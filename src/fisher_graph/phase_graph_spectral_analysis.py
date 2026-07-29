"""Phase-aware mode coupling and graph Fourier diagnostics.

The existing modal spectral mapper deliberately combines magnitude, phase,
and origin coherence into one real-valued similarity.  That representation is
useful for conservative clustering, but an exactly sign-reversed response is
not represented as an explicit negative edge.  This module keeps the complex
spectral inner product intact and builds two complementary mode graphs:

* a complex connection graph whose edge angle retains relative phase; and
* a signed real graph whose negative edges retain opposition.

Both graphs use normalized Laplacians.  Their eigenvectors are graph Fourier
bases over source modes, not temporal Fourier bases.  The diagnostics are
descriptive and source-artifact-bound; they do not identify semantics,
authorize compression, or establish model fidelity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .modal_spectral_mapping import ModalSpectralResponse


__all__ = [
    "DEFAULT_EIGENVALUE_BLOCK_TOLERANCE",
    "DEFAULT_RELATIVE_SUPPORT_FLOOR",
    "PhaseGraphSpectralAnalysis",
    "analyze_phase_graph_spectral_response",
]


_ARTIFACT_KIND = "fisher_graph.phase_graph_spectral_analysis"
_FORMAT_VERSION = 1
_HASH_DOMAIN = b"fisher-graph:phase-graph-spectral-analysis:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_RELATIVE_SUPPORT_FLOOR = math.sqrt(torch.finfo(torch.float64).eps)
DEFAULT_EIGENVALUE_BLOCK_TOLERANCE = 1e-10
_FLOAT_TENSOR_FIELDS = (
    "source_response_norms",
    "legacy_similarity",
    "directional_quadrature",
    "complex_coherency_real",
    "complex_coherency_imag",
    "coherence_magnitude",
    "phase_blind_magnitude_similarity",
    "phase_angle",
    "connection_adjacency_real",
    "connection_adjacency_imag",
    "signed_adjacency",
    "connection_laplacian_real",
    "connection_laplacian_imag",
    "signed_laplacian",
    "magnitude_laplacian",
    "connection_eigenvalues",
    "connection_eigenvectors_real",
    "connection_eigenvectors_imag",
    "signed_eigenvalues",
    "signed_eigenvectors",
    "magnitude_eigenvalues",
    "magnitude_eigenvectors",
    "connection_graph_fourier_energy",
    "signed_graph_fourier_energy",
    "magnitude_graph_fourier_energy",
)
_BOOL_TENSOR_FIELDS = ("selected_edge_mask",)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        _HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _float_tensor(
    value: object,
    *,
    label: str,
    dimensions: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != dimensions or any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{label} has an invalid shape")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must contain finite floating values")
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


def _bool_tensor(
    value: object,
    *,
    label: str,
    dimensions: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != dimensions or any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{label} has an invalid shape")
    if value.dtype != torch.bool:
        raise TypeError(f"{label} must use torch.bool")
    return value.detach().to(device="cpu").contiguous().clone()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _same_eigenvalue(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(
        1.0,
        abs(left),
        abs(right),
    )


def _eigenvalue_blocks(
    eigenvalues: Tensor,
    *,
    tolerance: float,
) -> tuple[tuple[int, int], ...]:
    count = int(eigenvalues.numel())
    if count == 0:
        return ()
    blocks: list[tuple[int, int]] = []
    start = 0
    for index in range(1, count):
        if not _same_eigenvalue(
            float(eigenvalues[index - 1]),
            float(eigenvalues[index]),
            tolerance=tolerance,
        ):
            blocks.append((start, index))
            start = index
    blocks.append((start, count))
    return tuple(blocks)


def _rank_at_energy(
    energy: Tensor,
    eigenvalues: Tensor,
    fraction: float,
    *,
    tolerance: float,
) -> int:
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps:
        return 0
    cumulative = 0.0
    for start, end in _eigenvalue_blocks(
        eigenvalues,
        tolerance=tolerance,
    ):
        cumulative += float(energy[start:end].sum()) / total
        if cumulative >= fraction:
            return end
    return int(energy.numel())


def _block_complete_prefix_energy(
    energy: Tensor,
    eigenvalues: Tensor,
    count: int,
    *,
    tolerance: float,
) -> float:
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps or count <= 0:
        return 0.0
    limit = min(count, int(energy.numel()))
    if limit < int(energy.numel()):
        while limit < int(energy.numel()) and _same_eigenvalue(
            float(eigenvalues[limit - 1]),
            float(eigenvalues[limit]),
            tolerance=tolerance,
        ):
            limit += 1
    return float(energy[:limit].sum()) / total


def _normalized_graph_laplacian(adjacency: Tensor) -> Tensor:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("graph adjacency must be square")
    degree = adjacency.abs().sum(dim=1).real
    active = degree > torch.finfo(torch.float64).eps
    inverse = torch.zeros_like(degree)
    inverse[active] = degree[active].rsqrt()
    normalized = inverse[:, None] * adjacency * inverse[None, :]
    diagonal = torch.diag(active.to(dtype=adjacency.real.dtype))
    if adjacency.is_complex():
        diagonal = diagonal.to(dtype=adjacency.dtype)
    laplacian = diagonal - normalized
    return (laplacian + laplacian.mH) / 2


def _deterministic_edge_mask(
    magnitude: Tensor,
    *,
    neighbor_count: int,
    minimum_coherence: float,
    active: Tensor,
) -> Tensor:
    mode_count = int(magnitude.shape[0])
    mask = torch.zeros((mode_count, mode_count), dtype=torch.bool)
    keep_count = min(neighbor_count, max(0, mode_count - 1))
    for source in range(mode_count):
        if not bool(active[source]) or keep_count == 0:
            continue
        ordered = sorted(
            (
                (-float(magnitude[source, target]), target)
                for target in range(mode_count)
                if target != source
                and bool(active[target])
                and float(magnitude[source, target]) >= minimum_coherence
            ),
        )
        for _, target in ordered[:keep_count]:
            mask[source, target] = True
    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask


def _graph_fourier_metrics(
    *,
    laplacian: Tensor,
    eigenvectors: Tensor,
    signal: Tensor,
) -> tuple[Tensor, float]:
    signal_energy = float(signal.abs().square().sum())
    if signal_energy <= torch.finfo(torch.float64).eps:
        return (
            torch.zeros(
                eigenvectors.shape[1],
                dtype=torch.float64,
            ),
            0.0,
        )
    coefficients = eigenvectors.mH @ signal
    energy = coefficients.abs().square().sum(dim=1).real
    energy /= energy.sum()
    smoothness = float(
        torch.real(
            torch.sum(signal.conj() * (laplacian @ signal))
        )
        / signal_energy
    )
    return energy.to(dtype=torch.float64), smoothness


def _pair_rows(
    *,
    coherency: Tensor,
    legacy: Tensor,
    directional_quadrature: Tensor,
    source_modes: tuple[int, ...],
    count: int,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    rows: list[dict[str, object]] = []
    for left in range(len(source_modes)):
        for right in range(left + 1, len(source_modes)):
            value = complex(coherency[left, right].item())
            legacy_value = float(legacy[left, right])
            rows.append(
                {
                    "left_mode": source_modes[left],
                    "right_mode": source_modes[right],
                    "coherence_magnitude": abs(value),
                    "phase_alignment": value.real,
                    "phase_quadrature": value.imag,
                    "directional_quadrature": float(
                        directional_quadrature[left, right]
                    ),
                    "phase_angle_radians": math.atan2(
                        value.imag,
                        value.real,
                    ),
                    "legacy_similarity": legacy_value,
                    "legacy_minus_phase_alignment": (
                        legacy_value - value.real
                    ),
                }
            )

    def stable(
        key,
        *,
        reverse: bool,
    ) -> tuple[dict[str, object], ...]:
        ordered = sorted(
            rows,
            key=lambda row: (
                key(row),
                -int(row["left_mode"]),
                -int(row["right_mode"]),
            ),
            reverse=reverse,
        )
        return tuple(ordered[: min(count, len(ordered))])

    aligned = tuple(
        row
        for row in stable(
            lambda row: float(row["phase_alignment"]),
            reverse=True,
        )
        if float(row["phase_alignment"]) > 0.0
    )
    opposed = tuple(
        row
        for row in stable(
            lambda row: float(row["phase_alignment"]),
            reverse=False,
        )
        if float(row["phase_alignment"]) < 0.0
    )
    quadrature = stable(
        lambda row: abs(float(row["phase_quadrature"])),
        reverse=True,
    )
    disagreement = stable(
        lambda row: abs(float(row["legacy_minus_phase_alignment"])),
        reverse=True,
    )
    return aligned, opposed, quadrature, disagreement


@dataclass(frozen=True, slots=True)
class PhaseGraphSpectralAnalysis:
    """Strict phase-aware graph Fourier analysis of one spectral response."""

    response_artifact_sha256: str
    response_label: str
    response_kind: str
    temporal_fft_length: int
    temporal_frequency_count: int
    rfft_energy_weighted: bool
    source_mode_indices: tuple[int, ...]
    neighbor_count: int
    minimum_coherence: float
    top_pair_count: int
    relative_support_floor: float
    eigenvalue_block_tolerance: float
    source_response_norms: Tensor
    legacy_similarity: Tensor
    directional_quadrature: Tensor
    complex_coherency_real: Tensor
    complex_coherency_imag: Tensor
    coherence_magnitude: Tensor
    phase_blind_magnitude_similarity: Tensor
    phase_angle: Tensor
    selected_edge_mask: Tensor
    connection_adjacency_real: Tensor
    connection_adjacency_imag: Tensor
    signed_adjacency: Tensor
    connection_laplacian_real: Tensor
    connection_laplacian_imag: Tensor
    signed_laplacian: Tensor
    magnitude_laplacian: Tensor
    connection_eigenvalues: Tensor
    connection_eigenvectors_real: Tensor
    connection_eigenvectors_imag: Tensor
    signed_eigenvalues: Tensor
    signed_eigenvectors: Tensor
    magnitude_eigenvalues: Tensor
    magnitude_eigenvectors: Tensor
    connection_graph_fourier_energy: Tensor
    signed_graph_fourier_energy: Tensor
    magnitude_graph_fourier_energy: Tensor
    connection_rank_90: int
    connection_rank_95: int
    connection_rank_99: int
    signed_rank_90: int
    signed_rank_95: int
    signed_rank_99: int
    magnitude_rank_90: int
    magnitude_rank_95: int
    magnitude_rank_99: int
    connection_low8_energy_fraction: float
    connection_low16_energy_fraction: float
    signed_low8_energy_fraction: float
    signed_low16_energy_fraction: float
    magnitude_low8_energy_fraction: float
    magnitude_low16_energy_fraction: float
    connection_smoothness: float
    signed_smoothness: float
    magnitude_smoothness: float
    active_mode_count: int
    minimum_active_response_norm: float
    maximum_active_response_norm: float
    active_response_norm_dynamic_range: float
    selected_edge_count: int
    opposed_selected_edge_count: int
    maximum_legacy_phase_gap: float
    mean_absolute_legacy_phase_gap: float
    strongest_aligned_pairs: tuple[dict[str, object], ...]
    strongest_opposed_pairs: tuple[dict[str, object], ...]
    strongest_quadrature_pairs: tuple[dict[str, object], ...]
    largest_legacy_phase_disagreements: tuple[dict[str, object], ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(
            self.response_artifact_sha256,
            label="response artifact",
        )
        if not isinstance(self.response_label, str) or not self.response_label:
            raise ValueError("response label must be nonempty")
        if not isinstance(self.response_kind, str) or not self.response_kind:
            raise ValueError("response kind must be nonempty")
        _positive_int(self.temporal_fft_length, label="temporal FFT length")
        _positive_int(
            self.temporal_frequency_count,
            label="temporal frequency count",
        )
        if (
            self.temporal_frequency_count
            != self.temporal_fft_length // 2 + 1
            or self.rfft_energy_weighted is not True
        ):
            raise ValueError("temporal FFT geometry differs")
        if (
            type(self.source_mode_indices) is not tuple
            or not self.source_mode_indices
            or any(type(value) is not int or value < 0 for value in self.source_mode_indices)
            or len(set(self.source_mode_indices)) != len(self.source_mode_indices)
        ):
            raise ValueError("source mode indices are invalid")
        _positive_int(self.neighbor_count, label="neighbor count")
        _finite_float(
            self.minimum_coherence,
            label="minimum coherence",
            minimum=0.0,
            maximum=1.0,
        )
        _positive_int(self.top_pair_count, label="top pair count")
        mode_count = len(self.source_mode_indices)

        _finite_float(
            self.relative_support_floor,
            label="relative support floor",
            minimum=0.0,
            maximum=1.0,
        )
        if self.relative_support_floor != DEFAULT_RELATIVE_SUPPORT_FLOOR:
            raise ValueError("relative support floor differs")
        _finite_float(
            self.eigenvalue_block_tolerance,
            label="eigenvalue block tolerance",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            self.eigenvalue_block_tolerance
            != DEFAULT_EIGENVALUE_BLOCK_TOLERANCE
        ):
            raise ValueError("eigenvalue block tolerance differs")
        for field in _FLOAT_TENSOR_FIELDS:
            dimensions = 1 if field.endswith(
                ("eigenvalues", "fourier_energy", "response_norms")
            ) else 2
            value = _float_tensor(
                getattr(self, field),
                label=field,
                dimensions=dimensions,
            )
            expected = (mode_count,) if dimensions == 1 else (
                mode_count,
                mode_count,
            )
            if tuple(value.shape) != expected:
                raise ValueError(f"{field} shape is invalid")
            object.__setattr__(self, field, value)
        for field in _BOOL_TENSOR_FIELDS:
            value = _bool_tensor(
                getattr(self, field),
                label=field,
                dimensions=2,
            )
            if tuple(value.shape) != (mode_count, mode_count):
                raise ValueError(f"{field} shape is invalid")
            object.__setattr__(self, field, value)

        coherency = torch.complex(
            self.complex_coherency_real,
            self.complex_coherency_imag,
        )
        directional_connection = torch.complex(
            self.legacy_similarity,
            self.directional_quadrature,
        )
        connection_adjacency = torch.complex(
            self.connection_adjacency_real,
            self.connection_adjacency_imag,
        )
        connection_laplacian = torch.complex(
            self.connection_laplacian_real,
            self.connection_laplacian_imag,
        )
        connection_eigenvectors = torch.complex(
            self.connection_eigenvectors_real,
            self.connection_eigenvectors_imag,
        )
        tolerance = 1e-9
        if (
            not torch.allclose(coherency, coherency.mH, atol=tolerance, rtol=0)
            or not torch.allclose(
                directional_connection,
                directional_connection.mH,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                connection_adjacency,
                connection_adjacency.mH,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                connection_laplacian,
                connection_laplacian.mH,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.signed_laplacian,
                self.signed_laplacian.T,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.magnitude_laplacian,
                self.magnitude_laplacian.T,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.phase_blind_magnitude_similarity,
                self.phase_blind_magnitude_similarity.T,
                atol=tolerance,
                rtol=0,
            )
            or not torch.equal(
                self.selected_edge_mask,
                self.selected_edge_mask.T,
            )
            or not torch.allclose(
                self.directional_quadrature,
                -self.directional_quadrature.T,
                atol=tolerance,
                rtol=0,
            )
        ):
            raise ValueError("phase graph matrices violate symmetry")
        identity = torch.eye(mode_count, dtype=torch.complex128)
        if not torch.allclose(
            connection_eigenvectors.mH @ connection_eigenvectors,
            identity,
            atol=2e-8,
            rtol=2e-8,
        ):
            raise ValueError("connection eigenvectors are not orthonormal")
        if not torch.allclose(
            self.signed_eigenvectors.T @ self.signed_eigenvectors,
            torch.eye(mode_count, dtype=torch.float64),
            atol=2e-8,
            rtol=2e-8,
        ):
            raise ValueError("signed eigenvectors are not orthonormal")
        if not torch.allclose(
            self.magnitude_eigenvectors.T @ self.magnitude_eigenvectors,
            torch.eye(mode_count, dtype=torch.float64),
            atol=2e-8,
            rtol=2e-8,
        ):
            raise ValueError("magnitude eigenvectors are not orthonormal")
        if (
            float(self.connection_eigenvalues.min()) < -1e-8
            or float(self.signed_eigenvalues.min()) < -1e-8
            or float(self.magnitude_eigenvalues.min()) < -1e-8
            or float(self.connection_eigenvalues.max()) > 2.0 + 1e-8
            or float(self.signed_eigenvalues.max()) > 2.0 + 1e-8
            or float(self.magnitude_eigenvalues.max()) > 2.0 + 1e-8
            or bool((self.coherence_magnitude < -tolerance).any())
            or bool((self.coherence_magnitude > 1.0 + tolerance).any())
            or bool(
                (self.phase_blind_magnitude_similarity < -tolerance).any()
            )
            or bool(
                (self.phase_blind_magnitude_similarity > 1.0 + tolerance).any()
            )
            or bool((self.source_response_norms < -tolerance).any())
        ):
            raise ValueError("phase graph spectrum is outside its domain")
        reconstructed_connection = (
            connection_eigenvectors
            @ torch.diag(self.connection_eigenvalues).to(torch.complex128)
            @ connection_eigenvectors.mH
        )
        reconstructed_signed = (
            self.signed_eigenvectors
            @ torch.diag(self.signed_eigenvalues)
            @ self.signed_eigenvectors.T
        )
        reconstructed_magnitude = (
            self.magnitude_eigenvectors
            @ torch.diag(self.magnitude_eigenvalues)
            @ self.magnitude_eigenvectors.T
        )
        if (
            not torch.allclose(
                reconstructed_connection,
                connection_laplacian,
                atol=2e-8,
                rtol=2e-8,
            )
            or not torch.allclose(
                reconstructed_signed,
                self.signed_laplacian,
                atol=2e-8,
                rtol=2e-8,
            )
            or not torch.allclose(
                reconstructed_magnitude,
                self.magnitude_laplacian,
                atol=2e-8,
                rtol=2e-8,
            )
        ):
            raise ValueError("phase graph eigendecomposition is invalid")
        for field in (
            "connection_graph_fourier_energy",
            "signed_graph_fourier_energy",
            "magnitude_graph_fourier_energy",
        ):
            energy = getattr(self, field)
            total = float(energy.sum())
            if bool((energy < -tolerance).any()) or (
                total > tolerance and not math.isclose(
                    total,
                    1.0,
                    abs_tol=1e-9,
                    rel_tol=1e-9,
                )
            ):
                raise ValueError(f"{field} is not normalized")
        for field in (
            "connection_rank_90",
            "connection_rank_95",
            "connection_rank_99",
            "signed_rank_90",
            "signed_rank_95",
            "signed_rank_99",
            "magnitude_rank_90",
            "magnitude_rank_95",
            "magnitude_rank_99",
            "active_mode_count",
        ):
            value = _nonnegative_int(getattr(self, field), label=field)
            if value > mode_count:
                raise ValueError(f"{field} exceeds mode count")
        maximum_edges = mode_count * (mode_count - 1) // 2
        for field in (
            "selected_edge_count",
            "opposed_selected_edge_count",
        ):
            value = _nonnegative_int(getattr(self, field), label=field)
            if value > maximum_edges:
                raise ValueError(f"{field} exceeds graph size")
        if self.opposed_selected_edge_count > self.selected_edge_count:
            raise ValueError("opposed edge count exceeds selected edge count")
        maximum_norm = float(self.source_response_norms.max())
        support_threshold = max(
            torch.finfo(torch.float64).eps,
            maximum_norm * self.relative_support_floor,
        )
        expected_active = self.source_response_norms > support_threshold
        active_norms = self.source_response_norms[expected_active]
        expected_minimum_norm = (
            float(active_norms.min()) if active_norms.numel() else 0.0
        )
        expected_maximum_norm = (
            float(active_norms.max()) if active_norms.numel() else 0.0
        )
        expected_dynamic_range = (
            expected_maximum_norm / expected_minimum_norm
            if expected_minimum_norm > 0.0
            else 0.0
        )
        if (
            self.active_mode_count != int(expected_active.sum())
            or self.minimum_active_response_norm != expected_minimum_norm
            or self.maximum_active_response_norm != expected_maximum_norm
            or self.active_response_norm_dynamic_range
            != expected_dynamic_range
            or not torch.allclose(
                coherency.diagonal().real,
                expected_active.to(torch.float64),
                atol=tolerance,
                rtol=0,
            )
        ):
            raise ValueError("response norm diagnostics differ")
        expected_connection_adjacency = coherency.clone()
        expected_connection_adjacency.fill_diagonal_(0.0)
        expected_signed_adjacency = coherency.real.clone()
        expected_signed_adjacency.fill_diagonal_(0.0)
        expected_magnitude_adjacency = (
            self.phase_blind_magnitude_similarity.clone()
        )
        expected_magnitude_adjacency.fill_diagonal_(0.0)
        expected_mask = _deterministic_edge_mask(
            self.coherence_magnitude,
            neighbor_count=self.neighbor_count,
            minimum_coherence=self.minimum_coherence,
            active=expected_active,
        )
        if (
            not torch.allclose(
                self.coherence_magnitude,
                coherency.abs(),
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.phase_angle,
                torch.angle(coherency),
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                connection_adjacency,
                expected_connection_adjacency,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.signed_adjacency,
                expected_signed_adjacency,
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                connection_laplacian,
                _normalized_graph_laplacian(
                    expected_connection_adjacency
                ),
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.signed_laplacian,
                _normalized_graph_laplacian(expected_signed_adjacency),
                atol=tolerance,
                rtol=0,
            )
            or not torch.allclose(
                self.magnitude_laplacian,
                _normalized_graph_laplacian(
                    expected_magnitude_adjacency
                ),
                atol=tolerance,
                rtol=0,
            )
            or not torch.equal(self.selected_edge_mask, expected_mask)
        ):
            raise ValueError("phase graph construction differs")
        upper_mask = torch.triu(self.selected_edge_mask, diagonal=1)
        if (
            self.selected_edge_count != int(upper_mask.sum())
            or self.opposed_selected_edge_count
            != int((upper_mask & (coherency.real < 0.0)).sum())
        ):
            raise ValueError("selected edge counts differ")
        for prefix in ("connection", "signed", "magnitude"):
            ranks = tuple(
                getattr(self, f"{prefix}_rank_{threshold}")
                for threshold in (90, 95, 99)
            )
            if ranks != tuple(sorted(ranks)):
                raise ValueError(f"{prefix} graph Fourier ranks are not monotone")
            energy = getattr(self, f"{prefix}_graph_fourier_energy")
            eigenvalues = getattr(self, f"{prefix}_eigenvalues")
            expected_ranks = tuple(
                _rank_at_energy(
                    energy,
                    eigenvalues,
                    fraction,
                    tolerance=self.eigenvalue_block_tolerance,
                )
                for fraction in (0.90, 0.95, 0.99)
            )
            expected_prefixes = tuple(
                _block_complete_prefix_energy(
                    energy,
                    eigenvalues,
                    count,
                    tolerance=self.eigenvalue_block_tolerance,
                )
                for count in (8, 16)
            )
            actual_prefixes = (
                getattr(self, f"{prefix}_low8_energy_fraction"),
                getattr(self, f"{prefix}_low16_energy_fraction"),
            )
            if ranks != expected_ranks or actual_prefixes != expected_prefixes:
                raise ValueError(
                    f"{prefix} graph Fourier summaries differ"
                )
        for field in (
            "connection_low8_energy_fraction",
            "connection_low16_energy_fraction",
            "signed_low8_energy_fraction",
            "signed_low16_energy_fraction",
            "magnitude_low8_energy_fraction",
            "magnitude_low16_energy_fraction",
        ):
            _finite_float(
                getattr(self, field),
                label=field,
                minimum=0.0,
                maximum=1.0,
            )
        for field in (
            "minimum_active_response_norm",
            "maximum_active_response_norm",
            "active_response_norm_dynamic_range",
            "connection_smoothness",
            "signed_smoothness",
            "magnitude_smoothness",
            "maximum_legacy_phase_gap",
            "mean_absolute_legacy_phase_gap",
        ):
            _finite_float(getattr(self, field), label=field, minimum=0.0)
        pair_fields = (
            "strongest_aligned_pairs",
            "strongest_opposed_pairs",
            "strongest_quadrature_pairs",
            "largest_legacy_phase_disagreements",
        )
        for field in pair_fields:
            rows = getattr(self, field)
            if type(rows) is not tuple or len(rows) > self.top_pair_count:
                raise ValueError(f"{field} is invalid")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError(f"{field} rows must be mappings")
                for key, value in row.items():
                    if not isinstance(key, str) or isinstance(value, Tensor):
                        raise TypeError(f"{field} contains an unsafe row")

        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="phase graph artifact",
            ) != computed:
                raise ValueError("phase graph artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _scalar_state(self) -> dict[str, object]:
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "response_artifact_sha256": self.response_artifact_sha256,
            "response_label": self.response_label,
            "response_kind": self.response_kind,
            "temporal_fft_length": self.temporal_fft_length,
            "temporal_frequency_count": self.temporal_frequency_count,
            "rfft_energy_weighted": self.rfft_energy_weighted,
            "source_mode_indices": self.source_mode_indices,
            "neighbor_count": self.neighbor_count,
            "minimum_coherence": self.minimum_coherence,
            "top_pair_count": self.top_pair_count,
            "relative_support_floor": self.relative_support_floor,
            "eigenvalue_block_tolerance": (
                self.eigenvalue_block_tolerance
            ),
            "connection_rank_90": self.connection_rank_90,
            "connection_rank_95": self.connection_rank_95,
            "connection_rank_99": self.connection_rank_99,
            "signed_rank_90": self.signed_rank_90,
            "signed_rank_95": self.signed_rank_95,
            "signed_rank_99": self.signed_rank_99,
            "magnitude_rank_90": self.magnitude_rank_90,
            "magnitude_rank_95": self.magnitude_rank_95,
            "magnitude_rank_99": self.magnitude_rank_99,
            "connection_low8_energy_fraction": (
                self.connection_low8_energy_fraction
            ),
            "connection_low16_energy_fraction": (
                self.connection_low16_energy_fraction
            ),
            "signed_low8_energy_fraction": self.signed_low8_energy_fraction,
            "signed_low16_energy_fraction": (
                self.signed_low16_energy_fraction
            ),
            "magnitude_low8_energy_fraction": (
                self.magnitude_low8_energy_fraction
            ),
            "magnitude_low16_energy_fraction": (
                self.magnitude_low16_energy_fraction
            ),
            "connection_smoothness": self.connection_smoothness,
            "signed_smoothness": self.signed_smoothness,
            "magnitude_smoothness": self.magnitude_smoothness,
            "active_mode_count": self.active_mode_count,
            "minimum_active_response_norm": (
                self.minimum_active_response_norm
            ),
            "maximum_active_response_norm": (
                self.maximum_active_response_norm
            ),
            "active_response_norm_dynamic_range": (
                self.active_response_norm_dynamic_range
            ),
            "selected_edge_count": self.selected_edge_count,
            "opposed_selected_edge_count": self.opposed_selected_edge_count,
            "maximum_legacy_phase_gap": self.maximum_legacy_phase_gap,
            "mean_absolute_legacy_phase_gap": (
                self.mean_absolute_legacy_phase_gap
            ),
            "strongest_aligned_pairs": self.strongest_aligned_pairs,
            "strongest_opposed_pairs": self.strongest_opposed_pairs,
            "strongest_quadrature_pairs": self.strongest_quadrature_pairs,
            "largest_legacy_phase_disagreements": (
                self.largest_legacy_phase_disagreements
            ),
            "phase_semantics": (
                "legacy_similarity_retains_existing_magnitude_cosine_phase_"
                "score;directional_quadrature_adds_coherence_weighted_sine_"
                "of_relative_phase;raw_complex_cosine_is_an_opposition_"
                "diagnostic"
            ),
            "connection_laplacian_semantics": (
                "normalized_dense_complex_connection_laplacian_of_raw_"
                "spectral_coherency_degree_uses_absolute_edge_weight"
            ),
            "signed_laplacian_semantics": (
                "normalized_dense_signed_real_part_of_raw_spectral_"
                "coherency_laplacian_degree_uses_absolute_edge_weight"
            ),
            "magnitude_laplacian_semantics": (
                "normalized_dense_nonnegative_cosine_similarity_of_per_bin_"
                "spectral_magnitudes_is_the_phase_blind_control"
            ),
            "graph_fourier_semantics": (
                "dense_source_mode_graph_eigenvectors_project_mean_complex_"
                "temporal_spectra;selected_edge_mask_is_visualization_only"
            ),
            "response_gain_semantics": (
                "supported_response_rows_are_l2_normalized_so_graph_edges_"
                "encode_shape_not_gain;raw_norm_diagnostics_are_retained"
            ),
            "pooled_edge_semantics": (
                "origin_temporal_frequency_and_target_mode_are_globally_"
                "flattened_after_one_sided_rfft_energy_weighting;real_"
                "alignment_is_parseval_consistent_but_pooled_complex_phase_"
                "remains_fft_length_bound_and_is_not_a_frequency_resolved_"
                "delay_or_directed_transport_edge"
            ),
            "rank_semantics": (
                "rank_90_95_99_are_low_to_high_laplacian_eigenvalue_block_"
                "complete_prefix_indices_not_matrix_rank_effective_rank_or_"
                "expert_count;low8_low16_include_the_full_boundary_"
                "eigenspace"
            ),
            "energy_tensor_semantics": (
                "per_eigenvector_energy_is_basis_dependent_inside_repeated_"
                "eigenspaces;reported_prefix_scalars_complete_tied_blocks"
            ),
            "causal_identification_claim": False,
            "coupling_strength_claim": False,
            "directed_transfer_graph_claim": False,
            "frequency_resolved_phase_claim": False,
            "semantic_cluster_claim": False,
            "compression_claim": False,
            "speed_claim": False,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            {
                **self._scalar_state(),
                "tensor_sha256s": {
                    field: _tensor_sha256(getattr(self, field))
                    for field in (*_FLOAT_TENSOR_FIELDS, *_BOOL_TENSOR_FIELDS)
                },
            }
        )

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("phase graph artifact integrity check failed")

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._scalar_state(),
            **{
                field: getattr(self, field).clone()
                for field in (*_FLOAT_TENSOR_FIELDS, *_BOOL_TENSOR_FIELDS)
            },
            "artifact_sha256": self.artifact_sha256,
        }

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._scalar_state(),
            "tensor_fields": {
                field: {
                    "shape": tuple(int(size) for size in getattr(self, field).shape),
                    "dtype": str(getattr(self, field).dtype),
                    "sha256": _tensor_sha256(getattr(self, field)),
                }
                for field in (*_FLOAT_TENSOR_FIELDS, *_BOOL_TENSOR_FIELDS)
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "PhaseGraphSpectralAnalysis":
        if not isinstance(raw, Mapping) or any(
            not isinstance(key, str) for key in raw
        ):
            raise TypeError("phase graph state must be a string-keyed mapping")
        exemplar_keys = {
            *cls.__dataclass_fields__,
            "artifact_kind",
            "format_version",
            "phase_semantics",
            "connection_laplacian_semantics",
            "signed_laplacian_semantics",
            "magnitude_laplacian_semantics",
            "graph_fourier_semantics",
            "response_gain_semantics",
            "pooled_edge_semantics",
            "rank_semantics",
            "energy_tensor_semantics",
            "causal_identification_claim",
            "coupling_strength_claim",
            "directed_transfer_graph_claim",
            "frequency_resolved_phase_claim",
            "semantic_cluster_claim",
            "compression_claim",
            "speed_claim",
        }
        if set(raw) != exemplar_keys:
            raise ValueError("phase graph state fields differ")
        if (
            raw["artifact_kind"] != _ARTIFACT_KIND
            or raw["format_version"] != _FORMAT_VERSION
            or raw["causal_identification_claim"] is not False
            or raw["coupling_strength_claim"] is not False
            or raw["directed_transfer_graph_claim"] is not False
            or raw["frequency_resolved_phase_claim"] is not False
            or raw["semantic_cluster_claim"] is not False
            or raw["compression_claim"] is not False
            or raw["speed_claim"] is not False
        ):
            raise ValueError("phase graph state provenance is invalid")
        _require_sha256(
            raw["artifact_sha256"],
            label="phase graph artifact",
        )
        kwargs = {
            field: raw[field]
            for field in cls.__dataclass_fields__
        }
        kwargs["source_mode_indices"] = tuple(raw["source_mode_indices"])
        for field in (
            "strongest_aligned_pairs",
            "strongest_opposed_pairs",
            "strongest_quadrature_pairs",
            "largest_legacy_phase_disagreements",
        ):
            kwargs[field] = tuple(dict(row) for row in raw[field])
        result = cls(**kwargs)
        canonical = result._scalar_state()
        for field in (
            "phase_semantics",
            "connection_laplacian_semantics",
            "signed_laplacian_semantics",
            "magnitude_laplacian_semantics",
            "graph_fourier_semantics",
            "response_gain_semantics",
            "pooled_edge_semantics",
            "rank_semantics",
            "energy_tensor_semantics",
        ):
            if raw[field] != canonical[field]:
                raise ValueError("phase graph semantics differ")
        return result


def analyze_phase_graph_spectral_response(
    response: ModalSpectralResponse,
    *,
    neighbor_count: int = 6,
    minimum_coherence: float = 0.10,
    top_pair_count: int = 8,
) -> PhaseGraphSpectralAnalysis:
    """Build phase-aware signed/connection graphs for one frozen response."""

    if not isinstance(response, ModalSpectralResponse):
        raise TypeError("response must be a ModalSpectralResponse")
    response.validate_integrity()
    neighbor_count = _positive_int(neighbor_count, label="neighbor count")
    minimum_coherence = _finite_float(
        minimum_coherence,
        label="minimum coherence",
        minimum=0.0,
        maximum=1.0,
    )
    top_pair_count = _positive_int(top_pair_count, label="top pair count")

    spectrum = torch.complex(
        response.spectral_fingerprint_real,
        response.spectral_fingerprint_imag,
    )
    frequency_count = response.fft_length // 2 + 1
    parseval_weights = torch.ones(
        frequency_count,
        dtype=torch.float64,
    )
    if response.fft_length % 2 == 0:
        parseval_weights[1:-1] = 2.0
    else:
        parseval_weights[1:] = 2.0
    parseval_scale = parseval_weights.sqrt()
    weighted_spectrum = spectrum * parseval_scale[None, None, :, None]
    flattened = weighted_spectrum.reshape(spectrum.shape[0], -1)
    norms = torch.linalg.vector_norm(flattened, dim=1)
    maximum_norm = float(norms.max())
    support_threshold = max(
        torch.finfo(torch.float64).eps,
        maximum_norm * DEFAULT_RELATIVE_SUPPORT_FLOOR,
    )
    active = norms > support_threshold
    active_norms = norms[active]
    minimum_active_norm = (
        float(active_norms.min()) if active_norms.numel() else 0.0
    )
    maximum_active_norm = (
        float(active_norms.max()) if active_norms.numel() else 0.0
    )
    active_norm_dynamic_range = (
        maximum_active_norm / minimum_active_norm
        if minimum_active_norm > 0.0
        else 0.0
    )
    normalized = torch.zeros_like(flattened)
    normalized[active] = flattened[active] / norms[active, None]
    coherency = normalized @ normalized.mH
    coherency = (coherency + coherency.mH) / 2
    diagonal = torch.arange(coherency.shape[0])
    coherency[diagonal[active], diagonal[active]] = 1.0 + 0.0j
    coherency[diagonal[~active], diagonal[~active]] = 0.0 + 0.0j
    magnitude = coherency.abs().clamp(0.0, 1.0)
    phase_angle = torch.angle(coherency)
    phase_blind_features = flattened.abs()
    phase_blind_norms = torch.linalg.vector_norm(
        phase_blind_features,
        dim=1,
    )
    phase_blind_normalized = torch.zeros_like(phase_blind_features)
    phase_blind_active = active
    phase_blind_normalized[phase_blind_active] = (
        phase_blind_features[phase_blind_active]
        / phase_blind_norms[phase_blind_active, None]
    )
    phase_blind_similarity = (
        phase_blind_normalized @ phase_blind_normalized.T
    ).clamp(0.0, 1.0)
    phase_blind_similarity[
        diagonal[phase_blind_active],
        diagonal[phase_blind_active],
    ] = 1.0
    phase_blind_similarity[
        diagonal[~phase_blind_active],
        diagonal[~phase_blind_active],
    ] = 0.0

    spectral_magnitude = (
        response.magnitude * parseval_scale[None, :, None]
    ).reshape(spectrum.shape[0], -1)
    spectral_magnitude_norm = torch.linalg.vector_norm(
        spectral_magnitude,
        dim=1,
    )
    normalized_magnitude = torch.zeros_like(spectral_magnitude)
    magnitude_active = (
        spectral_magnitude_norm > torch.finfo(torch.float64).eps
    )
    normalized_magnitude[magnitude_active] = (
        spectral_magnitude[magnitude_active]
        / spectral_magnitude_norm[magnitude_active, None]
    )
    coherence = response.coherence_like.reshape(spectrum.shape[0], -1)
    spectral_phase = response.phase.reshape(spectrum.shape[0], -1)
    phase_cos = normalized_magnitude * coherence * torch.cos(spectral_phase)
    phase_sin = normalized_magnitude * coherence * torch.sin(spectral_phase)
    signature_norm = torch.sqrt(
        normalized_magnitude.square().sum(dim=1)
        + phase_cos.square().sum(dim=1)
        + phase_sin.square().sum(dim=1)
    )
    quadrature_numerator = (
        phase_sin @ phase_cos.T - phase_cos @ phase_sin.T
    )
    signature_denominator = signature_norm[:, None] * signature_norm[None, :]
    directional_quadrature = torch.where(
        signature_denominator > torch.finfo(torch.float64).eps,
        quadrature_numerator / signature_denominator,
        torch.zeros_like(quadrature_numerator),
    )
    directional_quadrature = (
        directional_quadrature - directional_quadrature.T
    ) / 2
    directional_quadrature.fill_diagonal_(0.0)
    legacy = response.pairwise_spectral_similarity
    edge_mask = _deterministic_edge_mask(
        magnitude,
        neighbor_count=neighbor_count,
        minimum_coherence=minimum_coherence,
        active=active,
    )
    # The graph Fourier basis is built from the full authenticated graph.
    # Top-k selection is retained only as a deterministic readable view and
    # therefore cannot manufacture graph-frequency compaction.
    connection_adjacency = coherency.clone()
    connection_adjacency.fill_diagonal_(0.0)
    signed_adjacency = coherency.real.clone().contiguous()
    signed_adjacency.fill_diagonal_(0.0)
    connection_laplacian = _normalized_graph_laplacian(
        connection_adjacency
    )
    signed_laplacian = _normalized_graph_laplacian(signed_adjacency)
    magnitude_adjacency = phase_blind_similarity.clone()
    magnitude_adjacency.fill_diagonal_(0.0)
    magnitude_laplacian = _normalized_graph_laplacian(
        magnitude_adjacency
    )
    connection_values, connection_vectors = torch.linalg.eigh(
        connection_laplacian
    )
    signed_values, signed_vectors = torch.linalg.eigh(signed_laplacian)
    magnitude_values, magnitude_vectors = torch.linalg.eigh(
        magnitude_laplacian
    )
    if (
        float(connection_values.min()) < -1e-8
        or float(signed_values.min()) < -1e-8
        or float(magnitude_values.min()) < -1e-8
    ):
        raise RuntimeError("phase graph Laplacian is not positive semidefinite")
    connection_values = connection_values.clamp_min(0.0)
    signed_values = signed_values.clamp_min(0.0)
    magnitude_values = magnitude_values.clamp_min(0.0)

    mean_spectrum = torch.complex(
        response.mean_spectral_fingerprint_real,
        response.mean_spectral_fingerprint_imag,
    )
    mean_spectrum = (
        mean_spectrum * parseval_scale[None, :, None]
    ).reshape(spectrum.shape[0], -1)
    connection_energy, connection_smoothness = _graph_fourier_metrics(
        laplacian=connection_laplacian,
        eigenvectors=connection_vectors,
        signal=mean_spectrum,
    )
    signed_energy, signed_smoothness = _graph_fourier_metrics(
        laplacian=signed_laplacian.to(dtype=torch.complex128),
        eigenvectors=signed_vectors.to(dtype=torch.complex128),
        signal=mean_spectrum,
    )
    magnitude_energy, magnitude_smoothness = _graph_fourier_metrics(
        laplacian=magnitude_laplacian.to(dtype=torch.complex128),
        eigenvectors=magnitude_vectors.to(dtype=torch.complex128),
        signal=mean_spectrum,
    )
    gap = (legacy - coherency.real).abs()
    off_diagonal = ~torch.eye(gap.shape[0], dtype=torch.bool)
    gap_values = gap[off_diagonal]
    maximum_gap = float(gap_values.max()) if gap_values.numel() else 0.0
    mean_gap = float(gap_values.mean()) if gap_values.numel() else 0.0
    aligned, opposed, quadrature, disagreement = _pair_rows(
        coherency=coherency,
        legacy=legacy,
        directional_quadrature=directional_quadrature,
        source_modes=response.source_mode_indices,
        count=top_pair_count,
    )
    upper = torch.triu(edge_mask, diagonal=1)
    opposed_edges = upper & (coherency.real < 0.0)

    return PhaseGraphSpectralAnalysis(
        response_artifact_sha256=response.artifact_sha256,
        response_label=response.label,
        response_kind=response.response_kind,
        temporal_fft_length=response.fft_length,
        temporal_frequency_count=frequency_count,
        rfft_energy_weighted=True,
        source_mode_indices=response.source_mode_indices,
        neighbor_count=neighbor_count,
        minimum_coherence=minimum_coherence,
        top_pair_count=top_pair_count,
        relative_support_floor=DEFAULT_RELATIVE_SUPPORT_FLOOR,
        eigenvalue_block_tolerance=(
            DEFAULT_EIGENVALUE_BLOCK_TOLERANCE
        ),
        source_response_norms=norms,
        legacy_similarity=legacy,
        directional_quadrature=directional_quadrature,
        complex_coherency_real=coherency.real,
        complex_coherency_imag=coherency.imag,
        coherence_magnitude=magnitude,
        phase_blind_magnitude_similarity=phase_blind_similarity,
        phase_angle=phase_angle,
        selected_edge_mask=edge_mask,
        connection_adjacency_real=connection_adjacency.real,
        connection_adjacency_imag=connection_adjacency.imag,
        signed_adjacency=signed_adjacency,
        connection_laplacian_real=connection_laplacian.real,
        connection_laplacian_imag=connection_laplacian.imag,
        signed_laplacian=signed_laplacian,
        magnitude_laplacian=magnitude_laplacian,
        connection_eigenvalues=connection_values,
        connection_eigenvectors_real=connection_vectors.real,
        connection_eigenvectors_imag=connection_vectors.imag,
        signed_eigenvalues=signed_values,
        signed_eigenvectors=signed_vectors,
        magnitude_eigenvalues=magnitude_values,
        magnitude_eigenvectors=magnitude_vectors,
        connection_graph_fourier_energy=connection_energy,
        signed_graph_fourier_energy=signed_energy,
        magnitude_graph_fourier_energy=magnitude_energy,
        connection_rank_90=_rank_at_energy(
            connection_energy,
            connection_values,
            0.90,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        connection_rank_95=_rank_at_energy(
            connection_energy,
            connection_values,
            0.95,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        connection_rank_99=_rank_at_energy(
            connection_energy,
            connection_values,
            0.99,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        signed_rank_90=_rank_at_energy(
            signed_energy,
            signed_values,
            0.90,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        signed_rank_95=_rank_at_energy(
            signed_energy,
            signed_values,
            0.95,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        signed_rank_99=_rank_at_energy(
            signed_energy,
            signed_values,
            0.99,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        magnitude_rank_90=_rank_at_energy(
            magnitude_energy,
            magnitude_values,
            0.90,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        magnitude_rank_95=_rank_at_energy(
            magnitude_energy,
            magnitude_values,
            0.95,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        magnitude_rank_99=_rank_at_energy(
            magnitude_energy,
            magnitude_values,
            0.99,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        connection_low8_energy_fraction=_block_complete_prefix_energy(
            connection_energy,
            connection_values,
            8,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        connection_low16_energy_fraction=_block_complete_prefix_energy(
            connection_energy,
            connection_values,
            16,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        signed_low8_energy_fraction=_block_complete_prefix_energy(
            signed_energy,
            signed_values,
            8,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        signed_low16_energy_fraction=_block_complete_prefix_energy(
            signed_energy,
            signed_values,
            16,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        magnitude_low8_energy_fraction=_block_complete_prefix_energy(
            magnitude_energy,
            magnitude_values,
            8,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        magnitude_low16_energy_fraction=_block_complete_prefix_energy(
            magnitude_energy,
            magnitude_values,
            16,
            tolerance=DEFAULT_EIGENVALUE_BLOCK_TOLERANCE,
        ),
        connection_smoothness=max(0.0, connection_smoothness),
        signed_smoothness=max(0.0, signed_smoothness),
        magnitude_smoothness=max(0.0, magnitude_smoothness),
        active_mode_count=int(active.sum()),
        minimum_active_response_norm=minimum_active_norm,
        maximum_active_response_norm=maximum_active_norm,
        active_response_norm_dynamic_range=active_norm_dynamic_range,
        selected_edge_count=int(upper.sum()),
        opposed_selected_edge_count=int(opposed_edges.sum()),
        maximum_legacy_phase_gap=maximum_gap,
        mean_absolute_legacy_phase_gap=mean_gap,
        strongest_aligned_pairs=aligned,
        strongest_opposed_pairs=opposed,
        strongest_quadrature_pairs=quadrature,
        largest_legacy_phase_disagreements=disagreement,
    )
