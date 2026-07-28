"""Prompt-free causal-lag impulse and spectral fingerprints for modal maps.

This module probes a callable mapping from source modal sequences to target
modal sequences.  It records finite one-sided secants and optional symmetric
central secants at explicitly bound logical origins, aligns their outputs by
nonnegative logical lag, and computes zero-padded real FFT fingerprints.

The resulting spectra are descriptive, reference-dependent fingerprints.
They are not evidence of causal identification, shift invariance, or a
universal convolutional transfer function.  Precausal energy and agreement
across impulse origins are surfaced so those assumptions remain testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable

import torch
from torch import Tensor


__all__ = [
    "ModalSpectralMapping",
    "ModalSpectralResponse",
    "analyze_modal_spectral_mapping",
    "connected_components_from_spectral_similarity",
]


_RESPONSE_KIND = "fisher_graph.modal_spectral_response"
_MAPPING_KIND = "fisher_graph.modal_spectral_mapping"
_FORMAT_VERSION = 1
_RESPONSE_HASH_DOMAIN = b"fisher_graph.modal_spectral_response.v1\0"
_MAPPING_HASH_DOMAIN = b"fisher_graph.modal_spectral_mapping.v1\0"
_NO_CAUSALITY_CLAIM = (
    "origin_bound_descriptive_fingerprint_not_causal_identification"
)
_NO_SHIFT_INVARIANCE_CLAIM = (
    "multiple_origins_are_retained_and_shift_invariance_is_not_assumed"
)
_SPECTRAL_SEMANTICS = (
    "zero_padded_rfft_of_nonnegative_logical_lag_impulse_response"
)
_COHERENCE_SEMANTICS = (
    "squared_coherent_mean_over_mean_power_across_impulse_origins"
)
_FREQUENCY_BAND_SEMANTICS = (
    "dc=0;low=(0,1/6];mid=(1/6,1/3];high=(1/3,1/2]_cycles_per_token"
)
_RANK_SEMANTICS = (
    "smallest_singular_prefix_reaching_squared_singular_value_energy_fraction"
)
_CLUSTERING_SEMANTICS = (
    "deterministic_connected_components_of_descriptive_spectral_similarity"
)
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TENSOR_FIELDS = (
    "impulse_responses",
    "even_residual_impulse_responses",
    "spectral_fingerprint_real",
    "spectral_fingerprint_imag",
    "mean_spectral_fingerprint_real",
    "mean_spectral_fingerprint_imag",
    "magnitude",
    "phase",
    "coherence_like",
    "normalized_signatures",
    "pairwise_spectral_similarity",
    "origin_spectral_similarity",
    "lag_energy_fractions",
)


def _strict_keys(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    actual = set(state)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(actual)}"
        )


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    result = _require_nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite(
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


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Tensor) -> str:
    original_dtype = value.dtype
    if value.is_complex():
        tensor = value.detach().to(
            device="cpu",
            dtype=torch.complex128,
        ).contiguous()
    elif value.is_floating_point():
        tensor = value.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
    else:
        tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(original_dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, object], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    return digest.hexdigest()


def _as_cpu_float64(
    value: Tensor,
    *,
    label: str,
    dimensions: int,
    allow_empty: bool = False,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != dimensions:
        raise ValueError(f"{label} must be rank {dimensions}")
    if not value.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    if not allow_empty and any(int(width) <= 0 for width in value.shape):
        raise ValueError(f"{label} dimensions must be positive")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


def _positions_and_mask(
    logical_positions: Tensor,
    valid_mask: Tensor,
    *,
    sequence_length: int,
) -> tuple[Tensor, Tensor, tuple[int, ...], tuple[bool, ...]]:
    if not isinstance(logical_positions, Tensor):
        raise TypeError("logical_positions must be a Tensor")
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a Tensor")
    positions = logical_positions.detach()
    mask = valid_mask.detach()
    if positions.ndim == 2 and positions.shape[0] == 1:
        positions = positions[0]
    if mask.ndim == 2 and mask.shape[0] == 1:
        mask = mask[0]
    if positions.ndim != 1 or positions.shape[0] != sequence_length:
        raise ValueError("logical_positions must have shape [S] or [1, S]")
    if mask.ndim != 1 or mask.shape[0] != sequence_length:
        raise ValueError("valid_mask must have shape [S] or [1, S]")
    if positions.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("logical_positions must use an integer dtype")
    if mask.dtype != torch.bool:
        raise TypeError("valid_mask must use torch.bool")
    positions_cpu = positions.to(device="cpu", dtype=torch.int64).contiguous()
    mask_cpu = mask.to(device="cpu", dtype=torch.bool).contiguous()
    valid = positions_cpu[mask_cpu]
    if valid.numel() == 0:
        raise ValueError("valid_mask must select at least one position")
    if valid.numel() > 1 and not bool(torch.all(valid[1:] > valid[:-1])):
        raise ValueError(
            "valid logical positions must be strictly increasing in "
            "sequence order"
        )
    return (
        positions_cpu,
        mask_cpu,
        tuple(int(value) for value in positions_cpu.tolist()),
        tuple(bool(value) for value in mask_cpu.tolist()),
    )


def _validate_position_tuples(
    positions: object,
    mask: object,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    if (
        type(positions) is not tuple
        or not positions
        or any(type(value) is not int for value in positions)
    ):
        raise ValueError("logical_positions must be a nonempty integer tuple")
    if (
        type(mask) is not tuple
        or len(mask) != len(positions)
        or any(type(value) is not bool for value in mask)
    ):
        raise ValueError("valid_mask must be a matching boolean tuple")
    valid = tuple(
        position
        for position, selected in zip(positions, mask, strict=True)
        if selected
    )
    if not valid or any(
        right <= left for left, right in zip(valid, valid[1:])
    ):
        raise ValueError(
            "valid logical positions must be strictly increasing"
        )
    return positions, mask


def _strict_increasing_indices(
    values: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> tuple[int, ...]:
    if (
        type(values) is not tuple
        or not values
        or any(type(value) is not int for value in values)
    ):
        raise ValueError(f"{label} must be a nonempty integer tuple")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be strictly increasing")
    if minimum is not None and any(value < minimum for value in values):
        raise ValueError(f"{label} contains a value below its bound")
    if maximum is not None and any(value > maximum for value in values):
        raise ValueError(f"{label} contains a value above its bound")
    return values


def _maximum_fully_observed_lag(
    origins: tuple[int, ...],
    *,
    valid_positions: Sequence[int],
) -> int:
    valid = set(valid_positions)
    lag = 0
    while all(origin + lag in valid for origin in origins):
        lag += 1
    return lag - 1


def _function_output(
    function: Callable[[Tensor], Tensor],
    source: Tensor,
    *,
    sequence_length: int,
    target_rank: int | None,
) -> Tensor:
    with torch.no_grad():
        output = function(source)
    if not isinstance(output, Tensor):
        raise TypeError("mapping output must be a Tensor")
    if output.ndim != 3 or output.shape[:2] != (1, sequence_length):
        raise ValueError("mapping output must have shape [1, S, r_dst]")
    if output.shape[2] <= 0:
        raise ValueError("mapping target rank must be positive")
    if target_rank is not None and output.shape[2] != target_rank:
        raise ValueError("mapping target rank changed across evaluations")
    if not output.is_floating_point():
        raise TypeError("mapping output must use a floating dtype")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("mapping output must be finite")
    return output


def _rank_at_energy(singular_values: Tensor, fraction: float) -> int:
    energy = singular_values.square()
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, fraction).item()) + 1


def _spectral_ranks(
    mean_spectrum: Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int, int]:
    ranks = {0.90: [], 0.95: [], 0.99: []}
    for frequency in range(mean_spectrum.shape[1]):
        singular = torch.linalg.svdvals(mean_spectrum[:, frequency, :])
        for fraction in ranks:
            ranks[fraction].append(_rank_at_energy(singular, fraction))
    joint = mean_spectrum.reshape(mean_spectrum.shape[0], -1)
    joint_singular = torch.linalg.svdvals(joint)
    return (
        tuple(ranks[0.90]),
        tuple(ranks[0.95]),
        tuple(ranks[0.99]),
        _rank_at_energy(joint_singular, 0.90),
        _rank_at_energy(joint_singular, 0.95),
        _rank_at_energy(joint_singular, 0.99),
    )


def _normalized_rows(value: Tensor) -> Tensor:
    norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return torch.where(
        norms > torch.finfo(torch.float64).eps,
        value / torch.clamp_min(norms, torch.finfo(torch.float64).eps),
        torch.zeros_like(value),
    )


def connected_components_from_spectral_similarity(
    similarity: Tensor,
    *,
    source_mode_indices: Sequence[int],
    threshold: float,
) -> tuple[tuple[int, ...], ...]:
    """Deterministic components of a descriptive similarity graph."""

    matrix = _as_cpu_float64(
        similarity,
        label="similarity",
        dimensions=2,
    )
    labels = tuple(source_mode_indices)
    if (
        not labels
        or any(type(value) is not int or value < 0 for value in labels)
        or len(set(labels)) != len(labels)
        or matrix.shape != (len(labels), len(labels))
    ):
        raise ValueError("source_mode_indices do not match similarity")
    threshold = _finite(
        threshold,
        label="threshold",
        minimum=0.0,
        maximum=1.0,
    )
    if not torch.allclose(matrix, matrix.transpose(0, 1), atol=1e-10, rtol=0):
        raise ValueError("similarity must be symmetric")
    remaining = set(range(len(labels)))
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        selected: set[int] = set()
        while stack:
            index = stack.pop()
            if index in selected:
                continue
            selected.add(index)
            remaining.discard(index)
            neighbors = [
                candidate
                for candidate in sorted(remaining, reverse=True)
                if float(matrix[index, candidate]) >= threshold
            ]
            stack.extend(neighbors)
        components.append(tuple(sorted(labels[index] for index in selected)))
    return tuple(sorted(components, key=lambda component: component[0]))


@dataclass(frozen=True, slots=True)
class ModalSpectralResponse:
    """One amplitude regime of origin-bound modal impulse fingerprints."""

    label: str
    response_kind: str
    source_mode_indices: tuple[int, ...]
    impulse_logical_positions: tuple[int, ...]
    source_mode_amplitudes: tuple[float, ...]
    max_lag: int
    fft_length: int
    lag_observation_counts: tuple[int, ...]
    impulse_responses: Tensor
    even_residual_impulse_responses: Tensor | None
    spectral_fingerprint_real: Tensor
    spectral_fingerprint_imag: Tensor
    mean_spectral_fingerprint_real: Tensor
    mean_spectral_fingerprint_imag: Tensor
    magnitude: Tensor
    phase: Tensor
    coherence_like: Tensor
    normalized_signatures: Tensor
    pairwise_spectral_similarity: Tensor
    origin_spectral_similarity: Tensor
    lag_energy_fractions: Tensor
    impulse_response_sha256s: tuple[str, ...]
    per_frequency_rank_90: tuple[int, ...]
    per_frequency_rank_95: tuple[int, ...]
    per_frequency_rank_99: tuple[int, ...]
    joint_rank_90: int
    joint_rank_95: int
    joint_rank_99: int
    total_valid_response_frobenius: float
    causal_window_response_frobenius: float
    precausal_response_frobenius: float
    postwindow_response_frobenius: float
    even_residual_total_valid_frobenius: float
    even_residual_causal_window_frobenius: float
    relative_even_residual: float
    dc_energy_fraction: float
    low_energy_fraction: float
    mid_energy_fraction: float
    high_energy_fraction: float
    energy_beyond_lag4_fraction: float
    similarity_threshold: float
    function_evaluation_count: int
    no_causality_claim: str = _NO_CAUSALITY_CLAIM
    no_shift_invariance_claim: str = _NO_SHIFT_INVARIANCE_CLAIM
    spectral_semantics: str = _SPECTRAL_SEMANTICS
    coherence_semantics: str = _COHERENCE_SEMANTICS
    frequency_band_semantics: str = _FREQUENCY_BAND_SEMANTICS
    rank_semantics: str = _RANK_SEMANTICS
    clustering_semantics: str = _CLUSTERING_SEMANTICS
    connected_components: tuple[tuple[int, ...], ...] = ()
    artifact_sha256: str = ""
    artifact_kind: str = _RESPONSE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or _LABEL.fullmatch(self.label) is None:
            raise ValueError("response label is invalid")
        if self.response_kind not in (
            "one_sided_finite_secant",
            "symmetric_central_secant",
        ):
            raise ValueError("response_kind is invalid")
        modes = _strict_increasing_indices(
            self.source_mode_indices,
            label="source_mode_indices",
            minimum=0,
        )
        origins = _strict_increasing_indices(
            self.impulse_logical_positions,
            label="impulse_logical_positions",
        )
        if (
            type(self.source_mode_amplitudes) is not tuple
            or len(self.source_mode_amplitudes) != len(modes)
        ):
            raise ValueError(
                "source_mode_amplitudes must match source modes"
            )
        amplitudes = tuple(
            _finite(value, label="source mode amplitude", minimum=0.0)
            for value in self.source_mode_amplitudes
        )
        if any(value == 0.0 for value in amplitudes):
            raise ValueError("source mode amplitudes must be positive")
        object.__setattr__(self, "source_mode_amplitudes", amplitudes)
        max_lag = _require_nonnegative_int(self.max_lag, label="max_lag")
        fft_length = _require_positive_int(
            self.fft_length,
            label="fft_length",
        )
        if fft_length < max_lag + 1:
            raise ValueError("fft_length cannot truncate causal lags")
        if (
            type(self.lag_observation_counts) is not tuple
            or len(self.lag_observation_counts) != max_lag + 1
            or any(
                type(value) is not int
                or value < 0
                or value > len(origins)
                for value in self.lag_observation_counts
            )
        ):
            raise ValueError("lag_observation_counts are invalid")
        tensors: dict[str, Tensor | None] = {}
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if field == "even_residual_impulse_responses" and value is None:
                tensors[field] = None
                continue
            tensors[field] = _as_cpu_float64(
                value,  # type: ignore[arg-type]
                label=field,
                dimensions=(
                    4
                    if field
                    in (
                        "impulse_responses",
                        "even_residual_impulse_responses",
                        "spectral_fingerprint_real",
                        "spectral_fingerprint_imag",
                    )
                    else 3
                    if field
                    in (
                        "mean_spectral_fingerprint_real",
                        "mean_spectral_fingerprint_imag",
                        "magnitude",
                        "phase",
                        "coherence_like",
                        "origin_spectral_similarity",
                    )
                    else 2
                    if field
                    in (
                        "normalized_signatures",
                        "pairwise_spectral_similarity",
                    )
                    else 1
                ),
            )
        for field, value in tensors.items():
            object.__setattr__(self, field, value)
        responses = tensors["impulse_responses"]
        assert responses is not None
        mode_count = len(modes)
        origin_count = len(origins)
        if (
            responses.shape[:3]
            != (mode_count, origin_count, max_lag + 1)
        ):
            raise ValueError("impulse_responses shape is invalid")
        target_rank = int(responses.shape[3])
        frequency_count = fft_length // 2 + 1
        spectrum_shape = (
            mode_count,
            origin_count,
            frequency_count,
            target_rank,
        )
        for field in (
            "spectral_fingerprint_real",
            "spectral_fingerprint_imag",
        ):
            if tensors[field] is None or tensors[field].shape != spectrum_shape:
                raise ValueError(f"{field} shape is invalid")
        mean_shape = (mode_count, frequency_count, target_rank)
        for field in (
            "mean_spectral_fingerprint_real",
            "mean_spectral_fingerprint_imag",
            "magnitude",
            "phase",
            "coherence_like",
        ):
            if tensors[field] is None or tensors[field].shape != mean_shape:
                raise ValueError(f"{field} shape is invalid")
        even = tensors["even_residual_impulse_responses"]
        if self.response_kind == "symmetric_central_secant":
            if even is None or even.shape != responses.shape:
                raise ValueError(
                    "symmetric response requires an even residual tensor"
                )
        elif even is not None:
            raise ValueError(
                "one-sided response cannot carry an even residual tensor"
            )
        expected_signature_width = 3 * frequency_count * target_rank
        if tensors["normalized_signatures"].shape != (
            mode_count,
            expected_signature_width,
        ):
            raise ValueError("normalized_signatures shape is invalid")
        if tensors["pairwise_spectral_similarity"].shape != (
            mode_count,
            mode_count,
        ):
            raise ValueError("pairwise_spectral_similarity shape is invalid")
        if tensors["origin_spectral_similarity"].shape != (
            mode_count,
            origin_count,
            origin_count,
        ):
            raise ValueError("origin_spectral_similarity shape is invalid")
        if tensors["lag_energy_fractions"].shape != (max_lag + 1,):
            raise ValueError("lag_energy_fractions shape is invalid")
        if (
            type(self.impulse_response_sha256s) is not tuple
            or len(self.impulse_response_sha256s)
            != mode_count * origin_count
        ):
            raise ValueError("impulse response hashes are invalid")
        computed_response_hashes = tuple(
            _tensor_sha256(responses[mode, origin])
            for mode in range(mode_count)
            for origin in range(origin_count)
        )
        if self.impulse_response_sha256s != computed_response_hashes:
            raise ValueError("impulse response hashes do not match tensors")
        for digest in self.impulse_response_sha256s:
            _require_sha256(digest, label="impulse response hash")
        for field in (
            "per_frequency_rank_90",
            "per_frequency_rank_95",
            "per_frequency_rank_99",
        ):
            values = getattr(self, field)
            if (
                type(values) is not tuple
                or len(values) != frequency_count
                or any(
                    type(value) is not int
                    or value < 0
                    or value > min(mode_count, target_rank)
                    for value in values
                )
            ):
                raise ValueError(f"{field} is invalid")
        for field in ("joint_rank_90", "joint_rank_95", "joint_rank_99"):
            value = _require_nonnegative_int(getattr(self, field), label=field)
            if value > mode_count:
                raise ValueError(f"{field} exceeds the source-mode count")
        for field in (
            "total_valid_response_frobenius",
            "causal_window_response_frobenius",
            "precausal_response_frobenius",
            "postwindow_response_frobenius",
            "even_residual_total_valid_frobenius",
            "even_residual_causal_window_frobenius",
            "relative_even_residual",
            "dc_energy_fraction",
            "low_energy_fraction",
            "mid_energy_fraction",
            "high_energy_fraction",
            "energy_beyond_lag4_fraction",
        ):
            object.__setattr__(
                self,
                field,
                _finite(
                    getattr(self, field),
                    label=field,
                    minimum=0.0,
                    maximum=(
                        1.0
                        if field.endswith("_fraction")
                        and field != "relative_even_residual"
                        else None
                    ),
                ),
            )
        energy_sum = (
            self.precausal_response_frobenius**2
            + self.causal_window_response_frobenius**2
            + self.postwindow_response_frobenius**2
        )
        if not math.isclose(
            self.total_valid_response_frobenius**2,
            energy_sum,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("response energy accounting is inconsistent")
        if not math.isclose(
            self.dc_energy_fraction
            + self.low_energy_fraction
            + self.mid_energy_fraction
            + self.high_energy_fraction,
            (
                1.0
                if self.causal_window_response_frobenius > 0.0
                else 0.0
            ),
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError("spectral band energy fractions are inconsistent")
        threshold = _finite(
            self.similarity_threshold,
            label="similarity_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "similarity_threshold", threshold)
        expected_evaluations = mode_count * origin_count * (
            2 if self.response_kind == "symmetric_central_secant" else 1
        )
        if self.function_evaluation_count != expected_evaluations:
            raise ValueError("function evaluation accounting is invalid")
        similarity = tensors["pairwise_spectral_similarity"]
        assert similarity is not None
        expected_components = connected_components_from_spectral_similarity(
            similarity,
            source_mode_indices=modes,
            threshold=threshold,
        )
        if self.connected_components != expected_components:
            raise ValueError("connected components do not match similarity")
        if self.no_causality_claim != _NO_CAUSALITY_CLAIM:
            raise ValueError("no_causality_claim provenance is invalid")
        if self.no_shift_invariance_claim != _NO_SHIFT_INVARIANCE_CLAIM:
            raise ValueError(
                "no_shift_invariance_claim provenance is invalid"
            )
        if self.spectral_semantics != _SPECTRAL_SEMANTICS:
            raise ValueError("spectral_semantics provenance is invalid")
        if self.coherence_semantics != _COHERENCE_SEMANTICS:
            raise ValueError("coherence_semantics provenance is invalid")
        if self.frequency_band_semantics != _FREQUENCY_BAND_SEMANTICS:
            raise ValueError(
                "frequency_band_semantics provenance is invalid"
            )
        if self.rank_semantics != _RANK_SEMANTICS:
            raise ValueError("rank_semantics provenance is invalid")
        if self.clustering_semantics != _CLUSTERING_SEMANTICS:
            raise ValueError("clustering_semantics provenance is invalid")
        if (
            self.artifact_kind != _RESPONSE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal spectral response header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("modal spectral response hash mismatch")

    @property
    def source_mode_count(self) -> int:
        return len(self.source_mode_indices)

    @property
    def impulse_origin_count(self) -> int:
        return len(self.impulse_logical_positions)

    @property
    def target_rank(self) -> int:
        return int(self.impulse_responses.shape[3])

    @property
    def frequency_count(self) -> int:
        return self.fft_length // 2 + 1

    @property
    def mean_origin_spectral_similarity(self) -> float:
        if self.impulse_origin_count <= 1:
            return 1.0
        mask = ~torch.eye(
            self.impulse_origin_count,
            dtype=torch.bool,
        ).unsqueeze(0)
        values = self.origin_spectral_similarity[mask.expand_as(
            self.origin_spectral_similarity
        )]
        return float(values.mean())

    @property
    def minimum_origin_spectral_similarity(self) -> float:
        if self.impulse_origin_count <= 1:
            return 1.0
        mask = ~torch.eye(
            self.impulse_origin_count,
            dtype=torch.bool,
        ).unsqueeze(0)
        values = self.origin_spectral_similarity[mask.expand_as(
            self.origin_spectral_similarity
        )]
        return float(values.min())

    @property
    def serialized_tensor_scalar_count(self) -> int:
        total = 0
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None:
                total += value.numel()
        return int(total)

    def _hash_payload(self) -> dict[str, object]:
        tensor_bindings = {
            f"{field}_sha256": (
                None
                if getattr(self, field) is None
                else _tensor_sha256(getattr(self, field))
            )
            for field in _TENSOR_FIELDS
        }
        tensor_shapes = {
            f"{field}_shape": (
                None
                if getattr(self, field) is None
                else tuple(int(value) for value in getattr(self, field).shape)
            )
            for field in _TENSOR_FIELDS
        }
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "label": self.label,
            "response_kind": self.response_kind,
            "source_mode_indices": self.source_mode_indices,
            "impulse_logical_positions": self.impulse_logical_positions,
            "source_mode_amplitudes": self.source_mode_amplitudes,
            "max_lag": self.max_lag,
            "fft_length": self.fft_length,
            "lag_observation_counts": self.lag_observation_counts,
            **tensor_bindings,
            **tensor_shapes,
            "impulse_response_sha256s": self.impulse_response_sha256s,
            "per_frequency_rank_90": self.per_frequency_rank_90,
            "per_frequency_rank_95": self.per_frequency_rank_95,
            "per_frequency_rank_99": self.per_frequency_rank_99,
            "joint_rank_90": self.joint_rank_90,
            "joint_rank_95": self.joint_rank_95,
            "joint_rank_99": self.joint_rank_99,
            "total_valid_response_frobenius": (
                self.total_valid_response_frobenius
            ),
            "causal_window_response_frobenius": (
                self.causal_window_response_frobenius
            ),
            "precausal_response_frobenius": (
                self.precausal_response_frobenius
            ),
            "postwindow_response_frobenius": (
                self.postwindow_response_frobenius
            ),
            "even_residual_total_valid_frobenius": (
                self.even_residual_total_valid_frobenius
            ),
            "even_residual_causal_window_frobenius": (
                self.even_residual_causal_window_frobenius
            ),
            "relative_even_residual": self.relative_even_residual,
            "dc_energy_fraction": self.dc_energy_fraction,
            "low_energy_fraction": self.low_energy_fraction,
            "mid_energy_fraction": self.mid_energy_fraction,
            "high_energy_fraction": self.high_energy_fraction,
            "energy_beyond_lag4_fraction": (
                self.energy_beyond_lag4_fraction
            ),
            "similarity_threshold": self.similarity_threshold,
            "function_evaluation_count": self.function_evaluation_count,
            "no_causality_claim": self.no_causality_claim,
            "no_shift_invariance_claim": self.no_shift_invariance_claim,
            "spectral_semantics": self.spectral_semantics,
            "coherence_semantics": self.coherence_semantics,
            "frequency_band_semantics": self.frequency_band_semantics,
            "rank_semantics": self.rank_semantics,
            "clustering_semantics": self.clustering_semantics,
            "connected_components": self.connected_components,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_RESPONSE_HASH_DOMAIN,
        )

    def validate_integrity(self) -> None:
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError("modal spectral response tensor is nonfinite")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("modal spectral response integrity check failed")

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "source_mode_count": self.source_mode_count,
            "impulse_origin_count": self.impulse_origin_count,
            "target_rank": self.target_rank,
            "frequency_count": self.frequency_count,
            "mean_origin_spectral_similarity": (
                self.mean_origin_spectral_similarity
            ),
            "minimum_origin_spectral_similarity": (
                self.minimum_origin_spectral_similarity
            ),
            "serialized_tensor_scalar_count": (
                self.serialized_tensor_scalar_count
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            **{
                field: (
                    None
                    if getattr(self, field) is None
                    else getattr(self, field).clone()
                )
                for field in _TENSOR_FIELDS
            },
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalSpectralResponse:
        scalar_fields = {
            "artifact_kind",
            "format_version",
            "label",
            "response_kind",
            "source_mode_indices",
            "impulse_logical_positions",
            "source_mode_amplitudes",
            "max_lag",
            "fft_length",
            "lag_observation_counts",
            "impulse_response_sha256s",
            "per_frequency_rank_90",
            "per_frequency_rank_95",
            "per_frequency_rank_99",
            "joint_rank_90",
            "joint_rank_95",
            "joint_rank_99",
            "total_valid_response_frobenius",
            "causal_window_response_frobenius",
            "precausal_response_frobenius",
            "postwindow_response_frobenius",
            "even_residual_total_valid_frobenius",
            "even_residual_causal_window_frobenius",
            "relative_even_residual",
            "dc_energy_fraction",
            "low_energy_fraction",
            "mid_energy_fraction",
            "high_energy_fraction",
            "energy_beyond_lag4_fraction",
            "similarity_threshold",
            "function_evaluation_count",
            "no_causality_claim",
            "no_shift_invariance_claim",
            "spectral_semantics",
            "coherence_semantics",
            "frequency_band_semantics",
            "rank_semantics",
            "clustering_semantics",
            "connected_components",
            "artifact_sha256",
        }
        expected = scalar_fields | set(_TENSOR_FIELDS) | {
            f"{field}_sha256" for field in _TENSOR_FIELDS
        } | {
            f"{field}_shape" for field in _TENSOR_FIELDS
        }
        _strict_keys(
            state,
            expected=expected,
            label="modal spectral response",
        )
        tensors: dict[str, Tensor | None] = {}
        for field in _TENSOR_FIELDS:
            value = state[field]
            expected_hash = state[f"{field}_sha256"]
            expected_shape = state[f"{field}_shape"]
            if value is None:
                if expected_hash is not None or expected_shape is not None:
                    raise ValueError(f"serialized {field} null binding drifted")
                tensors[field] = None
                continue
            if not isinstance(value, Tensor):
                raise TypeError(f"serialized {field} must be a Tensor")
            if (
                not isinstance(expected_shape, tuple)
                or tuple(value.shape) != expected_shape
                or _tensor_sha256(value) != expected_hash
            ):
                raise ValueError(f"serialized {field} binding mismatch")
            tensors[field] = value
        return cls(
            label=state["label"],  # type: ignore[arg-type]
            response_kind=state["response_kind"],  # type: ignore[arg-type]
            source_mode_indices=state[
                "source_mode_indices"
            ],  # type: ignore[arg-type]
            impulse_logical_positions=state[
                "impulse_logical_positions"
            ],  # type: ignore[arg-type]
            source_mode_amplitudes=state[
                "source_mode_amplitudes"
            ],  # type: ignore[arg-type]
            max_lag=state["max_lag"],  # type: ignore[arg-type]
            fft_length=state["fft_length"],  # type: ignore[arg-type]
            lag_observation_counts=state[
                "lag_observation_counts"
            ],  # type: ignore[arg-type]
            **tensors,
            impulse_response_sha256s=state[
                "impulse_response_sha256s"
            ],  # type: ignore[arg-type]
            per_frequency_rank_90=state[
                "per_frequency_rank_90"
            ],  # type: ignore[arg-type]
            per_frequency_rank_95=state[
                "per_frequency_rank_95"
            ],  # type: ignore[arg-type]
            per_frequency_rank_99=state[
                "per_frequency_rank_99"
            ],  # type: ignore[arg-type]
            joint_rank_90=state["joint_rank_90"],  # type: ignore[arg-type]
            joint_rank_95=state["joint_rank_95"],  # type: ignore[arg-type]
            joint_rank_99=state["joint_rank_99"],  # type: ignore[arg-type]
            total_valid_response_frobenius=state[
                "total_valid_response_frobenius"
            ],  # type: ignore[arg-type]
            causal_window_response_frobenius=state[
                "causal_window_response_frobenius"
            ],  # type: ignore[arg-type]
            precausal_response_frobenius=state[
                "precausal_response_frobenius"
            ],  # type: ignore[arg-type]
            postwindow_response_frobenius=state[
                "postwindow_response_frobenius"
            ],  # type: ignore[arg-type]
            even_residual_total_valid_frobenius=state[
                "even_residual_total_valid_frobenius"
            ],  # type: ignore[arg-type]
            even_residual_causal_window_frobenius=state[
                "even_residual_causal_window_frobenius"
            ],  # type: ignore[arg-type]
            relative_even_residual=state[
                "relative_even_residual"
            ],  # type: ignore[arg-type]
            dc_energy_fraction=state[
                "dc_energy_fraction"
            ],  # type: ignore[arg-type]
            low_energy_fraction=state[
                "low_energy_fraction"
            ],  # type: ignore[arg-type]
            mid_energy_fraction=state[
                "mid_energy_fraction"
            ],  # type: ignore[arg-type]
            high_energy_fraction=state[
                "high_energy_fraction"
            ],  # type: ignore[arg-type]
            energy_beyond_lag4_fraction=state[
                "energy_beyond_lag4_fraction"
            ],  # type: ignore[arg-type]
            similarity_threshold=state[
                "similarity_threshold"
            ],  # type: ignore[arg-type]
            function_evaluation_count=state[
                "function_evaluation_count"
            ],  # type: ignore[arg-type]
            no_causality_claim=state[
                "no_causality_claim"
            ],  # type: ignore[arg-type]
            no_shift_invariance_claim=state[
                "no_shift_invariance_claim"
            ],  # type: ignore[arg-type]
            spectral_semantics=state[
                "spectral_semantics"
            ],  # type: ignore[arg-type]
            coherence_semantics=state[
                "coherence_semantics"
            ],  # type: ignore[arg-type]
            frequency_band_semantics=state[
                "frequency_band_semantics"
            ],  # type: ignore[arg-type]
            rank_semantics=state["rank_semantics"],  # type: ignore[arg-type]
            clustering_semantics=state[
                "clustering_semantics"
            ],  # type: ignore[arg-type]
            connected_components=state[
                "connected_components"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModalSpectralMapping:
    """Authenticated reference-bound collection of spectral responses."""

    baseline_modes_sha256: str
    baseline_output_sha256: str
    logical_positions: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    source_rank: int
    target_rank: int
    source_mode_indices: tuple[int, ...]
    impulse_logical_positions: tuple[int, ...]
    max_lag: int
    fft_length: int
    finite: ModalSpectralResponse
    symmetric_responses: tuple[ModalSpectralResponse, ...]
    symmetric_scale_pair_similarity: Tensor
    baseline_function_evaluation_count: int
    function_evaluation_count: int
    no_causality_claim: str = _NO_CAUSALITY_CLAIM
    reference_dependence: str = (
        "all_responses_are_bound_to_the_exact_baseline_modes_and_output"
    )
    artifact_sha256: str = ""
    artifact_kind: str = _MAPPING_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.baseline_modes_sha256,
            label="baseline_modes_sha256",
        )
        _require_sha256(
            self.baseline_output_sha256,
            label="baseline_output_sha256",
        )
        _validate_position_tuples(self.logical_positions, self.valid_mask)
        source_rank = _require_positive_int(
            self.source_rank,
            label="source_rank",
        )
        target_rank = _require_positive_int(
            self.target_rank,
            label="target_rank",
        )
        modes = _strict_increasing_indices(
            self.source_mode_indices,
            label="source_mode_indices",
            minimum=0,
            maximum=source_rank - 1,
        )
        valid_positions = {
            position
            for position, selected in zip(
                self.logical_positions,
                self.valid_mask,
                strict=True,
            )
            if selected
        }
        origins = _strict_increasing_indices(
            self.impulse_logical_positions,
            label="impulse_logical_positions",
        )
        if not set(origins).issubset(valid_positions):
            raise ValueError("impulse logical positions must be valid")
        max_lag = _require_nonnegative_int(self.max_lag, label="max_lag")
        if (
            len(origins) > 1
            and max_lag
            > _maximum_fully_observed_lag(
                origins,
                valid_positions=tuple(valid_positions),
            )
        ):
            raise ValueError(
                "max_lag must be fully observed from every impulse origin"
            )
        fft_length = _require_positive_int(
            self.fft_length,
            label="fft_length",
        )
        if fft_length < max_lag + 1:
            raise ValueError("fft_length cannot truncate causal lags")
        if not isinstance(self.finite, ModalSpectralResponse):
            raise TypeError("finite must be a ModalSpectralResponse")
        if (
            type(self.symmetric_responses) is not tuple
            or any(
                not isinstance(response, ModalSpectralResponse)
                for response in self.symmetric_responses
            )
        ):
            raise TypeError(
                "symmetric_responses must contain spectral responses"
            )
        labels = tuple(response.label for response in self.symmetric_responses)
        if labels != tuple(sorted(labels)) or len(labels) != len(set(labels)):
            raise ValueError(
                "symmetric response labels must be sorted and unique"
            )
        for response in (self.finite, *self.symmetric_responses):
            response.validate_integrity()
            if (
                response.source_mode_indices != modes
                or response.impulse_logical_positions != origins
                or response.max_lag != max_lag
                or response.fft_length != fft_length
                or response.target_rank != target_rank
            ):
                raise ValueError("spectral response binding drifted")
        if self.finite.response_kind != "one_sided_finite_secant":
            raise ValueError("finite response kind is invalid")
        if any(
            response.response_kind != "symmetric_central_secant"
            for response in self.symmetric_responses
        ):
            raise ValueError("symmetric response kind is invalid")
        scale_similarity = _as_cpu_float64(
            self.symmetric_scale_pair_similarity,
            label="symmetric_scale_pair_similarity",
            dimensions=3,
            allow_empty=True,
        )
        expected_shape = (len(labels), len(labels), len(modes))
        if scale_similarity.shape != expected_shape:
            raise ValueError(
                "symmetric_scale_pair_similarity shape is invalid"
            )
        object.__setattr__(
            self,
            "symmetric_scale_pair_similarity",
            scale_similarity,
        )
        if self.baseline_function_evaluation_count != 1:
            raise ValueError("baseline function evaluation count must be one")
        expected_evaluations = 1 + self.finite.function_evaluation_count + sum(
            response.function_evaluation_count
            for response in self.symmetric_responses
        )
        if self.function_evaluation_count != expected_evaluations:
            raise ValueError("total function evaluation count is invalid")
        if self.no_causality_claim != _NO_CAUSALITY_CLAIM:
            raise ValueError("no_causality_claim provenance is invalid")
        if self.reference_dependence != (
            "all_responses_are_bound_to_the_exact_baseline_modes_and_output"
        ):
            raise ValueError("reference_dependence provenance is invalid")
        if (
            self.artifact_kind != _MAPPING_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal spectral mapping header is invalid")
        computed = self._computed_sha256()
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("modal spectral mapping hash mismatch")

    @property
    def symmetric_by_label(self) -> dict[str, ModalSpectralResponse]:
        return {
            response.label: response for response in self.symmetric_responses
        }

    @property
    def symmetric_labels(self) -> tuple[str, ...]:
        return tuple(
            response.label for response in self.symmetric_responses
        )

    def scale_similarity(self, left_label: str, right_label: str) -> Tensor:
        labels = self.symmetric_labels
        try:
            left = labels.index(left_label)
            right = labels.index(right_label)
        except ValueError as error:
            raise KeyError("unknown symmetric response label") from error
        return self.symmetric_scale_pair_similarity[left, right].clone()

    def validate_reference(
        self,
        *,
        baseline_modes: Tensor,
        baseline_output: Tensor | None = None,
    ) -> None:
        """Authenticate the exact modal reference used by this mapping."""

        self.validate_integrity()
        if not isinstance(baseline_modes, Tensor):
            raise TypeError("baseline_modes must be a Tensor")
        if baseline_modes.shape != (
            1,
            len(self.logical_positions),
            self.source_rank,
        ):
            raise ValueError("baseline_modes shape does not match the mapping")
        if _tensor_sha256(baseline_modes) != self.baseline_modes_sha256:
            raise ValueError("baseline_modes do not match the mapping")
        if baseline_output is not None:
            if not isinstance(baseline_output, Tensor):
                raise TypeError("baseline_output must be a Tensor")
            if baseline_output.shape != (
                1,
                len(self.logical_positions),
                self.target_rank,
            ):
                raise ValueError(
                    "baseline_output shape does not match the mapping"
                )
            if (
                _tensor_sha256(baseline_output)
                != self.baseline_output_sha256
            ):
                raise ValueError("baseline_output does not match the mapping")

    def accounting(self) -> dict[str, object]:
        response_scalars = self.finite.serialized_tensor_scalar_count + sum(
            response.serialized_tensor_scalar_count
            for response in self.symmetric_responses
        )
        scale_scalars = self.symmetric_scale_pair_similarity.numel()
        experiment_count = (
            len(self.source_mode_indices)
            * len(self.impulse_logical_positions)
        )
        return {
            "baseline_function_evaluations": 1,
            "finite_function_evaluations": (
                self.finite.function_evaluation_count
            ),
            "symmetric_function_evaluations": sum(
                response.function_evaluation_count
                for response in self.symmetric_responses
            ),
            "function_evaluation_count": self.function_evaluation_count,
            "impulse_experiment_count_per_scale": experiment_count,
            "symmetric_scale_count": len(self.symmetric_responses),
            "serialized_tensor_scalar_count": (
                response_scalars + scale_scalars
            ),
            "serialized_tensor_bytes_float64": (
                response_scalars + scale_scalars
            )
            * 8,
            "mapping_function_macs": None,
            "mapping_function_parameters": None,
            "runtime_speedup_claim": False,
            "causality_claim": False,
            "shift_invariance_claim": False,
        }

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "baseline_modes_sha256": self.baseline_modes_sha256,
            "baseline_output_sha256": self.baseline_output_sha256,
            "logical_positions": self.logical_positions,
            "valid_mask": self.valid_mask,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "source_mode_indices": self.source_mode_indices,
            "impulse_logical_positions": self.impulse_logical_positions,
            "max_lag": self.max_lag,
            "fft_length": self.fft_length,
            "finite_artifact_sha256": self.finite.artifact_sha256,
            "symmetric_artifact_sha256s": tuple(
                response.artifact_sha256
                for response in self.symmetric_responses
            ),
            "symmetric_labels": self.symmetric_labels,
            "symmetric_scale_pair_similarity_sha256": _tensor_sha256(
                self.symmetric_scale_pair_similarity
            ),
            "symmetric_scale_pair_similarity_shape": tuple(
                int(value)
                for value in self.symmetric_scale_pair_similarity.shape
            ),
            "baseline_function_evaluation_count": (
                self.baseline_function_evaluation_count
            ),
            "function_evaluation_count": self.function_evaluation_count,
            "no_causality_claim": self.no_causality_claim,
            "reference_dependence": self.reference_dependence,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(
            self._hash_payload(),
            domain=_MAPPING_HASH_DOMAIN,
        )

    def validate_integrity(self) -> None:
        self.finite.validate_integrity()
        for response in self.symmetric_responses:
            response.validate_integrity()
        if (
            not bool(torch.isfinite(
                self.symmetric_scale_pair_similarity
            ).all())
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise ValueError("modal spectral mapping integrity check failed")

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "artifact_sha256": self.artifact_sha256,
            "finite": self.finite.metadata(),
            "symmetric": tuple(
                response.metadata()
                for response in self.symmetric_responses
            ),
            "accounting": self.accounting(),
            "contains_prompt_text": False,
            "reference_bound": True,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "finite": self.finite.state_dict(),
            "symmetric_responses": tuple(
                response.state_dict()
                for response in self.symmetric_responses
            ),
            "symmetric_scale_pair_similarity": (
                self.symmetric_scale_pair_similarity.clone()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    to_state_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalSpectralMapping:
        expected = {
            "artifact_kind",
            "format_version",
            "baseline_modes_sha256",
            "baseline_output_sha256",
            "logical_positions",
            "valid_mask",
            "source_rank",
            "target_rank",
            "source_mode_indices",
            "impulse_logical_positions",
            "max_lag",
            "fft_length",
            "finite_artifact_sha256",
            "symmetric_artifact_sha256s",
            "symmetric_labels",
            "symmetric_scale_pair_similarity_sha256",
            "symmetric_scale_pair_similarity_shape",
            "baseline_function_evaluation_count",
            "function_evaluation_count",
            "no_causality_claim",
            "reference_dependence",
            "finite",
            "symmetric_responses",
            "symmetric_scale_pair_similarity",
            "artifact_sha256",
        }
        _strict_keys(
            state,
            expected=expected,
            label="modal spectral mapping",
        )
        finite_state = state["finite"]
        symmetric_states = state["symmetric_responses"]
        scale_similarity = state["symmetric_scale_pair_similarity"]
        if not isinstance(finite_state, Mapping):
            raise TypeError("serialized finite response must be a mapping")
        if (
            type(symmetric_states) is not tuple
            or any(not isinstance(value, Mapping) for value in symmetric_states)
        ):
            raise TypeError(
                "serialized symmetric responses must be mappings"
            )
        if not isinstance(scale_similarity, Tensor):
            raise TypeError(
                "serialized symmetric scale similarity must be a Tensor"
            )
        if (
            tuple(scale_similarity.shape)
            != state["symmetric_scale_pair_similarity_shape"]
            or _tensor_sha256(scale_similarity)
            != state["symmetric_scale_pair_similarity_sha256"]
        ):
            raise ValueError(
                "serialized symmetric scale similarity binding mismatch"
            )
        finite = ModalSpectralResponse.from_state_dict(finite_state)
        symmetric = tuple(
            ModalSpectralResponse.from_state_dict(value)
            for value in symmetric_states
        )
        if finite.artifact_sha256 != state["finite_artifact_sha256"]:
            raise ValueError("serialized finite response binding drifted")
        if (
            tuple(value.artifact_sha256 for value in symmetric)
            != state["symmetric_artifact_sha256s"]
            or tuple(value.label for value in symmetric)
            != state["symmetric_labels"]
        ):
            raise ValueError("serialized symmetric response binding drifted")
        return cls(
            baseline_modes_sha256=state[
                "baseline_modes_sha256"
            ],  # type: ignore[arg-type]
            baseline_output_sha256=state[
                "baseline_output_sha256"
            ],  # type: ignore[arg-type]
            logical_positions=state[
                "logical_positions"
            ],  # type: ignore[arg-type]
            valid_mask=state["valid_mask"],  # type: ignore[arg-type]
            source_rank=state["source_rank"],  # type: ignore[arg-type]
            target_rank=state["target_rank"],  # type: ignore[arg-type]
            source_mode_indices=state[
                "source_mode_indices"
            ],  # type: ignore[arg-type]
            impulse_logical_positions=state[
                "impulse_logical_positions"
            ],  # type: ignore[arg-type]
            max_lag=state["max_lag"],  # type: ignore[arg-type]
            fft_length=state["fft_length"],  # type: ignore[arg-type]
            finite=finite,
            symmetric_responses=symmetric,
            symmetric_scale_pair_similarity=scale_similarity,
            baseline_function_evaluation_count=state[
                "baseline_function_evaluation_count"
            ],  # type: ignore[arg-type]
            function_evaluation_count=state[
                "function_evaluation_count"
            ],  # type: ignore[arg-type]
            no_causality_claim=state[
                "no_causality_claim"
            ],  # type: ignore[arg-type]
            reference_dependence=state[
                "reference_dependence"
            ],  # type: ignore[arg-type]
            artifact_sha256=state[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


def _resolve_modes(
    source_mode_indices: Sequence[int] | None,
    *,
    source_rank: int,
) -> tuple[int, ...]:
    if source_mode_indices is None:
        return tuple(range(source_rank))
    if isinstance(source_mode_indices, (str, bytes)):
        raise TypeError("source_mode_indices must be integer indices")
    values = tuple(source_mode_indices)
    return _strict_increasing_indices(
        values,
        label="source_mode_indices",
        minimum=0,
        maximum=source_rank - 1,
    )


def _resolve_origins(
    impulse_logical_positions: Sequence[int] | None,
    *,
    valid_positions: tuple[int, ...],
) -> tuple[int, ...]:
    if impulse_logical_positions is None:
        return valid_positions
    if isinstance(impulse_logical_positions, (str, bytes)):
        raise TypeError("impulse_logical_positions must be integers")
    values = _strict_increasing_indices(
        tuple(impulse_logical_positions),
        label="impulse_logical_positions",
    )
    if not set(values).issubset(valid_positions):
        raise ValueError("impulse logical positions must be valid")
    return values


def _resolve_amplitudes(
    value: float | Tensor,
    *,
    source_rank: int,
    source_mode_indices: tuple[int, ...],
    label: str,
) -> tuple[float, ...]:
    if isinstance(value, Tensor):
        if (
            value.ndim != 1
            or value.shape[0] != source_rank
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{label} must have shape [r_src] and be finite")
        values = value.detach().to(device="cpu", dtype=torch.float64)
        selected = tuple(float(values[index]) for index in source_mode_indices)
    else:
        amplitude = _finite(value, label=label, minimum=0.0)
        selected = (amplitude,) * len(source_mode_indices)
    if any(amplitude <= 0.0 for amplitude in selected):
        raise ValueError(f"{label} values must be positive")
    return selected


def _response_hashes(responses: Tensor) -> tuple[str, ...]:
    return tuple(
        _tensor_sha256(responses[mode, origin])
        for mode in range(responses.shape[0])
        for origin in range(responses.shape[1])
    )


def _spectral_products(
    responses: Tensor,
    *,
    fft_length: int,
    source_mode_indices: tuple[int, ...],
    similarity_threshold: float,
) -> dict[str, object]:
    spectrum = torch.fft.rfft(responses, n=fft_length, dim=2)
    mean_spectrum = spectrum.mean(dim=1)
    magnitude = torch.abs(mean_spectrum)
    phase = torch.angle(mean_spectrum)
    mean_power = torch.mean(torch.abs(spectrum).square(), dim=1)
    coherence = torch.where(
        mean_power > torch.finfo(torch.float64).eps,
        torch.abs(mean_spectrum).square()
        / torch.clamp_min(mean_power, torch.finfo(torch.float64).eps),
        torch.zeros_like(mean_power),
    ).clamp(0.0, 1.0)
    magnitude_rows = magnitude.reshape(magnitude.shape[0], -1)
    normalized_magnitude = _normalized_rows(magnitude_rows)
    phase_cos = (
        normalized_magnitude
        * coherence.reshape(coherence.shape[0], -1)
        * torch.cos(phase).reshape(phase.shape[0], -1)
    )
    phase_sin = (
        normalized_magnitude
        * coherence.reshape(coherence.shape[0], -1)
        * torch.sin(phase).reshape(phase.shape[0], -1)
    )
    signatures = _normalized_rows(
        torch.cat((normalized_magnitude, phase_cos, phase_sin), dim=1)
    )
    pairwise = (signatures @ signatures.transpose(0, 1)).clamp(-1.0, 1.0)
    pairwise.fill_diagonal_(1.0)
    mode_count, origin_count = spectrum.shape[:2]
    origin_similarity = torch.empty(
        (mode_count, origin_count, origin_count),
        dtype=torch.float64,
    )
    for mode in range(mode_count):
        origin_vectors = torch.cat(
            (
                spectrum[mode].real.reshape(origin_count, -1),
                spectrum[mode].imag.reshape(origin_count, -1),
            ),
            dim=1,
        )
        normalized = _normalized_rows(origin_vectors)
        matrix = (normalized @ normalized.transpose(0, 1)).clamp(-1.0, 1.0)
        matrix.fill_diagonal_(1.0)
        origin_similarity[mode] = matrix
    lag_energy = responses.square().sum(dim=(0, 1, 3))
    lag_total = float(lag_energy.sum())
    lag_fractions = (
        lag_energy / lag_total
        if lag_total > torch.finfo(torch.float64).eps
        else torch.zeros_like(lag_energy)
    )
    frequencies = torch.fft.rfftfreq(fft_length, d=1.0)
    weights = torch.full_like(frequencies, 2.0)
    weights[0] = 1.0
    if fft_length % 2 == 0:
        weights[-1] = 1.0
    spectral_energy = (
        torch.abs(spectrum).square() * weights.reshape(1, 1, -1, 1)
    ).sum(dim=(0, 1, 3))
    total_spectral = float(spectral_energy.sum())
    band_masks = (
        frequencies == 0.0,
        (frequencies > 0.0) & (frequencies <= 1.0 / 6.0),
        (frequencies > 1.0 / 6.0) & (frequencies <= 1.0 / 3.0),
        frequencies > 1.0 / 3.0,
    )
    if total_spectral > torch.finfo(torch.float64).eps:
        band_fractions = tuple(
            float(spectral_energy[mask].sum()) / total_spectral
            for mask in band_masks
        )
    else:
        band_fractions = (0.0, 0.0, 0.0, 0.0)
    (
        rank_90,
        rank_95,
        rank_99,
        joint_90,
        joint_95,
        joint_99,
    ) = _spectral_ranks(mean_spectrum)
    return {
        "spectral_fingerprint_real": spectrum.real,
        "spectral_fingerprint_imag": spectrum.imag,
        "mean_spectral_fingerprint_real": mean_spectrum.real,
        "mean_spectral_fingerprint_imag": mean_spectrum.imag,
        "magnitude": magnitude,
        "phase": phase,
        "coherence_like": coherence,
        "normalized_signatures": signatures,
        "pairwise_spectral_similarity": pairwise,
        "origin_spectral_similarity": origin_similarity,
        "lag_energy_fractions": lag_fractions,
        "per_frequency_rank_90": rank_90,
        "per_frequency_rank_95": rank_95,
        "per_frequency_rank_99": rank_99,
        "joint_rank_90": joint_90,
        "joint_rank_95": joint_95,
        "joint_rank_99": joint_99,
        "dc_energy_fraction": band_fractions[0],
        "low_energy_fraction": band_fractions[1],
        "mid_energy_fraction": band_fractions[2],
        "high_energy_fraction": band_fractions[3],
        "energy_beyond_lag4_fraction": (
            float(lag_fractions[5:].sum())
            if lag_fractions.numel() > 5
            else 0.0
        ),
        "connected_components": (
            connected_components_from_spectral_similarity(
                pairwise,
                source_mode_indices=source_mode_indices,
                threshold=similarity_threshold,
            )
        ),
    }


def _measure_response(
    function: Callable[[Tensor], Tensor],
    *,
    baseline: Tensor,
    baseline_output: Tensor,
    positions: Tensor,
    mask: Tensor,
    source_mode_indices: tuple[int, ...],
    origins: tuple[int, ...],
    amplitudes: tuple[float, ...],
    max_lag: int,
    fft_length: int,
    label: str,
    symmetric: bool,
    similarity_threshold: float,
) -> ModalSpectralResponse:
    target_rank = int(baseline_output.shape[2])
    position_to_index = {
        int(positions[index]): int(index)
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist()
    }
    valid_indices = tuple(sorted(position_to_index.values()))
    responses = torch.zeros(
        (
            len(source_mode_indices),
            len(origins),
            max_lag + 1,
            target_rank,
        ),
        dtype=torch.float64,
    )
    even_responses = torch.zeros_like(responses) if symmetric else None
    precausal_squared = 0.0
    causal_squared = 0.0
    postwindow_squared = 0.0
    total_squared = 0.0
    all_causal_squared = 0.0
    beyond_lag4_squared = 0.0
    even_total_squared = 0.0
    even_causal_squared = 0.0
    for mode_ordinal, (source_mode, amplitude) in enumerate(
        zip(source_mode_indices, amplitudes, strict=True)
    ):
        for origin_ordinal, origin in enumerate(origins):
            impulse = torch.zeros_like(baseline)
            impulse[0, position_to_index[origin], source_mode] = amplitude
            plus = _function_output(
                function,
                baseline + impulse,
                sequence_length=baseline.shape[1],
                target_rank=target_rank,
            )
            if symmetric:
                minus = _function_output(
                    function,
                    baseline - impulse,
                    sequence_length=baseline.shape[1],
                    target_rank=target_rank,
                )
                normalized = (plus - minus) / (2.0 * amplitude)
                even = (
                    plus + minus - 2.0 * baseline_output
                ) / (2.0 * amplitude)
            else:
                normalized = (plus - baseline_output) / amplitude
                even = None
            selected = normalized[0].detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            selected_even = (
                None
                if even is None
                else even[0].detach().to(device="cpu", dtype=torch.float64)
            )
            for target_index in valid_indices:
                lag = int(positions[target_index]) - origin
                row_squared = float(selected[target_index].square().sum())
                total_squared += row_squared
                if lag < 0:
                    precausal_squared += row_squared
                else:
                    all_causal_squared += row_squared
                    if lag > 4:
                        beyond_lag4_squared += row_squared
                    if lag <= max_lag:
                        responses[
                            mode_ordinal,
                            origin_ordinal,
                            lag,
                        ] = selected[target_index]
                        causal_squared += row_squared
                    else:
                        postwindow_squared += row_squared
                if selected_even is not None:
                    even_row_squared = float(
                        selected_even[target_index].square().sum()
                    )
                    even_total_squared += even_row_squared
                    if 0 <= lag <= max_lag:
                        assert even_responses is not None
                        even_responses[
                            mode_ordinal,
                            origin_ordinal,
                            lag,
                        ] = selected_even[target_index]
                        even_causal_squared += even_row_squared
    products = _spectral_products(
        responses,
        fft_length=fft_length,
        source_mode_indices=source_mode_indices,
        similarity_threshold=similarity_threshold,
    )
    products["energy_beyond_lag4_fraction"] = (
        beyond_lag4_squared / all_causal_squared
        if all_causal_squared > torch.finfo(torch.float64).eps
        else 0.0
    )
    lag_counts = tuple(
        sum((origin + lag) in position_to_index for origin in origins)
        for lag in range(max_lag + 1)
    )
    total_frobenius = math.sqrt(total_squared)
    even_total_frobenius = math.sqrt(even_total_squared)
    return ModalSpectralResponse(
        label=label,
        response_kind=(
            "symmetric_central_secant"
            if symmetric
            else "one_sided_finite_secant"
        ),
        source_mode_indices=source_mode_indices,
        impulse_logical_positions=origins,
        source_mode_amplitudes=amplitudes,
        max_lag=max_lag,
        fft_length=fft_length,
        lag_observation_counts=lag_counts,
        impulse_responses=responses,
        even_residual_impulse_responses=even_responses,
        impulse_response_sha256s=_response_hashes(responses),
        total_valid_response_frobenius=total_frobenius,
        causal_window_response_frobenius=math.sqrt(causal_squared),
        precausal_response_frobenius=math.sqrt(precausal_squared),
        postwindow_response_frobenius=math.sqrt(postwindow_squared),
        even_residual_total_valid_frobenius=even_total_frobenius,
        even_residual_causal_window_frobenius=math.sqrt(
            even_causal_squared
        ),
        relative_even_residual=even_total_frobenius
        / max(total_frobenius, torch.finfo(torch.float64).eps),
        similarity_threshold=similarity_threshold,
        function_evaluation_count=(
            len(source_mode_indices) * len(origins) * (2 if symmetric else 1)
        ),
        **products,
    )


def analyze_modal_spectral_mapping(
    function: Callable[[Tensor], Tensor],
    *,
    baseline_modes: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
    source_mode_indices: Sequence[int] | None = None,
    impulse_logical_positions: Sequence[int] | None = None,
    max_lag: int | None = None,
    fft_length: int | None = None,
    finite_impulse_amplitudes: float | Tensor = 1.0,
    symmetric_amplitude_sets: Mapping[str, Tensor] | None = None,
    similarity_threshold: float = 0.9,
) -> ModalSpectralMapping:
    """Measure reference-bound modal impulse and spectral fingerprints.

    ``symmetric_amplitude_sets`` maps stable labels to positive tensors of
    shape ``[r_src]``.  Each set evaluates a central secant and an even
    residual.  For example, callers can provide local ``0.05 * sigma`` and
    operating-scale ``1.0 * sigma`` tensors in one authenticated artifact.
    With multiple impulse origins, the lag window must be observed at every
    origin.  An omitted ``max_lag`` selects the longest such contiguous window.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(baseline_modes, Tensor):
        raise TypeError("baseline_modes must be a Tensor")
    if (
        baseline_modes.ndim != 3
        or baseline_modes.shape[0] != 1
        or baseline_modes.shape[1] <= 0
        or baseline_modes.shape[2] <= 0
    ):
        raise ValueError("baseline_modes must have shape [1, S, r_src]")
    if (
        not baseline_modes.is_floating_point()
        or not bool(torch.isfinite(baseline_modes).all())
    ):
        raise ValueError("baseline_modes must be finite floating data")
    sequence_length = int(baseline_modes.shape[1])
    source_rank = int(baseline_modes.shape[2])
    positions, mask, position_tuple, mask_tuple = _positions_and_mask(
        logical_positions,
        valid_mask,
        sequence_length=sequence_length,
    )
    valid_positions = tuple(
        int(value) for value in positions[mask].tolist()
    )
    modes = _resolve_modes(
        source_mode_indices,
        source_rank=source_rank,
    )
    origins = _resolve_origins(
        impulse_logical_positions,
        valid_positions=valid_positions,
    )
    maximum_fully_observed_lag = (
        _maximum_fully_observed_lag(
            origins,
            valid_positions=valid_positions,
        )
        if len(origins) > 1
        else None
    )
    if max_lag is None:
        max_lag = (
            maximum_fully_observed_lag
            if maximum_fully_observed_lag is not None
            else max(valid_positions) - min(origins)
        )
    max_lag = _require_nonnegative_int(max_lag, label="max_lag")
    if (
        maximum_fully_observed_lag is not None
        and max_lag > maximum_fully_observed_lag
    ):
        raise ValueError(
            "max_lag must be fully observed from every impulse origin"
        )
    if fft_length is None:
        required = 2 * (max_lag + 1)
        fft_length = 1 << (required - 1).bit_length()
    fft_length = _require_positive_int(fft_length, label="fft_length")
    if fft_length < max_lag + 1:
        raise ValueError("fft_length cannot truncate causal lags")
    similarity_threshold = _finite(
        similarity_threshold,
        label="similarity_threshold",
        minimum=0.0,
        maximum=1.0,
    )
    finite_amplitudes = _resolve_amplitudes(
        finite_impulse_amplitudes,
        source_rank=source_rank,
        source_mode_indices=modes,
        label="finite_impulse_amplitudes",
    )
    if symmetric_amplitude_sets is None:
        symmetric_amplitude_sets = {}
    if not isinstance(symmetric_amplitude_sets, Mapping):
        raise TypeError("symmetric_amplitude_sets must be a mapping")
    symmetric_amplitudes: list[tuple[str, tuple[float, ...]]] = []
    for label in sorted(symmetric_amplitude_sets):
        if not isinstance(label, str) or _LABEL.fullmatch(label) is None:
            raise ValueError("symmetric amplitude label is invalid")
        value = symmetric_amplitude_sets[label]
        if not isinstance(value, Tensor):
            raise TypeError(
                "symmetric amplitude sets must contain Tensor values"
            )
        symmetric_amplitudes.append(
            (
                label,
                _resolve_amplitudes(
                    value,
                    source_rank=source_rank,
                    source_mode_indices=modes,
                    label=f"symmetric_amplitude_sets[{label!r}]",
                ),
            )
        )
    baseline = baseline_modes.detach().clone()
    baseline_output = _function_output(
        function,
        baseline,
        sequence_length=sequence_length,
        target_rank=None,
    )
    target_rank = int(baseline_output.shape[2])
    finite = _measure_response(
        function,
        baseline=baseline,
        baseline_output=baseline_output,
        positions=positions,
        mask=mask,
        source_mode_indices=modes,
        origins=origins,
        amplitudes=finite_amplitudes,
        max_lag=max_lag,
        fft_length=fft_length,
        label="finite",
        symmetric=False,
        similarity_threshold=similarity_threshold,
    )
    symmetric = tuple(
        _measure_response(
            function,
            baseline=baseline,
            baseline_output=baseline_output,
            positions=positions,
            mask=mask,
            source_mode_indices=modes,
            origins=origins,
            amplitudes=amplitudes,
            max_lag=max_lag,
            fft_length=fft_length,
            label=label,
            symmetric=True,
            similarity_threshold=similarity_threshold,
        )
        for label, amplitudes in symmetric_amplitudes
    )
    scale_count = len(symmetric)
    scale_similarity = torch.empty(
        (scale_count, scale_count, len(modes)),
        dtype=torch.float64,
    )
    for left in range(scale_count):
        for right in range(scale_count):
            scale_similarity[left, right] = (
                symmetric[left].normalized_signatures
                * symmetric[right].normalized_signatures
            ).sum(dim=1).clamp(-1.0, 1.0)
    return ModalSpectralMapping(
        baseline_modes_sha256=_tensor_sha256(baseline),
        baseline_output_sha256=_tensor_sha256(baseline_output),
        logical_positions=position_tuple,
        valid_mask=mask_tuple,
        source_rank=source_rank,
        target_rank=target_rank,
        source_mode_indices=modes,
        impulse_logical_positions=origins,
        max_lag=max_lag,
        fft_length=fft_length,
        finite=finite,
        symmetric_responses=symmetric,
        symmetric_scale_pair_similarity=scale_similarity,
        baseline_function_evaluation_count=1,
        function_evaluation_count=(
            1
            + finite.function_evaluation_count
            + sum(
                response.function_evaluation_count
                for response in symmetric
            )
        ),
    )
