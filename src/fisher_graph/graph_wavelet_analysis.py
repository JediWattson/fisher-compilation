"""Deterministic spectral graph-wavelet analysis and matched controls.

The core object in this module is a real Parseval frame built from a
symmetric positive-semidefinite graph Laplacian.  A smooth scaling kernel and
smooth band-pass kernels are evaluated on the graph spectrum, then normalized
*at every eigenvalue* so their squared responses sum to one.  For filter
``f`` and center node ``n``, the node-centered atom is

``kernel_f(L) e_n``.

Consequently, analysis is just filtering followed by reading each node, and
synthesis applies the same filters and sums.  The normalized spectral
partition makes this analysis/synthesis pair a tight frame:

``sum_f kernel_f(L)^2 = I``.

The module also supplies equal-budget graph-Fourier, native-node, and seeded
random-orthonormal controls.  Reports contain hashes and aggregate metrics,
not raw graph signals.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Literal

import torch
from torch import Tensor


GraphCompressionMethod = Literal[
    "graph_wavelet_tight_frame",
    "graph_fourier",
    "native_nodes",
    "random_orthonormal",
]

__all__ = [
    "DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES",
    "GRAPH_COMPRESSION_METHOD_ORDER",
    "CoefficientBudgetPoint",
    "FitOnlyGraphWaveletOMPSubspace",
    "GraphCompressionMethod",
    "GraphWaveletAnalysisReport",
    "GraphWaveletCoefficients",
    "FrozenWaveletGroupOrder",
    "MatchedGraphSignalCoefficients",
    "SpectralGraphWaveletFrame",
    "analyze_graph_wavelet_compression",
    "build_spectral_graph_wavelet_frame",
    "fit_wavelet_group_order",
    "fit_graph_wavelet_omp_subspace",
    "matched_graph_signal_coefficients",
    "reconstruct_frozen_wavelet_groups",
    "reconstruct_matched_graph_signal",
    "wavelet_group_scores",
]


GRAPH_COMPRESSION_METHOD_ORDER: tuple[GraphCompressionMethod, ...] = (
    "graph_wavelet_tight_frame",
    "graph_fourier",
    "native_nodes",
    "random_orthonormal",
)
DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES = (0.5, 1.0, 2.0, 4.0)

_FRAME_KIND = "fisher_graph.spectral_graph_wavelet_frame"
_REPORT_KIND = "fisher_graph.graph_wavelet_analysis_report"
_OMP_SUBSPACE_KIND = "fisher_graph.fit_only_graph_wavelet_omp_subspace"
_FORMAT_VERSION = 1
_FRAME_DOMAIN = b"fisher-graph:spectral-graph-wavelet-frame:v1\0"
_REPORT_DOMAIN = b"fisher-graph:graph-wavelet-analysis-report:v1\0"
_GROUP_ORDER_DOMAIN = b"fisher-graph:frozen-wavelet-group-order:v1\0"
_OMP_SUBSPACE_DOMAIN = (
    b"fisher-graph:fit-only-graph-wavelet-omp-subspace:v1\0"
)
_TENSOR_DOMAIN = b"fisher-graph:canonical-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        _canonical_json_bytes(tuple(int(width) for width in tensor.shape))
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _float_tensor(
    value: object,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if (
        (ndim is not None and result.ndim != ndim)
        or result.ndim == 0
        or any(int(width) <= 0 for width in result.shape)
        or not bool(torch.isfinite(result).all())
    ):
        expected = f" rank-{ndim}" if ndim is not None else ""
        raise ValueError(f"{label} must be finite, nonempty{expected} data")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and strictly positive")
    return float(value)


def _diffusion_scales(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("diffusion_scales must be a sequence")
    result = tuple(
        _positive_float(value, label=f"diffusion_scales[{index}]")
        for index, value in enumerate(values)
    )
    if (
        not result
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(
            "diffusion_scales must be unique strictly increasing values"
        )
    return result


def _canonicalize_column_signs(value: Tensor) -> Tensor:
    result = value.clone()
    for column in range(result.shape[1]):
        pivot = int(torch.argmax(result[:, column].abs()))
        if float(result[pivot, column]) < 0.0:
            result[:, column] *= -1.0
    return result.contiguous()


def _canonical_subspace_basis(projector: Tensor, rank: int) -> Tensor:
    """Choose a coordinate-anchored basis for a degenerate eigenspace."""

    node_count = int(projector.shape[0])
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(1, node_count)
    columns: list[Tensor] = []
    for coordinate in range(node_count):
        candidate = projector[:, coordinate].clone()
        for column in columns:
            candidate -= torch.dot(column, candidate) * column
        norm = float(torch.linalg.vector_norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(torch.argmax(candidate.abs()))
        if float(candidate[pivot]) < 0.0:
            candidate *= -1.0
        columns.append(candidate)
        if len(columns) == rank:
            break
    if len(columns) != rank:
        raise RuntimeError("could not canonicalize a graph eigenspace")
    return torch.stack(columns, dim=1).contiguous()


def _deterministic_psd_eigh(
    laplacian: Tensor,
    *,
    eigenvalue_tolerance: float,
) -> tuple[Tensor, Tensor, float]:
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    scale = max(1.0, float(torch.linalg.matrix_norm(laplacian, ord=2)))
    absolute_tolerance = eigenvalue_tolerance * scale
    minimum = float(eigenvalues.min())
    if minimum < -absolute_tolerance:
        raise ValueError(
            "laplacian must be positive semidefinite within tolerance"
        )
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    canonical_values = eigenvalues.clone()
    canonical_vectors = torch.empty_like(eigenvectors)
    start = 0
    node_count = int(eigenvalues.numel())
    while start < node_count:
        end = start + 1
        while (
            end < node_count
            and abs(float(eigenvalues[end] - eigenvalues[start]))
            <= absolute_tolerance
        ):
            end += 1
        width = end - start
        if width == 1:
            canonical_vectors[:, start:end] = _canonicalize_column_signs(
                eigenvectors[:, start:end]
            )
        else:
            source = eigenvectors[:, start:end]
            projector = source @ source.T
            canonical_vectors[:, start:end] = _canonical_subspace_basis(
                projector,
                width,
            )
            canonical_values[start:end] = eigenvalues[start:end].mean()
        start = end

    identity = torch.eye(node_count, dtype=torch.float64)
    if not torch.allclose(
        canonical_vectors.T @ canonical_vectors,
        identity,
        atol=5.0e-11,
        rtol=5.0e-11,
    ):
        raise RuntimeError("canonical graph eigenvectors are not orthonormal")
    residual = float(
        torch.linalg.matrix_norm(
            laplacian @ canonical_vectors
            - canonical_vectors * canonical_values.unsqueeze(0),
            ord="fro",
        )
        / max(
            float(torch.linalg.matrix_norm(laplacian, ord="fro")),
            torch.finfo(torch.float64).tiny,
        )
    )
    return (
        canonical_values.contiguous(),
        canonical_vectors.contiguous(),
        residual,
    )


def _normalized_spectral_kernels(
    eigenvalues: Tensor,
    *,
    diffusion_scales: tuple[float, ...],
) -> Tensor:
    """Build heat-diffusion windows and normalize a Parseval partition."""

    maximum = float(eigenvalues.max())
    if maximum <= torch.finfo(torch.float64).eps:
        scaling = torch.ones_like(eigenvalues)
        bands = torch.zeros(
            (len(diffusion_scales), eigenvalues.numel()),
            dtype=torch.float64,
        )
    else:
        normalized = eigenvalues / maximum
        scales = torch.tensor(diffusion_scales, dtype=torch.float64)
        heat = torch.exp(
            -scales.unsqueeze(1) * normalized.unsqueeze(0)
        )
        scaling = heat[-1]
        differences = tuple(
            heat[index] - heat[index + 1]
            for index in range(len(diffusion_scales) - 2, -1, -1)
        )
        highpass = 1.0 - heat[0]
        bands = torch.stack(
            (*differences, highpass),
            dim=0,
        )
    raw = torch.cat((scaling.unsqueeze(0), bands), dim=0)
    denominator = raw.square().sum(dim=0).sqrt()
    if bool((denominator <= torch.finfo(torch.float64).tiny).any()):
        raise RuntimeError("spectral graph-wavelet kernels left a frequency uncovered")
    return (raw / denominator.unsqueeze(0)).contiguous()


def _filter_matrices(
    eigenvectors: Tensor,
    kernels: Tensor,
) -> Tensor:
    filters = torch.einsum(
        "ik,fk,jk->fij",
        eigenvectors,
        kernels,
        eigenvectors,
    )
    return ((filters + filters.transpose(1, 2)) / 2.0).contiguous()


@dataclass(frozen=True, slots=True, eq=False)
class GraphWaveletCoefficients:
    """Analysis coefficients ordered ``[filter, node, *signal_tail]``."""

    frame_artifact_sha256: str
    values: Tensor

    def __post_init__(self) -> None:
        _require_sha256(
            self.frame_artifact_sha256,
            label="frame_artifact_sha256",
        )
        values = _float_tensor(self.values, label="wavelet coefficients")
        if values.ndim < 2:
            raise ValueError(
                "wavelet coefficients must have filter and node axes"
            )
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True, eq=False)
class SpectralGraphWaveletFrame:
    """Authenticated scaling-plus-bandpass Parseval graph frame."""

    laplacian: Tensor
    eigenvalues: Tensor
    eigenvectors: Tensor
    spectral_kernels: Tensor
    filter_matrices: Tensor
    filter_names: tuple[str, ...]
    diffusion_scales: tuple[float, ...]
    eigenvalue_tolerance: float
    eigensystem_relative_residual: float
    tight_partition_maximum_error: float
    tight_operator_maximum_error: float
    artifact_sha256: str = ""
    artifact_kind: str = _FRAME_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        laplacian = _float_tensor(self.laplacian, label="laplacian", ndim=2)
        if laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError("laplacian must be square")
        node_count = int(laplacian.shape[0])
        eigenvalues = _float_tensor(
            self.eigenvalues,
            label="eigenvalues",
            ndim=1,
        )
        eigenvectors = _float_tensor(
            self.eigenvectors,
            label="eigenvectors",
            ndim=2,
        )
        kernels = _float_tensor(
            self.spectral_kernels,
            label="spectral_kernels",
            ndim=2,
        )
        filters = _float_tensor(
            self.filter_matrices,
            label="filter_matrices",
            ndim=3,
        )
        diffusion_scales = _diffusion_scales(self.diffusion_scales)
        bandpass_count = len(diffusion_scales)
        filter_count = bandpass_count + 1
        names = tuple(self.filter_names)
        if (
            eigenvalues.shape != (node_count,)
            or eigenvectors.shape != (node_count, node_count)
            or kernels.shape != (filter_count, node_count)
            or filters.shape != (filter_count, node_count, node_count)
            or names
            != (
                "scaling",
                *(
                    f"bandpass_{index:02d}"
                    for index in range(bandpass_count)
                ),
            )
        ):
            raise ValueError("graph-wavelet frame tensor geometry differs")
        if not torch.allclose(
            laplacian,
            laplacian.T,
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise ValueError("laplacian must be symmetric")
        identity = torch.eye(node_count, dtype=torch.float64)
        if not torch.allclose(
            eigenvectors.T @ eigenvectors,
            identity,
            atol=5.0e-10,
            rtol=5.0e-10,
        ):
            raise ValueError("graph eigenvectors must be orthonormal")
        if bool((eigenvalues < 0.0).any()) or (
            node_count > 1
            and bool((eigenvalues[1:] < eigenvalues[:-1]).any())
        ):
            raise ValueError("graph eigenvalues must be sorted and nonnegative")
        eigenvalue_tolerance = _positive_float(
            self.eigenvalue_tolerance,
            label="eigenvalue_tolerance",
        )
        laplacian_scale = max(
            1.0,
            float(torch.linalg.matrix_norm(laplacian, ord=2)),
        )
        eigen_left = laplacian @ eigenvectors
        eigen_right = eigenvectors * eigenvalues.unsqueeze(0)
        if not torch.allclose(
            eigen_left,
            eigen_right,
            atol=eigenvalue_tolerance * laplacian_scale,
            rtol=eigenvalue_tolerance,
        ):
            raise ValueError("laplacian and graph eigensystem differ")
        actual_eigensystem_residual = float(
            torch.linalg.matrix_norm(
                eigen_left - eigen_right,
                ord="fro",
            )
            / max(
                float(torch.linalg.matrix_norm(laplacian, ord="fro")),
                torch.finfo(torch.float64).tiny,
            )
        )
        declared_eigensystem_residual = float(
            self.eigensystem_relative_residual
        )
        residual_receipt_tolerance = (
            128.0
            * torch.finfo(torch.float64).eps
            * max(
                1.0,
                abs(actual_eigensystem_residual),
                abs(declared_eigensystem_residual),
            )
        )
        if (
            not math.isfinite(declared_eigensystem_residual)
            or declared_eigensystem_residual < 0.0
            or abs(
                declared_eigensystem_residual
                - actual_eigensystem_residual
            )
            > residual_receipt_tolerance
        ):
            raise ValueError("declared eigensystem residual differs")
        expected_kernels = _normalized_spectral_kernels(
            eigenvalues,
            diffusion_scales=diffusion_scales,
        )
        kernel_tolerance = 128.0 * torch.finfo(torch.float64).eps
        if not torch.allclose(
            kernels,
            expected_kernels,
            atol=kernel_tolerance,
            rtol=kernel_tolerance,
        ):
            raise ValueError(
                "spectral kernels differ from deterministic diffusion kernels"
            )
        partition_error = float(
            torch.max(torch.abs(kernels.square().sum(dim=0) - 1.0))
        )
        operator = torch.einsum("fij,fjk->ik", filters, filters)
        operator_error = float(torch.max(torch.abs(operator - identity)))
        rebuilt_filters = _filter_matrices(eigenvectors, kernels)
        if (
            partition_error > 5.0e-10
            or operator_error > 5.0e-9
            or not torch.allclose(
                filters,
                rebuilt_filters,
                atol=5.0e-10,
                rtol=5.0e-10,
            )
        ):
            raise ValueError("graph-wavelet filters are not a Parseval frame")
        declared_partition = float(self.tight_partition_maximum_error)
        declared_operator = float(self.tight_operator_maximum_error)
        if (
            not math.isfinite(declared_partition)
            or not math.isfinite(declared_operator)
            or abs(declared_partition - partition_error) > 1.0e-15
            or abs(declared_operator - operator_error) > 1.0e-15
        ):
            raise ValueError("declared tight-frame residuals differ")
        if (
            self.artifact_kind != _FRAME_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("graph-wavelet frame provenance differs")

        object.__setattr__(self, "laplacian", laplacian)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "spectral_kernels", kernels)
        object.__setattr__(self, "filter_matrices", filters)
        object.__setattr__(self, "filter_names", names)
        object.__setattr__(
            self,
            "diffusion_scales",
            diffusion_scales,
        )
        object.__setattr__(
            self,
            "eigenvalue_tolerance",
            eigenvalue_tolerance,
        )
        object.__setattr__(
            self,
            "eigensystem_relative_residual",
            actual_eigensystem_residual,
        )
        object.__setattr__(
            self,
            "tight_partition_maximum_error",
            partition_error,
        )
        object.__setattr__(
            self,
            "tight_operator_maximum_error",
            operator_error,
        )
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("graph-wavelet frame artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def node_count(self) -> int:
        return int(self.laplacian.shape[0])

    @property
    def filter_count(self) -> int:
        return int(self.spectral_kernels.shape[0])

    @property
    def bandpass_count(self) -> int:
        return len(self.diffusion_scales)

    @property
    def frame_coefficient_count(self) -> int:
        return self.filter_count * self.node_count

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "node_count": self.node_count,
            "filter_names": self.filter_names,
            "diffusion_scales": self.diffusion_scales,
            "bandpass_count": self.bandpass_count,
            "eigenvalue_tolerance": self.eigenvalue_tolerance,
            "eigensystem_relative_residual": (
                self.eigensystem_relative_residual
            ),
            "tight_partition_maximum_error": (
                self.tight_partition_maximum_error
            ),
            "tight_operator_maximum_error": self.tight_operator_maximum_error,
            "tensor_sha256s": {
                "laplacian": _tensor_sha256(self.laplacian),
                "eigenvalues": _tensor_sha256(self.eigenvalues),
                "eigenvectors": _tensor_sha256(self.eigenvectors),
                "spectral_kernels": _tensor_sha256(self.spectral_kernels),
                "filter_matrices": _tensor_sha256(self.filter_matrices),
            },
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(_FRAME_DOMAIN, self._hash_payload())

    def validate_integrity(self) -> None:
        for name in (
            "laplacian",
            "eigenvalues",
            "eigenvectors",
            "spectral_kernels",
            "filter_matrices",
        ):
            value = getattr(self, name)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("graph-wavelet frame artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "frame_coefficient_count": self.frame_coefficient_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self.metadata(),
            "laplacian": self.laplacian.clone(),
            "eigenvalues": self.eigenvalues.clone(),
            "eigenvectors": self.eigenvectors.clone(),
            "spectral_kernels": self.spectral_kernels.clone(),
            "filter_matrices": self.filter_matrices.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, object],
    ) -> "SpectralGraphWaveletFrame":
        if not isinstance(value, Mapping):
            raise TypeError("graph-wavelet frame state must be a mapping")
        expected = {
            "artifact_kind",
            "format_version",
            "node_count",
            "filter_names",
            "diffusion_scales",
            "bandpass_count",
            "eigenvalue_tolerance",
            "eigensystem_relative_residual",
            "tight_partition_maximum_error",
            "tight_operator_maximum_error",
            "tensor_sha256s",
            "frame_coefficient_count",
            "artifact_sha256",
            "laplacian",
            "eigenvalues",
            "eigenvectors",
            "spectral_kernels",
            "filter_matrices",
        }
        if set(value) != expected:
            raise ValueError("graph-wavelet frame state fields differ")
        result = cls(
            laplacian=value["laplacian"],  # type: ignore[arg-type]
            eigenvalues=value["eigenvalues"],  # type: ignore[arg-type]
            eigenvectors=value["eigenvectors"],  # type: ignore[arg-type]
            spectral_kernels=value["spectral_kernels"],  # type: ignore[arg-type]
            filter_matrices=value["filter_matrices"],  # type: ignore[arg-type]
            filter_names=tuple(value["filter_names"]),  # type: ignore[arg-type]
            diffusion_scales=tuple(
                value["diffusion_scales"]  # type: ignore[arg-type]
            ),
            eigenvalue_tolerance=value[
                "eigenvalue_tolerance"
            ],  # type: ignore[arg-type]
            eigensystem_relative_residual=value[
                "eigensystem_relative_residual"
            ],  # type: ignore[arg-type]
            tight_partition_maximum_error=value[
                "tight_partition_maximum_error"
            ],  # type: ignore[arg-type]
            tight_operator_maximum_error=value[
                "tight_operator_maximum_error"
            ],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=value["artifact_kind"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )
        if (
            value["node_count"] != result.node_count
            or value["bandpass_count"] != result.bandpass_count
            or value["frame_coefficient_count"]
            != result.frame_coefficient_count
            or value["tensor_sha256s"]
            != result.metadata()["tensor_sha256s"]
        ):
            raise ValueError("graph-wavelet frame state receipt differs")
        return result

    def atom(self, filter_index: int, center_node: int) -> Tensor:
        self.validate_integrity()
        if (
            type(filter_index) is not int
            or not 0 <= filter_index < self.filter_count
        ):
            raise ValueError("filter_index is out of range")
        if (
            type(center_node) is not int
            or not 0 <= center_node < self.node_count
        ):
            raise ValueError("center_node is out of range")
        return self.filter_matrices[filter_index, :, center_node].clone()

    def analyze(self, signal: Tensor) -> GraphWaveletCoefficients:
        self.validate_integrity()
        canonical = _float_tensor(signal, label="graph signal")
        if canonical.shape[0] != self.node_count:
            raise ValueError("graph signal node axis differs from frame")
        tail = tuple(int(width) for width in canonical.shape[1:])
        flat = canonical.reshape(self.node_count, -1)
        coefficients = torch.einsum(
            "fij,jc->fic",
            self.filter_matrices,
            flat,
        ).reshape((self.filter_count, self.node_count, *tail))
        return GraphWaveletCoefficients(
            frame_artifact_sha256=self.artifact_sha256,
            values=coefficients,
        )

    def synthesize(self, coefficients: GraphWaveletCoefficients) -> Tensor:
        self.validate_integrity()
        if not isinstance(coefficients, GraphWaveletCoefficients):
            raise TypeError("coefficients must be GraphWaveletCoefficients")
        if coefficients.frame_artifact_sha256 != self.artifact_sha256:
            raise ValueError("wavelet coefficients belong to another frame")
        values = coefficients.values
        if values.shape[:2] != (self.filter_count, self.node_count):
            raise ValueError("wavelet coefficient geometry differs from frame")
        tail = tuple(int(width) for width in values.shape[2:])
        flat = values.reshape(self.filter_count, self.node_count, -1)
        signal = torch.einsum(
            "fij,fjc->ic",
            self.filter_matrices,
            flat,
        )
        return signal.reshape((self.node_count, *tail)).contiguous()

    def reconstruct(self, signal: Tensor) -> Tensor:
        return self.synthesize(self.analyze(signal))

    def localization_metrics(self) -> tuple[dict[str, object], ...]:
        """Return spatial and spectral localization for every centered atom."""

        self.validate_integrity()
        distances = _all_pairs_hop_distances(
            self.laplacian,
            tolerance=self.eigenvalue_tolerance,
        )
        rows: list[dict[str, object]] = []
        tiny = torch.finfo(torch.float64).tiny
        for filter_index, filter_name in enumerate(self.filter_names):
            kernel = self.spectral_kernels[filter_index]
            for center in range(self.node_count):
                atom = self.filter_matrices[filter_index, :, center]
                energy = atom.square()
                total = float(energy.sum())
                if total <= tiny:
                    probability = torch.zeros_like(energy)
                    center_fraction = 0.0
                    effective_support = 0.0
                    reachable_fraction = 0.0
                    mean_hop = 0.0
                    rms_hop = 0.0
                    spectral_center = 0.0
                    spectral_scale = 0.0
                    support_90 = 0
                    support_95 = 0
                    graph_total_variation = 0.0
                else:
                    probability = energy / total
                    center_fraction = float(probability[center])
                    effective_support = 1.0 / float(
                        probability.square().sum()
                    )
                    finite = torch.isfinite(distances[center])
                    reachable_fraction = float(probability[finite].sum())
                    if reachable_fraction > 0.0:
                        conditional = probability[finite] / reachable_fraction
                        hops = distances[center, finite]
                        mean_hop = float(torch.sum(conditional * hops))
                        rms_hop = math.sqrt(
                            max(
                                0.0,
                                float(torch.sum(conditional * hops.square())),
                            )
                        )
                    else:
                        mean_hop = 0.0
                        rms_hop = 0.0
                    spectral_energy = (
                        kernel * self.eigenvectors[center]
                    ).square()
                    spectral_total = float(spectral_energy.sum())
                    if spectral_total <= tiny:
                        spectral_center = 0.0
                        spectral_scale = 0.0
                    else:
                        spectral_probability = (
                            spectral_energy / spectral_total
                        )
                        spectral_center = float(
                            torch.sum(
                                spectral_probability * self.eigenvalues
                            )
                        )
                        spectral_scale = math.sqrt(
                            max(
                                0.0,
                                float(
                                    torch.sum(
                                        spectral_probability
                                        * (
                                            self.eigenvalues
                                            - spectral_center
                                        )
                                        .square()
                                    )
                                ),
                            )
                        )
                    ordered_energy = torch.sort(
                        probability,
                        descending=True,
                    ).values
                    cumulative_energy = torch.cumsum(ordered_energy, dim=0)
                    support_90 = min(
                        self.node_count,
                        int(
                            torch.searchsorted(
                                cumulative_energy,
                                torch.tensor(0.9, dtype=torch.float64),
                            )
                        )
                        + 1,
                    )
                    support_95 = min(
                        self.node_count,
                        int(
                            torch.searchsorted(
                                cumulative_energy,
                                torch.tensor(0.95, dtype=torch.float64),
                            )
                        )
                        + 1,
                    )
                    graph_total_variation = float(
                        atom @ (self.laplacian @ atom)
                    ) / total
                rows.append(
                    {
                        "filter_index": filter_index,
                        "filter_name": filter_name,
                        "center_node": center,
                        "atom_l2_norm": math.sqrt(max(0.0, total)),
                        "center_energy_fraction": center_fraction,
                        "effective_node_support": effective_support,
                        "energy_support_90_node_count": support_90,
                        "energy_support_95_node_count": support_95,
                        "normalized_graph_total_variation": (
                            graph_total_variation
                        ),
                        "reachable_energy_fraction": reachable_fraction,
                        "unreachable_energy_fraction": 1.0 - reachable_fraction,
                        "mean_reachable_hop_distance": mean_hop,
                        "rms_reachable_hop_distance": rms_hop,
                        "spectral_center": spectral_center,
                        "spectral_scale": spectral_scale,
                    }
                )
        return tuple(rows)

    def scale_localization_summary(self) -> tuple[dict[str, object], ...]:
        rows = self.localization_metrics()
        result: list[dict[str, object]] = []
        for filter_index, filter_name in enumerate(self.filter_names):
            selected = tuple(
                row for row in rows if row["filter_index"] == filter_index
            )
            result.append(
                {
                    "filter_index": filter_index,
                    "filter_name": filter_name,
                    "center_count": len(selected),
                    "mean_center_energy_fraction": math.fsum(
                        float(row["center_energy_fraction"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_effective_node_support": math.fsum(
                        float(row["effective_node_support"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_energy_support_90_node_count": math.fsum(
                        float(row["energy_support_90_node_count"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_energy_support_95_node_count": math.fsum(
                        float(row["energy_support_95_node_count"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_normalized_graph_total_variation": math.fsum(
                        float(row["normalized_graph_total_variation"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_reachable_hop_distance": math.fsum(
                        float(row["mean_reachable_hop_distance"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_rms_reachable_hop_distance": math.fsum(
                        float(row["rms_reachable_hop_distance"])
                        for row in selected
                    )
                    / len(selected),
                    "mean_spectral_center": math.fsum(
                        float(row["spectral_center"]) for row in selected
                    )
                    / len(selected),
                    "mean_spectral_scale": math.fsum(
                        float(row["spectral_scale"]) for row in selected
                    )
                    / len(selected),
                }
            )
        return tuple(result)


def _all_pairs_hop_distances(
    laplacian: Tensor,
    *,
    tolerance: float,
) -> Tensor:
    node_count = int(laplacian.shape[0])
    scale = max(1.0, float(laplacian.abs().max()))
    edge = laplacian.abs() > tolerance * scale
    edge.fill_diagonal_(False)
    distances = torch.full(
        (node_count, node_count),
        float("inf"),
        dtype=torch.float64,
    )
    for source in range(node_count):
        distances[source, source] = 0.0
        queue: deque[int] = deque((source,))
        while queue:
            current = queue.popleft()
            next_distance = float(distances[source, current]) + 1.0
            neighbors = torch.nonzero(edge[current], as_tuple=False).flatten()
            for neighbor_tensor in neighbors:
                neighbor = int(neighbor_tensor)
                if math.isinf(float(distances[source, neighbor])):
                    distances[source, neighbor] = next_distance
                    queue.append(neighbor)
    return distances


def build_spectral_graph_wavelet_frame(
    laplacian: Tensor,
    *,
    diffusion_scales: Sequence[float] = (
        DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES
    ),
    eigenvalue_tolerance: float = 1.0e-12,
) -> SpectralGraphWaveletFrame:
    """Construct a deterministic smooth Parseval frame from a PSD Laplacian."""

    canonical = _float_tensor(laplacian, label="laplacian", ndim=2)
    if canonical.shape[0] != canonical.shape[1]:
        raise ValueError("laplacian must be square")
    symmetry_tolerance = _positive_float(
        eigenvalue_tolerance,
        label="eigenvalue_tolerance",
    )
    if not torch.allclose(
        canonical,
        canonical.T,
        atol=symmetry_tolerance,
        rtol=symmetry_tolerance,
    ):
        raise ValueError("laplacian must be symmetric")
    canonical = ((canonical + canonical.T) / 2.0).contiguous()
    scales = _diffusion_scales(diffusion_scales)
    eigenvalues, eigenvectors, eigensystem_residual = (
        _deterministic_psd_eigh(
            canonical,
            eigenvalue_tolerance=symmetry_tolerance,
        )
    )
    kernels = _normalized_spectral_kernels(
        eigenvalues,
        diffusion_scales=scales,
    )
    filters = _filter_matrices(eigenvectors, kernels)
    identity = torch.eye(canonical.shape[0], dtype=torch.float64)
    partition_error = float(
        torch.max(torch.abs(kernels.square().sum(dim=0) - 1.0))
    )
    operator_error = float(
        torch.max(
            torch.abs(
                torch.einsum("fij,fjk->ik", filters, filters) - identity
            )
        )
    )
    return SpectralGraphWaveletFrame(
        laplacian=canonical,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        spectral_kernels=kernels,
        filter_matrices=filters,
        filter_names=(
            "scaling",
            *(f"bandpass_{index:02d}" for index in range(len(scales))),
        ),
        diffusion_scales=scales,
        eigenvalue_tolerance=symmetry_tolerance,
        eigensystem_relative_residual=eigensystem_residual,
        tight_partition_maximum_error=partition_error,
        tight_operator_maximum_error=operator_error,
    )


def _random_orthonormal_basis(node_count: int, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        (node_count, node_count),
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(raw)
    return _canonicalize_column_signs(basis).contiguous()


def wavelet_group_scores(
    coefficients: GraphWaveletCoefficients,
) -> Tensor:
    """Score each center-scale group by energy summed over the signal tail."""

    if not isinstance(coefficients, GraphWaveletCoefficients):
        raise TypeError("coefficients must be GraphWaveletCoefficients")
    values = coefficients.values
    energy = values.square()
    if energy.ndim > 2:
        energy = energy.sum(
            dim=tuple(range(2, energy.ndim)),
        )
    return energy.to(device="cpu", dtype=torch.float64).contiguous()


def _ordered_groups(scores: Tensor) -> tuple[tuple[int, int], ...]:
    if scores.ndim != 2:
        raise ValueError("wavelet group scores must have shape [filter, node]")
    filter_count, node_count = (int(width) for width in scores.shape)
    return tuple(
        (flat_index // node_count, flat_index % node_count)
        for flat_index in sorted(
            range(filter_count * node_count),
            key=lambda index: (
                -float(scores.reshape(-1)[index]),
                index,
            ),
        )
    )


@dataclass(frozen=True, slots=True, eq=False)
class FrozenWaveletGroupOrder:
    """Fit-only center-scale order reusable without reading heldout signals."""

    frame_artifact_sha256: str
    fit_signal_sha256s: tuple[str, ...]
    group_scores: Tensor
    ordered_groups: tuple[tuple[int, int], ...]
    selection_semantics: str = (
        "fit_only_sum_of_center_scale_coefficient_energy_over_all_"
        "signal_tail_dimensions"
    )
    heldout_signal_used_for_order: bool = False
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(
            self.frame_artifact_sha256,
            label="frame_artifact_sha256",
        )
        signal_hashes = tuple(self.fit_signal_sha256s)
        if not signal_hashes:
            raise ValueError("fit_signal_sha256s cannot be empty")
        for value in signal_hashes:
            _require_sha256(value, label="fit signal SHA-256")
        scores = _float_tensor(
            self.group_scores,
            label="group_scores",
            ndim=2,
        )
        if bool((scores < 0.0).any()):
            raise ValueError("group_scores must be nonnegative")
        order = tuple(tuple(row) for row in self.ordered_groups)
        expected_order = _ordered_groups(scores)
        if order != expected_order:
            raise ValueError("ordered_groups differ from deterministic scores")
        if (
            self.selection_semantics
            != (
                "fit_only_sum_of_center_scale_coefficient_energy_over_all_"
                "signal_tail_dimensions"
            )
            or self.heldout_signal_used_for_order is not False
        ):
            raise ValueError("frozen wavelet group provenance differs")
        object.__setattr__(self, "fit_signal_sha256s", signal_hashes)
        object.__setattr__(self, "group_scores", scores)
        object.__setattr__(self, "ordered_groups", order)
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("frozen wavelet group hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def filter_count(self) -> int:
        return int(self.group_scores.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.group_scores.shape[1])

    @property
    def group_count(self) -> int:
        return int(self.group_scores.numel())

    def _payload(self) -> dict[str, object]:
        return {
            "frame_artifact_sha256": self.frame_artifact_sha256,
            "fit_signal_sha256s": self.fit_signal_sha256s,
            "fit_signal_count": len(self.fit_signal_sha256s),
            "group_score_sha256": _tensor_sha256(self.group_scores),
            "group_score_shape": tuple(int(x) for x in self.group_scores.shape),
            "ordered_groups": self.ordered_groups,
            "selection_semantics": self.selection_semantics,
            "heldout_signal_used_for_order": (
                self.heldout_signal_used_for_order
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(_GROUP_ORDER_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if (
            _ordered_groups(self.group_scores) != self.ordered_groups
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise ValueError("frozen wavelet group order drifted")

    def mask(self, budget: int) -> Tensor:
        """Return a whole-group boolean mask for the first fit-ranked groups."""

        self.validate_integrity()
        retained = _nonnegative_int(budget, label="budget")
        if retained > self.group_count:
            raise ValueError("group budget exceeds the frozen group count")
        mask = torch.zeros(
            (self.filter_count, self.node_count),
            dtype=torch.bool,
        )
        for filter_index, node_index in self.ordered_groups[:retained]:
            mask[filter_index, node_index] = True
        return mask

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

def fit_wavelet_group_order(
    frame: SpectralGraphWaveletFrame,
    fit_signals: Sequence[Tensor],
) -> FrozenWaveletGroupOrder:
    """Freeze one simultaneous group-energy order from declared fit signals."""

    if not isinstance(frame, SpectralGraphWaveletFrame):
        raise TypeError("frame must be a SpectralGraphWaveletFrame")
    frame.validate_integrity()
    if (
        isinstance(fit_signals, (str, bytes))
        or not isinstance(fit_signals, Sequence)
        or not fit_signals
    ):
        raise ValueError("fit_signals must be a nonempty sequence")
    aggregate = torch.zeros(
        (frame.filter_count, frame.node_count),
        dtype=torch.float64,
    )
    hashes: list[str] = []
    for signal in fit_signals:
        canonical = _float_tensor(signal, label="fit graph signal")
        if canonical.shape[0] != frame.node_count:
            raise ValueError("fit graph signal node axis differs from frame")
        aggregate += wavelet_group_scores(frame.analyze(canonical))
        hashes.append(_tensor_sha256(canonical))
    return FrozenWaveletGroupOrder(
        frame_artifact_sha256=frame.artifact_sha256,
        fit_signal_sha256s=tuple(hashes),
        group_scores=aggregate,
        ordered_groups=_ordered_groups(aggregate),
    )


def reconstruct_frozen_wavelet_groups(
    frame: SpectralGraphWaveletFrame,
    coefficients: GraphWaveletCoefficients,
    frozen_order: FrozenWaveletGroupOrder,
    budget: int,
) -> Tensor:
    """Apply a fit-only group mask and synthesize an arbitrary heldout signal."""

    if not isinstance(frame, SpectralGraphWaveletFrame):
        raise TypeError("frame must be a SpectralGraphWaveletFrame")
    if not isinstance(coefficients, GraphWaveletCoefficients):
        raise TypeError("coefficients must be GraphWaveletCoefficients")
    if not isinstance(frozen_order, FrozenWaveletGroupOrder):
        raise TypeError("frozen_order must be FrozenWaveletGroupOrder")
    frame.validate_integrity()
    frozen_order.validate_integrity()
    if (
        coefficients.frame_artifact_sha256 != frame.artifact_sha256
        or frozen_order.frame_artifact_sha256 != frame.artifact_sha256
        or frozen_order.group_scores.shape
        != (frame.filter_count, frame.node_count)
    ):
        raise ValueError("wavelet group artifacts belong to another frame")
    mask = frozen_order.mask(budget)
    expanded = mask
    for _ in coefficients.values.shape[2:]:
        expanded = expanded.unsqueeze(-1)
    retained = torch.where(
        expanded,
        coefficients.values,
        torch.zeros_like(coefficients.values),
    )
    return frame.synthesize(
        GraphWaveletCoefficients(
            frame_artifact_sha256=frame.artifact_sha256,
            values=retained,
        )
    )


def _coordinate_locality(value: Tensor) -> dict[str, object]:
    vector = _float_tensor(value, label="locality vector", ndim=1)
    energy = vector.square()
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).tiny:
        return {
            "effective_node_support": 0.0,
            "energy_support_90_node_count": 0,
            "energy_support_95_node_count": 0,
        }
    probability = energy / total
    ordered = torch.sort(probability, descending=True).values
    cumulative = torch.cumsum(ordered, dim=0)

    def support(fraction: float) -> int:
        return min(
            int(vector.numel()),
            int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(fraction, dtype=torch.float64),
                )
            )
            + 1,
        )

    return {
        "effective_node_support": 1.0 / float(probability.square().sum()),
        "energy_support_90_node_count": support(0.9),
        "energy_support_95_node_count": support(0.95),
    }


def _canonicalize_vector_sign(value: Tensor) -> Tensor:
    result = value.clone()
    pivot = int(torch.argmax(result.abs()))
    if float(result[pivot]) < 0.0:
        result *= -1.0
    return result.contiguous()


def _frame_atom_dictionary(
    frame: SpectralGraphWaveletFrame,
) -> tuple[Tensor, Tensor]:
    """Return filter-major raw atoms ``[node, filter * center]`` and norms."""

    raw = (
        frame.filter_matrices.permute(1, 0, 2)
        .reshape(frame.node_count, frame.frame_coefficient_count)
        .contiguous()
    )
    norms = torch.linalg.vector_norm(raw, dim=0)
    return raw, norms.contiguous()


@dataclass(frozen=True, slots=True, eq=False)
class FitOnlyGraphWaveletOMPSubspace:
    """Nested localized subspace selected from fit signals by simultaneous OMP.

    The selected atom at each step has the largest normalized *raw-atom*
    correlation with the current multi-output residual.  Only after selection
    is it orthogonalized against the existing basis; candidates with
    insufficient QR novelty are skipped.  The residual is then recomputed by
    an orthogonal least-squares projection onto the full selected span.  Exact
    score ties keep the smaller filter-major flat atom index.
    """

    frame_artifact_sha256: str
    node_count: int
    filter_names: tuple[str, ...]
    fit_signal_sha256s: tuple[str, ...]
    dependency_tolerance: float
    selected_flat_atom_indices: tuple[int, ...]
    selected_raw_atom_norms: Tensor
    selected_qr_novelty: Tensor
    raw_selected_dictionary_condition_by_rank: Tensor
    orthonormal_basis: Tensor
    fit_relative_residual_by_rank: Tensor
    raw_selected_atom_locality: tuple[Mapping[str, object], ...]
    orthonormal_basis_locality: tuple[Mapping[str, object], ...]
    selection_semantics: str = (
        "fit_only_simultaneous_group_omp_normalized_raw_atom_frobenius_"
        "correlation_then_qr_novelty_deterministic_flat_index_ties"
    )
    heldout_signal_used_for_fit: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _OMP_SUBSPACE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.frame_artifact_sha256,
            label="frame_artifact_sha256",
        )
        node_count = _positive_int(self.node_count, label="node_count")
        filter_names = tuple(self.filter_names)
        if (
            not filter_names
            or any(
                not isinstance(name, str) or not name
                for name in filter_names
            )
        ):
            raise ValueError("filter_names must be nonempty strings")
        fit_hashes = tuple(self.fit_signal_sha256s)
        if not fit_hashes:
            raise ValueError("fit_signal_sha256s cannot be empty")
        for value in fit_hashes:
            _require_sha256(value, label="fit signal SHA-256")
        tolerance = _positive_float(
            self.dependency_tolerance,
            label="dependency_tolerance",
        )
        if tolerance >= 1.0:
            raise ValueError("dependency_tolerance must be less than one")
        selected = tuple(self.selected_flat_atom_indices)
        rank = len(selected)
        dictionary_width = node_count * len(filter_names)
        if (
            rank <= 0
            or rank > node_count
            or any(
                type(index) is not int
                or not 0 <= index < dictionary_width
                for index in selected
            )
            or len(set(selected)) != rank
        ):
            raise ValueError("selected flat atom indices are invalid")
        raw_norms = _float_tensor(
            self.selected_raw_atom_norms,
            label="selected_raw_atom_norms",
            ndim=1,
        )
        novelty = _float_tensor(
            self.selected_qr_novelty,
            label="selected_qr_novelty",
            ndim=1,
        )
        conditions = _float_tensor(
            self.raw_selected_dictionary_condition_by_rank,
            label="raw_selected_dictionary_condition_by_rank",
            ndim=1,
        )
        basis = _float_tensor(
            self.orthonormal_basis,
            label="orthonormal_basis",
            ndim=2,
        )
        residual = _float_tensor(
            self.fit_relative_residual_by_rank,
            label="fit_relative_residual_by_rank",
            ndim=1,
        )
        if (
            raw_norms.shape != (rank,)
            or novelty.shape != (rank,)
            or conditions.shape != (rank,)
            or basis.shape != (node_count, rank)
            or residual.shape != (rank + 1,)
            or bool((raw_norms <= 0.0).any())
            or bool((novelty <= tolerance).any())
            or bool((novelty > 1.0 + 1.0e-10).any())
            or bool((conditions < 1.0 - 1.0e-10).any())
            or bool((residual < 0.0).any())
            or bool((residual[1:] > residual[:-1] + 1.0e-12).any())
            or abs(float(residual[0]) - 1.0) > 1.0e-12
        ):
            raise ValueError("OMP subspace tensor geometry or residuals differ")
        if not torch.allclose(
            basis.T @ basis,
            torch.eye(rank, dtype=torch.float64),
            atol=5.0e-10,
            rtol=5.0e-10,
        ):
            raise ValueError("OMP subspace basis must be orthonormal")
        for column in range(rank):
            pivot = int(torch.argmax(basis[:, column].abs()))
            if float(basis[pivot, column]) < 0.0:
                raise ValueError("OMP subspace basis signs are not canonical")
        raw_locality = tuple(
            dict(row) for row in self.raw_selected_atom_locality
        )
        q_locality = tuple(
            dict(row) for row in self.orthonormal_basis_locality
        )
        if (
            len(raw_locality) != rank
            or len(q_locality) != rank
            or tuple(row.get("selection_rank") for row in raw_locality)
            != tuple(range(1, rank + 1))
            or tuple(row.get("selection_rank") for row in q_locality)
            != tuple(range(1, rank + 1))
        ):
            raise ValueError("OMP locality rows differ from selected rank")
        expected_metadata = tuple(
            (
                index // node_count,
                index % node_count,
            )
            for index in selected
        )
        for ordinal, (filter_index, center_node) in enumerate(
            expected_metadata
        ):
            raw_row = raw_locality[ordinal]
            if (
                raw_row.get("flat_atom_index") != selected[ordinal]
                or raw_row.get("filter_index") != filter_index
                or raw_row.get("filter_name") != filter_names[filter_index]
                or raw_row.get("center_node") != center_node
                or abs(
                    float(raw_row.get("raw_atom_norm", -1.0))
                    - float(raw_norms[ordinal])
                )
                > 1.0e-12
                or abs(
                    float(raw_row.get("qr_novelty", -1.0))
                    - float(novelty[ordinal])
                )
                > 1.0e-12
                or abs(
                    float(
                        raw_row.get(
                            "raw_selected_dictionary_condition",
                            -1.0,
                        )
                    )
                    - float(conditions[ordinal])
                )
                > 1.0e-10 * max(1.0, float(conditions[ordinal]))
            ):
                raise ValueError("raw selected atom metadata differs")
        if (
            self.selection_semantics
            != (
                "fit_only_simultaneous_group_omp_normalized_raw_atom_"
                "frobenius_correlation_then_qr_novelty_deterministic_"
                "flat_index_ties"
            )
            or self.heldout_signal_used_for_fit is not False
            or self.artifact_kind != _OMP_SUBSPACE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("OMP subspace provenance differs")

        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "filter_names", filter_names)
        object.__setattr__(self, "fit_signal_sha256s", fit_hashes)
        object.__setattr__(self, "dependency_tolerance", tolerance)
        object.__setattr__(
            self,
            "selected_flat_atom_indices",
            selected,
        )
        object.__setattr__(
            self,
            "selected_raw_atom_norms",
            raw_norms,
        )
        object.__setattr__(self, "selected_qr_novelty", novelty)
        object.__setattr__(
            self,
            "raw_selected_dictionary_condition_by_rank",
            conditions,
        )
        object.__setattr__(self, "orthonormal_basis", basis)
        object.__setattr__(
            self,
            "fit_relative_residual_by_rank",
            residual,
        )
        object.__setattr__(
            self,
            "raw_selected_atom_locality",
            raw_locality,
        )
        object.__setattr__(
            self,
            "orthonormal_basis_locality",
            q_locality,
        )
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("OMP subspace artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def max_rank(self) -> int:
        return int(self.orthonormal_basis.shape[1])

    @property
    def filter_count(self) -> int:
        return len(self.filter_names)

    @property
    def selected_atom_metadata(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(row) for row in self.raw_selected_atom_locality)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "frame_artifact_sha256": self.frame_artifact_sha256,
            "node_count": self.node_count,
            "filter_names": self.filter_names,
            "fit_signal_sha256s": self.fit_signal_sha256s,
            "fit_signal_count": len(self.fit_signal_sha256s),
            "dependency_tolerance": self.dependency_tolerance,
            "max_rank": self.max_rank,
            "selected_flat_atom_indices": self.selected_flat_atom_indices,
            "selected_raw_atom_norms_sha256": _tensor_sha256(
                self.selected_raw_atom_norms
            ),
            "selected_qr_novelty_sha256": _tensor_sha256(
                self.selected_qr_novelty
            ),
            "raw_selected_dictionary_condition_by_rank_sha256": (
                _tensor_sha256(
                    self.raw_selected_dictionary_condition_by_rank
                )
            ),
            "orthonormal_basis_sha256": _tensor_sha256(
                self.orthonormal_basis
            ),
            "fit_relative_residual_by_rank_sha256": _tensor_sha256(
                self.fit_relative_residual_by_rank
            ),
            "raw_selected_atom_locality": tuple(
                dict(row) for row in self.raw_selected_atom_locality
            ),
            "orthonormal_basis_locality": tuple(
                dict(row) for row in self.orthonormal_basis_locality
            ),
            "selection_semantics": self.selection_semantics,
            "heldout_signal_used_for_fit": self.heldout_signal_used_for_fit,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(_OMP_SUBSPACE_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if (
            self.selected_raw_atom_norms.dtype != torch.float64
            or self.selected_qr_novelty.dtype != torch.float64
            or self.raw_selected_dictionary_condition_by_rank.dtype
            != torch.float64
            or self.orthonormal_basis.dtype != torch.float64
            or self.fit_relative_residual_by_rank.dtype != torch.float64
            or self.selected_raw_atom_norms.device.type != "cpu"
            or self.selected_qr_novelty.device.type != "cpu"
            or self.raw_selected_dictionary_condition_by_rank.device.type
            != "cpu"
            or self.orthonormal_basis.device.type != "cpu"
            or self.fit_relative_residual_by_rank.device.type != "cpu"
            or not torch.allclose(
                self.orthonormal_basis.T @ self.orthonormal_basis,
                torch.eye(self.max_rank, dtype=torch.float64),
                atol=5.0e-10,
                rtol=5.0e-10,
            )
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise ValueError("OMP subspace artifact drifted")

    def validate_against_frame(
        self,
        frame: SpectralGraphWaveletFrame,
    ) -> None:
        """Verify selected raw atoms and every nested Q direction."""

        self.validate_integrity()
        if not isinstance(frame, SpectralGraphWaveletFrame):
            raise TypeError("frame must be a SpectralGraphWaveletFrame")
        frame.validate_integrity()
        if (
            frame.artifact_sha256 != self.frame_artifact_sha256
            or frame.node_count != self.node_count
            or frame.filter_names != self.filter_names
        ):
            raise ValueError("OMP subspace belongs to another frame")
        raw_atoms, raw_norms = _frame_atom_dictionary(frame)
        selected_norms = raw_norms[
            torch.tensor(
                self.selected_flat_atom_indices,
                dtype=torch.int64,
            )
        ]
        if not torch.allclose(
            selected_norms,
            self.selected_raw_atom_norms,
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise ValueError("selected raw atom norms differ from frame")
        rebuilt: list[Tensor] = []
        for rank, flat_index in enumerate(self.selected_flat_atom_indices):
            candidate = raw_atoms[:, flat_index] / raw_norms[flat_index]
            for column in rebuilt:
                candidate -= torch.dot(column, candidate) * column
            for column in rebuilt:
                candidate -= torch.dot(column, candidate) * column
            norm = float(torch.linalg.vector_norm(candidate))
            if norm <= self.dependency_tolerance:
                raise ValueError("selected frame atom became dependent")
            if abs(norm - float(self.selected_qr_novelty[rank])) > 1.0e-12:
                raise ValueError("selected QR novelty differs from frame atoms")
            rebuilt.append(_canonicalize_vector_sign(candidate / norm))
            if not torch.allclose(
                rebuilt[-1],
                self.orthonormal_basis[:, rank],
                atol=5.0e-10,
                rtol=5.0e-10,
            ):
                raise ValueError("OMP basis direction differs from frame atoms")
            selected_dictionary = torch.stack(
                (
                    *(
                        raw_atoms[:, index]
                        for index in self.selected_flat_atom_indices[
                            : rank + 1
                        ]
                    ),
                ),
                dim=1,
            )
            singular_values = torch.linalg.svdvals(selected_dictionary)
            condition = float(singular_values[0] / singular_values[-1])
            expected_condition = float(
                self.raw_selected_dictionary_condition_by_rank[rank]
            )
            if abs(condition - expected_condition) > 1.0e-10 * max(
                1.0,
                expected_condition,
            ):
                raise ValueError(
                    "raw selected dictionary condition differs from frame"
                )
            filter_index = flat_index // self.node_count
            center_node = flat_index % self.node_count
            expected_raw_locality = {
                "selection_rank": rank + 1,
                "flat_atom_index": flat_index,
                "filter_index": filter_index,
                "filter_name": self.filter_names[filter_index],
                "center_node": center_node,
                "raw_atom_norm": float(raw_norms[flat_index]),
                "qr_novelty": norm,
                "raw_selected_dictionary_condition": condition,
                **_coordinate_locality(raw_atoms[:, flat_index]),
            }
            expected_q_locality = {
                "selection_rank": rank + 1,
                **_coordinate_locality(rebuilt[-1]),
            }
            if (
                _canonical_json_bytes(expected_raw_locality)
                != _canonical_json_bytes(
                    dict(self.raw_selected_atom_locality[rank])
                )
                or _canonical_json_bytes(expected_q_locality)
                != _canonical_json_bytes(
                    dict(self.orthonormal_basis_locality[rank])
                )
            ):
                raise ValueError("OMP locality metrics differ from frame")

    def basis(self, rank: int) -> Tensor:
        """Return the canonical orthonormal prefix for one nested rank."""

        self.validate_integrity()
        if type(rank) is not int or not 0 <= rank <= self.max_rank:
            raise ValueError("rank must lie in [0, max_rank]")
        return self.orthonormal_basis[:, :rank].clone()

    def project(self, signal: Tensor, rank: int) -> Tensor:
        """Project an arbitrary signal without changing the fit-only order."""

        canonical = _float_tensor(signal, label="graph signal")
        if canonical.shape[0] != self.node_count:
            raise ValueError("graph signal node axis differs from OMP subspace")
        basis = self.basis(rank)
        flat = canonical.reshape(self.node_count, -1)
        projected = basis @ (basis.T @ flat)
        return projected.reshape(canonical.shape).contiguous()

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        """Return the frozen basis tensors plus their authenticated receipt."""

        self.validate_integrity()
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "frame_artifact_sha256": self.frame_artifact_sha256,
            "node_count": self.node_count,
            "filter_names": self.filter_names,
            "fit_signal_sha256s": self.fit_signal_sha256s,
            "dependency_tolerance": self.dependency_tolerance,
            "selected_flat_atom_indices": self.selected_flat_atom_indices,
            "selected_raw_atom_norms": (
                self.selected_raw_atom_norms.clone()
            ),
            "selected_qr_novelty": self.selected_qr_novelty.clone(),
            "raw_selected_dictionary_condition_by_rank": (
                self.raw_selected_dictionary_condition_by_rank.clone()
            ),
            "orthonormal_basis": self.orthonormal_basis.clone(),
            "fit_relative_residual_by_rank": (
                self.fit_relative_residual_by_rank.clone()
            ),
            "raw_selected_atom_locality": tuple(
                dict(row) for row in self.raw_selected_atom_locality
            ),
            "orthonormal_basis_locality": tuple(
                dict(row) for row in self.orthonormal_basis_locality
            ),
            "selection_semantics": self.selection_semantics,
            "heldout_signal_used_for_fit": self.heldout_signal_used_for_fit,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, object],
    ) -> "FitOnlyGraphWaveletOMPSubspace":
        if not isinstance(value, Mapping):
            raise TypeError("OMP subspace state must be a mapping")
        expected = {
            "artifact_kind",
            "format_version",
            "frame_artifact_sha256",
            "node_count",
            "filter_names",
            "fit_signal_sha256s",
            "dependency_tolerance",
            "selected_flat_atom_indices",
            "selected_raw_atom_norms",
            "selected_qr_novelty",
            "raw_selected_dictionary_condition_by_rank",
            "orthonormal_basis",
            "fit_relative_residual_by_rank",
            "raw_selected_atom_locality",
            "orthonormal_basis_locality",
            "selection_semantics",
            "heldout_signal_used_for_fit",
            "artifact_sha256",
        }
        if set(value) != expected:
            raise ValueError("OMP subspace state fields differ")
        return cls(
            frame_artifact_sha256=value[
                "frame_artifact_sha256"
            ],  # type: ignore[arg-type]
            node_count=value["node_count"],  # type: ignore[arg-type]
            filter_names=tuple(value["filter_names"]),  # type: ignore[arg-type]
            fit_signal_sha256s=tuple(
                value["fit_signal_sha256s"]  # type: ignore[arg-type]
            ),
            dependency_tolerance=value[
                "dependency_tolerance"
            ],  # type: ignore[arg-type]
            selected_flat_atom_indices=tuple(
                value[
                    "selected_flat_atom_indices"
                ]  # type: ignore[arg-type]
            ),
            selected_raw_atom_norms=value[
                "selected_raw_atom_norms"
            ],  # type: ignore[arg-type]
            selected_qr_novelty=value[
                "selected_qr_novelty"
            ],  # type: ignore[arg-type]
            raw_selected_dictionary_condition_by_rank=value[
                "raw_selected_dictionary_condition_by_rank"
            ],  # type: ignore[arg-type]
            orthonormal_basis=value[
                "orthonormal_basis"
            ],  # type: ignore[arg-type]
            fit_relative_residual_by_rank=value[
                "fit_relative_residual_by_rank"
            ],  # type: ignore[arg-type]
            raw_selected_atom_locality=tuple(
                value[
                    "raw_selected_atom_locality"
                ]  # type: ignore[arg-type]
            ),
            orthonormal_basis_locality=tuple(
                value[
                    "orthonormal_basis_locality"
                ]  # type: ignore[arg-type]
            ),
            selection_semantics=value[
                "selection_semantics"
            ],  # type: ignore[arg-type]
            heldout_signal_used_for_fit=value[
                "heldout_signal_used_for_fit"
            ],  # type: ignore[arg-type]
            artifact_sha256=value[
                "artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_kind=value["artifact_kind"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )


def _fit_signal_matrix(
    frame: SpectralGraphWaveletFrame,
    fit_signals: Sequence[Tensor],
) -> tuple[Tensor, tuple[str, ...]]:
    if (
        isinstance(fit_signals, (str, bytes))
        or not isinstance(fit_signals, Sequence)
        or not fit_signals
    ):
        raise ValueError("fit_signals must be a nonempty sequence")
    columns: list[Tensor] = []
    hashes: list[str] = []
    for signal in fit_signals:
        canonical = _float_tensor(signal, label="fit graph signal")
        if canonical.shape[0] != frame.node_count:
            raise ValueError("fit graph signal node axis differs from frame")
        columns.append(canonical.reshape(frame.node_count, -1))
        hashes.append(_tensor_sha256(canonical))
    matrix = torch.cat(columns, dim=1).contiguous()
    if float(matrix.square().sum()) <= torch.finfo(torch.float64).tiny:
        raise ValueError("fit graph signals must contain nonzero energy")
    return matrix, tuple(hashes)


def fit_graph_wavelet_omp_subspace(
    frame: SpectralGraphWaveletFrame,
    fit_signals: Sequence[Tensor],
    *,
    max_rank: int | None = None,
    dependency_tolerance: float = 1.0e-10,
) -> FitOnlyGraphWaveletOMPSubspace:
    """Fit a deterministic nested simultaneous-OMP localized subspace."""

    if not isinstance(frame, SpectralGraphWaveletFrame):
        raise TypeError("frame must be a SpectralGraphWaveletFrame")
    frame.validate_integrity()
    if max_rank is None:
        selected_max_rank = frame.node_count
    else:
        selected_max_rank = _positive_int(max_rank, label="max_rank")
    if selected_max_rank > frame.node_count:
        raise ValueError("max_rank cannot exceed node_count")
    tolerance = _positive_float(
        dependency_tolerance,
        label="dependency_tolerance",
    )
    if tolerance >= 1.0:
        raise ValueError("dependency_tolerance must be less than one")
    fit_matrix, fit_hashes = _fit_signal_matrix(frame, fit_signals)
    total_fit_energy = float(fit_matrix.square().sum())
    raw_atoms, raw_norms = _frame_atom_dictionary(frame)
    maximum_raw_norm = float(raw_norms.max())
    nonzero_threshold = max(
        torch.finfo(torch.float64).tiny,
        128.0
        * torch.finfo(torch.float64).eps
        * max(1.0, maximum_raw_norm)
        * frame.node_count,
    )
    candidate_indices = tuple(
        index
        for index in range(frame.frame_coefficient_count)
        if float(raw_norms[index]) > nonzero_threshold
    )
    if not candidate_indices:
        raise ValueError("graph-wavelet frame has no nonzero atoms")
    normalized_atoms = raw_atoms.clone()
    candidate_tensor = torch.tensor(candidate_indices, dtype=torch.int64)
    normalized_atoms[:, candidate_tensor] = (
        normalized_atoms[:, candidate_tensor]
        / raw_norms[candidate_tensor].unsqueeze(0)
    )

    selected_indices: list[int] = []
    skipped_indices: set[int] = set()
    q_columns: list[Tensor] = []
    selected_novelties: list[float] = []
    selected_conditions: list[float] = []
    residual = fit_matrix.clone()
    relative_residuals = [1.0]
    for _selection_rank in range(selected_max_rank):
        selected_this_rank = False
        while not selected_this_rank:
            best_index: int | None = None
            best_score = -1.0
            for flat_index in candidate_indices:
                if (
                    flat_index in selected_indices
                    or flat_index in skipped_indices
                ):
                    continue
                raw_direction = normalized_atoms[:, flat_index]
                score = float((raw_direction @ residual).square().sum())
                if score > best_score:
                    best_score = score
                    best_index = flat_index
            if best_index is None:
                break
            candidate = normalized_atoms[:, best_index].clone()
            for column in q_columns:
                candidate -= torch.dot(column, candidate) * column
            for column in q_columns:
                candidate -= torch.dot(column, candidate) * column
            novelty = float(torch.linalg.vector_norm(candidate))
            if novelty <= tolerance:
                skipped_indices.add(best_index)
                continue
            direction = _canonicalize_vector_sign(candidate / novelty)
            selected_indices.append(best_index)
            selected_novelties.append(novelty)
            q_columns.append(direction)
            selected_dictionary = raw_atoms[
                :,
                torch.tensor(selected_indices, dtype=torch.int64),
            ]
            singular_values = torch.linalg.svdvals(selected_dictionary)
            condition = float(
                singular_values[0]
                / max(
                    float(singular_values[-1]),
                    torch.finfo(torch.float64).tiny,
                )
            )
            if not math.isfinite(condition):
                raise RuntimeError(
                    "selected raw graph-wavelet dictionary became singular"
                )
            selected_conditions.append(condition)
            selected_this_rank = True
        if not selected_this_rank:
            break
        basis = torch.stack(q_columns, dim=1)
        residual = fit_matrix - basis @ (basis.T @ fit_matrix)
        relative = math.sqrt(
            max(0.0, float(residual.square().sum()) / total_fit_energy)
        )
        relative_residuals.append(
            min(relative_residuals[-1], relative)
        )
    if len(selected_indices) != selected_max_rank:
        raise RuntimeError(
            "graph-wavelet atoms could not span the requested max_rank"
        )
    basis = torch.stack(q_columns, dim=1).contiguous()
    selected_norms = raw_norms[
        torch.tensor(selected_indices, dtype=torch.int64)
    ].contiguous()
    raw_locality: list[dict[str, object]] = []
    q_locality: list[dict[str, object]] = []
    for ordinal, flat_index in enumerate(selected_indices):
        filter_index = flat_index // frame.node_count
        center_node = flat_index % frame.node_count
        raw_locality.append(
            {
                "selection_rank": ordinal + 1,
                "flat_atom_index": flat_index,
                "filter_index": filter_index,
                "filter_name": frame.filter_names[filter_index],
                "center_node": center_node,
                "raw_atom_norm": float(selected_norms[ordinal]),
                "qr_novelty": selected_novelties[ordinal],
                "raw_selected_dictionary_condition": (
                    selected_conditions[ordinal]
                ),
                **_coordinate_locality(raw_atoms[:, flat_index]),
            }
        )
        q_locality.append(
            {
                "selection_rank": ordinal + 1,
                **_coordinate_locality(basis[:, ordinal]),
            }
        )
    result = FitOnlyGraphWaveletOMPSubspace(
        frame_artifact_sha256=frame.artifact_sha256,
        node_count=frame.node_count,
        filter_names=frame.filter_names,
        fit_signal_sha256s=fit_hashes,
        dependency_tolerance=tolerance,
        selected_flat_atom_indices=tuple(selected_indices),
        selected_raw_atom_norms=selected_norms,
        selected_qr_novelty=torch.tensor(
            selected_novelties,
            dtype=torch.float64,
        ),
        raw_selected_dictionary_condition_by_rank=torch.tensor(
            selected_conditions,
            dtype=torch.float64,
        ),
        orthonormal_basis=basis,
        fit_relative_residual_by_rank=torch.tensor(
            relative_residuals,
            dtype=torch.float64,
        ),
        raw_selected_atom_locality=tuple(raw_locality),
        orthonormal_basis_locality=tuple(q_locality),
    )
    result.validate_against_frame(frame)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class MatchedGraphSignalCoefficients:
    """Transient coefficients for the frame and three equal-budget controls."""

    frame_artifact_sha256: str
    signal_sha256: str
    random_seed: int
    wavelet: Tensor
    graph_fourier: Tensor
    native_nodes: Tensor
    random_orthonormal: Tensor
    random_basis: Tensor

    def __post_init__(self) -> None:
        _require_sha256(
            self.frame_artifact_sha256,
            label="frame_artifact_sha256",
        )
        _require_sha256(self.signal_sha256, label="signal_sha256")
        seed = _nonnegative_int(self.random_seed, label="random_seed")
        wavelet = _float_tensor(self.wavelet, label="wavelet", ndim=2)
        node_count = int(wavelet.shape[1])
        graph_fourier = _float_tensor(
            self.graph_fourier,
            label="graph_fourier",
            ndim=1,
        )
        native = _float_tensor(
            self.native_nodes,
            label="native_nodes",
            ndim=1,
        )
        random = _float_tensor(
            self.random_orthonormal,
            label="random_orthonormal",
            ndim=1,
        )
        random_basis = _float_tensor(
            self.random_basis,
            label="random_basis",
            ndim=2,
        )
        if (
            graph_fourier.shape != (node_count,)
            or native.shape != (node_count,)
            or random.shape != (node_count,)
            or random_basis.shape != (node_count, node_count)
            or not torch.allclose(
                random_basis.T @ random_basis,
                torch.eye(node_count, dtype=torch.float64),
                atol=1.0e-10,
                rtol=1.0e-10,
            )
        ):
            raise ValueError("matched graph coefficient geometry differs")
        object.__setattr__(self, "random_seed", seed)
        object.__setattr__(self, "wavelet", wavelet)
        object.__setattr__(self, "graph_fourier", graph_fourier)
        object.__setattr__(self, "native_nodes", native)
        object.__setattr__(self, "random_orthonormal", random)
        object.__setattr__(self, "random_basis", random_basis)

    @property
    def node_count(self) -> int:
        return int(self.native_nodes.numel())

    def values(self, method: GraphCompressionMethod) -> Tensor:
        if method == "graph_wavelet_tight_frame":
            return self.wavelet
        if method == "graph_fourier":
            return self.graph_fourier
        if method == "native_nodes":
            return self.native_nodes
        if method == "random_orthonormal":
            return self.random_orthonormal
        raise ValueError("unknown graph compression method")


def matched_graph_signal_coefficients(
    frame: SpectralGraphWaveletFrame,
    signal: Tensor,
    *,
    random_seed: int = 0,
) -> MatchedGraphSignalCoefficients:
    """Analyze one scalar graph signal in the frame and matched controls."""

    if not isinstance(frame, SpectralGraphWaveletFrame):
        raise TypeError("frame must be a SpectralGraphWaveletFrame")
    frame.validate_integrity()
    canonical = _float_tensor(signal, label="graph signal", ndim=1)
    if canonical.shape != (frame.node_count,):
        raise ValueError("matched controls require one scalar per graph node")
    seed = _nonnegative_int(random_seed, label="random_seed")
    random_basis = _random_orthonormal_basis(frame.node_count, seed=seed)
    return MatchedGraphSignalCoefficients(
        frame_artifact_sha256=frame.artifact_sha256,
        signal_sha256=_tensor_sha256(canonical),
        random_seed=seed,
        wavelet=frame.analyze(canonical).values,
        graph_fourier=frame.eigenvectors.T @ canonical,
        native_nodes=canonical,
        random_orthonormal=random_basis.T @ canonical,
        random_basis=random_basis,
    )


def _top_budget_mask(values: Tensor, budget: int) -> Tensor:
    flat = values.reshape(-1)
    keep = min(budget, int(flat.numel()))
    mask = torch.zeros(flat.numel(), dtype=torch.bool)
    if keep == 0:
        return mask.reshape(values.shape)
    order = sorted(
        range(int(flat.numel())),
        key=lambda index: (-abs(float(flat[index])), index),
    )
    mask[torch.tensor(order[:keep], dtype=torch.int64)] = True
    return mask.reshape(values.shape)


def reconstruct_matched_graph_signal(
    frame: SpectralGraphWaveletFrame,
    coefficients: MatchedGraphSignalCoefficients,
    method: GraphCompressionMethod,
    budget: int,
) -> Tensor:
    """Keep the largest ``budget`` scalar coefficients and reconstruct."""

    if not isinstance(frame, SpectralGraphWaveletFrame):
        raise TypeError("frame must be a SpectralGraphWaveletFrame")
    if not isinstance(coefficients, MatchedGraphSignalCoefficients):
        raise TypeError(
            "coefficients must be MatchedGraphSignalCoefficients"
        )
    frame.validate_integrity()
    if coefficients.frame_artifact_sha256 != frame.artifact_sha256:
        raise ValueError("matched coefficients belong to another frame")
    retained_budget = _nonnegative_int(budget, label="budget")
    values = coefficients.values(method)
    mask = _top_budget_mask(values, retained_budget)
    retained = torch.where(mask, values, torch.zeros_like(values))
    if method == "graph_wavelet_tight_frame":
        return frame.synthesize(
            GraphWaveletCoefficients(
                frame_artifact_sha256=frame.artifact_sha256,
                values=retained,
            )
        )
    if method == "graph_fourier":
        return (frame.eigenvectors @ retained).contiguous()
    if method == "native_nodes":
        return retained.contiguous()
    if method == "random_orthonormal":
        return (coefficients.random_basis @ retained).contiguous()
    raise ValueError("unknown graph compression method")


@dataclass(frozen=True, slots=True)
class CoefficientBudgetPoint:
    """One equal-budget reconstruction result."""

    method: GraphCompressionMethod
    budget: int
    retained_coefficient_count: int
    available_coefficient_count: int
    retained_analysis_energy_fraction: float
    relative_l2_error: float
    reconstruction_sha256: str

    def __post_init__(self) -> None:
        if self.method not in GRAPH_COMPRESSION_METHOD_ORDER:
            raise ValueError("coefficient-budget method differs")
        budget = _nonnegative_int(self.budget, label="budget")
        retained = _nonnegative_int(
            self.retained_coefficient_count,
            label="retained_coefficient_count",
        )
        available = _positive_int(
            self.available_coefficient_count,
            label="available_coefficient_count",
        )
        if retained != min(budget, available):
            raise ValueError("retained coefficient count differs from budget")
        for name in (
            "retained_analysis_energy_fraction",
            "relative_l2_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        _require_sha256(
            self.reconstruction_sha256,
            label="reconstruction_sha256",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "budget": self.budget,
            "retained_coefficient_count": self.retained_coefficient_count,
            "available_coefficient_count": self.available_coefficient_count,
            "retained_analysis_energy_fraction": (
                self.retained_analysis_energy_fraction
            ),
            "relative_l2_error": self.relative_l2_error,
            "reconstruction_sha256": self.reconstruction_sha256,
        }


def _default_budgets(node_count: int) -> tuple[int, ...]:
    values = {0, 1, node_count}
    current = 2
    while current < node_count:
        values.add(current)
        current *= 2
    return tuple(sorted(values))


def _budgets(
    values: Sequence[int] | None,
    *,
    node_count: int,
) -> tuple[int, ...]:
    if values is None:
        return _default_budgets(node_count)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("budgets must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int for value in result)
        or tuple(sorted(set(result))) != result
        or result[0] < 0
        or result[-1] > node_count
    ):
        raise ValueError(
            "matched budgets must be unique increasing integers in "
            "[0, node_count]"
        )
    return result


@dataclass(frozen=True, slots=True, eq=False)
class GraphWaveletAnalysisReport:
    """Hashed rate-distortion and localization results for one graph signal."""

    frame_artifact_sha256: str
    signal_sha256: str
    node_count: int
    frame_coefficient_count: int
    random_seed: int
    budgets: tuple[int, ...]
    curves: Mapping[str, tuple[CoefficientBudgetPoint, ...]]
    scale_localization: tuple[Mapping[str, object], ...]
    parseval_energy_relative_error: float
    full_frame_reconstruction_relative_error: float
    selection_semantics: str = (
        "per_signal_oracle_largest_absolute_scalar_coefficients_"
        "diagnostic_only"
    )
    heldout_evidence_claim: bool = False
    raw_signal_serialized: bool = False
    report_sha256: str = ""
    artifact_kind: str = _REPORT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.frame_artifact_sha256,
            label="frame_artifact_sha256",
        )
        _require_sha256(self.signal_sha256, label="signal_sha256")
        node_count = _positive_int(self.node_count, label="node_count")
        frame_count = _positive_int(
            self.frame_coefficient_count,
            label="frame_coefficient_count",
        )
        seed = _nonnegative_int(self.random_seed, label="random_seed")
        budgets = _budgets(self.budgets, node_count=node_count)
        curves = {
            method: tuple(points)
            for method, points in self.curves.items()
        }
        if tuple(curves) != GRAPH_COMPRESSION_METHOD_ORDER:
            raise ValueError("report curve order differs")
        for method in GRAPH_COMPRESSION_METHOD_ORDER:
            points = curves[method]
            if (
                tuple(point.budget for point in points) != budgets
                or any(point.method != method for point in points)
            ):
                raise ValueError("report coefficient-budget rows differ")
        localization = tuple(dict(row) for row in self.scale_localization)
        if not localization:
            raise ValueError("scale localization report cannot be empty")
        for name in (
            "parseval_energy_relative_error",
            "full_frame_reconstruction_relative_error",
        ):
            result = float(getattr(self, name))
            if not math.isfinite(result) or result < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, result)
        if (
            self.selection_semantics
            != (
                "per_signal_oracle_largest_absolute_scalar_coefficients_"
                "diagnostic_only"
            )
            or self.heldout_evidence_claim is not False
            or
            self.raw_signal_serialized is not False
            or self.artifact_kind != _REPORT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("graph-wavelet report provenance differs")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(
            self,
            "frame_coefficient_count",
            frame_count,
        )
        object.__setattr__(self, "random_seed", seed)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "curves", MappingProxyType(curves))
        object.__setattr__(self, "scale_localization", localization)
        computed = self._computed_sha256()
        if self.report_sha256:
            if (
                _require_sha256(
                    self.report_sha256,
                    label="report_sha256",
                )
                != computed
            ):
                raise ValueError("graph-wavelet report hash mismatch")
        else:
            object.__setattr__(self, "report_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "frame_artifact_sha256": self.frame_artifact_sha256,
            "signal_sha256": self.signal_sha256,
            "node_count": self.node_count,
            "frame_coefficient_count": self.frame_coefficient_count,
            "random_seed": self.random_seed,
            "budgets": self.budgets,
            "curves": {
                method: tuple(point.to_dict() for point in self.curves[method])
                for method in GRAPH_COMPRESSION_METHOD_ORDER
            },
            "scale_localization": tuple(
                dict(row) for row in self.scale_localization
            ),
            "parseval_energy_relative_error": (
                self.parseval_energy_relative_error
            ),
            "full_frame_reconstruction_relative_error": (
                self.full_frame_reconstruction_relative_error
            ),
            "selection_semantics": self.selection_semantics,
            "heldout_evidence_claim": self.heldout_evidence_claim,
            "raw_signal_serialized": self.raw_signal_serialized,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(_REPORT_DOMAIN, self._payload())

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.report_sha256:
            raise ValueError("graph-wavelet report hash mismatch")

    def to_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "report_sha256": self.report_sha256}


def _relative_error(reference: Tensor, candidate: Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference))
    numerator = float(torch.linalg.vector_norm(candidate - reference))
    if denominator <= torch.finfo(torch.float64).tiny:
        return 0.0 if numerator <= torch.finfo(torch.float64).tiny else numerator
    return numerator / denominator


def analyze_graph_wavelet_compression(
    frame: SpectralGraphWaveletFrame,
    signal: Tensor,
    *,
    budgets: Sequence[int] | None = None,
    random_seed: int = 0,
) -> GraphWaveletAnalysisReport:
    """Build equal-budget wavelet/Fourier/native/random compression curves."""

    if not isinstance(frame, SpectralGraphWaveletFrame):
        raise TypeError("frame must be a SpectralGraphWaveletFrame")
    frame.validate_integrity()
    canonical = _float_tensor(signal, label="graph signal", ndim=1)
    if canonical.shape != (frame.node_count,):
        raise ValueError("compression report requires one scalar per node")
    selected_budgets = _budgets(budgets, node_count=frame.node_count)
    coefficients = matched_graph_signal_coefficients(
        frame,
        canonical,
        random_seed=random_seed,
    )
    signal_energy = float(canonical.square().sum())
    wavelet_energy = float(coefficients.wavelet.square().sum())
    parseval_error = abs(wavelet_energy - signal_energy) / max(
        signal_energy,
        torch.finfo(torch.float64).tiny,
    )
    full_reconstruction = frame.reconstruct(canonical)
    full_error = _relative_error(canonical, full_reconstruction)

    curves: dict[str, tuple[CoefficientBudgetPoint, ...]] = {}
    for method in GRAPH_COMPRESSION_METHOD_ORDER:
        values = coefficients.values(method)
        available = int(values.numel())
        total_coefficient_energy = float(values.square().sum())
        points: list[CoefficientBudgetPoint] = []
        for budget in selected_budgets:
            mask = _top_budget_mask(values, budget)
            retained_energy = float(values[mask].square().sum())
            fraction = (
                retained_energy / total_coefficient_energy
                if total_coefficient_energy
                > torch.finfo(torch.float64).tiny
                else 0.0
            )
            reconstruction = reconstruct_matched_graph_signal(
                frame,
                coefficients,
                method,  # type: ignore[arg-type]
                budget,
            )
            points.append(
                CoefficientBudgetPoint(
                    method=method,  # type: ignore[arg-type]
                    budget=budget,
                    retained_coefficient_count=min(budget, available),
                    available_coefficient_count=available,
                    retained_analysis_energy_fraction=fraction,
                    relative_l2_error=_relative_error(
                        canonical,
                        reconstruction,
                    ),
                    reconstruction_sha256=_tensor_sha256(reconstruction),
                )
            )
        curves[method] = tuple(points)
    return GraphWaveletAnalysisReport(
        frame_artifact_sha256=frame.artifact_sha256,
        signal_sha256=_tensor_sha256(canonical),
        node_count=frame.node_count,
        frame_coefficient_count=frame.frame_coefficient_count,
        random_seed=random_seed,
        budgets=selected_budgets,
        curves=curves,
        scale_localization=frame.scale_localization_summary(),
        parseval_energy_relative_error=parseval_error,
        full_frame_reconstruction_relative_error=full_error,
    )
