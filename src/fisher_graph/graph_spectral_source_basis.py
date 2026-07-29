"""Fit-only graph Fourier source bases for conditional modal executors.

This module turns a real causal response tensor

``H[source_mode, source_origin, lag, target_mode]``

into two real, orthonormal source-mode bases:

* a signed phase-aware graph basis built from the real part of complex
  spectral coherency; and
* a phase-blind magnitude-similarity control basis.

Only declared fit origins are indexed before the graph is built.  The graph
basis is therefore suitable for a held-out-origin rate-distortion experiment.
It remains a source basis, not a transfer executor by itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor

from .conditional_spectral_generator import _tensor_sha256


GraphSourceBasisKind = Literal[
    "signed_phase_graph_low_frequency",
    "phase_blind_magnitude_graph_low_frequency",
]

__all__ = [
    "FitOnlyGraphSourceBasis",
    "GraphSourceBasisKind",
    "fit_graph_source_bases",
]


_ARTIFACT_KIND = "fisher_graph.fit_only_graph_source_basis"
_FORMAT_VERSION = 1
_HASH_DOMAIN = b"fisher-graph:fit-only-graph-source-basis:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RELATIVE_SUPPORT_FLOOR = math.sqrt(torch.finfo(torch.float64).eps)
_SIGNED_SEMANTICS = (
    "normalized_signed_real_complex_coherency_laplacian_fit_origins_only"
)
_MAGNITUDE_SEMANTICS = (
    "normalized_phase_blind_magnitude_similarity_laplacian_fit_origins_only"
)
_ENERGY_SEMANTICS = (
    "parseval_weighted_complex_fit_response_energy_after_real_graph_basis_"
    "projection"
)
_TENSOR_FIELDS = (
    "source_response_norms",
    "signed_eigenvalues",
    "signed_eigenvectors",
    "magnitude_eigenvalues",
    "magnitude_eigenvectors",
    "signed_projection_energy",
    "magnitude_projection_energy",
)


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


def _canonical_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
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
        result.ndim != ndim
        or any(int(width) <= 0 for width in result.shape)
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError(f"{label} must be finite, nonempty rank-{ndim} data")
    return result


def _tensor_sha256_local(value: Tensor) -> str:
    tensor = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(width) for width in tensor.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _origins(
    values: Sequence[int],
    *,
    label: str,
    minimum_count: int = 1,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        len(result) < minimum_count
        or any(type(value) is not int or value < 0 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(f"{label} must be strictly increasing integers")
    return result


def _canonicalize_column_signs(value: Tensor) -> Tensor:
    result = value.clone()
    for column in range(result.shape[1]):
        pivot = int(torch.argmax(result[:, column].abs()))
        if float(result[pivot, column]) < 0.0:
            result[:, column] *= -1.0
    return result.contiguous()


def _normalized_laplacian(adjacency: Tensor) -> Tensor:
    degree = adjacency.abs().sum(dim=1)
    active = degree > torch.finfo(torch.float64).eps
    inverse = torch.zeros_like(degree)
    inverse[active] = degree[active].rsqrt()
    normalized = inverse[:, None] * adjacency * inverse[None, :]
    laplacian = torch.diag(active.to(dtype=torch.float64)) - normalized
    return ((laplacian + laplacian.T) / 2).contiguous()


def _projection_energy(basis: Tensor, signal: Tensor) -> Tensor:
    coefficients = basis.T.to(dtype=torch.complex128) @ signal
    energy = coefficients.abs().square().sum(dim=1).real
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps:
        return torch.zeros_like(energy)
    return (energy / total).contiguous()


@dataclass(frozen=True, slots=True)
class FitOnlyGraphSourceBasis:
    """Authenticated full signed and magnitude graph source bases."""

    response_binding_sha256: str
    fit_weighted_kernels_sha256: str
    fit_origins: tuple[int, ...]
    fft_length: int
    relative_support_floor: float
    source_response_norms: Tensor
    signed_eigenvalues: Tensor
    signed_eigenvectors: Tensor
    magnitude_eigenvalues: Tensor
    magnitude_eigenvectors: Tensor
    signed_projection_energy: Tensor
    magnitude_projection_energy: Tensor
    signed_graph_semantics: str = _SIGNED_SEMANTICS
    magnitude_graph_semantics: str = _MAGNITUDE_SEMANTICS
    projection_energy_semantics: str = _ENERGY_SEMANTICS
    heldout_origins_used_for_basis: bool = False
    transfer_executor_claim: bool = False
    compression_claim: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.response_binding_sha256,
            label="response_binding_sha256",
        )
        _require_sha256(
            self.fit_weighted_kernels_sha256,
            label="fit_weighted_kernels_sha256",
        )
        object.__setattr__(
            self,
            "fit_origins",
            _origins(self.fit_origins, label="fit_origins", minimum_count=2),
        )
        if type(self.fft_length) is not int or self.fft_length <= 0:
            raise ValueError("fft_length must be a positive integer")
        if (
            isinstance(self.relative_support_floor, bool)
            or not isinstance(self.relative_support_floor, (int, float))
            or not math.isfinite(float(self.relative_support_floor))
            or float(self.relative_support_floor) <= 0.0
        ):
            raise ValueError("relative_support_floor must be finite and positive")
        object.__setattr__(
            self,
            "relative_support_floor",
            float(self.relative_support_floor),
        )
        for field in _TENSOR_FIELDS:
            object.__setattr__(
                self,
                field,
                _canonical_tensor(
                    getattr(self, field),
                    label=field,
                    ndim=2 if field.endswith("eigenvectors") else 1,
                ),
            )
        modes = int(self.source_response_norms.numel())
        if any(
            tensor.shape != (modes,)
            for tensor in (
                self.signed_eigenvalues,
                self.magnitude_eigenvalues,
                self.signed_projection_energy,
                self.magnitude_projection_energy,
            )
        ) or any(
            tensor.shape != (modes, modes)
            for tensor in (
                self.signed_eigenvectors,
                self.magnitude_eigenvectors,
            )
        ):
            raise ValueError("graph basis tensor shapes differ")
        identity = torch.eye(modes, dtype=torch.float64)
        for label, vectors in (
            ("signed", self.signed_eigenvectors),
            ("magnitude", self.magnitude_eigenvectors),
        ):
            if not torch.allclose(
                vectors.T @ vectors,
                identity,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError(f"{label} graph basis is not orthonormal")
        for label, values in (
            ("signed", self.signed_eigenvalues),
            ("magnitude", self.magnitude_eigenvalues),
        ):
            if (
                bool((values < -1e-10).any())
                or bool((values > 2.0 + 1e-10).any())
                or (
                    values.numel() > 1
                    and bool((values[1:] < values[:-1] - 1e-12).any())
                )
            ):
                raise ValueError(f"{label} graph eigenvalues are invalid")
        if bool((self.source_response_norms < 0.0).any()):
            raise ValueError("source response norms cannot be negative")
        for label, energy in (
            ("signed", self.signed_projection_energy),
            ("magnitude", self.magnitude_projection_energy),
        ):
            total = float(energy.sum())
            if bool((energy < -1e-12).any()) or not (
                abs(total) <= 1e-10 or abs(total - 1.0) <= 1e-10
            ):
                raise ValueError(f"{label} projection energy is invalid")
        if (
            self.signed_graph_semantics != _SIGNED_SEMANTICS
            or self.magnitude_graph_semantics != _MAGNITUDE_SEMANTICS
            or self.projection_energy_semantics != _ENERGY_SEMANTICS
            or self.heldout_origins_used_for_basis is not False
            or self.transfer_executor_claim is not False
            or self.compression_claim is not False
            or self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("graph source basis provenance drifted")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("graph source basis artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_modes(self) -> int:
        return int(self.source_response_norms.numel())

    def basis(
        self,
        kind: GraphSourceBasisKind,
        rank: int,
    ) -> Tensor:
        self.validate_integrity()
        if type(rank) is not int or not 1 <= rank <= self.source_modes:
            raise ValueError("rank must lie within the source mode count")
        if kind == "signed_phase_graph_low_frequency":
            value = self.signed_eigenvectors
        elif kind == "phase_blind_magnitude_graph_low_frequency":
            value = self.magnitude_eigenvectors
        else:
            raise ValueError("graph source basis kind is invalid")
        return value[:, :rank].clone()

    def projection_relative_error(
        self,
        kind: GraphSourceBasisKind,
        rank: int,
    ) -> float:
        if kind == "signed_phase_graph_low_frequency":
            energy = self.signed_projection_energy
        elif kind == "phase_blind_magnitude_graph_low_frequency":
            energy = self.magnitude_projection_energy
        else:
            raise ValueError("graph source basis kind is invalid")
        if type(rank) is not int or not 1 <= rank <= self.source_modes:
            raise ValueError("rank must lie within the source mode count")
        return math.sqrt(max(0.0, 1.0 - float(energy[:rank].sum())))

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "response_binding_sha256": self.response_binding_sha256,
            "fit_weighted_kernels_sha256": (
                self.fit_weighted_kernels_sha256
            ),
            "fit_origins": self.fit_origins,
            "fft_length": self.fft_length,
            "relative_support_floor": self.relative_support_floor,
            "source_modes": self.source_modes,
            "tensor_sha256s": {
                field: _tensor_sha256_local(getattr(self, field))
                for field in _TENSOR_FIELDS
            },
            "tensor_shapes": {
                field: tuple(int(width) for width in getattr(self, field).shape)
                for field in _TENSOR_FIELDS
            },
            "signed_graph_semantics": self.signed_graph_semantics,
            "magnitude_graph_semantics": self.magnitude_graph_semantics,
            "projection_energy_semantics": self.projection_energy_semantics,
            "heldout_origins_used_for_basis": (
                self.heldout_origins_used_for_basis
            ),
            "transfer_executor_claim": self.transfer_executor_claim,
            "compression_claim": self.compression_claim,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload())

    def validate_integrity(self) -> None:
        for field in _TENSOR_FIELDS:
            value = getattr(self, field)
            if (
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{field} drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("graph source basis artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "signed_projection_relative_error_by_rank": {
                str(rank): self.projection_relative_error(
                    "signed_phase_graph_low_frequency",
                    rank,
                )
                for rank in range(1, self.source_modes + 1)
            },
            "magnitude_projection_relative_error_by_rank": {
                str(rank): self.projection_relative_error(
                    "phase_blind_magnitude_graph_low_frequency",
                    rank,
                )
                for rank in range(1, self.source_modes + 1)
            },
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            **{
                field: getattr(self, field).clone()
                for field in _TENSOR_FIELDS
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "FitOnlyGraphSourceBasis":
        if not isinstance(raw, Mapping):
            raise TypeError("graph source basis state must be a mapping")
        expected = {
            "artifact_kind",
            "format_version",
            "response_binding_sha256",
            "fit_weighted_kernels_sha256",
            "fit_origins",
            "fft_length",
            "relative_support_floor",
            "source_modes",
            "tensor_sha256s",
            "tensor_shapes",
            "signed_graph_semantics",
            "magnitude_graph_semantics",
            "projection_energy_semantics",
            "heldout_origins_used_for_basis",
            "transfer_executor_claim",
            "compression_claim",
            *_TENSOR_FIELDS,
            "artifact_sha256",
        }
        if set(raw) != expected:
            raise ValueError("graph source basis state fields differ")
        tensor_hashes = raw["tensor_sha256s"]
        tensor_shapes = raw["tensor_shapes"]
        if (
            not isinstance(tensor_hashes, Mapping)
            or set(tensor_hashes) != set(_TENSOR_FIELDS)
            or not isinstance(tensor_shapes, Mapping)
            or set(tensor_shapes) != set(_TENSOR_FIELDS)
        ):
            raise ValueError("graph source basis tensor declarations differ")
        tensors: dict[str, Tensor] = {}
        for field in _TENSOR_FIELDS:
            value = raw[field]
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or _tensor_sha256_local(value) != tensor_hashes[field]
                or tuple(value.shape) != tuple(tensor_shapes[field])
            ):
                raise ValueError(f"serialized {field} is invalid")
            tensors[field] = value
        result = cls(
            response_binding_sha256=raw[
                "response_binding_sha256"
            ],  # type: ignore[arg-type]
            fit_weighted_kernels_sha256=raw[
                "fit_weighted_kernels_sha256"
            ],  # type: ignore[arg-type]
            fit_origins=tuple(raw["fit_origins"]),  # type: ignore[arg-type]
            fft_length=raw["fft_length"],  # type: ignore[arg-type]
            relative_support_floor=raw[
                "relative_support_floor"
            ],  # type: ignore[arg-type]
            **tensors,
            signed_graph_semantics=raw[
                "signed_graph_semantics"
            ],  # type: ignore[arg-type]
            magnitude_graph_semantics=raw[
                "magnitude_graph_semantics"
            ],  # type: ignore[arg-type]
            projection_energy_semantics=raw[
                "projection_energy_semantics"
            ],  # type: ignore[arg-type]
            heldout_origins_used_for_basis=raw[
                "heldout_origins_used_for_basis"
            ],  # type: ignore[arg-type]
            transfer_executor_claim=raw[
                "transfer_executor_claim"
            ],  # type: ignore[arg-type]
            compression_claim=raw["compression_claim"],  # type: ignore[arg-type]
            artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=raw["artifact_kind"],  # type: ignore[arg-type]
            format_version=raw["format_version"],  # type: ignore[arg-type]
        )
        if raw["source_modes"] != result.source_modes:
            raise ValueError("serialized source mode count differs")
        return result


def fit_graph_source_bases(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    fit_origins: Sequence[int],
    *,
    response_binding_sha256: str,
    fft_length: int | None = None,
) -> FitOnlyGraphSourceBasis:
    """Build signed and magnitude graph bases without reading held-out origins."""

    _require_sha256(
        response_binding_sha256,
        label="response_binding_sha256",
    )
    kernels = _canonical_tensor(responses, label="responses", ndim=4)
    scales = _canonical_tensor(
        source_scales,
        label="source_scales",
        ndim=1,
    )
    if kernels.shape[0] != scales.numel():
        raise ValueError("responses and source_scales have different widths")
    if bool((scales <= 0.0).any()):
        raise ValueError("source_scales must be strictly positive")
    measured = _origins(origins, label="origins")
    knots = _origins(fit_origins, label="fit_origins", minimum_count=2)
    if len(measured) != kernels.shape[1] or not set(knots).issubset(measured):
        raise ValueError("origin axes do not match the response tensor")
    lag_count = int(kernels.shape[2])
    if fft_length is None:
        fft_length = 1 << (lag_count - 1).bit_length()
    if (
        type(fft_length) is not int
        or fft_length <= 0
        or fft_length < lag_count
    ):
        raise ValueError("fft_length cannot truncate causal lags")
    ordinals = torch.tensor(
        [measured.index(origin) for origin in knots],
        dtype=torch.int64,
    )
    # This selection is the only read of the origin axis used by the graph.
    fit_kernels = kernels.index_select(1, ordinals).contiguous()
    weighted = (
        fit_kernels * scales.view(-1, 1, 1, 1)
    ).contiguous()
    spectrum = torch.fft.rfft(weighted, n=fft_length, dim=2)
    frequency_count = fft_length // 2 + 1
    parseval_weights = torch.ones(frequency_count, dtype=torch.float64)
    if fft_length % 2 == 0:
        parseval_weights[1:-1] = 2.0
    else:
        parseval_weights[1:] = 2.0
    flattened = (
        spectrum * parseval_weights.sqrt()[None, None, :, None]
    ).reshape(kernels.shape[0], -1)
    norms = torch.linalg.vector_norm(flattened, dim=1)
    threshold = max(
        torch.finfo(torch.float64).eps,
        float(norms.max()) * _RELATIVE_SUPPORT_FLOOR,
    )
    active = norms > threshold
    normalized = torch.zeros_like(flattened)
    normalized[active] = flattened[active] / norms[active, None]
    coherency = normalized @ normalized.mH
    coherency = (coherency + coherency.mH) / 2
    indices = torch.arange(coherency.shape[0])
    coherency[indices[active], indices[active]] = 1.0 + 0.0j
    coherency[indices[~active], indices[~active]] = 0.0 + 0.0j
    signed_adjacency = coherency.real.contiguous()
    signed_adjacency.fill_diagonal_(0.0)

    magnitude_features = flattened.abs()
    magnitude_norms = torch.linalg.vector_norm(magnitude_features, dim=1)
    magnitude_normalized = torch.zeros_like(magnitude_features)
    magnitude_normalized[active] = (
        magnitude_features[active] / magnitude_norms[active, None]
    )
    magnitude_adjacency = (
        magnitude_normalized @ magnitude_normalized.T
    ).clamp(0.0, 1.0)
    magnitude_adjacency.fill_diagonal_(0.0)

    signed_laplacian = _normalized_laplacian(signed_adjacency)
    magnitude_laplacian = _normalized_laplacian(magnitude_adjacency)
    signed_values, signed_vectors = torch.linalg.eigh(signed_laplacian)
    magnitude_values, magnitude_vectors = torch.linalg.eigh(
        magnitude_laplacian
    )
    signed_values = signed_values.clamp_min(0.0).contiguous()
    magnitude_values = magnitude_values.clamp_min(0.0).contiguous()
    signed_vectors = _canonicalize_column_signs(signed_vectors)
    magnitude_vectors = _canonicalize_column_signs(magnitude_vectors)
    return FitOnlyGraphSourceBasis(
        response_binding_sha256=response_binding_sha256,
        fit_weighted_kernels_sha256=_tensor_sha256(weighted),
        fit_origins=knots,
        fft_length=fft_length,
        relative_support_floor=_RELATIVE_SUPPORT_FLOOR,
        source_response_norms=norms,
        signed_eigenvalues=signed_values,
        signed_eigenvectors=signed_vectors,
        magnitude_eigenvalues=magnitude_values,
        magnitude_eigenvectors=magnitude_vectors,
        signed_projection_energy=_projection_energy(
            signed_vectors,
            flattened,
        ),
        magnitude_projection_energy=_projection_energy(
            magnitude_vectors,
            flattened,
        ),
    )
