"""Graph-organized grouping and routing for a frozen global-SVD executor.

The global SVD remains the numerical compression basis.  A fit-only graph
Fourier basis supplies signatures over the retained SVD generator columns.
Those signatures organize generators into graph-frequency packs without
changing any weight.  Enabling every pack is therefore algebraically
equivalent to the source SVD plan.

The prepared runtime accepts an explicit per-source-row pack mask.  A small
reference router can derive such a mask from latent energy and certified core
operator-norm bounds.  This is conditional-compute infrastructure, not a
latency claim.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor, nn

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    _FACTORIZATION,
    _RANK_SEMANTICS,
    _canonical_runtime_grid,
)
from .graph_spectral_source_basis import FitOnlyGraphSourceBasis


OrganizationKind = Literal[
    "signed_gfa_dyadic",
    "magnitude_gfa_dyadic",
    "signed_row_permutation_dyadic",
    "singular_contiguous_control",
    "random_size_matched_control",
]

__all__ = [
    "GraphOrganizedSVDExecutionAccounting",
    "GraphOrganizedSVDPlan",
    "OrganizationKind",
    "PreparedGraphOrganizedSVD",
    "organize_conditional_svd_with_graph",
]


_ARTIFACT_KIND = "fisher_graph.graph_organized_svd_plan"
_FORMAT_VERSION = 1
_HASH_DOMAIN = b"fisher-graph:graph-organized-svd-plan:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:graph-organized-svd-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)
_ORGANIZATION_KINDS = frozenset(
    {
        "signed_gfa_dyadic",
        "magnitude_gfa_dyadic",
        "signed_row_permutation_dyadic",
        "singular_contiguous_control",
        "random_size_matched_control",
    }
)
_BAND_SEMANTICS = (
    "fit_only_graph_frequency_signatures_with_assignment_rule_named_by_"
    "organization_kind_and_size_matched_controls"
)
_EXECUTION_SEMANTICS = (
    "global_svd_source_projection_then_pack_masked_causal_core_transport_"
    "with_full_target_basis_folded_into_core"
)
_ROUTER_BOUND_SEMANTICS = (
    "fit_knot_pack_core_spectral_norms_linearly_upper_bound_interpolated_"
    "pack_contribution"
)
_FLOAT_TENSORS = (
    "source_scales",
    "source_basis",
    "knot_cores",
    "source_singular_values",
    "target_singular_values",
    "graph_eigenvalues",
    "graph_eigenvectors",
    "component_graph_energy",
    "band_mass",
    "core_operator_norm_bounds",
)
_INT_TENSORS = (
    "pack_assignments",
    "component_permutation",
    "pack_offsets",
    "graph_row_permutation",
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


def _float_tensor(
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
        raise ValueError(f"{label} must be finite nonempty rank-{ndim} data")
    return result


def _int_tensor(
    value: object,
    *,
    label: str,
    ndim: int = 1,
) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.int64:
        raise TypeError(f"{label} must be a torch.int64 Tensor")
    result = value.detach().to(device="cpu").contiguous().clone()
    if (
        result.ndim != ndim
        or any(int(width) <= 0 for width in result.shape)
    ):
        raise ValueError(f"{label} must be nonempty rank-{ndim} data")
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(int(width) for width in tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _origin_tuple(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("fit origins must be a sequence")
    result = tuple(values)
    if (
        len(result) < 2
        or any(type(value) is not int or value < 0 for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError("fit origins must be strictly increasing integers")
    return result


def _boundaries(
    values: Sequence[int],
    *,
    width: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("frequency band boundaries must be a sequence")
    result = tuple(values)
    if (
        len(result) < 3
        or result[0] != 0
        or result[-1] != width
        or tuple(sorted(set(result))) != result
        or any(type(value) is not int for value in result)
    ):
        raise ValueError(
            "frequency bands must strictly partition the graph frequencies"
        )
    return result


def _default_boundaries(width: int) -> tuple[int, ...]:
    if width < 2:
        raise ValueError("graph organization requires at least two modes")
    return tuple(
        sorted(
            {
                0,
                max(1, width // 8),
                max(1, width // 4),
                max(1, width // 2),
                width,
            }
        )
    )


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _close(
    left: Tensor | float,
    right: Tensor | float,
    *,
    scale: float = 1.0,
) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return bool(
            torch.allclose(
                torch.as_tensor(left, dtype=torch.float64),
                torch.as_tensor(right, dtype=torch.float64),
                atol=2e-10 * max(scale, 1.0),
                rtol=2e-10,
            )
        )
    return math.isclose(
        float(left),
        float(right),
        abs_tol=2e-10 * max(scale, 1.0),
        rel_tol=2e-10,
    )


def _inflated_operator_norm(value: Tensor) -> Tensor:
    """Return a float64 one-sided certificate for one matrix 2-norm."""

    norm = torch.linalg.matrix_norm(value, ord=2)
    inflation = norm * 1e-12 + torch.finfo(torch.float64).tiny
    upper = norm + inflation
    return torch.nextafter(
        upper,
        torch.full_like(upper, torch.inf),
    )


def _pack_permutation(assignments: Tensor) -> Tensor:
    return torch.tensor(
        sorted(
            range(assignments.numel()),
            key=lambda index: (int(assignments[index]), index),
        ),
        dtype=torch.int64,
    )


def _pack_offsets(assignments: Tensor, pack_count: int) -> Tensor:
    counts = torch.bincount(assignments, minlength=pack_count)
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int64),
            counts.cumsum(dim=0),
        )
    ).contiguous()


def _size_matched_assignments(
    counts: Sequence[int],
    *,
    kind: OrganizationKind,
    seed: int,
) -> Tensor:
    count_tuple = tuple(counts)
    rank = sum(count_tuple)
    if (
        not count_tuple
        or any(type(value) is not int or value <= 0 for value in count_tuple)
    ):
        raise ValueError("matched pack counts must be positive integers")
    assignments = torch.empty(rank, dtype=torch.int64)
    if kind == "singular_contiguous_control":
        ordering = torch.arange(rank, dtype=torch.int64)
    elif kind == "random_size_matched_control":
        ordering = torch.randperm(
            rank,
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        raise ValueError("size-matched assignment kind is invalid")
    cursor = 0
    for pack, count in enumerate(count_tuple):
        assignments[ordering[cursor : cursor + count]] = pack
        cursor += count
    return assignments.contiguous()


@dataclass(frozen=True, slots=True)
class GraphOrganizedSVDExecutionAccounting:
    """Exact executor work plus explicit router primitives before sorting."""

    batch_size: int
    sequence_length: int
    valid_source_rows: int
    valid_target_rows: int
    admitted_causal_pairs: int
    active_pack_instances: int
    active_rank_instances: int
    interpolated_active_rank_instances: int
    admitted_active_rank_pairs: int
    admitted_active_pack_pairs: int
    source_modes: int
    target_modes: int
    source_rank: int
    pack_count: int
    lag_count: int
    router_evaluated: bool = False

    def __post_init__(self) -> None:
        if type(self.router_evaluated) is not bool:
            raise TypeError("router_evaluated must be a bool")
        for field in (
            "batch_size",
            "sequence_length",
            "source_modes",
            "target_modes",
            "source_rank",
            "pack_count",
            "lag_count",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in (
            "valid_source_rows",
            "valid_target_rows",
            "admitted_causal_pairs",
            "active_pack_instances",
            "active_rank_instances",
            "interpolated_active_rank_instances",
            "admitted_active_rank_pairs",
            "admitted_active_pack_pairs",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if (
            self.pack_count > self.source_rank
            or self.valid_source_rows > self.batch_size * self.sequence_length
            or self.valid_target_rows > self.batch_size * self.sequence_length
            or self.active_rank_instances
            > self.valid_source_rows * self.source_rank
            or self.active_pack_instances
            > self.valid_source_rows * self.pack_count
            or self.interpolated_active_rank_instances
            > self.active_rank_instances
            or self.admitted_active_rank_pairs
            > self.admitted_causal_pairs * self.source_rank
            or self.admitted_active_pack_pairs
            > self.admitted_causal_pairs * self.pack_count
        ):
            raise ValueError("execution accounting bounds are inconsistent")

    @property
    def source_projection_macs(self) -> int:
        return self.valid_source_rows * self.source_modes * self.source_rank

    @property
    def source_standardization_divisions(self) -> int:
        return self.valid_source_rows * self.source_modes

    @property
    def routed_core_transport_macs(self) -> int:
        return self.admitted_active_rank_pairs * self.target_modes

    @property
    def factorized_linear_macs(self) -> int:
        return self.source_projection_macs + self.routed_core_transport_macs

    @property
    def dense_linear_macs(self) -> int:
        return (
            self.admitted_causal_pairs
            * self.source_modes
            * self.target_modes
        )

    @property
    def core_accumulation_additions(self) -> int:
        return self.admitted_active_pack_pairs * self.target_modes

    @property
    def lazy_core_interpolation_multiplies(self) -> int:
        return (
            2
            * self.interpolated_active_rank_instances
            * self.lag_count
            * self.target_modes
        )

    @property
    def lazy_core_interpolation_additions(self) -> int:
        return (
            self.interpolated_active_rank_instances
            * self.lag_count
            * self.target_modes
        )

    @property
    def reference_router_latent_square_multiplies(self) -> int:
        return (
            self.valid_source_rows * self.source_rank
            if self.router_evaluated
            else 0
        )

    @property
    def reference_router_bound_interpolation_multiplies(self) -> int:
        return 0 if not self.router_evaluated else (
            2
            * self.valid_source_rows
            * self.pack_count
            * self.lag_count
        )

    @property
    def reference_router_bound_interpolation_additions(self) -> int:
        return 0 if not self.router_evaluated else (
            self.valid_source_rows
            * self.pack_count
            * self.lag_count
        )

    @property
    def reference_router_latent_norm_reduction_additions(self) -> int:
        return 0 if not self.router_evaluated else (
            self.valid_source_rows * (self.source_rank - self.pack_count)
        )

    @property
    def reference_router_bound_norm_square_multiplies(self) -> int:
        return 0 if not self.router_evaluated else (
            self.valid_source_rows
            * self.pack_count
            * self.lag_count
        )

    @property
    def reference_router_bound_norm_reduction_additions(self) -> int:
        return 0 if not self.router_evaluated else (
            self.valid_source_rows
            * self.pack_count
            * (self.lag_count - 1)
        )

    @property
    def reference_router_square_roots(self) -> int:
        return (
            2 * self.valid_source_rows * self.pack_count
            if self.router_evaluated
            else 0
        )

    @property
    def reference_router_score_multiplies(self) -> int:
        return (
            self.valid_source_rows * self.pack_count
            if self.router_evaluated
            else 0
        )

    @property
    def reference_router_sort_rows(self) -> int:
        return self.valid_source_rows if self.router_evaluated else 0

    @property
    def reference_router_score_total_additions(self) -> int:
        return (
            self.valid_source_rows * (self.pack_count - 1)
            if self.router_evaluated
            else 0
        )

    @property
    def reference_router_retained_accumulation_additions(self) -> int:
        return self.active_pack_instances if self.router_evaluated else 0

    @property
    def reference_router_retention_threshold_comparisons(self) -> int:
        return self.active_pack_instances if self.router_evaluated else 0

    @property
    def active_rank_fraction(self) -> float:
        denominator = self.valid_source_rows * self.source_rank
        return self.active_rank_instances / denominator if denominator else 0.0

    def metadata(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "batch_size",
                "sequence_length",
                "valid_source_rows",
                "valid_target_rows",
                "admitted_causal_pairs",
                "active_pack_instances",
                "active_rank_instances",
                "interpolated_active_rank_instances",
                "admitted_active_rank_pairs",
                "admitted_active_pack_pairs",
                "source_modes",
                "target_modes",
                "source_rank",
                "pack_count",
                "lag_count",
                "router_evaluated",
                "source_standardization_divisions",
                "source_projection_macs",
                "routed_core_transport_macs",
                "factorized_linear_macs",
                "dense_linear_macs",
                "core_accumulation_additions",
                "lazy_core_interpolation_multiplies",
                "lazy_core_interpolation_additions",
                "reference_router_latent_square_multiplies",
                "reference_router_latent_norm_reduction_additions",
                "reference_router_bound_interpolation_multiplies",
                "reference_router_bound_interpolation_additions",
                "reference_router_bound_norm_square_multiplies",
                "reference_router_bound_norm_reduction_additions",
                "reference_router_square_roots",
                "reference_router_score_multiplies",
                "reference_router_score_total_additions",
                "reference_router_retained_accumulation_additions",
                "reference_router_retention_threshold_comparisons",
                "reference_router_sort_rows",
                "active_rank_fraction",
            )
        } | {
            "reference_router_sort_comparisons_counted": False,
        }


@dataclass(frozen=True, slots=True)
class GraphOrganizedSVDPlan:
    """Strict global-SVD weights plus graph-derived generator packs."""

    response_binding_sha256: str
    source_plan_artifact_sha256: str
    graph_basis_artifact_sha256: str
    fit_weighted_kernels_sha256: str
    fit_knot_origins: tuple[int, ...]
    fft_length: int
    organization_kind: OrganizationKind
    organization_seed: int
    frequency_band_boundaries: tuple[int, ...]
    source_scales: Tensor
    source_basis: Tensor
    knot_cores: Tensor
    source_singular_values: Tensor
    target_singular_values: Tensor
    graph_eigenvalues: Tensor
    graph_eigenvectors: Tensor
    component_graph_energy: Tensor
    band_mass: Tensor
    pack_assignments: Tensor
    component_permutation: Tensor
    pack_offsets: Tensor
    graph_row_permutation: Tensor
    core_operator_norm_bounds: Tensor
    weighted_total_energy: float
    weighted_retained_energy: float
    weighted_relative_error: float
    source_parseval_relative_error: float
    target_parseval_relative_error: float
    band_semantics: str = _BAND_SEMANTICS
    execution_semantics: str = _EXECUTION_SEMANTICS
    router_bound_semantics: str = _ROUTER_BOUND_SEMANTICS
    heldout_origins_used_for_organization: bool = False
    conditional_compute_claim: bool = False
    latency_claim: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        for field in (
            "response_binding_sha256",
            "source_plan_artifact_sha256",
            "graph_basis_artifact_sha256",
            "fit_weighted_kernels_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        object.__setattr__(
            self,
            "fit_knot_origins",
            _origin_tuple(self.fit_knot_origins),
        )
        if type(self.fft_length) is not int or self.fft_length <= 0:
            raise ValueError("fft_length must be a positive integer")
        if self.organization_kind not in _ORGANIZATION_KINDS:
            raise ValueError("organization_kind is invalid")
        if type(self.organization_seed) is not int or self.organization_seed < 0:
            raise ValueError("organization_seed must be nonnegative")
        if (
            self.organization_kind
            in {
                "signed_gfa_dyadic",
                "magnitude_gfa_dyadic",
                "singular_contiguous_control",
            }
            and self.organization_seed != 0
        ):
            raise ValueError(
                "deterministic organization kinds require seed zero"
            )
        for field in _FLOAT_TENSORS:
            object.__setattr__(
                self,
                field,
                _float_tensor(
                    getattr(self, field),
                    label=field,
                    ndim=(
                        4
                        if field == "knot_cores"
                        else 3
                        if field == "core_operator_norm_bounds"
                        else 2
                        if field
                        in (
                            "source_basis",
                            "graph_eigenvectors",
                            "component_graph_energy",
                            "band_mass",
                        )
                        else 1
                    ),
                ),
            )
        for field in _INT_TENSORS:
            object.__setattr__(
                self,
                field,
                _int_tensor(getattr(self, field), label=field),
            )
        source_modes = int(self.source_basis.shape[0])
        source_rank = int(self.source_basis.shape[1])
        target_modes = int(self.knot_cores.shape[3])
        pack_count = int(self.pack_offsets.numel() - 1)
        object.__setattr__(
            self,
            "frequency_band_boundaries",
            _boundaries(
                self.frequency_band_boundaries,
                width=source_modes,
            ),
        )
        if pack_count != len(self.frequency_band_boundaries) - 1:
            raise ValueError("pack count and graph frequency bands differ")
        if (
            self.source_scales.shape != (source_modes,)
            or self.graph_eigenvalues.shape != (source_modes,)
            or self.graph_eigenvectors.shape != (source_modes, source_modes)
            or self.component_graph_energy.shape
            != (source_modes, source_rank)
            or self.band_mass.shape != (pack_count, source_rank)
            or self.pack_assignments.shape != (source_rank,)
            or self.component_permutation.shape != (source_rank,)
            or self.pack_offsets.shape != (pack_count + 1,)
            or self.graph_row_permutation.shape != (source_modes,)
            or self.knot_cores.shape[:3]
            != (
                len(self.fit_knot_origins),
                self.knot_cores.shape[1],
                source_rank,
            )
            or self.core_operator_norm_bounds.shape
            != (
                len(self.fit_knot_origins),
                self.knot_cores.shape[1],
                pack_count,
            )
        ):
            raise ValueError("graph-organized tensor shapes differ")
        if bool((self.source_scales <= 0.0).any()):
            raise ValueError("source scales must be strictly positive")
        expected_row_permutation = (
            torch.randperm(
                source_modes,
                generator=torch.Generator().manual_seed(
                    self.organization_seed
                ),
            )
            if self.organization_kind == "signed_row_permutation_dyadic"
            else torch.arange(source_modes, dtype=torch.int64)
        )
        if not torch.equal(
            self.graph_row_permutation,
            expected_row_permutation,
        ):
            raise ValueError("graph row permutation differs from its seed")
        source_identity = torch.eye(source_rank, dtype=torch.float64)
        graph_identity = torch.eye(source_modes, dtype=torch.float64)
        if not _close(self.source_basis.T @ self.source_basis, source_identity):
            raise ValueError("source basis is not orthonormal")
        if not _close(
            self.graph_eigenvectors.T @ self.graph_eigenvectors,
            graph_identity,
        ):
            raise ValueError("graph eigenvectors are not orthonormal")
        if (
            bool((self.graph_eigenvalues < -1e-10).any())
            or bool((self.graph_eigenvalues > 2.0 + 1e-10).any())
            or bool(
                (
                    self.graph_eigenvalues[1:]
                    < self.graph_eigenvalues[:-1] - 1e-12
                ).any()
            )
        ):
            raise ValueError("graph eigenvalues are invalid")
        for boundary in self.frequency_band_boundaries[1:-1]:
            gap = float(
                self.graph_eigenvalues[boundary]
                - self.graph_eigenvalues[boundary - 1]
            )
            if abs(gap) <= 1e-10 * max(
                1.0,
                abs(float(self.graph_eigenvalues[boundary])),
                abs(float(self.graph_eigenvalues[boundary - 1])),
            ):
                raise ValueError("a graph band splits a tied eigenspace")
        if (
            int(self.pack_assignments.min()) < 0
            or int(self.pack_assignments.max()) >= pack_count
        ):
            raise ValueError("pack assignments are outside the pack range")
        counts = torch.bincount(
            self.pack_assignments,
            minlength=pack_count,
        )
        if bool((counts <= 0).any()):
            raise ValueError("every graph-organized pack must be nonempty")
        expected_permutation = _pack_permutation(self.pack_assignments)
        expected_offsets = _pack_offsets(
            self.pack_assignments,
            pack_count,
        )
        if (
            not torch.equal(self.component_permutation, expected_permutation)
            or not torch.equal(self.pack_offsets, expected_offsets)
        ):
            raise ValueError("pack permutation or offsets are noncanonical")
        unpacked_basis = torch.empty_like(self.source_basis)
        unpacked_basis[:, self.component_permutation] = self.source_basis
        expected_energy = (
            self.graph_eigenvectors.T @ unpacked_basis
        ).square()
        if not _close(self.component_graph_energy, expected_energy):
            raise ValueError("component graph signatures differ")
        expected_band_mass = torch.stack(
            tuple(
                self.component_graph_energy[start:stop].sum(dim=0)
                for start, stop in zip(
                    self.frequency_band_boundaries[:-1],
                    self.frequency_band_boundaries[1:],
                    strict=True,
                )
            )
        )
        if not _close(self.band_mass, expected_band_mass):
            raise ValueError("graph band masses differ")
        if not _close(
            self.component_graph_energy.sum(dim=0),
            torch.ones(source_rank, dtype=torch.float64),
        ):
            raise ValueError("component graph energy is not normalized")
        if self.organization_kind in {
            "signed_gfa_dyadic",
            "magnitude_gfa_dyadic",
            "signed_row_permutation_dyadic",
        } and not torch.equal(
            self.pack_assignments,
            self.band_mass.argmax(dim=0),
        ):
            raise ValueError("graph pack assignments differ from band mass")
        if self.organization_kind in {
            "singular_contiguous_control",
            "random_size_matched_control",
        } and not torch.equal(
            self.pack_assignments,
            _size_matched_assignments(
                tuple(int(value) for value in counts),
                kind=self.organization_kind,
                seed=self.organization_seed,
            ),
        ):
            raise ValueError("control pack assignments are noncanonical")
        if not torch.equal(
            torch.sort(self.component_permutation).values,
            torch.arange(source_rank, dtype=torch.int64),
        ):
            raise ValueError("component_permutation is not a permutation")
        for values in (
            self.source_singular_values,
            self.target_singular_values,
        ):
            if bool((values < 0.0).any()) or bool(
                (values[1:] > values[:-1]).any()
            ):
                raise ValueError("singular values are invalid")
        if (
            self.source_singular_values.numel() < source_rank
            or self.target_singular_values.numel() < target_modes
        ):
            raise ValueError("retained ranks exceed the fitted spectra")
        total = _finite_nonnegative(
            self.weighted_total_energy,
            label="weighted_total_energy",
        )
        retained = _finite_nonnegative(
            self.weighted_retained_energy,
            label="weighted_retained_energy",
        )
        relative = _finite_nonnegative(
            self.weighted_relative_error,
            label="weighted_relative_error",
        )
        source_parseval = _finite_nonnegative(
            self.source_parseval_relative_error,
            label="source_parseval_relative_error",
        )
        target_parseval = _finite_nonnegative(
            self.target_parseval_relative_error,
            label="target_parseval_relative_error",
        )
        for field, value in (
            ("weighted_total_energy", total),
            ("weighted_retained_energy", retained),
            ("weighted_relative_error", relative),
            ("source_parseval_relative_error", source_parseval),
            ("target_parseval_relative_error", target_parseval),
        ):
            object.__setattr__(self, field, value)
        scale = max(total, 1.0)
        expected_relative = (
            math.sqrt(max(total - retained, 0.0) / total)
            if total > torch.finfo(torch.float64).eps
            else 0.0
        )
        if (
            retained > total + 2e-10 * scale
            or not _close(relative, expected_relative)
            or not _close(
                float(self.knot_cores.square().sum()),
                retained,
                scale=scale,
            )
            or not _close(
                float(self.source_singular_values.square().sum()),
                total,
                scale=scale,
            )
            or not _close(
                float(self.target_singular_values.square().sum()),
                total,
                scale=scale,
            )
            or not _close(
                float(
                    self.source_singular_values[:source_rank]
                    .square()
                    .sum()
                ),
                retained,
                scale=scale,
            )
            or source_parseval > 1e-10
            or target_parseval > 1e-10
        ):
            raise ValueError("graph-organized energy accounting differs")
        expected_bounds = torch.empty_like(self.core_operator_norm_bounds)
        for knot in range(self.knot_count):
            for lag in range(self.lag_count):
                for pack in range(self.pack_count):
                    start = int(self.pack_offsets[pack])
                    stop = int(self.pack_offsets[pack + 1])
                    expected_bounds[knot, lag, pack] = (
                        torch.linalg.matrix_norm(
                            self.knot_cores[knot, lag, start:stop],
                            ord=2,
                        )
                    )
        bound_scale = max(float(expected_bounds.max()), 1.0)
        if (
            bool(
                (
                    self.core_operator_norm_bounds
                    < expected_bounds
                ).any()
            )
            or not bool(
                torch.allclose(
                    self.core_operator_norm_bounds,
                    expected_bounds,
                    atol=2e-11 * bound_scale,
                    rtol=2e-11,
                )
            )
        ):
            raise ValueError("pack core operator-norm bounds differ")
        if (
            self.band_semantics != _BAND_SEMANTICS
            or self.execution_semantics != _EXECUTION_SEMANTICS
            or self.router_bound_semantics != _ROUTER_BOUND_SEMANTICS
            or self.heldout_origins_used_for_organization is not False
            or self.conditional_compute_claim is not False
            or self.latency_claim is not False
            or self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("graph-organized plan provenance drifted")
        if self.fft_length < self.lag_count:
            raise ValueError("fft_length cannot truncate causal lags")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="artifact_sha256",
                )
                != computed
            ):
                raise ValueError("graph-organized plan hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def source_modes(self) -> int:
        return int(self.source_basis.shape[0])

    @property
    def source_rank(self) -> int:
        return int(self.source_basis.shape[1])

    @property
    def target_modes(self) -> int:
        return int(self.knot_cores.shape[3])

    @property
    def knot_count(self) -> int:
        return len(self.fit_knot_origins)

    @property
    def lag_count(self) -> int:
        return int(self.knot_cores.shape[1])

    @property
    def pack_count(self) -> int:
        return int(self.pack_offsets.numel() - 1)

    @property
    def pack_counts(self) -> tuple[int, ...]:
        return tuple(
            int(self.pack_offsets[index + 1] - self.pack_offsets[index])
            for index in range(self.pack_count)
        )

    @property
    def stored_coefficient_count(self) -> int:
        """Deployable float coefficients, excluding analysis diagnostics."""

        return (
            self.source_basis.numel()
            + self.knot_cores.numel()
            + self.core_operator_norm_bounds.numel()
        )

    @property
    def prepared_float_scalar_count(self) -> int:
        return self.stored_coefficient_count + self.source_scales.numel()

    @property
    def prepared_integer_value_count(self) -> int:
        return self.pack_offsets.numel() + self.knot_count

    @property
    def dense_fit_knot_coefficient_count(self) -> int:
        return (
            self.knot_count
            * self.lag_count
            * self.source_modes
            * self.target_modes
        )

    @property
    def coefficient_fraction_of_dense_fit_knots(self) -> float:
        return (
            self.stored_coefficient_count
            / self.dense_fit_knot_coefficient_count
        )

    @property
    def analysis_only_float_scalar_count(self) -> int:
        return sum(
            getattr(self, field).numel()
            for field in (
                "source_singular_values",
                "target_singular_values",
                "graph_eigenvalues",
                "graph_eigenvectors",
                "component_graph_energy",
                "band_mass",
            )
        )

    @property
    def artifact_float_scalar_count(self) -> int:
        return sum(getattr(self, field).numel() for field in _FLOAT_TENSORS)

    @property
    def artifact_integer_scalar_count(self) -> int:
        return sum(getattr(self, field).numel() for field in _INT_TENSORS)

    @property
    def artifact_tensor_bytes(self) -> int:
        return 8 * (
            self.artifact_float_scalar_count
            + self.artifact_integer_scalar_count
        )

    def _interpolation(self, origin: int) -> tuple[int, int, float]:
        if type(origin) is not int or origin < 0:
            raise ValueError("origin must be a nonnegative integer")
        knots = self.fit_knot_origins
        if origin < knots[0] or origin > knots[-1]:
            raise ValueError("origin lies outside the fitted knot interval")
        right = min(max(bisect_right(knots, origin), 1), len(knots) - 1)
        left = right - 1
        alpha = (origin - knots[left]) / (knots[right] - knots[left])
        return left, right, float(alpha)

    def core_at_origin(self, origin: int) -> Tensor:
        self.validate_integrity()
        left, right, alpha = self._interpolation(origin)
        return (
            self.knot_cores[left] * (1.0 - alpha)
            + self.knot_cores[right] * alpha
        ).contiguous()

    def norm_bounds_at_origin(self, origin: int) -> Tensor:
        self.validate_integrity()
        left, right, alpha = self._interpolation(origin)
        return (
            self.core_operator_norm_bounds[left] * (1.0 - alpha)
            + self.core_operator_norm_bounds[right] * alpha
        ).contiguous()

    def weighted_kernel_at_origin(self, origin: int) -> Tensor:
        return torch.einsum(
            "sr,lrt->slt",
            self.source_basis,
            self.core_at_origin(origin),
        ).contiguous()

    def standardized_output_at_origin(
        self,
        standardized_source: Tensor,
        *,
        origin: int,
        pack_mask: Tensor | None = None,
    ) -> Tensor:
        self.validate_integrity()
        source = _float_tensor(
            standardized_source,
            label="standardized_source",
            ndim=2,
        )
        if source.shape[1] != self.source_modes:
            raise ValueError("standardized source width differs")
        latent = source @ self.source_basis
        if pack_mask is not None:
            if (
                not isinstance(pack_mask, Tensor)
                or pack_mask.dtype != torch.bool
                or pack_mask.device.type != "cpu"
                or pack_mask.shape != (source.shape[0], self.pack_count)
            ):
                raise ValueError("pack_mask must have shape [rows, packs]")
            rank_mask = torch.zeros(
                (source.shape[0], self.source_rank),
                dtype=torch.bool,
            )
            for pack in range(self.pack_count):
                start = int(self.pack_offsets[pack])
                stop = int(self.pack_offsets[pack + 1])
                rank_mask[:, start:stop] = pack_mask[:, pack : pack + 1]
            latent = torch.where(rank_mask, latent, torch.zeros_like(latent))
        return torch.einsum(
            "nr,lrt->nlt",
            latent,
            self.core_at_origin(origin),
        ).contiguous()

    def bound_mass_route_mask(
        self,
        standardized_source: Tensor,
        *,
        origin: int,
        retained_bound_fraction: float,
    ) -> tuple[Tensor, Tensor]:
        self.validate_integrity()
        source = _float_tensor(
            standardized_source,
            label="standardized_source",
            ndim=2,
        )
        if source.shape[1] != self.source_modes:
            raise ValueError("standardized source width differs")
        if (
            isinstance(retained_bound_fraction, bool)
            or not isinstance(retained_bound_fraction, (int, float))
            or not math.isfinite(float(retained_bound_fraction))
            or not 0.0 < float(retained_bound_fraction) <= 1.0
        ):
            raise ValueError("retained_bound_fraction must lie in (0, 1]")
        latent = source @ self.source_basis
        lag_bounds = self.norm_bounds_at_origin(origin)
        scores = torch.empty(
            (source.shape[0], self.pack_count),
            dtype=torch.float64,
        )
        for pack in range(self.pack_count):
            start = int(self.pack_offsets[pack])
            stop = int(self.pack_offsets[pack + 1])
            scores[:, pack] = (
                torch.linalg.vector_norm(latent[:, start:stop], dim=1)
                * torch.linalg.vector_norm(lag_bounds[:, pack])
            )
        mask = torch.zeros_like(scores, dtype=torch.bool)
        fraction = float(retained_bound_fraction)
        for row in range(source.shape[0]):
            if fraction == 1.0:
                mask[row] = True
                continue
            total = float(scores[row].sum())
            if total == 0.0:
                continue
            ordered = sorted(
                range(self.pack_count),
                key=lambda pack: (-float(scores[row, pack]), pack),
            )
            retained = 0.0
            for pack in ordered:
                mask[row, pack] = True
                retained += float(scores[row, pack])
                if retained >= fraction * total:
                    break
        return mask.contiguous(), scores.contiguous()

    def omitted_source_response_bound(
        self,
        standardized_source: Tensor,
        *,
        origin: int,
        pack_mask: Tensor,
    ) -> Tensor:
        """Bound each source row's omitted full ``[lag, target]`` response."""

        self.validate_integrity()
        source = _float_tensor(
            standardized_source,
            label="standardized_source",
            ndim=2,
        )
        if (
            not isinstance(pack_mask, Tensor)
            or pack_mask.dtype != torch.bool
            or pack_mask.device.type != "cpu"
            or pack_mask.shape != (source.shape[0], self.pack_count)
        ):
            raise ValueError("pack_mask must match source rows and pack count")
        latent = source @ self.source_basis
        lag_bounds = self.norm_bounds_at_origin(origin)
        scores = torch.empty(
            (source.shape[0], self.pack_count),
            dtype=torch.float64,
        )
        for pack in range(self.pack_count):
            start = int(self.pack_offsets[pack])
            stop = int(self.pack_offsets[pack + 1])
            scores[:, pack] = (
                torch.linalg.vector_norm(latent[:, start:stop], dim=1)
                * torch.linalg.vector_norm(lag_bounds[:, pack])
            )
        return torch.where(
            pack_mask,
            torch.zeros_like(scores),
            scores,
        ).sum(dim=1)

    def omitted_same_origin_sequence_response_bound(
        self,
        standardized_source: Tensor,
        *,
        origin: int,
        pack_mask: Tensor,
    ) -> Tensor:
        """Bound shifted source responses when every row shares one origin."""

        return self.omitted_source_response_bound(
            standardized_source,
            origin=origin,
            pack_mask=pack_mask,
        ).sum()

    def prepare(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "PreparedGraphOrganizedSVD":
        return PreparedGraphOrganizedSVD(self, device=device, dtype=dtype)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "response_binding_sha256": self.response_binding_sha256,
            "source_plan_artifact_sha256": (
                self.source_plan_artifact_sha256
            ),
            "graph_basis_artifact_sha256": (
                self.graph_basis_artifact_sha256
            ),
            "fit_weighted_kernels_sha256": (
                self.fit_weighted_kernels_sha256
            ),
            "fit_knot_origins": self.fit_knot_origins,
            "fft_length": self.fft_length,
            "organization_kind": self.organization_kind,
            "organization_seed": self.organization_seed,
            "frequency_band_boundaries": self.frequency_band_boundaries,
            "source_modes": self.source_modes,
            "source_rank": self.source_rank,
            "target_modes": self.target_modes,
            "knot_count": self.knot_count,
            "lag_count": self.lag_count,
            "pack_count": self.pack_count,
            "pack_counts": self.pack_counts,
            "tensor_sha256s": {
                field: _tensor_sha256(getattr(self, field))
                for field in (*_FLOAT_TENSORS, *_INT_TENSORS)
            },
            "tensor_shapes": {
                field: tuple(int(width) for width in getattr(self, field).shape)
                for field in (*_FLOAT_TENSORS, *_INT_TENSORS)
            },
            "weighted_total_energy": self.weighted_total_energy,
            "weighted_retained_energy": self.weighted_retained_energy,
            "weighted_relative_error": self.weighted_relative_error,
            "source_parseval_relative_error": (
                self.source_parseval_relative_error
            ),
            "target_parseval_relative_error": (
                self.target_parseval_relative_error
            ),
            "stored_coefficient_count": self.stored_coefficient_count,
            "prepared_float_scalar_count": (
                self.prepared_float_scalar_count
            ),
            "prepared_integer_value_count": (
                self.prepared_integer_value_count
            ),
            "dense_fit_knot_coefficient_count": (
                self.dense_fit_knot_coefficient_count
            ),
            "coefficient_fraction_of_dense_fit_knots": (
                self.coefficient_fraction_of_dense_fit_knots
            ),
            "analysis_only_float_scalar_count": (
                self.analysis_only_float_scalar_count
            ),
            "artifact_float_scalar_count": (
                self.artifact_float_scalar_count
            ),
            "artifact_integer_scalar_count": (
                self.artifact_integer_scalar_count
            ),
            "artifact_tensor_bytes": self.artifact_tensor_bytes,
            "band_semantics": self.band_semantics,
            "execution_semantics": self.execution_semantics,
            "router_bound_semantics": self.router_bound_semantics,
            "heldout_origins_used_for_organization": (
                self.heldout_origins_used_for_organization
            ),
            "conditional_compute_claim": self.conditional_compute_claim,
            "latency_claim": self.latency_claim,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._hash_payload())

    def validate_integrity(self) -> None:
        for field in (*_FLOAT_TENSORS, *_INT_TENSORS):
            value = getattr(self, field)
            if (
                value.device.type != "cpu"
                or not value.is_contiguous()
                or (
                    field in _FLOAT_TENSORS
                    and (
                        value.dtype != torch.float64
                        or not bool(torch.isfinite(value).all())
                    )
                )
                or (
                    field in _INT_TENSORS
                    and value.dtype != torch.int64
                )
            ):
                raise ValueError(f"{field} drifted from canonical storage")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("graph-organized plan hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        top_mass = torch.topk(
            self.component_graph_energy,
            k=min(16, self.source_modes),
            dim=0,
        ).values.cumsum(dim=0)
        participation = 1.0 / self.component_graph_energy.square().sum(dim=0)
        return {
            **self._hash_payload(),
            "mean_top4_graph_frequency_mass": float(
                top_mass[min(3, top_mass.shape[0] - 1)].mean()
            ),
            "mean_top8_graph_frequency_mass": float(
                top_mass[min(7, top_mass.shape[0] - 1)].mean()
            ),
            "mean_top16_graph_frequency_mass": float(
                top_mass[min(15, top_mass.shape[0] - 1)].mean()
            ),
            "mean_graph_frequency_participation_ratio": float(
                participation.mean()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            **{
                field: getattr(self, field).clone()
                for field in (*_FLOAT_TENSORS, *_INT_TENSORS)
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "GraphOrganizedSVDPlan":
        if not isinstance(raw, Mapping):
            raise TypeError("graph-organized state must be a mapping")
        scalar_keys = {
            "artifact_kind",
            "format_version",
            "response_binding_sha256",
            "source_plan_artifact_sha256",
            "graph_basis_artifact_sha256",
            "fit_weighted_kernels_sha256",
            "fit_knot_origins",
            "fft_length",
            "organization_kind",
            "organization_seed",
            "frequency_band_boundaries",
            "source_modes",
            "source_rank",
            "target_modes",
            "knot_count",
            "lag_count",
            "pack_count",
            "pack_counts",
            "tensor_sha256s",
            "tensor_shapes",
            "weighted_total_energy",
            "weighted_retained_energy",
            "weighted_relative_error",
            "source_parseval_relative_error",
            "target_parseval_relative_error",
            "stored_coefficient_count",
            "prepared_float_scalar_count",
            "prepared_integer_value_count",
            "dense_fit_knot_coefficient_count",
            "coefficient_fraction_of_dense_fit_knots",
            "analysis_only_float_scalar_count",
            "artifact_float_scalar_count",
            "artifact_integer_scalar_count",
            "artifact_tensor_bytes",
            "band_semantics",
            "execution_semantics",
            "router_bound_semantics",
            "heldout_origins_used_for_organization",
            "conditional_compute_claim",
            "latency_claim",
            "artifact_sha256",
        }
        expected = {*scalar_keys, *_FLOAT_TENSORS, *_INT_TENSORS}
        if set(raw) != expected:
            raise ValueError("graph-organized state fields differ")
        tensor_hashes = raw["tensor_sha256s"]
        tensor_shapes = raw["tensor_shapes"]
        tensor_fields = (*_FLOAT_TENSORS, *_INT_TENSORS)
        if (
            not isinstance(tensor_hashes, Mapping)
            or set(tensor_hashes) != set(tensor_fields)
            or not isinstance(tensor_shapes, Mapping)
            or set(tensor_shapes) != set(tensor_fields)
        ):
            raise ValueError("graph-organized tensor declarations differ")
        tensors: dict[str, Tensor] = {}
        for field in tensor_fields:
            value = raw[field]
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or (
                    field in _FLOAT_TENSORS
                    and (
                        value.dtype != torch.float64
                        or not bool(torch.isfinite(value).all())
                    )
                )
                or (
                    field in _INT_TENSORS
                    and value.dtype != torch.int64
                )
                or _tensor_sha256(value) != tensor_hashes[field]
                or tuple(value.shape) != tuple(tensor_shapes[field])
            ):
                raise ValueError(f"serialized {field} is invalid")
            tensors[field] = value
        result = cls(
            response_binding_sha256=raw[
                "response_binding_sha256"
            ],  # type: ignore[arg-type]
            source_plan_artifact_sha256=raw[
                "source_plan_artifact_sha256"
            ],  # type: ignore[arg-type]
            graph_basis_artifact_sha256=raw[
                "graph_basis_artifact_sha256"
            ],  # type: ignore[arg-type]
            fit_weighted_kernels_sha256=raw[
                "fit_weighted_kernels_sha256"
            ],  # type: ignore[arg-type]
            fit_knot_origins=tuple(raw["fit_knot_origins"]),  # type: ignore[arg-type]
            fft_length=raw["fft_length"],  # type: ignore[arg-type]
            organization_kind=raw["organization_kind"],  # type: ignore[arg-type]
            organization_seed=raw["organization_seed"],  # type: ignore[arg-type]
            frequency_band_boundaries=tuple(
                raw["frequency_band_boundaries"]  # type: ignore[arg-type]
            ),
            **tensors,
            weighted_total_energy=raw[
                "weighted_total_energy"
            ],  # type: ignore[arg-type]
            weighted_retained_energy=raw[
                "weighted_retained_energy"
            ],  # type: ignore[arg-type]
            weighted_relative_error=raw[
                "weighted_relative_error"
            ],  # type: ignore[arg-type]
            source_parseval_relative_error=raw[
                "source_parseval_relative_error"
            ],  # type: ignore[arg-type]
            target_parseval_relative_error=raw[
                "target_parseval_relative_error"
            ],  # type: ignore[arg-type]
            band_semantics=raw["band_semantics"],  # type: ignore[arg-type]
            execution_semantics=raw[
                "execution_semantics"
            ],  # type: ignore[arg-type]
            router_bound_semantics=raw[
                "router_bound_semantics"
            ],  # type: ignore[arg-type]
            heldout_origins_used_for_organization=raw[
                "heldout_origins_used_for_organization"
            ],  # type: ignore[arg-type]
            conditional_compute_claim=raw[
                "conditional_compute_claim"
            ],  # type: ignore[arg-type]
            latency_claim=raw["latency_claim"],  # type: ignore[arg-type]
            artifact_sha256=raw["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=raw["artifact_kind"],  # type: ignore[arg-type]
            format_version=raw["format_version"],  # type: ignore[arg-type]
        )
        for field in (
            "source_modes",
            "source_rank",
            "target_modes",
            "knot_count",
            "lag_count",
            "pack_count",
            "pack_counts",
            "stored_coefficient_count",
            "prepared_float_scalar_count",
            "prepared_integer_value_count",
            "dense_fit_knot_coefficient_count",
            "coefficient_fraction_of_dense_fit_knots",
            "analysis_only_float_scalar_count",
            "artifact_float_scalar_count",
            "artifact_integer_scalar_count",
            "artifact_tensor_bytes",
        ):
            if raw[field] != getattr(result, field):
                raise ValueError(f"serialized {field} differs")
        expected_payload = result._hash_payload()
        if (
            raw["tensor_sha256s"] != expected_payload["tensor_sha256s"]
            or raw["tensor_shapes"] != expected_payload["tensor_shapes"]
        ):
            raise ValueError("serialized tensor declarations differ")
        return result


class PreparedGraphOrganizedSVD(nn.Module):
    """Validate-once runtime for all-on or explicitly routed SVD packs."""

    def __init__(
        self,
        plan: GraphOrganizedSVDPlan,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(plan, GraphOrganizedSVDPlan):
            raise TypeError("plan must be a GraphOrganizedSVDPlan")
        if dtype not in _RUNTIME_DTYPES:
            raise ValueError("dtype must be a supported floating Torch dtype")
        runtime_device = torch.device(device)
        plan.validate_integrity()
        self.plan_sha256 = plan.artifact_sha256
        self.fit_knot_origins = plan.fit_knot_origins
        self.source_modes = plan.source_modes
        self.source_rank = plan.source_rank
        self.target_modes = plan.target_modes
        self.pack_count = plan.pack_count
        self.lag_count = plan.lag_count
        self.register_buffer(
            "source_scales",
            plan.source_scales.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "source_basis",
            plan.source_basis.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "knot_cores",
            plan.knot_cores.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "core_operator_norm_bounds",
            plan.core_operator_norm_bounds.to(
                device=runtime_device,
                dtype=dtype,
            ).contiguous().clone(),
        )
        self.register_buffer(
            "pack_offsets",
            plan.pack_offsets.to(device=runtime_device).clone(),
        )

    @property
    def device(self) -> torch.device:
        return self.source_basis.device

    @property
    def dtype(self) -> torch.dtype:
        return self.source_basis.dtype

    @property
    def learned_parameter_count(self) -> int:
        return 0

    @property
    def stored_coefficient_count(self) -> int:
        return (
            self.source_basis.numel()
            + self.knot_cores.numel()
            + self.core_operator_norm_bounds.numel()
        )

    def _source(
        self,
        source_modes: Tensor,
    ) -> tuple[Tensor, bool]:
        if (
            not isinstance(source_modes, Tensor)
            or source_modes.ndim not in (2, 3)
            or any(int(width) <= 0 for width in source_modes.shape)
            or source_modes.shape[-1] != self.source_modes
            or source_modes.dtype != self.dtype
            or source_modes.device != self.device
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError(
                "source_modes must be finite [S, source] or "
                "[B, S, source] runtime data"
            )
        squeeze = source_modes.ndim == 2
        return (source_modes.unsqueeze(0) if squeeze else source_modes), squeeze

    def _grid(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        batched, squeeze = self._source(source_modes)
        positions, targets, sources = _canonical_runtime_grid(
            logical_positions,
            valid_mask,
            source_mask,
            batch_size=int(batched.shape[0]),
            sequence_length=int(batched.shape[1]),
            device=self.device,
            minimum_origin=self.fit_knot_origins[0],
            maximum_origin=self.fit_knot_origins[-1],
        )
        return batched, positions, targets, sources, squeeze

    def _mask(
        self,
        pack_mask: Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        active_source_mask: Tensor,
    ) -> Tensor:
        if pack_mask is None:
            return active_source_mask.unsqueeze(-1).expand(
                batch_size,
                sequence_length,
                self.pack_count,
            ).clone()
        if (
            not isinstance(pack_mask, Tensor)
            or pack_mask.dtype != torch.bool
            or pack_mask.device != self.device
        ):
            raise TypeError("pack_mask must be a boolean runtime Tensor")
        if pack_mask.ndim == 2 and batch_size == 1:
            pack_mask = pack_mask.unsqueeze(0)
        if pack_mask.shape != (
            batch_size,
            sequence_length,
            self.pack_count,
        ):
            raise ValueError("pack_mask must match [B, S, pack_count]")
        if bool((pack_mask & ~active_source_mask.unsqueeze(-1)).any()):
            raise ValueError("inactive source rows cannot activate packs")
        return pack_mask

    def _interpolation(self, origin: int) -> tuple[int, int, float]:
        knots = self.fit_knot_origins
        right = min(max(bisect_right(knots, origin), 1), len(knots) - 1)
        left = right - 1
        alpha = (origin - knots[left]) / (knots[right] - knots[left])
        return left, right, float(alpha)

    def _pack_core(self, origin: int, pack: int) -> Tensor:
        left, right, alpha = self._interpolation(origin)
        start = int(self.pack_offsets[pack])
        stop = int(self.pack_offsets[pack + 1])
        return (
            self.knot_cores[left, :, start:stop] * (1.0 - alpha)
            + self.knot_cores[right, :, start:stop] * alpha
        )

    def _norm_bounds(self, origin: int) -> Tensor:
        left, right, alpha = self._interpolation(origin)
        return (
            self.core_operator_norm_bounds[left] * (1.0 - alpha)
            + self.core_operator_norm_bounds[right] * alpha
        )

    def _latent(self, batched: Tensor, sources: Tensor) -> Tensor:
        features = torch.zeros_like(batched)
        features[sources] = batched[sources] / self.source_scales
        latent = batched.new_zeros(
            (batched.shape[0], batched.shape[1], self.source_rank)
        )
        latent[sources] = features[sources] @ self.source_basis
        return latent

    def _route_from_latent(
        self,
        latent: Tensor,
        positions: Tensor,
        sources: Tensor,
        *,
        retained_bound_fraction: float,
    ) -> tuple[Tensor, Tensor]:
        if self.dtype != torch.float64:
            raise ValueError(
                "certified bound-mass routing currently requires float64"
            )
        if (
            isinstance(retained_bound_fraction, bool)
            or not isinstance(retained_bound_fraction, (int, float))
            or not math.isfinite(float(retained_bound_fraction))
            or not 0.0 < float(retained_bound_fraction) <= 1.0
        ):
            raise ValueError("retained_bound_fraction must lie in (0, 1]")
        mask = torch.zeros(
            (*sources.shape, self.pack_count),
            dtype=torch.bool,
            device=self.device,
        )
        scores = torch.zeros(
            (*sources.shape, self.pack_count),
            dtype=self.dtype,
            device=self.device,
        )
        fraction = float(retained_bound_fraction)
        for batch in range(latent.shape[0]):
            for source_index in torch.nonzero(
                sources[batch],
                as_tuple=False,
            ).flatten().tolist():
                bounds = self._norm_bounds(
                    int(positions[batch, source_index])
                )
                for pack in range(self.pack_count):
                    start = int(self.pack_offsets[pack])
                    stop = int(self.pack_offsets[pack + 1])
                    scores[batch, source_index, pack] = (
                        torch.linalg.vector_norm(
                            latent[batch, source_index, start:stop]
                        )
                        * torch.linalg.vector_norm(bounds[:, pack])
                    )
                if fraction == 1.0:
                    mask[batch, source_index] = True
                    continue
                total = float(scores[batch, source_index].sum())
                if total == 0.0:
                    continue
                ordering = torch.argsort(
                    scores[batch, source_index],
                    descending=True,
                    stable=True,
                ).tolist()
                retained = 0.0
                for pack in ordering:
                    mask[batch, source_index, pack] = True
                    retained += float(scores[batch, source_index, pack])
                    if retained >= fraction * total:
                        break
        return mask.contiguous(), scores.contiguous()

    def _execute_grid(
        self,
        batched: Tensor,
        positions: Tensor,
        targets: Tensor,
        sources: Tensor,
        mask: Tensor,
        latent: Tensor,
        *,
        squeeze: bool,
    ) -> Tensor:
        result = batched.new_zeros(
            (batched.shape[0], batched.shape[1], self.target_modes)
        )
        core_cache: dict[tuple[int, int], Tensor] = {}
        for batch in range(batched.shape[0]):
            source_indices = torch.nonzero(
                sources[batch],
                as_tuple=False,
            ).flatten().tolist()
            by_position = {
                int(positions[batch, index]): int(index)
                for index in source_indices
            }
            active_packs = {
                source_index: torch.nonzero(
                    mask[batch, source_index],
                    as_tuple=False,
                ).flatten().tolist()
                for source_index in source_indices
            }
            for source_index in source_indices:
                origin = int(positions[batch, source_index])
                for pack in active_packs[source_index]:
                    key = (origin, pack)
                    if key not in core_cache:
                        core_cache[key] = self._pack_core(origin, pack)
            for target_index in torch.nonzero(
                targets[batch],
                as_tuple=False,
            ).flatten().tolist():
                target_position = int(positions[batch, target_index])
                value = result[batch, target_index]
                for lag in range(self.lag_count):
                    source_index = by_position.get(target_position - lag)
                    if source_index is None:
                        continue
                    for pack in active_packs[source_index]:
                        start = int(self.pack_offsets[pack])
                        stop = int(self.pack_offsets[pack + 1])
                        value = value + (
                            latent[batch, source_index, start:stop]
                            @ core_cache[
                                (
                                    int(positions[batch, source_index]),
                                    pack,
                                )
                            ][lag]
                        )
                result[batch, target_index] = value
        return result[0] if squeeze else result

    def _execute(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None,
        pack_mask: Tensor | None,
    ) -> Tensor:
        batched, positions, targets, sources, squeeze = self._grid(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        mask = self._mask(
            pack_mask,
            batch_size=int(batched.shape[0]),
            sequence_length=int(batched.shape[1]),
            active_source_mask=sources,
        )
        return self._execute_grid(
            batched,
            positions,
            targets,
            sources,
            mask,
            self._latent(batched, sources),
            squeeze=squeeze,
        )

    def forward(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        return self._execute(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
            pack_mask=None,
        )

    def forward_with_pack_mask(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        pack_mask: Tensor,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        return self._execute(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
            pack_mask=pack_mask,
        )

    def forward_bound_routed(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        retained_bound_fraction: float,
        source_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Route and execute once, reusing the full source projection."""

        if self.dtype != torch.float64:
            raise ValueError(
                "certified bound-mass routing currently requires float64"
            )
        batched, positions, targets, sources, squeeze = self._grid(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        latent = self._latent(batched, sources)
        mask, scores = self._route_from_latent(
            latent,
            positions,
            sources,
            retained_bound_fraction=retained_bound_fraction,
        )
        result = self._execute_grid(
            batched,
            positions,
            targets,
            sources,
            mask,
            latent,
            squeeze=squeeze,
        )
        return (
            result,
            mask[0] if squeeze else mask,
            scores[0] if squeeze else scores,
        )

    def execution_accounting(
        self,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        pack_mask: Tensor | None = None,
        source_mask: Tensor | None = None,
        router_evaluated: bool = False,
    ) -> GraphOrganizedSVDExecutionAccounting:
        if type(router_evaluated) is not bool:
            raise TypeError("router_evaluated must be a bool")
        if router_evaluated and pack_mask is None:
            raise ValueError("routed accounting requires an explicit pack mask")
        if router_evaluated and self.dtype != torch.float64:
            raise ValueError(
                "certified routed accounting currently requires float64"
            )
        if not isinstance(logical_positions, Tensor):
            raise TypeError("logical_positions must be a Tensor")
        if logical_positions.ndim not in (1, 2):
            raise ValueError("logical_positions must have shape [S] or [B, S]")
        sequence_length = logical_positions.shape[-1]
        batch_sizes = (
            [int(logical_positions.shape[0])]
            if logical_positions.ndim == 2
            else []
        )
        for value in (valid_mask, source_mask):
            if isinstance(value, Tensor) and value.ndim == 2:
                batch_sizes.append(int(value.shape[0]))
        if isinstance(pack_mask, Tensor) and pack_mask.ndim == 3:
            batch_sizes.append(int(pack_mask.shape[0]))
        if len(set(batch_sizes)) > 1:
            raise ValueError("batched runtime grids have different batch sizes")
        batch_size = batch_sizes[0] if batch_sizes else 1
        dummy = torch.zeros(
            (batch_size, sequence_length, self.source_modes),
            dtype=self.dtype,
            device=self.device,
        )
        _, positions, targets, sources, _ = self._grid(
            dummy,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        mask = self._mask(
            pack_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            active_source_mask=sources,
        )
        pack_ranks = (
            self.pack_offsets[1:] - self.pack_offsets[:-1]
        ).to(dtype=torch.int64)
        active_ranks = (
            mask.to(dtype=torch.int64) * pack_ranks.view(1, 1, -1)
        ).sum(dim=2)
        admitted_pairs = 0
        admitted_rank_pairs = 0
        admitted_pack_pairs = 0
        interpolated_origin_packs: set[tuple[int, int]] = set()
        for batch in range(batch_size):
            target_positions = positions[batch][targets[batch]]
            source_positions = positions[batch][sources[batch]]
            source_ranks = active_ranks[batch][sources[batch]]
            source_packs = mask[batch][sources[batch]].sum(
                dim=1,
                dtype=torch.int64,
            )
            for source_index in torch.nonzero(
                sources[batch],
                as_tuple=False,
            ).flatten().tolist():
                origin = int(positions[batch, source_index])
                for pack in torch.nonzero(
                    mask[batch, source_index],
                    as_tuple=False,
                ).flatten().tolist():
                    interpolated_origin_packs.add((origin, pack))
            lags = (
                target_positions.unsqueeze(1)
                - source_positions.unsqueeze(0)
            )
            admitted = (lags >= 0) & (lags < self.lag_count)
            admitted_pairs += int(admitted.sum())
            admitted_rank_pairs += int(
                (
                    admitted.to(dtype=torch.int64)
                    * source_ranks.unsqueeze(0)
                ).sum()
            )
            admitted_pack_pairs += int(
                (
                    admitted.to(dtype=torch.int64)
                    * source_packs.unsqueeze(0)
                ).sum()
            )
        return GraphOrganizedSVDExecutionAccounting(
            batch_size=batch_size,
            sequence_length=sequence_length,
            valid_source_rows=int(sources.sum()),
            valid_target_rows=int(targets.sum()),
            admitted_causal_pairs=admitted_pairs,
            active_pack_instances=int(mask.sum()),
            active_rank_instances=int(active_ranks.sum()),
            interpolated_active_rank_instances=sum(
                int(pack_ranks[pack])
                for _origin, pack in interpolated_origin_packs
            ),
            admitted_active_rank_pairs=admitted_rank_pairs,
            admitted_active_pack_pairs=admitted_pack_pairs,
            source_modes=self.source_modes,
            target_modes=self.target_modes,
            source_rank=self.source_rank,
            pack_count=self.pack_count,
            lag_count=self.lag_count,
            router_evaluated=router_evaluated,
        )


def organize_conditional_svd_with_graph(
    base_plan: ConditionalSpectralGeneratorPlan,
    graph_basis: FitOnlyGraphSourceBasis,
    *,
    organization_kind: OrganizationKind = "signed_gfa_dyadic",
    frequency_band_boundaries: Sequence[int] | None = None,
    organization_seed: int = 0,
    matched_pack_counts: Sequence[int] | None = None,
) -> GraphOrganizedSVDPlan:
    """Fold a full-target SVD plan and organize its generators into packs."""

    if not isinstance(base_plan, ConditionalSpectralGeneratorPlan):
        raise TypeError("base_plan must be a conditional spectral plan")
    if not isinstance(graph_basis, FitOnlyGraphSourceBasis):
        raise TypeError("graph_basis must be a FitOnlyGraphSourceBasis")
    base_plan.validate_integrity()
    graph_basis.validate_integrity()
    if (
        base_plan.input_transform != "standardized_linear"
        or base_plan.target_rank != base_plan.target_modes
        or base_plan.factorization_semantics != _FACTORIZATION
        or base_plan.rank_semantics != _RANK_SEMANTICS
    ):
        raise ValueError(
            "organization requires a global-SVD linear plan with a full "
            "target basis"
        )
    if (
        graph_basis.response_binding_sha256
        != base_plan.response_binding_sha256
        or graph_basis.fit_weighted_kernels_sha256
        != base_plan.fit_weighted_kernels_sha256
        or graph_basis.fit_origins != base_plan.fit_knot_origins
        or graph_basis.fft_length != base_plan.fft_length
        or graph_basis.source_modes != base_plan.source_modes
    ):
        raise ValueError("graph and SVD plan fit provenance differ")
    if organization_kind not in _ORGANIZATION_KINDS:
        raise ValueError("organization_kind is invalid")
    if type(organization_seed) is not int or organization_seed < 0:
        raise ValueError("organization_seed must be nonnegative")
    if (
        organization_kind
        in {
            "signed_gfa_dyadic",
            "magnitude_gfa_dyadic",
            "singular_contiguous_control",
        }
        and organization_seed != 0
    ):
        raise ValueError("deterministic organization kinds require seed zero")
    boundaries = _boundaries(
        (
            _default_boundaries(base_plan.source_modes)
            if frequency_band_boundaries is None
            else frequency_band_boundaries
        ),
        width=base_plan.source_modes,
    )
    if organization_kind == "magnitude_gfa_dyadic":
        eigenvalues = graph_basis.magnitude_eigenvalues
        eigenvectors = graph_basis.magnitude_eigenvectors
    else:
        eigenvalues = graph_basis.signed_eigenvalues
        eigenvectors = graph_basis.signed_eigenvectors
    graph_row_permutation = torch.arange(
        base_plan.source_modes,
        dtype=torch.int64,
    )
    if organization_kind == "signed_row_permutation_dyadic":
        graph_row_permutation = torch.randperm(
            base_plan.source_modes,
            generator=torch.Generator().manual_seed(organization_seed),
        )
        eigenvectors = eigenvectors[graph_row_permutation].contiguous()
    component_energy = (
        eigenvectors.T @ base_plan.source_basis
    ).square().contiguous()
    band_mass = torch.stack(
        tuple(
            component_energy[start:stop].sum(dim=0)
            for start, stop in zip(
                boundaries[:-1],
                boundaries[1:],
                strict=True,
            )
        )
    ).contiguous()
    if organization_kind in {
        "signed_gfa_dyadic",
        "magnitude_gfa_dyadic",
        "signed_row_permutation_dyadic",
    }:
        assignments = band_mass.argmax(dim=0).to(dtype=torch.int64)
        counts = torch.bincount(
            assignments,
            minlength=len(boundaries) - 1,
        )
        if bool((counts == 0).any()):
            raise ValueError("graph organization produced an empty pack")
    else:
        if matched_pack_counts is None:
            raise ValueError("control organization requires matched counts")
        if len(tuple(matched_pack_counts)) != len(boundaries) - 1:
            raise ValueError(
                "matched pack counts must match the frequency band count"
            )
        assignments = _size_matched_assignments(
            matched_pack_counts,
            kind=organization_kind,
            seed=organization_seed,
        )
        if assignments.numel() != base_plan.source_rank:
            raise ValueError("matched pack counts do not cover source rank")
    component_permutation = _pack_permutation(assignments)
    offsets = _pack_offsets(assignments, len(boundaries) - 1)
    source_basis = base_plan.source_basis[
        :,
        component_permutation,
    ].contiguous()
    folded = torch.einsum(
        "klrq,tq->klrt",
        base_plan.knot_cores,
        base_plan.target_basis,
    ).contiguous()
    packed_cores = folded[:, :, component_permutation].contiguous()
    bounds = torch.empty(
        (
            base_plan.knot_count,
            base_plan.lag_count,
            len(boundaries) - 1,
        ),
        dtype=torch.float64,
    )
    for knot in range(base_plan.knot_count):
        for lag in range(base_plan.lag_count):
            for pack in range(len(boundaries) - 1):
                start = int(offsets[pack])
                stop = int(offsets[pack + 1])
                bounds[knot, lag, pack] = _inflated_operator_norm(
                    packed_cores[knot, lag, start:stop]
                )
    result = GraphOrganizedSVDPlan(
        response_binding_sha256=base_plan.response_binding_sha256,
        source_plan_artifact_sha256=base_plan.artifact_sha256,
        graph_basis_artifact_sha256=graph_basis.artifact_sha256,
        fit_weighted_kernels_sha256=(
            base_plan.fit_weighted_kernels_sha256
        ),
        fit_knot_origins=base_plan.fit_knot_origins,
        fft_length=base_plan.fft_length,
        organization_kind=organization_kind,
        organization_seed=organization_seed,
        frequency_band_boundaries=boundaries,
        source_scales=base_plan.source_scales,
        source_basis=source_basis,
        knot_cores=packed_cores,
        source_singular_values=base_plan.source_singular_values,
        target_singular_values=base_plan.target_singular_values,
        graph_eigenvalues=eigenvalues,
        graph_eigenvectors=eigenvectors,
        component_graph_energy=component_energy,
        band_mass=band_mass,
        pack_assignments=assignments,
        component_permutation=component_permutation,
        pack_offsets=offsets,
        graph_row_permutation=graph_row_permutation,
        core_operator_norm_bounds=bounds,
        weighted_total_energy=base_plan.weighted_total_energy,
        weighted_retained_energy=base_plan.weighted_retained_energy,
        weighted_relative_error=base_plan.weighted_relative_error,
        source_parseval_relative_error=(
            base_plan.source_parseval_relative_error
        ),
        target_parseval_relative_error=(
            base_plan.target_parseval_relative_error
        ),
    )
    for origin in base_plan.fit_knot_origins:
        expected = base_plan.weighted_kernel_at_origin(origin)
        actual = result.weighted_kernel_at_origin(origin)
        scale = max(float(torch.linalg.vector_norm(expected)), 1.0)
        if not torch.allclose(
            actual,
            expected,
            atol=2e-10 * scale,
            rtol=2e-10,
        ):
            raise RuntimeError(
                "graph packing failed all-on source-plan equivalence"
            )
    return result
