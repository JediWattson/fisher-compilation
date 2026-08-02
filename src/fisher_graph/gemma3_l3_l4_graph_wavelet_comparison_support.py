"""Shared fit-only support for matched Gemma graph-wavelet comparisons.

The module deliberately separates fit construction from held-out evaluation.
``FrozenGraphWaveletFitContext`` and ``FrozenMatchedPlanPanel`` contain only
fit-origin response values.  The full response tensor is canonicalized for
the first time by :func:`evaluate_frozen_plan_panel`, after every basis and
conditional plan has already been frozen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from types import MappingProxyType

import torch
from torch import Tensor

from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorAccounting,
    ConditionalSpectralGeneratorEvaluation,
    ConditionalSpectralGeneratorPlan,
    evaluate_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    FIT_ORIGINS,
    INTERIOR_ORIGINS,
    SELECTION_ORIGINS,
)
from .gemma3_l3_l4_graph_wavelet_experiment import (
    DIFFUSION_SCALES,
    EIGENVALUE_TOLERANCE,
    Gemma3GraphWaveletCandidate,
    _float_tensor,
    _floating_tensor_geometry,
    _laplacian,
    _tensor_sha256,
)
from .graph_spectral_source_basis import (
    FitOnlyGraphSourceBasis,
    fit_graph_source_bases,
)
from .graph_wavelet_analysis import (
    FitOnlyGraphWaveletOMPSubspace,
    SpectralGraphWaveletFrame,
    build_spectral_graph_wavelet_frame,
    fit_graph_wavelet_omp_subspace,
)


__all__ = [
    "FrozenGraphWaveletFitContext",
    "FrozenMatchedPlanPanel",
    "MatchedBasisFamily",
    "basis_locality",
    "evaluate_frozen_plan_panel",
    "freeze_matched_plan_panel",
    "full_rank_plan_accounting",
    "matched_plan_row_prefix",
    "reconstruct_authenticated_q64_fit_context",
]


def _metadata_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be a string-keyed mapping")
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain JSON metadata only") from error
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must normalize to an object")
    return MappingProxyType(normalized)


def _parent_rate_row(
    parent: Gemma3GraphWaveletCandidate,
    *,
    rank: int,
) -> Mapping[str, object]:
    matches = tuple(
        row
        for row in parent.rate_rows
        if row.get("method") == "signed_graph_wavelet_omp"
        and row.get("source_rank") == rank
    )
    if len(matches) != 1:
        raise ValueError("parent signed graph-wavelet row is missing")
    return matches[0]


@dataclass(frozen=True, slots=True, eq=False)
class FrozenGraphWaveletFitContext:
    """Authenticated Q64 reconstruction containing fit values only."""

    response_shape: tuple[int, int, int, int]
    measured_origins: tuple[int, ...]
    source_scales: Tensor
    fit_kernels: Tensor
    weighted_fit: Tensor
    fit_folds: tuple[Tensor, ...]
    graph: FitOnlyGraphSourceBasis
    signed_laplacian: Tensor
    signed_frame: SpectralGraphWaveletFrame
    parent_subspace: FitOnlyGraphWaveletOMPSubspace
    q64: Tensor
    parent_packet_order: tuple[int, ...]
    response_binding_sha256: str
    source_receipt: Mapping[str, object]
    parent_receipt: Mapping[str, object]
    fft_length: int

    @property
    def source_modes(self) -> int:
        return self.response_shape[0]

    @property
    def target_modes(self) -> int:
        return self.response_shape[3]

    @property
    def lag_count(self) -> int:
        return self.response_shape[2]


def reconstruct_authenticated_q64_fit_context(
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    parent: Gemma3GraphWaveletCandidate,
    *,
    response_binding_sha256: str,
    expected_graph_basis_artifact_sha256: str,
    source_receipt: Mapping[str, object],
    fft_length: int,
    expected_parent_rank: int = 64,
) -> FrozenGraphWaveletFitContext:
    """Rebuild and authenticate the parent's full GOMP basis from fit origins."""

    if not isinstance(parent, Gemma3GraphWaveletCandidate):
        raise TypeError("parent must be a graph-wavelet candidate")
    parent.validate_integrity()
    geometry = _floating_tensor_geometry(
        responses,
        label="responses",
        ndim=4,
    )
    scales = _float_tensor(source_scales, label="source_scales", ndim=1)
    measured_origins = tuple(origins)
    if (
        measured_origins != INTERIOR_ORIGINS
        or len(measured_origins) != geometry.shape[1]
        or geometry.shape[0] != expected_parent_rank
        or scales.numel() != expected_parent_rank
        or parent.protocol.get("source_modes") != expected_parent_rank
    ):
        raise ValueError("graph-wavelet comparison geometry or origins differ")
    if geometry.shape[3] != parent.protocol.get("target_modes"):
        raise ValueError("graph-wavelet comparison requires the parent target width")

    fit_indices = torch.tensor(
        [measured_origins.index(origin) for origin in FIT_ORIGINS],
        dtype=torch.int64,
        device=geometry.device,
    )
    fit_kernels = _float_tensor(
        geometry.index_select(1, fit_indices),
        label="fit responses",
        ndim=4,
    )
    weighted_fit = (
        fit_kernels * scales.view(-1, 1, 1, 1)
    ).contiguous()
    fit_folds = tuple(
        weighted_fit[:, index : index + 1].contiguous()
        for index in range(len(FIT_ORIGINS))
    )
    graph = fit_graph_source_bases(
        fit_kernels,
        scales,
        FIT_ORIGINS,
        FIT_ORIGINS,
        response_binding_sha256=response_binding_sha256,
        fft_length=fft_length,
    )
    if (
        graph.artifact_sha256 != expected_graph_basis_artifact_sha256
        or graph.artifact_sha256
        != parent.graph_receipt.get("artifact_sha256")
        or graph.fit_weighted_kernels_sha256
        != parent.graph_receipt.get("fit_weighted_kernels_sha256")
    ):
        raise ValueError("graph-wavelet comparison graph provenance differs")
    signed_laplacian = _laplacian(
        graph.signed_eigenvalues,
        graph.signed_eigenvectors,
    )
    signed_frame = build_spectral_graph_wavelet_frame(
        signed_laplacian,
        diffusion_scales=DIFFUSION_SCALES,
        eigenvalue_tolerance=EIGENVALUE_TOLERANCE,
    )
    parent_signed_frame = parent.frame_receipts.get("signed")
    if (
        not isinstance(parent_signed_frame, Mapping)
        or signed_frame.artifact_sha256
        != parent_signed_frame.get("artifact_sha256")
    ):
        raise ValueError("graph-wavelet comparison frame provenance differs")
    fit_signals = tuple(
        weighted_fit[:, index].contiguous()
        for index in range(weighted_fit.shape[1])
    )
    parent_subspace = fit_graph_wavelet_omp_subspace(
        signed_frame,
        fit_signals,
        max_rank=expected_parent_rank,
    )
    parent_subspace_receipt = parent_signed_frame.get(
        "fit_only_omp_subspace"
    )
    if (
        not isinstance(parent_subspace_receipt, Mapping)
        or parent_subspace.artifact_sha256
        != parent_subspace_receipt.get("artifact_sha256")
    ):
        raise ValueError("graph-wavelet comparison GOMP provenance differs")
    q64 = parent_subspace.orthonormal_basis.clone()
    packet_order = tuple(parent_subspace.selected_flat_atom_indices)
    parent_rank_row = _parent_rate_row(parent, rank=expected_parent_rank)
    if (
        _tensor_sha256(q64) != parent_rank_row.get("source_basis_sha256")
        or packet_order
        != tuple(parent_rank_row.get("selected_packet_indices", ()))
    ):
        raise ValueError("graph-wavelet comparison Q64 differs from parent")
    if any(
        parent.source_receipt.get(key) != value
        for key, value in source_receipt.items()
    ):
        raise ValueError("graph-wavelet comparison source receipt differs")

    return FrozenGraphWaveletFitContext(
        response_shape=tuple(int(width) for width in geometry.shape),
        measured_origins=measured_origins,
        source_scales=scales,
        fit_kernels=fit_kernels,
        weighted_fit=weighted_fit,
        fit_folds=fit_folds,
        graph=graph,
        signed_laplacian=signed_laplacian,
        signed_frame=signed_frame,
        parent_subspace=parent_subspace,
        q64=q64,
        parent_packet_order=packet_order,
        response_binding_sha256=response_binding_sha256,
        source_receipt=_metadata_mapping(
            source_receipt,
            label="source receipt",
        ),
        parent_receipt=_metadata_mapping(
            {
                "artifact_sha256": parent.artifact_sha256,
                "graph_artifact_sha256": graph.artifact_sha256,
                "signed_frame_artifact_sha256": signed_frame.artifact_sha256,
                "signed_gomp_subspace_artifact_sha256": (
                    parent_subspace.artifact_sha256
                ),
                "q64_source_basis_sha256": (
                    parent_rank_row["source_basis_sha256"]
                ),
                "q64_selected_packet_order_sha256": (
                    parent_rank_row["selected_packet_order_sha256"]
                ),
            },
            label="parent receipt",
        ),
        fft_length=fft_length,
    )


@dataclass(frozen=True, slots=True, eq=False)
class MatchedBasisFamily:
    """One fit-frozen nested source-basis family."""

    method: str
    source_basis_kind: str
    bases: Mapping[int, Tensor]
    construction_receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be nonempty")
        if not isinstance(self.source_basis_kind, str):
            raise TypeError("source_basis_kind must be a string")
        if not isinstance(self.bases, Mapping) or not self.bases:
            raise ValueError("bases must be a nonempty rank mapping")
        frozen: dict[int, Tensor] = {}
        row_count: int | None = None
        for rank, value in sorted(self.bases.items()):
            if type(rank) is not int or rank <= 0:
                raise ValueError("basis ranks must be positive integers")
            basis = _float_tensor(value, label="matched basis", ndim=2)
            if basis.shape[1] != rank:
                raise ValueError("basis column count must equal its rank")
            if row_count is None:
                row_count = int(basis.shape[0])
            if basis.shape[0] != row_count or not torch.allclose(
                basis.T @ basis,
                torch.eye(rank, dtype=torch.float64),
                atol=5.0e-10,
                rtol=5.0e-10,
            ):
                raise ValueError("matched bases must be equally wide and orthonormal")
            frozen[rank] = basis
        ranks = tuple(frozen)
        for lower, upper in zip(ranks, ranks[1:]):
            small = frozen[lower]
            large = frozen[upper]
            residual = small - large @ (large.T @ small)
            if float(torch.linalg.vector_norm(residual)) > 5.0e-9:
                raise ValueError("matched basis subspaces must be nested")
        object.__setattr__(self, "bases", MappingProxyType(frozen))
        object.__setattr__(
            self,
            "construction_receipt",
            _metadata_mapping(
                self.construction_receipt,
                label="construction receipt",
            ),
        )

    @property
    def available_ranks(self) -> tuple[int, ...]:
        return tuple(self.bases)


def basis_locality(basis: Tensor, laplacian: Tensor) -> dict[str, object]:
    """Return basis-locality diagnostics shared by every comparison arm."""

    probability = basis.square()
    effective_support = 1.0 / probability.square().sum(dim=0)
    graph_variation = torch.diagonal(basis.T @ laplacian @ basis)
    return {
        "column_count": int(basis.shape[1]),
        "mean_effective_node_support": float(effective_support.mean()),
        "maximum_effective_node_support": float(effective_support.max()),
        "mean_signed_graph_quadratic_form": float(graph_variation.mean()),
        "maximum_signed_graph_quadratic_form": float(graph_variation.max()),
        "hop_locality_claim": False,
    }


def full_rank_plan_accounting(
    context: FrozenGraphWaveletFitContext,
) -> ConditionalSpectralGeneratorAccounting:
    spectrum_bins = context.fft_length // 2 + 1
    return ConditionalSpectralGeneratorAccounting(
        source_modes=context.source_modes,
        target_modes=context.target_modes,
        source_rank=context.source_modes,
        target_rank=context.target_modes,
        knot_count=len(FIT_ORIGINS),
        lag_count=context.lag_count,
        source_spectrum_rank=min(
            context.source_modes,
            2 * len(FIT_ORIGINS) * spectrum_bins * context.target_modes,
        ),
        target_spectrum_rank=min(
            context.target_modes,
            2 * context.source_modes * len(FIT_ORIGINS) * spectrum_bins,
        ),
    )


def matched_plan_row_prefix(
    *,
    context: FrozenGraphWaveletFitContext,
    family: MatchedBasisFamily,
    rank: int,
    plan: ConditionalSpectralGeneratorPlan,
    fit_evaluation: ConditionalSpectralGeneratorEvaluation,
    full_accounting: ConditionalSpectralGeneratorAccounting,
) -> Mapping[str, object]:
    accounting = plan.accounting().metadata()
    prepared_bytes = int(accounting["prepared_storage_bytes"])
    return MappingProxyType(
        {
            "method": family.method,
            "rank": rank,
            "source_rank": plan.source_rank,
            "target_rank": plan.target_rank,
            "source_basis_kind": family.source_basis_kind,
            "source_basis_sha256": _tensor_sha256(plan.source_basis),
            "plan_artifact_sha256": plan.artifact_sha256,
            "fit_weighted_kernels_sha256": (
                plan.fit_weighted_kernels_sha256
            ),
            "fit_evaluation": fit_evaluation.metadata(),
            "heldout_evaluation": None,
            "basis_locality": basis_locality(
                plan.source_basis,
                context.signed_laplacian,
            ),
            "construction_receipt": dict(family.construction_receipt),
            "coefficient_payload": {
                "source_basis_float64_scalars": int(
                    plan.source_basis.numel()
                ),
                "target_basis_float64_scalars": int(
                    plan.target_basis.numel()
                ),
                "fit_knot_core_float64_scalars": int(
                    plan.knot_cores.numel()
                ),
                "compiled_plan_float64_scalars": (
                    plan.stored_coefficient_count
                ),
                "compiled_plan_coefficient_fraction_of_full_rank": (
                    plan.stored_coefficient_count
                    / full_accounting.stored_coefficient_count
                ),
                "prepared_runtime_storage_bytes": prepared_bytes,
                "prepared_runtime_storage_fraction_of_full_rank": (
                    prepared_bytes / full_accounting.prepared_storage_bytes
                ),
                "storage_gate_uses_complete_prepared_runtime_bytes": True,
                "equal_rank_payload_match": True,
                "basis_construction_metadata_excluded_from_plan_payload": True,
            },
            "plan_accounting": accounting,
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class FrozenMatchedPlanPanel:
    """Fit-frozen plans and fit metrics; no held-out response values."""

    context: FrozenGraphWaveletFitContext
    families: tuple[MatchedBasisFamily, ...]
    method_order: tuple[str, ...]
    ranks: tuple[int, ...]
    plans: tuple[ConditionalSpectralGeneratorPlan, ...]
    fit_evaluations: tuple[ConditionalSpectralGeneratorEvaluation, ...]
    row_prefixes: tuple[Mapping[str, object], ...]
    full_accounting: ConditionalSpectralGeneratorAccounting


def freeze_matched_plan_panel(
    context: FrozenGraphWaveletFitContext,
    families: Sequence[MatchedBasisFamily],
    *,
    ranks: Sequence[int] | None = None,
) -> FrozenMatchedPlanPanel:
    """Freeze every basis, compensated plan, and fit metric before selection."""

    if not isinstance(context, FrozenGraphWaveletFitContext):
        raise TypeError("context must be a FrozenGraphWaveletFitContext")
    family_values = tuple(families)
    if (
        not family_values
        or any(not isinstance(value, MatchedBasisFamily) for value in family_values)
        or len({value.method for value in family_values}) != len(family_values)
    ):
        raise ValueError("families must be nonempty with unique methods")
    selected_ranks = (
        family_values[0].available_ranks
        if ranks is None
        else tuple(ranks)
    )
    if (
        not selected_ranks
        or tuple(sorted(set(selected_ranks))) != selected_ranks
        or any(
            family.available_ranks != selected_ranks
            for family in family_values
        )
    ):
        raise ValueError("every family must expose the same ordered ranks")
    for family in family_values:
        if any(
            family.bases[rank].shape[0] != context.source_modes
            for rank in selected_ranks
        ):
            raise ValueError("matched basis source width differs from context")

    full_accounting = full_rank_plan_accounting(context)
    plans: list[ConditionalSpectralGeneratorPlan] = []
    fit_evaluations: list[ConditionalSpectralGeneratorEvaluation] = []
    prefixes: list[Mapping[str, object]] = []
    for family in family_values:
        for rank in selected_ranks:
            plan = fit_conditional_spectral_generator_with_source_basis(
                context.fit_kernels,
                context.source_scales,
                FIT_ORIGINS,
                FIT_ORIGINS,
                family.bases[rank],
                context.target_modes,
                source_basis_kind=family.source_basis_kind,  # type: ignore[arg-type]
                source_basis_fit_weighted_kernels_sha256=(
                    context.graph.fit_weighted_kernels_sha256
                ),
                response_binding_sha256=context.response_binding_sha256,
                input_transform="standardized_linear",
                fft_length=context.fft_length,
            )
            fit_evaluation = evaluate_conditional_spectral_generator(
                plan,
                context.fit_kernels,
                FIT_ORIGINS,
                FIT_ORIGINS,
                response_binding_sha256=context.response_binding_sha256,
            )
            plans.append(plan)
            fit_evaluations.append(fit_evaluation)
            prefixes.append(
                matched_plan_row_prefix(
                    context=context,
                    family=family,
                    rank=rank,
                    plan=plan,
                    fit_evaluation=fit_evaluation,
                    full_accounting=full_accounting,
                )
            )
    return FrozenMatchedPlanPanel(
        context=context,
        families=family_values,
        method_order=tuple(family.method for family in family_values),
        ranks=selected_ranks,
        plans=tuple(plans),
        fit_evaluations=tuple(fit_evaluations),
        row_prefixes=tuple(prefixes),
        full_accounting=full_accounting,
    )


def evaluate_frozen_plan_panel(
    panel: FrozenMatchedPlanPanel,
    responses: Tensor,
    origins: Sequence[int],
    *,
    selection_origins: Sequence[int] = SELECTION_ORIGINS,
) -> tuple[Mapping[str, object], ...]:
    """Open held-out response values only after the complete panel is frozen."""

    if not isinstance(panel, FrozenMatchedPlanPanel):
        raise TypeError("panel must be a FrozenMatchedPlanPanel")
    measured_origins = tuple(origins)
    selected_origins = tuple(selection_origins)
    if (
        measured_origins != panel.context.measured_origins
        or not selected_origins
        or set(selected_origins) & set(FIT_ORIGINS)
    ):
        raise ValueError("held-out origin split differs from the frozen panel")
    kernels = _float_tensor(responses, label="responses", ndim=4)
    if tuple(kernels.shape) != panel.context.response_shape:
        raise ValueError("held-out response geometry differs from the panel")
    fit_indices = torch.tensor(
        [measured_origins.index(origin) for origin in FIT_ORIGINS],
        dtype=torch.int64,
    )
    if not torch.equal(
        kernels.index_select(1, fit_indices),
        panel.context.fit_kernels,
    ):
        raise ValueError("fit response values differ from the frozen panel")
    rows: list[Mapping[str, object]] = []
    for plan, prefix in zip(
        panel.plans,
        panel.row_prefixes,
        strict=True,
    ):
        heldout = evaluate_conditional_spectral_generator(
            plan,
            kernels,
            measured_origins,
            selected_origins,
            response_binding_sha256=panel.context.response_binding_sha256,
            require_heldout=True,
        )
        row = dict(prefix)
        row["heldout_evaluation"] = heldout.metadata()
        rows.append(MappingProxyType(row))
    return tuple(rows)
