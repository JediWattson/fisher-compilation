"""Analytic state-gain capacity screen on the pinned same-A K64 candidate.

This v6 rung asks whether a four-feature, row-conditioned amplitude field has
held-family *local* capacity beyond the static v5 one-over-64 update.  It
recollects the authenticated parent and the exact v4 unit-candidate VJP bank,
retaining row resolution only in memory.  Nested family-disjoint analytic
fits compare the state field with an identically trained scalar amplitude
control.  No state-conditioned candidate is executed and no finite behavior,
compression, serving, or speed claim is authorized by this report.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic as v5diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic as v3diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic as v4diag
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from . import complete_h4_tail_candidate_gain_microstep as microstep
from .complete_h4_tail_candidate_gain_refit_v4 import (
    CANDIDATE_GAIN_RANK,
    CandidateConditionedK64GainGradientExampleV4,
    CandidateConditionedK64MeanKLRefit,
    contract_candidate_teacher_kl_gain_scores,
    fit_candidate_conditioned_k64_mean_kl_gains,
)
from .complete_h4_tail_candidate_state_gain_field import (
    STATE_FEATURE_RANK,
    STATE_GAIN_BASE_STEP,
    CandidateConditionedK64StateFeatureExample,
    CandidateConditionedK64StateFeatureCodec,
    CandidateConditionedK64StateGainAnalyticScreen,
    CandidateConditionedK64StateGainFoldAnalyticRecord,
    CandidateConditionedK64StateGainGradientExample,
    build_candidate_conditioned_k64_inner_family_analytic_record,
    build_candidate_conditioned_k64_state_gain_fold_analytic_record,
    contract_candidate_conditioned_k64_row_direction_scores,
    encode_candidate_conditioned_k64_state_features,
    fit_candidate_conditioned_k64_state_feature_codec,
    fit_candidate_conditioned_k64_state_gain_field,
    fit_candidate_conditioned_k64_static_amplitude_control,
    reduce_candidate_conditioned_k64_row_mode_scores,
    screen_candidate_conditioned_k64_state_gain_capacity,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    fit_complete_h4_tail_held_family,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import _load_committed_basis
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
)


__all__ = [
    "DEFAULT_EXPANDED_PARENT_REPORT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRANSFER_REPORT",
    "DEFAULT_V3_REPORT",
    "DEFAULT_V4_REPORT",
    "DEFAULT_V5_REPORT",
    "V5_REPORT_FILE_SHA256",
    "V5_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v5diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v5diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v5diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v4diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v5diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v5diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-state-gain-capacity-lofo-a-fit16-dev-v6.json"
)

V5_REPORT_FILE_SHA256 = (
    "488edc027f3265a624762af7f0ad6ec0ca9e7ff81bba469862a5ef0fdc72427b"
)
V5_REPORT_SHA256 = (
    "dc87205ee91c0e854155de7f27acf7aacd14a90a2e16b32f9288d843dc911459"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_state_gain_capacity_lofo.v6"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-state-gain-capacity:v6\0"
_ROW_BANK_DOMAIN = b"fisher-graph:complete-h4-k64-state-gain-row-bank:v6\0"
_V5_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-state-gain-v5-binding:v6\0"
_V5_CARRIER_DOMAIN = b"fisher-graph:complete-h4-k64-state-gain-v5-carrier:v6\0"
_V5_CARRIER_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-state-gain-v5-carrier-set:v6\0"
)
_EXPECTED_PARENT_FORWARDS = 48
_EXPECTED_PARENT_BACKWARDS = 109
_EXPECTED_GRADIENT_NATIVE_FORWARDS = 8
_EXPECTED_GRADIENT_CANDIDATE_FORWARDS = 56
_EXPECTED_GRADIENT_BACKWARDS = 385
_EXPECTED_TOTAL_FORWARDS = 112
_EXPECTED_TOTAL_BACKWARDS = 494
_EXPECTED_FOLDS = 8
_EXPECTED_INNER_FOLDS = 7
_STATE_FEATURE_RANK = STATE_FEATURE_RANK
_BASE_STEP = STATE_GAIN_BASE_STEP

_PromptRoles = v4diag._PromptRoles
_checkerboard_prompt_roles = v4diag._checkerboard_prompt_roles
_fresh_native_teacher = v4diag._fresh_native_teacher
_endpoint_indices = v4diag._endpoint_indices
_ordered_k64 = v4diag._ordered_k64
_ordered_k64_relevance = v4diag._ordered_k64_relevance
_authenticate_parent_recollection = v4diag._authenticate_parent_recollection
_authenticate_static_unit_k64_replay = v4diag._authenticate_static_unit_k64_replay
_endpoint_prompt_receipts = v4diag._endpoint_prompt_receipts


def _canonical(value: object) -> object:
    return v3diag._canonical(value)


def _load_v5_report(path: Path | str) -> dict[str, object]:
    """Load v5 only when file, logical report, schema, and result all match."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V5_REPORT_FILE_SHA256,
        expected_report_sha256=V5_REPORT_SHA256,
        label="candidate symmetric-microstep v5 control",
    )
    if (
        report.get("schema") != v5diag._SCHEMA
        or report.get("classification")
        != "symmetric_microstep_static_cross_family_transfer_blocker_same_a"
        or report.get("passed") is not False
    ):
        raise RuntimeError("candidate symmetric-microstep v5 control differs")
    return report


def _mean_delta(refit: CandidateConditionedK64MeanKLRefit) -> Tensor:
    value = getattr(refit, "mean_proposed_gains_tensor")
    gains = value() if callable(value) else value
    if not isinstance(gains, Tensor):
        raise TypeError("v4 mean proposed gain accessor must return a tensor")
    delta = (
        gains.detach().to(device="cpu", dtype=torch.float64).contiguous() - 1.0
    )
    if delta.shape != (CANDIDATE_GAIN_RANK,) or not bool(torch.isfinite(delta).all()):
        raise ValueError("v4 mean gain delta geometry differs")
    return delta


def _row_resolved_contractions(
    *,
    base_rows: Tensor,
    tail_rows: Tensor,
    ordered_directions: Tensor,
    token_h4_gradients: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return raw K4 features, row/mode scores, and exact frozen-v4 scores.

    The frozen v4 contraction remains the artifact authority.  The row sum is
    checked within a tight float64 reduction tolerance because materializing
    the row axis changes floating-point association.  This lets v6 add row
    resolution without changing any v4 artifact.
    """

    base = base_rows.detach().to(device="cpu", dtype=torch.float64).contiguous()
    tail = tail_rows.detach().to(device="cpu", dtype=torch.float64).contiguous()
    directions = (
        ordered_directions.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    gradients = (
        token_h4_gradients.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    if (
        base.ndim != 2
        or tail.ndim != 2
        or gradients.ndim != 3
        or base.shape != tail.shape
        or gradients.shape[1:] != tail.shape
        or directions.shape != (CANDIDATE_GAIN_RANK, tail.shape[1])
    ):
        raise ValueError("v6 row-resolved contraction geometry differs")
    raw_features = (base @ directions[:_STATE_FEATURE_RANK].T).contiguous()
    amplitudes = (tail @ directions.T).contiguous()
    gradient_coordinates = torch.einsum(
        "trw,kw->trk", gradients, directions
    ).contiguous()
    row_mode_scores = (amplitudes.unsqueeze(0) * gradient_coordinates).contiguous()
    expected = contract_candidate_teacher_kl_gain_scores(
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradients,
    )
    row_sum = row_mode_scores.sum(dim=1).contiguous()
    if not torch.allclose(row_sum, expected, rtol=0.0, atol=1.0e-12):
        maximum = float((row_sum - expected).abs().max())
        raise RuntimeError(
            "v6 row-resolved scores do not reconstruct v4 within float64 "
            f"reduction tolerance (maximum_abs_error={maximum})"
        )
    # The exact frozen-v4 contraction is authoritative for artifact hashes;
    # the unmodified row tensor remains authoritative for the new state fit.
    return raw_features, row_mode_scores, expected


def _execute_row_resolved_candidate_teacher_kl_vjp(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    model_inputs: Mapping[str, Tensor],
    teacher_logits: Tensor,
    endpoint_indices: Tensor,
    endpoint_grid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, object], int]:
    """Execute the exact v4 unit VJP while retaining ephemeral row scores."""

    gains = torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
    directions, tail, _correction_rows, correction = v3diag._candidate_components(
        trace, basis=basis, fit=fit, gains=gains
    )
    directions_sha256 = _runtime_tensor_sha256(directions)
    provider = v3diag._AuthenticatedCandidateGainProvider(
        stage="gradient",
        gain_kind="unit",
        fold_artifact_sha256=fit.artifact_sha256,
        ordered_directions_sha256=directions_sha256,
        gains=gains,
        refit_artifact_sha256=None,
        selection_artifact_sha256=None,
        alpha=None,
        model_inputs_sha256=trace.model_inputs_sha256,
        bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
        prefix_artifact_sha256=trace.prefix.artifact_sha256,
        base_h4=trace.base_h4,
        support_mask=trace.prefix.complete_h4_causal_support_mask(),
        correction=correction,
    )
    vjp = getattr(context, "bridge").execute_h4_token_teacher_kl_vjps(
        getattr(context, "adapter"),
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=endpoint_grid,
        vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
        h4_head=provider,
    )
    vjp.validate_integrity()
    v3diag._validate_candidate_execution(
        trace=trace, provider=provider, execution=vjp.execution
    )
    if (
        vjp.teacher_logits_sha256 != _runtime_tensor_sha256(teacher_logits)
        or not torch.equal(
            vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
        )
    ):
        raise RuntimeError("v6 candidate teacher-KL VJP authority differs")
    token_kl = (
        vjp.token_kl_divergences.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    independent = teacher_kl._selected_token_teacher_kl(
        teacher_logits, vjp.execution.logits, endpoint_indices
    )
    if not torch.allclose(token_kl, independent, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("v6 candidate teacher-KL objective authority differs")
    gradient_rows = (
        vjp.h4_gradients.detach()
        .to(device="cpu", dtype=torch.float64)[:, 0]
        .index_select(1, trace.support_indices)
        .contiguous()
    )
    maximum_future, future_nonzero = v3diag._future_gradient_evidence(
        trace=trace,
        endpoint_indices=endpoint_indices,
        gradient_rows=gradient_rows,
    )
    base_rows = (
        trace.base_h4.detach()
        .to(device="cpu")[0]
        .index_select(0, trace.support_indices)
        .contiguous()
    )
    raw_features, row_mode_scores, static_scores = _row_resolved_contractions(
        base_rows=base_rows,
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradient_rows,
    )
    evidence = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "held_family_id": fit.held_family_id,
        "fold_artifact_sha256": fit.artifact_sha256,
        "ordered_directions_sha256": directions_sha256,
        "gains_sha256": provider.gains_sha256,
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_artifact_sha256": vjp.execution.artifact_sha256,
        "teacher_kl_vjp_artifact_sha256": vjp.artifact_sha256,
        "teacher_logits_sha256": vjp.teacher_logits_sha256,
        "supervised_grid_sha256": _runtime_tensor_sha256(endpoint_grid),
        "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
        "token_teacher_kl_mean": float(token_kl.mean()),
        "token_gain_gradients_runtime_sha256": _runtime_tensor_sha256(static_scores),
        "backward_call_count": vjp.backward_call_count,
        "maximum_future_gradient_abs": maximum_future,
        "future_gradient_nonzero_count": future_nonzero,
        "stage": "fit_gradient",
        "candidate_is_realized_unit_gain_k64": True,
        "held_family_used": False,
        "raw_tensors_serialized": False,
    }
    calls = int(vjp.backward_call_count)
    del vjp, provider, correction, independent, gradient_rows, tail, directions
    return (
        token_kl,
        base_rows,
        raw_features,
        row_mode_scores,
        static_scores,
        evidence,
        calls,
    )


@dataclass(slots=True)
class _RowBankCell:
    example_id: str
    family_id: str
    held_family_id: str
    base_h4_support_rows: Tensor
    raw_features: Tensor
    row_mode_scores: Tensor
    token_teacher_kl: Tensor


@dataclass(slots=True)
class _RowBankResult:
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit]
    cells: Mapping[str, tuple[_RowBankCell, ...]]
    gradient_receipts: tuple[dict[str, object], ...]
    row_bank_receipts: tuple[dict[str, object], ...]
    resources: Mapping[str, int]


def _feature_example(cell: _RowBankCell) -> CandidateConditionedK64StateFeatureExample:
    return CandidateConditionedK64StateFeatureExample(
        example_id=cell.example_id,
        family_id=cell.family_id,
        base_h4_support_rows=cell.base_h4_support_rows,
    )


def _tie_row_direction_scores(
    *, row_mode_scores: Tensor, mean_gain_delta: Tensor
) -> Tensor:
    """Apply the one frozen K64 delta and one-over-64 carrier to row scores."""

    scores = row_mode_scores.detach().to(device="cpu", dtype=torch.float64)
    delta = mean_gain_delta.detach().to(device="cpu", dtype=torch.float64)
    if (
        scores.ndim != 3
        or scores.shape[2] != CANDIDATE_GAIN_RANK
        or delta.shape != (CANDIDATE_GAIN_RANK,)
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(delta).all())
    ):
        raise ValueError("v6 tied row-direction score geometry differs")
    return reduce_candidate_conditioned_k64_row_mode_scores(
        token_row_mode_scores=scores,
        mean_gain_delta=delta,
    )


def _state_gradient_example(
    *,
    cell: _RowBankCell,
    codec: CandidateConditionedK64StateFeatureCodec,
    ordered_directions: Tensor,
    mean_gain_delta: Tensor,
) -> CandidateConditionedK64StateGainGradientExample:
    standardized = encode_candidate_conditioned_k64_state_features(
        codec,
        base_h4_support_rows=cell.base_h4_support_rows,
        ordered_directions=ordered_directions,
    )
    expected = (
        (cell.raw_features - codec.feature_center)
        / codec.feature_scale
    ).contiguous()
    if not torch.equal(standardized, expected):
        raise RuntimeError("v6 raw K4 feature replay differs from codec encoding")
    row_scores = _tie_row_direction_scores(
        row_mode_scores=cell.row_mode_scores,
        mean_gain_delta=mean_gain_delta,
    )
    return CandidateConditionedK64StateGainGradientExample(
        example_id=cell.example_id,
        family_id=cell.family_id,
        codec_artifact_sha256=codec.artifact_sha256,
        standardized_state_features=standardized,
        token_row_direction_scores=row_scores,
        unit_token_teacher_kl=cell.token_teacher_kl,
    )


def _fit_nested_capacity_screen(
    *,
    row_bank: _RowBankResult,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> tuple[
    CandidateConditionedK64StateGainAnalyticScreen,
    tuple[CandidateConditionedK64StateGainFoldAnalyticRecord, ...],
]:
    """Fit eight outer and 56 nested-inner analytic state/control screens."""

    fold_records: list[CandidateConditionedK64StateGainFoldAnalyticRecord] = []
    for outer_family in sorted(fits):
        fit = fits[outer_family]
        refit = row_bank.refits[outer_family]
        directions = _ordered_k64(fit)
        delta = _mean_delta(refit)
        outer_cells = tuple(
            sorted(row_bank.cells[outer_family], key=lambda cell: cell.family_id)
        )
        if (
            len(outer_cells) != _EXPECTED_INNER_FOLDS
            or {cell.family_id for cell in outer_cells}
            != set(refit.training_family_ids)
            or outer_family in {cell.family_id for cell in outer_cells}
        ):
            raise RuntimeError("v6 outer row-bank family grid differs")
        full_features = tuple(_feature_example(cell) for cell in outer_cells)
        full_codec = fit_candidate_conditioned_k64_state_feature_codec(
            full_features,
            held_family_id=outer_family,
            ordered_directions=directions,
        )
        full_examples = tuple(
            _state_gradient_example(
                cell=cell,
                codec=full_codec,
                ordered_directions=directions,
                mean_gain_delta=delta,
            )
            for cell in outer_cells
        )
        full_field = fit_candidate_conditioned_k64_state_gain_field(
            refit, full_codec, full_examples
        )
        full_static = fit_candidate_conditioned_k64_static_amplitude_control(
            refit, full_codec, full_examples
        )
        inner_records: list[object] = []
        for inner_family in sorted(cell.family_id for cell in outer_cells):
            training_cells = tuple(
                cell for cell in outer_cells if cell.family_id != inner_family
            )
            held_cell = next(
                cell for cell in outer_cells if cell.family_id == inner_family
            )
            inner_codec = fit_candidate_conditioned_k64_state_feature_codec(
                tuple(_feature_example(cell) for cell in training_cells),
                held_family_id=outer_family,
                ordered_directions=directions,
            )
            inner_examples = tuple(
                _state_gradient_example(
                    cell=cell,
                    codec=inner_codec,
                    ordered_directions=directions,
                    mean_gain_delta=delta,
                )
                for cell in training_cells
            )
            inner_field = fit_candidate_conditioned_k64_state_gain_field(
                refit, inner_codec, inner_examples
            )
            inner_static = fit_candidate_conditioned_k64_static_amplitude_control(
                refit, inner_codec, inner_examples
            )
            held_example = _state_gradient_example(
                cell=held_cell,
                codec=inner_codec,
                ordered_directions=directions,
                mean_gain_delta=delta,
            )
            inner_records.append(
                build_candidate_conditioned_k64_inner_family_analytic_record(
                    full_field,
                    inner_field,
                    inner_static,
                    held_example,
                )
            )
        if len(inner_records) != _EXPECTED_INNER_FOLDS:
            raise RuntimeError("v6 nested held-inner family count differs")
        fold_records.append(
            build_candidate_conditioned_k64_state_gain_fold_analytic_record(
                full_field,
                full_static,
                tuple(inner_records),
            )
        )
    if len(fold_records) != _EXPECTED_FOLDS:
        raise RuntimeError("v6 outer capacity fold count differs")
    screen = screen_candidate_conditioned_k64_state_gain_capacity(
        tuple(fold_records)
    )
    return screen, tuple(fold_records)


def _collect_row_resolved_v4_bank(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    roles: _PromptRoles,
) -> _RowBankResult:
    """Collect one 8x7 bank and reproduce every exact v4 static artifact."""

    by_id = {trace.example_id: trace for trace in traces}
    examples: dict[str, list[CandidateConditionedK64GainGradientExampleV4]] = {
        family: [] for family in sorted(fits)
    }
    cells: dict[str, list[_RowBankCell]] = {family: [] for family in sorted(fits)}
    gradient_receipts: list[dict[str, object]] = []
    row_receipts: list[dict[str, object]] = []
    native_forwards = 0
    candidate_forwards = 0
    backward_calls = 0
    for example_id in roles.fit_example_ids:
        trace = by_id[example_id]
        model_inputs, indices, targets, teacher_logits = _fresh_native_teacher(
            context=context, trace=trace
        )
        native_forwards += 1
        endpoint_indices, _endpoint_targets, endpoint_grid = _endpoint_indices(
            trace, indices, targets
        )
        for held_family in sorted(fits):
            if held_family == trace.family_id:
                continue
            fit = fits[held_family]
            (
                token_kl,
                base_h4_support_rows,
                raw_features,
                row_mode_scores,
                static_scores,
                receipt,
                calls,
            ) = _execute_row_resolved_candidate_teacher_kl_vjp(
                context=context,
                trace=trace,
                basis=basis,
                fit=fit,
                model_inputs=model_inputs,
                teacher_logits=teacher_logits,
                endpoint_indices=endpoint_indices,
                endpoint_grid=endpoint_grid,
            )
            example_v4 = CandidateConditionedK64GainGradientExampleV4(
                example_id=trace.example_id,
                family_id=trace.family_id,
                token_gain_gradients=static_scores,
                token_teacher_kl=token_kl,
            )
            example_v3 = v3diag.CandidateConditionedK64GainGradientExample(
                example_id=trace.example_id,
                family_id=trace.family_id,
                token_gain_gradients=static_scores,
                token_teacher_kl=token_kl,
            )
            bound = {
                **receipt,
                "v4_gradient_example_artifact_sha256": example_v4.artifact_sha256,
                "v3_replay_gradient_example_artifact_sha256": example_v3.artifact_sha256,
                "token_gain_gradients_sha256": example_v4.metadata()[
                    "token_gain_gradients_sha256"
                ],
                "same_vjp_bank_used_for_mean_and_residual_directions": True,
            }
            bound["receipt_sha256"] = token_v1._domain_sha256(
                bound, domain=v4diag._GRADIENT_RECEIPT_DOMAIN
            )
            gradient_receipts.append(bound)
            row_bound = {
                "example_id": trace.example_id,
                "family_id": trace.family_id,
                "held_family_id": held_family,
                "v4_gradient_receipt_sha256": bound["receipt_sha256"],
                "raw_state_features_sha256": _runtime_tensor_sha256(raw_features),
                "row_mode_scores_sha256": _runtime_tensor_sha256(row_mode_scores),
                "row_sum_static_scores_sha256": _runtime_tensor_sha256(static_scores),
                "support_row_count": int(raw_features.shape[0]),
                "supervised_token_count": int(token_kl.numel()),
                "state_feature_rank": int(raw_features.shape[1]),
                "mode_rank": int(row_mode_scores.shape[2]),
                "row_sum_reconstructs_v4_static_scores_within_float64_reduction_tolerance": True,
                "row_sum_static_scores_maximum_abs_error": float(
                    (row_mode_scores.sum(dim=1) - static_scores).abs().max()
                ),
                "exact_v4_static_contraction_used_for_artifact_authentication": True,
                "raw_tensors_serialized": False,
            }
            row_bound["receipt_sha256"] = token_v1._domain_sha256(
                row_bound, domain=_ROW_BANK_DOMAIN
            )
            row_receipts.append(row_bound)
            examples[held_family].append(example_v4)
            cells[held_family].append(
                _RowBankCell(
                    example_id=trace.example_id,
                    family_id=trace.family_id,
                    held_family_id=held_family,
                    base_h4_support_rows=base_h4_support_rows,
                    raw_features=raw_features,
                    row_mode_scores=row_mode_scores,
                    token_teacher_kl=token_kl,
                )
            )
            candidate_forwards += 1
            backward_calls += calls
        del model_inputs, indices, targets, teacher_logits
    expected_backward = sum(
        (
            by_id[example_id].endpoint.supervised_tokens
            + token_v1._VJP_CHUNK_SIZE
            - 1
        )
        // token_v1._VJP_CHUNK_SIZE
        * 7
        for example_id in roles.fit_example_ids
    )
    if (
        native_forwards != _EXPECTED_GRADIENT_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or backward_calls != _EXPECTED_GRADIENT_BACKWARDS
        or expected_backward != _EXPECTED_GRADIENT_BACKWARDS
        or len(gradient_receipts) != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or len(row_receipts) != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("v6 row-resolved VJP accounting differs")
    refits: dict[str, CandidateConditionedK64MeanKLRefit] = {}
    for held_family in sorted(fits):
        fit = fits[held_family]
        refits[held_family] = fit_candidate_conditioned_k64_mean_kl_gains(
            examples[held_family],
            held_family_id=held_family,
            parent_fold_artifact_sha256=fit.artifact_sha256,
            ordered_directions_sha256=_runtime_tensor_sha256(_ordered_k64(fit)),
            ordered_token_fisher_relevance=_ordered_k64_relevance(fit),
        )
        expected_grid = {
            (by_id[example_id].family_id, example_id)
            for example_id in roles.fit_example_ids
            if by_id[example_id].family_id != held_family
        }
        actual_grid = {
            (cell.family_id, cell.example_id) for cell in cells[held_family]
        }
        refit = refits[held_family]
        if (
            len(cells[held_family]) != _EXPECTED_INNER_FOLDS
            or any(cell.held_family_id != held_family for cell in cells[held_family])
            or actual_grid != expected_grid
            or tuple(sorted(family for family, _example in expected_grid))
            != refit.training_family_ids
            or tuple(sorted(example for _family, example in expected_grid))
            != refit.training_example_ids
        ):
            raise RuntimeError("v6 row-bank cell grid differs from its v4 refit")
    return _RowBankResult(
        refits=refits,
        cells={family: tuple(cells[family]) for family in sorted(cells)},
        gradient_receipts=tuple(gradient_receipts),
        row_bank_receipts=tuple(row_receipts),
        resources={
            "gradient_native_forward_count": native_forwards,
            "gradient_candidate_vjp_forward_count": candidate_forwards,
            "gradient_candidate_vjp_backward_call_count": backward_calls,
            "gradient_prompt_fold_count": len(gradient_receipts),
            "row_resolved_candidate_vjp_bank_count": len(row_receipts),
            "additional_model_work_for_row_resolution": 0,
        },
    )


def _authenticate_v3_v4_v5_lineage(
    *,
    v3_report: Mapping[str, object],
    v4_report: Mapping[str, object],
    v5_report: Mapping[str, object],
) -> dict[str, object]:
    """Bind the three exact failed controls that define the v6 question."""

    raw_v3 = v4_report.get("failed_v3_control_binding")
    raw_v4 = v5_report.get("v4_control_binding")
    protocol = v5_report.get("protocol")
    resources = v5_report.get("resources")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_v3, raw_v4, protocol, resources)
    ):
        raise ValueError("pinned v3/v4/v5 lineage evidence differs")
    assert isinstance(raw_v3, Mapping)
    assert isinstance(raw_v4, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(resources, Mapping)
    expected_v3 = {
        "file_sha256": v4diag.V3_REPORT_FILE_SHA256,
        "report_sha256": v4diag.V3_REPORT_SHA256,
        "schema": v3diag._SCHEMA,
        "classification": "candidate_conditioned_k64_gain_refit_not_supported_same_a",
    }
    expected_v4 = {
        "file_sha256": v5diag.V4_REPORT_FILE_SHA256,
        "report_sha256": v5diag.V4_REPORT_SHA256,
        "schema": v4diag._SCHEMA,
        "classification": "tested_mean_KL_OPG_direction_not_supported_same_a",
    }
    if (
        any(raw_v3.get(key) != value for key, value in expected_v3.items())
        or any(raw_v4.get(key) != value for key, value in expected_v4.items())
        or v3_report.get("report_sha256") != v4diag.V3_REPORT_SHA256
        or v4_report.get("report_sha256") != v5diag.V4_REPORT_SHA256
        or v5_report.get("report_sha256") != V5_REPORT_SHA256
        or protocol.get("microstep_epsilon") != _BASE_STEP
        or protocol.get("microstep_epsilon_hex") != _BASE_STEP.hex()
        or resources.get("total_model_forward_count") != 264
        or resources.get("total_backward_call_count") != 494
    ):
        raise RuntimeError("pinned v3/v4/v5 lineage binding differs")
    payload = {
        "v3_report_file_sha256": v4diag.V3_REPORT_FILE_SHA256,
        "v3_report_sha256": v4diag.V3_REPORT_SHA256,
        "v3_schema": v3_report.get("schema"),
        "v3_classification": v3_report.get("classification"),
        "v4_report_file_sha256": v5diag.V4_REPORT_FILE_SHA256,
        "v4_report_sha256": v5diag.V4_REPORT_SHA256,
        "v4_schema": v4_report.get("schema"),
        "v4_classification": v4_report.get("classification"),
        "v5_report_file_sha256": V5_REPORT_FILE_SHA256,
        "v5_report_sha256": V5_REPORT_SHA256,
        "v5_schema": v5_report.get("schema"),
        "v5_classification": v5_report.get("classification"),
        "v5_base_step_hex": _BASE_STEP.hex(),
        "all_three_controls_authenticated_before_state_fit": True,
    }
    return {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V5_BINDING_DOMAIN
        ),
    }


def _authenticate_v5_plus_carriers(
    *,
    v5_report: Mapping[str, object],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
) -> dict[str, object]:
    """Bind v6 to V5's executed +epsilon arm, not V5's selected arm."""

    raw = v5_report.get("candidate_microstep_tune_selections")
    if not isinstance(raw, list):
        raise ValueError("pinned v5 microstep selection evidence differs")
    by_family = {
        str(row["held_family_id"]): dict(row)
        for row in raw
        if isinstance(row, Mapping)
    }
    if len(raw) != _EXPECTED_FOLDS or set(by_family) != set(refits):
        raise RuntimeError("pinned v5 carrier fold grid differs")
    receipts: list[dict[str, object]] = []
    selected_counts = {"plus_epsilon": 0, "unit": 0}
    for family in sorted(refits):
        refit = refits[family]
        row = by_family[family]
        mean_gains = refit.mean_proposed_gains_tensor()
        plus_gains = (
            1.0 + _BASE_STEP * (mean_gains - 1.0)
        ).contiguous()
        selected_arm = str(row.get("selected_arm"))
        if selected_arm not in selected_counts:
            raise RuntimeError("pinned v5 selected carrier arm differs")
        selected_counts[selected_arm] += 1
        if (
            row.get("refit_artifact_sha256") != refit.artifact_sha256
            or row.get("mean_proposed_gains_sha256")
            != microstep._tensor_sha256(mean_gains)
            or row.get("plus_gains_sha256")
            != microstep._tensor_sha256(plus_gains)
            or row.get("microstep_epsilon") != _BASE_STEP
            or row.get("microstep_epsilon_hex") != _BASE_STEP.hex()
        ):
            raise RuntimeError("live v4 refit did not reproduce a v5 plus carrier")
        receipt = {
            "held_family_id": family,
            "v5_selection_artifact_sha256": row.get("artifact_sha256"),
            "v4_refit_artifact_sha256": refit.artifact_sha256,
            "mean_proposed_gains_sha256": row.get(
                "mean_proposed_gains_sha256"
            ),
            "mean_proposed_gains_runtime_sha256": _runtime_tensor_sha256(
                mean_gains
            ),
            "plus_gains_sha256": row.get("plus_gains_sha256"),
            "plus_gains_runtime_sha256": _runtime_tensor_sha256(plus_gains),
            "microstep_epsilon": _BASE_STEP,
            "microstep_epsilon_hex": _BASE_STEP.hex(),
            "v5_selected_arm": selected_arm,
            "v6_carrier_arm": "plus_epsilon",
            "v6_carrier_was_v5_selected_arm": selected_arm == "plus_epsilon",
            "unit_point_linearization_incremental_to_executed_v5_plus_arm": True,
            "raw_tensors_serialized": False,
        }
        receipt["receipt_sha256"] = token_v1._domain_sha256(
            receipt, domain=_V5_CARRIER_DOMAIN
        )
        receipts.append(receipt)
    if selected_counts != {"plus_epsilon": 6, "unit": 2}:
        raise RuntimeError("pinned v5 selected-arm counts differ")
    receipt_set = token_v1._domain_sha256(
        tuple(row["receipt_sha256"] for row in receipts),
        domain=_V5_CARRIER_SET_DOMAIN,
    )
    return {
        "carrier_receipts": tuple(receipts),
        "carrier_receipt_set_sha256": receipt_set,
        "carrier_fold_count": len(receipts),
        "v5_plus_selected_fold_count": selected_counts["plus_epsilon"],
        "v5_unit_selected_fold_count": selected_counts["unit"],
        "counterfactual_plus_carrier_used_in_v5_unit_selected_folds": (
            selected_counts["unit"]
        ),
        "all_carriers_authenticate_live_v4_delta_and_v5_plus_gain_hash": True,
        "carrier_is_v5_selected_arm_in_all_folds": False,
    }


@dataclass(slots=True)
class _CapacityPhaseResults:
    recollection_receipt: str
    static_unit_replay_receipt: str
    roles: _PromptRoles
    row_bank: _RowBankResult
    live_v4_binding: Mapping[str, object]
    lineage_binding: Mapping[str, object]
    carrier_binding: Mapping[str, object]
    screen: CandidateConditionedK64StateGainAnalyticScreen
    fold_records: tuple[CandidateConditionedK64StateGainFoldAnalyticRecord, ...]


def _execute_capacity_phases(
    *,
    context: object,
    parent: Mapping[str, object],
    v3_report: Mapping[str, object],
    v4_report: Mapping[str, object],
    v5_report: Mapping[str, object],
    traces: Sequence[object],
    endpoint_resources: Mapping[str, int],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> _CapacityPhaseResults:
    """Execute the locked auth/recollect/analytic order and nothing else."""

    recollection = _authenticate_parent_recollection(
        parent=parent,
        traces=traces,
        endpoint_resources=endpoint_resources,
        fits=fits,
    )
    static_replay = _authenticate_static_unit_k64_replay(
        parent=parent, traces=traces, basis=basis, fits=fits
    )
    roles = _checkerboard_prompt_roles(traces)
    row_bank = _collect_row_resolved_v4_bank(
        context=context,
        traces=traces,
        basis=basis,
        fits=fits,
        roles=roles,
    )
    live_v4 = v5diag._authenticate_live_v4_refits_and_gradients(
        v4_report=v4_report,
        refits=row_bank.refits,
        gradient_receipts=row_bank.gradient_receipts,
    )
    lineage = _authenticate_v3_v4_v5_lineage(
        v3_report=v3_report,
        v4_report=v4_report,
        v5_report=v5_report,
    )
    carrier = _authenticate_v5_plus_carriers(
        v5_report=v5_report, refits=row_bank.refits
    )
    screen, fold_records = _fit_nested_capacity_screen(
        row_bank=row_bank, fits=fits
    )
    return _CapacityPhaseResults(
        recollection_receipt=recollection,
        static_unit_replay_receipt=static_replay,
        roles=roles,
        row_bank=row_bank,
        live_v4_binding=live_v4,
        lineage_binding=lineage,
        carrier_binding=carrier,
        screen=screen,
        fold_records=fold_records,
    )


def _resource_accounting(
    *,
    endpoint_resources: Mapping[str, int],
    gradient_resources: Mapping[str, int],
) -> dict[str, object]:
    parent_forwards = (
        endpoint_resources["base_forward_count"]
        + endpoint_resources["native_forward_count"]
        + endpoint_resources["endpoint_token_vjp_forward_count"]
    )
    parent_backwards = endpoint_resources["endpoint_token_vjp_backward_call_count"]
    gradient_forwards = (
        gradient_resources["gradient_native_forward_count"]
        + gradient_resources["gradient_candidate_vjp_forward_count"]
    )
    total_forwards = parent_forwards + gradient_forwards
    total_backwards = (
        parent_backwards
        + gradient_resources["gradient_candidate_vjp_backward_call_count"]
    )
    if (
        parent_forwards != _EXPECTED_PARENT_FORWARDS
        or parent_backwards != _EXPECTED_PARENT_BACKWARDS
        or gradient_forwards
        != _EXPECTED_GRADIENT_NATIVE_FORWARDS
        + _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or total_forwards != _EXPECTED_TOTAL_FORWARDS
        or total_backwards != _EXPECTED_TOTAL_BACKWARDS
    ):
        raise RuntimeError("v6 analytic capacity resource accounting differs")
    return {
        **endpoint_resources,
        **gradient_resources,
        "phase_order": (
            "parent_endpoint_recollection",
            "static_unit_k64_reconstruction",
            "row_resolved_unit_candidate_vjp_recollection",
            "exact_v4_refit_and_receipt_authentication",
            "exact_v5_plus_carrier_authentication",
            "nested_held_inner_family_analytic_capacity_screen",
            "scalar_hash_only_report_publication",
        ),
        "parent_collection_model_forward_count": parent_forwards,
        "parent_collection_backward_call_count": parent_backwards,
        "gradient_stage_model_forward_count": gradient_forwards,
        "analytic_fit_model_forward_count": 0,
        "finite_state_candidate_model_forward_count": 0,
        "total_model_forward_count": total_forwards,
        "total_backward_call_count": total_backwards,
        "exact_model_forward_count_is_112": total_forwards
        == _EXPECTED_TOTAL_FORWARDS,
        "exact_backward_call_count_is_494": total_backwards
        == _EXPECTED_TOTAL_BACKWARDS,
        "raw_row_resolved_vjp_banks_retained_in_report": False,
        "serving_learned_parameter_count": "not_applicable_capacity_screen_only",
        "serving_logical_macs_per_token": "not_applicable_capacity_screen_only",
    }


def _capacity_outcome(
    screen: CandidateConditionedK64StateGainAnalyticScreen,
) -> tuple[str, str]:
    """Collapse detailed analytic failures into the predeclared v6 partition."""

    detailed = str(screen.outcome)
    if bool(screen.capacity_screen_passed):
        return "capacity_supported", detailed
    if not bool(screen.feature_and_design_gate_passed) or not bool(
        screen.residual_energy_gate_passed
    ):
        return "degenerate_design", detailed
    if not bool(screen.non_noop_gate_passed):
        return "structural_no_op", detailed
    if (
        not bool(screen.negative_inner_global_gate_passed)
        or not bool(screen.negative_inner_local_gate_passed)
        or not bool(screen.cosine_stability_gate_passed)
    ):
        return "structural_no_op", detailed
    if not bool(screen.state_beats_scalar_gate_passed):
        return "state_not_better_than_scalar", detailed
    raise RuntimeError("v6 capacity outcome partition is incomplete")


def _safety_metadata() -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_state_feature_tensors": False,
        "contains_row_score_tensors": False,
        "contains_token_score_matrices": False,
        "contains_gain_vectors": False,
        "contains_state_weight_vectors": False,
        "contains_basis_coefficients": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    v5_report_path: Path | str = DEFAULT_V5_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the locked same-A v6 analytic state-gain capacity screen."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite candidate state-gain capacity v6 report"
        )
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = v4diag._load_v3_report(v3_report_path)
    v4_report = v5diag._load_v4_report(v4_report_path)
    v5_report = _load_v5_report(v5_report_path)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate state-gain v6 rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate state-gain v6 rank320 transfer",
    )
    transfer_receipts = expanded._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = token_v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if (
            len(traces) != token_v1._EXPECTED_EXAMPLES
            or len(families) != token_v1._EXPECTED_FAMILIES
        ):
            raise RuntimeError("candidate state-gain v6 A16 panel shape differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        phases = _execute_capacity_phases(
            context=context,
            parent=parent,
            v3_report=v3_report,
            v4_report=v4_report,
            v5_report=v5_report,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    resources = _resource_accounting(
        endpoint_resources=endpoint_resources,
        gradient_resources=phases.row_bank.resources,
    )
    row_errors = tuple(
        float(row["row_sum_static_scores_maximum_abs_error"])
        for row in phases.row_bank.row_bank_receipts
    )
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    ) and all(
        row["maximum_future_gradient_abs"] == 0.0
        and row["future_gradient_nonzero_count"] == 0
        for row in phases.row_bank.gradient_receipts
    )
    screen = phases.screen
    capacity_outcome, detailed_outcome = _capacity_outcome(screen)
    integrity_gates = {
        "expanded_v2_parent_authenticated": True,
        "v3_v4_v5_controls_authenticated_by_exact_file_and_report_sha256": True,
        "live_row_bank_canonically_reproduced_all_eight_v4_refits": (
            phases.live_v4_binding["live_refit_count"] == _EXPECTED_FOLDS
        ),
        "live_row_bank_canonically_reproduced_all_56_v4_gradient_receipts": (
            phases.live_v4_binding["live_gradient_receipt_count"]
            == _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        ),
        "all_v4_evidence_authenticated_before_nested_state_fit": bool(
            phases.live_v4_binding["authenticated_before_microstep_execution"]
        ),
        "all_eight_v5_plus_carriers_authenticate_live_refit_and_gain_hashes": (
            phases.carrier_binding["carrier_fold_count"] == _EXPECTED_FOLDS
            and phases.carrier_binding[
                "all_carriers_authenticate_live_v4_delta_and_v5_plus_gain_hash"
            ]
            is True
        ),
        "v5_selected_arm_counts_are_exactly_six_plus_and_two_unit": (
            phases.carrier_binding["v5_plus_selected_fold_count"] == 6
            and phases.carrier_binding["v5_unit_selected_fold_count"] == 2
        ),
        "parent_endpoint_traces_folds_and_resources_recollected_exactly": True,
        "unit_k64_static_projection_and_cast_rows_replayed_before_vjp": True,
        "checkerboard_fit_role_has_one_example_per_eight_families": (
            len(phases.roles.fit_example_ids) == _EXPECTED_FOLDS
            and phases.roles.fit_support_tokens == 398
            and phases.roles.tune_support_tokens == 405
        ),
        "every_candidate_vjp_excludes_its_outer_held_family": all(
            row["family_id"] != row["held_family_id"]
            for row in phases.row_bank.gradient_receipts
        ),
        "all_parent_and_candidate_teacher_KL_vjps_have_zero_future_gradient": (
            causality_passed
        ),
        "every_row_sum_reconstructs_exact_v4_contraction_within_float64_tolerance": (
            len(row_errors) == _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
            and max(row_errors, default=0.0) <= 1.0e-12
        ),
        "all_eight_outer_and_56_inner_analytic_fits_present": (
            len(phases.fold_records) == _EXPECTED_FOLDS
            and sum(
                len(record.inner_family_records)
                for record in phases.fold_records
            )
            == _EXPECTED_FOLDS * _EXPECTED_INNER_FOLDS
        ),
        "all_56_inner_derivative_records_are_scalar_hash_inspectable": (
            sum(
                len(record.inner_family_records)
                for record in phases.fold_records
            )
            == 56
        ),
        "exact_model_forward_count_is_112": resources[
            "total_model_forward_count"
        ]
        == _EXPECTED_TOTAL_FORWARDS,
        "exact_backward_call_count_is_494": resources[
            "total_backward_call_count"
        ]
        == _EXPECTED_TOTAL_BACKWARDS,
        "zero_finite_state_candidate_forwards": resources[
            "finite_state_candidate_model_forward_count"
        ]
        == 0,
    }
    integrity_passed = all(integrity_gates.values())
    outcome = capacity_outcome if integrity_passed else "integrity_failure"
    passed = integrity_passed and outcome == "capacity_supported"
    primary_gates = {
        **integrity_gates,
        "feature_and_design_rank_four_condition_at_most_100_all_folds": bool(
            screen.feature_and_design_gate_passed
        ),
        "residual_conditional_fisher_energy_at_least_five_percent": bool(
            screen.residual_energy_gate_passed
        ),
        "at_least_six_of_eight_full_state_fits_are_non_noop": bool(
            screen.non_noop_gate_passed
        ),
        "at_least_42_of_56_inner_derivatives_negative_and_4_of_7_in_6_folds": bool(
            screen.negative_inner_global_gate_passed
            and screen.negative_inner_local_gate_passed
        ),
        "state_beats_identically_trained_scalar_in_at_least_six_folds": bool(
            screen.state_beats_scalar_gate_passed
        ),
        "median_inner_full_raw_slope_cosine_at_least_point_90_in_six_folds": bool(
            screen.cosine_stability_gate_passed
        ),
    }
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_hypothesis_use_only",
            "capacity_screen_only": True,
            "finite_state_candidate_forwards": 0,
            "frozen_tail_rank": CANDIDATE_GAIN_RANK,
            "frozen_outer_direction": (
                "v4_family_equal_mean_KL_OPG_delta_per_outer_fold"
            ),
            "base_step": _BASE_STEP,
            "base_step_hex": _BASE_STEP.hex(),
            "state_feature_rank": _STATE_FEATURE_RANK,
            "state_feature_source": (
                "raw_pre_gate_base_h4_support_rows_projected_on_first_four_"
                "frozen_held_fold_K64_fisher_directions"
            ),
            "state_feature_normalization": (
                "fit_family_equal_center_and_diagonal_scale"
            ),
            "row_amplitude_formula": "one_plus_tanh_z_dot_w",
            "analytic_derivative_definition": (
                "unit_point_linearization_incremental_to_executed_v5_plus_arm"
            ),
            "w_zero_exactly_reproduces_executed_v5_plus_one_over_64_arm": True,
            "carrier_is_v5_selected_arm_in_every_fold": False,
            "counterfactual_plus_carrier_used_in_v5_unit_selected_folds": 2,
            "state_field_learned_parameter_count": _STATE_FEATURE_RANK,
            "prompt_static_control_learned_parameter_count": 1,
            "outer_family_fold_count": _EXPECTED_FOLDS,
            "nested_held_inner_family_count_per_outer_fold": (
                _EXPECTED_INNER_FOLDS
            ),
            "inner_codec_and_state_fit_family_count": 6,
            "full_codec_and_state_fit_family_count": 7,
            "candidate_vjp_chunk_size": token_v1._VJP_CHUNK_SIZE,
            "posthoc_hyperparameter_search_performed": False,
            "finite_execution_required_before_selection_or_serving": True,
        },
        "expanded_parent_binding": {
            "file": str(expanded_parent_report_path),
            "file_sha256": v3diag.EXPANDED_PARENT_REPORT_FILE_SHA256,
            "report_sha256": v3diag.EXPANDED_PARENT_REPORT_SHA256,
            "schema": parent.get("schema"),
            "classification": parent.get("classification"),
            "k320_reexecuted": False,
        },
        "control_lineage_binding": {
            "v3_file": str(v3_report_path),
            "v4_file": str(v4_report_path),
            "v5_file": str(v5_report_path),
            **dict(phases.lineage_binding),
        },
        "live_v4_refit_and_gradient_binding": phases.live_v4_binding,
        "v5_plus_carrier_binding": phases.carrier_binding,
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": (
                token_v1.MATERIALIZATION_REPORT_FILE_SHA256
            ),
            "materialization_report_sha256": (
                token_v1.MATERIALIZATION_REPORT_SHA256
            ),
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": token_v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": token_v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
            "parent_recollection_receipt_sha256": phases.recollection_receipt,
            "unit_k64_static_replay_receipt_sha256": (
                phases.static_unit_replay_receipt
            ),
        },
        "prompt_role_receipt": {
            "artifact_sha256": phases.roles.artifact_sha256,
            "fit_example_ids": phases.roles.fit_example_ids,
            "fit_support_supervised_token_count": phases.roles.fit_support_tokens,
            "tune_role_used_for_model_execution": False,
            "family_order": families,
            "outer_held_family_excluded_from_every_live_vjp": True,
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": _endpoint_prompt_receipts(traces),
        "candidate_gain_refits": tuple(
            phases.row_bank.refits[family].metadata() for family in families
        ),
        "candidate_gradient_receipts": phases.row_bank.gradient_receipts,
        "candidate_gradient_receipt_set_sha256": phases.live_v4_binding[
            "v4_candidate_gradient_receipt_set_sha256"
        ],
        "row_resolved_vjp_receipts": phases.row_bank.row_bank_receipts,
        "analytic_fold_records": tuple(
            record.metadata() for record in phases.fold_records
        ),
        "analytic_inner_family_records": tuple(
            inner.metadata()
            for record in phases.fold_records
            for inner in record.inner_family_records
        ),
        "analytic_capacity_screen": screen.metadata(),
        "outcome_matrix": {
            "outcome": outcome,
            "detailed_analytic_outcome": detailed_outcome,
            "capacity_supported": outcome == "capacity_supported",
            "finite_validation_authorized_next": outcome
            == "capacity_supported",
            "selection_or_serving_authorized": False,
        },
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "passed": passed,
        "classification": outcome,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "native_teacher_logits_and_held_native_tails_used": True,
            "tail_basis_and_order_are_family_disjoint": True,
            "outer_gain_fit_excludes_outer_held_family": True,
            "nested_state_fit_excludes_inner_held_family": True,
            "fit_family_equal_normalization_used": True,
            "identically_trained_one_scalar_control_used": True,
            "executed_v5_plus_arm_is_nominal_carrier_in_all_eight_folds": True,
            "v5_selected_arm_is_not_claimed_as_nominal_carrier": True,
            "counterfactual_plus_carrier_used_in_v5_unit_selected_folds": 2,
            "state_conditioned_candidate_executed": False,
            "finite_teacher_KL_or_behavioral_fidelity_validated": False,
            "analytic_derivative_is_not_finite_displacement_authority": True,
            "capacity_supported_means_only_eligible_for_finite_validation": True,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
        },
        "safety": _safety_metadata(),
    }
    return _publish(report, output=destination)


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(report, domain=_REPORT_DOMAIN)
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": token_v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """Return the deliberately no-knob v6 command-line interface."""

    return argparse.ArgumentParser(
        description=(
            "Run the pinned same-A K64 four-feature state-gain capacity screen."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
