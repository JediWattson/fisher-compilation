"""Analytic joint scalar-plus-state K64 gain-capacity screen.

This v7 rung authenticates and exactly reproduces the pinned v6 state-only
capacity analysis, then asks the missing nested question: can one joint
five-parameter amplitude logit ``u + z @ w`` outperform the exact scalar-only
comparator on held inner families?  The frozen v4 K64 delta and the executed
v5 plus-one-over-64 carrier are unchanged.  All joint fitting is analytic;
no joint candidate is executed and no serving, compression, or speed claim
is authorized.
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
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic as v6diag
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from .complete_h4_tail_candidate_gain_refit_v4 import CANDIDATE_GAIN_RANK
from .complete_h4_tail_candidate_joint_state_gain_field import (
    CandidateConditionedK64JointStateGainAnalyticScreen,
    CandidateConditionedK64JointStateGainFoldAnalyticRecord,
    build_candidate_conditioned_k64_joint_inner_family_analytic_record,
    build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record,
    fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control,
    screen_candidate_conditioned_k64_joint_state_gain_capacity,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    fit_complete_h4_tail_held_family,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import _load_committed_basis
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_EXPANDED_PARENT_REPORT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRANSFER_REPORT",
    "DEFAULT_V3_REPORT",
    "DEFAULT_V4_REPORT",
    "DEFAULT_V5_REPORT",
    "DEFAULT_V6_REPORT",
    "V6_REPORT_FILE_SHA256",
    "V6_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v6diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v6diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v6diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v6diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v6diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v6diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v6diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-gain-capacity-lofo-a-fit16-dev-v7.json"
)

V6_REPORT_FILE_SHA256 = (
    "95e49e10f802121436f081f381c8622653e79767fc2ff4d265b4c976b8193a28"
)
V6_REPORT_SHA256 = (
    "001738439b4c2bdd052f6220283103c592b96742b27f105778584e2529a758eb"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_gain_capacity_lofo.v7"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-joint-state-gain-capacity:v7\0"
_V6_BINDING_DOMAIN = (
    b"fisher-graph:complete-h4-k64-joint-state-gain-v6-binding:v7\0"
)
_V6_SCALAR_BINDING_DOMAIN = (
    b"fisher-graph:complete-h4-k64-joint-state-gain-v6-scalar-binding:v7\0"
)
_EXPECTED_OUTER_FOLDS = 8
_EXPECTED_INNER_FOLDS = 7
_EXPECTED_INNER_RECORDS = 56
_EXPECTED_TOTAL_FORWARDS = 112
_EXPECTED_TOTAL_BACKWARDS = 494


def _canonical(value: object) -> object:
    return v3diag._canonical(value)


def _load_v6_report(path: Path | str) -> dict[str, object]:
    """Load only the exact live v6 scalar-attribution failure."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V6_REPORT_FILE_SHA256,
        expected_report_sha256=V6_REPORT_SHA256,
        label="candidate state-gain capacity v6 control",
    )
    if (
        report.get("schema") != v6diag._SCHEMA
        or report.get("classification") != "state_not_better_than_scalar"
        or report.get("passed") is not False
    ):
        raise RuntimeError("candidate state-gain capacity v6 control differs")
    return report


def _authenticate_live_v6_evidence(
    *,
    v6_report: Mapping[str, object],
    phases: v6diag._CapacityPhaseResults,
) -> dict[str, object]:
    """Require exact live reproduction of every scalar/hash v6 result row."""

    expected_screen = v6_report.get("analytic_capacity_screen")
    expected_folds = v6_report.get("analytic_fold_records")
    expected_inner = v6_report.get("analytic_inner_family_records")
    expected_refits = v6_report.get("candidate_gain_refits")
    expected_gradients = v6_report.get("candidate_gradient_receipts")
    expected_rows = v6_report.get("row_resolved_vjp_receipts")
    expected_live_v4 = v6_report.get("live_v4_refit_and_gradient_binding")
    expected_carriers = v6_report.get("v5_plus_carrier_binding")
    expected_resources = v6_report.get("resources")
    if (
        not isinstance(expected_screen, Mapping)
        or not isinstance(expected_folds, list)
        or not isinstance(expected_inner, list)
        or not isinstance(expected_refits, list)
        or not isinstance(expected_gradients, list)
        or not isinstance(expected_rows, list)
        or not isinstance(expected_live_v4, Mapping)
        or not isinstance(expected_carriers, Mapping)
        or not isinstance(expected_resources, Mapping)
    ):
        raise ValueError("pinned v6 analytic evidence differs")
    live_folds = tuple(record.metadata() for record in phases.fold_records)
    live_inner = tuple(
        inner.metadata()
        for record in phases.fold_records
        for inner in record.inner_family_records
    )
    live_refits = tuple(
        phases.row_bank.refits[family].metadata()
        for family in sorted(phases.row_bank.refits)
    )
    comparisons = (
        (phases.screen.metadata(), expected_screen, "screen"),
        (live_folds, expected_folds, "fold records"),
        (live_inner, expected_inner, "inner records"),
        (live_refits, expected_refits, "v4 refits"),
        (phases.row_bank.gradient_receipts, expected_gradients, "v4 receipts"),
        (phases.row_bank.row_bank_receipts, expected_rows, "row receipts"),
        (phases.live_v4_binding, expected_live_v4, "live v4 binding"),
        (phases.carrier_binding, expected_carriers, "v5 carrier binding"),
    )
    for live, expected, label in comparisons:
        if _canonical(live) != _canonical(expected):
            raise RuntimeError(f"live v7 recollection did not reproduce v6 {label}")
    if (
        len(live_folds) != _EXPECTED_OUTER_FOLDS
        or len(live_inner) != _EXPECTED_INNER_RECORDS
        or phases.screen.outcome != "fail_state_vs_scalar_attribution"
        or phases.screen.capacity_screen_passed
        or expected_resources.get("total_model_forward_count")
        != _EXPECTED_TOTAL_FORWARDS
        or expected_resources.get("total_backward_call_count")
        != _EXPECTED_TOTAL_BACKWARDS
        or expected_resources.get("finite_state_candidate_model_forward_count")
        != 0
    ):
        raise RuntimeError("pinned v6 analytic result geometry differs")
    payload = {
        "v6_report_file_sha256": V6_REPORT_FILE_SHA256,
        "v6_report_sha256": V6_REPORT_SHA256,
        "v6_screen_artifact_sha256": phases.screen.artifact_sha256,
        "v6_fold_artifact_sha256s": tuple(
            record.artifact_sha256 for record in phases.fold_records
        ),
        "v6_inner_artifact_sha256s": tuple(
            inner.artifact_sha256
            for record in phases.fold_records
            for inner in record.inner_family_records
        ),
        "v6_live_refit_count": len(live_refits),
        "v6_live_gradient_receipt_count": len(
            phases.row_bank.gradient_receipts
        ),
        "v6_live_row_receipt_count": len(phases.row_bank.row_bank_receipts),
        "screen_metadata_canonically_equal": True,
        "fold_metadata_canonically_equal": True,
        "all_56_inner_metadata_rows_canonically_equal": True,
        "v4_refits_and_receipts_canonically_equal": True,
        "v5_carriers_canonically_equal": True,
        "authenticated_before_joint_fit": True,
    }
    return {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V6_BINDING_DOMAIN
        ),
    }


def _fit_nested_joint_capacity_screen(
    *,
    v6_phases: v6diag._CapacityPhaseResults,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> tuple[
    CandidateConditionedK64JointStateGainAnalyticScreen,
    tuple[CandidateConditionedK64JointStateGainFoldAnalyticRecord, ...],
    dict[str, object],
]:
    """Fit 8 full and 56 nested joint5D rows with exact v6 scalars."""

    v6_folds = {
        record.outer_held_family_id: record for record in v6_phases.fold_records
    }
    if set(v6_folds) != set(fits) or len(v6_folds) != _EXPECTED_OUTER_FOLDS:
        raise RuntimeError("live v6 fold grid differs before joint fitting")
    fold_records: list[CandidateConditionedK64JointStateGainFoldAnalyticRecord] = []
    scalar_receipts: list[dict[str, object]] = []
    for outer_family in sorted(fits):
        fit = fits[outer_family]
        refit = v6_phases.row_bank.refits[outer_family]
        directions = v6diag._ordered_k64(fit)
        delta = v6diag._mean_delta(refit)
        outer_cells = tuple(
            sorted(
                v6_phases.row_bank.cells[outer_family],
                key=lambda cell: cell.family_id,
            )
        )
        if (
            len(outer_cells) != _EXPECTED_INNER_FOLDS
            or {cell.family_id for cell in outer_cells}
            != set(refit.training_family_ids)
            or any(cell.held_family_id != outer_family for cell in outer_cells)
        ):
            raise RuntimeError("v7 outer row-bank family grid differs")
        v6_fold = v6_folds[outer_family]
        full_codec = v6diag.fit_candidate_conditioned_k64_state_feature_codec(
            tuple(v6diag._feature_example(cell) for cell in outer_cells),
            held_family_id=outer_family,
            ordered_directions=directions,
        )
        full_examples = tuple(
            v6diag._state_gradient_example(
                cell=cell,
                codec=full_codec,
                ordered_directions=directions,
                mean_gain_delta=delta,
            )
            for cell in outer_cells
        )
        full_joint, full_scalar = (
            fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control(
                refit,
                full_codec,
                full_examples,
                ordered_directions=directions,
            )
        )
        expected_full_scalar = v6_fold.full_static_control_fit
        if _canonical(full_scalar.metadata()) != _canonical(
            expected_full_scalar.metadata()
        ):
            raise RuntimeError("v7 full scalar comparator did not reproduce v6")
        full_receipt = {
            "outer_held_family_id": outer_family,
            "split": "full_seven",
            "inner_held_family_id": None,
            "v6_scalar_fit_artifact_sha256": (
                expected_full_scalar.artifact_sha256
            ),
            "live_scalar_fit_artifact_sha256": full_scalar.artifact_sha256,
            "joint_fit_artifact_sha256": full_joint.artifact_sha256,
            "scalar_metadata_canonically_equal": True,
            "raw_tensors_serialized": False,
        }
        full_receipt["receipt_sha256"] = token_v1._domain_sha256(
            full_receipt, domain=_V6_SCALAR_BINDING_DOMAIN
        )
        scalar_receipts.append(full_receipt)

        v6_inner_by_family = {
            record.inner_held_family_id: record
            for record in v6_fold.inner_family_records
        }
        if set(v6_inner_by_family) != {cell.family_id for cell in outer_cells}:
            raise RuntimeError("live v6 inner-family grid differs")
        inner_records: list[object] = []
        for inner_family in sorted(v6_inner_by_family):
            training_cells = tuple(
                cell for cell in outer_cells if cell.family_id != inner_family
            )
            held_cell = next(
                cell for cell in outer_cells if cell.family_id == inner_family
            )
            inner_codec = v6diag.fit_candidate_conditioned_k64_state_feature_codec(
                tuple(v6diag._feature_example(cell) for cell in training_cells),
                held_family_id=outer_family,
                ordered_directions=directions,
            )
            inner_examples = tuple(
                v6diag._state_gradient_example(
                    cell=cell,
                    codec=inner_codec,
                    ordered_directions=directions,
                    mean_gain_delta=delta,
                )
                for cell in training_cells
            )
            inner_joint, inner_scalar = (
                fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control(
                    refit,
                    inner_codec,
                    inner_examples,
                    ordered_directions=directions,
                )
            )
            expected_inner_scalar = v6_inner_by_family[
                inner_family
            ].inner_static_control_fit
            if _canonical(inner_scalar.metadata()) != _canonical(
                expected_inner_scalar.metadata()
            ):
                raise RuntimeError(
                    "v7 inner scalar comparator did not reproduce v6"
                )
            held_example = v6diag._state_gradient_example(
                cell=held_cell,
                codec=inner_codec,
                ordered_directions=directions,
                mean_gain_delta=delta,
            )
            inner_records.append(
                build_candidate_conditioned_k64_joint_inner_family_analytic_record(
                    full_joint,
                    inner_joint,
                    inner_scalar,
                    held_example,
                )
            )
            receipt = {
                "outer_held_family_id": outer_family,
                "split": "inner_six",
                "inner_held_family_id": inner_family,
                "v6_scalar_fit_artifact_sha256": (
                    expected_inner_scalar.artifact_sha256
                ),
                "live_scalar_fit_artifact_sha256": inner_scalar.artifact_sha256,
                "joint_fit_artifact_sha256": inner_joint.artifact_sha256,
                "scalar_metadata_canonically_equal": True,
                "raw_tensors_serialized": False,
            }
            receipt["receipt_sha256"] = token_v1._domain_sha256(
                receipt, domain=_V6_SCALAR_BINDING_DOMAIN
            )
            scalar_receipts.append(receipt)
        if len(inner_records) != _EXPECTED_INNER_FOLDS:
            raise RuntimeError("v7 inner joint record count differs")
        fold_records.append(
            build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record(
                full_joint,
                full_scalar,
                tuple(inner_records),
            )
        )
    if (
        len(fold_records) != _EXPECTED_OUTER_FOLDS
        or len(scalar_receipts)
        != _EXPECTED_OUTER_FOLDS + _EXPECTED_INNER_RECORDS
    ):
        raise RuntimeError("v7 joint/scalar analytic grid differs")
    screen = screen_candidate_conditioned_k64_joint_state_gain_capacity(
        tuple(fold_records)
    )
    binding_payload = {
        "scalar_comparator_receipts": tuple(scalar_receipts),
        "full_scalar_comparator_count": _EXPECTED_OUTER_FOLDS,
        "inner_scalar_comparator_count": _EXPECTED_INNER_RECORDS,
        "total_scalar_comparator_count": len(scalar_receipts),
        "every_scalar_comparator_canonically_reproduces_v6": True,
        "scalar_comparator_refit_or_retuned_for_joint": False,
        "raw_tensors_serialized": False,
    }
    binding_payload["artifact_sha256"] = token_v1._domain_sha256(
        binding_payload, domain=_V6_SCALAR_BINDING_DOMAIN
    )
    return screen, tuple(fold_records), binding_payload


@dataclass(slots=True)
class _JointPhaseResults:
    v6_phases: v6diag._CapacityPhaseResults
    v6_binding: Mapping[str, object]
    joint_screen: CandidateConditionedK64JointStateGainAnalyticScreen
    joint_fold_records: tuple[
        CandidateConditionedK64JointStateGainFoldAnalyticRecord, ...
    ]
    scalar_comparator_binding: Mapping[str, object]


def _execute_joint_phases(
    *,
    context: object,
    parent: Mapping[str, object],
    v3_report: Mapping[str, object],
    v4_report: Mapping[str, object],
    v5_report: Mapping[str, object],
    v6_report: Mapping[str, object],
    traces: Sequence[object],
    endpoint_resources: Mapping[str, int],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> _JointPhaseResults:
    """Reproduce all v6 evidence before constructing any joint fit."""

    v6_phases = v6diag._execute_capacity_phases(
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
    v6_binding = _authenticate_live_v6_evidence(
        v6_report=v6_report, phases=v6_phases
    )
    joint_screen, joint_folds, scalar_binding = (
        _fit_nested_joint_capacity_screen(
            v6_phases=v6_phases,
            fits=fits,
        )
    )
    return _JointPhaseResults(
        v6_phases=v6_phases,
        v6_binding=v6_binding,
        joint_screen=joint_screen,
        joint_fold_records=joint_folds,
        scalar_comparator_binding=scalar_binding,
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
        parent_forwards != 48
        or parent_backwards != 109
        or gradient_forwards != 64
        or gradient_resources["gradient_candidate_vjp_backward_call_count"]
        != 385
        or total_forwards != _EXPECTED_TOTAL_FORWARDS
        or total_backwards != _EXPECTED_TOTAL_BACKWARDS
    ):
        raise RuntimeError("v7 joint analytic resource accounting differs")
    return {
        **endpoint_resources,
        **gradient_resources,
        "phase_order": (
            "parent_endpoint_recollection",
            "static_unit_k64_reconstruction",
            "row_resolved_unit_candidate_vjp_recollection",
            "exact_v4_refit_and_receipt_authentication",
            "exact_v5_plus_carrier_authentication",
            "exact_v6_screen_fold_and_inner_reproduction",
            "nested_held_inner_family_joint5d_analytic_capacity_screen",
            "scalar_hash_only_report_publication",
        ),
        "parent_collection_model_forward_count": parent_forwards,
        "parent_collection_backward_call_count": parent_backwards,
        "gradient_stage_model_forward_count": gradient_forwards,
        "v6_reproduction_additional_model_forward_count": 0,
        "joint_fit_model_forward_count": 0,
        "finite_joint_candidate_model_forward_count": 0,
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
        "contains_joint_parameter_vectors": False,
        "contains_basis_coefficients": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


def _joint_gate_results(
    screen: CandidateConditionedK64JointStateGainAnalyticScreen,
) -> dict[str, bool]:
    return {
        "augmented_feature_and_joint_design_rank_five_condition_at_most_100_all_fits": (
            screen.feature_and_design_gate_passed
        ),
        "residual_conditional_state_fisher_at_least_five_percent_in_six_folds": (
            screen.residual_energy_gate_passed
        ),
        "at_least_six_of_eight_full_joint_fits_are_non_noop": (
            screen.non_noop_gate_passed
        ),
        "at_least_42_of_56_joint_derivatives_negative_and_4_of_7_in_six_folds": (
            screen.negative_inner_global_gate_passed
            and screen.negative_inner_local_gate_passed
        ),
        "joint_inner_macro_strictly_beats_exact_v6_scalar_in_six_folds": (
            screen.joint_beats_scalar_gate_passed
        ),
        "median_inner_full_state_slope_cosine_at_least_point_90_in_six_folds": (
            screen.cosine_stability_gate_passed
        ),
    }


def _integrity_gate_results(
    *,
    phases: _JointPhaseResults,
    traces: Sequence[object],
    resources: Mapping[str, object],
    joint_inner_record_count: int,
) -> dict[str, bool]:
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    ) and all(
        row["maximum_future_gradient_abs"] == 0.0
        and row["future_gradient_nonzero_count"] == 0
        for row in phases.v6_phases.row_bank.gradient_receipts
    )
    return {
        "expanded_v2_parent_authenticated": True,
        "v3_v4_v5_v6_controls_authenticated_by_exact_file_and_report_sha256": True,
        "live_recollection_canonically_reproduced_exact_v6_screen": (
            phases.v6_binding["screen_metadata_canonically_equal"] is True
        ),
        "live_recollection_canonically_reproduced_all_eight_v6_folds": (
            phases.v6_binding["fold_metadata_canonically_equal"] is True
        ),
        "live_recollection_canonically_reproduced_all_56_v6_inner_rows": (
            phases.v6_binding[
                "all_56_inner_metadata_rows_canonically_equal"
            ]
            is True
        ),
        "live_recollection_reproduced_v4_refits_receipts_and_v5_carriers": (
            phases.v6_binding["v4_refits_and_receipts_canonically_equal"]
            is True
            and phases.v6_binding["v5_carriers_canonically_equal"] is True
        ),
        "all_v6_evidence_authenticated_before_any_joint_fit": (
            phases.v6_binding["authenticated_before_joint_fit"] is True
        ),
        "every_full_and_inner_scalar_comparator_exactly_reproduces_v6": (
            phases.scalar_comparator_binding[
                "every_scalar_comparator_canonically_reproduces_v6"
            ]
            is True
            and phases.scalar_comparator_binding["total_scalar_comparator_count"]
            == _EXPECTED_OUTER_FOLDS + _EXPECTED_INNER_RECORDS
        ),
        "all_eight_outer_and_56_inner_joint_analytic_records_present": (
            len(phases.joint_fold_records) == _EXPECTED_OUTER_FOLDS
            and joint_inner_record_count == _EXPECTED_INNER_RECORDS
        ),
        "all_parent_and_candidate_teacher_KL_vjps_have_zero_future_gradient": (
            causality_passed
        ),
        "exact_model_forward_count_is_112": resources[
            "total_model_forward_count"
        ]
        == _EXPECTED_TOTAL_FORWARDS,
        "exact_backward_call_count_is_494": resources[
            "total_backward_call_count"
        ]
        == _EXPECTED_TOTAL_BACKWARDS,
        "zero_finite_joint_candidate_forwards": resources[
            "finite_joint_candidate_model_forward_count"
        ]
        == 0,
    }


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    v5_report_path: Path | str = DEFAULT_V5_REPORT,
    v6_report_path: Path | str = DEFAULT_V6_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the locked same-A v7 joint5D analytic capacity screen."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite candidate joint state-gain v7 report"
        )
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = v4diag._load_v3_report(v3_report_path)
    v4_report = v5diag._load_v4_report(v4_report_path)
    v5_report = v6diag._load_v5_report(v5_report_path)
    v6_report = _load_v6_report(v6_report_path)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate joint state-gain v7 rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate joint state-gain v7 rank320 transfer",
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
            raise RuntimeError("candidate joint state-gain v7 A16 panel differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        phases = _execute_joint_phases(
            context=context,
            parent=parent,
            v3_report=v3_report,
            v4_report=v4_report,
            v5_report=v5_report,
            v6_report=v6_report,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    v6_phases = phases.v6_phases
    resources = _resource_accounting(
        endpoint_resources=endpoint_resources,
        gradient_resources=v6_phases.row_bank.resources,
    )
    joint_screen_metadata = phases.joint_screen.metadata()
    joint_inner_records = tuple(
        inner.metadata()
        for record in phases.joint_fold_records
        for inner in record.inner_family_records
    )
    integrity_gates = _integrity_gate_results(
        phases=phases,
        traces=traces,
        resources=resources,
        joint_inner_record_count=len(joint_inner_records),
    )
    integrity_passed = all(integrity_gates.values())
    joint_gates = _joint_gate_results(phases.joint_screen)
    capacity_passed = all(joint_gates.values())
    if capacity_passed is not bool(
        joint_screen_metadata["capacity_screen_passed"]
    ):
        raise RuntimeError("v7 diagnostic and pure-core gate results differ")
    detailed_outcome = str(joint_screen_metadata["outcome"])
    classification = detailed_outcome if integrity_passed else "integrity_failure"
    passed = integrity_passed and capacity_passed
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_hypothesis_use_only",
            "capacity_screen_only": True,
            "finite_joint_candidate_forwards": 0,
            "frozen_tail_rank": CANDIDATE_GAIN_RANK,
            "frozen_outer_direction": (
                "v4_family_equal_mean_KL_OPG_delta_per_outer_fold"
            ),
            "nominal_carrier": "executed_v5_plus_one_over_64_all_eight_folds",
            "joint_amplitude_logit": "u_plus_z_dot_w",
            "joint_amplitude_formula": "one_plus_tanh_u_plus_z_dot_w",
            "joint_learned_parameter_count": 5,
            "scalar_intercept_parameter_count": 1,
            "state_slope_parameter_count": 4,
            "joint_fit": "family_equal_uncentered_augmented_5D_OPG",
            "joint_tangent_design": "concatenate_static_q_and_state_J",
            "single_joint_logit_RMS_trust_bound": True,
            "separately_fitted_scalar_and_state_weights_combined": False,
            "scalar_comparator": "exact_v6_one_parameter_fit_per_split",
            "outer_family_fold_count": _EXPECTED_OUTER_FOLDS,
            "nested_held_inner_family_count_per_outer_fold": (
                _EXPECTED_INNER_FOLDS
            ),
            "full_fit_family_count": 7,
            "inner_fit_family_count": 6,
            "joint_cell_wins_reported_but_not_a_primary_gate": True,
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
        "v6_control_binding": {
            "file": str(v6_report_path),
            "file_sha256": V6_REPORT_FILE_SHA256,
            "report_sha256": V6_REPORT_SHA256,
            "schema": v6_report.get("schema"),
            "classification": v6_report.get("classification"),
            "live_evidence_reproduction": phases.v6_binding,
        },
        "v6_scalar_comparator_binding": phases.scalar_comparator_binding,
        "live_v4_refit_and_gradient_binding": v6_phases.live_v4_binding,
        "v5_plus_carrier_binding": v6_phases.carrier_binding,
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
            "parent_recollection_receipt_sha256": (
                v6_phases.recollection_receipt
            ),
            "unit_k64_static_replay_receipt_sha256": (
                v6_phases.static_unit_replay_receipt
            ),
        },
        "prompt_role_receipt": {
            "artifact_sha256": v6_phases.roles.artifact_sha256,
            "fit_example_ids": v6_phases.roles.fit_example_ids,
            "fit_support_supervised_token_count": v6_phases.roles.fit_support_tokens,
            "tune_role_used_for_model_execution": False,
            "family_order": families,
            "outer_and_inner_held_families_excluded_from_their_fits": True,
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": v6diag._endpoint_prompt_receipts(traces),
        "candidate_gain_refits": tuple(
            v6_phases.row_bank.refits[family].metadata() for family in families
        ),
        "candidate_gradient_receipts": v6_phases.row_bank.gradient_receipts,
        "candidate_gradient_receipt_set_sha256": v6_phases.live_v4_binding[
            "v4_candidate_gradient_receipt_set_sha256"
        ],
        "row_resolved_vjp_receipts": v6_phases.row_bank.row_bank_receipts,
        "joint_analytic_fold_records": tuple(
            record.metadata() for record in phases.joint_fold_records
        ),
        "joint_analytic_inner_family_records": joint_inner_records,
        "joint_analytic_capacity_screen": joint_screen_metadata,
        "outcome_matrix": {
            "outcome": classification,
            "detailed_joint_analytic_outcome": detailed_outcome,
            "joint_capacity_supported": passed,
            "finite_validation_authorized_next": passed,
            "selection_or_serving_authorized": False,
        },
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "joint_capacity_gate_results": tuple(sorted(joint_gates.items())),
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "native_teacher_logits_and_held_native_tails_used": True,
            "tail_basis_and_order_are_family_disjoint": True,
            "outer_gain_fit_excludes_outer_held_family": True,
            "nested_joint_fit_excludes_inner_held_family": True,
            "exact_v6_scalar_comparator_used_without_retuning": True,
            "joint_state_candidate_executed": False,
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
    """Return the deliberately no-knob v7 command-line interface."""

    return argparse.ArgumentParser(
        description=(
            "Run the pinned same-A K64 joint scalar-plus-state capacity screen."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
