"""Exact replay analysis for the sign-dispatched two-expert Gemma repair.

Iteration three leaves the accepted-X4 plus lag-B parent frozen and replaces
iteration two's single 2-by-2 route with two independently bounded 2-by-2
experts.  The sign of the same causal cumulative-balance feature dispatches
each active row to exactly one expert.  The live campaign owns tensors and
model execution; this module independently replays the scalar-only fit,
leave-one-family-out providers, exact finite observations, resource envelope,
and retained full-data refit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
    _fidelity_from_observations,
    _paired_comparison,
)
from .gemma3_l3_l4_iterative_state_experts import (
    GemmaIterativeStateExpertsFitRecord,
    fit_gemma_iterative_state_experts_fold,
    gemma_causal_top2_state_experts_provider_artifact_sha256,
)
from .gemma3_l3_l4_iterative_state_router_analysis import (
    _assert_scalar_hash_only,
    _canonical_json_bytes,
    _canonical_observations,
    _correct_prompt_disagreement_quantile_labels,
    _finite,
    _identifier,
    _int2,
    _mapping,
    _observation_dict,
    _observation_from_dict,
    _relative_improvement,
    _require_sha256,
    _sha256,
    _source_grid,
    _validate_manifest,
)


__all__ = [
    "build_gemma_iterative_state_experts_report",
    "validate_gemma_iterative_state_experts_report",
]


_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_state_experts_analysis.sign_v1"
)
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma-iterative-state-experts-sign-v1-report:v1\0"
)
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-state-experts-sign-v1-collection:v1\0"
)
_RESOURCE_DOMAIN = b"fisher-graph:gemma-iterative-residual-resources:v1\0"
_RETENTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-retained-provider:v1\0"
)
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PER_FAMILY = 2
_EXPERT_REGIME_ORDER = ("negative", "nonnegative")
_EXPERT_ROUTE_MATRIX_SHAPE = (2, 2, 2)
_EXPERT_ROUTE_EDGE_ORDER = (
    "negative_0_to_0",
    "negative_0_to_1",
    "negative_1_to_0",
    "negative_1_to_1",
    "nonnegative_0_to_0",
    "nonnegative_0_to_1",
    "nonnegative_1_to_0",
    "nonnegative_1_to_1",
)
_ROUTE_STATE_SEMANTICS = (
    "top2_parent_lag_b_modal_cumulative_balance_v1"
)
_EXPERT_DISPATCH_SEMANTICS = (
    "negative_if_balance_lt_0_else_nonnegative"
)
_TRUST_PROJECTION_INTERPRETATION = (
    "independent_expert_operator_norm_projection_is_linearization_"
    "extrapolation"
)
_MACRO_RELATIVE_IMPROVEMENT_MIN = -0.02
_MINIMUM_FAMILY_WIN_COUNT = 6
_WORST_FAMILY_RELATIVE_IMPROVEMENT_MIN = -0.02
_BALANCE_FEATURE_STD_MIN = 0.05
_TOP2_MODAL_ENERGY_FRACTION_MIN = 0.5
_REGIME_FAMILY_MACRO_ACTIVE_FRACTION_MIN = 0.10
_REQUIRED_LINEAGE_KEYS = frozenset(
    {
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "accepted_x4_head_sha256",
        "bridge_binding_sha256",
        "model_sha256",
        "adapter_execution_sha256",
        "fit_manifest_sha256",
        "factorial_report_sha256",
        "factorial_report_file_sha256",
        "prior_iteration_report_sha256",
        "prior_iteration_report_file_sha256",
        "prior_iteration_collection_sha256",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "learned_parameter_count",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_count_per_sequence",
        "derived_constant_float_count",
        "nonlinear_scalar_ops_per_token_upper_bound",
        "serving_model_forward_count",
        "parent_head_reused_not_duplicated",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "candidate_provider_artifact_sha256_by_family",
        "residual_width",
        "resource_receipt_sha256",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "execution_mode",
        "example_count",
        "family_count",
        "outer_fold_count",
        "expert_route_matrix_shape",
        "expert_regime_order",
        "expert_route_edge_order",
        "route_state_semantics",
        "expert_dispatch_semantics",
        "routed_parent_decoder_mode_indices",
        "phase_a_source_forward_count",
        "phase_a_parent_vjp_forward_count",
        "phase_b_source_forward_count",
        "phase_b_candidate_forward_count",
        "total_model_forward_count",
        "model_forward_count_per_example",
        "one_semantic_candidate_per_iteration",
        "family_blocked_leave_one_family_out",
        "source_rerun_between_phases",
        "source_identity_equal_across_phases",
        "parent_observation_count",
        "candidate_observation_count",
        "fit_record_count",
        "fit_records_scalar_hash_only",
        "candidate_executions_released_between_examples",
        "raw_prompts_retained",
        "raw_token_ids_retained",
        "raw_logits_retained",
        "raw_activations_retained",
        "gradient_tensors_retained",
        "model_weights_retained",
        "source_model_sha256",
        "source_execution_sha256",
        "parent_artifact_sha256",
        "parent_h4_artifact_sha256",
        "parent_h4_decoder_sha256",
        "parent_h4_lag_kernel_sha256",
        "accepted_x4_head_sha256",
        "fit_manifest_sha256",
        "residual_width",
        "parent_prepared_float_scalar_count",
        "parent_logical_macs_per_token_upper_bound",
        "bridge_binding_sha256",
        "parent_execution_sha256s",
        "parent_execution_sha256_by_example",
        "model_inputs_sha256_by_example",
        "candidate_execution_sha256s",
        "candidate_execution_sha256_by_example",
        "fold_provider_artifact_sha256s",
        "fold_provider_artifact_sha256_by_family",
        "fold_trust_projection_count",
        "trust_projection_interpretation",
        "selection_input_opened",
        "guard_input_opened",
        "calibration_b_opened",
        "assessment_input_opened",
        "development_only",
    }
)
_SAFETY = {
    "development_only": True,
    "reusable_development_inputs_only": True,
    "source_outputs_authoritative": True,
    "candidate_outputs_metrics_only": True,
    "selection_input_opened": False,
    "guard_input_opened": False,
    "calibration_b_opened": False,
    "assessment_input_opened": False,
    "raw_prompts_retained": False,
    "raw_token_ids_retained": False,
    "raw_logits_retained": False,
    "raw_activations_retained": False,
    "gradient_tensors_retained": False,
    "model_weights_retained": False,
    "deployment_claim": False,
    "generalization_claim": False,
    "compression_qualification_claim": False,
}


def _float8(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 8:
        raise ValueError(f"{label} must contain exactly eight values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _float2(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _nonnegative_int2(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(
            f"{label} must contain two nonnegative integer values"
        )
    return tuple(value)  # type: ignore[return-value]


def _bool2(value: object, *, label: str) -> tuple[bool, bool]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not bool for item in value)
    ):
        raise ValueError(f"{label} must contain exactly two booleans")
    return tuple(value)  # type: ignore[return-value]


def _fit_record_dict(value: object) -> dict[str, object]:
    if isinstance(value, GemmaIterativeStateExpertsFitRecord):
        return value.to_dict()
    row = _mapping(value, label="state-experts fit record")
    expected = {
        "example_id",
        "family_id",
        "model_inputs_sha256",
        "parent_execution_sha256",
        "parent_observation_sha256",
        "parent_h4_artifact_sha256",
        "prefix_sha256",
        "gradient_sha256",
        "parent_modal_sha256",
        "balance_feature_sha256",
        "negative_gated_modal_sha256",
        "nonnegative_gated_modal_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_expert_route_edge",
        "active_row_count",
        "active_row_count_by_expert",
        "active_expert_mask",
        "jacobian_support_by_expert",
        "top_mode_indices",
        "top_mode_norms",
        "balance_feature_std",
        "top2_modal_energy_fraction",
        "fit_record_sha256",
    }
    if set(row) != expected:
        raise ValueError("state-experts fit-record fields differ")
    result = GemmaIterativeStateExpertsFitRecord(
        example_id=row["example_id"],  # type: ignore[arg-type]
        family_id=row["family_id"],  # type: ignore[arg-type]
        model_inputs_sha256=row["model_inputs_sha256"],  # type: ignore[arg-type]
        parent_execution_sha256=row[
            "parent_execution_sha256"
        ],  # type: ignore[arg-type]
        parent_observation_sha256=row[
            "parent_observation_sha256"
        ],  # type: ignore[arg-type]
        parent_h4_artifact_sha256=row[
            "parent_h4_artifact_sha256"
        ],  # type: ignore[arg-type]
        prefix_sha256=row["prefix_sha256"],  # type: ignore[arg-type]
        gradient_sha256=row["gradient_sha256"],  # type: ignore[arg-type]
        parent_modal_sha256=row[
            "parent_modal_sha256"
        ],  # type: ignore[arg-type]
        balance_feature_sha256=row[
            "balance_feature_sha256"
        ],  # type: ignore[arg-type]
        negative_gated_modal_sha256=row[
            "negative_gated_modal_sha256"
        ],  # type: ignore[arg-type]
        nonnegative_gated_modal_sha256=row[
            "nonnegative_gated_modal_sha256"
        ],  # type: ignore[arg-type]
        supervised_tokens=row["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=row[
            "parent_signed_delta_nll_per_token"
        ],  # type: ignore[arg-type]
        jacobian_by_expert_route_edge=row[
            "jacobian_by_expert_route_edge"
        ],  # type: ignore[arg-type]
        active_row_count=row["active_row_count"],  # type: ignore[arg-type]
        active_row_count_by_expert=row[
            "active_row_count_by_expert"
        ],  # type: ignore[arg-type]
        active_expert_mask=row["active_expert_mask"],  # type: ignore[arg-type]
        jacobian_support_by_expert=row[
            "jacobian_support_by_expert"
        ],  # type: ignore[arg-type]
        top_mode_indices=row["top_mode_indices"],  # type: ignore[arg-type]
        top_mode_norms=row["top_mode_norms"],  # type: ignore[arg-type]
        balance_feature_std=row["balance_feature_std"],  # type: ignore[arg-type]
        top2_modal_energy_fraction=row[
            "top2_modal_energy_fraction"
        ],  # type: ignore[arg-type]
    )
    if result.fit_record_sha256 != row["fit_record_sha256"]:
        raise ValueError("state-experts fit-record hash mismatch")
    return result.to_dict()


def _label_state_experts_comparison(
    value: Mapping[str, object],
) -> dict[str, object]:
    result = dict(value)
    result["baseline_arm_id"] = "accepted_x4_plus_lag_b_parent"
    result["challenger_arm_id"] = "causal_top2_sign_state_experts"
    primary_name = (
        "family_macro_mean_prompt_absolute_delta_nll_per_token"
    )
    primary = dict(_mapping(result[primary_name], label="paired primary"))
    primary["parent"] = primary.pop("matched_alpha0")
    primary["state_experts"] = primary.pop("challenger")
    result[primary_name] = primary
    family_rows: list[dict[str, object]] = []
    for raw in result["family_rows"]:
        row = dict(_mapping(raw, label="paired family row"))
        row["parent_mean_prompt_absolute_delta_nll_per_token"] = row.pop(
            "matched_alpha0_mean_prompt_absolute_delta_nll_per_token"
        )
        row[
            "state_experts_mean_prompt_absolute_delta_nll_per_token"
        ] = row.pop(
            "challenger_mean_prompt_absolute_delta_nll_per_token"
        )
        family_rows.append(row)
    result["family_rows"] = family_rows
    secondary: list[dict[str, object]] = []
    for raw in result["secondary_metrics"]:
        row = dict(_mapping(raw, label="paired secondary metric"))
        row["parent"] = row.pop("matched_alpha0")
        row["state_experts"] = row.pop("challenger")
        secondary.append(row)
    result["secondary_metrics"] = secondary
    return result


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    left_norm = math.sqrt(math.fsum(value * value for value in centered_left))
    right_norm = math.sqrt(
        math.fsum(value * value for value in centered_right)
    )
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if tuple(left) == tuple(right) else 0.0
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(
            centered_left,
            centered_right,
            strict=True,
        )
    ) / (left_norm * right_norm)


def _linearization_diagnostics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    predicted = [
        _finite(
            row["predicted_candidate_signed_delta_nll_per_token"],
            label="predicted candidate signed delta",
        )
        for row in rows
    ]
    exact = [
        _finite(
            row["exact_candidate_signed_delta_nll_per_token"],
            label="exact candidate signed delta",
        )
        for row in rows
    ]
    parent = [
        _finite(
            row["parent_signed_delta_nll_per_token"],
            label="parent signed delta",
        )
        for row in rows
    ]
    errors = [
        predicted_value - exact_value
        for predicted_value, exact_value in zip(predicted, exact, strict=True)
    ]
    sign_matches = sum(
        (predicted_value > 0.0) == (exact_value > 0.0)
        and (predicted_value < 0.0) == (exact_value < 0.0)
        for predicted_value, exact_value in zip(predicted, exact, strict=True)
    )
    parent_abs = math.fsum(abs(value) for value in parent) / len(rows)
    predicted_abs = math.fsum(abs(value) for value in predicted) / len(rows)
    exact_abs = math.fsum(abs(value) for value in exact) / len(rows)
    return {
        "objective": (
            "parent_point_candidate_nll_vjp_d_plus_j_expert_route_theta"
        ),
        "prompt_count": len(rows),
        "predicted_vs_exact_correlation": _correlation(predicted, exact),
        "predicted_vs_exact_rmse": math.sqrt(
            math.fsum(value * value for value in errors) / len(errors)
        ),
        "predicted_vs_exact_sign_agreement": sign_matches / len(rows),
        "predicted_vs_exact_worst_absolute_error": max(
            abs(value) for value in errors
        ),
        "mean_prompt_absolute_signed_delta": {
            "parent": parent_abs,
            "predicted_candidate": predicted_abs,
            "exact_candidate": exact_abs,
            "predicted_relative_improvement": _relative_improvement(
                predicted_abs,
                parent_abs,
            ),
            "exact_relative_improvement": _relative_improvement(
                exact_abs,
                parent_abs,
            ),
        },
    }


def _validate_lineage(value: object) -> dict[str, str]:
    row = _mapping(value, label="state-experts lineage")
    if set(row) != _REQUIRED_LINEAGE_KEYS:
        raise ValueError("state-experts report lineage fields differ")
    result = {
        key: _require_sha256(row[key], label=f"lineage {key}")
        for key in sorted(row)
    }
    if (
        result["prior_iteration_report_sha256"]
        == result["factorial_report_sha256"]
        or result["prior_iteration_report_file_sha256"]
        == result["factorial_report_file_sha256"]
        or result["prior_iteration_collection_sha256"]
        in {
            result["prior_iteration_report_sha256"],
            result["prior_iteration_report_file_sha256"],
            result["factorial_report_sha256"],
            result["factorial_report_file_sha256"],
        }
    ):
        raise ValueError("prior state-router lineage aliases its prerequisite")
    return result


def _validate_resources(value: object) -> dict[str, object]:
    row = _mapping(value, label="state-experts resources")
    if set(row) != _RESOURCE_KEYS:
        raise ValueError("state-experts resource fields differ")
    integer_fields = (
        "learned_parameter_count",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_count_per_sequence",
        "derived_constant_float_count",
        "nonlinear_scalar_ops_per_token_upper_bound",
        "serving_model_forward_count",
        "residual_width",
    )
    for name in integer_fields:
        if type(row[name]) is not int or int(row[name]) < 0:
            raise ValueError(f"state-experts resource {name} is invalid")
    if int(row["residual_width"]) <= 0:
        raise ValueError("state-experts residual width must be positive")
    if type(row["parent_head_reused_not_duplicated"]) is not bool:
        raise ValueError("state-experts parent reuse receipt is invalid")
    provider_map = _mapping(
        row["candidate_provider_artifact_sha256_by_family"],
        label="state-experts resource provider map",
    )
    if len(provider_map) != _EXPECTED_FAMILIES:
        raise ValueError("state-experts resource provider map differs")
    canonical_provider_map = {
        _identifier(key, label="resource family_id"): _require_sha256(
            provider,
            label="resource provider artifact",
        )
        for key, provider in sorted(provider_map.items())
    }
    payload = {
        name: row[name]
        for name in integer_fields
        if name != "residual_width"
    }
    payload.update(
        {
            "parent_head_reused_not_duplicated": row[
                "parent_head_reused_not_duplicated"
            ],
            "parent_artifact_sha256": _require_sha256(
                row["parent_artifact_sha256"],
                label="resource parent artifact",
            ),
            "parent_h4_head_sha256": _require_sha256(
                row["parent_h4_head_sha256"],
                label="resource parent H4",
            ),
            "candidate_provider_artifact_sha256_by_family": (
                canonical_provider_map
            ),
            "residual_width": row["residual_width"],
        }
    )
    if _sha256(_RESOURCE_DOMAIN, payload) != _require_sha256(
        row["resource_receipt_sha256"],
        label="state-experts resource receipt",
    ):
        raise ValueError("state-experts resource receipt hash mismatch")
    return {
        **payload,
        "resource_receipt_sha256": row["resource_receipt_sha256"],
    }


def _validate_audit(value: object) -> dict[str, object]:
    row = dict(_mapping(value, label="state-experts execution audit"))
    if set(row) != _AUDIT_KEYS:
        raise ValueError("state-experts execution audit fields differ")
    _assert_scalar_hash_only(row, path="state-experts execution audit")
    tuple_fields = {
        "expert_route_matrix_shape": _EXPERT_ROUTE_MATRIX_SHAPE,
        "expert_regime_order": _EXPERT_REGIME_ORDER,
        "expert_route_edge_order": _EXPERT_ROUTE_EDGE_ORDER,
    }
    for name, expected in tuple_fields.items():
        if (
            not isinstance(row[name], (tuple, list))
            or tuple(row[name]) != expected
        ):
            raise ValueError(f"state-experts execution {name} differs")
        row[name] = expected
    row["routed_parent_decoder_mode_indices"] = _int2(
        row["routed_parent_decoder_mode_indices"],
        label="execution routed parent mode indices",
    )
    required = {
        "execution_mode": (
            "fit_only_two_phase_family_blocked_iterative_state_experts"
        ),
        "example_count": 16,
        "family_count": 8,
        "outer_fold_count": 8,
        "route_state_semantics": _ROUTE_STATE_SEMANTICS,
        "expert_dispatch_semantics": _EXPERT_DISPATCH_SEMANTICS,
        "phase_a_source_forward_count": 16,
        "phase_a_parent_vjp_forward_count": 16,
        "phase_b_source_forward_count": 16,
        "phase_b_candidate_forward_count": 16,
        "total_model_forward_count": 64,
        "model_forward_count_per_example": 4,
        "one_semantic_candidate_per_iteration": True,
        "family_blocked_leave_one_family_out": True,
        "source_rerun_between_phases": True,
        "source_identity_equal_across_phases": True,
        "parent_observation_count": 16,
        "candidate_observation_count": 16,
        "fit_record_count": 16,
        "fit_records_scalar_hash_only": True,
        "candidate_executions_released_between_examples": True,
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_activations_retained": False,
        "gradient_tensors_retained": False,
        "model_weights_retained": False,
        "trust_projection_interpretation": (
            _TRUST_PROJECTION_INTERPRETATION
        ),
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }
    if any(row.get(key) != expected for key, expected in required.items()):
        raise ValueError("state-experts execution audit invariants differ")
    for name in (
        "residual_width",
        "parent_prepared_float_scalar_count",
        "parent_logical_macs_per_token_upper_bound",
        "fold_trust_projection_count",
    ):
        if type(row[name]) is not int or int(row[name]) < 0:
            raise ValueError(f"execution {name} must be nonnegative integer")
    if (
        int(row["residual_width"]) <= 0
        or int(row["fold_trust_projection_count"]) > _EXPECTED_FAMILIES
    ):
        raise ValueError("state-experts execution count is outside its bound")
    for name in (
        "source_model_sha256",
        "source_execution_sha256",
        "parent_artifact_sha256",
        "parent_h4_artifact_sha256",
        "parent_h4_decoder_sha256",
        "parent_h4_lag_kernel_sha256",
        "accepted_x4_head_sha256",
        "bridge_binding_sha256",
        "fit_manifest_sha256",
    ):
        _require_sha256(row.get(name), label=f"execution {name}")
    for name, count in (
        ("model_inputs_sha256_by_example", _EXPECTED_EXAMPLES),
        ("parent_execution_sha256_by_example", _EXPECTED_EXAMPLES),
        ("candidate_execution_sha256_by_example", _EXPECTED_EXAMPLES),
        (
            "fold_provider_artifact_sha256_by_family",
            _EXPECTED_FAMILIES,
        ),
    ):
        values = _mapping(row.get(name), label=f"execution {name}")
        if len(values) != count:
            raise ValueError(f"execution {name} membership differs")
        for key, receipt in values.items():
            _identifier(key, label=f"execution {name} key")
            _require_sha256(receipt, label=f"execution {name} receipt")
    if (
        tuple(sorted(row["parent_execution_sha256_by_example"].values()))
        != tuple(row.get("parent_execution_sha256s", ()))
        or tuple(
            sorted(row["candidate_execution_sha256_by_example"].values())
        )
        != tuple(row.get("candidate_execution_sha256s", ()))
        or tuple(
            sorted(
                row[
                    "fold_provider_artifact_sha256_by_family"
                ].values()
            )
        )
        != tuple(row.get("fold_provider_artifact_sha256s", ()))
    ):
        raise ValueError("execution hash lists differ from identity maps")
    return row


def _validate_folds(
    folds: Sequence[Mapping[str, object]],
    *,
    manifest: Mapping[str, str],
    fit_by_example: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    canonical = tuple(
        sorted(
            (
                dict(_mapping(value, label="state-experts fold receipt"))
                for value in folds
            ),
            key=lambda row: str(row.get("held_family_id")),
        )
    )
    families = set(manifest.values())
    if (
        len(canonical) != _EXPECTED_FAMILIES
        or {row.get("held_family_id") for row in canonical} != families
    ):
        raise ValueError("state-experts folds do not cover held families")
    expected_fields = {
        "held_family_id",
        "train_example_ids",
        "train_family_ids",
        "train_fit_record_sha256s",
        "coefficients_by_expert_route_edge",
        "unsupported_expert_route_edge_indices",
        "active_row_count",
        "active_row_count_by_expert",
        "supported_route_edge_count_by_expert",
        "weighted_column_norm_by_expert_route_edge",
        "weighted_design_rank",
        "weighted_design_rank_by_expert",
        "normal_condition_number",
        "pre_projection_operator_norm_by_expert",
        "post_projection_operator_norm_by_expert",
        "trust_projection_applied_by_expert",
        "trust_projection_applied",
        "linearized_rmse_before",
        "linearized_rmse_after",
        "ridge",
        "operator_norm_bound",
        "fold_receipt_sha256",
    }
    for row in canonical:
        if set(row) != expected_fields:
            raise ValueError("serialized state-experts fold fields differ")
        held = _identifier(row["held_family_id"], label="held family")
        train_examples = tuple(row["train_example_ids"])
        train_families = tuple(row["train_family_ids"])
        expected_train_examples = tuple(
            sorted(
                example_id
                for example_id, family_id in manifest.items()
                if family_id != held
            )
        )
        expected_train_families = tuple(sorted(families - {held}))
        if (
            train_examples != expected_train_examples
            or train_families != expected_train_families
            or len(train_examples) != 14
            or len(train_families) != 7
            or held in train_families
        ):
            raise ValueError("state-experts fold leaks or omits a family")
        replayed = fit_gemma_iterative_state_experts_fold(
            tuple(fit_by_example[item] for item in train_examples),
            held_family_id=held,
        )
        if _canonical_json_bytes(replayed.to_dict()) != _canonical_json_bytes(
            row
        ):
            raise ValueError(
                "state-experts fold coefficients do not replay from training"
            )
        _float8(
            row["coefficients_by_expert_route_edge"],
            label="fold expert-route coefficients",
        )
        unsupported = tuple(row["unsupported_expert_route_edge_indices"])
        if (
            unsupported != tuple(sorted(set(unsupported)))
            or any(
                type(index) is not int or not 0 <= index < 8
                for index in unsupported
            )
        ):
            raise ValueError("fold unsupported expert edges are invalid")
        active_by_expert = _nonnegative_int2(
            row["active_row_count_by_expert"],
            label="fold active rows by expert",
        )
        supported_by_expert = _nonnegative_int2(
            row["supported_route_edge_count_by_expert"],
            label="fold supported edges by expert",
        )
        rank_by_expert = _nonnegative_int2(
            row["weighted_design_rank_by_expert"],
            label="fold weighted rank by expert",
        )
        pre_norms = _float2(
            row["pre_projection_operator_norm_by_expert"],
            label="fold pre-projection norms",
        )
        post_norms = _float2(
            row["post_projection_operator_norm_by_expert"],
            label="fold post-projection norms",
        )
        projected = _bool2(
            row["trust_projection_applied_by_expert"],
            label="fold trust projection by expert",
        )
        if (
            type(row["active_row_count"]) is not int
            or row["active_row_count"] <= 0
            or sum(active_by_expert) != row["active_row_count"]
            or any(value > 4 for value in supported_by_expert)
            or any(value > 4 for value in rank_by_expert)
            or int(row["weighted_design_rank"]) > 8
            or any(value < 0.0 for value in (*pre_norms, *post_norms))
            or any(value > 0.25 + 1.0e-12 for value in post_norms)
            or bool(row["trust_projection_applied"]) != any(projected)
            or any(
                flag != (pre_norm > 0.25)
                for flag, pre_norm in zip(
                    projected,
                    pre_norms,
                    strict=True,
                )
            )
        ):
            raise ValueError("state-experts fold expert receipt is invalid")
        expected_supported = (
            sum(index not in unsupported for index in range(0, 4)),
            sum(index not in unsupported for index in range(4, 8)),
        )
        if supported_by_expert != expected_supported:
            raise ValueError("fold expert support counts contradict edges")
        _assert_scalar_hash_only(row, path=f"state-experts fold {held}")
    return canonical


def _validate_oof_rows(
    values: Sequence[Mapping[str, object]],
    *,
    parent_by_example: Mapping[
        str, GemmaH4DampingFiniteNLLObservation
    ],
    candidate_by_example: Mapping[
        str, GemmaH4DampingFiniteNLLObservation
    ],
    fit_by_example: Mapping[str, Mapping[str, object]],
    folds_by_family: Mapping[str, Mapping[str, object]],
    provider_by_family: Mapping[str, str],
    candidate_execution_by_example: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    rows = tuple(
        sorted(
            (
                dict(_mapping(value, label="state-experts OOF row"))
                for value in values
            ),
            key=lambda row: str(row.get("example_id")),
        )
    )
    if (
        len(rows) != _EXPECTED_EXAMPLES
        or len({row.get("example_id") for row in rows})
        != _EXPECTED_EXAMPLES
    ):
        raise ValueError("state-experts OOF rows do not cover all examples")
    expected_fields = {
        "example_id",
        "family_id",
        "held_family_id",
        "parent_signed_delta_nll_per_token",
        "predicted_candidate_signed_delta_nll_per_token",
        "exact_candidate_signed_delta_nll_per_token",
        "jacobian_by_expert_route_edge",
        "coefficients_by_expert_route_edge",
        "train_example_ids",
        "train_family_ids",
        "fit_record_sha256",
        "fold_receipt_sha256",
        "provider_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_observation_sha256",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("state-experts OOF-row fields differ")
        example_id = _identifier(row["example_id"], label="OOF example")
        family_id = _identifier(row["family_id"], label="OOF family")
        parent = parent_by_example.get(example_id)
        candidate = candidate_by_example.get(example_id)
        fit = fit_by_example.get(example_id)
        fold = folds_by_family.get(family_id)
        if parent is None or candidate is None or fit is None or fold is None:
            raise ValueError("state-experts OOF references unknown identity")
        jacobian = _float8(
            row["jacobian_by_expert_route_edge"],
            label="OOF expert-route Jacobian",
        )
        coefficients = _float8(
            row["coefficients_by_expert_route_edge"],
            label="OOF expert-route coefficients",
        )
        parent_signed = (
            parent.candidate_summed_nll - parent.source_summed_nll
        ) / parent.supervised_tokens
        predicted = parent_signed + math.fsum(
            left * right
            for left, right in zip(jacobian, coefficients, strict=True)
        )
        exact_signed = (
            candidate.candidate_summed_nll - candidate.source_summed_nll
        ) / candidate.supervised_tokens
        if (
            family_id != parent.family_id
            or row["held_family_id"] != family_id
            or tuple(fit["jacobian_by_expert_route_edge"]) != jacobian
            or tuple(fold["coefficients_by_expert_route_edge"])
            != coefficients
            or row["fit_record_sha256"] != fit["fit_record_sha256"]
            or row["fold_receipt_sha256"] != fold["fold_receipt_sha256"]
            or row["provider_artifact_sha256"]
            != provider_by_family[family_id]
            or row["candidate_execution_sha256"]
            != candidate_execution_by_example[example_id]
            or row["candidate_observation_sha256"]
            != candidate.observation_sha256
            or tuple(row["train_example_ids"])
            != tuple(fold["train_example_ids"])
            or tuple(row["train_family_ids"])
            != tuple(fold["train_family_ids"])
            or not math.isclose(
                _finite(
                    row["parent_signed_delta_nll_per_token"],
                    label="OOF parent signed delta",
                ),
                parent_signed,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _finite(
                    row["predicted_candidate_signed_delta_nll_per_token"],
                    label="OOF predicted signed delta",
                ),
                predicted,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _finite(
                    row["exact_candidate_signed_delta_nll_per_token"],
                    label="OOF exact signed delta",
                ),
                exact_signed,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("state-experts OOF row does not replay")
    return rows


def _scientific_metrics(
    fits: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, bool]]:
    by_family: dict[str, list[Mapping[str, object]]] = {}
    for row in fits:
        by_family.setdefault(str(row["family_id"]), []).append(row)
    family_rows: list[dict[str, object]] = []
    for family_id in sorted(by_family):
        records = by_family[family_id]
        active_by_expert = tuple(
            sum(
                _nonnegative_int2(
                    row["active_row_count_by_expert"],
                    label="fit active rows by expert",
                )[expert_index]
                for row in records
            )
            for expert_index in range(2)
        )
        total_active = sum(active_by_expert)
        if total_active <= 0:
            raise ValueError("state-experts family has no active rows")
        balance = math.fsum(
            _finite(
                row["balance_feature_std"],
                label="balance feature std",
            )
            for row in records
        ) / len(records)
        energy = math.fsum(
            _finite(
                row["top2_modal_energy_fraction"],
                label="top2 modal energy fraction",
            )
            for row in records
        ) / len(records)
        fractions = tuple(value / total_active for value in active_by_expert)
        if not 0.0 <= energy <= 1.0 or balance < 0.0:
            raise ValueError("state-experts scientific statistic is invalid")
        family_rows.append(
            {
                "family_id": family_id,
                "prompt_count": len(records),
                "active_row_count": total_active,
                "active_row_count_by_expert": active_by_expert,
                "active_row_fraction_by_expert": fractions,
                "mean_prompt_balance_feature_std": balance,
                "mean_prompt_top2_modal_energy_fraction": energy,
            }
        )
    macro_balance = math.fsum(
        float(row["mean_prompt_balance_feature_std"])
        for row in family_rows
    ) / len(family_rows)
    macro_energy = math.fsum(
        float(row["mean_prompt_top2_modal_energy_fraction"])
        for row in family_rows
    ) / len(family_rows)
    macro_fraction_by_expert = tuple(
        math.fsum(
            row["active_row_fraction_by_expert"][expert_index]
            for row in family_rows
        )
        / len(family_rows)
        for expert_index in range(2)
    )
    rank_by_family = {
        str(row["held_family_id"]): int(row["weighted_design_rank"])
        for row in folds
    }
    rank_by_expert_by_family = {
        str(row["held_family_id"]): _nonnegative_int2(
            row["weighted_design_rank_by_expert"],
            label="fold weighted rank by expert",
        )
        for row in folds
    }
    support_count_by_expert_by_family = {
        str(row["held_family_id"]): _nonnegative_int2(
            row["supported_route_edge_count_by_expert"],
            label="fold support count by expert",
        )
        for row in folds
    }
    active_rows_by_expert_by_family = {
        str(row["held_family_id"]): _nonnegative_int2(
            row["active_row_count_by_expert"],
            label="fold active rows by expert",
        )
        for row in folds
    }
    supported_fold_count_by_edge = {
        edge: sum(
            index
            not in tuple(row["unsupported_expert_route_edge_indices"])
            for row in folds
        )
        for index, edge in enumerate(_EXPERT_ROUTE_EDGE_ORDER)
    }
    supported_every_fold = all(
        value == _EXPECTED_FAMILIES
        for value in supported_fold_count_by_edge.values()
    )
    projected_family_ids_by_expert = {
        regime: [
            str(row["held_family_id"])
            for row in folds
            if _bool2(
                row["trust_projection_applied_by_expert"],
                label="trust projection by expert",
            )[expert_index]
        ]
        for expert_index, regime in enumerate(_EXPERT_REGIME_ORDER)
    }
    gates = {
        "all_fold_weighted_design_ranks_exactly_8": (
            len(rank_by_family) == _EXPECTED_FAMILIES
            and all(value == 8 for value in rank_by_family.values())
        ),
        "all_8_expert_edges_supported_in_every_fold": (
            supported_every_fold
        ),
        "negative_family_macro_active_row_fraction_at_least_0_10": (
            macro_fraction_by_expert[0]
            >= _REGIME_FAMILY_MACRO_ACTIVE_FRACTION_MIN
        ),
        "nonnegative_family_macro_active_row_fraction_at_least_0_10": (
            macro_fraction_by_expert[1]
            >= _REGIME_FAMILY_MACRO_ACTIVE_FRACTION_MIN
        ),
        "family_macro_balance_feature_std_at_least_0_05": (
            macro_balance >= _BALANCE_FEATURE_STD_MIN
        ),
        "family_macro_top2_modal_energy_fraction_at_least_0_5": (
            macro_energy >= _TOP2_MODAL_ENERGY_FRACTION_MIN
        ),
    }
    gates["passed"] = all(gates.values())
    return (
        {
            "aggregation": (
                "active_rows_sum_within_family_then_fraction_then_equal_"
                "family_macro"
            ),
            "expert_regime_order": _EXPERT_REGIME_ORDER,
            "minimum_family_macro_active_row_fraction_per_regime": (
                _REGIME_FAMILY_MACRO_ACTIVE_FRACTION_MIN
            ),
            "family_rows": family_rows,
            "family_macro_active_row_fraction_by_expert": (
                macro_fraction_by_expert
            ),
            "family_macro_balance_feature_std": macro_balance,
            "minimum_family_macro_balance_feature_std": (
                _BALANCE_FEATURE_STD_MIN
            ),
            "family_macro_top2_modal_energy_fraction": macro_energy,
            "minimum_family_macro_top2_modal_energy_fraction": (
                _TOP2_MODAL_ENERGY_FRACTION_MIN
            ),
            "weighted_design_rank_by_held_family": rank_by_family,
            "weighted_design_rank_by_expert_by_held_family": (
                rank_by_expert_by_family
            ),
            "full_rank_fold_count": sum(
                value == 8 for value in rank_by_family.values()
            ),
            "supported_route_edge_count_by_expert_by_held_family": (
                support_count_by_expert_by_family
            ),
            "active_row_count_by_expert_by_held_family": (
                active_rows_by_expert_by_family
            ),
            "supported_fold_count_by_expert_route_edge": (
                supported_fold_count_by_edge
            ),
            "all_edges_supported_fold_count": sum(
                not tuple(row["unsupported_expert_route_edge_indices"])
                for row in folds
            ),
            "total_active_row_count": sum(
                int(row["active_row_count"]) for row in fits
            ),
            "trust_projection_fold_count_by_expert": {
                regime: len(family_ids)
                for regime, family_ids in (
                    projected_family_ids_by_expert.items()
                )
            },
            "trust_projection_family_ids_by_expert": (
                projected_family_ids_by_expert
            ),
        },
        gates,
    )


def _provider_artifact_sha256(
    *,
    lineage: Mapping[str, str],
    audit: Mapping[str, object],
    fold: Mapping[str, object],
    top_mode_indices: tuple[int, int],
    top_mode_norms: tuple[float, float],
) -> str:
    return gemma_causal_top2_state_experts_provider_artifact_sha256(
        parent_artifact_sha256=lineage["parent_artifact_sha256"],
        parent_h4_artifact_sha256=lineage["parent_h4_head_sha256"],
        bridge_binding_sha256=lineage["bridge_binding_sha256"],
        decoder_sha256=str(audit["parent_h4_decoder_sha256"]),
        lag_kernel_sha256=str(audit["parent_h4_lag_kernel_sha256"]),
        fold_receipt_sha256=str(fold["fold_receipt_sha256"]),
        top_mode_indices=top_mode_indices,
        top_mode_norms=top_mode_norms,
        coefficients_by_expert_route_edge=fold[
            "coefficients_by_expert_route_edge"
        ],
    )


def _validate_retained_full_fit(
    value: object,
    *,
    retained: bool,
    provisional: bool,
    fits: Sequence[Mapping[str, object]],
    lineage: Mapping[str, str],
    resources: Mapping[str, object],
    audit: Mapping[str, object],
    top_mode_indices: tuple[int, int],
    top_mode_norms: tuple[float, float],
) -> Mapping[str, object] | None:
    if value is None:
        if retained and not provisional:
            raise ValueError(
                "retained state-experts iteration must bind its full fit"
            )
        return None
    if not retained:
        raise ValueError(
            "rejected state-experts iteration cannot publish a full fit"
        )
    row = dict(_mapping(value, label="retained state-experts full fit"))
    expected_fields = {
        "provider_artifact_sha256",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "bridge_binding_sha256",
        "learned_parameter_count",
        "logical_macs_per_token_upper_bound",
        "runtime_state_float_count_per_sequence",
        "derived_constant_float_count",
        "nonlinear_scalar_ops_per_token_upper_bound",
        "full_fit",
        "retention_receipt_sha256",
    }
    if set(row) != expected_fields:
        raise ValueError("retained state-experts full-fit fields differ")
    replayed_fit = fit_gemma_iterative_state_experts_fold(
        fits,
        held_family_id="__full_fit__",
    )
    submitted_fit = _mapping(
        row["full_fit"],
        label="retained state-experts full fit",
    )
    if _canonical_json_bytes(replayed_fit.to_dict()) != _canonical_json_bytes(
        submitted_fit
    ):
        raise ValueError("retained state-experts full fit does not replay")
    expected_provider = _provider_artifact_sha256(
        lineage=lineage,
        audit=audit,
        fold=replayed_fit.to_dict(),
        top_mode_indices=top_mode_indices,
        top_mode_norms=top_mode_norms,
    )
    payload = {
        "provider_artifact_sha256": expected_provider,
        "parent_artifact_sha256": lineage["parent_artifact_sha256"],
        "parent_h4_head_sha256": lineage["parent_h4_head_sha256"],
        "bridge_binding_sha256": lineage["bridge_binding_sha256"],
        "learned_parameter_count": resources["learned_parameter_count"],
        "logical_macs_per_token_upper_bound": resources[
            "logical_macs_per_token_upper_bound"
        ],
        "runtime_state_float_count_per_sequence": resources[
            "runtime_state_float_count_per_sequence"
        ],
        "derived_constant_float_count": resources[
            "derived_constant_float_count"
        ],
        "nonlinear_scalar_ops_per_token_upper_bound": resources[
            "nonlinear_scalar_ops_per_token_upper_bound"
        ],
        "full_fit": replayed_fit.to_dict(),
    }
    if any(
        row[key] != expected
        for key, expected in payload.items()
        if key != "full_fit"
    ):
        raise ValueError(
            "retained state-experts provider lineage or resources differ"
        )
    if _sha256(_RETENTION_DOMAIN, payload) != _require_sha256(
        row["retention_receipt_sha256"],
        label="state-experts retention receipt",
    ):
        raise ValueError("state-experts retention receipt hash mismatch")
    return {
        **payload,
        "retention_receipt_sha256": row["retention_receipt_sha256"],
    }


def build_gemma_iterative_state_experts_report(
    *,
    parent_observations: Sequence[
        GemmaH4DampingFiniteNLLObservation
    ],
    candidate_observations: Sequence[
        GemmaH4DampingFiniteNLLObservation
    ],
    oof_rows: Sequence[Mapping[str, object]],
    fit_records: Sequence[object],
    fold_receipts: Sequence[Mapping[str, object]],
    manifest: Mapping[str, str],
    lineage: Mapping[str, object],
    resources: Mapping[str, object],
    audit: Mapping[str, object],
    retained_full_fit_receipt: Mapping[str, object] | None = None,
    provisional: bool = False,
) -> dict[str, object]:
    """Build iteration three's exact family-disjoint retention report."""

    parent = _canonical_observations(
        parent_observations,
        label="state-experts parent observations",
    )
    candidate = _canonical_observations(
        candidate_observations,
        label="state-experts candidate observations",
    )
    if _source_grid(parent) != _source_grid(candidate):
        raise ValueError("state-experts parent and candidate grids differ")
    canonical_manifest = _validate_manifest(manifest, observations=parent)
    canonical_lineage = _validate_lineage(lineage)
    canonical_resources = _validate_resources(resources)
    canonical_audit = _validate_audit(audit)
    if type(provisional) is not bool:
        raise TypeError("provisional must be boolean")
    if (
        canonical_lineage["parent_artifact_sha256"]
        != canonical_audit["parent_artifact_sha256"]
        or canonical_lineage["parent_artifact_sha256"]
        != canonical_resources["parent_artifact_sha256"]
        or canonical_lineage["parent_h4_head_sha256"]
        != canonical_audit["parent_h4_artifact_sha256"]
        or canonical_lineage["parent_h4_head_sha256"]
        != canonical_resources["parent_h4_head_sha256"]
        or canonical_lineage["accepted_x4_head_sha256"]
        != canonical_audit["accepted_x4_head_sha256"]
        or canonical_lineage["bridge_binding_sha256"]
        != canonical_audit["bridge_binding_sha256"]
        or canonical_lineage["model_sha256"]
        != canonical_audit["source_model_sha256"]
        or canonical_lineage["adapter_execution_sha256"]
        != canonical_audit["source_execution_sha256"]
        or canonical_lineage["fit_manifest_sha256"]
        != canonical_audit["fit_manifest_sha256"]
        or canonical_resources["residual_width"]
        != canonical_audit["residual_width"]
    ):
        raise ValueError(
            "state-experts lineage, execution, and resources contradict"
        )
    model_inputs_by_example = dict(
        _mapping(
            canonical_audit["model_inputs_sha256_by_example"],
            label="state-experts model inputs",
        )
    )
    parent_execution_by_example = dict(
        _mapping(
            canonical_audit["parent_execution_sha256_by_example"],
            label="state-experts parent executions",
        )
    )
    candidate_execution_by_example = dict(
        _mapping(
            canonical_audit["candidate_execution_sha256_by_example"],
            label="state-experts candidate executions",
        )
    )
    audit_provider_by_family = dict(
        _mapping(
            canonical_audit["fold_provider_artifact_sha256_by_family"],
            label="state-experts fold providers",
        )
    )
    resource_provider_by_family = dict(
        _mapping(
            canonical_resources[
                "candidate_provider_artifact_sha256_by_family"
            ],
            label="state-experts resource providers",
        )
    )
    if (
        set(model_inputs_by_example) != set(canonical_manifest)
        or set(parent_execution_by_example) != set(canonical_manifest)
        or set(candidate_execution_by_example) != set(canonical_manifest)
        or set(audit_provider_by_family)
        != set(canonical_manifest.values())
        or audit_provider_by_family != resource_provider_by_family
    ):
        raise ValueError("state-experts execution maps differ from manifest")

    fits = tuple(
        sorted(
            (_fit_record_dict(value) for value in fit_records),
            key=lambda row: str(row["example_id"]),
        )
    )
    if (
        len(fits) != _EXPECTED_EXAMPLES
        or len({str(row["example_id"]) for row in fits})
        != _EXPECTED_EXAMPLES
    ):
        raise ValueError("state-experts fit records differ from manifest")
    parent_by_example = {row.example_id: row for row in parent}
    candidate_by_example = {row.example_id: row for row in candidate}
    fit_by_example = {str(row["example_id"]): row for row in fits}
    top_mode_indices_set = {
        _int2(row["top_mode_indices"], label="fit top-mode indices")
        for row in fits
    }
    top_mode_norms_set = {
        _float2(row["top_mode_norms"], label="fit top-mode norms")
        for row in fits
    }
    if len(top_mode_indices_set) != 1 or len(top_mode_norms_set) != 1:
        raise ValueError("state-experts fits disagree on mode constants")
    top_mode_indices = next(iter(top_mode_indices_set))
    top_mode_norms = next(iter(top_mode_norms_set))
    if top_mode_indices != tuple(
        canonical_audit["routed_parent_decoder_mode_indices"]
    ):
        raise ValueError("state-experts fit modes differ from execution")
    for example_id, fit in fit_by_example.items():
        observation = parent_by_example.get(example_id)
        expected_parent_signed = (
            0.0
            if observation is None
            else (
                observation.candidate_summed_nll
                - observation.source_summed_nll
            )
            / observation.supervised_tokens
        )
        active_by_expert = _nonnegative_int2(
            fit["active_row_count_by_expert"],
            label="fit active rows by expert",
        )
        active_mask = _bool2(
            fit["active_expert_mask"],
            label="fit active expert mask",
        )
        jacobian_support = _bool2(
            fit["jacobian_support_by_expert"],
            label="fit Jacobian support by expert",
        )
        if (
            observation is None
            or fit["family_id"] != observation.family_id
            or fit["parent_observation_sha256"]
            != observation.observation_sha256
            or fit["parent_h4_artifact_sha256"]
            != canonical_lineage["parent_h4_head_sha256"]
            or fit["supervised_tokens"] != observation.supervised_tokens
            or fit["model_inputs_sha256"]
            != model_inputs_by_example[example_id]
            or fit["parent_execution_sha256"]
            != parent_execution_by_example[example_id]
            or int(fit["active_row_count"]) != sum(active_by_expert)
            or active_mask
            != tuple(value > 0 for value in active_by_expert)
            or any(
                supported and not active
                for supported, active in zip(
                    jacobian_support,
                    active_mask,
                    strict=True,
                )
            )
            or not math.isclose(
                float(fit["parent_signed_delta_nll_per_token"]),
                expected_parent_signed,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "state-experts fit differs from parent observation"
            )

    folds = _validate_folds(
        fold_receipts,
        manifest=canonical_manifest,
        fit_by_example=fit_by_example,
    )
    folds_by_family = {
        str(row["held_family_id"]): row for row in folds
    }
    if canonical_audit["fold_trust_projection_count"] != sum(
        bool(row["trust_projection_applied"]) for row in folds
    ):
        raise ValueError(
            "state-experts audit projection count differs from folds"
        )
    for family_id, fold in folds_by_family.items():
        expected_provider = _provider_artifact_sha256(
            lineage=canonical_lineage,
            audit=canonical_audit,
            fold=fold,
            top_mode_indices=top_mode_indices,
            top_mode_norms=top_mode_norms,
        )
        if resource_provider_by_family[family_id] != expected_provider:
            raise ValueError(
                "state-experts provider does not replay from fold"
            )
    canonical_oof = _validate_oof_rows(
        oof_rows,
        parent_by_example=parent_by_example,
        candidate_by_example=candidate_by_example,
        fit_by_example=fit_by_example,
        folds_by_family=folds_by_family,
        provider_by_family=resource_provider_by_family,
        candidate_execution_by_example=candidate_execution_by_example,
    )

    parent_payloads = [_observation_dict(row) for row in parent]
    candidate_payloads = [_observation_dict(row) for row in candidate]
    parent_fidelity = _fidelity_from_observations(parent_payloads)
    candidate_fidelity = _fidelity_from_observations(candidate_payloads)
    paired = _label_state_experts_comparison(
        _correct_prompt_disagreement_quantile_labels(
            _paired_comparison(
                parent_fidelity,
                candidate_fidelity,
                baseline_observations=parent_payloads,
                challenger_observations=candidate_payloads,
            )
        )
    )
    paired_gates = _mapping(
        paired["gates"],
        label="state-experts paired gates",
    )
    paired_primary = _mapping(
        paired[
            "family_macro_mean_prompt_absolute_delta_nll_per_token"
        ],
        label="state-experts paired primary",
    )
    behavior_gates = {
        "family_macro_error_regression_at_most_2pct": (
            _finite(
                paired_primary["relative_improvement"],
                label="family macro improvement",
            )
            >= _MACRO_RELATIVE_IMPROVEMENT_MIN
        ),
        "strict_family_win_count_at_least_6_of_8": (
            int(paired["strict_family_win_count"])
            >= _MINIMUM_FAMILY_WIN_COUNT
        ),
        "worst_family_improvement_at_least_minus_2pct": (
            _finite(
                paired["worst_family_relative_improvement"],
                label="worst family improvement",
            )
            >= _WORST_FAMILY_RELATIVE_IMPROVEMENT_MIN
        ),
        "family_macro_kl_regression_at_most_2pct": bool(
            paired_gates["family_macro_kl_regression_at_most_2pct"]
        ),
        "family_macro_top1_disagreement_regression_at_most_2pct": bool(
            paired_gates[
                "family_macro_top1_disagreement_regression_at_most_2pct"
            ]
        ),
        "prompt_p90_absolute_delta_nll_regression_at_most_2pct": bool(
            paired_gates[
                "prompt_p90_absolute_delta_nll_regression_at_most_2pct"
            ]
        ),
        "prompt_p90_top1_disagreement_regression_at_most_2pct": bool(
            paired_gates[
                "prompt_p90_top1_disagreement_regression_at_most_2pct"
            ]
        ),
    }
    behavior_gates["passed"] = all(behavior_gates.values())
    scientific_metrics, scientific_gates = _scientific_metrics(fits, folds)
    resource_gates = {
        "learned_parameter_count_exactly_8": (
            canonical_resources["learned_parameter_count"] == 8
        ),
        "logical_macs_per_token_exactly_6": (
            canonical_resources["logical_macs_per_token_upper_bound"] == 6
        ),
        "runtime_state_float_count_per_sequence_exactly_2": (
            canonical_resources[
                "runtime_state_float_count_per_sequence"
            ]
            == 2
        ),
        "derived_constant_float_count_exactly_2": (
            canonical_resources["derived_constant_float_count"] == 2
        ),
        "nonlinear_scalar_ops_per_token_upper_bound_exactly_6": (
            canonical_resources[
                "nonlinear_scalar_ops_per_token_upper_bound"
            ]
            == 6
        ),
        "serving_model_forward_count_exactly_1": (
            canonical_resources["serving_model_forward_count"] == 1
        ),
        "parent_head_reused_not_duplicated": (
            canonical_resources["parent_head_reused_not_duplicated"] is True
        ),
    }
    resource_gates["passed"] = all(resource_gates.values())
    retained = (
        bool(behavior_gates["passed"])
        and bool(scientific_gates["passed"])
        and bool(resource_gates["passed"])
    )
    candidate_absolute_gates = _mapping(
        candidate_fidelity["gates"],
        label="state-experts candidate absolute gates",
    )
    ready_for_new_selection = retained and bool(
        candidate_absolute_gates["passed"]
    )
    canonical_retained_full_fit = _validate_retained_full_fit(
        retained_full_fit_receipt,
        retained=retained,
        provisional=provisional,
        fits=fits,
        lineage=canonical_lineage,
        resources=canonical_resources,
        audit=canonical_audit,
        top_mode_indices=top_mode_indices,
        top_mode_norms=top_mode_norms,
    )
    linearization = _linearization_diagnostics(canonical_oof)
    any_projected_families = [
        str(row["held_family_id"])
        for row in folds
        if bool(row["trust_projection_applied"])
    ]
    collection_payload = {
        "manifest": canonical_manifest,
        "lineage": canonical_lineage,
        "resources": canonical_resources,
        "execution": canonical_audit,
        "fit_records": list(fits),
        "fold_receipts": list(folds),
        "oof_rows": list(canonical_oof),
        "observations": {
            "parent": parent_payloads,
            "candidate": candidate_payloads,
        },
        "retained_full_fit": canonical_retained_full_fit,
    }
    collection_sha256 = _sha256(_COLLECTION_DOMAIN, collection_payload)
    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "semantics": {
            "iteration": 3,
            "candidate": (
                "causal_top2_lag_b_modal_sign_dispatched_state_experts_v1"
            ),
            "theta_zero_is_parent": True,
            "expert_route_matrix_shape": _EXPERT_ROUTE_MATRIX_SHAPE,
            "expert_regime_order": _EXPERT_REGIME_ORDER,
            "expert_route_edge_order": _EXPERT_ROUTE_EDGE_ORDER,
            "route_state_semantics": _ROUTE_STATE_SEMANTICS,
            "expert_dispatch_semantics": _EXPERT_DISPATCH_SEMANTICS,
            "fit_objective": (
                "per_prompt_d_plus_j_expert_route_theta_family_weighted_"
                "ridge"
            ),
            "retention_authority": (
                "exact_family_disjoint_out_of_fold_finite_metrics"
            ),
            "prior_iteration_must_be_rejected": True,
            "failure_scope": (
                "rejects_only_sign_dispatched_top2_lag_b_state_experts"
            ),
        },
        "manifest": {
            "example_count": _EXPECTED_EXAMPLES,
            "family_count": _EXPECTED_FAMILIES,
            "examples_per_family": _EXPECTED_PER_FAMILY,
            "family_by_example": canonical_manifest,
        },
        "lineage": canonical_lineage,
        "resources": canonical_resources,
        "execution": canonical_audit,
        "collection_sha256": collection_sha256,
        "fit_records": list(fits),
        "fold_receipts": list(folds),
        "oof_rows": list(canonical_oof),
        "observations": {
            "parent": parent_payloads,
            "candidate": candidate_payloads,
        },
        "retained_full_fit": canonical_retained_full_fit,
        "metrics": {
            "parent": parent_fidelity,
            "candidate": candidate_fidelity,
            "paired": paired,
            "linearization": {
                **linearization,
                "trust_projection_fold_count": len(
                    any_projected_families
                ),
                "trust_projection_family_ids": any_projected_families,
            },
            "expert_activity_support_and_coverage": scientific_metrics,
        },
        "decision": {
            "behavior_relative_gates": behavior_gates,
            "scientific_gates": scientific_gates,
            "resource_gates": resource_gates,
            "retained": retained,
            "ready_for_new_selection": ready_for_new_selection,
            "absolute_fidelity_gates_passed": bool(
                candidate_absolute_gates["passed"]
            ),
            "deployment_authorized": False,
            "generalization_claim": False,
        },
        "safety": dict(_SAFETY),
    }
    _assert_scalar_hash_only(payload, path="state-experts report")
    return {**payload, "report_sha256": _sha256(_REPORT_DOMAIN, payload)}


def validate_gemma_iterative_state_experts_report(
    report: Mapping[str, object],
) -> None:
    """Replay every scalar receipt, metric, gate, and decision."""

    row = dict(_mapping(report, label="state-experts report"))
    report_sha256 = _require_sha256(
        row.pop("report_sha256", None),
        label="state-experts report",
    )
    if _sha256(_REPORT_DOMAIN, row) != report_sha256:
        raise ValueError("state-experts report hash mismatch")
    if (
        row.get("schema") != _SCHEMA
        or row.get("format_version") != _FORMAT_VERSION
        or row.get("safety") != _SAFETY
    ):
        raise ValueError("state-experts report header or safety differs")
    manifest_payload = _mapping(row.get("manifest"), label="manifest payload")
    observations = _mapping(row.get("observations"), label="observations")
    parent_raw = observations.get("parent")
    candidate_raw = observations.get("candidate")
    if not isinstance(parent_raw, list) or not isinstance(candidate_raw, list):
        raise TypeError("serialized observations must be lists")
    parent = tuple(
        _observation_from_dict(
            value,
            label="state-experts parent observation",
        )
        for value in parent_raw
    )
    candidate = tuple(
        _observation_from_dict(
            value,
            label="state-experts candidate observation",
        )
        for value in candidate_raw
    )
    fit_records = row.get("fit_records")
    folds = row.get("fold_receipts")
    oof = row.get("oof_rows")
    if (
        not isinstance(fit_records, list)
        or not isinstance(folds, list)
        or not isinstance(oof, list)
    ):
        raise TypeError(
            "serialized state-experts fit/fold/OOF rows must be lists"
        )
    rebuilt = build_gemma_iterative_state_experts_report(
        parent_observations=parent,
        candidate_observations=candidate,
        oof_rows=oof,
        fit_records=fit_records,
        fold_receipts=folds,
        manifest=_mapping(
            manifest_payload.get("family_by_example"),
            label="manifest family map",
        ),  # type: ignore[arg-type]
        lineage=_mapping(
            row.get("lineage"),
            label="state-experts lineage",
        ),
        resources=_mapping(
            row.get("resources"),
            label="state-experts resources",
        ),
        audit=_mapping(
            row.get("execution"),
            label="state-experts execution",
        ),
        retained_full_fit_receipt=(
            None
            if row.get("retained_full_fit") is None
            else _mapping(
                row.get("retained_full_fit"),
                label="retained state-experts full fit",
            )
        ),
        provisional=False,
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(report):
        raise ValueError("state-experts report derived state does not replay")
