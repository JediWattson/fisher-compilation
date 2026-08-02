"""Exact replay analysis for iteration four's affine conformal route.

The accepted-X4 plus lag-B parent remains frozen.  Iteration four fits one
four-coordinate conformal route from parent-point NLL Jacobians, evaluates it
with family-blocked leave-one-family-out providers, and permits a retained
full-data refit only when behavioral, scientific, and resource gates all pass.
This module is deliberately scalar-only: the live campaign owns tensors and
model execution while the report independently replays every fit, provider
identity, finite observation, prediction, projection, and decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
    _fidelity_from_observations,
    _paired_comparison,
)
from .gemma3_l3_l4_iterative_conformal_route import (
    CONFORMAL_COEFFICIENT_COUNT,
    CONFORMAL_OPERATOR_NORM_BOUND,
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE,
    GemmaIterativeConformalRouteFitRecord,
    fit_gemma_iterative_conformal_route_fold,
    gemma_causal_top2_conformal_route_provider_artifact_sha256,
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
    "build_gemma_iterative_conformal_route_report",
    "validate_gemma_iterative_conformal_route_report",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_iterative_conformal_route_analysis"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma-iterative-conformal-route-report:v1\0"
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-conformal-route-collection:v1\0"
)
_RESOURCE_DOMAIN = b"fisher-graph:gemma-iterative-residual-resources:v1\0"
_RETENTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-retained-provider:v1\0"
)
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PER_FAMILY = 2
_COEFFICIENT_ORDER = (
    "shared_real",
    "shared_imag",
    "contrast_real",
    "contrast_imag",
)
_ENDPOINT_ORDER = ("g=-1", "g=+1")
_MACRO_RELATIVE_IMPROVEMENT_MIN = -0.02
_MINIMUM_FAMILY_WIN_COUNT = 6
_WORST_FAMILY_RELATIVE_IMPROVEMENT_MIN = -0.02
_BALANCE_FEATURE_STD_MIN = 0.05
_TOP2_MODAL_ENERGY_FRACTION_MIN = 0.5
_MEDIAN_DESIGN_CONDITION_MAX = 100.0
_MEAN_FOLD_COEFFICIENT_COSINE_MIN = 0.90
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
_PROVIDER_RESOURCE_FIELD_ORDER = (
    "learned_parameter_count",
    "logical_macs_per_token_upper_bound",
    "derived_constant_float_count",
    "prepared_float_scalar_count",
    "runtime_state_float_count_per_sequence",
    "nonlinear_scalar_ops_per_token_upper_bound",
    "linear_accumulator_scalar_ops_per_token_upper_bound",
    "zero_denominator_comparisons_per_token_upper_bound",
    "parent_decoder_invocations_per_token",
)
_PROVIDER_RESOURCE_KEYS = frozenset(_PROVIDER_RESOURCE_FIELD_ORDER)
_RESOURCE_KEYS = frozenset(
    {
        *_PROVIDER_RESOURCE_KEYS,
        "serving_model_forward_count",
        "parent_head_reused_not_duplicated",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "candidate_provider_artifact_sha256_by_family",
        "residual_width",
        "resource_receipt_sha256",
    }
)
_BASE_AUDIT_KEYS = frozenset(
    {
        "example_count",
        "family_count",
        "outer_fold_count",
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
        "selection_input_opened",
        "guard_input_opened",
        "calibration_b_opened",
        "assessment_input_opened",
        "development_only",
    }
)
_RECIPE_AUDIT_FIELDS = dict(
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE.audit_recipe_fields
)
_PROVIDER_AUDIT_FIELDS = dict(
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE.provider_audit_fields
)
_PARENT_TENSOR_AUDIT_FIELDS = dict(
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE.parent_tensor_audit_fields
)
_PROJECTION_COUNT_FIELD = (
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE
    .fold_projection_count_audit_field
)
_PROJECTION_INTERPRETATION_FIELD = (
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE
    .projection_interpretation_audit_field
)
_PROJECTION_INTERPRETATION = (
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE
    .projection_interpretation
)
_AUDIT_KEYS = frozenset(
    {
        *_BASE_AUDIT_KEYS,
        *_RECIPE_AUDIT_FIELDS,
        *_PROVIDER_AUDIT_FIELDS,
        *_PARENT_TENSOR_AUDIT_FIELDS,
        _PROJECTION_COUNT_FIELD,
        _PROJECTION_INTERPRETATION_FIELD,
    }
)
_FIT_RECORD_FIELDS = frozenset(
    {
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
        "shared_gated_feature_sha256",
        "contrast_gated_feature_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_conformal_coefficient",
        "active_row_count",
        "top_mode_indices",
        "top_mode_norms",
        "balance_feature_std",
        "top2_modal_energy_fraction",
        "fit_record_sha256",
    }
)
_FOLD_FIELDS = frozenset(
    {
        "held_family_id",
        "train_example_ids",
        "train_family_ids",
        "train_fit_record_sha256s",
        "coefficients_by_conformal_coefficient",
        "unsupported_conformal_coefficient_indices",
        "active_row_count",
        "weighted_column_norm_by_conformal_coefficient",
        "weighted_design_rank",
        "normal_condition_number",
        "pre_projection_endpoint_operator_norms",
        "post_projection_endpoint_operator_norms",
        "trust_projection_scale",
        "linearized_rmse_before",
        "linearized_rmse_after",
        "trust_projection_applied",
        "ridge",
        "operator_norm_bound",
        "fold_receipt_sha256",
    }
)
_OOF_FIELDS = frozenset(
    {
        "example_id",
        "family_id",
        "held_family_id",
        "parent_signed_delta_nll_per_token",
        "predicted_candidate_signed_delta_nll_per_token",
        "exact_candidate_signed_delta_nll_per_token",
        "jacobian_by_conformal_coefficient",
        "coefficients_by_conformal_coefficient",
        "train_example_ids",
        "train_family_ids",
        "fit_record_sha256",
        "fold_receipt_sha256",
        "provider_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_observation_sha256",
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


def _float2(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _float4(
    value: object,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != CONFORMAL_COEFFICIENT_COUNT
    ):
        raise ValueError(f"{label} must contain exactly four values")
    return tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _fit_record_dict(value: object) -> dict[str, object]:
    row = (
        value.to_dict()
        if isinstance(value, GemmaIterativeConformalRouteFitRecord)
        else dict(_mapping(value, label="conformal-route fit record"))
    )
    if set(row) != _FIT_RECORD_FIELDS:
        raise ValueError("conformal-route fit-record fields differ")
    result = GemmaIterativeConformalRouteFitRecord(
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
        shared_gated_feature_sha256=row[
            "shared_gated_feature_sha256"
        ],  # type: ignore[arg-type]
        contrast_gated_feature_sha256=row[
            "contrast_gated_feature_sha256"
        ],  # type: ignore[arg-type]
        supervised_tokens=row["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=row[
            "parent_signed_delta_nll_per_token"
        ],  # type: ignore[arg-type]
        jacobian_by_conformal_coefficient=row[
            "jacobian_by_conformal_coefficient"
        ],  # type: ignore[arg-type]
        active_row_count=row["active_row_count"],  # type: ignore[arg-type]
        top_mode_indices=row["top_mode_indices"],  # type: ignore[arg-type]
        top_mode_norms=row["top_mode_norms"],  # type: ignore[arg-type]
        balance_feature_std=row["balance_feature_std"],  # type: ignore[arg-type]
        top2_modal_energy_fraction=row[
            "top2_modal_energy_fraction"
        ],  # type: ignore[arg-type]
    )
    if _canonical_json_bytes(result.to_dict()) != _canonical_json_bytes(row):
        raise ValueError("conformal-route fit-record hash mismatch")
    return result.to_dict()


def _validate_lineage(value: object) -> dict[str, str]:
    row = _mapping(value, label="conformal-route lineage")
    if set(row) != _REQUIRED_LINEAGE_KEYS:
        raise ValueError("conformal-route report lineage fields differ")
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
        raise ValueError("prior state-experts lineage aliases its prerequisite")
    return result


def _validate_resources(value: object) -> dict[str, object]:
    row = _mapping(value, label="conformal-route resources")
    if set(row) != _RESOURCE_KEYS:
        raise ValueError("conformal-route resource fields differ")
    integer_fields = (
        *_PROVIDER_RESOURCE_FIELD_ORDER,
        "serving_model_forward_count",
        "residual_width",
    )
    for name in integer_fields:
        if type(row[name]) is not int or int(row[name]) < 0:
            raise ValueError(f"conformal-route resource {name} is invalid")
    if int(row["residual_width"]) <= 0:
        raise ValueError("conformal-route residual width must be positive")
    if type(row["parent_head_reused_not_duplicated"]) is not bool:
        raise ValueError("conformal-route parent reuse receipt is invalid")
    provider_map = _mapping(
        row["candidate_provider_artifact_sha256_by_family"],
        label="conformal-route resource provider map",
    )
    if len(provider_map) != _EXPECTED_FAMILIES:
        raise ValueError("conformal-route resource provider map differs")
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
        label="conformal-route resource receipt",
    ):
        raise ValueError("conformal-route resource receipt hash mismatch")
    return {
        **payload,
        "resource_receipt_sha256": row["resource_receipt_sha256"],
    }


def _expected_recipe_value(value: object) -> object:
    if isinstance(value, tuple):
        return tuple(_expected_recipe_value(item) for item in value)
    return value


def _validate_audit(value: object) -> dict[str, object]:
    row = dict(_mapping(value, label="conformal-route execution audit"))
    if set(row) != _AUDIT_KEYS:
        raise ValueError("conformal-route execution audit fields differ")
    _assert_scalar_hash_only(row, path="conformal-route execution audit")
    for name, expected_raw in _RECIPE_AUDIT_FIELDS.items():
        expected = _expected_recipe_value(expected_raw)
        observed = row[name]
        if isinstance(expected, tuple):
            if not isinstance(observed, (tuple, list)):
                raise ValueError(f"conformal-route execution {name} differs")
            observed = tuple(observed)
            row[name] = observed
        if observed != expected:
            raise ValueError(f"conformal-route execution {name} differs")
    required = {
        "example_count": 16,
        "family_count": 8,
        "outer_fold_count": 8,
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
        _PROJECTION_INTERPRETATION_FIELD: _PROJECTION_INTERPRETATION,
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }
    if any(row.get(key) != expected for key, expected in required.items()):
        raise ValueError("conformal-route execution audit invariants differ")
    for name in (
        "residual_width",
        "parent_prepared_float_scalar_count",
        "parent_logical_macs_per_token_upper_bound",
        _PROJECTION_COUNT_FIELD,
    ):
        if type(row[name]) is not int or int(row[name]) < 0:
            raise ValueError(f"execution {name} must be nonnegative integer")
    if (
        int(row["residual_width"]) <= 0
        or int(row[_PROJECTION_COUNT_FIELD]) > _EXPECTED_FAMILIES
    ):
        raise ValueError("conformal-route execution count is outside its bound")
    for name in (
        "source_model_sha256",
        "source_execution_sha256",
        "parent_artifact_sha256",
        "parent_h4_artifact_sha256",
        "accepted_x4_head_sha256",
        "fit_manifest_sha256",
        "bridge_binding_sha256",
        *_PARENT_TENSOR_AUDIT_FIELDS,
    ):
        _require_sha256(row.get(name), label=f"execution {name}")
    for name in _PROVIDER_AUDIT_FIELDS:
        if name.endswith("_indices"):
            row[name] = _int2(row[name], label=f"execution {name}")
    for name, count in (
        ("model_inputs_sha256_by_example", _EXPECTED_EXAMPLES),
        ("parent_execution_sha256_by_example", _EXPECTED_EXAMPLES),
        ("candidate_execution_sha256_by_example", _EXPECTED_EXAMPLES),
        ("fold_provider_artifact_sha256_by_family", _EXPECTED_FAMILIES),
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
            sorted(row["fold_provider_artifact_sha256_by_family"].values())
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
                dict(_mapping(value, label="conformal-route fold receipt"))
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
        raise ValueError("conformal-route folds do not cover held families")
    for row in canonical:
        if set(row) != _FOLD_FIELDS:
            raise ValueError("serialized conformal-route fold fields differ")
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
            raise ValueError("conformal-route fold leaks or omits a family")
        replayed = fit_gemma_iterative_conformal_route_fold(
            tuple(fit_by_example[item] for item in train_examples),
            held_family_id=held,
        )
        if _canonical_json_bytes(replayed.to_dict()) != _canonical_json_bytes(
            row
        ):
            raise ValueError(
                "conformal-route fold coefficients do not replay from training"
            )
        coefficients = _float4(
            row["coefficients_by_conformal_coefficient"],
            label="fold conformal coefficients",
        )
        unsupported = tuple(
            row["unsupported_conformal_coefficient_indices"]
        )
        if (
            unsupported != tuple(sorted(set(unsupported)))
            or any(
                type(index) is not int
                or not 0 <= index < CONFORMAL_COEFFICIENT_COUNT
                for index in unsupported
            )
            or any(coefficients[index] != 0.0 for index in unsupported)
        ):
            raise ValueError("fold unsupported conformal coordinates invalid")
        column_norms = _float4(
            row["weighted_column_norm_by_conformal_coefficient"],
            label="fold conformal weighted column norms",
        )
        pre = _float2(
            row["pre_projection_endpoint_operator_norms"],
            label="fold pre-projection endpoint norms",
        )
        post = _float2(
            row["post_projection_endpoint_operator_norms"],
            label="fold post-projection endpoint norms",
        )
        scale = _finite(
            row["trust_projection_scale"],
            label="fold trust projection scale",
        )
        if (
            type(row["active_row_count"]) is not int
            or row["active_row_count"] <= 0
            or type(row["weighted_design_rank"]) is not int
            or not 0 <= row["weighted_design_rank"] <= 4
            or any(value < 0.0 for value in (*column_norms, *pre, *post))
            or not 0.0 < scale <= 1.0
            or max(post) > CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12
            or type(row["trust_projection_applied"]) is not bool
            or bool(row["trust_projection_applied"]) != (scale < 1.0)
        ):
            raise ValueError("conformal-route fold receipt is invalid")
        _finite(
            row["normal_condition_number"],
            label="fold normal condition number",
        )
        _assert_scalar_hash_only(row, path=f"conformal-route fold {held}")
    return canonical


def _provider_artifact_sha256(
    *,
    lineage: Mapping[str, str],
    audit: Mapping[str, object],
    fold: Mapping[str, object],
    top_mode_indices: tuple[int, int],
    top_mode_norms: tuple[float, float],
) -> str:
    return gemma_causal_top2_conformal_route_provider_artifact_sha256(
        parent_artifact_sha256=lineage["parent_artifact_sha256"],
        parent_h4_artifact_sha256=lineage["parent_h4_head_sha256"],
        bridge_binding_sha256=lineage["bridge_binding_sha256"],
        decoder_sha256=str(audit["parent_h4_decoder_sha256"]),
        lag_kernel_sha256=str(audit["parent_h4_lag_kernel_sha256"]),
        fold_receipt_sha256=str(fold["fold_receipt_sha256"]),
        top_mode_indices=top_mode_indices,
        top_mode_norms=top_mode_norms,
        coefficients_by_conformal_coefficient=fold[
            "coefficients_by_conformal_coefficient"
        ],
    )


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
                dict(_mapping(value, label="conformal-route OOF row"))
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
        raise ValueError("conformal-route OOF rows do not cover all examples")
    for row in rows:
        if set(row) != _OOF_FIELDS:
            raise ValueError("conformal-route OOF-row fields differ")
        example_id = _identifier(row["example_id"], label="OOF example")
        family_id = _identifier(row["family_id"], label="OOF family")
        parent = parent_by_example.get(example_id)
        candidate = candidate_by_example.get(example_id)
        fit = fit_by_example.get(example_id)
        fold = folds_by_family.get(family_id)
        if parent is None or candidate is None or fit is None or fold is None:
            raise ValueError("conformal-route OOF references unknown identity")
        jacobian = _float4(
            row["jacobian_by_conformal_coefficient"],
            label="OOF conformal Jacobian",
        )
        coefficients = _float4(
            row["coefficients_by_conformal_coefficient"],
            label="OOF conformal coefficients",
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
            or tuple(fit["jacobian_by_conformal_coefficient"]) != jacobian
            or tuple(fold["coefficients_by_conformal_coefficient"])
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
            raise ValueError("conformal-route OOF row does not replay")
    return rows


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    left_norm = math.sqrt(math.fsum(value * value for value in centered_left))
    right_norm = math.sqrt(math.fsum(value * value for value in centered_right))
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
            "parent_point_candidate_nll_vjp_d_plus_j_affine_conformal_theta"
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


def _median(values: Sequence[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("median requires values")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _coefficient_cosine(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    result = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
    if not math.isfinite(result):
        raise ValueError("fold coefficient cosine must be finite")
    return max(-1.0, min(1.0, result))


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
        if balance < 0.0 or not 0.0 <= energy <= 1.0:
            raise ValueError("conformal-route scientific statistic is invalid")
        family_rows.append(
            {
                "family_id": family_id,
                "prompt_count": len(records),
                "active_row_count": sum(
                    int(row["active_row_count"]) for row in records
                ),
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
    rank_by_family = {
        str(row["held_family_id"]): int(row["weighted_design_rank"])
        for row in folds
    }
    supported_fold_count_by_coordinate = {
        coordinate: sum(
            index
            not in tuple(
                row["unsupported_conformal_coefficient_indices"]
            )
            for row in folds
        )
        for index, coordinate in enumerate(_COEFFICIENT_ORDER)
    }
    conditions = tuple(
        _finite(
            row["normal_condition_number"],
            label="fold normal condition number",
        )
        for row in folds
    )
    median_condition = _median(conditions)
    coefficients = tuple(
        _float4(
            row["coefficients_by_conformal_coefficient"],
            label="fold conformal coefficients",
        )
        for row in folds
    )
    coefficient_norm_by_family = {
        str(row["held_family_id"]): math.sqrt(
            math.fsum(value * value for value in coefficient)
        )
        for row, coefficient in zip(folds, coefficients, strict=True)
    }
    zero_norm_family_ids = [
        family_id
        for family_id, norm in coefficient_norm_by_family.items()
        if norm == 0.0
    ]
    pairwise_cosines = tuple(
        _coefficient_cosine(coefficients[left], coefficients[right])
        for left in range(len(coefficients))
        for right in range(left + 1, len(coefficients))
    )
    if len(pairwise_cosines) != 28:
        raise ValueError("fold coefficient stability requires 28 pairs")
    mean_pairwise_cosine = math.fsum(pairwise_cosines) / len(
        pairwise_cosines
    )
    projected_family_ids = [
        str(row["held_family_id"])
        for row in folds
        if bool(row["trust_projection_applied"])
    ]
    gates = {
        "all_fold_weighted_design_ranks_exactly_4": (
            len(rank_by_family) == _EXPECTED_FAMILIES
            and all(value == 4 for value in rank_by_family.values())
        ),
        "all_4_conformal_coordinates_supported_in_every_fold": all(
            count == _EXPECTED_FAMILIES
            for count in supported_fold_count_by_coordinate.values()
        ),
        "family_macro_balance_feature_std_at_least_0_05": (
            macro_balance >= _BALANCE_FEATURE_STD_MIN
        ),
        "family_macro_top2_modal_energy_fraction_at_least_0_5": (
            macro_energy >= _TOP2_MODAL_ENERGY_FRACTION_MIN
        ),
        "median_fold_normal_condition_number_at_most_100": (
            median_condition <= _MEDIAN_DESIGN_CONDITION_MAX
        ),
        "mean_pairwise_fold_coefficient_cosine_at_least_0_90": (
            mean_pairwise_cosine
            >= _MEAN_FOLD_COEFFICIENT_COSINE_MIN
        ),
        "all_fold_coefficient_norms_positive": (
            not zero_norm_family_ids
        ),
    }
    gates["passed"] = all(gates.values())
    return (
        {
            "aggregation": (
                "prompt_mean_within_family_then_equal_family_macro"
            ),
            "conformal_coefficient_order": _COEFFICIENT_ORDER,
            "endpoint_order": _ENDPOINT_ORDER,
            "family_rows": family_rows,
            "family_macro_balance_feature_std": macro_balance,
            "minimum_family_macro_balance_feature_std": (
                _BALANCE_FEATURE_STD_MIN
            ),
            "family_macro_top2_modal_energy_fraction": macro_energy,
            "minimum_family_macro_top2_modal_energy_fraction": (
                _TOP2_MODAL_ENERGY_FRACTION_MIN
            ),
            "weighted_design_rank_by_held_family": rank_by_family,
            "full_rank_fold_count": sum(
                value == 4 for value in rank_by_family.values()
            ),
            "supported_fold_count_by_conformal_coordinate": (
                supported_fold_count_by_coordinate
            ),
            "all_coordinates_supported_fold_count": sum(
                not tuple(
                    row["unsupported_conformal_coefficient_indices"]
                )
                for row in folds
            ),
            "normal_condition_number_by_held_family": {
                str(row["held_family_id"]): _finite(
                    row["normal_condition_number"],
                    label="fold normal condition number",
                )
                for row in folds
            },
            "median_fold_normal_condition_number": median_condition,
            "maximum_median_fold_normal_condition_number": (
                _MEDIAN_DESIGN_CONDITION_MAX
            ),
            "fold_coefficient_pair_count": len(pairwise_cosines),
            "fold_coefficient_norm_by_held_family": (
                coefficient_norm_by_family
            ),
            "zero_norm_fold_count": len(zero_norm_family_ids),
            "zero_norm_held_family_ids": zero_norm_family_ids,
            "pairwise_fold_coefficient_cosines": pairwise_cosines,
            "mean_pairwise_fold_coefficient_cosine": (
                mean_pairwise_cosine
            ),
            "minimum_mean_pairwise_fold_coefficient_cosine": (
                _MEAN_FOLD_COEFFICIENT_COSINE_MIN
            ),
            "total_active_row_count": sum(
                int(row["active_row_count"]) for row in fits
            ),
            "trust_projection_fold_count": len(projected_family_ids),
            "trust_projection_family_ids": projected_family_ids,
            "pre_projection_endpoint_operator_norms_by_held_family": {
                str(row["held_family_id"]): _float2(
                    row["pre_projection_endpoint_operator_norms"],
                    label="pre-projection endpoint norms",
                )
                for row in folds
            },
            "post_projection_endpoint_operator_norms_by_held_family": {
                str(row["held_family_id"]): _float2(
                    row["post_projection_endpoint_operator_norms"],
                    label="post-projection endpoint norms",
                )
                for row in folds
            },
            "trust_projection_scale_by_held_family": {
                str(row["held_family_id"]): _finite(
                    row["trust_projection_scale"],
                    label="trust projection scale",
                )
                for row in folds
            },
        },
        gates,
    )


def _label_conformal_comparison(
    value: Mapping[str, object],
) -> dict[str, object]:
    result = dict(value)
    result["baseline_arm_id"] = "accepted_x4_plus_lag_b_parent"
    result["challenger_arm_id"] = (
        "causal_top2_lag_b_modal_balance_affine_conformal_route"
    )
    primary_name = (
        "family_macro_mean_prompt_absolute_delta_nll_per_token"
    )
    primary = dict(_mapping(result[primary_name], label="paired primary"))
    primary["parent"] = primary.pop("matched_alpha0")
    primary["conformal_route"] = primary.pop("challenger")
    result[primary_name] = primary
    family_rows: list[dict[str, object]] = []
    for raw in result["family_rows"]:
        row = dict(_mapping(raw, label="paired family row"))
        row["parent_mean_prompt_absolute_delta_nll_per_token"] = row.pop(
            "matched_alpha0_mean_prompt_absolute_delta_nll_per_token"
        )
        row[
            "conformal_route_mean_prompt_absolute_delta_nll_per_token"
        ] = row.pop("challenger_mean_prompt_absolute_delta_nll_per_token")
        family_rows.append(row)
    result["family_rows"] = family_rows
    secondary: list[dict[str, object]] = []
    for raw in result["secondary_metrics"]:
        row = dict(_mapping(raw, label="paired secondary metric"))
        row["parent"] = row.pop("matched_alpha0")
        row["conformal_route"] = row.pop("challenger")
        secondary.append(row)
    result["secondary_metrics"] = secondary
    return result


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
                "retained conformal-route iteration must bind its full fit"
            )
        return None
    if not retained:
        raise ValueError(
            "rejected conformal-route iteration cannot publish a full fit"
        )
    row = dict(
        _mapping(value, label="retained conformal-route full fit")
    )
    expected_fields = {
        "provider_artifact_sha256",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "bridge_binding_sha256",
        *_PROVIDER_RESOURCE_KEYS,
        "full_fit",
        "retention_receipt_sha256",
    }
    if set(row) != expected_fields:
        raise ValueError("retained conformal-route full-fit fields differ")
    replayed_fit = fit_gemma_iterative_conformal_route_fold(
        fits,
        held_family_id="__full_fit__",
    )
    submitted_fit = _mapping(
        row["full_fit"],
        label="retained conformal-route full fit",
    )
    if _canonical_json_bytes(replayed_fit.to_dict()) != _canonical_json_bytes(
        submitted_fit
    ):
        raise ValueError("retained conformal-route full fit does not replay")
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
        **{
            key: resources[key]
            for key in _PROVIDER_RESOURCE_FIELD_ORDER
        },
        "full_fit": replayed_fit.to_dict(),
    }
    if any(
        row[key] != expected
        for key, expected in payload.items()
        if key != "full_fit"
    ):
        raise ValueError(
            "retained conformal-route provider lineage or resources differ"
        )
    if _sha256(_RETENTION_DOMAIN, payload) != _require_sha256(
        row["retention_receipt_sha256"],
        label="conformal-route retention receipt",
    ):
        raise ValueError("conformal-route retention receipt hash mismatch")
    return {
        **payload,
        "retention_receipt_sha256": row["retention_receipt_sha256"],
    }


def build_gemma_iterative_conformal_route_report(
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
    """Build iteration four's exact family-disjoint retention report."""

    parent = _canonical_observations(
        parent_observations,
        label="conformal-route parent observations",
    )
    candidate = _canonical_observations(
        candidate_observations,
        label="conformal-route candidate observations",
    )
    if _source_grid(parent) != _source_grid(candidate):
        raise ValueError("conformal-route parent and candidate grids differ")
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
            "conformal-route lineage, execution, and resources contradict"
        )
    model_inputs_by_example = dict(
        _mapping(
            canonical_audit["model_inputs_sha256_by_example"],
            label="conformal-route model inputs",
        )
    )
    parent_execution_by_example = dict(
        _mapping(
            canonical_audit["parent_execution_sha256_by_example"],
            label="conformal-route parent executions",
        )
    )
    candidate_execution_by_example = dict(
        _mapping(
            canonical_audit["candidate_execution_sha256_by_example"],
            label="conformal-route candidate executions",
        )
    )
    audit_provider_by_family = dict(
        _mapping(
            canonical_audit["fold_provider_artifact_sha256_by_family"],
            label="conformal-route fold providers",
        )
    )
    resource_provider_by_family = dict(
        _mapping(
            canonical_resources[
                "candidate_provider_artifact_sha256_by_family"
            ],
            label="conformal-route resource providers",
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
        raise ValueError("conformal-route execution maps differ from manifest")

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
        raise ValueError("conformal-route fit records differ from manifest")
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
        raise ValueError("conformal-route fits disagree on mode constants")
    top_mode_indices = next(iter(top_mode_indices_set))
    top_mode_norms = next(iter(top_mode_norms_set))
    if top_mode_indices != tuple(
        canonical_audit["routed_parent_decoder_mode_indices"]
    ):
        raise ValueError("conformal-route fit modes differ from execution")
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
            or not math.isclose(
                float(fit["parent_signed_delta_nll_per_token"]),
                expected_parent_signed,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "conformal-route fit differs from parent observation"
            )

    folds = _validate_folds(
        fold_receipts,
        manifest=canonical_manifest,
        fit_by_example=fit_by_example,
    )
    folds_by_family = {
        str(row["held_family_id"]): row for row in folds
    }
    if canonical_audit[_PROJECTION_COUNT_FIELD] != sum(
        bool(row["trust_projection_applied"]) for row in folds
    ):
        raise ValueError(
            "conformal-route audit projection count differs from folds"
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
                "conformal-route provider does not replay from fold"
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
    paired = _label_conformal_comparison(
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
        label="conformal-route paired gates",
    )
    paired_primary = _mapping(
        paired[
            "family_macro_mean_prompt_absolute_delta_nll_per_token"
        ],
        label="conformal-route paired primary",
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
        "learned_parameter_count_exactly_4": (
            canonical_resources["learned_parameter_count"] == 4
        ),
        "derived_constant_float_count_exactly_2": (
            canonical_resources["derived_constant_float_count"] == 2
        ),
        "prepared_float_scalar_count_exactly_6": (
            canonical_resources["prepared_float_scalar_count"] == 6
        ),
        "runtime_state_float_count_per_sequence_exactly_2": (
            canonical_resources[
                "runtime_state_float_count_per_sequence"
            ]
            == 2
        ),
        "logical_macs_per_token_exactly_8": (
            canonical_resources["logical_macs_per_token_upper_bound"] == 8
        ),
        "nonlinear_scalar_ops_per_token_upper_bound_exactly_5": (
            canonical_resources[
                "nonlinear_scalar_ops_per_token_upper_bound"
            ]
            == 5
        ),
        "linear_accumulator_scalar_ops_per_token_upper_bound_exactly_4": (
            canonical_resources[
                "linear_accumulator_scalar_ops_per_token_upper_bound"
            ]
            == 4
        ),
        "zero_denominator_comparisons_per_token_upper_bound_exactly_1": (
            canonical_resources[
                "zero_denominator_comparisons_per_token_upper_bound"
            ]
            == 1
        ),
        "parent_decoder_invocations_per_token_exactly_1": (
            canonical_resources[
                "parent_decoder_invocations_per_token"
            ]
            == 1
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
        label="conformal-route candidate absolute gates",
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
            "iteration": 4,
            "candidate": (
                "causal_top2_lag_b_modal_balance_affine_conformal_route"
            ),
            "theta_zero_is_parent": True,
            "conformal_matrix_shape": (2, 2),
            "conformal_coefficient_order": _COEFFICIENT_ORDER,
            "endpoint_order": _ENDPOINT_ORDER,
            "route_state_semantics": _RECIPE_AUDIT_FIELDS[
                "route_state_semantics"
            ],
            "conformal_route_semantics": _RECIPE_AUDIT_FIELDS[
                "conformal_route_semantics"
            ],
            "fit_objective": (
                "per_prompt_d_plus_j_affine_conformal_theta_"
                "family_weighted_ridge"
            ),
            "retention_authority": (
                "exact_family_disjoint_out_of_fold_finite_metrics"
            ),
            "prior_iteration_must_be_rejected": True,
            "failure_scope": (
                "rejects_only_top2_lag_b_affine_conformal_balance_routing"
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
            "linearization": linearization,
            "conformal_support_condition_and_stability": scientific_metrics,
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
    _assert_scalar_hash_only(payload, path="conformal-route report")
    return {**payload, "report_sha256": _sha256(_REPORT_DOMAIN, payload)}


def validate_gemma_iterative_conformal_route_report(
    report: Mapping[str, object],
) -> None:
    """Replay every scalar receipt, metric, gate, and decision."""

    row = dict(_mapping(report, label="conformal-route report"))
    report_sha256 = _require_sha256(
        row.pop("report_sha256", None),
        label="conformal-route report",
    )
    if _sha256(_REPORT_DOMAIN, row) != report_sha256:
        raise ValueError("conformal-route report hash mismatch")
    if (
        row.get("schema") != _SCHEMA
        or row.get("format_version") != _FORMAT_VERSION
        or row.get("safety") != _SAFETY
    ):
        raise ValueError("conformal-route report header or safety differs")
    manifest_payload = _mapping(row.get("manifest"), label="manifest payload")
    observations = _mapping(row.get("observations"), label="observations")
    parent_raw = observations.get("parent")
    candidate_raw = observations.get("candidate")
    if not isinstance(parent_raw, list) or not isinstance(candidate_raw, list):
        raise TypeError("serialized observations must be lists")
    parent = tuple(
        _observation_from_dict(
            value,
            label="conformal-route parent observation",
        )
        for value in parent_raw
    )
    candidate = tuple(
        _observation_from_dict(
            value,
            label="conformal-route candidate observation",
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
            "serialized conformal-route fit/fold/OOF rows must be lists"
        )
    rebuilt = build_gemma_iterative_conformal_route_report(
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
            label="conformal-route lineage",
        ),
        resources=_mapping(
            row.get("resources"),
            label="conformal-route resources",
        ),
        audit=_mapping(
            row.get("execution"),
            label="conformal-route execution",
        ),
        retained_full_fit_receipt=(
            None
            if row.get("retained_full_fit") is None
            else _mapping(
                row.get("retained_full_fit"),
                label="retained conformal-route full fit",
            )
        ),
        provisional=False,
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(report):
        raise ValueError("conformal-route report derived state does not replay")
