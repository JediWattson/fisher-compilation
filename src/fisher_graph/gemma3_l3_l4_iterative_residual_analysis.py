"""Tensor-free replay analysis for the first iterative residual boost.

The live campaign owns model forwards.  This module owns the retention
decision and reconstructs it entirely from scalar observations, four-value
behavioral linearizations, family-blocked fold receipts, and hashes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re

from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
    _fidelity_from_observations,
    _paired_comparison,
)
from .gemma3_l3_l4_iterative_residual_boost import (
    GemmaIterativeResidualFitRecord,
    GemmaIterativeResidualFoldFit,
    fit_gemma_iterative_residual_fold,
    gemma_causal_position_scale_provider_artifact_sha256,
)


__all__ = [
    "build_gemma_iterative_residual_report",
    "validate_gemma_iterative_residual_report",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_iterative_residual_analysis"
_FORMAT_VERSION = 1
_REPORT_DOMAIN = b"fisher-graph:gemma-iterative-residual-report:v1\0"
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-collection:v1\0"
)
_RESOURCE_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-resources:v1\0"
)
_RETENTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-residual-retained-provider:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PER_FAMILY = 2
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
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "learned_parameter_count",
        "logical_macs_per_token_upper_bound",
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
        "position_bin_count",
        "position_bin_semantics",
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
        "fold_linearization_extrapolation_count",
        "coefficient_clipping_interpretation",
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


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float4(value: object, *, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four values")
    result = tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _assert_scalar_hash_only(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _assert_scalar_hash_only(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_scalar_hash_only(nested, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported payload {type(value)!r}")


def _observation_dict(
    value: GemmaH4DampingFiniteNLLObservation,
) -> dict[str, object]:
    if not isinstance(value, GemmaH4DampingFiniteNLLObservation):
        raise TypeError("finite-NLL observations have the wrong type")
    return value.to_dict()


def _observation_from_dict(
    value: object,
    *,
    label: str,
) -> GemmaH4DampingFiniteNLLObservation:
    row = _mapping(value, label=label)
    expected = {
        "example_id",
        "family_id",
        "supervised_tokens",
        "source_summed_nll",
        "candidate_summed_nll",
        "source_to_candidate_summed_kl",
        "top1_matches",
        "source_logits_sha256",
        "candidate_logits_sha256",
        "targets_sha256",
        "observation_sha256",
    }
    if set(row) != expected:
        raise ValueError(f"{label} fields differ")
    result = GemmaH4DampingFiniteNLLObservation(
        example_id=row["example_id"],  # type: ignore[arg-type]
        family_id=row["family_id"],  # type: ignore[arg-type]
        supervised_tokens=row["supervised_tokens"],  # type: ignore[arg-type]
        source_summed_nll=row["source_summed_nll"],  # type: ignore[arg-type]
        candidate_summed_nll=row["candidate_summed_nll"],  # type: ignore[arg-type]
        source_to_candidate_summed_kl=row[
            "source_to_candidate_summed_kl"
        ],  # type: ignore[arg-type]
        top1_matches=row["top1_matches"],  # type: ignore[arg-type]
        source_logits_sha256=row["source_logits_sha256"],  # type: ignore[arg-type]
        candidate_logits_sha256=row[
            "candidate_logits_sha256"
        ],  # type: ignore[arg-type]
        targets_sha256=row["targets_sha256"],  # type: ignore[arg-type]
    )
    if result.observation_sha256 != row["observation_sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return result


def _fit_record_dict(value: object) -> dict[str, object]:
    if isinstance(value, GemmaIterativeResidualFitRecord):
        return value.to_dict()
    row = _mapping(value, label="fit record")
    expected = {
        "example_id",
        "family_id",
        "model_inputs_sha256",
        "parent_execution_sha256",
        "parent_observation_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_bin",
        "active_rows_by_bin",
        "fit_record_sha256",
    }
    if set(row) != expected:
        raise ValueError("fit-record fields differ")
    result = GemmaIterativeResidualFitRecord(
        example_id=row["example_id"],  # type: ignore[arg-type]
        family_id=row["family_id"],  # type: ignore[arg-type]
        model_inputs_sha256=row["model_inputs_sha256"],  # type: ignore[arg-type]
        parent_execution_sha256=row[
            "parent_execution_sha256"
        ],  # type: ignore[arg-type]
        parent_observation_sha256=row[
            "parent_observation_sha256"
        ],  # type: ignore[arg-type]
        supervised_tokens=row["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=row[
            "parent_signed_delta_nll_per_token"
        ],  # type: ignore[arg-type]
        jacobian_by_bin=row["jacobian_by_bin"],  # type: ignore[arg-type]
        active_rows_by_bin=row["active_rows_by_bin"],  # type: ignore[arg-type]
    )
    if result.fit_record_sha256 != row["fit_record_sha256"]:
        raise ValueError("fit-record hash mismatch")
    return result.to_dict()


def _canonical_observations(
    values: Sequence[GemmaH4DampingFiniteNLLObservation],
    *,
    label: str,
) -> tuple[GemmaH4DampingFiniteNLLObservation, ...]:
    result = tuple(sorted(values, key=lambda row: row.example_id))
    family_counts = Counter(row.family_id for row in result)
    if (
        len(result) != _EXPECTED_EXAMPLES
        or len({row.example_id for row in result}) != _EXPECTED_EXAMPLES
        or len(family_counts) != _EXPECTED_FAMILIES
        or set(family_counts.values()) != {_EXPECTED_PER_FAMILY}
    ):
        raise ValueError(f"{label} must contain a strict 16-by-8 panel")
    return result


def _source_grid(
    values: Sequence[GemmaH4DampingFiniteNLLObservation],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            row.example_id,
            row.family_id,
            row.supervised_tokens,
            row.source_summed_nll,
            row.source_logits_sha256,
            row.targets_sha256,
        )
        for row in values
    )


def _relative_improvement(new: float, parent: float) -> float:
    if new < 0.0 or parent < 0.0:
        raise ValueError("relative metrics must be nonnegative")
    if parent == 0.0:
        return 0.0 if new == 0.0 else -1.0
    return 1.0 - new / parent


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


def _correct_prompt_disagreement_quantile_labels(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Name ``1 - p10(agreement)`` as p90 disagreement in this report."""

    result = dict(value)
    secondary = [
        dict(_mapping(row, label="paired secondary metric"))
        for row in result.get("secondary_metrics", ())
    ]
    old_metric = "per_prompt_p10_top1_disagreement_to_source"
    new_metric = "per_prompt_p90_top1_disagreement_to_source"
    renamed = 0
    for row in secondary:
        if row.get("metric") == old_metric:
            row["metric"] = new_metric
            renamed += 1
    gates = dict(_mapping(result.get("gates"), label="paired gates"))
    old_gate = "prompt_p10_top1_disagreement_regression_at_most_2pct"
    new_gate = "prompt_p90_top1_disagreement_regression_at_most_2pct"
    if renamed != 1 or old_gate not in gates or new_gate in gates:
        raise ValueError("paired prompt-tail metric schema differs")
    gates[new_gate] = gates.pop(old_gate)
    result["secondary_metrics"] = secondary
    result["gates"] = gates
    return result


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
    predicted_abs = math.fsum(abs(value) for value in predicted) / len(rows)
    exact_abs = math.fsum(abs(value) for value in exact) / len(rows)
    parent_abs = math.fsum(abs(value) for value in parent) / len(rows)
    sign_matches = sum(
        (predicted_value > 0) == (exact_value > 0)
        and (predicted_value < 0) == (exact_value < 0)
        for predicted_value, exact_value in zip(predicted, exact, strict=True)
    )
    return {
        "objective": "parent_point_candidate_nll_vjp_d_plus_j_theta",
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
    row = _mapping(value, label="lineage")
    if set(row) != _REQUIRED_LINEAGE_KEYS:
        raise ValueError("iterative report lineage fields differ")
    return {
        key: _require_sha256(row[key], label=f"lineage {key}")
        for key in sorted(row)
    }


def _validate_resources(value: object) -> dict[str, object]:
    row = _mapping(value, label="resources")
    if set(row) != _RESOURCE_KEYS:
        raise ValueError("iterative report resource fields differ")
    parameters = row["learned_parameter_count"]
    macs = row["logical_macs_per_token_upper_bound"]
    forwards = row["serving_model_forward_count"]
    reused = row["parent_head_reused_not_duplicated"]
    residual_width = row["residual_width"]
    provider_map = _mapping(
        row["candidate_provider_artifact_sha256_by_family"],
        label="resource provider map",
    )
    if (
        type(parameters) is not int
        or parameters < 0
        or type(macs) is not int
        or macs < 0
        or type(forwards) is not int
        or forwards < 0
        or type(reused) is not bool
        or type(residual_width) is not int
        or residual_width <= 0
        or len(provider_map) != _EXPECTED_FAMILIES
    ):
        raise ValueError("iterative report resources are invalid")
    canonical_provider_map = {
        _identifier(key, label="resource family_id"): _require_sha256(
            provider,
            label="resource provider artifact",
        )
        for key, provider in sorted(provider_map.items())
    }
    payload = {
        "learned_parameter_count": parameters,
        "logical_macs_per_token_upper_bound": macs,
        "serving_model_forward_count": forwards,
        "parent_head_reused_not_duplicated": reused,
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
        "residual_width": residual_width,
    }
    if _sha256(_RESOURCE_DOMAIN, payload) != _require_sha256(
        row["resource_receipt_sha256"],
        label="resource receipt",
    ):
        raise ValueError("resource receipt hash mismatch")
    return {
        **payload,
        "resource_receipt_sha256": row["resource_receipt_sha256"],
    }


def _validate_audit(value: object) -> dict[str, object]:
    row = dict(_mapping(value, label="execution audit"))
    if set(row) != _AUDIT_KEYS:
        raise ValueError("iterative execution audit fields differ")
    _assert_scalar_hash_only(row, path="execution audit")
    required = {
        "execution_mode": (
            "fit_only_two_phase_family_blocked_iterative_residual"
        ),
        "example_count": 16,
        "family_count": 8,
        "outer_fold_count": 8,
        "position_bin_count": 4,
        "position_bin_semantics": (
            "causal_logical_position_[0_3]_[4_7]_[8_15]_[16_plus]"
        ),
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
        "coefficient_clipping_interpretation": (
            "linearization_extrapolation_not_free_improvement"
        ),
        "selection_input_opened": False,
        "guard_input_opened": False,
        "calibration_b_opened": False,
        "assessment_input_opened": False,
        "development_only": True,
    }
    if any(row.get(key) != expected for key, expected in required.items()):
        raise ValueError("iterative execution audit invariants differ")
    for name in (
        "residual_width",
        "parent_prepared_float_scalar_count",
        "parent_logical_macs_per_token_upper_bound",
        "fold_linearization_extrapolation_count",
    ):
        if type(row[name]) is not int or int(row[name]) < 0:
            raise ValueError(f"execution {name} must be nonnegative integer")
    if int(row["residual_width"]) <= 0:
        raise ValueError("execution residual_width must be positive")
    for name in (
        "source_model_sha256",
        "source_execution_sha256",
        "parent_artifact_sha256",
        "parent_h4_artifact_sha256",
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


def _validate_manifest(
    value: object,
    *,
    observations: Sequence[GemmaH4DampingFiniteNLLObservation],
) -> dict[str, str]:
    row = _mapping(value, label="manifest")
    manifest = {
        _identifier(key, label="manifest example_id"): _identifier(
            family,
            label="manifest family_id",
        )
        for key, family in row.items()
    }
    observed = {item.example_id: item.family_id for item in observations}
    if manifest != observed:
        raise ValueError("manifest differs from observations")
    return dict(sorted(manifest.items()))


def _validate_folds(
    folds: Sequence[Mapping[str, object]],
    *,
    manifest: Mapping[str, str],
    fit_by_example: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    canonical = tuple(
        sorted(
            (dict(_mapping(value, label="fold receipt")) for value in folds),
            key=lambda row: str(row.get("held_family_id")),
        )
    )
    families = set(manifest.values())
    if (
        len(canonical) != _EXPECTED_FAMILIES
        or {row.get("held_family_id") for row in canonical} != families
    ):
        raise ValueError("fold receipts do not cover all held families")
    for row in canonical:
        expected_fields = {
            "held_family_id",
            "train_example_ids",
            "train_family_ids",
            "train_fit_record_sha256s",
            "coefficients_by_bin",
            "unsupported_bin_indices",
            "active_rows_by_bin",
            "weighted_column_norm_by_bin",
            "normal_condition_number",
            "linearized_rmse_before",
            "linearized_rmse_after",
            "linearization_extrapolation",
            "ridge",
            "trust_bound",
            "fold_receipt_sha256",
        }
        if set(row) != expected_fields:
            raise ValueError("serialized fold-fit fields differ")
        held = _identifier(row.get("held_family_id"), label="held family")
        train_examples = tuple(row.get("train_example_ids", ()))
        train_families = tuple(row.get("train_family_ids", ()))
        coefficients = _float4(
            row.get("coefficients_by_bin"),
            label="fold coefficients",
        )
        expected_train_examples = tuple(
            sorted(
                example_id
                for example_id, family_id in manifest.items()
                if family_id != held
            )
        )
        expected_train_families = tuple(sorted(families - {held}))
        if (
            train_examples != tuple(sorted(set(train_examples)))
            or train_families != tuple(sorted(set(train_families)))
            or len(train_examples) != 14
            or len(train_families) != 7
            or held in train_families
            or any(manifest.get(str(item)) == held for item in train_examples)
            or set(str(item) for item in train_families)
            != families - {held}
            or train_examples != expected_train_examples
            or train_families != expected_train_families
        ):
            raise ValueError("fold receipt leaks or omits a family")
        replayed = fit_gemma_iterative_residual_fold(
            tuple(fit_by_example[example_id] for example_id in train_examples),
            held_family_id=held,
        )
        if _canonical_json_bytes(replayed.to_dict()) != _canonical_json_bytes(
            row
        ):
            raise ValueError(
                "fold coefficients do not replay from declared training rows"
            )
        if tuple(row.get("coefficients_by_bin", ())) != coefficients:
            raise ValueError("fold coefficients are not canonical")
        _assert_scalar_hash_only(row, path=f"fold {held}")
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
            (dict(_mapping(value, label="OOF row")) for value in values),
            key=lambda row: str(row.get("example_id")),
        )
    )
    expected_fields = {
        "example_id",
        "family_id",
        "held_family_id",
        "parent_signed_delta_nll_per_token",
        "predicted_candidate_signed_delta_nll_per_token",
        "exact_candidate_signed_delta_nll_per_token",
        "jacobian_by_bin",
        "coefficients_by_bin",
        "train_example_ids",
        "train_family_ids",
        "fit_record_sha256",
        "fold_receipt_sha256",
        "provider_artifact_sha256",
        "candidate_execution_sha256",
        "candidate_observation_sha256",
    }
    if (
        len(rows) != _EXPECTED_EXAMPLES
        or {str(row.get("example_id")) for row in rows}
        != set(parent_by_example)
    ):
        raise ValueError("OOF rows differ from observation membership")
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("OOF row fields differ")
        example_id = _identifier(row["example_id"], label="OOF example_id")
        family_id = _identifier(row["family_id"], label="OOF family_id")
        if row["held_family_id"] != family_id:
            raise ValueError("OOF row was not evaluated by its held fold")
        parent = parent_by_example[example_id]
        candidate = candidate_by_example[example_id]
        fit = fit_by_example[example_id]
        fold = folds_by_family[family_id]
        jacobian = _float4(row["jacobian_by_bin"], label="OOF jacobian")
        coefficients = _float4(
            row["coefficients_by_bin"],
            label="OOF coefficients",
        )
        parent_signed = (
            parent.candidate_summed_nll - parent.source_summed_nll
        ) / parent.supervised_tokens
        exact_signed = (
            candidate.candidate_summed_nll - candidate.source_summed_nll
        ) / candidate.supervised_tokens
        predicted = parent_signed + math.fsum(
            left * right
            for left, right in zip(jacobian, coefficients, strict=True)
        )
        if (
            family_id != parent.family_id
            or candidate.family_id != family_id
            or tuple(fit["jacobian_by_bin"]) != jacobian
            or tuple(fold["coefficients_by_bin"]) != coefficients
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
            raise ValueError("OOF row does not replay from its inputs")
    return rows


def _validate_retained_full_fit(
    value: object,
    *,
    retained: bool,
    provisional: bool,
    fits: Sequence[Mapping[str, object]],
    lineage: Mapping[str, str],
    resources: Mapping[str, object],
) -> Mapping[str, object] | None:
    if value is None:
        if retained and not provisional:
            raise ValueError(
                "a retained iteration must bind its full-fit provider"
            )
        return None
    if not retained:
        raise ValueError("a rejected iteration cannot publish a full fit")
    row = dict(_mapping(value, label="retained full fit"))
    expected_fields = {
        "provider_artifact_sha256",
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "bridge_binding_sha256",
        "learned_parameter_count",
        "logical_macs_per_token_upper_bound",
        "full_fit",
        "retention_receipt_sha256",
    }
    if set(row) != expected_fields:
        raise ValueError("retained full-fit receipt fields differ")
    replayed_fit = fit_gemma_iterative_residual_fold(
        fits,
        held_family_id="__full_fit__",
    )
    submitted_fit = _mapping(row["full_fit"], label="retained full fit")
    if _canonical_json_bytes(replayed_fit.to_dict()) != _canonical_json_bytes(
        submitted_fit
    ):
        raise ValueError("retained full fit does not replay from all records")
    expected_provider = (
        gemma_causal_position_scale_provider_artifact_sha256(
            parent_artifact_sha256=lineage["parent_artifact_sha256"],
            parent_h4_artifact_sha256=lineage[
                "parent_h4_head_sha256"
            ],
            bridge_binding_sha256=lineage["bridge_binding_sha256"],
            fold_receipt_sha256=replayed_fit.fold_receipt_sha256,
            coefficients_by_bin=replayed_fit.coefficients_by_bin,
        )
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
        "full_fit": replayed_fit.to_dict(),
    }
    if any(row[key] != expected for key, expected in payload.items()):
        raise ValueError("retained provider lineage or resources differ")
    if _sha256(_RETENTION_DOMAIN, payload) != _require_sha256(
        row["retention_receipt_sha256"],
        label="retention receipt",
    ):
        raise ValueError("retention receipt hash mismatch")
    return {
        **payload,
        "retention_receipt_sha256": row["retention_receipt_sha256"],
    }


def build_gemma_iterative_residual_report(
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
    """Build the exact OOF retention report from scalar/hash-only inputs."""

    parent = _canonical_observations(
        parent_observations,
        label="parent observations",
    )
    candidate = _canonical_observations(
        candidate_observations,
        label="candidate observations",
    )
    if _source_grid(parent) != _source_grid(candidate):
        raise ValueError("parent and candidate source authority grids differ")
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
            "lineage, execution, and resource receipts contradict"
        )
    model_inputs_by_example = dict(
        _mapping(
            canonical_audit["model_inputs_sha256_by_example"],
            label="model input receipts",
        )
    )
    parent_execution_by_example = dict(
        _mapping(
            canonical_audit["parent_execution_sha256_by_example"],
            label="parent execution receipts",
        )
    )
    candidate_execution_by_example = dict(
        _mapping(
            canonical_audit["candidate_execution_sha256_by_example"],
            label="candidate execution receipts",
        )
    )
    audit_provider_by_family = dict(
        _mapping(
            canonical_audit[
                "fold_provider_artifact_sha256_by_family"
            ],
            label="fold provider receipts",
        )
    )
    resource_provider_by_family = dict(
        _mapping(
            canonical_resources[
                "candidate_provider_artifact_sha256_by_family"
            ],
            label="resource provider receipts",
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
        raise ValueError("execution identity maps differ from the manifest")

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
        raise ValueError("fit records differ from the 16-example manifest")
    parent_by_example = {row.example_id: row for row in parent}
    candidate_by_example = {row.example_id: row for row in candidate}
    fit_by_example = {str(row["example_id"]): row for row in fits}
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
            raise ValueError("fit record differs from its parent observation")

    folds = _validate_folds(
        fold_receipts,
        manifest=canonical_manifest,
        fit_by_example=fit_by_example,
    )
    folds_by_family = {
        str(row["held_family_id"]): row for row in folds
    }
    if canonical_audit["fold_linearization_extrapolation_count"] != sum(
        bool(row["linearization_extrapolation"]) for row in folds
    ):
        raise ValueError(
            "execution clipping count differs from replayed fold fits"
        )
    for family_id, fold in folds_by_family.items():
        expected_provider = (
            gemma_causal_position_scale_provider_artifact_sha256(
                parent_artifact_sha256=canonical_lineage[
                    "parent_artifact_sha256"
                ],
                parent_h4_artifact_sha256=canonical_lineage[
                    "parent_h4_head_sha256"
                ],
                bridge_binding_sha256=canonical_lineage[
                    "bridge_binding_sha256"
                ],
                fold_receipt_sha256=str(
                    fold["fold_receipt_sha256"]
                ),
                coefficients_by_bin=fold["coefficients_by_bin"],  # type: ignore[arg-type]
            )
        )
        if resource_provider_by_family[family_id] != expected_provider:
            raise ValueError(
                "OOF provider artifact does not replay from its fold"
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
    paired = _correct_prompt_disagreement_quantile_labels(
        _paired_comparison(
            parent_fidelity,
            candidate_fidelity,
            baseline_observations=parent_payloads,
            challenger_observations=candidate_payloads,
        )
    )
    resource_gates = {
        "learned_parameter_count_exactly_4": (
            canonical_resources["learned_parameter_count"] == 4
        ),
        "logical_macs_per_token_equals_residual_width": (
            canonical_resources["logical_macs_per_token_upper_bound"]
            == canonical_resources["residual_width"]
        ),
        "logical_macs_per_token_at_most_1024": (
            int(
                canonical_resources[
                    "logical_macs_per_token_upper_bound"
                ]
            )
            <= 1_024
        ),
        "serving_model_forward_count_exactly_1": (
            canonical_resources["serving_model_forward_count"] == 1
        ),
        "parent_head_reused_not_duplicated": (
            canonical_resources["parent_head_reused_not_duplicated"] is True
        ),
    }
    resource_gates["passed"] = all(resource_gates.values())
    paired_gates = _mapping(paired["gates"], label="paired gates")
    retained = bool(paired_gates["passed"]) and bool(
        resource_gates["passed"]
    )
    candidate_absolute_gates = _mapping(
        candidate_fidelity["gates"],
        label="candidate absolute gates",
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
    )
    linearization = _linearization_diagnostics(canonical_oof)
    clipping_folds = [
        str(row["held_family_id"])
        for row in folds
        if bool(row.get("linearization_extrapolation", False))
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
            "iteration": 1,
            "candidate": "four_causal_position_scales_over_frozen_lag_b",
            "theta_zero_is_parent": True,
            "position_bins": ("0_3", "4_7", "8_15", "16_plus"),
            "fit_objective": "per_prompt_d_plus_j_theta_weighted_ridge",
            "retention_authority": (
                "exact_family_disjoint_out_of_fold_finite_metrics"
            ),
            "h4_rmse_is_diagnostic_only": True,
            "failure_scope": (
                "rejects_only_position_dependent_lag_b_amplitude"
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
                "trust_bound_hit_fold_count": len(clipping_folds),
                "trust_bound_hit_family_ids": clipping_folds,
            },
        },
        "decision": {
            "behavior_relative_gates_passed": bool(
                paired_gates["passed"]
            ),
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
    _assert_scalar_hash_only(payload, path="iterative report")
    return {**payload, "report_sha256": _sha256(_REPORT_DOMAIN, payload)}


def validate_gemma_iterative_residual_report(
    report: Mapping[str, object],
) -> None:
    """Replay every derived metric and decision in a serialized report."""

    row = dict(_mapping(report, label="iterative report"))
    report_sha256 = _require_sha256(
        row.pop("report_sha256", None),
        label="iterative report",
    )
    if _sha256(_REPORT_DOMAIN, row) != report_sha256:
        raise ValueError("iterative report hash mismatch")
    if (
        row.get("schema") != _SCHEMA
        or row.get("format_version") != _FORMAT_VERSION
        or row.get("safety") != _SAFETY
    ):
        raise ValueError("iterative report header or safety differs")
    manifest_payload = _mapping(row.get("manifest"), label="manifest payload")
    observations = _mapping(row.get("observations"), label="observations")
    parent_raw = observations.get("parent")
    candidate_raw = observations.get("candidate")
    if not isinstance(parent_raw, list) or not isinstance(candidate_raw, list):
        raise TypeError("serialized observations must be lists")
    parent = tuple(
        _observation_from_dict(value, label="parent observation")
        for value in parent_raw
    )
    candidate = tuple(
        _observation_from_dict(value, label="candidate observation")
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
        raise TypeError("serialized fit/fold/OOF rows must be lists")
    rebuilt = build_gemma_iterative_residual_report(
        parent_observations=parent,
        candidate_observations=candidate,
        oof_rows=oof,
        fit_records=fit_records,
        fold_receipts=folds,
        manifest=_mapping(
            manifest_payload.get("family_by_example"),
            label="manifest family map",
        ),  # type: ignore[arg-type]
        lineage=_mapping(row.get("lineage"), label="lineage"),
        resources=_mapping(row.get("resources"), label="resources"),
        audit=_mapping(row.get("execution"), label="execution"),
        retained_full_fit_receipt=(
            None
            if row.get("retained_full_fit") is None
            else _mapping(
                row.get("retained_full_fit"),
                label="retained full fit",
            )
        ),
        provisional=False,
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(report):
        raise ValueError("iterative report derived state does not replay")
