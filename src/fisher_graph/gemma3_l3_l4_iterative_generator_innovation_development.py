"""Gemma development report for the fixed-basis innovation controller.

This is the authenticated Phase-B boundary from the frozen generator plan.
It consumes two aligned panels of prompt-local sufficient statistics:

``legacy Q6``
    Exact token-loss directional moments for the six cumulative occupancy
    coordinates.

``generator R4``
    Exact token-loss directional moments for two fixed-basis generator
    coordinates and their two causal-innovation-conditioned counterparts.

The raw token score rows, modal rows, activations, gradients, logits, token
ids, and prompt text are deliberately absent from the serialized report.
Per-example causal features are represented only by aggregate moments, sign
counts, causal/chunk audit flags, and a hash of the transient bounded feature
trace.

The fixed plan is validated before fitting, and its exact 6-by-2 basis is
passed to the nested family-held-out screen.  A successful derivative screen
authorizes opening the preregistered finite-displacement evidence exactly
once; it never compiles a provider by itself.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
)
from .gemma3_l3_l4_iterative_generator_innovation_edges import (
    GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
)
from .gemma3_l3_l4_iterative_generator_innovation_plan import (
    GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA,
    validate_gemma_iterative_generator_innovation_plan,
)
from .gemma3_l3_l4_iterative_occupancy_route import (
    OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
)
from .gemma3_l3_l4_progressive_worker import (
    gemma_progressive_panel_membership_receipt_sha256,
)
from .token_loss_fisher import (
    TokenLossFisherPromptRecord,
    token_loss_fisher_prompt_record_from_dict,
)
from .token_loss_fisher_generator_innovation import (
    GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS,
    GENERATOR_INNOVATION_GATE_CONFIG,
    GENERATOR_INNOVATION_SCHEMA,
    GENERATOR_INNOVATION_SHARED_RIDGE,
    build_generator_innovation_nested_lofo_report,
    replay_generator_innovation_nested_lofo_report,
    validate_generator_innovation_nested_lofo_report,
)


__all__ = [
    "GEMMA_ITERATIVE_GENERATOR_INNOVATION_DEVELOPMENT_SCHEMA",
    "build_gemma_iterative_generator_innovation_development_report",
    "publish_gemma_iterative_generator_innovation_development_report",
    "replay_gemma_iterative_generator_innovation_development_report",
    "validate_gemma_iterative_generator_innovation_development_report",
]


GEMMA_ITERATIVE_GENERATOR_INNOVATION_DEVELOPMENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_generator_innovation_development.v1"
)

_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS_PER_FAMILY = 2
_COLLECTION_ROLE = "calibration_a_fit"
_REPORT_DOMAIN = (
    b"fisher-graph:gemma-iterative-generator-innovation-development:v1\0"
)
_FEATURE_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-feature-receipt:v1\0"
)
_FEATURE_AUDIT_DOMAIN = (
    b"fisher-graph:gemma-generator-innovation-feature-audit:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_SUMMARY_FIELDS = {
    "active_activation_row_count",
    "mean_by_channel",
    "second_moment_by_channel",
    "mean_absolute_by_channel",
    "maximum_absolute_by_channel",
    "positive_count_by_channel",
    "negative_count_by_channel",
    "zero_count_by_channel",
    "bounded_innovation_trace_sha256",
    "whole_sequence_equals_two_chunks",
    "prior_excludes_current_activation",
    "padding_updates_state",
}
_TOP_MODE_FIELDS = {"top_mode_indices", "top_mode_norms"}
_COLLECTION_LINEAGE_REQUIRED_FIELDS = {
    "plan_sha256",
    "plan_file_sha256",
    "basis_sha256",
    "collection_role_input_file_sha256",
    "collection_manifest_sha256",
    "collection_membership_receipt_sha256",
}
_COLLECTION_LINEAGE_OPTIONAL_FIELD = (
    "prompt_free_panel_artifact_receipt_sha256"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float_pair(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{label} must contain two values")
    return (
        _finite(value[0], label=f"{label}[0]"),
        _finite(value[1], label=f"{label}[1]"),
    )


def _count_pair(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{label} must contain two nonnegative integers")
    return int(value[0]), int(value[1])


def _records(
    legacy_records: Sequence[object],
    generator_records: Sequence[object],
) -> tuple[
    tuple[TokenLossFisherPromptRecord, ...],
    tuple[TokenLossFisherPromptRecord, ...],
]:
    def parse(
        rows: Sequence[object],
        *,
        label: str,
    ) -> tuple[TokenLossFisherPromptRecord, ...]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError(f"{label} must be a sequence")
        result = tuple(
            sorted(
                (
                    row
                    if isinstance(row, TokenLossFisherPromptRecord)
                    else token_loss_fisher_prompt_record_from_dict(
                        _mapping(row, label=f"{label} row")
                    )
                    for row in rows
                ),
                key=lambda row: row.example_id,
            )
        )
        for row in result:
            row.validate_integrity()
        return result

    legacy = parse(legacy_records, label="legacy prompt records")
    generator = parse(generator_records, label="generator prompt records")
    if (
        len(legacy) != _EXPECTED_EXAMPLES
        or len(generator) != _EXPECTED_EXAMPLES
        or len({row.example_id for row in legacy}) != _EXPECTED_EXAMPLES
        or len({row.example_id for row in generator}) != _EXPECTED_EXAMPLES
        or tuple(row.example_id for row in legacy)
        != tuple(row.example_id for row in generator)
    ):
        raise ValueError("generator innovation example geometry differs")
    if any(
        row.coordinate_names != GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER
        for row in legacy
    ):
        raise ValueError("legacy Q6 coordinate order differs")
    if any(
        row.coordinate_names != GENERATOR_INNOVATION_TANGENT_ORDER
        for row in generator
    ):
        raise ValueError("generator R4 coordinate order differs")
    for left, right in zip(legacy, generator, strict=True):
        if (
            left.family_id != right.family_id
            or left.supervised_tokens != right.supervised_tokens
            or left.compensation_target_sha256
            != right.compensation_target_sha256
            or left.target_second_moment != right.target_second_moment
        ):
            raise ValueError(
                f"legacy and generator targets differ for {left.example_id}"
            )
    counts = Counter(row.family_id for row in generator)
    if (
        len(counts) != _EXPECTED_FAMILIES
        or set(counts.values()) != {_EXPECTED_PROMPTS_PER_FAMILY}
    ):
        raise ValueError(
            "generator innovation requires eight families with two examples each"
        )
    return legacy, generator


def _plan_binding(
    *,
    plan: Mapping[str, object],
    plan_file_sha256: str,
) -> tuple[dict[str, object], tuple[tuple[float, float], ...]]:
    validate_gemma_iterative_generator_innovation_plan(plan)
    logical = _require_sha256(
        plan.get("plan_sha256"),
        label="generator innovation plan",
    )
    file_receipt = _require_sha256(
        plan_file_sha256,
        label="generator innovation plan file",
    )
    basis = _mapping(
        plan.get("frozen_generator_basis"),
        label="frozen generator basis",
    )
    basis_sha256 = _require_sha256(
        basis.get("basis_sha256"),
        label="frozen generator basis",
    )
    raw_rows = basis.get("basis_matrix_source_coordinates_by_generator")
    if not isinstance(raw_rows, (tuple, list)) or len(raw_rows) != 6:
        raise ValueError("frozen generator basis must contain six rows")
    basis_rows = tuple(
        _float_pair(row, label=f"fixed basis row {index}")
        for index, row in enumerate(raw_rows)
    )
    source_names = tuple(basis.get("source_coordinate_names", ()))
    if source_names != GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER:
        raise ValueError("plan and exact cumulative coordinate order differ")
    tangent_design = _mapping(
        plan.get("activation_tangent_design"),
        label="activation tangent design",
    )
    if (
        tuple(tangent_design.get("coordinate_order", ()))
        != GENERATOR_INNOVATION_TANGENT_ORDER
    ):
        raise ValueError("plan and exact generator tangent order differ")
    family_fishers = _mapping(
        basis.get("family_fisher_second_moments"),
        label="basis-source family Fishers",
    )
    source_families = tuple(sorted(
        _identifier(name, label="basis-source family ID")
        for name in family_fishers
    ))
    if len(source_families) != _EXPECTED_FAMILIES:
        raise ValueError("basis source must contain eight families")
    _validate_plan_fitter_compatibility(plan)
    return (
        {
            "schema": GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA,
            "plan_sha256": logical,
            "plan_file_sha256": file_receipt,
            "basis_sha256": basis_sha256,
            "basis_matrix_source_coordinates_by_generator": basis_rows,
            "source_coordinate_order": (
                GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER
            ),
            "source_family_ids": source_families,
        },
        basis_rows,
    )


def _validate_plan_fitter_compatibility(
    plan: Mapping[str, object],
) -> None:
    """Fail closed if fitter constants drift from the frozen plan."""

    screen = _mapping(
        plan.get("nested_family_screen"),
        label="plan nested family screen",
    )
    ridge = _mapping(
        screen.get("conditional_ridge"),
        label="plan conditional ridge",
    )
    finite_labels = tuple(
        format(
            _finite(value, label="plan finite ridge"),
            "g",
        )
        for value in ridge.get("finite_lambda_grid", ())
    )
    expected_labels = (
        *finite_labels,
        "inf",
    )
    if (
        _finite(
            ridge.get("shared_coordinate_ridge"),
            label="plan shared ridge",
        )
        != GENERATOR_INNOVATION_SHARED_RIDGE
        or expected_labels
        != GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS
        or ridge.get("shared_only_candidate")
        != (
            "exact_infinite_lambda_with_both_innovation_coefficients_zero"
        )
    ):
        raise ValueError("fitter ridge constants drifted from the frozen plan")

    gates = _mapping(screen.get("gates"), label="plan nested gates")
    fitter = GENERATOR_INNOVATION_GATE_CONFIG
    direct = {
        "all_outer_standardized_design_rank": (
            "required_outer_standardized_rank"
        ),
        "maximum_median_outer_standardized_condition_number": (
            "maximum_median_outer_standardized_condition"
        ),
        "minimum_mean_pairwise_outer_coefficient_cosine": (
            "minimum_mean_pairwise_standardized_coefficient_cosine"
        ),
        "minimum_conditional_residual_design_energy_fraction": (
            "minimum_conditional_residual_design_energy_fraction"
        ),
        "minimum_parent_family_macro_relative_rmse_improvement": (
            "minimum_family_macro_relative_rmse_improvement_vs_parent"
        ),
        "minimum_conditional_minus_fixed_u_shared_macro_improvement": (
            "minimum_family_macro_relative_rmse_improvement_vs_static_generator"
        ),
        "minimum_conditional_minus_legacy_shared_macro_improvement": (
            "minimum_family_macro_relative_rmse_improvement_vs_legacy_shared"
        ),
        "minimum_material_conditional_family_improvement": (
            "minimum_material_family_relative_rmse_improvement_vs_static_generator"
        ),
        "minimum_fixed_u_new_panel_fisher_trace_coverage_fraction": (
            "minimum_fixed_basis_fisher_trace_coverage"
        ),
    }
    for plan_key, fitter_key in direct.items():
        if gates.get(plan_key) != fitter.get(fitter_key):
            raise ValueError(
                f"fitter gate {fitter_key} drifted from plan gate {plan_key}"
            )
    if gates.get("minimum_worst_family_relative_rmse_improvement") != -float(
        fitter["maximum_worst_family_relative_rmse_regression_vs_parent"]
    ):
        raise ValueError("fitter worst-family gate drifted from the plan")
    count_fraction = {
        "minimum_parent_family_win_count": (
            "minimum_parent_family_win_fraction"
        ),
        "minimum_material_conditional_family_win_count": (
            "minimum_material_static_family_win_fraction"
        ),
        "minimum_outer_folds_with_nonzero_conditional_route": (
            "minimum_materially_nonzero_conditional_fold_fraction"
        ),
    }
    for plan_key, fitter_key in count_fraction.items():
        count = gates.get(plan_key)
        if (
            type(count) is not int
            or count / _EXPECTED_FAMILIES != fitter.get(fitter_key)
        ):
            raise ValueError(
                f"fitter fraction {fitter_key} drifted from plan count {plan_key}"
            )
    if set(gates) != set(direct) | {
        "minimum_worst_family_relative_rmse_improvement",
        *count_fraction,
    }:
        raise ValueError("plan nested gate set and fitter mapping differ")

    trust = _mapping(plan.get("trust_region"), label="plan trust region")
    if (
        trust.get("operator_norm_bound")
        != float(OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND)
        or trust.get("required_corner_count") != 16
        or tuple(trust.get("corner_axis_values", ())) != (-1.0, 1.0)
        or trust.get("coordinatewise_clipping_allowed") is not False
        or trust.get("projection_must_be_fit_without_held_family") is not True
    ):
        raise ValueError("fitter trust constants drifted from the frozen plan")


def _normalize_collection_lineage(
    value: Mapping[str, object],
    *,
    plan_binding: Mapping[str, object],
    family_by_example: Mapping[str, str],
) -> dict[str, object]:
    raw = _mapping(value, label="collection lineage")
    allowed = (
        _COLLECTION_LINEAGE_REQUIRED_FIELDS
        | {_COLLECTION_LINEAGE_OPTIONAL_FIELD}
    )
    if (
        not _COLLECTION_LINEAGE_REQUIRED_FIELDS.issubset(raw)
        or not set(raw).issubset(allowed)
    ):
        raise ValueError("collection lineage fields differ")
    result: dict[str, object] = {
        "role": _COLLECTION_ROLE,
        **{
            key: _require_sha256(
                raw.get(key),
                label=f"collection lineage {key}",
            )
            for key in sorted(_COLLECTION_LINEAGE_REQUIRED_FIELDS)
        },
    }
    optional = raw.get(_COLLECTION_LINEAGE_OPTIONAL_FIELD)
    result[_COLLECTION_LINEAGE_OPTIONAL_FIELD] = (
        None
        if optional is None
        else _require_sha256(
            optional,
            label="prompt-free panel artifact receipt",
        )
    )
    for key in ("plan_sha256", "plan_file_sha256", "basis_sha256"):
        if result[key] != plan_binding[key]:
            raise ValueError(f"collection {key} differs from the fixed plan")
    expected_membership = gemma_progressive_panel_membership_receipt_sha256(
        role=_COLLECTION_ROLE,
        manifest_sha256=str(result["collection_manifest_sha256"]),
        family_by_example=dict(family_by_example),
    )
    if result["collection_membership_receipt_sha256"] != expected_membership:
        raise ValueError(
            "collection membership does not bind the fitted example families"
        )
    return result


def _normalize_live_lineage(
    value: Mapping[str, object],
    *,
    planned_parent_lineage: Mapping[str, object],
    collection_lineage: Mapping[str, object],
) -> dict[str, str]:
    raw = _mapping(value, label="live lineage")
    if not raw:
        raise ValueError("live lineage must be nonempty")
    result: dict[str, str] = {}
    for key, receipt in sorted(raw.items()):
        name = _identifier(key, label="live lineage key")
        result[name] = _require_sha256(
            receipt,
            label=f"live lineage {name}",
        )
    for key, receipt in planned_parent_lineage.items():
        if result.get(key) != receipt:
            raise ValueError(
                f"live parent lineage differs from the plan for {key}"
            )
    for key in _COLLECTION_LINEAGE_REQUIRED_FIELDS:
        if result.get(key) != collection_lineage[key]:
            raise ValueError(
                f"live collection lineage differs for {key}"
            )
    optional = collection_lineage[
        _COLLECTION_LINEAGE_OPTIONAL_FIELD
    ]
    if optional is not None and result.get(
        _COLLECTION_LINEAGE_OPTIONAL_FIELD
    ) != optional:
        raise ValueError("live prompt-free panel receipt differs")
    return result


def _feature_summary(value: object, *, label: str) -> dict[str, object]:
    raw = _mapping(value, label=label)
    if set(raw) != _FEATURE_SUMMARY_FIELDS:
        raise ValueError(f"{label} fields differ")
    active = raw["active_activation_row_count"]
    if type(active) is not int or active <= 0:
        raise ValueError(f"{label} active row count must be positive")
    mean = _float_pair(raw["mean_by_channel"], label=f"{label} mean")
    second = _float_pair(
        raw["second_moment_by_channel"],
        label=f"{label} second moment",
    )
    mean_absolute = _float_pair(
        raw["mean_absolute_by_channel"],
        label=f"{label} mean absolute",
    )
    maximum = _float_pair(
        raw["maximum_absolute_by_channel"],
        label=f"{label} maximum absolute",
    )
    positive = _count_pair(
        raw["positive_count_by_channel"],
        label=f"{label} positive counts",
    )
    negative = _count_pair(
        raw["negative_count_by_channel"],
        label=f"{label} negative counts",
    )
    zero = _count_pair(
        raw["zero_count_by_channel"],
        label=f"{label} zero counts",
    )
    for channel in range(2):
        if (
            second[channel] < -1.0e-15
            or second[channel] + 1.0e-12
            < mean[channel] * mean[channel]
            or mean_absolute[channel] < -1.0e-15
            or mean_absolute[channel] + 1.0e-12 < abs(mean[channel])
            or maximum[channel] < 0.0
            or maximum[channel] > 1.0
            or mean_absolute[channel] > maximum[channel] + 1.0e-12
            or second[channel]
            > maximum[channel] * maximum[channel] + 1.0e-12
            or positive[channel] + negative[channel] + zero[channel] != active
        ):
            raise ValueError(f"{label} channel statistics are inconsistent")
    if (
        raw["whole_sequence_equals_two_chunks"] is not True
        or raw["prior_excludes_current_activation"] is not True
        or raw["padding_updates_state"] is not False
    ):
        raise ValueError(f"{label} causal feature audit failed")
    return {
        "active_activation_row_count": active,
        "mean_by_channel": mean,
        "second_moment_by_channel": second,
        "mean_absolute_by_channel": mean_absolute,
        "maximum_absolute_by_channel": maximum,
        "positive_count_by_channel": positive,
        "negative_count_by_channel": negative,
        "zero_count_by_channel": zero,
        "bounded_innovation_trace_sha256": _require_sha256(
            raw["bounded_innovation_trace_sha256"],
            label=f"{label} bounded feature trace",
        ),
        "whole_sequence_equals_two_chunks": True,
        "prior_excludes_current_activation": True,
        "padding_updates_state": False,
    }


def _top_mode_receipt(value: object, *, label: str) -> dict[str, object]:
    raw = _mapping(value, label=label)
    if set(raw) != _TOP_MODE_FIELDS:
        raise ValueError(f"{label} fields differ")
    indices_value = raw["top_mode_indices"]
    if (
        not isinstance(indices_value, (tuple, list))
        or len(indices_value) != 2
        or any(type(item) is not int or item < 0 for item in indices_value)
        or len(set(indices_value)) != 2
    ):
        raise ValueError(f"{label} indices differ")
    norms = _float_pair(raw["top_mode_norms"], label=f"{label} norms")
    if any(value <= 0.0 for value in norms):
        raise ValueError(f"{label} norms must be positive")
    return {
        "top_mode_indices": tuple(int(item) for item in indices_value),
        "top_mode_norms": norms,
    }


def _receipt_map(
    value: Mapping[str, object],
    *,
    example_ids: set[str],
    label: str,
) -> dict[str, str]:
    raw = _mapping(value, label=label)
    if set(raw) != example_ids:
        raise ValueError(f"{label} example identities differ")
    return {
        example_id: _require_sha256(
            raw[example_id],
            label=f"{label} for {example_id}",
        )
        for example_id in sorted(example_ids)
    }


def _feature_audit(
    *,
    records: Sequence[TokenLossFisherPromptRecord],
    feature_summary_by_example: Mapping[str, object],
    top_mode_receipt_by_example: Mapping[str, object],
    token_vjp_receipts: Mapping[str, str],
    source_tangent_receipts: Mapping[str, str],
) -> dict[str, object]:
    by_record = {row.example_id: row for row in records}
    examples = set(by_record)
    if (
        set(feature_summary_by_example) != examples
        or set(top_mode_receipt_by_example) != examples
    ):
        raise ValueError("causal feature audit example identities differ")
    rows: dict[str, object] = {}
    global_indices: set[tuple[int, int]] = set()
    global_norms: set[tuple[float, float]] = set()
    for example_id in sorted(examples):
        summary = _feature_summary(
            feature_summary_by_example[example_id],
            label=f"feature summary for {example_id}",
        )
        modes = _top_mode_receipt(
            top_mode_receipt_by_example[example_id],
            label=f"top-mode receipt for {example_id}",
        )
        indices = tuple(modes["top_mode_indices"])
        norms = tuple(modes["top_mode_norms"])
        global_indices.add(indices)  # type: ignore[arg-type]
        global_norms.add(norms)  # type: ignore[arg-type]
        payload = {
            "example_id": example_id,
            "family_id": by_record[example_id].family_id,
            "generator_prompt_record_sha256": (
                by_record[example_id].prompt_record_sha256
            ),
            "token_vjp_artifact_sha256": token_vjp_receipts[example_id],
            "source_tangent_record_sha256": (
                source_tangent_receipts[example_id]
            ),
            "feature_summary": summary,
            "top_mode_receipt": modes,
        }
        rows[example_id] = {
            **payload,
            "feature_receipt_sha256": _sha256(
                _FEATURE_RECEIPT_DOMAIN,
                payload,
            ),
        }
    if len(global_indices) != 1 or len(global_norms) != 1:
        raise ValueError("fixed parent top modes differ across examples")
    payload: dict[str, object] = {
        "top_mode_indices": next(iter(global_indices)),
        "top_mode_norms": next(iter(global_norms)),
        "by_example_id": rows,
        "causal_prior_before_update_for_every_example": True,
        "whole_sequence_chunk_equivalence_for_every_example": True,
        "padding_never_updates_state_for_every_example": True,
        "bounded_feature_rows_retained": False,
    }
    return {
        **payload,
        "feature_audit_sha256": _sha256(_FEATURE_AUDIT_DOMAIN, payload),
    }


def _decision(nested: Mapping[str, object]) -> dict[str, object]:
    passed = nested.get("passed")
    gates = nested.get("gate_results")
    if type(passed) is not bool or not isinstance(gates, (tuple, list)):
        raise ValueError("nested generator decision receipt differs")
    every_gate = bool(gates) and all(
        isinstance(row, (tuple, list))
        and len(row) == 2
        and type(row[1]) is bool
        and row[1]
        for row in gates
    )
    if passed is not every_gate:
        raise ValueError("nested pass flag differs from its gate conjunction")
    return {
        "nested_family_derivative_screen_passed": passed,
        "every_preregistered_nested_gate_passed": every_gate,
        "once_only_finite_displacement_authorized": every_gate,
        "finite_displacement_opened": False,
        "exact_finite_outputs_may_refit_or_select": False,
        "provider_compiled": False,
        "runtime_or_compression_claim_authorized": False,
        "next_step": (
            "open_once_only_out_of_fold_finite_displacement_without_refit"
            if every_gate
            else "stop_do_not_open_finite_displacement_or_compile_provider"
        ),
    }


def build_gemma_iterative_generator_innovation_development_report(
    *,
    legacy_records: Sequence[object],
    generator_records: Sequence[object],
    plan: Mapping[str, object],
    plan_file_sha256: str,
    feature_summary_by_example: Mapping[str, Mapping[str, object]],
    top_mode_receipt_by_example: Mapping[str, Mapping[str, object]],
    token_vjp_artifact_sha256_by_example: Mapping[str, str],
    source_tangent_record_sha256_by_example: Mapping[str, str],
    total_backward_call_count: int,
    vjp_chunk_size: int,
    lineage: Mapping[str, str],
    collection_lineage: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact 16-example nested generator-innovation screen."""

    legacy, generator = _records(legacy_records, generator_records)
    plan_receipt, fixed_basis = _plan_binding(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
    )
    new_families = {row.family_id for row in generator}
    if new_families & set(plan_receipt["source_family_ids"]):
        raise ValueError("generator collection reuses a basis-source family")
    family_by_example = {
        row.example_id: row.family_id for row in generator
    }
    collection = _normalize_collection_lineage(
        collection_lineage,
        plan_binding=plan_receipt,
        family_by_example=family_by_example,
    )
    plan_lineage = _mapping(plan.get("lineage"), label="plan lineage")
    planned_parent = dict(sorted(
        _mapping(
            plan_lineage.get("token_fisher_model_and_parent_lineage"),
            label="planned parent lineage",
        ).items()
    ))
    live = _normalize_live_lineage(
        lineage,
        planned_parent_lineage=planned_parent,
        collection_lineage=collection,
    )
    examples = set(family_by_example)
    vjp_receipts = _receipt_map(
        token_vjp_artifact_sha256_by_example,
        example_ids=examples,
        label="token VJP artifact receipts",
    )
    source_receipts = _receipt_map(
        source_tangent_record_sha256_by_example,
        example_ids=examples,
        label="source tangent record receipts",
    )
    if (
        type(vjp_chunk_size) is not int
        or vjp_chunk_size <= 0
        or type(total_backward_call_count) is not int
        or total_backward_call_count <= 0
    ):
        raise ValueError("generator innovation backward accounting is invalid")
    expected_backward_calls = sum(
        (row.supervised_tokens + vjp_chunk_size - 1) // vjp_chunk_size
        for row in generator
    )
    if total_backward_call_count != expected_backward_calls:
        raise ValueError("generator innovation backward call count differs")
    feature_audit = _feature_audit(
        records=generator,
        feature_summary_by_example=feature_summary_by_example,
        top_mode_receipt_by_example=top_mode_receipt_by_example,
        token_vjp_receipts=vjp_receipts,
        source_tangent_receipts=source_receipts,
    )
    nested = build_generator_innovation_nested_lofo_report(
        legacy,
        generator,
        fixed_basis=fixed_basis,
    )
    validate_generator_innovation_nested_lofo_report(nested)
    nested_basis = _mapping(
        nested.get("fixed_basis"),
        label="nested fixed basis",
    )
    if not _canonical_equal(nested_basis.get("rows"), fixed_basis):
        raise RuntimeError("nested fit did not use the exact frozen plan basis")
    supervised_tokens = sum(row.supervised_tokens for row in generator)
    payload: dict[str, object] = {
        "schema": (
            GEMMA_ITERATIVE_GENERATOR_INNOVATION_DEVELOPMENT_SCHEMA
        ),
        "lineage": {
            "plan": plan_receipt,
            "planned_parent_lineage": planned_parent,
            "live_lineage": live,
            "collection": collection,
        },
        "coordinate_orders": {
            "legacy_q6": GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
            "generator_r4": GENERATOR_INNOVATION_TANGENT_ORDER,
        },
        "legacy_prompt_fisher_records": tuple(
            row.to_dict() for row in legacy
        ),
        "generator_prompt_fisher_records": tuple(
            row.to_dict() for row in generator
        ),
        "feature_audit": feature_audit,
        "nested_family_screen": nested,
        "decision": _decision(nested),
        "resources": {
            "fit_example_count": _EXPECTED_EXAMPLES,
            "fit_family_count": _EXPECTED_FAMILIES,
            "examples_per_family": _EXPECTED_PROMPTS_PER_FAMILY,
            "supervised_token_count": supervised_tokens,
            "source_forward_count": _EXPECTED_EXAMPLES,
            "retained_parent_token_vjp_forward_count": _EXPECTED_EXAMPLES,
            "total_model_forward_count": 2 * _EXPECTED_EXAMPLES,
            "model_forward_count_per_example": 2,
            "token_vjp_backward_call_count": total_backward_call_count,
            "token_vjp_chunk_size": vjp_chunk_size,
            "candidate_forward_count": 0,
            "finite_displacement_forward_count": 0,
            "fresh_shadow_forward_count": 0,
        },
        "audit": {
            "execution_mode": (
                "development_exact_token_jacobian_fixed_u_nested_family_lofo"
            ),
            "development_only": True,
            "family_blocked_outer_and_inner_splits": True,
            "tokens_used_as_independent_split_units": False,
            "fixed_basis_refit_on_new_panel": False,
            "innovation_multiplied_before_token_gradient_contraction": True,
            "raw_prompt_text_retained": False,
            "raw_token_ids_retained": False,
            "raw_logits_retained": False,
            "raw_modal_rows_retained": False,
            "raw_feature_rows_retained": False,
            "raw_activation_rows_retained": False,
            "raw_gradient_rows_retained": False,
            "raw_token_score_rows_retained": False,
            "prompt_sufficient_statistics_retained": True,
            "finite_displacements_opened": False,
            "provider_compiled": False,
            "token_vjp_artifact_sha256_by_example": vjp_receipts,
            "source_tangent_record_sha256_by_example": source_receipts,
        },
    }
    return {
        **payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, payload),
    }


def _validate_lineage(
    value: object,
    *,
    records: Sequence[TokenLossFisherPromptRecord],
) -> None:
    lineage = _mapping(value, label="development lineage")
    if set(lineage) != {
        "plan",
        "planned_parent_lineage",
        "live_lineage",
        "collection",
    }:
        raise ValueError("generator development lineage fields differ")
    plan = _mapping(lineage["plan"], label="plan receipt")
    if set(plan) != {
        "schema",
        "plan_sha256",
        "plan_file_sha256",
        "basis_sha256",
        "basis_matrix_source_coordinates_by_generator",
        "source_coordinate_order",
        "source_family_ids",
    }:
        raise ValueError("generator development plan receipt fields differ")
    if (
        plan.get("schema")
        != GEMMA_ITERATIVE_GENERATOR_INNOVATION_PLAN_SCHEMA
        or tuple(plan.get("source_coordinate_order", ()))
        != GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER
    ):
        raise ValueError("generator development plan receipt differs")
    for key in ("plan_sha256", "plan_file_sha256", "basis_sha256"):
        _require_sha256(plan.get(key), label=f"plan receipt {key}")
    basis_rows = plan.get("basis_matrix_source_coordinates_by_generator")
    if not isinstance(basis_rows, (tuple, list)) or len(basis_rows) != 6:
        raise ValueError("plan receipt fixed basis must contain six rows")
    for index, row in enumerate(basis_rows):
        _float_pair(row, label=f"plan receipt fixed basis row {index}")
    source_families = tuple(plan.get("source_family_ids", ()))
    if (
        len(source_families) != _EXPECTED_FAMILIES
        or source_families != tuple(sorted(set(source_families)))
    ):
        raise ValueError("plan source family receipt differs")
    for family in source_families:
        _identifier(family, label="plan source family")
    if set(source_families) & {row.family_id for row in records}:
        raise ValueError("new collection overlaps a plan source family")
    planned_parent = _mapping(
        lineage["planned_parent_lineage"],
        label="planned parent lineage",
    )
    if not planned_parent:
        raise ValueError("planned parent lineage must be nonempty")
    for key, receipt in planned_parent.items():
        _identifier(key, label="planned parent lineage key")
        _require_sha256(receipt, label=f"planned parent lineage {key}")
    family_by_example = {
        row.example_id: row.family_id for row in records
    }
    collection_raw = _mapping(
        lineage["collection"],
        label="collection lineage",
    )
    if collection_raw.get("role") != _COLLECTION_ROLE:
        raise ValueError("collection role differs")
    collection = _normalize_collection_lineage(
        {
            key: value
            for key, value in collection_raw.items()
            if key != "role"
        },
        plan_binding=plan,
        family_by_example=family_by_example,
    )
    if not _canonical_equal(collection, collection_raw):
        raise ValueError("collection lineage is not canonical")
    live = _normalize_live_lineage(
        _mapping(lineage["live_lineage"], label="live lineage"),
        planned_parent_lineage=planned_parent,
        collection_lineage=collection,
    )
    if not _canonical_equal(live, lineage["live_lineage"]):
        raise ValueError("live lineage is not canonical")


def _validate_feature_audit(
    value: object,
    *,
    records: Sequence[TokenLossFisherPromptRecord],
    vjp_receipts: Mapping[str, str],
    source_receipts: Mapping[str, str],
) -> None:
    audit = _mapping(value, label="feature audit")
    if set(audit) != {
        "top_mode_indices",
        "top_mode_norms",
        "by_example_id",
        "causal_prior_before_update_for_every_example",
        "whole_sequence_chunk_equivalence_for_every_example",
        "padding_never_updates_state_for_every_example",
        "bounded_feature_rows_retained",
        "feature_audit_sha256",
    }:
        raise ValueError("feature audit fields differ")
    by_example = _mapping(
        audit["by_example_id"],
        label="feature audit examples",
    )
    examples = {row.example_id for row in records}
    if set(by_example) != examples:
        raise ValueError("feature audit example identities differ")
    summaries: dict[str, Mapping[str, object]] = {}
    modes: dict[str, Mapping[str, object]] = {}
    for example_id in examples:
        row = _mapping(
            by_example[example_id],
            label=f"feature audit for {example_id}",
        )
        expected_fields = {
            "example_id",
            "family_id",
            "generator_prompt_record_sha256",
            "token_vjp_artifact_sha256",
            "source_tangent_record_sha256",
            "feature_summary",
            "top_mode_receipt",
            "feature_receipt_sha256",
        }
        if set(row) != expected_fields:
            raise ValueError("per-example feature audit fields differ")
        summaries[example_id] = _mapping(
            row["feature_summary"],
            label=f"feature summary for {example_id}",
        )
        modes[example_id] = _mapping(
            row["top_mode_receipt"],
            label=f"top mode receipt for {example_id}",
        )
    rebuilt = _feature_audit(
        records=records,
        feature_summary_by_example=summaries,
        top_mode_receipt_by_example=modes,
        token_vjp_receipts=vjp_receipts,
        source_tangent_receipts=source_receipts,
    )
    if not _canonical_equal(rebuilt, audit):
        raise ValueError("feature audit receipt differs")


def validate_gemma_iterative_generator_innovation_development_report(
    report: object,
) -> None:
    """Validate sufficient statistics, nested fits, gates, and all receipts."""

    value = _mapping(report, label="generator innovation development report")
    expected_fields = {
        "schema",
        "lineage",
        "coordinate_orders",
        "legacy_prompt_fisher_records",
        "generator_prompt_fisher_records",
        "feature_audit",
        "nested_family_screen",
        "decision",
        "resources",
        "audit",
        "report_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("generator innovation development fields differ")
    if (
        value.get("schema")
        != GEMMA_ITERATIVE_GENERATOR_INNOVATION_DEVELOPMENT_SCHEMA
    ):
        raise ValueError("generator innovation development schema differs")
    if value.get("coordinate_orders") != {
        "legacy_q6": GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
        "generator_r4": GENERATOR_INNOVATION_TANGENT_ORDER,
    } and not _canonical_equal(
        value.get("coordinate_orders"),
        {
            "legacy_q6": GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
            "generator_r4": GENERATOR_INNOVATION_TANGENT_ORDER,
        },
    ):
        raise ValueError("generator innovation coordinate orders differ")
    legacy, generator = _records(
        value["legacy_prompt_fisher_records"],  # type: ignore[arg-type]
        value["generator_prompt_fisher_records"],  # type: ignore[arg-type]
    )
    _validate_lineage(value["lineage"], records=generator)
    nested = _mapping(
        value["nested_family_screen"],
        label="nested family screen",
    )
    if nested.get("schema") != GENERATOR_INNOVATION_SCHEMA:
        raise ValueError("nested generator innovation schema differs")
    validate_generator_innovation_nested_lofo_report(nested)
    plan_receipt = _mapping(
        _mapping(value["lineage"], label="development lineage")["plan"],
        label="plan receipt",
    )
    if not _canonical_equal(
        _mapping(nested["fixed_basis"], label="nested fixed basis")["rows"],
        plan_receipt["basis_matrix_source_coordinates_by_generator"],
    ):
        raise ValueError("nested screen did not use the exact plan basis")
    replay_generator_innovation_nested_lofo_report(
        legacy,
        generator,
        fixed_basis=_mapping(
            nested["fixed_basis"],
            label="nested fixed basis",
        )["rows"],  # type: ignore[arg-type]
        expected_report=nested,
    )
    audit = _mapping(value["audit"], label="development audit")
    expected_audit_fields = {
        "execution_mode",
        "development_only",
        "family_blocked_outer_and_inner_splits",
        "tokens_used_as_independent_split_units",
        "fixed_basis_refit_on_new_panel",
        "innovation_multiplied_before_token_gradient_contraction",
        "raw_prompt_text_retained",
        "raw_token_ids_retained",
        "raw_logits_retained",
        "raw_modal_rows_retained",
        "raw_feature_rows_retained",
        "raw_activation_rows_retained",
        "raw_gradient_rows_retained",
        "raw_token_score_rows_retained",
        "prompt_sufficient_statistics_retained",
        "finite_displacements_opened",
        "provider_compiled",
        "token_vjp_artifact_sha256_by_example",
        "source_tangent_record_sha256_by_example",
    }
    if (
        set(audit) != expected_audit_fields
        or audit.get("execution_mode")
        != (
            "development_exact_token_jacobian_fixed_u_"
            "nested_family_lofo"
        )
    ):
        raise ValueError("generator innovation audit fields differ")
    expected_flags = {
        "development_only": True,
        "family_blocked_outer_and_inner_splits": True,
        "tokens_used_as_independent_split_units": False,
        "fixed_basis_refit_on_new_panel": False,
        "innovation_multiplied_before_token_gradient_contraction": True,
        "raw_prompt_text_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_modal_rows_retained": False,
        "raw_feature_rows_retained": False,
        "raw_activation_rows_retained": False,
        "raw_gradient_rows_retained": False,
        "raw_token_score_rows_retained": False,
        "prompt_sufficient_statistics_retained": True,
        "finite_displacements_opened": False,
        "provider_compiled": False,
    }
    if any(audit.get(key) is not expected for key, expected in expected_flags.items()):
        raise ValueError("generator innovation safety audit differs")
    examples = {row.example_id for row in generator}
    vjp_receipts = _receipt_map(
        _mapping(
            audit["token_vjp_artifact_sha256_by_example"],
            label="token VJP artifact receipts",
        ),
        example_ids=examples,
        label="token VJP artifact receipts",
    )
    source_receipts = _receipt_map(
        _mapping(
            audit["source_tangent_record_sha256_by_example"],
            label="source tangent receipts",
        ),
        example_ids=examples,
        label="source tangent receipts",
    )
    _validate_feature_audit(
        value["feature_audit"],
        records=generator,
        vjp_receipts=vjp_receipts,
        source_receipts=source_receipts,
    )
    resources = _mapping(value["resources"], label="development resources")
    expected_resource_fields = {
        "fit_example_count",
        "fit_family_count",
        "examples_per_family",
        "supervised_token_count",
        "source_forward_count",
        "retained_parent_token_vjp_forward_count",
        "total_model_forward_count",
        "model_forward_count_per_example",
        "token_vjp_backward_call_count",
        "token_vjp_chunk_size",
        "candidate_forward_count",
        "finite_displacement_forward_count",
        "fresh_shadow_forward_count",
    }
    if set(resources) != expected_resource_fields:
        raise ValueError("generator innovation resource fields differ")
    chunk = resources.get("token_vjp_chunk_size")
    backwards = resources.get("token_vjp_backward_call_count")
    token_count = sum(row.supervised_tokens for row in generator)
    if (
        type(chunk) is not int
        or chunk <= 0
        or type(backwards) is not int
        or backwards
        != sum(
            (row.supervised_tokens + chunk - 1) // chunk
            for row in generator
        )
        or resources.get("fit_example_count") != _EXPECTED_EXAMPLES
        or resources.get("fit_family_count") != _EXPECTED_FAMILIES
        or resources.get("examples_per_family")
        != _EXPECTED_PROMPTS_PER_FAMILY
        or resources.get("supervised_token_count") != token_count
        or resources.get("source_forward_count") != _EXPECTED_EXAMPLES
        or resources.get("retained_parent_token_vjp_forward_count")
        != _EXPECTED_EXAMPLES
        or resources.get("total_model_forward_count") != 2 * _EXPECTED_EXAMPLES
        or resources.get("model_forward_count_per_example") != 2
        or resources.get("candidate_forward_count") != 0
        or resources.get("finite_displacement_forward_count") != 0
        or resources.get("fresh_shadow_forward_count") != 0
    ):
        raise ValueError("generator innovation resources differ")
    if not _canonical_equal(value["decision"], _decision(nested)):
        raise ValueError("generator innovation decision differs")
    payload = dict(value)
    receipt = payload.pop("report_sha256", None)
    if receipt != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("generator innovation development hash mismatch")


def replay_gemma_iterative_generator_innovation_development_report(
    *,
    report: Mapping[str, object],
    plan: Mapping[str, object],
    plan_file_sha256: str,
) -> dict[str, object]:
    """Rebuild the complete report from retained sufficient statistics."""

    validate_gemma_iterative_generator_innovation_development_report(report)
    plan_receipt, fixed_basis = _plan_binding(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
    )
    lineage = _mapping(report["lineage"], label="development lineage")
    if not _canonical_equal(lineage["plan"], plan_receipt):
        raise ValueError("development report is not bound to this exact plan")
    nested = _mapping(
        report["nested_family_screen"],
        label="nested family screen",
    )
    if not _canonical_equal(
        _mapping(nested["fixed_basis"], label="nested fixed basis")["rows"],
        fixed_basis,
    ):
        raise ValueError("development report basis differs from the exact plan")
    audit = _mapping(report["audit"], label="development audit")
    feature = _mapping(report["feature_audit"], label="feature audit")
    by_example = _mapping(
        feature["by_example_id"],
        label="feature audit examples",
    )
    summaries = {
        example_id: _mapping(row, label=f"feature row {example_id}")[
            "feature_summary"
        ]
        for example_id, row in by_example.items()
    }
    modes = {
        example_id: _mapping(row, label=f"feature row {example_id}")[
            "top_mode_receipt"
        ]
        for example_id, row in by_example.items()
    }
    collection = {
        key: item
        for key, item in _mapping(
            lineage["collection"],
            label="collection lineage",
        ).items()
        if key != "role"
    }
    resources = _mapping(report["resources"], label="development resources")
    rebuilt = build_gemma_iterative_generator_innovation_development_report(
        legacy_records=report["legacy_prompt_fisher_records"],  # type: ignore[arg-type]
        generator_records=report[  # type: ignore[arg-type]
            "generator_prompt_fisher_records"
        ],
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        feature_summary_by_example=summaries,  # type: ignore[arg-type]
        top_mode_receipt_by_example=modes,  # type: ignore[arg-type]
        token_vjp_artifact_sha256_by_example=_mapping(
            audit["token_vjp_artifact_sha256_by_example"],
            label="token VJP receipts",
        ),  # type: ignore[arg-type]
        source_tangent_record_sha256_by_example=_mapping(
            audit["source_tangent_record_sha256_by_example"],
            label="source tangent receipts",
        ),  # type: ignore[arg-type]
        total_backward_call_count=int(
            resources["token_vjp_backward_call_count"]
        ),
        vjp_chunk_size=int(resources["token_vjp_chunk_size"]),
        lineage=_mapping(
            lineage["live_lineage"],
            label="live lineage",
        ),  # type: ignore[arg-type]
        collection_lineage=collection,
    )
    if not _canonical_equal(rebuilt, report):
        raise ValueError("generator innovation development replay differs")
    return rebuilt


def publish_gemma_iterative_generator_innovation_development_report(
    destination: Path,
    report: Mapping[str, object],
) -> None:
    """Atomically publish one validated, non-overwriting local report."""

    validate_gemma_iterative_generator_innovation_development_report(report)
    if not isinstance(destination, Path):
        raise TypeError("generator innovation destination must be a Path")
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite generator innovation development report"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_path)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
            installed = True
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite generator innovation development report"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    if not installed:
        raise RuntimeError("generator innovation report was not installed")
