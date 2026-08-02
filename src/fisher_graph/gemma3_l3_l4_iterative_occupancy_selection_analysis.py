"""Scalar-only replay for the Iteration-5 occupancy-route selection.

The cumulative and exponentially weighted occupancy routes are fit and chosen
using already-open development data.  A fresh one-shot panel then evaluates
the frozen parent and both frozen routes.  Only the arm selected before that
panel opened is eligible to advance.

This module never loads model weights or tensors.  It rebuilds fidelity
metrics, paired comparisons, qualification gates, collection identity, and
the final report identity from scalar observations and hash receipts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math

from .gemma3_l3_l4_h4_damping_selection_runtime import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    GemmaH4DampingFiniteNLLObservation,
    _fidelity_from_observations,
    _paired_comparison,
)
from .gemma3_l3_l4_iterative_state_router_analysis import (
    _assert_scalar_hash_only,
    _canonical_observations,
    _correct_prompt_disagreement_quantile_labels,
    _mapping,
    _observation_dict,
    _observation_from_dict,
    _source_grid,
    _validate_manifest,
)


__all__ = [
    "CUMULATIVE_OCCUPANCY_ARM",
    "EW_OCCUPANCY_ARM",
    "OCCUPANCY_PARENT_ARM",
    "build_gemma_iterative_occupancy_selection_report",
    "validate_gemma_iterative_occupancy_selection_report",
]


OCCUPANCY_PARENT_ARM = "accepted_x4_plus_lag_b_parent"
CUMULATIVE_OCCUPANCY_ARM = "centered_cumulative_occupancy_route"
EW_OCCUPANCY_ARM = "centered_ew_occupancy_route"
_OCCUPANCY_ARMS = (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
)
_ALL_ARMS = (OCCUPANCY_PARENT_ARM, *_OCCUPANCY_ARMS)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_analysis"
)
_FORMAT_VERSION = 1
_COLLECTION_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-selection-collection:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma-iterative-occupancy-selection-report:v1\0"
)
_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PER_FAMILY = 2
_MACRO_IMPROVEMENT_MIN = 0.02
_MINIMUM_FAMILY_WIN_COUNT = 6
_WORST_FAMILY_IMPROVEMENT_MIN = -0.02
_SECONDARY_IMPROVEMENT_MIN = -0.02


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


def _strict_mapping(value: object, *, label: str) -> dict[str, object]:
    result = dict(_mapping(value, label=label))
    _assert_scalar_hash_only(result, path=label)
    return result


def _selected_arm(development: Mapping[str, object]) -> str:
    selected = development.get("selected_arm_id")
    if selected not in _OCCUPANCY_ARMS:
        raise ValueError(
            "development selection must freeze exactly one occupancy arm"
        )
    if development.get("selection_opened") is not False:
        raise ValueError(
            "development arm selection must precede fresh-panel opening"
        )
    if development.get("selection_rule_frozen") is not True:
        raise ValueError("development arm-selection rule was not frozen")
    gates = _mapping(
        development.get("scientific_gates_by_arm"),
        label="occupancy development scientific gates",
    )
    if set(gates) != set(_OCCUPANCY_ARMS):
        raise ValueError("development scientific arms differ")
    for arm_id in _OCCUPANCY_ARMS:
        arm_gates = _mapping(
            gates[arm_id],
            label=f"{arm_id} development scientific gates",
        )
        if type(arm_gates.get("passed")) is not bool:
            raise ValueError("development scientific decision is not boolean")
    if (
        _mapping(
            gates[selected],
            label="selected-arm development scientific gates",
        ).get("passed")
        is not True
    ):
        raise ValueError("development selected an unsupported occupancy arm")
    return str(selected)


def _canonical_arm_observations(
    value: Sequence[GemmaH4DampingFiniteNLLObservation | Mapping[str, object]],
    *,
    label: str,
) -> tuple[GemmaH4DampingFiniteNLLObservation, ...]:
    materialized = tuple(
        row
        if isinstance(row, GemmaH4DampingFiniteNLLObservation)
        else _observation_from_dict(
            row,
            label=f"{label}[{index}]",
        )
        for index, row in enumerate(value)
    )
    result = _canonical_observations(materialized, label=label)
    counts: dict[str, int] = {}
    for row in result:
        counts[row.family_id] = counts.get(row.family_id, 0) + 1
    if (
        len(result) != _EXPECTED_EXAMPLES
        or len(counts) != _EXPECTED_FAMILIES
        or set(counts.values()) != {_EXPECTED_PER_FAMILY}
    ):
        raise ValueError(f"{label} must be a strict 16-by-8 panel")
    return result


def _label_comparison(
    value: Mapping[str, object],
    *,
    baseline_arm_id: str,
    challenger_arm_id: str,
) -> dict[str, object]:
    """Replace damping-specific labels without changing replayed numbers."""

    result = dict(
        _correct_prompt_disagreement_quantile_labels(value)
    )
    result["baseline_arm_id"] = baseline_arm_id
    result["challenger_arm_id"] = challenger_arm_id
    primary_name = (
        "family_macro_mean_prompt_absolute_delta_nll_per_token"
    )
    primary = dict(
        _mapping(result[primary_name], label="occupancy paired primary")
    )
    primary["baseline"] = primary.pop("matched_alpha0")
    primary["challenger"] = primary.pop("challenger")
    result[primary_name] = primary

    family_rows: list[dict[str, object]] = []
    for raw in result["family_rows"]:
        row = dict(_mapping(raw, label="occupancy paired family row"))
        row["baseline_mean_prompt_absolute_delta_nll_per_token"] = row.pop(
            "matched_alpha0_mean_prompt_absolute_delta_nll_per_token"
        )
        row["challenger_mean_prompt_absolute_delta_nll_per_token"] = row.pop(
            "challenger_mean_prompt_absolute_delta_nll_per_token"
        )
        family_rows.append(row)
    result["family_rows"] = family_rows

    secondary: list[dict[str, object]] = []
    for raw in result["secondary_metrics"]:
        row = dict(_mapping(raw, label="occupancy paired secondary"))
        row["baseline"] = row.pop("matched_alpha0")
        row["challenger"] = row.pop("challenger")
        secondary.append(row)
    result["secondary_metrics"] = secondary
    return result


def _paired(
    baseline_fidelity: Mapping[str, object],
    challenger_fidelity: Mapping[str, object],
    *,
    baseline_observations: list[Mapping[str, object]],
    challenger_observations: list[Mapping[str, object]],
    baseline_arm_id: str,
    challenger_arm_id: str,
) -> dict[str, object]:
    return _label_comparison(
        _paired_comparison(
            baseline_fidelity,
            challenger_fidelity,
            baseline_observations=baseline_observations,
            challenger_observations=challenger_observations,
        ),
        baseline_arm_id=baseline_arm_id,
        challenger_arm_id=challenger_arm_id,
    )


def _validate_execution_audit(value: object) -> dict[str, object]:
    audit = _strict_mapping(value, label="occupancy selection audit")
    expected = {
        "development_example_count": 16,
        "selection_example_count": 16,
        "development_source_forward_count": 16,
        "development_parent_vjp_forward_count": 16,
        "selection_source_forward_count": 16,
        "selection_parent_forward_count": 16,
        "selection_cumulative_forward_count": 16,
        "selection_ew_forward_count": 16,
        "selection_vjp_forward_count": 0,
        "total_model_forward_count": 96,
        "model_forward_count_per_development_example": 2,
        "model_forward_count_per_selection_example": 4,
        "development_fit_records_shared_across_arms": True,
        "selection_source_reused_within_prompt": True,
        "selection_input_open_count": 1,
        "candidate_changes_after_selection_open": False,
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "raw_activations_retained": False,
        "gradient_tensors_retained": False,
        "model_weights_retained": False,
    }
    for key, expected_value in expected.items():
        if audit.get(key) != expected_value:
            raise ValueError(f"occupancy selection audit field {key} differs")
    return audit


def _comparison_gates_pass(value: Mapping[str, object]) -> bool:
    gates = _mapping(
        value.get("gates"),
        label="occupancy paired gates",
    )
    return gates.get("passed") is True


def _absolute_gates_pass(value: Mapping[str, object]) -> bool:
    gates = _mapping(
        value.get("gates"),
        label="occupancy absolute fidelity gates",
    )
    return gates.get("passed") is True


def build_gemma_iterative_occupancy_selection_report(
    *,
    development: Mapping[str, object],
    parent_observations: Sequence[
        GemmaH4DampingFiniteNLLObservation | Mapping[str, object]
    ],
    cumulative_observations: Sequence[
        GemmaH4DampingFiniteNLLObservation | Mapping[str, object]
    ],
    ew_observations: Sequence[
        GemmaH4DampingFiniteNLLObservation | Mapping[str, object]
    ],
    manifest: Mapping[str, str],
    lineage: Mapping[str, object],
    resources: Mapping[str, object],
    audit: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact scalar report for the frozen two-arm comparison."""

    canonical_development = _strict_mapping(
        development,
        label="occupancy development receipt",
    )
    selected = _selected_arm(canonical_development)
    canonical_audit = _validate_execution_audit(audit)
    canonical_lineage = _strict_mapping(
        lineage,
        label="occupancy selection lineage",
    )
    canonical_resources = _strict_mapping(
        resources,
        label="occupancy route resources",
    )
    if set(canonical_resources) != set(_OCCUPANCY_ARMS):
        raise ValueError("occupancy resource arms differ")

    canonical = {
        OCCUPANCY_PARENT_ARM: _canonical_arm_observations(
            parent_observations,
            label="occupancy parent observations",
        ),
        CUMULATIVE_OCCUPANCY_ARM: _canonical_arm_observations(
            cumulative_observations,
            label="cumulative occupancy observations",
        ),
        EW_OCCUPANCY_ARM: _canonical_arm_observations(
            ew_observations,
            label="EW occupancy observations",
        ),
    }
    grids = {
        arm_id: _source_grid(rows)
        for arm_id, rows in canonical.items()
    }
    if any(
        grid != grids[OCCUPANCY_PARENT_ARM]
        for grid in grids.values()
    ):
        raise ValueError("occupancy selection source grids differ")
    canonical_manifest = _validate_manifest(
        manifest,
        observations=canonical[OCCUPANCY_PARENT_ARM],
    )

    observation_payloads = {
        arm_id: [_observation_dict(row) for row in rows]
        for arm_id, rows in canonical.items()
    }
    fidelity = {
        arm_id: _fidelity_from_observations(payloads)
        for arm_id, payloads in observation_payloads.items()
    }
    comparisons = {
        "parent_to_cumulative": _paired(
            fidelity[OCCUPANCY_PARENT_ARM],
            fidelity[CUMULATIVE_OCCUPANCY_ARM],
            baseline_observations=observation_payloads[
                OCCUPANCY_PARENT_ARM
            ],
            challenger_observations=observation_payloads[
                CUMULATIVE_OCCUPANCY_ARM
            ],
            baseline_arm_id=OCCUPANCY_PARENT_ARM,
            challenger_arm_id=CUMULATIVE_OCCUPANCY_ARM,
        ),
        "parent_to_ew": _paired(
            fidelity[OCCUPANCY_PARENT_ARM],
            fidelity[EW_OCCUPANCY_ARM],
            baseline_observations=observation_payloads[
                OCCUPANCY_PARENT_ARM
            ],
            challenger_observations=observation_payloads[
                EW_OCCUPANCY_ARM
            ],
            baseline_arm_id=OCCUPANCY_PARENT_ARM,
            challenger_arm_id=EW_OCCUPANCY_ARM,
        ),
        "cumulative_to_ew": _paired(
            fidelity[CUMULATIVE_OCCUPANCY_ARM],
            fidelity[EW_OCCUPANCY_ARM],
            baseline_observations=observation_payloads[
                CUMULATIVE_OCCUPANCY_ARM
            ],
            challenger_observations=observation_payloads[
                EW_OCCUPANCY_ARM
            ],
            baseline_arm_id=CUMULATIVE_OCCUPANCY_ARM,
            challenger_arm_id=EW_OCCUPANCY_ARM,
        ),
    }
    selected_comparison_key = (
        "parent_to_cumulative"
        if selected == CUMULATIVE_OCCUPANCY_ARM
        else "parent_to_ew"
    )
    selected_comparison = comparisons[selected_comparison_key]
    parent_relative_passed = _comparison_gates_pass(selected_comparison)
    absolute_passed = _absolute_gates_pass(fidelity[selected])
    qualified_for_guard = parent_relative_passed and absolute_passed

    collection_payload = {
        "manifest": canonical_manifest,
        "lineage": canonical_lineage,
        "resources": canonical_resources,
        "audit": canonical_audit,
        "development": canonical_development,
        "observations_by_arm": observation_payloads,
    }
    collection_sha256 = _sha256(
        _COLLECTION_DOMAIN,
        collection_payload,
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "semantics": {
            "iteration": 5,
            "parent_arm_id": OCCUPANCY_PARENT_ARM,
            "occupancy_arm_ids": _OCCUPANCY_ARMS,
            "selected_before_fresh_panel_open": True,
            "nonselected_arm_metrics_only": True,
            "route": (
                "delta=(balance*selected_top2)@"
                "C(C0+balance*Cg+centered_occupancy*Co)"
            ),
            "occupancy_sign": (
                "+1_if_current_cumulative_balance_is_negative_else_-1"
            ),
            "occupancy_updates_current_active_token_before_route": True,
        },
        "thresholds": {
            "absolute": ESTABLISHED_SHADOW_FIDELITY_GATES.metadata(),
            "paired": {
                "family_macro_improvement_min": _MACRO_IMPROVEMENT_MIN,
                "minimum_strict_family_win_count": (
                    _MINIMUM_FAMILY_WIN_COUNT
                ),
                "worst_family_improvement_min": (
                    _WORST_FAMILY_IMPROVEMENT_MIN
                ),
                "secondary_improvement_min": (
                    _SECONDARY_IMPROVEMENT_MIN
                ),
            },
        },
        "collection": collection_payload,
        "collection_sha256": collection_sha256,
        "selection": {
            "fidelity_by_arm": fidelity,
            "paired_comparisons": comparisons,
        },
        "decision": {
            "development_selected_arm_id": selected,
            "only_development_selected_arm_eligible": True,
            "selected_parent_relative_gates_passed": (
                parent_relative_passed
            ),
            "selected_absolute_fidelity_gates_passed": absolute_passed,
            "qualified_for_guard": qualified_for_guard,
            "deployment_authorized": False,
            "retained_full_fit": None,
        },
        "safety": {
            "development_only": True,
            "fresh_selection_panel_opened_once": True,
            "source_outputs_authoritative": True,
            "candidate_outputs_metrics_only": True,
            "raw_prompts_in_report": False,
            "raw_token_ids_in_report": False,
            "raw_logits_in_report": False,
            "raw_activations_in_report": False,
            "gradient_tensors_in_report": False,
            "model_weights_in_report": False,
            "generalization_claim": False,
            "compression_qualification_claim": False,
            "deployment_claim": False,
        },
    }
    _assert_scalar_hash_only(report, path="occupancy selection report")
    report["report_sha256"] = _sha256(_REPORT_DOMAIN, report)
    return report


def validate_gemma_iterative_occupancy_selection_report(
    report: Mapping[str, object],
) -> None:
    """Rebuild every scalar-derived field and both report identities."""

    root = _strict_mapping(
        report,
        label="occupancy selection report",
    )
    expected_keys = {
        "schema",
        "format_version",
        "semantics",
        "thresholds",
        "collection",
        "collection_sha256",
        "selection",
        "decision",
        "safety",
        "report_sha256",
    }
    if set(root) != expected_keys:
        raise ValueError("occupancy selection report fields differ")
    if root["schema"] != _SCHEMA or root["format_version"] != _FORMAT_VERSION:
        raise ValueError("occupancy selection report schema differs")
    observed_report_sha256 = root["report_sha256"]
    if not isinstance(observed_report_sha256, str):
        raise ValueError("occupancy selection report hash is invalid")
    payload = dict(root)
    payload.pop("report_sha256")
    if _sha256(_REPORT_DOMAIN, payload) != observed_report_sha256:
        raise ValueError("occupancy selection report hash differs")

    collection = _strict_mapping(
        root["collection"],
        label="occupancy selection collection",
    )
    if _sha256(_COLLECTION_DOMAIN, collection) != root["collection_sha256"]:
        raise ValueError("occupancy selection collection hash differs")
    observations = _mapping(
        collection.get("observations_by_arm"),
        label="occupancy selection observations",
    )
    if set(observations) != set(_ALL_ARMS):
        raise ValueError("occupancy selection observation arms differ")
    rebuilt = build_gemma_iterative_occupancy_selection_report(
        development=_mapping(
            collection.get("development"),
            label="occupancy development receipt",
        ),
        parent_observations=observations[OCCUPANCY_PARENT_ARM],  # type: ignore[arg-type]
        cumulative_observations=observations[CUMULATIVE_OCCUPANCY_ARM],  # type: ignore[arg-type]
        ew_observations=observations[EW_OCCUPANCY_ARM],  # type: ignore[arg-type]
        manifest=_mapping(
            collection.get("manifest"),
            label="occupancy manifest",
        ),  # type: ignore[arg-type]
        lineage=_mapping(
            collection.get("lineage"),
            label="occupancy lineage",
        ),
        resources=_mapping(
            collection.get("resources"),
            label="occupancy resources",
        ),
        audit=_mapping(
            collection.get("audit"),
            label="occupancy audit",
        ),
    )
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(root):
        raise ValueError("occupancy selection report does not replay exactly")
