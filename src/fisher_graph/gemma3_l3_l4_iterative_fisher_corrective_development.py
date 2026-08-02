"""Adaptive partially pooled corrective screen over the frozen token Fisher map.

This rung reuses only authenticated prompt sufficient statistics from the
exact token-loss Fisher development report.  It performs no model forward,
does not open a selection panel, and cannot compile a provider.  The
cumulative occupancy arm is the fixed primary candidate; EW is a descriptive
sensitivity analysis and cannot rescue a failed primary decision.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re

from .gemma3_l3_l4_iterative_token_fisher_development import (
    TOKEN_FISHER_DEVELOPMENT_SCHEMA,
    validate_gemma_iterative_token_fisher_development_report,
)
from .gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES,
    TOKEN_OCCUPANCY_EW_COORDINATE_INDICES,
)
from .token_loss_fisher import (
    token_loss_fisher_prompt_record_from_dict,
)
from .token_loss_fisher_corrective import (
    build_token_loss_fisher_corrective_report,
    replay_token_loss_fisher_corrective_report,
    validate_token_loss_fisher_corrective_report,
)


__all__ = [
    "GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA",
    "build_gemma_iterative_fisher_corrective_development_report",
    "replay_gemma_iterative_fisher_corrective_development_report",
    "validate_gemma_iterative_fisher_corrective_development_report",
]


GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4."
    "iterative_fisher_corrective_development.v1"
)

_REPORT_DOMAIN = (
    b"fisher-graph:gemma-iterative-fisher-corrective-development:v1\0"
)
_TOPOLOGY_DOMAIN = (
    b"fisher-graph:gemma-iterative-fisher-corrective-topology:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = ("cumulative", "ew")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _graph_receipt(
    graph: object,
    *,
    arm_id: str,
) -> dict[str, object]:
    value = _mapping(graph, label=f"{arm_id} coupling graph")
    if (
        bool(value.get("fisher_coupling_is_symmetric")) is not True
        or bool(value.get("causal_direction_inferred")) is not False
        or int(value.get("stable_edge_count", -1)) != 6
    ):
        raise ValueError(
            f"{arm_id} frozen topology lacks six symmetric stable couplings"
        )
    coordinates = tuple(value.get("coordinate_names", ()))
    if len(coordinates) != 6 or len(set(coordinates)) != 6:
        raise ValueError(f"{arm_id} topology coordinates differ")
    edges = tuple(
        sorted(
            (
                str(edge["left_coordinate"]),
                str(edge["right_coordinate"]),
                float(edge["global_correlation"]),
                int(edge["stable_fold_count"]),
            )
            for edge in value.get("edges", ())
            if bool(edge.get("stable"))
        )
    )
    if (
        len(edges) != 6
        or len(
            {
                tuple(sorted((left, right)))
                for left, right, *_rest in edges
            }
        )
        != 6
        or any(
            left not in coordinates
            or right not in coordinates
            or left == right
            or not math.isfinite(correlation)
            for left, right, correlation, _count in edges
        )
        or any(count != 8 for *_rest, count in edges)
    ):
        raise ValueError(f"{arm_id} topology edge stability differs")
    canonical = json.loads(json.dumps(value, sort_keys=True))
    return {
        "arm_id": arm_id,
        "coordinate_names": coordinates,
        "stable_edges": edges,
        "stable_edge_count": 6,
        "fisher_coupling_is_symmetric": True,
        "causal_direction_inferred": False,
        "full_coupling_graph_sha256": _sha256(
            _TOPOLOGY_DOMAIN, canonical
        ),
    }


def _upstream_records(
    report: Mapping[str, object],
) -> tuple[object, ...]:
    rows = report.get("prompt_fisher_records")
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("token Fisher report omitted prompt records")
    return tuple(
        token_loss_fisher_prompt_record_from_dict(row) for row in rows
    )


def build_gemma_iterative_fisher_corrective_development_report(
    *,
    token_fisher_report: Mapping[str, object],
    token_fisher_report_file_sha256: str,
) -> dict[str, object]:
    """Build the cumulative-primary adaptive corrective development report."""

    validate_gemma_iterative_token_fisher_development_report(
        token_fisher_report
    )
    upstream_logical = _require_sha256(
        token_fisher_report.get("report_sha256"),
        label="token Fisher report",
    )
    upstream_file = _require_sha256(
        token_fisher_report_file_sha256,
        label="token Fisher report file",
    )
    records = _upstream_records(token_fisher_report)
    analysis = _mapping(
        token_fisher_report.get("analysis"),
        label="token Fisher analysis",
    )
    topology = {
        "cumulative": _graph_receipt(
            analysis.get("cumulative_coupling_graph"),
            arm_id="cumulative",
        ),
        "ew": _graph_receipt(
            analysis.get("ew_coupling_graph"),
            arm_id="ew",
        ),
    }
    primary = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=(
            TOKEN_OCCUPANCY_CUMULATIVE_COORDINATE_INDICES
        ),
    )
    sensitivity = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=TOKEN_OCCUPANCY_EW_COORDINATE_INDICES,
    )
    if (
        tuple(topology["cumulative"]["coordinate_names"])
        != tuple(primary["coordinate_names"])
        or tuple(topology["ew"]["coordinate_names"])
        != tuple(sensitivity["coordinate_names"])
    ):
        raise ValueError(
            "corrective frozen topology and fit coordinates differ"
        )
    primary_passed = bool(primary["passed"])
    payload = {
        "schema": GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA,
        "lineage": {
            "token_fisher_report_sha256": upstream_logical,
            "token_fisher_report_file_sha256": upstream_file,
            "token_fisher_schema": token_fisher_report.get("schema"),
            "token_fisher_prompt_record_sha256s": tuple(
                sorted(
                    str(row["prompt_record_sha256"])
                    for row in token_fisher_report[
                        "prompt_fisher_records"
                    ]
                )
            ),
            "token_fisher_prompt_record_sha256_by_example_id": {
                str(row["example_id"]): str(row["prompt_record_sha256"])
                for row in sorted(
                    token_fisher_report["prompt_fisher_records"],
                    key=lambda item: str(item["example_id"]),
                )
            },
        },
        "frozen_topology": topology,
        "primary_arm": "cumulative",
        "sensitivity_arm": "ew",
        "analysis": {
            "cumulative": primary,
            "ew": sensitivity,
        },
        "decision": {
            "primary_corrective_screen_passed": primary_passed,
            "sensitivity_corrective_screen_passed": bool(
                sensitivity["passed"]
            ),
            "sensitivity_can_rescue_primary": False,
            "provider_compiled": False,
            "runtime_claim_authorized": False,
            "causal_direction_authorized": False,
            "fresh_confirmation_authorized": False,
            "next_step": (
                "freeze_new_family_disjoint_corrective_recipe"
                if primary_passed
                else (
                    "collect_preregistered_new_causal_feature_on_"
                    "new_family_disjoint_data"
                )
            ),
        },
        "resources": {
            "source_model_forwards": 0,
            "parent_model_forwards": 0,
            "candidate_model_forwards": 0,
            "fresh_panel_forwards": 0,
            "backward_calls": 0,
            "reused_prompt_sufficient_statistics": len(records),
        },
        "audit": {
            "adaptive_development_only": True,
            "same_hypothesis_generation_panel_reused": True,
            "family_blocked_outer_lofo": True,
            "family_blocked_inner_lofo": True,
            "tokens_used_as_independent_split_units": False,
            "held_family_used_for_ridge_selection": False,
            "cumulative_arm_fixed_primary_before_screen": True,
            "ew_arm_sensitivity_only": True,
            "fisher_coupling_used_as_causal_direction": False,
            "selection_panel_referenced": False,
            "selection_panel_opened": False,
            "selection_claim_created": False,
            "provider_compiled": False,
        },
    }
    return {
        **payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, payload),
    }


def validate_gemma_iterative_fisher_corrective_development_report(
    report: object,
) -> None:
    """Validate one serialized adaptive corrective report."""

    value = _mapping(report, label="Gemma corrective development report")
    expected = {
        "schema",
        "lineage",
        "frozen_topology",
        "primary_arm",
        "sensitivity_arm",
        "analysis",
        "decision",
        "resources",
        "audit",
        "report_sha256",
    }
    if set(value) != expected:
        raise ValueError("Gemma corrective report fields differ")
    if value["schema"] != GEMMA_ITERATIVE_FISHER_CORRECTIVE_SCHEMA:
        raise ValueError("Gemma corrective report schema differs")
    if (
        value["primary_arm"] != "cumulative"
        or value["sensitivity_arm"] != "ew"
    ):
        raise ValueError("Gemma corrective arm roles differ")
    lineage = _mapping(value["lineage"], label="corrective lineage")
    if set(lineage) != {
        "token_fisher_report_sha256",
        "token_fisher_report_file_sha256",
        "token_fisher_schema",
        "token_fisher_prompt_record_sha256s",
        "token_fisher_prompt_record_sha256_by_example_id",
    }:
        raise ValueError("corrective lineage fields differ")
    _require_sha256(
        lineage.get("token_fisher_report_sha256"),
        label="corrective token Fisher lineage",
    )
    _require_sha256(
        lineage.get("token_fisher_report_file_sha256"),
        label="corrective token Fisher file lineage",
    )
    if lineage.get("token_fisher_schema") != TOKEN_FISHER_DEVELOPMENT_SCHEMA:
        raise ValueError("corrective token Fisher schema lineage differs")
    prompt_receipts = tuple(
        lineage["token_fisher_prompt_record_sha256s"]
    )
    if prompt_receipts != tuple(sorted(set(prompt_receipts))):
        raise ValueError("corrective prompt-record lineage is not canonical")
    for receipt in prompt_receipts:
        _require_sha256(receipt, label="corrective prompt-record lineage")
    prompt_receipt_map = _mapping(
        lineage["token_fisher_prompt_record_sha256_by_example_id"],
        label="corrective example-to-prompt lineage",
    )
    for example_id, receipt in prompt_receipt_map.items():
        if (
            not isinstance(example_id, str)
            or not example_id
            or example_id != example_id.strip()
        ):
            raise ValueError("corrective lineage example ID is invalid")
        _require_sha256(
            receipt,
            label=f"corrective prompt-record lineage for {example_id}",
        )
    if (
        tuple(prompt_receipt_map) != tuple(sorted(prompt_receipt_map))
        or len(prompt_receipt_map) != len(prompt_receipts)
        or tuple(sorted(prompt_receipt_map.values())) != prompt_receipts
    ):
        raise ValueError("corrective example-to-prompt lineage differs")
    topology = _mapping(
        value["frozen_topology"], label="corrective topology"
    )
    if set(topology) != set(_ARMS):
        raise ValueError("corrective topology arms differ")
    for arm_id in _ARMS:
        receipt = _mapping(
            topology[arm_id], label=f"{arm_id} topology receipt"
        )
        if set(receipt) != {
            "arm_id",
            "coordinate_names",
            "stable_edges",
            "stable_edge_count",
            "fisher_coupling_is_symmetric",
            "causal_direction_inferred",
            "full_coupling_graph_sha256",
        }:
            raise ValueError("corrective topology receipt fields differ")
        if (
            receipt.get("arm_id") != arm_id
            or receipt.get("stable_edge_count") != 6
            or receipt.get("fisher_coupling_is_symmetric") is not True
            or receipt.get("causal_direction_inferred") is not False
        ):
            raise ValueError("corrective frozen topology receipt differs")
        coordinates = tuple(receipt.get("coordinate_names", ()))
        edges = tuple(receipt.get("stable_edges", ()))
        normalized_edges = tuple(
            (
                str(edge[0]),
                str(edge[1]),
                float(edge[2]),
                edge[3],
            )
            for edge in edges
            if isinstance(edge, (tuple, list)) and len(edge) == 4
        )
        if (
            len(coordinates) != 6
            or len(set(coordinates)) != 6
            or len(edges) != 6
            or len(normalized_edges) != len(edges)
            or normalized_edges != tuple(sorted(normalized_edges))
            or any(
                not isinstance(edge, (tuple, list))
                or len(edge) != 4
                or edge[0] not in coordinates
                or edge[1] not in coordinates
                or edge[0] == edge[1]
                or not isinstance(edge[2], (int, float))
                or isinstance(edge[2], bool)
                or not math.isfinite(float(edge[2]))
                or type(edge[3]) is not int
                or edge[3] != 8
                for edge in edges
            )
            or len(
                {
                    tuple(sorted((str(edge[0]), str(edge[1]))))
                    for edge in edges
                }
            )
            != 6
        ):
            raise ValueError("corrective topology edge receipt differs")
        _require_sha256(
            receipt.get("full_coupling_graph_sha256"),
            label=f"{arm_id} topology graph",
        )
    analysis = _mapping(value["analysis"], label="corrective analysis")
    if set(analysis) != set(_ARMS):
        raise ValueError("corrective analysis arms differ")
    for arm_id in _ARMS:
        validate_token_loss_fisher_corrective_report(analysis[arm_id])
        if tuple(topology[arm_id]["coordinate_names"]) != tuple(
            analysis[arm_id]["coordinate_names"]
        ):
            raise ValueError(
                "corrective topology and analysis coordinates differ"
            )
    if (
        prompt_receipts
        != tuple(analysis["cumulative"]["prompt_record_sha256s"])
        or prompt_receipts
        != tuple(analysis["ew"]["prompt_record_sha256s"])
        or dict(prompt_receipt_map)
        != dict(
            analysis["cumulative"][
                "prompt_record_sha256_by_example_id"
            ]
        )
        or dict(prompt_receipt_map)
        != dict(
            analysis["ew"]["prompt_record_sha256_by_example_id"]
        )
    ):
        raise ValueError("corrective prompt-record lineage differs")
    decision = _mapping(value["decision"], label="corrective decision")
    if set(decision) != {
        "primary_corrective_screen_passed",
        "sensitivity_corrective_screen_passed",
        "sensitivity_can_rescue_primary",
        "provider_compiled",
        "runtime_claim_authorized",
        "causal_direction_authorized",
        "fresh_confirmation_authorized",
        "next_step",
    }:
        raise ValueError("corrective decision fields differ")
    expected_next_step = (
        "freeze_new_family_disjoint_corrective_recipe"
        if bool(analysis["cumulative"]["passed"])
        else (
            "collect_preregistered_new_causal_feature_on_"
            "new_family_disjoint_data"
        )
    )
    if (
        bool(decision.get("primary_corrective_screen_passed"))
        != bool(analysis["cumulative"]["passed"])
        or bool(decision.get("sensitivity_corrective_screen_passed"))
        != bool(analysis["ew"]["passed"])
        or decision.get("sensitivity_can_rescue_primary") is not False
        or decision.get("provider_compiled") is not False
        or decision.get("runtime_claim_authorized") is not False
        or decision.get("causal_direction_authorized") is not False
        or decision.get("fresh_confirmation_authorized") is not False
        or decision.get("next_step") != expected_next_step
    ):
        raise ValueError("corrective development decision differs")
    resources = _mapping(value["resources"], label="corrective resources")
    if resources != {
        "source_model_forwards": 0,
        "parent_model_forwards": 0,
        "candidate_model_forwards": 0,
        "fresh_panel_forwards": 0,
        "backward_calls": 0,
        "reused_prompt_sufficient_statistics": len(prompt_receipts),
    }:
        raise ValueError("corrective resource receipt differs")
    audit = _mapping(value["audit"], label="corrective audit")
    if set(audit) != {
        "adaptive_development_only",
        "same_hypothesis_generation_panel_reused",
        "family_blocked_outer_lofo",
        "family_blocked_inner_lofo",
        "tokens_used_as_independent_split_units",
        "held_family_used_for_ridge_selection",
        "cumulative_arm_fixed_primary_before_screen",
        "ew_arm_sensitivity_only",
        "fisher_coupling_used_as_causal_direction",
        "selection_panel_referenced",
        "selection_panel_opened",
        "selection_claim_created",
        "provider_compiled",
    }:
        raise ValueError("corrective audit fields differ")
    for key in (
        "adaptive_development_only",
        "same_hypothesis_generation_panel_reused",
        "family_blocked_outer_lofo",
        "family_blocked_inner_lofo",
        "cumulative_arm_fixed_primary_before_screen",
        "ew_arm_sensitivity_only",
    ):
        if audit.get(key) is not True:
            raise ValueError(f"corrective audit {key} must be true")
    for key in (
        "tokens_used_as_independent_split_units",
        "held_family_used_for_ridge_selection",
        "fisher_coupling_used_as_causal_direction",
        "selection_panel_referenced",
        "selection_panel_opened",
        "selection_claim_created",
        "provider_compiled",
    ):
        if audit.get(key) is not False:
            raise ValueError(f"corrective audit {key} must be false")
    payload = dict(value)
    receipt = payload.pop("report_sha256")
    if receipt != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("Gemma corrective report hash mismatch")


def replay_gemma_iterative_fisher_corrective_development_report(
    *,
    token_fisher_report: Mapping[str, object],
    token_fisher_report_file_sha256: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild every nested fit from the authenticated upstream moments."""

    validate_gemma_iterative_fisher_corrective_development_report(report)
    records = _upstream_records(token_fisher_report)
    analysis = _mapping(report["analysis"], label="corrective analysis")
    replay_token_loss_fisher_corrective_report(
        records, analysis["cumulative"]
    )
    replay_token_loss_fisher_corrective_report(records, analysis["ew"])
    rebuilt = (
        build_gemma_iterative_fisher_corrective_development_report(
            token_fisher_report=token_fisher_report,
            token_fisher_report_file_sha256=(
                token_fisher_report_file_sha256
            ),
        )
    )
    if _canonical_bytes(rebuilt) != _canonical_bytes(report):
        raise ValueError(
            "Gemma corrective report does not replay from upstream moments"
        )
    return rebuilt
