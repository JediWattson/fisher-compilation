"""Development report for exact Gemma loss-token Fisher edge fitting.

This rung uses exact supervised-token NLL Jacobians to estimate Fisher
coupling between the eight declared occupancy-route directions.  Token rows
determine the within-prompt geometry; prompts and families remain the
statistical units for fitting and held-out validation.

The Fisher coupling graph is intentionally undirected.  Its vertices are
already-declared causal route tangents, but a symmetric Fisher off-diagonal
cannot infer a new causal arrow.  A later JVP/intervention rung may orient
only the stable couplings nominated here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import Tensor

from .gemma3_l3_l4_iterative_token_fisher_edges import (
    TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
    GemmaIterativeTokenOccupancyTangentRecord,
    parse_gemma_iterative_token_occupancy_tangent_record,
)
from .token_loss_fisher import (
    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
    CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    TokenLossFisherPromptRecord,
    _tensor_sha256 as _token_fisher_tensor_sha256,
    analyze_cumulative_occupancy_token_loss_fisher_lofo,
    analyze_ew_occupancy_token_loss_fisher_lofo,
    token_loss_fisher_prompt_record_from_dict,
)


__all__ = [
    "TOKEN_FISHER_EDGE_MINIMUM_ABSOLUTE_CORRELATION",
    "TOKEN_FISHER_EDGE_MINIMUM_STABLE_FOLDS",
    "TOKEN_FISHER_DEVELOPMENT_SCHEMA",
    "build_gemma_iterative_token_fisher_development_report",
    "publish_gemma_iterative_token_fisher_development_report",
    "validate_gemma_iterative_token_fisher_development_report",
]


TOKEN_FISHER_DEVELOPMENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.iterative_token_fisher_development.v1"
)
TOKEN_FISHER_EDGE_MINIMUM_ABSOLUTE_CORRELATION = 0.25
TOKEN_FISHER_EDGE_MINIMUM_STABLE_FOLDS = 7

_EXPECTED_EXAMPLES = 16
_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS_PER_FAMILY = 2
_REPORT_DOMAIN = (
    b"fisher-graph:gemma-iterative-token-fisher-development:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(
        _REPORT_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _canonical_equal(left: object, right: object) -> bool:
    """Compare report values across tuple/list JSON round trips."""

    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _records(
    token_tangent_records: Sequence[object],
    prompt_records: Sequence[object],
) -> tuple[
    tuple[GemmaIterativeTokenOccupancyTangentRecord, ...],
    tuple[TokenLossFisherPromptRecord, ...],
]:
    tangents = tuple(
        sorted(
            (
                parse_gemma_iterative_token_occupancy_tangent_record(value)
                for value in token_tangent_records
            ),
            key=lambda row: row.example_id,
        )
    )
    prompts = tuple(
        sorted(
            (
                value
                if isinstance(value, TokenLossFisherPromptRecord)
                else token_loss_fisher_prompt_record_from_dict(value)
                for value in prompt_records
            ),
            key=lambda row: row.example_id,
        )
    )
    for record in prompts:
        record.validate_integrity()
    if (
        len(tangents) != _EXPECTED_EXAMPLES
        or len(prompts) != _EXPECTED_EXAMPLES
        or len({row.example_id for row in tangents}) != _EXPECTED_EXAMPLES
        or len({row.example_id for row in prompts}) != _EXPECTED_EXAMPLES
        or tuple(row.example_id for row in tangents)
        != tuple(row.example_id for row in prompts)
    ):
        raise ValueError("token Fisher development record geometry differs")
    tangent_families = {
        row.example_id: row.family_id for row in tangents
    }
    prompt_families = {
        row.example_id: row.family_id for row in prompts
    }
    counts = Counter(tangent_families.values())
    if (
        tangent_families != prompt_families
        or len(counts) != _EXPECTED_FAMILIES
        or set(counts.values()) != {_EXPECTED_PROMPTS_PER_FAMILY}
    ):
        raise ValueError("token Fisher development family geometry differs")
    for tangent, prompt in zip(tangents, prompts, strict=True):
        if (
            tangent.coordinate_order
            != TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER
            or prompt.coordinate_names
            != COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES
            or tangent.supervised_token_count
            != prompt.supervised_tokens
        ):
            raise ValueError("token Fisher coordinate/count binding differs")
        scores = torch.tensor(
            tuple(
                row.tangent_by_combined_occupancy_coordinate
                for row in tangent.rows
            ),
            dtype=torch.float64,
        )
        fisher = (scores.T @ scores) / prompt.supervised_tokens
        fisher = (fisher + fisher.T) * 0.5
        mean = scores.mean(dim=0)
        if (
            _token_fisher_tensor_sha256(scores)
            != prompt.token_score_matrix_sha256
        ):
            raise ValueError("token Fisher score-matrix hash differs")
        torch.testing.assert_close(
            fisher,
            torch.tensor(
                prompt.fisher_second_moment,
                dtype=torch.float64,
            ),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            mean,
            torch.tensor(prompt.mean_score, dtype=torch.float64),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    return tangents, prompts


def _family_fisher(
    records: Sequence[TokenLossFisherPromptRecord],
    indices: tuple[int, ...],
) -> dict[str, Tensor]:
    selected = torch.tensor(indices, dtype=torch.int64)
    grouped: dict[str, list[Tensor]] = defaultdict(list)
    for record in records:
        full = torch.tensor(
            record.fisher_second_moment,
            dtype=torch.float64,
        )
        grouped[record.family_id].append(
            full.index_select(0, selected)
            .index_select(1, selected)
            .contiguous()
        )
    return {
        family: (
            sum(values, torch.zeros_like(values[0])) / len(values)
        ).contiguous()
        for family, values in sorted(grouped.items())
    }


def _mean_tensors(values: Sequence[Tensor]) -> Tensor:
    if not values:
        raise ValueError("cannot average empty Fisher tensors")
    result = sum(values, torch.zeros_like(values[0])) / len(values)
    return ((result + result.T) * 0.5).contiguous()


def _correlation(fisher: Tensor, left: int, right: int) -> float:
    denominator = math.sqrt(
        max(float(fisher[left, left]), 0.0)
        * max(float(fisher[right, right]), 0.0)
    )
    if denominator <= 1.0e-12:
        return 0.0
    value = float(fisher[left, right]) / denominator
    return min(max(value, -1.0), 1.0)


def _coupling_graph(
    records: Sequence[TokenLossFisherPromptRecord],
    *,
    coordinate_indices: tuple[int, ...],
) -> dict[str, object]:
    family = _family_fisher(records, coordinate_indices)
    families = tuple(sorted(family))
    global_fisher = _mean_tensors(tuple(family.values()))
    names = tuple(
        COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES[index]
        for index in coordinate_indices
    )
    fold_fishers = {
        held: _mean_tensors(
            tuple(
                family[name] for name in families if name != held
            )
        )
        for held in families
    }
    edges: list[dict[str, object]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            global_value = _correlation(global_fisher, left, right)
            folds = tuple(
                _correlation(fold_fishers[held], left, right)
                for held in families
            )
            support = sum(
                abs(value)
                >= TOKEN_FISHER_EDGE_MINIMUM_ABSOLUTE_CORRELATION
                and (
                    global_value == 0.0
                    or math.copysign(1.0, value)
                    == math.copysign(1.0, global_value)
                )
                for value in folds
            )
            edges.append(
                {
                    "left_coordinate": names[left],
                    "right_coordinate": names[right],
                    "global_correlation": global_value,
                    "fold_correlations_by_held_family": {
                        held: value
                        for held, value in zip(
                            families, folds, strict=True
                        )
                    },
                    "stable_fold_count": support,
                    "stable": (
                        abs(global_value)
                        >= TOKEN_FISHER_EDGE_MINIMUM_ABSOLUTE_CORRELATION
                        and support
                        >= TOKEN_FISHER_EDGE_MINIMUM_STABLE_FOLDS
                    ),
                }
            )
    canonical_edges = tuple(
        sorted(
            edges,
            key=lambda row: (
                -abs(float(row["global_correlation"])),
                str(row["left_coordinate"]),
                str(row["right_coordinate"]),
            ),
        )
    )
    return {
        "coordinate_names": names,
        "family_balanced_fisher_second_moment": tuple(
            tuple(float(value) for value in row)
            for row in global_fisher
        ),
        "edges": canonical_edges,
        "stable_edge_count": sum(
            bool(row["stable"]) for row in canonical_edges
        ),
        "minimum_absolute_correlation": (
            TOKEN_FISHER_EDGE_MINIMUM_ABSOLUTE_CORRELATION
        ),
        "minimum_stable_folds": (
            TOKEN_FISHER_EDGE_MINIMUM_STABLE_FOLDS
        ),
        "fisher_coupling_is_symmetric": True,
        "causal_direction_inferred": False,
    }


def _decision(
    cumulative: Mapping[str, object],
    ew: Mapping[str, object],
    cumulative_graph: Mapping[str, object],
    ew_graph: Mapping[str, object],
) -> dict[str, object]:
    cumulative_passed = bool(cumulative["passed"]) and int(
        cumulative_graph["stable_edge_count"]
    ) > 0
    ew_passed = bool(ew["passed"]) and int(
        ew_graph["stable_edge_count"]
    ) > 0
    selected: str | None = None
    if cumulative_passed or ew_passed:
        cumulative_score = (
            float(cumulative["family_macro_relative_rmse_improvement"])
            if cumulative_passed
            else -math.inf
        )
        ew_score = (
            float(ew["family_macro_relative_rmse_improvement"])
            if ew_passed
            else -math.inf
        )
        selected = "ew" if ew_score > cumulative_score else "cumulative"
    return {
        "cumulative_science_and_edge_gates_passed": cumulative_passed,
        "ew_science_and_edge_gates_passed": ew_passed,
        "any_arm_passed": cumulative_passed or ew_passed,
        "recommended_arm": selected,
        "provider_compiled": False,
        "runtime_claim_authorized": False,
        "next_step": (
            "held_family_finite_displacement_and_causal_jvp_orientation"
            if selected is not None
            else "do_not_compile_expand_or_revise_feature_hypothesis"
        ),
    }


def build_gemma_iterative_token_fisher_development_report(
    *,
    token_tangent_records: Sequence[object],
    prompt_records: Sequence[object],
    lineage: Mapping[str, object],
    token_vjp_artifact_sha256_by_example: Mapping[str, str],
    total_backward_call_count: int,
    vjp_chunk_size: int,
) -> dict[str, object]:
    """Build the strict development-only Fisher/LOFO report."""

    tangents, prompts = _records(token_tangent_records, prompt_records)
    lineage_payload = dict(sorted(lineage.items()))
    if not lineage_payload:
        raise ValueError("token Fisher lineage must be nonempty")
    for name, value in lineage_payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError("token Fisher lineage key is invalid")
        _require_sha256(value, label=f"token Fisher lineage {name}")
    vjp_receipts = dict(
        sorted(token_vjp_artifact_sha256_by_example.items())
    )
    if (
        set(vjp_receipts)
        != {row.example_id for row in tangents}
        or any(
            _SHA256.fullmatch(value) is None
            for value in vjp_receipts.values()
        )
    ):
        raise ValueError("token Fisher VJP receipts differ from records")
    if (
        type(total_backward_call_count) is not int
        or total_backward_call_count <= 0
        or type(vjp_chunk_size) is not int
        or vjp_chunk_size <= 0
    ):
        raise ValueError("token Fisher backward accounting is invalid")
    expected_backward_calls = sum(
        (
            row.supervised_token_count
            + vjp_chunk_size
            - 1
        )
        // vjp_chunk_size
        for row in tangents
    )
    if total_backward_call_count != expected_backward_calls:
        raise ValueError("token Fisher backward call count differs")
    cumulative_report = (
        analyze_cumulative_occupancy_token_loss_fisher_lofo(prompts)
    )
    ew_report = analyze_ew_occupancy_token_loss_fisher_lofo(prompts)
    cumulative = cumulative_report.to_dict()
    ew = ew_report.to_dict()
    cumulative_graph = _coupling_graph(
        prompts,
        coordinate_indices=(
            CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES
        ),
    )
    ew_graph = _coupling_graph(
        prompts,
        coordinate_indices=EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    )
    supervised_tokens = sum(
        row.supervised_token_count for row in tangents
    )
    payload: dict[str, object] = {
        "schema": TOKEN_FISHER_DEVELOPMENT_SCHEMA,
        "lineage": lineage_payload,
        "coordinate_order": TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER,
        "token_tangent_records": tuple(
            row.to_dict() for row in tangents
        ),
        "prompt_fisher_records": tuple(
            row.to_dict() for row in prompts
        ),
        "analysis": {
            "cumulative": cumulative,
            "ew": ew,
            "cumulative_coupling_graph": cumulative_graph,
            "ew_coupling_graph": ew_graph,
        },
        "decision": _decision(
            cumulative,
            ew,
            cumulative_graph,
            ew_graph,
        ),
        "resources": {
            "fit_example_count": len(tangents),
            "fit_family_count": len(
                {row.family_id for row in tangents}
            ),
            "supervised_token_count": supervised_tokens,
            "source_forward_count": len(tangents),
            "parent_token_vjp_forward_count": len(tangents),
            "total_model_forward_count": 2 * len(tangents),
            "model_forward_count_per_example": 2,
            "token_vjp_backward_call_count": (
                total_backward_call_count
            ),
            "token_vjp_chunk_size": vjp_chunk_size,
            "candidate_forward_count": 0,
            "fresh_forward_count": 0,
            "serving_learned_parameter_count": 0,
            "serving_logical_macs_per_token": 0,
        },
        "audit": {
            "execution_mode": (
                "development_exact_token_loss_fisher_family_lofo"
            ),
            "family_blocked_leave_one_family_out": True,
            "tokens_used_as_independent_split_units": False,
            "exact_loss_token_vjps": True,
            "activation_site_pseudo_fisher_used": False,
            "fisher_coupling_is_symmetric": True,
            "causal_direction_inferred": False,
            "development_only": True,
            "selection_panel_referenced": False,
            "selection_panel_opened": False,
            "selection_claim_created": False,
            "raw_prompts_retained": False,
            "raw_token_ids_retained": False,
            "raw_logits_retained": False,
            "raw_activations_retained": False,
            "raw_gradients_retained": False,
            "model_weights_retained": False,
            "reduced_token_jacobian_rows_retained": True,
            "token_vjp_artifact_sha256_by_example": vjp_receipts,
        },
    }
    return {**payload, "report_sha256": _sha256(payload)}


def validate_gemma_iterative_token_fisher_development_report(
    report: Mapping[str, object],
) -> None:
    """Replay all derived fits, coupling edges, gates, and the outer receipt."""

    if not isinstance(report, Mapping):
        raise TypeError("token Fisher development report must be a mapping")
    expected = {
        "schema",
        "lineage",
        "coordinate_order",
        "token_tangent_records",
        "prompt_fisher_records",
        "analysis",
        "decision",
        "resources",
        "audit",
        "report_sha256",
    }
    if set(report) != expected:
        raise ValueError("token Fisher development report fields differ")
    if report["schema"] != TOKEN_FISHER_DEVELOPMENT_SCHEMA:
        raise ValueError("token Fisher development schema differs")
    if tuple(report["coordinate_order"]) != (
        TOKEN_OCCUPANCY_TANGENT_COORDINATE_ORDER
    ):
        raise ValueError("token Fisher development coordinates differ")
    tangents, prompts = _records(
        report["token_tangent_records"],  # type: ignore[arg-type]
        report["prompt_fisher_records"],  # type: ignore[arg-type]
    )
    analysis = report["analysis"]
    if not isinstance(analysis, Mapping):
        raise ValueError("token Fisher analysis must be a mapping")
    expected_analysis = {
        "cumulative",
        "ew",
        "cumulative_coupling_graph",
        "ew_coupling_graph",
    }
    if set(analysis) != expected_analysis:
        raise ValueError("token Fisher analysis fields differ")
    cumulative = (
        analyze_cumulative_occupancy_token_loss_fisher_lofo(prompts)
        .to_dict()
    )
    ew = (
        analyze_ew_occupancy_token_loss_fisher_lofo(prompts).to_dict()
    )
    cumulative_graph = _coupling_graph(
        prompts,
        coordinate_indices=(
            CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES
        ),
    )
    ew_graph = _coupling_graph(
        prompts,
        coordinate_indices=EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    )
    expected_decision = _decision(
        cumulative,
        ew,
        cumulative_graph,
        ew_graph,
    )
    if (
        not _canonical_equal(analysis["cumulative"], cumulative)
        or not _canonical_equal(analysis["ew"], ew)
        or not _canonical_equal(
            analysis["cumulative_coupling_graph"],
            cumulative_graph,
        )
        or not _canonical_equal(
            analysis["ew_coupling_graph"],
            ew_graph,
        )
        or not _canonical_equal(report["decision"], expected_decision)
    ):
        raise ValueError("token Fisher derived analysis differs")
    resources = report["resources"]
    audit = report["audit"]
    lineage = report["lineage"]
    if (
        not isinstance(resources, Mapping)
        or not isinstance(audit, Mapping)
        or not isinstance(lineage, Mapping)
    ):
        raise ValueError("token Fisher report bindings must be mappings")
    if not lineage:
        raise ValueError("token Fisher lineage must be nonempty")
    for name, value in lineage.items():
        if not isinstance(name, str) or not name:
            raise ValueError("token Fisher lineage key is invalid")
        _require_sha256(value, label=f"token Fisher lineage {name}")
    expected_tokens = sum(
        row.supervised_token_count for row in tangents
    )
    chunk_size = resources.get("token_vjp_chunk_size")
    backward_calls = resources.get("token_vjp_backward_call_count")
    if (
        type(chunk_size) is not int
        or chunk_size <= 0
        or type(backward_calls) is not int
        or backward_calls
        != sum(
            (row.supervised_token_count + chunk_size - 1)
            // chunk_size
            for row in tangents
        )
    ):
        raise ValueError("token Fisher backward resource receipt differs")
    if (
        resources.get("fit_example_count") != _EXPECTED_EXAMPLES
        or resources.get("fit_family_count") != _EXPECTED_FAMILIES
        or resources.get("supervised_token_count") != expected_tokens
        or resources.get("source_forward_count") != _EXPECTED_EXAMPLES
        or resources.get("parent_token_vjp_forward_count")
        != _EXPECTED_EXAMPLES
        or resources.get("total_model_forward_count")
        != 2 * _EXPECTED_EXAMPLES
        or resources.get("model_forward_count_per_example") != 2
        or resources.get("candidate_forward_count") != 0
        or resources.get("fresh_forward_count") != 0
        or resources.get("serving_learned_parameter_count") != 0
        or resources.get("serving_logical_macs_per_token") != 0
    ):
        raise ValueError("token Fisher resource receipt differs")
    expected_resource_fields = {
        "fit_example_count",
        "fit_family_count",
        "supervised_token_count",
        "source_forward_count",
        "parent_token_vjp_forward_count",
        "total_model_forward_count",
        "model_forward_count_per_example",
        "token_vjp_backward_call_count",
        "token_vjp_chunk_size",
        "candidate_forward_count",
        "fresh_forward_count",
        "serving_learned_parameter_count",
        "serving_logical_macs_per_token",
    }
    if set(resources) != expected_resource_fields:
        raise ValueError("token Fisher resource fields differ")
    expected_audit_fields = {
        "execution_mode",
        "family_blocked_leave_one_family_out",
        "tokens_used_as_independent_split_units",
        "exact_loss_token_vjps",
        "activation_site_pseudo_fisher_used",
        "fisher_coupling_is_symmetric",
        "causal_direction_inferred",
        "development_only",
        "selection_panel_referenced",
        "selection_panel_opened",
        "selection_claim_created",
        "raw_prompts_retained",
        "raw_token_ids_retained",
        "raw_logits_retained",
        "raw_activations_retained",
        "raw_gradients_retained",
        "model_weights_retained",
        "reduced_token_jacobian_rows_retained",
        "token_vjp_artifact_sha256_by_example",
    }
    if (
        set(audit) != expected_audit_fields
        or audit.get("execution_mode")
        != "development_exact_token_loss_fisher_family_lofo"
    ):
        raise ValueError("token Fisher audit fields differ")
    if any(
        audit.get(name) is not expected_value
        for name, expected_value in (
            ("family_blocked_leave_one_family_out", True),
            ("tokens_used_as_independent_split_units", False),
            ("exact_loss_token_vjps", True),
            ("activation_site_pseudo_fisher_used", False),
            ("fisher_coupling_is_symmetric", True),
            ("causal_direction_inferred", False),
            ("development_only", True),
            ("selection_panel_referenced", False),
            ("selection_panel_opened", False),
            ("selection_claim_created", False),
            ("raw_prompts_retained", False),
            ("raw_token_ids_retained", False),
            ("raw_logits_retained", False),
            ("raw_activations_retained", False),
            ("raw_gradients_retained", False),
            ("model_weights_retained", False),
            ("reduced_token_jacobian_rows_retained", True),
        )
    ):
        raise ValueError("token Fisher safety audit differs")
    vjp_receipts = audit.get(
        "token_vjp_artifact_sha256_by_example"
    )
    if (
        not isinstance(vjp_receipts, Mapping)
        or set(vjp_receipts)
        != {row.example_id for row in tangents}
    ):
        raise ValueError("token Fisher VJP receipt map differs")
    for value in vjp_receipts.values():
        _require_sha256(value, label="token Fisher VJP artifact")
    payload = {key: report[key] for key in expected if key != "report_sha256"}
    if _require_sha256(
        report["report_sha256"],
        label="token Fisher development report",
    ) != _sha256(payload):
        raise ValueError("token Fisher development report hash mismatch")


def publish_gemma_iterative_token_fisher_development_report(
    destination: Path,
    report: Mapping[str, object],
) -> None:
    """Atomically publish one validated, non-overwriting local report."""

    validate_gemma_iterative_token_fisher_development_report(report)
    if not isinstance(destination, Path):
        raise TypeError("token Fisher destination must be a Path")
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite token Fisher development report"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError("token Fisher temporary report already exists")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
