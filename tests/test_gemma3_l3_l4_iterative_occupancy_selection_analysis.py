from __future__ import annotations

import copy
import hashlib

import pytest

from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_iterative_occupancy_selection_analysis import (
    CUMULATIVE_OCCUPANCY_ARM,
    EW_OCCUPANCY_ARM,
    build_gemma_iterative_occupancy_selection_report,
    validate_gemma_iterative_occupancy_selection_report,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _observations(
    arm: str,
    *,
    candidate_offset: float,
    kl: float,
) -> tuple[GemmaH4DampingFiniteNLLObservation, ...]:
    rows = []
    for index in range(16):
        rows.append(
            GemmaH4DampingFiniteNLLObservation(
                example_id=f"selection-example-{index:02d}",
                family_id=f"selection-family-{index % 8}",
                supervised_tokens=10,
                source_summed_nll=100.0 + index,
                candidate_summed_nll=100.0 + index + candidate_offset,
                source_to_candidate_summed_kl=kl,
                top1_matches=10,
                source_logits_sha256=_sha(f"source-{index}"),
                candidate_logits_sha256=_sha(f"{arm}-{index}"),
                targets_sha256=_sha(f"targets-{index}"),
            )
        )
    return tuple(rows)


def _development(selected: str) -> dict[str, object]:
    return {
        "selected_arm_id": selected,
        "selection_opened": False,
        "selection_rule_frozen": True,
        "scientific_gates_by_arm": {
            CUMULATIVE_OCCUPANCY_ARM: {"passed": True},
            EW_OCCUPANCY_ARM: {"passed": True},
        },
        "selection_rule": (
            "minimum_family_macro_predicted_absolute_delta_nll"
        ),
    }


def _audit() -> dict[str, object]:
    return {
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
        "execution_receipt_sha256": _sha("execution"),
    }


def _report(*, selected: str = CUMULATIVE_OCCUPANCY_ARM):
    parent = _observations(
        "parent",
        candidate_offset=0.8,
        kl=0.01,
    )
    cumulative = _observations(
        "cumulative",
        candidate_offset=0.2,
        kl=0.002,
    )
    ew = _observations(
        "ew",
        candidate_offset=0.3,
        kl=0.003,
    )
    manifest = {
        row.example_id: row.family_id
        for row in parent
    }
    return build_gemma_iterative_occupancy_selection_report(
        development=_development(selected),
        parent_observations=parent,
        cumulative_observations=cumulative,
        ew_observations=ew,
        manifest=manifest,
        lineage={"parent_artifact_sha256": _sha("parent")},
        resources={
            CUMULATIVE_OCCUPANCY_ARM: {
                "learned_float_scalar_count": 6,
            },
            EW_OCCUPANCY_ARM: {
                "learned_float_scalar_count": 6,
            },
        },
        audit=_audit(),
    )


def test_selection_report_replays_and_only_preselected_arm_qualifies() -> None:
    report = _report()

    validate_gemma_iterative_occupancy_selection_report(report)

    assert report["decision"]["development_selected_arm_id"] == (  # type: ignore[index]
        CUMULATIVE_OCCUPANCY_ARM
    )
    assert report["decision"]["qualified_for_guard"] is True  # type: ignore[index]
    comparisons = report["selection"]["paired_comparisons"]  # type: ignore[index]
    assert comparisons["parent_to_cumulative"][  # type: ignore[index]
        "challenger_arm_id"
    ] == CUMULATIVE_OCCUPANCY_ARM
    assert comparisons["parent_to_ew"]["challenger_arm_id"] == (  # type: ignore[index]
        EW_OCCUPANCY_ARM
    )


def test_report_rejects_tampering_and_post_open_arm_selection() -> None:
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["decision"]["qualified_for_guard"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="hash differs"):
        validate_gemma_iterative_occupancy_selection_report(tampered)

    development = _development(CUMULATIVE_OCCUPANCY_ARM)
    development["selection_opened"] = True
    parent = _observations("parent", candidate_offset=0.8, kl=0.01)
    manifest = {row.example_id: row.family_id for row in parent}
    with pytest.raises(ValueError, match="precede fresh-panel"):
        build_gemma_iterative_occupancy_selection_report(
            development=development,
            parent_observations=parent,
            cumulative_observations=_observations(
                "cumulative",
                candidate_offset=0.2,
                kl=0.002,
            ),
            ew_observations=_observations(
                "ew",
                candidate_offset=0.3,
                kl=0.003,
            ),
            manifest=manifest,
            lineage={"parent_artifact_sha256": _sha("parent")},
            resources={
                CUMULATIVE_OCCUPANCY_ARM: {},
                EW_OCCUPANCY_ARM: {},
            },
            audit=_audit(),
        )
