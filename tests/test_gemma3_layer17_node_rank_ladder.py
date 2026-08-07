from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json

import pytest

from fisher_graph.gemma3_layer17_node_rank_ladder import (
    DEFAULT_LAYER17_NODE_RANK_ARM_SPECS,
    LAYER17_SOURCE_MACS_PER_TOKEN,
    LAYER17_SOURCE_PARAMETERS,
    Layer17NodeRankLadderPlan,
    Layer17NodeRankResourceRow,
    build_default_layer17_node_rank_ladder_plan,
    build_layer17_node_rank_resource_row,
    resolve_layer17_node_ranks,
    validate_layer17_node_rank_ladder_plan,
)


def _row(label: str) -> Layer17NodeRankResourceRow:
    plan = build_default_layer17_node_rank_ladder_plan()
    return next(row for row in plan.rows if row.spec.label == label)


def _assert_no_runtime_payload(value: object) -> None:
    assert not hasattr(value, "shape")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_runtime_payload(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_runtime_payload(item)

def test_rank_caps_preserve_frozen_fragments_and_saturate_independently() -> None:
    assert resolve_layer17_node_ranks(32) == (32, 32, 32, 32)
    assert resolve_layer17_node_ranks(48) == (48, 38, 48, 48)
    assert resolve_layer17_node_ranks(64) == (54, 38, 64, 53)
    assert resolve_layer17_node_ranks(1000) == (54, 38, 85, 53)

    with pytest.raises(ValueError, match="positive integer"):
        resolve_layer17_node_ranks(0)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_layer17_node_ranks(True)


def test_baseline_dynamic_reproduces_the_existing_layer17_accounting() -> None:
    row = _row("baseline-dynamic")

    assert row.node_ranks == (32, 32, 32, 32)
    assert row.node_parameter_count == 127_616
    assert row.interaction_parameter_count == 5_571
    assert row.graph_parameter_count == 133_187
    assert row.node_macs_per_token == 124_928
    assert row.interaction_dense_macs_per_token == 5_472
    assert row.graph_dense_macs_per_token == 130_400
    assert row.conditional_routing_macs_per_token == 96
    assert row.conditional_dense_message_macs_per_token == 5_376
    assert (
        row.conditional_selected_message_macs_per_token_upper_bound == 1_792
    )
    assert row.executed_graph_macs_per_token_upper_bound == 126_816
    assert row.net_parameter_savings == 308_413
    assert row.net_executed_macs_saved_per_token == 314_784


def test_edgeless_baseline_removes_all_interaction_costs() -> None:
    dynamic = _row("baseline-dynamic")
    edgeless = _row("baseline-edgeless")

    assert edgeless.node_ranks == dynamic.node_ranks
    assert edgeless.node_parameter_count == dynamic.node_parameter_count
    assert edgeless.node_macs_per_token == dynamic.node_macs_per_token
    assert edgeless.interaction_count == 0
    assert edgeless.interaction_parameter_count == 0
    assert edgeless.interaction_dense_macs_per_token == 0
    assert edgeless.conditional_routing_macs_per_token == 0
    assert edgeless.graph_parameter_count == 127_616
    assert edgeless.executed_graph_macs_per_token_upper_bound == 124_928
    assert edgeless.net_parameter_savings == 313_984
    assert edgeless.net_executed_macs_saved_per_token == 316_672


def test_latent_lift_uses_the_general_generator_rank_formula() -> None:
    row = _row("latent-lift-edgeless")
    width = 640
    node_rank = 32
    generator_rank = 32
    expected_node_macs = (
        width * generator_rank
        + generator_rank * node_rank
        + node_rank * width
    )
    expected_node_parameters = expected_node_macs + node_rank + width

    assert row.node_parameter_count == 4 * expected_node_parameters == 170_624
    assert row.node_macs_per_token == 4 * expected_node_macs == 167_936
    assert row.graph_parameter_count == row.node_parameter_count
    assert (
        row.executed_graph_macs_per_token_upper_bound == row.node_macs_per_token
    )
    assert row.net_parameter_savings == 270_976
    assert row.net_executed_macs_saved_per_token == 273_664


@pytest.mark.parametrize(
    ("label", "ranks", "parameters", "dense_macs", "executed_macs"),
    (
        (
            "cap48-dynamic-diagnostic",
            (48, 38, 48, 48),
            173_183,
            170_304,
            163_952,
        ),
        (
            "cap64-dynamic-diagnostic",
            (54, 38, 64, 53),
            193_355,
            190_428,
            183_058,
        ),
    ),
)
def test_capped_dynamic_diagnostic_rows(
    label: str,
    ranks: tuple[int, ...],
    parameters: int,
    dense_macs: int,
    executed_macs: int,
) -> None:
    row = _row(label)

    assert row.node_ranks == ranks
    assert row.graph_parameter_count == parameters
    assert row.graph_dense_macs_per_token == dense_macs
    assert row.executed_graph_macs_per_token_upper_bound == executed_macs
    assert row.net_parameter_savings == LAYER17_SOURCE_PARAMETERS - parameters
    assert row.net_executed_macs_saved_per_token == (
        LAYER17_SOURCE_MACS_PER_TOKEN - executed_macs
    )


def test_planner_is_general_over_generator_rank_and_edge_policy() -> None:
    lower = build_layer17_node_rank_resource_row(
        label="synthetic-r8",
        mode_rank_cap=32,
        generator_rank=8,
        edge_policy="edgeless",
    )
    higher = build_layer17_node_rank_resource_row(
        label="synthetic-r24",
        mode_rank_cap=32,
        generator_rank=24,
        edge_policy="edgeless",
    )

    assert higher.node_parameter_count > lower.node_parameter_count
    assert higher.node_macs_per_token > lower.node_macs_per_token
    with pytest.raises(ValueError, match="smallest resolved node rank"):
        build_layer17_node_rank_resource_row(
            label="invalid-rank",
            mode_rank_cap=32,
            generator_rank=33,
            edge_policy="edgeless",
        )
    with pytest.raises(ValueError, match="unsupported.*edge policy"):
        build_layer17_node_rank_resource_row(
            label="invalid-policy",
            mode_rank_cap=32,
            generator_rank=16,
            edge_policy="dense",  # type: ignore[arg-type]
        )


def test_analytic_invariants_reject_tampered_resource_values() -> None:
    row = _row("latent-lift-edgeless")

    with pytest.raises(ValueError, match="analytic invariant"):
        replace(row, graph_parameter_count=row.graph_parameter_count + 1)
    with pytest.raises(ValueError, match="analytic invariant"):
        replace(
            row,
            executed_graph_macs_per_token_upper_bound=(
                row.executed_graph_macs_per_token_upper_bound - 1
            ),
        )


def test_source_safe_json_round_trip_is_strict_and_deterministic() -> None:
    first = build_default_layer17_node_rank_ladder_plan()
    second = build_default_layer17_node_rank_ladder_plan()
    state = json.loads(json.dumps(first.state_dict()))
    restored = validate_layer17_node_rank_ladder_plan(state)

    assert first.artifact_sha256 == second.artifact_sha256
    assert restored.state_dict() == first.state_dict()
    assert tuple(spec.label for spec in DEFAULT_LAYER17_NODE_RANK_ARM_SPECS) == (
        "baseline-dynamic",
        "baseline-edgeless",
        "latent-lift-edgeless",
        "cap48-dynamic-diagnostic",
        "cap64-dynamic-diagnostic",
    )
    for field in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_activation_tensors",
        "contains_gradient_tensors",
        "contains_model_or_candidate_weights",
        "model_or_tokenizer_accessed",
        "calibration_b_opened",
        "validation_opened",
        "test_opened",
    ):
        assert state[field] is False
    _assert_no_runtime_payload(state)

    state["unexpected"] = True
    with pytest.raises(ValueError, match="fields are invalid"):
        Layer17NodeRankLadderPlan.from_state_dict(state)


def test_hash_and_nested_resource_tampering_are_rejected() -> None:
    state = build_default_layer17_node_rank_ladder_plan().state_dict()
    bad_hash = dict(state)
    bad_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_layer17_node_rank_ladder_plan(bad_hash)

    bad_row = json.loads(json.dumps(state))
    bad_row["rows"][0]["graph_parameter_count"] += 1
    with pytest.raises(ValueError, match="analytic invariant"):
        validate_layer17_node_rank_ladder_plan(bad_row)
