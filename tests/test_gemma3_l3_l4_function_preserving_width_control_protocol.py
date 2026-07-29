from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from fisher_graph.gemma3_l3_l4_function_preserving_width_control_protocol import (
    DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256,
    DeterministicInitializationBindings,
    FunctionPreservingWidthControlProtocol,
    NestedLiftSpec,
    PairedTrainingSpec,
    SourceArtifactBindings,
    WidthExecutorSpec,
    default_function_preserving_width_control_protocol,
)


def test_protocol_literal_trust_anchor_and_round_trip() -> None:
    protocol = default_function_preserving_width_control_protocol()

    assert protocol.protocol_sha256 == (
        DEFAULT_FUNCTION_PRESERVING_WIDTH_CONTROL_PROTOCOL_SHA256
    )
    assert protocol.protocol_sha256 == (
        "c3ad81c84d41108839b5fcab13e3b5d47d99a55ae9a9223c3f116edb6b457597"
    )
    assert (
        FunctionPreservingWidthControlProtocol.from_state_dict(
            protocol.state_dict()
        )
        == protocol
    )


def test_protocol_module_imports_only_standard_library() -> None:
    path = Path(
        "src/fisher_graph/"
        "gemma3_l3_l4_function_preserving_width_control_protocol.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])

    assert roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "re",
    }


def test_protocol_freezes_paired_geometry_and_authorities() -> None:
    protocol = default_function_preserving_width_control_protocol()
    state = protocol.state_dict()

    assert protocol.rank16_executor.output_modes == 16
    assert protocol.rank64_executor.output_modes == 64
    assert protocol.rank16_executor.expert_rank == 16
    assert protocol.rank64_executor.expert_rank == 16
    assert protocol.lift.added_rank == 48
    assert state["primary_arms"] == [
        "rank16_exact_d3_replay",
        "rank64_function_preserving_lift",
    ]
    assert state["c2_selection_allowed"] is False
    assert state["c3_allowed"] is False
    assert state["compression_claim_allowed"] is False
    assert state["longer_optimization_mixed_into_primary"] is False
    assert state["expert_rank_change_mixed_into_primary"] is False
    assert state["conditional_residual_mixed_into_primary"] is False


def test_training_is_exact_d3_contract() -> None:
    training = PairedTrainingSpec()

    assert training.primary_seed == 20_260_728_402
    assert training.replication_seed == 20_260_729_402
    assert training.steps == 600
    assert training.learning_rate == 1e-3
    assert (
        training.pointwise_weight,
        training.relative_delta_weight,
        training.direction_weight,
        training.midpoint_jvp_weight,
        training.intended_null_weight,
    ) == (1.0, 2.0, 2.0, 1.0, 1.0)
    assert training.primary_initial_metrics_sha256 == (
        "56ee6759438f98b1ba9628c055358d29622af48899d49a51e23297ad9779aa20"
    )


def test_lift_freezes_equivalence_and_gradient_open_gates() -> None:
    lift = NestedLiftSpec()
    state = lift.state_dict()

    assert lift.equivalence_absolute_tolerance == 1e-12
    assert lift.equivalence_relative_tolerance == 1e-12
    assert lift.gradient_norm_floor == 1e-12
    assert lift.extra_encoder_initialization == (
        "identity_remaining_modal_modes"
    )
    assert lift.extra_decoder_initialization == "exact_zero_rows"
    assert "step1" in state["gradient_open_rule"]
    assert "step2" in state["gradient_open_rule"]


def test_protocol_binds_both_seeded_initial_parameter_pairs() -> None:
    bindings = DeterministicInitializationBindings()

    assert bindings.hashes_for(
        seed_role="primary",
        arm="rank16",
    )["executor_sha256"].startswith("94a0d18e")
    assert bindings.hashes_for(
        seed_role="primary",
        arm="rank64",
    )["executor_sha256"].startswith("717bdc26")
    assert bindings.hashes_for(
        seed_role="replication",
        arm="rank16",
    )["executor_sha256"].startswith("9e7498e6")
    assert bindings.hashes_for(
        seed_role="replication",
        arm="rank64",
    )["executor_sha256"].startswith("94d03be0")
    tampered = replace(
        bindings,
        primary_rank16_encoder_sha256="0" * 64,
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="initialization drifted"):
        replace(
            default_function_preserving_width_control_protocol(),
            initialization=tampered,
            artifact_sha256="",
        )


def test_both_predecessor_artifacts_are_bound_exactly() -> None:
    sources = SourceArtifactBindings()

    assert sources.d3_logical_artifact_sha256.startswith("e171361c")
    assert sources.d3_primary_plan_sha256.startswith("45912143")
    assert sources.rank64_logical_artifact_sha256.startswith("cccc3fc5")
    assert sources.rank64_primary_plan_sha256.startswith("78a63958")
    assert sources.rank64_tensor_file_sha256.startswith("3d44215a")


@pytest.mark.parametrize(
    ("factory", "change"),
    [
        (
            lambda: WidthExecutorSpec(input_modes=18, output_modes=16),
            {"expert_rank": 32},
        ),
        (PairedTrainingSpec, {"steps": 601}),
        (NestedLiftSpec, {"gradient_norm_floor": 1e-11}),
    ],
)
def test_nested_declarations_reject_scientific_drift(
    factory: object,
    change: dict[str, object],
) -> None:
    value = factory()  # type: ignore[operator]
    with pytest.raises(ValueError):
        replace(value, **change)


def test_protocol_rejects_semantic_tampering() -> None:
    state = default_function_preserving_width_control_protocol().state_dict()
    state["c3_allowed"] = True

    with pytest.raises(ValueError, match="semantics drifted"):
        FunctionPreservingWidthControlProtocol.from_state_dict(state)
