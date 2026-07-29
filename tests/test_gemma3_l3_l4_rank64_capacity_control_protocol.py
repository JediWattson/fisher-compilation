from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_rank64_capacity_control_protocol as protocol_module,
)
from fisher_graph.gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    default_objective_balance_diagnostic_protocol,
)
from fisher_graph.gemma3_l3_l4_rank64_capacity_control_protocol import (
    CAPACITY_CONTROL_BASELINE_LATENT_RANK,
    CAPACITY_CONTROL_EXECUTION_DEVICE,
    CAPACITY_CONTROL_EXECUTION_DTYPE,
    CAPACITY_CONTROL_LATENT_RANK,
    CAPACITY_CONTROL_PRIMARY_SEED,
    CAPACITY_CONTROL_REPLICATION_SEED,
    DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256,
    MatchedD3TrainingSpec,
    ObjectiveBalanceResultBinding,
    Rank64CapacityControlProtocol,
    Rank64ExecutorSpec,
    default_rank64_capacity_control_protocol,
)


def test_default_protocol_authenticates_fit_only_rank64_control() -> None:
    protocol = default_rank64_capacity_control_protocol()

    assert protocol.protocol_sha256 == (
        DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256
    )
    assert protocol.protocol_sha256 == (
        "03b1e595836ee325b83f5c2fc7355b31f7e5e6deceba92f9ad98ae27c29e6cf5"
    )
    assert protocol.baseline_latent_rank == (
        CAPACITY_CONTROL_BASELINE_LATENT_RANK
    ) == 16
    assert protocol.latent_rank == CAPACITY_CONTROL_LATENT_RANK == 64
    assert protocol.execution_device == (
        CAPACITY_CONTROL_EXECUTION_DEVICE
    ) == "cpu"
    assert protocol.execution_dtype == (
        CAPACITY_CONTROL_EXECUTION_DTYPE
    ) == "float32"
    assert protocol.primary_seed == (
        CAPACITY_CONTROL_PRIMARY_SEED
    ) == 20_260_728_402
    assert protocol.replication_seed == (
        CAPACITY_CONTROL_REPLICATION_SEED
    ) == 20_260_729_402
    assert protocol.state_dict()["scientific_scope"] == (
        "fit_only_rank64_capacity_control_not_compression_or_generalization"
    )


def test_control_matches_d3_objective_optimizer_and_seeds() -> None:
    protocol = default_rank64_capacity_control_protocol()
    training = protocol.training
    source = default_objective_balance_diagnostic_protocol()
    d3 = source.recipe("d3_unit_rms_family_balanced_direction")

    assert protocol.baseline_latent_rank == source.latent_rank
    assert protocol.primary_seed == source.primary_seed == d3.primary_seed
    assert protocol.replication_seed == source.replication_seed
    assert training.recipe_id == d3.recipe_id
    assert training.training_metric == d3.training_metric
    assert training.signed_pair_multiplicity == d3.signed_pair_multiplicity
    assert training.pointwise_weight == d3.pointwise_weight
    assert (
        training.sensitivity_relative_delta_weight
        == d3.sensitivity_relative_delta_weight
    )
    assert training.sensitivity_direction_weight == d3.direction_weight
    assert training.midpoint_jvp_weight == d3.midpoint_jvp_weight
    assert training.intended_null_weight == d3.intended_null_weight
    assert training.steps == d3.steps == 600
    assert training.learning_rate == d3.learning_rate == 1e-3
    assert training.sensitivity_relative_floor == 1e-6
    assert training.direction_norm_floor == 1e-8
    assert training.jvp_relative_floor == 1e-6


def test_fit_data_order_and_fit_gates_are_frozen() -> None:
    training = default_rank64_capacity_control_protocol().training
    state = training.state_dict()

    assert training.fit_data_binding_sha256 == (
        "a84f73269fd3bf71c350c79309ef7539a2728b67007c75950aa9f87fb2447c17"
    )
    assert training.ordinary_gates_sha256 == (
        "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
    )
    assert training.contrast_gates_sha256 == (
        "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
    )
    assert state["fit_materialization_roles"] == ["pilot", "fit"]
    assert state["batch_order_semantics"] == (
        "exact_recipe_independent_c2_fit_data_binding_and_index_order"
    )
    assert state["canonical_scoring_semantics"] == (
        "raw_canonical_fisher_metric_for_all_fit_scoring"
    )
    assert state["ordinary_gate_semantics"] == (
        "all_12_existing_gate_flags_must_pass"
    )
    assert state["contrast_gate_semantics"] == (
        "all_null_radial_and_signed_fit_contrasts_must_pass"
    )


def test_source_result_and_failed_d3_primary_are_bound_exactly() -> None:
    source = default_rank64_capacity_control_protocol().source_result
    state = source.state_dict()

    assert source.protocol_sha256 == (
        "d502d003fd86f6ef7322e35854d0bb738fdc4cfa6fa5089c2812e366a142d2eb"
    )
    assert source.logical_artifact_sha256 == (
        "e171361c2a2f43083f9e591d27c1d7b4555302c89d9d6e6e2f54e6be7cfd9cb0"
    )
    assert source.tensor_sha256 == (
        "a76519a832519103030cafe6646ecd1912212a4f9154a0f4a246277c516c9d5a"
    )
    assert source.report_sha256 == (
        "88394ae24648eca541a7b83ad48afec0681772bb231ff0fa5866ab75d74510ed"
    )
    assert source.code_bundle_sha256 == (
        "c88cd41db520e953c08b389df47e5befb6e3207c336e2dea79231e76ac4bed31"
    )
    assert source.d3_recipe_sha256 == (
        "853f09e644b2020cb90b056a06b594e7cc8af2f72c89a447e0a4988ebabb6c3b"
    )
    assert source.d3_primary_plan_sha256 == (
        "4591214359dd37a39c95f79419870d871814690e5dad546696d93da045813142"
    )
    assert source.d3_primary_result_sha256 == (
        "601bfbdddd91fe37364947a3703810bfb88db4043ac5ed0b1817e9b35d4950f1"
    )
    assert source.d3_fit_batch_sequence_sha256 == (
        "775da8a3506c7f7574aba6ee651cf472f45cbb35cf390d925f891bcb40bdcde5"
    )
    assert source.d3_fit_batch_content_sequence_sha256 == (
        "72e634d78bab6002b5bb88b58ca05cab03094ea97658679608fa06c96818cde0"
    )
    assert source.d3_fit_indexed_batch_sequence_sha256 == (
        "f80a057a292db7a15e3a5673e5c7d002ac3e5ccd74942c9a915f9700b004cdde"
    )
    assert source.d3_fit_endpoint_sequence_sha256 == (
        "ea3dcc77851607f3267343164fb9f4690f4a1c9e610a29099ed1668823a5cf04"
    )
    assert source.d3_fit_pair_sequence_sha256 == (
        "a3823fd453cc13770132cab9e57cc2b9159af16fa3d9bb913f043475361d62df"
    )
    assert source.d3_natural_pair_sequence_sha256 == (
        "26cadfb562be90f5719a95733d6cbdd23c64865605d864b122a1144e1e21282b"
    )
    assert source.d3_balanced_pair_sequence_sha256 == (
        "fb2234ff3342585dd1e0098241e6351fe5e30a2aa9edb2646ffa59a0b37fb379"
    )
    assert source.d3_source_replay_binding_sha256 == (
        "cb033006bbbeb6e2692bb9957d4b21b9fedeeefe03fc7c39f9fbe7fb10cc211b"
    )
    assert state["source_outcome"] == (
        "no_primary_treatment_passed_fit_gates"
    )
    assert state["source_d3_passed_all_fit_gates"] is False
    assert state["source_authorized_fresh_c3_recipe_id"] is None


def test_only_packed_rank_and_derived_core_widths_change() -> None:
    protocol = default_rank64_capacity_control_protocol()
    executor = protocol.executor
    state = protocol.state_dict()

    assert executor == Rank64ExecutorSpec()
    assert executor.input_modes == protocol.latent_rank + 2 == 66
    assert executor.output_modes == protocol.latent_rank == 64
    assert executor.expert_count == 2
    assert executor.expert_rank == 16
    assert executor.router_width == 16
    assert executor.same_position_skip is False
    assert executor.max_positive_lag is None
    assert executor.router_activation == "tanh"
    assert executor.source_normalized_routing is True
    assert state["controlled_change"] == (
        "packed_latent_rank_16_to_64_only_with_derived_core_"
        "input_18_to_66_and_output_16_to_64"
    )
    assert state["packing_semantics"] == (
        "learned_64_to_r_to_64_modal_packing"
    )


def test_authority_boundaries_separate_capacity_validity_and_c3() -> None:
    state = default_rank64_capacity_control_protocol().state_dict()

    assert state["c2_selection_role_allowed"] is False
    assert state["c2_provider_artifact_loading_allowed"] is False
    assert state["authenticated_source_result_artifact_loading_allowed"] is True
    assert state["source_result_loading_semantics"] == (
        "strict_hash_authenticated_source_safe_result_only"
    )
    assert state["primary_pass_authority"] == (
        "authorize_replication_seed_only"
    )
    assert state["valid_full_rank_fit_failure_authority"] == (
        "investigate_executor_objective_or_optimization_budget_"
        "at_full_outer_rank"
    )
    assert state["treatment_validity_failure_authority"] == (
        "invalidate_rank_comparison_no_capacity_conclusion"
    )
    assert state["two_seed_pass_authority"] == (
        "supports_capacity_conclusion_and_separate_compressed_width_"
        "ladder_preregistration_only"
    )
    assert state["fresh_c3_authorized"] is False
    assert state["compression_claim_authorized"] is False
    assert state["full_rank_parameter_reduction_claim_authorized"] is False


def test_protocol_and_nested_artifacts_round_trip_strictly() -> None:
    protocol = default_rank64_capacity_control_protocol()
    restored = Rank64CapacityControlProtocol.from_state_dict(
        protocol.state_dict()
    )

    assert restored == protocol
    assert restored.protocol_sha256 == protocol.protocol_sha256
    assert Rank64ExecutorSpec.from_state_dict(
        protocol.executor.state_dict()
    ) == protocol.executor
    assert MatchedD3TrainingSpec.from_state_dict(
        protocol.training.state_dict()
    ) == protocol.training
    assert ObjectiveBalanceResultBinding.from_state_dict(
        protocol.source_result.state_dict()
    ) == protocol.source_result


@pytest.mark.parametrize(
    "path,value",
    (
        (("artifact_sha256",), "0" * 64),
        (("latent_rank",), 63),
        (("execution_device",), "mps"),
        (("execution_dtype",), "bfloat16"),
        (("executor", "expert_rank"), 64),
        (("executor", "source_normalized_routing"), False),
        (("training", "steps"), 601),
        (("training", "sensitivity_direction_weight"), 0.5),
        (("source_result", "d3_primary_result_sha256"), "1" * 64),
        (("source_result", "d3_fit_pair_sequence_sha256"), "1" * 64),
        (("c2_selection_role_allowed",), True),
        (("fresh_c3_authorized",), True),
    ),
)
def test_protocol_rejects_tampering(
    path: tuple[object, ...],
    value: object,
) -> None:
    state = copy.deepcopy(
        default_rank64_capacity_control_protocol().state_dict()
    )
    target: object = state
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        Rank64CapacityControlProtocol.from_state_dict(state)


def test_protocol_rejects_extra_or_missing_state_keys() -> None:
    state = default_rank64_capacity_control_protocol().state_dict()
    state["extra"] = False
    with pytest.raises(ValueError, match="keys differ"):
        Rank64CapacityControlProtocol.from_state_dict(state)

    state = default_rank64_capacity_control_protocol().state_dict()
    del state["treatment_validity_failure_authority"]
    with pytest.raises(ValueError, match="keys differ"):
        Rank64CapacityControlProtocol.from_state_dict(state)


def test_default_factory_fails_if_literal_trust_anchor_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "DEFAULT_RANK64_CAPACITY_CONTROL_PROTOCOL_SHA256",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="trust anchor drifted"):
        protocol_module.default_rank64_capacity_control_protocol()


def test_protocol_module_uses_only_the_python_standard_library() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fisher_graph"
        / "gemma3_l3_l4_rank64_capacity_control_protocol.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {
        "__future__",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "re",
    }
