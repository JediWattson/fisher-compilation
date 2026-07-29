from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_objective_balance_diagnostic_protocol as protocol_module,
)
from fisher_graph.gemma3_l3_l4_objective_balance_diagnostic_protocol import (
    DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256,
    DIAGNOSTIC_EXECUTION_DEVICE,
    DIAGNOSTIC_EXECUTION_DTYPE,
    DIAGNOSTIC_LATENT_RANK,
    DIAGNOSTIC_LEARNING_RATE,
    DIAGNOSTIC_PRIMARY_SEED,
    DIAGNOSTIC_RECIPE_STEPS,
    DIAGNOSTIC_REPLICATION_SEED,
    OBJECTIVE_BALANCE_RECIPE_IDS,
    C2ObjectiveBalanceProvenance,
    ObjectiveBalanceDiagnosticGates,
    ObjectiveBalanceDiagnosticProtocol,
    ObjectiveBalanceRecipe,
    default_objective_balance_diagnostic_protocol,
    family_balance_copy_binding_sha256,
    family_balance_copy_id,
)


def test_default_protocol_authenticates_literal_fit_only_declaration() -> None:
    protocol = default_objective_balance_diagnostic_protocol()

    assert protocol.protocol_sha256 == (
        DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
    )
    assert protocol.protocol_sha256 == (
        "d502d003fd86f6ef7322e35854d0bb738fdc4cfa6fa5089c2812e366a142d2eb"
    )
    assert protocol.latent_rank == DIAGNOSTIC_LATENT_RANK == 16
    assert protocol.execution_device == DIAGNOSTIC_EXECUTION_DEVICE == "cpu"
    assert protocol.execution_dtype == DIAGNOSTIC_EXECUTION_DTYPE == "float32"
    assert protocol.primary_seed == (
        DIAGNOSTIC_PRIMARY_SEED
    ) == 20_260_728_402
    assert protocol.replication_seed == (
        DIAGNOSTIC_REPLICATION_SEED
    ) == 20_260_729_402
    state = protocol.state_dict()
    assert state["scientific_scope"] == (
        "fit_only_rank16_optimization_diagnostic_not_generalization"
    )
    assert state["schema"] == (
        "fisher_graph.gemma3_l3_l4_objective_balance_diagnostic."
        "d0_d3_protocol.v1"
    )
    assert state["materialization_roles_allowed"] == ["pilot", "fit"]
    assert state["selection_role_allowed"] is False
    assert state["fresh_c3_required"] is True
    assert state["success_authority"] == (
        "may_lock_recipe_for_fresh_c3_only"
    )
    assert state["d3_treatment_semantics"] == (
        "composite_direction_weight_and_doubled_steps_"
        "no_component_attribution"
    )
    assert state["fit_data_binding_semantics"] == (
        "exact_c2_rank16_raw_control_binding_reused_for_all_recipes_"
        "to_freeze_batch_order"
    )


def test_recipe_ladder_freezes_d0_through_d3() -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    recipes = protocol.recipes

    assert tuple(value.recipe_id for value in recipes) == (
        OBJECTIVE_BALANCE_RECIPE_IDS
    )
    assert tuple(value.training_metric for value in recipes) == (
        "canonical_fisher",
        "fit_teacher_weighted_rms",
        "fit_teacher_weighted_rms",
        "fit_teacher_weighted_rms",
    )
    assert tuple(value.signed_pair_multiplicity for value in recipes) == (
        1,
        1,
        2,
        2,
    )
    assert tuple(value.direction_weight for value in recipes) == (
        0.5,
        0.5,
        0.5,
        2.0,
    )
    assert tuple(value.steps for value in recipes) == (
        DIAGNOSTIC_RECIPE_STEPS
    ) == (300, 300, 300, 600)
    assert {value.learning_rate for value in recipes} == {
        DIAGNOSTIC_LEARNING_RATE
    } == {1e-3}
    assert {value.primary_seed for value in recipes} == {
        DIAGNOSTIC_PRIMARY_SEED
    }
    assert recipes[0].advancement_eligible is False
    assert all(value.advancement_eligible for value in recipes[1:])
    assert protocol.advancement_recipe_ids == OBJECTIVE_BALANCE_RECIPE_IDS[1:]
    assert protocol.recipe("d2_unit_rms_family_balanced") is recipes[2]
    with pytest.raises(KeyError):
        protocol.recipe("missing")


def test_balance_and_advancement_gates_are_frozen() -> None:
    gates = default_objective_balance_diagnostic_protocol().gates

    assert gates.minimum_gauge_energy == 1e-12
    assert gates.normalized_energy_absolute_tolerance == 1e-12
    assert gates.minimum_teacher_mse_floor_multiple == 100.0
    assert gates.minimum_initial_pointwise_share == 0.10
    assert gates.maximum_initial_pointwise_share == 0.40
    assert gates.minimum_initial_contrast_share == 0.50
    assert gates.maximum_initial_active_component_share == 0.65
    assert gates.required_ordinary_gate_count == 12
    assert gates.required_null_candidate_pass_count == 24
    assert gates.require_all_ordinary_gates is True
    assert gates.require_every_eligible_sensitivity_contrast is True
    assert gates.require_all_contrast_families is True
    assert gates.require_two_seed_pass is True
    assert gates.ordinary_gates_sha256 == (
        "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
    )
    assert gates.contrast_gates_sha256 == (
        "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
    )


def test_c2_provenance_denies_both_selection_panel_bindings() -> None:
    provenance = (
        default_objective_balance_diagnostic_protocol().c2_provenance
    )

    assert provenance.protocol_sha256 == (
        "033020dc9a0da819bd5753eb10090bff1bd9b4fcf61f33cd7186b1c1e3cb5254"
    )
    assert provenance.pilot_panel_sha256 == (
        "2b4d4efc8dbbdbc1c6afdb1a3134068f31b39ebc65e8a226fbbb9c92fe07b5e3"
    )
    assert provenance.fit_panel_sha256 == (
        "e6244896f2f6823f61d5ac57fad0756be320961e0b2272612579f9ba2f5b3e75"
    )
    assert provenance.calibrated_fit_panel_sha256 == (
        "f3175e8a08fe368614763c13e3a3eb89629b613ca7ca1a49710b44b227854fdd"
    )
    assert provenance.calibration_sha256 == (
        "aedb23de65ed6a37d645539001311ddb415cd2713400777dac448cb96bd5bfa8"
    )
    assert provenance.selected_amplitude == 8.0
    assert provenance.forbidden_selection_panel_sha256s == (
        "4e86a5acf6fcaa4a1e03b61e387938fb442f11e2d82aee73b56d1ee0189013f6",
        "2c991c815d2f18d5b410fb296a6a9659d0a594c6abc93daee2ec21ef972a6aec",
    )
    assert provenance.forbidden_probe_prefixes == (
        "development_c2.selection.",
    )
    assert provenance.selection_materialization_allowed is False
    assert provenance.c2_artifact_loading_allowed is False


def test_protocol_and_nested_artifacts_round_trip_strictly() -> None:
    protocol = default_objective_balance_diagnostic_protocol()
    restored = ObjectiveBalanceDiagnosticProtocol.from_state_dict(
        protocol.state_dict()
    )

    assert restored == protocol
    assert restored.protocol_sha256 == protocol.protocol_sha256
    assert ObjectiveBalanceDiagnosticGates.from_state_dict(
        protocol.gates.state_dict()
    ) == protocol.gates
    assert C2ObjectiveBalanceProvenance.from_state_dict(
        protocol.c2_provenance.state_dict()
    ) == protocol.c2_provenance
    assert tuple(
        ObjectiveBalanceRecipe.from_state_dict(value.state_dict())
        for value in protocol.recipes
    ) == protocol.recipes


@pytest.mark.parametrize(
    "path,value",
    (
        (("artifact_sha256",), "0" * 64),
        (("selection_role_allowed",), True),
        (("execution_device",), "mps"),
        (("execution_dtype",), "bfloat16"),
        (("latent_rank",), 8),
        (("recipes", 1, "steps"), 301),
        (
            ("gates", "maximum_initial_pointwise_share"),
            0.9,
        ),
        (
            (
                "c2_provenance",
                "forbidden_selection_panel_sha256s",
                1,
            ),
            "1" * 64,
        ),
    ),
)
def test_protocol_rejects_tampering(
    path: tuple[object, ...],
    value: object,
) -> None:
    state = copy.deepcopy(
        default_objective_balance_diagnostic_protocol().state_dict()
    )
    target: object = state
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        ObjectiveBalanceDiagnosticProtocol.from_state_dict(state)


def test_protocol_rejects_extra_or_missing_state_keys() -> None:
    state = default_objective_balance_diagnostic_protocol().state_dict()
    state["extra"] = False
    with pytest.raises(ValueError, match="keys differ"):
        ObjectiveBalanceDiagnosticProtocol.from_state_dict(state)

    state = default_objective_balance_diagnostic_protocol().state_dict()
    del state["fresh_c3_required"]
    with pytest.raises(ValueError, match="keys differ"):
        ObjectiveBalanceDiagnosticProtocol.from_state_dict(state)


def test_frozen_recipe_and_gate_construction_fails_closed() -> None:
    with pytest.raises(ValueError, match="frozen value"):
        ObjectiveBalanceRecipe(
            recipe_id="d1_unit_rms",
            training_metric="canonical_fisher",
            signed_pair_multiplicity=1,
            pointwise_weight=1.0,
            sensitivity_relative_delta_weight=2.0,
            direction_weight=0.5,
            midpoint_jvp_weight=1.0,
            intended_null_weight=1.0,
            steps=300,
            learning_rate=1e-3,
            primary_seed=DIAGNOSTIC_PRIMARY_SEED,
            advancement_eligible=True,
        )
    with pytest.raises(ValueError, match="interval"):
        ObjectiveBalanceDiagnosticGates(
            minimum_initial_pointwise_share=0.8,
            maximum_initial_pointwise_share=0.2,
        )


def test_signed_pair_copy_identity_is_deterministic_and_authenticated() -> None:
    pair_sha = "a" * 64
    binding = family_balance_copy_binding_sha256(
        "development_c2.fit.signed.pair.00",
        pair_sha,
    )

    assert len(binding) == 64
    assert family_balance_copy_id(
        "development_c2.fit.signed.pair.00",
        pair_sha,
    ) == f"objective_balance_d1.signed_copy.{binding}"
    assert binding != family_balance_copy_binding_sha256(
        "development_c2.fit.signed.pair.01",
        pair_sha,
    )
    with pytest.raises(ValueError):
        family_balance_copy_id("", pair_sha)
    with pytest.raises(ValueError):
        family_balance_copy_binding_sha256("pair", "A" * 64)


def test_default_factory_fails_if_literal_trust_anchor_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol_module,
        "DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="trust anchor drifted"):
        protocol_module.default_objective_balance_diagnostic_protocol()


def test_protocol_module_uses_only_the_python_standard_library() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fisher_graph"
        / "gemma3_l3_l4_objective_balance_diagnostic_protocol.py"
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
        "math",
        "re",
        "typing",
    }
