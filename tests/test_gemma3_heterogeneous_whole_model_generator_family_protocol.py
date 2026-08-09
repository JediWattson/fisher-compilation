from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_heterogeneous_whole_model_generator_family_protocol import (
    GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION,
    GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA,
    PINNED_EDGE_OCCURRENCE_COUNT,
    PINNED_EVENT_COUNT,
    PINNED_HETEROGENEOUS_ATOM_COUNT,
    PINNED_MAXIMUM_TARGET_CANDIDATE_SCALARS,
    PINNED_MINIMUM_TARGET_REMOVED_SCALARS,
    PINNED_MUTATION_COUNT,
    PINNED_SOURCE_SCALARS,
    build_gemma3_heterogeneous_generator_family_protocol,
    load_gemma3_heterogeneous_generator_family_protocol,
    validate_gemma3_heterogeneous_generator_family_protocol,
    write_gemma3_heterogeneous_generator_family_protocol,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / (
    "src/fisher_graph/"
    "gemma3_heterogeneous_whole_model_generator_family_protocol.py"
)
_SHIM = _ROOT / "src/fisher_graph_heterogeneous_generator_family_setup.py"
_ARTIFACT = _ROOT / (
    "artifacts/research/"
    "gemma3-heterogeneous-whole-model-generator-family-protocol-v1.json"
)


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_setup_is_standard_library_only_and_performs_no_run() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    assert protocol["schema"] == (
        GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_SCHEMA
    )
    assert protocol["format_version"] == (
        GEMMA3_HETEROGENEOUS_GENERATOR_FAMILY_PROTOCOL_FORMAT_VERSION
    )
    assert not any(protocol["setup_receipt"].values())
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "runpy",
        "typing",
    }
    assert _import_roots(_MODULE) <= allowed
    assert _import_roots(_SHIM) <= allowed
    assert json.loads(json.dumps(protocol)) == protocol


def test_exact_lift_arm_order_is_frozen_before_compression() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    expected = [
        "exact.causal_carrier",
        "exact.cross_block_mlp_family_lift",
        "exact.compatible_attention_family_lift",
        "exact.heterogeneous_full_rank_atom_bank_lift",
    ]
    gate = protocol["exact_lift_gate_chain"]
    assert gate["exact_arm_order"] == expected
    assert protocol["arm_order"][:4] == expected
    assert protocol["arm_order"][4:6] == [
        "primary.shared_family_core_residual_adapter_ladder",
        "primary.heterogeneous_shared_hidden_atom_bank_ladder",
    ]
    assert all(
        arm["candidate_stored_scalars"] == PINNED_SOURCE_SCALARS
        for arm in protocol["arms"][:4]
    )
    assert gate["all_exact_arms_must_pass_before_any_compression_fit"] is True
    assert gate["maximum_absolute_logit_error"] == 0.0
    assert gate["native_to_candidate_KL_per_query"] == 0.0
    assert gate["top1_agreement"] == 1.0
    assert gate["all_75_ordered_edge_occurrence_hashes_exact"] is True


def test_full_invocation_and_edge_occurrence_inventory_closes() -> None:
    graph = build_gemma3_heterogeneous_generator_family_protocol()[
        "execution_graph"
    ]
    assert graph["invocation_count"] == PINNED_EVENT_COUNT == 39
    assert graph["body_invocation_count"] == PINNED_MUTATION_COUNT == 36
    assert graph["edge_occurrence_count"] == PINNED_EDGE_OCCURRENCE_COUNT == 75
    assert graph["lexical_edge_occurrence_count"] == 2
    assert graph["identity_carry_occurrence_count"] == 36
    assert graph["primary_generator_edge_occurrence_count"] == 36
    assert graph["terminal_edge_occurrence_count"] == 1
    assert graph["learned_correction_edge_count"] == 0
    assert graph["optional_learned_correction_edges_allowed"] is False

    invocations = graph["invocations"]
    assert [row["invocation_ordinal"] for row in invocations] == list(range(39))
    assert len({row["invocation_id"] for row in invocations}) == 39
    assert len({row["invocation_adapter_id"] for row in invocations}) == 39
    assert len({row["atom_id"] for row in invocations}) == 38
    assert len({row["family_id"] for row in invocations if row["family_id"]}) == 4

    ingress, terminal, egress = invocations[0], invocations[-2], invocations[-1]
    assert ingress["invocation_id"] == "lexical.ingress"
    assert egress["invocation_id"] == "lexical.egress"
    assert ingress["atom_id"] == egress["atom_id"] == "atom.lexical.tied"
    assert terminal["family_id"] is None
    assert terminal["atom_id"] == "atom.terminal.RMSNorm"

    body = [row for row in invocations if row["event_kind"] == "body_mutation"]
    assert len({row["atom_id"] for row in body}) == 36
    occurrences = graph["edge_occurrences"]
    assert len({row["edge_occurrence_id"] for row in occurrences}) == 75
    kinds = [row["occurrence_kind"] for row in occurrences]
    assert kinds.count("identity_carry") == 36
    assert kinds.count("primary_generator") == 36
    assert not any("correction" in kind for kind in kinds)
    assert all(
        row["stored_scalars"] == 0
        for row in occurrences
        if row["occurrence_kind"] == "identity_carry"
    )


def test_four_typed_family_banks_include_tied_lexical_and_close_inventory() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    contract = protocol["family_and_atom_bank_contract"]
    assert contract["family_count"] == 4
    assert [row["member_invocation_count"] for row in contract["families"]] == [
        2,
        18,
        15,
        3,
    ]
    assert [row["full_rank_physical_atom_count"] for row in contract["families"]] == [
        1,
        18,
        15,
        3,
    ]
    lexical = contract["families"][0]
    assert lexical["family_id"] == "family.lexical.tied"
    assert lexical["exact_stored_scalars"] == 167_772_160
    assert lexical["tied_physical_storage_alias_required"] is True

    bank = contract["heterogeneous_atom_bank"]
    assert bank["typed_subbank_count"] == 4
    assert bank["physical_atom_count"] == PINNED_HETEROGENEOUS_ATOM_COUNT == 37
    assert bank["lexical_physical_atom_count"] == 1
    assert bank["body_physical_atom_count"] == 36
    assert bank["stored_scalars"] == 268_097_536
    assert bank["terminal_exact_atom_is_outside_bank"] is True
    assert bank["incompatible_atom_shapes_are_never_flattened_or_averaged"]

    resources = protocol["resource_accounting"]
    assert resources["source_inventory_closes"] is True
    assert resources["exact_family_bank_plus_terminal_closes"] is True
    assert sum(resources["source_pools"].values()) == PINNED_SOURCE_SCALARS


def test_shared_family_primary_ladder_counts_core_once_plus_residuals() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    ladders = protocol["compression_ladders"]
    assert ladders["primary_ladder_order"] == [
        "primary.shared_family_core_residual_adapter_ladder",
        "primary.heterogeneous_shared_hidden_atom_bank_ladder",
    ]
    shared = ladders["shared_family_core_residual_adapters"]
    assert shared["phase_order"] == [
        "one_nonadjacent_MLP_pair",
        "compatible_attention_pairs_separately",
        "family_size_growth",
        "disjoint_full_body_family_joint",
    ]
    assert shared["full_rank_control_is_dense_not_low_rank_r640"] is True

    mlp_pair = shared["one_nonadjacent_MLP_pair"]["ordered_points"]
    assert [point["residual_rank"] for point in mlp_pair] == [
        None,
        256,
        128,
        64,
        32,
    ]
    assert [point["candidate_stored_scalars"] for point in mlp_pair] == [
        268_098_176,
        266_230_400,
        265_198_208,
        264_682_112,
        264_424_064,
    ]
    assert all(point["shared_core_count"] == 1 for point in mlp_pair)
    assert all(point["residual_adapter_count"] == 1 for point in mlp_pair)

    attention_pairs = shared["compatible_attention_pairs_separately"]
    assert len(attention_pairs) == 2
    expected_attention_candidates = [
        268_098_176,
        267_442_816,
        266_951_296,
        266_705_536,
        266_582_656,
    ]
    for pair in attention_pairs:
        assert [
            point["candidate_stored_scalars"]
            for point in pair["ordered_points"]
        ] == expected_attention_candidates

    full_body = shared["disjoint_full_body_family_joint"]
    assert full_body["duplicate_invocation_membership_allowed"] is False
    assert [
        point["candidate_stored_scalars"]
        for point in full_body["ordered_points"]
    ] == [
        268_098_176,
        225_860_224,
        200_448_640,
        187_742_848,
        181_389_952,
    ]
    assert [
        point["meets_twenty_percent_resource_target"]
        for point in full_body["ordered_points"]
    ] == [False, False, True, True, True]
    assert all(
        point["learned_correction_edge_count"] == 0
        for point in full_body["ordered_points"]
    )


def test_family_growth_and_assignment_are_frozen_without_ordinal_features() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    assignment = protocol["family_and_atom_bank_contract"][
        "family_assignment_contract"
    ]
    assert assignment["compressed_members_and_shared_core_anchor_selected_on"] == (
        "A_fit"
    )
    assert assignment["selection_uses_activation_and_Fisher_statistics_only"]
    assert assignment["block_ordinal_used_as_numerical_feature"] is False
    assert assignment["layer_distance_used_as_numerical_feature"] is False
    assert assignment["family_assignment_frozen_before_A_select"] is True
    assert assignment["duplicate_invocation_membership_allowed"] is False

    growth = protocol["compression_ladders"][
        "shared_family_core_residual_adapters"
    ]["family_size_growth"]
    curves = growth["ordered_family_curves"]
    assert [row["ordered_growth_member_counts"] for row in curves] == [
        [4, 8, 12, 18],
        [4, 8, 15],
        [3],
    ]
    mlp_full = curves[0]["ordered_curves"][-1]["ordered_points"]
    assert [point["candidate_stored_scalars"] for point in mlp_full] == [
        268_098_176,
        236_345_984,
        218_798_720,
        210_025_088,
        205_638_272,
    ]


def test_heterogeneous_hidden_bank_arithmetic_and_target_shortfall_are_frozen() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    resources = protocol["resource_accounting"]
    assert resources["maximum_candidate_scalars_to_meet_target"] == (
        PINNED_MAXIMUM_TARGET_CANDIDATE_SCALARS
    )
    assert resources["minimum_removed_scalars_to_meet_target"] == (
        PINNED_MINIMUM_TARGET_REMOVED_SCALARS
    )
    ladder = protocol["compression_ladders"][
        "heterogeneous_shared_hidden_atom_bank"
    ]
    assert ladder["typed_subbank_count"] == 4
    assert ladder["rank_order"] == [640, 608, 576, 544, 512, 496]
    points = ladder["ordered_points"]
    assert [point["candidate_stored_scalars"] for point in points] == [
        268_098_176,
        255_085_184,
        241_662_592,
        228_240_000,
        214_817_408,
        208_106_112,
    ]
    assert [point["removed_scalars"] for point in points] == [
        0,
        13_012_992,
        26_435_584,
        39_858_176,
        53_280_768,
        59_992_064,
    ]
    assert [
        point["meets_twenty_percent_resource_target"] for point in points
    ] == [False, False, False, False, False, True]
    assert ladder["rank_512_misses_twenty_percent_target_by_scalars"] == 338_868
    assert ladder["largest_integer_rank_meeting_target_under_formula"] == 511
    assert ladder["rank_511_is_not_a_preregistered_execution_point"] is True
    assert ladder["first_preregistered_point_meeting_target"] == 496
    assert points[0]["shared_hidden_basis_stored_scalars"] == 0
    assert points[-2]["shared_hidden_basis_stored_scalars"] == 327_680
    assert points[-1]["shared_hidden_basis_stored_scalars"] == 317_440
    assert all(point["learned_correction_edge_count"] == 0 for point in points)


def test_old_factorizations_are_controls_and_transports_are_excluded() -> None:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    ladders = protocol["compression_ladders"]
    controls = ladders["matched_controls"]
    assert controls["control_only"] is True
    assert [
        point["candidate_stored_scalars"] for point in controls["ordered_points"]
    ] == [
        234_871_424,
        229_751_120,
        214_071_632,
        213_342_544,
        204_933_456,
        196_524_368,
    ]
    assert all(point["control_only"] for point in controls["ordered_points"])
    assert all(
        point["recommended_primary"] is False
        for point in controls["ordered_points"]
    )
    assert all(
        point["learned_correction_edge_count"] == 0
        for point in controls["ordered_points"]
    )
    prior = ladders["excluded_historical_negative_prior"]
    assert prior["learned_correction_edge_count"] == 136
    assert prior["recommended_primary"] is False
    assert prior["execution_allowed_under_this_protocol"] is False
    assert prior["historical_metric_values_imported_into_protocol"] is False
    assert ladders["learned_correction_edge_count_ladder"] == [0]
    overhead = protocol["resource_accounting"][
        "chart_and_transport_overhead_policy"
    ]
    assert overhead["primary_chart_count"] == 0
    assert overhead["primary_learned_transport_count"] == 0
    assert overhead["adding_either_requires_new_protocol"] is True


def test_fit_select_guard_assessment_firewall_is_one_way() -> None:
    firewall = build_gemma3_heterogeneous_generator_family_protocol()[
        "fit_selection_guard_assessment_firewall"
    ]
    assert firewall["required_manifests"] == [
        "A_fit",
        "A_select",
        "B_guard",
        "C_assessment",
    ]
    assert firewall["prompt_example_and_family_overlap_across_roles_allowed"] is False
    assert firewall["A_fit_may_open_A_select_B_guard_or_C_assessment"] is False
    assert firewall["A_fit_may_use_block_ordinal_or_layer_distance_as_feature"] is False
    assert firewall["family_assignment_manifest_frozen_and_hashed_before_A_select"]
    assert firewall["A_select_may_change_family_membership_or_shared_core_anchor"] is False
    assert firewall["A_select_may_refit_change_topology_or_change_resource_points"] is False
    assert firewall["pareto_frontier_frozen_before_B_guard"] is True
    assert firewall["B_guard_opened_once"] is True
    assert firewall["B_guard_may_refit_reorder_prune_or_change_thresholds"] is False
    assert firewall["B_guard_may_change_family_assignment"] is False
    assert firewall["C_assessment_candidate_count"] == 1
    assert firewall["C_assessment_opened_only_after_guard_pass"] is True
    assert firewall["C_assessment_may_refit_select_reorder_or_change_thresholds"] is False


def test_strict_validator_rejects_rehashed_semantic_changes() -> None:
    original = build_gemma3_heterogeneous_generator_family_protocol()
    mutations = (
        ("execution_graph", "optional_learned_correction_edges_allowed", True),
        ("exact_lift_gate_chain", "maximum_absolute_logit_error", 1.0e-6),
        ("resource_accounting", "minimum_removed_scalars_to_meet_target", 0),
        (
            "fit_selection_guard_assessment_firewall",
            "C_assessment_candidate_count",
            2,
        ),
        (
            "claim_firewall",
            "no_compression_or_speed_claim_authorized_by_this_setup",
            False,
        ),
    )
    for section, key, value in mutations:
        changed = deepcopy(original)
        changed[section][key] = value
        changed["protocol_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="not canonical"):
            validate_gemma3_heterogeneous_generator_family_protocol(changed)


def test_writer_is_exclusive_and_artifact_matches_builder(tmp_path: Path) -> None:
    destination = tmp_path / "protocol.json"
    written = write_gemma3_heterogeneous_generator_family_protocol(destination)
    assert load_gemma3_heterogeneous_generator_family_protocol(destination) == written
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_gemma3_heterogeneous_generator_family_protocol(destination)
    assert load_gemma3_heterogeneous_generator_family_protocol(_ARTIFACT) == (
        build_gemma3_heterogeneous_generator_family_protocol()
    )
