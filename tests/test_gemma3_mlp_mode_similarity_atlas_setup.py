from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_mlp_mode_similarity_atlas_setup import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ATLAS_CONFIG,
    CROSS_EXECUTION_COUNT,
    COMPARISON_ABI,
    DEFAULT_OUTPUT,
    DIRECTED_OFF_DIAGONAL_COUNT,
    FAMILY_DIRECTED_STAT_COUNT,
    FAMILY_EXAMPLE_DIRECTED_STAT_COUNT,
    FAMILY_SELF_REPLAY_STAT_COUNT,
    MLP_INVOCATION_IDS,
    MLP_MODE_IDS,
    ORDERED_FAMILY_IDS,
    PAIR_COUNT,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_PANEL_SHA256,
    PROBES_PER_MODE_EXAMPLE,
    RESIDUAL_PROFILE_RANKS,
    TOTAL_ORIGIN_PROBES,
    Gemma3MLPModeSimilarityAtlasSetup,
    build_gemma3_mlp_mode_similarity_atlas_setup,
    load_gemma3_mlp_mode_similarity_atlas_setup,
    main,
    validate_gemma3_mlp_mode_similarity_atlas_setup,
    write_gemma3_mlp_mode_similarity_atlas_setup,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "src/fisher_graph/gemma3_mlp_mode_similarity_atlas_setup.py"
_SHIM = _ROOT / "src/fisher_graph_mlp_mode_similarity_atlas_setup.py"
_ARTIFACT = _ROOT / DEFAULT_OUTPUT


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_setup_is_standard_library_only_and_receipt_is_empty() -> None:
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
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    assert not any(setup["setup_receipt"].values())
    assert setup["claim_boundary"]["setup_only"] is True
    assert json.loads(json.dumps(setup)) == setup


def test_model_panel_and_negative_predecessor_are_frozen() -> None:
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    assert setup["model_contract"]["model_id"] == PINNED_MODEL_ID == (
        "google/gemma-3-270m"
    )
    assert setup["model_contract"]["revision"] == PINNED_MODEL_REVISION == (
        "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
    )
    assert setup["discovery_data_firewall"]["panel_sha256"] == (
        PINNED_PANEL_SHA256
    )
    chain = setup["predecessor_chain"]
    assert chain["setup"]["setup_sha256"] == (
        "39d4debc38f680d8318bced5b9f553fe9ec8aa9866a990d9852c10e8f55059e8"
    )
    assert chain["report"]["report_sha256"] == (
        "11bec66d79aaf585c6dbfe835485c3b2aba15a2be5dcd10189bc9d26455f68b4"
    )
    assert chain["required_outcome"] == {
        "diagnostic_outcome": "shared_family_pair_mechanism_not_qualified",
        "dense_exact_control_passed_before_compressed_fit": True,
        "pair_mechanism_qualified": False,
        "B_guard_winner": None,
        "C_assessment_opened": False,
        "selected_pair_is_not_inherited_as_a_family": True,
    }
    assert chain["predecessor_pair_or_metric_may_seed_order_or_break_ties"] is False


def test_shared_mode_ontology_is_typed_and_mlp_is_only_provenance() -> None:
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    ontology = setup["mode_ontology"]
    assert setup["comparison_abi"] == COMPARISON_ABI == {
        "abi_name": "carrier_mutation.token_local.h640",
        "abi_version": 1,
        "input_signature": "raw_carrier_row.float.h640",
        "output_signature": "additive_full_atom_delta.float.h640",
        "context_signature": "token_local.no_external_context",
    }
    assert ontology["implementation_names_are_provenance_not_ontology"] is True
    assert ontology["layer_block_invocation_and_component_labels_are_not_features"]
    by_id = {entry["mode_abi_id"]: entry for entry in ontology["abi_catalog"]}
    assert by_id["abi.carrier_mutation.token_local.h640.v1"][
        "eligible_in_this_setup"
    ] is True
    assert by_id["abi.carrier_mutation.sequence_routed.h640.v1"][
        "eligible_in_this_setup"
    ] is False
    assert by_id["abi.lexical_boundary.vocab_h640.v1"][
        "eligible_in_this_setup"
    ] is False
    assert ontology["no_cross_ABI_adapter_is_present_here"] is True


def test_all_18_token_local_modes_and_all_153_pairs_are_in_scope() -> None:
    scope = build_gemma3_mlp_mode_similarity_atlas_setup()["scope_freeze"]
    assert len(MLP_MODE_IDS) == len(MLP_INVOCATION_IDS) == 18
    assert MLP_INVOCATION_IDS == tuple(f"mutation.{2 * i + 2}" for i in range(18))
    assert scope["ordered_mode_ids"] == list(MLP_MODE_IDS)
    assert [entry["source_invocation_id"] for entry in scope["mode_catalog"]] == (
        list(MLP_INVOCATION_IDS)
    )
    assert all(entry["cross_execution_eligible"] for entry in scope["mode_catalog"])
    assert {
        entry["generic_core_mode_kind"] for entry in scope["mode_catalog"]
    } == {"carrier_mutation.token_local.h640"}
    assert all(
        entry["provenance_fields_allowed_as_numerical_features"] is False
        for entry in scope["mode_catalog"]
    )
    assert PAIR_COUNT == scope["unordered_pair_count"] == 153
    assert DIRECTED_OFF_DIAGONAL_COUNT == 306
    assert scope["per_family_directed_cross_mode_stat_count"] == (
        FAMILY_DIRECTED_STAT_COUNT
    ) == 1_836
    assert scope["per_family_self_replay_stat_count"] == (
        FAMILY_SELF_REPLAY_STAT_COUNT
    ) == 108
    assert scope["per_family_example_directed_cross_mode_stat_count"] == (
        FAMILY_EXAMPLE_DIRECTED_STAT_COUNT
    ) == 18_360
    assert scope["nonadjacency_filter_present"] is False


def test_all_six_families_are_spent_descriptive_discovery_data() -> None:
    firewall = build_gemma3_mlp_mode_similarity_atlas_setup()[
        "discovery_data_firewall"
    ]
    assert firewall["ordered_family_ids"] == list(ORDERED_FAMILY_IDS)
    assert firewall["all_six_families_visible_to_descriptive_geometry"] is True
    assert firewall["leave_one_family_out_views_are_stability_analyses_not_holdouts"]
    examples = [
        example_id
        for family in firewall["family_manifests"]
        for example_id in family["ordered_example_ids"]
    ]
    assert len(examples) == len(set(examples)) == 60
    assert all(
        len(family["family_manifest_sha256"]) == 64
        for family in firewall["family_manifests"]
    )
    assert firewall["all_60_examples_are_spent_after_this_atlas"] is True
    requirements = firewall["every_future_merge_or_compression_protocol_requires"]
    assert "zero_example_id_overlap" in requirements
    assert "zero_exact_prompt_text_overlap" in requirements
    assert "zero_paraphrase_or_template_instance_overlap" in requirements


def test_top_two_fisher_probe_budget_and_cross_execution_are_exact() -> None:
    probes = build_gemma3_mlp_mode_similarity_atlas_setup()["probe_sampling"]
    assert PROBES_PER_MODE_EXAMPLE == 2
    assert probes["all_rows_seen_before_top_two_selection"] is True
    tokenization = probes["choice_tokenization_gate"]
    assert tokenization["choice_encoding"] == (
        "tokenize_one_leading_space_plus_choice"
    )
    assert tokenization["required_token_count_per_choice"] == 1
    assert tokenization["four_choice_token_ids_must_be_distinct_within_example"]
    assert tokenization["query_position"] == (
        "last_valid_nonpadding_prompt_position"
    )
    assert tokenization["loss_reduction_across_examples"] == "sum_not_mean"
    assert probes["Fisher_objective"].startswith(
        "sum_across_examples_of_gold_cross_entropy"
    )
    assert probes["probes_per_origin_mode_family"] == 20
    assert probes["probes_per_origin_mode"] == 120
    assert TOTAL_ORIGIN_PROBES == probes["total_origin_probes"] == 2_160
    assert CROSS_EXECUTION_COUNT == probes["full_atom_cross_executions"] == 38_880
    assert probes["foreign_atom_cross_executions"] == 36_720
    assert probes["origin_atom_self_replays"] == 2_160
    assert probes["self_replay_is_an_exact_runtime_gate_not_a_pair_distance"]
    parity = probes["self_replay_numerical_parity_gate"]
    assert parity["receipt_count"] == 108
    assert parity["maximum_absolute_delta_error"] == 1.0e-6
    assert parity["delta_NRMSE_maximum"] == 1.0e-7
    assert parity["claim"] == "same_operator_float32_numerical_parity"
    assert probes["cross_evaluate_all_18_exact_atoms_per_origin_probe"] is True
    assert probes["ordered_tie_break"][-1].startswith("fail_closed")


def test_primary_distance_is_diagonal_fisher_directed_then_symmetric() -> None:
    contract = build_gemma3_mlp_mode_similarity_atlas_setup()[
        "directed_distance_contract"
    ]
    assert contract["direction_definition"].startswith("i_to_j_uses_only_i_origin")
    assert contract["coordinate_weight"] == (
        "g_i_squared_diagonal_output_Fisher"
    )
    stats = contract["per_family_sufficient_statistics"]
    assert stats == {
        "weighted_reference_energy": "sum(g_i^2*Gi(x_i)^2)",
        "weighted_candidate_energy": "sum(g_i^2*Gj(x_i)^2)",
        "weighted_dot": "sum(g_i^2*Gi(x_i)*Gj(x_i))",
        "weighted_residual_SSE": (
            "reference_energy+candidate_energy-2*weighted_dot"
        ),
    }
    assert contract["per_family_relative_SSE"] == (
        "weighted_residual_SSE/max(weighted_reference_energy,"
        "atlas_config.energy_epsilon)"
    )
    assert contract["family_weighting"].startswith("equal_one_sixth")
    assert contract["primary_symmetric_distance"] == (
        "0.5*(d(i_to_j)+d(j_to_i))"
    )
    assert contract["distance_component_standardization_or_mixing_present"] is False


def test_cosine_raw_geometry_and_residual_rank_are_diagnostics_only() -> None:
    diagnostics = build_gemma3_mlp_mode_similarity_atlas_setup()[
        "diagnostic_geometry"
    ]
    assert diagnostics["diagnostics_never_change_primary_distance_or_order"]
    assert diagnostics["raw_unweighted_diagnostics"][
        "may_enter_primary_distance"
    ] is False
    profile = diagnostics["weighted_residual_rank_profile"]
    assert profile["rank_order"] == list(RESIDUAL_PROFILE_RANKS)
    assert profile["report_both_directions_separately"] is True
    assert profile["may_enter_primary_distance_or_seriation"] is False


def test_dense_fiedler_seriation_has_no_provenance_tie_break() -> None:
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    assert ATLAS_CONFIG == {
        "energy_epsilon": 1.0e-12,
        "sufficient_stat_relative_tolerance": 1.0e-7,
        "connectivity_tolerance": 1.0e-12,
        "fiedler_eigengap_relative_tolerance": 1.0e-8,
        "fiedler_coordinate_tie_tolerance": 1.0e-10,
        "nearest_neighbor_tie_tolerance": 1.0e-10,
        "affinity_max_exponent": 700.0,
    }
    assert {
        name: setup["atlas_config"][name] for name in ATLAS_CONFIG
    } == ATLAS_CONFIG
    assert setup["atlas_config"]["laplacian"] == "symmetric_normalized_v1"
    seriation = setup["affinity_and_seriation"]
    assert seriation["distance_scale"].startswith("median_of_all_153")
    assert seriation["sparsification_or_nearest_neighbor_graph_used"] is False
    assert seriation["laplacian"] == (
        "symmetric_normalized_I-D^(-1/2)*A*D^(-1/2)"
    )
    assert seriation["Fiedler_vector"] == "eigenvector_at_zero_based_index_1"
    assert len(seriation["fail_closed_conditions"]) == 5
    assert seriation["state_fingerprint_role"] == (
        "canonical_input_axis_and_global_reversal_only"
    )
    assert seriation[
        "state_fingerprint_never_enters_distance_affinity_or_numerical_"
        "tie_breaking"
    ]
    assert seriation["mode_ID_invocation_block_ordinal_or_source_label_tie_break"] is False
    assert seriation["numerical_ties_are_never_broken_by_provenance"] is True


def test_leave_one_family_out_and_bootstrap_only_form_hypotheses() -> None:
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    stability = setup["stability_contract"]
    loo = stability["leave_one_family_out"]
    assert loo["fold_count"] == 6
    assert loo["fold_order"] == list(ORDERED_FAMILY_IDS)
    assert loo["held_out_family_is_not_a_clean_assessment_role"] is True
    bootstrap = stability["within_family_bootstrap"]
    assert bootstrap["seed"] == BOOTSTRAP_SEED == 2_311
    assert bootstrap["resamples"] == BOOTSTRAP_RESAMPLES == 10_000
    assert bytes.fromhex(bootstrap["counter_hash_domain_hex"]).endswith(b"v1\0")
    assert bootstrap["family_ordinal_order"] == list(ORDERED_FAMILY_IDS)
    assert bootstrap["same_drawn_example_indices_shared_by_all_modes_and_pairs"]
    assert bootstrap["report_unordered_adjacency_frequency_for_all_153_pairs"]
    compact = setup["within_family_bootstrap_config"]
    assert compact["draw_protocol"] == (
        "sha256_counter_modulo_example_count_v1"
    )
    assert compact["draw_protocol_domain_hex"] == bootstrap[
        "counter_hash_domain_hex"
    ]
    assert compact["seed"] == 2_311
    assert compact["resample_count"] == 10_000
    assert compact["draws_per_family"] == 10
    assert compact["maximum_degenerate_fraction"] == 0.01
    assert [family["family_id"] for family in compact["families"]] == list(
        ORDERED_FAMILY_IDS
    )
    assert all(len(family["example_cluster_ids"]) == 10 for family in compact["families"])
    hypothesis = stability["stable_neighbor_hypothesis_rule"]
    assert hypothesis["leave_one_family_out_adjacency_count_minimum"] == 5
    assert hypothesis["bootstrap_adjacency_frequency_minimum"] == 0.80
    assert hypothesis["does_not_establish_a_generator_family_or_merge"] is True
    assert stability["absence_of_stable_neighbors_is_a_valid_result"] is True


def test_claim_boundary_forbids_compression_edges_and_family_claims() -> None:
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    scope = setup["scope_freeze"]
    assert scope["generator_shared_merged_deleted_or_replaced"] is False
    assert scope["compression_or_low_rank_fit_performed"] is False
    assert scope["carrier_edge_changed_or_learned"] is False
    claims = setup["claim_boundary"]
    assert claims["descriptive_exact_geometry_only"] is True
    assert claims["shared_generator_family_or_merge_claimed"] is False
    assert claims["causal_or_functional_interchangeability_claimed"] is False
    assert claims["stable_neighbors_are_hypotheses_requiring_fresh_data"] is True


def test_generic_core_configs_parse_without_metadata_leaking_into_dataclass() -> None:
    from fisher_graph.mode_similarity_atlas import (
        ModeABI,
        parse_atlas_config_payload,
        parse_within_family_bootstrap_config_payload,
    )

    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    atlas_config = parse_atlas_config_payload(setup["atlas_config"])
    assert atlas_config.energy_epsilon == 1.0e-12
    assert atlas_config.canonical_payload()["laplacian"] == (
        "symmetric_normalized_v1"
    )
    abi = ModeABI(**setup["comparison_abi"])
    assert len(abi.sha256) == 64
    bootstrap = parse_within_family_bootstrap_config_payload(
        setup["within_family_bootstrap_config"]
    )
    assert bootstrap.seed == 2_311
    assert bootstrap.resample_count == 10_000
    assert bootstrap.canonical_payload() == setup[
        "within_family_bootstrap_config"
    ]


def test_validation_wrapper_file_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    setup = build_gemma3_mlp_mode_similarity_atlas_setup()
    validate_gemma3_mlp_mode_similarity_atlas_setup(setup)
    assert Gemma3MLPModeSimilarityAtlasSetup.build().to_dict() == setup

    destination = tmp_path / "setup.json"
    assert write_gemma3_mlp_mode_similarity_atlas_setup(destination) == setup
    assert load_gemma3_mlp_mode_similarity_atlas_setup(destination) == setup
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_gemma3_mlp_mode_similarity_atlas_setup(destination)

    tampered = deepcopy(setup)
    tampered["scope_freeze"]["carrier_edge_changed_or_learned"] = True
    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_mlp_mode_similarity_atlas_setup(tampered)
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_gemma3_mlp_mode_similarity_atlas_setup([])  # type: ignore[arg-type]


def test_checked_in_artifact_and_both_cli_surfaces_are_canonical(
    tmp_path: Path,
) -> None:
    expected = build_gemma3_mlp_mode_similarity_atlas_setup()
    assert load_gemma3_mlp_mode_similarity_atlas_setup(_ARTIFACT) == expected
    assert main(["--check", "--output", str(_ARTIFACT)]) == 0

    direct_output = tmp_path / "direct.json"
    shim_output = tmp_path / "shim.json"
    assert main(["--output", str(direct_output)]) == 0
    assert load_gemma3_mlp_mode_similarity_atlas_setup(direct_output) == expected

    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            str(_SHIM),
            "--output",
            str(shim_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert load_gemma3_mlp_mode_similarity_atlas_setup(shim_output) == expected
