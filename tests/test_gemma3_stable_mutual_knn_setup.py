from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_stable_mutual_knn_setup import (
    BOOTSTRAP_CATALOG_DIGEST_DOMAIN,
    BOOTSTRAP_CATALOG_DIGEST_ENCODING,
    BOOTSTRAP_DRAW_DIGEST_SELECTION,
    BOOTSTRAP_DRAW_PREIMAGE_ENCODING,
    BOOTSTRAP_DRAW_PROTOCOL,
    BOOTSTRAP_DRAW_PROTOCOL_DOMAIN,
    BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY,
    BOOTSTRAP_RESAMPLE_COUNT,
    DEFAULT_OUTPUT,
    DISCOVERY_CANDIDATE_EDGE_COUNT,
    FRESH_EXAMPLE_COUNT,
    GEMMA3_STABLE_MUTUAL_KNN_SETUP_SCHEMA,
    GRAPH_NODE_COUNT,
    Gemma3StableMutualKNNSetup,
    LOFO_MINIMUM_MUTUAL_COUNT,
    LOGICAL_MODE_COUNT,
    LONG_VALID_TOKEN_MINIMUM,
    MAX_ABSOLUTE_WEIGHTED_RMSE,
    MAX_DIRECTIONAL_RELATIVE_SSE,
    MUTUAL_K,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    build_gemma3_stable_mutual_knn_setup,
    load_gemma3_stable_mutual_knn_setup,
    main,
    validate_gemma3_stable_mutual_knn_setup,
    write_gemma3_stable_mutual_knn_setup,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "src/fisher_graph/gemma3_stable_mutual_knn_setup.py"
_SHIM = _ROOT / "src/fisher_graph_stable_mutual_knn_setup.py"
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_setup_is_standard_library_only_import_inert_and_empty() -> None:
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
    setup = build_gemma3_stable_mutual_knn_setup()
    assert setup["schema"] == GEMMA3_STABLE_MUTUAL_KNN_SETUP_SCHEMA
    assert not any(setup["setup_receipt"].values())
    assert setup["claim_boundary"]["setup_only"] is True
    assert json.loads(json.dumps(setup)) == setup


def test_exact_predecessor_and_runtime_measurement_are_fully_pinned() -> None:
    predecessor = build_gemma3_stable_mutual_knn_setup()[
        "exact_predecessor"
    ]
    assert predecessor["typed_response_setup_sha256"] == (
        "189887111e12acd8e86bb56574ac69ebaf44660491a99a333ebcc75f7942de0f"
    )
    assert predecessor["typed_response_setup_file_sha256"] == (
        "a821b963c491b710784f4bf7745321208dfee8ba23201f4ece4f017d1be00aa0"
    )
    assert predecessor["typed_response_report_sha256"] == (
        "1ae47951899bf7f3ff4ddae4e3de69ce1bd5a798e2897ffe42931f3ed7005faa"
    )
    assert predecessor["typed_response_report_file_sha256"] == (
        "eeb4d735263309f9050c7ae3a61a49dfe9830e7b1d1ac23b47266a3cc3186996"
    )
    assert predecessor["measurement_input_sha256"] == (
        "cae68825f4d8f81e5fe106495bcab2eaa8f0f167ccc40f3fe9daf9487e7cad9f"
    )
    assert predecessor["response_ABI_sha256"] == (
        "a55b5fc1c50c7d435a8b26d22b0da81df32914685a2fff4f9b1ea69e6cd72367"
    )
    assert predecessor["runtime_output_Fisher_weight_catalog_sha256"] == (
        "b99b42a17ccd3dcf4082ba8a8dba4ff943c65c3ef7456dd8e65440ff50b1b1b0"
    )


def test_37_atomic_nodes_expand_to_38_logical_modes() -> None:
    ontology = build_gemma3_stable_mutual_knn_setup()["node_ontology"]
    nodes = ontology["ordered_node_catalog"]
    assert len(nodes) == GRAPH_NODE_COUNT == 37
    assert sum(node["member_count"] for node in nodes) == (
        LOGICAL_MODE_COUNT
    ) == 38
    lexical = [node for node in nodes if node["member_count"] == 2]
    assert len(lexical) == 1
    assert lexical[0]["member_mode_ids"] == [
        "mode.lexical.egress",
        "mode.lexical.ingress",
    ]
    assert ontology["tied_lexical_class_consumes_one_neighbor_slot"]
    assert ontology["predecessor_37_node_universe_is_immutable_for_fresh_graph"]
    assert ontology["fresh_tied_lexical_ingress_egress_exact_equality_must_be_reproved"]
    assert ontology["failed_fresh_tied_lexical_equality_invalidates_the_run"]
    assert ontology["new_fresh_exact_equalities_may_not_collapse_nodes_or_change_ranking_universe"]
    assert ontology["new_fresh_exact_equalities_are_diagnostic_only"]
    assert ontology["response_equivalence_does_not_imply_native_ABI_equivalence"]


def test_k_sweep_selects_sparse_middle_rung_before_fresh_data() -> None:
    sweep = build_gemma3_stable_mutual_knn_setup()["discovery_k_sweep"]
    assert sweep["selected_k"] == MUTUAL_K == 2
    assert sweep["selection_was_frozen_before_fresh_panel"]
    assert sweep["k_selection_may_not_be_retuned_on_fresh_data"]
    assert (
        sweep["k_1"]["edge_count"],
        sweep["k_1"]["covered_node_count"],
        sweep["k_1"]["isolated_node_count"],
        sweep["k_1"]["nonisolated_connected_component_sizes"],
    ) == (6, 12, 25, [2, 2, 2, 2, 2, 2])
    assert (
        sweep["k_2"]["edge_count"],
        sweep["k_2"]["covered_node_count"],
        sweep["k_2"]["isolated_node_count"],
        sweep["k_2"]["nonisolated_connected_component_sizes"],
    ) == (14, 20, 17, [6, 4, 3, 3, 2, 2])
    assert (
        sweep["k_3"]["edge_count"],
        sweep["k_3"]["covered_node_count"],
        sweep["k_3"]["isolated_node_count"],
        sweep["k_3"]["nonisolated_connected_component_sizes"],
    ) == (20, 24, 13, [11, 3, 3, 3, 2, 2])


def test_all_14_discovery_mutual_k2_edges_are_candidates() -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    graph = setup["discovery_candidate_graph"]
    assert graph["candidate_edge_count"] == DISCOVERY_CANDIDATE_EDGE_COUNT == 14
    assert graph["all_14_edges_retained_irrespective_of_discovery_stability_or_error_gates"]
    assert graph["no_discovery_edge_is_confirmed"]
    pairs = {
        frozenset((row["member_label_a"], row["member_label_b"]))
        for row in graph["candidate_edges"]
    }
    assert frozenset(("mode.mutation.14", "mode.mutation.16")) in pairs
    assert frozenset(("mode.mutation.9", "mode.mutation.6")) in pairs
    assert frozenset(("mode.lexical.egress", "mode.mutation.3")) in pairs
    assert sorted(
        [row["node_a_id"], row["node_b_id"]]
        for row in graph["candidate_edges"]
    ) == setup["discovery_k_sweep"]["k_2"]["edge_catalog"]


def test_discovery_candidate_stability_values_are_frozen_not_claimed() -> None:
    rows = build_gemma3_stable_mutual_knn_setup()[
        "discovery_candidate_graph"
    ]["candidate_edges"]
    by_members = {
        frozenset((row["member_label_a"], row["member_label_b"])): row
        for row in rows
    }
    strongest = by_members[
        frozenset(("mode.mutation.14", "mode.mutation.16"))
    ]
    assert strongest["discovery_bootstrap_mutual_count"] == 9_997
    assert strongest["discovery_bootstrap_mutual_frequency"] == 0.9997
    assert strongest["discovery_LOFO_mutual_count"] == 6
    lexical = by_members[
        frozenset(("mode.lexical.egress", "mode.mutation.3"))
    ]
    assert lexical["discovery_bootstrap_mutual_count"] == 9_155
    assert lexical["discovery_LOFO_mutual_count"] == 6
    assert all(row["discovery_only_not_confirmed"] for row in rows)


def test_ranking_and_tie_rules_are_label_free_and_fail_closed() -> None:
    ranking = build_gemma3_stable_mutual_knn_setup()["ranking_protocol"]
    assert ranking["k"] == 2
    assert ranking["rank_metric"] == "symmetric_distance"
    assert ranking["rank_boundary_tie_absolute_tolerance"] == 1.0e-10
    assert ranking["rank_boundary_tie_policy"] == (
        "fail_closed_undefined_node_neighbors"
    )
    assert ranking["mode_or_node_label_tie_break_forbidden"]
    assert ranking["equivalence_classes_collapsed_before_ranking"]


def test_confirmation_is_strict_conjunction_of_point_and_replication_gates() -> None:
    gates = build_gemma3_stable_mutual_knn_setup()[
        "fresh_confirmation_gates"
    ]
    error = gates["point_error_gates"]
    assert error["both_directions_required"]
    assert error["maximum_directional_relative_SSE"] == (
        MAX_DIRECTIONAL_RELATIVE_SSE
    ) == 1.0
    assert error["maximum_absolute_weighted_RMSE"] == (
        MAX_ABSOLUTE_WEIGHTED_RMSE
    ) == 0.125
    assert error["asymmetry_ratio_is_diagnostic_only"]
    bootstrap = gates["within_family_bootstrap"]
    assert bootstrap["resample_count"] == BOOTSTRAP_RESAMPLE_COUNT == 10_000
    assert bootstrap["minimum_mutual_frequency"] == (
        BOOTSTRAP_MINIMUM_MUTUAL_FREQUENCY
    ) == 0.80
    assert bootstrap["draw_protocol"] == BOOTSTRAP_DRAW_PROTOCOL
    assert bootstrap["draw_protocol_domain_hex"] == (
        BOOTSTRAP_DRAW_PROTOCOL_DOMAIN.hex()
    )
    assert bootstrap["draw_preimage_encoding"] == (
        BOOTSTRAP_DRAW_PREIMAGE_ENCODING
    )
    assert bootstrap["draw_digest_selection"] == (
        BOOTSTRAP_DRAW_DIGEST_SELECTION
    )
    assert bootstrap["draw_catalog_digest_domain_hex"] == (
        BOOTSTRAP_CATALOG_DIGEST_DOMAIN.hex()
    )
    assert bootstrap["draw_catalog_digest_encoding"] == (
        BOOTSTRAP_CATALOG_DIGEST_ENCODING
    )
    assert bootstrap["draw_catalog_iteration_order"] == (
        "resample_then_configured_family_then_draw"
    )
    lofo = gates["leave_one_family_out"]
    assert lofo["replicate_count"] == 6
    assert lofo["minimum_mutual_count"] == LOFO_MINIMUM_MUTUAL_COUNT == 5
    assert gates["long_context_robustness_veto"]["failure_vetoes_confirmation"]
    assert gates["partial_gate_pass_never_confirms_an_edge"]


def test_fresh_panel_is_family_disjoint_and_hits_exact_window_schedule() -> None:
    panel = build_gemma3_stable_mutual_knn_setup()["fresh_panel_contract"]
    assert panel["panel_not_authored_by_this_setup"]
    assert panel["manifest_schema"] == (
        "fisher_graph.gemma3_stable_mutual_knn_confirmation_panel"
    )
    assert panel["manifest_format_version"] == 1
    assert panel["panel_id"] == "gemma3_stable_mutual_knn_confirmation_v1"
    assert panel["corpus_schema"] == (
        "fisher_graph.gemma3_stable_mutual_knn_confirmation_corpus"
    )
    assert panel["corpus_format_version"] == 1
    assert panel["example_count"] == FRESH_EXAMPLE_COUNT == 60
    assert panel["examples_per_family"] == 10
    assert panel["fresh_semantic_families_are_disjoint_from_discovery_families"]
    assert panel["ordered_family_ids"] == [
        "C_confirmation.observational_diagnosis",
        "C_confirmation.quantitative_reasoning",
        "C_confirmation.technical_explanation",
        "C_confirmation.procedural_planning",
        "C_confirmation.formal_logic",
        "C_confirmation.causal_evidence_synthesis",
    ]
    assert panel["exact_valid_prompt_token_targets_per_family"] == [
        32,
        64,
        96,
        511,
        512,
        513,
        514,
        576,
        640,
        768,
    ]
    assert panel["context_strata"]["short"]["global_example_count"] == 18
    assert panel["context_strata"]["boundary"]["global_example_count"] == 24
    assert panel["context_strata"]["long"]["global_example_count"] == 18
    assert LONG_VALID_TOKEN_MINIMUM == 576
    assert panel["context_strata"]["boundary"][
        "token_514_is_diagnostic_only_and_carries_no_long_context_evidence_claim"
    ]
    assert panel["context_strata"]["long"][
        "full_minimal_sufficient_evidence_span_outside_final_512_token_window_required"
    ]
    reserve = panel["reserve_contract"]
    assert reserve["example_count"] == 12
    assert reserve["tokenizer_access_forbidden"]
    assert reserve["model_access_forbidden"]


def test_fresh_overlap_firewall_is_multilevel_and_pre_model() -> None:
    firewall = build_gemma3_stable_mutual_knn_setup()[
        "discovery_data_firewall"
    ]
    assert firewall["all_discovery_rows_spent"]
    assert firewall["required_zero_intersections"] == [
        "family_id",
        "text_sha256",
        "canonical_sha256",
        "instance_id",
        "template_id",
        "domain_template",
        "semantic_lineage_id",
        "paraphrase_lineage_id",
        "template_signature_sha256",
    ]
    assert firewall["instance_id_disjointness_is_the_paraphrase_firewall"]
    assert firewall["fresh_overlap_audit_receipt_required_before_tokenization"]
    assert firewall["fresh_token_count_and_long_evidence_audit_required_before_model_weights_load"]
    assert firewall["any_overlap_or_missing_lineage_metadata_is_fail_closed"]


def test_components_are_diagnostic_and_cliques_never_authorize_clusters() -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    extraction = setup["graph_extraction_and_reporting"]
    assert extraction["connected_components_role"] == (
        "diagnostic_topology_only"
    )
    assert extraction["connected_components_are_not_clusters"]
    assert extraction["maximal_cliques_role"] == (
        "stable_neighbor_groups_only"
    )
    assert extraction["maximal_cliques_may_overlap"]
    assert extraction["new_edges_cannot_enter_confirmatory_graph"]
    forbidden = setup["claim_taxonomy"][
        "never_authorized_by_this_protocol"
    ]
    assert all(forbidden.values())
    assert not setup["claim_boundary"]["confirmed_neighborhood_edge_claimed"]


def test_wrapper_validation_corruption_and_exclusive_write(tmp_path: Path) -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    wrapper = Gemma3StableMutualKNNSetup.build()
    assert wrapper.to_dict() == setup
    corrupted = deepcopy(setup)
    corrupted["ranking_protocol"]["k"] = 3
    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_stable_mutual_knn_setup(corrupted)
    target = tmp_path / "setup.json"
    assert write_gemma3_stable_mutual_knn_setup(target) == setup
    assert load_gemma3_stable_mutual_knn_setup(target) == setup
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_gemma3_stable_mutual_knn_setup(target)


def test_checked_in_artifact_and_cli_are_canonical_and_pinned() -> None:
    setup = build_gemma3_stable_mutual_knn_setup()
    assert load_gemma3_stable_mutual_knn_setup(_ARTIFACT) == setup
    assert setup["setup_sha256"] == PINNED_SETUP_SHA256
    assert _sha256_file(_ARTIFACT) == PINNED_SETUP_FILE_SHA256
    assert main(["--check", "--output", str(_ARTIFACT)]) == 0
