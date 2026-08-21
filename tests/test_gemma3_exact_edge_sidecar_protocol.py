from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_exact_edge_sidecar_protocol import (
    EXACT_EDGE_SIDECAR_PROTOCOL_FORMAT_VERSION,
    EXACT_EDGE_SIDECAR_PROTOCOL_SCHEMA,
    PINNED_CHART_SITE_COUNT,
    PINNED_EVENT_COUNT,
    PINNED_HIDDEN_WIDTH,
    PINNED_PANEL_PATH,
    PINNED_PANEL_SHA256,
    PINNED_SOURCE_PARAMETER_COUNT,
    PINNED_SPLIT_ID_SHA256S,
    ExactEdgeSidecarProtocol,
    build_exact_edge_sidecar_protocol,
    load_exact_edge_sidecar_protocol,
    main,
    validate_exact_edge_sidecar_protocol,
    write_exact_edge_sidecar_protocol,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _REPO_ROOT / "src/fisher_graph/gemma3_exact_edge_sidecar_protocol.py"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_setup_is_exact_model_edge_analysis_only() -> None:
    protocol = build_exact_edge_sidecar_protocol()
    assert protocol["schema"] == EXACT_EDGE_SIDECAR_PROTOCOL_SCHEMA
    assert protocol["format_version"] == (
        EXACT_EDGE_SIDECAR_PROTOCOL_FORMAT_VERSION
    )
    assert protocol["artifact_role"] == "setup_only_preregistered_edge_axis"
    assert protocol["setup_receipt"] == {
        "model_weights_loaded": False,
        "prompt_data_opened": False,
        "tokenization_performed": False,
        "activations_captured": False,
        "charts_fitted": False,
        "edges_fitted": False,
        "JVPs_computed": False,
        "finite_interventions_executed": False,
        "experiment_executed": False,
        "metrics_present": False,
    }
    scope = protocol["scope_firewall"]
    assert scope["axis"] == "edge_structure_only"
    assert scope["all_model_weights_exact"] is True
    assert scope["all_model_ranks_exact"] is True
    assert scope["baseline_model_output_must_remain_byte_identical"] is True
    assert scope["sidecar_edges_may_write_model_state"] is False
    assert scope["sidecar_edges_may_write_model_logits"] is False
    assert scope["compression"] is False
    assert scope["compression_axis_enabled"] is False
    assert scope["compressed_candidate_present"] is False
    assert scope["parameter_removal"] is False
    assert scope["parameter_merging"] is False
    assert scope["rank_reduction"] is False
    assert scope["weight_approximation"] is False
    assert json.loads(json.dumps(protocol)) == protocol


def test_exact_schedule_and_unified_chart_catalog_are_complete() -> None:
    protocol = build_exact_edge_sidecar_protocol()
    gate = protocol["exact_39_event_gate"]
    events = gate["events"]
    assert gate["event_count"] == PINNED_EVENT_COUNT == 39
    assert [event["event_index"] for event in events] == list(range(39))
    assert gate["event_ids"] == [event["event_id"] for event in events]
    assert gate["attention_mutation_count"] == 18
    assert gate["mlp_mutation_count"] == 18
    assert gate["global_attention_block_ordinals"] == [5, 11, 17]
    assert gate["tied_lexical_ingress_and_egress_exact"] is True
    assert gate["residual_carrier_initial_version"] == 0
    assert gate["residual_carrier_final_version"] == 36
    assert gate["source_parameter_inventory_parity_required"] is True
    parity = gate["sidecar_disabled_vs_enabled"]
    for name in (
        "event_order_equal",
        "every_mutation_delta_byte_identical",
        "final_carrier_byte_identical",
        "normalized_output_byte_identical",
        "logits_byte_identical",
        "source_state_fingerprint_unchanged",
    ):
        assert parity[name] is True
    assert parity["delta_nll_per_token"] == 0.0
    assert parity["native_to_sidecar_kl_per_query"] == 0.0
    assert parity["top1_agreement"] == 1.0

    charts = protocol["unified_exact_chart_contract"]
    sites = charts["sites"]
    assert charts["site_count"] == PINNED_CHART_SITE_COUNT == 73
    assert len(sites) == 1 + 2 * 36
    assert [site["site_index"] for site in sites] == list(range(73))
    assert len({site["site_id"] for site in sites}) == 73
    assert charts["coordinate_rank"] == PINNED_HIDDEN_WIDTH == 640
    assert charts["coordinate_rank_is_full_width"] is True
    assert charts["chart_count_per_site"] == 1
    assert charts["chart_fitting_changes_model_execution"] is False
    assert charts["capture_callback_is_observational_and_detached"] is True
    assert charts["callback_return_value_is_ignored"] is True
    assert {key: sites[0][key] for key in (
        "site_id",
        "site_index",
        "event_id",
        "event_index",
        "block_ordinal",
        "component_family",
        "boundary_role",
        "capture_tensor",
        "local_predecessor_site_id",
    )} == {
        "site_id": "lexical.ingress.output",
        "site_index": 0,
        "event_id": "lexical.ingress",
        "event_index": 0,
        "block_ordinal": -1,
        "component_family": "lexical",
        "boundary_role": "output",
        "capture_tensor": "exact_lexical_ingress_carrier",
        "local_predecessor_site_id": None,
    }
    assert sites[0]["source_eligible"] is True
    assert sites[0]["target_eligible"] is False
    assert {site["component_family"] for site in sites} == {
        "lexical",
        "attention",
        "mlp",
    }
    assert sum(site["component_family"] == "attention" for site in sites) == 36
    assert sum(site["component_family"] == "mlp" for site in sites) == 36
    for previous, current in zip(sites, sites[1:]):
        assert current["local_predecessor_site_id"] == previous["site_id"]


def test_edge_discovery_is_provenance_blind_but_topology_is_strict() -> None:
    protocol = build_exact_edge_sidecar_protocol()
    task = protocol["predictive_task"]
    assert task["source_block_must_precede_target_block"] is True
    assert task["source_component_family_must_differ_from_target"] is True
    assert task["real_edge_addition"] == (
        "one_strictly_earlier_cross_block_same_token_source"
    )
    assert task["layer_distance_is_a_numerical_feature"] is False
    assert task["source_event_id_is_a_numerical_feature"] is False
    assert task["source_component_family_is_a_numerical_feature"] is False
    assert task["provenance_may_only_filter_edge_eligibility"] is True
    assert task["source_label_may_break_numerical_score_ties"] is False
    assert task["label_only_renaming_invariance_required"] is True
    assert task["score_tie_break"] == (
        "canonical_hash_of_design_tensor_bytes_then_chart_index"
    )
    assert task["edge_coefficient_rank"] == 640
    assert task["edge_coefficient_shape"] == "640_by_640"
    assert task["edge_low_rank_factorization_used"] is False
    assert task["edge_numerical_rank_may_be_design_limited"] is True
    assert task["edge_numerical_rank_must_be_reported"] is True
    assert task["edge_selection_role"] == "A_fit_only"
    assert task["maximum_incoming_edges_per_target"] == 1
    assert task["all_edge_bearing_controls_share_real_edge_budget"] is True
    assert task["every_arm_uses_the_same_local_baseline"] is True

    arms = {arm["arm_id"]: arm for arm in protocol["arms"]}
    assert list(arms) == protocol["arm_order"]
    assert set(arms) == {
        "local_target_baseline",
        "real_forward_cross_block_edges",
        "zero_edge_control",
        "row_shuffled_control",
        "family_shuffled_control",
        "source_provenance_permutation_control",
    }
    real = arms["real_forward_cross_block_edges"]
    assert real["strictly_forward"] is True
    assert real["cross_block"] is True
    assert real["cross_component_family"] is True
    assert real["same_logical_token"] is True
    assert real["source_provenance_used_as_numerical_feature"] is False
    zero = arms["zero_edge_control"]
    assert zero["topology_copied_from"] == "real_forward_cross_block_edges"
    assert zero["edge_weights_refitted"] is False
    provenance = arms["source_provenance_permutation_control"]
    assert provenance["source_tensor_multiset_unchanged"] is True
    assert provenance["source_site_labels_fixed"] is True
    assert provenance["packet_assignment_to_source_labels_deranged"] is True
    assert provenance["permutation_within_target_eligible_shape_bucket"] is True
    assert provenance["same_permutation_frozen_for_B_and_C"] is True
    assert all(arm["exact_model_forward"] is True for arm in arms.values())
    assert all(arm["model_weights_mutated"] is False for arm in arms.values())
    assert all(
        arm["model_carrier_mutated_by_sidecar"] is False
        for arm in arms.values()
    )
    assert all(arm["model_logits_mutated_by_sidecar"] is False for arm in arms.values())
    assert all(arm["compression"] is False for arm in arms.values())


def test_A_B_C_firewall_hashes_the_real_disjoint_panel_partition() -> None:
    protocol = build_exact_edge_sidecar_protocol()
    firewall = protocol["data_firewall"]
    panel_path = _REPO_ROOT / PINNED_PANEL_PATH
    assert hashlib.sha256(panel_path.read_bytes()).hexdigest() == (
        PINNED_PANEL_SHA256
    )
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict[str, object]]] = {}
    for example in panel["examples"]:
        by_family.setdefault(str(example["family_id"]), []).append(example)
    ranges = {
        "A_fit": (0, 6),
        "B_predictive": (6, 8),
        "C_intervention": (8, 10),
    }
    role_ids: dict[str, list[str]] = {}
    for role, (start, stop) in ranges.items():
        ids = [
            str(example["example_id"])
            for family in sorted(by_family)
            for example in by_family[family][start:stop]
        ]
        role_ids[role] = ids
        assert len(ids) == firewall["roles"][role]["example_count"]
        assert hashlib.sha256(_canonical_json_bytes(ids)).hexdigest() == (
            PINNED_SPLIT_ID_SHA256S[role]
        )
        assert firewall["roles"][role]["example_ids_sha256"] == (
            PINNED_SPLIT_ID_SHA256S[role]
        )
    assert not (set(role_ids["A_fit"]) & set(role_ids["B_predictive"]))
    assert not (set(role_ids["A_fit"]) & set(role_ids["C_intervention"]))
    assert not (set(role_ids["B_predictive"]) & set(role_ids["C_intervention"]))
    assert firewall["example_and_prompt_disjoint_across_roles"] is True
    assert firewall["families_overlap_across_roles"] is True
    assert firewall["family_disjoint_confirmation"] is False
    assert firewall["B_and_C_rows_may_enter_fit"] is False
    assert firewall["B_metrics_may_change_edge_selection"] is False
    assert firewall["C_metrics_may_change_edge_selection"] is False
    assert firewall["raw_prompt_text_or_token_ids_written_to_report"] is False
    assert firewall["row_identity_sha256_required_for_every_capture"] is True


def test_predictive_and_causal_gates_require_heldout_falsification() -> None:
    protocol = build_exact_edge_sidecar_protocol()
    predictive = protocol["heldout_predictive_gate"]
    assert predictive == {
        "role": "B_predictive",
        "primary_metric": "Fisher_weighted_relative_SSE_gain_vs_local",
        "minimum_real_gain_vs_local": 0.02,
        "minimum_real_gain_margin_over_each_negative_control": 0.01,
        "paired_family_bootstrap_lower_bound_must_exceed_zero": True,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": 10_000,
        "edge_catalog_is_not_pruned_or_refitted_on_B": True,
        "any_selected_edge_with_nonpositive_B_gain_fails_the_arm": True,
    }
    causal = protocol["heldout_causal_gate"]
    assert causal["role"] == "C_intervention"
    assert causal["predictive_gain_alone_authorizes_causal_language"] is False
    assert causal["both_JVP_and_finite_intervention_must_pass"] is True
    jvp = causal["JVP"]
    assert jvp["direction_count_per_edge"] == 8
    assert jvp["exact_autograd_reference_required"] is True
    assert jvp["minimum_real_direction_cosine"] == 0.90
    assert jvp["maximum_real_relative_norm_error"] == 0.25
    assert jvp["minimum_cosine_margin_over_each_negative_control"] == 0.05
    finite = causal["finite_intervention"]
    assert finite["shadow_execution_only"] is True
    assert finite["baseline_model_output_is_never_replaced"] is True
    assert finite["source_activation_perturbation_is_ephemeral"] is True
    assert finite["central_plus_minus_epsilon"] is True
    assert finite["epsilon_times_source_RMS"] == [0.001, 0.003, 0.01]
    assert finite["minimum_real_direction_cosine"] == 0.80
    assert finite["maximum_real_relative_norm_error"] == 0.35
    assert finite["maximum_adjacent_epsilon_derivative_disagreement"] == 0.10


def test_resource_ledger_counts_sidecar_cost_and_forbids_compression_claims() -> None:
    protocol = build_exact_edge_sidecar_protocol()
    cost = protocol["analysis_cost_and_claim_firewall"]
    assert cost["source_model_parameter_count"] == PINNED_SOURCE_PARAMETER_COUNT
    assert cost["model_parameter_reduction_fraction"] == 0.0
    assert cost["model_storage_reduction_fraction"] == 0.0
    assert cost["model_MAC_reduction_fraction"] == 0.0
    assert cost["compression"] is False
    assert cost["compression_success_claimed"] is False
    assert cost["latency_improvement_claimed"] is False
    for name in (
        "sidecar_chart_scalar_count_must_be_reported",
        "sidecar_edge_scalar_count_must_be_reported",
        "fit_FLOPs_or_MACs_must_be_reported",
        "exact_forward_count_must_be_reported",
        "backward_or_JVP_count_must_be_reported",
        "finite_intervention_forward_count_must_be_reported",
        "peak_resident_bytes_must_be_reported",
    ):
        assert cost[name] is True
    assert cost["sidecar_cost_may_not_be_subtracted_from_model_cost"] is True
    assert cost["edge_predictive_success_is_not_compression_success"] is True
    promotion = protocol["promotion_firewall"]
    assert promotion["edge_axis_pass_requires_exact_gate"] is True
    assert promotion["edge_axis_pass_requires_predictive_gate"] is True
    assert promotion["edge_axis_pass_requires_JVP_gate"] is True
    assert promotion["edge_axis_pass_requires_finite_intervention_gate"] is True
    assert promotion["combined_edge_and_compression_arm_locked"] is True
    assert promotion["this_setup_authorizes_model_mutation"] is False
    assert promotion["this_setup_authorizes_compression_claim"] is False


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["scope_firewall"].__setitem__(
            "all_model_weights_exact", False
        ),
        lambda value: value["exact_39_event_gate"].__setitem__(
            "event_count", 38
        ),
        lambda value: value["unified_exact_chart_contract"].__setitem__(
            "coordinate_rank", 512
        ),
        lambda value: value["predictive_task"].__setitem__(
            "layer_distance_is_a_numerical_feature", True
        ),
        lambda value: value["predictive_task"].__setitem__(
            "edge_low_rank_factorization_used", True
        ),
        lambda value: value["predictive_task"].__setitem__(
            "edge_numerical_rank_must_be_reported", False
        ),
        lambda value: value["arms"][1].__setitem__("cross_block", False),
        lambda value: value["data_firewall"].__setitem__(
            "B_and_C_rows_may_enter_fit", True
        ),
        lambda value: value["heldout_predictive_gate"].__setitem__(
            "minimum_real_gain_vs_local", 0.0
        ),
        lambda value: value["heldout_causal_gate"]["JVP"].__setitem__(
            "exact_autograd_reference_required", False
        ),
        lambda value: value["analysis_cost_and_claim_firewall"].__setitem__(
            "compression_success_claimed", True
        ),
        lambda value: value["setup_receipt"].__setitem__(
            "experiment_executed", True
        ),
    ),
)
def test_validator_rejects_semantic_tampering(mutate: object) -> None:
    protocol = deepcopy(build_exact_edge_sidecar_protocol())
    mutate(protocol)  # type: ignore[operator]
    with pytest.raises(ValueError, match="does not match frozen v1"):
        validate_exact_edge_sidecar_protocol(protocol)


def test_protocol_construction_is_standard_library_only() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "typing",
    }
    assert "torch" not in imported_roots
    assert "transformers" not in imported_roots


def test_frozen_constructor_rejects_scope_broadening() -> None:
    with pytest.raises(ValueError, match="exact full-width"):
        ExactEdgeSidecarProtocol(coordinate_rank=512)
    with pytest.raises(ValueError, match="full-width edge coefficients"):
        ExactEdgeSidecarProtocol(edge_coefficient_rank=512)
    with pytest.raises(ValueError, match="one full-width chart"):
        ExactEdgeSidecarProtocol(chart_count_per_site=2)
    with pytest.raises(ValueError, match="one incoming skip edge"):
        ExactEdgeSidecarProtocol(maximum_real_incoming_edges_per_target=2)
    with pytest.raises(ValueError, match="ridge is frozen"):
        ExactEdgeSidecarProtocol(ridge=1.0e-3)
    with pytest.raises(ValueError, match="intervention ladder is frozen"):
        ExactEdgeSidecarProtocol(intervention_epsilon_ladder=(0.01,))


def test_write_load_and_check_are_strict_and_non_overwriting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "protocol.json"
    written = write_exact_edge_sidecar_protocol(path)
    assert load_exact_edge_sidecar_protocol(path) == written
    with pytest.raises(FileExistsError):
        write_exact_edge_sidecar_protocol(path)
    assert main(("--check", "--output", str(path))) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "action": "checked",
        "compression": False,
        "experiment_executed": False,
        "model_weights_loaded": False,
        "output": str(path),
        "protocol_sha256": written["protocol_sha256"],
        "setup_only": True,
    }


def test_checked_in_protocol_artifact_is_canonical() -> None:
    artifact = (
        _REPO_ROOT / "artifacts/research/gemma3-exact-edge-sidecar-protocol-v1.json"
    )
    assert load_exact_edge_sidecar_protocol(artifact) == (
        build_exact_edge_sidecar_protocol()
    )
