from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_typed_mode_response_setup import (
    ATTENTION_FULL_SEQUENCE_VALID_TOKEN_EXECUTION_COUNT,
    ATTENTION_LOGICAL_MAC_ESTIMATE,
    ATTENTION_MODE_IDS,
    ATTENTION_NATIVE_PREFIX_CALL_COUNT,
    ATTENTION_PROBE_COUNT,
    ATTENTION_SELECTED_CANDIDATE_OUTPUT_COUNT,
    DEFAULT_OUTPUT,
    GEMMA3_TYPED_MODE_RESPONSE_SETUP_SCHEMA,
    Gemma3TypedModeResponseSetup,
    LEXICAL_MODE_IDS,
    MLP_MODE_IDS,
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    RESPONSE_EXAMPLE_PAIR_STAT_COUNT,
    RESPONSE_FAMILY_PAIR_STAT_COUNT,
    RESPONSE_MODE_IDS,
    RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT,
    RESPONSE_UNORDERED_PAIR_COUNT,
    TERMINAL_NORM_MODE_ID,
    build_gemma3_typed_mode_response_setup,
    load_gemma3_typed_mode_response_setup,
    main,
    validate_gemma3_typed_mode_response_setup,
    write_gemma3_typed_mode_response_setup,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "src/fisher_graph/gemma3_typed_mode_response_setup.py"
_SHIM = _ROOT / "src/fisher_graph_typed_mode_response_setup.py"
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
    import hashlib

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
    setup = build_gemma3_typed_mode_response_setup()
    assert not any(setup["setup_receipt"].values())
    assert setup["schema"] == GEMMA3_TYPED_MODE_RESPONSE_SETUP_SCHEMA
    assert setup["claim_boundary"]["setup_only"] is True
    assert json.loads(json.dumps(setup)) == setup


def test_exactly_38_response_modes_have_four_native_abis() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    scope = setup["scope_freeze"]
    catalog = setup["typed_mode_catalog"]
    assert len(RESPONSE_MODE_IDS) == len(catalog) == 38
    assert [entry["mode_id"] for entry in catalog] == list(RESPONSE_MODE_IDS)
    assert len(set(RESPONSE_MODE_IDS)) == 38
    assert scope["response_mode_count"] == 38
    assert scope["native_ABI_count"] == 4
    assert set(setup["native_abis"]) == {
        "attention",
        "mlp",
        "lexical_ingress",
        "lexical_egress",
    }
    assert len(set(setup["native_abi_sha256s"].values())) == 4


def test_attention_and_mlp_are_provenance_on_generic_modes() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    assert ATTENTION_MODE_IDS == tuple(
        f"mode.mutation.{2 * block + 1}" for block in range(18)
    )
    assert MLP_MODE_IDS == tuple(
        f"mode.mutation.{2 * block + 2}" for block in range(18)
    )
    by_id = {entry["mode_id"]: entry for entry in setup["typed_mode_catalog"]}
    for block, mode_id in enumerate(ATTENTION_MODE_IDS):
        entry = by_id[mode_id]
        assert entry["source_invocation_id"] == f"mutation.{2 * block + 1}"
        assert entry["source_block_ordinal"] == block
        assert entry["operator_kind"] == "attention"
        assert entry["generic_core_mode_kind"] == (
            "carrier_mutation.sequence_routed.h640"
        )
        assert "token_local" not in entry["native_abi_id"]
    for block, mode_id in enumerate(MLP_MODE_IDS):
        entry = by_id[mode_id]
        assert entry["source_invocation_id"] == f"mutation.{2 * block + 2}"
        assert entry["source_block_ordinal"] == block
        assert entry["operator_kind"] == "mlp"
    assert setup["mode_ontology"][
        "implementation_names_are_provenance_not_ontology"
    ] is True
    assert setup["mode_ontology"]["cross_ABI_native_execution_forbidden"]


def test_native_abi_payloads_authenticate_with_the_generic_core() -> None:
    from fisher_graph.mode_similarity_atlas import ModeABI

    setup = build_gemma3_typed_mode_response_setup()
    for name, payload in setup["native_abis"].items():
        assert ModeABI(**payload).sha256 == setup["native_abi_sha256s"][name]
    assert setup["native_abi_sha256s"]["mlp"] == (
        "1cf7faee029f7f1ccce70676015533cc6247d19f87eaee49b445f9f3238238b0"
    )


def test_existing_mlp_native_atlas_is_fully_authenticated() -> None:
    mlp = build_gemma3_typed_mode_response_setup()["predecessors"][
        "mlp_native_atlas"
    ]
    assert mlp["setup_sha256"] == (
        "7edb27bc2a2778459764a855bba0d0f358b55b06b547e2a4b6c1029ee0110dd0"
    )
    assert mlp["setup_file_sha256"] == (
        "4c3f39fdf18917be77f246ecdca852d001810915f9fdecc516e1e2860f088290"
    )
    assert mlp["report_sha256"] == (
        "0fc453081460885c2d07015ad4cf3ffae47970bec47855a34f1211a188da5161"
    )
    assert mlp["report_file_sha256"] == (
        "82a15b23b4a073a8874206a769ea5e8fbc05da082808e79ba13b2b78e7a69fb2"
    )
    assert mlp["required_diagnostic_outcome"] == "mode_similarity_discovery_only"
    assert mlp["ordered_mode_ids"] == list(MLP_MODE_IDS)
    assert mlp["native_distances_may_not_be_reused_as_common_response_stats"]
    assert mlp["stable_neighbor_hypotheses_are_not_inherited_as_claims"]


def test_attention_native_atlas_is_new_full_sequence_work() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    required = setup["required_native_atlases"]["attention"]
    contract = setup["attention_experiment_contract"]
    assert required["must_be_newly_executed"] is True
    assert required["ordered_mode_ids"] == list(ATTENTION_MODE_IDS)
    assert required["full_native_sequence_operator_required"] is True
    assert contract["rowwise_attention_cross_execution_forbidden"] is True
    assert contract["full_atom_ownership"] == [
        "input_RMSNorm",
        "Q_K_V_projections",
        "Q_and_K_normalization",
        "RoPE",
        "causal_grouped_query_routing",
        "output_projection",
        "post_attention_RMSNorm",
    ]
    assert contract["capture_batch_count"] == 18
    assert contract["capture_forward_count"] == 18
    assert contract["capture_backward_count"] == 18
    assert contract["selected_probe_count"] == ATTENTION_PROBE_COUNT == 2_160
    assert contract["selected_candidate_row_output_count"] == (
        ATTENTION_SELECTED_CANDIDATE_OUTPUT_COUNT
    ) == 38_880
    assert contract["native_prefix_call_count"] == (
        ATTENTION_NATIVE_PREFIX_CALL_COUNT
    ) == 324
    assert contract["full_sequence_valid_token_execution_count"] == (
        ATTENTION_FULL_SEQUENCE_VALID_TOKEN_EXECUTION_COUNT
    ) == 263_736
    assert contract["logical_MAC_estimate"] == (
        ATTENTION_LOGICAL_MAC_ESTIMATE
    ) == 436_467_253_248


def test_attention_and_response_bootstrap_units_are_distinct() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    attention = setup["attention_within_family_bootstrap_config"]
    response = setup["common_response_within_family_bootstrap_config"]
    assert attention["sampling_unit"] == "example_plus_selected_probes_cluster"
    assert response["sampling_unit"] == (
        "example_plus_centered_logits4_response_cluster"
    )
    assert response["response_coordinates_per_example"] == 4
    assert response["all_four_coordinates_resampled_together"] is True
    assert attention["seed"] == response["seed"] == 2_311
    assert attention["resample_count"] == response["resample_count"] == 10_000


def test_common_response_abi_uses_finite_baseline_minus_ablation() -> None:
    from fisher_graph.multi_abi_mode_atlas import DownstreamResponseABI

    setup = build_gemma3_typed_mode_response_setup()
    payload = setup["common_response_abi"]
    abi = DownstreamResponseABI(
        abi_name=payload["abi_name"],
        abi_version=payload["abi_version"],
        response_coordinate_signature=payload["response_coordinate_signature"],
        response_width=payload["response_width"],
        baseline_signature=payload["baseline_signature"],
        intervention_signature=payload["intervention_signature"],
        row_catalog_sha256=payload["row_catalog_sha256"],
        weighting_catalog_sha256=payload["weighting_catalog_sha256"],
    )
    assert abi.canonical_payload() == payload
    contract = setup["common_response_contract"]
    assert contract["primary_measurement"] == (
        "exact_native_output_scale_one_to_zero_finite_ablation"
    )
    assert contract["response_formula"] == (
        "r_m=C4*(restricted_logits4_at_scale1-"
        "restricted_logits4_at_scale0)"
    )
    assert contract["centering_matrix"] == "C4=I4-(1/4)*ones4x4"
    assert contract["response_width"] == 4
    assert contract["response_effective_rank_after_centering"] == 3


def test_primary_weights_are_analytic_baseline_output_fisher() -> None:
    contract = build_gemma3_typed_mode_response_setup()[
        "common_response_contract"
    ]
    weights = contract["weighting_catalog"]
    assert weights["analytic_gradient_formula"] == (
        "g_k=softmax(z_scale1)_k-one[k=gold]"
    )
    assert weights["output_Fisher_formula"] == "w_k=g_k^2_computed_in_float64"
    assert weights["gold_choice_or_Fisher_weighting_used"] is True
    binding = contract["output_Fisher_weight_catalog_binding"]
    assert binding["runtime_float64_value_catalog_SHA256_required"] is True
    assert binding["runtime_catalog_must_be_authenticated_before_pair_stats"]
    normalization = contract["distance_normalization"]
    assert normalization["per_example_weight_renormalization"] is False
    assert normalization["directed_a_to_b"].startswith("SSE/max(mode_a_energy")
    assert normalization["symmetric_distance"] == "0.5*(d(a_to_b)+d(b_to_a))"


def test_lexical_occurrences_are_distinct_but_share_tied_state() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    by_id = {entry["mode_id"]: entry for entry in setup["typed_mode_catalog"]}
    ingress, egress = (by_id[mode_id] for mode_id in LEXICAL_MODE_IDS)
    assert ingress["native_abi_id"] != egress["native_abi_id"]
    assert ingress["response_intervention_id"] != egress["response_intervention_id"]
    assert ingress["source_atom_id"] == egress["source_atom_id"]
    assert ingress["shared_physical_state_id"] == egress["shared_physical_state_id"]
    tied = setup["tied_lexical_state_contract"]
    assert tied["physical_parameter_state_group_count"] == 1
    assert tied["native_occurrence_intervention_count"] == 2
    assert tied["both_logical_modes_are_executed_independently"] is True
    assert tied["both_finite_responses_must_be_exactly_equal"] is True
    assert tied["pairwise_family_and_example_SSE_must_be_exactly_zero"]


def test_response_inventory_counts_and_exact_forward_ledger() -> None:
    contract = build_gemma3_typed_mode_response_setup()[
        "common_response_contract"
    ]
    assert contract["mode_count"] == 38
    assert contract["logical_intervention_count"] == 38
    assert contract["native_occurrence_intervention_count"] == 38
    assert contract["physical_parameter_state_group_count"] == 37
    assert contract["token_id_baseline_parity_forward_count"] == 18
    assert contract["canonical_inputs_embeds_baseline_forward_count"] == 18
    assert contract["logical_intervention_forward_count"] == 684
    assert contract["total_full_model_forward_count"] == (
        RESPONSE_TOTAL_FULL_MODEL_FORWARD_COUNT
    ) == 720
    assert contract["backward_count_for_primary_response"] == 0
    assert contract["unordered_pair_count"] == RESPONSE_UNORDERED_PAIR_COUNT == 703
    assert contract["family_pair_sufficient_stat_count"] == (
        RESPONSE_FAMILY_PAIR_STAT_COUNT
    ) == 4_218
    assert contract["example_pair_sufficient_stat_count"] == (
        RESPONSE_EXAMPLE_PAIR_STAT_COUNT
    ) == 42_180


def test_terminal_norm_is_exact_baseline_infrastructure_only() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    terminal = setup["terminal_norm_baseline"]
    assert terminal["mode_id"] == TERMINAL_NORM_MODE_ID
    assert TERMINAL_NORM_MODE_ID not in RESPONSE_MODE_IDS
    assert terminal["included_in_38_response_modes"] is False
    assert terminal["native_atlas_built"] is False
    assert terminal["finite_ablation_response_measured"] is False
    assert terminal["generator_family_neighbor_cluster_merge_or_compression_eligible"] is False
    assert terminal[
        "terminal_RMSNorm_tied_lexical_readout_and_logit_softcap_exact"
    ] is True


def test_short_panel_cannot_support_long_context_attention_claims() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    limitation = setup["short_context_limitation"]
    assert limitation["observed_valid_prompt_token_count_minimum"] == 11
    assert limitation["observed_valid_prompt_token_count_maximum"] == 26
    assert limitation["sliding_window_token_count"] == 512
    assert limitation["sliding_window_cutoff_exercised"] is False
    assert limitation[
        "global_versus_sliding_attention_long_context_behavior_tested"
    ] is False
    future = limitation[
        "future_attention_neighbor_cluster_merge_or_compression_claim_requires"
    ]
    assert future["fresh_prompt_and_paraphrase_disjoint_panel"] is True
    assert future["at_least_one_valid_prompt_token_count_greater_than"] == 512
    assert future["minimum_qualifying_valid_prompt_token_count"] == 513
    assert future["must_include_queries_with_keys_outside_the_512_token_window"]


def test_all_60_prompts_are_spent_and_no_structural_claim_is_made() -> None:
    setup = build_gemma3_typed_mode_response_setup()
    firewall = setup["discovery_data_firewall"]
    assert firewall["prompt_count"] == 60
    assert len(firewall["ordered_example_ids"]) == 60
    assert len(set(firewall["ordered_example_ids"])) == 60
    assert firewall["all_prompts_spent"] is True
    claims = setup["claim_boundary"]
    for field in (
        "neighbor_hypothesis_claimed",
        "cluster_claimed",
        "generator_family_claimed",
        "merge_or_compression_claimed",
        "causal_or_functional_interchangeability_claimed",
        "long_context_attention_behavior_claimed",
    ):
        assert claims[field] is False
    tangent = setup["common_response_contract"]["optional_tangent_diagnostic"]
    assert tangent["allowed"] is True
    assert tangent["finite_ablation_remains_primary"] is True
    assert tangent[
        "may_enter_primary_pair_statistics_distance_affinity_order_or_claims"
    ] is False


def test_validation_wrapper_file_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    setup = build_gemma3_typed_mode_response_setup()
    validate_gemma3_typed_mode_response_setup(setup)
    assert Gemma3TypedModeResponseSetup.build().to_dict() == setup
    destination = tmp_path / "setup.json"
    assert write_gemma3_typed_mode_response_setup(destination) == setup
    assert load_gemma3_typed_mode_response_setup(destination) == setup
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_gemma3_typed_mode_response_setup(destination)
    tampered = deepcopy(setup)
    tampered["claim_boundary"]["cluster_claimed"] = True
    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_typed_mode_response_setup(tampered)
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_gemma3_typed_mode_response_setup([])  # type: ignore[arg-type]


def test_checked_in_artifact_and_both_cli_surfaces_are_canonical(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    expected = build_gemma3_typed_mode_response_setup()
    assert expected["setup_sha256"] == PINNED_SETUP_SHA256
    assert load_gemma3_typed_mode_response_setup(_ARTIFACT) == expected
    assert _sha256_file(_ARTIFACT) == PINNED_SETUP_FILE_SHA256
    assert main(["--check", "--output", str(_ARTIFACT)]) == 0

    direct_output = tmp_path / "direct.json"
    shim_output = tmp_path / "shim.json"
    assert main(["--output", str(direct_output)]) == 0
    assert load_gemma3_typed_mode_response_setup(direct_output) == expected
    completed = subprocess.run(
        [sys.executable, str(_SHIM), "--output", str(shim_output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert load_gemma3_typed_mode_response_setup(shim_output) == expected
