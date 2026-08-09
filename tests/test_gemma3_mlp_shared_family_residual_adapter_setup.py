from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisher_graph.gemma3_mlp_shared_family_residual_adapter_setup import (
    DEFAULT_OUTPUT,
    LOW_RANK_POINTS,
    ORDERED_RESIDUAL_RANKS,
    ORDERED_ROLES,
    PINNED_PREDECESSOR_PROTOCOL_SHA256,
    PINNED_PREDECESSOR_REPORT_SHA256,
    SEEDS,
    Gemma3MLPSharedFamilyResidualAdapterSetup,
    build_gemma3_mlp_shared_family_residual_adapter_setup,
    load_gemma3_mlp_shared_family_residual_adapter_setup,
    main,
    mlp_residual_adapter_scalars,
    validate_gemma3_mlp_shared_family_residual_adapter_setup,
    write_gemma3_mlp_shared_family_residual_adapter_setup,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODULE = _ROOT / (
    "src/fisher_graph/gemma3_mlp_shared_family_residual_adapter_setup.py"
)
_SHIM = _ROOT / "src/fisher_graph_mlp_shared_family_residual_adapter_setup.py"
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


def test_setup_is_standard_library_only_and_has_no_run_receipt() -> None:
    setup = build_gemma3_mlp_shared_family_residual_adapter_setup()
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
    assert not any(setup["setup_receipt"].values())
    assert setup["claim_boundary"]["setup_only"] is True
    assert json.loads(json.dumps(setup)) == setup


def test_predecessor_protocol_and_report_are_cryptographically_frozen() -> None:
    chain = build_gemma3_mlp_shared_family_residual_adapter_setup()[
        "predecessor_chain"
    ]
    assert PINNED_PREDECESSOR_PROTOCOL_SHA256 == (
        "32747a59ce2aa7ad0633904b54680d19b7f2da1545bb136c4816d025667f61f2"
    )
    assert chain["protocol"]["protocol_sha256"] == (
        PINNED_PREDECESSOR_PROTOCOL_SHA256
    )
    assert chain["protocol"]["file_sha256"] == (
        "b2caaae36742864f7a0bb491ed80ab08f8c94029d6d554d7d68cdbd1f7b1d243"
    )
    assert PINNED_PREDECESSOR_REPORT_SHA256 == (
        "7f062b40a2af2fe61364b208189773b82569bbe88902c0b8f3e7bbe607401926"
    )
    assert chain["report"]["report_sha256"] == (
        PINNED_PREDECESSOR_REPORT_SHA256
    )
    assert chain["report"]["file_sha256"] == (
        "2ad135209ff1c780944c7d0efd5dbd7b52724266dc880345abb31cfc9bbcf64a"
    )
    assert chain["required_predecessor_outcome"] == {
        "all_five_arms_passed": True,
        "maximum_valid_query_logit_error": 0.0,
        "delta_nll_per_token": 0.0,
        "native_to_candidate_kl_per_query": 0.0,
        "top1_agreement": 1.0,
        "compiled_source_calls": 0,
        "invocation_count": 39,
        "edge_occurrence_count": 75,
        "stored_scalars": 268_098_176,
    }


def test_four_role_manifests_are_family_disjoint_and_exhaust_the_panel() -> None:
    firewall = build_gemma3_mlp_shared_family_residual_adapter_setup()[
        "data_firewall"
    ]
    assert firewall["ordered_roles"] == list(ORDERED_ROLES)
    roles = firewall["roles"]
    assert [roles[role]["example_count"] for role in ORDERED_ROLES] == [
        20,
        10,
        20,
        10,
    ]
    assert roles["A_fit"]["family_ids"] == [
        "integer_addition",
        "integer_subtraction",
    ]
    assert roles["A_select"]["family_ids"] == ["integer_comparison"]
    assert roles["B_guard"]["family_ids"] == [
        "lexical_antonym",
        "category_completion",
    ]
    assert roles["C_assessment"]["family_ids"] == ["world_capitals"]
    families = [
        family
        for role in ORDERED_ROLES
        for family in roles[role]["family_ids"]
    ]
    examples = [
        example
        for role in ORDERED_ROLES
        for example in roles[role]["ordered_example_ids"]
    ]
    assert len(families) == len(set(families)) == 6
    assert len(examples) == len(set(examples)) == 60
    assert all(len(roles[role]["role_manifest_sha256"]) == 64 for role in roles)
    assert firewall["A_fit_is_the_only_role_visible_to_pair_search_or_fitting"]


def test_nonadjacent_pair_selection_uses_only_activation_and_fisher() -> None:
    selection = build_gemma3_mlp_shared_family_residual_adapter_setup()[
        "pair_and_anchor_selection"
    ]
    assert selection["role"] == "A_fit"
    assert selection["nonadjacent_definition"] == (
        "absolute_block_difference_at_least_2"
    )
    assert selection["unordered_eligible_pair_count"] == 136
    assert selection["directed_pair_and_anchor_candidate_count"] == 272
    assert selection["primary_objective_direction"] == "minimize"
    assert selection["primary_objective_name"] == (
        "zero_residual_shared_core_Fisher_weighted_relative_SSE"
    )
    assert selection["adapter_probe_fit_during_pair_selection_allowed"] is False
    assert selection["activation_and_Fisher_evidence_only"] is True
    assert selection["block_ordinal_or_distance_used_as_numerical_feature"] is False
    assert selection["nonadjacency_is_eligibility_only"] is True
    assert selection["ordered_tie_break"] == [
        "lexicographically_smaller_SHA256_of_canonical_float32_aligned_design_"
        "and_target_tensor_bytes",
        "fail_closed_if_the_numerical_tensor_digest_is_identical",
    ]
    assert selection["pair_anchor_and_A_fit_row_manifest_frozen_before_A_select"]


def test_exact_and_control_arm_order_is_frozen_per_rank() -> None:
    contract = build_gemma3_mlp_shared_family_residual_adapter_setup()[
        "arm_contract"
    ]
    assert contract["residual_rank_order"] == list(ORDERED_RESIDUAL_RANKS)
    expected = [
        "native",
        "exact.heterogeneous_full_rank_atom_bank_lift",
        "exact.shared_core_dense_residual",
        "ablation.shared_core_zero_residual",
    ]
    for rank in LOW_RANK_POINTS:
        expected.extend(
            [
                f"primary.shared_core_residual_r{rank}",
                f"control.independent_member_r{rank}",
                f"control.shuffled_residual_r{rank}",
            ]
        )
    assert contract["arm_order"] == expected
    assert contract["dense_exact_control_must_pass_before_any_compressed_fit"]
    assert contract["zero_residual_arm_is_control_only_despite_highest_reduction"]


def test_resource_ladder_counts_one_core_and_one_residual_only() -> None:
    resources = build_gemma3_mlp_shared_family_residual_adapter_setup()[
        "resource_accounting"
    ]
    assert [mlp_residual_adapter_scalars(rank) for rank in LOW_RANK_POINTS] == [
        2_065_664,
        1_033_472,
        517_376,
        259_328,
    ]
    assert resources["ordered_candidate_stored_scalars"] == [
        268_098_176,
        266_230_400,
        265_198_208,
        264_682_112,
        264_424_064,
    ]
    assert resources["zero_residual_candidate_stored_scalars"] == 264_164_736
    points = resources["ordered_points"]
    assert [point["residual_rank"] for point in points] == [
        "dense",
        256,
        128,
        64,
        32,
    ]
    assert all(point["physical_shared_core_count"] == 1 for point in points)
    assert all(point["learned_correction_edge_count"] == 0 for point in points)
    assert resources["learned_correction_edge_scalars"] == 0
    assert resources["charts_and_learned_transport_scalars"] == 0


def test_fidelity_control_bootstrap_and_seed_gates_are_frozen() -> None:
    setup = build_gemma3_mlp_shared_family_residual_adapter_setup()
    gates = setup["acceptance_gates"]
    exact = gates["exact_dense_residual_control"]
    assert exact["maximum_valid_query_logit_absolute_error"] == 0.0
    assert exact["native_to_candidate_kl_per_query_maximum"] == 0.0
    assert exact["top1_agreement_minimum"] == 1.0
    fidelity = gates["compressed_candidate_fidelity"]
    assert fidelity["delta_nll_per_token_maximum"] == 0.10
    assert fidelity["native_to_candidate_kl_per_query_maximum"] == 0.10
    assert fidelity["top1_agreement_minimum"] == 0.90
    assert fidelity["valid_query_logit_nrmse_maximum"] == 0.10
    assert fidelity["worst_family_bootstrap_lower_bound_minimum"] == 0.85
    dominance = gates["control_dominance"]
    assert dominance[
        "real_residual_gain_simultaneous_lower_bound_minimum_exclusive"
    ] == 0.0
    assert dominance[
        "gain_over_matched_independent_simultaneous_lower_bound_minimum_exclusive"
    ] == 0.0
    assert dominance[
        "shuffled_gain_fraction_of_real_simultaneous_upper_bound_maximum"
    ] == 0.25
    bootstrap = gates["B_guard_bootstrap"]
    assert bootstrap["seed"] == SEEDS["bootstrap"] == 1_733
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["confidence_level"] == 0.95
    assert bootstrap["cluster_unit"] == "prompt_or_paraphrase_family"
    assert bootstrap["paired_resample_indices_shared_by_all_arms_and_metrics"]
    assert list(setup["fit_contract"]["seeds"].values()) == [
        1_729,
        1_730,
        1_731,
        1_732,
        1_733,
        1_734,
    ]
    residual_fit = setup["fit_contract"]["low_rank_residual"]
    assert residual_fit["gate_and_up_design"] == (
        "X_equals_exact_member_normalized_MLP_input"
    )
    assert residual_fit["down_design"] == "H_equals_exact_member_gated_activation"
    assert residual_fit["relative_ridge"] == 1.0e-4
    assert residual_fit["ridge_scale"] == (
        "trace(X_weighted_transpose_X_weighted)_divided_by_design_width"
    )
    assert residual_fit["absolute_ridge"] == (
        "relative_ridge_times_ridge_scale"
    )
    assert residual_fit["solver_order"] == [
        "multiply_design_and_target_rows_by_sqrt_weight",
        "form_ridge_gram_and_solve_the_full_weighted_coefficient",
        "left_whiten_the_coefficient_by_the_ridge_gram_Cholesky",
        "compute_canonical_rank_r_SVD_of_the_whitened_coefficient",
        "unwhiten_the_truncated_coefficient",
        "factor_the_rank_r_coefficient_as_output_by_r_times_r_by_input",
    ]
    assert residual_fit["factors_for_each_rank_fitted_independently"] is True
    assert residual_fit["optimizer_or_gradient_descent_used"] is False
    fit_contract = setup["fit_contract"]
    assert fit_contract[
        "selected_member_prefix_is_exact_by_single_approximation"
    ] is True
    assert fit_contract[
        "exact_prefix_input_hash_equality_authenticated_on_A_fit"
    ] is True
    assert fit_contract["candidate_prefix_recapture_required"] is False


def test_scope_keeps_all_occurrences_and_forbids_learned_edges() -> None:
    scope = build_gemma3_mlp_shared_family_residual_adapter_setup()[
        "scope_freeze"
    ]
    assert scope["compressed_component"] == "one_nonadjacent_MLP_pair_only"
    assert scope["invocation_count"] == 39
    assert scope["edge_occurrence_count"] == 75
    assert scope["learned_correction_edge_count"] == 0
    assert scope["optional_correction_edge_budget_present"] is False
    assert scope["all_18_attention_invocations"] == "exact_and_frozen"
    assert scope["tied_lexical_ingress_and_egress"] == "exact_and_frozen"


def test_canonical_artifact_and_typed_wrapper_round_trip() -> None:
    checked_in = load_gemma3_mlp_shared_family_residual_adapter_setup(_ARTIFACT)
    built = build_gemma3_mlp_shared_family_residual_adapter_setup()
    assert checked_in == built
    assert Gemma3MLPSharedFamilyResidualAdapterSetup.build().to_dict() == built
    validate_gemma3_mlp_shared_family_residual_adapter_setup(checked_in)
    changed = deepcopy(checked_in)
    changed["scope_freeze"]["learned_correction_edge_count"] = 1
    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_mlp_shared_family_residual_adapter_setup(changed)


def test_write_is_exclusive_and_check_mode_is_read_only(tmp_path: Path) -> None:
    output = tmp_path / "setup.json"
    written = write_gemma3_mlp_shared_family_residual_adapter_setup(output)
    assert load_gemma3_mlp_shared_family_residual_adapter_setup(output) == written
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_gemma3_mlp_shared_family_residual_adapter_setup(output)
    before = output.read_bytes()
    assert main(["--check", "--output", str(output)]) == 0
    assert output.read_bytes() == before
