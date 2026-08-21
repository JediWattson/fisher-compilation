from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path

import pytest

from fisher_graph import gemma3_chart_native_neighbor_v2_setup as setup
from fisher_graph import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup as diagnostic_setup,
)


def test_v2_freezes_full_intrinsic_descending_capacity_rule() -> None:
    state = setup.build_setup()
    selection = state["chart_capacity_selection"]
    assert selection["candidate_order"] == list(range(8, 0, -1))
    assert selection["selection_rule"] == (
        "first_descending_candidate_completing_the_entire_frozen_"
        "intrinsic_context_only_atlas_contract"
    )
    assert selection["intrinsic_reject_codes_in_gate_order"] == list(
        setup.INTRINSIC_REJECT_CODES
    )
    assert selection["full_pass_status"] == "ADMISSIBLE"
    assert selection["only_typed_K_dependent_intrinsic_rejects_descend"]
    assert selection["unexpected_exception_is_terminal"]
    assert selection[
        "selected_atlas_serialization_application_or_transport_failure_"
        "is_terminal_and_cannot_resume_K_search"
    ]
    assert selection["generalized_contributions_forbidden_for_selection"]
    assert selection[
        "B_C_bootstrap_LOFO_or_response_results_forbidden_for_selection"
    ]
    assert selection[
        "early_N_less_than_K_times_rank_plus_one_abort_forbidden"
    ]
    assert selection[
        "smaller_K_descent_continues_after_typed_low_row_reject"
    ]


def test_v2_dynamic_width_and_storage_table_is_exact() -> None:
    contract = setup.build_setup()["dynamic_transport_contract"]
    table = contract["resource_table"]
    assert [row["source_feature_width"] for row in table] == [
        136,
        119,
        102,
        85,
        68,
        51,
        34,
        17,
    ]
    assert [row["all_ordered_coefficient_bytes"] for row in table] == [
        979_025_920,
        856_647_680,
        734_269_440,
        611_891_200,
        489_512_960,
        367_134_720,
        244_756_480,
        122_378_240,
    ]
    assert table[-1]["chart_count"] == 1
    assert table[-1]["applied_store_payload_bytes_per_role"] == 2_571_336_576
    assert contract["selection_streaming_resource_config"] == {
        "row_chunk_size": 256,
        "atlas_chunk_size": 256,
        "max_resident_bytes": 8 * 1024**3,
        "max_artifact_bytes": 4 * 1024**3,
        "exact_payload_bound_by_traced_selection_kernel_receipt": True,
    }


def test_v2_requires_full_post_occupancy_reject_statistics() -> None:
    from fisher_graph import streaming_shared_chart_atlas_selection_v1 as numeric

    trace = setup.build_setup()["selection_receipt"][
        "post_occupancy_gate_sufficient_statistics"
    ]
    assert trace["PASS_evidence_required_for_every_reached_predicate"]
    assert set(trace["gate_evidence_required_fields"]) == {
        member.name
        for member in fields(
            numeric.StreamingSharedChartAtlasGateEvidenceV1
        )
    } | {"schema"}
    assert trace["radius_contract"]["observed_binary64"] == [
        "published_radius_squared"
    ]
    assert trace["radius_contract"][
        "radius_numerator_unfloored_value_and_floor_not_exposed_or_claimed"
    ]
    soft = trace["soft_occupancy_contract"]
    assert soft["minimum_membership_is_per_chart_not_global"]
    assert soft["simplex_effective_bound_binary64"] == "3d819799812dea11"
    covariance = trace["covariance_contract"]
    assert covariance["covariance_bytes_remain_original_and_unsymmetrized"]
    assert covariance["observed_binary64"] == [
        "C_ij",
        "C_ji",
        "absolute_pair_residual",
        "scale_max_1_abs_Cij_abs_Cji",
    ]
    assert covariance["maximum_covariance_residual_not_exposed_or_claimed"]
    eigengap = trace["eigengap_orientation_contract"]
    assert eigengap["selected_atlas_cardinality"] == "K_chart_times_16"
    assert eigengap["partial_spectrum_hash_policy"] == "option_A"
    assert eigengap["spectrum_hashes_only_for_complete_16_row_charts"]
    spd = trace["SPD_contract"]
    assert spd["basis_gram_atol_binary64"] == "3e112e0be826d695"
    assert spd["tangent_variance"][
        "variance_floor_value_not_exposed_or_claimed"
    ]
    assert trace["hash_only_rejected_candidate_diagnostics_forbidden"]


def test_v2_preserves_B_C_and_downgrades_K1_claim() -> None:
    state = setup.build_setup()
    heldout = state["B_C_neighbor_confirmation"]
    assert heldout["mutual_neighbor_k"] == 2
    assert heldout["K_chart_and_k_neighbor_are_distinct"]
    assert heldout["within_family_bootstrap"] == {
        "seed": 7901,
        "resample_count": 10_000,
        "sampling_unit": "whole_example_with_all_target_token_rows",
        "primary_examples_per_family": 7,
        "minimum_mutual_frequency": 0.8,
        "maximum_degenerate_fraction": 0.01,
        "unchanged_from_v1": True,
    }
    assert heldout["leave_one_family_out"]["minimum_mutual_count"] == 5
    assert state["K1_boundary"]["K1_is_mathematically_admissible_with_rank16"]
    assert state["K1_boundary"][
        "multi_chart_local_chart_mixture_or_routing_claim_forbidden"
    ]


def test_v2_discloses_post_A_status_and_contains_no_result() -> None:
    state = setup.build_setup()
    provenance = state["methodology_provenance"]
    assert provenance["v2_is_post_A_informed"]
    assert provenance["v2_is_prospective_to_B_and_C"]
    assert provenance["whole_protocol_preregistered_from_scratch_claimed"] is False
    assert provenance[
        "A_only_K8_diagnostic_result_is_required_prior_evidence_for_workflow"
    ]
    assert provenance["A_only_K8_diagnostic_required_outcome"] == (
        "BALANCED_HARD_MASS_BELOW_MINIMUM"
    )
    assert provenance["A_only_K8_diagnostic_required_terminal_phase"] == (
        "final_hard_mass_gate"
    )
    assert provenance["A_only_K8_diagnostic_result_cannot_select_or_tune_K"]
    assert provenance["v2_independently_reproduces_K8_before_descending"]
    assert provenance["diagnostic_and_v2_K8_semantic_agreement_required"]
    assert provenance["A_only_K8_diagnostic_setup_sha256"] == (
        diagnostic_setup.PINNED_SETUP_SHA256
    )
    assert provenance["A_only_K8_diagnostic_setup_file_sha256"] == (
        diagnostic_setup.PINNED_SETUP_FILE_SHA256
    )
    assert state["implementation_status"]["A_selected_K_known"] is False
    assert all(value is False for value in state["setup_receipt"].values())


def test_v2_live_outputs_are_distinct_and_outside_frozen_artifacts() -> None:
    state = setup.build_setup()
    outputs = state["output_contract"]
    expected_parent = Path(".local-runs/google--gemma-3-270m")
    assert {Path(outputs[name]).parent for name in ("report", "svg", "atlas")} == {
        expected_parent
    }
    assert len({outputs[name] for name in ("report", "svg", "atlas")}) == 3
    assert Path(
        state["methodology_provenance"]["A_only_K8_diagnostic_result_path"]
    ) == setup.A_DIAGNOSTIC_RESULT_PATH
    qualification = state["implementation_sources"][
        "future_implementation_qualification"
    ]
    assert Path(qualification["path"]) == (
        setup.FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT
    )
    assert qualification["must_not_modify_or_regenerate_this_scientific_setup"]


def test_v2_setup_rejects_selection_or_gate_drift() -> None:
    state = copy.deepcopy(setup.build_setup())
    state["chart_capacity_selection"]["candidate_order"] = [8, 4, 2, 1]
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="differs from source"):
        setup.validate_setup(state)


def test_v2_setup_rejects_boolean_integer_substitution() -> None:
    state = copy.deepcopy(setup.build_setup())
    state["format_version"] = True
    state["setup_sha256"] = setup.setup_sha256(state)
    with pytest.raises(ValueError, match="differs from source"):
        setup.validate_setup(state)


def test_v2_setup_roundtrip_no_clobber_and_artifact_pin(tmp_path) -> None:
    output = tmp_path / "v2-setup.json"
    expected = setup.write_setup(output)
    assert setup.load_setup(output) == expected
    with pytest.raises(FileExistsError):
        setup.write_setup(output)
    assert setup.check_setup(setup.DEFAULT_OUTPUT) == setup.build_setup()


def test_v2_setup_cli_requires_explicit_action() -> None:
    with pytest.raises(SystemExit) as error:
        setup.main([])
    assert error.value.code == 2
