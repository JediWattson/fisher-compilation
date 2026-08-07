from __future__ import annotations

import builtins
from copy import deepcopy
import json
import re

import pytest

from fisher_graph.gemma3_l10_l17_trajectory_correction_protocol import (
    FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256,
    build_default_gemma3_l10_l17_trajectory_correction_protocol,
    validate_gemma3_l10_l17_trajectory_correction_protocol,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _walk_mappings(value: object):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_mappings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mappings(item)


def test_protocol_freezes_exact_composition_and_fit_only_authorities() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()

    assert protocol["artifact_sha256"] == (
        FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256
    )
    assert _SHA256.fullmatch(protocol["artifact_sha256"])
    authority = protocol["source_authority"]
    assert authority["model"] == {
        "model_id": "google/gemma-3-270m",
        "requested_revision": (
            "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
        ),
        "adapter_model_fingerprint": (
            "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
        ),
    }
    assert authority["composition_bundle"]["tensor_file_sha256"] == (
        "394906f8e84a50e18922de0dc8c114be1ea9889f0995ccca180b9f6a8d303d8d"
    )
    assert authority["composition_bundle"][
        "composition_payload_sha256"
    ] == "2f7c2179656fc16c614cd84b7a0b29d3250443a5d8c80db221b220e3d3f082bf"
    layer10 = authority["layer10"]
    assert layer10["candidate_tensor_file_sha256"] == (
        "feffc023ba37aee10591cc4313238dd6936181a5e77c5a61d12cfe6be04b8a1b"
    )
    assert layer10["candidate_scientific_payload_sha256"] == (
        "eae90f334b34dc76d7ef38585e394f77825e1514189cecc3e38e36ec3842fcbb"
    )
    assert layer10["primary_graph_sha256"] == (
        "67327f1ba3cff3bd9a49897245d0301d109ac1564eff2c4f70409d29a28a8b94"
    )
    assert layer10["must_remain_bitwise_unchanged"] is True

    fit = authority["calibration_a_fit"]
    assert fit["exclusive_role_use"] == "calibration_a_fit"
    assert fit["example_count"] == 256
    assert fit["family_count"] == 8
    assert fit["family_aliases"] == [
        f"family_{index:02d}" for index in range(8)
    ]
    assert authority["forbidden_roles"] == [
        "calibration_a_selection",
        "calibration_a_guard",
        "calibration_b",
        "validation",
        "test",
    ]


def test_a3_target_is_raw_mlp_correction_not_a1_a2_or_a3_plus() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    target = protocol["target_contract"]

    assert target["target_variant"] == "A3"
    assert target["pairing_key"] == ["example_id", "logical_position"]
    assert target["pairing_requires_identical_token_stream"] is True
    assert target["input"] == (
        "layer10_compiled_layer17_mlp_normalized_input"
    )
    assert target["native_pass"] == "fully_native_model"
    assert target["native_capture"] == "layer17_mlp_operator_output"
    assert target["compiled_keep_pass"] == (
        "authenticated_layer17_compact_mlp_replay_on_captured_"
        "layer10_compiled_normalized_input"
    )
    assert target["compiled_keep_replay_matches_runtime_deletion_operator"] is True
    assert target["algebraic_full_minus_selected_used_as_target"] is False
    assert target[
        "algebraic_full_minus_selected_equivalence_audit_required"
    ] is True
    assert target["raw_target_formula"] == (
        "native_full_layer17_mlp_operator_output-"
        "compiled_keep_layer17_mlp_operator_output"
    )
    assert target["includes_layer17_residual_stream_offset"] is False
    assert target["a3_plus_is_forbidden"] is True
    assert target["target_width"] == 640
    assert target["fisher_weight_source"] == (
        "fully_native_selected_mode_virtual_gate_empirical_fisher_"
        "at_layer17_mlp_down_input"
    )
    assert target["fisher_weight_formula"] == (
        "sum_over_selected_modes((z*dNLL_dz)^2)"
    )
    assert target["fisher_family_normalization"] == (
        "equal_total_mass_per_training_family"
    )


def test_joint_frozen_decoder_projection_has_exact_coordinate_split() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    projection = protocol["projection_contract"]
    candidate = protocol["candidate_contract"]

    assert projection["concatenated_decoder_shape"] == [182, 640]
    assert projection["concatenated_coordinate_width"] == 182
    assert projection["decoder_span_sha256"] == (
        "35cd9f78095855f71eb06660c2c93cee55cb4956e95fad1afaf78d4680aae9cb"
    )
    assert projection["geometry"] == "affine_sum_of_frozen_mode_codecs"
    assert projection["affine_offset_definition"] == (
        "mu_cat=sum_i(frozen_mean_bias_i)"
    )
    assert projection["summed_mean_shape"] == [640]
    assert projection["summed_mean_sha256"] == (
        "f906a94712c4026c03a4eb5aeb62ea188d5987ccdaeba9e5fb6ab4492df9dc1e"
    )
    assert projection["projection_formula"] == (
        "c_star=(r_star-mu_cat)*D_cat_T*pinv(D_cat*D_cat_T)"
    )
    assert projection["reconstruction_formula"] == (
        "r_projected=mu_cat+c_star*D_cat"
    )
    assert projection["joint_projection_required"] is True
    assert projection["independent_generator_fits_after_split"] is True
    assert projection[
        "direct_independent_decoder_projection_forbidden"
    ] is True
    assert projection["pseudoinverse_solver"] == "svd_moore_penrose"
    assert projection["accumulation_dtype"] == "float64"
    assert projection["coordinate_slices"] == [
        {
            "node_name": row["node_name"],
            "start": start,
            "stop": stop,
        }
        for row, start, stop in zip(
            projection["ordered_decoders"],
            (0, 48, 86, 134),
            (48, 86, 134, 182),
            strict=True,
        )
    ]
    assert [row["node_rank"] for row in projection["ordered_decoders"]] == [
        48,
        38,
        48,
        48,
    ]
    assert all(
        _SHA256.fullmatch(row["computational_mode_basis_sha256"])
        and _SHA256.fullmatch(row["mean_bias_sha256"])
        and _SHA256.fullmatch(row["decoder_basis_sha256"])
        for row in projection["ordered_decoders"]
    )
    assert candidate["layer17_node_ranks_in_execution_order"] == [
        48,
        38,
        48,
        48,
    ]
    assert candidate["layer17_generator_rank"] == 16
    assert candidate["layer17_ridge"] == 0.0
    assert candidate["layer17_edge_policy"] == "edgeless"
    assert candidate["decoder_basis_refit_forbidden"] is True
    assert candidate["frozen_decoder_means_required"] is True


def test_lofo_gates_and_resources_are_predeclared() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    folds = protocol["folds"]

    assert len(folds) == 8
    assert [fold["held_family_alias"] for fold in folds] == [
        f"family_{index:02d}" for index in range(8)
    ]
    for index, fold in enumerate(folds):
        assert fold["fold_index"] == index
        assert fold["held_example_count"] == 32
        assert fold["training_example_count"] == 224
        assert len(fold["training_family_aliases"]) == 7
        assert fold["held_family_alias"] not in fold[
            "training_family_aliases"
        ]
        assert _SHA256.fullmatch(fold["held_membership_sha256"])
        assert _SHA256.fullmatch(fold["training_membership_sha256"])
        assert _SHA256.fullmatch(fold["artifact_sha256"])

    assert protocol["gates"] == {
        "decision_policy": "all_required_gates_must_pass",
        "required_completed_fold_count": 8,
        "maximum_failed_fold_count": 0,
        "maximum_family_macro_delta_nll_per_token": 0.08,
        "maximum_worst_family_delta_nll_per_token": 0.1,
        "maximum_family_macro_native_to_candidate_kl_per_token": 0.09,
        "minimum_family_macro_top1_agreement_to_native": 0.84,
        "maximum_family_macro_interaction_excess_nll": 0.01,
        "require_strict_family_macro_kl_improvement_vs_frozen": True,
        "minimum_held_family_kl_improvement_count": 6,
        "required_held_family_count": 8,
        "maximum_family_macro_nll_regression_vs_frozen": 0.001,
        "maximum_family_macro_top1_regression_vs_frozen": 0.005,
        "minimum_family_macro_deletion_nll_recovery_fraction": 0.6,
        "minimum_worst_family_deletion_nll_recovery_fraction": 0.4,
        "require_exact_resources": True,
        "require_exact_projection_metadata": True,
        "require_compact_replay_algebraic_equivalence_audit": True,
        "require_source_model_unchanged": True,
        "require_layer10_unchanged": True,
        "permit_latency_or_kernel_speed_claim": False,
    }
    assert protocol["candidate_contract"]["exact_resources"] == {
        "source_whole_model_learned_parameters": 268_098_176,
        "replaced_layer_count": 2,
        "graph_node_count": 8,
        "dynamic_interaction_count": 3,
        "native_removed_parameters": 1_082_880,
        "primary_graph_parameters": 295_129,
        "net_stored_parameter_savings": 787_751,
        "candidate_whole_model_learned_parameters": 267_310_425,
        "dense_graph_macs_per_token": 289_600,
        "executed_graph_macs_per_token": 286_784,
        "native_removed_macs_per_token": 1_082_880,
        "net_executed_macs_saved_per_token": 796_096,
    }


def test_protocol_is_pure_source_safe_and_never_opens_a_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("the pure protocol attempted filesystem access")

    monkeypatch.setattr(builtins, "open", forbid_open)
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    safety = protocol["safety"]

    assert all(value is False for value in safety.values())
    forbidden_identity_fields = {
        "prompts",
        "prompt_text",
        "token_ids",
        "ordered_prompt_sha256s",
        "ordered_family_ids",
        "ordered_members",
        "identity_sha256",
    }
    assert not any(
        forbidden_identity_fields & set(mapping)
        for mapping in _walk_mappings(protocol)
    )
    serialized = json.dumps(protocol)
    assert "structured-strong-v8-calibration_a-" not in serialized
    assert '"calibration_a_selection", "manifest_sha256"' not in serialized


def test_json_round_trip_and_tampering_fail_closed() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    round_tripped = json.loads(json.dumps(protocol))

    assert (
        validate_gemma3_l10_l17_trajectory_correction_protocol(round_tripped)
        == protocol
    )

    changed_target = deepcopy(protocol)
    changed_target["target_contract"][
        "includes_layer17_residual_stream_offset"
    ] = True
    with pytest.raises(ValueError, match="differs from frozen plan"):
        validate_gemma3_l10_l17_trajectory_correction_protocol(changed_target)

    changed_decoder = deepcopy(protocol)
    changed_decoder["projection_contract"]["ordered_decoders"][0][
        "decoder_basis_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="differs from frozen plan"):
        validate_gemma3_l10_l17_trajectory_correction_protocol(changed_decoder)

    changed_resource = deepcopy(protocol)
    changed_resource["candidate_contract"]["exact_resources"][
        "primary_graph_parameters"
    ] += 1
    with pytest.raises(ValueError, match="differs from frozen plan"):
        validate_gemma3_l10_l17_trajectory_correction_protocol(
            changed_resource
        )

    changed_hash = deepcopy(protocol)
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_gemma3_l10_l17_trajectory_correction_protocol(changed_hash)

    extra_field = deepcopy(protocol)
    extra_field["result"] = "pass"
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_gemma3_l10_l17_trajectory_correction_protocol(extra_field)


def test_claim_boundary_is_fit_only_and_does_not_overstate_result() -> None:
    protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    boundary = protocol["claim_boundary"]

    assert boundary["scientific_role"] == (
        "calibration_a_fit_adaptive_trajectory_correction_development"
    )
    assert boundary["supports_family_blocked_internal_estimate"] is True
    assert boundary["supports_heldout_confirmation"] is False
    assert boundary["supports_serving_authorization"] is False
    assert boundary["supports_whole_model_compiled_claim"] is False
    assert boundary["full_model_logits_may_be_scored"] is True
    assert boundary["compiled_layer_count"] == 2
    assert boundary["supports_lossless_claim"] is False
    assert boundary["supports_latency_or_kernel_speed_claim"] is False
    assert boundary[
        "selection_guard_b_validation_and_test_remain_untouched"
    ] is True
