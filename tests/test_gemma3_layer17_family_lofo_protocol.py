from __future__ import annotations

import builtins
from copy import deepcopy
import json
import re

import pytest

from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    V8_FAMILY_LOFO_ROLES,
    build_authenticated_v8_layer17_family_lofo_protocol,
    build_default_v8_layer17_family_lofo_protocol,
    validate_v8_layer17_family_lofo_protocol,
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


def test_default_protocol_freezes_roles_folds_and_first_arm() -> None:
    protocol = build_default_v8_layer17_family_lofo_protocol()

    assert protocol["artifact_sha256"] == (
        FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
    )
    assert _SHA256.fullmatch(protocol["artifact_sha256"])
    assert V8_FAMILY_LOFO_ROLES == (
        "calibration_a_fit",
        "calibration_a_selection",
        "calibration_a_guard",
    )
    role_bindings = protocol["role_bindings"]
    assert role_bindings["fit"]["role"] == "calibration_a_fit"
    assert role_bindings["fit"]["family_aliases"] == list(
        V8_FAMILY_LOFO_FAMILY_ALIASES
    )
    assert _SHA256.fullmatch(
        role_bindings["fit"]["family_alias_mapping_sha256"]
    )
    assert (
        role_bindings["fit"]["family_alias_mapping_sha256"]
        == protocol["corpus_authority"]["family_alias_mapping_sha256"]
    )
    assert role_bindings["open_development_assessment"]["role"] == (
        "calibration_a_selection"
    )
    assert role_bindings["open_development_assessment"][
        "historical_status"
    ] == "open_development"
    assert role_bindings["sealed_guard"]["role"] == "calibration_a_guard"
    assert role_bindings["sealed_guard"]["must_remain_sealed"] is True
    assert role_bindings["forbidden_external_roles"] == [
        "calibration_b",
        "validation",
        "test",
    ]

    arm = protocol["first_arm"]
    assert arm["arm_id"] == "cap48-r16-edgeless-v1"
    assert arm["layer_ordinal"] == 17
    assert arm["mode_rank_cap"] == 48
    assert arm["resolved_node_ranks_in_execution_order"] == [48, 38, 48, 48]
    assert arm["generator_rank"] == 16
    assert arm["edge_policy"] == "edgeless"
    assert arm["interaction_count"] == 0
    assert arm["ridge"] == 0.0

    folds = protocol["folds"]
    assert len(folds) == 8
    assert tuple(fold["held_family_alias"] for fold in folds) == (
        V8_FAMILY_LOFO_FAMILY_ALIASES
    )
    for index, fold in enumerate(folds):
        held_family = V8_FAMILY_LOFO_FAMILY_ALIASES[index]
        assert fold["fold_index"] == index
        assert fold["fold_id"] == f"family-{index + 1:02d}-of-08"
        assert fold["held_example_count"] == 32
        assert fold["training_example_count"] == 224
        assert fold["training_family_aliases"] == [
            family
            for family in V8_FAMILY_LOFO_FAMILY_ALIASES
            if family != held_family
        ]
        assert _SHA256.fullmatch(fold["held_membership_sha256"])
        assert _SHA256.fullmatch(fold["training_membership_sha256"])
        assert _SHA256.fullmatch(fold["artifact_sha256"])


def test_gates_are_predeclared_and_cannot_be_selected_after_results() -> None:
    protocol = build_default_v8_layer17_family_lofo_protocol()
    gates = protocol["gates"]
    evaluation = protocol["evaluation_contract"]

    assert gates == {
        "decision_policy": "all_required_gates_must_pass",
        "required_completed_fold_count": 8,
        "maximum_failed_fold_count": 0,
        "maximum_family_macro_delta_nll_per_token": 0.075,
        "maximum_worst_family_delta_nll_per_token": 0.1,
        "maximum_family_macro_native_to_candidate_kl_per_token": 0.06,
        "minimum_family_macro_top1_agreement_to_native": 0.875,
        "minimum_family_macro_deletion_nll_recovery_fraction": 0.6,
        "minimum_worst_family_deletion_nll_recovery_fraction": 0.4,
        "require_positive_exact_parameter_savings": True,
        "require_positive_logical_mac_savings": True,
        "permit_latency_or_kernel_speed_claim": False,
    }
    assert evaluation["fold_local_refit_required"] is True
    assert evaluation["reuse_fitted_parameters_across_folds"] is False
    assert evaluation["arm_selection_inside_lofo"] is False
    assert evaluation[
        "held_family_used_for_fitting_or_early_stopping"
    ] is False
    assert evaluation["deletion_control_required"] is True
    assert evaluation["worst_family_metrics_required"] is True
    assert evaluation["deletion_nll_recovery"] == {
        "candidate_condition": "lofo_refit",
        "control_condition": "matched_deletion",
        "formula": (
            "(matched_deletion_delta_nll_per_token-"
            "lofo_refit_delta_nll_per_token)/"
            "matched_deletion_delta_nll_per_token"
        ),
        "denominator": "matched_deletion_delta_nll_per_token",
        "denominator_requirement": "strictly_greater_than_zero",
        "invalid_denominator_policy": "fail_closed",
        "macro_aggregation": "equal_family_arithmetic_mean",
        "worst_aggregation": "minimum_over_held_families",
    }


def test_protocol_is_source_safe_and_never_opens_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbid_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("the pure planner attempted filesystem access")

    monkeypatch.setattr(builtins, "open", _forbid_open)
    protocol = build_default_v8_layer17_family_lofo_protocol()
    safety = protocol["safety"]

    for key in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_per_example_identity",
        "contains_activation_or_gradient_tensors",
        "contains_model_or_candidate_weights",
        "role_input_file_opened",
        "model_or_tokenizer_accessed",
        "model_executed",
        "metrics_read",
        "calibration_a_selection_opened_by_planner",
        "calibration_a_guard_opened",
        "calibration_b_opened",
        "validation_opened",
        "test_opened",
    ):
        assert safety[key] is False

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
    assert "structured-strong-v8-calibration_a-" not in json.dumps(protocol)
    assert max(
        len(item)
        for mapping in _walk_mappings(protocol)
        for item in mapping.values()
        if isinstance(item, list)
    ) <= 8


def test_protocol_json_round_trip_is_exact_and_tampering_fails_closed() -> None:
    protocol = build_default_v8_layer17_family_lofo_protocol()
    round_tripped = json.loads(json.dumps(protocol))

    assert validate_v8_layer17_family_lofo_protocol(round_tripped) == protocol
    assert (
        build_default_v8_layer17_family_lofo_protocol()["artifact_sha256"]
        == protocol["artifact_sha256"]
    )

    changed_gate = deepcopy(protocol)
    changed_gate["gates"][
        "maximum_family_macro_delta_nll_per_token"
    ] = 0.076
    with pytest.raises(ValueError, match="differs from frozen plan"):
        validate_v8_layer17_family_lofo_protocol(changed_gate)

    changed_membership = deepcopy(protocol)
    changed_membership["folds"][0]["held_membership_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from frozen plan"):
        validate_v8_layer17_family_lofo_protocol(changed_membership)

    changed_hash = deepcopy(protocol)
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_v8_layer17_family_lofo_protocol(changed_hash)

    extra_field = deepcopy(protocol)
    extra_field["result"] = "pass"
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_v8_layer17_family_lofo_protocol(extra_field)


def test_authenticated_builder_accepts_only_an_in_memory_frozen_corpus() -> None:
    with pytest.raises(ValueError, match="fields are invalid"):
        build_authenticated_v8_layer17_family_lofo_protocol({})
    with pytest.raises(ValueError, match="fields are invalid"):
        build_authenticated_v8_layer17_family_lofo_protocol(  # type: ignore[arg-type]
            "corpus.json"
        )


def test_claim_boundary_does_not_overstate_the_lofo_protocol() -> None:
    boundary = build_default_v8_layer17_family_lofo_protocol()[
        "claim_boundary"
    ]

    assert boundary["supports_family_blocked_internal_estimate"] is True
    assert boundary["supports_heldout_confirmation"] is False
    assert boundary["supports_lossless_claim"] is False
    assert boundary["supports_compression_claim_before_gates"] is False
    assert boundary[
        "guard_and_external_assessment_roles_remain_untouched"
    ] is True
