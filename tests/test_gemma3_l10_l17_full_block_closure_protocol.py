from __future__ import annotations

import copy

import pytest

from fisher_graph.gemma3_l10_l17_full_block_closure_protocol import (
    FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256,
    build_default_gemma3_l10_l17_full_block_closure_protocol,
    validate_gemma3_l10_l17_full_block_closure_protocol,
)


def test_default_full_block_protocol_is_frozen_and_source_safe() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()

    assert (
        protocol["artifact_sha256"]
        == FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256
    )
    assert validate_gemma3_l10_l17_full_block_closure_protocol(protocol) == (
        protocol
    )
    assert len(protocol["folds"]) == 8
    assert protocol["safety"]["filesystem_accessed"] is False
    assert protocol["safety"]["metrics_read"] is False
    assert protocol["source_authority"]["forbidden_roles"] == [
        "calibration_a_selection",
        "calibration_a_guard",
        "calibration_b",
        "validation",
        "test",
    ]


def test_full_block_target_and_runtime_boundary_are_postnorm_exact() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()
    target = protocol["target_contract"]
    projection = protocol["projection_contract"]
    candidate = protocol["candidate_contract"]

    assert target["target_variant"] == "A4_full_block_closure"
    assert target["raw_target_formula"] == (
        "native_layer17_block_output-"
        "compiled_layer17_post_attention_residual-"
        "compiled_keep_layer17_post_feedforward_delta"
    )
    assert target["generator_application_boundary"] == "layer.17.mlp.delta"
    assert target["pre_norm_target_inversion_forbidden"] is True
    assert target["includes_layer17_residual_stream_offset"] is True
    assert projection["source_decoder_output_boundary"] == (
        "layer.17.mlp.operator_output"
    )
    assert projection["runtime_decoder_output_boundary"] == (
        "layer.17.mlp.delta"
    )
    assert projection["decoder_tensor_values_preserved_exactly"] is True
    assert projection["relocated_basis_binding_required"] is True
    assert candidate["runtime_application_boundary"] == "layer.17.mlp.delta"
    assert candidate["learned_capacity_unchanged_from_a3"] is True


def test_full_block_protocol_binds_failed_a3_without_opening_other_roles() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()
    evidence = protocol["source_authority"]["prior_a3_evidence"]
    evaluation = protocol["evaluation_contract"]

    assert evidence["all_required_gates_pass"] is False
    assert evidence["report_sha256"] == (
        "b90949076c560efa17181a5b47c6191d9358dd5b3d6b5c4e2ba57d6937bdd9ca"
    )
    assert evaluation["eligible_condition"] == (
        "a4_full_block_corrected_composition"
    )
    assert evaluation["interaction_factorial"]["diagnostic_only"] is True
    assert (
        evaluation["interaction_factorial"][
            "may_select_or_mutate_interactions"
        ]
        is False
    )
    assert protocol["gates"][
        "require_strict_family_macro_kl_improvement_vs_prior_a3"
    ] is True


def test_full_block_protocol_rejects_any_mutation() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()
    changed = copy.deepcopy(protocol)
    changed["target_contract"]["generator_application_boundary"] = (
        "layer.17.mlp.operator_output"
    )

    with pytest.raises(ValueError, match="differs from frozen plan"):
        validate_gemma3_l10_l17_full_block_closure_protocol(changed)


def test_full_block_protocol_rejects_hash_mutation() -> None:
    protocol = build_default_gemma3_l10_l17_full_block_closure_protocol()
    protocol["artifact_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_gemma3_l10_l17_full_block_closure_protocol(protocol)
