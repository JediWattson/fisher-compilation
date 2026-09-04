from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from fisher_graph import gemma3_attention_transport_full_model_partition as full


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _members(*, receipt_prefix: str | None = None) -> tuple[dict[str, object], ...]:
    members: list[dict[str, object]] = []
    for family in range(6):
        for target in full.A_TARGET_TOKEN_COUNTS:
            member: dict[str, object] = {
                "canonical_prompt_sha256": _sha(
                    f"context-family-{family}-length-{target}"
                ),
                "family_id": f"family.{family}",
                "objective_row_stream_sha256": _sha(
                    f"objective-family-{family}-length-{target}"
                ),
                "target_token_count": target,
            }
            if receipt_prefix is not None:
                member["runtime_execution_receipt_sha256"] = _sha(
                    f"{receipt_prefix}-{family}-{target}"
                )
            members.append(member)
    return tuple(members)


def _split(
    *,
    panel_label: str = "A-panel",
    receipt_prefix: str | None = None,
) -> full.FullModelStableASplit:
    return full.build_full_model_stable_A_split(
        role_panel_binding_sha256=_sha(panel_label),
        ordered_members=_members(receipt_prefix=receipt_prefix),
    )


def _protocol(
    *, panel_label: str = "A-panel"
) -> full.AttentionTransportFullModelProtocol:
    return full.build_attention_transport_full_model_protocol(
        _split(panel_label=panel_label)
    )


def _firewall(
    protocol: full.AttentionTransportFullModelProtocol,
    **overrides: object,
) -> full.FullModelFirewallReceipt:
    values: dict[str, object] = {
        "protocol_sha256": protocol.protocol_sha256,
        "candidate_artifact_sha256": _sha("candidate"),
        "candidate_frozen_before_outer_A_dev": True,
        "heldout_teacher_in_candidate_lineage": False,
        "heldout_teacher_reads_before_freeze": 0,
        "outer_A_dev_reads_before_freeze": 0,
        "outer_A_dev_opened_after_freeze": True,
        "B_guard_opened_before_A_pass": False,
        "B_guard_reads_before_A_pass": 0,
        "C_assessment_opened_before_B_pass": False,
        "C_assessment_reads_before_B_pass": 0,
        "refit_after_sealed_read": False,
        "reselection_after_sealed_read": False,
        "oracle_used_in_full_model_trajectory": False,
        "native_internal_states_used_in_candidate_trajectory": False,
        "free_layer_parameter_count": 0,
        "replaced_value_output_site_count": 18,
        "native_value_output_parameters_in_executor": 0,
        "teacher_payload_values_in_executor": 0,
        "control_permutation_frozen_before_outer_A_dev": True,
        "candidate_published_before_decision": False,
    }
    values.update(overrides)
    return full.FullModelFirewallReceipt(**values)  # type: ignore[arg-type]


def _measurements(**overrides: object) -> full.FullModelGateMeasurements:
    values: dict[str, object] = {
        "held_block_relative_fisher_loss": 0.08,
        "mutation_23_relative_fisher_loss": 0.09,
        "held_block_oracle_relative_fisher_loss": 0.07,
        "delta_nll_per_token": 0.07,
        "kl_per_token": 0.08,
        "top1_agreement": 0.92,
        "real_gain_over_zeroed_control": 0.20,
        "permuted_control_gain": 0.04,
    }
    values.update(overrides)
    return full.FullModelGateMeasurements(**values)  # type: ignore[arg-type]


def test_import_adds_no_tensor_or_model_runtime_modules() -> None:
    code = """
import sys
import fisher_graph
baseline = set(sys.modules)
import fisher_graph.gemma3_attention_transport_full_model_partition
new_modules = set(sys.modules) - baseline
assert 'torch' not in new_modules
assert 'transformers' not in new_modules
assert 'mlx' not in new_modules
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_fixed_layer_catalog_and_zero_shot_partition_are_exact() -> None:
    assert len(full.ATTENTION_LAYER_CATALOG) == 18
    assert tuple(site.layer_ordinal for site in full.ATTENTION_LAYER_CATALOG) == tuple(
        range(18)
    )
    assert tuple(site.invocation_id for site in full.ATTENTION_LAYER_CATALOG) == (
        full.INVOCATION_IDS
    )
    assert tuple(
        site.invocation_id
        for site in full.ATTENTION_LAYER_CATALOG
        if site.macroblock_ordinal in full.ENDPOINT_MACROBLOCK_ORDINALS
    ) == full.ENDPOINT_TRAIN_INVOCATION_IDS
    assert tuple(
        site.invocation_id
        for site in full.ATTENTION_LAYER_CATALOG
        if site.macroblock_ordinal == full.MIDDLE_HOLDOUT_MACROBLOCK_ORDINAL
    ) == full.MIDDLE_HOLDOUT_INVOCATION_IDS
    assert set(full.ENDPOINT_TRAIN_INVOCATION_IDS).isdisjoint(
        full.MIDDLE_HOLDOUT_INVOCATION_IDS
    )
    assert set(full.ENDPOINT_TRAIN_INVOCATION_IDS) | set(
        full.MIDDLE_HOLDOUT_INVOCATION_IDS
    ) == set(full.INVOCATION_IDS)
    assert tuple(
        site.invocation_id
        for site in full.ATTENTION_LAYER_CATALOG
        if site.is_global_attention
    ) == full.GLOBAL_ATTENTION_INVOCATION_IDS
    assert full.HELD_GLOBAL_INVOCATION_ID == "mutation.23"
    held_global = full.attention_layer_site(full.HELD_GLOBAL_INVOCATION_ID)
    assert held_global.macroblock_ordinal == 1
    assert held_global.repeated_site_ordinal == 5
    assert held_global.is_global_attention
    assert len(full.ATTENTION_LAYER_CATALOG_SHA256) == 64

    with pytest.raises(ValueError, match="catalog entry drifted"):
        full.AttentionLayerSite(
            layer_ordinal=6,
            invocation_id="mutation.13",
            macroblock_ordinal=0,
            repeated_site_ordinal=0,
            attention_kind="local",
        )


def test_no_free_layer_feature_contract_is_fixed_and_repeated_across_blocks() -> None:
    contract = full.DEFAULT_NO_FREE_LAYER_FEATURE_ENCODING_CONTRACT
    assert contract.feature_names == full.NO_FREE_LAYER_FEATURE_NAMES
    assert contract.feature_count == 9
    assert contract.learned_layer_embedding_count == 0
    assert contract.free_per_layer_parameter_count == 0
    assert contract.includes_layer_one_hot is False
    assert contract.includes_macroblock_one_hot is False

    encodings = {
        site.invocation_id: contract.encode(site.invocation_id)
        for site in full.ATTENTION_LAYER_CATALOG
    }
    assert encodings["mutation.1"][0] == -1.0
    assert encodings["mutation.35"][0] == 1.0
    for site in full.ATTENTION_LAYER_CATALOG:
        encoded = encodings[site.invocation_id]
        assert len(encoded) == 9
        assert sum(encoded[1:7]) == 1.0
        assert encoded[1 + site.repeated_site_ordinal] == 1.0
        assert encoded[-2:] == (
            (0.0, 1.0) if site.is_global_attention else (1.0, 0.0)
        )

    # Same repeated-site and attention features across all macroblocks; only
    # the fixed analytic depth coordinate may differ.
    for repeated_site in range(6):
        rows = [
            encodings[full.INVOCATION_IDS[block * 6 + repeated_site]]
            for block in range(3)
        ]
        assert rows[0][1:] == rows[1][1:] == rows[2][1:]

    with pytest.raises(FrozenInstanceError):
        contract.free_per_layer_parameter_count = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replace(contract, artifact_sha256=_sha("forged-feature-contract"))


def test_stable_A_split_is_nested_disjoint_and_ignores_runtime_receipt_fields() -> None:
    baseline = _split()
    first_receipts = _split(receipt_prefix="first")
    second_receipts = _split(receipt_prefix="second")
    assert baseline == first_receipts == second_receipts
    assert len(baseline.direction_train_batch_identity_sha256s) == 30
    assert len(baseline.allocation_validation_batch_identity_sha256s) == 6
    assert len(baseline.outer_fit_batch_identity_sha256s) == 36
    assert len(baseline.outer_dev_batch_identity_sha256s) == 9
    assert set(baseline.direction_train_batch_identity_sha256s).isdisjoint(
        baseline.allocation_validation_batch_identity_sha256s
    )
    assert set(baseline.direction_train_batch_identity_sha256s) | set(
        baseline.allocation_validation_batch_identity_sha256s
    ) == set(baseline.outer_fit_batch_identity_sha256s)
    assert set(baseline.outer_fit_batch_identity_sha256s).isdisjoint(
        baseline.outer_dev_batch_identity_sha256s
    )
    assert baseline.state_dict()["contains_runtime_execution_receipts"] is False
    serialized = json.dumps(baseline.state_dict(), sort_keys=True)
    assert "runtime_execution_receipt_sha256" not in serialized
    assert baseline == _split()

    changed = _split(panel_label="changed-A-panel")
    assert changed.catalog.batch_plan_artifact_sha256 != (
        baseline.catalog.batch_plan_artifact_sha256
    )
    assert changed.artifact_sha256 != baseline.artifact_sha256


def test_stable_A_split_hash_and_nested_source_binding_reject_tampering() -> None:
    split = _split()
    with pytest.raises(ValueError, match="nested split drifted"):
        replace(split, inner_split_artifact_sha256=_sha("forged-inner"))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replace(split, artifact_sha256=_sha("forged-split"))


def test_protocol_binds_split_structure_features_budgets_and_gates() -> None:
    split = _split()
    protocol = full.build_attention_transport_full_model_protocol(split)
    state = protocol.state_dict()
    assert state["schema"] == full.FULL_MODEL_PROTOCOL_SCHEMA
    assert state["endpoint_train_invocation_ids"] == (
        full.ENDPOINT_TRAIN_INVOCATION_IDS
    )
    assert state["middle_holdout_invocation_ids"] == (
        full.MIDDLE_HOLDOUT_INVOCATION_IDS
    )
    assert state["generator_count"] == 8
    assert state["generator_rank"] == 8
    assert state["active_generator_count"] == 2
    assert state["all_18_value_output_sites_replaced"] is True
    assert state["candidate_trajectory_execution"] is True
    assert state["free_per_layer_parameter_count"] == 0
    assert state["maximum_generator_bank_parameters"] == 134_208
    assert state["maximum_candidate_parameters"] == 186_049
    assert state["independent_baseline_parameter_cap"] == 186_049
    assert state["native_value_output_parameters"] == 14_745_600
    assert protocol.maximum_parameter_reduction_fraction == pytest.approx(
        0.9873827446831597
    )
    assert protocol.stable_A_split_artifact_sha256 == split.artifact_sha256
    protocol.validate_integrity()
    protocol.validate_against_split(split)

    assert _protocol(panel_label="other-panel").protocol_sha256 != (
        protocol.protocol_sha256
    )
    with pytest.raises(FrozenInstanceError):
        protocol.feature_contract_sha256 = _sha("new")  # type: ignore[misc]
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        replace(protocol, protocol_sha256=_sha("forged-protocol"))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_frozen_before_outer_A_dev", False),
        ("heldout_teacher_in_candidate_lineage", True),
        ("heldout_teacher_reads_before_freeze", 1),
        ("outer_A_dev_reads_before_freeze", 1),
        ("outer_A_dev_opened_after_freeze", False),
        ("B_guard_opened_before_A_pass", True),
        ("B_guard_reads_before_A_pass", 1),
        ("C_assessment_opened_before_B_pass", True),
        ("C_assessment_reads_before_B_pass", 1),
        ("refit_after_sealed_read", True),
        ("reselection_after_sealed_read", True),
        ("oracle_used_in_full_model_trajectory", True),
        ("native_internal_states_used_in_candidate_trajectory", True),
        ("free_layer_parameter_count", 1),
        ("replaced_value_output_site_count", 17),
        ("native_value_output_parameters_in_executor", 1),
        ("teacher_payload_values_in_executor", 1),
        ("control_permutation_frozen_before_outer_A_dev", False),
        ("candidate_published_before_decision", True),
    ),
)
def test_firewall_fails_closed_for_every_forbidden_read_or_content(
    field: str,
    value: object,
) -> None:
    protocol = _protocol()
    clean = _firewall(protocol)
    assert clean.passed
    unsafe = _firewall(protocol, **{field: value})
    assert not unsafe.passed


def test_firewall_is_hash_bound_and_strictly_typed() -> None:
    protocol = _protocol()
    firewall = _firewall(protocol)
    firewall.validate_integrity()
    assert firewall.state_dict()["passed"] is True
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replace(firewall, artifact_sha256=_sha("forged-firewall"))
    with pytest.raises(ValueError, match="nonnegative integer"):
        _firewall(protocol, heldout_teacher_reads_before_freeze=True)
    with pytest.raises(TypeError, match="must be a bool"):
        _firewall(protocol, refit_after_sealed_read=0)


def test_gate_threshold_boundaries_pass_and_decision_is_recomputed() -> None:
    protocol = _protocol()
    firewall = _firewall(protocol)
    measurements = _measurements(
        held_block_relative_fisher_loss=0.10,
        mutation_23_relative_fisher_loss=0.10,
        held_block_oracle_relative_fisher_loss=0.08,
        delta_nll_per_token=0.10,
        kl_per_token=0.10,
        top1_agreement=0.90,
        real_gain_over_zeroed_control=0.20,
        permuted_control_gain=0.05,
    )
    decision = full.evaluate_full_model_gate_decision(
        protocol=protocol,
        firewall=firewall,
        measurements=measurements,
    )
    assert decision.passed
    assert decision.failed_gates == ()
    assert measurements.router_oracle_relative_fisher_loss_gap == pytest.approx(
        0.02
    )
    assert measurements.permuted_control_gain_fraction == pytest.approx(0.25)
    decision.validate_against(protocol)
    assert decision.state_dict()["measurements"]["top1_agreement"] == 0.90


@pytest.mark.parametrize(
    ("measurement_overrides", "expected_gate"),
    (
        (
            {
                "held_block_relative_fisher_loss": 0.100001,
                "held_block_oracle_relative_fisher_loss": 0.081,
            },
            "held_block_relative_fisher_loss",
        ),
        (
            {"mutation_23_relative_fisher_loss": 0.100001},
            "mutation_23_relative_fisher_loss",
        ),
        (
            {
                "held_block_relative_fisher_loss": 0.09,
                "held_block_oracle_relative_fisher_loss": 0.069999,
            },
            "router_oracle_relative_fisher_loss_gap",
        ),
        ({"delta_nll_per_token": 0.100001}, "delta_nll_per_token"),
        ({"kl_per_token": 0.100001}, "kl_per_token"),
        ({"top1_agreement": 0.899999}, "top1_agreement"),
        (
            {"permuted_control_gain": 0.050001},
            "permuted_control_gain_fraction",
        ),
    ),
)
def test_each_numeric_gate_fails_independently(
    measurement_overrides: dict[str, object],
    expected_gate: str,
) -> None:
    protocol = _protocol()
    decision = full.evaluate_full_model_gate_decision(
        protocol=protocol,
        firewall=_firewall(protocol),
        measurements=_measurements(**measurement_overrides),
    )
    assert not decision.passed
    assert decision.failed_gates == (expected_gate,)


def test_control_and_firewall_gates_fail_closed() -> None:
    protocol = _protocol()
    no_gain = full.evaluate_full_model_gate_decision(
        protocol=protocol,
        firewall=_firewall(protocol),
        measurements=_measurements(
            real_gain_over_zeroed_control=0.0,
            permuted_control_gain=-0.1,
        ),
    )
    assert no_gain.failed_gates == (
        "real_gain_over_zeroed_control",
        "permuted_control_gain_fraction",
    )

    negative_permuted = full.evaluate_full_model_gate_decision(
        protocol=protocol,
        firewall=_firewall(protocol),
        measurements=_measurements(permuted_control_gain=-1.0),
    )
    assert negative_permuted.passed

    unsafe = full.evaluate_full_model_gate_decision(
        protocol=protocol,
        firewall=_firewall(protocol, heldout_teacher_reads_before_freeze=1),
        measurements=_measurements(),
    )
    assert unsafe.failed_gates == ("firewall",)


def test_measurements_reject_impossible_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="below the exact oracle"):
        _measurements(
            held_block_relative_fisher_loss=0.07,
            held_block_oracle_relative_fisher_loss=0.08,
        )
    with pytest.raises(ValueError, match="must lie"):
        _measurements(top1_agreement=1.01)
    with pytest.raises(ValueError, match="must be nonnegative"):
        _measurements(kl_per_token=-0.01)
    with pytest.raises(ValueError, match="must be finite"):
        _measurements(delta_nll_per_token=float("nan"))


def test_decision_rejects_lineage_pass_bit_gate_inventory_and_hash_tampering() -> None:
    protocol = _protocol()
    firewall = _firewall(protocol)
    measurements = _measurements()
    decision = full.evaluate_full_model_gate_decision(
        protocol=protocol,
        firewall=firewall,
        measurements=measurements,
    )
    with pytest.raises(ValueError, match="pass bit drifted"):
        replace(decision, passed=False)
    with pytest.raises(ValueError, match="failed-gate inventory drifted"):
        replace(decision, failed_gates=("delta_nll_per_token",))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replace(decision, artifact_sha256=_sha("forged-decision"))

    other_protocol = _protocol(panel_label="other-panel")
    with pytest.raises(ValueError, match="protocol binding drifted"):
        decision.validate_against(other_protocol)
    with pytest.raises(ValueError, match="lineage drifted"):
        full.evaluate_full_model_gate_decision(
            protocol=protocol,
            firewall=_firewall(other_protocol),
            measurements=measurements,
        )
