from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph import gemma3_heterogeneous_generator_family_compiler as family
from fisher_graph.heterogeneous_generator_atom_bank import (
    HeterogeneousGeneratorExecutionReceipt,
    LIFT_SCOPE_COMPATIBLE_ATTENTION,
    LIFT_SCOPE_FULL_HETEROGENEOUS,
    LIFT_SCOPE_MLP_FAMILY,
)
from test_gemma3_causal_carrier_transport_compiler import (
    _sequence,
    _transport_executor,
)


_SCOPES = (
    LIFT_SCOPE_MLP_FAMILY,
    LIFT_SCOPE_COMPATIBLE_ATTENTION,
    LIFT_SCOPE_FULL_HETEROGENEOUS,
)


def _compilation() -> family.Gemma3HeterogeneousGeneratorCompilation:
    return family.compile_gemma3_exact_heterogeneous_generator_plan(
        _transport_executor(),
        require_pinned_gemma=False,
    )


def test_three_exact_lift_manifests_preserve_one_physical_inventory() -> None:
    compilation = _compilation()
    plan = compilation.plan
    manifests = plan.manifests

    assert plan.exact_gate_chain == family.EXACT_LIFT_GATE_CHAIN
    assert tuple(item.lift_scope for item in manifests) == _SCOPES
    assert tuple(len(item.families) for item in manifests) == (22, 6, 5)
    assert tuple(
        sum(row.component_kind != "terminal" for row in item.families)
        for item in manifests
    ) == (21, 5, 4)

    reference = manifests[0]
    assert len(reference.invocations) == 39
    assert len(reference.edge_occurrences) == 75
    assert len(reference.atom_banks) == 4
    assert len(reference.atoms) == 37
    assert reference.invocations[0].invocation_id == "lexical.ingress"
    assert reference.invocations[1].invocation_id == "mutation.1"
    assert reference.invocations[-2].invocation_id == "terminal.norm_softcap"
    assert reference.invocations[-1].invocation_id == "lexical.egress"
    assert reference.edge_occurrences[0].occurrence_id == (
        "occurrence.lexical.ingress"
    )
    assert reference.edge_occurrences[1].occurrence_id == (
        "occurrence.carry.mutation.1"
    )
    assert reference.edge_occurrences[2].occurrence_id == (
        "occurrence.primary.mutation.1"
    )
    assert reference.edge_occurrences[-2].occurrence_id == (
        "occurrence.terminal.norm_softcap"
    )
    assert reference.edge_occurrences[-1].occurrence_id == (
        "occurrence.lexical.egress"
    )
    assert all(item.carrier_manifest.correction_count == 0 for item in manifests)
    assert all(item.compression_attempted is False for item in manifests)
    assert all(item.deployed_scalar_count == item.source_stored_scalar_count for item in manifests)
    assert all(item.atoms == reference.atoms for item in manifests)
    assert all(item.atom_banks == reference.atom_banks for item in manifests)

    final = manifests[-1]
    lexical_invocations = (
        final.invocations[0],
        final.invocations[-1],
    )
    assert lexical_invocations[0].family_id == lexical_invocations[1].family_id
    assert lexical_invocations[0].atom_id == lexical_invocations[1].atom_id
    assert lexical_invocations[0].atom_bank_id == lexical_invocations[1].atom_bank_id
    lexical_bank = compilation.atom_banks[family.LEXICAL_BANK_ID]
    assert lexical_bank.atom("atom.lexical.tied") is lexical_bank.atom(
        final.invocations[-1].atom_id
    )
    assert compilation.metadata()["generator_family_count"] == 4
    assert compilation.metadata()["typed_terminal_boundary_count"] == 1


def test_all_three_scopes_execute_live_with_literal_exact_carrier_parity() -> None:
    compilation = _compilation()
    input_ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    sequence = _sequence()
    with torch.inference_mode():
        reference = compilation.exact_carrier_executor.execute(
            input_ids,
            sequence,
        )
        executions = tuple(
            (
                scope,
                compilation.assemble(scope).execute(
                    input_ids,
                    sequence,
                    execution_nonce=f"toy.{index}",
                ),
            )
            for index, scope in enumerate(_SCOPES)
        )

    for scope, execution in executions:
        assert torch.equal(execution.output, reference.output)
        assert torch.equal(execution.normalized_output, reference.normalized_output)
        assert torch.equal(execution.logits, reference.logits)
        receipt = execution.receipt
        assert isinstance(receipt, HeterogeneousGeneratorExecutionReceipt)
        assert len(receipt.invocation_receipts) == 39
        assert all(
            item.exact_full_rank_lift_literal_parity
            for item in receipt.invocation_receipts
        )
        assert receipt.source_projection_call_count == 0
        assert receipt.source_fallback_call_count == 0
        assert receipt.learned_cross_event_correction_count == 0
        assert receipt.compression_attempted is False
        assert receipt.compression_eligible is False
        assert receipt.carrier_execution_receipt.correction_message_count == 0
        assert receipt.carrier_execution_receipt.aggregate_commit_count == 36
        assert tuple(
            boundary.output_carrier_sha256
            for boundary in receipt.carrier_execution_receipt.boundaries
        ) == tuple(
            boundary.output_carrier_sha256
            for boundary in reference.core_execution_receipt.boundaries
        )
        assert family.verify_gemma3_heterogeneous_literal_parity(
            lift_scope=scope,
            reference=reference,
            candidate=execution,
        ).passed


def test_plan_round_trip_and_tamper_validation_fail_closed() -> None:
    plan = _compilation().plan
    assert family.Gemma3HeterogeneousGeneratorPlan.from_dict(
        plan.to_dict()
    ) == plan

    tampered = copy.deepcopy(plan.to_dict())
    tampered["exact_gate_chain"][1] = "exact.skipped_gate"
    with pytest.raises(ValueError):
        family.Gemma3HeterogeneousGeneratorPlan.from_dict(tampered)

    tampered = copy.deepcopy(plan.to_dict())
    tampered["manifests"][-1]["invocations"][0]["atom_id"] = (
        "atom.lexical.untied"
    )
    with pytest.raises(ValueError):
        family.Gemma3HeterogeneousGeneratorPlan.from_dict(tampered)

    tampered = copy.deepcopy(plan.to_dict())
    tampered["manifests"][-1]["claims"]["compression_eligible"] = True
    with pytest.raises(ValueError):
        family.Gemma3HeterogeneousGeneratorPlan.from_dict(tampered)
