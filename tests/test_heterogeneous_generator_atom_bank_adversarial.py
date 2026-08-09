from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from fisher_graph import gemma3_heterogeneous_generator_family_compiler as gemma_family
from fisher_graph.causal_carrier_transport_graph import (
    CarrierEventSpec,
    build_causal_carrier_transport_manifest,
)
from fisher_graph.gemma3_heterogeneous_whole_model_generator_family_protocol import (
    build_gemma3_heterogeneous_generator_family_protocol,
    validate_gemma3_heterogeneous_generator_family_protocol,
)
from fisher_graph.heterogeneous_generator_atom_bank import (
    LIFT_SCOPE_FULL_HETEROGENEOUS,
    AttentionAtomRuntime,
    AttentionRuntimeContract,
    AtomEmission,
    FamilyEdgeOccurrenceSpec,
    GeneratorAtomBankRuntime,
    GeneratorAtomBankSpec,
    GeneratorAtomRuntime,
    GeneratorAtomSpec,
    GeneratorFamilySpec,
    GeneratorInvocationSpec,
    HeterogeneousGeneratorExecutor,
    LexicalAtomRuntime,
    LexicalRuntimeContract,
    MLPAtomRuntime,
    MLPRuntimeContract,
    TerminalRuntime,
    TerminalRuntimeContract,
    assemble_heterogeneous_generator_executor,
    build_exact_heterogeneous_generator_manifest,
    build_generator_atom_bank_spec,
    build_generator_atom_spec,
    validate_heterogeneous_generator_execution_receipt,
    validate_heterogeneous_generator_manifest,
)
from test_gemma3_causal_carrier_transport_compiler import (
    _sequence as _gemma_sequence,
    _transport_executor as _gemma_transport_executor,
)


_WIDTH = 4
_VOCABULARY = 7
_INTERMEDIATE = 6
_GLOBAL_BLOCKS = frozenset({5, 11, 17})

_LEXICAL_FAMILY = "family.lexical.tied"
_MLP_FAMILY = "family.mlp.shared"
_SLIDING_FAMILY = "family.attention.sliding"
_GLOBAL_FAMILY = "family.attention.global"
_TERMINAL_FAMILY = "family.terminal.exact"

_LEXICAL_BANK = "atom_bank.lexical.tied"
_MLP_BANK = "atom_bank.mlp"
_SLIDING_BANK = "atom_bank.attention.sliding"
_GLOBAL_BANK = "atom_bank.attention.global"


class _ToyLexicalAtom(LexicalAtomRuntime):
    def __init__(self) -> None:
        super().__init__(atom_id="atom.lexical.tied", output_rank=_WIDTH)
        values = torch.arange(
            _VOCABULARY * _WIDTH,
            dtype=torch.float32,
        ).view(_VOCABULARY, _WIDTH)
        self.weight = nn.Parameter(values / 100.0, requires_grad=False)

    def ingress(self, token_input: Tensor) -> Tensor:
        return self.weight[token_input]

    def egress(self, hidden_state: Tensor) -> Tensor:
        return hidden_state @ self.weight.transpose(0, 1)


class _ToyAttentionAtom(AttentionAtomRuntime):
    def __init__(
        self,
        *,
        atom_id: str,
        seed: int,
        shared_weight: Tensor | None = None,
    ) -> None:
        super().__init__(atom_id=atom_id, output_rank=_WIDTH)
        if shared_weight is None:
            generator = torch.Generator().manual_seed(seed)
            weight = torch.randn(_WIDTH, _WIDTH, generator=generator) / 200.0
        else:
            weight = shared_weight
        self.weight = nn.Parameter(weight, requires_grad=False)

    def emit(self, carrier: Tensor, sequence: object) -> AtomEmission:
        del sequence
        token_count = carrier.shape[-2]
        routing = torch.tril(
            torch.ones(
                token_count,
                token_count,
                dtype=carrier.dtype,
                device=carrier.device,
            )
        )
        routing = routing / routing.sum(dim=-1, keepdim=True)
        routing = routing.expand(carrier.shape[0], 1, -1, -1).contiguous()
        return AtomEmission(
            value=(carrier @ self.weight.transpose(0, 1)),
            attention_routing_weights=routing,
        )


class _ToyMLPAtom(MLPAtomRuntime):
    def __init__(
        self,
        *,
        atom_id: str,
        seed: int,
        shared_weight: Tensor | None = None,
    ) -> None:
        super().__init__(atom_id=atom_id, output_rank=_WIDTH)
        if shared_weight is None:
            generator = torch.Generator().manual_seed(seed)
            weight = torch.randn(_WIDTH, _WIDTH, generator=generator) / 200.0
        else:
            weight = shared_weight
        self.weight = nn.Parameter(weight, requires_grad=False)

    def emit(self, carrier: Tensor, sequence: object) -> AtomEmission:
        del sequence
        return AtomEmission(value=carrier @ self.weight.transpose(0, 1))


class _ToyTerminalRuntime(TerminalRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(_WIDTH), requires_grad=False)

    def normalize(self, carrier: Tensor) -> Tensor:
        return carrier * self.scale


@dataclass(frozen=True)
class _ToyFixture:
    executor: HeterogeneousGeneratorExecutor
    atoms: dict[str, GeneratorAtomRuntime]
    atom_banks: dict[str, GeneratorAtomBankRuntime]
    terminal: _ToyTerminalRuntime


def _family_specs() -> tuple[GeneratorFamilySpec, ...]:
    return (
        GeneratorFamilySpec(
            family_index=0,
            family_id=_LEXICAL_FAMILY,
            component_kind="lexical",
            runtime_contract=LexicalRuntimeContract(
                hidden_width=_WIDTH,
                vocabulary_size=_VOCABULARY,
            ),
            compatible_atom_bank_ids=(_LEXICAL_BANK,),
        ),
        GeneratorFamilySpec(
            family_index=1,
            family_id=_MLP_FAMILY,
            component_kind="mlp",
            runtime_contract=MLPRuntimeContract(
                hidden_width=_WIDTH,
                intermediate_width=_INTERMEDIATE,
                activation="gelu",
            ),
            compatible_atom_bank_ids=(_MLP_BANK,),
        ),
        GeneratorFamilySpec(
            family_index=2,
            family_id=_SLIDING_FAMILY,
            component_kind="attention",
            runtime_contract=AttentionRuntimeContract(
                hidden_width=_WIDTH,
                query_heads=2,
                key_value_heads=1,
                head_dimension=2,
                attention_kind="sliding_causal",
                window_size=2,
                qk_norm=True,
                rope=True,
            ),
            compatible_atom_bank_ids=(_SLIDING_BANK,),
        ),
        GeneratorFamilySpec(
            family_index=3,
            family_id=_GLOBAL_FAMILY,
            component_kind="attention",
            runtime_contract=AttentionRuntimeContract(
                hidden_width=_WIDTH,
                query_heads=2,
                key_value_heads=1,
                head_dimension=2,
                attention_kind="global_causal",
                window_size=None,
                qk_norm=True,
                rope=True,
            ),
            compatible_atom_bank_ids=(_GLOBAL_BANK,),
        ),
        GeneratorFamilySpec(
            family_index=4,
            family_id=_TERMINAL_FAMILY,
            component_kind="terminal",
            runtime_contract=TerminalRuntimeContract(hidden_width=_WIDTH),
            compatible_atom_bank_ids=(),
        ),
    )


def _build_fixture() -> _ToyFixture:
    lexical = _ToyLexicalAtom()
    atoms: dict[str, GeneratorAtomRuntime] = {lexical.atom_id: lexical}
    events: list[CarrierEventSpec] = []
    invocations: list[GeneratorInvocationSpec] = [
        GeneratorInvocationSpec(
            invocation_index=0,
            invocation_id="invocation.lexical.ingress",
            event_id="lexical.ingress",
            event_kind="lexical_ingress",
            family_id=_LEXICAL_FAMILY,
            atom_bank_id=_LEXICAL_BANK,
            atom_id=lexical.atom_id,
            exact_full_rank_lift=True,
            lift_input_rank=_WIDTH,
            lift_output_rank=_WIDTH,
        )
    ]
    primary_stored_scalar_counts: dict[str, int] = {}
    for block in range(18):
        attention_index = 1 + 2 * block
        attention_event_id = f"mutation.{attention_index}"
        global_attention = block in _GLOBAL_BLOCKS
        attention_bank = _GLOBAL_BANK if global_attention else _SLIDING_BANK
        attention_family = (
            _GLOBAL_FAMILY if global_attention else _SLIDING_FAMILY
        )
        attention = _ToyAttentionAtom(
            atom_id=f"atom.attention.layer.{block}",
            seed=10_000 + block,
        )
        atoms[attention.atom_id] = attention
        events.append(
            CarrierEventSpec(
                event_index=attention_index,
                event_id=attention_event_id,
                event_kind="attention",
                generator_invocation_id=(
                    f"invocation.attention.layer.{block}"
                ),
                generator_family_id=attention_family,
                input_carrier_version=attention_index - 1,
                output_carrier_version=attention_index,
            )
        )
        invocations.append(
            GeneratorInvocationSpec(
                invocation_index=attention_index,
                invocation_id=f"invocation.attention.layer.{block}",
                event_id=attention_event_id,
                event_kind="attention",
                family_id=attention_family,
                atom_bank_id=attention_bank,
                atom_id=attention.atom_id,
                exact_full_rank_lift=True,
                lift_input_rank=_WIDTH,
                lift_output_rank=_WIDTH,
            )
        )
        primary_stored_scalar_counts[attention_event_id] = (
            attention.stored_scalar_count
        )

        mlp_index = attention_index + 1
        mlp_event_id = f"mutation.{mlp_index}"
        mlp = _ToyMLPAtom(
            atom_id=f"atom.mlp.layer.{block}",
            seed=20_000 + block,
        )
        atoms[mlp.atom_id] = mlp
        events.append(
            CarrierEventSpec(
                event_index=mlp_index,
                event_id=mlp_event_id,
                event_kind="mlp",
                generator_invocation_id=f"invocation.mlp.layer.{block}",
                generator_family_id=_MLP_FAMILY,
                input_carrier_version=mlp_index - 1,
                output_carrier_version=mlp_index,
            )
        )
        invocations.append(
            GeneratorInvocationSpec(
                invocation_index=mlp_index,
                invocation_id=f"invocation.mlp.layer.{block}",
                event_id=mlp_event_id,
                event_kind="mlp",
                family_id=_MLP_FAMILY,
                atom_bank_id=_MLP_BANK,
                atom_id=mlp.atom_id,
                exact_full_rank_lift=True,
                lift_input_rank=_WIDTH,
                lift_output_rank=_WIDTH,
            )
        )
        primary_stored_scalar_counts[mlp_event_id] = mlp.stored_scalar_count

    invocations.extend(
        (
            GeneratorInvocationSpec(
                invocation_index=37,
                invocation_id="invocation.terminal.rmsnorm",
                event_id="terminal.rmsnorm",
                event_kind="terminal",
                family_id=_TERMINAL_FAMILY,
                atom_bank_id=None,
                atom_id=None,
                exact_full_rank_lift=True,
                lift_input_rank=_WIDTH,
                lift_output_rank=_WIDTH,
            ),
            GeneratorInvocationSpec(
                invocation_index=38,
                invocation_id="invocation.lexical.egress",
                event_id="lexical.egress",
                event_kind="lexical_egress",
                family_id=_LEXICAL_FAMILY,
                atom_bank_id=_LEXICAL_BANK,
                atom_id=lexical.atom_id,
                exact_full_rank_lift=True,
                lift_input_rank=_WIDTH,
                lift_output_rank=_WIDTH,
            ),
        )
    )
    carrier_manifest = build_causal_carrier_transport_manifest(
        lexical_invocation_id=invocations[0].invocation_id,
        lexical_generator_family_id=_LEXICAL_FAMILY,
        events=tuple(events),
        lexical_stored_scalar_count=lexical.stored_scalar_count,
        primary_stored_scalar_counts=primary_stored_scalar_counts,
    )

    atom_specs = tuple(
        build_generator_atom_spec(
            atom_index=index,
            atom_bank_id=(
                _LEXICAL_BANK
                if atom.component_kind == "lexical"
                else (
                    _MLP_BANK
                    if atom.component_kind == "mlp"
                    else (
                        _GLOBAL_BANK
                        if int(atom.atom_id.rsplit(".", 1)[1])
                        in _GLOBAL_BLOCKS
                        else _SLIDING_BANK
                    )
                )
            ),
            atom=atom,
        )
        for index, atom in enumerate(atoms.values())
    )
    bank_inputs = (
        (0, _LEXICAL_BANK, "lexical"),
        (1, _MLP_BANK, "mlp"),
        (2, _SLIDING_BANK, "attention.sliding"),
        (3, _GLOBAL_BANK, "attention.global"),
    )
    bank_specs: list[GeneratorAtomBankSpec] = []
    atom_banks: dict[str, GeneratorAtomBankRuntime] = {}
    for bank_index, bank_id, component_kind in bank_inputs:
        members = tuple(
            atom for atom in atom_specs if atom.atom_bank_id == bank_id
        )
        spec = build_generator_atom_bank_spec(
            bank_index=bank_index,
            atom_bank_id=bank_id,
            component_kind=component_kind,
            atoms=members,
        )
        bank_specs.append(spec)
        atom_banks[bank_id] = GeneratorAtomBankRuntime(
            spec=spec,
            atoms={atom.atom_id: atoms[atom.atom_id] for atom in members},
        )

    terminal = _ToyTerminalRuntime()
    source_scalar_count = (
        sum(spec.physical_stored_scalar_count for spec in bank_specs)
        + terminal.stored_scalar_count
    )
    manifest = build_exact_heterogeneous_generator_manifest(
        arm_id="exact.heterogeneous.toy",
        lift_scope=LIFT_SCOPE_FULL_HETEROGENEOUS,
        hidden_width=_WIDTH,
        source_stored_scalar_count=source_scalar_count,
        terminal_stored_scalar_count=terminal.stored_scalar_count,
        terminal_state_sha256=terminal.state_sha256,
        carrier_manifest=carrier_manifest,
        families=_family_specs(),
        atom_banks=bank_specs,
        atoms=atom_specs,
        invocations=invocations,
    )
    executor = assemble_heterogeneous_generator_executor(
        manifest=manifest,
        atom_banks=atom_banks,
        terminal_runtime=terminal,
    )
    return _ToyFixture(
        executor=executor,
        atoms=atoms,
        atom_banks=atom_banks,
        terminal=terminal,
    )


@pytest.fixture(scope="module")
def toy() -> _ToyFixture:
    return _build_fixture()


def _tokens() -> Tensor:
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)


def _manual_exact(toy: _ToyFixture, tokens: Tensor) -> tuple[Tensor, Tensor]:
    invocations = toy.executor.manifest.invocations
    lexical = toy.executor.atom_for(invocations[0])
    assert isinstance(lexical, _ToyLexicalAtom)
    carrier = lexical.ingress(tokens)
    for invocation in invocations[1:37]:
        atom = toy.executor.atom_for(invocation)
        if isinstance(atom, _ToyAttentionAtom):
            delta = atom.emit(carrier, None).value
        else:
            assert isinstance(atom, _ToyMLPAtom)
            delta = atom.emit(carrier, None).value
        carrier = carrier + delta
    normalized = toy.terminal.normalize(carrier)
    return carrier, lexical.egress(normalized)


def test_exact_manifest_closes_39_invocations_75_occurrences_and_37_atoms(
    toy: _ToyFixture,
) -> None:
    manifest = toy.executor.manifest
    payload = manifest.to_dict()

    assert len(manifest.invocations) == 39
    assert len(manifest.edge_occurrences) == 75
    assert len(manifest.atom_banks) == 4
    assert len(manifest.atoms) == 37
    assert payload["accounting"] == {
        "invocation_count": 39,
        "family_catalog_entry_count": 5,
        "generator_family_count": 4,
        "typed_terminal_boundary_count": 1,
        "atom_bank_count": 4,
        "physical_atom_count": 37,
        "edge_occurrence_count": 75,
        "physical_atom_scalar_count": 604,
        "terminal_stored_scalar_count": 4,
        "deployed_scalar_count": 608,
        "source_stored_scalar_count": 608,
        "parameter_difference": 0,
        "learned_cross_event_correction_count": 0,
    }
    assert validate_heterogeneous_generator_manifest(payload) == manifest


def test_tied_lexical_atom_is_one_physical_object_and_counted_once(
    toy: _ToyFixture,
) -> None:
    manifest = toy.executor.manifest
    ingress, egress = manifest.invocations[0], manifest.invocations[38]

    assert ingress.family_id == egress.family_id == _LEXICAL_FAMILY
    assert ingress.atom_bank_id == egress.atom_bank_id == _LEXICAL_BANK
    assert ingress.atom_id == egress.atom_id == "atom.lexical.tied"
    assert toy.executor.atom_for(ingress) is toy.executor.atom_for(egress)
    lexical_bank = next(
        bank for bank in manifest.atom_banks if bank.atom_bank_id == _LEXICAL_BANK
    )
    assert lexical_bank.physical_stored_scalar_count == _VOCABULARY * _WIDTH


def test_repeated_family_uses_keep_distinct_invocation_receipts(
    toy: _ToyFixture,
) -> None:
    execution = toy.executor.execute(
        _tokens(),
        None,
        execution_nonce="repeat-family-receipts",
    )
    mlp = tuple(
        receipt
        for receipt in execution.receipt.invocation_receipts
        if receipt.family_id == _MLP_FAMILY
    )
    sliding = tuple(
        receipt
        for receipt in execution.receipt.invocation_receipts
        if receipt.family_id == _SLIDING_FAMILY
    )

    assert len(mlp) == 18
    assert len(sliding) == 15
    assert len({item.invocation_id for item in mlp}) == 18
    assert len({item.atom_id for item in mlp}) == 18
    assert [item.invocation_index for item in mlp] == list(range(2, 37, 2))
    assert len(execution.receipt.invocation_receipts) == 39


def test_same_family_edge_occurrences_remain_strictly_forward(
    toy: _ToyFixture,
) -> None:
    source = toy.executor.manifest.invocations[2]
    target = toy.executor.manifest.invocations[4]
    assert source.family_id == target.family_id == _MLP_FAMILY
    occurrence = FamilyEdgeOccurrenceSpec(
        occurrence_index=0,
        occurrence_id="occurrence.mlp.0.to.1",
        message_id="message.mlp.0.to.1",
        message_kind="family_transport",
        source_invocation_id=source.invocation_id,
        target_invocation_id=target.invocation_id,
        source_invocation_index=source.invocation_index,
        target_invocation_index=target.invocation_index,
        emitting_family_id=source.family_id,
        atom_bank_id=_MLP_BANK,
        atom_id=source.atom_id,
        carrier_edge_id=None,
        learned_cross_event_correction=False,
    )
    assert occurrence.source_invocation_index < occurrence.target_invocation_index
    assert occurrence.emitting_family_id == _MLP_FAMILY

    with pytest.raises(ValueError, match="strictly forward"):
        replace(
            occurrence,
            source_invocation_index=4,
            target_invocation_index=4,
        )
    with pytest.raises(ValueError, match="strictly forward"):
        replace(
            occurrence,
            source_invocation_index=6,
            target_invocation_index=4,
        )


def test_typed_runtime_abi_rejects_wrong_atom_class(
    toy: _ToyFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = toy.executor.atom_for

    def wrong_atom(invocation: GeneratorInvocationSpec) -> GeneratorAtomRuntime:
        if invocation.event_kind == "attention":
            return toy.atoms["atom.mlp.layer.0"]
        return original(invocation)

    monkeypatch.setattr(toy.executor, "atom_for", wrong_atom)
    with pytest.raises(TypeError, match="attention invocation resolved"):
        toy.executor.execute(
            _tokens(),
            None,
            execution_nonce="wrong-attention-abi",
        )


def test_manifest_rejects_dangling_atom_created_by_duplicate_binding(
    toy: _ToyFixture,
) -> None:
    manifest = toy.executor.manifest
    invocations = list(manifest.invocations)
    first_sliding = invocations[1]
    later_sliding = invocations[3]
    invocations[1] = replace(first_sliding, atom_id=later_sliding.atom_id)

    with pytest.raises(ValueError, match="every physical atom must be used"):
        replace(manifest, invocations=tuple(invocations))


def test_atom_bank_rejects_object_and_tensor_storage_aliasing(
    toy: _ToyFixture,
) -> None:
    spec = next(
        bank for bank in toy.executor.manifest.atom_banks
        if bank.atom_bank_id == _MLP_BANK
    )
    atoms = {
        atom_id: toy.atoms[atom_id]
        for atom_id in spec.atom_ids
    }
    atoms[spec.atom_ids[1]] = atoms[spec.atom_ids[0]]
    with pytest.raises(ValueError, match="one runtime object"):
        GeneratorAtomBankRuntime(spec=spec, atoms=atoms)

    distinct = {
        atom_id: toy.atoms[atom_id]
        for atom_id in spec.atom_ids
    }
    first = distinct[spec.atom_ids[0]]
    assert isinstance(first, _ToyMLPAtom)
    distinct[spec.atom_ids[1]] = _ToyMLPAtom(
        atom_id=spec.atom_ids[1],
        seed=0,
        shared_weight=first.weight,
    )
    with pytest.raises(ValueError, match="alias one physical tensor"):
        GeneratorAtomBankRuntime(spec=spec, atoms=distinct)


def test_executor_rejects_cross_bank_persistent_storage_aliasing() -> None:
    toy = _build_fixture()
    sliding = toy.atom_banks[_SLIDING_BANK].atom("atom.attention.layer.0")
    global_attention = toy.atom_banks[_GLOBAL_BANK].atom(
        "atom.attention.layer.5"
    )
    assert isinstance(sliding, _ToyAttentionAtom)
    assert isinstance(global_attention, _ToyAttentionAtom)
    global_attention.weight = sliding.weight

    with pytest.raises(ValueError, match="two atom banks cannot alias"):
        assemble_heterogeneous_generator_executor(
            manifest=toy.executor.manifest,
            atom_banks=toy.atom_banks,
            terminal_runtime=toy.terminal,
        )


def test_executor_rejects_terminal_to_atom_storage_aliasing() -> None:
    toy = _build_fixture()
    mlp = toy.atom_banks[_MLP_BANK].atom("atom.mlp.layer.0")
    assert isinstance(mlp, _ToyMLPAtom)
    toy.terminal.scale = nn.Parameter(
        mlp.weight.view(-1)[:_WIDTH],
        requires_grad=False,
    )

    with pytest.raises(ValueError, match="terminal storage cannot alias"):
        assemble_heterogeneous_generator_executor(
            manifest=toy.executor.manifest,
            atom_banks=toy.atom_banks,
            terminal_runtime=toy.terminal,
        )


def test_atom_bank_detects_post_manifest_tensor_tampering() -> None:
    toy = _build_fixture()
    bank = toy.atom_banks[_MLP_BANK]
    atom = bank.atom("atom.mlp.layer.0")
    with torch.no_grad():
        next(atom.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="state drifted"):
        bank.assert_matches_spec()


def test_full_rank_lift_is_literal_and_matches_manual_exact_execution(
    toy: _ToyFixture,
) -> None:
    tokens = _tokens()
    expected_carrier, expected_logits = _manual_exact(toy, tokens)
    execution = toy.executor.execute(
        tokens,
        None,
        execution_nonce="literal-full-rank-parity",
    )

    assert torch.equal(execution.output, expected_carrier)
    assert torch.equal(execution.logits, expected_logits)
    assert all(
        receipt.exact_full_rank_lift_literal_parity is True
        and receipt.atom_output_sha256 == receipt.lifted_output_sha256
        for receipt in execution.receipt.invocation_receipts
    )
    assert validate_heterogeneous_generator_execution_receipt(
        execution.receipt.to_dict(),
        manifest=toy.executor.manifest,
    ) == execution.receipt


def test_execution_receipt_rejects_rehashed_terminal_state_tampering(
    toy: _ToyFixture,
) -> None:
    execution = toy.executor.execute(
        _tokens(),
        None,
        execution_nonce="terminal-state-tamper",
    )
    tampered = replace(
        execution.receipt,
        terminal_state_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="physical-state accounting drifted"):
        validate_heterogeneous_generator_execution_receipt(
            tampered,
            manifest=toy.executor.manifest,
        )


def test_storage_closes_without_compression_credit(
    toy: _ToyFixture,
) -> None:
    manifest = toy.executor.manifest
    execution = toy.executor.execute(
        _tokens(),
        None,
        execution_nonce="storage-and-claim-firewall",
    )
    payload = manifest.to_dict()

    assert manifest.physical_atom_scalar_count == sum(
        bank.physical_stored_scalar_count for bank in manifest.atom_banks
    )
    assert manifest.deployed_scalar_count == manifest.source_stored_scalar_count
    assert execution.receipt.physical_stored_scalar_count == (
        manifest.deployed_scalar_count
    )
    assert payload["accounting"]["parameter_difference"] == 0
    assert payload["claims"]["compression_eligible"] is False
    assert payload["claims"]["compression_success_claimed"] is False
    assert manifest.compression_attempted is False
    assert execution.receipt.compression_attempted is False
    assert execution.receipt.compression_eligible is False
    assert all(
        occurrence.learned_cross_event_correction is False
        for occurrence in manifest.edge_occurrences
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("model_contract", "event_count"), 40),
        (("execution_graph", "invocation_count"), 38),
        (
            (
                "family_and_atom_bank_contract",
                "heterogeneous_atom_bank",
                "physical_atom_count",
            ),
            36,
        ),
        (("exact_lift_gate_chain", "maximum_absolute_logit_error"), 1e-6),
        (
            (
                "claim_firewall",
                "no_compression_or_speed_claim_authorized_by_this_setup",
            ),
            False,
        ),
    ),
)
def test_strict_protocol_rejects_authenticated_field_tampering(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    protocol = deepcopy(
        build_gemma3_heterogeneous_generator_family_protocol()
    )
    current: dict[str, object] = protocol
    for key in path[:-1]:
        value = current[key]
        assert isinstance(value, dict)
        current = value
    current[path[-1]] = replacement

    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_heterogeneous_generator_family_protocol(protocol)


def test_strict_protocol_rejects_extra_fields_and_digest_tampering() -> None:
    protocol = deepcopy(
        build_gemma3_heterogeneous_generator_family_protocol()
    )
    protocol["unexpected"] = True
    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_heterogeneous_generator_family_protocol(protocol)

    protocol = deepcopy(
        build_gemma3_heterogeneous_generator_family_protocol()
    )
    protocol["protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not canonical"):
        validate_gemma3_heterogeneous_generator_family_protocol(protocol)


@pytest.fixture(scope="module")
def gemma_compilation() -> gemma_family.Gemma3HeterogeneousGeneratorCompilation:
    return gemma_family.compile_gemma3_exact_heterogeneous_generator_plan(
        _gemma_transport_executor(),
        require_pinned_gemma=False,
    )


def test_compiler_freezes_protocol_exact_invocation_and_occurrence_ids(
    gemma_compilation: gemma_family.Gemma3HeterogeneousGeneratorCompilation,
) -> None:
    expected_invocations = (
        "lexical.ingress",
        *(f"mutation.{index}" for index in range(1, 37)),
        "terminal.norm_softcap",
        "lexical.egress",
    )
    expected_occurrences = ["occurrence.lexical.ingress"]
    for index in range(1, 37):
        expected_occurrences.extend(
            (
                f"occurrence.carry.mutation.{index}",
                f"occurrence.primary.mutation.{index}",
            )
        )
    expected_occurrences.extend(
        ("occurrence.terminal.norm_softcap", "occurrence.lexical.egress")
    )

    for manifest in gemma_compilation.plan.manifests:
        assert tuple(
            invocation.invocation_id for invocation in manifest.invocations
        ) == expected_invocations
        assert tuple(
            occurrence.occurrence_id
            for occurrence in manifest.edge_occurrences
        ) == tuple(expected_occurrences)
        assert len(manifest.invocations) == 39
        assert len(manifest.edge_occurrences) == 75
        assert len(manifest.atom_banks) == 4
        assert len(manifest.atoms) == 37


def test_compiler_three_scopes_execute_without_exact_forward_delegation(
    gemma_compilation: gemma_family.Gemma3HeterogeneousGeneratorCompilation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_input = torch.tensor([[1, 2, 3], [0, 4, 5]])
    sequence = _gemma_sequence()
    reference = gemma_compilation.exact_carrier_executor.execute(
        token_input,
        sequence,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("heterogeneous lift delegated to the exact forward")

    monkeypatch.setattr(
        gemma_compilation.exact_carrier_executor,
        "execute",
        forbidden,
    )
    monkeypatch.setattr(
        gemma_compilation.exact_carrier_executor.exact_executor,
        "forward_components",
        forbidden,
    )
    scopes = (
        gemma_family.LIFT_SCOPE_MLP_FAMILY,
        gemma_family.LIFT_SCOPE_COMPATIBLE_ATTENTION,
        gemma_family.LIFT_SCOPE_FULL_HETEROGENEOUS,
    )
    for index, scope in enumerate(scopes):
        execution = gemma_compilation.assemble(scope).execute(
            token_input,
            sequence,
            execution_nonce=f"adversarial.scope.{index}",
        )
        assert torch.equal(execution.output, reference.output)
        assert torch.equal(
            execution.normalized_output,
            reference.normalized_output,
        )
        assert torch.equal(execution.logits, reference.logits)
        assert len(execution.receipt.invocation_receipts) == 39
        assert execution.receipt.source_projection_call_count == 0
        assert execution.receipt.source_fallback_call_count == 0


def test_compiler_counts_shareable_families_separately_from_terminal(
    gemma_compilation: gemma_family.Gemma3HeterogeneousGeneratorCompilation,
) -> None:
    metadata = gemma_compilation.metadata()
    final = gemma_compilation.plan.manifests[-1]
    generator_families = tuple(
        family
        for family in final.families
        if family.component_kind != "terminal"
    )
    terminal_families = tuple(
        family
        for family in final.families
        if family.component_kind == "terminal"
    )

    assert len(generator_families) == 4
    assert len(terminal_families) == 1
    assert metadata["generator_family_count"] == 4
    assert metadata["typed_terminal_boundary_count"] == 1
    assert metadata["family_catalog_entry_count"] == 5
    assert metadata["typed_atom_bank_count"] == 4
    assert metadata["physical_atom_count"] == 37
    assert metadata["exact_full_rank_scalar_parity"] is True
    assert metadata["family_schema_sharing_parameter_credit"] == 0
    assert metadata["compression_attempted"] is False
    assert metadata["compression_eligible"] is False


def test_compiler_plan_strict_load_round_trip_and_extra_field_rejection(
    gemma_compilation: gemma_family.Gemma3HeterogeneousGeneratorCompilation,
    tmp_path: Path,
) -> None:
    path = tmp_path / "heterogeneous-plan.json"
    payload = gemma_compilation.plan.to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert gemma_family.load_gemma3_heterogeneous_generator_plan(path) == (
        gemma_compilation.plan
    )

    tampered = deepcopy(payload)
    tampered["claims"]["unexpected"] = True
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        gemma_family.load_gemma3_heterogeneous_generator_plan(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("terminal_digest", "authentication failed"),
        ("dangling_atom", "invocation atom binding is invalid"),
        ("storage_count", "state/accounting drifted"),
        ("compression_claim", "authentication failed"),
        ("gate_skip", "gate order drifted|authentication failed"),
    ),
)
def test_compiler_plan_rejects_nested_authenticated_tampering(
    gemma_compilation: gemma_family.Gemma3HeterogeneousGeneratorCompilation,
    mutation: str,
    match: str,
) -> None:
    payload = deepcopy(gemma_compilation.plan.to_dict())
    manifest = payload["manifests"][-1]
    assert isinstance(manifest, dict)
    if mutation == "terminal_digest":
        manifest["terminal_state_sha256"] = "0" * 64
    elif mutation == "dangling_atom":
        manifest["invocations"][1]["atom_id"] = "atom.attention.missing"
    elif mutation == "storage_count":
        manifest["atom_banks"][0]["physical_stored_scalar_count"] += 1
    elif mutation == "compression_claim":
        manifest["claims"]["compression_eligible"] = True
    elif mutation == "gate_skip":
        payload["exact_gate_chain"][1] = "exact.skipped"
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    with pytest.raises(ValueError, match=match):
        gemma_family.validate_gemma3_heterogeneous_generator_plan(payload)
