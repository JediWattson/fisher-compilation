"""Exact Gemma lift into typed generator families and physical atom banks.

This compiler sits directly above the exact causal carrier-transport runtime.
It does not fit, merge, prune, or reduce anything.  Instead, it makes three
successive *exact* views explicit:

1. all MLP invocations adopt one compatible logical family contract;
2. attention invocations adopt separate global/sliding family contracts;
3. tied lexical ingress/egress adopt one family and one physical atom.

Every view retains the same 37 full-rank physical atoms and terminal RMSNorm,
so family labels receive no parameter credit.  The returned executor owns the
live lexical -> carrier -> terminal -> egress path; the prior exact executor is
kept only as a separately callable parity oracle on the compilation product.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn

from .adapters import SequenceContext
from .gemma3_attention_generator import Gemma3AttentionGenerator
from .gemma3_causal_carrier_transport_compiler import (
    Gemma3CausalCarrierTransportExecution,
    Gemma3CausalCarrierTransportExecutor,
)
from .gemma3_heterogeneous_whole_model_generator_family_protocol import (
    PINNED_ATTENTION_SCALARS,
    PINNED_EVENT_COUNT,
    PINNED_GLOBAL_ATTENTION_BLOCKS,
    PINNED_HIDDEN_WIDTH,
    PINNED_HETEROGENEOUS_ATOM_COUNT,
    PINNED_LEXICAL_SCALARS,
    PINNED_MLP_SCALARS,
    PINNED_MLP_WIDTH,
    PINNED_MUTATION_COUNT,
    PINNED_SOURCE_SCALARS,
    PINNED_TERMINAL_NORM_SCALARS,
    PINNED_VOCABULARY_SIZE,
    build_gemma3_heterogeneous_generator_family_protocol,
    validate_gemma3_heterogeneous_generator_family_protocol,
)
from .gemma3_joint_cross_block_chart_executor import (
    JointGraphGeneratorMutation,
)
from .heterogeneous_generator_atom_bank import (
    AttentionAtomRuntime,
    AttentionRuntimeContract,
    AtomEmission,
    GeneratorAtomBankRuntime,
    GeneratorAtomBankSpec,
    GeneratorAtomSpec,
    GeneratorFamilySpec,
    GeneratorInvocationSpec,
    HeterogeneousGeneratorExecution,
    HeterogeneousGeneratorExecutor,
    HeterogeneousGeneratorManifest,
    LIFT_SCOPE_COMPATIBLE_ATTENTION,
    LIFT_SCOPE_FULL_HETEROGENEOUS,
    LIFT_SCOPE_MLP_FAMILY,
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
    module_state_sha256,
    validate_heterogeneous_generator_manifest,
)


GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA = (
    "fisher_graph.gemma3_heterogeneous_generator_family.plan.v1"
)
GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_FORMAT_VERSION = 1

EXACT_CARRIER_ARM_ID: Final = "exact.causal_carrier"
MLP_FAMILY_ARM_ID: Final = "exact.cross_block_mlp_family_lift"
COMPATIBLE_ATTENTION_ARM_ID: Final = (
    "exact.compatible_attention_family_lift"
)
FULL_HETEROGENEOUS_ARM_ID: Final = (
    "exact.heterogeneous_full_rank_atom_bank_lift"
)

EXACT_LIFT_GATE_CHAIN: Final = (
    EXACT_CARRIER_ARM_ID,
    MLP_FAMILY_ARM_ID,
    COMPATIBLE_ATTENTION_ARM_ID,
    FULL_HETEROGENEOUS_ARM_ID,
)
_SCOPE_ARM_IDS: Final = {
    LIFT_SCOPE_MLP_FAMILY: MLP_FAMILY_ARM_ID,
    LIFT_SCOPE_COMPATIBLE_ATTENTION: COMPATIBLE_ATTENTION_ARM_ID,
    LIFT_SCOPE_FULL_HETEROGENEOUS: FULL_HETEROGENEOUS_ARM_ID,
}
_ORDERED_SCOPES: Final = tuple(_SCOPE_ARM_IDS)

LEXICAL_BANK_ID: Final = "atom_bank.lexical.tied.full_rank"
MLP_BANK_ID: Final = "atom_bank.mlp.full_rank"
SLIDING_ATTENTION_BANK_ID: Final = (
    "atom_bank.attention.sliding.full_rank"
)
GLOBAL_ATTENTION_BANK_ID: Final = "atom_bank.attention.global.full_rank"
ORDERED_BANK_IDS: Final = (
    LEXICAL_BANK_ID,
    MLP_BANK_ID,
    SLIDING_ATTENTION_BANK_ID,
    GLOBAL_ATTENTION_BANK_ID,
)

_PLAN_HASH_DOMAIN = b"fisher_graph:gemma3-heterogeneous-plan:v1\0"
_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_PLAN_HASH_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _require_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value[0] not in _ID_CHARS
        or any(character not in _ID_CHARS for character in value)
    ):
        raise ValueError(f"{label} is not a canonical identifier")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _protocol_sha256() -> str:
    protocol = build_gemma3_heterogeneous_generator_family_protocol()
    validate_gemma3_heterogeneous_generator_family_protocol(protocol)
    digest = protocol.get("protocol_sha256")
    return _require_sha256(digest, label="heterogeneous protocol SHA-256")


class Gemma3TiedLexicalAtom(LexicalAtomRuntime):
    """One exact physical matrix used for both embedding and LM-head calls."""

    def __init__(
        self,
        *,
        lexical: nn.Module,
        hidden_width: int,
        vocabulary_size: int,
        final_logit_softcap: float | None,
    ) -> None:
        super().__init__(atom_id="atom.lexical.tied", output_rank=hidden_width)
        if not isinstance(lexical, nn.Module):
            raise TypeError("lexical runtime must be an nn.Module")
        if type(vocabulary_size) is not int or vocabulary_size <= 1:
            raise ValueError("vocabulary size must be positive")
        if final_logit_softcap is not None and (
            isinstance(final_logit_softcap, bool)
            or not isinstance(final_logit_softcap, (float, int))
            or not math.isfinite(float(final_logit_softcap))
            or float(final_logit_softcap) <= 0
        ):
            raise ValueError("final logit softcap must be finite and positive")
        self.lexical = lexical
        self.vocabulary_size = vocabulary_size
        self.final_logit_softcap = (
            None if final_logit_softcap is None else float(final_logit_softcap)
        )

    def ingress(self, token_input: Tensor) -> Tensor:
        ingress = getattr(self.lexical, "ingress", None)
        if not callable(ingress):
            raise RuntimeError("exact lexical ingress method disappeared")
        value = ingress(token_input)
        if not isinstance(value, Tensor) or value.shape != (
            *token_input.shape,
            self.output_rank,
        ):
            raise ValueError("exact lexical ingress returned the wrong shape")
        return value

    def egress(self, hidden_state: Tensor) -> Tensor:
        egress = getattr(self.lexical, "egress", None)
        if not callable(egress):
            raise RuntimeError("exact lexical egress method disappeared")
        logits = egress(hidden_state)
        if not isinstance(logits, Tensor) or logits.shape != (
            *hidden_state.shape[:2],
            self.vocabulary_size,
        ):
            raise ValueError("exact lexical egress returned the wrong shape")
        if self.final_logit_softcap is not None:
            cap = self.final_logit_softcap
            logits = torch.tanh(logits / cap) * cap
        return logits


class Gemma3AttentionMutationAtom(AttentionAtomRuntime):
    """Typed exact attention mutation, including both Gemma RMSNorms."""

    def __init__(
        self,
        *,
        atom_id: str,
        mutation: JointGraphGeneratorMutation,
    ) -> None:
        super().__init__(atom_id=atom_id, output_rank=mutation.hidden_width)
        if (
            not isinstance(mutation, JointGraphGeneratorMutation)
            or mutation.event_spec.event_kind != "attention_mutation"
            or not isinstance(mutation.compiled_operator, Gemma3AttentionGenerator)
        ):
            raise TypeError("attention atom requires an exact Gemma mutation")
        self.mutation = mutation

    @property
    def block_ordinal(self) -> int:
        value = self.mutation.event_spec.block_ordinal
        if type(value) is not int:
            raise RuntimeError("attention block ordinal disappeared")
        return value

    def emit(self, carrier: Tensor, sequence: object) -> AtomEmission:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("Gemma attention requires a SequenceContext")
        incoming = torch.zeros(
            *carrier.shape[:2],
            self.mutation.incoming_chart_count,
            self.mutation.incoming_generator_rank,
            dtype=carrier.dtype,
            device=carrier.device,
        )
        generated = self.mutation.execute_with_incoming(
            carrier,
            sequence,
            incoming,
        )
        operator = self.mutation.compiled_operator
        assert isinstance(operator, Gemma3AttentionGenerator)
        # The exact carrier control already defines this deterministic replay
        # as the routing receipt.  Reuse it so causal/sliding masks, RoPE, GQA,
        # QK norms, and softcap semantics stay byte-for-byte aligned.
        _, routing = Gemma3CausalCarrierTransportExecutor._attention_routing(
            operator=operator,
            normalized_input=generated.chart_input,
            sequence=sequence,
            event=self.mutation.event_spec,
            runtime_logical_pairs=(
                generated.runtime_receipt.logical_causal_key_pairs
            ),
        )
        return AtomEmission(
            value=generated.delta,
            attention_routing_weights=routing,
        )


class Gemma3MLPMutationAtom(MLPAtomRuntime):
    """Typed exact token-local gated-GELU mutation with both RMSNorms."""

    def __init__(
        self,
        *,
        atom_id: str,
        mutation: JointGraphGeneratorMutation,
    ) -> None:
        super().__init__(atom_id=atom_id, output_rank=mutation.hidden_width)
        if (
            not isinstance(mutation, JointGraphGeneratorMutation)
            or mutation.event_spec.event_kind != "mlp_mutation"
        ):
            raise TypeError("MLP atom requires an exact Gemma mutation")
        self.mutation = mutation

    @property
    def block_ordinal(self) -> int:
        value = self.mutation.event_spec.block_ordinal
        if type(value) is not int:
            raise RuntimeError("MLP block ordinal disappeared")
        return value

    def emit(self, carrier: Tensor, sequence: object) -> AtomEmission:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("Gemma MLP requires a SequenceContext")
        incoming = torch.zeros(
            *carrier.shape[:2],
            self.mutation.incoming_chart_count,
            self.mutation.incoming_generator_rank,
            dtype=carrier.dtype,
            device=carrier.device,
        )
        generated = self.mutation.execute_with_incoming(
            carrier,
            sequence,
            incoming,
        )
        return AtomEmission(value=generated.delta)


class Gemma3ExactTerminalRuntime(TerminalRuntime):
    """Exact terminal RMSNorm; logit softcap remains in lexical egress."""

    def __init__(self, terminal_norm: nn.Module) -> None:
        super().__init__()
        if not isinstance(terminal_norm, nn.Module):
            raise TypeError("terminal norm must be an nn.Module")
        self.terminal_norm = terminal_norm

    def normalize(self, carrier: Tensor) -> Tensor:
        value = self.terminal_norm(carrier)
        if not isinstance(value, Tensor) or value.shape != carrier.shape:
            raise ValueError("terminal RMSNorm must preserve carrier shape")
        return value


@dataclass(frozen=True, slots=True)
class Gemma3HeterogeneousGeneratorPlan:
    """Serializable, authenticated three-successor exact lift plan."""

    source_model_fingerprint: str
    exact_carrier_manifest_sha256: str
    manifests: tuple[HeterogeneousGeneratorManifest, ...]
    protocol_sha256: str
    exact_gate_chain: tuple[str, ...] = EXACT_LIFT_GATE_CHAIN
    schema: str = GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA
    format_version: int = GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA:
            raise ValueError("heterogeneous Gemma plan schema is unsupported")
        if self.format_version != GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_FORMAT_VERSION:
            raise ValueError("heterogeneous Gemma plan version is unsupported")
        _require_sha256(
            self.source_model_fingerprint,
            label="source model fingerprint",
        )
        _require_sha256(
            self.exact_carrier_manifest_sha256,
            label="exact carrier manifest SHA-256",
        )
        if self.protocol_sha256 != _protocol_sha256():
            raise ValueError("heterogeneous protocol binding drifted")
        if self.exact_gate_chain != EXACT_LIFT_GATE_CHAIN:
            raise ValueError("exact lift gate order drifted")
        if type(self.manifests) is not tuple or tuple(
            item.lift_scope for item in self.manifests
        ) != _ORDERED_SCOPES:
            raise ValueError("plan must contain the three ordered lift scopes")
        validated = tuple(
            validate_heterogeneous_generator_manifest(item)
            for item in self.manifests
        )
        if tuple(item.arm_id for item in validated) != tuple(
            _SCOPE_ARM_IDS[scope] for scope in _ORDERED_SCOPES
        ):
            raise ValueError("exact lift arm IDs drifted")
        if any(
            item.carrier_manifest.fingerprint_sha256
            != self.exact_carrier_manifest_sha256
            for item in validated
        ):
            raise ValueError("lift plan no longer binds the exact carrier base")
        source_counts = {
            item.source_stored_scalar_count for item in validated
        }
        if len(source_counts) != 1:
            raise ValueError("lift plan source scalar inventory drifted")
        reference = validated[0]
        expected_invocation_ids = (
            "lexical.ingress",
            *(f"mutation.{index}" for index in range(1, 37)),
            "terminal.norm_softcap",
            "lexical.egress",
        )
        expected_occurrence_ids = (
            "occurrence.lexical.ingress",
            *(
                occurrence_id
                for index in range(1, 37)
                for occurrence_id in (
                    f"occurrence.carry.mutation.{index}",
                    f"occurrence.primary.mutation.{index}",
                )
            ),
            "occurrence.terminal.norm_softcap",
            "occurrence.lexical.egress",
        )
        if (
            tuple(item.invocation_id for item in reference.invocations)
            != expected_invocation_ids
            or tuple(
                item.occurrence_id for item in reference.edge_occurrences
            )
            != expected_occurrence_ids
        ):
            raise ValueError("ordered protocol invocation/occurrence IDs drifted")
        for successor in validated[1:]:
            if (
                successor.atoms != reference.atoms
                or successor.atom_banks != reference.atom_banks
                or tuple(
                    (
                        item.invocation_index,
                        item.invocation_id,
                        item.event_id,
                        item.event_kind,
                        item.atom_bank_id,
                        item.atom_id,
                    )
                    for item in successor.invocations
                )
                != tuple(
                    (
                        item.invocation_index,
                        item.invocation_id,
                        item.event_id,
                        item.event_kind,
                        item.atom_bank_id,
                        item.atom_id,
                    )
                    for item in reference.invocations
                )
            ):
                raise ValueError("exact lift arms changed physical atom bindings")
        family_counts = tuple(len(item.families) for item in validated)
        if family_counts != (22, 6, 5):
            raise ValueError("exact family generalization ladder drifted")
        generator_family_counts = tuple(
            sum(family.component_kind != "terminal" for family in item.families)
            for item in validated
        )
        if generator_family_counts != (21, 5, 4):
            raise ValueError("generator-family versus terminal boundary drifted")

    @property
    def fingerprint_sha256(self) -> str:
        return _fingerprint(self.payload())

    def manifest_for_scope(self, lift_scope: str) -> HeterogeneousGeneratorManifest:
        for manifest in self.manifests:
            if manifest.lift_scope == lift_scope:
                return manifest
        raise KeyError(f"unknown lift scope {lift_scope}")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_model_fingerprint": self.source_model_fingerprint,
            "exact_carrier_manifest_sha256": (
                self.exact_carrier_manifest_sha256
            ),
            "protocol_sha256": self.protocol_sha256,
            "exact_gate_chain": list(self.exact_gate_chain),
            "manifests": [item.to_dict() for item in self.manifests],
            "claims": {
                "live_39_event_executor_required": True,
                "literal_parity_gate_required_per_successor": True,
                "learned_cross_event_corrections_enabled": False,
                "compression_attempted": False,
                "compression_eligible": False,
                "family_schema_sharing_receives_parameter_credit": False,
                "pinned_gemma_resource_profile": (
                    self.manifests[0].source_stored_scalar_count
                    == PINNED_SOURCE_SCALARS
                    and self.manifests[0].hidden_width == PINNED_HIDDEN_WIDTH
                ),
            },
        }

    def to_dict(self) -> dict[str, object]:
        value = self.payload()
        value["plan_sha256"] = self.fingerprint_sha256
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Gemma3HeterogeneousGeneratorPlan:
        expected = {
            "schema",
            "format_version",
            "source_model_fingerprint",
            "exact_carrier_manifest_sha256",
            "protocol_sha256",
            "exact_gate_chain",
            "manifests",
            "claims",
            "plan_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("heterogeneous Gemma plan fields are not exact")
        manifests = value["manifests"]
        chain = value["exact_gate_chain"]
        if (
            not isinstance(manifests, Sequence)
            or isinstance(manifests, (str, bytes))
            or not isinstance(chain, Sequence)
            or isinstance(chain, (str, bytes))
        ):
            raise TypeError("serialized plan sequences are invalid")
        result = cls(
            source_model_fingerprint=value["source_model_fingerprint"],
            exact_carrier_manifest_sha256=value[
                "exact_carrier_manifest_sha256"
            ],
            manifests=tuple(
                HeterogeneousGeneratorManifest.from_dict(item)
                for item in manifests
            ),
            protocol_sha256=value["protocol_sha256"],
            exact_gate_chain=tuple(chain),
            schema=value["schema"],
            format_version=value["format_version"],
        )
        _require_sha256(value["plan_sha256"], label="plan SHA-256")
        if result.to_dict() != dict(value):
            raise ValueError("heterogeneous Gemma plan authentication failed")
        return result


def validate_gemma3_heterogeneous_generator_plan(
    value: Gemma3HeterogeneousGeneratorPlan | Mapping[str, object],
) -> Gemma3HeterogeneousGeneratorPlan:
    if isinstance(value, Gemma3HeterogeneousGeneratorPlan):
        return Gemma3HeterogeneousGeneratorPlan.from_dict(value.to_dict())
    return Gemma3HeterogeneousGeneratorPlan.from_dict(value)


def load_gemma3_heterogeneous_generator_plan(
    source: Path | str,
) -> Gemma3HeterogeneousGeneratorPlan:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_gemma3_heterogeneous_generator_plan(value)


def _family_id(
    *,
    lift_scope: str,
    event_kind: str,
    event_id: str,
    attention_kind: str | None,
) -> str:
    if event_kind == "mlp":
        return "family.mlp.gated_gelu_h640_i2048"
    if event_kind == "attention":
        if lift_scope == LIFT_SCOPE_MLP_FAMILY:
            return f"family.attention.invocation.{event_id}"
        return (
            "family.attention.global_causal"
            if attention_kind == "global"
            else "family.attention.sliding_causal_512"
        )
    if event_kind in {"lexical_ingress", "lexical_egress"}:
        if lift_scope == LIFT_SCOPE_FULL_HETEROGENEOUS:
            return "family.lexical.tied"
        return f"family.{event_kind.replace('_', '.')}"
    if event_kind == "terminal":
        return "family.terminal.exact"
    raise ValueError("unknown event kind")


def _attention_contract(
    mutation: JointGraphGeneratorMutation,
) -> AttentionRuntimeContract:
    operator = mutation.compiled_operator
    if not isinstance(operator, Gemma3AttentionGenerator):
        raise TypeError("attention mutation lost its exact operator")
    attention = operator.config.attention
    return AttentionRuntimeContract(
        hidden_width=operator.config.residual_width,
        query_heads=attention.query_heads,
        key_value_heads=attention.key_value_heads,
        head_dimension=attention.head_dimension,
        attention_kind=attention.kind,
        window_size=attention.window_size,
        qk_norm=attention.qk_norm,
        rope=attention.rope is not None,
    )


def _build_invocations_and_families(
    *,
    lift_scope: str,
    exact_executor: Gemma3CausalCarrierTransportExecutor,
    atom_binding_by_event: Mapping[str, tuple[str, str]],
    hidden_width: int,
    vocabulary_size: int,
    mlp_width: int,
) -> tuple[tuple[GeneratorInvocationSpec, ...], tuple[GeneratorFamilySpec, ...]]:
    lexical_contract = LexicalRuntimeContract(
        hidden_width=hidden_width,
        vocabulary_size=vocabulary_size,
        tied_ingress_egress=True,
        final_logit_softcap=exact_executor.exact_executor.final_logit_softcap,
    )
    mlp_contract = MLPRuntimeContract(
        hidden_width=hidden_width,
        intermediate_width=mlp_width,
        activation="gelu_pytorch_tanh",
    )
    terminal_contract = TerminalRuntimeContract(
        hidden_width=hidden_width,
        normalization_kind="rmsnorm",
    )
    mutation_by_event = {
        mutation.event_spec.event_id: mutation
        for mutation in exact_executor.exact_executor.mutations
    }
    rows: list[tuple[str, str, str | None, str | None, object]] = []
    rows.append(
        (
            "lexical_ingress",
            "lexical.ingress",
            LEXICAL_BANK_ID,
            "atom.lexical.tied",
            lexical_contract,
        )
    )
    for event in exact_executor.transport_manifest.events:
        mutation = mutation_by_event[event.event_id]
        bank_id, atom_id = atom_binding_by_event[event.event_id]
        contract: object = (
            _attention_contract(mutation)
            if event.event_kind == "attention"
            else mlp_contract
        )
        rows.append((event.event_kind, event.event_id, bank_id, atom_id, contract))
    rows.extend(
        (
            (
                "terminal",
                "terminal.rmsnorm",
                None,
                None,
                terminal_contract,
            ),
            (
                "lexical_egress",
                "lexical.egress",
                LEXICAL_BANK_ID,
                "atom.lexical.tied",
                lexical_contract,
            ),
        )
    )
    invocations: list[GeneratorInvocationSpec] = []
    family_contracts: dict[str, object] = {}
    family_banks: dict[str, tuple[str, ...]] = {}
    for index, (event_kind, event_id, bank_id, atom_id, contract) in enumerate(rows):
        mutation = mutation_by_event.get(event_id)
        attention_kind = (
            None if mutation is None else mutation.event_spec.attention_kind
        )
        family_id = _family_id(
            lift_scope=lift_scope,
            event_kind=event_kind,
            event_id=event_id,
            attention_kind=attention_kind,
        )
        invocations.append(
            GeneratorInvocationSpec(
                invocation_index=index,
                invocation_id=(
                    "terminal.norm_softcap"
                    if event_kind == "terminal"
                    else event_id
                ),
                event_id=event_id,
                event_kind=event_kind,
                family_id=family_id,
                atom_bank_id=bank_id,
                atom_id=atom_id,
                exact_full_rank_lift=True,
                lift_input_rank=hidden_width,
                lift_output_rank=hidden_width,
            )
        )
        previous = family_contracts.setdefault(family_id, contract)
        if previous != contract:
            raise ValueError("one family received incompatible runtime contracts")
        family_banks.setdefault(
            family_id,
            () if bank_id is None else (bank_id,),
        )
    families = tuple(
        GeneratorFamilySpec(
            family_index=index,
            family_id=family_id,
            component_kind=contract.contract_kind,
            runtime_contract=contract,
            compatible_atom_bank_ids=family_banks[family_id],
        )
        for index, (family_id, contract) in enumerate(family_contracts.items())
    )
    return tuple(invocations), families


def _validate_exact_carrier_base(
    executor: Gemma3CausalCarrierTransportExecutor,
    *,
    require_pinned_gemma: bool,
) -> None:
    if not isinstance(executor, Gemma3CausalCarrierTransportExecutor):
        raise TypeError("exact carrier executor has the wrong type")
    inventory = executor.resource_inventory()
    architecture = executor.architecture_manifest()
    structural_failure = (
        inventory.get("exact_stored_scalar_parity") is not True
        or architecture.get("event_count") != PINNED_EVENT_COUNT
        or architecture.get("carrier_boundary_count") != PINNED_MUTATION_COUNT
        or architecture.get("cross_block_correction_message_count") != 0
        or architecture.get("compression_attempted") is not False
        or architecture.get("source_independent_runtime") is not True
        or executor.transport_manifest.correction_count != 0
    )
    pinned_failure = require_pinned_gemma and (
        inventory.get("exact_deployed_scalar_count") != PINNED_SOURCE_SCALARS
        or inventory.get("lexical_shared_physical_scalars")
        != PINNED_LEXICAL_SCALARS
        or inventory.get("mutation_scalars")
        != PINNED_ATTENTION_SCALARS + PINNED_MLP_SCALARS
        or inventory.get("terminal_norm_scalars")
        != PINNED_TERMINAL_NORM_SCALARS
        or executor.hidden_width != PINNED_HIDDEN_WIDTH
    )
    if structural_failure or pinned_failure:
        raise ValueError("exact carrier base does not match the pinned Gemma rung")


def _compile_physical_runtime(
    executor: Gemma3CausalCarrierTransportExecutor,
    *,
    hidden_width: int,
    vocabulary_size: int,
    source_stored_scalar_count: int,
    terminal_stored_scalar_count: int,
) -> tuple[
    tuple[GeneratorAtomSpec, ...],
    tuple[GeneratorAtomBankSpec, ...],
    dict[str, GeneratorAtomBankRuntime],
    Gemma3ExactTerminalRuntime,
    dict[str, tuple[str, str]],
]:
    exact = executor.exact_executor
    lexical = Gemma3TiedLexicalAtom(
        lexical=exact.lexical,
        hidden_width=hidden_width,
        vocabulary_size=vocabulary_size,
        final_logit_softcap=exact.final_logit_softcap,
    )
    grouped: dict[str, list[LexicalAtomRuntime | AttentionAtomRuntime | MLPAtomRuntime]] = {
        LEXICAL_BANK_ID: [lexical],
        MLP_BANK_ID: [],
        SLIDING_ATTENTION_BANK_ID: [],
        GLOBAL_ATTENTION_BANK_ID: [],
    }
    atom_binding_by_event: dict[str, tuple[str, str]] = {}
    for mutation in exact.mutations:
        block = mutation.event_spec.block_ordinal
        if type(block) is not int:
            raise ValueError("body mutation lost its block ordinal")
        if mutation.event_spec.event_kind == "attention_mutation":
            bank_id = (
                GLOBAL_ATTENTION_BANK_ID
                if block in PINNED_GLOBAL_ATTENTION_BLOCKS
                else SLIDING_ATTENTION_BANK_ID
            )
            atom_id = f"atom.attention.layer.{block}"
            atom = Gemma3AttentionMutationAtom(
                atom_id=atom_id,
                mutation=mutation,
            )
        else:
            bank_id = MLP_BANK_ID
            atom_id = f"atom.mlp.layer.{block}"
            atom = Gemma3MLPMutationAtom(
                atom_id=atom_id,
                mutation=mutation,
            )
        grouped[bank_id].append(atom)
        atom_binding_by_event[mutation.event_spec.event_id] = (
            bank_id,
            atom_id,
        )
    bank_kinds = {
        LEXICAL_BANK_ID: "lexical",
        MLP_BANK_ID: "mlp",
        SLIDING_ATTENTION_BANK_ID: "attention.sliding",
        GLOBAL_ATTENTION_BANK_ID: "attention.global",
    }
    atom_specs: list[GeneratorAtomSpec] = []
    bank_specs: list[GeneratorAtomBankSpec] = []
    bank_runtimes: dict[str, GeneratorAtomBankRuntime] = {}
    for bank_index, bank_id in enumerate(ORDERED_BANK_IDS):
        runtimes = grouped[bank_id]
        current_specs = tuple(
            build_generator_atom_spec(
                atom_index=len(atom_specs) + local_index,
                atom_bank_id=bank_id,
                atom=atom,
            )
            for local_index, atom in enumerate(runtimes)
        )
        atom_specs.extend(current_specs)
        bank_spec = build_generator_atom_bank_spec(
            bank_index=bank_index,
            atom_bank_id=bank_id,
            component_kind=bank_kinds[bank_id],
            atoms=current_specs,
        )
        bank_specs.append(bank_spec)
        bank_runtimes[bank_id] = GeneratorAtomBankRuntime(
            spec=bank_spec,
            atoms={atom.atom_id: atom for atom in runtimes},
        )
    terminal = Gemma3ExactTerminalRuntime(exact.terminal_norm)
    if (
        len(atom_specs) != PINNED_HETEROGENEOUS_ATOM_COUNT
        or sum(item.stored_scalar_count for item in atom_specs)
        != source_stored_scalar_count - terminal_stored_scalar_count
        or terminal.stored_scalar_count != terminal_stored_scalar_count
    ):
        raise RuntimeError("full-rank physical atom inventory did not close")
    return (
        tuple(atom_specs),
        tuple(bank_specs),
        bank_runtimes,
        terminal,
        atom_binding_by_event,
    )


@dataclass(slots=True)
class Gemma3HeterogeneousGeneratorCompilation:
    """Live atom inventory plus the strict three-arm plan and exact oracle."""

    exact_carrier_executor: Gemma3CausalCarrierTransportExecutor
    plan: Gemma3HeterogeneousGeneratorPlan
    atom_banks: Mapping[str, GeneratorAtomBankRuntime]
    terminal_runtime: Gemma3ExactTerminalRuntime

    def __post_init__(self) -> None:
        self.plan = validate_gemma3_heterogeneous_generator_plan(self.plan)
        _validate_exact_carrier_base(
            self.exact_carrier_executor,
            require_pinned_gemma=(
                self.plan.manifests[0].source_stored_scalar_count
                == PINNED_SOURCE_SCALARS
                and self.plan.manifests[0].hidden_width == PINNED_HIDDEN_WIDTH
            ),
        )
        if self.plan.source_model_fingerprint != self.exact_carrier_executor.source_model_fingerprint:
            raise ValueError("compilation plan belongs to another exact source")
        manifest = self.plan.manifests[-1]
        expected = {item.atom_bank_id: item for item in manifest.atom_banks}
        if not isinstance(self.atom_banks, Mapping) or set(self.atom_banks) != set(expected):
            raise ValueError("compilation atom-bank inventory drifted")
        for bank_id, runtime in self.atom_banks.items():
            if (
                not isinstance(runtime, GeneratorAtomBankRuntime)
                or runtime.spec != expected[bank_id]
            ):
                raise ValueError("compilation atom-bank runtime is invalid")
            runtime.assert_matches_spec()
        if (
            not isinstance(self.terminal_runtime, Gemma3ExactTerminalRuntime)
            or self.terminal_runtime.state_sha256
            != manifest.terminal_state_sha256
        ):
            raise ValueError("compilation terminal runtime drifted")

    def assemble(self, lift_scope: str) -> HeterogeneousGeneratorExecutor:
        return assemble_heterogeneous_generator_executor(
            manifest=self.plan.manifest_for_scope(lift_scope),
            atom_banks=self.atom_banks,
            terminal_runtime=self.terminal_runtime,
        )

    def metadata(self) -> dict[str, object]:
        final = self.plan.manifests[-1]
        return {
            "schema": GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA,
            "plan_sha256": self.plan.fingerprint_sha256,
            "source_model_fingerprint": self.plan.source_model_fingerprint,
            "exact_gate_chain": self.plan.exact_gate_chain,
            "lift_scopes": _ORDERED_SCOPES,
            "invocation_count": len(final.invocations),
            "edge_occurrence_count": len(final.edge_occurrences),
            "typed_atom_bank_count": len(final.atom_banks),
            "physical_atom_count": len(final.atoms),
            "generator_family_count": sum(
                item.component_kind != "terminal" for item in final.families
            ),
            "family_catalog_entry_count": len(final.families),
            "typed_terminal_boundary_count": 1,
            "deployed_scalar_count": final.deployed_scalar_count,
            "source_stored_scalar_count": final.source_stored_scalar_count,
            "exact_full_rank_scalar_parity": (
                final.deployed_scalar_count == final.source_stored_scalar_count
            ),
            "family_schema_sharing_parameter_credit": 0,
            "learned_cross_event_correction_count": 0,
            "compression_attempted": False,
            "compression_eligible": False,
            "model_loaded_by_compiler": False,
            "prompt_data_opened": False,
            "experiment_executed": False,
        }


@dataclass(frozen=True, slots=True)
class Gemma3HeterogeneousLiteralParityReceipt:
    """Literal carrier/message/output gate for one exact lift successor."""

    lift_scope: str
    compared_mutation_count: int
    lexical_message_literal_parity: bool
    mutation_message_literal_parity: tuple[bool, ...]
    carrier_boundary_literal_parity: tuple[bool, ...]
    final_carrier_literal_parity: bool
    normalized_output_literal_parity: bool
    logits_literal_parity: bool
    source_projection_call_count: int
    source_fallback_call_count: int
    learned_cross_event_correction_count: int
    compression_attempted: bool

    @property
    def passed(self) -> bool:
        return (
            self.compared_mutation_count == PINNED_MUTATION_COUNT
            and self.lexical_message_literal_parity
            and all(self.mutation_message_literal_parity)
            and all(self.carrier_boundary_literal_parity)
            and self.final_carrier_literal_parity
            and self.normalized_output_literal_parity
            and self.logits_literal_parity
            and self.source_projection_call_count == 0
            and self.source_fallback_call_count == 0
            and self.learned_cross_event_correction_count == 0
            and self.compression_attempted is False
        )

    def metadata(self) -> dict[str, object]:
        return {
            "lift_scope": self.lift_scope,
            "compared_mutation_count": self.compared_mutation_count,
            "lexical_message_literal_parity": (
                self.lexical_message_literal_parity
            ),
            "mutation_message_literal_parity": (
                self.mutation_message_literal_parity
            ),
            "carrier_boundary_literal_parity": (
                self.carrier_boundary_literal_parity
            ),
            "final_carrier_literal_parity": self.final_carrier_literal_parity,
            "normalized_output_literal_parity": (
                self.normalized_output_literal_parity
            ),
            "logits_literal_parity": self.logits_literal_parity,
            "source_projection_call_count": self.source_projection_call_count,
            "source_fallback_call_count": self.source_fallback_call_count,
            "learned_cross_event_correction_count": (
                self.learned_cross_event_correction_count
            ),
            "compression_attempted": self.compression_attempted,
            "passed": self.passed,
        }


def verify_gemma3_heterogeneous_literal_parity(
    *,
    lift_scope: str,
    reference: Gemma3CausalCarrierTransportExecution,
    candidate: HeterogeneousGeneratorExecution,
) -> Gemma3HeterogeneousLiteralParityReceipt:
    """Compare all 36 core commits plus terminal/logits with no tolerance."""

    if lift_scope not in _ORDERED_SCOPES:
        raise ValueError("literal parity lift scope is invalid")
    if not isinstance(reference, Gemma3CausalCarrierTransportExecution):
        raise TypeError("reference execution has the wrong type")
    if not isinstance(candidate, HeterogeneousGeneratorExecution):
        raise TypeError("candidate execution has the wrong type")
    reference_core = reference.core_execution_receipt
    candidate_core = candidate.receipt.carrier_execution_receipt
    if (
        reference_core.manifest_sha256 != candidate_core.manifest_sha256
        or len(reference_core.boundaries) != PINNED_MUTATION_COUNT
        or len(candidate_core.boundaries) != PINNED_MUTATION_COUNT
    ):
        raise ValueError("literal parity executions do not share one carrier base")
    mutation_parity: list[bool] = []
    boundary_parity: list[bool] = []
    for left, right in zip(
        reference_core.boundaries,
        candidate_core.boundaries,
        strict=True,
    ):
        mutation_parity.append(
            tuple(
                (
                    item.edge_id,
                    item.value_sha256,
                    item.routing_sha256,
                )
                for item in left.message_receipts
            )
            == tuple(
                (
                    item.edge_id,
                    item.value_sha256,
                    item.routing_sha256,
                )
                for item in right.message_receipts
            )
        )
        boundary_parity.append(
            left.event_id == right.event_id
            and left.input_carrier_sha256 == right.input_carrier_sha256
            and left.output_carrier_sha256 == right.output_carrier_sha256
        )
    receipt = candidate.receipt
    result = Gemma3HeterogeneousLiteralParityReceipt(
        lift_scope=lift_scope,
        compared_mutation_count=len(mutation_parity),
        lexical_message_literal_parity=(
            reference_core.lexical_message_receipt.value_sha256
            == candidate_core.lexical_message_receipt.value_sha256
        ),
        mutation_message_literal_parity=tuple(mutation_parity),
        carrier_boundary_literal_parity=tuple(boundary_parity),
        final_carrier_literal_parity=torch.equal(
            reference.output,
            candidate.output,
        ),
        normalized_output_literal_parity=torch.equal(
            reference.normalized_output,
            candidate.normalized_output,
        ),
        logits_literal_parity=torch.equal(reference.logits, candidate.logits),
        source_projection_call_count=receipt.source_projection_call_count,
        source_fallback_call_count=receipt.source_fallback_call_count,
        learned_cross_event_correction_count=(
            receipt.learned_cross_event_correction_count
        ),
        compression_attempted=receipt.compression_attempted,
    )
    return result


def compile_gemma3_exact_heterogeneous_generator_plan(
    exact_carrier_executor: Gemma3CausalCarrierTransportExecutor,
    *,
    require_pinned_gemma: bool = True,
) -> Gemma3HeterogeneousGeneratorCompilation:
    """Lift an already-compiled exact Gemma carrier into all exact arms."""

    if type(require_pinned_gemma) is not bool:
        raise TypeError("require_pinned_gemma must be bool")
    _validate_exact_carrier_base(
        exact_carrier_executor,
        require_pinned_gemma=require_pinned_gemma,
    )
    inventory = exact_carrier_executor.resource_inventory()
    hidden_width = exact_carrier_executor.hidden_width
    source_stored_scalar_count = int(inventory["exact_deployed_scalar_count"])
    lexical_stored_scalar_count = int(
        inventory["lexical_shared_physical_scalars"]
    )
    terminal_stored_scalar_count = int(inventory["terminal_norm_scalars"])
    if lexical_stored_scalar_count % hidden_width:
        raise ValueError("tied lexical matrix does not close vocabulary width")
    vocabulary_size = lexical_stored_scalar_count // hidden_width
    if require_pinned_gemma:
        vocabulary_size = PINNED_VOCABULARY_SIZE
        mlp_width = PINNED_MLP_WIDTH
    else:
        mlp_shapes = [
            dimension
            for mutation in exact_carrier_executor.exact_executor.mutations
            if mutation.event_spec.event_kind == "mlp_mutation"
            for parameter in mutation.parameters()
            for dimension in parameter.shape
        ]
        mlp_width = max((hidden_width, *mlp_shapes))
    (
        atoms,
        banks,
        bank_runtimes,
        terminal,
        atom_binding_by_event,
    ) = _compile_physical_runtime(
        exact_carrier_executor,
        hidden_width=hidden_width,
        vocabulary_size=vocabulary_size,
        source_stored_scalar_count=source_stored_scalar_count,
        terminal_stored_scalar_count=terminal_stored_scalar_count,
    )
    manifests: list[HeterogeneousGeneratorManifest] = []
    for scope in _ORDERED_SCOPES:
        invocations, families = _build_invocations_and_families(
            lift_scope=scope,
            exact_executor=exact_carrier_executor,
            atom_binding_by_event=atom_binding_by_event,
            hidden_width=hidden_width,
            vocabulary_size=vocabulary_size,
            mlp_width=mlp_width,
        )
        manifests.append(
            build_exact_heterogeneous_generator_manifest(
                arm_id=_SCOPE_ARM_IDS[scope],
                lift_scope=scope,
                hidden_width=hidden_width,
                source_stored_scalar_count=source_stored_scalar_count,
                terminal_stored_scalar_count=terminal.stored_scalar_count,
                terminal_state_sha256=terminal.state_sha256,
                carrier_manifest=exact_carrier_executor.transport_manifest,
                families=families,
                atom_banks=banks,
                atoms=atoms,
                invocations=invocations,
            )
        )
    plan = Gemma3HeterogeneousGeneratorPlan(
        source_model_fingerprint=exact_carrier_executor.source_model_fingerprint,
        exact_carrier_manifest_sha256=(
            exact_carrier_executor.transport_manifest.fingerprint_sha256
        ),
        manifests=tuple(manifests),
        protocol_sha256=_protocol_sha256(),
    )
    return Gemma3HeterogeneousGeneratorCompilation(
        exact_carrier_executor=exact_carrier_executor,
        plan=plan,
        atom_banks=bank_runtimes,
        terminal_runtime=terminal,
    )


def assemble_gemma3_heterogeneous_generator_executor(
    compilation: Gemma3HeterogeneousGeneratorCompilation,
    *,
    lift_scope: str,
) -> HeterogeneousGeneratorExecutor:
    if not isinstance(compilation, Gemma3HeterogeneousGeneratorCompilation):
        raise TypeError("compilation has the wrong type")
    return compilation.assemble(lift_scope)


__all__ = [
    "COMPATIBLE_ATTENTION_ARM_ID",
    "EXACT_CARRIER_ARM_ID",
    "EXACT_LIFT_GATE_CHAIN",
    "FULL_HETEROGENEOUS_ARM_ID",
    "GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_FORMAT_VERSION",
    "GEMMA3_HETEROGENEOUS_GENERATOR_PLAN_SCHEMA",
    "GLOBAL_ATTENTION_BANK_ID",
    "Gemma3AttentionMutationAtom",
    "Gemma3ExactTerminalRuntime",
    "Gemma3HeterogeneousGeneratorCompilation",
    "Gemma3HeterogeneousLiteralParityReceipt",
    "Gemma3HeterogeneousGeneratorPlan",
    "Gemma3MLPMutationAtom",
    "Gemma3TiedLexicalAtom",
    "LEXICAL_BANK_ID",
    "MLP_BANK_ID",
    "MLP_FAMILY_ARM_ID",
    "ORDERED_BANK_IDS",
    "SLIDING_ATTENTION_BANK_ID",
    "assemble_gemma3_heterogeneous_generator_executor",
    "compile_gemma3_exact_heterogeneous_generator_plan",
    "load_gemma3_heterogeneous_generator_plan",
    "validate_gemma3_heterogeneous_generator_plan",
    "verify_gemma3_heterogeneous_literal_parity",
]
