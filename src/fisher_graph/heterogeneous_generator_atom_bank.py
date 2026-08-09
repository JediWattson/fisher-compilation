"""Heterogeneous generator-family and physical atom-bank runtime.

This module adds one identity layer above the exact causal carrier transport:

``temporal invocation -> logical family -> physical atom bank -> exact atom``.

The identities are deliberately independent.  Repeating a family never
collapses temporal edge occurrences, and physical storage is counted only by
atom identity.  Version 1 is an exact, full-rank representation rung: every
family lift is the literal atom output, learned cross-event corrections are
disabled, and no compression claim is permitted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor, nn

from .causal_carrier_transport_graph import (
    CarrierTransportExecution,
    CarrierTransportExecutionReceipt,
    CarrierTransportManifest,
    TransportContext,
    TransportMessage,
    execute_causal_carrier_transport_graph,
    validate_carrier_transport_execution_receipt,
    validate_carrier_transport_manifest,
)


HETEROGENEOUS_GENERATOR_MANIFEST_SCHEMA = (
    "fisher_graph.heterogeneous_generator_atom_bank.manifest.v1"
)
HETEROGENEOUS_GENERATOR_RECEIPT_SCHEMA = (
    "fisher_graph.heterogeneous_generator_atom_bank.execution.v1"
)
HETEROGENEOUS_GENERATOR_FORMAT_VERSION = 1

LIFT_SCOPE_MLP_FAMILY = "mlp_family_lift"
LIFT_SCOPE_COMPATIBLE_ATTENTION = "compatible_attention_lift"
LIFT_SCOPE_FULL_HETEROGENEOUS = "heterogeneous_full_rank_lift"
LIFT_SCOPES = (
    LIFT_SCOPE_MLP_FAMILY,
    LIFT_SCOPE_COMPATIBLE_ATTENTION,
    LIFT_SCOPE_FULL_HETEROGENEOUS,
)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_MANIFEST_DOMAIN = b"fisher_graph:heterogeneous-generator-manifest:v1\0"
_RECEIPT_DOMAIN = b"fisher_graph:heterogeneous-generator-receipt:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:heterogeneous-generator-tensor:v1\0"
_MODULE_DOMAIN = b"fisher_graph:heterogeneous-generator-module:v1\0"
_BANK_DOMAIN = b"fisher_graph:heterogeneous-generator-bank:v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def tensor_sha256(value: Tensor) -> str:
    """Hash exact dense tensor bytes with a dtype/shape domain."""

    if not isinstance(value, Tensor):
        raise TypeError("only torch tensors can be authenticated")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(_TENSOR_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    # Flatten first because PyTorch cannot reinterpret a rank-zero scalar as
    # bytes directly when the element sizes differ.
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def module_physical_scalar_count(module: nn.Module) -> int:
    """Count each parameter/persistent-buffer storage once inside a module."""

    if not isinstance(module, nn.Module):
        raise TypeError("physical scalar accounting requires an nn.Module")
    values = tuple(module.state_dict(keep_vars=True).values())
    seen: set[int] = set()
    total = 0
    for value in values:
        if value.numel() == 0:
            continue
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += value.numel()
    return total


def module_state_sha256(module: nn.Module) -> str:
    """Authenticate unique parameter/persistent-buffer state by stable name."""

    if not isinstance(module, nn.Module):
        raise TypeError("module state hashing requires an nn.Module")
    values = list(module.state_dict(keep_vars=True).items())
    digest = hashlib.sha256(_MODULE_DOMAIN)
    seen: set[int] = set()
    for name, value in sorted(values, key=lambda item: item[0]):
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


def _module_persistent_storage_keys(module: nn.Module) -> frozenset[tuple[str, int]]:
    """Identify nonempty parameter/persistent-buffer storages for alias audits."""

    if not isinstance(module, nn.Module):
        raise TypeError("storage-key audit requires an nn.Module")
    return frozenset(
        (str(value.device), value.untyped_storage().data_ptr())
        for value in module.state_dict(keep_vars=True).values()
        if value.numel()
    )


@dataclass(frozen=True, slots=True)
class LexicalRuntimeContract:
    hidden_width: int
    vocabulary_size: int
    tied_ingress_egress: bool = True
    final_logit_softcap: float | None = None
    contract_kind: str = "lexical"

    def __post_init__(self) -> None:
        _require_int(self.hidden_width, label="lexical hidden_width", minimum=1)
        _require_int(
            self.vocabulary_size,
            label="lexical vocabulary_size",
            minimum=2,
        )
        if self.tied_ingress_egress is not True:
            raise ValueError("exact lexical runtime must remain tied")
        if self.final_logit_softcap is not None and (
            isinstance(self.final_logit_softcap, bool)
            or not isinstance(self.final_logit_softcap, (float, int))
            or not math.isfinite(float(self.final_logit_softcap))
            or float(self.final_logit_softcap) <= 0
        ):
            raise ValueError("final logit softcap must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_kind": self.contract_kind,
            "hidden_width": self.hidden_width,
            "vocabulary_size": self.vocabulary_size,
            "tied_ingress_egress": self.tied_ingress_egress,
            "final_logit_softcap": self.final_logit_softcap,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LexicalRuntimeContract:
        if not isinstance(value, Mapping) or set(value) != {
            "contract_kind",
            "hidden_width",
            "vocabulary_size",
            "tied_ingress_egress",
            "final_logit_softcap",
        }:
            raise ValueError("lexical contract fields are not exact")
        if value["contract_kind"] != "lexical":
            raise ValueError("lexical contract kind drifted")
        return cls(
            hidden_width=value["hidden_width"],
            vocabulary_size=value["vocabulary_size"],
            tied_ingress_egress=value["tied_ingress_egress"],
            final_logit_softcap=value["final_logit_softcap"],
        )


@dataclass(frozen=True, slots=True)
class AttentionRuntimeContract:
    hidden_width: int
    query_heads: int
    key_value_heads: int
    head_dimension: int
    attention_kind: str
    window_size: int | None
    qk_norm: bool
    rope: bool
    contract_kind: str = "attention"

    def __post_init__(self) -> None:
        for label, value in (
            ("hidden_width", self.hidden_width),
            ("query_heads", self.query_heads),
            ("key_value_heads", self.key_value_heads),
            ("head_dimension", self.head_dimension),
        ):
            _require_int(value, label=f"attention {label}", minimum=1)
        if self.query_heads % self.key_value_heads:
            raise ValueError("attention heads do not define valid GQA groups")
        if self.attention_kind not in {"global_causal", "sliding_causal"}:
            raise ValueError("attention kind is invalid")
        if self.attention_kind == "global_causal":
            if self.window_size is not None:
                raise ValueError("global attention cannot declare a window")
        elif type(self.window_size) is not int or self.window_size <= 0:
            raise ValueError("sliding attention requires a positive window")
        if type(self.qk_norm) is not bool or self.rope is not True:
            raise ValueError("attention qk_norm/rope contract drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_kind": self.contract_kind,
            "hidden_width": self.hidden_width,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dimension": self.head_dimension,
            "attention_kind": self.attention_kind,
            "window_size": self.window_size,
            "qk_norm": self.qk_norm,
            "rope": self.rope,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AttentionRuntimeContract:
        expected = {
            "contract_kind",
            "hidden_width",
            "query_heads",
            "key_value_heads",
            "head_dimension",
            "attention_kind",
            "window_size",
            "qk_norm",
            "rope",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("attention contract fields are not exact")
        if value["contract_kind"] != "attention":
            raise ValueError("attention contract kind drifted")
        return cls(**{key: value[key] for key in expected if key != "contract_kind"})


@dataclass(frozen=True, slots=True)
class MLPRuntimeContract:
    hidden_width: int
    intermediate_width: int
    activation: str
    gated: bool = True
    token_local: bool = True
    contract_kind: str = "mlp"

    def __post_init__(self) -> None:
        _require_int(self.hidden_width, label="MLP hidden_width", minimum=1)
        _require_int(
            self.intermediate_width,
            label="MLP intermediate_width",
            minimum=1,
        )
        _require_id(self.activation, label="MLP activation")
        if self.gated is not True or self.token_local is not True:
            raise ValueError("exact MLP must remain gated and token-local")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_kind": self.contract_kind,
            "hidden_width": self.hidden_width,
            "intermediate_width": self.intermediate_width,
            "activation": self.activation,
            "gated": self.gated,
            "token_local": self.token_local,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MLPRuntimeContract:
        expected = {
            "contract_kind",
            "hidden_width",
            "intermediate_width",
            "activation",
            "gated",
            "token_local",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("MLP contract fields are not exact")
        if value["contract_kind"] != "mlp":
            raise ValueError("MLP contract kind drifted")
        return cls(**{key: value[key] for key in expected if key != "contract_kind"})


@dataclass(frozen=True, slots=True)
class TerminalRuntimeContract:
    hidden_width: int
    normalization_kind: str = "rmsnorm"
    contract_kind: str = "terminal"

    def __post_init__(self) -> None:
        _require_int(self.hidden_width, label="terminal hidden_width", minimum=1)
        if self.normalization_kind != "rmsnorm":
            raise ValueError("terminal normalization must be RMSNorm")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_kind": self.contract_kind,
            "hidden_width": self.hidden_width,
            "normalization_kind": self.normalization_kind,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TerminalRuntimeContract:
        if not isinstance(value, Mapping) or set(value) != {
            "contract_kind",
            "hidden_width",
            "normalization_kind",
        }:
            raise ValueError("terminal contract fields are not exact")
        if value["contract_kind"] != "terminal":
            raise ValueError("terminal contract kind drifted")
        return cls(
            hidden_width=value["hidden_width"],
            normalization_kind=value["normalization_kind"],
        )


RuntimeContract = (
    LexicalRuntimeContract
    | AttentionRuntimeContract
    | MLPRuntimeContract
    | TerminalRuntimeContract
)


def runtime_contract_from_dict(value: Mapping[str, object]) -> RuntimeContract:
    if not isinstance(value, Mapping):
        raise TypeError("runtime contract must be a mapping")
    kind = value.get("contract_kind")
    parser = {
        "lexical": LexicalRuntimeContract.from_dict,
        "attention": AttentionRuntimeContract.from_dict,
        "mlp": MLPRuntimeContract.from_dict,
        "terminal": TerminalRuntimeContract.from_dict,
    }.get(kind)
    if parser is None:
        raise ValueError("runtime contract kind is invalid")
    return parser(value)


@dataclass(frozen=True, slots=True)
class GeneratorAtomSpec:
    atom_index: int
    atom_id: str
    atom_bank_id: str
    component_kind: str
    output_rank: int
    stored_scalar_count: int
    state_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.atom_index, label="atom_index")
        _require_id(self.atom_id, label="atom_id")
        _require_id(self.atom_bank_id, label="atom_bank_id")
        if self.atom_id == self.atom_bank_id:
            raise ValueError("atom identity must differ from bank identity")
        if self.component_kind not in {"lexical", "attention", "mlp"}:
            raise ValueError("physical atom component kind is invalid")
        _require_int(self.output_rank, label="atom output_rank", minimum=1)
        _require_int(
            self.stored_scalar_count,
            label="atom stored_scalar_count",
            minimum=1,
        )
        _require_sha256(self.state_sha256, label="atom state_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_index": self.atom_index,
            "atom_id": self.atom_id,
            "atom_bank_id": self.atom_bank_id,
            "component_kind": self.component_kind,
            "output_rank": self.output_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GeneratorAtomSpec:
        expected = {
            "atom_index",
            "atom_id",
            "atom_bank_id",
            "component_kind",
            "output_rank",
            "stored_scalar_count",
            "state_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("atom fields are not exact")
        return cls(**{key: value[key] for key in expected})


def _bank_sha256(atoms: Sequence[GeneratorAtomSpec]) -> str:
    # ``atom_index`` belongs to the manifest-wide catalog, not to physical
    # bank state.  Excluding it lets a bank authenticate itself without being
    # coupled to where another bank happened to appear in that catalog.
    return _fingerprint(
        _BANK_DOMAIN,
        [
            {
                key: value
                for key, value in atom.to_dict().items()
                if key != "atom_index"
            }
            for atom in atoms
        ],
    )


def build_generator_atom_spec(
    *,
    atom_index: int,
    atom_bank_id: str,
    atom: GeneratorAtomRuntime,
) -> GeneratorAtomSpec:
    """Describe one live physical atom without conflating its identities."""

    if not isinstance(atom, GeneratorAtomRuntime):
        raise TypeError("atom must implement GeneratorAtomRuntime")
    return GeneratorAtomSpec(
        atom_index=atom_index,
        atom_id=atom.atom_id,
        atom_bank_id=atom_bank_id,
        component_kind=atom.component_kind,
        output_rank=atom.output_rank,
        stored_scalar_count=atom.stored_scalar_count,
        state_sha256=atom.state_sha256,
    )


def build_generator_atom_bank_spec(
    *,
    bank_index: int,
    atom_bank_id: str,
    component_kind: str,
    atoms: Sequence[GeneratorAtomSpec],
) -> GeneratorAtomBankSpec:
    """Build an authenticated bank catalog from manifest-wide atom specs."""

    ordered = tuple(atoms)
    if not ordered:
        raise ValueError("atom bank cannot be empty")
    if any(atom.atom_bank_id != atom_bank_id for atom in ordered):
        raise ValueError("atom-bank ownership disagrees with atom specs")
    return GeneratorAtomBankSpec(
        bank_index=bank_index,
        atom_bank_id=atom_bank_id,
        component_kind=component_kind,
        atom_ids=tuple(atom.atom_id for atom in ordered),
        physical_stored_scalar_count=sum(
            atom.stored_scalar_count for atom in ordered
        ),
        state_sha256=_bank_sha256(ordered),
    )


@dataclass(frozen=True, slots=True)
class GeneratorAtomBankSpec:
    bank_index: int
    atom_bank_id: str
    component_kind: str
    atom_ids: tuple[str, ...]
    physical_stored_scalar_count: int
    state_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.bank_index, label="bank_index")
        _require_id(self.atom_bank_id, label="atom_bank_id")
        if self.component_kind not in {
            "lexical",
            "attention.global",
            "attention.sliding",
            "mlp",
        }:
            raise ValueError("atom bank component kind is invalid")
        if type(self.atom_ids) is not tuple or not self.atom_ids:
            raise ValueError("atom bank must contain atom IDs")
        if len(set(self.atom_ids)) != len(self.atom_ids):
            raise ValueError("atom bank atom IDs must be unique")
        for atom_id in self.atom_ids:
            _require_id(atom_id, label="bank atom_id")
            if atom_id == self.atom_bank_id:
                raise ValueError("atom identity must differ from bank identity")
        _require_int(
            self.physical_stored_scalar_count,
            label="bank physical_stored_scalar_count",
            minimum=1,
        )
        _require_sha256(self.state_sha256, label="bank state_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "bank_index": self.bank_index,
            "atom_bank_id": self.atom_bank_id,
            "component_kind": self.component_kind,
            "atom_ids": list(self.atom_ids),
            "physical_stored_scalar_count": self.physical_stored_scalar_count,
            "state_sha256": self.state_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GeneratorAtomBankSpec:
        expected = {
            "bank_index",
            "atom_bank_id",
            "component_kind",
            "atom_ids",
            "physical_stored_scalar_count",
            "state_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("atom-bank fields are not exact")
        atom_ids = value["atom_ids"]
        if not isinstance(atom_ids, Sequence) or isinstance(atom_ids, (str, bytes)):
            raise TypeError("serialized atom IDs must be a sequence")
        return cls(
            bank_index=value["bank_index"],
            atom_bank_id=value["atom_bank_id"],
            component_kind=value["component_kind"],
            atom_ids=tuple(atom_ids),
            physical_stored_scalar_count=value["physical_stored_scalar_count"],
            state_sha256=value["state_sha256"],
        )


@dataclass(frozen=True, slots=True)
class GeneratorFamilySpec:
    family_index: int
    family_id: str
    component_kind: str
    runtime_contract: RuntimeContract
    compatible_atom_bank_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_int(self.family_index, label="family_index")
        _require_id(self.family_id, label="family_id")
        if self.component_kind not in {
            "lexical",
            "attention",
            "mlp",
            "terminal",
        }:
            raise ValueError("family component kind is invalid")
        if self.runtime_contract.contract_kind != self.component_kind:
            raise ValueError("family kind disagrees with runtime contract")
        if type(self.compatible_atom_bank_ids) is not tuple:
            raise TypeError("compatible atom-bank IDs must be a tuple")
        if self.component_kind == "terminal":
            if self.compatible_atom_bank_ids:
                raise ValueError("terminal family remains outside physical banks")
        elif not self.compatible_atom_bank_ids:
            raise ValueError("generator family must name a physical atom bank")
        for bank_id in self.compatible_atom_bank_ids:
            _require_id(bank_id, label="compatible atom_bank_id")
            if bank_id == self.family_id:
                raise ValueError("family identity must differ from bank identity")

    @property
    def contract_sha256(self) -> str:
        return _fingerprint(_MANIFEST_DOMAIN, self.runtime_contract.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "family_index": self.family_index,
            "family_id": self.family_id,
            "component_kind": self.component_kind,
            "runtime_contract": self.runtime_contract.to_dict(),
            "runtime_contract_sha256": self.contract_sha256,
            "compatible_atom_bank_ids": list(self.compatible_atom_bank_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GeneratorFamilySpec:
        expected = {
            "family_index",
            "family_id",
            "component_kind",
            "runtime_contract",
            "runtime_contract_sha256",
            "compatible_atom_bank_ids",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("generator-family fields are not exact")
        contract = runtime_contract_from_dict(value["runtime_contract"])
        banks = value["compatible_atom_bank_ids"]
        if not isinstance(banks, Sequence) or isinstance(banks, (str, bytes)):
            raise TypeError("compatible atom-bank IDs must be a sequence")
        result = cls(
            family_index=value["family_index"],
            family_id=value["family_id"],
            component_kind=value["component_kind"],
            runtime_contract=contract,
            compatible_atom_bank_ids=tuple(banks),
        )
        _require_sha256(
            value["runtime_contract_sha256"],
            label="runtime_contract_sha256",
        )
        if result.to_dict() != dict(value):
            raise ValueError("generator-family authentication failed")
        return result


@dataclass(frozen=True, slots=True)
class GeneratorInvocationSpec:
    invocation_index: int
    invocation_id: str
    event_id: str
    event_kind: str
    family_id: str
    atom_bank_id: str | None
    atom_id: str | None
    exact_full_rank_lift: bool
    lift_input_rank: int
    lift_output_rank: int

    def __post_init__(self) -> None:
        _require_int(self.invocation_index, label="invocation_index")
        _require_id(self.invocation_id, label="invocation_id")
        _require_id(self.event_id, label="invocation event_id")
        _require_id(self.family_id, label="invocation family_id")
        if self.invocation_id == self.family_id:
            raise ValueError("invocation identity must differ from family identity")
        if self.event_kind not in {
            "lexical_ingress",
            "attention",
            "mlp",
            "terminal",
            "lexical_egress",
        }:
            raise ValueError("invocation event kind is invalid")
        if self.event_kind == "terminal":
            if self.atom_bank_id is not None or self.atom_id is not None:
                raise ValueError("terminal invocation is an exact external boundary")
        else:
            _require_id(self.atom_bank_id, label="invocation atom_bank_id")
            _require_id(self.atom_id, label="invocation atom_id")
            if self.family_id in {self.atom_bank_id, self.atom_id}:
                raise ValueError("family identity must differ from bank and atom")
            if self.invocation_id in {self.atom_bank_id, self.atom_id}:
                raise ValueError("invocation identity must differ from bank and atom")
        if self.exact_full_rank_lift is not True:
            raise ValueError("version 1 permits only exact full-rank family lifts")
        _require_int(self.lift_input_rank, label="lift_input_rank", minimum=1)
        _require_int(self.lift_output_rank, label="lift_output_rank", minimum=1)
        if self.lift_input_rank != self.lift_output_rank:
            raise ValueError("exact full-rank lift cannot change rank")

    def to_dict(self) -> dict[str, object]:
        return {
            "invocation_index": self.invocation_index,
            "invocation_id": self.invocation_id,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "family_id": self.family_id,
            "atom_bank_id": self.atom_bank_id,
            "atom_id": self.atom_id,
            "exact_full_rank_lift": self.exact_full_rank_lift,
            "lift_input_rank": self.lift_input_rank,
            "lift_output_rank": self.lift_output_rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GeneratorInvocationSpec:
        expected = {
            "invocation_index",
            "invocation_id",
            "event_id",
            "event_kind",
            "family_id",
            "atom_bank_id",
            "atom_id",
            "exact_full_rank_lift",
            "lift_input_rank",
            "lift_output_rank",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("generator-invocation fields are not exact")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class FamilyEdgeOccurrenceSpec:
    occurrence_index: int
    occurrence_id: str
    message_id: str
    message_kind: str
    source_invocation_id: str | None
    target_invocation_id: str
    source_invocation_index: int | None
    target_invocation_index: int
    emitting_family_id: str
    atom_bank_id: str | None
    atom_id: str | None
    carrier_edge_id: str | None
    learned_cross_event_correction: bool = False

    def __post_init__(self) -> None:
        _require_int(self.occurrence_index, label="occurrence_index")
        _require_id(self.occurrence_id, label="occurrence_id")
        _require_id(self.message_id, label="message_id")
        _require_id(self.message_kind, label="message_kind")
        _require_id(self.target_invocation_id, label="target_invocation_id")
        _require_int(
            self.target_invocation_index,
            label="target_invocation_index",
        )
        _require_id(self.emitting_family_id, label="emitting_family_id")
        if self.source_invocation_id is None:
            if self.source_invocation_index is not None:
                raise ValueError("source invocation ID/index optionality drifted")
        else:
            _require_id(self.source_invocation_id, label="source_invocation_id")
            _require_int(
                self.source_invocation_index,
                label="source_invocation_index",
            )
            if self.source_invocation_index >= self.target_invocation_index:
                raise ValueError("edge occurrences must remain strictly forward")
        if (self.atom_bank_id is None) != (self.atom_id is None):
            raise ValueError("edge atom bank/atom optionality drifted")
        if self.atom_bank_id is not None:
            _require_id(self.atom_bank_id, label="edge atom_bank_id")
            _require_id(self.atom_id, label="edge atom_id")
        if self.carrier_edge_id is not None:
            _require_id(self.carrier_edge_id, label="carrier_edge_id")
        if type(self.learned_cross_event_correction) is not bool:
            raise TypeError("learned correction marker must be bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrence_index": self.occurrence_index,
            "occurrence_id": self.occurrence_id,
            "message_id": self.message_id,
            "message_kind": self.message_kind,
            "source_invocation_id": self.source_invocation_id,
            "target_invocation_id": self.target_invocation_id,
            "source_invocation_index": self.source_invocation_index,
            "target_invocation_index": self.target_invocation_index,
            "emitting_family_id": self.emitting_family_id,
            "atom_bank_id": self.atom_bank_id,
            "atom_id": self.atom_id,
            "carrier_edge_id": self.carrier_edge_id,
            "learned_cross_event_correction": (
                self.learned_cross_event_correction
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FamilyEdgeOccurrenceSpec:
        expected = {
            "occurrence_index",
            "occurrence_id",
            "message_id",
            "message_kind",
            "source_invocation_id",
            "target_invocation_id",
            "source_invocation_index",
            "target_invocation_index",
            "emitting_family_id",
            "atom_bank_id",
            "atom_id",
            "carrier_edge_id",
            "learned_cross_event_correction",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("family-edge occurrence fields are not exact")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class HeterogeneousGeneratorManifest:
    arm_id: str
    lift_scope: str
    hidden_width: int
    source_stored_scalar_count: int
    carrier_manifest: CarrierTransportManifest
    families: tuple[GeneratorFamilySpec, ...]
    atom_banks: tuple[GeneratorAtomBankSpec, ...]
    atoms: tuple[GeneratorAtomSpec, ...]
    invocations: tuple[GeneratorInvocationSpec, ...]
    edge_occurrences: tuple[FamilyEdgeOccurrenceSpec, ...]
    terminal_stored_scalar_count: int
    terminal_state_sha256: str
    learned_cross_event_corrections_enabled: bool = False
    compression_attempted: bool = False
    schema: str = HETEROGENEOUS_GENERATOR_MANIFEST_SCHEMA
    format_version: int = HETEROGENEOUS_GENERATOR_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != HETEROGENEOUS_GENERATOR_MANIFEST_SCHEMA:
            raise ValueError("heterogeneous manifest schema is unsupported")
        if self.format_version != HETEROGENEOUS_GENERATOR_FORMAT_VERSION:
            raise ValueError("heterogeneous manifest version is unsupported")
        _require_id(self.arm_id, label="arm_id")
        if self.lift_scope not in LIFT_SCOPES:
            raise ValueError("lift_scope is invalid")
        _require_int(self.hidden_width, label="hidden_width", minimum=1)
        _require_int(
            self.source_stored_scalar_count,
            label="source_stored_scalar_count",
            minimum=1,
        )
        _require_int(
            self.terminal_stored_scalar_count,
            label="terminal_stored_scalar_count",
            minimum=1,
        )
        _require_sha256(
            self.terminal_state_sha256,
            label="terminal_state_sha256",
        )
        if self.learned_cross_event_corrections_enabled is not False:
            raise ValueError("learned cross-event corrections are disabled in v1")
        if self.compression_attempted is not False:
            raise ValueError("exact family lift cannot attempt compression")
        validate_carrier_transport_manifest(self.carrier_manifest)
        _validate_manifest(self)

    @property
    def physical_atom_scalar_count(self) -> int:
        return sum(bank.physical_stored_scalar_count for bank in self.atom_banks)

    @property
    def deployed_scalar_count(self) -> int:
        return self.physical_atom_scalar_count + self.terminal_stored_scalar_count

    @property
    def fingerprint_sha256(self) -> str:
        return _fingerprint(_MANIFEST_DOMAIN, self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "arm_id": self.arm_id,
            "lift_scope": self.lift_scope,
            "hidden_width": self.hidden_width,
            "source_stored_scalar_count": self.source_stored_scalar_count,
            "terminal_stored_scalar_count": self.terminal_stored_scalar_count,
            "terminal_state_sha256": self.terminal_state_sha256,
            "carrier_manifest": self.carrier_manifest.to_dict(),
            "families": [item.to_dict() for item in self.families],
            "atom_banks": [item.to_dict() for item in self.atom_banks],
            "atoms": [item.to_dict() for item in self.atoms],
            "invocations": [item.to_dict() for item in self.invocations],
            "edge_occurrences": [item.to_dict() for item in self.edge_occurrences],
            "learned_cross_event_corrections_enabled": (
                self.learned_cross_event_corrections_enabled
            ),
            "compression_attempted": self.compression_attempted,
            "accounting": {
                "invocation_count": len(self.invocations),
                "family_catalog_entry_count": len(self.families),
                "generator_family_count": sum(
                    item.component_kind != "terminal" for item in self.families
                ),
                "typed_terminal_boundary_count": sum(
                    item.component_kind == "terminal" for item in self.families
                ),
                "atom_bank_count": len(self.atom_banks),
                "physical_atom_count": len(self.atoms),
                "edge_occurrence_count": len(self.edge_occurrences),
                "physical_atom_scalar_count": self.physical_atom_scalar_count,
                "terminal_stored_scalar_count": self.terminal_stored_scalar_count,
                "deployed_scalar_count": self.deployed_scalar_count,
                "source_stored_scalar_count": self.source_stored_scalar_count,
                "parameter_difference": (
                    self.source_stored_scalar_count - self.deployed_scalar_count
                ),
                "learned_cross_event_correction_count": 0,
            },
            "claims": {
                "exact_full_rank_family_lift": True,
                "invocation_family_atom_identities_distinct": True,
                "family_repetition_collapses_invocations": False,
                "physical_storage_counted_by_atom_identity": True,
                "terminal_remains_typed_exact": True,
                "terminal_is_generator_family": False,
                "compression_eligible": False,
                "compression_success_claimed": False,
                "source_free_serialized_artifact_established": False,
            },
        }

    def to_dict(self) -> dict[str, object]:
        result = self.payload()
        result["manifest_sha256"] = self.fingerprint_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HeterogeneousGeneratorManifest:
        expected = {
            "schema",
            "format_version",
            "arm_id",
            "lift_scope",
            "hidden_width",
            "source_stored_scalar_count",
            "terminal_stored_scalar_count",
            "terminal_state_sha256",
            "carrier_manifest",
            "families",
            "atom_banks",
            "atoms",
            "invocations",
            "edge_occurrences",
            "learned_cross_event_corrections_enabled",
            "compression_attempted",
            "accounting",
            "claims",
            "manifest_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("heterogeneous manifest fields are not exact")
        for name in (
            "families",
            "atom_banks",
            "atoms",
            "invocations",
            "edge_occurrences",
        ):
            if not isinstance(value[name], Sequence) or isinstance(
                value[name], (str, bytes)
            ):
                raise TypeError(f"serialized {name} must be a sequence")
        result = cls(
            arm_id=value["arm_id"],
            lift_scope=value["lift_scope"],
            hidden_width=value["hidden_width"],
            source_stored_scalar_count=value["source_stored_scalar_count"],
            terminal_stored_scalar_count=value["terminal_stored_scalar_count"],
            terminal_state_sha256=value["terminal_state_sha256"],
            carrier_manifest=CarrierTransportManifest.from_dict(
                value["carrier_manifest"]
            ),
            families=tuple(GeneratorFamilySpec.from_dict(item) for item in value["families"]),
            atom_banks=tuple(GeneratorAtomBankSpec.from_dict(item) for item in value["atom_banks"]),
            atoms=tuple(GeneratorAtomSpec.from_dict(item) for item in value["atoms"]),
            invocations=tuple(GeneratorInvocationSpec.from_dict(item) for item in value["invocations"]),
            edge_occurrences=tuple(
                FamilyEdgeOccurrenceSpec.from_dict(item)
                for item in value["edge_occurrences"]
            ),
            learned_cross_event_corrections_enabled=value[
                "learned_cross_event_corrections_enabled"
            ],
            compression_attempted=value["compression_attempted"],
            schema=value["schema"],
            format_version=value["format_version"],
        )
        _require_sha256(value["manifest_sha256"], label="manifest_sha256")
        if result.to_dict() != dict(value):
            raise ValueError("heterogeneous manifest authentication failed")
        return result


def _validate_manifest(manifest: HeterogeneousGeneratorManifest) -> None:
    families = manifest.families
    banks = manifest.atom_banks
    atoms = manifest.atoms
    invocations = manifest.invocations
    occurrences = manifest.edge_occurrences
    if [item.family_index for item in families] != list(range(len(families))):
        raise ValueError("family indexes must be contiguous")
    if [item.bank_index for item in banks] != list(range(len(banks))):
        raise ValueError("atom-bank indexes must be contiguous")
    if [item.atom_index for item in atoms] != list(range(len(atoms))):
        raise ValueError("atom indexes must be contiguous")
    if [item.invocation_index for item in invocations] != list(
        range(len(invocations))
    ):
        raise ValueError("invocation indexes must be contiguous")
    if [item.occurrence_index for item in occurrences] != list(
        range(len(occurrences))
    ):
        raise ValueError("edge-occurrence indexes must be contiguous")
    for label, values in (
        ("family", [item.family_id for item in families]),
        ("atom bank", [item.atom_bank_id for item in banks]),
        ("atom", [item.atom_id for item in atoms]),
        ("invocation", [item.invocation_id for item in invocations]),
        ("occurrence", [item.occurrence_id for item in occurrences]),
        ("message", [item.message_id for item in occurrences]),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} IDs must be unique")
    family_by_id = {item.family_id: item for item in families}
    bank_by_id = {item.atom_bank_id: item for item in banks}
    atom_by_id = {item.atom_id: item for item in atoms}
    invocation_by_id = {item.invocation_id: item for item in invocations}
    atom_uses: set[str] = set()
    for bank in banks:
        members = tuple(atom_by_id.get(atom_id) for atom_id in bank.atom_ids)
        if any(item is None for item in members):
            raise ValueError("atom bank names an unknown atom")
        typed_members = tuple(item for item in members if item is not None)
        if any(item.atom_bank_id != bank.atom_bank_id for item in typed_members):
            raise ValueError("atom bank ownership drifted")
        if bank.physical_stored_scalar_count != sum(
            item.stored_scalar_count for item in typed_members
        ) or bank.state_sha256 != _bank_sha256(typed_members):
            raise ValueError("atom-bank state/accounting drifted")
    if set(atom.atom_bank_id for atom in atoms) != set(bank_by_id):
        raise ValueError("every atom bank must own atoms")
    if len(invocations) != 39 or len(manifest.carrier_manifest.events) != 36:
        raise ValueError("exact heterogeneous Gemma base must contain 39 events")
    if manifest.carrier_manifest.correction_count != 0:
        raise ValueError("exact heterogeneous base cannot contain corrections")
    expected_event_kinds = (
        "lexical_ingress",
        *("attention" if index % 2 else "mlp" for index in range(1, 37)),
        "terminal",
        "lexical_egress",
    )
    if tuple(item.event_kind for item in invocations) != expected_event_kinds:
        raise ValueError("39-event heterogeneous schedule drifted")
    if tuple(item.event_id for item in invocations[1:37]) != tuple(
        event.event_id for event in manifest.carrier_manifest.events
    ):
        raise ValueError("heterogeneous invocations disagree with carrier events")
    for invocation in invocations:
        family = family_by_id.get(invocation.family_id)
        if family is None:
            raise ValueError("invocation names an unknown family")
        expected_kind = (
            "lexical"
            if invocation.event_kind.startswith("lexical")
            else invocation.event_kind
        )
        if family.component_kind != expected_kind:
            raise ValueError("invocation and family component kinds disagree")
        if invocation.event_kind == "terminal":
            continue
        assert invocation.atom_bank_id is not None
        assert invocation.atom_id is not None
        atom = atom_by_id.get(invocation.atom_id)
        if atom is None or atom.atom_bank_id != invocation.atom_bank_id:
            raise ValueError("invocation atom binding is invalid")
        if invocation.atom_bank_id not in family.compatible_atom_bank_ids:
            raise ValueError("family cannot lift from the selected atom bank")
        if invocation.lift_input_rank != atom.output_rank:
            raise ValueError("full-rank lift input disagrees with atom rank")
        atom_uses.add(atom.atom_id)
    if atom_uses != set(atom_by_id):
        raise ValueError("every physical atom must be used by an invocation")
    if len(banks) != 4 or len(atoms) != 37:
        raise ValueError("full exact ontology requires four banks and 37 atoms")
    if manifest.deployed_scalar_count != manifest.source_stored_scalar_count:
        raise ValueError("exact full-rank family lift must preserve scalar count")
    if len(occurrences) != 75:
        raise ValueError("full exact ontology requires 75 edge occurrences")
    if any(item.learned_cross_event_correction for item in occurrences):
        raise ValueError("learned correction occurrences are disabled")
    if any(item.target_invocation_id not in invocation_by_id for item in occurrences):
        raise ValueError("edge occurrence targets an unknown invocation")
    for occurrence in occurrences:
        target = invocation_by_id[occurrence.target_invocation_id]
        if target.invocation_index != occurrence.target_invocation_index:
            raise ValueError("edge target invocation index drifted")
        if occurrence.source_invocation_id is not None:
            source = invocation_by_id.get(occurrence.source_invocation_id)
            if source is None or source.invocation_index != occurrence.source_invocation_index:
                raise ValueError("edge source invocation index drifted")
    carrier_edges = [item.carrier_edge_id for item in occurrences[:73]]
    if carrier_edges != [edge.edge_id for edge in manifest.carrier_manifest.edges]:
        raise ValueError("edge occurrences do not bind the exact carrier topology")
    if any(item.carrier_edge_id is not None for item in occurrences[73:]):
        raise ValueError("terminal/egress occurrences cannot name carrier edges")


def validate_heterogeneous_generator_manifest(
    value: HeterogeneousGeneratorManifest | Mapping[str, object],
) -> HeterogeneousGeneratorManifest:
    if isinstance(value, HeterogeneousGeneratorManifest):
        return HeterogeneousGeneratorManifest.from_dict(value.to_dict())
    return HeterogeneousGeneratorManifest.from_dict(value)


def load_heterogeneous_generator_manifest(
    source: Path | str,
) -> HeterogeneousGeneratorManifest:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_heterogeneous_generator_manifest(value)


def build_family_edge_occurrences(
    *,
    carrier_manifest: CarrierTransportManifest,
    invocations: Sequence[GeneratorInvocationSpec],
) -> tuple[FamilyEdgeOccurrenceSpec, ...]:
    """Bind 75 message occurrences to 39 temporal invocations."""

    carrier = validate_carrier_transport_manifest(carrier_manifest)
    ordered = tuple(invocations)
    if len(ordered) != 39:
        raise ValueError("edge construction requires 39 invocations")
    result: list[FamilyEdgeOccurrenceSpec] = []

    def add(
        *,
        occurrence_id: str,
        message_id: str,
        message_kind: str,
        source: GeneratorInvocationSpec | None,
        target: GeneratorInvocationSpec,
        emitting_family_id: str,
        atom_bank_id: str | None,
        atom_id: str | None,
        carrier_edge_id: str | None,
    ) -> None:
        result.append(
            FamilyEdgeOccurrenceSpec(
                occurrence_index=len(result),
                occurrence_id=occurrence_id,
                message_id=message_id,
                message_kind=message_kind,
                source_invocation_id=None if source is None else source.invocation_id,
                target_invocation_id=target.invocation_id,
                source_invocation_index=(
                    None if source is None else source.invocation_index
                ),
                target_invocation_index=target.invocation_index,
                emitting_family_id=emitting_family_id,
                atom_bank_id=atom_bank_id,
                atom_id=atom_id,
                carrier_edge_id=carrier_edge_id,
            )
        )

    ingress = ordered[0]
    add(
        occurrence_id="occurrence.lexical.ingress",
        message_id="message.lexical.ingress",
        message_kind="lexical_ingress",
        source=None,
        target=ingress,
        emitting_family_id=ingress.family_id,
        atom_bank_id=ingress.atom_bank_id,
        atom_id=ingress.atom_id,
        carrier_edge_id=carrier.edges[0].edge_id,
    )
    carrier_offset = 1
    for mutation_index, target in enumerate(ordered[1:37], start=1):
        source = ordered[mutation_index - 1]
        carry_edge = carrier.edges[carrier_offset]
        primary_edge = carrier.edges[carrier_offset + 1]
        carrier_offset += 2
        add(
            occurrence_id=f"occurrence.carry.{target.invocation_id}",
            message_id=f"message.{target.event_id}.carry",
            message_kind="identity_carry",
            source=source,
            target=target,
            emitting_family_id="carrier.identity",
            atom_bank_id=None,
            atom_id=None,
            carrier_edge_id=carry_edge.edge_id,
        )
        add(
            occurrence_id=f"occurrence.primary.{target.invocation_id}",
            message_id=f"message.{target.event_id}.generator",
            message_kind=(
                "state_conditioned_causal_attention"
                if target.event_kind == "attention"
                else "state_conditioned_token_local_mlp"
            ),
            source=source,
            target=target,
            emitting_family_id=target.family_id,
            atom_bank_id=target.atom_bank_id,
            atom_id=target.atom_id,
            carrier_edge_id=primary_edge.edge_id,
        )
    terminal = ordered[37]
    egress = ordered[38]
    add(
        occurrence_id="occurrence.terminal.norm_softcap",
        message_id="message.terminal.rmsnorm",
        message_kind="terminal_normalization",
        source=ordered[36],
        target=terminal,
        emitting_family_id=terminal.family_id,
        atom_bank_id=None,
        atom_id=None,
        carrier_edge_id=None,
    )
    add(
        occurrence_id="occurrence.lexical.egress",
        message_id="message.lexical.egress",
        message_kind="lexical_egress",
        source=terminal,
        target=egress,
        emitting_family_id=egress.family_id,
        atom_bank_id=egress.atom_bank_id,
        atom_id=egress.atom_id,
        carrier_edge_id=None,
    )
    return tuple(result)


def build_exact_heterogeneous_generator_manifest(
    *,
    arm_id: str,
    lift_scope: str,
    hidden_width: int,
    source_stored_scalar_count: int,
    terminal_stored_scalar_count: int,
    terminal_state_sha256: str,
    carrier_manifest: CarrierTransportManifest,
    families: Sequence[GeneratorFamilySpec],
    atom_banks: Sequence[GeneratorAtomBankSpec],
    atoms: Sequence[GeneratorAtomSpec],
    invocations: Sequence[GeneratorInvocationSpec],
) -> HeterogeneousGeneratorManifest:
    invocation_tuple = tuple(invocations)
    return HeterogeneousGeneratorManifest(
        arm_id=arm_id,
        lift_scope=lift_scope,
        hidden_width=hidden_width,
        source_stored_scalar_count=source_stored_scalar_count,
        terminal_stored_scalar_count=terminal_stored_scalar_count,
        terminal_state_sha256=terminal_state_sha256,
        carrier_manifest=validate_carrier_transport_manifest(carrier_manifest),
        families=tuple(families),
        atom_banks=tuple(atom_banks),
        atoms=tuple(atoms),
        invocations=invocation_tuple,
        edge_occurrences=build_family_edge_occurrences(
            carrier_manifest=carrier_manifest,
            invocations=invocation_tuple,
        ),
    )


@dataclass(frozen=True, slots=True)
class AtomEmission:
    value: Tensor
    attention_routing_weights: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor):
            raise TypeError("atom emission value must be a Tensor")
        if self.attention_routing_weights is not None and not isinstance(
            self.attention_routing_weights, Tensor
        ):
            raise TypeError("atom routing weights must be a Tensor or None")


class GeneratorAtomRuntime(nn.Module, ABC):
    """One physical, source-independent exact generator atom."""

    component_kind: str

    def __init__(self, *, atom_id: str, output_rank: int) -> None:
        super().__init__()
        self.atom_id = _require_id(atom_id, label="runtime atom_id")
        self.output_rank = _require_int(
            output_rank,
            label="runtime output_rank",
            minimum=1,
        )

    @property
    def stored_scalar_count(self) -> int:
        return module_physical_scalar_count(self)

    @property
    def state_sha256(self) -> str:
        return module_state_sha256(self)

    @property
    def source_projection_call_count(self) -> int:
        return 0

    @property
    def source_fallback_call_count(self) -> int:
        return 0


class LexicalAtomRuntime(GeneratorAtomRuntime):
    component_kind = "lexical"

    @abstractmethod
    def ingress(self, token_input: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def egress(self, hidden_state: Tensor) -> Tensor:
        raise NotImplementedError


class AttentionAtomRuntime(GeneratorAtomRuntime):
    component_kind = "attention"

    @abstractmethod
    def emit(self, carrier: Tensor, sequence: object) -> AtomEmission:
        raise NotImplementedError


class MLPAtomRuntime(GeneratorAtomRuntime):
    component_kind = "mlp"

    @abstractmethod
    def emit(self, carrier: Tensor, sequence: object) -> AtomEmission:
        raise NotImplementedError


class TerminalRuntime(nn.Module, ABC):
    @abstractmethod
    def normalize(self, carrier: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def stored_scalar_count(self) -> int:
        return module_physical_scalar_count(self)

    @property
    def state_sha256(self) -> str:
        return module_state_sha256(self)


class GeneratorAtomBankRuntime(nn.Module):
    """Register every physical atom once and resolve it by canonical ID."""

    def __init__(
        self,
        *,
        spec: GeneratorAtomBankSpec,
        atoms: Mapping[str, GeneratorAtomRuntime],
    ) -> None:
        super().__init__()
        if not isinstance(spec, GeneratorAtomBankSpec):
            raise TypeError("atom-bank runtime requires a bank spec")
        if not isinstance(atoms, Mapping) or set(atoms) != set(spec.atom_ids):
            raise ValueError("atom runtime inventory disagrees with bank spec")
        ordered = tuple(atoms[atom_id] for atom_id in spec.atom_ids)
        if any(not isinstance(atom, GeneratorAtomRuntime) for atom in ordered):
            raise TypeError("atom banks require GeneratorAtomRuntime values")
        if len({id(atom) for atom in ordered}) != len(ordered):
            raise ValueError("one runtime object cannot represent two atom IDs")
        pointers: set[int] = set()
        for atom in ordered:
            current = {
                value.untyped_storage().data_ptr()
                for value in atom.state_dict(keep_vars=True).values()
                if value.numel()
            }
            if pointers & current:
                raise ValueError(
                    "two atom IDs cannot alias one physical tensor storage"
                )
            pointers.update(current)
        self.spec = spec
        self._atom_ids = spec.atom_ids
        self.atoms = nn.ModuleList(ordered)
        self.eval().requires_grad_(False)
        self.assert_matches_spec()

    def atom(self, atom_id: str) -> GeneratorAtomRuntime:
        try:
            index = self._atom_ids.index(atom_id)
        except ValueError as error:
            raise KeyError(f"unknown atom ID {atom_id}") from error
        atom = self.atoms[index]
        assert isinstance(atom, GeneratorAtomRuntime)
        return atom

    @property
    def state_sha256(self) -> str:
        specs = tuple(
            GeneratorAtomSpec(
                atom_index=index,
                atom_id=atom_id,
                atom_bank_id=self.spec.atom_bank_id,
                component_kind=self.atom(atom_id).component_kind,
                output_rank=self.atom(atom_id).output_rank,
                stored_scalar_count=self.atom(atom_id).stored_scalar_count,
                state_sha256=self.atom(atom_id).state_sha256,
            )
            for index, atom_id in enumerate(self._atom_ids)
        )
        return _bank_sha256(specs)

    def assert_matches_spec(self) -> None:
        if (
            sum(self.atom(atom_id).stored_scalar_count for atom_id in self._atom_ids)
            != self.spec.physical_stored_scalar_count
            or self.state_sha256 != self.spec.state_sha256
        ):
            raise RuntimeError("physical atom-bank state drifted from manifest")


@dataclass(frozen=True, slots=True)
class InvocationExecutionReceipt:
    invocation_index: int
    invocation_id: str
    event_id: str
    event_kind: str
    family_id: str
    atom_bank_id: str | None
    atom_id: str | None
    atom_output_sha256: str
    lifted_output_sha256: str
    routing_sha256: str | None
    exact_full_rank_lift_literal_parity: bool

    def __post_init__(self) -> None:
        _require_int(self.invocation_index, label="receipt invocation_index")
        for label, value in (
            ("invocation_id", self.invocation_id),
            ("event_id", self.event_id),
            ("event_kind", self.event_kind),
            ("family_id", self.family_id),
        ):
            _require_id(value, label=label)
        if (self.atom_bank_id is None) != (self.atom_id is None):
            raise ValueError("receipt atom bank/atom optionality drifted")
        if self.atom_bank_id is not None:
            _require_id(self.atom_bank_id, label="receipt atom_bank_id")
            _require_id(self.atom_id, label="receipt atom_id")
        _require_sha256(self.atom_output_sha256, label="atom_output_sha256")
        _require_sha256(self.lifted_output_sha256, label="lifted_output_sha256")
        if self.routing_sha256 is not None:
            _require_sha256(self.routing_sha256, label="routing_sha256")
        if (
            self.exact_full_rank_lift_literal_parity is not True
            or self.atom_output_sha256 != self.lifted_output_sha256
        ):
            raise ValueError("exact full-rank atom lift lost literal parity")

    def to_dict(self) -> dict[str, object]:
        return {
            "invocation_index": self.invocation_index,
            "invocation_id": self.invocation_id,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "family_id": self.family_id,
            "atom_bank_id": self.atom_bank_id,
            "atom_id": self.atom_id,
            "atom_output_sha256": self.atom_output_sha256,
            "lifted_output_sha256": self.lifted_output_sha256,
            "routing_sha256": self.routing_sha256,
            "exact_full_rank_lift_literal_parity": (
                self.exact_full_rank_lift_literal_parity
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> InvocationExecutionReceipt:
        expected = {
            "invocation_index",
            "invocation_id",
            "event_id",
            "event_kind",
            "family_id",
            "atom_bank_id",
            "atom_id",
            "atom_output_sha256",
            "lifted_output_sha256",
            "routing_sha256",
            "exact_full_rank_lift_literal_parity",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("invocation receipt fields are not exact")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class HeterogeneousGeneratorExecutionReceipt:
    manifest_sha256: str
    execution_nonce: str
    carrier_execution_receipt: CarrierTransportExecutionReceipt
    invocation_receipts: tuple[InvocationExecutionReceipt, ...]
    atom_bank_state_before: tuple[tuple[str, str], ...]
    atom_bank_state_after: tuple[tuple[str, str], ...]
    terminal_state_sha256: str
    final_carrier_sha256: str
    normalized_output_sha256: str
    logits_sha256: str
    physical_stored_scalar_count: int
    source_projection_call_count: int = 0
    source_fallback_call_count: int = 0
    learned_cross_event_correction_count: int = 0
    compression_attempted: bool = False
    compression_eligible: bool = False
    schema: str = HETEROGENEOUS_GENERATOR_RECEIPT_SCHEMA
    format_version: int = HETEROGENEOUS_GENERATOR_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != HETEROGENEOUS_GENERATOR_RECEIPT_SCHEMA:
            raise ValueError("heterogeneous receipt schema is unsupported")
        if self.format_version != HETEROGENEOUS_GENERATOR_FORMAT_VERSION:
            raise ValueError("heterogeneous receipt version is unsupported")
        _require_sha256(self.manifest_sha256, label="receipt manifest_sha256")
        _require_id(self.execution_nonce, label="execution_nonce")
        if not isinstance(
            self.carrier_execution_receipt,
            CarrierTransportExecutionReceipt,
        ):
            raise TypeError("carrier execution receipt has the wrong type")
        if type(self.invocation_receipts) is not tuple:
            raise TypeError("invocation receipts must be a tuple")
        if len(self.invocation_receipts) != 39:
            raise ValueError("execution receipt must cover 39 invocations")
        for catalog in (self.atom_bank_state_before, self.atom_bank_state_after):
            if type(catalog) is not tuple or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                for item in catalog
            ):
                raise TypeError("atom-bank state catalog is invalid")
            for bank_id, digest in catalog:
                _require_id(bank_id, label="receipt atom_bank_id")
                _require_sha256(digest, label="receipt atom-bank state")
        if self.atom_bank_state_before != self.atom_bank_state_after:
            raise ValueError("atom-bank state changed during execution")
        for label, digest in (
            ("terminal_state_sha256", self.terminal_state_sha256),
            ("final_carrier_sha256", self.final_carrier_sha256),
            ("normalized_output_sha256", self.normalized_output_sha256),
            ("logits_sha256", self.logits_sha256),
        ):
            _require_sha256(digest, label=label)
        _require_int(
            self.physical_stored_scalar_count,
            label="physical_stored_scalar_count",
            minimum=1,
        )
        if (
            self.source_projection_call_count != 0
            or self.source_fallback_call_count != 0
            or self.learned_cross_event_correction_count != 0
            or self.compression_attempted is not False
            or self.compression_eligible is not False
        ):
            raise ValueError("exact heterogeneous receipt crossed its claim firewall")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "manifest_sha256": self.manifest_sha256,
            "execution_nonce": self.execution_nonce,
            "carrier_execution_receipt": self.carrier_execution_receipt.to_dict(),
            "invocation_receipts": [item.to_dict() for item in self.invocation_receipts],
            "atom_bank_state_before": dict(self.atom_bank_state_before),
            "atom_bank_state_after": dict(self.atom_bank_state_after),
            "terminal_state_sha256": self.terminal_state_sha256,
            "final_carrier_sha256": self.final_carrier_sha256,
            "normalized_output_sha256": self.normalized_output_sha256,
            "logits_sha256": self.logits_sha256,
            "physical_stored_scalar_count": self.physical_stored_scalar_count,
            "source_projection_call_count": self.source_projection_call_count,
            "source_fallback_call_count": self.source_fallback_call_count,
            "learned_cross_event_correction_count": (
                self.learned_cross_event_correction_count
            ),
            "compression_attempted": self.compression_attempted,
            "compression_eligible": self.compression_eligible,
        }

    @property
    def fingerprint_sha256(self) -> str:
        return _fingerprint(_RECEIPT_DOMAIN, self.payload())

    def to_dict(self) -> dict[str, object]:
        result = self.payload()
        result["receipt_sha256"] = self.fingerprint_sha256
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> HeterogeneousGeneratorExecutionReceipt:
        expected = {
            "schema",
            "format_version",
            "manifest_sha256",
            "execution_nonce",
            "carrier_execution_receipt",
            "invocation_receipts",
            "atom_bank_state_before",
            "atom_bank_state_after",
            "terminal_state_sha256",
            "final_carrier_sha256",
            "normalized_output_sha256",
            "logits_sha256",
            "physical_stored_scalar_count",
            "source_projection_call_count",
            "source_fallback_call_count",
            "learned_cross_event_correction_count",
            "compression_attempted",
            "compression_eligible",
            "receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("heterogeneous execution receipt fields are not exact")
        invocations = value["invocation_receipts"]
        if not isinstance(invocations, Sequence) or isinstance(
            invocations, (str, bytes)
        ):
            raise TypeError("serialized invocation receipts must be a sequence")
        before = value["atom_bank_state_before"]
        after = value["atom_bank_state_after"]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise TypeError("serialized atom-bank states must be mappings")
        result = cls(
            manifest_sha256=value["manifest_sha256"],
            execution_nonce=value["execution_nonce"],
            carrier_execution_receipt=CarrierTransportExecutionReceipt.from_dict(
                value["carrier_execution_receipt"]
            ),
            invocation_receipts=tuple(
                InvocationExecutionReceipt.from_dict(item) for item in invocations
            ),
            atom_bank_state_before=tuple(sorted(before.items())),
            atom_bank_state_after=tuple(sorted(after.items())),
            terminal_state_sha256=value["terminal_state_sha256"],
            final_carrier_sha256=value["final_carrier_sha256"],
            normalized_output_sha256=value["normalized_output_sha256"],
            logits_sha256=value["logits_sha256"],
            physical_stored_scalar_count=value["physical_stored_scalar_count"],
            source_projection_call_count=value["source_projection_call_count"],
            source_fallback_call_count=value["source_fallback_call_count"],
            learned_cross_event_correction_count=value[
                "learned_cross_event_correction_count"
            ],
            compression_attempted=value["compression_attempted"],
            compression_eligible=value["compression_eligible"],
            schema=value["schema"],
            format_version=value["format_version"],
        )
        _require_sha256(value["receipt_sha256"], label="receipt_sha256")
        if result.to_dict() != dict(value):
            raise ValueError("heterogeneous execution receipt authentication failed")
        return result


@dataclass(frozen=True, slots=True)
class HeterogeneousGeneratorExecution:
    output: Tensor
    normalized_output: Tensor
    logits: Tensor
    carrier_execution: CarrierTransportExecution
    receipt: HeterogeneousGeneratorExecutionReceipt

    def __post_init__(self) -> None:
        for name, value in (
            ("output", self.output),
            ("normalized_output", self.normalized_output),
            ("logits", self.logits),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a Tensor")
        if not isinstance(self.carrier_execution, CarrierTransportExecution):
            raise TypeError("carrier_execution has the wrong type")
        if not isinstance(self.receipt, HeterogeneousGeneratorExecutionReceipt):
            raise TypeError("heterogeneous receipt has the wrong type")
        if (
            tensor_sha256(self.output) != self.receipt.final_carrier_sha256
            or tensor_sha256(self.normalized_output)
            != self.receipt.normalized_output_sha256
            or tensor_sha256(self.logits) != self.receipt.logits_sha256
        ):
            raise ValueError("execution tensors disagree with receipt")


class HeterogeneousGeneratorExecutor(nn.Module):
    """Own the live lexical -> 36-carrier -> terminal -> egress path."""

    def __init__(
        self,
        *,
        manifest: HeterogeneousGeneratorManifest,
        atom_banks: Mapping[str, GeneratorAtomBankRuntime],
        terminal_runtime: TerminalRuntime,
    ) -> None:
        super().__init__()
        self.manifest = validate_heterogeneous_generator_manifest(manifest)
        if not isinstance(atom_banks, Mapping) or set(atom_banks) != {
            item.atom_bank_id for item in self.manifest.atom_banks
        }:
            raise ValueError("runtime atom-bank inventory disagrees with manifest")
        ordered = tuple(
            atom_banks[item.atom_bank_id] for item in self.manifest.atom_banks
        )
        if any(not isinstance(item, GeneratorAtomBankRuntime) for item in ordered):
            raise TypeError("executor requires GeneratorAtomBankRuntime values")
        if not isinstance(terminal_runtime, TerminalRuntime):
            raise TypeError("executor requires a typed terminal runtime")
        seen_storages: set[tuple[str, int]] = set()
        for bank in ordered:
            current = set(_module_persistent_storage_keys(bank))
            if seen_storages & current:
                raise ValueError(
                    "two atom banks cannot alias one physical tensor storage"
                )
            seen_storages.update(current)
        terminal_storages = set(_module_persistent_storage_keys(terminal_runtime))
        if seen_storages & terminal_storages:
            raise ValueError(
                "terminal storage cannot alias a physical generator atom"
            )
        self._bank_ids = tuple(item.atom_bank_id for item in self.manifest.atom_banks)
        self.atom_banks = nn.ModuleList(ordered)
        self.terminal_runtime = terminal_runtime
        if (
            terminal_runtime.stored_scalar_count
            != self.manifest.terminal_stored_scalar_count
            or terminal_runtime.state_sha256
            != self.manifest.terminal_state_sha256
        ):
            raise ValueError("terminal scalar accounting disagrees with manifest")
        for bank in ordered:
            bank.assert_matches_spec()
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> HeterogeneousGeneratorExecutor:
        if type(mode) is not bool:
            raise TypeError("mode must be bool")
        if mode:
            raise RuntimeError("exact heterogeneous executor is inference-only")
        return super().train(False)

    def bank(self, bank_id: str) -> GeneratorAtomBankRuntime:
        try:
            index = self._bank_ids.index(bank_id)
        except ValueError as error:
            raise KeyError(f"unknown atom-bank ID {bank_id}") from error
        bank = self.atom_banks[index]
        assert isinstance(bank, GeneratorAtomBankRuntime)
        return bank

    def atom_for(self, invocation: GeneratorInvocationSpec) -> GeneratorAtomRuntime:
        if invocation.atom_bank_id is None or invocation.atom_id is None:
            raise ValueError("terminal invocation has no physical atom")
        return self.bank(invocation.atom_bank_id).atom(invocation.atom_id)

    def _bank_states(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted((bank_id, self.bank(bank_id).state_sha256) for bank_id in self._bank_ids)
        )

    @staticmethod
    def _receipt(
        invocation: GeneratorInvocationSpec,
        output: Tensor,
        routing: Tensor | None,
    ) -> InvocationExecutionReceipt:
        digest = tensor_sha256(output)
        return InvocationExecutionReceipt(
            invocation_index=invocation.invocation_index,
            invocation_id=invocation.invocation_id,
            event_id=invocation.event_id,
            event_kind=invocation.event_kind,
            family_id=invocation.family_id,
            atom_bank_id=invocation.atom_bank_id,
            atom_id=invocation.atom_id,
            atom_output_sha256=digest,
            lifted_output_sha256=digest,
            routing_sha256=None if routing is None else tensor_sha256(routing),
            exact_full_rank_lift_literal_parity=True,
        )

    def execute(
        self,
        token_input: Tensor,
        sequence: object,
        *,
        execution_nonce: str,
    ) -> HeterogeneousGeneratorExecution:
        if not isinstance(token_input, Tensor) or token_input.ndim != 2:
            raise ValueError("token_input must be rank-two")
        _require_id(execution_nonce, label="execution_nonce")
        before = self._bank_states()
        invocations = self.manifest.invocations
        by_event = {item.event_id: item for item in invocations[1:37]}
        invocation_receipts: dict[int, InvocationExecutionReceipt] = {}
        producers: dict[str, Any] = {}
        carrier = self.manifest.carrier_manifest
        lexical_invocation = invocations[0]
        lexical_atom = self.atom_for(lexical_invocation)
        if not isinstance(lexical_atom, LexicalAtomRuntime):
            raise TypeError("lexical invocation resolved to the wrong atom type")

        def lexical_producer(context: TransportContext) -> TransportMessage:
            if context.edge.edge_id != carrier.edges[0].edge_id:
                raise RuntimeError("lexical carrier edge drifted")
            value = lexical_atom.ingress(token_input)
            if value.shape != (*token_input.shape, self.manifest.hidden_width):
                raise ValueError("lexical atom returned the wrong hidden shape")
            invocation_receipts[0] = self._receipt(
                lexical_invocation,
                value,
                None,
            )
            return TransportMessage(edge_id=context.edge.edge_id, value=value)

        producers[carrier.edges[0].edge_id] = lexical_producer
        for edge in carrier.edges:
            if edge.message_kind not in {"attention_transport", "mlp_mutation"}:
                continue
            assert edge.target_event_id is not None
            invocation = by_event[edge.target_event_id]
            atom = self.atom_for(invocation)

            def producer(
                context: TransportContext,
                *,
                invocation: GeneratorInvocationSpec = invocation,
                atom: GeneratorAtomRuntime = atom,
            ) -> TransportMessage:
                if context.carrier_input is None:
                    raise RuntimeError("mutation atom did not receive a carrier")
                if invocation.event_kind == "attention":
                    if not isinstance(atom, AttentionAtomRuntime):
                        raise TypeError("attention invocation resolved to wrong atom type")
                    emission = atom.emit(context.carrier_input, sequence)
                    if emission.attention_routing_weights is None:
                        raise ValueError("attention atom omitted routing weights")
                else:
                    if not isinstance(atom, MLPAtomRuntime):
                        raise TypeError("MLP invocation resolved to wrong atom type")
                    emission = atom.emit(context.carrier_input, sequence)
                    if emission.attention_routing_weights is not None:
                        raise ValueError("MLP atom cannot emit attention routing")
                if emission.value.shape != context.carrier_input.shape:
                    raise ValueError("generator atom must emit a full-width delta")
                invocation_receipts[invocation.invocation_index] = self._receipt(
                    invocation,
                    emission.value,
                    emission.attention_routing_weights,
                )
                return TransportMessage(
                    edge_id=context.edge.edge_id,
                    value=emission.value,
                    attention_routing_weights=emission.attention_routing_weights,
                )

            producers[edge.edge_id] = producer
        carrier_execution = execute_causal_carrier_transport_graph(
            carrier,
            token_input=token_input,
            message_producers=producers,
            execution_nonce=execution_nonce,
        )
        validate_carrier_transport_execution_receipt(
            carrier_execution.receipt,
            manifest=carrier,
        )
        terminal_invocation = invocations[37]
        normalized = self.terminal_runtime.normalize(carrier_execution.final_carrier)
        if normalized.shape != carrier_execution.final_carrier.shape:
            raise ValueError("terminal runtime must preserve hidden shape")
        invocation_receipts[37] = self._receipt(
            terminal_invocation,
            normalized,
            None,
        )
        egress_invocation = invocations[38]
        egress_atom = self.atom_for(egress_invocation)
        if egress_atom is not lexical_atom:
            raise RuntimeError("tied lexical ingress/egress did not share one atom")
        logits = lexical_atom.egress(normalized)
        if logits.ndim != 3 or logits.shape[:2] != token_input.shape:
            raise ValueError("lexical egress returned invalid logits")
        invocation_receipts[38] = self._receipt(egress_invocation, logits, None)
        if set(invocation_receipts) != set(range(39)):
            raise RuntimeError("live path did not execute every invocation once")
        after = self._bank_states()
        if before != after:
            raise RuntimeError("atom-bank state changed during execution")
        receipt = HeterogeneousGeneratorExecutionReceipt(
            manifest_sha256=self.manifest.fingerprint_sha256,
            execution_nonce=execution_nonce,
            carrier_execution_receipt=carrier_execution.receipt,
            invocation_receipts=tuple(
                invocation_receipts[index] for index in range(39)
            ),
            atom_bank_state_before=before,
            atom_bank_state_after=after,
            terminal_state_sha256=self.terminal_runtime.state_sha256,
            final_carrier_sha256=tensor_sha256(carrier_execution.final_carrier),
            normalized_output_sha256=tensor_sha256(normalized),
            logits_sha256=tensor_sha256(logits),
            physical_stored_scalar_count=self.manifest.deployed_scalar_count,
        )
        validate_heterogeneous_generator_execution_receipt(
            receipt,
            manifest=self.manifest,
        )
        return HeterogeneousGeneratorExecution(
            output=carrier_execution.final_carrier,
            normalized_output=normalized,
            logits=logits,
            carrier_execution=carrier_execution,
            receipt=receipt,
        )

    forward_components = execute


def assemble_heterogeneous_generator_executor(
    *,
    manifest: HeterogeneousGeneratorManifest,
    atom_banks: Mapping[str, GeneratorAtomBankRuntime],
    terminal_runtime: TerminalRuntime,
) -> HeterogeneousGeneratorExecutor:
    return HeterogeneousGeneratorExecutor(
        manifest=manifest,
        atom_banks=atom_banks,
        terminal_runtime=terminal_runtime,
    )


def validate_heterogeneous_generator_execution_receipt(
    value: HeterogeneousGeneratorExecutionReceipt | Mapping[str, object],
    *,
    manifest: HeterogeneousGeneratorManifest,
) -> HeterogeneousGeneratorExecutionReceipt:
    manifest = validate_heterogeneous_generator_manifest(manifest)
    receipt = (
        value
        if isinstance(value, HeterogeneousGeneratorExecutionReceipt)
        else HeterogeneousGeneratorExecutionReceipt.from_dict(value)
    )
    if receipt.manifest_sha256 != manifest.fingerprint_sha256:
        raise ValueError("execution receipt belongs to another manifest")
    validate_carrier_transport_execution_receipt(
        receipt.carrier_execution_receipt,
        manifest=manifest.carrier_manifest,
    )
    if tuple(item.invocation_id for item in receipt.invocation_receipts) != tuple(
        item.invocation_id for item in manifest.invocations
    ):
        raise ValueError("execution invocation order disagrees with manifest")
    for spec, item in zip(
        manifest.invocations,
        receipt.invocation_receipts,
        strict=True,
    ):
        if (
            item.invocation_index != spec.invocation_index
            or item.event_id != spec.event_id
            or item.event_kind != spec.event_kind
            or item.family_id != spec.family_id
            or item.atom_bank_id != spec.atom_bank_id
            or item.atom_id != spec.atom_id
        ):
            raise ValueError("execution invocation metadata drifted")
    expected_states = tuple(
        sorted((bank.atom_bank_id, bank.state_sha256) for bank in manifest.atom_banks)
    )
    if (
        receipt.atom_bank_state_before != expected_states
        or receipt.terminal_state_sha256 != manifest.terminal_state_sha256
        or receipt.physical_stored_scalar_count != manifest.deployed_scalar_count
    ):
        raise ValueError("execution physical-state accounting drifted")
    return receipt


def load_heterogeneous_generator_execution_receipt(
    source: Path | str,
    *,
    manifest: HeterogeneousGeneratorManifest,
) -> HeterogeneousGeneratorExecutionReceipt:
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    return validate_heterogeneous_generator_execution_receipt(
        value,
        manifest=manifest,
    )


__all__ = [
    "AttentionAtomRuntime",
    "AttentionRuntimeContract",
    "AtomEmission",
    "FamilyEdgeOccurrenceSpec",
    "GeneratorAtomBankRuntime",
    "GeneratorAtomBankSpec",
    "GeneratorAtomRuntime",
    "GeneratorAtomSpec",
    "GeneratorFamilySpec",
    "GeneratorInvocationSpec",
    "HETEROGENEOUS_GENERATOR_FORMAT_VERSION",
    "HETEROGENEOUS_GENERATOR_MANIFEST_SCHEMA",
    "HETEROGENEOUS_GENERATOR_RECEIPT_SCHEMA",
    "HeterogeneousGeneratorExecution",
    "HeterogeneousGeneratorExecutionReceipt",
    "HeterogeneousGeneratorExecutor",
    "HeterogeneousGeneratorManifest",
    "InvocationExecutionReceipt",
    "LIFT_SCOPE_COMPATIBLE_ATTENTION",
    "LIFT_SCOPE_FULL_HETEROGENEOUS",
    "LIFT_SCOPE_MLP_FAMILY",
    "LIFT_SCOPES",
    "LexicalAtomRuntime",
    "LexicalRuntimeContract",
    "MLPAtomRuntime",
    "MLPRuntimeContract",
    "TerminalRuntime",
    "TerminalRuntimeContract",
    "assemble_heterogeneous_generator_executor",
    "build_exact_heterogeneous_generator_manifest",
    "build_family_edge_occurrences",
    "build_generator_atom_bank_spec",
    "build_generator_atom_spec",
    "load_heterogeneous_generator_execution_receipt",
    "load_heterogeneous_generator_manifest",
    "module_physical_scalar_count",
    "module_state_sha256",
    "runtime_contract_from_dict",
    "tensor_sha256",
    "validate_heterogeneous_generator_execution_receipt",
    "validate_heterogeneous_generator_manifest",
]
