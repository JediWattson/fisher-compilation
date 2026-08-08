"""Model-agnostic causal carrier-transport graph and live executor.

The graph treats residual carry, attention, MLP mutations, lexical ingress,
and optional earlier-event corrections as one kind of object: a message into
a versioned carrier boundary.  A mutation boundary commits exactly once::

    carrier[v + 1] = identity(carrier[v]) + primary_message + corrections

The executor owns that addition.  Generator callbacks can produce messages,
but cannot replace the carrier or commit a boundary themselves.  This keeps
the residual wire explicit when generator families are later shared, merged,
or removed.

This module does not load a model and makes no compression claim.  It uses
PyTorch tensors so exact model wrappers can retain their native dtype, device,
and autograd graph while the topology and receipts remain model-independent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import TypeAlias

import torch
from torch import Tensor


CARRIER_TRANSPORT_MANIFEST_SCHEMA = (
    "fisher_graph.causal_carrier_transport.manifest.v1"
)
CARRIER_TRANSPORT_RECEIPT_SCHEMA = (
    "fisher_graph.causal_carrier_transport.execution_receipt.v1"
)
CARRIER_TRANSPORT_FORMAT_VERSION = 1

_MANIFEST_HASH_DOMAIN = b"fisher_graph:carrier-transport-manifest:v1\0"
_RECEIPT_HASH_DOMAIN = b"fisher_graph:carrier-transport-receipt:v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph:carrier-transport-tensor:v1\0"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")

_EVENT_KINDS = frozenset({"attention", "mlp"})
_MESSAGE_KINDS = frozenset(
    {
        "lexical_ingress",
        "identity_carry",
        "attention_transport",
        "mlp_mutation",
        "forward_correction",
    }
)
_ROUTING_BY_MESSAGE_KIND = {
    "lexical_ingress": "vocabulary_lookup",
    "identity_carry": "same_token_fixed_one",
    "attention_transport": "state_conditioned_causal_token_routing",
    "mlp_mutation": "same_token",
    "forward_correction": "strictly_forward_same_token",
}
_TRANSFORM_BY_MESSAGE_KIND = {
    "lexical_ingress": "embedding_lookup",
    "identity_carry": "identity",
    "attention_transport": "attention_value_output_projection",
    "mlp_mutation": "token_local_nonlinear",
    "forward_correction": "learned_correction",
}


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


def _require_exact_int(value: object, *, label: str, minimum: int = 0) -> int:
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


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("only torch tensors can be authenticated")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    # NumPy cannot materialize every torch dtype (notably bfloat16).  A uint8
    # storage view preserves the exact native bytes for all dense dtypes.
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_metadata(value: Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device_type": value.device.type,
        "sha256": _tensor_sha256(value),
    }


def _freeze_tensor_metadata(
    value: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    if set(value) != {"shape", "dtype", "device_type", "sha256"}:
        raise ValueError(f"{label} fields are not exact")
    shape = value["shape"]
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError(f"{label} shape must be a sequence")
    frozen_shape = tuple(shape)
    if any(
        type(dimension) is not int or dimension < 0
        for dimension in frozen_shape
    ):
        raise ValueError(f"{label} shape dimensions must be nonnegative integers")
    dtype = value["dtype"]
    if not isinstance(dtype, str) or not dtype.startswith("torch."):
        raise ValueError(f"{label} dtype must be a canonical torch dtype")
    device_type = value["device_type"]
    _require_id(device_type, label=f"{label} device_type")
    digest = _require_sha256(value["sha256"], label=f"{label} sha256")
    return MappingProxyType(
        {
            "shape": frozen_shape,
            "dtype": dtype,
            "device_type": device_type,
            "sha256": digest,
        }
    )


def _tensor_metadata_payload(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "shape": list(value["shape"]),
        "dtype": value["dtype"],
        "device_type": value["device_type"],
        "sha256": value["sha256"],
    }


def _clone(value: Tensor) -> Tensor:
    # Clone rather than detach: a compiled model may need autograd through the
    # live graph.  Receipt hashing always detaches its private CPU view.
    return value.clone()


@dataclass(frozen=True, slots=True)
class CarrierEventSpec:
    """One causal generator invocation and its carrier boundary.

    ``generator_family_id`` identifies shared parameters or behavior.
    ``generator_invocation_id`` identifies this temporal call.  Families may
    repeat across events; invocation identifiers may not.
    """

    event_index: int
    event_id: str
    event_kind: str
    generator_invocation_id: str
    generator_family_id: str
    input_carrier_version: int
    output_carrier_version: int

    def __post_init__(self) -> None:
        _require_exact_int(self.event_index, label="event_index")
        _require_id(self.event_id, label="event_id")
        if self.event_kind not in _EVENT_KINDS:
            raise ValueError("event_kind must be 'attention' or 'mlp'")
        _require_id(
            self.generator_invocation_id,
            label="generator_invocation_id",
        )
        _require_id(self.generator_family_id, label="generator_family_id")
        if self.generator_invocation_id == self.generator_family_id:
            raise ValueError("generator family identity must differ from invocation")
        _require_exact_int(
            self.input_carrier_version,
            label="input_carrier_version",
        )
        _require_exact_int(
            self.output_carrier_version,
            label="output_carrier_version",
            minimum=1,
        )
        if self.output_carrier_version != self.input_carrier_version + 1:
            raise ValueError("one event must advance the carrier by one version")

    @property
    def primary_message_kind(self) -> str:
        return {
            "attention": "attention_transport",
            "mlp": "mlp_mutation",
        }[self.event_kind]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "generator_invocation_id": self.generator_invocation_id,
            "generator_family_id": self.generator_family_id,
            "input_carrier_version": self.input_carrier_version,
            "output_carrier_version": self.output_carrier_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CarrierEventSpec:
        expected = {
            "event_index",
            "event_id",
            "event_kind",
            "generator_invocation_id",
            "generator_family_id",
            "input_carrier_version",
            "output_carrier_version",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("carrier-event fields do not match format version 1")
        return cls(
            event_index=value["event_index"],
            event_id=value["event_id"],
            event_kind=value["event_kind"],
            generator_invocation_id=value["generator_invocation_id"],
            generator_family_id=value["generator_family_id"],
            input_carrier_version=value["input_carrier_version"],
            output_carrier_version=value["output_carrier_version"],
        )


@dataclass(frozen=True, slots=True)
class ForwardCorrectionSpec:
    """Declaration for an optional earlier-event correction message."""

    correction_id: str
    generator_invocation_id: str
    generator_family_id: str
    source_event_id: str
    target_event_id: str

    def __post_init__(self) -> None:
        _require_id(self.correction_id, label="correction_id")
        _require_id(
            self.generator_invocation_id,
            label="correction generator_invocation_id",
        )
        _require_id(
            self.generator_family_id,
            label="correction generator_family_id",
        )
        if self.generator_invocation_id == self.generator_family_id:
            raise ValueError("generator family identity must differ from invocation")
        _require_id(self.source_event_id, label="source_event_id")
        _require_id(self.target_event_id, label="target_event_id")
        if self.source_event_id == self.target_event_id:
            raise ValueError("a correction source and target must differ")


@dataclass(frozen=True, slots=True)
class TransportEdgeSpec:
    """One typed message edge into a versioned carrier boundary."""

    edge_index: int
    edge_id: str
    message_kind: str
    generator_invocation_id: str
    generator_family_id: str
    source_event_id: str | None
    target_event_id: str | None
    source_carrier_version: int | None
    target_carrier_version: int
    routing_kind: str
    transform_kind: str
    fixed_weight: float | None
    stored_scalar_count: int

    def __post_init__(self) -> None:
        _require_exact_int(self.edge_index, label="edge_index")
        _require_id(self.edge_id, label="edge_id")
        if self.message_kind not in _MESSAGE_KINDS:
            raise ValueError("message_kind is invalid")
        _require_id(
            self.generator_invocation_id,
            label="edge generator_invocation_id",
        )
        _require_id(
            self.generator_family_id,
            label="edge generator_family_id",
        )
        if self.generator_invocation_id == self.generator_family_id:
            raise ValueError("generator family identity must differ from invocation")
        if self.source_event_id is not None:
            _require_id(self.source_event_id, label="source_event_id")
        if self.target_event_id is not None:
            _require_id(self.target_event_id, label="target_event_id")
        if self.source_carrier_version is not None:
            _require_exact_int(
                self.source_carrier_version,
                label="source_carrier_version",
            )
        _require_exact_int(
            self.target_carrier_version,
            label="target_carrier_version",
        )
        if self.routing_kind != _ROUTING_BY_MESSAGE_KIND[self.message_kind]:
            raise ValueError("routing_kind disagrees with message_kind")
        if self.transform_kind != _TRANSFORM_BY_MESSAGE_KIND[self.message_kind]:
            raise ValueError("transform_kind disagrees with message_kind")
        _require_exact_int(
            self.stored_scalar_count,
            label="stored_scalar_count",
        )
        if self.fixed_weight is not None:
            if type(self.fixed_weight) not in (float, int):
                raise TypeError("fixed_weight must be a real number or None")
            if not math.isfinite(float(self.fixed_weight)):
                raise ValueError("fixed_weight must be finite")
        if self.message_kind == "identity_carry":
            if self.fixed_weight != 1.0 or self.stored_scalar_count != 0:
                raise ValueError(
                    "identity carry must have fixed weight 1 and zero scalars"
                )
        elif self.fixed_weight is not None:
            raise ValueError("only identity carry may have a fixed weight")

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "edge_id": self.edge_id,
            "message_kind": self.message_kind,
            "generator_invocation_id": self.generator_invocation_id,
            "generator_family_id": self.generator_family_id,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "source_carrier_version": self.source_carrier_version,
            "target_carrier_version": self.target_carrier_version,
            "routing_kind": self.routing_kind,
            "transform_kind": self.transform_kind,
            "fixed_weight": self.fixed_weight,
            "stored_scalar_count": self.stored_scalar_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TransportEdgeSpec:
        expected = {
            "edge_index",
            "edge_id",
            "message_kind",
            "generator_invocation_id",
            "generator_family_id",
            "source_event_id",
            "target_event_id",
            "source_carrier_version",
            "target_carrier_version",
            "routing_kind",
            "transform_kind",
            "fixed_weight",
            "stored_scalar_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("transport-edge fields do not match format version 1")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True)
class CarrierTransportManifest:
    """Immutable, authenticated graph topology."""

    lexical_invocation_id: str
    lexical_generator_family_id: str
    events: tuple[CarrierEventSpec, ...]
    edges: tuple[TransportEdgeSpec, ...]
    schema: str = CARRIER_TRANSPORT_MANIFEST_SCHEMA
    format_version: int = CARRIER_TRANSPORT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != CARRIER_TRANSPORT_MANIFEST_SCHEMA:
            raise ValueError("unsupported carrier-transport manifest schema")
        if self.format_version != CARRIER_TRANSPORT_FORMAT_VERSION:
            raise ValueError("unsupported carrier-transport manifest version")
        _require_id(self.lexical_invocation_id, label="lexical_invocation_id")
        _require_id(
            self.lexical_generator_family_id,
            label="lexical_generator_family_id",
        )
        if self.lexical_invocation_id == self.lexical_generator_family_id:
            raise ValueError("lexical family identity must differ from invocation")
        if type(self.events) is not tuple or not self.events:
            raise ValueError("events must be a nonempty tuple")
        if type(self.edges) is not tuple or not self.edges:
            raise ValueError("edges must be a nonempty tuple")
        if any(not isinstance(item, CarrierEventSpec) for item in self.events):
            raise TypeError("events must contain CarrierEventSpec values")
        if any(not isinstance(item, TransportEdgeSpec) for item in self.edges):
            raise TypeError("edges must contain TransportEdgeSpec values")
        _validate_manifest_topology(self)

    @property
    def initial_carrier_version(self) -> int:
        return 0

    @property
    def final_carrier_version(self) -> int:
        return self.events[-1].output_carrier_version

    @property
    def correction_count(self) -> int:
        return sum(edge.message_kind == "forward_correction" for edge in self.edges)

    @property
    def stored_scalar_count(self) -> int:
        family_counts: dict[str, int] = {}
        for edge in self.edges:
            previous = family_counts.setdefault(
                edge.generator_family_id,
                edge.stored_scalar_count,
            )
            if previous != edge.stored_scalar_count:
                raise ValueError("one generator family has inconsistent scalar counts")
        return sum(family_counts.values())

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "lexical_invocation_id": self.lexical_invocation_id,
            "lexical_generator_family_id": self.lexical_generator_family_id,
            "events": [event.to_dict() for event in self.events],
            "edges": [edge.to_dict() for edge in self.edges],
            "accounting": {
                "event_count": len(self.events),
                "edge_count": len(self.edges),
                "lexical_message_count": 1,
                "identity_carry_count": len(self.events),
                "primary_generator_message_count": len(self.events),
                "forward_correction_count": self.correction_count,
                "stored_scalar_count": self.stored_scalar_count,
                "initial_carrier_version": self.initial_carrier_version,
                "final_carrier_version": self.final_carrier_version,
                "commits_per_event": 1,
            },
            "semantics": {
                "aggregation": "ordered_identity_plus_primary_plus_corrections",
                "identity_carry_fixed_weight": 1.0,
                "identity_carry_stored_scalar_count": 0,
                "carrier_closure_required": True,
                "generator_family_is_invocation_identity": False,
                "attention_routing_is_state_conditioned": True,
                "attention_routing_is_causal": True,
                "corrections_must_be_strictly_forward": True,
            },
        }

    @property
    def fingerprint_sha256(self) -> str:
        return _fingerprint(_MANIFEST_HASH_DOMAIN, self.payload())

    def to_dict(self) -> dict[str, object]:
        result = self.payload()
        result["manifest_sha256"] = self.fingerprint_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CarrierTransportManifest:
        expected = {
            "schema",
            "format_version",
            "lexical_invocation_id",
            "lexical_generator_family_id",
            "events",
            "edges",
            "accounting",
            "semantics",
            "manifest_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("carrier-transport manifest fields are not exact")
        if not isinstance(value["events"], Sequence) or isinstance(
            value["events"], (str, bytes)
        ):
            raise TypeError("serialized events must be a sequence")
        if not isinstance(value["edges"], Sequence) or isinstance(
            value["edges"], (str, bytes)
        ):
            raise TypeError("serialized edges must be a sequence")
        manifest = cls(
            lexical_invocation_id=value["lexical_invocation_id"],
            lexical_generator_family_id=value["lexical_generator_family_id"],
            events=tuple(CarrierEventSpec.from_dict(item) for item in value["events"]),
            edges=tuple(TransportEdgeSpec.from_dict(item) for item in value["edges"]),
            schema=value["schema"],
            format_version=value["format_version"],
        )
        _require_sha256(value["manifest_sha256"], label="manifest_sha256")
        if manifest.to_dict() != dict(value):
            raise ValueError("carrier-transport manifest authentication failed")
        return manifest


def _validate_manifest_topology(manifest: CarrierTransportManifest) -> None:
    events = manifest.events
    edges = manifest.edges
    if [event.event_index for event in events] != sorted(
        event.event_index for event in events
    ):
        raise ValueError("event indexes must be strictly ordered")
    if len({event.event_index for event in events}) != len(events):
        raise ValueError("event indexes must be unique")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("event IDs must be unique")
    if [event.input_carrier_version for event in events] != list(
        range(len(events))
    ):
        raise ValueError("event inputs must consume every carrier version once")
    if [event.output_carrier_version for event in events] != list(
        range(1, len(events) + 1)
    ):
        raise ValueError("event outputs must close the sequential carrier wire")

    primary_invocations = [event.generator_invocation_id for event in events]
    if len(set(primary_invocations)) != len(primary_invocations):
        raise ValueError("generator invocation IDs must be unique")
    if manifest.lexical_invocation_id in set(primary_invocations):
        raise ValueError("lexical and mutation invocation IDs must differ")
    if [edge.edge_index for edge in edges] != list(range(len(edges))):
        raise ValueError("edge indexes must be contiguous and ordered")
    if len({edge.edge_id for edge in edges}) != len(edges):
        raise ValueError("edge IDs must be unique")
    if len({edge.generator_invocation_id for edge in edges}) != len(edges):
        raise ValueError("every message edge must have a unique invocation ID")
    family_contracts: dict[str, tuple[str, int]] = {}
    for edge in edges:
        contract = (edge.message_kind, edge.stored_scalar_count)
        previous = family_contracts.setdefault(edge.generator_family_id, contract)
        if previous != contract:
            raise ValueError(
                "one generator family cannot change message kind or scalar count"
            )

    event_by_id = {event.event_id: event for event in events}
    ingress = [edge for edge in edges if edge.message_kind == "lexical_ingress"]
    if len(ingress) != 1 or ingress[0] != edges[0]:
        raise ValueError("the first edge must be the only lexical ingress")
    ingress_edge = ingress[0]
    if (
        ingress_edge.generator_invocation_id != manifest.lexical_invocation_id
        or ingress_edge.generator_family_id
        != manifest.lexical_generator_family_id
        or ingress_edge.source_event_id is not None
        or ingress_edge.target_event_id is not None
        or ingress_edge.source_carrier_version is not None
        or ingress_edge.target_carrier_version != 0
    ):
        raise ValueError("lexical ingress edge contract is invalid")

    expected_order = [ingress_edge.edge_id]
    for event in events:
        target_edges = [
            edge for edge in edges if edge.target_event_id == event.event_id
        ]
        carries = [
            edge for edge in target_edges if edge.message_kind == "identity_carry"
        ]
        primary = [
            edge
            for edge in target_edges
            if edge.message_kind == event.primary_message_kind
        ]
        corrections = [
            edge
            for edge in target_edges
            if edge.message_kind == "forward_correction"
        ]
        if corrections != sorted(
            corrections,
            key=lambda edge: (edge.source_event_id, edge.edge_id),
        ):
            raise ValueError("correction edges are not in canonical order")
        if len(carries) != 1 or len(primary) != 1:
            raise ValueError("each event needs one carry and one primary message")
        if len(target_edges) != 2 + len(corrections):
            raise ValueError("an event contains an incompatible message kind")
        carry = carries[0]
        generator = primary[0]
        if (
            carry.source_event_id is not None
            or carry.source_carrier_version != event.input_carrier_version
            or carry.target_carrier_version != event.output_carrier_version
        ):
            raise ValueError("identity carry is wired to the wrong boundary")
        if (
            generator.source_event_id != event.event_id
            or generator.source_carrier_version != event.input_carrier_version
            or generator.target_carrier_version != event.output_carrier_version
            or generator.generator_invocation_id
            != event.generator_invocation_id
            or generator.generator_family_id != event.generator_family_id
        ):
            raise ValueError("primary generator is wired to the wrong boundary")
        expected_order.extend((carry.edge_id, generator.edge_id))
        for correction in corrections:
            source = event_by_id.get(correction.source_event_id)
            if source is None:
                raise ValueError("correction names an unknown source event")
            if source.event_index >= event.event_index:
                raise ValueError("correction messages must be strictly forward")
            if (
                correction.source_carrier_version
                != source.output_carrier_version
                or correction.source_carrier_version
                > event.input_carrier_version
                or correction.target_carrier_version
                != event.output_carrier_version
            ):
                raise ValueError("correction is wired to an invalid carrier version")
            expected_order.append(correction.edge_id)
    if [edge.edge_id for edge in edges] != expected_order:
        raise ValueError(
            "edges must be ordered ingress then carry, primary, corrections"
        )


def validate_carrier_transport_manifest(
    value: CarrierTransportManifest | Mapping[str, object],
) -> CarrierTransportManifest:
    """Authenticate a manifest object or its JSON-safe representation."""

    if isinstance(value, CarrierTransportManifest):
        # Reconstructing catches illicit nested tensor/object mutation and makes
        # the validation behavior identical to a loaded artifact.
        return CarrierTransportManifest.from_dict(value.to_dict())
    return CarrierTransportManifest.from_dict(value)


def build_causal_carrier_transport_manifest(
    *,
    lexical_invocation_id: str,
    lexical_generator_family_id: str,
    events: Sequence[CarrierEventSpec],
    corrections: Sequence[ForwardCorrectionSpec] = (),
    lexical_stored_scalar_count: int = 0,
    primary_stored_scalar_counts: Mapping[str, int] | None = None,
    correction_stored_scalar_counts: Mapping[str, int] | None = None,
) -> CarrierTransportManifest:
    """Build the canonical edge catalog for a sequential mutation schedule."""

    event_tuple = tuple(events)
    correction_tuple = tuple(corrections)
    _require_exact_int(
        lexical_stored_scalar_count,
        label="lexical_stored_scalar_count",
    )
    primary_counts = dict(primary_stored_scalar_counts or {})
    correction_counts = dict(correction_stored_scalar_counts or {})
    if set(primary_counts) - {event.event_id for event in event_tuple}:
        raise ValueError("primary scalar counts name an unknown event")
    if set(correction_counts) - {
        correction.correction_id for correction in correction_tuple
    }:
        raise ValueError("correction scalar counts name an unknown correction")
    for key, count in (*primary_counts.items(), *correction_counts.items()):
        _require_id(key, label="scalar-count key")
        _require_exact_int(count, label=f"stored scalar count for {key}")

    event_by_id = {event.event_id: event for event in event_tuple}
    if len(event_by_id) != len(event_tuple):
        raise ValueError("event IDs must be unique")
    corrections_by_target: dict[str, list[ForwardCorrectionSpec]] = {
        event.event_id: [] for event in event_tuple
    }
    correction_ids: set[str] = set()
    for correction in correction_tuple:
        if correction.correction_id in correction_ids:
            raise ValueError("correction IDs must be unique")
        correction_ids.add(correction.correction_id)
        source = event_by_id.get(correction.source_event_id)
        target = event_by_id.get(correction.target_event_id)
        if source is None or target is None:
            raise ValueError("correction source and target events must exist")
        if source.event_index >= target.event_index:
            raise ValueError("correction messages must be strictly forward")
        corrections_by_target[target.event_id].append(correction)

    edges: list[TransportEdgeSpec] = []

    def add_edge(**values: object) -> None:
        edges.append(TransportEdgeSpec(edge_index=len(edges), **values))

    add_edge(
        edge_id="edge.lexical.ingress",
        message_kind="lexical_ingress",
        generator_invocation_id=lexical_invocation_id,
        generator_family_id=lexical_generator_family_id,
        source_event_id=None,
        target_event_id=None,
        source_carrier_version=None,
        target_carrier_version=0,
        routing_kind=_ROUTING_BY_MESSAGE_KIND["lexical_ingress"],
        transform_kind=_TRANSFORM_BY_MESSAGE_KIND["lexical_ingress"],
        fixed_weight=None,
        stored_scalar_count=lexical_stored_scalar_count,
    )
    for event in event_tuple:
        add_edge(
            edge_id=f"edge.{event.event_id}.carry",
            message_kind="identity_carry",
            generator_invocation_id=f"carry:{event.event_id}",
            generator_family_id="carrier.identity",
            source_event_id=None,
            target_event_id=event.event_id,
            source_carrier_version=event.input_carrier_version,
            target_carrier_version=event.output_carrier_version,
            routing_kind=_ROUTING_BY_MESSAGE_KIND["identity_carry"],
            transform_kind=_TRANSFORM_BY_MESSAGE_KIND["identity_carry"],
            fixed_weight=1.0,
            stored_scalar_count=0,
        )
        add_edge(
            edge_id=f"edge.{event.event_id}.generator",
            message_kind=event.primary_message_kind,
            generator_invocation_id=event.generator_invocation_id,
            generator_family_id=event.generator_family_id,
            source_event_id=event.event_id,
            target_event_id=event.event_id,
            source_carrier_version=event.input_carrier_version,
            target_carrier_version=event.output_carrier_version,
            routing_kind=_ROUTING_BY_MESSAGE_KIND[event.primary_message_kind],
            transform_kind=_TRANSFORM_BY_MESSAGE_KIND[event.primary_message_kind],
            fixed_weight=None,
            stored_scalar_count=primary_counts.get(event.event_id, 0),
        )
        for correction in sorted(
            corrections_by_target[event.event_id],
            key=lambda item: (item.source_event_id, item.correction_id),
        ):
            source = event_by_id[correction.source_event_id]
            add_edge(
                edge_id=(
                    f"edge.{event.event_id}.correction.{correction.correction_id}"
                ),
                message_kind="forward_correction",
                generator_invocation_id=correction.generator_invocation_id,
                generator_family_id=correction.generator_family_id,
                source_event_id=source.event_id,
                target_event_id=event.event_id,
                source_carrier_version=source.output_carrier_version,
                target_carrier_version=event.output_carrier_version,
                routing_kind=_ROUTING_BY_MESSAGE_KIND["forward_correction"],
                transform_kind=_TRANSFORM_BY_MESSAGE_KIND["forward_correction"],
                fixed_weight=None,
                stored_scalar_count=correction_counts.get(
                    correction.correction_id,
                    0,
                ),
            )
    return CarrierTransportManifest(
        lexical_invocation_id=lexical_invocation_id,
        lexical_generator_family_id=lexical_generator_family_id,
        events=event_tuple,
        edges=tuple(edges),
    )


@dataclass(frozen=True, slots=True)
class TransportMessage:
    """A callback-produced message; only the core may aggregate it."""

    edge_id: str
    value: Tensor
    attention_routing_weights: Tensor | None = None

    def __post_init__(self) -> None:
        _require_id(self.edge_id, label="message edge_id")
        if not isinstance(self.value, Tensor):
            raise TypeError("message value must be a torch Tensor")
        if self.attention_routing_weights is not None and not isinstance(
            self.attention_routing_weights, Tensor
        ):
            raise TypeError("attention_routing_weights must be a Tensor or None")


@dataclass(frozen=True, slots=True)
class TransportContext:
    """Read-only routing context supplied to exactly one message callback."""

    manifest_sha256: str
    execution_nonce: str
    edge: TransportEdgeSpec
    event: CarrierEventSpec | None
    token_input: Tensor
    source_value: Tensor
    carrier_input: Tensor | None

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, label="manifest_sha256")
        _require_id(self.execution_nonce, label="execution_nonce")
        if not isinstance(self.edge, TransportEdgeSpec):
            raise TypeError("edge must be a TransportEdgeSpec")
        if self.event is not None and not isinstance(self.event, CarrierEventSpec):
            raise TypeError("event must be a CarrierEventSpec or None")
        if not isinstance(self.token_input, Tensor):
            raise TypeError("token_input must be a Tensor")
        if not isinstance(self.source_value, Tensor):
            raise TypeError("source_value must be a Tensor")
        if self.carrier_input is not None and not isinstance(
            self.carrier_input, Tensor
        ):
            raise TypeError("carrier_input must be a Tensor or None")


TransportProducer: TypeAlias = Callable[[TransportContext], TransportMessage]


def _validate_attention_routing(weights: Tensor, carrier: Tensor) -> None:
    if not weights.dtype.is_floating_point or not torch.isfinite(weights).all():
        raise ValueError("attention routing weights must be finite floating point")
    if weights.ndim < 2:
        raise ValueError("attention routing weights need query and key axes")
    if carrier.ndim < 2:
        raise ValueError("attention carrier needs token and feature axes")
    token_count = carrier.shape[-2]
    if tuple(weights.shape[-2:]) != (token_count, token_count):
        raise ValueError("attention routing axes must match carrier tokens")
    tolerance = 5e-3 if weights.dtype in (torch.float16, torch.bfloat16) else 1e-6
    if bool((weights < -tolerance).any()):
        raise ValueError("attention routing weights cannot be negative")
    row_sums = weights.to(dtype=torch.float64).sum(dim=-1)
    valid_rows = torch.isclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=0.0,
        atol=tolerance,
    )
    inert_rows = torch.isclose(
        row_sums,
        torch.zeros_like(row_sums),
        rtol=0.0,
        atol=tolerance,
    )
    if not bool(valid_rows.any()):
        raise ValueError("attention routing must contain at least one valid row")
    if not bool((valid_rows | inert_rows).all()):
        raise ValueError(
            "attention routing rows must be stochastic or zero and inert"
        )
    forbidden = torch.triu(
        weights.to(dtype=torch.float64),
        diagonal=1,
    )
    if float(forbidden.abs().max().item()) > tolerance:
        raise ValueError("attention routing weights must be causal")


def _validate_message(
    message: TransportMessage,
    *,
    edge: TransportEdgeSpec,
    carrier_reference: Tensor | None,
) -> None:
    if message.edge_id != edge.edge_id:
        raise ValueError("producer returned a message for the wrong edge")
    if not message.value.dtype.is_floating_point:
        raise ValueError("transport message values must be floating point")
    if not torch.isfinite(message.value).all():
        raise ValueError("transport message values must be finite")
    if carrier_reference is not None:
        if message.value.shape != carrier_reference.shape:
            raise ValueError("message shape must equal the target carrier shape")
        if message.value.dtype != carrier_reference.dtype:
            raise ValueError("message dtype must equal the carrier dtype")
        if message.value.device != carrier_reference.device:
            raise ValueError("message device must equal the carrier device")
    if edge.message_kind == "attention_transport":
        if message.attention_routing_weights is None:
            raise ValueError("attention messages require routing weights")
        assert carrier_reference is not None
        _validate_attention_routing(
            message.attention_routing_weights,
            carrier_reference,
        )
    elif message.attention_routing_weights is not None:
        raise ValueError("only attention messages may carry routing weights")


@dataclass(frozen=True, slots=True)
class TransportMessageReceipt:
    edge_index: int
    edge_id: str
    message_kind: str
    generator_invocation_id: str
    generator_family_id: str
    source_carrier_version: int | None
    target_carrier_version: int
    value_sha256: str
    routing_sha256: str | None
    stored_scalar_count: int

    def __post_init__(self) -> None:
        _require_exact_int(self.edge_index, label="receipt edge_index")
        _require_id(self.edge_id, label="receipt edge_id")
        if self.message_kind not in _MESSAGE_KINDS:
            raise ValueError("receipt message_kind is invalid")
        _require_id(
            self.generator_invocation_id,
            label="receipt generator_invocation_id",
        )
        _require_id(
            self.generator_family_id,
            label="receipt generator_family_id",
        )
        if self.generator_invocation_id == self.generator_family_id:
            raise ValueError("receipt family identity must differ from invocation")
        if self.source_carrier_version is not None:
            _require_exact_int(
                self.source_carrier_version,
                label="receipt source_carrier_version",
            )
        _require_exact_int(
            self.target_carrier_version,
            label="receipt target_carrier_version",
        )
        _require_sha256(self.value_sha256, label="message value_sha256")
        if self.routing_sha256 is not None:
            _require_sha256(self.routing_sha256, label="routing_sha256")
        _require_exact_int(
            self.stored_scalar_count,
            label="receipt stored_scalar_count",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_index": self.edge_index,
            "edge_id": self.edge_id,
            "message_kind": self.message_kind,
            "generator_invocation_id": self.generator_invocation_id,
            "generator_family_id": self.generator_family_id,
            "source_carrier_version": self.source_carrier_version,
            "target_carrier_version": self.target_carrier_version,
            "value_sha256": self.value_sha256,
            "routing_sha256": self.routing_sha256,
            "stored_scalar_count": self.stored_scalar_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TransportMessageReceipt:
        expected = {
            "edge_index",
            "edge_id",
            "message_kind",
            "generator_invocation_id",
            "generator_family_id",
            "source_carrier_version",
            "target_carrier_version",
            "value_sha256",
            "routing_sha256",
            "stored_scalar_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("message-receipt fields are not exact")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True)
class CarrierBoundaryReceipt:
    event_index: int
    event_id: str
    input_carrier_version: int
    output_carrier_version: int
    input_carrier_sha256: str
    message_receipts: tuple[TransportMessageReceipt, ...]
    output_carrier_sha256: str
    aggregate_commit_count: int = 1

    def __post_init__(self) -> None:
        _require_exact_int(self.event_index, label="boundary event_index")
        _require_id(self.event_id, label="boundary event_id")
        _require_exact_int(
            self.input_carrier_version,
            label="boundary input_carrier_version",
        )
        if self.output_carrier_version != self.input_carrier_version + 1:
            raise ValueError("boundary carrier versions are not consecutive")
        _require_sha256(
            self.input_carrier_sha256,
            label="input_carrier_sha256",
        )
        _require_sha256(
            self.output_carrier_sha256,
            label="output_carrier_sha256",
        )
        if type(self.message_receipts) is not tuple or len(
            self.message_receipts
        ) < 2:
            raise ValueError("boundary needs carry and primary message receipts")
        if self.message_receipts[0].message_kind != "identity_carry":
            raise ValueError("the first boundary message must be identity carry")
        if self.message_receipts[0].value_sha256 != self.input_carrier_sha256:
            raise ValueError("identity carry must hash exactly as boundary input")
        if self.aggregate_commit_count != 1:
            raise ValueError("each boundary must be committed exactly once")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_id": self.event_id,
            "input_carrier_version": self.input_carrier_version,
            "output_carrier_version": self.output_carrier_version,
            "input_carrier_sha256": self.input_carrier_sha256,
            "message_receipts": [item.to_dict() for item in self.message_receipts],
            "output_carrier_sha256": self.output_carrier_sha256,
            "aggregate_commit_count": self.aggregate_commit_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CarrierBoundaryReceipt:
        expected = {
            "event_index",
            "event_id",
            "input_carrier_version",
            "output_carrier_version",
            "input_carrier_sha256",
            "message_receipts",
            "output_carrier_sha256",
            "aggregate_commit_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("boundary-receipt fields are not exact")
        messages = value["message_receipts"]
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("serialized message_receipts must be a sequence")
        return cls(
            event_index=value["event_index"],
            event_id=value["event_id"],
            input_carrier_version=value["input_carrier_version"],
            output_carrier_version=value["output_carrier_version"],
            input_carrier_sha256=value["input_carrier_sha256"],
            message_receipts=tuple(
                TransportMessageReceipt.from_dict(item) for item in messages
            ),
            output_carrier_sha256=value["output_carrier_sha256"],
            aggregate_commit_count=value["aggregate_commit_count"],
        )


@dataclass(frozen=True, slots=True)
class CarrierTransportExecutionReceipt:
    manifest_sha256: str
    execution_nonce: str
    token_input_metadata: Mapping[str, object]
    lexical_message_receipt: TransportMessageReceipt
    initial_carrier_metadata: Mapping[str, object]
    boundaries: tuple[CarrierBoundaryReceipt, ...]
    final_carrier_metadata: Mapping[str, object]
    callback_call_count: int
    identity_carry_count: int
    primary_generator_message_count: int
    correction_message_count: int
    aggregate_commit_count: int
    stored_scalar_count: int
    schema: str = CARRIER_TRANSPORT_RECEIPT_SCHEMA
    format_version: int = CARRIER_TRANSPORT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != CARRIER_TRANSPORT_RECEIPT_SCHEMA:
            raise ValueError("unsupported carrier-transport receipt schema")
        if self.format_version != CARRIER_TRANSPORT_FORMAT_VERSION:
            raise ValueError("unsupported carrier-transport receipt version")
        _require_sha256(self.manifest_sha256, label="receipt manifest_sha256")
        _require_id(self.execution_nonce, label="receipt execution_nonce")
        if not isinstance(self.lexical_message_receipt, TransportMessageReceipt):
            raise TypeError("lexical_message_receipt has the wrong type")
        if self.lexical_message_receipt.message_kind != "lexical_ingress":
            raise ValueError("lexical receipt has the wrong message kind")
        if type(self.boundaries) is not tuple or not self.boundaries:
            raise ValueError("receipt boundaries must be a nonempty tuple")
        for name, metadata in (
            ("token_input_metadata", self.token_input_metadata),
            ("initial_carrier_metadata", self.initial_carrier_metadata),
            ("final_carrier_metadata", self.final_carrier_metadata),
        ):
            if not isinstance(metadata, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(
                self,
                name,
                _freeze_tensor_metadata(metadata, label=name),
            )
        event_count = len(self.boundaries)
        expected_callbacks = 1 + event_count + self.correction_message_count
        counts = {
            "callback_call_count": (self.callback_call_count, expected_callbacks),
            "identity_carry_count": (self.identity_carry_count, event_count),
            "primary_generator_message_count": (
                self.primary_generator_message_count,
                event_count,
            ),
            "aggregate_commit_count": (self.aggregate_commit_count, event_count),
        }
        for label, (actual, expected) in counts.items():
            _require_exact_int(actual, label=label)
            if actual != expected:
                raise ValueError(f"{label} does not close execution accounting")
        _require_exact_int(
            self.correction_message_count,
            label="correction_message_count",
        )
        _require_exact_int(self.stored_scalar_count, label="stored_scalar_count")
        if (
            self.initial_carrier_metadata["sha256"]
            != self.lexical_message_receipt.value_sha256
        ):
            raise ValueError("lexical message does not authenticate initial carrier")
        if (
            self.final_carrier_metadata["sha256"]
            != self.boundaries[-1].output_carrier_sha256
        ):
            raise ValueError("last boundary does not authenticate final carrier")
        previous = self.initial_carrier_metadata["sha256"]
        correction_count = 0
        family_counts = {
            self.lexical_message_receipt.generator_family_id: (
                self.lexical_message_receipt.stored_scalar_count
            )
        }
        for ordinal, boundary in enumerate(self.boundaries):
            if boundary.input_carrier_version != ordinal:
                raise ValueError("receipt boundary input versions are not closed")
            if boundary.input_carrier_sha256 != previous:
                raise ValueError("receipt carrier hash chain is broken")
            previous = boundary.output_carrier_sha256
            correction_count += sum(
                item.message_kind == "forward_correction"
                for item in boundary.message_receipts
            )
            for item in boundary.message_receipts:
                previous_count = family_counts.setdefault(
                    item.generator_family_id,
                    item.stored_scalar_count,
                )
                if previous_count != item.stored_scalar_count:
                    raise ValueError(
                        "one receipt family has inconsistent scalar counts"
                    )
        if correction_count != self.correction_message_count:
            raise ValueError("correction receipt accounting does not close")
        if sum(family_counts.values()) != self.stored_scalar_count:
            raise ValueError("stored scalar accounting does not close")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "manifest_sha256": self.manifest_sha256,
            "execution_nonce": self.execution_nonce,
            "token_input_metadata": _tensor_metadata_payload(
                self.token_input_metadata
            ),
            "lexical_message_receipt": self.lexical_message_receipt.to_dict(),
            "initial_carrier_metadata": _tensor_metadata_payload(
                self.initial_carrier_metadata
            ),
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "final_carrier_metadata": _tensor_metadata_payload(
                self.final_carrier_metadata
            ),
            "accounting": {
                "callback_call_count": self.callback_call_count,
                "identity_carry_count": self.identity_carry_count,
                "primary_generator_message_count": (
                    self.primary_generator_message_count
                ),
                "correction_message_count": self.correction_message_count,
                "aggregate_commit_count": self.aggregate_commit_count,
                "stored_scalar_count": self.stored_scalar_count,
            },
            "claims": {
                "live_carrier_advanced_by_core": True,
                "one_aggregate_commit_per_event": True,
                "identity_carry_exact": True,
                "residual_carrier_closure": True,
                "post_hoc_reconstruction": False,
                "compression": False,
            },
        }

    @property
    def fingerprint_sha256(self) -> str:
        return _fingerprint(_RECEIPT_HASH_DOMAIN, self.payload())

    def to_dict(self) -> dict[str, object]:
        result = self.payload()
        result["receipt_sha256"] = self.fingerprint_sha256
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> CarrierTransportExecutionReceipt:
        expected = {
            "schema",
            "format_version",
            "manifest_sha256",
            "execution_nonce",
            "token_input_metadata",
            "lexical_message_receipt",
            "initial_carrier_metadata",
            "boundaries",
            "final_carrier_metadata",
            "accounting",
            "claims",
            "receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("carrier-transport receipt fields are not exact")
        accounting = value["accounting"]
        expected_accounting = {
            "callback_call_count",
            "identity_carry_count",
            "primary_generator_message_count",
            "correction_message_count",
            "aggregate_commit_count",
            "stored_scalar_count",
        }
        if not isinstance(accounting, Mapping) or set(accounting) != (
            expected_accounting
        ):
            raise ValueError("carrier-transport receipt accounting is not exact")
        expected_claims = {
            "live_carrier_advanced_by_core": True,
            "one_aggregate_commit_per_event": True,
            "identity_carry_exact": True,
            "residual_carrier_closure": True,
            "post_hoc_reconstruction": False,
            "compression": False,
        }
        if value["claims"] != expected_claims:
            raise ValueError("carrier-transport receipt claims are not exact")
        boundaries = value["boundaries"]
        if not isinstance(boundaries, Sequence) or isinstance(
            boundaries, (str, bytes)
        ):
            raise TypeError("serialized receipt boundaries must be a sequence")
        for name in (
            "token_input_metadata",
            "initial_carrier_metadata",
            "final_carrier_metadata",
        ):
            if not isinstance(value[name], Mapping):
                raise TypeError(f"serialized {name} must be a mapping")
        lexical = value["lexical_message_receipt"]
        if not isinstance(lexical, Mapping):
            raise TypeError("serialized lexical message receipt must be a mapping")
        receipt = cls(
            manifest_sha256=value["manifest_sha256"],
            execution_nonce=value["execution_nonce"],
            token_input_metadata=value["token_input_metadata"],
            lexical_message_receipt=TransportMessageReceipt.from_dict(lexical),
            initial_carrier_metadata=value["initial_carrier_metadata"],
            boundaries=tuple(
                CarrierBoundaryReceipt.from_dict(item) for item in boundaries
            ),
            final_carrier_metadata=value["final_carrier_metadata"],
            callback_call_count=accounting["callback_call_count"],
            identity_carry_count=accounting["identity_carry_count"],
            primary_generator_message_count=accounting[
                "primary_generator_message_count"
            ],
            correction_message_count=accounting["correction_message_count"],
            aggregate_commit_count=accounting["aggregate_commit_count"],
            stored_scalar_count=accounting["stored_scalar_count"],
            schema=value["schema"],
            format_version=value["format_version"],
        )
        _require_sha256(value["receipt_sha256"], label="receipt_sha256")
        if receipt.to_dict() != dict(value):
            raise ValueError("carrier-transport receipt authentication failed")
        return receipt


@dataclass(frozen=True, slots=True)
class CarrierTransportExecution:
    """Live carrier result plus immutable JSON-safe execution receipt."""

    final_carrier: Tensor
    carrier_snapshots: tuple[Tensor, ...]
    receipt: CarrierTransportExecutionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.final_carrier, Tensor):
            raise TypeError("final_carrier must be a Tensor")
        if type(self.carrier_snapshots) is not tuple or not self.carrier_snapshots:
            raise ValueError("carrier_snapshots must be a nonempty tuple")
        if not isinstance(self.receipt, CarrierTransportExecutionReceipt):
            raise TypeError("receipt has the wrong type")
        if len(self.carrier_snapshots) != len(self.receipt.boundaries) + 1:
            raise ValueError("carrier snapshot count does not close boundaries")
        for version, snapshot in enumerate(self.carrier_snapshots):
            if not isinstance(snapshot, Tensor):
                raise TypeError("carrier snapshots must be tensors")
            expected = (
                self.receipt.initial_carrier_metadata["sha256"]
                if version == 0
                else self.receipt.boundaries[version - 1].output_carrier_sha256
            )
            if _tensor_sha256(snapshot) != expected:
                raise ValueError("carrier snapshot does not match its receipt")
        if _tensor_sha256(self.final_carrier) != self.receipt.final_carrier_metadata[
            "sha256"
        ]:
            raise ValueError("final carrier does not match its receipt")


def _message_receipt(
    edge: TransportEdgeSpec,
    value: Tensor,
    routing: Tensor | None,
) -> TransportMessageReceipt:
    return TransportMessageReceipt(
        edge_index=edge.edge_index,
        edge_id=edge.edge_id,
        message_kind=edge.message_kind,
        generator_invocation_id=edge.generator_invocation_id,
        generator_family_id=edge.generator_family_id,
        source_carrier_version=edge.source_carrier_version,
        target_carrier_version=edge.target_carrier_version,
        value_sha256=_tensor_sha256(value),
        routing_sha256=None if routing is None else _tensor_sha256(routing),
        stored_scalar_count=edge.stored_scalar_count,
    )


def execute_causal_carrier_transport_graph(
    manifest: CarrierTransportManifest,
    *,
    token_input: Tensor,
    message_producers: Mapping[str, TransportProducer],
    execution_nonce: str,
) -> CarrierTransportExecution:
    """Advance a live carrier through the authenticated transport graph.

    ``message_producers`` is keyed by every non-carry edge ID.  Each producer
    is called once.  Identity carry is owned by this function and cannot be
    overridden.  Existing carrier inputs are hashed before and after callbacks
    so in-place mutation fails closed.
    """

    manifest = validate_carrier_transport_manifest(manifest)
    if not isinstance(token_input, Tensor):
        raise TypeError("token_input must be a torch Tensor")
    if not isinstance(message_producers, Mapping):
        raise TypeError("message_producers must be a mapping")
    _require_id(execution_nonce, label="execution_nonce")
    required = {
        edge.edge_id
        for edge in manifest.edges
        if edge.message_kind != "identity_carry"
    }
    if set(message_producers) != required:
        raise ValueError("message_producers must exactly cover non-carry edges")
    if any(not callable(producer) for producer in message_producers.values()):
        raise TypeError("every message producer must be callable")

    event_by_id = {event.event_id: event for event in manifest.events}
    edges_by_target: dict[str, tuple[TransportEdgeSpec, ...]] = {}
    for event in manifest.events:
        edges_by_target[event.event_id] = tuple(
            edge for edge in manifest.edges if edge.target_event_id == event.event_id
        )

    ingress = manifest.edges[0]
    input_hash_before = _tensor_sha256(token_input)
    ingress_message = message_producers[ingress.edge_id](
        TransportContext(
            manifest_sha256=manifest.fingerprint_sha256,
            execution_nonce=execution_nonce,
            edge=ingress,
            event=None,
            token_input=token_input,
            source_value=token_input,
            carrier_input=None,
        )
    )
    if _tensor_sha256(token_input) != input_hash_before:
        raise RuntimeError("lexical producer mutated token_input in place")
    if not isinstance(ingress_message, TransportMessage):
        raise TypeError("message producers must return TransportMessage")
    _validate_message(ingress_message, edge=ingress, carrier_reference=None)
    carrier = _clone(ingress_message.value)
    snapshots: list[Tensor] = [_clone(carrier)]
    ingress_receipt = _message_receipt(
        ingress,
        carrier,
        ingress_message.attention_routing_weights,
    )
    boundaries: list[CarrierBoundaryReceipt] = []

    callback_count = 1
    correction_count = 0
    for event in manifest.events:
        target_edges = edges_by_target[event.event_id]
        carry_edge = target_edges[0]
        primary_edge = target_edges[1]
        correction_edges = target_edges[2:]
        input_hash = _tensor_sha256(carrier)

        produced: list[tuple[TransportEdgeSpec, Tensor, Tensor | None]] = [
            (carry_edge, carrier, None)
        ]
        callback_edges = (primary_edge, *correction_edges)
        for edge in callback_edges:
            if edge.message_kind == "forward_correction":
                assert edge.source_carrier_version is not None
                source_value = snapshots[edge.source_carrier_version]
            else:
                source_value = carrier
            source_hash = _tensor_sha256(source_value)
            carrier_hash = _tensor_sha256(carrier)
            token_hash = _tensor_sha256(token_input)
            message = message_producers[edge.edge_id](
                TransportContext(
                    manifest_sha256=manifest.fingerprint_sha256,
                    execution_nonce=execution_nonce,
                    edge=edge,
                    event=event_by_id[edge.target_event_id],
                    token_input=token_input,
                    source_value=source_value,
                    carrier_input=carrier,
                )
            )
            callback_count += 1
            if (
                _tensor_sha256(source_value) != source_hash
                or _tensor_sha256(carrier) != carrier_hash
                or _tensor_sha256(token_input) != token_hash
            ):
                raise RuntimeError("message producer mutated its context in place")
            if not isinstance(message, TransportMessage):
                raise TypeError("message producers must return TransportMessage")
            _validate_message(message, edge=edge, carrier_reference=carrier)
            produced.append(
                (
                    edge,
                    message.value,
                    message.attention_routing_weights,
                )
            )

        # One live commit.  The addition order is topology-authenticated and
        # deliberately begins with the exact identity carrier.
        next_carrier = produced[0][1] + produced[1][1]
        for _, value, _ in produced[2:]:
            next_carrier = next_carrier + value
        if not torch.isfinite(next_carrier).all():
            raise ValueError("aggregated carrier must be finite")
        message_receipts = tuple(
            _message_receipt(edge, value, routing)
            for edge, value, routing in produced
        )
        carrier = next_carrier
        snapshots.append(_clone(carrier))
        boundaries.append(
            CarrierBoundaryReceipt(
                event_index=event.event_index,
                event_id=event.event_id,
                input_carrier_version=event.input_carrier_version,
                output_carrier_version=event.output_carrier_version,
                input_carrier_sha256=input_hash,
                message_receipts=message_receipts,
                output_carrier_sha256=_tensor_sha256(carrier),
            )
        )
        correction_count += len(correction_edges)

    stored_scalar_count = manifest.stored_scalar_count
    receipt = CarrierTransportExecutionReceipt(
        manifest_sha256=manifest.fingerprint_sha256,
        execution_nonce=execution_nonce,
        token_input_metadata=_tensor_metadata(token_input),
        lexical_message_receipt=ingress_receipt,
        initial_carrier_metadata=_tensor_metadata(snapshots[0]),
        boundaries=tuple(boundaries),
        final_carrier_metadata=_tensor_metadata(carrier),
        callback_call_count=callback_count,
        identity_carry_count=len(manifest.events),
        primary_generator_message_count=len(manifest.events),
        correction_message_count=correction_count,
        aggregate_commit_count=len(manifest.events),
        stored_scalar_count=stored_scalar_count,
    )
    return CarrierTransportExecution(
        final_carrier=_clone(carrier),
        carrier_snapshots=tuple(snapshots),
        receipt=receipt,
    )


def validate_carrier_transport_execution_receipt(
    receipt: CarrierTransportExecutionReceipt | Mapping[str, object],
    *,
    manifest: CarrierTransportManifest,
) -> CarrierTransportExecutionReceipt:
    """Validate exact receipt accounting against an authenticated manifest."""

    manifest = validate_carrier_transport_manifest(manifest)
    if isinstance(receipt, Mapping):
        receipt = CarrierTransportExecutionReceipt.from_dict(receipt)
    if not isinstance(receipt, CarrierTransportExecutionReceipt):
        raise TypeError("receipt must be a receipt or serialized mapping")
    if receipt.manifest_sha256 != manifest.fingerprint_sha256:
        raise ValueError("receipt belongs to a different manifest")
    if len(receipt.boundaries) != len(manifest.events):
        raise ValueError("receipt event count disagrees with manifest")
    if receipt.correction_message_count != manifest.correction_count:
        raise ValueError("receipt correction count disagrees with manifest")
    if receipt.stored_scalar_count != manifest.stored_scalar_count:
        raise ValueError("receipt scalar count disagrees with manifest")
    flat_receipts = [receipt.lexical_message_receipt]
    for boundary in receipt.boundaries:
        flat_receipts.extend(boundary.message_receipts)
    if [item.edge_id for item in flat_receipts] != [
        edge.edge_id for edge in manifest.edges
    ]:
        raise ValueError("receipt edge order disagrees with manifest")
    for edge, item in zip(manifest.edges, flat_receipts, strict=True):
        if (
            edge.edge_index != item.edge_index
            or edge.message_kind != item.message_kind
            or edge.generator_invocation_id != item.generator_invocation_id
            or edge.generator_family_id != item.generator_family_id
            or edge.source_carrier_version != item.source_carrier_version
            or edge.target_carrier_version != item.target_carrier_version
            or edge.stored_scalar_count != item.stored_scalar_count
        ):
            raise ValueError("receipt edge metadata disagrees with manifest")
        if (edge.message_kind == "attention_transport") != (
            item.routing_sha256 is not None
        ):
            raise ValueError("attention routing receipt coverage is invalid")
    return receipt


__all__ = [
    "CARRIER_TRANSPORT_FORMAT_VERSION",
    "CARRIER_TRANSPORT_MANIFEST_SCHEMA",
    "CARRIER_TRANSPORT_RECEIPT_SCHEMA",
    "CarrierBoundaryReceipt",
    "CarrierEventSpec",
    "CarrierTransportExecution",
    "CarrierTransportExecutionReceipt",
    "CarrierTransportManifest",
    "ForwardCorrectionSpec",
    "TransportContext",
    "TransportEdgeSpec",
    "TransportMessage",
    "TransportMessageReceipt",
    "TransportProducer",
    "build_causal_carrier_transport_manifest",
    "execute_causal_carrier_transport_graph",
    "validate_carrier_transport_execution_receipt",
    "validate_carrier_transport_manifest",
]
