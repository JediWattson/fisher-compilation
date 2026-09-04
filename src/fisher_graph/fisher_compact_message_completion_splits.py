"""Exact unit-level split mechanics for compact-message campaigns.

This module implements the H9 split-key byte contract, integer Hamilton
allocation, and G10 whole-panel stratum coarsening.  It is deliberately
independent of tensors and corpus payloads: only irreversible analysis-unit
commitments and sanitized stratum labels are accepted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Final, Literal, TypeAlias


__all__ = [
    "FROZEN_SPLIT_PURPOSES",
    "SplitAssignment",
    "SplitProtocolError",
    "SplitUnit",
    "StratumDecision",
    "StratumLabel",
    "canonical_stratum_label_bytes",
    "choose_strata",
    "hamilton_allocations",
    "split_key",
    "stratified_binary_split",
    "stratum_order_key",
]


FROZEN_SPLIT_PURPOSES: Final = frozenset(
    {
        "A-fit-train-vs-fit-monitor",
        "D-tune-vs-development-confirm",
        "O1-calibration-vs-validation",
    }
)
_SPLIT_DOMAIN: Final = b"fisher-cmc/split-key/v1"
_STRATUM_DOMAIN: Final = b"fisher-cmc/stratum-order/v1"
_HEX: Final = frozenset("0123456789abcdef")

JsonLabel: TypeAlias = str | int | bool | None | list["JsonLabel"] | dict[str, "JsonLabel"]
IndependenceMode: TypeAlias = Literal["mode_a_template_lineage", "mode_b_semantic_instance"]
StratumScheme: TypeAlias = Literal["family_template", "family", "global"]


class SplitProtocolError(ValueError):
    """Split inputs violate the frozen deterministic protocol."""


def _raw_commitment(value: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise SplitProtocolError(
            "analysis-unit commitments must be 64 lowercase hexadecimal characters"
        )
    return bytes.fromhex(value)


def _validate_json_label(value: object, *, path: str = "$") -> None:
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        if not -(2**53) + 1 <= value <= (2**53) - 1:
            raise SplitProtocolError(f"{path} integer is outside exact JSON range")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_label(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SplitProtocolError(f"{path} contains a non-string key")
            _validate_json_label(item, path=f"{path}.{key}")
        return
    raise SplitProtocolError(f"{path} contains unsupported label type")


def canonical_stratum_label_bytes(value: JsonLabel) -> bytes:
    """Return RFC-8785-equivalent bytes for the protocol's exact JSON subset."""

    _validate_json_label(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def split_key(purpose: str, analysis_unit_commitment: str) -> bytes:
    """Compute the exact H9 raw 32-byte split key."""

    if purpose not in FROZEN_SPLIT_PURPOSES:
        raise SplitProtocolError("purpose is not one of the three frozen identifiers")
    purpose_bytes = purpose.encode("ascii")
    if len(purpose_bytes) > 0xFFFF:
        raise SplitProtocolError("purpose exceeds uint16 length")
    return hashlib.sha256(
        _SPLIT_DOMAIN
        + b"\x00"
        + struct.pack(">H", len(purpose_bytes))
        + purpose_bytes
        + b"\x00"
        + _raw_commitment(analysis_unit_commitment)
    ).digest()


def stratum_order_key(label: JsonLabel) -> tuple[bytes, bytes]:
    """Return the H9 Hamilton tie key and canonical collision fallback."""

    encoded = canonical_stratum_label_bytes(label)
    digest = hashlib.sha256(
        _STRATUM_DOMAIN
        + b"\x00"
        + struct.pack(">I", len(encoded))
        + encoded
    ).digest()
    return digest, encoded


@dataclass(frozen=True, slots=True)
class SplitUnit:
    analysis_unit_commitment: str
    family_id: str
    template_id: str | None

    def __post_init__(self) -> None:
        _raw_commitment(self.analysis_unit_commitment)
        if not isinstance(self.family_id, str) or not self.family_id:
            raise SplitProtocolError("family_id must be a nonempty string")
        if self.template_id is not None and (
            not isinstance(self.template_id, str) or not self.template_id
        ):
            raise SplitProtocolError("template_id must be null or nonempty")


@dataclass(frozen=True, slots=True, order=True)
class StratumLabel:
    scheme: StratumScheme
    family_id: str | None
    template_id: str | None

    def as_json(self) -> dict[str, JsonLabel]:
        return {
            "family_id": self.family_id,
            "scheme": self.scheme,
            "template_id": self.template_id,
        }


@dataclass(frozen=True, slots=True)
class StratumDecision:
    initial_scheme: StratumScheme
    initial_counts: tuple[tuple[StratumLabel, int], ...]
    final_scheme: StratumScheme
    final_counts: tuple[tuple[StratumLabel, int], ...]
    coarsening_reason: str | None


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    purpose: str
    left_name: str
    right_name: str
    requested_left_count: int
    stratum_decision: StratumDecision
    allocation_by_stratum: tuple[tuple[StratumLabel, int], ...]
    left_units: tuple[str, ...]
    right_units: tuple[str, ...]


def _label(unit: SplitUnit, scheme: StratumScheme) -> StratumLabel:
    if scheme == "family_template":
        if unit.template_id is None:
            raise SplitProtocolError("family-template strata require template_id")
        return StratumLabel(scheme, unit.family_id, unit.template_id)
    if scheme == "family":
        return StratumLabel(scheme, unit.family_id, None)
    return StratumLabel("global", None, None)


def _counts(
    units: Sequence[SplitUnit], scheme: StratumScheme
) -> tuple[tuple[StratumLabel, int], ...]:
    counter = Counter(_label(unit, scheme) for unit in units)
    return tuple(sorted(counter.items()))


def choose_strata(
    units: Sequence[SplitUnit],
    independence_mode: IndependenceMode,
    *,
    force_family: bool = False,
) -> StratumDecision:
    """Apply G10 coarsening to the entire panel, never selected cells."""

    if not units:
        raise SplitProtocolError("a split panel must contain units")
    if independence_mode not in {
        "mode_a_template_lineage",
        "mode_b_semantic_instance",
    }:
        raise SplitProtocolError("unknown independence mode")
    if force_family or independence_mode == "mode_a_template_lineage":
        initial: StratumScheme = "family"
    else:
        initial = "family_template"
    initial_counts = _counts(units, initial)
    if all(count >= 2 for _, count in initial_counts):
        return StratumDecision(initial, initial_counts, initial, initial_counts, None)

    family_counts = _counts(units, "family")
    if all(count >= 2 for _, count in family_counts):
        return StratumDecision(
            initial,
            initial_counts,
            "family",
            family_counts,
            "initial_scheme_contains_cell_below_two_units",
        )
    global_counts = _counts(units, "global")
    return StratumDecision(
        initial,
        initial_counts,
        "global",
        global_counts,
        "family_scheme_contains_cell_below_two_units",
    )


def hamilton_allocations(
    stratum_counts: Mapping[StratumLabel, int],
    requested: int,
) -> dict[StratumLabel, int]:
    """Allocate an exact count using only integer quotient and remainder."""

    if not isinstance(requested, int) or isinstance(requested, bool):
        raise SplitProtocolError("requested allocation must be an integer")
    if not stratum_counts:
        raise SplitProtocolError("stratum counts must not be empty")
    normalized: dict[StratumLabel, int] = {}
    for label, count in stratum_counts.items():
        if not isinstance(label, StratumLabel):
            raise SplitProtocolError("stratum keys must be StratumLabel values")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SplitProtocolError("stratum counts must be positive integers")
        normalized[label] = count
    total = sum(normalized.values())
    if not 0 <= requested <= total:
        raise SplitProtocolError("requested allocation is outside panel size")

    allocations = {
        label: (requested * count) // total
        for label, count in normalized.items()
    }
    remainders = {
        label: (requested * count) % total
        for label, count in normalized.items()
    }
    remaining = requested - sum(allocations.values())
    ordered = sorted(
        normalized,
        key=lambda label: (
            -remainders[label],
            *stratum_order_key(label.as_json()),
        ),
    )
    for label in ordered[:remaining]:
        allocations[label] += 1
    if sum(allocations.values()) != requested:
        raise AssertionError("integer Hamilton allocation failed to close")
    return allocations


def stratified_binary_split(
    units: Iterable[SplitUnit],
    *,
    purpose: str,
    requested_left_count: int,
    left_name: str,
    right_name: str,
    independence_mode: IndependenceMode,
    force_family_strata: bool = False,
) -> SplitAssignment:
    """Assign whole analysis units by Hamilton allocation and raw split keys."""

    materialized = tuple(units)
    if not isinstance(left_name, str) or not left_name:
        raise SplitProtocolError("left_name must be nonempty")
    if not isinstance(right_name, str) or not right_name:
        raise SplitProtocolError("right_name must be nonempty")
    commitments = [unit.analysis_unit_commitment for unit in materialized]
    if len(commitments) != len(set(commitments)):
        raise SplitProtocolError("analysis-unit commitments must be unique")

    decision = choose_strata(
        materialized,
        independence_mode,
        force_family=force_family_strata,
    )
    by_stratum: dict[StratumLabel, list[SplitUnit]] = {
        label: [] for label, _ in decision.final_counts
    }
    for unit in materialized:
        by_stratum[_label(unit, decision.final_scheme)].append(unit)
    allocations = hamilton_allocations(
        dict(decision.final_counts), requested_left_count
    )

    left: list[tuple[bytes, str]] = []
    right: list[tuple[bytes, str]] = []
    for label in sorted(by_stratum):
        ordered = sorted(
            by_stratum[label],
            key=lambda unit: (
                split_key(purpose, unit.analysis_unit_commitment),
                _raw_commitment(unit.analysis_unit_commitment),
            ),
        )
        boundary = allocations[label]
        for index, unit in enumerate(ordered):
            keyed = (
                split_key(purpose, unit.analysis_unit_commitment),
                unit.analysis_unit_commitment,
            )
            (left if index < boundary else right).append(keyed)
    left.sort()
    right.sort()
    return SplitAssignment(
        purpose=purpose,
        left_name=left_name,
        right_name=right_name,
        requested_left_count=requested_left_count,
        stratum_decision=decision,
        allocation_by_stratum=tuple(sorted(allocations.items())),
        left_units=tuple(commitment for _, commitment in left),
        right_units=tuple(commitment for _, commitment in right),
    )
