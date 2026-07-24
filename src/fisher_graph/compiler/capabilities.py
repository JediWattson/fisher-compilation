"""Canonical, fail-closed sequence capability matching.

Adapter sequence specifications, manifest-v1 guards, and executable backends
describe different parts of the same runtime contract.  This module converts
those facts into one in-memory representation.  Missing facts remain explicit
``Unknown`` values; they are never treated as permission to execute compiled
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from ..adapters.base import SequenceContext
from .manifest import SequenceSpec as ManifestSequenceSpec


class MatchStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CapabilityMatch:
    status: MatchStatus
    reasons: tuple[str, ...] = ()

    @property
    def matched(self) -> bool:
        return self.status is MatchStatus.MATCH


@dataclass(frozen=True, slots=True)
class CapabilityValues:
    """A known finite set or one explicitly unknown capability field."""

    values: frozenset[str] | None
    unknown_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.values is None) == (self.unknown_reason is None):
            raise ValueError(
                "capability values require exactly one of values or "
                "unknown_reason"
            )
        if self.values is not None and any(
            not isinstance(value, str) or not value for value in self.values
        ):
            raise ValueError("known capability values must be nonempty strings")
        if self.unknown_reason is not None and not self.unknown_reason:
            raise ValueError("unknown capability reason cannot be empty")

    @classmethod
    def known(cls, *values: str) -> CapabilityValues:
        return cls(frozenset(values))

    @classmethod
    def unknown(cls, reason: str) -> CapabilityValues:
        return cls(None, reason)

    @property
    def is_known(self) -> bool:
        return self.values is not None

    def match(self, value: str, *, field: str) -> CapabilityMatch:
        if self.values is None:
            return CapabilityMatch(
                MatchStatus.UNKNOWN,
                (f"{field}: {self.unknown_reason}",),
            )
        if value not in self.values:
            return CapabilityMatch(
                MatchStatus.MISMATCH,
                (f"{field}: {value!r} is unsupported",),
            )
        return CapabilityMatch(MatchStatus.MATCH)


@dataclass(frozen=True, slots=True)
class LengthDomain:
    minimum: int
    maximum: int | None

    def __post_init__(self) -> None:
        if type(self.minimum) is not int or self.minimum <= 0:
            raise ValueError("minimum length must be positive")
        if self.maximum is not None and (
            type(self.maximum) is not int or self.maximum < self.minimum
        ):
            raise ValueError("maximum length must be at least minimum length")

    def contains(self, length: int) -> bool:
        return (
            type(length) is int
            and length >= self.minimum
            and (self.maximum is None or length <= self.maximum)
        )


@dataclass(frozen=True, slots=True)
class SequenceCapabilitySet:
    """All facts needed to authorize one compiled sequence execution."""

    length: LengthDomain
    executions: CapabilityValues
    qk_relations: CapabilityValues
    position_relations: CapabilityValues
    mask_origins: CapabilityValues
    mask_patterns: CapabilityValues
    mask_representations: CapabilityValues
    visibility_families: CapabilityValues
    position_origins: CapabilityValues
    position_domains: CapabilityValues
    cache_kinds: CapabilityValues
    dtypes: CapabilityValues
    devices: CapabilityValues
    layouts: CapabilityValues


@dataclass(frozen=True, slots=True)
class SequenceRequest:
    """Concrete normalized semantics plus caller-input provenance."""

    query_length: int
    key_length: int
    execution: str
    qk_relation: str
    position_relation: str
    mask_origin: str
    mask_pattern: str
    mask_representation: str
    visibility_family: str
    position_origin: str
    position_domain: str
    cache_kind: str
    dtype: str
    device: str
    layout: str

    def __post_init__(self) -> None:
        if type(self.query_length) is not int or self.query_length <= 0:
            raise ValueError("query_length must be positive")
        if type(self.key_length) is not int or self.key_length <= 0:
            raise ValueError("key_length must be positive")
        for field, value in (
            ("execution", self.execution),
            ("qk_relation", self.qk_relation),
            ("position_relation", self.position_relation),
            ("mask_origin", self.mask_origin),
            ("mask_pattern", self.mask_pattern),
            ("mask_representation", self.mask_representation),
            ("visibility_family", self.visibility_family),
            ("position_origin", self.position_origin),
            ("position_domain", self.position_domain),
            ("cache_kind", self.cache_kind),
            ("dtype", self.dtype),
            ("device", self.device),
            ("layout", self.layout),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string")


def _row_mask_pattern(row: Tensor) -> str:
    row = row.to(dtype=torch.bool)
    length = row.numel()
    valid = int(row.sum().item())
    if valid == length:
        return "all_valid"
    if valid == 0:
        return "sparse"
    indices = torch.arange(length, device=row.device)
    if torch.equal(row, indices < valid):
        return "right_padded"
    if torch.equal(row, indices >= length - valid):
        return "left_padded"
    return "sparse"


def _mask_pattern(context: SequenceContext) -> str:
    if (
        context.query_valid_mask.shape != context.key_valid_mask.shape
        or not torch.equal(
            context.query_valid_mask,
            context.key_valid_mask,
        )
    ):
        return "custom"
    patterns = {
        _row_mask_pattern(row) for row in context.key_valid_mask
    }
    if patterns == {"all_valid"}:
        return "all_valid"
    if patterns <= {"all_valid", "right_padded"}:
        return "right_padded"
    if patterns <= {"all_valid", "left_padded"}:
        return "left_padded"
    if patterns <= {"all_valid", "left_padded", "right_padded"}:
        return "mixed_padded"
    return "sparse"


def _one_position_domain(positions: Tensor, valid_mask: Tensor) -> str:
    domains: set[str] = set()
    for row_positions, row_valid in zip(
        positions,
        valid_mask,
        strict=True,
    ):
        valid_positions = row_positions[row_valid]
        if valid_positions.numel() == 0:
            return "invalid"
        if (valid_positions < 0).any():
            return "invalid"
        if valid_positions.numel() > 1:
            differences = valid_positions[1:] - valid_positions[:-1]
            if (differences <= 0).any():
                return "invalid"
            contiguous = bool((differences == 1).all())
        else:
            contiguous = True
        if contiguous and int(valid_positions[0].item()) == 0:
            domains.add("zero_contiguous")
        elif contiguous:
            domains.add("offset_contiguous")
        else:
            domains.add("arbitrary")
    if domains <= {"zero_contiguous"}:
        return "zero_contiguous"
    if domains <= {"zero_contiguous", "offset_contiguous"}:
        return "offset_contiguous"
    return "arbitrary"


def _position_domain(context: SequenceContext) -> str:
    query = _one_position_domain(
        context.logical_positions,
        context.query_valid_mask,
    )
    key = _one_position_domain(
        context.key_logical_positions,
        context.key_valid_mask,
    )
    if "invalid" in (query, key):
        return "invalid"
    domains = {query, key}
    if domains == {"zero_contiguous"}:
        return "zero_contiguous"
    if domains <= {"zero_contiguous", "offset_contiguous"}:
        return "offset_contiguous"
    return "arbitrary"


def _position_relation(context: SequenceContext) -> str:
    """Classify the semantic relationship between valid query/key positions."""

    relations: set[str] = set()
    for query_positions, query_valid, key_positions, key_valid in zip(
        context.logical_positions,
        context.query_valid_mask,
        context.key_logical_positions,
        context.key_valid_mask,
        strict=True,
    ):
        query = query_positions[query_valid]
        key = key_positions[key_valid]
        if torch.equal(query, key):
            relations.add("equal")
        elif (
            query.numel() <= key.numel()
            and torch.equal(query, key[-query.numel() :])
        ):
            relations.add("query_suffix")
        else:
            relations.add("arbitrary")
    if relations == {"equal"}:
        return "equal"
    if relations <= {"equal", "query_suffix"}:
        return "query_suffix"
    return "arbitrary"


def request_from_context(
    context: SequenceContext,
    hidden_states: Tensor,
    *,
    mask_representation: str,
    visibility_family: str,
    cache_kind: str,
) -> SequenceRequest:
    """Describe one call without guessing facts erased by normalization."""

    if not isinstance(context, SequenceContext):
        raise TypeError("context must be a SequenceContext")
    if not isinstance(hidden_states, Tensor):
        raise TypeError("hidden_states must be a Tensor")
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, query, feature]"
        )
    if hidden_states.shape[:2] != (
        context.batch_size,
        context.query_length,
    ):
        raise ValueError(
            "hidden_states batch/query axes do not match the sequence context"
        )

    if context.phase == "decode":
        execution = "decode"
    elif (
        context.query_length != context.key_length
        or context.cache_state is not None
    ):
        execution = "chunked_prefill"
    else:
        execution = "prefill"

    if context.query_length == context.key_length:
        qk_relation = "equal"
    elif context.query_length < context.key_length:
        qk_relation = "query_le_key"
    else:
        qk_relation = "query_gt_key"

    active_cache_kind = (
        "none"
        if (
            context.cache_state is None
            and context.cache_positions is None
            and context.phase == "prefill"
        )
        else cache_kind
    )
    dtype = str(hidden_states.dtype).removeprefix("torch.")
    return SequenceRequest(
        query_length=context.query_length,
        key_length=context.key_length,
        execution=execution,
        qk_relation=qk_relation,
        position_relation=_position_relation(context),
        mask_origin=(
            "provided"
            if context.input_origin.attention_mask_supplied
            else "omitted"
        ),
        mask_pattern=_mask_pattern(context),
        mask_representation=mask_representation,
        visibility_family=visibility_family,
        position_origin=(
            "provided"
            if context.input_origin.position_ids_supplied
            else "omitted"
        ),
        position_domain=_position_domain(context),
        cache_kind=active_cache_kind,
        dtype=dtype,
        device=hidden_states.device.type,
        layout=("contiguous" if hidden_states.is_contiguous() else "strided"),
    )


def _combine_matches(
    matches: list[CapabilityMatch],
) -> CapabilityMatch:
    mismatches = tuple(
        reason
        for match in matches
        if match.status is MatchStatus.MISMATCH
        for reason in match.reasons
    )
    if mismatches:
        return CapabilityMatch(MatchStatus.MISMATCH, mismatches)
    unknowns = tuple(
        reason
        for match in matches
        if match.status is MatchStatus.UNKNOWN
        for reason in match.reasons
    )
    if unknowns:
        return CapabilityMatch(MatchStatus.UNKNOWN, unknowns)
    return CapabilityMatch(MatchStatus.MATCH)


def match_capabilities(
    capabilities: SequenceCapabilitySet,
    request: SequenceRequest,
) -> CapabilityMatch:
    if not isinstance(capabilities, SequenceCapabilitySet):
        raise TypeError("capabilities must be a SequenceCapabilitySet")
    if not isinstance(request, SequenceRequest):
        raise TypeError("request must be a SequenceRequest")
    matches: list[CapabilityMatch] = []
    for label, length in (
        ("query_length", request.query_length),
        ("key_length", request.key_length),
    ):
        if not capabilities.length.contains(length):
            matches.append(
                CapabilityMatch(
                    MatchStatus.MISMATCH,
                    (f"{label}: {length} is outside the compiled domain",),
                )
            )
    for field, capability, value in (
        ("execution", capabilities.executions, request.execution),
        ("qk_relation", capabilities.qk_relations, request.qk_relation),
        (
            "position_relation",
            capabilities.position_relations,
            request.position_relation,
        ),
        ("mask_origin", capabilities.mask_origins, request.mask_origin),
        ("mask_pattern", capabilities.mask_patterns, request.mask_pattern),
        (
            "mask_representation",
            capabilities.mask_representations,
            request.mask_representation,
        ),
        (
            "visibility_family",
            capabilities.visibility_families,
            request.visibility_family,
        ),
        (
            "position_origin",
            capabilities.position_origins,
            request.position_origin,
        ),
        (
            "position_domain",
            capabilities.position_domains,
            request.position_domain,
        ),
        ("cache_kind", capabilities.cache_kinds, request.cache_kind),
        ("dtype", capabilities.dtypes, request.dtype),
        ("device", capabilities.devices, request.device),
        ("layout", capabilities.layouts, request.layout),
    ):
        matches.append(capability.match(value, field=field))
    return _combine_matches(matches)


def capabilities_from_manifest_v1(
    sequence: ManifestSequenceSpec,
) -> SequenceCapabilitySet:
    """Convert only facts actually represented by the v1 JSON schema."""

    if not isinstance(sequence, ManifestSequenceSpec):
        raise TypeError("sequence must be a manifest SequenceSpec")
    mask_origins = {
        "unsupported": ("omitted",),
        "optional_all_true": ("omitted", "provided"),
        "optional": ("omitted", "provided"),
        "required": ("provided",),
    }[sequence.attention_mask]
    position_origins = {
        "unsupported": ("omitted",),
        "optional": ("omitted", "provided"),
        "required": ("provided",),
    }[sequence.position_ids]
    executions = {
        "none": ("prefill",),
        "prefill": ("prefill", "chunked_prefill"),
        "prefill_decode": ("prefill", "chunked_prefill", "decode"),
    }[sequence.cache]
    qk_relations = (
        ("equal",)
        if sequence.cache == "none"
        else ("equal", "query_le_key")
    )
    if (
        sequence.attention_mask in ("unsupported", "optional_all_true")
        and sequence.padding == "none"
    ):
        mask_patterns = CapabilityValues.known("all_valid")
    else:
        padding_patterns = {
            "none": ("all_valid",),
            "left": ("all_valid", "left_padded"),
            "right": ("all_valid", "right_padded"),
            "either": (
                "all_valid",
                "left_padded",
                "right_padded",
                "mixed_padded",
            ),
        }[sequence.padding]
        mask_patterns = CapabilityValues.known(*padding_patterns)
    return SequenceCapabilitySet(
        length=LengthDomain(
            sequence.minimum_length,
            sequence.maximum_length,
        ),
        executions=CapabilityValues.known(*executions),
        qk_relations=CapabilityValues.known(*qk_relations),
        position_relations=CapabilityValues.unknown(
            "manifest_v1_unexpressed:position_relation"
        ),
        mask_origins=CapabilityValues.known(*mask_origins),
        mask_patterns=mask_patterns,
        mask_representations=CapabilityValues.unknown(
            "manifest_v1_unexpressed:mask_representation"
        ),
        visibility_families=CapabilityValues.unknown(
            "manifest_v1_unexpressed:visibility_family"
        ),
        position_origins=CapabilityValues.known(*position_origins),
        position_domains=CapabilityValues.unknown(
            "manifest_v1_unexpressed:position_domain"
        ),
        cache_kinds=(
            CapabilityValues.known("none")
            if sequence.cache == "none"
            else CapabilityValues.unknown(
                "manifest_v1_unexpressed:cache_kind"
            )
        ),
        dtypes=CapabilityValues.unknown(
            "manifest_v1_unexpressed:dtype"
        ),
        devices=CapabilityValues.unknown(
            "manifest_v1_unexpressed:device"
        ),
        layouts=CapabilityValues.unknown(
            "manifest_v1_unexpressed:layout"
        ),
    )


def _overlay_values(
    base: CapabilityValues,
    overlay: CapabilityValues,
) -> CapabilityValues:
    if base.values is None:
        return overlay
    if overlay.values is None:
        return base
    return CapabilityValues(base.values.intersection(overlay.values))


def overlay_capabilities(
    base: SequenceCapabilitySet,
    overlay: SequenceCapabilitySet,
) -> SequenceCapabilitySet:
    """Fill unknown v1 facts while never widening a known manifest guard."""

    minimum = max(base.length.minimum, overlay.length.minimum)
    base_max = math.inf if base.length.maximum is None else base.length.maximum
    overlay_max = (
        math.inf if overlay.length.maximum is None else overlay.length.maximum
    )
    maximum_value = min(base_max, overlay_max)
    if maximum_value < minimum:
        raise ValueError("capability overlay has an empty length domain")
    maximum = None if math.isinf(maximum_value) else int(maximum_value)
    fields = {}
    for name in (
        "executions",
        "qk_relations",
        "position_relations",
        "mask_origins",
        "mask_patterns",
        "mask_representations",
        "visibility_families",
        "position_origins",
        "position_domains",
        "cache_kinds",
        "dtypes",
        "devices",
        "layouts",
    ):
        fields[name] = _overlay_values(
            getattr(base, name),
            getattr(overlay, name),
        )
    return SequenceCapabilitySet(
        length=LengthDomain(minimum, maximum),
        **fields,
    )


__all__ = [
    "CapabilityMatch",
    "CapabilityValues",
    "LengthDomain",
    "MatchStatus",
    "SequenceCapabilitySet",
    "SequenceRequest",
    "capabilities_from_manifest_v1",
    "match_capabilities",
    "overlay_capabilities",
    "request_from_context",
]
