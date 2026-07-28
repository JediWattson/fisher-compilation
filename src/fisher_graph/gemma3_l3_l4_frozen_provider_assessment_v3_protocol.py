"""Fresh V3 assessment specifications for the frozen Gemma L3->L4 provider.

This module is deliberately independent of the hash-bound V2 protocol.  It
contains specifications only: importing it cannot load Torch, a model, an
artifact, prompts, tokens, or measured activations.

The panel has three distinct scientific roles:

* new multitone and block-sparse probes test ordinary full-width fidelity;
* radial and signed blocks are declared sensitivity challenges; and
* exact-gain-null variants are declared invariance challenges.

Teacher eligibility and candidate fidelity are intentionally left to the
future assessment scorer.  This file only freezes the panel, its intended
contrasts, and the already-selected V2 rank-8 candidate identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal


__all__ = [
    "DEFAULT_V3_PANEL_SPEC_SHA256",
    "DEFAULT_V3_PROTOCOL_SHA256",
    "FrozenV2CandidateBinding",
    "V3AssessmentProtocol",
    "V3ContrastGroupSpec",
    "V3ProbeSpec",
    "V3SparseBlock",
    "default_v3_assessment_protocol",
    "v3_panel_spec_sha256",
]


_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol.v1"
)
_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MASK64 = (1 << 64) - 1
_PROBE_DOMAIN = b"fisher-graph:l3-l4-frozen-provider-v3-probe:v1\0"
_GROUP_DOMAIN = b"fisher-graph:l3-l4-frozen-provider-v3-group:v1\0"
_PROTOCOL_DOMAIN = b"fisher-graph:l3-l4-frozen-provider-v3-protocol:v1\0"
_PANEL_SPEC_DOMAIN = (
    b"fisher-graph:l3-l4-frozen-provider-v3-panel-spec:v1\0"
)
_DIRECTION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-frozen-v2-provider-assessment-v3\0"
)

ProbeFamily = Literal[
    "multitone",
    "block_sparse",
    "radial_block_sensitivity",
    "signed_block_sensitivity",
    "null_single_invariance",
]
ContrastIntent = Literal["sensitivity", "invariance"]
RankStratum = Literal["retained", "discarded"]

_FAMILIES = {
    "multitone",
    "block_sparse",
    "radial_block_sensitivity",
    "signed_block_sensitivity",
    "null_single_invariance",
}
_FIDELITY_FAMILIES = {"multitone", "block_sparse"}
_SENSITIVITY_FAMILIES = {
    "radial_block_sensitivity",
    "signed_block_sensitivity",
}
_INVARIANCE_FAMILIES = {"null_single_invariance"}

# Literal trust anchors for the complete declaration below.  Keeping both
# anchors makes accidental probe or grouping drift fail before a live teacher
# can be called.
DEFAULT_V3_PANEL_SPEC_SHA256 = (
    "919126906cc6f07074d76599843504ea81462485e8f93ee6d35c71732979249e"
)
DEFAULT_V3_PROTOCOL_SHA256 = (
    "65959324d2815621a1d6420bdb4d41a9db74c4214205088da9545088bc19ce03"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return value


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(value)}"
        )


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{label} must be finite")
    return float(value)


def _positive_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _derived_u64(*parts: object) -> int:
    digest = hashlib.sha256(_DIRECTION_DOMAIN)
    for part in parts:
        encoded = str(part).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "little") & _MASK64


def _derived_distinct_modes(
    *,
    family: str,
    ordinal: int,
    block_index: int,
    count: int,
) -> tuple[int, ...]:
    result: list[int] = []
    cursor = 0
    while len(result) < count:
        candidate = int(
            _derived_u64(family, ordinal, block_index, "mode", cursor)
            % _MODAL_WIDTH
        )
        cursor += 1
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class V3SparseBlock:
    """One contiguous block with four fixed signed modal coordinates."""

    start: int
    length: int
    mode_signs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        start = _exact_int(self.start, label="sparse block start")
        length = _exact_int(
            self.length,
            label="sparse block length",
            minimum=1,
        )
        if type(self.mode_signs) is not tuple or len(self.mode_signs) != 4:
            raise ValueError("sparse block must contain exactly four modes")
        modes: list[int] = []
        for value in self.mode_signs:
            if (
                type(value) is not tuple
                or len(value) != 2
                or type(value[0]) is not int
                or not 0 <= value[0] < _MODAL_WIDTH
                or value[1] not in (-1, 1)
            ):
                raise ValueError("sparse block mode/sign entry is invalid")
            modes.append(value[0])
        if len(set(modes)) != len(modes):
            raise ValueError("sparse block modes must be unique")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "length", length)

    def state_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "length": self.length,
            "mode_signs": [list(value) for value in self.mode_signs],
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "V3SparseBlock":
        state = _mapping(raw, label="V3 sparse block")
        _strict_keys(
            state,
            expected={"start", "length", "mode_signs"},
            label="V3 sparse block",
        )
        entries = _sequence(
            state["mode_signs"],
            label="V3 sparse block mode signs",
        )
        parsed: list[tuple[int, int]] = []
        for index, raw_entry in enumerate(entries):
            entry = _sequence(
                raw_entry,
                label=f"V3 sparse block mode sign {index}",
            )
            if len(entry) != 2:
                raise ValueError("V3 sparse block mode sign must have length two")
            parsed.append(
                (
                    _exact_int(
                        entry[0],
                        label=f"V3 sparse block mode {index}",
                    ),
                    (
                        _exact_int(
                            entry[1],
                            label=f"V3 sparse block sign {index}",
                        )
                        if entry[1] == 1
                        else (
                            -1
                            if type(entry[1]) is int and entry[1] == -1
                            else 0
                        )
                    ),
                )
            )
        return cls(
            start=_exact_int(state["start"], label="V3 sparse block start"),
            length=_exact_int(
                state["length"],
                label="V3 sparse block length",
                minimum=1,
            ),
            mode_signs=tuple(parsed),
        )


@dataclass(frozen=True, slots=True)
class V3ProbeSpec:
    """One immutable synthetic assessment probe."""

    probe_id: str
    ordinal: int
    family: ProbeFamily
    sequence_length: int
    source_offset: int
    modal_amplitude: float
    radial_scale: float
    null_coordinate: float
    direction_seed: int
    active_block_length: int
    axis_mode: int | None = None
    axis_sign: int | None = None
    multitone_temporal_frequencies: tuple[int, int] | None = None
    multitone_modal_frequencies: tuple[int, int] | None = None
    multitone_phase_quadrants: tuple[int, int] | None = None
    sparse_blocks: tuple[V3SparseBlock, ...] = ()
    contrast_group: str | None = None
    contrast_intent: ContrastIntent | None = None
    contrast_variant: str | None = None
    rank_stratum: RankStratum | None = None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        probe_id = _nonempty_string(self.probe_id, label="V3 probe id")
        if not probe_id.startswith("assessment_v3."):
            raise ValueError("V3 probe id must use the assessment_v3 namespace")
        ordinal = _exact_int(self.ordinal, label="V3 probe ordinal")
        if self.family not in _FAMILIES:
            raise ValueError("V3 probe family is invalid")
        length = _exact_int(
            self.sequence_length,
            label="V3 sequence length",
            minimum=1,
        )
        offset = _exact_int(self.source_offset, label="V3 source offset")
        if offset >= length:
            raise ValueError("V3 source offset must lie inside the sequence")
        amplitude = _positive_float(
            self.modal_amplitude,
            label="V3 modal amplitude",
        )
        radial = _positive_float(self.radial_scale, label="V3 radial scale")
        null = _finite_float(
            self.null_coordinate,
            label="V3 null coordinate",
        )
        seed = _exact_int(self.direction_seed, label="V3 direction seed")
        if seed > _MASK64:
            raise ValueError("V3 direction seed exceeds uint64")
        block_length = _exact_int(
            self.active_block_length,
            label="V3 active block length",
            minimum=1,
        )
        if offset + block_length > length:
            raise ValueError("V3 active block exceeds the sequence")

        axis_fields = (self.axis_mode, self.axis_sign)
        has_axis = any(value is not None for value in axis_fields)
        if has_axis:
            if (
                type(self.axis_mode) is not int
                or not 0 <= self.axis_mode < _MODAL_WIDTH
                or self.axis_sign not in (-1, 1)
            ):
                raise ValueError("V3 axis coordinate is invalid")

        multitone_fields = (
            self.multitone_temporal_frequencies,
            self.multitone_modal_frequencies,
            self.multitone_phase_quadrants,
        )
        has_multitone = any(value is not None for value in multitone_fields)
        if has_multitone:
            temporal, modal, phase = multitone_fields
            if (
                type(temporal) is not tuple
                or type(modal) is not tuple
                or type(phase) is not tuple
                or len(temporal) != 2
                or len(modal) != 2
                or len(phase) != 2
                or any(type(value) is not int or value <= 0 for value in temporal)
                or any(
                    type(value) is not int
                    or not 0 <= value < _MODAL_WIDTH
                    for value in modal
                )
                or any(value not in (0, 1, 2, 3) for value in phase)
                or temporal[0] == temporal[1]
                or modal[0] == modal[1]
            ):
                raise ValueError("V3 multitone parameters are invalid")

        if type(self.sparse_blocks) is not tuple or any(
            not isinstance(value, V3SparseBlock)
            for value in self.sparse_blocks
        ):
            raise TypeError("V3 sparse blocks must be a tuple of V3SparseBlock")
        for block in self.sparse_blocks:
            if (
                block.start < offset
                or block.start + block.length > length
            ):
                raise ValueError("V3 sparse block lies outside active sequence")
        ordered_blocks = tuple(
            sorted(self.sparse_blocks, key=lambda value: value.start)
        )
        if ordered_blocks != self.sparse_blocks:
            raise ValueError("V3 sparse blocks must be sorted")
        for left, right in zip(ordered_blocks, ordered_blocks[1:]):
            if left.start + left.length > right.start:
                raise ValueError("V3 sparse blocks must not overlap")

        contrast_fields = (
            self.contrast_group,
            self.contrast_intent,
            self.contrast_variant,
            self.rank_stratum,
        )
        has_contrast = any(value is not None for value in contrast_fields)
        if has_contrast and any(value is None for value in contrast_fields):
            raise ValueError("V3 contrast metadata must be all present or absent")
        if has_contrast:
            _nonempty_string(self.contrast_group, label="V3 contrast group")
            _nonempty_string(self.contrast_variant, label="V3 contrast variant")
            if self.contrast_intent not in {"sensitivity", "invariance"}:
                raise ValueError("V3 contrast intent is invalid")
            if self.rank_stratum not in {"retained", "discarded"}:
                raise ValueError("V3 rank stratum is invalid")

        if self.family == "multitone":
            if (
                not has_multitone
                or has_axis
                or self.sparse_blocks
                or has_contrast
                or block_length != length - offset
            ):
                raise ValueError("V3 multitone probe fields are inconsistent")
        elif self.family == "block_sparse":
            if (
                has_multitone
                or has_axis
                or len(self.sparse_blocks) != 2
                or has_contrast
            ):
                raise ValueError("V3 block-sparse probe fields are inconsistent")
        elif self.family in _SENSITIVITY_FAMILIES:
            if (
                has_multitone
                or self.sparse_blocks
                or not has_axis
                or not has_contrast
                or self.contrast_intent != "sensitivity"
            ):
                raise ValueError("V3 sensitivity probe fields are inconsistent")
        elif self.family in _INVARIANCE_FAMILIES:
            if (
                has_multitone
                or self.sparse_blocks
                or not has_axis
                or not has_contrast
                or self.contrast_intent != "invariance"
                or block_length != 1
            ):
                raise ValueError("V3 invariance probe fields are inconsistent")

        object.__setattr__(self, "probe_id", probe_id)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "sequence_length", length)
        object.__setattr__(self, "source_offset", offset)
        object.__setattr__(self, "modal_amplitude", amplitude)
        object.__setattr__(self, "radial_scale", radial)
        object.__setattr__(self, "null_coordinate", null)
        object.__setattr__(self, "direction_seed", seed)
        object.__setattr__(self, "active_block_length", block_length)
        computed = _digest(self._payload(), domain=_PROBE_DOMAIN)
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="V3 probe artifact SHA",
            ) != computed:
                raise ValueError("V3 probe artifact hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.v3_assessment_probe.v1",
            "format_version": _FORMAT_VERSION,
            "probe_id": self.probe_id,
            "split": "assessment",
            "ordinal": self.ordinal,
            "family": self.family,
            "sequence_length": self.sequence_length,
            "source_offset": self.source_offset,
            "modal_amplitude": self.modal_amplitude,
            "radial_scale": self.radial_scale,
            "null_coordinate": self.null_coordinate,
            "direction_seed": self.direction_seed,
            "active_block_length": self.active_block_length,
            "axis_mode": self.axis_mode,
            "axis_sign": self.axis_sign,
            "multitone_temporal_frequencies": (
                None
                if self.multitone_temporal_frequencies is None
                else list(self.multitone_temporal_frequencies)
            ),
            "multitone_modal_frequencies": (
                None
                if self.multitone_modal_frequencies is None
                else list(self.multitone_modal_frequencies)
            ),
            "multitone_phase_quadrants": (
                None
                if self.multitone_phase_quadrants is None
                else list(self.multitone_phase_quadrants)
            ),
            "sparse_blocks": [
                value.state_dict() for value in self.sparse_blocks
            ],
            "contrast_group": self.contrast_group,
            "contrast_intent": self.contrast_intent,
            "contrast_variant": self.contrast_variant,
            "rank_stratum": self.rank_stratum,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "V3ProbeSpec":
        state = _mapping(raw, label="V3 probe")
        expected = {
            "schema",
            "format_version",
            "probe_id",
            "split",
            "ordinal",
            "family",
            "sequence_length",
            "source_offset",
            "modal_amplitude",
            "radial_scale",
            "null_coordinate",
            "direction_seed",
            "active_block_length",
            "axis_mode",
            "axis_sign",
            "multitone_temporal_frequencies",
            "multitone_modal_frequencies",
            "multitone_phase_quadrants",
            "sparse_blocks",
            "contrast_group",
            "contrast_intent",
            "contrast_variant",
            "rank_stratum",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="V3 probe")
        if (
            state["schema"] != "fisher_graph.v3_assessment_probe.v1"
            or state["format_version"] != _FORMAT_VERSION
            or state["split"] != "assessment"
        ):
            raise ValueError("V3 probe schema, version, or split is invalid")

        def optional_pair(name: str) -> tuple[int, int] | None:
            raw_value = state[name]
            if raw_value is None:
                return None
            values = _sequence(raw_value, label=f"V3 probe {name}")
            if len(values) != 2:
                raise ValueError(f"V3 probe {name} must have length two")
            return (
                _exact_int(values[0], label=f"V3 probe {name}[0]"),
                _exact_int(values[1], label=f"V3 probe {name}[1]"),
            )

        blocks = _sequence(state["sparse_blocks"], label="V3 sparse blocks")
        return cls(
            probe_id=_nonempty_string(state["probe_id"], label="V3 probe id"),
            ordinal=_exact_int(state["ordinal"], label="V3 probe ordinal"),
            family=str(state["family"]),  # type: ignore[arg-type]
            sequence_length=_exact_int(
                state["sequence_length"],
                label="V3 sequence length",
                minimum=1,
            ),
            source_offset=_exact_int(
                state["source_offset"],
                label="V3 source offset",
            ),
            modal_amplitude=_positive_float(
                state["modal_amplitude"],
                label="V3 modal amplitude",
            ),
            radial_scale=_positive_float(
                state["radial_scale"],
                label="V3 radial scale",
            ),
            null_coordinate=_finite_float(
                state["null_coordinate"],
                label="V3 null coordinate",
            ),
            direction_seed=_exact_int(
                state["direction_seed"],
                label="V3 direction seed",
            ),
            active_block_length=_exact_int(
                state["active_block_length"],
                label="V3 active block length",
                minimum=1,
            ),
            axis_mode=(
                None
                if state["axis_mode"] is None
                else _exact_int(state["axis_mode"], label="V3 axis mode")
            ),
            axis_sign=(
                None
                if state["axis_sign"] is None
                else int(state["axis_sign"])
            ),
            multitone_temporal_frequencies=optional_pair(
                "multitone_temporal_frequencies"
            ),
            multitone_modal_frequencies=optional_pair(
                "multitone_modal_frequencies"
            ),
            multitone_phase_quadrants=optional_pair(
                "multitone_phase_quadrants"
            ),
            sparse_blocks=tuple(
                V3SparseBlock.from_state_dict(value) for value in blocks
            ),
            contrast_group=(
                None
                if state["contrast_group"] is None
                else str(state["contrast_group"])
            ),
            contrast_intent=(
                None
                if state["contrast_intent"] is None
                else str(state["contrast_intent"])
            ),  # type: ignore[arg-type]
            contrast_variant=(
                None
                if state["contrast_variant"] is None
                else str(state["contrast_variant"])
            ),
            rank_stratum=(
                None
                if state["rank_stratum"] is None
                else str(state["rank_stratum"])
            ),  # type: ignore[arg-type]
            artifact_sha256=_require_sha256(
                state["artifact_sha256"],
                label="V3 probe artifact SHA",
            ),
        )


@dataclass(frozen=True, slots=True)
class V3ContrastGroupSpec:
    """Frozen variants and canonical comparisons for one contrast group."""

    group_id: str
    family: ProbeFamily
    intent: ContrastIntent
    rank_stratum: RankStratum
    variant_probe_ids: tuple[str, ...]
    canonical_variant_pairs: tuple[tuple[str, str], ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        group = _nonempty_string(self.group_id, label="V3 contrast group")
        if not group.startswith("assessment_v3."):
            raise ValueError("V3 contrast group must use assessment_v3 namespace")
        if self.family not in _SENSITIVITY_FAMILIES | _INVARIANCE_FAMILIES:
            raise ValueError("V3 contrast group family is invalid")
        if self.intent not in {"sensitivity", "invariance"}:
            raise ValueError("V3 contrast group intent is invalid")
        if (
            self.intent == "sensitivity"
            and self.family not in _SENSITIVITY_FAMILIES
        ) or (
            self.intent == "invariance"
            and self.family not in _INVARIANCE_FAMILIES
        ):
            raise ValueError("V3 contrast group family and intent disagree")
        if self.rank_stratum not in {"retained", "discarded"}:
            raise ValueError("V3 contrast rank stratum is invalid")
        if (
            type(self.variant_probe_ids) is not tuple
            or len(self.variant_probe_ids) < 2
            or len(set(self.variant_probe_ids)) != len(self.variant_probe_ids)
            or any(not isinstance(value, str) or not value for value in self.variant_probe_ids)
        ):
            raise ValueError("V3 contrast variants are invalid")
        if (
            type(self.canonical_variant_pairs) is not tuple
            or not self.canonical_variant_pairs
        ):
            raise ValueError("V3 canonical variant pairs cannot be empty")
        seen_pairs: set[tuple[str, str]] = set()
        variants = set(self.variant_probe_ids)
        for pair in self.canonical_variant_pairs:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or pair[0] == pair[1]
                or pair[0] not in variants
                or pair[1] not in variants
                or pair in seen_pairs
            ):
                raise ValueError("V3 canonical variant pair is invalid")
            seen_pairs.add(pair)
        expected_pair_count = (
            len(self.variant_probe_ids) * (len(self.variant_probe_ids) - 1) // 2
        )
        if self.intent == "invariance" and len(seen_pairs) != expected_pair_count:
            raise ValueError("V3 invariance group must compare every pair")
        computed = _digest(self._payload(), domain=_GROUP_DOMAIN)
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="V3 contrast group SHA",
            ) != computed:
                raise ValueError("V3 contrast group hash mismatch")
        object.__setattr__(self, "group_id", group)
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.v3_contrast_group.v1",
            "format_version": _FORMAT_VERSION,
            "group_id": self.group_id,
            "family": self.family,
            "intent": self.intent,
            "rank_stratum": self.rank_stratum,
            "variant_probe_ids": list(self.variant_probe_ids),
            "canonical_variant_pairs": [
                list(value) for value in self.canonical_variant_pairs
            ],
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "V3ContrastGroupSpec":
        state = _mapping(raw, label="V3 contrast group")
        _strict_keys(
            state,
            expected={
                "schema",
                "format_version",
                "group_id",
                "family",
                "intent",
                "rank_stratum",
                "variant_probe_ids",
                "canonical_variant_pairs",
                "artifact_sha256",
            },
            label="V3 contrast group",
        )
        if (
            state["schema"] != "fisher_graph.v3_contrast_group.v1"
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("V3 contrast group schema or version is invalid")
        variants = tuple(
            _nonempty_string(value, label="V3 contrast variant id")
            for value in _sequence(
                state["variant_probe_ids"],
                label="V3 contrast variant ids",
            )
        )
        raw_pairs = _sequence(
            state["canonical_variant_pairs"],
            label="V3 canonical variant pairs",
        )
        pairs: list[tuple[str, str]] = []
        for raw_pair in raw_pairs:
            pair = _sequence(raw_pair, label="V3 canonical variant pair")
            if len(pair) != 2:
                raise ValueError("V3 canonical variant pair must have length two")
            pairs.append(
                (
                    _nonempty_string(pair[0], label="V3 pair left"),
                    _nonempty_string(pair[1], label="V3 pair right"),
                )
            )
        return cls(
            group_id=_nonempty_string(
                state["group_id"],
                label="V3 contrast group",
            ),
            family=str(state["family"]),  # type: ignore[arg-type]
            intent=str(state["intent"]),  # type: ignore[arg-type]
            rank_stratum=str(state["rank_stratum"]),  # type: ignore[arg-type]
            variant_probe_ids=variants,
            canonical_variant_pairs=tuple(pairs),
            artifact_sha256=_require_sha256(
                state["artifact_sha256"],
                label="V3 contrast group SHA",
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenV2CandidateBinding:
    """Literal identity of the already-selected V2 rank-8 provider."""

    artifact_sha256: str = (
        "973bab7c72d456247a535137fd3bbfa8fd064b4710718dc905dea94963144f46"
    )
    file_sha256: str = (
        "37bd6fbda9b3660777f0388561e4e8d7d1a28e3958bcb98c69ca302cd1f77ae1"
    )
    report_sha256: str = (
        "1e14518f915821aa7448b6f4799e322e2451074b3030ba4107c6a2a0924be4d9"
    )
    candidate_id: str = "spectral-r08-t08"
    source_rank: int = 8
    target_rank: int = 8
    stored_scalar_count: int = 910
    selected_plan_sha256: str = (
        "7ab42890daece95eeedbf08ba0e5727f2bccfd7be20e00a4e404539cd1bf9cee"
    )
    selection_sha256: str = (
        "f378ceeabc45a1f084687ce6a7444db60620c71d1e5a141f29563dd75b92ce4c"
    )
    controls_sha256: str = (
        "8d08b222f73b17715965cadf8cd0d19c5fe37abb09a1e578351e6f85e98a24d6"
    )
    standardized_gauge_sha256: str = (
        "5b6d48d48e5b4aab8306b542b6d85c498052373adb145cc6aa82199fd36a59e4"
    )
    metric_weight_sha256: str = (
        "8f223ba744cdcbd7903766517b4fc27a3c27db2d3ca192450b06f9ae7406a75e"
    )
    training_protocol_sha256: str = (
        "4eb3bc860539683802355bd156dd59ff6007e4de86c1b98558f51d45b798fbaf"
    )
    basis_payload_sha256: str = (
        "b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01"
    )
    source_model_sha256: str = (
        "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
    )

    def __post_init__(self) -> None:
        for name in (
            "artifact_sha256",
            "file_sha256",
            "report_sha256",
            "selected_plan_sha256",
            "selection_sha256",
            "controls_sha256",
            "standardized_gauge_sha256",
            "metric_weight_sha256",
            "training_protocol_sha256",
            "basis_payload_sha256",
            "source_model_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"frozen provider {name}")
        if self.candidate_id != "spectral-r08-t08":
            raise ValueError("frozen V2 candidate id drifted")
        if (
            type(self.source_rank) is not int
            or type(self.target_rank) is not int
            or type(self.stored_scalar_count) is not int
            or
            self.source_rank != 8
            or self.target_rank != 8
            or self.stored_scalar_count != 910
        ):
            raise ValueError("frozen V2 candidate geometry drifted")

    def state_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "artifact_sha256",
                "file_sha256",
                "report_sha256",
                "candidate_id",
                "source_rank",
                "target_rank",
                "stored_scalar_count",
                "selected_plan_sha256",
                "selection_sha256",
                "controls_sha256",
                "standardized_gauge_sha256",
                "metric_weight_sha256",
                "training_protocol_sha256",
                "basis_payload_sha256",
                "source_model_sha256",
            )
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "FrozenV2CandidateBinding":
        state = _mapping(raw, label="frozen V2 provider binding")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="frozen V2 provider binding",
        )
        result = cls(
            **{
                name: state[name]
                for name in expected
            }
        )
        if result != cls():
            raise ValueError("frozen V2 provider binding differs from trust anchor")
        return result


def v3_panel_spec_sha256(
    ordered_probe_sha256s: Sequence[str],
) -> str:
    hashes = tuple(
        _require_sha256(value, label="V3 panel probe SHA")
        for value in ordered_probe_sha256s
    )
    if len(hashes) != 48 or len(set(hashes)) != len(hashes):
        raise ValueError("V3 panel must bind 48 unique probe hashes")
    return _digest(
        {
            "schema": "fisher_graph.v3_assessment_panel_spec.v1",
            "format_version": _FORMAT_VERSION,
            "ordered_probe_sha256s": list(hashes),
        },
        domain=_PANEL_SPEC_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class V3AssessmentProtocol:
    """Complete authenticated declaration of the fresh V3 panel."""

    frozen_candidate: FrozenV2CandidateBinding
    probes: tuple[V3ProbeSpec, ...]
    contrast_groups: tuple[V3ContrastGroupSpec, ...]
    protocol_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_candidate, FrozenV2CandidateBinding):
            raise TypeError("V3 frozen candidate binding is invalid")
        if (
            type(self.probes) is not tuple
            or len(self.probes) != 48
            or any(not isinstance(value, V3ProbeSpec) for value in self.probes)
        ):
            raise ValueError("V3 protocol must contain 48 probes")
        if tuple(value.ordinal for value in self.probes) != tuple(range(48)):
            raise ValueError("V3 probe ordinals must be contiguous and ordered")
        if len({value.probe_id for value in self.probes}) != 48 or len(
            {value.artifact_sha256 for value in self.probes}
        ) != 48:
            raise ValueError("V3 probe identities must be unique")
        family_counts = {
            family: sum(value.family == family for value in self.probes)
            for family in _FAMILIES
        }
        if family_counts != {
            "multitone": 8,
            "block_sparse": 8,
            "radial_block_sensitivity": 12,
            "signed_block_sensitivity": 8,
            "null_single_invariance": 12,
        }:
            raise ValueError("V3 probe family counts drifted")
        if (
            type(self.contrast_groups) is not tuple
            or len(self.contrast_groups) != 12
            or any(
                not isinstance(value, V3ContrastGroupSpec)
                for value in self.contrast_groups
            )
            or tuple(
                sorted(
                    self.contrast_groups,
                    key=lambda value: value.group_id,
                )
            )
            != self.contrast_groups
        ):
            raise ValueError("V3 contrast groups must be 12 sorted groups")
        if len({value.group_id for value in self.contrast_groups}) != 12:
            raise ValueError("V3 contrast groups must be unique")

        probes_by_id = {value.probe_id: value for value in self.probes}
        grouped_ids: set[str] = set()
        for group in self.contrast_groups:
            for probe_id in group.variant_probe_ids:
                probe = probes_by_id.get(probe_id)
                if (
                    probe is None
                    or probe.contrast_group != group.group_id
                    or probe.contrast_intent != group.intent
                    or probe.rank_stratum != group.rank_stratum
                    or probe.family != group.family
                    or probe_id in grouped_ids
                ):
                    raise ValueError(
                        "V3 contrast group differs from its probe declarations"
                    )
                grouped_ids.add(probe_id)
        declared_group_ids = {
            value.probe_id
            for value in self.probes
            if value.contrast_group is not None
        }
        if grouped_ids != declared_group_ids:
            raise ValueError("V3 contrast probes are not grouped exactly once")

        panel_sha = self.panel_spec_sha256
        if (
            DEFAULT_V3_PANEL_SPEC_SHA256
            and panel_sha != DEFAULT_V3_PANEL_SPEC_SHA256
        ):
            raise ValueError("V3 assessment panel specification drifted")
        computed = _digest(self._payload(), domain=_PROTOCOL_DOMAIN)
        if self.protocol_sha256:
            if _require_sha256(
                self.protocol_sha256,
                label="V3 protocol SHA",
            ) != computed:
                raise ValueError("V3 assessment protocol hash mismatch")
        if DEFAULT_V3_PROTOCOL_SHA256 and computed != DEFAULT_V3_PROTOCOL_SHA256:
            raise ValueError("V3 assessment protocol differs from trust anchor")
        object.__setattr__(self, "protocol_sha256", computed)

    @property
    def panel_spec_sha256(self) -> str:
        return v3_panel_spec_sha256(
            tuple(value.artifact_sha256 for value in self.probes)
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scope": "fresh_assessment_only_for_exact_frozen_v2_rank8_provider",
            "assessment_may_fit": False,
            "assessment_may_select": False,
            "assessment_may_refit": False,
            "prompt_blind_after_frozen_prompt_conditioned_basis": True,
            "natural_prompt_claim": False,
            "whole_model_claim": False,
            "panel_spec_sha256": self.panel_spec_sha256,
            "frozen_candidate": self.frozen_candidate.state_dict(),
            "probes": [value.state_dict() for value in self.probes],
            "contrast_groups": [
                value.state_dict() for value in self.contrast_groups
            ],
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "protocol_sha256": self.protocol_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "V3AssessmentProtocol":
        state = _mapping(raw, label="V3 assessment protocol")
        _strict_keys(
            state,
            expected={
                "schema",
                "format_version",
                "scope",
                "assessment_may_fit",
                "assessment_may_select",
                "assessment_may_refit",
                "prompt_blind_after_frozen_prompt_conditioned_basis",
                "natural_prompt_claim",
                "whole_model_claim",
                "panel_spec_sha256",
                "frozen_candidate",
                "probes",
                "contrast_groups",
                "protocol_sha256",
            },
            label="V3 assessment protocol",
        )
        if (
            state["schema"] != _SCHEMA
            or state["format_version"] != _FORMAT_VERSION
            or state["scope"]
            != "fresh_assessment_only_for_exact_frozen_v2_rank8_provider"
            or state["assessment_may_fit"] is not False
            or state["assessment_may_select"] is not False
            or state["assessment_may_refit"] is not False
            or state["prompt_blind_after_frozen_prompt_conditioned_basis"]
            is not True
            or state["natural_prompt_claim"] is not False
            or state["whole_model_claim"] is not False
        ):
            raise ValueError("V3 protocol declaration is invalid")
        probes = tuple(
            V3ProbeSpec.from_state_dict(value)
            for value in _sequence(state["probes"], label="V3 protocol probes")
        )
        groups = tuple(
            V3ContrastGroupSpec.from_state_dict(value)
            for value in _sequence(
                state["contrast_groups"],
                label="V3 protocol contrast groups",
            )
        )
        result = cls(
            frozen_candidate=FrozenV2CandidateBinding.from_state_dict(
                state["frozen_candidate"]
            ),
            probes=probes,
            contrast_groups=groups,
            protocol_sha256=_require_sha256(
                state["protocol_sha256"],
                label="V3 protocol SHA",
            ),
        )
        if _require_sha256(
            state["panel_spec_sha256"],
            label="V3 panel spec SHA",
        ) != result.panel_spec_sha256:
            raise ValueError("V3 panel spec hash mismatch")
        return result


def _multitone_probe(
    *,
    ordinal: int,
    index: int,
    length: int,
    offset: int,
) -> V3ProbeSpec:
    active_length = length - offset
    temporal_first = 1 + int(
        _derived_u64("multitone", index, "temporal", 0)
        % max(2, active_length // 4)
    )
    temporal_second = 1 + int(
        _derived_u64("multitone", index, "temporal", 1)
        % max(2, active_length // 4)
    )
    if temporal_second == temporal_first:
        temporal_second = 1 + temporal_second % max(2, active_length // 4)
    modal_first = int(
        _derived_u64("multitone", index, "modal", 0) % _MODAL_WIDTH
    )
    modal_second = int(
        _derived_u64("multitone", index, "modal", 1) % _MODAL_WIDTH
    )
    if modal_second == modal_first:
        modal_second = (modal_second + 1) % _MODAL_WIDTH
    amplitudes = (0.125, 0.375, 0.625, 0.875)
    radials = (0.625, 1.125, 1.875, 1.125, 0.625, 1.875, 1.125, 0.625)
    nulls = (-0.75, 0.0, 0.75, 0.0, -0.75, 0.75, 0.0, -0.75)
    return V3ProbeSpec(
        probe_id=f"assessment_v3.fidelity.multitone.{index:02d}",
        ordinal=ordinal,
        family="multitone",
        sequence_length=length,
        source_offset=offset,
        modal_amplitude=amplitudes[index % len(amplitudes)],
        radial_scale=radials[index],
        null_coordinate=nulls[index],
        direction_seed=_derived_u64("multitone", index, "direction"),
        active_block_length=active_length,
        multitone_temporal_frequencies=(
            temporal_first,
            temporal_second,
        ),
        multitone_modal_frequencies=(modal_first, modal_second),
        multitone_phase_quadrants=(
            int(_derived_u64("multitone", index, "phase", 0) % 4),
            int(_derived_u64("multitone", index, "phase", 1) % 4),
        ),
    )


def _block_sparse_probe(
    *,
    ordinal: int,
    index: int,
    length: int,
    offset: int,
) -> V3ProbeSpec:
    suffix = length - offset
    block_length = max(2, suffix // 8)
    second_start = offset + suffix // 2
    if second_start + block_length > length:
        second_start = length - block_length
    blocks: list[V3SparseBlock] = []
    for block_index, start in enumerate((offset, second_start)):
        modes = _derived_distinct_modes(
            family="block_sparse",
            ordinal=index,
            block_index=block_index,
            count=4,
        )
        signs = tuple(
            1
            if _derived_u64(
                "block_sparse",
                index,
                block_index,
                "sign",
                position,
            )
            & 1
            else -1
            for position in range(4)
        )
        blocks.append(
            V3SparseBlock(
                start=start,
                length=block_length,
                mode_signs=tuple(zip(modes, signs, strict=True)),
            )
        )
    amplitudes = (0.125, 0.375, 0.625, 0.875)
    radials = (1.125, 0.625, 1.875, 0.625, 1.125, 1.875, 0.625, 1.125)
    nulls = (0.0, 0.75, -0.75, 0.0, 0.75, -0.75, 0.0, 0.75)
    return V3ProbeSpec(
        probe_id=f"assessment_v3.fidelity.block_sparse.{index:02d}",
        ordinal=ordinal,
        family="block_sparse",
        sequence_length=length,
        source_offset=offset,
        modal_amplitude=amplitudes[(index + 1) % len(amplitudes)],
        radial_scale=radials[index],
        null_coordinate=nulls[index],
        direction_seed=_derived_u64("block_sparse", index, "direction"),
        active_block_length=length - offset,
        sparse_blocks=tuple(blocks),
    )


def _build_protocol() -> V3AssessmentProtocol:
    probes: list[V3ProbeSpec] = []
    groups: list[V3ContrastGroupSpec] = []
    lengths = (48, 80, 112, 144, 176, 208, 240, 256)
    offsets = (6, 20, 42, 72, 44, 104, 90, 128)
    for index, (length, offset) in enumerate(
        zip(lengths, offsets, strict=True)
    ):
        probes.append(
            _multitone_probe(
                ordinal=len(probes),
                index=index,
                length=length,
                offset=offset,
            )
        )
    for index, (length, offset) in enumerate(
        zip(lengths, offsets, strict=True)
    ):
        probes.append(
            _block_sparse_probe(
                ordinal=len(probes),
                index=index,
                length=length,
                offset=offset,
            )
        )

    bases = (
        ("b0", 3, "retained", 40, 5, 5),
        ("b1", 6, "retained", 88, 22, 7),
        ("b2", 19, "discarded", 152, 57, 9),
        ("b3", 37, "discarded", 232, 116, 11),
    )
    radial_variants = (
        ("radius_0p625", 0.625),
        ("radius_1p125", 1.125),
        ("radius_1p875", 1.875),
    )
    for base, mode, stratum, length, offset, block_length in bases:
        group_id = f"assessment_v3.sensitivity.radial.{base}"
        variant_ids: list[str] = []
        for variant, radial in radial_variants:
            probe_id = f"{group_id}.{variant}"
            variant_ids.append(probe_id)
            probes.append(
                V3ProbeSpec(
                    probe_id=probe_id,
                    ordinal=len(probes),
                    family="radial_block_sensitivity",
                    sequence_length=length,
                    source_offset=offset,
                    modal_amplitude=0.5,
                    radial_scale=radial,
                    null_coordinate=0.0,
                    direction_seed=_derived_u64(group_id, "direction"),
                    active_block_length=block_length,
                    axis_mode=mode,
                    axis_sign=1,
                    contrast_group=group_id,
                    contrast_intent="sensitivity",
                    contrast_variant=variant,
                    rank_stratum=stratum,  # type: ignore[arg-type]
                )
            )
        groups.append(
            V3ContrastGroupSpec(
                group_id=group_id,
                family="radial_block_sensitivity",
                intent="sensitivity",
                rank_stratum=stratum,  # type: ignore[arg-type]
                variant_probe_ids=tuple(variant_ids),
                canonical_variant_pairs=(
                    (variant_ids[0], variant_ids[1]),
                    (variant_ids[1], variant_ids[2]),
                ),
            )
        )

    for base, mode, stratum, length, offset, block_length in bases:
        group_id = f"assessment_v3.sensitivity.sign_block.{base}"
        variant_ids: list[str] = []
        for variant, sign in (("negative", -1), ("positive", 1)):
            probe_id = f"{group_id}.{variant}"
            variant_ids.append(probe_id)
            probes.append(
                V3ProbeSpec(
                    probe_id=probe_id,
                    ordinal=len(probes),
                    family="signed_block_sensitivity",
                    sequence_length=length,
                    source_offset=offset,
                    modal_amplitude=0.75,
                    radial_scale=1.25,
                    null_coordinate=0.0,
                    direction_seed=_derived_u64(group_id, "direction"),
                    active_block_length=block_length,
                    axis_mode=mode,
                    axis_sign=sign,
                    contrast_group=group_id,
                    contrast_intent="sensitivity",
                    contrast_variant=variant,
                    rank_stratum=stratum,  # type: ignore[arg-type]
                )
            )
        groups.append(
            V3ContrastGroupSpec(
                group_id=group_id,
                family="signed_block_sensitivity",
                intent="sensitivity",
                rank_stratum=stratum,  # type: ignore[arg-type]
                variant_probe_ids=tuple(variant_ids),
                canonical_variant_pairs=((variant_ids[0], variant_ids[1]),),
            )
        )

    null_variants = (
        ("null_m0p75", -0.75),
        ("null_0", 0.0),
        ("null_p0p75", 0.75),
    )
    for base, mode, stratum, length, offset, _block_length in bases:
        group_id = f"assessment_v3.invariance.null.{base}"
        variant_ids: list[str] = []
        for variant, null in null_variants:
            probe_id = f"{group_id}.{variant}"
            variant_ids.append(probe_id)
            probes.append(
                V3ProbeSpec(
                    probe_id=probe_id,
                    ordinal=len(probes),
                    family="null_single_invariance",
                    sequence_length=length,
                    source_offset=offset,
                    modal_amplitude=0.5,
                    radial_scale=1.0,
                    null_coordinate=null,
                    direction_seed=_derived_u64(group_id, "direction"),
                    active_block_length=1,
                    axis_mode=mode,
                    axis_sign=1,
                    contrast_group=group_id,
                    contrast_intent="invariance",
                    contrast_variant=variant,
                    rank_stratum=stratum,  # type: ignore[arg-type]
                )
            )
        groups.append(
            V3ContrastGroupSpec(
                group_id=group_id,
                family="null_single_invariance",
                intent="invariance",
                rank_stratum=stratum,  # type: ignore[arg-type]
                variant_probe_ids=tuple(variant_ids),
                canonical_variant_pairs=(
                    (variant_ids[0], variant_ids[1]),
                    (variant_ids[0], variant_ids[2]),
                    (variant_ids[1], variant_ids[2]),
                ),
            )
        )

    groups.sort(key=lambda value: value.group_id)
    return V3AssessmentProtocol(
        frozen_candidate=FrozenV2CandidateBinding(),
        probes=tuple(probes),
        contrast_groups=tuple(groups),
    )


def default_v3_assessment_protocol() -> V3AssessmentProtocol:
    """Return the authenticated fresh V3 assessment declaration."""

    return _build_protocol()
