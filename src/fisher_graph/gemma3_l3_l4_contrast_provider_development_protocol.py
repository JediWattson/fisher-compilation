"""Fresh development-only protocol for contrast-aware Gemma L3/L4 providers.

This declaration is additive and independent of the sealed V2 and V3
protocols.  It contains no tensor or model imports.  Its three roles have
strictly different authority:

* ``pilot`` may choose one signed-displacement amplitude by a frozen rule;
* ``fit`` may train each preregistered latent-rank candidate; and
* ``selection`` may only choose among an already-frozen candidate set.

The later materializer requires an authenticated calibration binding for fit
and selection, and additionally requires an authenticated frozen-candidate
binding before selection probes can be opened.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal


__all__ = [
    "CALIBRATION_AMPLITUDE_GRID",
    "CALIBRATION_EXACT_HALF_PAIRS",
    "CONSUMED_C1_PILOT_PANEL_SHA256",
    "CONSUMED_C1_PROTOCOL_SHA256",
    "DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256",
    "DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256",
    "DEFAULT_DEVELOPMENT_PROTOCOL_SHA256",
    "DEFAULT_DEVELOPMENT_SELECTION_PANEL_SHA256",
    "DEVELOPMENT_CANDIDATE_IDS",
    "DEVELOPMENT_RANK_LADDER",
    "CalibrationPilotMetric",
    "ContrastProviderDevelopmentProtocol",
    "DevelopmentCalibrationBinding",
    "DevelopmentContrastGroupSpec",
    "DevelopmentProbeSpec",
    "DevelopmentSparseBlock",
    "FrozenDevelopmentCandidateSet",
    "SignedCalibrationRule",
    "calibrated_role_panel_sha256",
    "default_contrast_provider_development_protocol",
    "freeze_development_candidates",
    "select_global_calibration_amplitude",
]


DevelopmentRole = Literal["pilot", "fit", "selection"]
DevelopmentFamily = Literal[
    "calibration_signed",
    "multitone",
    "block_sparse",
    "radial_sensitivity",
    "signed_sensitivity",
    "null_invariance",
]
ContrastIntent = Literal["sensitivity", "invariance"]
RankBand = Literal[
    "band_00_07",
    "band_08_15",
    "band_16_31",
    "band_32_63",
]

CALIBRATION_AMPLITUDE_GRID = (2.0, 4.0, 6.0, 8.0, 12.0)
CALIBRATION_EXACT_HALF_PAIRS = (
    (4.0, 2.0),
    (8.0, 4.0),
    (12.0, 6.0),
)
DEVELOPMENT_RANK_LADDER = (8, 16, 32)
DEVELOPMENT_CANDIDATE_IDS = (
    "latent-r08",
    "latent-r16",
    "latent-r32",
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_contrast_provider_development_c2.v1"
_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_MASK64 = (1 << 64) - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = ("pilot", "fit", "selection")
_BANDS = (
    "band_00_07",
    "band_08_15",
    "band_16_31",
    "band_32_63",
)
_FAMILIES = {
    "calibration_signed",
    "multitone",
    "block_sparse",
    "radial_sensitivity",
    "signed_sensitivity",
    "null_invariance",
}
_SENSITIVITY_FAMILIES = {
    "calibration_signed",
    "radial_sensitivity",
    "signed_sensitivity",
}
_PROBE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:probe:v1\0"
)
_GROUP_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:group:v1\0"
)
_PANEL_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:panel:v1\0"
)
_PROTOCOL_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:protocol:v1\0"
)
_DIRECTION_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:direction:v1\0"
)
_CALIBRATION_RULE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"calibration-rule:v1\0"
)
_PILOT_METRIC_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"pilot-metric:v1\0"
)
_CALIBRATION_BINDING_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"calibration-binding:v1\0"
)
_CALIBRATED_PANEL_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"calibrated-panel:v1\0"
)
_CANDIDATE_SET_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"candidate-set:v1\0"
)

# The C1 pilot is consumed.  These are audit-only denylist anchors; C1 fit and
# selection were never opened.
CONSUMED_C1_PROTOCOL_SHA256 = (
    "829fe983b1a221b888d683d71e658a86038dbbb61fd5b8fd1e3ebd979e40aadf"
)
CONSUMED_C1_PILOT_PANEL_SHA256 = (
    "5268c22ecc9b8154d0f1f70a653bab9b146f88815b7f11f8c0576f278f5bb085"
)

# Literal trust anchors for the complete C2 declaration.
DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256 = (
    "2b4d4efc8dbbdbc1c6afdb1a3134068f31b39ebc65e8a226fbbb9c92fe07b5e3"
)
DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256 = (
    "e6244896f2f6823f61d5ac57fad0756be320961e0b2272612579f9ba2f5b3e75"
)
DEFAULT_DEVELOPMENT_SELECTION_PANEL_SHA256 = (
    "4e86a5acf6fcaa4a1e03b61e387938fb442f11e2d82aee73b56d1ee0189013f6"
)
DEFAULT_DEVELOPMENT_PROTOCOL_SHA256 = (
    "033020dc9a0da819bd5753eb10090bff1bd9b4fcf61f33cd7186b1c1e3cb5254"
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


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _positive(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _derived_u64(*parts: object) -> int:
    digest = hashlib.sha256(_DIRECTION_DOMAIN)
    for part in parts:
        encoded = str(part).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "little") & _MASK64


def _rank_band(mode: int) -> RankBand:
    if mode < 8:
        return "band_00_07"
    if mode < 16:
        return "band_08_15"
    if mode < 32:
        return "band_16_31"
    return "band_32_63"


def _predecessor_pilot_summary() -> dict[str, object]:
    """Return tensor-free provenance for the consumed, failed-closed C1 pilot."""

    return {
        "protocol_sha256": CONSUMED_C1_PROTOCOL_SHA256,
        "pilot_panel_sha256": CONSUMED_C1_PILOT_PANEL_SHA256,
        "outcome": "failed_closed_no_eligible_global_amplitude",
        "maximum_tested_amplitude": 2.0,
        "teacher_relative_effect_at_maximum_by_band": {
            "band_00_07": 0.008251,
            "band_08_15": 0.003732,
            "band_16_31": 0.011773,
            "band_32_63": 0.003672,
        },
        "fd_cosine_summary": "approximately_one",
        "fd_gain_range_summary": [0.998, 0.999],
        "unchanged_minimum_effect_lower": 0.02,
        "failure_reason": "no_amplitude_met_unchanged_effect_floor",
        "fit_opened": False,
        "selection_opened": False,
        "c2_change_scope": (
            "fresh_pilot_identities_and_amplitude_grid_not_gate_tuning"
        ),
    }


@dataclass(frozen=True, slots=True)
class SignedCalibrationRule:
    """Frozen fit-only rule for choosing one global displacement amplitude."""

    amplitude_grid: tuple[float, ...] = CALIBRATION_AMPLITUDE_GRID
    exact_half_pairs: tuple[tuple[float, float], ...] = (
        CALIBRATION_EXACT_HALF_PAIRS
    )
    minimum_eligible_fraction: float = 0.75
    minimum_effect_lower: float = 0.02
    maximum_effect_upper: float = 0.25
    minimum_half_full_fd_cosine: float = 0.98
    minimum_half_full_fd_gain: float = 0.80
    maximum_half_full_fd_gain: float = 1.25
    selection_rule: str = "smallest_globally_eligible_amplitude"
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        grid = tuple(
            _positive(value, label="calibration amplitude")
            for value in self.amplitude_grid
        )
        if grid != CALIBRATION_AMPLITUDE_GRID:
            raise ValueError("calibration amplitude grid is not frozen")
        pairs = tuple(
            (
                _positive(full, label="full calibration amplitude"),
                _positive(half, label="half calibration amplitude"),
            )
            for full, half in self.exact_half_pairs
        )
        if (
            pairs != CALIBRATION_EXACT_HALF_PAIRS
            or any(not math.isclose(full / 2.0, half) for full, half in pairs)
        ):
            raise ValueError("calibration exact-half pairs are not frozen")
        for name in (
            "minimum_eligible_fraction",
            "minimum_half_full_fd_cosine",
        ):
            value = _finite(getattr(self, name), label=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        lower = _positive(
            self.minimum_effect_lower,
            label="minimum_effect_lower",
        )
        upper = _positive(
            self.maximum_effect_upper,
            label="maximum_effect_upper",
        )
        if lower >= upper:
            raise ValueError("calibration effect interval is invalid")
        minimum_gain = _positive(
            self.minimum_half_full_fd_gain,
            label="minimum_half_full_fd_gain",
        )
        maximum_gain = _positive(
            self.maximum_half_full_fd_gain,
            label="maximum_half_full_fd_gain",
        )
        if minimum_gain > maximum_gain:
            raise ValueError("calibration gain interval is invalid")
        if self.selection_rule != "smallest_globally_eligible_amplitude":
            raise ValueError("calibration selection rule is not frozen")
        computed = _digest(self._payload(), domain=_CALIBRATION_RULE_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("calibration rule hash mismatch")
        object.__setattr__(self, "amplitude_grid", grid)
        object.__setattr__(self, "exact_half_pairs", pairs)
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "amplitude_grid": list(self.amplitude_grid),
            "exact_half_pairs": [list(value) for value in self.exact_half_pairs],
            "minimum_eligible_fraction": self.minimum_eligible_fraction,
            "minimum_effect_lower": self.minimum_effect_lower,
            "maximum_effect_upper": self.maximum_effect_upper,
            "minimum_half_full_fd_cosine": (
                self.minimum_half_full_fd_cosine
            ),
            "minimum_half_full_fd_gain": self.minimum_half_full_fd_gain,
            "maximum_half_full_fd_gain": self.maximum_half_full_fd_gain,
            "selection_rule": self.selection_rule,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class DevelopmentSparseBlock:
    start: int
    length: int
    mode_signs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _exact_int(self.start, label="sparse block start")
        _exact_int(self.length, label="sparse block length", minimum=1)
        if type(self.mode_signs) is not tuple or len(self.mode_signs) != 4:
            raise ValueError("sparse block must contain four modes")
        modes: list[int] = []
        for value in self.mode_signs:
            if (
                type(value) is not tuple
                or len(value) != 2
                or type(value[0]) is not int
                or not 0 <= value[0] < _MODAL_WIDTH
                or value[1] not in (-1, 1)
            ):
                raise ValueError("sparse block mode/sign is invalid")
            modes.append(value[0])
        if len(set(modes)) != 4:
            raise ValueError("sparse block modes must be unique")

    def state_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "length": self.length,
            "mode_signs": [list(value) for value in self.mode_signs],
        }


@dataclass(frozen=True, slots=True)
class DevelopmentProbeSpec:
    """One immutable development probe endpoint."""

    probe_id: str
    role: DevelopmentRole
    ordinal: int
    family: DevelopmentFamily
    sequence_length: int
    source_offset: int
    active_block_length: int
    modal_amplitude: float
    radial_scale: float
    null_coordinate: float
    direction_seed: int
    uses_calibrated_amplitude: bool = False
    rank_band: RankBand | None = None
    axis_mode: int | None = None
    axis_sign: int | None = None
    multitone_temporal_frequencies: tuple[int, int] | None = None
    multitone_modal_frequencies: tuple[int, int] | None = None
    multitone_phase_quadrants: tuple[int, int] | None = None
    sparse_blocks: tuple[DevelopmentSparseBlock, ...] = ()
    contrast_group: str | None = None
    contrast_intent: ContrastIntent | None = None
    contrast_variant: str | None = None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.probe_id, str)
            or not self.probe_id.startswith(f"development_c2.{self.role}.")
        ):
            raise ValueError("development probe id has the wrong namespace")
        if self.role not in _ROLES or self.family not in _FAMILIES:
            raise ValueError("development probe role or family is invalid")
        _exact_int(self.ordinal, label="probe ordinal")
        length = _exact_int(
            self.sequence_length,
            label="sequence length",
            minimum=1,
        )
        offset = _exact_int(self.source_offset, label="source offset")
        block_length = _exact_int(
            self.active_block_length,
            label="active block length",
            minimum=1,
        )
        if offset >= length or offset + block_length > length:
            raise ValueError("probe active block lies outside its sequence")
        _positive(self.modal_amplitude, label="modal amplitude")
        _positive(self.radial_scale, label="radial scale")
        _finite(self.null_coordinate, label="null coordinate")
        seed = _exact_int(self.direction_seed, label="direction seed")
        if seed > _MASK64:
            raise ValueError("direction seed exceeds uint64")
        if type(self.uses_calibrated_amplitude) is not bool:
            raise TypeError("uses_calibrated_amplitude must be boolean")

        is_ordinary = self.family in {"multitone", "block_sparse"}
        is_contrast = not is_ordinary
        contrast_fields = (
            self.rank_band,
            self.contrast_group,
            self.contrast_intent,
            self.contrast_variant,
        )
        if is_ordinary and any(value is not None for value in contrast_fields):
            raise ValueError("ordinary probe cannot carry contrast metadata")
        if is_contrast and any(value is None for value in contrast_fields):
            raise ValueError("contrast probe metadata must be complete")
        if self.rank_band is not None and self.rank_band not in _BANDS:
            raise ValueError("probe rank band is invalid")
        if self.contrast_intent is not None and self.contrast_intent not in {
            "sensitivity",
            "invariance",
        }:
            raise ValueError("probe contrast intent is invalid")
        if (
            self.family in _SENSITIVITY_FAMILIES
            and self.contrast_intent != "sensitivity"
        ) or (
            self.family == "null_invariance"
            and self.contrast_intent != "invariance"
        ):
            raise ValueError("probe family and contrast intent disagree")

        has_axis = self.axis_mode is not None or self.axis_sign is not None
        if has_axis and (
            type(self.axis_mode) is not int
            or not 0 <= self.axis_mode < _MODAL_WIDTH
            or self.axis_sign not in (-1, 1)
            or _rank_band(self.axis_mode) != self.rank_band
        ):
            raise ValueError("probe axis coordinate is invalid")
        if is_contrast and not has_axis:
            raise ValueError("contrast probe requires an axis coordinate")

        has_multitone = any(
            value is not None
            for value in (
                self.multitone_temporal_frequencies,
                self.multitone_modal_frequencies,
                self.multitone_phase_quadrants,
            )
        )
        if self.family == "multitone":
            temporal = self.multitone_temporal_frequencies
            modal = self.multitone_modal_frequencies
            phase = self.multitone_phase_quadrants
            if (
                type(temporal) is not tuple
                or type(modal) is not tuple
                or type(phase) is not tuple
                or len(temporal) != 2
                or len(modal) != 2
                or len(phase) != 2
                or temporal[0] == temporal[1]
                or modal[0] == modal[1]
                or any(value <= 0 for value in temporal)
                or any(not 0 <= value < _MODAL_WIDTH for value in modal)
                or any(value not in (0, 1, 2, 3) for value in phase)
                or block_length != length - offset
                or self.sparse_blocks
                or has_axis
            ):
                raise ValueError("multitone probe fields are inconsistent")
        elif has_multitone:
            raise ValueError("only multitone probes may carry frequencies")

        if self.family == "block_sparse":
            if len(self.sparse_blocks) != 2 or has_axis:
                raise ValueError("block-sparse probe requires two blocks")
            ordered = tuple(
                sorted(self.sparse_blocks, key=lambda value: value.start)
            )
            if ordered != self.sparse_blocks:
                raise ValueError("sparse blocks must be ordered")
            for block in ordered:
                if (
                    block.start < offset
                    or block.start + block.length > length
                ):
                    raise ValueError("sparse block lies outside active suffix")
            if ordered[0].start + ordered[0].length > ordered[1].start:
                raise ValueError("sparse blocks overlap")
        elif self.sparse_blocks:
            raise ValueError("only block-sparse probes may carry sparse blocks")

        if self.role == "pilot":
            if (
                self.family != "calibration_signed"
                or self.uses_calibrated_amplitude
                or self.modal_amplitude not in CALIBRATION_AMPLITUDE_GRID
            ):
                raise ValueError("pilot probes must be fixed calibration steps")
        elif self.family == "calibration_signed":
            raise ValueError("calibration-signed probes are pilot-only")
        if self.uses_calibrated_amplitude and self.family not in {
            "radial_sensitivity",
            "signed_sensitivity",
        }:
            raise ValueError("only radial/signed probes may use calibrated h")

        computed = _digest(self._payload(), domain=_PROBE_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("development probe hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.contrast_provider_development_c2_probe.v1"
            ),
            "format_version": _FORMAT_VERSION,
            "probe_id": self.probe_id,
            "role": self.role,
            "ordinal": self.ordinal,
            "family": self.family,
            "sequence_length": self.sequence_length,
            "source_offset": self.source_offset,
            "active_block_length": self.active_block_length,
            "modal_amplitude": self.modal_amplitude,
            "radial_scale": self.radial_scale,
            "null_coordinate": self.null_coordinate,
            "direction_seed": self.direction_seed,
            "uses_calibrated_amplitude": self.uses_calibrated_amplitude,
            "rank_band": self.rank_band,
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
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class DevelopmentContrastGroupSpec:
    group_id: str
    role: DevelopmentRole
    family: DevelopmentFamily
    intent: ContrastIntent
    rank_band: RankBand
    variant_probe_ids: tuple[str, ...]
    canonical_variant_pairs: tuple[tuple[str, str], ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.group_id, str)
            or not self.group_id.startswith(f"development_c2.{self.role}.")
        ):
            raise ValueError("contrast group has the wrong namespace")
        if self.role not in _ROLES or self.family not in _FAMILIES:
            raise ValueError("contrast group role or family is invalid")
        if self.family in _SENSITIVITY_FAMILIES:
            if self.intent != "sensitivity":
                raise ValueError("sensitivity group has the wrong intent")
        elif self.family == "null_invariance":
            if self.intent != "invariance":
                raise ValueError("null group has the wrong intent")
        else:
            raise ValueError("ordinary families cannot form contrast groups")
        if self.rank_band not in _BANDS:
            raise ValueError("contrast group rank band is invalid")
        if (
            type(self.variant_probe_ids) is not tuple
            or len(self.variant_probe_ids) < 2
            or len(set(self.variant_probe_ids)) != len(self.variant_probe_ids)
        ):
            raise ValueError("contrast group variants are invalid")
        variants = set(self.variant_probe_ids)
        if (
            type(self.canonical_variant_pairs) is not tuple
            or not self.canonical_variant_pairs
        ):
            raise ValueError("contrast group pairs are empty")
        seen: set[tuple[str, str]] = set()
        for pair in self.canonical_variant_pairs:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or pair[0] == pair[1]
                or pair[0] not in variants
                or pair[1] not in variants
                or pair in seen
            ):
                raise ValueError("contrast group pair is invalid")
            seen.add(pair)
        if self.intent == "invariance" and len(seen) != (
            len(variants) * (len(variants) - 1) // 2
        ):
            raise ValueError("null group must compare every endpoint pair")
        computed = _digest(self._payload(), domain=_GROUP_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("contrast group hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.contrast_provider_development_c2_group.v1"
            ),
            "format_version": _FORMAT_VERSION,
            "group_id": self.group_id,
            "role": self.role,
            "family": self.family,
            "intent": self.intent,
            "rank_band": self.rank_band,
            "variant_probe_ids": list(self.variant_probe_ids),
            "canonical_variant_pairs": [
                list(value) for value in self.canonical_variant_pairs
            ],
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _panel_sha256(
    role: DevelopmentRole,
    probes: Sequence[DevelopmentProbeSpec],
    groups: Sequence[DevelopmentContrastGroupSpec],
) -> str:
    return _digest(
        {
            "schema": (
                "fisher_graph.contrast_provider_development_c2_panel.v1"
            ),
            "format_version": _FORMAT_VERSION,
            "role": role,
            "ordered_probe_sha256s": [
                value.artifact_sha256 for value in probes
            ],
            "ordered_group_sha256s": [
                value.artifact_sha256 for value in groups
            ],
        },
        domain=_PANEL_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class ContrastProviderDevelopmentProtocol:
    calibration_rule: SignedCalibrationRule
    rank_ladder: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    probes: tuple[DevelopmentProbeSpec, ...]
    contrast_groups: tuple[DevelopmentContrastGroupSpec, ...]
    protocol_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_rule, SignedCalibrationRule):
            raise TypeError("protocol requires a calibration rule")
        if self.rank_ladder != DEVELOPMENT_RANK_LADDER:
            raise ValueError("development rank ladder is not frozen")
        if self.candidate_ids != DEVELOPMENT_CANDIDATE_IDS:
            raise ValueError("development candidate ids are not frozen")
        if type(self.probes) is not tuple or len(self.probes) != 200:
            raise ValueError("development protocol must contain 200 probes")
        if (
            len({value.probe_id for value in self.probes}) != 200
            or len({value.artifact_sha256 for value in self.probes}) != 200
        ):
            raise ValueError("development probe identities must be unique")
        expected_counts = {"pilot": 40, "fit": 80, "selection": 80}
        if Counter(value.role for value in self.probes) != expected_counts:
            raise ValueError("development role counts drifted")
        for role in _ROLES:
            role_probes = self.probes_for_role(role)  # type: ignore[arg-type]
            if tuple(value.ordinal for value in role_probes) != tuple(
                range(expected_counts[role])
            ):
                raise ValueError(f"{role} probe ordinals drifted")
        expected_families = {
            "pilot": {"calibration_signed": 40},
            "fit": {
                "multitone": 8,
                "block_sparse": 8,
                "radial_sensitivity": 24,
                "signed_sensitivity": 16,
                "null_invariance": 24,
            },
            "selection": {
                "multitone": 8,
                "block_sparse": 8,
                "radial_sensitivity": 24,
                "signed_sensitivity": 16,
                "null_invariance": 24,
            },
        }
        for role, expected in expected_families.items():
            actual = Counter(
                value.family
                for value in self.probes_for_role(role)  # type: ignore[arg-type]
            )
            if dict(actual) != expected:
                raise ValueError(f"{role} family counts drifted")
        if type(self.contrast_groups) is not tuple or len(
            self.contrast_groups
        ) != 68:
            raise ValueError("development protocol must contain 68 groups")
        if len({value.group_id for value in self.contrast_groups}) != 68:
            raise ValueError("development contrast groups must be unique")
        expected_group_counts = {"pilot": 20, "fit": 24, "selection": 24}
        if Counter(
            value.role for value in self.contrast_groups
        ) != expected_group_counts:
            raise ValueError("development contrast-group counts drifted")
        probes_by_id = {value.probe_id: value for value in self.probes}
        grouped: set[str] = set()
        for group in self.contrast_groups:
            for probe_id in group.variant_probe_ids:
                probe = probes_by_id.get(probe_id)
                if (
                    probe is None
                    or probe.role != group.role
                    or probe.family != group.family
                    or probe.rank_band != group.rank_band
                    or probe.contrast_group != group.group_id
                    or probe.contrast_intent != group.intent
                    or probe_id in grouped
                ):
                    raise ValueError("contrast group differs from probe metadata")
                grouped.add(probe_id)
        expected_grouped = {
            value.probe_id
            for value in self.probes
            if value.contrast_group is not None
        }
        if grouped != expected_grouped:
            raise ValueError("contrast endpoints are not grouped exactly once")
        anchors = {
            "pilot": DEFAULT_DEVELOPMENT_PILOT_PANEL_SHA256,
            "fit": DEFAULT_DEVELOPMENT_FIT_PANEL_SHA256,
            "selection": DEFAULT_DEVELOPMENT_SELECTION_PANEL_SHA256,
        }
        for role, anchor in anchors.items():
            if anchor and self.panel_sha256(role) != anchor:  # type: ignore[arg-type]
                raise ValueError(f"{role} development panel drifted")
        computed = _digest(self._payload(), domain=_PROTOCOL_DOMAIN)
        if self.protocol_sha256 and self.protocol_sha256 != computed:
            raise ValueError("development protocol hash mismatch")
        if (
            DEFAULT_DEVELOPMENT_PROTOCOL_SHA256
            and computed != DEFAULT_DEVELOPMENT_PROTOCOL_SHA256
        ):
            raise ValueError("development protocol differs from trust anchor")
        object.__setattr__(self, "protocol_sha256", computed)

    def probes_for_role(
        self,
        role: DevelopmentRole,
    ) -> tuple[DevelopmentProbeSpec, ...]:
        if role not in _ROLES:
            raise ValueError("development role is invalid")
        return tuple(value for value in self.probes if value.role == role)

    def groups_for_role(
        self,
        role: DevelopmentRole,
    ) -> tuple[DevelopmentContrastGroupSpec, ...]:
        if role not in _ROLES:
            raise ValueError("development role is invalid")
        return tuple(
            value for value in self.contrast_groups if value.role == role
        )

    def panel_sha256(self, role: DevelopmentRole) -> str:
        return _panel_sha256(
            role,
            self.probes_for_role(role),
            self.groups_for_role(role),
        )

    def calibrated_panel_sha256(
        self,
        role: Literal["fit", "selection"],
        calibration: DevelopmentCalibrationBinding,
    ) -> str:
        return calibrated_role_panel_sha256(self, role, calibration)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scope": "development_only_no_assessment_authority",
            "rank_semantics": "full_width_io_with_latent_executor_rank",
            "rank_ladder": list(self.rank_ladder),
            "candidate_ids": list(self.candidate_ids),
            "calibration_rule": self.calibration_rule.state_dict(),
            "role_panel_sha256s": {
                role: self.panel_sha256(role) for role in _ROLES
            },
            "probes": [value.state_dict() for value in self.probes],
            "contrast_groups": [
                value.state_dict() for value in self.contrast_groups
            ],
            "selection_may_fit_or_refit": False,
            "selection_requires_frozen_candidates": True,
            "v2_or_v3_targets_loaded": False,
            "predecessor_pilot": _predecessor_pilot_summary(),
            "natural_prompt_claim": False,
            "model_replacement_claim": False,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "protocol_sha256": self.protocol_sha256}


@dataclass(frozen=True, slots=True)
class CalibrationPilotMetric:
    """Tensor-free result for one pilot base and amplitude."""

    metric_id: str
    rank_band: RankBand
    amplitude: float
    teacher_relative_effect_lower: float
    teacher_relative_effect_upper: float
    half_full_fd_cosine: float | None
    half_full_fd_gain: float | None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.metric_id, str)
            or not self.metric_id.startswith("development_c2.pilot.metric.")
        ):
            raise ValueError("pilot metric id has the wrong namespace")
        if self.rank_band not in _BANDS:
            raise ValueError("pilot metric rank band is invalid")
        if self.amplitude not in CALIBRATION_AMPLITUDE_GRID:
            raise ValueError("pilot metric amplitude is not frozen")
        lower = _finite(
            self.teacher_relative_effect_lower,
            label="teacher effect lower",
        )
        upper = _finite(
            self.teacher_relative_effect_upper,
            label="teacher effect upper",
        )
        if lower < 0.0 or upper < lower:
            raise ValueError("pilot teacher effect interval is invalid")
        exact_full_steps = {
            full for full, _half in CALIBRATION_EXACT_HALF_PAIRS
        }
        if self.amplitude in exact_full_steps:
            if self.half_full_fd_cosine is None or self.half_full_fd_gain is None:
                raise ValueError(
                    "exact-half pilot metric requires cosine and gain"
                )
            cosine = _finite(
                self.half_full_fd_cosine,
                label="half/full FD cosine",
            )
            if not -1.0 <= cosine <= 1.0:
                raise ValueError("half/full FD cosine lies outside [-1, 1]")
            _finite(self.half_full_fd_gain, label="half/full FD gain")
        elif (
            self.half_full_fd_cosine is not None
            or self.half_full_fd_gain is not None
        ):
            raise ValueError(
                "pilot amplitude without an exact half must not claim FD metrics"
            )
        computed = _digest(self._payload(), domain=_PILOT_METRIC_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("pilot metric hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "rank_band": self.rank_band,
            "amplitude": self.amplitude,
            "teacher_relative_effect_lower": (
                self.teacher_relative_effect_lower
            ),
            "teacher_relative_effect_upper": (
                self.teacher_relative_effect_upper
            ),
            "half_full_fd_cosine": self.half_full_fd_cosine,
            "half_full_fd_gain": self.half_full_fd_gain,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class DevelopmentCalibrationBinding:
    protocol_sha256: str
    pilot_panel_sha256: str
    calibration_rule_sha256: str
    selected_amplitude: float
    pilot_metric_sha256s: tuple[str, ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "pilot_panel_sha256",
            "calibration_rule_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.selected_amplitude not in CALIBRATION_AMPLITUDE_GRID:
            raise ValueError("selected amplitude is not on the frozen grid")
        if (
            type(self.pilot_metric_sha256s) is not tuple
            or len(self.pilot_metric_sha256s) != 20
            or len(set(self.pilot_metric_sha256s)) != 20
        ):
            raise ValueError("calibration binding requires 20 pilot metrics")
        for value in self.pilot_metric_sha256s:
            _require_sha256(value, label="pilot metric hash")
        computed = _digest(self._payload(), domain=_CALIBRATION_BINDING_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("calibration binding hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "pilot_panel_sha256": self.pilot_panel_sha256,
            "calibration_rule_sha256": self.calibration_rule_sha256,
            "selected_amplitude": self.selected_amplitude,
            "pilot_metric_sha256s": list(self.pilot_metric_sha256s),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def select_global_calibration_amplitude(
    protocol: ContrastProviderDevelopmentProtocol,
    metrics: Sequence[CalibrationPilotMetric],
) -> DevelopmentCalibrationBinding:
    """Apply the frozen fit-only rule and return an authenticated binding."""

    if not isinstance(protocol, ContrastProviderDevelopmentProtocol):
        raise TypeError("protocol must be a development protocol")
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise TypeError("pilot metrics must be a sequence")
    values = tuple(metrics)
    if (
        len(values) != 20
        or any(not isinstance(value, CalibrationPilotMetric) for value in values)
        or len({value.metric_id for value in values}) != 20
        or len({value.artifact_sha256 for value in values}) != 20
    ):
        raise ValueError("calibration requires 20 unique pilot metrics")
    expected_coordinates = {
        (band, amplitude)
        for band in _BANDS
        for amplitude in CALIBRATION_AMPLITUDE_GRID
    }
    if {(value.rank_band, value.amplitude) for value in values} != (
        expected_coordinates
    ):
        raise ValueError("pilot metric coordinates are incomplete")
    rule = protocol.calibration_rule
    required = math.ceil(rule.minimum_eligible_fraction * len(_BANDS))
    selected: float | None = None
    eligible_amplitudes = tuple(full for full, _half in rule.exact_half_pairs)
    for amplitude in eligible_amplitudes:
        rows = tuple(value for value in values if value.amplitude == amplitude)
        effect_eligible = sum(
            value.teacher_relative_effect_lower
            >= rule.minimum_effect_lower
            for value in rows
        )
        stable = all(
            value.teacher_relative_effect_upper
            <= rule.maximum_effect_upper
            and value.half_full_fd_cosine is not None
            and value.half_full_fd_gain is not None
            and value.half_full_fd_cosine
            >= rule.minimum_half_full_fd_cosine
            and rule.minimum_half_full_fd_gain
            <= value.half_full_fd_gain
            <= rule.maximum_half_full_fd_gain
            for value in rows
        )
        if effect_eligible >= required and stable:
            selected = amplitude
            break
    if selected is None:
        raise ValueError("no globally eligible calibration amplitude")
    ordered = tuple(
        sorted(values, key=lambda value: (value.rank_band, value.amplitude))
    )
    return DevelopmentCalibrationBinding(
        protocol_sha256=protocol.protocol_sha256,
        pilot_panel_sha256=protocol.panel_sha256("pilot"),
        calibration_rule_sha256=rule.artifact_sha256,
        selected_amplitude=selected,
        pilot_metric_sha256s=tuple(
            value.artifact_sha256 for value in ordered
        ),
    )


def _authenticate_calibration(
    protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
) -> None:
    if not isinstance(calibration, DevelopmentCalibrationBinding):
        raise TypeError("calibration must be a development binding")
    if (
        calibration.protocol_sha256 != protocol.protocol_sha256
        or calibration.pilot_panel_sha256 != protocol.panel_sha256("pilot")
        or calibration.calibration_rule_sha256
        != protocol.calibration_rule.artifact_sha256
    ):
        raise ValueError("calibration binding is incompatible with protocol")


def calibrated_role_panel_sha256(
    protocol: ContrastProviderDevelopmentProtocol,
    role: Literal["fit", "selection"],
    calibration: DevelopmentCalibrationBinding,
) -> str:
    if role not in {"fit", "selection"}:
        raise ValueError("only fit/selection panels are calibrated")
    _authenticate_calibration(protocol, calibration)
    return _digest(
        {
            "schema": "fisher_graph.calibrated_development_c2_panel.v1",
            "format_version": _FORMAT_VERSION,
            "role": role,
            "base_panel_sha256": protocol.panel_sha256(role),
            "calibration_binding_sha256": calibration.artifact_sha256,
            "selected_amplitude": calibration.selected_amplitude,
        },
        domain=_CALIBRATED_PANEL_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class FrozenDevelopmentCandidateSet:
    protocol_sha256: str
    calibration_sha256: str
    calibrated_fit_panel_sha256: str
    rank_ladder: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    candidate_artifact_sha256s: tuple[str, ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "calibration_sha256",
            "calibrated_fit_panel_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.rank_ladder != DEVELOPMENT_RANK_LADDER:
            raise ValueError("frozen candidate rank ladder drifted")
        if self.candidate_ids != DEVELOPMENT_CANDIDATE_IDS:
            raise ValueError("frozen candidate ids drifted")
        if (
            type(self.candidate_artifact_sha256s) is not tuple
            or len(self.candidate_artifact_sha256s) != 3
            or len(set(self.candidate_artifact_sha256s)) != 3
        ):
            raise ValueError("frozen candidate set requires three unique hashes")
        for value in self.candidate_artifact_sha256s:
            _require_sha256(value, label="candidate artifact hash")
        computed = _digest(self._payload(), domain=_CANDIDATE_SET_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("frozen candidate-set hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "calibration_sha256": self.calibration_sha256,
            "calibrated_fit_panel_sha256": (
                self.calibrated_fit_panel_sha256
            ),
            "rank_ladder": list(self.rank_ladder),
            "candidate_ids": list(self.candidate_ids),
            "candidate_artifact_sha256s": list(
                self.candidate_artifact_sha256s
            ),
            "all_candidates_fitted_and_frozen_before_selection": True,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def freeze_development_candidates(
    protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
    candidate_artifact_sha256s: Sequence[str],
) -> FrozenDevelopmentCandidateSet:
    _authenticate_calibration(protocol, calibration)
    values = tuple(candidate_artifact_sha256s)
    return FrozenDevelopmentCandidateSet(
        protocol_sha256=protocol.protocol_sha256,
        calibration_sha256=calibration.artifact_sha256,
        calibrated_fit_panel_sha256=calibrated_role_panel_sha256(
            protocol,
            "fit",
            calibration,
        ),
        rank_ladder=protocol.rank_ladder,
        candidate_ids=protocol.candidate_ids,
        candidate_artifact_sha256s=values,
    )


def _sparse_blocks(
    *,
    role: DevelopmentRole,
    base_index: int,
    length: int,
    offset: int,
) -> tuple[DevelopmentSparseBlock, ...]:
    suffix = length - offset
    block_length = max(2, suffix // 10)
    starts = (offset, min(length - block_length, offset + suffix // 2))
    result: list[DevelopmentSparseBlock] = []
    for block_index, start in enumerate(starts):
        modes: list[int] = []
        cursor = 0
        while len(modes) < 4:
            mode = int(
                _derived_u64(
                    role,
                    "block_sparse",
                    base_index,
                    block_index,
                    "mode",
                    cursor,
                )
                % _MODAL_WIDTH
            )
            cursor += 1
            if mode not in modes:
                modes.append(mode)
        signs = tuple(
            1
            if _derived_u64(
                role,
                "block_sparse",
                base_index,
                block_index,
                "sign",
                index,
            )
            & 1
            else -1
            for index in range(4)
        )
        result.append(
            DevelopmentSparseBlock(
                start=start,
                length=block_length,
                mode_signs=tuple(zip(modes, signs, strict=True)),
            )
        )
    return tuple(result)


def _ordinary_probes(
    *,
    role: Literal["fit", "selection"],
    schedule: Sequence[tuple[int, int, int, int]],
    ordinal_start: int,
) -> tuple[DevelopmentProbeSpec, ...]:
    probes: list[DevelopmentProbeSpec] = []
    for base_index, (mode, length, offset, _block_length) in enumerate(schedule):
        active = length - offset
        first_t = 1 + int(
            _derived_u64(role, "multitone", base_index, "t0")
            % max(2, active // 5)
        )
        second_t = 1 + int(
            _derived_u64(role, "multitone", base_index, "t1")
            % max(2, active // 5)
        )
        if second_t == first_t:
            second_t = 1 + second_t % max(2, active // 5)
        first_m = int(
            _derived_u64(role, "multitone", base_index, "m0")
            % _MODAL_WIDTH
        )
        second_m = int(
            _derived_u64(role, "multitone", base_index, "m1")
            % _MODAL_WIDTH
        )
        if second_m == first_m:
            second_m = (second_m + 1) % _MODAL_WIDTH
        probes.append(
            DevelopmentProbeSpec(
                probe_id=(
                    f"development_c2.{role}.ordinary.multitone."
                    f"{base_index:02d}"
                ),
                role=role,
                ordinal=ordinal_start + len(probes),
                family="multitone",
                sequence_length=length,
                source_offset=offset,
                active_block_length=active,
                modal_amplitude=(0.25, 0.5, 0.75, 1.0)[base_index % 4],
                radial_scale=(0.7, 1.1, 1.6, 1.9)[base_index % 4],
                null_coordinate=(-0.6, 0.0, 0.6, 0.0)[base_index % 4],
                direction_seed=_derived_u64(
                    role,
                    "multitone",
                    base_index,
                    "direction",
                ),
                multitone_temporal_frequencies=(first_t, second_t),
                multitone_modal_frequencies=(first_m, second_m),
                multitone_phase_quadrants=(
                    int(
                        _derived_u64(role, "multitone", base_index, "p0") % 4
                    ),
                    int(
                        _derived_u64(role, "multitone", base_index, "p1") % 4
                    ),
                ),
            )
        )
        probes.append(
            DevelopmentProbeSpec(
                probe_id=(
                    f"development_c2.{role}.ordinary.block_sparse."
                    f"{base_index:02d}"
                ),
                role=role,
                ordinal=ordinal_start + len(probes),
                family="block_sparse",
                sequence_length=length,
                source_offset=offset,
                active_block_length=active,
                modal_amplitude=(0.375, 0.625, 0.875, 0.5)[base_index % 4],
                radial_scale=(1.2, 0.8, 1.7, 1.0)[base_index % 4],
                null_coordinate=(0.0, 0.6, -0.6, 0.0)[base_index % 4],
                direction_seed=_derived_u64(
                    role,
                    "block_sparse",
                    base_index,
                    "direction",
                ),
                sparse_blocks=_sparse_blocks(
                    role=role,
                    base_index=base_index,
                    length=length,
                    offset=offset,
                ),
            )
        )
    return tuple(probes)


def _contrast_role(
    *,
    role: Literal["fit", "selection"],
    schedule: Sequence[tuple[int, int, int, int]],
    ordinal_start: int,
) -> tuple[
    tuple[DevelopmentProbeSpec, ...],
    tuple[DevelopmentContrastGroupSpec, ...],
]:
    probes: list[DevelopmentProbeSpec] = []
    groups: list[DevelopmentContrastGroupSpec] = []
    for base_index, (mode, length, offset, block_length) in enumerate(schedule):
        band = _rank_band(mode)
        declarations = (
            (
                "radial_sensitivity",
                "sensitivity",
                (
                    ("radius_0p625", 1, 0.625, 0.0),
                    ("radius_1p125", 1, 1.125, 0.0),
                    ("radius_1p875", 1, 1.875, 0.0),
                ),
                ((0, 1), (1, 2)),
                True,
            ),
            (
                "signed_sensitivity",
                "sensitivity",
                (
                    ("negative", -1, 1.25, 0.0),
                    ("positive", 1, 1.25, 0.0),
                ),
                ((0, 1),),
                True,
            ),
            (
                "null_invariance",
                "invariance",
                (
                    ("null_m0p75", 1, 1.0, -0.75),
                    ("null_0", 1, 1.0, 0.0),
                    ("null_p0p75", 1, 1.0, 0.75),
                ),
                ((0, 1), (0, 2), (1, 2)),
                False,
            ),
        )
        for (
            family,
            intent,
            variants,
            pair_indices,
            calibrated,
        ) in declarations:
            group_id = (
                f"development_c2.{role}.{family}.base_{base_index:02d}"
            )
            ids: list[str] = []
            for variant, sign, radial, null in variants:
                probe_id = f"{group_id}.{variant}"
                ids.append(probe_id)
                probes.append(
                    DevelopmentProbeSpec(
                        probe_id=probe_id,
                        role=role,
                        ordinal=ordinal_start + len(probes),
                        family=family,  # type: ignore[arg-type]
                        sequence_length=length,
                        source_offset=offset,
                        active_block_length=(
                            1 if family == "null_invariance" else block_length
                        ),
                        modal_amplitude=(
                            0.5 if family == "null_invariance" else 1.0
                        ),
                        radial_scale=radial,
                        null_coordinate=null,
                        direction_seed=_derived_u64(
                            role,
                            family,
                            base_index,
                            "direction",
                        ),
                        uses_calibrated_amplitude=calibrated,
                        rank_band=band,
                        axis_mode=mode,
                        axis_sign=sign,
                        contrast_group=group_id,
                        contrast_intent=intent,  # type: ignore[arg-type]
                        contrast_variant=variant,
                    )
                )
            groups.append(
                DevelopmentContrastGroupSpec(
                    group_id=group_id,
                    role=role,
                    family=family,  # type: ignore[arg-type]
                    intent=intent,  # type: ignore[arg-type]
                    rank_band=band,
                    variant_probe_ids=tuple(ids),
                    canonical_variant_pairs=tuple(
                        (ids[left], ids[right])
                        for left, right in pair_indices
                    ),
                )
            )
    return tuple(probes), tuple(groups)


def _pilot_panel() -> tuple[
    tuple[DevelopmentProbeSpec, ...],
    tuple[DevelopmentContrastGroupSpec, ...],
]:
    schedule = (
        (2, 68, 13, 6),
        (12, 108, 37, 8),
        (27, 164, 81, 10),
        (52, 228, 149, 12),
    )
    probes: list[DevelopmentProbeSpec] = []
    groups: list[DevelopmentContrastGroupSpec] = []
    for base_index, (mode, length, offset, block_length) in enumerate(schedule):
        band = _rank_band(mode)
        for amplitude in CALIBRATION_AMPLITUDE_GRID:
            label = str(amplitude).replace(".", "p")
            group_id = (
                f"development_c2.pilot.calibration_signed."
                f"base_{base_index:02d}.h_{label}"
            )
            ids: list[str] = []
            for variant, sign in (("negative", -1), ("positive", 1)):
                probe_id = f"{group_id}.{variant}"
                ids.append(probe_id)
                probes.append(
                    DevelopmentProbeSpec(
                        probe_id=probe_id,
                        role="pilot",
                        ordinal=len(probes),
                        family="calibration_signed",
                        sequence_length=length,
                        source_offset=offset,
                        active_block_length=block_length,
                        modal_amplitude=amplitude,
                        radial_scale=1.25,
                        null_coordinate=0.0,
                        direction_seed=_derived_u64(
                            "pilot",
                            "calibration_signed",
                            base_index,
                            "direction",
                        ),
                        rank_band=band,
                        axis_mode=mode,
                        axis_sign=sign,
                        contrast_group=group_id,
                        contrast_intent="sensitivity",
                        contrast_variant=variant,
                    )
                )
            groups.append(
                DevelopmentContrastGroupSpec(
                    group_id=group_id,
                    role="pilot",
                    family="calibration_signed",
                    intent="sensitivity",
                    rank_band=band,
                    variant_probe_ids=tuple(ids),
                    canonical_variant_pairs=((ids[0], ids[1]),),
                )
            )
    return tuple(probes), tuple(groups)


def default_contrast_provider_development_protocol(
) -> ContrastProviderDevelopmentProtocol:
    """Return the authenticated fresh 40/80/80 development declaration."""

    pilot_probes, pilot_groups = _pilot_panel()
    fit_schedule = (
        (4, 56, 7, 5),
        (5, 96, 29, 7),
        (9, 136, 61, 9),
        (13, 184, 83, 11),
        (18, 216, 119, 7),
        (23, 248, 155, 9),
        (34, 120, 37, 5),
        (46, 200, 101, 11),
    )
    selection_schedule = (
        (0, 60, 11, 7),
        (7, 100, 33, 9),
        (8, 140, 65, 5),
        (14, 188, 87, 7),
        (17, 220, 123, 9),
        (31, 252, 159, 11),
        (35, 124, 41, 7),
        (58, 204, 105, 9),
    )
    fit_ordinary = _ordinary_probes(
        role="fit",
        schedule=fit_schedule,
        ordinal_start=0,
    )
    fit_contrast, fit_groups = _contrast_role(
        role="fit",
        schedule=fit_schedule,
        ordinal_start=len(fit_ordinary),
    )
    selection_ordinary = _ordinary_probes(
        role="selection",
        schedule=selection_schedule,
        ordinal_start=0,
    )
    selection_contrast, selection_groups = _contrast_role(
        role="selection",
        schedule=selection_schedule,
        ordinal_start=len(selection_ordinary),
    )
    return ContrastProviderDevelopmentProtocol(
        calibration_rule=SignedCalibrationRule(),
        rank_ladder=DEVELOPMENT_RANK_LADDER,
        candidate_ids=DEVELOPMENT_CANDIDATE_IDS,
        probes=(
            pilot_probes
            + fit_ordinary
            + fit_contrast
            + selection_ordinary
            + selection_contrast
        ),
        contrast_groups=pilot_groups + fit_groups + selection_groups,
    )
