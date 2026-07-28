"""Frozen prompt-blind probes for the Gemma L3-to-L4 reference provider.

This module contains specifications only.  It does not import a tensor
library, load an artifact, tokenize text, or call a model.  A model-specific
runner may later materialize the declared residual-state coordinates against
an authenticated, already-frozen Fisher basis package.

The three roles are intentionally separate:

* ``fit`` may fit provider coefficients from Rademacher, AR(1), and axis
  anchors;
* ``selection`` may choose the smallest passing candidate on seed-disjoint
  Rademacher and AR(1) probes; and
* ``assessment`` may only evaluate a sealed candidate on sparse, chirp, axis,
  radial-collision, and null-collision probes.

Radial and null collisions preserve the declared normalized modal direction
while changing information lost by RMS normalization.  They are therefore
architecture-identifiability controls, not additional fit rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math


__all__ = [
    "DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256",
    "DEFAULT_PROTOCOL_SHA256",
    "CandidateRatePoint",
    "ProbeRole",
    "SyntheticReferenceGates",
    "SyntheticReferenceProbe",
    "SyntheticReferenceProtocol",
    "assessment_panel_spec_sha256",
    "default_synthetic_reference_protocol",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_synthetic_reference_protocol.v2"
_FORMAT_VERSION = 2
_FROZEN_PROBE_DOMAIN_V1 = (
    b"fisher-graph:l3-l4-synthetic-reference-probe:v1\0"
)
_SELECTION_PROBE_DOMAIN_V2 = (
    b"fisher-graph:l3-l4-synthetic-reference-selection-probe:v2\0"
)
_PROTOCOL_DOMAIN = b"fisher-graph:l3-l4-synthetic-reference-protocol:v2\0"
_ASSESSMENT_PANEL_SPEC_DOMAIN = (
    b"fisher-graph:l3-l4-synthetic-reference-assessment-panel-spec:v1\0"
)
_MASK64 = (1 << 64) - 1
_MASK63 = (1 << 63) - 1

_MODAL_WIDTH = 64
_SEQUENCE_LENGTHS = (32, 72, 128, 256)
_OFFSET_FRACTIONS = ((0, 1), (1, 4), (1, 2), (3, 4))
_MODAL_AMPLITUDES = (0.05, 0.25, 0.5, 1.0)
_RADIAL_SCALES = (0.5, 1.0, 2.0)
_NULL_COORDINATES = (-1.0, 0.0, 1.0)
_AR_COEFFICIENTS = (0.25, 0.65, 0.9)
_AXIS_MODES = (0, 1, 2, 7, 15, 28, 42, 43)
_COLLISION_MODES = (0, 7, 28, 63)
_SPARSE_CARDINALITY = 4

_FIT_SEED = 20_260_728_031
_SELECTION_SEED = 20_260_728_071
_ASSESSMENT_SEED = 20_260_728_059

_FIT_RADEMACHER_COUNT = 32
_FIT_AR_COUNT = 32
_FIT_AXIS_COUNT = 16
_SELECTION_RADEMACHER_COUNT = 16
_SELECTION_AR_COUNT = 16
_ASSESSMENT_SPARSE_COUNT = 24
_ASSESSMENT_CHIRP_COUNT = 24
_ASSESSMENT_AXIS_COUNT = 16
_ASSESSMENT_RADIAL_COLLISION_COUNT = 12
_ASSESSMENT_NULL_COLLISION_COUNT = 12

_FAMILIES = {
    "rademacher",
    "ar1",
    "sparse",
    "chirp",
    "axis",
    "radial_collision",
    "null_collision",
}
_STOCHASTIC_FAMILIES = {"rademacher", "ar1", "sparse", "chirp"}
_ROLES = ("fit", "selection", "assessment")

_SCIENTIFIC_STATUS = {
    "scope": "synthetic_residual_state_probe_specifications_only",
    "prompt_blind_after_frozen_basis_package": True,
    "frozen_basis_package_is_upstream_prompt_conditioned": True,
    "prompt_text_loaded": False,
    "token_ids_loaded": False,
    "tokenizer_loaded": False,
    "natural_activation_rows_loaded": False,
    "score_gradient_rows_loaded": False,
    "prompt_local_kernel_loaded": False,
    "model_called": False,
    "artifact_loaded": False,
    "live_tensor_materialization_performed": False,
    "natural_prompt_fidelity_claim": False,
    "model_replacement_claim": False,
    "compression_claim": False,
    "speed_or_latency_claim": False,
}

_MATERIALIZATION_SEMANTICS = {
    "coordinate_system": (
        "frozen_L3_modal_coordinates_plus_explicit_pre_norm_radial_"
        "and_null_coordinates"
    ),
    "source_offset": "first_logical_position_at_which_direction_is_active",
    "modal_direction_normalization": (
        "per_active_logical_position_unit_L2_across_nonzero_modal_"
        "coordinates_before_modal_amplitude_sigma;modal_amplitude_is_"
        "the_standardized_radial_norm_not_a_per_mode_multiplier"
    ),
    "rademacher": (
        "splitmix64_sign_by_direction_seed_logical_position_and_modal_index"
    ),
    "ar1": (
        "unit_variance_AR1_along_logical_position_from_splitmix64_"
        "Rademacher_innovations"
    ),
    "sparse": "explicit_unique_logical_position_mode_sign_coordinates",
    "chirp": (
        "cosine_chirp_from_integer_temporal_and_modal_frequencies_"
        "and_phase_quadrant"
    ),
    "axis": "one_signed_modal_coordinate_at_source_offset",
    "radial_collision": (
        "same_normalized_modal_direction_with_distinct_pre_norm_radial_scale"
    ),
    "null_collision": (
        "same_normalized_modal_direction_with_distinct_RMSNorm_null_coordinate"
    ),
    "invalid_or_noninvertible_materialization": "fail_closed",
    "padding_and_causal_masks": "runner_must_materialize_declared_length_exactly",
}

# Literal trust anchor for the declaration built by
# :func:`default_synthetic_reference_protocol`.  Accidental drift fails during
# construction, before a model-specific runner can materialize any probe.
DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256 = (
    "c690e9f85f5629ab2701fc5db487aea1404864256f5fe24034e35143047af102"
)
DEFAULT_PROTOCOL_SHA256 = (
    "82b6d07830c3410a89f24233fc0d2ddfb0f3c1972739b6fe55144183485b3fb3"
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


def assessment_panel_spec_sha256(
    ordered_probe_artifact_sha256s: Sequence[str],
) -> str:
    """Hash only the ordered frozen assessment probe artifact identities."""

    if isinstance(
        ordered_probe_artifact_sha256s,
        (str, bytes),
    ) or not isinstance(ordered_probe_artifact_sha256s, Sequence):
        raise TypeError(
            "ordered assessment probe artifact SHA-256 values "
            "must be a sequence"
        )
    values = tuple(ordered_probe_artifact_sha256s)
    if len(values) != 88 or len(set(values)) != len(values):
        raise ValueError(
            "ordered assessment probe artifact SHA-256 values "
            "must contain 88 unique entries"
        )
    encoded: list[bytes] = []
    for value in values:
        if (
            type(value) is not str
            or len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "assessment probe artifact identity must be a lowercase "
                "SHA-256 digest"
            )
        try:
            encoded.append(bytes.fromhex(value))
        except ValueError as error:
            raise ValueError(
                "assessment probe artifact identity must be a lowercase "
                "SHA-256 digest"
            ) from error
    return hashlib.sha256(
        _ASSESSMENT_PANEL_SPEC_DOMAIN + b"".join(encoded)
    ).hexdigest()


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match frozen format")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _exact_int(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _finite_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(float(value))
    ):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _splitmix64(value: int) -> int:
    """One fully specified SplitMix64 step."""

    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _derived_seed(role_seed: int, family: str, index: int) -> int:
    family_digest = int.from_bytes(
        hashlib.sha256(family.encode("ascii")).digest()[:8],
        "big",
    )
    value = (
        role_seed
        ^ family_digest
        ^ ((index + 1) * 0xD1342543DE82EF95)
    )
    result = _splitmix64(value) & _MASK63
    return result or 1


def _offset(length: int, index: int) -> int:
    numerator, denominator = _OFFSET_FRACTIONS[index % len(_OFFSET_FRACTIONS)]
    return min(length - 1, (length * numerator) // denominator)


def _common_coordinates(
    index: int,
) -> tuple[int, int, float, float, float]:
    length = _SEQUENCE_LENGTHS[index % len(_SEQUENCE_LENGTHS)]
    source_offset = _offset(length, index // len(_SEQUENCE_LENGTHS))
    amplitude = _MODAL_AMPLITUDES[
        (index // 2) % len(_MODAL_AMPLITUDES)
    ]
    radial_scale = _RADIAL_SCALES[
        (index // 3) % len(_RADIAL_SCALES)
    ]
    null_coordinate = _NULL_COORDINATES[
        (index // 5) % len(_NULL_COORDINATES)
    ]
    return length, source_offset, amplitude, radial_scale, null_coordinate


def _unique_sparse_coordinates(
    *,
    seed: int,
    length: int,
    source_offset: int,
) -> tuple[tuple[int, int, int], ...]:
    available_positions = length - source_offset
    modulus = available_positions * _MODAL_WIDTH
    chosen: list[int] = []
    step = 0
    while len(chosen) < _SPARSE_CARDINALITY:
        candidate = _splitmix64(seed + step) % modulus
        if candidate not in chosen:
            chosen.append(candidate)
        step += 1
    coordinates = []
    for ordinal, flat in enumerate(chosen):
        logical_position = source_offset + flat // _MODAL_WIDTH
        mode = flat % _MODAL_WIDTH
        sign = 1 if _splitmix64(seed ^ (ordinal + 1)) & 1 else -1
        coordinates.append((logical_position, mode, sign))
    return tuple(sorted(coordinates))


@dataclass(frozen=True, slots=True)
class ProbeRole:
    """One frozen probe role and its mutation authority."""

    name: str
    seed: int
    expected_count: int
    families: tuple[str, ...]
    may_fit_coefficients: bool
    may_select_candidate: bool
    may_assess_sealed_candidate: bool
    requires_frozen_candidate: bool

    def __post_init__(self) -> None:
        if self.name not in _ROLES:
            raise ValueError("probe role name is invalid")
        _exact_int(self.seed, label="role seed", minimum=1)
        _exact_int(
            self.expected_count,
            label="role expected_count",
            minimum=1,
        )
        if (
            type(self.families) is not tuple
            or not self.families
            or len(set(self.families)) != len(self.families)
            or any(family not in _FAMILIES for family in self.families)
        ):
            raise ValueError("probe role families are invalid")
        authorities = (
            self.may_fit_coefficients,
            self.may_select_candidate,
            self.may_assess_sealed_candidate,
        )
        if any(type(value) is not bool for value in authorities):
            raise TypeError("probe role authorities must be booleans")
        if sum(authorities) != 1:
            raise ValueError("each probe role must have exactly one authority")
        if self.requires_frozen_candidate != (
            self.may_assess_sealed_candidate
        ):
            raise ValueError("only assessment may require a frozen candidate")

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "expected_count": self.expected_count,
            "families": list(self.families),
            "may_fit_coefficients": self.may_fit_coefficients,
            "may_select_candidate": self.may_select_candidate,
            "may_assess_sealed_candidate": (
                self.may_assess_sealed_candidate
            ),
            "requires_frozen_candidate": self.requires_frozen_candidate,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "ProbeRole":
        state = _mapping(raw, label="probe role")
        _strict_keys(
            state,
            expected={
                "name",
                "seed",
                "expected_count",
                "families",
                "may_fit_coefficients",
                "may_select_candidate",
                "may_assess_sealed_candidate",
                "requires_frozen_candidate",
            },
            label="probe role",
        )
        families = state["families"]
        if not isinstance(families, list):
            raise TypeError("probe role families must be a list")
        return cls(
            name=str(state["name"]),
            seed=_exact_int(state["seed"], label="role seed", minimum=1),
            expected_count=_exact_int(
                state["expected_count"],
                label="role expected_count",
                minimum=1,
            ),
            families=tuple(str(value) for value in families),
            may_fit_coefficients=state[  # type: ignore[arg-type]
                "may_fit_coefficients"
            ],
            may_select_candidate=state[  # type: ignore[arg-type]
                "may_select_candidate"
            ],
            may_assess_sealed_candidate=state[  # type: ignore[arg-type]
                "may_assess_sealed_candidate"
            ],
            requires_frozen_candidate=state[  # type: ignore[arg-type]
                "requires_frozen_candidate"
            ],
        )


@dataclass(frozen=True, slots=True)
class CandidateRatePoint:
    """One frozen provider rank candidate."""

    kind: str
    source_rank: int
    target_rank: int

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "spectral", "dense"}:
            raise ValueError("candidate kind is invalid")
        _exact_int(self.source_rank, label="source rank", minimum=0)
        _exact_int(self.target_rank, label="target rank", minimum=0)
        if self.kind == "constant" and (
            self.source_rank != 0 or self.target_rank != 0
        ):
            raise ValueError("constant candidate must have zero ranks")
        if self.kind != "constant" and (
            self.source_rank <= 0 or self.target_rank <= 0
        ):
            raise ValueError("nonconstant candidate ranks must be positive")
        if self.kind == "dense" and (
            self.source_rank != _MODAL_WIDTH
            or self.target_rank != _MODAL_WIDTH
        ):
            raise ValueError("dense candidate must retain the full modal width")

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "CandidateRatePoint":
        state = _mapping(raw, label="candidate rate point")
        _strict_keys(
            state,
            expected={"kind", "source_rank", "target_rank"},
            label="candidate rate point",
        )
        return cls(
            kind=str(state["kind"]),
            source_rank=_exact_int(
                state["source_rank"],
                label="source rank",
                minimum=0,
            ),
            target_rank=_exact_int(
                state["target_rank"],
                label="target rank",
                minimum=0,
            ),
        )


@dataclass(frozen=True, slots=True)
class SyntheticReferenceGates:
    """Frozen gates applied without rounding by a model-specific runner."""

    maximum_fisher_weighted_relative_error: float = 0.225
    minimum_reference_cosine: float = 0.975
    minimum_error_reduction_vs_constant: float = 0.10
    minimum_error_reduction_vs_position_only: float = 0.10
    maximum_per_probe_p90_relative_error: float = 0.35
    maximum_worst_panel_relative_error: float = 0.35
    maximum_prepared_vs_analytic_relative_error: float = 1e-5
    maximum_causality_violation: float = 1e-6
    maximum_padding_violation: float = 1e-6
    maximum_repeat_relative_error: float = 1e-7
    minimum_collision_target_relative_difference: float = 0.01
    minimum_in_support_fraction: float = 0.99

    def __post_init__(self) -> None:
        unit_interval = (
            self.minimum_reference_cosine,
            self.minimum_error_reduction_vs_constant,
            self.minimum_error_reduction_vs_position_only,
            self.minimum_in_support_fraction,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in unit_interval
        ):
            raise ValueError("unit-interval reference gates are invalid")
        nonnegative = (
            self.maximum_fisher_weighted_relative_error,
            self.maximum_per_probe_p90_relative_error,
            self.maximum_worst_panel_relative_error,
            self.maximum_prepared_vs_analytic_relative_error,
            self.maximum_causality_violation,
            self.maximum_padding_violation,
            self.maximum_repeat_relative_error,
            self.minimum_collision_target_relative_difference,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("nonnegative reference gates are invalid")

    def state_dict(self) -> dict[str, float]:
        return {
            "maximum_fisher_weighted_relative_error": (
                self.maximum_fisher_weighted_relative_error
            ),
            "minimum_reference_cosine": self.minimum_reference_cosine,
            "minimum_error_reduction_vs_constant": (
                self.minimum_error_reduction_vs_constant
            ),
            "minimum_error_reduction_vs_position_only": (
                self.minimum_error_reduction_vs_position_only
            ),
            "maximum_per_probe_p90_relative_error": (
                self.maximum_per_probe_p90_relative_error
            ),
            "maximum_worst_panel_relative_error": (
                self.maximum_worst_panel_relative_error
            ),
            "maximum_prepared_vs_analytic_relative_error": (
                self.maximum_prepared_vs_analytic_relative_error
            ),
            "maximum_causality_violation": self.maximum_causality_violation,
            "maximum_padding_violation": self.maximum_padding_violation,
            "maximum_repeat_relative_error": (
                self.maximum_repeat_relative_error
            ),
            "minimum_collision_target_relative_difference": (
                self.minimum_collision_target_relative_difference
            ),
            "minimum_in_support_fraction": self.minimum_in_support_fraction,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "SyntheticReferenceGates":
        state = _mapping(raw, label="synthetic reference gates")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="synthetic reference gates",
        )
        return cls(
            **{
                key: _finite_float(state[key], label=f"gate {key}")
                for key in sorted(expected)
            }
        )


@dataclass(frozen=True, slots=True)
class SyntheticReferenceProbe:
    """One lightweight residual-state direction specification."""

    role: str
    ordinal: int
    family: str
    sequence_length: int
    source_offset: int
    modal_amplitude: float
    radial_scale: float
    null_coordinate: float
    direction_seed: int
    axis_mode: int | None = None
    axis_sign: int | None = None
    ar_coefficient: float | None = None
    sparse_coordinates: tuple[tuple[int, int, int], ...] = ()
    chirp_temporal_frequency: int | None = None
    chirp_modal_frequency: int | None = None
    chirp_phase_quadrant: int | None = None
    collision_group: str | None = None
    collision_variant: str | None = None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError("probe role is invalid")
        _exact_int(self.ordinal, label="probe ordinal", minimum=0)
        if self.family not in _FAMILIES:
            raise ValueError("probe family is invalid")
        if self.sequence_length not in _SEQUENCE_LENGTHS:
            raise ValueError("probe sequence length is not frozen")
        if (
            type(self.source_offset) is not int
            or not 0 <= self.source_offset < self.sequence_length
        ):
            raise ValueError("probe source offset is invalid")
        if self.modal_amplitude not in _MODAL_AMPLITUDES:
            raise ValueError("probe modal amplitude is not frozen")
        if self.radial_scale not in _RADIAL_SCALES:
            raise ValueError("probe radial scale is not frozen")
        if self.null_coordinate not in _NULL_COORDINATES:
            raise ValueError("probe null coordinate is not frozen")
        _exact_int(self.direction_seed, label="direction seed", minimum=1)

        axis_family = self.family in {
            "axis",
            "radial_collision",
            "null_collision",
        }
        if axis_family:
            if (
                type(self.axis_mode) is not int
                or not 0 <= self.axis_mode < _MODAL_WIDTH
                or self.axis_sign not in (-1, 1)
            ):
                raise ValueError("axis probe coordinates are invalid")
        elif self.axis_mode is not None or self.axis_sign is not None:
            raise ValueError("non-axis probe cannot declare an axis")

        if self.family == "ar1":
            if self.ar_coefficient not in _AR_COEFFICIENTS:
                raise ValueError("AR coefficient is not frozen")
        elif self.ar_coefficient is not None:
            raise ValueError("non-AR probe cannot declare an AR coefficient")

        if self.family == "sparse":
            if (
                type(self.sparse_coordinates) is not tuple
                or len(self.sparse_coordinates) != _SPARSE_CARDINALITY
                or len(set(self.sparse_coordinates))
                != len(self.sparse_coordinates)
            ):
                raise ValueError("sparse probe cardinality is invalid")
            for position, mode, sign in self.sparse_coordinates:
                if (
                    type(position) is not int
                    or not self.source_offset <= position < self.sequence_length
                    or type(mode) is not int
                    or not 0 <= mode < _MODAL_WIDTH
                    or sign not in (-1, 1)
                ):
                    raise ValueError("sparse probe coordinate is invalid")
        elif self.sparse_coordinates:
            raise ValueError("nonsparse probe cannot declare sparse coordinates")

        chirp_values = (
            self.chirp_temporal_frequency,
            self.chirp_modal_frequency,
            self.chirp_phase_quadrant,
        )
        if self.family == "chirp":
            if (
                type(chirp_values[0]) is not int
                or not 1 <= chirp_values[0] <= self.sequence_length
                or type(chirp_values[1]) is not int
                or not 1 <= chirp_values[1] < _MODAL_WIDTH
                or type(chirp_values[2]) is not int
                or chirp_values[2] not in (0, 1, 2, 3)
            ):
                raise ValueError("chirp parameters are invalid")
        elif any(value is not None for value in chirp_values):
            raise ValueError("nonchirp probe cannot declare chirp parameters")

        is_collision = self.family in {
            "radial_collision",
            "null_collision",
        }
        is_axis_assessment = self.family == "axis" and self.role == "assessment"
        if is_collision or is_axis_assessment:
            if (
                not isinstance(self.collision_group, str)
                or not self.collision_group
                or not isinstance(self.collision_variant, str)
                or not self.collision_variant
            ):
                raise ValueError("control probe lacks collision identity")
        elif (
            self.collision_group is not None
            or self.collision_variant is not None
        ):
            raise ValueError("noncontrol probe cannot declare collision fields")

        probe_domain = (
            _SELECTION_PROBE_DOMAIN_V2
            if self.role == "selection"
            else _FROZEN_PROBE_DOMAIN_V1
        )
        computed = _digest(self._payload(), domain=probe_domain)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("synthetic probe hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def probe_id(self) -> str:
        return (
            f"{self.role}.{self.ordinal:04d}.{self.family}."
            f"{self.artifact_sha256[:12]}"
        )

    def _payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "ordinal": self.ordinal,
            "family": self.family,
            "sequence_length": self.sequence_length,
            "source_offset": self.source_offset,
            "modal_amplitude": self.modal_amplitude,
            "radial_scale": self.radial_scale,
            "null_coordinate": self.null_coordinate,
            "direction_seed": self.direction_seed,
            "axis_mode": self.axis_mode,
            "axis_sign": self.axis_sign,
            "ar_coefficient": self.ar_coefficient,
            "sparse_coordinates": [
                list(coordinate) for coordinate in self.sparse_coordinates
            ],
            "chirp_temporal_frequency": self.chirp_temporal_frequency,
            "chirp_modal_frequency": self.chirp_modal_frequency,
            "chirp_phase_quadrant": self.chirp_phase_quadrant,
            "collision_group": self.collision_group,
            "collision_variant": self.collision_variant,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "probe_id": self.probe_id,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "SyntheticReferenceProbe":
        state = _mapping(raw, label="synthetic probe")
        expected = {
            "role",
            "ordinal",
            "family",
            "sequence_length",
            "source_offset",
            "modal_amplitude",
            "radial_scale",
            "null_coordinate",
            "direction_seed",
            "axis_mode",
            "axis_sign",
            "ar_coefficient",
            "sparse_coordinates",
            "chirp_temporal_frequency",
            "chirp_modal_frequency",
            "chirp_phase_quadrant",
            "collision_group",
            "collision_variant",
            "probe_id",
            "artifact_sha256",
        }
        _strict_keys(state, expected=expected, label="synthetic probe")
        sparse = state["sparse_coordinates"]
        if not isinstance(sparse, list):
            raise TypeError("sparse_coordinates must be a list")
        coordinates = []
        for value in sparse:
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("sparse coordinate must have three integers")
            coordinates.append(
                tuple(
                    _exact_int(item, label="sparse coordinate")
                    for item in value
                )
            )
        optional_ints = {}
        for key in (
            "axis_mode",
            "axis_sign",
            "chirp_temporal_frequency",
            "chirp_modal_frequency",
            "chirp_phase_quadrant",
        ):
            value = state[key]
            optional_ints[key] = (
                None
                if value is None
                else _exact_int(value, label=f"probe {key}")
            )
        ar = state["ar_coefficient"]
        probe = cls(
            role=str(state["role"]),
            ordinal=_exact_int(
                state["ordinal"],
                label="probe ordinal",
                minimum=0,
            ),
            family=str(state["family"]),
            sequence_length=_exact_int(
                state["sequence_length"],
                label="sequence length",
                minimum=1,
            ),
            source_offset=_exact_int(
                state["source_offset"],
                label="source offset",
                minimum=0,
            ),
            modal_amplitude=_finite_float(
                state["modal_amplitude"],
                label="modal amplitude",
                minimum=0.0,
            ),
            radial_scale=_finite_float(
                state["radial_scale"],
                label="radial scale",
                minimum=0.0,
            ),
            null_coordinate=_finite_float(
                state["null_coordinate"],
                label="null coordinate",
            ),
            direction_seed=_exact_int(
                state["direction_seed"],
                label="direction seed",
                minimum=1,
            ),
            axis_mode=optional_ints["axis_mode"],
            axis_sign=optional_ints["axis_sign"],
            ar_coefficient=(
                None
                if ar is None
                else _finite_float(ar, label="AR coefficient")
            ),
            sparse_coordinates=tuple(coordinates),  # type: ignore[arg-type]
            chirp_temporal_frequency=optional_ints[
                "chirp_temporal_frequency"
            ],
            chirp_modal_frequency=optional_ints["chirp_modal_frequency"],
            chirp_phase_quadrant=optional_ints["chirp_phase_quadrant"],
            collision_group=(
                None
                if state["collision_group"] is None
                else str(state["collision_group"])
            ),
            collision_variant=(
                None
                if state["collision_variant"] is None
                else str(state["collision_variant"])
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )
        if state["probe_id"] != probe.probe_id:
            raise ValueError("synthetic probe id mismatch")
        return probe


def _build_fit_probes() -> tuple[SyntheticReferenceProbe, ...]:
    probes: list[SyntheticReferenceProbe] = []

    def ordinal() -> int:
        return len(probes)

    for index in range(_FIT_RADEMACHER_COUNT):
        length, offset, amplitude, radial, null = _common_coordinates(index)
        probes.append(
            SyntheticReferenceProbe(
                role="fit",
                ordinal=ordinal(),
                family="rademacher",
                sequence_length=length,
                source_offset=offset,
                modal_amplitude=amplitude,
                radial_scale=radial,
                null_coordinate=null,
                direction_seed=_derived_seed(
                    _FIT_SEED,
                    "rademacher",
                    index,
                ),
            )
        )
    for index in range(_FIT_AR_COUNT):
        length, offset, amplitude, radial, null = _common_coordinates(
            index + 11
        )
        probes.append(
            SyntheticReferenceProbe(
                role="fit",
                ordinal=ordinal(),
                family="ar1",
                sequence_length=length,
                source_offset=offset,
                modal_amplitude=amplitude,
                radial_scale=radial,
                null_coordinate=null,
                direction_seed=_derived_seed(_FIT_SEED, "ar1", index),
                ar_coefficient=_AR_COEFFICIENTS[
                    index % len(_AR_COEFFICIENTS)
                ],
            )
        )
    for group_index, mode in enumerate(_AXIS_MODES):
        length, offset, amplitude, radial, null = _common_coordinates(
            group_index + 23
        )
        for sign_index, sign in enumerate((1, -1)):
            index = group_index * 2 + sign_index
            probes.append(
                SyntheticReferenceProbe(
                    role="fit",
                    ordinal=ordinal(),
                    family="axis",
                    sequence_length=length,
                    source_offset=offset,
                    modal_amplitude=amplitude,
                    radial_scale=radial,
                    null_coordinate=null,
                    direction_seed=_derived_seed(_FIT_SEED, "axis", index),
                    axis_mode=mode,
                    axis_sign=sign,
                )
            )
    return tuple(probes)


def _build_selection_probes() -> tuple[SyntheticReferenceProbe, ...]:
    probes: list[SyntheticReferenceProbe] = []

    def ordinal() -> int:
        return len(probes)

    for index in range(_SELECTION_RADEMACHER_COUNT):
        length, offset, amplitude, radial, null = _common_coordinates(
            index + 37
        )
        probes.append(
            SyntheticReferenceProbe(
                role="selection",
                ordinal=ordinal(),
                family="rademacher",
                sequence_length=length,
                source_offset=offset,
                modal_amplitude=amplitude,
                radial_scale=radial,
                null_coordinate=null,
                direction_seed=_derived_seed(
                    _SELECTION_SEED,
                    "rademacher",
                    index,
                ),
            )
        )
    for index in range(_SELECTION_AR_COUNT):
        length, offset, amplitude, radial, null = _common_coordinates(
            index + 53
        )
        probes.append(
            SyntheticReferenceProbe(
                role="selection",
                ordinal=ordinal(),
                family="ar1",
                sequence_length=length,
                source_offset=offset,
                modal_amplitude=amplitude,
                radial_scale=radial,
                null_coordinate=null,
                direction_seed=_derived_seed(_SELECTION_SEED, "ar1", index),
                ar_coefficient=_AR_COEFFICIENTS[
                    (index + 1) % len(_AR_COEFFICIENTS)
                ],
            )
        )
    return tuple(probes)


def _build_assessment_probes() -> tuple[SyntheticReferenceProbe, ...]:
    probes: list[SyntheticReferenceProbe] = []

    def ordinal() -> int:
        return len(probes)

    for index in range(_ASSESSMENT_SPARSE_COUNT):
        length, offset, amplitude, radial, null = _common_coordinates(
            index + 71
        )
        seed = _derived_seed(_ASSESSMENT_SEED, "sparse", index)
        probes.append(
            SyntheticReferenceProbe(
                role="assessment",
                ordinal=ordinal(),
                family="sparse",
                sequence_length=length,
                source_offset=offset,
                modal_amplitude=amplitude,
                radial_scale=radial,
                null_coordinate=null,
                direction_seed=seed,
                sparse_coordinates=_unique_sparse_coordinates(
                    seed=seed,
                    length=length,
                    source_offset=offset,
                ),
            )
        )
    for index in range(_ASSESSMENT_CHIRP_COUNT):
        length, offset, amplitude, radial, null = _common_coordinates(
            index + 89
        )
        seed = _derived_seed(_ASSESSMENT_SEED, "chirp", index)
        active_length = length - offset
        probes.append(
            SyntheticReferenceProbe(
                role="assessment",
                ordinal=ordinal(),
                family="chirp",
                sequence_length=length,
                source_offset=offset,
                modal_amplitude=amplitude,
                radial_scale=radial,
                null_coordinate=null,
                direction_seed=seed,
                chirp_temporal_frequency=1 + seed % active_length,
                chirp_modal_frequency=1 + (seed >> 13) % (_MODAL_WIDTH - 1),
                chirp_phase_quadrant=(seed >> 29) % 4,
            )
        )
    for group_index, mode in enumerate(_AXIS_MODES):
        length, offset, amplitude, radial, _ = _common_coordinates(
            group_index + 107
        )
        for sign_index, sign in enumerate((1, -1)):
            index = group_index * 2 + sign_index
            probes.append(
                SyntheticReferenceProbe(
                    role="assessment",
                    ordinal=ordinal(),
                    family="axis",
                    sequence_length=length,
                    source_offset=offset,
                    modal_amplitude=amplitude,
                    radial_scale=radial,
                    null_coordinate=0.0,
                    direction_seed=_derived_seed(
                        _ASSESSMENT_SEED,
                        "axis",
                        index,
                    ),
                    axis_mode=mode,
                    axis_sign=sign,
                    collision_group=f"assessment.axis.mode_{mode:02d}",
                    collision_variant=(
                        "positive" if sign > 0 else "negative"
                    ),
                )
            )
    for group_index, mode in enumerate(_COLLISION_MODES):
        length = _SEQUENCE_LENGTHS[group_index]
        source_offset = _offset(length, group_index + 1)
        seed = _derived_seed(
            _ASSESSMENT_SEED,
            "radial_collision",
            group_index,
        )
        for radial_scale in _RADIAL_SCALES:
            probes.append(
                SyntheticReferenceProbe(
                    role="assessment",
                    ordinal=ordinal(),
                    family="radial_collision",
                    sequence_length=length,
                    source_offset=source_offset,
                    modal_amplitude=0.5,
                    radial_scale=radial_scale,
                    null_coordinate=0.0,
                    direction_seed=seed,
                    axis_mode=mode,
                    axis_sign=1,
                    collision_group=(
                        f"assessment.radial.mode_{mode:02d}"
                    ),
                    collision_variant=f"radial_{radial_scale:g}",
                )
            )
    for group_index, mode in enumerate(_COLLISION_MODES):
        length = _SEQUENCE_LENGTHS[group_index]
        source_offset = _offset(length, group_index + 2)
        seed = _derived_seed(
            _ASSESSMENT_SEED,
            "null_collision",
            group_index,
        )
        for null_coordinate in _NULL_COORDINATES:
            probes.append(
                SyntheticReferenceProbe(
                    role="assessment",
                    ordinal=ordinal(),
                    family="null_collision",
                    sequence_length=length,
                    source_offset=source_offset,
                    modal_amplitude=0.5,
                    radial_scale=1.0,
                    null_coordinate=null_coordinate,
                    direction_seed=seed,
                    axis_mode=mode,
                    axis_sign=1,
                    collision_group=f"assessment.null.mode_{mode:02d}",
                    collision_variant=f"null_{null_coordinate:g}",
                )
            )
    return tuple(probes)


@dataclass(frozen=True, slots=True)
class SyntheticReferenceProtocol:
    """Authenticated, immutable synthetic provider protocol."""

    roles: tuple[ProbeRole, ...]
    candidate_ladder: tuple[CandidateRatePoint, ...]
    gates: SyntheticReferenceGates
    probes: tuple[SyntheticReferenceProbe, ...]
    protocol_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.roles) is not tuple
            or tuple(role.name for role in self.roles) != _ROLES
        ):
            raise ValueError("protocol roles are not frozen")
        if len({role.seed for role in self.roles}) != len(self.roles):
            raise ValueError("protocol role seeds must be disjoint")
        if (
            type(self.candidate_ladder) is not tuple
            or not self.candidate_ladder
            or self.candidate_ladder[0].kind != "constant"
            or self.candidate_ladder[-1].kind != "dense"
        ):
            raise ValueError("candidate ladder endpoints are invalid")
        if not isinstance(self.gates, SyntheticReferenceGates):
            raise TypeError("protocol gates are invalid")
        if type(self.probes) is not tuple or not self.probes:
            raise ValueError("protocol probes cannot be empty")

        hashes = [probe.artifact_sha256 for probe in self.probes]
        if len(set(hashes)) != len(hashes):
            raise ValueError("protocol probe hashes must be unique")
        for role in self.roles:
            role_probes = [
                probe for probe in self.probes if probe.role == role.name
            ]
            if len(role_probes) != role.expected_count:
                raise ValueError(f"{role.name} probe count drifted")
            if tuple(probe.ordinal for probe in role_probes) != tuple(
                range(role.expected_count)
            ):
                raise ValueError(f"{role.name} probe ordinals drifted")
            if set(probe.family for probe in role_probes) != set(role.families):
                raise ValueError(f"{role.name} probe families drifted")
        if (
            self.assessment_panel_spec_sha256
            != DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256
        ):
            raise ValueError(
                "synthetic assessment panel specification drifted"
            )

        stochastic_by_role = {
            role: {
                probe.direction_seed
                for probe in self.probes
                if probe.role == role
                and probe.family in _STOCHASTIC_FAMILIES
            }
            for role in _ROLES
        }
        for left_index, left in enumerate(_ROLES):
            for right in _ROLES[left_index + 1 :]:
                if stochastic_by_role[left] & stochastic_by_role[right]:
                    raise ValueError("stochastic probe seeds cross role boundaries")

        computed = _digest(self._payload(), domain=_PROTOCOL_DOMAIN)
        if self.protocol_sha256:
            if self.protocol_sha256 != computed:
                raise ValueError("synthetic reference protocol hash mismatch")
        else:
            object.__setattr__(self, "protocol_sha256", computed)
        if (
            DEFAULT_PROTOCOL_SHA256
            and self.protocol_sha256 != DEFAULT_PROTOCOL_SHA256
        ):
            raise ValueError("synthetic reference protocol differs from default")

    @property
    def assessment_panel_spec_sha256(self) -> str:
        return assessment_panel_spec_sha256(
            tuple(
                probe.artifact_sha256
                for probe in self.probes
                if probe.role == "assessment"
            )
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "assessment_panel_spec_sha256": (
                self.assessment_panel_spec_sha256
            ),
            "scientific_status": dict(_SCIENTIFIC_STATUS),
            "geometry": {
                "modal_width": _MODAL_WIDTH,
                "sequence_lengths": list(_SEQUENCE_LENGTHS),
                "offset_fractions": [
                    list(value) for value in _OFFSET_FRACTIONS
                ],
                "modal_amplitudes_sigma": list(_MODAL_AMPLITUDES),
                "pre_norm_radial_scale_multipliers": list(_RADIAL_SCALES),
                "rmsnorm_null_coordinates": list(_NULL_COORDINATES),
                "sparse_cardinality": _SPARSE_CARDINALITY,
            },
            "materialization_semantics": dict(_MATERIALIZATION_SEMANTICS),
            "roles": [role.state_dict() for role in self.roles],
            "candidate_selection": {
                "rule": (
                    "minimum_stored_coefficients_then_source_rank_"
                    "then_target_rank"
                ),
                "fit_all_rows_before_opening_selection": True,
                "assessment_may_change_candidate": False,
                "gate_values_applied_without_rounding": True,
                "ladder": [
                    value.state_dict() for value in self.candidate_ladder
                ],
            },
            "gates": self.gates.state_dict(),
            "probes": [probe.state_dict() for probe in self.probes],
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "protocol_sha256": self.protocol_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "SyntheticReferenceProtocol":
        state = _mapping(raw, label="synthetic reference protocol")
        _strict_keys(
            state,
            expected={
                "schema",
                "format_version",
                "assessment_panel_spec_sha256",
                "scientific_status",
                "geometry",
                "materialization_semantics",
                "roles",
                "candidate_selection",
                "gates",
                "probes",
                "protocol_sha256",
            },
            label="synthetic reference protocol",
        )
        if (
            state["schema"] != _SCHEMA
            or state["format_version"] != _FORMAT_VERSION
            or state["assessment_panel_spec_sha256"]
            != DEFAULT_ASSESSMENT_PANEL_SPEC_SHA256
            or state["scientific_status"] != _SCIENTIFIC_STATUS
            or state["materialization_semantics"]
            != _MATERIALIZATION_SEMANTICS
        ):
            raise ValueError("synthetic reference protocol declaration drifted")
        geometry = _mapping(state["geometry"], label="protocol geometry")
        expected_geometry = default_synthetic_reference_protocol().state_dict()[
            "geometry"
        ]
        if dict(geometry) != expected_geometry:
            raise ValueError("synthetic reference geometry drifted")
        roles = state["roles"]
        probes = state["probes"]
        selection = _mapping(
            state["candidate_selection"],
            label="candidate selection",
        )
        _strict_keys(
            selection,
            expected={
                "rule",
                "fit_all_rows_before_opening_selection",
                "assessment_may_change_candidate",
                "gate_values_applied_without_rounding",
                "ladder",
            },
            label="candidate selection",
        )
        if (
            selection["rule"]
            != "minimum_stored_coefficients_then_source_rank_then_target_rank"
            or selection["fit_all_rows_before_opening_selection"] is not True
            or selection["assessment_may_change_candidate"] is not False
            or selection["gate_values_applied_without_rounding"] is not True
            or not isinstance(roles, list)
            or not isinstance(probes, list)
            or not isinstance(selection["ladder"], list)
        ):
            raise ValueError("candidate selection declaration drifted")
        return cls(
            roles=tuple(ProbeRole.from_state_dict(value) for value in roles),
            candidate_ladder=tuple(
                CandidateRatePoint.from_state_dict(value)
                for value in selection["ladder"]
            ),
            gates=SyntheticReferenceGates.from_state_dict(state["gates"]),
            probes=tuple(
                SyntheticReferenceProbe.from_state_dict(value)
                for value in probes
            ),
            protocol_sha256=str(state["protocol_sha256"]),
        )


def default_synthetic_reference_protocol() -> SyntheticReferenceProtocol:
    """Return the one frozen prompt-blind synthetic reference protocol."""

    fit = _build_fit_probes()
    selection = _build_selection_probes()
    assessment = _build_assessment_probes()
    return SyntheticReferenceProtocol(
        roles=(
            ProbeRole(
                name="fit",
                seed=_FIT_SEED,
                expected_count=len(fit),
                families=("rademacher", "ar1", "axis"),
                may_fit_coefficients=True,
                may_select_candidate=False,
                may_assess_sealed_candidate=False,
                requires_frozen_candidate=False,
            ),
            ProbeRole(
                name="selection",
                seed=_SELECTION_SEED,
                expected_count=len(selection),
                families=("rademacher", "ar1"),
                may_fit_coefficients=False,
                may_select_candidate=True,
                may_assess_sealed_candidate=False,
                requires_frozen_candidate=False,
            ),
            ProbeRole(
                name="assessment",
                seed=_ASSESSMENT_SEED,
                expected_count=len(assessment),
                families=(
                    "sparse",
                    "chirp",
                    "axis",
                    "radial_collision",
                    "null_collision",
                ),
                may_fit_coefficients=False,
                may_select_candidate=False,
                may_assess_sealed_candidate=True,
                requires_frozen_candidate=True,
            ),
        ),
        candidate_ladder=(
            CandidateRatePoint("constant", 0, 0),
            CandidateRatePoint("spectral", 8, 8),
            CandidateRatePoint("spectral", 16, 16),
            CandidateRatePoint("spectral", 24, 24),
            CandidateRatePoint("spectral", 32, 32),
            CandidateRatePoint("spectral", 48, 48),
            CandidateRatePoint("dense", 64, 64),
        ),
        gates=SyntheticReferenceGates(),
        probes=fit + selection + assessment,
    )
