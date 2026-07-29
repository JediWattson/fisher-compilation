"""Pure declaration for the fit-only Gemma L3/L4 objective-balance rung.

This diagnostic may replay the consumed C2 pilot to authenticate its selected
amplitude and may measure the C2 fit panel.  It must never materialize,
measure, load, or score the consumed C2 selection panel.  A pass only locks a
training recipe for a later, fresh C3 protocol; it is not held-out evidence.

The declaration intentionally imports no tensor or model library.  In
particular, the committed C2 implementation and its code-bundle hash remain
untouched.
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
    "C2ObjectiveBalanceProvenance",
    "DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256",
    "DIAGNOSTIC_EXECUTION_DEVICE",
    "DIAGNOSTIC_EXECUTION_DTYPE",
    "DIAGNOSTIC_LATENT_RANK",
    "DIAGNOSTIC_LEARNING_RATE",
    "DIAGNOSTIC_PRIMARY_SEED",
    "DIAGNOSTIC_REPLICATION_SEED",
    "DIAGNOSTIC_RECIPE_STEPS",
    "OBJECTIVE_BALANCE_RECIPE_IDS",
    "ObjectiveBalanceDiagnosticGates",
    "ObjectiveBalanceDiagnosticProtocol",
    "ObjectiveBalanceRecipe",
    "default_objective_balance_diagnostic_protocol",
    "family_balance_copy_binding_sha256",
    "family_balance_copy_id",
]


TrainingMetric = Literal[
    "canonical_fisher",
    "fit_teacher_weighted_rms",
]

DIAGNOSTIC_LATENT_RANK = 16
DIAGNOSTIC_EXECUTION_DEVICE = "cpu"
DIAGNOSTIC_EXECUTION_DTYPE = "float32"
DIAGNOSTIC_PRIMARY_SEED = 20_260_728_402
DIAGNOSTIC_REPLICATION_SEED = 20_260_729_402
DIAGNOSTIC_LEARNING_RATE = 1e-3
DIAGNOSTIC_RECIPE_STEPS = (300, 300, 300, 600)
OBJECTIVE_BALANCE_RECIPE_IDS = (
    "d0_raw_c2_control",
    "d1_unit_rms",
    "d2_unit_rms_family_balanced",
    "d3_unit_rms_family_balanced_direction",
)
_FROZEN_RECIPE_SPECS = {
    "d0_raw_c2_control": {
        "training_metric": "canonical_fisher",
        "signed_pair_multiplicity": 1,
        "direction_weight": 0.5,
        "steps": 300,
        "advancement_eligible": False,
    },
    "d1_unit_rms": {
        "training_metric": "fit_teacher_weighted_rms",
        "signed_pair_multiplicity": 1,
        "direction_weight": 0.5,
        "steps": 300,
        "advancement_eligible": True,
    },
    "d2_unit_rms_family_balanced": {
        "training_metric": "fit_teacher_weighted_rms",
        "signed_pair_multiplicity": 2,
        "direction_weight": 0.5,
        "steps": 300,
        "advancement_eligible": True,
    },
    "d3_unit_rms_family_balanced_direction": {
        "training_metric": "fit_teacher_weighted_rms",
        "signed_pair_multiplicity": 2,
        "direction_weight": 2.0,
        "steps": 600,
        "advancement_eligible": True,
    },
}

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_objective_balance_diagnostic."
    "d0_d3_protocol.v1"
)
_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECIPE_DOMAIN = b"fisher-graph:objective-balance-diagnostic:recipe:v1\0"
_GATES_DOMAIN = b"fisher-graph:objective-balance-diagnostic:gates:v1\0"
_PROVENANCE_DOMAIN = (
    b"fisher-graph:objective-balance-diagnostic:c2-provenance:v1\0"
)
_PROTOCOL_DOMAIN = b"fisher-graph:objective-balance-diagnostic:protocol:v1\0"
_PAIR_COPY_DOMAIN = (
    b"fisher-graph:objective-balance-diagnostic:signed-pair-copy:v1\0"
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


def _nonnegative(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


@dataclass(frozen=True, slots=True)
class ObjectiveBalanceRecipe:
    """One deterministic rank-16 fit recipe."""

    recipe_id: str
    training_metric: TrainingMetric
    signed_pair_multiplicity: int
    pointwise_weight: float
    sensitivity_relative_delta_weight: float
    direction_weight: float
    midpoint_jvp_weight: float
    intended_null_weight: float
    steps: int
    learning_rate: float
    primary_seed: int
    advancement_eligible: bool
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.recipe_id, str)
            or self.recipe_id not in OBJECTIVE_BALANCE_RECIPE_IDS
        ):
            raise ValueError("objective-balance recipe id is invalid")
        if self.training_metric not in (
            "canonical_fisher",
            "fit_teacher_weighted_rms",
        ):
            raise ValueError("objective-balance training metric is invalid")
        object.__setattr__(
            self,
            "signed_pair_multiplicity",
            _exact_int(
                self.signed_pair_multiplicity,
                label="signed pair multiplicity",
                minimum=1,
            ),
        )
        for name in (
            "pointwise_weight",
            "sensitivity_relative_delta_weight",
            "direction_weight",
            "midpoint_jvp_weight",
            "intended_null_weight",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), label=name),
            )
        if not any(
            getattr(self, name) > 0.0
            for name in (
                "pointwise_weight",
                "sensitivity_relative_delta_weight",
                "direction_weight",
                "midpoint_jvp_weight",
                "intended_null_weight",
            )
        ):
            raise ValueError("recipe must have an active objective component")
        object.__setattr__(
            self,
            "steps",
            _exact_int(self.steps, label="training steps", minimum=1),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive(self.learning_rate, label="learning rate"),
        )
        object.__setattr__(
            self,
            "primary_seed",
            _exact_int(self.primary_seed, label="primary seed", minimum=1),
        )
        _bool(self.advancement_eligible, label="advancement eligibility")
        frozen = _FROZEN_RECIPE_SPECS[self.recipe_id]
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{self.recipe_id} {name} differs from its frozen value"
                )
        if (
            self.pointwise_weight != 1.0
            or self.sensitivity_relative_delta_weight != 2.0
            or self.midpoint_jvp_weight != 1.0
            or self.intended_null_weight != 1.0
            or self.learning_rate != DIAGNOSTIC_LEARNING_RATE
            or self.primary_seed != DIAGNOSTIC_PRIMARY_SEED
        ):
            raise ValueError("objective-balance common recipe values drifted")
        object.__setattr__(
            self,
            "artifact_sha256",
            self._digest_or_validate(self.artifact_sha256),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.objective_balance_recipe",
            "format_version": _FORMAT_VERSION,
            "recipe_id": self.recipe_id,
            "training_metric": self.training_metric,
            "signed_pair_multiplicity": self.signed_pair_multiplicity,
            "pointwise_weight": self.pointwise_weight,
            "sensitivity_relative_delta_weight": (
                self.sensitivity_relative_delta_weight
            ),
            "direction_weight": self.direction_weight,
            "midpoint_jvp_weight": self.midpoint_jvp_weight,
            "intended_null_weight": self.intended_null_weight,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "primary_seed": self.primary_seed,
            "advancement_eligible": self.advancement_eligible,
        }

    def _digest_or_validate(self, supplied: str) -> str:
        computed = _digest(self._payload(), domain=_RECIPE_DOMAIN)
        if supplied:
            if _require_sha256(supplied, label="recipe artifact") != computed:
                raise ValueError("objective-balance recipe hash mismatch")
            return supplied
        return computed

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "ObjectiveBalanceRecipe":
        state = _mapping(raw, label="objective-balance recipe")
        expected = set(cls(
            recipe_id=OBJECTIVE_BALANCE_RECIPE_IDS[0],
            training_metric="canonical_fisher",
            signed_pair_multiplicity=1,
            pointwise_weight=1.0,
            sensitivity_relative_delta_weight=2.0,
            direction_weight=0.5,
            midpoint_jvp_weight=1.0,
            intended_null_weight=1.0,
            steps=300,
            learning_rate=DIAGNOSTIC_LEARNING_RATE,
            primary_seed=DIAGNOSTIC_PRIMARY_SEED,
            advancement_eligible=False,
        ).state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="objective-balance recipe",
        )
        if (
            state["artifact_kind"]
            != "fisher_graph.objective_balance_recipe"
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("objective-balance recipe semantics drifted")
        return cls(
            recipe_id=str(state["recipe_id"]),
            training_metric=str(state["training_metric"]),  # type: ignore[arg-type]
            signed_pair_multiplicity=_exact_int(
                state["signed_pair_multiplicity"],
                label="signed pair multiplicity",
                minimum=1,
            ),
            pointwise_weight=_nonnegative(
                state["pointwise_weight"],
                label="pointwise weight",
            ),
            sensitivity_relative_delta_weight=_nonnegative(
                state["sensitivity_relative_delta_weight"],
                label="sensitivity relative delta weight",
            ),
            direction_weight=_nonnegative(
                state["direction_weight"],
                label="direction weight",
            ),
            midpoint_jvp_weight=_nonnegative(
                state["midpoint_jvp_weight"],
                label="midpoint JVP weight",
            ),
            intended_null_weight=_nonnegative(
                state["intended_null_weight"],
                label="intended-null weight",
            ),
            steps=_exact_int(
                state["steps"],
                label="training steps",
                minimum=1,
            ),
            learning_rate=_positive(
                state["learning_rate"],
                label="learning rate",
            ),
            primary_seed=_exact_int(
                state["primary_seed"],
                label="primary seed",
                minimum=1,
            ),
            advancement_eligible=_bool(
                state["advancement_eligible"],
                label="advancement eligibility",
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveBalanceDiagnosticGates:
    """Frozen fit-only optimization and advancement gates."""

    minimum_gauge_energy: float = 1e-12
    normalized_energy_absolute_tolerance: float = 1e-12
    minimum_teacher_mse_floor_multiple: float = 100.0
    minimum_initial_pointwise_share: float = 0.10
    maximum_initial_pointwise_share: float = 0.40
    minimum_initial_contrast_share: float = 0.50
    maximum_initial_active_component_share: float = 0.65
    required_ordinary_gate_count: int = 12
    required_null_candidate_pass_count: int = 24
    require_all_ordinary_gates: bool = True
    require_every_eligible_sensitivity_contrast: bool = True
    require_all_contrast_families: bool = True
    require_two_seed_pass: bool = True
    ordinary_gates_sha256: str = (
        "0ef08515366888ff11f83269c68fc154477202b64bf2308e6b68985b05e91cd5"
    )
    contrast_gates_sha256: str = (
        "bedf561b190b04f880aabad6020ddb680d187734d572c9ec1abba7573cad0de1"
    )
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "minimum_gauge_energy",
            "normalized_energy_absolute_tolerance",
            "minimum_teacher_mse_floor_multiple",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), label=name),
            )
        for name in (
            "minimum_initial_pointwise_share",
            "maximum_initial_pointwise_share",
            "minimum_initial_contrast_share",
            "maximum_initial_active_component_share",
        ):
            value = _finite(getattr(self, name), label=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        if (
            self.minimum_initial_pointwise_share
            > self.maximum_initial_pointwise_share
        ):
            raise ValueError("initial pointwise share interval is reversed")
        for name in (
            "required_ordinary_gate_count",
            "required_null_candidate_pass_count",
        ):
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), label=name, minimum=1),
            )
        for name in (
            "require_all_ordinary_gates",
            "require_every_eligible_sensitivity_contrast",
            "require_all_contrast_families",
            "require_two_seed_pass",
        ):
            _bool(getattr(self, name), label=name)
        for name in ("ordinary_gates_sha256", "contrast_gates_sha256"):
            _require_sha256(getattr(self, name), label=name)
        computed = _digest(self._payload(), domain=_GATES_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="diagnostic gates artifact",
                )
                != computed
            ):
                raise ValueError("objective-balance gate hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.objective_balance_diagnostic_gates",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "artifact_sha256"
            },
            "ordinary_decision": "all_existing_gate_flags_must_pass",
            "contrast_decision": (
                "null_radial_and_signed_families_must_formally_pass"
            ),
            "advancement_decision": (
                "first_eligible_recipe_passing_primary_and_replication_seeds"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "ObjectiveBalanceDiagnosticGates":
        state = _mapping(raw, label="objective-balance gates")
        expected = set(cls().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="objective-balance gates",
        )
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "ordinary_decision",
            "contrast_decision",
            "advancement_decision",
        ):
            if state[name] != defaults[name]:
                raise ValueError("objective-balance gate semantics drifted")
        return cls(
            **{
                name: state[name]
                for name in cls.__dataclass_fields__
            }
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class C2ObjectiveBalanceProvenance:
    """Literal C2 anchors and the consumed-selection firewall."""

    protocol_sha256: str
    pilot_panel_sha256: str
    fit_panel_sha256: str
    calibrated_fit_panel_sha256: str
    calibration_sha256: str
    selected_amplitude: float
    objective_sha256: str
    training_sha256: str
    forbidden_selection_panel_sha256s: tuple[str, ...]
    forbidden_probe_prefixes: tuple[str, ...]
    selection_materialization_allowed: bool = False
    c2_artifact_loading_allowed: bool = False
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "pilot_panel_sha256",
            "fit_panel_sha256",
            "calibrated_fit_panel_sha256",
            "calibration_sha256",
            "objective_sha256",
            "training_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        object.__setattr__(
            self,
            "selected_amplitude",
            _positive(self.selected_amplitude, label="selected amplitude"),
        )
        forbidden = tuple(self.forbidden_selection_panel_sha256s)
        if len(forbidden) != 2 or len(set(forbidden)) != 2:
            raise ValueError("exactly two selection panel hashes are denied")
        for value in forbidden:
            _require_sha256(value, label="forbidden selection panel")
        object.__setattr__(
            self,
            "forbidden_selection_panel_sha256s",
            forbidden,
        )
        prefixes = tuple(self.forbidden_probe_prefixes)
        if prefixes != ("development_c2.selection.",):
            raise ValueError("C2 selection probe denylist drifted")
        object.__setattr__(self, "forbidden_probe_prefixes", prefixes)
        if self.selection_materialization_allowed:
            raise ValueError("C2 selection materialization must be forbidden")
        if self.c2_artifact_loading_allowed:
            raise ValueError("loading the selection-bearing C2 artifact is forbidden")
        computed = _digest(self._payload(), domain=_PROVENANCE_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="C2 provenance artifact",
                )
                != computed
            ):
                raise ValueError("C2 provenance hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.objective_balance_c2_provenance",
            "format_version": _FORMAT_VERSION,
            "protocol_sha256": self.protocol_sha256,
            "pilot_panel_sha256": self.pilot_panel_sha256,
            "fit_panel_sha256": self.fit_panel_sha256,
            "calibrated_fit_panel_sha256": self.calibrated_fit_panel_sha256,
            "calibration_sha256": self.calibration_sha256,
            "selected_amplitude": self.selected_amplitude,
            "objective_sha256": self.objective_sha256,
            "training_sha256": self.training_sha256,
            "forbidden_selection_panel_sha256s": list(
                self.forbidden_selection_panel_sha256s
            ),
            "forbidden_probe_prefixes": list(self.forbidden_probe_prefixes),
            "selection_materialization_allowed": (
                self.selection_materialization_allowed
            ),
            "c2_artifact_loading_allowed": self.c2_artifact_loading_allowed,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "C2ObjectiveBalanceProvenance":
        state = _mapping(raw, label="C2 objective-balance provenance")
        expected = set(_default_c2_provenance().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="C2 objective-balance provenance",
        )
        if (
            state["artifact_kind"]
            != "fisher_graph.objective_balance_c2_provenance"
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("C2 objective-balance provenance semantics drifted")
        denied = state["forbidden_selection_panel_sha256s"]
        prefixes = state["forbidden_probe_prefixes"]
        if not isinstance(denied, Sequence) or isinstance(denied, (str, bytes)):
            raise TypeError("forbidden selection panels must be a sequence")
        if not isinstance(prefixes, Sequence) or isinstance(
            prefixes,
            (str, bytes),
        ):
            raise TypeError("forbidden probe prefixes must be a sequence")
        return cls(
            protocol_sha256=str(state["protocol_sha256"]),
            pilot_panel_sha256=str(state["pilot_panel_sha256"]),
            fit_panel_sha256=str(state["fit_panel_sha256"]),
            calibrated_fit_panel_sha256=str(
                state["calibrated_fit_panel_sha256"]
            ),
            calibration_sha256=str(state["calibration_sha256"]),
            selected_amplitude=_positive(
                state["selected_amplitude"],
                label="selected amplitude",
            ),
            objective_sha256=str(state["objective_sha256"]),
            training_sha256=str(state["training_sha256"]),
            forbidden_selection_panel_sha256s=tuple(
                str(value) for value in denied
            ),
            forbidden_probe_prefixes=tuple(str(value) for value in prefixes),
            selection_materialization_allowed=_bool(
                state["selection_materialization_allowed"],
                label="selection materialization allowed",
            ),
            c2_artifact_loading_allowed=_bool(
                state["c2_artifact_loading_allowed"],
                label="C2 artifact loading allowed",
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveBalanceDiagnosticProtocol:
    """Complete pure declaration for the rank-16 diagnostic."""

    latent_rank: int
    execution_device: str
    execution_dtype: str
    primary_seed: int
    replication_seed: int
    recipes: tuple[ObjectiveBalanceRecipe, ...]
    gates: ObjectiveBalanceDiagnosticGates
    c2_provenance: C2ObjectiveBalanceProvenance
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if _exact_int(self.latent_rank, label="latent rank", minimum=1) != (
            DIAGNOSTIC_LATENT_RANK
        ):
            raise ValueError("objective-balance latent rank is not frozen")
        if self.execution_device != DIAGNOSTIC_EXECUTION_DEVICE:
            raise ValueError("objective-balance execution device is not frozen")
        if self.execution_dtype != DIAGNOSTIC_EXECUTION_DTYPE:
            raise ValueError("objective-balance execution dtype is not frozen")
        if _exact_int(self.primary_seed, label="primary seed", minimum=1) != (
            DIAGNOSTIC_PRIMARY_SEED
        ):
            raise ValueError("objective-balance primary seed is not frozen")
        if _exact_int(
            self.replication_seed,
            label="replication seed",
            minimum=1,
        ) != DIAGNOSTIC_REPLICATION_SEED:
            raise ValueError("objective-balance replication seed is not frozen")
        recipes = tuple(self.recipes)
        if tuple(value.recipe_id for value in recipes) != (
            OBJECTIVE_BALANCE_RECIPE_IDS
        ):
            raise ValueError("objective-balance recipe ladder drifted")
        if any(value.primary_seed != self.primary_seed for value in recipes):
            raise ValueError("recipe primary seed differs from protocol")
        if recipes[0].advancement_eligible or not all(
            value.advancement_eligible for value in recipes[1:]
        ):
            raise ValueError("objective-balance advancement eligibility drifted")
        object.__setattr__(self, "recipes", recipes)
        if not isinstance(self.gates, ObjectiveBalanceDiagnosticGates):
            raise TypeError("diagnostic gates have the wrong type")
        if not isinstance(self.c2_provenance, C2ObjectiveBalanceProvenance):
            raise TypeError("C2 provenance has the wrong type")
        if self.gates != ObjectiveBalanceDiagnosticGates():
            raise ValueError("objective-balance diagnostic gates drifted")
        if self.c2_provenance != _default_c2_provenance():
            raise ValueError("objective-balance C2 provenance drifted")
        computed = _digest(self._payload(), domain=_PROTOCOL_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="objective-balance protocol",
                )
                != computed
            ):
                raise ValueError("objective-balance protocol hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def protocol_sha256(self) -> str:
        return self.artifact_sha256

    @property
    def advancement_recipe_ids(self) -> tuple[str, ...]:
        return tuple(
            value.recipe_id for value in self.recipes
            if value.advancement_eligible
        )

    def recipe(self, recipe_id: str) -> ObjectiveBalanceRecipe:
        for value in self.recipes:
            if value.recipe_id == recipe_id:
                return value
        raise KeyError(recipe_id)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "scientific_scope": (
                "fit_only_rank16_optimization_diagnostic_not_generalization"
            ),
            "latent_rank": self.latent_rank,
            "execution_device": self.execution_device,
            "execution_dtype": self.execution_dtype,
            "primary_seed": self.primary_seed,
            "replication_seed": self.replication_seed,
            "recipes": [value.state_dict() for value in self.recipes],
            "gates": self.gates.state_dict(),
            "c2_provenance": self.c2_provenance.state_dict(),
            "materialization_roles_allowed": ["pilot", "fit"],
            "selection_role_allowed": False,
            "recipe_decision_rule": (
                "first_advancement_eligible_recipe_in_declared_order_"
                "passing_every_gate_on_primary_and_replication_seeds"
            ),
            "success_authority": "may_lock_recipe_for_fresh_c3_only",
            "fresh_c3_required": True,
            "d3_treatment_semantics": (
                "composite_direction_weight_and_doubled_steps_"
                "no_component_attribution"
            ),
            "fit_data_binding_semantics": (
                "exact_c2_rank16_raw_control_binding_reused_for_all_recipes_"
                "to_freeze_batch_order"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: object,
    ) -> "ObjectiveBalanceDiagnosticProtocol":
        state = _mapping(raw, label="objective-balance protocol")
        expected = set(default_objective_balance_diagnostic_protocol().state_dict())
        _strict_keys(
            state,
            expected=expected,
            label="objective-balance protocol",
        )
        defaults = default_objective_balance_diagnostic_protocol()._payload()
        for name in (
            "schema",
            "format_version",
            "scientific_scope",
            "materialization_roles_allowed",
            "selection_role_allowed",
            "recipe_decision_rule",
            "success_authority",
            "fresh_c3_required",
            "d3_treatment_semantics",
            "fit_data_binding_semantics",
        ):
            if state[name] != defaults[name]:
                raise ValueError("objective-balance protocol semantics drifted")
        recipes = state["recipes"]
        if not isinstance(recipes, Sequence) or isinstance(
            recipes,
            (str, bytes),
        ):
            raise TypeError("objective-balance recipes must be a sequence")
        return cls(
            latent_rank=_exact_int(
                state["latent_rank"],
                label="latent rank",
                minimum=1,
            ),
            execution_device=str(state["execution_device"]),
            execution_dtype=str(state["execution_dtype"]),
            primary_seed=_exact_int(
                state["primary_seed"],
                label="primary seed",
                minimum=1,
            ),
            replication_seed=_exact_int(
                state["replication_seed"],
                label="replication seed",
                minimum=1,
            ),
            recipes=tuple(
                ObjectiveBalanceRecipe.from_state_dict(value)
                for value in recipes
            ),
            gates=ObjectiveBalanceDiagnosticGates.from_state_dict(
                state["gates"]
            ),
            c2_provenance=C2ObjectiveBalanceProvenance.from_state_dict(
                state["c2_provenance"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
        )


def family_balance_copy_binding_sha256(
    original_pair_id: str,
    original_pair_sha256: str,
) -> str:
    """Authenticate one exact signed-pair multiplicity copy."""

    if not isinstance(original_pair_id, str) or not original_pair_id:
        raise ValueError("original pair id must be nonempty")
    original_sha = _require_sha256(
        original_pair_sha256,
        label="original pair artifact",
    )
    return _digest(
        {
            "schema": "fisher_graph.objective_balance_signed_pair_copy.v1",
            "original_pair_id": original_pair_id,
            "original_pair_sha256": original_sha,
            "copy_ordinal": 1,
            "copy_semantics": (
                "same_endpoints_role_family_rank_and_teacher_chart_tensors"
            ),
        },
        domain=_PAIR_COPY_DOMAIN,
    )


def family_balance_copy_id(
    original_pair_id: str,
    original_pair_sha256: str,
) -> str:
    binding = family_balance_copy_binding_sha256(
        original_pair_id,
        original_pair_sha256,
    )
    return f"objective_balance_d1.signed_copy.{binding}"


def _default_c2_provenance() -> C2ObjectiveBalanceProvenance:
    return C2ObjectiveBalanceProvenance(
        protocol_sha256=(
            "033020dc9a0da819bd5753eb10090bff1bd9b4fcf61f33cd7186b1c1e3cb5254"
        ),
        pilot_panel_sha256=(
            "2b4d4efc8dbbdbc1c6afdb1a3134068f31b39ebc65e8a226fbbb9c92fe07b5e3"
        ),
        fit_panel_sha256=(
            "e6244896f2f6823f61d5ac57fad0756be320961e0b2272612579f9ba2f5b3e75"
        ),
        calibrated_fit_panel_sha256=(
            "f3175e8a08fe368614763c13e3a3eb89629b613ca7ca1a49710b44b227854fdd"
        ),
        calibration_sha256=(
            "aedb23de65ed6a37d645539001311ddb415cd2713400777dac448cb96bd5bfa8"
        ),
        selected_amplitude=8.0,
        objective_sha256=(
            "98fcafab29a9721990edcce3db3e18c68202f5907033d17e1390c425a5728881"
        ),
        training_sha256=(
            "da3d152cd5e9156e3b4fb27ad63af38cbdd9994bb4d5d7f1bb4e2eb621876322"
        ),
        forbidden_selection_panel_sha256s=(
            "4e86a5acf6fcaa4a1e03b61e387938fb442f11e2d82aee73b56d1ee0189013f6",
            "2c991c815d2f18d5b410fb296a6a9659d0a594c6abc93daee2ec21ef972a6aec",
        ),
        forbidden_probe_prefixes=("development_c2.selection.",),
    )


def _default_recipes() -> tuple[ObjectiveBalanceRecipe, ...]:
    common = {
        "pointwise_weight": 1.0,
        "sensitivity_relative_delta_weight": 2.0,
        "midpoint_jvp_weight": 1.0,
        "intended_null_weight": 1.0,
        "learning_rate": DIAGNOSTIC_LEARNING_RATE,
        "primary_seed": DIAGNOSTIC_PRIMARY_SEED,
    }
    return (
        ObjectiveBalanceRecipe(
            recipe_id=OBJECTIVE_BALANCE_RECIPE_IDS[0],
            training_metric="canonical_fisher",
            signed_pair_multiplicity=1,
            direction_weight=0.5,
            steps=300,
            advancement_eligible=False,
            **common,
        ),
        ObjectiveBalanceRecipe(
            recipe_id=OBJECTIVE_BALANCE_RECIPE_IDS[1],
            training_metric="fit_teacher_weighted_rms",
            signed_pair_multiplicity=1,
            direction_weight=0.5,
            steps=300,
            advancement_eligible=True,
            **common,
        ),
        ObjectiveBalanceRecipe(
            recipe_id=OBJECTIVE_BALANCE_RECIPE_IDS[2],
            training_metric="fit_teacher_weighted_rms",
            signed_pair_multiplicity=2,
            direction_weight=0.5,
            steps=300,
            advancement_eligible=True,
            **common,
        ),
        ObjectiveBalanceRecipe(
            recipe_id=OBJECTIVE_BALANCE_RECIPE_IDS[3],
            training_metric="fit_teacher_weighted_rms",
            signed_pair_multiplicity=2,
            direction_weight=2.0,
            steps=600,
            advancement_eligible=True,
            **common,
        ),
    )


def default_objective_balance_diagnostic_protocol(
) -> ObjectiveBalanceDiagnosticProtocol:
    """Return the complete immutable D0-D3 declaration."""

    protocol = ObjectiveBalanceDiagnosticProtocol(
        latent_rank=DIAGNOSTIC_LATENT_RANK,
        execution_device=DIAGNOSTIC_EXECUTION_DEVICE,
        execution_dtype=DIAGNOSTIC_EXECUTION_DTYPE,
        primary_seed=DIAGNOSTIC_PRIMARY_SEED,
        replication_seed=DIAGNOSTIC_REPLICATION_SEED,
        recipes=_default_recipes(),
        gates=ObjectiveBalanceDiagnosticGates(),
        c2_provenance=_default_c2_provenance(),
    )
    if (
        protocol.protocol_sha256
        != DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256
    ):
        raise RuntimeError(
            "objective-balance diagnostic protocol trust anchor drifted"
        )
    return protocol


# Literal trust anchor for the complete declaration.  It is intentionally not
# computed at import time; tests bind this value to the canonical state.
DEFAULT_OBJECTIVE_BALANCE_DIAGNOSTIC_PROTOCOL_SHA256 = (
    "d502d003fd86f6ef7322e35854d0bb738fdc4cfa6fa5089c2812e366a142d2eb"
)
