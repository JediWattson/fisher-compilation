"""Model-free held-out evaluation for Gemma-3 layer-local multimode flows.

This module is the scalar-only side of the B3/C3/D3 firewall.  It does not
import a model, a tensor library, a prompt panel, or a held-out role loader.
Future role workers may emit the authenticated additive records defined here,
but no constructor that opens B3, C3, or D3 data exists in this source.

B3 evaluates teacher-forced layer-local velocity fidelity.  C3 evaluates the
source-free compiled stack from whole-example additive sums.  D3 repeats C3
with an unchanged candidate and adds deterministic whole-example simultaneous
bootstrap bounds, exact resource evidence, and a no-native-MLP execution trace.

The D3 interval procedure is deliberately described as an approximate
bootstrap, not as distribution-free coverage: per-token NLL differences and
KL values are not bounded.  It uses 10,000 stratified whole-example draws and
one max-T critical value across the pooled, every-family, and hard-long harm
coordinates (delta NLL, KL, and top-1 error).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct


__all__ = [
    "B3_AGGREGATE_COSINE_MINIMUM",
    "B3_FAMILY_NRMSE_MAXIMUM",
    "B3_MODE_MASS_MINIMUM",
    "B3_STATE_MODE_MI_MINIMUM_NATS",
    "B3_VELOCITY_NRMSE_MAXIMUM",
    "BOOTSTRAP_REPLICATE_COUNT",
    "BOOTSTRAP_SEED_SHA256",
    "C3_FAMILY_DELTA_NLL_MAXIMUM",
    "C3_FAMILY_KL_MAXIMUM",
    "C3_FAMILY_TOP1_AGREEMENT_MINIMUM",
    "C3_POOLED_DELTA_NLL_MAXIMUM",
    "C3_POOLED_KL_MAXIMUM",
    "C3_POOLED_TOP1_AGREEMENT_MINIMUM",
    "CompiledCandidateIdentityReceipt",
    "D3FinalEvaluationReceipt",
    "D3NoNativeMLPTraceReceipt",
    "D3ResourceComparisonReceipt",
    "D3SimultaneousBoundReceipt",
    "FullStackExampleAggregate",
    "FullStackMetricSliceReceipt",
    "LayerTeacherForcedAdditiveEvidence",
    "LayerTeacherForcedDecisionReceipt",
    "LayerTeacherForcedFamilyAggregate",
    "LayerTeacherForcedModeAggregate",
    "B3TeacherForcedEvaluationReceipt",
    "C3FullStackEvaluationReceipt",
    "d3_bootstrap_policy",
    "d3_bootstrap_policy_sha256",
    "evaluate_b3_teacher_forced",
    "evaluate_c3_full_stack",
    "evaluate_d3_final",
    "heldout_evaluation_policy",
    "heldout_evaluation_policy_sha256",
]


LAYER_COUNT = 18
HIDDEN_WIDTH = 640
FAMILY_COUNT = 6

B3_VELOCITY_NRMSE_MAXIMUM = 0.30
B3_AGGREGATE_COSINE_MINIMUM = 0.90
B3_FAMILY_NRMSE_MAXIMUM = 0.40
B3_MODE_MASS_MINIMUM = 0.03
B3_STATE_MODE_MI_MINIMUM_NATS = 0.03

C3_POOLED_DELTA_NLL_MAXIMUM = 0.02
C3_POOLED_KL_MAXIMUM = 0.02
C3_POOLED_TOP1_AGREEMENT_MINIMUM = 0.95
C3_FAMILY_DELTA_NLL_MAXIMUM = 0.04
C3_FAMILY_KL_MAXIMUM = 0.04
C3_FAMILY_TOP1_AGREEMENT_MINIMUM = 0.90

BOOTSTRAP_REPLICATE_COUNT = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED_SHA256 = hashlib.sha256(
    b"fisher_graph:gemma3-layer-multimode:d3-bootstrap:v1"
).hexdigest()

B3_ROLE = "B3_teacher_forced_layerwise"
C3_ROLE = "C3_full_stack_source_free"
D3_ROLE = "D3_final_heldout_validation"

_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REL_TOL = 1e-12
_ABS_TOL = 1e-12

_FAMILY_KIND = "fisher_graph.gemma3_layer_multimode_b3_family_additive"
_MODE_KIND = "fisher_graph.gemma3_layer_multimode_b3_mode_additive"
_LAYER_KIND = "fisher_graph.gemma3_layer_multimode_b3_layer_evidence"
_LAYER_DECISION_KIND = "fisher_graph.gemma3_layer_multimode_b3_layer_decision"
_B3_KIND = "fisher_graph.gemma3_layer_multimode_b3_evaluation"
_CANDIDATE_KIND = "fisher_graph.gemma3_layer_multimode_candidate_identity"
_EXAMPLE_KIND = "fisher_graph.gemma3_layer_multimode_full_stack_example"
_SLICE_KIND = "fisher_graph.gemma3_layer_multimode_full_stack_slice"
_C3_KIND = "fisher_graph.gemma3_layer_multimode_c3_evaluation"
_BOUND_KIND = "fisher_graph.gemma3_layer_multimode_d3_bound"
_RESOURCE_KIND = "fisher_graph.gemma3_layer_multimode_d3_resources"
_TRACE_KIND = "fisher_graph.gemma3_layer_multimode_d3_no_native_mlp_trace"
_D3_KIND = "fisher_graph.gemma3_layer_multimode_d3_evaluation"

_FAMILY_DOMAIN = b"fisher_graph:layer-multimode:b3-family:v1\0"
_MODE_DOMAIN = b"fisher_graph:layer-multimode:b3-mode:v1\0"
_LAYER_DOMAIN = b"fisher_graph:layer-multimode:b3-layer:v1\0"
_LAYER_DECISION_DOMAIN = b"fisher_graph:layer-multimode:b3-decision:v1\0"
_B3_DOMAIN = b"fisher_graph:layer-multimode:b3-evaluation:v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph:layer-multimode:candidate:v1\0"
_EXAMPLE_DOMAIN = b"fisher_graph:layer-multimode:full-stack-example:v1\0"
_SLICE_DOMAIN = b"fisher_graph:layer-multimode:full-stack-slice:v1\0"
_C3_DOMAIN = b"fisher_graph:layer-multimode:c3-evaluation:v1\0"
_BOUND_DOMAIN = b"fisher_graph:layer-multimode:d3-bound:v1\0"
_RESOURCE_DOMAIN = b"fisher_graph:layer-multimode:d3-resource:v1\0"
_TRACE_DOMAIN = b"fisher_graph:layer-multimode:d3-trace:v1\0"
_D3_DOMAIN = b"fisher_graph:layer-multimode:d3-evaluation:v1\0"
_BOOTSTRAP_POLICY_DOMAIN = b"fisher_graph:layer-multimode:d3-policy:v1\0"
_EVALUATION_POLICY_DOMAIN = b"fisher_graph:layer-multimode:evaluation-policy:v1\0"
_BOOTSTRAP_REPLICATE_DOMAIN = b"fisher_graph:layer-multimode:d3-replicates:v1\0"
_SAMPLER_DOMAIN = b"fisher_graph:layer-multimode:d3-sampler:v1\0"

_SAFETY_FIELDS = {
    "contains_prompt_text": False,
    "contains_raw_tokens": False,
    "contains_raw_logits": False,
    "contains_model_weights": False,
    "contains_activation_rows": False,
    "contains_scalar_example_aggregates": True,
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite float")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _strict(
    state: Mapping[str, object],
    *,
    fields: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != fields:
        raise ValueError(f"{label} fields are invalid")


def _list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON list")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _state(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _state(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_state(item) for item in value]
    if type(value) is list:
        return [_state(item) for item in value]
    return value


def _install_hash(instance: object, payload: object, domain: bytes) -> None:
    computed = _hash(payload, domain=domain)
    current = getattr(instance, "artifact_sha256")
    if current == "":
        object.__setattr__(instance, "artifact_sha256", computed)
    elif _require_sha(current, label="artifact_sha256") != computed:
        raise ValueError("artifact SHA-256 mismatch")


def _validate_header(instance: object, expected_kind: str) -> None:
    if (
        getattr(instance, "artifact_kind") != expected_kind
        or type(getattr(instance, "format_version")) is not int
        or getattr(instance, "format_version") != _FORMAT_VERSION
    ):
        raise ValueError("receipt header is invalid")


@dataclass(frozen=True, slots=True)
class LayerTeacherForcedFamilyAggregate:
    """Additive B3 velocity sums for one family and one layer."""

    family_id: str
    scalar_count: int
    prediction_sse: float
    teacher_energy: float
    prediction_energy: float
    prediction_teacher_dot: float
    artifact_sha256: str = ""
    artifact_kind: str = _FAMILY_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_string(self.family_id, label="family_id")
        _require_int(self.scalar_count, label="scalar_count", minimum=1)
        _require_float(self.prediction_sse, label="prediction_sse", minimum=0.0)
        _require_float(self.teacher_energy, label="teacher_energy", minimum=0.0)
        _require_float(
            self.prediction_energy,
            label="prediction_energy",
            minimum=0.0,
        )
        _require_float(
            self.prediction_teacher_dot,
            label="prediction_teacher_dot",
        )
        if self.teacher_energy == 0.0 or self.prediction_energy == 0.0:
            raise ValueError("teacher and prediction energy must be positive")
        cosine = self.velocity_cosine
        if cosine < -1.0 - _ABS_TOL or cosine > 1.0 + _ABS_TOL:
            raise ValueError("prediction-teacher dot violates Cauchy-Schwarz")
        _validate_header(self, _FAMILY_KIND)
        _install_hash(self, self._payload(), _FAMILY_DOMAIN)

    @property
    def velocity_nrmse(self) -> float:
        return math.sqrt(self.prediction_sse / self.teacher_energy)

    @property
    def velocity_cosine(self) -> float:
        return self.prediction_teacher_dot / math.sqrt(
            self.prediction_energy * self.teacher_energy
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "family_id": self.family_id,
            "scalar_count": self.scalar_count,
            "prediction_sse": self.prediction_sse,
            "teacher_energy": self.teacher_energy,
            "prediction_energy": self.prediction_energy,
            "prediction_teacher_dot": self.prediction_teacher_dot,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerTeacherForcedFamilyAggregate:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "family_id",
                "scalar_count",
                "prediction_sse",
                "teacher_energy",
                "prediction_energy",
                "prediction_teacher_dot",
                "artifact_sha256",
            },
            label="B3 family aggregate",
        )
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LayerTeacherForcedModeAggregate:
    """Additive q and q*pi sums for one B3 layer chart."""

    chart_ordinal: int
    chart_weight_sum: float
    chart_mode_weight_sums: tuple[float, ...]
    weighted_router_entropy_sum_nats: float
    artifact_sha256: str = ""
    artifact_kind: str = _MODE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_int(self.chart_ordinal, label="chart_ordinal")
        _require_float(
            self.chart_weight_sum,
            label="chart_weight_sum",
            minimum=0.0,
        )
        if self.chart_weight_sum == 0.0:
            raise ValueError("chart_weight_sum must be positive")
        if (
            type(self.chart_mode_weight_sums) is not tuple
            or not self.chart_mode_weight_sums
        ):
            raise ValueError("chart_mode_weight_sums must be a nonempty tuple")
        for index, value in enumerate(self.chart_mode_weight_sums):
            _require_float(
                value,
                label=f"chart_mode_weight_sums[{index}]",
                minimum=0.0,
            )
        if not _close(sum(self.chart_mode_weight_sums), self.chart_weight_sum):
            raise ValueError("mode weights do not sum to chart weight")
        _require_float(
            self.weighted_router_entropy_sum_nats,
            label="weighted_router_entropy_sum_nats",
            minimum=0.0,
        )
        maximum_entropy = self.chart_weight_sum * math.log(
            len(self.chart_mode_weight_sums)
        )
        if self.weighted_router_entropy_sum_nats > maximum_entropy + _ABS_TOL:
            raise ValueError("weighted router entropy exceeds log(mode_count)")
        conditional = self.conditional_mode_masses
        mean_entropy = -sum(
            value * math.log(value) for value in conditional if value > 0.0
        )
        if self.mean_router_entropy_nats > mean_entropy + _ABS_TOL:
            raise ValueError("negative state-mode mutual information")
        if len(conditional) == 1 and (
            conditional != (1.0,)
            or self.weighted_router_entropy_sum_nats != 0.0
        ):
            raise ValueError("M=1 mode diagnostics must be exact sentinels")
        _validate_header(self, _MODE_KIND)
        _install_hash(self, self._payload(), _MODE_DOMAIN)

    @property
    def conditional_mode_masses(self) -> tuple[float, ...]:
        return tuple(
            value / self.chart_weight_sum
            for value in self.chart_mode_weight_sums
        )

    @property
    def mean_router_entropy_nats(self) -> float:
        return self.weighted_router_entropy_sum_nats / self.chart_weight_sum

    @property
    def state_mode_mutual_information_nats(self) -> float:
        entropy = -sum(
            value * math.log(value)
            for value in self.conditional_mode_masses
            if value > 0.0
        )
        value = entropy - self.mean_router_entropy_nats
        return 0.0 if abs(value) <= _ABS_TOL else value

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "chart_ordinal": self.chart_ordinal,
            "chart_weight_sum": self.chart_weight_sum,
            "chart_mode_weight_sums": self.chart_mode_weight_sums,
            "weighted_router_entropy_sum_nats": (
                self.weighted_router_entropy_sum_nats
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerTeacherForcedModeAggregate:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "chart_ordinal",
                "chart_weight_sum",
                "chart_mode_weight_sums",
                "weighted_router_entropy_sum_nats",
                "artifact_sha256",
            },
            label="B3 mode aggregate",
        )
        arguments = dict(state)
        arguments["chart_mode_weight_sums"] = tuple(
            _list(
                state["chart_mode_weight_sums"],
                label="chart_mode_weight_sums",
            )
        )
        return cls(**arguments)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LayerTeacherForcedAdditiveEvidence:
    """Scalar-only B3 evidence for one frozen selected layer candidate."""

    layer_ordinal: int
    chart_count: int
    dynamic_mode_count: int
    tangent_rank: int
    selected_candidate_sha256: str
    role_capture_receipt_sha256: str
    family_aggregates: tuple[LayerTeacherForcedFamilyAggregate, ...]
    mode_aggregates: tuple[LayerTeacherForcedModeAggregate, ...]
    row_weight_sum: float
    artifact_sha256: str = ""
    artifact_kind: str = _LAYER_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_int(self.layer_ordinal, label="layer_ordinal")
        if self.layer_ordinal >= LAYER_COUNT:
            raise ValueError("layer_ordinal is outside the 18-layer model")
        for label, value in (
            ("chart_count", self.chart_count),
            ("dynamic_mode_count", self.dynamic_mode_count),
            ("tangent_rank", self.tangent_rank),
        ):
            _require_int(value, label=label, minimum=1)
        _require_sha(
            self.selected_candidate_sha256,
            label="selected_candidate_sha256",
        )
        _require_sha(
            self.role_capture_receipt_sha256,
            label="role_capture_receipt_sha256",
        )
        if (
            type(self.family_aggregates) is not tuple
            or len(self.family_aggregates) != FAMILY_COUNT
            or any(
                type(value) is not LayerTeacherForcedFamilyAggregate
                for value in self.family_aggregates
            )
        ):
            raise ValueError("family_aggregates must contain six receipts")
        family_ids = tuple(value.family_id for value in self.family_aggregates)
        if (
            family_ids != tuple(sorted(family_ids))
            or len(set(family_ids)) != FAMILY_COUNT
        ):
            raise ValueError("family aggregates must use canonical unique order")
        if any(
            value.scalar_count % HIDDEN_WIDTH != 0
            for value in self.family_aggregates
        ):
            raise ValueError(
                "B3 family scalar counts must contain complete hidden rows"
            )
        if (
            type(self.mode_aggregates) is not tuple
            or len(self.mode_aggregates) != self.chart_count
            or any(
                type(value) is not LayerTeacherForcedModeAggregate
                for value in self.mode_aggregates
            )
        ):
            raise ValueError("mode_aggregates must contain one receipt per chart")
        if tuple(value.chart_ordinal for value in self.mode_aggregates) != tuple(
            range(self.chart_count)
        ):
            raise ValueError("mode aggregates must use canonical chart order")
        if any(
            len(value.chart_mode_weight_sums) != self.dynamic_mode_count
            for value in self.mode_aggregates
        ):
            raise ValueError("mode aggregate width does not match selected M")
        _require_float(self.row_weight_sum, label="row_weight_sum", minimum=0.0)
        if self.row_weight_sum == 0.0:
            raise ValueError("row_weight_sum must be positive")
        evaluated_row_count = self.scalar_count // HIDDEN_WIDTH
        if not _close(self.row_weight_sum, float(evaluated_row_count)):
            raise ValueError(
                "B3 routing row weight does not close to fidelity rows"
            )
        if not _close(
            sum(value.chart_weight_sum for value in self.mode_aggregates),
            self.row_weight_sum,
        ):
            raise ValueError("chart weights do not sum to row_weight_sum")
        if self.dynamic_mode_count == 1 and any(
            value.state_mode_mutual_information_nats != 0.0
            for value in self.mode_aggregates
        ):
            raise ValueError("M=1 mutual information must be zero")
        _validate_header(self, _LAYER_KIND)
        _install_hash(self, self._payload(), _LAYER_DOMAIN)

    @property
    def scalar_count(self) -> int:
        return sum(value.scalar_count for value in self.family_aggregates)

    @property
    def prediction_sse(self) -> float:
        return sum(value.prediction_sse for value in self.family_aggregates)

    @property
    def teacher_energy(self) -> float:
        return sum(value.teacher_energy for value in self.family_aggregates)

    @property
    def prediction_energy(self) -> float:
        return sum(value.prediction_energy for value in self.family_aggregates)

    @property
    def prediction_teacher_dot(self) -> float:
        return sum(
            value.prediction_teacher_dot for value in self.family_aggregates
        )

    @property
    def velocity_nrmse(self) -> float:
        return math.sqrt(self.prediction_sse / self.teacher_energy)

    @property
    def aggregate_velocity_cosine(self) -> float:
        return self.prediction_teacher_dot / math.sqrt(
            self.prediction_energy * self.teacher_energy
        )

    @property
    def chart_soft_masses(self) -> tuple[float, ...]:
        return tuple(
            value.chart_weight_sum / self.row_weight_sum
            for value in self.mode_aggregates
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "layer_ordinal": self.layer_ordinal,
            "chart_count": self.chart_count,
            "dynamic_mode_count": self.dynamic_mode_count,
            "tangent_rank": self.tangent_rank,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "role_capture_receipt_sha256": self.role_capture_receipt_sha256,
            "family_aggregates": tuple(
                value.state_dict() for value in self.family_aggregates
            ),
            "mode_aggregates": tuple(
                value.state_dict() for value in self.mode_aggregates
            ),
            "row_weight_sum": self.row_weight_sum,
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerTeacherForcedAdditiveEvidence:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "layer_ordinal",
                "chart_count",
                "dynamic_mode_count",
                "tangent_rank",
                "selected_candidate_sha256",
                "role_capture_receipt_sha256",
                "family_aggregates",
                "mode_aggregates",
                "row_weight_sum",
                "artifact_sha256",
            },
            label="B3 layer evidence",
        )
        arguments = dict(state)
        arguments["family_aggregates"] = tuple(
            LayerTeacherForcedFamilyAggregate.from_state_dict(value)
            for value in _list(
                state["family_aggregates"], label="family_aggregates"
            )
        )
        arguments["mode_aggregates"] = tuple(
            LayerTeacherForcedModeAggregate.from_state_dict(value)
            for value in _list(
                state["mode_aggregates"], label="mode_aggregates"
            )
        )
        return cls(**arguments)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LayerTeacherForcedDecisionReceipt:
    layer_evidence_sha256: str
    dynamic_mode_count: int
    velocity_nrmse: float
    aggregate_velocity_cosine: float
    maximum_family_velocity_nrmse: float
    minimum_conditional_mode_mass: float
    minimum_state_mode_mutual_information_nats: float
    failed_gates: tuple[str, ...]
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _LAYER_DECISION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha(self.layer_evidence_sha256, label="layer_evidence_sha256")
        _require_int(
            self.dynamic_mode_count,
            label="dynamic_mode_count",
            minimum=1,
        )
        _require_float(self.velocity_nrmse, label="velocity_nrmse", minimum=0.0)
        _require_float(
            self.aggregate_velocity_cosine,
            label="aggregate_velocity_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        _require_float(
            self.maximum_family_velocity_nrmse,
            label="maximum_family_velocity_nrmse",
            minimum=0.0,
        )
        _require_float(
            self.minimum_conditional_mode_mass,
            label="minimum_conditional_mode_mass",
            minimum=0.0,
            maximum=1.0,
        )
        _require_float(
            self.minimum_state_mode_mutual_information_nats,
            label="minimum_state_mode_mutual_information_nats",
            minimum=0.0,
        )
        if (
            type(self.failed_gates) is not tuple
            or self.failed_gates != tuple(sorted(set(self.failed_gates)))
            or any(type(value) is not str or not value for value in self.failed_gates)
        ):
            raise ValueError("failed_gates must be canonical unique strings")
        _require_bool(self.passed, label="passed")
        expected_failures: list[str] = []
        if self.velocity_nrmse > B3_VELOCITY_NRMSE_MAXIMUM:
            expected_failures.append("velocity_nrmse")
        if self.aggregate_velocity_cosine < B3_AGGREGATE_COSINE_MINIMUM:
            expected_failures.append("aggregate_velocity_cosine")
        if self.maximum_family_velocity_nrmse > B3_FAMILY_NRMSE_MAXIMUM:
            expected_failures.append("family_velocity_nrmse")
        if self.dynamic_mode_count > 1:
            if self.minimum_conditional_mode_mass < B3_MODE_MASS_MINIMUM:
                expected_failures.append("chart_conditional_mode_mass")
            if (
                self.minimum_state_mode_mutual_information_nats
                < B3_STATE_MODE_MI_MINIMUM_NATS
            ):
                expected_failures.append(
                    "chart_state_mode_mutual_information"
                )
        if self.failed_gates != tuple(sorted(expected_failures)):
            raise ValueError("B3 decision failed_gates do not recompute")
        if self.passed is not (not expected_failures):
            raise ValueError("passed does not recompute from failed_gates")
        _validate_header(self, _LAYER_DECISION_KIND)
        _install_hash(self, self._payload(), _LAYER_DECISION_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "layer_evidence_sha256": self.layer_evidence_sha256,
            "dynamic_mode_count": self.dynamic_mode_count,
            "velocity_nrmse": self.velocity_nrmse,
            "aggregate_velocity_cosine": self.aggregate_velocity_cosine,
            "maximum_family_velocity_nrmse": self.maximum_family_velocity_nrmse,
            "minimum_conditional_mode_mass": self.minimum_conditional_mode_mass,
            "minimum_state_mode_mutual_information_nats": (
                self.minimum_state_mode_mutual_information_nats
            ),
            "failed_gates": self.failed_gates,
            "passed": self.passed,
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> LayerTeacherForcedDecisionReceipt:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "layer_evidence_sha256",
                "dynamic_mode_count",
                "velocity_nrmse",
                "aggregate_velocity_cosine",
                "maximum_family_velocity_nrmse",
                "minimum_conditional_mode_mass",
                "minimum_state_mode_mutual_information_nats",
                "failed_gates",
                "passed",
                "artifact_sha256",
            },
            label="B3 layer decision",
        )
        arguments = dict(state)
        arguments["failed_gates"] = tuple(
            _list(state["failed_gates"], label="failed_gates")
        )
        return cls(**arguments)  # type: ignore[arg-type]


def _decide_b3_layer(
    evidence: LayerTeacherForcedAdditiveEvidence,
) -> LayerTeacherForcedDecisionReceipt:
    minimum_mass = min(
        min(value.conditional_mode_masses) for value in evidence.mode_aggregates
    )
    minimum_mi = min(
        value.state_mode_mutual_information_nats
        for value in evidence.mode_aggregates
    )
    failures: list[str] = []
    if evidence.velocity_nrmse > B3_VELOCITY_NRMSE_MAXIMUM:
        failures.append("velocity_nrmse")
    if evidence.aggregate_velocity_cosine < B3_AGGREGATE_COSINE_MINIMUM:
        failures.append("aggregate_velocity_cosine")
    maximum_family = max(
        value.velocity_nrmse for value in evidence.family_aggregates
    )
    if maximum_family > B3_FAMILY_NRMSE_MAXIMUM:
        failures.append("family_velocity_nrmse")
    if evidence.dynamic_mode_count > 1:
        if minimum_mass < B3_MODE_MASS_MINIMUM:
            failures.append("chart_conditional_mode_mass")
        if minimum_mi < B3_STATE_MODE_MI_MINIMUM_NATS:
            failures.append("chart_state_mode_mutual_information")
    return LayerTeacherForcedDecisionReceipt(
        layer_evidence_sha256=evidence.artifact_sha256,
        dynamic_mode_count=evidence.dynamic_mode_count,
        velocity_nrmse=float(evidence.velocity_nrmse),
        aggregate_velocity_cosine=float(evidence.aggregate_velocity_cosine),
        maximum_family_velocity_nrmse=float(maximum_family),
        minimum_conditional_mode_mass=float(minimum_mass),
        minimum_state_mode_mutual_information_nats=float(minimum_mi),
        failed_gates=tuple(sorted(failures)),
        passed=not failures,
    )


@dataclass(frozen=True, slots=True)
class B3TeacherForcedEvaluationReceipt:
    source_model_sha256: str
    selection_receipt_sha256: str
    panel_manifest_sha256: str
    evaluation_protocol_sha256: str
    layer_evidence: tuple[LayerTeacherForcedAdditiveEvidence, ...]
    layer_decisions: tuple[LayerTeacherForcedDecisionReceipt, ...]
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _B3_KIND
    format_version: int = _FORMAT_VERSION
    role: str = B3_ROLE
    contains_prompt_text: bool = False
    contains_raw_tokens: bool = False
    contains_raw_logits: bool = False
    contains_model_weights: bool = False
    contains_activation_rows: bool = False
    contains_scalar_example_aggregates: bool = True

    def __post_init__(self) -> None:
        for label in (
            "source_model_sha256",
            "selection_receipt_sha256",
            "panel_manifest_sha256",
            "evaluation_protocol_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        if self.role != B3_ROLE:
            raise ValueError("B3 receipt role is invalid")
        if self.evaluation_protocol_sha256 != heldout_evaluation_policy_sha256():
            raise ValueError("B3 evaluation protocol SHA-256 is not frozen")
        if (
            type(self.layer_evidence) is not tuple
            or len(self.layer_evidence) != LAYER_COUNT
            or tuple(value.layer_ordinal for value in self.layer_evidence)
            != tuple(range(LAYER_COUNT))
        ):
            raise ValueError("B3 evidence must contain all layers in order")
        family_ids = tuple(
            value.family_id for value in self.layer_evidence[0].family_aggregates
        )
        role_capture = self.layer_evidence[0].role_capture_receipt_sha256
        for evidence in self.layer_evidence:
            if (
                tuple(
                    value.family_id for value in evidence.family_aggregates
                )
                != family_ids
            ):
                raise ValueError("B3 family order differs across layers")
            if evidence.role_capture_receipt_sha256 != role_capture:
                raise ValueError("B3 layers do not share one role capture receipt")
        if (
            type(self.layer_decisions) is not tuple
            or len(self.layer_decisions) != LAYER_COUNT
        ):
            raise ValueError("B3 decisions must contain all layers")
        expected = tuple(_decide_b3_layer(value) for value in self.layer_evidence)
        if tuple(value.state_dict() for value in self.layer_decisions) != tuple(
            value.state_dict() for value in expected
        ):
            raise ValueError("B3 layer decisions do not recompute")
        _require_bool(self.passed, label="passed")
        if self.passed is not all(value.passed for value in self.layer_decisions):
            raise ValueError("B3 passed does not recompute")
        for label, expected_value in _SAFETY_FIELDS.items():
            if getattr(self, label) is not expected_value:
                raise ValueError("B3 safety metadata is invalid")
        _validate_header(self, _B3_KIND)
        _install_hash(self, self._payload(), _B3_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "role": self.role,
            "source_model_sha256": self.source_model_sha256,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "panel_manifest_sha256": self.panel_manifest_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "layer_evidence": tuple(
                value.state_dict() for value in self.layer_evidence
            ),
            "layer_decisions": tuple(
                value.state_dict() for value in self.layer_decisions
            ),
            "passed": self.passed,
            **_SAFETY_FIELDS,
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> B3TeacherForcedEvaluationReceipt:
        fields = {
            "artifact_kind",
            "format_version",
            "role",
            "source_model_sha256",
            "selection_receipt_sha256",
            "panel_manifest_sha256",
            "evaluation_protocol_sha256",
            "layer_evidence",
            "layer_decisions",
            "passed",
            "artifact_sha256",
            *_SAFETY_FIELDS,
        }
        _strict(state, fields=fields, label="B3 evaluation")
        arguments = dict(state)
        arguments["layer_evidence"] = tuple(
            LayerTeacherForcedAdditiveEvidence.from_state_dict(value)
            for value in _list(state["layer_evidence"], label="layer_evidence")
        )
        arguments["layer_decisions"] = tuple(
            LayerTeacherForcedDecisionReceipt.from_state_dict(value)
            for value in _list(state["layer_decisions"], label="layer_decisions")
        )
        return cls(**arguments)  # type: ignore[arg-type]


def evaluate_b3_teacher_forced(
    layer_evidence: Sequence[LayerTeacherForcedAdditiveEvidence],
    *,
    source_model_sha256: str,
    selection_receipt_sha256: str,
    panel_manifest_sha256: str,
    evaluation_protocol_sha256: str,
) -> B3TeacherForcedEvaluationReceipt:
    """Evaluate already-authenticated B3 scalar evidence; opens no role data."""

    evidence = tuple(layer_evidence)
    decisions = tuple(_decide_b3_layer(value) for value in evidence)
    return B3TeacherForcedEvaluationReceipt(
        source_model_sha256=source_model_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        panel_manifest_sha256=panel_manifest_sha256,
        evaluation_protocol_sha256=evaluation_protocol_sha256,
        layer_evidence=evidence,
        layer_decisions=decisions,
        passed=all(value.passed for value in decisions),
    )


@dataclass(frozen=True, slots=True)
class CompiledCandidateIdentityReceipt:
    """Frozen identity shared byte-for-byte by C3 and D3."""

    selection_receipt_sha256: str
    compiler_protocol_sha256: str
    compile_receipt_sha256: str
    bundle_semantic_sha256: str
    compiled_model_state_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _CANDIDATE_KIND
    format_version: int = _FORMAT_VERSION
    source_free_compiled_stack: bool = True
    native_mlp_fallback_available: bool = False

    def __post_init__(self) -> None:
        for label in (
            "selection_receipt_sha256",
            "compiler_protocol_sha256",
            "compile_receipt_sha256",
            "bundle_semantic_sha256",
            "compiled_model_state_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        if self.source_free_compiled_stack is not True:
            raise ValueError("candidate must be a source-free compiled stack")
        if self.native_mlp_fallback_available is not False:
            raise ValueError("candidate may not expose a native MLP fallback")
        _validate_header(self, _CANDIDATE_KIND)
        _install_hash(self, self._payload(), _CANDIDATE_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "compiler_protocol_sha256": self.compiler_protocol_sha256,
            "compile_receipt_sha256": self.compile_receipt_sha256,
            "bundle_semantic_sha256": self.bundle_semantic_sha256,
            "compiled_model_state_sha256": self.compiled_model_state_sha256,
            "source_free_compiled_stack": self.source_free_compiled_stack,
            "native_mlp_fallback_available": self.native_mlp_fallback_available,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CompiledCandidateIdentityReceipt:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "selection_receipt_sha256",
                "compiler_protocol_sha256",
                "compile_receipt_sha256",
                "bundle_semantic_sha256",
                "compiled_model_state_sha256",
                "source_free_compiled_stack",
                "native_mlp_fallback_available",
                "artifact_sha256",
            },
            label="compiled candidate identity",
        )
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FullStackExampleAggregate:
    """One whole-example additive record; never contains token-level data.

    Delta NLL is candidate minus native on identical scored positions.  KL is
    native-to-candidate over the full vocabulary, and top-1 agreement compares
    the two deterministic argmax indices on those same positions.
    """

    role: str
    example_identity_sha256: str
    family_id: str
    hard_long: bool
    predicted_token_count: int
    delta_nll_sum_nats: float
    native_to_candidate_kl_sum_nats: float
    top1_agreement_count: int
    example_execution_receipt_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _EXAMPLE_KIND
    format_version: int = _FORMAT_VERSION
    contains_prompt_text: bool = False
    contains_raw_tokens: bool = False
    contains_raw_logits: bool = False
    contains_model_weights: bool = False
    contains_activation_rows: bool = False
    contains_scalar_example_aggregates: bool = True

    def __post_init__(self) -> None:
        if self.role not in (C3_ROLE, D3_ROLE):
            raise ValueError("full-stack example role must be C3 or D3")
        _require_sha(
            self.example_identity_sha256,
            label="example_identity_sha256",
        )
        _require_string(self.family_id, label="family_id")
        _require_bool(self.hard_long, label="hard_long")
        _require_int(
            self.predicted_token_count,
            label="predicted_token_count",
            minimum=1,
        )
        _require_float(self.delta_nll_sum_nats, label="delta_nll_sum_nats")
        _require_float(
            self.native_to_candidate_kl_sum_nats,
            label="native_to_candidate_kl_sum_nats",
            minimum=0.0,
        )
        _require_int(
            self.top1_agreement_count,
            label="top1_agreement_count",
        )
        if self.top1_agreement_count > self.predicted_token_count:
            raise ValueError("top1 agreement count exceeds predicted tokens")
        _require_sha(
            self.example_execution_receipt_sha256,
            label="example_execution_receipt_sha256",
        )
        for label, expected in _SAFETY_FIELDS.items():
            if getattr(self, label) is not expected:
                raise ValueError("full-stack example safety metadata is invalid")
        _validate_header(self, _EXAMPLE_KIND)
        _install_hash(self, self._payload(), _EXAMPLE_DOMAIN)

    @property
    def delta_nll_nats_per_token(self) -> float:
        return self.delta_nll_sum_nats / self.predicted_token_count

    @property
    def native_to_candidate_kl_per_token(self) -> float:
        return self.native_to_candidate_kl_sum_nats / self.predicted_token_count

    @property
    def top1_error_rate(self) -> float:
        return 1.0 - self.top1_agreement_count / self.predicted_token_count

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "role": self.role,
            "example_identity_sha256": self.example_identity_sha256,
            "family_id": self.family_id,
            "hard_long": self.hard_long,
            "predicted_token_count": self.predicted_token_count,
            "delta_nll_sum_nats": self.delta_nll_sum_nats,
            "native_to_candidate_kl_sum_nats": (
                self.native_to_candidate_kl_sum_nats
            ),
            "top1_agreement_count": self.top1_agreement_count,
            "example_execution_receipt_sha256": (
                self.example_execution_receipt_sha256
            ),
            **_SAFETY_FIELDS,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FullStackExampleAggregate:
        fields = {
            "artifact_kind",
            "format_version",
            "role",
            "example_identity_sha256",
            "family_id",
            "hard_long",
            "predicted_token_count",
            "delta_nll_sum_nats",
            "native_to_candidate_kl_sum_nats",
            "top1_agreement_count",
            "example_execution_receipt_sha256",
            "artifact_sha256",
            *_SAFETY_FIELDS,
        }
        _strict(state, fields=fields, label="full-stack example aggregate")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FullStackMetricSliceReceipt:
    """Additive and derived full-stack metrics for one canonical slice."""

    slice_id: str
    example_count: int
    predicted_token_count: int
    delta_nll_sum_nats: float
    native_to_candidate_kl_sum_nats: float
    top1_agreement_count: int
    delta_nll_nats_per_token: float
    native_to_candidate_kl_per_token: float
    top1_agreement: float
    top1_error: float
    artifact_sha256: str = ""
    artifact_kind: str = _SLICE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_string(self.slice_id, label="slice_id")
        _require_int(self.example_count, label="example_count", minimum=1)
        _require_int(
            self.predicted_token_count,
            label="predicted_token_count",
            minimum=1,
        )
        _require_float(self.delta_nll_sum_nats, label="delta_nll_sum_nats")
        _require_float(
            self.native_to_candidate_kl_sum_nats,
            label="native_to_candidate_kl_sum_nats",
            minimum=0.0,
        )
        _require_int(self.top1_agreement_count, label="top1_agreement_count")
        if self.top1_agreement_count > self.predicted_token_count:
            raise ValueError("slice top1 agreement count exceeds token count")
        _require_float(
            self.delta_nll_nats_per_token,
            label="delta_nll_nats_per_token",
        )
        _require_float(
            self.native_to_candidate_kl_per_token,
            label="native_to_candidate_kl_per_token",
            minimum=0.0,
        )
        _require_float(
            self.top1_agreement,
            label="top1_agreement",
            minimum=0.0,
            maximum=1.0,
        )
        _require_float(
            self.top1_error,
            label="top1_error",
            minimum=0.0,
            maximum=1.0,
        )
        expected_delta = self.delta_nll_sum_nats / self.predicted_token_count
        expected_kl = (
            self.native_to_candidate_kl_sum_nats / self.predicted_token_count
        )
        expected_agreement = (
            self.top1_agreement_count / self.predicted_token_count
        )
        if not _close(self.delta_nll_nats_per_token, expected_delta):
            raise ValueError("slice delta NLL does not recompute")
        if not _close(self.native_to_candidate_kl_per_token, expected_kl):
            raise ValueError("slice KL does not recompute")
        if not _close(self.top1_agreement, expected_agreement):
            raise ValueError("slice top1 agreement does not recompute")
        if not _close(self.top1_error, 1.0 - expected_agreement):
            raise ValueError("slice top1 error does not recompute")
        _validate_header(self, _SLICE_KIND)
        _install_hash(self, self._payload(), _SLICE_DOMAIN)

    @property
    def harm_coordinates(self) -> tuple[float, float, float]:
        return (
            self.delta_nll_nats_per_token,
            self.native_to_candidate_kl_per_token,
            self.top1_error,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "slice_id": self.slice_id,
            "example_count": self.example_count,
            "predicted_token_count": self.predicted_token_count,
            "delta_nll_sum_nats": self.delta_nll_sum_nats,
            "native_to_candidate_kl_sum_nats": (
                self.native_to_candidate_kl_sum_nats
            ),
            "top1_agreement_count": self.top1_agreement_count,
            "delta_nll_nats_per_token": self.delta_nll_nats_per_token,
            "native_to_candidate_kl_per_token": (
                self.native_to_candidate_kl_per_token
            ),
            "top1_agreement": self.top1_agreement,
            "top1_error": self.top1_error,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> FullStackMetricSliceReceipt:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "slice_id",
                "example_count",
                "predicted_token_count",
                "delta_nll_sum_nats",
                "native_to_candidate_kl_sum_nats",
                "top1_agreement_count",
                "delta_nll_nats_per_token",
                "native_to_candidate_kl_per_token",
                "top1_agreement",
                "top1_error",
                "artifact_sha256",
            },
            label="full-stack metric slice",
        )
        return cls(**state)  # type: ignore[arg-type]


def _slice(
    slice_id: str,
    records: Sequence[FullStackExampleAggregate],
) -> FullStackMetricSliceReceipt:
    if not records:
        raise ValueError(f"slice {slice_id} is empty")
    token_count = sum(value.predicted_token_count for value in records)
    delta_sum = math.fsum(value.delta_nll_sum_nats for value in records)
    kl_sum = math.fsum(
        value.native_to_candidate_kl_sum_nats for value in records
    )
    agreement = sum(value.top1_agreement_count for value in records)
    agreement_rate = agreement / token_count
    return FullStackMetricSliceReceipt(
        slice_id=slice_id,
        example_count=len(records),
        predicted_token_count=token_count,
        delta_nll_sum_nats=float(delta_sum),
        native_to_candidate_kl_sum_nats=float(kl_sum),
        top1_agreement_count=agreement,
        delta_nll_nats_per_token=float(delta_sum / token_count),
        native_to_candidate_kl_per_token=float(kl_sum / token_count),
        top1_agreement=float(agreement_rate),
        top1_error=float(1.0 - agreement_rate),
    )


def _canonical_records(
    records: Sequence[FullStackExampleAggregate],
    *,
    role: str,
) -> tuple[FullStackExampleAggregate, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("full-stack evidence must be nonempty")
    if any(type(value) is not FullStackExampleAggregate for value in values):
        raise TypeError("full-stack evidence contains an invalid record")
    if any(value.role != role for value in values):
        raise ValueError("full-stack evidence role mismatch")
    identities = tuple(value.example_identity_sha256 for value in values)
    if identities != tuple(sorted(identities)) or len(set(identities)) != len(values):
        raise ValueError("full-stack evidence must use canonical unique identity order")
    families = tuple(sorted({value.family_id for value in values}))
    if len(families) != FAMILY_COUNT:
        raise ValueError("full-stack evidence must contain exactly six families")
    if any(
        not any(value.family_id == family for value in values)
        for family in families
    ):
        raise ValueError("full-stack family strata must be nonempty")
    if not any(value.hard_long for value in values):
        raise ValueError("full-stack hard-long stratum must be nonempty")
    return values


def _metric_slices(
    records: Sequence[FullStackExampleAggregate],
) -> tuple[FullStackMetricSliceReceipt, ...]:
    families = tuple(sorted({value.family_id for value in records}))
    result = [_slice("pooled", records)]
    result.extend(
        _slice(
            f"family:{family}",
            tuple(value for value in records if value.family_id == family),
        )
        for family in families
    )
    result.append(
        _slice("hard_long", tuple(value for value in records if value.hard_long))
    )
    return tuple(result)


def _point_gate_failures(
    slices: Sequence[FullStackMetricSliceReceipt],
) -> tuple[str, ...]:
    by_id = {value.slice_id: value for value in slices}
    failures: list[str] = []

    def check(
        value: FullStackMetricSliceReceipt,
        *,
        prefix: str,
        delta_max: float,
        kl_max: float,
        agreement_min: float,
    ) -> None:
        if value.delta_nll_nats_per_token > delta_max:
            failures.append(f"{prefix}.delta_nll")
        if value.native_to_candidate_kl_per_token > kl_max:
            failures.append(f"{prefix}.kl")
        if value.top1_agreement < agreement_min:
            failures.append(f"{prefix}.top1_agreement")

    check(
        by_id["pooled"],
        prefix="pooled",
        delta_max=C3_POOLED_DELTA_NLL_MAXIMUM,
        kl_max=C3_POOLED_KL_MAXIMUM,
        agreement_min=C3_POOLED_TOP1_AGREEMENT_MINIMUM,
    )
    check(
        by_id["hard_long"],
        prefix="hard_long",
        delta_max=C3_POOLED_DELTA_NLL_MAXIMUM,
        kl_max=C3_POOLED_KL_MAXIMUM,
        agreement_min=C3_POOLED_TOP1_AGREEMENT_MINIMUM,
    )
    for slice_id in sorted(value for value in by_id if value.startswith("family:")):
        check(
            by_id[slice_id],
            prefix=slice_id,
            delta_max=C3_FAMILY_DELTA_NLL_MAXIMUM,
            kl_max=C3_FAMILY_KL_MAXIMUM,
            agreement_min=C3_FAMILY_TOP1_AGREEMENT_MINIMUM,
        )
    return tuple(sorted(failures))


@dataclass(frozen=True, slots=True)
class C3FullStackEvaluationReceipt:
    source_model_sha256: str
    b3_evaluation_sha256: str
    panel_manifest_sha256: str
    role_capture_receipt_sha256: str
    evaluation_protocol_sha256: str
    candidate_identity: CompiledCandidateIdentityReceipt
    example_aggregates: tuple[FullStackExampleAggregate, ...]
    metric_slices: tuple[FullStackMetricSliceReceipt, ...]
    failed_gates: tuple[str, ...]
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _C3_KIND
    format_version: int = _FORMAT_VERSION
    role: str = C3_ROLE
    contains_prompt_text: bool = False
    contains_raw_tokens: bool = False
    contains_raw_logits: bool = False
    contains_model_weights: bool = False
    contains_activation_rows: bool = False
    contains_scalar_example_aggregates: bool = True

    def __post_init__(self) -> None:
        for label in (
            "source_model_sha256",
            "b3_evaluation_sha256",
            "panel_manifest_sha256",
            "role_capture_receipt_sha256",
            "evaluation_protocol_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        if self.role != C3_ROLE:
            raise ValueError("C3 receipt role is invalid")
        if self.evaluation_protocol_sha256 != heldout_evaluation_policy_sha256():
            raise ValueError("C3 evaluation protocol SHA-256 is not frozen")
        if type(self.candidate_identity) is not CompiledCandidateIdentityReceipt:
            raise TypeError("candidate_identity is invalid")
        if type(self.example_aggregates) is not tuple:
            raise TypeError("example_aggregates must be a tuple")
        if type(self.metric_slices) is not tuple:
            raise TypeError("metric_slices must be a tuple")
        records = _canonical_records(self.example_aggregates, role=C3_ROLE)
        expected_slices = _metric_slices(records)
        if tuple(value.state_dict() for value in self.metric_slices) != tuple(
            value.state_dict() for value in expected_slices
        ):
            raise ValueError("C3 metric slices do not recompute")
        expected_failures = _point_gate_failures(expected_slices)
        if self.failed_gates != expected_failures:
            raise ValueError("C3 failed_gates do not recompute")
        _require_bool(self.passed, label="passed")
        if self.passed is not (not self.failed_gates):
            raise ValueError("C3 passed does not recompute")
        for label, expected in _SAFETY_FIELDS.items():
            if getattr(self, label) is not expected:
                raise ValueError("C3 safety metadata is invalid")
        _validate_header(self, _C3_KIND)
        _install_hash(self, self._payload(), _C3_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "role": self.role,
            "source_model_sha256": self.source_model_sha256,
            "b3_evaluation_sha256": self.b3_evaluation_sha256,
            "panel_manifest_sha256": self.panel_manifest_sha256,
            "role_capture_receipt_sha256": self.role_capture_receipt_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "candidate_identity": self.candidate_identity.state_dict(),
            "example_aggregates": tuple(
                value.state_dict() for value in self.example_aggregates
            ),
            "metric_slices": tuple(value.state_dict() for value in self.metric_slices),
            "failed_gates": self.failed_gates,
            "passed": self.passed,
            **_SAFETY_FIELDS,
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> C3FullStackEvaluationReceipt:
        fields = {
            "artifact_kind",
            "format_version",
            "role",
            "source_model_sha256",
            "b3_evaluation_sha256",
            "panel_manifest_sha256",
            "role_capture_receipt_sha256",
            "evaluation_protocol_sha256",
            "candidate_identity",
            "example_aggregates",
            "metric_slices",
            "failed_gates",
            "passed",
            "artifact_sha256",
            *_SAFETY_FIELDS,
        }
        _strict(state, fields=fields, label="C3 full-stack evaluation")
        arguments = dict(state)
        candidate_state = state["candidate_identity"]
        if not isinstance(candidate_state, Mapping):
            raise TypeError("candidate_identity must be a JSON object")
        arguments["candidate_identity"] = (
            CompiledCandidateIdentityReceipt.from_state_dict(candidate_state)
        )
        arguments["example_aggregates"] = tuple(
            FullStackExampleAggregate.from_state_dict(value)
            for value in _list(
                state["example_aggregates"], label="example_aggregates"
            )
        )
        arguments["metric_slices"] = tuple(
            FullStackMetricSliceReceipt.from_state_dict(value)
            for value in _list(state["metric_slices"], label="metric_slices")
        )
        arguments["failed_gates"] = tuple(
            _list(state["failed_gates"], label="failed_gates")
        )
        return cls(**arguments)  # type: ignore[arg-type]


def evaluate_c3_full_stack(
    example_aggregates: Sequence[FullStackExampleAggregate],
    *,
    b3_evaluation: B3TeacherForcedEvaluationReceipt,
    source_model_sha256: str,
    panel_manifest_sha256: str,
    role_capture_receipt_sha256: str,
    evaluation_protocol_sha256: str,
    candidate_identity: CompiledCandidateIdentityReceipt,
) -> C3FullStackEvaluationReceipt:
    """Evaluate C3 only after strict replay of a terminal passing B3 receipt."""

    if type(b3_evaluation) is not B3TeacherForcedEvaluationReceipt:
        raise TypeError("b3_evaluation must be the exact typed B3 receipt")
    replayed_b3 = B3TeacherForcedEvaluationReceipt.from_state_dict(
        json.loads(json.dumps(b3_evaluation.state_dict()))
    )
    if replayed_b3.state_dict() != b3_evaluation.state_dict():
        raise ValueError("B3 evaluation did not survive strict replay")
    if not b3_evaluation.passed:
        raise ValueError("C3 is unavailable until all B3 gates pass")
    if b3_evaluation.source_model_sha256 != source_model_sha256:
        raise ValueError("B3 and C3 source-model identities differ")
    if b3_evaluation.panel_manifest_sha256 != panel_manifest_sha256:
        raise ValueError("B3 and C3 do not share the frozen panel manifest")
    if (
        b3_evaluation.evaluation_protocol_sha256
        != evaluation_protocol_sha256
    ):
        raise ValueError("B3 and C3 evaluation protocols differ")
    if (
        b3_evaluation.selection_receipt_sha256
        != candidate_identity.selection_receipt_sha256
    ):
        raise ValueError("B3 and C3 candidate selections differ")

    records = _canonical_records(example_aggregates, role=C3_ROLE)
    slices = _metric_slices(records)
    failures = _point_gate_failures(slices)
    return C3FullStackEvaluationReceipt(
        source_model_sha256=source_model_sha256,
        b3_evaluation_sha256=b3_evaluation.artifact_sha256,
        panel_manifest_sha256=panel_manifest_sha256,
        role_capture_receipt_sha256=role_capture_receipt_sha256,
        evaluation_protocol_sha256=evaluation_protocol_sha256,
        candidate_identity=candidate_identity,
        example_aggregates=records,
        metric_slices=slices,
        failed_gates=failures,
        passed=not failures,
    )


def d3_bootstrap_policy() -> dict[str, object]:
    """Return the exact prospective D3 simultaneous-bound protocol."""

    return {
        "schema": "fisher_graph.gemma3_layer_multimode_d3_bootstrap_policy",
        "format_version": 1,
        "scientific_status": "approximate_bootstrap_not_distribution_free",
        "reason_not_distribution_free": "delta_NLL_and_KL_are_unbounded",
        "resampling_unit": "whole_example",
        "replicate_count": BOOTSTRAP_REPLICATE_COUNT,
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "seed_sha256": BOOTSTRAP_SEED_SHA256,
        "record_order": "ascending_example_identity_sha256",
        "stratification": "all_nonempty_family_id_x_hard_long_cells",
        "required_strata": "both_hard_long_values_within_each_of_six_families",
        "stratum_order": "ascending_family_id_then_false_before_true",
        "draw_count_per_stratum": "observed_stratum_example_count",
        "sampler": "sha256_counter_uint64_rejection_v1",
        "sampler_float_encoding": "IEEE754_binary64_big_endian",
        "statistic": "ratio_of_additive_sums",
        "harm_metric_order": [
            "delta_nll_nats_per_token",
            "native_to_candidate_kl_per_token",
            "top1_error",
        ],
        "slice_order": "pooled_then_family_ascending_then_hard_long",
        "standard_error": "sample_sd_across_bootstrap_replicates",
        "studentization": "fixed_coordinate_bootstrap_standard_error",
        "max_t": "one_shared_critical_across_every_slice_and_harm_metric",
        "quantile_rank": "ceil((B+1)*0.95)_clamped_to_B_one_based",
        "critical_value_floor": 0.0,
        "zero_standard_error": (
            "allowed_only_if_every_centered_replicate_is_exactly_zero"
        ),
        "top1_error_upper_bound_clipped_to_known_range": [0.0, 1.0],
        "presentation_only_conversion": "top1_agreement=1-top1_error",
        "pooled_gate_uses_simultaneous_bounds": True,
        "family_and_hard_long_gates_use_frozen_C3_point_thresholds": True,
    }


def d3_bootstrap_policy_sha256() -> str:
    return _hash(d3_bootstrap_policy(), domain=_BOOTSTRAP_POLICY_DOMAIN)


def heldout_evaluation_policy() -> dict[str, object]:
    """Return the canonical model-free B3/C3/D3 statistics contract."""

    return {
        "schema": "fisher_graph.gemma3_layer_multimode_evaluation_policy",
        "format_version": 1,
        "layer_count": LAYER_COUNT,
        "hidden_width": HIDDEN_WIDTH,
        "family_count": FAMILY_COUNT,
        "role_order": [B3_ROLE, C3_ROLE, D3_ROLE],
        "role_data_access": (
            "no_panel_loader_or_role_constructor_in_statistics_core"
        ),
        "B3": {
            "teacher_forced_native_layer_inputs": True,
            "velocity_statistics": [
                "prediction_sse",
                "teacher_energy",
                "prediction_energy",
                "prediction_teacher_dot",
            ],
            "fidelity_accumulation": (
                "unweighted_complete_hidden_rows_with_family_scalar_counts_"
                "divisible_by_H"
            ),
            "routing_accumulation": (
                "unit_weight_per_evaluated_hidden_row_and_sum_chart_q_"
                "equals_row_count"
            ),
            "velocity_nrmse": "sqrt(sum_sse/sum_teacher_energy)",
            "velocity_cosine": (
                "sum_prediction_teacher_dot/"
                "sqrt(sum_prediction_energy*sum_teacher_energy)"
            ),
            "every_layer_velocity_nrmse_maximum": (
                B3_VELOCITY_NRMSE_MAXIMUM
            ),
            "every_layer_velocity_cosine_minimum": (
                B3_AGGREGATE_COSINE_MINIMUM
            ),
            "every_layer_family_nrmse_maximum": (
                B3_FAMILY_NRMSE_MAXIMUM
            ),
            "M_greater_than_1": {
                "chart_conditional_mode_mass": "sum(q*pi)/sum(q)",
                "chart_state_mode_MI_nats": (
                    "H(sum(q*pi)/sum(q))-sum(q*H(pi))/sum(q)"
                ),
                "minimum_conditional_mode_mass": B3_MODE_MASS_MINIMUM,
                "minimum_state_mode_MI_nats": (
                    B3_STATE_MODE_MI_MINIMUM_NATS
                ),
            },
        },
        "C3": {
            "requires_strict_passing_B3_receipt": True,
            "B3_C3_must_match": [
                "source_model_sha256",
                "panel_manifest_sha256",
                "evaluation_protocol_sha256",
                "selection_receipt_sha256",
            ],
            "candidate_execution": "recursive_source_free_full_stack",
            "resampling_unit_record": "whole_example_additive_sums",
            "per_example_fields": [
                "predicted_token_count",
                "delta_nll_sum_nats",
                "native_to_candidate_kl_sum_nats",
                "top1_agreement_count",
            ],
            "scored_positions": (
                "identical_teacher_forced_next_token_positions_for_native_"
                "and_candidate"
            ),
            "delta_nll_sum_nats": (
                "sum(candidate_token_NLL-native_token_NLL)"
            ),
            "native_to_candidate_kl_sum_nats": (
                "sum_positions_sum_full_vocabulary_"
                "p_native*log(p_native/p_candidate)"
            ),
            "top1_agreement_count": (
                "count(argmax_native==argmax_candidate)_with_lowest_"
                "vocabulary_index_tie_break"
            ),
            "slice_order": "pooled_then_family_ascending_then_hard_long",
            "metric_denominator": "sum_predicted_token_count_within_slice",
            "top1_error": "1-sum_agreement_count/sum_token_count",
            "pooled_and_hard_long": {
                "delta_nll_maximum": C3_POOLED_DELTA_NLL_MAXIMUM,
                "KL_maximum": C3_POOLED_KL_MAXIMUM,
                "top1_agreement_minimum": (
                    C3_POOLED_TOP1_AGREEMENT_MINIMUM
                ),
            },
            "every_family": {
                "delta_nll_maximum": C3_FAMILY_DELTA_NLL_MAXIMUM,
                "KL_maximum": C3_FAMILY_KL_MAXIMUM,
                "top1_agreement_minimum": (
                    C3_FAMILY_TOP1_AGREEMENT_MINIMUM
                ),
            },
        },
        "D3": {
            "candidate_identity_must_equal_C3": True,
            "C3_D3_example_identity_sets_must_be_disjoint": True,
            "bootstrap_policy_sha256": d3_bootstrap_policy_sha256(),
            "pooled_gate_uses_simultaneous_bounds": True,
            "family_and_hard_long_repeat_C3_point_gates": True,
            "resource_closure": (
                "compiled_full=native_full-native_boundary+candidate_boundary"
            ),
            "formula_and_measured_parameter_reduction_required": True,
            "serialized_full_model_byte_reduction_required": True,
            "serialized_byte_comparison": (
                "native_and_compiled_regular_file_sizes_from_the_same_frozen_"
                "serialization_protocol_dtype_metadata_compression_and_"
                "sharding_policy"
            ),
            "all_18_compiled_layer_call_counts_equal_forward_call_count": True,
            "native_module_projection_fallback_alias_inventory_counts": (
                "all_exactly_zero"
            ),
            "no_generic_operator_absence_MAC_or_latency_claim": True,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_raw_tokens": False,
            "contains_raw_logits": False,
            "contains_model_weights": False,
            "contains_activation_rows": False,
            "contains_only_scalar_example_aggregates_for_full_stack": True,
        },
    }


def heldout_evaluation_policy_sha256() -> str:
    return _hash(
        heldout_evaluation_policy(),
        domain=_EVALUATION_POLICY_DOMAIN,
    )


def _strata(
    records: Sequence[FullStackExampleAggregate],
) -> tuple[tuple[str, bool, tuple[FullStackExampleAggregate, ...]], ...]:
    families = tuple(sorted({value.family_id for value in records}))
    result: list[tuple[str, bool, tuple[FullStackExampleAggregate, ...]]] = []
    for family in families:
        for hard_long in (False, True):
            members = tuple(
                value
                for value in records
                if value.family_id == family and value.hard_long is hard_long
            )
            if not members:
                raise ValueError(
                    "every family x hard_long bootstrap stratum must be nonempty"
                )
            result.append((family, hard_long, members))
    return tuple(result)


def _sample_index(
    population_size: int,
    *,
    replicate_ordinal: int,
    stratum_ordinal: int,
    draw_ordinal: int,
) -> int:
    """Version-independent SHA-256 counter sampler with rejection."""

    _require_int(population_size, label="population_size", minimum=1)
    limit = (1 << 64) - ((1 << 64) % population_size)
    attempt = 0
    while True:
        payload = (
            bytes.fromhex(BOOTSTRAP_SEED_SHA256)
            + struct.pack(
                ">QQQQ",
                replicate_ordinal,
                stratum_ordinal,
                draw_ordinal,
                attempt,
            )
        )
        value = int.from_bytes(
            hashlib.sha256(_SAMPLER_DOMAIN + payload).digest()[:8],
            "big",
        )
        if value < limit:
            return value % population_size
        attempt += 1


def _bootstrap_matrix(
    records: Sequence[FullStackExampleAggregate],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[float, ...], ...],
    str,
]:
    """Return coordinate ids, strata ids, replicates, and binary digest."""

    observed_slices = _metric_slices(records)
    slice_ids = tuple(value.slice_id for value in observed_slices)
    coordinate_ids = tuple(
        f"{slice_id}|{metric}"
        for slice_id in slice_ids
        for metric in ("delta_nll", "native_to_candidate_kl", "top1_error")
    )
    strata = _strata(records)
    stratum_ids = tuple(
        f"family:{family}|hard_long:{int(hard_long)}"
        for family, hard_long, _members in strata
    )
    family_slice_index = {
        slice_id.removeprefix("family:"): index
        for index, slice_id in enumerate(slice_ids)
        if slice_id.startswith("family:")
    }
    hard_long_index = slice_ids.index("hard_long")
    rows: list[tuple[float, ...]] = []
    digest = hashlib.sha256(_BOOTSTRAP_REPLICATE_DOMAIN)
    for replicate in range(BOOTSTRAP_REPLICATE_COUNT):
        # Token count, delta terms, KL terms, agreement count for each slice.
        # Float sums use fsum so observed and resampled additive closures do
        # not depend on the traversal order of identical evidence.
        totals = [[0, [], [], 0] for _ in slice_ids]
        for stratum_ordinal, (family, hard_long, members) in enumerate(strata):
            for draw in range(len(members)):
                selected = members[
                    _sample_index(
                        len(members),
                        replicate_ordinal=replicate,
                        stratum_ordinal=stratum_ordinal,
                        draw_ordinal=draw,
                    )
                ]
                target_indices = [0, family_slice_index[family]]
                if hard_long:
                    target_indices.append(hard_long_index)
                for index in target_indices:
                    totals[index][0] += selected.predicted_token_count
                    totals[index][1].append(selected.delta_nll_sum_nats)
                    totals[index][2].append(
                        selected.native_to_candidate_kl_sum_nats
                    )
                    totals[index][3] += selected.top1_agreement_count
        coordinates: list[float] = []
        for token_count, delta_terms, kl_terms, agreement_count in totals:
            if token_count == 0:
                raise ValueError("bootstrap produced an empty metric slice")
            delta_sum = math.fsum(delta_terms)
            kl_sum = math.fsum(kl_terms)
            coordinates.extend(
                (
                    float(delta_sum / token_count),
                    float(kl_sum / token_count),
                    float(1.0 - agreement_count / token_count),
                )
            )
        row = tuple(coordinates)
        for value in row:
            if not math.isfinite(value):
                raise ValueError("bootstrap produced a non-finite coordinate")
            digest.update(struct.pack(">d", value))
        rows.append(row)
    return coordinate_ids, stratum_ids, tuple(rows), digest.hexdigest()


@dataclass(frozen=True, slots=True)
class D3SimultaneousBoundReceipt:
    coordinate_ordinal: int
    slice_id: str
    harm_metric: str
    point_estimate: float
    bootstrap_standard_error: float
    simultaneous_upper_bound: float
    presentation_metric: str
    presentation_point_estimate: float
    presentation_one_sided_bound: float
    artifact_sha256: str = ""
    artifact_kind: str = _BOUND_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_int(self.coordinate_ordinal, label="coordinate_ordinal")
        _require_string(self.slice_id, label="slice_id")
        if self.harm_metric not in (
            "delta_nll",
            "native_to_candidate_kl",
            "top1_error",
        ):
            raise ValueError("harm_metric is invalid")
        _require_float(self.point_estimate, label="point_estimate")
        _require_float(
            self.bootstrap_standard_error,
            label="bootstrap_standard_error",
            minimum=0.0,
        )
        _require_float(
            self.simultaneous_upper_bound,
            label="simultaneous_upper_bound",
        )
        if self.simultaneous_upper_bound + _ABS_TOL < self.point_estimate:
            raise ValueError("simultaneous upper bound is below its point")
        _require_string(self.presentation_metric, label="presentation_metric")
        _require_float(
            self.presentation_point_estimate,
            label="presentation_point_estimate",
        )
        _require_float(
            self.presentation_one_sided_bound,
            label="presentation_one_sided_bound",
        )
        if self.harm_metric == "top1_error":
            if self.presentation_metric != "top1_agreement":
                raise ValueError("top1 error presentation metric is invalid")
            if not _close(
                self.presentation_point_estimate,
                1.0 - self.point_estimate,
            ):
                raise ValueError("top1 presentation point does not recompute")
            if not _close(
                self.presentation_one_sided_bound,
                1.0 - self.simultaneous_upper_bound,
            ):
                raise ValueError("top1 presentation lower bound does not recompute")
        else:
            expected_metric = (
                "delta_nll_nats_per_token"
                if self.harm_metric == "delta_nll"
                else "native_to_candidate_kl_per_token"
            )
            if self.presentation_metric != expected_metric:
                raise ValueError("harm presentation metric is invalid")
            if not _close(self.presentation_point_estimate, self.point_estimate):
                raise ValueError("harm presentation point does not recompute")
            if not _close(
                self.presentation_one_sided_bound,
                self.simultaneous_upper_bound,
            ):
                raise ValueError("harm presentation upper bound does not recompute")
        _validate_header(self, _BOUND_KIND)
        _install_hash(self, self._payload(), _BOUND_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "coordinate_ordinal": self.coordinate_ordinal,
            "slice_id": self.slice_id,
            "harm_metric": self.harm_metric,
            "point_estimate": self.point_estimate,
            "bootstrap_standard_error": self.bootstrap_standard_error,
            "simultaneous_upper_bound": self.simultaneous_upper_bound,
            "presentation_metric": self.presentation_metric,
            "presentation_point_estimate": self.presentation_point_estimate,
            "presentation_one_sided_bound": self.presentation_one_sided_bound,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> D3SimultaneousBoundReceipt:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "coordinate_ordinal",
                "slice_id",
                "harm_metric",
                "point_estimate",
                "bootstrap_standard_error",
                "simultaneous_upper_bound",
                "presentation_metric",
                "presentation_point_estimate",
                "presentation_one_sided_bound",
                "artifact_sha256",
            },
            label="D3 simultaneous bound",
        )
        return cls(**state)  # type: ignore[arg-type]


def _bootstrap_bounds(
    records: Sequence[FullStackExampleAggregate],
) -> tuple[
    tuple[str, ...],
    str,
    float,
    tuple[D3SimultaneousBoundReceipt, ...],
]:
    coordinate_ids, stratum_ids, matrix, replicate_digest = _bootstrap_matrix(records)
    observed = tuple(
        coordinate
        for metric_slice in _metric_slices(records)
        for coordinate in metric_slice.harm_coordinates
    )
    if len(observed) != len(coordinate_ids):
        raise AssertionError("bootstrap coordinate shape mismatch")
    replicate_count = len(matrix)
    means = tuple(
        math.fsum(row[index] for row in matrix) / replicate_count
        for index in range(len(observed))
    )
    standard_errors = tuple(
        math.sqrt(
            math.fsum(
                (row[index] - means[index]) ** 2 for row in matrix
            )
            / (replicate_count - 1)
        )
        for index in range(len(observed))
    )
    active: list[int] = []
    for index, standard_error in enumerate(standard_errors):
        centered = tuple(row[index] - observed[index] for row in matrix)
        if standard_error == 0.0:
            if any(value != 0.0 for value in centered):
                raise ValueError(
                    "zero bootstrap SE has a nonzero centered replicate"
                )
        else:
            active.append(index)
    max_t_values = []
    for row in matrix:
        if active:
            max_t_values.append(
                max(
                    (row[index] - observed[index]) / standard_errors[index]
                    for index in active
                )
            )
        else:
            max_t_values.append(0.0)
    rank = math.ceil(
        (replicate_count + 1) * BOOTSTRAP_CONFIDENCE_LEVEL
    )
    rank = min(replicate_count, max(1, rank))
    critical = max(0.0, sorted(max_t_values)[rank - 1])
    bounds: list[D3SimultaneousBoundReceipt] = []
    metric_names = ("delta_nll", "native_to_candidate_kl", "top1_error")
    for index, coordinate_id in enumerate(coordinate_ids):
        slice_id, metric_name = coordinate_id.rsplit("|", 1)
        if metric_name != metric_names[index % len(metric_names)]:
            raise AssertionError("bootstrap coordinate order mismatch")
        point = observed[index]
        upper = point + critical * standard_errors[index]
        if metric_name == "top1_error":
            upper = min(1.0, max(0.0, upper))
            presentation_metric = "top1_agreement"
            presentation_point = 1.0 - point
            presentation_bound = 1.0 - upper
        else:
            presentation_metric = (
                "delta_nll_nats_per_token"
                if metric_name == "delta_nll"
                else "native_to_candidate_kl_per_token"
            )
            presentation_point = point
            presentation_bound = upper
        bounds.append(
            D3SimultaneousBoundReceipt(
                coordinate_ordinal=index,
                slice_id=slice_id,
                harm_metric=metric_name,
                point_estimate=float(point),
                bootstrap_standard_error=float(standard_errors[index]),
                simultaneous_upper_bound=float(upper),
                presentation_metric=presentation_metric,
                presentation_point_estimate=float(presentation_point),
                presentation_one_sided_bound=float(presentation_bound),
            )
        )
    return stratum_ids, replicate_digest, float(critical), tuple(bounds)


@dataclass(frozen=True, slots=True)
class D3ResourceComparisonReceipt:
    """Exact parameter and same-protocol full-model byte evidence.

    Native and compiled byte counts are regular-file sizes produced by the
    single canonical serialization protocol digest.  Unlike containers,
    compression modes, dtypes, metadata policies, or sharding layouts cannot
    be compared by this receipt.
    """

    candidate_identity_sha256: str
    parameter_formula_receipt_sha256: str
    serialization_protocol_sha256: str
    native_model_file_sha256: str
    compiled_model_file_sha256: str
    native_full_model_parameter_count: int
    native_replaced_boundary_parameter_count: int
    compiled_replacement_boundary_parameter_count: int
    expected_compiled_full_model_parameter_count: int
    measured_compiled_full_model_parameter_count: int
    native_serialized_model_bytes: int
    compiled_serialized_model_bytes: int
    formula_parameter_reduction_passed: bool
    measured_parameter_reduction_passed: bool
    serialized_byte_reduction_passed: bool
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _RESOURCE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        for label in (
            "candidate_identity_sha256",
            "parameter_formula_receipt_sha256",
            "serialization_protocol_sha256",
            "native_model_file_sha256",
            "compiled_model_file_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        for label in (
            "native_full_model_parameter_count",
            "native_replaced_boundary_parameter_count",
            "compiled_replacement_boundary_parameter_count",
            "expected_compiled_full_model_parameter_count",
            "measured_compiled_full_model_parameter_count",
            "native_serialized_model_bytes",
            "compiled_serialized_model_bytes",
        ):
            _require_int(getattr(self, label), label=label, minimum=1)
        expected_full = (
            self.native_full_model_parameter_count
            - self.native_replaced_boundary_parameter_count
            + self.compiled_replacement_boundary_parameter_count
        )
        if self.expected_compiled_full_model_parameter_count != expected_full:
            raise ValueError("expected compiled parameter count does not close")
        formula_pass = (
            self.compiled_replacement_boundary_parameter_count
            < self.native_replaced_boundary_parameter_count
        )
        measured_pass = (
            self.measured_compiled_full_model_parameter_count
            == self.expected_compiled_full_model_parameter_count
            and self.measured_compiled_full_model_parameter_count
            < self.native_full_model_parameter_count
        )
        bytes_pass = (
            self.compiled_serialized_model_bytes
            < self.native_serialized_model_bytes
        )
        for label, value, expected in (
            (
                "formula_parameter_reduction_passed",
                self.formula_parameter_reduction_passed,
                formula_pass,
            ),
            (
                "measured_parameter_reduction_passed",
                self.measured_parameter_reduction_passed,
                measured_pass,
            ),
            (
                "serialized_byte_reduction_passed",
                self.serialized_byte_reduction_passed,
                bytes_pass,
            ),
        ):
            _require_bool(value, label=label)
            if value is not expected:
                raise ValueError(f"{label} does not recompute")
        _require_bool(self.passed, label="passed")
        if self.passed is not (formula_pass and measured_pass and bytes_pass):
            raise ValueError("resource comparison passed does not recompute")
        _validate_header(self, _RESOURCE_KIND)
        _install_hash(self, self._payload(), _RESOURCE_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "parameter_formula_receipt_sha256": self.parameter_formula_receipt_sha256,
            "serialization_protocol_sha256": self.serialization_protocol_sha256,
            "native_model_file_sha256": self.native_model_file_sha256,
            "compiled_model_file_sha256": self.compiled_model_file_sha256,
            "native_full_model_parameter_count": self.native_full_model_parameter_count,
            "native_replaced_boundary_parameter_count": (
                self.native_replaced_boundary_parameter_count
            ),
            "compiled_replacement_boundary_parameter_count": (
                self.compiled_replacement_boundary_parameter_count
            ),
            "expected_compiled_full_model_parameter_count": (
                self.expected_compiled_full_model_parameter_count
            ),
            "measured_compiled_full_model_parameter_count": (
                self.measured_compiled_full_model_parameter_count
            ),
            "native_serialized_model_bytes": self.native_serialized_model_bytes,
            "compiled_serialized_model_bytes": self.compiled_serialized_model_bytes,
            "formula_parameter_reduction_passed": (
                self.formula_parameter_reduction_passed
            ),
            "measured_parameter_reduction_passed": (
                self.measured_parameter_reduction_passed
            ),
            "serialized_byte_reduction_passed": (
                self.serialized_byte_reduction_passed
            ),
            "passed": self.passed,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> D3ResourceComparisonReceipt:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "candidate_identity_sha256",
                "parameter_formula_receipt_sha256",
                "serialization_protocol_sha256",
                "native_model_file_sha256",
                "compiled_model_file_sha256",
                "native_full_model_parameter_count",
                "native_replaced_boundary_parameter_count",
                "compiled_replacement_boundary_parameter_count",
                "expected_compiled_full_model_parameter_count",
                "measured_compiled_full_model_parameter_count",
                "native_serialized_model_bytes",
                "compiled_serialized_model_bytes",
                "formula_parameter_reduction_passed",
                "measured_parameter_reduction_passed",
                "serialized_byte_reduction_passed",
                "passed",
                "artifact_sha256",
            },
            label="D3 resource comparison",
        )
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class D3NoNativeMLPTraceReceipt:
    candidate_identity_sha256: str
    role_capture_receipt_sha256: str
    trace_protocol_sha256: str
    module_inventory_sha256: str
    trace_event_stream_sha256: str
    proof_backend: str
    traced_forward_call_count: int
    traced_example_count: int
    traced_predicted_token_count: int
    compiled_layer_call_counts: tuple[int, ...]
    native_mlp_module_inventory_count: int
    native_mlp_projection_inventory_count: int
    native_mlp_fallback_inventory_count: int
    native_mlp_parameter_storage_alias_count: int
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _TRACE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        for label in (
            "candidate_identity_sha256",
            "role_capture_receipt_sha256",
            "trace_protocol_sha256",
            "module_inventory_sha256",
            "trace_event_stream_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        _require_string(self.proof_backend, label="proof_backend")
        for label in (
            "traced_forward_call_count",
            "traced_example_count",
            "traced_predicted_token_count",
        ):
            _require_int(getattr(self, label), label=label, minimum=1)
        if (
            type(self.compiled_layer_call_counts) is not tuple
            or len(self.compiled_layer_call_counts) != LAYER_COUNT
        ):
            raise ValueError("compiled_layer_call_counts must contain 18 entries")
        for index, value in enumerate(self.compiled_layer_call_counts):
            _require_int(value, label=f"compiled_layer_call_counts[{index}]")
        zero_fields = (
            self.native_mlp_module_inventory_count,
            self.native_mlp_projection_inventory_count,
            self.native_mlp_fallback_inventory_count,
            self.native_mlp_parameter_storage_alias_count,
        )
        for label, value in zip(
            (
                "native_mlp_module_inventory_count",
                "native_mlp_projection_inventory_count",
                "native_mlp_fallback_inventory_count",
                "native_mlp_parameter_storage_alias_count",
            ),
            zero_fields,
            strict=True,
        ):
            _require_int(value, label=label)
        expected_pass = (
            all(
                value == self.traced_forward_call_count
                for value in self.compiled_layer_call_counts
            )
            and all(value == 0 for value in zero_fields)
        )
        _require_bool(self.passed, label="passed")
        if self.passed is not expected_pass:
            raise ValueError("no-native-MLP trace passed does not recompute")
        _validate_header(self, _TRACE_KIND)
        _install_hash(self, self._payload(), _TRACE_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "role_capture_receipt_sha256": self.role_capture_receipt_sha256,
            "trace_protocol_sha256": self.trace_protocol_sha256,
            "module_inventory_sha256": self.module_inventory_sha256,
            "trace_event_stream_sha256": self.trace_event_stream_sha256,
            "proof_backend": self.proof_backend,
            "traced_forward_call_count": self.traced_forward_call_count,
            "traced_example_count": self.traced_example_count,
            "traced_predicted_token_count": self.traced_predicted_token_count,
            "compiled_layer_call_counts": self.compiled_layer_call_counts,
            "native_mlp_module_inventory_count": (
                self.native_mlp_module_inventory_count
            ),
            "native_mlp_projection_inventory_count": (
                self.native_mlp_projection_inventory_count
            ),
            "native_mlp_fallback_inventory_count": (
                self.native_mlp_fallback_inventory_count
            ),
            "native_mlp_parameter_storage_alias_count": (
                self.native_mlp_parameter_storage_alias_count
            ),
            "passed": self.passed,
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> D3NoNativeMLPTraceReceipt:
        _strict(
            state,
            fields={
                "artifact_kind",
                "format_version",
                "candidate_identity_sha256",
                "role_capture_receipt_sha256",
                "trace_protocol_sha256",
                "module_inventory_sha256",
                "trace_event_stream_sha256",
                "proof_backend",
                "traced_forward_call_count",
                "traced_example_count",
                "traced_predicted_token_count",
                "compiled_layer_call_counts",
                "native_mlp_module_inventory_count",
                "native_mlp_projection_inventory_count",
                "native_mlp_fallback_inventory_count",
                "native_mlp_parameter_storage_alias_count",
                "passed",
                "artifact_sha256",
            },
            label="D3 no-native-MLP trace",
        )
        arguments = dict(state)
        arguments["compiled_layer_call_counts"] = tuple(
            _list(
                state["compiled_layer_call_counts"],
                label="compiled_layer_call_counts",
            )
        )
        return cls(**arguments)  # type: ignore[arg-type]


def _d3_bound_failures(
    bounds: Sequence[D3SimultaneousBoundReceipt],
) -> tuple[str, ...]:
    pooled = {
        value.harm_metric: value
        for value in bounds
        if value.slice_id == "pooled"
    }
    if set(pooled) != {
        "delta_nll",
        "native_to_candidate_kl",
        "top1_error",
    }:
        raise ValueError("D3 pooled simultaneous bounds are incomplete")
    failures: list[str] = []
    if (
        pooled["delta_nll"].simultaneous_upper_bound
        > C3_POOLED_DELTA_NLL_MAXIMUM
    ):
        failures.append("pooled_bound.delta_nll")
    if (
        pooled["native_to_candidate_kl"].simultaneous_upper_bound
        > C3_POOLED_KL_MAXIMUM
    ):
        failures.append("pooled_bound.kl")
    if (
        pooled["top1_error"].presentation_one_sided_bound
        < C3_POOLED_TOP1_AGREEMENT_MINIMUM
    ):
        failures.append("pooled_bound.top1_agreement")
    return tuple(sorted(failures))


@dataclass(frozen=True, slots=True)
class D3FinalEvaluationReceipt:
    source_model_sha256: str
    c3_evaluation_sha256: str
    c3_candidate_identity_sha256: str
    c3_example_identity_sha256s: tuple[str, ...]
    panel_manifest_sha256: str
    role_capture_receipt_sha256: str
    evaluation_protocol_sha256: str
    candidate_identity: CompiledCandidateIdentityReceipt
    example_aggregates: tuple[FullStackExampleAggregate, ...]
    metric_slices: tuple[FullStackMetricSliceReceipt, ...]
    bootstrap_policy_sha256: str
    bootstrap_stratum_ids: tuple[str, ...]
    bootstrap_replicate_digest_sha256: str
    bootstrap_max_t_critical_value: float
    simultaneous_bounds: tuple[D3SimultaneousBoundReceipt, ...]
    resource_comparison: D3ResourceComparisonReceipt
    no_native_mlp_trace: D3NoNativeMLPTraceReceipt
    point_failed_gates: tuple[str, ...]
    bound_failed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    passed: bool
    artifact_sha256: str = ""
    artifact_kind: str = _D3_KIND
    format_version: int = _FORMAT_VERSION
    role: str = D3_ROLE
    bootstrap_is_distribution_free: bool = False
    contains_prompt_text: bool = False
    contains_raw_tokens: bool = False
    contains_raw_logits: bool = False
    contains_model_weights: bool = False
    contains_activation_rows: bool = False
    contains_scalar_example_aggregates: bool = True

    def __post_init__(self) -> None:
        for label in (
            "source_model_sha256",
            "c3_evaluation_sha256",
            "c3_candidate_identity_sha256",
            "panel_manifest_sha256",
            "role_capture_receipt_sha256",
            "evaluation_protocol_sha256",
            "bootstrap_policy_sha256",
            "bootstrap_replicate_digest_sha256",
        ):
            _require_sha(getattr(self, label), label=label)
        if self.role != D3_ROLE:
            raise ValueError("D3 receipt role is invalid")
        if self.evaluation_protocol_sha256 != heldout_evaluation_policy_sha256():
            raise ValueError("D3 evaluation protocol SHA-256 is not frozen")
        if self.bootstrap_is_distribution_free is not False:
            raise ValueError("D3 bootstrap may not claim distribution-free coverage")
        if type(self.candidate_identity) is not CompiledCandidateIdentityReceipt:
            raise TypeError("candidate_identity is invalid")
        if self.candidate_identity.artifact_sha256 != self.c3_candidate_identity_sha256:
            raise ValueError("D3 candidate is not unchanged from C3")
        if (
            type(self.c3_example_identity_sha256s) is not tuple
            or not self.c3_example_identity_sha256s
        ):
            raise ValueError("C3 example identities must be a nonempty tuple")
        for index, value in enumerate(self.c3_example_identity_sha256s):
            _require_sha(value, label=f"c3_example_identity_sha256s[{index}]")
        if self.c3_example_identity_sha256s != tuple(
            sorted(set(self.c3_example_identity_sha256s))
        ):
            raise ValueError("C3 example identities must be canonical and unique")
        records = _canonical_records(self.example_aggregates, role=D3_ROLE)
        d3_identities = {value.example_identity_sha256 for value in records}
        if d3_identities.intersection(self.c3_example_identity_sha256s):
            raise ValueError("C3 and D3 example identities must be disjoint")
        expected_slices = _metric_slices(records)
        if type(self.metric_slices) is not tuple:
            raise TypeError("metric_slices must be a tuple")
        if tuple(value.state_dict() for value in self.metric_slices) != tuple(
            value.state_dict() for value in expected_slices
        ):
            raise ValueError("D3 metric slices do not recompute")
        if self.bootstrap_policy_sha256 != d3_bootstrap_policy_sha256():
            raise ValueError("D3 bootstrap policy SHA-256 is not frozen")
        if type(self.bootstrap_stratum_ids) is not tuple:
            raise TypeError("bootstrap_stratum_ids must be a tuple")
        if type(self.simultaneous_bounds) is not tuple:
            raise TypeError("simultaneous_bounds must be a tuple")
        _require_float(
            self.bootstrap_max_t_critical_value,
            label="bootstrap_max_t_critical_value",
            minimum=0.0,
        )
        expected_strata, expected_digest, expected_critical, expected_bounds = (
            _bootstrap_bounds(records)
        )
        if self.bootstrap_stratum_ids != expected_strata:
            raise ValueError("D3 bootstrap strata do not recompute")
        if self.bootstrap_replicate_digest_sha256 != expected_digest:
            raise ValueError("D3 bootstrap replicate digest does not recompute")
        if not _close(self.bootstrap_max_t_critical_value, expected_critical):
            raise ValueError("D3 bootstrap max-T critical value does not recompute")
        if tuple(value.state_dict() for value in self.simultaneous_bounds) != tuple(
            value.state_dict() for value in expected_bounds
        ):
            raise ValueError("D3 simultaneous bounds do not recompute")
        expected_point_failures = _point_gate_failures(expected_slices)
        expected_bound_failures = _d3_bound_failures(expected_bounds)
        if self.point_failed_gates != expected_point_failures:
            raise ValueError("D3 point failed gates do not recompute")
        if self.bound_failed_gates != expected_bound_failures:
            raise ValueError("D3 bound failed gates do not recompute")
        if type(self.resource_comparison) is not D3ResourceComparisonReceipt:
            raise TypeError("resource_comparison is invalid")
        if type(self.no_native_mlp_trace) is not D3NoNativeMLPTraceReceipt:
            raise TypeError("no_native_mlp_trace is invalid")
        if (
            self.resource_comparison.candidate_identity_sha256
            != self.candidate_identity.artifact_sha256
        ):
            raise ValueError("D3 resource comparison candidate identity mismatch")
        trace = self.no_native_mlp_trace
        if trace.candidate_identity_sha256 != self.candidate_identity.artifact_sha256:
            raise ValueError("D3 trace candidate identity mismatch")
        if trace.role_capture_receipt_sha256 != self.role_capture_receipt_sha256:
            raise ValueError("D3 trace role-capture identity mismatch")
        if trace.traced_example_count != len(records):
            raise ValueError("D3 trace example count does not close")
        if trace.traced_predicted_token_count != sum(
            value.predicted_token_count for value in records
        ):
            raise ValueError("D3 trace token count does not close")
        expected_failures = list(expected_point_failures)
        expected_failures.extend(expected_bound_failures)
        if not self.resource_comparison.passed:
            expected_failures.append("resource_comparison")
        if not trace.passed:
            expected_failures.append("no_native_mlp_trace")
        canonical_failures = tuple(sorted(expected_failures))
        if self.failed_gates != canonical_failures:
            raise ValueError("D3 failed_gates do not recompute")
        _require_bool(self.passed, label="passed")
        if self.passed is not (not self.failed_gates):
            raise ValueError("D3 passed does not recompute")
        for label, expected in _SAFETY_FIELDS.items():
            if getattr(self, label) is not expected:
                raise ValueError("D3 safety metadata is invalid")
        _validate_header(self, _D3_KIND)
        _install_hash(self, self._payload(), _D3_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "role": self.role,
            "source_model_sha256": self.source_model_sha256,
            "c3_evaluation_sha256": self.c3_evaluation_sha256,
            "c3_candidate_identity_sha256": self.c3_candidate_identity_sha256,
            "c3_example_identity_sha256s": self.c3_example_identity_sha256s,
            "panel_manifest_sha256": self.panel_manifest_sha256,
            "role_capture_receipt_sha256": self.role_capture_receipt_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "candidate_identity": self.candidate_identity.state_dict(),
            "example_aggregates": tuple(
                value.state_dict() for value in self.example_aggregates
            ),
            "metric_slices": tuple(value.state_dict() for value in self.metric_slices),
            "bootstrap_policy_sha256": self.bootstrap_policy_sha256,
            "bootstrap_stratum_ids": self.bootstrap_stratum_ids,
            "bootstrap_replicate_digest_sha256": (
                self.bootstrap_replicate_digest_sha256
            ),
            "bootstrap_max_t_critical_value": (
                self.bootstrap_max_t_critical_value
            ),
            "simultaneous_bounds": tuple(
                value.state_dict() for value in self.simultaneous_bounds
            ),
            "resource_comparison": self.resource_comparison.state_dict(),
            "no_native_mlp_trace": self.no_native_mlp_trace.state_dict(),
            "point_failed_gates": self.point_failed_gates,
            "bound_failed_gates": self.bound_failed_gates,
            "failed_gates": self.failed_gates,
            "passed": self.passed,
            "bootstrap_is_distribution_free": self.bootstrap_is_distribution_free,
            **_SAFETY_FIELDS,
        }

    def state_dict(self) -> dict[str, object]:
        return _state(
            {**self._payload(), "artifact_sha256": self.artifact_sha256}
        )  # type: ignore[return-value]

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> D3FinalEvaluationReceipt:
        fields = {
            "artifact_kind",
            "format_version",
            "role",
            "source_model_sha256",
            "c3_evaluation_sha256",
            "c3_candidate_identity_sha256",
            "c3_example_identity_sha256s",
            "panel_manifest_sha256",
            "role_capture_receipt_sha256",
            "evaluation_protocol_sha256",
            "candidate_identity",
            "example_aggregates",
            "metric_slices",
            "bootstrap_policy_sha256",
            "bootstrap_stratum_ids",
            "bootstrap_replicate_digest_sha256",
            "bootstrap_max_t_critical_value",
            "simultaneous_bounds",
            "resource_comparison",
            "no_native_mlp_trace",
            "point_failed_gates",
            "bound_failed_gates",
            "failed_gates",
            "passed",
            "bootstrap_is_distribution_free",
            "artifact_sha256",
            *_SAFETY_FIELDS,
        }
        _strict(state, fields=fields, label="D3 final evaluation")
        arguments = dict(state)
        arguments["c3_example_identity_sha256s"] = tuple(
            _list(
                state["c3_example_identity_sha256s"],
                label="c3_example_identity_sha256s",
            )
        )
        candidate_state = state["candidate_identity"]
        if not isinstance(candidate_state, Mapping):
            raise TypeError("candidate_identity must be a JSON object")
        arguments["candidate_identity"] = (
            CompiledCandidateIdentityReceipt.from_state_dict(candidate_state)
        )
        arguments["example_aggregates"] = tuple(
            FullStackExampleAggregate.from_state_dict(value)
            for value in _list(
                state["example_aggregates"], label="example_aggregates"
            )
        )
        arguments["metric_slices"] = tuple(
            FullStackMetricSliceReceipt.from_state_dict(value)
            for value in _list(state["metric_slices"], label="metric_slices")
        )
        arguments["bootstrap_stratum_ids"] = tuple(
            _list(
                state["bootstrap_stratum_ids"], label="bootstrap_stratum_ids"
            )
        )
        arguments["simultaneous_bounds"] = tuple(
            D3SimultaneousBoundReceipt.from_state_dict(value)
            for value in _list(
                state["simultaneous_bounds"], label="simultaneous_bounds"
            )
        )
        resource_state = state["resource_comparison"]
        if not isinstance(resource_state, Mapping):
            raise TypeError("resource_comparison must be a JSON object")
        arguments["resource_comparison"] = (
            D3ResourceComparisonReceipt.from_state_dict(resource_state)
        )
        trace_state = state["no_native_mlp_trace"]
        if not isinstance(trace_state, Mapping):
            raise TypeError("no_native_mlp_trace must be a JSON object")
        arguments["no_native_mlp_trace"] = (
            D3NoNativeMLPTraceReceipt.from_state_dict(trace_state)
        )
        for field in (
            "point_failed_gates",
            "bound_failed_gates",
            "failed_gates",
        ):
            arguments[field] = tuple(_list(state[field], label=field))
        return cls(**arguments)  # type: ignore[arg-type]


def evaluate_d3_final(
    example_aggregates: Sequence[FullStackExampleAggregate],
    *,
    c3_evaluation: C3FullStackEvaluationReceipt,
    source_model_sha256: str,
    panel_manifest_sha256: str,
    role_capture_receipt_sha256: str,
    evaluation_protocol_sha256: str,
    candidate_identity: CompiledCandidateIdentityReceipt,
    resource_comparison: D3ResourceComparisonReceipt,
    no_native_mlp_trace: D3NoNativeMLPTraceReceipt,
) -> D3FinalEvaluationReceipt:
    """Evaluate D3 scalar records after a terminal passing C3 receipt."""

    if type(c3_evaluation) is not C3FullStackEvaluationReceipt:
        raise TypeError("c3_evaluation must be the exact typed C3 receipt")
    replayed_c3 = C3FullStackEvaluationReceipt.from_state_dict(
        json.loads(json.dumps(c3_evaluation.state_dict()))
    )
    if replayed_c3.state_dict() != c3_evaluation.state_dict():
        raise ValueError("C3 evaluation did not survive strict replay")
    if not c3_evaluation.passed:
        raise ValueError("D3 is unavailable until C3 passes")
    if c3_evaluation.source_model_sha256 != source_model_sha256:
        raise ValueError("C3 and D3 source-model identities differ")
    if c3_evaluation.panel_manifest_sha256 != panel_manifest_sha256:
        raise ValueError("C3 and D3 do not share the frozen panel manifest")
    if (
        c3_evaluation.candidate_identity.artifact_sha256
        != candidate_identity.artifact_sha256
    ):
        raise ValueError("D3 candidate is not unchanged from C3")
    records = _canonical_records(example_aggregates, role=D3_ROLE)
    c3_identities = tuple(
        value.example_identity_sha256
        for value in c3_evaluation.example_aggregates
    )
    if set(c3_identities).intersection(
        value.example_identity_sha256 for value in records
    ):
        raise ValueError("C3 and D3 example identities are not disjoint")
    slices = _metric_slices(records)
    strata, replicate_digest, critical, bounds = _bootstrap_bounds(records)
    point_failures = _point_gate_failures(slices)
    bound_failures = _d3_bound_failures(bounds)
    failures = list(point_failures)
    failures.extend(bound_failures)
    if not resource_comparison.passed:
        failures.append("resource_comparison")
    if not no_native_mlp_trace.passed:
        failures.append("no_native_mlp_trace")
    return D3FinalEvaluationReceipt(
        source_model_sha256=source_model_sha256,
        c3_evaluation_sha256=c3_evaluation.artifact_sha256,
        c3_candidate_identity_sha256=(
            c3_evaluation.candidate_identity.artifact_sha256
        ),
        c3_example_identity_sha256s=c3_identities,
        panel_manifest_sha256=panel_manifest_sha256,
        role_capture_receipt_sha256=role_capture_receipt_sha256,
        evaluation_protocol_sha256=evaluation_protocol_sha256,
        candidate_identity=candidate_identity,
        example_aggregates=records,
        metric_slices=slices,
        bootstrap_policy_sha256=d3_bootstrap_policy_sha256(),
        bootstrap_stratum_ids=strata,
        bootstrap_replicate_digest_sha256=replicate_digest,
        bootstrap_max_t_critical_value=critical,
        simultaneous_bounds=bounds,
        resource_comparison=resource_comparison,
        no_native_mlp_trace=no_native_mlp_trace,
        point_failed_gates=point_failures,
        bound_failed_gates=bound_failures,
        failed_gates=tuple(sorted(failures)),
        passed=not failures,
    )
