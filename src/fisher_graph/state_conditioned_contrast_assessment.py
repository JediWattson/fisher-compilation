"""Generic, source-safe contrast assessment for state-conditioned executors.

The scorer deliberately separates teacher-panel eligibility from candidate
recovery.  Callers supply endpoint tensors that have already been restricted
to one frozen scoring mask.  The returned state contains hashes and scalar
metrics only; it never serializes endpoint tensors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import torch
from torch import Tensor


__all__ = [
    "ContrastAssessmentGates",
    "ContrastAssessmentResult",
    "ContrastDefinition",
    "ContrastFamilyScore",
    "ContrastObservation",
    "ContrastScore",
    "assess_state_conditioned_contrasts",
    "score_state_conditioned_contrast",
]


ContrastRole = Literal["expected_sensitivity", "intended_null"]
DecisionStatus = Literal[
    "invalid",
    "teacher_null_failure",
    "panel_inconclusive",
    "candidate_fail",
    "pass",
]

_ROLES = {"expected_sensitivity", "intended_null"}
_DECISION_PRIORITY = {
    "pass": 0,
    "candidate_fail": 1,
    "panel_inconclusive": 2,
    "teacher_null_failure": 3,
    "invalid": 4,
}
_GATES_DOMAIN = b"fisher-graph:contrast-assessment-gates:v1\0"
_DEFINITION_DOMAIN = b"fisher-graph:contrast-definition:v1\0"
_SCORE_DOMAIN = b"fisher-graph:contrast-score:v1\0"
_FAMILY_DOMAIN = b"fisher-graph:contrast-family-score:v1\0"
_ASSESSMENT_DOMAIN = b"fisher-graph:contrast-assessment:v1\0"


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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    number = _finite(value, label=label, minimum=float(minimum))
    result = int(number)
    if result != number:
        raise ValueError(f"{label} must be an integer")
    return result


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the frozen format")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": tuple(int(width) for width in tensor.shape),
        }
    )
    return hashlib.sha256(
        header + b"\0" + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _tensor_l2(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)))


def _optional_finite(value: float | None, *, label: str) -> None:
    if value is not None:
        _finite(value, label=label)


@dataclass(frozen=True, slots=True)
class ContrastAssessmentGates:
    """Frozen numerical, teacher-eligibility, and candidate-recovery gates."""

    repeat_noise_multiplier: float = 8.0
    epsilon_scale_multiplier: float = 4.0
    minimum_sensitivity_relative_effect: float = 0.01
    maximum_teacher_null_relative_effect: float = 0.001
    maximum_sensitivity_contrast_relative_error: float = 0.35
    minimum_sensitivity_direction_cosine: float = 0.95
    minimum_sensitivity_projection_gain: float = 0.70
    maximum_sensitivity_projection_gain: float = 1.30
    maximum_sensitivity_orthogonal_leakage: float = 0.25
    maximum_candidate_null_relative_effect: float = 0.01
    maximum_candidate_null_relative_error: float = 0.01
    minimum_family_eligible_fraction: float = 0.75
    minimum_family_eligible_count: int = 4
    maximum_family_macro_rms_relative_error: float = 0.25
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        positive = (
            "repeat_noise_multiplier",
            "epsilon_scale_multiplier",
            "minimum_sensitivity_relative_effect",
            "maximum_teacher_null_relative_effect",
            "maximum_sensitivity_contrast_relative_error",
            "maximum_sensitivity_orthogonal_leakage",
            "maximum_candidate_null_relative_effect",
            "maximum_candidate_null_relative_error",
            "maximum_family_macro_rms_relative_error",
        )
        for name in positive:
            if _finite(getattr(self, name), label=name, minimum=0.0) == 0.0:
                raise ValueError(f"{name} must be positive")
        unit_interval = (
            "minimum_sensitivity_direction_cosine",
            "minimum_family_eligible_fraction",
        )
        for name in unit_interval:
            value = _finite(getattr(self, name), label=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        minimum_gain = _finite(
            self.minimum_sensitivity_projection_gain,
            label="minimum_sensitivity_projection_gain",
        )
        maximum_gain = _finite(
            self.maximum_sensitivity_projection_gain,
            label="maximum_sensitivity_projection_gain",
        )
        if minimum_gain > maximum_gain:
            raise ValueError("sensitivity projection-gain interval is reversed")
        _integer(
            self.minimum_family_eligible_count,
            label="minimum_family_eligible_count",
            minimum=1,
        )
        computed = _digest(self._payload(), domain=_GATES_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("contrast-assessment gate hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "artifact_sha256"
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "ContrastAssessmentGates":
        if not isinstance(raw, Mapping):
            raise TypeError("contrast-assessment gates must be a mapping")
        expected = set(cls.__dataclass_fields__)
        _strict_keys(raw, expected=expected, label="contrast-assessment gates")
        values = {
            name: raw[name]
            for name in expected
            if name != "artifact_sha256"
        }
        values["minimum_family_eligible_count"] = _integer(
            values["minimum_family_eligible_count"],
            label="minimum_family_eligible_count",
            minimum=1,
        )
        for name in tuple(values):
            if name != "minimum_family_eligible_count":
                values[name] = _finite(values[name], label=name)
        return cls(
            **values,  # type: ignore[arg-type]
            artifact_sha256=_require_sha256(
                raw["artifact_sha256"],
                label="contrast-assessment gate hash",
            ),
        )


@dataclass(frozen=True, slots=True)
class ContrastDefinition:
    """One preregistered zero-sum, L1-normalized endpoint contrast."""

    contrast_id: str
    family: str
    role: ContrastRole
    coefficients: tuple[float, ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.contrast_id, str) or not self.contrast_id:
            raise ValueError("contrast_id must be nonempty")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("contrast family must be nonempty")
        if self.role not in _ROLES:
            raise ValueError("contrast role is invalid")
        if type(self.coefficients) is not tuple or len(self.coefficients) < 2:
            raise ValueError("contrast coefficients must contain at least two values")
        coefficients = tuple(
            _finite(value, label="contrast coefficient")
            for value in self.coefficients
        )
        if not math.isclose(sum(coefficients), 0.0, abs_tol=1e-12):
            raise ValueError("contrast coefficients must sum to zero")
        if not math.isclose(
            sum(abs(value) for value in coefficients),
            2.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("contrast coefficient L1 norm must equal two")
        object.__setattr__(self, "coefficients", coefficients)
        computed = _digest(self._payload(), domain=_DEFINITION_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("contrast definition hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "family": self.family,
            "role": self.role,
            "coefficients": list(self.coefficients),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(cls, raw: object) -> "ContrastDefinition":
        if not isinstance(raw, Mapping):
            raise TypeError("contrast definition must be a mapping")
        _strict_keys(
            raw,
            expected={
                "contrast_id",
                "family",
                "role",
                "coefficients",
                "artifact_sha256",
            },
            label="contrast definition",
        )
        coefficients = raw["coefficients"]
        if not isinstance(coefficients, list):
            raise TypeError("contrast coefficients must be a list")
        return cls(
            contrast_id=str(raw["contrast_id"]),
            family=str(raw["family"]),
            role=str(raw["role"]),  # type: ignore[arg-type]
            coefficients=tuple(
                _finite(value, label="contrast coefficient")
                for value in coefficients
            ),
            artifact_sha256=_require_sha256(
                raw["artifact_sha256"],
                label="contrast definition hash",
            ),
        )


@dataclass(frozen=True, slots=True)
class ContrastObservation:
    """Immutable teacher and candidate endpoint replays for one contrast."""

    definition: ContrastDefinition
    teacher_endpoints: tuple[Tensor, ...]
    repeated_teacher_endpoints: tuple[Tensor, ...]
    candidate_endpoints: tuple[Tensor, ...]
    repeated_candidate_endpoints: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ContrastDefinition):
            raise TypeError("contrast observation requires a definition")
        endpoint_sets = (
            self.teacher_endpoints,
            self.repeated_teacher_endpoints,
            self.candidate_endpoints,
            self.repeated_candidate_endpoints,
        )
        expected_count = len(self.definition.coefficients)
        if any(
            type(values) is not tuple or len(values) != expected_count
            for values in endpoint_sets
        ):
            raise ValueError("endpoint counts must match contrast coefficients")
        anchor = self.teacher_endpoints[0]
        if not isinstance(anchor, Tensor) or not anchor.is_floating_point():
            raise TypeError("contrast endpoints must be floating tensors")
        shape = anchor.shape
        for values in endpoint_sets:
            for value in values:
                if (
                    not isinstance(value, Tensor)
                    or not value.is_floating_point()
                    or value.shape != shape
                ):
                    raise ValueError(
                        "contrast endpoints must be aligned floating tensors"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("contrast endpoints must be finite")


@dataclass(frozen=True, slots=True)
class ContrastScore:
    """Tensor-free scalar and hash evidence for one frozen contrast."""

    contrast_id: str
    definition_sha256: str
    family: str
    role: ContrastRole
    teacher_endpoint_sha256s: tuple[str, ...]
    repeated_teacher_endpoint_sha256s: tuple[str, ...]
    candidate_endpoint_sha256s: tuple[str, ...]
    repeated_candidate_endpoint_sha256s: tuple[str, ...]
    teacher_baseline_l2: float
    teacher_contrast_l2: float
    teacher_repeat_noise_l2: float
    teacher_numeric_floor_l2: float
    teacher_effective_noise_l2: float
    teacher_relative_effect_lower: float | None
    teacher_relative_effect_upper: float | None
    teacher_status: str
    candidate_scored: bool
    candidate_contrast_l2: float | None
    candidate_repeat_noise_l2: float | None
    candidate_numeric_floor_l2: float | None
    candidate_effective_noise_l2: float | None
    candidate_contrast_relative_error: float | None
    candidate_direction_cosine: float | None
    candidate_projection_gain: float | None
    candidate_orthogonal_leakage: float | None
    candidate_magnitude_ratio: float | None
    candidate_null_relative_effect_upper: float | None
    candidate_null_relative_error_upper: float | None
    candidate_gate_flags: tuple[tuple[str, bool], ...]
    candidate_status: str
    decision_status: DecisionStatus
    reason_codes: tuple[str, ...]
    gates_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.decision_status not in _DECISION_PRIORITY:
            raise ValueError("contrast decision status is invalid")
        for name in (
            "teacher_baseline_l2",
            "teacher_contrast_l2",
            "teacher_repeat_noise_l2",
            "teacher_numeric_floor_l2",
            "teacher_effective_noise_l2",
        ):
            _finite(getattr(self, name), label=name, minimum=0.0)
        for name in (
            "teacher_relative_effect_lower",
            "teacher_relative_effect_upper",
            "candidate_contrast_l2",
            "candidate_repeat_noise_l2",
            "candidate_numeric_floor_l2",
            "candidate_effective_noise_l2",
            "candidate_contrast_relative_error",
            "candidate_direction_cosine",
            "candidate_projection_gain",
            "candidate_orthogonal_leakage",
            "candidate_magnitude_ratio",
            "candidate_null_relative_effect_upper",
            "candidate_null_relative_error_upper",
        ):
            _optional_finite(getattr(self, name), label=name)
        computed = _digest(self._payload(), domain=_SCORE_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("contrast score hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                list(value)
                if isinstance(value, tuple)
                else value
            )
            for name, value in (
                (field_name, getattr(self, field_name))
                for field_name in self.__dataclass_fields__
                if field_name != "artifact_sha256"
            )
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class ContrastFamilyScore:
    """Macro and worst-case decision for one preregistered family."""

    family: str
    role: ContrastRole
    planned_contrast_count: int
    required_eligible_count: int
    eligible_contrast_count: int
    teacher_null_valid_count: int
    panel_inconclusive_count: int
    teacher_null_failure_count: int
    candidate_scored_count: int
    candidate_pass_count: int
    macro_rms_contrast_relative_error: float | None
    worst_contrast_relative_error: float | None
    minimum_direction_cosine: float | None
    minimum_projection_gain: float | None
    maximum_projection_gain: float | None
    maximum_orthogonal_leakage: float | None
    maximum_candidate_null_relative_effect_upper: float | None
    maximum_candidate_null_relative_error_upper: float | None
    decision_status: DecisionStatus
    reason_codes: tuple[str, ...]
    contrast_score_sha256s: tuple[str, ...]
    gates_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.decision_status not in _DECISION_PRIORITY:
            raise ValueError("family decision status is invalid")
        computed = _digest(self._payload(), domain=_FAMILY_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("contrast family score hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                list(value)
                if isinstance(value, tuple)
                else value
            )
            for name, value in (
                (field_name, getattr(self, field_name))
                for field_name in self.__dataclass_fields__
                if field_name != "artifact_sha256"
            )
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class ContrastAssessmentResult:
    """Source-safe family aggregation and formal overall decision."""

    gates_sha256: str
    contrast_scores: tuple[ContrastScore, ...]
    family_scores: tuple[ContrastFamilyScore, ...]
    invalid_family_count: int
    teacher_null_failure_family_count: int
    panel_inconclusive_family_count: int
    candidate_failed_family_count: int
    passed_family_count: int
    overall_status: DecisionStatus
    reason_codes: tuple[str, ...]
    weak_teacher_contrasts_entered_candidate_relative_metrics: bool = False
    intended_null_contrasts_entered_direction_metrics: bool = False
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.overall_status not in _DECISION_PRIORITY:
            raise ValueError("overall contrast-assessment status is invalid")
        if (
            self.weak_teacher_contrasts_entered_candidate_relative_metrics
            or self.intended_null_contrasts_entered_direction_metrics
        ):
            raise ValueError("contrast-assessment metric firewall was violated")
        computed = _digest(self._payload(), domain=_ASSESSMENT_DOMAIN)
        if self.artifact_sha256:
            if self.artifact_sha256 != computed:
                raise ValueError("contrast assessment hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "gates_sha256": self.gates_sha256,
            "contrast_score_sha256s": [
                value.artifact_sha256 for value in self.contrast_scores
            ],
            "family_score_sha256s": [
                value.artifact_sha256 for value in self.family_scores
            ],
            "invalid_family_count": self.invalid_family_count,
            "teacher_null_failure_family_count": (
                self.teacher_null_failure_family_count
            ),
            "panel_inconclusive_family_count": (
                self.panel_inconclusive_family_count
            ),
            "candidate_failed_family_count": self.candidate_failed_family_count,
            "passed_family_count": self.passed_family_count,
            "overall_status": self.overall_status,
            "reason_codes": list(self.reason_codes),
            "weak_teacher_contrasts_entered_candidate_relative_metrics": (
                self.weak_teacher_contrasts_entered_candidate_relative_metrics
            ),
            "intended_null_contrasts_entered_direction_metrics": (
                self.intended_null_contrasts_entered_direction_metrics
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "contrast_scores": [
                value.state_dict() for value in self.contrast_scores
            ],
            "family_scores": [
                value.state_dict() for value in self.family_scores
            ],
            "artifact_sha256": self.artifact_sha256,
        }


def _linear_contrast(
    endpoints: tuple[Tensor, ...],
    coefficients: tuple[float, ...],
) -> Tensor:
    values = tuple(
        endpoint.detach().to(device="cpu", dtype=torch.float64)
        for endpoint in endpoints
    )
    result = torch.zeros_like(values[0])
    for coefficient, value in zip(coefficients, values, strict=True):
        result = result + coefficient * value
    return result


def _endpoint_hashes(values: tuple[Tensor, ...]) -> tuple[str, ...]:
    return tuple(_tensor_sha256(value) for value in values)


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = _tensor_l2(left) * _tensor_l2(right)
    if denominator == 0.0:
        raise ValueError("cosine requires nonzero inputs")
    return float(
        torch.dot(left.flatten(), right.flatten()).item() / denominator
    )


def _invalid_score(
    observation: ContrastObservation,
    gates: ContrastAssessmentGates,
    *,
    baseline: float,
    teacher_contrast: Tensor,
    teacher_repeat_noise: float,
    numeric_floor: float,
    teacher_effective_noise: float,
    reason: str,
) -> ContrastScore:
    definition = observation.definition
    return ContrastScore(
        contrast_id=definition.contrast_id,
        definition_sha256=definition.artifact_sha256,
        family=definition.family,
        role=definition.role,
        teacher_endpoint_sha256s=_endpoint_hashes(
            observation.teacher_endpoints
        ),
        repeated_teacher_endpoint_sha256s=_endpoint_hashes(
            observation.repeated_teacher_endpoints
        ),
        candidate_endpoint_sha256s=_endpoint_hashes(
            observation.candidate_endpoints
        ),
        repeated_candidate_endpoint_sha256s=_endpoint_hashes(
            observation.repeated_candidate_endpoints
        ),
        teacher_baseline_l2=baseline,
        teacher_contrast_l2=_tensor_l2(teacher_contrast),
        teacher_repeat_noise_l2=teacher_repeat_noise,
        teacher_numeric_floor_l2=numeric_floor,
        teacher_effective_noise_l2=teacher_effective_noise,
        teacher_relative_effect_lower=None,
        teacher_relative_effect_upper=None,
        teacher_status="invalid",
        candidate_scored=False,
        candidate_contrast_l2=None,
        candidate_repeat_noise_l2=None,
        candidate_numeric_floor_l2=None,
        candidate_effective_noise_l2=None,
        candidate_contrast_relative_error=None,
        candidate_direction_cosine=None,
        candidate_projection_gain=None,
        candidate_orthogonal_leakage=None,
        candidate_magnitude_ratio=None,
        candidate_null_relative_effect_upper=None,
        candidate_null_relative_error_upper=None,
        candidate_gate_flags=(),
        candidate_status="not_scored",
        decision_status="invalid",
        reason_codes=(reason,),
        gates_sha256=gates.artifact_sha256,
    )


def score_state_conditioned_contrast(
    observation: ContrastObservation,
    *,
    gates: ContrastAssessmentGates | None = None,
) -> ContrastScore:
    """Score one frozen contrast without leaking weak targets into candidate gates."""

    selected_gates = gates or ContrastAssessmentGates()
    definition = observation.definition
    teacher = _linear_contrast(
        observation.teacher_endpoints,
        definition.coefficients,
    )
    teacher_repeat = _linear_contrast(
        observation.repeated_teacher_endpoints,
        definition.coefficients,
    )
    candidate = _linear_contrast(
        observation.candidate_endpoints,
        definition.coefficients,
    )
    candidate_repeat = _linear_contrast(
        observation.repeated_candidate_endpoints,
        definition.coefficients,
    )
    baseline = 0.5 * sum(
        abs(coefficient) * _tensor_l2(endpoint)
        for coefficient, endpoint in zip(
            definition.coefficients,
            observation.teacher_endpoints,
            strict=True,
        )
    )
    teacher_signal = _tensor_l2(teacher)
    candidate_signal = _tensor_l2(candidate)
    teacher_repeat_noise = _tensor_l2(teacher - teacher_repeat)
    candidate_repeat_noise = _tensor_l2(candidate - candidate_repeat)
    dtype = observation.teacher_endpoints[0].dtype
    epsilon = (
        torch.finfo(dtype).eps
        if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
        else torch.finfo(torch.float64).eps
    )
    numeric_floor = (
        selected_gates.epsilon_scale_multiplier * epsilon * baseline
    )
    teacher_noise = max(teacher_repeat_noise, numeric_floor)
    candidate_noise = max(candidate_repeat_noise, numeric_floor)
    if baseline <= max(numeric_floor, torch.finfo(torch.float64).tiny):
        return _invalid_score(
            observation,
            selected_gates,
            baseline=baseline,
            teacher_contrast=teacher,
            teacher_repeat_noise=teacher_repeat_noise,
            numeric_floor=numeric_floor,
            teacher_effective_noise=teacher_noise,
            reason="teacher_baseline_is_numerically_unresolved",
        )

    uncertainty = selected_gates.repeat_noise_multiplier * teacher_noise
    lower = max(teacher_signal - uncertainty, 0.0) / baseline
    upper = (teacher_signal + uncertainty) / baseline
    teacher_status: str
    decision: DecisionStatus
    reasons: list[str] = []
    candidate_scored = False

    if definition.role == "expected_sensitivity":
        if teacher_signal < uncertainty:
            teacher_status = "numerically_unresolved_sensitivity"
            decision = "panel_inconclusive"
            reasons.append("teacher_sensitivity_not_numerically_resolved")
        elif lower >= selected_gates.minimum_sensitivity_relative_effect:
            teacher_status = "eligible_sensitivity"
            decision = "pass"
            candidate_scored = True
        elif upper < selected_gates.minimum_sensitivity_relative_effect:
            teacher_status = "underpowered_sensitivity"
            decision = "panel_inconclusive"
            reasons.append("teacher_sensitivity_below_effect_floor")
        else:
            teacher_status = "boundary_inconclusive_sensitivity"
            decision = "panel_inconclusive"
            reasons.append("teacher_sensitivity_interval_crosses_effect_floor")
    else:
        if upper <= selected_gates.maximum_teacher_null_relative_effect:
            teacher_status = "valid_intended_null"
            decision = "pass"
            candidate_scored = True
        elif lower > selected_gates.maximum_teacher_null_relative_effect:
            teacher_status = "violated_intended_null"
            decision = "teacher_null_failure"
            reasons.append("teacher_null_effect_exceeds_ceiling")
        else:
            teacher_status = "boundary_inconclusive_null"
            decision = "panel_inconclusive"
            reasons.append("teacher_null_interval_crosses_ceiling")

    candidate_relative_error: float | None = None
    candidate_cosine: float | None = None
    candidate_gain: float | None = None
    candidate_orthogonal: float | None = None
    candidate_magnitude_ratio: float | None = None
    candidate_null_effect: float | None = None
    candidate_null_error: float | None = None
    candidate_flags: tuple[tuple[str, bool], ...] = ()
    candidate_status = "not_scored"

    if candidate_scored and definition.role == "expected_sensitivity":
        candidate_status = "pass"
        error = candidate - teacher
        teacher_energy = float(torch.dot(teacher.flatten(), teacher.flatten()))
        candidate_relative_error = _tensor_l2(error) / teacher_signal
        candidate_gain = float(
            torch.dot(candidate.flatten(), teacher.flatten()).item()
            / teacher_energy
        )
        candidate_orthogonal = (
            _tensor_l2(candidate - candidate_gain * teacher) / teacher_signal
        )
        candidate_magnitude_ratio = candidate_signal / teacher_signal
        candidate_resolved = (
            candidate_signal
            > selected_gates.repeat_noise_multiplier * candidate_noise
        )
        candidate_cosine = (
            _cosine(candidate, teacher) if candidate_resolved else None
        )
        flags = {
            "contrast_relative_error": (
                candidate_relative_error
                <= selected_gates.maximum_sensitivity_contrast_relative_error
            ),
            "direction_cosine": (
                candidate_cosine is not None
                and candidate_cosine
                >= selected_gates.minimum_sensitivity_direction_cosine
            ),
            "projection_gain": (
                selected_gates.minimum_sensitivity_projection_gain
                <= candidate_gain
                <= selected_gates.maximum_sensitivity_projection_gain
            ),
            "orthogonal_leakage": (
                candidate_orthogonal
                <= selected_gates.maximum_sensitivity_orthogonal_leakage
            ),
        }
        candidate_flags = tuple(sorted(flags.items()))
        if not candidate_resolved:
            reasons.append("candidate_sensitivity_not_numerically_resolved")
        if not all(flags.values()):
            candidate_status = "fail"
            decision = "candidate_fail"
            reasons.extend(
                f"candidate_failed_{name}"
                for name, passed in candidate_flags
                if not passed
            )
    elif candidate_scored:
        candidate_status = "pass"
        candidate_null_effect = (
            candidate_signal
            + selected_gates.repeat_noise_multiplier * candidate_noise
        ) / baseline
        candidate_null_error = (
            _tensor_l2(candidate - teacher)
            + selected_gates.repeat_noise_multiplier
            * (candidate_noise + teacher_noise)
        ) / baseline
        flags = {
            "null_relative_effect": (
                candidate_null_effect
                <= selected_gates.maximum_candidate_null_relative_effect
            ),
            "null_relative_error": (
                candidate_null_error
                <= selected_gates.maximum_candidate_null_relative_error
            ),
        }
        candidate_flags = tuple(sorted(flags.items()))
        if not all(flags.values()):
            candidate_status = "fail"
            decision = "candidate_fail"
            reasons.extend(
                f"candidate_failed_{name}"
                for name, passed in candidate_flags
                if not passed
            )

    return ContrastScore(
        contrast_id=definition.contrast_id,
        definition_sha256=definition.artifact_sha256,
        family=definition.family,
        role=definition.role,
        teacher_endpoint_sha256s=_endpoint_hashes(
            observation.teacher_endpoints
        ),
        repeated_teacher_endpoint_sha256s=_endpoint_hashes(
            observation.repeated_teacher_endpoints
        ),
        candidate_endpoint_sha256s=_endpoint_hashes(
            observation.candidate_endpoints
        ),
        repeated_candidate_endpoint_sha256s=_endpoint_hashes(
            observation.repeated_candidate_endpoints
        ),
        teacher_baseline_l2=baseline,
        teacher_contrast_l2=teacher_signal,
        teacher_repeat_noise_l2=teacher_repeat_noise,
        teacher_numeric_floor_l2=numeric_floor,
        teacher_effective_noise_l2=teacher_noise,
        teacher_relative_effect_lower=lower,
        teacher_relative_effect_upper=upper,
        teacher_status=teacher_status,
        candidate_scored=candidate_scored,
        candidate_contrast_l2=(
            candidate_signal if candidate_scored else None
        ),
        candidate_repeat_noise_l2=(
            candidate_repeat_noise if candidate_scored else None
        ),
        candidate_numeric_floor_l2=(
            numeric_floor if candidate_scored else None
        ),
        candidate_effective_noise_l2=(
            candidate_noise if candidate_scored else None
        ),
        candidate_contrast_relative_error=candidate_relative_error,
        candidate_direction_cosine=candidate_cosine,
        candidate_projection_gain=candidate_gain,
        candidate_orthogonal_leakage=candidate_orthogonal,
        candidate_magnitude_ratio=candidate_magnitude_ratio,
        candidate_null_relative_effect_upper=candidate_null_effect,
        candidate_null_relative_error_upper=candidate_null_error,
        candidate_gate_flags=candidate_flags,
        candidate_status=candidate_status,
        decision_status=decision,
        reason_codes=tuple(sorted(set(reasons))),
        gates_sha256=selected_gates.artifact_sha256,
    )


def _family_score(
    family: str,
    scores: tuple[ContrastScore, ...],
    *,
    gates: ContrastAssessmentGates,
) -> ContrastFamilyScore:
    roles = {score.role for score in scores}
    if len(roles) != 1:
        raise ValueError("one contrast family cannot mix semantic roles")
    role = next(iter(roles))
    planned = len(scores)
    required = (
        max(
            gates.minimum_family_eligible_count,
            math.ceil(gates.minimum_family_eligible_fraction * planned),
        )
        if role == "expected_sensitivity"
        else planned
    )
    eligible = sum(
        score.teacher_status == "eligible_sensitivity" for score in scores
    )
    null_valid = sum(
        score.teacher_status == "valid_intended_null" for score in scores
    )
    inconclusive = sum(
        score.decision_status == "panel_inconclusive" for score in scores
    )
    null_failures = sum(
        score.decision_status == "teacher_null_failure" for score in scores
    )
    candidate_scores = tuple(score for score in scores if score.candidate_scored)
    candidate_passes = sum(
        score.candidate_status == "pass" for score in candidate_scores
    )
    sensitivity_errors = tuple(
        score.candidate_contrast_relative_error
        for score in candidate_scores
        if score.role == "expected_sensitivity"
        and score.candidate_contrast_relative_error is not None
    )
    macro_error = (
        math.sqrt(
            sum(value * value for value in sensitivity_errors)
            / len(sensitivity_errors)
        )
        if sensitivity_errors
        else None
    )

    decision: DecisionStatus
    reasons: list[str] = []
    if any(score.decision_status == "invalid" for score in scores):
        decision = "invalid"
        reasons.append("family_contains_invalid_contrast")
    elif null_failures:
        decision = "teacher_null_failure"
        reasons.append("family_teacher_null_control_failed")
    elif role == "expected_sensitivity" and eligible < required:
        decision = "panel_inconclusive"
        reasons.append("family_has_insufficient_eligible_sensitivities")
    elif role == "intended_null" and null_valid != planned:
        decision = "panel_inconclusive"
        reasons.append("family_contains_inconclusive_null")
    elif any(
        score.decision_status == "candidate_fail" for score in candidate_scores
    ):
        decision = "candidate_fail"
        reasons.append("family_contains_candidate_contrast_failure")
    elif (
        macro_error is not None
        and macro_error > gates.maximum_family_macro_rms_relative_error
    ):
        decision = "candidate_fail"
        reasons.append("family_macro_rms_error_exceeds_gate")
    else:
        decision = "pass"

    sensitivity_cosines = tuple(
        score.candidate_direction_cosine
        for score in candidate_scores
        if score.candidate_direction_cosine is not None
    )
    sensitivity_gains = tuple(
        score.candidate_projection_gain
        for score in candidate_scores
        if score.candidate_projection_gain is not None
    )
    sensitivity_orthogonal = tuple(
        score.candidate_orthogonal_leakage
        for score in candidate_scores
        if score.candidate_orthogonal_leakage is not None
    )
    null_effects = tuple(
        score.candidate_null_relative_effect_upper
        for score in candidate_scores
        if score.candidate_null_relative_effect_upper is not None
    )
    null_errors = tuple(
        score.candidate_null_relative_error_upper
        for score in candidate_scores
        if score.candidate_null_relative_error_upper is not None
    )
    return ContrastFamilyScore(
        family=family,
        role=role,  # type: ignore[arg-type]
        planned_contrast_count=planned,
        required_eligible_count=required,
        eligible_contrast_count=eligible,
        teacher_null_valid_count=null_valid,
        panel_inconclusive_count=inconclusive,
        teacher_null_failure_count=null_failures,
        candidate_scored_count=len(candidate_scores),
        candidate_pass_count=candidate_passes,
        macro_rms_contrast_relative_error=macro_error,
        worst_contrast_relative_error=(
            max(sensitivity_errors) if sensitivity_errors else None
        ),
        minimum_direction_cosine=(
            min(sensitivity_cosines) if sensitivity_cosines else None
        ),
        minimum_projection_gain=(
            min(sensitivity_gains) if sensitivity_gains else None
        ),
        maximum_projection_gain=(
            max(sensitivity_gains) if sensitivity_gains else None
        ),
        maximum_orthogonal_leakage=(
            max(sensitivity_orthogonal)
            if sensitivity_orthogonal
            else None
        ),
        maximum_candidate_null_relative_effect_upper=(
            max(null_effects) if null_effects else None
        ),
        maximum_candidate_null_relative_error_upper=(
            max(null_errors) if null_errors else None
        ),
        decision_status=decision,
        reason_codes=tuple(reasons),
        contrast_score_sha256s=tuple(
            score.artifact_sha256 for score in scores
        ),
        gates_sha256=gates.artifact_sha256,
    )


def assess_state_conditioned_contrasts(
    observations: Sequence[ContrastObservation],
    *,
    gates: ContrastAssessmentGates | None = None,
) -> ContrastAssessmentResult:
    """Score and aggregate a complete generic contrast panel."""

    selected_gates = gates or ContrastAssessmentGates()
    if not observations:
        raise ValueError("contrast assessment requires at least one observation")
    definitions = [observation.definition for observation in observations]
    ids = [definition.contrast_id for definition in definitions]
    if len(set(ids)) != len(ids):
        raise ValueError("contrast assessment contains duplicate contrast ids")
    scores = tuple(
        sorted(
            (
                score_state_conditioned_contrast(
                    observation,
                    gates=selected_gates,
                )
                for observation in observations
            ),
            key=lambda value: value.contrast_id,
        )
    )
    by_family: dict[str, list[ContrastScore]] = defaultdict(list)
    for score in scores:
        by_family[score.family].append(score)
    families = tuple(
        _family_score(
            family,
            tuple(sorted(values, key=lambda value: value.contrast_id)),
            gates=selected_gates,
        )
        for family, values in sorted(by_family.items())
    )
    counts = {
        status: sum(family.decision_status == status for family in families)
        for status in _DECISION_PRIORITY
    }
    overall = max(
        (family.decision_status for family in families),
        key=lambda status: _DECISION_PRIORITY[status],
    )
    reasons = tuple(
        sorted(
            {
                f"{family.family}:{reason}"
                for family in families
                for reason in family.reason_codes
            }
        )
    )
    return ContrastAssessmentResult(
        gates_sha256=selected_gates.artifact_sha256,
        contrast_scores=scores,
        family_scores=families,
        invalid_family_count=counts["invalid"],
        teacher_null_failure_family_count=counts["teacher_null_failure"],
        panel_inconclusive_family_count=counts["panel_inconclusive"],
        candidate_failed_family_count=counts["candidate_fail"],
        passed_family_count=counts["pass"],
        overall_status=overall,  # type: ignore[arg-type]
        reason_codes=reasons,
    )
