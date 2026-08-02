"""Pure symmetric microstep validation for candidate-conditioned K64 gains.

The V4 mean-KL proposal supplies one fixed direction.  This module evaluates
that direction at exactly ``+1/64`` and ``-1/64`` around the authenticated V4
unit point.  Only the positive microstep can be selected; the negative arm is
retained solely to estimate a central slope and local curve shape.

Raw token losses and gain vectors stay in typed in-memory records.  Serialized
metadata contains only hashes, scalar summaries, and protocol identities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    CANDIDATE_GAIN_RANK,
    CandidateConditionedK64MeanKLRefit,
)


__all__ = [
    "SYMMETRIC_GAIN_MICROSTEP_EPSILON",
    "CandidateConditionedK64SymmetricMicrostepExample",
    "CandidateConditionedK64SymmetricMicrostepSelection",
    "symmetric_microstep_gains",
    "select_candidate_conditioned_k64_symmetric_microstep",
]


SYMMETRIC_GAIN_MICROSTEP_EPSILON = 1.0 / 64.0
_EXPECTED_TRAINING_FAMILIES = 7
_GAIN_MINIMUM = 0.0
_GAIN_MAXIMUM = 1.5
_ABSOLUTE_KL_TOLERANCE = 1.0e-8
_RELATIVE_IMPROVEMENT_FRACTION = 1.0e-4
_WORST_FAMILY_RATIO = 1.05
_EXAMPLE_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-symmetric-microstep-example:v5\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-symmetric-microstep-selection:v5\0"
)
_TENSOR_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-symmetric-microstep-tensor:v5\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _float64(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or 0 in value.shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating tensor")
    return (
        value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    )


def _tensor_sha256(value: Tensor) -> str:
    tensor = _float64(value, label="hashed tensor", ndim=value.ndim)
    payload = tensor.numpy().astype("<f8", copy=False).tobytes(order="C")
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + _canonical_json_bytes(
            {"dtype": "float64-little-endian", "shape": tuple(tensor.shape)}
        )
        + payload
    ).hexdigest()


def _finite_nonnegative_scalar(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative scalar")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a finite nonnegative scalar"
        ) from error
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative scalar")
    return converted


def _finite_report_ratio(candidate: float, source: float) -> float:
    denominator = max(source, 1.0e-12)
    maximum = torch.finfo(torch.float64).max
    if candidate > maximum * denominator:
        return maximum
    return candidate / denominator


def symmetric_microstep_gains(
    refit: CandidateConditionedK64MeanKLRefit, sign: int
) -> Tensor:
    """Return the fixed positive or negative ``1/64`` gain microstep."""

    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("refit must be a candidate v4 mean-KL refit")
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError("symmetric microstep sign must be exactly -1 or +1")
    refit.validate_integrity()
    proposal = refit.mean_proposed_gains_tensor()
    gains = (
        1.0
        + float(sign)
        * SYMMETRIC_GAIN_MICROSTEP_EPSILON
        * (proposal - 1.0)
    ).contiguous()
    if (
        gains.shape != (CANDIDATE_GAIN_RANK,)
        or not bool(torch.isfinite(gains).all())
        or bool((gains < _GAIN_MINIMUM).any())
        or bool((gains > _GAIN_MAXIMUM).any())
    ):
        raise RuntimeError("symmetric microstep gains left the provider bounds")
    return gains.clone().contiguous()


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64SymmetricMicrostepExample:
    """One tune-family unit binding and its newly executed +/- observations."""

    held_family_id: str
    example_id: str
    family_id: str
    v4_refit_artifact_sha256: str
    pinned_v4_tune_example_artifact_sha256: str
    pinned_v4_unit_mean_teacher_kl: float
    pinned_v4_unit_token_teacher_kl_sha256: str
    pinned_v4_unit_receipt_sha256: str
    structural_no_op_replayed_pinned_v4_unit_exactly: bool | None
    plus_token_teacher_kl: Tensor = field(repr=False)
    minus_token_teacher_kl: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        held = _identifier(self.held_family_id, label="held_family_id")
        example = _identifier(self.example_id, label="microstep example_id")
        family = _identifier(self.family_id, label="microstep family_id")
        refit_sha = _require_sha256(
            self.v4_refit_artifact_sha256, label="candidate v4 refit"
        )
        tune_sha = _require_sha256(
            self.pinned_v4_tune_example_artifact_sha256,
            label="pinned v4 tune example",
        )
        unit_hash = _require_sha256(
            self.pinned_v4_unit_token_teacher_kl_sha256,
            label="pinned v4 unit token teacher KL",
        )
        unit_receipt = _require_sha256(
            self.pinned_v4_unit_receipt_sha256,
            label="pinned v4 unit receipt",
        )
        unit_mean = _finite_nonnegative_scalar(
            self.pinned_v4_unit_mean_teacher_kl,
            label="pinned v4 unit mean teacher KL",
        )
        plus = _float64(
            self.plus_token_teacher_kl,
            label="plus-epsilon token teacher KL",
            ndim=1,
        )
        minus = _float64(
            self.minus_token_teacher_kl,
            label="minus-epsilon token teacher KL",
            ndim=1,
        )
        replay = self.structural_no_op_replayed_pinned_v4_unit_exactly
        if (
            held == family
            or plus.shape != minus.shape
            or bool((plus < 0.0).any())
            or bool((minus < 0.0).any())
            or (replay is not None and replay is not True)
        ):
            raise ValueError("symmetric microstep example geometry differs")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "example_id", example)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "v4_refit_artifact_sha256", refit_sha)
        object.__setattr__(
            self, "pinned_v4_tune_example_artifact_sha256", tune_sha
        )
        object.__setattr__(
            self, "pinned_v4_unit_token_teacher_kl_sha256", unit_hash
        )
        object.__setattr__(
            self, "pinned_v4_unit_receipt_sha256", unit_receipt
        )
        object.__setattr__(
            self, "pinned_v4_unit_mean_teacher_kl", unit_mean
        )
        object.__setattr__(self, "plus_token_teacher_kl", plus)
        object.__setattr__(self, "minus_token_teacher_kl", minus)
        object.__setattr__(
            self,
            "structural_no_op_replayed_pinned_v4_unit_exactly",
            replay,
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return int(self.plus_token_teacher_kl.numel())

    def plus_token_kl_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.plus_token_teacher_kl.clone().contiguous()

    def minus_token_kl_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.minus_token_teacher_kl.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        plus_mean = float(self.plus_token_teacher_kl.mean())
        minus_mean = float(self.minus_token_teacher_kl.mean())
        central_slope = (plus_mean - minus_mean) / (
            2.0 * SYMMETRIC_GAIN_MICROSTEP_EPSILON
        )
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "v4_refit_artifact_sha256": self.v4_refit_artifact_sha256,
            "pinned_v4_tune_example_artifact_sha256": (
                self.pinned_v4_tune_example_artifact_sha256
            ),
            "pinned_v4_unit_mean_teacher_kl": (
                self.pinned_v4_unit_mean_teacher_kl
            ),
            "pinned_v4_unit_mean_teacher_kl_hex": (
                self.pinned_v4_unit_mean_teacher_kl.hex()
            ),
            "pinned_v4_unit_token_teacher_kl_sha256": (
                self.pinned_v4_unit_token_teacher_kl_sha256
            ),
            "pinned_v4_unit_receipt_sha256": (
                self.pinned_v4_unit_receipt_sha256
            ),
            "structural_no_op_replayed_pinned_v4_unit_exactly": (
                self.structural_no_op_replayed_pinned_v4_unit_exactly
            ),
            "supervised_token_count": self.supervised_tokens,
            "microstep_epsilon_hex": SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
            "plus_token_teacher_kl_sha256": _tensor_sha256(
                self.plus_token_teacher_kl
            ),
            "minus_token_teacher_kl_sha256": _tensor_sha256(
                self.minus_token_teacher_kl
            ),
            "plus_mean_teacher_kl": plus_mean,
            "minus_mean_teacher_kl": minus_mean,
            "family_central_slope": central_slope,
            "minus_arm_role": "diagnostic_only",
            "minus_arm_participates_in_central_slope_sign_estimate": True,
            "minus_arm_is_selectable": False,
            "minus_arm_independently_authorizes_selection": False,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.plus_token_teacher_kl.dtype != torch.float64
            or self.plus_token_teacher_kl.device.type != "cpu"
            or self.plus_token_teacher_kl.requires_grad
            or not self.plus_token_teacher_kl.is_contiguous()
            or self.minus_token_teacher_kl.dtype != torch.float64
            or self.minus_token_teacher_kl.device.type != "cpu"
            or self.minus_token_teacher_kl.requires_grad
            or not self.minus_token_teacher_kl.is_contiguous()
            or _sha256(_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256, label="symmetric microstep example"
            )
        ):
            raise RuntimeError("symmetric microstep example payload drifted")


def _scalar_summary(values: Tensor, *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_minimum": float(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_maximum": float(values.max()),
    }


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64SymmetricMicrostepSelection:
    """Authenticated fixed-microstep tune ledger and its safe selection."""

    held_family_id: str
    refit_artifact_sha256: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    tune_example_ids: tuple[str, ...]
    tune_example_artifact_sha256s: tuple[str, ...]
    mean_proposed_gains: Tensor = field(repr=False)
    unit_family_mean_teacher_kl: Tensor = field(repr=False)
    plus_family_mean_teacher_kl: Tensor = field(repr=False)
    minus_family_mean_teacher_kl: Tensor = field(repr=False)
    mean_fit_predicted_derivative: float
    mean_refit_was_no_op: bool
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        held = _identifier(self.held_family_id, label="held_family_id")
        refit_sha = _require_sha256(
            self.refit_artifact_sha256, label="candidate v4 refit"
        )
        families = tuple(
            _identifier(value, label="training family_id")
            for value in self.training_family_ids
        )
        training_ids = tuple(
            _identifier(value, label="training example_id")
            for value in self.training_example_ids
        )
        tune_ids = tuple(
            _identifier(value, label="tune example_id")
            for value in self.tune_example_ids
        )
        evidence = tuple(
            _require_sha256(value, label="microstep example artifact")
            for value in self.tune_example_artifact_sha256s
        )
        proposal = _float64(
            self.mean_proposed_gains, label="mean proposed gains", ndim=1
        )
        unit = _float64(
            self.unit_family_mean_teacher_kl,
            label="unit family mean teacher KL",
            ndim=1,
        )
        plus = _float64(
            self.plus_family_mean_teacher_kl,
            label="plus family mean teacher KL",
            ndim=1,
        )
        minus = _float64(
            self.minus_family_mean_teacher_kl,
            label="minus family mean teacher KL",
            ndim=1,
        )
        fit_derivative = float(self.mean_fit_predicted_derivative)
        if (
            len(families) != _EXPECTED_TRAINING_FAMILIES
            or families != tuple(sorted(set(families)))
            or held in families
            or len(training_ids) != _EXPECTED_TRAINING_FAMILIES
            or training_ids != tuple(sorted(set(training_ids)))
            or len(tune_ids) != _EXPECTED_TRAINING_FAMILIES
            or len(set(tune_ids)) != len(tune_ids)
            or not set(tune_ids).isdisjoint(training_ids)
            or len(evidence) != len(tune_ids)
            or proposal.shape != (CANDIDATE_GAIN_RANK,)
            or unit.shape != (_EXPECTED_TRAINING_FAMILIES,)
            or plus.shape != unit.shape
            or minus.shape != unit.shape
            or bool((unit < 0.0).any())
            or bool((plus < 0.0).any())
            or bool((minus < 0.0).any())
            or not math.isfinite(fit_derivative)
            or type(self.mean_refit_was_no_op) is not bool
            or (self.mean_refit_was_no_op and fit_derivative != 0.0)
            or self.mean_refit_was_no_op is not (fit_derivative >= 0.0)
            or (
                self.mean_refit_was_no_op
                and not torch.equal(proposal, torch.ones_like(proposal))
            )
        ):
            raise ValueError("symmetric microstep selection payload is invalid")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "refit_artifact_sha256", refit_sha)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", training_ids)
        object.__setattr__(self, "tune_example_ids", tune_ids)
        object.__setattr__(self, "tune_example_artifact_sha256s", evidence)
        object.__setattr__(self, "mean_proposed_gains", proposal)
        object.__setattr__(self, "unit_family_mean_teacher_kl", unit)
        object.__setattr__(self, "plus_family_mean_teacher_kl", plus)
        object.__setattr__(self, "minus_family_mean_teacher_kl", minus)
        object.__setattr__(
            self, "mean_fit_predicted_derivative", fit_derivative
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SELECTION_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def aggregate_unit_mean_teacher_kl(self) -> float:
        return math.fsum(
            float(value) for value in self.unit_family_mean_teacher_kl
        ) / _EXPECTED_TRAINING_FAMILIES

    @property
    def aggregate_plus_mean_teacher_kl(self) -> float:
        return math.fsum(
            float(value) for value in self.plus_family_mean_teacher_kl
        ) / _EXPECTED_TRAINING_FAMILIES

    @property
    def aggregate_minus_mean_teacher_kl(self) -> float:
        return math.fsum(
            float(value) for value in self.minus_family_mean_teacher_kl
        ) / _EXPECTED_TRAINING_FAMILIES

    @property
    def minimum_required_plus_improvement(self) -> float:
        return max(
            _ABSOLUTE_KL_TOLERANCE,
            _RELATIVE_IMPROVEMENT_FRACTION
            * self.aggregate_unit_mean_teacher_kl,
        )

    @property
    def plus_improvement(self) -> float:
        return (
            self.aggregate_unit_mean_teacher_kl
            - self.aggregate_plus_mean_teacher_kl
        )

    @property
    def central_slope(self) -> float:
        if self.refit_no_op_or_zero_delta:
            return 0.0
        return (
            self.aggregate_plus_mean_teacher_kl
            - self.aggregate_minus_mean_teacher_kl
        ) / (2.0 * SYMMETRIC_GAIN_MICROSTEP_EPSILON)

    @property
    def forward_slope(self) -> float:
        if self.refit_no_op_or_zero_delta:
            return 0.0
        return (
            self.aggregate_plus_mean_teacher_kl
            - self.aggregate_unit_mean_teacher_kl
        ) / SYMMETRIC_GAIN_MICROSTEP_EPSILON

    @property
    def backward_slope(self) -> float:
        if self.refit_no_op_or_zero_delta:
            return 0.0
        return (
            self.aggregate_unit_mean_teacher_kl
            - self.aggregate_minus_mean_teacher_kl
        ) / SYMMETRIC_GAIN_MICROSTEP_EPSILON

    @property
    def central_curvature(self) -> float:
        if self.refit_no_op_or_zero_delta:
            return 0.0
        epsilon = SYMMETRIC_GAIN_MICROSTEP_EPSILON
        return (
            self.aggregate_plus_mean_teacher_kl
            - 2.0 * self.aggregate_unit_mean_teacher_kl
            + self.aggregate_minus_mean_teacher_kl
        ) / (epsilon * epsilon)

    @property
    def plus_family_nonworse_count(self) -> int:
        return sum(
            float(candidate) <= float(source) + _ABSOLUTE_KL_TOLERANCE
            for candidate, source in zip(
                self.plus_family_mean_teacher_kl,
                self.unit_family_mean_teacher_kl,
            )
        )

    @property
    def plus_worst_family_ratio(self) -> float:
        return max(
            _finite_report_ratio(float(candidate), float(source))
            for candidate, source in zip(
                self.plus_family_mean_teacher_kl,
                self.unit_family_mean_teacher_kl,
            )
        )

    @property
    def plus_family_cap_passed(self) -> bool:
        return all(
            float(candidate)
            <= _WORST_FAMILY_RATIO * float(source) + _ABSOLUTE_KL_TOLERANCE
            for candidate, source in zip(
                self.plus_family_mean_teacher_kl,
                self.unit_family_mean_teacher_kl,
            )
        )

    @property
    def refit_no_op_or_zero_delta(self) -> bool:
        return self.mean_refit_was_no_op or bool(
            torch.equal(
                self.mean_proposed_gains,
                torch.ones_like(self.mean_proposed_gains),
            )
        )

    @property
    def plus_is_eligible(self) -> bool:
        return (
            not self.refit_no_op_or_zero_delta
            and self.central_slope < 0.0
            and self.plus_improvement >= self.minimum_required_plus_improvement
            and self.plus_family_nonworse_count >= 4
            and self.plus_family_cap_passed
        )

    @property
    def selected_step(self) -> float:
        return SYMMETRIC_GAIN_MICROSTEP_EPSILON if self.plus_is_eligible else 0.0

    @property
    def selected_arm(self) -> str:
        return "plus_epsilon" if self.plus_is_eligible else "unit"

    @property
    def selection_reason(self) -> str:
        if self.refit_no_op_or_zero_delta:
            return "unit_mean_refit_no_op_or_zero_delta"
        if self.central_slope >= 0.0:
            return "unit_nonnegative_family_equal_central_slope"
        if self.plus_improvement < self.minimum_required_plus_improvement:
            return "unit_plus_macro_improvement_below_threshold"
        if self.plus_family_nonworse_count < 4:
            return "unit_fewer_than_four_of_seven_plus_families_nonworse"
        if not self.plus_family_cap_passed:
            return "unit_plus_family_cap_failed"
        return "plus_epsilon_cleared_symmetric_microstep_guards"

    def plus_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return (
            1.0
            + SYMMETRIC_GAIN_MICROSTEP_EPSILON
            * (self.mean_proposed_gains - 1.0)
        ).clone().contiguous()

    def minus_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return (
            1.0
            - SYMMETRIC_GAIN_MICROSTEP_EPSILON
            * (self.mean_proposed_gains - 1.0)
        ).clone().contiguous()

    def selected_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        if self.plus_is_eligible:
            return self.plus_gains_tensor()
        return torch.ones_like(self.mean_proposed_gains).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        plus_gains = (
            1.0
            + SYMMETRIC_GAIN_MICROSTEP_EPSILON
            * (self.mean_proposed_gains - 1.0)
        ).contiguous()
        minus_gains = (
            1.0
            - SYMMETRIC_GAIN_MICROSTEP_EPSILON
            * (self.mean_proposed_gains - 1.0)
        ).contiguous()
        selected_gains = (
            plus_gains
            if self.plus_is_eligible
            else torch.ones_like(self.mean_proposed_gains)
        )
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "tune_example_ids": self.tune_example_ids,
            "tune_example_artifact_sha256s": (
                self.tune_example_artifact_sha256s
            ),
            "microstep_epsilon": SYMMETRIC_GAIN_MICROSTEP_EPSILON,
            "microstep_epsilon_hex": SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
            "v5_executed_tune_steps_hex": (
                (-SYMMETRIC_GAIN_MICROSTEP_EPSILON).hex(),
                SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
            ),
            "selection_candidate_steps_hex": (
                0.0.hex(),
                SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
            ),
            "posthoc_step_grid_searched": False,
            "zero_step_source": "pinned_v4_not_reexecuted",
            "mean_proposed_gains_sha256": _tensor_sha256(
                self.mean_proposed_gains
            ),
            "plus_gains_sha256": _tensor_sha256(plus_gains),
            "minus_gains_sha256": _tensor_sha256(minus_gains),
            "unit_family_mean_teacher_kl_sha256": _tensor_sha256(
                self.unit_family_mean_teacher_kl
            ),
            "plus_family_mean_teacher_kl_sha256": _tensor_sha256(
                self.plus_family_mean_teacher_kl
            ),
            "minus_family_mean_teacher_kl_sha256": _tensor_sha256(
                self.minus_family_mean_teacher_kl
            ),
            "aggregate_unit_mean_teacher_kl": (
                self.aggregate_unit_mean_teacher_kl
            ),
            "aggregate_plus_mean_teacher_kl": (
                self.aggregate_plus_mean_teacher_kl
            ),
            "aggregate_minus_mean_teacher_kl": (
                self.aggregate_minus_mean_teacher_kl
            ),
            "plus_improvement": self.plus_improvement,
            "minimum_required_plus_improvement": (
                self.minimum_required_plus_improvement
            ),
            "central_slope": self.central_slope,
            "forward_slope": self.forward_slope,
            "backward_slope": self.backward_slope,
            "central_curvature": self.central_curvature,
            "curve_diagnostics_defined": (
                not self.refit_no_op_or_zero_delta
            ),
            "central_slope_is_negative": self.central_slope < 0.0,
            "forward_slope_is_negative": self.forward_slope < 0.0,
            "backward_slope_is_negative": self.backward_slope < 0.0,
            "forward_and_backward_slope_signs_agree": (
                (self.forward_slope < 0.0) == (self.backward_slope < 0.0)
            ),
            "plus_family_nonworse_count": self.plus_family_nonworse_count,
            "minimum_plus_family_nonworse_count": 4,
            "plus_worst_family_ratio": self.plus_worst_family_ratio,
            "plus_family_ratio_cap": _WORST_FAMILY_RATIO,
            "plus_family_cap_passed": self.plus_family_cap_passed,
            "v4_mean_fit_predicted_derivative": (
                self.mean_fit_predicted_derivative
            ),
            "v4_predicted_slope_is_negative": (
                self.mean_fit_predicted_derivative < 0.0
            ),
            "central_slope_sign_agrees_with_v4_prediction": (
                not self.refit_no_op_or_zero_delta
                and (self.central_slope < 0.0)
                == (self.mean_fit_predicted_derivative < 0.0)
            ),
            "mean_refit_was_no_op": self.mean_refit_was_no_op,
            "refit_no_op_or_zero_delta": self.refit_no_op_or_zero_delta,
            "plus_is_eligible": self.plus_is_eligible,
            "selected_step_hex": self.selected_step.hex(),
            "selected_arm": self.selected_arm,
            "selected_gains_sha256": _tensor_sha256(selected_gains),
            "selection_reason": self.selection_reason,
            "minus_arm_diagnostic_only": True,
            "minus_arm_participates_in_central_slope_sign_guard": True,
            "minus_arm_is_selectable": False,
            "minus_arm_independently_authorizes_selection": False,
            "family_aggregation": "equal_over_exactly_seven_tune_families",
            "held_family_used_for_tune": False,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        result.update(
            _scalar_summary(
                self.unit_family_mean_teacher_kl,
                prefix="unit_family_mean_teacher_kl",
            )
        )
        result.update(
            _scalar_summary(
                self.plus_family_mean_teacher_kl,
                prefix="plus_family_mean_teacher_kl",
            )
        )
        result.update(
            _scalar_summary(
                self.minus_family_mean_teacher_kl,
                prefix="minus_family_mean_teacher_kl",
            )
        )
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        plus_gains = (
            1.0
            + SYMMETRIC_GAIN_MICROSTEP_EPSILON
            * (self.mean_proposed_gains - 1.0)
        )
        minus_gains = (
            1.0
            - SYMMETRIC_GAIN_MICROSTEP_EPSILON
            * (self.mean_proposed_gains - 1.0)
        )
        tensors = (
            self.mean_proposed_gains,
            self.unit_family_mean_teacher_kl,
            self.plus_family_mean_teacher_kl,
            self.minus_family_mean_teacher_kl,
        )
        if (
            any(
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or value.requires_grad
                or not value.is_contiguous()
                for value in tensors
            )
            or bool((plus_gains < _GAIN_MINIMUM).any())
            or bool((plus_gains > _GAIN_MAXIMUM).any())
            or bool((minus_gains < _GAIN_MINIMUM).any())
            or bool((minus_gains > _GAIN_MAXIMUM).any())
            or (
                self.refit_no_op_or_zero_delta
                and self.plus_is_eligible
            )
            or _sha256(_SELECTION_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256, label="symmetric microstep selection"
            )
        ):
            raise RuntimeError("symmetric microstep selection payload drifted")


def select_candidate_conditioned_k64_symmetric_microstep(
    refit: CandidateConditionedK64MeanKLRefit,
    tune_examples: Sequence[CandidateConditionedK64SymmetricMicrostepExample],
) -> CandidateConditionedK64SymmetricMicrostepSelection:
    """Choose exactly the positive 1/64 microstep or authenticated V4 unit."""

    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("refit must be a candidate v4 mean-KL refit")
    refit.validate_integrity()
    supplied = tuple(tune_examples)
    if any(
        not isinstance(value, CandidateConditionedK64SymmetricMicrostepExample)
        for value in supplied
    ):
        raise TypeError("microstep tune examples must be typed records")
    values = tuple(
        sorted(
            supplied,
            key=lambda value: (value.family_id, value.example_id),
        )
    )
    for value in values:
        value.validate_integrity()
    families = tuple(value.family_id for value in values)
    tune_ids = tuple(value.example_id for value in values)
    if (
        len(values) != _EXPECTED_TRAINING_FAMILIES
        or families != refit.training_family_ids
        or len(set(tune_ids)) != len(tune_ids)
        or not set(tune_ids).isdisjoint(refit.training_example_ids)
        or len(
            {
                value.pinned_v4_tune_example_artifact_sha256
                for value in values
            }
        )
        != len(values)
        or len(
            {value.pinned_v4_unit_receipt_sha256 for value in values}
        )
        != len(values)
        or any(value.held_family_id != refit.held_family_id for value in values)
        or any(
            value.v4_refit_artifact_sha256 != refit.artifact_sha256
            for value in values
        )
    ):
        raise ValueError(
            "microstep tune evidence must be disjoint and match the v4 fold"
        )
    mean_proposal = refit.mean_proposed_gains_tensor()
    structural_no_op = refit.mean_no_op or torch.equal(
        mean_proposal, torch.ones_like(mean_proposal)
    )
    if structural_no_op and any(
        value.structural_no_op_replayed_pinned_v4_unit_exactly is not True
        for value in values
    ):
        raise ValueError(
            "no-op microstep evidence must authenticate exact v4 unit replay"
        )
    if not structural_no_op and any(
        value.structural_no_op_replayed_pinned_v4_unit_exactly is not None
        for value in values
    ):
        raise ValueError("non-no-op microstep evidence cannot claim unit replay")
    unit = torch.tensor(
        [value.pinned_v4_unit_mean_teacher_kl for value in values],
        dtype=torch.float64,
    ).contiguous()
    plus = torch.tensor(
        [float(value.plus_token_teacher_kl.mean()) for value in values],
        dtype=torch.float64,
    ).contiguous()
    minus = torch.tensor(
        [float(value.minus_token_teacher_kl.mean()) for value in values],
        dtype=torch.float64,
    ).contiguous()
    return CandidateConditionedK64SymmetricMicrostepSelection(
        held_family_id=refit.held_family_id,
        refit_artifact_sha256=refit.artifact_sha256,
        training_family_ids=refit.training_family_ids,
        training_example_ids=refit.training_example_ids,
        tune_example_ids=tune_ids,
        tune_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        mean_proposed_gains=mean_proposal,
        unit_family_mean_teacher_kl=unit,
        plus_family_mean_teacher_kl=plus,
        minus_family_mean_teacher_kl=minus,
        mean_fit_predicted_derivative=refit.mean_predicted_derivative,
        mean_refit_was_no_op=refit.mean_no_op,
    )
