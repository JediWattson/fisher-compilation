"""Pure candidate-conditioned K64 gain-refit primitives.

The ordered K64 directions are frozen by an upstream whole-family-LOFO
token-Fisher fit.  This module only fits a bounded gain update from exact
teacher-KL VJPs with respect to the realized, post-cast H4 state at the
unit-gain K64 candidate, then selects a predeclared finite step on separate
tune prompts.  The analytic gain contraction uses the differentiable
interpretation of that final cast; exact finite-alpha execution remains the
authority for accepting a step.

Raw score matrices, gradient-Gram matrices, gradients, and gains remain ephemeral.
Metadata contains hashes and scalar summaries only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor


__all__ = [
    "CANDIDATE_GAIN_ALPHAS",
    "CANDIDATE_GAIN_RANK",
    "CandidateConditionedK64GainGradientExample",
    "CandidateConditionedK64GainRefit",
    "CandidateConditionedK64GainTuneExample",
    "CandidateConditionedK64GainTuneSelection",
    "contract_candidate_teacher_kl_gain_scores",
    "fit_candidate_conditioned_k64_gains",
    "select_candidate_conditioned_k64_gain_alpha",
]


CANDIDATE_GAIN_RANK = 64
CANDIDATE_GAIN_ALPHAS = (0.0, 0.25, 0.5, 1.0)
_EXPECTED_TRAINING_FAMILIES = 7
_DAMPING_FRACTION = 0.1
_DAMPING_FLOOR = 1.0e-12
_TRUST_RMS = 0.25
_GAIN_MINIMUM = 0.0
_GAIN_MAXIMUM = 1.5
_GRADIENT_EXAMPLE_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-gain-gradient-example:v1\0"
)
_REFIT_DOMAIN = b"fisher-graph:complete-h4-candidate-k64-gain-refit:v1\0"
_TUNE_EXAMPLE_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-gain-tune-example:v1\0"
)
_TUNE_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-gain-tune-selection:v1\0"
)
_TENSOR_DOMAIN = b"fisher-graph:complete-h4-candidate-k64-gain-tensor:v1\0"
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


def _validated_alphas() -> tuple[float, ...]:
    values = tuple(float(value) for value in CANDIDATE_GAIN_ALPHAS)
    if (
        values != CANDIDATE_GAIN_ALPHAS
        or values[0] != 0.0
        or values != tuple(sorted(set(values)))
        or any(not 0.0 <= value <= 1.0 for value in values)
    ):
        raise RuntimeError("candidate gain alpha protocol drifted")
    return values


def _expected_relevance_regularizer(relevance: Tensor) -> Tensor:
    tiny = torch.finfo(torch.float64).tiny
    positive = relevance[relevance > 0.0]
    scale = max(float(positive.median()) if positive.numel() else 0.0, tiny)
    normalized = torch.clamp(relevance / scale, min=tiny)
    return torch.diag(normalized).contiguous()


def _expected_damping(gradient_gram: Tensor) -> float:
    return max(
        _DAMPING_FRACTION * float(gradient_gram.diagonal().median()),
        _DAMPING_FLOOR,
    )


def _solve_damped_system(damped: Tensor, c: Tensor) -> Tensor:
    """Solve the GN system after a scalar normalization for tiny SPD floors."""

    if bool((c == 0.0).all()):
        return torch.zeros_like(c)
    scale = max(float(damped.abs().max()), float(c.abs().max()))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("candidate gain damped solve scale was invalid")
    try:
        result = torch.linalg.solve(damped / scale, -c / scale).contiguous()
    except RuntimeError as error:
        raise RuntimeError("candidate gain damped solve failed") from error
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("candidate gain damped solve was nonfinite")
    return result


def _finite_report_ratio(candidate: float, source: float) -> float:
    """Return a finite diagnostic ratio without changing the additive gate."""

    denominator = max(source, 1.0e-12)
    maximum = torch.finfo(torch.float64).max
    if candidate > maximum * denominator:
        return maximum
    return candidate / denominator


def contract_candidate_teacher_kl_gain_scores(
    *,
    tail_rows: Tensor,
    ordered_directions: Tensor,
    token_h4_gradients: Tensor,
) -> Tensor:
    """Return post-cast-H4 token pullbacks into continuous gain directions.

    The H4 VJP is exact at the realized candidate.  Contracting it with the
    pre-cast mode field treats the final float cast as locally differentiable;
    finite-alpha teacher-KL evaluation therefore remains the decision
    authority for the discrete executed candidate.
    """

    tail = _float64(tail_rows, label="candidate tail rows", ndim=2)
    directions = _float64(
        ordered_directions, label="candidate ordered directions", ndim=2
    )
    gradients = _float64(
        token_h4_gradients, label="candidate token H4 gradients", ndim=3
    )
    if (
        directions.shape != (CANDIDATE_GAIN_RANK, tail.shape[1])
        or gradients.shape[1:] != tail.shape
        or not torch.allclose(
            directions @ directions.T,
            torch.eye(CANDIDATE_GAIN_RANK, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError("candidate gain contraction geometry differs")
    amplitudes = (tail @ directions.T).contiguous()
    gradient_coordinates = torch.einsum(
        "trw,kw->trk", gradients, directions
    ).contiguous()
    return torch.einsum(
        "rk,trk->tk", amplitudes, gradient_coordinates
    ).contiguous()


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64GainGradientExample:
    example_id: str
    family_id: str
    token_gain_gradients: Tensor = field(repr=False)
    token_teacher_kl: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        example_id = _identifier(self.example_id, label="gradient example_id")
        family_id = _identifier(self.family_id, label="gradient family_id")
        gradients = _float64(
            self.token_gain_gradients, label="token gain gradients", ndim=2
        )
        teacher_kl = _float64(
            self.token_teacher_kl, label="candidate token teacher KL", ndim=1
        )
        if (
            gradients.shape != (teacher_kl.numel(), CANDIDATE_GAIN_RANK)
            or bool((teacher_kl < 0.0).any())
        ):
            raise ValueError("candidate gradient evidence geometry differs")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "token_gain_gradients", gradients)
        object.__setattr__(self, "token_teacher_kl", teacher_kl)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_GRADIENT_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return int(self.token_teacher_kl.numel())

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "supervised_token_count": self.supervised_tokens,
            "token_gain_gradients_shape": tuple(self.token_gain_gradients.shape),
            "token_gain_gradients_sha256": _tensor_sha256(
                self.token_gain_gradients
            ),
            "token_teacher_kl_sha256": _tensor_sha256(self.token_teacher_kl),
            "mean_teacher_kl": float(self.token_teacher_kl.mean()),
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.token_gain_gradients.dtype != torch.float64
            or self.token_gain_gradients.device.type != "cpu"
            or self.token_gain_gradients.requires_grad
            or not self.token_gain_gradients.is_contiguous()
            or self.token_teacher_kl.dtype != torch.float64
            or self.token_teacher_kl.device.type != "cpu"
            or self.token_teacher_kl.requires_grad
            or not self.token_teacher_kl.is_contiguous()
            or _sha256(
                _GRADIENT_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)
            )
            != _require_sha256(self.artifact_sha256, label="gradient example")
        ):
            raise RuntimeError("candidate gradient example payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64GainRefit:
    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    parent_fold_artifact_sha256: str
    ordered_directions_sha256: str
    ordered_token_fisher_relevance: Tensor = field(repr=False)
    residual_gradient_c: Tensor = field(repr=False)
    gradient_gram: Tensor = field(repr=False)
    relevance_regularizer: Tensor = field(repr=False)
    damped_system: Tensor = field(repr=False)
    raw_delta: Tensor = field(repr=False)
    proposed_gains: Tensor = field(repr=False)
    damping: float
    raw_delta_rms: float
    trust_scale: float
    predicted_derivative: float
    no_op: bool
    no_op_reason: str | None
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        held = _identifier(self.held_family_id, label="held_family_id")
        families = tuple(
            _identifier(value, label="training family_id")
            for value in self.training_family_ids
        )
        examples = tuple(
            _identifier(value, label="training example_id")
            for value in self.training_example_ids
        )
        evidence = tuple(
            _require_sha256(value, label="gradient example artifact")
            for value in self.training_example_artifact_sha256s
        )
        relevance = _float64(
            self.ordered_token_fisher_relevance,
            label="ordered token-Fisher relevance",
            ndim=1,
        )
        c = _float64(
            self.residual_gradient_c,
            label="candidate residual gradient c",
            ndim=1,
        )
        gradient_gram = _float64(
            self.gradient_gram, label="candidate gradient Gram", ndim=2
        )
        regularizer = _float64(
            self.relevance_regularizer,
            label="candidate relevance regularizer",
            ndim=2,
        )
        damped_system = _float64(
            self.damped_system, label="candidate damped system", ndim=2
        )
        raw_delta = _float64(self.raw_delta, label="raw gain delta", ndim=1)
        gains = _float64(self.proposed_gains, label="proposed gains", ndim=1)
        damping = float(self.damping)
        raw_rms = float(self.raw_delta_rms)
        trust_scale = float(self.trust_scale)
        derivative = float(self.predicted_derivative)
        expected_regularizer = _expected_relevance_regularizer(relevance)
        expected_damping = _expected_damping(gradient_gram)
        expected_damped = (
            gradient_gram + expected_damping * expected_regularizer
        ).contiguous()
        try:
            expected_raw_delta = _solve_damped_system(expected_damped, c)
        except RuntimeError as error:
            raise ValueError("candidate gain refit system is not solvable") from error
        expected_raw_rms = float(expected_raw_delta.square().mean().sqrt())
        expected_trust_scale = min(
            1.0,
            _TRUST_RMS
            / max(expected_raw_rms, torch.finfo(torch.float64).tiny),
        )
        expected_candidate_gains = torch.clamp(
            1.0 + expected_trust_scale * expected_raw_delta,
            min=_GAIN_MINIMUM,
            max=_GAIN_MAXIMUM,
        ).contiguous()
        candidate_derivative = float(
            torch.dot(c, expected_candidate_gains - 1.0)
        )
        expected_no_op = candidate_derivative >= 0.0
        expected_gains = (
            torch.ones_like(expected_candidate_gains)
            if expected_no_op
            else expected_candidate_gains
        )
        expected_derivative = 0.0 if expected_no_op else candidate_derivative
        expected_reason = (
            "nonnegative_predicted_derivative_after_trust_and_box"
            if expected_no_op
            else None
        )
        if (
            len(families) != _EXPECTED_TRAINING_FAMILIES
            or families != tuple(sorted(set(families)))
            or held in families
            or len(examples) != _EXPECTED_TRAINING_FAMILIES
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(examples)
            or relevance.shape != (CANDIDATE_GAIN_RANK,)
            or c.shape != relevance.shape
            or gradient_gram.shape
            != (CANDIDATE_GAIN_RANK, CANDIDATE_GAIN_RANK)
            or regularizer.shape != gradient_gram.shape
            or damped_system.shape != gradient_gram.shape
            or raw_delta.shape != relevance.shape
            or gains.shape != relevance.shape
            or bool((relevance < 0.0).any())
            or not torch.allclose(
                gradient_gram, gradient_gram.T, rtol=0.0, atol=1.0e-12
            )
            or not torch.allclose(
                regularizer, regularizer.T, rtol=0.0, atol=1.0e-12
            )
            or not torch.allclose(
                damped_system, damped_system.T, rtol=0.0, atol=1.0e-12
            )
            or float(torch.linalg.eigvalsh(gradient_gram)[0]) < -1.0e-10
            or not torch.equal(regularizer, expected_regularizer)
            or not math.isclose(
                damping, expected_damping, rel_tol=0.0, abs_tol=0.0
            )
            or not torch.allclose(
                damped_system, expected_damped, rtol=1.0e-12, atol=0.0
            )
            or not torch.allclose(
                raw_delta, expected_raw_delta, rtol=1.0e-9, atol=1.0e-12
            )
            or not math.isclose(
                raw_rms, expected_raw_rms, rel_tol=1.0e-12, abs_tol=1.0e-15
            )
            or not math.isclose(
                trust_scale,
                expected_trust_scale,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            or not torch.allclose(
                gains, expected_gains, rtol=1.0e-12, atol=1.0e-12
            )
            or not math.isclose(
                derivative,
                expected_derivative,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            or type(self.no_op) is not bool
            or self.no_op is not expected_no_op
            or self.no_op_reason != expected_reason
            or bool((gains < _GAIN_MINIMUM).any())
            or bool((gains > _GAIN_MAXIMUM).any())
            or not math.isfinite(damping)
            or damping < 0.0
            or not math.isfinite(raw_rms)
            or raw_rms < 0.0
            or not math.isfinite(trust_scale)
            or not 0.0 < trust_scale <= 1.0
            or not math.isfinite(derivative)
        ):
            raise ValueError("candidate gain refit payload is invalid")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(
            self,
            "parent_fold_artifact_sha256",
            _require_sha256(self.parent_fold_artifact_sha256, label="parent fold"),
        )
        object.__setattr__(
            self,
            "ordered_directions_sha256",
            _require_sha256(
                self.ordered_directions_sha256, label="ordered directions"
            ),
        )
        object.__setattr__(self, "ordered_token_fisher_relevance", relevance)
        object.__setattr__(self, "residual_gradient_c", c)
        object.__setattr__(self, "gradient_gram", gradient_gram)
        object.__setattr__(self, "relevance_regularizer", regularizer)
        object.__setattr__(self, "damped_system", damped_system)
        object.__setattr__(self, "raw_delta", raw_delta)
        object.__setattr__(self, "proposed_gains", gains)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "raw_delta_rms", raw_rms)
        object.__setattr__(self, "trust_scale", trust_scale)
        object.__setattr__(self, "predicted_derivative", derivative)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_REFIT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def proposed_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.proposed_gains.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        applied_delta = self.proposed_gains - 1.0
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "training_example_artifact_sha256s": (
                self.training_example_artifact_sha256s
            ),
            "parent_fold_artifact_sha256": self.parent_fold_artifact_sha256,
            "ordered_directions_sha256": self.ordered_directions_sha256,
            "rank": CANDIDATE_GAIN_RANK,
            "initial_gains": "all_ones",
            "ordered_token_fisher_relevance_sha256": _tensor_sha256(
                self.ordered_token_fisher_relevance
            ),
            "ordered_token_fisher_relevance_minimum": float(
                self.ordered_token_fisher_relevance.min()
            ),
            "ordered_token_fisher_relevance_median": float(
                self.ordered_token_fisher_relevance.median()
            ),
            "ordered_token_fisher_relevance_maximum": float(
                self.ordered_token_fisher_relevance.max()
            ),
            "residual_gradient_c_sha256": _tensor_sha256(
                self.residual_gradient_c
            ),
            "residual_gradient_c_l2_norm": float(
                torch.linalg.vector_norm(self.residual_gradient_c)
            ),
            "gradient_gram_sha256": _tensor_sha256(self.gradient_gram),
            "gradient_gram_trace": float(torch.trace(self.gradient_gram)),
            "gradient_gram_minimum_eigenvalue": float(
                torch.linalg.eigvalsh(self.gradient_gram)[0]
            ),
            "relevance_regularizer_sha256": _tensor_sha256(
                self.relevance_regularizer
            ),
            "relevance_regularizer_diagonal_median": float(
                self.relevance_regularizer.diagonal().median()
            ),
            "damped_system_sha256": _tensor_sha256(self.damped_system),
            "damping_fraction_of_gradient_gram_diagonal_median": (
                _DAMPING_FRACTION
            ),
            "damping_floor": _DAMPING_FLOOR,
            "damping": self.damping,
            "relevance_regularizer_normalized_diagonal_floor": (
                torch.finfo(torch.float64).tiny
            ),
            "zero_relevance_policy": (
                "normalize_by_positive_median_then_clamp_each_diagonal_at_"
                "float64_tiny"
            ),
            "raw_delta_sha256": _tensor_sha256(self.raw_delta),
            "raw_delta_rms": self.raw_delta_rms,
            "trust_rms_maximum": _TRUST_RMS,
            "trust_scale": self.trust_scale,
            "proposed_gains_sha256": _tensor_sha256(self.proposed_gains),
            "proposed_gain_minimum": float(self.proposed_gains.min()),
            "proposed_gain_mean": float(self.proposed_gains.mean()),
            "proposed_gain_maximum": float(self.proposed_gains.max()),
            "applied_delta_rms": float(applied_delta.square().mean().sqrt()),
            "gain_bounds": (_GAIN_MINIMUM, _GAIN_MAXIMUM),
            "predicted_derivative": self.predicted_derivative,
            "negative_predicted_derivative": self.predicted_derivative < 0.0,
            "no_op": self.no_op,
            "no_op_reason": self.no_op_reason,
            "aggregation": "equal_over_seven_families_then_equal_tokens",
            "fit_objective": "one_half_expected_squared_token_teacher_KL",
            "method": "one_step_damped_residual_Gauss_Newton",
            "not_claimed_as": "mean_KL_natural_gradient_or_exact_GGN",
            "matrix": (
                "gradient_Gram_plus_mu_times_diag_parent_relevance_over_"
                "positive_median"
            ),
            "solve": "negative_damped_system_inverse_times_residual_gradient_c",
            "held_family_used_for_refit": False,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _sha256(_REFIT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="candidate gain refit")
        ):
            raise RuntimeError("candidate gain refit payload drifted")


def fit_candidate_conditioned_k64_gains(
    examples: Iterable[CandidateConditionedK64GainGradientExample],
    *,
    held_family_id: str,
    parent_fold_artifact_sha256: str,
    ordered_directions_sha256: str,
    ordered_token_fisher_relevance: Tensor,
) -> CandidateConditionedK64GainRefit:
    """Fit one damped trust-clipped gain step without held-family evidence."""

    values = tuple(sorted(tuple(examples), key=lambda value: value.example_id))
    if not values or any(
        not isinstance(value, CandidateConditionedK64GainGradientExample)
        for value in values
    ):
        raise TypeError("candidate gain examples must be typed records")
    for value in values:
        value.validate_integrity()
    held = _identifier(held_family_id, label="held_family_id")
    families = tuple(sorted({value.family_id for value in values}))
    if (
        len(values) != _EXPECTED_TRAINING_FAMILIES
        or len(families) != _EXPECTED_TRAINING_FAMILIES
        or held in families
        or len({value.example_id for value in values}) != len(values)
    ):
        raise ValueError("candidate gain refit requires seven disjoint families")
    relevance = _float64(
        ordered_token_fisher_relevance,
        label="ordered token-Fisher relevance",
        ndim=1,
    )
    if relevance.shape != (CANDIDATE_GAIN_RANK,) or bool((relevance < 0.0).any()):
        raise ValueError("candidate gain relevance must be nonnegative K64")
    family_residual_gradients = tuple(
        (
            value.token_gain_gradients
            * value.token_teacher_kl.unsqueeze(1)
        ).mean(dim=0)
        for value in values
    )
    family_gradient_grams = tuple(
        value.token_gain_gradients.T @ value.token_gain_gradients
        / value.supervised_tokens
        for value in values
    )
    c = torch.stack(family_residual_gradients).mean(dim=0).contiguous()
    gradient_gram = torch.stack(family_gradient_grams).mean(dim=0)
    gradient_gram = (0.5 * (gradient_gram + gradient_gram.T)).contiguous()
    relevance_regularizer = _expected_relevance_regularizer(relevance)
    damping = _expected_damping(gradient_gram)
    damped = (gradient_gram + damping * relevance_regularizer).contiguous()
    raw_delta = _solve_damped_system(damped, c)
    raw_delta_rms = float(raw_delta.square().mean().sqrt())
    trust_scale = min(1.0, _TRUST_RMS / max(raw_delta_rms, torch.finfo(torch.float64).tiny))
    trusted_delta = (trust_scale * raw_delta).contiguous()
    proposed = torch.clamp(
        1.0 + trusted_delta, min=_GAIN_MINIMUM, max=_GAIN_MAXIMUM
    ).contiguous()
    applied_delta = proposed - 1.0
    predicted_derivative = float(torch.dot(c, applied_delta))
    no_op = predicted_derivative >= 0.0
    reason: str | None = None
    if no_op:
        proposed = torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
        predicted_derivative = 0.0
        reason = "nonnegative_predicted_derivative_after_trust_and_box"
    return CandidateConditionedK64GainRefit(
        held_family_id=held,
        training_family_ids=families,
        training_example_ids=tuple(value.example_id for value in values),
        training_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        parent_fold_artifact_sha256=parent_fold_artifact_sha256,
        ordered_directions_sha256=ordered_directions_sha256,
        ordered_token_fisher_relevance=relevance,
        residual_gradient_c=c,
        gradient_gram=gradient_gram,
        relevance_regularizer=relevance_regularizer,
        damped_system=damped,
        raw_delta=raw_delta,
        proposed_gains=proposed,
        damping=damping,
        raw_delta_rms=raw_delta_rms,
        trust_scale=trust_scale,
        predicted_derivative=predicted_derivative,
        no_op=no_op,
        no_op_reason=reason,
    )


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64GainTuneExample:
    example_id: str
    family_id: str
    token_teacher_kl_by_alpha: tuple[Tensor, ...] = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        alphas = _validated_alphas()
        example_id = _identifier(self.example_id, label="tune example_id")
        family_id = _identifier(self.family_id, label="tune family_id")
        values = tuple(
            _float64(value, label="tune token teacher KL", ndim=1)
            for value in self.token_teacher_kl_by_alpha
        )
        if (
            len(values) != len(alphas)
            or len({value.numel() for value in values}) != 1
            or any(bool((value < 0.0).any()) for value in values)
        ):
            raise ValueError("candidate gain tune grid differs")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "token_teacher_kl_by_alpha", values)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_TUNE_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def token_kl(self, alpha: float) -> Tensor:
        alphas = _validated_alphas()
        if type(alpha) is not float or alpha not in alphas:
            raise ValueError("tune alpha is outside the fixed grid")
        self.validate_integrity()
        return self.token_teacher_kl_by_alpha[alphas.index(alpha)].clone()

    @property
    def supervised_tokens(self) -> int:
        return int(self.token_teacher_kl_by_alpha[0].numel())

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "supervised_token_count": self.supervised_tokens,
            "alpha_hex_grid": tuple(alpha.hex() for alpha in _validated_alphas()),
            "token_teacher_kl_sha256_by_alpha": tuple(
                _tensor_sha256(value) for value in self.token_teacher_kl_by_alpha
            ),
            "mean_teacher_kl_by_alpha": tuple(
                float(value.mean()) for value in self.token_teacher_kl_by_alpha
            ),
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _sha256(_TUNE_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="gain tune example")
        ):
            raise RuntimeError("candidate gain tune example payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64GainTuneSelection:
    held_family_id: str
    refit_artifact_sha256: str
    tune_example_artifact_sha256s: tuple[str, ...]
    family_half_mean_squared_teacher_kl_by_alpha: Tensor = field(repr=False)
    family_mean_teacher_kl_by_alpha: Tensor = field(repr=False)
    aggregate_half_mean_squared_teacher_kl_by_alpha: tuple[float, ...]
    aggregate_mean_teacher_kl_by_alpha: tuple[float, ...]
    worst_family_ratio_by_alpha: tuple[float, ...]
    selected_alpha: float
    minimum_required_squared_KL_improvement: float
    minimum_required_mean_KL_improvement: float
    family_improved_or_equal_count_by_alpha: tuple[int, ...]
    proposed_gains: Tensor = field(repr=False)
    fit_predicted_derivative: float
    refit_was_no_op: bool
    selected_gains: Tensor = field(repr=False)
    selection_reason: str
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        alphas = _validated_alphas()
        family_squared = _float64(
            self.family_half_mean_squared_teacher_kl_by_alpha,
            label="tune family squared-KL ledger",
            ndim=2,
        )
        family_means = _float64(
            self.family_mean_teacher_kl_by_alpha,
            label="tune family mean-KL ledger",
            ndim=2,
        )
        squared_objectives = tuple(
            float(value)
            for value in self.aggregate_half_mean_squared_teacher_kl_by_alpha
        )
        mean_kls = tuple(
            float(value) for value in self.aggregate_mean_teacher_kl_by_alpha
        )
        ratios = tuple(float(value) for value in self.worst_family_ratio_by_alpha)
        selected_alpha = float(self.selected_alpha)
        minimum_squared = float(self.minimum_required_squared_KL_improvement)
        minimum_mean = float(self.minimum_required_mean_KL_improvement)
        family_counts = tuple(self.family_improved_or_equal_count_by_alpha)
        proposed = _float64(self.proposed_gains, label="refit proposed gains", ndim=1)
        fit_derivative = float(self.fit_predicted_derivative)
        gains = _float64(self.selected_gains, label="selected gains", ndim=1)
        if (
            family_squared.shape != (len(alphas), _EXPECTED_TRAINING_FAMILIES)
            or family_means.shape != family_squared.shape
            or bool((family_squared < 0.0).any())
            or bool((family_means < 0.0).any())
        ):
            raise ValueError("candidate gain tune family ledgers are invalid")
        expected_squared = tuple(
            math.fsum(float(value) for value in row) / len(row)
            for row in family_squared
        )
        expected_means = tuple(
            math.fsum(float(value) for value in row) / len(row)
            for row in family_means
        )
        baseline = tuple(float(value) for value in family_means[0])
        expected_ratios = tuple(
            max(
                _finite_report_ratio(candidate, source)
                for candidate, source in zip(
                    (float(value) for value in family_means[index]), baseline
                )
            )
            for index in range(len(alphas))
        )
        expected_counts = tuple(
            sum(
                float(candidate) <= source + 1.0e-8
                for candidate, source in zip(family_means[index], baseline)
            )
            for index in range(len(alphas))
        )
        expected_minimum_squared = max(1.0e-10, 1.0e-4 * expected_squared[0])
        expected_minimum_mean = max(1.0e-8, 1.0e-4 * expected_means[0])
        eligible: list[float] = []
        if fit_derivative < 0.0 and self.refit_was_no_op is False:
            for index, alpha in enumerate(alphas[1:], start=1):
                if (
                    expected_squared[0] - expected_squared[index]
                    >= expected_minimum_squared
                    and expected_means[0] - expected_means[index]
                    >= expected_minimum_mean
                    and all(
                        float(candidate) <= 1.05 * source + 1.0e-8
                        for candidate, source in zip(family_means[index], baseline)
                    )
                    and expected_counts[index] >= 4
                ):
                    eligible.append(alpha)
        expected_alpha = max(eligible, default=0.0)
        if expected_alpha > 0.0:
            expected_reason = (
                "largest_positive_alpha_cleared_improvement_and_family_guard"
            )
        elif self.refit_was_no_op:
            expected_reason = "alpha_zero_refit_was_no_op"
        elif fit_derivative >= 0.0:
            expected_reason = "alpha_zero_nonnegative_fit_predicted_derivative"
        else:
            expected_reason = "alpha_zero_fallback"
        expected_selected_gains = (
            1.0 + expected_alpha * (proposed - 1.0)
        ).contiguous()
        if (
            len(self.tune_example_artifact_sha256s)
            != _EXPECTED_TRAINING_FAMILIES
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.tune_example_artifact_sha256s
            )
            or len(squared_objectives) != len(alphas)
            or len(mean_kls) != len(alphas)
            or len(ratios) != len(alphas)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in squared_objectives
            )
            or any(not math.isfinite(value) or value < 0.0 for value in mean_kls)
            or any(not math.isfinite(value) or value < 0.0 for value in ratios)
            or selected_alpha not in alphas
            or squared_objectives != expected_squared
            or mean_kls != expected_means
            or ratios != expected_ratios
            or selected_alpha != expected_alpha
            or not math.isfinite(minimum_squared)
            or minimum_squared < 0.0
            or not math.isfinite(minimum_mean)
            or minimum_mean < 0.0
            or len(family_counts) != len(alphas)
            or any(type(value) is not int or not 0 <= value <= 7 for value in family_counts)
            or family_counts != expected_counts
            or minimum_squared != expected_minimum_squared
            or minimum_mean != expected_minimum_mean
            or proposed.shape != (CANDIDATE_GAIN_RANK,)
            or bool((proposed < _GAIN_MINIMUM).any())
            or bool((proposed > _GAIN_MAXIMUM).any())
            or not math.isfinite(fit_derivative)
            or type(self.refit_was_no_op) is not bool
            or (self.refit_was_no_op and fit_derivative != 0.0)
            or gains.shape != (CANDIDATE_GAIN_RANK,)
            or bool((gains < _GAIN_MINIMUM).any())
            or bool((gains > _GAIN_MAXIMUM).any())
            or (selected_alpha == 0.0 and not torch.equal(gains, torch.ones_like(gains)))
            or not torch.allclose(
                gains, expected_selected_gains, rtol=1.0e-12, atol=1.0e-12
            )
            or self.selection_reason != expected_reason
        ):
            raise ValueError("candidate gain tune selection payload is invalid")
        object.__setattr__(
            self,
            "held_family_id",
            _identifier(self.held_family_id, label="held_family_id"),
        )
        object.__setattr__(
            self,
            "refit_artifact_sha256",
            _require_sha256(self.refit_artifact_sha256, label="candidate refit"),
        )
        object.__setattr__(
            self,
            "family_half_mean_squared_teacher_kl_by_alpha",
            family_squared,
        )
        object.__setattr__(
            self,
            "family_mean_teacher_kl_by_alpha",
            family_means,
        )
        object.__setattr__(
            self,
            "aggregate_half_mean_squared_teacher_kl_by_alpha",
            squared_objectives,
        )
        object.__setattr__(self, "aggregate_mean_teacher_kl_by_alpha", mean_kls)
        object.__setattr__(self, "worst_family_ratio_by_alpha", ratios)
        object.__setattr__(self, "selected_alpha", selected_alpha)
        object.__setattr__(
            self,
            "minimum_required_squared_KL_improvement",
            minimum_squared,
        )
        object.__setattr__(
            self,
            "minimum_required_mean_KL_improvement",
            minimum_mean,
        )
        object.__setattr__(
            self,
            "family_improved_or_equal_count_by_alpha",
            family_counts,
        )
        object.__setattr__(self, "proposed_gains", proposed)
        object.__setattr__(self, "fit_predicted_derivative", fit_derivative)
        object.__setattr__(self, "selected_gains", gains)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_TUNE_SELECTION_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def selected_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.selected_gains.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "tune_example_artifact_sha256s": self.tune_example_artifact_sha256s,
            "alpha_hex_grid": tuple(alpha.hex() for alpha in _validated_alphas()),
            "family_half_mean_squared_teacher_kl_by_alpha_sha256": (
                _tensor_sha256(self.family_half_mean_squared_teacher_kl_by_alpha)
            ),
            "family_mean_teacher_kl_by_alpha_sha256": _tensor_sha256(
                self.family_mean_teacher_kl_by_alpha
            ),
            "aggregate_half_mean_squared_teacher_kl_by_alpha": (
                self.aggregate_half_mean_squared_teacher_kl_by_alpha
            ),
            "aggregate_mean_teacher_kl_by_alpha": (
                self.aggregate_mean_teacher_kl_by_alpha
            ),
            "worst_family_ratio_by_alpha": self.worst_family_ratio_by_alpha,
            "minimum_required_squared_KL_improvement": (
                self.minimum_required_squared_KL_improvement
            ),
            "minimum_required_mean_KL_improvement": (
                self.minimum_required_mean_KL_improvement
            ),
            "family_improved_or_equal_count_by_alpha": (
                self.family_improved_or_equal_count_by_alpha
            ),
            "proposed_gains_sha256": _tensor_sha256(self.proposed_gains),
            "fit_predicted_derivative": self.fit_predicted_derivative,
            "refit_was_no_op": self.refit_was_no_op,
            "selected_alpha_hex": self.selected_alpha.hex(),
            "selected_gains_sha256": _tensor_sha256(self.selected_gains),
            "selected_gain_minimum": float(self.selected_gains.min()),
            "selected_gain_mean": float(self.selected_gains.mean()),
            "selected_gain_maximum": float(self.selected_gains.max()),
            "selection_reason": self.selection_reason,
            "largest_eligible_positive_alpha_selected": self.selected_alpha > 0.0,
            "alpha_zero_is_abstention": self.selected_alpha == 0.0,
            "held_family_used_for_tune": False,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _sha256(_TUNE_SELECTION_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="gain tune selection")
        ):
            raise RuntimeError("candidate gain tune selection payload drifted")


def select_candidate_conditioned_k64_gain_alpha(
    refit: CandidateConditionedK64GainRefit,
    tune_examples: Sequence[CandidateConditionedK64GainTuneExample],
) -> CandidateConditionedK64GainTuneSelection:
    """Select the largest safe positive finite step, otherwise alpha zero."""

    if not isinstance(refit, CandidateConditionedK64GainRefit):
        raise TypeError("refit must be a candidate gain refit")
    refit.validate_integrity()
    values = tuple(sorted(tuple(tune_examples), key=lambda value: value.example_id))
    if any(
        not isinstance(value, CandidateConditionedK64GainTuneExample)
        for value in values
    ):
        raise TypeError("tune examples must be typed records")
    for value in values:
        value.validate_integrity()
    families = tuple(sorted({value.family_id for value in values}))
    tune_ids = {value.example_id for value in values}
    if (
        len(values) != _EXPECTED_TRAINING_FAMILIES
        or families != refit.training_family_ids
        or len({value.example_id for value in values}) != len(values)
        or not tune_ids.isdisjoint(refit.training_example_ids)
    ):
        raise ValueError(
            "gain tune examples must be disjoint and match the seven training families"
        )
    alphas = _validated_alphas()
    means_by_alpha = tuple(
        tuple(float(value.token_kl(alpha).mean()) for value in values)
        for alpha in alphas
    )
    mean_kls = tuple(
        math.fsum(means) / len(means) for means in means_by_alpha
    )
    squared_by_alpha = tuple(
        tuple(
            0.5 * float(value.token_kl(alpha).square().mean())
            for value in values
        )
        for alpha in alphas
    )
    squared_objectives = tuple(
        math.fsum(values_at_alpha) / len(values_at_alpha)
        for values_at_alpha in squared_by_alpha
    )
    baseline = means_by_alpha[0]
    ratios = tuple(
        max(
            _finite_report_ratio(candidate, source)
            for candidate, source in zip(means, baseline)
        )
        for means in means_by_alpha
    )
    minimum_squared = max(1.0e-10, 1.0e-4 * squared_objectives[0])
    minimum_mean = max(1.0e-8, 1.0e-4 * mean_kls[0])
    family_counts = tuple(
        sum(
            candidate <= source + 1.0e-8
            for candidate, source in zip(means, baseline)
        )
        for means in means_by_alpha
    )
    selected = 0.0
    reason = "alpha_zero_fallback"
    if refit.predicted_derivative < 0.0 and not refit.no_op:
        for alpha in reversed(alphas[1:]):
            index = alphas.index(alpha)
            if (
                squared_objectives[0] - squared_objectives[index]
                >= minimum_squared
                and mean_kls[0] - mean_kls[index] >= minimum_mean
                and all(
                    candidate <= 1.05 * source + 1.0e-8
                    for candidate, source in zip(means_by_alpha[index], baseline)
                )
                and family_counts[index] >= 4
            ):
                selected = alpha
                reason = "largest_positive_alpha_cleared_improvement_and_family_guard"
                break
    elif refit.no_op:
        reason = "alpha_zero_refit_was_no_op"
    else:
        reason = "alpha_zero_nonnegative_fit_predicted_derivative"
    proposed = refit.proposed_gains_tensor()
    selected_gains = (1.0 + selected * (proposed - 1.0)).contiguous()
    return CandidateConditionedK64GainTuneSelection(
        held_family_id=refit.held_family_id,
        refit_artifact_sha256=refit.artifact_sha256,
        tune_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        family_half_mean_squared_teacher_kl_by_alpha=torch.tensor(
            squared_by_alpha, dtype=torch.float64
        ).contiguous(),
        family_mean_teacher_kl_by_alpha=torch.tensor(
            means_by_alpha, dtype=torch.float64
        ).contiguous(),
        aggregate_half_mean_squared_teacher_kl_by_alpha=squared_objectives,
        aggregate_mean_teacher_kl_by_alpha=mean_kls,
        worst_family_ratio_by_alpha=ratios,
        selected_alpha=selected,
        minimum_required_squared_KL_improvement=minimum_squared,
        minimum_required_mean_KL_improvement=minimum_mean,
        family_improved_or_equal_count_by_alpha=family_counts,
        proposed_gains=proposed,
        fit_predicted_derivative=refit.predicted_derivative,
        refit_was_no_op=refit.no_op,
        selected_gains=selected_gains,
        selection_reason=reason,
    )
