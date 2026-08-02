"""Pure dual-objective candidate-conditioned K64 gain-refit primitives.

One teacher-KL VJP bank supplies the exact family-equal mean-KL gradient,
the squared-residual gradient, and their shared empirical gradient Gram.  The
mean-KL direction is the only optimization arm.  Reversing the residual
direction is retained as an explicitly diagnostic sign-control arm.

Raw score matrices, Gram matrices, gradients, and gains remain ephemeral.
Metadata contains hashes and scalar summaries only.  Exact finite-step tune
executions remain authoritative for selecting either predeclared grid point.
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
    "CANDIDATE_GAIN_RANK",
    "MEAN_KL_GAIN_ALPHAS",
    "REVERSE_RESIDUAL_GAIN_BETAS",
    "CandidateConditionedK64GainGradientExampleV4",
    "CandidateConditionedK64MeanKLRefit",
    "CandidateConditionedK64DualTuneExample",
    "CandidateConditionedK64DualTuneSelection",
    "contract_candidate_teacher_kl_gain_scores",
    "fit_candidate_conditioned_k64_mean_kl_gains",
    "select_candidate_conditioned_k64_dual_tune_steps",
]


CANDIDATE_GAIN_RANK = 64
MEAN_KL_GAIN_ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)
REVERSE_RESIDUAL_GAIN_BETAS = (0.0, 0.125, 0.25, 0.5)
_EXPECTED_TRAINING_FAMILIES = 7
_DAMPING_FRACTION = 0.1
_DAMPING_FLOOR = 1.0e-12
_TRUST_RMS = 0.25
_GAIN_MINIMUM = 0.0
_GAIN_MAXIMUM = 1.5
_GRADIENT_EXAMPLE_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-gain-gradient-example:v4\0"
)
_REFIT_DOMAIN = b"fisher-graph:complete-h4-candidate-k64-mean-kl-refit:v4\0"
_TUNE_EXAMPLE_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-dual-tune-example:v4\0"
)
_TUNE_SELECTION_DOMAIN = (
    b"fisher-graph:complete-h4-candidate-k64-dual-tune-selection:v4\0"
)
_TENSOR_DOMAIN = b"fisher-graph:complete-h4-candidate-k64-gain-tensor:v4\0"
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


def _validated_grid(
    values: tuple[float, ...], *, label: str, maximum: float
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if (
        converted != values
        or converted[0] != 0.0
        or converted != tuple(sorted(set(converted)))
        or any(not 0.0 <= value <= maximum for value in converted)
    ):
        raise RuntimeError(f"{label} protocol drifted")
    return converted


def _mean_alphas() -> tuple[float, ...]:
    return _validated_grid(
        MEAN_KL_GAIN_ALPHAS, label="mean-KL alpha", maximum=1.0
    )


def _reverse_betas() -> tuple[float, ...]:
    return _validated_grid(
        REVERSE_RESIDUAL_GAIN_BETAS,
        label="reverse-residual beta",
        maximum=0.5,
    )


def _expected_relevance_regularizer(relevance: Tensor) -> Tensor:
    tiny = torch.finfo(torch.float64).tiny
    positive = relevance[relevance > 0.0]
    scale = max(float(positive.median()) if positive.numel() else 0.0, tiny)
    return torch.diag(torch.clamp(relevance / scale, min=tiny)).contiguous()


def _expected_damping(gradient_gram: Tensor) -> float:
    return max(
        _DAMPING_FRACTION * float(gradient_gram.diagonal().median()),
        _DAMPING_FLOOR,
    )


def _solve_damped_system(damped: Tensor, gradient: Tensor) -> Tensor:
    if bool((gradient == 0.0).all()):
        return torch.zeros_like(gradient)
    scale = max(float(damped.abs().max()), float(gradient.abs().max()))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("candidate gain damped solve scale was invalid")
    try:
        result = torch.linalg.solve(
            damped / scale, -gradient / scale
        ).contiguous()
    except RuntimeError as error:
        raise RuntimeError("candidate gain damped solve failed") from error
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("candidate gain damped solve was nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class _DirectionDerivation:
    raw_delta: Tensor
    raw_delta_rms: float
    trust_scale: float
    proposed_gains: Tensor
    predicted_derivative: float
    no_op: bool
    no_op_reason: str | None


def _derive_direction(damped: Tensor, gradient: Tensor) -> _DirectionDerivation:
    raw_delta = _solve_damped_system(damped, gradient)
    raw_rms = float(raw_delta.square().mean().sqrt())
    trust_scale = min(
        1.0,
        _TRUST_RMS / max(raw_rms, torch.finfo(torch.float64).tiny),
    )
    candidate = torch.clamp(
        1.0 + trust_scale * raw_delta,
        min=_GAIN_MINIMUM,
        max=_GAIN_MAXIMUM,
    ).contiguous()
    derivative = float(torch.dot(gradient, candidate - 1.0))
    no_op = derivative >= 0.0
    if no_op:
        candidate = torch.ones_like(candidate)
        derivative = 0.0
    return _DirectionDerivation(
        raw_delta=raw_delta,
        raw_delta_rms=raw_rms,
        trust_scale=trust_scale,
        proposed_gains=candidate,
        predicted_derivative=derivative,
        no_op=no_op,
        no_op_reason=(
            "nonnegative_predicted_derivative_after_trust_and_box"
            if no_op
            else None
        ),
    )


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = float(
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(left, right)) / denominator


def _finite_report_ratio(candidate: float, source: float) -> float:
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
    """Contract exact realized-H4 teacher-KL VJPs into K64 gain scores."""

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
class CandidateConditionedK64GainGradientExampleV4:
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
            "shared_vjp_bank_supplies_b_c_and_F": True,
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
            raise RuntimeError("candidate v4 gradient example payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64MeanKLRefit:
    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    parent_fold_artifact_sha256: str
    ordered_directions_sha256: str
    ordered_token_fisher_relevance: Tensor = field(repr=False)
    mean_gradient_b: Tensor = field(repr=False)
    residual_gradient_c: Tensor = field(repr=False)
    gradient_gram: Tensor = field(repr=False)
    relevance_regularizer: Tensor = field(repr=False)
    damped_system: Tensor = field(repr=False)
    mean_raw_delta: Tensor = field(repr=False)
    residual_raw_delta: Tensor = field(repr=False)
    mean_proposed_gains: Tensor = field(repr=False)
    residual_proposed_gains: Tensor = field(repr=False)
    damping: float
    mean_raw_delta_rms: float
    residual_raw_delta_rms: float
    mean_trust_scale: float
    residual_trust_scale: float
    mean_predicted_derivative: float
    residual_predicted_derivative: float
    mean_gradient_on_residual_delta: float
    residual_gradient_on_mean_delta: float
    gradient_cosine: float
    applied_delta_cosine: float
    mean_no_op: bool
    residual_no_op: bool
    mean_no_op_reason: str | None
    residual_no_op_reason: str | None
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
        b = _float64(self.mean_gradient_b, label="mean-KL gradient b", ndim=1)
        c = _float64(
            self.residual_gradient_c, label="residual gradient c", ndim=1
        )
        gram = _float64(self.gradient_gram, label="gradient Gram", ndim=2)
        regularizer = _float64(
            self.relevance_regularizer, label="relevance regularizer", ndim=2
        )
        damped = _float64(self.damped_system, label="damped system", ndim=2)
        mean_raw = _float64(self.mean_raw_delta, label="mean raw delta", ndim=1)
        residual_raw = _float64(
            self.residual_raw_delta, label="residual raw delta", ndim=1
        )
        mean_gains = _float64(
            self.mean_proposed_gains, label="mean proposed gains", ndim=1
        )
        residual_gains = _float64(
            self.residual_proposed_gains,
            label="residual proposed gains",
            ndim=1,
        )

        expected_regularizer = _expected_relevance_regularizer(relevance)
        expected_damping = _expected_damping(gram)
        expected_damped = (
            gram + expected_damping * expected_regularizer
        ).contiguous()
        try:
            expected_mean = _derive_direction(expected_damped, b)
            expected_residual = _derive_direction(expected_damped, c)
        except RuntimeError as error:
            raise ValueError(
                "candidate v4 gain refit system is not solvable"
            ) from error
        mean_delta = expected_mean.proposed_gains - 1.0
        residual_delta = expected_residual.proposed_gains - 1.0
        expected_cross_mean = float(torch.dot(b, residual_delta))
        expected_cross_residual = float(torch.dot(c, mean_delta))
        expected_gradient_cosine = _cosine(b, c)
        expected_delta_cosine = _cosine(mean_delta, residual_delta)
        scalar_pairs = (
            (self.damping, expected_damping),
            (self.mean_raw_delta_rms, expected_mean.raw_delta_rms),
            (self.residual_raw_delta_rms, expected_residual.raw_delta_rms),
            (self.mean_trust_scale, expected_mean.trust_scale),
            (self.residual_trust_scale, expected_residual.trust_scale),
            (
                self.mean_predicted_derivative,
                expected_mean.predicted_derivative,
            ),
            (
                self.residual_predicted_derivative,
                expected_residual.predicted_derivative,
            ),
            (self.mean_gradient_on_residual_delta, expected_cross_mean),
            (self.residual_gradient_on_mean_delta, expected_cross_residual),
            (self.gradient_cosine, expected_gradient_cosine),
            (self.applied_delta_cosine, expected_delta_cosine),
        )
        rank_shape = (CANDIDATE_GAIN_RANK,)
        matrix_shape = (CANDIDATE_GAIN_RANK, CANDIDATE_GAIN_RANK)
        if (
            len(families) != _EXPECTED_TRAINING_FAMILIES
            or families != tuple(sorted(set(families)))
            or held in families
            or len(examples) != _EXPECTED_TRAINING_FAMILIES
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(examples)
            or relevance.shape != rank_shape
            or b.shape != rank_shape
            or c.shape != rank_shape
            or gram.shape != matrix_shape
            or regularizer.shape != matrix_shape
            or damped.shape != matrix_shape
            or mean_raw.shape != rank_shape
            or residual_raw.shape != rank_shape
            or mean_gains.shape != rank_shape
            or residual_gains.shape != rank_shape
            or bool((relevance < 0.0).any())
            or not torch.allclose(gram, gram.T, rtol=0.0, atol=1.0e-12)
            or not torch.allclose(
                regularizer, regularizer.T, rtol=0.0, atol=1.0e-12
            )
            or not torch.allclose(damped, damped.T, rtol=0.0, atol=1.0e-12)
            or float(torch.linalg.eigvalsh(gram)[0]) < -1.0e-10
            or not torch.equal(regularizer, expected_regularizer)
            or not torch.allclose(
                damped, expected_damped, rtol=1.0e-12, atol=0.0
            )
            or not torch.allclose(
                mean_raw, expected_mean.raw_delta, rtol=1.0e-9, atol=1.0e-12
            )
            or not torch.allclose(
                residual_raw,
                expected_residual.raw_delta,
                rtol=1.0e-9,
                atol=1.0e-12,
            )
            or not torch.allclose(
                mean_gains,
                expected_mean.proposed_gains,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            or not torch.allclose(
                residual_gains,
                expected_residual.proposed_gains,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            or any(
                not math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                for actual, expected in scalar_pairs
            )
            or self.mean_no_op is not expected_mean.no_op
            or self.residual_no_op is not expected_residual.no_op
            or self.mean_no_op_reason != expected_mean.no_op_reason
            or self.residual_no_op_reason != expected_residual.no_op_reason
            or type(self.mean_no_op) is not bool
            or type(self.residual_no_op) is not bool
            or bool((mean_gains < _GAIN_MINIMUM).any())
            or bool((mean_gains > _GAIN_MAXIMUM).any())
            or bool((residual_gains < _GAIN_MINIMUM).any())
            or bool((residual_gains > _GAIN_MAXIMUM).any())
        ):
            raise ValueError("candidate v4 gain refit payload is invalid")

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
        for name, value in (
            ("ordered_token_fisher_relevance", relevance),
            ("mean_gradient_b", b),
            ("residual_gradient_c", c),
            ("gradient_gram", gram),
            ("relevance_regularizer", regularizer),
            ("damped_system", damped),
            ("mean_raw_delta", mean_raw),
            ("residual_raw_delta", residual_raw),
            ("mean_proposed_gains", mean_gains),
            ("residual_proposed_gains", residual_gains),
        ):
            object.__setattr__(self, name, value)
        for name, value in (
            ("damping", self.damping),
            ("mean_raw_delta_rms", self.mean_raw_delta_rms),
            ("residual_raw_delta_rms", self.residual_raw_delta_rms),
            ("mean_trust_scale", self.mean_trust_scale),
            ("residual_trust_scale", self.residual_trust_scale),
            ("mean_predicted_derivative", self.mean_predicted_derivative),
            (
                "residual_predicted_derivative",
                self.residual_predicted_derivative,
            ),
            (
                "mean_gradient_on_residual_delta",
                self.mean_gradient_on_residual_delta,
            ),
            (
                "residual_gradient_on_mean_delta",
                self.residual_gradient_on_mean_delta,
            ),
            ("gradient_cosine", self.gradient_cosine),
            ("applied_delta_cosine", self.applied_delta_cosine),
        ):
            object.__setattr__(self, name, float(value))
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_REFIT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def mean_proposed_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.mean_proposed_gains.clone().contiguous()

    def residual_proposed_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.residual_proposed_gains.clone().contiguous()

    def reverse_residual_gains_tensor(self, beta: float) -> Tensor:
        betas = _reverse_betas()
        if type(beta) is not float or beta not in betas:
            raise ValueError("reverse beta is outside the fixed diagnostic grid")
        self.validate_integrity()
        return (
            1.0 - beta * (self.residual_proposed_gains - 1.0)
        ).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        mean_delta = self.mean_proposed_gains - 1.0
        residual_delta = self.residual_proposed_gains - 1.0
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
            "mean_gradient_b_sha256": _tensor_sha256(self.mean_gradient_b),
            "mean_gradient_b_l2_norm": float(
                torch.linalg.vector_norm(self.mean_gradient_b)
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
            "mean_raw_delta_sha256": _tensor_sha256(self.mean_raw_delta),
            "residual_raw_delta_sha256": _tensor_sha256(
                self.residual_raw_delta
            ),
            "mean_raw_delta_rms": self.mean_raw_delta_rms,
            "residual_raw_delta_rms": self.residual_raw_delta_rms,
            "trust_rms_maximum": _TRUST_RMS,
            "mean_trust_scale": self.mean_trust_scale,
            "residual_trust_scale": self.residual_trust_scale,
            "mean_proposed_gains_sha256": _tensor_sha256(
                self.mean_proposed_gains
            ),
            "residual_proposed_gains_sha256": _tensor_sha256(
                self.residual_proposed_gains
            ),
            "mean_proposed_gain_minimum": float(self.mean_proposed_gains.min()),
            "mean_proposed_gain_mean": float(self.mean_proposed_gains.mean()),
            "mean_proposed_gain_maximum": float(self.mean_proposed_gains.max()),
            "residual_proposed_gain_minimum": float(
                self.residual_proposed_gains.min()
            ),
            "residual_proposed_gain_mean": float(
                self.residual_proposed_gains.mean()
            ),
            "residual_proposed_gain_maximum": float(
                self.residual_proposed_gains.max()
            ),
            "mean_applied_delta_rms": float(mean_delta.square().mean().sqrt()),
            "residual_applied_delta_rms": float(
                residual_delta.square().mean().sqrt()
            ),
            "gain_bounds": (_GAIN_MINIMUM, _GAIN_MAXIMUM),
            "mean_predicted_derivative": self.mean_predicted_derivative,
            "residual_predicted_derivative": self.residual_predicted_derivative,
            "mean_gradient_on_residual_delta": (
                self.mean_gradient_on_residual_delta
            ),
            "residual_gradient_on_mean_delta": (
                self.residual_gradient_on_mean_delta
            ),
            "gradient_cosine": self.gradient_cosine,
            "applied_delta_cosine": self.applied_delta_cosine,
            "mean_no_op": self.mean_no_op,
            "residual_no_op": self.residual_no_op,
            "mean_no_op_reason": self.mean_no_op_reason,
            "residual_no_op_reason": self.residual_no_op_reason,
            "aggregation": "equal_over_seven_families_then_equal_tokens",
            "mean_fit_objective": "expected_token_teacher_KL",
            "residual_fit_objective": "one_half_expected_squared_token_teacher_KL",
            "shared_curvature": "family_equal_empirical_gain_score_Gram",
            "mean_method": "one_step_OPG_preconditioned_mean_KL_descent",
            "residual_method": "one_step_damped_residual_Gauss_Newton",
            "reverse_residual_use": "diagnostic_sign_control_only",
            "reverse_residual_is_optimizer": False,
            "matrix": (
                "gradient_Gram_plus_mu_times_diag_parent_relevance_over_"
                "positive_median"
            ),
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
            != _require_sha256(self.artifact_sha256, label="candidate v4 refit")
        ):
            raise RuntimeError("candidate v4 gain refit payload drifted")


def fit_candidate_conditioned_k64_mean_kl_gains(
    examples: Iterable[CandidateConditionedK64GainGradientExampleV4],
    *,
    held_family_id: str,
    parent_fold_artifact_sha256: str,
    ordered_directions_sha256: str,
    ordered_token_fisher_relevance: Tensor,
) -> CandidateConditionedK64MeanKLRefit:
    """Fit mean-KL and residual controls from one seven-family VJP bank."""

    values = tuple(sorted(tuple(examples), key=lambda value: value.example_id))
    if not values or any(
        not isinstance(value, CandidateConditionedK64GainGradientExampleV4)
        for value in values
    ):
        raise TypeError("candidate v4 gain examples must be typed records")
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
        raise ValueError("candidate v4 refit requires seven disjoint families")
    relevance = _float64(
        ordered_token_fisher_relevance,
        label="ordered token-Fisher relevance",
        ndim=1,
    )
    if relevance.shape != (CANDIDATE_GAIN_RANK,) or bool((relevance < 0.0).any()):
        raise ValueError("candidate v4 relevance must be nonnegative K64")

    family_b = tuple(value.token_gain_gradients.mean(dim=0) for value in values)
    family_c = tuple(
        (
            value.token_gain_gradients
            * value.token_teacher_kl.unsqueeze(1)
        ).mean(dim=0)
        for value in values
    )
    family_grams = tuple(
        value.token_gain_gradients.T
        @ value.token_gain_gradients
        / value.supervised_tokens
        for value in values
    )
    b = torch.stack(family_b).mean(dim=0).contiguous()
    c = torch.stack(family_c).mean(dim=0).contiguous()
    gram = torch.stack(family_grams).mean(dim=0)
    gram = (0.5 * (gram + gram.T)).contiguous()
    regularizer = _expected_relevance_regularizer(relevance)
    damping = _expected_damping(gram)
    damped = (gram + damping * regularizer).contiguous()
    mean = _derive_direction(damped, b)
    residual = _derive_direction(damped, c)
    mean_delta = mean.proposed_gains - 1.0
    residual_delta = residual.proposed_gains - 1.0

    return CandidateConditionedK64MeanKLRefit(
        held_family_id=held,
        training_family_ids=families,
        training_example_ids=tuple(value.example_id for value in values),
        training_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        parent_fold_artifact_sha256=parent_fold_artifact_sha256,
        ordered_directions_sha256=ordered_directions_sha256,
        ordered_token_fisher_relevance=relevance,
        mean_gradient_b=b,
        residual_gradient_c=c,
        gradient_gram=gram,
        relevance_regularizer=regularizer,
        damped_system=damped,
        mean_raw_delta=mean.raw_delta,
        residual_raw_delta=residual.raw_delta,
        mean_proposed_gains=mean.proposed_gains,
        residual_proposed_gains=residual.proposed_gains,
        damping=damping,
        mean_raw_delta_rms=mean.raw_delta_rms,
        residual_raw_delta_rms=residual.raw_delta_rms,
        mean_trust_scale=mean.trust_scale,
        residual_trust_scale=residual.trust_scale,
        mean_predicted_derivative=mean.predicted_derivative,
        residual_predicted_derivative=residual.predicted_derivative,
        mean_gradient_on_residual_delta=float(torch.dot(b, residual_delta)),
        residual_gradient_on_mean_delta=float(torch.dot(c, mean_delta)),
        gradient_cosine=_cosine(b, c),
        applied_delta_cosine=_cosine(mean_delta, residual_delta),
        mean_no_op=mean.no_op,
        residual_no_op=residual.no_op,
        mean_no_op_reason=mean.no_op_reason,
        residual_no_op_reason=residual.no_op_reason,
    )


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64DualTuneExample:
    example_id: str
    family_id: str
    unit_token_teacher_kl: Tensor = field(repr=False)
    mean_token_teacher_kl_by_positive_alpha: tuple[Tensor, ...] = field(repr=False)
    reverse_token_teacher_kl_by_positive_beta: tuple[Tensor, ...] = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        example_id = _identifier(self.example_id, label="tune example_id")
        family_id = _identifier(self.family_id, label="tune family_id")
        unit = _float64(
            self.unit_token_teacher_kl, label="unit token teacher KL", ndim=1
        )
        mean_values = tuple(
            _float64(value, label="mean-arm token teacher KL", ndim=1)
            for value in self.mean_token_teacher_kl_by_positive_alpha
        )
        reverse_values = tuple(
            _float64(value, label="reverse-arm token teacher KL", ndim=1)
            for value in self.reverse_token_teacher_kl_by_positive_beta
        )
        if (
            len(mean_values) != len(_mean_alphas()) - 1
            or len(reverse_values) != len(_reverse_betas()) - 1
            or any(value.numel() != unit.numel() for value in mean_values)
            or any(value.numel() != unit.numel() for value in reverse_values)
            or bool((unit < 0.0).any())
            or any(bool((value < 0.0).any()) for value in mean_values)
            or any(bool((value < 0.0).any()) for value in reverse_values)
        ):
            raise ValueError("candidate v4 dual tune grid differs")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "unit_token_teacher_kl", unit)
        object.__setattr__(
            self, "mean_token_teacher_kl_by_positive_alpha", mean_values
        )
        object.__setattr__(
            self, "reverse_token_teacher_kl_by_positive_beta", reverse_values
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_TUNE_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return int(self.unit_token_teacher_kl.numel())

    def mean_token_kl(self, alpha: float) -> Tensor:
        alphas = _mean_alphas()
        if type(alpha) is not float or alpha not in alphas:
            raise ValueError("mean alpha is outside the fixed grid")
        self.validate_integrity()
        if alpha == 0.0:
            return self.unit_token_teacher_kl.clone()
        return self.mean_token_teacher_kl_by_positive_alpha[
            alphas.index(alpha) - 1
        ].clone()

    def reverse_token_kl(self, beta: float) -> Tensor:
        betas = _reverse_betas()
        if type(beta) is not float or beta not in betas:
            raise ValueError("reverse beta is outside the fixed grid")
        self.validate_integrity()
        if beta == 0.0:
            return self.unit_token_teacher_kl.clone()
        return self.reverse_token_teacher_kl_by_positive_beta[
            betas.index(beta) - 1
        ].clone()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        mean_values = (
            (self.unit_token_teacher_kl,)
            + self.mean_token_teacher_kl_by_positive_alpha
        )
        reverse_values = (
            (self.unit_token_teacher_kl,)
            + self.reverse_token_teacher_kl_by_positive_beta
        )
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "supervised_token_count": self.supervised_tokens,
            "mean_alpha_hex_grid": tuple(
                alpha.hex() for alpha in _mean_alphas()
            ),
            "reverse_beta_hex_grid": tuple(
                beta.hex() for beta in _reverse_betas()
            ),
            "unit_token_teacher_kl_sha256": _tensor_sha256(
                self.unit_token_teacher_kl
            ),
            "mean_positive_token_teacher_kl_sha256s": tuple(
                _tensor_sha256(value)
                for value in self.mean_token_teacher_kl_by_positive_alpha
            ),
            "reverse_positive_token_teacher_kl_sha256s": tuple(
                _tensor_sha256(value)
                for value in self.reverse_token_teacher_kl_by_positive_beta
            ),
            "mean_teacher_kl_by_alpha": tuple(
                float(value.mean()) for value in mean_values
            ),
            "reverse_mean_teacher_kl_by_beta": tuple(
                float(value.mean()) for value in reverse_values
            ),
            "alpha_zero_and_beta_zero_share_one_unit_execution": True,
            "reverse_arm_role": "diagnostic_sign_control_only",
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _sha256(_TUNE_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="dual tune example")
        ):
            raise RuntimeError("candidate v4 dual tune example payload drifted")


def _ledger_summary(
    family_means: Tensor,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    aggregate = tuple(
        math.fsum(float(value) for value in row) / len(row)
        for row in family_means
    )
    baseline = tuple(float(value) for value in family_means[0])
    ratios = tuple(
        max(
            _finite_report_ratio(float(candidate), source)
            for candidate, source in zip(row, baseline)
        )
        for row in family_means
    )
    counts = tuple(
        sum(
            float(candidate) <= source + 1.0e-8
            for candidate, source in zip(row, baseline)
        )
        for row in family_means
    )
    return aggregate, ratios, counts


def _largest_eligible_step(
    *,
    grid: tuple[float, ...],
    family_means: Tensor,
    aggregate_means: tuple[float, ...],
    family_counts: tuple[int, ...],
    fit_derivative: float,
    refit_no_op: bool,
) -> float:
    if fit_derivative >= 0.0 or refit_no_op:
        return 0.0
    baseline = tuple(float(value) for value in family_means[0])
    minimum = max(1.0e-8, 1.0e-4 * aggregate_means[0])
    eligible = []
    for index, step in enumerate(grid[1:], start=1):
        if (
            aggregate_means[0] - aggregate_means[index] >= minimum
            and all(
                float(candidate) <= 1.05 * source + 1.0e-8
                for candidate, source in zip(family_means[index], baseline)
            )
            and family_counts[index] >= 4
        ):
            eligible.append(step)
    return max(eligible, default=0.0)


def _selection_reason(
    *, step: float, refit_no_op: bool, fit_derivative: float, label: str
) -> str:
    if step > 0.0:
        return f"largest_positive_{label}_cleared_exact_mean_KL_family_guards"
    if refit_no_op:
        return f"{label}_zero_refit_was_no_op"
    if fit_derivative >= 0.0:
        return f"{label}_zero_nonnegative_fit_predicted_derivative"
    return f"{label}_zero_fallback"


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64DualTuneSelection:
    held_family_id: str
    refit_artifact_sha256: str
    tune_example_artifact_sha256s: tuple[str, ...]
    mean_family_mean_teacher_kl_by_alpha: Tensor = field(repr=False)
    mean_family_half_mean_squared_teacher_kl_by_alpha: Tensor = field(repr=False)
    reverse_family_mean_teacher_kl_by_beta: Tensor = field(repr=False)
    reverse_family_half_mean_squared_teacher_kl_by_beta: Tensor = field(repr=False)
    mean_aggregate_mean_teacher_kl_by_alpha: tuple[float, ...]
    mean_aggregate_half_mean_squared_teacher_kl_by_alpha: tuple[float, ...]
    reverse_aggregate_mean_teacher_kl_by_beta: tuple[float, ...]
    reverse_aggregate_half_mean_squared_teacher_kl_by_beta: tuple[float, ...]
    mean_worst_family_ratio_by_alpha: tuple[float, ...]
    reverse_worst_family_ratio_by_beta: tuple[float, ...]
    mean_family_improved_or_equal_count_by_alpha: tuple[int, ...]
    reverse_family_improved_or_equal_count_by_beta: tuple[int, ...]
    minimum_required_mean_KL_improvement: float
    selected_mean_alpha: float
    selected_reverse_beta: float
    mean_proposed_gains: Tensor = field(repr=False)
    residual_proposed_gains: Tensor = field(repr=False)
    mean_fit_predicted_derivative: float
    residual_fit_predicted_derivative: float
    mean_refit_was_no_op: bool
    residual_refit_was_no_op: bool
    selected_mean_gains: Tensor = field(repr=False)
    selected_reverse_gains: Tensor = field(repr=False)
    mean_selection_reason: str
    reverse_selection_reason: str
    reverse_diagnostic_only: bool
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        alphas = _mean_alphas()
        betas = _reverse_betas()
        mean_family = _float64(
            self.mean_family_mean_teacher_kl_by_alpha,
            label="mean-arm family mean-KL ledger",
            ndim=2,
        )
        mean_squared = _float64(
            self.mean_family_half_mean_squared_teacher_kl_by_alpha,
            label="mean-arm family squared-KL ledger",
            ndim=2,
        )
        reverse_family = _float64(
            self.reverse_family_mean_teacher_kl_by_beta,
            label="reverse-arm family mean-KL ledger",
            ndim=2,
        )
        reverse_squared = _float64(
            self.reverse_family_half_mean_squared_teacher_kl_by_beta,
            label="reverse-arm family squared-KL ledger",
            ndim=2,
        )
        if (
            mean_family.shape != (len(alphas), _EXPECTED_TRAINING_FAMILIES)
            or mean_squared.shape != mean_family.shape
            or reverse_family.shape != (len(betas), _EXPECTED_TRAINING_FAMILIES)
            or reverse_squared.shape != reverse_family.shape
            or bool((mean_family < 0.0).any())
            or bool((mean_squared < 0.0).any())
            or bool((reverse_family < 0.0).any())
            or bool((reverse_squared < 0.0).any())
            or not torch.equal(mean_family[0], reverse_family[0])
            or not torch.equal(mean_squared[0], reverse_squared[0])
        ):
            raise ValueError("candidate v4 dual tune ledgers are invalid")

        expected_mean_aggregate, expected_mean_ratios, expected_mean_counts = (
            _ledger_summary(mean_family)
        )
        (
            expected_reverse_aggregate,
            expected_reverse_ratios,
            expected_reverse_counts,
        ) = _ledger_summary(reverse_family)
        expected_mean_squared = tuple(
            math.fsum(float(value) for value in row) / len(row)
            for row in mean_squared
        )
        expected_reverse_squared = tuple(
            math.fsum(float(value) for value in row) / len(row)
            for row in reverse_squared
        )
        mean_fit = float(self.mean_fit_predicted_derivative)
        residual_fit = float(self.residual_fit_predicted_derivative)
        expected_alpha = _largest_eligible_step(
            grid=alphas,
            family_means=mean_family,
            aggregate_means=expected_mean_aggregate,
            family_counts=expected_mean_counts,
            fit_derivative=mean_fit,
            refit_no_op=self.mean_refit_was_no_op,
        )
        expected_beta = _largest_eligible_step(
            grid=betas,
            family_means=reverse_family,
            aggregate_means=expected_reverse_aggregate,
            family_counts=expected_reverse_counts,
            fit_derivative=residual_fit,
            refit_no_op=self.residual_refit_was_no_op,
        )
        mean_proposed = _float64(
            self.mean_proposed_gains, label="mean proposed gains", ndim=1
        )
        residual_proposed = _float64(
            self.residual_proposed_gains,
            label="residual proposed gains",
            ndim=1,
        )
        selected_mean = _float64(
            self.selected_mean_gains, label="selected mean gains", ndim=1
        )
        selected_reverse = _float64(
            self.selected_reverse_gains, label="selected reverse gains", ndim=1
        )
        expected_selected_mean = (
            1.0 + expected_alpha * (mean_proposed - 1.0)
        ).contiguous()
        expected_selected_reverse = (
            1.0 - expected_beta * (residual_proposed - 1.0)
        ).contiguous()
        expected_minimum = max(
            1.0e-8, 1.0e-4 * expected_mean_aggregate[0]
        )
        expected_mean_reason = _selection_reason(
            step=expected_alpha,
            refit_no_op=self.mean_refit_was_no_op,
            fit_derivative=mean_fit,
            label="mean_alpha",
        )
        expected_reverse_reason = _selection_reason(
            step=expected_beta,
            refit_no_op=self.residual_refit_was_no_op,
            fit_derivative=residual_fit,
            label="reverse_beta",
        )
        tuple_checks = (
            (
                tuple(
                    float(value)
                    for value in self.mean_aggregate_mean_teacher_kl_by_alpha
                ),
                expected_mean_aggregate,
            ),
            (
                tuple(
                    float(value)
                    for value in (
                        self.mean_aggregate_half_mean_squared_teacher_kl_by_alpha
                    )
                ),
                expected_mean_squared,
            ),
            (
                tuple(
                    float(value)
                    for value in self.reverse_aggregate_mean_teacher_kl_by_beta
                ),
                expected_reverse_aggregate,
            ),
            (
                tuple(
                    float(value)
                    for value in (
                        self.reverse_aggregate_half_mean_squared_teacher_kl_by_beta
                    )
                ),
                expected_reverse_squared,
            ),
            (
                tuple(float(value) for value in self.mean_worst_family_ratio_by_alpha),
                expected_mean_ratios,
            ),
            (
                tuple(
                    float(value)
                    for value in self.reverse_worst_family_ratio_by_beta
                ),
                expected_reverse_ratios,
            ),
            (
                tuple(self.mean_family_improved_or_equal_count_by_alpha),
                expected_mean_counts,
            ),
            (
                tuple(self.reverse_family_improved_or_equal_count_by_beta),
                expected_reverse_counts,
            ),
        )
        if (
            len(self.tune_example_artifact_sha256s)
            != _EXPECTED_TRAINING_FAMILIES
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.tune_example_artifact_sha256s
            )
            or any(actual != expected for actual, expected in tuple_checks)
            or float(self.minimum_required_mean_KL_improvement)
            != expected_minimum
            or float(self.selected_mean_alpha) != expected_alpha
            or float(self.selected_reverse_beta) != expected_beta
            or mean_proposed.shape != (CANDIDATE_GAIN_RANK,)
            or residual_proposed.shape != (CANDIDATE_GAIN_RANK,)
            or selected_mean.shape != (CANDIDATE_GAIN_RANK,)
            or selected_reverse.shape != (CANDIDATE_GAIN_RANK,)
            or bool((mean_proposed < _GAIN_MINIMUM).any())
            or bool((mean_proposed > _GAIN_MAXIMUM).any())
            or bool((residual_proposed < _GAIN_MINIMUM).any())
            or bool((residual_proposed > _GAIN_MAXIMUM).any())
            or not torch.allclose(
                selected_mean, expected_selected_mean, rtol=1.0e-12, atol=1.0e-12
            )
            or not torch.allclose(
                selected_reverse,
                expected_selected_reverse,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            or bool((selected_mean < _GAIN_MINIMUM).any())
            or bool((selected_mean > _GAIN_MAXIMUM).any())
            or bool((selected_reverse < _GAIN_MINIMUM).any())
            or bool((selected_reverse > _GAIN_MAXIMUM).any())
            or self.mean_selection_reason != expected_mean_reason
            or self.reverse_selection_reason != expected_reverse_reason
            or type(self.mean_refit_was_no_op) is not bool
            or type(self.residual_refit_was_no_op) is not bool
            or type(self.reverse_diagnostic_only) is not bool
            or self.reverse_diagnostic_only is not True
            or (self.mean_refit_was_no_op and mean_fit != 0.0)
            or (self.residual_refit_was_no_op and residual_fit != 0.0)
            or self.mean_refit_was_no_op is not (mean_fit >= 0.0)
            or self.residual_refit_was_no_op is not (residual_fit >= 0.0)
            or (
                self.mean_refit_was_no_op
                and not torch.equal(mean_proposed, torch.ones_like(mean_proposed))
            )
            or (
                self.residual_refit_was_no_op
                and not torch.equal(
                    residual_proposed, torch.ones_like(residual_proposed)
                )
            )
            or not math.isfinite(mean_fit)
            or not math.isfinite(residual_fit)
        ):
            raise ValueError("candidate v4 dual tune selection payload is invalid")

        object.__setattr__(
            self,
            "held_family_id",
            _identifier(self.held_family_id, label="held_family_id"),
        )
        object.__setattr__(
            self,
            "refit_artifact_sha256",
            _require_sha256(self.refit_artifact_sha256, label="candidate v4 refit"),
        )
        object.__setattr__(
            self,
            "tune_example_artifact_sha256s",
            tuple(self.tune_example_artifact_sha256s),
        )
        for name, value in (
            ("mean_family_mean_teacher_kl_by_alpha", mean_family),
            ("mean_family_half_mean_squared_teacher_kl_by_alpha", mean_squared),
            ("reverse_family_mean_teacher_kl_by_beta", reverse_family),
            (
                "reverse_family_half_mean_squared_teacher_kl_by_beta",
                reverse_squared,
            ),
            ("mean_proposed_gains", mean_proposed),
            ("residual_proposed_gains", residual_proposed),
            ("selected_mean_gains", selected_mean),
            ("selected_reverse_gains", selected_reverse),
        ):
            object.__setattr__(self, name, value)
        for name, value in (
            (
                "mean_aggregate_mean_teacher_kl_by_alpha",
                expected_mean_aggregate,
            ),
            (
                "mean_aggregate_half_mean_squared_teacher_kl_by_alpha",
                expected_mean_squared,
            ),
            (
                "reverse_aggregate_mean_teacher_kl_by_beta",
                expected_reverse_aggregate,
            ),
            (
                "reverse_aggregate_half_mean_squared_teacher_kl_by_beta",
                expected_reverse_squared,
            ),
            ("mean_worst_family_ratio_by_alpha", expected_mean_ratios),
            ("reverse_worst_family_ratio_by_beta", expected_reverse_ratios),
            (
                "mean_family_improved_or_equal_count_by_alpha",
                expected_mean_counts,
            ),
            (
                "reverse_family_improved_or_equal_count_by_beta",
                expected_reverse_counts,
            ),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "minimum_required_mean_KL_improvement", expected_minimum
        )
        object.__setattr__(self, "selected_mean_alpha", expected_alpha)
        object.__setattr__(self, "selected_reverse_beta", expected_beta)
        object.__setattr__(self, "mean_fit_predicted_derivative", mean_fit)
        object.__setattr__(
            self, "residual_fit_predicted_derivative", residual_fit
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(
                _TUNE_SELECTION_DOMAIN, self.metadata(include_artifact=False)
            ),
        )
        self.validate_integrity()

    def selected_mean_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.selected_mean_gains.clone().contiguous()

    def selected_reverse_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.selected_reverse_gains.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "tune_example_artifact_sha256s": self.tune_example_artifact_sha256s,
            "mean_alpha_hex_grid": tuple(
                alpha.hex() for alpha in _mean_alphas()
            ),
            "reverse_beta_hex_grid": tuple(
                beta.hex() for beta in _reverse_betas()
            ),
            "mean_family_mean_teacher_kl_by_alpha_sha256": _tensor_sha256(
                self.mean_family_mean_teacher_kl_by_alpha
            ),
            "mean_family_half_mean_squared_teacher_kl_by_alpha_sha256": (
                _tensor_sha256(
                    self.mean_family_half_mean_squared_teacher_kl_by_alpha
                )
            ),
            "reverse_family_mean_teacher_kl_by_beta_sha256": _tensor_sha256(
                self.reverse_family_mean_teacher_kl_by_beta
            ),
            "reverse_family_half_mean_squared_teacher_kl_by_beta_sha256": (
                _tensor_sha256(
                    self.reverse_family_half_mean_squared_teacher_kl_by_beta
                )
            ),
            "mean_aggregate_mean_teacher_kl_by_alpha": (
                self.mean_aggregate_mean_teacher_kl_by_alpha
            ),
            "mean_aggregate_half_mean_squared_teacher_kl_by_alpha": (
                self.mean_aggregate_half_mean_squared_teacher_kl_by_alpha
            ),
            "reverse_aggregate_mean_teacher_kl_by_beta": (
                self.reverse_aggregate_mean_teacher_kl_by_beta
            ),
            "reverse_aggregate_half_mean_squared_teacher_kl_by_beta": (
                self.reverse_aggregate_half_mean_squared_teacher_kl_by_beta
            ),
            "mean_worst_family_ratio_by_alpha": (
                self.mean_worst_family_ratio_by_alpha
            ),
            "reverse_worst_family_ratio_by_beta": (
                self.reverse_worst_family_ratio_by_beta
            ),
            "mean_family_improved_or_equal_count_by_alpha": (
                self.mean_family_improved_or_equal_count_by_alpha
            ),
            "reverse_family_improved_or_equal_count_by_beta": (
                self.reverse_family_improved_or_equal_count_by_beta
            ),
            "minimum_required_mean_KL_improvement": (
                self.minimum_required_mean_KL_improvement
            ),
            "squared_KL_is_selection_gate": False,
            "mean_proposed_gains_sha256": _tensor_sha256(
                self.mean_proposed_gains
            ),
            "residual_proposed_gains_sha256": _tensor_sha256(
                self.residual_proposed_gains
            ),
            "mean_fit_predicted_derivative": self.mean_fit_predicted_derivative,
            "residual_fit_predicted_derivative": (
                self.residual_fit_predicted_derivative
            ),
            "mean_refit_was_no_op": self.mean_refit_was_no_op,
            "residual_refit_was_no_op": self.residual_refit_was_no_op,
            "selected_mean_alpha_hex": self.selected_mean_alpha.hex(),
            "selected_reverse_beta_hex": self.selected_reverse_beta.hex(),
            "selected_mean_gains_sha256": _tensor_sha256(
                self.selected_mean_gains
            ),
            "selected_reverse_gains_sha256": _tensor_sha256(
                self.selected_reverse_gains
            ),
            "mean_selection_reason": self.mean_selection_reason,
            "reverse_selection_reason": self.reverse_selection_reason,
            "reverse_diagnostic_only": self.reverse_diagnostic_only,
            "reverse_arm_can_authorize_primary_refit": False,
            "alpha_zero_and_beta_zero_share_one_unit_execution": True,
            "held_family_used_for_tune": False,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _sha256(
                _TUNE_SELECTION_DOMAIN, self.metadata(include_artifact=False)
            )
            != _require_sha256(
                self.artifact_sha256, label="candidate v4 tune selection"
            )
        ):
            raise RuntimeError("candidate v4 dual tune selection payload drifted")


def select_candidate_conditioned_k64_dual_tune_steps(
    refit: CandidateConditionedK64MeanKLRefit,
    tune_examples: Sequence[CandidateConditionedK64DualTuneExample],
) -> CandidateConditionedK64DualTuneSelection:
    """Select mean alpha and diagnostic reverse beta on exact tune mean KL."""

    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("refit must be a candidate v4 mean-KL refit")
    refit.validate_integrity()
    values = tuple(sorted(tuple(tune_examples), key=lambda value: value.example_id))
    if any(
        not isinstance(value, CandidateConditionedK64DualTuneExample)
        for value in values
    ):
        raise TypeError("v4 tune examples must be typed records")
    for value in values:
        value.validate_integrity()
    families = tuple(sorted({value.family_id for value in values}))
    tune_ids = {value.example_id for value in values}
    if (
        len(values) != _EXPECTED_TRAINING_FAMILIES
        or families != refit.training_family_ids
        or len(tune_ids) != len(values)
        or not tune_ids.isdisjoint(refit.training_example_ids)
    ):
        raise ValueError(
            "v4 tune examples must be disjoint and match seven training families"
        )

    alphas = _mean_alphas()
    betas = _reverse_betas()
    mean_means = tuple(
        tuple(float(value.mean_token_kl(alpha).mean()) for value in values)
        for alpha in alphas
    )
    mean_squared = tuple(
        tuple(
            0.5 * float(value.mean_token_kl(alpha).square().mean())
            for value in values
        )
        for alpha in alphas
    )
    reverse_means = tuple(
        tuple(float(value.reverse_token_kl(beta).mean()) for value in values)
        for beta in betas
    )
    reverse_squared = tuple(
        tuple(
            0.5 * float(value.reverse_token_kl(beta).square().mean())
            for value in values
        )
        for beta in betas
    )
    mean_family = torch.tensor(mean_means, dtype=torch.float64).contiguous()
    reverse_family = torch.tensor(reverse_means, dtype=torch.float64).contiguous()
    mean_squared_tensor = torch.tensor(
        mean_squared, dtype=torch.float64
    ).contiguous()
    reverse_squared_tensor = torch.tensor(
        reverse_squared, dtype=torch.float64
    ).contiguous()
    mean_aggregate, mean_ratios, mean_counts = _ledger_summary(mean_family)
    reverse_aggregate, reverse_ratios, reverse_counts = _ledger_summary(
        reverse_family
    )
    mean_squared_aggregate = tuple(
        math.fsum(row) / len(row) for row in mean_squared
    )
    reverse_squared_aggregate = tuple(
        math.fsum(row) / len(row) for row in reverse_squared
    )
    alpha = _largest_eligible_step(
        grid=alphas,
        family_means=mean_family,
        aggregate_means=mean_aggregate,
        family_counts=mean_counts,
        fit_derivative=refit.mean_predicted_derivative,
        refit_no_op=refit.mean_no_op,
    )
    beta = _largest_eligible_step(
        grid=betas,
        family_means=reverse_family,
        aggregate_means=reverse_aggregate,
        family_counts=reverse_counts,
        fit_derivative=refit.residual_predicted_derivative,
        refit_no_op=refit.residual_no_op,
    )
    mean_proposed = refit.mean_proposed_gains_tensor()
    residual_proposed = refit.residual_proposed_gains_tensor()
    selected_mean = (1.0 + alpha * (mean_proposed - 1.0)).contiguous()
    selected_reverse = (
        1.0 - beta * (residual_proposed - 1.0)
    ).contiguous()
    return CandidateConditionedK64DualTuneSelection(
        held_family_id=refit.held_family_id,
        refit_artifact_sha256=refit.artifact_sha256,
        tune_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        mean_family_mean_teacher_kl_by_alpha=mean_family,
        mean_family_half_mean_squared_teacher_kl_by_alpha=mean_squared_tensor,
        reverse_family_mean_teacher_kl_by_beta=reverse_family,
        reverse_family_half_mean_squared_teacher_kl_by_beta=(
            reverse_squared_tensor
        ),
        mean_aggregate_mean_teacher_kl_by_alpha=mean_aggregate,
        mean_aggregate_half_mean_squared_teacher_kl_by_alpha=(
            mean_squared_aggregate
        ),
        reverse_aggregate_mean_teacher_kl_by_beta=reverse_aggregate,
        reverse_aggregate_half_mean_squared_teacher_kl_by_beta=(
            reverse_squared_aggregate
        ),
        mean_worst_family_ratio_by_alpha=mean_ratios,
        reverse_worst_family_ratio_by_beta=reverse_ratios,
        mean_family_improved_or_equal_count_by_alpha=mean_counts,
        reverse_family_improved_or_equal_count_by_beta=reverse_counts,
        minimum_required_mean_KL_improvement=max(
            1.0e-8, 1.0e-4 * mean_aggregate[0]
        ),
        selected_mean_alpha=alpha,
        selected_reverse_beta=beta,
        mean_proposed_gains=mean_proposed,
        residual_proposed_gains=residual_proposed,
        mean_fit_predicted_derivative=refit.mean_predicted_derivative,
        residual_fit_predicted_derivative=refit.residual_predicted_derivative,
        mean_refit_was_no_op=refit.mean_no_op,
        residual_refit_was_no_op=refit.residual_no_op,
        selected_mean_gains=selected_mean,
        selected_reverse_gains=selected_reverse,
        mean_selection_reason=_selection_reason(
            step=alpha,
            refit_no_op=refit.mean_no_op,
            fit_derivative=refit.mean_predicted_derivative,
            label="mean_alpha",
        ),
        reverse_selection_reason=_selection_reason(
            step=beta,
            refit_no_op=refit.residual_no_op,
            fit_derivative=refit.residual_predicted_derivative,
            label="reverse_beta",
        ),
        reverse_diagnostic_only=True,
    )
