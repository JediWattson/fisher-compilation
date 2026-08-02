"""Pure V7 joint scalar-plus-state K64 gain-field capacity screen.

The frozen V4 direction and V5 ``+1/64`` carrier are unchanged.  V7 only
adds one affine row logit over the four V6 state coordinates::

    ell_r = u + z_r @ w
    a_r = 1 + tanh(ell_r)
    g_rk = 1 + (1/64) * a_r * delta_k

The five coefficients are fit together from the same authenticated V6 row
bank.  This module performs no finite candidate execution and cannot select
or authorize a serving candidate.
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
from fisher_graph.complete_h4_tail_candidate_state_gain_field import (
    STATE_DESIGN_RANK_RELATIVE_TOLERANCE,
    STATE_FEATURE_RANK,
    STATE_FEATURE_SCALE_MINIMUM,
    STATE_GAIN_BASE_STEP,
    STATE_GAIN_LOGIT_RMS_TRUST,
    STATE_RESIDUAL_FISHER_MINIMUM,
    STATE_STANDARDIZED_CONDITION_MAXIMUM,
    CandidateConditionedK64StateFeatureCodec,
    CandidateConditionedK64StateGainGradientExample,
    CandidateConditionedK64StaticAmplitudeControlFit,
    fit_candidate_conditioned_k64_static_amplitude_control,
    _tensor_sha256 as _v6_tensor_sha256,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
)


__all__ = [
    "JOINT_STATE_GAIN_PARAMETER_COUNT",
    "CandidateConditionedK64JointStateGainFieldFit",
    "CandidateConditionedK64JointInnerFamilyAnalyticRecord",
    "CandidateConditionedK64JointStateGainFoldAnalyticRecord",
    "CandidateConditionedK64JointStateGainAnalyticScreen",
    "candidate_conditioned_k64_joint_tangent_design",
    "fit_candidate_conditioned_k64_joint_state_gain_field",
    "fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control",
    "build_candidate_conditioned_k64_joint_inner_family_analytic_record",
    "build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record",
    "screen_candidate_conditioned_k64_joint_state_gain_capacity",
]


JOINT_STATE_GAIN_PARAMETER_COUNT = 1 + STATE_FEATURE_RANK
_OUTER_FAMILY_COUNT = 8
_OUTER_TRAINING_FAMILY_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_DAMPING_FRACTION = 0.1
_DAMPING_FLOOR = 1.0e-12
_TINY = torch.finfo(torch.float64).tiny
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_TENSOR_DOMAIN = b"fisher-graph:candidate-k64-joint-state-tensor:v7\0"
_FIT_DOMAIN = b"fisher-graph:candidate-k64-joint-state-fit:v7\0"
_INNER_DOMAIN = b"fisher-graph:candidate-k64-joint-state-inner:v7\0"
_FOLD_DOMAIN = b"fisher-graph:candidate-k64-joint-state-fold:v7\0"
_SCREEN_DOMAIN = b"fisher-graph:candidate-k64-joint-state-screen:v7\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
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
        or any(int(width) <= 0 for width in value.shape)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


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


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _family_equal_mean(values: Sequence[Tensor]) -> Tensor:
    if not values:
        raise ValueError("family-equal mean requires evidence")
    return torch.stack(tuple(value.mean(dim=0) for value in values)).mean(dim=0)


def _family_equal_second_moment(values: Sequence[Tensor]) -> Tensor:
    if not values:
        raise ValueError("family-equal second moment requires evidence")
    return torch.stack(
        tuple((value.T @ value) / value.shape[0] for value in values)
    ).mean(dim=0).contiguous()


def _is_positive_semidefinite(value: Tensor) -> bool:
    symmetric = 0.5 * (value + value.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    scale = max(float(torch.linalg.matrix_norm(symmetric, ord=2)), _TINY)
    return float(eigenvalues[0]) >= -STATE_DESIGN_RANK_RELATIVE_TOLERANCE * scale


def _rank_condition_from_gram(value: Tensor) -> tuple[int, float]:
    symmetric = 0.5 * (value + value.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    maximum = max(float(eigenvalues[-1]), 0.0)
    tolerance = STATE_DESIGN_RANK_RELATIVE_TOLERANCE * maximum
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.numel())
    if rank != value.shape[0] or positive.numel() == 0:
        return rank, torch.finfo(torch.float64).max
    return rank, math.sqrt(maximum / float(positive[0]))


def _column_standardized_gram(value: Tensor) -> Tensor:
    diagonal = torch.clamp(value.diagonal(), min=0.0)
    inverse = torch.where(
        diagonal > 0.0,
        diagonal.rsqrt(),
        torch.zeros_like(diagonal),
    )
    return (inverse[:, None] * value * inverse[None, :]).contiguous()


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = float(
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )
    if denominator == 0.0:
        return 0.0
    value = float(torch.dot(left, right)) / denominator
    return max(-1.0, min(1.0, value))


def _residual_conditional_state_fisher_fraction(joint_fisher: Tensor) -> float:
    state = 0.5 * (joint_fisher[1:, 1:] + joint_fisher[1:, 1:].T)
    cross = joint_fisher[1:, 0]
    scalar = float(joint_fisher[0, 0])
    total = float(torch.trace(state))
    if total <= 0.0:
        return 0.0
    residual = state - torch.outer(cross, cross) / max(scalar, _TINY)
    residual = 0.5 * (residual + residual.T)
    eigenvalues = torch.linalg.eigvalsh(residual)
    scale = max(float(torch.linalg.matrix_norm(state, ord=2)), _TINY)
    tolerance = STATE_DESIGN_RANK_RELATIVE_TOLERANCE * scale
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("joint conditional state Fisher residual is non-PSD")
    fraction = float(torch.trace(residual)) / total
    if fraction < -STATE_DESIGN_RANK_RELATIVE_TOLERANCE or fraction > (
        1.0 + STATE_DESIGN_RANK_RELATIVE_TOLERANCE
    ):
        raise ValueError("joint conditional state Fisher fraction is invalid")
    return max(0.0, min(1.0, fraction))


def candidate_conditioned_k64_joint_tangent_design(
    example: CandidateConditionedK64StateGainGradientExample,
) -> Tensor:
    """Return the token-by-five direct joint tangent design ``[s, J]``."""

    if not isinstance(example, CandidateConditionedK64StateGainGradientExample):
        raise TypeError("joint tangent design requires a V6 gradient record")
    example.validate_integrity()
    scalar = example.static_tangent_tensor()
    state = example.state_tangent_design_tensor()
    return torch.cat((scalar[:, None], state), dim=1).contiguous()


def _joint_logit_rows(
    example: CandidateConditionedK64StateGainGradientExample,
) -> Tensor:
    ones = torch.ones(
        example.standardized_state_features.shape[0],
        1,
        dtype=torch.float64,
    )
    return torch.cat((ones, example.standardized_state_features), dim=1).contiguous()


def _validated_examples(
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
    *,
    codec: CandidateConditionedK64StateFeatureCodec,
) -> tuple[CandidateConditionedK64StateGainGradientExample, ...]:
    supplied = tuple(examples)
    if any(
        not isinstance(value, CandidateConditionedK64StateGainGradientExample)
        for value in supplied
    ):
        raise TypeError("joint fit evidence must be V6 gradient records")
    values = tuple(sorted(supplied, key=lambda value: value.example_id))
    for value in values:
        value.validate_integrity()
    families = tuple(sorted(value.family_id for value in values))
    ids = tuple(value.example_id for value in values)
    if (
        len(values) not in (_INNER_TRAINING_FAMILY_COUNT, _OUTER_TRAINING_FAMILY_COUNT)
        or len(set(families)) != len(values)
        or len(set(ids)) != len(values)
        or families != codec.training_family_ids
        or ids != codec.training_example_ids
        or any(value.codec_artifact_sha256 != codec.artifact_sha256 for value in values)
    ):
        raise ValueError("joint fit evidence differs from its feature codec")
    return values


def _joint_statistics(
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
) -> tuple[Tensor, Tensor, Tensor]:
    state_designs = tuple(
        value.state_tangent_design_tensor() for value in examples
    )
    static_designs = tuple(value.static_tangent_tensor() for value in examples)
    state_features = tuple(
        value.standardized_state_features for value in examples
    )
    state_gradient = _family_equal_mean(state_designs)
    static_gradient = float(
        _family_equal_mean(
            tuple(value[:, None] for value in static_designs)
        )[0]
    )
    state_fisher = _family_equal_second_moment(state_designs)
    static_fisher = float(
        torch.stack(
            tuple(value.square().mean() for value in static_designs)
        ).mean()
    )
    cross = torch.stack(
        tuple(
            (state * scalar[:, None]).mean(dim=0)
            for state, scalar in zip(state_designs, static_designs)
        )
    ).mean(dim=0).contiguous()
    gradient = torch.cat(
        (torch.tensor([static_gradient], dtype=torch.float64), state_gradient)
    ).contiguous()
    fisher = torch.zeros(
        JOINT_STATE_GAIN_PARAMETER_COUNT,
        JOINT_STATE_GAIN_PARAMETER_COUNT,
        dtype=torch.float64,
    )
    fisher[0, 0] = static_fisher
    fisher[0, 1:] = cross
    fisher[1:, 0] = cross
    fisher[1:, 1:] = state_fisher
    state_mean = _family_equal_mean(state_features)
    state_second_moment = _family_equal_second_moment(state_features)
    logit = torch.zeros_like(fisher)
    logit[0, 0] = 1.0
    logit[0, 1:] = state_mean
    logit[1:, 0] = state_mean
    logit[1:, 1:] = state_second_moment
    return gradient, fisher.contiguous(), logit.contiguous()


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64JointStateGainFieldFit:
    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    refit_artifact_sha256: str
    codec_artifact_sha256: str
    ordered_directions_codec_sha256: str
    ordered_directions_refit_sha256: str
    feature_scale: Tensor = field(repr=False)
    mean_gain_delta: Tensor = field(repr=False)
    mean_gradient: Tensor = field(repr=False)
    joint_fisher_gram: Tensor = field(repr=False)
    joint_logit_second_moment: Tensor = field(repr=False)
    raw_parameter: Tensor = field(init=False, repr=False)
    applied_parameter: Tensor = field(init=False, repr=False)
    damping: float = field(init=False)
    raw_logit_rms: float = field(init=False)
    applied_logit_rms: float = field(init=False)
    trust_scale: float = field(init=False)
    predicted_incremental_derivative: float = field(init=False)
    no_op: bool = field(init=False)
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
        scale = _float64(self.feature_scale, label="feature scale", ndim=1)
        delta = _float64(self.mean_gain_delta, label="mean gain delta", ndim=1)
        gradient = _float64(self.mean_gradient, label="joint mean gradient", ndim=1)
        fisher = _float64(self.joint_fisher_gram, label="joint Fisher Gram", ndim=2)
        logit = _float64(
            self.joint_logit_second_moment,
            label="joint logit second moment",
            ndim=2,
        )
        if (
            len(families) not in (_INNER_TRAINING_FAMILY_COUNT, _OUTER_TRAINING_FAMILY_COUNT)
            or families != tuple(sorted(set(families)))
            or held in families
            or len(examples) != len(families)
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(families)
            or scale.shape != (STATE_FEATURE_RANK,)
            or bool((scale <= STATE_FEATURE_SCALE_MINIMUM).any())
            or delta.shape != (CANDIDATE_GAIN_RANK,)
            or gradient.shape != (JOINT_STATE_GAIN_PARAMETER_COUNT,)
            or fisher.shape != (
                JOINT_STATE_GAIN_PARAMETER_COUNT,
                JOINT_STATE_GAIN_PARAMETER_COUNT,
            )
            or logit.shape != fisher.shape
            or bool((1.0 + delta < 0.0).any())
            or bool((1.0 + delta > 1.5).any())
            or not torch.allclose(fisher, fisher.T, rtol=0.0, atol=1.0e-12)
            or not torch.allclose(logit, logit.T, rtol=0.0, atol=1.0e-12)
            or not _is_positive_semidefinite(fisher)
            or not _is_positive_semidefinite(logit)
        ):
            raise ValueError("joint state gain field fit payload is invalid")
        _residual_conditional_state_fisher_fraction(fisher)
        damping = max(
            _DAMPING_FRACTION * float(fisher.diagonal().median()),
            _DAMPING_FLOOR,
        )
        system = fisher + damping * torch.eye(
            JOINT_STATE_GAIN_PARAMETER_COUNT, dtype=torch.float64
        )
        if bool((gradient == 0.0).all()):
            raw = torch.zeros_like(gradient)
        else:
            numeric_scale = max(
                float(system.abs().max()),
                float(gradient.abs().max()),
                _TINY,
            )
            try:
                raw = torch.linalg.solve(
                    system / numeric_scale,
                    -gradient / numeric_scale,
                ).contiguous()
            except RuntimeError as error:
                raise RuntimeError("joint state gain OPG solve failed") from error
        if not bool(torch.isfinite(raw).all()):
            raise RuntimeError("joint state gain OPG solve was nonfinite")
        raw_rms = math.sqrt(max(float(raw @ logit @ raw), 0.0))
        trust = min(
            1.0,
            STATE_GAIN_LOGIT_RMS_TRUST / max(raw_rms, _TINY),
        )
        applied = (trust * raw).contiguous()
        derivative = float(torch.dot(gradient, applied))
        no_op = derivative >= 0.0
        if no_op:
            applied = torch.zeros_like(applied)
            derivative = 0.0
        applied_rms = math.sqrt(max(float(applied @ logit @ applied), 0.0))
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(
            self,
            "refit_artifact_sha256",
            _require_sha256(self.refit_artifact_sha256, label="V4 refit"),
        )
        object.__setattr__(
            self,
            "codec_artifact_sha256",
            _require_sha256(self.codec_artifact_sha256, label="state codec"),
        )
        object.__setattr__(
            self,
            "ordered_directions_codec_sha256",
            _require_sha256(
                self.ordered_directions_codec_sha256,
                label="V6 codec directions",
            ),
        )
        object.__setattr__(
            self,
            "ordered_directions_refit_sha256",
            _require_sha256(
                self.ordered_directions_refit_sha256,
                label="V4 refit directions",
            ),
        )
        for name, value in (
            ("feature_scale", scale),
            ("mean_gain_delta", delta),
            ("mean_gradient", gradient),
            ("joint_fisher_gram", fisher),
            ("joint_logit_second_moment", logit),
            ("raw_parameter", raw),
            ("applied_parameter", applied),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "raw_logit_rms", raw_rms)
        object.__setattr__(self, "applied_logit_rms", applied_rms)
        object.__setattr__(self, "trust_scale", trust)
        object.__setattr__(self, "predicted_incremental_derivative", derivative)
        object.__setattr__(self, "no_op", no_op)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def applied_intercept(self) -> float:
        return float(self.applied_parameter[0])

    @property
    def applied_state_weight(self) -> Tensor:
        self.validate_integrity()
        return self.applied_parameter[1:].clone().contiguous()

    @property
    def augmented_feature_rank(self) -> int:
        return _rank_condition_from_gram(
            _column_standardized_gram(self.joint_logit_second_moment)
        )[0]

    @property
    def augmented_feature_condition(self) -> float:
        return _rank_condition_from_gram(
            _column_standardized_gram(self.joint_logit_second_moment)
        )[1]

    @property
    def standardized_design_rank(self) -> int:
        return _rank_condition_from_gram(
            _column_standardized_gram(self.joint_fisher_gram)
        )[0]

    @property
    def standardized_design_condition(self) -> float:
        return _rank_condition_from_gram(
            _column_standardized_gram(self.joint_fisher_gram)
        )[1]

    @property
    def residual_conditional_state_fisher_fraction(self) -> float:
        return _residual_conditional_state_fisher_fraction(
            self.joint_fisher_gram
        )

    def raw_state_slope_tensor(self) -> Tensor:
        self.validate_integrity()
        return (self.applied_parameter[1:] / self.feature_scale).contiguous()

    def row_logits_tensor(self, standardized_features: Tensor) -> Tensor:
        features = _float64(
            standardized_features,
            label="runtime standardized state features",
            ndim=2,
        )
        if features.shape[1] != STATE_FEATURE_RANK:
            raise ValueError("runtime state feature width differs")
        self.validate_integrity()
        return (
            self.applied_parameter[0]
            + features @ self.applied_parameter[1:]
        ).contiguous()

    def row_amplitudes_tensor(self, standardized_features: Tensor) -> Tensor:
        return (1.0 + torch.tanh(self.row_logits_tensor(standardized_features))).contiguous()

    def row_gains_tensor(self, standardized_features: Tensor) -> Tensor:
        amplitude = self.row_amplitudes_tensor(standardized_features)
        return (
            1.0
            + STATE_GAIN_BASE_STEP
            * amplitude[:, None]
            * self.mean_gain_delta[None, :]
        ).contiguous()

    def static_plus_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return (1.0 + STATE_GAIN_BASE_STEP * self.mean_gain_delta).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "training_example_artifact_sha256s": (
                self.training_example_artifact_sha256s
            ),
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "codec_artifact_sha256": self.codec_artifact_sha256,
            "ordered_directions_codec_sha256": (
                self.ordered_directions_codec_sha256
            ),
            "ordered_directions_refit_sha256": (
                self.ordered_directions_refit_sha256
            ),
            "ordered_directions_dual_hash_authenticated": True,
            "feature_scale_sha256": _tensor_sha256(self.feature_scale),
            "mean_gain_delta_sha256": _tensor_sha256(self.mean_gain_delta),
            "mean_gradient_sha256": _tensor_sha256(self.mean_gradient),
            "joint_fisher_gram_sha256": _tensor_sha256(self.joint_fisher_gram),
            "joint_logit_second_moment_sha256": _tensor_sha256(
                self.joint_logit_second_moment
            ),
            "raw_parameter_sha256": _tensor_sha256(self.raw_parameter),
            "applied_parameter_sha256": _tensor_sha256(self.applied_parameter),
            "raw_state_slope_sha256": _tensor_sha256(
                self.applied_parameter[1:] / self.feature_scale
            ),
            "applied_intercept": self.applied_intercept,
            "damping_fraction": _DAMPING_FRACTION,
            "damping_floor": _DAMPING_FLOOR,
            "damping": self.damping,
            "raw_logit_rms": self.raw_logit_rms,
            "applied_logit_rms": self.applied_logit_rms,
            "joint_logit_rms_trust": STATE_GAIN_LOGIT_RMS_TRUST,
            "trust_scale": self.trust_scale,
            "predicted_incremental_derivative": (
                self.predicted_incremental_derivative
            ),
            "no_op": self.no_op,
            "augmented_feature_rank": self.augmented_feature_rank,
            "augmented_feature_condition": self.augmented_feature_condition,
            "standardized_design_rank": self.standardized_design_rank,
            "standardized_design_condition": self.standardized_design_condition,
            "residual_conditional_state_fisher_fraction": (
                self.residual_conditional_state_fisher_fraction
            ),
            "learned_parameter_count": JOINT_STATE_GAIN_PARAMETER_COUNT,
            "bias_parameter_count": 1,
            "state_parameter_count": STATE_FEATURE_RANK,
            "codec_derived_float_count": 2 * STATE_FEATURE_RANK,
            "incremental_float_count_including_codec": (
                JOINT_STATE_GAIN_PARAMETER_COUNT + 2 * STATE_FEATURE_RANK
            ),
            "row_logit_formula": "u_plus_z_dot_w",
            "row_amplitude_formula": "one_plus_tanh_u_plus_z_dot_w",
            "zero_parameters_exactly_reproduce_v5_static_plus": True,
            "direction_reversal_possible": False,
            "fit_objective": "family_equal_uncentered_direct_joint5D_OPG",
            "finite_execution_authority_required_later": True,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        tensors = (
            self.feature_scale,
            self.mean_gain_delta,
            self.mean_gradient,
            self.joint_fisher_gram,
            self.joint_logit_second_moment,
            self.raw_parameter,
            self.applied_parameter,
        )
        if (
            any(
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or value.requires_grad
                or not value.is_contiguous()
                for value in tensors
            )
            or self.applied_parameter.shape != (JOINT_STATE_GAIN_PARAMETER_COUNT,)
            or self.applied_logit_rms > STATE_GAIN_LOGIT_RMS_TRUST + 1.0e-12
            or _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="joint state fit")
        ):
            raise RuntimeError("joint state gain field fit payload drifted")


def fit_candidate_conditioned_k64_joint_state_gain_field(
    refit: CandidateConditionedK64MeanKLRefit,
    codec: CandidateConditionedK64StateFeatureCodec,
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
    *,
    ordered_directions: Tensor,
) -> CandidateConditionedK64JointStateGainFieldFit:
    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("joint fit requires a V4 mean-KL refit")
    if not isinstance(codec, CandidateConditionedK64StateFeatureCodec):
        raise TypeError("joint fit requires a V6 state feature codec")
    refit.validate_integrity()
    codec.validate_integrity()
    values = _validated_examples(examples, codec=codec)
    if codec.held_family_id != refit.held_family_id:
        raise ValueError("joint codec and V4 refit held families differ")
    directions = _float64(
        ordered_directions,
        label="ordered K64 directions",
        ndim=2,
    )
    if (
        directions.shape[0] != CANDIDATE_GAIN_RANK
        or _v6_tensor_sha256(directions) != codec.ordered_directions_sha256
    ):
        raise ValueError("joint directions differ from the V6 feature codec")
    if _runtime_tensor_sha256(ordered_directions) != refit.ordered_directions_sha256:
        raise ValueError("joint directions differ from the V4 refit")
    gradient, fisher, logit = _joint_statistics(values)
    if not torch.equal(logit[1:, 1:], codec.standardized_feature_gram):
        raise ValueError("joint row features do not replay the fitted codec")
    return CandidateConditionedK64JointStateGainFieldFit(
        held_family_id=refit.held_family_id,
        training_family_ids=codec.training_family_ids,
        training_example_ids=codec.training_example_ids,
        training_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        refit_artifact_sha256=refit.artifact_sha256,
        codec_artifact_sha256=codec.artifact_sha256,
        ordered_directions_codec_sha256=codec.ordered_directions_sha256,
        ordered_directions_refit_sha256=refit.ordered_directions_sha256,
        feature_scale=codec.feature_scale,
        mean_gain_delta=refit.mean_proposed_gains_tensor() - 1.0,
        mean_gradient=gradient,
        joint_fisher_gram=fisher,
        joint_logit_second_moment=logit,
    )


def _scalar_control_matches_joint(
    joint: CandidateConditionedK64JointStateGainFieldFit,
    scalar: CandidateConditionedK64StaticAmplitudeControlFit,
) -> bool:
    return (
        scalar.held_family_id == joint.held_family_id
        and scalar.training_family_ids == joint.training_family_ids
        and scalar.training_example_ids == joint.training_example_ids
        and scalar.training_example_artifact_sha256s
        == joint.training_example_artifact_sha256s
        and scalar.refit_artifact_sha256 == joint.refit_artifact_sha256
        and torch.equal(scalar.mean_gain_delta, joint.mean_gain_delta)
        and scalar.mean_gradient == float(joint.mean_gradient[0])
        and scalar.fisher_energy == float(joint.joint_fisher_gram[0, 0])
    )


def fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control(
    refit: CandidateConditionedK64MeanKLRefit,
    codec: CandidateConditionedK64StateFeatureCodec,
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
    *,
    ordered_directions: Tensor,
) -> tuple[
    CandidateConditionedK64JointStateGainFieldFit,
    CandidateConditionedK64StaticAmplitudeControlFit,
]:
    joint = fit_candidate_conditioned_k64_joint_state_gain_field(
        refit,
        codec,
        examples,
        ordered_directions=ordered_directions,
    )
    scalar = fit_candidate_conditioned_k64_static_amplitude_control(
        refit, codec, examples
    )
    if not _scalar_control_matches_joint(joint, scalar):
        raise RuntimeError("V7 scalar control does not reproduce the V6 fit")
    return joint, scalar


def _fit_is_identifiable(
    value: CandidateConditionedK64JointStateGainFieldFit,
) -> bool:
    return (
        value.augmented_feature_rank == JOINT_STATE_GAIN_PARAMETER_COUNT
        and value.standardized_design_rank == JOINT_STATE_GAIN_PARAMETER_COUNT
        and value.augmented_feature_condition
        <= STATE_STANDARDIZED_CONDITION_MAXIMUM
        and value.standardized_design_condition
        <= STATE_STANDARDIZED_CONDITION_MAXIMUM
    )


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64JointInnerFamilyAnalyticRecord:
    full_joint_fit: CandidateConditionedK64JointStateGainFieldFit = field(
        repr=False
    )
    inner_joint_fit: CandidateConditionedK64JointStateGainFieldFit = field(
        repr=False
    )
    inner_scalar_control_fit: CandidateConditionedK64StaticAmplitudeControlFit = (
        field(repr=False)
    )
    held_example: CandidateConditionedK64StateGainGradientExample = field(
        repr=False
    )
    outer_held_family_id: str = field(init=False)
    inner_held_family_id: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.full_joint_fit,
            CandidateConditionedK64JointStateGainFieldFit,
        ) or not isinstance(
            self.inner_joint_fit,
            CandidateConditionedK64JointStateGainFieldFit,
        ):
            raise TypeError("joint inner record requires typed joint fits")
        if not isinstance(
            self.inner_scalar_control_fit,
            CandidateConditionedK64StaticAmplitudeControlFit,
        ):
            raise TypeError("joint inner record requires a V6 scalar control")
        if not isinstance(
            self.held_example,
            CandidateConditionedK64StateGainGradientExample,
        ):
            raise TypeError("joint inner record requires typed held evidence")
        self.full_joint_fit.validate_integrity()
        self.inner_joint_fit.validate_integrity()
        self.inner_scalar_control_fit.validate_integrity()
        self.held_example.validate_integrity()
        outer = self.full_joint_fit.held_family_id
        inner = self.held_example.family_id
        full_families = set(self.full_joint_fit.training_family_ids)
        inner_families = set(self.inner_joint_fit.training_family_ids)
        if (
            len(full_families) != _OUTER_TRAINING_FAMILY_COUNT
            or len(inner_families) != _INNER_TRAINING_FAMILY_COUNT
            or self.inner_joint_fit.held_family_id != outer
            or inner not in full_families
            or inner in inner_families
            or full_families != inner_families | {inner}
            or set(self.full_joint_fit.training_example_ids)
            != set(self.inner_joint_fit.training_example_ids)
            | {self.held_example.example_id}
            or self.held_example.example_id
            in self.inner_joint_fit.training_example_ids
            or self.held_example.codec_artifact_sha256
            != self.inner_joint_fit.codec_artifact_sha256
            or self.full_joint_fit.refit_artifact_sha256
            != self.inner_joint_fit.refit_artifact_sha256
            or not torch.equal(
                self.full_joint_fit.mean_gain_delta,
                self.inner_joint_fit.mean_gain_delta,
            )
            or not _scalar_control_matches_joint(
                self.inner_joint_fit,
                self.inner_scalar_control_fit,
            )
        ):
            raise ValueError("joint inner evidence does not match its outer fold")
        object.__setattr__(self, "outer_held_family_id", outer)
        object.__setattr__(self, "inner_held_family_id", inner)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_INNER_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def held_scalar_contribution(self) -> float:
        scalar_gradient = float(self.held_example.static_tangent_tensor().mean())
        return scalar_gradient * self.inner_joint_fit.applied_intercept

    @property
    def held_state_contribution(self) -> float:
        state_gradient = self.held_example.state_tangent_design_tensor().mean(dim=0)
        return float(
            torch.dot(state_gradient, self.inner_joint_fit.applied_parameter[1:])
        )

    @property
    def held_joint_total_derivative(self) -> float:
        return self.held_scalar_contribution + self.held_state_contribution

    @property
    def held_scalar_comparator_derivative(self) -> float:
        scalar_gradient = float(self.held_example.static_tangent_tensor().mean())
        return scalar_gradient * self.inner_scalar_control_fit.applied_coefficient

    @property
    def joint_minus_scalar_margin(self) -> float:
        return (
            self.held_joint_total_derivative
            - self.held_scalar_comparator_derivative
        )

    @property
    def joint_derivative_is_negative(self) -> bool:
        return self.held_joint_total_derivative < 0.0

    @property
    def joint_beats_scalar(self) -> bool:
        return self.joint_minus_scalar_margin < 0.0

    @property
    def state_raw_slope_cosine(self) -> float:
        return _cosine(
            self.full_joint_fit.raw_state_slope_tensor(),
            self.inner_joint_fit.raw_state_slope_tensor(),
        )

    @property
    def inner_feature_and_design_identifiable(self) -> bool:
        return _fit_is_identifiable(self.inner_joint_fit)

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "outer_held_family_id": self.outer_held_family_id,
            "inner_held_family_id": self.inner_held_family_id,
            "full_joint_fit_artifact_sha256": self.full_joint_fit.artifact_sha256,
            "inner_joint_fit_artifact_sha256": self.inner_joint_fit.artifact_sha256,
            "inner_scalar_control_fit_artifact_sha256": (
                self.inner_scalar_control_fit.artifact_sha256
            ),
            "held_example_id": self.held_example.example_id,
            "held_example_artifact_sha256": self.held_example.artifact_sha256,
            "held_scalar_contribution": self.held_scalar_contribution,
            "held_state_contribution": self.held_state_contribution,
            "held_joint_total_derivative": self.held_joint_total_derivative,
            "held_scalar_comparator_derivative": (
                self.held_scalar_comparator_derivative
            ),
            "joint_minus_scalar_margin": self.joint_minus_scalar_margin,
            "joint_derivative_is_negative": self.joint_derivative_is_negative,
            "joint_beats_scalar": self.joint_beats_scalar,
            "state_raw_slope_cosine": self.state_raw_slope_cosine,
            "cosine_excludes_intercept": True,
            "inner_augmented_feature_rank": (
                self.inner_joint_fit.augmented_feature_rank
            ),
            "inner_augmented_feature_condition": (
                self.inner_joint_fit.augmented_feature_condition
            ),
            "inner_standardized_design_rank": (
                self.inner_joint_fit.standardized_design_rank
            ),
            "inner_standardized_design_condition": (
                self.inner_joint_fit.standardized_design_condition
            ),
            "inner_feature_and_design_identifiable": (
                self.inner_feature_and_design_identifiable
            ),
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.full_joint_fit.validate_integrity()
        self.inner_joint_fit.validate_integrity()
        self.inner_scalar_control_fit.validate_integrity()
        self.held_example.validate_integrity()
        if (
            self.outer_held_family_id != self.full_joint_fit.held_family_id
            or self.inner_held_family_id != self.held_example.family_id
            or not _scalar_control_matches_joint(
                self.inner_joint_fit,
                self.inner_scalar_control_fit,
            )
            or _sha256(_INNER_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="joint inner record")
        ):
            raise RuntimeError("joint inner analytic record payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64JointStateGainFoldAnalyticRecord:
    full_joint_fit: CandidateConditionedK64JointStateGainFieldFit = field(
        repr=False
    )
    full_scalar_control_fit: CandidateConditionedK64StaticAmplitudeControlFit = (
        field(repr=False)
    )
    inner_family_records: tuple[
        CandidateConditionedK64JointInnerFamilyAnalyticRecord, ...
    ] = field(repr=False)
    outer_held_family_id: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.full_joint_fit,
            CandidateConditionedK64JointStateGainFieldFit,
        ) or not isinstance(
            self.full_scalar_control_fit,
            CandidateConditionedK64StaticAmplitudeControlFit,
        ):
            raise TypeError("joint fold record requires typed full fits")
        supplied = tuple(self.inner_family_records)
        if any(
            not isinstance(
                value,
                CandidateConditionedK64JointInnerFamilyAnalyticRecord,
            )
            for value in supplied
        ):
            raise TypeError("joint fold record requires typed inner records")
        records = tuple(
            sorted(supplied, key=lambda value: value.inner_held_family_id)
        )
        self.full_joint_fit.validate_integrity()
        self.full_scalar_control_fit.validate_integrity()
        for value in records:
            value.validate_integrity()
        outer = self.full_joint_fit.held_family_id
        inner_ids = tuple(value.inner_held_family_id for value in records)
        if (
            len(records) != _OUTER_TRAINING_FAMILY_COUNT
            or len(set(inner_ids)) != len(inner_ids)
            or inner_ids != self.full_joint_fit.training_family_ids
            or any(value.outer_held_family_id != outer for value in records)
            or any(
                value.full_joint_fit.artifact_sha256
                != self.full_joint_fit.artifact_sha256
                for value in records
            )
            or not _scalar_control_matches_joint(
                self.full_joint_fit,
                self.full_scalar_control_fit,
            )
        ):
            raise ValueError("joint fold evidence is incomplete or mismatched")
        object.__setattr__(self, "inner_family_records", records)
        object.__setattr__(self, "outer_held_family_id", outer)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FOLD_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def feature_and_design_gate_passed(self) -> bool:
        return _fit_is_identifiable(self.full_joint_fit) and all(
            value.inner_feature_and_design_identifiable
            for value in self.inner_family_records
        )

    @property
    def residual_energy_gate_passed(self) -> bool:
        return (
            self.full_joint_fit.residual_conditional_state_fisher_fraction
            >= STATE_RESIDUAL_FISHER_MINIMUM
        )

    @property
    def full_joint_non_noop(self) -> bool:
        return not self.full_joint_fit.no_op

    @property
    def negative_inner_derivative_count(self) -> int:
        return sum(
            value.joint_derivative_is_negative for value in self.inner_family_records
        )

    @property
    def negative_inner_local_gate_passed(self) -> bool:
        return self.negative_inner_derivative_count >= 4

    @property
    def joint_inner_macro_derivative(self) -> float:
        return sum(
            value.held_joint_total_derivative
            for value in self.inner_family_records
        ) / len(self.inner_family_records)

    @property
    def scalar_inner_macro_derivative(self) -> float:
        return sum(
            value.held_scalar_comparator_derivative
            for value in self.inner_family_records
        ) / len(self.inner_family_records)

    @property
    def joint_beats_scalar_inner_macro(self) -> bool:
        return self.joint_inner_macro_derivative < self.scalar_inner_macro_derivative

    @property
    def joint_beats_scalar_cell_count(self) -> int:
        return sum(value.joint_beats_scalar for value in self.inner_family_records)

    @property
    def median_inner_full_state_slope_cosine(self) -> float:
        values = sorted(
            value.state_raw_slope_cosine for value in self.inner_family_records
        )
        return values[len(values) // 2]

    @property
    def cosine_stability_gate_passed(self) -> bool:
        return self.median_inner_full_state_slope_cosine >= 0.90

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "outer_held_family_id": self.outer_held_family_id,
            "full_joint_fit_artifact_sha256": self.full_joint_fit.artifact_sha256,
            "full_scalar_control_fit_artifact_sha256": (
                self.full_scalar_control_fit.artifact_sha256
            ),
            "inner_family_record_artifact_sha256s": tuple(
                value.artifact_sha256 for value in self.inner_family_records
            ),
            "feature_and_design_gate_passed": (
                self.feature_and_design_gate_passed
            ),
            "full_residual_conditional_state_fisher_fraction": (
                self.full_joint_fit.residual_conditional_state_fisher_fraction
            ),
            "residual_energy_gate_passed": self.residual_energy_gate_passed,
            "full_joint_non_noop": self.full_joint_non_noop,
            "negative_inner_derivative_count": (
                self.negative_inner_derivative_count
            ),
            "negative_inner_local_gate_passed": (
                self.negative_inner_local_gate_passed
            ),
            "joint_inner_macro_derivative": self.joint_inner_macro_derivative,
            "scalar_inner_macro_derivative": self.scalar_inner_macro_derivative,
            "joint_beats_scalar_inner_macro": (
                self.joint_beats_scalar_inner_macro
            ),
            "joint_beats_scalar_cell_count": self.joint_beats_scalar_cell_count,
            "joint_beats_scalar_cell_count_is_gate": False,
            "median_inner_full_state_slope_cosine": (
                self.median_inner_full_state_slope_cosine
            ),
            "cosine_stability_gate_passed": self.cosine_stability_gate_passed,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.full_joint_fit.validate_integrity()
        self.full_scalar_control_fit.validate_integrity()
        for value in self.inner_family_records:
            value.validate_integrity()
        if (
            self.outer_held_family_id != self.full_joint_fit.held_family_id
            or not _scalar_control_matches_joint(
                self.full_joint_fit,
                self.full_scalar_control_fit,
            )
            or _sha256(_FOLD_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="joint fold record")
        ):
            raise RuntimeError("joint fold analytic record payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64JointStateGainAnalyticScreen:
    fold_records: tuple[
        CandidateConditionedK64JointStateGainFoldAnalyticRecord, ...
    ] = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        supplied = tuple(self.fold_records)
        if any(
            not isinstance(
                value,
                CandidateConditionedK64JointStateGainFoldAnalyticRecord,
            )
            for value in supplied
        ):
            raise TypeError("joint screen requires typed fold records")
        records = tuple(
            sorted(supplied, key=lambda value: value.outer_held_family_id)
        )
        for value in records:
            value.validate_integrity()
        outer_ids = tuple(value.outer_held_family_id for value in records)
        if (
            len(records) != _OUTER_FAMILY_COUNT
            or len(set(outer_ids)) != len(outer_ids)
            or any(
                set(value.full_joint_fit.training_family_ids)
                != set(outer_ids) - {value.outer_held_family_id}
                for value in records
            )
        ):
            raise ValueError("joint screen requires one coherent eight-fold panel")
        object.__setattr__(self, "fold_records", records)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SCREEN_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def feature_and_design_gate_passed(self) -> bool:
        return all(value.feature_and_design_gate_passed for value in self.fold_records)

    @property
    def residual_energy_pass_count(self) -> int:
        return sum(value.residual_energy_gate_passed for value in self.fold_records)

    @property
    def residual_energy_gate_passed(self) -> bool:
        return self.residual_energy_pass_count >= 6

    @property
    def non_noop_fold_count(self) -> int:
        return sum(value.full_joint_non_noop for value in self.fold_records)

    @property
    def non_noop_gate_passed(self) -> bool:
        return self.non_noop_fold_count >= 6

    @property
    def negative_inner_derivative_count(self) -> int:
        return sum(
            value.negative_inner_derivative_count for value in self.fold_records
        )

    @property
    def negative_inner_global_gate_passed(self) -> bool:
        return self.negative_inner_derivative_count >= 42

    @property
    def negative_inner_local_fold_count(self) -> int:
        return sum(
            value.negative_inner_local_gate_passed for value in self.fold_records
        )

    @property
    def negative_inner_local_gate_passed(self) -> bool:
        return self.negative_inner_local_fold_count >= 6

    @property
    def joint_beats_scalar_fold_count(self) -> int:
        return sum(
            value.joint_beats_scalar_inner_macro for value in self.fold_records
        )

    @property
    def joint_beats_scalar_gate_passed(self) -> bool:
        return self.joint_beats_scalar_fold_count >= 6

    @property
    def joint_beats_scalar_cell_count(self) -> int:
        return sum(
            value.joint_beats_scalar_cell_count for value in self.fold_records
        )

    @property
    def cosine_stability_fold_count(self) -> int:
        return sum(
            value.cosine_stability_gate_passed for value in self.fold_records
        )

    @property
    def cosine_stability_gate_passed(self) -> bool:
        return self.cosine_stability_fold_count >= 6

    @property
    def capacity_screen_passed(self) -> bool:
        return (
            self.feature_and_design_gate_passed
            and self.residual_energy_gate_passed
            and self.non_noop_gate_passed
            and self.negative_inner_global_gate_passed
            and self.negative_inner_local_gate_passed
            and self.joint_beats_scalar_gate_passed
            and self.cosine_stability_gate_passed
        )

    @property
    def outcome(self) -> str:
        if not self.feature_and_design_gate_passed:
            return "fail_augmented_feature_or_design_identifiability"
        if not self.residual_energy_gate_passed:
            return "fail_residual_conditional_state_fisher_energy"
        if not self.non_noop_gate_passed:
            return "fail_joint_non_noop_support"
        if not self.negative_inner_global_gate_passed:
            return "fail_global_joint_derivative_support"
        if not self.negative_inner_local_gate_passed:
            return "fail_fold_local_joint_derivative_support"
        if not self.joint_beats_scalar_gate_passed:
            return "fail_joint_vs_scalar_attribution"
        if not self.cosine_stability_gate_passed:
            return "fail_state_slope_stability"
        return "joint_capacity_supported_for_finite_validation"

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "fold_record_artifact_sha256s": tuple(
                value.artifact_sha256 for value in self.fold_records
            ),
            "outer_held_family_ids": tuple(
                value.outer_held_family_id for value in self.fold_records
            ),
            "augmented_feature_rank_required": JOINT_STATE_GAIN_PARAMETER_COUNT,
            "joint_design_rank_required": JOINT_STATE_GAIN_PARAMETER_COUNT,
            "standardized_condition_maximum": (
                STATE_STANDARDIZED_CONDITION_MAXIMUM
            ),
            "feature_and_design_gate_passed": (
                self.feature_and_design_gate_passed
            ),
            "residual_conditional_state_fisher_minimum": (
                STATE_RESIDUAL_FISHER_MINIMUM
            ),
            "residual_energy_pass_count": self.residual_energy_pass_count,
            "residual_energy_required_fold_count": 6,
            "residual_energy_gate_passed": self.residual_energy_gate_passed,
            "non_noop_fold_count": self.non_noop_fold_count,
            "non_noop_required_fold_count": 6,
            "non_noop_gate_passed": self.non_noop_gate_passed,
            "negative_inner_derivative_count": (
                self.negative_inner_derivative_count
            ),
            "negative_inner_derivative_required_count": 42,
            "negative_inner_global_gate_passed": (
                self.negative_inner_global_gate_passed
            ),
            "negative_inner_local_fold_count": (
                self.negative_inner_local_fold_count
            ),
            "negative_inner_local_required_per_fold": 4,
            "negative_inner_local_required_fold_count": 6,
            "negative_inner_local_gate_passed": (
                self.negative_inner_local_gate_passed
            ),
            "joint_beats_scalar_fold_count": self.joint_beats_scalar_fold_count,
            "joint_beats_scalar_required_fold_count": 6,
            "joint_beats_scalar_gate_passed": (
                self.joint_beats_scalar_gate_passed
            ),
            "joint_beats_scalar_cell_count": self.joint_beats_scalar_cell_count,
            "joint_beats_scalar_cell_count_is_gate": False,
            "cosine_stability_fold_count": self.cosine_stability_fold_count,
            "median_inner_full_state_slope_cosine_minimum": 0.90,
            "cosine_stability_required_fold_count": 6,
            "cosine_stability_gate_passed": (
                self.cosine_stability_gate_passed
            ),
            "capacity_screen_passed": self.capacity_screen_passed,
            "outcome": self.outcome,
            "analytic_forward_count": 112,
            "analytic_backward_count": 494,
            "parent_forward_count": 48,
            "parent_backward_count": 109,
            "shared_row_bank_forward_count": 64,
            "shared_row_bank_backward_count": 385,
            "extra_backward_count": 0,
            "capacity_screen_only": True,
            "finite_joint_candidate_execution_performed": False,
            "authorizes_final_selection": False,
            "authorizes_serving_or_model_mutation": False,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        for value in self.fold_records:
            value.validate_integrity()
        if _sha256(
            _SCREEN_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(self.artifact_sha256, label="joint analytic screen"):
            raise RuntimeError("joint analytic screen payload drifted")


def build_candidate_conditioned_k64_joint_inner_family_analytic_record(
    full_joint_fit: CandidateConditionedK64JointStateGainFieldFit,
    inner_joint_fit: CandidateConditionedK64JointStateGainFieldFit,
    inner_scalar_control_fit: CandidateConditionedK64StaticAmplitudeControlFit,
    held_example: CandidateConditionedK64StateGainGradientExample,
) -> CandidateConditionedK64JointInnerFamilyAnalyticRecord:
    return CandidateConditionedK64JointInnerFamilyAnalyticRecord(
        full_joint_fit=full_joint_fit,
        inner_joint_fit=inner_joint_fit,
        inner_scalar_control_fit=inner_scalar_control_fit,
        held_example=held_example,
    )


def build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record(
    full_joint_fit: CandidateConditionedK64JointStateGainFieldFit,
    full_scalar_control_fit: CandidateConditionedK64StaticAmplitudeControlFit,
    inner_family_records: Sequence[
        CandidateConditionedK64JointInnerFamilyAnalyticRecord
    ],
) -> CandidateConditionedK64JointStateGainFoldAnalyticRecord:
    return CandidateConditionedK64JointStateGainFoldAnalyticRecord(
        full_joint_fit=full_joint_fit,
        full_scalar_control_fit=full_scalar_control_fit,
        inner_family_records=tuple(inner_family_records),
    )


def screen_candidate_conditioned_k64_joint_state_gain_capacity(
    fold_records: Sequence[
        CandidateConditionedK64JointStateGainFoldAnalyticRecord
    ],
) -> CandidateConditionedK64JointStateGainAnalyticScreen:
    return CandidateConditionedK64JointStateGainAnalyticScreen(
        fold_records=tuple(fold_records)
    )
