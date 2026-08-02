"""Pure analytic capacity screen for a nested state-conditioned K64 gain field.

The frozen V4 mean-KL proposal supplies one K64 direction ``delta``.  The
field only redistributes the already-tested V5 ``+1/64`` step across rows::

    a_r = 1 + tanh(z_r @ w)
    g_rk = 1 + (1/64) * a_r * delta_k

``z`` contains four fit-only standardized coordinates of the *pre-gate*
base-H4 row along the first four held-fold K64 directions.  There is no bias:
``w == 0`` is exactly the V5 static arm, and ``a_r`` is bounded in ``[0, 2]``
so this rung cannot reverse the frozen direction.

This module performs no finite tune or held execution.  It fits the four
state coefficients and an identically trained one-scalar static-amplitude
control from row-resolved unit-point teacher-KL VJPs, builds nested
inner-family analytic records, and applies predeclared capacity gates.  Raw
tensors remain in defensive typed records; metadata exposes hashes and scalar
summaries only.
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
    "STATE_FEATURE_RANK",
    "STATE_GAIN_BASE_STEP",
    "STATE_GAIN_LOGIT_RMS_TRUST",
    "STATE_DESIGN_RANK_RELATIVE_TOLERANCE",
    "STATE_FEATURE_SCALE_MINIMUM",
    "STATE_STANDARDIZED_CONDITION_MAXIMUM",
    "STATE_RESIDUAL_FISHER_MINIMUM",
    "CandidateConditionedK64StateFeatureExample",
    "CandidateConditionedK64StateFeatureCodec",
    "CandidateConditionedK64StateGainGradientExample",
    "CandidateConditionedK64StateGainFieldFit",
    "CandidateConditionedK64StaticAmplitudeControlFit",
    "CandidateConditionedK64InnerFamilyAnalyticRecord",
    "CandidateConditionedK64StateGainFoldAnalyticRecord",
    "CandidateConditionedK64StateGainAnalyticScreen",
    "fit_candidate_conditioned_k64_state_feature_codec",
    "encode_candidate_conditioned_k64_state_features",
    "reduce_candidate_conditioned_k64_row_mode_scores",
    "contract_candidate_conditioned_k64_row_direction_scores",
    "fit_candidate_conditioned_k64_state_gain_field",
    "fit_candidate_conditioned_k64_static_amplitude_control",
    "build_candidate_conditioned_k64_inner_family_analytic_record",
    "build_candidate_conditioned_k64_state_gain_fold_analytic_record",
    "screen_candidate_conditioned_k64_state_gain_capacity",
]


STATE_FEATURE_RANK = 4
STATE_GAIN_BASE_STEP = 1.0 / 64.0
STATE_GAIN_LOGIT_RMS_TRUST = 1.0
STATE_DESIGN_RANK_RELATIVE_TOLERANCE = 1.0e-10
STATE_FEATURE_SCALE_MINIMUM = 1.0e-12
STATE_STANDARDIZED_CONDITION_MAXIMUM = 100.0
STATE_RESIDUAL_FISHER_MINIMUM = 0.05
_OUTER_FAMILY_COUNT = 8
_OUTER_TRAINING_FAMILY_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_DAMPING_FRACTION = 0.1
_DAMPING_FLOOR = 1.0e-12
_TINY = torch.finfo(torch.float64).tiny
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_TENSOR_DOMAIN = b"fisher-graph:candidate-k64-state-gain-tensor:v6\0"
_FEATURE_EXAMPLE_DOMAIN = b"fisher-graph:candidate-k64-state-feature-example:v6\0"
_CODEC_DOMAIN = b"fisher-graph:candidate-k64-state-feature-codec:v6\0"
_GRADIENT_EXAMPLE_DOMAIN = b"fisher-graph:candidate-k64-state-gradient-example:v6\0"
_FIELD_FIT_DOMAIN = b"fisher-graph:candidate-k64-state-gain-field-fit:v6\0"
_STATIC_FIT_DOMAIN = b"fisher-graph:candidate-k64-static-amplitude-fit:v6\0"
_INNER_DOMAIN = b"fisher-graph:candidate-k64-state-gain-inner-record:v6\0"
_FOLD_DOMAIN = b"fisher-graph:candidate-k64-state-gain-fold-record:v6\0"
_SCREEN_DOMAIN = b"fisher-graph:candidate-k64-state-gain-screen:v6\0"


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


def _families_and_examples(
    values: Sequence[object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    families = tuple(sorted({str(getattr(value, "family_id")) for value in values}))
    examples = tuple(sorted(str(getattr(value, "example_id")) for value in values))
    if (
        len(values) not in (_INNER_TRAINING_FAMILY_COUNT, _OUTER_TRAINING_FAMILY_COUNT)
        or len(families) != len(values)
        or len(set(examples)) != len(values)
    ):
        raise ValueError("state gain evidence requires one unique example per family")
    return families, examples


def _weighted_mean(vectors: Sequence[Tensor]) -> Tensor:
    if not vectors:
        raise ValueError("family-equal mean requires evidence")
    return torch.stack(tuple(value.mean(dim=0) for value in vectors)).mean(dim=0)


def _family_equal_second_moment(matrices: Sequence[Tensor]) -> Tensor:
    return torch.stack(
        tuple((value.T @ value) / value.shape[0] for value in matrices)
    ).mean(dim=0).contiguous()


def _rank_condition_from_gram(gram: Tensor) -> tuple[int, float]:
    symmetric = 0.5 * (gram + gram.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    maximum = max(float(eigenvalues[-1]), 0.0)
    tolerance = STATE_DESIGN_RANK_RELATIVE_TOLERANCE * maximum
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.numel())
    if rank != gram.shape[0] or positive.numel() == 0:
        return rank, torch.finfo(torch.float64).max
    return rank, math.sqrt(maximum / float(positive[0]))


def _column_standardized_gram(gram: Tensor) -> Tensor:
    diagonal = torch.clamp(gram.diagonal(), min=0.0)
    inverse = torch.where(
        diagonal > 0.0,
        diagonal.rsqrt(),
        torch.zeros_like(diagonal),
    )
    return (inverse[:, None] * gram * inverse[None, :]).contiguous()


def _expected_damping(gram: Tensor) -> float:
    return max(
        _DAMPING_FRACTION * float(gram.diagonal().median()),
        _DAMPING_FLOOR,
    )


def _solve(gram: Tensor, gradient: Tensor, damping: float) -> Tensor:
    system = gram + damping * torch.eye(gram.shape[0], dtype=torch.float64)
    if bool((gradient == 0.0).all()):
        return torch.zeros_like(gradient)
    scale = max(float(system.abs().max()), float(gradient.abs().max()), _TINY)
    try:
        result = torch.linalg.solve(system / scale, -gradient / scale)
    except RuntimeError as error:
        raise RuntimeError("state gain OPG solve failed") from error
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("state gain OPG solve was nonfinite")
    return result.contiguous()


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = float(
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )
    if denominator == 0.0:
        return 0.0
    value = float(torch.dot(left, right)) / denominator
    return max(-1.0, min(1.0, value))


def _is_positive_semidefinite(value: Tensor) -> bool:
    symmetric = 0.5 * (value + value.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    scale = max(float(torch.linalg.matrix_norm(symmetric, ord=2)), _TINY)
    return float(eigenvalues[0]) >= -STATE_DESIGN_RANK_RELATIVE_TOLERANCE * scale


def _residual_conditional_fisher_fraction(
    fisher: Tensor,
    cross: Tensor,
    static_energy: float,
) -> float:
    symmetric_fisher = 0.5 * (fisher + fisher.T)
    total = float(torch.trace(symmetric_fisher))
    if total <= 0.0:
        return 0.0
    explained = torch.outer(cross, cross) / max(static_energy, _TINY)
    residual = 0.5 * (
        symmetric_fisher - explained + (symmetric_fisher - explained).T
    )
    residual_eigenvalues = torch.linalg.eigvalsh(residual)
    scale = max(float(torch.linalg.matrix_norm(symmetric_fisher, ord=2)), _TINY)
    tolerance = STATE_DESIGN_RANK_RELATIVE_TOLERANCE * scale
    if float(residual_eigenvalues[0]) < -tolerance:
        raise ValueError("conditional Fisher residual is materially non-PSD")
    fraction = float(torch.trace(residual)) / total
    if fraction < -STATE_DESIGN_RANK_RELATIVE_TOLERANCE or fraction > (
        1.0 + STATE_DESIGN_RANK_RELATIVE_TOLERANCE
    ):
        raise ValueError("conditional Fisher residual fraction is invalid")
    return max(0.0, min(1.0, fraction))


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StateFeatureExample:
    example_id: str
    family_id: str
    base_h4_support_rows: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        rows = _float64(
            self.base_h4_support_rows,
            label="pre-gate base-H4 support rows",
            ndim=2,
        )
        object.__setattr__(self, "example_id", _identifier(self.example_id, label="example_id"))
        object.__setattr__(self, "family_id", _identifier(self.family_id, label="family_id"))
        object.__setattr__(self, "base_h4_support_rows", rows)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FEATURE_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "support_row_count": int(self.base_h4_support_rows.shape[0]),
            "width": int(self.base_h4_support_rows.shape[1]),
            "base_h4_support_rows_sha256": _tensor_sha256(
                self.base_h4_support_rows
            ),
            "feature_source": "pre_gate_bridge_base_H4_support_rows",
            "post_gate_state_used": False,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.base_h4_support_rows.dtype != torch.float64
            or self.base_h4_support_rows.device.type != "cpu"
            or self.base_h4_support_rows.requires_grad
            or not self.base_h4_support_rows.is_contiguous()
            or _sha256(_FEATURE_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="state feature example")
        ):
            raise RuntimeError("state feature example payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StateFeatureCodec:
    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    ordered_directions_sha256: str
    feature_center: Tensor = field(repr=False)
    feature_scale: Tensor = field(repr=False)
    standardized_feature_gram: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        center = _float64(self.feature_center, label="feature center", ndim=1)
        scale = _float64(self.feature_scale, label="feature scale", ndim=1)
        gram = _float64(
            self.standardized_feature_gram,
            label="standardized feature Gram",
            ndim=2,
        )
        families = tuple(_identifier(v, label="training family_id") for v in self.training_family_ids)
        examples = tuple(_identifier(v, label="training example_id") for v in self.training_example_ids)
        evidence = tuple(_require_sha256(v, label="feature example artifact") for v in self.training_example_artifact_sha256s)
        held = _identifier(self.held_family_id, label="held_family_id")
        if (
            len(families) not in (_INNER_TRAINING_FAMILY_COUNT, _OUTER_TRAINING_FAMILY_COUNT)
            or families != tuple(sorted(set(families)))
            or len(examples) != len(families)
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(families)
            or held in families
            or center.shape != (STATE_FEATURE_RANK,)
            or scale.shape != center.shape
            or gram.shape != (STATE_FEATURE_RANK, STATE_FEATURE_RANK)
            or bool((scale <= STATE_FEATURE_SCALE_MINIMUM).any())
            or not torch.allclose(gram, gram.T, rtol=0.0, atol=1.0e-12)
            or not _is_positive_semidefinite(gram)
        ):
            raise ValueError("state feature codec payload is invalid")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(self, "ordered_directions_sha256", _require_sha256(self.ordered_directions_sha256, label="ordered directions"))
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "standardized_feature_gram", gram)
        object.__setattr__(self, "artifact_sha256", _sha256(_CODEC_DOMAIN, self.metadata(include_artifact=False)))
        self.validate_integrity()

    @property
    def feature_rank(self) -> int:
        return _rank_condition_from_gram(self.standardized_feature_gram)[0]

    @property
    def feature_condition(self) -> float:
        return _rank_condition_from_gram(self.standardized_feature_gram)[1]

    def raw_projection_slope(self, standardized_weight: Tensor) -> Tensor:
        weight = _float64(standardized_weight, label="standardized weight", ndim=1)
        if weight.shape != (STATE_FEATURE_RANK,):
            raise ValueError("standardized weight width differs")
        self.validate_integrity()
        return (weight / self.feature_scale).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "training_example_artifact_sha256s": self.training_example_artifact_sha256s,
            "ordered_directions_sha256": self.ordered_directions_sha256,
            "feature_rank": self.feature_rank,
            "feature_condition": self.feature_condition,
            "feature_center_sha256": _tensor_sha256(self.feature_center),
            "feature_scale_sha256": _tensor_sha256(self.feature_scale),
            "standardized_feature_gram_sha256": _tensor_sha256(
                self.standardized_feature_gram
            ),
            "feature_center_minimum": float(self.feature_center.min()),
            "feature_center_maximum": float(self.feature_center.max()),
            "feature_scale_minimum": float(self.feature_scale.min()),
            "feature_scale_maximum": float(self.feature_scale.max()),
            "feature_scale_strict_minimum": STATE_FEATURE_SCALE_MINIMUM,
            "feature_coordinate_count": STATE_FEATURE_RANK,
            "feature_direction_indices": tuple(range(STATE_FEATURE_RANK)),
            "feature_source": "pre_gate_bridge_base_H4_support_rows",
            "normalization": "fit_only_family_equal_center_and_diagonal_scale",
            "learned_parameter_count": 0,
            "derived_float_scalar_count": 2 * STATE_FEATURE_RANK,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(_CODEC_DOMAIN, self.metadata(include_artifact=False)) != _require_sha256(self.artifact_sha256, label="state feature codec"):
            raise RuntimeError("state feature codec payload drifted")


def fit_candidate_conditioned_k64_state_feature_codec(
    examples: Sequence[CandidateConditionedK64StateFeatureExample],
    *,
    held_family_id: str,
    ordered_directions: Tensor,
) -> CandidateConditionedK64StateFeatureCodec:
    values = tuple(sorted(tuple(examples), key=lambda value: value.example_id))
    if any(not isinstance(value, CandidateConditionedK64StateFeatureExample) for value in values):
        raise TypeError("state feature examples must be typed records")
    for value in values:
        value.validate_integrity()
    families, example_ids = _families_and_examples(values)
    held = _identifier(held_family_id, label="held_family_id")
    if held in families:
        raise ValueError("held family cannot fit state feature codec")
    directions = _float64(ordered_directions, label="ordered K64 directions", ndim=2)
    widths = {int(value.base_h4_support_rows.shape[1]) for value in values}
    if (
        directions.shape[0] != CANDIDATE_GAIN_RANK
        or widths != {int(directions.shape[1])}
        or not torch.allclose(
            directions @ directions.T,
            torch.eye(CANDIDATE_GAIN_RANK, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError("state feature direction geometry differs")
    feature_directions = directions[:STATE_FEATURE_RANK]
    raw = tuple(
        (value.base_h4_support_rows @ feature_directions.T).contiguous()
        for value in values
    )
    center = _weighted_mean(raw)
    centered = tuple((value - center).contiguous() for value in raw)
    variance = _weighted_mean(tuple(value.square() for value in centered))
    if bool((variance <= STATE_FEATURE_SCALE_MINIMUM**2).any()):
        raise ValueError(
            "state feature fit has a zero or near-zero family-equal variance"
        )
    scale = torch.sqrt(variance).contiguous()
    standardized = tuple((value / scale).contiguous() for value in centered)
    gram = _family_equal_second_moment(standardized)
    return CandidateConditionedK64StateFeatureCodec(
        held_family_id=held,
        training_family_ids=families,
        training_example_ids=example_ids,
        training_example_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        ordered_directions_sha256=_tensor_sha256(directions),
        feature_center=center,
        feature_scale=scale,
        standardized_feature_gram=gram,
    )


def encode_candidate_conditioned_k64_state_features(
    codec: CandidateConditionedK64StateFeatureCodec,
    *,
    base_h4_support_rows: Tensor,
    ordered_directions: Tensor,
) -> Tensor:
    if not isinstance(codec, CandidateConditionedK64StateFeatureCodec):
        raise TypeError("codec must be a state feature codec")
    codec.validate_integrity()
    rows = _float64(base_h4_support_rows, label="pre-gate base-H4 rows", ndim=2)
    directions = _float64(ordered_directions, label="ordered K64 directions", ndim=2)
    if (
        directions.shape != (CANDIDATE_GAIN_RANK, rows.shape[1])
        or _tensor_sha256(directions) != codec.ordered_directions_sha256
    ):
        raise ValueError("state feature runtime directions differ from codec")
    raw = rows @ directions[:STATE_FEATURE_RANK].T
    return ((raw - codec.feature_center) / codec.feature_scale).contiguous()


def contract_candidate_conditioned_k64_row_direction_scores(
    *,
    tail_rows: Tensor,
    ordered_directions: Tensor,
    token_h4_gradients: Tensor,
    mean_gain_delta: Tensor,
) -> Tensor:
    """Return token-by-row derivative scores for the nested amplitude field."""

    tail = _float64(tail_rows, label="candidate tail rows", ndim=2)
    directions = _float64(ordered_directions, label="ordered directions", ndim=2)
    gradients = _float64(token_h4_gradients, label="token H4 gradients", ndim=3)
    delta = _float64(mean_gain_delta, label="mean gain delta", ndim=1)
    if (
        directions.shape != (CANDIDATE_GAIN_RANK, tail.shape[1])
        or gradients.shape[1:] != tail.shape
        or delta.shape != (CANDIDATE_GAIN_RANK,)
        or not torch.allclose(
            directions @ directions.T,
            torch.eye(CANDIDATE_GAIN_RANK, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError("row-resolved gain score geometry differs")
    amplitudes = tail @ directions.T
    gradient_coordinates = torch.einsum("trw,kw->trk", gradients, directions)
    row_mode_scores = amplitudes.unsqueeze(0) * gradient_coordinates
    return reduce_candidate_conditioned_k64_row_mode_scores(
        token_row_mode_scores=row_mode_scores,
        mean_gain_delta=delta,
    )


def reduce_candidate_conditioned_k64_row_mode_scores(
    *,
    token_row_mode_scores: Tensor,
    mean_gain_delta: Tensor,
) -> Tensor:
    """Reduce authenticated ``T x R x K64`` mode scores to row scores."""

    scores = _float64(
        token_row_mode_scores,
        label="token-row-mode direction scores",
        ndim=3,
    )
    delta = _float64(mean_gain_delta, label="mean gain delta", ndim=1)
    if scores.shape[2] != CANDIDATE_GAIN_RANK or delta.shape != (
        CANDIDATE_GAIN_RANK,
    ):
        raise ValueError("row-mode score reduction geometry differs")
    return (
        STATE_GAIN_BASE_STEP * torch.einsum("trk,k->tr", scores, delta)
    ).contiguous()


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StateGainGradientExample:
    example_id: str
    family_id: str
    codec_artifact_sha256: str
    standardized_state_features: Tensor = field(repr=False)
    token_row_direction_scores: Tensor = field(repr=False)
    unit_token_teacher_kl: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        features = _float64(self.standardized_state_features, label="standardized state features", ndim=2)
        scores = _float64(self.token_row_direction_scores, label="token-row direction scores", ndim=2)
        teacher_kl = _float64(self.unit_token_teacher_kl, label="unit token teacher KL", ndim=1)
        if (
            features.shape[1] != STATE_FEATURE_RANK
            or scores.shape != (teacher_kl.numel(), features.shape[0])
            or bool((teacher_kl < 0.0).any())
        ):
            raise ValueError("state gain gradient example geometry differs")
        object.__setattr__(self, "example_id", _identifier(self.example_id, label="example_id"))
        object.__setattr__(self, "family_id", _identifier(self.family_id, label="family_id"))
        object.__setattr__(self, "codec_artifact_sha256", _require_sha256(self.codec_artifact_sha256, label="state feature codec"))
        object.__setattr__(self, "standardized_state_features", features)
        object.__setattr__(self, "token_row_direction_scores", scores)
        object.__setattr__(self, "unit_token_teacher_kl", teacher_kl)
        object.__setattr__(self, "artifact_sha256", _sha256(_GRADIENT_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)))
        self.validate_integrity()

    def state_tangent_design_tensor(self) -> Tensor:
        self.validate_integrity()
        return (self.token_row_direction_scores @ self.standardized_state_features).contiguous()

    def static_tangent_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.token_row_direction_scores.sum(dim=1).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        design = self.token_row_direction_scores @ self.standardized_state_features
        static = self.token_row_direction_scores.sum(dim=1)
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "codec_artifact_sha256": self.codec_artifact_sha256,
            "support_row_count": int(self.standardized_state_features.shape[0]),
            "supervised_token_count": int(self.unit_token_teacher_kl.numel()),
            "standardized_state_features_sha256": _tensor_sha256(self.standardized_state_features),
            "token_row_direction_scores_sha256": _tensor_sha256(self.token_row_direction_scores),
            "unit_token_teacher_kl_sha256": _tensor_sha256(self.unit_token_teacher_kl),
            "state_tangent_design_sha256": _tensor_sha256(design),
            "static_tangent_sha256": _tensor_sha256(static),
            "mean_unit_teacher_kl": float(self.unit_token_teacher_kl.mean()),
            "row_scores_include_one_over_64_step": True,
            "row_axis_retained_before_global_K64_reduction": True,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        tensors = (
            self.standardized_state_features,
            self.token_row_direction_scores,
            self.unit_token_teacher_kl,
        )
        if (
            any(
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or value.requires_grad
                or not value.is_contiguous()
                for value in tensors
            )
            or _sha256(_GRADIENT_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="state gradient example")
        ):
            raise RuntimeError("state gain gradient example payload drifted")


def _gradient_statistics(
    values: Sequence[CandidateConditionedK64StateGainGradientExample],
) -> dict[str, Tensor | float]:
    designs = tuple(value.state_tangent_design_tensor() for value in values)
    static = tuple(value.static_tangent_tensor() for value in values)
    gradients = _weighted_mean(designs)
    gram = _family_equal_second_moment(designs)
    feature_gram = _family_equal_second_moment(
        tuple(value.standardized_state_features for value in values)
    )
    static_gradient = float(_weighted_mean(tuple(value[:, None] for value in static))[0])
    static_energy = float(
        torch.stack(tuple(value.square().mean() for value in static)).mean()
    )
    cross = torch.stack(
        tuple((design * scalar[:, None]).mean(dim=0) for design, scalar in zip(designs, static))
    ).mean(dim=0).contiguous()
    return {
        "gradient": gradients,
        "gram": gram,
        "feature_gram": feature_gram,
        "static_gradient": static_gradient,
        "static_energy": static_energy,
        "cross": cross,
    }


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StateGainFieldFit:
    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    refit_artifact_sha256: str
    codec_artifact_sha256: str
    feature_scale: Tensor = field(repr=False)
    mean_gain_delta: Tensor = field(repr=False)
    mean_gradient: Tensor = field(repr=False)
    state_fisher_gram: Tensor = field(repr=False)
    feature_second_moment: Tensor = field(repr=False)
    state_static_cross_moment: Tensor = field(repr=False)
    static_fisher_energy: float
    raw_weight: Tensor = field(init=False, repr=False)
    applied_weight: Tensor = field(init=False, repr=False)
    damping: float = field(init=False)
    raw_logit_rms: float = field(init=False)
    trust_scale: float = field(init=False)
    predicted_incremental_derivative: float = field(init=False)
    no_op: bool = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        families = tuple(_identifier(v, label="training family_id") for v in self.training_family_ids)
        examples = tuple(_identifier(v, label="training example_id") for v in self.training_example_ids)
        evidence = tuple(_require_sha256(v, label="gradient example artifact") for v in self.training_example_artifact_sha256s)
        scale = _float64(self.feature_scale, label="feature scale", ndim=1)
        delta = _float64(self.mean_gain_delta, label="mean gain delta", ndim=1)
        gradient = _float64(self.mean_gradient, label="state mean gradient", ndim=1)
        fisher = _float64(self.state_fisher_gram, label="state Fisher Gram", ndim=2)
        feature = _float64(self.feature_second_moment, label="feature second moment", ndim=2)
        cross = _float64(self.state_static_cross_moment, label="state-static cross moment", ndim=1)
        static_energy = _finite(self.static_fisher_energy, label="static Fisher energy")
        held = _identifier(self.held_family_id, label="held_family_id")
        if (
            len(families) not in (_INNER_TRAINING_FAMILY_COUNT, _OUTER_TRAINING_FAMILY_COUNT)
            or families != tuple(sorted(set(families)))
            or len(examples) != len(families)
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(families)
            or held in families
            or scale.shape != (STATE_FEATURE_RANK,)
            or delta.shape != (CANDIDATE_GAIN_RANK,)
            or gradient.shape != (STATE_FEATURE_RANK,)
            or fisher.shape != (STATE_FEATURE_RANK, STATE_FEATURE_RANK)
            or feature.shape != fisher.shape
            or cross.shape != gradient.shape
            or static_energy < 0.0
            or bool((scale <= STATE_FEATURE_SCALE_MINIMUM).any())
            or bool((1.0 + delta < 0.0).any())
            or bool((1.0 + delta > 1.5).any())
            or not torch.allclose(fisher, fisher.T, rtol=0.0, atol=1.0e-12)
            or not torch.allclose(feature, feature.T, rtol=0.0, atol=1.0e-12)
            or not _is_positive_semidefinite(fisher)
            or not _is_positive_semidefinite(feature)
        ):
            raise ValueError("state gain field fit payload is invalid")
        _residual_conditional_fisher_fraction(
            fisher,
            cross,
            static_energy,
        )
        damping = _expected_damping(fisher)
        raw = _solve(fisher, gradient, damping)
        raw_rms = math.sqrt(max(float(raw @ feature @ raw), 0.0))
        trust = min(1.0, STATE_GAIN_LOGIT_RMS_TRUST / max(raw_rms, _TINY))
        applied = (trust * raw).contiguous()
        derivative = float(torch.dot(gradient, applied))
        no_op = derivative >= 0.0
        if no_op:
            applied = torch.zeros_like(applied)
            derivative = 0.0
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(self, "refit_artifact_sha256", _require_sha256(self.refit_artifact_sha256, label="V4 refit"))
        object.__setattr__(self, "codec_artifact_sha256", _require_sha256(self.codec_artifact_sha256, label="state codec"))
        for name, value in (
            ("feature_scale", scale),
            ("mean_gain_delta", delta),
            ("mean_gradient", gradient),
            ("state_fisher_gram", fisher),
            ("feature_second_moment", feature),
            ("state_static_cross_moment", cross),
            ("raw_weight", raw),
            ("applied_weight", applied),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "static_fisher_energy", static_energy)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "raw_logit_rms", raw_rms)
        object.__setattr__(self, "trust_scale", trust)
        object.__setattr__(self, "predicted_incremental_derivative", derivative)
        object.__setattr__(self, "no_op", no_op)
        object.__setattr__(self, "artifact_sha256", _sha256(_FIELD_FIT_DOMAIN, self.metadata(include_artifact=False)))
        self.validate_integrity()

    @property
    def standardized_design_rank(self) -> int:
        return _rank_condition_from_gram(
            _column_standardized_gram(self.state_fisher_gram)
        )[0]

    @property
    def standardized_design_condition(self) -> float:
        return _rank_condition_from_gram(
            _column_standardized_gram(self.state_fisher_gram)
        )[1]

    @property
    def feature_rank(self) -> int:
        return _rank_condition_from_gram(self.feature_second_moment)[0]

    @property
    def feature_condition(self) -> float:
        return _rank_condition_from_gram(self.feature_second_moment)[1]

    @property
    def residual_conditional_fisher_fraction(self) -> float:
        return _residual_conditional_fisher_fraction(
            self.state_fisher_gram,
            self.state_static_cross_moment,
            self.static_fisher_energy,
        )

    def raw_projection_slope_tensor(self) -> Tensor:
        self.validate_integrity()
        return (self.applied_weight / self.feature_scale).clone().contiguous()

    def row_amplitudes_tensor(self, standardized_features: Tensor) -> Tensor:
        features = _float64(standardized_features, label="runtime state features", ndim=2)
        if features.shape[1] != STATE_FEATURE_RANK:
            raise ValueError("runtime state feature width differs")
        self.validate_integrity()
        return (1.0 + torch.tanh(features @ self.applied_weight)).contiguous()

    def row_gains_tensor(self, standardized_features: Tensor) -> Tensor:
        amplitudes = self.row_amplitudes_tensor(standardized_features)
        return (
            1.0
            + STATE_GAIN_BASE_STEP
            * amplitudes[:, None]
            * self.mean_gain_delta[None, :]
        ).contiguous()

    def static_plus_gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return (1.0 + STATE_GAIN_BASE_STEP * self.mean_gain_delta).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        residual_fraction = self.residual_conditional_fisher_fraction
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "training_example_artifact_sha256s": self.training_example_artifact_sha256s,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "codec_artifact_sha256": self.codec_artifact_sha256,
            "feature_scale_sha256": _tensor_sha256(self.feature_scale),
            "mean_gain_delta_sha256": _tensor_sha256(self.mean_gain_delta),
            "mean_gradient_sha256": _tensor_sha256(self.mean_gradient),
            "state_fisher_gram_sha256": _tensor_sha256(self.state_fisher_gram),
            "feature_second_moment_sha256": _tensor_sha256(self.feature_second_moment),
            "state_static_cross_moment_sha256": _tensor_sha256(self.state_static_cross_moment),
            "static_fisher_energy": self.static_fisher_energy,
            "residual_conditional_fisher_fraction": residual_fraction,
            "feature_rank": self.feature_rank,
            "feature_condition": self.feature_condition,
            "standardized_design_rank": self.standardized_design_rank,
            "standardized_design_condition": self.standardized_design_condition,
            "rank_relative_tolerance": STATE_DESIGN_RANK_RELATIVE_TOLERANCE,
            "condition_definition": (
                "smax_over_smin_of_family_weighted_column_rms_standardized_design"
            ),
            "damping_fraction": _DAMPING_FRACTION,
            "damping_floor": _DAMPING_FLOOR,
            "damping": self.damping,
            "raw_weight_sha256": _tensor_sha256(self.raw_weight),
            "applied_weight_sha256": _tensor_sha256(self.applied_weight),
            "raw_projection_slope_sha256": _tensor_sha256(self.applied_weight / self.feature_scale),
            "raw_logit_rms": self.raw_logit_rms,
            "gate_logit_rms_trust": STATE_GAIN_LOGIT_RMS_TRUST,
            "trust_scale": self.trust_scale,
            "predicted_incremental_derivative": self.predicted_incremental_derivative,
            "no_op": self.no_op,
            "learned_parameter_count": STATE_FEATURE_RANK,
            "bias_parameter_count": 0,
            "incremental_derived_float_scalar_count": 2 * STATE_FEATURE_RANK,
            "incremental_float_count_including_codec": (
                3 * STATE_FEATURE_RANK
            ),
            "base_step_hex": STATE_GAIN_BASE_STEP.hex(),
            "amplitude_formula": "one_plus_tanh_z_dot_w",
            "amplitude_bounds": (0.0, 2.0),
            "w_zero_exactly_reproduces_v5_static_plus": True,
            "direction_reversal_possible": False,
            "fit_objective": "family_equal_unit_point_linear_teacher_KL_OPG",
            "finite_execution_authority_required_later": True,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.applied_weight.shape != (STATE_FEATURE_RANK,)
            or math.sqrt(max(float(self.applied_weight @ self.feature_second_moment @ self.applied_weight), 0.0))
            > STATE_GAIN_LOGIT_RMS_TRUST + 1.0e-12
            or _sha256(_FIELD_FIT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="state gain field fit")
        ):
            raise RuntimeError("state gain field fit payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StaticAmplitudeControlFit:
    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    refit_artifact_sha256: str
    mean_gain_delta: Tensor = field(repr=False)
    mean_gradient: float
    fisher_energy: float
    raw_coefficient: float = field(init=False)
    applied_coefficient: float = field(init=False)
    damping: float = field(init=False)
    trust_scale: float = field(init=False)
    predicted_incremental_derivative: float = field(init=False)
    no_op: bool = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        delta = _float64(self.mean_gain_delta, label="mean gain delta", ndim=1)
        gradient = _finite(self.mean_gradient, label="static mean gradient")
        energy = _finite(self.fisher_energy, label="static Fisher energy")
        families = tuple(_identifier(v, label="training family_id") for v in self.training_family_ids)
        examples = tuple(_identifier(v, label="training example_id") for v in self.training_example_ids)
        evidence = tuple(_require_sha256(v, label="gradient example artifact") for v in self.training_example_artifact_sha256s)
        held = _identifier(self.held_family_id, label="held_family_id")
        if (
            len(families) not in (_INNER_TRAINING_FAMILY_COUNT, _OUTER_TRAINING_FAMILY_COUNT)
            or families != tuple(sorted(set(families)))
            or len(examples) != len(families)
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(families)
            or held in families
            or delta.shape != (CANDIDATE_GAIN_RANK,)
            or energy < 0.0
        ):
            raise ValueError("static amplitude control fit payload is invalid")
        damping = max(_DAMPING_FRACTION * energy, _DAMPING_FLOOR)
        raw = 0.0 if gradient == 0.0 else -gradient / (energy + damping)
        trust = min(1.0, STATE_GAIN_LOGIT_RMS_TRUST / max(abs(raw), _TINY))
        applied = trust * raw
        derivative = gradient * applied
        no_op = derivative >= 0.0
        if no_op:
            applied = 0.0
            derivative = 0.0
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(self, "refit_artifact_sha256", _require_sha256(self.refit_artifact_sha256, label="V4 refit"))
        object.__setattr__(self, "mean_gain_delta", delta)
        object.__setattr__(self, "mean_gradient", gradient)
        object.__setattr__(self, "fisher_energy", energy)
        object.__setattr__(self, "raw_coefficient", raw)
        object.__setattr__(self, "applied_coefficient", applied)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "trust_scale", trust)
        object.__setattr__(self, "predicted_incremental_derivative", derivative)
        object.__setattr__(self, "no_op", no_op)
        object.__setattr__(self, "artifact_sha256", _sha256(_STATIC_FIT_DOMAIN, self.metadata(include_artifact=False)))
        self.validate_integrity()

    @property
    def amplitude(self) -> float:
        return 1.0 + math.tanh(self.applied_coefficient)

    def gains_tensor(self) -> Tensor:
        self.validate_integrity()
        return (
            1.0
            + STATE_GAIN_BASE_STEP * self.amplitude * self.mean_gain_delta
        ).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "training_example_artifact_sha256s": self.training_example_artifact_sha256s,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "mean_gain_delta_sha256": _tensor_sha256(self.mean_gain_delta),
            "mean_gradient": self.mean_gradient,
            "fisher_energy": self.fisher_energy,
            "damping": self.damping,
            "raw_coefficient": self.raw_coefficient,
            "applied_coefficient": self.applied_coefficient,
            "trust_scale": self.trust_scale,
            "predicted_incremental_derivative": self.predicted_incremental_derivative,
            "no_op": self.no_op,
            "amplitude": self.amplitude,
            "learned_parameter_count": 1,
            "bias_is_prompt_static_control_not_state_field_bias": True,
            "can_rescue_state_attribution": False,
            "base_step_hex": STATE_GAIN_BASE_STEP.hex(),
            "finite_execution_authority_required_later": True,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            abs(self.applied_coefficient) > STATE_GAIN_LOGIT_RMS_TRUST + 1.0e-12
            or _sha256(_STATIC_FIT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="static amplitude fit")
        ):
            raise RuntimeError("static amplitude control fit payload drifted")


def _validated_gradient_values(
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
    *,
    codec: CandidateConditionedK64StateFeatureCodec,
) -> tuple[CandidateConditionedK64StateGainGradientExample, ...]:
    values = tuple(sorted(tuple(examples), key=lambda value: value.example_id))
    if any(not isinstance(value, CandidateConditionedK64StateGainGradientExample) for value in values):
        raise TypeError("state gradient examples must be typed records")
    for value in values:
        value.validate_integrity()
    families, ids = _families_and_examples(values)
    if (
        families != codec.training_family_ids
        or ids != codec.training_example_ids
        or any(value.codec_artifact_sha256 != codec.artifact_sha256 for value in values)
    ):
        raise ValueError("state gradient examples differ from feature codec fit set")
    return values


def fit_candidate_conditioned_k64_state_gain_field(
    refit: CandidateConditionedK64MeanKLRefit,
    codec: CandidateConditionedK64StateFeatureCodec,
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
) -> CandidateConditionedK64StateGainFieldFit:
    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("refit must be a candidate V4 mean-KL refit")
    if not isinstance(codec, CandidateConditionedK64StateFeatureCodec):
        raise TypeError("codec must be a state feature codec")
    refit.validate_integrity()
    codec.validate_integrity()
    values = _validated_gradient_values(examples, codec=codec)
    if codec.held_family_id != refit.held_family_id:
        raise ValueError("state codec and V4 refit held families differ")
    stats = _gradient_statistics(values)
    if not torch.equal(
        stats["feature_gram"], codec.standardized_feature_gram
    ):
        raise ValueError("state gradient features do not replay the fitted codec")
    return CandidateConditionedK64StateGainFieldFit(
        held_family_id=refit.held_family_id,
        training_family_ids=codec.training_family_ids,
        training_example_ids=codec.training_example_ids,
        training_example_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        refit_artifact_sha256=refit.artifact_sha256,
        codec_artifact_sha256=codec.artifact_sha256,
        feature_scale=codec.feature_scale,
        mean_gain_delta=refit.mean_proposed_gains_tensor() - 1.0,
        mean_gradient=stats["gradient"],
        state_fisher_gram=stats["gram"],
        feature_second_moment=stats["feature_gram"],
        state_static_cross_moment=stats["cross"],
        static_fisher_energy=float(stats["static_energy"]),
    )


def fit_candidate_conditioned_k64_static_amplitude_control(
    refit: CandidateConditionedK64MeanKLRefit,
    codec: CandidateConditionedK64StateFeatureCodec,
    examples: Sequence[CandidateConditionedK64StateGainGradientExample],
) -> CandidateConditionedK64StaticAmplitudeControlFit:
    if not isinstance(refit, CandidateConditionedK64MeanKLRefit):
        raise TypeError("refit must be a candidate V4 mean-KL refit")
    refit.validate_integrity()
    if not isinstance(codec, CandidateConditionedK64StateFeatureCodec):
        raise TypeError("codec must be a state feature codec")
    codec.validate_integrity()
    values = _validated_gradient_values(examples, codec=codec)
    if codec.held_family_id != refit.held_family_id:
        raise ValueError("state codec and V4 refit held families differ")
    stats = _gradient_statistics(values)
    if not torch.equal(
        stats["feature_gram"], codec.standardized_feature_gram
    ):
        raise ValueError("state gradient features do not replay the fitted codec")
    return CandidateConditionedK64StaticAmplitudeControlFit(
        held_family_id=refit.held_family_id,
        training_family_ids=codec.training_family_ids,
        training_example_ids=codec.training_example_ids,
        training_example_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        refit_artifact_sha256=refit.artifact_sha256,
        mean_gain_delta=refit.mean_proposed_gains_tensor() - 1.0,
        mean_gradient=float(stats["static_gradient"]),
        fisher_energy=float(stats["static_energy"]),
    )


def _field_fit_is_identifiable(
    value: CandidateConditionedK64StateGainFieldFit,
) -> bool:
    return (
        value.feature_rank == STATE_FEATURE_RANK
        and value.standardized_design_rank == STATE_FEATURE_RANK
        and value.feature_condition <= STATE_STANDARDIZED_CONDITION_MAXIMUM
        and value.standardized_design_condition
        <= STATE_STANDARDIZED_CONDITION_MAXIMUM
    )


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64InnerFamilyAnalyticRecord:
    """One inner-family analytic generalization result within an outer fold."""

    full_field_fit: CandidateConditionedK64StateGainFieldFit = field(repr=False)
    inner_field_fit: CandidateConditionedK64StateGainFieldFit = field(repr=False)
    inner_static_control_fit: CandidateConditionedK64StaticAmplitudeControlFit = (
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
            self.full_field_fit, CandidateConditionedK64StateGainFieldFit
        ) or not isinstance(
            self.inner_field_fit, CandidateConditionedK64StateGainFieldFit
        ):
            raise TypeError("inner analytic record requires typed state field fits")
        if not isinstance(
            self.inner_static_control_fit,
            CandidateConditionedK64StaticAmplitudeControlFit,
        ):
            raise TypeError("inner analytic record requires a typed scalar control")
        if not isinstance(
            self.held_example,
            CandidateConditionedK64StateGainGradientExample,
        ):
            raise TypeError("inner analytic record requires typed held evidence")
        self.full_field_fit.validate_integrity()
        self.inner_field_fit.validate_integrity()
        self.inner_static_control_fit.validate_integrity()
        self.held_example.validate_integrity()
        outer = self.full_field_fit.held_family_id
        inner = self.held_example.family_id
        full_families = set(self.full_field_fit.training_family_ids)
        inner_families = set(self.inner_field_fit.training_family_ids)
        if (
            len(self.full_field_fit.training_family_ids)
            != _OUTER_TRAINING_FAMILY_COUNT
            or len(self.inner_field_fit.training_family_ids)
            != _INNER_TRAINING_FAMILY_COUNT
            or self.inner_field_fit.held_family_id != outer
            or self.inner_static_control_fit.held_family_id != outer
            or outer in full_families
            or outer in inner_families
            or inner not in full_families
            or inner in inner_families
            or full_families != inner_families | {inner}
            or self.full_field_fit.refit_artifact_sha256
            != self.inner_field_fit.refit_artifact_sha256
            or self.inner_field_fit.refit_artifact_sha256
            != self.inner_static_control_fit.refit_artifact_sha256
            or self.inner_field_fit.training_family_ids
            != self.inner_static_control_fit.training_family_ids
            or self.inner_field_fit.training_example_ids
            != self.inner_static_control_fit.training_example_ids
            or self.inner_field_fit.training_example_artifact_sha256s
            != self.inner_static_control_fit.training_example_artifact_sha256s
            or self.inner_field_fit.static_fisher_energy
            != self.inner_static_control_fit.fisher_energy
            or self.held_example.codec_artifact_sha256
            != self.inner_field_fit.codec_artifact_sha256
            or self.held_example.example_id
            in self.inner_field_fit.training_example_ids
            or self.held_example.example_id
            not in self.full_field_fit.training_example_ids
            or set(self.full_field_fit.training_example_ids)
            != set(self.inner_field_fit.training_example_ids)
            | {self.held_example.example_id}
            or not torch.equal(
                self.full_field_fit.mean_gain_delta,
                self.inner_field_fit.mean_gain_delta,
            )
            or not torch.equal(
                self.inner_field_fit.mean_gain_delta,
                self.inner_static_control_fit.mean_gain_delta,
            )
        ):
            raise ValueError("inner analytic evidence does not match its outer fold")
        object.__setattr__(self, "outer_held_family_id", outer)
        object.__setattr__(self, "inner_held_family_id", inner)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_INNER_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def state_predicted_incremental_derivative(self) -> float:
        held_gradient = self.held_example.state_tangent_design_tensor().mean(dim=0)
        return float(torch.dot(held_gradient, self.inner_field_fit.applied_weight))

    @property
    def scalar_predicted_incremental_derivative(self) -> float:
        held_gradient = float(self.held_example.static_tangent_tensor().mean())
        return held_gradient * self.inner_static_control_fit.applied_coefficient

    @property
    def codec_invariant_weight_cosine(self) -> float:
        return _cosine(
            self.full_field_fit.raw_projection_slope_tensor(),
            self.inner_field_fit.raw_projection_slope_tensor(),
        )

    @property
    def state_increment_is_negative(self) -> bool:
        return self.state_predicted_incremental_derivative < 0.0

    @property
    def inner_feature_and_design_identifiable(self) -> bool:
        return _field_fit_is_identifiable(self.inner_field_fit)

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "outer_held_family_id": self.outer_held_family_id,
            "inner_held_family_id": self.inner_held_family_id,
            "full_field_fit_artifact_sha256": (
                self.full_field_fit.artifact_sha256
            ),
            "inner_field_fit_artifact_sha256": (
                self.inner_field_fit.artifact_sha256
            ),
            "inner_static_control_fit_artifact_sha256": (
                self.inner_static_control_fit.artifact_sha256
            ),
            "held_example_id": self.held_example.example_id,
            "held_example_artifact_sha256": self.held_example.artifact_sha256,
            "inner_training_family_ids": self.inner_field_fit.training_family_ids,
            "state_predicted_incremental_derivative": (
                self.state_predicted_incremental_derivative
            ),
            "scalar_predicted_incremental_derivative": (
                self.scalar_predicted_incremental_derivative
            ),
            "state_increment_is_negative": self.state_increment_is_negative,
            "codec_invariant_weight_cosine": (
                self.codec_invariant_weight_cosine
            ),
            "inner_feature_rank": self.inner_field_fit.feature_rank,
            "inner_feature_condition": self.inner_field_fit.feature_condition,
            "inner_standardized_design_rank": (
                self.inner_field_fit.standardized_design_rank
            ),
            "inner_standardized_design_condition": (
                self.inner_field_fit.standardized_design_condition
            ),
            "inner_feature_and_design_identifiable": (
                self.inner_feature_and_design_identifiable
            ),
            "inner_field_no_op": self.inner_field_fit.no_op,
            "held_derivative_definition": (
                "post_trust_incremental_vs_fixed_v5_static_plus"
            ),
            "cosine_coordinates": "applied_weight_divided_by_codec_scale",
            "zero_slope_cosine": 0.0,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.full_field_fit.validate_integrity()
        self.inner_field_fit.validate_integrity()
        self.inner_static_control_fit.validate_integrity()
        self.held_example.validate_integrity()
        if (
            self.outer_held_family_id != self.full_field_fit.held_family_id
            or self.inner_held_family_id != self.held_example.family_id
            or _sha256(_INNER_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="inner analytic record")
        ):
            raise RuntimeError("inner analytic record payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StateGainFoldAnalyticRecord:
    """Seven inner-family checks and the full-seven fit for one outer fold."""

    full_field_fit: CandidateConditionedK64StateGainFieldFit = field(repr=False)
    full_static_control_fit: CandidateConditionedK64StaticAmplitudeControlFit = (
        field(repr=False)
    )
    inner_family_records: tuple[
        CandidateConditionedK64InnerFamilyAnalyticRecord, ...
    ] = field(repr=False)
    outer_held_family_id: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.full_field_fit, CandidateConditionedK64StateGainFieldFit
        ) or not isinstance(
            self.full_static_control_fit,
            CandidateConditionedK64StaticAmplitudeControlFit,
        ):
            raise TypeError("fold analytic record requires typed full fits")
        supplied = tuple(self.inner_family_records)
        if any(
            not isinstance(
                value, CandidateConditionedK64InnerFamilyAnalyticRecord
            )
            for value in supplied
        ):
            raise TypeError("fold analytic record requires typed inner records")
        records = tuple(sorted(supplied, key=lambda value: value.inner_held_family_id))
        self.full_field_fit.validate_integrity()
        self.full_static_control_fit.validate_integrity()
        for value in records:
            value.validate_integrity()
        outer = self.full_field_fit.held_family_id
        inner_ids = tuple(value.inner_held_family_id for value in records)
        if (
            len(records) != _OUTER_TRAINING_FAMILY_COUNT
            or len(set(inner_ids)) != len(inner_ids)
            or inner_ids != self.full_field_fit.training_family_ids
            or any(value.outer_held_family_id != outer for value in records)
            or any(
                value.full_field_fit.artifact_sha256
                != self.full_field_fit.artifact_sha256
                for value in records
            )
            or self.full_static_control_fit.held_family_id != outer
            or self.full_static_control_fit.training_family_ids
            != self.full_field_fit.training_family_ids
            or self.full_static_control_fit.training_example_ids
            != self.full_field_fit.training_example_ids
            or self.full_static_control_fit.training_example_artifact_sha256s
            != self.full_field_fit.training_example_artifact_sha256s
            or self.full_static_control_fit.refit_artifact_sha256
            != self.full_field_fit.refit_artifact_sha256
            or self.full_static_control_fit.fisher_energy
            != self.full_field_fit.static_fisher_energy
            or not torch.equal(
                self.full_static_control_fit.mean_gain_delta,
                self.full_field_fit.mean_gain_delta,
            )
        ):
            raise ValueError("fold analytic evidence is incomplete or mismatched")
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
        return _field_fit_is_identifiable(self.full_field_fit) and all(
            value.inner_feature_and_design_identifiable
            for value in self.inner_family_records
        )

    @property
    def residual_energy_gate_passed(self) -> bool:
        return (
            self.full_field_fit.residual_conditional_fisher_fraction
            >= STATE_RESIDUAL_FISHER_MINIMUM
        )

    @property
    def full_field_non_noop(self) -> bool:
        return not self.full_field_fit.no_op

    @property
    def negative_inner_derivative_count(self) -> int:
        return sum(
            value.state_increment_is_negative
            for value in self.inner_family_records
        )

    @property
    def negative_inner_local_gate_passed(self) -> bool:
        return self.negative_inner_derivative_count >= 4

    @property
    def state_inner_macro_derivative(self) -> float:
        return sum(
            value.state_predicted_incremental_derivative
            for value in self.inner_family_records
        ) / len(self.inner_family_records)

    @property
    def scalar_inner_macro_derivative(self) -> float:
        return sum(
            value.scalar_predicted_incremental_derivative
            for value in self.inner_family_records
        ) / len(self.inner_family_records)

    @property
    def state_beats_scalar_inner_macro(self) -> bool:
        return self.state_inner_macro_derivative < self.scalar_inner_macro_derivative

    @property
    def median_inner_full_weight_cosine(self) -> float:
        values = sorted(
            value.codec_invariant_weight_cosine
            for value in self.inner_family_records
        )
        return values[len(values) // 2]

    @property
    def cosine_stability_gate_passed(self) -> bool:
        return self.median_inner_full_weight_cosine >= 0.90

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "outer_held_family_id": self.outer_held_family_id,
            "full_field_fit_artifact_sha256": self.full_field_fit.artifact_sha256,
            "full_static_control_fit_artifact_sha256": (
                self.full_static_control_fit.artifact_sha256
            ),
            "inner_family_record_artifact_sha256s": tuple(
                value.artifact_sha256 for value in self.inner_family_records
            ),
            "inner_held_family_ids": tuple(
                value.inner_held_family_id for value in self.inner_family_records
            ),
            "full_feature_rank": self.full_field_fit.feature_rank,
            "full_feature_condition": self.full_field_fit.feature_condition,
            "full_standardized_design_rank": (
                self.full_field_fit.standardized_design_rank
            ),
            "full_standardized_design_condition": (
                self.full_field_fit.standardized_design_condition
            ),
            "feature_and_design_gate_passed": (
                self.feature_and_design_gate_passed
            ),
            "full_residual_conditional_fisher_fraction": (
                self.full_field_fit.residual_conditional_fisher_fraction
            ),
            "residual_energy_gate_passed": self.residual_energy_gate_passed,
            "full_field_non_noop": self.full_field_non_noop,
            "negative_inner_derivative_count": (
                self.negative_inner_derivative_count
            ),
            "negative_inner_local_gate_passed": (
                self.negative_inner_local_gate_passed
            ),
            "state_inner_macro_derivative": self.state_inner_macro_derivative,
            "scalar_inner_macro_derivative": self.scalar_inner_macro_derivative,
            "state_beats_scalar_inner_macro": self.state_beats_scalar_inner_macro,
            "median_inner_full_weight_cosine": (
                self.median_inner_full_weight_cosine
            ),
            "cosine_stability_gate_passed": self.cosine_stability_gate_passed,
            "raw_tensors_serialized": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.full_field_fit.validate_integrity()
        self.full_static_control_fit.validate_integrity()
        for value in self.inner_family_records:
            value.validate_integrity()
        if (
            self.outer_held_family_id != self.full_field_fit.held_family_id
            or _sha256(_FOLD_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="fold analytic record")
        ):
            raise RuntimeError("fold analytic record payload drifted")


@dataclass(frozen=True, slots=True)
class CandidateConditionedK64StateGainAnalyticScreen:
    """Eight-fold analytic capacity screen; never a serving selection."""

    fold_records: tuple[CandidateConditionedK64StateGainFoldAnalyticRecord, ...] = (
        field(repr=False)
    )
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        supplied = tuple(self.fold_records)
        if any(
            not isinstance(
                value, CandidateConditionedK64StateGainFoldAnalyticRecord
            )
            for value in supplied
        ):
            raise TypeError("analytic capacity screen requires typed fold records")
        records = tuple(sorted(supplied, key=lambda value: value.outer_held_family_id))
        for value in records:
            value.validate_integrity()
        outer_ids = tuple(value.outer_held_family_id for value in records)
        if (
            len(records) != _OUTER_FAMILY_COUNT
            or len(set(outer_ids)) != len(outer_ids)
            or any(
                set(value.full_field_fit.training_family_ids)
                != set(outer_ids) - {value.outer_held_family_id}
                for value in records
            )
        ):
            raise ValueError("analytic capacity screen requires eight outer folds")
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
        return sum(value.full_field_non_noop for value in self.fold_records)

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
    def state_beats_scalar_fold_count(self) -> int:
        return sum(
            value.state_beats_scalar_inner_macro for value in self.fold_records
        )

    @property
    def state_beats_scalar_gate_passed(self) -> bool:
        return self.state_beats_scalar_fold_count >= 6

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
            and self.state_beats_scalar_gate_passed
            and self.cosine_stability_gate_passed
        )

    @property
    def outcome(self) -> str:
        if not self.feature_and_design_gate_passed:
            return "fail_feature_or_design_identifiability"
        if not self.residual_energy_gate_passed:
            return "fail_residual_conditional_fisher_energy"
        if not self.non_noop_gate_passed:
            return "fail_non_noop_support"
        if not self.negative_inner_global_gate_passed:
            return "fail_global_inner_derivative_support"
        if not self.negative_inner_local_gate_passed:
            return "fail_fold_local_inner_derivative_support"
        if not self.state_beats_scalar_gate_passed:
            return "fail_state_vs_scalar_attribution"
        if not self.cosine_stability_gate_passed:
            return "fail_inner_full_weight_stability"
        return "capacity_supported_for_finite_validation"

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "fold_record_artifact_sha256s": tuple(
                value.artifact_sha256 for value in self.fold_records
            ),
            "outer_held_family_ids": tuple(
                value.outer_held_family_id for value in self.fold_records
            ),
            "feature_rank_required": STATE_FEATURE_RANK,
            "design_rank_required": STATE_FEATURE_RANK,
            "standardized_condition_maximum": (
                STATE_STANDARDIZED_CONDITION_MAXIMUM
            ),
            "feature_and_design_gate_passed": (
                self.feature_and_design_gate_passed
            ),
            "residual_conditional_fisher_minimum": (
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
            "state_beats_scalar_fold_count": self.state_beats_scalar_fold_count,
            "state_beats_scalar_required_fold_count": 6,
            "state_beats_scalar_gate_passed": (
                self.state_beats_scalar_gate_passed
            ),
            "cosine_stability_fold_count": self.cosine_stability_fold_count,
            "median_inner_full_weight_cosine_minimum": 0.90,
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
            "finite_tune_or_held_execution_performed": False,
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
        ) != _require_sha256(self.artifact_sha256, label="analytic capacity screen"):
            raise RuntimeError("analytic capacity screen payload drifted")


def build_candidate_conditioned_k64_inner_family_analytic_record(
    full_field_fit: CandidateConditionedK64StateGainFieldFit,
    inner_field_fit: CandidateConditionedK64StateGainFieldFit,
    inner_static_control_fit: CandidateConditionedK64StaticAmplitudeControlFit,
    held_example: CandidateConditionedK64StateGainGradientExample,
) -> CandidateConditionedK64InnerFamilyAnalyticRecord:
    return CandidateConditionedK64InnerFamilyAnalyticRecord(
        full_field_fit=full_field_fit,
        inner_field_fit=inner_field_fit,
        inner_static_control_fit=inner_static_control_fit,
        held_example=held_example,
    )


def build_candidate_conditioned_k64_state_gain_fold_analytic_record(
    full_field_fit: CandidateConditionedK64StateGainFieldFit,
    full_static_control_fit: CandidateConditionedK64StaticAmplitudeControlFit,
    inner_family_records: Sequence[
        CandidateConditionedK64InnerFamilyAnalyticRecord
    ],
) -> CandidateConditionedK64StateGainFoldAnalyticRecord:
    return CandidateConditionedK64StateGainFoldAnalyticRecord(
        full_field_fit=full_field_fit,
        full_static_control_fit=full_static_control_fit,
        inner_family_records=tuple(inner_family_records),
    )


def screen_candidate_conditioned_k64_state_gain_capacity(
    fold_records: Sequence[CandidateConditionedK64StateGainFoldAnalyticRecord],
) -> CandidateConditionedK64StateGainAnalyticScreen:
    return CandidateConditionedK64StateGainAnalyticScreen(
        fold_records=tuple(fold_records)
    )
