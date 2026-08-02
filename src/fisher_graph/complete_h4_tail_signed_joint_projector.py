"""Pure held-family endpoint signed-joint projector fitting.

The ordinary tail-Fisher ladder scores already chosen directions one at a
time.  This module implements the preceding *direction discovery* rung.  It
uses the signed compensation target and exact endpoint token VJPs to rotate
inside the deterministic complement of a frozen supported basis.

Raw residuals, VJPs, and compensation targets are hypothesis-use evidence.
They are consumed by the fit but are never serialized by the fit artifact.
The result contains only ambient directions, scalar diagnostics, and hashes
of the authenticated evidence that was used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
    canonical_orthogonal_complement_rows,
)


__all__ = [
    "CompleteH4TailSignedJointHeldFamilyFit",
    "CompleteH4TailSignedJointStep",
    "complete_h4_tail_signed_joint_prediction",
    "complete_h4_tail_signed_joint_scores",
    "fit_complete_h4_tail_signed_joint_held_family",
]


_FIT_DOMAIN = b"fisher-graph:complete-h4-tail-signed-joint-fit:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:complete-h4-tail-signed-joint-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STOP_REASONS = frozenset(
    {
        "requested_rank_reached",
        "complement_exhausted",
        "nonpositive_curvature",
        "numerical_rank_exhausted",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


def _float64_matrix(
    value: Tensor,
    *,
    label: str,
    allow_empty_rows: bool = False,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or value.shape[1] == 0
        or (value.shape[0] == 0 and not allow_empty_rows)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating matrix")
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .contiguous()
    )


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("hashed tensor must be finite and floating")
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    payload = tensor.numpy().astype("<f8", copy=False).tobytes(order="C")
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + _canonical_json_bytes(
            {"dtype": "float64-little-endian", "shape": tuple(tensor.shape)}
        )
        + payload
    ).hexdigest()


def _canonical_examples(
    examples: Iterable[CompleteH4TailEndpointExample],
) -> tuple[CompleteH4TailEndpointExample, ...]:
    values = tuple(examples)
    if not values or any(
        not isinstance(value, CompleteH4TailEndpointExample) for value in values
    ):
        raise TypeError("signed-joint examples must be nonempty endpoint records")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("signed-joint endpoint example ids must be unique")
    if len({value.width for value in ordered}) != 1:
        raise ValueError("signed-joint endpoint widths differ")
    return ordered


def _validated_supported_basis(value: Tensor) -> Tensor:
    basis = _float64_matrix(value, label="supported basis")
    if basis.shape[0] >= basis.shape[1]:
        raise ValueError("supported basis must leave a nonempty complement")
    if not torch.allclose(
        basis @ basis.T,
        torch.eye(basis.shape[0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("supported basis rows must be orthonormal")
    return basis


def _canonical_ambient_sign(coordinates: Tensor, complement: Tensor) -> Tensor:
    result = coordinates.clone()
    ambient = result @ complement
    pivot = int(ambient.abs().argmax())
    if float(ambient[pivot]) < 0.0:
        result.neg_()
    return result.contiguous()


@dataclass(frozen=True, slots=True)
class CompleteH4TailSignedJointStep:
    """Scalar receipt for one nested signed-joint direction."""

    index: int
    selected_eigenvalue: float
    eigenvalue_tolerance: float
    gain: float
    gain_epsilon: float
    q_second_moment: float
    rmse_before: float
    rmse_after: float

    def __post_init__(self) -> None:
        values = (
            self.selected_eigenvalue,
            self.eigenvalue_tolerance,
            self.gain,
            self.gain_epsilon,
            self.q_second_moment,
            self.rmse_before,
            self.rmse_after,
        )
        if (
            type(self.index) is not int
            or self.index < 0
            or any(not math.isfinite(float(value)) for value in values)
            or self.selected_eigenvalue <= self.eigenvalue_tolerance
            or self.eigenvalue_tolerance < 0.0
            or not 0.0 <= self.gain <= 2.0
            or self.gain_epsilon < 0.0
            or self.q_second_moment <= 0.0
            or self.rmse_before < 0.0
            or self.rmse_after < 0.0
        ):
            raise ValueError("signed-joint step is invalid")
        scale = max(self.rmse_before, self.rmse_after, 1.0)
        tolerance = 512.0 * torch.finfo(torch.float64).eps * scale
        if self.rmse_after > self.rmse_before + tolerance:
            raise ValueError("signed-joint step increased training RMSE")

    def metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "selected_eigenvalue": self.selected_eigenvalue,
            "eigenvalue_tolerance": self.eigenvalue_tolerance,
            "gain": self.gain,
            "gain_epsilon": self.gain_epsilon,
            "q_second_moment": self.q_second_moment,
            "rmse_before": self.rmse_before,
            "rmse_after": self.rmse_after,
        }


@dataclass(frozen=True, slots=True)
class CompleteH4TailSignedJointHeldFamilyFit:
    """Training-only nested signed-joint directions for one held-family fold."""

    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    supported_basis_sha256: str
    complement_basis_sha256: str
    ambient_width: int
    requested_max_directions: int
    ambient_directions: Tensor = field(repr=False)
    steps: tuple[CompleteH4TailSignedJointStep, ...]
    initial_rmse: float
    final_rmse: float
    stop_reason: str
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
            _require_sha256(value, label="training endpoint artifact")
            for value in self.training_example_artifact_sha256s
        )
        if (
            not families
            or families != tuple(sorted(set(families)))
            or held in families
            or not examples
            or examples != tuple(sorted(set(examples)))
            or len(evidence) != len(examples)
        ):
            raise ValueError("signed-joint held-family split is invalid")
        if type(self.ambient_width) is not int or self.ambient_width <= 1:
            raise ValueError("signed-joint ambient width is invalid")
        if (
            type(self.requested_max_directions) is not int
            or not 1 <= self.requested_max_directions <= 64
        ):
            raise ValueError("signed-joint requested maximum must be in [1, 64]")
        directions = _float64_matrix(
            self.ambient_directions,
            label="signed-joint ambient directions",
            allow_empty_rows=True,
        )
        if directions.shape[1] != self.ambient_width:
            raise ValueError("signed-joint direction width differs")
        steps = tuple(self.steps)
        if (
            len(steps) != directions.shape[0]
            or len(steps) > self.requested_max_directions
            or any(
                not isinstance(step, CompleteH4TailSignedJointStep)
                or step.index != index
                for index, step in enumerate(steps)
            )
        ):
            raise ValueError("signed-joint directions and steps differ")
        if directions.shape[0] and not torch.allclose(
            directions @ directions.T,
            torch.eye(directions.shape[0], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("signed-joint ambient directions are not orthonormal")
        for row in directions:
            pivot = int(row.abs().argmax())
            if float(row[pivot]) <= 0.0:
                raise ValueError("signed-joint direction sign is not canonical")
        initial_rmse = float(self.initial_rmse)
        final_rmse = float(self.final_rmse)
        if (
            not math.isfinite(initial_rmse)
            or not math.isfinite(final_rmse)
            or initial_rmse < 0.0
            or final_rmse < 0.0
            or (steps and initial_rmse != steps[0].rmse_before)
            or (steps and final_rmse != steps[-1].rmse_after)
            or (not steps and initial_rmse != final_rmse)
            or self.stop_reason not in _STOP_REASONS
        ):
            raise ValueError("signed-joint RMSE history or stop reason is invalid")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(
            self,
            "supported_basis_sha256",
            _require_sha256(self.supported_basis_sha256, label="supported basis"),
        )
        object.__setattr__(
            self,
            "complement_basis_sha256",
            _require_sha256(self.complement_basis_sha256, label="complement basis"),
        )
        object.__setattr__(self, "ambient_directions", directions)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "initial_rmse", initial_rmse)
        object.__setattr__(self, "final_rmse", final_rmse)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def rank(self) -> int:
        return int(self.ambient_directions.shape[0])

    @property
    def gains(self) -> tuple[float, ...]:
        return tuple(step.gain for step in self.steps)

    def directions_tensor(self) -> Tensor:
        self.validate_integrity()
        return self.ambient_directions.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "held_family_id": self.held_family_id,
            "training_family_ids": self.training_family_ids,
            "training_example_ids": self.training_example_ids,
            "training_example_artifact_sha256s": (
                self.training_example_artifact_sha256s
            ),
            "supported_basis_sha256": self.supported_basis_sha256,
            "complement_basis_sha256": self.complement_basis_sha256,
            "ambient_width": self.ambient_width,
            "requested_max_directions": self.requested_max_directions,
            "ambient_directions_shape": tuple(self.ambient_directions.shape),
            "ambient_directions_sha256": _tensor_sha256(
                self.ambient_directions
            ),
            "steps": tuple(step.metadata() for step in self.steps),
            "initial_rmse": self.initial_rmse,
            "final_rmse": self.final_rmse,
            "stop_reason": self.stop_reason,
            "objective": (
                "signed_compensation_endpoint_quadratic_joint_projector"
            ),
            "weighting": "equal_family_then_equal_prompt_then_equal_token",
            "direction_frame": "deterministic_canonical_null_supported_basis",
            "gain_constraint": "closed_interval_0_2",
            "held_family_used_for_direction_gain_or_stop": False,
            "raw_evidence_serialized": False,
            "endpoint_first_order_hypothesis_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        directions = self.ambient_directions
        if (
            directions.dtype != torch.float64
            or directions.device.type != "cpu"
            or directions.requires_grad
            or not directions.is_contiguous()
            or not bool(torch.isfinite(directions).all())
            or (directions.shape[0] > 0 and not torch.allclose(
                directions @ directions.T,
                torch.eye(directions.shape[0], dtype=torch.float64),
                rtol=0.0,
                atol=1.0e-9,
            ))
            or _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256, label="signed-joint fit artifact"
            )
        ):
            raise RuntimeError("signed-joint fit payload drifted")


@dataclass(frozen=True, slots=True)
class _CoordinateEvidence:
    example: CompleteH4TailEndpointExample
    amplitudes: Tensor
    gradients: Tensor


def _coordinate_evidence(
    examples: Sequence[CompleteH4TailEndpointExample],
    complement: Tensor,
) -> tuple[_CoordinateEvidence, ...]:
    return tuple(
        _CoordinateEvidence(
            example=example,
            amplitudes=(example.residual_rows @ complement.T).contiguous(),
            gradients=torch.einsum(
                "trw,cw->trc", example.token_h4_gradients, complement
            ).contiguous(),
        )
        for example in examples
    )


def _nested_expectation(
    evidence: Sequence[_CoordinateEvidence],
    values: Mapping[str, Tensor],
) -> Tensor:
    """Average tokens, then prompts, then families with equal weights."""

    by_family: dict[str, list[Tensor]] = defaultdict(list)
    for item in evidence:
        value = values[item.example.example_id]
        if value.shape[0] != item.example.supervised_tokens:
            raise RuntimeError("signed-joint token statistic length drifted")
        by_family[item.example.family_id].append(value.mean(dim=0))
    family_means = tuple(
        torch.stack(by_family[family]).mean(dim=0)
        for family in sorted(by_family)
    )
    return torch.stack(family_means).mean(dim=0).contiguous()


def _signed_joint_operator(
    evidence: Sequence[_CoordinateEvidence],
    residuals: Mapping[str, Tensor],
) -> Tensor:
    """Return the exact family/prompt/token-equal signed joint operator.

    The tempting implementation constructs one ``[C, C]`` outer product per
    token.  At the real complement width that is a prohibitive ``[T,C,C]``
    temporary.  Linearity lets us contract the signed token residual into the
    gradients first:

    ``mean_t rho_t sum_r A_r.T G_tr``
    ``= A.T @ mean_t(rho_t G_t)``.

    We still average tokens within each prompt, prompts within each family,
    and families equally; only the order of the algebraic contraction changes.
    """

    family_sums: dict[str, Tensor] = {}
    family_prompt_counts: dict[str, int] = defaultdict(int)
    for item in evidence:
        residual = residuals[item.example.example_id]
        if residual.shape != (item.example.supervised_tokens,):
            raise RuntimeError("signed-joint compensation residual length drifted")
        weighted_gradient = torch.einsum(
            "t,trc->rc", residual, item.gradients
        ) / float(item.example.supervised_tokens)
        prompt_operator = item.amplitudes.T @ weighted_gradient
        prompt_operator = 0.5 * (prompt_operator + prompt_operator.T)
        family = item.example.family_id
        if family in family_sums:
            family_sums[family].add_(prompt_operator)
        else:
            family_sums[family] = prompt_operator.clone()
        family_prompt_counts[family] += 1
    operator = torch.zeros_like(next(iter(family_sums.values())))
    for family in sorted(family_sums):
        operator.add_(family_sums[family] / family_prompt_counts[family])
    operator.div_(len(family_sums))
    return (0.5 * (operator + operator.T)).contiguous()


def _low_rank_deflate_symmetric_operator(
    operator: Tensor,
    prior_rows: Tensor,
) -> Tensor:
    """Return ``(I - U.T U) M (I - U.T U)`` without dense projectors.

    Expanding the product gives

    ``M - U.T(UM) - (MU.T)U + U.T(UMU.T)U``.

    For ``k`` prior rows and complement width ``C`` this uses low-rank
    ``O(k C^2 + k^2 C)`` contractions instead of two ``C x C`` products.
    It is algebraically identical to dense projection.  The changed
    parenthesization can change floating-point results in their final bits, so
    callers must expect tolerance equivalence rather than a universal promise
    of bitwise identity.  Canonical ordering and signs remain deterministic
    for a fixed implementation and input.
    """

    if (
        not isinstance(operator, Tensor)
        or operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.dtype != torch.float64
        or operator.device.type != "cpu"
        or not bool(torch.isfinite(operator).all())
    ):
        raise ValueError("deflated operator must be a finite CPU float64 square")
    if (
        not isinstance(prior_rows, Tensor)
        or prior_rows.ndim != 2
        or prior_rows.shape[1] != operator.shape[0]
        or prior_rows.dtype != torch.float64
        or prior_rows.device.type != "cpu"
        or not bool(torch.isfinite(prior_rows).all())
    ):
        raise ValueError("deflation rows must match the operator")
    if not torch.allclose(
        operator, operator.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("deflated operator must be symmetric")
    if prior_rows.shape[0] == 0:
        return operator.clone().contiguous()
    if not torch.allclose(
        prior_rows @ prior_rows.T,
        torch.eye(prior_rows.shape[0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("deflation rows must be orthonormal")
    u_m = prior_rows @ operator
    left = prior_rows.T @ u_m
    middle = prior_rows.T @ ((u_m @ prior_rows.T) @ prior_rows)
    result = operator - left - left.T + middle
    return (0.5 * (result + result.T)).contiguous()


def _scores(item: _CoordinateEvidence, direction: Tensor) -> Tensor:
    amplitudes = item.amplitudes @ direction
    gradients = torch.einsum("trc,c->tr", item.gradients, direction)
    return torch.einsum("r,tr->t", amplitudes, gradients).contiguous()


def _weighted_rmse(
    evidence: Sequence[_CoordinateEvidence],
    residuals: Mapping[str, Tensor],
) -> float:
    squared = {
        key: value.square()
        for key, value in residuals.items()
    }
    return math.sqrt(float(_nested_expectation(evidence, squared)))


def fit_complete_h4_tail_signed_joint_held_family(
    examples: Iterable[CompleteH4TailEndpointExample],
    *,
    supported_basis: Tensor,
    held_family_id: str,
    max_directions: int = 64,
) -> CompleteH4TailSignedJointHeldFamilyFit:
    """Greedily fit a nested signed-joint projector without held evidence.

    For a direction ``u`` the token response is
    ``q_t(u) = sum_r (A_r @ u) (G_tr @ u)``.  Each step forms the symmetric
    signed operator from the current compensation residual, deflates already
    selected directions, takes the largest positive eigenvector, and fits a
    nonnegative gain in ``[0, 2]``.
    """

    values = _canonical_examples(examples)
    held = _identifier(held_family_id, label="held_family_id")
    if held not in {value.family_id for value in values}:
        raise ValueError("held family is absent from signed-joint evidence")
    if type(max_directions) is not int or not 1 <= max_directions <= 64:
        raise ValueError("max_directions must be in [1, 64]")
    training = tuple(value for value in values if value.family_id != held)
    training_families = tuple(sorted({value.family_id for value in training}))
    if len(training_families) < 2:
        raise ValueError("signed-joint fold requires at least two training families")
    supported = _validated_supported_basis(supported_basis)
    if supported.shape[1] != values[0].width:
        raise ValueError("supported basis width differs from signed-joint evidence")
    complement = canonical_orthogonal_complement_rows(supported)
    evidence = _coordinate_evidence(training, complement)
    residuals = {
        item.example.example_id: item.example.compensation_target.clone()
        for item in evidence
    }
    coordinate_directions: list[Tensor] = []
    ambient_directions: list[Tensor] = []
    steps: list[CompleteH4TailSignedJointStep] = []
    initial_rmse = _weighted_rmse(evidence, residuals)
    dimension = int(complement.shape[0])
    target_rank = min(max_directions, dimension)
    total_tokens = sum(item.example.supervised_tokens for item in evidence)
    machine_epsilon = torch.finfo(torch.float64).eps
    stop_reason = "requested_rank_reached"

    for index in range(target_rank):
        operator = _signed_joint_operator(evidence, residuals)
        if coordinate_directions:
            prior = torch.stack(coordinate_directions)
            operator = _low_rank_deflate_symmetric_operator(operator, prior)
        operator_scale = float(operator.abs().max())
        eigenvalue_tolerance = (
            256.0 * machine_epsilon * max(dimension, 1) * operator_scale
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(operator)
        selected_eigenvalue = float(eigenvalues[-1])
        if selected_eigenvalue <= eigenvalue_tolerance:
            stop_reason = "nonpositive_curvature"
            break
        direction = eigenvectors[:, -1].clone()
        for _ in range(2):
            for prior_direction in coordinate_directions:
                direction -= prior_direction * torch.dot(prior_direction, direction)
        norm = float(torch.linalg.vector_norm(direction))
        rank_tolerance = (
            256.0 * machine_epsilon * max(dimension, 1)
        )
        if norm <= rank_tolerance:
            stop_reason = "numerical_rank_exhausted"
            break
        direction /= norm
        direction = _canonical_ambient_sign(direction, complement)
        q_values = {
            item.example.example_id: _scores(item, direction)
            for item in evidence
        }
        numerator = float(
            _nested_expectation(
                evidence,
                {
                    key: residuals[key] * q_values[key]
                    for key in residuals
                },
            )
        )
        q_second_moment = float(
            _nested_expectation(
                evidence,
                {key: value.square() for key, value in q_values.items()},
            )
        )
        if q_second_moment <= 0.0:
            stop_reason = "numerical_rank_exhausted"
            break
        gain_epsilon = (
            256.0
            * machine_epsilon
            * max(dimension, len(evidence), total_tokens, 1)
            * q_second_moment
        )
        gain = min(max(numerator / (q_second_moment + gain_epsilon), 0.0), 2.0)
        rmse_before = _weighted_rmse(evidence, residuals)
        for key in residuals:
            residuals[key] = (residuals[key] - gain * q_values[key]).contiguous()
        rmse_after = _weighted_rmse(evidence, residuals)
        coordinate_directions.append(direction.contiguous())
        ambient = (direction @ complement).contiguous()
        ambient_directions.append(ambient)
        steps.append(
            CompleteH4TailSignedJointStep(
                index=index,
                selected_eigenvalue=selected_eigenvalue,
                eigenvalue_tolerance=eigenvalue_tolerance,
                gain=gain,
                gain_epsilon=gain_epsilon,
                q_second_moment=q_second_moment,
                rmse_before=rmse_before,
                rmse_after=rmse_after,
            )
        )
    else:
        stop_reason = (
            "complement_exhausted"
            if target_rank == dimension and max_directions >= dimension
            else "requested_rank_reached"
        )

    if ambient_directions:
        directions_tensor = torch.stack(ambient_directions).contiguous()
    else:
        directions_tensor = torch.empty(
            (0, supported.shape[1]), dtype=torch.float64
        )
    if directions_tensor.shape[0] and not torch.allclose(
        directions_tensor @ supported.T,
        torch.zeros(
            directions_tensor.shape[0], supported.shape[0], dtype=torch.float64
        ),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError("signed-joint directions left null(supported basis)")
    return CompleteH4TailSignedJointHeldFamilyFit(
        held_family_id=held,
        training_family_ids=training_families,
        training_example_ids=tuple(item.example.example_id for item in evidence),
        training_example_artifact_sha256s=tuple(
            item.example.artifact_sha256 for item in evidence
        ),
        supported_basis_sha256=_tensor_sha256(supported),
        complement_basis_sha256=_tensor_sha256(complement),
        ambient_width=int(supported.shape[1]),
        requested_max_directions=max_directions,
        ambient_directions=directions_tensor,
        steps=tuple(steps),
        initial_rmse=initial_rmse,
        final_rmse=_weighted_rmse(evidence, residuals),
        stop_reason=stop_reason,
    )


def complete_h4_tail_signed_joint_scores(
    example: CompleteH4TailEndpointExample,
    ambient_directions: Tensor,
) -> Tensor:
    """Return ``q_t(u_j)`` for arbitrary orthonormal ambient directions."""

    if not isinstance(example, CompleteH4TailEndpointExample):
        raise TypeError("example must be a complete-H4 endpoint record")
    example.validate_integrity()
    directions = _float64_matrix(
        ambient_directions,
        label="signed-joint scoring directions",
        allow_empty_rows=True,
    )
    if directions.shape[1] != example.width:
        raise ValueError("signed-joint scoring direction width differs")
    if directions.shape[0] and not torch.allclose(
        directions @ directions.T,
        torch.eye(directions.shape[0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("signed-joint scoring directions must be orthonormal")
    if directions.shape[0] == 0:
        return torch.empty(
            (example.supervised_tokens, 0), dtype=torch.float64
        )
    amplitudes = example.residual_rows @ directions.T
    gradients = torch.einsum(
        "trw,kw->trk", example.token_h4_gradients, directions
    )
    return torch.einsum("rk,trk->tk", amplitudes, gradients).contiguous()


def complete_h4_tail_signed_joint_prediction(
    example: CompleteH4TailEndpointExample,
    fit: CompleteH4TailSignedJointHeldFamilyFit,
    *,
    rank: int | None = None,
) -> Tensor:
    """Apply a fitted prefix as an endpoint first-order compensation estimate."""

    if not isinstance(example, CompleteH4TailEndpointExample):
        raise TypeError("example must be a complete-H4 endpoint record")
    example.validate_integrity()
    if not isinstance(fit, CompleteH4TailSignedJointHeldFamilyFit):
        raise TypeError("fit must be a signed-joint held-family fit")
    fit.validate_integrity()
    use_rank = fit.rank if rank is None else rank
    if type(use_rank) is not int or not 0 <= use_rank <= fit.rank:
        raise ValueError("signed-joint prediction rank is outside the fit")
    if use_rank == 0:
        return torch.zeros(example.supervised_tokens, dtype=torch.float64)
    scores = complete_h4_tail_signed_joint_scores(
        example, fit.directions_tensor()[:use_rank]
    )
    gains = torch.tensor(fit.gains[:use_rank], dtype=torch.float64)
    return (scores @ gains).contiguous()
