"""Pure held-family tail-basis and endpoint token-Fisher primitives.

This module is deliberately model agnostic.  It fits a *full* orthogonal
complement to a frozen supported basis from training families only, orders
that complement with exact per-token endpoint VJPs from those same training
families, and exposes finite prefix projections for held-family evaluation.

The input tensors are hypothesis-use research evidence.  ``metadata`` methods
publish hashes and scalar summaries only; raw residuals, gradients, token
scores, and basis coefficients are intentionally not serialized.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
    fit_complete_h4_projection_basis,
)


__all__ = [
    "CompleteH4TailEndpointExample",
    "CompleteH4TailHeldFamilyFit",
    "canonical_orthogonal_complement_rows",
    "complete_h4_tail_gate_scores",
    "fit_complete_h4_tail_held_family",
    "project_complete_h4_tail_prefix",
    "project_complete_h4_tail_rows",
]


_EXAMPLE_DOMAIN = b"fisher-graph:complete-h4-tail-endpoint-example:v1\0"
_FIT_DOMAIN = b"fisher-graph:complete-h4-tail-held-family-fit:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:complete-h4-tail-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _float64(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or 0 in value.shape
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating tensor")
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .contiguous()
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


def _canonicalize_row_signs(rows: Tensor) -> Tensor:
    result = rows.clone()
    for index in range(result.shape[0]):
        pivot = int(result[index].abs().argmax())
        if float(result[index, pivot]) < 0.0:
            result[index].neg_()
    return result.contiguous()


def _validated_supported_basis(value: Tensor) -> Tensor:
    rows = _float64(value, label="supported basis", ndim=2)
    if rows.shape[0] >= rows.shape[1]:
        raise ValueError("supported basis must leave a nonempty complement")
    identity = torch.eye(rows.shape[0], dtype=torch.float64)
    if not torch.allclose(rows @ rows.T, identity, rtol=0.0, atol=1.0e-10):
        raise ValueError("supported basis rows must be orthonormal")
    return rows


def canonical_orthogonal_complement_rows(supported_basis: Tensor) -> Tensor:
    """Return a deterministic coordinate-ordered basis for ``null(D)``.

    Coordinate axes are projected in ascending order and accepted with
    exactly two modified-Gram-Schmidt passes.  Consequently the result is a
    function of the supported subspace projector, not an arbitrary SVD
    rotation in its nullspace.
    """

    supported = _validated_supported_basis(supported_basis)
    width = int(supported.shape[1])
    dimension = width - int(supported.shape[0])
    tolerance = 64.0 * torch.finfo(torch.float64).eps * max(width, 1)
    accepted: list[Tensor] = []
    for coordinate in range(width):
        candidate = torch.zeros(width, dtype=torch.float64)
        candidate[coordinate] = 1.0
        for _ in range(2):
            candidate -= (candidate @ supported.T) @ supported
            for prior in accepted:
                candidate -= prior * torch.dot(prior, candidate)
        norm = float(torch.linalg.vector_norm(candidate))
        if norm <= tolerance:
            continue
        accepted.append(candidate / norm)
        if len(accepted) == dimension:
            break
    if len(accepted) != dimension:
        raise RuntimeError("could not construct the complete orthogonal complement")
    result = _canonicalize_row_signs(torch.stack(accepted).contiguous())
    identity = torch.eye(dimension, dtype=torch.float64)
    if (
        not torch.allclose(result @ result.T, identity, rtol=0.0, atol=1.0e-10)
        or not torch.allclose(
            result @ supported.T,
            torch.zeros(dimension, supported.shape[0], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise RuntimeError("constructed complement lost orthogonality")
    return result


def project_complete_h4_tail_rows(
    residual_rows: Tensor,
    supported_basis: Tensor,
) -> Tensor:
    """Return ``(I - D.T D) R`` in row-vector convention."""

    supported = _validated_supported_basis(supported_basis)
    residual = _float64(residual_rows, label="complete-H4 residual", ndim=2)
    if residual.shape[1] != supported.shape[1]:
        raise ValueError("residual width differs from the supported basis")
    return (residual - (residual @ supported.T) @ supported).contiguous()


@dataclass(frozen=True, slots=True)
class CompleteH4TailEndpointExample:
    """Ephemeral raw evidence for one endpoint-VJP prompt."""

    example_id: str
    family_id: str
    residual_rows: Tensor = field(repr=False)
    token_h4_gradients: Tensor = field(repr=False)
    compensation_target: Tensor = field(repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        example_id = _identifier(self.example_id, label="example_id")
        family_id = _identifier(self.family_id, label="family_id")
        residual = _float64(
            self.residual_rows, label="tail endpoint residual", ndim=2
        )
        gradients = _float64(
            self.token_h4_gradients,
            label="tail endpoint token gradients",
            ndim=3,
        )
        target = _float64(
            self.compensation_target,
            label="tail endpoint compensation target",
            ndim=1,
        )
        if (
            gradients.shape[0] != target.shape[0]
            or gradients.shape[1:] != residual.shape
        ):
            raise ValueError("endpoint residual, token gradients, and target differ")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "residual_rows", residual)
        object.__setattr__(self, "token_h4_gradients", gradients)
        object.__setattr__(self, "compensation_target", target)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EXAMPLE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def width(self) -> int:
        return int(self.residual_rows.shape[1])

    @property
    def supervised_tokens(self) -> int:
        return int(self.compensation_target.shape[0])

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "residual_shape": tuple(self.residual_rows.shape),
            "supervised_token_count": self.supervised_tokens,
            "residual_sha256": _tensor_sha256(self.residual_rows),
            "token_h4_gradients_sha256": _tensor_sha256(
                self.token_h4_gradients
            ),
            "compensation_target_sha256": _tensor_sha256(
                self.compensation_target
            ),
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            self.residual_rows.dtype != torch.float64
            or self.residual_rows.device.type != "cpu"
            or self.residual_rows.requires_grad
            or not self.residual_rows.is_contiguous()
            or self.token_h4_gradients.dtype != torch.float64
            or self.token_h4_gradients.device.type != "cpu"
            or self.token_h4_gradients.requires_grad
            or not self.token_h4_gradients.is_contiguous()
            or self.compensation_target.dtype != torch.float64
            or self.compensation_target.device.type != "cpu"
            or self.compensation_target.requires_grad
            or not self.compensation_target.is_contiguous()
            or not bool(torch.isfinite(self.residual_rows).all())
            or not bool(torch.isfinite(self.token_h4_gradients).all())
            or not bool(torch.isfinite(self.compensation_target).all())
            or self.token_h4_gradients.shape[0]
            != self.compensation_target.shape[0]
            or self.token_h4_gradients.shape[1:] != self.residual_rows.shape
            or _sha256(_EXAMPLE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256, label="endpoint example artifact"
            )
        ):
            raise RuntimeError("tail endpoint example payload drifted")


def _canonical_examples(
    examples: Iterable[CompleteH4TailEndpointExample],
) -> tuple[CompleteH4TailEndpointExample, ...]:
    values = tuple(examples)
    if not values or any(
        not isinstance(value, CompleteH4TailEndpointExample) for value in values
    ):
        raise TypeError("tail endpoint examples must be nonempty typed records")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("tail endpoint example ids must be unique")
    if len({value.width for value in ordered}) != 1:
        raise ValueError("tail endpoint widths differ")
    return ordered


def complete_h4_tail_gate_scores(
    example: CompleteH4TailEndpointExample,
    basis_rows: Tensor,
) -> Tensor:
    """Contract exact token VJPs with prompt-local residual mode fields."""

    if not isinstance(example, CompleteH4TailEndpointExample):
        raise TypeError("example must be a tail endpoint record")
    example.validate_integrity()
    basis = _float64(basis_rows, label="tail gate basis", ndim=2)
    if basis.shape[1] != example.width:
        raise ValueError("tail gate basis width differs")
    identity = torch.eye(basis.shape[0], dtype=torch.float64)
    if not torch.allclose(basis @ basis.T, identity, rtol=0.0, atol=1.0e-10):
        raise ValueError("tail gate basis rows must be orthonormal")
    amplitudes = example.residual_rows @ basis.T
    gradient_coordinates = torch.einsum(
        "trw,kw->trk", example.token_h4_gradients, basis
    )
    return torch.einsum("rk,trk->tk", amplitudes, gradient_coordinates).contiguous()


def _family_equal_token_fisher(
    examples: Sequence[CompleteH4TailEndpointExample],
    basis_rows: Tensor,
) -> Tensor:
    by_family: dict[str, list[Tensor]] = defaultdict(list)
    for example in examples:
        scores = complete_h4_tail_gate_scores(example, basis_rows)
        by_family[example.family_id].append(scores.square().mean(dim=0))
    family_means = tuple(
        torch.stack(by_family[family]).mean(dim=0)
        for family in sorted(by_family)
    )
    return torch.stack(family_means).mean(dim=0).contiguous()


@dataclass(frozen=True, slots=True)
class CompleteH4TailHeldFamilyFit:
    """Training-only full complement and endpoint token-Fisher order."""

    held_family_id: str
    training_family_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    training_example_artifact_sha256s: tuple[str, ...]
    supported_basis_sha256: str
    complement_basis_sha256: str
    fitted_basis_rows: Tensor = field(repr=False)
    residual_eigenvalues: tuple[float, ...]
    token_fisher_relevance: tuple[float, ...]
    token_fisher_order: tuple[int, ...]
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
            _require_sha256(value, label="training example artifact")
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
            raise ValueError("held-family training split is invalid")
        supported_sha256 = _require_sha256(
            self.supported_basis_sha256, label="supported basis"
        )
        complement_sha256 = _require_sha256(
            self.complement_basis_sha256, label="complement basis"
        )
        basis = _float64(self.fitted_basis_rows, label="fitted tail basis", ndim=2)
        rank, width = basis.shape
        if rank <= 0 or rank >= width:
            raise ValueError("fitted tail basis must be a proper complement")
        if not torch.allclose(
            basis @ basis.T,
            torch.eye(rank, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("fitted tail basis rows must be orthonormal")
        eigenvalues = tuple(float(value) for value in self.residual_eigenvalues)
        relevance = tuple(float(value) for value in self.token_fisher_relevance)
        order = tuple(self.token_fisher_order)
        if (
            len(eigenvalues) != rank
            or len(relevance) != rank
            or len(order) != rank
            or any(not math.isfinite(value) or value < 0.0 for value in eigenvalues)
            or any(not math.isfinite(value) or value < 0.0 for value in relevance)
            or sorted(order) != list(range(rank))
            or order
            != tuple(sorted(range(rank), key=lambda i: (-relevance[i], i)))
        ):
            raise ValueError("tail fit spectrum or Fisher ordering is invalid")
        object.__setattr__(self, "held_family_id", held)
        object.__setattr__(self, "training_family_ids", families)
        object.__setattr__(self, "training_example_ids", examples)
        object.__setattr__(self, "training_example_artifact_sha256s", evidence)
        object.__setattr__(self, "supported_basis_sha256", supported_sha256)
        object.__setattr__(self, "complement_basis_sha256", complement_sha256)
        object.__setattr__(self, "fitted_basis_rows", basis)
        object.__setattr__(self, "residual_eigenvalues", eigenvalues)
        object.__setattr__(self, "token_fisher_relevance", relevance)
        object.__setattr__(self, "token_fisher_order", order)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def rank(self) -> int:
        return int(self.fitted_basis_rows.shape[0])

    @property
    def width(self) -> int:
        return int(self.fitted_basis_rows.shape[1])

    def ordered_basis_rows(self) -> Tensor:
        self.validate_integrity()
        return self.fitted_basis_rows[list(self.token_fisher_order)].contiguous()

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
            "fitted_basis_shape": tuple(self.fitted_basis_rows.shape),
            "fitted_basis_sha256": _tensor_sha256(self.fitted_basis_rows),
            "residual_eigenvalues": self.residual_eigenvalues,
            "token_fisher_relevance": self.token_fisher_relevance,
            "token_fisher_order": self.token_fisher_order,
            "basis_fit_weighting": (
                "equal_family_then_equal_prompt_unweighted_tail_residual_"
                "covariance"
            ),
            "ordering_weighting": (
                "equal_family_then_equal_prompt_then_equal_token_endpoint_"
                "vjp_square"
            ),
            "prompt_mean_fisher_used_for_ordering": False,
            "held_family_used_for_fit_or_ordering": False,
            "full_complement_span_fitted": True,
            "truth_leaking_hypothesis_use_only": True,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        rank = int(self.fitted_basis_rows.shape[0])
        if (
            self.fitted_basis_rows.dtype != torch.float64
            or self.fitted_basis_rows.device.type != "cpu"
            or self.fitted_basis_rows.requires_grad
            or not self.fitted_basis_rows.is_contiguous()
            or not bool(torch.isfinite(self.fitted_basis_rows).all())
            or not torch.allclose(
                self.fitted_basis_rows @ self.fitted_basis_rows.T,
                torch.eye(rank, dtype=torch.float64),
                rtol=0.0,
                atol=1.0e-9,
            )
            or _sha256(_FIT_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="tail fit artifact")
        ):
            raise RuntimeError("held-family tail fit payload drifted")


def fit_complete_h4_tail_held_family(
    examples: Iterable[CompleteH4TailEndpointExample],
    *,
    supported_basis: Tensor,
    held_family_id: str,
) -> CompleteH4TailHeldFamilyFit:
    """Fit the full tail basis and token-Fisher order without held data."""

    values = _canonical_examples(examples)
    held = _identifier(held_family_id, label="held_family_id")
    all_families = {value.family_id for value in values}
    if held not in all_families:
        raise ValueError("held family is absent from endpoint examples")
    training = tuple(value for value in values if value.family_id != held)
    training_families = tuple(sorted({value.family_id for value in training}))
    if len(training_families) < 2:
        raise ValueError("tail fold requires at least two training families")
    supported = _validated_supported_basis(supported_basis)
    if supported.shape[1] != values[0].width:
        raise ValueError("supported basis width differs from endpoint examples")
    complement = canonical_orthogonal_complement_rows(supported)
    sequences: list[CompleteH4ProjectionFitSequence] = []
    training_tail_by_example: dict[str, Tensor] = {}
    for example in training:
        tail = project_complete_h4_tail_rows(example.residual_rows, supported)
        training_tail_by_example[example.example_id] = tail
        coordinates = (tail @ complement.T).contiguous()
        sequences.append(
            CompleteH4ProjectionFitSequence(
                example_id=example.example_id,
                family_id=example.family_id,
                residual_rows=coordinates,
            )
        )
    coordinate_fit = fit_complete_h4_projection_basis(
        sequences,
        max_rank=int(complement.shape[0]),
        fit_weighting="unweighted",
    )
    ambient_basis = (coordinate_fit.basis_tensor() @ complement).contiguous()
    complete_frame = torch.cat((supported, ambient_basis), dim=0)
    if complete_frame.shape != (supported.shape[1], supported.shape[1]):
        raise RuntimeError("supported and fitted tail bases do not fill the width")
    if not torch.allclose(
        complete_frame @ complete_frame.T,
        torch.eye(complete_frame.shape[0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError("[supported; fitted-tail] is not an orthonormal frame")
    # This is an explicit full-prefix sentinel over the actual omitted field,
    # not an inference from PCA rank or the eigensolver's return shape.
    for example in training:
        tail = training_tail_by_example[example.example_id]
        reconstructed = (tail @ ambient_basis.T) @ ambient_basis
        if not torch.allclose(
            reconstructed,
            tail,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise RuntimeError("full fitted complement did not reconstruct tail")
    relevance = _family_equal_token_fisher(training, ambient_basis)
    order = tuple(
        sorted(
            range(int(relevance.numel())),
            key=lambda index: (-float(relevance[index]), index),
        )
    )
    return CompleteH4TailHeldFamilyFit(
        held_family_id=held,
        training_family_ids=training_families,
        training_example_ids=tuple(value.example_id for value in training),
        training_example_artifact_sha256s=tuple(
            value.artifact_sha256 for value in training
        ),
        supported_basis_sha256=_tensor_sha256(supported),
        complement_basis_sha256=_tensor_sha256(complement),
        fitted_basis_rows=ambient_basis,
        residual_eigenvalues=coordinate_fit.residual_eigenvalues,
        token_fisher_relevance=tuple(float(value) for value in relevance),
        token_fisher_order=order,
    )


def project_complete_h4_tail_prefix(
    residual_rows: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    *,
    rank: int,
) -> Tensor:
    """Project a held residual onto a training-only Fisher-ordered prefix."""

    if not isinstance(fit, CompleteH4TailHeldFamilyFit):
        raise TypeError("fit must be a held-family tail fit")
    fit.validate_integrity()
    if type(rank) is not int or rank <= 0 or rank > fit.rank:
        raise ValueError("tail prefix rank is outside the fitted complement")
    residual = _float64(residual_rows, label="tail prefix residual", ndim=2)
    if residual.shape[1] != fit.width:
        raise ValueError("tail prefix residual width differs")
    directions = fit.ordered_basis_rows()[:rank]
    return ((residual @ directions.T) @ directions).contiguous()
