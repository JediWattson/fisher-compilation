"""Pure-Torch fitting math for the first shared-MLP-family rung.

The functions in this module do not load or execute Gemma.  They consume
already-aligned A-fit tensors and produce two kinds of deterministic evidence:

* a directed-pair score for the zero-residual shared-core comparison; and
* independently fitted ``U @ V`` factors for one MLP projection and one rank.

Block ordinals are deliberately absent from the pair score and its numerical
tie-break.  The selector uses them only to enforce the preregistered
non-adjacency eligibility rule.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

import torch
from torch import Tensor


PRIMARY_RESIDUAL_TARGET: Final = "primary_member_minus_anchor"
INDEPENDENT_MEMBER_TARGET: Final = "independent_member"
DEFAULT_RELATIVE_RIDGE: Final = 1.0e-4

_PROJECTION_EVIDENCE_DOMAIN = (
    b"fisher_graph:gemma3-mlp-family-projection-evidence:v1\0"
)
_PROJECTION_FACTORS_DOMAIN = (
    b"fisher_graph:gemma3-mlp-family-projection-factors:v1\0"
)
_PROJECTION_RECEIPT_DOMAIN = (
    b"fisher_graph:gemma3-mlp-family-projection-receipt:v1\0"
)
_PAIR_NUMERICAL_TENSOR_DOMAIN = (
    b"fisher_graph:gemma3-mlp-family-pair-numerical-tensors:v1\0"
)
_PAIR_SCORE_RECEIPT_DOMAIN = (
    b"fisher_graph:gemma3-mlp-family-pair-score-receipt:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_id(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} must be a nonempty canonical identifier")
    return value


def _finite_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating matrix")
    return value


def _normalized_fisher_weights(value: Tensor, *, row_count: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.shape[0] != row_count
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value < 0).any())
    ):
        raise ValueError(
            "row_fisher_weights must be a finite nonnegative row vector"
        )
    result = value.detach().to(device="cpu", dtype=torch.float64)
    mean = result.mean()
    if not bool(torch.isfinite(mean)) or float(mean) <= 0.0:
        raise ValueError("row_fisher_weights must contain positive energy")
    return (result / mean).contiguous()


def _update_with_canonical_float32_tensor(
    digest: "hashlib._Hash",
    value: Tensor,
) -> None:
    canonical = (
        value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    header = _canonical_json_bytes(
        {"dtype": "float32", "shape": list(canonical.shape)}
    )
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    raw = canonical.view(torch.uint8).numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, byteorder="big", signed=False))
    digest.update(raw)


def canonical_float32_tensor_sha256(
    *values: Tensor,
    domain: bytes = b"fisher_graph:canonical-float32-tensors:v1\0",
) -> str:
    """Hash tensor shapes and canonical contiguous float32 bytes in order."""

    if not values:
        raise ValueError("at least one tensor is required")
    digest = hashlib.sha256(domain)
    for value in values:
        _finite_matrix(value, label="digest tensor")
        _update_with_canonical_float32_tensor(digest, value)
    return digest.hexdigest()


def _canonicalize_svd_output_signs(
    left: Tensor,
    right: Tensor,
    *,
    component_count: int,
) -> None:
    """Canonicalize paired SVD signs using the output vector as authority."""

    for component in range(component_count):
        output_vector = right[component]
        pivot = int(output_vector.abs().argmax().item())
        if float(output_vector[pivot].item()) < 0.0:
            right[component].neg_()
            left[:, component].neg_()


@dataclass(frozen=True, slots=True)
class ProjectionResidualFitReceipt:
    """JSON-safe receipt for one projection, target, and requested rank."""

    projection_name: str
    target_kind: str
    row_count: int
    input_width: int
    output_width: int
    requested_rank: int
    numerical_rank: int
    effective_rank: int
    zero_padded_rank: int
    relative_ridge: float
    ridge_scale: float
    absolute_ridge: float
    numerical_rank_tolerance: float
    weighted_target_energy: float
    weighted_fit_sse: float
    weighted_relative_sse: float
    evidence_sha256: str
    factor_sha256: str
    stored_dtype: str
    solver_dtype: str = "torch.float64"
    factor_gauge: str = "sqrt_singular_value_canonical_output_sign"
    schema: str = "fisher_graph.gemma3_mlp_family.projection_fit_receipt.v1"
    format_version: int = 1

    def __post_init__(self) -> None:
        _require_id(self.projection_name, label="projection_name")
        if self.target_kind not in {
            PRIMARY_RESIDUAL_TARGET,
            INDEPENDENT_MEMBER_TARGET,
        }:
            raise ValueError("target_kind is unsupported")
        for label, value, minimum in (
            ("row_count", self.row_count, 1),
            ("input_width", self.input_width, 1),
            ("output_width", self.output_width, 1),
            ("requested_rank", self.requested_rank, 1),
            ("numerical_rank", self.numerical_rank, 0),
            ("effective_rank", self.effective_rank, 0),
            ("zero_padded_rank", self.zero_padded_rank, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        if self.effective_rank > self.numerical_rank:
            raise ValueError("effective_rank exceeds numerical_rank")
        if self.effective_rank + self.zero_padded_rank != self.requested_rank:
            raise ValueError("requested rank is not fully accounted")
        for label, value in (
            ("relative_ridge", self.relative_ridge),
            ("ridge_scale", self.ridge_scale),
            ("absolute_ridge", self.absolute_ridge),
            ("numerical_rank_tolerance", self.numerical_rank_tolerance),
            ("weighted_target_energy", self.weighted_target_energy),
            ("weighted_fit_sse", self.weighted_fit_sse),
            ("weighted_relative_sse", self.weighted_relative_sse),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")
        if self.relative_ridge <= 0.0 or self.absolute_ridge <= 0.0:
            raise ValueError("ridge values must be positive")
        _require_sha256(self.evidence_sha256, label="evidence_sha256")
        _require_sha256(self.factor_sha256, label="factor_sha256")
        if not self.stored_dtype.startswith("torch."):
            raise ValueError("stored_dtype must name a torch dtype")
        if self.solver_dtype != "torch.float64":
            raise ValueError("projection solver must use float64")
        if self.factor_gauge != "sqrt_singular_value_canonical_output_sign":
            raise ValueError("projection factor gauge drifted")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "projection_name": self.projection_name,
            "target_kind": self.target_kind,
            "row_count": self.row_count,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "requested_rank": self.requested_rank,
            "numerical_rank": self.numerical_rank,
            "effective_rank": self.effective_rank,
            "zero_padded_rank": self.zero_padded_rank,
            "relative_ridge": self.relative_ridge,
            "ridge_scale": self.ridge_scale,
            "absolute_ridge": self.absolute_ridge,
            "numerical_rank_tolerance": self.numerical_rank_tolerance,
            "weighted_target_energy": self.weighted_target_energy,
            "weighted_fit_sse": self.weighted_fit_sse,
            "weighted_relative_sse": self.weighted_relative_sse,
            "evidence_sha256": self.evidence_sha256,
            "factor_sha256": self.factor_sha256,
            "stored_dtype": self.stored_dtype,
            "solver_dtype": self.solver_dtype,
            "factor_gauge": self.factor_gauge,
        }

    @property
    def receipt_sha256(self) -> str:
        return _fingerprint(_PROJECTION_RECEIPT_DOMAIN, self.payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class ProjectionResidualFit:
    """Requested-shape factors plus their authenticated fit receipt."""

    U: Tensor
    V: Tensor
    receipt: ProjectionResidualFitReceipt

    def __post_init__(self) -> None:
        if (
            not isinstance(self.U, Tensor)
            or not isinstance(self.V, Tensor)
            or self.U.shape
            != (self.receipt.output_width, self.receipt.requested_rank)
            or self.V.shape
            != (self.receipt.requested_rank, self.receipt.input_width)
            or self.U.dtype != self.V.dtype
            or self.U.device != self.V.device
            or not self.U.is_floating_point()
            or not bool(torch.isfinite(self.U).all())
            or not bool(torch.isfinite(self.V).all())
        ):
            raise ValueError("projection factors do not match their receipt")
        observed = canonical_float32_tensor_sha256(
            self.U,
            self.V,
            domain=_PROJECTION_FACTORS_DOMAIN,
        )
        if observed != self.receipt.factor_sha256:
            raise ValueError("projection factor authentication failed")

    @property
    def left(self) -> Tensor:
        return self.U

    @property
    def right(self) -> Tensor:
        return self.V

    def __iter__(self) -> Iterator[object]:
        yield self.U
        yield self.V
        yield self.receipt


@dataclass(frozen=True, slots=True)
class MLPProjectionFactorCatalogFit(
    Mapping[str, tuple[Tensor, Tensor]]
):
    """Three-projection fit that is itself a compiler-compatible mapping."""

    fits: tuple[tuple[str, ProjectionResidualFit], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.fits)
        if not names or len(names) != len(set(names)):
            raise ValueError("projection fit names must be nonempty and unique")
        if any(
            not isinstance(fit, ProjectionResidualFit)
            or fit.receipt.projection_name != name
            for name, fit in self.fits
        ):
            raise ValueError("projection fit catalog is inconsistent")

    def __getitem__(self, name: str) -> tuple[Tensor, Tensor]:
        for observed_name, fit in self.fits:
            if observed_name == name:
                return fit.U, fit.V
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(name for name, _ in self.fits)

    def __len__(self) -> int:
        return len(self.fits)

    @property
    def projection_factors(self) -> dict[str, tuple[Tensor, Tensor]]:
        return {name: (fit.U, fit.V) for name, fit in self.fits}

    @property
    def receipts(self) -> dict[str, ProjectionResidualFitReceipt]:
        return {name: fit.receipt for name, fit in self.fits}

    def to_dict(self) -> dict[str, object]:
        return {
            "projection_order": [name for name, _ in self.fits],
            "projection_receipts": {
                name: fit.receipt.to_dict() for name, fit in self.fits
            },
        }


def fit_fisher_weighted_reduced_rank_projection(
    *,
    projection_name: str,
    design: Tensor,
    member_weight: Tensor,
    anchor_weight: Tensor | None,
    row_fisher_weights: Tensor,
    requested_rank: int | None = None,
    rank: int | None = None,
    independent_nonsharing: bool = False,
    relative_ridge: float = DEFAULT_RELATIVE_RIDGE,
) -> ProjectionResidualFit:
    """Fit one projection as requested-shape ``U[out,r] @ V[r,in]``.

    The weighted target is ``design @ (W_member - W_anchor).T`` for the
    primary shared-core residual and ``design @ W_member.T`` for the matched
    independent nonsharing control.  All linear algebra is performed on CPU
    in float64.  Factors are returned on the member weight's device and in its
    dtype; unavailable or numerically zero components are explicit zero
    columns/rows, so the physical shapes always encode the requested rank.
    """

    _require_id(projection_name, label="projection_name")
    design = _finite_matrix(design, label="design")
    member_weight = _finite_matrix(member_weight, label="member_weight")
    if design.shape[1] != member_weight.shape[1]:
        raise ValueError("design and member_weight input widths differ")
    if requested_rank is None:
        requested_rank = rank
    elif rank is not None and rank != requested_rank:
        raise ValueError("rank and requested_rank disagree")
    if type(requested_rank) is not int or requested_rank <= 0:
        raise ValueError("requested_rank must be a positive integer")
    if not isinstance(independent_nonsharing, bool):
        raise TypeError("independent_nonsharing must be bool")
    if (
        not isinstance(relative_ridge, (int, float))
        or isinstance(relative_ridge, bool)
        or not math.isfinite(float(relative_ridge))
        or float(relative_ridge) <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")

    if independent_nonsharing:
        target_weight = member_weight
        target_kind = INDEPENDENT_MEMBER_TARGET
    else:
        anchor_weight = _finite_matrix(anchor_weight, label="anchor_weight")
        if (
            anchor_weight.shape != member_weight.shape
            or anchor_weight.dtype != member_weight.dtype
        ):
            raise ValueError("anchor_weight must match member_weight")
        target_weight = member_weight - anchor_weight.to(
            device=member_weight.device
        )
        target_kind = PRIMARY_RESIDUAL_TARGET

    rows, input_width = design.shape
    output_width = member_weight.shape[0]
    weights = _normalized_fisher_weights(
        row_fisher_weights,
        row_count=rows,
    )
    design64 = design.detach().to(device="cpu", dtype=torch.float64)
    target_weight64 = target_weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    target64 = design64 @ target_weight64.T
    square_root_weight = weights.sqrt().unsqueeze(1)
    weighted_design = square_root_weight * design64
    weighted_target = square_root_weight * target64

    gram = weighted_design.T @ weighted_design
    ridge_scale = float(torch.trace(gram).item()) / input_width
    if not math.isfinite(ridge_scale) or ridge_scale <= 0.0:
        raise ValueError("weighted design has no positive finite energy")
    absolute_ridge = float(relative_ridge) * ridge_scale

    identity = torch.eye(input_width, dtype=torch.float64)
    ridge_gram = gram + absolute_ridge * identity
    cholesky = torch.linalg.cholesky(ridge_gram)
    full_coefficient = torch.cholesky_solve(
        weighted_design.T @ weighted_target,
        cholesky,
    )
    whitened_coefficient = cholesky.T @ full_coefficient
    left, singular, right = torch.linalg.svd(
        whitened_coefficient,
        full_matrices=False,
    )
    maximum_singular = float(singular[0].item()) if singular.numel() else 0.0
    rank_tolerance = (
        max(whitened_coefficient.shape)
        * torch.finfo(torch.float64).eps
        * maximum_singular
    )
    numerical_rank = int((singular > rank_tolerance).sum().item())
    effective_rank = min(requested_rank, numerical_rank)
    _canonicalize_svd_output_signs(
        left,
        right,
        component_count=effective_rank,
    )

    U64 = torch.zeros(output_width, requested_rank, dtype=torch.float64)
    V64 = torch.zeros(requested_rank, input_width, dtype=torch.float64)
    if effective_rank:
        square_root_singular = singular[:effective_rank].sqrt()
        U64[:, :effective_rank] = (
            right[:effective_rank].T * square_root_singular
        )
        inverse_cholesky = torch.linalg.solve_triangular(
            cholesky,
            identity,
            upper=False,
        )
        V64[:effective_rank] = (
            square_root_singular.unsqueeze(1)
            * (left[:, :effective_rank].T @ inverse_cholesky)
        )

    U = U64.to(device=member_weight.device, dtype=member_weight.dtype).contiguous()
    V = V64.to(device=member_weight.device, dtype=member_weight.dtype).contiguous()
    deployed_U64 = U.detach().to(device="cpu", dtype=torch.float64)
    deployed_V64 = V.detach().to(device="cpu", dtype=torch.float64)
    weighted_prediction = weighted_design @ deployed_V64.T @ deployed_U64.T
    weighted_target_energy = float(weighted_target.square().sum().item())
    weighted_fit_sse = float(
        (weighted_target - weighted_prediction).square().sum().item()
    )
    denominator = max(
        weighted_target_energy,
        torch.finfo(torch.float64).eps
        * max(1.0, weighted_target_energy),
    )
    weighted_relative_sse = weighted_fit_sse / denominator

    evidence_sha256 = canonical_float32_tensor_sha256(
        design,
        target64,
        row_fisher_weights.unsqueeze(1),
        domain=_PROJECTION_EVIDENCE_DOMAIN,
    )
    factor_sha256 = canonical_float32_tensor_sha256(
        U,
        V,
        domain=_PROJECTION_FACTORS_DOMAIN,
    )
    receipt = ProjectionResidualFitReceipt(
        projection_name=projection_name,
        target_kind=target_kind,
        row_count=rows,
        input_width=input_width,
        output_width=output_width,
        requested_rank=requested_rank,
        numerical_rank=numerical_rank,
        effective_rank=effective_rank,
        zero_padded_rank=requested_rank - effective_rank,
        relative_ridge=float(relative_ridge),
        ridge_scale=ridge_scale,
        absolute_ridge=absolute_ridge,
        numerical_rank_tolerance=rank_tolerance,
        weighted_target_energy=weighted_target_energy,
        weighted_fit_sse=weighted_fit_sse,
        weighted_relative_sse=weighted_relative_sse,
        evidence_sha256=evidence_sha256,
        factor_sha256=factor_sha256,
        stored_dtype=str(member_weight.dtype),
    )
    return ProjectionResidualFit(U=U, V=V, receipt=receipt)


def fit_mlp_projection_factor_catalog(
    *,
    projection_designs: Mapping[str, Tensor],
    member_projection_weights: Mapping[str, Tensor],
    anchor_projection_weights: Mapping[str, Tensor] | None,
    projection_row_fisher_weights: Mapping[str, Tensor] | Tensor,
    requested_rank: int | None = None,
    rank: int | None = None,
    independent_nonsharing: bool = False,
    relative_ridge: float = DEFAULT_RELATIVE_RIDGE,
    projection_order: Sequence[str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
) -> MLPProjectionFactorCatalogFit:
    """Fit a compiler-ready catalog without nesting factors across ranks.

    A single Fisher vector may be supplied when every projection shares row
    weights; otherwise pass one vector per projection.  Every call fits a
    fresh artifact for exactly one requested rank.
    """

    if (
        not isinstance(projection_designs, Mapping)
        or not isinstance(member_projection_weights, Mapping)
        or (
            anchor_projection_weights is not None
            and not isinstance(anchor_projection_weights, Mapping)
        )
        or isinstance(projection_order, (str, bytes))
        or not isinstance(projection_order, Sequence)
        or not projection_order
    ):
        raise TypeError("projection factor catalog inputs are invalid")
    names = tuple(projection_order)
    if (
        len(names) != len(set(names))
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("projection_order must contain unique names")
    expected = set(names)
    if set(projection_designs) != expected:
        raise ValueError("projection design catalog is incomplete")
    if set(member_projection_weights) != expected:
        raise ValueError("member projection-weight catalog is incomplete")
    if not independent_nonsharing and (
        anchor_projection_weights is None
        or set(anchor_projection_weights) != expected
    ):
        raise ValueError("anchor projection-weight catalog is incomplete")
    if isinstance(projection_row_fisher_weights, Mapping):
        if set(projection_row_fisher_weights) != expected:
            raise ValueError("projection Fisher-weight catalog is incomplete")
        fisher_by_name = projection_row_fisher_weights
    elif isinstance(projection_row_fisher_weights, Tensor):
        fisher_by_name = {
            name: projection_row_fisher_weights for name in names
        }
    else:
        raise TypeError("projection_row_fisher_weights has the wrong type")

    fitted: list[tuple[str, ProjectionResidualFit]] = []
    for name in names:
        fitted.append(
            (
                name,
                fit_fisher_weighted_reduced_rank_projection(
                    projection_name=name,
                    design=projection_designs[name],
                    member_weight=member_projection_weights[name],
                    anchor_weight=(
                        None
                        if anchor_projection_weights is None
                        else anchor_projection_weights[name]
                    ),
                    row_fisher_weights=fisher_by_name[name],
                    requested_rank=requested_rank,
                    rank=rank,
                    independent_nonsharing=independent_nonsharing,
                    relative_ridge=relative_ridge,
                ),
            )
        )
    return MLPProjectionFactorCatalogFit(fits=tuple(fitted))


@dataclass(frozen=True, slots=True)
class FamilyZeroResidualScore:
    family_id: str
    row_count: int
    normalized_weight_sum: float
    target_energy: float
    residual_sse: float
    relative_sse: float

    def __post_init__(self) -> None:
        _require_id(self.family_id, label="family_id")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("row_count must be positive")
        for label, value in (
            ("normalized_weight_sum", self.normalized_weight_sum),
            ("target_energy", self.target_energy),
            ("residual_sse", self.residual_sse),
            ("relative_sse", self.relative_sse),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{label} must be finite and nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "row_count": self.row_count,
            "normalized_weight_sum": self.normalized_weight_sum,
            "target_energy": self.target_energy,
            "residual_sse": self.residual_sse,
            "relative_sse": self.relative_sse,
        }


@dataclass(frozen=True, slots=True)
class ZeroResidualPairScore:
    """Equal-family score and label-free numerical tensor tie-break."""

    score: float
    numerical_tensor_sha256: str
    family_scores: tuple[FamilyZeroResidualScore, ...]
    row_count: int
    family_count: int
    schema: str = "fisher_graph.gemma3_mlp_family.zero_residual_pair_score.v1"
    format_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(float(self.score))
            or float(self.score) < 0.0
        ):
            raise ValueError("score must be finite and nonnegative")
        _require_sha256(
            self.numerical_tensor_sha256,
            label="numerical_tensor_sha256",
        )
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.family_count) is not int
            or self.family_count <= 0
            or len(self.family_scores) != self.family_count
            or sum(item.row_count for item in self.family_scores)
            != self.row_count
        ):
            raise ValueError("pair score inventory is inconsistent")
        ids = tuple(item.family_id for item in self.family_scores)
        if len(set(ids)) != len(ids):
            raise ValueError("pair score family ids must be unique")

    @property
    def tensor_digest_sha256(self) -> str:
        return self.numerical_tensor_sha256

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "score": self.score,
            "numerical_tensor_sha256": self.numerical_tensor_sha256,
            "row_count": self.row_count,
            "family_count": self.family_count,
            "family_aggregation": "equal_mean",
            "family_scores": [item.to_dict() for item in self.family_scores],
            "ordinal_or_distance_numerical_feature": False,
        }

    @property
    def receipt_sha256(self) -> str:
        return _fingerprint(_PAIR_SCORE_RECEIPT_DOMAIN, self.payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.payload(), "receipt_sha256": self.receipt_sha256}


def score_fisher_weighted_zero_residual_pair(
    *,
    aligned_design: Tensor,
    exact_member_delta: Tensor,
    anchor_core_delta: Tensor,
    row_fisher_weights: Tensor,
    family_ids: Sequence[str],
) -> ZeroResidualPairScore:
    """Score an anchor on the member rows, averaging families equally."""

    aligned_design = _finite_matrix(aligned_design, label="aligned_design")
    exact_member_delta = _finite_matrix(
        exact_member_delta,
        label="exact_member_delta",
    )
    anchor_core_delta = _finite_matrix(
        anchor_core_delta,
        label="anchor_core_delta",
    )
    rows = aligned_design.shape[0]
    if (
        exact_member_delta.shape != anchor_core_delta.shape
        or exact_member_delta.shape[0] != rows
    ):
        raise ValueError("pair score tensors must have aligned rows and outputs")
    if (
        isinstance(family_ids, (str, bytes))
        or not isinstance(family_ids, Sequence)
        or len(family_ids) != rows
        or any(
            not isinstance(family_id, str) or not family_id
            for family_id in family_ids
        )
    ):
        raise ValueError("family_ids must identify every aligned row")
    base_weights = _normalized_fisher_weights(
        row_fisher_weights,
        row_count=rows,
    )
    member64 = exact_member_delta.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    anchor64 = anchor_core_delta.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    family_ids_tuple = tuple(family_ids)
    scores: list[FamilyZeroResidualScore] = []
    for family_id in sorted(set(family_ids_tuple)):
        indices = torch.tensor(
            [
                index
                for index, observed in enumerate(family_ids_tuple)
                if observed == family_id
            ],
            dtype=torch.long,
        )
        weights = base_weights.index_select(0, indices)
        family_mean = weights.mean()
        if not bool(torch.isfinite(family_mean)) or float(family_mean) <= 0.0:
            raise ValueError(f"family {family_id!r} has no Fisher energy")
        weights = weights / family_mean
        member = member64.index_select(0, indices)
        residual = member - anchor64.index_select(0, indices)
        target_energy = float(
            (weights[:, None] * member.square()).sum().item()
        )
        residual_sse = float(
            (weights[:, None] * residual.square()).sum().item()
        )
        epsilon = torch.finfo(torch.float64).eps * max(1.0, target_energy)
        relative_sse = residual_sse / max(target_energy, epsilon)
        scores.append(
            FamilyZeroResidualScore(
                family_id=family_id,
                row_count=int(indices.numel()),
                normalized_weight_sum=float(weights.sum().item()),
                target_energy=target_energy,
                residual_sse=residual_sse,
                relative_sse=relative_sse,
            )
        )
    score = math.fsum(item.relative_sse for item in scores) / len(scores)
    numerical_tensor_sha256 = canonical_float32_tensor_sha256(
        aligned_design,
        exact_member_delta,
        domain=_PAIR_NUMERICAL_TENSOR_DOMAIN,
    )
    return ZeroResidualPairScore(
        score=score,
        numerical_tensor_sha256=numerical_tensor_sha256,
        family_scores=tuple(scores),
        row_count=rows,
        family_count=len(scores),
    )


@dataclass(frozen=True, slots=True)
class DirectedMLPPairCandidate:
    """One directed anchor/member score; ordinals encode eligibility only."""

    anchor_invocation_id: str
    member_invocation_id: str
    anchor_block_ordinal: int
    member_block_ordinal: int
    evidence: ZeroResidualPairScore

    def __post_init__(self) -> None:
        _require_id(self.anchor_invocation_id, label="anchor_invocation_id")
        _require_id(self.member_invocation_id, label="member_invocation_id")
        if self.anchor_invocation_id == self.member_invocation_id:
            raise ValueError("directed pair members must be distinct")
        for label, value in (
            ("anchor_block_ordinal", self.anchor_block_ordinal),
            ("member_block_ordinal", self.member_block_ordinal),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if not isinstance(self.evidence, ZeroResidualPairScore):
            raise TypeError("evidence must be ZeroResidualPairScore")

    @property
    def eligible(self) -> bool:
        return abs(self.anchor_block_ordinal - self.member_block_ordinal) >= 2

    @property
    def score(self) -> float:
        return self.evidence.score

    @property
    def numerical_tensor_sha256(self) -> str:
        return self.evidence.numerical_tensor_sha256


def build_directed_mlp_pair_candidate(
    *,
    anchor_invocation_id: str,
    member_invocation_id: str,
    anchor_block_ordinal: int,
    member_block_ordinal: int,
    aligned_design: Tensor,
    exact_member_delta: Tensor,
    anchor_core_delta: Tensor,
    row_fisher_weights: Tensor,
    family_ids: Sequence[str],
) -> DirectedMLPPairCandidate:
    return DirectedMLPPairCandidate(
        anchor_invocation_id=anchor_invocation_id,
        member_invocation_id=member_invocation_id,
        anchor_block_ordinal=anchor_block_ordinal,
        member_block_ordinal=member_block_ordinal,
        evidence=score_fisher_weighted_zero_residual_pair(
            aligned_design=aligned_design,
            exact_member_delta=exact_member_delta,
            anchor_core_delta=anchor_core_delta,
            row_fisher_weights=row_fisher_weights,
            family_ids=family_ids,
        ),
    )


def select_directed_mlp_pair(
    candidates: Sequence[DirectedMLPPairCandidate],
) -> DirectedMLPPairCandidate:
    """Select by score then tensor digest after non-adjacency filtering.

    Invocation ids, ordinals, block distance, and input order never break a
    numerical tie.  An identical score and numerical tensor digest therefore
    fails closed instead of falling back to a layer label.
    """

    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or not candidates
        or any(
            not isinstance(candidate, DirectedMLPPairCandidate)
            for candidate in candidates
        )
    ):
        raise ValueError("candidates must be a nonempty materialized sequence")
    eligible = tuple(candidate for candidate in candidates if candidate.eligible)
    if not eligible:
        raise ValueError("no non-adjacent directed MLP pair is eligible")
    ordered = sorted(
        eligible,
        key=lambda candidate: (
            candidate.score,
            candidate.numerical_tensor_sha256,
        ),
    )
    if (
        len(ordered) > 1
        and ordered[0].score == ordered[1].score
        and ordered[0].numerical_tensor_sha256
        == ordered[1].numerical_tensor_sha256
    ):
        raise RuntimeError(
            "directed-pair tie has an identical numerical tensor digest"
        )
    return ordered[0]


def select_directed_mlp_family_pair(
    candidates: Sequence[DirectedMLPPairCandidate],
) -> DirectedMLPPairCandidate:
    """High-level worker seam for the frozen directed MLP pair search."""

    return select_directed_mlp_pair(candidates)


# Explicit aliases keep call sites readable while retaining one implementation.
fit_projection_residual_adapter = fit_fisher_weighted_reduced_rank_projection
fisher_weighted_zero_residual_pair_score = (
    score_fisher_weighted_zero_residual_pair
)
select_directed_nonadjacent_mlp_pair = select_directed_mlp_pair


__all__ = [
    "DEFAULT_RELATIVE_RIDGE",
    "DirectedMLPPairCandidate",
    "FamilyZeroResidualScore",
    "INDEPENDENT_MEMBER_TARGET",
    "MLPProjectionFactorCatalogFit",
    "PRIMARY_RESIDUAL_TARGET",
    "ProjectionResidualFit",
    "ProjectionResidualFitReceipt",
    "ZeroResidualPairScore",
    "build_directed_mlp_pair_candidate",
    "canonical_float32_tensor_sha256",
    "fisher_weighted_zero_residual_pair_score",
    "fit_fisher_weighted_reduced_rank_projection",
    "fit_mlp_projection_factor_catalog",
    "fit_projection_residual_adapter",
    "score_fisher_weighted_zero_residual_pair",
    "select_directed_mlp_pair",
    "select_directed_mlp_family_pair",
    "select_directed_nonadjacent_mlp_pair",
]
