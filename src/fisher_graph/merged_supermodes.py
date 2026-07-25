"""Fisher-aware dense supermodes inside a locked residual span."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .codimension_projection import (
    OrthogonalDeltaProjector,
    canonical_orthonormal_basis,
    canonical_unit_direction,
)
from .linear_codec import (
    LinearActivationCodec,
    build_generalized_fisher_codec,
)
from .rotated_span_executor import deterministic_orthogonal_complement


_LEGACY_MAXIMUM_RANK_PROJECTION = (
    "factorized_generalized_fisher_replay"
)
_AUTHENTICATED_MAXIMUM_RANK_PROJECTION = (
    "authenticated_one_normal"
)


def _symmetric_finite_matrix(
    value: Tensor,
    *,
    width: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.shape != (width, width)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be a finite square floating matrix")
    result = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).clone()
    scale = max(float(result.abs().max().item()), 1.0)
    tolerance = 256 * torch.finfo(torch.float64).eps * width * scale
    if not torch.allclose(
        result,
        result.T,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(f"{label} must be symmetric")
    result = ((result + result.T) * 0.5).contiguous()
    eigenvalues = torch.linalg.eigvalsh(result)
    if float(eigenvalues.min().item()) < -2.0 * tolerance:
        raise ValueError(f"{label} must be positive semidefinite")
    return result


def _canonicalize_supermode_signs(
    coordinates: Tensor,
    supermodes: Tensor,
) -> tuple[Tensor, Tensor]:
    result_coordinates = coordinates.clone()
    result_supermodes = supermodes.clone()
    for index in range(result_supermodes.shape[1]):
        column = result_supermodes[:, index]
        pivot = int(column.abs().argmax().item())
        if float(column[pivot].item()) < 0.0:
            result_coordinates[:, index].neg_()
            result_supermodes[:, index].neg_()
    return (
        result_coordinates.contiguous(),
        result_supermodes.contiguous(),
    )


@dataclass(frozen=True, slots=True)
class MergedSupermodeBasis:
    """An ordered orthonormal mixture basis inside one locked span."""

    family: str
    locked_normal: Tensor
    operator: Tensor
    eigenvalues_descending: Tensor
    coordinate_vectors_descending: Tensor
    supermodes: Tensor

    def __post_init__(self) -> None:
        if self.family not in {
            "balanced_score_fisher_delta",
            "delta_only",
        }:
            raise ValueError("unsupported merged-supermode family")
        normal = canonical_unit_direction(
            self.locked_normal,
            label="locked normal",
        )
        supplied_normal = self.locked_normal.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.allclose(
            supplied_normal,
            normal,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("locked normal must already be canonical")
        width = normal.numel()
        rank = width - 1
        operator = _symmetric_finite_matrix(
            self.operator,
            width=rank,
            label="merged-supermode operator",
        )
        eigenvalues = self.eigenvalues_descending.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        coordinates = self.coordinate_vectors_descending.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        supermodes = self.supermodes.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if (
            eigenvalues.shape != (rank,)
            or coordinates.shape != (rank, rank)
            or supermodes.shape != (width, rank)
            or not torch.isfinite(eigenvalues).all()
            or not torch.isfinite(coordinates).all()
            or not torch.isfinite(supermodes).all()
            or (eigenvalues[:-1] < eigenvalues[1:]).any()
            or float(eigenvalues[-1].item()) < -1e-12
        ):
            raise ValueError("merged-supermode eigensystem is invalid")
        identity = torch.eye(rank, dtype=torch.float64)
        if (
            not torch.allclose(
                coordinates.T @ coordinates,
                identity,
                rtol=0.0,
                atol=1e-9,
            )
            or not torch.allclose(
                supermodes.T @ supermodes,
                identity,
                rtol=0.0,
                atol=1e-9,
            )
            or float((supermodes.T @ normal).abs().max().item()) > 1e-9
        ):
            raise ValueError(
                "merged-supermode columns must be orthonormal in the "
                "locked span"
            )
        span = deterministic_orthogonal_complement(normal)
        reconstructed = span @ coordinates
        if (
            not torch.allclose(
                reconstructed,
                supermodes,
                rtol=1e-9,
                atol=1e-10,
            )
            or not torch.allclose(
                operator @ coordinates,
                coordinates * eigenvalues.unsqueeze(0),
                rtol=1e-8,
                atol=1e-10,
            )
        ):
            raise ValueError(
                "merged-supermode vectors do not diagonalize the operator"
            )
        canonical_coordinates, canonical_supermodes = (
            _canonicalize_supermode_signs(coordinates, supermodes)
        )
        if not torch.allclose(
            canonical_supermodes,
            supermodes,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "merged-supermode columns must use canonical signs"
            )
        object.__setattr__(self, "locked_normal", normal)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "eigenvalues_descending",
            eigenvalues.contiguous(),
        )
        object.__setattr__(
            self,
            "coordinate_vectors_descending",
            canonical_coordinates,
        )
        object.__setattr__(
            self,
            "supermodes",
            canonical_supermodes,
        )

    @property
    def width(self) -> int:
        return int(self.locked_normal.numel())

    @property
    def maximum_rank(self) -> int:
        return self.width - 1

    def retained_basis(self, retained_rank: int) -> Tensor:
        if (
            type(retained_rank) is not int
            or not 1 <= retained_rank <= self.maximum_rank
        ):
            raise ValueError(
                "retained_rank must be between one and the locked span rank"
            )
        return self.supermodes[:, :retained_rank].clone()

    def projector(
        self,
        retained_rank: int,
    ) -> OrthogonalDeltaProjector:
        retained = self.retained_basis(retained_rank)
        del retained
        omitted = torch.column_stack(
            (
                self.locked_normal,
                self.supermodes[:, retained_rank:],
            )
        )
        omitted = canonical_orthonormal_basis(
            omitted,
            label="merged-supermode omitted basis",
        )
        return OrthogonalDeltaProjector(omitted)

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "family": self.family,
            "locked_normal": self.locked_normal.clone(),
            "operator": self.operator.clone(),
            "eigenvalues_descending": (
                self.eigenvalues_descending.clone()
            ),
            "coordinate_vectors_descending": (
                self.coordinate_vectors_descending.clone()
            ),
            "supermodes": self.supermodes.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> MergedSupermodeBasis:
        fields = {
            "format_version",
            "family",
            "locked_normal",
            "operator",
            "eigenvalues_descending",
            "coordinate_vectors_descending",
            "supermodes",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError("merged-supermode basis fields are invalid")
        if state["format_version"] != 1:
            raise ValueError("unsupported merged-supermode basis format")
        tensors = {
            name: state[name]
            for name in (
                "locked_normal",
                "operator",
                "eigenvalues_descending",
                "coordinate_vectors_descending",
                "supermodes",
            )
        }
        if any(not isinstance(value, Tensor) for value in tensors.values()):
            raise TypeError("merged-supermode basis values must be Tensors")
        family = state["family"]
        if not isinstance(family, str):
            raise TypeError("merged-supermode family must be a string")
        return cls(
            family=family,
            locked_normal=tensors["locked_normal"],  # type: ignore[arg-type]
            operator=tensors["operator"],  # type: ignore[arg-type]
            eigenvalues_descending=tensors[
                "eigenvalues_descending"
            ],  # type: ignore[arg-type]
            coordinate_vectors_descending=tensors[
                "coordinate_vectors_descending"
            ],  # type: ignore[arg-type]
            supermodes=tensors["supermodes"],  # type: ignore[arg-type]
        )


def build_merged_supermode_basis(
    *,
    family: str,
    locked_normal: Tensor,
    score_fisher: Tensor,
    delta_second_moment: Tensor,
) -> MergedSupermodeBasis:
    """Diagonalize a preregistered modal importance operator."""

    normal = canonical_unit_direction(
        locked_normal,
        label="locked normal",
    )
    rank = normal.numel() - 1
    score = _symmetric_finite_matrix(
        score_fisher,
        width=rank,
        label="score Fisher",
    )
    delta = _symmetric_finite_matrix(
        delta_second_moment,
        width=rank,
        label="delta second moment",
    )
    score_trace = float(torch.trace(score).item())
    delta_trace = float(torch.trace(delta).item())
    if score_trace <= 0.0 or delta_trace <= 0.0:
        raise ValueError("merged-supermode moments must have positive trace")
    if family == "balanced_score_fisher_delta":
        operator = (
            0.5 * score / score_trace
            + 0.5 * delta / delta_trace
        )
    elif family == "delta_only":
        operator = delta / delta_trace
    else:
        raise ValueError("unsupported merged-supermode family")
    operator = ((operator + operator.T) * 0.5).contiguous()
    eigenvalues, coordinate_vectors = torch.linalg.eigh(operator)
    order = torch.arange(
        rank - 1,
        -1,
        -1,
        dtype=torch.long,
    )
    eigenvalues = eigenvalues[order].clamp_min(0).contiguous()
    coordinate_vectors = coordinate_vectors[:, order].contiguous()
    span = deterministic_orthogonal_complement(normal)
    supermodes = (span @ coordinate_vectors).contiguous()
    coordinate_vectors, supermodes = _canonicalize_supermode_signs(
        coordinate_vectors,
        supermodes,
    )
    return MergedSupermodeBasis(
        family=family,
        locked_normal=normal,
        operator=operator,
        eigenvalues_descending=eigenvalues,
        coordinate_vectors_descending=coordinate_vectors,
        supermodes=supermodes,
    )


@dataclass(frozen=True, slots=True)
class AnchoredTailSupermodeMerge:
    """Merge a validated tail span through a generalized Fisher codec."""

    tail_basis: Tensor
    locked_normal: Tensor
    surviving_coordinates: Tensor
    codec: LinearActivationCodec
    maximum_rank_projection: str = (
        _AUTHENTICATED_MAXIMUM_RANK_PROJECTION
    )

    def __post_init__(self) -> None:
        if self.maximum_rank_projection not in {
            _LEGACY_MAXIMUM_RANK_PROJECTION,
            _AUTHENTICATED_MAXIMUM_RANK_PROJECTION,
        }:
            raise ValueError(
                "unknown anchored-tail maximum-rank projection"
            )
        tail = canonical_orthonormal_basis(
            self.tail_basis,
            label="tail basis",
        )
        supplied_tail = self.tail_basis.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.allclose(
            supplied_tail,
            tail,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("tail basis must use canonical column signs")
        normal = canonical_unit_direction(
            self.locked_normal,
            label="locked normal",
        )
        supplied_normal = self.locked_normal.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.allclose(
            supplied_normal,
            normal,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("locked normal must already be canonical")
        if normal.numel() != tail.shape[0]:
            raise ValueError("tail basis and locked normal widths differ")
        tail_normal = tail.T @ normal
        if (
            abs(float(torch.linalg.vector_norm(tail_normal).item()) - 1.0)
            > 1e-9
            or float(
                (
                    normal - tail @ tail_normal
                ).abs().max().item()
            )
            > 1e-9
        ):
            raise ValueError("locked normal must lie in the tail span")
        tail_normal = canonical_unit_direction(
            tail_normal,
            label="tail normal",
        )
        expected_surviving = deterministic_orthogonal_complement(
            tail_normal
        )
        surviving = self.surviving_coordinates.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if (
            surviving.shape != expected_surviving.shape
            or not torch.allclose(
                surviving,
                expected_surviving,
                rtol=1e-10,
                atol=1e-11,
            )
        ):
            raise ValueError(
                "surviving coordinates do not match the deterministic "
                "locked-normal complement"
            )
        if (
            not isinstance(self.codec, LinearActivationCodec)
            or self.codec.width != surviving.shape[1]
            or self.codec.method != "generalized_fisher"
            or self.codec.alpha_floor != 0.0
            or self.codec.beta_floor != 0.0
            or not torch.equal(
                self.codec.mean,
                torch.zeros_like(self.codec.mean),
            )
        ):
            raise ValueError(
                "tail merge requires a zero-mean unregularized "
                "generalized Fisher codec"
            )
        object.__setattr__(self, "tail_basis", tail)
        object.__setattr__(self, "locked_normal", normal)
        object.__setattr__(
            self,
            "surviving_coordinates",
            expected_surviving,
        )

    @property
    def width(self) -> int:
        return int(self.tail_basis.shape[0])

    @property
    def tail_width(self) -> int:
        return int(self.tail_basis.shape[1])

    @property
    def maximum_supermodes(self) -> int:
        return self.tail_width - 1

    @property
    def preserved_prefix_rank(self) -> int:
        return self.width - self.tail_width

    def total_rank(self, supermode_rank: int) -> int:
        if (
            type(supermode_rank) is not int
            or not 0 <= supermode_rank <= self.maximum_supermodes
        ):
            raise ValueError(
                "supermode_rank must be between zero and the surviving "
                "tail rank"
            )
        return self.preserved_prefix_rank + supermode_rank

    def retained_weighted_fraction(self, supermode_rank: int) -> float:
        self.total_rank(supermode_rank)
        total = float(self.codec.importance_scores.sum().item())
        if total <= 0.0:
            raise RuntimeError("supermode importance has zero total")
        return float(
            self.codec.importance_scores[:supermode_rank].sum().item()
        ) / total

    def project_delta(
        self,
        delta: Tensor,
        *,
        supermode_rank: int,
    ) -> Tensor:
        self.total_rank(supermode_rank)
        if (
            not isinstance(delta, Tensor)
            or not delta.is_floating_point()
            or delta.ndim < 1
            or delta.shape[-1] != self.width
        ):
            raise ValueError(
                "delta must be floating with the merge width on its "
                "final axis"
            )
        compute_dtype = (
            torch.float32
            if delta.dtype in (torch.float16, torch.bfloat16)
            else delta.dtype
        )
        values = delta.to(dtype=compute_dtype)
        if (
            supermode_rank == self.maximum_supermodes
            and self.maximum_rank_projection
            == _AUTHENTICATED_MAXIMUM_RANK_PROJECTION
        ):
            normal = self.locked_normal.to(
                device=delta.device,
                dtype=compute_dtype,
            )
            removed = (values @ normal).unsqueeze(-1) * normal
            return (values - removed).to(dtype=delta.dtype)
        tail = self.tail_basis.to(
            device=delta.device,
            dtype=compute_dtype,
        )
        surviving = self.surviving_coordinates.to(
            device=delta.device,
            dtype=compute_dtype,
        )
        tail_coordinates = values @ tail
        preserved_prefix = values - tail_coordinates @ tail.T
        surviving_values = tail_coordinates @ surviving
        merged_values = self.codec.reconstruct(
            surviving_values,
            rank=supermode_rank,
        )
        reconstructed_tail = merged_values @ surviving.T @ tail.T
        return (preserved_prefix + reconstructed_tail).to(
            dtype=delta.dtype
        )

    def project_output(
        self,
        source: Tensor,
        target: Tensor,
        *,
        valid_positions: Tensor,
        supermode_rank: int,
    ) -> Tensor:
        if (
            not isinstance(source, Tensor)
            or not isinstance(target, Tensor)
            or source.shape != target.shape
            or source.ndim != 3
            or source.shape[-1] != self.width
            or not source.is_floating_point()
            or not target.is_floating_point()
        ):
            raise ValueError(
                "source and target must be aligned floating "
                "[batch, sequence, width] tensors"
            )
        if (
            not isinstance(valid_positions, Tensor)
            or valid_positions.dtype is not torch.bool
            or valid_positions.shape != source.shape[:2]
        ):
            raise ValueError(
                "valid_positions must be a matching boolean mask"
            )
        projected = source + self.project_delta(
            target - source,
            supermode_rank=supermode_rank,
        )
        return torch.where(
            valid_positions.to(device=source.device).unsqueeze(-1),
            projected,
            target,
        )

    def state_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "format_version": (
                1
                if self.maximum_rank_projection
                == _LEGACY_MAXIMUM_RANK_PROJECTION
                else 2
            ),
            "tail_basis": self.tail_basis.clone(),
            "locked_normal": self.locked_normal.clone(),
            "surviving_coordinates": (
                self.surviving_coordinates.clone()
            ),
            "codec": self.codec.state_dict(),
        }
        if result["format_version"] == 2:
            result["maximum_rank_projection"] = (
                self.maximum_rank_projection
            )
        return result

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> AnchoredTailSupermodeMerge:
        legacy_fields = {
            "format_version",
            "tail_basis",
            "locked_normal",
            "surviving_coordinates",
            "codec",
        }
        current_fields = legacy_fields | {"maximum_rank_projection"}
        if not isinstance(state, Mapping):
            raise ValueError("anchored tail merge fields are invalid")
        format_version = state.get("format_version")
        if (
            format_version == 1
            and set(state) != legacy_fields
        ) or (
            format_version == 2
            and (
                set(state) != current_fields
                or state.get("maximum_rank_projection")
                != _AUTHENTICATED_MAXIMUM_RANK_PROJECTION
            )
        ) or format_version not in {1, 2}:
            raise ValueError("unsupported anchored tail merge format")
        for field in (
            "tail_basis",
            "locked_normal",
            "surviving_coordinates",
        ):
            if not isinstance(state[field], Tensor):
                raise TypeError(
                    f"anchored tail merge {field} must be a Tensor"
                )
        codec_state = state["codec"]
        if not isinstance(codec_state, Mapping):
            raise TypeError("anchored tail merge codec must be a mapping")
        return cls(
            tail_basis=state["tail_basis"],  # type: ignore[arg-type]
            locked_normal=state["locked_normal"],  # type: ignore[arg-type]
            surviving_coordinates=state[
                "surviving_coordinates"
            ],  # type: ignore[arg-type]
            codec=LinearActivationCodec.from_state_dict(codec_state),
            maximum_rank_projection=(
                _LEGACY_MAXIMUM_RANK_PROJECTION
                if format_version == 1
                else _AUTHENTICATED_MAXIMUM_RANK_PROJECTION
            ),
        )


def build_anchored_tail_supermode_merge(
    *,
    tail_basis: Tensor,
    locked_normal: Tensor,
    score_fisher: Tensor,
    delta_second_moment: Tensor,
    maximum_rank_projection: str = (
        _AUTHENTICATED_MAXIMUM_RANK_PROJECTION
    ),
) -> AnchoredTailSupermodeMerge:
    """Fit unregularized generalized Fisher modes in the surviving tail."""

    tail = canonical_orthonormal_basis(
        tail_basis,
        label="tail basis",
    )
    normal = canonical_unit_direction(
        locked_normal,
        label="locked normal",
    )
    tail_normal = canonical_unit_direction(
        tail.T @ normal,
        label="tail normal",
    )
    surviving = deterministic_orthogonal_complement(tail_normal)
    tail_width = tail.shape[1]
    score = _symmetric_finite_matrix(
        score_fisher,
        width=tail_width,
        label="tail score Fisher",
    )
    delta = _symmetric_finite_matrix(
        delta_second_moment,
        width=tail_width,
        label="tail delta second moment",
    )
    surviving_score = (
        surviving.T @ score @ surviving
    ).contiguous()
    surviving_delta = (
        surviving.T @ delta @ surviving
    ).contiguous()
    codec = build_generalized_fisher_codec(
        covariance=surviving_delta,
        fisher_matrix=surviving_score,
        alpha=0.0,
        beta=0.0,
        activation_name="locked_rotated_tail_block_delta",
        mean=torch.zeros(
            surviving.shape[1],
            dtype=torch.float64,
        ),
    )
    return AnchoredTailSupermodeMerge(
        tail_basis=tail,
        locked_normal=normal,
        surviving_coordinates=surviving,
        codec=codec,
        maximum_rank_projection=maximum_rank_projection,
    )


__all__ = [
    "AnchoredTailSupermodeMerge",
    "MergedSupermodeBasis",
    "build_anchored_tail_supermode_merge",
    "build_merged_supermode_basis",
]
