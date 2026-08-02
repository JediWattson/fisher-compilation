"""Family-balanced complete-H4 residual projection geometry.

This module is deliberately independent of the Gemma runtime.  A runtime may
stream the complete-H4 correction target ``native_h4 - carrier_h4`` and the
optional NLL gradient at that boundary into
:class:`CompleteH4ProjectionFitSequence`.  The math layer then answers a
narrow capacity question: how much of that correction lies in a rank-limited
linear subspace?

Fit weighting and direction ordering are kept explicitly separate.  The fit
may use either the established Fisher-alignment row tilt or the unweighted
family/example-macro residual second moment.  After either fit, two direction
orderings are available:

``euclidean``
    Principal order of the selected fit covariance.  Rank-grid retention is
    always measured on the unweighted residual, so it remains a Euclidean
    capacity measurement even when the fitted directions use the bounded
    Fisher-alignment tilt.
``fisher``
    A diagnostic ordering of the *same* Euclidean directions by
    ``residual_variance * E[(gradient @ direction)^2]``.  It describes NLL
    relevance; it does not redefine, rotate, or improve the Euclidean basis.

Fit tensors are copied into immutable byte-backed float64 matrices and bound
by domain-separated SHA-256 receipts.  Basis and geometry metadata retain no
prompts, token ids, logits, or model weights.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Literal

import torch
from torch import Tensor


__all__ = [
    "COMPLETE_H4_DEFAULT_RANK_GRID",
    "CompleteH4ProjectionBasis",
    "CompleteH4ProjectionFitSequence",
    "CompleteH4ProjectionGeometry",
    "CompleteH4ProjectionRankGeometry",
    "ImmutableFloat64Matrix",
    "ProjectionFitWeighting",
    "ProjectionOrdering",
    "canonical_complete_h4_rank_grid",
    "canonicalize_orthonormal_basis_signs",
    "fit_complete_h4_projection_basis",
    "project_complete_h4_residual_rows",
    "summarize_complete_h4_projection_geometry",
]


ProjectionOrdering = Literal["euclidean", "fisher"]
ProjectionFitWeighting = Literal["fisher_alignment_tilted", "unweighted"]
COMPLETE_H4_DEFAULT_RANK_GRID: tuple[int, ...] = (8, 16, 32, 64)

_FISHER_ALIGNMENT_TILTED_COVARIANCE = (
    "family_example_macro_residual_second_moment_with_"
    "one_plus_cosine_squared_fisher_alignment_tilt"
)
_UNWEIGHTED_COVARIANCE = (
    "family_example_macro_unweighted_residual_second_moment"
)

_MATRIX_DOMAIN = b"fisher-graph:complete-h4:immutable-f64-matrix:v1\0"
_SEQUENCE_DOMAIN = b"fisher-graph:complete-h4:fit-sequence:v1\0"
_BASIS_DOMAIN = b"fisher-graph:complete-h4:projection-basis:v1\0"
_GEOMETRY_DOMAIN = b"fisher-graph:complete-h4:projection-geometry:v1\0"


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
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _strict_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(
    value: object,
    *,
    label: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _float_hex_tuple(values: Sequence[float]) -> tuple[str, ...]:
    return tuple(float(value).hex() for value in values)


@dataclass(frozen=True, slots=True)
class ImmutableFloat64Matrix:
    """An immutable, exact, CPU float64 matrix with a content receipt.

    ``to_tensor`` always returns a new writable tensor, so neither caller
    mutation nor downstream linear algebra can alter the hash-bound payload.
    """

    row_count: int
    width: int
    _little_endian_bytes: bytes = field(repr=False)
    matrix_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _strict_positive_int(self.row_count, label="matrix row_count")
        _strict_positive_int(self.width, label="matrix width")
        if not isinstance(self._little_endian_bytes, bytes):
            raise TypeError("matrix payload must be immutable bytes")
        expected_bytes = self.row_count * self.width * 8
        if len(self._little_endian_bytes) != expected_bytes:
            raise ValueError(
                "matrix payload byte length does not match [row_count,width]"
            )
        decoded = torch.frombuffer(
            bytearray(self._little_endian_bytes),
            dtype=torch.float64,
        )
        if not bool(torch.isfinite(decoded).all().item()):
            raise ValueError("matrix payload must contain only finite values")
        object.__setattr__(
            self,
            "matrix_sha256",
            hashlib.sha256(
                _MATRIX_DOMAIN
                + _canonical_json_bytes(
                    {
                        "dtype": "float64-little-endian",
                        "shape": (self.row_count, self.width),
                    }
                )
                + self._little_endian_bytes
            ).hexdigest(),
        )

    @classmethod
    def from_tensor(cls, value: Tensor, *, label: str) -> "ImmutableFloat64Matrix":
        if not isinstance(value, Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError(f"{label} must have nonempty shape [N,D]")
        if value.dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise TypeError(f"{label} must have a floating dtype")
        normalized = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        # NumPy emits native-endian float64 bytes.  All supported execution
        # targets are little endian, but spell the conversion out to bind the
        # serialization contract rather than the host default.
        payload = normalized.numpy().astype("<f8", copy=False).tobytes(order="C")
        return cls(
            row_count=int(normalized.shape[0]),
            width=int(normalized.shape[1]),
            _little_endian_bytes=payload,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_count, self.width)

    def to_tensor(self) -> Tensor:
        # bytearray gives torch a writable temporary buffer; clone severs the
        # returned tensor from it before this method returns.
        return (
            torch.frombuffer(
                bytearray(self._little_endian_bytes),
                dtype=torch.float64,
            )
            .reshape(self.row_count, self.width)
            .clone()
        )

    def metadata(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "dtype": "float64",
            "matrix_sha256": self.matrix_sha256,
        }


MatrixInput = Tensor | ImmutableFloat64Matrix


def _immutable_matrix(value: MatrixInput, *, label: str) -> ImmutableFloat64Matrix:
    if isinstance(value, ImmutableFloat64Matrix):
        # Reconstructing performs byte-length and digest validation again.
        return ImmutableFloat64Matrix(
            row_count=value.row_count,
            width=value.width,
            _little_endian_bytes=value._little_endian_bytes,
        )
    return ImmutableFloat64Matrix.from_tensor(value, label=label)


@dataclass(frozen=True, slots=True)
class CompleteH4ProjectionFitSequence:
    """One example's complete-H4 correction rows and optional NLL scores."""

    example_id: str
    family_id: str
    residual_rows: MatrixInput
    gradient_rows: MatrixInput | None = None
    sequence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="fit sequence example_id")
        _identifier(self.family_id, label="fit sequence family_id")
        residual = _immutable_matrix(
            self.residual_rows,
            label="fit sequence residual_rows",
        )
        gradient = (
            None
            if self.gradient_rows is None
            else _immutable_matrix(
                self.gradient_rows,
                label="fit sequence gradient_rows",
            )
        )
        if gradient is not None and gradient.shape != residual.shape:
            raise ValueError(
                "fit sequence gradient_rows must match residual_rows shape"
            )
        object.__setattr__(self, "residual_rows", residual)
        object.__setattr__(self, "gradient_rows", gradient)
        object.__setattr__(
            self,
            "sequence_sha256",
            _sha256(_SEQUENCE_DOMAIN, self._payload()),
        )

    @property
    def row_count(self) -> int:
        return self.residual_rows.row_count  # type: ignore[union-attr]

    @property
    def width(self) -> int:
        return self.residual_rows.width  # type: ignore[union-attr]

    @property
    def has_gradients(self) -> bool:
        return self.gradient_rows is not None

    def _payload(self) -> dict[str, object]:
        residual = self.residual_rows
        gradient = self.gradient_rows
        assert isinstance(residual, ImmutableFloat64Matrix)
        assert gradient is None or isinstance(gradient, ImmutableFloat64Matrix)
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "residual_rows": residual.metadata(),
            "gradient_rows": (
                None if gradient is None else gradient.metadata()
            ),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "sequence_sha256": self.sequence_sha256}


def canonicalize_orthonormal_basis_signs(basis_rows: Tensor) -> Tensor:
    """Choose a deterministic sign using each row's first max-|value| pivot."""

    if not isinstance(basis_rows, Tensor):
        raise TypeError("basis_rows must be a torch.Tensor")
    if basis_rows.ndim != 2 or basis_rows.shape[0] <= 0 or basis_rows.shape[1] <= 0:
        raise ValueError("basis_rows must have nonempty shape [rank,width]")
    if basis_rows.dtype not in {torch.float32, torch.float64}:
        raise TypeError("basis_rows must have float32 or float64 dtype")
    result = basis_rows.detach().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("basis_rows must contain only finite values")
    gram = result @ result.transpose(0, 1)
    identity = torch.eye(
        result.shape[0],
        dtype=result.dtype,
        device=result.device,
    )
    tolerance = 1e-9 if result.dtype == torch.float64 else 2e-5
    if not bool(torch.allclose(gram, identity, atol=tolerance, rtol=tolerance)):
        raise ValueError("basis_rows must be orthonormal")
    for row_index in range(result.shape[0]):
        row = result[row_index]
        pivot = int(torch.argmax(torch.abs(row)).item())
        if float(row[pivot].item()) < 0.0:
            result[row_index].neg_()
    return result


def _validated_sequences(
    sequences: Iterable[CompleteH4ProjectionFitSequence],
    *,
    require_unique_examples: bool = True,
) -> tuple[CompleteH4ProjectionFitSequence, ...]:
    result = tuple(sequences)
    if not result:
        raise ValueError("complete-H4 projection requires at least one sequence")
    if any(not isinstance(value, CompleteH4ProjectionFitSequence) for value in result):
        raise TypeError(
            "complete-H4 projection sequences must be "
            "CompleteH4ProjectionFitSequence instances"
        )
    widths = {value.width for value in result}
    if len(widths) != 1:
        raise ValueError("complete-H4 projection sequence widths must match")
    example_ids = [value.example_id for value in result]
    if require_unique_examples and len(set(example_ids)) != len(example_ids):
        raise ValueError("complete-H4 projection example_id values must be unique")
    gradient_flags = {value.has_gradients for value in result}
    if len(gradient_flags) != 1:
        raise ValueError(
            "gradient_rows must be supplied for either every sequence or none"
        )
    return tuple(sorted(result, key=lambda value: (value.family_id, value.example_id)))


def _family_example_macro_moments(
    sequences: Sequence[CompleteH4ProjectionFitSequence],
    *,
    fit_weighting: ProjectionFitWeighting,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Match the established progressive map's family/example weighting.

    Each example contributes its row-mean moment, examples are averaged inside
    a family, and families are averaged last.  The tilted fit applies the
    established bounded row weight ``1 + cos^2(residual, gradient)``; the
    unweighted fit uses the raw residual second moment.  The returned Fisher
    moment remains a separate diagnostic operator for both fits.
    """

    if fit_weighting not in ("fisher_alignment_tilted", "unweighted"):
        raise ValueError(
            "projection fit_weighting must be "
            "'fisher_alignment_tilted' or 'unweighted'"
        )

    family_covariances: dict[str, Tensor] = {}
    family_unweighted_covariances: dict[str, Tensor] = {}
    family_fishers: dict[str, Tensor] = {}
    family_examples: dict[str, int] = defaultdict(int)
    for sequence in sequences:
        residual = sequence.residual_rows.to_tensor()  # type: ignore[union-attr]
        gradient = (
            None
            if sequence.gradient_rows is None
            else sequence.gradient_rows.to_tensor()  # type: ignore[union-attr]
        )
        weighted = residual
        if gradient is not None and fit_weighting == "fisher_alignment_tilted":
            residual_square = residual.square().sum(dim=1)
            gradient_square = gradient.square().sum(dim=1)
            alignment = (residual * gradient).sum(dim=1).square() / (
                residual_square * gradient_square + 1.0e-30
            )
            fisher_weight = 1.0 + alignment.clamp(min=0.0, max=1.0)
            weighted = residual * fisher_weight.sqrt().unsqueeze(1)
        covariance = weighted.transpose(0, 1) @ weighted / residual.shape[0]
        unweighted_covariance = (
            residual.transpose(0, 1) @ residual / residual.shape[0]
        )
        family_covariances[sequence.family_id] = (
            family_covariances.get(sequence.family_id, torch.zeros_like(covariance))
            + covariance
        )
        family_unweighted_covariances[sequence.family_id] = (
            family_unweighted_covariances.get(
                sequence.family_id,
                torch.zeros_like(unweighted_covariance),
            )
            + unweighted_covariance
        )
        if gradient is not None:
            fisher = gradient.transpose(0, 1) @ gradient / gradient.shape[0]
            family_fishers[sequence.family_id] = (
                family_fishers.get(sequence.family_id, torch.zeros_like(fisher))
                + fisher
            )
        family_examples[sequence.family_id] += 1

    family_ids = sorted(family_covariances)
    covariance = sum(
        family_covariances[family_id] / family_examples[family_id]
        for family_id in family_ids
    ) / len(family_ids)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    unweighted_covariance = sum(
        family_unweighted_covariances[family_id]
        / family_examples[family_id]
        for family_id in family_ids
    ) / len(family_ids)
    unweighted_covariance = 0.5 * (
        unweighted_covariance + unweighted_covariance.transpose(0, 1)
    )
    if not family_fishers:
        return covariance, unweighted_covariance, None
    fisher = sum(
        family_fishers[family_id] / family_examples[family_id]
        for family_id in family_ids
    ) / len(family_ids)
    fisher = 0.5 * (fisher + fisher.transpose(0, 1))
    return covariance, unweighted_covariance, fisher


@dataclass(frozen=True, slots=True)
class CompleteH4ProjectionBasis:
    """Hash-bound residual basis plus separate Fisher relevance metadata."""

    width: int
    max_rank: int
    basis_rows: MatrixInput
    residual_eigenvalues: tuple[float, ...]
    residual_energy_fractions: tuple[float, ...]
    directional_residual_variance: tuple[float, ...]
    next_residual_eigenvalue: float
    cutoff_spectral_gap: float
    source_example_ids: tuple[str, ...]
    source_family_ids: tuple[str, ...]
    source_sequence_sha256s: tuple[str, ...]
    directional_fisher: tuple[float, ...] | None = None
    fisher_relevance: tuple[float, ...] | None = None
    fisher_first_order_mode_coupling: tuple[float, ...] | None = None
    fisher_rank_order: tuple[int, ...] | None = None
    fit_weighting: ProjectionFitWeighting = "fisher_alignment_tilted"
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.fit_weighting not in (
            "fisher_alignment_tilted",
            "unweighted",
        ):
            raise ValueError(
                "projection fit_weighting must be "
                "'fisher_alignment_tilted' or 'unweighted'"
            )
        width = _strict_positive_int(self.width, label="projection basis width")
        rank = _strict_positive_int(self.max_rank, label="projection basis max_rank")
        matrix = _immutable_matrix(self.basis_rows, label="projection basis_rows")
        if matrix.shape != (rank, width):
            raise ValueError("projection basis_rows shape must be [max_rank,width]")
        basis = matrix.to_tensor()
        gram = basis @ basis.transpose(0, 1)
        if not bool(
            torch.allclose(
                gram,
                torch.eye(rank, dtype=torch.float64),
                atol=1e-9,
                rtol=1e-9,
            )
        ):
            raise ValueError("projection basis_rows must be orthonormal")
        canonical = canonicalize_orthonormal_basis_signs(basis)
        if not bool(torch.equal(canonical, basis)):
            raise ValueError("projection basis_rows signs are not canonical")

        eigenvalues = tuple(
            _finite_float(
                value,
                label="projection residual eigenvalue",
                nonnegative=True,
            )
            for value in self.residual_eigenvalues
        )
        fractions = tuple(
            _finite_float(
                value,
                label="projection residual energy fraction",
                nonnegative=True,
            )
            for value in self.residual_energy_fractions
        )
        if len(eigenvalues) != rank or len(fractions) != rank:
            raise ValueError(
                "projection eigenvalue and energy-fraction lengths must equal max_rank"
            )
        residual_variance = tuple(
            _finite_float(
                value,
                label="projection directional residual variance",
                nonnegative=True,
            )
            for value in self.directional_residual_variance
        )
        if len(residual_variance) != rank:
            raise ValueError(
                "projection directional residual variance length must equal max_rank"
            )
        if any(
            eigenvalues[index] + 1e-12 < eigenvalues[index + 1]
            for index in range(rank - 1)
        ):
            raise ValueError("projection residual eigenvalues must be descending")
        next_eigenvalue = _finite_float(
            self.next_residual_eigenvalue,
            label="projection next residual eigenvalue",
            nonnegative=True,
        )
        spectral_gap = _finite_float(
            self.cutoff_spectral_gap,
            label="projection cutoff spectral gap",
            nonnegative=True,
        )
        if next_eigenvalue > eigenvalues[-1] + 1e-12:
            raise ValueError(
                "projection next eigenvalue exceeds the cutoff eigenvalue"
            )
        expected_gap = max(0.0, eigenvalues[-1] - next_eigenvalue)
        if not math.isclose(
            spectral_gap,
            expected_gap,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("projection cutoff spectral gap is inconsistent")
        if sum(fractions) > 1.0 + 1e-9:
            raise ValueError("projection residual energy fractions cannot sum above one")

        examples = tuple(
            _identifier(value, label="projection source example_id")
            for value in self.source_example_ids
        )
        families = tuple(
            _identifier(value, label="projection source family_id")
            for value in self.source_family_ids
        )
        receipts = tuple(self.source_sequence_sha256s)
        if not examples or len(set(examples)) != len(examples):
            raise ValueError("projection source example ids must be nonempty and unique")
        if not families or tuple(sorted(set(families))) != families:
            raise ValueError("projection source family ids must be sorted and unique")
        if len(receipts) != len(examples) or any(
            not isinstance(value, str) or len(value) != 64
            for value in receipts
        ):
            raise ValueError("projection source sequence receipts are invalid")

        fisher_fields = (
            self.directional_fisher,
            self.fisher_relevance,
            self.fisher_first_order_mode_coupling,
            self.fisher_rank_order,
        )
        if any(value is None for value in fisher_fields) and not all(
            value is None for value in fisher_fields
        ):
            raise ValueError("projection Fisher metadata must be entirely present or absent")
        if self.directional_fisher is not None:
            directional = tuple(
                _finite_float(
                    value,
                    label="projection directional Fisher",
                    nonnegative=True,
                )
                for value in self.directional_fisher
            )
            relevance = tuple(
                _finite_float(
                    value,
                    label="projection Fisher relevance",
                    nonnegative=True,
                )
                for value in self.fisher_relevance or ()
            )
            coupling = tuple(
                _finite_float(
                    value,
                    label="projection Fisher first-order mode coupling",
                )
                for value in self.fisher_first_order_mode_coupling or ()
            )
            order = tuple(self.fisher_rank_order or ())
            if not all(len(value) == rank for value in (directional, relevance, coupling, order)):
                raise ValueError("projection Fisher metadata lengths must equal max_rank")
            if sorted(order) != list(range(rank)):
                raise ValueError("projection fisher_rank_order must be a permutation")
            expected_order = tuple(sorted(range(rank), key=lambda index: (-relevance[index], index)))
            if order != expected_order:
                raise ValueError("projection fisher_rank_order does not match relevance")
            object.__setattr__(self, "directional_fisher", directional)
            object.__setattr__(self, "fisher_relevance", relevance)
            object.__setattr__(self, "fisher_first_order_mode_coupling", coupling)
            object.__setattr__(self, "fisher_rank_order", order)

        object.__setattr__(self, "basis_rows", matrix)
        object.__setattr__(self, "residual_eigenvalues", eigenvalues)
        object.__setattr__(self, "residual_energy_fractions", fractions)
        object.__setattr__(
            self,
            "directional_residual_variance",
            residual_variance,
        )
        object.__setattr__(self, "next_residual_eigenvalue", next_eigenvalue)
        object.__setattr__(self, "cutoff_spectral_gap", spectral_gap)
        object.__setattr__(self, "source_example_ids", examples)
        object.__setattr__(self, "source_family_ids", families)
        object.__setattr__(self, "source_sequence_sha256s", receipts)
        object.__setattr__(self, "fit_weighting", self.fit_weighting)
        object.__setattr__(self, "artifact_sha256", _sha256(_BASIS_DOMAIN, self._payload()))

    @property
    def has_fisher(self) -> bool:
        return self.directional_fisher is not None

    def basis_tensor(self, *, ordering: ProjectionOrdering = "euclidean") -> Tensor:
        basis = self.basis_rows.to_tensor()  # type: ignore[union-attr]
        if ordering == "euclidean":
            return basis
        if ordering != "fisher":
            raise ValueError("projection ordering must be 'euclidean' or 'fisher'")
        if self.fisher_rank_order is None:
            raise ValueError("Fisher ordering requires gradient_rows during basis fitting")
        return basis[list(self.fisher_rank_order)]

    def _payload(self) -> dict[str, object]:
        matrix = self.basis_rows
        assert isinstance(matrix, ImmutableFloat64Matrix)
        return {
            "width": self.width,
            "max_rank": self.max_rank,
            "basis_rows": matrix.metadata(),
            "residual_eigenvalues_hex": _float_hex_tuple(self.residual_eigenvalues),
            "residual_energy_fractions_hex": _float_hex_tuple(
                self.residual_energy_fractions
            ),
            "directional_residual_variance_hex": _float_hex_tuple(
                self.directional_residual_variance
            ),
            "next_residual_eigenvalue_hex": float(
                self.next_residual_eigenvalue
            ).hex(),
            "cutoff_spectral_gap_hex": float(self.cutoff_spectral_gap).hex(),
            "cutoff_relative_spectral_gap": (
                self.cutoff_spectral_gap
                / max(
                    self.residual_eigenvalues[-1],
                    torch.finfo(torch.float64).tiny,
                )
            ),
            "source_example_ids": self.source_example_ids,
            "source_family_ids": self.source_family_ids,
            "source_sequence_sha256s": self.source_sequence_sha256s,
            "directional_fisher_hex": (
                None
                if self.directional_fisher is None
                else _float_hex_tuple(self.directional_fisher)
            ),
            "fisher_relevance_hex": (
                None
                if self.fisher_relevance is None
                else _float_hex_tuple(self.fisher_relevance)
            ),
            "fisher_first_order_mode_coupling_hex": (
                None
                if self.fisher_first_order_mode_coupling is None
                else _float_hex_tuple(self.fisher_first_order_mode_coupling)
            ),
            "fisher_rank_order": self.fisher_rank_order,
            "fisher_relevance_definition": (
                "unweighted_directional_residual_variance_times_"
                "family_balanced_directional_empirical_fisher"
            ),
            "basis_fit_covariance": (
                _FISHER_ALIGNMENT_TILTED_COVARIANCE
                if self.fit_weighting == "fisher_alignment_tilted"
                else _UNWEIGHTED_COVARIANCE
            ),
            "residual_eigenvalue_semantics": (
                "eigenvalue_of_fisher_alignment_tilted_basis_fit_covariance"
                if self.fit_weighting == "fisher_alignment_tilted"
                else "eigenvalue_of_unweighted_basis_fit_covariance"
            ),
            "euclidean_basis_reordered_by_fisher": False,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def fit_complete_h4_projection_basis(
    sequences: Iterable[CompleteH4ProjectionFitSequence],
    *,
    max_rank: int = 64,
    fit_weighting: ProjectionFitWeighting = "fisher_alignment_tilted",
) -> CompleteH4ProjectionBasis:
    """Fit a family-balanced residual basis under the selected weighting."""

    values = _validated_sequences(sequences)
    if fit_weighting not in ("fisher_alignment_tilted", "unweighted"):
        raise ValueError(
            "projection fit_weighting must be "
            "'fisher_alignment_tilted' or 'unweighted'"
        )
    width = values[0].width
    requested_rank = _strict_positive_int(max_rank, label="projection max_rank")
    effective_rank = min(requested_rank, width)
    covariance, unweighted_covariance, macro_fisher = (
        _family_example_macro_moments(
            values,
            fit_weighting=fit_weighting,
        )
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    negative_tolerance = max(1.0, float(torch.max(torch.abs(eigenvalues)).item())) * 1e-10
    if float(torch.min(eigenvalues).item()) < -negative_tolerance:
        raise RuntimeError("family-balanced residual second moment is not positive semidefinite")
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    total_energy = float(eigenvalues.sum().item())
    if total_energy <= torch.finfo(torch.float64).eps:
        raise ValueError("complete-H4 residual rows have zero family-balanced energy")

    selected_values = eigenvalues[:effective_rank]
    next_eigenvalue = (
        float(eigenvalues[effective_rank])
        if effective_rank < int(eigenvalues.numel())
        else 0.0
    )
    cutoff_spectral_gap = max(
        0.0,
        float(selected_values[-1]) - next_eigenvalue,
    )
    basis_rows = eigenvectors[:, :effective_rank].transpose(0, 1).contiguous()
    basis_rows = canonicalize_orthonormal_basis_signs(basis_rows)
    fractions = selected_values / total_energy
    residual_variance_values = torch.einsum(
        "kw,wx,kx->k",
        basis_rows,
        unweighted_covariance,
        basis_rows,
    ).clamp_min(0.0)

    directional_fisher: tuple[float, ...] | None = None
    fisher_relevance: tuple[float, ...] | None = None
    fisher_coupling: tuple[float, ...] | None = None
    fisher_order: tuple[int, ...] | None = None
    if values[0].has_gradients:
        assert macro_fisher is not None
        fisher_values = torch.einsum(
            "kw,wx,kx->k",
            basis_rows,
            macro_fisher,
            basis_rows,
        ).clamp_min(0.0)
        coupling_values = torch.zeros(effective_rank, dtype=torch.float64)
        coupling_by_family: dict[str, Tensor] = {}
        family_examples: dict[str, int] = defaultdict(int)
        for sequence in values:
            residual = sequence.residual_rows.to_tensor()  # type: ignore[union-attr]
            assert sequence.gradient_rows is not None
            gradient = sequence.gradient_rows.to_tensor()  # type: ignore[union-attr]
            residual_coordinates = residual @ basis_rows.transpose(0, 1)
            gradient_coordinates = gradient @ basis_rows.transpose(0, 1)
            example_coupling = (
                residual_coordinates * gradient_coordinates
            ).mean(dim=0)
            coupling_by_family[sequence.family_id] = (
                coupling_by_family.get(
                    sequence.family_id,
                    torch.zeros_like(example_coupling),
                )
                + example_coupling
            )
            family_examples[sequence.family_id] += 1
        coupling_values = sum(
            coupling_by_family[family_id] / family_examples[family_id]
            for family_id in sorted(coupling_by_family)
        ) / len(coupling_by_family)
        relevance_values = residual_variance_values * fisher_values
        directional_fisher = tuple(float(value) for value in fisher_values.tolist())
        fisher_relevance = tuple(float(value) for value in relevance_values.tolist())
        fisher_coupling = tuple(float(value) for value in coupling_values.tolist())
        fisher_order = tuple(
            sorted(
                range(effective_rank),
                key=lambda index: (-fisher_relevance[index], index),
            )
        )

    return CompleteH4ProjectionBasis(
        width=width,
        max_rank=effective_rank,
        basis_rows=basis_rows,
        residual_eigenvalues=tuple(float(value) for value in selected_values.tolist()),
        residual_energy_fractions=tuple(float(value) for value in fractions.tolist()),
        directional_residual_variance=tuple(
            float(value) for value in residual_variance_values.tolist()
        ),
        next_residual_eigenvalue=next_eigenvalue,
        cutoff_spectral_gap=cutoff_spectral_gap,
        source_example_ids=tuple(value.example_id for value in values),
        source_family_ids=tuple(sorted({value.family_id for value in values})),
        source_sequence_sha256s=tuple(value.sequence_sha256 for value in values),
        directional_fisher=directional_fisher,
        fisher_relevance=fisher_relevance,
        fisher_first_order_mode_coupling=fisher_coupling,
        fisher_rank_order=fisher_order,
        fit_weighting=fit_weighting,
    )


def canonical_complete_h4_rank_grid(
    max_rank: int,
    ranks: Iterable[int] = COMPLETE_H4_DEFAULT_RANK_GRID,
) -> tuple[int, ...]:
    """Cap a positive rank grid at the available width/rank, removing repeats."""

    available = _strict_positive_int(max_rank, label="available projection rank")
    requested = tuple(ranks)
    if not requested:
        raise ValueError("projection rank grid must be nonempty")
    capped: list[int] = []
    for value in requested:
        rank = _strict_positive_int(value, label="projection grid rank")
        candidate = min(rank, available)
        if candidate not in capped:
            capped.append(candidate)
    return tuple(sorted(capped))


def _projection_input(value: MatrixInput, *, width: int) -> tuple[Tensor, torch.device, torch.dtype]:
    if isinstance(value, ImmutableFloat64Matrix):
        rows = value.to_tensor()
        device = torch.device("cpu")
        dtype = torch.float64
    elif isinstance(value, Tensor):
        if value.ndim != 2 or value.shape[0] <= 0:
            raise ValueError("projection residual_rows must have nonempty shape [N,D]")
        if value.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise TypeError("projection residual_rows must have a floating dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("projection residual_rows must contain only finite values")
        device = value.device
        dtype = value.dtype
        rows = value
    else:
        raise TypeError("projection residual_rows must be a tensor or immutable matrix")
    if rows.shape[1] != width:
        raise ValueError("projection residual_rows width does not match basis")
    return rows, device, dtype


def project_complete_h4_residual_rows(
    residual_rows: MatrixInput,
    basis: CompleteH4ProjectionBasis,
    *,
    rank: int,
    ordering: ProjectionOrdering = "euclidean",
) -> Tensor:
    """Return the orthogonal rank-limited correction projection."""

    if not isinstance(basis, CompleteH4ProjectionBasis):
        raise TypeError("basis must be CompleteH4ProjectionBasis")
    selected_rank = _strict_positive_int(rank, label="projection rank")
    if selected_rank > basis.max_rank:
        raise ValueError("projection rank exceeds fitted basis max_rank")
    rows, device, dtype = _projection_input(residual_rows, width=basis.width)
    directions = basis.basis_tensor(ordering=ordering)[:selected_rank].to(
        device=device,
        dtype=dtype,
    )
    return (rows @ directions.transpose(0, 1)) @ directions


def _family_example_macro_scalar(
    values_by_sequence: Sequence[tuple[str, Tensor]],
) -> float:
    family_values: dict[str, list[float]] = defaultdict(list)
    for family_id, value in values_by_sequence:
        family_values[family_id].append(float(value.mean().item()))
    return sum(
        sum(values) / len(values)
        for values in family_values.values()
    ) / len(family_values)


def _row_weighted_scalar(
    values_by_sequence: Sequence[tuple[str, Tensor]],
) -> float:
    total = sum(float(value.sum().item()) for _, value in values_by_sequence)
    count = sum(value.numel() for _, value in values_by_sequence)
    return total / count


@dataclass(frozen=True, slots=True)
class CompleteH4ProjectionRankGeometry:
    """Scalar geometry for one rank; no row-level values are retained."""

    rank: int
    coefficient_count: int
    family_balanced_residual_energy_retention: float
    family_balanced_residual_rmse: float
    row_weighted_residual_energy_retention: float
    row_weighted_residual_rmse: float
    fisher_first_order_residual_coupling: float | None
    fisher_first_order_error_coupling: float | None
    fisher_absolute_first_order_residual_coupling: float | None
    fisher_absolute_first_order_error_coupling: float | None

    def __post_init__(self) -> None:
        _strict_positive_int(self.rank, label="projection geometry rank")
        _strict_positive_int(
            self.coefficient_count,
            label="projection geometry coefficient_count",
        )
        for name in (
            "family_balanced_residual_energy_retention",
            "row_weighted_residual_energy_retention",
        ):
            value = _finite_float(getattr(self, name), label=name)
            if not -1e-9 <= value <= 1.0 + 1e-9:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, min(1.0, max(0.0, value)))
        for name in (
            "family_balanced_residual_rmse",
            "row_weighted_residual_rmse",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), label=name, nonnegative=True),
            )
        fisher = (
            self.fisher_first_order_residual_coupling,
            self.fisher_first_order_error_coupling,
            self.fisher_absolute_first_order_residual_coupling,
            self.fisher_absolute_first_order_error_coupling,
        )
        if any(value is None for value in fisher) and not all(value is None for value in fisher):
            raise ValueError("projection geometry Fisher coupling fields must be all present or absent")
        for name in (
            "fisher_first_order_residual_coupling",
            "fisher_first_order_error_coupling",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, label=name))
        for name in (
            "fisher_absolute_first_order_residual_coupling",
            "fisher_absolute_first_order_error_coupling",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_float(value, label=name, nonnegative=True),
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "coefficient_count": self.coefficient_count,
            "family_balanced_residual_energy_retention": (
                self.family_balanced_residual_energy_retention
            ),
            "family_balanced_residual_rmse": self.family_balanced_residual_rmse,
            "row_weighted_residual_energy_retention": (
                self.row_weighted_residual_energy_retention
            ),
            "row_weighted_residual_rmse": self.row_weighted_residual_rmse,
            "fisher_first_order_residual_coupling": (
                self.fisher_first_order_residual_coupling
            ),
            "fisher_first_order_error_coupling": (
                self.fisher_first_order_error_coupling
            ),
            "fisher_absolute_first_order_residual_coupling": (
                self.fisher_absolute_first_order_residual_coupling
            ),
            "fisher_absolute_first_order_error_coupling": (
                self.fisher_absolute_first_order_error_coupling
            ),
        }


@dataclass(frozen=True, slots=True)
class CompleteH4ProjectionGeometry:
    """Hash-bound scalar-only rank-grid evaluation."""

    basis_artifact_sha256: str
    ordering: ProjectionOrdering
    evaluation_sequence_sha256s: tuple[str, ...]
    evaluation_family_ids: tuple[str, ...]
    rank_rows: tuple[CompleteH4ProjectionRankGeometry, ...]
    geometry_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.basis_artifact_sha256, str) or len(self.basis_artifact_sha256) != 64:
            raise ValueError("geometry basis_artifact_sha256 is invalid")
        if self.ordering not in {"euclidean", "fisher"}:
            raise ValueError("geometry ordering must be 'euclidean' or 'fisher'")
        if not self.evaluation_sequence_sha256s or any(
            not isinstance(value, str) or len(value) != 64
            for value in self.evaluation_sequence_sha256s
        ):
            raise ValueError("geometry evaluation sequence receipts are invalid")
        if not self.evaluation_family_ids or tuple(sorted(set(self.evaluation_family_ids))) != self.evaluation_family_ids:
            raise ValueError("geometry evaluation family ids must be sorted and unique")
        if not self.rank_rows or any(
            not isinstance(value, CompleteH4ProjectionRankGeometry)
            for value in self.rank_rows
        ):
            raise ValueError("geometry rank_rows must be nonempty")
        ranks = tuple(value.rank for value in self.rank_rows)
        if tuple(sorted(set(ranks))) != ranks:
            raise ValueError("geometry ranks must be strictly increasing")
        object.__setattr__(self, "geometry_sha256", _sha256(_GEOMETRY_DOMAIN, self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "basis_artifact_sha256": self.basis_artifact_sha256,
            "ordering": self.ordering,
            "evaluation_sequence_sha256s": self.evaluation_sequence_sha256s,
            "evaluation_family_ids": self.evaluation_family_ids,
            "rank_rows": tuple(value.to_dict() for value in self.rank_rows),
            "safety": {
                "raw_prompts_retained": False,
                "raw_token_ids_retained": False,
                "raw_logits_retained": False,
                "row_level_activations_retained": False,
                "model_weights_retained": False,
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "geometry_sha256": self.geometry_sha256}


def summarize_complete_h4_projection_geometry(
    sequences: Iterable[CompleteH4ProjectionFitSequence],
    basis: CompleteH4ProjectionBasis,
    *,
    ranks: Iterable[int] = COMPLETE_H4_DEFAULT_RANK_GRID,
    ordering: ProjectionOrdering = "euclidean",
) -> CompleteH4ProjectionGeometry:
    """Evaluate scalar residual and NLL-gradient geometry over a rank grid."""

    values = _validated_sequences(sequences)
    if values[0].width != basis.width:
        raise ValueError("geometry sequence width does not match projection basis")
    if ordering == "fisher" and not basis.has_fisher:
        raise ValueError("Fisher geometry ordering requires a Fisher-ranked basis")
    grid = canonical_complete_h4_rank_grid(basis.max_rank, ranks)
    residual_sequences = tuple(
        (
            value.family_id,
            value.residual_rows.to_tensor(),  # type: ignore[union-attr]
            (
                None
                if value.gradient_rows is None
                else value.gradient_rows.to_tensor()  # type: ignore[union-attr]
            ),
        )
        for value in values
    )
    residual_energy_by_sequence = tuple(
        (family_id, rows.square().sum(dim=1))
        for family_id, rows, _ in residual_sequences
    )
    family_residual_energy = _family_example_macro_scalar(
        residual_energy_by_sequence
    )
    row_residual_energy = _row_weighted_scalar(residual_energy_by_sequence)

    rank_rows: list[CompleteH4ProjectionRankGeometry] = []
    for rank in grid:
        error_by_sequence: list[tuple[str, Tensor]] = []
        signed_residual_by_sequence: list[tuple[str, Tensor]] = []
        signed_error_by_sequence: list[tuple[str, Tensor]] = []
        absolute_residual_by_sequence: list[tuple[str, Tensor]] = []
        absolute_error_by_sequence: list[tuple[str, Tensor]] = []
        for family_id, residual, gradient in residual_sequences:
            projected = project_complete_h4_residual_rows(
                residual,
                basis,
                rank=rank,
                ordering=ordering,
            )
            error = residual - projected
            error_by_sequence.append((family_id, error.square().sum(dim=1)))
            if gradient is not None:
                residual_coupling = (gradient * residual).sum(dim=1)
                error_coupling = (gradient * error).sum(dim=1)
                signed_residual_by_sequence.append((family_id, residual_coupling))
                signed_error_by_sequence.append((family_id, error_coupling))
                absolute_residual_by_sequence.append((family_id, residual_coupling.abs()))
                absolute_error_by_sequence.append((family_id, error_coupling.abs()))

        family_error_energy = _family_example_macro_scalar(error_by_sequence)
        row_error_energy = _row_weighted_scalar(error_by_sequence)
        family_retention = (
            1.0
            if family_residual_energy == 0.0
            else 1.0 - family_error_energy / family_residual_energy
        )
        row_retention = (
            1.0
            if row_residual_energy == 0.0
            else 1.0 - row_error_energy / row_residual_energy
        )
        rank_rows.append(
            CompleteH4ProjectionRankGeometry(
                rank=rank,
                coefficient_count=rank * basis.width,
                family_balanced_residual_energy_retention=family_retention,
                family_balanced_residual_rmse=math.sqrt(
                    max(0.0, family_error_energy) / basis.width
                ),
                row_weighted_residual_energy_retention=row_retention,
                row_weighted_residual_rmse=math.sqrt(
                    max(0.0, row_error_energy) / basis.width
                ),
                fisher_first_order_residual_coupling=(
                    None
                    if not signed_residual_by_sequence
                    else _family_example_macro_scalar(signed_residual_by_sequence)
                ),
                fisher_first_order_error_coupling=(
                    None
                    if not signed_error_by_sequence
                    else _family_example_macro_scalar(signed_error_by_sequence)
                ),
                fisher_absolute_first_order_residual_coupling=(
                    None
                    if not absolute_residual_by_sequence
                    else _family_example_macro_scalar(absolute_residual_by_sequence)
                ),
                fisher_absolute_first_order_error_coupling=(
                    None
                    if not absolute_error_by_sequence
                    else _family_example_macro_scalar(absolute_error_by_sequence)
                ),
            )
        )

    return CompleteH4ProjectionGeometry(
        basis_artifact_sha256=basis.artifact_sha256,
        ordering=ordering,
        evaluation_sequence_sha256s=tuple(value.sequence_sha256 for value in values),
        evaluation_family_ids=tuple(sorted({value.family_id for value in values})),
        rank_rows=tuple(rank_rows),
    )
