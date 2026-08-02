"""Deterministic tail-informed complete-H4 projection geometry.

The treatment ordering preserves the first 192 rows of an authenticated
unweighted global basis, appends the complete numerical row span of the
structural-tail residual left by that prefix, and then fills to rank 320 from
the later global directions with deterministic two-pass modified
Gram--Schmidt (MGS).  All linear algebra is CPU float64.

The numerical SVD rank tolerance is fixed by this module's format contract:
``max(N, D) * eps(float64) * largest_singular_value``.  This is the standard
backward-error threshold for the ordered ``N x D`` tail-residual matrix.  It
is deliberately not tunable by callers, so identical authenticated inputs
cannot silently produce different treatment bases.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
from torch import Tensor

from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionBasis,
    CompleteH4ProjectionFitSequence,
    ImmutableFloat64Matrix,
    MatrixInput,
    canonicalize_orthonormal_basis_signs,
)


__all__ = [
    "TAIL_INFORMED_PROJECTION_ORDERING",
    "CompleteH4TailInformedProjectionFit",
    "CompleteH4TailProjectionTrace",
    "fit_complete_h4_tail_informed_projection",
]


TAIL_INFORMED_PROJECTION_ORDERING = (
    "unweighted_u192_then_tail_residual_svd_span_then_mgs_u320"
)

_TRACE_DOMAIN = b"fisher-graph:complete-h4:tail-trace:v1\0"
_MASK_DOMAIN = b"fisher-graph:complete-h4:tail-graph-core-mask:v1\0"
_FIT_DOMAIN = b"fisher-graph:complete-h4:tail-informed-fit:v1\0"
_PREFIX_DOMAIN = b"fisher-graph:complete-h4:tail-informed-prefix:v1\0"
_LINEAGE_DOMAIN = b"fisher-graph:complete-h4:tail-informed-lineage:v1\0"
_MGS_RELATIVE_TOLERANCE = 128.0 * torch.finfo(torch.float64).eps


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _receipt(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _immutable_matrix(value: MatrixInput, *, label: str) -> ImmutableFloat64Matrix:
    if isinstance(value, ImmutableFloat64Matrix):
        return ImmutableFloat64Matrix(
            row_count=value.row_count,
            width=value.width,
            _little_endian_bytes=value._little_endian_bytes,
        )
    return ImmutableFloat64Matrix.from_tensor(value, label=label)


def _canonical_mask(value: Sequence[bool] | Tensor, *, rows: int) -> tuple[bool, ...]:
    if isinstance(value, Tensor):
        if value.ndim != 1 or int(value.numel()) != rows:
            raise ValueError("graph_core_mask must have shape [row_count]")
        if value.dtype != torch.bool:
            raise TypeError("graph_core_mask tensor must have bool dtype")
        result = tuple(bool(item) for item in value.detach().to("cpu").tolist())
    else:
        result = tuple(value)
        if len(result) != rows or any(type(item) is not bool for item in result):
            raise ValueError("graph_core_mask must contain one bool per residual row")
    return result


@dataclass(frozen=True, slots=True)
class CompleteH4TailProjectionTrace:
    """One authenticated residual trace and its structural core/tail split."""

    example_id: str
    family_id: str
    residual_rows: MatrixInput = field(repr=False)
    graph_core_mask: Sequence[bool] | Tensor = field(repr=False)
    source_sequence_sha256: str
    source_pair_sha256: str
    source_graph_core_mask_sha256: str
    graph_core_mask_sha256: str = field(init=False)
    trace_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="tail trace example_id")
        _identifier(self.family_id, label="tail trace family_id")
        rows = _immutable_matrix(self.residual_rows, label="tail trace residual_rows")
        mask = _canonical_mask(self.graph_core_mask, rows=rows.row_count)
        sequence_sha = _sha256(
            self.source_sequence_sha256,
            label="tail trace source_sequence_sha256",
        )
        pair_sha = _sha256(
            self.source_pair_sha256,
            label="tail trace source_pair_sha256",
        )
        source_mask_sha = _sha256(
            self.source_graph_core_mask_sha256,
            label="tail trace source_graph_core_mask_sha256",
        )
        mask_sha = _receipt(
            _MASK_DOMAIN,
            {"row_count": rows.row_count, "graph_core_mask": mask},
        )
        payload = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "residual_rows_sha256": rows.matrix_sha256,
            "row_count": rows.row_count,
            "width": rows.width,
            "graph_core_rows": sum(mask),
            "structural_tail_rows": len(mask) - sum(mask),
            "graph_core_mask_sha256": mask_sha,
            "source_sequence_sha256": sequence_sha,
            "source_pair_sha256": pair_sha,
            "source_graph_core_mask_sha256": source_mask_sha,
        }
        object.__setattr__(self, "residual_rows", rows)
        object.__setattr__(self, "graph_core_mask", mask)
        object.__setattr__(self, "source_sequence_sha256", sequence_sha)
        object.__setattr__(self, "source_pair_sha256", pair_sha)
        object.__setattr__(self, "source_graph_core_mask_sha256", source_mask_sha)
        object.__setattr__(self, "graph_core_mask_sha256", mask_sha)
        object.__setattr__(self, "trace_sha256", _receipt(_TRACE_DOMAIN, payload))

    @classmethod
    def from_fit_sequence(
        cls,
        sequence: CompleteH4ProjectionFitSequence,
        graph_core_mask: Sequence[bool] | Tensor,
        *,
        source_pair_sha256: str,
        source_graph_core_mask_sha256: str,
    ) -> "CompleteH4TailProjectionTrace":
        if not isinstance(sequence, CompleteH4ProjectionFitSequence):
            raise TypeError("sequence must be CompleteH4ProjectionFitSequence")
        return cls(
            example_id=sequence.example_id,
            family_id=sequence.family_id,
            residual_rows=sequence.residual_rows,
            graph_core_mask=graph_core_mask,
            source_sequence_sha256=sequence.sequence_sha256,
            source_pair_sha256=source_pair_sha256,
            source_graph_core_mask_sha256=source_graph_core_mask_sha256,
        )

    @property
    def row_count(self) -> int:
        return self.residual_rows.row_count  # type: ignore[union-attr]

    @property
    def width(self) -> int:
        return self.residual_rows.width  # type: ignore[union-attr]

    @property
    def tail_row_count(self) -> int:
        return len(self.graph_core_mask) - sum(self.graph_core_mask)  # type: ignore[arg-type]

    def tail_rows_tensor(self) -> Tensor:
        rows = self.residual_rows.to_tensor()  # type: ignore[union-attr]
        mask = torch.tensor(self.graph_core_mask, dtype=torch.bool)
        return rows[~mask]

    def metadata(self) -> dict[str, object]:
        rows = self.residual_rows
        assert isinstance(rows, ImmutableFloat64Matrix)
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "row_count": self.row_count,
            "width": self.width,
            "graph_core_rows": self.row_count - self.tail_row_count,
            "structural_tail_rows": self.tail_row_count,
            "residual_rows_sha256": rows.matrix_sha256,
            "graph_core_mask_sha256": self.graph_core_mask_sha256,
            "source_sequence_sha256": self.source_sequence_sha256,
            "source_pair_sha256": self.source_pair_sha256,
            "source_graph_core_mask_sha256": self.source_graph_core_mask_sha256,
            "trace_sha256": self.trace_sha256,
        }


def _canonicalize_vector_sign(vector: Tensor) -> Tensor:
    result = vector.detach().to(device="cpu", dtype=torch.float64).clone()
    pivot = int(torch.argmax(torch.abs(result)).item())
    if float(result[pivot]) < 0.0:
        result.neg_()
    return result


def _two_pass_mgs(vector: Tensor, accepted_rows: list[Tensor]) -> tuple[Tensor, float]:
    """Orthogonalize in a fixed row order with exactly two MGS passes."""

    candidate = vector.detach().to(device="cpu", dtype=torch.float64).clone()
    original_norm = float(torch.linalg.vector_norm(candidate).item())
    for _ in range(2):
        for row in accepted_rows:
            candidate.add_(row, alpha=-float(torch.dot(candidate, row).item()))
    norm = float(torch.linalg.vector_norm(candidate).item())
    threshold = max(1.0, original_norm) * _MGS_RELATIVE_TOLERANCE
    if norm <= threshold:
        return candidate, norm
    candidate.div_(norm)
    return _canonicalize_vector_sign(candidate), norm


@dataclass(frozen=True, slots=True)
class CompleteH4TailInformedProjectionFit:
    """Hash-bound U192 + structural-tail-span + later-global treatment basis."""

    width: int
    anchor_rank: int
    tail_rank: int
    max_rank: int
    treatment_basis_rows: MatrixInput = field(repr=False)
    global_fit_basis_artifact_sha256: str
    global_u192_matrix_sha256: str
    global_u320_matrix_sha256: str
    tail_basis_matrix_sha256: str
    source_trace_sha256s: tuple[str, ...]
    source_sequence_sha256s: tuple[str, ...]
    source_pair_sha256s: tuple[str, ...]
    source_graph_core_mask_sha256s: tuple[str, ...]
    graph_core_mask_sha256s: tuple[str, ...]
    source_example_count: int
    source_row_count: int
    source_tail_row_count: int
    tail_largest_singular_value: float
    tail_smallest_retained_singular_value: float
    tail_svd_absolute_tolerance: float
    tail_reconstruction_max_abs_error: float
    tail_reconstruction_l2_error: float
    tail_reconstruction_absolute_tolerance: float
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        width = _positive_int(self.width, label="tail-informed width")
        anchor_rank = _positive_int(
            self.anchor_rank,
            label="tail-informed anchor_rank",
        )
        tail_rank = _positive_int(self.tail_rank, label="tail-informed tail_rank")
        max_rank = _positive_int(self.max_rank, label="tail-informed max_rank")
        if anchor_rank != 192 or max_rank != 320:
            raise ValueError("tail-informed v1 requires anchor_rank=192 and max_rank=320")
        if not anchor_rank + tail_rank <= max_rank <= width:
            raise ValueError("tail-informed ranks are inconsistent with width")
        matrix = _immutable_matrix(
            self.treatment_basis_rows,
            label="tail-informed treatment_basis_rows",
        )
        if matrix.shape != (max_rank, width):
            raise ValueError("tail-informed treatment basis must have shape [320,width]")
        basis = matrix.to_tensor()
        identity = torch.eye(max_rank, dtype=torch.float64)
        if not bool(torch.allclose(basis @ basis.T, identity, atol=2e-10, rtol=2e-10)):
            raise ValueError("tail-informed treatment basis must be orthonormal")
        if not bool(torch.equal(canonicalize_orthonormal_basis_signs(basis), basis)):
            raise ValueError("tail-informed treatment basis signs are not canonical")

        digest_fields = (
            (self.global_fit_basis_artifact_sha256, "global fit basis artifact"),
            (self.global_u192_matrix_sha256, "global U192 matrix"),
            (self.global_u320_matrix_sha256, "global U320 matrix"),
            (self.tail_basis_matrix_sha256, "tail basis matrix"),
        )
        for value, label in digest_fields:
            _sha256(value, label=label)
        receipt_groups = (
            self.source_trace_sha256s,
            self.source_sequence_sha256s,
            self.source_pair_sha256s,
            self.source_graph_core_mask_sha256s,
            self.graph_core_mask_sha256s,
        )
        if any(len(group) != self.source_example_count for group in receipt_groups):
            raise ValueError("tail-informed source receipt lengths must equal example count")
        for group in receipt_groups:
            for digest in group:
                _sha256(digest, label="tail-informed source receipt")
        _positive_int(self.source_example_count, label="tail-informed source_example_count")
        _positive_int(self.source_row_count, label="tail-informed source_row_count")
        _positive_int(self.source_tail_row_count, label="tail-informed source_tail_row_count")
        if self.source_tail_row_count > self.source_row_count:
            raise ValueError("tail-informed tail rows exceed all source rows")

        largest = _finite_nonnegative(
            self.tail_largest_singular_value,
            label="tail largest singular value",
        )
        smallest = _finite_nonnegative(
            self.tail_smallest_retained_singular_value,
            label="tail smallest retained singular value",
        )
        svd_tolerance = _finite_nonnegative(
            self.tail_svd_absolute_tolerance,
            label="tail SVD absolute tolerance",
        )
        max_error = _finite_nonnegative(
            self.tail_reconstruction_max_abs_error,
            label="tail reconstruction max error",
        )
        l2_error = _finite_nonnegative(
            self.tail_reconstruction_l2_error,
            label="tail reconstruction L2 error",
        )
        reconstruction_tolerance = _finite_nonnegative(
            self.tail_reconstruction_absolute_tolerance,
            label="tail reconstruction absolute tolerance",
        )
        if not largest >= smallest > svd_tolerance > 0.0:
            raise ValueError("tail singular-value rank proof is inconsistent")
        if max_error > reconstruction_tolerance:
            raise ValueError("structural-tail reconstruction is not exact in float64")

        actual_u192 = ImmutableFloat64Matrix.from_tensor(
            basis[:anchor_rank],
            label="tail-informed U192 prefix",
        ).matrix_sha256
        if actual_u192 != self.global_u192_matrix_sha256:
            raise ValueError("tail-informed treatment basis does not preserve global U192")
        actual_tail = ImmutableFloat64Matrix.from_tensor(
            basis[anchor_rank : anchor_rank + tail_rank],
            label="tail-informed tail basis",
        ).matrix_sha256
        if actual_tail != self.tail_basis_matrix_sha256:
            raise ValueError("tail-informed tail basis receipt differs")

        object.__setattr__(self, "treatment_basis_rows", matrix)
        object.__setattr__(self, "tail_largest_singular_value", largest)
        object.__setattr__(self, "tail_smallest_retained_singular_value", smallest)
        object.__setattr__(self, "tail_svd_absolute_tolerance", svd_tolerance)
        object.__setattr__(self, "tail_reconstruction_max_abs_error", max_error)
        object.__setattr__(self, "tail_reconstruction_l2_error", l2_error)
        object.__setattr__(
            self,
            "tail_reconstruction_absolute_tolerance",
            reconstruction_tolerance,
        )
        expected_artifact = _receipt(_FIT_DOMAIN, self._payload())
        if self.artifact_sha256:
            if _sha256(self.artifact_sha256, label="tail-informed artifact") != expected_artifact:
                raise ValueError("tail-informed artifact receipt differs")
        else:
            object.__setattr__(self, "artifact_sha256", expected_artifact)

    def _payload(self) -> dict[str, object]:
        matrix = self.treatment_basis_rows
        assert isinstance(matrix, ImmutableFloat64Matrix)
        return {
            "schema": "fisher_graph.complete_h4_tail_informed_projection_fit",
            "format_version": 1,
            "ordering": TAIL_INFORMED_PROJECTION_ORDERING,
            "dtype": "cpu_float64",
            "width": self.width,
            "anchor_rank": self.anchor_rank,
            "tail_rank": self.tail_rank,
            "tail_boundary_rank": self.anchor_rank + self.tail_rank,
            "max_rank": self.max_rank,
            "treatment_basis_rows": matrix.metadata(),
            "global_fit_basis_artifact_sha256": self.global_fit_basis_artifact_sha256,
            "global_u192_matrix_sha256": self.global_u192_matrix_sha256,
            "global_u320_matrix_sha256": self.global_u320_matrix_sha256,
            "tail_basis_matrix_sha256": self.tail_basis_matrix_sha256,
            "source_trace_sha256s": self.source_trace_sha256s,
            "source_sequence_sha256s": self.source_sequence_sha256s,
            "source_pair_sha256s": self.source_pair_sha256s,
            "source_graph_core_mask_sha256s": self.source_graph_core_mask_sha256s,
            "graph_core_mask_sha256s": self.graph_core_mask_sha256s,
            "source_example_count": self.source_example_count,
            "source_row_count": self.source_row_count,
            "source_tail_row_count": self.source_tail_row_count,
            "tail_largest_singular_value_hex": self.tail_largest_singular_value.hex(),
            "tail_smallest_retained_singular_value_hex": (
                self.tail_smallest_retained_singular_value.hex()
            ),
            "tail_svd_absolute_tolerance_hex": self.tail_svd_absolute_tolerance.hex(),
            "tail_reconstruction_max_abs_error_hex": (
                self.tail_reconstruction_max_abs_error.hex()
            ),
            "tail_reconstruction_l2_error_hex": self.tail_reconstruction_l2_error.hex(),
            "tail_reconstruction_absolute_tolerance_hex": (
                self.tail_reconstruction_absolute_tolerance.hex()
            ),
            "tail_reconstruction_exact_float64": True,
            "tail_reconstruction_tolerance_definition": (
                "512*max(row_count,width)*float64_epsilon*max(1,max_abs_source)"
            ),
            "tail_svd_rank_tolerance_definition": (
                "max(row_count,width)*float64_epsilon*largest_singular_value"
            ),
            "global_fill_orthogonalization": "fixed_order_two_pass_modified_gram_schmidt",
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def validate_integrity(self) -> bool:
        if _receipt(_FIT_DOMAIN, self._payload()) != self.artifact_sha256:
            raise ValueError("tail-informed artifact failed integrity validation")
        # Rebuild the immutable wrapper to revalidate byte length, finiteness,
        # and its content receipt independently of the dataclass construction.
        matrix = self.treatment_basis_rows
        assert isinstance(matrix, ImmutableFloat64Matrix)
        _immutable_matrix(matrix, label="tail-informed treatment basis integrity")
        return True

    def basis_tensor(self, rank: int) -> Tensor:
        selected_rank = _positive_int(rank, label="tail-informed prefix rank")
        if selected_rank > self.max_rank:
            raise ValueError("tail-informed prefix rank exceeds max_rank")
        return self.treatment_basis_rows.to_tensor()[:selected_rank]  # type: ignore[union-attr]

    def prefix_artifact_sha256(self, rank: int) -> str:
        prefix = ImmutableFloat64Matrix.from_tensor(
            self.basis_tensor(rank),
            label="tail-informed prefix",
        )
        return _receipt(
            _PREFIX_DOMAIN,
            {
                "fit_artifact_sha256": self.artifact_sha256,
                "ordering": TAIL_INFORMED_PROJECTION_ORDERING,
                "rank": rank,
                "prefix_matrix_sha256": prefix.matrix_sha256,
            },
        )

    def lineage(
        self,
        rank: int,
        execution_basis_artifact_sha256: str,
    ) -> dict[str, object]:
        execution_sha = _sha256(
            execution_basis_artifact_sha256,
            label="tail-informed execution basis artifact",
        )
        prefix = ImmutableFloat64Matrix.from_tensor(
            self.basis_tensor(rank),
            label="tail-informed lineage prefix",
        )
        payload = {
            "schema": "fisher_graph.complete_h4_tail_informed_projection_lineage",
            "format_version": 1,
            "ordering": TAIL_INFORMED_PROJECTION_ORDERING,
            "rank": rank,
            "tail_rank": self.tail_rank,
            "fit_artifact_sha256": self.artifact_sha256,
            "global_fit_basis_artifact_sha256": self.global_fit_basis_artifact_sha256,
            "prefix_matrix_sha256": prefix.matrix_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256(rank),
            "execution_basis_artifact_sha256": execution_sha,
        }
        return {**payload, "lineage_sha256": _receipt(_LINEAGE_DOMAIN, payload)}


def fit_complete_h4_tail_informed_projection(
    traces: Iterable[CompleteH4TailProjectionTrace],
    global_basis: CompleteH4ProjectionBasis,
    *,
    anchor_rank: int = 192,
    max_rank: int = 320,
) -> CompleteH4TailInformedProjectionFit:
    """Build the deterministic U192 + tail span + later-global treatment basis.

    Traces are ordered by ``(family_id, example_id)`` and structural-tail rows
    retain their original within-example order.  The global basis must be the
    unweighted fit over exactly those authenticated sequence receipts.
    """

    if not isinstance(global_basis, CompleteH4ProjectionBasis):
        raise TypeError("global_basis must be CompleteH4ProjectionBasis")
    if global_basis.fit_weighting != "unweighted":
        raise ValueError("tail-informed projection requires an unweighted global basis")
    if anchor_rank != 192 or max_rank != 320:
        raise ValueError("tail-informed v1 requires anchor_rank=192 and max_rank=320")
    if global_basis.max_rank < max_rank or global_basis.width < max_rank:
        raise ValueError("tail-informed projection requires global max_rank >= 320")

    ordered = tuple(sorted(tuple(traces), key=lambda row: (row.family_id, row.example_id)))
    if not ordered:
        raise ValueError("tail-informed projection requires at least one trace")
    if any(not isinstance(row, CompleteH4TailProjectionTrace) for row in ordered):
        raise TypeError("tail-informed traces must be CompleteH4TailProjectionTrace")
    if len({row.example_id for row in ordered}) != len(ordered):
        raise ValueError("tail-informed trace example_id values must be unique")
    if {row.width for row in ordered} != {global_basis.width}:
        raise ValueError("tail-informed trace widths must match the global basis")
    if tuple(row.example_id for row in ordered) != global_basis.source_example_ids:
        raise ValueError("tail-informed traces differ from global basis example lineage")
    if tuple(row.source_sequence_sha256 for row in ordered) != (
        global_basis.source_sequence_sha256s
    ):
        raise ValueError("tail-informed traces differ from global basis sequence lineage")
    if tuple(sorted({row.family_id for row in ordered})) != global_basis.source_family_ids:
        raise ValueError("tail-informed traces differ from global basis family lineage")

    tail_parts = [row.tail_rows_tensor() for row in ordered if row.tail_row_count]
    if not tail_parts:
        raise ValueError("tail-informed projection requires structural-tail rows")
    tail_rows = torch.cat(tail_parts, dim=0).to(device="cpu", dtype=torch.float64)
    global_rows = global_basis.basis_tensor(ordering="euclidean").to(
        device="cpu",
        dtype=torch.float64,
    )
    global_u320 = global_rows[:max_rank].contiguous()
    anchor = global_u320[:anchor_rank].contiguous()

    # The subtraction is deliberately written as one ordered pair of matrix
    # products.  Runtimes and tests can reproduce these exact float64 rows.
    tail_after_anchor = tail_rows - (tail_rows @ anchor.T) @ anchor
    if not bool(torch.isfinite(tail_after_anchor).all().item()):
        raise ValueError("tail residual after U192 contains non-finite values")
    if float(torch.linalg.vector_norm(tail_after_anchor).item()) == 0.0:
        raise ValueError("structural-tail residual after U192 has zero energy")

    _, singular_values, right_rows = torch.linalg.svd(
        tail_after_anchor,
        full_matrices=False,
    )
    largest = float(singular_values[0].item())
    svd_tolerance = (
        max(int(tail_after_anchor.shape[0]), int(tail_after_anchor.shape[1]))
        * torch.finfo(torch.float64).eps
        * largest
    )
    numerical_rank = int((singular_values > svd_tolerance).sum().item())
    if numerical_rank <= 0:
        raise ValueError("structural-tail residual has zero numerical SVD rank")
    if anchor_rank + numerical_rank > max_rank:
        raise ValueError("structural-tail numerical rank exceeds treatment capacity")

    accepted: list[Tensor] = [row.clone() for row in anchor]
    tail_directions: list[Tensor] = []
    for candidate in right_rows[:numerical_rank]:
        direction, retained_norm = _two_pass_mgs(candidate, accepted)
        if retained_norm <= max(1.0, float(torch.linalg.vector_norm(candidate).item())) * (
            _MGS_RELATIVE_TOLERANCE
        ):
            raise RuntimeError("numerical tail direction collapsed during deterministic MGS")
        accepted.append(direction)
        tail_directions.append(direction)

    tail_basis = torch.stack(tail_directions, dim=0)
    tail_boundary = torch.stack(accepted, dim=0)
    reconstructed_tail = (tail_rows @ tail_boundary.T) @ tail_boundary
    reconstruction_error = tail_rows - reconstructed_tail
    max_abs_error = float(torch.max(torch.abs(reconstruction_error)).item())
    l2_error = float(torch.linalg.vector_norm(reconstruction_error).item())
    source_scale = max(1.0, float(torch.max(torch.abs(tail_rows)).item()))
    reconstruction_tolerance = (
        512.0
        * max(int(tail_rows.shape[0]), int(tail_rows.shape[1]))
        * torch.finfo(torch.float64).eps
        * source_scale
    )
    if max_abs_error > reconstruction_tolerance:
        raise RuntimeError("structural-tail span failed exact float64 reconstruction")

    # Try the later authenticated global directions in their original order.
    # Directions made redundant by the inserted tail span are skipped.
    for candidate in global_u320[anchor_rank:max_rank]:
        if len(accepted) == max_rank:
            break
        direction, retained_norm = _two_pass_mgs(candidate, accepted)
        threshold = max(1.0, float(torch.linalg.vector_norm(candidate).item())) * (
            _MGS_RELATIVE_TOLERANCE
        )
        if retained_norm > threshold:
            accepted.append(direction)
    if len(accepted) != max_rank:
        raise RuntimeError("later global directions could not fill treatment rank 320")

    treatment = torch.stack(accepted, dim=0).contiguous()
    # This validation cannot alter the first 192 values; it only rejects an
    # unexpected loss of orthogonality or noncanonical sign.
    if not bool(
        torch.allclose(
            treatment @ treatment.T,
            torch.eye(max_rank, dtype=torch.float64),
            atol=2e-10,
            rtol=2e-10,
        )
    ):
        raise RuntimeError("tail-informed treatment basis lost orthonormality")
    if not bool(torch.equal(treatment[:anchor_rank], anchor)):
        raise RuntimeError("tail-informed treatment construction altered U192")

    global_u192_matrix = ImmutableFloat64Matrix.from_tensor(
        anchor,
        label="global U192",
    )
    global_u320_matrix = ImmutableFloat64Matrix.from_tensor(
        global_u320,
        label="global U320",
    )
    tail_basis_matrix = ImmutableFloat64Matrix.from_tensor(
        tail_basis,
        label="structural-tail SVD span",
    )
    return CompleteH4TailInformedProjectionFit(
        width=global_basis.width,
        anchor_rank=anchor_rank,
        tail_rank=numerical_rank,
        max_rank=max_rank,
        treatment_basis_rows=treatment,
        global_fit_basis_artifact_sha256=global_basis.artifact_sha256,
        global_u192_matrix_sha256=global_u192_matrix.matrix_sha256,
        global_u320_matrix_sha256=global_u320_matrix.matrix_sha256,
        tail_basis_matrix_sha256=tail_basis_matrix.matrix_sha256,
        source_trace_sha256s=tuple(row.trace_sha256 for row in ordered),
        source_sequence_sha256s=tuple(row.source_sequence_sha256 for row in ordered),
        source_pair_sha256s=tuple(row.source_pair_sha256 for row in ordered),
        source_graph_core_mask_sha256s=tuple(
            row.source_graph_core_mask_sha256 for row in ordered
        ),
        graph_core_mask_sha256s=tuple(row.graph_core_mask_sha256 for row in ordered),
        source_example_count=len(ordered),
        source_row_count=sum(row.row_count for row in ordered),
        source_tail_row_count=sum(row.tail_row_count for row in ordered),
        tail_largest_singular_value=largest,
        tail_smallest_retained_singular_value=float(
            singular_values[numerical_rank - 1].item()
        ),
        tail_svd_absolute_tolerance=svd_tolerance,
        tail_reconstruction_max_abs_error=max_abs_error,
        tail_reconstruction_l2_error=l2_error,
        tail_reconstruction_absolute_tolerance=reconstruction_tolerance,
    )
