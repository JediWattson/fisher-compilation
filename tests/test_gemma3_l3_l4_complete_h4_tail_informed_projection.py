from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionBasis,
    CompleteH4ProjectionFitSequence,
)
from fisher_graph.gemma3_l3_l4_complete_h4_tail_informed_projection import (
    TAIL_INFORMED_PROJECTION_ORDERING,
    CompleteH4TailProjectionTrace,
    fit_complete_h4_tail_informed_projection,
)


_WIDTH = 322


def _row(*values: tuple[int, float]) -> torch.Tensor:
    result = torch.zeros(_WIDTH, dtype=torch.float64)
    for index, value in values:
        result[index] = value
    return result


def _fixture() -> tuple[
    CompleteH4ProjectionBasis,
    tuple[CompleteH4TailProjectionTrace, ...],
]:
    sequence_a = CompleteH4ProjectionFitSequence(
        example_id="example-a",
        family_id="family-a",
        residual_rows=torch.stack(
            (
                _row((0, 1.0), (320, 2.0)),
                _row((1, 1.0), (320, 3.0)),
                _row((2, 1.0), (321, 4.0)),
            )
        ),
    )
    sequence_b = CompleteH4ProjectionFitSequence(
        example_id="example-b",
        family_id="family-b",
        residual_rows=torch.stack(
            (
                _row((3, 1.0)),
                _row((4, 1.0), (320, 5.0), (321, 6.0)),
            )
        ),
    )
    traces = (
        CompleteH4TailProjectionTrace.from_fit_sequence(
            sequence_a,
            torch.tensor((True, False, False), dtype=torch.bool),
            source_pair_sha256="1" * 64,
            source_graph_core_mask_sha256="2" * 64,
        ),
        CompleteH4TailProjectionTrace.from_fit_sequence(
            sequence_b,
            (True, False),
            source_pair_sha256="3" * 64,
            source_graph_core_mask_sha256="4" * 64,
        ),
    )
    eigenvalues = tuple(float(320 - index) for index in range(320))
    basis = CompleteH4ProjectionBasis(
        width=_WIDTH,
        max_rank=320,
        basis_rows=torch.eye(_WIDTH, dtype=torch.float64)[:320],
        residual_eigenvalues=eigenvalues,
        residual_energy_fractions=tuple(1.0 / 640.0 for _ in range(320)),
        directional_residual_variance=eigenvalues,
        next_residual_eigenvalue=0.5,
        cutoff_spectral_gap=0.5,
        source_example_ids=("example-a", "example-b"),
        source_family_ids=("family-a", "family-b"),
        source_sequence_sha256s=(
            sequence_a.sequence_sha256,
            sequence_b.sequence_sha256,
        ),
        fit_weighting="unweighted",
    )
    return basis, traces


def test_trace_binds_rows_mask_and_upstream_receipts_without_exposing_values() -> None:
    _, traces = _fixture()
    trace = traces[0]
    assert trace.tail_row_count == 2
    assert trace.tail_rows_tensor().shape == (2, _WIDTH)
    metadata = trace.metadata()
    assert metadata["source_sequence_sha256"] == trace.source_sequence_sha256
    assert metadata["source_pair_sha256"] == "1" * 64
    assert metadata["graph_core_mask_sha256"] == trace.graph_core_mask_sha256
    assert "graph_core_mask" not in metadata
    assert "residual_rows" not in metadata


def test_fit_preserves_u192_and_exactly_reconstructs_structural_tail() -> None:
    global_basis, traces = _fixture()
    fit = fit_complete_h4_tail_informed_projection(traces, global_basis)

    assert fit.tail_rank == 2
    assert fit.max_rank == 320
    assert fit.global_fit_basis_artifact_sha256 == global_basis.artifact_sha256
    assert fit.validate_integrity() is True
    expected_u192 = global_basis.basis_tensor()[:192]
    actual_u192 = fit.basis_tensor(192)
    assert torch.equal(actual_u192, expected_u192)
    assert actual_u192.numpy().tobytes() == expected_u192.numpy().tobytes()

    treatment = fit.basis_tensor(320)
    assert torch.allclose(
        treatment @ treatment.T,
        torch.eye(320, dtype=torch.float64),
        atol=2e-10,
        rtol=2e-10,
    )
    tail_rows = torch.cat(tuple(trace.tail_rows_tensor() for trace in traces))
    tail_boundary = fit.basis_tensor(192 + fit.tail_rank)
    reconstruction = (tail_rows @ tail_boundary.T) @ tail_boundary
    max_error = float(torch.max(torch.abs(tail_rows - reconstruction)).item())
    assert max_error == fit.tail_reconstruction_max_abs_error
    assert max_error <= fit.tail_reconstruction_absolute_tolerance
    assert fit.metadata()["tail_reconstruction_exact_float64"] is True


def test_fit_is_deterministic_and_prefixes_are_exactly_nested() -> None:
    global_basis, traces = _fixture()
    fit = fit_complete_h4_tail_informed_projection(reversed(traces), global_basis)
    repeated = fit_complete_h4_tail_informed_projection(traces, global_basis)
    assert fit.artifact_sha256 == repeated.artifact_sha256
    assert torch.equal(fit.basis_tensor(64), fit.basis_tensor(320)[:64])
    assert torch.equal(fit.basis_tensor(194), fit.basis_tensor(320)[:194])
    assert fit.prefix_artifact_sha256(192) != fit.prefix_artifact_sha256(194)
    assert fit.metadata()["ordering"] == TAIL_INFORMED_PROJECTION_ORDERING


def test_lineage_is_hash_bound_and_rejects_invalid_receipts() -> None:
    global_basis, traces = _fixture()
    fit = fit_complete_h4_tail_informed_projection(traces, global_basis)
    lineage = fit.lineage(194, "a" * 64)
    assert lineage["fit_artifact_sha256"] == fit.artifact_sha256
    assert lineage["prefix_artifact_sha256"] == fit.prefix_artifact_sha256(194)
    assert lineage["execution_basis_artifact_sha256"] == "a" * 64
    assert len(lineage["lineage_sha256"]) == 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        fit.lineage(194, "not-a-receipt")


def test_claimed_artifact_and_matrix_tampering_are_rejected() -> None:
    global_basis, traces = _fixture()
    fit = fit_complete_h4_tail_informed_projection(traces, global_basis)
    with pytest.raises(ValueError, match="artifact receipt differs"):
        replace(fit, source_row_count=fit.source_row_count + 1)

    tampered = fit.basis_tensor(320)
    tampered[0, 0] = -1.0
    with pytest.raises(ValueError, match="signs are not canonical"):
        replace(fit, treatment_basis_rows=tampered)


def test_no_structural_tail_is_rejected() -> None:
    global_basis, traces = _fixture()
    no_tail = tuple(
        CompleteH4TailProjectionTrace(
            example_id=trace.example_id,
            family_id=trace.family_id,
            residual_rows=trace.residual_rows,
            graph_core_mask=(True,) * trace.row_count,
            source_sequence_sha256=trace.source_sequence_sha256,
            source_pair_sha256=trace.source_pair_sha256,
            source_graph_core_mask_sha256=trace.source_graph_core_mask_sha256,
        )
        for trace in traces
    )
    with pytest.raises(ValueError, match="requires structural-tail rows"):
        fit_complete_h4_tail_informed_projection(no_tail, global_basis)


def test_zero_tail_residual_after_u192_is_rejected() -> None:
    global_basis, traces = _fixture()
    zero_tail = tuple(
        CompleteH4TailProjectionTrace(
            example_id=trace.example_id,
            family_id=trace.family_id,
            residual_rows=torch.stack(
                tuple(_row((index, 1.0)) for index in range(trace.row_count))
            ),
            graph_core_mask=trace.graph_core_mask,
            source_sequence_sha256=trace.source_sequence_sha256,
            source_pair_sha256=trace.source_pair_sha256,
            source_graph_core_mask_sha256=trace.source_graph_core_mask_sha256,
        )
        for trace in traces
    )
    with pytest.raises(ValueError, match="zero energy"):
        fit_complete_h4_tail_informed_projection(zero_tail, global_basis)


def test_closed_basis_contract_and_prefix_bounds() -> None:
    global_basis, traces = _fixture()
    tilted = replace(global_basis, fit_weighting="fisher_alignment_tilted")
    with pytest.raises(ValueError, match="unweighted global basis"):
        fit_complete_h4_tail_informed_projection(traces, tilted)
    fit = fit_complete_h4_tail_informed_projection(traces, global_basis)
    with pytest.raises(ValueError, match="positive integer"):
        fit.basis_tensor(0)
    with pytest.raises(ValueError, match="exceeds max_rank"):
        fit.basis_tensor(321)
