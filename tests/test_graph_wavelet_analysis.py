from __future__ import annotations

import json

import pytest
import torch

from fisher_graph.graph_wavelet_analysis import (
    DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES,
    GRAPH_COMPRESSION_METHOD_ORDER,
    FitOnlyGraphWaveletOMPSubspace,
    GraphWaveletCoefficients,
    SpectralGraphWaveletFrame,
    analyze_graph_wavelet_compression,
    build_spectral_graph_wavelet_frame,
    fit_graph_wavelet_omp_subspace,
    fit_wavelet_group_order,
    matched_graph_signal_coefficients,
    reconstruct_frozen_wavelet_groups,
    reconstruct_matched_graph_signal,
    wavelet_group_scores,
)


def _laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    return torch.diag(adjacency.sum(dim=1)) - adjacency


def _path_laplacian(node_count: int = 8) -> torch.Tensor:
    adjacency = torch.zeros(
        (node_count, node_count),
        dtype=torch.float64,
    )
    for node in range(node_count - 1):
        adjacency[node, node + 1] = 1.0
        adjacency[node + 1, node] = 1.0
    return _laplacian(adjacency)


def _community_laplacian() -> torch.Tensor:
    adjacency = torch.zeros((8, 8), dtype=torch.float64)
    for group in (range(4), range(4, 8)):
        for left in group:
            for right in group:
                if left < right:
                    adjacency[left, right] = 1.0
                    adjacency[right, left] = 1.0
    adjacency[3, 4] = 0.03
    adjacency[4, 3] = 0.03
    return _laplacian(adjacency)


def test_path_frame_is_deterministic_parseval_and_roundtrips_state() -> None:
    laplacian = _path_laplacian()
    first = build_spectral_graph_wavelet_frame(laplacian)
    second = build_spectral_graph_wavelet_frame(laplacian)

    assert first.diffusion_scales == DEFAULT_GRAPH_WAVELET_DIFFUSION_SCALES
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.filter_names == (
        "scaling",
        "bandpass_00",
        "bandpass_01",
        "bandpass_02",
        "bandpass_03",
    )
    assert first.tight_partition_maximum_error < 1.0e-12
    assert first.tight_operator_maximum_error < 1.0e-12
    assert first.eigensystem_relative_residual < 1.0e-12
    torch.testing.assert_close(
        first.spectral_kernels.square().sum(dim=0),
        torch.ones(8, dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        torch.einsum(
            "fij,fjk->ik",
            first.filter_matrices,
            first.filter_matrices,
        ),
        torch.eye(8, dtype=torch.float64),
        atol=2.0e-12,
        rtol=2.0e-12,
    )

    signal = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 7.0
    coefficients = first.analyze(signal)
    reconstructed = first.synthesize(coefficients)
    assert coefficients.values.shape == (5, 8, 3)
    torch.testing.assert_close(
        reconstructed,
        signal,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert float(coefficients.values.square().sum()) == pytest.approx(
        float(signal.square().sum()),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    for filter_index in range(first.filter_count):
        for center in (0, 3, 7):
            torch.testing.assert_close(
                first.atom(filter_index, center),
                first.filter_matrices[filter_index, :, center],
            )

    restored = SpectralGraphWaveletFrame.from_state_dict(first.state_dict())
    assert restored.artifact_sha256 == first.artifact_sha256
    torch.testing.assert_close(restored.filter_matrices, first.filter_matrices)


def test_frame_replay_rejects_semantically_mismatched_graph_receipts() -> None:
    frame = build_spectral_graph_wavelet_frame(_path_laplacian())
    common = {
        "eigenvalues": frame.eigenvalues,
        "eigenvectors": frame.eigenvectors,
        "spectral_kernels": frame.spectral_kernels,
        "filter_matrices": frame.filter_matrices,
        "filter_names": frame.filter_names,
        "diffusion_scales": frame.diffusion_scales,
        "eigenvalue_tolerance": frame.eigenvalue_tolerance,
        "eigensystem_relative_residual": (
            frame.eigensystem_relative_residual
        ),
        "tight_partition_maximum_error": (
            frame.tight_partition_maximum_error
        ),
        "tight_operator_maximum_error": (
            frame.tight_operator_maximum_error
        ),
    }

    with pytest.raises(ValueError, match="laplacian and graph eigensystem"):
        SpectralGraphWaveletFrame(
            laplacian=frame.laplacian
            + torch.eye(frame.node_count, dtype=torch.float64),
            **common,
        )

    false_residual = dict(common)
    false_residual["eigensystem_relative_residual"] = (
        frame.eigensystem_relative_residual + 0.25
    )
    with pytest.raises(ValueError, match="declared eigensystem residual"):
        SpectralGraphWaveletFrame(
            laplacian=frame.laplacian,
            **false_residual,
        )

    mutated_kernels = frame.spectral_kernels.roll(shifts=1, dims=0)
    mutated_filters = frame.filter_matrices.roll(shifts=1, dims=0)
    mutated_frame = dict(common)
    mutated_frame["spectral_kernels"] = mutated_kernels
    mutated_frame["filter_matrices"] = mutated_filters
    with pytest.raises(ValueError, match="deterministic diffusion kernels"):
        SpectralGraphWaveletFrame(
            laplacian=frame.laplacian,
            **mutated_frame,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tight_partition_maximum_error", float("nan")),
        ("tight_partition_maximum_error", float("inf")),
        ("tight_operator_maximum_error", float("nan")),
        ("tight_operator_maximum_error", float("inf")),
    ),
)
def test_frame_replay_rejects_nonfinite_tight_residual_receipts(
    field: str,
    value: float,
) -> None:
    frame = build_spectral_graph_wavelet_frame(_path_laplacian())
    state = {
        "laplacian": frame.laplacian,
        "eigenvalues": frame.eigenvalues,
        "eigenvectors": frame.eigenvectors,
        "spectral_kernels": frame.spectral_kernels,
        "filter_matrices": frame.filter_matrices,
        "filter_names": frame.filter_names,
        "diffusion_scales": frame.diffusion_scales,
        "eigenvalue_tolerance": frame.eigenvalue_tolerance,
        "eigensystem_relative_residual": (
            frame.eigensystem_relative_residual
        ),
        "tight_partition_maximum_error": (
            frame.tight_partition_maximum_error
        ),
        "tight_operator_maximum_error": (
            frame.tight_operator_maximum_error
        ),
    }
    state[field] = value

    with pytest.raises(ValueError, match="declared tight-frame residuals"):
        SpectralGraphWaveletFrame(**state)


def test_community_scaling_atoms_localize_and_metrics_expose_real_support() -> None:
    first = build_spectral_graph_wavelet_frame(_community_laplacian())
    second = build_spectral_graph_wavelet_frame(_community_laplacian())
    assert first.artifact_sha256 == second.artifact_sha256

    scaling_at_zero = first.atom(0, 0)
    energy = scaling_at_zero.square()
    within_community = float(energy[:4].sum() / energy.sum())
    assert within_community > 0.99

    rows = first.localization_metrics()
    assert len(rows) == first.filter_count * first.node_count
    selected = next(
        row
        for row in rows
        if row["filter_name"] == "scaling" and row["center_node"] == 0
    )
    assert 1 <= selected["energy_support_90_node_count"] <= 4
    assert 1 <= selected["energy_support_95_node_count"] <= 4
    assert selected["effective_node_support"] <= 4.1
    assert selected["center_energy_fraction"] > 0.20
    assert selected["normalized_graph_total_variation"] >= 0.0
    assert selected["unreachable_energy_fraction"] == pytest.approx(0.0)

    summaries = first.scale_localization_summary()
    assert tuple(row["filter_name"] for row in summaries) == first.filter_names
    assert summaries[0]["mean_spectral_center"] < min(
        row["mean_spectral_center"] for row in summaries[1:]
    )
    assert all(
        row["mean_energy_support_90_node_count"]
        <= row["mean_energy_support_95_node_count"]
        for row in summaries
    )


def test_fit_only_group_order_masks_whole_multifeature_vectors() -> None:
    frame = build_spectral_graph_wavelet_frame(_path_laplacian())
    fit_first = torch.stack(
        (
            torch.linspace(0.0, 1.0, 8, dtype=torch.float64),
            torch.linspace(1.0, -1.0, 8, dtype=torch.float64),
            torch.eye(8, dtype=torch.float64)[:, 2],
        ),
        dim=1,
    )
    fit_second = torch.roll(fit_first, shifts=1, dims=0)
    frozen = fit_wavelet_group_order(
        frame,
        (fit_first, fit_second),
    )
    replayed = fit_wavelet_group_order(
        frame,
        (fit_first, fit_second),
    )
    assert frozen.artifact_sha256 == replayed.artifact_sha256
    assert frozen.heldout_signal_used_for_order is False
    assert frozen.mask(6).shape == (frame.filter_count, frame.node_count)
    assert int(frozen.mask(6).sum()) == 6

    heldout = torch.randn(
        (8, 4, 2),
        generator=torch.Generator().manual_seed(99),
        dtype=torch.float64,
    )
    coefficients = frame.analyze(heldout)
    scores = wavelet_group_scores(coefficients)
    assert scores.shape == (frame.filter_count, frame.node_count)
    torch.testing.assert_close(
        scores,
        coefficients.values.square().sum(dim=(2, 3)),
    )
    reconstruction = reconstruct_frozen_wavelet_groups(
        frame,
        coefficients,
        frozen,
        6,
    )

    mask = frozen.mask(6).unsqueeze(-1).unsqueeze(-1)
    expected = frame.synthesize(
        GraphWaveletCoefficients(
            frame_artifact_sha256=frame.artifact_sha256,
            values=torch.where(
                mask,
                coefficients.values,
                torch.zeros_like(coefficients.values),
            ),
        )
    )
    torch.testing.assert_close(reconstruction, expected)
    assert reconstruction.shape == heldout.shape
    with pytest.raises(ValueError, match="exceeds"):
        frozen.mask(frozen.group_count + 1)


def test_equal_budget_report_has_matched_controls_and_oracle_warning() -> None:
    frame = build_spectral_graph_wavelet_frame(_path_laplacian())
    signal = torch.tensor(
        (0.0, 1.0, 1.5, -0.5, 2.0, 0.25, -1.0, 0.75),
        dtype=torch.float64,
    )
    budgets = (0, 1, 2, 4, 8)
    first = analyze_graph_wavelet_compression(
        frame,
        signal,
        budgets=budgets,
        random_seed=71,
    )
    second = analyze_graph_wavelet_compression(
        frame,
        signal,
        budgets=budgets,
        random_seed=71,
    )
    changed_seed = analyze_graph_wavelet_compression(
        frame,
        signal,
        budgets=budgets,
        random_seed=72,
    )

    assert first.report_sha256 == second.report_sha256
    assert first.report_sha256 != changed_seed.report_sha256
    assert tuple(first.curves) == GRAPH_COMPRESSION_METHOD_ORDER
    assert first.budgets == budgets
    assert first.parseval_energy_relative_error < 1.0e-12
    assert first.full_frame_reconstruction_relative_error < 1.0e-12
    assert first.heldout_evidence_claim is False
    assert "oracle" in first.selection_semantics
    assert first.raw_signal_serialized is False
    for method in GRAPH_COMPRESSION_METHOD_ORDER:
        points = first.curves[method]
        assert tuple(point.budget for point in points) == budgets
        retained_energy = tuple(
            point.retained_analysis_energy_fraction for point in points
        )
        assert retained_energy == tuple(sorted(retained_energy))
        if method == "graph_wavelet_tight_frame":
            assert points[-1].available_coefficient_count == 40
        else:
            assert points[-1].available_coefficient_count == 8
            assert points[-1].relative_l2_error < 1.0e-12
    for method in GRAPH_COMPRESSION_METHOD_ORDER[:-1]:
        assert (
            first.curves[method][-1].reconstruction_sha256
            == changed_seed.curves[method][-1].reconstruction_sha256
        )
    assert (
        first.curves["random_orthonormal"][1].reconstruction_sha256
        != changed_seed.curves["random_orthonormal"][1].reconstruction_sha256
    )

    payload = first.to_dict()
    assert "signal" not in payload
    assert payload["raw_signal_serialized"] is False
    json.dumps(payload, allow_nan=False)
    first.validate_integrity()


def test_controls_reconstruct_and_invalid_graphs_or_protocols_fail() -> None:
    frame = build_spectral_graph_wavelet_frame(_path_laplacian())
    signal = torch.arange(8, dtype=torch.float64)
    controls = matched_graph_signal_coefficients(
        frame,
        signal,
        random_seed=5,
    )
    for method in ("graph_fourier", "native_nodes", "random_orthonormal"):
        torch.testing.assert_close(
            reconstruct_matched_graph_signal(
                frame,
                controls,
                method,
                frame.node_count,
            ),
            signal,
            atol=2.0e-12,
            rtol=2.0e-12,
        )

    asymmetric = _path_laplacian()
    asymmetric[0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetric"):
        build_spectral_graph_wavelet_frame(asymmetric)
    with pytest.raises(ValueError, match="positive semidefinite"):
        build_spectral_graph_wavelet_frame(
            -torch.eye(4, dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_spectral_graph_wavelet_frame(
            _path_laplacian(),
            diffusion_scales=(1.0, 1.0, 2.0),
        )
    with pytest.raises(ValueError, match="node_count"):
        analyze_graph_wavelet_compression(
            frame,
            signal,
            budgets=(0, 9),
        )


def test_simultaneous_group_omp_is_deterministic_nested_and_localized() -> None:
    frame = build_spectral_graph_wavelet_frame(_path_laplacian())
    generator = torch.Generator().manual_seed(812)
    fit_signals = (
        torch.randn(
            (8, 3),
            generator=generator,
            dtype=torch.float64,
        ),
        torch.randn(
            (8, 2, 2),
            generator=generator,
            dtype=torch.float64,
        ),
    )
    first = fit_graph_wavelet_omp_subspace(frame, fit_signals)
    second = fit_graph_wavelet_omp_subspace(frame, fit_signals)

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.max_rank == frame.node_count
    assert len(set(first.selected_flat_atom_indices)) == frame.node_count
    assert first.heldout_signal_used_for_fit is False
    assert "fit_only" in first.selection_semantics
    torch.testing.assert_close(
        first.orthonormal_basis.T @ first.orthonormal_basis,
        torch.eye(8, dtype=torch.float64),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    for column in range(first.max_rank):
        pivot = int(torch.argmax(first.orthonormal_basis[:, column].abs()))
        assert float(first.orthonormal_basis[pivot, column]) >= 0.0
    residuals = first.fit_relative_residual_by_rank
    assert residuals.shape == (9,)
    assert residuals[0] == 1.0
    assert bool((residuals[1:] <= residuals[:-1] + 1.0e-14).all())
    assert float(residuals[-1]) < 1.0e-12
    assert bool(
        (
            first.selected_qr_novelty
            > first.dependency_tolerance
        ).all()
    )
    assert bool(torch.isfinite(first.selected_qr_novelty).all())
    assert bool(
        torch.isfinite(
            first.raw_selected_dictionary_condition_by_rank
        ).all()
    )
    assert float(
        first.raw_selected_dictionary_condition_by_rank.max()
    ) < 100.0

    raw_atoms = (
        frame.filter_matrices.permute(1, 0, 2)
        .reshape(frame.node_count, -1)
    )
    raw_norms = torch.linalg.vector_norm(raw_atoms, dim=0)
    nonzero = raw_norms > 1.0e-14
    normalized = torch.zeros_like(raw_atoms)
    normalized[:, nonzero] = (
        raw_atoms[:, nonzero] / raw_norms[nonzero].unsqueeze(0)
    )
    fit_matrix = torch.cat(
        tuple(signal.reshape(frame.node_count, -1) for signal in fit_signals),
        dim=1,
    )
    first_scores = (normalized.T @ fit_matrix).square().sum(dim=1)
    first_scores[~nonzero] = -1.0
    expected_first = min(
        index
        for index in range(first_scores.numel())
        if float(first_scores[index]) == float(first_scores.max())
    )
    assert first.selected_flat_atom_indices[0] == expected_first
    for rank in range(first.max_rank + 1):
        torch.testing.assert_close(
            first.basis(rank),
            first.orthonormal_basis[:, :rank],
        )

    assert len(first.selected_atom_metadata) == first.max_rank
    assert len(first.orthonormal_basis_locality) == first.max_rank
    for raw_row, q_row in zip(
        first.selected_atom_metadata,
        first.orthonormal_basis_locality,
    ):
        assert raw_row["raw_atom_norm"] > 0.0
        assert 1.0 <= raw_row["effective_node_support"] <= 8.0
        assert (
            1
            <= raw_row["energy_support_90_node_count"]
            <= raw_row["energy_support_95_node_count"]
            <= 8
        )
        assert 1.0 <= q_row["effective_node_support"] <= 8.0
        assert "hop" not in " ".join(q_row)
    first.validate_against_frame(frame)
    first.validate_integrity()
    restored = FitOnlyGraphWaveletOMPSubspace.from_state_dict(
        first.state_dict()
    )
    assert restored.artifact_sha256 == first.artifact_sha256
    assert (
        restored.selected_flat_atom_indices
        == first.selected_flat_atom_indices
    )
    torch.testing.assert_close(
        restored.orthonormal_basis,
        first.orthonormal_basis,
    )
    restored.validate_against_frame(frame)


def test_full_rank_omp_reconstructs_heldout_without_affecting_fit_artifact() -> None:
    frame = build_spectral_graph_wavelet_frame(_community_laplacian())
    fit = (
        torch.stack(
            (
                torch.tensor(
                    (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0),
                    dtype=torch.float64,
                ),
                torch.arange(8, dtype=torch.float64) / 8.0,
            ),
            dim=1,
        ),
    )
    fitted = fit_graph_wavelet_omp_subspace(frame, fit)
    heldout = torch.randn(
        (8, 4, 2),
        generator=torch.Generator().manual_seed(17),
        dtype=torch.float64,
    )
    changed_heldout = heldout.clone()
    changed_heldout[0].mul_(1000.0).add_(71.0)

    artifact_before = fitted.artifact_sha256
    full = fitted.project(heldout, fitted.max_rank)
    changed_full = fitted.project(changed_heldout, fitted.max_rank)
    torch.testing.assert_close(
        full,
        heldout,
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    torch.testing.assert_close(
        changed_full,
        changed_heldout,
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    assert not torch.equal(full, changed_full)
    assert fitted.artifact_sha256 == artifact_before
    assert (
        fit_graph_wavelet_omp_subspace(frame, fit).artifact_sha256
        == artifact_before
    )

    relative_errors = []
    denominator = float(torch.linalg.vector_norm(heldout))
    for rank in range(fitted.max_rank + 1):
        projected = fitted.project(heldout, rank)
        relative_errors.append(
            float(torch.linalg.vector_norm(projected - heldout))
            / denominator
        )
    assert all(
        right <= left + 1.0e-12
        for left, right in zip(relative_errors, relative_errors[1:])
    )
    assert relative_errors[-1] < 1.0e-12


def test_omp_flat_ties_dependencies_and_invalid_inputs_are_explicit() -> None:
    zero_frame = build_spectral_graph_wavelet_frame(
        torch.zeros((4, 4), dtype=torch.float64)
    )
    tied = fit_graph_wavelet_omp_subspace(
        zero_frame,
        (torch.ones(4, dtype=torch.float64),),
    )
    assert tied.selected_flat_atom_indices == (0, 1, 2, 3)
    torch.testing.assert_close(
        tied.selected_raw_atom_norms,
        torch.ones(4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        tied.selected_qr_novelty,
        torch.ones(4, dtype=torch.float64),
    )
    torch.testing.assert_close(
        tied.raw_selected_dictionary_condition_by_rank,
        torch.ones(4, dtype=torch.float64),
    )
    assert all(
        row["filter_name"] == "scaling"
        for row in tied.selected_atom_metadata
    )

    with pytest.raises(ValueError, match="nonempty"):
        fit_graph_wavelet_omp_subspace(zero_frame, ())
    with pytest.raises(ValueError, match="nonzero energy"):
        fit_graph_wavelet_omp_subspace(
            zero_frame,
            (torch.zeros(4, dtype=torch.float64),),
        )
    with pytest.raises(ValueError, match="node axis"):
        fit_graph_wavelet_omp_subspace(
            zero_frame,
            (torch.ones(5, dtype=torch.float64),),
        )
    with pytest.raises(ValueError, match="positive integer"):
        fit_graph_wavelet_omp_subspace(
            zero_frame,
            (torch.ones(4, dtype=torch.float64),),
            max_rank=0,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_graph_wavelet_omp_subspace(
            zero_frame,
            (torch.ones(4, dtype=torch.float64),),
            max_rank=5,
        )
    with pytest.raises(ValueError, match="less than one"):
        fit_graph_wavelet_omp_subspace(
            zero_frame,
            (torch.ones(4, dtype=torch.float64),),
            dependency_tolerance=1.0,
        )
    with pytest.raises(ValueError, match=r"\[0, max_rank\]"):
        tied.basis(5)
    with pytest.raises(ValueError, match="node axis"):
        tied.project(torch.ones(5, dtype=torch.float64), 2)

    other_frame = build_spectral_graph_wavelet_frame(
        torch.zeros((4, 4), dtype=torch.float64),
        diffusion_scales=(0.25, 0.5, 1.0),
    )
    with pytest.raises(ValueError, match="another frame"):
        tied.validate_against_frame(other_frame)

    drifted = fit_graph_wavelet_omp_subspace(
        zero_frame,
        (torch.ones(4, dtype=torch.float64),),
    )
    drifted.orthonormal_basis[0, 0] += 0.25
    with pytest.raises(ValueError, match="drifted"):
        drifted.validate_integrity()
