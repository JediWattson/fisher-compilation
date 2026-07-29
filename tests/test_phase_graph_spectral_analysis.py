from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.modal_spectral_mapping import analyze_modal_spectral_mapping
from fisher_graph.phase_graph_spectral_analysis import (
    PhaseGraphSpectralAnalysis,
    analyze_phase_graph_spectral_response,
)


def _mapping(
    function,
    *,
    source_modes: int,
    max_lag: int = 0,
    fft_length: int = 8,
):
    return analyze_modal_spectral_mapping(
        function,
        baseline_modes=torch.zeros(
            1,
            8,
            source_modes,
            dtype=torch.float64,
        ),
        logical_positions=torch.arange(8),
        valid_mask=torch.ones(8, dtype=torch.bool),
        source_mode_indices=tuple(range(source_modes)),
        impulse_logical_positions=(1, 3),
        max_lag=max_lag,
        fft_length=fft_length,
        finite_impulse_amplitudes=torch.ones(
            source_modes,
            dtype=torch.float64,
        ),
        symmetric_amplitude_sets={
            "local": torch.full(
                (source_modes,),
                0.1,
                dtype=torch.float64,
            ),
        },
        similarity_threshold=0.9,
    )


def test_exact_sign_reversal_is_an_explicit_negative_phase_edge() -> None:
    mapping = _mapping(
        lambda source: source[..., :1] - source[..., 1:2],
        source_modes=2,
    )

    result = analyze_phase_graph_spectral_response(
        mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )

    assert mapping.finite.pairwise_spectral_similarity[0, 1] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result.complex_coherency_real[0, 1] == pytest.approx(
        -1.0,
        abs=1e-12,
    )
    assert result.complex_coherency_imag[0, 1] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert result.connection_adjacency_real[0, 1] == pytest.approx(-1.0)
    assert result.signed_adjacency[0, 1] == pytest.approx(-1.0)
    assert result.opposed_selected_edge_count == 1
    assert result.connection_rank_90 == 1
    assert result.magnitude_rank_90 == 2
    assert result.maximum_legacy_phase_gap == pytest.approx(1.0)
    assert result.strongest_opposed_pairs[0]["left_mode"] == 0
    assert result.strongest_opposed_pairs[0]["right_mode"] == 1
    assert result.strongest_opposed_pairs[0][
        "phase_alignment"
    ] == pytest.approx(-1.0)


def test_directional_quadrature_retains_lead_lag_orientation() -> None:
    def lagged(source: torch.Tensor) -> torch.Tensor:
        first = source[..., :1]
        second = source[..., 1:2]
        delayed = torch.cat(
            (torch.zeros_like(second[:, :1]), second[:, :-1]),
            dim=1,
        )
        return first + delayed

    mapping = _mapping(lagged, source_modes=2, max_lag=1)
    result = analyze_phase_graph_spectral_response(
        mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )

    assert result.directional_quadrature[0, 0] == 0.0
    assert result.directional_quadrature[1, 1] == 0.0
    assert result.directional_quadrature[0, 1] == pytest.approx(
        -float(result.directional_quadrature[1, 0]),
    )
    assert abs(float(result.directional_quadrature[0, 1])) > 0.01
    torch.testing.assert_close(
        result.connection_adjacency_imag,
        -result.connection_adjacency_imag.T,
    )


def test_graph_laplacians_are_psd_and_fourier_bases_are_orthonormal() -> None:
    def three_mode_map(source: torch.Tensor) -> torch.Tensor:
        first = source[..., :1]
        second = 0.75 * source[..., 1:2]
        third = -0.5 * source[..., 2:3]
        delayed = torch.cat(
            (torch.zeros_like(second[:, :1]), second[:, :-1]),
            dim=1,
        )
        return torch.cat(
            (
                first + delayed + third,
                first - delayed,
            ),
            dim=2,
        )

    mapping = _mapping(three_mode_map, source_modes=3, max_lag=1)
    first = analyze_phase_graph_spectral_response(
        mapping.symmetric_by_label["local"],
        neighbor_count=2,
        minimum_coherence=0.0,
    )
    second = analyze_phase_graph_spectral_response(
        mapping.symmetric_by_label["local"],
        neighbor_count=2,
        minimum_coherence=0.0,
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert float(first.connection_eigenvalues.min()) >= 0.0
    assert float(first.signed_eigenvalues.min()) >= 0.0
    connection_vectors = torch.complex(
        first.connection_eigenvectors_real,
        first.connection_eigenvectors_imag,
    )
    torch.testing.assert_close(
        connection_vectors.mH @ connection_vectors,
        torch.eye(3, dtype=torch.complex128),
        atol=2e-8,
        rtol=2e-8,
    )
    torch.testing.assert_close(
        first.signed_eigenvectors.T @ first.signed_eigenvectors,
        torch.eye(3, dtype=torch.float64),
        atol=2e-8,
        rtol=2e-8,
    )
    assert float(first.connection_graph_fourier_energy.sum()) == pytest.approx(
        1.0,
    )
    assert float(first.signed_graph_fourier_energy.sum()) == pytest.approx(
        1.0,
    )
    assert float(first.magnitude_graph_fourier_energy.sum()) == pytest.approx(
        1.0,
    )
    assert (
        first.connection_rank_90
        <= first.connection_rank_95
        <= first.connection_rank_99
    )
    assert first.signed_rank_90 <= first.signed_rank_95 <= first.signed_rank_99
    assert first.connection_low8_energy_fraction == pytest.approx(1.0)
    assert first.connection_low16_energy_fraction == pytest.approx(1.0)
    assert first.signed_low8_energy_fraction == pytest.approx(1.0)
    assert first.signed_low16_energy_fraction == pytest.approx(1.0)
    assert first.magnitude_low8_energy_fraction == pytest.approx(1.0)
    assert first.magnitude_low16_energy_fraction == pytest.approx(1.0)


def test_top_k_view_does_not_change_dense_graph_fourier_basis() -> None:
    mapping = _mapping(
        lambda source: (
            source[..., :1]
            + 0.5 * source[..., 1:2]
            - 0.25 * source[..., 2:3]
        ),
        source_modes=3,
    )

    inclusive = analyze_phase_graph_spectral_response(
        mapping.finite,
        neighbor_count=2,
        minimum_coherence=0.0,
    )
    selective = analyze_phase_graph_spectral_response(
        mapping.finite,
        neighbor_count=1,
        minimum_coherence=1.0,
    )

    assert inclusive.selected_edge_count > selective.selected_edge_count
    torch.testing.assert_close(
        inclusive.connection_laplacian_real,
        selective.connection_laplacian_real,
    )
    torch.testing.assert_close(
        inclusive.connection_laplacian_imag,
        selective.connection_laplacian_imag,
    )
    torch.testing.assert_close(
        inclusive.connection_eigenvalues,
        selective.connection_eigenvalues,
    )
    torch.testing.assert_close(
        inclusive.connection_graph_fourier_energy,
        selective.connection_graph_fourier_energy,
    )


def test_phase_blind_control_is_invariant_to_pure_delay_phase() -> None:
    aligned_mapping = _mapping(
        lambda source: source[..., :1] + source[..., 1:2],
        source_modes=2,
        max_lag=1,
    )

    def delayed_map(source: torch.Tensor) -> torch.Tensor:
        second = source[..., 1:2]
        delayed = torch.cat(
            (torch.zeros_like(second[:, :1]), second[:, :-1]),
            dim=1,
        )
        return source[..., :1] + delayed

    delayed_mapping = _mapping(
        delayed_map,
        source_modes=2,
        max_lag=1,
    )
    aligned = analyze_phase_graph_spectral_response(
        aligned_mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )
    delayed = analyze_phase_graph_spectral_response(
        delayed_mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )

    torch.testing.assert_close(
        aligned.phase_blind_magnitude_similarity,
        delayed.phase_blind_magnitude_similarity,
    )
    torch.testing.assert_close(
        aligned.magnitude_laplacian,
        delayed.magnitude_laplacian,
    )
    assert not torch.allclose(
        aligned.connection_laplacian_real,
        delayed.connection_laplacian_real,
    ) or not torch.allclose(
        aligned.connection_laplacian_imag,
        delayed.connection_laplacian_imag,
    )


def test_real_alignment_is_parseval_consistent_but_phase_binds_fft_length() -> None:
    def delayed_map(source: torch.Tensor) -> torch.Tensor:
        second = source[..., 1:2]
        delayed = torch.cat(
            (torch.zeros_like(second[:, :1]), second[:, :-1]),
            dim=1,
        )
        return source[..., :1] + delayed - 0.25 * source[..., 2:3]

    short_mapping = _mapping(
        delayed_map,
        source_modes=3,
        max_lag=1,
        fft_length=8,
    )
    padded_mapping = _mapping(
        delayed_map,
        source_modes=3,
        max_lag=1,
        fft_length=16,
    )
    short = analyze_phase_graph_spectral_response(short_mapping.finite)
    padded = analyze_phase_graph_spectral_response(padded_mapping.finite)

    torch.testing.assert_close(
        short.complex_coherency_real,
        padded.complex_coherency_real,
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        short.phase_blind_magnitude_similarity,
        padded.phase_blind_magnitude_similarity,
        atol=1e-10,
        rtol=1e-10,
    )
    assert not torch.allclose(
        short.complex_coherency_imag,
        padded.complex_coherency_imag,
        atol=1e-10,
        rtol=1e-10,
    )


def test_tied_eigenvalue_blocks_are_not_split_by_energy_ranks() -> None:
    increasing = torch.arange(1, 11, dtype=torch.float64)
    decreasing = torch.flip(increasing, dims=(0,))
    first_mapping = _mapping(
        lambda source: source * increasing,
        source_modes=10,
    )
    second_mapping = _mapping(
        lambda source: source * decreasing,
        source_modes=10,
    )
    first = analyze_phase_graph_spectral_response(
        first_mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )
    second = analyze_phase_graph_spectral_response(
        second_mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )

    torch.testing.assert_close(
        first.connection_eigenvalues,
        torch.zeros(10, dtype=torch.float64),
    )
    assert first.connection_rank_90 == second.connection_rank_90 == 10
    assert first.connection_rank_95 == second.connection_rank_95 == 10
    assert first.connection_low8_energy_fraction == pytest.approx(1.0)
    assert second.connection_low8_energy_fraction == pytest.approx(1.0)


def test_relative_support_floor_rejects_negligible_gain_rows() -> None:
    mapping = _mapping(
        lambda source: source[..., :1] + 1e-15 * source[..., 1:2],
        source_modes=2,
    )

    result = analyze_phase_graph_spectral_response(mapping.finite)

    assert result.active_mode_count == 1
    assert result.source_response_norms[1] / result.source_response_norms[
        0
    ] == pytest.approx(1e-15)
    assert result.complex_coherency_real[0, 1] == 0.0
    assert result.selected_edge_count == 0


def test_zero_response_has_no_edges_or_graph_fourier_energy() -> None:
    mapping = _mapping(
        lambda source: torch.zeros_like(source[..., :1]),
        source_modes=2,
    )

    result = analyze_phase_graph_spectral_response(
        mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )

    assert result.active_mode_count == 0
    assert result.selected_edge_count == 0
    assert result.connection_rank_99 == 0
    assert result.signed_rank_99 == 0
    assert float(result.connection_graph_fourier_energy.sum()) == 0.0
    assert float(result.signed_graph_fourier_energy.sum()) == 0.0


def test_strict_roundtrip_and_tamper_rejection() -> None:
    mapping = _mapping(
        lambda source: source[..., :1] + 0.5 * source[..., 1:2],
        source_modes=2,
    )
    result = analyze_phase_graph_spectral_response(
        mapping.finite,
        neighbor_count=1,
        minimum_coherence=0.0,
    )

    restored = PhaseGraphSpectralAnalysis.from_state_dict(
        result.state_dict()
    )
    assert restored.artifact_sha256 == result.artifact_sha256
    torch.testing.assert_close(
        restored.complex_coherency_real,
        result.complex_coherency_real,
    )
    assert result.strongest_opposed_pairs == ()

    unknown = result.state_dict()
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="fields differ"):
        PhaseGraphSpectralAnalysis.from_state_dict(unknown)

    tampered = copy.deepcopy(result.state_dict())
    tampered["directional_quadrature"][0, 1] += 0.25
    with pytest.raises(
        ValueError,
        match="symmetry|hash mismatch",
    ):
        PhaseGraphSpectralAnalysis.from_state_dict(tampered)

    semantic_tamper = result.state_dict()
    semantic_tamper["phase_semantics"] = "arbitrary"
    with pytest.raises(ValueError, match="semantics differ"):
        PhaseGraphSpectralAnalysis.from_state_dict(semantic_tamper)

    blank_receipt = result.state_dict()
    blank_receipt["neighbor_count"] += 1
    blank_receipt["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PhaseGraphSpectralAnalysis.from_state_dict(blank_receipt)
