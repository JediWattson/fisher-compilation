from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.modal_spectral_mapping import (
    ModalSpectralMapping,
    analyze_modal_spectral_mapping,
    connected_components_from_spectral_similarity,
)


def _logical_causal_function(
    kernel: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
):
    valid_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    by_position = {
        int(positions[index]): int(index) for index in valid_indices
    }

    def function(source: torch.Tensor) -> torch.Tensor:
        rows = []
        for target_index in range(source.shape[1]):
            zero = source[:, target_index, :] @ torch.zeros_like(kernel[0])
            if not bool(mask[target_index]):
                rows.append(zero)
                continue
            target_position = int(positions[target_index])
            value = zero
            for lag in range(kernel.shape[0]):
                source_index = by_position.get(target_position - lag)
                if source_index is not None:
                    value = value + source[:, source_index, :] @ kernel[lag]
            rows.append(value)
        return torch.stack(rows, dim=1)

    return function


def test_linear_impulses_fft_multiscale_and_accounting_are_exact() -> None:
    kernel = torch.tensor(
        [
            [[1.0, 0.0, -0.5], [0.0, 2.0, 0.25]],
            [[0.5, 0.1, 0.0], [-0.4, 0.0, 0.75]],
            [[-0.2, 0.3, 0.2], [0.6, -0.1, 0.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.arange(8)
    mask = torch.ones(8, dtype=torch.bool)
    artifact = analyze_modal_spectral_mapping(
        _logical_causal_function(kernel, positions, mask),
        baseline_modes=torch.zeros(1, 8, 2, dtype=torch.float64),
        logical_positions=positions,
        valid_mask=mask,
        source_mode_indices=(0, 1),
        impulse_logical_positions=(1, 3, 5),
        max_lag=2,
        fft_length=16,
        finite_impulse_amplitudes=torch.tensor(
            [2.0, 3.0],
            dtype=torch.float64,
        ),
        symmetric_amplitude_sets={
            "local_0p05_sigma": torch.tensor(
                [0.05, 0.1],
                dtype=torch.float64,
            ),
            "operating_1sigma": torch.tensor(
                [1.0, 2.0],
                dtype=torch.float64,
            ),
        },
        similarity_threshold=0.95,
    )

    expected = kernel.permute(1, 0, 2).unsqueeze(1).expand(-1, 3, -1, -1)
    torch.testing.assert_close(
        artifact.finite.impulse_responses,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )
    expected_spectrum = torch.fft.rfft(expected, n=16, dim=2)
    torch.testing.assert_close(
        artifact.finite.spectral_fingerprint_real,
        expected_spectrum.real,
    )
    torch.testing.assert_close(
        artifact.finite.spectral_fingerprint_imag,
        expected_spectrum.imag,
    )
    assert artifact.symmetric_labels == (
        "local_0p05_sigma",
        "operating_1sigma",
    )
    for response in artifact.symmetric_responses:
        torch.testing.assert_close(response.impulse_responses, expected)
        torch.testing.assert_close(
            response.even_residual_impulse_responses,
            torch.zeros_like(expected),
            atol=1e-12,
            rtol=0,
        )
        assert response.relative_even_residual < 1e-12
        assert response.minimum_origin_spectral_similarity > 1 - 1e-12
        assert response.energy_beyond_lag4_fraction == 0.0
        assert len(response.per_frequency_rank_90) == 9
    torch.testing.assert_close(
        artifact.scale_similarity(
            "local_0p05_sigma",
            "operating_1sigma",
        ),
        torch.ones(2, dtype=torch.float64),
    )
    assert artifact.function_evaluation_count == 31
    accounting = artifact.accounting()
    assert accounting["baseline_function_evaluations"] == 1
    assert accounting["finite_function_evaluations"] == 6
    assert accounting["symmetric_function_evaluations"] == 24
    assert accounting["mapping_function_macs"] is None
    assert accounting["runtime_speedup_claim"] is False
    assert artifact.finite.lag_observation_counts == (3, 3, 3)
    assert sum(
        (
            artifact.finite.dc_energy_fraction,
            artifact.finite.low_energy_fraction,
            artifact.finite.mid_energy_fraction,
            artifact.finite.high_energy_fraction,
        )
    ) == pytest.approx(1.0)


def test_symmetric_secants_retain_scale_dependent_even_residual() -> None:
    def nonlinear(source: torch.Tensor) -> torch.Tensor:
        return source + 0.5 * source.square()

    artifact = analyze_modal_spectral_mapping(
        nonlinear,
        baseline_modes=torch.zeros(1, 5, 1, dtype=torch.float64),
        logical_positions=torch.arange(5),
        valid_mask=torch.ones(5, dtype=torch.bool),
        source_mode_indices=(0,),
        impulse_logical_positions=(1, 2),
        max_lag=0,
        fft_length=4,
        finite_impulse_amplitudes=torch.tensor([1.0]),
        symmetric_amplitude_sets={
            "local": torch.tensor([0.05]),
            "operating": torch.tensor([1.0]),
        },
    )

    local = artifact.symmetric_by_label["local"]
    operating = artifact.symmetric_by_label["operating"]
    torch.testing.assert_close(
        local.impulse_responses,
        torch.ones_like(local.impulse_responses),
    )
    torch.testing.assert_close(
        operating.impulse_responses,
        torch.ones_like(operating.impulse_responses),
    )
    torch.testing.assert_close(
        local.even_residual_impulse_responses,
        torch.full_like(local.impulse_responses, 0.025),
    )
    torch.testing.assert_close(
        operating.even_residual_impulse_responses,
        torch.full_like(operating.impulse_responses, 0.5),
    )
    assert operating.relative_even_residual > (
        10 * local.relative_even_residual
    )
    torch.testing.assert_close(
        artifact.scale_similarity("local", "operating"),
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        artifact.finite.impulse_responses,
        torch.full_like(artifact.finite.impulse_responses, 1.5),
    )


def test_precausal_energy_is_reported_without_a_causality_claim() -> None:
    def reads_future(source: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                source[:, 1:, :],
                torch.zeros_like(source[:, :1, :]),
            ),
            dim=1,
        )

    artifact = analyze_modal_spectral_mapping(
        reads_future,
        baseline_modes=torch.zeros(1, 5, 1, dtype=torch.float64),
        logical_positions=torch.arange(5),
        valid_mask=torch.ones(5, dtype=torch.bool),
        impulse_logical_positions=(2,),
        max_lag=1,
        fft_length=4,
    )

    assert artifact.finite.precausal_response_frobenius == pytest.approx(1.0)
    assert artifact.finite.causal_window_response_frobenius == 0.0
    assert artifact.finite.total_valid_response_frobenius == pytest.approx(1.0)
    assert "not_causal_identification" in artifact.no_causality_claim
    assert artifact.accounting()["causality_claim"] is False


def test_origin_bound_fingerprints_expose_shift_variation() -> None:
    signs = torch.tensor(
        [1.0, -1.0, 1.0, -1.0, 1.0],
        dtype=torch.float64,
    ).reshape(1, 5, 1)

    artifact = analyze_modal_spectral_mapping(
        lambda source: source * signs,
        baseline_modes=torch.zeros(1, 5, 1, dtype=torch.float64),
        logical_positions=torch.arange(5),
        valid_mask=torch.ones(5, dtype=torch.bool),
        impulse_logical_positions=(1, 2),
        max_lag=0,
        fft_length=4,
    )

    response = artifact.finite
    assert response.impulse_logical_positions == (1, 2)
    assert response.origin_spectral_similarity.shape == (1, 2, 2)
    assert response.origin_spectral_similarity[0, 0, 1] == pytest.approx(-1.0)
    assert response.minimum_origin_spectral_similarity == pytest.approx(-1.0)
    assert len(set(response.impulse_response_sha256s)) == 2
    assert "shift_invariance_is_not_assumed" in (
        response.no_shift_invariance_claim
    )


def test_multi_origin_fft_window_must_be_fully_observed() -> None:
    arguments = {
        "function": lambda source: source,
        "baseline_modes": torch.zeros(1, 5, 1, dtype=torch.float64),
        "logical_positions": torch.arange(5),
        "valid_mask": torch.ones(5, dtype=torch.bool),
        "impulse_logical_positions": (0, 3),
    }

    artifact = analyze_modal_spectral_mapping(**arguments)

    assert artifact.max_lag == 1
    assert artifact.finite.lag_observation_counts == (2, 2)
    with pytest.raises(ValueError, match="every impulse origin"):
        analyze_modal_spectral_mapping(
            max_lag=2,
            **arguments,
        )


def test_logical_lag_alignment_does_not_use_tensor_offsets() -> None:
    positions = torch.tensor([0, 2, 3, 5, 99])
    mask = torch.tensor([True, True, True, True, False])

    def previous_tensor_row(source: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros_like(source[:, :1, :]),
                source[:, :-1, :],
            ),
            dim=1,
        )

    artifact = analyze_modal_spectral_mapping(
        previous_tensor_row,
        baseline_modes=torch.zeros(1, 5, 1, dtype=torch.float64),
        logical_positions=positions,
        valid_mask=mask,
        impulse_logical_positions=(0,),
        max_lag=3,
        fft_length=8,
    )

    response = artifact.finite.impulse_responses[0, 0, :, 0]
    torch.testing.assert_close(
        response,
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64),
    )
    assert artifact.finite.lag_observation_counts == (1, 0, 1, 1)


def test_energy_beyond_lag4_includes_responses_outside_fft_window() -> None:
    kernel = torch.zeros(6, 1, 1, dtype=torch.float64)
    kernel[5, 0, 0] = 1.0
    positions = torch.arange(8)
    mask = torch.ones(8, dtype=torch.bool)
    artifact = analyze_modal_spectral_mapping(
        _logical_causal_function(kernel, positions, mask),
        baseline_modes=torch.zeros(1, 8, 1, dtype=torch.float64),
        logical_positions=positions,
        valid_mask=mask,
        impulse_logical_positions=(1,),
        max_lag=4,
        fft_length=16,
    )

    assert artifact.finite.causal_window_response_frobenius == 0.0
    assert artifact.finite.postwindow_response_frobenius == pytest.approx(1.0)
    assert artifact.finite.energy_beyond_lag4_fraction == pytest.approx(1.0)


def test_similarity_components_cluster_matching_modal_fingerprints() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    artifact = analyze_modal_spectral_mapping(
        lambda source: source @ matrix,
        baseline_modes=torch.zeros(1, 4, 3, dtype=torch.float64),
        logical_positions=torch.arange(4),
        valid_mask=torch.ones(4, dtype=torch.bool),
        impulse_logical_positions=(1,),
        max_lag=0,
        fft_length=4,
        similarity_threshold=0.99,
    )

    similarity = artifact.finite.pairwise_spectral_similarity
    assert similarity[0, 1] > 0.999
    assert abs(float(similarity[0, 2])) < 1e-12
    assert artifact.finite.connected_components == ((0, 1), (2,))
    assert connected_components_from_spectral_similarity(
        similarity,
        source_mode_indices=(0, 1, 2),
        threshold=0.99,
    ) == ((0, 1), (2,))
    assert "descriptive_spectral_similarity" in (
        artifact.finite.clustering_semantics
    )


def test_reference_binding_and_strict_state_roundtrip() -> None:
    def nonlinear(source: torch.Tensor) -> torch.Tensor:
        return source.square() + source

    arguments = {
        "function": nonlinear,
        "logical_positions": torch.arange(4),
        "valid_mask": torch.ones(4, dtype=torch.bool),
        "impulse_logical_positions": (1,),
        "max_lag": 0,
        "fft_length": 4,
        "symmetric_amplitude_sets": {
            "local": torch.tensor([0.1], dtype=torch.float64),
        },
    }
    zero = analyze_modal_spectral_mapping(
        baseline_modes=torch.zeros(1, 4, 1, dtype=torch.float64),
        **arguments,
    )
    shifted = analyze_modal_spectral_mapping(
        baseline_modes=torch.ones(1, 4, 1, dtype=torch.float64),
        **arguments,
    )

    assert zero.baseline_modes_sha256 != shifted.baseline_modes_sha256
    assert zero.artifact_sha256 != shifted.artifact_sha256
    restored = ModalSpectralMapping.from_state_dict(zero.state_dict())
    assert restored.artifact_sha256 == zero.artifact_sha256
    torch.testing.assert_close(
        restored.finite.impulse_responses,
        zero.finite.impulse_responses,
    )

    unknown = copy.deepcopy(zero.state_dict())
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        ModalSpectralMapping.from_state_dict(unknown)

    tampered = copy.deepcopy(zero.state_dict())
    tampered["finite"]["impulse_responses"][0, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="binding mismatch"):
        ModalSpectralMapping.from_state_dict(tampered)

    zero.symmetric_scale_pair_similarity[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="integrity check failed"):
        zero.validate_integrity()


def test_rejects_invalid_per_source_amplitudes() -> None:
    arguments = {
        "function": lambda source: source,
        "baseline_modes": torch.zeros(1, 3, 2, dtype=torch.float64),
        "logical_positions": torch.arange(3),
        "valid_mask": torch.ones(3, dtype=torch.bool),
        "impulse_logical_positions": (1,),
        "max_lag": 0,
        "fft_length": 2,
    }
    with pytest.raises(ValueError, match=r"shape \[r_src\]"):
        analyze_modal_spectral_mapping(
            finite_impulse_amplitudes=torch.ones(1, dtype=torch.float64),
            **arguments,
        )
    with pytest.raises(ValueError, match="positive"):
        analyze_modal_spectral_mapping(
            finite_impulse_amplitudes=torch.tensor([1.0, 0.0]),
            **arguments,
        )
    with pytest.raises(ValueError, match="positive"):
        analyze_modal_spectral_mapping(
            symmetric_amplitude_sets={
                "bad": torch.tensor([0.1, -0.1]),
            },
            **arguments,
        )
