from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    fit_conditional_spectral_generator_with_source_basis,
)
from fisher_graph.graph_spectral_source_basis import (
    FitOnlyGraphSourceBasis,
    fit_graph_source_bases,
)


_BINDING = "91" * 32
_ORIGINS = (0, 1, 2)
_FIT_ORIGINS = (0, 2)


def _responses() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(703)
    responses = torch.randn(
        4,
        len(_ORIGINS),
        3,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [0.5, 1.25, 2.0, 0.75],
        dtype=torch.float64,
    )
    return responses, scales


def _graph_and_plan() -> tuple[
    torch.Tensor,
    torch.Tensor,
    FitOnlyGraphSourceBasis,
    ConditionalSpectralGeneratorPlan,
]:
    responses, scales = _responses()
    graph = fit_graph_source_bases(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        response_binding_sha256=_BINDING,
    )
    plan = fit_conditional_spectral_generator_with_source_basis(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        graph.basis(
            "signed_phase_graph_low_frequency",
            graph.source_modes,
        ),
        responses.shape[-1],
        source_basis_kind="signed_phase_graph_low_frequency",
        source_basis_fit_weighted_kernels_sha256=(
            graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=_BINDING,
    )
    return responses, scales, graph, plan


def test_fit_origin_leakage_is_excluded_from_graph_and_executor_fit() -> None:
    responses, scales, first_graph, first_plan = _graph_and_plan()
    changed = responses.clone()
    changed[:, _ORIGINS.index(1)].mul_(10_000.0).add_(123.0)

    second_graph = fit_graph_source_bases(
        changed,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        response_binding_sha256=_BINDING,
    )
    second_plan = fit_conditional_spectral_generator_with_source_basis(
        changed,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        second_graph.basis(
            "signed_phase_graph_low_frequency",
            second_graph.source_modes,
        ),
        changed.shape[-1],
        source_basis_kind="signed_phase_graph_low_frequency",
        source_basis_fit_weighted_kernels_sha256=(
            second_graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=_BINDING,
    )

    assert first_graph.artifact_sha256 == second_graph.artifact_sha256
    assert (
        first_graph.fit_weighted_kernels_sha256
        == second_graph.fit_weighted_kernels_sha256
    )
    torch.testing.assert_close(
        first_graph.signed_eigenvectors,
        second_graph.signed_eigenvectors,
    )
    torch.testing.assert_close(
        first_graph.magnitude_eigenvectors,
        second_graph.magnitude_eigenvectors,
    )
    assert first_plan.artifact_sha256 == second_plan.artifact_sha256
    torch.testing.assert_close(first_plan.source_basis, second_plan.source_basis)
    torch.testing.assert_close(first_plan.target_basis, second_plan.target_basis)
    torch.testing.assert_close(first_plan.knot_cores, second_plan.knot_cores)


def test_full_rank_graph_plan_roundtrips_every_fit_kernel() -> None:
    responses, scales, graph, plan = _graph_and_plan()

    assert plan.source_rank == responses.shape[0]
    assert plan.target_rank == responses.shape[-1]
    assert plan.weighted_relative_error < 1e-7
    assert graph.projection_relative_error(
        "signed_phase_graph_low_frequency",
        graph.source_modes,
    ) < 1e-7
    for origin in _FIT_ORIGINS:
        ordinal = _ORIGINS.index(origin)
        expected = responses[:, ordinal] * scales.view(-1, 1, 1)
        torch.testing.assert_close(
            plan.weighted_kernel_at_origin(origin),
            expected,
            atol=2e-7,
            rtol=2e-7,
        )


def test_graph_wavelet_source_basis_has_distinct_rank_semantics() -> None:
    responses, scales = _responses()
    graph = fit_graph_source_bases(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        response_binding_sha256=_BINDING,
    )
    plan = fit_conditional_spectral_generator_with_source_basis(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        torch.eye(responses.shape[0], dtype=torch.float64),
        responses.shape[-1],
        source_basis_kind="fit_only_graph_wavelet_gomp",
        source_basis_fit_weighted_kernels_sha256=(
            graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=_BINDING,
    )

    assert plan.rank_semantics == (
        "fit_only_graph_wavelet_gomp_localized_orthonormal_source_subspace"
    )


def test_signed_graph_preserves_opposition_that_magnitude_graph_erases() -> None:
    positive = torch.tensor(
        [
            [[1.0], [2.0], [0.5]],
            [[0.75], [-1.0], [1.5]],
        ],
        dtype=torch.float64,
    )
    responses = torch.stack(
        (
            positive,
            -positive,
        ),
        dim=0,
    )
    graph = fit_graph_source_bases(
        responses,
        torch.ones(2, dtype=torch.float64),
        (0, 1),
        (0, 1),
        response_binding_sha256=_BINDING,
        fft_length=4,
    )

    signed_low = graph.basis(
        "signed_phase_graph_low_frequency",
        1,
    )[:, 0]
    magnitude_low = graph.basis(
        "phase_blind_magnitude_graph_low_frequency",
        1,
    )[:, 0]
    assert float(signed_low.prod()) < 0.0
    assert float(magnitude_low.prod()) > 0.0
    torch.testing.assert_close(
        signed_low.abs(),
        torch.full((2,), 2.0**-0.5, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        magnitude_low.abs(),
        torch.full((2,), 2.0**-0.5, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        graph.signed_eigenvalues,
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        graph.magnitude_eigenvalues,
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_graph_basis_strict_state_roundtrip_and_tamper_rejection() -> None:
    _, _, graph, _ = _graph_and_plan()
    restored = FitOnlyGraphSourceBasis.from_state_dict(graph.state_dict())

    assert restored.artifact_sha256 == graph.artifact_sha256
    assert restored.metadata() == graph.metadata()
    torch.testing.assert_close(
        restored.signed_eigenvectors,
        graph.signed_eigenvectors,
    )

    unknown = copy.deepcopy(graph.state_dict())
    unknown["future_field"] = True
    with pytest.raises(ValueError, match="state fields differ"):
        FitOnlyGraphSourceBasis.from_state_dict(unknown)

    tampered = copy.deepcopy(graph.state_dict())
    tampered["signed_eigenvectors"][0, 0].add_(0.25)
    with pytest.raises(ValueError, match="serialized signed_eigenvectors"):
        FitOnlyGraphSourceBasis.from_state_dict(tampered)

    live = FitOnlyGraphSourceBasis.from_state_dict(graph.state_dict())
    live.magnitude_projection_energy[0].add_(0.1)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        live.validate_integrity()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        live.state_dict()


def test_graph_plan_runtime_matches_its_dense_control() -> None:
    _, _, _, plan = _graph_and_plan()
    runtime = plan.prepare(device="cpu", dtype=torch.float64)
    generator = torch.Generator().manual_seed(704)
    source = torch.randn(
        2,
        3,
        plan.source_modes,
        generator=generator,
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0, 1, 2], [0, 1, 2]],
        dtype=torch.int64,
    )
    valid_mask = torch.tensor(
        [[True, True, True], [True, False, True]],
    )
    source_mask = torch.tensor(
        [[True, True, True], [True, False, True]],
    )

    factorized = runtime(
        source,
        logical_positions=positions,
        valid_mask=valid_mask,
        source_mask=source_mask,
    )
    dense = runtime.forward_dense_control(
        source,
        logical_positions=positions,
        valid_mask=valid_mask,
        source_mask=source_mask,
    )

    torch.testing.assert_close(factorized, dense, atol=1e-12, rtol=1e-12)
    assert bool((factorized[~valid_mask] == 0.0).all())


def test_graph_plan_rejects_a_basis_fit_hash_mismatch() -> None:
    responses, scales = _responses()
    graph = fit_graph_source_bases(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        response_binding_sha256=_BINDING,
    )

    with pytest.raises(ValueError, match="not bound"):
        fit_conditional_spectral_generator_with_source_basis(
            responses,
            scales,
            _ORIGINS,
            _FIT_ORIGINS,
            graph.basis("signed_phase_graph_low_frequency", 2),
            2,
            source_basis_kind="signed_phase_graph_low_frequency",
            source_basis_fit_weighted_kernels_sha256="cd" * 32,
            response_binding_sha256=_BINDING,
        )
