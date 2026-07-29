from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fisher_graph.conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    fit_conditional_spectral_generator,
    fit_conditional_spectral_generator_with_source_basis,
)
from fisher_graph.graph_organized_svd import (
    GraphOrganizedSVDPlan,
    organize_conditional_svd_with_graph,
)
from fisher_graph.graph_spectral_source_basis import (
    FitOnlyGraphSourceBasis,
    fit_graph_source_bases,
)


_BINDING = "e4" * 32
_ORIGINS = (0, 1, 2)
_FIT_ORIGINS = (0, 2)
_BANDS = (0, 4, 8, 16)


def _responses() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    responses = torch.randn(
        16,
        len(_ORIGINS),
        3,
        5,
        generator=generator,
        dtype=torch.float64,
    )
    scales = torch.linspace(0.5, 1.5, 16, dtype=torch.float64)
    return responses, scales


def _plans(
    responses: torch.Tensor | None = None,
) -> tuple[
    FitOnlyGraphSourceBasis,
    ConditionalSpectralGeneratorPlan,
    GraphOrganizedSVDPlan,
]:
    if responses is None:
        responses, scales = _responses()
    else:
        _, scales = _responses()
    graph = fit_graph_source_bases(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        response_binding_sha256=_BINDING,
        fft_length=4,
    )
    base = fit_conditional_spectral_generator(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        8,
        responses.shape[-1],
        response_binding_sha256=_BINDING,
        fft_length=4,
    )
    organized = organize_conditional_svd_with_graph(
        base,
        graph,
        frequency_band_boundaries=_BANDS,
    )
    return graph, base, organized


def test_all_on_plan_and_runtime_are_exactly_the_global_svd_operator() -> None:
    _, base, organized = _plans()

    assert organized.pack_counts == (4, 2, 2)
    assert organized.source_plan_artifact_sha256 == base.artifact_sha256
    assert organized.target_modes == base.target_modes
    for origin in _ORIGINS:
        torch.testing.assert_close(
            organized.weighted_kernel_at_origin(origin),
            base.weighted_kernel_at_origin(origin),
            atol=1e-12,
            rtol=1e-12,
        )

    base_runtime = base.prepare(device="cpu", dtype=torch.float64)
    runtime = organized.prepare(device="cpu", dtype=torch.float64)
    source = torch.randn(
        2,
        5,
        organized.source_modes,
        generator=torch.Generator().manual_seed(17),
        dtype=torch.float64,
    )
    positions = torch.arange(5, dtype=torch.int64)
    valid = torch.ones(2, 5, dtype=torch.bool)
    source_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, False, False],
        ]
    )
    expected = base_runtime(
        source,
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source_mask,
    )
    actual = runtime(
        source,
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source_mask,
    )
    all_on = source_mask.unsqueeze(-1).expand(-1, -1, organized.pack_count)
    explicitly_all_on = runtime.forward_with_pack_mask(
        source,
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source_mask,
        pack_mask=all_on,
    )

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        explicitly_all_on,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_router_bound_certifies_impulse_error_and_accounting_tracks_mask() -> None:
    _, _, organized = _plans()
    standardized = torch.randn(
        7,
        organized.source_modes,
        generator=torch.Generator().manual_seed(23),
        dtype=torch.float64,
    )
    mask, _scores = organized.bound_mass_route_mask(
        standardized,
        origin=1,
        retained_bound_fraction=0.7,
    )
    full = organized.standardized_output_at_origin(
        standardized,
        origin=1,
    )
    routed = organized.standardized_output_at_origin(
        standardized,
        origin=1,
        pack_mask=mask,
    )
    actual_error = torch.linalg.vector_norm(
        (full - routed).flatten(start_dim=1),
        dim=1,
    )
    certified_bound = organized.omitted_source_response_bound(
        standardized,
        origin=1,
        pack_mask=mask,
    )

    assert bool((actual_error <= certified_bound + 1e-12).all())
    assert bool(mask.any(dim=1).all())
    all_mask, _ = organized.bound_mass_route_mask(
        standardized,
        origin=1,
        retained_bound_fraction=1.0,
    )
    assert bool(all_mask.all())
    tiny_mask, tiny_scores = organized.bound_mass_route_mask(
        torch.full(
            (1, organized.source_modes),
            1e-20,
            dtype=torch.float64,
        ),
        origin=1,
        retained_bound_fraction=0.9,
    )
    assert float(tiny_scores.sum()) > 0.0
    assert bool(tiny_mask.any())

    runtime = organized.prepare(device="cpu", dtype=torch.float64)
    positions = torch.arange(3, dtype=torch.int64)
    valid = torch.ones(3, dtype=torch.bool)
    runtime_mask = mask[:3]
    accounting = runtime.execution_accounting(
        logical_positions=positions,
        valid_mask=valid,
        pack_mask=runtime_mask,
        router_evaluated=True,
    )
    all_on_accounting = runtime.execution_accounting(
        logical_positions=positions,
        valid_mask=valid,
    )
    pack_ranks = torch.tensor(organized.pack_counts, dtype=torch.int64)
    expected_active = int((runtime_mask * pack_ranks).sum())
    assert accounting.active_rank_instances == expected_active
    assert accounting.active_rank_fraction < 1.0
    assert (
        accounting.routed_core_transport_macs
        < all_on_accounting.routed_core_transport_macs
    )
    assert accounting.admitted_active_pack_pairs > 0
    assert accounting.reference_router_score_multiplies > 0


def test_every_pack_subset_and_interpolated_certificate_is_conservative() -> None:
    _, _, organized = _plans()
    standardized = torch.randn(
        3,
        organized.source_modes,
        generator=torch.Generator().manual_seed(27),
        dtype=torch.float64,
    )
    for origin in _ORIGINS:
        core = organized.core_at_origin(origin)
        bounds = organized.norm_bounds_at_origin(origin)
        for lag in range(organized.lag_count):
            for pack in range(organized.pack_count):
                start = int(organized.pack_offsets[pack])
                stop = int(organized.pack_offsets[pack + 1])
                assert (
                    torch.linalg.matrix_norm(
                        core[lag, start:stop],
                        ord=2,
                    )
                    <= bounds[lag, pack]
                )
        full = organized.standardized_output_at_origin(
            standardized,
            origin=origin,
        )
        for bits in range(1 << organized.pack_count):
            mask = torch.tensor(
                [
                    bool(bits & (1 << pack))
                    for pack in range(organized.pack_count)
                ],
                dtype=torch.bool,
            ).expand(standardized.shape[0], -1)
            routed = organized.standardized_output_at_origin(
                standardized,
                origin=origin,
                pack_mask=mask,
            )
            actual = torch.linalg.vector_norm(
                (full - routed).flatten(start_dim=1),
                dim=1,
            )
            bound = organized.omitted_source_response_bound(
                standardized,
                origin=origin,
                pack_mask=mask,
            )
            assert bool((actual <= bound + 1e-12).all())


def test_prepared_router_reuses_projection_and_matches_explicit_mask() -> None:
    _, _, organized = _plans()
    runtime = organized.prepare(device="cpu", dtype=torch.float64)
    standardized = torch.randn(
        3,
        organized.source_modes,
        generator=torch.Generator().manual_seed(31),
        dtype=torch.float64,
    )
    physical = standardized * organized.source_scales
    positions = torch.arange(3, dtype=torch.int64)
    valid = torch.ones(3, dtype=torch.bool)

    routed, mask, scores = runtime.forward_bound_routed(
        physical,
        logical_positions=positions,
        valid_mask=valid,
        retained_bound_fraction=0.8,
    )
    explicit = runtime.forward_with_pack_mask(
        physical,
        logical_positions=positions,
        valid_mask=valid,
        pack_mask=mask,
    )

    torch.testing.assert_close(routed, explicit, atol=1e-12, rtol=1e-12)
    assert mask.shape == (3, organized.pack_count)
    assert scores.shape == mask.shape
    assert bool((scores >= 0.0).all())
    full_routed, full_mask, _ = runtime.forward_bound_routed(
        physical,
        logical_positions=positions,
        valid_mask=valid,
        retained_bound_fraction=1.0,
    )
    torch.testing.assert_close(
        full_routed,
        runtime(
            physical,
            logical_positions=positions,
            valid_mask=valid,
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    assert bool(full_mask.all())


def test_execution_accounting_matches_pack_level_loop_work() -> None:
    _, _, organized = _plans()
    runtime = organized.prepare(device="cpu", dtype=torch.float64)
    positions = torch.arange(5, dtype=torch.int64)
    valid = torch.ones(5, dtype=torch.bool)
    source_mask = torch.tensor([True, True, True, False, False])
    mask = torch.zeros(5, organized.pack_count, dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, (0, 2)] = True

    accounting = runtime.execution_accounting(
        logical_positions=positions,
        valid_mask=valid,
        source_mask=source_mask,
        pack_mask=mask,
    )

    assert accounting.valid_source_rows == 3
    assert accounting.valid_target_rows == 5
    assert accounting.admitted_causal_pairs == 9
    assert accounting.active_pack_instances == 4
    assert accounting.active_rank_instances == 12
    assert accounting.interpolated_active_rank_instances == 12
    assert accounting.admitted_active_rank_pairs == 36
    assert accounting.admitted_active_pack_pairs == 12
    assert accounting.source_standardization_divisions == 48
    assert accounting.source_projection_macs == 384
    assert accounting.routed_core_transport_macs == 180
    assert accounting.factorized_linear_macs == 564
    assert accounting.dense_linear_macs == 720
    assert accounting.core_accumulation_additions == 60
    assert accounting.active_rank_fraction == 0.5

    batched = runtime.execution_accounting(
        logical_positions=torch.arange(3, dtype=torch.int64),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    assert batched.active_rank_instances == 48
    assert batched.interpolated_active_rank_instances == 24


def test_controls_preserve_pack_sizes_and_the_all_on_operator() -> None:
    graph, base, signed = _plans()

    for kind, seed in (
        ("singular_contiguous_control", 0),
        ("random_size_matched_control", 29),
    ):
        control = organize_conditional_svd_with_graph(
            base,
            graph,
            organization_kind=kind,
            frequency_band_boundaries=_BANDS,
            organization_seed=seed,
            matched_pack_counts=signed.pack_counts,
        )
        assert control.pack_counts == signed.pack_counts
        torch.testing.assert_close(
            control.weighted_kernel_at_origin(1),
            base.weighted_kernel_at_origin(1),
            atol=1e-12,
            rtol=1e-12,
        )


def test_organization_excludes_heldout_origin_and_state_is_strict() -> None:
    responses, _ = _responses()
    _, _, first = _plans(responses)
    changed = responses.clone()
    changed[:, _ORIGINS.index(1)].mul_(10_000.0).add_(123.0)
    _, _, second = _plans(changed)

    assert first.artifact_sha256 == second.artifact_sha256
    torch.testing.assert_close(first.source_basis, second.source_basis)
    torch.testing.assert_close(first.knot_cores, second.knot_cores)

    restored = GraphOrganizedSVDPlan.from_state_dict(first.state_dict())
    assert restored.artifact_sha256 == first.artifact_sha256
    assert restored.metadata() == first.metadata()

    unknown = copy.deepcopy(first.state_dict())
    unknown["future_field"] = True
    with pytest.raises(ValueError, match="state fields differ"):
        GraphOrganizedSVDPlan.from_state_dict(unknown)

    tampered = copy.deepcopy(first.state_dict())
    tampered["knot_cores"][0, 0, 0, 0].add_(0.25)
    with pytest.raises(ValueError, match="serialized knot_cores"):
        GraphOrganizedSVDPlan.from_state_dict(tampered)

    wrong_dtype = copy.deepcopy(first.state_dict())
    wrong_dtype["source_scales"] = wrong_dtype["source_scales"].float()
    with pytest.raises(ValueError, match="serialized source_scales"):
        GraphOrganizedSVDPlan.from_state_dict(wrong_dtype)

    live = GraphOrganizedSVDPlan.from_state_dict(first.state_dict())
    live.source_basis[0, 0].add_(0.1)
    with pytest.raises(ValueError, match="plan hash mismatch"):
        live.validate_integrity()
    with pytest.raises(ValueError, match="plan hash mismatch"):
        live.weighted_kernel_at_origin(1)


def test_non_svd_source_and_noncanonical_control_assignment_are_rejected() -> None:
    responses, scales = _responses()
    graph, base, signed = _plans(responses)
    graph_source_plan = fit_conditional_spectral_generator_with_source_basis(
        responses,
        scales,
        _ORIGINS,
        _FIT_ORIGINS,
        graph.signed_eigenvectors[:, :8],
        responses.shape[-1],
        source_basis_kind="signed_phase_graph_low_frequency",
        source_basis_fit_weighted_kernels_sha256=(
            graph.fit_weighted_kernels_sha256
        ),
        response_binding_sha256=_BINDING,
        fft_length=4,
    )
    with pytest.raises(ValueError, match="global-SVD"):
        organize_conditional_svd_with_graph(
            graph_source_plan,
            graph,
            frequency_band_boundaries=_BANDS,
        )

    contiguous = organize_conditional_svd_with_graph(
        base,
        graph,
        organization_kind="singular_contiguous_control",
        frequency_band_boundaries=_BANDS,
        matched_pack_counts=signed.pack_counts,
    )
    with pytest.raises(ValueError, match="control pack assignments"):
        replace(
            contiguous,
            organization_kind="random_size_matched_control",
            organization_seed=29,
            artifact_sha256="",
        )
