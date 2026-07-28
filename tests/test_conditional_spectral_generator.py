from __future__ import annotations

import copy
import math

import pytest
import torch

from fisher_graph.conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
    evaluate_conditional_spectral_generator,
    fit_conditional_spectral_generator,
)


_BINDING = "ab" * 32


def _orthonormal(
    rows: int,
    columns: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(
        rows,
        columns,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.linalg.qr(values, mode="reduced").Q


def _synthetic_responses() -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
]:
    source_basis = _orthonormal(5, 2, seed=11)
    target_basis = _orthonormal(4, 3, seed=12)
    generator = torch.Generator().manual_seed(13)
    left_core = torch.randn(
        3,
        2,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    right_core = torch.randn(
        3,
        2,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    origins = (1, 3, 5)
    weighted = []
    for origin in origins:
        alpha = (origin - origins[0]) / (origins[-1] - origins[0])
        core = left_core * (1.0 - alpha) + right_core * alpha
        weighted.append(
            torch.einsum(
                "sa,lab,tb->slt",
                source_basis,
                core,
                target_basis,
            )
        )
    source_scales = torch.tensor(
        [0.5, 1.25, 0.75, 2.0, 1.5],
        dtype=torch.float64,
    )
    responses = (
        torch.stack(weighted, dim=1)
        / source_scales.view(-1, 1, 1, 1)
    )
    return responses, source_scales, origins


def _fit(
    *,
    source_rank: int = 2,
    target_rank: int = 3,
    input_transform: str = "standardized_linear",
) -> ConditionalSpectralGeneratorPlan:
    responses, scales, origins = _synthetic_responses()
    return fit_conditional_spectral_generator(
        responses,
        scales,
        origins,
        (1, 5),
        source_rank,
        target_rank,
        response_binding_sha256=_BINDING,
        input_transform=input_transform,  # type: ignore[arg-type]
    )


def test_fit_recovers_two_sided_tucker_and_unseen_origin() -> None:
    responses, scales, origins = _synthetic_responses()
    plan = _fit()

    assert plan.source_rank == 2
    assert plan.target_rank == 3
    assert plan.fit_knot_origins == (1, 5)
    assert plan.heldout_origins_used_for_fit is False
    assert plan.source_parseval_relative_error < 1e-12
    assert plan.target_parseval_relative_error < 1e-12
    assert plan.weighted_relative_error < 1e-7
    torch.testing.assert_close(
        plan.weighted_kernel_at_origin(3),
        responses[:, origins.index(3)] * scales.view(-1, 1, 1),
        atol=2e-7,
        rtol=2e-7,
    )

    metrics = evaluate_conditional_spectral_generator(
        plan,
        responses,
        origins,
        (3,),
        response_binding_sha256=_BINDING,
        require_heldout=True,
    )
    assert metrics.fit_origin_overlap == ()
    assert metrics.weighted_relative_error < 1e-7
    assert metrics.weighted_cosine > 1.0 - 1e-12
    assert metrics.fit_was_not_recomputed is True


def test_heldout_origin_is_excluded_from_fit_and_binding_is_enforced() -> None:
    responses, scales, origins = _synthetic_responses()
    first = _fit()
    changed = responses.clone()
    changed[:, 1].mul_(1000.0)
    second = fit_conditional_spectral_generator(
        changed,
        scales,
        origins,
        (1, 5),
        2,
        3,
        response_binding_sha256=_BINDING,
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.fit_weighted_kernels_sha256 == (
        second.fit_weighted_kernels_sha256
    )
    torch.testing.assert_close(first.source_basis, second.source_basis)
    torch.testing.assert_close(first.target_basis, second.target_basis)
    torch.testing.assert_close(first.knot_cores, second.knot_cores)

    with pytest.raises(ValueError, match="overlap fit knots"):
        evaluate_conditional_spectral_generator(
            first,
            responses,
            origins,
            (1,),
            response_binding_sha256=_BINDING,
            require_heldout=True,
        )
    with pytest.raises(ValueError, match="binding"):
        evaluate_conditional_spectral_generator(
            first,
            responses,
            origins,
            (3,),
            response_binding_sha256="cd" * 32,
        )


def test_fused_runtime_matches_dense_control_for_batches_gaps_and_masks() -> None:
    plan = _fit()
    runtime = plan.prepare(device="cpu", dtype=torch.float64)
    generator = torch.Generator().manual_seed(21)
    source = torch.randn(
        2,
        7,
        plan.source_modes,
        generator=generator,
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 5, 6, 7, 9],
        ],
        dtype=torch.int64,
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, False, True, True, True],
            [True, False, True, True, True, True, False],
        ]
    )
    source_mask = torch.tensor(
        [
            [True, False, True, False, True, False, False],
            [True, False, True, True, False, False, False],
        ]
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
    accounting = runtime.execution_accounting(
        logical_positions=positions,
        valid_mask=valid_mask,
        source_mask=source_mask,
    )
    assert accounting.valid_source_rows == int(source_mask.sum())
    assert accounting.valid_target_rows == int(valid_mask.sum())
    assert accounting.target_projection_macs == (
        int(valid_mask.sum()) * plan.target_rank * plan.target_modes
    )
    assert accounting.factorized_linear_macs == (
        accounting.source_projection_macs
        + accounting.core_transport_macs
        + accounting.target_projection_macs
    )
    assert accounting.factorized_pair_accumulation_additions == (
        accounting.admitted_causal_pairs * plan.target_rank
    )
    assert accounting.dense_control_pair_accumulation_additions == (
        accounting.admitted_causal_pairs * plan.target_modes
    )
    assert accounting.dense_control_total_linear_macs == (
        accounting.dense_control_materialization_macs
        + accounting.dense_control_linear_macs
    )


def test_source_conditioning_is_causal_and_targets_may_follow_last_knot() -> None:
    plan = _fit()
    runtime = plan.prepare(device="cpu", dtype=torch.float64)
    positions = torch.arange(1, 9, dtype=torch.int64)
    valid_mask = torch.ones(8, dtype=torch.bool)
    source_mask = torch.tensor(
        [True, False, True, False, True, False, False, False]
    )
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(
        8,
        plan.source_modes,
        generator=generator,
        dtype=torch.float64,
    )

    baseline = runtime(
        source,
        logical_positions=positions,
        valid_mask=valid_mask,
        source_mask=source_mask,
    )
    changed = source.clone()
    changed[4].add_(4.0)
    after = runtime(
        changed,
        logical_positions=positions,
        valid_mask=valid_mask,
        source_mask=source_mask,
    )

    torch.testing.assert_close(after[:4], baseline[:4])
    assert not torch.allclose(after[4:], baseline[4:])
    assert bool(torch.isfinite(after[5:]).all())

    invalid_source_mask = source_mask.clone()
    invalid_source_mask[-1] = True
    with pytest.raises(ValueError, match="active source origin"):
        runtime(
            source,
            logical_positions=positions,
            valid_mask=valid_mask,
            source_mask=invalid_source_mask,
        )


def test_plan_and_default_runtime_forbid_source_extrapolation() -> None:
    plan = _fit()
    with pytest.raises(ValueError, match="extrapolation"):
        plan.core_at_origin(0)
    with pytest.raises(ValueError, match="extrapolation"):
        plan.weighted_kernel_at_origin(6)

    runtime = plan.prepare(device="cpu", dtype=torch.float64)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    source = torch.ones(3, plan.source_modes, dtype=torch.float64)
    with pytest.raises(ValueError, match="active source origin"):
        runtime(
            source,
            logical_positions=positions,
            valid_mask=torch.ones(3, dtype=torch.bool),
        )


def test_diagonal_square_alias_has_explicit_no_cross_term_semantics() -> None:
    plan = _fit(input_transform="standardized_diagonal_square")
    assert plan.input_transform == "standardized_square"
    metadata = plan.metadata()
    assert metadata["cross_mode_terms_measured"] is False
    assert metadata["square_transform_scope"] == (
        "diagonal_per_source_mode_only_no_cross_terms"
    )
    with pytest.raises(ValueError, match="no raw linear kernel"):
        plan.linear_kernel_at_origin(3)

    runtime = plan.prepare(device="cpu", dtype=torch.float64)
    source = torch.tensor(
        [
            [0.25, -0.5, 1.0, -1.25, 0.75],
            [-0.5, 0.25, -0.75, 0.5, 1.5],
            [1.0, -1.0, 0.5, 0.25, -0.25],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor([1, 3, 5], dtype=torch.int64)
    mask = torch.ones(3, dtype=torch.bool)
    torch.testing.assert_close(
        runtime(
            source,
            logical_positions=positions,
            valid_mask=mask,
        ),
        runtime.forward_dense_control(
            source,
            logical_positions=positions,
            valid_mask=mask,
        ),
        atol=1e-12,
        rtol=1e-12,
    )
    accounting = runtime.execution_accounting(
        logical_positions=positions,
        valid_mask=mask,
    )
    assert accounting.diagonal_square_multiplies == source.numel()


def test_strict_roundtrip_unknown_field_and_tensor_tampering() -> None:
    plan = _fit()
    restored = ConditionalSpectralGeneratorPlan.from_state_dict(
        plan.state_dict()
    )
    assert restored.artifact_sha256 == plan.artifact_sha256
    assert restored.metadata() == plan.metadata()
    assert restored.source_basis.dtype == torch.float64
    assert restored.source_basis.device.type == "cpu"

    unknown = copy.deepcopy(plan.state_dict())
    unknown["future_field"] = 1
    with pytest.raises(ValueError, match="fields mismatch"):
        ConditionalSpectralGeneratorPlan.from_state_dict(unknown)

    tampered = copy.deepcopy(plan.state_dict())
    tampered["knot_cores"][0, 0, 0, 0].add_(1.0)
    with pytest.raises(ValueError, match="hash or shape mismatch"):
        ConditionalSpectralGeneratorPlan.from_state_dict(tampered)

    noncanonical = copy.deepcopy(plan.state_dict())
    noncanonical["source_scales"] = noncanonical["source_scales"].float()
    with pytest.raises(ValueError, match="canonical CPU float64"):
        ConditionalSpectralGeneratorPlan.from_state_dict(noncanonical)

    live = _fit()
    live.knot_cores[0, 0, 0, 0].add_(1.0)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        live.validate_integrity()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        live.core_at_origin(3)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        live.state_dict()


def test_rank_energy_and_storage_accounting_are_exact() -> None:
    plan = _fit(source_rank=1, target_rank=2)
    torch.testing.assert_close(
        plan.source_singular_values.square().sum(),
        torch.tensor(plan.weighted_total_energy, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        plan.target_singular_values.square().sum(),
        torch.tensor(plan.weighted_total_energy, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    assert math.isclose(
        float(plan.knot_cores.square().sum()),
        plan.weighted_retained_energy,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    expected_error = math.sqrt(
        (
            plan.weighted_total_energy
            - plan.weighted_retained_energy
        )
        / plan.weighted_total_energy
    )
    assert math.isclose(
        plan.weighted_relative_error,
        expected_error,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    accounting = plan.accounting()
    expected_coefficients = (
        plan.source_modes * plan.source_rank
        + plan.target_modes * plan.target_rank
        + plan.knot_count
        * plan.lag_count
        * plan.source_rank
        * plan.target_rank
    )
    assert plan.stored_coefficient_count == expected_coefficients
    assert accounting.stored_coefficient_count == expected_coefficients
    assert accounting.normalization_scalar_count == plan.source_modes
    assert accounting.prepared_float_scalar_count == (
        expected_coefficients + plan.source_modes
    )
    assert accounting.artifact_float_scalar_count == (
        expected_coefficients
        + plan.source_modes
        + plan.source_singular_values.numel()
        + plan.target_singular_values.numel()
        + accounting.artifact_scalar_metric_count
    )
    assert accounting.artifact_scalar_metric_count == 6

    narrow_unfolding = torch.randn(6, 2, 1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="source unfolding rank capacity"):
        fit_conditional_spectral_generator(
            narrow_unfolding,
            torch.ones(6, dtype=torch.float64),
            (0, 1),
            (0, 1),
            5,
            1,
            response_binding_sha256=_BINDING,
        )
