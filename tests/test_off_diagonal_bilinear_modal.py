from __future__ import annotations

import copy
import math

import pytest
import torch

from fisher_graph.off_diagonal_bilinear_modal import (
    DenseBilinearKernelRecovery,
    ExplicitPairProductFeatureMap,
    OffDiagonalBilinearFeatureMap,
    StandardizedBilinearPairDesign,
    apply_dense_bilinear_feature_kernels,
    build_explicit_pair_product_feature_map,
    build_off_diagonal_bilinear_feature_map,
    build_standardized_bilinear_pair_design,
    fit_dense_bilinear_feature_kernels,
)


_SOURCE_BINDING = "12" * 32
_RESPONSE_BINDING = "34" * 32


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


def _complete_pairs(width: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left in range(width)
        for right in range(left + 1, width)
    )


def _explicit_map(width: int = 4) -> ExplicitPairProductFeatureMap:
    return build_explicit_pair_product_feature_map(
        torch.linspace(0.5, 2.0, width, dtype=torch.float64),
        source_pairs=_complete_pairs(width),
        source_binding_sha256=_SOURCE_BINDING,
    )


def test_fixed_basis_features_match_subtracted_quadratic_form() -> None:
    basis = _orthonormal(5, 3, seed=11)
    feature_map = build_off_diagonal_bilinear_feature_map(
        basis,
        source_basis_binding_sha256=_SOURCE_BINDING,
    )
    runtime = feature_map.prepare(device="cpu", dtype=torch.float64)
    source = torch.randn(
        2,
        4,
        5,
        generator=torch.Generator().manual_seed(12),
        dtype=torch.float64,
    )

    actual = runtime.features(source)
    projected = source @ basis
    diagonal = torch.einsum(
        "...i,ia,ib->...ab",
        source.square(),
        basis,
        basis,
    )
    expected_matrix = (
        projected.unsqueeze(-1) * projected.unsqueeze(-2) - diagonal
    )
    expected = torch.stack(
        [
            expected_matrix[..., left, right]
            for left, right in feature_map.feature_pairs
        ],
        dim=-1,
    )

    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    assert feature_map.feature_pairs == tuple(
        (left, right)
        for left in range(3)
        for right in range(left, 3)
    )


def test_fixed_basis_runtime_is_exactly_zero_on_every_singleton_axis() -> None:
    feature_map = build_off_diagonal_bilinear_feature_map(
        _orthonormal(6, 4, seed=21),
        source_basis_binding_sha256=_SOURCE_BINDING,
    )
    runtime = feature_map.prepare(device="cpu", dtype=torch.float64)

    for mode in range(feature_map.source_modes):
        source = torch.zeros(
            3,
            feature_map.source_modes,
            dtype=torch.float64,
        )
        source[:, mode] = torch.tensor([0.5, -2.0, 7.0])
        assert torch.equal(
            runtime(source),
            torch.zeros(
                3,
                feature_map.feature_count,
                dtype=torch.float64,
            ),
        )


def test_one_hot_basis_omits_structurally_zero_diagonal_features() -> None:
    feature_map = build_off_diagonal_bilinear_feature_map(
        torch.eye(4, dtype=torch.float64),
        source_basis_binding_sha256=_SOURCE_BINDING,
    )

    assert feature_map.feature_pairs == _complete_pairs(4)
    assert feature_map.feature_count == 6
    assert feature_map.omitted_structural_zero_count == 4
    with pytest.raises(ValueError, match="structurally zero"):
        build_off_diagonal_bilinear_feature_map(
            torch.eye(4, dtype=torch.float64),
            source_basis_binding_sha256=_SOURCE_BINDING,
            feature_pairs=((0, 0), (0, 1)),
        )
    with pytest.raises(ValueError, match="lexicographic"):
        build_off_diagonal_bilinear_feature_map(
            _orthonormal(4, 2, seed=22),
            source_basis_binding_sha256=_SOURCE_BINDING,
            feature_pairs=((1, 1), (0, 1)),
        )


def test_fixed_basis_artifact_round_trip_integrity_and_runtime_isolation() -> None:
    caller_basis = _orthonormal(5, 3, seed=31)
    feature_map = build_off_diagonal_bilinear_feature_map(
        caller_basis,
        source_basis_binding_sha256=_SOURCE_BINDING,
    )
    caller_basis.zero_()
    assert bool(feature_map.source_basis.abs().sum() > 0.0)

    restored = OffDiagonalBilinearFeatureMap.from_state_dict(
        feature_map.state_dict()
    )
    assert restored.artifact_sha256 == feature_map.artifact_sha256
    torch.testing.assert_close(restored.source_basis, feature_map.source_basis)

    runtime = feature_map.prepare(device="cpu", dtype=torch.float32)
    source = torch.randn(2, 5, dtype=torch.float32)
    before = runtime(source)
    feature_map.source_basis.zero_()
    after = runtime(source)
    torch.testing.assert_close(after, before)
    with pytest.raises(ValueError, match="hash mismatch"):
        feature_map.validate_integrity()

    state = restored.state_dict()
    tampered = copy.deepcopy(state)
    assert isinstance(tampered["source_basis"], torch.Tensor)
    tampered["source_basis"][0, 0].add_(1.0)
    with pytest.raises(ValueError, match="hash, shape, or storage"):
        OffDiagonalBilinearFeatureMap.from_state_dict(tampered)
    extra = copy.deepcopy(state)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        OffDiagonalBilinearFeatureMap.from_state_dict(extra)


def test_explicit_pair_products_standardize_scale_and_are_singleton_exact() -> None:
    scales = torch.tensor([0.5, 2.0, 3.0, 4.0], dtype=torch.float64)
    feature_map = build_explicit_pair_product_feature_map(
        scales,
        source_pairs=((0, 2), (1, 2), (1, 3)),
        source_binding_sha256=_SOURCE_BINDING,
    )
    runtime = feature_map.prepare(device="cpu", dtype=torch.float64)
    standardized = torch.tensor(
        [[1.5, -2.0, 0.25, 3.0]],
        dtype=torch.float64,
    )
    raw = standardized * scales

    torch.testing.assert_close(
        runtime.features(raw),
        torch.tensor(
            [[2 * 1.5 * 0.25, 2 * -2.0 * 0.25, 2 * -2.0 * 3.0]],
            dtype=torch.float64,
        ),
    )
    for mode in range(feature_map.source_modes):
        singleton = torch.zeros(4, dtype=torch.float64)
        singleton[mode] = 9.0
        assert torch.equal(
            runtime(singleton),
            torch.zeros(feature_map.feature_count, dtype=torch.float64),
        )

    accounting = runtime.execution_accounting(raw)
    assert accounting.input_row_count == 1
    assert accounting.total_multiplies == 2 * feature_map.feature_count
    assert feature_map.accounting().prepared_float_scalar_count == 3


def test_explicit_pair_artifact_is_strict_and_hash_bound() -> None:
    feature_map = _explicit_map()
    restored = ExplicitPairProductFeatureMap.from_state_dict(
        feature_map.state_dict()
    )

    assert restored.artifact_sha256 == feature_map.artifact_sha256
    assert restored.source_pairs == feature_map.source_pairs
    state = restored.state_dict()
    assert isinstance(state["source_scales"], torch.Tensor)
    state["source_scales"][0].mul_(2.0)
    with pytest.raises(ValueError, match="hash, shape, or storage"):
        ExplicitPairProductFeatureMap.from_state_dict(state)


def test_standardized_explicit_design_is_rho_squared_one_hot_and_full_rank() -> None:
    feature_map = _explicit_map()
    radii = torch.tensor([0.5, 1.0], dtype=torch.float64)
    design = build_standardized_bilinear_pair_design(
        feature_map,
        pair_indices=feature_map.source_pairs,
        radii=radii,
    )
    expected = torch.stack(
        [
            torch.eye(feature_map.feature_count, dtype=torch.float64) * radius**2
            for radius in radii
        ],
        dim=1,
    )

    torch.testing.assert_close(
        design.design_matrix,
        expected,
        atol=2e-16,
        rtol=2e-16,
    )
    diagnostics = design.diagnostics()
    assert diagnostics.full_column_rank is True
    assert diagnostics.numerical_rank == feature_map.feature_count
    assert diagnostics.condition_number == pytest.approx(1.0)
    design.validate_against(feature_map)
    restored = StandardizedBilinearPairDesign.from_state_dict(
        design.state_dict()
    )
    assert restored.artifact_sha256 == design.artifact_sha256


def test_dense_recovery_handles_arbitrary_trailing_axes_and_round_trips() -> None:
    feature_map = _explicit_map()
    design = build_standardized_bilinear_pair_design(
        feature_map,
        pair_indices=feature_map.source_pairs,
        radii=torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    kernels = torch.randn(
        feature_map.feature_count,
        2,
        3,
        4,
        generator=torch.Generator().manual_seed(41),
        dtype=torch.float64,
    )
    responses = apply_dense_bilinear_feature_kernels(
        design.design_matrix,
        kernels,
    )

    recovery = fit_dense_bilinear_feature_kernels(
        design,
        responses,
        response_binding_sha256=_RESPONSE_BINDING,
    )

    assert recovery.trailing_response_shape == (2, 3, 4)
    assert recovery.numerical_rank == feature_map.feature_count
    assert recovery.condition_number == pytest.approx(1.0)
    assert recovery.relative_error < 1e-14
    torch.testing.assert_close(
        recovery.feature_kernels,
        kernels,
        atol=2e-14,
        rtol=2e-14,
    )
    torch.testing.assert_close(
        recovery.predict(design),
        responses,
        atol=2e-14,
        rtol=2e-14,
    )

    restored = DenseBilinearKernelRecovery.from_state_dict(
        recovery.state_dict()
    )
    assert restored.artifact_sha256 == recovery.artifact_sha256
    state = restored.state_dict()
    assert isinstance(state["feature_kernels"], torch.Tensor)
    state["feature_kernels"][0, 0, 0, 0].add_(1.0)
    with pytest.raises(ValueError, match="hash, shape, or storage"):
        DenseBilinearKernelRecovery.from_state_dict(state)


def test_dense_recovery_reports_irreducible_fit_error() -> None:
    feature_map = _explicit_map()
    design = build_standardized_bilinear_pair_design(
        feature_map,
        pair_indices=feature_map.source_pairs,
        radii=torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    kernels = torch.randn(
        feature_map.feature_count,
        2,
        generator=torch.Generator().manual_seed(51),
        dtype=torch.float64,
    )
    responses = apply_dense_bilinear_feature_kernels(
        design.design_matrix,
        kernels,
    )
    # For each feature column the radius design is [0.25, 1].  [1, -0.25]
    # is orthogonal to it, so this perturbation cannot be absorbed by K.
    residual_direction = torch.tensor([1.0, -0.25], dtype=torch.float64)
    responses = responses + (
        0.1
        * residual_direction.view(1, 2, 1)
        * torch.ones_like(responses)
    )

    recovery = fit_dense_bilinear_feature_kernels(
        design,
        responses,
        response_binding_sha256=_RESPONSE_BINDING,
    )

    assert recovery.residual_frobenius > 0.0
    assert recovery.relative_error > 0.01
    assert recovery.diagnostics().design.full_column_rank is True


def test_rank_deficient_pair_design_fails_closed() -> None:
    feature_map = _explicit_map()
    design = build_standardized_bilinear_pair_design(
        feature_map,
        pair_indices=feature_map.source_pairs[:3],
        radii=torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    diagnostics = design.diagnostics()

    assert diagnostics.full_column_rank is False
    assert diagnostics.numerical_rank == 3
    assert math.isinf(diagnostics.condition_number)
    responses = torch.randn(
        design.pair_count,
        design.radius_count,
        2,
        3,
        4,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="rank deficient"):
        fit_dense_bilinear_feature_kernels(
            design,
            responses,
            response_binding_sha256=_RESPONSE_BINDING,
        )
