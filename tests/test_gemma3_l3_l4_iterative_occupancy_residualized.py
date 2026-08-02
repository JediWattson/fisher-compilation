from __future__ import annotations

from collections import Counter
import hashlib

import pytest
import torch

from fisher_graph import gemma3_l3_l4_iterative_occupancy_route as route
from fisher_graph.gemma3_l3_l4_iterative_occupancy_residualized import (
    fit_gemma_iterative_residualized_occupancy_route_fold,
    fit_gemma_iterative_residualized_occupancy_route_full_provider,
    occupancy_residual_basis_coefficients,
    occupancy_residual_projection_matrix,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _family_weights(families: tuple[str, ...]) -> torch.Tensor:
    counts = Counter(families)
    return torch.tensor(
        [
            1.0 / (len(counts) * counts[family])
            for family in families
        ],
        dtype=torch.float64,
    )


def _dense_base(row_count: int) -> torch.Tensor:
    return torch.tensor(
        [
            tuple(
                1.0 if row & (1 << coordinate) else -1.0
                for coordinate in range(4)
            )
            for row in range(row_count)
        ],
        dtype=torch.float64,
    )


def _weighted_orthogonal_residual(
    base: torch.Tensor,
    raw: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weighted_base = weights.sqrt().unsqueeze(1) * base
    weighted_raw = weights.sqrt().unsqueeze(1) * raw
    projection = torch.linalg.pinv(weighted_base) @ weighted_raw
    return (raw - base @ projection).contiguous()


def _records(
    *,
    base: torch.Tensor,
    cumulative_occupancy: torch.Tensor,
    ew_occupancy: torch.Tensor,
    target: torch.Tensor,
    families: tuple[str, ...] | None = None,
    index_offset: int = 0,
) -> tuple[route.GemmaIterativeOccupancyConformalRouteFitRecord, ...]:
    row_count = int(base.shape[0])
    assert base.shape == (row_count, 4)
    assert cumulative_occupancy.shape == (row_count, 2)
    assert ew_occupancy.shape == (row_count, 2)
    assert target.shape == (row_count,)
    if families is None:
        assert row_count % 2 == 0
        families = tuple(f"family-{index // 2}" for index in range(row_count))
    assert len(families) == row_count

    result = []
    for row in range(row_count):
        index = index_offset + row
        shared = tuple(float(value) for value in base[row])
        cumulative = tuple(
            float(value)
            for value in torch.cat((base[row], cumulative_occupancy[row]))
        )
        ew = tuple(
            float(value)
            for value in torch.cat((base[row], ew_occupancy[row]))
        )
        result.append(
            route.GemmaIterativeOccupancyConformalRouteFitRecord(
                example_id=f"example-{index:03d}",
                family_id=families[row],
                model_inputs_sha256=_sha(f"inputs-{index}"),
                parent_execution_sha256=_sha(f"execution-{index}"),
                parent_observation_sha256=_sha(f"observation-{index}"),
                parent_h4_artifact_sha256=_sha("parent-h4"),
                prefix_sha256=_sha(f"prefix-{index}"),
                gradient_sha256=_sha(f"gradient-{index}"),
                parent_modal_sha256=_sha(f"parent-modal-{index}"),
                balance_feature_sha256=_sha(f"balance-{index}"),
                cumulative_occupancy_feature_sha256=_sha(
                    f"cumulative-occupancy-{index}"
                ),
                ew_occupancy_feature_sha256=_sha(f"ew-occupancy-{index}"),
                shared_feature_sha256=_sha(f"shared-{index}"),
                balance_contrast_feature_sha256=_sha(
                    f"balance-contrast-{index}"
                ),
                cumulative_occupancy_contrast_feature_sha256=_sha(
                    f"cumulative-contrast-{index}"
                ),
                ew_occupancy_contrast_feature_sha256=_sha(
                    f"ew-contrast-{index}"
                ),
                supervised_tokens=5,
                parent_signed_delta_nll_per_token=-float(target[row]),
                jacobian_by_cumulative_occupancy_conformal_coefficient=(
                    cumulative  # type: ignore[arg-type]
                ),
                jacobian_by_ew_occupancy_conformal_coefficient=(
                    ew  # type: ignore[arg-type]
                ),
                active_row_count=5,
                negative_balance_row_count=2,
                nonnegative_balance_row_count=3,
                top_mode_indices=(0, 1),
                top_mode_norms=(2.0, 1.0),
                balance_feature_std=0.2,
                cumulative_occupancy_feature_std=0.3,
                ew_occupancy_feature_std=0.4,
                top2_modal_energy_fraction=0.8,
            )
        )
    return tuple(result)


def _dense_problem(
    *,
    families: tuple[str, ...] | None = None,
    target_scale: float = 0.05,
) -> tuple[
    tuple[route.GemmaIterativeOccupancyConformalRouteFitRecord, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
]:
    row_count = 14
    if families is None:
        families = tuple(f"family-{index // 2}" for index in range(row_count))
    base = _dense_base(row_count)
    weights = _family_weights(families)
    position = torch.arange(row_count, dtype=torch.float64)
    raw_residual = torch.stack(
        (
            position - position.mean(),
            (position.remainder(3.0) - 1.0)
            + 0.1 * position.square(),
        ),
        dim=1,
    )
    residual = _weighted_orthogonal_residual(
        base,
        raw_residual,
        weights,
    )
    cumulative_projection = torch.tensor(
        [
            [0.40, -0.20],
            [0.15, 0.35],
            [-0.30, 0.10],
            [0.25, -0.45],
        ],
        dtype=torch.float64,
    )
    ew_projection = torch.tensor(
        [
            [-0.10, 0.50],
            [0.30, -0.25],
            [0.20, 0.15],
            [-0.40, 0.05],
        ],
        dtype=torch.float64,
    )
    cumulative_occupancy = base @ cumulative_projection + residual
    ew_residual = residual @ torch.tensor(
        [[0.75, -0.20], [0.30, 0.90]],
        dtype=torch.float64,
    )
    ew_occupancy = base @ ew_projection + ew_residual
    target_gamma = target_scale * torch.tensor(
        [1.0, -0.8, 0.6, -0.4, 0.9, -0.7],
        dtype=torch.float64,
    )
    target = torch.cat((base, residual), dim=1) @ target_gamma
    return (
        _records(
            base=base,
            cumulative_occupancy=cumulative_occupancy,
            ew_occupancy=ew_occupancy,
            target=target,
            families=families,
        ),
        base,
        cumulative_occupancy,
        ew_occupancy,
        families,
    )


def _parent() -> GemmaCausalResidualHead:
    return GemmaCausalResidualHead(
        site="layer.4.output",
        parent_runtime_binding_sha256=_sha("runtime"),
        residual_map_sha256=_sha("residual-map"),
        analysis_artifact_sha256=_sha("analysis"),
        fit_manifest_sha256=_sha("fit-manifest"),
        bridge_binding_sha256=_sha("bridge"),
        decoder=torch.eye(3, dtype=torch.float64),
        lag_kernel=torch.tensor(
            [
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.1],
                ]
            ],
            dtype=torch.float64,
        ),
        state_kernel=torch.empty((0, 0), dtype=torch.float64),
        conditioning="l3_source_modes",
        ridge=1.0e-6,
        fit_row_count=14,
        family_ids=tuple(f"family-{index}" for index in range(7)),
        fit_sequence_sha256s=tuple(
            sorted(_sha(f"sequence-{index}") for index in range(7))
        ),
        fit_objective="candidate_nll_vjp_metric_ridge_v1",
        weighted_residual_rmse=1.0,
        normalized_nll_direction_rmse=1.0,
        linearized_nll_residual_rmse=1.0,
    )


def test_fold_is_training_only_rejects_held_leak_and_is_order_deterministic(
) -> None:
    records, _base, cumulative, ew, _families = _dense_problem()
    first = fit_gemma_iterative_residualized_occupancy_route_fold(
        records,
        held_family_id="held-family",
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    reversed_fit = fit_gemma_iterative_residualized_occupancy_route_fold(
        tuple(reversed(records)),
        held_family_id="held-family",
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )

    assert reversed_fit.to_dict() == first.to_dict()
    assert first.train_example_ids == tuple(
        sorted(record.example_id for record in records)
    )
    assert first.train_family_ids == tuple(
        sorted({record.family_id for record in records})
    )
    assert first.train_fit_record_sha256s == tuple(
        sorted(record.fit_record_sha256 for record in records)
    )

    held_base = _dense_base(2)
    held_records = _records(
        base=held_base,
        cumulative_occupancy=cumulative[:2],
        ew_occupancy=ew[:2],
        target=torch.zeros(2, dtype=torch.float64),
        families=("held-family", "held-family"),
        index_offset=100,
    )
    with pytest.raises(
        ValueError,
        match="held family leaked into residualized training",
    ):
        fit_gemma_iterative_residualized_occupancy_route_fold(
            (*records, *held_records),
            held_family_id="held-family",
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        )


def test_arm_specific_projectors_match_weighted_training_projection_and_are_orthogonal(
) -> None:
    families = (
        "family-a",
        "family-a",
        "family-a",
        "family-a",
        "family-a",
        "family-b",
        "family-b",
        "family-b",
        "family-b",
        "family-c",
        "family-c",
        "family-c",
        "family-d",
        "family-d",
    )
    records, base, cumulative, ew, families = _dense_problem(
        families=families
    )
    weights = _family_weights(families)
    square_root_weights = weights.sqrt().unsqueeze(1)
    expected_cumulative = torch.linalg.pinv(
        square_root_weights * base
    ) @ (square_root_weights * cumulative)
    expected_ew = torch.linalg.pinv(
        square_root_weights * base
    ) @ (square_root_weights * ew)

    cumulative_fit = (
        fit_gemma_iterative_residualized_occupancy_route_fold(
            records,
            held_family_id="held-family",
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        )
    )
    ew_fit = fit_gemma_iterative_residualized_occupancy_route_fold(
        records,
        held_family_id="held-family",
        occupancy_kind=route.CENTERED_EW_OCCUPANCY,
    )
    cumulative_projection = occupancy_residual_projection_matrix(
        cumulative_fit
    )
    ew_projection = occupancy_residual_projection_matrix(ew_fit)

    assert torch.allclose(
        cumulative_projection,
        expected_cumulative,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert torch.allclose(
        ew_projection,
        expected_ew,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert not torch.allclose(cumulative_projection, ew_projection)

    for fit, occupancy, projection in (
        (cumulative_fit, cumulative, cumulative_projection),
        (ew_fit, ew, ew_projection),
    ):
        residual = occupancy - base @ projection
        base_scales = torch.sqrt(
            (weights[:, None] * base.square()).sum(dim=0)
        )
        residual_scales = torch.sqrt(
            (weights[:, None] * residual.square()).sum(dim=0)
        )
        correlations = (
            base.T @ (weights[:, None] * residual)
        ).abs() / (base_scales[:, None] * residual_scales[None, :])
        expected_maximum = float(correlations.max())
        assert fit.residualization_base_weighted_design_rank == 4
        assert (
            fit.maximum_absolute_weighted_base_residual_correlation
            == pytest.approx(expected_maximum, abs=2.0e-15)
        )
        assert expected_maximum <= (
            route.OCCUPANCY_RESIDUAL_ORTHOGONALITY_TOLERANCE
        )


def test_map_back_identity_and_trust_projection_commute_exactly() -> None:
    records, base, cumulative, _ew, families = _dense_problem(
        target_scale=8.0
    )
    fit = fit_gemma_iterative_residualized_occupancy_route_fold(
        records,
        held_family_id="held-family",
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    projection = occupancy_residual_projection_matrix(fit)
    residual = cumulative - base @ projection
    residualized_design = torch.cat((base, residual), dim=1)
    original_design = torch.cat((base, cumulative), dim=1)
    weights = _family_weights(families)
    target = -torch.tensor(
        [record.parent_signed_delta_nll_per_token for record in records],
        dtype=torch.float64,
    )
    scales = torch.sqrt(
        (weights[:, None] * residualized_design.square()).sum(dim=0)
    )
    standardized = residualized_design / scales
    pre_projection_gamma = (
        torch.linalg.solve(
            standardized.T @ (weights[:, None] * standardized)
            + route.OCCUPANCY_STANDARDIZED_RIDGE
            * torch.eye(6, dtype=torch.float64),
            standardized.T @ (weights * target),
        )
        / scales
    )
    pre_projection_theta = torch.cat(
        (
            pre_projection_gamma[:4]
            - projection @ pre_projection_gamma[4:],
            pre_projection_gamma[4:],
        )
    )
    theta = torch.tensor(
        fit.coefficients_by_occupancy_conformal_coefficient,
        dtype=torch.float64,
    )
    gamma = torch.tensor(
        occupancy_residual_basis_coefficients(fit),
        dtype=torch.float64,
    )

    assert fit.trust_projection_applied is True
    assert max(fit.pre_projection_corner_operator_norms) > (
        route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
    )
    assert max(fit.post_projection_corner_operator_norms) <= (
        route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
    )
    assert torch.allclose(
        theta,
        pre_projection_theta * fit.trust_projection_scale,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert torch.allclose(
        gamma,
        pre_projection_gamma * fit.trust_projection_scale,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert torch.allclose(
        original_design @ theta,
        residualized_design @ gamma,
        atol=2.0e-14,
        rtol=2.0e-13,
    )


def test_exact_span_occupancy_is_unsupported_without_raw_fallback() -> None:
    row_count = 14
    families = tuple(f"family-{index // 2}" for index in range(row_count))
    weights = _family_weights(families)
    base = _dense_base(row_count)
    position = torch.arange(row_count, dtype=torch.float64)
    supported_residual = _weighted_orthogonal_residual(
        base,
        (position.square() + 0.25 * position).unsqueeze(1),
        weights,
    ).squeeze(1)
    projection = torch.tensor(
        [
            [0.50, -0.25],
            [0.10, 0.20],
            [-0.40, 0.30],
            [0.25, -0.15],
        ],
        dtype=torch.float64,
    )
    occupancy = base @ projection
    occupancy[:, 1] += supported_residual
    target = (
        base
        @ torch.tensor([0.02, -0.01, 0.03, -0.02], dtype=torch.float64)
        + 0.04 * supported_residual
    )
    records = _records(
        base=base,
        cumulative_occupancy=occupancy,
        ew_occupancy=occupancy,
        target=target,
        families=families,
    )
    fit = fit_gemma_iterative_residualized_occupancy_route_fold(
        records,
        held_family_id="held-family",
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    gamma = occupancy_residual_basis_coefficients(fit)

    assert (
        fit.unsupported_occupancy_conformal_coefficient_indices == (4,)
    )
    assert fit.weighted_column_scale_by_occupancy_conformal_coefficient[
        4
    ] < 1.0e-12
    assert fit.coefficients_by_occupancy_conformal_coefficient[4] == 0.0
    assert gamma[4] == 0.0
    assert fit.coefficients_by_occupancy_conformal_coefficient[5] != 0.0
    assert gamma[5] != 0.0
    assert fit.occupancy_residual_energy_fraction_by_coordinate[
        0
    ] < 1.0e-24
    assert fit.occupancy_residual_energy_fraction_by_coordinate[1] > 0.0
    assert fit.raw_weighted_design_rank == 5
    assert fit.standardized_weighted_design_rank == 5


def test_rank_zero_base_uses_zero_projection_and_keeps_occupancy_residuals(
) -> None:
    row_count = 14
    base = torch.zeros(row_count, 4, dtype=torch.float64)
    position = torch.arange(row_count, dtype=torch.float64)
    occupancy = torch.stack(
        (
            position - position.mean(),
            position.remainder(3.0) - 1.0,
        ),
        dim=1,
    )
    target = occupancy @ torch.tensor([0.03, -0.02], dtype=torch.float64)
    records = _records(
        base=base,
        cumulative_occupancy=occupancy,
        ew_occupancy=occupancy,
        target=target,
    )
    fit = fit_gemma_iterative_residualized_occupancy_route_fold(
        records,
        held_family_id="held-family",
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    theta = fit.coefficients_by_occupancy_conformal_coefficient

    assert fit.residualization_base_weighted_design_rank == 0
    assert torch.equal(
        occupancy_residual_projection_matrix(fit),
        torch.zeros(4, 2, dtype=torch.float64),
    )
    assert fit.maximum_absolute_weighted_base_residual_correlation == 0.0
    assert fit.occupancy_residual_energy_fraction_by_coordinate == (
        1.0,
        1.0,
    )
    assert fit.unsupported_occupancy_conformal_coefficient_indices == (
        0,
        1,
        2,
        3,
    )
    assert theta[:4] == (0.0, 0.0, 0.0, 0.0)
    assert occupancy_residual_basis_coefficients(fit) == theta
    assert fit.raw_weighted_design_rank == 2
    assert fit.standardized_weighted_design_rank == 2


def test_residual_receipt_rejects_in_memory_metadata_tampering() -> None:
    records, _base, _cumulative, _ew, _families = _dense_problem()
    mutations = (
        (
            "occupancy_projection_on_base_by_base_and_occupancy_coordinate",
            lambda fit: (
                fit.occupancy_projection_on_base_by_base_and_occupancy_coordinate[
                    0
                ]
                + 1.0e-6,
                *fit.occupancy_projection_on_base_by_base_and_occupancy_coordinate[
                    1:
                ],
            ),
        ),
        (
            "residualization_base_weighted_design_rank",
            lambda fit: fit.residualization_base_weighted_design_rank - 1,
        ),
        (
            "occupancy_residual_energy_fraction_by_coordinate",
            lambda fit: (
                fit.occupancy_residual_energy_fraction_by_coordinate[0]
                + 1.0e-6,
                fit.occupancy_residual_energy_fraction_by_coordinate[1],
            ),
        ),
        (
            "maximum_absolute_weighted_base_residual_correlation",
            lambda fit: (
                fit.maximum_absolute_weighted_base_residual_correlation
                + 1.0e-12
            ),
        ),
    )

    for field, replacement in mutations:
        fit = fit_gemma_iterative_residualized_occupancy_route_fold(
            records,
            held_family_id="held-family",
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        )
        object.__setattr__(fit, field, replacement(fit))
        with pytest.raises(RuntimeError, match="occupancy fold fit drifted"):
            fit.validate_integrity()

    projection_tampered = (
        fit_gemma_iterative_residualized_occupancy_route_fold(
            records,
            held_family_id="held-family",
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        )
    )
    projection_values = (
        projection_tampered
        .occupancy_projection_on_base_by_base_and_occupancy_coordinate
    )
    values = list(projection_values)
    values[0] += 1.0e-6
    object.__setattr__(
        projection_tampered,
        "occupancy_projection_on_base_by_base_and_occupancy_coordinate",
        tuple(values),
    )
    with pytest.raises(RuntimeError, match="occupancy fold fit drifted"):
        occupancy_residual_projection_matrix(projection_tampered)


@pytest.mark.parametrize(
    ("occupancy_kind", "prepared", "accumulator_ops", "multiplications"),
    (
        (route.CENTERED_CUMULATIVE_OCCUPANCY, 8, 6, 0),
        (route.CENTERED_EW_OCCUPANCY, 9, 8, 2),
    ),
)
def test_residualized_full_provider_keeps_the_existing_runtime_envelope(
    occupancy_kind: route.OccupancyKind,
    prepared: int,
    accumulator_ops: int,
    multiplications: int,
) -> None:
    records, _base, _cumulative, _ew, _families = _dense_problem()
    provider = fit_gemma_iterative_residualized_occupancy_route_full_provider(
        records=records,
        occupancy_kind=occupancy_kind,
        parent_h4=_parent(),
    )
    provider.validate_integrity()
    receipt = dict(provider.resource_receipt)

    assert provider.fold_fit.fit_coordinate_system == (
        route.OCCUPANCY_FIT_COORDINATE_RESIDUALIZED
    )
    assert receipt["learned_float_scalar_count"] == 6
    assert receipt["prepared_float_scalar_count"] == prepared
    assert receipt["runtime_state_float_scalars_per_sequence"] == 4
    assert receipt["logical_linear_macs_per_token_upper_bound"] == 10
    assert receipt[
        "linear_accumulator_scalar_ops_per_token_upper_bound"
    ] == accumulator_ops
    assert receipt[
        "explicit_scalar_multiplications_per_token_upper_bound"
    ] == multiplications
    assert receipt["parent_decoder_invocations_per_token"] == 1
    assert not any(
        "residual" in field or "projection" in field for field in receipt
    )
