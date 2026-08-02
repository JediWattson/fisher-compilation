from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import pytest
import torch

from fisher_graph import gemma3_l3_l4_iterative_occupancy_route as route
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _parent(*, history: bool = False) -> GemmaCausalResidualHead:
    lag_kernel = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.1],
            ]
        ],
        dtype=torch.float64,
    )
    if history:
        lag_kernel = torch.cat(
            (
                lag_kernel,
                torch.tensor(
                    [
                        [
                            [0.5, 0.0, 0.0],
                            [0.0, 0.25, 0.0],
                        ]
                    ],
                    dtype=torch.float64,
                ),
            )
        )
    return GemmaCausalResidualHead(
        site="layer.4.output",
        parent_runtime_binding_sha256=_sha(1),
        residual_map_sha256=_sha(2),
        analysis_artifact_sha256=_sha(3),
        fit_manifest_sha256=_sha(4),
        bridge_binding_sha256=_sha(5),
        decoder=torch.eye(3, dtype=torch.float64),
        lag_kernel=lag_kernel,
        state_kernel=torch.empty((0, 0), dtype=torch.float64),
        conditioning="l3_source_modes",
        ridge=1.0e-6,
        fit_row_count=8,
        family_ids=("fit-a", "fit-b"),
        fit_sequence_sha256s=(_sha(6), _sha(7)),
        fit_objective="candidate_nll_vjp_metric_ridge_v1",
        weighted_residual_rmse=1.0,
        normalized_nll_direction_rmse=1.0,
        linearized_nll_residual_rmse=1.0,
    )


def _prefix(
    source_modes: torch.Tensor | None = None,
    *,
    active: torch.Tensor | None = None,
) -> Gemma3L3L4OnePassPrefix:
    if source_modes is None:
        source_modes = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 0.5],
                    [0.5, -1.0],
                ]
            ],
            dtype=torch.float64,
        )
    batch, length, _rank = source_modes.shape
    valid = torch.ones((batch, length), dtype=torch.bool)
    if active is None:
        active = valid.clone()
    return Gemma3L3L4OnePassPrefix(
        source_modes=source_modes,
        clamped_y3=torch.zeros(batch, length, 3, dtype=torch.float64),
        predicted_target_modal_delta=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        decoded_base_x4_delta=torch.zeros(
            batch,
            length,
            3,
            dtype=torch.float64,
        ),
        logical_positions=torch.arange(
            length,
            dtype=torch.int64,
        ).expand(batch, -1),
        valid_target_mask=valid,
        source_eligible_mask=valid.clone(),
        target_affected_mask=active,
        bridge_binding_sha256=_sha(5),
    )


def _slice_prefix(
    prefix: Gemma3L3L4OnePassPrefix,
    start: int,
    stop: int,
) -> Gemma3L3L4OnePassPrefix:
    return Gemma3L3L4OnePassPrefix(
        source_modes=prefix.source_modes[:, start:stop].clone(),
        clamped_y3=prefix.clamped_y3[:, start:stop].clone(),
        predicted_target_modal_delta=(
            prefix.predicted_target_modal_delta[:, start:stop].clone()
        ),
        decoded_base_x4_delta=(
            prefix.decoded_base_x4_delta[:, start:stop].clone()
        ),
        logical_positions=prefix.logical_positions[:, start:stop].clone(),
        valid_target_mask=prefix.valid_target_mask[:, start:stop].clone(),
        source_eligible_mask=(
            prefix.source_eligible_mask[:, start:stop].clone()
        ),
        target_affected_mask=(
            prefix.target_affected_mask[:, start:stop].clone()
        ),
        bridge_binding_sha256=prefix.bridge_binding_sha256,
    )


def _fit_record(
    example: str,
    family: str,
    *,
    cumulative: tuple[float, float, float, float, float, float],
    ew: tuple[float, float, float, float, float, float] | None = None,
    delta: float = 0.0,
) -> route.GemmaIterativeOccupancyConformalRouteFitRecord:
    if ew is None:
        ew = cumulative
    assert ew[:4] == cumulative[:4]
    return route.GemmaIterativeOccupancyConformalRouteFitRecord(
        example_id=example,
        family_id=family,
        model_inputs_sha256=_sha(10),
        parent_execution_sha256=_sha(11),
        parent_observation_sha256=_sha(12),
        parent_h4_artifact_sha256=_parent().artifact_sha256,
        prefix_sha256=_sha(13),
        gradient_sha256=_sha(14),
        parent_modal_sha256=_sha(15),
        balance_feature_sha256=_sha(16),
        cumulative_occupancy_feature_sha256=_sha(17),
        ew_occupancy_feature_sha256=_sha(18),
        shared_feature_sha256=_sha(19),
        balance_contrast_feature_sha256=_sha(20),
        cumulative_occupancy_contrast_feature_sha256=_sha(21),
        ew_occupancy_contrast_feature_sha256=_sha(22),
        supervised_tokens=5,
        parent_signed_delta_nll_per_token=delta,
        jacobian_by_cumulative_occupancy_conformal_coefficient=(
            cumulative
        ),
        jacobian_by_ew_occupancy_conformal_coefficient=ew,
        active_row_count=5,
        negative_balance_row_count=2,
        nonnegative_balance_row_count=3,
        top_mode_indices=(0, 1),
        top_mode_norms=(2.0, 1.0),
        balance_feature_std=0.25,
        cumulative_occupancy_feature_std=0.5,
        ew_occupancy_feature_std=0.4,
        top2_modal_energy_fraction=0.99,
    )


def _fold(
    coefficients: tuple[float, float, float, float, float, float],
    *,
    occupancy_kind: route.OccupancyKind,
) -> route.GemmaIterativeOccupancyConformalRouteFoldFit:
    corners = route._corner_operator_norms(coefficients)
    return route.GemmaIterativeOccupancyConformalRouteFoldFit(
        occupancy_kind=occupancy_kind,
        held_family_id="held",
        train_example_ids=("train",),
        train_family_ids=("family",),
        train_fit_record_sha256s=(_sha(23),),
        coefficients_by_occupancy_conformal_coefficient=coefficients,
        unsupported_occupancy_conformal_coefficient_indices=(),
        active_row_count=5,
        weighted_column_scale_by_occupancy_conformal_coefficient=(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        raw_weighted_design_rank=6,
        standardized_weighted_design_rank=6,
        raw_normal_condition_number=1.0,
        standardized_normal_condition_number=1.0,
        pre_projection_corner_operator_norms=corners,
        post_projection_corner_operator_norms=corners,
        trust_projection_scale=1.0,
        linearized_rmse_before=1.0,
        linearized_rmse_after=0.5,
        trust_projection_applied=False,
    )


def _provider(
    coefficients: tuple[float, float, float, float, float, float],
    *,
    occupancy_kind: route.OccupancyKind,
    parent: GemmaCausalResidualHead | None = None,
) -> route.GemmaCausalTop2OccupancyConformalRouteH4Provider:
    return route.GemmaCausalTop2OccupancyConformalRouteH4Provider(
        parent_h4=_parent() if parent is None else parent,
        parent_artifact_sha256=_sha(24),
        fold_fit=_fold(coefficients, occupancy_kind=occupancy_kind),
    )


@pytest.mark.parametrize(
    "occupancy_kind",
    (
        route.CENTERED_CUMULATIVE_OCCUPANCY,
        route.CENTERED_EW_OCCUPANCY,
    ),
)
def test_centered_occupancy_consumes_current_token_and_tie_is_nonnegative(
    occupancy_kind: route.OccupancyKind,
) -> None:
    balance = torch.tensor(
        [[1.0, 0.0, -0.5, 1.0 / 7.0]],
        dtype=torch.float64,
    )
    active = torch.ones_like(balance, dtype=torch.bool)
    zeros = torch.zeros(1, dtype=torch.float64)
    observed, numerator, denominator, negative, nonnegative = (
        route._occupancy_feature(
            balance=balance,
            active=active,
            initial_numerator=zeros,
            initial_denominator=zeros.clone(),
            occupancy_kind=occupancy_kind,
        )
    )
    signed = (-1.0, -1.0, 1.0, -1.0)
    expected: list[float] = []
    expected_numerator = 0.0
    expected_denominator = 0.0
    decay = (
        1.0
        if occupancy_kind == route.CENTERED_CUMULATIVE_OCCUPANCY
        else route.OCCUPANCY_EW_DECAY
    )
    for value in signed:
        expected_numerator = decay * expected_numerator + value
        expected_denominator = decay * expected_denominator + 1.0
        expected.append(expected_numerator / expected_denominator)

    assert torch.equal(
        observed,
        torch.tensor([expected], dtype=torch.float64),
    )
    assert float(numerator[0]) == expected_numerator
    assert float(denominator[0]) == expected_denominator
    assert (negative, nonnegative) == (1, 3)
    if occupancy_kind == route.CENTERED_CUMULATIVE_OCCUPANCY:
        assert expected == [-1.0, -1.0, -1.0 / 3.0, -0.5]


@pytest.mark.parametrize(
    "occupancy_kind",
    (
        route.CENTERED_CUMULATIVE_OCCUPANCY,
        route.CENTERED_EW_OCCUPANCY,
    ),
)
def test_padding_and_every_chunk_partition_are_exactly_invariant(
    occupancy_kind: route.OccupancyKind,
) -> None:
    balance = torch.tensor(
        [[-0.5, 0.75, -0.25, 0.0, -0.8]],
        dtype=torch.float64,
    )
    active = torch.tensor([[True, False, True, False, True]])
    zeros = torch.zeros(1, dtype=torch.float64)
    full, full_numerator, full_denominator, _, _ = (
        route._occupancy_feature(
            balance=balance,
            active=active,
            initial_numerator=zeros,
            initial_denominator=zeros.clone(),
            occupancy_kind=occupancy_kind,
        )
    )
    assert torch.equal(
        full[~active],
        torch.zeros(2, dtype=torch.float64),
    )

    for cut_mask in itertools.product((False, True), repeat=4):
        cuts = [0]
        cuts.extend(
            index
            for index, selected in enumerate(cut_mask, start=1)
            if selected
        )
        cuts.append(balance.shape[1])
        numerator = zeros
        denominator = zeros.clone()
        pieces: list[torch.Tensor] = []
        for start, stop in itertools.pairwise(cuts):
            piece, numerator, denominator, _, _ = (
                route._occupancy_feature(
                    balance=balance[:, start:stop],
                    active=active[:, start:stop],
                    initial_numerator=numerator,
                    initial_denominator=denominator,
                    occupancy_kind=occupancy_kind,
                )
            )
            pieces.append(piece)
        assert torch.equal(torch.cat(pieces, dim=1), full)
        assert torch.equal(numerator, full_numerator)
        assert torch.equal(denominator, full_denominator)

    compact, compact_numerator, compact_denominator, _, _ = (
        route._occupancy_feature(
            balance=balance[active].reshape(1, -1),
            active=torch.ones((1, 3), dtype=torch.bool),
            initial_numerator=zeros,
            initial_denominator=zeros.clone(),
            occupancy_kind=occupancy_kind,
        )
    )
    assert torch.equal(full[active], compact.flatten())
    assert torch.equal(full_numerator, compact_numerator)
    assert torch.equal(full_denominator, compact_denominator)


def test_four_corner_projection_is_global_radial_and_interior_safe() -> None:
    exact = torch.tensor(
        [
            route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float64,
    )
    projected, pre, post, scale, applied = route._project_coefficients(
        exact,
        supported=(0, 1, 2, 3, 4, 5),
    )
    assert torch.equal(projected, exact)
    assert pre == (route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND,) * 4
    assert post == pre
    assert scale == 1.0
    assert applied is False

    raw = torch.tensor(
        [3.0, -4.0, 2.0, 0.5, -1.0, 0.0],
        dtype=torch.float64,
    )
    projected, pre, post, scale, applied = route._project_coefficients(
        raw,
        supported=(0, 1, 2, 3, 4),
    )
    assert applied is True
    assert scale < 1.0
    assert max(pre) > route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
    assert max(post) <= route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND
    assert projected[5].item() == 0.0
    assert torch.equal(projected[:5], raw[:5] * scale)
    for g in torch.linspace(-1.0, 1.0, 31):
        for occupancy in torch.linspace(-1.0, 1.0, 31):
            a = float(
                projected[0]
                + g * projected[2]
                + occupancy * projected[4]
            )
            b = float(
                projected[1]
                + g * projected[3]
                + occupancy * projected[5]
            )
            assert math.hypot(a, b) <= (
                route.OCCUPANCY_CONFORMAL_OPERATOR_NORM_BOUND + 1.0e-12
            )


@pytest.mark.parametrize(
    "occupancy_kind",
    (
        route.CENTERED_CUMULATIVE_OCCUPANCY,
        route.CENTERED_EW_OCCUPANCY,
    ),
)
def test_standardized_ridge_is_scale_fair_and_unsupported_is_exact_zero(
    occupancy_kind: route.OccupancyKind,
) -> None:
    scales = (1.0, 10.0, 0.1, 4.0, 1.0e-4, 0.0)
    target_coefficients = (0.01, -0.02, 0.03, -0.01, 0.04, 0.0)
    rows: list[
        route.GemmaIterativeOccupancyConformalRouteFitRecord
    ] = []
    for family in ("a", "b"):
        for index in range(5):
            jacobian = tuple(
                scales[column] if column == index else 0.0
                for column in range(6)
            )
            rows.append(
                _fit_record(
                    f"{family}-{index}",
                    family,
                    cumulative=jacobian,  # type: ignore[arg-type]
                    delta=-(scales[index] * target_coefficients[index]),
                )
            )
    fit = route.fit_gemma_iterative_occupancy_conformal_route_fold(
        rows,
        held_family_id="held",
        occupancy_kind=occupancy_kind,
    )
    assert fit.unsupported_occupancy_conformal_coefficient_indices == (5,)
    assert fit.coefficients_by_occupancy_conformal_coefficient[5] == 0.0
    assert fit.raw_weighted_design_rank == 5
    assert fit.standardized_weighted_design_rank == 5
    assert fit.raw_normal_condition_number > 1.0e9
    assert fit.standardized_normal_condition_number == pytest.approx(
        1.0,
        abs=1.0e-12,
    )
    assert fit.weighted_column_scale_by_occupancy_conformal_coefficient[
        4
    ] == pytest.approx(1.0e-4 / math.sqrt(5.0))
    for actual, expected in zip(
        fit.coefficients_by_occupancy_conformal_coefficient[:5],
        target_coefficients[:5],
        strict=True,
    ):
        assert actual == pytest.approx(expected, rel=2.0e-6, abs=1.0e-10)


@dataclass
class _Example:
    example_id: str = "example"
    family_id: str = "family"
    model_inputs_sha256: str = _sha(30)


@dataclass
class _Execution:
    prefix: Gemma3L3L4OnePassPrefix
    candidate_h4: torch.Tensor
    h4_head_sha256: str
    artifact_sha256: str = _sha(31)
    model_inputs_sha256: str = _sha(30)

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()


def _observation() -> GemmaH4DampingFiniteNLLObservation:
    return GemmaH4DampingFiniteNLLObservation(
        example_id="example",
        family_id="family",
        supervised_tokens=5,
        source_summed_nll=4.0,
        candidate_summed_nll=4.5,
        source_to_candidate_summed_kl=0.2,
        top1_matches=3,
        source_logits_sha256=_sha(32),
        candidate_logits_sha256=_sha(33),
        targets_sha256=_sha(34),
    )


def _shared_vjp_record() -> tuple[
    route.GemmaIterativeOccupancyConformalRouteFitRecord,
    GemmaCausalResidualHead,
    Gemma3L3L4OnePassPrefix,
    torch.Tensor,
    torch.Tensor,
]:
    parent = _parent()
    prefix = _prefix()
    candidate_h4 = torch.zeros(1, 5, 3, dtype=torch.float64)
    gradient = torch.tensor(
        [
            [
                [0.25, -0.5, 0.1],
                [1.0, 0.25, -0.2],
                [-0.5, 0.75, 0.3],
                [0.2, -0.1, 0.4],
                [-0.7, 0.5, -0.6],
            ]
        ],
        dtype=torch.float64,
    )
    record = route.build_gemma_iterative_occupancy_conformal_route_fit_record(
        example=_Example(),
        parent_execution=_Execution(
            prefix=prefix,
            candidate_h4=candidate_h4,
            h4_head_sha256=parent.artifact_sha256,
        ),
        gradient=gradient,
        parent_h4=parent,
        parent_observation=_observation(),
    )
    return record, parent, prefix, candidate_h4, gradient


def test_one_parent_vjp_binds_shared_first_four_and_distinct_occupancy() -> None:
    record, _parent_head, _prefix_value, _candidate_h4, _gradient = (
        _shared_vjp_record()
    )
    cumulative = (
        record.jacobian_by_cumulative_occupancy_conformal_coefficient
    )
    ew = record.jacobian_by_ew_occupancy_conformal_coefficient
    assert cumulative[:4] == ew[:4]
    assert cumulative[4:] != ew[4:]
    assert record.cumulative_occupancy_feature_sha256 != (
        record.ew_occupancy_feature_sha256
    )
    assert record.gradient_sha256


@pytest.mark.parametrize(
    "occupancy_kind",
    (
        route.CENTERED_CUMULATIVE_OCCUPANCY,
        route.CENTERED_EW_OCCUPANCY,
    ),
)
def test_all_six_analytic_coordinates_match_finite_displacement(
    occupancy_kind: route.OccupancyKind,
) -> None:
    record, parent, prefix, candidate_h4, gradient = _shared_vjp_record()
    expected = (
        record.jacobian_by_cumulative_occupancy_conformal_coefficient
        if occupancy_kind == route.CENTERED_CUMULATIVE_OCCUPANCY
        else record.jacobian_by_ew_occupancy_conformal_coefficient
    )
    epsilon = 1.0e-6
    for coefficient in range(6):
        plus = [0.0] * 6
        minus = [0.0] * 6
        plus[coefficient] = epsilon
        minus[coefficient] = -epsilon
        plus_correction = _provider(
            tuple(plus),  # type: ignore[arg-type]
            occupancy_kind=occupancy_kind,
            parent=parent,
        ).correction(prefix, candidate_h4)
        minus_correction = _provider(
            tuple(minus),  # type: ignore[arg-type]
            occupancy_kind=occupancy_kind,
            parent=parent,
        ).correction(prefix, candidate_h4)
        finite = float(
            (
                gradient * (plus_correction - minus_correction)
            ).sum()
            / (2.0 * epsilon * _observation().supervised_tokens)
        )
        assert finite == pytest.approx(
            expected[coefficient],
            rel=1.0e-9,
            abs=1.0e-9,
        )


@pytest.mark.parametrize(
    "occupancy_kind",
    (
        route.CENTERED_CUMULATIVE_OCCUPANCY,
        route.CENTERED_EW_OCCUPANCY,
    ),
)
def test_provider_parent_modal_chunks_and_state_are_exactly_invariant(
    occupancy_kind: route.OccupancyKind,
) -> None:
    provider = _provider(
        (0.03, -0.02, 0.01, 0.04, -0.03, 0.02),
        occupancy_kind=occupancy_kind,
        parent=_parent(history=True),
    )
    prefix = _prefix(
        active=torch.tensor([[True, True, False, True, True]])
    )
    realized = torch.zeros(1, 5, 3, dtype=torch.float64)
    parent_modal = provider.parent_h4.modal_correction(prefix, realized)
    initial = provider.initial_state(1, device="cpu", dtype=torch.float64)
    full, full_state = provider.correction_from_parent_modal_with_state(
        prefix,
        parent_modal,
        initial,
    )

    pieces: list[torch.Tensor] = []
    state = provider.initial_state(1, device="cpu", dtype=torch.float64)
    for start, stop in ((0, 1), (1, 3), (3, 4), (4, 5)):
        piece, state = provider.correction_from_parent_modal_with_state(
            _slice_prefix(prefix, start, stop),
            parent_modal[:, start:stop],
            state,
        )
        pieces.append(piece)
    assert torch.equal(torch.cat(pieces, dim=1), full)
    for name in (
        "balance_numerator",
        "balance_denominator",
        "occupancy_numerator",
        "occupancy_denominator",
    ):
        assert torch.equal(getattr(state, name), getattr(full_state, name))
    assert torch.equal(initial.balance_numerator, torch.zeros(1))
    assert torch.equal(initial.occupancy_numerator, torch.zeros(1))

    reset_parent_modal = provider.parent_h4.modal_correction(
        _slice_prefix(prefix, 3, 5),
        realized[:, 3:5],
    )
    assert not torch.equal(reset_parent_modal, parent_modal[:, 3:5])


@pytest.mark.parametrize(
    ("occupancy_kind", "prepared", "state_ops", "state_mults"),
    (
        (route.CENTERED_CUMULATIVE_OCCUPANCY, 8, 6, 0),
        (route.CENTERED_EW_OCCUPANCY, 9, 8, 2),
    ),
)
def test_provider_resources_and_artifact_integrity_are_exact(
    occupancy_kind: route.OccupancyKind,
    prepared: int,
    state_ops: int,
    state_mults: int,
) -> None:
    provider = _provider(
        (0.03, -0.02, 0.01, 0.04, -0.03, 0.02),
        occupancy_kind=occupancy_kind,
    )
    provider.validate_integrity()
    receipt = dict(provider.resource_receipt)
    assert receipt["learned_float_scalar_count"] == 6
    assert receipt["derived_prepared_float_scalar_count"] == 2
    assert receipt["fixed_decay_float_scalar_count"] == (
        int(occupancy_kind == route.CENTERED_EW_OCCUPANCY)
    )
    assert receipt["prepared_float_scalar_count"] == prepared
    assert receipt["runtime_state_float_scalars_per_sequence"] == 4
    assert receipt["logical_linear_macs_per_token_upper_bound"] == 10
    assert receipt[
        "linear_accumulator_scalar_ops_per_token_upper_bound"
    ] == state_ops
    assert receipt[
        "explicit_scalar_multiplications_per_token_upper_bound"
    ] == state_mults
    assert receipt["nonlinear_scalar_ops_per_token_upper_bound"] == 6
    assert receipt[
        "zero_denominator_comparisons_per_token_upper_bound"
    ] == 1
    assert receipt[
        "negative_balance_comparisons_per_token_upper_bound"
    ] == 1
    assert receipt["parent_decoder_invocations_per_token"] == 1

    other_kind = (
        route.CENTERED_EW_OCCUPANCY
        if occupancy_kind == route.CENTERED_CUMULATIVE_OCCUPANCY
        else route.CENTERED_CUMULATIVE_OCCUPANCY
    )
    wrong_state = route.GemmaCausalTop2OccupancyConformalRouteState(
        balance_numerator=torch.zeros(1, dtype=torch.float64),
        balance_denominator=torch.zeros(1, dtype=torch.float64),
        occupancy_numerator=torch.zeros(1, dtype=torch.float64),
        occupancy_denominator=torch.zeros(1, dtype=torch.float64),
        occupancy_kind=other_kind,
        provider_artifact_sha256=provider.artifact_sha256,
    )
    prefix = _prefix()
    parent_modal = provider.parent_h4.modal_correction(
        prefix,
        torch.zeros(1, 5, 3, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="state"):
        provider.route_parent_modal_with_state(
            prefix,
            parent_modal,
            wrong_state,
        )

    provider.parent_h4.lag_kernel[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="drifted"):
        provider.validate_integrity()


@pytest.mark.parametrize(
    ("numerator_field", "denominator_field"),
    (
        ("balance_numerator", "balance_denominator"),
        ("occupancy_numerator", "occupancy_denominator"),
    ),
)
def test_state_rejects_forged_tiny_denominator_carry(
    numerator_field: str,
    denominator_field: str,
) -> None:
    provider = _provider(
        (0.03, -0.02, 0.01, 0.04, -0.03, 0.02),
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    values = {
        "balance_numerator": torch.zeros(1, dtype=torch.float64),
        "balance_denominator": torch.zeros(1, dtype=torch.float64),
        "occupancy_numerator": torch.zeros(1, dtype=torch.float64),
        "occupancy_denominator": torch.zeros(1, dtype=torch.float64),
    }
    values[numerator_field] = torch.tensor([1.0e-6], dtype=torch.float64)
    values[denominator_field] = torch.tensor(
        [1.0e-12],
        dtype=torch.float64,
    )

    with pytest.raises(ValueError, match="occupancy route state is invalid"):
        route.GemmaCausalTop2OccupancyConformalRouteState(
            **values,
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
            provider_artifact_sha256=provider.artifact_sha256,
        )


def test_fit_record_in_memory_tampering_is_rejected_by_record_and_fitter(
) -> None:
    tampered = _fit_record(
        "a-0",
        "a",
        cumulative=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    valid = _fit_record(
        "b-0",
        "b",
        cumulative=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    )
    object.__setattr__(
        tampered,
        "jacobian_by_cumulative_occupancy_conformal_coefficient",
        (9.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="occupancy fit record drifted"):
        tampered.validate_integrity()
    with pytest.raises(RuntimeError, match="occupancy fit record drifted"):
        route.fit_gemma_iterative_occupancy_conformal_route_fold(
            (tampered, valid),
            held_family_id="held",
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        )


def test_fold_fit_in_memory_tampering_is_rejected_by_fold_and_provider(
) -> None:
    provider = _provider(
        (0.03, -0.02, 0.01, 0.04, -0.03, 0.02),
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    object.__setattr__(
        provider.fold_fit,
        "raw_weighted_design_rank",
        provider.fold_fit.raw_weighted_design_rank - 1,
    )

    with pytest.raises(RuntimeError, match="occupancy fold fit drifted"):
        provider.fold_fit.validate_integrity()
    with pytest.raises(RuntimeError, match="occupancy fold fit drifted"):
        provider.validate_integrity()


def test_cumulative_recurrence_has_additive_accounting_and_optional_counters(
) -> None:
    balance = torch.tensor(
        [[-0.5, 0.75, -0.25, 0.0]],
        dtype=torch.float64,
    )
    active = torch.tensor([[True, False, True, True]])
    zeros = torch.zeros(1, dtype=torch.float64)
    counted = route._occupancy_feature(
        balance=balance,
        active=active,
        initial_numerator=zeros,
        initial_denominator=zeros.clone(),
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    uncounted = route._occupancy_feature(
        balance=balance,
        active=active,
        initial_numerator=zeros,
        initial_denominator=zeros.clone(),
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        collect_sign_counts=False,
    )

    assert torch.equal(
        counted[0],
        torch.tensor([[1.0, 0.0, 1.0, 1.0 / 3.0]], dtype=torch.float64),
    )
    assert torch.equal(counted[1], torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(counted[2], torch.tensor([3.0], dtype=torch.float64))
    assert counted[3:] == (2, 1)
    assert all(
        torch.equal(left, right)
        for left, right in zip(counted[:3], uncounted[:3], strict=True)
    )
    assert uncounted[3:] == (0, 0)

    receipt = dict(
        _provider(
            (0.03, -0.02, 0.01, 0.04, -0.03, 0.02),
            occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
        ).resource_receipt
    )
    assert receipt[
        "linear_accumulator_scalar_ops_per_token_upper_bound"
    ] == 6
    assert receipt[
        "explicit_scalar_multiplications_per_token_upper_bound"
    ] == 0


def test_fitter_selects_the_requested_occupancy_arm_jacobian() -> None:
    records = tuple(
        _fit_record(
            f"{family}-0",
            family,
            cumulative=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ew=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            delta=-0.04,
        )
        for family in ("a", "b")
    )
    cumulative = route.fit_gemma_iterative_occupancy_conformal_route_fold(
        records,
        held_family_id="held",
        occupancy_kind=route.CENTERED_CUMULATIVE_OCCUPANCY,
    )
    ew = route.fit_gemma_iterative_occupancy_conformal_route_fold(
        records,
        held_family_id="held",
        occupancy_kind=route.CENTERED_EW_OCCUPANCY,
    )

    assert (
        cumulative.unsupported_occupancy_conformal_coefficient_indices
        == (0, 1, 2, 3, 5)
    )
    assert ew.unsupported_occupancy_conformal_coefficient_indices == (
        0,
        1,
        2,
        3,
        4,
    )
    assert cumulative.coefficients_by_occupancy_conformal_coefficient[
        4
    ] == pytest.approx(0.04, rel=2.0e-6)
    assert cumulative.coefficients_by_occupancy_conformal_coefficient[5] == 0.0
    assert ew.coefficients_by_occupancy_conformal_coefficient[4] == 0.0
    assert ew.coefficients_by_occupancy_conformal_coefficient[
        5
    ] == pytest.approx(0.04, rel=2.0e-6)


@pytest.mark.parametrize(
    ("occupancy_kind", "recipe", "jacobian_field", "expected_resources"),
    (
        (
            route.CENTERED_CUMULATIVE_OCCUPANCY,
            route.GEMMA_ITERATIVE_CUMULATIVE_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE,
            "jacobian_by_cumulative_occupancy_conformal_coefficient",
            {
                "learned_parameter_count": 6,
                "logical_macs_per_token_upper_bound": 10,
                "prepared_float_scalar_count": 8,
                "derived_constant_float_count": 2,
                "fixed_decay_float_scalar_count": 0,
                "runtime_state_float_count_per_sequence": 4,
                "nonlinear_scalar_ops_per_token_upper_bound": 6,
                "linear_accumulator_scalar_ops_per_token_upper_bound": 6,
                "explicit_scalar_multiplications_per_token_upper_bound": 0,
                "zero_denominator_comparisons_per_token_upper_bound": 1,
                "negative_balance_comparisons_per_token_upper_bound": 1,
                "parent_decoder_invocations_per_token": 1,
            },
        ),
        (
            route.CENTERED_EW_OCCUPANCY,
            route.GEMMA_ITERATIVE_EW_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE,
            "jacobian_by_ew_occupancy_conformal_coefficient",
            {
                "learned_parameter_count": 6,
                "logical_macs_per_token_upper_bound": 10,
                "prepared_float_scalar_count": 9,
                "derived_constant_float_count": 2,
                "fixed_decay_float_scalar_count": 1,
                "runtime_state_float_count_per_sequence": 4,
                "nonlinear_scalar_ops_per_token_upper_bound": 6,
                "linear_accumulator_scalar_ops_per_token_upper_bound": 8,
                "explicit_scalar_multiplications_per_token_upper_bound": 2,
                "zero_denominator_comparisons_per_token_upper_bound": 1,
                "negative_balance_comparisons_per_token_upper_bound": 1,
                "parent_decoder_invocations_per_token": 1,
            },
        ),
    ),
)
def test_campaign_recipe_binds_arm_and_exact_resource_envelope(
    occupancy_kind: route.OccupancyKind,
    recipe: object,
    jacobian_field: str,
    expected_resources: dict[str, int],
) -> None:
    provider = _provider(
        (0.03, -0.02, 0.01, 0.04, -0.03, 0.02),
        occupancy_kind=occupancy_kind,
    )
    assert recipe.fit_record_jacobian_field == jacobian_field
    assert recipe.coefficient_count == 6
    resources = recipe.provider_resource_receipt(provider)
    assert resources == expected_resources
    recipe.validate_resource_envelope(
        resources=resources,
        residual_width=provider.width,
    )

    other_recipe = (
        route.GEMMA_ITERATIVE_EW_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE
        if occupancy_kind == route.CENTERED_CUMULATIVE_OCCUPANCY
        else route.GEMMA_ITERATIVE_CUMULATIVE_OCCUPANCY_ROUTE_CAMPAIGN_RECIPE
    )
    with pytest.raises(
        RuntimeError,
        match="fixed occupancy conformal route exceeds its resource envelope",
    ):
        other_recipe.provider_resource_receipt(provider)
