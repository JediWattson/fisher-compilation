from __future__ import annotations

import math

import pytest
import torch

import fisher_graph.fisher_response_geometry as geometry
from fisher_graph.fisher_response_geometry import (
    AffineResponseChart,
    jvp_actions,
    normalization_update,
)
from fisher_graph.fisher_response_geometry_failure_diagnosis import (
    compact_intercept,
    endpoint_eligibility,
    frozen_predictions,
    run_diagnosis,
    score_assignments,
    support_categories,
    supported_oracle,
)


DTYPE = torch.float64


def _tensor(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=DTYPE)


def _chart(
    *,
    slope: float = 1.0,
    beta: float = 0.0,
    message_anchor: float = 0.0,
    message_radius: float = 10.0,
) -> AffineResponseChart:
    return AffineResponseChart(
        state_anchor=_tensor([0.0]),
        message_anchor=_tensor([message_anchor]),
        beta=_tensor([beta]),
        B=_tensor([[0.0]]),
        A=_tensor([[slope]]),
        U=_tensor([[1.0]]),
        Q=_tensor([[1.0]]),
        G=_tensor([[1.0]]),
        state_radius=10.0,
        message_radius=message_radius,
    )


def test_oracle_selects_one_chart_by_mean_probe_loss_not_one_chart_per_probe() -> None:
    target = _tensor([[[1.0], [1.0]]])
    predictions = _tensor([[[[1.0], [11.0]], [[7.0], [7.0]]]])
    eligible = torch.tensor([[True, True]])
    assignment = supported_oracle(predictions, target, _tensor([[1.0]]), eligible)
    assert assignment.tolist() == [1]
    scores = score_assignments(predictions, target, _tensor([[1.0]]), assignment, torch.tensor([True]))
    assert scores["fisher_error_numerator"] == pytest.approx(72.0)
    assert scores["fisher_target_denominator"] == pytest.approx(2.0)
    assert scores["fisher_relative_l2"] == pytest.approx(6.0)
    # Coordinate/probe-wise selection would report a different, unattainable loss.
    per_probe_cheating = (predictions - target[:, None]).square().sum(dim=-1).min(dim=1).values.sum()
    assert per_probe_cheating.item() == 36.0
    assert scores["fisher_error_numerator"] > per_probe_cheating.item()


def test_oracle_cannot_combine_different_charts_output_coordinates() -> None:
    target = _tensor([[[1.0, 1.0]]])
    predictions = _tensor([[[[1.0, 4.0]], [[3.0, 1.0]]]])
    factor = torch.eye(2, dtype=DTYPE)
    assignment = supported_oracle(predictions, target, factor, torch.tensor([[True, True]]))
    assert assignment.tolist() == [1]
    scores = score_assignments(predictions, target, factor, assignment, torch.tensor([True]))
    assert scores["fisher_error_numerator"] == pytest.approx(4.0)
    assert scores["fisher_relative_l2"] == pytest.approx(math.sqrt(2.0))


def test_fisher_oracle_assignments_also_determine_reported_euclidean_error() -> None:
    target = _tensor([[[1.0, 1.0]]])
    predictions = _tensor([[[[3.0, 1.0]], [[1.0, 2.0]]]])
    factor = torch.diag(_tensor([0.1, 10.0]))
    assignment = supported_oracle(predictions, target, factor, torch.tensor([[True, True]]))
    assert assignment.tolist() == [0]
    scores = score_assignments(predictions, target, factor, assignment, torch.tensor([True]))
    assert scores["fisher_error_numerator"] == pytest.approx(0.04)
    assert scores["fisher_target_denominator"] == pytest.approx(100.01)
    assert scores["euclidean_error_numerator"] == pytest.approx(4.0)
    assert scores["euclidean_target_denominator"] == pytest.approx(2.0)
    assert scores["euclidean_relative_l2"] == pytest.approx(math.sqrt(2.0))
    euclidean_better_assignment = torch.tensor([1])
    alternative = score_assignments(predictions, target, factor, euclidean_better_assignment, torch.tensor([True]))
    assert alternative["euclidean_relative_l2"] < scores["euclidean_relative_l2"]
    assert alternative["fisher_relative_l2"] > scores["fisher_relative_l2"]


def test_each_prediction_target_has_an_independent_oracle_assignment() -> None:
    charts = [_chart(slope=1.0, beta=100.0), _chart(slope=2.0, beta=0.0)]
    h, m, probes = _tensor([[0.0]]), _tensor([[1.0]]), _tensor([[[1.0]]])
    eligible = torch.tensor([[True, True]])
    factor = _tensor([[1.0]])
    derivative = frozen_predictions(charts, h, m, probes, "derivative_action")
    residual = frozen_predictions(charts, h, m, None, "residual_update")
    derivative_choice = supported_oracle(derivative, _tensor([[[1.0]]]), factor, eligible)
    residual_choice = supported_oracle(residual, _tensor([[[2.0]]]), factor, eligible)
    assert derivative_choice.tolist() == [0]
    assert residual_choice.tolist() == [1]


def test_supported_oracle_excludes_ineligible_perfect_predictions_and_breaks_ties_by_chart_id() -> None:
    target = _tensor([[[1.0]], [[1.0]], [[1.0]]])
    predictions = _tensor([
        [[[1.0]], [[3.0]], [[3.0]]],
        [[[3.0]], [[3.0]], [[3.0]]],
        [[[1.0]], [[1.0]], [[1.0]]],
    ])
    eligible = torch.tensor([[False, True, True], [True, True, True], [False, False, False]])
    assignments = supported_oracle(predictions, target, _tensor([[1.0]]), eligible)
    assert assignments.tolist() == [1, 0, -1]
    scored_mask = torch.tensor([True, True, False])
    scores = score_assignments(predictions, target, _tensor([[1.0]]), assignments, scored_mask)
    assert scores["row_mask"] == [True, True, False]
    assert scores["row_count"] == 2
    assert scores["fisher_error_numerator"] == pytest.approx(8.0)
    with pytest.raises(ValueError):
        score_assignments(predictions, target, _tensor([[1.0]]), assignments, torch.ones(3, dtype=torch.bool))


def test_empty_evaluation_mask_reports_missing_errors_instead_of_zero_error() -> None:
    predictions = _tensor([[[[1.0]]]])
    target = _tensor([[[1.0]]])
    scores = score_assignments(predictions, target, _tensor([[1.0]]), torch.tensor([-1]), torch.tensor([False]))
    assert scores["row_mask"] == [False]
    assert scores["row_count"] == 0
    for key in (
        "fisher_error_numerator", "fisher_target_denominator", "fisher_relative_l2",
        "euclidean_error_numerator", "euclidean_target_denominator", "euclidean_relative_l2",
    ):
        assert scores[key] is None


def test_common_mask_preserves_exact_rows_and_denominators_across_methods_and_families() -> None:
    target = _tensor([[[1.0, 2.0]], [[100.0, 200.0]], [[3.0, 4.0]]])
    factor = torch.diag(_tensor([2.0, 1.0]))
    predictions_a = torch.stack((target + 1.0, target - 1.0), dim=1)
    predictions_b = torch.stack((target + 2.0, target - 2.0), dim=1)
    common_mask = torch.tensor([True, False, True])
    first = score_assignments(predictions_a, target, factor, torch.tensor([0, -1, 1]), common_mask)
    second = score_assignments(predictions_b, target, factor, torch.tensor([1, 0, 0]), common_mask)
    for scores in (first, second):
        assert scores["row_mask"] == [True, False, True]
        assert scores["row_count"] == 2
        assert scores["fisher_target_denominator"] == pytest.approx(60.0)
        assert scores["euclidean_target_denominator"] == pytest.approx(30.0)
    assert first["fisher_error_numerator"] == pytest.approx(10.0)
    assert second["fisher_error_numerator"] == pytest.approx(40.0)
    assert first["fisher_relative_l2"] == pytest.approx(math.sqrt(10.0 / 60.0))


def test_support_categories_separate_no_predictor_from_unsupported_selection() -> None:
    eligible = torch.tensor([[False, False], [True, False], [False, True], [True, True]])
    assignments = torch.tensor([-1, 1, 1, 0])
    categories = support_categories(eligible, assignments)
    assert categories["no_eligible_mask"] == [True, False, False, False]
    assert categories["unsupported_selection_mask"] == [False, True, False, False]
    assert categories["supported_selection_mask"] == [False, False, True, True]
    assert categories["no_eligible_count"] == 1
    assert categories["unsupported_selection_count"] == 1
    assert categories["supported_selection_count"] == 2
    assert sum(categories[name] for name in ("no_eligible_count", "unsupported_selection_count", "supported_selection_count")) == 4
    # A runtime prediction can exist mathematically even when no chart supports it.
    extrapolated = support_categories(eligible, torch.tensor([0, 1, 1, 0]))
    assert extrapolated["no_eligible_mask"] == categories["no_eligible_mask"]
    with pytest.raises(ValueError):
        support_categories(eligible, torch.tensor([-1, -1, 1, 0]))


def test_finite_eligibility_requires_one_chart_for_base_and_every_probe_endpoint() -> None:
    charts = [
        _chart(message_anchor=0.0, message_radius=0.75),
        _chart(message_anchor=1.0, message_radius=0.75),
    ]
    h, m = _tensor([[0.0]]), _tensor([[0.5]])
    assert endpoint_eligibility(charts, h, m).tolist() == [[True, True]]
    probes = _tensor([[[-0.5], [0.5]]])
    # Either endpoint can be supported by some chart, but neither chart supports both.
    assert endpoint_eligibility(charts, h, m, probes).tolist() == [[False, False]]
    assert endpoint_eligibility(charts, h, m, probes[:, :1]).tolist() == [[True, False]]
    assert endpoint_eligibility(charts, h, m, probes[:, 1:]).tolist() == [[False, True]]


def test_finite_eligibility_does_not_accept_an_endpoint_when_base_is_unsupported() -> None:
    chart = _chart(message_anchor=1.0, message_radius=0.25)
    h, m = _tensor([[0.0]]), _tensor([[0.0]])
    assert endpoint_eligibility([chart], h, m, _tensor([[[1.0]]])).tolist() == [[False]]


def test_support_checks_full_inputs_including_omitted_state_and_message_directions() -> None:
    axis = _tensor([[1.0], [0.0]])
    chart = AffineResponseChart(
        state_anchor=_tensor([0.0, 0.0]), message_anchor=_tensor([0.0, 0.0]),
        beta=_tensor([0.0]), B=_tensor([[0.0]]), A=_tensor([[1.0]]),
        U=axis, Q=axis, G=axis, state_radius=0.5, message_radius=0.5,
    )
    h = _tensor([[0.0, 2.0], [0.0, 0.0]])
    m = _tensor([[0.0, 0.0], [0.0, 2.0]])
    assert endpoint_eligibility([chart], h, m).tolist() == [[False], [False]]
    assert endpoint_eligibility([chart], h[1:], m[:1], _tensor([[[0.0, 2.0]]])).tolist() == [[False]]


def test_frozen_predictions_use_stored_compact_operators_and_one_chart_at_both_endpoints() -> None:
    charts = [_chart(slope=2.0, beta=100.0), _chart(slope=-3.0, beta=-200.0)]
    h, m, probes = _tensor([[0.0]]), _tensor([[0.25]]), _tensor([[[0.5], [-0.2]]])
    expected_changes = _tensor([[[[1.0], [-0.4]], [[-1.5], [0.6]]]])
    for target in ("derivative_action", "finite_response"):
        observed = frozen_predictions(charts, h, m, probes, target)
        torch.testing.assert_close(observed, expected_changes, rtol=1e-12, atol=1e-12)
    absolute = frozen_predictions(charts, h, m, None, "residual_update")
    torch.testing.assert_close(absolute, _tensor([[[[100.5]], [[-200.75]]]]))
    # These arbitrary stored slopes deliberately disagree with any normalization teacher.
    assert absolute.shape == (1, 2, 1, 1)


def test_frozen_derivative_predictions_retain_input_and_output_projection_losses() -> None:
    axis = _tensor([[1.0], [0.0]])
    chart = AffineResponseChart(
        state_anchor=_tensor([0.0, 0.0]), message_anchor=_tensor([0.0, 0.0]),
        beta=_tensor([50.0]), B=_tensor([[0.0]]), A=_tensor([[3.0]]),
        U=axis, Q=axis, G=axis, state_radius=10.0, message_radius=10.0,
    )
    probes = _tensor([[[0.0, 1.0], [1.0, 0.0]]])
    actual = frozen_predictions([chart], _tensor([[0.0, 0.0]]), _tensor([[0.0, 0.0]]), probes, "derivative_action")
    torch.testing.assert_close(actual, _tensor([[[[0.0, 0.0], [3.0, 0.0]]]]))


def test_compact_intercept_is_the_stored_affine_intercept_and_rejects_state_dependence() -> None:
    axis = _tensor([[1.0], [0.0]])
    chart = AffineResponseChart(
        state_anchor=_tensor([0.3, -0.2]), message_anchor=_tensor([5.0, 7.0]),
        beta=_tensor([2.0]), B=_tensor([[0.0]]), A=_tensor([[3.0]]),
        U=axis, Q=axis, G=axis, state_radius=10.0, message_radius=10.0,
    )
    intercept = compact_intercept(chart)
    torch.testing.assert_close(intercept, _tensor([-13.0, 0.0]))
    h, m = _tensor([0.8, -0.4]), _tensor([1.1, -2.0])
    torch.testing.assert_close(chart.predict(h, m), intercept + chart.G @ chart.A @ chart.Q.T @ m)
    identity = torch.eye(2, dtype=DTYPE)
    teacher = lambda message: normalization_update(message, identity, _tensor([0.0, 0.0]), 0.01)
    R = jvp_actions(teacher, chart.message_anchor, identity)
    exact_intercept = teacher(chart.message_anchor) - R @ chart.message_anchor
    assert not torch.allclose(intercept, exact_intercept)
    state_dependent = AffineResponseChart(
        state_anchor=chart.state_anchor, message_anchor=chart.message_anchor,
        beta=chart.beta, B=_tensor([[0.5]]), A=chart.A,
        U=chart.U, Q=chart.Q, G=chart.G,
        state_radius=10.0, message_radius=10.0,
    )
    with pytest.raises(ValueError):
        compact_intercept(state_dependent)
    with pytest.raises(ValueError):
        frozen_predictions([state_dependent], h[None], m[None], None, "residual_update")


def test_normalization_sign_symmetry_preserves_response_but_reverses_exact_and_compact_intercepts() -> None:
    identity = torch.eye(2, dtype=DTYPE)
    projection = _tensor([[1.1, 0.2], [-0.3, 0.8]])
    weight = _tensor([0.2, -0.1])
    message = _tensor([0.7, -1.2])
    teacher = lambda m: normalization_update(m, projection, weight, 0.01)
    R_plus = jvp_actions(teacher, message, identity)
    R_minus = jvp_actions(teacher, -message, identity)
    torch.testing.assert_close(R_plus, R_minus, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(teacher(-message), -teacher(message), rtol=1e-12, atol=1e-12)
    exact_plus = teacher(message) - R_plus @ message
    exact_minus = teacher(-message) - R_minus @ (-message)
    torch.testing.assert_close(exact_minus, -exact_plus, rtol=1e-12, atol=1e-12)
    assert torch.linalg.vector_norm(exact_plus).item() > 0.5
    charts = [
        AffineResponseChart(
            state_anchor=torch.zeros(2, dtype=DTYPE), message_anchor=m,
            beta=teacher(m), B=torch.zeros((2, 2), dtype=DTYPE), A=R,
            U=identity, Q=identity, G=identity,
            state_radius=10.0, message_radius=10.0,
        )
        for m, R in ((message, R_plus), (-message, R_minus))
    ]
    torch.testing.assert_close(compact_intercept(charts[0]), exact_plus)
    torch.testing.assert_close(compact_intercept(charts[1]), exact_minus)


def test_complete_diagnostic_panels_keep_common_masks_and_explicit_unsupported_routing() -> None:
    h, m = _tensor([[0.0], [0.0], [0.0]]), _tensor([[0.0], [1.0], [3.0]])
    response_bank = [
        _chart(slope=2.0, message_anchor=0.0, message_radius=0.6),
        _chart(slope=0.2, beta=0.5, message_anchor=1.0, message_radius=0.6),
    ]
    activation_bank = [
        _chart(slope=2.0, message_anchor=0.0, message_radius=10.0),
        _chart(slope=0.2, beta=0.5, message_anchor=1.0, message_radius=10.0),
    ]
    direct = {"response_charts": torch.tensor([0, 1, 0]), "activation_charts": torch.tensor([0, 1, 0])}
    runtime = {"response_charts": torch.tensor([1, 1, 0]), "activation_charts": torch.tensor([0, 1, 0])}
    identity = _tensor([[1.0]])
    replay = {
        "kernel": geometry,
        "tensors": {
            "W": identity, "norm_weight": _tensor([0.0]), "F_factor": identity,
            "G": identity, "Q": identity, "D_G": identity,
        },
        "config": {"epsilon": 0.01, "finite_radii": [0.1], "rows_per_family": 1},
        "train_h": h, "train_m": m,
        "eval_h": h, "eval_m": m,
        "eval_ids": ["unit-family-0", "unit-family-1", "unit-family-2"],
        "probe_directions": torch.ones((3, 1), dtype=DTYPE),
        "banks": {"response_charts": response_bank, "activation_charts": activation_bank},
        "train_direct_assignments": direct,
        "runtime_assignments": runtime, "direct_assignments": direct,
        "original_identity": {"scope": "generated_unit_fixture"},
        "reproduction": {"scope": "generated_unit_fixture"},
    }
    report = run_diagnosis(replay)
    assert len(report["target_panels"]) == 3
    for panel in report["target_panels"]:
        assert panel["masks"]["common_any_family_support"] == [True, True, False]
        assert panel["masks"]["common_selected_support"] == [False, True, False]
        primary = [row for row in panel["rows"] if row["panel"] == "common_any_family_support"]
        assert len(primary) == 6
        assert {tuple(row["row_mask"]) for row in primary} == {(True, True, False)}
        assert {row["row_count"] for row in primary} == {2}
        assert len({row["fisher_target_denominator"] for row in primary}) == 1
        assert len({row["euclidean_target_denominator"] for row in primary}) == 1
        assert panel["assignments"]["response_charts"]["best_supported_oracle"] == [0, 1, -1]
        response_runtime = next(row for row in primary if row["chart_family"] == "response_charts" and row["assignment_method"] == "runtime_router")
        assert response_runtime["unsupported_selected_rows_in_scoring_mask"] == 1
        assert response_runtime["full_fixture_support"]["no_eligible_count"] == 1
        assert response_runtime["full_fixture_support"]["unsupported_selection_count"] == 1
        assert response_runtime["full_fixture_support"]["supported_selection_count"] == 1
        all_rows = [row for row in panel["rows"] if row["panel"] == "all_rows_extrapolating_runtime_and_discovery"]
        assert len(all_rows) == 4
        assert all(row["row_count"] == 3 and row["row_mask"] == [True, True, True] for row in all_rows)
        assert all(row["assignment_method"] != "best_supported_oracle" for row in all_rows)
        for reference in panel["references"]:
            if reference["panel"] == "common_any_family_support":
                assert reference["row_mask"] == primary[0]["row_mask"]
                assert reference["fisher_target_denominator"] == primary[0]["fisher_target_denominator"]
                assert reference["euclidean_target_denominator"] == primary[0]["euclidean_target_denominator"]
    assert report["evidence_boundary"]["charts_refitted"] is False
    assert report["evidence_boundary"]["proposed_fix_implemented"] is False
