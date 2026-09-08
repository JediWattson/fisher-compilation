from __future__ import annotations

import math

import pytest
import torch

import fisher_graph.fisher_response_geometry_failure_diagnosis as diagnosis
from fisher_graph.fisher_response_geometry_intercept_compatibility_evaluation import (
    comparison_masks,
    oracle_lower_bound,
    score_with_abstentions,
)


DTYPE = torch.float64


def _tensor(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=DTYPE)


def _mask_fixture():
    row_ids = [f"unit-row-{i}" for i in range(48)]
    legacy = row_ids[:38]
    eligible = {
        "activation_charts": torch.ones((48, 2), dtype=torch.bool),
        "response_charts": torch.ones((48, 2), dtype=torch.bool),
        "intercept_compatible_response_charts": torch.ones((48, 2), dtype=torch.bool),
    }
    eligible["response_charts"][3] = torch.tensor([True, False])
    candidate = eligible["intercept_compatible_response_charts"]
    candidate[4] = torch.tensor([False, True])
    candidate[6] = candidate[20] = torch.tensor([False, False])
    support = {}
    for family, values in eligible.items():
        direct = values.long().argmax(dim=1)
        oracle = direct.masked_fill(~values.any(dim=1), -1)
        runtime = direct.clone()
        if family == "response_charts":
            runtime[3] = 1
        elif family == "intercept_compatible_response_charts":
            runtime[4] = 0
        support[family] = {
            "runtime_router": diagnosis.support_categories(values, runtime),
            "direct_discovery": diagnosis.support_categories(values, direct),
            "best_supported_oracle": diagnosis.support_categories(values, oracle),
        }
    return row_ids, legacy, eligible, support


def test_three_family_primary_mask_uses_availability_instead_of_successful_selection() -> None:
    row_ids, legacy, eligible, support = _mask_fixture()
    masks = comparison_masks(eligible, support, row_ids, legacy)
    assert int(masks["all_rows"].sum()) == 48
    assert int(masks["common_availability"].sum()) == 46
    assert int(masks["common_selected_support"].sum()) == 44
    assert masks["common_availability"][3].item() is True
    assert masks["common_availability"][4].item() is True
    assert masks["common_selected_support"][3].item() is False
    assert masks["common_selected_support"][4].item() is False
    assert not bool(masks["common_availability"][[6, 20]].any())
    assert torch.equal(masks["legacy_38"], torch.arange(48) < 38)
    assert masks["legacy_38"][6].item() is True
    assert int(masks["legacy_common_availability"].sum()) == 36


def test_legacy_mask_follows_saved_row_identity_when_rows_are_reordered() -> None:
    row_ids, legacy, eligible, support = _mask_fixture()
    permutation = torch.arange(47, -1, -1)
    reordered_ids = [row_ids[i] for i in permutation.tolist()]
    reordered_eligible = {family: values[permutation] for family, values in eligible.items()}
    reordered_support = {
        family: {
            method: {
                "supported_selection_mask": torch.tensor(category["supported_selection_mask"])[permutation].tolist()
            }
            for method, category in methods.items()
        }
        for family, methods in support.items()
    }
    masks = comparison_masks(reordered_eligible, reordered_support, reordered_ids, legacy)
    assert [row for row, included in zip(reordered_ids, masks["legacy_38"]) if included] == list(reversed(legacy))
    assert int(masks["legacy_38"].sum()) == 38
    assert not bool(masks["legacy_38"][:10].any())


def test_legacy_mask_rejects_missing_or_duplicated_identities() -> None:
    row_ids, legacy, eligible, support = _mask_fixture()
    with pytest.raises(ValueError):
        comparison_masks(eligible, support, row_ids, [*legacy[:-1], "missing-row"])
    with pytest.raises(ValueError):
        comparison_masks(eligible, support, [row_ids[0], *row_ids[:-1]], legacy)
    with pytest.raises(ValueError):
        comparison_masks(eligible, support, row_ids, [legacy[0], *legacy[:-1]])


def test_full_mask_abstention_keeps_population_and_denominators_but_marks_error_missing() -> None:
    target = _tensor([[[1.0, 2.0]], [[10.0, 20.0]], [[3.0, 4.0]]])
    predictions = torch.stack((target + 1.0, target - 1.0), dim=1)
    factor = torch.diag(_tensor([2.0, 1.0]))
    assignments = torch.tensor([0, -1, 1])
    fixed_mask = torch.ones(3, dtype=torch.bool)
    score = score_with_abstentions(diagnosis, predictions, target, factor, assignments, fixed_mask)
    assert score["measurement_status"] == "unavailable_oracle_abstention_in_fixed_mask"
    assert score["row_mask"] == [True, True, True]
    assert score["row_count"] == score["requested_row_count"] == 3
    assert score["evaluated_row_count"] == 0
    assert score["available_prediction_row_count"] == 2
    assert score["abstaining_row_indices"] == [1]
    assert score["fisher_target_denominator"] == pytest.approx(860.0)
    assert score["euclidean_target_denominator"] == pytest.approx(530.0)
    for key in ("fisher_error_numerator", "fisher_relative_l2", "euclidean_error_numerator", "euclidean_relative_l2", "euclidean_rmse"):
        assert score[key] is None

    reduced = score_with_abstentions(diagnosis, predictions, target, factor, assignments, torch.tensor([True, False, True]))
    assert reduced["measurement_status"] == "measured"
    assert reduced["row_count"] == reduced["evaluated_row_count"] == 2
    assert reduced["fisher_target_denominator"] == pytest.approx(60.0)
    assert reduced["euclidean_target_denominator"] == pytest.approx(30.0)
    assert reduced["fisher_relative_l2"] == pytest.approx(math.sqrt(10.0 / 60.0))
    assert reduced["abstaining_row_indices"] == []


def test_abstention_outside_requested_mask_does_not_change_existing_score() -> None:
    target = _tensor([[[1.0]], [[10.0]], [[3.0]]])
    predictions = torch.stack((target + 1.0, target - 1.0), dim=1)
    factor = _tensor([[2.0]])
    assignments, mask = torch.tensor([0, -1, 1]), torch.tensor([True, False, True])
    expected = diagnosis.score_assignments(predictions, target, factor, assignments, mask)
    actual = score_with_abstentions(diagnosis, predictions, target, factor, assignments, mask)
    assert all(actual[key] == value for key, value in expected.items())
    assert actual["evaluated_row_count"] == 2
    assert actual["available_prediction_row_count"] == 2


def test_empty_mask_retains_unavailable_scores_instead_of_inventing_zero_loss() -> None:
    target = _tensor([[[1.0]]])
    predictions = _tensor([[[[0.0]]]])
    actual = score_with_abstentions(diagnosis, predictions, target, _tensor([[1.0]]), torch.tensor([-1]), torch.tensor([False]))
    assert actual["row_count"] == actual["requested_row_count"] == actual["evaluated_row_count"] == 0
    assert actual["fisher_relative_l2"] is None
    assert actual["euclidean_relative_l2"] is None
    assert actual["fisher_target_denominator"] is None


def _lower_bound_fixture():
    target = _tensor([[[1.0]], [[1.0]], [[1.0]]])
    predictions = _tensor([[[[3.0]], [[1.0]]], [[[1.0]], [[4.0]]], [[[0.0]], [[0.0]]]])
    eligible = torch.tensor([[True, True], [False, True], [False, False]])
    assignments = {
        "best_supported_oracle": torch.tensor([1, 1, -1]),
        "runtime_router": torch.tensor([0, 0, 0]),
        "direct_discovery": torch.tensor([1, 1, 0]),
    }
    return predictions, target, _tensor([[1.0]]), eligible, assignments, torch.tensor([True, True, False])


def test_oracle_lower_bound_excludes_unsupported_comparators_with_explicit_shared_masks() -> None:
    predictions, target, factor, eligible, assignments, mask = _lower_bound_fixture()
    result = oracle_lower_bound(diagnosis, predictions, target, factor, eligible, assignments, mask)
    assert result["status"] == "passed"
    checks = {row["comparator"]: row for row in result["checks"]}
    runtime = checks["runtime_router"]
    assert runtime["row_mask"] == [True, False, False]
    assert runtime["row_count"] == 1
    assert runtime["oracle_fisher_error_numerator"] == 0.0
    assert runtime["comparator_fisher_error_numerator"] == 4.0
    assert runtime["fisher_target_denominator"] == 1.0
    direct = checks["direct_discovery"]
    assert direct["row_mask"] == [True, True, False]
    assert direct["oracle_fisher_error_numerator"] == direct["comparator_fisher_error_numerator"] == 9.0
    assert direct["fisher_target_denominator"] == 2.0


def test_oracle_lower_bound_detects_wrong_eligible_oracle_prediction() -> None:
    predictions, target, factor, eligible, assignments, mask = _lower_bound_fixture()
    assignments["best_supported_oracle"] = torch.tensor([0, 1, -1])
    result = oracle_lower_bound(diagnosis, predictions, target, factor, eligible, assignments, mask)
    assert result["status"] == "failed"
    failure = next(row for row in result["checks"] if row["comparator"] == "direct_discovery")
    assert failure["passed"] is False
    assert failure["maximum_per_row_violation"] == 4.0


def test_oracle_lower_bound_rejects_ineligible_selection_or_unnecessary_abstention() -> None:
    predictions, target, factor, eligible, assignments, mask = _lower_bound_fixture()
    for invalid in (torch.tensor([1, 0, -1]), torch.tensor([-1, 1, -1])):
        assignments["best_supported_oracle"] = invalid
        with pytest.raises(ValueError):
            oracle_lower_bound(diagnosis, predictions, target, factor, eligible, assignments, mask)


def test_oracle_lower_bound_is_unavailable_when_no_eligible_comparator_rows_exist() -> None:
    predictions, target, factor, eligible, assignments, _mask = _lower_bound_fixture()
    result = oracle_lower_bound(diagnosis, predictions, target, factor, eligible, assignments, torch.zeros(3, dtype=torch.bool))
    assert result["status"] == "unavailable"
    assert all(row["row_count"] == 0 and row["passed"] is None for row in result["checks"])
