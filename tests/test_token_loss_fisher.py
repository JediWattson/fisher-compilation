from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
import torch

from fisher_graph.token_loss_fisher import (
    COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
    CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES,
    TokenLossFisherGateConfig,
    analyze_cumulative_occupancy_token_loss_fisher_lofo,
    analyze_ew_occupancy_token_loss_fisher_lofo,
    analyze_token_loss_fisher_lofo,
    build_token_loss_fisher_prompt_record,
    fit_token_loss_fisher_fold,
    token_loss_fisher_prompt_record_from_dict,
)


def _record(
    example: str,
    family: str,
    scores: torch.Tensor,
    target: torch.Tensor,
    *,
    names: tuple[str, ...] | None = None,
):
    if names is None:
        names = tuple(f"coordinate-{index}" for index in range(scores.shape[1]))
    return build_token_loss_fisher_prompt_record(
        example_id=example,
        family_id=family,
        coordinate_names=names,
        token_scores=scores,
        compensation_target=target,
    )


def _stable_records(
    *,
    coordinate_count: int = 2,
    family_count: int = 4,
):
    scores = torch.eye(coordinate_count, dtype=torch.float64)
    coefficients = torch.arange(
        1, coordinate_count + 1, dtype=torch.float64
    )
    target = scores @ coefficients
    return tuple(
        _record(
            f"example-{index}",
            f"family-{index}",
            scores,
            target,
        )
        for index in range(family_count)
    )


def _assert_no_tensor(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_tensor(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_tensor(item)


def test_prompt_record_contains_exact_scalar_sufficient_statistics() -> None:
    scores = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    target = torch.tensor([2.0, -1.0, 4.0], dtype=torch.float32)
    record = _record("example", "family", scores, target)

    expected_scores = scores.to(torch.float64)
    expected_target = target.to(torch.float64)
    expected_fisher = expected_scores.T @ expected_scores / 3
    expected_cross = expected_scores.T @ expected_target / 3

    assert record.supervised_tokens == 3
    assert torch.allclose(
        torch.tensor(record.fisher_second_moment, dtype=torch.float64),
        expected_fisher,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        torch.tensor(record.target_cross_moment, dtype=torch.float64),
        expected_cross,
        atol=0.0,
        rtol=0.0,
    )
    assert record.target_second_moment == pytest.approx(
        float(expected_target.square().mean())
    )
    assert record.mean_score == pytest.approx(
        tuple(float(value) for value in expected_scores.mean(dim=0))
    )
    payload = record.to_dict()
    assert "token_scores" not in payload
    assert "compensation_target" not in payload
    _assert_no_tensor(payload)


def test_prompt_record_rejects_bad_geometry_nonfinite_and_coordinates() -> None:
    scores = torch.ones((3, 2), dtype=torch.float64)
    target = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="geometry"):
        _record(
            "example",
            "family",
            scores,
            target[:2],
        )
    with pytest.raises(ValueError, match="finite"):
        _record(
            "example",
            "family",
            scores.clone().index_put_(
                (torch.tensor([0]), torch.tensor([0])),
                torch.tensor([math.nan], dtype=torch.float64),
            ),
            target,
        )
    with pytest.raises(ValueError, match="unique"):
        _record(
            "example",
            "family",
            scores,
            target,
            names=("same", "same"),
        )


def test_prompt_record_round_trip_and_receipt_fail_closed() -> None:
    record = _record(
        "example",
        "family",
        torch.eye(2, dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    replay = token_loss_fisher_prompt_record_from_dict(record.to_dict())
    assert replay == record

    forged = record.to_dict()
    forged["target_second_moment"] = record.target_second_moment + 0.25
    with pytest.raises(ValueError, match="hash mismatch"):
        token_loss_fisher_prompt_record_from_dict(forged)

    with pytest.raises(FrozenInstanceError):
        record.family_id = "other"  # type: ignore[misc]


def test_fold_fit_is_ridge_free_exact_and_excludes_held_family() -> None:
    records = _stable_records()
    fold = fit_token_loss_fisher_fold(
        records,
        held_family_id="family-0",
    )

    assert fold.coefficients == pytest.approx((1.0, 2.0), abs=1.0e-12)
    assert fold.raw_normal_rank == 2
    assert fold.standardized_normal_rank == 2
    assert fold.standardized_positive_spectrum_condition_number == pytest.approx(
        1.0
    )
    assert fold.train_rmse_after == pytest.approx(0.0, abs=1.0e-7)
    assert fold.held_rmse_after == pytest.approx(0.0, abs=1.0e-7)
    assert fold.held_relative_rmse_improvement == pytest.approx(
        1.0, abs=1.0e-7
    )
    assert "family-0" not in fold.train_family_ids
    assert fold.held_example_ids == ("example-0",)
    assert "example-0" not in fold.train_example_ids


def test_training_moments_balance_families_then_prompts() -> None:
    records = (
        _record(
            "a-0",
            "family-a",
            torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        ),
        _record(
            "a-1",
            "family-a",
            torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        ),
        _record(
            "b-0",
            "family-b",
            torch.tensor([[2.0]], dtype=torch.float64),
            torch.tensor([4.0], dtype=torch.float64),
        ),
        _record(
            "c-0",
            "family-c",
            torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        ),
    )
    fold = fit_token_loss_fisher_fold(
        records,
        held_family_id="family-c",
    )

    # Family A contributes A=1,b=1 despite having two prompts. Family B
    # contributes A=4,b=8. Equal family mass gives beta=(1+8)/(1+4)=1.8.
    assert fold.coefficients == pytest.approx((1.8,), abs=1.0e-12)


def test_lofo_report_has_exact_metrics_stability_gates_and_no_tensors() -> None:
    report = analyze_token_loss_fisher_lofo(_stable_records())

    assert report.family_ids == (
        "family-0",
        "family-1",
        "family-2",
        "family-3",
    )
    assert report.family_macro_rmse_after == pytest.approx(0.0, abs=1.0e-7)
    assert report.family_macro_relative_rmse_improvement == pytest.approx(
        1.0, abs=1.0e-7
    )
    assert report.family_win_count == 4
    assert report.minimum_family_win_count == 3
    assert report.mean_pairwise_fold_coefficient_cosine == pytest.approx(1.0)
    assert report.passed is True
    assert all(value for _name, value in report.gate_results)
    report.validate_integrity()
    _assert_no_tensor(report.to_dict())


def test_lofo_is_canonical_under_record_reordering() -> None:
    records = _stable_records()
    forward = analyze_token_loss_fisher_lofo(records)
    reverse = analyze_token_loss_fisher_lofo(tuple(reversed(records)))
    assert reverse.report_sha256 == forward.report_sha256
    assert reverse.to_dict() == forward.to_dict()


def test_rank_deficiency_fails_closed_without_ridge() -> None:
    scores = torch.tensor(
        [[1.0, 1.0], [-1.0, -1.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([1.0, -1.0], dtype=torch.float64)
    records = tuple(
        _record(f"example-{index}", f"family-{index}", scores, target)
        for index in range(4)
    )
    report = analyze_token_loss_fisher_lofo(records)
    gates = dict(report.gate_results)

    assert all(fold.raw_normal_rank == 1 for fold in report.folds)
    assert gates["all_fold_raw_normal_ranks_full"] is False
    assert gates["all_fold_standardized_normal_ranks_full"] is False
    assert report.passed is False


def test_combined_occupancy_views_are_fixed_and_share_one_record() -> None:
    scores = torch.eye(8, dtype=torch.float64)
    coefficients = torch.arange(1, 9, dtype=torch.float64)
    target = scores @ coefficients
    records = tuple(
        _record(
            f"example-{index}",
            f"family-{index}",
            scores,
            target,
            names=COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
        )
        for index in range(4)
    )

    cumulative = analyze_cumulative_occupancy_token_loss_fisher_lofo(records)
    ew = analyze_ew_occupancy_token_loss_fisher_lofo(records)

    assert (
        cumulative.coordinate_indices
        == CUMULATIVE_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES
    )
    assert ew.coordinate_indices == EW_OCCUPANCY_TOKEN_FISHER_COORDINATE_INDICES
    assert cumulative.coordinate_names[-2:] == (
        "cumulative_occupancy_contrast_real",
        "cumulative_occupancy_contrast_imag",
    )
    assert ew.coordinate_names[-2:] == (
        "ew_occupancy_contrast_real",
        "ew_occupancy_contrast_imag",
    )
    assert cumulative.folds[0].coefficients == pytest.approx(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    )
    assert ew.folds[0].coefficients == pytest.approx(
        (1.0, 2.0, 3.0, 4.0, 7.0, 8.0)
    )
    assert cumulative.minimum_fold_incremental_energy_fraction == pytest.approx(
        1.0
    )
    assert ew.minimum_fold_incremental_energy_fraction == pytest.approx(1.0)


def test_incremental_schur_energy_rejects_duplicated_occupancy() -> None:
    base = torch.eye(4, dtype=torch.float64)
    scores = torch.cat(
        (
            base,
            base[:, :2],
            base[:, 2:],
        ),
        dim=1,
    )
    target = base @ torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )
    records = tuple(
        _record(
            f"example-{index}",
            f"family-{index}",
            scores,
            target,
            names=COMBINED_OCCUPANCY_TOKEN_FISHER_COORDINATE_NAMES,
        )
        for index in range(4)
    )
    report = analyze_cumulative_occupancy_token_loss_fisher_lofo(records)
    gates = dict(report.gate_results)

    assert report.minimum_fold_incremental_energy_fraction == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert (
        gates[
            "all_fold_incremental_energy_fractions_at_least_minimum"
        ]
        is False
    )
    assert report.passed is False


def test_report_gate_thresholds_are_frozen_and_validated() -> None:
    with pytest.raises(ValueError, match="configuration"):
        TokenLossFisherGateConfig(
            minimum_family_win_fraction=1.1,
        )
    config = TokenLossFisherGateConfig(
        minimum_family_macro_relative_rmse_improvement=1.0,
    )
    report = analyze_token_loss_fisher_lofo(
        _stable_records(),
        gate_config=config,
    )
    assert dict(report.gate_results)[
        "family_macro_relative_rmse_improvement_at_least_minimum"
    ]


def test_coordinate_views_reject_duplicates_and_missing_held_family() -> None:
    records = _stable_records()
    with pytest.raises(ValueError, match="duplicate"):
        analyze_token_loss_fisher_lofo(records, coordinate_indices=(0, 0))
    with pytest.raises(ValueError, match="absent"):
        fit_token_loss_fisher_fold(
            records,
            held_family_id="not-a-family",
        )
