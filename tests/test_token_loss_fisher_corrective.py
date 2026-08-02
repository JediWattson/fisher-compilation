from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph import token_loss_fisher_corrective as corrective
from fisher_graph.token_loss_fisher import (
    build_token_loss_fisher_prompt_record,
)
from fisher_graph.token_loss_fisher_corrective import (
    build_token_loss_fisher_corrective_report,
    replay_token_loss_fisher_corrective_report,
    validate_token_loss_fisher_corrective_report,
)


_NAMES = (
    "shared_real",
    "shared_imag",
    "balance_real",
    "balance_imag",
    "occupancy_real",
    "occupancy_imag",
)


def _records(
    coefficients: tuple[float, float, float, float, float, float],
) -> tuple[object, ...]:
    generator = torch.Generator().manual_seed(17)
    coefficient_tensor = torch.tensor(coefficients, dtype=torch.float64)
    records = []
    for family in range(4):
        scores = torch.randn(
            48,
            6,
            generator=generator,
            dtype=torch.float64,
        )
        records.append(
            build_token_loss_fisher_prompt_record(
                example_id=f"example-{family}",
                family_id=f"family-{family}",
                coordinate_names=_NAMES,
                token_scores=scores,
                compensation_target=scores @ coefficient_tensor,
            )
        )
    return tuple(records)


def _resign_fold(fold: dict[str, object]) -> None:
    payload = dict(fold)
    payload.pop("fold_sha256")
    fold["fold_sha256"] = corrective._sha256(
        corrective._FOLD_DOMAIN, payload
    )


def _resign_report(report: dict[str, object]) -> None:
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = corrective._sha256(
        corrective._REPORT_DOMAIN, payload
    )


def test_shared_only_is_selected_when_conditional_columns_add_nothing() -> None:
    records = _records((0.04, -0.03, 0.0, 0.0, 0.0, 0.0))
    report = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=tuple(range(6)),
    )

    validate_token_loss_fisher_corrective_report(report)
    assert report["metrics"]["conditional_active_fold_count"] == 0
    assert set(
        report["metrics"][
            "selected_deviation_ridge_label_by_held_family"
        ].values()
    ) == {"inf"}
    assert (
        report["metrics"][
            "conditional_family_macro_relative_rmse_improvement_vs_shared_only"
        ]
        == 0.0
    )
    assert report["passed"] is False


def test_shared_only_numerical_dust_cannot_pass_conditional_gates() -> None:
    report = build_token_loss_fisher_corrective_report(
        _records((0.3, -0.2, 0.0, 0.0, 0.0, 0.0)),
        coordinate_indices=tuple(range(6)),
    )
    gates = dict(report["gate_results"])

    assert report["passed"] is False
    assert gates[
        (
            "conditional_family_macro_relative_rmse_improvement_"
            "vs_shared_only_at_least_minimum"
        )
    ] is False
    assert gates[
        "conditional_family_win_count_vs_shared_only_at_least_minimum"
    ] is False


def test_nested_screen_retains_transferable_conditional_correction() -> None:
    records = _records((0.03, -0.02, 0.08, 0.04, 0.06, -0.03))
    report = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=tuple(range(6)),
    )

    validate_token_loss_fisher_corrective_report(report)
    assert report["metrics"]["conditional_active_fold_count"] == 4
    assert set(
        report["metrics"][
            "selected_deviation_ridge_label_by_held_family"
        ].values()
    ) != {"inf"}
    assert (
        report["metrics"][
            "conditional_family_macro_relative_rmse_improvement_vs_shared_only"
        ]
        > 0.0
    )
    assert report["passed"] is True


def test_report_is_order_invariant_and_replays_from_prompt_moments() -> None:
    records = _records((0.03, -0.02, 0.08, 0.04, 0.06, -0.03))
    forward = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=tuple(range(6)),
    )
    reverse = build_token_loss_fisher_corrective_report(
        tuple(reversed(records)),
        coordinate_indices=tuple(range(6)),
    )

    assert reverse == forward
    assert (
        replay_token_loss_fisher_corrective_report(
            tuple(row.to_dict() for row in records),
            forward,
        )
        == forward
    )


def test_report_and_fold_tampering_fail_closed() -> None:
    records = _records((0.04, -0.03, 0.0, 0.0, 0.0, 0.0))
    report = build_token_loss_fisher_corrective_report(
        records,
        coordinate_indices=tuple(range(6)),
    )
    changed = copy.deepcopy(report)
    coefficients = list(changed["folds"][0]["coefficients"])
    coefficients[0] += 0.01
    changed["folds"][0]["coefficients"] = coefficients

    with pytest.raises(ValueError, match="fold hash mismatch"):
        validate_token_loss_fisher_corrective_report(changed)

    changed = copy.deepcopy(report)
    changed["metrics"]["family_win_count"] += 1
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        validate_token_loss_fisher_corrective_report(changed)


def test_fully_resigned_semantic_tampering_fails_closed() -> None:
    report = build_token_loss_fisher_corrective_report(
        _records((0.04, -0.03, 0.0, 0.0, 0.0, 0.0)),
        coordinate_indices=tuple(range(6)),
    )

    fabricated_gates = copy.deepcopy(report)
    fabricated_gates["gate_results"] = (("fabricated_gate", True),)
    fabricated_gates["passed"] = True
    _resign_report(fabricated_gates)
    with pytest.raises(ValueError, match="gate results differ"):
        validate_token_loss_fisher_corrective_report(fabricated_gates)

    leaked_family = copy.deepcopy(report)
    first_fold = leaked_family["folds"][0]
    first_fold["train_family_ids"] = tuple(leaked_family["family_ids"])
    _resign_fold(first_fold)
    _resign_report(leaked_family)
    with pytest.raises(ValueError, match="train-family partition differs"):
        validate_token_loss_fisher_corrective_report(leaked_family)

    fabricated_example = copy.deepcopy(report)
    first_fold = fabricated_example["folds"][0]
    first_fold["held_example_ids"] = ("fabricated-example",)
    _resign_fold(first_fold)
    _resign_report(fabricated_example)
    with pytest.raises(ValueError, match="fold prompt partition differs"):
        validate_token_loss_fisher_corrective_report(fabricated_example)

    moved_selection = copy.deepcopy(report)
    first_fold = moved_selection["folds"][0]
    candidates = first_fold["inner_ridge_candidates"]
    selected_index = next(
        index for index, candidate in enumerate(candidates)
        if candidate["selected"]
    )
    replacement_index = (selected_index + 1) % len(candidates)
    candidates[selected_index]["selected"] = False
    candidates[replacement_index]["selected"] = True
    _resign_fold(first_fold)
    _resign_report(moved_selection)
    with pytest.raises(ValueError, match="inner selection flags differ"):
        validate_token_loss_fisher_corrective_report(moved_selection)


def test_corrective_view_requires_exactly_six_coordinates() -> None:
    records = _records((0.04, -0.03, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="exactly six"):
        build_token_loss_fisher_corrective_report(
            records,
            coordinate_indices=(0, 1, 2, 3),
        )
    with pytest.raises(ValueError, match="canonical source order"):
        build_token_loss_fisher_corrective_report(
            records,
            coordinate_indices=(1, 0, 2, 3, 4, 5),
        )
