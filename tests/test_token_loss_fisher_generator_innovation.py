from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.token_loss_fisher import (
    build_token_loss_fisher_prompt_record,
)
from fisher_graph.token_loss_fisher_generator_innovation import (
    build_generator_innovation_nested_lofo_report,
    generator_innovation_corner_operator_norms,
    project_generator_innovation_coefficients,
    replay_generator_innovation_nested_lofo_report,
    validate_generator_innovation_nested_lofo_report,
)


_LEGACY_NAMES = (
    "shared_real",
    "shared_imag",
    "balance_contrast_real",
    "balance_contrast_imag",
    "cumulative_occupancy_contrast_real",
    "cumulative_occupancy_contrast_imag",
)
_GENERATOR_NAMES = (
    "generator_real_shared",
    "generator_imag_shared",
    "generator_real_innovation",
    "generator_imag_innovation",
)
_BASIS = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
)


def _records() -> tuple[tuple[object, ...], tuple[object, ...]]:
    generator = torch.Generator().manual_seed(71)
    legacy_records = []
    innovation_records = []
    for index in range(16):
        token_scores = torch.randn(
            72,
            4,
            generator=generator,
            dtype=torch.float64,
        )
        token_scores[:, 2:] += 0.2 * token_scores[:, :2]
        target = (
            token_scores
            @ torch.tensor((0.04, -0.03, 0.025, -0.02), dtype=torch.float64)
            + 0.01
            * torch.randn(72, generator=generator, dtype=torch.float64)
        )
        legacy = torch.randn(
            72,
            6,
            generator=generator,
            dtype=torch.float64,
        )
        legacy[:, :2] = token_scores[:, :2]
        example_id = f"example-{index:02d}"
        family_id = f"family-{index % 8}"
        legacy_records.append(
            build_token_loss_fisher_prompt_record(
                example_id=example_id,
                family_id=family_id,
                coordinate_names=_LEGACY_NAMES,
                token_scores=legacy,
                compensation_target=target,
            )
        )
        innovation_records.append(
            build_token_loss_fisher_prompt_record(
                example_id=example_id,
                family_id=family_id,
                coordinate_names=_GENERATOR_NAMES,
                token_scores=token_scores,
                compensation_target=target,
            )
        )
    return tuple(legacy_records), tuple(innovation_records)


def test_nested_generator_report_validates_and_exactly_replays() -> None:
    legacy, generator = _records()
    report = build_generator_innovation_nested_lofo_report(
        legacy,
        generator,
        fixed_basis=_BASIS,
    )
    validate_generator_innovation_nested_lofo_report(report)
    assert len(report["folds"]) == 8
    assert report["recipe"]["held_family_used_for_ridge_selection"] is False
    assert all(
        len(fold["conditional_fit"]["post_projection_corner_operator_norms"])
        == 16
        for fold in report["folds"]
    )
    assert (
        replay_generator_innovation_nested_lofo_report(
            legacy,
            generator,
            fixed_basis=_BASIS,
            expected_report=report,
        )
        == report
    )


def test_one_scale_projection_proves_all_sixteen_corners() -> None:
    coefficients = (2.0, -3.0, 1.5, -2.5)
    projected, pre, post, scale, applied = (
        project_generator_innovation_coefficients(
            coefficients,
            fixed_basis=_BASIS,
        )
    )
    assert applied is True
    assert scale < 1.0
    assert len(pre) == len(post) == 16
    assert max(post) <= 0.25 * (1.0 + 1.0e-12)
    assert generator_innovation_corner_operator_norms(
        projected,
        fixed_basis=_BASIS,
    ) == post
    ratios = tuple(
        projected[index] / coefficients[index] for index in range(4)
    )
    assert ratios == pytest.approx((scale,) * 4)


def test_report_gate_tamper_and_target_misalignment_fail_closed() -> None:
    legacy, generator = _records()
    report = build_generator_innovation_nested_lofo_report(
        legacy,
        generator,
        fixed_basis=_BASIS,
    )
    changed = copy.deepcopy(report)
    changed["gate_results"] = tuple(
        (name, not value) if index == 0 else (name, value)
        for index, (name, value) in enumerate(changed["gate_results"])
    )
    with pytest.raises(ValueError, match="gate results differ"):
        validate_generator_innovation_nested_lofo_report(changed)

    misaligned = list(generator)
    other_target = torch.ones(72, dtype=torch.float64)
    source = generator[0]
    misaligned[0] = build_token_loss_fisher_prompt_record(
        example_id=source.example_id,
        family_id=source.family_id,
        coordinate_names=_GENERATOR_NAMES,
        token_scores=torch.eye(72, 4, dtype=torch.float64),
        compensation_target=other_target,
    )
    with pytest.raises(ValueError, match="target bindings differ"):
        build_generator_innovation_nested_lofo_report(
            legacy,
            misaligned,
            fixed_basis=_BASIS,
        )
