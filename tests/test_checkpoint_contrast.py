from __future__ import annotations

import torch

from fisher_graph.checkpoint_contrast import (
    CheckpointContrastReport,
    CheckpointContrastThresholds,
    analyze_checkpoint_contrast,
)


def test_linear_checkpoint_contrast_satisfies_jvp_vjp_adjoint_identity() -> None:
    matrix = torch.tensor(
        [[1.5, -0.25], [0.5, 2.0]],
        dtype=torch.float64,
    )

    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mixed = value @ matrix.T
        return value, mixed, mixed * 0.5

    left = torch.tensor([[[0.2, -0.4], [0.7, 0.3]]], dtype=torch.float64)
    right = left + torch.tensor(
        [[[0.3, 0.1], [-0.2, 0.4]]],
        dtype=torch.float64,
    )
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("input", "mixed", "output"),
        left_input=left,
        right_input=right,
    )

    assert isinstance(report, CheckpointContrastReport)
    assert report.output_resolved
    assert report.output_jvp_secant_cosine is not None
    assert abs(report.output_jvp_secant_cosine - 1.0) < 1e-14
    assert report.output_midpoint_linearization_relative_error < 1e-14
    assert report.adjoint_relative_error < 1e-14
    assert report.classification == "distributed_or_inconclusive"
    assert len(report.artifact_sha256) == 64
    assert (
        CheckpointContrastReport(
            checkpoint_names=report.checkpoint_names,
            input_left_sha256=report.input_left_sha256,
            input_right_sha256=report.input_right_sha256,
            input_midpoint_sha256=report.input_midpoint_sha256,
            input_tangent_sha256=report.input_tangent_sha256,
            output_checkpoint_index=report.output_checkpoint_index,
            output_secant_l2=report.output_secant_l2,
            output_symmetric_relative_separation=(
                report.output_symmetric_relative_separation
            ),
            output_resolved=report.output_resolved,
            output_jvp_secant_cosine=report.output_jvp_secant_cosine,
            output_midpoint_linearization_relative_error=(
                report.output_midpoint_linearization_relative_error
            ),
            adjoint_left_inner_product=report.adjoint_left_inner_product,
            adjoint_right_inner_product=report.adjoint_right_inner_product,
            adjoint_relative_error=report.adjoint_relative_error,
            causal_leakage_fraction=report.causal_leakage_fraction,
            classification=report.classification,
            localized_transition=report.localized_transition,
            reason_codes=report.reason_codes,
            rows=report.rows,
            thresholds=report.thresholds,
            artifact_sha256=report.artifact_sha256,
        ).state_dict()
        == report.state_dict()
    )


def test_single_sharp_linear_drop_is_localized() -> None:
    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = value * 2.0
        suppressed = first * 0.05 + 10.0
        return first, suppressed, suppressed + value * 0.001

    left = torch.tensor([[[-1.0, 0.5]]], dtype=torch.float64)
    right = torch.tensor([[[1.0, -0.5]]], dtype=torch.float64)
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("first", "suppressed", "output"),
        left_input=left,
        right_input=right,
    )

    assert report.classification == "localized_attenuation"
    assert report.localized_transition == "first -> suppressed"


def test_nonlinear_finite_displacement_is_not_falsely_localized() -> None:
    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        squared = value.square()
        return value, squared, squared + value.pow(3)

    left = torch.tensor([[[-1.0, -0.5]]], dtype=torch.float64)
    right = torch.tensor([[[1.5, 0.75]]], dtype=torch.float64)
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("input", "square", "output"),
        left_input=left,
        right_input=right,
    )

    assert report.output_resolved
    assert report.classification == "nonlinear_or_finite_displacement"
    assert report.localized_transition is None
    assert (
        "midpoint_jvp_does_not_match_endpoint_secant"
        in report.reason_codes
    )


def test_low_output_signal_is_uninformative_not_localized() -> None:
    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        visible = value * 3.0
        return visible, visible * 0.0

    left = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    right = torch.tensor([[[2.0, -1.0]]], dtype=torch.float64)
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("visible", "gain_null"),
        left_input=left,
        right_input=right,
    )

    assert not report.output_resolved
    assert report.classification == "uninformative_low_output_contrast"
    assert report.localized_transition is None


def test_causal_jvp_has_no_response_before_changed_source() -> None:
    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        causal = torch.cumsum(value, dim=1)
        return value, torch.tanh(causal)

    left = torch.zeros(1, 4, 2, dtype=torch.float64)
    right = left.clone()
    right[:, 2, 0] = 0.25
    mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.tensor([[4, 7, 9, 15]], dtype=torch.int64)
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("input", "causal_output"),
        left_input=left,
        right_input=right,
        output_mask=mask,
        input_valid_mask=mask,
        logical_positions=positions,
    )

    assert report.causal_leakage_fraction == 0.0
    assert report.classification != "noncausal_or_invalid"


def test_future_dependency_is_rejected_by_causal_check() -> None:
    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        reverse = torch.flip(
            torch.cumsum(torch.flip(value, dims=(1,)), dim=1),
            dims=(1,),
        )
        return value, reverse

    left = torch.zeros(1, 4, 1, dtype=torch.float64)
    right = left.clone()
    right[:, 2, 0] = 1.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    positions = torch.arange(4, dtype=torch.int64).unsqueeze(0)
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("input", "future_dependent"),
        left_input=left,
        right_input=right,
        output_mask=mask,
        input_valid_mask=mask,
        logical_positions=positions,
    )

    assert report.causal_leakage_fraction > 0.0
    assert report.classification == "noncausal_or_invalid"


def test_unit_offset_rmsnorm_null_output_stays_uninformative() -> None:
    epsilon = 1e-6
    gain = torch.tensor([1.5, 0.0, 0.75], dtype=torch.float64)

    def checkpoints(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        denominator = (value.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
        normalized = gain * value / denominator
        return value, normalized, normalized[..., 1:2]

    left = torch.tensor([[[0.4, -0.5, 0.2]]], dtype=torch.float64)
    right = torch.tensor([[[0.4, 0.5, 0.2]]], dtype=torch.float64)
    report = analyze_checkpoint_contrast(
        checkpoints,
        checkpoint_names=("hidden", "normalized", "null_output"),
        left_input=left,
        right_input=right,
    )

    assert not report.output_resolved
    assert report.classification == "uninformative_low_output_contrast"
    assert report.rows[-1].jvp_l2 == 0.0
    assert report.rows[-1].vjp_l2 == 0.0


def test_threshold_validation_rejects_invalid_values() -> None:
    try:
        CheckpointContrastThresholds(
            maximum_linearization_relative_error=1.1
        )
    except ValueError as error:
        assert "must not exceed one" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid threshold was accepted")
