from __future__ import annotations

import copy

import pytest
import torch

from fisher_graph.causal_edge_jvp import (
    CausalEdgeJVPFit,
    apply_causal_lag_convolution,
    estimate_causal_edge_jvp,
)


def _causal_linear_function(kernel: torch.Tensor):
    def function(source: torch.Tensor) -> torch.Tensor:
        rows = []
        for target in range(source.shape[1]):
            value = source[:, target, :] @ kernel[0]
            for lag in range(1, kernel.shape[0]):
                if target >= lag:
                    value = value + source[:, target - lag, :] @ kernel[lag]
            rows.append(value)
        return torch.stack(rows, dim=1)

    return function


def test_exact_projected_stationary_causal_kernel_is_recovered() -> None:
    expected = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 2.0]],
            [[-0.75, 0.3], [1.25, -1.0]],
            [[0.2, 0.4], [-0.6, 0.8]],
        ],
        dtype=torch.float64,
    )
    baseline = torch.randn(1, 7, 2, dtype=torch.float64)
    positions = torch.arange(7, dtype=torch.int64)
    mask = torch.ones(7, dtype=torch.bool)

    fit = estimate_causal_edge_jvp(
        _causal_linear_function(expected),
        baseline_source=baseline,
        logical_positions=positions,
        valid_mask=mask,
        source_decoder=torch.eye(2, dtype=torch.float64),
        target_encoder=torch.eye(2, dtype=torch.float64),
        max_lag=2,
        probe_count=32,
        probe_seed=101,
        ridge=0.0,
    )

    torch.testing.assert_close(fit.kernel, expected, atol=1e-11, rtol=1e-11)
    assert fit.design_rank == 6
    assert fit.jvp_evaluation_count == 32
    assert fit.relative_output_residual < 1e-12
    assert fit.causal_direction == "source_at_or_before_target"
    assert fit.lag_definition == (
        "target_logical_position_minus_source_logical_position"
    )
    assert set(fit.per_lag_matrices) == {0, 1, 2}
    fit.validate_binding(
        baseline_source=baseline,
        source_decoder=torch.eye(2, dtype=torch.float64),
        target_encoder=torch.eye(2, dtype=torch.float64),
        logical_positions=positions,
        valid_mask=mask,
    )

    source_modes = torch.randn(3, 7, 2, dtype=torch.float64)
    actual = fit.execute(
        source_modes,
        logical_positions=positions,
        valid_mask=mask,
    )
    expected_output = torch.cat(
        [
            _causal_linear_function(expected)(row.unsqueeze(0))
            for row in source_modes
        ],
        dim=0,
    )
    torch.testing.assert_close(actual, expected_output)


def test_binding_validation_rejects_every_drifted_input() -> None:
    baseline = torch.randn(1, 5, 1, dtype=torch.float64)
    positions = torch.arange(5)
    mask = torch.ones(5, dtype=torch.bool)
    decoder = torch.ones(1, 1, dtype=torch.float64)
    encoder = torch.ones(1, 1, dtype=torch.float64)
    fit = estimate_causal_edge_jvp(
        lambda value: value,
        baseline_source=baseline,
        logical_positions=positions,
        valid_mask=mask,
        source_decoder=decoder,
        target_encoder=encoder,
        max_lag=0,
        probe_count=4,
        probe_seed=3,
        ridge=0.0,
    )

    base = {
        "baseline_source": baseline,
        "source_decoder": decoder,
        "target_encoder": encoder,
        "logical_positions": positions,
        "valid_mask": mask,
    }
    cases = (
        (
            "baseline_source",
            baseline + 1.0,
            "baseline_source does not match",
        ),
        ("source_decoder", decoder * 2.0, "source_decoder does not match"),
        ("target_encoder", encoder * 2.0, "target_encoder does not match"),
        (
            "logical_positions",
            torch.tensor([0, 1, 2, 3, 5]),
            "logical_positions do not match",
        ),
        (
            "valid_mask",
            torch.tensor([True, True, True, True, False]),
            "valid_mask does not match",
        ),
    )
    for key, value, message in cases:
        arguments = dict(base)
        arguments[key] = value
        with pytest.raises(ValueError, match=message):
            fit.validate_binding(**arguments)


def test_decoder_and_encoder_projection_orientations_are_explicit() -> None:
    hidden_kernel = torch.tensor(
        [
            [
                [1.0, 0.2, -0.3, 0.5],
                [0.4, -0.8, 0.7, 0.1],
                [-0.2, 0.6, 0.9, -0.4],
            ],
            [
                [-0.5, 0.1, 0.4, 0.3],
                [0.8, 0.2, -0.6, 0.7],
                [0.3, -0.9, 0.2, 0.5],
            ],
        ],
        dtype=torch.float64,
    )
    decoder = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [-0.25, 0.75]],
        dtype=torch.float64,
    )
    encoder = torch.tensor(
        [[1.0, 0.0], [0.25, 0.5], [-0.5, 1.0], [0.75, -0.25]],
        dtype=torch.float64,
    )
    expected = torch.stack(
        [
            decoder.transpose(0, 1) @ matrix @ encoder
            for matrix in hidden_kernel
        ]
    )

    fit = estimate_causal_edge_jvp(
        _causal_linear_function(hidden_kernel),
        baseline_source=torch.randn(1, 6, 3, dtype=torch.float64),
        logical_positions=torch.arange(6),
        valid_mask=torch.ones(6, dtype=torch.bool),
        source_decoder=decoder,
        target_encoder=encoder,
        max_lag=1,
        probe_count=24,
        probe_seed=7,
        ridge=0.0,
    )

    torch.testing.assert_close(fit.kernel, expected, atol=1e-11, rtol=1e-11)


def test_logical_position_gaps_do_not_become_tensor_offset_edges() -> None:
    kernel = torch.tensor(
        [
            [[2.0]],
            [[3.0]],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor([0, 1, 3, 4, 99], dtype=torch.int64)
    mask = torch.tensor([True, True, True, True, False])
    source_modes = torch.tensor(
        [[1.0], [2.0], [4.0], [8.0], [1000.0]],
        dtype=torch.float64,
    )

    output = apply_causal_lag_convolution(
        source_modes,
        kernel=kernel,
        logical_positions=positions,
        valid_mask=mask,
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[2.0], [7.0], [8.0], [28.0], [0.0]],
            dtype=torch.float64,
        ),
    )


def test_truncated_causal_model_reports_uncaptured_output() -> None:
    kernel = torch.tensor(
        [
            [[1.0]],
            [[0.0]],
            [[2.0]],
        ],
        dtype=torch.float64,
    )
    fit = estimate_causal_edge_jvp(
        _causal_linear_function(kernel),
        baseline_source=torch.zeros(1, 8, 1, dtype=torch.float64),
        logical_positions=torch.arange(8),
        valid_mask=torch.ones(8, dtype=torch.bool),
        source_decoder=torch.ones(1, 1, dtype=torch.float64),
        target_encoder=torch.ones(1, 1, dtype=torch.float64),
        max_lag=1,
        probe_count=24,
        probe_seed=4,
        ridge=0.0,
    )

    assert fit.relative_output_residual > 0.5
    assert fit.output_residual_frobenius > 0.0


def test_fit_is_deterministic_and_strictly_round_trips() -> None:
    kernel = torch.tensor(
        [
            [[1.5]],
            [[-0.25]],
        ],
        dtype=torch.float64,
    )
    arguments = {
        "baseline_source": torch.randn(1, 5, 1, dtype=torch.float64),
        "logical_positions": torch.arange(5),
        "valid_mask": torch.ones(5, dtype=torch.bool),
        "source_decoder": torch.ones(1, 1, dtype=torch.float64),
        "target_encoder": torch.ones(1, 1, dtype=torch.float64),
        "max_lag": 1,
        "probe_count": 12,
        "probe_seed": 88,
        "ridge": 1e-8,
    }
    first = estimate_causal_edge_jvp(
        _causal_linear_function(kernel),
        **arguments,
    )
    second = estimate_causal_edge_jvp(
        _causal_linear_function(kernel),
        **arguments,
    )

    assert first.artifact_sha256 == second.artifact_sha256
    torch.testing.assert_close(first.kernel, second.kernel)
    restored = CausalEdgeJVPFit.from_state_dict(first.state_dict())
    assert restored.artifact_sha256 == first.artifact_sha256
    torch.testing.assert_close(restored.kernel, first.kernel)

    unknown = copy.deepcopy(first.state_dict())
    unknown["surprise"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        CausalEdgeJVPFit.from_state_dict(unknown)

    tampered = copy.deepcopy(first.state_dict())
    tampered["kernel"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="kernel hash mismatch"):
        CausalEdgeJVPFit.from_state_dict(tampered)


def test_validation_rejects_noncausal_position_order() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_causal_edge_jvp(
            lambda value: value,
            baseline_source=torch.zeros(1, 3, 1),
            logical_positions=torch.tensor([0, 2, 1]),
            valid_mask=torch.ones(3, dtype=torch.bool),
            source_decoder=torch.ones(1, 1),
            target_encoder=torch.ones(1, 1),
            max_lag=1,
            probe_count=2,
            probe_seed=0,
            ridge=0.0,
        )


def test_estimator_rejects_nonfinite_primal_with_finite_jvp() -> None:
    def nonfinite_primal(value: torch.Tensor) -> torch.Tensor:
        return value * 0.0 + torch.tensor(
            float("nan"),
            device=value.device,
            dtype=value.dtype,
        )

    with pytest.raises(ValueError, match="function output must be finite"):
        estimate_causal_edge_jvp(
            nonfinite_primal,
            baseline_source=torch.zeros(1, 3, 1, dtype=torch.float64),
            logical_positions=torch.arange(3),
            valid_mask=torch.ones(3, dtype=torch.bool),
            source_decoder=torch.ones(1, 1, dtype=torch.float64),
            target_encoder=torch.ones(1, 1, dtype=torch.float64),
            max_lag=1,
            probe_count=2,
            probe_seed=0,
            ridge=0.0,
        )
