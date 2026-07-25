import copy

import pytest
import torch

from fisher_graph.adapters import SequenceContext, SequenceInputOrigin
from fisher_graph.gated_executor import (
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)
from fisher_graph.rotated_span_executor import (
    RotatedSpanBlockExecutor,
    deterministic_orthogonal_complement,
)


def _sequence(mask: torch.Tensor) -> SequenceContext:
    positions = torch.arange(
        mask.shape[1],
        dtype=torch.long,
        device=mask.device,
    ).unsqueeze(0).expand(mask.shape[0], -1)
    return SequenceContext(
        query_valid_mask=mask,
        key_valid_mask=mask,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=False,
            cache_positions_supplied=False,
        ),
    )


def _executor(width: int = 6) -> RotatedSpanBlockExecutor:
    torch.manual_seed(901)
    graph = ResidualGatedCausalModalExecutor(
        GatedCausalModalExecutorConfig(
            input_modes=width,
            output_modes=width - 1,
            expert_count=2,
            expert_rank=3,
            router_width=4,
            same_position_skip=False,
            max_positive_lag=None,
        ),
        dtype=torch.float32,
    )
    normal = torch.linspace(-0.4, 0.8, width, dtype=torch.float64)
    pivot = int(normal.abs().argmax().item())
    if normal[pivot] < 0:
        normal.neg_()
    normal /= torch.linalg.vector_norm(normal)
    return RotatedSpanBlockExecutor(
        normal=normal,
        input_mean=torch.linspace(-1.0, 1.0, width),
        input_scale=torch.linspace(0.5, 1.5, width),
        graph=graph,
    )


@pytest.mark.parametrize("pivot", [0, 2, 4])
def test_deterministic_complement_handles_coordinate_axes(pivot: int) -> None:
    normal = torch.zeros(5, dtype=torch.float64)
    normal[pivot] = 1.0
    first = deterministic_orthogonal_complement(normal)
    second = deterministic_orthogonal_complement(normal.clone())
    assert torch.equal(first, second)
    torch.testing.assert_close(
        first.T @ first,
        torch.eye(4, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first @ first.T,
        torch.eye(5, dtype=torch.float64)
        - torch.outer(normal, normal),
        rtol=0.0,
        atol=0.0,
    )


def test_output_delta_is_in_span_and_invalid_positions_are_unchanged() -> None:
    executor = _executor()
    hidden = torch.randn(2, 5, 6, requires_grad=True)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    result = executor.forward_components(hidden, _sequence(mask))
    valid_delta = (result.output - hidden)[mask]
    normal_component = (
        valid_delta.to(torch.float64) @ executor.normal
    )
    torch.testing.assert_close(
        normal_component,
        torch.zeros_like(normal_component),
        rtol=0.0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.output[~mask],
        hidden[~mask],
        rtol=0.0,
        atol=0.0,
    )
    result.output.square().sum().backward()
    assert hidden.grad is not None
    assert all(parameter.grad is not None for parameter in executor.parameters())


def test_executor_cannot_read_future_tensor_slots() -> None:
    executor = _executor().eval()
    mask = torch.ones(1, 6, dtype=torch.bool)
    sequence = _sequence(mask)
    baseline = torch.randn(1, 6, 6)
    changed = baseline.clone()
    changed[:, 4:] += 100.0
    first = executor(baseline, sequence)
    second = executor(changed, sequence)
    torch.testing.assert_close(
        first[:, :4],
        second[:, :4],
        rtol=0.0,
        atol=0.0,
    )


def test_strict_artifact_roundtrip_and_fingerprint_tamper() -> None:
    executor = _executor().eval()
    state = executor.artifact_state_dict()
    loaded = RotatedSpanBlockExecutor.from_artifact_state_dict(state)
    assert loaded.execution_fingerprint() == executor.execution_fingerprint()
    hidden = torch.randn(2, 4, 6)
    mask = torch.ones(2, 4, dtype=torch.bool)
    torch.testing.assert_close(
        loaded(hidden, _sequence(mask)),
        executor(hidden, _sequence(mask)),
    )

    changed = copy.deepcopy(state)
    changed["input_mean"][0] += 0.25
    with pytest.raises(
        ValueError,
        match="execution fingerprint mismatch",
    ):
        RotatedSpanBlockExecutor.from_artifact_state_dict(changed)


def test_shape_and_config_guards_are_fail_closed() -> None:
    graph = ResidualGatedCausalModalExecutor(
        GatedCausalModalExecutorConfig(
            input_modes=5,
            output_modes=5,
            expert_count=1,
            expert_rank=2,
            router_width=2,
            same_position_skip=True,
        )
    )
    with pytest.raises(ValueError, match="same_position_skip"):
        RotatedSpanBlockExecutor(
            normal=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            input_mean=torch.zeros(5),
            input_scale=torch.ones(5),
            graph=graph,
        )
