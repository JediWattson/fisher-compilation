from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.gemma3_cross_block_row_pruned_executor import (
    Gemma3DirectedCrossBlockMergedSupermodeExecutor,
    Gemma3CrossBlockRowPrunedBinding,
)


class _GemmaMLP(nn.Module):
    def __init__(self, width: int = 4, intermediate: int = 6) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def features(self, values: Tensor) -> Tensor:
        return F.gelu(
            self.gate_proj(values),
            approximate="tanh",
        ) * self.up_proj(values)

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(self.features(values))


def _binding(*, carry_scale: float = -0.5):
    return Gemma3CrossBlockRowPrunedBinding(
        source_model_fingerprint="1" * 64,
        source_execution_fingerprint="4" * 64,
        source_plan_artifact_sha256="2" * 64,
        source_replacement_oracle_artifact_sha256="3" * 64,
        proposal_id="layer-6-to-15",
        anchor_layer_id="layer.6",
        anchor_source_index=2,
        consumer_layer_id="layer.15",
        consumer_source_index=4,
        carry_scale=carry_scale,
        activation="gelu_pytorch_tanh",
    )


def _source_pair() -> tuple[_GemmaMLP, _GemmaMLP]:
    torch.manual_seed(733)
    anchor = _GemmaMLP().eval()
    consumer = _GemmaMLP().eval()
    for module in (anchor, consumer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return anchor, consumer


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }


def test_physically_prunes_generator_rows_and_matches_native_replacement() -> None:
    anchor, consumer = _source_pair()
    anchor_fingerprint = module_state_fingerprint(anchor)
    consumer_fingerprint = module_state_fingerprint(consumer)
    executor = Gemma3DirectedCrossBlockMergedSupermodeExecutor(
        anchor,
        consumer,
        binding=_binding(),
    )
    anchor_input = torch.randn(2, 3, 4)
    consumer_input = torch.randn(2, 3, 4)
    valid = torch.tensor(
        [[True, True, False], [True, False, False]]
    )

    execution = executor(anchor_input, consumer_input, valid)
    native_anchor_features = anchor.features(anchor_input)
    expected_consumer_features = consumer.features(consumer_input)
    replaced = expected_consumer_features.clone()
    signed_carry = (
        -0.5 * native_anchor_features[..., 2]
    ).masked_fill(~valid, 0)
    replaced[..., 4] = signed_carry
    expected_consumer = consumer.down_proj(replaced)

    torch.testing.assert_close(
        execution.anchor_output,
        anchor(anchor_input),
    )
    torch.testing.assert_close(
        execution.native_anchor_feature,
        native_anchor_features[..., 2],
    )
    torch.testing.assert_close(execution.carried_scalar, signed_carry)
    assert torch.equal(execution.consumer_output, expected_consumer)
    torch.testing.assert_close(
        execution.retained_consumer_features,
        expected_consumer_features.index_select(
            -1,
            torch.tensor([0, 1, 2, 3, 5]),
        ),
    )

    assert executor.consumer_gate_proj.weight.shape == (5, 4)
    assert executor.consumer_up_proj.weight.shape == (5, 4)
    assert executor.consumer_down_proj.weight.shape == (4, 6)
    torch.testing.assert_close(
        executor.consumer_decoder_column,
        consumer.down_proj.weight[:, 4],
    )
    torch.testing.assert_close(
        executor.consumer_down_proj.weight,
        consumer.down_proj.weight,
    )
    assert module_state_fingerprint(anchor) == anchor_fingerprint
    assert module_state_fingerprint(consumer) == consumer_fingerprint
    assert not (
        (_storage_pointers(anchor) | _storage_pointers(consumer))
        & _storage_pointers(executor)
    )
    manifest = executor.architecture_manifest()
    assert manifest["physically_skipped_consumer_gate_rows"] == 1
    assert manifest["physically_skipped_consumer_up_rows"] == 1
    assert manifest["preserved_consumer_down_columns"] == 1
    assert manifest["anchor_generator_retained_and_shared"]
    assert manifest["consumer_generator_removed"]
    assert manifest["consumer_decoder_retained"]
    assert manifest["full_consumer_down_projection_preserved"]
    assert manifest["consumer_down_input_scatter_width"] == 6
    assert manifest["deletion_is_ablation_control_only"]
    assert (
        manifest["compression_semantics"]
        == "directed_cross_block_merged_supermode"
    )
    assert not manifest["source_free_window_claimed"]
    assert not manifest["kernel_speedup_claimed"]


def test_accounting_reports_rows_decoder_scale_and_whole_model_delta() -> None:
    anchor, consumer = _source_pair()
    executor = Gemma3DirectedCrossBlockMergedSupermodeExecutor(
        anchor,
        consumer,
        binding=_binding(),
    )
    valid = torch.tensor(
        [[True, True, False], [True, False, False]]
    )

    accounting = executor.logical_accounting(
        valid,
        source_whole_model_parameters=1_000,
    )

    assert accounting.query_tokens == 6
    assert accounting.valid_tokens == 3
    assert accounting.source_pair_learned_parameters == 144
    assert accounting.candidate_pair_learned_parameters == 136
    assert accounting.removed_learned_parameters == 8
    assert accounting.fixed_carry_scale_coefficients == 1
    assert accounting.net_stored_coefficient_savings == 7
    assert accounting.preserved_consumer_decoder_parameters == 4
    assert accounting.source_pair_logical_linear_macs == 432
    assert accounting.candidate_pair_logical_linear_macs == 408
    assert accounting.removed_logical_linear_macs == 24
    assert accounting.carry_scale_logical_macs == 3
    assert accounting.net_logical_arithmetic_macs_saved == 21
    assert accounting.source_pair_dense_linear_macs == 864
    assert accounting.candidate_pair_dense_linear_macs == 816
    assert accounting.removed_dense_linear_macs == 48
    assert accounting.carry_scale_dense_macs == 6
    assert accounting.net_dense_arithmetic_macs_saved == 42
    assert accounting.candidate_whole_model_learned_parameters == 992
    assert accounting.candidate_whole_model_stored_coefficients == 993


def test_consumer_rejects_unmasked_invalid_carry() -> None:
    anchor, consumer = _source_pair()
    executor = Gemma3DirectedCrossBlockMergedSupermodeExecutor(
        anchor,
        consumer,
        binding=_binding(),
    )
    values = torch.randn(1, 2, 4)
    valid = torch.tensor([[True, False]])
    carry = torch.tensor([[0.5, 0.25]])

    with pytest.raises(ValueError, match="invalid-row-zeroed"):
        executor.forward_consumer(values, carry, valid)


def test_valid_query_equivalence_with_left_and_right_padding() -> None:
    anchor, consumer = _source_pair()
    executor = Gemma3DirectedCrossBlockMergedSupermodeExecutor(
        anchor,
        consumer,
        binding=_binding(),
    )
    anchor_input = torch.randn(2, 4, 4)
    consumer_input = torch.randn(2, 4, 4)
    valid = torch.tensor(
        [
            [False, False, True, True],
            [True, True, False, False],
        ]
    )

    execution = executor(anchor_input, consumer_input, valid)
    native_anchor = anchor.features(anchor_input)
    oracle_features = consumer.features(consumer_input)
    oracle_features[..., 4] = torch.where(
        valid,
        -0.5 * native_anchor[..., 2],
        oracle_features[..., 4],
    )
    oracle_output = consumer.down_proj(oracle_features)

    torch.testing.assert_close(
        execution.consumer_output[valid],
        oracle_output[valid],
    )
    assert not execution.carried_scalar[~valid].count_nonzero()
    assert executor.architecture_manifest()["equivalence_domain"] == (
        "valid_query_positions"
    )


def test_strict_artifact_round_trip_and_tamper_rejection() -> None:
    anchor, consumer = _source_pair()
    executor = Gemma3DirectedCrossBlockMergedSupermodeExecutor(
        anchor,
        consumer,
        binding=_binding(),
    )
    artifact = executor.artifact_state_dict()

    assert (
        "retained_consumer_source_indices"
        not in artifact["model_state_dict"]
    )
    assert artifact["contains_complete_source_model"] is False
    assert artifact["requires_compatible_base_model"] is True
    restored = executor.from_artifact_state_dict(artifact)
    assert restored.binding == executor.binding
    assert restored.architecture_manifest() == executor.architecture_manifest()
    assert restored.execution_fingerprint() == executor.execution_fingerprint()
    anchor_input = torch.randn(2, 3, 4)
    consumer_input = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    expected = executor(anchor_input, consumer_input, valid)
    actual = restored(anchor_input, consumer_input, valid)
    torch.testing.assert_close(actual.anchor_output, expected.anchor_output)
    torch.testing.assert_close(
        actual.consumer_output,
        expected.consumer_output,
    )

    tampered = copy.deepcopy(artifact)
    tampered["model_state_dict"][
        "consumer_gate_proj.weight"
    ][0, 0] += 0.25
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        executor.from_artifact_state_dict(tampered)
