from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from fisher_graph.adapters import Gemma3CausalLMAdapter
import fisher_graph.gemma3_l3_l4_h4_suffix_vjp_runtime as vjp_module
from fisher_graph.gemma3_l3_l4_h4_suffix_jvp_runtime import (
    Gemma3L3L4H4SuffixJVPRuntime,
)
from fisher_graph.gemma3_l3_l4_h4_suffix_vjp_runtime import (
    Gemma3L3L4H4SuffixVJPRuntime,
    gemma3_l3_l4_h4_suffix_vjp_resource_accounting,
    require_gemma3_l3_l4_h4_suffix_vjp_complete_panel,
)
from test_gemma3_l3_l4_h4_suffix_jvp_runtime import (
    _CausalLM,
    _runtime_fixture,
)


def _execute_three_tokens() -> tuple[object, Tensor]:
    adapter, sequence, h4, logits, teacher, supervised = _runtime_fixture()
    runtime = Gemma3L3L4H4SuffixVJPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    direction = torch.zeros_like(h4, dtype=torch.float64)
    torch.manual_seed(2713)
    direction[:, [0, 2]] = torch.randn(
        (1, 2, h4.shape[-1]), dtype=torch.float64
    )
    result = runtime.execute(
        h4.to(torch.float64).contiguous(),
        direction.contiguous(),
        support_indices=torch.tensor([0, 2], dtype=torch.int64),
        full_h4=h4,
        full_logits=logits,
    )
    return result, direction


def _runtime_fixture_length(
    sequence_length: int,
) -> tuple[object, object, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(2819)
    adapter = Gemma3CausalLMAdapter(_CausalLM().float().eval())
    input_ids = (
        torch.arange(sequence_length, dtype=torch.int64)
        .remainder(adapter.module.config.vocab_size)
        .unsqueeze(0)
    )
    with torch.no_grad():
        full = adapter.forward(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            },
            capture_sites={"layer.4.output"},
        )
    h4 = full.activations["layer.4.output"].detach().contiguous()
    logits = full.logits.detach().contiguous()
    teacher = logits.clone()
    teacher[..., 0] += 0.4
    teacher[..., 2] -= 0.2
    supervised = torch.stack(
        (
            torch.zeros(sequence_length, dtype=torch.int64),
            torch.arange(sequence_length, dtype=torch.int64),
        ),
        dim=1,
    ).contiguous()
    return adapter, full.sequence, h4, logits, teacher, supervised


def test_native_vjp_replays_exact_suffix_and_matches_jvp_direction() -> None:
    adapter, sequence, h4, logits, teacher, supervised = _runtime_fixture()
    runtime = Gemma3L3L4H4SuffixVJPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    direction = torch.zeros_like(h4, dtype=torch.float64)
    torch.manual_seed(2713)
    direction[:, [0, 2]] = torch.randn(
        (1, 2, h4.shape[-1]), dtype=torch.float64
    )
    real_vjp = torch.func.vjp
    real_vmap = torch.func.vmap
    with (
        patch("torch.func.vjp", wraps=real_vjp) as vjp,
        patch("torch.func.vmap", wraps=real_vmap) as vmap,
        patch.object(
            adapter, "run_segment", wraps=adapter.run_segment
        ) as run_segment,
        patch.object(
            adapter, "project_logits", wraps=adapter.project_logits
        ) as project_logits,
    ):
        result = runtime.execute(
            h4.to(torch.float64).contiguous(),
            direction.contiguous(),
            support_indices=torch.tensor([0, 2], dtype=torch.int64),
            full_h4=h4,
            full_logits=logits,
        )
    assert vjp.call_count == 1
    assert vjp.call_args.kwargs == {"has_aux": True}
    assert vmap.call_count == 1
    assert run_segment.call_count == 13
    assert project_logits.call_count == 1

    jvp_runtime = Gemma3L3L4H4SuffixJVPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    jvp_result = jvp_runtime.execute(
        h4.to(torch.float64).contiguous(),
        direction.contiguous(),
        full_h4=h4,
        full_logits=logits,
    )
    torch.testing.assert_close(
        result.primal_token_teacher_kl,
        jvp_result.primal_token_teacher_kl,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.directional_token_teacher_kl,
        jvp_result.directional_token_teacher_kl,
        rtol=2.0e-6,
        atol=2.0e-9,
    )

    receipt = result.receipt
    assert receipt.vjp_transform_count == 1
    assert receipt.vjp_has_aux is True
    assert receipt.vjp_pullback_chunk_call_count == 1
    assert receipt.vmap_pullback_call_count == 1
    assert receipt.h4_dtype_cast_count == 1
    assert receipt.suffix_segment_call_count == 13
    assert receipt.logit_projection_call_count == 1
    assert receipt.outside_direction_nonzero_count == 0
    assert receipt.outside_direction_max_abs == 0.0
    assert receipt.contraction_scope == "authenticated_support_rows_only"
    assert receipt.full_h4_cotangent_coordinate_count == 36
    assert receipt.support_h4_cotangent_coordinate_count == 24
    assert receipt.outside_support_h4_cotangent_coordinate_count == 12
    assert receipt.direction_contraction_coordinate_product_count == 24
    chunk = receipt.chunk_receipts[0]
    assert chunk.full_h4_cotangent_shape == (3, 1, 3, 4)
    assert chunk.support_h4_cotangent_shape == (3, 1, 2, 4)
    assert chunk.full_h4_cotangent_hashed is True
    assert chunk.full_h4_cotangent_retained is False
    assert chunk.full_h4_cotangent_serialized is False
    assert chunk.support_h4_cotangent_hashed is True
    assert chunk.support_h4_cotangent_retained is False
    assert chunk.support_h4_cotangent_serialized is False
    assert chunk.pullback_mechanism == "torch.func.vmap(pullback)"
    assert chunk.canonical_one_hot_token_cotangents is True
    result.validate_integrity()


def test_outside_support_is_rejected_before_vjp() -> None:
    adapter, sequence, h4, logits, teacher, supervised = _runtime_fixture()
    runtime = Gemma3L3L4H4SuffixVJPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    direction = torch.zeros_like(h4, dtype=torch.float64)
    direction[:, 1, 0] = 1.0
    with (
        patch("torch.func.vjp", wraps=torch.func.vjp) as vjp,
        pytest.raises(ValueError, match="nonzero outside support"),
    ):
        runtime.execute(
            h4.to(torch.float64).contiguous(),
            direction.contiguous(),
            support_indices=torch.tensor([0, 2], dtype=torch.int64),
            full_h4=h4,
            full_logits=logits,
        )
    assert vjp.call_count == 0


def test_receipt_and_result_mutations_are_detected() -> None:
    result, _direction = _execute_three_tokens()
    result.directional_token_teacher_kl[0] += 1.0
    with pytest.raises(RuntimeError, match="tensor payload drifted"):
        result.validate_integrity()

    result, _direction = _execute_three_tokens()
    chunk = result.receipt.chunk_receipts[0]
    object.__setattr__(chunk, "full_h4_cotangent_sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="chunk receipt drifted"):
        chunk.validate_integrity()

    result, _direction = _execute_three_tokens()
    object.__setattr__(result.receipt, "vjp_transform_count", 2)
    with pytest.raises(RuntimeError, match="receipt drifted"):
        result.receipt.validate_integrity()


def test_single_node_resource_accounting_and_complete_gate() -> None:
    result, _direction = _execute_three_tokens()
    resources = gemma3_l3_l4_h4_suffix_vjp_resource_accounting(
        [result.receipt]
    )
    assert resources == {
        "suffix_vjp_node_count": 1,
        "suffix_vjp_primal_vector_count": 1,
        "suffix_vjp_token_directional_derivative_count": 3,
        "suffix_segment_call_count": 13,
        "logit_projection_call_count": 1,
        "h4_dtype_cast_count": 1,
        "vjp_transform_count": 1,
        "vjp_pullback_chunk_call_count": 1,
        "vmap_pullback_call_count": 1,
        "canonical_token_cotangent_coverage_count": 3,
        "canonical_token_cotangent_nonzero_count": 3,
        "canonical_token_cotangent_element_observation_count": 9,
        "full_h4_row_observation_count": 3,
        "support_h4_row_observation_count": 2,
        "outside_support_h4_row_observation_count": 1,
        "direction_coordinate_validation_count": 12,
        "outside_support_direction_zero_validation_count": 4,
        "full_h4_cotangent_coordinate_observation_count": 36,
        "support_h4_cotangent_coordinate_observation_count": 24,
        "outside_support_h4_cotangent_coordinate_observation_count": 12,
        "direction_contraction_coordinate_product_count": 24,
        "full_h4_cotangent_sha256_count": 1,
        "support_h4_cotangent_sha256_count": 1,
        "contracted_directional_chunk_sha256_count": 1,
        "retained_full_h4_cotangent_count": 0,
        "serialized_full_h4_cotangent_count": 0,
        "resource_counts_are_not_FLOPs_or_total_model_compute": True,
    }

    complete = dict(vjp_module._COMPLETE)
    assert complete["suffix_vjp_node_count"] == 64
    assert complete["suffix_vjp_primal_vector_count"] == 64
    assert complete["suffix_vjp_token_directional_derivative_count"] == 3_212
    assert (
        complete["full_h4_cotangent_coordinate_observation_count"]
        == 130_048_000
    )
    assert (
        complete["direction_contraction_coordinate_product_count"]
        == 113_602_560
    )
    require_gemma3_l3_l4_h4_suffix_vjp_complete_panel(complete)
    complete["suffix_vjp_primal_vector_count"] = 65
    with pytest.raises(RuntimeError, match="complete-panel"):
        require_gemma3_l3_l4_h4_suffix_vjp_complete_panel(complete)


def test_canonical_cotangents_are_chunked_at_eight_without_lost_tokens() -> None:
    adapter, sequence, h4, logits, teacher, supervised = (
        _runtime_fixture_length(10)
    )
    runtime = Gemma3L3L4H4SuffixVJPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    support = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64)
    direction = torch.zeros_like(h4, dtype=torch.float64)
    direction[:, support] = 0.125
    with patch(
        "torch.func.vmap", wraps=torch.func.vmap
    ) as vmap:
        result = runtime.execute(
            h4.to(torch.float64).contiguous(),
            direction.contiguous(),
            support_indices=support,
            full_h4=h4,
            full_logits=logits,
        )
    assert vmap.call_count == 2
    assert tuple(
        (chunk.token_start, chunk.token_stop)
        for chunk in result.receipt.chunk_receipts
    ) == ((0, 8), (8, 10))
    assert result.receipt.token_cotangent_coverage_count == 10
    assert result.receipt.token_cotangent_nonzero_count == 10
    assert result.receipt.token_cotangent_element_count == 100
    assert result.receipt.full_h4_cotangent_coordinate_count == 400
    assert result.receipt.support_h4_cotangent_coordinate_count == 200
    assert result.receipt.outside_support_h4_cotangent_coordinate_count == 200
    assert result.receipt.direction_contraction_coordinate_product_count == 200
    jvp_runtime = Gemma3L3L4H4SuffixJVPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    jvp_result = jvp_runtime.execute(
        h4.to(torch.float64).contiguous(),
        direction.contiguous(),
        full_h4=h4,
        full_logits=logits,
    )
    torch.testing.assert_close(
        result.directional_token_teacher_kl,
        jvp_result.directional_token_teacher_kl,
        rtol=2.0e-6,
        atol=2.0e-9,
    )
    result.validate_integrity()


def test_support_must_be_canonical_and_no_full_gradient_tensor_survives() -> None:
    adapter, sequence, h4, logits, teacher, supervised = _runtime_fixture()
    runtime = Gemma3L3L4H4SuffixVJPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher,
        supervised_indices=supervised,
    )
    direction = torch.zeros_like(h4, dtype=torch.float64)
    with pytest.raises(ValueError, match="escape or reorder"):
        runtime.execute(
            h4.to(torch.float64).contiguous(),
            direction.contiguous(),
            support_indices=torch.tensor([2, 0], dtype=torch.int64),
            full_h4=h4,
            full_logits=logits,
        )

    result, _direction = _execute_three_tokens()
    receipt_values = (
        getattr(result.receipt, name)
        for name in result.receipt.__slots__
        if name not in {"chunk_receipts"}
    )
    assert not any(isinstance(value, Tensor) for value in receipt_values)
    for chunk in result.receipt.chunk_receipts:
        assert not any(
            isinstance(getattr(chunk, name), Tensor)
            for name in chunk.__slots__
        )

    def contains_tensor(value: object) -> bool:
        if isinstance(value, Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains_tensor(item) for item in value)
        return False

    assert contains_tensor(result.metadata()) is False
    assert set(result.__slots__) == {
        "primal_token_teacher_kl",
        "directional_token_teacher_kl",
        "receipt",
    }
