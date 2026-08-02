from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.gemma3_l3_l4_h4_suffix_jvp_runtime import (
    GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS,
    Gemma3L3L4H4SuffixJVPRuntime,
    gemma3_l3_l4_h4_discrete_cast_interval_stats,
)


class _Config:
    model_type = "gemma3_text"
    hidden_size = 4
    intermediate_size = 8
    vocab_size = 7
    num_hidden_layers = 18
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    query_pre_attn_scalar = 2
    max_position_embeddings = 32
    sliding_window = 4
    layer_types = [
        "sliding_attention" if index % 2 == 0 else "full_attention"
        for index in range(18)
    ]
    rope_parameters = {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
    }
    rms_norm_eps = 1e-6
    attention_dropout = 0.0
    attention_bias = False
    hidden_activation = "gelu_pytorch_tanh"
    final_logit_softcapping = 7.0
    attn_logit_softcapping = None
    use_bidirectional_attention = False
    _attn_implementation = "eager"


class _Rotary(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        layer_type: str,
    ) -> tuple[Tensor, Tensor]:
        del layer_type
        values = position_ids.to(hidden_states.dtype).unsqueeze(-1)
        return values.cos(), values.sin()


class _Attention(nn.Module):
    def __init__(self, width: int, *, sliding: bool) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.is_sliding = sliding

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor]:
        del position_embeddings, attention_mask, kwargs
        return (self.projection(hidden_states),)


class _MLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch.tanh(self.projection(hidden_states))


class _Layer(nn.Module):
    def __init__(self, width: int, *, sliding: bool) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.self_attn = _Attention(width, sliding=sliding)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.pre_feedforward_layernorm = nn.LayerNorm(width)
        self.mlp = _MLP(width)
        self.post_feedforward_layernorm = nn.LayerNorm(width)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        position_embeddings_global: tuple[Tensor, Tensor] | None = None,
        position_embeddings_local: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        if position_embeddings is None:
            position_embeddings = (
                position_embeddings_local
                if self.self_attn.is_sliding
                else position_embeddings_global
            )
        assert position_embeddings is not None
        residual = hidden_states
        attention = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )[0]
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        hidden_states = self.mlp(
            self.pre_feedforward_layernorm(hidden_states)
        )
        return residual + self.post_feedforward_layernorm(hidden_states)


class _TextModel(nn.Module):
    def __init__(self, config: _Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            _Layer(
                config.hidden_size,
                sliding=kind == "sliding_attention",
            )
            for kind in config.layer_types
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = _Rotary()

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device,
            ).unsqueeze(0).expand(hidden_states.shape[0], -1)
        for layer, kind in zip(
            self.layers,
            self.config.layer_types,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                position_embeddings=self.rotary_emb(
                    hidden_states,
                    position_ids,
                    kind,
                ),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return SimpleNamespace(last_hidden_state=self.norm(hidden_states))


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _TextModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        hidden = self.model(**kwargs).last_hidden_state
        logits = self.lm_head(hidden)
        cap = self.config.final_logit_softcapping
        return SimpleNamespace(logits=torch.tanh(logits / cap) * cap)


def _runtime_fixture() -> tuple[
    Gemma3CausalLMAdapter,
    object,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    torch.manual_seed(2601)
    adapter = Gemma3CausalLMAdapter(_CausalLM().float().eval())
    model_inputs = {
        "input_ids": torch.tensor([[1, 3, 2]], dtype=torch.int64),
        "attention_mask": torch.ones((1, 3), dtype=torch.int64),
    }
    with torch.no_grad():
        full = adapter.forward(
            model_inputs,
            capture_sites={"layer.4.output"},
        )
    full_h4 = full.activations["layer.4.output"].detach().contiguous()
    full_logits = full.logits.detach().contiguous()
    teacher_logits = full_logits.clone()
    teacher_logits[..., 0] += 0.4
    teacher_logits[..., 2] -= 0.2
    supervised_indices = torch.tensor(
        [[0, 0], [0, 1], [0, 2]],
        dtype=torch.int64,
    )
    return (
        adapter,
        full.sequence,
        full_h4,
        full_logits,
        teacher_logits,
        supervised_indices,
    )


def _manual_token_kl(
    adapter: Gemma3CausalLMAdapter,
    sequence: object,
    path_h4_f64: Tensor,
    teacher_logits: Tensor,
    supervised_indices: Tensor,
) -> Tensor:
    hidden = path_h4_f64.to(dtype=torch.float32)
    for segment in adapter.segments[5:18]:
        hidden = adapter.run_segment(segment, hidden, sequence).hidden_states
    logits = adapter.project_logits(hidden, sequence)
    indices = supervised_indices.to(logits.device)
    candidate = logits[indices[:, 0], indices[:, 1]].to(torch.float64)
    teacher = teacher_logits[indices[:, 0], indices[:, 1]].to(torch.float64)
    teacher_log = F.log_softmax(teacher, dim=-1)
    candidate_log = F.log_softmax(candidate, dim=-1)
    return (teacher_log.exp() * (teacher_log - candidate_log)).sum(dim=-1)


def test_authenticated_suffix_jvp_replays_exact_primal_and_direction() -> None:
    (
        adapter,
        sequence,
        full_h4,
        full_logits,
        teacher_logits,
        supervised_indices,
    ) = _runtime_fixture()
    runtime = Gemma3L3L4H4SuffixJVPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
    )
    path_h4_f64 = full_h4.to(torch.float64).contiguous()
    direction_h4_f64 = torch.randn_like(path_h4_f64).contiguous()

    with (
        patch.object(
            adapter,
            "run_segment",
            wraps=adapter.run_segment,
        ) as run_segment,
        patch.object(
            adapter,
            "project_logits",
            wraps=adapter.project_logits,
        ) as project_logits,
    ):
        result = runtime.execute(
            path_h4_f64,
            direction_h4_f64,
            full_h4=full_h4,
            full_logits=full_logits,
        )
    assert run_segment.call_count == 13
    assert project_logits.call_count == 1
    assert result.receipt.suffix_segment_ids == (
        GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS
    )
    assert result.receipt.suffix_segment_call_count == 13
    assert result.receipt.logit_projection_call_count == 1
    assert result.receipt.h4_dtype_cast_count == 1
    assert result.receipt.ad_mechanism == "torch.func.jvp.forward_mode"
    assert result.receipt.jvp_strict is True
    assert result.receipt.jvp_has_aux is True
    assert result.primal_token_teacher_kl.dtype == torch.float64
    assert result.directional_token_teacher_kl.dtype == torch.float64
    assert result.primal_token_teacher_kl_sha256 == (
        result.receipt.full_token_teacher_kl_sha256
    )
    assert result.path_h4_sha256 == result.receipt.path_h4_sha256
    assert result.full_h4_sha256 == result.receipt.cast_h4_sha256
    assert result.full_logits_sha256 == result.receipt.suffix_logits_sha256
    assert result.metadata()["artifact_sha256"] == result.artifact_sha256
    result.validate_integrity()

    leaf = path_h4_f64.detach().clone().requires_grad_(True)
    token_kl = _manual_token_kl(
        adapter,
        sequence,
        leaf,
        teacher_logits,
        supervised_indices,
    )
    manual_directional = []
    for index in range(token_kl.numel()):
        gradient = torch.autograd.grad(
            token_kl[index],
            leaf,
            retain_graph=index + 1 < token_kl.numel(),
        )[0]
        manual_directional.append(
            torch.sum(gradient.to(torch.float64) * direction_h4_f64)
        )
    torch.testing.assert_close(
        result.primal_token_teacher_kl,
        token_kl.detach(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.directional_token_teacher_kl,
        torch.stack(manual_directional),
        rtol=2.0e-6,
        atol=2.0e-9,
    )

    result.primal_token_teacher_kl[0] += 1.0
    with pytest.raises(RuntimeError, match="tensor payload drifted"):
        result.validate_integrity()


def test_suffix_runtime_rejects_noncanonical_inputs_and_state_drift() -> None:
    (
        adapter,
        sequence,
        full_h4,
        full_logits,
        teacher_logits,
        supervised_indices,
    ) = _runtime_fixture()
    runtime = Gemma3L3L4H4SuffixJVPRuntime(
        adapter,
        sequence,
        teacher_logits=teacher_logits,
        supervised_indices=supervised_indices,
    )
    path = full_h4.to(torch.float64).contiguous()
    direction = torch.ones_like(path)
    with pytest.raises(ValueError, match="float64 peers"):
        runtime.execute(
            path.to(torch.float32),
            direction,
            full_h4=full_h4,
            full_logits=full_logits,
        )
    with pytest.raises(ValueError, match="does not cast bitwise"):
        runtime.execute(
            path + 1.0,
            direction,
            full_h4=full_h4,
            full_logits=full_logits,
        )
    corrupted_logits = full_logits.clone()
    corrupted_logits[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="suffix logits"):
        runtime.execute(
            path,
            direction,
            full_h4=full_h4,
            full_logits=corrupted_logits,
        )

    with torch.no_grad():
        next(adapter.module.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="model state drifted"):
        runtime.validate_integrity(deep=True)

    adapter.module.eval()
    adapter.module.model.layers[0].train()
    assert adapter.module.training is False
    with pytest.raises(ValueError, match="complete Gemma module hierarchy"):
        Gemma3L3L4H4SuffixJVPRuntime(
            adapter,
            sequence,
            teacher_logits=teacher_logits,
            supervised_indices=supervised_indices,
        )
    adapter.module.eval()
    adapter.module.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager attention"):
        Gemma3L3L4H4SuffixJVPRuntime(
            adapter,
            sequence,
            teacher_logits=teacher_logits,
            supervised_indices=supervised_indices,
        )


def test_discrete_cast_interval_stats_bind_collisions_energy_and_kl() -> None:
    ideal_left = torch.tensor(
        [[[0.0, 1.0, -2.0], [3.0, 4.0, 5.0]]],
        dtype=torch.float64,
    )
    ideal_right = ideal_left.clone()
    ideal_right[0, 0, 1] += 1.0e-9  # Ideal change erased by float32.
    ideal_right[0, 1, 0] += 1.0e-4  # Ideal change retained by float32.
    ideal_right[0, 1, 2] += 0.5  # Ideal change retained by float32.
    left = ideal_left.to(torch.float32)
    right = ideal_right.to(torch.float32)
    left_kl = torch.tensor([0.25, 0.5], dtype=torch.float64)
    right_kl = torch.tensor([0.5, 0.125], dtype=torch.float64)
    stats = gemma3_l3_l4_h4_discrete_cast_interval_stats(
        left,
        right,
        ideal_left_h4_f64=ideal_left,
        ideal_right_h4_f64=ideal_right,
        left_path_fraction=0.2,
        right_path_fraction=0.5,
        left_token_teacher_kl=left_kl,
        right_token_teacher_kl=right_kl,
    )
    assert stats.coordinate_count == 6
    assert stats.ideal_changed_coordinate_count == 3
    assert stats.live_changed_coordinate_count == 2
    assert stats.preserved_change_coordinate_count == 2
    assert stats.cast_collision_coordinate_count == 1
    assert stats.static_coordinate_count == 3
    assert stats.unchanged_live_coordinate_count == 4
    assert stats.interval_width == pytest.approx(0.3)
    assert stats.ideal_displacement_squared_l2 > 0.25
    assert stats.live_displacement_squared_l2 > 0.25
    assert stats.token_count == 2
    assert stats.token_teacher_kl_delta_mean == pytest.approx(-0.0625)
    assert stats.token_teacher_kl_normalized_secant_mean == pytest.approx(
        -0.0625 / 0.3
    )
    assert stats.token_teacher_kl_delta_sha256 is not None
    assert stats.token_teacher_kl_normalized_secant_sha256 is not None
    metadata = stats.metadata()
    assert metadata["serialized_h4_or_token_kl_tensors"] is False
    assert metadata["cast_collision_coordinate_count"] == 1
    assert "collision_coordinate_count" not in metadata
    assert "token_teacher_kl_secant_sha256" not in metadata
    stats.validate_integrity()

    bad_right = right.clone()
    bad_right[0, 0, 0] = torch.nextafter(
        bad_right[0, 0, 0],
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="differ from ideal endpoint casts"):
        gemma3_l3_l4_h4_discrete_cast_interval_stats(
            left,
            bad_right,
            ideal_left_h4_f64=ideal_left,
            ideal_right_h4_f64=ideal_right,
            left_path_fraction=0.2,
            right_path_fraction=0.5,
        )

    object.__setattr__(stats, "ideal_changed_coordinate_count", 2)
    with pytest.raises(RuntimeError, match="receipt drifted"):
        stats.validate_integrity()


def test_suffix_runtime_requires_canonical_supervised_grid() -> None:
    (
        adapter,
        sequence,
        _full_h4,
        _full_logits,
        teacher_logits,
        _supervised_indices,
    ) = _runtime_fixture()
    with pytest.raises(ValueError, match="escape or reorder"):
        Gemma3L3L4H4SuffixJVPRuntime(
            adapter,
            sequence,
            teacher_logits=teacher_logits,
            supervised_indices=torch.tensor(
                [[0, 2], [0, 1]],
                dtype=torch.int64,
            ),
        )
