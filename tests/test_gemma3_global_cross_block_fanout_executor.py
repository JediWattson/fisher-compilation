from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fisher_graph.adapters import (
    Gemma3CausalLMAdapter,
    module_state_fingerprint,
)
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_global_cross_block_fanout_executor import (
    Gemma3GlobalCrossBlockFanoutExecutor,
    Gemma3GlobalFanoutMLP,
)
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
    ModeKey,
)
from fisher_graph.structured_mlp_cross_block_fanout import (
    create_cross_block_fanout_group,
    create_global_cross_block_fanout_plan,
)


class _GemmaMLP(nn.Module):
    def __init__(self, width: int = 4, intermediate: int = 7) -> None:
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


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }


def test_fused_mlp_replaces_three_consumers_from_two_native_roots() -> None:
    torch.manual_seed(7401)
    source = _GemmaMLP().eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    source_fingerprint = module_state_fingerprint(source)
    consumers = (1, 3, 6)
    mixer = torch.tensor(
        (
            (0.5, -0.25, 0.75),
            (-0.4, 0.8, 0.2),
        ),
        dtype=torch.float64,
    )
    consumer_tensor = torch.tensor(consumers, dtype=torch.long)
    decoder = (
        source.down_proj.weight.detach()
        .double()
        .index_select(1, consumer_tensor)
        @ mixer.T
    )
    compiled = Gemma3GlobalFanoutMLP(
        source,
        consumer_source_indices=consumers,
        fused_decoder=decoder,
        activation="gelu_pytorch_tanh",
    )
    values = torch.randn(2, 3, 4)
    roots = torch.randn(2, 3, 2)
    expected_features = source.features(values)
    expected_features[..., consumer_tensor] = (
        roots.double() @ mixer
    ).to(expected_features.dtype)
    expected = source.down_proj(expected_features)
    actual = compiled(values, roots)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert tuple(compiled.retained_source_indices.tolist()) == (0, 2, 4, 5)
    assert compiled.gate_proj.weight.shape == (4, 4)
    assert compiled.up_proj.weight.shape == (4, 4)
    assert compiled.down_proj.weight.shape == (4, 6)
    assert compiled.down_proj.weight.is_contiguous()
    torch.testing.assert_close(
        compiled.down_proj.weight[:, -2:],
        decoder.to(dtype=compiled.dtype),
    )
    assert set(compiled.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }
    assert sum(parameter.numel() for parameter in source.parameters()) == 84
    assert sum(parameter.numel() for parameter in compiled.parameters()) == 56
    assert module_state_fingerprint(source) == source_fingerprint
    assert not (_storage_pointers(source) & _storage_pointers(compiled))
    assert not torch.equal(
        compiled(values, torch.zeros_like(roots)),
        actual,
    )


def test_fused_mlp_rejects_decoder_that_overflows_runtime_dtype() -> None:
    source = _GemmaMLP().half().eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="runtime model dtype"):
        Gemma3GlobalFanoutMLP(
            source,
            consumer_source_indices=(1,),
            fused_decoder=torch.full(
                (4, 1),
                1e100,
                dtype=torch.float64,
            ),
            activation="gelu_pytorch_tanh",
        )


class _Config:
    model_type = "gemma3_text"
    hidden_size = 4
    intermediate_size = 6
    vocab_size = 17
    num_hidden_layers = 3
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    query_pre_attn_scalar = 4
    max_position_embeddings = 32
    sliding_window = 4
    layer_types = [
        "full_attention",
        "full_attention",
        "full_attention",
    ]
    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }
    }
    rms_norm_eps = 1e-6
    attention_dropout = 0.0
    attention_bias = False
    hidden_activation = "gelu_pytorch_tanh"
    final_logit_softcapping = None
    attn_logit_softcapping = None
    use_bidirectional_attention = False
    _attn_implementation = "eager"


class _Norm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, values: Tensor) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return (normalized * (1.0 + self.weight.float())).to(values.dtype)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = _Norm(2)
        self.k_norm = _Norm(2)

    def forward(self, hidden_states: Tensor, **_: object) -> tuple[Tensor, None]:
        return torch.zeros_like(hidden_states), None


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def features(self, values: Tensor) -> Tensor:
        return F.gelu(
            self.gate_proj(values),
            approximate="tanh",
        ) * self.up_proj(values)

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(self.features(values))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _Norm(4)
        self.self_attn = _Attention()
        self.post_attention_layernorm = _Norm(4)
        self.pre_feedforward_layernorm = _Norm(4)
        self.mlp = _MLP()
        self.post_feedforward_layernorm = _Norm(4)

    def forward(self, hidden_states: Tensor, **kwargs: object) -> Tensor:
        residual = hidden_states
        attention, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        feed_forward = self.mlp(
            self.pre_feedforward_layernorm(hidden_states)
        )
        return residual + self.post_feedforward_layernorm(feed_forward)


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.embed_tokens = nn.Embedding(17, 4)
        self.layers = nn.ModuleList((_Layer(), _Layer(), _Layer()))
        self.norm = _Norm(4)
        self.rotary_emb = nn.Identity()

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _TextModel()
        self.lm_head = nn.Linear(4, 17, bias=False)
        with torch.no_grad():
            for layer in self.model.layers:
                for parameter in layer.self_attn.parameters():
                    parameter.zero_()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        result = self.model(**kwargs)
        return SimpleNamespace(logits=self.lm_head(result.last_hidden_state))


def _adapter() -> Gemma3CausalLMAdapter:
    torch.manual_seed(7402)
    model = _CausalLM().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Gemma3CausalLMAdapter(model)


def _layer_specs(
    adapter: Gemma3CausalLMAdapter,
) -> tuple[CrossBlockLayerSpec, ...]:
    return tuple(
        CrossBlockLayerSpec(
            layer_id=layer.id,
            layer_ordinal=layer.ordinal,
            activation_site=(
                layer.transformer.operator_sites.feed_forward_down_input
            ),
            width=6,
        )
        for layer in adapter.layers
        if layer.transformer is not None
        and layer.transformer.operator_sites is not None
    )


def _mode(
    spec: CrossBlockLayerSpec,
    index: int,
) -> ModeKey:
    return ModeKey(
        layer_ordinal=spec.layer_ordinal,
        layer_id=spec.layer_id,
        activation_site=spec.activation_site,
        mode_index=index,
        fisher_rank=index,
    )


def _executor_fixture():
    adapter = _adapter()
    specs = _layer_specs(adapter)
    anchors = (_mode(specs[0], 0), _mode(specs[0], 1))
    consumers = tuple(_mode(specs[2], index) for index in (0, 2, 4))
    mixer = torch.tensor(
        (
            (0.6, -0.35, 0.2),
            (-0.1, 0.5, 0.8),
        ),
        dtype=torch.float64,
    )
    source_consumer = adapter.source_module(specs[2].layer_id).mlp
    consumer_indices = torch.tensor((0, 2, 4), dtype=torch.long)
    decoder = (
        source_consumer.down_proj.weight.detach()
        .double()
        .index_select(1, consumer_indices)
        @ mixer.T
    )
    group = create_cross_block_fanout_group(
        anchors=anchors,
        consumers=consumers,
        fused_decoder=decoder,
    )
    plan = create_global_cross_block_fanout_plan(
        source_discovery_artifact_sha256="a" * 64,
        source_model_fingerprint=adapter.model_fingerprint(),
        layer_specs=specs,
        groups=(group,),
    )
    return (
        adapter,
        Gemma3GlobalCrossBlockFanoutExecutor(adapter, plan),
        group,
        mixer,
    )


def _batch() -> CalibrationBatch:
    input_ids = torch.tensor(
        (
            (1, 2, 3, 4),
            (5, 6, 0, 0),
        )
    )
    valid = torch.tensor(
        (
            (True, True, True, True),
            (True, True, False, False),
        )
    )
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": valid,
        },
        targets=torch.where(
            valid,
            input_ids.roll(-1, dims=1),
            torch.full_like(input_ids, -100),
        ),
        valid_positions=valid,
        example_ids=("fanout-a", "fanout-b"),
    )


def test_executor_rejects_layer_catalog_width_drift() -> None:
    adapter = _adapter()
    specs = _layer_specs(adapter)
    anchors = (_mode(specs[0], 0),)
    consumers = (_mode(specs[2], 0),)
    group = create_cross_block_fanout_group(
        anchors=anchors,
        consumers=consumers,
        fused_decoder=torch.ones((4, 1), dtype=torch.float64),
    )
    drifted_specs = (
        specs[0],
        replace(specs[1], width=specs[1].width + 1),
        specs[2],
    )
    plan = create_global_cross_block_fanout_plan(
        source_discovery_artifact_sha256="a" * 64,
        source_model_fingerprint=adapter.model_fingerprint(),
        layer_specs=drifted_specs,
        groups=(group,),
    )

    with pytest.raises(ValueError, match="live adapter"):
        Gemma3GlobalCrossBlockFanoutExecutor(adapter, plan)


def test_full_model_fanout_matches_unfused_oracle_and_restores_source() -> None:
    adapter, executor, group, mixer = _executor_fixture()
    batch = _batch()
    valid = batch.valid_positions
    observed: dict[str, Tensor] = {}

    def observe_roots(values: Tensor) -> Tensor:
        observed["roots"] = values[..., (0, 1)].clone()
        return values

    def replace_consumers(values: Tensor) -> Tensor:
        updated = values.clone()
        replacement = (
            observed["roots"].double() @ mixer
        ).to(values.dtype)
        replacement = replacement.masked_fill(~valid.unsqueeze(-1), 0)
        updated[..., (0, 2, 4)] = replacement
        return updated

    oracle = adapter.forward(
        batch.model_inputs,
        interventions={
            group.anchors[0].activation_site: observe_roots,
            group.target_activation_site: replace_consumers,
        },
    )
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()
    merged = executor.run(batch.model_inputs, condition="merged")
    deletion = executor.run(batch.model_inputs, condition="deletion")

    torch.testing.assert_close(
        merged.model_output.logits,
        oracle.logits,
        rtol=3e-5,
        atol=3e-5,
    )
    assert not torch.equal(
        deletion.model_output.logits[valid],
        merged.model_output.logits[valid],
    )
    assert merged.group_count == 1
    assert merged.consumer_count == 3
    assert merged.native_root_count == 2
    assert merged.fused_anchor_input_count == 2
    assert merged.native_removed_learned_parameters == 36
    assert merged.fused_decoder_coefficients == 8
    assert merged.net_stored_coefficient_savings == 28
    assert merged.valid_tokens == 6
    assert merged.logical_linear_macs_native_removed == 216
    assert merged.logical_linear_macs_fused_decoder == 48
    assert merged.net_logical_macs_saved == 168
    assert merged.peak_live_root_scalars_per_token == 2
    assert (
        merged.candidate_whole_model_learned_parameters
        == merged.source_whole_model_learned_parameters - 28
    )
    target_compiled = executor.compiled_mlps["2"]
    assert target_compiled.gate_proj.weight.shape == (3, 4)
    assert target_compiled.up_proj.weight.shape == (3, 4)
    assert target_compiled.down_proj.weight.shape == (4, 5)
    assert all(
        "mixer" not in name.lower()
        and "decoder" not in name.lower()
        and "consumer" not in name.lower()
        for name in executor.state_dict()
    )
    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == source_mlps
    assert adapter.model_fingerprint() == source_fingerprint


def test_full_model_overlay_restores_source_after_failure() -> None:
    adapter, executor, _, _ = _executor_fixture()
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()

    def fail_target(
        _module: nn.Module,
        _args: tuple[object, ...],
    ) -> None:
        raise RuntimeError("sentinel fused target failure")

    handle = executor.compiled_mlps["2"].gate_proj.register_forward_pre_hook(
        fail_target
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="sentinel fused target failure",
        ):
            executor.run(batch.model_inputs)
    finally:
        handle.remove()
    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == source_mlps
    assert adapter.model_fingerprint() == source_fingerprint
