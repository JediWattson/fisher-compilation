import unittest
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from fisher_graph import ActivationTrace
from fisher_graph.adapters import Gemma3CausalLMAdapter


class FakeGemma3Config:
    model_type = "gemma3_text"
    hidden_size = 8
    intermediate_size = 16
    vocab_size = 19
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 2
    query_pre_attn_scalar = 4
    max_position_embeddings = 32
    sliding_window = 3
    layer_types = ["sliding_attention", "full_attention"]
    rope_parameters = {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        },
        "full_attention": {
            "rope_type": "linear",
            "rope_theta": 1_000_000.0,
            "factor": 2.0,
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


class FakeRotaryEmbedding(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        layer_type: str,
    ) -> tuple[Tensor, Tensor]:
        del layer_type
        values = position_ids.to(hidden_states.dtype).unsqueeze(-1)
        return values.cos(), values.sin()


class FakeLegacyRotaryEmbedding(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = position_ids.to(hidden_states.dtype).unsqueeze(-1)
        return values.cos(), values.sin()


class FakeGemma3Layer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.last_attention_mask: Tensor | None = None
        self.last_position_ids: Tensor | None = None

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: object | None = None,
        **kwargs: object,
    ) -> Tensor:
        del position_embeddings, past_key_values, kwargs
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        return hidden_states + torch.tanh(self.projection(hidden_states))


class FakeLegacyGemma3Layer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.saw_global = False
        self.saw_local = False

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_embeddings_global: tuple[Tensor, Tensor],
        position_embeddings_local: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        del attention_mask, position_ids, kwargs
        self.saw_global = bool(position_embeddings_global)
        self.saw_local = bool(position_embeddings_local)
        return hidden_states + torch.tanh(self.projection(hidden_states))


class FakeGemma3TextModel(nn.Module):
    def __init__(self, config: FakeGemma3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            FakeGemma3Layer(config.hidden_size)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.rotary_emb = FakeRotaryEmbedding()

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs: object,
    ) -> SimpleNamespace:
        del use_cache, return_dict, kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one input representation is required")
        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        batch_size, sequence_length = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)
        for layer, layer_type in zip(
            self.layers,
            self.config.layer_types,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=self.rotary_emb(
                    hidden_states,
                    position_ids,
                    layer_type,
                ),
                position_ids=position_ids,
                past_key_values=None,
            )
        return SimpleNamespace(last_hidden_state=self.norm(hidden_states))


class FakeGemma3ForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FakeGemma3Config()
        self.model = FakeGemma3TextModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        output = self.model(**kwargs)
        logits = self.lm_head(output.last_hidden_state)
        cap = self.config.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap
        return SimpleNamespace(logits=logits)


class ZeroGemma3Layer(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        return torch.zeros_like(hidden_states)


class Gemma3CausalLMAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(103)
        self.model = FakeGemma3ForCausalLM().eval()
        self.adapter = Gemma3CausalLMAdapter(self.model)
        self.input_ids = torch.tensor(
            [
                [0, 0, 2, 3],
                [0, 4, 5, 6],
            ],
            dtype=torch.long,
        )
        self.attention_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, True, True, True],
            ]
        )

    def test_describes_text_gemma_layers_attention_and_sites(self) -> None:
        sequence = self.adapter.sequence_spec
        self.assertEqual(sequence.length_policy, "bounded_dynamic")
        self.assertEqual(sequence.maximum_length, 32)
        self.assertTrue(sequence.mask.causal)
        self.assertEqual(sequence.mask.padding_side, "either")
        self.assertEqual(sequence.mask.representation, "adapter_owned")
        self.assertFalse(sequence.supports_decode)

        self.assertEqual(
            tuple(layer.id for layer in self.adapter.layers),
            ("layer.0", "layer.1"),
        )
        local = self.adapter.layers[0].attention
        global_attention = self.adapter.layers[1].attention
        assert local is not None
        assert global_attention is not None
        self.assertEqual(local.kind, "sliding_causal")
        self.assertEqual(local.query_heads, 4)
        self.assertEqual(local.key_value_heads, 2)
        self.assertEqual(local.head_dimension, 2)
        self.assertEqual(local.query_scale, 0.5)
        self.assertTrue(local.qk_norm)
        self.assertEqual(local.window_size, 3)
        assert local.rope is not None
        self.assertEqual(local.rope.theta, 10_000.0)
        self.assertEqual(global_attention.kind, "global_causal")
        self.assertIsNone(global_attention.window_size)
        assert global_attention.rope is not None
        self.assertEqual(global_attention.rope.scaling_kind, "linear")
        self.assertEqual(global_attention.rope.scaling_factor, 2.0)
        transformer = self.adapter.layers[0].transformer
        assert transformer is not None
        self.assertEqual(
            transformer.residual_layout,
            "sequential_attention_then_feed_forward_residual",
        )
        self.assertEqual(
            transformer.attention_input_norm.kind,
            "rms_norm",
        )
        self.assertEqual(
            transformer.attention_input_norm.scale_parameterization,
            "unit_offset",
        )
        self.assertEqual(
            transformer.attention_input_norm.compute_dtype,
            "float32",
        )
        self.assertEqual(transformer.qk_norm.width, 2)
        self.assertEqual(
            transformer.feed_forward.kind,
            "gated_multiplicative",
        )
        self.assertEqual(
            transformer.feed_forward.intermediate_width,
            16,
        )
        self.assertEqual(
            tuple(stage.kind for stage in transformer.stages),
            ("attention", "feed_forward"),
        )
        operator_sites = transformer.operator_sites
        assert operator_sites is not None
        self.assertEqual(
            operator_sites.values(),
            (
                "layer.0.attention.query_projection",
                "layer.0.attention.query_normalized",
                "layer.0.attention.key_projection",
                "layer.0.attention.key_normalized",
                "layer.0.attention.value_projection",
                "layer.0.attention.context",
                "layer.0.mlp.gate_projection",
                "layer.0.mlp.up_projection",
                "layer.0.mlp.down_input",
            ),
        )

        sites = {site.id: site for site in self.adapter.activation_sites}
        self.assertEqual(
            tuple(sites),
            (
                "layer.0.input",
                "layer.0.attention.normalized_input",
                "layer.0.attention.operator_output",
                "layer.0.attention.delta",
                "layer.0.attention.query_projection",
                "layer.0.attention.query_normalized",
                "layer.0.attention.key_projection",
                "layer.0.attention.key_normalized",
                "layer.0.attention.value_projection",
                "layer.0.attention.context",
                "layer.0.post_attention",
                "layer.0.mlp.normalized_input",
                "layer.0.mlp.operator_output",
                "layer.0.mlp.delta",
                "layer.0.mlp.gate_projection",
                "layer.0.mlp.up_projection",
                "layer.0.mlp.down_input",
                "layer.0.output",
                "layer.1.input",
                "layer.1.attention.normalized_input",
                "layer.1.attention.operator_output",
                "layer.1.attention.delta",
                "layer.1.attention.query_projection",
                "layer.1.attention.query_normalized",
                "layer.1.attention.key_projection",
                "layer.1.attention.key_normalized",
                "layer.1.attention.value_projection",
                "layer.1.attention.context",
                "layer.1.post_attention",
                "layer.1.mlp.normalized_input",
                "layer.1.mlp.operator_output",
                "layer.1.mlp.delta",
                "layer.1.mlp.gate_projection",
                "layer.1.mlp.up_projection",
                "layer.1.mlp.down_input",
                "layer.1.output",
                "final_norm",
                "logits",
            ),
        )
        self.assertFalse(sites["layer.0.post_attention"].intervenable)
        for site_id in operator_sites.values()[:-1]:
            self.assertFalse(sites[site_id].intervenable)
        self.assertTrue(
            sites[operator_sites.feed_forward_down_input].intervenable
        )
        self.assertEqual(
            sites["layer.1.input"].alias_of,
            "layer.0.output",
        )
        self.assertEqual(
            self.adapter.default_fisher_sites,
            (
                "layer.0.input",
                "layer.0.output",
                "layer.1.output",
            ),
        )
        block = self.adapter.plan_layer_block(0, 1)
        self.assertEqual(
            block.activation_sites,
            (
                "layer.0.input",
                "layer.0.output",
                "layer.1.output",
            ),
        )
        self.assertEqual(
            block.transitions,
            (
                ("layer.0.input", "layer.0.output"),
                ("layer.0.output", "layer.1.output"),
            ),
        )

    def test_legacy_config_preserves_distinct_local_and_global_rope(self) -> None:
        model = FakeGemma3ForCausalLM()
        model.config.rope_parameters = None
        model.config.rope_scaling = None
        model.config.rope_theta = 1_000_000.0
        model.config.rope_local_base_freq = 10_000.0
        adapter = Gemma3CausalLMAdapter(model)
        local = adapter.layers[0].attention
        global_attention = adapter.layers[1].attention
        assert local is not None and local.rope is not None
        assert global_attention is not None
        assert global_attention.rope is not None

        self.assertEqual(local.rope.theta, 10_000.0)
        self.assertEqual(global_attention.rope.theta, 1_000_000.0)

    def test_forward_hooks_capture_aliases_and_retain_gradients(self) -> None:
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        expected = self.model(**inputs).logits
        run = self.adapter.forward(
            inputs,
            capture_sites=(
                "layer.0.input",
                "layer.0.output",
                "layer.1.input",
                "layer.1.output",
                "final_norm",
                "logits",
            ),
            retain_gradients=True,
        )

        torch.testing.assert_close(run.logits, expected)
        self.assertIs(
            run.activations["layer.0.output"],
            run.activations["layer.1.input"],
        )
        run.logits.square().sum().backward()
        for name in (
            "layer.0.input",
            "layer.0.output",
            "layer.1.output",
            "final_norm",
        ):
            self.assertIsNotNone(run.activations[name].grad, name)

    def test_interventions_propagate_without_changing_layer_call_shape(self) -> None:
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        baseline = self.adapter.forward(inputs)
        intervened = self.adapter.forward(
            inputs,
            capture_sites=("layer.0.input", "logits"),
            interventions={
                "layer.0.input": torch.zeros_like,
                "logits": lambda value: value + 1,
            },
        )

        self.assertTrue(
            torch.equal(
                intervened.activations["layer.0.input"],
                torch.zeros_like(
                    intervened.activations["layer.0.input"]
                ),
            )
        )
        self.assertFalse(torch.equal(intervened.logits, baseline.logits))
        self.assertEqual(intervened.logits.shape, baseline.logits.shape)
        for layer in self.model.model.layers:
            assert isinstance(layer, FakeGemma3Layer)
            self.assertEqual(
                tuple(layer.last_position_ids.shape),
                tuple(self.input_ids.shape),
            )

    def test_hooks_are_removed_after_forward(self) -> None:
        hook_counts = tuple(
            (len(layer._forward_pre_hooks), len(layer._forward_hooks))
            for layer in self.model.model.layers
        )
        norm_hooks = len(self.model.model.norm._forward_hooks)
        self.adapter.forward(
            {"input_ids": self.input_ids},
            capture_sites=("layer.0.output",),
        )
        self.assertEqual(
            tuple(
                (len(layer._forward_pre_hooks), len(layer._forward_hooks))
                for layer in self.model.model.layers
            ),
            hook_counts,
        )
        self.assertEqual(
            len(self.model.model.norm._forward_hooks),
            norm_hooks,
        )

    def test_sequence_context_tracks_mask_positions_and_origin(self) -> None:
        position_ids = torch.tensor([[5, 6, 7, 8]])
        sequence = self.adapter.prepare_sequence(
            {
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
                "position_ids": position_ids,
            }
        )

        self.assertEqual((sequence.batch_size, sequence.query_length), (2, 4))
        self.assertTrue(
            torch.equal(sequence.query_valid_mask, self.attention_mask)
        )
        self.assertTrue(
            torch.equal(
                sequence.logical_positions,
                position_ids.expand(2, -1),
            )
        )
        self.assertTrue(sequence.input_origin.attention_mask_supplied)
        self.assertTrue(sequence.input_origin.position_ids_supplied)

        with self.assertRaisesRegex(
            ValueError,
            "provide an explicit attention_mask",
        ):
            self.adapter.prepare_sequence(
                {
                    "input_ids": self.input_ids,
                    "position_ids": torch.tensor(
                        [[0, 1, 0, 1]],
                    ),
                }
            )

        with self.assertRaisesRegex(ValueError, "cached decode"):
            self.adapter.prepare_sequence(
                {"input_ids": self.input_ids[:, :1]},
                phase="decode",
                cache_state=object(),
            )

    def test_explicit_segment_primitives_reproduce_fake_forward(self) -> None:
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        expected = self.adapter.forward(inputs).logits
        sequence = self.adapter.prepare_sequence(inputs)
        trace = ActivationTrace(retain_grad=False)
        current = self.adapter.embed(
            inputs,
            sequence,
            trace=trace,
        ).hidden_states
        for segment in self.adapter.segments:
            current = self.adapter.run_segment(
                segment,
                current,
                sequence,
                trace=trace,
            ).hidden_states
        logits = self.adapter.project_logits(
            current,
            sequence,
            trace=trace,
        )

        torch.testing.assert_close(logits, expected)
        self.assertEqual(
            trace.names,
            (
                "layer.0.input",
                "layer.0.output",
                "layer.1.input",
                "layer.1.output",
                "final_norm",
                "logits",
            ),
        )
        first_layer = self.model.model.layers[0]
        assert isinstance(first_layer, FakeGemma3Layer)
        assert first_layer.last_attention_mask is not None
        self.assertEqual(
            tuple(first_layer.last_attention_mask.shape),
            (2, 1, 4, 4),
        )
        minimum = torch.finfo(
            first_layer.last_attention_mask.dtype
        ).min
        self.assertEqual(
            first_layer.last_attention_mask[0, 0, 3, 0].item(),
            minimum,
        )

    def test_segment_supports_released_dual_rotary_transformers_abi(
        self,
    ) -> None:
        model = FakeGemma3ForCausalLM()
        model.model.rotary_emb = FakeLegacyRotaryEmbedding()
        model.model.rotary_emb_local = FakeLegacyRotaryEmbedding()
        model.model.layers = nn.ModuleList(
            FakeLegacyGemma3Layer(model.config.hidden_size)
            for _ in range(model.config.num_hidden_layers)
        )
        adapter = Gemma3CausalLMAdapter(model)
        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        sequence = adapter.prepare_sequence(inputs)
        hidden_states = adapter.embed(inputs, sequence).hidden_states

        result = adapter.run_segment(
            adapter.segments[0],
            hidden_states,
            sequence,
        )

        self.assertEqual(result.hidden_states.shape, hidden_states.shape)
        layer = model.model.layers[0]
        assert isinstance(layer, FakeLegacyGemma3Layer)
        self.assertTrue(layer.saw_global)
        self.assertTrue(layer.saw_local)

    def test_replaced_segments_restores_native_layers(self) -> None:
        original = self.model.model.layers[0]
        replacement = ZeroGemma3Layer()
        with self.adapter.replaced_segments({"layer.0": replacement}):
            self.assertIs(self.model.model.layers[0], replacement)
        self.assertIs(self.model.model.layers[0], original)

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with self.adapter.replaced_segments({"layer.0": replacement}):
                raise RuntimeError("deliberate")
        self.assertIs(self.model.model.layers[0], original)

    def test_rejects_multimodal_gemma_config_without_transformers_import(
        self,
    ) -> None:
        model = FakeGemma3ForCausalLM()
        model.config.model_type = "gemma3"
        with self.assertRaisesRegex(TypeError, "multimodal"):
            Gemma3CausalLMAdapter(model)

    def test_optional_real_transformers_tiny_model_segment_parity(
        self,
    ) -> None:
        try:
            from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        except ImportError:
            self.skipTest("optional Transformers dependency is not installed")

        torch.manual_seed(991)
        config = Gemma3TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        model = Gemma3ForCausalLM(config).eval()
        adapter = Gemma3CausalLMAdapter(model)
        inputs = {
            "input_ids": torch.tensor([[1, 7, 3, 9, 2]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        }

        expected = adapter.forward(inputs).logits
        sequence = adapter.prepare_sequence(inputs)
        hidden_states = adapter.embed(inputs, sequence).hidden_states
        for segment in adapter.segments:
            hidden_states = adapter.run_segment(
                segment,
                hidden_states,
                sequence,
            ).hidden_states
        actual = adapter.project_logits(hidden_states, sequence)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
