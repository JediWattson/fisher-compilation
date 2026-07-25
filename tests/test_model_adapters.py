import unittest
from dataclasses import replace

import torch
from torch import Tensor

from fisher_graph import (
    ActivationTrace,
    LayerExecutor,
    ToyTransformer,
    TransformerConfig,
)
from fisher_graph.adapters import (
    ToyTransformerAdapter,
    as_model_adapter,
    module_state_fingerprint,
)
from fisher_graph.compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
)
from fisher_graph.modes import (
    build_fisher_mode_bases,
    collect_activation_score_gradients,
    collect_adapter_score_gradients,
    extract_modal_jacobian,
    extract_segment_modal_jacobian,
)


class ZeroLayer(LayerExecutor):
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace=None,
        prefix: str,
    ) -> Tensor:
        del attention_mask, trace, prefix
        return torch.zeros_like(hidden_states)


class ModelAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.model = ToyTransformer(
            TransformerConfig(
                vocab_size=17,
                max_sequence_length=8,
                d_model=8,
                n_heads=2,
                n_layers=2,
                d_ff=12,
                dropout=0.0,
            )
        ).eval()
        self.adapter = ToyTransformerAdapter(self.model)

    def test_toy_adapter_describes_model_sites_layers_and_segments(self) -> None:
        adapter = self.adapter

        self.assertIs(adapter.module, self.model)
        self.assertIs(as_model_adapter(adapter), adapter)
        wrapped = as_model_adapter(self.model)
        self.assertIsInstance(wrapped, ToyTransformerAdapter)
        self.assertIs(wrapped.module, self.model)

        sequence = adapter.sequence_spec
        self.assertEqual(sequence.length_policy, "bounded_dynamic")
        self.assertEqual(sequence.minimum_length, 1)
        self.assertEqual(sequence.maximum_length, 8)
        self.assertTrue(sequence.mask.causal)
        self.assertEqual(sequence.mask.padding_side, "sparse")
        self.assertEqual(sequence.position_kind, "learned_absolute")
        self.assertTrue(sequence.supports_prefill)
        self.assertFalse(sequence.supports_decode)
        self.assertEqual(sequence.cache_kind, "none")

        self.assertEqual(tuple(layer.id for layer in adapter.layers), (
            "layer.0",
            "layer.1",
        ))
        self.assertEqual(adapter.layers[0].input_site, "layer.0.input")
        self.assertEqual(adapter.layers[0].output_site, "layer.0.output")
        self.assertEqual(adapter.layers[1].input_site, "layer.0.output")
        self.assertEqual(adapter.layers[1].output_site, "layer.1.output")
        for layer in adapter.layers:
            self.assertEqual(layer.residual_width, 8)
            self.assertEqual(layer.kind, "pre_norm_decoder")
            self.assertIsNotNone(layer.attention)
            assert layer.attention is not None
            self.assertEqual(layer.attention.kind, "global_causal")
            self.assertEqual(layer.attention.query_heads, 2)
            self.assertEqual(layer.attention.key_value_heads, 2)
            self.assertEqual(layer.attention.head_dimension, 4)
            self.assertEqual(layer.attention.query_scale, 0.5)
            self.assertFalse(layer.attention.qk_norm)
            self.assertEqual(layer.attention.cache_kind, "none")

        self.assertEqual(tuple(segment.id for segment in adapter.segments), (
            "layer.0",
            "layer.1",
        ))
        self.assertEqual(
            adapter.default_fisher_sites,
            (
                "layer.0.input",
                "layer.0.post_attention",
                "layer.0.output",
                "layer.1.post_attention",
                "layer.1.output",
                "final_norm",
            ),
        )
        self.assertEqual(adapter.segments[0].layer_ids, ("layer.0",))
        self.assertEqual(adapter.segments[1].layer_ids, ("layer.1",))
        self.assertEqual(adapter.segments[1].input_site, "layer.0.output")

        sites = {site.id: site for site in adapter.activation_sites}
        self.assertIn("embedding.output", sites)
        self.assertIn("layer.0.attention.probabilities", sites)
        self.assertIn("layer.1.output", sites)
        self.assertIn("logits", sites)
        self.assertTrue(sites["layer.0.output"].modal_eligible)
        self.assertFalse(
            sites["layer.0.attention.probabilities"].modal_eligible
        )
        self.assertEqual(
            sites["layer.1.input"].alias_of,
            "layer.0.output",
        )

        block = adapter.plan_layer_block(0, 1)
        self.assertEqual(block.layer_ids, ("layer.0", "layer.1"))
        self.assertEqual(block.layer_ordinals, (0, 1))
        self.assertEqual(
            block.activation_sites,
            (
                "layer.0.input",
                "layer.0.output",
                "layer.1.output",
            ),
        )
        self.assertEqual(block.widths, (8, 8, 8))
        self.assertEqual(block.leaf_activation_name, "layer.0.input")
        self.assertEqual(
            block.transitions,
            (
                ("layer.0.input", "layer.0.output"),
                ("layer.0.output", "layer.1.output"),
            ),
        )

    def test_layer_block_plan_validates_ordinal_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            self.adapter.plan_layer_block(1, 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.adapter.plan_layer_block(0, 2)

    def test_sequence_context_retains_normalized_input_origin(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        omitted = self.adapter.prepare_sequence({"input_ids": input_ids})
        explicit = self.adapter.prepare_sequence(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )

        torch.testing.assert_close(
            omitted.query_valid_mask,
            explicit.query_valid_mask,
        )
        torch.testing.assert_close(
            omitted.logical_positions,
            explicit.logical_positions,
        )
        self.assertFalse(omitted.input_origin.attention_mask_supplied)
        self.assertTrue(explicit.input_origin.attention_mask_supplied)
        self.assertFalse(omitted.input_origin.position_ids_supplied)
        self.assertFalse(explicit.input_origin.position_ids_supplied)

    def test_embedding_segment_and_head_primitives_reproduce_forward(self) -> None:
        model_inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[True, True, True, False]]
            ),
        }
        baseline = self.adapter.forward(
            model_inputs,
            capture_sites=(
                "embedding.output",
                "layer.0.input",
                "layer.1.output",
                "final_norm",
                "logits",
            ),
        )
        sequence = self.adapter.prepare_sequence(model_inputs)
        trace = ActivationTrace(retain_grad=False)
        current = self.adapter.embed(
            model_inputs,
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

        torch.testing.assert_close(logits, baseline.logits, rtol=0, atol=0)
        for name in baseline.activations:
            torch.testing.assert_close(
                trace[name],
                baseline.activations[name],
                rtol=0,
                atol=0,
            )
        self.assertIs(trace["embedding.output"], trace["layer.0.input"])

    def test_forward_matches_explicit_trace_and_preserves_alias_identity(
        self,
    ) -> None:
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ]
        )
        requested = (
            "embedding.output",
            "layer.0.output",
            "layer.1.input",
            "layer.1.output",
        )

        direct = self.model(
            input_ids,
            capture_activations=True,
            retain_activation_gradients=False,
        )
        run = self.adapter.forward(
            {"input_ids": input_ids},
            capture_sites=requested,
        )

        self.assertTrue(torch.equal(run.logits, direct.logits))
        self.assertEqual(tuple(run.activations), requested)
        assert direct.activations is not None
        for name in requested:
            self.assertTrue(
                torch.equal(run.activations[name], direct.activations[name])
            )
        self.assertIs(
            run.activations["layer.0.output"],
            run.activations["layer.1.input"],
        )
        self.assertIs(
            direct.activations["layer.0.output"],
            direct.activations["layer.1.input"],
        )

    def test_prepare_sequence_accepts_dynamic_lengths_and_rejects_decode(
        self,
    ) -> None:
        short_ids = torch.tensor([[1, 2, 3]])
        short_mask = torch.tensor([[True, True, False]])
        short = self.adapter.prepare_sequence(
            {
                "input_ids": short_ids,
                "attention_mask": short_mask,
            }
        )
        self.assertEqual(short.query_length, 3)
        self.assertEqual(short.key_length, 3)
        self.assertTrue(torch.equal(short.query_valid_mask, short_mask))
        self.assertTrue(torch.equal(short.key_valid_mask, short_mask))
        self.assertTrue(
            torch.equal(
                short.logical_positions,
                torch.tensor([[0, 1, 2]]),
            )
        )
        self.assertIsNone(short.cache_positions)
        self.assertEqual(short.phase, "prefill")

        long_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1],
            ]
        )
        long = self.adapter.prepare_sequence({"input_ids": long_ids})
        self.assertEqual((long.batch_size, long.query_length), (2, 7))
        self.assertTrue(long.query_valid_mask.all())
        self.assertTrue(
            torch.equal(
                long.logical_positions,
                torch.arange(7).unsqueeze(0).expand(2, -1),
            )
        )

        with self.assertRaisesRegex(ValueError, "cached decode"):
            self.adapter.prepare_sequence(
                {"input_ids": torch.tensor([[1]])},
                phase="decode",
            )
        with self.assertRaisesRegex(ValueError, "cached decode"):
            self.adapter.forward(
                {"input_ids": torch.tensor([[1]])},
                phase="decode",
                cache_state=object(),
            )

    def test_run_segment_matches_each_explicit_layer_boundary(self) -> None:
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ]
        )
        run = self.adapter.forward(
            {"input_ids": input_ids},
            capture_sites=(
                "layer.0.input",
                "layer.0.output",
                "layer.1.output",
            ),
        )

        first = self.adapter.run_segment(
            self.adapter.segments[0],
            run.activations["layer.0.input"],
            run.sequence,
        )
        second = self.adapter.run_segment(
            self.adapter.segments[1],
            first.hidden_states,
            run.sequence,
        )

        self.assertTrue(
            torch.equal(
                first.hidden_states,
                run.activations["layer.0.output"],
            )
        )
        self.assertTrue(
            torch.equal(
                second.hidden_states,
                run.activations["layer.1.output"],
            )
        )

    def test_replaced_segments_restores_on_normal_and_exceptional_exit(
        self,
    ) -> None:
        original = self.model.layers[0]
        replacement = ZeroLayer()

        with self.adapter.replaced_segments({"layer.0": replacement}):
            self.assertIs(self.model.layers[0], replacement)
        self.assertIs(self.model.layers[0], original)

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with self.adapter.replaced_segments({"layer.0": replacement}):
                self.assertIs(self.model.layers[0], replacement)
                raise RuntimeError("deliberate")
        self.assertIs(self.model.layers[0], original)

    def test_fingerprints_bind_attention_semantics_as_well_as_weights(
        self,
    ) -> None:
        configs = (
            TransformerConfig(
                vocab_size=17,
                max_sequence_length=8,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=12,
            ),
            TransformerConfig(
                vocab_size=17,
                max_sequence_length=8,
                d_model=8,
                n_heads=4,
                n_layers=1,
                d_ff=12,
            ),
        )
        models = []
        for config in configs:
            torch.manual_seed(91)
            models.append(ToyTransformer(config).eval())
        adapters = tuple(ToyTransformerAdapter(model) for model in models)

        self.assertEqual(
            module_state_fingerprint(models[0]),
            module_state_fingerprint(models[1]),
        )
        self.assertNotEqual(
            adapters[0].semantic_fingerprint(),
            adapters[1].semantic_fingerprint(),
        )
        self.assertNotEqual(
            adapters[0].model_fingerprint(),
            adapters[1].model_fingerprint(),
        )
        self.assertNotEqual(
            adapters[0].segment_fingerprint(adapters[0].segments[0]),
            adapters[1].segment_fingerprint(adapters[1].segments[0]),
        )

    def test_generic_score_collection_accepts_mixed_sequence_lengths(
        self,
    ) -> None:
        short_inputs = torch.tensor([[1, 2, 3]])
        short_targets = torch.tensor([[-100, -100, 4]])
        short_valid = torch.ones_like(short_inputs, dtype=torch.bool)
        long_inputs = torch.tensor([[5, 6, 7, 8, 9]])
        long_targets = torch.tensor([[-100, -100, -100, -100, 10]])
        long_valid = torch.ones_like(long_inputs, dtype=torch.bool)
        batches = (
            CalibrationBatch(
                model_inputs={
                    "input_ids": short_inputs,
                    "attention_mask": short_valid,
                },
                targets=short_targets,
                valid_positions=short_valid,
            ),
            CalibrationBatch(
                model_inputs={
                    "input_ids": long_inputs,
                    "attention_mask": long_valid,
                },
                targets=long_targets,
                valid_positions=long_valid,
            ),
        )

        collection = collect_adapter_score_gradients(
            self.adapter,
            batches,
            activation_names=("layer.0.output",),
            score_objective=CausalLanguageModelNLL(),
        )
        samples = collection.samples["layer.0.output"]

        self.assertEqual(collection.sequences, 2)
        self.assertEqual(samples.sequences, 2)
        self.assertEqual(samples.observations, 8)
        self.assertEqual(samples.activations.shape, (8, 8))
        self.assertEqual(samples.score_gradients.shape, (8, 8))
        self.assertTrue(
            torch.equal(
                samples.locations,
                torch.tensor(
                    [
                        [0, 0],
                        [0, 1],
                        [0, 2],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [1, 3],
                        [1, 4],
                    ]
                ),
            )
        )
        self.assertTrue(torch.isfinite(samples.activations).all())
        self.assertTrue(torch.isfinite(samples.score_gradients).all())
        self.assertIsNone(
            build_fisher_mode_bases(collection)[
                "layer.0.output"
            ].position_means
        )

    def test_score_collection_checks_declared_activation_width(self) -> None:
        self.adapter._activation_sites = tuple(
            replace(site, width=site.width + 1)
            if site.id == "layer.0.output" and site.width is not None
            else site
            for site in self.adapter.activation_sites
        )
        with self.assertRaisesRegex(ValueError, "declared width"):
            collect_activation_score_gradients(
                self.adapter,
                torch.tensor([[1, 2]]),
                torch.tensor([[-100, 3]]),
                activation_names=("layer.0.output",),
            )

    def test_calibration_declares_shared_unbatched_model_inputs(self) -> None:
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.full_like(input_ids, -100)
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        cache_position = torch.tensor([7, 8, 9])

        with self.assertRaisesRegex(ValueError, "declared shared"):
            CalibrationBatch(
                model_inputs={
                    "input_ids": input_ids,
                    "cache_position": cache_position,
                },
                targets=targets,
                valid_positions=valid,
            )

        batch = CalibrationBatch(
            model_inputs={
                "input_ids": input_ids,
                "cache_position": cache_position,
            },
            targets=targets,
            valid_positions=valid,
            shared_input_names=frozenset({"cache_position"}),
            example_ids=("first", "second"),
        )
        second = batch.sample(1)
        self.assertTrue(
            torch.equal(second.model_inputs["input_ids"], input_ids[1:2])
        )
        self.assertIs(
            second.model_inputs["cache_position"],
            cache_position,
        )
        self.assertEqual(second.example_ids, ("second",))

    def test_compatibility_score_collector_matches_generic_collector(
        self,
    ) -> None:
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ]
        )
        targets = torch.tensor(
            [
                [-100, -100, -100, 5],
                [-100, -100, -100, 6],
            ]
        )
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        names = ("layer.0.output", "layer.1.output")

        generic = collect_adapter_score_gradients(
            self.adapter,
            (
                CalibrationBatch(
                    model_inputs={
                        "input_ids": input_ids,
                        "attention_mask": valid,
                    },
                    targets=targets,
                    valid_positions=valid,
                    example_ids=("sequence.0", "sequence.1"),
                ),
            ),
            activation_names=names,
            score_objective=CausalLanguageModelNLL(),
        )
        compatibility = collect_activation_score_gradients(
            self.model,
            input_ids,
            targets,
            attention_mask=valid,
            activation_names=names,
        )

        self.assertEqual(generic.sequences, compatibility.sequences)
        self.assertEqual(generic.mean_loss, compatibility.mean_loss)
        self.assertEqual(tuple(generic.samples), tuple(compatibility.samples))
        for name in names:
            generic_samples = generic.samples[name]
            compatibility_samples = compatibility.samples[name]
            self.assertEqual(
                generic_samples.sequences,
                compatibility_samples.sequences,
            )
            self.assertTrue(
                torch.equal(
                    generic_samples.activations,
                    compatibility_samples.activations,
                )
            )
            self.assertTrue(
                torch.equal(
                    generic_samples.score_gradients,
                    compatibility_samples.score_gradients,
                )
            )
            self.assertTrue(
                torch.equal(
                    generic_samples.locations,
                    compatibility_samples.locations,
                )
            )

    def test_segment_jacobian_matches_legacy_layer_executor_path(self) -> None:
        input_ids = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        targets = torch.tensor(
            [
                [-100, -100, 7],
                [-100, -100, 8],
            ]
        )
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        names = ("layer.0.input", "layer.0.output")
        collection = collect_activation_score_gradients(
            self.adapter,
            input_ids,
            targets,
            attention_mask=valid,
            activation_names=names,
        )
        bases = build_fisher_mode_bases(collection)
        kwargs = {
            "input_modes": 2,
            "output_modes": 2,
            "max_sequences": 2,
        }

        legacy = extract_modal_jacobian(
            self.model.layers[0],
            collection.samples[names[0]],
            bases[names[0]],
            bases[names[1]],
            **kwargs,
        )
        adapted = extract_segment_modal_jacobian(
            self.adapter,
            self.adapter.segments[0],
            (
                CalibrationBatch(
                    model_inputs={
                        "input_ids": input_ids.flip(0),
                        "attention_mask": valid.flip(0),
                    },
                    targets=targets.flip(0),
                    valid_positions=valid.flip(0),
                    example_ids=("sequence.1", "sequence.0"),
                ),
            ),
            collection.samples[names[0]],
            bases[names[0]],
            bases[names[1]],
            **kwargs,
        )

        self.assertEqual(adapted.input_activation, legacy.input_activation)
        self.assertEqual(adapted.output_activation, legacy.output_activation)
        self.assertEqual(adapted.sequence_length, legacy.sequence_length)
        self.assertEqual(adapted.samples, legacy.samples)
        self.assertTrue(torch.equal(adapted.mean, legacy.mean))
        self.assertTrue(torch.equal(adapted.rms, legacy.rms))


if __name__ == "__main__":
    unittest.main()
