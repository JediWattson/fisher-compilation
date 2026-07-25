import io
import unittest

import torch

from fisher_graph import (
    ActivationScoreGradientRows,
    AdapterRun,
    CalibrationBatch,
    CausalLanguageModelNLL,
    SequenceContext,
    SequenceInputOrigin,
    ToyTransformer,
    ToyTransformerAdapter,
    TransformerConfig,
    iter_activation_score_gradient_rows,
)
from fisher_graph.modes import (
    build_fisher_mode_bases,
    collect_adapter_score_gradients,
)
from fisher_graph.streaming_analysis import (
    StreamingFisherCollection,
    collect_streaming_fisher_modes,
)


class StreamingAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(811)
        self.model = ToyTransformer(
            TransformerConfig(
                vocab_size=19,
                max_sequence_length=6,
                d_model=8,
                n_heads=2,
                n_layers=2,
                d_ff=12,
                dropout=0.0,
            )
        )
        self.adapter = ToyTransformerAdapter(self.model)
        tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 0],
            ]
        )
        valid = torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
            ]
        )
        targets = torch.tensor(
            [
                [2, 3, 4, -100],
                [6, 7, -100, -100],
            ]
        )
        self.batch = CalibrationBatch(
            model_inputs={
                "input_ids": tokens,
                "attention_mask": valid,
            },
            targets=targets,
            valid_positions=valid,
            example_ids=("first", "second"),
        )

    def test_per_sequence_row_stream_matches_materialized_collection(
        self,
    ) -> None:
        names = ("layer.0.input", "layer.0.output")
        objective = CausalLanguageModelNLL()
        exact = collect_adapter_score_gradients(
            self.adapter,
            (self.batch,),
            activation_names=names,
            score_objective=objective,
        )
        self.model.requires_grad_(False)
        self.model.train()

        rows = tuple(
            iter_activation_score_gradient_rows(
                self.adapter,
                (self.batch,),
                activation_names=names,
                score_objective=objective,
                leaf_activation_name="layer.0.input",
            )
        )

        self.assertTrue(self.model.training)
        self.assertEqual([row.example_id for row in rows], ["first", "second"])
        self.assertEqual([row.observations for row in rows], [4, 3])
        self.assertAlmostEqual(
            sum(row.loss for row in rows) / len(rows),
            exact.mean_loss,
        )
        for name in names:
            activations = torch.cat(
                [row.activations[name] for row in rows],
                dim=0,
            )
            gradients = torch.cat(
                [row.score_gradients[name] for row in rows],
                dim=0,
            )
            self.assertEqual(activations.device.type, "cpu")
            self.assertEqual(gradients.device.type, "cpu")
            self.assertEqual(activations.dtype, torch.float64)
            self.assertEqual(gradients.dtype, torch.float64)
            torch.testing.assert_close(
                activations,
                exact.samples[name].activations.double(),
            )
            torch.testing.assert_close(
                gradients,
                exact.samples[name].score_gradients.double(),
            )

    def test_row_stream_close_restores_training_state(self) -> None:
        self.model.requires_grad_(False)
        self.model.train()
        rows = iter_activation_score_gradient_rows(
            self.adapter,
            (self.batch,),
            activation_names=("layer.0.input",),
            score_objective=CausalLanguageModelNLL(),
            leaf_activation_name="layer.0.input",
        )

        first = next(rows)

        self.assertEqual(first.example_id, "first")
        self.assertFalse(self.model.training)
        rows.close()
        self.assertTrue(self.model.training)

    def test_row_stream_does_not_leak_enabled_grad_mode_at_yield(self) -> None:
        self.model.requires_grad_(False)
        rows = iter_activation_score_gradient_rows(
            self.adapter,
            (self.batch,),
            activation_names=("layer.0.input",),
            score_objective=CausalLanguageModelNLL(),
            leaf_activation_name="layer.0.input",
        )

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            first = next(rows)
            self.assertIsInstance(first, ActivationScoreGradientRows)
            self.assertFalse(torch.is_grad_enabled())
            rows.close()
            self.assertFalse(torch.is_grad_enabled())

    def test_row_stream_preserves_missing_example_ids(self) -> None:
        anonymous = CalibrationBatch(
            model_inputs=self.batch.model_inputs,
            targets=self.batch.targets,
            valid_positions=self.batch.valid_positions,
        )
        self.model.requires_grad_(False)

        rows = tuple(
            iter_activation_score_gradient_rows(
                self.adapter,
                (anonymous,),
                activation_names=("layer.0.input",),
                score_objective=CausalLanguageModelNLL(),
                leaf_activation_name="layer.0.input",
                accumulation_dtype=torch.float32,
            )
        )

        self.assertEqual([row.example_id for row in rows], [None, None])
        self.assertTrue(
            all(
                row.activations["layer.0.input"].dtype == torch.float32
                for row in rows
            )
        )

    def test_streamed_full_rank_matches_materialized_fisher(self) -> None:
        names = ("layer.0.input", "layer.0.output")
        objective = CausalLanguageModelNLL()
        exact = collect_adapter_score_gradients(
            self.adapter,
            (self.batch,),
            activation_names=names,
            score_objective=objective,
        )
        exact_bases = build_fisher_mode_bases(exact)
        self.model.requires_grad_(False)
        self.model.train()

        streamed = collect_streaming_fisher_modes(
            self.adapter,
            (self.batch,),
            activation_names=names,
            score_objective=objective,
            rank=8,
            sketch_rows=9,
            leaf_activation_name="layer.0.input",
        )

        self.assertTrue(self.model.training)
        self.assertEqual(streamed.sequences, 2)
        self.assertAlmostEqual(streamed.mean_loss, exact.mean_loss)
        for name in names:
            expected = exact_bases[name]
            actual = streamed.bases[name]
            torch.testing.assert_close(actual.mean, expected.mean)
            torch.testing.assert_close(
                actual.fisher.approximate_matrix(),
                expected.matrix,
                rtol=1e-10,
                atol=1e-10,
            )
            self.assertAlmostEqual(
                actual.fisher.fisher_trace,
                expected.fisher_trace,
            )
            self.assertEqual(actual.observations, 7)
            self.assertEqual(actual.sequences, 2)

    def test_state_round_trip_contains_analysis_tensors_only(self) -> None:
        self.model.requires_grad_(False)
        collection = collect_streaming_fisher_modes(
            self.adapter,
            (self.batch,),
            activation_names=("layer.0.input",),
            score_objective=CausalLanguageModelNLL(),
            rank=3,
            sketch_rows=6,
            leaf_activation_name="layer.0.input",
        )
        payload = io.BytesIO()
        torch.save(collection.state_dict(), payload)
        payload.seek(0)

        restored = StreamingFisherCollection.from_state_dict(
            torch.load(payload, weights_only=True)
        )

        self.assertEqual(restored.metadata(), collection.metadata())
        torch.testing.assert_close(
            restored.bases["layer.0.input"].mean,
            collection.bases["layer.0.input"].mean,
        )
        torch.testing.assert_close(
            restored.bases["layer.0.input"].vectors,
            collection.bases["layer.0.input"].vectors,
        )
        self.assertNotIn("state_dict", restored.state_dict())

    def test_nested_analysis_state_rejects_unknown_weight_fields(self) -> None:
        self.model.requires_grad_(False)
        collection = collect_streaming_fisher_modes(
            self.adapter,
            (self.batch,),
            activation_names=("layer.0.input",),
            score_objective=CausalLanguageModelNLL(),
            rank=3,
            sketch_rows=6,
            leaf_activation_name="layer.0.input",
        )
        state = collection.state_dict()
        state["model_state_dict"] = {"weight": torch.ones(2)}
        with self.assertRaisesRegex(ValueError, "collection fields"):
            StreamingFisherCollection.from_state_dict(state)

        state = collection.state_dict()
        basis = state["bases"]["layer.0.input"]
        basis["model_state_dict"] = {"weight": torch.ones(2)}
        with self.assertRaisesRegex(ValueError, "basis fields"):
            StreamingFisherCollection.from_state_dict(state)

        state = collection.state_dict()
        fisher = state["bases"]["layer.0.input"]["fisher"]
        fisher["model_state_dict"] = {"weight": torch.ones(2)}
        with self.assertRaisesRegex(ValueError, "result fields"):
            StreamingFisherCollection.from_state_dict(state)

    def test_nll_upcasts_low_precision_logits_for_score_computation(
        self,
    ) -> None:
        logits = torch.tensor(
            [[[1.0, -1.0, 0.5], [0.0, 0.0, 0.0]]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        valid = torch.ones((1, 2), dtype=torch.bool)
        positions = torch.arange(2).unsqueeze(0)
        run = AdapterRun(
            logits=logits,
            activations={},
            sequence=SequenceContext(
                query_valid_mask=valid,
                key_valid_mask=valid,
                logical_positions=positions,
                key_logical_positions=positions,
                cache_positions=positions[0],
                phase="prefill",
                input_origin=SequenceInputOrigin(
                    attention_mask_supplied=True,
                    position_ids_supplied=False,
                    cache_positions_supplied=False,
                ),
            ),
        )
        batch = CalibrationBatch(
            model_inputs={"input_ids": torch.tensor([[1, 2]])},
            targets=torch.tensor([[2, -100]]),
            valid_positions=valid,
        )

        loss = CausalLanguageModelNLL()(run, batch)
        expected = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 3),
            batch.targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        gradient = torch.autograd.grad(loss, logits)[0]

        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_frozen_model_without_leaf_has_actionable_error(self) -> None:
        self.model.requires_grad_(False)

        with self.assertRaisesRegex(ValueError, "leaf_activation_name"):
            collect_streaming_fisher_modes(
                self.adapter,
                (self.batch,),
                activation_names=("layer.0.input",),
                score_objective=CausalLanguageModelNLL(),
                rank=2,
            )

    def test_leaf_must_be_a_requested_activation(self) -> None:
        with self.assertRaisesRegex(ValueError, "one of activation_names"):
            collect_streaming_fisher_modes(
                self.adapter,
                (self.batch,),
                activation_names=("layer.0.output",),
                score_objective=CausalLanguageModelNLL(),
                rank=2,
                leaf_activation_name="layer.0.input",
            )


if __name__ == "__main__":
    unittest.main()
