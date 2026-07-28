import io
import unittest
from dataclasses import FrozenInstanceError

import torch

from fisher_graph import (
    CalibrationBatch,
    CausalLanguageModelNLL,
    ToyTransformer,
    ToyTransformerAdapter,
    TransformerConfig,
)
from fisher_graph.generator_message_analysis import (
    GeneratorMessageCapturePlan,
    GeneratorMessageScoreGradientRows,
    JointMessageMomentsResult,
    StreamingJointMessageMoments,
    iter_generator_message_score_gradient_rows,
)


class GeneratorMessageAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1493)
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
        self.plan = GeneratorMessageCapturePlan(
            value_sites=(
                "layer.0.input",
                "layer.0.output",
                "layer.1.input",
                "layer.1.output",
            ),
            gradient_sites=(
                "layer.0.output",
                "layer.1.output",
            ),
            leaf_site="layer.0.output",
        )
        self.objective = CausalLanguageModelNLL()

    def _collect_rows(
        self,
    ) -> tuple[GeneratorMessageScoreGradientRows, ...]:
        self.model.requires_grad_(False)
        return tuple(
            iter_generator_message_score_gradient_rows(
                self.adapter,
                (self.batch,),
                plan=self.plan,
                score_objective=self.objective,
            )
        )

    def test_capture_plan_is_immutable_and_round_trips_strictly(self) -> None:
        restored = GeneratorMessageCapturePlan.from_state_dict(
            self.plan.state_dict()
        )

        self.assertEqual(restored, self.plan)
        with self.assertRaises(FrozenInstanceError):
            restored.leaf_site = "layer.1.output"
        with self.assertRaisesRegex(ValueError, "subset"):
            GeneratorMessageCapturePlan(
                value_sites=("layer.0.output",),
                gradient_sites=("layer.1.output",),
                leaf_site="layer.0.output",
            )
        with self.assertRaisesRegex(ValueError, "fields"):
            GeneratorMessageCapturePlan.from_state_dict(
                {**self.plan.state_dict(), "extra": True}
            )

    def test_row_stream_separates_values_from_selected_gradients(self) -> None:
        first_sample = self.batch.sample(0)
        self.model.requires_grad_(False)
        self.model.train()

        with torch.enable_grad():
            manual_run = self.adapter.forward(
                first_sample.model_inputs,
                capture_sites=self.plan.value_sites,
                interventions={
                    self.plan.leaf_site: (
                        lambda values: values.detach().requires_grad_(True)
                    )
                },
                retain_gradients=False,
            )
            manual_loss = self.objective(manual_run, first_sample)
            manual_tensors = tuple(
                manual_run.activations[name]
                for name in self.plan.gradient_sites
            )
            manual_gradients = torch.autograd.grad(
                manual_loss,
                manual_tensors,
            )
        valid = first_sample.valid_positions[0]

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            stream = iter_generator_message_score_gradient_rows(
                self.adapter,
                (self.batch,),
                plan=self.plan,
                score_objective=self.objective,
            )
            rows = tuple(stream)
            self.assertFalse(torch.is_grad_enabled())

        self.assertTrue(self.model.training)
        self.assertEqual(
            [row.example_id for row in rows],
            ["first", "second"],
        )
        self.assertEqual([row.observations for row in rows], [4, 3])
        self.assertEqual(
            tuple(rows[0].activations),
            self.plan.value_sites,
        )
        self.assertEqual(
            tuple(rows[0].score_gradients),
            self.plan.gradient_sites,
        )
        self.assertNotIn(
            "layer.0.input",
            rows[0].score_gradients,
        )
        for tensor in (
            *rows[0].activations.values(),
            *rows[0].score_gradients.values(),
        ):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.float64)
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        torch.testing.assert_close(
            rows[0].logical_positions,
            torch.tensor([0, 1, 2, 3]),
        )
        for name, expected in zip(
            self.plan.gradient_sites,
            manual_gradients,
            strict=True,
        ):
            torch.testing.assert_close(
                rows[0].score_gradients[name],
                expected.detach()[0, valid].double(),
            )

    def test_row_stream_requires_frozen_model_and_unique_example_ids(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "frozen model"):
            iter_generator_message_score_gradient_rows(
                self.adapter,
                (self.batch,),
                plan=self.plan,
                score_objective=self.objective,
            )

        self.model.requires_grad_(False)
        anonymous = CalibrationBatch(
            model_inputs=self.batch.model_inputs,
            targets=self.batch.targets,
            valid_positions=self.batch.valid_positions,
        )
        with self.assertRaisesRegex(ValueError, "requires example_ids"):
            tuple(
                iter_generator_message_score_gradient_rows(
                    self.adapter,
                    (anonymous,),
                    plan=self.plan,
                    score_objective=self.objective,
                )
            )

        first = self.batch.sample(0)
        with self.assertRaisesRegex(ValueError, "duplicate 'first'"):
            tuple(
                iter_generator_message_score_gradient_rows(
                    self.adapter,
                    (first, first),
                    plan=self.plan,
                    score_objective=self.objective,
                )
            )

    def test_close_restores_training_state(self) -> None:
        self.model.requires_grad_(False)
        self.model.train()
        stream = iter_generator_message_score_gradient_rows(
            self.adapter,
            (self.batch,),
            plan=self.plan,
            score_objective=self.objective,
        )

        first = next(stream)

        self.assertEqual(first.example_id, "first")
        self.assertFalse(self.model.training)
        stream.close()
        self.assertTrue(self.model.training)

    def test_joint_moments_match_materialized_exact_matrices(self) -> None:
        rows = self._collect_rows()
        estimator = StreamingJointMessageMoments(
            self.plan.gradient_sites
        )
        for row in rows:
            estimator.update(row)
        result = estimator.finalize()

        activations = torch.cat(
            [
                torch.cat(
                    [
                        row.activations[name]
                        for name in self.plan.gradient_sites
                    ],
                    dim=1,
                )
                for row in rows
            ],
            dim=0,
        )
        gradients = torch.cat(
            [
                torch.cat(
                    [
                        row.score_gradients[name]
                        for name in self.plan.gradient_sites
                    ],
                    dim=1,
                )
                for row in rows
            ],
            dim=0,
        )
        expected_mean = activations.mean(dim=0)
        centered = activations - expected_mean
        expected_covariance = centered.T @ centered / activations.shape[0]
        expected_fisher = gradients.T @ gradients / gradients.shape[0]

        self.assertEqual(result.site_names, self.plan.gradient_sites)
        self.assertEqual(result.site_widths, (8, 8))
        self.assertEqual(result.observations, 7)
        self.assertEqual(result.sequences, 2)
        self.assertEqual(
            estimator.storage_shapes,
            ((16,), (16, 16), (16, 16)),
        )
        torch.testing.assert_close(result.mean, expected_mean)
        torch.testing.assert_close(
            result.covariance,
            expected_covariance,
        )
        torch.testing.assert_close(result.fisher, expected_fisher)
        self.assertEqual(
            {
                name: (site_slice.start, site_slice.stop)
                for name, site_slice in result.port_slices.items()
            },
            {
                "layer.0.output": (0, 8),
                "layer.1.output": (8, 16),
            },
        )
        self.assertEqual(
            dict(result.per_site_counts),
            {
                "layer.0.output": 7,
                "layer.1.output": 7,
            },
        )
        torch.testing.assert_close(
            result.per_site_means["layer.1.output"],
            expected_mean[8:],
        )
        torch.testing.assert_close(
            result.covariance_cross_blocks[
                ("layer.0.output", "layer.1.output")
            ],
            expected_covariance[:8, 8:],
        )
        torch.testing.assert_close(
            result.fisher_cross_blocks[
                ("layer.0.output", "layer.1.output")
            ],
            expected_fisher[:8, 8:],
        )

    def test_joint_moment_state_round_trip_and_tamper_checks(self) -> None:
        estimator = StreamingJointMessageMoments(
            self.plan.gradient_sites
        )
        for rows in self._collect_rows():
            estimator.update(rows)
        result = estimator.finalize()
        payload = io.BytesIO()
        torch.save(result.state_dict(), payload)
        payload.seek(0)

        restored = JointMessageMomentsResult.from_state_dict(
            torch.load(payload, weights_only=True)
        )

        self.assertEqual(restored.metadata(), result.metadata())
        torch.testing.assert_close(restored.mean, result.mean)
        torch.testing.assert_close(
            restored.covariance,
            result.covariance,
        )
        torch.testing.assert_close(restored.fisher, result.fisher)

        unknown = {**result.state_dict(), "unknown": 1}
        with self.assertRaisesRegex(ValueError, "fields"):
            JointMessageMomentsResult.from_state_dict(unknown)

        bad_slices = result.state_dict()
        bad_slices["port_slices"] = (
            ("layer.0.output", 0, 7),
            ("layer.1.output", 7, 16),
        )
        with self.assertRaisesRegex(ValueError, "port_slices"):
            JointMessageMomentsResult.from_state_dict(bad_slices)

        bad_trace = result.state_dict()
        bad_trace["squared_score_gradient_norm_sum"] = (
            result.squared_score_gradient_norm_sum + 1.0
        )
        with self.assertRaisesRegex(ValueError, "matrix trace"):
            JointMessageMomentsResult.from_state_dict(bad_trace)


if __name__ == "__main__":
    unittest.main()
