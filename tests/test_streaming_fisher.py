import io
import json
import unittest

import torch

from fisher_graph.streaming_fisher import (
    StreamingActivationFisherEstimator,
    StreamingFisherResult,
)


def exact_fisher(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.double()
    return scores.T @ scores / scores.shape[0]


class StreamingActivationFisherTests(unittest.TestCase):
    def test_full_width_no_shrink_stream_is_an_exact_eigensystem(self) -> None:
        generator = torch.Generator().manual_seed(700)
        scores = torch.randn(
            29,
            8,
            generator=generator,
            dtype=torch.float64,
        )
        estimator = StreamingActivationFisherEstimator(
            activation_name="full",
            rank=8,
            width=8,
            sketch_rows=9,
        )

        for chunk in scores.split([3, 11, 1, 14]):
            estimator.update(chunk)
        result = estimator.finalize()
        expected = exact_fisher(scores)

        self.assertEqual(result.vectors.shape, (8, 8))
        torch.testing.assert_close(
            result.vectors.T @ result.vectors,
            torch.eye(8, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            result.approximate_matrix(),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertAlmostEqual(
            result.retained_trace,
            result.fisher_trace,
        )

    def test_low_rank_stream_matches_exact_fisher(self) -> None:
        generator = torch.Generator().manual_seed(701)
        left = torch.randn(137, 2, generator=generator, dtype=torch.float64)
        right = torch.randn(2, 7, generator=generator, dtype=torch.float64)
        scores = left @ right
        estimator = StreamingActivationFisherEstimator(
            activation_name="layer.3.input",
            rank=3,
            sketch_rows=5,
        )

        for chunk in scores.split([3, 17, 1, 52, 64]):
            estimator.update(chunk)
        result = estimator.finalize()
        expected = exact_fisher(scores)

        torch.testing.assert_close(
            result.approximate_matrix(),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )
        expected_eigenvalues = torch.linalg.eigvalsh(expected).flip(0)[:3]
        torch.testing.assert_close(
            result.eigenvalues,
            expected_eigenvalues.clamp_min(0),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertAlmostEqual(result.fisher_trace, expected.trace().item())
        self.assertAlmostEqual(result.retained_trace_fraction, 1.0)
        self.assertEqual(result.observations, scores.shape[0])
        self.assertEqual(result.nonzero_observations, scores.shape[0])
        self.assertEqual(result.rows_seen, scores.shape[0])
        self.assertAlmostEqual(
            result.squared_gradient_norm_sum,
            scores.square().sum().item(),
        )
        self.assertEqual(estimator.storage_shape, (10, 7))

    def test_general_sketch_is_psd_conservative_and_chunk_stable(self) -> None:
        generator = torch.Generator().manual_seed(702)
        scores = torch.randn(211, 12, generator=generator, dtype=torch.float64)
        first = StreamingActivationFisherEstimator(
            activation_name="tap",
            rank=3,
            sketch_rows=6,
        )
        second = StreamingActivationFisherEstimator(
            activation_name="tap",
            rank=3,
            sketch_rows=6,
        )

        first.update(scores)
        for chunk in scores.split([1, 7, 64, 3, 100, 36]):
            second.update(chunk)
        first_result = first.finalize()
        second_result = second.finalize()
        exact = exact_fisher(scores)
        approximation = first_result.approximate_matrix()

        torch.testing.assert_close(
            first_result.eigenvalues,
            second_result.eigenvalues,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            first_result.approximate_matrix(),
            second_result.approximate_matrix(),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(approximation).min().item(),
            -1e-12,
        )
        self.assertGreaterEqual(
            torch.linalg.eigvalsh(exact - approximation).min().item(),
            -1e-10,
        )
        self.assertLessEqual(
            first_result.retained_trace,
            first_result.fisher_trace + 1e-12,
        )
        self.assertAlmostEqual(
            first_result.fisher_trace,
            exact.trace().item(),
        )

    def test_mask_zeros_and_low_precision_inputs_are_accounted_for(self) -> None:
        scores = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [float("nan"), 9.0, 9.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 4.0],
            ],
            dtype=torch.float16,
        )
        mask = torch.tensor([True, False, True, True])
        estimator = StreamingActivationFisherEstimator(
            activation_name="masked",
            rank=2,
            sketch_rows=4,
            width=3,
        )

        returned = estimator.update(scores, mask=mask)
        result = estimator.finalize()
        selected = scores[mask].double()

        self.assertIs(returned, estimator)
        self.assertEqual(result.rows_seen, 4)
        self.assertEqual(result.observations, 3)
        self.assertEqual(result.nonzero_observations, 2)
        self.assertEqual(result.eigenvalues.dtype, torch.float64)
        self.assertEqual(result.eigenvalues.device.type, "cpu")
        self.assertEqual(result.vectors.device.type, "cpu")
        self.assertAlmostEqual(
            result.fisher_trace,
            selected.square().sum().item() / 3,
        )
        torch.testing.assert_close(
            result.approximate_matrix(),
            exact_fisher(selected),
        )

    def test_all_zero_scores_have_a_deterministic_basis(self) -> None:
        estimator = StreamingActivationFisherEstimator(
            activation_name="zero",
            rank=3,
            sketch_rows=6,
        )
        estimator.update(torch.zeros(8, 5))

        result = estimator.finalize()

        torch.testing.assert_close(result.eigenvalues, torch.zeros(3).double())
        torch.testing.assert_close(
            result.vectors,
            torch.eye(5, dtype=torch.float64)[:, :3],
        )
        self.assertEqual(result.fisher_trace, 0.0)
        self.assertEqual(result.retained_trace_fraction, 0.0)
        self.assertEqual(result.nonzero_observations, 0)

    def test_result_state_and_metadata_round_trip(self) -> None:
        scores = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        )
        result = (
            StreamingActivationFisherEstimator(
                activation_name="serial",
                rank=1,
                sketch_rows=3,
                score_reduction="summed_nll",
            )
            .update(scores)
            .finalize()
        )
        state = result.state_dict()
        payload = io.BytesIO()
        torch.save(state, payload)
        payload.seek(0)
        restored = StreamingFisherResult.from_state_dict(
            torch.load(payload, weights_only=True)
        )

        json.dumps(result.metadata())
        self.assertEqual(restored.metadata(), result.metadata())
        torch.testing.assert_close(restored.eigenvalues, result.eigenvalues)
        torch.testing.assert_close(restored.vectors, result.vectors)

    def test_result_state_rejects_unknown_or_inconsistent_fields(self) -> None:
        result = (
            StreamingActivationFisherEstimator(
                activation_name="serial",
                rank=1,
                sketch_rows=3,
            )
            .update(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
            .finalize()
        )
        state = result.state_dict()
        state["model_state_dict"] = {"weight": torch.ones(2)}
        with self.assertRaisesRegex(ValueError, "fields"):
            StreamingFisherResult.from_state_dict(state)

        state = result.state_dict()
        state["width"] = 99
        with self.assertRaisesRegex(ValueError, "width"):
            StreamingFisherResult.from_state_dict(state)

    def test_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            StreamingActivationFisherEstimator(
                activation_name="tap",
                rank=0,
            )
        with self.assertRaisesRegex(ValueError, "greater than rank"):
            StreamingActivationFisherEstimator(
                activation_name="tap",
                rank=2,
                sketch_rows=2,
            )
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            StreamingActivationFisherEstimator(
                activation_name="tap",
                rank=2,
                accumulation_dtype=torch.float16,
            )

        estimator = StreamingActivationFisherEstimator(
            activation_name="tap",
            rank=2,
            sketch_rows=4,
            width=3,
        )
        invalid_updates = (
            (
                torch.ones(3),
                None,
                "shape",
            ),
            (
                torch.ones(2, 4),
                None,
                "expected score vector width",
            ),
            (
                torch.ones(2, 3, dtype=torch.long),
                None,
                "real floating dtype",
            ),
            (
                torch.ones(2, 3),
                torch.ones(2),
                "boolean dtype",
            ),
            (
                torch.ones(2, 3),
                torch.ones(2, 1, dtype=torch.bool),
                "mask must have shape",
            ),
        )
        for scores, mask, message in invalid_updates:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    estimator.update(scores, mask=mask)
        with self.assertRaisesRegex(ValueError, "finite"):
            estimator.update(
                torch.tensor([[1.0, float("inf"), 2.0]]),
            )

        empty = StreamingActivationFisherEstimator(
            activation_name="empty",
            rank=1,
        )
        empty.update(
            torch.ones(2, 3),
            mask=torch.zeros(2, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "without any selected"):
            empty.finalize()


if __name__ == "__main__":
    unittest.main()
