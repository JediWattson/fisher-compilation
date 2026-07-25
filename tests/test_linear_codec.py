import io
import unittest

import torch

from fisher_graph.linear_codec import (
    ActivationCovarianceResult,
    LinearActivationCodec,
    StreamingActivationCovariance,
    build_generalized_fisher_codec,
    build_native_fisher_codec,
    build_variance_weighted_fisher_codec,
)


class StreamingActivationCovarianceTests(unittest.TestCase):
    def test_chunked_masked_low_precision_matches_exact_population_state(
        self,
    ) -> None:
        values = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [20.0, 30.0, 40.0],
                [3.0, 4.0, 9.0],
                [5.0, 8.0, 7.0],
                [6.0, 1.0, 2.0],
            ],
            dtype=torch.float16,
        )
        mask = torch.tensor([True, False, True, True, True])
        accumulator = StreamingActivationCovariance(
            activation_name="layer.4.output",
        )
        accumulator.update(values[:2], mask=mask[:2])
        accumulator.update(values[2:], mask=mask[2:])
        result = accumulator.finalize()

        selected = values[mask].double()
        centered = selected - selected.mean(dim=0)
        expected_covariance = centered.T @ centered / selected.shape[0]
        self.assertEqual(result.mean.dtype, torch.float64)
        self.assertEqual(result.covariance.dtype, torch.float64)
        self.assertEqual(result.mean.device.type, "cpu")
        self.assertEqual(result.observations, 4)
        self.assertEqual(result.rows_seen, 5)
        torch.testing.assert_close(result.mean, selected.mean(dim=0))
        torch.testing.assert_close(result.covariance, expected_covariance)
        self.assertAlmostEqual(
            result.centered_square_norm_sum,
            centered.square().sum().item(),
        )

    def test_rejected_update_is_finite_and_transactional(self) -> None:
        accumulator = StreamingActivationCovariance(
            activation_name="tap",
            width=2,
        )
        accumulator.update(torch.tensor([[1.0, 2.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.update(
                torch.tensor([[float("nan"), 1.0]]),
            )
        self.assertEqual(accumulator.observations, 1)
        self.assertEqual(accumulator.rows_seen, 1)
        torch.testing.assert_close(
            accumulator.finalize().mean,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
        )

    def test_result_strict_state_round_trip_and_tamper_rejection(self) -> None:
        result = (
            StreamingActivationCovariance(activation_name="serial")
            .update(
                torch.tensor(
                    [[1.0, 0.0], [3.0, 2.0], [5.0, 1.0]],
                )
            )
            .finalize()
        )
        payload = io.BytesIO()
        torch.save(result.state_dict(), payload)
        payload.seek(0)
        restored = ActivationCovarianceResult.from_state_dict(
            torch.load(payload, weights_only=True)
        )
        torch.testing.assert_close(restored.mean, result.mean)
        torch.testing.assert_close(restored.covariance, result.covariance)
        self.assertEqual(restored.metadata(), result.metadata())

        unknown = result.state_dict()
        unknown["surprise"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            ActivationCovarianceResult.from_state_dict(unknown)

        wrong_dtype = result.state_dict()
        wrong_dtype["mean"] = result.mean.float()
        with self.assertRaisesRegex(ValueError, "CPU float64"):
            ActivationCovarianceResult.from_state_dict(wrong_dtype)

        inconsistent = result.state_dict()
        inconsistent["covariance_trace"] = 999.0
        with self.assertRaisesRegex(ValueError, "covariance_trace"):
            ActivationCovarianceResult.from_state_dict(inconsistent)


class LinearActivationCodecTests(unittest.TestCase):
    def test_native_builder_preserves_fisher_order_and_round_trips(
        self,
    ) -> None:
        covariance = (
            StreamingActivationCovariance(activation_name="native")
            .update(
                torch.tensor(
                    [
                        [1.0, 2.0, 0.0],
                        [2.0, 0.0, 1.0],
                        [4.0, 3.0, 2.0],
                    ]
                )
            )
            .finalize()
        )
        fisher_vectors = torch.eye(3, dtype=torch.float64)[:, [2, 0, 1]]
        fisher_eigenvalues = torch.tensor(
            [9.0, 4.0, 1.0],
            dtype=torch.float64,
        )
        codec = build_native_fisher_codec(
            covariance=covariance,
            fisher_eigenvalues=fisher_eigenvalues,
            fisher_vectors=fisher_vectors,
        )

        self.assertEqual(codec.method, "native_fisher")
        torch.testing.assert_close(
            codec.importance_scores,
            fisher_eigenvalues,
        )
        torch.testing.assert_close(codec.eigenvalues, fisher_eigenvalues)
        torch.testing.assert_close(codec.encoder, fisher_vectors)
        torch.testing.assert_close(codec.decoder, fisher_vectors)
        self.assertEqual(codec.activation_observations, 3)
        value = torch.tensor([[5.0, 6.0, 7.0]])
        torch.testing.assert_close(
            codec.reconstruct(value, rank=codec.width),
            value,
        )

        restored = LinearActivationCodec.from_state_dict(codec.state_dict())
        self.assertEqual(restored.metadata(), codec.metadata())
        torch.testing.assert_close(restored.encoder, codec.encoder)
        self.assertEqual(
            restored.importance_semantics,
            "native_fisher_eigenvalue",
        )

    def test_variance_weighted_builder_reorders_fisher_modes(self) -> None:
        covariance = torch.diag(
            torch.tensor([0.01, 3.0, 2.0], dtype=torch.float64)
        )
        codec = build_variance_weighted_fisher_codec(
            activation_name="tap",
            mean=torch.zeros(3),
            covariance=covariance,
            fisher_eigenvalues=torch.tensor(
                [100.0, 2.0, 1.0],
                dtype=torch.float64,
            ),
            fisher_vectors=torch.eye(3, dtype=torch.float64),
        )

        self.assertEqual(codec.method, "variance_weighted_fisher")
        torch.testing.assert_close(
            codec.importance_scores,
            torch.tensor([6.0, 2.0, 1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            codec.eigenvalues,
            torch.tensor([2.0, 1.0, 100.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            codec.encoder,
            torch.eye(3, dtype=torch.float64)[:, [1, 2, 0]],
        )
        self.assertLessEqual(codec.full_rank_identity_residual, 1e-12)

        value = torch.tensor([[4.0, 5.0, 6.0]])
        torch.testing.assert_close(
            codec.reconstruct(value, rank=1),
            torch.tensor([[0.0, 5.0, 0.0]]),
        )
        torch.testing.assert_close(
            codec.reconstruct(value, rank=0),
            torch.zeros_like(value),
        )
        torch.testing.assert_close(
            codec.reconstruct(value, rank=3),
            value,
        )

    def test_generalized_codec_has_dual_full_identity_and_useful_prefix(
        self,
    ) -> None:
        covariance = torch.diag(
            torch.tensor([9.0, 1.0], dtype=torch.float64)
        )
        fisher = torch.diag(
            torch.tensor([1.0, 4.0], dtype=torch.float64)
        )
        codec = build_generalized_fisher_codec(
            activation_name="tap",
            mean=torch.tensor([1.0, -2.0]),
            covariance=covariance,
            fisher_matrix=fisher,
            alpha=0.0,
            beta=0.0,
        )

        self.assertEqual(codec.method, "generalized_fisher")
        torch.testing.assert_close(
            codec.eigenvalues,
            torch.tensor([9.0, 4.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            codec.encoder @ codec.decoder.T,
            torch.eye(2, dtype=torch.float64),
        )
        value = torch.tensor([[7.0, 5.0]], dtype=torch.float64)
        torch.testing.assert_close(codec.reconstruct(value, rank=2), value)
        # The leading generalized direction is residual coordinate zero.
        torch.testing.assert_close(
            codec.reconstruct(value, rank=1),
            torch.tensor([[7.0, -2.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            codec.reconstruct(value, rank=0),
            torch.tensor([[1.0, -2.0]], dtype=torch.float64),
        )

    def test_generalized_rank_deficiency_requires_matching_positive_floors(
        self,
    ) -> None:
        covariance = torch.diag(torch.tensor([2.0, 0.0]))
        fisher = torch.diag(torch.tensor([0.0, 3.0]))
        common = {
            "activation_name": "rank-deficient",
            "mean": torch.zeros(2),
            "covariance": covariance,
            "fisher_matrix": fisher,
        }
        with self.assertRaisesRegex(
            ValueError,
            "activation covariance requires a positive",
        ):
            build_generalized_fisher_codec(
                **common,
                alpha=0.0,
                beta=1e-4,
            )
        with self.assertRaisesRegex(
            ValueError,
            "Fisher matrix requires a positive",
        ):
            build_generalized_fisher_codec(
                **common,
                alpha=1e-4,
                beta=0.0,
            )

        codec = build_generalized_fisher_codec(
            **common,
            alpha=1e-3,
            beta=2e-3,
        )
        self.assertEqual(codec.alpha_floor, 1e-3)
        self.assertEqual(codec.beta_floor, 2e-3)
        self.assertTrue(
            torch.isfinite(
                torch.tensor(
                    [
                        codec.activation_condition_number,
                        codec.fisher_condition_number,
                        codec.operator_condition_number,
                    ]
                )
            ).all()
        )
        torch.testing.assert_close(
            codec.encoder @ codec.decoder.T,
            torch.eye(2, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_covariance_result_infers_provenance_and_half_is_promoted(
        self,
    ) -> None:
        covariance = (
            StreamingActivationCovariance(activation_name="inferred")
            .update(
                torch.tensor(
                    [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]],
                    dtype=torch.float16,
                )
            )
            .finalize()
        )
        codec = build_variance_weighted_fisher_codec(
            covariance=covariance,
            fisher_eigenvalues=torch.tensor([2.0, 1.0]),
            fisher_vectors=torch.eye(2),
        )
        self.assertEqual(codec.activation_name, "inferred")
        self.assertEqual(codec.activation_observations, 3)
        values = torch.tensor([[3.0, 7.0]], dtype=torch.float16)
        coordinates = codec.encode(values, rank=1)
        self.assertEqual(coordinates.dtype, torch.float32)
        reconstructed = codec.reconstruct(values, rank=2)
        self.assertEqual(reconstructed.dtype, torch.float16)
        torch.testing.assert_close(reconstructed, values)

    def test_codec_strict_state_round_trip_and_tamper_rejection(self) -> None:
        codec = build_generalized_fisher_codec(
            activation_name="serial",
            mean=torch.tensor([0.5, -0.5]),
            covariance=torch.tensor([[2.0, 0.25], [0.25, 1.0]]),
            fisher_matrix=torch.tensor([[1.0, 0.1], [0.1, 3.0]]),
            alpha=1e-6,
            beta=2e-6,
        )
        payload = io.BytesIO()
        torch.save(codec.state_dict(), payload)
        payload.seek(0)
        restored = LinearActivationCodec.from_state_dict(
            torch.load(payload, weights_only=True)
        )
        self.assertEqual(restored.metadata(), codec.metadata())
        torch.testing.assert_close(restored.encoder, codec.encoder)
        torch.testing.assert_close(restored.decoder, codec.decoder)

        unknown = codec.state_dict()
        unknown["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "fields"):
            LinearActivationCodec.from_state_dict(unknown)

        wrong_order = codec.state_dict()
        wrong_order["importance_scores"] = codec.importance_scores.flip(0)
        with self.assertRaisesRegex(ValueError, "ordered"):
            LinearActivationCodec.from_state_dict(wrong_order)

        broken_dual = codec.state_dict()
        broken_dual["decoder"] = codec.decoder.clone()
        broken_dual["decoder"][0, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "identity"):
            LinearActivationCodec.from_state_dict(broken_dual)

        false_residual = codec.state_dict()
        false_residual["full_rank_identity_residual"] = 0.5
        with self.assertRaisesRegex(ValueError, "residual"):
            LinearActivationCodec.from_state_dict(false_residual)

        bad_semantics = codec.state_dict()
        bad_semantics["importance_semantics"] = "unknown"
        with self.assertRaisesRegex(ValueError, "semantics"):
            LinearActivationCodec.from_state_dict(bad_semantics)

    def test_invalid_shapes_symmetry_psd_and_rank_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric"):
            build_generalized_fisher_codec(
                activation_name="bad",
                mean=torch.zeros(2),
                covariance=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
                fisher_matrix=torch.eye(2),
                alpha=1e-3,
                beta=1e-3,
            )
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            build_generalized_fisher_codec(
                activation_name="bad",
                mean=torch.zeros(2),
                covariance=torch.diag(torch.tensor([1.0, -1.0])),
                fisher_matrix=torch.eye(2),
                alpha=1e-3,
                beta=1e-3,
            )
        codec = build_variance_weighted_fisher_codec(
            activation_name="rank",
            mean=torch.zeros(2),
            covariance=torch.eye(2),
            fisher_eigenvalues=torch.tensor([2.0, 1.0]),
            fisher_vectors=torch.eye(2),
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 2"):
            codec.reconstruct(torch.ones(1, 2), rank=3)


if __name__ == "__main__":
    unittest.main()
