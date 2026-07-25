import copy
import io
import unittest

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.jacobian_probe import (
    CausalLagJacobianStatistics,
    collect_block_causal_lag_jacobian,
)
from fisher_graph.linear_codec import (
    LinearActivationCodec,
    build_variance_weighted_fisher_codec,
)
from fisher_graph.model import ToyTransformer
from fisher_graph.config import TransformerConfig


def identity_codec(
    activation_name: str,
    width: int,
) -> LinearActivationCodec:
    """Build a public full-width codec whose ordered basis is identity."""

    return build_variance_weighted_fisher_codec(
        activation_name=activation_name,
        mean=torch.zeros(width, dtype=torch.float64),
        covariance=torch.eye(width, dtype=torch.float64),
        fisher_eigenvalues=torch.arange(
            width,
            0,
            -1,
            dtype=torch.float64,
        ),
        fisher_vectors=torch.eye(width, dtype=torch.float64),
    )


def calibration_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    example_ids: tuple[str, ...],
) -> CalibrationBatch:
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        targets=torch.zeros_like(input_ids),
        valid_positions=attention_mask,
        example_ids=example_ids,
    )


class CausalLagJacobianProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)
        self.width = 4
        self.model = ToyTransformer(
            TransformerConfig(
                vocab_size=19,
                max_sequence_length=5,
                d_model=self.width,
                n_heads=2,
                n_layers=2,
                d_ff=8,
                dropout=0.0,
            )
        ).eval()
        self.model.requires_grad_(False)
        self.adapter = ToyTransformerAdapter(self.model)
        self.plan = self.adapter.plan_layer_block(0, 1)
        self.input_codec = identity_codec(
            self.plan.activation_sites[0],
            self.width,
        )
        self.output_codec = identity_codec(
            self.plan.activation_sites[-1],
            self.width,
        )

    def test_two_layer_full_width_probe_respects_masks_and_causality(
        self,
    ) -> None:
        attention_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, False, True, True],
            ]
        )
        batch = calibration_batch(
            torch.tensor(
                [
                    [1, 2, 3, 0],
                    [4, 0, 5, 6],
                ]
            ),
            attention_mask,
            example_ids=("masked-a", "masked-b"),
        )

        result = collect_block_causal_lag_jacobian(
            self.adapter,
            self.plan,
            (batch,),
            input_codec=self.input_codec,
            output_codec=self.output_codec,
            input_modes=self.width,
            output_modes=self.width,
            max_lag=1,
        )

        self.assertEqual(result.input_activation, "layer.0.input")
        self.assertEqual(result.output_activation, "layer.1.output")
        self.assertEqual(result.sequences, 2)
        # Six valid source positions, with one JVP per identity direction.
        self.assertEqual(result.jvp_calls, 6 * self.width)
        # Same-position pairs: 3 + 3. Lag-one pairs: 2 + 1.
        self.assertEqual(result.lag_pair_counts, (6, 3))
        self.assertEqual(
            result.mean.shape,
            (2, self.width, self.width),
        )
        self.assertEqual(
            result.lag_matrices.shape,
            (2, self.width, self.width),
        )
        self.assertTrue(torch.isfinite(result.mean).all())
        self.assertTrue(torch.isfinite(result.rms).all())
        self.assertGreater(result.captured_squared_sum, 0.0)
        # Both masks have valid causal pairs beyond the retained lag window.
        self.assertGreater(result.omitted_past_squared_sum, 0.0)
        self.assertAlmostEqual(
            result.captured_squared_sum
            + result.omitted_past_squared_sum
            + result.causal_leakage_squared_sum,
            result.total_squared_sum,
            places=11,
        )
        # Decoder-only attention must have no measurable future response.
        leakage_tolerance = (
            1e-20 * max(result.total_squared_sum, 1.0)
        )
        self.assertLessEqual(
            result.causal_leakage_squared_sum,
            leakage_tolerance,
        )
        self.assertLessEqual(result.causal_leakage_fraction, 1e-20)
        self.assertGreaterEqual(
            result.stationary_mean_energy_fraction_of_captured,
            0.0,
        )
        self.assertGreaterEqual(
            result.regime_variation_fraction_of_captured,
            0.0,
        )
        self.assertAlmostEqual(
            result.stationary_mean_energy_fraction_of_captured
            + result.regime_variation_fraction_of_captured,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result.stationary_mean_squared_sum
            + result.within_lag_variation_squared_sum,
            result.captured_squared_sum,
            places=11,
        )
        # Collection preserves the source module's mode and frozen weights.
        self.assertFalse(self.model.training)
        self.assertFalse(
            any(parameter.requires_grad for parameter in self.model.parameters())
        )

    def test_duplicate_ids_across_batches_are_rejected(self) -> None:
        first = calibration_batch(
            torch.tensor([[1]]),
            torch.tensor([[True]]),
            example_ids=("duplicate",),
        )
        second = calibration_batch(
            torch.tensor([[2]]),
            torch.tensor([[True]]),
            example_ids=("duplicate",),
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate calibration example_id",
        ):
            collect_block_causal_lag_jacobian(
                self.adapter,
                self.plan,
                (first, second),
                input_codec=self.input_codec,
                output_codec=self.output_codec,
                input_modes=self.width,
                output_modes=self.width,
                max_lag=0,
            )

    def test_statistics_strict_weights_only_round_trip_and_tamper(
        self,
    ) -> None:
        batch = calibration_batch(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[True, True, True]]),
            example_ids=("serial",),
        )
        result = collect_block_causal_lag_jacobian(
            self.adapter,
            self.plan,
            (batch,),
            input_codec=self.input_codec,
            output_codec=self.output_codec,
            input_modes=self.width,
            output_modes=self.width,
            max_lag=2,
        )
        payload = io.BytesIO()
        torch.save(result.state_dict(), payload)
        payload.seek(0)
        restored = CausalLagJacobianStatistics.from_state_dict(
            torch.load(payload, weights_only=True)
        )

        self.assertEqual(restored.metadata(), result.metadata())
        torch.testing.assert_close(restored.mean, result.mean)
        torch.testing.assert_close(restored.rms, result.rms)
        detached_state = result.state_dict()
        detached_state["mean"][0, 0, 0] += 1.0
        self.assertNotEqual(
            detached_state["mean"][0, 0, 0],
            result.mean[0, 0, 0],
        )

        unknown = result.state_dict()
        unknown["model_state_dict"] = {}
        with self.assertRaisesRegex(ValueError, "fields"):
            CausalLagJacobianStatistics.from_state_dict(unknown)

        wrong_type = copy.deepcopy(result.state_dict())
        wrong_type["sequences"] = 1.0
        with self.assertRaisesRegex(TypeError, "sequences"):
            CausalLagJacobianStatistics.from_state_dict(wrong_type)

        bad_fraction = copy.deepcopy(result.state_dict())
        bad_fraction["captured_energy_fraction"] = 0.123
        with self.assertRaisesRegex(ValueError, "derived metadata"):
            CausalLagJacobianStatistics.from_state_dict(bad_fraction)

        bad_regime_fraction = copy.deepcopy(result.state_dict())
        bad_regime_fraction[
            "regime_variation_fraction_of_captured"
        ] = 0.123
        with self.assertRaisesRegex(ValueError, "derived metadata"):
            CausalLagJacobianStatistics.from_state_dict(
                bad_regime_fraction
            )

        bad_accounting = copy.deepcopy(result.state_dict())
        bad_accounting["total_squared_sum"] += 1.0
        with self.assertRaisesRegex(ValueError, "energy accounting"):
            CausalLagJacobianStatistics.from_state_dict(bad_accounting)

        bad_rms = copy.deepcopy(result.state_dict())
        bad_rms["rms"][0, 0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "rms cannot"):
            CausalLagJacobianStatistics.from_state_dict(bad_rms)


if __name__ == "__main__":
    unittest.main()
