import unittest

import torch

from fisher_graph import (
    FisherModeBasis,
    FisherModeSuppression,
    GraphLayerExecutor,
    ToyTransformer,
    TransformerConfig,
)
from fisher_graph.intervention_experiment import _energy_matched_fraction


def identity_basis() -> FisherModeBasis:
    return FisherModeBasis(
        activation_name="layer.0.output",
        mean=torch.tensor([50.0, 50.0], dtype=torch.float64),
        matrix=torch.eye(2, dtype=torch.float64),
        eigenvalues=torch.tensor([2.0, 1.0], dtype=torch.float64),
        vectors=torch.eye(2, dtype=torch.float64),
        observations=4,
        sequences=2,
        position_means=torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]],
            dtype=torch.float64,
        ),
    )


class FisherInterventionTests(unittest.TestCase):
    def test_energy_match_fraction_is_exact_or_fails(self) -> None:
        self.assertAlmostEqual(
            _energy_matched_fraction(
                target_rms=0.2,
                control_rms_at_reference=0.1,
                reference_fraction=0.25,
            ),
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "exceeding full suppression"):
            _energy_matched_fraction(
                target_rms=1.0,
                control_rms_at_reference=0.1,
                reference_fraction=0.25,
            )

    def test_full_mode_suppression_uses_position_conditioned_mean(self) -> None:
        basis = identity_basis()
        activation = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0]]]
        )
        suppress_first = FisherModeSuppression(
            basis=basis,
            mode_indices=(0,),
        )

        actual = suppress_first(activation)

        expected = torch.tensor(
            [[[10.0, 24.0], [30.0, 47.0]]]
        )
        torch.testing.assert_close(actual, expected)

    def test_suppressing_all_modes_returns_each_positions_mean(self) -> None:
        basis = identity_basis()
        activation = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0]]]
        )

        actual = FisherModeSuppression(
            basis=basis,
            mode_indices=(0, 1),
        )(activation)

        torch.testing.assert_close(
            actual,
            basis.position_means.float().unsqueeze(0),
        )

    def test_fractional_and_position_specific_suppression(self) -> None:
        basis = identity_basis()
        activation = torch.tensor(
            [[[14.0, 22.0], [36.0, 44.0]]]
        )

        actual = FisherModeSuppression(
            basis=basis,
            mode_indices=(0,),
            suppression_fraction=0.5,
            positions=(1,),
        )(activation)

        expected = torch.tensor(
            [[[14.0, 22.0], [33.0, 44.0]]]
        )
        torch.testing.assert_close(actual, expected)

    def test_suppression_preserves_autograd(self) -> None:
        activation = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0]]],
            requires_grad=True,
        )

        output = FisherModeSuppression(
            basis=identity_basis(),
            mode_indices=(0,),
            suppression_fraction=0.25,
        )(activation)
        output.sum().backward()

        self.assertIsNotNone(activation.grad)
        torch.testing.assert_close(
            activation.grad,
            torch.tensor([[[0.75, 1.0], [0.75, 1.0]]]),
        )

    def test_model_intervention_changes_downstream_logits_without_capture(self) -> None:
        torch.manual_seed(9)
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=11,
                max_sequence_length=3,
                d_model=4,
                n_heads=2,
                n_layers=1,
                d_ff=8,
            )
        ).eval()
        inputs = torch.tensor([[1, 2, 3]])
        baseline = model(inputs).logits

        intervened = model(
            inputs,
            activation_interventions={
                "layer.0.output": lambda tensor: tensor * 0
            },
        )

        self.assertIsNone(intervened.activations)
        self.assertFalse(torch.equal(intervened.logits, baseline))

    def test_capture_contains_post_intervention_activation_and_gradients(self) -> None:
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=11,
                max_sequence_length=3,
                d_model=4,
                n_heads=2,
                n_layers=1,
                d_ff=8,
            )
        )
        output = model(
            torch.tensor([[1, 2, 3]]),
            capture_activations=True,
            activation_interventions={
                "layer.0.output": lambda tensor: tensor * 0.5
            },
        )
        assert output.activations is not None
        captured = output.activations["layer.0.output"]
        output.logits.sum().backward()

        self.assertIsNotNone(captured.grad)

    def test_unknown_or_invalid_intervention_fails(self) -> None:
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=7,
                max_sequence_length=2,
                d_model=4,
                n_heads=2,
                n_layers=1,
                d_ff=8,
            )
        )
        inputs = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(KeyError, "unknown"):
            model(
                inputs,
                activation_interventions={
                    "not.a.tap": lambda tensor: tensor
                },
            )
        with self.assertRaisesRegex(ValueError, "changed shape"):
            model(
                inputs,
                activation_interventions={
                    "layer.0.output": lambda tensor: tensor[..., :1]
                },
            )

    def test_graph_executor_matches_standard_block_under_intervention(self) -> None:
        torch.manual_seed(14)
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=9,
                max_sequence_length=3,
                d_model=4,
                n_heads=2,
                n_layers=1,
                d_ff=8,
            )
        ).eval()
        inputs = torch.tensor([[1, 2, 3]])
        interventions = {
            "layer.0.attention.output": lambda tensor: tensor * 0.25,
            "layer.0.mlp.activated": lambda tensor: tensor * 0.5,
        }
        expected = model(
            inputs,
            activation_interventions=interventions,
        ).logits

        block = model.layers[0]
        model.replace_layer(
            0,
            GraphLayerExecutor.from_transformer_block(block),
        )
        actual = model(
            inputs,
            activation_interventions=interventions,
        ).logits

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
