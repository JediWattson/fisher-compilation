import unittest

import torch
import torch.nn.functional as F

from fisher_graph import (
    GraphLayerExecutor,
    ToyTransformer,
    TransformerConfig,
    empirical_activation_fisher,
)


def make_model() -> ToyTransformer:
    torch.manual_seed(11)
    return ToyTransformer(
        TransformerConfig(
            vocab_size=19,
            max_sequence_length=8,
            d_model=12,
            n_heads=3,
            n_layers=2,
            d_ff=24,
        )
    )


class InstrumentedTransformerTests(unittest.TestCase):
    def test_forward_exposes_named_internal_activations_and_gradients(self) -> None:
        model = make_model()
        inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])

        output = model(inputs, capture_activations=True)
        self.assertEqual(output.logits.shape, (2, 4, 19))
        self.assertIsNotNone(output.activations)
        trace = output.activations
        assert trace is not None
        self.assertIn("layer.0.attention.probabilities", trace)
        self.assertIn("layer.1.mlp.activated", trace)
        self.assertEqual(
            trace["layer.0.attention.probabilities"].shape, (2, 3, 4, 4)
        )

        loss = F.cross_entropy(output.logits.flatten(0, 1), targets.flatten())
        loss.backward()
        gradients = trace.gradients(strict=True)
        self.assertEqual(
            gradients["layer.0.output"].shape,
            trace["layer.0.output"].shape,
        )

    def test_fisher_is_per_example_nonnegative_and_shape_preserving(self) -> None:
        model = make_model()
        inputs = torch.tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        targets = torch.tensor([[2, 3, 4], [4, 6, 8], [6, 9, 12]])
        names = {"layer.0.output", "layer.0.attention.probabilities"}

        report = empirical_activation_fisher(
            model, inputs, targets, activations=names
        )

        self.assertEqual(report.samples, 3)
        self.assertEqual(set(report.activations), names)
        self.assertEqual(report.activations["layer.0.output"].diagonal.shape, (3, 12))
        self.assertEqual(
            report.activations[
                "layer.0.attention.probabilities"
            ].diagonal.shape,
            (3, 3, 3),
        )
        for entry in report.activations.values():
            self.assertTrue(torch.isfinite(entry.diagonal).all())
            self.assertTrue((entry.diagonal >= 0).all())
            self.assertEqual(entry.samples, 3)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_batched_fisher_equals_mean_of_individual_estimates(self) -> None:
        model = make_model()
        model.train()
        inputs = torch.tensor([[1, 2, 3], [2, 4, 6]])
        targets = torch.tensor([[2, 3, 4], [4, 6, 8]])
        name = "layer.0.output"

        batched = empirical_activation_fisher(
            model, inputs, targets, activations={name}
        )
        individual = [
            empirical_activation_fisher(
                model,
                inputs[index : index + 1],
                targets[index : index + 1],
                activations={name},
            )
            for index in range(inputs.shape[0])
        ]
        expected = torch.stack(
            [report.activations[name].diagonal for report in individual]
        ).mean(dim=0)

        torch.testing.assert_close(
            batched.activations[name].diagonal,
            expected,
        )
        self.assertTrue(model.training)

    def test_graph_executor_is_numerically_equivalent_to_standard_block(self) -> None:
        model = make_model().eval()
        inputs = torch.tensor([[1, 2, 3, 4]])
        expected = model(inputs, capture_activations=True)

        block = model.layers[0]
        graph = GraphLayerExecutor.from_transformer_block(block)
        model.replace_layer(0, graph)
        actual = model(inputs, capture_activations=True)

        torch.testing.assert_close(actual.logits, expected.logits)
        assert actual.activations is not None
        self.assertIn("layer.0.attention.query", actual.activations)
        self.assertIn("layer.0.output", actual.activations)

    def test_validation_rejects_incompatible_head_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerConfig(d_model=10, n_heads=3)


if __name__ == "__main__":
    unittest.main()
