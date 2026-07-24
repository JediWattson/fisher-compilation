import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor

from fisher_graph import (
    ActivationGradientSamples,
    FisherModeBasis,
    LayerExecutor,
    ToyTransformer,
    TransformerConfig,
    collect_activation_score_gradients,
    decompose_fisher_modes,
    extract_modal_jacobian,
    fit_modal_transition,
    load_fisher_build,
    save_fisher_build,
)


def make_samples(
    *,
    name: str,
    activations: Tensor,
    gradients: Tensor,
) -> ActivationGradientSamples:
    observations = activations.shape[0]
    locations = torch.stack(
        (
            torch.arange(observations),
            torch.zeros(observations, dtype=torch.long),
        ),
        dim=1,
    )
    return ActivationGradientSamples(
        name=name,
        activations=activations,
        score_gradients=gradients,
        locations=locations,
        sequences=observations,
    )


class TokenMixingLinearLayer(LayerExecutor):
    def __init__(self, token_mix: Tensor, feature_map: Tensor) -> None:
        super().__init__()
        self.register_buffer("token_mix", token_mix)
        self.register_buffer("feature_map", feature_map)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace=None,
        prefix: str,
    ) -> Tensor:
        del attention_mask, trace, prefix
        mixed = torch.einsum("ts,bsd->btd", self.token_mix, hidden_states)
        return mixed @ self.feature_map


class FisherModeTests(unittest.TestCase):
    def test_decomposition_reconstructs_full_fisher_and_round_trips(self) -> None:
        torch.manual_seed(4)
        activations = torch.randn(12, 4)
        gradients = torch.randn(12, 4)
        samples = make_samples(
            name="tap",
            activations=activations,
            gradients=gradients,
        )

        basis = decompose_fisher_modes(samples)
        expected = gradients.double().T @ gradients.double() / gradients.shape[0]
        reconstructed = (
            basis.vectors
            @ torch.diag(basis.eigenvalues)
            @ basis.vectors.T
        )

        torch.testing.assert_close(basis.matrix, expected)
        torch.testing.assert_close(reconstructed, expected)
        torch.testing.assert_close(
            basis.vectors.T @ basis.vectors,
            torch.eye(4, dtype=torch.float64),
        )
        coordinates = basis.project(activations.double())
        torch.testing.assert_close(
            basis.reconstruct(coordinates),
            activations.double(),
        )
        self.assertTrue(
            torch.all(basis.retained_curve[1:] >= basis.retained_curve[:-1])
        )
        self.assertAlmostEqual(basis.retained_fraction(4), 1.0)

    def test_opposing_scores_have_nonzero_fisher(self) -> None:
        activations = torch.zeros(2, 2)
        gradients = torch.tensor([[1.0, -2.0], [-1.0, 2.0]])
        self.assertTrue(torch.equal(gradients.mean(dim=0), torch.zeros(2)))

        basis = decompose_fisher_modes(
            make_samples(
                name="opposing",
                activations=activations,
                gradients=gradients,
            )
        )

        self.assertGreater(basis.fisher_trace, 0.0)
        self.assertAlmostEqual(basis.eigenvalues[0].item(), 5.0)

    def test_position_centered_modal_round_trip(self) -> None:
        basis = FisherModeBasis(
            activation_name="tap",
            mean=torch.tensor([50.0, 60.0], dtype=torch.float64),
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
        activations = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0]]],
            dtype=torch.float64,
        )

        coordinates = basis.project(
            activations,
            centering="position",
        )

        torch.testing.assert_close(
            basis.reconstruct(coordinates, centering="position"),
            activations,
        )
        with self.assertRaisesRegex(ValueError, "sequence length"):
            basis.project(
                activations[:, :1],
                centering="position",
            )

    def test_score_collection_uses_summed_per_sequence_nll(self) -> None:
        torch.manual_seed(8)
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
        targets = torch.tensor([[-100, 4, 5]])
        name = "layer.0.output"

        output = model(
            inputs,
            capture_activations=True,
            retain_activation_gradients=False,
        )
        assert output.activations is not None
        activation = output.activations[name]
        loss = torch.nn.functional.cross_entropy(
            output.logits.flatten(0, 1),
            targets.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        expected_gradient = torch.autograd.grad(loss, activation)[0][0]

        collection = collect_activation_score_gradients(
            model,
            inputs,
            targets,
            activation_names={name},
        )
        samples = collection.samples[name]

        torch.testing.assert_close(
            samples.score_gradients,
            expected_gradient,
        )
        expected_fisher = (
            expected_gradient.double().T @ expected_gradient.double()
            / inputs.shape[1]
        )
        torch.testing.assert_close(
            decompose_fisher_modes(samples).matrix,
            expected_fisher,
        )

    def test_score_collection_rejects_supervision_on_padding(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "attention-valid"):
            collect_activation_score_gradients(
                model,
                torch.tensor([[1, 2]]),
                torch.tensor([[-100, 3]]),
                attention_mask=torch.tensor([[True, False]]),
                activation_names={"layer.0.output"},
            )

    def test_modal_jacobian_preserves_token_to_token_blocks(self) -> None:
        token_mix = torch.tensor([[1.0, 0.0], [0.25, 1.0]])
        feature_map = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        layer = TokenMixingLinearLayer(token_mix, feature_map)
        activations = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        locations = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        samples = ActivationGradientSamples(
            name="input",
            activations=activations,
            score_gradients=torch.ones_like(activations),
            locations=locations,
            sequences=2,
        )
        identity_basis = FisherModeBasis(
            activation_name="input",
            mean=torch.zeros(2, dtype=torch.float64),
            matrix=torch.eye(2, dtype=torch.float64),
            eigenvalues=torch.ones(2, dtype=torch.float64),
            vectors=torch.eye(2, dtype=torch.float64),
            observations=4,
            sequences=2,
        )
        output_basis = FisherModeBasis(
            activation_name="output",
            mean=torch.zeros(2, dtype=torch.float64),
            matrix=torch.eye(2, dtype=torch.float64),
            eigenvalues=torch.ones(2, dtype=torch.float64),
            vectors=torch.eye(2, dtype=torch.float64),
            observations=4,
            sequences=2,
        )

        jacobian = extract_modal_jacobian(
            layer,
            samples,
            identity_basis,
            output_basis,
            input_modes=2,
            output_modes=2,
        )

        expected = torch.empty(2, 2, 2, 2, dtype=torch.float64)
        for output_position in range(2):
            for output_mode in range(2):
                for input_position in range(2):
                    for input_mode in range(2):
                        expected[
                            output_position,
                            output_mode,
                            input_position,
                            input_mode,
                        ] = (
                            token_mix[output_position, input_position]
                            * feature_map[input_mode, output_mode]
                        )
        torch.testing.assert_close(jacobian.mean, expected)
        torch.testing.assert_close(jacobian.rms, expected.abs())
        self.assertEqual(jacobian.samples, 2)

    def test_modal_map_rejects_duplicate_and_missing_grid_locations(self) -> None:
        locations = torch.tensor([[0, 0], [0, 0], [1, 0], [1, 1]])
        samples = ActivationGradientSamples(
            name="tap",
            activations=torch.randn(4, 2),
            score_gradients=torch.randn(4, 2),
            locations=locations,
            sequences=2,
        )
        basis = decompose_fisher_modes(samples)

        with self.assertRaisesRegex(ValueError, "complete unique"):
            fit_modal_transition(
                samples,
                samples,
                basis,
                basis,
                input_modes=2,
                output_modes=2,
            )

    def test_fisher_artifact_round_trip(self) -> None:
        samples = make_samples(
            name="tap",
            activations=torch.eye(3),
            gradients=torch.eye(3),
        )
        basis = decompose_fisher_modes(samples)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modes.pt"
            save_fisher_build(
                path,
                bases={"tap": basis},
                transitions=[],
                metadata={"checkpoint_sha256": "abc"},
            )
            bases, transitions, jacobians, metadata = load_fisher_build(path)

        torch.testing.assert_close(bases["tap"].matrix, basis.matrix)
        torch.testing.assert_close(
            bases["tap"].position_means,
            basis.position_means,
        )
        self.assertEqual(transitions, [])
        self.assertEqual(jacobians, [])
        self.assertEqual(metadata["checkpoint_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
