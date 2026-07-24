import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor

from fisher_graph import (
    ActivationTrace,
    CausalModalGraph,
    CausalModalMLPGraph,
    FisherModeBasis,
    LayerExecutor,
    ModalExecutorConfig,
    PositionConditionedModalBottleneckExecutor,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
    load_position_modal_executor,
    save_position_modal_executor,
)


def identity_basis(name: str = "tap") -> FisherModeBasis:
    return FisherModeBasis(
        activation_name=name,
        mean=torch.tensor([100.0, 200.0], dtype=torch.float64),
        matrix=torch.eye(2, dtype=torch.float64),
        eigenvalues=torch.tensor([2.0, 1.0], dtype=torch.float64),
        vectors=torch.eye(2, dtype=torch.float64),
        observations=6,
        sequences=2,
        position_means=torch.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            dtype=torch.float64,
        ),
    )


class IdentityLayer(LayerExecutor):
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        del attention_mask, trace, prefix
        return hidden_states


class ModalExecutorTests(unittest.TestCase):
    def test_position_projection_uses_position_means_and_round_trips(self) -> None:
        basis = identity_basis()
        activations = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0], [59.0, 71.0]]]
        )
        full = PositionConditionedModalProjection.from_basis(
            basis,
            modes=2,
        )
        leading = PositionConditionedModalProjection.from_basis(
            basis,
            modes=1,
        )

        torch.testing.assert_close(full(activations), activations)
        torch.testing.assert_close(
            leading(activations),
            torch.tensor(
                [[[13.0, 20.0], [35.0, 40.0], [59.0, 60.0]]]
            ),
        )
        self.assertIn("position_mean", full.state_dict())
        self.assertEqual(full.to(torch.float64).vectors.dtype, torch.float64)

    def test_linear_modal_graph_is_causal_and_matches_manual_equation(self) -> None:
        weights = torch.zeros(3, 3, 2, 2)
        weights[0, 0] = torch.eye(2)
        weights[0, 2] = torch.full((2, 2), 100.0)
        weights[1, 0, 0, 0] = 2
        weights[1, 1] = torch.eye(2)
        weights[2, 0, 1, 1] = -1
        weights[2, 2] = torch.eye(2)
        bias = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        graph = CausalModalGraph(weights, bias)
        coordinates = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        )

        actual = graph(coordinates)
        causal_mask = torch.ones(3, 3, dtype=torch.bool).tril().view(
            3, 3, 1, 1
        )
        expected = torch.einsum(
            "bsi,tsio->bto",
            coordinates,
            weights * causal_mask,
        ) + bias
        torch.testing.assert_close(actual, expected)
        self.assertEqual(graph.edge_count, 24)
        self.assertEqual(graph.possible_edge_count, 24)

        changed_future = coordinates.clone()
        changed_future[:, 2] += 1_000
        torch.testing.assert_close(
            graph(changed_future)[:, :2],
            actual[:, :2],
        )
        invalid_mask = torch.ones_like(weights, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "noncausal"):
            CausalModalGraph(weights, bias, edge_mask=invalid_mask)

    def test_bottleneck_consumes_modal_interventions(self) -> None:
        basis = identity_basis()
        projection = PositionConditionedModalProjection.from_basis(
            basis,
            modes=1,
        )
        executor = PositionConditionedModalBottleneckExecutor(
            IdentityLayer(),
            projection,
            projection,
        )
        inputs = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0], [59.0, 71.0]]]
        )
        trace = ActivationTrace(
            interventions={
                "layer.0.modal.input": lambda value: value * 0,
            }
        )

        output = executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            trace=trace,
            prefix="layer.0",
        )

        assert basis.position_means is not None
        torch.testing.assert_close(
            output,
            basis.position_means.float().unsqueeze(0),
        )
        torch.testing.assert_close(
            trace["layer.0.modal.input"],
            torch.zeros(1, 3, 1),
        )

    def test_nonlinear_hidden_tap_is_causal_and_intervenable(self) -> None:
        graph = CausalModalMLPGraph(
            input_modes=2,
            output_modes=2,
            sequence_length=3,
            hidden_modes=3,
            input_scale=torch.ones(3, 2),
            output_scale=torch.ones(3, 2),
        )
        projection = PositionConditionedModalProjection.from_basis(
            identity_basis(),
            modes=2,
        )
        executor = PositionConditionedModalGraphExecutor(
            projection,
            graph,
            projection,
        )
        inputs = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0], [59.0, 71.0]]],
            requires_grad=True,
        )
        baseline = executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix="layer.0",
        )
        modal_inputs = projection.encode(inputs.detach())
        changed_future = modal_inputs.clone()
        changed_future[:, 2] += 1_000
        torch.testing.assert_close(
            graph(changed_future)[:, :2],
            graph(modal_inputs)[:, :2],
        )
        trace = ActivationTrace(
            interventions={
                "layer.0.modal.hidden": lambda value: value * 0,
            }
        )
        intervened = executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            trace=trace,
            prefix="layer.0",
        )
        self.assertFalse(torch.equal(intervened, baseline))
        intervened.sum().backward()
        self.assertIsNotNone(inputs.grad)
        with self.assertRaisesRegex(ValueError, "does not support padding"):
            executor(
                inputs.detach(),
                attention_mask=torch.tensor([[True, True, False]]),
                prefix="layer.0",
            )

    def test_nonlinear_executor_artifact_round_trip(self) -> None:
        input_basis = identity_basis("input")
        output_basis = identity_basis("output")
        graph = CausalModalMLPGraph(
            input_modes=2,
            output_modes=2,
            sequence_length=3,
            hidden_modes=3,
            input_scale=torch.ones(3, 2),
            output_scale=torch.ones(3, 2),
        )
        executor = PositionConditionedModalGraphExecutor(
            PositionConditionedModalProjection.from_basis(
                input_basis,
                modes=2,
            ),
            graph,
            PositionConditionedModalProjection.from_basis(
                output_basis,
                modes=2,
            ),
        ).to(torch.float64)
        config = ModalExecutorConfig(
            input_activation="input",
            output_activation="output",
            sequence_length=3,
            input_modes=2,
            output_modes=2,
            routing_width=3,
        )
        inputs = torch.tensor(
            [[[13.0, 24.0], [35.0, 47.0], [59.0, 71.0]]],
            dtype=torch.float64,
        )
        expected = executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix="layer.0",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "executor.pt"
            save_position_modal_executor(
                path,
                executor=executor,
                config=config,
                metadata={"checkpoint_sha256": "abc"},
            )
            loaded, loaded_config, metadata = (
                load_position_modal_executor(path)
            )

        actual = loaded(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix="layer.0",
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            loaded.graph.input_layers[0].weight.dtype,
            torch.float64,
        )
        self.assertEqual(loaded_config, config)
        self.assertEqual(metadata["checkpoint_sha256"], "abc")

    def test_artifact_save_rejects_mismatched_config(self) -> None:
        basis = identity_basis()
        executor = PositionConditionedModalGraphExecutor(
            PositionConditionedModalProjection.from_basis(basis, modes=2),
            CausalModalMLPGraph(
                input_modes=2,
                output_modes=2,
                sequence_length=3,
                hidden_modes=3,
                input_scale=torch.ones(3, 2),
                output_scale=torch.ones(3, 2),
            ),
            PositionConditionedModalProjection.from_basis(basis, modes=2),
        )
        mismatched = ModalExecutorConfig(
            input_activation="wrong",
            output_activation="tap",
            sequence_length=3,
            input_modes=2,
            output_modes=2,
            routing_width=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "does not match"):
                save_position_modal_executor(
                    Path(directory) / "executor.pt",
                    executor=executor,
                    config=mismatched,
                    metadata={},
                )


if __name__ == "__main__":
    unittest.main()
