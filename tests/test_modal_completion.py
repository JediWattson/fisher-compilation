import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor, nn

from fisher_graph import (
    ActivationTrace,
    CausalModalGraph,
    FisherModeBasis,
    LayerExecutor,
    LocalModalCompletionGraph,
    ModalCompletionFitConfig,
    PositionConditionedCompletedModalGraphExecutor,
    PositionConditionedModalCompletion,
    PositionConditionedModalCompletionBottleneckExecutor,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
    fit_local_modal_completion,
    load_position_modal_completion,
    save_position_modal_completion,
)


def rotated_basis(
    name: str = "boundary",
    *,
    sequence_length: int = 3,
) -> FisherModeBasis:
    root_two = 2.0**0.5
    vectors = torch.tensor(
        [
            [1.0 / root_two, -1.0 / root_two, 0.0],
            [1.0 / root_two, 1.0 / root_two, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    position_means = (
        torch.arange(
            sequence_length * 3,
            dtype=torch.float64,
        ).reshape(sequence_length, 3)
        / 10.0
    )
    return FisherModeBasis(
        activation_name=name,
        mean=position_means.mean(dim=0),
        matrix=torch.eye(3, dtype=torch.float64),
        eigenvalues=torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        vectors=vectors,
        observations=sequence_length * 8,
        sequences=8,
        position_means=position_means,
    )


def decode_coordinates(
    basis: FisherModeBasis,
    coordinates: Tensor,
) -> Tensor:
    return basis.reconstruct(
        coordinates,
        centering="position",
    )


def make_completion(
    basis: FisherModeBasis,
    *,
    shared_weights: bool = True,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> PositionConditionedModalCompletion:
    sequence_length = basis.position_means.shape[0]  # type: ignore[union-attr]
    if weight is None:
        if shared_weights:
            weight = torch.tensor([[0.75], [-0.5]], dtype=dtype)
        else:
            weight = torch.tensor(
                [
                    [[0.75], [-0.5]],
                    [[-0.25], [1.25]],
                    [[1.5], [0.125]],
                ][:sequence_length],
                dtype=dtype,
            )
    if bias is None:
        bias = torch.arange(
            sequence_length,
            dtype=dtype,
        ).reshape(sequence_length, 1) / 4.0
    graph = LocalModalCompletionGraph(
        weight,
        bias,
        shared_weights=shared_weights,
    )
    return PositionConditionedModalCompletion.from_basis(
        basis,
        graph,
        dtype=dtype,
    )


class FrozenMixingLayer(LayerExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, -0.25],
                    [0.2, 0.3, 1.0],
                ],
                dtype=torch.float64,
            ),
            requires_grad=False,
        )
        self.last_input: Tensor | None = None

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        del attention_mask, trace, prefix
        self.last_input = hidden_states.detach().clone()
        return hidden_states @ self.weight.transpose(0, 1)


class ModalCompletionTests(unittest.TestCase):
    def test_oracle_completion_round_trips_full_activation(self) -> None:
        basis = rotated_basis()
        kept = torch.tensor(
            [
                [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]],
                [[-2.0, 0.5], [1.5, 2.5], [3.0, -3.0]],
            ],
            dtype=torch.float64,
        )
        weight = torch.tensor([[2.0], [-0.75]], dtype=torch.float64)
        bias = torch.tensor(
            [[0.25], [-0.5], [1.25]],
            dtype=torch.float64,
        )
        tail = kept @ weight + bias
        coordinates = torch.cat((kept, tail), dim=-1)
        activations = decode_coordinates(basis, coordinates)
        completion = make_completion(
            basis,
            weight=weight,
            bias=bias,
        )

        actual = completion(
            activations,
            attention_mask=torch.ones(2, 3, dtype=torch.bool),
            prefix="bridge",
        )

        torch.testing.assert_close(
            actual,
            activations,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            completion.encode_tail(actual),
            tail,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_completion_equals_existing_truncated_projection(self) -> None:
        basis = rotated_basis()
        coordinates = torch.tensor(
            [
                [[1.0, 2.0, 9.0], [3.0, 4.0, -8.0], [5.0, 6.0, 7.0]]
            ],
            dtype=torch.float64,
        )
        activations = decode_coordinates(basis, coordinates)
        zero_completion = make_completion(
            basis,
            weight=torch.zeros(2, 1, dtype=torch.float64),
            bias=torch.zeros(3, 1, dtype=torch.float64),
        )
        truncated = PositionConditionedModalProjection.from_basis(
            basis,
            modes=2,
            dtype=torch.float64,
        )

        actual = zero_completion(
            activations,
            prefix="bridge",
        )

        torch.testing.assert_close(
            actual,
            truncated(activations),
            rtol=0,
            atol=0,
        )

    def test_shared_affine_fit_recovers_exact_map(self) -> None:
        basis = rotated_basis()
        generator = torch.Generator().manual_seed(123)
        kept = torch.randn(
            128,
            3,
            2,
            generator=generator,
            dtype=torch.float64,
        )
        weight = torch.tensor(
            [[1.25], [-0.375]],
            dtype=torch.float64,
        )
        bias = torch.tensor(
            [[0.5], [-1.0], [1.75]],
            dtype=torch.float64,
        )
        tail = kept @ weight + bias
        activations = decode_coordinates(
            basis,
            torch.cat((kept, tail), dim=-1),
        )

        completion, report = fit_local_modal_completion(
            activations,
            basis,
            kept_modes=2,
            shared_weights=True,
            fit_config=ModalCompletionFitConfig(ridge=0),
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            completion.graph.weight,
            weight,
            rtol=1e-11,
            atol=1e-11,
        )
        torch.testing.assert_close(
            completion.graph.bias,
            bias,
            rtol=1e-11,
            atol=1e-11,
        )
        torch.testing.assert_close(
            completion(activations, prefix="bridge"),
            activations,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertGreaterEqual(report.train_tail_r_squared, 1 - 1e-12)
        self.assertLess(report.train_tail_rmse, 1e-12)
        self.assertGreaterEqual(
            report.minimum_nonconstant_position_r_squared,
            1 - 1e-12,
        )
        self.assertEqual(report.constant_position_count, 0)
        self.assertEqual(report.learned_parameters, 5)
        self.assertEqual(report.map_multiplies_per_sequence, 6)

    def test_position_local_affine_fit_recovers_exact_maps(self) -> None:
        basis = rotated_basis()
        generator = torch.Generator().manual_seed(456)
        kept = torch.randn(
            128,
            3,
            2,
            generator=generator,
            dtype=torch.float64,
        )
        weight = torch.tensor(
            [
                [[1.0], [-0.5]],
                [[-0.25], [1.5]],
                [[2.0], [0.125]],
            ],
            dtype=torch.float64,
        )
        bias = torch.tensor(
            [[-0.25], [0.75], [1.5]],
            dtype=torch.float64,
        )
        tail = torch.einsum("bti,tio->bto", kept, weight) + bias
        activations = decode_coordinates(
            basis,
            torch.cat((kept, tail), dim=-1),
        )

        completion, report = fit_local_modal_completion(
            activations,
            basis,
            kept_modes=2,
            shared_weights=False,
            fit_config=ModalCompletionFitConfig(ridge=0),
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            completion.graph.weight,
            weight,
            rtol=1e-11,
            atol=1e-11,
        )
        torch.testing.assert_close(
            completion.graph.bias,
            bias,
            rtol=1e-11,
            atol=1e-11,
        )
        torch.testing.assert_close(
            completion(activations, prefix="bridge"),
            activations,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertGreaterEqual(report.train_tail_r_squared, 1 - 1e-12)
        self.assertLess(report.train_tail_rmse, 1e-12)
        self.assertGreaterEqual(
            report.minimum_nonconstant_position_r_squared,
            1 - 1e-12,
        )
        self.assertEqual(report.constant_position_count, 0)
        self.assertEqual(report.learned_parameters, 9)
        self.assertEqual(report.map_multiplies_per_sequence, 6)

    def test_local_completion_is_strictly_position_local_and_causal(self) -> None:
        basis = rotated_basis()
        generator = torch.Generator().manual_seed(789)
        for shared_weights in (True, False):
            with self.subTest(shared_weights=shared_weights):
                completion = make_completion(
                    basis,
                    shared_weights=shared_weights,
                )
                kept = torch.randn(
                    2,
                    3,
                    2,
                    generator=generator,
                    dtype=torch.float64,
                )
                completed = completion.complete(
                    kept,
                    prefix="bridge",
                )
                changed_future = kept.clone()
                changed_future[:, 2] += 1000
                changed_completed = completion.complete(
                    changed_future,
                    prefix="bridge",
                )
                torch.testing.assert_close(
                    changed_completed[:, :2],
                    completed[:, :2],
                    rtol=0,
                    atol=0,
                )
                self.assertFalse(
                    torch.equal(
                        changed_completed[:, 2],
                        completed[:, 2],
                    )
                )

                differentiable = kept.clone().requires_grad_(True)
                position_one = completion.complete(
                    differentiable,
                    prefix="bridge",
                )[:, 1].sum()
                gradient = torch.autograd.grad(
                    position_one,
                    differentiable,
                )[0]
                torch.testing.assert_close(
                    gradient[:, 0],
                    torch.zeros_like(gradient[:, 0]),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    gradient[:, 2],
                    torch.zeros_like(gradient[:, 2]),
                    rtol=0,
                    atol=0,
                )

    def test_frozen_bottleneck_consumes_every_completion_intervention(
        self,
    ) -> None:
        basis = rotated_basis()
        input_completion = make_completion(basis)
        output_completion = make_completion(
            basis,
            weight=torch.tensor([[-0.5], [0.75]], dtype=torch.float64),
            bias=torch.tensor(
                [[0.125], [-0.25], [0.5]],
                dtype=torch.float64,
            ),
        )
        inner = FrozenMixingLayer()
        executor = PositionConditionedModalCompletionBottleneckExecutor(
            inner,
            input_completion=input_completion,
            output_completion=output_completion,
        )
        coordinates = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, -1.0, 2.0], [0.5, 3.0, -2.0]]
            ],
            dtype=torch.float64,
        )
        inputs = decode_coordinates(basis, coordinates)
        attention_mask = torch.ones(1, 3, dtype=torch.bool)
        baseline = executor(
            inputs,
            attention_mask=attention_mask,
            prefix="layer.0",
        )
        parameter_snapshot = {
            name: value.detach().clone()
            for name, value in inner.state_dict().items()
        }
        tap_names = (
            "layer.0.modal.input",
            "layer.0.modal.input_completion.tail",
            "layer.0.modal.input_completion.coordinates",
            "layer.0.modal.input_reconstruction",
            "layer.0.modal.full_output",
            "layer.0.modal.output",
            "layer.0.modal.output_completion.tail",
            "layer.0.modal.output_completion.coordinates",
            "layer.0.output",
        )
        for tap_name in tap_names:
            with self.subTest(tap_name=tap_name):
                trace = ActivationTrace(
                    interventions={
                        tap_name: lambda value: value
                        + torch.ones_like(value) * 0.375,
                    }
                )
                intervened = executor(
                    inputs,
                    attention_mask=attention_mask,
                    trace=trace,
                    prefix="layer.0",
                )
                trace.assert_all_interventions_applied()
                self.assertIn(tap_name, trace)
                self.assertFalse(torch.equal(intervened, baseline))

        self.assertIsNotNone(inner.last_input)
        assert inner.last_input is not None
        self.assertEqual(inner.last_input.shape, inputs.shape)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in inner.parameters()
            )
        )
        for name, expected in parameter_snapshot.items():
            torch.testing.assert_close(
                inner.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    def test_completed_modal_graph_decodes_predicted_tail(self) -> None:
        basis = rotated_basis()
        input_projection = PositionConditionedModalProjection.from_basis(
            basis,
            modes=2,
            dtype=torch.float64,
        )
        output_projection = PositionConditionedModalProjection.from_basis(
            basis,
            modes=2,
            dtype=torch.float64,
        )
        weights = torch.zeros(3, 3, 2, 2, dtype=torch.float64)
        for position in range(3):
            weights[position, position] = torch.eye(
                2,
                dtype=torch.float64,
            )
        base_executor = PositionConditionedModalGraphExecutor(
            input_projection,
            CausalModalGraph(
                weights,
                torch.zeros(3, 2, dtype=torch.float64),
            ),
            output_projection,
        )
        completion_weight = torch.tensor(
            [[1.0], [-2.0]],
            dtype=torch.float64,
        )
        output_completion = make_completion(
            basis,
            weight=completion_weight,
            bias=torch.tensor(
                [[0.25], [0.5], [0.75]],
                dtype=torch.float64,
            ),
        )
        executor = PositionConditionedCompletedModalGraphExecutor(
            base_executor,
            output_completion,
        )
        input_coordinates = torch.tensor(
            [
                [[1.0, 2.0, 9.0], [3.0, 4.0, -8.0], [5.0, 6.0, 7.0]]
            ],
            dtype=torch.float64,
        )
        inputs = decode_coordinates(basis, input_coordinates)
        kept = input_coordinates[..., :2]
        expected_coordinates = torch.cat(
            (
                kept,
                kept @ completion_weight
                + torch.tensor(
                    [[0.25], [0.5], [0.75]],
                    dtype=torch.float64,
                ),
            ),
            dim=-1,
        )
        expected = decode_coordinates(basis, expected_coordinates)

        actual = executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix="layer.0",
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        zero_tail_trace = ActivationTrace(
            interventions={
                "layer.0.modal.output_completion.tail": torch.zeros_like,
            }
        )
        zero_tail = executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            trace=zero_tail_trace,
            prefix="layer.0",
        )
        base_output = base_executor(
            inputs,
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            prefix="layer.0",
        )
        zero_tail_trace.assert_all_interventions_applied()
        torch.testing.assert_close(
            zero_tail,
            base_output,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_float64_artifact_round_trip_for_both_graph_kinds(self) -> None:
        basis = rotated_basis()
        inputs = decode_coordinates(
            basis,
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ],
                dtype=torch.float64,
            ),
        )
        for shared_weights in (True, False):
            with self.subTest(shared_weights=shared_weights):
                completion = make_completion(
                    basis,
                    shared_weights=shared_weights,
                )
                expected = completion(inputs, prefix="bridge")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "completion.pt"
                    save_position_modal_completion(
                        path,
                        completion=completion,
                        metadata={
                            "checkpoint_sha256": "checkpoint",
                            "fisher_sha256": "fisher",
                        },
                    )
                    payload = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    loaded, config, metadata = (
                        load_position_modal_completion(path)
                    )

                actual = loaded(inputs, prefix="bridge")
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(loaded.graph.weight.dtype, torch.float64)
                self.assertEqual(
                    loaded.full_projection.vectors.dtype,
                    torch.float64,
                )
                self.assertEqual(
                    config.graph_kind,
                    (
                        "shared_local_linear"
                        if shared_weights
                        else "position_local_linear"
                    ),
                )
                self.assertEqual(
                    metadata["checkpoint_sha256"],
                    "checkpoint",
                )
                self.assertEqual(
                    set(payload["completion_state_dict"]),
                    {
                        "full_projection.position_mean",
                        "full_projection.vectors",
                        "graph.weight",
                        "graph.bias",
                    },
                )

    def test_invalid_graph_projection_shapes_dtypes_and_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "bias must have shape"):
            LocalModalCompletionGraph(
                torch.ones(2, 1),
                torch.ones(3),
                shared_weights=True,
            )
        with self.assertRaisesRegex(ValueError, "shared completion weight"):
            LocalModalCompletionGraph(
                torch.ones(3, 2, 1),
                torch.ones(3, 1),
                shared_weights=True,
            )
        with self.assertRaisesRegex(ValueError, "position completion weight"):
            LocalModalCompletionGraph(
                torch.ones(2, 2, 1),
                torch.ones(3, 1),
                shared_weights=False,
            )
        graph = LocalModalCompletionGraph(
            torch.ones(2, 1),
            torch.zeros(3, 1),
            shared_weights=True,
        )
        with self.assertRaisesRegex(ValueError, "completion coordinates"):
            graph(torch.ones(1, 2, 2))
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            graph(
                torch.ones(1, 3, 2),
                attention_mask=torch.ones(1, 2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "does not support padding"):
            graph(
                torch.ones(1, 3, 2),
                attention_mask=torch.tensor([[True, True, False]]),
            )
        with self.assertRaisesRegex(ValueError, "dtype and device"):
            graph(torch.ones(1, 3, 2, dtype=torch.float64))

        basis = rotated_basis()
        truncated = PositionConditionedModalProjection.from_basis(
            basis,
            modes=2,
        )
        with self.assertRaisesRegex(ValueError, "complete square"):
            PositionConditionedModalCompletion(truncated, graph)
        bad_projection = PositionConditionedModalProjection(
            activation_name="boundary",
            position_mean=basis.position_means.float(),  # type: ignore[union-attr]
            vectors=torch.diag(torch.tensor([2.0, 1.0, 1.0])),
        )
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            PositionConditionedModalCompletion(
                bad_projection,
                graph,
            )

        with self.assertRaisesRegex(ValueError, "at least one"):
            PositionConditionedModalCompletionBottleneckExecutor(
                FrozenMixingLayer(),
            )
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            fit_local_modal_completion(
                torch.ones(1, 3, 3),
                basis,
                kept_modes=2,
                shared_weights=True,
            )
        completion = make_completion(basis)
        with self.assertRaisesRegex(ValueError, "does not support padding"):
            completion(
                torch.ones(1, 3, 3, dtype=torch.float64),
                attention_mask=torch.tensor([[True, False, False]]),
                prefix="bridge",
            )


if __name__ == "__main__":
    unittest.main()
