import copy
import tempfile
import unittest
from pathlib import Path

import torch

from fisher_graph.gated_executor import (
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)


def make_executor(
    *,
    input_modes: int = 3,
    output_modes: int = 3,
    expert_count: int = 2,
    expert_rank: int = 2,
    router_width: int = 4,
    same_position_skip: bool = True,
    max_positive_lag: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> ResidualGatedCausalModalExecutor:
    return ResidualGatedCausalModalExecutor(
        GatedCausalModalExecutorConfig(
            input_modes=input_modes,
            output_modes=output_modes,
            expert_count=expert_count,
            expert_rank=expert_rank,
            router_width=router_width,
            same_position_skip=same_position_skip,
            max_positive_lag=max_positive_lag,
        ),
        dtype=dtype,
    )


class ResidualGatedCausalModalExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(107)

    def test_skip_initialization_is_exact_and_variable_length(self) -> None:
        executor = make_executor().eval()
        with torch.no_grad():
            executor.expert_output_weight.zero_()
        original_shapes = {
            name: tuple(value.shape)
            for name, value in executor.state_dict().items()
        }

        for length in (1, 4, 9):
            values = torch.randn(2, length, 3)
            actual = executor(values)
            torch.testing.assert_close(actual, values, rtol=0.0, atol=0.0)
            self.assertEqual(
                {
                    name: tuple(value.shape)
                    for name, value in executor.state_dict().items()
                },
                original_shapes,
            )

        with self.assertRaisesRegex(ValueError, "equal input and output"):
            GatedCausalModalExecutorConfig(
                input_modes=2,
                output_modes=3,
                expert_count=1,
                expert_rank=1,
                router_width=1,
                same_position_skip=True,
            )

    def test_same_position_and_positive_lag_paths_are_separate(self) -> None:
        executor = make_executor(
            input_modes=1,
            output_modes=1,
            expert_count=1,
            expert_rank=1,
            router_width=1,
            same_position_skip=False,
            dtype=torch.float64,
        )
        with torch.no_grad():
            executor.same_position_weight.fill_(2.0)
            executor.same_position_bias.fill_(0.5)
            executor.expert_input_weight.fill_(3.0)
            executor.expert_output_weight.fill_(5.0)
            # One expert always has router probability one.
            executor.router_query_weight.fill_(13.0)
            executor.router_key_weight.fill_(-7.0)
            executor.router_output_weight.fill_(11.0)
            executor.router_bias.fill_(17.0)

        values = torch.tensor(
            [[[1.0], [2.0], [4.0]]],
            dtype=torch.float64,
        )
        components = executor.forward_components(values)
        expected_same = 2.0 * values + 0.5
        expected_cross = torch.tensor(
            [[[0.0], [15.0], [45.0]]],
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            components.same_position_output,
            expected_same,
        )
        torch.testing.assert_close(
            components.positive_lag_output,
            expected_cross,
        )
        torch.testing.assert_close(
            components.output,
            expected_same + expected_cross,
        )
        self.assertFalse(
            components.positive_lag_mask.diagonal(dim1=1, dim2=2).any()
        )
        torch.testing.assert_close(
            components.router_probabilities[
                components.positive_lag_mask
            ],
            torch.ones(3, 1, dtype=torch.float64),
        )

    def test_future_slots_have_zero_forward_and_gradient_influence(self) -> None:
        executor = make_executor(expert_count=3).eval()
        prefix = torch.randn(2, 4, 3)
        prefix_output = executor(prefix)
        extended = torch.cat(
            (prefix, 1_000 * torch.randn(2, 5, 3)),
            dim=1,
        )
        extended_output = executor(extended)

        torch.testing.assert_close(
            prefix_output,
            extended_output[:, :4],
            rtol=0.0,
            atol=0.0,
        )

        extended.requires_grad_()
        first_three = executor(extended)[:, :3].sum()
        gradient = torch.autograd.grad(first_three, extended)[0]
        torch.testing.assert_close(
            gradient[:, 3:],
            torch.zeros_like(gradient[:, 3:]),
            rtol=0.0,
            atol=0.0,
        )

        components = executor.forward_components(extended.detach())
        self.assertFalse(
            components.router_probabilities.sum(dim=-1).triu(
                diagonal=0
            ).any()
        )

    def test_sparse_padding_matches_compact_valid_sequence(self) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 3, 3)
        positions = torch.tensor([[5, 7, 10]])
        compact = executor(
            values,
            logical_positions=positions,
        )

        padded_values = torch.randn(1, 6, 3) * 100_000
        padded_values[:, [1, 3, 5]] = values
        valid = torch.tensor(
            [[False, True, False, True, False, True]]
        )
        padded_positions = torch.tensor(
            [[-50, 5, -40, 7, -30, 10]]
        )
        padded = executor(
            padded_values,
            query_valid_mask=valid,
            key_valid_mask=valid,
            logical_positions=padded_positions,
            key_logical_positions=padded_positions,
        )

        torch.testing.assert_close(
            padded[:, [1, 3, 5]],
            compact,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            padded[:, [0, 2, 4]],
            torch.zeros(1, 3, 3),
            rtol=0.0,
            atol=0.0,
        )

    def test_max_positive_lag_uses_logical_not_tensor_distance(self) -> None:
        executor = make_executor(
            input_modes=1,
            output_modes=1,
            expert_count=1,
            expert_rank=1,
            router_width=1,
            same_position_skip=False,
            max_positive_lag=2,
        )
        with torch.no_grad():
            executor.same_position_weight.zero_()
            executor.same_position_bias.zero_()
            executor.expert_input_weight.fill_(1.0)
            executor.expert_output_weight.fill_(1.0)
        values = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]])
        positions = torch.tensor([[3, 4, 6, 7]])
        components = executor.forward_components(
            values,
            logical_positions=positions,
        )
        shifted = executor(
            values,
            logical_positions=positions + 1_000,
        )

        # lags: target 4 reads 3; target 6 reads 4; target 7 reads 6 only.
        torch.testing.assert_close(
            components.output,
            torch.tensor([[[0.0], [1.0], [2.0], [4.0]]]),
        )
        torch.testing.assert_close(shifted, components.output)
        self.assertEqual(
            int(components.positive_lag_mask.sum().item()),
            3,
        )

    def test_router_probabilities_normalize_only_on_allowed_edges(self) -> None:
        executor = make_executor(expert_count=3).eval()
        valid = torch.tensor(
            [
                [True, True, False, True],
                [False, True, True, True],
            ]
        )
        values = torch.randn(2, 4, 3)
        positions = torch.tensor(
            [
                [0, 1, -1, 4],
                [-1, 8, 9, 12],
            ]
        )
        components = executor.forward_components(
            values,
            query_valid_mask=valid,
            key_valid_mask=valid,
            logical_positions=positions,
            key_logical_positions=positions,
        )
        sums = components.router_probabilities.sum(dim=-1)

        torch.testing.assert_close(
            sums[components.positive_lag_mask],
            torch.ones_like(sums[components.positive_lag_mask]),
        )
        torch.testing.assert_close(
            sums[~components.positive_lag_mask],
            torch.zeros_like(sums[~components.positive_lag_mask]),
            rtol=0.0,
            atol=0.0,
        )

    def test_router_can_distinguish_lags_with_identical_coordinates(
        self,
    ) -> None:
        executor = make_executor(
            input_modes=1,
            output_modes=1,
            expert_count=2,
            expert_rank=1,
            router_width=1,
            same_position_skip=False,
            max_positive_lag=2,
        )
        with torch.no_grad():
            executor.router_query_weight.zero_()
            executor.router_key_weight.zero_()
            executor.router_lag_weight.fill_(1.0)
            # Expert zero wins below tanh(log(1 + lag)) ~= 0.7 and loses
            # above it.  Token coordinates contribute nothing to this choice.
            executor.router_output_weight.copy_(
                torch.tensor([[-10.0, 0.0]])
            )
            executor.router_bias.copy_(torch.tensor([7.0, 0.0]))

        values = torch.ones(1, 3, 1)
        components = executor.forward_components(
            values,
            logical_positions=torch.tensor([[0, 1, 2]]),
        )
        lag_two_expert_zero = components.router_probabilities[
            0,
            2,
            0,
            0,
        ]
        lag_one_expert_zero = components.router_probabilities[
            0,
            2,
            1,
            0,
        ]

        self.assertLess(lag_two_expert_zero.item(), 0.35)
        self.assertGreater(lag_one_expert_zero.item(), 0.65)
        self.assertGreater(
            (lag_one_expert_zero - lag_two_expert_zero).item(),
            0.35,
        )

    def test_parameter_and_logical_mac_accounting(self) -> None:
        executor = make_executor(
            input_modes=2,
            output_modes=3,
            expert_count=2,
            expert_rank=4,
            router_width=5,
            same_position_skip=False,
        )
        accounting = executor.execution_accounting(3)

        self.assertEqual(accounting.valid_query_tokens, 3)
        self.assertEqual(accounting.valid_key_tokens, 3)
        self.assertEqual(accounting.active_positive_lag_queries, 2)
        self.assertEqual(accounting.active_positive_lag_keys, 2)
        self.assertEqual(accounting.positive_lag_edges, 3)
        self.assertEqual(accounting.same_position_parameter_count, 9)
        self.assertEqual(accounting.expert_parameter_count, 40)
        self.assertEqual(accounting.router_parameter_count, 37)
        self.assertEqual(accounting.total_parameter_count, 86)
        self.assertEqual(
            accounting.total_parameter_count,
            sum(parameter.numel() for parameter in executor.parameters()),
        )
        self.assertEqual(accounting.same_position_mac_count, 18)
        self.assertEqual(accounting.expert_input_mac_count, 32)
        self.assertEqual(accounting.router_projection_mac_count, 40)
        self.assertEqual(accounting.router_lag_mac_count, 15)
        self.assertEqual(accounting.router_edge_mac_count, 30)
        self.assertEqual(accounting.expert_mixture_mac_count, 24)
        self.assertEqual(accounting.expert_output_mac_count, 48)
        self.assertEqual(accounting.total_mac_count, 207)
        self.assertEqual(accounting.dense_affine_reference_mac_count, 36)
        self.assertAlmostEqual(
            accounting.mac_to_dense_affine_ratio,
            207 / 36,
        )

        valid = torch.tensor([[False, True, True]])
        padded = executor.execution_accounting(
            3,
            query_valid_mask=valid,
            key_valid_mask=valid,
        )
        self.assertEqual(padded.valid_query_tokens, 2)
        self.assertEqual(padded.positive_lag_edges, 1)
        self.assertEqual(padded.active_positive_lag_queries, 1)
        self.assertEqual(padded.active_positive_lag_keys, 1)

    def test_strict_weights_only_artifact_round_trip_and_tamper_rejection(
        self,
    ) -> None:
        executor = make_executor(dtype=torch.float64).eval()
        values = torch.randn(2, 4, 3, dtype=torch.float64)
        expected = executor(values)
        artifact = executor.artifact_state_dict()
        state_keys = tuple(artifact["model_state_dict"])
        self.assertEqual(state_keys, tuple(sorted(state_keys)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gated.pt"
            torch.save(artifact, path)
            restored = (
                ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                    torch.load(path, weights_only=True)
                )
            ).eval()
        torch.testing.assert_close(restored(values), expected)
        self.assertEqual(restored.config, executor.config)

        # Artifact tensors are detached clones.
        artifact["model_state_dict"]["same_position_bias"][0] += 1.0
        torch.testing.assert_close(executor(values), expected)

        unknown = executor.artifact_state_dict()
        unknown["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "artifact fields"):
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                unknown
            )

        bad_config = executor.artifact_state_dict()
        bad_config["config"]["expert_count"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                bad_config
            )

        missing_weight = executor.artifact_state_dict()
        del missing_weight["model_state_dict"]["router_bias"]
        with self.assertRaisesRegex(ValueError, "model state fields"):
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                missing_weight
            )

        bad_shape = executor.artifact_state_dict()
        bad_shape["model_state_dict"]["router_bias"] = torch.zeros(
            99,
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                bad_shape
            )

        nonfinite = copy.deepcopy(executor.artifact_state_dict())
        nonfinite["model_state_dict"]["router_bias"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                nonfinite
            )

    def test_input_and_sequence_validation_is_strict(self) -> None:
        executor = make_executor()
        values = torch.randn(1, 3, 3)
        with self.assertRaisesRegex(ValueError, "finite"):
            executor(
                torch.tensor(
                    [[[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0]]]
                )
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            executor(
                values,
                logical_positions=torch.tensor([[0, 2, 1]]),
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            executor(
                values,
                query_valid_mask=torch.ones(1, 3),
            )

    def test_accounting_enforces_execution_position_invariants(self) -> None:
        executor = make_executor()
        all_valid = torch.ones(1, 3, dtype=torch.bool)

        with self.assertRaisesRegex(
            ValueError,
            "valid query logical positions cannot be negative",
        ):
            executor.execution_accounting(
                3,
                query_valid_mask=all_valid,
                key_valid_mask=all_valid,
                logical_positions=torch.tensor([[0, -1, 2]]),
                key_logical_positions=torch.tensor([[0, 1, 2]]),
            )
        with self.assertRaisesRegex(
            ValueError,
            "valid key logical positions cannot be negative",
        ):
            executor.execution_accounting(
                3,
                query_valid_mask=all_valid,
                key_valid_mask=all_valid,
                logical_positions=torch.tensor([[0, 1, 2]]),
                key_logical_positions=torch.tensor([[0, -1, 2]]),
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            executor.execution_accounting(
                3,
                query_valid_mask=all_valid,
                key_valid_mask=all_valid,
                logical_positions=torch.tensor([[0, 1, 2]]),
                key_logical_positions=torch.tensor([[0, 2, 1]]),
            )


if __name__ == "__main__":
    unittest.main()
