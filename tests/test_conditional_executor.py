import unittest

import torch
from torch import Tensor, nn

from fisher_graph.activations import ActivationTrace
from fisher_graph.conditional_executor import (
    ConditionalModalProjectionOracleExecutor,
    HardRoutedFisherProjection,
    HardRoutedSpecialistBank,
)
from fisher_graph.conditional_routing import (
    ConditionalModalRoutingPlan,
    ConditionalModeTable,
    PointwiseCausalRouter,
)
from fisher_graph.layers import LayerExecutor
from fisher_graph.modes import FisherModeBasis


def routing_plan() -> ConditionalModalRoutingPlan:
    table = ConditionalModeTable.from_masks(
        torch.tensor(
            [
                [True, False, True],
                [False, True, False],
                [False, False, False],
            ]
        )
    )
    router = PointwiseCausalRouter(
        feature_mean=torch.zeros(3, dtype=torch.float64),
        feature_scale=torch.ones(3, dtype=torch.float64),
        weight=torch.eye(3, dtype=torch.float64),
        bias=torch.zeros(3, dtype=torch.float64),
        ridge=1e-3,
        observations=12,
    )
    return ConditionalModalRoutingPlan(mode_table=table, router=router)


def fisher_basis() -> FisherModeBasis:
    # A signed permutation keeps the dense reference bit-exact while ensuring
    # modal coordinates are not in their original feature order.
    vectors = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    return FisherModeBasis(
        activation_name="layer.output",
        mean=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
        matrix=torch.eye(3, dtype=torch.float64),
        eigenvalues=torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        vectors=vectors,
        observations=12,
        sequences=3,
    )


def routed_inputs() -> Tensor:
    return torch.tensor(
        [
            [[9.0, 1.0, 0.0], [0.0, 8.0, 1.0], [0.0, 1.0, 7.0]],
            [[0.0, 1.0, 6.0], [5.0, 1.0, 0.0], [0.0, 4.0, 1.0]],
        ]
    )


class CountingSourceLayer(LayerExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.received_trace: list[ActivationTrace | None] = []

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        del attention_mask, prefix
        self.calls += 1
        self.received_trace.append(trace)
        return hidden_states * 2 + 1


class CountingSpecialist(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.calls = 0
        self.rows: list[int] = []

    def forward(self, values: Tensor) -> Tensor:
        self.calls += 1
        self.rows.append(values.shape[0])
        return values + self.offset


class ConditionalExecutorTests(unittest.TestCase):
    def test_route_grouped_projection_matches_exact_dense_reference(self) -> None:
        plan = routing_plan()
        basis = fisher_basis()
        projector = HardRoutedFisherProjection(basis, plan)
        block_inputs = routed_inputs()
        native = torch.tensor(
            [
                [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0], [17.0, 28.0, 39.0]],
                [[12.0, 23.0, 34.0], [15.0, 26.0, 37.0], [18.0, 29.0, 40.0]],
            ]
        )
        valid = torch.tensor(
            [[True, True, True], [False, True, False]]
        )

        result = projector.project_with_accounting(
            native,
            block_inputs,
            valid_mask=valid,
        )
        routes = plan.route(block_inputs)
        vectors = basis.vectors.float()
        mean = basis.mean.float()
        coordinates = (native - mean) @ vectors
        masks = plan.mode_table.masks_for(routes).to(dtype=native.dtype)
        dense = (coordinates * masks) @ vectors.T + mean
        expected = torch.where(valid.unsqueeze(-1), dense, native)

        self.assertTrue(torch.equal(result.output, expected))
        self.assertTrue(torch.equal(result.route_ids, routes))
        self.assertEqual(
            result.accounting.route_token_counts,
            (2, 1, 1),
        )
        self.assertEqual(
            result.accounting.active_modes_per_route,
            (2, 1, 0),
        )
        self.assertEqual(result.accounting.active_mode_applications, 5)
        self.assertEqual(result.accounting.dense_mode_applications, 12)
        self.assertEqual(result.accounting.average_active_modes, 1.25)
        self.assertEqual(result.accounting.active_mode_ratio, 5 / 12)
        self.assertEqual(
            result.accounting.executed_route_groups,
            (0, 1, 2),
        )
        # Rank zero executes no matrix multiply; the two positive-rank groups
        # each execute one gather/encode and one decode.
        self.assertEqual(result.accounting.modal_matmul_calls, 4)

    def test_rank_zero_maps_valid_tokens_to_center_and_padding_is_unchanged(
        self,
    ) -> None:
        projector = HardRoutedFisherProjection(
            fisher_basis(),
            routing_plan(),
        )
        inputs = torch.tensor(
            [[[0.0, 0.0, 9.0], [0.0, 0.0, 8.0]]]
        )
        native = torch.tensor(
            [[[111.0, 222.0, 333.0], [444.0, 555.0, 666.0]]]
        )
        valid = torch.tensor([[True, False]])

        result = projector.project_with_accounting(
            native,
            inputs,
            valid_mask=valid,
        )

        self.assertTrue(
            torch.equal(result.output[0, 0], fisher_basis().mean.float())
        )
        self.assertTrue(torch.equal(result.output[0, 1], native[0, 1]))
        self.assertEqual(result.accounting.route_token_counts, (0, 0, 1))
        self.assertEqual(result.accounting.modal_matmul_calls, 0)
        self.assertEqual(result.accounting.active_mode_applications, 0)

    def test_routing_and_projection_are_invariant_to_appended_future(self) -> None:
        projector = HardRoutedFisherProjection(
            fisher_basis(),
            routing_plan(),
        )
        prefix_inputs = routed_inputs()[:1, :2]
        prefix_native = torch.tensor(
            [[[12.0, 21.0, 31.0], [13.0, 22.0, 32.0]]]
        )
        future_inputs = torch.tensor(
            [[[0.0, 0.0, 1_000.0], [1_000.0, 0.0, 0.0]]]
        )
        future_native = torch.tensor(
            [[[-9_000.0, 8_000.0, 7_000.0], [6_000.0, -5_000.0, 4_000.0]]]
        )
        extended_inputs = torch.cat((prefix_inputs, future_inputs), dim=1)
        extended_native = torch.cat((prefix_native, future_native), dim=1)

        prefix = projector.project_with_accounting(
            prefix_native,
            prefix_inputs,
        )
        extended = projector.project_with_accounting(
            extended_native,
            extended_inputs,
        )

        self.assertTrue(
            torch.equal(prefix.route_ids, extended.route_ids[:, :2])
        )
        self.assertTrue(
            torch.equal(prefix.output, extended.output[:, :2])
        )

    def test_oracle_records_native_execution_routes_and_cumulative_activity(
        self,
    ) -> None:
        source = CountingSourceLayer()
        oracle = ConditionalModalProjectionOracleExecutor(
            source,
            HardRoutedFisherProjection(fisher_basis(), routing_plan()),
        )
        inputs = routed_inputs()
        valid = torch.tensor(
            [[True, True, True], [False, True, False]]
        )
        trace = ActivationTrace()

        first = oracle(
            inputs,
            attention_mask=valid,
            trace=trace,
            prefix="layer.4",
        )
        second = oracle(
            inputs,
            attention_mask=valid,
            prefix="layer.4",
        )

        native = inputs * 2 + 1
        expected = oracle.projector(native, inputs, valid_mask=valid)
        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(second, expected))
        self.assertEqual(source.calls, 2)
        self.assertEqual(source.received_trace, [None, None])
        self.assertIn("layer.4.conditional.route_ids", trace)
        self.assertIn("layer.4.conditional.native_source_output", trace)
        self.assertIn("layer.4.conditional.active_mode_mask", trace)

        status = oracle.execution_status()
        self.assertEqual(status.executor_calls, 2)
        self.assertEqual(status.native_source_block_calls, 2)
        self.assertEqual(status.routed_valid_tokens, 8)
        self.assertEqual(status.route_token_counts, (4, 2, 2))
        self.assertEqual(status.route_group_executions, (2, 2, 2))
        self.assertEqual(status.active_mode_applications, 10)
        self.assertEqual(status.dense_mode_applications, 24)
        self.assertEqual(status.logical_active_mode_ratio, 10 / 24)
        self.assertFalse(status.source_block_savings_claimed)

        oracle.reset_execution_counters()
        reset = oracle.execution_status()
        self.assertEqual(reset.executor_calls, 0)
        self.assertEqual(reset.native_source_block_calls, 0)
        self.assertIsNone(oracle.last_execution)

    def test_specialist_bank_invokes_only_selected_nonempty_branches(self) -> None:
        specialists = [
            CountingSpecialist(10.0),
            CountingSpecialist(20.0),
            CountingSpecialist(30.0),
        ]
        bank = HardRoutedSpecialistBank(routing_plan(), specialists)
        inputs = routed_inputs()[:1]
        # Route 2 is present, but its only row is padding.
        valid = torch.tensor([[True, True, False]])
        invalid_output = torch.full_like(inputs, -99.0)

        result = bank.execute_with_accounting(
            inputs,
            valid_mask=valid,
            invalid_output=invalid_output,
        )

        torch.testing.assert_close(result.output[0, 0], inputs[0, 0] + 10)
        torch.testing.assert_close(result.output[0, 1], inputs[0, 1] + 20)
        torch.testing.assert_close(
            result.output[0, 2],
            invalid_output[0, 2],
        )
        self.assertEqual([specialist.calls for specialist in specialists], [1, 1, 0])
        self.assertEqual(
            [specialist.rows for specialist in specialists],
            [[1], [1], []],
        )
        self.assertEqual(result.accounting.route_token_counts, (1, 1, 0))
        self.assertEqual(result.accounting.specialist_calls, (1, 1, 0))
        self.assertEqual(result.accounting.executed_routes, (0, 1))

        # A second call selects only route 2, proving the other branches are
        # not speculatively evaluated.
        route_two = torch.tensor([[[0.0, 0.0, 5.0]]])
        second = bank.execute_with_accounting(route_two)
        torch.testing.assert_close(second.output, route_two + 30)
        self.assertEqual([specialist.calls for specialist in specialists], [1, 1, 1])
        status = bank.execution_status()
        self.assertEqual(status.executor_calls, 2)
        self.assertEqual(status.valid_tokens, 3)
        self.assertEqual(status.route_token_counts, (1, 1, 1))
        self.assertEqual(status.specialist_calls, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
