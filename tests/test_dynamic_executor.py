import math
import unittest

import torch

from fisher_graph import ActivationTrace
from fisher_graph.adapters.base import SequenceContext, SequenceInputOrigin
from fisher_graph.dynamic_executor import (
    SharedModalProjection,
    StatefulCausalModalGraph,
    VariableLengthCausalModalExecutor,
)
from fisher_graph.compiler.capabilities import (
    MatchStatus,
    match_capabilities,
    request_from_context,
)
from fisher_graph.modes import FisherModeBasis


def identity_basis(width: int, name: str) -> FisherModeBasis:
    return FisherModeBasis(
        activation_name=name,
        mean=torch.zeros(width),
        matrix=torch.eye(width),
        eigenvalues=torch.arange(width, 0, -1, dtype=torch.float32),
        vectors=torch.eye(width),
        observations=20,
        sequences=4,
    )


def make_executor(
    *,
    width: int = 4,
    input_modes: int = 3,
    output_modes: int = 3,
    state_channels: int = 2,
    routing_width: int = 5,
    activation: str = "gelu",
    window_size: int | None = None,
) -> VariableLengthCausalModalExecutor:
    return VariableLengthCausalModalExecutor.from_bases(
        identity_basis(width, "input"),
        identity_basis(width, "output"),
        input_modes=input_modes,
        output_modes=output_modes,
        state_channels=state_channels,
        routing_width=routing_width,
        activation=activation,
        window_size=window_size,
    )


def sequence_context(
    valid: torch.Tensor,
    positions: torch.Tensor,
    *,
    query_valid: torch.Tensor | None = None,
    phase: str = "prefill",
    cache_state=None,
    cache_positions: torch.Tensor | None = None,
) -> SequenceContext:
    return SequenceContext(
        query_valid_mask=valid if query_valid is None else query_valid,
        key_valid_mask=valid,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=cache_positions,
        phase=phase,
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=cache_positions is not None,
        ),
        cache_state=cache_state,
        adapter_payload=None,
    )


class VariableLengthCausalModalExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)

    def test_parameters_are_independent_of_runtime_sequence_length(self) -> None:
        executor = make_executor().eval()
        original_shapes = {
            name: tuple(value.shape)
            for name, value in executor.state_dict().items()
        }
        parameter_count = sum(
            parameter.numel() for parameter in executor.parameters()
        )

        for length in (1, 3, 11):
            output = executor(
                torch.randn(2, length, 4),
                prefix="layer.dynamic",
            )
            self.assertEqual(output.shape, (2, length, 4))
            self.assertEqual(
                {
                    name: tuple(value.shape)
                    for name, value in executor.state_dict().items()
                },
                original_shapes,
            )
            self.assertEqual(
                sum(
                    parameter.numel()
                    for parameter in executor.parameters()
                ),
                parameter_count,
            )

        self.assertEqual(executor.sequence_spec.length_policy, "dynamic")
        self.assertIsNone(executor.sequence_spec.maximum_length)
        self.assertFalse(executor.sequence_spec.supports_decode)
        self.assertEqual(executor.sequence_spec.mask.padding_side, "sparse")

        capabilities = executor.capabilities
        self.assertEqual(capabilities.executions.values, {"prefill"})
        self.assertEqual(capabilities.qk_relations.values, {"equal"})
        self.assertEqual(
            capabilities.visibility_families.values,
            {"global_causal"},
        )
        self.assertEqual(capabilities.dtypes.values, {"float32"})
        self.assertEqual(capabilities.devices.values, {"cpu"})
        self.assertEqual(
            capabilities.layouts.values,
            {"contiguous", "strided"},
        )

        strided = torch.randn(2, 3, 8)[..., ::2]
        self.assertFalse(strided.is_contiguous())
        strided_output = executor(strided, prefix="layer.dynamic")
        contiguous_output = executor(
            strided.contiguous(),
            prefix="layer.dynamic",
        )
        torch.testing.assert_close(strided_output, contiguous_output)

    def test_extended_future_cannot_change_prefix_outputs(self) -> None:
        executor = make_executor().eval()
        prefix_values = torch.randn(2, 4, 4)
        prefix_output = executor(
            prefix_values,
            prefix="layer.dynamic",
        )
        extended = torch.cat(
            (prefix_values, torch.randn(2, 5, 4) * 1_000),
            dim=1,
        )
        extended_output = executor(
            extended,
            prefix="layer.dynamic",
        )

        torch.testing.assert_close(
            extended_output[:, :4],
            prefix_output,
            rtol=0,
            atol=0,
        )

        extended.requires_grad_()
        first_three = executor(
            extended,
            prefix="layer.dynamic",
        )[:, :3].sum()
        gradient = torch.autograd.grad(first_three, extended)[0]
        torch.testing.assert_close(
            gradient[:, 3:],
            torch.zeros_like(gradient[:, 3:]),
            rtol=0,
            atol=0,
        )

    def test_sparse_padding_and_arbitrary_offset_preserve_valid_outputs(
        self,
    ) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 3, 4)
        compact_context = sequence_context(
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[5, 7, 10]]),
        )
        compact = executor.forward_context(
            values,
            sequence=compact_context,
            prefix="layer.dynamic",
        )
        shifted = executor.forward_context(
            values,
            sequence=sequence_context(
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([[105, 107, 110]]),
            ),
            prefix="layer.dynamic",
        )
        torch.testing.assert_close(shifted, compact, rtol=0, atol=0)

        padded_values = torch.randn(1, 6, 4) * 10_000
        padded_values[:, [1, 3, 5]] = values
        valid = torch.tensor([[False, True, False, True, False, True]])
        padded_context = sequence_context(
            valid,
            torch.tensor([[999, 5, -40, 7, 10_000, 10]]),
        )
        padded = executor.forward_context(
            padded_values,
            sequence=padded_context,
            prefix="layer.dynamic",
        )

        torch.testing.assert_close(
            padded[:, [1, 3, 5]],
            compact,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            padded[:, [0, 2, 4]],
            torch.zeros(1, 3, 4),
            rtol=0,
            atol=0,
        )

    def test_mixed_left_and_right_padding_match_unpadded_calls(self) -> None:
        executor = make_executor().eval()
        first = torch.randn(1, 2, 4)
        second = torch.randn(1, 3, 4)
        expected_first = executor.forward_context(
            first,
            sequence=sequence_context(
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[4, 6]]),
            ),
            prefix="layer.dynamic",
        )
        expected_second = executor.forward_context(
            second,
            sequence=sequence_context(
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([[10, 11, 14]]),
            ),
            prefix="layer.dynamic",
        )

        padded = torch.randn(2, 5, 4) * 100_000
        padded[0, :2] = first[0]
        padded[1, 2:] = second[0]
        valid = torch.tensor(
            [
                [True, True, False, False, False],
                [False, False, True, True, True],
            ]
        )
        context = sequence_context(
            valid,
            torch.tensor(
                [
                    [4, 6, -1, -1, -1],
                    [-1, -1, 10, 11, 14],
                ]
            ),
        )
        actual = executor.forward_context(
            padded,
            sequence=context,
            prefix="layer.dynamic",
        )

        torch.testing.assert_close(
            actual[0, :2],
            expected_first[0],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            actual[1, 2:],
            expected_second[0],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            actual[~valid],
            torch.zeros_like(actual[~valid]),
            rtol=0,
            atol=0,
        )

    def test_logical_position_gaps_control_relative_decay(self) -> None:
        projection = SharedModalProjection(
            activation_name="tap",
            mean=torch.zeros(1),
            vectors=torch.ones(1, 1),
        )
        graph = StatefulCausalModalGraph(
            input_modes=1,
            output_modes=1,
            state_channels=1,
            routing_width=1,
            activation="identity",
        )
        with torch.no_grad():
            graph.state_input_weight.fill_(1)
            # softplus(0) == log(2), so one position step decays by 1/2.
            graph.raw_decay_rate.zero_()
            graph.hidden_weight.fill_(1)
            graph.hidden_bias.zero_()
            graph.output_weight.fill_(1)
            graph.output_bias.zero_()
        executor = VariableLengthCausalModalExecutor(
            projection,
            graph,
            projection,
        ).eval()
        values = torch.ones(1, 2, 1)
        valid = torch.ones(1, 2, dtype=torch.bool)

        adjacent = executor.forward_context(
            values,
            sequence=sequence_context(
                valid,
                torch.tensor([[40, 41]]),
            ),
            prefix="layer.dynamic",
        )
        gap_three = executor.forward_context(
            values,
            sequence=sequence_context(
                valid,
                torch.tensor([[40, 43]]),
            ),
            prefix="layer.dynamic",
        )

        torch.testing.assert_close(
            adjacent,
            torch.tensor([[[1.0], [1.5]]]),
        )
        torch.testing.assert_close(
            gap_three,
            torch.tensor([[[1.0], [1.125]]]),
        )
        self.assertAlmostEqual(graph.decay_rate.item(), math.log(2))

    def test_sliding_window_excludes_exact_outside_boundary(self) -> None:
        projection = SharedModalProjection(
            activation_name="tap",
            mean=torch.zeros(1),
            vectors=torch.ones(1, 1),
        )
        graph = StatefulCausalModalGraph(
            input_modes=1,
            output_modes=1,
            state_channels=1,
            routing_width=1,
            activation="identity",
            window_size=2,
        )
        with torch.no_grad():
            graph.state_input_weight.fill_(1)
            graph.raw_decay_rate.zero_()
            graph.hidden_weight.fill_(1)
            graph.hidden_bias.zero_()
            graph.output_weight.fill_(1)
            graph.output_bias.zero_()
        executor = VariableLengthCausalModalExecutor(
            projection,
            graph,
            projection,
        ).eval()
        valid = torch.ones(1, 3, dtype=torch.bool)
        context = sequence_context(
            valid,
            torch.tensor([[0, 1, 2]]),
        )
        values = torch.tensor([[[1.0], [2.0], [3.0]]])
        baseline = executor.forward_context(
            values,
            sequence=context,
            prefix="layer.dynamic",
        )
        changed = values.clone()
        changed[:, 0] += 1_000
        edited = executor.forward_context(
            changed,
            sequence=context,
            prefix="layer.dynamic",
        )

        # At logical position 2, position 0 is exactly outside a size-2 window.
        torch.testing.assert_close(
            edited[:, 2],
            baseline[:, 2],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(edited[:, 1], baseline[:, 1]))
        self.assertEqual(
            executor.capabilities.visibility_families.values,
            {"sliding_causal"},
        )

    def test_query_and_key_masks_have_distinct_context_semantics(self) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 4, 4)
        keys = torch.ones(1, 4, dtype=torch.bool)
        queries = torch.tensor([[False, False, True, True]])
        context = sequence_context(
            keys,
            torch.tensor([[20, 21, 22, 23]]),
            query_valid=queries,
        )

        output = executor.forward_context(
            values,
            sequence=context,
            prefix="layer.dynamic",
        )
        all_queries = executor.forward_context(
            values,
            sequence=sequence_context(
                keys,
                torch.tensor([[20, 21, 22, 23]]),
            ),
            prefix="layer.dynamic",
        )

        torch.testing.assert_close(
            output[:, :2],
            torch.zeros_like(output[:, :2]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output[:, 2:],
            all_queries[:, 2:],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_custom_masks_cannot_expose_semantically_future_keys(
        self,
    ) -> None:
        graph = StatefulCausalModalGraph(
            input_modes=1,
            output_modes=1,
            state_channels=1,
            routing_width=1,
            activation="identity",
        )
        with torch.no_grad():
            graph.state_input_weight.fill_(1)
            graph.raw_decay_rate.zero_()
            graph.hidden_weight.fill_(1)
            graph.hidden_bias.zero_()
            graph.output_weight.fill_(1)
            graph.output_bias.zero_()
        positions = torch.tensor([[10, 5]])
        output = graph(
            torch.ones(1, 2, 1),
            query_valid_mask=torch.tensor([[False, True]]),
            key_valid_mask=torch.tensor([[True, False]]),
            logical_positions=positions,
            key_logical_positions=positions,
        )

        # The valid query is at 5; the only valid key is at 10 and is future.
        torch.testing.assert_close(
            output,
            torch.zeros_like(output),
            rtol=0,
            atol=0,
        )

    def test_distinct_query_and_key_positions_use_semantic_causality(
        self,
    ) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 3, 4)
        valid = torch.ones(1, 3, dtype=torch.bool)
        context = SequenceContext(
            query_valid_mask=valid,
            key_valid_mask=valid,
            logical_positions=torch.tensor([[10, 20, 100]]),
            key_logical_positions=torch.tensor([[10, 30, 50]]),
            cache_positions=None,
            phase="prefill",
            input_origin=SequenceInputOrigin(
                attention_mask_supplied=True,
                position_ids_supplied=True,
                cache_positions_supplied=False,
            ),
            cache_state=None,
            adapter_payload=None,
        )
        baseline = executor.forward_context(
            values,
            sequence=context,
            prefix="layer.dynamic",
        )
        changed = values.clone()
        changed[:, 1:] += 1_000
        edited = executor.forward_context(
            changed,
            sequence=context,
            prefix="layer.dynamic",
        )

        # Queries at 10 and 20 cannot see keys at 30 and 50.
        torch.testing.assert_close(
            edited[:, :2],
            baseline[:, :2],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(edited[:, 2], baseline[:, 2]))

        request = request_from_context(
            context,
            values,
            mask_representation="boolean_valid",
            visibility_family="global_causal",
            cache_kind="none",
        )
        match = match_capabilities(executor.capabilities, request)
        self.assertIs(match.status, MatchStatus.MATCH)

    def test_layer_forward_accepts_binary_mask_and_context_consistency(
        self,
    ) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 4, 4)
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.int64)
        output = executor(
            values,
            attention_mask=mask,
            prefix="layer.dynamic",
        )
        torch.testing.assert_close(
            output[:, 2:],
            torch.zeros_like(output[:, 2:]),
            rtol=0,
            atol=0,
        )

        context = sequence_context(
            mask.bool(),
            torch.arange(4).unsqueeze(0),
        )
        matching = executor(
            values,
            attention_mask=mask,
            sequence_context=context,
            prefix="layer.dynamic",
        )
        torch.testing.assert_close(matching, output)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            executor(
                values,
                attention_mask=torch.ones_like(mask),
                sequence_context=context,
                prefix="layer.dynamic",
            )

    def test_trace_exposes_intervenable_dynamic_graph_stages(self) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 4, 4)
        baseline_trace = ActivationTrace()
        baseline = executor(
            values,
            trace=baseline_trace,
            prefix="layer.dynamic",
        )
        self.assertEqual(
            baseline_trace.names,
            (
                "layer.dynamic.input",
                "layer.dynamic.modal.input",
                "layer.dynamic.modal.causal_state",
                "layer.dynamic.modal.hidden",
                "layer.dynamic.modal.output",
                "layer.dynamic.output",
            ),
        )

        edited_trace = ActivationTrace(
            interventions={
                "layer.dynamic.modal.hidden": torch.zeros_like,
            }
        )
        edited = executor(
            values,
            trace=edited_trace,
            prefix="layer.dynamic",
        )
        self.assertFalse(torch.equal(edited, baseline))
        torch.testing.assert_close(
            edited_trace["layer.dynamic.modal.hidden"],
            torch.zeros_like(
                edited_trace["layer.dynamic.modal.hidden"]
            ),
        )
        edited_trace.assert_all_interventions_applied()

    def test_same_parameters_train_across_mixed_lengths(self) -> None:
        executor = make_executor().train()
        optimizer = torch.optim.Adam(executor.parameters(), lr=1e-3)
        original_parameter_ids = tuple(
            id(parameter) for parameter in executor.parameters()
        )

        optimizer.zero_grad()
        loss = torch.zeros(())
        for length in (2, 7, 3):
            inputs = torch.randn(2, length, 4)
            targets = torch.randn(2, length, 4)
            output = executor(inputs, prefix="layer.dynamic")
            loss = loss + (output - targets).square().mean()
        loss.backward()

        for name, parameter in executor.graph.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertEqual(
            tuple(id(parameter) for parameter in executor.parameters()),
            original_parameter_ids,
        )

    def test_nonmonotonic_positions_and_decode_fail_explicitly(self) -> None:
        executor = make_executor().eval()
        values = torch.randn(1, 3, 4)
        valid = torch.ones(1, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            executor.forward_context(
                values,
                sequence=sequence_context(
                    valid,
                    torch.tensor([[3, 2, 4]]),
                ),
                prefix="layer.dynamic",
            )

        with self.assertRaisesRegex(ValueError, "cached decode"):
            executor.forward_context(
                values,
                sequence=sequence_context(
                    valid,
                    torch.tensor([[3, 4, 5]]),
                    phase="decode",
                ),
                prefix="layer.dynamic",
            )
        with self.assertRaisesRegex(ValueError, "cache state"):
            executor.forward_context(
                values,
                sequence=sequence_context(
                    valid,
                    torch.tensor([[3, 4, 5]]),
                    cache_state={"unsupported": True},
                ),
                prefix="layer.dynamic",
            )
        with self.assertRaisesRegex(ValueError, "cache positions"):
            executor.forward_context(
                values,
                sequence=sequence_context(
                    valid,
                    torch.tensor([[3, 4, 5]]),
                    cache_positions=torch.tensor([0, 1, 2]),
                ),
                prefix="layer.dynamic",
            )


if __name__ == "__main__":
    unittest.main()
