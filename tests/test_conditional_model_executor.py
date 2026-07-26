import copy
import unittest
from unittest.mock import patch

import torch

from fisher_graph.adapters.base import (
    SequenceContext,
    SequenceInputOrigin,
)
from fisher_graph.conditional_model_executor import (
    ConditionalCausalModalBlockExecutor,
    RouteSource,
)
from fisher_graph.conditional_routing import (
    ConditionalModalRoutingPlan,
    ConditionalModeTable,
    PointwiseCausalRouter,
)
from fisher_graph.compiler.manifest import (
    BackendSpec,
    CompiledSegment,
    SegmentProvenance,
    SegmentValidation,
    SequenceSpec as ManifestSequenceSpec,
)
from fisher_graph.dynamic_executor import StatefulCausalModalGraph
from fisher_graph.layers import LayerExecutor
from fisher_graph.linear_codec import (
    LinearActivationCodec,
    build_generalized_fisher_codec,
    build_native_fisher_codec,
)


def _codec(
    width: int,
    *,
    mean: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> LinearActivationCodec:
    if mean is None:
        mean = torch.zeros(width, dtype=torch.float64)
    if vectors is None:
        vectors = torch.eye(width, dtype=torch.float64)
    return build_native_fisher_codec(
        activation_name="block.output",
        mean=mean,
        covariance=torch.eye(width, dtype=torch.float64),
        fisher_eigenvalues=torch.arange(
            width,
            0,
            -1,
            dtype=torch.float64,
        ),
        fisher_vectors=vectors,
    )


def _routing_plan(
    masks: torch.Tensor,
    *,
    feature_width: int,
) -> ConditionalModalRoutingPlan:
    routes = masks.shape[0]
    if routes == 1:
        weight = torch.zeros(
            feature_width,
            1,
            dtype=torch.float64,
        )
    else:
        weight = torch.zeros(
            feature_width,
            routes,
            dtype=torch.float64,
        )
        for route in range(min(feature_width, routes)):
            weight[route, route] = 1.0
    router = PointwiseCausalRouter(
        feature_mean=torch.zeros(feature_width, dtype=torch.float64),
        feature_scale=torch.ones(feature_width, dtype=torch.float64),
        weight=weight,
        bias=torch.zeros(routes, dtype=torch.float64),
        ridge=1e-3,
        observations=32,
    )
    return ConditionalModalRoutingPlan(
        mode_table=ConditionalModeTable.from_masks(masks),
        router=router,
    )


def _identity_trunk(
    width: int,
    codec: LinearActivationCodec,
    *,
    window_size: int | None = 1,
    dtype: torch.dtype = torch.float32,
) -> StatefulCausalModalGraph:
    graph = StatefulCausalModalGraph(
        input_modes=width,
        output_modes=width,
        state_channels=1,
        routing_width=width,
        activation="identity",
        window_size=window_size,
    ).to(dtype=dtype)
    with torch.no_grad():
        graph.state_input_weight[0].copy_(torch.eye(width, dtype=dtype))
        # softplus(0) = log(2), useful for the padding/causality test.
        graph.raw_decay_rate.zero_()
        graph.hidden_weight.copy_(torch.eye(width, dtype=dtype))
        graph.hidden_bias.zero_()
        graph.output_weight.copy_(codec.encoder.to(dtype=dtype))
        graph.output_bias.copy_(
            -(codec.mean @ codec.encoder).to(dtype=dtype)
        )
    return graph


def _sequence(
    valid: torch.Tensor,
    positions: torch.Tensor,
) -> SequenceContext:
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid,
        logical_positions=positions,
        key_logical_positions=positions,
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


def _executor(
    masks: torch.Tensor,
    *,
    codec: LinearActivationCodec | None = None,
    window_size: int | None = 1,
    dtype: torch.dtype = torch.float32,
    route_source: RouteSource = "block_input",
) -> ConditionalCausalModalBlockExecutor:
    width = masks.shape[1]
    resolved_codec = _codec(width) if codec is None else codec
    return ConditionalCausalModalBlockExecutor(
        graph=_identity_trunk(
            width,
            resolved_codec,
            window_size=window_size,
            dtype=dtype,
        ),
        routing_plan=_routing_plan(masks, feature_width=width),
        output_codec=resolved_codec,
        input_activation_name="block.input",
        route_source=route_source,
    )


def _compiled_segment(
    *,
    input_activation: str = "block.input",
    output_activation: str = "block.output",
) -> CompiledSegment:
    return CompiledSegment(
        id="compiled.block",
        order=0,
        source_layers=("layer.0",),
        input_activation=input_activation,
        output_activation=output_activation,
        backend=BackendSpec(id="unit.conditional", abi_version=1),
        sequence=ManifestSequenceSpec(
            policy="dynamic",
            minimum_length=1,
            maximum_length=None,
            causal=True,
            attention_mask="optional",
            padding="either",
            position_ids="optional",
            cache="none",
        ),
        fast_resources=("executor",),
        instrumentation_resources=(),
        instrumentation_policy="none",
        fallback_policy="disabled",
        provenance=SegmentProvenance(
            source_model_state_sha256="0" * 64,
            source_model_config_sha256="1" * 64,
            dependency_resources=("executor",),
            compile_config_sha256=None,
        ),
        validation=SegmentValidation(
            status="passed",
            validator_id="unit.validator",
            validator_version=1,
            report_resource="report",
        ),
    )


class ConditionalModelExecutorTests(unittest.TestCase):
    def test_block_input_remains_the_default_route_source(self) -> None:
        implicit = _executor(
            torch.eye(2, dtype=torch.bool),
            window_size=None,
        ).eval()
        explicit = _executor(
            torch.eye(2, dtype=torch.bool),
            window_size=None,
            route_source="block_input",
        ).eval()
        values = torch.tensor(
            [[[0.0, 4.0], [1.0, 0.0]]]
        )

        implicit_result = implicit.execute_context_with_accounting(
            values,
            sequence=_sequence(
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[0, 1]]),
            ),
            prefix="compiled.block",
        )
        explicit_result = explicit.execute_context_with_accounting(
            values,
            sequence=_sequence(
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[0, 1]]),
            ),
            prefix="compiled.block",
        )

        self.assertEqual(implicit.route_source, "block_input")
        self.assertTrue(
            torch.equal(
                implicit_result.route_ids,
                torch.tensor([[1, 0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                implicit_result.route_ids,
                explicit_result.route_ids,
            )
        )
        torch.testing.assert_close(
            implicit_result.output,
            explicit_result.output,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            implicit.execution_fingerprint(),
            explicit.execution_fingerprint(),
        )
        self.assertEqual(
            implicit.artifact_state_dict()["route_source"],
            "block_input",
        )

    def test_causal_hidden_route_can_depend_on_the_prefix(self) -> None:
        executor = _executor(
            torch.eye(2, dtype=torch.bool),
            window_size=None,
            route_source="causal_hidden",
        ).eval()
        values = torch.tensor(
            [
                [[0.0, 4.0], [1.0, 0.0]],
                [[4.0, 0.0], [1.0, 0.0]],
            ]
        )
        sequence = _sequence(
            torch.ones(2, 2, dtype=torch.bool),
            torch.tensor([[0, 1], [0, 1]]),
        )

        result = executor.execute_context_with_accounting(
            values,
            sequence=sequence,
            prefix="compiled.block",
        )
        block_input_routes = executor.routing_plan.route(values)

        self.assertTrue(torch.equal(values[0, 1], values[1, 1]))
        self.assertTrue(
            torch.equal(block_input_routes[:, 1], torch.tensor([0, 0]))
        )
        self.assertTrue(
            torch.equal(result.route_ids[:, 1], torch.tensor([1, 0]))
        )

    def test_route_source_enforces_the_selected_feature_width(self) -> None:
        codec = _codec(2)
        graph = StatefulCausalModalGraph(
            input_modes=2,
            output_modes=2,
            state_channels=1,
            routing_width=3,
            activation="identity",
        )
        three_feature_plan = _routing_plan(
            torch.eye(2, dtype=torch.bool),
            feature_width=3,
        )
        two_feature_plan = _routing_plan(
            torch.eye(2, dtype=torch.bool),
            feature_width=2,
        )

        causal = ConditionalCausalModalBlockExecutor(
            graph=graph,
            routing_plan=three_feature_plan,
            output_codec=codec,
            input_activation_name="block.input",
            route_source="causal_hidden",
        )
        self.assertEqual(causal.route_source, "causal_hidden")
        with self.assertRaisesRegex(ValueError, "selected route source"):
            ConditionalCausalModalBlockExecutor(
                graph=graph,
                routing_plan=three_feature_plan,
                output_codec=codec,
                input_activation_name="block.input",
            )
        with self.assertRaisesRegex(ValueError, "selected route source"):
            ConditionalCausalModalBlockExecutor(
                graph=graph,
                routing_plan=two_feature_plan,
                output_codec=codec,
                input_activation_name="block.input",
                route_source="causal_hidden",
            )
        with self.assertRaisesRegex(ValueError, "route_source"):
            ConditionalCausalModalBlockExecutor(
                graph=graph,
                routing_plan=two_feature_plan,
                output_codec=codec,
                input_activation_name="block.input",
                route_source="future_state",  # type: ignore[arg-type]
            )

    def test_sequence_capabilities_match_the_shared_dynamic_backend(self) -> None:
        global_executor = _executor(
            torch.ones(1, 2, dtype=torch.bool),
            window_size=None,
        )
        sliding_executor = _executor(
            torch.ones(1, 2, dtype=torch.bool),
            window_size=2,
        )

        capabilities = global_executor.capabilities
        self.assertTrue(capabilities.length.contains(1_000))
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
        self.assertEqual(
            sliding_executor.capabilities.visibility_families.values,
            {"sliding_causal"},
        )

    def test_full_width_codec_identity_matches_configured_dense_head(
        self,
    ) -> None:
        vectors = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        codec = _codec(
            3,
            mean=torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64),
            vectors=vectors,
        )
        executor = _executor(
            torch.ones(1, 3, dtype=torch.bool),
            codec=codec,
        ).eval()
        values = torch.tensor(
            [
                [
                    [0.25, -0.5, 1.0],
                    [2.0, 0.5, -1.5],
                    [-0.25, 1.5, 3.0],
                ]
            ]
        )

        # The conditional executor must select output_weight columns itself;
        # invoking the graph's dense coordinate head is an implementation bug.
        with patch.object(
            executor.graph,
            "compute_output",
            side_effect=AssertionError("dense coordinate head was executed"),
        ):
            output = executor(values, prefix="compiled.block")

        # The configured head emits (values - mean) @ encoder.  Full-width
        # codec decoding therefore reconstructs values as the block delta.
        torch.testing.assert_close(output, values * 2, rtol=0.0, atol=0.0)
        accounting = executor.last_execution
        assert accounting is not None
        self.assertEqual(accounting.executed_compute_routes, (0,))
        self.assertEqual(accounting.output_head_column_applications, 9)
        self.assertEqual(accounting.output_head_macs, 27)
        self.assertEqual(accounting.decoder_macs, 27)
        self.assertEqual(accounting.ideal_tail_mac_reduction_fraction, 0.0)

    def test_rank_zero_is_bit_exact_residual_bypass(self) -> None:
        codec = _codec(
            3,
            mean=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),
        )
        executor = _executor(
            torch.zeros(1, 3, dtype=torch.bool),
            codec=codec,
        ).eval()
        with torch.no_grad():
            executor.graph.output_weight.fill_(float("nan"))
            executor.graph.output_bias.fill_(float("nan"))
            executor.output_decoder.fill_(float("nan"))
        values = torch.randn(2, 4, 3)

        output = executor(values, prefix="compiled.block")

        self.assertTrue(torch.equal(output, values))
        accounting = executor.last_execution
        assert accounting is not None
        self.assertEqual(accounting.populated_routes, (0,))
        self.assertEqual(accounting.executed_compute_routes, ())
        self.assertEqual(accounting.route_group_calls, (0,))
        self.assertEqual(accounting.output_head_matmul_calls, 0)
        self.assertEqual(accounting.decoder_matmul_calls, 0)
        self.assertEqual(accounting.output_head_macs, 0)
        self.assertEqual(accounting.decoder_macs, 0)

    def test_empty_route_branches_and_unselected_columns_are_skipped(
        self,
    ) -> None:
        executor = _executor(torch.eye(3, dtype=torch.bool)).eval()
        # Route 1 is empty below.  NaN sentinels prove its head and decoder
        # columns cannot leak through a dense full-coordinate computation.
        with torch.no_grad():
            executor.graph.output_weight[:, 1].fill_(float("nan"))
            executor.graph.output_bias[1] = float("nan")
            executor.output_decoder[:, 1].fill_(float("nan"))
        values = torch.tensor(
            [[[4.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 1.0, 3.0]]]
        )
        sequence = _sequence(
            torch.ones(1, 3, dtype=torch.bool),
            torch.arange(3).unsqueeze(0),
        )

        first = executor.execute_context_with_accounting(
            values,
            sequence=sequence,
            prefix="compiled.block",
        )
        second = executor.execute_context_with_accounting(
            values,
            sequence=sequence,
            prefix="compiled.block",
        )

        self.assertTrue(torch.isfinite(first.output).all())
        self.assertTrue(torch.equal(first.route_ids, torch.tensor([[0, 0, 2]])))
        self.assertEqual(first.accounting.route_token_counts, (2, 0, 1))
        self.assertEqual(first.accounting.populated_routes, (0, 2))
        self.assertEqual(
            first.accounting.head_column_indices_per_route,
            ((0,), (1,), (2,)),
        )
        self.assertEqual(
            first.accounting.executed_compute_routes,
            (0, 2),
        )
        self.assertEqual(first.accounting.route_group_calls, (1, 0, 1))
        self.assertEqual(first.accounting.output_head_matmul_calls, 2)
        self.assertEqual(first.accounting.decoder_matmul_calls, 2)
        self.assertEqual(first.accounting.output_head_column_applications, 3)
        self.assertEqual(first.accounting.output_head_macs, 9)
        self.assertEqual(first.accounting.decoder_macs, 9)

        status = executor.execution_status()
        self.assertEqual(status.executor_calls, 2)
        self.assertEqual(status.valid_tokens, 6)
        self.assertEqual(status.route_token_counts, (4, 0, 2))
        self.assertEqual(status.route_group_executions, (2, 0, 2))
        self.assertEqual(
            status.head_column_indices_per_route,
            ((0,), (1,), (2,)),
        )
        self.assertEqual(status.output_head_matmul_calls, 4)
        self.assertEqual(status.decoder_matmul_calls, 4)
        self.assertEqual(status.output_head_macs, 18)
        self.assertEqual(status.decoder_macs, 18)
        self.assertTrue(status.executor_local_source_free)
        self.assertTrue(torch.equal(second.output, first.output))

    def test_sparse_padding_preserves_valid_causal_outputs_and_bypasses_rows(
        self,
    ) -> None:
        executor = _executor(
            torch.ones(1, 3, dtype=torch.bool),
            window_size=None,
        ).eval()
        compact_values = torch.tensor(
            [[[1.0, 2.0, 3.0], [2.0, -1.0, 1.0], [3.0, 0.5, -2.0]]]
        )
        compact = executor.forward_context(
            compact_values,
            sequence=_sequence(
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([[5, 7, 10]]),
            ),
            prefix="compiled.block",
        )

        padded = torch.randn(1, 6, 3) * 100_000
        padded[:, [1, 3, 5]] = compact_values
        original_padding = padded[:, [0, 2, 4]].clone()
        valid = torch.tensor(
            [[False, True, False, True, False, True]]
        )
        padded_output = executor.forward_context(
            padded,
            sequence=_sequence(
                valid,
                torch.tensor([[999, 5, -40, 7, 10_000, 10]]),
            ),
            prefix="compiled.block",
        )

        torch.testing.assert_close(
            padded_output[:, [1, 3, 5]],
            compact,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            torch.equal(
                padded_output[:, [0, 2, 4]],
                original_padding,
            )
        )
        accounting = executor.last_execution
        assert accounting is not None
        self.assertEqual(accounting.total_tokens, 6)
        self.assertEqual(accounting.valid_tokens, 3)
        self.assertEqual(accounting.invalid_tokens, 3)

    def test_appended_future_cannot_change_prefix_routes_or_outputs(
        self,
    ) -> None:
        executor = _executor(
            torch.eye(3, dtype=torch.bool),
            window_size=None,
        ).eval()
        prefix_values = torch.tensor(
            [[[4.0, 1.0, 0.0], [0.0, 5.0, 1.0], [0.0, 1.0, 6.0]]]
        )
        prefix = executor.execute_context_with_accounting(
            prefix_values,
            sequence=_sequence(
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([[10, 11, 14]]),
            ),
            prefix="compiled.block",
        )
        future = torch.tensor(
            [[[1_000.0, 0.0, 0.0], [0.0, 0.0, 2_000.0]]]
        )
        extended_values = torch.cat((prefix_values, future), dim=1)
        extended_values.requires_grad_()
        extended = executor.execute_context_with_accounting(
            extended_values,
            sequence=_sequence(
                torch.ones(1, 5, dtype=torch.bool),
                torch.tensor([[10, 11, 14, 20, 25]]),
            ),
            prefix="compiled.block",
        )

        self.assertTrue(
            torch.equal(prefix.route_ids, extended.route_ids[:, :3])
        )
        torch.testing.assert_close(
            prefix.output,
            extended.output[:, :3],
            rtol=0.0,
            atol=0.0,
        )
        gradient = torch.autograd.grad(
            extended.output[:, :3].sum(),
            extended_values,
        )[0]
        torch.testing.assert_close(
            gradient[:, 3:],
            torch.zeros_like(gradient[:, 3:]),
            rtol=0.0,
            atol=0.0,
        )

    def test_causal_hidden_routes_and_outputs_are_future_invariant(
        self,
    ) -> None:
        executor = _executor(
            torch.eye(3, dtype=torch.bool),
            window_size=None,
            route_source="causal_hidden",
        ).eval()
        prefix_values = torch.tensor(
            [[[0.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 3.0]]]
        )
        prefix = executor.execute_context_with_accounting(
            prefix_values,
            sequence=_sequence(
                torch.ones(1, 3, dtype=torch.bool),
                torch.tensor([[10, 11, 14]]),
            ),
            prefix="compiled.block",
        )
        extended_values = torch.cat(
            (
                prefix_values,
                torch.tensor(
                    [[[10_000.0, 0.0, 0.0], [0.0, 0.0, 20_000.0]]]
                ),
            ),
            dim=1,
        ).requires_grad_()
        extended = executor.execute_context_with_accounting(
            extended_values,
            sequence=_sequence(
                torch.ones(1, 5, dtype=torch.bool),
                torch.tensor([[10, 11, 14, 20, 25]]),
            ),
            prefix="compiled.block",
        )

        self.assertTrue(
            torch.equal(prefix.route_ids, extended.route_ids[:, :3])
        )
        torch.testing.assert_close(
            prefix.output,
            extended.output[:, :3],
            rtol=0.0,
            atol=0.0,
        )
        gradient = torch.autograd.grad(
            extended.output[:, :3].sum(),
            extended_values,
        )[0]
        torch.testing.assert_close(
            gradient[:, 3:],
            torch.zeros_like(gradient[:, 3:]),
            rtol=0.0,
            atol=0.0,
        )

    def test_oblique_codec_decodes_only_the_selected_columns(self) -> None:
        codec = build_generalized_fisher_codec(
            activation_name="block.output",
            mean=torch.tensor([0.5, -0.25], dtype=torch.float64),
            covariance=torch.tensor(
                [[4.0, 1.0], [1.0, 1.0]],
                dtype=torch.float64,
            ),
            fisher_matrix=torch.tensor(
                [[2.0, 0.3], [0.3, 1.0]],
                dtype=torch.float64,
            ),
            alpha=0.1,
            beta=0.1,
        )
        self.assertFalse(
            torch.allclose(codec.encoder, codec.decoder)
        )
        executor = _executor(
            torch.tensor([[True, False]]),
            codec=codec,
            dtype=torch.float64,
        ).eval()
        values = torch.tensor(
            [[[2.0, -1.0], [-0.5, 3.0]]],
            dtype=torch.float64,
        )

        actual = executor(values, prefix="compiled.block")
        coordinates = (values - codec.mean) @ codec.encoder[:, :1]
        decoded_delta = (
            coordinates @ codec.decoder[:, :1].T + codec.mean
        )

        torch.testing.assert_close(
            actual,
            values + decoded_delta,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_codec_runtime_buffers_do_not_alias_the_caller_codec(self) -> None:
        codec = _codec(
            2,
            mean=torch.tensor([1.0, -2.0], dtype=torch.float64),
        )
        executor = _executor(
            torch.ones(1, 2, dtype=torch.bool),
            codec=codec,
            dtype=torch.float64,
        )
        expected_mean = executor.output_delta_mean.clone()
        expected_decoder = executor.output_decoder.clone()
        self.assertNotEqual(
            executor.output_delta_mean.data_ptr(),
            codec.mean.data_ptr(),
        )
        self.assertNotEqual(
            executor.output_decoder.data_ptr(),
            codec.decoder.data_ptr(),
        )

        codec.mean.add_(1_000.0)
        codec.decoder.fill_(-1_000.0)

        torch.testing.assert_close(executor.output_delta_mean, expected_mean)
        torch.testing.assert_close(executor.output_decoder, expected_decoder)

    def test_compiled_run_authenticates_both_boundaries(self) -> None:
        executor = _executor(torch.ones(1, 3, dtype=torch.bool)).eval()
        values = torch.tensor(
            [[[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]]
        )
        sequence = _sequence(
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[4, 7]]),
        )

        result = executor.run(
            _compiled_segment(),
            values,
            sequence,
        )

        torch.testing.assert_close(
            result.hidden_states,
            values * 2,
            rtol=0.0,
            atol=0.0,
        )
        self.assertIs(result.sequence, sequence)
        self.assertEqual(
            result.raw_output["route_ids"].tolist(),  # type: ignore[index]
            [[0, 0]],
        )
        for segment in (
            _compiled_segment(input_activation="wrong.input"),
            _compiled_segment(output_activation="wrong.output"),
        ):
            with self.subTest(segment=segment), self.assertRaisesRegex(
                ValueError,
                "boundaries",
            ):
                executor.run(segment, values, sequence)

    def test_strict_artifact_roundtrip_is_complete_and_source_free(self) -> None:
        codec = build_generalized_fisher_codec(
            activation_name="block.output",
            mean=torch.tensor([0.25, -0.5], dtype=torch.float64),
            covariance=torch.tensor(
                [[3.0, 0.5], [0.5, 1.0]],
                dtype=torch.float64,
            ),
            fisher_matrix=torch.tensor(
                [[1.0, 0.2], [0.2, 2.0]],
                dtype=torch.float64,
            ),
            alpha=0.1,
            beta=0.1,
        )
        executor = _executor(
            torch.tensor(
                [[False, False], [True, False], [True, True]]
            ),
            codec=codec,
            dtype=torch.float64,
            route_source="causal_hidden",
        ).eval()
        values = torch.tensor(
            [[[3.0, 0.0], [0.0, 4.0], [5.0, 1.0]]],
            dtype=torch.float64,
        )
        artifact = executor.artifact_state_dict()

        restored = (
            ConditionalCausalModalBlockExecutor.from_artifact_state_dict(
                artifact
            )
        )

        self.assertEqual(
            restored.execution_fingerprint(),
            executor.execution_fingerprint(),
        )
        self.assertEqual(restored.input_activation_name, "block.input")
        self.assertEqual(restored.output_activation_name, "block.output")
        self.assertEqual(restored.route_source, "causal_hidden")
        self.assertEqual(artifact["route_source"], "causal_hidden")
        self.assertEqual(
            restored.routing_plan.profile_semantics,
            executor.routing_plan.profile_semantics,
        )
        torch.testing.assert_close(
            restored.routing_plan.mode_table.mode_masks,
            executor.routing_plan.mode_table.mode_masks,
        )
        torch.testing.assert_close(
            restored.routing_plan.router.weight,
            executor.routing_plan.router.weight,
        )
        torch.testing.assert_close(
            restored(values, prefix="compiled.block"),
            executor(values, prefix="compiled.block"),
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotIn("encoder", artifact)
        self.assertNotIn("source", artifact)
        self.assertIn("routing_plan", artifact)

        unknown_field = copy.deepcopy(artifact)
        unknown_field["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            ConditionalCausalModalBlockExecutor.from_artifact_state_dict(
                unknown_field
            )

        missing_route_source = copy.deepcopy(artifact)
        del missing_route_source["route_source"]
        with self.assertRaisesRegex(ValueError, "fields"):
            ConditionalCausalModalBlockExecutor.from_artifact_state_dict(
                missing_route_source
            )

        invalid_route_source = copy.deepcopy(artifact)
        invalid_route_source["route_source"] = "future_state"
        with self.assertRaisesRegex(ValueError, "route_source"):
            ConditionalCausalModalBlockExecutor.from_artifact_state_dict(
                invalid_route_source
            )

        changed_route_source = copy.deepcopy(artifact)
        changed_route_source["route_source"] = "block_input"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            ConditionalCausalModalBlockExecutor.from_artifact_state_dict(
                changed_route_source
            )

        corrupted = copy.deepcopy(artifact)
        corrupted["graph_state_dict"]["output_bias"][0] += 1.0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            ConditionalCausalModalBlockExecutor.from_artifact_state_dict(
                corrupted
            )

    def test_executor_artifact_contains_no_source_module_or_fallback(self) -> None:
        executor = _executor(torch.ones(1, 3, dtype=torch.bool)).eval()
        self.assertIsInstance(executor, LayerExecutor)
        self.assertFalse(hasattr(executor, "source"))
        self.assertFalse(hasattr(executor, "native_source"))
        self.assertFalse(hasattr(executor, "fallback"))
        self.assertTrue(
            all(
                "source" not in name and "fallback" not in name
                for name in executor.state_dict()
            )
        )

        before = executor.execution_fingerprint()
        values = torch.randn(1, 2, 3)
        executor(values, prefix="compiled.block")
        after = executor.execution_fingerprint()
        self.assertEqual(before, after)
        status = executor.execution_status()
        self.assertTrue(status.executor_local_source_free)
        self.assertFalse(hasattr(status, "source_fallback_calls"))
        self.assertFalse(hasattr(status, "native_source_calls"))


if __name__ == "__main__":
    unittest.main()
