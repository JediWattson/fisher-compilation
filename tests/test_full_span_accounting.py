import unittest
from dataclasses import FrozenInstanceError

from fisher_graph.dynamic_executor import StatefulCausalModalGraph
from fisher_graph.full_span_accounting import (
    conditional_causal_graph_accounting,
    native_transformer_span_accounting,
)


class NativeTransformerSpanAccountingTests(unittest.TestCase):
    def test_exact_logical_padded_mac_and_parameter_formulas(self) -> None:
        report = native_transformer_span_accounting(
            valid_rows=10,
            causal_pairs=30,
            width=4,
            feed_forward_width=8,
            layer_count=3,
            padded_rows=12,
            padded_causal_pairs=45,
        )

        self.assertEqual(report.qkv_projection_macs, 1_440)
        self.assertEqual(report.attention_output_projection_macs, 480)
        self.assertEqual(report.attention_score_macs, 360)
        self.assertEqual(report.attention_value_macs, 360)
        self.assertEqual(report.feed_forward_input_macs, 960)
        self.assertEqual(report.feed_forward_output_macs, 960)
        self.assertEqual(report.logical_attention_projection_macs, 1_920)
        self.assertEqual(report.logical_attention_pair_macs, 720)
        self.assertEqual(report.logical_feed_forward_macs, 1_920)
        self.assertEqual(report.logical_total_macs, 4_560)
        self.assertEqual(report.padded_total_macs, 5_688)
        self.assertEqual(report.padding_row_count, 2)
        self.assertEqual(report.padding_pair_count, 15)
        self.assertEqual(report.padding_mac_overhead, 1_128)
        self.assertAlmostEqual(
            report.logical_to_padded_mac_ratio,
            4_560 / 5_688,
        )

        self.assertEqual(report.linear_weight_parameter_count, 384)
        self.assertEqual(report.affine_bias_parameter_count, 84)
        self.assertEqual(report.normalization_parameter_count, 48)
        self.assertEqual(report.total_parameter_count, 516)

    def test_default_padded_counts_equal_logical_counts(self) -> None:
        report = native_transformer_span_accounting(
            valid_rows=7,
            causal_pairs=19,
            width=3,
            feed_forward_width=5,
            layer_count=2,
        )

        self.assertEqual(report.padded_rows, 7)
        self.assertEqual(report.padded_causal_pairs, 19)
        self.assertEqual(report.padded_total_macs, report.logical_total_macs)
        self.assertEqual(report.padding_mac_overhead, 0)
        self.assertEqual(report.logical_to_padded_mac_ratio, 1.0)

    def test_zero_workload_has_finite_zero_accounting(self) -> None:
        report = native_transformer_span_accounting(
            valid_rows=0,
            causal_pairs=0,
            width=0,
            feed_forward_width=0,
            layer_count=0,
        )

        self.assertEqual(report.logical_total_macs, 0)
        self.assertEqual(report.padded_total_macs, 0)
        self.assertEqual(report.total_parameter_count, 0)
        self.assertEqual(report.logical_to_padded_mac_ratio, 0.0)

    def test_rejects_negative_boolean_and_inverted_padded_counts(self) -> None:
        valid = {
            "valid_rows": 2,
            "causal_pairs": 3,
            "width": 4,
            "feed_forward_width": 8,
            "layer_count": 1,
        }
        for name in valid:
            with self.subTest(name=name):
                arguments = dict(valid)
                arguments[name] = -1
                with self.assertRaisesRegex(ValueError, name):
                    native_transformer_span_accounting(**arguments)
                arguments[name] = True
                with self.assertRaisesRegex(ValueError, name):
                    native_transformer_span_accounting(**arguments)

        with self.assertRaisesRegex(ValueError, "padded_rows"):
            native_transformer_span_accounting(
                **valid,
                padded_rows=1,
            )
        with self.assertRaisesRegex(ValueError, "padded_causal_pairs"):
            native_transformer_span_accounting(
                **valid,
                padded_causal_pairs=2,
            )

    def test_report_is_immutable(self) -> None:
        report = native_transformer_span_accounting(
            valid_rows=1,
            causal_pairs=1,
            width=2,
            feed_forward_width=3,
            layer_count=1,
        )
        with self.assertRaises(FrozenInstanceError):
            report.valid_rows = 4


class ConditionalCausalGraphAccountingTests(unittest.TestCase):
    def test_exact_mac_recurrence_and_coefficient_formulas(self) -> None:
        report = conditional_causal_graph_accounting(
            key_rows=10,
            query_rows=2,
            width=4,
            state_channels=3,
            hidden_width=5,
            routes=2,
            active_ranks=(1, 3),
            padded_key_rows=12,
            padded_query_rows=4,
            logical_causal_pairs=30,
            padded_causal_pairs=45,
        )

        self.assertEqual(report.active_rank_applications, 4)
        self.assertEqual(report.average_active_rank, 2.0)
        self.assertTrue(report.include_router)
        self.assertEqual(report.stored_output_modes, 4)
        self.assertEqual(report.state_input_macs, 600)
        self.assertEqual(report.hidden_projection_macs, 150)
        self.assertEqual(report.shared_trunk_macs, 750)
        self.assertEqual(report.router_macs, 20)
        self.assertEqual(report.output_head_macs, 20)
        self.assertEqual(report.decoder_macs, 16)
        self.assertEqual(report.route_tail_macs, 36)
        self.assertEqual(report.ideal_total_macs, 806)
        self.assertEqual(report.dense_route_tail_macs, 72)
        self.assertEqual(report.route_tail_to_dense_ratio, 0.5)

        self.assertEqual(
            report.recurrence_decay_argument_multiplications,
            30,
        )
        self.assertEqual(report.recurrence_exponential_evaluations, 30)
        self.assertEqual(report.recurrence_state_multiplications, 150)
        self.assertEqual(report.recurrence_state_additions, 150)
        self.assertEqual(report.recurrence_elementwise_arithmetic_ops, 330)

        self.assertEqual(report.state_input_parameter_count, 60)
        self.assertEqual(report.decay_parameter_count, 3)
        self.assertEqual(report.hidden_weight_parameter_count, 75)
        self.assertEqual(report.hidden_bias_parameter_count, 5)
        self.assertEqual(report.shared_trunk_parameter_count, 143)
        self.assertEqual(report.output_head_weight_parameter_count, 20)
        self.assertEqual(report.output_head_bias_parameter_count, 4)
        self.assertEqual(report.graph_trainable_parameter_count, 167)
        self.assertEqual(report.router_weight_coefficient_count, 10)
        self.assertEqual(report.router_bias_coefficient_count, 2)
        self.assertEqual(report.router_normalization_coefficient_count, 10)
        self.assertEqual(report.router_stored_coefficient_count, 22)
        self.assertEqual(report.decoder_coefficient_count, 16)
        self.assertEqual(report.delta_mean_coefficient_count, 4)
        self.assertEqual(report.codec_fixed_coefficient_count, 20)
        self.assertEqual(report.total_learned_coefficient_count, 179)
        self.assertEqual(
            report.total_floating_runtime_coefficient_count,
            209,
        )
        self.assertEqual(report.route_mask_boolean_count, 8)
        self.assertEqual(report.total_runtime_scalar_count, 217)

        self.assertEqual(report.padding_key_rows, 2)
        self.assertEqual(report.padding_query_rows, 2)
        self.assertEqual(report.padding_pair_count, 15)

    def test_graph_parameter_formula_matches_runtime_module(self) -> None:
        graph = StatefulCausalModalGraph(
            input_modes=4,
            output_modes=4,
            state_channels=3,
            routing_width=5,
        )
        report = conditional_causal_graph_accounting(
            key_rows=1,
            query_rows=1,
            width=4,
            state_channels=3,
            hidden_width=5,
            routes=2,
            active_rank_applications=2,
        )

        self.assertEqual(
            report.graph_trainable_parameter_count,
            graph.learned_parameters,
        )

    def test_router_free_structurally_pruned_static_comparator(self) -> None:
        report = conditional_causal_graph_accounting(
            key_rows=10,
            query_rows=2,
            width=4,
            state_channels=3,
            hidden_width=5,
            routes=2,
            include_router=False,
            stored_output_modes=3,
            active_ranks=(3, 3),
        )

        self.assertFalse(report.include_router)
        self.assertEqual(report.stored_output_modes, 3)
        self.assertEqual(report.active_rank_applications, 6)
        self.assertEqual(report.average_active_rank, 3.0)
        self.assertEqual(report.shared_trunk_macs, 750)
        self.assertEqual(report.router_macs, 0)
        self.assertEqual(report.output_head_macs, 30)
        self.assertEqual(report.decoder_macs, 24)
        self.assertEqual(report.route_tail_macs, 54)
        self.assertEqual(report.dense_route_tail_macs, 54)
        self.assertEqual(report.route_tail_to_dense_ratio, 1.0)
        self.assertEqual(report.ideal_total_macs, 804)

        self.assertEqual(report.shared_trunk_parameter_count, 143)
        self.assertEqual(report.output_head_weight_parameter_count, 15)
        self.assertEqual(report.output_head_bias_parameter_count, 3)
        self.assertEqual(report.graph_trainable_parameter_count, 161)
        self.assertEqual(report.router_weight_coefficient_count, 0)
        self.assertEqual(report.router_bias_coefficient_count, 0)
        self.assertEqual(report.router_normalization_coefficient_count, 0)
        self.assertEqual(report.router_stored_coefficient_count, 0)
        self.assertEqual(report.decoder_coefficient_count, 12)
        self.assertEqual(report.delta_mean_coefficient_count, 4)
        self.assertEqual(report.codec_fixed_coefficient_count, 16)
        self.assertEqual(report.total_learned_coefficient_count, 161)
        self.assertEqual(
            report.total_floating_runtime_coefficient_count,
            177,
        )
        self.assertEqual(report.route_mask_boolean_count, 0)
        self.assertEqual(report.total_runtime_scalar_count, 177)

    def test_structural_mode_count_sizes_routed_storage(self) -> None:
        report = conditional_causal_graph_accounting(
            key_rows=10,
            query_rows=2,
            width=4,
            state_channels=3,
            hidden_width=5,
            routes=2,
            include_router=True,
            stored_output_modes=3,
            active_ranks=(1, 3),
        )

        self.assertEqual(report.router_macs, 20)
        self.assertEqual(report.output_head_weight_parameter_count, 15)
        self.assertEqual(report.output_head_bias_parameter_count, 3)
        self.assertEqual(report.decoder_coefficient_count, 12)
        self.assertEqual(report.delta_mean_coefficient_count, 4)
        self.assertEqual(report.route_mask_boolean_count, 6)
        self.assertEqual(report.dense_route_tail_macs, 54)

    def test_rank_sequence_and_aggregate_are_equivalent(self) -> None:
        common = {
            "key_rows": 9,
            "query_rows": 3,
            "width": 6,
            "state_channels": 2,
            "hidden_width": 4,
            "routes": 5,
        }
        per_query = conditional_causal_graph_accounting(
            **common,
            active_ranks=(0, 2, 6),
        )
        aggregate = conditional_causal_graph_accounting(
            **common,
            active_rank_applications=8,
        )
        both = conditional_causal_graph_accounting(
            **common,
            active_ranks=(0, 2, 6),
            active_rank_applications=8,
        )

        self.assertEqual(per_query, aggregate)
        self.assertEqual(aggregate, both)

    def test_pair_counts_are_provenance_not_recurrent_work(self) -> None:
        arguments = {
            "key_rows": 10,
            "query_rows": 2,
            "width": 4,
            "state_channels": 3,
            "hidden_width": 5,
            "routes": 2,
            "active_rank_applications": 4,
        }
        without_pairs = conditional_causal_graph_accounting(**arguments)
        with_pairs = conditional_causal_graph_accounting(
            **arguments,
            padded_key_rows=20,
            padded_query_rows=8,
            logical_causal_pairs=30,
            padded_causal_pairs=150,
        )

        self.assertIsNone(without_pairs.logical_causal_pairs)
        self.assertIsNone(without_pairs.padding_pair_count)
        self.assertEqual(with_pairs.padding_pair_count, 120)
        self.assertEqual(
            with_pairs.ideal_total_macs,
            without_pairs.ideal_total_macs,
        )
        self.assertEqual(
            with_pairs.recurrence_elementwise_arithmetic_ops,
            without_pairs.recurrence_elementwise_arithmetic_ops,
        )

    def test_zero_query_rows_support_empty_rank_sequence(self) -> None:
        report = conditional_causal_graph_accounting(
            key_rows=5,
            query_rows=0,
            width=3,
            state_channels=2,
            hidden_width=4,
            routes=2,
            active_ranks=(),
        )

        self.assertEqual(report.average_active_rank, 0.0)
        self.assertEqual(report.router_macs, 0)
        self.assertEqual(report.route_tail_macs, 0)
        self.assertEqual(report.route_tail_to_dense_ratio, 0.0)
        self.assertEqual(report.shared_trunk_macs, 120)

    def test_rejects_invalid_dimensions_rank_inputs_and_pair_provenance(
        self,
    ) -> None:
        valid = {
            "key_rows": 5,
            "query_rows": 2,
            "width": 4,
            "state_channels": 3,
            "hidden_width": 6,
            "routes": 2,
            "active_ranks": (1, 2),
        }
        for name in (
            "key_rows",
            "query_rows",
            "width",
            "state_channels",
            "hidden_width",
            "routes",
        ):
            with self.subTest(name=name):
                arguments = dict(valid)
                arguments[name] = -1
                with self.assertRaisesRegex(ValueError, name):
                    conditional_causal_graph_accounting(**arguments)
                arguments[name] = True
                with self.assertRaisesRegex(ValueError, name):
                    conditional_causal_graph_accounting(**arguments)

        with self.assertRaisesRegex(ValueError, "provide active"):
            conditional_causal_graph_accounting(
                **{key: value for key, value in valid.items() if key != "active_ranks"}
            )
        with self.assertRaisesRegex(ValueError, "one rank"):
            conditional_causal_graph_accounting(
                **{**valid, "active_ranks": (1,)},
            )
        with self.assertRaisesRegex(
            ValueError,
            "between zero and stored_output_modes",
        ):
            conditional_causal_graph_accounting(
                **{**valid, "active_ranks": (1, 5)},
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            conditional_causal_graph_accounting(
                **{
                    key: value
                    for key, value in valid.items()
                    if key != "active_ranks"
                },
                active_rank_applications=9,
            )
        with self.assertRaisesRegex(ValueError, "disagree"):
            conditional_causal_graph_accounting(
                **valid,
                active_rank_applications=4,
            )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            conditional_causal_graph_accounting(
                **valid,
                logical_causal_pairs=7,
            )
        with self.assertRaisesRegex(ValueError, "padded_causal_pairs"):
            conditional_causal_graph_accounting(
                **valid,
                logical_causal_pairs=7,
                padded_causal_pairs=6,
            )
        with self.assertRaisesRegex(ValueError, "padded_key_rows"):
            conditional_causal_graph_accounting(
                **valid,
                padded_key_rows=4,
            )
        with self.assertRaisesRegex(ValueError, "padded_query_rows"):
            conditional_causal_graph_accounting(
                **valid,
                padded_query_rows=1,
            )

    def test_rejects_invalid_router_flag_and_structural_mode_count(
        self,
    ) -> None:
        valid = {
            "key_rows": 5,
            "query_rows": 2,
            "width": 4,
            "state_channels": 3,
            "hidden_width": 6,
            "routes": 2,
            "active_ranks": (1, 2),
        }
        for invalid in (0, 1, "yes", None):
            with self.subTest(include_router=invalid):
                with self.assertRaisesRegex(ValueError, "include_router"):
                    conditional_causal_graph_accounting(
                        **valid,
                        include_router=invalid,
                    )
        for invalid in (-1, True, 5):
            with self.subTest(stored_output_modes=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "stored_output_modes",
                ):
                    conditional_causal_graph_accounting(
                        **valid,
                        stored_output_modes=invalid,
                    )

        with self.assertRaisesRegex(
            ValueError,
            "between zero and stored_output_modes",
        ):
            conditional_causal_graph_accounting(
                **{**valid, "active_ranks": (1, 3)},
                stored_output_modes=2,
            )
        with self.assertRaisesRegex(
            ValueError,
            "query_rows \\* stored_output_modes",
        ):
            conditional_causal_graph_accounting(
                **{
                    key: value
                    for key, value in valid.items()
                    if key != "active_ranks"
                },
                active_rank_applications=5,
                stored_output_modes=2,
            )


if __name__ == "__main__":
    unittest.main()
