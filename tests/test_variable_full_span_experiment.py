import unittest

import torch

from fisher_graph.conditional_routing import ConditionalModeTable
from fisher_graph.modes import ActivationGradientSamples, FisherModeBasis
from fisher_graph.model import ToyTransformer
from fisher_graph.variable_associative import (
    VariableAssociativeRecallTaskConfig,
    build_variable_associative_recall_splits,
    variable_associative_recall_model_config,
)
from fisher_graph.variable_associative_training import (
    variable_associative_answer_logits,
)
from fisher_graph.variable_full_span_experiment import (
    ROLE_NAMES,
    _answer_rows,
    _assert_evaluation_contexts_are_disjoint,
    _assert_sample_alignment,
    _context_role_splits,
    _decision,
    _hypothetical_graph_envelopes,
    _key_mask,
    _projected_answer_logits,
    _query_mask,
    _select_router_candidate,
)


class VariableFullSpanExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = VariableAssociativeRecallTaskConfig(
            n_keys=4,
            n_values=4,
        )
        self.splits = build_variable_associative_recall_splits(self.task)

    def test_context_roles_are_complete_and_semantically_disjoint(
        self,
    ) -> None:
        roles = _context_role_splits(
            self.splits.train,
            contexts_per_role=2,
        )

        self.assertEqual(tuple(roles), ROLE_NAMES)
        for split in roles.values():
            self.assertEqual(split.contexts, 2)
            self.assertEqual(
                split.samples,
                2 * self.task.variants_per_context,
            )
        hashes = [
            set(split.semantic_context_hashes)
            for split in roles.values()
        ]
        for left in range(len(hashes)):
            for right in range(left + 1, len(hashes)):
                self.assertTrue(hashes[left].isdisjoint(hashes[right]))
        _assert_evaluation_contexts_are_disjoint(
            roles,
            self.splits.validation,
            self.splits.test,
        )
        with self.assertRaisesRegex(RuntimeError, "validation"):
            _assert_evaluation_contexts_are_disjoint(
                roles,
                roles["basis_a"],
                self.splits.test,
            )

    def test_sample_alignment_rejects_permuted_collector_rows(self) -> None:
        split = _context_role_splits(
            self.splits.train,
            contexts_per_role=1,
        )["basis_a"]
        metadata = split.valid_token_metadata()
        locations = torch.stack(
            (metadata.example_indices, metadata.logical_positions),
            dim=1,
        )
        sample = ActivationGradientSamples(
            name="layer.0.input",
            activations=torch.zeros(metadata.observations, 2),
            score_gradients=torch.zeros(metadata.observations, 2),
            locations=locations.flip(0),
            sequences=split.samples,
            sequence_ids=split.example_ids,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "activation-gradient rows",
        ):
            _assert_sample_alignment(split, sample)

    def test_query_and_key_masks_separate_output_demand_from_prefix(
        self,
    ) -> None:
        split = _context_role_splits(
            self.splits.train,
            contexts_per_role=1,
        )["calibration_b"]

        query = _query_mask(split)
        keys = _key_mask(split)

        self.assertEqual(int(query.sum().item()), split.samples)
        self.assertTrue((query <= keys).all())
        self.assertTrue((keys <= split.attention_mask).all())
        positions = torch.arange(split.maximum_sequence_length).unsqueeze(0)
        future = positions > split.supervised_positions.unsqueeze(1)
        self.assertFalse(keys[future].any())
        self.assertTrue(split.attention_mask[future].any())

    def test_answer_row_gather_uses_each_examples_supervised_position(
        self,
    ) -> None:
        split = self.splits.validation
        values = torch.arange(
            split.samples * split.maximum_sequence_length,
        ).reshape(split.samples, split.maximum_sequence_length)
        expected = values[
            torch.arange(split.samples),
            split.supervised_positions,
        ]

        torch.testing.assert_close(_answer_rows(values, split), expected)

    def test_full_rank_projection_is_identity_across_all_source_layers(
        self,
    ) -> None:
        torch.manual_seed(91_401)
        model = ToyTransformer(
            variable_associative_recall_model_config(self.task)
        ).eval()
        split = _context_role_splits(
            self.splits.train,
            contexts_per_role=1,
        )["calibration_b"]
        width = model.config.d_model
        basis = FisherModeBasis(
            activation_name="layer.2.output",
            mean=torch.zeros(width, dtype=torch.float64),
            matrix=torch.eye(width, dtype=torch.float64),
            eigenvalues=torch.ones(width, dtype=torch.float64),
            vectors=torch.eye(width, dtype=torch.float64),
            observations=split.samples,
            sequences=split.samples,
        )
        baseline = variable_associative_answer_logits(model, split)

        projected = _projected_answer_logits(
            model,
            split,
            basis,
            input_site="layer.0.input",
            output_site="layer.2.output",
            static_mask=torch.ones(width, dtype=torch.bool),
        )

        torch.testing.assert_close(
            projected,
            baseline,
            rtol=0.0,
            atol=0.0,
        )

    def test_router_decisions_and_candidate_selection_are_deterministic(
        self,
    ) -> None:
        logits = torch.tensor(
            [[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]]
        )
        self.assertEqual(_decision(logits, "argmax").tolist(), [0, 2])
        conservative = _decision(logits, "posterior_q95")
        self.assertTrue((conservative >= _decision(logits, "argmax")).all())

        candidates = [
            {
                "name": "larger",
                "router_plus_ideal_selective_projection_macs": 20,
                "behavior": {
                    "passed": True,
                    "hard_nll": 0.1,
                    "gates": {"quality": True},
                },
            },
            {
                "name": "smaller",
                "router_plus_ideal_selective_projection_macs": 10,
                "behavior": {
                    "passed": True,
                    "hard_nll": 0.2,
                    "gates": {"quality": True},
                },
            },
        ]
        self.assertEqual(
            _select_router_candidate(candidates)["name"],
            "smaller",
        )

    def test_hypothetical_envelopes_are_explicitly_unfitted(
        self,
    ) -> None:
        model = ToyTransformer(
            variable_associative_recall_model_config(self.task)
        )
        split = _context_role_splits(
            self.splits.train,
            contexts_per_role=1,
        )["calibration_b"]
        width = model.config.d_model
        masks = torch.zeros(4, width, dtype=torch.bool)
        for route, rank in enumerate((4, 8, 16, width)):
            masks[route, :rank] = True
        table = ConditionalModeTable.from_masks(masks)
        routes = torch.arange(split.samples) % table.routes

        records = _hypothetical_graph_envelopes(
            model,
            split,
            table=table,
            learned_routes=routes,
            static_rank=16,
        )

        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["fitted"] is False for record in records))
        self.assertTrue(
            all(
                record["behavior_validated"] is False
                for record in records
            )
        )
        self.assertTrue(
            all(
                0 < record["conditional_to_native_ratio"]
                for record in records
            )
        )
        for record in records:
            conditional = record["conditional"]
            static = record["same_trunk_static"]
            self.assertEqual(
                conditional["padded_key_rows"],
                static["padded_key_rows"],
            )
            self.assertEqual(
                conditional["padded_causal_pairs"],
                static["padded_causal_pairs"],
            )
            self.assertFalse(static["include_router"])
            self.assertEqual(static["stored_output_modes"], 16)


if __name__ == "__main__":
    unittest.main()
