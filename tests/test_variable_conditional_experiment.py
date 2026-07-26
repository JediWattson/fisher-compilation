import unittest

import torch

from fisher_graph.modes import FisherModeBasis
from fisher_graph.model import ToyTransformer
from fisher_graph.variable_associative import (
    VariableAssociativeRecallTaskConfig,
    build_variable_associative_recall_splits,
    variable_associative_recall_model_config,
)
from fisher_graph.variable_associative_training import (
    variable_associative_answer_logits,
)
from fisher_graph.variable_conditional_experiment import (
    _behavior_record,
    _bootstrap_advantage,
    _context_role_splits,
    _projected_answer_logits,
)


class VariableConditionalExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = VariableAssociativeRecallTaskConfig(
            n_keys=4,
            n_values=4,
        )
        self.splits = build_variable_associative_recall_splits(self.task)

    def test_compiler_context_roles_are_complete_and_disjoint(self) -> None:
        roles = _context_role_splits(
            self.splits.train,
            contexts_per_role=2,
        )
        self.assertEqual(
            set(roles),
            {"basis_a", "mask_a", "router_a", "calibration_b"},
        )
        for split in roles.values():
            self.assertEqual(split.contexts, 2)
            self.assertEqual(
                split.samples,
                2 * self.task.variants_per_context,
            )
        hash_sets = [
            set(split.semantic_context_hashes)
            for split in roles.values()
        ]
        for left in range(len(hash_sets)):
            for right in range(left + 1, len(hash_sets)):
                self.assertTrue(
                    hash_sets[left].isdisjoint(hash_sets[right])
                )

    def test_full_residual_delta_projection_is_an_identity(self) -> None:
        torch.manual_seed(701)
        model = ToyTransformer(
            variable_associative_recall_model_config(self.task)
        ).eval()
        split = _context_role_splits(
            self.splits.train,
            contexts_per_role=1,
        )["calibration_b"]
        width = model.config.d_model
        basis = FisherModeBasis(
            activation_name="layer.1.output",
            mean=torch.zeros(width, dtype=torch.float64),
            matrix=torch.eye(width, dtype=torch.float64),
            eigenvalues=torch.ones(width, dtype=torch.float64),
            vectors=torch.eye(width, dtype=torch.float64),
            observations=1,
            sequences=1,
        )
        baseline = variable_associative_answer_logits(model, split)
        projected, routes = _projected_answer_logits(
            model,
            split,
            input_site="layer.1.input",
            output_site="layer.1.output",
            basis=basis,
            static_mask=torch.ones(width, dtype=torch.bool),
        )
        self.assertIsNone(routes)
        torch.testing.assert_close(
            projected,
            baseline,
            rtol=0.0,
            atol=0.0,
        )

    def test_behavior_and_bootstrap_evidence_use_per_example_nll(self) -> None:
        split = self.splits.validation
        baseline = torch.full(
            (split.samples, self.task.vocab_size),
            -5.0,
        )
        rows = torch.arange(split.samples)
        baseline[rows, split.answer_token_ids] = 5.0
        record = _behavior_record(baseline, baseline, split)
        self.assertTrue(record["passed"])
        self.assertEqual(record["delta_nll"], 0.0)

        worse = baseline.clone()
        worse[rows, split.answer_token_ids] -= 1.0
        first = _bootstrap_advantage(
            baseline,
            worse,
            split,
            seed=99,
            samples=200,
        )
        second = _bootstrap_advantage(
            baseline,
            worse,
            split,
            seed=99,
            samples=200,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["resampling_unit"], "semantic_context")
        self.assertEqual(first["resampling_units"], split.contexts)
        self.assertGreater(first["lower_95_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
