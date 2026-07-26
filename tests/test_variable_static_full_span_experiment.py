from pathlib import Path
import tempfile
import unittest

import torch

from fisher_graph.modes import FisherModeBasis
from fisher_graph.model import ToyTransformer
from fisher_graph.static_transformer_span_executor import (
    StaticTransformerSpanExecutor,
)
from fisher_graph.variable_associative import (
    VariableAssociativeRecallTaskConfig,
    build_variable_associative_recall_splits,
    variable_associative_recall_model_config,
)
from fisher_graph.variable_static_full_span_experiment import (
    ROLE_NAMES,
    StaticGraphCandidate,
    _allocate_context_roles,
    _behavior_summary,
    _bootstrap_nll_degradation,
    _compute_accounting,
    _development_overlap_audit,
    _direct_replacement_answer_logits,
    _save_result,
    _select_architecture_and_seed,
    _sequence_context,
    verify_variable_static_full_span_artifacts,
)


class VariableStaticFullSpanExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = VariableAssociativeRecallTaskConfig(
            n_keys=4,
            n_values=4,
        )
        self.splits = build_variable_associative_recall_splits(self.task)

    def test_role_allocator_excludes_predecessor_and_evaluation_contexts(
        self,
    ) -> None:
        predecessor = {
            "old_fit": self.splits.train.semantic_context_hashes[:3],
            "old_calibration": self.splits.train.semantic_context_hashes[3:6],
        }
        role_sizes = {name: 2 for name in ROLE_NAMES}
        first, first_reserve = _allocate_context_roles(
            self.splits.train,
            predecessor_role_hashes=predecessor,
            validation_hashes=self.splits.validation.semantic_context_hashes,
            test_hashes=self.splits.test.semantic_context_hashes,
            role_sizes=role_sizes,
            salt="unit-test",
        )
        second, second_reserve = _allocate_context_roles(
            self.splits.train,
            predecessor_role_hashes=predecessor,
            validation_hashes=self.splits.validation.semantic_context_hashes,
            test_hashes=self.splits.test.semantic_context_hashes,
            role_sizes=role_sizes,
            salt="unit-test",
        )

        self.assertEqual(tuple(first), ROLE_NAMES)
        self.assertEqual(
            {
                name: split.semantic_context_hashes
                for name, split in first.items()
            },
            {
                name: split.semantic_context_hashes
                for name, split in second.items()
            },
        )
        self.assertEqual(first_reserve, second_reserve)
        excluded = (
            set(predecessor["old_fit"])
            | set(predecessor["old_calibration"])
            | set(self.splits.validation.semantic_context_hashes)
            | set(self.splits.test.semantic_context_hashes)
        )
        used = set()
        for split in first.values():
            self.assertEqual(split.contexts, 2)
            self.assertTrue(set(split.semantic_context_hashes).isdisjoint(excluded))
            self.assertTrue(set(split.semantic_context_hashes).isdisjoint(used))
            used.update(split.semantic_context_hashes)
        self.assertTrue(set(first_reserve).isdisjoint(used | excluded))

        audit = _development_overlap_audit(
            first,
            first_reserve,
            ad_hoc_probe_role_hashes={
                "probe": first["calibration_c"].semantic_context_hashes[:1],
            },
        )
        self.assertEqual(audit["calibration_c_overlap_count"], 1)
        self.assertFalse(audit["calibration_c_confirmatory"])

    def test_strong_behavior_is_stric_superset_of_minimum_gate(self) -> None:
        split = self.splits.validation
        logits = torch.full(
            (split.samples, self.task.vocab_size),
            -12.0,
        )
        logits.scatter_(1, split.answer_token_ids.unsqueeze(1), 12.0)

        record = _behavior_summary(logits, logits, split)

        self.assertTrue(record["minimum_viability_passed"])
        self.assertTrue(record["strong_passed"])
        self.assertTrue(all(record["strong_gates"].values()))

        degraded = torch.zeros_like(logits)
        failed = _behavior_summary(degraded, logits, split)
        self.assertFalse(failed["strong_passed"])

    def test_architecture_requires_two_strong_seeds_and_selects_cheapest(
        self,
    ) -> None:
        def record(
            name: str,
            seed: int,
            *,
            strong: bool,
            nll: float,
            macs: int,
        ) -> dict[str, object]:
            return {
                "candidate": {
                    "name": name,
                    "hidden_width": 8,
                    "layer_count": 2,
                    "head_count": 2,
                    "feed_forward_width": 16,
                },
                "seed": seed,
                "graph_select_a": {
                    "strong_passed": strong,
                    "hard_nll": nll,
                },
                "accounting": {
                    "static_graph": {
                        "complete_macs": macs,
                        "runtime_stored_coefficients": macs // 2,
                    }
                },
            }

        records = [
            record("cheap", 1, strong=True, nll=0.06, macs=100),
            record("cheap", 2, strong=True, nll=0.05, macs=100),
            record("cheap", 3, strong=False, nll=0.20, macs=100),
            record("expensive", 1, strong=True, nll=0.04, macs=200),
            record("expensive", 2, strong=True, nll=0.04, macs=200),
            record("expensive", 3, strong=True, nll=0.04, macs=200),
            record("unstable", 1, strong=True, nll=0.03, macs=50),
            record("unstable", 2, strong=False, nll=0.30, macs=50),
            record("unstable", 3, strong=False, nll=0.31, macs=50),
        ]

        selected = _select_architecture_and_seed(records)

        self.assertEqual(selected["selected_runtime_key"], ("cheap", 1))
        summaries = {
            item["candidate"]["name"]: item
            for item in selected["architectures"]
        }
        self.assertTrue(summaries["cheap"]["architecture_passed"])
        self.assertFalse(summaries["unstable"]["architecture_passed"])

    def test_bootstrap_resamples_whole_semantic_contexts_deterministically(
        self,
    ) -> None:
        split = self.splits.validation
        baseline = torch.zeros(split.samples, self.task.vocab_size)
        candidate = baseline.clone()
        candidate.scatter_(
            1,
            split.answer_token_ids.unsqueeze(1),
            torch.full((split.samples, 1), 0.25),
        )

        first = _bootstrap_nll_degradation(
            candidate,
            baseline,
            split,
            seed=811,
            samples=100,
        )
        second = _bootstrap_nll_degradation(
            candidate,
            baseline,
            split,
            seed=811,
            samples=100,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["resampling_units"], split.contexts)
        self.assertLess(first["mean_nll_degradation"], 0.0)

    def test_accounting_and_direct_runner_bypass_all_source_layers(
        self,
    ) -> None:
        torch.manual_seed(901)
        model = ToyTransformer(
            variable_associative_recall_model_config(self.task)
        ).eval()
        split = self.splits.validation
        candidate = StaticGraphCandidate("test", 16, 2, 4, 32)
        decoder = torch.linalg.qr(torch.randn(32, 4)).Q
        executor = StaticTransformerSpanExecutor(
            candidate.executor_config(
                residual_width=32,
                retained_rank=4,
            ),
            decoder,
        ).eval()

        logits, calls = _direct_replacement_answer_logits(
            model,
            executor,
            split,
        )
        accounting = _compute_accounting(model, split, executor)
        executor_accounting = executor.logical_accounting(
            _sequence_context(split)
        )

        self.assertEqual(logits.shape, (split.samples, self.task.vocab_size))
        self.assertEqual(calls, (0, 0, 0))
        self.assertEqual(
            accounting["static_graph"]["input_projection_macs"],
            executor_accounting.input_projection_macs,
        )
        self.assertEqual(
            accounting["static_graph"]["mini_transformer_macs"],
            executor_accounting.transformer_trunk_macs,
        )
        self.assertEqual(
            accounting["static_graph_reference_dense_prefix"][
                "reference_dense_prefix_total_macs"
            ],
            executor_accounting.reference_dense_prefix_total_macs,
        )
        self.assertLess(
            accounting["graph_to_native_complete_mac_ratio"],
            1.0,
        )
        self.assertLess(
            accounting["graph_to_native_source_storage_ratio"],
            1.0,
        )

    def test_outer_artifact_is_weights_only_source_free_and_bound_to_decoder(
        self,
    ) -> None:
        torch.manual_seed(902)
        width = 8
        rank = 3
        vectors = torch.linalg.qr(torch.randn(width, width)).Q.to(torch.float64)
        basis = FisherModeBasis(
            activation_name="layer.2.output",
            mean=torch.zeros(width, dtype=torch.float64),
            matrix=torch.eye(width, dtype=torch.float64),
            eigenvalues=torch.ones(width, dtype=torch.float64),
            vectors=vectors,
            observations=10,
            sequences=10,
        )
        scale = torch.tensor([0.5, 1.0, 2.0])
        candidate = StaticGraphCandidate("artifact", 8, 2, 2, 16)
        executor = StaticTransformerSpanExecutor(
            candidate.executor_config(
                residual_width=width,
                retained_rank=rank,
            ),
            vectors[:, :rank].to(torch.float32) * scale.unsqueeze(0),
        ).eval()
        candidate_record = {
            "name": candidate.name,
            "hidden_width": candidate.hidden_width,
            "layer_count": candidate.layer_count,
            "head_count": candidate.head_count,
            "feed_forward_width": candidate.feed_forward_width,
        }
        analysis = {
            "selection_a": {
                "runs": [
                    {
                        "candidate": candidate_record,
                        "seed": 1,
                        "executor_fingerprint": (
                            executor.execution_fingerprint()
                        ),
                        "runtime_stored_coefficient_count": (
                            executor.total_runtime_coefficient_count
                        ),
                    }
                ],
                "selected": {
                    "candidate": candidate_record,
                    "selected_median_strong_seed": 1,
                },
            },
            "calibration_c": None,
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.pt"
            _save_result(
                output=output,
                source={"checkpoint_sha256": "a" * 64},
                predecessor={"artifact_sha256": "b" * 64},
                protocol={"test_policy": "untouched"},
                basis=basis,
                coordinate_scale=scale,
                executor_artifact=executor.artifact_state_dict(),
                analysis=analysis,
                scientific_status={"test_evaluated": False},
            )
            report = verify_variable_static_full_span_artifacts(output)
            payload = torch.load(output, map_location="cpu", weights_only=True)
            changed = StaticTransformerSpanExecutor.from_artifact_state_dict(
                payload["executor"]
            )
            with torch.no_grad():
                changed.output_head.bias[0] += 0.25
            changed.eval()
            payload["executor"] = changed.artifact_state_dict()
            torch.save(payload, output)
            with self.assertRaisesRegex(ValueError, "fingerprint binding"):
                verify_variable_static_full_span_artifacts(output)

        self.assertFalse(report["contains_source_model_weights"])
        self.assertTrue(report["contains_compiled_executor_weights"])
        self.assertNotIn("model_state_dict", payload["source"])
        self.assertFalse(
            payload["executor"]["contains_source_model_weights"]
        )
        self.assertFalse(payload["executor"]["contains_source_fallback"])


if __name__ == "__main__":
    unittest.main()
