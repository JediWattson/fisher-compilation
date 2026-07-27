import copy
from dataclasses import asdict, replace
import hashlib
import json
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
    variable_associative_recall_model_config,
)
from fisher_graph.variable_static_full_span_v2_experiment import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CANDIDATE,
    DEFAULT_EMA_DECAY,
    DEFAULT_INPUT_SITE,
    DEFAULT_MAXIMUM_RELATIVE_STORAGE,
    DEFAULT_MAXIMUM_RELATIVE_WORK,
    DEFAULT_OUTPUT_SITE,
    DEFAULT_REQUIRED_STRONG_SEEDS,
    DEFAULT_RETAINED_RANKS,
    DEFAULT_SEEDS,
    DEFAULT_TRAINING,
    _claim_protected_panel,
    _checkpoint_score_v2,
    _deployment_parameter_accounting,
    _save_result,
    _select_rank_and_seed,
    _validate_frozen_v2_recipe,
    _v2_behavior_summary,
    verify_variable_static_full_span_v2_artifacts,
)
from fisher_graph.variable_static_full_span_v2_protocol import (
    V2_TASK_CONFIG,
    build_variable_static_full_span_v2_protocol,
)


def _canonical_hash_set_digest(values: tuple[str, ...]) -> str:
    encoded = json.dumps(
        sorted(values),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class VariableStaticFullSpanV2ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = build_variable_static_full_span_v2_protocol()

    def _frozen_protocol_manifest(self) -> dict[str, object]:
        manifest = self.protocol.manifest()
        role_hashes = manifest["role_context_hashes"]
        assert isinstance(role_hashes, dict)
        manifest.update(
            {
                "input_site": DEFAULT_INPUT_SITE,
                "output_site": DEFAULT_OUTPUT_SITE,
                "source_layers": 3,
                "fisher_scope": (
                    "supervised answer row at output boundary"
                ),
                "retained_rank_candidates": DEFAULT_RETAINED_RANKS,
                "candidate": asdict(DEFAULT_CANDIDATE),
                "seeds": DEFAULT_SEEDS,
                "rank_pass_rule": (
                    "at least 4 of 5 graph_select_a seeds pass every "
                    "strong gate"
                ),
                "training": asdict(DEFAULT_TRAINING),
                "executor_parameter_ema_decay": DEFAULT_EMA_DECAY,
                "deterministic_algorithms": True,
                "strong_behavior_thresholds": {
                    "delta_nll": 0.007,
                    "answer_accuracy": 1.0,
                    "paired_context_accuracy": 1.0,
                    "minimum_layout_query_order_length_accuracy": 1.0,
                    "top1_agreement": 1.0,
                    "native_teacher_kl": 0.007,
                    "p90_absolute_delta_nll": 0.020,
                    "new_key_accuracy": 1.0,
                    "new_value_accuracy": 1.0,
                    "new_key_only_accuracy": 1.0,
                    "new_value_only_accuracy": 1.0,
                    "new_key_and_value_accuracy": 1.0,
                    "minimum_queried_key_accuracy": 1.0,
                    "minimum_answer_value_accuracy": 1.0,
                },
                "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
                "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
                "bootstrap_upper_degradation_threshold": 0.0085,
                "maximum_relative_work": DEFAULT_MAXIMUM_RELATIVE_WORK,
                "maximum_relative_storage": (
                    DEFAULT_MAXIMUM_RELATIVE_STORAGE
                ),
                "hash_set_digests": {
                    "excluded_baseline": _canonical_hash_set_digest(
                        manifest["excluded_baseline_context_hashes"]
                    ),
                    **{
                        name: _canonical_hash_set_digest(hashes)
                        for name, hashes in role_hashes.items()
                    },
                    "reserve": _canonical_hash_set_digest(
                        manifest["reserve_context_hashes"]
                    ),
                    "fresh_validation": _canonical_hash_set_digest(
                        manifest["fresh_validation_context_hashes"]
                    ),
                    "fresh_test": _canonical_hash_set_digest(
                        manifest["fresh_test_context_hashes"]
                    ),
                },
            }
        )
        return manifest

    def _canonical_source(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": "a" * 64,
            "dataset_sha256": self.protocol.source_dataset_sha256,
            "task_fingerprint": V2_TASK_CONFIG.fingerprint,
            "task_config": asdict(V2_TASK_CONFIG),
        }

    def _minimal_artifact_components(self) -> dict[str, object]:
        width = 4
        basis = FisherModeBasis(
            activation_name="layer.2.output",
            mean=torch.zeros(width, dtype=torch.float64),
            matrix=torch.eye(width, dtype=torch.float64),
            eigenvalues=torch.ones(width, dtype=torch.float64),
            vectors=torch.eye(width, dtype=torch.float64),
            observations=64,
            sequences=32,
        )
        return {
            "source": self._canonical_source(),
            "hypothesis": {"artifact_sha256": "b" * 64},
            "protocol": self._frozen_protocol_manifest(),
            "basis": basis,
            "coordinate_scale": torch.ones(
                width,
                dtype=torch.float32,
            ),
            "executor_artifact": None,
            "analysis": {
                "selection_a": {
                    "runs": [],
                    "ranks": [],
                    "selected": None,
                },
                "calibration_b": None,
                "fresh_validation": None,
            },
            "scientific_status": {
                "full_transformer_span_replaced": False,
                "source_independent_graph_fitted": False,
                "calibration_b_evaluated": False,
                "calibration_b_passed": False,
                "calibration_b_confirmatory": False,
                "fresh_validation_evaluated": False,
                "fresh_validation_passed": False,
                "executor_test_evaluated": False,
                "model_level_viable": False,
            },
        }

    def _save_minimal_artifact(self, output: Path) -> None:
        _save_result(
            output=output,
            **self._minimal_artifact_components(),
        )

    def test_perfect_logits_pass_behavior_and_all_novelty_gates(self) -> None:
        split = self.protocol.fresh_validation
        logits = torch.full(
            (split.samples, V2_TASK_CONFIG.vocab_size),
            -12.0,
        )
        logits[
            torch.arange(split.samples),
            split.answer_token_ids,
        ] = 12.0

        summary = _v2_behavior_summary(logits, logits.clone(), split)

        self.assertTrue(summary["minimum_viability_passed"])
        self.assertTrue(summary["strong_passed"])
        self.assertTrue(all(summary["strong_gates"].values()))
        self.assertEqual(summary["delta_nll"], 0.0)
        self.assertEqual(summary["top1_agreement"], 1.0)
        self.assertEqual(summary["answer_accuracy"], 1.0)
        self.assertEqual(summary["paired_context_accuracy"], 1.0)
        self.assertEqual(summary["minimum_stratum_accuracy"], 1.0)
        self.assertEqual(summary["native_teacher_kl"], 0.0)
        self.assertEqual(summary["p90_absolute_delta_nll"], 0.0)
        novelty = summary["novelty"]
        self.assertTrue(novelty["passes"])
        for name in ("new_key", "new_value", "both"):
            self.assertGreater(novelty[name]["contexts"], 0)
            self.assertEqual(novelty[name]["accuracy"], 1.0)

    def test_checkpoint_score_prioritizes_teacher_fidelity_then_delta(
        self,
    ) -> None:
        common = {
            "strong_passed": True,
            "minimum_viability_passed": True,
            "p90_absolute_delta_nll": 0.005,
        }
        lower_kl = {
            **common,
            "native_teacher_kl": 0.001,
            "delta_nll": -0.050,
        }
        closer_mean_but_higher_kl = {
            **common,
            "native_teacher_kl": 0.002,
            "delta_nll": 0.003,
        }
        self.assertLess(
            _checkpoint_score_v2(lower_kl, step=200),
            _checkpoint_score_v2(
                closer_mean_but_higher_kl,
                step=400,
            ),
        )

        closer_mean_at_equal_fidelity = {
            **lower_kl,
            "delta_nll": 0.003,
        }
        self.assertLess(
            _checkpoint_score_v2(
                closer_mean_at_equal_fidelity,
                step=200,
            ),
            _checkpoint_score_v2(lower_kl, step=400),
        )

    def test_selector_requires_four_of_five_and_uses_upper_median_seed(
        self,
    ) -> None:
        candidate = asdict(DEFAULT_CANDIDATE)

        def record(
            rank: int,
            seed: int,
            *,
            strong: bool,
            hard_nll: float,
            macs: int,
        ) -> dict[str, object]:
            return {
                "candidate": candidate,
                "retained_rank": rank,
                "seed": seed,
                "graph_select_a": {
                    "strong_passed": strong,
                    "hard_nll": hard_nll,
                },
                "accounting": {
                    "static_graph": {
                        "complete_macs": macs,
                        "runtime_stored_coefficients": macs // 4,
                    }
                },
            }

        records = [
            # Three passing seeds is insufficient even though this rank is
            # arithmetically cheapest.
            record(12, 1, strong=True, hard_nll=0.010, macs=80),
            record(12, 2, strong=True, hard_nll=0.011, macs=80),
            record(12, 3, strong=True, hard_nll=0.012, macs=80),
            record(12, 4, strong=False, hard_nll=0.050, macs=80),
            record(12, 5, strong=False, hard_nll=0.060, macs=80),
            # Four strong NLLs sort as seeds 4, 2, 3, 1. The conservative
            # upper median is therefore seed 3.
            record(14, 1, strong=True, hard_nll=0.050, macs=100),
            record(14, 2, strong=True, hard_nll=0.030, macs=100),
            record(14, 3, strong=True, hard_nll=0.040, macs=100),
            record(14, 4, strong=True, hard_nll=0.020, macs=100),
            record(14, 5, strong=False, hard_nll=0.001, macs=100),
            # This rank is stable but more expensive, so it cannot displace
            # the cheaper passing rank.
            record(18, 1, strong=True, hard_nll=0.010, macs=120),
            record(18, 2, strong=True, hard_nll=0.011, macs=120),
            record(18, 3, strong=True, hard_nll=0.012, macs=120),
            record(18, 4, strong=True, hard_nll=0.013, macs=120),
            record(18, 5, strong=True, hard_nll=0.014, macs=120),
        ]

        selection = _select_rank_and_seed(
            records,
            required_strong_seeds=4,
        )

        summaries = {
            item["retained_rank"]: item for item in selection["ranks"]
        }
        self.assertFalse(summaries[12]["rank_passed"])
        self.assertEqual(summaries[12]["strong_seed_count"], 3)
        self.assertTrue(summaries[14]["rank_passed"])
        self.assertEqual(summaries[14]["strong_seed_count"], 4)
        self.assertEqual(
            summaries[14]["selected_median_strong_seed"],
            3,
        )
        self.assertEqual(
            selection["selected_runtime_key"],
            (DEFAULT_CANDIDATE.name, 14, 3),
        )

    def test_deployment_parameter_accounting_includes_shared_shell(self) -> None:
        torch.manual_seed(311)
        model = ToyTransformer(
            variable_associative_recall_model_config(V2_TASK_CONFIG)
        ).eval()
        rank = 14
        decoder = torch.linalg.qr(
            torch.randn(model.config.d_model, rank),
            mode="reduced",
        ).Q
        executor = StaticTransformerSpanExecutor(
            DEFAULT_CANDIDATE.executor_config(
                residual_width=model.config.d_model,
                retained_rank=rank,
            ),
            decoder,
        ).eval()

        accounting = _deployment_parameter_accounting(model, executor)

        self.assertEqual(accounting["source_model_parameters"], 27_872)
        self.assertEqual(
            accounting["source_transformer_span_parameters"],
            25_632,
        )
        self.assertEqual(
            accounting["shared_embedding_position_norm_head_parameters"],
            2_240,
        )
        self.assertEqual(
            accounting["compiled_executor_runtime_coefficients"],
            executor.total_runtime_coefficient_count,
        )
        self.assertEqual(
            accounting["compiled_total_deployed_parameters"],
            2_240 + executor.total_runtime_coefficient_count,
        )
        self.assertLess(
            accounting["compiled_total_deployed_parameters"],
            accounting["source_model_parameters"],
        )
        self.assertLess(
            accounting["compiled_to_source_total_parameter_ratio"],
            0.90,
        )

    def test_verifier_rejects_jointly_edited_canonical_protocol_roles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v2-result.pt"
            self._save_minimal_artifact(output)
            payload = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            report = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )

            for container in (payload, report):
                protocol = container["protocol"]
                role_hashes = list(
                    protocol["role_context_hashes"]["basis_fit_a"]
                )
                reserve_hashes = list(
                    protocol["reserve_context_hashes"]
                )
                role_hashes[0], reserve_hashes[0] = (
                    reserve_hashes[0],
                    role_hashes[0],
                )
                protocol["role_context_hashes"]["basis_fit_a"] = tuple(
                    role_hashes
                )
                protocol["reserve_context_hashes"] = tuple(
                    reserve_hashes
                )
                protocol["context_set_sha256"]["basis_fit_a"] = (
                    _canonical_hash_set_digest(tuple(role_hashes))
                )
                protocol["context_set_sha256"]["reserve"] = (
                    _canonical_hash_set_digest(tuple(reserve_hashes))
                )
                protocol["hash_set_digests"]["basis_fit_a"] = (
                    _canonical_hash_set_digest(tuple(role_hashes))
                )
                protocol["hash_set_digests"]["reserve"] = (
                    _canonical_hash_set_digest(tuple(reserve_hashes))
                )

            torch.save(payload, output)
            output.with_suffix(".json").write_text(
                json.dumps(report, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "canonical|frozen|protocol manifest",
            ):
                verify_variable_static_full_span_v2_artifacts(output)

    def test_verifier_rejects_scientific_status_analysis_mismatch(
        self,
    ) -> None:
        mismatches = (
            {
                "calibration_b_evaluated": True,
                "calibration_b_passed": True,
                "calibration_b_confirmatory": True,
            },
            {
                "fresh_validation_evaluated": True,
                "fresh_validation_passed": True,
                "model_level_viable": True,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mismatch in enumerate(mismatches):
                with self.subTest(mismatch=mismatch):
                    output = Path(directory) / f"v2-result-{index}.pt"
                    self._save_minimal_artifact(output)
                    payload = torch.load(
                        output,
                        map_location="cpu",
                        weights_only=True,
                    )
                    report = json.loads(
                        output.with_suffix(".json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for container in (payload, report):
                        container["scientific_status"].update(mismatch)
                    torch.save(payload, output)
                    output.with_suffix(".json").write_text(
                        json.dumps(report, sort_keys=True),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "scientific status|calibration|validation",
                    ):
                        verify_variable_static_full_span_v2_artifacts(
                            output
                        )

    def test_frozen_recipe_rejects_relaxed_promotion_settings(self) -> None:
        frozen = {
            "retained_ranks": DEFAULT_RETAINED_RANKS,
            "candidate": DEFAULT_CANDIDATE,
            "seeds": DEFAULT_SEEDS,
            "required_strong_seeds": DEFAULT_REQUIRED_STRONG_SEEDS,
            "training": DEFAULT_TRAINING,
            "ema_decay": DEFAULT_EMA_DECAY,
            "maximum_relative_work": DEFAULT_MAXIMUM_RELATIVE_WORK,
            "maximum_relative_storage": (
                DEFAULT_MAXIMUM_RELATIVE_STORAGE
            ),
        }
        _validate_frozen_v2_recipe(**frozen)
        relaxed = (
            {"required_strong_seeds": 3},
            {"maximum_relative_work": 0.95},
            {"maximum_relative_storage": 0.95},
            {
                "training": replace(
                    DEFAULT_TRAINING,
                    teacher_kl_weight=3.0,
                )
            },
            {
                "training": replace(
                    DEFAULT_TRAINING,
                    max_steps=4_000,
                )
            },
        )
        for change in relaxed:
            with self.subTest(change=change):
                with self.assertRaisesRegex(
                    ValueError,
                    "frozen|recipe",
                ):
                    _validate_frozen_v2_recipe(
                        **{
                            **frozen,
                            **change,
                        }
                    )

    def test_save_result_refuses_existing_artifact_or_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for existing in ("artifact", "report"):
                with self.subTest(existing=existing):
                    output = root / existing / "v2-result.pt"
                    output.parent.mkdir(parents=True)
                    report = output.with_suffix(".json")
                    target = output if existing == "artifact" else report
                    sentinel = b"do-not-overwrite"
                    target.write_bytes(sentinel)

                    with self.assertRaisesRegex(
                        FileExistsError,
                        "exist|overwrite",
                    ):
                        _save_result(
                            output=output,
                            **self._minimal_artifact_components(),
                        )

                    self.assertEqual(target.read_bytes(), sentinel)
                    other = report if existing == "artifact" else output
                    self.assertFalse(other.exists())

    def test_protected_panel_receipt_is_exclusive_and_hash_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"frozen source checkpoint")
            hashes = self.protocol.calibration_b.semantic_context_hashes
            receipt = _claim_protected_panel(
                checkpoint,
                panel="calibration_b",
                context_hashes=hashes,
                protocol_manifest=self.protocol.manifest(),
            )

            stored = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(stored["panel"], "calibration_b")
            self.assertEqual(
                stored["context_set_sha256"],
                _canonical_hash_set_digest(hashes),
            )
            with self.assertRaises(FileExistsError):
                _claim_protected_panel(
                    checkpoint,
                    panel="calibration_b",
                    context_hashes=hashes,
                    protocol_manifest=self.protocol.manifest(),
                )

    def test_artifact_binds_selected_executor_fingerprint_and_decoder(
        self,
    ) -> None:
        torch.manual_seed(312)
        width = 32
        rank = DEFAULT_RETAINED_RANKS[0]
        vectors = torch.linalg.qr(
            torch.randn(width, width, dtype=torch.float64)
        ).Q
        basis = FisherModeBasis(
            activation_name="layer.2.output",
            mean=torch.zeros(width, dtype=torch.float64),
            matrix=torch.eye(width, dtype=torch.float64),
            eigenvalues=torch.ones(width, dtype=torch.float64),
            vectors=vectors,
            observations=64,
            sequences=32,
        )
        scale = torch.linspace(0.5, 2.0, rank, dtype=torch.float32)
        candidate = DEFAULT_CANDIDATE
        decoder = (
            basis.vectors[:, :rank].to(torch.float32)
            * scale.unsqueeze(0)
        )
        executor = StaticTransformerSpanExecutor(
            candidate.executor_config(
                residual_width=width,
                retained_rank=rank,
            ),
            decoder,
        ).eval()
        candidate_record = asdict(candidate)
        selected_run = {
            "candidate": candidate_record,
            "retained_rank": rank,
            "seed": DEFAULT_SEEDS[0],
            "executor_fingerprint": executor.execution_fingerprint(),
            "runtime_stored_coefficient_count": (
                executor.total_runtime_coefficient_count
            ),
        }
        analysis = {
            "selection_a": {
                "runs": [selected_run],
                "ranks": [],
                "selected": {
                    "candidate": candidate_record,
                    "retained_rank": rank,
                    "selected_median_strong_seed": DEFAULT_SEEDS[0],
                },
            },
            "calibration_b": None,
            "fresh_validation": None,
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v2-result.pt"
            _save_result(
                output=output,
                source=self._canonical_source(),
                hypothesis={"artifact_sha256": "b" * 64},
                protocol=self._frozen_protocol_manifest(),
                basis=basis,
                coordinate_scale=scale,
                executor_artifact=executor.artifact_state_dict(),
                analysis=analysis,
                scientific_status={"executor_test_evaluated": False},
            )
            report = verify_variable_static_full_span_v2_artifacts(output)
            original_payload = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            original_report = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )
            restored = StaticTransformerSpanExecutor.from_artifact_state_dict(
                original_payload["executor"]
            )

            self.assertEqual(
                restored.execution_fingerprint(),
                executor.execution_fingerprint(),
            )
            torch.testing.assert_close(
                restored.decoder,
                decoder,
                rtol=0.0,
                atol=0.0,
            )
            self.assertFalse(report["contains_source_model_weights"])
            self.assertTrue(report["contains_compiled_executor_weights"])

            fingerprint_payload = copy.deepcopy(original_payload)
            fingerprint_report = copy.deepcopy(original_report)
            bad_fingerprint = "0" * 64
            fingerprint_payload["analysis"]["selection_a"]["runs"][0][
                "executor_fingerprint"
            ] = bad_fingerprint
            fingerprint_report["analysis"]["selection_a"]["runs"][0][
                "executor_fingerprint"
            ] = bad_fingerprint
            torch.save(fingerprint_payload, output)
            output.with_suffix(".json").write_text(
                json.dumps(fingerprint_report, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint binding"):
                verify_variable_static_full_span_v2_artifacts(output)

            changed_decoder = decoder.clone()
            changed_decoder[0, 0] += 0.25
            changed_executor = StaticTransformerSpanExecutor(
                candidate.executor_config(
                    residual_width=width,
                    retained_rank=rank,
                ),
                changed_decoder,
            ).eval()
            decoder_payload = copy.deepcopy(original_payload)
            decoder_report = copy.deepcopy(original_report)
            decoder_payload["executor"] = (
                changed_executor.artifact_state_dict()
            )
            for container in (decoder_payload, decoder_report):
                container["analysis"]["selection_a"]["runs"][0][
                    "executor_fingerprint"
                ] = changed_executor.execution_fingerprint()
            torch.save(decoder_payload, output)
            output.with_suffix(".json").write_text(
                json.dumps(decoder_report, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaises((AssertionError, ValueError)):
                verify_variable_static_full_span_v2_artifacts(output)


if __name__ == "__main__":
    unittest.main()
