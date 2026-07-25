import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_gated_executor_experiment import (
    run_gemma3_gated_executor,
)
from fisher_graph.gemma3_projection_ladder_experiment import (
    DEFAULT_RANKS,
    ProjectionCandidate,
    _lock_candidate,
    _scientific_payload_sha256,
    build_parser,
    load_gemma3_projection_ladder_artifact,
    run_gemma3_projection_ladder,
)
from fisher_graph.gemma3_weighted_jacobian_experiment import (
    run_gemma3_weighted_jacobian,
)
from fisher_graph.model import ToyTransformer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"
    name_or_path = "projection-ladder-test-tokenizer"
    vocab_size = 31
    model_max_length = 8
    bos_token_id = 2
    eos_token_id = 1
    init_kwargs: dict[str, object] = {}

    def __init__(self) -> None:
        self.prompt_calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        prompts: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        self.prompt_calls.append(tuple(prompts))
        rows = [
            [2, 4 + index, 9 + index, 1]
            for index in range(len(prompts))
        ]
        input_ids = torch.tensor(rows)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(
                input_ids,
                dtype=torch.bool,
            ),
        }


def prompt_payload(prefix: str) -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": f"{prefix}_synthetic_test_only",
        "calibration_a": [f"{prefix} a one", f"{prefix} a two"],
        "calibration_b": [f"{prefix} b one", f"{prefix} b two"],
        "validation": [f"{prefix} v one", f"{prefix} v two"],
        "test": [f"{prefix} t one", f"{prefix} t two"],
    }


def make_model() -> ToyTransformer:
    model = ToyTransformer(
        TransformerConfig(
            vocab_size=31,
            max_sequence_length=8,
            d_model=8,
            n_heads=2,
            n_layers=3,
            d_ff=12,
            dropout=0.0,
        )
    ).eval()
    model.requires_grad_(False)
    return model


class Gemma3ProjectionLadderExperimentTests(unittest.TestCase):
    def test_fresh_b_only_ladder_lock_validation_and_strict_artifact(
        self,
    ) -> None:
        torch.manual_seed(5519)
        model = make_model()
        parameter_versions = {
            name: int(parameter._version)
            for name, parameter in model.named_parameters()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_prompts = root / "source.json"
            source_prompts.write_text(
                json.dumps(prompt_payload("source")),
                encoding="utf-8",
            )
            weighted_artifact = root / "weighted.pt"
            with patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "load_gemma3",
                return_value=(RecordingTokenizer(), model),
            ), patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                run_gemma3_weighted_jacobian(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=source_prompts,
                    start_layer=0,
                    end_layer=1,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(4, 8),
                    generalized_regularization_pairs=((1e-4, 1e-4),),
                    sketch_rows=9,
                    selection_nll_atol=1e6,
                    selection_top1_min=0.0,
                    full_rank_nll_atol=1e-4,
                    jacobian_max_sequences=0,
                    device_name="cpu",
                    dtype="float32",
                    output=weighted_artifact,
                )

            gated_prompts = root / "gated.json"
            gated_prompts.write_text(
                json.dumps(prompt_payload("gated")),
                encoding="utf-8",
            )
            gated_artifact = root / "gated.pt"
            with patch(
                "fisher_graph.gemma3_gated_executor_experiment.load_gemma3",
                return_value=(RecordingTokenizer(), model),
            ), patch(
                "fisher_graph.gemma3_gated_executor_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                run_gemma3_gated_executor(
                    weighted_artifact_path=weighted_artifact,
                    cache_dir=root / "external-cache",
                    prompt_splits_path=gated_prompts,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(4, 6),
                    expert_counts=(1,),
                    expert_ranks=(2,),
                    router_widths=(2,),
                    max_positive_lags=(2,),
                    fit_steps=2,
                    learning_rate=1e-3,
                    max_retained_fraction=1.0,
                    max_stored_coefficient_ratio=100.0,
                    max_analytic_mac_ratio=100.0,
                    max_block_delta_nrmse=0.0,
                    min_block_delta_cosine=1.0,
                    selection_nll_atol=0.0,
                    selection_top1_min=1.0,
                    identity_nll_atol=1e-4,
                    device_name="cpu",
                    dtype="float32",
                    output=gated_artifact,
                )

            fresh_prompts = root / "fresh.json"
            fresh = prompt_payload("fresh")
            fresh_prompts.write_text(
                json.dumps(fresh),
                encoding="utf-8",
            )
            with patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "_codec_state_sha256",
                return_value="0" * 64,
            ), patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "load_gemma3",
            ) as load_model:
                with self.assertRaisesRegex(
                    ValueError,
                    "output codecs disagree",
                ):
                    run_gemma3_projection_ladder(
                        weighted_artifact_path=weighted_artifact,
                        gated_artifact_path=gated_artifact,
                        cache_dir=root / "external-cache",
                        prompt_splits_path=fresh_prompts,
                        max_length=8,
                        tokenization_batch_size=2,
                        ranks=(4, 6, 7, 8),
                        device_name="cpu",
                        dtype="float32",
                        output=root / "codec-mismatch.pt",
                    )
                load_model.assert_not_called()

            overlap_prompts = root / "overlap.json"
            overlap = prompt_payload("overlap")
            overlap["calibration_a"][0] = "source a one"
            overlap_prompts.write_text(
                json.dumps(overlap),
                encoding="utf-8",
            )
            with patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "load_gemma3",
            ) as load_model:
                with self.assertRaisesRegex(
                    ValueError,
                    "overlap weighted source prompts",
                ):
                    run_gemma3_projection_ladder(
                        weighted_artifact_path=weighted_artifact,
                        gated_artifact_path=gated_artifact,
                        cache_dir=root / "external-cache",
                        prompt_splits_path=overlap_prompts,
                        max_length=8,
                        tokenization_batch_size=2,
                        ranks=(4, 6, 7, 8),
                        device_name="cpu",
                        dtype="float32",
                        output=root / "overlap.pt",
                    )
                load_model.assert_not_called()

            failed_identity_tokenizer = RecordingTokenizer()
            with patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "load_gemma3",
                return_value=(failed_identity_tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ), patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "_full_width_controls_passed",
                return_value=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "full-width projection identity failed",
                ):
                    run_gemma3_projection_ladder(
                        weighted_artifact_path=weighted_artifact,
                        gated_artifact_path=gated_artifact,
                        cache_dir=root / "external-cache",
                        prompt_splits_path=fresh_prompts,
                        max_length=8,
                        tokenization_batch_size=2,
                        ranks=(4, 6, 7, 8),
                        device_name="cpu",
                        dtype="float32",
                        output=root / "failed-identity.pt",
                    )
            self.assertEqual(
                failed_identity_tokenizer.prompt_calls,
                [("fresh b one", "fresh b two")],
            )

            output = root / "projection.pt"
            tokenizer = RecordingTokenizer()
            with patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_projection_ladder_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_projection_ladder(
                    weighted_artifact_path=weighted_artifact,
                    gated_artifact_path=gated_artifact,
                    cache_dir=root / "external-cache",
                    prompt_splits_path=fresh_prompts,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(4, 6, 7, 8),
                    selection_nll_atol=1e6,
                    selection_top1_min=0.0,
                    identity_nll_atol=1e-4,
                    max_meaningful_retained_fraction=0.75,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )

            self.assertEqual(
                tokenizer.prompt_calls,
                [
                    ("fresh b one", "fresh b two"),
                    ("fresh v one", "fresh v two"),
                ],
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            status = report["scientific_status"]
            self.assertFalse(status["calibration_a_evaluated"])
            self.assertFalse(status["test_evaluated"])
            self.assertEqual(
                status["locked_validation_interventions_per_batch"],
                1,
            )
            self.assertTrue(status["fidelity_viable_reduced_rank"])
            self.assertTrue(status["meaningful_rank_compression"])

            selection = report["analysis"]["selection"]
            self.assertEqual(
                tuple(selection["candidate_behavior"]),
                (
                    "rank_4.target_ls_projection",
                    "rank_6.target_ls_projection",
                    "rank_7.target_ls_projection",
                    "rank_8.target_ls_projection",
                ),
            )
            self.assertEqual(
                selection["lock"]["fidelity_lock_rank"],
                4,
            )
            self.assertFalse(selection["lock"]["selection_failed"])
            self.assertTrue(
                selection["full_width_identity"]["passed"]
            )
            nested = selection["nested_projection_error_control"]
            self.assertTrue(nested["passed"])
            self.assertEqual(
                list(nested["values"]),
                sorted(nested["values"], reverse=True),
            )
            validation = report["analysis"]["validation"]
            self.assertEqual(
                validation["locked_candidate"]["retained_rank"],
                4,
            )
            self.assertTrue(validation["behavior_fidelity_passed"])
            self.assertNotIn(
                "test",
                report["protocol"]["tokenized_splits"],
            )
            self.assertNotIn(
                "calibration_a",
                report["protocol"]["tokenized_splits"],
            )
            self.assertEqual(
                report["predecessors"]["prompt_disjointness"][
                    "weighted_overlap_count"
                ],
                0,
            )
            self.assertEqual(
                report["predecessors"]["prompt_disjointness"][
                    "gated_overlap_count"
                ],
                0,
            )

            raw = torch.load(output, map_location="cpu", weights_only=True)
            self.assertFalse(raw["contains_model_weights"])
            self.assertFalse(raw["contains_prompt_text"])
            self.assertFalse(raw["contains_tokenizer_state"])
            loaded = load_gemma3_projection_ladder_artifact(output)
            self.assertEqual(
                loaded["locked_candidate"]["retained_rank"],
                4,
            )
            serialized = repr(raw)
            for split in fresh.values():
                if isinstance(split, list):
                    for prompt in split:
                        self.assertNotIn(prompt, serialized)

            tampered = output.with_name("tampered.pt")
            changed = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            changed["selection"]["lock"]["fidelity_lock_rank"] = 6
            payload = {
                key: value
                for key, value in changed.items()
                if key
                not in {
                    "scientific_payload_sha256",
                    "report_sha256",
                }
            }
            changed["scientific_payload_sha256"] = (
                _scientific_payload_sha256(payload)
            )
            torch.save(changed, tampered)
            shutil.copyfile(
                output.with_suffix(".json"),
                tampered.with_suffix(".json"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "selection lock",
            ):
                load_gemma3_projection_ladder_artifact(tampered)

            aggregate_tampered = output.with_name(
                "aggregate-tampered.pt"
            )
            changed = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            changed["selection"]["candidate_behavior"][
                "rank_4.target_ls_projection"
            ]["delta_nll_per_token"] += 1.0
            payload = {
                key: value
                for key, value in changed.items()
                if key
                not in {
                    "scientific_payload_sha256",
                    "report_sha256",
                }
            }
            changed["scientific_payload_sha256"] = (
                _scientific_payload_sha256(payload)
            )
            torch.save(changed, aggregate_tampered)
            shutil.copyfile(
                output.with_suffix(".json"),
                aggregate_tampered.with_suffix(".json"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "behavior aggregate",
            ):
                load_gemma3_projection_ladder_artifact(
                    aggregate_tampered
                )

            for name, parameter in model.named_parameters():
                self.assertEqual(
                    int(parameter._version),
                    parameter_versions[name],
                )
                self.assertFalse(parameter.requires_grad)

    def test_lock_falls_back_to_identity_when_no_reduced_rank_passes(
        self,
    ) -> None:
        candidates = (
            ProjectionCandidate(4, 8),
            ProjectionCandidate(6, 8),
            ProjectionCandidate(8, 8),
        )
        ledger = [
            {
                "behavior_fidelity_passed": False,
                "direct_diagnostic": {"mse": mse},
            }
            for mse in (0.01, 0.02, 0.0)
        ]
        locked, lock = _lock_candidate(candidates, ledger)
        self.assertEqual(locked.retained_rank, 8)
        self.assertTrue(lock["selection_failed"])
        self.assertFalse(lock["reduced_candidate_found"])
        self.assertEqual(
            lock["reason"],
            "no_reduced_rank_passed_full_width_identity_fallback",
        )

    def test_lock_uses_smallest_behavior_pass_not_direct_metrics(
        self,
    ) -> None:
        candidates = (
            ProjectionCandidate(4, 8),
            ProjectionCandidate(6, 8),
            ProjectionCandidate(7, 8),
            ProjectionCandidate(8, 8),
        )
        ledger = [
            {
                "behavior_fidelity_passed": passed,
                "direct_diagnostic": {"mse": mse},
            }
            for passed, mse in (
                (False, 0.0),
                (True, 100.0),
                (True, 0.001),
                (True, 0.0),
            )
        ]
        locked, lock = _lock_candidate(candidates, ledger)
        self.assertEqual(locked.retained_rank, 6)
        self.assertFalse(lock["selection_failed"])
        self.assertEqual(
            lock["reason"],
            "smallest_reduced_rank_passing_behavioral_gates",
        )

    def test_cli_defaults_are_the_preregistered_width_640_ladder(
        self,
    ) -> None:
        parsed = build_parser().parse_args(
            [
                "--weighted-artifact",
                "weighted.pt",
                "--gated-artifact",
                "gated.pt",
            ]
        )
        self.assertEqual(tuple(parsed.ranks), DEFAULT_RANKS)
        self.assertEqual(parsed.selection_nll_atol, 0.05)
        self.assertEqual(parsed.selection_top1_min, 0.95)
        self.assertEqual(parsed.device, "cpu")
        self.assertEqual(
            parsed.max_meaningful_retained_fraction,
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
