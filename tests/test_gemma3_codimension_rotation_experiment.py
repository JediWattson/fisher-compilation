import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_codimension_rotation_experiment import (
    _PROMPT_STATUS,
    _candidate_schedule,
    _lock_candidate,
    _scientific_payload_sha256,
    load_gemma3_codimension_rotation_artifact,
    run_gemma3_codimension_rotation,
)
from fisher_graph.gemma3_gated_executor_experiment import (
    run_gemma3_gated_executor,
)
from fisher_graph.gemma3_projection_ladder_experiment import (
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
    name_or_path = "codimension-rotation-test-tokenizer"
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


def prompt_payload(
    prefix: str,
    *,
    scientific_status: str | None = None,
) -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": (
            f"{prefix}_synthetic_test_only"
            if scientific_status is None
            else scientific_status
        ),
        "calibration_a": [f"{prefix} a one", f"{prefix} a two"],
        "calibration_b": [f"{prefix} b one", f"{prefix} b two"],
        "validation": [f"{prefix} v one", f"{prefix} v two"],
        "test": [f"{prefix} t one", f"{prefix} t two"],
    }


def write_prompts(
    root: Path,
    prefix: str,
    *,
    scientific_status: str | None = None,
) -> tuple[Path, dict[str, object]]:
    payload = prompt_payload(
        prefix,
        scientific_status=scientific_status,
    )
    path = root / f"{prefix}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


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


class Gemma3CodimensionRotationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(6601)
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.cache = cls.root / "external-cache"
        cls.model = make_model()
        cls.parameter_versions = {
            name: int(parameter._version)
            for name, parameter in cls.model.named_parameters()
        }

        weighted_prompts, cls.weighted_prompt_payload = write_prompts(
            cls.root,
            "rotation-weighted",
        )
        cls.weighted_artifact = cls.root / "weighted.pt"
        with patch(
            "fisher_graph.gemma3_weighted_jacobian_experiment."
            "load_gemma3",
            return_value=(RecordingTokenizer(), cls.model),
        ), patch(
            "fisher_graph.gemma3_weighted_jacobian_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ):
            run_gemma3_weighted_jacobian(
                cache_dir=cls.cache,
                prompt_splits_path=weighted_prompts,
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
                output=cls.weighted_artifact,
            )

        gated_prompts, cls.gated_prompt_payload = write_prompts(
            cls.root,
            "rotation-gated",
        )
        cls.gated_artifact = cls.root / "gated.pt"
        with patch(
            "fisher_graph.gemma3_gated_executor_experiment.load_gemma3",
            return_value=(RecordingTokenizer(), cls.model),
        ), patch(
            "fisher_graph.gemma3_gated_executor_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ):
            run_gemma3_gated_executor(
                weighted_artifact_path=cls.weighted_artifact,
                cache_dir=cls.cache,
                prompt_splits_path=gated_prompts,
                max_length=8,
                tokenization_batch_size=2,
                ranks=(4, 6),
                expert_counts=(1,),
                expert_ranks=(2,),
                router_widths=(2,),
                max_positive_lags=(2,),
                fit_steps=1,
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
                output=cls.gated_artifact,
            )

        projection_prompts, cls.projection_prompt_payload = write_prompts(
            cls.root,
            "rotation-projection",
        )
        cls.projection_artifact = cls.root / "projection.pt"
        with patch(
            "fisher_graph.gemma3_projection_ladder_experiment."
            "load_gemma3",
            return_value=(RecordingTokenizer(), cls.model),
        ), patch(
            "fisher_graph.gemma3_projection_ladder_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ):
            run_gemma3_projection_ladder(
                weighted_artifact_path=cls.weighted_artifact,
                gated_artifact_path=cls.gated_artifact,
                cache_dir=cls.cache,
                prompt_splits_path=projection_prompts,
                max_length=8,
                tokenization_batch_size=2,
                ranks=(4, 6, 7, 8),
                selection_nll_atol=0.0,
                selection_top1_min=1.0,
                identity_nll_atol=1e-4,
                device_name="cpu",
                dtype="float32",
                output=cls.projection_artifact,
            )
        projection = load_gemma3_projection_ladder_artifact(
            cls.projection_artifact
        )
        if (
            projection["selection"]["lock"]["selection_failed"]
            is not True
        ):
            raise AssertionError(
                "synthetic predecessor must be a negative rank ladder"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _run_rotation(
        self,
        *,
        prompts: Path,
        tokenizer: RecordingTokenizer,
        output: Path,
        selection_nll_atol: float,
        selection_top1_min: float,
        minimum_split_half_alignment: float = 0.0,
        minimum_relative_eigengap: float = 0.0,
    ) -> dict[str, object]:
        with patch(
            "fisher_graph.gemma3_codimension_rotation_experiment."
            "load_gemma3",
            return_value=(tokenizer, self.model),
        ), patch(
            "fisher_graph.gemma3_codimension_rotation_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ):
            return run_gemma3_codimension_rotation(
                projection_artifact_path=self.projection_artifact,
                cache_dir=self.cache,
                prompt_splits_path=prompts,
                max_length=8,
                tokenization_batch_size=2,
                tail_width=4,
                selection_nll_atol=selection_nll_atol,
                selection_top1_min=selection_top1_min,
                identity_nll_atol=1e-4,
                minimum_split_half_alignment=(
                    minimum_split_half_alignment
                ),
                minimum_relative_eigengap=(
                    minimum_relative_eigengap
                ),
                device_name="cpu",
                dtype="float32",
                output=output,
            )

    def test_stable_success_strict_load_and_resigned_tamper(
        self,
    ) -> None:
        prompts, fresh = write_prompts(
            self.root,
            "rotation-success",
            scientific_status=_PROMPT_STATUS,
        )
        tokenizer = RecordingTokenizer()
        output = self.root / "rotation-success.pt"
        report = self._run_rotation(
            prompts=prompts,
            tokenizer=tokenizer,
            output=output,
            selection_nll_atol=1e6,
            selection_top1_min=0.0,
        )

        self.assertEqual(
            tokenizer.prompt_calls,
            [
                (
                    "rotation-success a one",
                    "rotation-success a two",
                ),
                (
                    "rotation-success b one",
                    "rotation-success b two",
                ),
                (
                    "rotation-success v one",
                    "rotation-success v two",
                ),
            ],
        )
        status = report["scientific_status"]
        self.assertTrue(status["sensitivity_fit_stable"])
        self.assertFalse(status["selection_failed"])
        self.assertTrue(status["rank_639_fidelity_viable"])
        self.assertFalse(status["test_evaluated"])
        self.assertFalse(status["inference_executor"])
        self.assertFalse(status["compression_claim"])
        self.assertFalse(status["meaningful_rank_compression"])
        self.assertEqual(
            status["locked_validation_interventions_per_batch"],
            1,
        )
        selection = report["analysis"]["selection"]
        validation = report["analysis"]["validation"]
        self.assertEqual(
            selection["lock"]["locked_normal_source"],
            "calibration_a_balanced_tail_rotation",
        )
        self.assertEqual(
            validation["locked_candidate"]["retained_rank"],
            7,
        )
        self.assertTrue(validation["behavior_fidelity_passed"])
        self.assertNotIn("test", report["protocol"]["tokenized_splits"])

        raw = torch.load(output, map_location="cpu", weights_only=True)
        self.assertFalse(raw["contains_model_weights"])
        self.assertFalse(raw["contains_prompt_text"])
        self.assertFalse(raw["contains_tokenizer_state"])
        loaded = load_gemma3_codimension_rotation_artifact(output)
        self.assertEqual(
            loaded["locked_candidate"]["retained_rank"],
            7,
        )
        self.assertEqual(
            loaded["metadata"]["scientific_payload_sha256"],
            raw["scientific_payload_sha256"],
        )
        serialized = repr(raw)
        for prompt_set in (
            self.weighted_prompt_payload,
            self.gated_prompt_payload,
            self.projection_prompt_payload,
            fresh,
        ):
            for split in prompt_set.values():
                if isinstance(split, list):
                    for prompt in split:
                        self.assertNotIn(prompt, serialized)

        tampered = self.root / "rotation-tampered.pt"
        changed = torch.load(
            output,
            map_location="cpu",
            weights_only=True,
        )
        changed["selection"]["lock"]["locked_candidate_id"] = (
            "rank_8.identity"
        )
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
            load_gemma3_codimension_rotation_artifact(tampered)

        invalid_behavior = self.root / "rotation-invalid-behavior.pt"
        changed = torch.load(
            output,
            map_location="cpu",
            weights_only=True,
        )
        candidate_id = (
            "rank_7.calibration_a_balanced_tail_rotation"
        )
        behavior_example = changed["selection"][
            "candidate_behavior"
        ][candidate_id]["examples"][0]
        behavior_example["top1_matches"] = (
            behavior_example["supervised_tokens"] + 1
        )
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
        torch.save(changed, invalid_behavior)
        shutil.copyfile(
            output.with_suffix(".json"),
            invalid_behavior.with_suffix(".json"),
        )
        with self.assertRaisesRegex(
            ValueError,
            "behavior example identity",
        ):
            load_gemma3_codimension_rotation_artifact(
                invalid_behavior
            )

        for name, parameter in self.model.named_parameters():
            self.assertEqual(
                int(parameter._version),
                self.parameter_versions[name],
            )
            self.assertFalse(parameter.requires_grad)

    def test_selection_failure_stops_before_validation_tokenization(
        self,
    ) -> None:
        prompts, _ = write_prompts(
            self.root,
            "rotation-failure",
            scientific_status=_PROMPT_STATUS,
        )
        tokenizer = RecordingTokenizer()
        output = self.root / "rotation-failure.pt"
        report = self._run_rotation(
            prompts=prompts,
            tokenizer=tokenizer,
            output=output,
            selection_nll_atol=0.0,
            selection_top1_min=1.0,
        )
        self.assertEqual(
            tokenizer.prompt_calls,
            [
                (
                    "rotation-failure a one",
                    "rotation-failure a two",
                ),
                (
                    "rotation-failure b one",
                    "rotation-failure b two",
                ),
            ],
        )
        self.assertTrue(output.is_file())
        self.assertTrue(report["scientific_status"]["selection_failed"])
        self.assertFalse(
            report["analysis"]["validation"]["evaluated"]
        )
        self.assertEqual(
            tuple(report["protocol"]["tokenized_splits"]),
            ("calibration_a", "calibration_b"),
        )
        loaded = load_gemma3_codimension_rotation_artifact(output)
        self.assertTrue(
            loaded["selection"]["lock"]["selection_failed"]
        )
        self.assertFalse(loaded["validation"]["evaluated"])

    def test_unstable_calibration_a_stops_before_selection(self) -> None:
        prompts, _ = write_prompts(
            self.root,
            "rotation-unstable-a",
            scientific_status=_PROMPT_STATUS,
        )
        tokenizer = RecordingTokenizer()
        output = self.root / "rotation-unstable-a.pt"
        with self.assertRaisesRegex(
            RuntimeError,
            "balanced tail direction is not identifiable",
        ):
            self._run_rotation(
                prompts=prompts,
                tokenizer=tokenizer,
                output=output,
                selection_nll_atol=1e6,
                selection_top1_min=0.0,
                minimum_relative_eigengap=1e9,
            )
        self.assertEqual(
            tokenizer.prompt_calls,
            [
                (
                    "rotation-unstable-a a one",
                    "rotation-unstable-a a two",
                )
            ],
        )
        self.assertFalse(output.exists())

    def test_prompt_overlap_fails_before_model_load(self) -> None:
        prompts, payload = write_prompts(
            self.root,
            "rotation-overlap",
            scientific_status=_PROMPT_STATUS,
        )
        payload["calibration_a"][0] = "rotation-projection a one"
        prompts.write_text(json.dumps(payload), encoding="utf-8")
        with patch(
            "fisher_graph.gemma3_codimension_rotation_experiment."
            "load_gemma3",
        ) as load_model:
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_gemma3_codimension_rotation(
                    projection_artifact_path=self.projection_artifact,
                    cache_dir=self.cache,
                    prompt_splits_path=prompts,
                    max_length=8,
                    tokenization_batch_size=2,
                    tail_width=4,
                    selection_nll_atol=1e6,
                    selection_top1_min=0.0,
                    identity_nll_atol=1e-4,
                    minimum_split_half_alignment=0.0,
                    minimum_relative_eigengap=0.0,
                    device_name="cpu",
                    dtype="float32",
                    output=self.root / "rotation-overlap.pt",
                )
            load_model.assert_not_called()

    def test_selection_policy_prefers_rotation_then_codec_fallback(
        self,
    ) -> None:
        candidates = _candidate_schedule(8)

        def ledger(*passing: bool) -> list[dict[str, object]]:
            return [
                {
                    "behavior_fidelity_passed": passed,
                    "candidate": candidate.metadata(),
                }
                for candidate, passed in zip(
                    candidates,
                    passing,
                    strict=True,
                )
            ]

        both_pass_locked, both_pass = _lock_candidate(
            candidates,
            ledger(True, True, True),
        )
        self.assertEqual(
            both_pass_locked.normal_source,
            "calibration_a_balanced_tail_rotation",
        )
        self.assertFalse(both_pass["selection_failed"])

        codec_locked, codec_fallback = _lock_candidate(
            candidates,
            ledger(False, True, True),
        )
        self.assertEqual(
            codec_locked.normal_source,
            "source_codec_prefix",
        )
        self.assertFalse(codec_fallback["selection_failed"])

        identity_locked, failed = _lock_candidate(
            candidates,
            ledger(False, False, True),
        )
        self.assertEqual(identity_locked.normal_source, "identity")
        self.assertTrue(failed["selection_failed"])


if __name__ == "__main__":
    unittest.main()
