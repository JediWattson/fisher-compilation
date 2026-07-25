import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_weighted_jacobian_experiment import (
    _candidate_schedule,
    _full_rank_candidate,
    _lock_candidate,
    _modal_result_from_metadata,
    _scientific_payload_sha256,
    _validation_candidates,
    _variants,
    build_parser,
    default_gemma3_weighted_jacobian_output,
    load_gemma3_weighted_jacobian_artifact,
    run_gemma3_weighted_jacobian,
)
from fisher_graph.model import ToyTransformer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"
    name_or_path = "weighted-jacobian-test-tokenizer"
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


def prompt_payload() -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": "synthetic_test_only",
        "calibration_a": ["a one", "a two"],
        "calibration_b": ["b one", "b two"],
        "validation": ["v one", "v two"],
        "test": ["t one", "t two"],
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


class Gemma3WeightedJacobianExperimentTests(unittest.TestCase):
    def test_split_safe_selection_jvp_factor_and_strict_artifact(
        self,
    ) -> None:
        torch.manual_seed(9917)
        model = make_model()
        tokenizer = RecordingTokenizer()
        parameter_objects = dict(model.named_parameters())
        parameter_versions = {
            name: int(parameter._version)
            for name, parameter in parameter_objects.items()
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "splits.json"
            prompt_path.write_text(
                json.dumps(prompt_payload()),
                encoding="utf-8",
            )
            output = root / "analysis" / "weighted.pt"
            with patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_weighted_jacobian(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=prompt_path,
                    start_layer=0,
                    end_layer=1,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(8, 4),
                    generalized_regularization_pairs=(
                        (1e-4, 1e-4),
                    ),
                    sketch_rows=9,
                    selection_nll_atol=1e6,
                    selection_top1_min=0.0,
                    full_rank_nll_atol=1e-5,
                    jacobian_max_sequences=1,
                    jacobian_modes=2,
                    jacobian_max_lag=1,
                    jacobian_factor_rank=1,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )

            selection = report["analysis"]["calibration_b"]["selection"]
            self.assertEqual(
                selection["locked_candidate"]["candidate_id"],
                "native_fisher.joint.rank_4",
            )
            self.assertEqual(
                selection["reason"],
                "lowest_rank_passing_gates",
            )
            calibration_b_analysis = report["analysis"]["calibration_b"]
            family_identities = calibration_b_analysis[
                "family_full_rank_identities"
            ]
            self.assertEqual(
                tuple(family_identities),
                (
                    "native_fisher",
                    "variance_weighted_fisher",
                    "generalized_fisher.reg_00",
                ),
            )
            self.assertTrue(
                all(
                    control["passed"]
                    for control in family_identities.values()
                )
            )
            self.assertTrue(
                calibration_b_analysis[
                    "locked_family_full_rank_identity"
                ]["passed"]
            )
            self.assertTrue(
                calibration_b_analysis[
                    "native_full_rank_identity"
                ]["passed"]
            )
            calibration_b_conditions = {
                row["condition"]["name"]
                for row in calibration_b_analysis[
                    "candidate_evaluation"
                ]["conditions"]
            }
            self.assertTrue(
                {
                    "native_fisher.joint.rank_8",
                    "variance_weighted_fisher.joint.rank_8",
                    "generalized_fisher.reg_00.joint.rank_8",
                }.issubset(calibration_b_conditions)
            )
            self.assertEqual(
                len(
                    report["analysis"]["validation"][
                        "locked_evaluation"
                    ]["conditions"]
                ),
                2,
            )
            self.assertTrue(
                report["analysis"]["validation"][
                    "locked_family_full_rank_identity"
                ]["passed"]
            )
            self.assertTrue(
                report["analysis"]["validation"][
                    "native_full_rank_identity"
                ]["passed"]
            )
            jacobian = report["analysis"]["jacobian"]
            self.assertTrue(jacobian["enabled"])
            self.assertEqual(jacobian["statistics"]["input_modes"], 2)
            self.assertEqual(
                jacobian["merged_factor"]["retained_ranks"],
                (1, 1),
            )
            self.assertFalse(
                jacobian["merged_factor"]["compression_claim"]
            )
            self.assertEqual(
                jacobian["merged_factor"]["dense_ratio_denominator"],
                "unshared_dense_causal_operator_not_lag_shared_storage",
            )

            self.assertEqual(
                tokenizer.prompt_calls,
                [
                    ("a one", "a two"),
                    ("b one", "b two"),
                    ("a one",),
                    ("v one", "v two"),
                ],
            )
            self.assertNotIn(
                "t one",
                [
                    prompt
                    for call in tokenizer.prompt_calls
                    for prompt in call
                ],
            )
            self.assertFalse(model.training)
            for name, parameter in model.named_parameters():
                self.assertIs(parameter, parameter_objects[name])
                self.assertEqual(
                    int(parameter._version),
                    parameter_versions[name],
                )
                self.assertFalse(parameter.requires_grad)

            raw = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(raw["format_version"], 2)
            self.assertFalse(raw["contains_model_weights"])
            self.assertFalse(raw["contains_prompt_text"])
            self.assertFalse(raw["contains_tokenizer_state"])
            self.assertEqual(
                raw["calibration_b"]["family_full_rank_identities"],
                family_identities,
            )
            self.assertNotIn("model_state_dict", raw)
            self.assertNotIn(
                "test",
                raw["protocol"]["tokenized_splits"],
            )
            serialized = repr(raw)
            for prompt in (
                "a one",
                "a two",
                "b one",
                "b two",
                "v one",
                "v two",
                "t one",
                "t two",
            ):
                self.assertNotIn(prompt, serialized)

            loaded = load_gemma3_weighted_jacobian_artifact(output)
            self.assertEqual(
                loaded["selection"]["locked_candidate"][
                    "candidate_id"
                ],
                "native_fisher.joint.rank_4",
            )
            self.assertIsNotNone(loaded["jacobian"])
            self.assertIsNotNone(loaded["merged_factor"])

            variants = _variants(((1e-4, 1e-4),))
            schedule = _candidate_schedule(
                ranks=(4, 8),
                width=8,
                variants=variants,
                codecs=loaded["codecs"],
                sites=(
                    "layer.0.output",
                    "layer.1.output",
                ),
            )
            generalized_locked = next(
                candidate
                for candidate in schedule
                if candidate.variant.method == "generalized_fisher"
                and candidate.retained_rank == 4
            )
            generalized_full = _full_rank_candidate(
                schedule,
                variant_id=generalized_locked.variant.variant_id,
            )
            native_full = _full_rank_candidate(
                schedule,
                variant_id="native_fisher",
            )
            generalized_validation, _, _ = _validation_candidates(
                locked=generalized_locked,
                locked_family_full_rank=generalized_full,
                native_full_rank=native_full,
            )
            self.assertEqual(
                tuple(
                    candidate.candidate_id
                    for candidate in generalized_validation
                ),
                (
                    "locked_candidate",
                    "locked_family_full_rank_identity",
                    "native_full_rank_identity",
                ),
            )
            native_validation, _, _ = _validation_candidates(
                locked=next(
                    candidate
                    for candidate in schedule
                    if candidate.variant.method == "native_fisher"
                    and candidate.retained_rank == 4
                ),
                locked_family_full_rank=native_full,
                native_full_rank=native_full,
            )
            self.assertEqual(len(native_validation), 2)

            forced_generalized = copy.deepcopy(
                calibration_b_analysis["candidate_evaluation"]
            )
            for condition in forced_generalized["conditions"]:
                name = condition["condition"]["name"]
                if not name.endswith("rank_4"):
                    continue
                aggregate = condition["aggregate"]
                if name.startswith("generalized_fisher.reg_00"):
                    aggregate["delta_nll_per_token"] = 0.0
                    aggregate["top1_agreement_to_baseline"] = 1.0
                else:
                    aggregate["delta_nll_per_token"] = 1.0
                    aggregate["top1_agreement_to_baseline"] = 0.0
            generalized_result = _modal_result_from_metadata(
                forced_generalized
            )
            generalized_selection, _ = _lock_candidate(
                schedule=schedule,
                calibration_b=generalized_result,
                nll_atol=0.05,
                top1_min=0.95,
            )
            self.assertEqual(
                generalized_selection.candidate_id,
                "generalized_fisher.reg_00.joint.rank_4",
            )
            selected_family_full = _full_rank_candidate(
                schedule,
                variant_id=generalized_selection.variant.variant_id,
            )
            generalized_validation, _, _ = _validation_candidates(
                locked=generalized_selection,
                locked_family_full_rank=selected_family_full,
                native_full_rank=native_full,
            )
            self.assertEqual(
                tuple(
                    candidate.candidate_id
                    for candidate in generalized_validation
                ),
                (
                    "locked_candidate",
                    "locked_family_full_rank_identity",
                    "native_full_rank_identity",
                ),
            )

            tampered = output.with_name("tampered.pt")
            tampered_raw = torch.load(
                output,
                map_location="cpu",
                weights_only=True,
            )
            tampered_raw["calibration_b"]["selection"]["reason"] = (
                "native_full_rank_fallback"
            )
            payload = {
                key: value
                for key, value in tampered_raw.items()
                if key
                not in {
                    "scientific_payload_sha256",
                    "report_sha256",
                }
            }
            tampered_raw["scientific_payload_sha256"] = (
                _scientific_payload_sha256(payload)
            )
            torch.save(tampered_raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "locked selection",
            ):
                load_gemma3_weighted_jacobian_artifact(tampered)

            def write_resigned_tamper(
                *,
                name: str,
                mutate: object,
            ) -> Path:
                value = copy.deepcopy(raw)
                mutate(value)  # type: ignore[operator]
                payload = {
                    key: item
                    for key, item in value.items()
                    if key
                    not in {
                        "scientific_payload_sha256",
                        "report_sha256",
                    }
                }
                value["scientific_payload_sha256"] = (
                    _scientific_payload_sha256(payload)
                )
                path = output.with_name(name)
                torch.save(value, path)
                return path

            guard_state = write_resigned_tamper(
                name="guard-source-state.pt",
                mutate=lambda value: value["protocol"][
                    "model_state_guard"
                ].__setitem__(
                    "model_state_dict",
                    {"forbidden": [1.0, 2.0]},
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "model-state guard",
            ):
                load_gemma3_weighted_jacobian_artifact(guard_state)

            library_prompt = write_resigned_tamper(
                name="library-prompt-state.pt",
                mutate=lambda value: value["protocol"][
                    "library_versions"
                ].__setitem__("prompt_text", "reserved prompt"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "library provenance",
            ):
                load_gemma3_weighted_jacobian_artifact(library_prompt)

            tokenizer_state = write_resigned_tamper(
                name="tokenizer-source-state.pt",
                mutate=lambda value: value["protocol"][
                    "tokenizer"
                ].__setitem__(
                    "tokenizer_state",
                    {"vocabulary": ["forbidden"]},
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "tokenizer provenance",
            ):
                load_gemma3_weighted_jacobian_artifact(tokenizer_state)

            family_identity = write_resigned_tamper(
                name="family-identity.pt",
                mutate=lambda value: value["calibration_b"][
                    "family_full_rank_identities"
                ]["variance_weighted_fisher"].__setitem__(
                    "passed",
                    False,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "identity gate",
            ):
                load_gemma3_weighted_jacobian_artifact(family_identity)

    def test_deterministic_full_rank_fallback_and_cli(self) -> None:
        torch.manual_seed(2027)
        model = make_model()
        tokenizer = RecordingTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "splits.json"
            prompt_path.write_text(
                json.dumps(prompt_payload()),
                encoding="utf-8",
            )
            output = root / "fallback.pt"
            with patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_weighted_jacobian_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_weighted_jacobian(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=prompt_path,
                    start_layer=0,
                    end_layer=1,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(4, 8),
                    generalized_regularization_pairs=(
                        (1e-4, 1e-4),
                    ),
                    sketch_rows=9,
                    selection_nll_atol=0.0,
                    selection_top1_min=1.0,
                    full_rank_nll_atol=1e-5,
                    jacobian_max_sequences=0,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )
            selection = report["analysis"]["calibration_b"]["selection"]
            self.assertEqual(
                selection["reason"],
                "native_full_rank_fallback",
            )
            self.assertEqual(
                selection["locked_candidate"]["candidate_id"],
                "native_fisher.joint.rank_8",
            )
            self.assertFalse(report["analysis"]["jacobian"]["enabled"])
            self.assertEqual(
                tokenizer.prompt_calls,
                [
                    ("a one", "a two"),
                    ("b one", "b two"),
                    ("v one", "v two"),
                ],
            )
            loaded = load_gemma3_weighted_jacobian_artifact(output)
            self.assertIsNone(loaded["jacobian"])
            self.assertIsNone(loaded["merged_factor"])

        self.assertEqual(
            default_gemma3_weighted_jacobian_output(
                "google/gemma-3-270m",
                4,
                6,
            ),
            Path(
                ".local-runs/google--gemma-3-270m/"
                "layers-4-6-weighted-jacobian.pt"
            ),
        )
        parsed = build_parser().parse_args(
            [
                "--retained-ranks",
                "640",
                "512",
                "--generalized-regularization",
                "1e-6:1e-12",
                "--jacobian-max-sequences",
                "0",
            ]
        )
        self.assertEqual(parsed.ranks, [640, 512])
        self.assertEqual(
            parsed.regularization_pairs,
            [(1e-6, 1e-12)],
        )
        self.assertEqual(parsed.jacobian_max_sequences, 0)


if __name__ == "__main__":
    unittest.main()
