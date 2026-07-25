import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_stability_experiment import (
    default_gemma3_stability_output,
    load_gemma3_prompt_splits,
    load_gemma3_stability_artifact,
    run_gemma3_stability,
)
from fisher_graph.model import ToyTransformer


class StabilityTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(
        self,
        prompts: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        rows = [
            [2, 4 + index, 7 + index, 1]
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


class Gemma3StabilityExperimentTests(unittest.TestCase):
    def test_prompt_split_loader_is_strict_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(
                json.dumps(prompt_payload()),
                encoding="utf-8",
            )
            splits = load_gemma3_prompt_splits(path)

            self.assertEqual(len(splits.calibration_a), 2)
            self.assertEqual(len(splits.test), 2)
            self.assertEqual(
                splits.metadata()["counts"],
                {
                    "calibration_a": 2,
                    "calibration_b": 2,
                    "validation": 2,
                    "test": 2,
                },
            )

            overlapping = prompt_payload()
            overlapping["test"] = ["a one", "t two"]
            path.write_text(json.dumps(overlapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pairwise disjoint"):
                load_gemma3_prompt_splits(path)

            invalid = prompt_payload()
            invalid["unexpected"] = True
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields"):
                load_gemma3_prompt_splits(path)

    def test_synthetic_end_to_end_stability_artifact(self) -> None:
        torch.manual_seed(933)
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=31,
                max_sequence_length=8,
                d_model=8,
                n_heads=2,
                n_layers=2,
                d_ff=12,
                dropout=0.0,
            )
        ).eval()
        model.requires_grad_(False)
        tokenizer = StabilityTokenizer()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "splits.json"
            prompt_path.write_text(
                json.dumps(prompt_payload()),
                encoding="utf-8",
            )
            output = root / "analysis" / "stability.pt"
            with patch(
                "fisher_graph.gemma3_stability_experiment.load_gemma3",
                return_value=(tokenizer, model),
            ), patch(
                "fisher_graph.gemma3_stability_experiment."
                "Gemma3CausalLMAdapter",
                side_effect=ToyTransformerAdapter,
            ):
                report = run_gemma3_stability(
                    cache_dir=root / "external-cache",
                    prompt_splits_path=prompt_path,
                    layer_index=0,
                    max_length=8,
                    tokenization_batch_size=2,
                    ranks=(2, 4),
                    sketch_rows=6,
                    device_name="cpu",
                    dtype="float32",
                    output=output,
                )

            calibration, validation, metadata = (
                load_gemma3_stability_artifact(output)
            )
            raw = torch.load(output, map_location="cpu", weights_only=True)
            report_exists = output.with_suffix(".json").is_file()
            tampered = root / "analysis" / "tampered.pt"
            raw["validation"]["bases"]["calibration_a"]["layer.0.input"][
                "basis_sha256"
            ] = "0" * 64
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "not bound to its named frozen basis",
            ):
                load_gemma3_stability_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["protocol"]["activation_sites"] = (
                "invented.input",
                "invented.output",
            )
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "protocol activation sites",
            ):
                load_gemma3_stability_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["protocol"]["tokenized_splits"]["validation"][
                "serialized_sha256"
            ] = "f" * 64
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "tokenized provenance does not match payloads",
            ):
                load_gemma3_stability_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            prompt_metadata = raw["protocol"]["prompt_splits"]
            for field in ("normalized_sha256", "per_prompt_sha256"):
                (
                    prompt_metadata[field]["validation"],
                    prompt_metadata[field]["test"],
                ) = (
                    prompt_metadata[field]["test"],
                    prompt_metadata[field]["validation"],
                )
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "source prompts do not match",
            ):
                load_gemma3_stability_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["protocol"]["prompt_splits"]["normalized_sha256"][
                "test"
            ] = "0" * 64
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "aggregate digest does not match",
            ):
                load_gemma3_stability_artifact(tampered)

            raw = torch.load(output, map_location="cpu", weights_only=True)
            raw["calibration"]["calibration_a"]["collection"]["bases"][
                "layer.0.input"
            ]["fisher"]["rows_seen"] += 100
            torch.save(raw, tampered)
            with self.assertRaisesRegex(
                ValueError,
                "calibration Fisher payload",
            ):
                load_gemma3_stability_artifact(tampered)

        self.assertTrue(report_exists)
        self.assertFalse(raw["contains_model_weights"])
        self.assertFalse(
            report["scientific_status"]["test_split_evaluated"]
        )
        self.assertEqual(
            report["protocol"]["prompt_splits"]["counts"]["test"],
            2,
        )
        self.assertEqual(
            set(report["protocol"]["tokenized_splits"]),
            {
                "calibration_a",
                "calibration_b",
                "calibration_full",
                "validation",
            },
        )
        self.assertNotIn("test", report["protocol"]["tokenized_splits"])
        self.assertEqual(
            report["protocol"]["tokenized_splits"]["validation"][
                "sequences"
            ],
            2,
        )
        streams = report["protocol"]["tokenized_splits"]
        self.assertEqual(
            streams["calibration_full"]["source_prompt_sha256"],
            streams["calibration_a"]["source_prompt_sha256"]
            + streams["calibration_b"]["source_prompt_sha256"],
        )
        self.assertEqual(
            [
                example["content_sha256"]
                for example in streams["calibration_full"]["examples"]
            ],
            [
                example["content_sha256"]
                for split_name in ("calibration_a", "calibration_b")
                for example in streams[split_name]["examples"]
            ],
        )
        self.assertEqual(
            set(report["protocol"]["library_versions"]),
            {
                "python",
                "torch",
                "transformers",
                "tokenizers",
                "sentencepiece",
            },
        )
        self.assertEqual(
            len(
                report["protocol"]["prompt_splits"][
                    "per_prompt_sha256"
                ]["test"]
            ),
            2,
        )
        self.assertEqual(calibration["calibration_a"].sequences, 2)
        self.assertEqual(calibration["calibration_b"].sequences, 2)
        self.assertEqual(calibration["calibration_full"].sequences, 4)
        self.assertEqual(
            validation["calibration_a"]["layer.0.input"].modes,
            4,
        )
        self.assertEqual(
            validation["calibration_full"]["layer.0.input"].modes,
            4,
        )
        self.assertEqual(
            metadata["protocol"]["ranks"],
            (2, 4),
        )
        self.assertEqual(metadata["protocol"]["extraction_rank"], 5)
        for activation_name, curve in report["analysis"][
            "rank_curve"
        ].items():
            with self.subTest(activation_name=activation_name):
                self.assertEqual(
                    [point["rank"] for point in curve],
                    [2, 4],
                )
                self.assertIsNotNone(
                    curve[-1]["calibration_full_relative_eigengap"]
                )
                for point in curve:
                    self.assertGreaterEqual(
                        point["split_mean_squared_overlap"],
                        0.0,
                    )
                    self.assertLessEqual(
                        point["split_mean_squared_overlap"],
                        1.0,
                    )
                    self.assertGreaterEqual(
                        point[
                            "validation_exact_rayleigh_fraction_min"
                        ],
                        0.0,
                    )
                    self.assertLessEqual(
                        point[
                            "validation_exact_rayleigh_fraction_min"
                        ],
                        1.0,
                    )
                    self.assertGreaterEqual(
                        point[
                            "validation_full_exact_rayleigh_fraction"
                        ],
                        0.0,
                    )
                    self.assertLessEqual(
                        point[
                            "validation_full_exact_rayleigh_fraction"
                        ],
                        1.0,
                    )

    def test_default_output_and_invalid_options(self) -> None:
        self.assertEqual(
            default_gemma3_stability_output(
                "google/gemma-3-270m",
                5,
            ),
            Path(
                ".local-runs/google--gemma-3-270m/"
                "layer-5-fisher-stability.pt"
            ),
        )
        invalid = (
            ({"ranks": ()}, "ranks cannot be empty"),
            ({"ranks": (0,)}, "positive integers"),
            (
                {"ranks": (4,), "sketch_rows": 4},
                "greater than the maximum rank",
            ),
            ({"output": Path("report.json")}, ".pt suffix"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs), patch(
                "fisher_graph.gemma3_stability_experiment.load_gemma3"
            ) as loader:
                with self.assertRaisesRegex(ValueError, message):
                    run_gemma3_stability(**kwargs)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
