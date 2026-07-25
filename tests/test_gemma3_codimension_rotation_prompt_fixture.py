import hashlib
import json
import unittest
from pathlib import Path

from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"
ROTATION_FIXTURE = EXAMPLES / "gemma3_codimension_rotation_prompts.json"
REFERENCE_FIXTURES = (
    EXAMPLES / "gemma3_gated_executor_prompts.json",
    EXAMPLES / "gemma3_projection_ladder_prompts.json",
    EXAMPLES / "gemma3_stability_prompts.json",
)
LEGACY_PROMPTS = EXAMPLES / "gemma3_prompts.txt"
SPLIT_NAMES = (
    "calibration_a",
    "calibration_b",
    "validation",
    "test",
)


def _canonical_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _prompt_hash(value: str) -> str:
    payload = json.dumps(
        [value],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _length_bucket(word_count: int) -> str:
    if word_count <= 12:
        return "short"
    if word_count <= 24:
        return "medium"
    if word_count <= 39:
        return "long"
    return "extended"


class Gemma3CodimensionRotationPromptFixtureTests(unittest.TestCase):
    def test_fixture_is_strict_fresh_disjoint_and_diverse(self) -> None:
        splits = load_gemma3_prompt_splits(ROTATION_FIXTURE)
        self.assertEqual(
            splits.scientific_status,
            (
                "fresh_codimension_rotation_diagnostic_"
                "calibration_a_sensitivity_b_selection_"
                "validation_locked_test_hash_only"
            ),
        )
        split_values = {
            name: getattr(splits, name)
            for name in SPLIT_NAMES
        }
        self.assertEqual(
            {name: len(values) for name, values in split_values.items()},
            {
                "calibration_a": 16,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            },
        )

        new_prompts = {
            _canonical_prompt(prompt)
            for values in split_values.values()
            for prompt in values
        }
        new_hashes = {
            _prompt_hash(prompt)
            for values in split_values.values()
            for prompt in values
        }
        self.assertEqual(len(new_prompts), 64)
        self.assertEqual(len(new_hashes), 64)
        self.assertEqual(
            new_hashes,
            {
                digest
                for digests in splits.metadata()[
                    "per_prompt_sha256"
                ].values()
                for digest in digests
            },
        )

        for path in REFERENCE_FIXTURES:
            with self.subTest(reference_fixture=path.name):
                reference = load_gemma3_prompt_splits(path)
                reference_values = [
                    prompt
                    for name in SPLIT_NAMES
                    for prompt in getattr(reference, name)
                ]
                self.assertTrue(
                    new_prompts.isdisjoint(
                        _canonical_prompt(prompt)
                        for prompt in reference_values
                    )
                )
                self.assertTrue(
                    new_hashes.isdisjoint(
                        _prompt_hash(prompt)
                        for prompt in reference_values
                    )
                )

        legacy_values = [
            prompt.strip()
            for prompt in LEGACY_PROMPTS.read_text(
                encoding="utf-8"
            ).splitlines()
            if prompt.strip()
        ]
        self.assertTrue(
            new_prompts.isdisjoint(
                _canonical_prompt(prompt)
                for prompt in legacy_values
            )
        )
        self.assertTrue(
            new_hashes.isdisjoint(
                _prompt_hash(prompt)
                for prompt in legacy_values
            )
        )

        for name, prompts in split_values.items():
            word_counts = [len(prompt.split()) for prompt in prompts]
            first_words = {
                _canonical_prompt(prompt).split()[0]
                for prompt in prompts
            }
            with self.subTest(split=name):
                self.assertGreaterEqual(min(word_counts), 7)
                self.assertLessEqual(min(word_counts), 10)
                self.assertGreaterEqual(max(word_counts), 55)
                self.assertGreaterEqual(
                    max(word_counts) - min(word_counts),
                    45,
                )
                self.assertEqual(
                    {
                        _length_bucket(count)
                        for count in word_counts
                    },
                    {"short", "medium", "long", "extended"},
                )
                self.assertGreaterEqual(len(first_words), 10)
                self.assertTrue(
                    any(prompt.endswith("?") for prompt in prompts)
                )
                self.assertTrue(
                    any('"' in prompt for prompt in prompts)
                )
                self.assertTrue(
                    any("`" in prompt for prompt in prompts)
                )
                self.assertTrue(
                    any(not prompt.isascii() for prompt in prompts)
                )

        metadata = splits.metadata()
        self.assertEqual(
            metadata["counts"],
            {
                "calibration_a": 16,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            },
        )
        self.assertEqual(
            set(metadata["normalized_sha256"]),
            set(SPLIT_NAMES),
        )
        self.assertTrue(
            all(
                len(digest) == 64
                and set(digest) <= set("0123456789abcdef")
                for digest in metadata["normalized_sha256"].values()
            ),
        )


if __name__ == "__main__":
    unittest.main()
