import unittest
from pathlib import Path

from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECTION_FIXTURE = (
    REPOSITORY_ROOT
    / "examples"
    / "gemma3_projection_ladder_prompts.json"
)
REFERENCE_FIXTURES = (
    REPOSITORY_ROOT / "examples" / "gemma3_gated_executor_prompts.json",
    REPOSITORY_ROOT / "examples" / "gemma3_stability_prompts.json",
)
SPLIT_NAMES = (
    "calibration_a",
    "calibration_b",
    "validation",
    "test",
)


def _canonical_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


class Gemma3ProjectionPromptFixtureTests(unittest.TestCase):
    def test_fixture_is_strict_fresh_disjoint_and_diverse(self) -> None:
        splits = load_gemma3_prompt_splits(PROJECTION_FIXTURE)
        self.assertEqual(
            splits.scientific_status,
            (
                "fresh_projection_ladder_diagnostic_only_"
                "calibration_a_and_test_reserved_hash_only"
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
        self.assertEqual(len(new_prompts), 64)

        new_hashes = {
            digest
            for hashes in splits.metadata()["per_prompt_sha256"].values()
            for digest in hashes
        }
        self.assertEqual(len(new_hashes), 64)
        for path in REFERENCE_FIXTURES:
            with self.subTest(reference_fixture=path.name):
                reference = load_gemma3_prompt_splits(path)
                reference_prompts = {
                    _canonical_prompt(prompt)
                    for name in SPLIT_NAMES
                    for prompt in getattr(reference, name)
                }
                reference_hashes = {
                    digest
                    for hashes in reference.metadata()[
                        "per_prompt_sha256"
                    ].values()
                    for digest in hashes
                }
                self.assertTrue(new_prompts.isdisjoint(reference_prompts))
                self.assertTrue(new_hashes.isdisjoint(reference_hashes))

        for name, prompts in split_values.items():
            word_counts = [len(prompt.split()) for prompt in prompts]
            first_words = {
                _canonical_prompt(prompt).split()[0]
                for prompt in prompts
            }
            length_buckets = {
                min(count // 10, 4)
                for count in word_counts
            }
            with self.subTest(split=name):
                self.assertLessEqual(min(word_counts), 4)
                self.assertGreaterEqual(max(word_counts), 45)
                self.assertGreaterEqual(
                    max(word_counts) - min(word_counts),
                    40,
                )
                self.assertGreaterEqual(len(length_buckets), 4)
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
