import hashlib
import json
import unittest
from pathlib import Path

from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"
EXPANDED_FIXTURE = (
    EXAMPLES / "gemma3_codimension_rotation_expanded_a_prompts.json"
)
ORIGINAL_FIXTURE = EXAMPLES / "gemma3_codimension_rotation_prompts.json"
OTHER_FIXTURES = (
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


def _fixture_values(path: Path) -> list[str]:
    splits = load_gemma3_prompt_splits(path)
    return [
        prompt
        for name in SPLIT_NAMES
        for prompt in getattr(splits, name)
    ]


class Gemma3CodimensionRotationExpandedAPromptFixtureTests(
    unittest.TestCase
):
    def test_expanded_a_is_fresh_and_other_splits_are_untouched(
        self,
    ) -> None:
        expanded = load_gemma3_prompt_splits(EXPANDED_FIXTURE)
        original = load_gemma3_prompt_splits(ORIGINAL_FIXTURE)
        self.assertEqual(
            expanded.scientific_status,
            (
                "expanded_codimension_rotation_calibration_a_after_"
                "identifiability_failure_b_validation_test_untouched"
            ),
        )
        self.assertEqual(
            {
                name: len(getattr(expanded, name))
                for name in SPLIT_NAMES
            },
            {
                "calibration_a": 64,
                "calibration_b": 16,
                "validation": 16,
                "test": 16,
            },
        )

        self.assertEqual(
            expanded.calibration_a[:16],
            original.calibration_a,
        )
        for name in ("calibration_b", "validation", "test"):
            with self.subTest(untouched_split=name):
                expanded_values = getattr(expanded, name)
                original_values = getattr(original, name)
                self.assertEqual(expanded_values, original_values)
                self.assertEqual(
                    [value.encode("utf-8") for value in expanded_values],
                    [value.encode("utf-8") for value in original_values],
                )
                self.assertEqual(
                    expanded.metadata()["normalized_sha256"][name],
                    original.metadata()["normalized_sha256"][name],
                )

        all_values = [
            prompt
            for name in SPLIT_NAMES
            for prompt in getattr(expanded, name)
        ]
        all_prompts = {
            _canonical_prompt(prompt)
            for prompt in all_values
        }
        all_hashes = {
            _prompt_hash(prompt)
            for prompt in all_values
        }
        self.assertEqual(len(all_prompts), 112)
        self.assertEqual(len(all_hashes), 112)
        self.assertEqual(
            all_hashes,
            {
                digest
                for digests in expanded.metadata()[
                    "per_prompt_sha256"
                ].values()
                for digest in digests
            },
        )

        additions = expanded.calibration_a[16:]
        self.assertEqual(len(additions), 48)
        added_prompts = {
            _canonical_prompt(prompt)
            for prompt in additions
        }
        added_hashes = {
            _prompt_hash(prompt)
            for prompt in additions
        }
        self.assertEqual(len(added_prompts), 48)
        self.assertEqual(len(added_hashes), 48)

        reference_values = _fixture_values(ORIGINAL_FIXTURE)
        for path in OTHER_FIXTURES:
            reference_values.extend(_fixture_values(path))
        reference_values.extend(
            prompt.strip()
            for prompt in LEGACY_PROMPTS.read_text(
                encoding="utf-8"
            ).splitlines()
            if prompt.strip()
        )
        self.assertTrue(
            added_prompts.isdisjoint(
                _canonical_prompt(prompt)
                for prompt in reference_values
            )
        )
        self.assertTrue(
            added_hashes.isdisjoint(
                _prompt_hash(prompt)
                for prompt in reference_values
            )
        )

        for path in OTHER_FIXTURES:
            with self.subTest(disjoint_fixture=path.name):
                values = _fixture_values(path)
                self.assertTrue(
                    all_prompts.isdisjoint(
                        _canonical_prompt(prompt)
                        for prompt in values
                    )
                )
                self.assertTrue(
                    all_hashes.isdisjoint(
                        _prompt_hash(prompt)
                        for prompt in values
                    )
                )

        word_counts = [len(prompt.split()) for prompt in additions]
        self.assertLessEqual(min(word_counts), 9)
        self.assertGreaterEqual(max(word_counts), 60)
        self.assertGreaterEqual(max(word_counts) - min(word_counts), 50)
        self.assertGreaterEqual(
            len(
                {
                    min(count // 10, 5)
                    for count in word_counts
                }
            ),
            4,
        )
        self.assertGreaterEqual(
            len(
                {
                    _canonical_prompt(prompt).split()[0]
                    for prompt in additions
                }
            ),
            30,
        )
        self.assertGreaterEqual(
            sum(not prompt.isascii() for prompt in additions),
            3,
        )
        self.assertGreaterEqual(
            sum(prompt.endswith("?") for prompt in additions),
            3,
        )
        self.assertGreaterEqual(
            sum('"' in prompt for prompt in additions),
            3,
        )
        self.assertGreaterEqual(
            sum("`" in prompt for prompt in additions),
            3,
        )


if __name__ == "__main__":
    unittest.main()
