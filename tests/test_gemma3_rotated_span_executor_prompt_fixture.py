import hashlib
import json
import unittest
from pathlib import Path

from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"
FIXTURE = EXAMPLES / "gemma3_rotated_span_executor_prompts.json"
SPLIT_NAMES = (
    "calibration_a",
    "calibration_b",
    "validation",
    "test",
)
EXPECTED_COUNTS = {
    "calibration_a": 64,
    "calibration_b": 16,
    "validation": 16,
    "test": 16,
}


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


class Gemma3RotatedSpanExecutorPromptFixtureTests(unittest.TestCase):
    def test_fixture_is_exact_text_and_hash_disjoint(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            set(raw),
            {
                "schema",
                "format_version",
                "scientific_status",
                *SPLIT_NAMES,
            },
        )
        self.assertEqual(
            raw["schema"],
            "fisher_graph.gemma3_prompt_splits",
        )
        self.assertEqual(raw["format_version"], 1)

        splits = load_gemma3_prompt_splits(FIXTURE)
        self.assertEqual(
            splits.scientific_status,
            (
                "rotated_span_executor_fresh_train_b_"
                "validation_test_hash_only"
            ),
        )
        split_values = {
            name: getattr(splits, name)
            for name in SPLIT_NAMES
        }
        self.assertEqual(
            {
                name: len(values)
                for name, values in split_values.items()
            },
            EXPECTED_COUNTS,
        )
        self.assertTrue(
            all(
                isinstance(prompt, str) and prompt.strip()
                for values in split_values.values()
                for prompt in values
            )
        )

        values = [
            prompt
            for name in SPLIT_NAMES
            for prompt in split_values[name]
        ]
        canonical = {
            _canonical_prompt(prompt)
            for prompt in values
        }
        hashes = {
            _prompt_hash(prompt)
            for prompt in values
        }
        self.assertEqual(len(values), 112)
        self.assertEqual(len(canonical), 112)
        self.assertEqual(len(hashes), 112)
        self.assertEqual(
            hashes,
            {
                digest
                for digests in splits.metadata()[
                    "per_prompt_sha256"
                ].values()
                for digest in digests
            },
        )

        reference_fixtures = sorted(
            path
            for path in EXAMPLES.glob("gemma3_*_prompts.json")
            if path != FIXTURE
        )
        self.assertTrue(reference_fixtures)
        for path in reference_fixtures:
            with self.subTest(reference_fixture=path.name):
                reference_values = _fixture_values(path)
                self.assertTrue(
                    canonical.isdisjoint(
                        _canonical_prompt(prompt)
                        for prompt in reference_values
                    )
                )
                self.assertTrue(
                    hashes.isdisjoint(
                        _prompt_hash(prompt)
                        for prompt in reference_values
                    )
                )


if __name__ == "__main__":
    unittest.main()
