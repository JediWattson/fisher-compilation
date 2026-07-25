import json
import unittest
from pathlib import Path

from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_FIXTURE = (
    REPOSITORY_ROOT
    / "examples"
    / "gemma3_gated_executor_prompts.json"
)
STABILITY_FIXTURE = (
    REPOSITORY_ROOT / "examples" / "gemma3_stability_prompts.json"
)
LEGACY_PROMPTS = REPOSITORY_ROOT / "examples" / "gemma3_prompts.txt"


def _canonical_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


class Gemma3ExecutorPromptFixtureTests(unittest.TestCase):
    def test_fixture_is_strict_fresh_disjoint_and_length_diverse(self) -> None:
        splits = load_gemma3_prompt_splits(EXECUTOR_FIXTURE)
        self.assertEqual(
            splits.scientific_status,
            "fresh_executor_protocol_diagnostic_only_test_reserved_hash_only",
        )
        split_values = {
            "calibration_a": splits.calibration_a,
            "calibration_b": splits.calibration_b,
            "validation": splits.validation,
            "test": splits.test,
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

        stability_raw = json.loads(
            STABILITY_FIXTURE.read_text(encoding="utf-8")
        )
        old_prompts = {
            _canonical_prompt(prompt)
            for name in (
                "calibration_a",
                "calibration_b",
                "validation",
                "test",
            )
            for prompt in stability_raw[name]
        }
        old_prompts.update(
            _canonical_prompt(prompt)
            for prompt in LEGACY_PROMPTS.read_text(
                encoding="utf-8"
            ).splitlines()
            if prompt.strip()
        )
        self.assertTrue(new_prompts.isdisjoint(old_prompts))

        for name, prompts in split_values.items():
            word_counts = [len(prompt.split()) for prompt in prompts]
            with self.subTest(split=name):
                self.assertLessEqual(min(word_counts), 4)
                self.assertGreaterEqual(max(word_counts), 45)
                self.assertGreaterEqual(max(word_counts) - min(word_counts), 40)

        metadata = splits.metadata()
        self.assertEqual(
            metadata["normalized_sha256"],
            {
                "calibration_a": (
                    "e7b87e6e80bbabf26e7c7a3a437a7e317bd156907857bc"
                    "f2fa2b24101dda233e"
                ),
                "calibration_b": (
                    "c79d4d6a83bf396338553aaa31c74504eb82cffbbf2375e"
                    "0c249ee7fe788e36c"
                ),
                "validation": (
                    "a7603a1971a9ac159d6da935b7a8cb42b21cc0630bcd4b"
                    "61e72bc8edbbdd0756"
                ),
                "test": (
                    "7ba31678335db4edb4171624034d76dcada9927d8c5bb19"
                    "8e289ef8d83ee0849"
                ),
            },
        )
        all_hashes = [
            digest
            for hashes in metadata["per_prompt_sha256"].values()
            for digest in hashes
        ]
        self.assertEqual(len(all_hashes), 64)
        self.assertEqual(len(set(all_hashes)), 64)
        self.assertTrue(
            all(
                len(digest) == 64
                and set(digest) <= set("0123456789abcdef")
                for digest in all_hashes
            )
        )


if __name__ == "__main__":
    unittest.main()
