import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"
FIXTURE = EXAMPLES / "gemma3_merged_supermode_oracle_prompts.json"
FAMILY_MANIFEST = (
    EXAMPLES / "gemma3_merged_supermode_oracle_prompt_families.json"
)
LEGACY_PROMPTS = EXAMPLES / "gemma3_prompts.txt"
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
EXPECTED_NORMALIZED_SHA256 = {
    "calibration_a": (
        "2ab87a0567893b5f444e4f08481a0fec0209074955cd085c1100f46db45d2430"
    ),
    "calibration_b": (
        "00501ac58890a3f1cbb310f3ea7fafea2851e95f5ad4ad99065fa00da9d4e6f7"
    ),
    "validation": (
        "751713cd821914bf5a06b7627279b8e0cf2d7adbe301fbbd2912dccb91de78be"
    ),
    "test": (
        "0adb2cee2870f93db450a16dbd4579a10fd2b553204165957a511269afcbe273"
    ),
}
EXPECTED_FIXTURE_SHA256 = (
    "7076a630a286607f130e8060e62b1f8a17214a2ec8a80bcb1c36cd85e650d938"
)
EXPECTED_FAMILY_COUNTS = {
    "calibration_a": 8,
    "calibration_b": 4,
    "validation": 4,
    "test": 4,
}
EXPECTED_FAMILY_IDS = {
    "calibration_a": (
        "calibration_a.tide_timebase_diagnostics",
        "calibration_a.archive_chain_of_custody",
        "calibration_a.modular_quantity_problems",
        "calibration_a.persistent_structure_explanations",
        "calibration_a.spectrometer_protocols",
        "calibration_a.agroecology_causal_studies",
        "calibration_a.formal_rule_inference",
        "calibration_a.coordinate_geometry_analogies",
    ),
    "calibration_b": (
        "calibration_b.bell_tower_acoustics",
        "calibration_b.ceramic_workshop_arithmetic",
        "calibration_b.version_control_semantics",
        "calibration_b.glacier_field_protocols",
    ),
    "validation": (
        "validation.lighthouse_optics",
        "validation.orchard_allocation_arithmetic",
        "validation.queueing_system_semantics",
        "validation.textile_dye_protocols",
    ),
    "test": (
        "test.radio_telemetry_diagnostics",
        "test.library_shelving_arithmetic",
        "test.capability_security_semantics",
        "test.cave_survey_protocols",
    ),
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


class Gemma3MergedSupermodeOraclePromptFixtureTests(
    unittest.TestCase
):
    def test_fixture_is_fresh_frozen_and_disjoint(self) -> None:
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
        self.assertEqual(
            hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            EXPECTED_FIXTURE_SHA256,
        )

        splits = load_gemma3_prompt_splits(FIXTURE)
        self.assertEqual(
            splits.scientific_status,
            (
                "merged_supermode_oracle_fresh_calibration_a_"
                "train_b_selection_validation_locked_test_hash_only"
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

        metadata = splits.metadata()
        self.assertEqual(metadata["counts"], EXPECTED_COUNTS)
        self.assertEqual(
            metadata["normalized_sha256"],
            EXPECTED_NORMALIZED_SHA256,
        )
        self.assertEqual(
            hashes,
            {
                digest
                for digests in metadata[
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

        legacy_values = [
            prompt.strip()
            for prompt in LEGACY_PROMPTS.read_text(
                encoding="utf-8"
            ).splitlines()
            if prompt.strip()
        ]
        self.assertTrue(
            canonical.isdisjoint(
                _canonical_prompt(prompt)
                for prompt in legacy_values
            )
        )
        self.assertTrue(
            hashes.isdisjoint(
                _prompt_hash(prompt)
                for prompt in legacy_values
            )
        )

    def test_family_manifest_binds_every_prompt_to_one_role(self) -> None:
        manifest = json.loads(
            FAMILY_MANIFEST.read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(manifest),
            {
                "schema",
                "format_version",
                "prompt_fixture",
                "prompt_fixture_sha256",
                "scientific_policy",
                "roles",
            },
        )
        self.assertEqual(
            manifest["schema"],
            "fisher_graph.gemma3_prompt_family_manifest",
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["prompt_fixture"], FIXTURE.name)
        self.assertEqual(
            manifest["prompt_fixture_sha256"],
            EXPECTED_FIXTURE_SHA256,
        )
        self.assertEqual(
            manifest["prompt_fixture_sha256"],
            hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            manifest["scientific_policy"],
            "one_template_family_one_role_no_cross_split_reuse",
        )
        self.assertEqual(set(manifest["roles"]), set(SPLIT_NAMES))

        splits = load_gemma3_prompt_splits(FIXTURE)
        metadata = splits.metadata()
        family_roles: dict[str, set[str]] = defaultdict(set)
        family_suffixes: set[str] = set()
        family_descriptions: set[str] = set()
        assigned_hashes: dict[str, tuple[str, str]] = {}

        for role in SPLIT_NAMES:
            role_manifest = manifest["roles"][role]
            self.assertEqual(
                set(role_manifest),
                {"normalized_sha256", "families"},
            )
            self.assertEqual(
                role_manifest["normalized_sha256"],
                metadata["normalized_sha256"][role],
            )
            families = role_manifest["families"]
            self.assertEqual(
                len(families),
                EXPECTED_FAMILY_COUNTS[role],
            )
            self.assertEqual(
                tuple(family["id"] for family in families),
                EXPECTED_FAMILY_IDS[role],
            )

            expected_start = 0
            expected_family_size = (
                8 if role == "calibration_a" else 4
            )
            role_hashes = metadata["per_prompt_sha256"][role]
            for family in families:
                self.assertEqual(
                    set(family),
                    {"id", "description", "start", "count"},
                )
                family_id = family["id"]
                description = family["description"]
                self.assertIsInstance(family_id, str)
                self.assertTrue(family_id.startswith(f"{role}."))
                family_suffix = family_id.split(".", maxsplit=1)[1]
                self.assertNotIn(family_suffix, family_suffixes)
                family_suffixes.add(family_suffix)
                self.assertEqual(family["start"], expected_start)
                self.assertEqual(
                    family["count"],
                    expected_family_size,
                )
                self.assertIsInstance(description, str)
                self.assertTrue(description.strip())
                self.assertNotIn(description, family_descriptions)
                family_descriptions.add(description)
                family_roles[family_id].add(role)

                stop = family["start"] + family["count"]
                for digest in role_hashes[family["start"] : stop]:
                    self.assertNotIn(digest, assigned_hashes)
                    assigned_hashes[digest] = (role, family_id)
                expected_start = stop

            self.assertEqual(expected_start, EXPECTED_COUNTS[role])

        self.assertEqual(len(family_roles), 20)
        self.assertTrue(
            all(
                len(roles) == 1
                for roles in family_roles.values()
            )
        )
        self.assertEqual(len(assigned_hashes), 112)
        self.assertEqual(
            set(assigned_hashes),
            {
                digest
                for digests in metadata[
                    "per_prompt_sha256"
                ].values()
                for digest in digests
            },
        )


if __name__ == "__main__":
    unittest.main()
