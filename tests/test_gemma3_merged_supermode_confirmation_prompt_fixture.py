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
FIXTURE = (
    EXAMPLES / "gemma3_merged_supermode_confirmation_prompts.json"
)
FAMILY_MANIFEST = (
    EXAMPLES
    / "gemma3_merged_supermode_confirmation_prompt_families.json"
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
        "468bb4c2b0c68c051b5afdc5dce766ebe4f475f5f215d2816853f1a7770e6e0c"
    ),
    "calibration_b": (
        "38a09f36adf581b91da7144682400dd66fc4fd60529d953974f9170a7952353b"
    ),
    "validation": (
        "d7d275c94e5be8faa81c7ca71ea9980ebe2b8c4321cabb687bc9e673cb78298f"
    ),
    "test": (
        "9040a545db4382081444a4ffd60ab17c9190798306e4b367a67842af5b040774"
    ),
}
EXPECTED_FIXTURE_SHA256 = (
    "1d292f970a951fd15f85bc024c3fab7bb231971e9850d9e0110c4aab90f08c57"
)
EXPECTED_DOMAIN_TEMPLATES = {
    "calibration_a": (
        "river_lock_instrument_faults",
        "museum_accession_reconciliation",
        "astronomy_cadence_calculation",
        "database_index_explanation",
        "battery_cell_safety_protocol",
        "forest_canopy_causal_study",
        "set_membership_inference",
        "signal_basis_analogy",
    ),
    "calibration_b": (
        "clockwork_automaton_faults",
        "bakery_batch_calculation",
        "distributed_consensus_explanation",
        "coral_census_protocol",
    ),
    "validation": (
        "telescope_mount_faults",
        "rail_freight_calculation",
        "memory_allocator_explanation",
        "manuscript_ink_protocol",
    ),
    "test": (
        "hydraulic_elevator_faults",
        "solar_array_calculation",
        "encryption_key_explanation",
        "volcanic_gas_protocol",
    ),
}
MATCHED_BROAD_FORMS = {
    "observational_diagnosis",
    "quantitative_reasoning",
    "technical_explanation",
    "procedural_instruction",
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


class Gemma3MergedSupermodeConfirmationPromptFixtureTests(
    unittest.TestCase
):
    def test_confirmation_fixture_is_fresh_frozen_and_disjoint(
        self,
    ) -> None:
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
                "merged_supermode_q28_confirmation_fresh_"
                "calibration_a_train_b_selection_validation_"
                "locked_test_hash_only"
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

    def test_domain_templates_are_role_confined_and_forms_recorded(
        self,
    ) -> None:
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
                "broad_form_policy",
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
            (
                "one_domain_template_family_one_role_"
                "no_cross_split_reuse"
            ),
        )
        self.assertEqual(
            manifest["broad_form_policy"],
            (
                "broad_task_forms_may_repeat_across_roles_and_are_"
                "recorded_but_domain_template_families_must_"
                "remain_role_confined"
            ),
        )
        self.assertEqual(set(manifest["roles"]), set(SPLIT_NAMES))

        splits = load_gemma3_prompt_splits(FIXTURE)
        metadata = splits.metadata()
        domain_roles: dict[str, set[str]] = defaultdict(set)
        broad_form_roles: dict[str, set[str]] = defaultdict(set)
        family_ids: set[str] = set()
        descriptions: set[str] = set()
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
                tuple(
                    family["domain_template"]
                    for family in families
                ),
                EXPECTED_DOMAIN_TEMPLATES[role],
            )

            expected_start = 0
            expected_family_size = (
                8 if role == "calibration_a" else 4
            )
            role_hashes = metadata["per_prompt_sha256"][role]
            for family in families:
                self.assertEqual(
                    set(family),
                    {
                        "id",
                        "domain_template",
                        "broad_form",
                        "description",
                        "start",
                        "count",
                    },
                )
                family_id = family["id"]
                domain_template = family["domain_template"]
                broad_form = family["broad_form"]
                description = family["description"]
                self.assertEqual(
                    family_id,
                    f"{role}.{domain_template}",
                )
                self.assertNotIn(family_id, family_ids)
                family_ids.add(family_id)
                domain_roles[domain_template].add(role)
                broad_form_roles[broad_form].add(role)
                self.assertIsInstance(description, str)
                self.assertTrue(description.strip())
                self.assertNotIn(description, descriptions)
                descriptions.add(description)
                self.assertEqual(family["start"], expected_start)
                self.assertEqual(
                    family["count"],
                    expected_family_size,
                )

                stop = family["start"] + family["count"]
                for digest in role_hashes[family["start"] : stop]:
                    self.assertNotIn(digest, assigned_hashes)
                    assigned_hashes[digest] = (role, family_id)
                expected_start = stop

            self.assertEqual(expected_start, EXPECTED_COUNTS[role])

        self.assertEqual(len(domain_roles), 20)
        self.assertTrue(
            all(
                len(roles) == 1
                for roles in domain_roles.values()
            )
        )
        for broad_form in MATCHED_BROAD_FORMS:
            with self.subTest(broad_form=broad_form):
                self.assertEqual(
                    broad_form_roles[broad_form],
                    set(SPLIT_NAMES),
                )
        self.assertEqual(
            {
                broad_form
                for broad_form, roles in broad_form_roles.items()
                if roles == set(SPLIT_NAMES)
            },
            MATCHED_BROAD_FORMS,
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
