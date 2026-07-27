import json
import unittest

import torch

from fisher_graph.variable_static_full_span_v2_protocol import (
    V2_BASELINE_TASK_CONFIG,
    V2_ROLE_NAMES,
    V2_ROLE_SALT,
    V2_ROLE_SIZES,
    V2_TASK_CONFIG,
    _salted_rank,
    build_variable_static_full_span_v2_protocol,
    variable_static_full_span_v2_novelty_accuracy,
)


class VariableStaticFullSpanV2ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = build_variable_static_full_span_v2_protocol()

    def test_frozen_configuration_and_exact_context_accounting(self) -> None:
        protocol = self.protocol
        self.assertEqual(
            (
                V2_TASK_CONFIG.n_keys,
                V2_TASK_CONFIG.n_values,
                V2_TASK_CONFIG.split_seed,
            ),
            (10, 10, 26_071),
        )
        self.assertEqual(
            (
                V2_BASELINE_TASK_CONFIG.n_keys,
                V2_BASELINE_TASK_CONFIG.n_values,
                V2_BASELINE_TASK_CONFIG.split_seed,
            ),
            (8, 8, 26_071),
        )
        self.assertEqual(
            V2_ROLE_SALT,
            "fisher_graph.variable_static_full_span.roles.v2.n10",
        )
        self.assertEqual(
            tuple(V2_ROLE_SIZES.items()),
            (
                ("basis_fit_a", 128),
                ("graph_fit_a", 1_024),
                ("graph_stop_a", 192),
                ("graph_select_a", 192),
                ("calibration_b", 256),
            ),
        )
        self.assertEqual(tuple(protocol.roles), V2_ROLE_NAMES)

        audit = protocol.audit
        self.assertEqual(audit.baseline_contexts, 1_568)
        self.assertEqual(audit.source_contexts, 4_050)
        self.assertEqual(
            (
                audit.source_train_contexts,
                audit.source_validation_contexts,
                audit.source_test_contexts,
            ),
            (3_240, 405, 405),
        )
        self.assertEqual(
            (
                audit.fresh_train_contexts,
                audit.fresh_validation_contexts,
                audit.fresh_test_contexts,
            ),
            (1_986, 246, 250),
        )
        self.assertEqual(
            (
                audit.excluded_train_contexts,
                audit.excluded_validation_contexts,
                audit.excluded_test_contexts,
            ),
            (1_254, 159, 155),
        )
        self.assertEqual(audit.allocated_role_contexts, 1_792)
        self.assertEqual(audit.reserve_contexts, 194)
        self.assertEqual(
            dict(audit.role_context_counts),
            V2_ROLE_SIZES,
        )
        self.assertTrue(audit.all_overlap_checks_pass)

    def test_roles_are_the_declared_hash_rank_and_exhaust_fresh_train(self) -> None:
        protocol = self.protocol
        ranked = _salted_rank(protocol.fresh_train, salt=V2_ROLE_SALT)
        ranked_hashes = tuple(
            protocol.fresh_train.semantic_context_hashes[index]
            for index in ranked
        )
        allocated_hashes = tuple(
            semantic_hash
            for name in V2_ROLE_NAMES
            for semantic_hash in protocol.roles[name].semantic_context_hashes
        )
        self.assertEqual(
            allocated_hashes + protocol.reserve.semantic_context_hashes,
            ranked_hashes,
        )

        role_sets = tuple(
            set(protocol.roles[name].semantic_context_hashes)
            for name in V2_ROLE_NAMES
        )
        used = set().union(*role_sets)
        self.assertEqual(len(used), sum(V2_ROLE_SIZES.values()))
        self.assertTrue(used.isdisjoint(protocol.baseline_context_hashes))
        self.assertTrue(
            used.isdisjoint(
                protocol.fresh_validation.semantic_context_hashes
            )
        )
        self.assertTrue(
            used.isdisjoint(protocol.fresh_test.semantic_context_hashes)
        )
        self.assertEqual(
            used | set(protocol.reserve.semantic_context_hashes),
            set(protocol.fresh_train.semantic_context_hashes),
        )

    def test_manifest_is_json_compatible_and_records_every_overlap_check(
        self,
    ) -> None:
        manifest = self.protocol.manifest()
        encoded = json.dumps(manifest, allow_nan=False, sort_keys=True)
        self.assertTrue(encoded)
        self.assertEqual(
            manifest["schema"],
            "fisher_graph.variable_static_full_span_v2_protocol",
        )
        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["role_salt"], V2_ROLE_SALT)
        self.assertEqual(manifest["role_sizes"], V2_ROLE_SIZES)
        self.assertEqual(
            manifest["context_set_sha256"],
            {
                "excluded_baseline": (
                    "28e132822db129caac0878a9baf5f7734487670768e77dc28f74"
                    "dd1942a94f21"
                ),
                "basis_fit_a": (
                    "45c4c664d146ea153ca31c6ae45c66d229a6e4d7439ecd969e6"
                    "89666db393126"
                ),
                "graph_fit_a": (
                    "c46596ddf25ef50d953653c88fb7490edd0a2f5ea926066bea0c"
                    "efb633304ba3"
                ),
                "graph_stop_a": (
                    "24f4b0e6c9e8c6e98ab0024d144da324b433ac341f8c0ab98e7"
                    "a8ffc9a0a54c1"
                ),
                "graph_select_a": (
                    "e8aec505282f486b147dea709111f88016f29bd6a217e8953e92"
                    "9dbb3376a515"
                ),
                "calibration_b": (
                    "15e8068ed55d1dfc87f6c9c446e69bfddf4a9a0783a436ccc09"
                    "ac6917cd7a960"
                ),
                "reserve": (
                    "1068c48fdd8cb0a2986a535c81a6c8ce4e2a554e2dd4bb5d72"
                    "708bf8e0ee26e9"
                ),
                "fresh_validation": (
                    "9d7768b38356c64487f281459b2932682b2c08b106befe1962c5"
                    "3433efa1c808"
                ),
                "fresh_test": (
                    "3017769c01c12518247c01daf6624606362e290c05fcd4a09394"
                    "dbbd7586c3bd"
                ),
            },
        )
        self.assertTrue(manifest["audit"]["all_overlap_checks_pass"])
        overlap_values = {
            name: value
            for name, value in manifest["audit"].items()
            if name.endswith("_overlap")
        }
        self.assertEqual(len(overlap_values), 11)
        self.assertFalse(any(overlap_values.values()))

    def test_perfect_logits_pass_nonempty_novelty_strata(self) -> None:
        split = self.protocol.fresh_validation
        logits = torch.full(
            (split.samples, V2_TASK_CONFIG.vocab_size),
            -10.0,
        )
        logits[
            torch.arange(split.samples),
            split.answer_token_ids,
        ] = 10.0
        result = variable_static_full_span_v2_novelty_accuracy(split, logits)

        self.assertEqual(
            (result.new_key.contexts, result.new_key.samples),
            (143, 4_576),
        )
        self.assertEqual(
            (result.new_value.contexts, result.new_value.samples),
            (157, 5_024),
        )
        self.assertEqual(
            (result.key_only.contexts, result.key_only.samples),
            (89, 2_848),
        )
        self.assertEqual(
            (result.value_only.contexts, result.value_only.samples),
            (103, 3_296),
        )
        self.assertEqual(
            (result.both.contexts, result.both.samples),
            (54, 1_728),
        )
        self.assertEqual(
            result.key_only.contexts
            + result.value_only.contexts
            + result.both.contexts,
            split.contexts,
        )
        self.assertTrue(result.primary_strata_nonempty)
        self.assertTrue(result.new_key_pass)
        self.assertTrue(result.new_value_pass)
        self.assertTrue(result.both_pass)
        self.assertTrue(result.passes)

    def test_one_both_novelty_error_fails_all_primary_exact_gates(self) -> None:
        split = self.protocol.fresh_validation
        logits = torch.full(
            (split.samples, V2_TASK_CONFIG.vocab_size),
            -10.0,
        )
        logits[
            torch.arange(split.samples),
            split.answer_token_ids,
        ] = 10.0
        contexts = split.semantic_contexts
        both_contexts = (
            (contexts[:, :2] >= V2_BASELINE_TASK_CONFIG.n_keys).any(dim=1)
            & (contexts[:, 2:] >= V2_BASELINE_TASK_CONFIG.n_values).any(
                dim=1
            )
        )
        both_row = int(
            both_contexts.nonzero(as_tuple=False).flatten()[0].item()
        )
        sample = int(
            (split.example_context_indices == both_row)
            .nonzero(as_tuple=False)
            .flatten()[0]
            .item()
        )
        target = int(split.answer_token_ids[sample].item())
        wrong = (target + 1) % V2_TASK_CONFIG.vocab_size
        logits[sample, target] = -10.0
        logits[sample, wrong] = 10.0

        result = variable_static_full_span_v2_novelty_accuracy(split, logits)
        self.assertTrue(result.primary_strata_nonempty)
        self.assertFalse(result.new_key_pass)
        self.assertFalse(result.new_value_pass)
        self.assertFalse(result.both_pass)
        self.assertFalse(result.passes)
        self.assertEqual(result.both.correct_samples, result.both.samples - 1)

    def test_novelty_input_validation_is_fail_closed(self) -> None:
        split = self.protocol.fresh_validation
        with self.assertRaisesRegex(ValueError, "shape"):
            variable_static_full_span_v2_novelty_accuracy(
                split,
                torch.zeros(split.samples),
            )
        with self.assertRaisesRegex(ValueError, "minimum_accuracy"):
            variable_static_full_span_v2_novelty_accuracy(
                split,
                torch.zeros(split.samples, V2_TASK_CONFIG.vocab_size),
                minimum_accuracy=1.1,
            )


if __name__ == "__main__":
    unittest.main()
