import unittest
from collections import defaultdict

import torch

from fisher_graph.model import ToyTransformer
from fisher_graph.variable_associative import (
    DEFAULT_VARIABLE_ASSOCIATIVE_LAYOUTS,
    VariableAssociativeRecallTaskConfig,
    VariableAssociativeTokenRole,
    build_variable_associative_recall_splits,
    iter_variable_associative_calibration_batches,
    subset_variable_associative_recall_split,
    variable_associative_recall_model_config,
)


def small_config(
    *,
    split_seed: int = 26_071,
) -> VariableAssociativeRecallTaskConfig:
    return VariableAssociativeRecallTaskConfig(
        n_keys=4,
        n_values=4,
        split_seed=split_seed,
    )


def all_splits(splits):
    return (splits.train, splits.validation, splits.test)


class VariableAssociativeRecallTaskTests(unittest.TestCase):
    def test_default_layouts_break_position_length_equivalence(self) -> None:
        task = VariableAssociativeRecallTaskConfig()
        pairs = {
            (layout.valid_length, layout.supervised_position)
            for layout in task.layouts
        }
        positions_by_length = defaultdict(set)
        lengths_by_position = defaultdict(set)
        for length, position in pairs:
            positions_by_length[length].add(position)
            lengths_by_position[position].add(length)

        self.assertEqual(set(positions_by_length), set(range(8, 13)))
        self.assertTrue(
            any(len(positions) > 1 for positions in positions_by_length.values())
        )
        self.assertTrue(
            any(len(lengths) > 1 for lengths in lengths_by_position.values())
        )
        self.assertEqual(task.minimum_sequence_length, 8)
        self.assertEqual(task.maximum_sequence_length, 12)
        self.assertEqual(task.variants_per_context, 32)
        self.assertEqual(
            task.semantic_context_count,
            28 * 8 * 7,
        )

        causal_groups = defaultdict(set)
        for layout in DEFAULT_VARIABLE_ASSOCIATIVE_LAYOUTS:
            causal_groups[layout.causal_layout_key].add(
                layout.suffix_fillers
            )
        self.assertTrue(
            any(len(suffixes) > 1 for suffixes in causal_groups.values())
        )

    def test_splits_are_semantic_context_disjoint_grouped_and_complete(
        self,
    ) -> None:
        task = small_config()
        splits = build_variable_associative_recall_splits(task)

        self.assertEqual(
            (
                splits.train.contexts,
                splits.validation.contexts,
                splits.test.contexts,
            ),
            (57, 7, 8),
        )
        self.assertEqual(
            (
                splits.train.samples,
                splits.validation.samples,
                splits.test.samples,
            ),
            (57 * 32, 7 * 32, 8 * 32),
        )
        context_id_sets = [
            set(split.context_ids.tolist()) for split in all_splits(splits)
        ]
        context_hash_sets = [
            set(split.semantic_context_hashes) for split in all_splits(splits)
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(
                    context_id_sets[left].isdisjoint(context_id_sets[right])
                )
                self.assertTrue(
                    context_hash_sets[left].isdisjoint(
                        context_hash_sets[right]
                    )
                )
        self.assertEqual(
            len(set().union(*context_id_sets)),
            task.semantic_context_count,
        )
        self.assertEqual(
            len(set().union(*context_hash_sets)),
            task.semantic_context_count,
        )

        for split in all_splits(splits):
            counts = torch.bincount(
                split.example_context_indices,
                minlength=split.contexts,
            )
            self.assertTrue(
                torch.equal(
                    counts,
                    torch.full_like(counts, task.variants_per_context),
                )
            )
            self.assertTrue(
                bool((split.semantic_contexts[:, 0] < split.semantic_contexts[:, 1]).all())
            )
            for context_index in range(split.contexts):
                selected = split.example_context_indices == context_index
                variants = set(
                    zip(
                        split.query_slots[selected].tolist(),
                        split.pair_orders[selected].tolist(),
                        split.layout_indices[selected].tolist(),
                    )
                )
                self.assertEqual(
                    variants,
                    {
                        (query_slot, pair_order, layout_index)
                        for query_slot in range(2)
                        for pair_order in range(2)
                        for layout_index in range(len(task.layouts))
                    },
                )

    def test_examples_are_right_padded_and_only_answer_is_supervised(
        self,
    ) -> None:
        task = small_config()
        split = build_variable_associative_recall_splits(task).validation
        positions = torch.arange(task.maximum_sequence_length).unsqueeze(0)
        expected_mask = positions < split.valid_lengths.unsqueeze(1)

        self.assertTrue(torch.equal(split.attention_mask, expected_mask))
        self.assertTrue(
            bool(
                (
                    split.input_ids[~split.attention_mask]
                    == task.pad_token_id
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    split.token_role_ids[~split.attention_mask]
                    == int(VariableAssociativeTokenRole.PAD)
                ).all()
            )
        )
        supervised = split.targets != task.ignore_index
        self.assertTrue(torch.equal(supervised.sum(dim=1), torch.ones(split.samples, dtype=torch.long)))
        rows = torch.arange(split.samples)
        self.assertTrue(
            bool(supervised[rows, split.supervised_positions].all())
        )
        self.assertTrue(
            bool(
                (
                    split.token_role_ids[
                        rows, split.supervised_positions
                    ]
                    == int(VariableAssociativeTokenRole.ANSWER_MARKER)
                ).all()
            )
        )
        self.assertTrue(
            torch.equal(
                split.answer_token_ids,
                task.value_offset + split.answer_value_indices,
            )
        )

    def test_query_and_pair_order_variants_preserve_lookup_semantics(
        self,
    ) -> None:
        task = small_config()
        split = build_variable_associative_recall_splits(task).validation

        for row in range(split.samples):
            context = split.semantic_contexts[
                split.example_context_indices[row]
            ]
            keys = context[:2]
            values = context[2:]
            query_slot = int(split.query_slots[row].item())
            pair_order = int(split.pair_orders[row].item())
            expected_rendered_slots = (
                (0, 1) if pair_order == 0 else (1, 0)
            )
            key_positions = (
                split.token_role_ids[row]
                == int(VariableAssociativeTokenRole.KEY)
            ).nonzero(as_tuple=False).flatten()
            value_positions = (
                split.token_role_ids[row]
                == int(VariableAssociativeTokenRole.VALUE)
            ).nonzero(as_tuple=False).flatten()

            self.assertEqual(
                split.input_ids[row, key_positions].tolist(),
                [int(keys[index]) for index in expected_rendered_slots],
            )
            self.assertEqual(
                split.input_ids[row, value_positions].tolist(),
                [
                    task.value_offset + int(values[index])
                    for index in expected_rendered_slots
                ],
            )
            self.assertEqual(
                int(split.queried_key_ids[row].item()),
                int(keys[query_slot].item()),
            )
            self.assertEqual(
                int(split.answer_value_indices[row].item()),
                int(values[query_slot].item()),
            )
            self.assertEqual(
                int(split.rendered_query_slots[row].item()),
                expected_rendered_slots.index(query_slot),
            )

    def test_prefix_hashes_expose_future_only_length_variants(self) -> None:
        split = build_variable_associative_recall_splits(
            small_config()
        ).validation
        groups = defaultdict(list)
        for row, prefix_hash in enumerate(split.causal_prefix_hashes):
            groups[prefix_hash].append(row)
        matched_groups = [
            rows
            for rows in groups.values()
            if len({int(split.valid_lengths[row]) for row in rows}) > 1
        ]
        self.assertTrue(matched_groups)

        for group in matched_groups:
            reference = group[0]
            reference_position = int(split.supervised_positions[reference])
            for row in group[1:]:
                self.assertEqual(
                    int(split.example_context_indices[row]),
                    int(split.example_context_indices[reference]),
                )
                self.assertEqual(
                    int(split.query_slots[row]),
                    int(split.query_slots[reference]),
                )
                self.assertEqual(
                    int(split.pair_orders[row]),
                    int(split.pair_orders[reference]),
                )
                self.assertEqual(
                    int(split.supervised_positions[row]),
                    reference_position,
                )
                torch.testing.assert_close(
                    split.input_ids[row, : reference_position + 1],
                    split.input_ids[
                        reference, : reference_position + 1
                    ],
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(
                    int(split.answer_token_ids[row]),
                    int(split.answer_token_ids[reference]),
                )
                valid_length = int(split.valid_lengths[row])
                suffix_roles = split.token_role_ids[
                    row, reference_position + 1 : valid_length
                ]
                self.assertTrue(
                    bool(
                        (
                            suffix_roles
                            == int(VariableAssociativeTokenRole.FILLER)
                        ).all()
                    )
                )

    def test_valid_token_metadata_aligns_position_length_and_token_controls(
        self,
    ) -> None:
        split = build_variable_associative_recall_splits(
            small_config()
        ).validation
        metadata = split.valid_token_metadata()
        flat_valid = split.attention_mask.reshape(-1)

        self.assertEqual(
            metadata.observations,
            int(split.attention_mask.sum().item()),
        )
        self.assertTrue(
            torch.equal(
                metadata.selected_flat_indices,
                flat_valid.nonzero(as_tuple=False).flatten(),
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.token_ids,
                split.input_ids.reshape(-1)[flat_valid],
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.token_role_ids,
                split.token_role_ids.reshape(-1)[flat_valid],
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.is_supervised,
                metadata.logical_positions == metadata.supervised_positions,
            )
        )
        self.assertTrue(
            torch.equal(
                metadata.is_future,
                metadata.logical_positions > metadata.supervised_positions,
            )
        )
        self.assertEqual(
            int(metadata.is_supervised.sum().item()),
            split.samples,
        )
        self.assertTrue(
            bool(
                (
                    metadata.logical_positions
                    < metadata.valid_lengths
                ).all()
            )
        )

    def test_ids_and_hashes_are_stable_and_seed_independent_per_example(
        self,
    ) -> None:
        first = build_variable_associative_recall_splits(small_config())
        repeated = build_variable_associative_recall_splits(small_config())

        self.assertEqual(first.dataset_sha256, repeated.dataset_sha256)
        for left, right in zip(all_splits(first), all_splits(repeated)):
            self.assertTrue(torch.equal(left.input_ids, right.input_ids))
            self.assertTrue(
                torch.equal(left.attention_mask, right.attention_mask)
            )
            self.assertEqual(left.example_ids, right.example_ids)
            self.assertEqual(left.example_hashes, right.example_hashes)
            self.assertEqual(
                left.causal_prefix_hashes,
                right.causal_prefix_hashes,
            )
            self.assertEqual(left.content_sha256, right.content_sha256)

        changed_seed = build_variable_associative_recall_splits(
            small_config(split_seed=91)
        )
        self.assertNotEqual(first.dataset_sha256, changed_seed.dataset_sha256)

        def example_map(splits):
            return {
                example_id: example_hash
                for split in all_splits(splits)
                for example_id, example_hash in zip(
                    split.example_ids,
                    split.example_hashes,
                )
            }

        first_examples = example_map(first)
        changed_examples = example_map(changed_seed)
        self.assertEqual(first_examples, changed_examples)
        self.assertEqual(
            len(first_examples),
            len(set(first_examples)),
        )
        for value in (
            first.dataset_sha256,
            *first_examples.values(),
            *(
                context_hash
                for split in all_splits(first)
                for context_hash in split.semantic_context_hashes
            ),
        ):
            self.assertEqual(len(value), 64)
            int(value, 16)

    def test_calibration_batches_preserve_masks_targets_and_ids(self) -> None:
        split = build_variable_associative_recall_splits(
            small_config()
        ).validation
        batches = tuple(
            iter_variable_associative_calibration_batches(
                split,
                batch_size=37,
            )
        )

        self.assertEqual(sum(batch.batch_size for batch in batches), split.samples)
        self.assertEqual(
            tuple(
                example_id
                for batch in batches
                for example_id in batch.example_ids or ()
            ),
            split.example_ids,
        )
        self.assertTrue(
            torch.equal(
                torch.cat(
                    [batch.model_inputs["input_ids"] for batch in batches]
                ),
                split.input_ids,
            )
        )
        self.assertTrue(
            torch.equal(
                torch.cat(
                    [batch.model_inputs["attention_mask"] for batch in batches]
                ),
                split.attention_mask,
            )
        )
        self.assertTrue(
            torch.equal(
                torch.cat([batch.targets for batch in batches]),
                split.targets,
            )
        )
        self.assertTrue(
            torch.equal(
                torch.cat([batch.valid_positions for batch in batches]),
                split.attention_mask,
            )
        )

    def test_suffix_variants_are_invariant_at_the_answer_in_toy_model(
        self,
    ) -> None:
        torch.manual_seed(912)
        task = small_config()
        split = build_variable_associative_recall_splits(task).validation
        groups = defaultdict(list)
        for row, prefix_hash in enumerate(split.causal_prefix_hashes):
            groups[prefix_hash].append(row)
        rows = next(
            rows
            for rows in groups.values()
            if len({int(split.valid_lengths[row]) for row in rows}) > 1
        )
        model = ToyTransformer(
            variable_associative_recall_model_config(task)
        ).eval()
        selected = torch.tensor(rows)
        with torch.no_grad():
            output = model(
                split.input_ids.index_select(0, selected),
                attention_mask=split.attention_mask.index_select(0, selected),
            ).logits
        supervised_position = int(split.supervised_positions[rows[0]])
        answer_logits = output[:, supervised_position]

        torch.testing.assert_close(
            answer_logits,
            answer_logits[:1].expand_as(answer_logits),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_context_subset_keeps_whole_variants_and_remaps_rows(self) -> None:
        task = small_config()
        split = build_variable_associative_recall_splits(task).train
        selected = subset_variable_associative_recall_split(
            split,
            context_rows=torch.tensor([2, 0]),
            name="compiler_a",
        )
        other = subset_variable_associative_recall_split(
            split,
            context_rows=torch.tensor([1]),
            name="compiler_b",
        )

        self.assertEqual(selected.contexts, 2)
        self.assertEqual(selected.samples, 2 * task.variants_per_context)
        self.assertEqual(
            selected.example_context_indices.unique(sorted=True).tolist(),
            [0, 1],
        )
        self.assertEqual(
            selected.context_ids.tolist(),
            split.context_ids[torch.tensor([2, 0])].tolist(),
        )
        self.assertTrue(
            set(selected.example_hashes).isdisjoint(
                set(other.example_hashes)
            )
        )


if __name__ == "__main__":
    unittest.main()
