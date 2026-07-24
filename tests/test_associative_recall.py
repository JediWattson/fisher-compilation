import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fisher_graph import (
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    build_associative_recall_splits,
)


class _BufferOnlyModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.register_buffer("reference", torch.tensor(0.0))

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(
            logits=self.reference.new_zeros(
                *input_ids.shape,
                self.vocab_size,
            )
        )


class AssociativeRecallTaskTests(unittest.TestCase):
    def test_default_splits_are_grouped_disjoint_and_complete(self) -> None:
        task = AssociativeRecallTaskConfig()
        splits = build_associative_recall_splits(task)

        self.assertEqual(
            (splits.train.samples, splits.validation.samples, splits.test.samples),
            (5016, 628, 628),
        )
        self.assertEqual(
            (
                splits.train.contexts,
                splits.validation.contexts,
                splits.test.contexts,
            ),
            (2508, 314, 314),
        )
        train_ids = set(splits.train.context_ids.tolist())
        validation_ids = set(splits.validation.context_ids.tolist())
        test_ids = set(splits.test.context_ids.tolist())
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertEqual(
            len(train_ids | validation_ids | test_ids),
            task.context_count,
        )

    def test_both_query_variants_stay_in_each_context_group(self) -> None:
        split = build_associative_recall_splits().validation

        counts = torch.bincount(
            split.example_context_indices,
            minlength=split.contexts,
        )
        self.assertTrue(torch.equal(counts, torch.full_like(counts, 2)))
        for context_index in range(split.contexts):
            query_slots = split.query_slots[
                split.example_context_indices == context_index
            ]
            self.assertEqual(set(query_slots.tolist()), {0, 1})

    def test_answer_logits_support_a_frozen_buffer_only_runtime(self) -> None:
        task = AssociativeRecallTaskConfig()
        split = build_associative_recall_splits(task).validation
        model = _BufferOnlyModel(task.vocab_size)

        logits = associative_recall_answer_logits(
            model,
            split,
            batch_size=127,
        )

        self.assertEqual(logits.shape, (split.samples, task.vocab_size))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
