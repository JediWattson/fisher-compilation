from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fisher_graph.config import TransformerConfig
from fisher_graph.model import ToyTransformer
from fisher_graph.variable_associative import (
    VariableAssociativeRecallTaskConfig,
    build_variable_associative_recall_splits,
)
from fisher_graph.variable_associative_training import (
    VariableAssociativeTrainingConfig,
    VariableAssociativeTrainingResult,
    build_parser,
    evaluate_variable_associative_recall,
    load_variable_associative_checkpoint,
    main,
    save_variable_associative_checkpoint,
    variable_associative_metrics_from_logits,
)


class VariableAssociativeTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = VariableAssociativeRecallTaskConfig(
            n_keys=3,
            n_values=3,
            train_fraction=0.5,
        )
        self.splits = build_variable_associative_recall_splits(self.task)

    def test_metrics_use_per_example_supervised_positions_and_strata(self) -> None:
        split = self.splits.validation
        logits = torch.full((split.samples, self.task.vocab_size), -10.0)
        logits[
            torch.arange(split.samples),
            split.answer_token_ids,
        ] = 10.0
        metrics = variable_associative_metrics_from_logits(split, logits)
        self.assertEqual(metrics.answer_accuracy, 1.0)
        self.assertEqual(metrics.paired_context_accuracy, 1.0)
        self.assertEqual(metrics.minimum_stratum_accuracy, 1.0)
        self.assertEqual(
            tuple(length for length, _ in metrics.length_accuracies),
            (8, 9, 10, 11, 12),
        )

    def test_evaluation_passes_attention_mask_and_gathers_answer_marker(self) -> None:
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=self.task.vocab_size,
                max_sequence_length=self.task.maximum_sequence_length,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
            )
        )
        metrics = evaluate_variable_associative_recall(
            model,
            self.splits.validation,
            batch_size=7,
        )
        self.assertEqual(metrics.samples, self.splits.validation.samples)
        self.assertTrue(math_is_finite(metrics.hard_nll))

    def test_checkpoint_round_trip_rebinds_dataset(self) -> None:
        model_config = TransformerConfig(
            vocab_size=self.task.vocab_size,
            max_sequence_length=self.task.maximum_sequence_length,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        model = ToyTransformer(model_config)
        metrics = evaluate_variable_associative_recall(
            model,
            self.splits.validation,
        )
        from fisher_graph.variable_associative_training import (
            VariableAssociativeCheckpoint,
        )

        checkpoint = VariableAssociativeCheckpoint(
            step=1,
            model_state_dict={
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            },
            train_metrics=metrics,
            validation_metrics=metrics,
        )
        result = VariableAssociativeTrainingResult(
            model=model,
            model_config=model_config,
            task_config=self.task,
            training_config=VariableAssociativeTrainingConfig(max_steps=1),
            splits=self.splits,
            best_checkpoint=checkpoint,
            history=(),
            final_step=1,
            converged=False,
            test_metrics=metrics,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_variable_associative_checkpoint(result, path)
            loaded, splits, metadata = load_variable_associative_checkpoint(path)
        self.assertEqual(splits.dataset_sha256, self.splits.dataset_sha256)
        self.assertEqual(metadata["best_step"], 1)
        for expected, actual in zip(
            model.parameters(),
            loaded.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(expected, actual)

    def test_parser_exposes_expanded_task_geometry(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--n-keys",
                "12",
                "--n-values",
                "14",
                "--split-seed",
                "76025",
            ]
        )
        self.assertEqual(arguments.n_keys, 12)
        self.assertEqual(arguments.n_values, 14)
        self.assertEqual(arguments.split_seed, 76025)

        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.n_keys, 8)
        self.assertEqual(defaults.n_values, 8)
        self.assertEqual(defaults.split_seed, 26_071)

    def test_main_forwards_task_geometry_and_reports_split_sizes(self) -> None:
        task = VariableAssociativeRecallTaskConfig(
            n_keys=3,
            n_values=3,
            split_seed=77,
            train_fraction=0.5,
        )
        splits = build_variable_associative_recall_splits(task)
        model = ToyTransformer(
            TransformerConfig(
                vocab_size=task.vocab_size,
                max_sequence_length=task.maximum_sequence_length,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
            )
        )
        metrics = evaluate_variable_associative_recall(
            model,
            splits.validation,
        )
        result = SimpleNamespace(
            task_config=task,
            splits=splits,
            converged=False,
            final_step=1,
            best_checkpoint=SimpleNamespace(
                step=1,
                train_metrics=metrics,
                validation_metrics=metrics,
            ),
            test_metrics=metrics,
        )
        output = Path("/tmp/expanded-variable-associative.pt")
        stdout = io.StringIO()
        with (
            patch(
                "fisher_graph.variable_associative_training."
                "run_variable_associative_training",
                return_value=result,
            ) as run_training,
            patch(
                "fisher_graph.variable_associative_training."
                "save_variable_associative_checkpoint",
                return_value=output,
            ),
            redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--n-keys",
                    "3",
                    "--n-values",
                    "3",
                    "--split-seed",
                    "77",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(exit_code, 0)
        configured_task = run_training.call_args.kwargs["task_config"]
        self.assertEqual(configured_task.n_keys, 3)
        self.assertEqual(configured_task.n_values, 3)
        self.assertEqual(configured_task.split_seed, 77)
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(
            receipt["task"],
            {
                "n_keys": 3,
                "n_values": 3,
                "split_seed": 77,
                "semantic_contexts": task.semantic_context_count,
                "variants_per_context": task.variants_per_context,
                "train_contexts": splits.train.contexts,
                "validation_contexts": splits.validation.contexts,
                "test_contexts": splits.test.contexts,
            },
        )


def math_is_finite(value: float) -> bool:
    return not (value != value or value in (float("inf"), float("-inf")))


if __name__ == "__main__":
    unittest.main()
