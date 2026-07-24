import unittest

import torch

from fisher_graph.adapters.base import SequenceContext, SequenceInputOrigin
from fisher_graph.compiler.dynamic_fit import (
    DynamicExecutorFitConfig,
    TeacherBoundaryBatch,
    TeacherBoundarySplit,
    fit_variable_length_causal_modal_executor,
    split_teacher_boundary_batches,
    valid_query_position_mse,
)
from fisher_graph.dynamic_executor import VariableLengthCausalModalExecutor
from fisher_graph.modes import FisherModeBasis


def identity_basis(width: int, name: str) -> FisherModeBasis:
    return FisherModeBasis(
        activation_name=name,
        mean=torch.zeros(width),
        matrix=torch.eye(width),
        eigenvalues=torch.arange(width, 0, -1, dtype=torch.float32),
        vectors=torch.eye(width),
        observations=20,
        sequences=4,
    )


def make_executor(*, seed: int) -> VariableLengthCausalModalExecutor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return VariableLengthCausalModalExecutor.from_bases(
            identity_basis(2, "segment.input"),
            identity_basis(2, "segment.output"),
            input_modes=2,
            output_modes=2,
            state_channels=1,
            routing_width=2,
            activation="identity",
        )


def make_context(
    *,
    batch_size: int,
    sequence_length: int,
    valid_lengths: tuple[int, ...] | None = None,
) -> SequenceContext:
    if valid_lengths is None:
        valid_lengths = (sequence_length,) * batch_size
    valid = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    for example, length in enumerate(valid_lengths):
        valid[example, :length] = True
    positions = torch.arange(sequence_length).unsqueeze(0).expand(
        batch_size,
        -1,
    )
    return SequenceContext(
        query_valid_mask=valid,
        key_valid_mask=valid,
        logical_positions=positions,
        key_logical_positions=positions,
        cache_positions=None,
        phase="prefill",
        input_origin=SequenceInputOrigin(
            attention_mask_supplied=True,
            position_ids_supplied=True,
            cache_positions_supplied=False,
        ),
        cache_state=None,
        adapter_payload={"opaque": True},
    )


def teacher_batch(
    teacher: VariableLengthCausalModalExecutor,
    *,
    batch_id: str,
    sequence_length: int,
    batch_size: int,
    seed: int,
) -> TeacherBoundaryBatch:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        batch_size,
        sequence_length,
        2,
        generator=generator,
    )
    context = make_context(
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    with torch.no_grad():
        outputs = teacher.forward_context(
            inputs,
            sequence=context,
            prefix="teacher",
        )
    return TeacherBoundaryBatch(
        batch_id=batch_id,
        input_activation="segment.input",
        output_activation="segment.output",
        input_activations=inputs,
        output_activations=outputs,
        sequence=context,
    )


class DynamicFitTests(unittest.TestCase):
    def test_teacher_batch_rejects_live_teacher_graph(self) -> None:
        context = make_context(batch_size=1, sequence_length=2)
        inputs = torch.randn(1, 2, 2, requires_grad=True)

        with self.assertRaisesRegex(ValueError, "detached"):
            TeacherBoundaryBatch(
                batch_id="live-teacher",
                input_activation="segment.input",
                output_activation="segment.output",
                input_activations=inputs,
                output_activations=torch.zeros(1, 2, 2),
                sequence=context,
            )

    def test_valid_query_mse_ignores_padding(self) -> None:
        context = make_context(
            batch_size=2,
            sequence_length=4,
            valid_lengths=(2, 3),
        )
        predictions = torch.zeros(2, 4, 2)
        targets = torch.ones(2, 4, 2)
        targets[~context.query_valid_mask] = 10_000

        loss = valid_query_position_mse(
            predictions,
            targets,
            context,
        )

        self.assertEqual(loss.item(), 1.0)

    def test_seeded_fraction_split_is_input_order_independent(self) -> None:
        teacher = make_executor(seed=1).eval()
        batches = tuple(
            teacher_batch(
                teacher,
                batch_id=f"batch-{index}",
                sequence_length=2 + index,
                batch_size=2,
                seed=100 + index,
            )
            for index in range(5)
        )

        first = split_teacher_boundary_batches(
            batches,
            validation_fraction=0.4,
            seed=91,
        )
        second = split_teacher_boundary_batches(
            tuple(reversed(batches)),
            validation_fraction=0.4,
            seed=91,
        )

        self.assertEqual(
            first.training_batch_ids,
            second.training_batch_ids,
        )
        self.assertEqual(
            first.validation_batch_ids,
            second.validation_batch_ids,
        )
        self.assertEqual(len(first.validation), 2)
        self.assertTrue(
            set(first.training_batch_ids).isdisjoint(
                first.validation_batch_ids
            )
        )
        with self.assertRaisesRegex(ValueError, "not observed"):
            split_teacher_boundary_batches(
                batches,
                held_out_lengths=(6, 999),
            )

    def test_valid_position_weighted_batch_selection_matches_report(self) -> None:
        teacher = make_executor(seed=3).eval()
        small = teacher_batch(
            teacher,
            batch_id="small",
            sequence_length=1,
            batch_size=1,
            seed=20,
        )
        large = teacher_batch(
            teacher,
            batch_id="large",
            sequence_length=3,
            batch_size=3,
            seed=21,
        )
        validation = teacher_batch(
            teacher,
            batch_id="validation",
            sequence_length=2,
            batch_size=1,
            seed=22,
        )

        _, report = fit_variable_length_causal_modal_executor(
            make_executor(seed=4),
            TeacherBoundarySplit(
                training=(small, large),
                validation=(validation,),
            ),
            config=DynamicExecutorFitConfig(
                steps=100,
                learning_rate=1e-3,
                evaluation_interval=100,
                seed=12,
            ),
        )

        selection = {
            metric.batch_id: metric
            for metric in report.training_batch_selection
        }
        self.assertEqual(selection["small"].valid_query_positions, 1)
        self.assertEqual(selection["large"].valid_query_positions, 9)
        self.assertEqual(selection["small"].optimization_steps, 10)
        self.assertEqual(selection["large"].optimization_steps, 90)

    def test_best_validation_state_is_restored(self) -> None:
        student = make_executor(seed=5)
        with torch.no_grad():
            for parameter in student.graph.parameters():
                parameter.zero_()
            student.graph.output_bias.fill_(0.1)
        context = make_context(batch_size=1, sequence_length=2)

        def zero_target_batch(batch_id: str) -> TeacherBoundaryBatch:
            return TeacherBoundaryBatch(
                batch_id=batch_id,
                input_activation="segment.input",
                output_activation="segment.output",
                input_activations=torch.zeros(1, 2, 2),
                output_activations=torch.zeros(1, 2, 2),
                sequence=context,
            )

        _, report = fit_variable_length_causal_modal_executor(
            student,
            TeacherBoundarySplit(
                training=(zero_target_batch("train"),),
                validation=(zero_target_batch("validation"),),
            ),
            config=DynamicExecutorFitConfig(
                steps=1,
                learning_rate=1.0,
                weight_decay=0.0,
                evaluation_interval=1,
                seed=8,
            ),
        )

        self.assertEqual(report.best_step, 0)
        self.assertGreater(
            report.last_step_validation_mse,
            report.validation_mse,
        )
        torch.testing.assert_close(
            student.graph.output_bias,
            torch.full_like(student.graph.output_bias, 0.1),
        )

    def test_same_architecture_fit_improves_held_out_validation_lengths(
        self,
    ) -> None:
        # This is a fitter contract test against a representable synthetic
        # teacher. It is not evidence that an arbitrary transformer segment
        # has already been compiled.
        teacher = make_executor(seed=7).eval()
        with torch.no_grad():
            teacher.graph.state_input_weight.copy_(
                torch.tensor([[[0.8, -0.4], [0.3, 0.9]]])
            )
            teacher.graph.hidden_weight.copy_(
                torch.tensor([[1.0, 0.2], [-0.3, 0.7]])
            )
            teacher.graph.hidden_bias.copy_(torch.tensor([0.1, -0.2]))
            teacher.graph.output_weight.copy_(
                torch.tensor([[0.6, -0.5], [0.4, 0.8]])
            )
            teacher.graph.output_bias.copy_(torch.tensor([-0.1, 0.05]))

        batches = tuple(
            teacher_batch(
                teacher,
                batch_id=f"length-{length}",
                sequence_length=length,
                batch_size=10,
                seed=1_000 + length,
            )
            for length in (2, 3, 4, 5, 6)
        )
        split = split_teacher_boundary_batches(
            batches,
            held_out_lengths=(5, 6),
        )
        student = make_executor(seed=99)

        fitted, report = fit_variable_length_causal_modal_executor(
            student,
            split,
            config=DynamicExecutorFitConfig(
                steps=300,
                learning_rate=1e-2,
                weight_decay=0.0,
                evaluation_interval=20,
                seed=123,
            ),
        )

        self.assertIs(fitted, student)
        self.assertEqual(
            {metric.length for metric in report.train_by_length},
            {2, 3, 4},
        )
        self.assertEqual(
            {metric.length for metric in report.validation_by_length},
            {5, 6},
        )
        self.assertEqual(
            report.validation_mse,
            min(point.validation_mse for point in report.history),
        )
        self.assertEqual(
            report.best_step,
            min(
                report.history,
                key=lambda point: point.validation_mse,
            ).step,
        )
        self.assertLess(
            report.validation_mse,
            report.initial_validation_mse * 0.2,
        )
        self.assertGreater(report.learned_parameters, 0)
        self.assertFalse(fitted.training)


if __name__ == "__main__":
    unittest.main()
