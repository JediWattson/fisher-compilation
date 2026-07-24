"""Distillation utilities for the variable-length causal modal executor.

The dynamic executor is deliberately independent of any particular model
adapter.  This module gives compiler frontends an equally small training
boundary: named pairs of source-segment input/output activations accompanied
by the exact :class:`~fisher_graph.adapters.base.SequenceContext` under which
the source segment ran.

Splitting happens at whole-batch granularity.  This preserves opaque adapter
payloads and makes length-held-out-from-training validation explicit, at the
cost of requiring callers to construct separate batches when one padded batch
mixes a training length with a validation length.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..adapters.base import SequenceContext
from ..dynamic_executor import VariableLengthCausalModalExecutor


def _require_nonempty_string(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_positive_integer(value: int, *, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_finite_number(
    value: float,
    *,
    name: str,
    minimum: float,
    inclusive: bool,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    invalid = numeric < minimum if inclusive else numeric <= minimum
    if invalid:
        relation = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {relation} {minimum}")


@dataclass(frozen=True, slots=True)
class TeacherBoundaryBatch:
    """One clean source-segment boundary batch.

    ``input_activations`` and ``output_activations`` are the exact source
    segment boundary values for ``sequence``.  They must be detached from the
    teacher graph; the fitter never updates or retains a teacher model.
    """

    batch_id: str
    input_activation: str
    output_activation: str
    input_activations: Tensor
    output_activations: Tensor
    sequence: SequenceContext

    def __post_init__(self) -> None:
        _require_nonempty_string(self.batch_id, name="batch_id")
        _require_nonempty_string(
            self.input_activation,
            name="input_activation",
        )
        _require_nonempty_string(
            self.output_activation,
            name="output_activation",
        )
        for name, values in (
            ("input_activations", self.input_activations),
            ("output_activations", self.output_activations),
        ):
            if not isinstance(values, Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if values.ndim != 3:
                raise ValueError(
                    f"{name} must have shape [batch, sequence, width]"
                )
            if not values.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if values.requires_grad:
                raise ValueError(
                    f"{name} must be detached from the teacher graph"
                )
        if self.input_activations.shape[:2] != (
            self.output_activations.shape[:2]
        ):
            raise ValueError(
                "teacher input/output batch and sequence dimensions must align"
            )
        if (
            self.input_activations.shape[0] == 0
            or self.input_activations.shape[1] == 0
            or self.input_activations.shape[2] == 0
            or self.output_activations.shape[2] == 0
        ):
            raise ValueError("teacher boundary tensors cannot be empty")
        if self.input_activations.device != self.output_activations.device:
            raise ValueError(
                "teacher input/output activations must share a device"
            )
        if self.input_activations.dtype != self.output_activations.dtype:
            raise ValueError(
                "teacher input/output activations must share a dtype"
            )
        if not torch.isfinite(self.input_activations).all():
            raise ValueError("teacher input activations must be finite")
        if not isinstance(self.sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        batch_size, query_length = self.input_activations.shape[:2]
        if self.sequence.batch_size != batch_size:
            raise ValueError(
                "sequence context and teacher activations must share a batch"
            )
        if (
            self.sequence.query_length != query_length
            or self.sequence.key_length != query_length
        ):
            raise ValueError(
                "dynamic prefill boundary pairs require equal input, query, "
                "and key sequence lengths"
            )
        if self.sequence.device != self.input_activations.device:
            raise ValueError(
                "sequence context and teacher activations must share a device"
            )
        if self.sequence.phase != "prefill":
            raise ValueError(
                "dynamic executor fitting currently supports prefill only"
            )
        if self.sequence.cache_state is not None:
            raise ValueError(
                "dynamic executor fitting does not accept cache state"
            )
        if self.sequence.cache_positions is not None:
            raise ValueError(
                "dynamic executor fitting does not accept cache positions"
            )
        if not self.sequence.query_valid_mask.any(dim=1).all():
            raise ValueError(
                "every teacher example must contain a valid query position"
            )
        valid_outputs = self.output_activations[
            self.sequence.query_valid_mask
        ]
        if not torch.isfinite(valid_outputs).all():
            raise ValueError(
                "teacher outputs must be finite at valid query positions"
            )
        if (
            self.sequence.logical_positions[
                self.sequence.query_valid_mask
            ]
            < 0
        ).any():
            raise ValueError(
                "valid query logical positions cannot be negative"
            )
        if (
            self.sequence.key_logical_positions[
                self.sequence.key_valid_mask
            ]
            < 0
        ).any():
            raise ValueError(
                "valid key logical positions cannot be negative"
            )
        for example in range(batch_size):
            positions = self.sequence.key_logical_positions[example][
                self.sequence.key_valid_mask[example]
            ]
            if positions.numel() > 1 and (positions[1:] <= positions[:-1]).any():
                raise ValueError(
                    "valid key logical positions must be strictly increasing"
                )

    @property
    def valid_query_lengths(self) -> tuple[int, ...]:
        """Valid-query count for each example in the batch."""

        return tuple(
            int(length)
            for length in self.sequence.query_valid_mask.sum(dim=1).tolist()
        )


@dataclass(frozen=True, slots=True)
class TeacherBoundarySplit:
    """A nonoverlapping train/validation selection of teacher batches."""

    training: tuple[TeacherBoundaryBatch, ...]
    validation: tuple[TeacherBoundaryBatch, ...]

    def __post_init__(self) -> None:
        if not self.training:
            raise ValueError("the training split cannot be empty")
        if not self.validation:
            raise ValueError("the validation split cannot be empty")
        all_batches = self.training + self.validation
        if any(
            not isinstance(batch, TeacherBoundaryBatch)
            for batch in all_batches
        ):
            raise TypeError(
                "training and validation entries must be TeacherBoundaryBatch"
            )
        identifiers = tuple(batch.batch_id for batch in all_batches)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                "teacher batch identifiers must be unique across the split"
            )

    @property
    def training_batch_ids(self) -> tuple[str, ...]:
        return tuple(batch.batch_id for batch in self.training)

    @property
    def validation_batch_ids(self) -> tuple[str, ...]:
        return tuple(batch.batch_id for batch in self.validation)


def split_teacher_boundary_batches(
    batches: Sequence[TeacherBoundaryBatch],
    *,
    validation_fraction: float = 0.2,
    seed: int = 314_159,
    held_out_lengths: Iterable[int] | None = None,
) -> TeacherBoundarySplit:
    """Select deterministic whole-batch train and validation sets.

    When ``held_out_lengths`` is supplied, every example in a validation batch
    must have one of those valid-query counts and no training batch may contain
    one.  A batch that mixes held-out and training lengths is rejected rather
    than silently leaking examples across the split.

    Without held-out lengths, batch IDs are ranked by a seeded SHA-256 digest.
    The result therefore does not depend on caller order, Python hash
    randomization, or global random-number-generator state.
    """

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    normalized = tuple(batches)
    if len(normalized) < 2:
        raise ValueError("at least two teacher batches are required")
    if any(
        not isinstance(batch, TeacherBoundaryBatch)
        for batch in normalized
    ):
        raise TypeError("batches must contain TeacherBoundaryBatch values")
    by_id = {batch.batch_id: batch for batch in normalized}
    if len(by_id) != len(normalized):
        raise ValueError("teacher batch identifiers must be unique")

    if held_out_lengths is not None:
        held_out = tuple(held_out_lengths)
        if not held_out:
            raise ValueError("held_out_lengths cannot be empty")
        if any(type(length) is not int or length <= 0 for length in held_out):
            raise ValueError(
                "held_out_lengths must contain positive integers"
            )
        if len(set(held_out)) != len(held_out):
            raise ValueError("held_out_lengths cannot contain duplicates")
        held_out_set = frozenset(held_out)
        training: list[TeacherBoundaryBatch] = []
        validation: list[TeacherBoundaryBatch] = []
        for batch in normalized:
            batch_lengths = frozenset(batch.valid_query_lengths)
            overlap = batch_lengths & held_out_set
            if overlap and not batch_lengths <= held_out_set:
                raise ValueError(
                    f"teacher batch {batch.batch_id!r} mixes held-out and "
                    "training lengths"
                )
            (validation if overlap else training).append(batch)
        observed_validation_lengths = {
            length
            for batch in validation
            for length in batch.valid_query_lengths
        }
        missing_lengths = held_out_set - observed_validation_lengths
        if missing_lengths:
            raise ValueError(
                "held_out_lengths were not observed in teacher batches: "
                + ", ".join(str(length) for length in sorted(missing_lengths))
            )
        return TeacherBoundarySplit(
            training=tuple(sorted(training, key=lambda batch: batch.batch_id)),
            validation=tuple(
                sorted(validation, key=lambda batch: batch.batch_id)
            ),
        )

    _require_finite_number(
        validation_fraction,
        name="validation_fraction",
        minimum=0.0,
        inclusive=False,
    )
    if float(validation_fraction) >= 1.0:
        raise ValueError("validation_fraction must be less than one")
    validation_count = max(
        1,
        min(
            len(normalized) - 1,
            int(round(len(normalized) * float(validation_fraction))),
        ),
    )

    def selection_key(batch: TeacherBoundaryBatch) -> tuple[bytes, str]:
        payload = f"{seed}:{batch.batch_id}".encode("utf-8")
        return hashlib.sha256(payload).digest(), batch.batch_id

    ranked = sorted(normalized, key=selection_key)
    validation_ids = {
        batch.batch_id for batch in ranked[:validation_count]
    }
    return TeacherBoundarySplit(
        training=tuple(
            sorted(
                (
                    batch
                    for batch in normalized
                    if batch.batch_id not in validation_ids
                ),
                key=lambda batch: batch.batch_id,
            )
        ),
        validation=tuple(
            sorted(
                (
                    batch
                    for batch in normalized
                    if batch.batch_id in validation_ids
                ),
                key=lambda batch: batch.batch_id,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class DynamicExecutorFitConfig:
    """Optimization settings for shared causal modal distillation."""

    steps: int = 1_000
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    evaluation_interval: int = 50
    seed: int = 314_159
    maximum_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        _require_positive_integer(self.steps, name="steps")
        _require_positive_integer(
            self.evaluation_interval,
            name="evaluation_interval",
        )
        _require_finite_number(
            self.learning_rate,
            name="learning_rate",
            minimum=0.0,
            inclusive=False,
        )
        _require_finite_number(
            self.weight_decay,
            name="weight_decay",
            minimum=0.0,
            inclusive=True,
        )
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        if self.maximum_gradient_norm is not None:
            _require_finite_number(
                self.maximum_gradient_norm,
                name="maximum_gradient_norm",
                minimum=0.0,
                inclusive=False,
            )


@dataclass(frozen=True, slots=True)
class DynamicFitPoint:
    """One full-dataset evaluation during fitting."""

    step: int
    train_mse: float
    validation_mse: float


@dataclass(frozen=True, slots=True)
class LengthFitMetric:
    """MSE aggregated over examples with one valid-query count."""

    length: int
    examples: int
    valid_query_positions: int
    mse: float


@dataclass(frozen=True, slots=True)
class BatchSelectionMetric:
    """How often one batch contributed an optimization update."""

    batch_id: str
    valid_query_positions: int
    optimization_steps: int


@dataclass(frozen=True, slots=True)
class DynamicExecutorFitReport:
    """Best-checkpoint metrics for a dynamic modal executor fit."""

    seed: int
    best_step: int
    initial_train_mse: float
    initial_validation_mse: float
    train_mse: float
    validation_mse: float
    last_step_validation_mse: float
    learned_parameters: int
    training_batch_ids: tuple[str, ...]
    validation_batch_ids: tuple[str, ...]
    training_batch_selection: tuple[BatchSelectionMetric, ...]
    train_by_length: tuple[LengthFitMetric, ...]
    validation_by_length: tuple[LengthFitMetric, ...]
    history: tuple[DynamicFitPoint, ...]


def valid_query_position_mse(
    predictions: Tensor,
    targets: Tensor,
    sequence: SequenceContext,
) -> Tensor:
    """Return feature-wise MSE over valid query positions only."""

    if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("predictions and targets must be Tensors")
    if predictions.ndim != 3 or targets.shape != predictions.shape:
        raise ValueError(
            "predictions and targets must share shape "
            "[batch, sequence, width]"
        )
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise ValueError("predictions and targets must be floating point")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must share a device")
    if not isinstance(sequence, SequenceContext):
        raise TypeError("sequence must be a SequenceContext")
    if sequence.query_valid_mask.shape != predictions.shape[:2]:
        raise ValueError(
            "sequence query mask must match predictions and targets"
        )
    if sequence.device != predictions.device:
        raise ValueError(
            "sequence context, predictions, and targets must share a device"
        )
    valid = sequence.query_valid_mask
    if not valid.any():
        raise ValueError("MSE requires at least one valid query position")
    selected_predictions = predictions[valid]
    selected_targets = targets[valid]
    if not torch.isfinite(selected_predictions).all():
        raise ValueError("predictions must be finite at valid query positions")
    if not torch.isfinite(selected_targets).all():
        raise ValueError("targets must be finite at valid query positions")
    return torch.mean((selected_predictions - selected_targets).square())


def _move_sequence(
    sequence: SequenceContext,
    *,
    device: torch.device,
) -> SequenceContext:
    return SequenceContext(
        query_valid_mask=sequence.query_valid_mask.to(device=device),
        key_valid_mask=sequence.key_valid_mask.to(device=device),
        logical_positions=sequence.logical_positions.to(device=device),
        key_logical_positions=sequence.key_logical_positions.to(device=device),
        cache_positions=None,
        phase=sequence.phase,
        input_origin=sequence.input_origin,
        cache_state=None,
        # The dynamic executor consumes only the normalized fields above.
        # Keeping the payload preserves provenance without interpreting it.
        adapter_payload=sequence.adapter_payload,
    )


@dataclass(frozen=True, slots=True)
class _PreparedBoundaryBatch:
    """One boundary batch cast for the executor, without a second full scan."""

    batch_id: str
    input_activations: Tensor
    output_activations: Tensor
    sequence: SequenceContext
    valid_query_lengths: tuple[int, ...]


def _validate_batch_for_executor(
    batch: TeacherBoundaryBatch,
    executor: VariableLengthCausalModalExecutor,
) -> None:
    config = executor.config
    if batch.input_activation != config.input_activation:
        raise ValueError(
            f"teacher batch {batch.batch_id!r} input activation does not "
            "match the executor"
        )
    if batch.output_activation != config.output_activation:
        raise ValueError(
            f"teacher batch {batch.batch_id!r} output activation does not "
            "match the executor"
        )
    if batch.input_activations.shape[2] != config.width:
        raise ValueError(
            f"teacher batch {batch.batch_id!r} input width does not match "
            "the executor"
        )
    if batch.output_activations.shape[2] != config.width:
        raise ValueError(
            f"teacher batch {batch.batch_id!r} output width does not match "
            "the executor"
        )


def _prepare_batch(
    batch: TeacherBoundaryBatch,
    executor: VariableLengthCausalModalExecutor,
) -> _PreparedBoundaryBatch:
    reference = executor.graph.state_input_weight
    return _PreparedBoundaryBatch(
        batch_id=batch.batch_id,
        input_activations=batch.input_activations.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        ),
        output_activations=batch.output_activations.detach().to(
            device=reference.device,
            dtype=reference.dtype,
        ),
        sequence=_move_sequence(
            batch.sequence,
            device=reference.device,
        ),
        valid_query_lengths=batch.valid_query_lengths,
    )


def _predict(
    executor: VariableLengthCausalModalExecutor,
    batch: _PreparedBoundaryBatch,
) -> Tensor:
    return executor.forward_context(
        batch.input_activations,
        sequence=batch.sequence,
        prefix="dynamic_fit",
    )


def _evaluate(
    executor: VariableLengthCausalModalExecutor,
    batches: tuple[TeacherBoundaryBatch, ...],
) -> tuple[float, tuple[LengthFitMetric, ...]]:
    total_squared_error = 0.0
    total_values = 0
    per_length: dict[int, list[float | int]] = {}
    executor.eval()
    with torch.no_grad():
        for source_batch in batches:
            # Keep the calibration corpus on its original device.  Only one
            # variable-length batch is resident on the executor device at a
            # time, which matters for large-model activation captures.
            batch = _prepare_batch(source_batch, executor)
            predictions = _predict(executor, batch)
            for example, length in enumerate(batch.valid_query_lengths):
                valid = batch.sequence.query_valid_mask[example]
                predicted = predictions[example, valid]
                target = batch.output_activations[example, valid]
                squared_error = (
                    predicted - target
                ).square().sum().item()
                if not math.isfinite(squared_error):
                    raise RuntimeError(
                        "dynamic fit produced non-finite valid-position error"
                    )
                values = predicted.numel()
                total_squared_error += squared_error
                total_values += values
                aggregate = per_length.setdefault(length, [0.0, 0, 0])
                aggregate[0] = float(aggregate[0]) + squared_error
                aggregate[1] = int(aggregate[1]) + 1
                aggregate[2] = int(aggregate[2]) + length
    if total_values == 0:
        raise RuntimeError("fit evaluation contained no valid target values")
    metrics = tuple(
        LengthFitMetric(
            length=length,
            examples=int(aggregate[1]),
            valid_query_positions=int(aggregate[2]),
            mse=float(aggregate[0])
            / (int(aggregate[2]) * executor.config.width),
        )
        for length, aggregate in sorted(per_length.items())
    )
    return total_squared_error / total_values, metrics


def _clone_state(
    executor: VariableLengthCausalModalExecutor,
) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in executor.state_dict().items()
    }


class _ValidPositionBatchSelector:
    """Deterministic smooth weighted round-robin over valid positions."""

    def __init__(
        self,
        batches: tuple[TeacherBoundaryBatch, ...],
        *,
        seed: int,
    ) -> None:
        self._weights = tuple(
            sum(batch.valid_query_lengths) for batch in batches
        )
        if any(weight <= 0 for weight in self._weights):
            raise ValueError(
                "training batches must contain valid query positions"
            )
        self._total_weight = sum(self._weights)
        self._credit = [0] * len(batches)
        self._counts = [0] * len(batches)
        ranked = sorted(
            range(len(batches)),
            key=lambda index: (
                hashlib.sha256(
                    f"{seed}:{batches[index].batch_id}".encode("utf-8")
                ).digest(),
                batches[index].batch_id,
            ),
        )
        self._tie_rank = {
            batch_index: rank
            for rank, batch_index in enumerate(ranked)
        }

    def next_index(self) -> int:
        for index, weight in enumerate(self._weights):
            self._credit[index] += weight
        maximum = max(self._credit)
        selected = min(
            (
                index
                for index, credit in enumerate(self._credit)
                if credit == maximum
            ),
            key=self._tie_rank.__getitem__,
        )
        self._credit[selected] -= self._total_weight
        self._counts[selected] += 1
        return selected

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(self._counts)


def fit_variable_length_causal_modal_executor(
    executor: VariableLengthCausalModalExecutor,
    split: TeacherBoundarySplit,
    *,
    config: DynamicExecutorFitConfig | None = None,
) -> tuple[
    VariableLengthCausalModalExecutor,
    DynamicExecutorFitReport,
]:
    """Fit ``executor`` in place and restore its best validation checkpoint.

    Batch selection uses deterministic smooth weighted round-robin, with each
    batch weighted by its valid-query count.  Since each selected batch loss
    is a mean over its valid scalar targets and all targets have the same
    width, this makes the optimization schedule match the globally pooled MSE
    reported below.  The seed resolves deterministic scheduling ties; global
    RNG state and validation evaluation cannot change the schedule.
    """

    if not isinstance(executor, VariableLengthCausalModalExecutor):
        raise TypeError(
            "executor must be a VariableLengthCausalModalExecutor"
        )
    if not isinstance(split, TeacherBoundarySplit):
        raise TypeError("split must be a TeacherBoundarySplit")
    options = config or DynamicExecutorFitConfig()
    if not isinstance(options, DynamicExecutorFitConfig):
        raise TypeError("config must be a DynamicExecutorFitConfig")

    training = split.training
    validation = split.validation
    for batch in training + validation:
        _validate_batch_for_executor(batch, executor)
    parameters = tuple(
        parameter
        for parameter in executor.parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("executor has no trainable parameters")

    optimizer = torch.optim.AdamW(
        parameters,
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    selector = _ValidPositionBatchSelector(
        training,
        seed=options.seed,
    )

    initial_train_mse, _ = _evaluate(executor, training)
    initial_validation_mse, _ = _evaluate(executor, validation)
    best_step = 0
    best_validation_mse = initial_validation_mse
    best_state = _clone_state(executor)
    history = [
        DynamicFitPoint(
            step=0,
            train_mse=initial_train_mse,
            validation_mse=initial_validation_mse,
        )
    ]

    for step in range(1, options.steps + 1):
        batch = _prepare_batch(
            training[selector.next_index()],
            executor,
        )

        executor.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = _predict(executor, batch)
        loss = valid_query_position_mse(
            predictions,
            batch.output_activations,
            batch.sequence,
        )
        loss.backward()
        if options.maximum_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                options.maximum_gradient_norm,
            )
        optimizer.step()

        if (
            step % options.evaluation_interval == 0
            or step == options.steps
        ):
            train_mse, _ = _evaluate(executor, training)
            validation_mse, _ = _evaluate(executor, validation)
            history.append(
                DynamicFitPoint(
                    step=step,
                    train_mse=train_mse,
                    validation_mse=validation_mse,
                )
            )
            if validation_mse < best_validation_mse:
                best_step = step
                best_validation_mse = validation_mse
                best_state = _clone_state(executor)

    executor.load_state_dict(best_state)
    executor.eval()
    train_mse, train_by_length = _evaluate(executor, training)
    validation_mse, validation_by_length = _evaluate(
        executor,
        validation,
    )
    report = DynamicExecutorFitReport(
        seed=options.seed,
        best_step=best_step,
        initial_train_mse=initial_train_mse,
        initial_validation_mse=initial_validation_mse,
        train_mse=train_mse,
        validation_mse=validation_mse,
        last_step_validation_mse=history[-1].validation_mse,
        learned_parameters=sum(
            parameter.numel() for parameter in parameters
        ),
        training_batch_ids=split.training_batch_ids,
        validation_batch_ids=split.validation_batch_ids,
        training_batch_selection=tuple(
            BatchSelectionMetric(
                batch_id=batch.batch_id,
                valid_query_positions=sum(batch.valid_query_lengths),
                optimization_steps=selection_count,
            )
            for batch, selection_count in zip(
                training,
                selector.counts,
                strict=True,
            )
        ),
        train_by_length=train_by_length,
        validation_by_length=validation_by_length,
        history=tuple(history),
    )
    return executor, report


__all__ = [
    "BatchSelectionMetric",
    "DynamicExecutorFitConfig",
    "DynamicExecutorFitReport",
    "DynamicFitPoint",
    "LengthFitMetric",
    "TeacherBoundaryBatch",
    "TeacherBoundarySplit",
    "fit_variable_length_causal_modal_executor",
    "split_teacher_boundary_batches",
    "valid_query_position_mse",
]
