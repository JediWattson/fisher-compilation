"""One-pass activation capture and bounded-memory Fisher-mode analysis."""

from __future__ import annotations

import math
from collections.abc import Collection, Generator, Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .adapters import ActivationSite
from .compiler.calibration import CalibrationBatch, ScoreObjective
from .instrumentation import InstrumentedModel, validate_instrumented_model
from .streaming_fisher import (
    StreamingActivationFisherEstimator,
    StreamingFisherResult,
)


@dataclass(frozen=True, slots=True)
class ActivationScoreGradientRows:
    """CPU activation and score-gradient rows from one independent sequence."""

    activations: Mapping[str, Tensor]
    score_gradients: Mapping[str, Tensor]
    logical_positions: Tensor
    loss: float
    example_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.activations, Mapping) or not self.activations:
            raise ValueError("activations must be a nonempty mapping")
        if not isinstance(self.score_gradients, Mapping):
            raise TypeError("score_gradients must be a mapping")
        names = tuple(self.activations)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("activation names must be nonempty strings")
        if set(self.score_gradients) != set(names):
            raise ValueError(
                "activations and score_gradients must name the same sites"
            )

        observation_counts = set()
        for name in names:
            activation = self.activations[name]
            gradient = self.score_gradients[name]
            if not isinstance(activation, Tensor) or not isinstance(
                gradient,
                Tensor,
            ):
                raise TypeError(
                    "activations and score_gradients must contain Tensors"
                )
            if activation.ndim != 2 or activation.shape[1] <= 0:
                raise ValueError(
                    f"{name!r} activation rows must have shape "
                    "[observations, width]"
                )
            if gradient.shape != activation.shape:
                raise ValueError(
                    f"{name!r} score-gradient rows must match activations"
                )
            if activation.shape[0] <= 0:
                raise ValueError(
                    "each sequence must contain at least one observation"
                )
            if (
                activation.device.type != "cpu"
                or gradient.device.type != "cpu"
            ):
                raise ValueError(
                    "activation and score-gradient rows must be on CPU"
                )
            if activation.dtype not in (torch.float32, torch.float64):
                raise ValueError(
                    "activation rows must use float32 or float64"
                )
            if gradient.dtype != activation.dtype:
                raise ValueError(
                    "activation and score-gradient rows must share a dtype"
                )
            if not torch.isfinite(activation).all():
                raise ValueError(f"{name!r} contains non-finite activations")
            if not torch.isfinite(gradient).all():
                raise ValueError(f"{name!r} contains non-finite gradients")
            observation_counts.add(activation.shape[0])
        if len(observation_counts) != 1:
            raise ValueError(
                "every activation site must contain the same observations"
            )
        observations = next(iter(observation_counts))
        if (
            not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.shape != (observations,)
            or self.logical_positions.device.type != "cpu"
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "logical_positions must be a CPU integer vector with one "
                "entry per observation"
            )
        if observations > 1 and not torch.all(
            self.logical_positions[1:] > self.logical_positions[:-1]
        ):
            raise ValueError(
                "logical_positions must be strictly increasing"
            )
        if (self.logical_positions < 0).any():
            raise ValueError("logical_positions must be nonnegative")
        if not isinstance(self.loss, float) or not math.isfinite(self.loss):
            raise ValueError("loss must be a finite float")
        if self.example_id is not None and (
            not isinstance(self.example_id, str) or not self.example_id
        ):
            raise ValueError("example_id must be a nonempty string when set")

    @property
    def observations(self) -> int:
        """Number of valid activation positions in this sequence."""

        return next(iter(self.activations.values())).shape[0]


@dataclass(frozen=True, slots=True)
class StreamingActivationFisherBasis:
    """A pooled activation center and low-rank streaming Fisher result."""

    activation_name: str
    mean: Tensor
    fisher: StreamingFisherResult
    sequences: int

    def __post_init__(self) -> None:
        if not isinstance(self.activation_name, str) or not self.activation_name:
            raise ValueError("activation_name must be a nonempty string")
        if self.activation_name != self.fisher.activation_name:
            raise ValueError(
                "basis and Fisher result must name the same activation"
            )
        if self.mean.shape != (self.fisher.width,):
            raise ValueError("mean must have shape [activation width]")
        if self.mean.device.type != "cpu":
            raise ValueError("streaming activation mean must be on CPU")
        if self.mean.dtype not in (torch.float32, torch.float64):
            raise ValueError("streaming activation mean must be float32 or float64")
        if not torch.isfinite(self.mean).all():
            raise ValueError("streaming activation mean must be finite")
        if type(self.sequences) is not int or self.sequences <= 0:
            raise ValueError("sequences must be positive")

    @property
    def observations(self) -> int:
        return self.fisher.observations

    @property
    def eigenvalues(self) -> Tensor:
        return self.fisher.eigenvalues

    @property
    def vectors(self) -> Tensor:
        return self.fisher.vectors

    def metadata(self) -> dict[str, object]:
        return {
            **self.fisher.metadata(),
            "sequences": self.sequences,
            "centering": "pooled_activation_mean",
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "activation_name": self.activation_name,
            "mean": self.mean,
            "sequences": self.sequences,
            "fisher": self.fisher.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StreamingActivationFisherBasis:
        expected = {
            "activation_name",
            "mean",
            "sequences",
            "fisher",
        }
        if set(state) != expected:
            raise ValueError(
                "streaming activation basis fields do not match "
                "format version 1"
            )
        mean = state["mean"]
        fisher_state = state["fisher"]
        if not isinstance(mean, Tensor):
            raise TypeError("streaming activation mean must be a Tensor")
        if not isinstance(fisher_state, Mapping):
            raise TypeError("streaming Fisher state must be a mapping")
        return cls(
            activation_name=str(state["activation_name"]),
            mean=mean,
            fisher=StreamingFisherResult.from_state_dict(fisher_state),
            sequences=int(state["sequences"]),
        )


@dataclass(frozen=True, slots=True)
class StreamingFisherCollection:
    """Low-rank Fisher bases collected without retaining calibration rows."""

    bases: Mapping[str, StreamingActivationFisherBasis]
    mean_loss: float
    sequences: int

    def __post_init__(self) -> None:
        if not self.bases:
            raise ValueError("streaming Fisher collection cannot be empty")
        if any(
            name != basis.activation_name
            for name, basis in self.bases.items()
        ):
            raise ValueError("basis mapping keys must match activation names")
        if any(basis.sequences != self.sequences for basis in self.bases.values()):
            raise ValueError("every basis must share the collection sequence count")
        if not torch.isfinite(torch.tensor(self.mean_loss)):
            raise ValueError("mean_loss must be finite")
        if type(self.sequences) is not int or self.sequences <= 0:
            raise ValueError("sequences must be positive")

    def metadata(self) -> dict[str, object]:
        return {
            "mean_loss": self.mean_loss,
            "sequences": self.sequences,
            "bases": {
                name: basis.metadata()
                for name, basis in self.bases.items()
            },
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "mean_loss": self.mean_loss,
            "sequences": self.sequences,
            "bases": {
                name: basis.state_dict()
                for name, basis in self.bases.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StreamingFisherCollection:
        expected = {
            "format_version",
            "mean_loss",
            "sequences",
            "bases",
        }
        if set(state) != expected:
            raise ValueError(
                "streaming Fisher collection fields do not match "
                "format version 1"
            )
        if state.get("format_version") != 1:
            raise ValueError("unsupported streaming Fisher collection format")
        raw_bases = state["bases"]
        if not isinstance(raw_bases, Mapping):
            raise TypeError("streaming Fisher bases must be a mapping")
        bases: dict[str, StreamingActivationFisherBasis] = {}
        for raw_name, raw_basis in raw_bases.items():
            if not isinstance(raw_name, str):
                raise TypeError("streaming Fisher basis names must be strings")
            if not isinstance(raw_basis, Mapping):
                raise TypeError("streaming Fisher basis state must be a mapping")
            bases[raw_name] = StreamingActivationFisherBasis.from_state_dict(
                raw_basis
            )
        return cls(
            bases=bases,
            mean_loss=float(state["mean_loss"]),
            sequences=int(state["sequences"]),
        )


def _detached_leaf(values: Tensor) -> Tensor:
    return values.detach().requires_grad_(True)


def _activation_site_width(site: ActivationSite) -> int:
    width = site.width
    if width is None:
        raise ValueError(
            f"{site.id!r} does not declare a modal activation width"
        )
    return width


def _prepare_activation_row_stream(
    model: InstrumentedModel,
    *,
    activation_names: Collection[str],
    score_objective: ScoreObjective,
    leaf_activation_name: str | None,
    accumulation_dtype: torch.dtype,
) -> tuple[tuple[str, ...], dict[str, ActivationSite]]:
    validate_instrumented_model(model)
    if not activation_names:
        raise ValueError("activation_names cannot be empty")
    if not callable(score_objective):
        raise TypeError("score_objective must be callable")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be float32 or float64")
    requested = tuple(dict.fromkeys(activation_names))
    if any(not isinstance(name, str) or not name for name in requested):
        raise TypeError("activation names must be nonempty strings")
    if leaf_activation_name is not None and leaf_activation_name not in requested:
        raise ValueError("leaf_activation_name must be one of activation_names")

    site_catalog = {site.id: site for site in model.activation_sites}
    sites = {}
    for name in requested:
        try:
            site = site_catalog[name]
        except KeyError as error:
            raise KeyError(f"unknown activation site: {name!r}") from error
        if not site.modal_eligible:
            raise ValueError(
                f"{name!r} is not a canonical "
                "[batch, sequence, feature] activation site"
            )
        _activation_site_width(site)
        sites[name] = site
    return requested, sites


def _sequence_activation_score_gradient_rows(
    model: InstrumentedModel,
    sample: CalibrationBatch,
    *,
    requested: tuple[str, ...],
    sites: Mapping[str, ActivationSite],
    score_objective: ScoreObjective,
    leaf_activation_name: str | None,
    accumulation_dtype: torch.dtype,
) -> ActivationScoreGradientRows:
    interventions = (
        None
        if leaf_activation_name is None
        else {leaf_activation_name: _detached_leaf}
    )
    run = model.forward(
        sample.model_inputs,
        capture_sites=requested,
        interventions=interventions,
        retain_gradients=False,
    )
    missing = set(requested) - set(run.activations)
    if missing:
        raise KeyError(f"unknown activation taps: {sorted(missing)}")
    selected = {name: run.activations[name] for name in requested}
    expected_grid = (1, sample.valid_positions.shape[1])
    for name, tensor in selected.items():
        if (
            tensor.ndim != 3
            or tensor.shape[:2] != expected_grid
            or tensor.shape[2] != _activation_site_width(sites[name])
        ):
            raise ValueError(
                f"{name!r} must have shape "
                "[batch, sequence, declared width]"
            )

    loss = score_objective(run, sample)
    if (
        not isinstance(loss, Tensor)
        or loss.ndim != 0
        or not loss.is_floating_point()
    ):
        raise TypeError(
            "score_objective must return a floating scalar Tensor"
        )
    if not torch.isfinite(loss):
        raise ValueError("score_objective returned a non-finite value")
    if not loss.requires_grad:
        hint = (
            "; set leaf_activation_name when source weights are frozen"
            if leaf_activation_name is None
            else ""
        )
        raise ValueError(
            "score_objective result is not differentiable"
            f"{hint}"
        )

    unique_tensors: list[Tensor] = []
    tensor_indices: dict[int, int] = {}
    for tensor in selected.values():
        tensor_id = id(tensor)
        if tensor_id not in tensor_indices:
            tensor_indices[tensor_id] = len(unique_tensors)
            unique_tensors.append(tensor)
    unique_gradients = torch.autograd.grad(
        loss,
        tuple(unique_tensors),
        allow_unused=False,
    )

    valid_positions = sample.valid_positions[0].to(
        device=run.sequence.device
    )
    sequence_valid = run.sequence.query_valid_mask[0].to(
        device=run.sequence.device
    )
    if (valid_positions & ~sequence_valid).any():
        raise ValueError(
            "calibration positions must be valid in the adapter sequence context"
        )
    if not valid_positions.any():
        raise ValueError("a calibration sample has no valid positions")

    activation_rows = {}
    gradient_rows = {}
    for name, tensor in selected.items():
        tensor_mask = valid_positions.to(device=tensor.device)
        activation = tensor.detach()[0, tensor_mask]
        gradient = unique_gradients[tensor_indices[id(tensor)]].detach()[
            0,
            tensor_mask,
        ]
        if not torch.isfinite(activation).all():
            raise ValueError(f"{name!r} contains non-finite activations")
        if not torch.isfinite(gradient).all():
            raise ValueError(f"{name!r} contains non-finite gradients")
        activation_rows[name] = activation.to(
            device="cpu",
            dtype=accumulation_dtype,
        )
        gradient_rows[name] = gradient.to(
            device="cpu",
            dtype=accumulation_dtype,
        )
    return ActivationScoreGradientRows(
        activations=activation_rows,
        score_gradients=gradient_rows,
        logical_positions=(
            run.sequence.logical_positions[0, valid_positions]
            .detach()
            .to(device="cpu", dtype=torch.int64)
        ),
        loss=float(loss.detach().item()),
        example_id=(
            None if sample.example_ids is None else sample.example_ids[0]
        ),
    )


def _iter_prepared_activation_score_gradient_rows(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    requested: tuple[str, ...],
    sites: Mapping[str, ActivationSite],
    score_objective: ScoreObjective,
    leaf_activation_name: str | None,
    accumulation_dtype: torch.dtype,
) -> Generator[ActivationScoreGradientRows, None, None]:
    module = model.module
    was_training = module.training
    module.eval()
    try:
        for batch in calibration_batches:
            if not isinstance(batch, CalibrationBatch):
                raise TypeError(
                    "calibration_batches must contain CalibrationBatch"
                )
            for batch_index in range(batch.batch_size):
                # Leave the grad-mode context before yielding. Otherwise a
                # consumer iterating under torch.no_grad() observes grad mode
                # enabled while this generator is suspended.
                with torch.enable_grad():
                    sequence_rows = (
                        _sequence_activation_score_gradient_rows(
                            model,
                            batch.sample(batch_index),
                            requested=requested,
                            sites=sites,
                            score_objective=score_objective,
                            leaf_activation_name=leaf_activation_name,
                            accumulation_dtype=accumulation_dtype,
                        )
                    )
                yield sequence_rows
    finally:
        module.train(was_training)


def iter_activation_score_gradient_rows(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    activation_names: Collection[str],
    score_objective: ScoreObjective,
    leaf_activation_name: str | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
) -> Generator[ActivationScoreGradientRows, None, None]:
    """Yield detached CPU activation and score-gradient rows per sequence.

    Every input sequence is differentiated independently using the scalar
    returned by ``score_objective``. Only positions selected by
    ``CalibrationBatch.valid_positions`` are emitted. The returned iterator
    restores the model's train/eval state when it is exhausted or closed.
    """

    requested, sites = _prepare_activation_row_stream(
        model,
        activation_names=activation_names,
        score_objective=score_objective,
        leaf_activation_name=leaf_activation_name,
        accumulation_dtype=accumulation_dtype,
    )
    return _iter_prepared_activation_score_gradient_rows(
        model,
        calibration_batches,
        requested=requested,
        sites=sites,
        score_objective=score_objective,
        leaf_activation_name=leaf_activation_name,
        accumulation_dtype=accumulation_dtype,
    )


def collect_streaming_fisher_modes(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    activation_names: Collection[str],
    score_objective: ScoreObjective,
    rank: int,
    sketch_rows: int | None = None,
    leaf_activation_name: str | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
    score_reduction: str = "sum",
    normalizer: str = "valid_activation_positions",
) -> StreamingFisherCollection:
    """Stream shared-width Fisher modes from independent sequence scores.

    This preserves the repository's existing scientific definition: every
    sequence is differentiated independently using a summed scalar score, and
    valid activation positions are pooled as width-dimensional rows.  It does
    not retain those rows.  For frozen large models, ``leaf_activation_name``
    can replace one boundary with a detached differentiable leaf, eliminating
    the autograd graph before that boundary while preserving its value and all
    suffix gradients. ``score_reduction`` and ``normalizer`` are explicit
    provenance labels for custom objectives; they do not alter the supplied
    scalar or row weights.
    """

    requested, sites = _prepare_activation_row_stream(
        model,
        activation_names=activation_names,
        score_objective=score_objective,
        leaf_activation_name=leaf_activation_name,
        accumulation_dtype=accumulation_dtype,
    )

    estimators = {
        name: StreamingActivationFisherEstimator(
            activation_name=name,
            rank=rank,
            width=_activation_site_width(site),
            sketch_rows=sketch_rows,
            accumulation_dtype=accumulation_dtype,
            score_reduction=score_reduction,
            normalizer=normalizer,
        )
        for name, site in sites.items()
    }
    activation_sums = {
        name: torch.zeros(
            _activation_site_width(site),
            dtype=accumulation_dtype,
        )
        for name, site in sites.items()
    }
    observation_counts = {name: 0 for name in requested}
    loss_total = 0.0
    sequence_count = 0
    rows = iter_activation_score_gradient_rows(
        model,
        calibration_batches,
        activation_names=requested,
        score_objective=score_objective,
        leaf_activation_name=leaf_activation_name,
        accumulation_dtype=accumulation_dtype,
    )
    try:
        for sequence_rows in rows:
            loss_total += sequence_rows.loss
            for name in requested:
                estimators[name].update(
                    sequence_rows.score_gradients[name]
                )
                activation_sums[name] += sequence_rows.activations[name].sum(
                    dim=0
                )
                observation_counts[name] += (
                    sequence_rows.activations[name].shape[0]
                )
            sequence_count += 1
    finally:
        close = getattr(rows, "close", None)
        if callable(close):
            close()

    if sequence_count == 0:
        raise ValueError("cannot collect scores from an empty calibration stream")
    bases = {}
    for name in requested:
        result = estimators[name].finalize()
        if result.observations != observation_counts[name]:
            raise RuntimeError(
                "activation and gradient observation counts diverged"
            )
        bases[name] = StreamingActivationFisherBasis(
            activation_name=name,
            mean=activation_sums[name] / observation_counts[name],
            fisher=result,
            sequences=sequence_count,
        )
    return StreamingFisherCollection(
        bases=bases,
        mean_loss=loss_total / sequence_count,
        sequences=sequence_count,
    )


__all__ = [
    "ActivationScoreGradientRows",
    "StreamingActivationFisherBasis",
    "StreamingFisherCollection",
    "collect_streaming_fisher_modes",
    "iter_activation_score_gradient_rows",
]
