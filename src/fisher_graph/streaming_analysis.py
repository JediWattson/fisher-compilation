"""One-pass activation capture and bounded-memory Fisher-mode analysis."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .compiler.calibration import CalibrationBatch, ScoreObjective
from .instrumentation import InstrumentedModel, validate_instrumented_model
from .streaming_fisher import (
    StreamingActivationFisherEstimator,
    StreamingFisherResult,
)


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

    validate_instrumented_model(model)
    if not activation_names:
        raise ValueError("activation_names cannot be empty")
    if not callable(score_objective):
        raise TypeError("score_objective must be callable")
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
        assert site.width is not None
        sites[name] = site

    estimators = {
        name: StreamingActivationFisherEstimator(
            activation_name=name,
            rank=rank,
            width=site.width,
            sketch_rows=sketch_rows,
            accumulation_dtype=accumulation_dtype,
            score_reduction=score_reduction,
            normalizer=normalizer,
        )
        for name, site in sites.items()
    }
    activation_sums = {
        name: torch.zeros(site.width, dtype=accumulation_dtype)
        for name, site in sites.items()
    }
    observation_counts = {name: 0 for name in requested}
    loss_total = 0.0
    sequence_count = 0
    module = model.module
    was_training = module.training
    module.eval()

    try:
        with torch.enable_grad():
            for batch in calibration_batches:
                if not isinstance(batch, CalibrationBatch):
                    raise TypeError(
                        "calibration_batches must contain CalibrationBatch"
                    )
                for batch_index in range(batch.batch_size):
                    sample = batch.sample(batch_index)
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
                        raise KeyError(
                            f"unknown activation taps: {sorted(missing)}"
                        )
                    selected = {
                        name: run.activations[name] for name in requested
                    }
                    expected_grid = (1, sample.valid_positions.shape[1])
                    for name, tensor in selected.items():
                        if (
                            tensor.ndim != 3
                            or tensor.shape[:2] != expected_grid
                            or tensor.shape[2] != sites[name].width
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
                        raise ValueError(
                            "score_objective returned a non-finite value"
                        )
                    if not loss.requires_grad:
                        hint = (
                            "; set leaf_activation_name when source weights "
                            "are frozen"
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
                    loss_total += loss.detach().item()

                    valid_positions = sample.valid_positions[0].to(
                        device=run.sequence.device
                    )
                    sequence_valid = run.sequence.query_valid_mask[0].to(
                        device=run.sequence.device
                    )
                    if (valid_positions & ~sequence_valid).any():
                        raise ValueError(
                            "calibration positions must be valid in the "
                            "adapter sequence context"
                        )
                    if not valid_positions.any():
                        raise ValueError(
                            "a calibration sample has no valid positions"
                        )

                    for name, tensor in selected.items():
                        tensor_mask = valid_positions.to(device=tensor.device)
                        activation_rows = tensor.detach()[0, tensor_mask]
                        gradient = unique_gradients[
                            tensor_indices[id(tensor)]
                        ]
                        gradient_rows = gradient.detach()[0, tensor_mask]
                        if not torch.isfinite(activation_rows).all():
                            raise ValueError(
                                f"{name!r} contains non-finite activations"
                            )
                        if not torch.isfinite(gradient_rows).all():
                            raise ValueError(
                                f"{name!r} contains non-finite gradients"
                            )
                        estimators[name].update(gradient_rows)
                        activation_sums[name] += activation_rows.to(
                            device="cpu",
                            dtype=accumulation_dtype,
                        ).sum(dim=0)
                        observation_counts[name] += activation_rows.shape[0]
                    sequence_count += 1
    finally:
        module.train(was_training)

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
    "StreamingActivationFisherBasis",
    "StreamingFisherCollection",
    "collect_streaming_fisher_modes",
]
