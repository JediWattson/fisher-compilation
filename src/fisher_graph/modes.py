"""Full activation Fisher matrices and their compute-mode decompositions."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from .adapters import ModelAdapter, SegmentSpec, as_model_adapter
from .compiler.calibration import (
    CalibrationBatch,
    CausalLanguageModelNLL,
    ScoreObjective,
)
from .instrumentation import (
    InstrumentedModel,
    validate_instrumented_model,
)
from .layers import LayerExecutor
from .model import ToyTransformer


@dataclass(frozen=True, slots=True)
class ActivationGradientSamples:
    """Aligned activation and score-gradient rows for one named tap.

    A row is one valid sequence position and the columns are the activation's
    final (feature) dimension. ``locations`` stores ``[sequence, position]`` for
    every row. Treating positions as observations produces a shared width-wise
    Fisher basis that can be reused at every sequence position.
    """

    name: str
    activations: Tensor
    score_gradients: Tensor
    locations: Tensor
    sequences: int
    sequence_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.activations.ndim != 2:
            raise ValueError("activations must have shape [observations, width]")
        if self.activations.shape[0] == 0:
            raise ValueError("activation samples cannot be empty")
        if type(self.sequences) is not int or self.sequences <= 0:
            raise ValueError("sequences must be positive")
        if self.score_gradients.shape != self.activations.shape:
            raise ValueError("score_gradients must match the activation matrix")
        if self.locations.shape != (self.activations.shape[0], 2):
            raise ValueError("locations must have shape [observations, 2]")
        if not self.activations.is_floating_point():
            raise ValueError("activations must be floating point")
        if not self.score_gradients.is_floating_point():
            raise ValueError("score_gradients must be floating point")
        if not torch.isfinite(self.activations).all():
            raise ValueError("activations must be finite")
        if not torch.isfinite(self.score_gradients).all():
            raise ValueError("score_gradients must be finite")
        if self.locations.dtype not in (torch.int32, torch.int64):
            raise ValueError("locations must use an integer dtype")
        if (self.locations[:, 0] < 0).any():
            raise ValueError("sequence locations cannot be negative")
        if (self.locations[:, 0] >= self.sequences).any():
            raise ValueError("sequence location exceeds sequence count")
        if (self.locations[:, 1] < 0).any():
            raise ValueError("position locations cannot be negative")
        if self.sequence_ids is not None:
            if (
                type(self.sequence_ids) is not tuple
                or len(self.sequence_ids) != self.sequences
                or any(
                    not isinstance(sequence_id, str) or not sequence_id
                    for sequence_id in self.sequence_ids
                )
            ):
                raise ValueError(
                    "sequence_ids must contain one nonempty string per sequence"
                )
            if len(set(self.sequence_ids)) != len(self.sequence_ids):
                raise ValueError("sequence_ids must be unique")

    @property
    def observations(self) -> int:
        return self.activations.shape[0]

    @property
    def width(self) -> int:
        return self.activations.shape[1]


@dataclass(frozen=True, slots=True)
class ScoreGradientCollection:
    """Per-example score gradients collected from a model."""

    samples: Mapping[str, ActivationGradientSamples]
    mean_loss: float
    sequences: int


@dataclass(frozen=True, slots=True)
class FisherModeBasis:
    """Eigenbasis of a full width-wise empirical activation Fisher matrix."""

    activation_name: str
    mean: Tensor
    matrix: Tensor
    eigenvalues: Tensor
    vectors: Tensor
    observations: int
    sequences: int
    scope: str = "width_pooled"
    score_reduction: str = "sum"
    normalizer: str = "valid_activation_positions"
    position_means: Tensor | None = None

    def __post_init__(self) -> None:
        width = self.mean.numel()
        if self.mean.shape != (width,):
            raise ValueError("mean must be a one-dimensional feature vector")
        if self.matrix.shape != (width, width):
            raise ValueError("Fisher matrix must have shape [width, width]")
        if self.eigenvalues.shape != (width,):
            raise ValueError("eigenvalues must have shape [width]")
        if self.vectors.shape != (width, width):
            raise ValueError("vectors must have shape [width, width]")
        if self.position_means is not None:
            if (
                self.position_means.ndim != 2
                or self.position_means.shape[1] != width
            ):
                raise ValueError(
                    "position_means must have shape [sequence, width]"
                )
            if not torch.isfinite(self.position_means).all():
                raise ValueError("position_means must be finite")

    @property
    def width(self) -> int:
        return self.mean.numel()

    @property
    def fisher_trace(self) -> float:
        return self.eigenvalues.sum().item()

    @property
    def retained_curve(self) -> Tensor:
        total = self.eigenvalues.sum()
        if total <= 0:
            return torch.zeros_like(self.eigenvalues)
        return self.eigenvalues.cumsum(dim=0) / total

    def modes_for_fraction(self, fraction: float) -> int:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if self.eigenvalues.sum() <= 0:
            return self.width
        index = torch.searchsorted(
            self.retained_curve,
            torch.tensor(
                fraction,
                dtype=self.retained_curve.dtype,
                device=self.retained_curve.device,
            ),
        )
        return min(index.item() + 1, self.width)

    def retained_fraction(self, modes: int) -> float:
        if not 1 <= modes <= self.width:
            raise ValueError(f"modes must be between 1 and {self.width}")
        return self.retained_curve[modes - 1].item()

    def _centering_mean(
        self,
        values: Tensor,
        *,
        centering: Literal["pooled", "position"],
    ) -> Tensor:
        if centering == "pooled":
            return self.mean.to(device=values.device, dtype=values.dtype)
        if centering != "position":
            raise ValueError("centering must be 'pooled' or 'position'")
        if self.position_means is None:
            raise ValueError(
                "position centering requires position means in the basis"
            )
        if values.ndim < 2:
            raise ValueError(
                "position centering requires a sequence dimension"
            )
        if values.shape[-2] != self.position_means.shape[0]:
            raise ValueError(
                "value sequence length does not match position means"
            )
        return self.position_means.to(
            device=values.device,
            dtype=values.dtype,
        )

    def project(
        self,
        values: Tensor,
        *,
        modes: int | None = None,
        centering: Literal["pooled", "position"] = "pooled",
    ) -> Tensor:
        """Center and project values whose final dimension is ``width``."""

        mode_count = modes if modes is not None else self.width
        if not 1 <= mode_count <= self.width:
            raise ValueError(f"modes must be between 1 and {self.width}")
        if values.shape[-1] != self.width:
            raise ValueError(
                f"expected final dimension {self.width}, got {values.shape[-1]}"
            )
        mean = self._centering_mean(values, centering=centering)
        vectors = self.vectors[:, :mode_count].to(
            device=values.device, dtype=values.dtype
        )
        return (values - mean) @ vectors

    def reconstruct(
        self,
        coordinates: Tensor,
        *,
        centering: Literal["pooled", "position"] = "pooled",
    ) -> Tensor:
        """Reconstruct activation values from leading modal coordinates."""

        mode_count = coordinates.shape[-1]
        if not 1 <= mode_count <= self.width:
            raise ValueError(
                f"coordinate width must be between 1 and {self.width}"
            )
        vectors = self.vectors[:, :mode_count].to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        mean = self._centering_mean(
            coordinates,
            centering=centering,
        )
        return coordinates @ vectors.transpose(0, 1) + mean

    def state_dict(self) -> dict[str, object]:
        return {
            "activation_name": self.activation_name,
            "mean": self.mean,
            "matrix": self.matrix,
            "eigenvalues": self.eigenvalues,
            "vectors": self.vectors,
            "observations": self.observations,
            "sequences": self.sequences,
            "scope": self.scope,
            "score_reduction": self.score_reduction,
            "normalizer": self.normalizer,
            "position_means": self.position_means,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> FisherModeBasis:
        return cls(
            activation_name=str(state["activation_name"]),
            mean=state["mean"],  # type: ignore[arg-type]
            matrix=state["matrix"],  # type: ignore[arg-type]
            eigenvalues=state["eigenvalues"],  # type: ignore[arg-type]
            vectors=state["vectors"],  # type: ignore[arg-type]
            observations=int(state["observations"]),
            sequences=int(state["sequences"]),
            scope=str(state.get("scope", "width_pooled")),
            score_reduction=str(state.get("score_reduction", "sum")),
            normalizer=str(
                state.get("normalizer", "valid_activation_positions")
            ),
            position_means=state.get("position_means"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ModalTransition:
    """Position-coupled affine map between two Fisher mode bases."""

    input_activation: str
    output_activation: str
    sequence_length: int
    input_modes: int
    output_modes: int
    weights: Tensor
    bias: Tensor
    r_squared: float
    rmse: float

    def __post_init__(self) -> None:
        expected_weights = (
            self.sequence_length * self.input_modes,
            self.sequence_length * self.output_modes,
        )
        if self.weights.shape != expected_weights:
            raise ValueError(
                f"weights must have shape {expected_weights}, "
                f"got {tuple(self.weights.shape)}"
            )
        if self.bias.shape != (self.sequence_length * self.output_modes,):
            raise ValueError("bias shape does not match the output modal grid")

    def state_dict(self) -> dict[str, object]:
        return {
            "input_activation": self.input_activation,
            "output_activation": self.output_activation,
            "sequence_length": self.sequence_length,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "weights": self.weights,
            "bias": self.bias,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
        }

    def strongest_edges(self, count: int = 20) -> list[dict[str, float | int]]:
        """Return the largest position-mode couplings by absolute weight."""

        if count <= 0:
            return []
        flat = self.weights.abs().flatten()
        edge_count = min(count, flat.numel())
        indices = torch.topk(flat, edge_count).indices
        edges: list[dict[str, float | int]] = []
        for flat_index in indices.tolist():
            source_flat, target_flat = divmod(
                flat_index, self.weights.shape[1]
            )
            source_position, source_mode = divmod(
                source_flat, self.input_modes
            )
            target_position, target_mode = divmod(
                target_flat, self.output_modes
            )
            edges.append(
                {
                    "source_position": source_position,
                    "source_mode": source_mode,
                    "target_position": target_position,
                    "target_mode": target_mode,
                    "weight": self.weights[
                        source_flat, target_flat
                    ].item(),
                }
            )
        return edges


@dataclass(frozen=True, slots=True)
class ModalJacobian:
    """Mean and RMS token-to-token layer Jacobian in Fisher coordinates."""

    input_activation: str
    output_activation: str
    input_modes: int
    output_modes: int
    sequence_length: int
    samples: int
    mean: Tensor
    rms: Tensor

    def __post_init__(self) -> None:
        expected = (
            self.sequence_length,
            self.output_modes,
            self.sequence_length,
            self.input_modes,
        )
        if self.mean.shape != expected or self.rms.shape != expected:
            raise ValueError(
                "modal Jacobians must have shape "
                "[output_position, output_mode, input_position, input_mode]"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "input_activation": self.input_activation,
            "output_activation": self.output_activation,
            "input_modes": self.input_modes,
            "output_modes": self.output_modes,
            "sequence_length": self.sequence_length,
            "samples": self.samples,
            "mean": self.mean,
            "rms": self.rms,
        }

    def strongest_edges(self, count: int = 20) -> list[dict[str, float | int]]:
        """Rank modal edges by RMS magnitude without signed cancellation."""

        if count <= 0:
            return []
        flat = self.rms.flatten()
        edge_count = min(count, flat.numel())
        indices = torch.topk(flat, edge_count).indices
        edges: list[dict[str, float | int]] = []
        shape = self.rms.shape
        for flat_index in indices.tolist():
            output_position = flat_index // (shape[1] * shape[2] * shape[3])
            remainder = flat_index % (shape[1] * shape[2] * shape[3])
            output_mode = remainder // (shape[2] * shape[3])
            remainder %= shape[2] * shape[3]
            input_position, input_mode = divmod(remainder, shape[3])
            edges.append(
                {
                    "input_position": input_position,
                    "input_mode": input_mode,
                    "output_position": output_position,
                    "output_mode": output_mode,
                    "mean": self.mean[
                        output_position,
                        output_mode,
                        input_position,
                        input_mode,
                    ].item(),
                    "rms": self.rms[
                        output_position,
                        output_mode,
                        input_position,
                        input_mode,
                    ].item(),
                }
            )
        return edges


def collect_instrumented_score_gradients(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    activation_names: Collection[str],
    score_objective: ScoreObjective,
) -> ScoreGradientCollection:
    """Collect per-example score gradients through an instrumented model.

    Calibration batches may have different sequence lengths. Every sequence
    is still differentiated independently, avoiding cross-example terms and
    preserving the empirical-Fisher interpretation.  The model may be a
    source :class:`ModelAdapter` or a bound mixed/compiled runtime; activation
    layout metadata is read from the same typed ``activation_sites`` catalog in
    either case.
    """

    validate_instrumented_model(model)
    if not activation_names:
        raise ValueError("activation_names cannot be empty")
    if not callable(score_objective):
        raise TypeError("score_objective must be callable")

    requested = tuple(dict.fromkeys(activation_names))
    site_catalog = {site.id: site for site in model.activation_sites}
    activation_sites = {}
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
        activation_sites[name] = site
    activation_rows: dict[str, list[Tensor]] = {name: [] for name in requested}
    gradient_rows: dict[str, list[Tensor]] = {name: [] for name in requested}
    location_rows: dict[str, list[Tensor]] = {name: [] for name in requested}
    loss_total = 0.0
    sequence_count = 0
    sequence_ids: list[str] = []
    has_explicit_ids: bool | None = None
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
                    sample_has_id = sample.example_ids is not None
                    if has_explicit_ids is None:
                        has_explicit_ids = sample_has_id
                    elif has_explicit_ids != sample_has_id:
                        raise ValueError(
                            "calibration stream cannot mix identified and "
                            "unidentified examples"
                        )
                    if sample.example_ids is not None:
                        sequence_ids.append(sample.example_ids[0])
                    run = model.forward(
                        sample.model_inputs,
                        capture_sites=requested,
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
                    expected_grid = (
                        1,
                        sample.valid_positions.shape[1],
                    )
                    for name, tensor in selected.items():
                        if (
                            tensor.ndim != 3
                            or tensor.shape[:2] != expected_grid
                            or tensor.shape[2]
                            != activation_sites[name].width
                        ):
                            raise ValueError(
                                f"{name!r} must have shape "
                                "[batch, sequence, declared width] for "
                                "shared-width Fisher modes"
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
                        raise ValueError(
                            "score_objective result is not differentiable"
                        )

                    # Multiple names may alias one residual tensor. Differentiate
                    # every unique tensor once and share the resulting gradient.
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
                    positions = run.sequence.logical_positions[0].to(
                        device=valid_positions.device
                    )[valid_positions]
                    locations = torch.stack(
                        (
                            torch.full_like(positions, sequence_count),
                            positions,
                        ),
                        dim=1,
                    ).cpu()
                    for name, tensor in selected.items():
                        gradient = unique_gradients[
                            tensor_indices[id(tensor)]
                        ]
                        tensor_valid_positions = valid_positions.to(
                            device=tensor.device
                        )
                        activation_rows[name].append(
                            tensor.detach()[
                                0,
                                tensor_valid_positions,
                            ].cpu()
                        )
                        gradient_rows[name].append(
                            gradient.detach()[
                                0,
                                tensor_valid_positions,
                            ].cpu()
                        )
                        location_rows[name].append(locations)
                    sequence_count += 1
    finally:
        module.train(was_training)

    if sequence_count == 0:
        raise ValueError("cannot collect scores from an empty calibration stream")
    samples = {
        name: ActivationGradientSamples(
            name=name,
            activations=torch.cat(activation_rows[name], dim=0),
            score_gradients=torch.cat(gradient_rows[name], dim=0),
            locations=torch.cat(location_rows[name], dim=0),
            sequences=sequence_count,
            sequence_ids=(
                tuple(sequence_ids) if has_explicit_ids else None
            ),
        )
        for name in requested
    }
    return ScoreGradientCollection(
        samples=samples,
        mean_loss=loss_total / sequence_count,
        sequences=sequence_count,
    )


def collect_adapter_score_gradients(
    adapter: ModelAdapter,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    activation_names: Collection[str],
    score_objective: ScoreObjective,
) -> ScoreGradientCollection:
    """Compatibility facade for Fisher collection through a model adapter."""

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    return collect_instrumented_score_gradients(
        adapter,
        calibration_batches,
        activation_names=activation_names,
        score_objective=score_objective,
    )


def collect_activation_score_gradients(
    model: ToyTransformer | ModelAdapter,
    input_ids: Tensor,
    targets: Tensor,
    *,
    activation_names: Collection[str],
    attention_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> ScoreGradientCollection:
    """Compatibility facade for summed causal-language-model NLL scores."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] == 0:
        raise ValueError("cannot collect scores from an empty batch")
    if targets.shape != input_ids.shape:
        raise ValueError("targets must have the same shape as input_ids")
    if not activation_names:
        raise ValueError("activation_names cannot be empty")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    valid_positions = (
        attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.bool)
    )
    if ((targets != ignore_index) & ~valid_positions).any():
        raise ValueError(
            "supervised targets must be at attention-valid positions"
        )
    for sequence_index in range(input_ids.shape[0]):
        if not (targets[sequence_index] != ignore_index).any():
            raise ValueError(
                f"sequence {sequence_index} has no supervised target"
            )
    return collect_adapter_score_gradients(
        as_model_adapter(model),
        (
            CalibrationBatch(
                model_inputs={
                    "input_ids": input_ids,
                    "attention_mask": valid_positions,
                },
                targets=targets,
                valid_positions=valid_positions,
                example_ids=tuple(
                    f"sequence.{index}"
                    for index in range(input_ids.shape[0])
                ),
            ),
        ),
        activation_names=activation_names,
        score_objective=CausalLanguageModelNLL(
            ignore_index=ignore_index,
        ),
    )


def decompose_fisher_modes(
    samples: ActivationGradientSamples,
) -> FisherModeBasis:
    """Build and diagonalize ``G.T @ G / observations`` in float64."""

    gradients = samples.score_gradients.to(torch.float64)
    fisher = gradients.transpose(0, 1) @ gradients / samples.observations
    eigenvalues, vectors = torch.linalg.eigh(fisher)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    vectors = vectors.flip(1)

    # Eigenvector signs are arbitrary. Canonical signs make saved artifacts
    # stable across otherwise identical runs.
    pivots = vectors.abs().argmax(dim=0)
    columns = torch.arange(vectors.shape[1], device=vectors.device)
    signs = vectors[pivots, columns].sign()
    signs[signs == 0] = 1
    vectors = vectors * signs
    try:
        position_count = _validate_complete_location_grid(samples)
    except ValueError:
        # A width-pooled Fisher remains valid for ragged sequences and
        # nonzero logical-position offsets. Position-indexed means do not.
        position_means = None
    else:
        position_means = torch.stack(
            [
                samples.activations[
                    samples.locations[:, 1] == position
                ].to(torch.float64).mean(dim=0)
                for position in range(position_count)
            ]
        )

    return FisherModeBasis(
        activation_name=samples.name,
        mean=samples.activations.to(torch.float64).mean(dim=0),
        matrix=fisher,
        eigenvalues=eigenvalues,
        vectors=vectors,
        observations=samples.observations,
        sequences=samples.sequences,
        position_means=position_means,
    )


def build_fisher_mode_bases(
    collection: ScoreGradientCollection,
) -> dict[str, FisherModeBasis]:
    return {
        name: decompose_fisher_modes(samples)
        for name, samples in collection.samples.items()
    }


def _sequence_modal_matrix(
    samples: ActivationGradientSamples,
    basis: FisherModeBasis,
    modes: int,
) -> tuple[Tensor, int]:
    if samples.name != basis.activation_name:
        raise ValueError("samples and basis refer to different activation taps")
    if samples.width != basis.width:
        raise ValueError("sample width does not match the Fisher basis")
    coordinates = basis.project(
        samples.activations.to(torch.float64), modes=modes
    )
    sequence_length = _validate_complete_location_grid(samples)
    matrix = torch.empty(
        samples.sequences,
        sequence_length,
        modes,
        dtype=coordinates.dtype,
    )
    matrix[
        samples.locations[:, 0],
        samples.locations[:, 1],
    ] = coordinates
    return matrix.flatten(start_dim=1), sequence_length


def fit_modal_transition(
    input_samples: ActivationGradientSamples,
    output_samples: ActivationGradientSamples,
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    *,
    input_modes: int,
    output_modes: int,
    ridge: float = 1e-6,
) -> ModalTransition:
    """Fit a position-coupled affine map between modal activations.

    This is a dataset-local description of how a layer moves computation
    between Fisher modes. It is deliberately reported as a fitted map rather
    than a Jacobian: nonlinear layers need a later nonlinear graph executor.
    """

    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if input_samples.name != input_basis.activation_name:
        raise ValueError("input samples do not match the input basis")
    if output_samples.name != output_basis.activation_name:
        raise ValueError("output samples do not match the output basis")
    if input_samples.width != input_basis.width:
        raise ValueError("input sample width does not match the input basis")
    if output_samples.width != output_basis.width:
        raise ValueError("output sample width does not match the output basis")
    if not torch.equal(input_samples.locations, output_samples.locations):
        raise ValueError("input and output samples must have aligned locations")
    if input_samples.sequence_ids != output_samples.sequence_ids:
        raise ValueError(
            "input and output samples must have aligned sequence_ids"
        )

    inputs, input_length = _sequence_modal_matrix(
        input_samples, input_basis, input_modes
    )
    outputs, output_length = _sequence_modal_matrix(
        output_samples, output_basis, output_modes
    )
    if input_length != output_length:
        raise ValueError("input and output sequence lengths must match")

    ones = torch.ones(inputs.shape[0], 1, dtype=inputs.dtype)
    design = torch.cat((inputs, ones), dim=1)
    if ridge > 0:
        feature_count = inputs.shape[1]
        ridge_design = torch.zeros(
            feature_count,
            design.shape[1],
            dtype=design.dtype,
        )
        ridge_design[:, :feature_count] = (
            torch.eye(feature_count, dtype=design.dtype) * ridge**0.5
        )
        design = torch.cat((design, ridge_design), dim=0)
        outputs_for_fit = torch.cat(
            (
                outputs,
                torch.zeros(
                    feature_count,
                    outputs.shape[1],
                    dtype=outputs.dtype,
                ),
            ),
            dim=0,
        )
    else:
        outputs_for_fit = outputs
    coefficients = torch.linalg.lstsq(design, outputs_for_fit).solution
    weights = coefficients[:-1]
    bias = coefficients[-1]
    predictions = inputs @ weights + bias
    residual_sum = (outputs - predictions).square().sum()
    centered_sum = (outputs - outputs.mean(dim=0)).square().sum()
    r_squared = (
        1.0 - (residual_sum / centered_sum).item()
        if centered_sum > 0
        else 1.0
    )
    rmse = (outputs - predictions).square().mean().sqrt().item()

    return ModalTransition(
        input_activation=input_samples.name,
        output_activation=output_samples.name,
        sequence_length=input_length,
        input_modes=input_modes,
        output_modes=output_modes,
        weights=weights,
        bias=bias,
        r_squared=r_squared,
        rmse=rmse,
    )


def _sequence_activation_tensor(
    samples: ActivationGradientSamples,
    *,
    sequence_limit: int | None = None,
) -> Tensor:
    sequence_length = _validate_complete_location_grid(samples)
    result = torch.empty(
        samples.sequences,
        sequence_length,
        samples.width,
        dtype=samples.activations.dtype,
    )
    result[
        samples.locations[:, 0],
        samples.locations[:, 1],
    ] = samples.activations
    if sequence_limit is not None:
        result = result[:sequence_limit]
    return result


def _validate_complete_location_grid(
    samples: ActivationGradientSamples,
) -> int:
    """Return sequence length after validating a complete unique grid."""

    sequence_length = samples.locations[:, 1].max().item() + 1
    expected_locations = samples.sequences * sequence_length
    if samples.observations != expected_locations:
        raise ValueError(
            "modal maps currently require equal-length, unpadded sequences"
        )
    linear_locations = (
        samples.locations[:, 0] * sequence_length
        + samples.locations[:, 1]
    )
    expected = torch.arange(
        expected_locations,
        dtype=linear_locations.dtype,
        device=linear_locations.device,
    )
    if not torch.equal(linear_locations.sort().values, expected):
        raise ValueError(
            "locations must form a complete unique sequence-position grid"
        )
    return sequence_length


def extract_modal_jacobian(
    layer: LayerExecutor,
    input_samples: ActivationGradientSamples,
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    *,
    input_modes: int,
    output_modes: int,
    max_sequences: int | None = None,
) -> ModalJacobian:
    """Project sample-local layer JVPs into input/output Fisher modes.

    The result preserves token-to-token blocks. RMS is stored alongside the
    signed mean because context-dependent Jacobian edges can otherwise cancel.
    """

    if not 1 <= input_modes <= input_basis.width:
        raise ValueError("input_modes is outside the input basis")
    if not 1 <= output_modes <= output_basis.width:
        raise ValueError("output_modes is outside the output basis")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive when provided")
    if input_samples.name != input_basis.activation_name:
        raise ValueError("input samples do not match the input basis")
    if input_samples.width != input_basis.width:
        raise ValueError("input sample width does not match the input basis")
    sequence_inputs = _sequence_activation_tensor(
        input_samples, sequence_limit=max_sequences
    )
    reference_tensor = next(
        layer.parameters(),
        next(layer.buffers(), sequence_inputs),
    )
    sequence_inputs = sequence_inputs.to(
        device=reference_tensor.device,
        dtype=reference_tensor.dtype,
    )
    sample_count, sequence_length, width = sequence_inputs.shape
    input_vectors = input_basis.vectors[:, :input_modes].to(
        device=sequence_inputs.device,
        dtype=sequence_inputs.dtype,
    )
    output_vectors = output_basis.vectors[:, :output_modes].to(
        device=sequence_inputs.device,
        dtype=sequence_inputs.dtype,
    )
    mean = torch.zeros(
        sequence_length,
        output_modes,
        sequence_length,
        input_modes,
        dtype=torch.float64,
        device=sequence_inputs.device,
    )
    second_moment = torch.zeros_like(mean)
    was_training = layer.training
    layer.eval()

    try:
        for sample in sequence_inputs:
            hidden_states = sample.unsqueeze(0)
            attention_mask = torch.ones(
                1,
                sequence_length,
                dtype=torch.bool,
                device=sequence_inputs.device,
            )

            def layer_function(value: Tensor) -> Tensor:
                return layer(
                    value,
                    attention_mask=attention_mask,
                    trace=None,
                    prefix="jacobian",
                )

            for input_position in range(sequence_length):
                for input_mode in range(input_modes):
                    tangent = torch.zeros(
                        1,
                        sequence_length,
                        width,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    tangent[0, input_position] = input_vectors[:, input_mode]
                    _, output_tangent = torch.autograd.functional.jvp(
                        layer_function,
                        hidden_states,
                        tangent,
                        create_graph=False,
                        strict=True,
                    )
                    projected = (
                        output_tangent[0] @ output_vectors
                    ).to(torch.float64)
                    mean[
                        :,
                        :,
                        input_position,
                        input_mode,
                    ] += projected
                    second_moment[
                        :,
                        :,
                        input_position,
                        input_mode,
                    ] += projected.square()
    finally:
        layer.train(was_training)

    mean = (mean / sample_count).cpu()
    rms = (second_moment / sample_count).sqrt().cpu()
    return ModalJacobian(
        input_activation=input_basis.activation_name,
        output_activation=output_basis.activation_name,
        input_modes=input_modes,
        output_modes=output_modes,
        sequence_length=sequence_length,
        samples=sample_count,
        mean=mean,
        rms=rms,
    )


def extract_segment_modal_jacobian(
    adapter: ModelAdapter,
    segment: SegmentSpec,
    calibration_batches: Iterable[CalibrationBatch],
    input_samples: ActivationGradientSamples,
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    *,
    input_modes: int,
    output_modes: int,
    max_sequences: int | None = None,
) -> ModalJacobian:
    """Project adapter-owned segment JVPs into Fisher coordinates.

    Unlike :func:`extract_modal_jacobian`, this path never calls a concrete
    transformer block signature. The adapter prepares each sequence context
    and executes the requested segment, including model-specific mask,
    position, and cache semantics.

    The current ``ModalJacobian`` artifact is still a dense fixed-position
    tensor, so this function deliberately requires an equal-length,
    unpadded calibration grid. Variable-length Fisher collection is supported
    independently; a future dynamic graph backend will replace this dense
    artifact constraint. Stable example IDs bind each activation row to its
    sequence context even if calibration batches are replayed in a different
    order. This reference JVP path also fails explicitly for a sharded source
    segment; device-specific lowering remains adapter/backend work.
    """

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must implement ModelAdapter")
    if not isinstance(segment, SegmentSpec):
        raise TypeError("segment must be a SegmentSpec")
    if adapter.segment(segment.id) != segment:
        raise ValueError("segment specification does not match this adapter")
    if segment.input_site != input_basis.activation_name:
        raise ValueError("segment input site does not match the input basis")
    if segment.output_site != output_basis.activation_name:
        raise ValueError("segment output site does not match the output basis")
    if segment.input_width != input_basis.width:
        raise ValueError("segment input width does not match the input basis")
    if segment.output_width != output_basis.width:
        raise ValueError("segment output width does not match the output basis")
    if not 1 <= input_modes <= input_basis.width:
        raise ValueError("input_modes is outside the input basis")
    if not 1 <= output_modes <= output_basis.width:
        raise ValueError("output_modes is outside the output basis")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive when provided")
    if input_samples.name != input_basis.activation_name:
        raise ValueError("input samples do not match the input basis")
    if input_samples.width != input_basis.width:
        raise ValueError("input sample width does not match the input basis")

    sequence_inputs = _sequence_activation_tensor(
        input_samples,
        sequence_limit=max_sequences,
    )
    sample_limit = sequence_inputs.shape[0]
    if input_samples.sequence_ids is None:
        raise ValueError(
            "adapter segment Jacobians require stable sequence_ids"
        )
    target_ids = input_samples.sequence_ids[:sample_limit]
    needed_ids = set(target_ids)
    calibration_by_id: dict[str, CalibrationBatch] = {}
    for batch in calibration_batches:
        if not isinstance(batch, CalibrationBatch):
            raise TypeError(
                "calibration_batches must contain CalibrationBatch"
            )
        for batch_index in range(batch.batch_size):
            calibration = batch.sample(batch_index)
            if calibration.example_ids is None:
                raise ValueError(
                    "adapter segment Jacobians require calibration example_ids"
                )
            example_id = calibration.example_ids[0]
            if example_id in calibration_by_id:
                raise ValueError(
                    f"duplicate calibration example_id: {example_id!r}"
                )
            if example_id in needed_ids:
                calibration_by_id[example_id] = calibration
        if needed_ids.issubset(calibration_by_id):
            break
    missing_ids = needed_ids - set(calibration_by_id)
    if missing_ids:
        raise ValueError(
            "calibration stream is missing activation sequence ids: "
            f"{sorted(missing_ids)}"
        )
    calibration_samples = [
        calibration_by_id[example_id] for example_id in target_ids
    ]

    source = adapter.source_module(segment.layer_ids[0])
    floating_state = [
        tensor
        for tensor in (*source.parameters(), *source.buffers())
        if tensor.is_floating_point()
    ]
    if not floating_state:
        raise ValueError(
            "segment Jacobian extraction requires a floating-point "
            "single-device source segment"
        )
    source_devices = {tensor.device for tensor in floating_state}
    if len(source_devices) != 1:
        raise ValueError(
            "segment Jacobian extraction does not yet support sharded "
            "source segments"
        )
    sequence_inputs = sequence_inputs.to(device=source_devices.pop())
    sample_count, sequence_length, width = sequence_inputs.shape
    input_vectors = input_basis.vectors[:, :input_modes].to(
        device=sequence_inputs.device,
        dtype=sequence_inputs.dtype,
    )
    output_vectors = output_basis.vectors[:, :output_modes].to(
        device=sequence_inputs.device,
        dtype=sequence_inputs.dtype,
    )
    mean = torch.zeros(
        sequence_length,
        output_modes,
        sequence_length,
        input_modes,
        dtype=torch.float64,
        device=sequence_inputs.device,
    )
    second_moment = torch.zeros_like(mean)
    module = adapter.module
    was_training = module.training
    module.eval()

    try:
        for sample_index, sample in enumerate(sequence_inputs):
            calibration = calibration_samples[sample_index]
            if (
                calibration.valid_positions.shape != (1, sequence_length)
                or not calibration.valid_positions.all()
            ):
                raise ValueError(
                    "dense modal Jacobians require complete calibration rows"
                )
            model_inputs = {
                name: value.to(device=sequence_inputs.device)
                for name, value in calibration.model_inputs.items()
            }
            sequence = adapter.prepare_sequence(model_inputs)
            if (
                sequence.batch_size != 1
                or sequence.query_length != sequence_length
                or sequence.key_length != sequence_length
            ):
                raise ValueError(
                    "calibration context does not match the activation grid"
                )
            expected_positions = torch.arange(
                sequence_length,
                dtype=sequence.logical_positions.dtype,
                device=sequence.device,
            )
            if not torch.equal(
                sequence.logical_positions[0],
                expected_positions,
            ):
                raise ValueError(
                    "calibration logical positions do not match the "
                    "activation locations"
                )
            if (
                not sequence.query_valid_mask.all()
                or not sequence.key_valid_mask.all()
            ):
                raise ValueError(
                    "dense modal Jacobians require unpadded calibration sequences"
                )
            hidden_states = sample.unsqueeze(0)

            def segment_function(value: Tensor) -> Tensor:
                return adapter.run_segment(
                    segment,
                    value,
                    sequence,
                ).hidden_states

            for input_position in range(sequence_length):
                for input_mode in range(input_modes):
                    tangent = torch.zeros(
                        1,
                        sequence_length,
                        width,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                    tangent[0, input_position] = input_vectors[:, input_mode]
                    _, output_tangent = torch.autograd.functional.jvp(
                        segment_function,
                        hidden_states,
                        tangent,
                        create_graph=False,
                        strict=True,
                    )
                    projected = (
                        output_tangent[0] @ output_vectors
                    ).to(torch.float64)
                    mean[
                        :,
                        :,
                        input_position,
                        input_mode,
                    ] += projected
                    second_moment[
                        :,
                        :,
                        input_position,
                        input_mode,
                    ] += projected.square()
    finally:
        module.train(was_training)

    mean = (mean / sample_count).cpu()
    rms = (second_moment / sample_count).sqrt().cpu()
    return ModalJacobian(
        input_activation=input_basis.activation_name,
        output_activation=output_basis.activation_name,
        input_modes=input_modes,
        output_modes=output_modes,
        sequence_length=sequence_length,
        samples=sample_count,
        mean=mean,
        rms=rms,
    )


def save_fisher_build(
    path: str | Path,
    *,
    bases: Mapping[str, FisherModeBasis],
    transitions: Collection[ModalTransition],
    jacobians: Collection[ModalJacobian] = (),
    metadata: Mapping[str, object],
) -> None:
    """Save tensors and build metadata as a portable PyTorch artifact."""

    torch.save(
        {
            "format_version": 1,
            "bases": {
                name: basis.state_dict() for name, basis in bases.items()
            },
            "transitions": [
                transition.state_dict() for transition in transitions
            ],
            "jacobians": [
                jacobian.state_dict() for jacobian in jacobians
            ],
            "metadata": dict(metadata),
        },
        Path(path),
    )


def load_fisher_build(
    path: str | Path,
) -> tuple[
    dict[str, FisherModeBasis],
    list[ModalTransition],
    list[ModalJacobian],
    dict[str, object],
]:
    """Load an artifact written by :func:`save_fisher_build`."""

    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("unsupported Fisher build format")
    bases = {
        name: FisherModeBasis.from_state_dict(basis_state)
        for name, basis_state in state["bases"].items()
    }
    transitions = [
        ModalTransition(
            input_activation=str(item["input_activation"]),
            output_activation=str(item["output_activation"]),
            sequence_length=int(item["sequence_length"]),
            input_modes=int(item["input_modes"]),
            output_modes=int(item["output_modes"]),
            weights=item["weights"],
            bias=item["bias"],
            r_squared=float(item["r_squared"]),
            rmse=float(item["rmse"]),
        )
        for item in state["transitions"]
    ]
    jacobians = [
        ModalJacobian(
            input_activation=str(item["input_activation"]),
            output_activation=str(item["output_activation"]),
            input_modes=int(item["input_modes"]),
            output_modes=int(item["output_modes"]),
            sequence_length=int(item["sequence_length"]),
            samples=int(item["samples"]),
            mean=item["mean"],
            rms=item["rms"],
        )
        for item in state["jacobians"]
    ]
    return bases, transitions, jacobians, dict(state["metadata"])
