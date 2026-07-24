"""Causal completion of discarded Fisher coordinates at layer boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .layers import LayerExecutor
from .modal_executor import (
    CausalModalGraph,
    CausalModalMLPGraph,
    PositionConditionedModalGraphExecutor,
    PositionConditionedModalProjection,
)
from .modes import FisherModeBasis

ModalCompletionKind = Literal[
    "shared_local_linear",
    "position_local_linear",
]


@dataclass(frozen=True, slots=True)
class ModalCompletionConfig:
    """Portable shape and topology of one modal completion bridge."""

    activation_name: str
    sequence_length: int
    width: int
    kept_modes: int
    graph_kind: ModalCompletionKind

    def __post_init__(self) -> None:
        if not self.activation_name:
            raise ValueError("completion activation name cannot be empty")
        if self.sequence_length <= 0 or self.width <= 1:
            raise ValueError("completion dimensions must be positive")
        if not 1 <= self.kept_modes < self.width:
            raise ValueError("kept_modes must be between 1 and width - 1")
        if self.graph_kind not in (
            "shared_local_linear",
            "position_local_linear",
        ):
            raise ValueError("unsupported modal completion graph kind")

    @property
    def tail_modes(self) -> int:
        return self.width - self.kept_modes


@dataclass(frozen=True, slots=True)
class ModalCompletionFitConfig:
    """Deterministic standardized-ridge fitting options."""

    ridge: float = 1e-4
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if not torch.isfinite(torch.tensor(self.ridge)) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if (
            not torch.isfinite(torch.tensor(self.minimum_scale))
            or self.minimum_scale <= 0
        ):
            raise ValueError("minimum_scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class ModalCompletionFitReport:
    """Fit-set diagnostics for one analytic completion map."""

    samples: int
    observations: int
    train_tail_r_squared: float
    train_tail_rmse: float
    minimum_nonconstant_position_r_squared: float
    constant_position_count: int
    learned_parameters: int
    map_multiplies_per_sequence: int
    fit_config: ModalCompletionFitConfig


class LocalModalCompletionGraph(nn.Module):
    """A same-position modal map with shared or position-specific weights."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor,
        *,
        shared_weights: bool,
    ) -> None:
        super().__init__()
        if bias.ndim != 2:
            raise ValueError(
                "completion bias must have shape [sequence, tail_modes]"
            )
        sequence_length, tail_modes = bias.shape
        if min(sequence_length, tail_modes) <= 0:
            raise ValueError("completion graph dimensions must be positive")
        if shared_weights:
            if weight.ndim != 2 or weight.shape[1] != tail_modes:
                raise ValueError(
                    "shared completion weight must have shape "
                    "[kept_modes, tail_modes]"
                )
            input_modes = weight.shape[0]
        else:
            if (
                weight.ndim != 3
                or weight.shape[0] != sequence_length
                or weight.shape[2] != tail_modes
            ):
                raise ValueError(
                    "position completion weight must have shape "
                    "[sequence, kept_modes, tail_modes]"
                )
            input_modes = weight.shape[1]
        if input_modes <= 0:
            raise ValueError("completion graph must read at least one mode")
        if (
            not weight.is_floating_point()
            or not bias.is_floating_point()
            or weight.dtype != bias.dtype
            or weight.device != bias.device
        ):
            raise ValueError(
                "completion weight and bias must share a floating dtype/device"
            )
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("completion graph parameters must be finite")
        self.shared_weights = shared_weights
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone())

    @property
    def sequence_length(self) -> int:
        return self.bias.shape[0]

    @property
    def input_modes(self) -> int:
        return self.weight.shape[0] if self.shared_weights else self.weight.shape[1]

    @property
    def output_modes(self) -> int:
        return self.bias.shape[1]

    @property
    def graph_kind(self) -> ModalCompletionKind:
        return (
            "shared_local_linear"
            if self.shared_weights
            else "position_local_linear"
        )

    @property
    def edge_count(self) -> int:
        """Scalar connections executed across every sequence position."""

        return (
            self.sequence_length * self.input_modes * self.output_modes
        )

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        coordinates: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if coordinates.ndim != 3 or coordinates.shape[1:] != (
            self.sequence_length,
            self.input_modes,
        ):
            raise ValueError(
                "completion coordinates must have shape "
                "[batch, sequence, kept_modes]"
            )
        if attention_mask is not None:
            if attention_mask.shape != coordinates.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]"
                )
            if not attention_mask.to(torch.bool).all():
                raise ValueError(
                    "the fixed position-conditioned completion does not "
                    "support padding"
                )
        if (
            coordinates.dtype != self.weight.dtype
            or coordinates.device != self.weight.device
        ):
            raise ValueError(
                "completion inputs must match the graph dtype and device"
            )
        if self.shared_weights:
            return coordinates @ self.weight + self.bias
        return (
            torch.einsum("bti,tio->bto", coordinates, self.weight)
            + self.bias
        )


class PositionConditionedModalCompletion(nn.Module):
    """Predict discarded modes and lift retained modes back to full width."""

    def __init__(
        self,
        full_projection: PositionConditionedModalProjection,
        graph: LocalModalCompletionGraph,
    ) -> None:
        super().__init__()
        if full_projection.modes != full_projection.width:
            raise ValueError(
                "modal completion requires a complete square modal basis"
            )
        if graph.sequence_length != full_projection.sequence_length:
            raise ValueError(
                "completion graph and projection sequence lengths differ"
            )
        if graph.input_modes + graph.output_modes != full_projection.width:
            raise ValueError(
                "kept and tail modes must span the complete modal basis"
            )
        vectors = full_projection.vectors.detach()
        identity = torch.eye(
            vectors.shape[1],
            dtype=vectors.dtype,
            device=vectors.device,
        )
        tolerance = max(
            1e-9,
            torch.finfo(vectors.dtype).eps * vectors.shape[1] * 4,
        )
        if not torch.allclose(
            vectors.transpose(0, 1) @ vectors,
            identity,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError("completion modal vectors must be orthonormal")
        self.full_projection = full_projection
        self.graph = graph

    @classmethod
    def from_basis(
        cls,
        basis: FisherModeBasis,
        graph: LocalModalCompletionGraph,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> PositionConditionedModalCompletion:
        return cls(
            PositionConditionedModalProjection.from_basis(
                basis,
                modes=basis.width,
                dtype=dtype,
            ),
            graph.to(dtype=dtype),
        )

    @property
    def activation_name(self) -> str:
        return self.full_projection.activation_name

    @property
    def sequence_length(self) -> int:
        return self.full_projection.sequence_length

    @property
    def width(self) -> int:
        return self.full_projection.width

    @property
    def kept_modes(self) -> int:
        return self.graph.input_modes

    @property
    def tail_modes(self) -> int:
        return self.graph.output_modes

    @property
    def config(self) -> ModalCompletionConfig:
        return ModalCompletionConfig(
            activation_name=self.activation_name,
            sequence_length=self.sequence_length,
            width=self.width,
            kept_modes=self.kept_modes,
            graph_kind=self.graph.graph_kind,
        )

    def encode_kept(self, activations: Tensor) -> Tensor:
        self.full_projection._validate_values(
            activations,
            width=self.width,
        )
        mean = self.full_projection.position_mean.to(
            device=activations.device,
            dtype=activations.dtype,
        )
        vectors = self.full_projection.vectors[
            :, : self.kept_modes
        ].to(
            device=activations.device,
            dtype=activations.dtype,
        )
        return (activations - mean) @ vectors

    def encode_tail(self, activations: Tensor) -> Tensor:
        return self.full_projection.encode(activations)[
            ..., self.kept_modes :
        ]

    def complete(
        self,
        kept_coordinates: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        tail = self.graph(
            kept_coordinates,
            attention_mask=attention_mask,
        )
        tail = record(trace, f"{prefix}.tail", tail)
        coordinates = torch.cat((kept_coordinates, tail), dim=-1)
        return record(trace, f"{prefix}.coordinates", coordinates)

    def decode(self, full_coordinates: Tensor) -> Tensor:
        return self.full_projection.decode(full_coordinates)

    def forward(
        self,
        activations: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        kept = record(
            trace,
            f"{prefix}.kept",
            self.encode_kept(activations),
        )
        return self.decode(
            self.complete(
                kept,
                attention_mask=attention_mask,
                trace=trace,
                prefix=prefix,
            )
        )


def _tail_fit_metrics(
    actual: Tensor,
    predicted: Tensor,
) -> tuple[float, float, float, int]:
    residual = actual - predicted
    centered = actual - actual.mean(dim=0, keepdim=True)
    residual_sum = residual.square().sum()
    centered_sum = centered.square().sum()
    r_squared = (
        1.0 - (residual_sum / centered_sum).item()
        if centered_sum > 0
        else 1.0
    )
    position_r_squared: list[float] = []
    constant_positions = 0
    for position in range(actual.shape[1]):
        position_residual = residual[:, position].square().sum()
        position_centered = centered[:, position].square().sum()
        if position_centered <= torch.finfo(actual.dtype).eps:
            constant_positions += 1
            continue
        position_r_squared.append(
            1.0 - (position_residual / position_centered).item()
        )
    return (
        r_squared,
        residual.square().mean().sqrt().item(),
        min(position_r_squared) if position_r_squared else 1.0,
        constant_positions,
    )


def _standardized_ridge(
    design: Tensor,
    target: Tensor,
    *,
    fit_config: ModalCompletionFitConfig,
) -> Tensor:
    design_scale = design.std(dim=0).clamp_min(
        fit_config.minimum_scale
    )
    target_scale = target.std(dim=0).clamp_min(
        fit_config.minimum_scale
    )
    standardized_design = design / design_scale
    standardized_target = target / target_scale
    observations = standardized_design.shape[0]
    gram = (
        standardized_design.transpose(0, 1) @ standardized_design
        / observations
    )
    right_hand_side = (
        standardized_design.transpose(0, 1) @ standardized_target
        / observations
    )
    if fit_config.ridge > 0:
        gram = gram + torch.eye(
            gram.shape[0],
            dtype=gram.dtype,
            device=gram.device,
        ) * fit_config.ridge
        standardized_weight = torch.linalg.solve(
            gram,
            right_hand_side,
        )
    else:
        standardized_weight = torch.linalg.pinv(gram) @ right_hand_side
    return (
        standardized_weight
        / design_scale.unsqueeze(1)
        * target_scale.unsqueeze(0)
    )


def fit_local_modal_completion(
    activations: Tensor,
    basis: FisherModeBasis,
    *,
    kept_modes: int,
    shared_weights: bool,
    fit_config: ModalCompletionFitConfig | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    PositionConditionedModalCompletion,
    ModalCompletionFitReport,
]:
    """Fit a deterministic local ridge map from retained to discarded modes."""

    options = fit_config or ModalCompletionFitConfig()
    if activations.ndim != 3:
        raise ValueError(
            "completion activations must have shape "
            "[samples, sequence, width]"
        )
    if activations.shape[0] < 2:
        raise ValueError("completion fitting requires at least two samples")
    if activations.shape[-1] != basis.width:
        raise ValueError("completion activation width does not match basis")
    if basis.position_means is None:
        raise ValueError("completion fitting requires position means")
    if activations.shape[1] != basis.position_means.shape[0]:
        raise ValueError(
            "completion sequence length does not match position means"
        )
    if not 1 <= kept_modes < basis.width:
        raise ValueError("kept_modes must be between 1 and basis width - 1")
    full = basis.project(
        activations.to(torch.float64),
        modes=basis.width,
        centering="position",
    )
    kept = full[..., :kept_modes]
    tail = full[..., kept_modes:]
    kept_mean = kept.mean(dim=0)
    tail_mean = tail.mean(dim=0)

    if shared_weights:
        centered_kept = (kept - kept_mean).reshape(-1, kept_modes)
        centered_tail = (tail - tail_mean).reshape(
            -1,
            basis.width - kept_modes,
        )
        weight = _standardized_ridge(
            centered_kept,
            centered_tail,
            fit_config=options,
        )
        bias = tail_mean - torch.einsum(
            "ti,io->to",
            kept_mean,
            weight,
        )
    else:
        weights: list[Tensor] = []
        biases: list[Tensor] = []
        for position in range(full.shape[1]):
            centered_kept = kept[:, position] - kept_mean[position]
            centered_tail = tail[:, position] - tail_mean[position]
            position_weight = _standardized_ridge(
                centered_kept,
                centered_tail,
                fit_config=options,
            )
            weights.append(position_weight)
            biases.append(
                tail_mean[position]
                - kept_mean[position] @ position_weight
            )
        weight = torch.stack(weights)
        bias = torch.stack(biases)

    graph = LocalModalCompletionGraph(
        weight.to(dtype=dtype),
        bias.to(dtype=dtype),
        shared_weights=shared_weights,
    )
    completion = PositionConditionedModalCompletion.from_basis(
        basis,
        graph,
        dtype=dtype,
    )
    with torch.no_grad():
        predicted = graph(kept.to(dtype=dtype)).to(torch.float64)
    r_squared, rmse, minimum_position, constant_positions = _tail_fit_metrics(
        tail,
        predicted,
    )
    return completion, ModalCompletionFitReport(
        samples=activations.shape[0],
        observations=activations.shape[0] * activations.shape[1],
        train_tail_r_squared=r_squared,
        train_tail_rmse=rmse,
        minimum_nonconstant_position_r_squared=minimum_position,
        constant_position_count=constant_positions,
        learned_parameters=graph.learned_parameter_count,
        map_multiplies_per_sequence=graph.edge_count,
        fit_config=options,
    )


def make_mean_modal_completion(
    activations: Tensor,
    basis: FisherModeBasis,
    *,
    kept_modes: int,
    dtype: torch.dtype = torch.float32,
) -> PositionConditionedModalCompletion:
    """Build a fit-set, position-specific mean-tail control."""

    if (
        activations.ndim != 3
        or activations.shape[0] == 0
        or activations.shape[-1] != basis.width
    ):
        raise ValueError(
            "completion activations must have shape "
            "[samples, sequence, basis_width]"
        )
    if basis.position_means is None:
        raise ValueError("completion requires position means")
    if activations.shape[1] != basis.position_means.shape[0]:
        raise ValueError(
            "completion sequence length does not match position means"
        )
    if not 1 <= kept_modes < basis.width:
        raise ValueError("kept_modes must be between 1 and basis width - 1")
    full = basis.project(
        activations.to(torch.float64),
        modes=basis.width,
        centering="position",
    )
    tail_mean = full[..., kept_modes:].mean(dim=0)
    graph = LocalModalCompletionGraph(
        torch.zeros(
            kept_modes,
            basis.width - kept_modes,
            dtype=dtype,
        ),
        tail_mean.to(dtype=dtype),
        shared_weights=True,
    )
    return PositionConditionedModalCompletion.from_basis(
        basis,
        graph,
        dtype=dtype,
    )


def _projections_match(
    projection: PositionConditionedModalProjection,
    completion: PositionConditionedModalCompletion,
) -> bool:
    if (
        projection.activation_name != completion.activation_name
        or projection.sequence_length != completion.sequence_length
        or projection.width != completion.width
        or projection.modes != completion.kept_modes
    ):
        return False
    return bool(
        torch.allclose(
            projection.position_mean,
            completion.full_projection.position_mean.to(
                projection.position_mean
            ),
            rtol=0,
            atol=0,
        )
        and torch.allclose(
            projection.vectors,
            completion.full_projection.vectors[
                :, : completion.kept_modes
            ].to(projection.vectors),
            rtol=0,
            atol=0,
        )
    )


class PositionConditionedModalCompletionBottleneckExecutor(LayerExecutor):
    """Run a frozen layer through optional input/output completion bridges."""

    def __init__(
        self,
        inner: LayerExecutor,
        *,
        input_completion: PositionConditionedModalCompletion | None = None,
        output_completion: PositionConditionedModalCompletion | None = None,
    ) -> None:
        super().__init__()
        if input_completion is None and output_completion is None:
            raise ValueError("at least one completion bridge is required")
        if (
            input_completion is not None
            and output_completion is not None
            and input_completion.sequence_length
            != output_completion.sequence_length
        ):
            raise ValueError(
                "input and output completion sequence lengths differ"
            )
        self.inner = inner
        self.input_completion = input_completion
        self.output_completion = output_completion

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        hidden_states = record(trace, f"{prefix}.input", hidden_states)
        if self.input_completion is not None:
            input_modes = record(
                trace,
                f"{prefix}.modal.input",
                self.input_completion.encode_kept(hidden_states),
            )
            input_coordinates = self.input_completion.complete(
                input_modes,
                attention_mask=attention_mask,
                trace=trace,
                prefix=f"{prefix}.modal.input_completion",
            )
            hidden_states = record(
                trace,
                f"{prefix}.modal.input_reconstruction",
                self.input_completion.decode(input_coordinates),
            )
        full_output = self.inner(
            hidden_states,
            attention_mask=attention_mask,
            trace=None,
            prefix=prefix,
        )
        full_output = record(
            trace,
            f"{prefix}.modal.full_output",
            full_output,
        )
        if self.output_completion is not None:
            output_modes = record(
                trace,
                f"{prefix}.modal.output",
                self.output_completion.encode_kept(full_output),
            )
            output_coordinates = self.output_completion.complete(
                output_modes,
                attention_mask=attention_mask,
                trace=trace,
                prefix=f"{prefix}.modal.output_completion",
            )
            full_output = self.output_completion.decode(
                output_coordinates
            )
        return record(trace, f"{prefix}.output", full_output)


class PositionConditionedCompletedModalGraphExecutor(LayerExecutor):
    """Add a learned tail lift to an existing standalone modal executor."""

    def __init__(
        self,
        base_executor: PositionConditionedModalGraphExecutor,
        output_completion: PositionConditionedModalCompletion,
    ) -> None:
        super().__init__()
        if not _projections_match(
            base_executor.output_projection,
            output_completion,
        ):
            raise ValueError(
                "output completion does not match the modal executor"
            )
        self.base_executor = base_executor
        self.output_completion = output_completion

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        trace: ActivationTrace | None = None,
        prefix: str,
    ) -> Tensor:
        hidden_states = record(trace, f"{prefix}.input", hidden_states)
        input_modes = record(
            trace,
            f"{prefix}.modal.input",
            self.base_executor.input_projection.encode(hidden_states),
        )
        graph = self.base_executor.graph
        if isinstance(graph, CausalModalMLPGraph):
            hidden = graph.compute_hidden(
                input_modes,
                attention_mask=attention_mask,
            )
            hidden = record(
                trace,
                f"{prefix}.modal.hidden",
                hidden,
            )
            output_modes = graph.compute_output(hidden)
        elif isinstance(graph, CausalModalGraph):
            output_modes = graph(
                input_modes,
                attention_mask=attention_mask,
            )
        else:
            raise TypeError("unsupported base modal graph")
        output_modes = record(
            trace,
            f"{prefix}.modal.output",
            output_modes,
        )
        output_coordinates = self.output_completion.complete(
            output_modes,
            attention_mask=attention_mask,
            trace=trace,
            prefix=f"{prefix}.modal.output_completion",
        )
        output = self.output_completion.decode(output_coordinates)
        return record(trace, f"{prefix}.output", output)


def save_position_modal_completion(
    path: str | Path,
    *,
    completion: PositionConditionedModalCompletion,
    metadata: Mapping[str, object],
) -> None:
    """Save one completion bridge without duplicating a transformer layer."""

    torch.save(
        {
            "format_version": 1,
            "artifact_kind": "position_conditioned_modal_completion",
            "config": asdict(completion.config),
            "completion_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in completion.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        Path(path),
    )


def load_position_modal_completion(
    path: str | Path,
) -> tuple[
    PositionConditionedModalCompletion,
    ModalCompletionConfig,
    dict[str, object],
]:
    """Load an artifact written by :func:`save_position_modal_completion`."""

    state = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if state.get("format_version") != 1:
        raise ValueError("unsupported modal completion artifact format")
    if (
        state.get("artifact_kind")
        != "position_conditioned_modal_completion"
    ):
        raise ValueError("unsupported modal completion artifact kind")
    config = ModalCompletionConfig(**state["config"])
    completion_state = state["completion_state_dict"]
    projection = PositionConditionedModalProjection(
        activation_name=config.activation_name,
        position_mean=completion_state[
            "full_projection.position_mean"
        ],
        vectors=completion_state["full_projection.vectors"],
    )
    weight = completion_state["graph.weight"]
    bias = completion_state["graph.bias"]
    graph = LocalModalCompletionGraph(
        weight,
        bias,
        shared_weights=(
            config.graph_kind == "shared_local_linear"
        ),
    )
    completion = PositionConditionedModalCompletion(
        projection,
        graph,
    )
    completion.load_state_dict(completion_state)
    if completion.config != config:
        raise ValueError("modal completion state does not match its config")
    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("modal completion metadata must be an object")
    return completion, config, dict(metadata)
