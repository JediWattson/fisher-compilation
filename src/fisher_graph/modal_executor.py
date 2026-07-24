"""Position-conditioned execution in Fisher-mode coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .layers import LayerExecutor
from .modes import FisherModeBasis


@dataclass(frozen=True, slots=True)
class ModalExecutorConfig:
    input_activation: str
    output_activation: str
    sequence_length: int
    input_modes: int
    output_modes: int
    routing_width: int

    def __post_init__(self) -> None:
        if not self.input_activation or not self.output_activation:
            raise ValueError("modal activation names cannot be empty")
        if min(
            self.sequence_length,
            self.input_modes,
            self.output_modes,
            self.routing_width,
        ) <= 0:
            raise ValueError("modal executor dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ModalExecutorFitConfig:
    steps: int = 2_000
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    evaluation_interval: int = 100
    seed: int = 314_159
    device: str = "cpu"
    minimum_scale: float = 1e-4

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("fit steps and batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be nonnegative")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation interval must be positive")
        if self.minimum_scale <= 0:
            raise ValueError("minimum scale must be positive")


@dataclass(frozen=True, slots=True)
class ModalExecutorFitPoint:
    step: int
    batch_mse: float
    validation_mse: float


@dataclass(frozen=True, slots=True)
class ModalExecutorFitReport:
    best_step: int
    train_mse: float
    validation_mse: float
    learned_parameters: int
    graph_edges: int
    history: tuple[ModalExecutorFitPoint, ...]


class PositionConditionedModalProjection(nn.Module):
    """Project and reconstruct a fixed-length residual stream by position."""

    def __init__(
        self,
        *,
        activation_name: str,
        position_mean: Tensor,
        vectors: Tensor,
    ) -> None:
        super().__init__()
        if not activation_name:
            raise ValueError("activation_name cannot be empty")
        if position_mean.ndim != 2:
            raise ValueError(
                "position_mean must have shape [sequence, width]"
            )
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape [width, modes]")
        if position_mean.shape[1] != vectors.shape[0]:
            raise ValueError(
                "position mean width does not match the modal vectors"
            )
        if vectors.shape[1] == 0:
            raise ValueError("a projection must retain at least one mode")
        if not torch.isfinite(position_mean).all():
            raise ValueError("position means must be finite")
        if not torch.isfinite(vectors).all():
            raise ValueError("modal vectors must be finite")
        self.activation_name = activation_name
        self.register_buffer(
            "position_mean",
            position_mean.detach().clone(),
        )
        self.register_buffer(
            "vectors",
            vectors.detach().clone(),
        )

    @classmethod
    def from_basis(
        cls,
        basis: FisherModeBasis,
        *,
        modes: int,
        dtype: torch.dtype = torch.float32,
    ) -> PositionConditionedModalProjection:
        if not 1 <= modes <= basis.width:
            raise ValueError(
                f"modes must be between 1 and {basis.width}"
            )
        if basis.position_means is None:
            raise ValueError(
                "position-conditioned projection requires position means"
            )
        return cls(
            activation_name=basis.activation_name,
            position_mean=basis.position_means.to(dtype=dtype),
            vectors=basis.vectors[:, :modes].to(dtype=dtype),
        )

    @property
    def sequence_length(self) -> int:
        return self.position_mean.shape[0]

    @property
    def width(self) -> int:
        return self.position_mean.shape[1]

    @property
    def modes(self) -> int:
        return self.vectors.shape[1]

    def _validate_values(self, values: Tensor, *, width: int) -> None:
        if values.ndim != 3:
            raise ValueError(
                "modal projection values must have shape "
                "[batch, sequence, features]"
            )
        if values.shape[1] != self.sequence_length:
            raise ValueError(
                "value sequence length does not match the modal projection"
            )
        if values.shape[2] != width:
            raise ValueError(
                f"expected feature width {width}, got {values.shape[2]}"
            )

    def encode(self, activations: Tensor) -> Tensor:
        self._validate_values(activations, width=self.width)
        mean = self.position_mean.to(
            device=activations.device,
            dtype=activations.dtype,
        )
        vectors = self.vectors.to(
            device=activations.device,
            dtype=activations.dtype,
        )
        return (activations - mean) @ vectors

    def decode(self, coordinates: Tensor) -> Tensor:
        self._validate_values(coordinates, width=self.modes)
        mean = self.position_mean.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        vectors = self.vectors.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        return coordinates @ vectors.transpose(0, 1) + mean

    def forward(self, activations: Tensor) -> Tensor:
        return self.decode(self.encode(activations))


class CausalModalGraph(nn.Module):
    """A fixed-length DAG from input position-modes to output position-modes."""

    def __init__(
        self,
        weights: Tensor,
        bias: Tensor,
        *,
        edge_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        if weights.ndim != 4:
            raise ValueError(
                "weights must have shape "
                "[output_position, input_position, input_mode, output_mode]"
            )
        output_positions, input_positions, input_modes, output_modes = (
            weights.shape
        )
        if output_positions != input_positions:
            raise ValueError(
                "modal graph input and output sequence lengths must match"
            )
        if bias.shape != (output_positions, output_modes):
            raise ValueError(
                "bias must have shape [output_position, output_mode]"
            )
        causal = torch.ones(
            output_positions,
            input_positions,
            dtype=torch.bool,
            device=weights.device,
        ).tril().view(
            output_positions,
            input_positions,
            1,
            1,
        )
        if edge_mask is None:
            mask = causal.expand_as(weights)
        else:
            if edge_mask.shape != weights.shape:
                raise ValueError("edge_mask must match the graph weights")
            if edge_mask.dtype != torch.bool:
                raise ValueError("edge_mask must be boolean")
            if (edge_mask & ~causal).any():
                raise ValueError("edge_mask cannot contain noncausal edges")
            mask = edge_mask
        if not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
            raise ValueError("modal graph parameters must be finite")
        self.weights = nn.Parameter(weights.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone())
        self.register_buffer("edge_mask", mask.detach().clone())

    @property
    def sequence_length(self) -> int:
        return self.weights.shape[0]

    @property
    def input_modes(self) -> int:
        return self.weights.shape[2]

    @property
    def output_modes(self) -> int:
        return self.weights.shape[3]

    @property
    def edge_count(self) -> int:
        return int(self.edge_mask.sum().item())

    @property
    def possible_edge_count(self) -> int:
        return (
            self.sequence_length
            * (self.sequence_length + 1)
            // 2
            * self.input_modes
            * self.output_modes
        )

    @property
    def density(self) -> float:
        return self.edge_count / self.possible_edge_count

    def forward(
        self,
        coordinates: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if coordinates.ndim != 3:
            raise ValueError(
                "modal coordinates must have shape [batch, sequence, modes]"
            )
        if coordinates.shape[1:] != (
            self.sequence_length,
            self.input_modes,
        ):
            raise ValueError(
                "modal coordinate shape does not match the graph input"
            )
        if attention_mask is not None:
            if attention_mask.shape != coordinates.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]"
                )
            mask = attention_mask.to(
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            coordinates = coordinates * mask.unsqueeze(-1)
        weights = self.weights * self.edge_mask.to(
            device=self.weights.device,
            dtype=self.weights.dtype,
        )
        output = torch.einsum(
            "bsi,tsio->bto",
            coordinates,
            weights,
        ) + self.bias
        if attention_mask is not None:
            output = output * mask.unsqueeze(-1)
        return output

    def strongest_edges(
        self,
        count: int = 20,
    ) -> list[dict[str, float | int]]:
        if count <= 0 or self.edge_count == 0:
            return []
        magnitude = self.weights.detach().abs().masked_fill(
            ~self.edge_mask,
            -torch.inf,
        )
        edge_count = min(count, self.edge_count)
        flat_indices = torch.topk(
            magnitude.flatten(),
            edge_count,
        ).indices
        edges: list[dict[str, float | int]] = []
        for flat_index in flat_indices.tolist():
            output_position, remainder = divmod(
                flat_index,
                self.weights.shape[1]
                * self.input_modes
                * self.output_modes,
            )
            input_position, remainder = divmod(
                remainder,
                self.input_modes * self.output_modes,
            )
            input_mode, output_mode = divmod(
                remainder,
                self.output_modes,
            )
            edges.append(
                {
                    "input_position": input_position,
                    "input_mode": input_mode,
                    "output_position": output_position,
                    "output_mode": output_mode,
                    "weight": self.weights[
                        output_position,
                        input_position,
                        input_mode,
                        output_mode,
                    ].item(),
                }
            )
        return edges


class CausalModalMLPGraph(nn.Module):
    """A causal modal DAG with one nonlinear hidden bank per output position."""

    def __init__(
        self,
        *,
        input_modes: int,
        output_modes: int,
        sequence_length: int,
        hidden_modes: int,
        input_scale: Tensor,
        output_scale: Tensor,
    ) -> None:
        super().__init__()
        if min(
            input_modes,
            output_modes,
            sequence_length,
            hidden_modes,
        ) <= 0:
            raise ValueError("modal graph dimensions must be positive")
        self._input_modes = input_modes
        self._output_modes = output_modes
        self._sequence_length = sequence_length
        self.hidden_modes = hidden_modes
        expected_input_scale = (sequence_length, input_modes)
        expected_output_scale = (sequence_length, output_modes)
        if input_scale.shape != expected_input_scale:
            raise ValueError(
                f"input_scale must have shape {expected_input_scale}"
            )
        if output_scale.shape != expected_output_scale:
            raise ValueError(
                f"output_scale must have shape {expected_output_scale}"
            )
        for name, scale in (
            ("input_scale", input_scale),
            ("output_scale", output_scale),
        ):
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"{name} must be finite and positive")
        self.register_buffer(
            "input_scale",
            input_scale.detach().clone(),
        )
        self.register_buffer(
            "output_scale",
            output_scale.detach().clone(),
        )
        self.input_layers = nn.ModuleList(
            nn.Linear((position + 1) * input_modes, hidden_modes)
            for position in range(sequence_length)
        )
        self.output_layers = nn.ModuleList(
            nn.Linear(hidden_modes, output_modes)
            for _ in range(sequence_length)
        )

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def input_modes(self) -> int:
        return self._input_modes

    @property
    def output_modes(self) -> int:
        return self._output_modes

    @property
    def edge_count(self) -> int:
        input_edges = (
            self.sequence_length
            * (self.sequence_length + 1)
            // 2
            * self.input_modes
            * self.hidden_modes
        )
        output_edges = (
            self.sequence_length
            * self.hidden_modes
            * self.output_modes
        )
        return input_edges + output_edges

    def _validate(
        self,
        coordinates: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> None:
        if coordinates.ndim != 3 or coordinates.shape[1:] != (
            self.sequence_length,
            self.input_modes,
        ):
            raise ValueError(
                "modal coordinate shape does not match the nonlinear graph"
            )
        if attention_mask is not None:
            if attention_mask.shape != coordinates.shape[:2]:
                raise ValueError(
                    "attention_mask must have shape [batch, sequence]"
                )
            if not attention_mask.to(torch.bool).all():
                raise ValueError(
                    "the fixed position-conditioned modal graph does not "
                    "support padding"
                )

    def compute_hidden(
        self,
        coordinates: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate(
            coordinates,
            attention_mask=attention_mask,
        )
        scale = self.input_scale.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        normalized = coordinates / scale
        hidden = [
            torch.nn.functional.gelu(
                self.input_layers[position](
                    normalized[:, : position + 1].flatten(start_dim=1)
                )
            )
            for position in range(self.sequence_length)
        ]
        return torch.stack(hidden, dim=1)

    def compute_output(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (
            self.sequence_length,
            self.hidden_modes,
        ):
            raise ValueError(
                "hidden modal shape does not match the nonlinear graph"
            )
        standardized = torch.stack(
            [
                self.output_layers[position](hidden[:, position])
                for position in range(self.sequence_length)
            ],
            dim=1,
        )
        scale = self.output_scale.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return standardized * scale

    def forward(
        self,
        coordinates: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self.compute_output(
            self.compute_hidden(
                coordinates,
                attention_mask=attention_mask,
            )
        )


def _modal_training_coordinates(
    input_activations: Tensor,
    output_activations: Tensor,
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    config: ModalExecutorConfig,
) -> tuple[Tensor, Tensor]:
    if input_basis.activation_name != config.input_activation:
        raise ValueError("input basis does not match the executor config")
    if output_basis.activation_name != config.output_activation:
        raise ValueError("output basis does not match the executor config")
    if config.input_modes > input_basis.width:
        raise ValueError("input mode count exceeds the input basis")
    if config.output_modes > output_basis.width:
        raise ValueError("output mode count exceeds the output basis")
    if input_activations.ndim != 3 or output_activations.ndim != 3:
        raise ValueError(
            "fit activations must have shape [samples, sequence, width]"
        )
    if input_activations.shape[:2] != output_activations.shape[:2]:
        raise ValueError("input and output fit sample grids must align")
    if input_activations.shape[0] == 0:
        raise ValueError("fit activations cannot be empty")
    if input_activations.shape[1] != config.sequence_length:
        raise ValueError("fit sequence length does not match the config")
    inputs = input_basis.project(
        input_activations.to(torch.float64),
        modes=config.input_modes,
        centering="position",
    ).to(torch.float32)
    outputs = output_basis.project(
        output_activations.to(torch.float64),
        modes=config.output_modes,
        centering="position",
    ).to(torch.float32)
    return inputs, outputs


def fit_position_modal_executor(
    input_activations: Tensor,
    output_activations: Tensor,
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    *,
    config: ModalExecutorConfig,
    fit_config: ModalExecutorFitConfig | None = None,
    validation_input_activations: Tensor | None = None,
    validation_output_activations: Tensor | None = None,
) -> tuple[
    PositionConditionedModalGraphExecutor,
    ModalExecutorFitReport,
]:
    """Distill a causal nonlinear modal graph from teacher activations.

    The fitter consumes only aligned layer input/output activations. Dataset
    provenance and final held-out evaluation remain the experiment runner's
    responsibility.
    """

    fit_options = fit_config or ModalExecutorFitConfig()
    train_inputs, train_outputs = _modal_training_coordinates(
        input_activations,
        output_activations,
        input_basis,
        output_basis,
        config,
    )
    if (validation_input_activations is None) != (
        validation_output_activations is None
    ):
        raise ValueError(
            "validation input and output activations must be provided together"
        )
    if validation_input_activations is None:
        validation_inputs = train_inputs
        validation_outputs = train_outputs
    else:
        assert validation_output_activations is not None
        validation_inputs, validation_outputs = _modal_training_coordinates(
            validation_input_activations,
            validation_output_activations,
            input_basis,
            output_basis,
            config,
        )

    # Normalize each position/mode by variation across examples. The modal
    # coordinates are already position-centered by the Fisher basis, so their
    # sample standard deviation is the natural scale for the distillation
    # objective. A practical floor keeps nearly dormant modes finite.
    input_scale = train_inputs.std(dim=0).clamp_min(
        fit_options.minimum_scale
    )
    output_scale = train_outputs.std(dim=0).clamp_min(
        fit_options.minimum_scale
    )
    device = torch.device(fit_options.device)
    train_inputs = train_inputs.to(device)
    train_targets = (train_outputs / output_scale).to(device)
    validation_inputs = validation_inputs.to(device)
    validation_targets = (
        validation_outputs / output_scale
    ).to(device)

    with torch.random.fork_rng(
        devices=(
            [device]
            if device.type == "cuda"
            else []
        )
    ):
        torch.manual_seed(fit_options.seed)
        graph = CausalModalMLPGraph(
            input_modes=config.input_modes,
            output_modes=config.output_modes,
            sequence_length=config.sequence_length,
            hidden_modes=config.routing_width,
            input_scale=input_scale.to(device),
            output_scale=output_scale.to(device),
        ).to(device)
    optimizer = torch.optim.AdamW(
        graph.parameters(),
        lr=fit_options.learning_rate,
        weight_decay=fit_options.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        fit_options.seed + 1
    )
    history: list[ModalExecutorFitPoint] = []
    best_step = 0
    best_validation = float("inf")
    best_state: dict[str, Tensor] | None = None

    for step in range(1, fit_options.steps + 1):
        indices = torch.randint(
            0,
            train_inputs.shape[0],
            (min(fit_options.batch_size, train_inputs.shape[0]),),
            generator=generator,
        ).to(device)
        batch_inputs = train_inputs.index_select(0, indices)
        batch_targets = train_targets.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        predictions = graph(batch_inputs) / graph.output_scale
        loss = torch.nn.functional.mse_loss(
            predictions,
            batch_targets,
        )
        loss.backward()
        optimizer.step()

        if (
            step == 1
            or step % fit_options.evaluation_interval == 0
            or step == fit_options.steps
        ):
            with torch.no_grad():
                validation_predictions = (
                    graph(validation_inputs) / graph.output_scale
                )
                validation_mse = torch.nn.functional.mse_loss(
                    validation_predictions,
                    validation_targets,
                ).item()
            history.append(
                ModalExecutorFitPoint(
                    step=step,
                    batch_mse=loss.detach().item(),
                    validation_mse=validation_mse,
                )
            )
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in graph.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("modal executor fitting produced no checkpoint")
    graph.load_state_dict(best_state)
    graph = graph.cpu()
    with torch.no_grad():
        train_predictions = graph(train_inputs.cpu()) / graph.output_scale
        train_mse = torch.nn.functional.mse_loss(
            train_predictions,
            train_targets.cpu(),
        ).item()
        validation_predictions = (
            graph(validation_inputs.cpu()) / graph.output_scale
        )
        validation_mse = torch.nn.functional.mse_loss(
            validation_predictions,
            validation_targets.cpu(),
        ).item()
    executor = PositionConditionedModalGraphExecutor(
        PositionConditionedModalProjection.from_basis(
            input_basis,
            modes=config.input_modes,
            dtype=torch.float32,
        ),
        graph,
        PositionConditionedModalProjection.from_basis(
            output_basis,
            modes=config.output_modes,
            dtype=torch.float32,
        ),
    )
    learned_parameters = sum(
        parameter.numel() for parameter in graph.parameters()
    )
    return executor, ModalExecutorFitReport(
        best_step=best_step,
        train_mse=train_mse,
        validation_mse=validation_mse,
        learned_parameters=learned_parameters,
        graph_edges=graph.edge_count,
        history=tuple(history),
    )


@dataclass(frozen=True, slots=True)
class CausalModalGraphFit:
    """Analytic ridge fit for a causal modal graph."""

    weights: Tensor
    bias: Tensor
    train_r_squared: float
    train_rmse: float
    ridge: float
    samples: int

    def graph(self, *, dtype: torch.dtype = torch.float32) -> CausalModalGraph:
        return CausalModalGraph(
            self.weights.to(dtype=dtype),
            self.bias.to(dtype=dtype),
        )


def fit_causal_modal_graph(
    input_activations: Tensor,
    output_activations: Tensor,
    input_basis: FisherModeBasis,
    output_basis: FisherModeBasis,
    *,
    input_modes: int,
    output_modes: int,
    ridge: float = 1e-4,
) -> CausalModalGraphFit:
    """Fit a causal position-mode DAG by independent ridge regressions."""

    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if input_activations.ndim != 3:
        raise ValueError(
            "input_activations must have shape [samples, sequence, width]"
        )
    if output_activations.ndim != 3:
        raise ValueError(
            "output_activations must have shape [samples, sequence, width]"
        )
    if input_activations.shape[:2] != output_activations.shape[:2]:
        raise ValueError("input and output sample grids must align")
    if input_activations.shape[0] == 0:
        raise ValueError("cannot fit a modal graph without samples")
    inputs = input_basis.project(
        input_activations.to(torch.float64),
        modes=input_modes,
        centering="position",
    )
    outputs = output_basis.project(
        output_activations.to(torch.float64),
        modes=output_modes,
        centering="position",
    )
    samples, sequence_length, _ = inputs.shape
    weights = torch.zeros(
        sequence_length,
        sequence_length,
        input_modes,
        output_modes,
        dtype=torch.float64,
    )
    bias = torch.empty(
        sequence_length,
        output_modes,
        dtype=torch.float64,
    )

    for output_position in range(sequence_length):
        design = inputs[:, : output_position + 1].reshape(samples, -1)
        target = outputs[:, output_position]
        design_mean = design.mean(dim=0)
        target_mean = target.mean(dim=0)
        centered_design = design - design_mean
        centered_target = target - target_mean
        gram = (
            centered_design.transpose(0, 1) @ centered_design / samples
        )
        if ridge > 0:
            gram = gram + torch.eye(
                gram.shape[0],
                dtype=gram.dtype,
                device=gram.device,
            ) * ridge
        right_hand_side = (
            centered_design.transpose(0, 1) @ centered_target / samples
        )
        coefficients = torch.linalg.solve(gram, right_hand_side)
        weights[
            output_position,
            : output_position + 1,
        ] = coefficients.reshape(
            output_position + 1,
            input_modes,
            output_modes,
        )
        bias[output_position] = (
            target_mean - design_mean @ coefficients
        )

    predictions = torch.einsum(
        "bsi,tsio->bto",
        inputs,
        weights,
    ) + bias
    residual = outputs - predictions
    residual_sum = residual.square().sum()
    centered_sum = (
        outputs - outputs.mean(dim=0, keepdim=True)
    ).square().sum()
    r_squared = (
        1.0 - (residual_sum / centered_sum).item()
        if centered_sum > 0
        else 1.0
    )
    return CausalModalGraphFit(
        weights=weights,
        bias=bias,
        train_r_squared=r_squared,
        train_rmse=residual.square().mean().sqrt().item(),
        ridge=ridge,
        samples=samples,
    )


class PositionConditionedModalBottleneckExecutor(LayerExecutor):
    """Run a real layer between position-conditioned modal bottlenecks."""

    def __init__(
        self,
        inner: LayerExecutor,
        input_projection: PositionConditionedModalProjection,
        output_projection: PositionConditionedModalProjection,
    ) -> None:
        super().__init__()
        if input_projection.sequence_length != output_projection.sequence_length:
            raise ValueError(
                "input and output modal sequence lengths must match"
            )
        self.inner = inner
        self.input_projection = input_projection
        self.output_projection = output_projection

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
            self.input_projection.encode(hidden_states),
        )
        reconstructed_input = record(
            trace,
            f"{prefix}.modal.input_reconstruction",
            self.input_projection.decode(input_modes),
        )
        full_output = self.inner(
            reconstructed_input,
            attention_mask=attention_mask,
            trace=None,
            prefix=prefix,
        )
        full_output = record(
            trace,
            f"{prefix}.modal.full_output",
            full_output,
        )
        output_modes = record(
            trace,
            f"{prefix}.modal.output",
            self.output_projection.encode(full_output),
        )
        output = self.output_projection.decode(output_modes)
        return record(trace, f"{prefix}.output", output)


class PositionConditionedModalGraphExecutor(LayerExecutor):
    """Replace a transformer layer with a standalone causal modal DAG."""

    def __init__(
        self,
        input_projection: PositionConditionedModalProjection,
        graph: CausalModalGraph | CausalModalMLPGraph,
        output_projection: PositionConditionedModalProjection,
    ) -> None:
        super().__init__()
        if input_projection.sequence_length != graph.sequence_length:
            raise ValueError(
                "input projection sequence length does not match the graph"
            )
        if output_projection.sequence_length != graph.sequence_length:
            raise ValueError(
                "output projection sequence length does not match the graph"
            )
        if input_projection.modes != graph.input_modes:
            raise ValueError(
                "input projection mode count does not match the graph"
            )
        if output_projection.modes != graph.output_modes:
            raise ValueError(
                "output projection mode count does not match the graph"
            )
        self.input_projection = input_projection
        self.graph = graph
        self.output_projection = output_projection

    @classmethod
    def from_fit(
        cls,
        input_basis: FisherModeBasis,
        output_basis: FisherModeBasis,
        fit: CausalModalGraphFit,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> PositionConditionedModalGraphExecutor:
        return cls(
            PositionConditionedModalProjection.from_basis(
                input_basis,
                modes=fit.weights.shape[2],
                dtype=dtype,
            ),
            fit.graph(dtype=dtype),
            PositionConditionedModalProjection.from_basis(
                output_basis,
                modes=fit.weights.shape[3],
                dtype=dtype,
            ),
        )

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
            self.input_projection.encode(hidden_states),
        )
        if isinstance(self.graph, CausalModalMLPGraph):
            hidden = self.graph.compute_hidden(
                input_modes,
                attention_mask=attention_mask,
            )
            hidden = record(
                trace,
                f"{prefix}.modal.hidden",
                hidden,
            )
            output_modes = self.graph.compute_output(hidden)
        else:
            output_modes = self.graph(
                input_modes,
                attention_mask=attention_mask,
            )
        output_modes = record(
            trace,
            f"{prefix}.modal.output",
            output_modes,
        )
        output = self.output_projection.decode(output_modes)
        return record(trace, f"{prefix}.output", output)


def save_position_modal_executor(
    path: str | Path,
    *,
    executor: PositionConditionedModalGraphExecutor,
    config: ModalExecutorConfig,
    metadata: Mapping[str, object],
) -> None:
    """Save a standalone nonlinear position-modal graph artifact."""

    if not isinstance(executor.graph, CausalModalMLPGraph):
        raise TypeError(
            "the portable modal executor artifact currently supports "
            "CausalModalMLPGraph"
        )
    actual_config = ModalExecutorConfig(
        input_activation=executor.input_projection.activation_name,
        output_activation=executor.output_projection.activation_name,
        sequence_length=executor.graph.sequence_length,
        input_modes=executor.graph.input_modes,
        output_modes=executor.graph.output_modes,
        routing_width=executor.graph.hidden_modes,
    )
    if config != actual_config:
        raise ValueError("modal executor config does not match the executor")
    torch.save(
        {
            "format_version": 1,
            "graph_kind": "causal_position_modal_mlp",
            "config": asdict(config),
            "executor_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in executor.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        Path(path),
    )


def load_position_modal_executor(
    path: str | Path,
) -> tuple[
    PositionConditionedModalGraphExecutor,
    ModalExecutorConfig,
    dict[str, object],
]:
    """Load an artifact written by :func:`save_position_modal_executor`."""

    state = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    if state.get("format_version") != 1:
        raise ValueError("unsupported modal executor artifact format")
    if state.get("graph_kind") != "causal_position_modal_mlp":
        raise ValueError("unsupported modal executor graph kind")
    config = ModalExecutorConfig(**state["config"])
    executor_state = state["executor_state_dict"]
    input_projection = PositionConditionedModalProjection(
        activation_name=config.input_activation,
        position_mean=executor_state[
            "input_projection.position_mean"
        ],
        vectors=executor_state["input_projection.vectors"],
    )
    graph = CausalModalMLPGraph(
        input_modes=config.input_modes,
        output_modes=config.output_modes,
        sequence_length=config.sequence_length,
        hidden_modes=config.routing_width,
        input_scale=executor_state["graph.input_scale"],
        output_scale=executor_state["graph.output_scale"],
    )
    parameter_dtype = executor_state["graph.input_layers.0.weight"].dtype
    if not parameter_dtype.is_floating_point:
        raise ValueError("modal executor parameters must be floating point")
    graph = graph.to(dtype=parameter_dtype)
    output_projection = PositionConditionedModalProjection(
        activation_name=config.output_activation,
        position_mean=executor_state[
            "output_projection.position_mean"
        ],
        vectors=executor_state["output_projection.vectors"],
    )
    executor = PositionConditionedModalGraphExecutor(
        input_projection,
        graph,
        output_projection,
    )
    executor.load_state_dict(executor_state)
    return executor, config, dict(state["metadata"])
