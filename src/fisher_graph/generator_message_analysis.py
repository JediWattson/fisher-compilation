"""Stream generator-message values, score gradients, and joint moments.

Generator analysis needs a wider observation surface than Fisher collection:
some ports are captured only as values, while a smaller ordered subset carries
score gradients.  This module makes that distinction explicit and keeps the
large-model path bounded in calibration-set size.  Each sequence is processed
independently, converted to detached CPU float64 rows, and then discarded once
its exact joint activation covariance and empirical Fisher contribution have
been accumulated.
"""

from __future__ import annotations

import math
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from .adapters import ActivationSite
from .compiler.calibration import CalibrationBatch, ScoreObjective
from .instrumentation import InstrumentedModel, validate_instrumented_model


_PLAN_FORMAT_VERSION = 1
_MOMENTS_FORMAT_VERSION = 1
_MOMENTS_ALGORITHM = "exact_joint_generator_message_moments"
_MOMENTS_ALGORITHM_VERSION = 1


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
) -> float:
    if type(value) not in (float, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return converted


def _ordered_unique_names(
    values: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be an ordered sequence of names")
    names = tuple(values)
    if not names:
        raise ValueError(f"{label} cannot be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} cannot contain duplicate names")
    return names


def _validate_cpu_float64_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"{label} must be a CPU float64 Tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{label} must have shape {list(shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    return value


def _validate_symmetric_psd(value: Tensor, *, label: str) -> Tensor:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{label} must be a nonempty square matrix")
    scale = max(float(value.abs().max().item()), 1.0)
    width = value.shape[0]
    symmetry_tolerance = (
        256 * torch.finfo(torch.float64).eps * width * scale
    )
    if not torch.allclose(
        value,
        value.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        raise ValueError(f"{label} must be symmetric")
    symmetric = (value + value.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    psd_tolerance = (
        512 * torch.finfo(torch.float64).eps * width * scale
    )
    if float(eigenvalues.min().item()) < -psd_tolerance:
        raise ValueError(f"{label} must be positive semidefinite")
    return symmetric


@dataclass(frozen=True, slots=True)
class GeneratorMessageCapturePlan:
    """Immutable declaration of observed and differentiated message ports.

    ``value_sites`` defines capture order. ``gradient_sites`` is an ordered
    subset whose score gradients will be materialized. ``leaf_site`` is the
    one captured boundary replaced by a detached differentiable leaf, so a
    frozen source model builds autograd only for the suffix after that port.
    The leaf can be value-only when gradients are needed only at later ports.
    """

    value_sites: tuple[str, ...]
    gradient_sites: tuple[str, ...]
    leaf_site: str
    format_version: int = _PLAN_FORMAT_VERSION

    def __post_init__(self) -> None:
        value_sites = _ordered_unique_names(
            self.value_sites,
            label="value_sites",
        )
        gradient_sites = _ordered_unique_names(
            self.gradient_sites,
            label="gradient_sites",
        )
        leaf_site = _require_nonempty_string(
            self.leaf_site,
            label="leaf_site",
        )
        if not set(gradient_sites).issubset(value_sites):
            raise ValueError("gradient_sites must be a subset of value_sites")
        if leaf_site not in value_sites:
            raise ValueError("leaf_site must be one of value_sites")
        if self.format_version != _PLAN_FORMAT_VERSION:
            raise ValueError(
                "unsupported generator-message capture-plan format"
            )
        object.__setattr__(self, "value_sites", value_sites)
        object.__setattr__(self, "gradient_sites", gradient_sites)
        object.__setattr__(self, "leaf_site", leaf_site)

    def state_dict(self) -> dict[str, object]:
        return {
            "value_sites": self.value_sites,
            "gradient_sites": self.gradient_sites,
            "leaf_site": self.leaf_site,
            "format_version": self.format_version,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GeneratorMessageCapturePlan:
        expected = {
            "value_sites",
            "gradient_sites",
            "leaf_site",
            "format_version",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "generator-message capture-plan fields do not match "
                "format version 1"
            )
        if type(state["value_sites"]) is not tuple:
            raise TypeError("serialized value_sites must be a tuple")
        if type(state["gradient_sites"]) is not tuple:
            raise TypeError("serialized gradient_sites must be a tuple")
        return cls(
            value_sites=state["value_sites"],
            gradient_sites=state["gradient_sites"],
            leaf_site=_require_nonempty_string(
                state["leaf_site"],
                label="leaf_site",
            ),
            format_version=_require_exact_int(
                state["format_version"],
                label="format_version",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class GeneratorMessageScoreGradientRows:
    """Detached CPU float64 message rows from one uniquely named sequence."""

    plan: GeneratorMessageCapturePlan
    activations: Mapping[str, Tensor]
    score_gradients: Mapping[str, Tensor]
    logical_positions: Tensor
    loss: float
    example_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, GeneratorMessageCapturePlan):
            raise TypeError("plan must be a GeneratorMessageCapturePlan")
        if not isinstance(self.activations, Mapping):
            raise TypeError("activations must be a mapping")
        if tuple(self.activations) != self.plan.value_sites:
            raise ValueError(
                "activations must follow the plan's ordered value_sites"
            )
        if not isinstance(self.score_gradients, Mapping):
            raise TypeError("score_gradients must be a mapping")
        if tuple(self.score_gradients) != self.plan.gradient_sites:
            raise ValueError(
                "score_gradients must follow the plan's ordered "
                "gradient_sites"
            )

        activation_rows: dict[str, Tensor] = {}
        gradient_rows: dict[str, Tensor] = {}
        observations: int | None = None
        widths: dict[str, int] = {}
        for name in self.plan.value_sites:
            activation = _validate_cpu_float64_tensor(
                self.activations[name],
                label=f"{name!r} activations",
            )
            if activation.ndim != 2 or activation.shape[0] <= 0:
                raise ValueError(
                    f"{name!r} activations must have shape "
                    "[positive observations, width]"
                )
            if activation.shape[1] <= 0:
                raise ValueError(f"{name!r} activation width must be positive")
            if observations is None:
                observations = activation.shape[0]
            elif activation.shape[0] != observations:
                raise ValueError(
                    "all value sites must contain the same observations"
                )
            widths[name] = activation.shape[1]
            activation_rows[name] = activation.detach().clone().contiguous()

        assert observations is not None
        for name in self.plan.gradient_sites:
            gradient = _validate_cpu_float64_tensor(
                self.score_gradients[name],
                label=f"{name!r} score gradients",
                shape=(observations, widths[name]),
            )
            gradient_rows[name] = gradient.detach().clone().contiguous()

        logical_positions = self.logical_positions
        if (
            not isinstance(logical_positions, Tensor)
            or logical_positions.device.type != "cpu"
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or tuple(logical_positions.shape) != (observations,)
        ):
            raise ValueError(
                "logical_positions must be a CPU integer vector with one "
                "entry per observation"
            )
        if (logical_positions < 0).any():
            raise ValueError("logical_positions must be nonnegative")
        if observations > 1 and not torch.all(
            logical_positions[1:] > logical_positions[:-1]
        ):
            raise ValueError(
                "logical_positions must be strictly increasing"
            )
        loss = _require_finite_number(self.loss, label="loss")
        example_id = _require_nonempty_string(
            self.example_id,
            label="example_id",
        )

        object.__setattr__(
            self,
            "activations",
            MappingProxyType(activation_rows),
        )
        object.__setattr__(
            self,
            "score_gradients",
            MappingProxyType(gradient_rows),
        )
        object.__setattr__(
            self,
            "logical_positions",
            logical_positions.detach().clone().to(dtype=torch.int64),
        )
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "example_id", example_id)

    @property
    def observations(self) -> int:
        return next(iter(self.activations.values())).shape[0]


def _detached_leaf(values: Tensor) -> Tensor:
    return values.detach().requires_grad_(True)


def _site_width(site: ActivationSite) -> int:
    if site.width is None:
        raise ValueError(
            f"{site.id!r} does not declare a modal activation width"
        )
    return site.width


def _prepare_message_row_stream(
    model: InstrumentedModel,
    *,
    plan: GeneratorMessageCapturePlan,
    score_objective: ScoreObjective,
) -> dict[str, ActivationSite]:
    validate_instrumented_model(model)
    if not isinstance(plan, GeneratorMessageCapturePlan):
        raise TypeError("plan must be a GeneratorMessageCapturePlan")
    if not callable(score_objective):
        raise TypeError("score_objective must be callable")
    if any(parameter.requires_grad for parameter in model.module.parameters()):
        raise ValueError(
            "generator-message capture requires a frozen model; all "
            "parameters must have requires_grad=False"
        )

    catalog = {site.id: site for site in model.activation_sites}
    selected: dict[str, ActivationSite] = {}
    for name in plan.value_sites:
        try:
            site = catalog[name]
        except KeyError as error:
            raise KeyError(f"unknown activation site: {name!r}") from error
        if not site.modal_eligible:
            raise ValueError(
                f"{name!r} is not a canonical "
                "[batch, sequence, feature] activation site"
            )
        _site_width(site)
        selected[name] = site
    if not selected[plan.leaf_site].intervenable:
        raise ValueError(
            f"leaf site {plan.leaf_site!r} does not support intervention"
        )
    return selected


def _sequence_message_rows(
    model: InstrumentedModel,
    sample: CalibrationBatch,
    *,
    plan: GeneratorMessageCapturePlan,
    sites: Mapping[str, ActivationSite],
    score_objective: ScoreObjective,
) -> GeneratorMessageScoreGradientRows:
    run = model.forward(
        sample.model_inputs,
        capture_sites=plan.value_sites,
        interventions={plan.leaf_site: _detached_leaf},
        retain_gradients=False,
    )
    missing = set(plan.value_sites) - set(run.activations)
    if missing:
        raise KeyError(f"missing activation taps: {sorted(missing)}")
    selected = {
        name: run.activations[name]
        for name in plan.value_sites
    }
    expected_grid = (1, sample.valid_positions.shape[1])
    for name, values in selected.items():
        if (
            values.ndim != 3
            or tuple(values.shape[:2]) != expected_grid
            or values.shape[2] != _site_width(sites[name])
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
        raise ValueError(
            "score_objective result is not differentiable from the "
            "detached leaf"
        )

    gradient_tensors = {
        name: selected[name]
        for name in plan.gradient_sites
    }
    not_differentiable = [
        name
        for name, tensor in gradient_tensors.items()
        if not tensor.requires_grad
    ]
    if not_differentiable:
        raise ValueError(
            "gradient sites are not differentiable from the detached leaf: "
            f"{not_differentiable}"
        )
    unique_tensors: list[Tensor] = []
    tensor_indices: dict[int, int] = {}
    for tensor in gradient_tensors.values():
        tensor_id = id(tensor)
        if tensor_id not in tensor_indices:
            tensor_indices[tensor_id] = len(unique_tensors)
            unique_tensors.append(tensor)
    unique_gradients = torch.autograd.grad(
        loss,
        tuple(unique_tensors),
        allow_unused=False,
    )

    if run.sequence.query_valid_mask.shape != expected_grid:
        raise ValueError(
            "adapter sequence mask must match the calibration query grid"
        )
    valid_positions = sample.valid_positions[0].to(
        device=run.sequence.query_valid_mask.device
    )
    sequence_valid = run.sequence.query_valid_mask[0]
    if (valid_positions & ~sequence_valid).any():
        raise ValueError(
            "calibration positions must be valid in the adapter sequence "
            "context"
        )
    if not valid_positions.any():
        raise ValueError("a calibration sample has no valid positions")

    activation_rows: dict[str, Tensor] = {}
    for name, values in selected.items():
        mask = valid_positions.to(device=values.device)
        rows = values.detach()[0, mask].to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.isfinite(rows).all():
            raise ValueError(f"{name!r} contains non-finite activations")
        activation_rows[name] = rows

    gradient_rows: dict[str, Tensor] = {}
    for name, values in gradient_tensors.items():
        mask = valid_positions.to(device=values.device)
        gradient = unique_gradients[tensor_indices[id(values)]]
        rows = gradient.detach()[0, mask].to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.isfinite(rows).all():
            raise ValueError(f"{name!r} contains non-finite gradients")
        gradient_rows[name] = rows

    assert sample.example_ids is not None
    logical_mask = valid_positions.to(
        device=run.sequence.logical_positions.device
    )
    return GeneratorMessageScoreGradientRows(
        plan=plan,
        activations=activation_rows,
        score_gradients=gradient_rows,
        logical_positions=(
            run.sequence.logical_positions[0, logical_mask]
            .detach()
            .to(device="cpu", dtype=torch.int64)
        ),
        loss=float(loss.detach().item()),
        example_id=sample.example_ids[0],
    )


def _iter_prepared_message_rows(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    plan: GeneratorMessageCapturePlan,
    sites: Mapping[str, ActivationSite],
    score_objective: ScoreObjective,
) -> Generator[GeneratorMessageScoreGradientRows, None, None]:
    module = model.module
    was_training = module.training
    seen_example_ids: set[str] = set()
    module.eval()
    try:
        for batch in calibration_batches:
            if not isinstance(batch, CalibrationBatch):
                raise TypeError(
                    "calibration_batches must contain CalibrationBatch values"
                )
            if batch.example_ids is None:
                raise ValueError(
                    "generator-message capture requires example_ids"
                )
            for batch_index, example_id in enumerate(batch.example_ids):
                if example_id in seen_example_ids:
                    raise ValueError(
                        "example_ids must be unique across the complete "
                        f"capture stream; duplicate {example_id!r}"
                    )
                seen_example_ids.add(example_id)
                sample = batch.sample(batch_index)
                # Exit the context before yielding so callers suspended under
                # torch.no_grad() continue to observe their own grad mode.
                with torch.enable_grad():
                    rows = _sequence_message_rows(
                        model,
                        sample,
                        plan=plan,
                        sites=sites,
                        score_objective=score_objective,
                    )
                yield rows
    finally:
        module.train(was_training)


def iter_generator_message_score_gradient_rows(
    model: InstrumentedModel,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    plan: GeneratorMessageCapturePlan,
    score_objective: ScoreObjective,
) -> Generator[GeneratorMessageScoreGradientRows, None, None]:
    """Yield one detached generator-message record per unique sequence.

    The public call validates the model and plan eagerly. Execution itself is
    lazy, restores the module's train/eval state on exhaustion or close, and
    never yields while its internal ``torch.enable_grad`` context is active.
    """

    sites = _prepare_message_row_stream(
        model,
        plan=plan,
        score_objective=score_objective,
    )
    return _iter_prepared_message_rows(
        model,
        calibration_batches,
        plan=plan,
        sites=sites,
        score_objective=score_objective,
    )


def _site_slices(
    site_names: tuple[str, ...],
    site_widths: tuple[int, ...],
) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    for name, width in zip(site_names, site_widths, strict=True):
        stop = start + width
        slices[name] = slice(start, stop)
        start = stop
    return slices


@dataclass(frozen=True, slots=True)
class JointMessageMomentsResult:
    """Exact joint activation covariance and score-gradient Fisher."""

    site_names: tuple[str, ...]
    site_widths: tuple[int, ...]
    observations: int
    sequences: int
    mean: Tensor
    covariance: Tensor
    fisher: Tensor
    centered_activation_square_norm_sum: float
    squared_score_gradient_norm_sum: float
    normalizer: str = "valid_activation_positions"
    score_reduction: str = "sum"
    accumulation_dtype: str = "float64"
    algorithm: str = _MOMENTS_ALGORITHM
    algorithm_version: int = _MOMENTS_ALGORITHM_VERSION
    format_version: int = _MOMENTS_FORMAT_VERSION

    def __post_init__(self) -> None:
        site_names = _ordered_unique_names(
            self.site_names,
            label="site_names",
        )
        if (
            type(self.site_widths) is not tuple
            or len(self.site_widths) != len(site_names)
            or any(
                type(width) is not int or width <= 0
                for width in self.site_widths
            )
        ):
            raise ValueError(
                "site_widths must contain one positive integer per site"
            )
        width = sum(self.site_widths)
        observations = _require_exact_int(
            self.observations,
            label="observations",
            minimum=1,
        )
        _require_exact_int(
            self.sequences,
            label="sequences",
            minimum=1,
        )
        if self.sequences > observations:
            raise ValueError("sequences cannot exceed observations")
        mean = _validate_cpu_float64_tensor(
            self.mean,
            label="mean",
            shape=(width,),
        )
        covariance = _validate_symmetric_psd(
            _validate_cpu_float64_tensor(
                self.covariance,
                label="covariance",
                shape=(width, width),
            ),
            label="covariance",
        )
        fisher = _validate_symmetric_psd(
            _validate_cpu_float64_tensor(
                self.fisher,
                label="fisher",
                shape=(width, width),
            ),
            label="fisher",
        )
        centered_sum = _require_finite_number(
            self.centered_activation_square_norm_sum,
            label="centered_activation_square_norm_sum",
            minimum=0.0,
        )
        gradient_sum = _require_finite_number(
            self.squared_score_gradient_norm_sum,
            label="squared_score_gradient_norm_sum",
            minimum=0.0,
        )
        expected_centered = observations * float(
            covariance.trace().item()
        )
        expected_gradient = observations * float(fisher.trace().item())
        for actual, expected, label in (
            (
                centered_sum,
                expected_centered,
                "centered_activation_square_norm_sum",
            ),
            (
                gradient_sum,
                expected_gradient,
                "squared_score_gradient_norm_sum",
            ),
        ):
            scale = max(abs(actual), abs(expected), 1.0)
            if not math.isclose(
                actual,
                expected,
                rel_tol=1e-11,
                abs_tol=1e-12 * scale,
            ):
                raise ValueError(
                    f"{label} does not match observations times matrix trace"
                )
        _require_nonempty_string(self.normalizer, label="normalizer")
        _require_nonempty_string(
            self.score_reduction,
            label="score_reduction",
        )
        if self.accumulation_dtype != "float64":
            raise ValueError("accumulation_dtype must be 'float64'")
        if self.algorithm != _MOMENTS_ALGORITHM:
            raise ValueError("unsupported joint-message algorithm")
        if self.algorithm_version != _MOMENTS_ALGORITHM_VERSION:
            raise ValueError("unsupported joint-message algorithm version")
        if self.format_version != _MOMENTS_FORMAT_VERSION:
            raise ValueError("unsupported joint-message format version")

        object.__setattr__(self, "site_names", site_names)
        object.__setattr__(self, "mean", mean.detach().clone())
        object.__setattr__(self, "covariance", covariance.detach().clone())
        object.__setattr__(self, "fisher", fisher.detach().clone())
        object.__setattr__(
            self,
            "centered_activation_square_norm_sum",
            centered_sum,
        )
        object.__setattr__(
            self,
            "squared_score_gradient_norm_sum",
            gradient_sum,
        )

    @property
    def width(self) -> int:
        return sum(self.site_widths)

    @property
    def port_slices(self) -> Mapping[str, slice]:
        return MappingProxyType(_site_slices(self.site_names, self.site_widths))

    def site_slice(self, site_name: str) -> slice:
        try:
            return self.port_slices[site_name]
        except KeyError as error:
            raise KeyError(f"unknown message site: {site_name!r}") from error

    @property
    def per_site_means(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: self.mean[site_slice].clone()
                for name, site_slice in self.port_slices.items()
            }
        )

    @property
    def per_site_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {name: self.observations for name in self.site_names}
        )

    def covariance_block(self, left: str, right: str) -> Tensor:
        return self.covariance[
            self.site_slice(left),
            self.site_slice(right),
        ].clone()

    def fisher_block(self, left: str, right: str) -> Tensor:
        return self.fisher[
            self.site_slice(left),
            self.site_slice(right),
        ].clone()

    @property
    def covariance_cross_blocks(
        self,
    ) -> Mapping[tuple[str, str], Tensor]:
        return MappingProxyType(
            {
                (left, right): self.covariance_block(left, right)
                for index, left in enumerate(self.site_names)
                for right in self.site_names[index + 1 :]
            }
        )

    @property
    def fisher_cross_blocks(
        self,
    ) -> Mapping[tuple[str, str], Tensor]:
        return MappingProxyType(
            {
                (left, right): self.fisher_block(left, right)
                for index, left in enumerate(self.site_names)
                for right in self.site_names[index + 1 :]
            }
        )

    # Longer aliases describe the underlying quantities at schema boundaries.
    @property
    def activation_cross_covariances(
        self,
    ) -> Mapping[tuple[str, str], Tensor]:
        return self.covariance_cross_blocks

    @property
    def score_gradient_cross_fishers(
        self,
    ) -> Mapping[tuple[str, str], Tensor]:
        return self.fisher_cross_blocks

    def metadata(self) -> dict[str, object]:
        return {
            "site_names": self.site_names,
            "site_widths": self.site_widths,
            "port_slices": tuple(
                (name, value.start, value.stop)
                for name, value in self.port_slices.items()
            ),
            "width": self.width,
            "observations": self.observations,
            "sequences": self.sequences,
            "centered_activation_square_norm_sum": (
                self.centered_activation_square_norm_sum
            ),
            "squared_score_gradient_norm_sum": (
                self.squared_score_gradient_norm_sum
            ),
            "covariance_trace": float(self.covariance.trace().item()),
            "fisher_trace": float(self.fisher.trace().item()),
            "normalizer": self.normalizer,
            "score_reduction": self.score_reduction,
            "accumulation_dtype": self.accumulation_dtype,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "format_version": self.format_version,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.metadata(),
            "mean": self.mean.clone(),
            "covariance": self.covariance.clone(),
            "fisher": self.fisher.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> JointMessageMomentsResult:
        expected = {
            "site_names",
            "site_widths",
            "port_slices",
            "width",
            "observations",
            "sequences",
            "centered_activation_square_norm_sum",
            "squared_score_gradient_norm_sum",
            "covariance_trace",
            "fisher_trace",
            "normalizer",
            "score_reduction",
            "accumulation_dtype",
            "algorithm",
            "algorithm_version",
            "format_version",
            "mean",
            "covariance",
            "fisher",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "joint-message moment fields do not match format version 1"
            )
        if type(state["site_names"]) is not tuple:
            raise TypeError("serialized site_names must be a tuple")
        if type(state["site_widths"]) is not tuple:
            raise TypeError("serialized site_widths must be a tuple")
        result = cls(
            site_names=state["site_names"],
            site_widths=state["site_widths"],
            observations=_require_exact_int(
                state["observations"],
                label="observations",
                minimum=1,
            ),
            sequences=_require_exact_int(
                state["sequences"],
                label="sequences",
                minimum=1,
            ),
            mean=_validate_cpu_float64_tensor(
                state["mean"],
                label="mean",
            ),
            covariance=_validate_cpu_float64_tensor(
                state["covariance"],
                label="covariance",
            ),
            fisher=_validate_cpu_float64_tensor(
                state["fisher"],
                label="fisher",
            ),
            centered_activation_square_norm_sum=_require_finite_number(
                state["centered_activation_square_norm_sum"],
                label="centered_activation_square_norm_sum",
                minimum=0.0,
            ),
            squared_score_gradient_norm_sum=_require_finite_number(
                state["squared_score_gradient_norm_sum"],
                label="squared_score_gradient_norm_sum",
                minimum=0.0,
            ),
            normalizer=_require_nonempty_string(
                state["normalizer"],
                label="normalizer",
            ),
            score_reduction=_require_nonempty_string(
                state["score_reduction"],
                label="score_reduction",
            ),
            accumulation_dtype=_require_nonempty_string(
                state["accumulation_dtype"],
                label="accumulation_dtype",
            ),
            algorithm=_require_nonempty_string(
                state["algorithm"],
                label="algorithm",
            ),
            algorithm_version=_require_exact_int(
                state["algorithm_version"],
                label="algorithm_version",
                minimum=1,
            ),
            format_version=_require_exact_int(
                state["format_version"],
                label="format_version",
                minimum=1,
            ),
        )
        expected_slices = tuple(
            (name, value.start, value.stop)
            for name, value in result.port_slices.items()
        )
        if state["port_slices"] != expected_slices:
            raise ValueError(
                "serialized port_slices do not match site widths"
            )
        if _require_exact_int(
            state["width"],
            label="width",
            minimum=1,
        ) != result.width:
            raise ValueError("serialized width does not match site widths")
        for state_key, matrix, label in (
            ("covariance_trace", result.covariance, "covariance_trace"),
            ("fisher_trace", result.fisher, "fisher_trace"),
        ):
            serialized = _require_finite_number(
                state[state_key],
                label=label,
                minimum=0.0,
            )
            if not math.isclose(
                serialized,
                float(matrix.trace().item()),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"serialized {label} does not match its matrix"
                )
        return result


class StreamingJointMessageMoments:
    """Accumulate exact concatenated message moments without retaining rows."""

    def __init__(
        self,
        site_names: Sequence[str],
        *,
        site_widths: Mapping[str, int] | None = None,
        normalizer: str = "valid_activation_positions",
        score_reduction: str = "sum",
    ) -> None:
        self.site_names = _ordered_unique_names(
            site_names,
            label="site_names",
        )
        self.normalizer = _require_nonempty_string(
            normalizer,
            label="normalizer",
        )
        self.score_reduction = _require_nonempty_string(
            score_reduction,
            label="score_reduction",
        )
        if site_widths is not None:
            if not isinstance(site_widths, Mapping):
                raise TypeError("site_widths must be a mapping when provided")
            if set(site_widths) != set(self.site_names):
                raise ValueError(
                    "site_widths must name exactly the ordered sites"
                )
            widths = tuple(
                _require_exact_int(
                    site_widths[name],
                    label=f"site_widths[{name!r}]",
                    minimum=1,
                )
                for name in self.site_names
            )
        else:
            widths = None
        self._site_widths = widths
        self._mean: Tensor | None = None
        self._centered_second_moment: Tensor | None = None
        self._score_gradient_outer_sum: Tensor | None = None
        self._observations = 0
        self._sequences = 0
        if widths is not None:
            self._initialize_storage(widths)

    def _initialize_storage(self, widths: tuple[int, ...]) -> None:
        width = sum(widths)
        self._mean = torch.zeros(width, dtype=torch.float64)
        self._centered_second_moment = torch.zeros(
            (width, width),
            dtype=torch.float64,
        )
        self._score_gradient_outer_sum = torch.zeros(
            (width, width),
            dtype=torch.float64,
        )

    @property
    def site_widths(self) -> tuple[int, ...] | None:
        return self._site_widths

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def sequences(self) -> int:
        return self._sequences

    @property
    def storage_shapes(
        self,
    ) -> (
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        | None
    ):
        if (
            self._mean is None
            or self._centered_second_moment is None
            or self._score_gradient_outer_sum is None
        ):
            return None
        return (
            tuple(self._mean.shape),
            tuple(self._centered_second_moment.shape),
            tuple(self._score_gradient_outer_sum.shape),
        )

    def update(
        self,
        rows: GeneratorMessageScoreGradientRows,
    ) -> StreamingJointMessageMoments:
        if not isinstance(rows, GeneratorMessageScoreGradientRows):
            raise TypeError(
                "rows must be GeneratorMessageScoreGradientRows"
            )
        if tuple(rows.score_gradients) != self.site_names:
            raise ValueError(
                "row gradient sites must equal estimator sites in order"
            )
        if any(name not in rows.activations for name in self.site_names):
            raise ValueError(
                "every estimator site must have activation rows"
            )
        site_widths = tuple(
            rows.activations[name].shape[1]
            for name in self.site_names
        )
        if self._site_widths is not None and site_widths != self._site_widths:
            raise ValueError(
                "row site widths do not match estimator site widths"
            )

        activations = torch.cat(
            [rows.activations[name] for name in self.site_names],
            dim=1,
        ).detach().to(device="cpu", dtype=torch.float64)
        gradients = torch.cat(
            [rows.score_gradients[name] for name in self.site_names],
            dim=1,
        ).detach().to(device="cpu", dtype=torch.float64)
        if activations.shape != gradients.shape:
            raise ValueError(
                "concatenated activation and gradient rows must match"
            )
        if not torch.isfinite(activations).all():
            raise ValueError("joint activation rows must be finite")
        if not torch.isfinite(gradients).all():
            raise ValueError("joint gradient rows must be finite")

        selected_count = activations.shape[0]
        batch_mean = activations.mean(dim=0)
        centered = activations - batch_mean
        batch_second_moment = centered.T @ centered
        batch_gradient_outer = gradients.T @ gradients
        if (
            not torch.isfinite(batch_mean).all()
            or not torch.isfinite(batch_second_moment).all()
            or not torch.isfinite(batch_gradient_outer).all()
        ):
            raise ValueError("joint-message update overflowed float64")

        if self._observations == 0:
            candidate_mean = batch_mean
            candidate_second_moment = batch_second_moment
            candidate_gradient_outer = batch_gradient_outer
        else:
            assert self._mean is not None
            assert self._centered_second_moment is not None
            assert self._score_gradient_outer_sum is not None
            total = self._observations + selected_count
            delta = batch_mean - self._mean
            candidate_mean = (
                self._mean + delta * (selected_count / total)
            )
            correction = torch.outer(delta, delta) * (
                self._observations * selected_count / total
            )
            candidate_second_moment = (
                self._centered_second_moment
                + batch_second_moment
                + correction
            )
            candidate_gradient_outer = (
                self._score_gradient_outer_sum + batch_gradient_outer
            )
        if (
            not torch.isfinite(candidate_mean).all()
            or not torch.isfinite(candidate_second_moment).all()
            or not torch.isfinite(candidate_gradient_outer).all()
        ):
            raise ValueError("joint-message state overflowed float64")

        # Mutate only after the complete candidate update has been validated.
        if self._site_widths is None:
            self._site_widths = site_widths
        self._mean = candidate_mean
        self._centered_second_moment = (
            candidate_second_moment + candidate_second_moment.T
        ) * 0.5
        self._score_gradient_outer_sum = (
            candidate_gradient_outer + candidate_gradient_outer.T
        ) * 0.5
        self._observations += selected_count
        self._sequences += 1
        return self

    def finalize(self) -> JointMessageMomentsResult:
        if self._observations == 0:
            raise ValueError(
                "cannot finalize joint moments without observations"
            )
        assert self._site_widths is not None
        assert self._mean is not None
        assert self._centered_second_moment is not None
        assert self._score_gradient_outer_sum is not None
        covariance = (
            self._centered_second_moment / self._observations
        )
        fisher = (
            self._score_gradient_outer_sum / self._observations
        )
        return JointMessageMomentsResult(
            site_names=self.site_names,
            site_widths=self._site_widths,
            observations=self._observations,
            sequences=self._sequences,
            mean=self._mean.clone(),
            covariance=(covariance + covariance.T) * 0.5,
            fisher=(fisher + fisher.T) * 0.5,
            centered_activation_square_norm_sum=float(
                self._centered_second_moment.trace().item()
            ),
            squared_score_gradient_norm_sum=float(
                self._score_gradient_outer_sum.trace().item()
            ),
            normalizer=self.normalizer,
            score_reduction=self.score_reduction,
        )


__all__ = [
    "GeneratorMessageCapturePlan",
    "GeneratorMessageScoreGradientRows",
    "JointMessageMomentsResult",
    "StreamingJointMessageMoments",
    "iter_generator_message_score_gradient_rows",
]
