"""Model-aware teacher targets and local losses for structured executors."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .adapters.base import ModelAdapter, SequenceContext
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecution,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class StructuredLayerProvenance:
    """Exact source-layer identity shared by targets and calibration state."""

    layer_id: str
    output_site: str
    source_segment_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id:
            raise ValueError("layer_id must be a nonempty string")
        if not isinstance(self.output_site, str) or not self.output_site:
            raise ValueError("output_site must be a nonempty string")
        _require_sha256(
            self.source_segment_fingerprint,
            label="source_segment_fingerprint",
        )


def structured_layer_provenance(
    adapter: ModelAdapter,
    layer_id: str,
) -> StructuredLayerProvenance:
    """Bind one structured layer's semantics and source tensors."""

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    layer = adapter.layer(layer_id)
    if layer.transformer is None:
        raise ValueError(
            f"layer {layer_id!r} has no structured transformer semantics"
        )
    return StructuredLayerProvenance(
        layer_id=layer.id,
        output_site=layer.output_site,
        source_segment_fingerprint=adapter.segment_fingerprint(
            adapter.segment(layer.id)
        ),
    )


@dataclass(frozen=True, slots=True)
class StructuredLayerTargets:
    """Detached teacher activations around both residual branches.

    The optional projection inputs are internal activation pairs used for
    source-parameter-free identification of the two terminal linear maps.
    """

    provenance: StructuredLayerProvenance
    sequence: SequenceContext
    block_input: Tensor
    normalized_attention_input: Tensor
    attention_operator_output: Tensor
    attention_delta: Tensor
    post_attention: Tensor
    normalized_feed_forward_input: Tensor
    feed_forward_operator_output: Tensor
    feed_forward_delta: Tensor
    output: Tensor
    attention_projection_input: Tensor | None = None
    feed_forward_projection_input: Tensor | None = None
    teacher_logits: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError(
                "provenance must be StructuredLayerProvenance"
            )
        if not isinstance(self.sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        tensors = (
            self.block_input,
            self.normalized_attention_input,
            self.attention_operator_output,
            self.attention_delta,
            self.post_attention,
            self.normalized_feed_forward_input,
            self.feed_forward_operator_output,
            self.feed_forward_delta,
            self.output,
        )
        expected = self.block_input.shape
        if (
            self.block_input.ndim != 3
            or any(
                not isinstance(value, Tensor)
                or value.shape != expected
                or not value.is_floating_point()
                for value in tensors
            )
        ):
            raise ValueError(
                "structured layer targets must share one floating "
                "[batch, sequence, feature] shape"
            )
        if expected[:2] != (
            self.sequence.batch_size,
            self.sequence.query_length,
        ):
            raise ValueError(
                "structured targets do not match their sequence context"
            )
        projection_inputs = (
            self.attention_projection_input,
            self.feed_forward_projection_input,
        )
        if any(value is not None for value in projection_inputs) and (
            any(
                not isinstance(value, Tensor)
                or value.ndim != 3
                or value.shape[:2] != expected[:2]
                or not value.is_floating_point()
                for value in projection_inputs
            )
        ):
            raise ValueError(
                "structured terminal projection inputs must be provided "
                "together as floating [batch, sequence, feature] tensors"
            )
        if self.teacher_logits is not None and (
            not isinstance(self.teacher_logits, Tensor)
            or self.teacher_logits.ndim != 2
            or not self.teacher_logits.is_floating_point()
        ):
            raise ValueError(
                "teacher_logits must have shape "
                "[selected_positions, vocabulary] when provided"
            )

    @property
    def layer_id(self) -> str:
        return self.provenance.layer_id

    @property
    def output_site(self) -> str:
        return self.provenance.output_site


def capture_structured_layer_targets(
    adapter: ModelAdapter,
    layer_id: str,
    model_inputs: Mapping[str, Tensor],
    *,
    teacher_logit_positions: Tensor | None = None,
    provenance: StructuredLayerProvenance | None = None,
) -> StructuredLayerTargets:
    """Capture one native layer's model-described residual stages."""

    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter must be a ModelAdapter")
    layer = adapter.layer(layer_id)
    semantics = layer.transformer
    if semantics is None:
        raise ValueError(
            f"layer {layer_id!r} has no structured transformer semantics"
        )
    live_provenance = structured_layer_provenance(adapter, layer_id)
    resolved_provenance = (
        live_provenance if provenance is None else provenance
    )
    if (
        not isinstance(
            resolved_provenance,
            StructuredLayerProvenance,
        )
        or resolved_provenance != live_provenance
    ):
        raise ValueError(
            "structured target provenance does not match the source layer"
        )
    attention, feed_forward = semantics.stages
    operator_sites = semantics.operator_sites
    terminal_projection_sites = (
        ()
        if operator_sites is None
        else (
            operator_sites.attention_context,
            operator_sites.feed_forward_down_input,
        )
    )
    sites = (
        layer.input_site,
        attention.normalized_input_site,
        attention.operator_output_site,
        attention.delta_site,
        attention.output_site,
        feed_forward.normalized_input_site,
        feed_forward.operator_output_site,
        feed_forward.delta_site,
        layer.output_site,
        *terminal_projection_sites,
    )
    with torch.no_grad():
        run = adapter.forward(
            model_inputs,
            capture_sites=sites,
            retain_gradients=False,
        )
    values = {
        name: run.activations[name].detach().clone()
        for name in sites
    }
    teacher_logits = None
    if teacher_logit_positions is not None:
        if (
            not isinstance(teacher_logit_positions, Tensor)
            or teacher_logit_positions.dtype is not torch.bool
            or teacher_logit_positions.shape
            != (
                run.sequence.batch_size,
                run.sequence.query_length,
            )
            or not bool(teacher_logit_positions.any())
        ):
            raise ValueError(
                "teacher_logit_positions must be a nonempty boolean "
                "[batch, sequence] tensor"
            )
        teacher_logits = (
            run.logits[
                teacher_logit_positions.to(device=run.logits.device)
            ]
            .detach()
            .clone()
        )
    return StructuredLayerTargets(
        provenance=resolved_provenance,
        sequence=run.sequence,
        block_input=values[layer.input_site],
        normalized_attention_input=values[
            attention.normalized_input_site
        ],
        attention_operator_output=values[
            attention.operator_output_site
        ],
        attention_delta=values[attention.delta_site],
        post_attention=values[attention.output_site],
        normalized_feed_forward_input=values[
            feed_forward.normalized_input_site
        ],
        feed_forward_operator_output=values[
            feed_forward.operator_output_site
        ],
        feed_forward_delta=values[feed_forward.delta_site],
        output=values[layer.output_site],
        attention_projection_input=(
            None
            if operator_sites is None
            else values[operator_sites.attention_context]
        ),
        feed_forward_projection_input=(
            None
            if operator_sites is None
            else values[operator_sites.feed_forward_down_input]
        ),
        teacher_logits=teacher_logits,
    )


@dataclass(frozen=True, slots=True)
class StructuredLayerDistillationWeights:
    """Weights for explicitly supervised layer-stage boundaries."""

    normalized_attention_input: float = 0.10
    attention_operator_output: float = 0.25
    attention_delta: float = 1.0
    post_attention: float = 0.25
    normalized_feed_forward_input: float = 0.10
    feed_forward_operator_output: float = 0.25
    feed_forward_delta: float = 1.0
    output: float = 1.0
    output_fisher: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(
            float(getattr(self, name))
            for name in self.__dataclass_fields__
        )
        if any(
            not torch.isfinite(torch.tensor(value)) or value < 0
            for value in values
        ):
            raise ValueError(
                "structured distillation weights must be finite and "
                "nonnegative"
            )
        if not any(value > 0 for value in values):
            raise ValueError(
                "at least one structured distillation weight must be positive"
            )


@dataclass(frozen=True, slots=True)
class StructuredLayerDistillationScales:
    """Calibration-A per-coordinate scales for every supervised stage."""

    provenance: StructuredLayerProvenance
    calibration_split_sha256: str
    normalized_attention_input: Tensor
    attention_operator_output: Tensor
    attention_delta: Tensor
    post_attention: Tensor
    normalized_feed_forward_input: Tensor
    feed_forward_operator_output: Tensor
    feed_forward_delta: Tensor
    output: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError(
                "provenance must be StructuredLayerProvenance"
            )
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        names = (
            "normalized_attention_input",
            "attention_operator_output",
            "attention_delta",
            "post_attention",
            "normalized_feed_forward_input",
            "feed_forward_operator_output",
            "feed_forward_delta",
            "output",
        )
        values = tuple(getattr(self, name) for name in names)
        if (
            any(
                not isinstance(value, Tensor)
                or value.ndim != 1
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or not bool((value > 0).all())
                for value in values
            )
            or len({tuple(value.shape) for value in values}) != 1
        ):
            raise ValueError(
                "structured distillation scales must be finite positive "
                "vectors with one shared width"
            )
        for name, value in zip(names, values, strict=True):
            object.__setattr__(
                self,
                name,
                value.detach().to(
                    device="cpu",
                    dtype=torch.float64,
                ).clone(),
            )

    @property
    def width(self) -> int:
        return int(self.output.shape[0])


@dataclass(frozen=True, slots=True)
class StructuredOutputFisherMetric:
    """Validated Fisher quadratic in standardized output-error coordinates.

    Validation and canonicalization happen once when the calibration artifact
    is constructed.  Training steps can therefore consume the quadratic
    without repeatedly running an eigendecomposition.
    """

    provenance: StructuredLayerProvenance
    calibration_split_sha256: str
    delta_scale: Tensor
    standardized_coordinate_metric: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, StructuredLayerProvenance):
            raise TypeError(
                "provenance must be StructuredLayerProvenance"
            )
        _require_sha256(
            self.calibration_split_sha256,
            label="calibration_split_sha256",
        )
        if (
            not isinstance(self.delta_scale, Tensor)
            or self.delta_scale.ndim != 1
            or not self.delta_scale.is_floating_point()
            or self.delta_scale.numel() == 0
            or not isinstance(
                self.standardized_coordinate_metric,
                Tensor,
            )
            or self.standardized_coordinate_metric.ndim != 2
            or not self.standardized_coordinate_metric.is_floating_point()
            or self.standardized_coordinate_metric.shape
            != (self.delta_scale.numel(), self.delta_scale.numel())
        ):
            raise ValueError(
                "structured output Fisher requires a scale vector and "
                "square standardized-coordinate metric with one shared "
                "nonzero width"
            )
        scale = self.delta_scale.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).clone()
        metric = self.standardized_coordinate_metric.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).clone()
        if (
            not bool(torch.isfinite(scale).all())
            or not bool((scale > 0).all())
            or not bool(torch.isfinite(metric).all())
        ):
            raise ValueError(
                "structured output Fisher values must be finite and "
                "delta scales must be strictly positive"
        )
        magnitude = max(1.0, float(metric.abs().max().item()))
        symmetry_tolerance = 1e-7 * magnitude
        eigenvalue_tolerance = max(
            1e-12,
            1e-10 * max(1, int(metric.shape[0])) * magnitude,
        )
        if not torch.allclose(
            metric,
            metric.transpose(0, 1),
            rtol=1e-7,
            atol=symmetry_tolerance,
        ):
            raise ValueError(
                "structured output Fisher metric must be symmetric"
            )
        metric = 0.5 * (metric + metric.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        minimum_eigenvalue = float(eigenvalues.min().item())
        if minimum_eigenvalue < -eigenvalue_tolerance:
            raise ValueError(
                "structured output Fisher metric must be positive "
                "semidefinite"
            )
        eigenvalues = eigenvalues.clamp_min(0)
        if float(eigenvalues.sum().item()) <= eigenvalue_tolerance:
            raise ValueError(
                "structured output Fisher metric must contain nonzero "
                "sensitivity"
            )
        metric = (
            eigenvectors
            * eigenvalues.unsqueeze(0)
        ) @ eigenvectors.transpose(0, 1)
        metric = 0.5 * (metric + metric.transpose(0, 1))
        object.__setattr__(self, "delta_scale", scale)
        object.__setattr__(
            self,
            "standardized_coordinate_metric",
            metric,
        )

    @property
    def width(self) -> int:
        return int(self.delta_scale.shape[0])

    @classmethod
    def from_raw_fisher(
        cls,
        *,
        provenance: StructuredLayerProvenance,
        calibration_split_sha256: str,
        delta_scale: Tensor,
        raw_fisher: Tensor,
    ) -> StructuredOutputFisherMetric:
        """Transform ``e.T @ F @ e`` into coordinates ``z = e / scale``."""

        if (
            not isinstance(delta_scale, Tensor)
            or delta_scale.ndim != 1
            or not delta_scale.is_floating_point()
            or not isinstance(raw_fisher, Tensor)
            or raw_fisher.ndim != 2
            or not raw_fisher.is_floating_point()
            or raw_fisher.shape
            != (delta_scale.numel(), delta_scale.numel())
        ):
            raise ValueError(
                "raw Fisher and delta scale must share one nonzero width"
            )
        scale = delta_scale.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        fisher = raw_fisher.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        standardized = (
            scale.unsqueeze(1)
            * fisher
            * scale.unsqueeze(0)
        )
        return cls(
            provenance=provenance,
            calibration_split_sha256=calibration_split_sha256,
            delta_scale=scale,
            standardized_coordinate_metric=standardized,
        )


def estimate_structured_layer_scales(
    batches: Sequence[StructuredLayerTargets],
    *,
    calibration_split_sha256: str,
    floor: float = 1e-4,
    relative_median_floor: float = 0.0,
) -> StructuredLayerDistillationScales:
    """Estimate stable stage-local coordinate systems on calibration A."""

    if not batches:
        raise ValueError("structured scale batches cannot be empty")
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if (
        not isinstance(floor, (float, int))
        or isinstance(floor, bool)
        or not torch.isfinite(torch.tensor(float(floor)))
        or floor <= 0
    ):
        raise ValueError("structured scale floor must be finite and positive")
    if (
        not isinstance(relative_median_floor, (float, int))
        or isinstance(relative_median_floor, bool)
        or not torch.isfinite(
            torch.tensor(float(relative_median_floor))
        )
        or not 0 <= relative_median_floor <= 1
    ):
        raise ValueError(
            "structured relative median scale floor must be finite in [0, 1]"
        )
    width = int(batches[0].block_input.shape[-1])
    provenance = batches[0].provenance
    names = (
        "normalized_attention_input",
        "attention_operator_output",
        "attention_delta",
        "post_attention",
        "normalized_feed_forward_input",
        "feed_forward_operator_output",
        "feed_forward_delta",
        "output",
    )
    sums = {
        name: torch.zeros(width, dtype=torch.float64)
        for name in names
    }
    rows = 0
    for targets in batches:
        if (
            not isinstance(targets, StructuredLayerTargets)
            or targets.block_input.shape[-1] != width
            or targets.provenance != provenance
        ):
            raise ValueError(
                "structured scale batches must share one target width "
                "and source-layer provenance"
            )
        valid = targets.sequence.query_valid_mask
        if not torch.equal(valid, targets.sequence.key_valid_mask):
            raise ValueError(
                "structured scale estimation requires middle-layer demand"
            )
        count = int(valid.sum().item())
        if count == 0:
            continue
        rows += count
        values = {
            "normalized_attention_input": (
                targets.normalized_attention_input
            ),
            "attention_operator_output": (
                targets.attention_operator_output
            ),
            "attention_delta": targets.attention_delta,
            # Both residual-state errors are exactly branch-delta errors
            # because every candidate receives the same block input.
            "post_attention": targets.attention_delta,
            "normalized_feed_forward_input": (
                targets.normalized_feed_forward_input
            ),
            "feed_forward_operator_output": (
                targets.feed_forward_operator_output
            ),
            "feed_forward_delta": targets.feed_forward_delta,
            "output": targets.output - targets.block_input,
        }
        for name, value in values.items():
            sums[name].add_(
                value[valid]
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .square()
                .sum(dim=0)
            )
    if rows == 0:
        raise ValueError("structured scale batches have no valid rows")
    scales = {}
    for name, value in sums.items():
        raw = (value / rows).sqrt()
        robust_floor = max(
            float(floor),
            float(relative_median_floor) * float(raw.median().item()),
        )
        scales[name] = raw.clamp_min(robust_floor)
    return StructuredLayerDistillationScales(
        provenance=provenance,
        calibration_split_sha256=calibration_split_sha256,
        **scales,
    )


@dataclass(frozen=True, slots=True)
class StructuredLayerDistillationLoss:
    """Differentiable aggregate plus inspectable component losses."""

    total: Tensor
    coordinate_total: Tensor
    energy_total: Tensor
    normalized_attention_input: Tensor
    attention_operator_output: Tensor
    attention_delta: Tensor
    post_attention: Tensor
    normalized_feed_forward_input: Tensor
    feed_forward_operator_output: Tensor
    feed_forward_delta: Tensor
    output: Tensor
    output_fisher: Tensor


def _masked_losses(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    scale: Tensor | None,
) -> tuple[Tensor, Tensor]:
    error = prediction.float() - target.float()
    raw_mse = error[valid_mask].square().mean()
    if scale is not None:
        resolved_scale = scale.to(
            device=prediction.device,
            dtype=torch.float32,
        )
        coordinate_error = error / resolved_scale
        energy_denominator = resolved_scale.square().mean()
    else:
        coordinate_error = error
        energy_denominator = (
            target.float()[valid_mask].square().mean().detach()
        )
    coordinate = coordinate_error[valid_mask].square().mean()
    energy = raw_mse / energy_denominator.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return coordinate, energy


_RMSNORM_INITIALIZATION_ALGORITHM = (
    "calibration_a_activation_pair_coordinate_least_squares_v1"
)


def initialize_structured_rmsnorms_from_targets_(
    executor: StructuredTransformerLayerExecutor,
    batches: Sequence[StructuredLayerTargets],
    *,
    calibration_split_sha256: str,
) -> dict[str, object]:
    """Initialize four residual RMSNorm gains from calibration-A pairs.

    A Gemma unit-offset RMSNorm is linear in its learned coordinate gain once
    the input RMS has been computed.  The source-free executor can therefore
    recover those gains by coordinate-wise least squares over captured
    activation pairs without reading or copying a source module parameter.
    """

    if not isinstance(executor, StructuredTransformerLayerExecutor):
        raise TypeError(
            "executor must be StructuredTransformerLayerExecutor"
        )
    if executor.owns_source_model_weights:
        raise ValueError(
            "RMSNorm initialization refuses source-weight-contaminated "
            "executors"
        )
    if not batches:
        raise ValueError("RMSNorm initialization batches cannot be empty")
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    provenance = batches[0].provenance
    width = executor.width
    pairs = (
        (
            "attention_input_norm",
            "block_input",
            "normalized_attention_input",
        ),
        (
            "attention_output_norm",
            "attention_operator_output",
            "attention_delta",
        ),
        (
            "feed_forward_input_norm",
            "post_attention",
            "normalized_feed_forward_input",
        ),
        (
            "feed_forward_output_norm",
            "feed_forward_operator_output",
            "feed_forward_delta",
        ),
    )
    reports: dict[str, object] = {}
    valid_rows = 0
    for targets in batches:
        if (
            not isinstance(targets, StructuredLayerTargets)
            or targets.provenance != provenance
            or targets.block_input.shape[-1] != width
        ):
            raise ValueError(
                "RMSNorm initialization batches must share width and "
                "provenance"
            )
        valid_rows += int(targets.sequence.query_valid_mask.sum().item())
    if valid_rows <= 0:
        raise ValueError("RMSNorm initialization has no valid rows")

    for module_name, input_name, target_name in pairs:
        module = getattr(executor, module_name)
        numerator = torch.zeros(width, dtype=torch.float64)
        denominator = torch.zeros(width, dtype=torch.float64)
        target_energy = torch.zeros(width, dtype=torch.float64)
        epsilon = float(module.spec.epsilon)
        for targets in batches:
            valid = targets.sequence.query_valid_mask
            inputs = getattr(targets, input_name).float()
            desired = getattr(targets, target_name).float()
            base = inputs * torch.rsqrt(
                inputs.square().mean(dim=-1, keepdim=True) + epsilon
            )
            base_rows = base[valid].detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            desired_rows = desired[valid].detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            numerator.add_((base_rows * desired_rows).sum(dim=0))
            denominator.add_(base_rows.square().sum(dim=0))
            target_energy.add_(desired_rows.square().sum(dim=0))
        identifiable = denominator > torch.finfo(torch.float64).tiny
        effective_gain = torch.ones(width, dtype=torch.float64)
        effective_gain[identifiable] = (
            numerator[identifiable] / denominator[identifiable]
        )
        if not bool(torch.isfinite(effective_gain).all()):
            raise RuntimeError(
                f"RMSNorm initialization for {module_name!r} is nonfinite"
            )
        weight = effective_gain - 1.0
        with torch.no_grad():
            module.weight.copy_(
                weight.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
            )
        minimum_sse = target_energy.clone()
        minimum_sse[identifiable] -= (
            numerator[identifiable].square()
            / denominator[identifiable]
        )
        minimum_sse.clamp_min_(0)
        total_target_energy = float(target_energy.sum().item())
        fit_nrmse = math.sqrt(
            float(minimum_sse.sum().item())
            / max(total_target_energy, torch.finfo(torch.float64).tiny)
        )
        reports[module_name] = {
            "width": width,
            "identified_coordinates": int(identifiable.sum().item()),
            "fit_nrmse": fit_nrmse,
            "weight_minimum": float(weight.min().item()),
            "weight_median": float(weight.median().item()),
            "weight_maximum": float(weight.max().item()),
            "weight_rms": float(weight.square().mean().sqrt().item()),
        }

    return {
        "algorithm": _RMSNORM_INITIALIZATION_ALGORITHM,
        "calibration_split_sha256": calibration_split_sha256,
        "provenance": {
            "layer_id": provenance.layer_id,
            "output_site": provenance.output_site,
            "source_segment_fingerprint": (
                provenance.source_segment_fingerprint
            ),
        },
        "valid_rows": valid_rows,
        "source_module_or_parameter_read": False,
        "direct_source_tensor_copy": False,
        "normalizations": reports,
    }


_TERMINAL_PROJECTION_REFIT_ALGORITHM = (
    "calibration_activation_only_float64_ridge_normal_equations_v1"
)


class _Float64LinearNormalEquations:
    """CPU sufficient statistics for one activation-to-activation map."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        *,
        fit_bias: bool,
    ) -> None:
        self.input_width = input_width
        self.output_width = output_width
        self.fit_bias = fit_bias
        coefficient_width = input_width + int(fit_bias)
        self.gram = torch.zeros(
            coefficient_width,
            coefficient_width,
            dtype=torch.float64,
        )
        self.cross = torch.zeros(
            coefficient_width,
            output_width,
            dtype=torch.float64,
        )
        self.target_energy = torch.zeros((), dtype=torch.float64)
        self.rows = 0

    def add(
        self,
        features: Tensor,
        targets: Tensor,
        valid_mask: Tensor,
    ) -> None:
        if (
            features.ndim != 3
            or targets.ndim != 3
            or features.shape[:2] != targets.shape[:2]
            or features.shape[-1] != self.input_width
            or targets.shape[-1] != self.output_width
            or valid_mask.dtype is not torch.bool
            or valid_mask.shape != features.shape[:2]
            or valid_mask.device != features.device
            or targets.device != features.device
        ):
            raise ValueError(
                "terminal projection rows have incompatible shapes or "
                "devices"
            )
        feature_rows = features[valid_mask].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        target_rows = targets[valid_mask].detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if feature_rows.shape[0] == 0:
            return
        if (
            not bool(torch.isfinite(feature_rows).all())
            or not bool(torch.isfinite(target_rows).all())
        ):
            raise ValueError(
                "terminal projection valid rows must be finite"
            )
        if self.fit_bias:
            feature_rows = torch.cat(
                (
                    feature_rows,
                    torch.ones(
                        feature_rows.shape[0],
                        1,
                        dtype=torch.float64,
                    ),
                ),
                dim=1,
            )
        self.gram.add_(feature_rows.mT @ feature_rows)
        self.cross.add_(feature_rows.mT @ target_rows)
        self.target_energy.add_(target_rows.square().sum())
        self.rows += feature_rows.shape[0]

    def solve(self, ridge: float) -> Tensor:
        if self.rows <= 0:
            raise ValueError("terminal projection refit has no valid rows")
        system = self.gram.clone()
        system.diagonal()[: self.input_width].add_(ridge)
        solution = torch.linalg.solve(system, self.cross)
        if not bool(torch.isfinite(solution).all()):
            raise RuntimeError(
                "terminal projection ridge solution is nonfinite"
            )
        return solution

    def operator_nrmse(self, coefficients: Tensor) -> float:
        expected = (self.gram.shape[0], self.output_width)
        if (
            coefficients.shape != expected
            or coefficients.dtype is not torch.float64
            or coefficients.device.type != "cpu"
        ):
            raise ValueError(
                "terminal projection coefficients are not canonical"
            )
        quadratic = (coefficients * (self.gram @ coefficients)).sum()
        linear = (coefficients * self.cross).sum()
        squared_error = self.target_energy - 2.0 * linear + quadratic
        numerical_scale = max(
            float(self.target_energy.abs().item()),
            float(linear.abs().item()),
            float(quadratic.abs().item()),
            torch.finfo(torch.float64).tiny,
        )
        if float(squared_error.item()) < -1e-10 * numerical_scale:
            raise RuntimeError(
                "terminal projection error became materially negative"
            )
        squared_error.clamp_min_(0)
        denominator = max(
            float(self.target_energy.item()),
            torch.finfo(torch.float64).tiny,
        )
        return math.sqrt(float(squared_error.item()) / denominator)


def _linear_coefficients(linear: nn.Linear) -> Tensor:
    coefficients = linear.weight.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).mT
    if linear.bias is not None:
        coefficients = torch.cat(
            (
                coefficients,
                linear.bias.detach()
                .to(device="cpu", dtype=torch.float64)
                .unsqueeze(0),
            ),
            dim=0,
        )
    return coefficients


def _write_linear_coefficients_(
    linear: nn.Linear,
    coefficients: Tensor,
) -> None:
    input_width = linear.in_features
    expected = (
        input_width + int(linear.bias is not None),
        linear.out_features,
    )
    if coefficients.shape != expected:
        raise ValueError(
            "terminal projection solution has the wrong shape"
        )
    with torch.no_grad():
        linear.weight.copy_(
            coefficients[:input_width]
            .mT.to(
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
        )
        if linear.bias is not None:
            linear.bias.copy_(
                coefficients[-1].to(
                    device=linear.bias.device,
                    dtype=linear.bias.dtype,
                )
            )


def refit_structured_terminal_projections_from_targets_(
    executor: StructuredTransformerLayerExecutor,
    batches: Sequence[StructuredLayerTargets],
    *,
    calibration_split_sha256: str,
    ridge: float = 1e-6,
) -> dict[str, object]:
    """Ridge-fit ``o_proj`` and ``down_proj`` from activation pairs only.

    Captured pre-terminal features and operator outputs supply the regression
    rows. Only valid query rows enter CPU float64 normal equations; neither a
    source module nor source parameters are accepted by this API.
    """

    if not isinstance(executor, StructuredTransformerLayerExecutor):
        raise TypeError(
            "executor must be a StructuredTransformerLayerExecutor"
        )
    if executor.owns_source_model_weights:
        raise ValueError(
            "terminal projection refit refuses source-weight-contaminated "
            "executors"
        )
    if not batches:
        raise ValueError(
            "terminal projection refit batches cannot be empty"
        )
    _require_sha256(
        calibration_split_sha256,
        label="calibration_split_sha256",
    )
    if (
        not isinstance(ridge, (float, int))
        or isinstance(ridge, bool)
        or not math.isfinite(float(ridge))
        or float(ridge) <= 0
    ):
        raise ValueError("ridge must be finite and positive")
    resolved_ridge = float(ridge)
    provenance = batches[0].provenance
    width = executor.width
    valid_rows = 0
    for targets in batches:
        if (
            not isinstance(targets, StructuredLayerTargets)
            or targets.provenance != provenance
            or targets.block_input.shape[-1] != width
            or targets.attention_projection_input is None
            or targets.feed_forward_projection_input is None
        ):
            raise ValueError(
                "terminal projection refit batches must share width and "
                "provenance and contain both projection inputs"
            )
        valid_rows += int(
            targets.sequence.query_valid_mask.sum().item()
        )
    if valid_rows <= 0:
        raise ValueError("terminal projection refit has no valid rows")

    attention_linear = executor.attention.o_proj
    feed_forward_linear = executor.feed_forward.down_proj
    attention_equations = _Float64LinearNormalEquations(
        attention_linear.in_features,
        attention_linear.out_features,
        fit_bias=attention_linear.bias is not None,
    )
    feed_forward_equations = _Float64LinearNormalEquations(
        feed_forward_linear.in_features,
        feed_forward_linear.out_features,
        fit_bias=feed_forward_linear.bias is not None,
    )
    executor_fingerprint_before = executor.execution_fingerprint()
    attention_before = _linear_coefficients(attention_linear)
    feed_forward_before = _linear_coefficients(feed_forward_linear)
    with torch.no_grad():
        for targets in batches:
            valid = targets.sequence.query_valid_mask
            attention_features = targets.attention_projection_input
            feed_forward_features = targets.feed_forward_projection_input
            assert attention_features is not None
            assert feed_forward_features is not None
            attention_equations.add(
                attention_features,
                targets.attention_operator_output,
                valid,
            )
            feed_forward_equations.add(
                feed_forward_features,
                targets.feed_forward_operator_output,
                valid,
            )
    if (
        attention_equations.rows != valid_rows
        or feed_forward_equations.rows != valid_rows
    ):
        raise RuntimeError(
            "terminal projection normal-equation row accounting drifted"
        )

    attention_solution = attention_equations.solve(resolved_ridge)
    feed_forward_solution = feed_forward_equations.solve(resolved_ridge)
    projection_reports = {}
    for name, equations, before, solution, linear in (
        (
            "attention.o_proj",
            attention_equations,
            attention_before,
            attention_solution,
            attention_linear,
        ),
        (
            "feed_forward.down_proj",
            feed_forward_equations,
            feed_forward_before,
            feed_forward_solution,
            feed_forward_linear,
        ),
    ):
        stored_solution = solution.to(
            dtype=linear.weight.dtype
        ).to(dtype=torch.float64)
        projection_reports[name] = {
            "rows": equations.rows,
            "input_width": equations.input_width,
            "output_width": equations.output_width,
            "bias_fitted": equations.fit_bias,
            "bias_regularized": False,
            "ridge": resolved_ridge,
            "normal_equations_dtype": "float64",
            "normal_equations_device": "cpu",
            "pre_refit_operator_nrmse": equations.operator_nrmse(
                before
            ),
            "post_refit_operator_nrmse": equations.operator_nrmse(
                stored_solution
            ),
            "weight_rms": float(
                stored_solution[: equations.input_width]
                .square()
                .mean()
                .sqrt()
                .item()
            ),
            "bias_rms": (
                float(
                    stored_solution[-1]
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                )
                if equations.fit_bias
                else None
            ),
        }

    _write_linear_coefficients_(attention_linear, attention_solution)
    _write_linear_coefficients_(
        feed_forward_linear,
        feed_forward_solution,
    )
    if executor.owns_source_model_weights:
        raise RuntimeError(
            "terminal projection refit changed source-weight origin"
        )
    return {
        "algorithm": _TERMINAL_PROJECTION_REFIT_ALGORITHM,
        "calibration_split_sha256": calibration_split_sha256,
        "provenance": {
            "layer_id": provenance.layer_id,
            "output_site": provenance.output_site,
            "source_segment_fingerprint": (
                provenance.source_segment_fingerprint
            ),
        },
        "valid_rows": valid_rows,
        "regularization": {
            "ridge": resolved_ridge,
            "objective": (
                "summed_squared_operator_error_plus_ridge_weight_l2"
            ),
            "bias_regularized": False,
        },
        "activation_pairs": {
            "attention.o_proj": {
                "input": (
                    "StructuredLayerTargets.attention_projection_input"
                ),
                "target": (
                    "StructuredLayerTargets.attention_operator_output"
                ),
            },
            "feed_forward.down_proj": {
                "input": (
                    "StructuredLayerTargets.feed_forward_projection_input"
                ),
                "target": (
                    "StructuredLayerTargets.feed_forward_operator_output"
                ),
            },
        },
        "projections": projection_reports,
        "executor_fingerprint_before": executor_fingerprint_before,
        "executor_fingerprint_after": executor.execution_fingerprint(),
        "source_module_or_parameter_read": False,
        "direct_source_tensor_copy": False,
        "source_weight_origin_before": False,
        "source_weight_origin_after": False,
    }


def structured_layer_distillation_loss(
    prediction: StructuredTransformerLayerExecution,
    targets: StructuredLayerTargets,
    valid_mask: Tensor,
    *,
    weights: StructuredLayerDistillationWeights = (
        StructuredLayerDistillationWeights()
    ),
    scales: StructuredLayerDistillationScales | None = None,
    coordinate_loss_weight: float = 1.0,
    energy_loss_weight: float = 0.0,
    fisher_positions: Tensor | None = None,
    output_fisher: StructuredOutputFisherMetric | None = None,
) -> StructuredLayerDistillationLoss:
    """Supervise both Gemma residual branches and an optional Fisher metric."""

    if not isinstance(prediction, StructuredTransformerLayerExecution):
        raise TypeError(
            "prediction must be StructuredTransformerLayerExecution"
        )
    if not isinstance(targets, StructuredLayerTargets):
        raise TypeError("targets must be StructuredLayerTargets")
    if not isinstance(weights, StructuredLayerDistillationWeights):
        raise TypeError(
            "weights must be StructuredLayerDistillationWeights"
        )
    if scales is not None and (
        not isinstance(scales, StructuredLayerDistillationScales)
        or scales.width != targets.block_input.shape[-1]
        or scales.provenance != targets.provenance
    ):
        raise ValueError(
            "scales must match the structured target width and provenance"
        )
    for name, value in (
        ("coordinate_loss_weight", coordinate_loss_weight),
        ("energy_loss_weight", energy_loss_weight),
    ):
        if (
            not isinstance(value, (float, int))
            or isinstance(value, bool)
            or not bool(torch.isfinite(torch.tensor(float(value))))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if coordinate_loss_weight == 0 and energy_loss_weight == 0:
        raise ValueError(
            "coordinate or energy loss weight must be positive"
        )
    if (
        not isinstance(valid_mask, Tensor)
        or valid_mask.dtype is not torch.bool
        or valid_mask.shape != targets.block_input.shape[:2]
        or not bool(valid_mask.any())
    ):
        raise ValueError(
            "valid_mask must be a nonempty boolean [batch, sequence] tensor"
        )
    if valid_mask.device != prediction.output.device:
        raise ValueError(
            "valid_mask and structured prediction must share a device"
        )
    target_valid = targets.sequence.query_valid_mask
    if (
        target_valid.device != valid_mask.device
        or bool((valid_mask & ~target_valid).any())
    ):
        raise ValueError(
            "valid_mask must be a subset of the captured valid queries"
        )
    pairs = (
        (
            "normalized_attention_input",
            prediction.normalized_attention_input,
            targets.normalized_attention_input,
        ),
        (
            "attention_operator_output",
            prediction.attention_operator_output,
            targets.attention_operator_output,
        ),
        (
            "attention_delta",
            prediction.attention_delta,
            targets.attention_delta,
        ),
        (
            "post_attention",
            prediction.post_attention,
            targets.post_attention,
        ),
        (
            "normalized_feed_forward_input",
            prediction.normalized_feed_forward_input,
            targets.normalized_feed_forward_input,
        ),
        (
            "feed_forward_operator_output",
            prediction.feed_forward_operator_output,
            targets.feed_forward_operator_output,
        ),
        (
            "feed_forward_delta",
            prediction.feed_forward_delta,
            targets.feed_forward_delta,
        ),
        ("output", prediction.output, targets.output),
    )
    coordinate_component: dict[str, Tensor] = {}
    energy_component: dict[str, Tensor] = {}
    component: dict[str, Tensor] = {}
    for name, predicted, target in pairs:
        if predicted.shape != target.shape:
            raise ValueError(
                f"structured prediction {name!r} has the wrong shape"
            )
        coordinate, energy = _masked_losses(
            predicted,
            target,
            valid_mask,
            None if scales is None else getattr(scales, name),
        )
        coordinate_component[name] = coordinate
        energy_component[name] = energy
        component[name] = (
            float(coordinate_loss_weight) * coordinate
            + float(energy_loss_weight) * energy
        )

    if fisher_positions is None and output_fisher is None:
        fisher = prediction.output.new_zeros((), dtype=torch.float32)
        if weights.output_fisher != 0:
            raise ValueError(
                "output_fisher weight requires Fisher metric inputs"
            )
    elif fisher_positions is None or output_fisher is None:
        raise ValueError(
            "Fisher positions and output Fisher metric must be provided "
            "together"
        )
    else:
        width = targets.output.shape[-1]
        if (
            not isinstance(fisher_positions, Tensor)
            or fisher_positions.dtype is not torch.bool
            or fisher_positions.shape != valid_mask.shape
            or fisher_positions.device != valid_mask.device
            or bool((fisher_positions & ~valid_mask).any())
            or not bool(fisher_positions.any())
            or not isinstance(
                output_fisher,
                StructuredOutputFisherMetric,
            )
            or output_fisher.width != width
            or output_fisher.provenance != targets.provenance
            or (
                scales is not None
                and output_fisher.calibration_split_sha256
                != scales.calibration_split_sha256
            )
        ):
            raise ValueError("structured Fisher metric inputs are invalid")
        error = (
            prediction.output.float() - targets.output.float()
        ) / output_fisher.delta_scale.to(
            device=prediction.output.device,
            dtype=torch.float32,
        )
        selected = error[fisher_positions]
        metric = output_fisher.standardized_coordinate_metric.to(
            device=prediction.output.device,
            dtype=torch.float32,
        )
        fisher = torch.einsum(
            "ni,ij,nj->n",
            selected,
            metric,
            selected,
        ).clamp_min(0).mean() / width

    coordinate_total = sum(
        float(getattr(weights, name)) * loss
        for name, loss in coordinate_component.items()
    )
    energy_total = sum(
        float(getattr(weights, name)) * loss
        for name, loss in energy_component.items()
    )
    total = (
        float(coordinate_loss_weight) * coordinate_total
        + float(energy_loss_weight) * energy_total
        + float(weights.output_fisher) * fisher
    )
    return StructuredLayerDistillationLoss(
        total=total,
        coordinate_total=coordinate_total,
        energy_total=energy_total,
        normalized_attention_input=component[
            "normalized_attention_input"
        ],
        attention_operator_output=component[
            "attention_operator_output"
        ],
        attention_delta=component["attention_delta"],
        post_attention=component["post_attention"],
        normalized_feed_forward_input=component[
            "normalized_feed_forward_input"
        ],
        feed_forward_operator_output=component[
            "feed_forward_operator_output"
        ],
        feed_forward_delta=component["feed_forward_delta"],
        output=component["output"],
        output_fisher=fisher,
    )


__all__ = [
    "StructuredLayerDistillationLoss",
    "StructuredLayerDistillationScales",
    "StructuredLayerDistillationWeights",
    "StructuredLayerProvenance",
    "StructuredLayerTargets",
    "StructuredOutputFisherMetric",
    "capture_structured_layer_targets",
    "estimate_structured_layer_scales",
    "initialize_structured_rmsnorms_from_targets_",
    "refit_structured_terminal_projections_from_targets_",
    "structured_layer_provenance",
    "structured_layer_distillation_loss",
]
