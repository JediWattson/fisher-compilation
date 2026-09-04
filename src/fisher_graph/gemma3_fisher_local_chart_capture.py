"""Exact layer-4 capture primitives for the Fisher local-chart probe.

The Gemma 3 decoder layer is a sequential two-stage residual update::

    h_A    = h + post_attention_rms_norm(q_A)
    h_plus = h_A + post_feed_forward_rms_norm(q_M)

This module preserves those native boundaries and decomposes the two branch
outputs without independently normalizing their components.  A realized
Gemma RMSNorm scale is computed once from the complete branch output and is
then shared by every attention-head or MLP probe-block contribution.

The model-facing entry point intentionally performs one teacher forward.  It
temporarily observes eager attention probabilities, replaces
``layer.4.output`` with a detached leaf of the same value, and differentiates
the summed causal NLL through the untouched suffix.  The remaining helpers
are model-free so the scientific identities can be tested without loading a
checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Literal, TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import Gemma3CausalLMAdapter


LAYER4_CANONICAL_SITES = MappingProxyType(
    {
        "h": "layer.4.input",
        "q_A": "layer.4.attention.operator_output",
        "v_A": "layer.4.attention.delta",
        "h_A": "layer.4.post_attention",
        "X4": "layer.4.mlp.normalized_input",
        "q_M": "layer.4.mlp.operator_output",
        "v_M": "layer.4.mlp.delta",
        "h_plus": "layer.4.output",
    }
)
LAYER4_ATTENTION_CONTEXT_SITE = "layer.4.attention.context"
LAYER4_MLP_DOWN_INPUT_SITE = "layer.4.mlp.down_input"
ATTENTION_SUMMARY_FEATURE_NAMES = (
    "entropy",
    "top1_mass",
    "top4_mass",
    "relative_position_mean",
    "relative_position_std",
)
DEFAULT_MLP_PARTITION_SEED = 4_240_064


def _require_finite_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{label} cannot have an empty dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _cpu_clone(value: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    return value.detach().to(device="cpu", dtype=dtype).clone().contiguous()


def _validate_sequence_ids(
    sequence_ids: Sequence[str],
    *,
    batch_size: int,
) -> tuple[str, ...]:
    result = tuple(sequence_ids)
    if len(result) != batch_size:
        raise ValueError("one sequence id is required for every batch item")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError("sequence ids must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("sequence ids must be unique within one capture")
    return result


@dataclass(frozen=True, slots=True)
class TensorClosureDiagnostic:
    """Numerically safe diagnostics for one tensor identity."""

    name: str
    maximum_absolute_error: float
    relative_l2_error: float
    reference_maximum_absolute_value: float

    @classmethod
    def compare(
        cls,
        name: str,
        observed: Tensor,
        reference: Tensor,
    ) -> "TensorClosureDiagnostic":
        if observed.shape != reference.shape:
            raise ValueError(f"{name} tensors have different shapes")
        left = _require_finite_tensor(observed, label=f"{name} observed")
        right = _require_finite_tensor(reference, label=f"{name} reference")
        error = left.to(torch.float64) - right.to(torch.float64)
        maximum = float(error.abs().max().item())
        reference_maximum = float(right.to(torch.float64).abs().max().item())
        denominator = max(
            float(torch.linalg.vector_norm(right.to(torch.float64)).item()),
            torch.finfo(torch.float64).tiny,
        )
        relative = float(torch.linalg.vector_norm(error).item()) / denominator
        return cls(
            name=name,
            maximum_absolute_error=maximum,
            relative_l2_error=relative,
            reference_maximum_absolute_value=reference_maximum,
        )

    def passes(self, *, absolute_tolerance: float, relative_tolerance: float) -> bool:
        threshold = absolute_tolerance + (
            relative_tolerance * self.reference_maximum_absolute_value
        )
        return self.maximum_absolute_error <= threshold


@dataclass(frozen=True, slots=True)
class Layer4AdditivityDiagnostics:
    attention_stage_identity: TensorClosureDiagnostic
    mlp_stage_identity: TensorClosureDiagnostic
    complete_stage_identity: TensorClosureDiagnostic
    attention_q_additivity: TensorClosureDiagnostic
    attention_v_additivity: TensorClosureDiagnostic
    mlp_q_additivity: TensorClosureDiagnostic
    mlp_v_additivity: TensorClosureDiagnostic
    complete_component_additivity: TensorClosureDiagnostic

    @property
    def all(self) -> tuple[TensorClosureDiagnostic, ...]:
        return (
            self.attention_stage_identity,
            self.mlp_stage_identity,
            self.complete_stage_identity,
            self.attention_q_additivity,
            self.attention_v_additivity,
            self.mlp_q_additivity,
            self.mlp_v_additivity,
            self.complete_component_additivity,
        )

    def assert_within(
        self,
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> None:
        failures = tuple(
            item
            for item in self.all
            if not item.passes(
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )
        if failures:
            detail = ", ".join(
                f"{item.name}: max={item.maximum_absolute_error:.6g}, "
                f"rel_l2={item.relative_l2_error:.6g}"
                for item in failures
            )
            raise ValueError(f"layer-4 additive identities failed: {detail}")


@dataclass(frozen=True, slots=True)
class SharedGemmaRMSNormScale:
    """One realized diagonal Gemma RMSNorm scale computed from a total.

    ``scale`` is always fp32.  It includes both the input-dependent RMS
    reciprocal and Gemma's learned unit-offset factor ``1 + weight``.
    Components are multiplied in fp32 and promoted to fp64 for accumulation.
    """

    scale: Tensor
    learned_weight: Tensor
    epsilon: float
    source_dtype: str
    scale_parameterization: str = "unit_offset"
    compute_dtype: str = "float32"

    def __post_init__(self) -> None:
        scale = _require_finite_tensor(self.scale, label="realized RMS scale")
        weight = _require_finite_tensor(
            self.learned_weight,
            label="RMSNorm learned weight",
            ndim=1,
        )
        if scale.dtype != torch.float32:
            raise ValueError("realized Gemma RMSNorm scale must be fp32")
        if scale.shape[-1] != weight.shape[0]:
            raise ValueError("RMSNorm scale and learned weight widths differ")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not math.isfinite(float(self.epsilon))
            or float(self.epsilon) <= 0.0
        ):
            raise ValueError("RMSNorm epsilon must be finite and positive")
        if self.scale_parameterization != "unit_offset":
            raise ValueError("Gemma RMSNorm must use unit-offset scaling")
        if self.compute_dtype != "float32":
            raise ValueError("Gemma RMSNorm must compute in float32")
        object.__setattr__(self, "scale", _cpu_clone(scale, dtype=torch.float32))
        object.__setattr__(
            self,
            "learned_weight",
            _cpu_clone(weight, dtype=torch.float32),
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @classmethod
    def from_total(
        cls,
        total: Tensor,
        learned_weight: Tensor,
        *,
        epsilon: float,
    ) -> "SharedGemmaRMSNormScale":
        total = _require_finite_tensor(total, label="RMSNorm total")
        weight = _require_finite_tensor(
            learned_weight,
            label="RMSNorm learned weight",
            ndim=1,
        )
        if total.shape[-1] != weight.shape[0]:
            raise ValueError("RMSNorm total and learned weight widths differ")
        total_fp32 = total.detach().float()
        reciprocal_rms = torch.rsqrt(
            total_fp32.square().mean(dim=-1, keepdim=True) + float(epsilon)
        )
        effective_weight = 1.0 + weight.detach().to(
            device=total.device,
            dtype=torch.float32,
        )
        scale = reciprocal_rms * effective_weight
        return cls(
            scale=scale,
            learned_weight=weight,
            epsilon=float(epsilon),
            source_dtype=str(total.dtype),
        )

    def apply(self, components: Tensor) -> Tensor:
        """Apply this fixed scale; never recompute it from a component."""

        values = _require_finite_tensor(components, label="RMSNorm components")
        scale = self.scale.to(device=values.device)
        if values.shape[-1] != scale.shape[-1]:
            raise ValueError("RMSNorm component width is incompatible")
        if values.ndim == scale.ndim:
            expected_prefix = tuple(scale.shape[:-1])
        elif values.ndim == scale.ndim + 1:
            expected_prefix = tuple(scale.shape[:-1])
            scale = scale.unsqueeze(-2)
        else:
            raise ValueError(
                "RMSNorm components must match the total or add one component axis"
            )
        if tuple(values.shape[: len(expected_prefix)]) != expected_prefix:
            raise ValueError("RMSNorm component leading dimensions differ")
        return (values.float() * scale).to(torch.float64)


def gemma_rmsnorm(
    values: Tensor,
    learned_weight: Tensor,
    *,
    epsilon: float,
) -> Tensor:
    """Reproduce native Gemma RMSNorm, including its final dtype cast."""

    source_dtype = values.dtype
    realized = SharedGemmaRMSNormScale.from_total(
        values,
        learned_weight,
        epsilon=epsilon,
    )
    return realized.apply(values).to(device=values.device, dtype=source_dtype)


PartitionScheme = Literal["contiguous", "interleaved"]


@dataclass(frozen=True, slots=True)
class MLPProbePartitionManifest:
    """A deterministic, balanced and exhaustive MLP-channel partition."""

    intermediate_width: int
    block_count: int
    scheme: PartitionScheme
    seed: int
    channel_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.intermediate_width) is not int or self.intermediate_width <= 0:
            raise ValueError("intermediate width must be positive")
        if type(self.block_count) is not int or self.block_count <= 0:
            raise ValueError("block count must be positive")
        if self.block_count > self.intermediate_width:
            raise ValueError("block count cannot exceed intermediate width")
        if self.scheme not in ("contiguous", "interleaved"):
            raise ValueError("unknown MLP probe partition scheme")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("partition seed must be a nonnegative integer")
        if len(self.channel_indices) != self.block_count:
            raise ValueError("partition has the wrong number of blocks")
        flattened = tuple(index for block in self.channel_indices for index in block)
        if sorted(flattened) != list(range(self.intermediate_width)):
            raise ValueError("partition must contain every channel exactly once")
        sizes = tuple(len(block) for block in self.channel_indices)
        if min(sizes) <= 0 or max(sizes) - min(sizes) > 1:
            raise ValueError("partition blocks must be nonempty and balanced")
        if self.scheme == "contiguous":
            cursor = 0
            for block in self.channel_indices:
                if block != tuple(range(cursor, cursor + len(block))):
                    raise ValueError("contiguous partition contains a gap")
                cursor += len(block)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(len(block) for block in self.channel_indices)


def make_mlp_probe_partition(
    intermediate_width: int,
    *,
    scheme: PartitionScheme,
    block_count: int = 4,
    seed: int = DEFAULT_MLP_PARTITION_SEED,
) -> MLPProbePartitionManifest:
    if (
        type(intermediate_width) is not int
        or intermediate_width <= 0
        or type(block_count) is not int
        or block_count <= 0
        or block_count > intermediate_width
    ):
        raise ValueError("partition width and block count are invalid")
    base, remainder = divmod(intermediate_width, block_count)
    sizes = tuple(base + (index < remainder) for index in range(block_count))
    if scheme == "contiguous":
        blocks: list[tuple[int, ...]] = []
        cursor = 0
        for size in sizes:
            blocks.append(tuple(range(cursor, cursor + size)))
            cursor += size
    elif scheme == "interleaved":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(
            intermediate_width,
            generator=generator,
        ).tolist()
        # Round-robin the seeded channel order.  This is balanced for every
        # width and records the complete realized assignment in the manifest.
        blocks = [
            tuple(sorted(permutation[index::block_count]))
            for index in range(block_count)
        ]
    else:
        raise ValueError("unknown MLP probe partition scheme")
    return MLPProbePartitionManifest(
        intermediate_width=intermediate_width,
        block_count=block_count,
        scheme=scheme,
        seed=seed,
        channel_indices=tuple(blocks),
    )


def attention_head_q_contributions(
    attention_context: Tensor,
    output_projection_weight: Tensor,
    *,
    query_head_count: int,
    head_dimension: int,
) -> Tensor:
    """Project each query-head context through its matching ``o_proj`` slice."""

    context = _require_finite_tensor(
        attention_context,
        label="attention context",
        ndim=3,
    )
    weight = _require_finite_tensor(
        output_projection_weight,
        label="attention output-projection weight",
        ndim=2,
    )
    if type(query_head_count) is not int or query_head_count <= 0:
        raise ValueError("query head count must be positive")
    if type(head_dimension) is not int or head_dimension <= 0:
        raise ValueError("head dimension must be positive")
    context_width = query_head_count * head_dimension
    if context.shape[-1] != context_width or weight.shape[1] != context_width:
        raise ValueError("attention context and output projection do not align")
    contributions = []
    for head in range(query_head_count):
        start = head * head_dimension
        stop = start + head_dimension
        contributions.append(
            F.linear(context[..., start:stop], weight[:, start:stop])
        )
    return torch.stack(contributions, dim=-2)


def mlp_block_q_contributions(
    down_projection_input: Tensor,
    down_projection_weight: Tensor,
    partition: MLPProbePartitionManifest,
) -> Tensor:
    """Project deterministic MLP probe blocks through ``down_proj`` slices."""

    values = _require_finite_tensor(
        down_projection_input,
        label="MLP down-projection input",
        ndim=3,
    )
    weight = _require_finite_tensor(
        down_projection_weight,
        label="MLP down-projection weight",
        ndim=2,
    )
    if values.shape[-1] != partition.intermediate_width:
        raise ValueError("MLP input width differs from the partition")
    if weight.shape[1] != partition.intermediate_width:
        raise ValueError("MLP down projection differs from the partition")
    blocks = []
    for channel_indices in partition.channel_indices:
        indices = torch.tensor(
            channel_indices,
            device=values.device,
            dtype=torch.long,
        )
        blocks.append(
            F.linear(
                values.index_select(-1, indices),
                weight.index_select(1, indices),
            )
        )
    return torch.stack(blocks, dim=-2)


@dataclass(frozen=True, slots=True)
class CausalAttentionSummary:
    """Primary destination-free attention routing features."""

    features: Tensor
    query_valid_mask: Tensor
    feature_names: tuple[str, ...] = ATTENTION_SUMMARY_FEATURE_NAMES
    maximum_future_mass: float = 0.0
    maximum_invalid_key_mass: float = 0.0
    source_feature_contract: str = (
        "realized_attention_probabilities_masks_and_positions_only;"
        "no_destination_state_or_residual_contribution"
    )

    def __post_init__(self) -> None:
        features = _require_finite_tensor(
            self.features,
            label="causal attention summary",
            ndim=4,
        )
        if features.shape[-1] != len(ATTENTION_SUMMARY_FEATURE_NAMES):
            raise ValueError("attention summary has the wrong feature width")
        if self.feature_names != ATTENTION_SUMMARY_FEATURE_NAMES:
            raise ValueError("attention summary feature order is not canonical")
        if (
            not isinstance(self.query_valid_mask, Tensor)
            or self.query_valid_mask.dtype != torch.bool
            or self.query_valid_mask.ndim != 2
            or self.query_valid_mask.shape
            != (features.shape[0], features.shape[2])
        ):
            raise ValueError("attention summary query mask is invalid")
        if self.maximum_future_mass < 0 or self.maximum_invalid_key_mass < 0:
            raise ValueError("attention leakage diagnostics cannot be negative")
        object.__setattr__(self, "features", _cpu_clone(features, dtype=torch.float64))
        object.__setattr__(
            self,
            "query_valid_mask",
            self.query_valid_mask.detach().to(device="cpu").clone(),
        )


def causal_attention_summaries(
    attention_probabilities: Tensor,
    *,
    query_valid_mask: Tensor,
    key_valid_mask: Tensor,
    query_positions: Tensor,
    key_positions: Tensor,
    leakage_tolerance: float = 1.0e-7,
    normalization_tolerance: float = 1.0e-5,
) -> CausalAttentionSummary:
    """Build five causal-only routing features for every query/head row."""

    probabilities = _require_finite_tensor(
        attention_probabilities,
        label="attention probabilities",
        ndim=4,
    ).to(torch.float64)
    batch, _heads, queries, keys = probabilities.shape
    for label, value, shape in (
        ("query valid mask", query_valid_mask, (batch, queries)),
        ("key valid mask", key_valid_mask, (batch, keys)),
        ("query positions", query_positions, (batch, queries)),
        ("key positions", key_positions, (batch, keys)),
    ):
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{label} has the wrong shape")
    if query_valid_mask.dtype != torch.bool or key_valid_mask.dtype != torch.bool:
        raise ValueError("attention validity masks must be boolean")
    if query_positions.dtype not in (torch.int32, torch.int64) or key_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("attention positions must use integer dtypes")
    if (
        isinstance(leakage_tolerance, bool)
        or not math.isfinite(float(leakage_tolerance))
        or float(leakage_tolerance) < 0.0
    ):
        raise ValueError("leakage tolerance must be finite and nonnegative")
    if bool((probabilities < -float(leakage_tolerance)).any()):
        raise ValueError("attention probabilities contain negative mass")

    query_ordinals = torch.arange(queries, device=probabilities.device).view(
        1, 1, queries, 1
    )
    key_ordinals = torch.arange(keys, device=probabilities.device).view(
        1, 1, 1, keys
    )
    future = key_ordinals > query_ordinals
    invalid_keys = ~key_valid_mask.to(probabilities.device)[:, None, None, :]
    valid_queries = query_valid_mask.to(probabilities.device)[:, None, :, None]
    future_mass = (probabilities * future * valid_queries).sum(dim=-1)
    invalid_mass = (probabilities * invalid_keys * valid_queries).sum(dim=-1)
    maximum_future = float(future_mass.abs().max().item())
    maximum_invalid = float(invalid_mass.abs().max().item())
    if maximum_future > float(leakage_tolerance):
        raise ValueError(
            "attention probabilities leak future-token mass "
            f"({maximum_future:.6g})"
        )
    if maximum_invalid > float(leakage_tolerance):
        raise ValueError(
            "attention probabilities assign mass to invalid keys "
            f"({maximum_invalid:.6g})"
        )

    allowed = (~future) & (~invalid_keys) & valid_queries
    causal = probabilities.clamp_min(0).masked_fill(~allowed, 0)
    mass = causal.sum(dim=-1, keepdim=True)
    valid_rows = query_valid_mask.to(probabilities.device)[:, None, :, None]
    if bool((mass.masked_select(valid_rows) <= 0).any()):
        raise ValueError("a valid attention row has no causal mass")
    maximum_mass_error = float(
        (mass.masked_select(valid_rows) - 1.0).abs().max().item()
    )
    if maximum_mass_error > float(normalization_tolerance):
        raise ValueError(
            "valid attention rows are not normalized "
            f"({maximum_mass_error:.6g})"
        )
    normalized = torch.where(mass > 0, causal / mass.clamp_min(1.0e-300), causal)
    log_probabilities = torch.where(
        normalized > 0,
        normalized.log(),
        torch.zeros_like(normalized),
    )
    entropy = -(normalized * log_probabilities).sum(dim=-1)
    top1 = normalized.max(dim=-1).values
    top4 = normalized.topk(k=min(4, keys), dim=-1).values.sum(dim=-1)
    relative_positions = (
        query_positions.to(probabilities.device, dtype=torch.float64)[:, None, :, None]
        - key_positions.to(probabilities.device, dtype=torch.float64)[:, None, None, :]
    )
    relative_mean = (normalized * relative_positions).sum(dim=-1)
    relative_variance = (
        normalized * (relative_positions - relative_mean.unsqueeze(-1)).square()
    ).sum(dim=-1)
    relative_std = relative_variance.clamp_min(0).sqrt()
    features = torch.stack(
        (entropy, top1, top4, relative_mean, relative_std),
        dim=-1,
    )
    features = features.masked_fill(
        ~query_valid_mask.to(features.device)[:, None, :, None],
        0,
    )
    return CausalAttentionSummary(
        features=features,
        query_valid_mask=query_valid_mask,
        maximum_future_mass=maximum_future,
        maximum_invalid_key_mass=maximum_invalid,
    )


@dataclass(frozen=True, slots=True)
class FisherVJPRowID:
    sequence_id: str
    query_position: int
    target_position: int


@dataclass(frozen=True, slots=True)
class Layer4OutputFisherVJP:
    """Summed causal-NLL VJP at a detached ``layer.4.output`` leaf."""

    sequence_ids: tuple[str, ...]
    logical_positions: Tensor
    valid_row_mask: Tensor
    gradients: Tensor
    row_ids: tuple[FisherVJPRowID, ...]
    causal_nll_sum: float
    supervised_token_count: int
    boundary_site: str = LAYER4_CANONICAL_SITES["h_plus"]

    def __post_init__(self) -> None:
        gradients = _require_finite_tensor(
            self.gradients,
            label="layer-4 output Fisher VJP",
            ndim=3,
        )
        batch, length, _width = gradients.shape
        sequence_ids = _validate_sequence_ids(self.sequence_ids, batch_size=batch)
        if (
            not isinstance(self.valid_row_mask, Tensor)
            or self.valid_row_mask.dtype != torch.bool
            or tuple(self.valid_row_mask.shape) != (batch, length)
        ):
            raise ValueError("Fisher VJP valid-row mask is invalid")
        if (
            not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or tuple(self.logical_positions.shape) != (batch, length)
        ):
            raise ValueError("Fisher VJP logical positions are invalid")
        if type(self.supervised_token_count) is not int or self.supervised_token_count <= 0:
            raise ValueError("Fisher VJP must contain supervised tokens")
        if self.supervised_token_count != int(self.valid_row_mask.sum().item()):
            raise ValueError("Fisher VJP row count differs from its mask")
        if len(self.row_ids) != self.supervised_token_count:
            raise ValueError("Fisher VJP row identities are incomplete")
        if not math.isfinite(float(self.causal_nll_sum)):
            raise ValueError("causal NLL must be finite")
        invalid = ~self.valid_row_mask.unsqueeze(-1)
        if bool((gradients.masked_select(invalid).abs() > 0).any()):
            raise ValueError("invalid Fisher VJP rows must be zero-inert")
        object.__setattr__(self, "sequence_ids", sequence_ids)
        object.__setattr__(
            self,
            "logical_positions",
            self.logical_positions.detach().to(device="cpu", dtype=torch.int64).clone(),
        )
        object.__setattr__(
            self,
            "valid_row_mask",
            self.valid_row_mask.detach().to(device="cpu").clone(),
        )
        object.__setattr__(self, "gradients", _cpu_clone(gradients, dtype=torch.float64))
        object.__setattr__(self, "causal_nll_sum", float(self.causal_nll_sum))

    @property
    def valid_gradients(self) -> Tensor:
        return self.gradients[self.valid_row_mask]


def detach_layer4_output_as_leaf(output: Tensor) -> Tensor:
    """Create the exact suffix-VJP boundary without retaining its prefix graph."""

    output = _require_finite_tensor(output, label="layer.4.output", ndim=3)
    return output.detach().clone().requires_grad_(True)


def causal_nll_output_fisher_vjp(
    logits: Tensor,
    output_leaf: Tensor,
    *,
    token_ids: Tensor,
    query_valid_mask: Tensor,
    logical_positions: Tensor,
    sequence_ids: Sequence[str],
    ignore_index: int = -100,
) -> Layer4OutputFisherVJP:
    """Differentiate summed next-token NLL with respect to the detached leaf."""

    logits = _require_finite_tensor(logits, label="causal logits", ndim=3)
    leaf = _require_finite_tensor(output_leaf, label="layer.4.output leaf", ndim=3)
    if not leaf.is_leaf or not leaf.requires_grad:
        raise ValueError("layer.4.output boundary must be a detached gradient leaf")
    batch, length, _width = leaf.shape
    if tuple(logits.shape[:2]) != (batch, length):
        raise ValueError("causal logits and layer-4 output do not align")
    if (
        not isinstance(token_ids, Tensor)
        or token_ids.dtype not in (torch.int32, torch.int64)
        or tuple(token_ids.shape) != (batch, length)
    ):
        raise ValueError("causal token ids are invalid")
    if (
        not isinstance(query_valid_mask, Tensor)
        or query_valid_mask.dtype != torch.bool
        or tuple(query_valid_mask.shape) != (batch, length)
    ):
        raise ValueError("causal query-valid mask is invalid")
    if (
        not isinstance(logical_positions, Tensor)
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or tuple(logical_positions.shape) != (batch, length)
    ):
        raise ValueError("causal logical positions are invalid")
    sequence_ids = _validate_sequence_ids(sequence_ids, batch_size=batch)
    if length < 2:
        raise ValueError("causal NLL VJP requires at least two positions")

    labels = token_ids[:, 1:].to(device=logits.device, dtype=torch.long)
    valid_pairs = (
        query_valid_mask[:, :-1]
        & query_valid_mask[:, 1:]
        & token_ids[:, 1:].ne(ignore_index)
    ).to(device=logits.device)
    if not bool(valid_pairs.any()):
        raise ValueError("causal NLL VJP has no valid next-token rows")
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(batch, length - 1)
    loss = losses.masked_select(valid_pairs).sum()
    gradient = torch.autograd.grad(
        loss,
        leaf,
        retain_graph=False,
        create_graph=False,
    )[0]
    valid_rows = torch.zeros(
        batch,
        length,
        dtype=torch.bool,
        device=leaf.device,
    )
    valid_rows[:, :-1] = valid_pairs.to(device=leaf.device)
    gradient = gradient.masked_fill(~valid_rows.unsqueeze(-1), 0)
    row_ids = []
    positions_cpu = logical_positions.detach().to(device="cpu")
    valid_cpu = valid_rows.detach().to(device="cpu")
    for batch_index, sequence_id in enumerate(sequence_ids):
        for position in torch.nonzero(valid_cpu[batch_index], as_tuple=False).flatten():
            index = int(position.item())
            row_ids.append(
                FisherVJPRowID(
                    sequence_id=sequence_id,
                    query_position=int(positions_cpu[batch_index, index].item()),
                    target_position=int(positions_cpu[batch_index, index + 1].item()),
                )
            )
    return Layer4OutputFisherVJP(
        sequence_ids=sequence_ids,
        logical_positions=logical_positions,
        valid_row_mask=valid_rows,
        gradients=gradient,
        row_ids=tuple(row_ids),
        causal_nll_sum=float(loss.detach().item()),
        supervised_token_count=int(valid_rows.sum().item()),
    )


@dataclass(frozen=True, slots=True)
class Layer4SequenceCapture:
    """Validated exact stage/component capture for one sequence batch."""

    sequence_ids: tuple[str, ...]
    query_valid_mask: Tensor
    logical_positions: Tensor
    h: Tensor
    q_A: Tensor
    v_A: Tensor
    h_A: Tensor
    X4: Tensor
    q_M: Tensor
    v_M: Tensor
    h_plus: Tensor
    attention_head_q: Tensor
    attention_head_v: Tensor
    mlp_block_q: Tensor
    mlp_block_v: Tensor
    attention_probabilities: Tensor
    attention_summary: CausalAttentionSummary
    attention_scale: SharedGemmaRMSNormScale
    mlp_scale: SharedGemmaRMSNormScale
    mlp_partition: MLPProbePartitionManifest
    output_fisher_vjp: Layer4OutputFisherVJP
    runtime_dtypes: tuple[tuple[str, str], ...]
    absolute_tolerance: float = 1.0e-5
    relative_tolerance: float = 1.0e-4
    diagnostics: Layer4AdditivityDiagnostics = field(init=False)

    def __post_init__(self) -> None:
        h = _require_finite_tensor(self.h, label="h", ndim=3)
        batch, length, width = h.shape
        sequence_ids = _validate_sequence_ids(self.sequence_ids, batch_size=batch)
        stage_values: dict[str, Tensor] = {"h": h}
        for label in ("q_A", "v_A", "h_A", "X4", "q_M", "v_M", "h_plus"):
            value = _require_finite_tensor(getattr(self, label), label=label, ndim=3)
            if tuple(value.shape) != (batch, length, width):
                raise ValueError(f"{label} does not align with h")
            stage_values[label] = value
        if (
            not isinstance(self.query_valid_mask, Tensor)
            or self.query_valid_mask.dtype != torch.bool
            or tuple(self.query_valid_mask.shape) != (batch, length)
        ):
            raise ValueError("capture query-valid mask is invalid")
        if (
            not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or tuple(self.logical_positions.shape) != (batch, length)
        ):
            raise ValueError("capture logical positions are invalid")
        head_q = _require_finite_tensor(
            self.attention_head_q,
            label="attention head q contributions",
            ndim=4,
        )
        head_v = _require_finite_tensor(
            self.attention_head_v,
            label="attention head v contributions",
            ndim=4,
        )
        block_q = _require_finite_tensor(
            self.mlp_block_q,
            label="MLP block q contributions",
            ndim=4,
        )
        block_v = _require_finite_tensor(
            self.mlp_block_v,
            label="MLP block v contributions",
            ndim=4,
        )
        if head_q.shape != head_v.shape or tuple(head_q.shape[:2] + head_q.shape[3:]) != (
            batch,
            length,
            width,
        ):
            raise ValueError("attention component tensors are misaligned")
        if block_q.shape != block_v.shape or tuple(block_q.shape[:2] + block_q.shape[3:]) != (
            batch,
            length,
            width,
        ):
            raise ValueError("MLP component tensors are misaligned")
        if head_q.shape[2] != 4:
            raise ValueError("the Gemma-3 270M layer-4 capture requires four heads")
        if block_q.shape[2] != self.mlp_partition.block_count:
            raise ValueError("MLP components differ from their partition")
        probabilities = _require_finite_tensor(
            self.attention_probabilities,
            label="attention probabilities",
            ndim=4,
        )
        if tuple(probabilities.shape[:3]) != (batch, 4, length):
            raise ValueError("attention probabilities are misaligned")
        if self.attention_summary.features.shape[:3] != (batch, 4, length):
            raise ValueError("attention summary is misaligned")
        if self.output_fisher_vjp.sequence_ids != sequence_ids:
            raise ValueError("Fisher VJP sequence ids differ from the capture")
        if not torch.equal(
            self.output_fisher_vjp.logical_positions,
            self.logical_positions.detach().to(device="cpu", dtype=torch.int64),
        ):
            raise ValueError("Fisher VJP positions differ from the capture")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(dtype, str)
            or not dtype
            for name, dtype in self.runtime_dtypes
        ):
            raise ValueError("runtime dtype receipt is invalid")
        if len({name for name, _dtype in self.runtime_dtypes}) != len(self.runtime_dtypes):
            raise ValueError("runtime dtype receipt contains duplicate names")
        for label, tolerance in (
            ("absolute", self.absolute_tolerance),
            ("relative", self.relative_tolerance),
        ):
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or float(tolerance) < 0
            ):
                raise ValueError(f"{label} tolerance must be finite and nonnegative")

        diagnostics = Layer4AdditivityDiagnostics(
            attention_stage_identity=TensorClosureDiagnostic.compare(
                "h_A == h + v_A", stage_values["h_A"], h + stage_values["v_A"]
            ),
            mlp_stage_identity=TensorClosureDiagnostic.compare(
                "h_plus == h_A + v_M",
                stage_values["h_plus"],
                stage_values["h_A"] + stage_values["v_M"],
            ),
            complete_stage_identity=TensorClosureDiagnostic.compare(
                "h_plus == h + v_A + v_M",
                stage_values["h_plus"],
                h + stage_values["v_A"] + stage_values["v_M"],
            ),
            attention_q_additivity=TensorClosureDiagnostic.compare(
                "q_A == sum_j q_A_j",
                stage_values["q_A"],
                head_q.to(torch.float64).sum(dim=2),
            ),
            attention_v_additivity=TensorClosureDiagnostic.compare(
                "v_A == sum_j D_A(q_A) q_A_j",
                stage_values["v_A"],
                head_v.to(torch.float64).sum(dim=2),
            ),
            mlp_q_additivity=TensorClosureDiagnostic.compare(
                "q_M == sum_r q_M_r",
                stage_values["q_M"],
                block_q.to(torch.float64).sum(dim=2),
            ),
            mlp_v_additivity=TensorClosureDiagnostic.compare(
                "v_M == sum_r D_M(q_M) q_M_r",
                stage_values["v_M"],
                block_v.to(torch.float64).sum(dim=2),
            ),
            complete_component_additivity=TensorClosureDiagnostic.compare(
                "h_plus - h == sum_j v_A_j + sum_r v_M_r",
                stage_values["h_plus"].to(torch.float64) - h.to(torch.float64),
                head_v.to(torch.float64).sum(dim=2)
                + block_v.to(torch.float64).sum(dim=2),
            ),
        )
        diagnostics.assert_within(
            absolute_tolerance=float(self.absolute_tolerance),
            relative_tolerance=float(self.relative_tolerance),
        )
        object.__setattr__(self, "sequence_ids", sequence_ids)
        object.__setattr__(self, "query_valid_mask", self.query_valid_mask.detach().cpu().clone())
        object.__setattr__(
            self,
            "logical_positions",
            self.logical_positions.detach().to(device="cpu", dtype=torch.int64).clone(),
        )
        for label, value in stage_values.items():
            object.__setattr__(self, label, _cpu_clone(value))
        for label, value in (
            ("attention_head_q", head_q),
            ("attention_head_v", head_v),
            ("mlp_block_q", block_q),
            ("mlp_block_v", block_v),
            ("attention_probabilities", probabilities),
        ):
            object.__setattr__(self, label, _cpu_clone(value))
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def valid_stage_rows(self) -> Mapping[str, Tensor]:
        mask = self.query_valid_mask
        return MappingProxyType(
            {
                name: getattr(self, name)[mask]
                for name in ("h", "q_A", "v_A", "h_A", "X4", "q_M", "v_M", "h_plus")
            }
        )

    @property
    def v_total(self) -> Tensor:
        return self.h_plus.to(torch.float64) - self.h.to(torch.float64)

    @property
    def attention_component_sum(self) -> Tensor:
        return self.attention_head_v.to(torch.float64).sum(dim=2)

    @property
    def mlp_component_sum(self) -> Tensor:
        return self.mlp_block_v.to(torch.float64).sum(dim=2)


def build_layer4_sequence_capture(
    *,
    sequence_ids: Sequence[str],
    query_valid_mask: Tensor,
    key_valid_mask: Tensor,
    logical_positions: Tensor,
    key_positions: Tensor,
    h: Tensor,
    q_A: Tensor,
    v_A: Tensor,
    h_A: Tensor,
    X4: Tensor,
    q_M: Tensor,
    v_M: Tensor,
    h_plus: Tensor,
    attention_context: Tensor,
    attention_probabilities: Tensor,
    attention_output_projection_weight: Tensor,
    attention_norm_weight: Tensor,
    attention_norm_epsilon: float,
    query_head_count: int,
    head_dimension: int,
    mlp_down_projection_input: Tensor,
    mlp_down_projection_weight: Tensor,
    mlp_norm_weight: Tensor,
    mlp_norm_epsilon: float,
    mlp_partition: MLPProbePartitionManifest,
    output_fisher_vjp: Layer4OutputFisherVJP,
    runtime_dtypes: Sequence[tuple[str, str]] = (),
    absolute_tolerance: float = 1.0e-5,
    relative_tolerance: float = 1.0e-4,
) -> Layer4SequenceCapture:
    """Build and authenticate a capture from model-free native tensors."""

    if query_head_count != 4:
        raise ValueError("the pinned Gemma-3 270M capture requires four query heads")
    head_q = attention_head_q_contributions(
        attention_context,
        attention_output_projection_weight,
        query_head_count=query_head_count,
        head_dimension=head_dimension,
    )
    attention_scale = SharedGemmaRMSNormScale.from_total(
        q_A,
        attention_norm_weight,
        epsilon=attention_norm_epsilon,
    )
    head_v = attention_scale.apply(head_q)
    block_q = mlp_block_q_contributions(
        mlp_down_projection_input,
        mlp_down_projection_weight,
        mlp_partition,
    )
    mlp_scale = SharedGemmaRMSNormScale.from_total(
        q_M,
        mlp_norm_weight,
        epsilon=mlp_norm_epsilon,
    )
    block_v = mlp_scale.apply(block_q)
    summary = causal_attention_summaries(
        attention_probabilities,
        query_valid_mask=query_valid_mask,
        key_valid_mask=key_valid_mask,
        query_positions=logical_positions,
        key_positions=key_positions,
    )
    return Layer4SequenceCapture(
        sequence_ids=tuple(sequence_ids),
        query_valid_mask=query_valid_mask,
        logical_positions=logical_positions,
        h=h,
        q_A=q_A,
        v_A=v_A,
        h_A=h_A,
        X4=X4,
        q_M=q_M,
        v_M=v_M,
        h_plus=h_plus,
        attention_head_q=head_q,
        attention_head_v=head_v,
        mlp_block_q=block_q,
        mlp_block_v=block_v,
        attention_probabilities=attention_probabilities,
        attention_summary=summary,
        attention_scale=attention_scale,
        mlp_scale=mlp_scale,
        mlp_partition=mlp_partition,
        output_fisher_vjp=output_fisher_vjp,
        runtime_dtypes=tuple(runtime_dtypes),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


T = TypeVar("T")


def run_with_eager_attention_probability_capture(
    attention_module: nn.Module,
    forward: Callable[[], T],
) -> tuple[T, Tensor]:
    """Observe one eager-attention probability tensor and always remove hook."""

    if not isinstance(attention_module, nn.Module):
        raise TypeError("attention module must be a torch module")
    config = getattr(attention_module, "config", None)
    implementation = getattr(config, "_attn_implementation", "eager")
    if implementation not in (None, "eager"):
        raise ValueError("attention probability capture requires eager attention")
    captured: list[Tensor] = []

    def observe(_module: nn.Module, _args: tuple[Any, ...], output: object) -> None:
        if (
            not isinstance(output, (tuple, list))
            or len(output) < 2
            or not isinstance(output[1], Tensor)
        ):
            raise RuntimeError("eager Gemma attention did not return probabilities")
        captured.append(output[1])

    handle = attention_module.register_forward_hook(observe)
    try:
        result = forward()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "expected exactly one layer-4 eager attention invocation, "
            f"observed {len(captured)}"
        )
    return result, captured[0]


def _norm_epsilon(module: nn.Module, *, label: str) -> float:
    value = getattr(module, "eps", getattr(module, "variance_epsilon", None))
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} does not expose a valid epsilon")
    return float(value)


def capture_layer4_sequence(
    adapter: Gemma3CausalLMAdapter,
    model_inputs: Mapping[str, Tensor],
    *,
    sequence_ids: Sequence[str],
    mlp_partition_scheme: PartitionScheme = "contiguous",
    mlp_partition_seed: int = DEFAULT_MLP_PARTITION_SEED,
    absolute_tolerance: float = 1.0e-5,
    relative_tolerance: float = 1.0e-4,
) -> Layer4SequenceCapture:
    """Capture the native layer-4 transition and one fit-output suffix VJP."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    token_ids = model_inputs.get("labels", model_inputs.get("input_ids"))
    if not isinstance(token_ids, Tensor):
        raise ValueError("model inputs require input_ids or labels for causal NLL")
    layer_spec = adapter.layer("layer.4")
    attention_spec = layer_spec.attention
    transformer_spec = layer_spec.transformer
    if attention_spec is None or transformer_spec is None:
        raise RuntimeError("layer.4 lacks structured Gemma semantics")
    if attention_spec.query_heads != 4:
        raise RuntimeError("pinned Gemma-3 270M layer.4 must have four query heads")
    if adapter.module.training:
        raise RuntimeError("layer-4 capture requires model.eval()")
    attention_norm_spec = transformer_spec.attention_output_norm
    mlp_norm_spec = transformer_spec.feed_forward_output_norm
    for label, norm_spec in (
        ("attention", attention_norm_spec),
        ("MLP", mlp_norm_spec),
    ):
        if (
            norm_spec.kind != "rms_norm"
            or not norm_spec.affine
            or norm_spec.scale_parameterization != "unit_offset"
            or norm_spec.compute_dtype != "float32"
        ):
            raise RuntimeError(
                f"layer-4 {label} postnorm is not decomposable Gemma RMSNorm"
            )
    layer = adapter.source_module("layer.4")
    self_attention = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    attention_norm = getattr(layer, "post_attention_layernorm", None)
    mlp_norm = getattr(layer, "post_feedforward_layernorm", None)
    if not all(
        isinstance(value, nn.Module)
        for value in (self_attention, mlp, attention_norm, mlp_norm)
    ):
        raise TypeError("native layer.4 modules are incomplete")
    assert isinstance(self_attention, nn.Module)
    assert isinstance(mlp, nn.Module)
    assert isinstance(attention_norm, nn.Module)
    assert isinstance(mlp_norm, nn.Module)
    o_proj = getattr(self_attention, "o_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    if not isinstance(o_proj, nn.Linear) or not isinstance(down_proj, nn.Linear):
        raise TypeError("native layer.4 projections must be linear modules")
    if o_proj.bias is not None or down_proj.bias is not None:
        raise RuntimeError("exact component decomposition requires bias-free projections")
    attention_weight = getattr(attention_norm, "weight", None)
    mlp_weight = getattr(mlp_norm, "weight", None)
    if not isinstance(attention_weight, Tensor) or not isinstance(mlp_weight, Tensor):
        raise TypeError("native Gemma RMSNorm modules must expose learned weights")
    attention_epsilon = _norm_epsilon(
        attention_norm,
        label="attention postnorm",
    )
    mlp_epsilon = _norm_epsilon(mlp_norm, label="MLP postnorm")
    if not math.isclose(
        attention_epsilon,
        float(attention_norm_spec.epsilon),
        rel_tol=0.0,
        abs_tol=0.0,
    ) or not math.isclose(
        mlp_epsilon,
        float(mlp_norm_spec.epsilon),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("native and adapter layer-4 RMSNorm epsilon differ")

    capture_sites = (
        *LAYER4_CANONICAL_SITES.values(),
        LAYER4_ATTENTION_CONTEXT_SITE,
        LAYER4_MLP_DOWN_INPUT_SITE,
    )
    output_leaves: list[Tensor] = []

    def detach_output(output: Tensor) -> Tensor:
        if output_leaves:
            raise RuntimeError("layer.4.output intervention ran more than once")
        leaf = detach_layer4_output_as_leaf(output)
        output_leaves.append(leaf)
        return leaf

    def teacher_forward() -> Any:
        with torch.enable_grad():
            return adapter.forward(
                model_inputs,
                capture_sites=capture_sites,
                interventions={LAYER4_CANONICAL_SITES["h_plus"]: detach_output},
                retain_gradients=False,
            )

    run, probabilities = run_with_eager_attention_probability_capture(
        self_attention,
        teacher_forward,
    )
    if len(output_leaves) != 1:
        raise RuntimeError("layer.4.output detached leaf was not installed")
    leaf = output_leaves[0]
    fisher_vjp = causal_nll_output_fisher_vjp(
        run.logits,
        leaf,
        token_ids=token_ids,
        query_valid_mask=run.sequence.query_valid_mask,
        logical_positions=run.sequence.logical_positions,
        sequence_ids=sequence_ids,
    )
    partition = make_mlp_probe_partition(
        transformer_spec.feed_forward.intermediate_width,
        scheme=mlp_partition_scheme,
        block_count=4,
        seed=mlp_partition_seed,
    )
    activations = run.activations
    runtime_dtypes = tuple(
        (name, str(activations[site].dtype))
        for name, site in LAYER4_CANONICAL_SITES.items()
    ) + (
        ("attention_norm_weight", str(attention_weight.dtype)),
        ("mlp_norm_weight", str(mlp_weight.dtype)),
        ("attention_probabilities", str(probabilities.dtype)),
    )
    return build_layer4_sequence_capture(
        sequence_ids=sequence_ids,
        query_valid_mask=run.sequence.query_valid_mask,
        key_valid_mask=run.sequence.key_valid_mask,
        logical_positions=run.sequence.logical_positions,
        key_positions=run.sequence.key_logical_positions,
        h=activations[LAYER4_CANONICAL_SITES["h"]],
        q_A=activations[LAYER4_CANONICAL_SITES["q_A"]],
        v_A=activations[LAYER4_CANONICAL_SITES["v_A"]],
        h_A=activations[LAYER4_CANONICAL_SITES["h_A"]],
        X4=activations[LAYER4_CANONICAL_SITES["X4"]],
        q_M=activations[LAYER4_CANONICAL_SITES["q_M"]],
        v_M=activations[LAYER4_CANONICAL_SITES["v_M"]],
        h_plus=activations[LAYER4_CANONICAL_SITES["h_plus"]],
        attention_context=activations[LAYER4_ATTENTION_CONTEXT_SITE],
        attention_probabilities=probabilities,
        attention_output_projection_weight=o_proj.weight,
        attention_norm_weight=attention_weight,
        attention_norm_epsilon=attention_epsilon,
        query_head_count=attention_spec.query_heads,
        head_dimension=attention_spec.head_dimension,
        mlp_down_projection_input=activations[LAYER4_MLP_DOWN_INPUT_SITE],
        mlp_down_projection_weight=down_proj.weight,
        mlp_norm_weight=mlp_weight,
        mlp_norm_epsilon=mlp_epsilon,
        mlp_partition=partition,
        output_fisher_vjp=fisher_vjp,
        runtime_dtypes=runtime_dtypes,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


__all__ = [
    "ATTENTION_SUMMARY_FEATURE_NAMES",
    "CausalAttentionSummary",
    "DEFAULT_MLP_PARTITION_SEED",
    "FisherVJPRowID",
    "LAYER4_ATTENTION_CONTEXT_SITE",
    "LAYER4_CANONICAL_SITES",
    "LAYER4_MLP_DOWN_INPUT_SITE",
    "Layer4AdditivityDiagnostics",
    "Layer4OutputFisherVJP",
    "Layer4SequenceCapture",
    "MLPProbePartitionManifest",
    "SharedGemmaRMSNormScale",
    "TensorClosureDiagnostic",
    "attention_head_q_contributions",
    "build_layer4_sequence_capture",
    "capture_layer4_sequence",
    "causal_attention_summaries",
    "causal_nll_output_fisher_vjp",
    "detach_layer4_output_as_leaf",
    "gemma_rmsnorm",
    "make_mlp_probe_partition",
    "mlp_block_q_contributions",
    "run_with_eager_attention_probability_capture",
]
