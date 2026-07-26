"""Source-free execution of one model-described transformer layer.

The existing full-width replacement is intentionally architecture agnostic.
This executor is the complementary feasibility control: it consumes portable
``LayerSpec`` semantics and reproduces the source model's operator grammar
without retaining a source module or source fallback.

The first supported grammar is the Gemma 3 text-decoder block:

```
attention_state = x + post_attention_norm(
    grouped_query_rope_attention(input_norm(x))
)
output = attention_state + post_feed_forward_norm(
    gated_mlp(pre_feed_forward_norm(attention_state))
)
```

This module does not perform modal or width reduction.  A native-shape run is
therefore a generator-fidelity rung, not a compression result.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import ActivationTrace, record
from .adapters.base import (
    AttentionSpec,
    FeedForwardSpec,
    LayerSpec,
    NormalizationSpec,
    ResidualStageSpec,
    RopeSpec,
    SegmentRun,
    SequenceContext,
    StructuredOperatorSites,
    TransformerLayerSemantics,
    module_state_fingerprint,
)
from .compiler.manifest import CompiledSegment


_ARTIFACT_KIND = "fisher_graph.structured_transformer_layer_executor"
_FORMAT_VERSION = 1
_FINGERPRINT_DOMAIN = (
    b"fisher_graph.structured_transformer_layer_executor.v1\0"
)
_ARTIFACT_FIELDS = {
    "artifact_kind",
    "format_version",
    "contains_source_model_weights",
    "contains_source_fallback",
    "config",
    "dtype",
    "model_state_dict",
    "execution_fingerprint",
}
_CONFIG_FIELDS = {
    "attention",
    "transformer",
    "causal_edges_enabled",
}
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"unsupported structured executor dtype: {dtype}")
    return name


def _exact_mapping(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _normalization_from_dict(value: object) -> NormalizationSpec:
    raw = _exact_mapping(
        value,
        {
            "kind",
            "width",
            "epsilon",
            "affine",
            "scale_parameterization",
            "compute_dtype",
        },
        label="normalization",
    )
    return NormalizationSpec(**dict(raw))  # type: ignore[arg-type]


def _feed_forward_from_dict(value: object) -> FeedForwardSpec:
    raw = _exact_mapping(
        value,
        {
            "kind",
            "intermediate_width",
            "activation",
            "projection_bias",
        },
        label="feed-forward",
    )
    return FeedForwardSpec(**dict(raw))  # type: ignore[arg-type]


def _stage_from_dict(value: object) -> ResidualStageSpec:
    raw = _exact_mapping(
        value,
        {
            "id",
            "kind",
            "input_site",
            "normalized_input_site",
            "operator_output_site",
            "delta_site",
            "output_site",
        },
        label="residual stage",
    )
    return ResidualStageSpec(**dict(raw))  # type: ignore[arg-type]


def _operator_sites_from_dict(
    value: object,
) -> StructuredOperatorSites | None:
    if value is None:
        return None
    raw = _exact_mapping(
        value,
        {
            "attention_query_projection",
            "attention_query_normalized",
            "attention_key_projection",
            "attention_key_normalized",
            "attention_value_projection",
            "attention_context",
            "feed_forward_gate_projection",
            "feed_forward_up_projection",
            "feed_forward_down_input",
        },
        label="structured operator sites",
    )
    return StructuredOperatorSites(**dict(raw))  # type: ignore[arg-type]


def _rope_from_dict(value: object) -> RopeSpec | None:
    if value is None:
        return None
    raw = _exact_mapping(
        value,
        {
            "kind",
            "theta",
            "rotary_dimension",
            "scaling_kind",
            "scaling_factor",
        },
        label="RoPE",
    )
    return RopeSpec(**dict(raw))  # type: ignore[arg-type]


def _attention_from_dict(value: object) -> AttentionSpec:
    raw = _exact_mapping(
        value,
        {
            "kind",
            "query_heads",
            "key_value_heads",
            "head_dimension",
            "query_scale",
            "qk_norm",
            "window_size",
            "rope",
            "cache_kind",
        },
        label="attention",
    )
    values = dict(raw)
    values["rope"] = _rope_from_dict(values["rope"])
    return AttentionSpec(**values)  # type: ignore[arg-type]


def _transformer_from_dict(value: object) -> TransformerLayerSemantics:
    legacy_fields = {
        "residual_layout",
        "attention_input_norm",
        "attention_output_norm",
        "qk_norm",
        "attention_projection_bias",
        "attention_dropout",
        "attention_logit_softcap",
        "feed_forward_input_norm",
        "feed_forward_output_norm",
        "feed_forward",
        "stages",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) not in (legacy_fields, legacy_fields | {"operator_sites"})
    ):
        raise ValueError("transformer semantics fields are invalid")
    raw = value
    stages = raw["stages"]
    if not isinstance(stages, (tuple, list)):
        raise ValueError("transformer stages must be an array")
    return TransformerLayerSemantics(
        residual_layout=raw["residual_layout"],  # type: ignore[arg-type]
        attention_input_norm=_normalization_from_dict(
            raw["attention_input_norm"]
        ),
        attention_output_norm=_normalization_from_dict(
            raw["attention_output_norm"]
        ),
        qk_norm=(
            None
            if raw["qk_norm"] is None
            else _normalization_from_dict(raw["qk_norm"])
        ),
        attention_projection_bias=raw[  # type: ignore[arg-type]
            "attention_projection_bias"
        ],
        attention_dropout=raw["attention_dropout"],  # type: ignore[arg-type]
        attention_logit_softcap=raw[  # type: ignore[arg-type]
            "attention_logit_softcap"
        ],
        feed_forward_input_norm=_normalization_from_dict(
            raw["feed_forward_input_norm"]
        ),
        feed_forward_output_norm=_normalization_from_dict(
            raw["feed_forward_output_norm"]
        ),
        feed_forward=_feed_forward_from_dict(raw["feed_forward"]),
        stages=tuple(_stage_from_dict(stage) for stage in stages),
        operator_sites=_operator_sites_from_dict(
            raw.get("operator_sites")
        ),
    )


@dataclass(frozen=True, slots=True)
class StructuredTransformerLayerExecutorConfig:
    """Portable native-shape configuration for one structured layer."""

    attention: AttentionSpec
    transformer: TransformerLayerSemantics
    causal_edges_enabled: bool = True
    _operator_sites_schema_present: bool = field(
        default=True,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.attention, AttentionSpec):
            raise TypeError("attention must be an AttentionSpec")
        if not isinstance(
            self.transformer,
            TransformerLayerSemantics,
        ):
            raise TypeError(
                "transformer must be TransformerLayerSemantics"
            )
        if type(self.causal_edges_enabled) is not bool:
            raise TypeError("causal_edges_enabled must be boolean")
        if type(self._operator_sites_schema_present) is not bool:
            raise TypeError(
                "_operator_sites_schema_present must be boolean"
            )
        if self.attention.kind not in (
            "global_causal",
            "sliding_causal",
        ):
            raise ValueError(
                "structured executor supports causal attention only"
            )
        if (
            self.attention.kind == "sliding_causal"
            and self.attention.window_size is None
        ):
            raise ValueError(
                "sliding attention requires a window_size"
            )
        if (
            self.attention.kind == "global_causal"
            and self.attention.window_size is not None
        ):
            raise ValueError(
                "global attention cannot carry a sliding window"
            )
        if self.attention.cache_kind != "none":
            raise ValueError(
                "structured executor currently supports cache-free prefill"
            )
        if self.attention.query_scale is None:
            raise ValueError(
                "structured executor requires an explicit query scale"
            )
        rope = self.attention.rope
        if (
            rope is None
            or rope.kind != "rotary"
            or rope.theta is None
            or rope.rotary_dimension != self.attention.head_dimension
        ):
            raise ValueError(
                "attention requires full-head rotary position semantics"
            )
        if self.attention.head_dimension % 2:
            raise ValueError("rotary head_dimension must be even")
        if rope.scaling_kind not in (None, "linear"):
            raise ValueError(
                "only default and linear RoPE scaling are supported"
            )
        if rope.scaling_kind == "linear" and rope.scaling_factor is None:
            raise ValueError("linear RoPE requires a scaling factor")
        if rope.scaling_kind is None and rope.scaling_factor is not None:
            raise ValueError(
                "default RoPE cannot carry a scaling factor"
            )
        transformer = self.transformer
        if transformer.residual_layout != (
            "sequential_attention_then_feed_forward_residual"
        ):
            raise ValueError("unsupported residual layout")
        for norm in (
            transformer.attention_input_norm,
            transformer.attention_output_norm,
            transformer.feed_forward_input_norm,
            transformer.feed_forward_output_norm,
        ):
            _require_gemma_rms_norm(norm)
        if any(
            norm.width != self.residual_width
            for norm in (
                transformer.attention_input_norm,
                transformer.attention_output_norm,
                transformer.feed_forward_input_norm,
                transformer.feed_forward_output_norm,
            )
        ):
            raise ValueError(
                "branch normalization widths must match residual width"
            )
        if self.attention.qk_norm:
            if transformer.qk_norm is None:
                raise ValueError(
                    "attention requires detailed qk normalization semantics"
                )
            _require_gemma_rms_norm(transformer.qk_norm)
            if (
                transformer.qk_norm.width
                != self.attention.head_dimension
            ):
                raise ValueError(
                    "qk normalization width must match head dimension"
                )
        elif transformer.qk_norm is not None:
            raise ValueError(
                "qk normalization semantics conflict with attention metadata"
            )
        if transformer.feed_forward.kind != "gated_multiplicative":
            raise ValueError("unsupported feed-forward topology")
        if transformer.feed_forward.activation not in (
            "gelu_pytorch_tanh",
            "gelu",
            "silu",
        ):
            raise ValueError("unsupported feed-forward activation")

    @property
    def residual_width(self) -> int:
        return self.transformer.attention_input_norm.width

    def to_dict(self) -> dict[str, object]:
        transformer = asdict(self.transformer)
        if not self._operator_sites_schema_present:
            transformer.pop("operator_sites")
        return {
            "attention": asdict(self.attention),
            "transformer": transformer,
            "causal_edges_enabled": self.causal_edges_enabled,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> StructuredTransformerLayerExecutorConfig:
        raw = _exact_mapping(value, _CONFIG_FIELDS, label="config")
        transformer_raw = raw["transformer"]
        return cls(
            attention=_attention_from_dict(raw["attention"]),
            transformer=_transformer_from_dict(transformer_raw),
            causal_edges_enabled=raw[  # type: ignore[arg-type]
                "causal_edges_enabled"
            ],
            _operator_sites_schema_present=(
                isinstance(transformer_raw, Mapping)
                and "operator_sites" in transformer_raw
            ),
        )

    @classmethod
    def from_layer_spec(
        cls,
        layer: LayerSpec,
        *,
        causal_edges_enabled: bool = True,
    ) -> StructuredTransformerLayerExecutorConfig:
        if not isinstance(layer, LayerSpec):
            raise TypeError("layer must be a LayerSpec")
        if layer.attention is None or layer.transformer is None:
            raise ValueError(
                "layer does not expose structured transformer semantics"
            )
        return cls(
            attention=layer.attention,
            transformer=layer.transformer,
            causal_edges_enabled=causal_edges_enabled,
        )


def _require_gemma_rms_norm(spec: NormalizationSpec) -> None:
    if (
        spec.kind != "rms_norm"
        or not spec.affine
        or spec.scale_parameterization != "unit_offset"
        or spec.compute_dtype != "float32"
    ):
        raise ValueError(
            "structured executor currently requires Gemma-style RMSNorm"
        )


class _StructuredRMSNorm(nn.Module):
    def __init__(
        self,
        spec: NormalizationSpec,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        _require_gemma_rms_norm(spec)
        self.spec = spec
        self.weight = nn.Parameter(
            torch.zeros(spec.width, dtype=dtype, device=device)
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True)
            + float(self.spec.epsilon)
        )
        normalized = normalized * (1.0 + self.weight.float())
        return normalized.to(dtype=values.dtype)


class _GroupedQueryRoPEAttention(nn.Module):
    def __init__(
        self,
        residual_width: int,
        attention: AttentionSpec,
        qk_norm: NormalizationSpec | None,
        *,
        projection_bias: bool,
        dropout: float,
        logit_softcap: float | None,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.residual_width = residual_width
        self.spec = attention
        self.dropout = float(dropout)
        self.logit_softcap = logit_softcap
        query_width = attention.query_heads * attention.head_dimension
        key_value_width = (
            attention.key_value_heads * attention.head_dimension
        )
        self.q_proj = nn.Linear(
            residual_width,
            query_width,
            bias=projection_bias,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            residual_width,
            key_value_width,
            bias=projection_bias,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            residual_width,
            key_value_width,
            bias=projection_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            query_width,
            residual_width,
            bias=projection_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = (
            _StructuredRMSNorm(
                qk_norm,
                dtype=dtype,
                device=device,
            )
            if qk_norm is not None
            else nn.Identity()
        )
        self.k_norm = (
            _StructuredRMSNorm(
                qk_norm,
                dtype=dtype,
                device=device,
            )
            if qk_norm is not None
            else nn.Identity()
        )
        rope = attention.rope
        assert rope is not None and rope.theta is not None
        inv_frequency = 1.0 / (
            float(rope.theta)
            ** (
                torch.arange(
                    0,
                    attention.head_dimension,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / attention.head_dimension
            )
        )
        self.register_buffer(
            "inv_frequency",
            inv_frequency,
            persistent=False,
        )

    @property
    def query_width(self) -> int:
        return self.spec.query_heads * self.spec.head_dimension

    @property
    def key_value_width(self) -> int:
        return self.spec.key_value_heads * self.spec.head_dimension

    @staticmethod
    def _rotate_half(values: Tensor) -> Tensor:
        half = values.shape[-1] // 2
        return torch.cat(
            (-values[..., half:], values[..., :half]),
            dim=-1,
        )

    def _rotary_cos_sin(
        self,
        positions: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        rope = self.spec.rope
        assert rope is not None
        position_values = positions.float()
        if rope.scaling_kind == "linear":
            assert rope.scaling_factor is not None
            position_values = position_values / float(rope.scaling_factor)
        frequencies = torch.einsum(
            "bt,d->btd",
            position_values,
            self.inv_frequency.float(),
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype=dtype), embedding.sin().to(
            dtype=dtype
        )

    @staticmethod
    def _repeat_key_value(values: Tensor, repeats: int) -> Tensor:
        if repeats == 1:
            return values
        batch, heads, length, width = values.shape
        expanded = values[:, :, None, :, :].expand(
            batch,
            heads,
            repeats,
            length,
            width,
        )
        return expanded.reshape(
            batch,
            heads * repeats,
            length,
            width,
        )

    def allowed_pairs(self, sequence: SequenceContext) -> Tensor:
        query_ordinals = torch.arange(
            sequence.query_length,
            device=sequence.device,
        ).view(1, sequence.query_length, 1)
        key_ordinals = torch.arange(
            sequence.key_length,
            device=sequence.device,
        ).view(1, 1, sequence.key_length)
        allowed = key_ordinals <= query_ordinals
        allowed = allowed & sequence.key_valid_mask.unsqueeze(1)
        allowed = allowed & sequence.query_valid_mask.unsqueeze(2)
        if self.spec.window_size is not None:
            allowed = allowed & (
                query_ordinals - key_ordinals < self.spec.window_size
            )
        return allowed

    def projection_features(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        batch, length, _ = hidden_states.shape
        heads = self.spec.query_heads
        key_value_heads = self.spec.key_value_heads
        head_dimension = self.spec.head_dimension

        query = self.q_proj(hidden_states).view(
            batch,
            length,
            heads,
            head_dimension,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch,
            length,
            key_value_heads,
            head_dimension,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch,
            length,
            key_value_heads,
            head_dimension,
        ).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)

        cos, sin = self._rotary_cos_sin(
            sequence.logical_positions,
            dtype=query.dtype,
        )
        query = (
            query * cos.unsqueeze(1)
            + self._rotate_half(query) * sin.unsqueeze(1)
        )
        key = (
            key * cos.unsqueeze(1)
            + self._rotate_half(key) * sin.unsqueeze(1)
        )
        query = record(trace, f"{prefix}.query", query)
        key = record(trace, f"{prefix}.key", key)
        value = record(trace, f"{prefix}.value", value)

        repeats = heads // key_value_heads
        repeated_key = self._repeat_key_value(key, repeats)
        repeated_value = self._repeat_key_value(value, repeats)
        scores = torch.matmul(
            query,
            repeated_key.transpose(-2, -1),
        ) * float(self.spec.query_scale)
        if self.logit_softcap is not None:
            cap = float(self.logit_softcap)
            scores = torch.tanh(scores / cap) * cap
        allowed = self.allowed_pairs(sequence).unsqueeze(1)
        additive = torch.zeros_like(scores)
        additive.masked_fill_(
            ~allowed,
            torch.finfo(scores.dtype).min,
        )
        scores = record(
            trace,
            f"{prefix}.scores",
            scores + additive,
        )
        probabilities = F.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(dtype=query.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        probabilities = record(
            trace,
            f"{prefix}.probabilities",
            probabilities,
        )
        context = torch.matmul(probabilities, repeated_value)
        context = context.transpose(1, 2).contiguous().view(
            batch,
            length,
            self.query_width,
        )
        return record(
            trace,
            f"{prefix}.pre_o_proj",
            context,
        )

    def project(
        self,
        features: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        return record(
            trace,
            f"{prefix}.output",
            self.o_proj(features),
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        return self.project(
            self.projection_features(
                hidden_states,
                sequence,
                trace=trace,
                prefix=prefix,
            ),
            trace=trace,
            prefix=prefix,
        )


class _GatedFeedForward(nn.Module):
    def __init__(
        self,
        residual_width: int,
        spec: FeedForwardSpec,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.gate_proj = nn.Linear(
            residual_width,
            spec.intermediate_width,
            bias=spec.projection_bias,
            dtype=dtype,
            device=device,
        )
        self.up_proj = nn.Linear(
            residual_width,
            spec.intermediate_width,
            bias=spec.projection_bias,
            dtype=dtype,
            device=device,
        )
        self.down_proj = nn.Linear(
            spec.intermediate_width,
            residual_width,
            bias=spec.projection_bias,
            dtype=dtype,
            device=device,
        )

    def _activation(self, values: Tensor) -> Tensor:
        if self.spec.activation == "gelu_pytorch_tanh":
            return F.gelu(values, approximate="tanh")
        if self.spec.activation == "gelu":
            return F.gelu(values)
        if self.spec.activation == "silu":
            return F.silu(values)
        raise RuntimeError("validated activation became unsupported")

    def projection_features(
        self,
        hidden_states: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        gate = record(
            trace,
            f"{prefix}.gate",
            self.gate_proj(hidden_states),
        )
        up = record(
            trace,
            f"{prefix}.up",
            self.up_proj(hidden_states),
        )
        activated_gate = record(
            trace,
            f"{prefix}.activated_gate",
            self._activation(gate),
        )
        return record(
            trace,
            f"{prefix}.pre_down_proj",
            activated_gate * up,
        )

    def project(
        self,
        features: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        return record(
            trace,
            f"{prefix}.output",
            self.down_proj(features),
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        trace: ActivationTrace | None,
        prefix: str,
    ) -> Tensor:
        return self.project(
            self.projection_features(
                hidden_states,
                trace=trace,
                prefix=prefix,
            ),
            trace=trace,
            prefix=prefix,
        )


@dataclass(frozen=True, slots=True)
class StructuredTransformerLayerExecution:
    """Inspectable residual-stage outputs from one executor call."""

    normalized_attention_input: Tensor
    attention_projection_input: Tensor
    attention_operator_output: Tensor
    attention_delta: Tensor
    post_attention: Tensor
    normalized_feed_forward_input: Tensor
    feed_forward_projection_input: Tensor
    feed_forward_operator_output: Tensor
    feed_forward_delta: Tensor
    output: Tensor


@dataclass(frozen=True, slots=True)
class StructuredTransformerLayerAccounting:
    """Logical native-shaped MAC ledger.

    Counts exclude additions, normalization, activation, masking, softmax,
    and RoPE.  They are arithmetic-shape estimates, not latency claims.
    """

    valid_tokens: int
    logical_causal_key_pairs: int
    attention_projection_macs: int
    attention_score_macs: int
    attention_value_macs: int
    feed_forward_macs: int

    @property
    def logical_total_macs(self) -> int:
        return (
            self.attention_projection_macs
            + self.attention_score_macs
            + self.attention_value_macs
            + self.feed_forward_macs
        )


class StructuredTransformerLayerExecutor(nn.Module):
    """Execute a portable, Gemma-shaped transformer layer."""

    def __init__(
        self,
        config: StructuredTransformerLayerExecutorConfig,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(
            config,
            StructuredTransformerLayerExecutorConfig,
        ):
            raise TypeError(
                "config must be StructuredTransformerLayerExecutorConfig"
            )
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("executor dtype must be floating point")
        _dtype_name(dtype)
        self.config = config
        semantics = config.transformer
        self.attention_input_norm = _StructuredRMSNorm(
            semantics.attention_input_norm,
            dtype=dtype,
            device=device,
        )
        self.attention = _GroupedQueryRoPEAttention(
            config.residual_width,
            config.attention,
            semantics.qk_norm,
            projection_bias=semantics.attention_projection_bias,
            dropout=semantics.attention_dropout,
            logit_softcap=semantics.attention_logit_softcap,
            dtype=dtype,
            device=device,
        )
        self.attention_output_norm = _StructuredRMSNorm(
            semantics.attention_output_norm,
            dtype=dtype,
            device=device,
        )
        self.feed_forward_input_norm = _StructuredRMSNorm(
            semantics.feed_forward_input_norm,
            dtype=dtype,
            device=device,
        )
        self.feed_forward = _GatedFeedForward(
            config.residual_width,
            semantics.feed_forward,
            dtype=dtype,
            device=device,
        )
        self.feed_forward_output_norm = _StructuredRMSNorm(
            semantics.feed_forward_output_norm,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "_weight_origin",
            torch.zeros((), dtype=torch.uint8, device=device),
            persistent=True,
        )

    @property
    def width(self) -> int:
        return self.config.residual_width

    @property
    def retained_rank(self) -> int:
        return self.width

    @property
    def dtype(self) -> torch.dtype:
        return self.attention.q_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.attention.q_proj.weight.device

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def fixed_runtime_coefficient_count(self) -> int:
        return 0

    @property
    def total_runtime_coefficient_count(self) -> int:
        return self.learned_parameter_count

    @property
    def executor_local_source_free(self) -> bool:
        return not self.owns_source_model_weights

    @property
    def owns_source_model_weights(self) -> bool:
        return bool(self._weight_origin.item())

    @property
    def owns_source_fallback(self) -> bool:
        return False

    @property
    def causal_edge_control(self) -> str:
        return (
            "enabled"
            if self.config.causal_edges_enabled
            else "attention_output_zeroed_storage_matched"
        )

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "structured_transformer_layer",
            "config": self.config.to_dict(),
            "full_residual_width": True,
            "rank_reduction": False,
            "prefill_only": True,
            "cache_supported": False,
        }

    def _validate_inputs(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
    ) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError(
                "structured executor supports cache-free prefill only"
            )
        if sequence.cache_positions is not None:
            raise ValueError(
                "structured cache-free prefill rejects cache_positions"
            )
        if (
            not sequence.input_origin.attention_mask_supplied
            and sequence.query_length > 1
            and bool(
                (
                    sequence.logical_positions[:, 1:]
                    - sequence.logical_positions[:, :-1]
                    != 1
                ).any()
            )
        ):
            raise ValueError(
                "structured execution cannot represent packed or gapped "
                "position_ids without an explicit attention_mask"
            )
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.width,
        )
        if (
            not isinstance(hidden_states, Tensor)
            or tuple(hidden_states.shape) != expected
        ):
            raise ValueError(
                "hidden_states shape does not match executor and sequence"
            )
        if not hidden_states.is_floating_point():
            raise ValueError("hidden_states must be floating point")
        if hidden_states.device != sequence.device:
            raise ValueError(
                "hidden_states and sequence context must share a device"
            )
        if hidden_states.device != self.device:
            raise ValueError(
                "hidden_states and executor must share a device"
            )
        if sequence.key_length != sequence.query_length:
            raise ValueError(
                "structured prefill requires equal query and key lengths"
            )
        if not torch.equal(
            sequence.logical_positions,
            sequence.key_logical_positions,
        ):
            raise ValueError(
                "structured prefill requires identical query/key positions"
            )
        if not torch.equal(
            sequence.query_valid_mask,
            sequence.key_valid_mask,
        ):
            raise ValueError(
                "structured middle-layer prefill requires identical "
                "query/key validity"
            )

    def attention_projection_features(
        self,
        normalized_attention_input: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str = "structured_layer.attention",
    ) -> Tensor:
        """Return causal attention context immediately before ``o_proj``."""

        self._validate_inputs(normalized_attention_input, sequence)
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("attention feature prefix must be nonempty")
        return self.attention.projection_features(
            normalized_attention_input,
            sequence,
            trace=trace,
            prefix=prefix,
        )

    def feed_forward_projection_features(
        self,
        normalized_feed_forward_input: Tensor,
        *,
        trace: ActivationTrace | None = None,
        prefix: str = "structured_layer.mlp",
    ) -> Tensor:
        """Return activated gate-times-up features before ``down_proj``."""

        if (
            not isinstance(normalized_feed_forward_input, Tensor)
            or normalized_feed_forward_input.ndim != 3
            or normalized_feed_forward_input.shape[-1] != self.width
        ):
            raise ValueError(
                "normalized_feed_forward_input must have shape "
                "[batch, sequence, residual_width]"
            )
        if not normalized_feed_forward_input.is_floating_point():
            raise ValueError(
                "normalized_feed_forward_input must be floating point"
            )
        if normalized_feed_forward_input.device != self.device:
            raise ValueError(
                "feed-forward inputs and executor must share a device"
            )
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("feed-forward feature prefix must be nonempty")
        return self.feed_forward.projection_features(
            normalized_feed_forward_input,
            trace=trace,
            prefix=prefix,
        )

    def forward_components(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str | None = "structured_layer",
    ) -> StructuredTransformerLayerExecution:
        self._validate_inputs(hidden_states, sequence)
        attention_stage, feed_forward_stage = (
            self.config.transformer.stages
        )
        if prefix is None:
            sites = {
                "input": attention_stage.input_site,
                "normalized_attention": (
                    attention_stage.normalized_input_site
                ),
                "attention_operator": (
                    attention_stage.operator_output_site
                ),
                "attention_delta": attention_stage.delta_site,
                "post_attention": attention_stage.output_site,
                "normalized_feed_forward": (
                    feed_forward_stage.normalized_input_site
                ),
                "feed_forward_operator": (
                    feed_forward_stage.operator_output_site
                ),
                "feed_forward_delta": feed_forward_stage.delta_site,
                "output": feed_forward_stage.output_site,
                "attention_prefix": (
                    attention_stage.operator_output_site.rsplit(
                        ".operator_output",
                        1,
                    )[0]
                ),
                "feed_forward_prefix": (
                    feed_forward_stage.operator_output_site.rsplit(
                        ".operator_output",
                        1,
                    )[0]
                ),
            }
        else:
            sites = {
                "input": f"{prefix}.input",
                "normalized_attention": (
                    f"{prefix}.attention.normalized_input"
                ),
                "attention_operator": (
                    f"{prefix}.attention.operator_output"
                ),
                "attention_delta": f"{prefix}.attention.delta",
                "post_attention": f"{prefix}.post_attention",
                "normalized_feed_forward": (
                    f"{prefix}.mlp.normalized_input"
                ),
                "feed_forward_operator": (
                    f"{prefix}.mlp.operator_output"
                ),
                "feed_forward_delta": f"{prefix}.mlp.delta",
                "output": f"{prefix}.output",
                "attention_prefix": f"{prefix}.attention",
                "feed_forward_prefix": f"{prefix}.mlp",
            }
        residual = record(trace, sites["input"], hidden_states)
        normalized_attention = record(
            trace,
            sites["normalized_attention"],
            self.attention_input_norm(residual),
        )
        attention_projection_input = self.attention_projection_features(
            normalized_attention,
            sequence,
            trace=trace,
            prefix=sites["attention_prefix"],
        )
        attention_operator = self.attention.project(
            attention_projection_input,
            trace=trace,
            prefix=sites["attention_prefix"],
        )
        if not self.config.causal_edges_enabled:
            attention_operator = torch.zeros_like(attention_operator)
        attention_operator = record(
            trace,
            sites["attention_operator"],
            attention_operator,
        )
        attention_delta = self.attention_output_norm(attention_operator)
        attention_delta = attention_delta.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        attention_delta = record(
            trace,
            sites["attention_delta"],
            attention_delta,
        )
        post_attention = record(
            trace,
            sites["post_attention"],
            residual + attention_delta,
        )
        normalized_feed_forward = record(
            trace,
            sites["normalized_feed_forward"],
            self.feed_forward_input_norm(post_attention),
        )
        feed_forward_projection_input = (
            self.feed_forward_projection_features(
                normalized_feed_forward,
                trace=trace,
                prefix=sites["feed_forward_prefix"],
            )
        )
        feed_forward_operator = self.feed_forward.project(
            feed_forward_projection_input,
            trace=trace,
            prefix=sites["feed_forward_prefix"],
        )
        feed_forward_operator = record(
            trace,
            sites["feed_forward_operator"],
            feed_forward_operator,
        )
        feed_forward_delta = self.feed_forward_output_norm(
            feed_forward_operator
        )
        feed_forward_delta = feed_forward_delta.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        feed_forward_delta = record(
            trace,
            sites["feed_forward_delta"],
            feed_forward_delta,
        )
        output = record(
            trace,
            sites["output"],
            post_attention + feed_forward_delta,
        )
        return StructuredTransformerLayerExecution(
            normalized_attention_input=normalized_attention,
            attention_projection_input=attention_projection_input,
            attention_operator_output=attention_operator,
            attention_delta=attention_delta,
            post_attention=post_attention,
            normalized_feed_forward_input=normalized_feed_forward,
            feed_forward_projection_input=(
                feed_forward_projection_input
            ),
            feed_forward_operator_output=feed_forward_operator,
            feed_forward_delta=feed_forward_delta,
            output=output,
        )

    def forward(
        self,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
        prefix: str | None = "structured_layer",
    ) -> Tensor:
        return self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
            prefix=prefix,
        ).output

    def run(
        self,
        segment: CompiledSegment,
        hidden_states: Tensor,
        sequence: SequenceContext,
        *,
        trace: ActivationTrace | None = None,
    ) -> SegmentRun:
        if not isinstance(segment, CompiledSegment):
            raise TypeError("segment must be a CompiledSegment")
        attention_stage, feed_forward_stage = (
            self.config.transformer.stages
        )
        if (
            segment.input_activation != attention_stage.input_site
            or segment.output_activation != feed_forward_stage.output_site
        ):
            raise ValueError(
                "compiled segment activation binding does not match "
                "structured layer semantics"
            )
        components = self.forward_components(
            hidden_states,
            sequence,
            trace=trace,
            prefix=None,
        )
        return SegmentRun(
            hidden_states=components.output,
            sequence=sequence,
            raw_output={
                "attention_delta": components.attention_delta,
                "post_attention": components.post_attention,
                "feed_forward_delta": components.feed_forward_delta,
                "executor_local_source_free": (
                    self.executor_local_source_free
                ),
                "owns_source_fallback": self.owns_source_fallback,
            },
        )

    def logical_accounting(
        self,
        sequence: SequenceContext,
    ) -> StructuredTransformerLayerAccounting:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError("accounting supports cache-free prefill only")
        if sequence.cache_positions is not None:
            raise ValueError(
                "accounting rejects cache positions without cache support"
            )
        if (
            not sequence.input_origin.attention_mask_supplied
            and sequence.query_length > 1
            and bool(
                (
                    sequence.logical_positions[:, 1:]
                    - sequence.logical_positions[:, :-1]
                    != 1
                ).any()
            )
        ):
            raise ValueError(
                "accounting cannot represent packed or gapped position_ids "
                "without an explicit attention_mask"
            )
        if (
            sequence.key_length != sequence.query_length
            or not torch.equal(
                sequence.logical_positions,
                sequence.key_logical_positions,
            )
            or not torch.equal(
                sequence.query_valid_mask,
                sequence.key_valid_mask,
            )
        ):
            raise ValueError(
                "accounting requires full middle-layer prefill semantics"
            )
        valid_tokens = int(sequence.key_valid_mask.sum().item())
        pairs = int(self.attention.allowed_pairs(sequence).sum().item())
        attention = self.config.attention
        residual = self.width
        query_width = attention.query_heads * attention.head_dimension
        key_value_width = (
            attention.key_value_heads * attention.head_dimension
        )
        projections = valid_tokens * residual * (
            query_width + 2 * key_value_width
        )
        projections += valid_tokens * query_width * residual
        scores = (
            pairs * attention.query_heads * attention.head_dimension
        )
        values = scores
        intermediate = (
            self.config.transformer.feed_forward.intermediate_width
        )
        feed_forward = valid_tokens * (
            2 * residual * intermediate
            + intermediate * residual
        )
        return StructuredTransformerLayerAccounting(
            valid_tokens=valid_tokens,
            logical_causal_key_pairs=pairs,
            attention_projection_macs=projections,
            attention_score_macs=scores,
            attention_value_macs=values,
            feed_forward_macs=feed_forward,
        )

    def execution_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        digest.update(
            json.dumps(
                self.architecture_manifest(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(
                [
                    (name, module.training)
                    for name, module in self.named_modules()
                ],
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(module_state_fingerprint(self).encode("ascii"))
        return digest.hexdigest()

    def artifact_state_dict(self) -> dict[str, object]:
        training_modules = tuple(
            name or "<root>"
            for name, module in self.named_modules()
            if module.training
        )
        if training_modules:
            raise RuntimeError(
                "structured executor artifacts require every module in "
                f"eval mode; training modules: {list(training_modules)}"
            )
        if self.owns_source_model_weights:
            raise RuntimeError(
                "test-only source-weight transplants cannot be serialized"
            )
        model_state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }
        for name, value in model_state.items():
            valid = (
                value.dtype is torch.uint8
                and value.ndim == 0
                and int(value.item()) == 0
                if name == "_weight_origin"
                else (
                    value.is_floating_point()
                    and bool(torch.isfinite(value).all())
                )
            )
            if not valid:
                raise ValueError(
                    f"structured executor tensor {name!r} is invalid"
                )
        return {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "contains_source_model_weights": False,
            "contains_source_fallback": False,
            "config": self.config.to_dict(),
            "dtype": _dtype_name(self.dtype),
            "model_state_dict": model_state,
            "execution_fingerprint": self.execution_fingerprint(),
        }

    @classmethod
    def from_artifact_state_dict(
        cls,
        state: Mapping[str, object],
        *,
        map_location: torch.device | str = "cpu",
    ) -> StructuredTransformerLayerExecutor:
        raw = _exact_mapping(
            state,
            _ARTIFACT_FIELDS,
            label="structured executor artifact",
        )
        dtype_name = raw["dtype"]
        if (
            raw["artifact_kind"] != _ARTIFACT_KIND
            or raw["format_version"] != _FORMAT_VERSION
            or raw["contains_source_model_weights"] is not False
            or raw["contains_source_fallback"] is not False
            or not isinstance(dtype_name, str)
            or dtype_name not in _DTYPES
            or not isinstance(raw["execution_fingerprint"], str)
        ):
            raise ValueError("unsupported structured executor artifact")
        config = StructuredTransformerLayerExecutorConfig.from_dict(
            raw["config"]
        )
        result = cls(
            config,
            dtype=_DTYPES[dtype_name],
            device=map_location,
        )
        model_state = raw["model_state_dict"]
        if not isinstance(model_state, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Tensor)
            for name, value in model_state.items()
        ):
            raise ValueError("structured executor model state is invalid")
        expected = result.state_dict()
        if set(model_state) != set(expected):
            raise ValueError(
                "structured executor model-state fields are invalid"
            )
        converted: dict[str, Tensor] = {}
        for name, expected_value in expected.items():
            value = model_state[name]
            assert isinstance(value, Tensor)
            tensor_is_valid = (
                value.shape == expected_value.shape
                and value.dtype == expected_value.dtype
            )
            if name == "_weight_origin":
                tensor_is_valid = (
                    tensor_is_valid
                    and value.dtype is torch.uint8
                    and value.ndim == 0
                    and int(value.item()) == 0
                )
            else:
                tensor_is_valid = (
                    tensor_is_valid
                    and value.is_floating_point()
                    and bool(torch.isfinite(value).all())
                )
            if not tensor_is_valid:
                raise ValueError(
                    f"structured executor tensor {name!r} is invalid"
                )
            converted[name] = value.to(device=map_location).clone()
        result.load_state_dict(converted, strict=True)
        result.eval()
        if result.execution_fingerprint() != raw["execution_fingerprint"]:
            raise ValueError(
                "structured executor execution fingerprint mismatch"
            )
        return result

    def transplant_gemma3_layer_weights_(
        self,
        source_layer: nn.Module,
    ) -> StructuredTransformerLayerExecutor:
        """Copy a native Gemma layer solely for operator-parity tests.

        The method marks the executor as contaminated by source weights.
        :meth:`artifact_state_dict` then fails closed so a transplanted
        control cannot be mistaken for a compiled candidate.
        """

        if not isinstance(source_layer, nn.Module):
            raise TypeError("source_layer must be an nn.Module")
        module_pairs = (
            (self.attention_input_norm, "input_layernorm"),
            (self.attention, "self_attn"),
            (self.attention_output_norm, "post_attention_layernorm"),
            (self.feed_forward_input_norm, "pre_feedforward_layernorm"),
            (self.feed_forward, "mlp"),
            (self.feed_forward_output_norm, "post_feedforward_layernorm"),
        )
        source_modules: list[nn.Module] = []
        for _, name in module_pairs:
            module = getattr(source_layer, name, None)
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"source Gemma layer is missing module {name!r}"
                )
            source_modules.append(module)
        candidate = copy.deepcopy(self.state_dict())
        mappings = (
            (
                self.attention_input_norm,
                source_modules[0],
                ("weight",),
            ),
            (
                self.attention,
                source_modules[1],
                (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                    "q_proj.bias",
                    "k_proj.bias",
                    "v_proj.bias",
                    "o_proj.bias",
                ),
            ),
            (
                self.attention_output_norm,
                source_modules[2],
                ("weight",),
            ),
            (
                self.feed_forward_input_norm,
                source_modules[3],
                ("weight",),
            ),
            (
                self.feed_forward,
                source_modules[4],
                (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                    "gate_proj.bias",
                    "up_proj.bias",
                    "down_proj.bias",
                ),
            ),
            (
                self.feed_forward_output_norm,
                source_modules[5],
                ("weight",),
            ),
        )
        try:
            with torch.no_grad():
                for destination, source, names in mappings:
                    destination_state = destination.state_dict()
                    source_state = source.state_dict()
                    required = {
                        name
                        for name in names
                        if name in destination_state
                    }
                    if required != set(destination_state):
                        raise ValueError(
                            "destination/source transplant schema is "
                            "unsupported"
                        )
                    if not required <= set(source_state):
                        raise ValueError(
                            "source Gemma layer transplant tensors are "
                            "missing"
                        )
                    for name in required:
                        destination_value = destination_state[name]
                        source_value = source_state[name]
                        if (
                            destination_value.shape != source_value.shape
                            or not source_value.is_floating_point()
                        ):
                            raise ValueError(
                                f"source transplant tensor {name!r} "
                                "has an incompatible shape"
                            )
                    destination.load_state_dict(
                        {
                            name: source_state[name]
                            .detach()
                            .to(
                                device=self.device,
                                dtype=self.dtype,
                            )
                            .clone()
                            for name in required
                        },
                        strict=True,
                    )
        except BaseException:
            self.load_state_dict(candidate, strict=True)
            raise
        self._weight_origin.fill_(1)
        return self


__all__ = [
    "StructuredTransformerLayerAccounting",
    "StructuredTransformerLayerExecution",
    "StructuredTransformerLayerExecutor",
    "StructuredTransformerLayerExecutorConfig",
]
