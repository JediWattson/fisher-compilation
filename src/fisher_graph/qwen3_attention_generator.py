"""Qwen 3 attention generator: Gemma machinery, Llama-style norm layout.

Qwen 3 decoder layers apply no normalization to residual deltas (the
attention/MLP operator outputs are added back unmodified), and its RMSNorm
modules use the direct ``weight * normalized`` parameterization rather than
Gemma's unit-offset ``(1 + weight) * normalized``.

The generator's stored coefficients keep the established unit-offset
convention so the shared fitting stack (``(1 + w)`` Fisher anchors,
closed-form projections, serialized artifacts) needs no changes: direct
source weights are converted to offset form (``w - 1``) at extraction.
The absent delta norm is recorded as ``kind == "identity"`` in the config
and carried as a vestigial zeros weight; ``_rms_norm`` passes values
through unchanged under that kind.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .adapters.base import LayerSpec, NormalizationSpec
from .gemma3_attention_generator import (
    ExactAttentionGeneratorCoefficients,
    Gemma3AttentionGenerator,
    Gemma3AttentionGeneratorConfig,
)


def _require_qwen3_norm(
    spec: NormalizationSpec,
    *,
    width: int,
    label: str,
    allow_identity: bool = False,
) -> None:
    if not isinstance(spec, NormalizationSpec) or spec.width != width:
        raise ValueError(f"{label} must be a width-matched normalization spec")
    if spec.kind == "identity":
        if not allow_identity:
            raise ValueError(f"{label} may not be an identity norm")
        return
    if (
        spec.kind != "rms_norm"
        or not spec.affine
        or spec.scale_parameterization
        not in ("unit_offset", "direct")
        or spec.compute_dtype != "float32"
    ):
        raise ValueError(f"{label} must be a width-matched Qwen 3 RMSNorm")


class Qwen3AttentionGeneratorConfig(Gemma3AttentionGeneratorConfig):
    """Qwen 3 semantics: identity output norm allowed, direct scale allowed."""

    def __post_init__(self) -> None:  # noqa: C901 - mirrors parent validation
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("invocation_id must be nonempty")
        if type(self.residual_width) is not int or self.residual_width <= 0:
            raise ValueError("residual_width must be positive")
        attention = self.attention
        if attention.kind not in ("global_causal", "sliding_causal"):
            raise ValueError("attention generator requires causal attention")
        if (
            attention.kind == "global_causal"
            and attention.window_size is not None
        ):
            raise ValueError("global attention cannot have a sliding window")
        if (
            attention.kind == "sliding_causal"
            and attention.window_size is None
        ):
            raise ValueError("sliding attention requires a window")
        if attention.cache_kind != "none":
            raise ValueError("attention generator v1 is cache-free prefill")
        if attention.query_scale is None:
            raise ValueError("attention query scale must be explicit")
        rope = attention.rope
        if (
            rope is None
            or rope.kind != "rotary"
            or rope.theta is None
            or rope.rotary_dimension != attention.head_dimension
            or attention.head_dimension % 2
        ):
            raise ValueError("attention requires full even-head RoPE")
        if rope.scaling_kind not in (None, "linear"):
            raise ValueError("only default and linear RoPE are supported")
        if rope.scaling_kind == "linear" and rope.scaling_factor is None:
            raise ValueError("linear RoPE requires a scaling factor")
        if rope.scaling_kind is None and rope.scaling_factor is not None:
            raise ValueError("default RoPE cannot carry a scaling factor")
        _require_qwen3_norm(
            self.input_norm,
            width=self.residual_width,
            label="input_norm",
        )
        _require_qwen3_norm(
            self.output_norm,
            width=self.residual_width,
            label="output_norm",
            allow_identity=True,
        )
        if attention.qk_norm:
            if self.qk_norm is None:
                raise ValueError("qk_norm semantics are required")
            _require_qwen3_norm(
                self.qk_norm,
                width=attention.head_dimension,
                label="qk_norm",
            )
        elif self.qk_norm is not None:
            raise ValueError("qk_norm semantics contradict attention metadata")
        if type(self.projection_bias) is not bool:
            raise TypeError("projection_bias must be boolean")
        if (
            not isinstance(self.dropout, (float, int))
            or isinstance(self.dropout, bool)
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("dropout must be finite in [0, 1)")
        if self.logit_softcap is not None and (
            not isinstance(self.logit_softcap, (float, int))
            or isinstance(self.logit_softcap, bool)
            or not math.isfinite(float(self.logit_softcap))
            or float(self.logit_softcap) <= 0.0
        ):
            raise ValueError("logit_softcap must be finite and positive")


class Qwen3AttentionGenerator(Gemma3AttentionGenerator):
    """Gemma attention machinery with Qwen 3 norm semantics.

    ``_rms_norm`` honours ``kind == "identity"`` (pass-through for the
    absent Qwen 3 delta norm) and ``scale_parameterization == "direct"``
    (multiply by weight, not ``1 + weight``).  All other math — coordinate
    encoding, causal routing, RoPE, grouped-query attention, output
    projection — is shared with the Gemma implementation unchanged.
    """

    @staticmethod
    def _rms_norm(
        values: Tensor,
        weight: Tensor,
        spec: NormalizationSpec,
    ) -> Tensor:
        if spec.kind == "identity":
            return values
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True)
            + float(spec.epsilon)
        )
        if spec.scale_parameterization == "direct":
            return (normalized * weight.float()).to(dtype=values.dtype)
        return (normalized * (1.0 + weight.float())).to(
            dtype=values.dtype
        )


def _module_weight(module: nn.Module | None, label: str) -> Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError(f"{label} module does not expose a weight tensor")
    return weight.detach()


def qwen3_config_from_layer_spec(
    layer: LayerSpec,
) -> Qwen3AttentionGeneratorConfig:
    """Generator config from an adapter ``LayerSpec``.

    Real norms are stored under the unit-offset convention because the
    corresponding coefficient weights are converted to offset form at
    extraction; the absent delta norm records ``kind == "identity"``.
    """

    if layer.attention is None or layer.transformer is None:
        raise ValueError(
            "layer does not expose structured transformer semantics"
        )
    semantics = layer.transformer

    def _stored(norm: NormalizationSpec) -> NormalizationSpec:
        if norm.kind == "identity":
            return norm
        if norm.scale_parameterization == "direct":
            return NormalizationSpec(
                kind=norm.kind,
                width=norm.width,
                epsilon=norm.epsilon,
                affine=norm.affine,
                scale_parameterization="unit_offset",
                compute_dtype=norm.compute_dtype,
            )
        return norm

    return Qwen3AttentionGeneratorConfig(
        invocation_id=semantics.stages[0].id,
        residual_width=layer.residual_width,
        attention=layer.attention,
        input_norm=_stored(semantics.attention_input_norm),
        output_norm=_stored(semantics.attention_output_norm),
        qk_norm=(
            _stored(semantics.qk_norm)
            if semantics.qk_norm is not None
            else None
        ),
        projection_bias=semantics.attention_projection_bias,
        dropout=float(semantics.attention_dropout),
        logit_softcap=semantics.attention_logit_softcap,
    )


def build_qwen3_exact_attention_generator(
    adapter: object,
    layer_id: str,
) -> Qwen3AttentionGenerator:
    """Extract an exact attention generator from a live Qwen 3 layer module.

    Direct-parameterized norm weights are converted to the stored
    unit-offset convention (``w - 1``); the absent delta norm contributes a
    zeros weight that is inert under ``kind == "identity"``.
    """

    layer_spec = adapter.layer(layer_id)
    config = qwen3_config_from_layer_spec(layer_spec)
    module = adapter.source_module(layer_id)
    self_attn = getattr(module, "self_attn", None)
    input_norm = getattr(module, "input_layernorm", None)
    if not isinstance(self_attn, nn.Module) or not isinstance(
        input_norm, nn.Module
    ):
        raise TypeError(
            "Qwen 3 layer must expose self_attn and input_layernorm"
        )

    def _offset(module: nn.Module | None, label: str) -> Tensor:
        return _module_weight(module, label) - 1.0

    coefficients = ExactAttentionGeneratorCoefficients(
        input_norm_weight=_offset(input_norm, "input_layernorm"),
        q_weight=_module_weight(
            getattr(self_attn, "q_proj", None), "q_proj"
        ),
        k_weight=_module_weight(
            getattr(self_attn, "k_proj", None), "k_proj"
        ),
        v_weight=_module_weight(
            getattr(self_attn, "v_proj", None), "v_proj"
        ),
        o_weight=_module_weight(
            getattr(self_attn, "o_proj", None), "o_proj"
        ),
        output_norm_weight=torch.zeros(
            layer_spec.residual_width,
            dtype=_module_weight(input_norm, "input_layernorm").dtype,
        ),
        q_norm_weight=_offset(
            getattr(self_attn, "q_norm", None), "q_norm"
        ),
        k_norm_weight=_offset(
            getattr(self_attn, "k_norm", None), "k_norm"
        ),
        q_bias=getattr(getattr(self_attn, "q_proj", None), "bias", None),
        k_bias=getattr(getattr(self_attn, "k_proj", None), "bias", None),
        v_bias=getattr(getattr(self_attn, "v_proj", None), "bias", None),
        o_bias=getattr(getattr(self_attn, "o_proj", None), "bias", None),
    )
    return Qwen3AttentionGenerator(
        config,
        exact=coefficients,
        source_checkpoint_coefficients_retained=True,
    )
