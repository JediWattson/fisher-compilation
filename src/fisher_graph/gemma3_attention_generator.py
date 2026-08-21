"""Source-independent Gemma 3 attention generators.

This module is the attention-specific seam for a cross-block modal executor.
It owns the complete Gemma attention mutation grammar -- input RMSNorm,
Q/K/V projections, per-head Q/K RMSNorm, RoPE, grouped-query causal mixing,
the output projection, output RMSNorm, and valid-row masking -- without
retaining a native attention module or a source fallback.

Two deliberately separate controls are provided:

``exact_full_rank``
    Stores the native-shaped projection coefficients and is the source-free
    parity control.

``reduced_coordinates``
    Shares one incoming chart across Q/K/V and factorizes the output
    projection through a compact coordinate packet.  Attention heads and the
    sequence operator remain native-shaped, so this rung tests projection and
    cross-block coordinate compression rather than a different attention
    algorithm.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters.base import (
    AttentionSpec,
    LayerSpec,
    NormalizationSpec,
    SequenceContext,
)
from .structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)


__all__ = [
    "ATTENTION_GENERATOR_KINDS",
    "ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID",
    "ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT",
    "ATTENTION_ROUTE_PROBABILITY_SIMPLEX_SUPPORT_DERIVATION_ID",
    "AttentionRouteProbabilitySimplexAudit",
    "AttentionCoordinatePacket",
    "Gemma3AttentionGenerator",
    "Gemma3AttentionGeneratorConfig",
    "Gemma3AttentionGeneratorExecution",
    "Gemma3AttentionGeneratorLogicalAccounting",
    "Gemma3AttentionGeneratorParameterAccounting",
    "Gemma3AttentionGeneratorReceipt",
    "Gemma3AttentionRouteFactorSnapshot",
    "ExactAttentionGeneratorCoefficients",
    "ReducedAttentionGeneratorCoefficients",
    "compile_exact_attention_generator",
    "compile_exact_attention_generator_from_structured_executor",
    "compile_reduced_attention_generator",
    "compile_svd_reduced_attention_generator",
    "derive_attention_route_allowed_mask",
    "attention_route_probability_simplex_forward_error_bound",
    "transport_attention_coordinates",
    "validate_attention_route_probability_simplex",
]


ATTENTION_GENERATOR_KINDS = (
    "exact_full_rank",
    "reduced_coordinates",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_HASH_DOMAIN = b"fisher_graph:gemma3-attention-config:v1\0"
_ARTIFACT_HASH_DOMAIN = b"fisher_graph:gemma3-attention-artifact:v1\0"
_COORDINATE_HASH_DOMAIN = b"fisher_graph:gemma3-attention-coordinate:v1\0"
_ROUTE_FACTOR_HASH_DOMAIN = b"fisher_graph:gemma3-attention-route-factor:v1\0"
_ROUTE_SIMPLEX_AUDIT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-attention-route-simplex-audit:v1\0"
)
ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID = (
    "float32_softmax_simplex."
    "gamma32_support_plus_2_gamma64_audit_plus_ftz.v1"
)
ATTENTION_ROUTE_PROBABILITY_SIMPLEX_SUPPORT_DERIVATION_ID = (
    "query_key_valid_masks_logical_causal_optional_strict_window.v1"
)
ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT = 768
_FLOAT32_UNIT_ROUNDOFF = 2.0**-24
_FLOAT64_UNIT_ROUNDOFF = 2.0**-53
_FLOAT32_MIN_NORMAL = 2.0**-126


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def attention_route_probability_simplex_forward_error_bound(
    support_count: int,
) -> float:
    """Return the frozen forward-error bound for one stored f32 softmax row.

    The native CPU kernel stores float32 probabilities after a float32
    reduction, reciprocal, and multiply.  The audit then sums those exact
    stored bytes in float64.  For support ``k`` the bound is

    ``gamma32(k+2) + gamma64(k-1)*(1+gamma32(k+2)) + k*2^-126``.

    The final term is deliberately the float32 minimum *normal*, not the
    minimum subnormal, so the contract remains safe under flush-to-zero.
    """

    if (
        type(support_count) is not int
        or support_count < 1
        or support_count > ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT
    ):
        raise ValueError(
            "support_count must be an integer in [1, 768]"
        )
    gamma32_numerator = (support_count + 2) * _FLOAT32_UNIT_ROUNDOFF
    gamma32 = gamma32_numerator / (1.0 - gamma32_numerator)
    gamma64_numerator = (support_count - 1) * _FLOAT64_UNIT_ROUNDOFF
    gamma64 = gamma64_numerator / (1.0 - gamma64_numerator)
    return (
        gamma32
        + gamma64 * (1.0 + gamma32)
        + support_count * _FLOAT32_MIN_NORMAL
    )


@dataclass(frozen=True, slots=True)
class AttentionRouteProbabilitySimplexAudit:
    formula_id: str
    source_softmax_dtype: str
    source_softmax_device: str
    audit_accumulation_dtype: str
    valid_row_count: int
    support_minimum: int
    support_maximum: int
    maximum_absolute_mass_residual: float
    maximum_forward_error_bound: float
    minimum_error_margin: float

    def __post_init__(self) -> None:
        if self.formula_id != ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID:
            raise ValueError("probability simplex formula_id drifted")
        if self.source_softmax_dtype != "torch.float32":
            raise ValueError("source_softmax_dtype must be torch.float32")
        if self.source_softmax_device != "cpu":
            raise ValueError("source_softmax_device must be cpu")
        if self.audit_accumulation_dtype != "torch.float64":
            raise ValueError("audit_accumulation_dtype must be torch.float64")
        if type(self.valid_row_count) is not int or self.valid_row_count <= 0:
            raise ValueError("valid_row_count must be positive")
        if (
            type(self.support_minimum) is not int
            or type(self.support_maximum) is not int
            or self.support_minimum < 1
            or self.support_maximum < self.support_minimum
            or self.support_maximum
            > ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT
        ):
            raise ValueError("simplex audit support bounds are invalid")
        for label, value in (
            (
                "maximum_absolute_mass_residual",
                self.maximum_absolute_mass_residual,
            ),
            ("maximum_forward_error_bound", self.maximum_forward_error_bound),
            ("minimum_error_margin", self.minimum_error_margin),
        ):
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and nonnegative")
        if (
            self.maximum_absolute_mass_residual
            > self.maximum_forward_error_bound
        ):
            raise ValueError("simplex audit residual exceeds its bound")
        minimum_support_bound = (
            attention_route_probability_simplex_forward_error_bound(
                self.support_minimum
            )
        )
        if self.minimum_error_margin > minimum_support_bound:
            raise ValueError(
                "simplex audit error margin exceeds its minimum-support bound"
            )
        extrema_sum = (
            self.maximum_absolute_mass_residual + self.minimum_error_margin
        )
        if not minimum_support_bound <= extrema_sum <= (
            self.maximum_forward_error_bound
        ):
            raise ValueError(
                "simplex audit extrema cannot arise from the support inventory"
            )
        expected_maximum_bound = (
            attention_route_probability_simplex_forward_error_bound(
                self.support_maximum
            )
        )
        if self.maximum_forward_error_bound != expected_maximum_bound:
            raise ValueError(
                "maximum_forward_error_bound does not close to the frozen "
                "formula"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "formula_id": self.formula_id,
            "source_softmax_dtype": self.source_softmax_dtype,
            "source_softmax_device": self.source_softmax_device,
            "audit_accumulation_dtype": self.audit_accumulation_dtype,
            "valid_row_count": self.valid_row_count,
            "support_minimum": self.support_minimum,
            "support_maximum": self.support_maximum,
            "maximum_absolute_mass_residual": (
                self.maximum_absolute_mass_residual
            ),
            "maximum_forward_error_bound": self.maximum_forward_error_bound,
            "minimum_error_margin": self.minimum_error_margin,
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(
            _ROUTE_SIMPLEX_AUDIT_HASH_DOMAIN,
            self.state_dict(),
        )


def derive_attention_route_allowed_mask(
    *,
    query_valid_mask: Tensor,
    key_valid_mask: Tensor,
    query_logical_positions: Tensor,
    key_logical_positions: Tensor,
    window_size: int | None,
) -> Tensor:
    """Derive the authenticated causal route support from masks/positions."""

    for label, value in (
        ("query_valid_mask", query_valid_mask),
        ("key_valid_mask", key_valid_mask),
    ):
        if (
            not isinstance(value, Tensor)
            or value.ndim != 2
            or value.dtype != torch.bool
        ):
            raise ValueError(f"{label} must be a bool matrix")
    for label, value in (
        ("query_logical_positions", query_logical_positions),
        ("key_logical_positions", key_logical_positions),
    ):
        if (
            not isinstance(value, Tensor)
            or value.ndim != 2
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"{label} must be an integer matrix")
    if (
        query_valid_mask.shape != query_logical_positions.shape
        or key_valid_mask.shape != key_logical_positions.shape
        or query_valid_mask.shape[0] != key_valid_mask.shape[0]
    ):
        raise ValueError("route masks and logical positions do not align")
    devices = {
        query_valid_mask.device,
        key_valid_mask.device,
        query_logical_positions.device,
        key_logical_positions.device,
    }
    if len(devices) != 1:
        raise ValueError("route masks and logical positions must share a device")
    if window_size is not None and (
        type(window_size) is not int or window_size <= 0
    ):
        raise ValueError("window_size must be positive when present")
    allowed = (
        query_valid_mask[:, :, None]
        & key_valid_mask[:, None, :]
        & (
            key_logical_positions[:, None, :]
            <= query_logical_positions[:, :, None]
        )
    )
    if window_size is not None:
        allowed &= (
            key_logical_positions[:, None, :]
            > query_logical_positions[:, :, None] - window_size
        )
    return allowed


def validate_attention_route_probability_simplex(
    probabilities: Tensor,
    *,
    allowed_mask: Tensor,
    valid_row_mask: Tensor,
) -> AttentionRouteProbabilitySimplexAudit:
    """Validate exact stored float32 route probabilities without renormalizing."""

    if (
        not isinstance(probabilities, Tensor)
        or probabilities.ndim < 2
        or probabilities.dtype != torch.float32
        or probabilities.device.type != "cpu"
        or probabilities.requires_grad
        or not bool(torch.isfinite(probabilities).all())
    ):
        raise ValueError(
            "route probabilities must be finite detached CPU torch.float32"
        )
    if (
        not isinstance(allowed_mask, Tensor)
        or allowed_mask.dtype != torch.bool
        or allowed_mask.shape != probabilities.shape
        or allowed_mask.device != probabilities.device
    ):
        raise ValueError("allowed_mask must be a bool tensor matching probabilities")
    if (
        not isinstance(valid_row_mask, Tensor)
        or valid_row_mask.dtype != torch.bool
        or valid_row_mask.shape != probabilities.shape[:-1]
        or valid_row_mask.device != probabilities.device
    ):
        raise ValueError("valid_row_mask must match probability rows")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("route probabilities must lie in [0, 1]")
    if bool((probabilities.masked_select(~allowed_mask) != 0).any()):
        raise ValueError("masked route probabilities must be exactly zero")
    support_counts = allowed_mask.sum(dim=-1)
    if bool((support_counts[valid_row_mask] < 1).any()):
        raise ValueError("valid route rows must have positive support")
    if bool(
        (
            support_counts[valid_row_mask]
            > ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT
        ).any()
    ):
        raise ValueError("valid route support exceeds the frozen maximum")
    if bool((support_counts[~valid_row_mask] != 0).any()):
        raise ValueError("invalid route rows must have zero support")
    if bool((probabilities[~valid_row_mask] != 0).any()):
        raise ValueError("invalid route probabilities must be exactly zero")
    if not bool(valid_row_mask.any()):
        raise ValueError("probability simplex audit requires a valid route row")
    support = support_counts[valid_row_mask].to(torch.float64)
    gamma32_numerator = (support + 2.0) * _FLOAT32_UNIT_ROUNDOFF
    gamma32 = gamma32_numerator / (1.0 - gamma32_numerator)
    gamma64_numerator = (support - 1.0) * _FLOAT64_UNIT_ROUNDOFF
    gamma64 = gamma64_numerator / (1.0 - gamma64_numerator)
    bounds = (
        gamma32
        + gamma64 * (1.0 + gamma32)
        + support * _FLOAT32_MIN_NORMAL
    )
    masses = probabilities.to(torch.float64).sum(dim=-1)[valid_row_mask]
    residuals = (masses - 1.0).abs()
    if bool((residuals > bounds).any()):
        raise ValueError(
            "valid route probabilities exceed the frozen simplex error bound"
        )
    margins = bounds - residuals
    return AttentionRouteProbabilitySimplexAudit(
        formula_id=ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID,
        source_softmax_dtype="torch.float32",
        source_softmax_device="cpu",
        audit_accumulation_dtype="torch.float64",
        valid_row_count=int(valid_row_mask.sum()),
        support_minimum=int(support.min()),
        support_maximum=int(support.max()),
        maximum_absolute_mass_residual=float(residuals.max()),
        maximum_forward_error_bound=float(bounds.max()),
        minimum_error_margin=float(margins.min()),
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_gemma_norm(
    spec: NormalizationSpec,
    *,
    width: int,
    label: str,
) -> None:
    if (
        not isinstance(spec, NormalizationSpec)
        or spec.kind != "rms_norm"
        or spec.width != width
        or not spec.affine
        or spec.scale_parameterization != "unit_offset"
        or spec.compute_dtype != "float32"
    ):
        raise ValueError(f"{label} must be a width-matched Gemma RMSNorm")


def _clone_parameter(value: Tensor, *, label: str) -> nn.Parameter:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return nn.Parameter(
        value.detach().clone().contiguous(),
        requires_grad=False,
    )


def _optional_clone_parameter(
    value: Tensor | None,
    *,
    label: str,
) -> nn.Parameter | None:
    if value is None:
        return None
    return _clone_parameter(value, label=label)


def _tensor_payload(value: Tensor) -> bytes:
    canonical = value.detach().to(device="cpu").contiguous()
    return b"\0".join(
        (
            str(tuple(canonical.shape)).encode("ascii"),
            str(canonical.dtype).encode("ascii"),
            canonical.view(torch.uint8).numpy().tobytes(order="C"),
        )
    )


def _tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(
        _ROUTE_FACTOR_HASH_DOMAIN + _tensor_payload(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Gemma3AttentionGeneratorConfig:
    """Portable semantics for one Gemma attention mutation invocation."""

    invocation_id: str
    residual_width: int
    attention: AttentionSpec
    input_norm: NormalizationSpec
    output_norm: NormalizationSpec
    qk_norm: NormalizationSpec | None
    projection_bias: bool
    dropout: float
    logit_softcap: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("invocation_id must be nonempty")
        if type(self.residual_width) is not int or self.residual_width <= 0:
            raise ValueError("residual_width must be positive")
        if not isinstance(self.attention, AttentionSpec):
            raise TypeError("attention must be an AttentionSpec")
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
        _require_gemma_norm(
            self.input_norm,
            width=self.residual_width,
            label="input_norm",
        )
        _require_gemma_norm(
            self.output_norm,
            width=self.residual_width,
            label="output_norm",
        )
        if attention.qk_norm:
            if self.qk_norm is None:
                raise ValueError("qk_norm semantics are required")
            _require_gemma_norm(
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

    @property
    def query_width(self) -> int:
        return self.attention.query_heads * self.attention.head_dimension

    @property
    def key_value_width(self) -> int:
        return (
            self.attention.key_value_heads
            * self.attention.head_dimension
        )

    @property
    def config_sha256(self) -> str:
        return _json_sha256(_CONFIG_HASH_DOMAIN, asdict(self))

    @classmethod
    def from_structured_config(
        cls,
        invocation_id: str,
        config: StructuredTransformerLayerExecutorConfig,
    ) -> Gemma3AttentionGeneratorConfig:
        if not isinstance(config, StructuredTransformerLayerExecutorConfig):
            raise TypeError(
                "config must be StructuredTransformerLayerExecutorConfig"
            )
        semantics = config.transformer
        return cls(
            invocation_id=invocation_id,
            residual_width=config.residual_width,
            attention=config.attention,
            input_norm=semantics.attention_input_norm,
            output_norm=semantics.attention_output_norm,
            qk_norm=semantics.qk_norm,
            projection_bias=semantics.attention_projection_bias,
            dropout=float(semantics.attention_dropout),
            logit_softcap=semantics.attention_logit_softcap,
        )

    @classmethod
    def from_layer_spec(
        cls,
        layer: LayerSpec,
    ) -> Gemma3AttentionGeneratorConfig:
        structured = StructuredTransformerLayerExecutorConfig.from_layer_spec(
            layer
        )
        assert layer.transformer is not None
        return cls.from_structured_config(
            layer.transformer.stages[0].id,
            structured,
        )


@dataclass(frozen=True, slots=True)
class ExactAttentionGeneratorCoefficients:
    input_norm_weight: Tensor
    q_weight: Tensor
    k_weight: Tensor
    v_weight: Tensor
    o_weight: Tensor
    output_norm_weight: Tensor
    q_norm_weight: Tensor | None = None
    k_norm_weight: Tensor | None = None
    q_bias: Tensor | None = None
    k_bias: Tensor | None = None
    v_bias: Tensor | None = None
    o_bias: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ReducedAttentionGeneratorCoefficients:
    """Factorized Q/K/V input chart and attention-output generator."""

    input_norm_weight: Tensor
    input_basis: Tensor
    q_decoder_weight: Tensor
    k_decoder_weight: Tensor
    v_decoder_weight: Tensor
    output_encoder_weight: Tensor
    output_decoder_weight: Tensor
    output_norm_weight: Tensor
    q_norm_weight: Tensor | None = None
    k_norm_weight: Tensor | None = None
    q_bias: Tensor | None = None
    k_bias: Tensor | None = None
    v_bias: Tensor | None = None
    o_bias: Tensor | None = None


@dataclass(frozen=True, slots=True)
class AttentionCoordinatePacket:
    """Sequence-aligned coordinates that may cross generator boundaries."""

    values: Tensor
    valid_token_mask: Tensor
    coordinate_system_sha256: str
    semantic_role: str
    source_invocation_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.values, Tensor)
            or not self.values.is_floating_point()
            or self.values.ndim != 3
        ):
            raise TypeError("coordinate values must be [batch, token, rank]")
        if (
            not isinstance(self.valid_token_mask, Tensor)
            or self.valid_token_mask.dtype is not torch.bool
            or self.valid_token_mask.shape != self.values.shape[:2]
            or self.valid_token_mask.device != self.values.device
        ):
            raise ValueError("coordinate valid-token mask is inconsistent")
        _require_sha256(
            self.coordinate_system_sha256,
            label="coordinate_system_sha256",
        )
        if self.semantic_role not in (
            "normalized_input",
            "output_projection_input",
            "transported_input",
        ):
            raise ValueError("unsupported coordinate semantic_role")
        if (
            not isinstance(self.source_invocation_id, str)
            or not self.source_invocation_id
        ):
            raise ValueError("source_invocation_id must be nonempty")

    @property
    def rank(self) -> int:
        return int(self.values.shape[-1])


@dataclass(frozen=True, slots=True)
class Gemma3AttentionGeneratorReceipt:
    invocation_id: str
    generator_kind: str
    attention_kind: str
    input_coordinate_rank: int
    output_coordinate_rank: int
    logical_causal_key_pairs: int
    source_projection_call_count: int
    source_fallback_call_count: int


@dataclass(frozen=True, slots=True)
class Gemma3AttentionRouteFactorSnapshot:
    """Detached exact factors for scale-safe native-route aggregation.

    The snapshot keeps the factorization ``P`` and repeated ``V`` rather than
    materializing one residual-width vector for every ``(q, head, k)`` route.
    It is deliberately an execution sidecar: all tensors are detached CPU
    clones, and its digest binds the exact attention artifact and context.
    """

    invocation_id: str
    generator_kind: str
    attention_kind: str
    config_sha256: str
    artifact_sha256: str
    probabilities: Tensor
    repeated_values: Tensor
    output_projection_weight: Tensor
    query_valid_mask: Tensor
    key_valid_mask: Tensor
    query_logical_positions: Tensor
    key_logical_positions: Tensor
    query_head_count: int
    key_value_head_count: int
    head_dimension: int
    window_size: int | None
    rope_theta: float
    rope_scaling_kind: str
    rope_scaling_factor: float | None
    output_projection_bias_present: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("route-factor invocation_id must be nonempty")
        if self.generator_kind != "exact_full_rank":
            raise ValueError("route-factor snapshots require exact attention")
        if self.attention_kind not in {"global_causal", "sliding_causal"}:
            raise ValueError("route-factor attention_kind is invalid")
        if type(self.output_projection_bias_present) is not bool:
            raise TypeError("output_projection_bias_present must be bool")
        if self.output_projection_bias_present:
            raise ValueError(
                "factorized attention routes require a bias-free O projection"
            )
        _require_sha256(self.config_sha256, label="config_sha256")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if (
            not isinstance(self.probabilities, Tensor)
            or self.probabilities.dtype != torch.float32
            or self.probabilities.device.type != "cpu"
        ):
            raise ValueError(
                "source route probabilities must be CPU torch.float32"
            )
        tensors: dict[str, Tensor] = {}
        for label, value, ndim in (
            ("probabilities", self.probabilities, 4),
            ("repeated_values", self.repeated_values, 4),
            ("output_projection_weight", self.output_projection_weight, 2),
        ):
            if (
                not isinstance(value, Tensor)
                or value.ndim != ndim
                or not value.is_floating_point()
                or value.requires_grad
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"{label} must be a finite detached rank-{ndim} tensor"
                )
            tensors[label] = value.detach().to(
                device="cpu",
                copy=True,
            ).contiguous()
        masks: dict[str, Tensor] = {}
        positions: dict[str, Tensor] = {}
        for label, value in (
            ("query_valid_mask", self.query_valid_mask),
            ("key_valid_mask", self.key_valid_mask),
        ):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 2
                or value.dtype != torch.bool
                or value.requires_grad
            ):
                raise ValueError(f"{label} must be a detached bool matrix")
            masks[label] = value.detach().to(
                device="cpu",
                copy=True,
            ).contiguous()
        for label, value in (
            ("query_logical_positions", self.query_logical_positions),
            ("key_logical_positions", self.key_logical_positions),
        ):
            if (
                not isinstance(value, Tensor)
                or value.ndim != 2
                or value.dtype not in (torch.int32, torch.int64)
                or value.requires_grad
            ):
                raise ValueError(
                    f"{label} must be a detached integer matrix"
                )
            positions[label] = value.detach().to(
                device="cpu",
                dtype=torch.int64,
                copy=True,
            ).contiguous()
        probabilities = tensors["probabilities"]
        repeated = tensors["repeated_values"]
        output_weight = tensors["output_projection_weight"]
        query_mask = masks["query_valid_mask"]
        key_mask = masks["key_valid_mask"]
        query_positions = positions["query_logical_positions"]
        key_positions = positions["key_logical_positions"]
        for label, value in (
            ("query_head_count", self.query_head_count),
            ("key_value_head_count", self.key_value_head_count),
            ("head_dimension", self.head_dimension),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.query_head_count % self.key_value_head_count:
            raise ValueError("query heads must be divisible by key/value heads")
        batch, heads, query_width, key_width = probabilities.shape
        if (
            heads != self.query_head_count
            or repeated.shape
            != (batch, heads, key_width, self.head_dimension)
            or output_weight.shape[1]
            != self.query_head_count * self.head_dimension
            or query_mask.shape != (batch, query_width)
            or key_mask.shape != (batch, key_width)
            or query_positions.shape != query_mask.shape
            or key_positions.shape != key_mask.shape
        ):
            raise ValueError("route-factor tensor shapes do not close")
        allowed = derive_attention_route_allowed_mask(
            query_valid_mask=query_mask,
            key_valid_mask=key_mask,
            query_logical_positions=query_positions,
            key_logical_positions=key_positions,
            window_size=self.window_size,
        )
        if (
            self.attention_kind == "global_causal"
            and self.window_size is not None
        ):
            raise ValueError("global attention cannot carry a sliding window")
        if (
            self.attention_kind == "sliding_causal"
            and self.window_size is None
        ):
            raise ValueError("sliding attention requires a window")
        validate_attention_route_probability_simplex(
            probabilities,
            allowed_mask=allowed[:, None, :, :].expand_as(probabilities),
            valid_row_mask=query_mask[:, None, :].expand(
                batch,
                heads,
                query_width,
            ),
        )
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if self.rope_scaling_kind not in {"none", "linear"}:
            raise ValueError("rope_scaling_kind is unsupported")
        if self.rope_scaling_kind == "linear":
            if (
                self.rope_scaling_factor is None
                or not math.isfinite(self.rope_scaling_factor)
                or self.rope_scaling_factor <= 0
            ):
                raise ValueError("linear RoPE requires a positive scaling factor")
        elif self.rope_scaling_factor is not None:
            raise ValueError("unscaled RoPE cannot carry a scaling factor")
        for label, value in tensors.items():
            object.__setattr__(self, label, value)
        for label, value in masks.items():
            object.__setattr__(self, label, value)
        for label, value in positions.items():
            object.__setattr__(self, label, value)

    @property
    def tensor_sha256s(self) -> dict[str, str]:
        return {
            "key_logical_positions": _tensor_sha256(
                self.key_logical_positions
            ),
            "key_valid_mask": _tensor_sha256(self.key_valid_mask),
            "output_projection_weight": _tensor_sha256(
                self.output_projection_weight
            ),
            "probabilities": _tensor_sha256(self.probabilities),
            "query_logical_positions": _tensor_sha256(
                self.query_logical_positions
            ),
            "query_valid_mask": _tensor_sha256(self.query_valid_mask),
            "repeated_values": _tensor_sha256(self.repeated_values),
        }

    @property
    def probability_simplex_audit(
        self,
    ) -> AttentionRouteProbabilitySimplexAudit:
        allowed = derive_attention_route_allowed_mask(
            query_valid_mask=self.query_valid_mask,
            key_valid_mask=self.key_valid_mask,
            query_logical_positions=self.query_logical_positions,
            key_logical_positions=self.key_logical_positions,
            window_size=self.window_size,
        )
        batch, heads, query_width, _ = self.probabilities.shape
        return validate_attention_route_probability_simplex(
            self.probabilities,
            allowed_mask=allowed[:, None, :, :].expand_as(
                self.probabilities
            ),
            valid_row_mask=self.query_valid_mask[:, None, :].expand(
                batch,
                heads,
                query_width,
            ),
        )

    @property
    def sha256(self) -> str:
        simplex_audit = self.probability_simplex_audit
        return _json_sha256(
            _ROUTE_FACTOR_HASH_DOMAIN,
            {
                "artifact_sha256": self.artifact_sha256,
                "attention_kind": self.attention_kind,
                "config_sha256": self.config_sha256,
                "factorization": "probability_times_repeated_value_then_O_head_slice",
                "head_dimension": self.head_dimension,
                "invocation_id": self.invocation_id,
                "key_value_head_count": self.key_value_head_count,
                "output_projection_bias_present": (
                    self.output_projection_bias_present
                ),
                "query_head_count": self.query_head_count,
                "probability_simplex_audit_sha256": simplex_audit.sha256,
                "probability_simplex_formula_id": (
                    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID
                ),
                "probability_simplex_support_derivation_id": (
                    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_SUPPORT_DERIVATION_ID
                ),
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_scaling_kind": self.rope_scaling_kind,
                "rope_theta": float(self.rope_theta),
                "tensor_sha256s": self.tensor_sha256s,
                "window_size": self.window_size,
            },
        )

    def metadata(self) -> dict[str, object]:
        simplex_audit = self.probability_simplex_audit
        return {
            "snapshot_sha256": self.sha256,
            "invocation_id": self.invocation_id,
            "generator_kind": self.generator_kind,
            "attention_kind": self.attention_kind,
            "config_sha256": self.config_sha256,
            "artifact_sha256": self.artifact_sha256,
            "query_head_count": self.query_head_count,
            "key_value_head_count": self.key_value_head_count,
            "head_dimension": self.head_dimension,
            "output_projection_bias_present": (
                self.output_projection_bias_present
            ),
            "window_size": self.window_size,
            "rope_theta": float(self.rope_theta),
            "rope_scaling_kind": self.rope_scaling_kind,
            "rope_scaling_factor": self.rope_scaling_factor,
            "tensor_sha256s": self.tensor_sha256s,
            "probability_simplex_audit": simplex_audit.state_dict(),
            "probability_simplex_audit_sha256": simplex_audit.sha256,
            "probability_shape": list(self.probabilities.shape),
            "repeated_value_shape": list(self.repeated_values.shape),
            "full_route_vectors_materialized": False,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }


@dataclass(frozen=True, slots=True)
class Gemma3AttentionGeneratorExecution:
    normalized_input: Tensor | None
    input_coordinates: AttentionCoordinatePacket
    output_coordinates: AttentionCoordinatePacket
    operator_output: Tensor
    residual_delta: Tensor
    receipt: Gemma3AttentionGeneratorReceipt
    route_factor_snapshot: Gemma3AttentionRouteFactorSnapshot | None = None

    @property
    def delta(self) -> Tensor:
        """Residual-carrier mutation consumed by the joint graph runtime."""

        return self.residual_delta

    @property
    def emitted_generator_latents(self) -> Tensor:
        """Expose the one-chart output packet in graph transport layout."""

        return self.output_coordinates.values.unsqueeze(-2)


@dataclass(frozen=True, slots=True)
class Gemma3AttentionGeneratorParameterAccounting:
    generator_kind: str
    source_native_coefficient_count: int
    candidate_stored_coefficient_count: int
    removed_coefficient_count: int
    input_chart_coefficient_count: int
    qkv_generator_coefficient_count: int
    output_generator_coefficient_count: int
    normalization_coefficient_count: int
    bias_coefficient_count: int

    @property
    def compression_ratio(self) -> float:
        return (
            self.removed_coefficient_count
            / self.source_native_coefficient_count
        )


@dataclass(frozen=True, slots=True)
class Gemma3AttentionGeneratorLogicalAccounting:
    generator_kind: str
    valid_tokens: int
    logical_causal_key_pairs: int
    input_chart_macs: int
    qkv_projection_macs: int
    attention_score_macs: int
    attention_value_macs: int
    output_projection_macs: int

    @property
    def logical_total_macs(self) -> int:
        return (
            self.input_chart_macs
            + self.qkv_projection_macs
            + self.attention_score_macs
            + self.attention_value_macs
            + self.output_projection_macs
        )


def transport_attention_coordinates(
    packet: AttentionCoordinatePacket,
    transport_weight: Tensor,
    *,
    target_coordinate_system_sha256: str,
    target_invocation_id: str,
) -> AttentionCoordinatePacket:
    """Apply one explicit dense-linear edge between attention charts."""

    if not isinstance(packet, AttentionCoordinatePacket):
        raise TypeError("packet must be an AttentionCoordinatePacket")
    _require_sha256(
        target_coordinate_system_sha256,
        label="target_coordinate_system_sha256",
    )
    if not isinstance(target_invocation_id, str) or not target_invocation_id:
        raise ValueError("target_invocation_id must be nonempty")
    if (
        not isinstance(transport_weight, Tensor)
        or not transport_weight.is_floating_point()
        or transport_weight.ndim != 2
        or transport_weight.shape[1] != packet.rank
        or transport_weight.device != packet.values.device
        or transport_weight.dtype != packet.values.dtype
        or not bool(torch.isfinite(transport_weight).all())
    ):
        raise ValueError(
            "transport_weight must be finite [target_rank, source_rank] "
            "on the packet dtype/device"
        )
    values = F.linear(packet.values, transport_weight)
    values = values.masked_fill(
        ~packet.valid_token_mask.unsqueeze(-1),
        0,
    )
    return AttentionCoordinatePacket(
        values=values,
        valid_token_mask=packet.valid_token_mask,
        coordinate_system_sha256=target_coordinate_system_sha256,
        semantic_role="transported_input",
        source_invocation_id=target_invocation_id,
    )


class Gemma3AttentionGenerator(nn.Module):
    """Execute one exact or coordinate-reduced Gemma attention mutation."""

    def __init__(
        self,
        config: Gemma3AttentionGeneratorConfig,
        *,
        exact: ExactAttentionGeneratorCoefficients | None = None,
        reduced: ReducedAttentionGeneratorCoefficients | None = None,
        source_checkpoint_coefficients_retained: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(config, Gemma3AttentionGeneratorConfig):
            raise TypeError("config must be Gemma3AttentionGeneratorConfig")
        if (exact is None) == (reduced is None):
            raise ValueError("provide exactly one coefficient arm")
        if type(source_checkpoint_coefficients_retained) is not bool:
            raise TypeError(
                "source_checkpoint_coefficients_retained must be boolean"
            )
        if reduced is not None and source_checkpoint_coefficients_retained:
            raise ValueError(
                "reduced generators cannot claim copied checkpoint storage"
            )
        self.config = config
        self.generator_kind = (
            "exact_full_rank" if exact is not None else "reduced_coordinates"
        )
        self.source_checkpoint_coefficients_retained = (
            source_checkpoint_coefficients_retained
        )

        values = exact if exact is not None else reduced
        assert values is not None
        self.input_norm_weight = _clone_parameter(
            values.input_norm_weight,
            label="input_norm_weight",
        )
        self.output_norm_weight = _clone_parameter(
            values.output_norm_weight,
            label="output_norm_weight",
        )
        self.q_norm_weight = _optional_clone_parameter(
            values.q_norm_weight,
            label="q_norm_weight",
        )
        self.k_norm_weight = _optional_clone_parameter(
            values.k_norm_weight,
            label="k_norm_weight",
        )
        self.q_bias = _optional_clone_parameter(values.q_bias, label="q_bias")
        self.k_bias = _optional_clone_parameter(values.k_bias, label="k_bias")
        self.v_bias = _optional_clone_parameter(values.v_bias, label="v_bias")
        self.o_bias = _optional_clone_parameter(values.o_bias, label="o_bias")

        if exact is not None:
            self.input_basis = None
            self.q_weight = _clone_parameter(exact.q_weight, label="q_weight")
            self.k_weight = _clone_parameter(exact.k_weight, label="k_weight")
            self.v_weight = _clone_parameter(exact.v_weight, label="v_weight")
            self.output_encoder_weight = None
            self.output_decoder_weight = _clone_parameter(
                exact.o_weight,
                label="o_weight",
            )
            self._input_rank = config.residual_width
            self._output_rank = config.query_width
        else:
            assert reduced is not None
            self.input_basis = _clone_parameter(
                reduced.input_basis,
                label="input_basis",
            )
            self.q_weight = _clone_parameter(
                reduced.q_decoder_weight,
                label="q_decoder_weight",
            )
            self.k_weight = _clone_parameter(
                reduced.k_decoder_weight,
                label="k_decoder_weight",
            )
            self.v_weight = _clone_parameter(
                reduced.v_decoder_weight,
                label="v_decoder_weight",
            )
            self.output_encoder_weight = _clone_parameter(
                reduced.output_encoder_weight,
                label="output_encoder_weight",
            )
            self.output_decoder_weight = _clone_parameter(
                reduced.output_decoder_weight,
                label="output_decoder_weight",
            )
            self._input_rank = int(self.input_basis.shape[1])
            self._output_rank = int(self.output_encoder_weight.shape[0])

        self._validate_coefficient_shapes()
        rope = config.attention.rope
        assert rope is not None and rope.theta is not None
        inverse_frequency = 1.0 / (
            float(rope.theta)
            ** (
                torch.arange(
                    0,
                    config.attention.head_dimension,
                    2,
                    dtype=torch.float32,
                    device=self.device,
                )
                / config.attention.head_dimension
            )
        )
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def _validate_coefficient_shapes(self) -> None:
        config = self.config
        parameters = tuple(self.parameters())
        if len({value.device for value in parameters}) != 1:
            raise ValueError("attention coefficients must share one device")
        if len({value.dtype for value in parameters}) != 1:
            raise ValueError("attention coefficients must share one dtype")
        expected_norm = (config.residual_width,)
        if (
            self.input_norm_weight.shape != expected_norm
            or self.output_norm_weight.shape != expected_norm
        ):
            raise ValueError("branch normalization weights have wrong shape")
        expected_qk_norm = (config.attention.head_dimension,)
        if config.attention.qk_norm:
            if (
                self.q_norm_weight is None
                or self.k_norm_weight is None
                or self.q_norm_weight.shape != expected_qk_norm
                or self.k_norm_weight.shape != expected_qk_norm
            ):
                raise ValueError("Q/K normalization weights have wrong shape")
        elif self.q_norm_weight is not None or self.k_norm_weight is not None:
            raise ValueError("Q/K normalization weights are not permitted")

        if self.generator_kind == "exact_full_rank":
            if self.input_basis is not None or self.output_encoder_weight is not None:
                raise RuntimeError("exact generator contains reduced tensors")
            qkv_input = config.residual_width
            output_shape = (config.residual_width, config.query_width)
        else:
            assert self.input_basis is not None
            assert self.output_encoder_weight is not None
            if (
                self.input_basis.ndim != 2
                or self.input_basis.shape[0] != config.residual_width
                or not 0 < self._input_rank <= config.residual_width
                or self.output_encoder_weight.ndim != 2
                or self.output_encoder_weight.shape[1] != config.query_width
                or not 0 < self._output_rank <= min(
                    config.residual_width,
                    config.query_width,
                )
            ):
                raise ValueError("reduced coordinate bases have wrong shape")
            qkv_input = self._input_rank
            output_shape = (config.residual_width, self._output_rank)

        expected_weights = (
            (self.q_weight, (config.query_width, qkv_input), "Q"),
            (self.k_weight, (config.key_value_width, qkv_input), "K"),
            (self.v_weight, (config.key_value_width, qkv_input), "V"),
            (self.output_decoder_weight, output_shape, "O"),
        )
        for weight, shape, label in expected_weights:
            if weight.ndim != 2 or tuple(weight.shape) != shape:
                raise ValueError(f"{label} projection weight has wrong shape")

        bias_shapes = (
            (self.q_bias, config.query_width, "Q"),
            (self.k_bias, config.key_value_width, "K"),
            (self.v_bias, config.key_value_width, "V"),
            (self.o_bias, config.residual_width, "O"),
        )
        for bias, width, label in bias_shapes:
            if config.projection_bias:
                if bias is None or tuple(bias.shape) != (width,):
                    raise ValueError(f"{label} projection bias has wrong shape")
            elif bias is not None:
                raise ValueError(f"{label} projection bias is not permitted")

    def train(self, mode: bool = True) -> Gemma3AttentionGenerator:
        if type(mode) is not bool:
            raise TypeError("mode must be boolean")
        if mode:
            raise RuntimeError(
                "compiled attention generators are inference-only"
            )
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return self.q_weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.q_weight.dtype

    @property
    def input_coordinate_rank(self) -> int:
        return self._input_rank

    @property
    def output_coordinate_rank(self) -> int:
        return self._output_rank

    @property
    def hidden_width(self) -> int:
        return self.config.residual_width

    @property
    def chart_count(self) -> int:
        """Attention v1 exposes one shared Q/K/V coordinate chart."""

        return 1

    @property
    def incoming_chart_count(self) -> int:
        return 1

    @property
    def emitted_chart_count(self) -> int:
        return 1

    @property
    def incoming_generator_rank(self) -> int:
        return self.input_coordinate_rank

    @property
    def emitted_generator_rank(self) -> int:
        return self.output_coordinate_rank

    @property
    def owns_source_model_weights(self) -> bool:
        return self.source_checkpoint_coefficients_retained

    @property
    def owns_source_module(self) -> bool:
        return False

    @property
    def owns_source_fallback(self) -> bool:
        return False

    @property
    def executor_local_source_free(self) -> bool:
        return True

    @property
    def source_projection_call_count(self) -> int:
        return 0

    @property
    def supports_route_factor_capture(self) -> bool:
        return self.generator_kind == "exact_full_rank"

    @property
    def artifact_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(_ARTIFACT_HASH_DOMAIN)
        digest.update(self.config.config_sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(self.generator_kind.encode("ascii"))
        digest.update(b"\0source-checkpoint-coefficients-retained=")
        digest.update(
            str(self.source_checkpoint_coefficients_retained).encode("ascii")
        )
        for name, parameter in sorted(self.named_parameters()):
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_tensor_payload(parameter))
        return digest.hexdigest()

    def _coordinate_sha256(self, role: str) -> str:
        return _json_sha256(
            _COORDINATE_HASH_DOMAIN,
            {
                "artifact_sha256": self.artifact_sha256,
                "invocation_id": self.config.invocation_id,
                "generator_kind": self.generator_kind,
                "role": role,
                "rank": (
                    self.input_coordinate_rank
                    if role == "input"
                    else self.output_coordinate_rank
                ),
            },
        )

    @property
    def input_coordinate_system_sha256(self) -> str:
        return self._coordinate_sha256("input")

    @property
    def output_coordinate_system_sha256(self) -> str:
        return self._coordinate_sha256("output")

    @property
    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def stored_coefficient_count(self) -> int:
        return self.learned_parameter_count

    def architecture_manifest(self) -> dict[str, object]:
        attention = self.config.attention
        rope = attention.rope
        assert rope is not None
        return {
            "kind": "gemma3_attention_generator",
            "generator_kind": self.generator_kind,
            "invocation_id": self.config.invocation_id,
            "source_independent_runtime": True,
            "source_fallback": False,
            "source_checkpoint_coefficients_retained": (
                self.source_checkpoint_coefficients_retained
            ),
            "prefill_only": True,
            "input_coordinate_rank": self.input_coordinate_rank,
            "output_coordinate_rank": self.output_coordinate_rank,
            "attention_contract": {
                "kind": attention.kind,
                "query_heads": attention.query_heads,
                "key_value_heads": attention.key_value_heads,
                "head_dimension": attention.head_dimension,
                "qk_norm": attention.qk_norm,
                "rope_theta": float(rope.theta),
                "rope_scaling_kind": rope.scaling_kind,
                "rope_scaling_factor": rope.scaling_factor,
                "window_size": attention.window_size,
                "logit_softcap": self.config.logit_softcap,
            },
            "input_coordinate_system_sha256": (
                self.input_coordinate_system_sha256
            ),
            "output_coordinate_system_sha256": (
                self.output_coordinate_system_sha256
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def _validate_sequence(self, sequence: SequenceContext) -> None:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError("attention generator v1 supports prefill only")
        if sequence.cache_positions is not None:
            raise ValueError("cache-free prefill rejects cache_positions")
        if sequence.query_length != sequence.key_length:
            raise ValueError("prefill requires equal query and key lengths")
        if not torch.equal(
            sequence.logical_positions,
            sequence.key_logical_positions,
        ):
            raise ValueError("query/key logical positions must match")
        if not torch.equal(
            sequence.query_valid_mask,
            sequence.key_valid_mask,
        ):
            raise ValueError("query/key validity must match")
        if sequence.device != self.device:
            raise ValueError("sequence and generator must share a device")
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
                "packed or gapped positions require an explicit mask"
            )

    def _allowed_pairs(self, sequence: SequenceContext) -> Tensor:
        query = torch.arange(
            sequence.query_length,
            device=sequence.device,
        ).view(1, sequence.query_length, 1)
        key = torch.arange(
            sequence.key_length,
            device=sequence.device,
        ).view(1, 1, sequence.key_length)
        allowed = key <= query
        allowed = allowed & sequence.key_valid_mask.unsqueeze(1)
        allowed = allowed & sequence.query_valid_mask.unsqueeze(2)
        if self.config.attention.window_size is not None:
            allowed = allowed & (
                query - key < self.config.attention.window_size
            )
        return allowed

    @staticmethod
    def _rotate_half(values: Tensor) -> Tensor:
        half = values.shape[-1] // 2
        return torch.cat((-values[..., half:], values[..., :half]), dim=-1)

    @staticmethod
    def _repeat_key_value(values: Tensor, repeats: int) -> Tensor:
        if repeats == 1:
            return values
        batch, heads, length, width = values.shape
        return (
            values[:, :, None, :, :]
            .expand(batch, heads, repeats, length, width)
            .reshape(batch, heads * repeats, length, width)
        )

    @staticmethod
    def _rms_norm(
        values: Tensor,
        weight: Tensor,
        spec: NormalizationSpec,
    ) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True)
            + float(spec.epsilon)
        )
        return (
            normalized * (1.0 + weight.float())
        ).to(dtype=values.dtype)

    def _rotary_cos_sin(
        self,
        positions: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        rope = self.config.attention.rope
        assert rope is not None
        position_values = positions.float()
        if rope.scaling_kind == "linear":
            assert rope.scaling_factor is not None
            position_values = position_values / float(rope.scaling_factor)
        frequencies = torch.einsum(
            "bt,d->btd",
            position_values,
            self.inverse_frequency.float(),
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embedding.cos().to(dtype=dtype),
            embedding.sin().to(dtype=dtype),
        )

    def encode_input(
        self,
        residual_state: Tensor,
        sequence: SequenceContext,
    ) -> tuple[Tensor, AttentionCoordinatePacket]:
        self._validate_sequence(sequence)
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.config.residual_width,
        )
        if (
            not isinstance(residual_state, Tensor)
            or not residual_state.is_floating_point()
            or tuple(residual_state.shape) != expected
            or residual_state.device != self.device
            or residual_state.dtype != self.dtype
        ):
            raise ValueError(
                "residual_state must match sequence, width, dtype, and device"
            )
        normalized = self._rms_norm(
            residual_state,
            self.input_norm_weight,
            self.config.input_norm,
        )
        coordinates = (
            normalized
            if self.input_basis is None
            else torch.matmul(normalized, self.input_basis)
        )
        coordinates = coordinates.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        return normalized, AttentionCoordinatePacket(
            values=coordinates,
            valid_token_mask=sequence.query_valid_mask,
            coordinate_system_sha256=self.input_coordinate_system_sha256,
            semantic_role="normalized_input",
            source_invocation_id=self.config.invocation_id,
        )

    def _validate_input_packet(
        self,
        packet: AttentionCoordinatePacket,
        sequence: SequenceContext,
    ) -> None:
        self._validate_sequence(sequence)
        if not isinstance(packet, AttentionCoordinatePacket):
            raise TypeError("packet must be an AttentionCoordinatePacket")
        if (
            packet.coordinate_system_sha256
            != self.input_coordinate_system_sha256
            or packet.rank != self.input_coordinate_rank
            or packet.values.shape[:2]
            != (sequence.batch_size, sequence.query_length)
            or packet.values.device != self.device
            or packet.values.dtype != self.dtype
            or not torch.equal(
                packet.valid_token_mask,
                sequence.query_valid_mask,
            )
            or packet.semantic_role
            not in ("normalized_input", "transported_input")
        ):
            raise ValueError(
                "coordinate packet does not match this attention input chart"
            )

    def _execute_coordinates(
        self,
        packet: AttentionCoordinatePacket,
        sequence: SequenceContext,
        *,
        normalized_input: Tensor | None,
        capture_route_factors: bool = False,
    ) -> Gemma3AttentionGeneratorExecution:
        if type(capture_route_factors) is not bool:
            raise TypeError("capture_route_factors must be bool")
        if capture_route_factors and self.generator_kind != "exact_full_rank":
            raise ValueError(
                "route-factor snapshots are available only on exact attention"
            )
        if capture_route_factors and self.o_bias is not None:
            raise ValueError(
                "factorized route capture requires a bias-free O projection"
            )
        self._validate_input_packet(packet, sequence)
        attention = self.config.attention
        batch = sequence.batch_size
        length = sequence.query_length
        head_dimension = attention.head_dimension

        query = F.linear(packet.values, self.q_weight, self.q_bias).view(
            batch,
            length,
            attention.query_heads,
            head_dimension,
        ).transpose(1, 2)
        key = F.linear(packet.values, self.k_weight, self.k_bias).view(
            batch,
            length,
            attention.key_value_heads,
            head_dimension,
        ).transpose(1, 2)
        value = F.linear(packet.values, self.v_weight, self.v_bias).view(
            batch,
            length,
            attention.key_value_heads,
            head_dimension,
        ).transpose(1, 2)

        if attention.qk_norm:
            assert self.q_norm_weight is not None
            assert self.k_norm_weight is not None
            assert self.config.qk_norm is not None
            query = self._rms_norm(
                query,
                self.q_norm_weight,
                self.config.qk_norm,
            )
            key = self._rms_norm(
                key,
                self.k_norm_weight,
                self.config.qk_norm,
            )
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

        repeats = attention.query_heads // attention.key_value_heads
        repeated_key = self._repeat_key_value(key, repeats)
        repeated_value = self._repeat_key_value(value, repeats)
        scores = torch.matmul(
            query,
            repeated_key.transpose(-2, -1),
        ) * float(attention.query_scale)
        if self.config.logit_softcap is not None:
            cap = float(self.config.logit_softcap)
            scores = torch.tanh(scores / cap) * cap
        allowed = self._allowed_pairs(sequence)
        scores = scores.masked_fill(
            ~allowed.unsqueeze(1),
            torch.finfo(scores.dtype).min,
        )
        native_float32_probabilities = F.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        )
        probabilities = native_float32_probabilities.to(dtype=query.dtype)
        route_probabilities = native_float32_probabilities.masked_fill(
            ~allowed.unsqueeze(1),
            0,
        )
        context = torch.matmul(probabilities, repeated_value)
        context = context.transpose(1, 2).contiguous().view(
            batch,
            length,
            self.config.query_width,
        )

        if self.output_encoder_weight is None:
            output_coordinates = context
        else:
            output_coordinates = F.linear(
                context,
                self.output_encoder_weight,
            )
        operator_output = F.linear(
            output_coordinates,
            self.output_decoder_weight,
            self.o_bias,
        )
        residual_delta = self._rms_norm(
            operator_output,
            self.output_norm_weight,
            self.config.output_norm,
        )
        residual_delta = residual_delta.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        output_coordinates = output_coordinates.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        output_packet = AttentionCoordinatePacket(
            values=output_coordinates,
            valid_token_mask=sequence.query_valid_mask,
            coordinate_system_sha256=self.output_coordinate_system_sha256,
            semantic_role="output_projection_input",
            source_invocation_id=self.config.invocation_id,
        )
        pair_count = int(allowed.sum().item())
        route_factor_snapshot = None
        if capture_route_factors:
            if native_float32_probabilities.device.type != "cpu":
                raise ValueError(
                    "route-factor capture requires the frozen CPU softmax"
                )
            authenticated_allowed = derive_attention_route_allowed_mask(
                query_valid_mask=sequence.query_valid_mask,
                key_valid_mask=sequence.key_valid_mask,
                query_logical_positions=sequence.logical_positions,
                key_logical_positions=sequence.key_logical_positions,
                window_size=attention.window_size,
            )
            if not torch.equal(authenticated_allowed, allowed):
                raise ValueError(
                    "native dense attention support disagrees with the "
                    "authenticated logical route support"
                )
            rope = attention.rope
            assert rope is not None and rope.theta is not None
            route_factor_snapshot = Gemma3AttentionRouteFactorSnapshot(
                invocation_id=self.config.invocation_id,
                generator_kind=self.generator_kind,
                attention_kind=attention.kind,
                config_sha256=self.config.config_sha256,
                artifact_sha256=self.artifact_sha256,
                probabilities=route_probabilities.detach(),
                repeated_values=repeated_value.detach(),
                output_projection_weight=(
                    self.output_decoder_weight.detach()
                ),
                query_valid_mask=sequence.query_valid_mask.detach(),
                key_valid_mask=sequence.key_valid_mask.detach(),
                query_logical_positions=(
                    sequence.logical_positions.detach()
                ),
                key_logical_positions=(
                    sequence.key_logical_positions.detach()
                ),
                query_head_count=attention.query_heads,
                key_value_head_count=attention.key_value_heads,
                head_dimension=head_dimension,
                window_size=attention.window_size,
                rope_theta=float(rope.theta),
                rope_scaling_kind=(
                    "none"
                    if rope.scaling_kind is None
                    else rope.scaling_kind
                ),
                rope_scaling_factor=rope.scaling_factor,
                output_projection_bias_present=self.o_bias is not None,
            )
        return Gemma3AttentionGeneratorExecution(
            normalized_input=normalized_input,
            input_coordinates=packet,
            output_coordinates=output_packet,
            operator_output=operator_output,
            residual_delta=residual_delta,
            receipt=Gemma3AttentionGeneratorReceipt(
                invocation_id=self.config.invocation_id,
                generator_kind=self.generator_kind,
                attention_kind=attention.kind,
                input_coordinate_rank=self.input_coordinate_rank,
                output_coordinate_rank=self.output_coordinate_rank,
                logical_causal_key_pairs=pair_count,
                source_projection_call_count=0,
                source_fallback_call_count=0,
            ),
            route_factor_snapshot=route_factor_snapshot,
        )

    def forward_from_coordinates(
        self,
        packet: AttentionCoordinatePacket,
        sequence: SequenceContext,
        *,
        capture_route_factors: bool = False,
    ) -> Gemma3AttentionGeneratorExecution:
        return self._execute_coordinates(
            packet,
            sequence,
            normalized_input=None,
            capture_route_factors=capture_route_factors,
        )

    def execute_with_incoming(
        self,
        residual_state: Tensor,
        sequence: SequenceContext,
        incoming_generator_latents: Tensor,
        *,
        normalized_input_perturbation: Tensor | None = None,
        capture_route_factors: bool = False,
    ) -> Gemma3AttentionGeneratorExecution:
        """Execute after adding sparse cross-block messages to the input chart.

        The joint graph transport layout is ``[batch, token, chart, rank]``.
        Attention v1 has one shared incoming Q/K/V chart, while its emitted
        rank may differ because the output projection has a separate
        factorization.  Endpoint-specific edge matrices bridge those ranks.
        """

        normalized, packet = self.encode_input(residual_state, sequence)
        if normalized_input_perturbation is not None:
            if (
                not isinstance(normalized_input_perturbation, Tensor)
                or normalized_input_perturbation.shape != normalized.shape
                or normalized_input_perturbation.dtype != normalized.dtype
                or normalized_input_perturbation.device != normalized.device
                or not bool(torch.isfinite(normalized_input_perturbation).all())
            ):
                raise ValueError(
                    "normalized input perturbation must match attention input"
                )
            normalized = normalized + normalized_input_perturbation.masked_fill(
                ~sequence.query_valid_mask.unsqueeze(-1),
                0,
            )
            coordinates = (
                normalized
                if self.input_basis is None
                else torch.matmul(normalized, self.input_basis)
            ).masked_fill(
                ~sequence.query_valid_mask.unsqueeze(-1),
                0,
            )
            packet = AttentionCoordinatePacket(
                values=coordinates,
                valid_token_mask=sequence.query_valid_mask,
                coordinate_system_sha256=self.input_coordinate_system_sha256,
                semantic_role="normalized_input",
                source_invocation_id=self.config.invocation_id,
            )
        expected = (
            sequence.batch_size,
            sequence.query_length,
            self.chart_count,
            self.incoming_generator_rank,
        )
        if (
            not isinstance(incoming_generator_latents, Tensor)
            or tuple(incoming_generator_latents.shape) != expected
            or incoming_generator_latents.dtype != packet.values.dtype
            or incoming_generator_latents.device != packet.values.device
            or not bool(torch.isfinite(incoming_generator_latents).all())
        ):
            raise ValueError(
                "incoming attention latents must match the one-chart input"
            )
        combined = packet.values + incoming_generator_latents.squeeze(-2)
        combined = combined.masked_fill(
            ~sequence.query_valid_mask.unsqueeze(-1),
            0,
        )
        transported = AttentionCoordinatePacket(
            values=combined,
            valid_token_mask=sequence.query_valid_mask,
            coordinate_system_sha256=self.input_coordinate_system_sha256,
            semantic_role="transported_input",
            source_invocation_id=self.config.invocation_id,
        )
        return self._execute_coordinates(
            transported,
            sequence,
            normalized_input=normalized,
            capture_route_factors=capture_route_factors,
        )

    def forward(
        self,
        residual_state: Tensor,
        sequence: SequenceContext,
        *,
        capture_route_factors: bool = False,
    ) -> Gemma3AttentionGeneratorExecution:
        normalized, packet = self.encode_input(residual_state, sequence)
        return self._execute_coordinates(
            packet,
            sequence,
            normalized_input=normalized,
            capture_route_factors=capture_route_factors,
        )

    def forward_delta(
        self,
        residual_state: Tensor,
        sequence: SequenceContext,
    ) -> Tensor:
        return self(residual_state, sequence).residual_delta

    def parameter_accounting(
        self,
    ) -> Gemma3AttentionGeneratorParameterAccounting:
        config = self.config
        bias_count = (
            config.query_width
            + 2 * config.key_value_width
            + config.residual_width
            if config.projection_bias
            else 0
        )
        normalization_count = 2 * config.residual_width + (
            2 * config.attention.head_dimension
            if config.attention.qk_norm
            else 0
        )
        source_qkv = config.residual_width * (
            config.query_width + 2 * config.key_value_width
        )
        source_output = config.query_width * config.residual_width
        source = source_qkv + source_output + normalization_count + bias_count
        if self.generator_kind == "exact_full_rank":
            input_chart = 0
            candidate_qkv = source_qkv
            candidate_output = source_output
        else:
            input_chart = config.residual_width * self.input_coordinate_rank
            candidate_qkv = self.input_coordinate_rank * (
                config.query_width + 2 * config.key_value_width
            )
            candidate_output = self.output_coordinate_rank * (
                config.query_width + config.residual_width
            )
        candidate = (
            input_chart
            + candidate_qkv
            + candidate_output
            + normalization_count
            + bias_count
        )
        if candidate != self.stored_coefficient_count:
            raise RuntimeError("attention parameter ledger does not close")
        return Gemma3AttentionGeneratorParameterAccounting(
            generator_kind=self.generator_kind,
            source_native_coefficient_count=source,
            candidate_stored_coefficient_count=candidate,
            removed_coefficient_count=source - candidate,
            input_chart_coefficient_count=input_chart,
            qkv_generator_coefficient_count=candidate_qkv,
            output_generator_coefficient_count=candidate_output,
            normalization_coefficient_count=normalization_count,
            bias_coefficient_count=bias_count,
        )

    def logical_accounting(
        self,
        sequence: SequenceContext,
        *,
        input_coordinates_supplied: bool = False,
    ) -> Gemma3AttentionGeneratorLogicalAccounting:
        self._validate_sequence(sequence)
        if type(input_coordinates_supplied) is not bool:
            raise TypeError("input_coordinates_supplied must be boolean")
        valid_tokens = int(sequence.query_valid_mask.sum().item())
        pairs = int(self._allowed_pairs(sequence).sum().item())
        config = self.config
        input_macs = (
            0
            if self.generator_kind == "exact_full_rank"
            or input_coordinates_supplied
            else valid_tokens
            * config.residual_width
            * self.input_coordinate_rank
        )
        qkv_macs = valid_tokens * self.input_coordinate_rank * (
            config.query_width + 2 * config.key_value_width
        )
        score_macs = (
            pairs
            * config.attention.query_heads
            * config.attention.head_dimension
        )
        value_macs = score_macs
        output_macs = (
            valid_tokens * config.query_width * config.residual_width
            if self.generator_kind == "exact_full_rank"
            else valid_tokens
            * self.output_coordinate_rank
            * (config.query_width + config.residual_width)
        )
        return Gemma3AttentionGeneratorLogicalAccounting(
            generator_kind=self.generator_kind,
            valid_tokens=valid_tokens,
            logical_causal_key_pairs=pairs,
            input_chart_macs=input_macs,
            qkv_projection_macs=qkv_macs,
            attention_score_macs=score_macs,
            attention_value_macs=value_macs,
            output_projection_macs=output_macs,
        )


def compile_exact_attention_generator(
    config: Gemma3AttentionGeneratorConfig,
    coefficients: ExactAttentionGeneratorCoefficients,
    *,
    source_checkpoint_coefficients_retained: bool = False,
) -> Gemma3AttentionGenerator:
    if not isinstance(coefficients, ExactAttentionGeneratorCoefficients):
        raise TypeError(
            "coefficients must be ExactAttentionGeneratorCoefficients"
        )
    return Gemma3AttentionGenerator(
        config,
        exact=coefficients,
        source_checkpoint_coefficients_retained=(
            source_checkpoint_coefficients_retained
        ),
    )


def compile_reduced_attention_generator(
    config: Gemma3AttentionGeneratorConfig,
    coefficients: ReducedAttentionGeneratorCoefficients,
) -> Gemma3AttentionGenerator:
    """Compile caller-fitted Fisher/activation coordinates for one event."""

    if not isinstance(coefficients, ReducedAttentionGeneratorCoefficients):
        raise TypeError(
            "coefficients must be ReducedAttentionGeneratorCoefficients"
        )
    return Gemma3AttentionGenerator(config, reduced=coefficients)


def compile_exact_attention_generator_from_structured_executor(
    executor: StructuredTransformerLayerExecutor,
    *,
    invocation_id: str,
) -> Gemma3AttentionGenerator:
    """Clone an exact attention generator from a source-free layer control."""

    if not isinstance(executor, StructuredTransformerLayerExecutor):
        raise TypeError("executor must be StructuredTransformerLayerExecutor")
    config = Gemma3AttentionGeneratorConfig.from_structured_config(
        invocation_id,
        executor.config,
    )
    q_norm = getattr(executor.attention.q_norm, "weight", None)
    k_norm = getattr(executor.attention.k_norm, "weight", None)
    return compile_exact_attention_generator(
        config,
        ExactAttentionGeneratorCoefficients(
            input_norm_weight=executor.attention_input_norm.weight,
            q_weight=executor.attention.q_proj.weight,
            k_weight=executor.attention.k_proj.weight,
            v_weight=executor.attention.v_proj.weight,
            o_weight=executor.attention.o_proj.weight,
            output_norm_weight=executor.attention_output_norm.weight,
            q_norm_weight=q_norm,
            k_norm_weight=k_norm,
            q_bias=executor.attention.q_proj.bias,
            k_bias=executor.attention.k_proj.bias,
            v_bias=executor.attention.v_proj.bias,
            o_bias=executor.attention.o_proj.bias,
        ),
        source_checkpoint_coefficients_retained=(
            executor.owns_source_model_weights
        ),
    )


def compile_svd_reduced_attention_generator(
    exact: Gemma3AttentionGenerator,
    *,
    input_rank: int,
    output_rank: int,
) -> Gemma3AttentionGenerator:
    """Create the deterministic weight-SVD reduced control from an exact arm.

    Activation/Fisher chart fitting can supply the same reduced coefficient
    schema directly.  This helper is intentionally only a mechanical capacity
    control and does not claim that weight SVD is the selected compiler fit.
    """

    if (
        not isinstance(exact, Gemma3AttentionGenerator)
        or exact.generator_kind != "exact_full_rank"
    ):
        raise TypeError("exact must be an exact_full_rank generator")
    config = exact.config
    if (
        type(input_rank) is not int
        or not 0 < input_rank <= config.residual_width
        or type(output_rank) is not int
        or not 0 < output_rank <= min(
            config.residual_width,
            config.query_width,
        )
    ):
        raise ValueError("requested attention ranks are outside capacity")

    working_dtype = torch.float64
    working_device = torch.device("cpu")
    qkv = torch.cat(
        (exact.q_weight, exact.k_weight, exact.v_weight),
        dim=0,
    ).detach().to(device=working_device, dtype=working_dtype)
    _, _, input_vh = torch.linalg.svd(qkv, full_matrices=False)
    input_basis = input_vh[:input_rank].transpose(0, 1).contiguous()
    q_decoder = (
        exact.q_weight.detach().to(working_device, working_dtype)
        @ input_basis
    )
    k_decoder = (
        exact.k_weight.detach().to(working_device, working_dtype)
        @ input_basis
    )
    v_decoder = (
        exact.v_weight.detach().to(working_device, working_dtype)
        @ input_basis
    )

    output = exact.output_decoder_weight.detach().to(
        device=working_device,
        dtype=working_dtype,
    )
    output_u, output_s, output_vh = torch.linalg.svd(
        output,
        full_matrices=False,
    )
    root_s = output_s[:output_rank].sqrt()
    output_encoder = root_s.unsqueeze(1) * output_vh[:output_rank]
    output_decoder = output_u[:, :output_rank] * root_s.unsqueeze(0)
    target_dtype = exact.dtype
    target_device = exact.device
    return compile_reduced_attention_generator(
        config,
        ReducedAttentionGeneratorCoefficients(
            input_norm_weight=exact.input_norm_weight,
            input_basis=input_basis.to(
                device=target_device,
                dtype=target_dtype,
            ),
            q_decoder_weight=q_decoder.to(
                device=target_device,
                dtype=target_dtype,
            ),
            k_decoder_weight=k_decoder.to(
                device=target_device,
                dtype=target_dtype,
            ),
            v_decoder_weight=v_decoder.to(
                device=target_device,
                dtype=target_dtype,
            ),
            output_encoder_weight=output_encoder.to(
                device=target_device,
                dtype=target_dtype,
            ),
            output_decoder_weight=output_decoder.to(
                device=target_device,
                dtype=target_dtype,
            ),
            output_norm_weight=exact.output_norm_weight,
            q_norm_weight=exact.q_norm_weight,
            k_norm_weight=exact.k_norm_weight,
            q_bias=exact.q_bias,
            k_bias=exact.k_bias,
            v_bias=exact.v_bias,
            o_bias=exact.o_bias,
        ),
    )
