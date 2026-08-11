"""Committed-API capture bridge for parts-first Gemma 3 supermodes.

This module converts the Gemma adapter's native activation/score-gradient rows
into scalar responses for the physical parts catalog:

``response(part, row) = <part contribution, d score / d operator output>``.

It uses only activation sites and ``iter_activation_score_gradient_rows`` that
exist in the committed adapter API.  In particular:

* ``layer.0.input`` gives tied lexical-column responses at ingress;
* ``attention.context`` is the routed per-head input to ``o_proj``;
* ``mlp.down_input`` is the activated gate-times-up channel vector;
* native attention/MLP operator-output sites supply independent references and
  their score gradients; and
* ``final_norm`` supplies the exact readout-input adjoint, including any final
  logit softcap in the suffix.

Only responses inside one boundary are additive.  The returned 38 blocks stay
in execution order and are never summed across layers.  The production factory
binds the pinned live Gemma adapter; a tensor-only fixture factory exercises
the same response builder without loading a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Final

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, ScoreObjective
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    _source_lexical_modules,
)
from .gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    ATTENTION_HEAD_DIMENSION,
    GEMMA3_LAYER_COUNT,
    HIDDEN_WIDTH,
    MLP_CHANNEL_COUNT,
    SOURCE_UNIQUE_PARAMETER_COUNT,
    VOCABULARY_SIZE,
)
from .streaming_analysis import (
    ActivationScoreGradientRows,
    iter_activation_score_gradient_rows,
)


__all__ = [
    "CaptureRowKey",
    "Gemma3PartsFirstCaptureBinding",
    "NativePartResponseReceipt",
    "PartsFirstAtomResponseBatch",
    "PartsFirstResponseBlock",
    "PartsFirstShape",
    "bind_parts_first_fixture_surfaces",
    "bind_parts_first_gemma_adapter",
    "build_parts_first_atom_response_rows",
    "iter_parts_first_atom_response_batches",
    "parts_first_capture_site_ids",
]


PARTS_FIRST_CAPTURE_ABI: Final = (
    "fisher_graph.gemma3_parts_first_atom_capture.v1"
)
_HASH_DOMAIN = b"fisher_graph:gemma3-parts-first-atom-capture:v1\0"
_ROW_DOMAIN = b"fisher_graph:gemma3-parts-first-row-catalog:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:gemma3-parts-first-tensor:v1\0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCTION_SOURCE_FINGERPRINT = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_PRODUCTION_EXECUTION_FINGERPRINT = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
_OPERATOR_ATOL = 1.0e-4
_OPERATOR_RTOL = 1.0e-5
_RESPONSE_ATOL = 2.0e-4
_RESPONSE_RTOL = 2.0e-5


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("tensor hash input must be a Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(canonical.dtype), "shape": list(canonical.shape)}
        )
    )
    digest.update(b"\0")
    digest.update(
        canonical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


def _finite_float_tensor(
    value: object,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.dtype not in (torch.float32, torch.float64)
        or (ndim is not None and value.ndim != ndim)
        or not value.numel()
        or not bool(torch.isfinite(value).all())
    ):
        rank = "" if ndim is None else f" rank-{ndim}"
        raise ValueError(
            f"{label} must be a nonempty finite float32/float64{rank} tensor"
        )
    return value


def _cpu_clone(value: Tensor) -> Tensor:
    checked = _finite_float_tensor(value, label="captured tensor")
    return checked.detach().to(device="cpu").contiguous().clone()


def _close_metrics(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float, bool]:
    actual64 = actual.to(torch.float64)
    expected64 = expected.to(torch.float64)
    if actual64.shape != expected64.shape:
        raise ValueError("closure tensors must have identical shapes")
    error = actual64 - expected64
    max_abs = float(error.abs().max().item())
    limit = float(atol + rtol * float(expected64.abs().max().item()))
    scale = expected64.square().mean().sqrt().clamp_min(float(atol))
    nrmse = float(error.square().mean().sqrt().div(scale).item())
    # The mixed absolute/relative bound is the closure policy.  NRMSE remains
    # useful evidence, but it is ill-conditioned when a contraction cancels
    # near zero: an otherwise accepted float32 reconstruction error can then
    # look arbitrarily large relative to the tiny reference score.
    return max_abs, limit, nrmse, bool(max_abs <= limit)


@dataclass(frozen=True, slots=True)
class PartsFirstShape:
    layer_count: int
    hidden_width: int
    vocabulary_size: int
    query_head_count: int
    head_dimension: int
    mlp_width: int

    def __post_init__(self) -> None:
        for label, value in (
            ("layer_count", self.layer_count),
            ("hidden_width", self.hidden_width),
            ("vocabulary_size", self.vocabulary_size),
            ("query_head_count", self.query_head_count),
            ("head_dimension", self.head_dimension),
            ("mlp_width", self.mlp_width),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.layer_count != GEMMA3_LAYER_COUNT:
            raise ValueError("parts-first capture requires Gemma's 18 layers")
        if self.query_head_count * self.head_dimension <= 0:
            raise ValueError("attention head geometry is invalid")

    @property
    def lexical_atom_count(self) -> int:
        return self.hidden_width

    @property
    def attention_atom_count(self) -> int:
        return self.layer_count * self.query_head_count

    @property
    def mlp_atom_count(self) -> int:
        return self.layer_count * self.mlp_width

    @property
    def physical_atom_count(self) -> int:
        return (
            self.lexical_atom_count
            + self.attention_atom_count
            + self.mlp_atom_count
        )

    def attention_global_ordinal(self, layer: int, head: int) -> int:
        return self.lexical_atom_count + layer * self.query_head_count + head

    def mlp_global_ordinal(self, layer: int, channel: int) -> int:
        return (
            self.lexical_atom_count
            + self.attention_atom_count
            + layer * self.mlp_width
            + channel
        )

    def state_dict(self) -> dict[str, int]:
        return {
            "layer_count": self.layer_count,
            "hidden_width": self.hidden_width,
            "vocabulary_size": self.vocabulary_size,
            "query_head_count": self.query_head_count,
            "head_dimension": self.head_dimension,
            "mlp_width": self.mlp_width,
            "physical_atom_count": self.physical_atom_count,
        }


PINNED_PARTS_FIRST_SHAPE = PartsFirstShape(
    layer_count=GEMMA3_LAYER_COUNT,
    hidden_width=HIDDEN_WIDTH,
    vocabulary_size=VOCABULARY_SIZE,
    query_head_count=ATTENTION_HEAD_COUNT,
    head_dimension=ATTENTION_HEAD_DIMENSION,
    mlp_width=MLP_CHANNEL_COUNT,
)


@dataclass(frozen=True, slots=True)
class CaptureRowKey:
    example_id: str
    row_ordinal: int
    logical_position: int
    context_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be nonempty")
        for label, value in (
            ("row_ordinal", self.row_ordinal),
            ("logical_position", self.logical_position),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be nonnegative")
        _require_sha256(
            self.context_fingerprint_sha256,
            label="context_fingerprint_sha256",
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "row_ordinal": self.row_ordinal,
            "logical_position": self.logical_position,
            "context_fingerprint_sha256": self.context_fingerprint_sha256,
        }


def _row_catalog_sha256(rows: Sequence[CaptureRowKey]) -> str:
    return _domain_sha256(_ROW_DOMAIN, [row.state_dict() for row in rows])


@dataclass(frozen=True, slots=True)
class Gemma3PartsFirstCaptureBinding:
    """Physical parameter slices and adapter lineage for response capture."""

    adapter: Gemma3CausalLMAdapter | None = field(repr=False)
    shape: PartsFirstShape
    binding_kind: str
    source_model_fingerprint: str
    execution_fingerprint: str
    tied_weight: Tensor = field(repr=False)
    embed_scale: float
    attention_query_weights: tuple[Tensor, ...] = field(repr=False)
    attention_output_weights: tuple[Tensor, ...] = field(repr=False)
    mlp_gate_weights: tuple[Tensor, ...] = field(repr=False)
    mlp_up_weights: tuple[Tensor, ...] = field(repr=False)
    mlp_down_weights: tuple[Tensor, ...] = field(repr=False)
    binding_sha256: str

    def __post_init__(self) -> None:
        if self.binding_kind not in {"pinned_gemma_adapter", "fixture_surfaces"}:
            raise ValueError("binding_kind is invalid")
        if self.binding_kind == "pinned_gemma_adapter" and not isinstance(
            self.adapter,
            Gemma3CausalLMAdapter,
        ):
            raise TypeError("production binding requires a Gemma adapter")
        if self.binding_kind == "fixture_surfaces" and self.adapter is not None:
            raise ValueError("fixture surfaces cannot retain a live adapter")
        _require_sha256(
            self.source_model_fingerprint,
            label="source_model_fingerprint",
        )
        _require_sha256(
            self.execution_fingerprint,
            label="execution_fingerprint",
        )
        tied = _finite_float_tensor(
            self.tied_weight,
            label="tied_weight",
            ndim=2,
        )
        if tied.shape != (
            self.shape.vocabulary_size,
            self.shape.hidden_width,
        ):
            raise ValueError("tied weight does not match capture geometry")
        if (
            isinstance(self.embed_scale, bool)
            or not isinstance(self.embed_scale, (int, float))
            or not math.isfinite(float(self.embed_scale))
            or float(self.embed_scale) <= 0
        ):
            raise ValueError("embed_scale must be finite and positive")
        for label, values in (
            ("attention query", self.attention_query_weights),
            ("attention output", self.attention_output_weights),
            ("MLP gate", self.mlp_gate_weights),
            ("MLP up", self.mlp_up_weights),
            ("MLP down", self.mlp_down_weights),
        ):
            if len(values) != self.shape.layer_count:
                raise ValueError(f"{label}-weight inventory drifted")
        for weight in self.attention_query_weights:
            checked = _finite_float_tensor(
                weight,
                label="attention query weight",
                ndim=2,
            )
            if checked.shape != (
                self.shape.query_head_count * self.shape.head_dimension,
                self.shape.hidden_width,
            ):
                raise ValueError("attention query weight geometry drifted")
        for weight in self.attention_output_weights:
            checked = _finite_float_tensor(
                weight,
                label="attention output weight",
                ndim=2,
            )
            if checked.shape != (
                self.shape.hidden_width,
                self.shape.query_head_count * self.shape.head_dimension,
            ):
                raise ValueError("attention output weight geometry drifted")
        for label, values in (
            ("MLP gate", self.mlp_gate_weights),
            ("MLP up", self.mlp_up_weights),
        ):
            for weight in values:
                checked = _finite_float_tensor(
                    weight,
                    label=f"{label} weight",
                    ndim=2,
                )
                if checked.shape != (
                    self.shape.mlp_width,
                    self.shape.hidden_width,
                ):
                    raise ValueError(f"{label} weight geometry drifted")
        for weight in self.mlp_down_weights:
            checked = _finite_float_tensor(
                weight,
                label="MLP down weight",
                ndim=2,
            )
            if checked.shape != (
                self.shape.hidden_width,
                self.shape.mlp_width,
            ):
                raise ValueError("MLP down weight geometry drifted")
        if self.binding_sha256 != _binding_sha256(
            shape=self.shape,
            binding_kind=self.binding_kind,
            source_model_fingerprint=self.source_model_fingerprint,
            execution_fingerprint=self.execution_fingerprint,
            tied_weight=tied,
            embed_scale=float(self.embed_scale),
            attention_query_weights=self.attention_query_weights,
            attention_output_weights=self.attention_output_weights,
            mlp_gate_weights=self.mlp_gate_weights,
            mlp_up_weights=self.mlp_up_weights,
            mlp_down_weights=self.mlp_down_weights,
        ):
            raise ValueError("capture binding hash does not close")


def _coefficient_manifest_sha256(
    *,
    binding_kind: str,
    source_model_fingerprint: str,
    shape: PartsFirstShape,
    tied_weight: Tensor,
    attention_query_weights: Sequence[Tensor],
    attention_output_weights: Sequence[Tensor],
    mlp_gate_weights: Sequence[Tensor],
    mlp_up_weights: Sequence[Tensor],
    mlp_down_weights: Sequence[Tensor],
) -> str:
    # The pinned model fingerprint is already a content digest over model
    # state.  Paths and shapes bind the exact slices without copying or
    # re-hashing the 167M-scalar tied matrix on every streamed sequence.
    fixture_tensor_sha256s: dict[str, object] | None = None
    if binding_kind == "fixture_surfaces":
        fixture_tensor_sha256s = {
            "tied_weight": _tensor_sha256(tied_weight),
            "attention_query_weights": [
                _tensor_sha256(value) for value in attention_query_weights
            ],
            "attention_output_weights": [
                _tensor_sha256(value) for value in attention_output_weights
            ],
            "mlp_gate_weights": [
                _tensor_sha256(value) for value in mlp_gate_weights
            ],
            "mlp_up_weights": [
                _tensor_sha256(value) for value in mlp_up_weights
            ],
            "mlp_down_weights": [
                _tensor_sha256(value) for value in mlp_down_weights
            ],
        }
    return _domain_sha256(
        _HASH_DOMAIN + b"coefficient-manifest\0",
        {
            "binding_kind": binding_kind,
            "source_model_fingerprint": source_model_fingerprint,
            "shape": shape.state_dict(),
            "tied": {
                "path": "model.embed_tokens.weight",
                "shape": list(tied_weight.shape),
                "physical_occurrences": ["ingress", "raw_readout"],
                "storage_counted_once": True,
            },
            "attention": [
                {
                    "layer": layer,
                    "q_path": f"model.layers.{layer}.self_attn.q_proj.weight",
                    "q_shape": list(attention_query_weights[layer].shape),
                    "o_path": f"model.layers.{layer}.self_attn.o_proj.weight",
                    "o_shape": list(weight.shape),
                }
                for layer, weight in enumerate(attention_output_weights)
            ],
            "mlp": [
                {
                    "layer": layer,
                    "gate_path": f"model.layers.{layer}.mlp.gate_proj.weight",
                    "gate_shape": list(mlp_gate_weights[layer].shape),
                    "up_path": f"model.layers.{layer}.mlp.up_proj.weight",
                    "up_shape": list(mlp_up_weights[layer].shape),
                    "down_path": f"model.layers.{layer}.mlp.down_proj.weight",
                    "down_shape": list(weight.shape),
                }
                for layer, weight in enumerate(mlp_down_weights)
            ],
            "fixture_tensor_sha256s": fixture_tensor_sha256s,
        },
    )


def _binding_sha256(
    *,
    shape: PartsFirstShape,
    binding_kind: str,
    source_model_fingerprint: str,
    execution_fingerprint: str,
    tied_weight: Tensor,
    embed_scale: float,
    attention_query_weights: Sequence[Tensor],
    attention_output_weights: Sequence[Tensor],
    mlp_gate_weights: Sequence[Tensor],
    mlp_up_weights: Sequence[Tensor],
    mlp_down_weights: Sequence[Tensor],
) -> str:
    return _domain_sha256(
        _HASH_DOMAIN + b"binding\0",
        {
            "abi": PARTS_FIRST_CAPTURE_ABI,
            "binding_kind": binding_kind,
            "source_model_fingerprint": source_model_fingerprint,
            "execution_fingerprint": execution_fingerprint,
            "shape": shape.state_dict(),
            "embed_scale": embed_scale,
            "coefficient_manifest_sha256": _coefficient_manifest_sha256(
                binding_kind=binding_kind,
                source_model_fingerprint=source_model_fingerprint,
                shape=shape,
                tied_weight=tied_weight,
                attention_query_weights=attention_query_weights,
                attention_output_weights=attention_output_weights,
                mlp_gate_weights=mlp_gate_weights,
                mlp_up_weights=mlp_up_weights,
                mlp_down_weights=mlp_down_weights,
            ),
        },
    )


def _make_binding(
    *,
    adapter: Gemma3CausalLMAdapter | None,
    shape: PartsFirstShape,
    binding_kind: str,
    source_model_fingerprint: str,
    execution_fingerprint: str,
    tied_weight: Tensor,
    embed_scale: float,
    attention_query_weights: Sequence[Tensor],
    attention_output_weights: Sequence[Tensor],
    mlp_gate_weights: Sequence[Tensor],
    mlp_up_weights: Sequence[Tensor],
    mlp_down_weights: Sequence[Tensor],
) -> Gemma3PartsFirstCaptureBinding:
    attention_query = tuple(attention_query_weights)
    attention = tuple(attention_output_weights)
    mlp_gate = tuple(mlp_gate_weights)
    mlp_up = tuple(mlp_up_weights)
    mlp = tuple(mlp_down_weights)
    digest = _binding_sha256(
        shape=shape,
        binding_kind=binding_kind,
        source_model_fingerprint=source_model_fingerprint,
        execution_fingerprint=execution_fingerprint,
        tied_weight=tied_weight,
        embed_scale=float(embed_scale),
        attention_query_weights=attention_query,
        attention_output_weights=attention,
        mlp_gate_weights=mlp_gate,
        mlp_up_weights=mlp_up,
        mlp_down_weights=mlp,
    )
    return Gemma3PartsFirstCaptureBinding(
        adapter=adapter,
        shape=shape,
        binding_kind=binding_kind,
        source_model_fingerprint=source_model_fingerprint,
        execution_fingerprint=execution_fingerprint,
        tied_weight=tied_weight,
        embed_scale=float(embed_scale),
        attention_query_weights=attention_query,
        attention_output_weights=attention,
        mlp_gate_weights=mlp_gate,
        mlp_up_weights=mlp_up,
        mlp_down_weights=mlp,
        binding_sha256=digest,
    )


def bind_parts_first_gemma_adapter(
    adapter: Gemma3CausalLMAdapter,
) -> Gemma3PartsFirstCaptureBinding:
    """Bind the pinned live Gemma adapter and its exact physical slices."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    source_fingerprint = adapter.model_fingerprint()
    execution_fingerprint = adapter.execution_fingerprint()
    source_count = sum(parameter.numel() for parameter in adapter.module.parameters())
    if (
        source_fingerprint != _PRODUCTION_SOURCE_FINGERPRINT
        or execution_fingerprint != _PRODUCTION_EXECUTION_FINGERPRINT
        or source_count != SOURCE_UNIQUE_PARAMETER_COUNT
        or len(adapter.layers) != GEMMA3_LAYER_COUNT
        or any(layer.residual_width != HIDDEN_WIDTH for layer in adapter.layers)
    ):
        raise ValueError("adapter is not the pinned Gemma 3 source execution")
    _input_module, _output_module, tied_weight, embed_scale = (
        _source_lexical_modules(adapter)
    )
    attention_query_weights: list[Tensor] = []
    attention_output_weights: list[Tensor] = []
    mlp_gate_weights: list[Tensor] = []
    mlp_up_weights: list[Tensor] = []
    mlp_down_weights: list[Tensor] = []
    for ordinal, layer_spec in enumerate(adapter.layers):
        transformer = layer_spec.transformer
        attention = layer_spec.attention
        if (
            transformer is None
            or attention is None
            or attention.query_heads != ATTENTION_HEAD_COUNT
            or attention.head_dimension != ATTENTION_HEAD_DIMENSION
            or transformer.feed_forward.intermediate_width != MLP_CHANNEL_COUNT
        ):
            raise ValueError("adapter layer geometry drifted")
        source_layer = adapter.source_module(layer_spec.id)
        self_attention = getattr(source_layer, "self_attn", None)
        mlp = getattr(source_layer, "mlp", None)
        query_projection = getattr(self_attention, "q_proj", None)
        output_projection = getattr(self_attention, "o_proj", None)
        gate_projection = getattr(mlp, "gate_proj", None)
        up_projection = getattr(mlp, "up_proj", None)
        down_projection = getattr(mlp, "down_proj", None)
        query_weight = getattr(query_projection, "weight", None)
        output_weight = getattr(output_projection, "weight", None)
        gate_weight = getattr(gate_projection, "weight", None)
        up_weight = getattr(up_projection, "weight", None)
        down_weight = getattr(down_projection, "weight", None)
        if (
            not all(
                isinstance(value, Tensor)
                for value in (
                    query_weight,
                    output_weight,
                    gate_weight,
                    up_weight,
                    down_weight,
                )
            )
            or not isinstance(down_weight, Tensor)
            or getattr(query_projection, "bias", None) is not None
            or getattr(output_projection, "bias", None) is not None
            or getattr(gate_projection, "bias", None) is not None
            or getattr(up_projection, "bias", None) is not None
            or getattr(down_projection, "bias", None) is not None
        ):
            raise ValueError(f"layer {ordinal} lacks bias-free part seams")
        assert isinstance(query_weight, Tensor)
        assert isinstance(output_weight, Tensor)
        assert isinstance(gate_weight, Tensor)
        assert isinstance(up_weight, Tensor)
        attention_query_weights.append(query_weight)
        attention_output_weights.append(output_weight)
        mlp_gate_weights.append(gate_weight)
        mlp_up_weights.append(up_weight)
        mlp_down_weights.append(down_weight)
    return _make_binding(
        adapter=adapter,
        shape=PINNED_PARTS_FIRST_SHAPE,
        binding_kind="pinned_gemma_adapter",
        source_model_fingerprint=source_fingerprint,
        execution_fingerprint=execution_fingerprint,
        tied_weight=tied_weight,
        embed_scale=embed_scale,
        attention_query_weights=attention_query_weights,
        attention_output_weights=attention_output_weights,
        mlp_gate_weights=mlp_gate_weights,
        mlp_up_weights=mlp_up_weights,
        mlp_down_weights=mlp_down_weights,
    )


def bind_parts_first_fixture_surfaces(
    *,
    shape: PartsFirstShape,
    tied_weight: Tensor,
    embed_scale: float,
    attention_query_weights: Sequence[Tensor],
    attention_output_weights: Sequence[Tensor],
    mlp_gate_weights: Sequence[Tensor],
    mlp_up_weights: Sequence[Tensor],
    mlp_down_weights: Sequence[Tensor],
    source_model_fingerprint: str = "0" * 64,
    execution_fingerprint: str = "1" * 64,
) -> Gemma3PartsFirstCaptureBinding:
    """Build a model-free binding for fake captured activation rows."""

    if not isinstance(shape, PartsFirstShape):
        raise TypeError("shape must be a PartsFirstShape")
    _require_sha256(source_model_fingerprint, label="source_model_fingerprint")
    _require_sha256(execution_fingerprint, label="execution_fingerprint")
    return _make_binding(
        adapter=None,
        shape=shape,
        binding_kind="fixture_surfaces",
        source_model_fingerprint=source_model_fingerprint,
        execution_fingerprint=execution_fingerprint,
        tied_weight=tied_weight,
        embed_scale=embed_scale,
        attention_query_weights=attention_query_weights,
        attention_output_weights=attention_output_weights,
        mlp_gate_weights=mlp_gate_weights,
        mlp_up_weights=mlp_up_weights,
        mlp_down_weights=mlp_down_weights,
    )


@dataclass(frozen=True, slots=True)
class _LayerCaptureSites:
    layer_ordinal: int
    normalized_attention_input: str
    attention_context: str
    attention_operator_output: str
    normalized_mlp_input: str
    mlp_down_input: str
    mlp_operator_output: str


def _layer_capture_sites(
    binding: Gemma3PartsFirstCaptureBinding,
) -> tuple[_LayerCaptureSites, ...]:
    if binding.adapter is None:
        return tuple(
            _LayerCaptureSites(
                layer_ordinal=layer,
                normalized_attention_input=(
                    f"layer.{layer}.attention.normalized_input"
                ),
                attention_context=f"layer.{layer}.attention.context",
                attention_operator_output=(
                    f"layer.{layer}.attention.operator_output"
                ),
                normalized_mlp_input=f"layer.{layer}.mlp.normalized_input",
                mlp_down_input=f"layer.{layer}.mlp.down_input",
                mlp_operator_output=f"layer.{layer}.mlp.operator_output",
            )
            for layer in range(binding.shape.layer_count)
        )
    result: list[_LayerCaptureSites] = []
    for layer, spec in enumerate(binding.adapter.layers):
        transformer = spec.transformer
        if transformer is None or transformer.operator_sites is None:
            raise RuntimeError("adapter lost structured operator sites")
        attention_stage, mlp_stage = transformer.stages
        operators = transformer.operator_sites
        result.append(
            _LayerCaptureSites(
                layer_ordinal=layer,
                normalized_attention_input=attention_stage.normalized_input_site,
                attention_context=operators.attention_context,
                attention_operator_output=attention_stage.operator_output_site,
                normalized_mlp_input=mlp_stage.normalized_input_site,
                mlp_down_input=operators.feed_forward_down_input,
                mlp_operator_output=mlp_stage.operator_output_site,
            )
        )
    return tuple(result)


def _lexical_ingress_site(binding: Gemma3PartsFirstCaptureBinding) -> str:
    return (
        "layer.0.input"
        if binding.adapter is None
        else binding.adapter.layers[0].input_site
    )


def parts_first_capture_site_ids(
    binding: Gemma3PartsFirstCaptureBinding,
) -> tuple[str, ...]:
    """Return the committed adapter activation inventory for one capture."""

    if not isinstance(binding, Gemma3PartsFirstCaptureBinding):
        raise TypeError("binding must be a parts-first capture binding")
    return (
        _lexical_ingress_site(binding),
        *(
            site
            for layer in _layer_capture_sites(binding)
            for site in (
                layer.normalized_attention_input,
                layer.attention_context,
                layer.attention_operator_output,
                layer.normalized_mlp_input,
                layer.mlp_down_input,
                layer.mlp_operator_output,
            )
        ),
        "final_norm",
    )


def _context_rows(
    binding: Gemma3PartsFirstCaptureBinding,
    rows: ActivationScoreGradientRows,
    *,
    source_rows_sha256: str,
) -> tuple[CaptureRowKey, ...]:
    example_id = rows.example_id
    if example_id is None:
        raise ValueError("parts-first rows require a stable example_id")
    positions = rows.logical_positions.to(torch.int64)
    context_fingerprint = _domain_sha256(
        _HASH_DOMAIN + b"row-context\0",
        {
            "binding_sha256": binding.binding_sha256,
            "source_rows_sha256": source_rows_sha256,
            "example_id": example_id,
            "logical_positions_sha256": _tensor_sha256(positions),
            "observation_count": rows.observations,
        },
    )
    return tuple(
        CaptureRowKey(
            example_id=example_id,
            row_ordinal=ordinal,
            logical_position=int(position),
            context_fingerprint_sha256=context_fingerprint,
        )
        for ordinal, position in enumerate(positions.tolist())
    )


@dataclass(frozen=True, slots=True)
class NativePartResponseReceipt:
    seam_kind: str
    coefficient_binding_sha256: str
    operator_activation_sha256: str
    operator_gradient_sha256: str
    operator_max_abs_error: float
    operator_max_abs_limit: float
    operator_nrmse: float
    operator_closure_passed: bool
    response_max_abs_error: float
    response_max_abs_limit: float
    response_nrmse: float
    response_closure_passed: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.seam_kind not in {
            "adapter_layer0_embedding_output_columns",
            "adapter_attention_context_before_o_proj",
            "adapter_mlp_down_input_before_down_proj",
            "adapter_final_norm_readout_input_adjoint",
        }:
            raise ValueError("response seam_kind is invalid")
        for label, value in (
            ("coefficient_binding_sha256", self.coefficient_binding_sha256),
            ("operator_activation_sha256", self.operator_activation_sha256),
            ("operator_gradient_sha256", self.operator_gradient_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            _require_sha256(value, label=label)
        for label, value in (
            ("operator_max_abs_error", self.operator_max_abs_error),
            ("operator_max_abs_limit", self.operator_max_abs_limit),
            ("operator_nrmse", self.operator_nrmse),
            ("response_max_abs_error", self.response_max_abs_error),
            ("response_max_abs_limit", self.response_max_abs_limit),
            ("response_nrmse", self.response_nrmse),
        ):
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and nonnegative")
        if (
            type(self.operator_closure_passed) is not bool
            or type(self.response_closure_passed) is not bool
        ):
            raise TypeError("closure decisions must be bool")


@dataclass(frozen=True, slots=True)
class PartsFirstResponseBlock:
    event_index: int
    event_id: str
    boundary_id: str
    family_id: str
    functional_role_id: str
    layer_ordinal: int | None
    context_site_id: str
    contribution_site_id: str
    row_catalog_sha256: str
    local_part_ordinals: tuple[int, ...]
    physical_atom_ordinals: tuple[int, ...]
    context_values: Tensor
    part_responses: Tensor
    reference_scores: Tensor
    receipt: NativePartResponseReceipt
    block_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_values", _cpu_clone(self.context_values))
        object.__setattr__(self, "part_responses", _cpu_clone(self.part_responses))
        object.__setattr__(self, "reference_scores", _cpu_clone(self.reference_scores))
        self.assert_integrity()

    def assert_integrity(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be nonnegative")
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.event_id,
                self.boundary_id,
                self.family_id,
                self.functional_role_id,
                self.context_site_id,
                self.contribution_site_id,
            )
        ):
            raise ValueError("response identifiers must be nonempty")
        _require_sha256(self.row_catalog_sha256, label="row_catalog_sha256")
        if (
            self.local_part_ordinals != tuple(range(len(self.local_part_ordinals)))
            or len(self.local_part_ordinals) != len(self.physical_atom_ordinals)
            or len(set(self.physical_atom_ordinals))
            != len(self.physical_atom_ordinals)
        ):
            raise ValueError("part ordinal inventory drifted")
        context = _finite_float_tensor(
            self.context_values,
            label="context_values",
            ndim=2,
        )
        responses = _finite_float_tensor(
            self.part_responses,
            label="part_responses",
            ndim=2,
        )
        reference = _finite_float_tensor(
            self.reference_scores,
            label="reference_scores",
            ndim=1,
        )
        if (
            responses.shape[0] != context.shape[0]
            or reference.shape[0] != context.shape[0]
            or responses.shape[1] != len(self.local_part_ordinals)
        ):
            raise ValueError("response row/part axes are not aligned")
        response_metrics = _close_metrics(
            responses.to(torch.float64).sum(dim=-1),
            reference,
            atol=_RESPONSE_ATOL,
            rtol=_RESPONSE_RTOL,
        )
        expected_metrics = (
            self.receipt.response_max_abs_error,
            self.receipt.response_max_abs_limit,
            self.receipt.response_nrmse,
            self.receipt.response_closure_passed,
        )
        if response_metrics != expected_metrics:
            raise ValueError("part-response closure receipt does not close")
        if (
            not self.receipt.operator_closure_passed
            or not self.receipt.response_closure_passed
        ):
            raise ValueError("native operator/response closure failed")
        payload = _block_payload(self)
        if self.receipt.receipt_sha256 != _domain_sha256(
            _HASH_DOMAIN + b"response-receipt\0",
            payload["receipt_payload"],
        ):
            raise ValueError("response receipt hash does not close")
        if self.block_sha256 != _domain_sha256(
            _HASH_DOMAIN + b"response-block\0",
            payload,
        ):
            raise ValueError("response block hash does not close")


def _block_payload(block: PartsFirstResponseBlock) -> dict[str, object]:
    receipt_payload = {
        "seam_kind": block.receipt.seam_kind,
        "coefficient_binding_sha256": (
            block.receipt.coefficient_binding_sha256
        ),
        "operator_activation_sha256": (
            block.receipt.operator_activation_sha256
        ),
        "operator_gradient_sha256": block.receipt.operator_gradient_sha256,
        "operator_max_abs_error": block.receipt.operator_max_abs_error,
        "operator_max_abs_limit": block.receipt.operator_max_abs_limit,
        "operator_nrmse": block.receipt.operator_nrmse,
        "operator_closure_passed": block.receipt.operator_closure_passed,
        "response_max_abs_error": block.receipt.response_max_abs_error,
        "response_max_abs_limit": block.receipt.response_max_abs_limit,
        "response_nrmse": block.receipt.response_nrmse,
        "response_closure_passed": block.receipt.response_closure_passed,
    }
    return {
        "event_index": block.event_index,
        "event_id": block.event_id,
        "boundary_id": block.boundary_id,
        "family_id": block.family_id,
        "functional_role_id": block.functional_role_id,
        "layer_ordinal": block.layer_ordinal,
        "context_site_id": block.context_site_id,
        "contribution_site_id": block.contribution_site_id,
        "row_catalog_sha256": block.row_catalog_sha256,
        "local_part_ordinals": list(block.local_part_ordinals),
        "physical_atom_ordinals": list(block.physical_atom_ordinals),
        "context_values_sha256": _tensor_sha256(block.context_values),
        "part_responses_sha256": _tensor_sha256(block.part_responses),
        "reference_scores_sha256": _tensor_sha256(block.reference_scores),
        "receipt_payload": receipt_payload,
        "receipt_sha256": block.receipt.receipt_sha256,
    }


def _make_response_block(
    *,
    event_index: int,
    event_id: str,
    boundary_id: str,
    family_id: str,
    functional_role_id: str,
    layer_ordinal: int | None,
    context_site_id: str,
    contribution_site_id: str,
    row_catalog_sha256: str,
    physical_atom_ordinals: tuple[int, ...],
    context_values: Tensor,
    part_responses: Tensor,
    reference_scores: Tensor,
    seam_kind: str,
    coefficient_binding_sha256: str,
    operator_activation: Tensor,
    operator_gradient: Tensor,
    operator_metrics: tuple[float, float, float, bool],
) -> PartsFirstResponseBlock:
    context = _cpu_clone(context_values)
    responses = _cpu_clone(part_responses)
    reference = _cpu_clone(reference_scores)
    response_metrics = _close_metrics(
        responses.to(torch.float64).sum(dim=-1),
        reference,
        atol=_RESPONSE_ATOL,
        rtol=_RESPONSE_RTOL,
    )
    receipt_payload = {
        "seam_kind": seam_kind,
        "coefficient_binding_sha256": coefficient_binding_sha256,
        "operator_activation_sha256": _tensor_sha256(operator_activation),
        "operator_gradient_sha256": _tensor_sha256(operator_gradient),
        "operator_max_abs_error": operator_metrics[0],
        "operator_max_abs_limit": operator_metrics[1],
        "operator_nrmse": operator_metrics[2],
        "operator_closure_passed": operator_metrics[3],
        "response_max_abs_error": response_metrics[0],
        "response_max_abs_limit": response_metrics[1],
        "response_nrmse": response_metrics[2],
        "response_closure_passed": response_metrics[3],
    }
    receipt = NativePartResponseReceipt(
        **receipt_payload,
        receipt_sha256=_domain_sha256(
            _HASH_DOMAIN + b"response-receipt\0",
            receipt_payload,
        ),
    )
    temporary = PartsFirstResponseBlock.__new__(PartsFirstResponseBlock)
    for name, value in {
        "event_index": event_index,
        "event_id": event_id,
        "boundary_id": boundary_id,
        "family_id": family_id,
        "functional_role_id": functional_role_id,
        "layer_ordinal": layer_ordinal,
        "context_site_id": context_site_id,
        "contribution_site_id": contribution_site_id,
        "row_catalog_sha256": row_catalog_sha256,
        "local_part_ordinals": tuple(range(len(physical_atom_ordinals))),
        "physical_atom_ordinals": physical_atom_ordinals,
        "context_values": context,
        "part_responses": responses,
        "reference_scores": reference,
        "receipt": receipt,
    }.items():
        object.__setattr__(temporary, name, value)
    object.__setattr__(
        temporary,
        "block_sha256",
        _domain_sha256(
            _HASH_DOMAIN + b"response-block\0",
            _block_payload(temporary),
        ),
    )
    temporary.assert_integrity()
    return temporary


@dataclass(frozen=True, slots=True)
class PartsFirstAtomResponseBatch:
    binding_sha256: str
    source_rows_sha256: str
    shape: PartsFirstShape
    example_id: str
    loss: float
    row_keys: tuple[CaptureRowKey, ...]
    blocks: tuple[PartsFirstResponseBlock, ...]
    batch_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("binding_sha256", self.binding_sha256),
            ("source_rows_sha256", self.source_rows_sha256),
            ("batch_sha256", self.batch_sha256),
        ):
            _require_sha256(value, label=label)
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be nonempty")
        if not isinstance(self.loss, float) or not math.isfinite(self.loss):
            raise ValueError("loss must be finite")
        if not self.row_keys or len(set(self.row_keys)) != len(self.row_keys):
            raise ValueError("row keys must be nonempty and unique")
        expected_events = (0, *range(1, 2 * self.shape.layer_count + 1), 38)
        if (
            len(self.blocks) != 2 + 2 * self.shape.layer_count
            or tuple(block.event_index for block in self.blocks) != expected_events
        ):
            raise ValueError("response block schedule drifted")
        row_sha = _row_catalog_sha256(self.row_keys)
        for block in self.blocks:
            block.assert_integrity()
            if (
                block.row_catalog_sha256 != row_sha
                or block.context_values.shape[0] != len(self.row_keys)
            ):
                raise ValueError("response blocks do not share one row catalog")
        expected = _domain_sha256(
            _HASH_DOMAIN + b"response-batch\0",
            {
                "binding_sha256": self.binding_sha256,
                "source_rows_sha256": self.source_rows_sha256,
                "shape": self.shape.state_dict(),
                "example_id": self.example_id,
                "loss": self.loss,
                "row_catalog_sha256": row_sha,
                "block_sha256s": [block.block_sha256 for block in self.blocks],
            },
        )
        if self.batch_sha256 != expected:
            raise ValueError("response batch hash does not close")


def _source_rows_sha256(rows: ActivationScoreGradientRows) -> str:
    return _domain_sha256(
        _HASH_DOMAIN + b"activation-score-gradient-rows\0",
        {
            "example_id": rows.example_id,
            "loss": rows.loss,
            "logical_positions_sha256": _tensor_sha256(rows.logical_positions),
            "activation_sha256s": {
                name: _tensor_sha256(value)
                for name, value in rows.activations.items()
            },
            "score_gradient_sha256s": {
                name: _tensor_sha256(value)
                for name, value in rows.score_gradients.items()
            },
        },
    )


def _coefficient_slice_sha256(
    binding: Gemma3PartsFirstCaptureBinding,
    *,
    family: str,
    layer: int | None,
) -> str:
    return _domain_sha256(
        _HASH_DOMAIN + b"coefficient-slice\0",
        {
            "binding_sha256": binding.binding_sha256,
            "family": family,
            "layer": layer,
            "source_model_fingerprint": binding.source_model_fingerprint,
        },
    )


def build_parts_first_atom_response_rows(
    binding: Gemma3PartsFirstCaptureBinding,
    rows: ActivationScoreGradientRows,
) -> PartsFirstAtomResponseBatch:
    """Build all 38 ordered part-response blocks for one captured sequence."""

    if not isinstance(binding, Gemma3PartsFirstCaptureBinding):
        raise TypeError("binding must be a parts-first capture binding")
    if not isinstance(rows, ActivationScoreGradientRows):
        raise TypeError("rows must be ActivationScoreGradientRows")
    expected_sites = parts_first_capture_site_ids(binding)
    if tuple(rows.activations) != expected_sites or tuple(
        rows.score_gradients
    ) != expected_sites:
        raise ValueError("activation row site inventory/order drifted")
    source_sha = _source_rows_sha256(rows)
    row_keys = _context_rows(
        binding,
        rows,
        source_rows_sha256=source_sha,
    )
    row_sha = _row_catalog_sha256(row_keys)
    shape = binding.shape
    sites = _layer_capture_sites(binding)
    blocks: list[PartsFirstResponseBlock] = []

    ingress_site = _lexical_ingress_site(binding)
    ingress = rows.activations[ingress_site]
    ingress_gradient = rows.score_gradients[ingress_site]
    ingress_responses = ingress * ingress_gradient
    ingress_scores = ingress_responses.sum(dim=-1)
    blocks.append(
        _make_response_block(
            event_index=0,
            event_id="lexical.ingress",
            boundary_id="lexical_ingress",
            family_id="tied_lexical_hidden_column",
            functional_role_id="lexical_ingress",
            layer_ordinal=None,
            context_site_id=ingress_site,
            contribution_site_id=ingress_site,
            row_catalog_sha256=row_sha,
            physical_atom_ordinals=tuple(range(shape.hidden_width)),
            context_values=ingress,
            part_responses=ingress_responses,
            reference_scores=ingress_scores,
            seam_kind="adapter_layer0_embedding_output_columns",
            coefficient_binding_sha256=_coefficient_slice_sha256(
                binding,
                family="tied_lexical_hidden_column",
                layer=None,
            ),
            operator_activation=ingress,
            operator_gradient=ingress_gradient,
            operator_metrics=(0.0, _OPERATOR_ATOL, 0.0, True),
        )
    )

    for layer, site in enumerate(sites):
        attention_context = rows.activations[site.attention_context]
        attention_reference = rows.activations[site.attention_operator_output]
        attention_gradient = rows.score_gradients[
            site.attention_operator_output
        ]
        output_weight = binding.attention_output_weights[layer].detach().to(
            device="cpu",
            dtype=attention_context.dtype,
        )
        context_by_head = attention_context.reshape(
            rows.observations,
            shape.query_head_count,
            shape.head_dimension,
        )
        output_slices = output_weight.reshape(
            shape.hidden_width,
            shape.query_head_count,
            shape.head_dimension,
        ).permute(1, 2, 0)
        head_outputs = torch.einsum(
            "rhd,hdo->rho",
            context_by_head,
            output_slices,
        )
        attention_formula = head_outputs.sum(dim=1)
        attention_responses = (
            head_outputs * attention_gradient.unsqueeze(1)
        ).sum(dim=-1)
        attention_scores = (
            attention_reference * attention_gradient
        ).sum(dim=-1)
        attention_event = 1 + 2 * layer
        blocks.append(
            _make_response_block(
                event_index=attention_event,
                event_id=f"mutation.{attention_event}",
                boundary_id=f"layer.{layer}.attention",
                family_id="attention_qo_head",
                functional_role_id="attention_qo_contribution",
                layer_ordinal=layer,
                context_site_id=site.normalized_attention_input,
                contribution_site_id=site.attention_operator_output,
                row_catalog_sha256=row_sha,
                physical_atom_ordinals=tuple(
                    shape.attention_global_ordinal(layer, head)
                    for head in range(shape.query_head_count)
                ),
                context_values=rows.activations[
                    site.normalized_attention_input
                ],
                part_responses=attention_responses,
                reference_scores=attention_scores,
                seam_kind="adapter_attention_context_before_o_proj",
                coefficient_binding_sha256=_coefficient_slice_sha256(
                    binding,
                    family="attention_qo_head",
                    layer=layer,
                ),
                operator_activation=attention_reference,
                operator_gradient=attention_gradient,
                operator_metrics=_close_metrics(
                    attention_formula,
                    attention_reference,
                    atol=_OPERATOR_ATOL,
                    rtol=_OPERATOR_RTOL,
                ),
            )
        )

        down_input = rows.activations[site.mlp_down_input]
        mlp_reference = rows.activations[site.mlp_operator_output]
        mlp_gradient = rows.score_gradients[site.mlp_operator_output]
        down_weight = binding.mlp_down_weights[layer].detach().to(
            device="cpu",
            dtype=down_input.dtype,
        )
        mlp_formula = down_input @ down_weight.transpose(0, 1)
        mlp_responses = down_input * (mlp_gradient @ down_weight)
        mlp_scores = (mlp_reference * mlp_gradient).sum(dim=-1)
        mlp_event = 2 + 2 * layer
        blocks.append(
            _make_response_block(
                event_index=mlp_event,
                event_id=f"mutation.{mlp_event}",
                boundary_id=f"layer.{layer}.mlp",
                family_id="mlp_channel",
                functional_role_id="mlp_channel_contribution",
                layer_ordinal=layer,
                context_site_id=site.normalized_mlp_input,
                contribution_site_id=site.mlp_operator_output,
                row_catalog_sha256=row_sha,
                physical_atom_ordinals=tuple(
                    shape.mlp_global_ordinal(layer, channel)
                    for channel in range(shape.mlp_width)
                ),
                context_values=rows.activations[site.normalized_mlp_input],
                part_responses=mlp_responses,
                reference_scores=mlp_scores,
                seam_kind="adapter_mlp_down_input_before_down_proj",
                coefficient_binding_sha256=_coefficient_slice_sha256(
                    binding,
                    family="mlp_channel",
                    layer=layer,
                ),
                operator_activation=mlp_reference,
                operator_gradient=mlp_gradient,
                operator_metrics=_close_metrics(
                    mlp_formula,
                    mlp_reference,
                    atol=_OPERATOR_ATOL,
                    rtol=_OPERATOR_RTOL,
                ),
            )
        )

    terminal = rows.activations["final_norm"]
    terminal_gradient = rows.score_gradients["final_norm"]
    egress_responses = terminal * terminal_gradient
    egress_scores = egress_responses.sum(dim=-1)
    blocks.append(
        _make_response_block(
            event_index=38,
            event_id="lexical.egress",
            boundary_id="lexical_readout",
            family_id="tied_lexical_hidden_column",
            functional_role_id="lexical_readout",
            layer_ordinal=None,
            context_site_id="final_norm",
            contribution_site_id="lexical.raw_egress.adjoint",
            row_catalog_sha256=row_sha,
            physical_atom_ordinals=tuple(range(shape.hidden_width)),
            context_values=terminal,
            part_responses=egress_responses,
            reference_scores=egress_scores,
            seam_kind="adapter_final_norm_readout_input_adjoint",
            coefficient_binding_sha256=_coefficient_slice_sha256(
                binding,
                family="tied_lexical_hidden_column",
                layer=None,
            ),
            operator_activation=terminal,
            operator_gradient=terminal_gradient,
            operator_metrics=(0.0, _OPERATOR_ATOL, 0.0, True),
        )
    )

    ordered = tuple(blocks)
    batch_payload = {
        "binding_sha256": binding.binding_sha256,
        "source_rows_sha256": source_sha,
        "shape": shape.state_dict(),
        "example_id": rows.example_id,
        "loss": rows.loss,
        "row_catalog_sha256": row_sha,
        "block_sha256s": [block.block_sha256 for block in ordered],
    }
    assert rows.example_id is not None
    return PartsFirstAtomResponseBatch(
        binding_sha256=binding.binding_sha256,
        source_rows_sha256=source_sha,
        shape=shape,
        example_id=rows.example_id,
        loss=rows.loss,
        row_keys=row_keys,
        blocks=ordered,
        batch_sha256=_domain_sha256(
            _HASH_DOMAIN + b"response-batch\0",
            batch_payload,
        ),
    )


def iter_parts_first_atom_response_batches(
    binding: Gemma3PartsFirstCaptureBinding,
    calibration_batches: Iterable[CalibrationBatch],
    *,
    score_objective: ScoreObjective,
    accumulation_dtype: torch.dtype = torch.float64,
) -> Iterable[PartsFirstAtomResponseBatch]:
    """Run the live adapter and stream one exact response batch per sequence."""

    if not isinstance(binding, Gemma3PartsFirstCaptureBinding):
        raise TypeError("binding must be a parts-first capture binding")
    if binding.adapter is None:
        raise ValueError("live capture requires a pinned Gemma adapter binding")
    source_rows = iter_activation_score_gradient_rows(
        binding.adapter,
        calibration_batches,
        activation_names=parts_first_capture_site_ids(binding),
        score_objective=score_objective,
        leaf_activation_name=_lexical_ingress_site(binding),
        accumulation_dtype=accumulation_dtype,
    )
    return (
        build_parts_first_atom_response_rows(binding, rows)
        for rows in source_rows
    )
