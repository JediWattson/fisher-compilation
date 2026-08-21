"""Model-free Gemma chart surfaces for attention routes and lexical boundaries.

The stable-neighbor experiment compares one common target-token carrier chart
surface.  Attention additionally owns a causal source-to-target route surface;
that surface is authenticated supplemental evidence and is not silently folded
into the common distance.  The two lexical endpoints share one tied-state
lineage while retaining different native ABIs.  Lexical egress enters the
common carrier surface through an authenticated full-NLL terminal-gradient
pullback; a selected-row analytic pullback remains available for validation.

Nothing in this module loads a tokenizer or model.  Surface objects contain
detached CPU tensors, stable row identities, and hashes, but never token ids or
prompt text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import re
import struct
from typing import Final

import torch
from torch import Tensor

from .gemma3_attention_generator import (
    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID,
    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT,
    ATTENTION_ROUTE_PROBABILITY_SIMPLEX_SUPPORT_DERIVATION_ID,
    AttentionRouteProbabilitySimplexAudit,
    Gemma3AttentionRouteFactorSnapshot,
    derive_attention_route_allowed_mask,
    validate_attention_route_probability_simplex,
)


GEMMA3_CHART_MODE_SURFACE_SCHEMA: Final = (
    "fisher_graph.gemma3_chart_mode_surface.v1"
)
GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION: Final = 1
COMMON_CARRIER_COMPARISON_ABI_NAME: Final = "carrier_chart_energy.h640.v1"
PINNED_CARRIER_HIDDEN_WIDTH: Final = 640

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SURFACE_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface-tensor:v1\0"
_ROW_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface-rows:v1\0"
_ROPE_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface-rope:v1\0"
_LINEAGE_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface-lineage:v1\0"
_MAPPING_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface-mapping:v1\0"
_READOUT_DOMAIN = b"fisher_graph:gemma3-chart-mode-surface-readout:v1\0"
_FACTORIZED_ROUTE_DOMAIN = (
    b"fisher_graph:gemma3-chart-factorized-attention-route:v1\0"
)
_FACTORIZED_ROUTE_KERNEL_DOMAIN = (
    b"fisher_graph:gemma3-chart-factorized-attention-kernel:v1\0"
)
_ENERGY_ATOL = 1.0e-10
MAX_EXHAUSTIVE_ATTENTION_ROUTE_ROWS: Final = 65_536
_STREAM_AGGREGATE_BLOCK_ROWS: Final = 1_024
FACTORIZED_ATTENTION_ROUTE_KERNEL_ID: Final = (
    "factorized_attention_route_energy.p_u_g.float64_chunked.v1"
)
FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE: Final = 32
FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE: Final = 128


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
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable canonical name")
    return value


def _strict_int(value: object, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an exact integer")
    return value


def _strict_float(
    value: object,
    *,
    label: str,
    strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (strictly_positive and float(value) <= 0)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _strict_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a materialized list")
    return value


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _detached_cpu(value: Tensor, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a tensor")
    return value.detach().to(device="cpu").clone().contiguous()


def _finite_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
    nonnegative: bool = False,
) -> Tensor:
    result = _detached_cpu(value, label=label)
    if (
        result.ndim != ndim
        or not all(int(width) > 0 for width in result.shape)
        or not result.is_floating_point()
        or not bool(torch.isfinite(result).all())
        or (nonnegative and bool((result < 0).any()))
    ):
        qualifier = "nonnegative finite" if nonnegative else "finite"
        raise ValueError(f"{label} must be a {qualifier} floating {ndim}-tensor")
    return result


def _bool_tensor(value: Tensor, *, label: str, ndim: int) -> Tensor:
    result = _detached_cpu(value, label=label)
    if result.ndim != ndim or result.dtype != torch.bool:
        raise ValueError(f"{label} must be a boolean {ndim}-tensor")
    return result


def _integer_tensor(value: Tensor, *, label: str, ndim: int) -> Tensor:
    result = _detached_cpu(value, label=label)
    if result.ndim != ndim or result.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"{label} must be an integer {ndim}-tensor")
    return result.to(dtype=torch.int64)


def _tensor_receipt(value: Tensor) -> dict[str, object]:
    tensor = _detached_cpu(value, label="receipt tensor")
    byte_view = tensor.view(torch.uint8)
    payload_sha = hashlib.sha256(
        _TENSOR_DOMAIN + byte_view.numpy().tobytes(order="C")
    ).hexdigest()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "payload_sha256": payload_sha,
    }


def _tensor_sha256(value: Tensor) -> str:
    return _domain_sha256(_TENSOR_DOMAIN, _tensor_receipt(value))


def _row_catalog_sha256(rows: Sequence[object]) -> str:
    return _domain_sha256(_ROW_DOMAIN, list(rows))


def _canonical_example_ids(
    example_ids: Sequence[str],
    *,
    batch_size: int,
) -> tuple[str, ...]:
    if isinstance(example_ids, (str, bytes)) or not isinstance(
        example_ids, Sequence
    ):
        raise TypeError("example_ids must be a materialized sequence")
    result = tuple(example_ids)
    if (
        len(result) != batch_size
        or any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("example_ids must be unique nonempty batch identities")
    return result


def _canonical_context_fingerprints(
    values: Sequence[str],
    *,
    batch_size: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("context_fingerprints must be a materialized sequence")
    result = tuple(values)
    if len(result) != batch_size:
        raise ValueError("context_fingerprints must align with the batch")
    for index, value in enumerate(result):
        _require_sha256(value, label=f"context_fingerprints[{index}]")
    return result


def _validate_logical_positions(
    positions: Tensor,
    valid_mask: Tensor,
    *,
    label: str,
) -> None:
    if positions.shape != valid_mask.shape:
        raise ValueError(f"{label} shape must match its valid mask")
    if bool((positions[valid_mask] < 0).any()):
        raise ValueError(f"{label} valid positions must be nonnegative")
    for row in range(int(positions.shape[0])):
        valid = positions[row][valid_mask[row]].tolist()
        if any(left >= right for left, right in zip(valid, valid[1:])):
            raise ValueError(f"{label} valid positions must be strictly increasing")


def _normalized_energy(value: Tensor) -> Tensor:
    energy = value.to(dtype=torch.float64)
    total = energy.sum()
    if float(total) == 0.0:
        return torch.zeros_like(energy)
    return (energy / total).contiguous()


def _energy_close(left: Tensor, right: Tensor) -> bool:
    return bool(torch.allclose(left, right, rtol=1.0e-9, atol=_ENERGY_ATOL))


@dataclass(frozen=True, slots=True)
class NativeRoPESpec:
    """Exact rotary-position semantics authenticated by a route surface."""

    theta: float
    rotary_dimension: int
    scaling_kind: str
    scaling_factor: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.theta) or self.theta <= 0:
            raise ValueError("RoPE theta must be finite and positive")
        if (
            type(self.rotary_dimension) is not int
            or self.rotary_dimension <= 0
            or self.rotary_dimension % 2
        ):
            raise ValueError("RoPE rotary_dimension must be positive and even")
        _require_name(self.scaling_kind, label="RoPE scaling_kind")
        if self.scaling_factor is not None and (
            not math.isfinite(self.scaling_factor) or self.scaling_factor <= 0
        ):
            raise ValueError("RoPE scaling_factor must be finite and positive")
        if self.scaling_kind in {"none", "default"} and self.scaling_factor is not None:
            raise ValueError("unscaled RoPE cannot carry a scaling factor")

    def state_dict(self) -> dict[str, object]:
        return {
            "theta": float(self.theta),
            "rotary_dimension": self.rotary_dimension,
            "scaling_kind": self.scaling_kind,
            "scaling_factor": self.scaling_factor,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_ROPE_DOMAIN, self.state_dict())

    @classmethod
    def from_state_dict(cls, value: object) -> NativeRoPESpec:
        state = _require_exact_fields(
            value,
            {"theta", "rotary_dimension", "scaling_kind", "scaling_factor"},
            label="RoPE state",
        )
        return cls(
            theta=_strict_float(
                state["theta"], label="RoPE theta", strictly_positive=True
            ),
            rotary_dimension=_strict_int(
                state["rotary_dimension"],
                label="RoPE rotary_dimension",
                minimum=1,
            ),
            scaling_kind=_require_name(
                state["scaling_kind"], label="RoPE scaling_kind"
            ),
            scaling_factor=(
                None
                if state["scaling_factor"] is None
                else _strict_float(
                    state["scaling_factor"],
                    label="RoPE scaling_factor",
                    strictly_positive=True,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AttentionPostNormBridgeReceipt:
    """Bind additive route closure to the nonlinear post-attention boundary."""

    operator_output_sha256: str
    operator_output_score_gradient_sha256: str
    postnorm_state_sha256: str
    postnorm_config_sha256: str
    postnorm_delta_sha256: str
    execution_fingerprint_sha256: str

    def __post_init__(self) -> None:
        for label in (
            "operator_output_sha256",
            "operator_output_score_gradient_sha256",
            "postnorm_state_sha256",
            "postnorm_config_sha256",
            "postnorm_delta_sha256",
            "execution_fingerprint_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)

    def state_dict(self) -> dict[str, object]:
        return {
            "operator_output_sha256": self.operator_output_sha256,
            "operator_output_score_gradient_sha256": (
                self.operator_output_score_gradient_sha256
            ),
            "postnorm_state_sha256": self.postnorm_state_sha256,
            "postnorm_config_sha256": self.postnorm_config_sha256,
            "postnorm_delta_sha256": self.postnorm_delta_sha256,
            "execution_fingerprint_sha256": self.execution_fingerprint_sha256,
            "route_additive_closure_site": "attention.operator_output",
            "common_carrier_signature_site": "post_attention_norm.delta",
            "additive_equality_across_nonlinear_postnorm_claimed": False,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_MAPPING_DOMAIN, self.state_dict())

    @classmethod
    def from_state_dict(cls, value: object) -> AttentionPostNormBridgeReceipt:
        fields = {
            "operator_output_sha256",
            "operator_output_score_gradient_sha256",
            "postnorm_state_sha256",
            "postnorm_config_sha256",
            "postnorm_delta_sha256",
            "execution_fingerprint_sha256",
            "route_additive_closure_site",
            "common_carrier_signature_site",
            "additive_equality_across_nonlinear_postnorm_claimed",
        }
        state = _require_exact_fields(value, fields, label="attention postnorm bridge")
        if (
            state["route_additive_closure_site"] != "attention.operator_output"
            or state["common_carrier_signature_site"]
            != "post_attention_norm.delta"
            or state["additive_equality_across_nonlinear_postnorm_claimed"]
            is not False
        ):
            raise ValueError("attention postnorm bridge semantics drifted")
        result = cls(
            operator_output_sha256=_require_sha256(
                state["operator_output_sha256"], label="operator_output_sha256"
            ),
            operator_output_score_gradient_sha256=_require_sha256(
                state["operator_output_score_gradient_sha256"],
                label="operator_output_score_gradient_sha256",
            ),
            postnorm_state_sha256=_require_sha256(
                state["postnorm_state_sha256"], label="postnorm_state_sha256"
            ),
            postnorm_config_sha256=_require_sha256(
                state["postnorm_config_sha256"], label="postnorm_config_sha256"
            ),
            postnorm_delta_sha256=_require_sha256(
                state["postnorm_delta_sha256"], label="postnorm_delta_sha256"
            ),
            execution_fingerprint_sha256=_require_sha256(
                state["execution_fingerprint_sha256"],
                label="execution_fingerprint_sha256",
            ),
        )
        if dict(state) != result.state_dict():
            raise ValueError("attention postnorm bridge failed strict validation")
        return result


def build_attention_postnorm_bridge_receipt(
    *,
    operator_output: Tensor,
    operator_output_score_gradient: Tensor,
    postnorm_delta: Tensor,
    postnorm_state_sha256: str,
    postnorm_config_sha256: str,
    execution_fingerprint_sha256: str,
) -> AttentionPostNormBridgeReceipt:
    operator = _finite_float_tensor(
        operator_output,
        label="operator_output",
        ndim=3,
    )
    gradient = _finite_float_tensor(
        operator_output_score_gradient,
        label="operator_output_score_gradient",
        ndim=3,
    )
    delta = _finite_float_tensor(postnorm_delta, label="postnorm_delta", ndim=3)
    if operator.shape != gradient.shape or operator.shape != delta.shape:
        raise ValueError("attention bridge tensors must share [B,Q,H]")
    return AttentionPostNormBridgeReceipt(
        operator_output_sha256=_tensor_sha256(operator),
        operator_output_score_gradient_sha256=_tensor_sha256(gradient),
        postnorm_state_sha256=postnorm_state_sha256,
        postnorm_config_sha256=postnorm_config_sha256,
        postnorm_delta_sha256=_tensor_sha256(delta),
        execution_fingerprint_sha256=execution_fingerprint_sha256,
    )


@dataclass(frozen=True, slots=True)
class TiedLexicalStateLineage:
    """One tied parameter state viewed through two non-interchangeable ABIs."""

    tied_state_sha256: str
    vocabulary_size: int
    hidden_width: int
    embedding_scale: float

    def __post_init__(self) -> None:
        _require_sha256(self.tied_state_sha256, label="tied_state_sha256")
        if type(self.vocabulary_size) is not int or self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if self.hidden_width != PINNED_CARRIER_HIDDEN_WIDTH:
            raise ValueError("tied lexical hidden_width must be the pinned h640")
        if not math.isfinite(self.embedding_scale) or self.embedding_scale <= 0:
            raise ValueError("embedding_scale must be finite and positive")

    def state_dict(self) -> dict[str, object]:
        return {
            "tied_state_sha256": self.tied_state_sha256,
            "vocabulary_size": self.vocabulary_size,
            "hidden_width": self.hidden_width,
            "embedding_scale": float(self.embedding_scale),
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_LINEAGE_DOMAIN, self.state_dict())

    @classmethod
    def from_state_dict(cls, value: object) -> TiedLexicalStateLineage:
        state = _require_exact_fields(
            value,
            {
                "tied_state_sha256",
                "vocabulary_size",
                "hidden_width",
                "embedding_scale",
            },
            label="tied lexical lineage",
        )
        return cls(
            tied_state_sha256=_require_sha256(
                state["tied_state_sha256"], label="tied_state_sha256"
            ),
            vocabulary_size=_strict_int(
                state["vocabulary_size"], label="vocabulary_size", minimum=1
            ),
            hidden_width=_strict_int(
                state["hidden_width"], label="hidden_width", minimum=1
            ),
            embedding_scale=_strict_float(
                state["embedding_scale"],
                label="embedding_scale",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class CarrierChartSignature:
    """Aligned ``carrier value * score gradient`` comparison coordinates.

    The integer in each row key is the carrier's prediction-query position
    ``q``.  The authenticated score-objective receipt separately binds the
    next-token label position ``t = q + 1``.
    """

    mode_id: str
    native_abi_sha256: str
    comparison_abi_sha256: str
    mapping_receipt_sha256: str
    source_state_sha256: str
    execution_fingerprint_sha256: str
    score_objective_receipt_sha256: str
    row_keys: tuple[tuple[str, int], ...]
    carrier_values: Tensor
    carrier_score_gradients: Tensor
    coordinates: Tensor
    activation_energy: Tensor
    fisher_energy: Tensor
    normalized_activation_energy: Tensor
    normalized_fisher_energy: Tensor

    def __post_init__(self) -> None:
        _require_name(self.mode_id, label="mode_id")
        for label in (
            "native_abi_sha256",
            "comparison_abi_sha256",
            "mapping_receipt_sha256",
            "source_state_sha256",
            "execution_fingerprint_sha256",
            "score_objective_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if (
            not isinstance(self.row_keys, tuple)
            or not self.row_keys
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
                or type(row[1]) is not int
                or row[1] < 0
                for row in self.row_keys
            )
            or len(set(self.row_keys)) != len(self.row_keys)
        ):
            raise ValueError(
                "row_keys must be unique "
                "(example_id, prediction_query_position) pairs"
            )
        values = _finite_float_tensor(
            self.carrier_values,
            label="carrier_values",
            ndim=2,
        )
        gradients = _finite_float_tensor(
            self.carrier_score_gradients,
            label="carrier_score_gradients",
            ndim=2,
        )
        coordinates = _finite_float_tensor(
            self.coordinates,
            label="coordinates",
            ndim=2,
        )
        if (
            values.shape != gradients.shape
            or values.shape != coordinates.shape
            or values.shape[0] != len(self.row_keys)
        ):
            raise ValueError("carrier values, gradients, coordinates, and rows must align")
        if int(values.shape[1]) != PINNED_CARRIER_HIDDEN_WIDTH:
            raise ValueError("common carrier comparison requires exact h640 rows")
        expected_coordinates = values * gradients
        if not torch.equal(coordinates, expected_coordinates):
            raise ValueError("coordinates must be carrier value times score gradient")
        expected_activation = values.to(torch.float64).square().sum(dim=1)
        expected_fisher = coordinates.to(torch.float64).square().sum(dim=1)
        energies = (
            (self.activation_energy, expected_activation, "activation_energy"),
            (self.fisher_energy, expected_fisher, "fisher_energy"),
            (
                self.normalized_activation_energy,
                _normalized_energy(expected_activation),
                "normalized_activation_energy",
            ),
            (
                self.normalized_fisher_energy,
                _normalized_energy(expected_fisher),
                "normalized_fisher_energy",
            ),
        )
        for claimed, expected, label in energies:
            canonical = _finite_float_tensor(
                claimed,
                label=label,
                ndim=1,
                nonnegative=True,
            ).to(torch.float64)
            if canonical.shape != expected.shape or not _energy_close(
                canonical, expected
            ):
                raise ValueError(f"{label} is not derived from carrier coordinates")
        for source, label in (
            (self.normalized_activation_energy, "normalized activation energy"),
            (self.normalized_fisher_energy, "normalized Fisher energy"),
        ):
            value = source.to(torch.float64)
            expected_sum = 0.0 if float(value.sum()) == 0.0 else 1.0
            if not math.isclose(
                float(value.sum()), expected_sum, rel_tol=1.0e-9, abs_tol=_ENERGY_ATOL
            ):
                raise ValueError(f"{label} must sum to zero or one")
        object.__setattr__(self, "carrier_values", values)
        object.__setattr__(self, "carrier_score_gradients", gradients)
        object.__setattr__(self, "coordinates", coordinates)
        for field_name in (
            "activation_energy",
            "fisher_energy",
            "normalized_activation_energy",
            "normalized_fisher_energy",
        ):
            object.__setattr__(
                self,
                field_name,
                _detached_cpu(getattr(self, field_name), label=field_name),
            )

    @property
    def hidden_width(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def row_catalog_sha256(self) -> str:
        return _row_catalog_sha256(self.row_keys)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "mode_id": self.mode_id,
            "native_abi_sha256": self.native_abi_sha256,
            "comparison_abi_sha256": self.comparison_abi_sha256,
            "mapping_receipt_sha256": self.mapping_receipt_sha256,
            "source_state_sha256": self.source_state_sha256,
            "execution_fingerprint_sha256": self.execution_fingerprint_sha256,
            "score_objective_receipt_sha256": (
                self.score_objective_receipt_sha256
            ),
            "row_catalog_sha256": self.row_catalog_sha256,
            "row_count": len(self.row_keys),
            "hidden_width": self.hidden_width,
            "carrier_values_sha256": _tensor_sha256(self.carrier_values),
            "carrier_score_gradients_sha256": _tensor_sha256(
                self.carrier_score_gradients
            ),
            "coordinates_sha256": _tensor_sha256(self.coordinates),
            "activation_energy_sha256": _tensor_sha256(self.activation_energy),
            "fisher_energy_sha256": _tensor_sha256(self.fisher_energy),
            "normalized_activation_energy_sha256": _tensor_sha256(
                self.normalized_activation_energy
            ),
            "normalized_fisher_energy_sha256": _tensor_sha256(
                self.normalized_fisher_energy
            ),
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_SURFACE_DOMAIN, self._hash_payload())

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "schema": GEMMA3_CHART_MODE_SURFACE_SCHEMA,
            "format_version": GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION,
            "surface_kind": "common_carrier_signature",
            "comparison_abi_name": COMMON_CARRIER_COMPARISON_ABI_NAME,
            "row_keys": [list(row) for row in self.row_keys],
            "carrier_values": self.carrier_values.detach().clone(),
            "carrier_score_gradients": self.carrier_score_gradients.detach().clone(),
            "coordinates": self.coordinates.detach().clone(),
            "activation_energy": self.activation_energy.detach().clone(),
            "fisher_energy": self.fisher_energy.detach().clone(),
            "normalized_activation_energy": self.normalized_activation_energy.detach().clone(),
            "normalized_fisher_energy": self.normalized_fisher_energy.detach().clone(),
            "surface_sha256": self.sha256,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }

    @classmethod
    def from_state_dict(cls, value: object) -> CarrierChartSignature:
        fields = {
            "schema",
            "format_version",
            "surface_kind",
            "comparison_abi_name",
            "mode_id",
            "native_abi_sha256",
            "comparison_abi_sha256",
            "mapping_receipt_sha256",
            "source_state_sha256",
            "execution_fingerprint_sha256",
            "score_objective_receipt_sha256",
            "row_catalog_sha256",
            "row_count",
            "hidden_width",
            "row_keys",
            "carrier_values",
            "carrier_score_gradients",
            "coordinates",
            "activation_energy",
            "fisher_energy",
            "normalized_activation_energy",
            "normalized_fisher_energy",
            "carrier_values_sha256",
            "carrier_score_gradients_sha256",
            "coordinates_sha256",
            "activation_energy_sha256",
            "fisher_energy_sha256",
            "normalized_activation_energy_sha256",
            "normalized_fisher_energy_sha256",
            "surface_sha256",
            "contains_token_ids",
            "contains_prompt_text",
        }
        state = _require_exact_fields(value, fields, label="carrier signature")
        if (
            state["schema"] != GEMMA3_CHART_MODE_SURFACE_SCHEMA
            or type(state["format_version"]) is not int
            or state["format_version"] != GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION
            or state["surface_kind"] != "common_carrier_signature"
            or state["comparison_abi_name"] != COMMON_CARRIER_COMPARISON_ABI_NAME
            or state["contains_token_ids"] is not False
            or state["contains_prompt_text"] is not False
        ):
            raise ValueError("carrier signature contract drifted")
        raw_rows = state["row_keys"]
        if not isinstance(raw_rows, list):
            raise ValueError("carrier row_keys must be a list")
        rows_list: list[tuple[str, int]] = []
        for index, row in enumerate(raw_rows):
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
            ):
                raise ValueError(f"carrier row_keys[{index}] is malformed")
            rows_list.append(
                (
                    row[0],
                    _strict_int(
                        row[1],
                        label=f"carrier row_keys[{index}] position",
                        minimum=0,
                    ),
                )
            )
        rows = tuple(rows_list)
        _strict_int(state["row_count"], label="row_count", minimum=1)
        _strict_int(state["hidden_width"], label="hidden_width", minimum=1)
        result = cls(
            mode_id=_require_name(state["mode_id"], label="mode_id"),
            native_abi_sha256=_require_sha256(
                state["native_abi_sha256"], label="native_abi_sha256"
            ),
            comparison_abi_sha256=_require_sha256(
                state["comparison_abi_sha256"], label="comparison_abi_sha256"
            ),
            mapping_receipt_sha256=_require_sha256(
                state["mapping_receipt_sha256"], label="mapping_receipt_sha256"
            ),
            source_state_sha256=_require_sha256(
                state["source_state_sha256"], label="source_state_sha256"
            ),
            execution_fingerprint_sha256=_require_sha256(
                state["execution_fingerprint_sha256"],
                label="execution_fingerprint_sha256",
            ),
            score_objective_receipt_sha256=_require_sha256(
                state["score_objective_receipt_sha256"],
                label="score_objective_receipt_sha256",
            ),
            row_keys=rows,
            carrier_values=state["carrier_values"],  # type: ignore[arg-type]
            carrier_score_gradients=state[  # type: ignore[arg-type]
                "carrier_score_gradients"
            ],
            coordinates=state["coordinates"],  # type: ignore[arg-type]
            activation_energy=state["activation_energy"],  # type: ignore[arg-type]
            fisher_energy=state["fisher_energy"],  # type: ignore[arg-type]
            normalized_activation_energy=state[  # type: ignore[arg-type]
                "normalized_activation_energy"
            ],
            normalized_fisher_energy=state[  # type: ignore[arg-type]
                "normalized_fisher_energy"
            ],
        )
        expected = result.state_dict()
        for key in fields - {
            "carrier_values",
            "carrier_score_gradients",
            "coordinates",
            "activation_energy",
            "fisher_energy",
            "normalized_activation_energy",
            "normalized_fisher_energy",
        }:
            if state[key] != expected[key]:
                raise ValueError(f"carrier signature {key} failed strict validation")
        return result


def _build_carrier_signature(
    *,
    mode_id: str,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    mapping_receipt_sha256: str,
    source_state_sha256: str,
    execution_fingerprint_sha256: str,
    score_objective_receipt_sha256: str,
    example_ids: tuple[str, ...],
    valid_mask: Tensor,
    logical_positions: Tensor,
    dense_carrier_values: Tensor,
    dense_carrier_score_gradients: Tensor,
) -> CarrierChartSignature:
    valid = _bool_tensor(valid_mask, label="valid_mask", ndim=2)
    positions = _integer_tensor(
        logical_positions,
        label="logical_positions",
        ndim=2,
    )
    _validate_logical_positions(positions, valid, label="logical_positions")
    values = _finite_float_tensor(
        dense_carrier_values,
        label="dense_carrier_values",
        ndim=3,
    )
    gradients = _finite_float_tensor(
        dense_carrier_score_gradients,
        label="dense_carrier_score_gradients",
        ndim=3,
    )
    if values.shape != gradients.shape or values.shape[:2] != valid.shape:
        raise ValueError("dense carrier tensors must align with valid_mask")
    rows = tuple(
        (example_ids[batch], int(positions[batch, token]))
        for batch, token in valid.nonzero(as_tuple=False).tolist()
    )
    flat_values = values[valid].contiguous()
    flat_gradients = gradients[valid].contiguous()
    flat_coordinates = (flat_values * flat_gradients).contiguous()
    coordinate64 = flat_coordinates.to(torch.float64)
    activation = flat_values.to(torch.float64).square().sum(dim=1).contiguous()
    fisher_energy = coordinate64.square().sum(dim=1).contiguous()
    return CarrierChartSignature(
        mode_id=mode_id,
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        mapping_receipt_sha256=mapping_receipt_sha256,
        source_state_sha256=source_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=score_objective_receipt_sha256,
        row_keys=rows,
        carrier_values=flat_values,
        carrier_score_gradients=flat_gradients,
        coordinates=flat_coordinates,
        activation_energy=activation,
        fisher_energy=fisher_energy,
        normalized_activation_energy=_normalized_energy(activation),
        normalized_fisher_energy=_normalized_energy(fisher_energy),
    )


@dataclass(frozen=True, slots=True)
class MutationCarrierChartSurface:
    """Generic attention/MLP carrier with static and panel receipts split."""

    carrier: CarrierChartSignature
    context_site_id: str
    contribution_site_id: str
    context_values: Tensor
    context_catalog_sha256: str
    static_comparison_adapter_receipt_sha256: str
    panel_mapping_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.carrier, CarrierChartSignature):
            raise TypeError("carrier must be a CarrierChartSignature")
        _require_name(self.context_site_id, label="context_site_id")
        _require_name(self.contribution_site_id, label="contribution_site_id")
        if self.context_site_id == self.contribution_site_id:
            raise ValueError("mutation context and contribution sites must differ")
        _require_sha256(
            self.context_catalog_sha256,
            label="context_catalog_sha256",
        )
        _require_sha256(
            self.static_comparison_adapter_receipt_sha256,
            label="static_comparison_adapter_receipt_sha256",
        )
        _require_sha256(
            self.panel_mapping_receipt_sha256,
            label="panel_mapping_receipt_sha256",
        )
        if (
            self.carrier.mapping_receipt_sha256
            != self.static_comparison_adapter_receipt_sha256
        ):
            raise ValueError(
                "carrier mapping must bind the static comparison adapter"
            )
        contexts = _finite_float_tensor(
            self.context_values,
            label="context_values",
            ndim=2,
        )
        if contexts.shape != self.carrier.carrier_values.shape:
            raise ValueError("mutation context rows do not align with carrier rows")
        object.__setattr__(self, "context_values", contexts)

    @property
    def sha256(self) -> str:
        return _domain_sha256(
            _SURFACE_DOMAIN,
            {
                "carrier_sha256": self.carrier.sha256,
                "context_catalog_sha256": self.context_catalog_sha256,
                "context_site_id": self.context_site_id,
                "context_values_sha256": _tensor_sha256(
                    self.context_values
                ),
                "contribution_site_id": self.contribution_site_id,
                "panel_mapping_receipt_sha256": (
                    self.panel_mapping_receipt_sha256
                ),
                "static_comparison_adapter_receipt_sha256": (
                    self.static_comparison_adapter_receipt_sha256
                ),
                "surface_kind": "generic_mutation_carrier",
            },
        )


def build_mutation_carrier_signature(
    *,
    mode_id: str,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    static_comparison_adapter_receipt_sha256: str,
    source_state_sha256: str,
    execution_fingerprint_sha256: str,
    score_objective_receipt_sha256: str,
    example_ids: Sequence[str],
    valid_mask: Tensor,
    logical_positions: Tensor,
    mutation_output_values: Tensor,
    mutation_output_score_gradients: Tensor,
) -> CarrierChartSignature:
    """Build a body-mode carrier bound to its static ABI adapter receipt."""

    if not mode_id.startswith("mode.mutation."):
        raise ValueError("generic mutation mode_id must name a mutation")
    _require_sha256(
        static_comparison_adapter_receipt_sha256,
        label="static_comparison_adapter_receipt_sha256",
    )
    valid = _bool_tensor(valid_mask, label="valid_mask", ndim=2)
    examples = _canonical_example_ids(
        example_ids,
        batch_size=int(valid.shape[0]),
    )
    return _build_carrier_signature(
        mode_id=mode_id,
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        mapping_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        source_state_sha256=source_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=score_objective_receipt_sha256,
        example_ids=examples,
        valid_mask=valid,
        logical_positions=logical_positions,
        dense_carrier_values=mutation_output_values,
        dense_carrier_score_gradients=mutation_output_score_gradients,
    )


def build_mutation_carrier_chart_surface(
    *,
    mode_id: str,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    static_comparison_adapter_receipt_sha256: str,
    source_state_sha256: str,
    execution_fingerprint_sha256: str,
    score_objective_receipt_sha256: str,
    context_site_id: str,
    contribution_site_id: str,
    example_ids: Sequence[str],
    context_fingerprints: Sequence[str],
    valid_mask: Tensor,
    logical_positions: Tensor,
    mutation_context_values: Tensor,
    mutation_output_values: Tensor,
    mutation_output_score_gradients: Tensor,
) -> MutationCarrierChartSurface:
    """Build one generic body surface with panel lineage kept non-static."""

    _require_name(context_site_id, label="context_site_id")
    _require_name(contribution_site_id, label="contribution_site_id")
    if context_site_id == contribution_site_id:
        raise ValueError("mutation context and contribution sites must differ")
    valid = _bool_tensor(valid_mask, label="valid_mask", ndim=2)
    examples = _canonical_example_ids(
        example_ids,
        batch_size=int(valid.shape[0]),
    )
    contexts = _canonical_context_fingerprints(
        context_fingerprints,
        batch_size=int(valid.shape[0]),
    )
    positions = _integer_tensor(
        logical_positions,
        label="logical_positions",
        ndim=2,
    )
    _validate_logical_positions(positions, valid, label="logical_positions")
    dense_context = _finite_float_tensor(
        mutation_context_values,
        label="mutation_context_values",
        ndim=3,
    )
    if dense_context.shape[:2] != valid.shape:
        raise ValueError("mutation contexts do not align with valid rows")
    carrier = build_mutation_carrier_signature(
        mode_id=mode_id,
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        static_comparison_adapter_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        source_state_sha256=source_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=score_objective_receipt_sha256,
        example_ids=examples,
        valid_mask=valid,
        logical_positions=positions,
        mutation_output_values=mutation_output_values,
        mutation_output_score_gradients=(
            mutation_output_score_gradients
        ),
    )
    flat_context = dense_context[valid].contiguous()
    context_catalog_sha256 = _domain_sha256(
        _ROW_DOMAIN,
        {
            "context_fingerprints": list(contexts),
            "context_site_id": context_site_id,
            "context_values_sha256": _tensor_sha256(flat_context),
            "example_ids": list(examples),
            "logical_positions_sha256": _tensor_sha256(positions),
            "row_keys": [list(row) for row in carrier.row_keys],
        },
    )
    panel_mapping_receipt_sha256 = _domain_sha256(
        _MAPPING_DOMAIN,
        {
            "carrier_value_sha256": _tensor_sha256(
                carrier.carrier_values
            ),
            "context_catalog_sha256": context_catalog_sha256,
            "context_site_id": context_site_id,
            "contribution_site_id": contribution_site_id,
            "execution_fingerprint_sha256": (
                execution_fingerprint_sha256
            ),
            "mapping": "mutation_output_to_common_h640_carrier",
            "mode_id": mode_id,
            "score_gradient_sha256": _tensor_sha256(
                carrier.carrier_score_gradients
            ),
            "score_objective_receipt_sha256": (
                score_objective_receipt_sha256
            ),
            "static_comparison_adapter_receipt_sha256": (
                static_comparison_adapter_receipt_sha256
            ),
        },
    )
    return MutationCarrierChartSurface(
        carrier=carrier,
        context_site_id=context_site_id,
        contribution_site_id=contribution_site_id,
        context_values=flat_context,
        context_catalog_sha256=context_catalog_sha256,
        static_comparison_adapter_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        panel_mapping_receipt_sha256=panel_mapping_receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class AttentionNativeContextBinding:
    """The exact causal mask and RoPE positions underlying route coordinates."""

    example_ids: tuple[str, ...]
    context_fingerprints: tuple[str, ...]
    query_valid_mask: Tensor
    key_valid_mask: Tensor
    query_logical_positions: Tensor
    key_logical_positions: Tensor
    native_route_mask: Tensor
    query_head_count: int
    window_size: int | None
    rope: NativeRoPESpec

    def __post_init__(self) -> None:
        query_valid = _bool_tensor(
            self.query_valid_mask,
            label="query_valid_mask",
            ndim=2,
        )
        key_valid = _bool_tensor(
            self.key_valid_mask,
            label="key_valid_mask",
            ndim=2,
        )
        if query_valid.shape[0] != key_valid.shape[0]:
            raise ValueError("query and key masks must share a batch")
        _canonical_example_ids(self.example_ids, batch_size=int(query_valid.shape[0]))
        _canonical_context_fingerprints(
            self.context_fingerprints,
            batch_size=int(query_valid.shape[0]),
        )
        query_positions = _integer_tensor(
            self.query_logical_positions,
            label="query_logical_positions",
            ndim=2,
        )
        key_positions = _integer_tensor(
            self.key_logical_positions,
            label="key_logical_positions",
            ndim=2,
        )
        _validate_logical_positions(
            query_positions,
            query_valid,
            label="query_logical_positions",
        )
        _validate_logical_positions(
            key_positions,
            key_valid,
            label="key_logical_positions",
        )
        if type(self.query_head_count) is not int or self.query_head_count <= 0:
            raise ValueError("query_head_count must be positive")
        if self.window_size is not None and (
            type(self.window_size) is not int or self.window_size <= 0
        ):
            raise ValueError("window_size must be positive when present")
        native = _bool_tensor(
            self.native_route_mask,
            label="native_route_mask",
            ndim=4,
        )
        expected_shape = (
            int(query_valid.shape[0]),
            int(query_valid.shape[1]),
            self.query_head_count,
            int(key_valid.shape[1]),
        )
        if tuple(native.shape) != expected_shape:
            raise ValueError("native_route_mask must be [B,Q,query_head,K]")
        allowed = (
            query_valid[:, :, None, None]
            & key_valid[:, None, None, :]
            & (
                key_positions[:, None, None, :]
                <= query_positions[:, :, None, None]
            )
        ).expand(expected_shape)
        if self.window_size is not None:
            allowed = allowed & (
                key_positions[:, None, None, :]
                > query_positions[:, :, None, None] - self.window_size
            ).expand(expected_shape)
        if not torch.equal(native, allowed):
            raise ValueError(
                "native_route_mask must exactly match padding, causal, and window semantics"
            )
        valid_targets = query_valid.nonzero(as_tuple=False)
        for batch, target in valid_targets.tolist():
            if not bool(native[batch, target].any()):
                raise ValueError("every valid target must own at least one native route")
        object.__setattr__(self, "query_valid_mask", query_valid)
        object.__setattr__(self, "key_valid_mask", key_valid)
        object.__setattr__(self, "query_logical_positions", query_positions)
        object.__setattr__(self, "key_logical_positions", key_positions)
        object.__setattr__(self, "native_route_mask", native)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "example_ids": list(self.example_ids),
            "context_fingerprints": list(self.context_fingerprints),
            "query_valid_mask_sha256": _tensor_sha256(self.query_valid_mask),
            "key_valid_mask_sha256": _tensor_sha256(self.key_valid_mask),
            "query_logical_positions_sha256": _tensor_sha256(
                self.query_logical_positions
            ),
            "key_logical_positions_sha256": _tensor_sha256(
                self.key_logical_positions
            ),
            "native_route_mask_sha256": _tensor_sha256(self.native_route_mask),
            "query_head_count": self.query_head_count,
            "window_size": self.window_size,
            "rope": self.rope.state_dict(),
            "rope_sha256": self.rope.sha256,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_ROPE_DOMAIN, self._hash_payload())

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "query_valid_mask": self.query_valid_mask.detach().clone(),
            "key_valid_mask": self.key_valid_mask.detach().clone(),
            "query_logical_positions": self.query_logical_positions.detach().clone(),
            "key_logical_positions": self.key_logical_positions.detach().clone(),
            "native_route_mask": self.native_route_mask.detach().clone(),
            "binding_sha256": self.sha256,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }

    @classmethod
    def from_state_dict(cls, value: object) -> AttentionNativeContextBinding:
        fields = {
            "example_ids",
            "context_fingerprints",
            "query_valid_mask",
            "key_valid_mask",
            "query_logical_positions",
            "key_logical_positions",
            "native_route_mask",
            "query_valid_mask_sha256",
            "key_valid_mask_sha256",
            "query_logical_positions_sha256",
            "key_logical_positions_sha256",
            "native_route_mask_sha256",
            "query_head_count",
            "window_size",
            "rope",
            "rope_sha256",
            "binding_sha256",
            "contains_token_ids",
            "contains_prompt_text",
        }
        state = _require_exact_fields(value, fields, label="attention context binding")
        if state["contains_token_ids"] is not False or state["contains_prompt_text"] is not False:
            raise ValueError("attention context binding leaked prohibited content")
        result = cls(
            example_ids=tuple(
                _require_name(item, label="example_id")
                for item in _strict_list(state["example_ids"], label="example_ids")
            ),
            context_fingerprints=tuple(
                _require_sha256(item, label="context_fingerprint")
                for item in _strict_list(
                    state["context_fingerprints"], label="context_fingerprints"
                )
            ),
            query_valid_mask=state["query_valid_mask"],  # type: ignore[arg-type]
            key_valid_mask=state["key_valid_mask"],  # type: ignore[arg-type]
            query_logical_positions=state["query_logical_positions"],  # type: ignore[arg-type]
            key_logical_positions=state["key_logical_positions"],  # type: ignore[arg-type]
            native_route_mask=state["native_route_mask"],  # type: ignore[arg-type]
            query_head_count=_strict_int(
                state["query_head_count"], label="query_head_count", minimum=1
            ),
            window_size=(
                None
                if state["window_size"] is None
                else _strict_int(
                    state["window_size"], label="window_size", minimum=1
                )
            ),
            rope=NativeRoPESpec.from_state_dict(state["rope"]),
        )
        expected = result.state_dict()
        for key in fields - {
            "query_valid_mask",
            "key_valid_mask",
            "query_logical_positions",
            "key_logical_positions",
            "native_route_mask",
        }:
            if state[key] != expected[key]:
                raise ValueError(f"attention context {key} failed strict validation")
        return result


@dataclass(frozen=True, slots=True)
class AttentionRouteStreamContext:
    """Bounded native-mask/RoPE description without a ``[B,Q,H,K]`` tensor."""

    example_ids: tuple[str, ...]
    context_fingerprints: tuple[str, ...]
    query_valid_mask: Tensor
    key_valid_mask: Tensor
    query_logical_positions: Tensor
    key_logical_positions: Tensor
    query_head_count: int
    window_size: int | None
    rope: NativeRoPESpec

    def __post_init__(self) -> None:
        query_valid = _bool_tensor(
            self.query_valid_mask,
            label="query_valid_mask",
            ndim=2,
        )
        key_valid = _bool_tensor(
            self.key_valid_mask,
            label="key_valid_mask",
            ndim=2,
        )
        if query_valid.shape[0] != key_valid.shape[0]:
            raise ValueError("query and key masks must share a batch")
        _canonical_example_ids(self.example_ids, batch_size=int(query_valid.shape[0]))
        _canonical_context_fingerprints(
            self.context_fingerprints,
            batch_size=int(query_valid.shape[0]),
        )
        query_positions = _integer_tensor(
            self.query_logical_positions,
            label="query_logical_positions",
            ndim=2,
        )
        key_positions = _integer_tensor(
            self.key_logical_positions,
            label="key_logical_positions",
            ndim=2,
        )
        _validate_logical_positions(
            query_positions,
            query_valid,
            label="query_logical_positions",
        )
        _validate_logical_positions(
            key_positions,
            key_valid,
            label="key_logical_positions",
        )
        if type(self.query_head_count) is not int or self.query_head_count <= 0:
            raise ValueError("query_head_count must be positive")
        if self.window_size is not None and (
            type(self.window_size) is not int or self.window_size <= 0
        ):
            raise ValueError("window_size must be positive when present")
        if self.expected_route_count <= 0:
            raise ValueError("attention route context must contain a native route")
        object.__setattr__(self, "query_valid_mask", query_valid)
        object.__setattr__(self, "key_valid_mask", key_valid)
        object.__setattr__(self, "query_logical_positions", query_positions)
        object.__setattr__(self, "key_logical_positions", key_positions)

    def route_is_allowed(self, batch: int, target: int, head: int, source: int) -> bool:
        if not (
            0 <= batch < int(self.query_valid_mask.shape[0])
            and 0 <= target < int(self.query_valid_mask.shape[1])
            and 0 <= head < self.query_head_count
            and 0 <= source < int(self.key_valid_mask.shape[1])
            and bool(self.query_valid_mask[batch, target])
            and bool(self.key_valid_mask[batch, source])
        ):
            return False
        target_position = int(self.query_logical_positions[batch, target])
        source_position = int(self.key_logical_positions[batch, source])
        return source_position <= target_position and (
            self.window_size is None
            or source_position > target_position - self.window_size
        )

    @property
    def expected_route_count(self) -> int:
        count = 0
        for batch in range(int(self.query_valid_mask.shape[0])):
            valid_sources = self.key_logical_positions[batch][
                self.key_valid_mask[batch]
            ].tolist()
            for target in self.query_logical_positions[batch][
                self.query_valid_mask[batch]
            ].tolist():
                count += self.query_head_count * sum(
                    source <= target
                    and (
                        self.window_size is None
                        or source > target - self.window_size
                    )
                    for source in valid_sources
                )
        return count

    def _hash_payload(self) -> dict[str, object]:
        return {
            "example_ids": list(self.example_ids),
            "context_fingerprints": list(self.context_fingerprints),
            "query_valid_mask_sha256": _tensor_sha256(self.query_valid_mask),
            "key_valid_mask_sha256": _tensor_sha256(self.key_valid_mask),
            "query_logical_positions_sha256": _tensor_sha256(
                self.query_logical_positions
            ),
            "key_logical_positions_sha256": _tensor_sha256(
                self.key_logical_positions
            ),
            "query_head_count": self.query_head_count,
            "window_size": self.window_size,
            "rope_sha256": self.rope.sha256,
            "expected_route_count": self.expected_route_count,
            "route_semantics": "padding_and_causal_and_optional_strict_window",
            "native_route_mask_materialized": False,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_ROPE_DOMAIN, self._hash_payload())


@dataclass(frozen=True, slots=True)
class AttentionRouteAggregateReceipt:
    """Bounded sufficient statistics from a canonical stream of native routes."""

    carrier_sha256: str
    context_binding_sha256: str
    postnorm_bridge_sha256: str
    block_ordinal: int
    route_count: int
    expected_route_count: int
    route_row_catalog_sha256: str
    route_coordinate_stream_sha256: str
    route_energy_stream_sha256: str
    distance_bin_upper_bounds: tuple[int, ...]
    distance_bin_counts: tuple[int, ...]
    distance_bin_intrinsic_fisher_energy: tuple[float, ...]
    head_route_counts: tuple[int, ...]
    head_intrinsic_fisher_energy: tuple[float, ...]
    head_coordinate_mean: Tensor
    head_coordinate_second_moment: Tensor
    intrinsic_route_fisher_energy: float
    allocated_target_fisher_energy: float
    target_fisher_energy: float
    positive_target_count: int
    zero_target_count: int
    target_count: int
    operator_output_sha256: str
    reconstructed_operator_output_sha256: str
    operator_output_closure_max_abs: float
    operator_output_closure_max_abs_limit: float
    operator_output_closure_nrmse: float
    top_mass_route_count: int
    top_mass_route_catalog_sha256: str
    top_intrinsic_fisher_energy_fraction: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.carrier_sha256, "carrier_sha256"),
            (self.context_binding_sha256, "context_binding_sha256"),
            (self.postnorm_bridge_sha256, "postnorm_bridge_sha256"),
            (self.route_row_catalog_sha256, "route_row_catalog_sha256"),
            (
                self.route_coordinate_stream_sha256,
                "route_coordinate_stream_sha256",
            ),
            (self.route_energy_stream_sha256, "route_energy_stream_sha256"),
            (self.top_mass_route_catalog_sha256, "top_mass_route_catalog_sha256"),
            (self.operator_output_sha256, "operator_output_sha256"),
            (
                self.reconstructed_operator_output_sha256,
                "reconstructed_operator_output_sha256",
            ),
        ):
            _require_sha256(value, label=label)
        if type(self.block_ordinal) is not int or self.block_ordinal < 0:
            raise ValueError("block_ordinal must be nonnegative")
        if (
            self.route_count != self.expected_route_count
            or self.route_count <= 0
            or sum(self.distance_bin_counts) != self.route_count
            or sum(self.head_route_counts) != self.route_count
        ):
            raise ValueError("streamed route inventory does not close")
        if (
            any(type(value) is not int or value <= 0 for value in self.distance_bin_upper_bounds)
            or tuple(sorted(set(self.distance_bin_upper_bounds)))
            != self.distance_bin_upper_bounds
            or len(self.distance_bin_counts) != len(self.distance_bin_upper_bounds) + 1
            or len(self.distance_bin_intrinsic_fisher_energy)
            != len(self.distance_bin_counts)
            or len(self.head_route_counts) != len(self.head_intrinsic_fisher_energy)
        ):
            raise ValueError("route aggregate bins or heads are malformed")
        mean = _finite_float_tensor(
            self.head_coordinate_mean,
            label="head_coordinate_mean",
            ndim=2,
        )
        second = _finite_float_tensor(
            self.head_coordinate_second_moment,
            label="head_coordinate_second_moment",
            ndim=2,
            nonnegative=True,
        )
        if mean.shape != second.shape or mean.shape[0] != len(self.head_route_counts):
            raise ValueError("head coordinate moments do not align")
        if any(value <= 0 for value in self.head_route_counts):
            raise ValueError("every query head must own streamed routes")
        for sequence, label in (
            (self.distance_bin_intrinsic_fisher_energy, "distance energy"),
            (self.head_intrinsic_fisher_energy, "head energy"),
        ):
            if any(not math.isfinite(value) or value < 0 for value in sequence):
                raise ValueError(f"{label} must be finite and nonnegative")
            if not math.isclose(
                sum(sequence),
                self.intrinsic_route_fisher_energy,
                rel_tol=1.0e-9,
                abs_tol=_ENERGY_ATOL,
            ):
                raise ValueError(f"{label} does not conserve intrinsic route energy")
        if (
            not math.isfinite(self.target_fisher_energy)
            or self.target_fisher_energy < 0
            or not math.isclose(
                self.allocated_target_fisher_energy,
                self.target_fisher_energy,
                rel_tol=1.0e-9,
                abs_tol=_ENERGY_ATOL,
            )
            or self.positive_target_count + self.zero_target_count != self.target_count
            or self.zero_target_count < 0
            or self.target_count <= 0
        ):
            raise ValueError("streamed route allocation does not conserve target energy")
        if (
            not math.isfinite(self.operator_output_closure_max_abs)
            or self.operator_output_closure_max_abs < 0
            or not math.isfinite(self.operator_output_closure_max_abs_limit)
            or self.operator_output_closure_max_abs_limit <= 0
            or self.operator_output_closure_max_abs
            > self.operator_output_closure_max_abs_limit
            or not math.isfinite(self.operator_output_closure_nrmse)
            or self.operator_output_closure_nrmse < 0
            or self.operator_output_closure_nrmse > 1.0e-5
        ):
            raise ValueError("operator-output closure exceeds its verified tolerance")
        if (
            not 0 < self.top_mass_route_count <= self.route_count
            or not math.isfinite(self.top_intrinsic_fisher_energy_fraction)
            or not 0 <= self.top_intrinsic_fisher_energy_fraction <= 1 + _ENERGY_ATOL
        ):
            raise ValueError("top-mass route summary is invalid")
        object.__setattr__(self, "head_coordinate_mean", mean)
        object.__setattr__(self, "head_coordinate_second_moment", second)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "carrier_sha256": self.carrier_sha256,
            "context_binding_sha256": self.context_binding_sha256,
            "postnorm_bridge_sha256": self.postnorm_bridge_sha256,
            "block_ordinal": self.block_ordinal,
            "route_count": self.route_count,
            "expected_route_count": self.expected_route_count,
            "route_row_catalog_sha256": self.route_row_catalog_sha256,
            "route_coordinate_stream_sha256": self.route_coordinate_stream_sha256,
            "route_energy_stream_sha256": self.route_energy_stream_sha256,
            "distance_bin_upper_bounds": list(self.distance_bin_upper_bounds),
            "distance_bin_counts": list(self.distance_bin_counts),
            "distance_bin_intrinsic_fisher_energy": list(
                self.distance_bin_intrinsic_fisher_energy
            ),
            "head_route_counts": list(self.head_route_counts),
            "head_intrinsic_fisher_energy": list(
                self.head_intrinsic_fisher_energy
            ),
            "head_coordinate_mean_sha256": _tensor_sha256(
                self.head_coordinate_mean
            ),
            "head_coordinate_second_moment_sha256": _tensor_sha256(
                self.head_coordinate_second_moment
            ),
            "intrinsic_route_fisher_energy": self.intrinsic_route_fisher_energy,
            "allocated_target_fisher_energy": self.allocated_target_fisher_energy,
            "target_fisher_energy": self.target_fisher_energy,
            "positive_target_count": self.positive_target_count,
            "zero_target_count": self.zero_target_count,
            "target_count": self.target_count,
            "operator_output_sha256": self.operator_output_sha256,
            "reconstructed_operator_output_sha256": (
                self.reconstructed_operator_output_sha256
            ),
            "operator_output_closure_max_abs": self.operator_output_closure_max_abs,
            "operator_output_closure_max_abs_limit": (
                self.operator_output_closure_max_abs_limit
            ),
            "operator_output_closure_nrmse": self.operator_output_closure_nrmse,
            "operator_output_closure_verified": True,
            "top_mass_route_count": self.top_mass_route_count,
            "top_mass_route_catalog_sha256": self.top_mass_route_catalog_sha256,
            "top_intrinsic_fisher_energy_fraction": (
                self.top_intrinsic_fisher_energy_fraction
            ),
            "full_route_rows_persisted": False,
            "full_route_coordinates_persisted": False,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_SURFACE_DOMAIN, self._hash_payload())

    def supplemental_evidence(self) -> tuple[str, str]:
        """Return the exact pair consumed by core ``ChartExampleStats``."""

        return self.route_row_catalog_sha256, self.sha256

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "schema": GEMMA3_CHART_MODE_SURFACE_SCHEMA,
            "format_version": GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION,
            "surface_kind": "attention_route_stream_aggregate",
            "head_coordinate_mean": self.head_coordinate_mean.detach().clone(),
            "head_coordinate_second_moment": (
                self.head_coordinate_second_moment.detach().clone()
            ),
            "receipt_sha256": self.sha256,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }


class AttentionRouteChartAccumulator:
    """Chunked route aggregator with memory bounded by targets, heads, and H."""

    def __init__(
        self,
        *,
        carrier: CarrierChartSignature,
        block_ordinal: int,
        context: AttentionRouteStreamContext,
        postnorm_bridge: AttentionPostNormBridgeReceipt,
        attention_operator_output: Tensor,
        attention_operator_output_score_gradients: Tensor,
        distance_bin_upper_bounds: Sequence[int] = (1, 4, 16, 64, 256, 512),
        top_mass_route_count: int = 64,
    ) -> None:
        if type(block_ordinal) is not int or block_ordinal < 0:
            raise ValueError("block_ordinal must be nonnegative")
        bounds = tuple(distance_bin_upper_bounds)
        if (
            not bounds
            or any(type(value) is not int or value <= 0 for value in bounds)
            or tuple(sorted(set(bounds))) != bounds
        ):
            raise ValueError("distance bins must be unique increasing positive ints")
        if type(top_mass_route_count) is not int or top_mass_route_count <= 0:
            raise ValueError("top_mass_route_count must be positive")
        expected_rows = tuple(
            (
                context.example_ids[batch],
                int(context.query_logical_positions[batch, target]),
            )
            for batch, target in context.query_valid_mask.nonzero(
                as_tuple=False
            ).tolist()
        )
        if carrier.row_keys != expected_rows:
            raise ValueError("carrier rows do not match streamed attention targets")
        if (
            postnorm_bridge.execution_fingerprint_sha256
            != carrier.execution_fingerprint_sha256
        ):
            raise ValueError("attention bridge and carrier execution must match")
        operator_gradients = _finite_float_tensor(
            attention_operator_output_score_gradients,
            label="attention_operator_output_score_gradients",
            ndim=3,
        )
        if operator_gradients.shape[:2] != context.query_valid_mask.shape or int(
            operator_gradients.shape[2]
        ) != carrier.hidden_width:
            raise ValueError("operator-output gradients must be [B,Q,H]")
        operator_output = _finite_float_tensor(
            attention_operator_output,
            label="attention_operator_output",
            ndim=3,
        )
        if operator_output.shape != operator_gradients.shape:
            raise ValueError("operator output and its score gradient must align")
        if _tensor_sha256(operator_output) != postnorm_bridge.operator_output_sha256:
            raise ValueError("operator output disagrees with bridge receipt")
        if (
            _tensor_sha256(operator_gradients)
            != postnorm_bridge.operator_output_score_gradient_sha256
        ):
            raise ValueError("operator-output gradient disagrees with bridge receipt")
        self.carrier = carrier
        self.block_ordinal = block_ordinal
        self.context = context
        self.postnorm_bridge = postnorm_bridge
        self._operator_output = operator_output
        self._operator_output_score_gradients = operator_gradients
        self.distance_bin_upper_bounds = bounds
        self.top_mass_route_count = top_mass_route_count
        self._target_index_dense = torch.full(
            context.query_valid_mask.shape,
            -1,
            dtype=torch.int64,
        )
        self._target_index_dense[context.query_valid_mask] = torch.arange(
            len(carrier.row_keys), dtype=torch.int64
        )
        head_count = context.query_head_count
        hidden = carrier.hidden_width
        self._route_count = 0
        self._last_ordinal: int | None = None
        self._row_hasher = hashlib.sha256(
            _ROW_DOMAIN + context.sha256.encode("ascii") + b"\0"
        )
        self._coordinate_hasher = hashlib.sha256(
            _TENSOR_DOMAIN
            + _canonical_json_bytes({"dtype": "torch.float64", "width": hidden})
            + b"\0"
        )
        self._energy_hasher = hashlib.sha256(_TENSOR_DOMAIN + b"route-energy\0")
        self._head_count = [0] * head_count
        self._head_energy = [0.0] * head_count
        self._head_sum = torch.zeros((head_count, hidden), dtype=torch.float64)
        self._head_square_sum = torch.zeros((head_count, hidden), dtype=torch.float64)
        self._distance_count = [0] * (len(bounds) + 1)
        self._distance_energy = [0.0] * (len(bounds) + 1)
        self._target_intrinsic = torch.zeros(len(carrier.row_keys), dtype=torch.float64)
        self._target_route_value_sum = torch.zeros(
            (len(carrier.row_keys), hidden), dtype=torch.float64
        )
        self._pending_indices = torch.empty((0, 4), dtype=torch.int64)
        self._pending_values = torch.empty((0, hidden), dtype=torch.float64)
        self._top_heap: list[tuple[float, int, int, str]] = []
        self._closed = False

    @property
    def resident_scalar_count_upper_bound(self) -> int:
        return (
            2 * self.context.query_head_count * self.carrier.hidden_width
            + len(self.carrier.row_keys)
            + len(self.carrier.row_keys) * self.carrier.hidden_width
            + 2 * self.context.query_head_count
            + 2 * (len(self.distance_bin_upper_bounds) + 1)
            + 6 * self.top_mass_route_count
            + min(
                _STREAM_AGGREGATE_BLOCK_ROWS,
                self.context.expected_route_count,
            )
            * (self.carrier.hidden_width + 4)
        )

    def add_chunk(self, *, route_indices: Tensor, route_carrier_values: Tensor) -> None:
        if self._closed:
            raise RuntimeError("attention route accumulator is closed")
        indices = _integer_tensor(route_indices, label="route_indices", ndim=2)
        values = _finite_float_tensor(
            route_carrier_values,
            label="route_carrier_values",
            ndim=2,
        )
        if indices.shape[1] != 4 or values.shape != (
            indices.shape[0],
            self.carrier.hidden_width,
        ):
            raise ValueError("route chunk must be [R,4] indices and [R,H] values")
        if not int(indices.shape[0]):
            raise ValueError("route chunks must be nonempty")
        batch, target, head, source = indices.unbind(dim=1)
        shape = self.context.query_valid_mask.shape
        key_width = int(self.context.key_valid_mask.shape[1])
        in_bounds = (
            (batch >= 0)
            & (batch < shape[0])
            & (target >= 0)
            & (target < shape[1])
            & (head >= 0)
            & (head < self.context.query_head_count)
            & (source >= 0)
            & (source < key_width)
        )
        if not bool(in_bounds.all()):
            raise ValueError("route chunk index is out of bounds")
        ordinals = (
            ((batch * shape[1] + target) * self.context.query_head_count + head)
            * key_width
            + source
        )
        if (
            (self._last_ordinal is not None and int(ordinals[0]) <= self._last_ordinal)
            or bool((ordinals[1:] <= ordinals[:-1]).any())
        ):
            raise ValueError("route chunks must follow strict canonical B,Q,H,K order")
        query_valid = self.context.query_valid_mask[batch, target]
        key_valid = self.context.key_valid_mask[batch, source]
        target_position = self.context.query_logical_positions[batch, target]
        source_position = self.context.key_logical_positions[batch, source]
        allowed = query_valid & key_valid & (source_position <= target_position)
        if self.context.window_size is not None:
            allowed &= source_position > target_position - self.context.window_size
        if not bool(allowed.all()):
            raise ValueError("route chunk contains a masked or noncausal edge")
        self._last_ordinal = int(ordinals[-1])
        gradients = self._operator_output_score_gradients[batch, target].to(
            torch.float64
        )
        values64 = values.to(torch.float64).contiguous()
        coordinates = (values64 * gradients).contiguous()
        energies = coordinates.square().sum(dim=1).contiguous()
        self._row_hasher.update(
            indices.to(torch.int64).contiguous().numpy().tobytes(order="C")
        )
        self._coordinate_hasher.update(coordinates.numpy().tobytes(order="C"))
        self._energy_hasher.update(energies.numpy().tobytes(order="C"))
        self._enqueue_aggregate_rows(indices=indices, values=values64)
        self._route_count += len(indices)

    def _enqueue_aggregate_rows(self, *, indices: Tensor, values: Tensor) -> None:
        """Aggregate at fixed row boundaries, independent of caller chunking."""

        cursor = 0
        row_count = len(indices)
        pending_count = len(self._pending_indices)
        if pending_count:
            required = _STREAM_AGGREGATE_BLOCK_ROWS - pending_count
            take = min(required, row_count)
            self._pending_indices = torch.cat(
                (self._pending_indices, indices[:take]),
                dim=0,
            )
            self._pending_values = torch.cat(
                (self._pending_values, values[:take]),
                dim=0,
            )
            cursor = take
            if len(self._pending_indices) == _STREAM_AGGREGATE_BLOCK_ROWS:
                self._consume_aggregate_block(
                    indices=self._pending_indices,
                    values=self._pending_values,
                )
                self._pending_indices = torch.empty((0, 4), dtype=torch.int64)
                self._pending_values = torch.empty(
                    (0, self.carrier.hidden_width), dtype=torch.float64
                )
        while row_count - cursor >= _STREAM_AGGREGATE_BLOCK_ROWS:
            end = cursor + _STREAM_AGGREGATE_BLOCK_ROWS
            self._consume_aggregate_block(
                indices=indices[cursor:end],
                values=values[cursor:end],
            )
            cursor = end
        if cursor < row_count:
            self._pending_indices = indices[cursor:].clone().contiguous()
            self._pending_values = values[cursor:].clone().contiguous()

    def _consume_aggregate_block(self, *, indices: Tensor, values: Tensor) -> None:
        batch, target, head, source = indices.unbind(dim=1)
        shape = self.context.query_valid_mask.shape
        key_width = int(self.context.key_valid_mask.shape[1])
        ordinals = (
            ((batch * shape[1] + target) * self.context.query_head_count + head)
            * key_width
            + source
        )
        target_rows = self._target_index_dense[batch, target]
        target_position = self.context.query_logical_positions[batch, target]
        source_position = self.context.key_logical_positions[batch, source]
        gradients = self._operator_output_score_gradients[batch, target].to(
            torch.float64
        )
        coordinates = (values * gradients).contiguous()
        energies = coordinates.square().sum(dim=1).contiguous()
        head_counts = torch.bincount(
            head,
            minlength=self.context.query_head_count,
        )
        head_energies = torch.zeros(
            self.context.query_head_count, dtype=torch.float64
        ).index_add_(0, head, energies)
        for head_index in range(self.context.query_head_count):
            self._head_count[head_index] += int(head_counts[head_index])
            self._head_energy[head_index] += float(head_energies[head_index])
        self._head_sum.index_add_(0, head, coordinates)
        self._head_square_sum.index_add_(0, head, coordinates.square())
        distances = target_position - source_position
        unassigned = torch.ones(len(indices), dtype=torch.bool)
        for bin_index, upper in enumerate(self.distance_bin_upper_bounds):
            selector = unassigned & (distances < upper)
            self._distance_count[bin_index] += int(selector.sum())
            self._distance_energy[bin_index] += float(energies[selector].sum())
            unassigned &= ~selector
        self._distance_count[-1] += int(unassigned.sum())
        self._distance_energy[-1] += float(energies[unassigned].sum())
        self._target_intrinsic.index_add_(0, target_rows, energies)
        self._target_route_value_sum.index_add_(0, target_rows, values)
        candidate_count = min(self.top_mass_route_count, len(energies))
        candidate_indices = torch.argsort(
            energies,
            descending=True,
            stable=True,
        )[:candidate_count]
        for offset in candidate_indices.tolist():
            energy = float(energies[offset])
            ordinal = int(ordinals[offset])
            row_sha = hashlib.sha256(
                _ROW_DOMAIN + struct.pack(">Q", ordinal)
            ).hexdigest()
            item = (energy, -ordinal, ordinal, row_sha)
            if len(self._top_heap) < self.top_mass_route_count:
                heapq.heappush(self._top_heap, item)
            elif item > self._top_heap[0]:
                heapq.heapreplace(self._top_heap, item)

    def finish(self) -> AttentionRouteAggregateReceipt:
        if self._closed:
            raise RuntimeError("attention route accumulator is closed")
        self._closed = True
        if self._route_count != self.context.expected_route_count:
            raise ValueError("stream did not cover the complete native route catalog")
        if len(self._pending_indices):
            self._consume_aggregate_block(
                indices=self._pending_indices,
                values=self._pending_values,
            )
            self._pending_indices = torch.empty((0, 4), dtype=torch.int64)
            self._pending_values = torch.empty(
                (0, self.carrier.hidden_width), dtype=torch.float64
            )
        common_energy = self.carrier.fisher_energy.to(torch.float64)
        intrinsic_positive = self._target_intrinsic > _ENERGY_ATOL
        expected_operator = self._operator_output[
            self.context.query_valid_mask
        ].to(torch.float64)
        residual = self._target_route_value_sum - expected_operator
        max_abs = float(residual.abs().max())
        nrmse = float(
            residual.square().mean().sqrt()
            / expected_operator.square().mean().sqrt().clamp_min(1.0e-12)
        )
        closure_limit = 1.0e-4 + 1.0e-5 * float(expected_operator.abs().max())
        if nrmse > 1.0e-5 or max_abs > closure_limit:
            raise ValueError("streamed route values do not close to operator output")
        intrinsic = float(self._target_intrinsic.sum())
        top = sorted(self._top_heap, key=lambda item: (-item[0], item[2], item[3]))
        top_payload = [
            {
                "energy": energy,
                "canonical_ordinal": ordinal,
                "row_sha256": row_sha,
            }
            for energy, _, ordinal, row_sha in top
        ]
        head_counts = torch.tensor(self._head_count, dtype=torch.float64)[:, None]
        target_energy = float(self.carrier.fisher_energy.sum())
        return AttentionRouteAggregateReceipt(
            carrier_sha256=self.carrier.sha256,
            context_binding_sha256=self.context.sha256,
            postnorm_bridge_sha256=self.postnorm_bridge.sha256,
            block_ordinal=self.block_ordinal,
            route_count=self._route_count,
            expected_route_count=self.context.expected_route_count,
            route_row_catalog_sha256=self._row_hasher.hexdigest(),
            route_coordinate_stream_sha256=self._coordinate_hasher.hexdigest(),
            route_energy_stream_sha256=self._energy_hasher.hexdigest(),
            distance_bin_upper_bounds=self.distance_bin_upper_bounds,
            distance_bin_counts=tuple(self._distance_count),
            distance_bin_intrinsic_fisher_energy=tuple(self._distance_energy),
            head_route_counts=tuple(self._head_count),
            head_intrinsic_fisher_energy=tuple(self._head_energy),
            head_coordinate_mean=(self._head_sum / head_counts).contiguous(),
            head_coordinate_second_moment=(
                self._head_square_sum / head_counts
            ).contiguous(),
            intrinsic_route_fisher_energy=intrinsic,
            allocated_target_fisher_energy=target_energy,
            target_fisher_energy=target_energy,
            positive_target_count=int(intrinsic_positive.sum()),
            zero_target_count=int((self._target_intrinsic <= _ENERGY_ATOL).sum()),
            target_count=len(self.carrier.row_keys),
            operator_output_sha256=self.postnorm_bridge.operator_output_sha256,
            reconstructed_operator_output_sha256=_tensor_sha256(
                self._target_route_value_sum
            ),
            operator_output_closure_max_abs=max_abs,
            operator_output_closure_max_abs_limit=closure_limit,
            operator_output_closure_nrmse=nrmse,
            top_mass_route_count=len(top),
            top_mass_route_catalog_sha256=_domain_sha256(_ROW_DOMAIN, top_payload),
            top_intrinsic_fisher_energy_fraction=(
                0.0
                if intrinsic <= _ENERGY_ATOL
                else sum(item[0] for item in top) / intrinsic
            ),
        )


def strict_load_attention_route_aggregate(
    value: object,
) -> AttentionRouteAggregateReceipt:
    """Strictly reconstruct a bounded route receipt and verify every digest."""

    if not isinstance(value, Mapping):
        raise ValueError("attention route aggregate must be a mapping")
    state = dict(value)
    required = {
        "schema", "format_version", "surface_kind", "carrier_sha256",
        "context_binding_sha256", "postnorm_bridge_sha256", "block_ordinal",
        "route_count", "expected_route_count", "route_row_catalog_sha256",
        "route_coordinate_stream_sha256", "route_energy_stream_sha256",
        "distance_bin_upper_bounds", "distance_bin_counts",
        "distance_bin_intrinsic_fisher_energy", "head_route_counts",
        "head_intrinsic_fisher_energy", "head_coordinate_mean",
        "head_coordinate_mean_sha256", "head_coordinate_second_moment",
        "head_coordinate_second_moment_sha256", "intrinsic_route_fisher_energy",
        "allocated_target_fisher_energy", "target_fisher_energy",
        "positive_target_count", "zero_target_count", "target_count",
        "operator_output_sha256", "reconstructed_operator_output_sha256",
        "operator_output_closure_max_abs", "operator_output_closure_max_abs_limit",
        "operator_output_closure_nrmse",
        "operator_output_closure_verified", "top_mass_route_count",
        "top_mass_route_catalog_sha256", "top_intrinsic_fisher_energy_fraction",
        "full_route_rows_persisted", "full_route_coordinates_persisted",
        "receipt_sha256", "contains_token_ids", "contains_prompt_text",
    }
    if set(state) != required:
        raise ValueError("attention route aggregate fields are not exact")
    if (
        state["schema"] != GEMMA3_CHART_MODE_SURFACE_SCHEMA
        or type(state["format_version"]) is not int
        or state["format_version"] != GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION
        or state["surface_kind"] != "attention_route_stream_aggregate"
        or state["full_route_rows_persisted"] is not False
        or state["full_route_coordinates_persisted"] is not False
        or state["contains_token_ids"] is not False
        or state["contains_prompt_text"] is not False
        or state["operator_output_closure_verified"] is not True
    ):
        raise ValueError("attention route aggregate contract drifted")
    result = AttentionRouteAggregateReceipt(
        carrier_sha256=_require_sha256(state["carrier_sha256"], label="carrier_sha256"),
        context_binding_sha256=_require_sha256(
            state["context_binding_sha256"], label="context_binding_sha256"
        ),
        postnorm_bridge_sha256=_require_sha256(
            state["postnorm_bridge_sha256"], label="postnorm_bridge_sha256"
        ),
        block_ordinal=_strict_int(state["block_ordinal"], label="block_ordinal", minimum=0),
        route_count=_strict_int(state["route_count"], label="route_count", minimum=1),
        expected_route_count=_strict_int(
            state["expected_route_count"], label="expected_route_count", minimum=1
        ),
        route_row_catalog_sha256=_require_sha256(
            state["route_row_catalog_sha256"], label="route_row_catalog_sha256"
        ),
        route_coordinate_stream_sha256=_require_sha256(
            state["route_coordinate_stream_sha256"],
            label="route_coordinate_stream_sha256",
        ),
        route_energy_stream_sha256=_require_sha256(
            state["route_energy_stream_sha256"], label="route_energy_stream_sha256"
        ),
        distance_bin_upper_bounds=tuple(
            _strict_int(item, label="distance_bin_upper_bounds item", minimum=1)
            for item in _strict_list(
                state["distance_bin_upper_bounds"],
                label="distance_bin_upper_bounds",
            )
        ),
        distance_bin_counts=tuple(
            _strict_int(item, label="distance_bin_counts item", minimum=0)
            for item in _strict_list(
                state["distance_bin_counts"], label="distance_bin_counts"
            )
        ),
        distance_bin_intrinsic_fisher_energy=tuple(
            _strict_float(item, label="distance-bin energy")
            for item in _strict_list(
                state["distance_bin_intrinsic_fisher_energy"],
                label="distance_bin_intrinsic_fisher_energy",
            )
        ),
        head_route_counts=tuple(
            _strict_int(item, label="head_route_counts item", minimum=0)
            for item in _strict_list(
                state["head_route_counts"], label="head_route_counts"
            )
        ),
        head_intrinsic_fisher_energy=tuple(
            _strict_float(item, label="head energy")
            for item in _strict_list(
                state["head_intrinsic_fisher_energy"],
                label="head_intrinsic_fisher_energy",
            )
        ),
        head_coordinate_mean=state["head_coordinate_mean"],  # type: ignore[arg-type]
        head_coordinate_second_moment=state["head_coordinate_second_moment"],  # type: ignore[arg-type]
        intrinsic_route_fisher_energy=_strict_float(
            state["intrinsic_route_fisher_energy"],
            label="intrinsic_route_fisher_energy",
        ),
        allocated_target_fisher_energy=_strict_float(
            state["allocated_target_fisher_energy"],
            label="allocated_target_fisher_energy",
        ),
        target_fisher_energy=_strict_float(
            state["target_fisher_energy"], label="target_fisher_energy"
        ),
        positive_target_count=_strict_int(
            state["positive_target_count"], label="positive_target_count", minimum=0
        ),
        zero_target_count=_strict_int(
            state["zero_target_count"], label="zero_target_count", minimum=0
        ),
        target_count=_strict_int(state["target_count"], label="target_count", minimum=1),
        operator_output_sha256=_require_sha256(
            state["operator_output_sha256"], label="operator_output_sha256"
        ),
        reconstructed_operator_output_sha256=_require_sha256(
            state["reconstructed_operator_output_sha256"],
            label="reconstructed_operator_output_sha256",
        ),
        operator_output_closure_max_abs=_strict_float(
            state["operator_output_closure_max_abs"],
            label="operator_output_closure_max_abs",
        ),
        operator_output_closure_max_abs_limit=_strict_float(
            state["operator_output_closure_max_abs_limit"],
            label="operator_output_closure_max_abs_limit",
            strictly_positive=True,
        ),
        operator_output_closure_nrmse=_strict_float(
            state["operator_output_closure_nrmse"],
            label="operator_output_closure_nrmse",
        ),
        top_mass_route_count=_strict_int(
            state["top_mass_route_count"], label="top_mass_route_count", minimum=1
        ),
        top_mass_route_catalog_sha256=_require_sha256(
            state["top_mass_route_catalog_sha256"],
            label="top_mass_route_catalog_sha256",
        ),
        top_intrinsic_fisher_energy_fraction=_strict_float(
            state["top_intrinsic_fisher_energy_fraction"],
            label="top_intrinsic_fisher_energy_fraction",
        ),
    )
    expected = result.state_dict()
    for key in required - {"head_coordinate_mean", "head_coordinate_second_moment"}:
        if state[key] != expected[key]:
            raise ValueError(f"attention route aggregate {key} failed validation")
    return result


def factorized_attention_route_kernel_receipt() -> dict[str, object]:
    """Return the fixed CPU/float64 production-kernel receipt."""

    payload = {
        "kernel_id": FACTORIZED_ATTENTION_ROUTE_KERNEL_ID,
        "device": "cpu",
        "dtype": "float64",
        "query_chunk_size": FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE,
        "key_chunk_size": FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE,
        "route_value_factorization": "P_qhk_times_U_hkhidden",
        "route_energy_factorization": (
            "P_qhk_squared_times_dot_Gq_squared_Uhk_squared"
        ),
        "source_softmax_dtype": "torch.float32",
        "source_softmax_device": "cpu",
        "probability_simplex_audit_accumulation_dtype": "torch.float64",
        "probability_simplex_forward_error_formula_id": (
            ATTENTION_ROUTE_PROBABILITY_SIMPLEX_FORMULA_ID
        ),
        "probability_simplex_support_derivation_id": (
            ATTENTION_ROUTE_PROBABILITY_SIMPLEX_SUPPORT_DERIVATION_ID
        ),
        "probability_simplex_maximum_support": (
            ATTENTION_ROUTE_PROBABILITY_SIMPLEX_MAX_SUPPORT
        ),
        "probability_simplex_float32_unit_roundoff_power": -24,
        "probability_simplex_float64_unit_roundoff_power": -53,
        "probability_simplex_float32_FTZ_safety_power": -126,
        "stored_probability_bytes_renormalized": False,
        "full_width_route_vectors_materialized": False,
        "torch_version": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
    }
    return {
        **payload,
        "receipt_sha256": _domain_sha256(
            _FACTORIZED_ROUTE_KERNEL_DOMAIN,
            payload,
        ),
    }


@dataclass(frozen=True, slots=True)
class FactorizedAttentionRouteAggregateReceipt:
    """Per-example exact route statistics without ``[route,h640]`` rows."""

    carrier_sha256: str
    context_binding_sha256: str
    postnorm_bridge_sha256: str
    example_id: str
    context_fingerprint_sha256: str
    block_ordinal: int
    source_state_sha256: str
    execution_fingerprint_sha256: str
    score_objective_receipt_sha256: str
    attention_config_sha256: str
    attention_artifact_sha256: str
    route_factor_snapshot_sha256: str
    probability_sha256: str
    repeated_value_sha256: str
    output_projection_weight_sha256: str
    projected_value_sha256: str
    operator_output_score_gradient_sha256: str
    query_valid_mask_sha256: str
    key_valid_mask_sha256: str
    query_logical_positions_sha256: str
    key_logical_positions_sha256: str
    probability_simplex_audit_sha256: str
    probability_simplex_formula_id: str
    probability_simplex_valid_row_count: int
    probability_simplex_support_minimum: int
    probability_simplex_support_maximum: int
    probability_simplex_maximum_absolute_mass_residual: float
    probability_simplex_maximum_forward_error_bound: float
    probability_simplex_minimum_error_margin: float
    kernel_id: str
    kernel_receipt_sha256: str
    query_chunk_size: int
    key_chunk_size: int
    route_count: int
    expected_route_count: int
    route_catalog_sha256: str
    distance_bin_upper_bounds: tuple[int, ...]
    distance_bin_counts: tuple[int, ...]
    distance_bin_intrinsic_fisher_energy: tuple[float, ...]
    head_route_counts: tuple[int, ...]
    head_intrinsic_fisher_energy: tuple[float, ...]
    head_coordinate_mean: Tensor
    head_coordinate_second_moment: Tensor
    intrinsic_route_fisher_energy: float
    common_postnorm_target_fisher_energy: float
    intrinsic_positive_target_count: int
    intrinsic_zero_target_count: int
    common_positive_target_count: int
    common_zero_target_count: int
    target_count: int
    operator_output_sha256: str
    reconstructed_operator_output_sha256: str
    operator_output_closure_max_abs: float
    operator_output_closure_max_abs_limit: float
    operator_output_closure_nrmse: float
    operator_contribution_sha256: str
    reconstructed_operator_contribution_sha256: str
    operator_contribution_closure_max_abs: float
    operator_contribution_closure_max_abs_limit: float
    operator_contribution_closure_nrmse: float
    top_mass_route_count: int
    top_mass_route_catalog_sha256: str
    top_intrinsic_fisher_energy_fraction: float
    materialized_full_width_route_scalar_count: int
    peak_route_energy_scalar_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "carrier_sha256",
            "context_binding_sha256",
            "postnorm_bridge_sha256",
            "context_fingerprint_sha256",
            "source_state_sha256",
            "execution_fingerprint_sha256",
            "score_objective_receipt_sha256",
            "attention_config_sha256",
            "attention_artifact_sha256",
            "route_factor_snapshot_sha256",
            "probability_sha256",
            "repeated_value_sha256",
            "output_projection_weight_sha256",
            "projected_value_sha256",
            "operator_output_score_gradient_sha256",
            "query_valid_mask_sha256",
            "key_valid_mask_sha256",
            "query_logical_positions_sha256",
            "key_logical_positions_sha256",
            "probability_simplex_audit_sha256",
            "kernel_receipt_sha256",
            "route_catalog_sha256",
            "operator_output_sha256",
            "reconstructed_operator_output_sha256",
            "operator_contribution_sha256",
            "reconstructed_operator_contribution_sha256",
            "top_mass_route_catalog_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        _require_name(self.example_id, label="example_id")
        simplex_audit = AttentionRouteProbabilitySimplexAudit(
            formula_id=self.probability_simplex_formula_id,
            source_softmax_dtype="torch.float32",
            source_softmax_device="cpu",
            audit_accumulation_dtype="torch.float64",
            valid_row_count=self.probability_simplex_valid_row_count,
            support_minimum=self.probability_simplex_support_minimum,
            support_maximum=self.probability_simplex_support_maximum,
            maximum_absolute_mass_residual=(
                self.probability_simplex_maximum_absolute_mass_residual
            ),
            maximum_forward_error_bound=(
                self.probability_simplex_maximum_forward_error_bound
            ),
            minimum_error_margin=self.probability_simplex_minimum_error_margin,
        )
        if simplex_audit.sha256 != self.probability_simplex_audit_sha256:
            raise ValueError("probability simplex audit digest drifted")
        if type(self.block_ordinal) is not int or self.block_ordinal < 0:
            raise ValueError("block_ordinal must be nonnegative")
        if self.kernel_id != FACTORIZED_ATTENTION_ROUTE_KERNEL_ID:
            raise ValueError("factorized route kernel_id drifted")
        if (
            self.kernel_receipt_sha256
            != factorized_attention_route_kernel_receipt()["receipt_sha256"]
        ):
            raise ValueError("factorized route kernel receipt drifted")
        if (
            self.query_chunk_size
            != FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE
            or self.key_chunk_size
            != FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE
        ):
            raise ValueError("factorized route chunk schedule drifted")
        if (
            self.route_count != self.expected_route_count
            or self.route_count <= 0
            or sum(self.distance_bin_counts) != self.route_count
            or sum(self.head_route_counts) != self.route_count
        ):
            raise ValueError("factorized route inventory does not close")
        if self.probability_simplex_valid_row_count != (
            len(self.head_route_counts) * self.target_count
        ):
            raise ValueError(
                "probability simplex audit row coverage does not close"
            )
        head_count = len(self.head_route_counts)
        if (
            head_count <= 0
            or len(set(self.head_route_counts)) != 1
            or self.route_count != head_count * self.head_route_counts[0]
            or any(count % head_count for count in self.distance_bin_counts)
        ):
            raise ValueError(
                "shared route support does not close identically across heads"
            )
        target_rows = self.target_count
        per_head_route_count = self.head_route_counts[0]
        support_minimum = self.probability_simplex_support_minimum
        support_maximum = self.probability_simplex_support_maximum
        if support_minimum == support_maximum:
            support_inventory_closes = (
                per_head_route_count == target_rows * support_minimum
            )
        else:
            support_inventory_closes = (
                target_rows >= 2
                and (target_rows - 1) * support_minimum + support_maximum
                <= per_head_route_count
                <= support_minimum + (target_rows - 1) * support_maximum
            )
        if not support_inventory_closes:
            raise ValueError("probability simplex support inventory does not close")
        if (
            not self.distance_bin_upper_bounds
            or tuple(sorted(set(self.distance_bin_upper_bounds)))
            != self.distance_bin_upper_bounds
            or any(value <= 0 for value in self.distance_bin_upper_bounds)
            or len(self.distance_bin_counts)
            != len(self.distance_bin_upper_bounds) + 1
            or len(self.distance_bin_intrinsic_fisher_energy)
            != len(self.distance_bin_counts)
            or len(self.head_route_counts)
            != len(self.head_intrinsic_fisher_energy)
            or any(value <= 0 for value in self.head_route_counts)
        ):
            raise ValueError("factorized route bins or heads are malformed")
        mean = _finite_float_tensor(
            self.head_coordinate_mean,
            label="head_coordinate_mean",
            ndim=2,
        )
        second = _finite_float_tensor(
            self.head_coordinate_second_moment,
            label="head_coordinate_second_moment",
            ndim=2,
            nonnegative=True,
        )
        if mean.shape != second.shape or mean.shape[0] != len(
            self.head_route_counts
        ):
            raise ValueError("factorized head coordinate moments do not align")
        for values, label in (
            (
                self.distance_bin_intrinsic_fisher_energy,
                "distance energy",
            ),
            (self.head_intrinsic_fisher_energy, "head energy"),
        ):
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{label} must be finite and nonnegative")
            if not math.isclose(
                sum(values),
                self.intrinsic_route_fisher_energy,
                rel_tol=1.0e-9,
                abs_tol=_ENERGY_ATOL,
            ):
                raise ValueError(f"{label} does not conserve route energy")
        if (
            not math.isfinite(self.intrinsic_route_fisher_energy)
            or self.intrinsic_route_fisher_energy < 0
            or not math.isfinite(self.common_postnorm_target_fisher_energy)
            or self.common_postnorm_target_fisher_energy < 0
            or self.intrinsic_positive_target_count
            + self.intrinsic_zero_target_count
            != self.target_count
            or self.common_positive_target_count
            + self.common_zero_target_count
            != self.target_count
            or self.target_count <= 0
        ):
            raise ValueError("factorized target energy diagnostics are invalid")
        for prefix in ("operator_output", "operator_contribution"):
            maximum = getattr(self, f"{prefix}_closure_max_abs")
            limit = getattr(self, f"{prefix}_closure_max_abs_limit")
            nrmse = getattr(self, f"{prefix}_closure_nrmse")
            if (
                not math.isfinite(maximum)
                or maximum < 0
                or not math.isfinite(limit)
                or limit <= 0
                or maximum > limit
                or not math.isfinite(nrmse)
                or nrmse < 0
                or nrmse > 1.0e-5
            ):
                raise ValueError(f"{prefix} closure exceeds verified tolerance")
        if (
            not 0 < self.top_mass_route_count <= self.route_count
            or not math.isfinite(self.top_intrinsic_fisher_energy_fraction)
            or not 0
            <= self.top_intrinsic_fisher_energy_fraction
            <= 1 + _ENERGY_ATOL
            or self.materialized_full_width_route_scalar_count != 0
            or self.peak_route_energy_scalar_count <= 0
            or self.peak_route_energy_scalar_count
            > len(self.head_route_counts)
            * self.query_chunk_size
            * self.key_chunk_size
        ):
            raise ValueError("factorized route scale accounting is invalid")
        object.__setattr__(self, "head_coordinate_mean", mean)
        object.__setattr__(self, "head_coordinate_second_moment", second)

    def _hash_payload(self) -> dict[str, object]:
        return {
            field_name: (
                _tensor_sha256(getattr(self, field_name))
                if field_name
                in {"head_coordinate_mean", "head_coordinate_second_moment"}
                else list(getattr(self, field_name))
                if field_name
                in {
                    "distance_bin_upper_bounds",
                    "distance_bin_counts",
                    "distance_bin_intrinsic_fisher_energy",
                    "head_route_counts",
                    "head_intrinsic_fisher_energy",
                }
                else getattr(self, field_name)
            )
            for field_name in self.__dataclass_fields__
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_FACTORIZED_ROUTE_DOMAIN, self._hash_payload())

    def supplemental_evidence(self) -> tuple[str, str]:
        return self.route_catalog_sha256, self.sha256

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "schema": GEMMA3_CHART_MODE_SURFACE_SCHEMA,
            "format_version": GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION,
            "surface_kind": "factorized_attention_route_aggregate",
            "head_coordinate_mean": self.head_coordinate_mean.detach().clone(),
            "head_coordinate_second_moment": (
                self.head_coordinate_second_moment.detach().clone()
            ),
            "receipt_sha256": self.sha256,
            "full_route_rows_persisted": False,
            "full_route_coordinates_persisted": False,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }


def _closure_statistics(
    reconstructed: Tensor,
    expected: Tensor,
) -> tuple[float, float, float]:
    residual = reconstructed - expected
    max_abs = float(residual.abs().max())
    nrmse = float(
        residual.square().mean().sqrt()
        / expected.square().mean().sqrt().clamp_min(1.0e-12)
    )
    limit = 1.0e-4 + 1.0e-5 * float(expected.abs().max())
    return max_abs, limit, nrmse


def build_factorized_attention_route_aggregates(
    *,
    carrier: CarrierChartSignature,
    block_ordinal: int,
    context: AttentionRouteStreamContext,
    postnorm_bridge: AttentionPostNormBridgeReceipt,
    route_factors: Gemma3AttentionRouteFactorSnapshot,
    attention_operator_output: Tensor,
    attention_operator_output_score_gradients: Tensor,
    distance_bin_upper_bounds: Sequence[int] = (1, 4, 16, 64, 256, 512),
    top_mass_route_count: int = 64,
) -> tuple[FactorizedAttentionRouteAggregateReceipt, ...]:
    """Aggregate exact routes from ``P``, repeated ``V``, ``O``, and ``G``.

    Resident route-sized work is only a scalar energy tile ``[H,qc,kc]``;
    no residual-width route tensor or literal coordinate stream is formed.
    """

    if not isinstance(carrier, CarrierChartSignature):
        raise TypeError("carrier must be a CarrierChartSignature")
    if not isinstance(context, AttentionRouteStreamContext):
        raise TypeError("context must be an AttentionRouteStreamContext")
    if not isinstance(postnorm_bridge, AttentionPostNormBridgeReceipt):
        raise TypeError("postnorm_bridge must be an AttentionPostNormBridgeReceipt")
    if not isinstance(route_factors, Gemma3AttentionRouteFactorSnapshot):
        raise TypeError(
            "route_factors must be a Gemma3AttentionRouteFactorSnapshot"
        )
    if route_factors.output_projection_bias_present:
        raise ValueError(
            "factorized attention routes require a bias-free O projection"
        )
    if type(block_ordinal) is not int or block_ordinal < 0:
        raise ValueError("block_ordinal must be nonnegative")
    bounds = tuple(distance_bin_upper_bounds)
    if (
        not bounds
        or any(type(value) is not int or value <= 0 for value in bounds)
        or tuple(sorted(set(bounds))) != bounds
    ):
        raise ValueError("distance bins must be unique increasing positive ints")
    if type(top_mass_route_count) is not int or top_mass_route_count <= 0:
        raise ValueError("top_mass_route_count must be positive")
    if carrier.mode_id != f"mode.{route_factors.invocation_id}":
        raise ValueError("route factors do not match the carrier mode")
    if (
        carrier.execution_fingerprint_sha256
        != postnorm_bridge.execution_fingerprint_sha256
    ):
        raise ValueError("carrier and postnorm bridge execution disagree")
    expected_rows = tuple(
        (
            context.example_ids[batch],
            int(context.query_logical_positions[batch, query]),
        )
        for batch, query in context.query_valid_mask.nonzero(
            as_tuple=False
        ).tolist()
    )
    if carrier.row_keys != expected_rows:
        raise ValueError("carrier rows do not match factorized route targets")
    operator = _finite_float_tensor(
        attention_operator_output,
        label="attention_operator_output",
        ndim=3,
    )
    gradient = _finite_float_tensor(
        attention_operator_output_score_gradients,
        label="attention_operator_output_score_gradients",
        ndim=3,
    )
    if operator.shape != gradient.shape or operator.shape[:2] != (
        context.query_valid_mask.shape
    ):
        raise ValueError("operator output/gradient must align with route context")
    if int(operator.shape[2]) != carrier.hidden_width:
        raise ValueError("operator output width disagrees with carrier")
    full_operator_output_sha256 = _tensor_sha256(operator)
    full_operator_gradient_sha256 = _tensor_sha256(gradient)
    if full_operator_output_sha256 != postnorm_bridge.operator_output_sha256:
        raise ValueError("operator output disagrees with postnorm bridge")
    if (
        full_operator_gradient_sha256
        != postnorm_bridge.operator_output_score_gradient_sha256
    ):
        raise ValueError("operator gradient disagrees with postnorm bridge")
    if (
        route_factors.probabilities.shape[0]
        != context.query_valid_mask.shape[0]
        or route_factors.probabilities.shape[2:]
        != (
            context.query_valid_mask.shape[1],
            context.key_valid_mask.shape[1],
        )
        or route_factors.query_head_count != context.query_head_count
        or route_factors.window_size != context.window_size
        or not torch.equal(
            route_factors.key_valid_mask,
            context.key_valid_mask,
        )
        or not torch.equal(
            route_factors.query_logical_positions,
            context.query_logical_positions,
        )
        or not torch.equal(
            route_factors.key_logical_positions,
            context.key_logical_positions,
        )
        or bool(
            (
                context.query_valid_mask
                & ~route_factors.query_valid_mask
            ).any()
        )
    ):
        raise ValueError("route factors do not match native q/k context")
    context_route_allowed = derive_attention_route_allowed_mask(
        query_valid_mask=context.query_valid_mask,
        key_valid_mask=context.key_valid_mask,
        query_logical_positions=context.query_logical_positions,
        key_logical_positions=context.key_logical_positions,
        window_size=context.window_size,
    )
    rope = context.rope
    if (
        not math.isclose(route_factors.rope_theta, rope.theta)
        or route_factors.head_dimension != rope.rotary_dimension
        or route_factors.rope_scaling_kind != rope.scaling_kind
        or route_factors.rope_scaling_factor != rope.scaling_factor
    ):
        raise ValueError("route factors disagree with native RoPE context")
    output_weight = route_factors.output_projection_weight.to(torch.float64)
    if output_weight.shape[0] != carrier.hidden_width:
        raise ValueError("route O projection does not produce the carrier width")
    kernel = factorized_attention_route_kernel_receipt()
    route_factor_snapshot_sha256 = route_factors.sha256
    output_projection_weight_sha256 = _tensor_sha256(output_weight)
    carrier_sha256 = carrier.sha256
    context_binding_sha256 = context.sha256
    postnorm_bridge_sha256 = postnorm_bridge.sha256
    carrier_index = {
        row_key: index for index, row_key in enumerate(carrier.row_keys)
    }
    receipts: list[FactorizedAttentionRouteAggregateReceipt] = []
    heads = context.query_head_count
    head_dimension = route_factors.head_dimension
    full_key_width = int(context.key_valid_mask.shape[1])
    output_head_slices = output_weight.reshape(
        carrier.hidden_width,
        heads,
        head_dimension,
    ).permute(1, 2, 0).contiguous()
    for batch, (example_id, context_fingerprint) in enumerate(
        zip(
            context.example_ids,
            context.context_fingerprints,
            strict=True,
        )
    ):
        query_indices = context.query_valid_mask[batch].nonzero(
            as_tuple=False
        ).flatten()
        key_indices = context.key_valid_mask[batch].nonzero(
            as_tuple=False
        ).flatten()
        query_positions = context.query_logical_positions[
            batch, query_indices
        ].contiguous()
        key_positions = context.key_logical_positions[
            batch, key_indices
        ].contiguous()
        allowed = (
            context_route_allowed[batch]
            .index_select(0, query_indices)
            .index_select(1, key_indices)
            .contiguous()
        )
        expected_route_count = int(allowed.sum()) * heads
        if expected_route_count <= 0:
            raise ValueError("factorized example has no native route")
        stored_probabilities = (
            route_factors.probabilities[batch]
            .index_select(1, query_indices)
            .index_select(2, key_indices)
            .contiguous()
        )
        simplex_audit = validate_attention_route_probability_simplex(
            stored_probabilities,
            allowed_mask=allowed[None, :, :].expand_as(stored_probabilities),
            valid_row_mask=torch.ones(
                (heads, len(query_indices)),
                dtype=torch.bool,
            ),
        )
        probabilities = stored_probabilities.to(torch.float64).contiguous()
        repeated_values = (
            route_factors.repeated_values[batch]
            .index_select(1, key_indices)
            .to(torch.float64)
            .contiguous()
        )
        projected_values = torch.einsum(
            "hkd,hdo->hko",
            repeated_values,
            output_head_slices,
        ).contiguous()
        score_gradient = gradient[batch, query_indices].to(
            torch.float64
        ).contiguous()
        expected_operator = operator[batch, query_indices].to(
            torch.float64
        ).contiguous()
        reconstructed_operator = torch.empty_like(expected_operator)
        head_sum = torch.zeros(
            (heads, carrier.hidden_width), dtype=torch.float64
        )
        head_square_sum = torch.zeros_like(head_sum)
        head_energy = torch.zeros(heads, dtype=torch.float64)
        target_intrinsic = torch.zeros(
            len(query_indices), dtype=torch.float64
        )
        head_counts = torch.zeros(heads, dtype=torch.int64)
        distance_counts = [0] * (len(bounds) + 1)
        distance_energy = [0.0] * (len(bounds) + 1)
        top_heap: list[tuple[float, int, int, str]] = []
        peak_route_energy_scalars = 0
        for query_start in range(
            0,
            len(query_indices),
            FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE,
        ):
            query_end = min(
                len(query_indices),
                query_start + FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE,
            )
            probability_query = probabilities[:, query_start:query_end]
            gradient_query = score_gradient[query_start:query_end]
            route_context = torch.einsum(
                "hqk,hko->hqo",
                probability_query,
                projected_values,
            )
            reconstructed_operator[query_start:query_end] = route_context.sum(
                dim=0
            )
            head_sum += (
                route_context * gradient_query.unsqueeze(0)
            ).sum(dim=1)
            squared_route_context = torch.einsum(
                "hqk,hko->hqo",
                probability_query.square(),
                projected_values.square(),
            )
            head_square_sum += (
                squared_route_context * gradient_query.square().unsqueeze(0)
            ).sum(dim=1)
            for key_start in range(
                0,
                len(key_indices),
                FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE,
            ):
                key_end = min(
                    len(key_indices),
                    key_start + FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE,
                )
                probability_tile = probability_query[:, :, key_start:key_end]
                projected_square = projected_values[
                    :, key_start:key_end
                ].square()
                dot_square = torch.einsum(
                    "qo,hko->hqk",
                    gradient_query.square(),
                    projected_square,
                )
                energy = probability_tile.square() * dot_square
                allowed_tile = allowed[
                    query_start:query_end,
                    key_start:key_end,
                ]
                energy = energy.masked_fill(~allowed_tile.unsqueeze(0), 0)
                peak_route_energy_scalars = max(
                    peak_route_energy_scalars,
                    energy.numel(),
                )
                head_energy += energy.sum(dim=(1, 2))
                target_intrinsic[query_start:query_end] += energy.sum(
                    dim=(0, 2)
                )
                allowed_count = int(allowed_tile.sum())
                head_counts += allowed_count
                distances = (
                    query_positions[query_start:query_end, None]
                    - key_positions[None, key_start:key_end]
                )
                unassigned = allowed_tile.clone()
                for bin_index, upper in enumerate(bounds):
                    selector = unassigned & (distances < upper)
                    distance_counts[bin_index] += int(selector.sum()) * heads
                    distance_energy[bin_index] += float(
                        energy.masked_select(selector.unsqueeze(0)).sum()
                    )
                    unassigned &= ~selector
                distance_counts[-1] += int(unassigned.sum()) * heads
                distance_energy[-1] += float(
                    energy.masked_select(unassigned.unsqueeze(0)).sum()
                )
                energy_qhk = energy.permute(1, 0, 2).contiguous()
                valid_qhk = allowed_tile[:, None, :].expand_as(energy_qhk)
                candidate_energy = energy_qhk[valid_qhk]
                q_local, head_local, key_local = valid_qhk.nonzero(
                    as_tuple=True
                )
                ordinals = (
                    (
                        query_indices[query_start:query_end][q_local]
                        * heads
                        + head_local
                    )
                    * full_key_width
                    + key_indices[key_start:key_end][key_local]
                )
                take = min(top_mass_route_count, len(candidate_energy))
                candidate_order = torch.argsort(
                    candidate_energy,
                    descending=True,
                    stable=True,
                )[:take]
                for candidate in candidate_order.tolist():
                    ordinal = int(ordinals[candidate])
                    energy_value = float(candidate_energy[candidate])
                    query_dense = int(
                        query_indices[query_start:query_end][q_local[candidate]]
                    )
                    key_dense = int(
                        key_indices[key_start:key_end][key_local[candidate]]
                    )
                    head = int(head_local[candidate])
                    row_sha = _domain_sha256(
                        _ROW_DOMAIN,
                        {
                            "context_fingerprint_sha256": context_fingerprint,
                            "example_id": example_id,
                            "query_head": head,
                            "source_position": int(
                                context.key_logical_positions[batch, key_dense]
                            ),
                            "target_position": int(
                                context.query_logical_positions[
                                    batch, query_dense
                                ]
                            ),
                        },
                    )
                    item = (energy_value, -ordinal, ordinal, row_sha)
                    if len(top_heap) < top_mass_route_count:
                        heapq.heappush(top_heap, item)
                    elif item > top_heap[0]:
                        heapq.heapreplace(top_heap, item)
        route_count = int(head_counts.sum())
        if route_count != expected_route_count:
            raise RuntimeError("factorized route enumeration did not close")
        max_abs, max_abs_limit, nrmse = _closure_statistics(
            reconstructed_operator,
            expected_operator,
        )
        if max_abs > max_abs_limit or nrmse > 1.0e-5:
            raise ValueError("factorized routes do not close to operator output")
        reconstructed_contribution = (
            reconstructed_operator * score_gradient
        )
        operator_contribution = expected_operator * score_gradient
        contribution_max_abs, contribution_limit, contribution_nrmse = (
            _closure_statistics(
                reconstructed_contribution,
                operator_contribution,
            )
        )
        if (
            contribution_max_abs > contribution_limit
            or contribution_nrmse > 1.0e-5
        ):
            raise ValueError(
                "factorized routes do not close to operator contribution"
            )
        common_indices = [
            carrier_index[(example_id, int(position))]
            for position in query_positions.tolist()
        ]
        common_energy = carrier.fisher_energy[common_indices].to(torch.float64)
        intrinsic = float(target_intrinsic.sum())
        top = sorted(
            top_heap,
            key=lambda item: (-item[0], item[2], item[3]),
        )
        top_payload = [
            {
                "energy": energy_value,
                "canonical_ordinal": ordinal,
                "row_sha256": row_sha,
            }
            for energy_value, _, ordinal, row_sha in top
        ]
        query_mask = context.query_valid_mask[batch].contiguous()
        key_mask = context.key_valid_mask[batch].contiguous()
        query_position_row = context.query_logical_positions[batch].contiguous()
        key_position_row = context.key_logical_positions[batch].contiguous()
        route_catalog_sha256 = _domain_sha256(
            _ROW_DOMAIN,
            {
                "canonical_order": "query_then_head_then_key",
                "context_binding_sha256": context_binding_sha256,
                "context_fingerprint_sha256": context_fingerprint,
                "example_id": example_id,
                "expected_route_count": expected_route_count,
                "key_logical_positions_sha256": _tensor_sha256(
                    key_position_row
                ),
                "key_valid_mask_sha256": _tensor_sha256(key_mask),
                "query_head_count": heads,
                "query_logical_positions_sha256": _tensor_sha256(
                    query_position_row
                ),
                "query_valid_mask_sha256": _tensor_sha256(query_mask),
                "window_size": context.window_size,
            },
        )
        receipts.append(
            FactorizedAttentionRouteAggregateReceipt(
                carrier_sha256=carrier_sha256,
                context_binding_sha256=context_binding_sha256,
                postnorm_bridge_sha256=postnorm_bridge_sha256,
                example_id=example_id,
                context_fingerprint_sha256=context_fingerprint,
                block_ordinal=block_ordinal,
                source_state_sha256=carrier.source_state_sha256,
                execution_fingerprint_sha256=(
                    carrier.execution_fingerprint_sha256
                ),
                score_objective_receipt_sha256=(
                    carrier.score_objective_receipt_sha256
                ),
                attention_config_sha256=route_factors.config_sha256,
                attention_artifact_sha256=route_factors.artifact_sha256,
                route_factor_snapshot_sha256=route_factor_snapshot_sha256,
                probability_sha256=_tensor_sha256(stored_probabilities),
                repeated_value_sha256=_tensor_sha256(repeated_values),
                output_projection_weight_sha256=(
                    output_projection_weight_sha256
                ),
                projected_value_sha256=_tensor_sha256(projected_values),
                operator_output_score_gradient_sha256=_tensor_sha256(
                    score_gradient
                ),
                query_valid_mask_sha256=_tensor_sha256(query_mask),
                key_valid_mask_sha256=_tensor_sha256(key_mask),
                query_logical_positions_sha256=_tensor_sha256(
                    query_position_row
                ),
                key_logical_positions_sha256=_tensor_sha256(
                    key_position_row
                ),
                probability_simplex_audit_sha256=simplex_audit.sha256,
                probability_simplex_formula_id=simplex_audit.formula_id,
                probability_simplex_valid_row_count=(
                    simplex_audit.valid_row_count
                ),
                probability_simplex_support_minimum=(
                    simplex_audit.support_minimum
                ),
                probability_simplex_support_maximum=(
                    simplex_audit.support_maximum
                ),
                probability_simplex_maximum_absolute_mass_residual=(
                    simplex_audit.maximum_absolute_mass_residual
                ),
                probability_simplex_maximum_forward_error_bound=(
                    simplex_audit.maximum_forward_error_bound
                ),
                probability_simplex_minimum_error_margin=(
                    simplex_audit.minimum_error_margin
                ),
                kernel_id=FACTORIZED_ATTENTION_ROUTE_KERNEL_ID,
                kernel_receipt_sha256=kernel["receipt_sha256"],
                query_chunk_size=FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE,
                key_chunk_size=FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE,
                route_count=route_count,
                expected_route_count=expected_route_count,
                route_catalog_sha256=route_catalog_sha256,
                distance_bin_upper_bounds=bounds,
                distance_bin_counts=tuple(distance_counts),
                distance_bin_intrinsic_fisher_energy=tuple(distance_energy),
                head_route_counts=tuple(int(value) for value in head_counts),
                head_intrinsic_fisher_energy=tuple(
                    float(value) for value in head_energy
                ),
                head_coordinate_mean=(
                    head_sum / head_counts.to(torch.float64)[:, None]
                ).contiguous(),
                head_coordinate_second_moment=(
                    head_square_sum
                    / head_counts.to(torch.float64)[:, None]
                ).contiguous(),
                intrinsic_route_fisher_energy=intrinsic,
                common_postnorm_target_fisher_energy=float(
                    common_energy.sum()
                ),
                intrinsic_positive_target_count=int(
                    (target_intrinsic > _ENERGY_ATOL).sum()
                ),
                intrinsic_zero_target_count=int(
                    (target_intrinsic <= _ENERGY_ATOL).sum()
                ),
                common_positive_target_count=int(
                    (common_energy > _ENERGY_ATOL).sum()
                ),
                common_zero_target_count=int(
                    (common_energy <= _ENERGY_ATOL).sum()
                ),
                target_count=len(query_indices),
                operator_output_sha256=_tensor_sha256(expected_operator),
                reconstructed_operator_output_sha256=_tensor_sha256(
                    reconstructed_operator
                ),
                operator_output_closure_max_abs=max_abs,
                operator_output_closure_max_abs_limit=max_abs_limit,
                operator_output_closure_nrmse=nrmse,
                operator_contribution_sha256=_tensor_sha256(
                    operator_contribution
                ),
                reconstructed_operator_contribution_sha256=_tensor_sha256(
                    reconstructed_contribution
                ),
                operator_contribution_closure_max_abs=contribution_max_abs,
                operator_contribution_closure_max_abs_limit=(
                    contribution_limit
                ),
                operator_contribution_closure_nrmse=contribution_nrmse,
                top_mass_route_count=len(top),
                top_mass_route_catalog_sha256=_domain_sha256(
                    _ROW_DOMAIN,
                    top_payload,
                ),
                top_intrinsic_fisher_energy_fraction=(
                    0.0
                    if intrinsic <= _ENERGY_ATOL
                    else sum(item[0] for item in top) / intrinsic
                ),
                materialized_full_width_route_scalar_count=0,
                peak_route_energy_scalar_count=peak_route_energy_scalars,
            )
        )
    return tuple(receipts)


def bind_carrier_signature_to_chart_mode_spec(
    signature: CarrierChartSignature,
    mode_spec: object,
) -> CarrierChartSignature:
    """Verify a surface against the model-independent graph-core mode spec."""

    from .chart_energy_neighbor_graph import (
        ChartModeSpec,
        canonical_carrier_comparison_abi,
    )

    if not isinstance(signature, CarrierChartSignature):
        raise TypeError("signature must be a CarrierChartSignature")
    if not isinstance(mode_spec, ChartModeSpec):
        raise TypeError("mode_spec must be a ChartModeSpec")
    canonical = canonical_carrier_comparison_abi()
    if mode_spec.comparison_abi.sha256 != canonical.sha256:
        raise ValueError("mode spec does not use the canonical carrier comparison ABI")
    checks = {
        "mode_id": (mode_spec.mode_id, signature.mode_id),
        "state fingerprint": (
            mode_spec.state_fingerprint_sha256,
            signature.source_state_sha256,
        ),
        "native ABI": (mode_spec.native_abi.sha256, signature.native_abi_sha256),
        "comparison ABI": (
            mode_spec.comparison_abi.sha256,
            signature.comparison_abi_sha256,
        ),
        "comparison adapter receipt": (
            mode_spec.comparison_adapter_receipt_sha256,
            signature.mapping_receipt_sha256,
        ),
    }
    for label, (claimed, observed) in checks.items():
        if claimed != observed:
            raise ValueError(f"carrier signature {label} disagrees with mode spec")
    return signature


@dataclass(frozen=True, slots=True)
class AttentionRouteRowKey:
    context_fingerprint_sha256: str
    example_id: str
    source_position: int
    target_position: int
    query_head: int

    def __post_init__(self) -> None:
        _require_sha256(
            self.context_fingerprint_sha256,
            label="route context fingerprint",
        )
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("route example_id must be nonempty")
        for value, label in (
            (self.source_position, "source_position"),
            (self.target_position, "target_position"),
            (self.query_head, "query_head"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be nonnegative")
        if self.source_position > self.target_position:
            raise ValueError("a causal route cannot point from the future")

    def state_dict(self) -> dict[str, object]:
        return {
            "context_fingerprint_sha256": self.context_fingerprint_sha256,
            "example_id": self.example_id,
            "source_position": self.source_position,
            "target_position": self.target_position,
            "query_head": self.query_head,
        }


@dataclass(frozen=True, slots=True)
class AttentionRouteChartSurface:
    """Common target signature plus supplemental causal route coordinates."""

    carrier: CarrierChartSignature
    block_ordinal: int
    context_binding: AttentionNativeContextBinding
    postnorm_bridge: AttentionPostNormBridgeReceipt
    route_row_keys: tuple[AttentionRouteRowKey, ...]
    route_target_row_indices: Tensor
    route_coordinates: Tensor
    route_intrinsic_fisher_energy: Tensor
    route_energy_fraction: Tensor
    route_allocated_target_fisher_energy: Tensor

    def __post_init__(self) -> None:
        if type(self.block_ordinal) is not int or self.block_ordinal < 0:
            raise ValueError("block_ordinal must be nonnegative")
        if not self.carrier.mode_id.startswith("mode.mutation."):
            raise ValueError("attention route mode_id must be a mutation mode")
        if (
            self.postnorm_bridge.execution_fingerprint_sha256
            != self.carrier.execution_fingerprint_sha256
        ):
            raise ValueError("attention bridge and carrier execution must match")
        expected_rows = int(self.context_binding.native_route_mask.sum().item())
        if len(self.route_row_keys) != expected_rows or not self.route_row_keys:
            raise ValueError("route rows must cover every native route exactly once")
        if len({tuple(row.state_dict().values()) for row in self.route_row_keys}) != len(
            self.route_row_keys
        ):
            raise ValueError("attention route row keys must be unique")
        target_indices = _integer_tensor(
            self.route_target_row_indices,
            label="route_target_row_indices",
            ndim=1,
        )
        if (
            target_indices.shape[0] != expected_rows
            or bool((target_indices < 0).any())
            or bool((target_indices >= len(self.carrier.row_keys)).any())
        ):
            raise ValueError("route target row indices are out of bounds")
        coordinates = _finite_float_tensor(
            self.route_coordinates,
            label="route_coordinates",
            ndim=2,
        )
        if coordinates.shape[0] != expected_rows:
            raise ValueError("route coordinates do not align with route rows")
        for value, label in (
            (self.route_intrinsic_fisher_energy, "route_intrinsic_fisher_energy"),
            (self.route_energy_fraction, "route_energy_fraction"),
            (
                self.route_allocated_target_fisher_energy,
                "route_allocated_target_fisher_energy",
            ),
        ):
            tensor = _finite_float_tensor(
                value,
                label=label,
                ndim=1,
                nonnegative=True,
            )
            if tensor.shape[0] != expected_rows:
                raise ValueError(f"{label} does not align with route rows")
        binding = self.context_binding
        dense_target_indices = {
            (batch, target): index
            for index, (batch, target) in enumerate(
                binding.query_valid_mask.nonzero(as_tuple=False).tolist()
            )
        }
        native_indices = binding.native_route_mask.nonzero(as_tuple=False).tolist()
        expected_keys = tuple(
            AttentionRouteRowKey(
                context_fingerprint_sha256=binding.context_fingerprints[batch],
                example_id=binding.example_ids[batch],
                source_position=int(binding.key_logical_positions[batch, source]),
                target_position=int(binding.query_logical_positions[batch, target]),
                query_head=head,
            )
            for batch, target, head, source in native_indices
        )
        expected_target_indices = torch.tensor(
            [dense_target_indices[(batch, target)] for batch, target, _, _ in native_indices],
            dtype=torch.int64,
        )
        if self.route_row_keys != expected_keys or not torch.equal(
            target_indices, expected_target_indices
        ):
            raise ValueError("route row identity does not match the native mask")
        expected_intrinsic = coordinates.to(torch.float64).square().sum(dim=1)
        if not _energy_close(
            self.route_intrinsic_fisher_energy.to(torch.float64),
            expected_intrinsic,
        ):
            raise ValueError("intrinsic route energy is not derived from coordinates")
        for target_index in range(len(self.carrier.row_keys)):
            selector = target_indices == target_index
            if not bool(selector.any()):
                raise ValueError("every target carrier row must own a route")
            fractions = self.route_energy_fraction[selector].to(torch.float64)
            intrinsic_total = expected_intrinsic[selector].sum()
            if float(intrinsic_total) <= 0.0:
                if float(self.carrier.fisher_energy[target_index]) > _ENERGY_ATOL:
                    raise ValueError(
                        "zero route energy is allowed only for zero common target energy"
                    )
                if float(fractions.sum()) != 0.0 or float(
                    self.route_allocated_target_fisher_energy[selector].sum()
                ) != 0.0:
                    raise ValueError("zero-energy targets require zero route allocation")
                continue
            expected_fractions = expected_intrinsic[selector] / intrinsic_total
            if not _energy_close(fractions, expected_fractions):
                raise ValueError("route energy fractions are not intrinsic-energy shares")
            if not math.isclose(
                float(fractions.sum()), 1.0, rel_tol=1.0e-9, abs_tol=_ENERGY_ATOL
            ):
                raise ValueError("route energy fractions must normalize per target")
            allocated = self.route_allocated_target_fisher_energy[selector].to(
                torch.float64
            )
            target_energy = float(self.carrier.fisher_energy[target_index])
            if not math.isclose(
                float(allocated.sum()),
                target_energy,
                rel_tol=1.0e-9,
                abs_tol=_ENERGY_ATOL,
            ):
                raise ValueError("route allocation must conserve target Fisher energy")
        object.__setattr__(self, "route_target_row_indices", target_indices)
        object.__setattr__(self, "route_coordinates", coordinates)
        for field_name in (
            "route_intrinsic_fisher_energy",
            "route_energy_fraction",
            "route_allocated_target_fisher_energy",
        ):
            object.__setattr__(
                self,
                field_name,
                _detached_cpu(getattr(self, field_name), label=field_name),
            )

    @property
    def route_row_catalog_sha256(self) -> str:
        return _row_catalog_sha256(
            tuple(row.state_dict() for row in self.route_row_keys)
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "carrier_sha256": self.carrier.sha256,
            "block_ordinal": self.block_ordinal,
            "context_binding_sha256": self.context_binding.sha256,
            "postnorm_bridge_sha256": self.postnorm_bridge.sha256,
            "route_row_catalog_sha256": self.route_row_catalog_sha256,
            "route_row_count": len(self.route_row_keys),
            "route_target_row_indices_sha256": _tensor_sha256(
                self.route_target_row_indices
            ),
            "route_coordinates_sha256": _tensor_sha256(self.route_coordinates),
            "route_intrinsic_fisher_energy_sha256": _tensor_sha256(
                self.route_intrinsic_fisher_energy
            ),
            "route_energy_fraction_sha256": _tensor_sha256(
                self.route_energy_fraction
            ),
            "route_allocated_target_fisher_energy_sha256": _tensor_sha256(
                self.route_allocated_target_fisher_energy
            ),
            "route_diagnostics_enter_common_distance": False,
        }

    @property
    def sha256(self) -> str:
        return _domain_sha256(_SURFACE_DOMAIN, self._hash_payload())

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "schema": GEMMA3_CHART_MODE_SURFACE_SCHEMA,
            "format_version": GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION,
            "surface_kind": "attention_route",
            "carrier": self.carrier.state_dict(),
            "context_binding": self.context_binding.state_dict(),
            "postnorm_bridge": self.postnorm_bridge.state_dict(),
            "route_row_keys": [row.state_dict() for row in self.route_row_keys],
            "route_target_row_indices": self.route_target_row_indices.detach().clone(),
            "route_coordinates": self.route_coordinates.detach().clone(),
            "route_intrinsic_fisher_energy": self.route_intrinsic_fisher_energy.detach().clone(),
            "route_energy_fraction": self.route_energy_fraction.detach().clone(),
            "route_allocated_target_fisher_energy": self.route_allocated_target_fisher_energy.detach().clone(),
            "surface_sha256": self.sha256,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }

    @classmethod
    def from_state_dict(cls, value: object) -> AttentionRouteChartSurface:
        tensor_fields = {
            "route_target_row_indices",
            "route_coordinates",
            "route_intrinsic_fisher_energy",
            "route_energy_fraction",
            "route_allocated_target_fisher_energy",
        }
        fields = {
            "schema",
            "format_version",
            "surface_kind",
            "carrier",
            "carrier_sha256",
            "block_ordinal",
            "context_binding",
            "context_binding_sha256",
            "postnorm_bridge",
            "postnorm_bridge_sha256",
            "route_row_keys",
            "route_row_catalog_sha256",
            "route_row_count",
            *tensor_fields,
            "route_target_row_indices_sha256",
            "route_coordinates_sha256",
            "route_intrinsic_fisher_energy_sha256",
            "route_energy_fraction_sha256",
            "route_allocated_target_fisher_energy_sha256",
            "route_diagnostics_enter_common_distance",
            "surface_sha256",
            "contains_token_ids",
            "contains_prompt_text",
        }
        state = _require_exact_fields(value, fields, label="attention route surface")
        if (
            state["schema"] != GEMMA3_CHART_MODE_SURFACE_SCHEMA
            or type(state["format_version"]) is not int
            or state["format_version"] != GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION
            or state["surface_kind"] != "attention_route"
            or state["route_diagnostics_enter_common_distance"] is not False
            or state["contains_token_ids"] is not False
            or state["contains_prompt_text"] is not False
        ):
            raise ValueError("attention route surface contract drifted")
        raw_rows = state["route_row_keys"]
        if not isinstance(raw_rows, list):
            raise ValueError("route_row_keys must be a list")
        parsed_route_rows: list[AttentionRouteRowKey] = []
        for index, raw_row in enumerate(raw_rows):
            row = _require_exact_fields(
                raw_row,
                {
                    "context_fingerprint_sha256",
                    "example_id",
                    "source_position",
                    "target_position",
                    "query_head",
                },
                label=f"route_row_keys[{index}]",
            )
            parsed_route_rows.append(
                AttentionRouteRowKey(
                context_fingerprint_sha256=_require_sha256(
                    row["context_fingerprint_sha256"],
                    label="route context fingerprint",
                ),
                example_id=_require_name(row["example_id"], label="route example_id"),
                source_position=_strict_int(
                    row["source_position"], label="source_position", minimum=0
                ),
                target_position=_strict_int(
                    row["target_position"], label="target_position", minimum=0
                ),
                query_head=_strict_int(
                    row["query_head"], label="query_head", minimum=0
                ),
                )
            )
        route_rows = tuple(parsed_route_rows)
        result = cls(
            carrier=CarrierChartSignature.from_state_dict(state["carrier"]),
            block_ordinal=_strict_int(
                state["block_ordinal"], label="block_ordinal", minimum=0
            ),
            context_binding=AttentionNativeContextBinding.from_state_dict(
                state["context_binding"]
            ),
            postnorm_bridge=AttentionPostNormBridgeReceipt.from_state_dict(
                state["postnorm_bridge"]
            ),
            route_row_keys=route_rows,
            route_target_row_indices=state["route_target_row_indices"],  # type: ignore[arg-type]
            route_coordinates=state["route_coordinates"],  # type: ignore[arg-type]
            route_intrinsic_fisher_energy=state[  # type: ignore[arg-type]
                "route_intrinsic_fisher_energy"
            ],
            route_energy_fraction=state["route_energy_fraction"],  # type: ignore[arg-type]
            route_allocated_target_fisher_energy=state[  # type: ignore[arg-type]
                "route_allocated_target_fisher_energy"
            ],
        )
        expected = result.state_dict()
        for key in fields - tensor_fields - {
            "carrier",
            "context_binding",
            "postnorm_bridge",
        }:
            if state[key] != expected[key]:
                raise ValueError(f"attention route {key} failed strict validation")
        return result


def build_attention_target_carrier_signature(
    *,
    mode_id: str,
    block_ordinal: int,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    source_state_sha256: str,
    execution_fingerprint_sha256: str,
    score_objective_receipt_sha256: str,
    attention_context_binding_sha256: str,
    postnorm_bridge: AttentionPostNormBridgeReceipt,
    example_ids: Sequence[str],
    query_valid_mask: Tensor,
    query_logical_positions: Tensor,
    target_carrier_values: Tensor,
    target_carrier_score_gradients: Tensor,
) -> CarrierChartSignature:
    """Build the production postnorm target signature without route materialization."""

    if type(block_ordinal) is not int or block_ordinal < 0:
        raise ValueError("block_ordinal must be nonnegative")
    if not mode_id.startswith("mode.mutation."):
        raise ValueError("attention target mode_id must be a mutation mode")
    _require_sha256(
        attention_context_binding_sha256,
        label="attention_context_binding_sha256",
    )
    if postnorm_bridge.execution_fingerprint_sha256 != execution_fingerprint_sha256:
        raise ValueError("attention bridge and target execution must match")
    target_values = _finite_float_tensor(
        target_carrier_values,
        label="target_carrier_values",
        ndim=3,
    )
    if _tensor_sha256(target_values) != postnorm_bridge.postnorm_delta_sha256:
        raise ValueError("postnorm target delta disagrees with bridge receipt")
    valid = _bool_tensor(
        query_valid_mask,
        label="query_valid_mask",
        ndim=2,
    )
    examples = _canonical_example_ids(example_ids, batch_size=int(valid.shape[0]))
    mapping_receipt = _domain_sha256(
        _MAPPING_DOMAIN,
        {
            "mapping": "attention_full_atom_postnorm_delta_to_common_carrier",
            "mode_id": mode_id,
            "block_ordinal": block_ordinal,
            "context_binding_sha256": attention_context_binding_sha256,
            "postnorm_bridge_sha256": postnorm_bridge.sha256,
            "native_abi_sha256": native_abi_sha256,
            "comparison_abi_sha256": comparison_abi_sha256,
            "score_objective_receipt_sha256": score_objective_receipt_sha256,
        },
    )
    return _build_carrier_signature(
        mode_id=mode_id,
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        mapping_receipt_sha256=mapping_receipt,
        source_state_sha256=source_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=score_objective_receipt_sha256,
        example_ids=examples,
        valid_mask=valid,
        logical_positions=query_logical_positions,
        dense_carrier_values=target_values,
        dense_carrier_score_gradients=target_carrier_score_gradients,
    )


def build_attention_route_chart_surface(
    *,
    mode_id: str,
    block_ordinal: int,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    source_state_sha256: str,
    execution_fingerprint_sha256: str,
    score_objective_receipt_sha256: str,
    example_ids: Sequence[str],
    context_fingerprints: Sequence[str],
    query_valid_mask: Tensor,
    key_valid_mask: Tensor,
    query_logical_positions: Tensor,
    key_logical_positions: Tensor,
    native_route_mask: Tensor,
    target_carrier_values: Tensor,
    target_carrier_score_gradients: Tensor,
    attention_operator_output: Tensor,
    attention_operator_output_score_gradients: Tensor,
    route_operator_output_values: Tensor,
    postnorm_state_sha256: str,
    postnorm_config_sha256: str,
    rope: NativeRoPESpec,
    window_size: int | None,
) -> AttentionRouteChartSurface:
    """Build one exact causal route chart without running an attention module."""

    query_valid = _bool_tensor(
        query_valid_mask,
        label="query_valid_mask",
        ndim=2,
    )
    batch_size, query_width = map(int, query_valid.shape)
    examples = _canonical_example_ids(example_ids, batch_size=batch_size)
    contexts = _canonical_context_fingerprints(
        context_fingerprints,
        batch_size=batch_size,
    )
    native = _bool_tensor(native_route_mask, label="native_route_mask", ndim=4)
    query_heads = int(native.shape[2])
    binding = AttentionNativeContextBinding(
        example_ids=examples,
        context_fingerprints=contexts,
        query_valid_mask=query_valid,
        key_valid_mask=_bool_tensor(key_valid_mask, label="key_valid_mask", ndim=2),
        query_logical_positions=_integer_tensor(
            query_logical_positions,
            label="query_logical_positions",
            ndim=2,
        ),
        key_logical_positions=_integer_tensor(
            key_logical_positions,
            label="key_logical_positions",
            ndim=2,
        ),
        native_route_mask=native,
        query_head_count=query_heads,
        window_size=window_size,
        rope=rope,
    )
    if int(native.sum()) > MAX_EXHAUSTIVE_ATTENTION_ROUTE_ROWS:
        raise ValueError(
            "exhaustive route surfaces are small-fixture-only; use "
            "AttentionRouteChartAccumulator for production panels"
        )
    bridge = build_attention_postnorm_bridge_receipt(
        operator_output=attention_operator_output,
        operator_output_score_gradient=attention_operator_output_score_gradients,
        postnorm_delta=target_carrier_values,
        postnorm_state_sha256=postnorm_state_sha256,
        postnorm_config_sha256=postnorm_config_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
    )
    carrier = build_attention_target_carrier_signature(
        mode_id=mode_id,
        block_ordinal=block_ordinal,
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        source_state_sha256=source_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=score_objective_receipt_sha256,
        attention_context_binding_sha256=binding.sha256,
        postnorm_bridge=bridge,
        example_ids=examples,
        query_valid_mask=query_valid,
        query_logical_positions=binding.query_logical_positions,
        target_carrier_values=target_carrier_values,
        target_carrier_score_gradients=target_carrier_score_gradients,
    )
    route_values = _finite_float_tensor(
        route_operator_output_values,
        label="route_operator_output_values",
        ndim=5,
    )
    expected_route_shape = (*native.shape, carrier.hidden_width)
    if tuple(route_values.shape) != expected_route_shape:
        raise ValueError(
            "route_operator_output_values must be [B,Q,query_head,K,H]"
        )
    reconstructed_operator = (
        route_values * native[..., None].to(route_values.dtype)
    ).sum(dim=(2, 3))
    operator_output = _finite_float_tensor(
        attention_operator_output,
        label="attention_operator_output",
        ndim=3,
    )
    closure_residual = reconstructed_operator.to(torch.float64) - operator_output.to(
        torch.float64
    )
    closure_max_abs = float(closure_residual.abs().max())
    closure_nrmse = float(
        closure_residual.square().mean().sqrt()
        / operator_output.to(torch.float64).square().mean().sqrt().clamp_min(1.0e-12)
    )
    closure_limit = 1.0e-4 + 1.0e-5 * float(operator_output.abs().max())
    if closure_nrmse > 1.0e-5 or closure_max_abs > closure_limit:
        raise ValueError("route operator values do not close to operator output")
    route_indices = native.nonzero(as_tuple=False)
    dense_gradients = _finite_float_tensor(
        attention_operator_output_score_gradients,
        label="attention_operator_output_score_gradients",
        ndim=3,
    )
    route_values_flat = route_values[native].contiguous()
    query_positions = binding.query_logical_positions
    key_positions = binding.key_logical_positions
    target_index_by_dense: dict[tuple[int, int], int] = {}
    for target_index, (batch, target) in enumerate(
        query_valid.nonzero(as_tuple=False).tolist()
    ):
        target_index_by_dense[(batch, target)] = target_index
    route_target_indices = torch.tensor(
        [target_index_by_dense[(int(row[0]), int(row[1]))] for row in route_indices],
        dtype=torch.int64,
    )
    route_rows = tuple(
        AttentionRouteRowKey(
            context_fingerprint_sha256=contexts[int(batch)],
            example_id=examples[int(batch)],
            source_position=int(key_positions[int(batch), int(source)]),
            target_position=int(query_positions[int(batch), int(target)]),
            query_head=int(head),
        )
        for batch, target, head, source in route_indices.tolist()
    )
    route_gradients = dense_gradients[
        route_indices[:, 0], route_indices[:, 1]
    ].contiguous()
    route_coordinates = (route_values_flat * route_gradients).contiguous()
    route64 = route_coordinates.to(torch.float64)
    intrinsic = route64.square().sum(dim=1).contiguous()
    fractions = torch.empty_like(intrinsic)
    allocated = torch.empty_like(intrinsic)
    for target_index in range(len(carrier.row_keys)):
        selector = route_target_indices == target_index
        subtotal = intrinsic[selector].sum()
        if float(subtotal) <= 0.0:
            if float(carrier.fisher_energy[target_index]) > _ENERGY_ATOL:
                raise ValueError(
                    "zero route energy is allowed only for zero common target energy"
                )
            fractions[selector] = 0.0
            allocated[selector] = 0.0
            continue
        fractions[selector] = intrinsic[selector] / subtotal
        allocated[selector] = fractions[selector] * carrier.fisher_energy[target_index]
    return AttentionRouteChartSurface(
        carrier=carrier,
        block_ordinal=block_ordinal,
        context_binding=binding,
        postnorm_bridge=bridge,
        route_row_keys=route_rows,
        route_target_row_indices=route_target_indices,
        route_coordinates=route_coordinates,
        route_intrinsic_fisher_energy=intrinsic,
        route_energy_fraction=fractions,
        route_allocated_target_fisher_energy=allocated,
    )


@dataclass(frozen=True, slots=True)
class LexicalIngressChartSurface:
    carrier: CarrierChartSignature
    lineage: TiedLexicalStateLineage
    context_catalog_sha256: str
    static_comparison_adapter_receipt_sha256: str
    panel_mapping_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.carrier.mode_id != "mode.lexical.ingress":
            raise ValueError("lexical ingress surface has the wrong mode_id")
        if self.carrier.hidden_width != self.lineage.hidden_width:
            raise ValueError("lexical ingress hidden width disagrees with tied lineage")
        _require_sha256(self.context_catalog_sha256, label="context_catalog_sha256")
        _require_sha256(
            self.static_comparison_adapter_receipt_sha256,
            label="static_comparison_adapter_receipt_sha256",
        )
        _require_sha256(
            self.panel_mapping_receipt_sha256,
            label="panel_mapping_receipt_sha256",
        )
        if (
            self.carrier.mapping_receipt_sha256
            != self.static_comparison_adapter_receipt_sha256
        ):
            raise ValueError(
                "lexical ingress carrier must bind the static comparison adapter"
            )

    @property
    def sha256(self) -> str:
        return _domain_sha256(
            _SURFACE_DOMAIN,
            {
                "surface_kind": "lexical_ingress",
                "carrier_sha256": self.carrier.sha256,
                "lineage_sha256": self.lineage.sha256,
                "context_catalog_sha256": self.context_catalog_sha256,
                "static_comparison_adapter_receipt_sha256": (
                    self.static_comparison_adapter_receipt_sha256
                ),
                "panel_mapping_receipt_sha256": (
                    self.panel_mapping_receipt_sha256
                ),
                "native_direction": "vocabulary_identity_to_initial_carrier",
            },
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": GEMMA3_CHART_MODE_SURFACE_SCHEMA,
            "format_version": GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION,
            "surface_kind": "lexical_ingress",
            "carrier": self.carrier.state_dict(),
            "carrier_sha256": self.carrier.sha256,
            "lineage": self.lineage.state_dict(),
            "lineage_sha256": self.lineage.sha256,
            "context_catalog_sha256": self.context_catalog_sha256,
            "static_comparison_adapter_receipt_sha256": (
                self.static_comparison_adapter_receipt_sha256
            ),
            "panel_mapping_receipt_sha256": (
                self.panel_mapping_receipt_sha256
            ),
            "native_direction": "vocabulary_identity_to_initial_carrier",
            "surface_sha256": self.sha256,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }

    @classmethod
    def from_state_dict(cls, value: object) -> LexicalIngressChartSurface:
        fields = {
            "schema",
            "format_version",
            "surface_kind",
            "carrier",
            "carrier_sha256",
            "lineage",
            "lineage_sha256",
            "context_catalog_sha256",
            "static_comparison_adapter_receipt_sha256",
            "panel_mapping_receipt_sha256",
            "native_direction",
            "surface_sha256",
            "contains_token_ids",
            "contains_prompt_text",
        }
        state = _require_exact_fields(value, fields, label="lexical ingress surface")
        if (
            state["schema"] != GEMMA3_CHART_MODE_SURFACE_SCHEMA
            or type(state["format_version"]) is not int
            or state["format_version"] != GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION
            or state["surface_kind"] != "lexical_ingress"
            or state["native_direction"] != "vocabulary_identity_to_initial_carrier"
            or state["contains_token_ids"] is not False
            or state["contains_prompt_text"] is not False
        ):
            raise ValueError("lexical ingress contract drifted")
        result = cls(
            carrier=CarrierChartSignature.from_state_dict(state["carrier"]),
            lineage=TiedLexicalStateLineage.from_state_dict(state["lineage"]),
            context_catalog_sha256=_require_sha256(
                state["context_catalog_sha256"], label="context_catalog_sha256"
            ),
            static_comparison_adapter_receipt_sha256=_require_sha256(
                state["static_comparison_adapter_receipt_sha256"],
                label="static_comparison_adapter_receipt_sha256",
            ),
            panel_mapping_receipt_sha256=_require_sha256(
                state["panel_mapping_receipt_sha256"],
                label="panel_mapping_receipt_sha256",
            ),
        )
        expected = result.state_dict()
        for key in fields - {"carrier", "lineage"}:
            if state[key] != expected[key]:
                raise ValueError(f"lexical ingress {key} failed strict validation")
        return result


def build_lexical_ingress_chart_surface(
    *,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    static_comparison_adapter_receipt_sha256: str,
    execution_fingerprint_sha256: str,
    score_objective_receipt_sha256: str,
    ingress_lookup_receipt_sha256: str,
    lineage: TiedLexicalStateLineage,
    example_ids: Sequence[str],
    context_fingerprints: Sequence[str],
    valid_mask: Tensor,
    logical_positions: Tensor,
    ingress_carrier_values: Tensor,
    ingress_carrier_score_gradients: Tensor,
) -> LexicalIngressChartSurface:
    _require_sha256(
        static_comparison_adapter_receipt_sha256,
        label="static_comparison_adapter_receipt_sha256",
    )
    _require_sha256(
        ingress_lookup_receipt_sha256,
        label="ingress_lookup_receipt_sha256",
    )
    valid = _bool_tensor(valid_mask, label="valid_mask", ndim=2)
    examples = _canonical_example_ids(example_ids, batch_size=int(valid.shape[0]))
    contexts = _canonical_context_fingerprints(
        context_fingerprints,
        batch_size=int(valid.shape[0]),
    )
    carrier_values = _finite_float_tensor(
        ingress_carrier_values,
        label="ingress_carrier_values",
        ndim=3,
    )
    if int(carrier_values.shape[2]) != lineage.hidden_width:
        raise ValueError("ingress carrier width disagrees with tied lineage")
    context_catalog = _domain_sha256(
        _ROW_DOMAIN,
        {
            "example_ids": list(examples),
            "context_fingerprints": list(contexts),
            "logical_positions_sha256": _tensor_sha256(logical_positions),
        },
    )
    carrier = _build_carrier_signature(
        mode_id="mode.lexical.ingress",
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        mapping_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        source_state_sha256=lineage.tied_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=score_objective_receipt_sha256,
        example_ids=examples,
        valid_mask=valid,
        logical_positions=logical_positions,
        dense_carrier_values=carrier_values,
        dense_carrier_score_gradients=ingress_carrier_score_gradients,
    )
    panel_mapping_receipt = _domain_sha256(
        _MAPPING_DOMAIN,
        {
            "carrier_score_gradient_sha256": _tensor_sha256(
                carrier.carrier_score_gradients
            ),
            "carrier_value_sha256": _tensor_sha256(
                carrier.carrier_values
            ),
            "comparison_abi_sha256": comparison_abi_sha256,
            "context_catalog_sha256": context_catalog,
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "ingress_lookup_receipt_sha256": (
                ingress_lookup_receipt_sha256
            ),
            "lineage_sha256": lineage.sha256,
            "mapping": "tied_embedding_output_to_common_initial_carrier",
            "native_abi_sha256": native_abi_sha256,
            "score_objective_receipt_sha256": (
                score_objective_receipt_sha256
            ),
            "static_comparison_adapter_receipt_sha256": (
                static_comparison_adapter_receipt_sha256
            ),
        },
    )
    return LexicalIngressChartSurface(
        carrier=carrier,
        lineage=lineage,
        context_catalog_sha256=context_catalog,
        static_comparison_adapter_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        panel_mapping_receipt_sha256=panel_mapping_receipt,
    )


def pull_back_post_softcap_logit_covector(
    *,
    terminal_carrier: Tensor,
    selected_readout_weight_rows: Tensor,
    post_softcap_logit_covector: Tensor,
    final_logit_softcap: float | None,
) -> Tensor:
    """Pull any selected post-softcap logit covector into carrier space.

    The selected rows may be the full vocabulary (the canonical next-token NLL
    score path) or a restricted-choice validation catalog.  Vocabulary indices
    are deliberately absent from this API and every returned surface.
    """

    carrier = _finite_float_tensor(
        terminal_carrier,
        label="terminal_carrier",
        ndim=3,
    )
    weights = _finite_float_tensor(
        selected_readout_weight_rows,
        label="selected_readout_weight_rows",
        ndim=3,
    )
    covector = _finite_float_tensor(
        post_softcap_logit_covector,
        label="post_softcap_logit_covector",
        ndim=3,
    )
    if (
        weights.shape[0] != carrier.shape[0]
        or weights.shape[2] != carrier.shape[2]
        or covector.shape
        != (carrier.shape[0], carrier.shape[1], weights.shape[1])
    ):
        raise ValueError("selected readout tensors do not align")
    raw_logits = torch.einsum("bsh,bch->bsc", carrier, weights)
    if final_logit_softcap is None:
        derivative = torch.ones_like(raw_logits)
    else:
        if (
            isinstance(final_logit_softcap, bool)
            or not isinstance(final_logit_softcap, (int, float))
            or not math.isfinite(float(final_logit_softcap))
            or float(final_logit_softcap) <= 0
        ):
            raise ValueError("final_logit_softcap must be finite and positive")
        cap = float(final_logit_softcap)
        derivative = 1.0 - torch.tanh(raw_logits / cap).square()
    return torch.einsum("bsc,bch->bsh", covector * derivative, weights).contiguous()


@dataclass(frozen=True, slots=True)
class LexicalEgressChartSurface:
    carrier: CarrierChartSignature
    lineage: TiedLexicalStateLineage
    context_catalog_sha256: str
    static_comparison_adapter_receipt_sha256: str
    panel_mapping_receipt_sha256: str
    terminal_norm_state_sha256: str
    readout_state_sha256: str
    objective_receipt_sha256: str
    readout_mapping_receipt_sha256: str
    score_gradient_sha256: str
    readout_pullback_sha256: str
    analytic_validation_receipt_sha256: str | None
    pullback_kind: str
    final_logit_softcap: float | None

    def __post_init__(self) -> None:
        if self.carrier.mode_id != "mode.lexical.egress":
            raise ValueError("lexical egress surface has the wrong mode_id")
        if self.carrier.hidden_width != self.lineage.hidden_width:
            raise ValueError("lexical egress hidden width disagrees with tied lineage")
        for label in (
            "context_catalog_sha256",
            "static_comparison_adapter_receipt_sha256",
            "panel_mapping_receipt_sha256",
            "terminal_norm_state_sha256",
            "readout_state_sha256",
            "objective_receipt_sha256",
            "readout_mapping_receipt_sha256",
            "score_gradient_sha256",
            "readout_pullback_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if (
            self.carrier.mapping_receipt_sha256
            != self.static_comparison_adapter_receipt_sha256
        ):
            raise ValueError(
                "lexical egress carrier must bind the static comparison adapter"
            )
        if self.pullback_kind not in {
            "full_next_token_nll_direct_terminal_gradient",
            "validation_selected_logit_covector",
        }:
            raise ValueError("pullback_kind is unsupported")
        if self.analytic_validation_receipt_sha256 is not None:
            _require_sha256(
                self.analytic_validation_receipt_sha256,
                label="analytic_validation_receipt_sha256",
            )
        if (
            self.pullback_kind == "validation_selected_logit_covector"
            and self.analytic_validation_receipt_sha256 is None
        ):
            raise ValueError("selected-logit validation requires an analytic receipt")
        if self.final_logit_softcap is not None and (
            not math.isfinite(self.final_logit_softcap)
            or self.final_logit_softcap <= 0
        ):
            raise ValueError("final_logit_softcap must be finite and positive")

    @property
    def sha256(self) -> str:
        return _domain_sha256(
            _SURFACE_DOMAIN,
            {
                "surface_kind": "lexical_egress",
                "carrier_sha256": self.carrier.sha256,
                "lineage_sha256": self.lineage.sha256,
                "context_catalog_sha256": self.context_catalog_sha256,
                "static_comparison_adapter_receipt_sha256": (
                    self.static_comparison_adapter_receipt_sha256
                ),
                "panel_mapping_receipt_sha256": (
                    self.panel_mapping_receipt_sha256
                ),
                "terminal_norm_state_sha256": self.terminal_norm_state_sha256,
                "readout_state_sha256": self.readout_state_sha256,
                "objective_receipt_sha256": self.objective_receipt_sha256,
                "readout_mapping_receipt_sha256": self.readout_mapping_receipt_sha256,
                "score_gradient_sha256": self.score_gradient_sha256,
                "readout_pullback_sha256": self.readout_pullback_sha256,
                "analytic_validation_receipt_sha256": self.analytic_validation_receipt_sha256,
                "pullback_kind": self.pullback_kind,
                "final_logit_softcap": self.final_logit_softcap,
                "native_direction": "terminal_carrier_to_vocabulary_logits",
                "common_mapping": "terminal_carrier_value_times_terminal_score_gradient",
            },
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": GEMMA3_CHART_MODE_SURFACE_SCHEMA,
            "format_version": GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION,
            "surface_kind": "lexical_egress",
            "carrier": self.carrier.state_dict(),
            "carrier_sha256": self.carrier.sha256,
            "lineage": self.lineage.state_dict(),
            "lineage_sha256": self.lineage.sha256,
            "context_catalog_sha256": self.context_catalog_sha256,
            "static_comparison_adapter_receipt_sha256": (
                self.static_comparison_adapter_receipt_sha256
            ),
            "panel_mapping_receipt_sha256": (
                self.panel_mapping_receipt_sha256
            ),
            "terminal_norm_state_sha256": self.terminal_norm_state_sha256,
            "readout_state_sha256": self.readout_state_sha256,
            "objective_receipt_sha256": self.objective_receipt_sha256,
            "readout_mapping_receipt_sha256": self.readout_mapping_receipt_sha256,
            "score_gradient_sha256": self.score_gradient_sha256,
            "readout_pullback_sha256": self.readout_pullback_sha256,
            "analytic_validation_receipt_sha256": self.analytic_validation_receipt_sha256,
            "pullback_kind": self.pullback_kind,
            "final_logit_softcap": self.final_logit_softcap,
            "native_direction": "terminal_carrier_to_vocabulary_logits",
            "common_mapping": "terminal_carrier_value_times_terminal_score_gradient",
            "surface_sha256": self.sha256,
            "contains_token_ids": False,
            "contains_prompt_text": False,
        }

    @classmethod
    def from_state_dict(cls, value: object) -> LexicalEgressChartSurface:
        fields = {
            "schema",
            "format_version",
            "surface_kind",
            "carrier",
            "carrier_sha256",
            "lineage",
            "lineage_sha256",
            "context_catalog_sha256",
            "static_comparison_adapter_receipt_sha256",
            "panel_mapping_receipt_sha256",
            "terminal_norm_state_sha256",
            "readout_state_sha256",
            "objective_receipt_sha256",
            "readout_mapping_receipt_sha256",
            "score_gradient_sha256",
            "readout_pullback_sha256",
            "analytic_validation_receipt_sha256",
            "pullback_kind",
            "final_logit_softcap",
            "native_direction",
            "common_mapping",
            "surface_sha256",
            "contains_token_ids",
            "contains_prompt_text",
        }
        state = _require_exact_fields(value, fields, label="lexical egress surface")
        if (
            state["schema"] != GEMMA3_CHART_MODE_SURFACE_SCHEMA
            or type(state["format_version"]) is not int
            or state["format_version"] != GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION
            or state["surface_kind"] != "lexical_egress"
            or state["native_direction"] != "terminal_carrier_to_vocabulary_logits"
            or state["common_mapping"]
            != "terminal_carrier_value_times_terminal_score_gradient"
            or state["contains_token_ids"] is not False
            or state["contains_prompt_text"] is not False
        ):
            raise ValueError("lexical egress contract drifted")
        result = cls(
            carrier=CarrierChartSignature.from_state_dict(state["carrier"]),
            lineage=TiedLexicalStateLineage.from_state_dict(state["lineage"]),
            context_catalog_sha256=_require_sha256(
                state["context_catalog_sha256"], label="context_catalog_sha256"
            ),
            static_comparison_adapter_receipt_sha256=_require_sha256(
                state["static_comparison_adapter_receipt_sha256"],
                label="static_comparison_adapter_receipt_sha256",
            ),
            panel_mapping_receipt_sha256=_require_sha256(
                state["panel_mapping_receipt_sha256"],
                label="panel_mapping_receipt_sha256",
            ),
            terminal_norm_state_sha256=_require_sha256(
                state["terminal_norm_state_sha256"],
                label="terminal_norm_state_sha256",
            ),
            readout_state_sha256=_require_sha256(
                state["readout_state_sha256"], label="readout_state_sha256"
            ),
            objective_receipt_sha256=_require_sha256(
                state["objective_receipt_sha256"], label="objective_receipt_sha256"
            ),
            readout_mapping_receipt_sha256=_require_sha256(
                state["readout_mapping_receipt_sha256"],
                label="readout_mapping_receipt_sha256",
            ),
            score_gradient_sha256=_require_sha256(
                state["score_gradient_sha256"], label="score_gradient_sha256"
            ),
            readout_pullback_sha256=_require_sha256(
                state["readout_pullback_sha256"], label="readout_pullback_sha256"
            ),
            analytic_validation_receipt_sha256=(
                None
                if state["analytic_validation_receipt_sha256"] is None
                else _require_sha256(
                    state["analytic_validation_receipt_sha256"],
                    label="analytic_validation_receipt_sha256",
                )
            ),
            pullback_kind=_require_name(
                state["pullback_kind"], label="pullback_kind"
            ),
            final_logit_softcap=(
                None
                if state["final_logit_softcap"] is None
                else _strict_float(
                    state["final_logit_softcap"],
                    label="final_logit_softcap",
                    strictly_positive=True,
                )
            ),
        )
        expected = result.state_dict()
        for key in fields - {"carrier", "lineage"}:
            if state[key] != expected[key]:
                raise ValueError(f"lexical egress {key} failed strict validation")
        return result


def build_lexical_egress_chart_surface(
    *,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    static_comparison_adapter_receipt_sha256: str,
    execution_fingerprint_sha256: str,
    lineage: TiedLexicalStateLineage,
    example_ids: Sequence[str],
    context_fingerprints: Sequence[str],
    valid_mask: Tensor,
    logical_positions: Tensor,
    terminal_carrier: Tensor,
    selected_readout_weight_rows: Tensor,
    post_softcap_logit_covector: Tensor,
    readout_row_catalog_sha256: str,
    objective_receipt_sha256: str,
    terminal_norm_state_sha256: str,
    readout_state_sha256: str,
    final_logit_softcap: float | None,
) -> LexicalEgressChartSurface:
    """Build a small selected-logit validation surface.

    The canonical full-NLL worker must use
    :func:`build_lexical_egress_from_terminal_score_gradient`; this analytic
    helper intentionally rejects full-vocabulary materialization.
    """

    _require_sha256(
        readout_row_catalog_sha256,
        label="readout_row_catalog_sha256",
    )
    _require_sha256(
        static_comparison_adapter_receipt_sha256,
        label="static_comparison_adapter_receipt_sha256",
    )
    _require_sha256(objective_receipt_sha256, label="objective_receipt_sha256")
    _require_sha256(terminal_norm_state_sha256, label="terminal_norm_state_sha256")
    _require_sha256(readout_state_sha256, label="readout_state_sha256")
    valid = _bool_tensor(valid_mask, label="valid_mask", ndim=2)
    examples = _canonical_example_ids(example_ids, batch_size=int(valid.shape[0]))
    contexts = _canonical_context_fingerprints(
        context_fingerprints,
        batch_size=int(valid.shape[0]),
    )
    terminal = _finite_float_tensor(
        terminal_carrier,
        label="terminal_carrier",
        ndim=3,
    )
    weights = _finite_float_tensor(
        selected_readout_weight_rows,
        label="selected_readout_weight_rows",
        ndim=3,
    )
    covector = _finite_float_tensor(
        post_softcap_logit_covector,
        label="post_softcap_logit_covector",
        ndim=3,
    )
    if int(terminal.shape[2]) != lineage.hidden_width:
        raise ValueError("terminal carrier width disagrees with tied lineage")
    if int(weights.shape[1]) >= lineage.vocabulary_size:
        raise ValueError("analytic selected-row validation must be smaller than vocabulary")
    if not bool(
        torch.allclose(
            covector.to(torch.float64).sum(dim=-1),
            torch.zeros(covector.shape[:2], dtype=torch.float64),
            rtol=1.0e-9,
            atol=1.0e-7,
        )
    ):
        raise ValueError("selected-logit validation covectors must be centered")
    pullback = pull_back_post_softcap_logit_covector(
        terminal_carrier=terminal,
        selected_readout_weight_rows=weights,
        post_softcap_logit_covector=covector,
        final_logit_softcap=final_logit_softcap,
    )
    context_catalog = _domain_sha256(
        _ROW_DOMAIN,
        {
            "example_ids": list(examples),
            "context_fingerprints": list(contexts),
            "logical_positions_sha256": _tensor_sha256(logical_positions),
        },
    )
    selected_rows_sha = _tensor_sha256(weights)
    covector_sha = _tensor_sha256(covector)
    readout_mapping_receipt = _domain_sha256(
        _MAPPING_DOMAIN,
        {
            "lineage_sha256": lineage.sha256,
            "readout_state_sha256": readout_state_sha256,
            "readout_row_catalog_sha256": readout_row_catalog_sha256,
            "selected_readout_rows_sha256": selected_rows_sha,
        },
    )
    readout_receipt = _domain_sha256(
        _READOUT_DOMAIN,
        {
            "lineage_sha256": lineage.sha256,
            "terminal_norm_state_sha256": terminal_norm_state_sha256,
            "readout_state_sha256": readout_state_sha256,
            "terminal_carrier_sha256": _tensor_sha256(terminal),
            "selected_readout_rows_sha256": selected_rows_sha,
            "post_softcap_covector_sha256": covector_sha,
            "readout_row_catalog_sha256": readout_row_catalog_sha256,
            "pullback_kind": "validation_selected_logit_covector",
            "final_logit_softcap": final_logit_softcap,
            "pullback_sha256": _tensor_sha256(pullback),
        },
    )
    panel_mapping_receipt = _domain_sha256(
        _MAPPING_DOMAIN,
        {
            "comparison_abi_sha256": comparison_abi_sha256,
            "context_catalog_sha256": context_catalog,
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "lineage_sha256": lineage.sha256,
            "mapping": "post_softcap_readout_score_pullback_to_common_terminal_carrier",
            "native_abi_sha256": native_abi_sha256,
            "objective_receipt_sha256": objective_receipt_sha256,
            "readout_pullback_sha256": readout_receipt,
            "static_comparison_adapter_receipt_sha256": (
                static_comparison_adapter_receipt_sha256
            ),
        },
    )
    carrier = _build_carrier_signature(
        mode_id="mode.lexical.egress",
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        mapping_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        source_state_sha256=lineage.tied_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=objective_receipt_sha256,
        example_ids=examples,
        valid_mask=valid,
        logical_positions=logical_positions,
        dense_carrier_values=terminal,
        dense_carrier_score_gradients=pullback,
    )
    return LexicalEgressChartSurface(
        carrier=carrier,
        lineage=lineage,
        context_catalog_sha256=context_catalog,
        static_comparison_adapter_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        panel_mapping_receipt_sha256=panel_mapping_receipt,
        terminal_norm_state_sha256=terminal_norm_state_sha256,
        readout_state_sha256=readout_state_sha256,
        objective_receipt_sha256=objective_receipt_sha256,
        readout_mapping_receipt_sha256=readout_mapping_receipt,
        score_gradient_sha256=_tensor_sha256(pullback),
        readout_pullback_sha256=readout_receipt,
        analytic_validation_receipt_sha256=readout_receipt,
        pullback_kind="validation_selected_logit_covector",
        final_logit_softcap=final_logit_softcap,
    )


def build_lexical_egress_from_terminal_score_gradient(
    *,
    native_abi_sha256: str,
    comparison_abi_sha256: str,
    static_comparison_adapter_receipt_sha256: str,
    execution_fingerprint_sha256: str,
    lineage: TiedLexicalStateLineage,
    example_ids: Sequence[str],
    context_fingerprints: Sequence[str],
    valid_mask: Tensor,
    logical_positions: Tensor,
    terminal_carrier: Tensor,
    terminal_carrier_score_gradient: Tensor,
    objective_receipt_sha256: str,
    terminal_norm_state_sha256: str,
    readout_state_sha256: str,
    readout_mapping_receipt_sha256: str,
    terminal_score_gradient_receipt_sha256: str,
    analytic_parity_receipt_sha256: str,
    final_logit_softcap: float | None,
) -> LexicalEgressChartSurface:
    """Build the scalable canonical egress chart from a direct terminal VJP.

    The caller authenticates the full next-token NLL objective and the direct
    autograd/VJP receipt.  This path never expands tied readout weights to
    ``[B,V,H]`` and never persists full-vocabulary logits or token ids.
    """

    for value, label in (
        (
            static_comparison_adapter_receipt_sha256,
            "static_comparison_adapter_receipt_sha256",
        ),
        (objective_receipt_sha256, "objective_receipt_sha256"),
        (terminal_norm_state_sha256, "terminal_norm_state_sha256"),
        (readout_state_sha256, "readout_state_sha256"),
        (readout_mapping_receipt_sha256, "readout_mapping_receipt_sha256"),
        (
            terminal_score_gradient_receipt_sha256,
            "terminal_score_gradient_receipt_sha256",
        ),
        (analytic_parity_receipt_sha256, "analytic_parity_receipt_sha256"),
    ):
        _require_sha256(value, label=label)
    if final_logit_softcap is not None and (
        isinstance(final_logit_softcap, bool)
        or not isinstance(final_logit_softcap, (int, float))
        or not math.isfinite(float(final_logit_softcap))
        or float(final_logit_softcap) <= 0
    ):
        raise ValueError("final_logit_softcap must be finite and positive")
    valid = _bool_tensor(valid_mask, label="valid_mask", ndim=2)
    examples = _canonical_example_ids(example_ids, batch_size=int(valid.shape[0]))
    contexts = _canonical_context_fingerprints(
        context_fingerprints,
        batch_size=int(valid.shape[0]),
    )
    terminal = _finite_float_tensor(
        terminal_carrier,
        label="terminal_carrier",
        ndim=3,
    )
    score_gradient = _finite_float_tensor(
        terminal_carrier_score_gradient,
        label="terminal_carrier_score_gradient",
        ndim=3,
    )
    if terminal.shape != score_gradient.shape:
        raise ValueError("terminal carrier and direct score gradient must align")
    if int(terminal.shape[2]) != lineage.hidden_width:
        raise ValueError("terminal carrier width disagrees with tied lineage")
    context_catalog = _domain_sha256(
        _ROW_DOMAIN,
        {
            "example_ids": list(examples),
            "context_fingerprints": list(contexts),
            "logical_positions_sha256": _tensor_sha256(logical_positions),
        },
    )
    score_gradient_sha = _tensor_sha256(score_gradient)
    readout_receipt = _domain_sha256(
        _READOUT_DOMAIN,
        {
            "pullback_kind": "full_next_token_nll_direct_terminal_gradient",
            "objective_receipt_sha256": objective_receipt_sha256,
            "lineage_sha256": lineage.sha256,
            "terminal_norm_state_sha256": terminal_norm_state_sha256,
            "readout_state_sha256": readout_state_sha256,
            "readout_mapping_receipt_sha256": readout_mapping_receipt_sha256,
            "terminal_score_gradient_receipt_sha256": (
                terminal_score_gradient_receipt_sha256
            ),
            "analytic_parity_receipt_sha256": analytic_parity_receipt_sha256,
            "terminal_carrier_sha256": _tensor_sha256(terminal),
            "terminal_score_gradient_sha256": score_gradient_sha,
            "final_logit_softcap": final_logit_softcap,
            "full_vocabulary_tensors_persisted": False,
        },
    )
    panel_mapping_receipt = _domain_sha256(
        _MAPPING_DOMAIN,
        {
            "comparison_abi_sha256": comparison_abi_sha256,
            "context_catalog_sha256": context_catalog,
            "execution_fingerprint_sha256": execution_fingerprint_sha256,
            "lineage_sha256": lineage.sha256,
            "mapping": "terminal_carrier_value_times_direct_nll_score_gradient",
            "native_abi_sha256": native_abi_sha256,
            "objective_receipt_sha256": objective_receipt_sha256,
            "readout_pullback_sha256": readout_receipt,
            "static_comparison_adapter_receipt_sha256": (
                static_comparison_adapter_receipt_sha256
            ),
        },
    )
    carrier = _build_carrier_signature(
        mode_id="mode.lexical.egress",
        native_abi_sha256=native_abi_sha256,
        comparison_abi_sha256=comparison_abi_sha256,
        mapping_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        source_state_sha256=lineage.tied_state_sha256,
        execution_fingerprint_sha256=execution_fingerprint_sha256,
        score_objective_receipt_sha256=objective_receipt_sha256,
        example_ids=examples,
        valid_mask=valid,
        logical_positions=logical_positions,
        dense_carrier_values=terminal,
        dense_carrier_score_gradients=score_gradient,
    )
    return LexicalEgressChartSurface(
        carrier=carrier,
        lineage=lineage,
        context_catalog_sha256=context_catalog,
        static_comparison_adapter_receipt_sha256=(
            static_comparison_adapter_receipt_sha256
        ),
        panel_mapping_receipt_sha256=panel_mapping_receipt,
        terminal_norm_state_sha256=terminal_norm_state_sha256,
        readout_state_sha256=readout_state_sha256,
        objective_receipt_sha256=objective_receipt_sha256,
        readout_mapping_receipt_sha256=readout_mapping_receipt_sha256,
        score_gradient_sha256=score_gradient_sha,
        readout_pullback_sha256=readout_receipt,
        analytic_validation_receipt_sha256=analytic_parity_receipt_sha256,
        pullback_kind="full_next_token_nll_direct_terminal_gradient",
        final_logit_softcap=final_logit_softcap,
    )


__all__ = [
    "COMMON_CARRIER_COMPARISON_ABI_NAME",
    "GEMMA3_CHART_MODE_SURFACE_FORMAT_VERSION",
    "GEMMA3_CHART_MODE_SURFACE_SCHEMA",
    "FACTORIZED_ATTENTION_ROUTE_KERNEL_ID",
    "FACTORIZED_ATTENTION_ROUTE_KEY_CHUNK_SIZE",
    "FACTORIZED_ATTENTION_ROUTE_QUERY_CHUNK_SIZE",
    "MAX_EXHAUSTIVE_ATTENTION_ROUTE_ROWS",
    "PINNED_CARRIER_HIDDEN_WIDTH",
    "AttentionNativeContextBinding",
    "AttentionPostNormBridgeReceipt",
    "AttentionRouteAggregateReceipt",
    "AttentionRouteChartAccumulator",
    "AttentionRouteChartSurface",
    "AttentionRouteRowKey",
    "AttentionRouteStreamContext",
    "CarrierChartSignature",
    "FactorizedAttentionRouteAggregateReceipt",
    "LexicalEgressChartSurface",
    "LexicalIngressChartSurface",
    "MutationCarrierChartSurface",
    "NativeRoPESpec",
    "TiedLexicalStateLineage",
    "build_attention_postnorm_bridge_receipt",
    "build_attention_route_chart_surface",
    "build_attention_target_carrier_signature",
    "build_factorized_attention_route_aggregates",
    "build_lexical_egress_chart_surface",
    "build_lexical_egress_from_terminal_score_gradient",
    "build_lexical_ingress_chart_surface",
    "build_mutation_carrier_chart_surface",
    "build_mutation_carrier_signature",
    "bind_carrier_signature_to_chart_mode_spec",
    "factorized_attention_route_kernel_receipt",
    "pull_back_post_softcap_logit_covector",
    "strict_load_attention_route_aggregate",
]
