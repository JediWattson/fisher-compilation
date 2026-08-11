"""HEAD-safe exact attention-edge extraction for parts-first Gemma 3.

The committed parts-first atom capture closes one scalar response per query
head.  This module refines that head response into its native token-to-token
routes without depending on an attention implementation hook or a stored
``[route, hidden]`` tensor.

For an exact, CPU-float32, bias-free :class:`Gemma3AttentionGenerator`, the
caller supplies the captured normalized attention input and the score gradient
at the post-attention RMSNorm delta.  The kernel then:

* replays the public exact generator and pulls the gradient back locally from
  its post-norm delta to its operator output;
* reconstructs Q/K/V, RoPE, grouped-query repetition, the ordinal causal or
  sliding mask, and the native float32 softmax from public coefficients and
  configuration;
* emits the full valid ``[batch, query-head, query, key]`` probability matrix;
* computes the additive signed route response ``R = P * <G, U>``;
* computes the cancellation-free intrinsic Fisher energy
  ``E = P^2 * sum_d((G_d * U_d)^2)``; and
* computes the softmax-logit tangent
  ``Z = R - P * sum_key(R)``.

Here ``U`` is the repeated value vector mapped through the corresponding
output-projection head slice, and ``G`` is the local operator-output score
gradient.  Causality and sliding windows are defined by tensor ordinals, as in
the committed Gemma adapter and exact generator.  Logical positions affect
RoPE and row identity only; they do not redefine visibility.

Returned arrays are stored as immutable bytes.  Tensor properties reconstruct
fresh defensive copies, so mutating a returned Tensor cannot change the
authenticated extraction.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import hashlib
import json
import math
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor

from .adapters.base import NormalizationSpec, SequenceContext
from .gemma3_attention_generator import (
    AttentionCoordinatePacket,
    Gemma3AttentionGenerator,
)


__all__ = [
    "ATTENTION_EDGE_PROBABILITY_FORMULA_ID",
    "PARTS_FIRST_ATTENTION_EDGE_FORMAT_VERSION",
    "PARTS_FIRST_ATTENTION_EDGE_SCHEMA",
    "AttentionEdgeClosureReceipt",
    "PartsFirstAttentionEdgeExtraction",
    "compile_parts_first_attention_edges",
]


PARTS_FIRST_ATTENTION_EDGE_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edges.v1"
)
PARTS_FIRST_ATTENTION_EDGE_FORMAT_VERSION: Final = 1
ATTENTION_EDGE_PROBABILITY_FORMULA_ID: Final = (
    "stored_float32_softmax.gamma32_support_plus_2_gamma64_plus_ftz.v1"
)

_TENSOR_DOMAIN = b"fisher_graph:parts-first-attention-edge-tensor:v1\0"
_CONTEXT_DOMAIN = b"fisher_graph:parts-first-attention-edge-context:v1\0"
_CLOSURE_DOMAIN = b"fisher_graph:parts-first-attention-edge-closure:v1\0"
_EXTRACTION_DOMAIN = b"fisher_graph:parts-first-attention-edge:v1\0"
_COMPILE_AUTHORITY: Final[object] = object()

_FLOAT32_UNIT_ROUNDOFF = 2.0**-24
_FLOAT64_UNIT_ROUNDOFF = 2.0**-53
_FLOAT32_MIN_NORMAL = 2.0**-126

_OPERATOR_ATOL = 2.0e-4
_OPERATOR_RTOL = 2.0e-5
_POSTNORM_ATOL = 2.0e-4
_POSTNORM_RTOL = 2.0e-5
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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor_bytes(value: Tensor) -> bytes:
    canonical = value.detach().to(device="cpu").contiguous()
    return canonical.view(torch.uint8).numpy().tobytes(order="C")


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(canonical.dtype),
                "shape": list(canonical.shape),
            }
        )
    )
    digest.update(b"\0")
    digest.update(_tensor_bytes(canonical))
    return digest.hexdigest()


def _tensor_from_bytes(
    payload: bytes,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    label: str,
) -> Tensor:
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} payload must be immutable bytes")
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = math.prod(shape)
    if len(payload) != element_count * element_size:
        raise ValueError(f"{label} byte length does not match its shape")
    # ``bytearray`` gives torch a writable temporary buffer.  The final clone
    # owns its storage, so neither the immutable source bytes nor a returned
    # Tensor aliases authenticated state.
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    return raw.view(dtype).reshape(shape).clone().contiguous()


def _finite_float_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.device.type != "cpu"
        or (shape is not None and tuple(value.shape) != shape)
        or (dtype is not None and value.dtype != dtype)
        or not bool(torch.isfinite(value).all())
    ):
        expected = "" if shape is None else f" with shape {shape}"
        expected_dtype = "" if dtype is None else f" and dtype {dtype}"
        raise ValueError(
            f"{label} must be a finite CPU floating tensor{expected}"
            f"{expected_dtype}"
        )
    return value.detach().clone().contiguous()


def _bool_tensor(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or value.device.type != "cpu"
        or (shape is not None and tuple(value.shape) != shape)
    ):
        expected = "" if shape is None else f" with shape {shape}"
        raise ValueError(f"{label} must be a CPU bool tensor{expected}")
    return value.detach().clone().contiguous()


def _closure_metrics(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float]:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    expected64 = expected.detach().to(device="cpu", dtype=torch.float64)
    if actual64.shape != expected64.shape or not actual64.numel():
        raise ValueError("closure tensors must be nonempty and shape-aligned")
    error = actual64 - expected64
    max_abs = float(error.abs().max().item())
    limit = float(atol + rtol * float(expected64.abs().max().item()))
    scale = expected64.square().mean().sqrt().clamp_min(float(atol))
    nrmse = float(error.square().mean().sqrt().div(scale).item())
    return max_abs, limit, nrmse


def _probability_forward_error_bound(support_count: int) -> float:
    if type(support_count) is not int or support_count <= 0:
        raise ValueError("softmax support_count must be positive")
    float32_terms = support_count + 2
    numerator32 = float32_terms * _FLOAT32_UNIT_ROUNDOFF
    numerator64 = (support_count - 1) * _FLOAT64_UNIT_ROUNDOFF
    if numerator32 >= 1.0 or numerator64 >= 1.0:
        raise ValueError("softmax support exceeds the frozen error model")
    gamma32 = numerator32 / (1.0 - numerator32)
    gamma64 = numerator64 / (1.0 - numerator64)
    return (
        gamma32
        + gamma64 * (1.0 + gamma32)
        + support_count * _FLOAT32_MIN_NORMAL
    )


def _ordinal_allowed_mask(
    sequence: SequenceContext,
    *,
    window_size: int | None,
) -> Tensor:
    query = torch.arange(
        sequence.query_length,
        dtype=torch.int64,
        device="cpu",
    ).view(1, sequence.query_length, 1)
    key = torch.arange(
        sequence.key_length,
        dtype=torch.int64,
        device="cpu",
    ).view(1, 1, sequence.key_length)
    allowed = (
        sequence.query_valid_mask.to(device="cpu")[:, :, None]
        & sequence.key_valid_mask.to(device="cpu")[:, None, :]
        & (key <= query)
    )
    if window_size is not None:
        allowed &= query - key < window_size
    return allowed.contiguous()


def _rms_norm(
    values: Tensor,
    weight: Tensor,
    spec: NormalizationSpec,
) -> Tensor:
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True)
        + float(spec.epsilon)
    )
    return (normalized * (1.0 + weight.float())).to(dtype=values.dtype)


def _rotate_half(values: Tensor) -> Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def _rope_cos_sin(
    positions: Tensor,
    *,
    head_dimension: int,
    theta: float,
    scaling_kind: str | None,
    scaling_factor: float | None,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    inverse_frequency = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dimension,
                2,
                dtype=torch.float32,
                device="cpu",
            )
            / head_dimension
        )
    )
    position_values = positions.to(device="cpu", dtype=torch.float32)
    if scaling_kind == "linear":
        if scaling_factor is None:
            raise ValueError("linear RoPE requires a scaling factor")
        position_values = position_values / float(scaling_factor)
    frequencies = torch.einsum(
        "bt,d->btd",
        position_values,
        inverse_frequency,
    )
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)


def _repeat_key_value(values: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return values
    batch, heads, length, width = values.shape
    return (
        values[:, :, None, :, :]
        .expand(batch, heads, repeats, length, width)
        .reshape(batch, heads * repeats, length, width)
    )


def _context_sha256(sequence: SequenceContext) -> str:
    origin = sequence.input_origin
    return _domain_sha256(
        _CONTEXT_DOMAIN,
        {
            "cache_positions_sha256": (
                None
                if sequence.cache_positions is None
                else _tensor_sha256(sequence.cache_positions)
            ),
            "cache_state_present": sequence.cache_state is not None,
            "input_origin": {
                "attention_mask_supplied": origin.attention_mask_supplied,
                "cache_positions_supplied": origin.cache_positions_supplied,
                "position_ids_supplied": origin.position_ids_supplied,
            },
            "key_logical_positions_sha256": _tensor_sha256(
                sequence.key_logical_positions
            ),
            "key_valid_mask_sha256": _tensor_sha256(
                sequence.key_valid_mask
            ),
            "phase": sequence.phase,
            "query_logical_positions_sha256": _tensor_sha256(
                sequence.logical_positions
            ),
            "query_valid_mask_sha256": _tensor_sha256(
                sequence.query_valid_mask
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class AttentionEdgeClosureReceipt:
    """Numerical and inventory closure for one full attention edge tensor."""

    probability_formula_id: str
    batch_size: int
    query_head_count: int
    query_length: int
    key_length: int
    hidden_width: int
    valid_query_count: int
    native_edge_count: int
    support_minimum: int
    support_maximum: int
    probability_mass_max_abs_error: float
    probability_mass_max_forward_error_bound: float
    masked_probability_max_abs: float
    operator_output_max_abs_error: float
    operator_output_max_abs_limit: float
    operator_output_nrmse: float
    postnorm_delta_max_abs_error: float
    postnorm_delta_max_abs_limit: float
    postnorm_delta_nrmse: float
    head_response_max_abs_error: float
    head_response_max_abs_limit: float
    head_response_nrmse: float
    total_response_max_abs_error: float
    total_response_max_abs_limit: float
    total_response_nrmse: float
    softmax_tangent_row_sum_max_abs: float
    softmax_tangent_row_sum_max_abs_limit: float
    intrinsic_fisher_energy_total: float
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.probability_formula_id != ATTENTION_EDGE_PROBABILITY_FORMULA_ID:
            raise ValueError("attention probability formula drifted")
        for label, value in (
            ("batch_size", self.batch_size),
            ("query_head_count", self.query_head_count),
            ("query_length", self.query_length),
            ("key_length", self.key_length),
            ("hidden_width", self.hidden_width),
            ("valid_query_count", self.valid_query_count),
            ("native_edge_count", self.native_edge_count),
            ("support_minimum", self.support_minimum),
            ("support_maximum", self.support_maximum),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.support_minimum > self.support_maximum:
            raise ValueError("attention support bounds are reversed")
        for label, value in (
            (
                "probability_mass_max_abs_error",
                self.probability_mass_max_abs_error,
            ),
            (
                "probability_mass_max_forward_error_bound",
                self.probability_mass_max_forward_error_bound,
            ),
            ("masked_probability_max_abs", self.masked_probability_max_abs),
            ("operator_output_max_abs_error", self.operator_output_max_abs_error),
            ("operator_output_max_abs_limit", self.operator_output_max_abs_limit),
            ("operator_output_nrmse", self.operator_output_nrmse),
            ("postnorm_delta_max_abs_error", self.postnorm_delta_max_abs_error),
            ("postnorm_delta_max_abs_limit", self.postnorm_delta_max_abs_limit),
            ("postnorm_delta_nrmse", self.postnorm_delta_nrmse),
            ("head_response_max_abs_error", self.head_response_max_abs_error),
            ("head_response_max_abs_limit", self.head_response_max_abs_limit),
            ("head_response_nrmse", self.head_response_nrmse),
            ("total_response_max_abs_error", self.total_response_max_abs_error),
            ("total_response_max_abs_limit", self.total_response_max_abs_limit),
            ("total_response_nrmse", self.total_response_nrmse),
            (
                "softmax_tangent_row_sum_max_abs",
                self.softmax_tangent_row_sum_max_abs,
            ),
            (
                "softmax_tangent_row_sum_max_abs_limit",
                self.softmax_tangent_row_sum_max_abs_limit,
            ),
            ("intrinsic_fisher_energy_total", self.intrinsic_fisher_energy_total),
        ):
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and nonnegative")
        if (
            self.probability_mass_max_abs_error
            > self.probability_mass_max_forward_error_bound
            or self.masked_probability_max_abs != 0.0
            or self.operator_output_max_abs_error
            > self.operator_output_max_abs_limit
            or self.postnorm_delta_max_abs_error
            > self.postnorm_delta_max_abs_limit
            or self.head_response_max_abs_error
            > self.head_response_max_abs_limit
            or self.total_response_max_abs_error
            > self.total_response_max_abs_limit
            or self.softmax_tangent_row_sum_max_abs
            > self.softmax_tangent_row_sum_max_abs_limit
        ):
            raise ValueError("attention edge numerical closure failed")
        computed = _domain_sha256(_CLOSURE_DOMAIN, self._payload())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif self.receipt_sha256 != computed:
            raise ValueError("attention edge closure receipt hash drifted")

    def _payload(self) -> dict[str, object]:
        return {
            "probability_formula_id": self.probability_formula_id,
            "batch_size": self.batch_size,
            "query_head_count": self.query_head_count,
            "query_length": self.query_length,
            "key_length": self.key_length,
            "hidden_width": self.hidden_width,
            "valid_query_count": self.valid_query_count,
            "native_edge_count": self.native_edge_count,
            "support_minimum": self.support_minimum,
            "support_maximum": self.support_maximum,
            "probability_mass_max_abs_error": (
                self.probability_mass_max_abs_error
            ),
            "probability_mass_max_forward_error_bound": (
                self.probability_mass_max_forward_error_bound
            ),
            "masked_probability_max_abs": self.masked_probability_max_abs,
            "operator_output_max_abs_error": self.operator_output_max_abs_error,
            "operator_output_max_abs_limit": self.operator_output_max_abs_limit,
            "operator_output_nrmse": self.operator_output_nrmse,
            "postnorm_delta_max_abs_error": self.postnorm_delta_max_abs_error,
            "postnorm_delta_max_abs_limit": self.postnorm_delta_max_abs_limit,
            "postnorm_delta_nrmse": self.postnorm_delta_nrmse,
            "head_response_max_abs_error": self.head_response_max_abs_error,
            "head_response_max_abs_limit": self.head_response_max_abs_limit,
            "head_response_nrmse": self.head_response_nrmse,
            "total_response_max_abs_error": self.total_response_max_abs_error,
            "total_response_max_abs_limit": self.total_response_max_abs_limit,
            "total_response_nrmse": self.total_response_nrmse,
            "softmax_tangent_row_sum_max_abs": (
                self.softmax_tangent_row_sum_max_abs
            ),
            "softmax_tangent_row_sum_max_abs_limit": (
                self.softmax_tangent_row_sum_max_abs_limit
            ),
            "intrinsic_fisher_energy_total": self.intrinsic_fisher_energy_total,
        }

    @property
    def passed(self) -> bool:
        return True

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}


def _validate_stored_tensor_closure(
    *,
    allowed: Tensor,
    probabilities: Tensor,
    responses: Tensor,
    energies: Tensor,
    tangents: Tensor,
    reconstructed_operator: Tensor,
    reference_operator: Tensor,
    reconstructed_postnorm_delta: Tensor,
    reference_postnorm_delta: Tensor,
    reference_head_response: Tensor,
    reference_total_response: Tensor,
    closure: AttentionEdgeClosureReceipt,
) -> None:
    """Re-derive every receipt relation available from stored edge tensors."""

    batch, heads, queries, keys = probabilities.shape
    valid_queries = allowed.any(dim=-1)
    support = allowed.sum(dim=-1)
    valid_support = support[valid_queries]
    if not valid_support.numel() or bool((valid_support <= 0).any()):
        raise ValueError("stored attention edge support is invalid")
    if (
        closure.valid_query_count != int(valid_queries.sum().item())
        or closure.native_edge_count != int(allowed.sum().item()) * heads
        or closure.support_minimum != int(valid_support.min().item())
        or closure.support_maximum != int(valid_support.max().item())
    ):
        raise ValueError("stored attention edge inventory drifted")

    probability64 = probabilities.to(torch.float64)
    expanded_valid_queries = valid_queries[:, None, :].expand(
        batch,
        heads,
        queries,
    )
    valid_mass = probability64.sum(dim=-1).masked_select(
        expanded_valid_queries
    )
    probability_error = float((valid_mass - 1.0).abs().max().item())
    probability_bound = _probability_forward_error_bound(
        int(valid_support.max().item())
    )
    if (
        closure.probability_mass_max_abs_error != probability_error
        or closure.probability_mass_max_forward_error_bound
        != probability_bound
    ):
        raise ValueError("stored attention probability receipt drifted")

    expanded_allowed = allowed[:, None, :, :].expand_as(probabilities)
    expected_tangents = (
        responses
        - probability64 * responses.sum(dim=-1, keepdim=True)
    ).masked_fill(~expanded_allowed, 0)
    if not torch.equal(tangents, expected_tangents):
        raise ValueError("stored attention softmax tangent relation drifted")

    valid_tangent_rows = tangents.sum(dim=-1).masked_select(
        expanded_valid_queries
    )
    tangent_error = float(valid_tangent_rows.abs().max().item())
    tangent_reduction_bound = (
        keys
        * _FLOAT64_UNIT_ROUNDOFF
        * float(tangents.abs().sum(dim=-1).max().item())
    )
    tangent_limit = float(
        responses.sum(dim=-1).abs().max().item() * probability_bound
        + tangent_reduction_bound
        + _FLOAT64_UNIT_ROUNDOFF
    )
    if (
        closure.softmax_tangent_row_sum_max_abs != tangent_error
        or closure.softmax_tangent_row_sum_max_abs_limit != tangent_limit
    ):
        raise ValueError("stored attention tangent receipt drifted")

    energy_total = float(energies.sum().item())
    if closure.intrinsic_fisher_energy_total != energy_total:
        raise ValueError("stored intrinsic Fisher energy receipt drifted")
    # Cauchy-Schwarz must hold route-wise because R sums the same hidden-width
    # coordinate products whose squared norm defines E.
    response_square = responses.square()
    cauchy_bound = float(closure.hidden_width) * energies
    cauchy_slack = (
        64.0
        * _FLOAT64_UNIT_ROUNDOFF
        * torch.maximum(cauchy_bound, torch.ones_like(cauchy_bound))
    )
    if bool((response_square > cauchy_bound + cauchy_slack).any()):
        raise ValueError("stored attention response/energy relation drifted")

    operator_metrics = _closure_metrics(
        reconstructed_operator,
        reference_operator,
        atol=_OPERATOR_ATOL,
        rtol=_OPERATOR_RTOL,
    )
    if operator_metrics != (
        closure.operator_output_max_abs_error,
        closure.operator_output_max_abs_limit,
        closure.operator_output_nrmse,
    ):
        raise ValueError("stored attention operator closure receipt drifted")
    postnorm_metrics = _closure_metrics(
        reconstructed_postnorm_delta,
        reference_postnorm_delta,
        atol=_POSTNORM_ATOL,
        rtol=_POSTNORM_RTOL,
    )
    if postnorm_metrics != (
        closure.postnorm_delta_max_abs_error,
        closure.postnorm_delta_max_abs_limit,
        closure.postnorm_delta_nrmse,
    ):
        raise ValueError("stored attention postnorm closure receipt drifted")
    head_metrics = _closure_metrics(
        responses.sum(dim=-1),
        reference_head_response,
        atol=_RESPONSE_ATOL,
        rtol=_RESPONSE_RTOL,
    )
    if head_metrics != (
        closure.head_response_max_abs_error,
        closure.head_response_max_abs_limit,
        closure.head_response_nrmse,
    ):
        raise ValueError("stored attention head response receipt drifted")
    total_metrics = _closure_metrics(
        responses.sum(dim=-1).sum(dim=1),
        reference_total_response,
        atol=_RESPONSE_ATOL,
        rtol=_RESPONSE_RTOL,
    )
    if total_metrics != (
        closure.total_response_max_abs_error,
        closure.total_response_max_abs_limit,
        closure.total_response_nrmse,
    ):
        raise ValueError("stored attention total response receipt drifted")


@dataclass(frozen=True, slots=True)
class PartsFirstAttentionEdgeExtraction:
    """Authenticated, byte-backed full attention-edge tensors."""

    generator_config_sha256: str
    generator_artifact_sha256: str
    sequence_context_sha256: str
    normalized_input_sha256: str
    postnorm_delta_score_gradient_sha256: str
    tensor_shape: tuple[int, int, int, int]
    allowed_mask_sha256: str
    probabilities_sha256: str
    signed_responses_sha256: str
    intrinsic_fisher_energies_sha256: str
    softmax_tangents_sha256: str
    reconstructed_operator_sha256: str
    reference_operator_sha256: str
    reconstructed_postnorm_delta_sha256: str
    reference_postnorm_delta_sha256: str
    reference_head_response_sha256: str
    reference_total_response_sha256: str
    closure: AttentionEdgeClosureReceipt
    _allowed_mask_bytes: bytes
    _probabilities_bytes: bytes
    _signed_responses_bytes: bytes
    _intrinsic_fisher_energies_bytes: bytes
    _softmax_tangents_bytes: bytes
    _reconstructed_operator_bytes: bytes
    _reference_operator_bytes: bytes
    _reconstructed_postnorm_delta_bytes: bytes
    _reference_postnorm_delta_bytes: bytes
    _reference_head_response_bytes: bytes
    _reference_total_response_bytes: bytes
    _compile_authority: InitVar[object | None] = None
    schema: str = PARTS_FIRST_ATTENTION_EDGE_SCHEMA
    format_version: int = PARTS_FIRST_ATTENTION_EDGE_FORMAT_VERSION
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    extraction_sha256: str = ""

    def __post_init__(self, _compile_authority: object | None) -> None:
        if (
            self.schema != PARTS_FIRST_ATTENTION_EDGE_SCHEMA
            or self.format_version != PARTS_FIRST_ATTENTION_EDGE_FORMAT_VERSION
            or self.contains_prompt_text is not False
            or self.contains_token_ids is not False
        ):
            raise ValueError("attention edge extraction header drifted")
        for label, value in (
            ("generator_config_sha256", self.generator_config_sha256),
            ("generator_artifact_sha256", self.generator_artifact_sha256),
            ("sequence_context_sha256", self.sequence_context_sha256),
            ("normalized_input_sha256", self.normalized_input_sha256),
            (
                "postnorm_delta_score_gradient_sha256",
                self.postnorm_delta_score_gradient_sha256,
            ),
            ("allowed_mask_sha256", self.allowed_mask_sha256),
            ("probabilities_sha256", self.probabilities_sha256),
            ("signed_responses_sha256", self.signed_responses_sha256),
            (
                "intrinsic_fisher_energies_sha256",
                self.intrinsic_fisher_energies_sha256,
            ),
            ("softmax_tangents_sha256", self.softmax_tangents_sha256),
            (
                "reconstructed_operator_sha256",
                self.reconstructed_operator_sha256,
            ),
            ("reference_operator_sha256", self.reference_operator_sha256),
            (
                "reconstructed_postnorm_delta_sha256",
                self.reconstructed_postnorm_delta_sha256,
            ),
            (
                "reference_postnorm_delta_sha256",
                self.reference_postnorm_delta_sha256,
            ),
            (
                "reference_head_response_sha256",
                self.reference_head_response_sha256,
            ),
            (
                "reference_total_response_sha256",
                self.reference_total_response_sha256,
            ),
        ):
            _require_sha256(value, label=label)
        if (
            not isinstance(self.tensor_shape, tuple)
            or len(self.tensor_shape) != 4
            or any(type(value) is not int or value <= 0 for value in self.tensor_shape)
        ):
            raise ValueError("attention edge tensor_shape must be four positive ints")
        batch, heads, queries, keys = self.tensor_shape
        if not isinstance(self.closure, AttentionEdgeClosureReceipt) or (
            self.closure.batch_size,
            self.closure.query_head_count,
            self.closure.query_length,
            self.closure.key_length,
        ) != (batch, heads, queries, keys):
            raise ValueError("attention edge closure dimensions drifted")
        allowed = _tensor_from_bytes(
            self._allowed_mask_bytes,
            dtype=torch.bool,
            shape=(batch, queries, keys),
            label="allowed_mask",
        )
        probabilities = _tensor_from_bytes(
            self._probabilities_bytes,
            dtype=torch.float32,
            shape=self.tensor_shape,
            label="probabilities",
        )
        responses = _tensor_from_bytes(
            self._signed_responses_bytes,
            dtype=torch.float64,
            shape=self.tensor_shape,
            label="signed_responses",
        )
        energies = _tensor_from_bytes(
            self._intrinsic_fisher_energies_bytes,
            dtype=torch.float64,
            shape=self.tensor_shape,
            label="intrinsic_fisher_energies",
        )
        tangents = _tensor_from_bytes(
            self._softmax_tangents_bytes,
            dtype=torch.float64,
            shape=self.tensor_shape,
            label="softmax_tangents",
        )
        operator_shape = (batch, queries, self.closure.hidden_width)
        reconstructed_operator = _tensor_from_bytes(
            self._reconstructed_operator_bytes,
            dtype=torch.float64,
            shape=operator_shape,
            label="reconstructed_operator",
        )
        reference_operator = _tensor_from_bytes(
            self._reference_operator_bytes,
            dtype=torch.float64,
            shape=operator_shape,
            label="reference_operator",
        )
        reconstructed_postnorm_delta = _tensor_from_bytes(
            self._reconstructed_postnorm_delta_bytes,
            dtype=torch.float32,
            shape=operator_shape,
            label="reconstructed_postnorm_delta",
        )
        reference_postnorm_delta = _tensor_from_bytes(
            self._reference_postnorm_delta_bytes,
            dtype=torch.float32,
            shape=operator_shape,
            label="reference_postnorm_delta",
        )
        reference_head_response = _tensor_from_bytes(
            self._reference_head_response_bytes,
            dtype=torch.float64,
            shape=(batch, heads, queries),
            label="reference_head_response",
        )
        reference_total_response = _tensor_from_bytes(
            self._reference_total_response_bytes,
            dtype=torch.float64,
            shape=(batch, queries),
            label="reference_total_response",
        )
        if (
            not bool(torch.isfinite(probabilities).all())
            or not bool(torch.isfinite(responses).all())
            or not bool(torch.isfinite(energies).all())
            or not bool(torch.isfinite(tangents).all())
            or not bool(torch.isfinite(reconstructed_operator).all())
            or not bool(torch.isfinite(reference_operator).all())
            or not bool(torch.isfinite(reconstructed_postnorm_delta).all())
            or not bool(torch.isfinite(reference_postnorm_delta).all())
            or not bool(torch.isfinite(reference_head_response).all())
            or not bool(torch.isfinite(reference_total_response).all())
            or bool((probabilities < 0).any())
            or bool((probabilities > 1).any())
            or bool((energies < 0).any())
        ):
            raise ValueError("attention edge tensor payload is invalid")
        expanded_allowed = allowed[:, None, :, :].expand_as(probabilities)
        if (
            bool((probabilities.masked_select(~expanded_allowed) != 0).any())
            or bool((responses.masked_select(~expanded_allowed) != 0).any())
            or bool((energies.masked_select(~expanded_allowed) != 0).any())
            or bool((tangents.masked_select(~expanded_allowed) != 0).any())
        ):
            raise ValueError("masked attention edge entries must be exactly zero")
        _validate_stored_tensor_closure(
            allowed=allowed,
            probabilities=probabilities,
            responses=responses,
            energies=energies,
            tangents=tangents,
            reconstructed_operator=reconstructed_operator,
            reference_operator=reference_operator,
            reconstructed_postnorm_delta=reconstructed_postnorm_delta,
            reference_postnorm_delta=reference_postnorm_delta,
            reference_head_response=reference_head_response,
            reference_total_response=reference_total_response,
            closure=self.closure,
        )
        for actual, expected, label in (
            (_tensor_sha256(allowed), self.allowed_mask_sha256, "allowed mask"),
            (
                _tensor_sha256(probabilities),
                self.probabilities_sha256,
                "probabilities",
            ),
            (
                _tensor_sha256(responses),
                self.signed_responses_sha256,
                "signed responses",
            ),
            (
                _tensor_sha256(energies),
                self.intrinsic_fisher_energies_sha256,
                "intrinsic Fisher energies",
            ),
            (
                _tensor_sha256(tangents),
                self.softmax_tangents_sha256,
                "softmax tangents",
            ),
            (
                _tensor_sha256(reconstructed_operator),
                self.reconstructed_operator_sha256,
                "reconstructed operator",
            ),
            (
                _tensor_sha256(reference_operator),
                self.reference_operator_sha256,
                "reference operator",
            ),
            (
                _tensor_sha256(reconstructed_postnorm_delta),
                self.reconstructed_postnorm_delta_sha256,
                "reconstructed postnorm delta",
            ),
            (
                _tensor_sha256(reference_postnorm_delta),
                self.reference_postnorm_delta_sha256,
                "reference postnorm delta",
            ),
            (
                _tensor_sha256(reference_head_response),
                self.reference_head_response_sha256,
                "reference head response",
            ),
            (
                _tensor_sha256(reference_total_response),
                self.reference_total_response_sha256,
                "reference total response",
            ),
        ):
            if actual != expected:
                raise ValueError(f"attention edge {label} hash drifted")
        computed = _domain_sha256(_EXTRACTION_DOMAIN, self._hash_payload())
        if self.extraction_sha256 == "":
            object.__setattr__(self, "extraction_sha256", computed)
        elif self.extraction_sha256 != computed:
            raise ValueError("attention edge extraction hash drifted")
        if _compile_authority is not _COMPILE_AUTHORITY:
            raise ValueError(
                "attention edge extractions may only be built by "
                "compile_parts_first_attention_edges"
            )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "generator_config_sha256": self.generator_config_sha256,
            "generator_artifact_sha256": self.generator_artifact_sha256,
            "sequence_context_sha256": self.sequence_context_sha256,
            "normalized_input_sha256": self.normalized_input_sha256,
            "postnorm_delta_score_gradient_sha256": (
                self.postnorm_delta_score_gradient_sha256
            ),
            "tensor_shape": list(self.tensor_shape),
            "allowed_mask_sha256": self.allowed_mask_sha256,
            "probabilities_sha256": self.probabilities_sha256,
            "signed_responses_sha256": self.signed_responses_sha256,
            "intrinsic_fisher_energies_sha256": (
                self.intrinsic_fisher_energies_sha256
            ),
            "softmax_tangents_sha256": self.softmax_tangents_sha256,
            "reconstructed_operator_sha256": (
                self.reconstructed_operator_sha256
            ),
            "reference_operator_sha256": self.reference_operator_sha256,
            "reconstructed_postnorm_delta_sha256": (
                self.reconstructed_postnorm_delta_sha256
            ),
            "reference_postnorm_delta_sha256": (
                self.reference_postnorm_delta_sha256
            ),
            "reference_head_response_sha256": (
                self.reference_head_response_sha256
            ),
            "reference_total_response_sha256": (
                self.reference_total_response_sha256
            ),
            "closure_receipt_sha256": self.closure.receipt_sha256,
            "contains_prompt_text": self.contains_prompt_text,
            "contains_token_ids": self.contains_token_ids,
        }

    @property
    def sha256(self) -> str:
        return self.extraction_sha256

    @property
    def allowed_mask(self) -> Tensor:
        batch, _heads, queries, keys = self.tensor_shape
        return _tensor_from_bytes(
            self._allowed_mask_bytes,
            dtype=torch.bool,
            shape=(batch, queries, keys),
            label="allowed_mask",
        )

    @property
    def probabilities(self) -> Tensor:
        return _tensor_from_bytes(
            self._probabilities_bytes,
            dtype=torch.float32,
            shape=self.tensor_shape,
            label="probabilities",
        )

    @property
    def signed_responses(self) -> Tensor:
        return _tensor_from_bytes(
            self._signed_responses_bytes,
            dtype=torch.float64,
            shape=self.tensor_shape,
            label="signed_responses",
        )

    @property
    def intrinsic_fisher_energies(self) -> Tensor:
        return _tensor_from_bytes(
            self._intrinsic_fisher_energies_bytes,
            dtype=torch.float64,
            shape=self.tensor_shape,
            label="intrinsic_fisher_energies",
        )

    @property
    def softmax_tangents(self) -> Tensor:
        return _tensor_from_bytes(
            self._softmax_tangents_bytes,
            dtype=torch.float64,
            shape=self.tensor_shape,
            label="softmax_tangents",
        )

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "extraction_sha256": self.extraction_sha256,
            "closure": self.closure.state_dict(),
            "full_qk_probability_matrix_present": True,
            "full_qk_signed_response_matrix_present": True,
            "full_qk_intrinsic_fisher_energy_matrix_present": True,
            "full_qk_softmax_tangent_matrix_present": True,
            "full_width_route_vectors_materialized": False,
            "compact_closure_witness_tensors_present": True,
            "closure_witness_payloads_exposed": False,
            "causal_visibility_uses_tensor_ordinals": True,
            "logical_positions_define_visibility": False,
        }

    @classmethod
    def _from_tensors(
        cls,
        *,
        generator: Gemma3AttentionGenerator,
        sequence_context_sha256: str,
        normalized_input_sha256: str,
        postnorm_delta_score_gradient_sha256: str,
        allowed_mask: Tensor,
        probabilities: Tensor,
        signed_responses: Tensor,
        intrinsic_fisher_energies: Tensor,
        softmax_tangents: Tensor,
        reconstructed_operator: Tensor,
        reference_operator: Tensor,
        reconstructed_postnorm_delta: Tensor,
        reference_postnorm_delta: Tensor,
        reference_head_response: Tensor,
        reference_total_response: Tensor,
        closure: AttentionEdgeClosureReceipt,
    ) -> "PartsFirstAttentionEdgeExtraction":
        return cls(
            generator_config_sha256=generator.config.config_sha256,
            generator_artifact_sha256=generator.artifact_sha256,
            sequence_context_sha256=sequence_context_sha256,
            normalized_input_sha256=normalized_input_sha256,
            postnorm_delta_score_gradient_sha256=(
                postnorm_delta_score_gradient_sha256
            ),
            tensor_shape=tuple(probabilities.shape),  # type: ignore[arg-type]
            allowed_mask_sha256=_tensor_sha256(allowed_mask),
            probabilities_sha256=_tensor_sha256(probabilities),
            signed_responses_sha256=_tensor_sha256(signed_responses),
            intrinsic_fisher_energies_sha256=_tensor_sha256(
                intrinsic_fisher_energies
            ),
            softmax_tangents_sha256=_tensor_sha256(softmax_tangents),
            reconstructed_operator_sha256=_tensor_sha256(
                reconstructed_operator
            ),
            reference_operator_sha256=_tensor_sha256(reference_operator),
            reconstructed_postnorm_delta_sha256=_tensor_sha256(
                reconstructed_postnorm_delta
            ),
            reference_postnorm_delta_sha256=_tensor_sha256(
                reference_postnorm_delta
            ),
            reference_head_response_sha256=_tensor_sha256(
                reference_head_response
            ),
            reference_total_response_sha256=_tensor_sha256(
                reference_total_response
            ),
            closure=closure,
            _allowed_mask_bytes=_tensor_bytes(allowed_mask),
            _probabilities_bytes=_tensor_bytes(probabilities),
            _signed_responses_bytes=_tensor_bytes(signed_responses),
            _intrinsic_fisher_energies_bytes=_tensor_bytes(
                intrinsic_fisher_energies
            ),
            _softmax_tangents_bytes=_tensor_bytes(softmax_tangents),
            _reconstructed_operator_bytes=_tensor_bytes(
                reconstructed_operator
            ),
            _reference_operator_bytes=_tensor_bytes(reference_operator),
            _reconstructed_postnorm_delta_bytes=_tensor_bytes(
                reconstructed_postnorm_delta
            ),
            _reference_postnorm_delta_bytes=_tensor_bytes(
                reference_postnorm_delta
            ),
            _reference_head_response_bytes=_tensor_bytes(
                reference_head_response
            ),
            _reference_total_response_bytes=_tensor_bytes(
                reference_total_response
            ),
            _compile_authority=_COMPILE_AUTHORITY,
        )


def _validate_kernel_inputs(
    generator: Gemma3AttentionGenerator,
    normalized_input: Tensor,
    postnorm_delta_score_gradient: Tensor,
    sequence: SequenceContext,
) -> tuple[Tensor, Tensor]:
    if not isinstance(generator, Gemma3AttentionGenerator):
        raise TypeError("generator must be a Gemma3AttentionGenerator")
    if generator.generator_kind != "exact_full_rank":
        raise ValueError("attention edges require an exact_full_rank generator")
    if generator.device.type != "cpu" or generator.dtype != torch.float32:
        raise ValueError("attention edges require an exact CPU-float32 generator")
    if generator.training or any(
        parameter.requires_grad for parameter in generator.parameters()
    ):
        raise ValueError("attention edge generator must be frozen in eval mode")
    if (
        generator.config.projection_bias
        or generator.q_bias is not None
        or generator.k_bias is not None
        or generator.v_bias is not None
        or generator.o_bias is not None
    ):
        raise ValueError("attention edges require bias-free Q/K/V/O projections")
    if float(generator.config.dropout) != 0.0:
        raise ValueError("attention edges require zero attention dropout")
    if not isinstance(sequence, SequenceContext):
        raise TypeError("sequence must be a SequenceContext")
    if sequence.device.type != "cpu":
        raise ValueError("attention edge sequence must be on CPU")
    # Public replay owns the authoritative sequence validation.
    expected = (
        sequence.batch_size,
        sequence.query_length,
        generator.config.residual_width,
    )
    normalized = _finite_float_tensor(
        normalized_input,
        label="normalized_input",
        shape=expected,
        dtype=torch.float32,
    )
    gradient = _finite_float_tensor(
        postnorm_delta_score_gradient,
        label="postnorm_delta_score_gradient",
        shape=expected,
        dtype=torch.float32,
    )
    return normalized, gradient


def compile_parts_first_attention_edges(
    generator: Gemma3AttentionGenerator,
    normalized_input: Tensor,
    postnorm_delta_score_gradient: Tensor,
    sequence: SequenceContext,
) -> PartsFirstAttentionEdgeExtraction:
    """Extract all valid exact attention edges for one captured invocation.

    ``normalized_input`` is the full ``mutation.<odd>.input`` snapshot from
    the committed 73-site observer. ``postnorm_delta_score_gradient`` is the
    matching entry returned by ``JointGraphEventGradientTape.score_gradients``
    for ``mutation.<odd>.output``.
    """

    normalized, delta_gradient = _validate_kernel_inputs(
        generator,
        normalized_input,
        postnorm_delta_score_gradient,
        sequence,
    )
    valid_queries = sequence.query_valid_mask.detach().to(device="cpu")
    effective_input = normalized.masked_fill(~valid_queries.unsqueeze(-1), 0)

    # One exact public replay is both the operator-output reference and the
    # local post-norm graph used for the VJP.  No source model or private
    # attention hook participates.
    replay_input = effective_input.detach().clone().requires_grad_(True)
    packet = AttentionCoordinatePacket(
        values=replay_input,
        valid_token_mask=valid_queries.clone(),
        coordinate_system_sha256=generator.input_coordinate_system_sha256,
        semantic_role="normalized_input",
        source_invocation_id=generator.config.invocation_id,
    )
    with torch.enable_grad():
        replay = generator.forward_from_coordinates(packet, sequence)
        operator_gradient = torch.autograd.grad(
            replay.residual_delta,
            replay.operator_output,
            grad_outputs=delta_gradient,
            allow_unused=False,
            retain_graph=False,
            create_graph=False,
        )[0]
    reference_operator = replay.operator_output.detach().to(torch.float32)
    reference_delta = replay.residual_delta.detach().to(torch.float32)
    operator_gradient = operator_gradient.detach().to(torch.float32)

    attention = generator.config.attention
    batch = sequence.batch_size
    queries = sequence.query_length
    keys = sequence.key_length
    heads = attention.query_heads
    key_value_heads = attention.key_value_heads
    head_dimension = attention.head_dimension

    with torch.no_grad():
        query = F.linear(effective_input, generator.q_weight).view(
            batch,
            queries,
            heads,
            head_dimension,
        ).transpose(1, 2)
        key = F.linear(effective_input, generator.k_weight).view(
            batch,
            keys,
            key_value_heads,
            head_dimension,
        ).transpose(1, 2)
        value = F.linear(effective_input, generator.v_weight).view(
            batch,
            keys,
            key_value_heads,
            head_dimension,
        ).transpose(1, 2)
        if attention.qk_norm:
            if (
                generator.q_norm_weight is None
                or generator.k_norm_weight is None
                or generator.config.qk_norm is None
            ):
                raise RuntimeError("exact attention omitted Q/K norm coefficients")
            query = _rms_norm(
                query,
                generator.q_norm_weight,
                generator.config.qk_norm,
            )
            key = _rms_norm(
                key,
                generator.k_norm_weight,
                generator.config.qk_norm,
            )
        rope = attention.rope
        if rope is None or rope.theta is None:
            raise RuntimeError("exact attention omitted RoPE semantics")
        cos, sin = _rope_cos_sin(
            sequence.logical_positions,
            head_dimension=head_dimension,
            theta=float(rope.theta),
            scaling_kind=rope.scaling_kind,
            scaling_factor=rope.scaling_factor,
            dtype=torch.float32,
        )
        query = query * cos.unsqueeze(1) + _rotate_half(query) * sin.unsqueeze(1)
        key = key * cos.unsqueeze(1) + _rotate_half(key) * sin.unsqueeze(1)
        repeats = heads // key_value_heads
        repeated_key = _repeat_key_value(key, repeats)
        repeated_value = _repeat_key_value(value, repeats)
        scores = torch.matmul(
            query,
            repeated_key.transpose(-2, -1),
        ) * float(attention.query_scale)
        if generator.config.logit_softcap is not None:
            cap = float(generator.config.logit_softcap)
            scores = torch.tanh(scores / cap) * cap
        allowed = _ordinal_allowed_mask(
            sequence,
            window_size=attention.window_size,
        )
        expanded_allowed = allowed[:, None, :, :].expand(
            batch,
            heads,
            queries,
            keys,
        )
        scores = scores.masked_fill(
            ~expanded_allowed,
            torch.finfo(torch.float32).min,
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
        # Invalid query rows have no semantic simplex in the exact generator.
        # Mask every forbidden route after the native softmax without
        # renormalizing stored bytes.
        probabilities = probabilities.masked_fill(~expanded_allowed, 0).contiguous()

        output_slices = generator.output_decoder_weight.detach().reshape(
            generator.config.residual_width,
            heads,
            head_dimension,
        ).permute(1, 2, 0).contiguous()
        projected_values = torch.einsum(
            "bhkd,hdo->bhko",
            repeated_value,
            output_slices,
        ).contiguous()

    probability64 = probabilities.to(torch.float64)
    projected64 = projected_values.to(torch.float64)
    gradient64 = operator_gradient.to(torch.float64)
    unit_signed = torch.einsum(
        "bqo,bhko->bhqk",
        gradient64,
        projected64,
    )
    signed_responses = (probability64 * unit_signed).masked_fill(
        ~expanded_allowed,
        0,
    ).contiguous()
    unit_energy = torch.einsum(
        "bqo,bhko->bhqk",
        gradient64.square(),
        projected64.square(),
    )
    intrinsic_energies = (probability64.square() * unit_energy).masked_fill(
        ~expanded_allowed,
        0,
    ).contiguous()
    head_response = signed_responses.sum(dim=-1)
    softmax_tangents = (
        signed_responses - probability64 * head_response.unsqueeze(-1)
    ).masked_fill(~expanded_allowed, 0).contiguous()

    reconstructed_operator = torch.einsum(
        "bhqk,bhko->bqo",
        probability64,
        projected64,
    ).contiguous()
    expected_operator = reference_operator.to(torch.float64).masked_fill(
        ~valid_queries.unsqueeze(-1),
        0,
    )
    reconstructed_operator = reconstructed_operator.masked_fill(
        ~valid_queries.unsqueeze(-1),
        0,
    )
    operator_metrics = _closure_metrics(
        reconstructed_operator,
        expected_operator,
        atol=_OPERATOR_ATOL,
        rtol=_OPERATOR_RTOL,
    )
    reconstructed_delta = _rms_norm(
        reconstructed_operator.to(torch.float32),
        generator.output_norm_weight.detach(),
        generator.config.output_norm,
    ).masked_fill(~valid_queries.unsqueeze(-1), 0)
    postnorm_metrics = _closure_metrics(
        reconstructed_delta,
        reference_delta,
        atol=_POSTNORM_ATOL,
        rtol=_POSTNORM_RTOL,
    )

    native_head_context = torch.matmul(
        probabilities,
        repeated_value,
    )
    native_head_output = torch.einsum(
        "bhqd,hdo->bhqo",
        native_head_context,
        output_slices,
    ).to(torch.float64)
    reference_head_response = torch.einsum(
        "bhqo,bqo->bhq",
        native_head_output,
        gradient64,
    ).masked_fill(~valid_queries[:, None, :], 0)
    head_metrics = _closure_metrics(
        head_response,
        reference_head_response,
        atol=_RESPONSE_ATOL,
        rtol=_RESPONSE_RTOL,
    )
    total_response = head_response.sum(dim=1)
    reference_total_response = (
        expected_operator * gradient64
    ).sum(dim=-1).masked_fill(~valid_queries, 0)
    total_metrics = _closure_metrics(
        total_response,
        reference_total_response,
        atol=_RESPONSE_ATOL,
        rtol=_RESPONSE_RTOL,
    )

    support = allowed.sum(dim=-1)
    valid_support = support[valid_queries]
    if not valid_support.numel() or bool((valid_support <= 0).any()):
        raise ValueError("every valid attention query must have positive support")
    probability_mass = probability64.sum(dim=-1)
    valid_mass = probability_mass.masked_select(
        valid_queries[:, None, :].expand(batch, heads, queries)
    )
    probability_error = float((valid_mass - 1.0).abs().max().item())
    probability_bound = _probability_forward_error_bound(
        int(valid_support.max().item())
    )
    masked_values = probabilities.masked_select(~expanded_allowed)
    masked_max = (
        0.0
        if not masked_values.numel()
        else float(masked_values.abs().max().item())
    )
    valid_tangent_rows = softmax_tangents.sum(dim=-1).masked_select(
        valid_queries[:, None, :].expand(batch, heads, queries)
    )
    tangent_error = float(valid_tangent_rows.abs().max().item())
    # Stored float32 softmax rows are intentionally not renormalized.  The
    # algebraic tangent row sum is therefore bounded by the head response
    # times the frozen simplex forward-error bound, plus float64 reduction
    # roundoff over the materialized tangent row.
    tangent_reduction_bound = (
        keys
        * _FLOAT64_UNIT_ROUNDOFF
        * float(softmax_tangents.abs().sum(dim=-1).max().item())
    )
    tangent_limit = float(
        head_response.abs().max().item() * probability_bound
        + tangent_reduction_bound
        + _FLOAT64_UNIT_ROUNDOFF
    )

    closure = AttentionEdgeClosureReceipt(
        probability_formula_id=ATTENTION_EDGE_PROBABILITY_FORMULA_ID,
        batch_size=batch,
        query_head_count=heads,
        query_length=queries,
        key_length=keys,
        hidden_width=generator.config.residual_width,
        valid_query_count=int(valid_queries.sum().item()),
        native_edge_count=int(allowed.sum().item()) * heads,
        support_minimum=int(valid_support.min().item()),
        support_maximum=int(valid_support.max().item()),
        probability_mass_max_abs_error=probability_error,
        probability_mass_max_forward_error_bound=probability_bound,
        masked_probability_max_abs=masked_max,
        operator_output_max_abs_error=operator_metrics[0],
        operator_output_max_abs_limit=operator_metrics[1],
        operator_output_nrmse=operator_metrics[2],
        postnorm_delta_max_abs_error=postnorm_metrics[0],
        postnorm_delta_max_abs_limit=postnorm_metrics[1],
        postnorm_delta_nrmse=postnorm_metrics[2],
        head_response_max_abs_error=head_metrics[0],
        head_response_max_abs_limit=head_metrics[1],
        head_response_nrmse=head_metrics[2],
        total_response_max_abs_error=total_metrics[0],
        total_response_max_abs_limit=total_metrics[1],
        total_response_nrmse=total_metrics[2],
        softmax_tangent_row_sum_max_abs=tangent_error,
        softmax_tangent_row_sum_max_abs_limit=tangent_limit,
        intrinsic_fisher_energy_total=float(intrinsic_energies.sum().item()),
    )
    return PartsFirstAttentionEdgeExtraction._from_tensors(
        generator=generator,
        sequence_context_sha256=_context_sha256(sequence),
        normalized_input_sha256=_tensor_sha256(normalized),
        postnorm_delta_score_gradient_sha256=_tensor_sha256(delta_gradient),
        allowed_mask=allowed,
        probabilities=probabilities,
        signed_responses=signed_responses,
        intrinsic_fisher_energies=intrinsic_energies,
        softmax_tangents=softmax_tangents,
        reconstructed_operator=reconstructed_operator,
        reference_operator=expected_operator,
        reconstructed_postnorm_delta=reconstructed_delta,
        reference_postnorm_delta=reference_delta,
        reference_head_response=reference_head_response,
        reference_total_response=reference_total_response,
        closure=closure,
    )
