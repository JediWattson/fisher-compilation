"""Compile native Gemma attention V/O transport into shared generators.

This module implements the first contextual transport rung while deliberately
leaving native Q/K routing and the post-attention RMSNorm outside the compiled
object.  A capture supplies normalized residual inputs, exact native attention
probabilities, the pre-RMSNorm operator output, and its downstream score
gradient.  The compiler then fits

``M_a ~= sum_k rho[a, k] G_k``

jointly at the complete operator-output boundary.  Every ``G_k`` is stored as
an explicit low-rank SVD factorization.  The fitting loss is samplewise and
joint over heads, so cross-head cancellation and coalitions are retained::

    e = o_hat - o
    loss = mean((g.T @ e) ** 2 + epsilon * ||e|| ** 2)

The chart router is a second, separate fit.  Its oracle solves a signed,
Fisher-weighted sparse reconstruction over the frozen generator outputs.  The
deployable router uses stopped frozen-chart memberships, TopK-renormalized
selection probabilities, and independent signed amplitudes.  Neither fitted
runtime object retains teacher outputs, score gradients, native V/O weights,
token ids, or prompt text.

The implementation reports parameter storage separately from logical work.
Computing residual-width contextual inputs can cost more than native value
mixing; parameter eligibility is therefore never presented as a speedup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import math

import torch
from torch import Tensor, nn

from .gemma3_attention_generator import (
    Gemma3AttentionGenerator,
    Gemma3AttentionRouteFactorSnapshot,
)


_BANK_DOMAIN = b"fisher_graph:attention-transport-generator-bank:v1\0"
_ATLAS_DOMAIN = b"fisher_graph:attention-transport-chart-atlas:v1\0"
_ROUTER_DOMAIN = b"fisher_graph:attention-transport-chart-router:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:attention-transport-tensor:v1\0"
_HEAD_OPERATOR_CATALOG_DOMAIN = (
    b"fisher_graph:attention-transport-head-operator-catalog:v1\0"
)
EXACT_SPARSE_ORACLE_SOLVER = (
    "exact_support_enumeration_sizes_0_through_active_count"
)
HELDOUT_ORACLE_COEFFICIENT_RIDGE = 0.0


def _cpu_float64(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or not value.is_floating_point()
        or not all(int(size) > 0 for size in value.shape)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating rank-{ndim} tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


def _cpu_bool(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim or value.dtype != torch.bool:
        raise ValueError(f"{label} must be a boolean rank-{ndim} tensor")
    return value.detach().to(device="cpu").clone().contiguous()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _head_operator_sha256(value: Tensor) -> str:
    """Match the split-lineage tensor digest used by experiment core."""

    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _batch_head_operators_sha256(value: object) -> str:
    digest = getattr(value, "head_operators_sha256", None)
    if isinstance(digest, str) and len(digest) == 64:
        return digest
    operators = getattr(value, "head_operators", None)
    if not isinstance(operators, Tensor):
        raise TypeError("attention transport batch has no head operators")
    return _head_operator_sha256(operators)


def samplewise_joint_fisher_loss(
    prediction: Tensor,
    target: Tensor,
    score_gradient: Tensor,
    *,
    euclidean_floor: float,
) -> Tensor:
    """Return ``mean((g^T e)^2 + epsilon ||e||^2)`` over complete rows."""

    if (
        not isinstance(prediction, Tensor)
        or not isinstance(target, Tensor)
        or not isinstance(score_gradient, Tensor)
        or prediction.ndim != 2
        or target.shape != prediction.shape
        or score_gradient.shape != prediction.shape
        or not prediction.is_floating_point()
        or not target.is_floating_point()
        or not score_gradient.is_floating_point()
        or not bool(torch.isfinite(prediction).all())
        or not bool(torch.isfinite(target).all())
        or not bool(torch.isfinite(score_gradient).all())
    ):
        raise ValueError("Fisher loss inputs must be aligned finite matrices")
    if (
        isinstance(euclidean_floor, bool)
        or not isinstance(euclidean_floor, (int, float))
        or not math.isfinite(float(euclidean_floor))
        or float(euclidean_floor) <= 0.0
    ):
        raise ValueError("euclidean_floor must be finite and positive")
    error = prediction - target
    directional = (error * score_gradient).sum(dim=-1).square()
    euclidean = error.square().sum(dim=-1)
    return (directional + float(euclidean_floor) * euclidean).mean()


@dataclass(frozen=True, slots=True)
class AttentionTransportCapture:
    """One exact, bias-free attention invocation captured for compilation."""

    invocation_id: str
    normalized_inputs: Tensor
    attention_probabilities: Tensor
    operator_outputs: Tensor
    operator_output_score_gradients: Tensor
    query_valid_mask: Tensor
    key_valid_mask: Tensor
    value_projection_weight: Tensor
    output_projection_weight: Tensor
    query_head_count: int
    key_value_head_count: int
    head_dimension: int

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("invocation_id must be nonempty")
        for label, value in (
            ("query_head_count", self.query_head_count),
            ("key_value_head_count", self.key_value_head_count),
            ("head_dimension", self.head_dimension),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.query_head_count % self.key_value_head_count:
            raise ValueError("query heads must be divisible by key/value heads")

        source_precision = max(
            float(torch.finfo(value.dtype).eps)
            for value in (
                self.normalized_inputs,
                self.attention_probabilities,
                self.operator_outputs,
                self.value_projection_weight,
                self.output_projection_weight,
            )
            if isinstance(value, Tensor) and value.is_floating_point()
        )
        normalized = _cpu_float64(
            self.normalized_inputs,
            label="normalized_inputs",
            ndim=3,
        )
        probabilities = _cpu_float64(
            self.attention_probabilities,
            label="attention_probabilities",
            ndim=4,
        )
        outputs = _cpu_float64(
            self.operator_outputs,
            label="operator_outputs",
            ndim=3,
        )
        gradients = _cpu_float64(
            self.operator_output_score_gradients,
            label="operator_output_score_gradients",
            ndim=3,
        )
        query_mask = _cpu_bool(
            self.query_valid_mask,
            label="query_valid_mask",
            ndim=2,
        )
        key_mask = _cpu_bool(
            self.key_valid_mask,
            label="key_valid_mask",
            ndim=2,
        )
        value_weight = _cpu_float64(
            self.value_projection_weight,
            label="value_projection_weight",
            ndim=2,
        )
        output_weight = _cpu_float64(
            self.output_projection_weight,
            label="output_projection_weight",
            ndim=2,
        )

        batch, key_length, hidden = normalized.shape
        probability_shape = (
            batch,
            self.query_head_count,
            outputs.shape[1],
            key_length,
        )
        if tuple(probabilities.shape) != probability_shape:
            raise ValueError("attention probabilities do not match capture geometry")
        if (
            outputs.shape != gradients.shape
            or outputs.shape[0] != batch
            or outputs.shape[2] != hidden
            or query_mask.shape != outputs.shape[:2]
            or key_mask.shape != normalized.shape[:2]
            or value_weight.shape
            != (self.key_value_head_count * self.head_dimension, hidden)
            or output_weight.shape
            != (hidden, self.query_head_count * self.head_dimension)
        ):
            raise ValueError("attention capture tensor shapes do not close")
        if bool((probabilities < 0.0).any()):
            raise ValueError("attention probabilities must be nonnegative")
        invalid_keys = ~key_mask[:, None, None, :].expand_as(probabilities)
        invalid_queries = ~query_mask[:, None, :, None].expand_as(probabilities)
        if bool((probabilities.masked_select(invalid_keys) != 0.0).any()):
            raise ValueError("invalid keys carry attention probability")
        if bool((probabilities.masked_select(invalid_queries) != 0.0).any()):
            raise ValueError("invalid queries carry attention probability")
        masses = probabilities.sum(dim=-1)
        valid_rows = query_mask[:, None, :].expand_as(masses)
        if not torch.allclose(
            masses[valid_rows],
            torch.ones_like(masses[valid_rows]),
            atol=2.0e-6,
            rtol=2.0e-6,
        ):
            raise ValueError("valid attention rows must be probability simplexes")

        for name, value in (
            ("normalized_inputs", normalized),
            ("attention_probabilities", probabilities),
            ("operator_outputs", outputs),
            ("operator_output_score_gradients", gradients),
            ("query_valid_mask", query_mask),
            ("key_valid_mask", key_mask),
            ("value_projection_weight", value_weight),
            ("output_projection_weight", output_weight),
        ):
            object.__setattr__(self, name, value)

        exact = self.reconstruct_native_operator_outputs()
        target = self.valid_operator_outputs
        relative_tolerance = max(5.0e-7, 64.0 * source_precision)
        tolerance = 5.0e-8 + relative_tolerance * max(
            1.0,
            float(target.abs().max()),
        )
        if not torch.allclose(
            exact,
            target,
            atol=tolerance,
            rtol=relative_tolerance,
        ):
            maximum = float((exact - target).abs().max())
            raise ValueError(
                "native head decomposition does not reconstruct operator output "
                f"(maximum absolute error {maximum:.6g})"
            )

    @property
    def hidden_width(self) -> int:
        return int(self.normalized_inputs.shape[-1])

    @property
    def row_count(self) -> int:
        return int(self.query_valid_mask.sum())

    @property
    def head_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.invocation_id}/query-head.{index}"
            for index in range(self.query_head_count)
        )

    @property
    def head_operators(self) -> Tensor:
        operators = []
        repeats = self.query_head_count // self.key_value_head_count
        for query_head in range(self.query_head_count):
            kv_head = query_head // repeats
            head_slice = slice(
                query_head * self.head_dimension,
                (query_head + 1) * self.head_dimension,
            )
            kv_slice = slice(
                kv_head * self.head_dimension,
                (kv_head + 1) * self.head_dimension,
            )
            operators.append(
                self.output_projection_weight[:, head_slice]
                @ self.value_projection_weight[kv_slice]
            )
        return torch.stack(operators).contiguous()

    @property
    def head_contexts(self) -> Tensor:
        contexts = torch.einsum(
            "baqs,bsh->bqah",
            self.attention_probabilities,
            self.normalized_inputs,
        )
        return contexts[self.query_valid_mask].contiguous()

    @property
    def valid_operator_outputs(self) -> Tensor:
        return self.operator_outputs[self.query_valid_mask].contiguous()

    @property
    def valid_score_gradients(self) -> Tensor:
        return self.operator_output_score_gradients[
            self.query_valid_mask
        ].contiguous()

    @property
    def valid_chart_inputs(self) -> Tensor:
        if self.normalized_inputs.shape[:2] != self.query_valid_mask.shape:
            raise ValueError("chart inputs require self-attention query/key alignment")
        return self.normalized_inputs[self.query_valid_mask].contiguous()

    def reconstruct_native_operator_outputs(self) -> Tensor:
        return torch.einsum(
            "aoh,nah->no",
            self.head_operators,
            self.head_contexts,
        ).contiguous()

    @property
    def native_v_o_parameter_count(self) -> int:
        return int(
            self.value_projection_weight.numel()
            + self.output_projection_weight.numel()
        )


@dataclass(frozen=True, slots=True)
class AttentionTransportHeadOperatorCatalog:
    """One authenticated, shared V/O head-operator catalog per invocation.

    Compact row batches deliberately retain this catalog by reference.  The
    catalog owns the only float64 clone; its digest, storage identity, and
    tensor version make later split-lineage checks independent of object
    aliasing while avoiding one ``[A,H,H]`` clone per captured batch.
    """

    invocation_id: str
    head_operators: Tensor
    native_v_o_parameter_count: int
    _head_operators_sha256: str = field(init=False, repr=False, compare=False)
    _tensor_version: int = field(init=False, repr=False, compare=False)
    _data_ptr: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("head-operator catalog invocation_id must be nonempty")
        if (
            type(self.native_v_o_parameter_count) is not int
            or self.native_v_o_parameter_count <= 0
        ):
            raise ValueError("head-operator catalog native parameter count is invalid")
        operators = _cpu_float64(
            self.head_operators,
            label="head-operator catalog",
            ndim=3,
        )
        if (
            operators.shape[0] <= 0
            or operators.shape[1] != operators.shape[2]
        ):
            raise ValueError("head-operator catalog geometry is invalid")
        object.__setattr__(self, "head_operators", operators)
        object.__setattr__(
            self,
            "_head_operators_sha256",
            _head_operator_sha256(operators),
        )
        object.__setattr__(self, "_tensor_version", int(operators._version))
        object.__setattr__(self, "_data_ptr", int(operators.data_ptr()))

    @property
    def query_head_count(self) -> int:
        return int(self.head_operators.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.head_operators.shape[1])

    @property
    def head_operators_sha256(self) -> str:
        self.validate_unchanged()
        return self._head_operators_sha256

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "head_operators_sha256": self.head_operators_sha256,
            "hidden_width": self.hidden_width,
            "invocation_id": self.invocation_id,
            "native_v_o_parameter_count": self.native_v_o_parameter_count,
            "query_head_count": self.query_head_count,
        }
        return hashlib.sha256(
            _HEAD_OPERATOR_CATALOG_DOMAIN
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def validate_unchanged(self, *, rehash: bool = False) -> None:
        operators = self.head_operators
        if (
            not isinstance(operators, Tensor)
            or operators.device.type != "cpu"
            or operators.dtype != torch.float64
            or operators.layout != torch.strided
            or not operators.is_contiguous()
            or int(operators.data_ptr()) != self._data_ptr
            or int(operators._version) != self._tensor_version
        ):
            raise RuntimeError("shared head-operator catalog changed after capture")
        if (
            rehash
            and _head_operator_sha256(operators)
            != self._head_operators_sha256
        ):
            raise RuntimeError(
                "shared head-operator catalog bytes changed after capture"
            )

    def validate_capture(self, capture: AttentionTransportCapture) -> None:
        if not isinstance(capture, AttentionTransportCapture):
            raise TypeError("catalog capture must be an attention transport capture")
        if (
            capture.invocation_id != self.invocation_id
            or capture.query_head_count != self.query_head_count
            or capture.hidden_width != self.hidden_width
            or capture.native_v_o_parameter_count
            != self.native_v_o_parameter_count
            or _head_operator_sha256(capture.head_operators)
            != self.head_operators_sha256
        ):
            raise ValueError("capture does not match the shared head-operator catalog")


_HEAD_OPERATOR_REFERENCE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _AuthenticatedHeadOperatorReference:
    catalog: AttentionTransportHeadOperatorCatalog
    factory_token: object

    def __post_init__(self) -> None:
        if (
            self.factory_token is not _HEAD_OPERATOR_REFERENCE_TOKEN
            or not isinstance(self.catalog, AttentionTransportHeadOperatorCatalog)
        ):
            raise TypeError(
                "head-operator references require authenticated construction"
            )


@dataclass(frozen=True, slots=True)
class AttentionTransportRowBatch:
    """Prompt-free sufficient rows retained after authenticating a capture.

    The full attention probabilities and native V/O tensors are needed only at
    the capture boundary.  Fitting needs their authenticated consequences: the
    contextual head inputs, complete operator target/cotangent rows, normalized
    chart inputs, and one frozen head-operator catalog per invocation.
    """

    invocation_id: str
    head_contexts: Tensor
    operator_outputs: Tensor
    operator_output_score_gradients: Tensor
    chart_inputs: Tensor
    head_operators: Tensor | _AuthenticatedHeadOperatorReference
    native_v_o_parameter_count: int
    _operator_catalog: AttentionTransportHeadOperatorCatalog | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _head_operators_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _head_operators_version: int = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("row-batch invocation_id must be nonempty")
        contexts = _cpu_float64(
            self.head_contexts,
            label="row-batch head_contexts",
            ndim=3,
        )
        outputs = _cpu_float64(
            self.operator_outputs,
            label="row-batch operator_outputs",
            ndim=2,
        )
        gradients = _cpu_float64(
            self.operator_output_score_gradients,
            label="row-batch operator_output_score_gradients",
            ndim=2,
        )
        chart_inputs = _cpu_float64(
            self.chart_inputs,
            label="row-batch chart_inputs",
            ndim=2,
        )
        reference = self.head_operators
        catalog = (
            reference.catalog
            if isinstance(reference, _AuthenticatedHeadOperatorReference)
            else None
        )
        if catalog is None:
            operators = _cpu_float64(
                reference,
                label="row-batch head_operators",
                ndim=3,
            )
            operator_sha256 = _head_operator_sha256(operators)
        else:
            catalog.validate_unchanged()
            operators = catalog.head_operators
            operator_sha256 = catalog.head_operators_sha256
        rows, heads, hidden = contexts.shape
        if (
            outputs.shape != (rows, hidden)
            or gradients.shape != outputs.shape
            or chart_inputs.shape != outputs.shape
            or operators.shape != (heads, hidden, hidden)
            or type(self.native_v_o_parameter_count) is not int
            or self.native_v_o_parameter_count <= 0
        ):
            raise ValueError("attention transport row-batch shapes do not close")
        if catalog is not None and (
            catalog.invocation_id != self.invocation_id
            or catalog.query_head_count != heads
            or catalog.hidden_width != hidden
            or catalog.native_v_o_parameter_count
            != self.native_v_o_parameter_count
        ):
            raise ValueError("row-batch shared head-operator catalog is misaligned")
        if catalog is None:
            exact = torch.einsum("aoh,nah->no", operators, contexts)
            if not torch.allclose(exact, outputs, atol=2.0e-5, rtol=2.0e-5):
                maximum = float((exact - outputs).abs().max())
                raise ValueError(
                    "row-batch head operators do not reconstruct operator output "
                    f"(maximum absolute error {maximum:.6g})"
                )
        for name, value in (
            ("head_contexts", contexts),
            ("operator_outputs", outputs),
            ("operator_output_score_gradients", gradients),
            ("chart_inputs", chart_inputs),
            ("head_operators", operators),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_operator_catalog", catalog)
        object.__setattr__(self, "_head_operators_sha256", operator_sha256)
        object.__setattr__(
            self,
            "_head_operators_version",
            int(operators._version),
        )

    @property
    def head_operators_sha256(self) -> str:
        self.validate_head_operator_catalog()
        return self._head_operators_sha256

    @property
    def head_operator_catalog_identity(self) -> int:
        return (
            id(self._operator_catalog)
            if self._operator_catalog is not None
            else id(self)
        )

    def validate_head_operator_catalog(self, *, rehash: bool = False) -> None:
        if self._operator_catalog is not None:
            self._operator_catalog.validate_unchanged(rehash=rehash)
            return
        if int(self.head_operators._version) != self._head_operators_version:
            raise RuntimeError("row-batch head operators changed after construction")
        if (
            rehash
            and _head_operator_sha256(self.head_operators)
            != self._head_operators_sha256
        ):
            raise RuntimeError(
                "row-batch head-operator bytes changed after construction"
            )

    @property
    def hidden_width(self) -> int:
        return int(self.head_contexts.shape[-1])

    @property
    def query_head_count(self) -> int:
        return int(self.head_contexts.shape[1])

    @property
    def row_count(self) -> int:
        return int(self.head_contexts.shape[0])

    @property
    def head_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.invocation_id}/query-head.{index}"
            for index in range(self.query_head_count)
        )

    @property
    def valid_operator_outputs(self) -> Tensor:
        return self.operator_outputs

    @property
    def valid_score_gradients(self) -> Tensor:
        return self.operator_output_score_gradients

    @property
    def valid_chart_inputs(self) -> Tensor:
        return self.chart_inputs


AttentionTransportBatch = AttentionTransportCapture | AttentionTransportRowBatch


def attention_transport_row_batch_from_capture(
    capture: AttentionTransportCapture,
    *,
    row_indices: Tensor | None = None,
    head_operator_catalog: AttentionTransportHeadOperatorCatalog | None = None,
) -> AttentionTransportRowBatch:
    """Discard raw attention/native weights after deterministic row selection."""

    if not isinstance(capture, AttentionTransportCapture):
        raise TypeError("capture must be an AttentionTransportCapture")
    if row_indices is None:
        indices = torch.arange(capture.row_count, dtype=torch.int64)
    else:
        if (
            not isinstance(row_indices, Tensor)
            or row_indices.ndim != 1
            or row_indices.dtype != torch.int64
            or row_indices.numel() == 0
        ):
            raise ValueError("row_indices must be a nonempty int64 vector")
        indices = row_indices.detach().to(device="cpu").clone().contiguous()
        if (
            int(indices.min()) < 0
            or int(indices.max()) >= capture.row_count
            or not bool((indices[1:] > indices[:-1]).all())
        ):
            raise ValueError("row_indices must be unique, sorted, and in range")
    operator_value: Tensor | _AuthenticatedHeadOperatorReference
    if head_operator_catalog is None:
        operator_value = capture.head_operators
    else:
        head_operator_catalog.validate_capture(capture)
        operator_value = _AuthenticatedHeadOperatorReference(
            catalog=head_operator_catalog,
            factory_token=_HEAD_OPERATOR_REFERENCE_TOKEN,
        )
    return AttentionTransportRowBatch(
        invocation_id=capture.invocation_id,
        head_contexts=capture.head_contexts.index_select(0, indices),
        operator_outputs=capture.valid_operator_outputs.index_select(0, indices),
        operator_output_score_gradients=(
            capture.valid_score_gradients.index_select(0, indices)
        ),
        chart_inputs=capture.valid_chart_inputs.index_select(0, indices),
        head_operators=operator_value,
        native_v_o_parameter_count=capture.native_v_o_parameter_count,
    )


def attention_transport_capture_from_exact_attention(
    generator: Gemma3AttentionGenerator,
    *,
    normalized_inputs: Tensor,
    route_factors: Gemma3AttentionRouteFactorSnapshot,
    operator_outputs: Tensor,
    operator_output_score_gradients: Tensor,
) -> AttentionTransportCapture:
    """Bridge one exact generator execution into the transport compiler.

    This authenticates that the captured repeated values are exactly the
    bias-free native V projection of the normalized residual inputs.  Q/K,
    masks, and softmax remain represented by the frozen route probabilities.
    """

    if not isinstance(generator, Gemma3AttentionGenerator):
        raise TypeError("generator must be a Gemma3AttentionGenerator")
    if not isinstance(route_factors, Gemma3AttentionRouteFactorSnapshot):
        raise TypeError("route_factors must be a native route-factor snapshot")
    if (
        generator.generator_kind != "exact_full_rank"
        or generator.input_basis is not None
        or generator.output_encoder_weight is not None
    ):
        raise ValueError("transport capture requires exact full-rank attention")
    if generator.v_bias is not None or generator.o_bias is not None:
        raise ValueError("transport capture requires bias-free V and O projections")
    if (
        route_factors.invocation_id != generator.config.invocation_id
        or route_factors.artifact_sha256 != generator.artifact_sha256
    ):
        raise ValueError("route factors are not bound to the exact generator")
    if not torch.equal(
        route_factors.output_projection_weight,
        generator.output_decoder_weight.detach().to(
            device="cpu",
            dtype=route_factors.output_projection_weight.dtype,
        ),
    ):
        raise ValueError("route-factor output projection drifted")

    normalized = _cpu_float64(
        normalized_inputs,
        label="normalized_inputs",
        ndim=3,
    )
    if (
        normalized_inputs.device != generator.v_weight.device
        or normalized_inputs.dtype != generator.v_weight.dtype
    ):
        raise ValueError(
            "normalized inputs must retain the native V projection dtype and device"
        )
    batch, key_length, _ = normalized.shape
    key_value_heads = route_factors.key_value_head_count
    head_dimension = route_factors.head_dimension
    values = torch.nn.functional.linear(
        normalized_inputs.detach(),
        generator.v_weight.detach(),
    )
    values = values.reshape(batch, key_length, key_value_heads, head_dimension)
    values = values.permute(0, 2, 1, 3)
    expected_repeated = values.repeat_interleave(
        route_factors.query_head_count // key_value_heads,
        dim=1,
    ).detach().to(device="cpu").contiguous()
    repeated = route_factors.repeated_values
    if not torch.equal(expected_repeated, repeated):
        maximum = float(
            (
                expected_repeated.to(dtype=torch.float64)
                - repeated.to(dtype=torch.float64)
            )
            .abs()
            .max()
        )
        raise ValueError(
            "native repeated values do not close through the frozen V projection "
            f"(maximum absolute error {maximum:.6g})"
        )
    return AttentionTransportCapture(
        invocation_id=route_factors.invocation_id,
        normalized_inputs=normalized_inputs,
        attention_probabilities=route_factors.probabilities,
        operator_outputs=operator_outputs,
        operator_output_score_gradients=operator_output_score_gradients,
        query_valid_mask=route_factors.query_valid_mask,
        key_valid_mask=route_factors.key_valid_mask,
        value_projection_weight=generator.v_weight,
        output_projection_weight=route_factors.output_projection_weight,
        query_head_count=route_factors.query_head_count,
        key_value_head_count=route_factors.key_value_head_count,
        head_dimension=route_factors.head_dimension,
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportFitConfig:
    generator_count: int
    generator_rank: int
    steps: int = 256
    learning_rate: float = 2.0e-2
    euclidean_floor: float = 1.0e-3
    coefficient_ridge: float = 1.0e-8
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        for label, value, minimum in (
            ("generator_count", self.generator_count, 1),
            ("generator_rank", self.generator_rank, 1),
            ("steps", self.steps, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{label} is outside its valid range")
        for label, value, positive in (
            ("learning_rate", self.learning_rate, True),
            ("euclidean_floor", self.euclidean_floor, True),
            ("coefficient_ridge", self.coefficient_ridge, False),
            ("gradient_clip_norm", self.gradient_clip_norm, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (positive and float(value) <= 0.0)
                or (not positive and float(value) < 0.0)
            ):
                raise ValueError(f"{label} is invalid")


@dataclass(frozen=True, slots=True)
class AttentionTransportMetrics:
    row_count: int
    fisher_loss: float
    directional_mse: float
    euclidean_mse: float
    normalized_rmse: float
    cosine: float


def _metrics(
    prediction: Tensor,
    target: Tensor,
    gradient: Tensor,
    epsilon: float,
) -> AttentionTransportMetrics:
    error = prediction - target
    directional = (error * gradient).sum(dim=-1)
    target_power = float(target.square().mean())
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    denominator = float(
        torch.linalg.vector_norm(prediction_flat)
        * torch.linalg.vector_norm(target_flat)
    )
    return AttentionTransportMetrics(
        row_count=int(target.shape[0]),
        fisher_loss=float(
            samplewise_joint_fisher_loss(
                prediction,
                target,
                gradient,
                euclidean_floor=epsilon,
            )
        ),
        directional_mse=float(directional.square().mean()),
        euclidean_mse=float(error.square().mean()),
        normalized_rmse=(
            math.sqrt(float(error.square().mean()) / target_power)
            if target_power > 0.0
            else 0.0
        ),
        cosine=(
            float(torch.dot(prediction_flat, target_flat)) / denominator
            if denominator > 0.0
            else 0.0
        ),
    )


@dataclass(frozen=True, slots=True)
class FisherAttentionTransportGeneratorBank:
    """Source-free shared low-rank V/O transport generator bank."""

    left_factors: Tensor
    singular_scales: Tensor
    right_factors: Tensor
    head_coefficients: Tensor
    head_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        left = _cpu_float64(self.left_factors, label="left_factors", ndim=3)
        scales = _cpu_float64(
            self.singular_scales,
            label="singular_scales",
            ndim=2,
        )
        right = _cpu_float64(self.right_factors, label="right_factors", ndim=3)
        coefficients = _cpu_float64(
            self.head_coefficients,
            label="head_coefficients",
            ndim=2,
        )
        if (
            left.shape != right.shape
            or scales.shape != (left.shape[0], left.shape[2])
            or coefficients.shape[1] != left.shape[0]
            or coefficients.shape[0] != len(self.head_ids)
            or len(set(self.head_ids)) != len(self.head_ids)
            or any(not isinstance(value, str) or not value for value in self.head_ids)
            or bool((scales < 0.0).any())
        ):
            raise ValueError("generator bank shapes or head catalog are invalid")
        identity = torch.eye(left.shape[2], dtype=torch.float64)
        for label, factors in (("left", left), ("right", right)):
            for generator in factors:
                if not torch.allclose(
                    generator.T @ generator,
                    identity,
                    atol=1.0e-8,
                    rtol=1.0e-8,
                ):
                    raise ValueError(f"{label} generator factors must be orthonormal")
        object.__setattr__(self, "left_factors", left)
        object.__setattr__(self, "singular_scales", scales)
        object.__setattr__(self, "right_factors", right)
        object.__setattr__(self, "head_coefficients", coefficients)

    @property
    def generator_count(self) -> int:
        return int(self.left_factors.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.left_factors.shape[1])

    @property
    def generator_rank(self) -> int:
        return int(self.left_factors.shape[2])

    @property
    def generator_matrices(self) -> Tensor:
        return torch.einsum(
            "kor,kr,kir->koi",
            self.left_factors,
            self.singular_scales,
            self.right_factors,
        ).contiguous()

    @property
    def stored_parameter_count(self) -> int:
        k, h, r = self.left_factors.shape
        return int(k * (2 * h * r + r) + self.head_coefficients.numel())

    def _head_indices(self, invocation_id: str, head_count: int) -> tuple[int, ...]:
        requested = tuple(
            f"{invocation_id}/query-head.{index}" for index in range(head_count)
        )
        lookup = {value: index for index, value in enumerate(self.head_ids)}
        try:
            return tuple(lookup[value] for value in requested)
        except KeyError as error:
            raise ValueError(
                "invocation head catalog is absent from generator bank"
            ) from error

    def contextual_inputs_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        contexts = _cpu_float64(head_contexts, label="head_contexts", ndim=3)
        if contexts.shape[2] != self.hidden_width:
            raise ValueError("head contexts have the wrong hidden width")
        indices = self._head_indices(invocation_id, int(contexts.shape[1]))
        coefficients = self.head_coefficients[list(indices)]
        return torch.einsum("nah,ak->nkh", contexts, coefficients).contiguous()

    def candidate_outputs_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        latent = torch.einsum(
            "nki,kir->nkr",
            contextual,
            self.right_factors,
        )
        latent = latent * self.singular_scales[None]
        return torch.einsum(
            "nkr,kor->nko",
            latent,
            self.left_factors,
        ).contiguous()

    def router_context_features_from_head_contexts(
        self,
        *,
        invocation_id: str,
        head_contexts: Tensor,
    ) -> Tensor:
        """Return source-free per-generator RMS features of ``xi[k, t]``."""

        contextual = self.contextual_inputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=head_contexts,
        )
        return contextual.square().mean(dim=-1).sqrt().contiguous()

    def candidate_outputs(self, capture: AttentionTransportBatch) -> Tensor:
        return self.candidate_outputs_from_head_contexts(
            invocation_id=capture.invocation_id,
            head_contexts=capture.head_contexts,
        )

    def predict_operator_outputs(
        self,
        capture: AttentionTransportBatch,
        coefficients: Tensor | None = None,
    ) -> Tensor:
        candidates = self.candidate_outputs(capture)
        if coefficients is None:
            weights = torch.ones(
                candidates.shape[:2],
                dtype=torch.float64,
            )
        else:
            weights = _cpu_float64(coefficients, label="coefficients", ndim=2)
            if weights.shape != candidates.shape[:2]:
                raise ValueError("generator coefficients do not align with rows")
        return torch.einsum("nk,nkh->nh", weights, candidates).contiguous()

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "head_ids": list(self.head_ids),
            "left_factors_sha256": _tensor_sha256(self.left_factors),
            "singular_scales_sha256": _tensor_sha256(self.singular_scales),
            "right_factors_sha256": _tensor_sha256(self.right_factors),
            "head_coefficients_sha256": _tensor_sha256(self.head_coefficients),
            "generator_count": self.generator_count,
            "generator_rank": self.generator_rank,
            "hidden_width": self.hidden_width,
            "native_qk_retained": True,
            "native_output_rmsnorm_retained": True,
            "contains_teacher_outputs_or_gradients": False,
            "contains_native_value_or_output_weights": False,
        }
        return hashlib.sha256(
            _BANK_DOMAIN
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "head_ids": list(self.head_ids),
            "left_factors": self.left_factors.clone(),
            "singular_scales": self.singular_scales.clone(),
            "right_factors": self.right_factors.clone(),
            "head_coefficients": self.head_coefficients.clone(),
            "native_qk_retained": True,
            "native_output_rmsnorm_retained": True,
            "contains_teacher_outputs_or_gradients": False,
            "contains_native_value_or_output_weights": False,
        }


def fisher_attention_transport_generator_bank_from_state_dict(
    state: Mapping[str, object],
) -> FisherAttentionTransportGeneratorBank:
    expected = {
        "artifact_sha256",
        "contains_native_value_or_output_weights",
        "contains_teacher_outputs_or_gradients",
        "head_coefficients",
        "head_ids",
        "left_factors",
        "native_output_rmsnorm_retained",
        "native_qk_retained",
        "right_factors",
        "singular_scales",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("generator bank state fields are invalid")
    if (
        state["contains_native_value_or_output_weights"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
        or state["native_output_rmsnorm_retained"] is not True
        or state["native_qk_retained"] is not True
    ):
        raise ValueError("generator bank state safety declarations are invalid")
    raw_head_ids = state["head_ids"]
    if not isinstance(raw_head_ids, list):
        raise ValueError("generator bank head catalog must be a list")
    bank = FisherAttentionTransportGeneratorBank(
        left_factors=state["left_factors"],
        singular_scales=state["singular_scales"],
        right_factors=state["right_factors"],
        head_coefficients=state["head_coefficients"],
        head_ids=tuple(raw_head_ids),
    )
    if state["artifact_sha256"] != bank.artifact_sha256:
        raise ValueError("generator bank artifact hash drifted")
    return bank


@dataclass(frozen=True, slots=True)
class AttentionTransportBankFit:
    bank: FisherAttentionTransportGeneratorBank
    initial_fit_metrics: AttentionTransportMetrics
    final_fit_metrics: AttentionTransportMetrics
    evaluation_metrics: AttentionTransportMetrics
    native_v_o_parameter_count: int
    compiled_bank_parameter_count: int
    parameter_reduction_fraction: float
    parameter_compression_eligible: bool


def _capture_catalog(
    captures: Sequence[AttentionTransportBatch],
    *,
    materialize_operators: bool = False,
) -> tuple[
    tuple[AttentionTransportBatch, ...],
    tuple[AttentionTransportBatch, ...],
    tuple[str, ...],
    Tensor | None,
]:
    if type(materialize_operators) is not bool:
        raise TypeError("materialize_operators must be boolean")
    if isinstance(captures, (str, bytes)) or not isinstance(captures, Sequence):
        raise TypeError("captures must be a materialized sequence")
    values = tuple(captures)
    if not values or any(
        not isinstance(value, (AttentionTransportCapture, AttentionTransportRowBatch))
        for value in values
    ):
        raise ValueError("captures must contain attention transport captures")
    if len({value.hidden_width for value in values}) != 1:
        raise ValueError("all captures must share one residual width")

    representatives_by_invocation: dict[str, AttentionTransportBatch] = {}
    for value in values:
        representative = representatives_by_invocation.get(value.invocation_id)
        if representative is None:
            representatives_by_invocation[value.invocation_id] = value
            continue
        if (
            value.query_head_count != representative.query_head_count
            or value.native_v_o_parameter_count
            != representative.native_v_o_parameter_count
            or _batch_head_operators_sha256(value)
            != _batch_head_operators_sha256(representative)
        ):
            raise ValueError(
                "captures for one invocation must share identical frozen V/O "
                "weights and head geometry; derived operator or parameter "
                "identity drifted"
            )

    representatives = tuple(representatives_by_invocation.values())
    head_ids = tuple(item for value in representatives for item in value.head_ids)
    operators = (
        torch.cat(
            tuple(value.head_operators for value in representatives),
            dim=0,
        )
        if materialize_operators
        else None
    )
    return values, representatives, head_ids, operators


def _principal_initialization(
    operators: Tensor,
    *,
    generator_count: int,
    generator_rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    gram = torch.einsum("pij,qij->pq", operators, operators)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True, stable=True)
    selected_values = eigenvalues[order[:generator_count]]
    selected_vectors = eigenvectors[:, order[:generator_count]]
    tolerance = 256.0 * torch.finfo(torch.float64).eps * max(
        1.0,
        float(eigenvalues.abs().max()),
    )
    if bool((selected_values <= tolerance).any()):
        raise ValueError("generator_count exceeds the numerical head-operator rank")
    left = []
    right = []
    coefficients = selected_vectors * selected_values.sqrt()[None, :]
    for index in range(generator_count):
        atom = torch.einsum(
            "p,pij->ij",
            selected_vectors[:, index],
            operators,
        ) / selected_values[index].sqrt()
        u, singular, vh = torch.linalg.svd(atom, full_matrices=False)
        root = singular[:generator_rank].clamp_min(0.0).sqrt()
        left.append(u[:, :generator_rank] * root[None, :])
        right.append(vh[:generator_rank].T * root[None, :])
    return torch.stack(left), torch.stack(right), coefficients


def _parameter_prediction(
    capture: AttentionTransportBatch,
    left: Tensor,
    right: Tensor,
    coefficients: Tensor,
) -> Tensor:
    contextual = torch.einsum(
        "nah,ak->nkh",
        capture.head_contexts,
        coefficients,
    )
    latent = torch.einsum("nkh,khr->nkr", contextual, right)
    return torch.einsum("nkr,khr->nh", latent, left)


def _joined_metrics(
    bank: FisherAttentionTransportGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    *,
    epsilon: float,
) -> AttentionTransportMetrics:
    predictions = torch.cat(
        tuple(bank.predict_operator_outputs(value) for value in captures)
    )
    targets = torch.cat(tuple(value.valid_operator_outputs for value in captures))
    gradients = torch.cat(tuple(value.valid_score_gradients for value in captures))
    return _metrics(predictions, targets, gradients, epsilon)


def _canonical_bank(
    left: Tensor,
    right: Tensor,
    coefficients: Tensor,
    head_ids: tuple[str, ...],
) -> FisherAttentionTransportGeneratorBank:
    canonical_left = []
    canonical_scales = []
    canonical_right = []
    rank = int(left.shape[2])
    for left_factor, right_factor in zip(left, right, strict=True):
        matrix = left_factor @ right_factor.T
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        canonical_left.append(u[:, :rank])
        canonical_scales.append(singular[:rank].clamp_min(0.0))
        canonical_right.append(vh[:rank].T)
    return FisherAttentionTransportGeneratorBank(
        left_factors=torch.stack(canonical_left),
        singular_scales=torch.stack(canonical_scales),
        right_factors=torch.stack(canonical_right),
        head_coefficients=coefficients,
        head_ids=head_ids,
    )


def fit_fisher_attention_transport_generator_bank(
    fit_captures: Sequence[AttentionTransportBatch],
    evaluation_captures: Sequence[AttentionTransportBatch],
    *,
    config: AttentionTransportFitConfig,
) -> AttentionTransportBankFit:
    """Fit a shared structured bank using complete samplewise Fisher errors."""

    if not isinstance(config, AttentionTransportFitConfig):
        raise TypeError("config must be an AttentionTransportFitConfig")
    fit, fit_invocations, head_ids, operators = _capture_catalog(
        fit_captures,
        materialize_operators=True,
    )
    evaluation, evaluation_invocations, _, _ = _capture_catalog(
        evaluation_captures
    )
    if operators is None:
        raise RuntimeError("bank initialization omitted the head-operator catalog")
    evaluation_by_invocation = {
        value.invocation_id: value for value in evaluation_invocations
    }
    if {
        value.invocation_id for value in fit_invocations
    } != set(evaluation_by_invocation):
        raise ValueError("fit and evaluation captures must share frozen V/O operators")
    for fit_invocation in fit_invocations:
        evaluation_invocation = evaluation_by_invocation[
            fit_invocation.invocation_id
        ]
        if (
            fit_invocation.query_head_count
            != evaluation_invocation.query_head_count
            or fit_invocation.native_v_o_parameter_count
            != evaluation_invocation.native_v_o_parameter_count
            or _batch_head_operators_sha256(fit_invocation)
            != _batch_head_operators_sha256(evaluation_invocation)
        ):
            raise ValueError(
                "fit and evaluation captures must share frozen V/O operators"
            )
    hidden = int(operators.shape[-1])
    if config.generator_count > operators.shape[0]:
        raise ValueError("generator_count cannot exceed the head operator count")
    if config.generator_rank > hidden:
        raise ValueError("generator_rank cannot exceed the residual width")

    initial_left, initial_right, initial_coefficients = _principal_initialization(
        operators,
        generator_count=config.generator_count,
        generator_rank=config.generator_rank,
    )
    del operators
    initial_bank = _canonical_bank(
        initial_left,
        initial_right,
        initial_coefficients,
        head_ids,
    )
    initial_metrics = _joined_metrics(
        initial_bank,
        fit,
        epsilon=config.euclidean_floor,
    )

    left = nn.Parameter(initial_left.clone())
    right = nn.Parameter(initial_right.clone())
    coefficients = nn.Parameter(initial_coefficients.clone())
    optimizer = torch.optim.Adam(
        (left, right, coefficients),
        lr=float(config.learning_rate),
    )
    offsets_by_invocation: dict[str, tuple[int, int]] = {}
    start = 0
    for capture in fit_invocations:
        stop = start + capture.query_head_count
        offsets_by_invocation[capture.invocation_id] = (start, stop)
        start = stop

    def current_objective() -> Tensor:
        losses = []
        rows = []
        for capture in fit:
            start, stop = offsets_by_invocation[capture.invocation_id]
            prediction = _parameter_prediction(
                capture,
                left,
                right,
                coefficients[start:stop],
            )
            losses.append(
                samplewise_joint_fisher_loss(
                    prediction,
                    capture.valid_operator_outputs,
                    capture.valid_score_gradients,
                    euclidean_floor=config.euclidean_floor,
                )
            )
            rows.append(capture.row_count)
        denominator = float(sum(rows))
        objective = sum(
            value * count for value, count in zip(losses, rows, strict=True)
        ) / denominator
        if config.coefficient_ridge:
            objective = objective + float(
                config.coefficient_ridge
            ) * coefficients.square().mean()
        return objective

    best_loss = math.inf
    best_parameters: tuple[Tensor, Tensor, Tensor] | None = None
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = current_objective()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("attention transport fit became nonfinite")
        observed_loss = float(loss.detach())
        if observed_loss < best_loss:
            best_loss = observed_loss
            best_parameters = (
                left.detach().clone(),
                right.detach().clone(),
                coefficients.detach().clone(),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (left, right, coefficients),
            max_norm=float(config.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()

    terminal_loss = current_objective()
    if not bool(torch.isfinite(terminal_loss)):
        raise RuntimeError("attention transport fit became nonfinite")
    if float(terminal_loss.detach()) < best_loss:
        best_parameters = (
            left.detach().clone(),
            right.detach().clone(),
            coefficients.detach().clone(),
        )
    if best_parameters is None:
        raise RuntimeError("attention transport fit did not evaluate one state")

    bank = _canonical_bank(
        *best_parameters,
        head_ids,
    )
    final_metrics = _joined_metrics(bank, fit, epsilon=config.euclidean_floor)
    evaluation_metrics = _joined_metrics(
        bank,
        evaluation,
        epsilon=config.euclidean_floor,
    )
    native_parameters = sum(
        value.native_v_o_parameter_count for value in fit_invocations
    )
    compiled_parameters = bank.stored_parameter_count
    reduction = 1.0 - compiled_parameters / native_parameters
    return AttentionTransportBankFit(
        bank=bank,
        initial_fit_metrics=initial_metrics,
        final_fit_metrics=final_metrics,
        evaluation_metrics=evaluation_metrics,
        native_v_o_parameter_count=native_parameters,
        compiled_bank_parameter_count=compiled_parameters,
        parameter_reduction_fraction=float(reduction),
        parameter_compression_eligible=compiled_parameters < native_parameters,
    )


@dataclass(frozen=True, slots=True)
class FrozenAttentionChartAtlas:
    centers: Tensor
    bases: Tensor
    temperature: float

    def __post_init__(self) -> None:
        centers = _cpu_float64(self.centers, label="chart centers", ndim=2)
        bases = _cpu_float64(self.bases, label="chart bases", ndim=3)
        if bases.shape[:2] != centers.shape or bases.shape[2] > centers.shape[1]:
            raise ValueError("chart bases do not match chart centers")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) <= 0.0
        ):
            raise ValueError("chart temperature must be finite and positive")
        identity = torch.eye(bases.shape[2], dtype=torch.float64)
        for basis in bases:
            if not torch.allclose(basis.T @ basis, identity, atol=1.0e-8, rtol=1.0e-8):
                raise ValueError("chart bases must have orthonormal columns")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "bases", bases)

    @property
    def chart_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def hidden_width(self) -> int:
        return int(self.centers.shape[1])

    @property
    def chart_rank(self) -> int:
        return int(self.bases.shape[2])

    @property
    def stored_parameter_count(self) -> int:
        return int(self.centers.numel() + self.bases.numel() + 1)

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "bases_sha256": _tensor_sha256(self.bases),
            "centers_sha256": _tensor_sha256(self.centers),
            "chart_count": self.chart_count,
            "chart_rank": self.chart_rank,
            "contains_teacher_outputs_or_gradients": False,
            "hidden_width": self.hidden_width,
            "temperature": float(self.temperature),
        }
        return hashlib.sha256(
            _ATLAS_DOMAIN
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "bases": self.bases.clone(),
            "centers": self.centers.clone(),
            "contains_teacher_outputs_or_gradients": False,
            "temperature": float(self.temperature),
        }

    def coordinates_and_memberships(
        self,
        chart_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        inputs = _cpu_float64(chart_inputs, label="chart_inputs", ndim=2)
        if inputs.shape[1] != self.hidden_width:
            raise ValueError("chart inputs have the wrong hidden width")
        centered = inputs[:, None, :] - self.centers[None, :, :]
        coordinates = torch.einsum("nch,chr->ncr", centered, self.bases)
        normal_squared = (
            centered.square().sum(dim=-1) - coordinates.square().sum(dim=-1)
        ).clamp_min(0.0)
        memberships = torch.softmax(
            -normal_squared / float(self.temperature),
            dim=-1,
        )
        return coordinates, memberships


def frozen_attention_chart_atlas_from_state_dict(
    state: Mapping[str, object],
) -> FrozenAttentionChartAtlas:
    if not isinstance(state, Mapping) or set(state) != {
        "artifact_sha256",
        "bases",
        "centers",
        "contains_teacher_outputs_or_gradients",
        "temperature",
    }:
        raise ValueError("frozen attention atlas state fields are invalid")
    if state["contains_teacher_outputs_or_gradients"] is not False:
        raise ValueError("frozen attention atlas retained teacher state")
    atlas = FrozenAttentionChartAtlas(
        centers=state["centers"],
        bases=state["bases"],
        temperature=state["temperature"],
    )
    if state["artifact_sha256"] != atlas.artifact_sha256:
        raise ValueError("frozen attention atlas artifact hash drifted")
    return atlas


def _canonicalize_basis_signs(basis: Tensor) -> Tensor:
    result = basis.clone()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(torch.argmax(vector.abs()))
        if float(vector[pivot]) < 0.0:
            result[:, column] = -vector
    return result


def fit_frozen_attention_chart_atlas(
    captures: Sequence[AttentionTransportBatch],
    *,
    chart_count: int,
    chart_rank: int,
    temperature_scale: float = 1.0,
    lloyd_iterations: int = 12,
) -> FrozenAttentionChartAtlas:
    """Fit deterministic normalized-input charts with balanced invocations."""

    values, invocations, _, _ = _capture_catalog(captures)
    hidden = values[0].hidden_width
    if type(chart_count) is not int or chart_count <= 0:
        raise ValueError("chart_count must be positive")
    if type(chart_rank) is not int or not 0 < chart_rank < hidden:
        raise ValueError("chart_rank must be positive and smaller than hidden width")
    if type(lloyd_iterations) is not int or lloyd_iterations <= 0:
        raise ValueError("lloyd_iterations must be positive")
    if (
        isinstance(temperature_scale, bool)
        or not isinstance(temperature_scale, (int, float))
        or not math.isfinite(float(temperature_scale))
        or float(temperature_scale) <= 0.0
    ):
        raise ValueError("temperature_scale must be finite and positive")

    rows_by_invocation = {
        invocation.invocation_id: sum(
            capture.row_count
            for capture in values
            if capture.invocation_id == invocation.invocation_id
        )
        for invocation in invocations
    }
    contexts = torch.cat(tuple(value.valid_chart_inputs for value in values))
    weights = torch.cat(
        tuple(
            torch.full(
                (value.row_count,),
                1.0
                / (
                    len(invocations)
                    * rows_by_invocation[value.invocation_id]
                ),
                dtype=torch.float64,
            )
            for value in values
        )
    )
    if contexts.shape[0] < chart_count * (chart_rank + 1):
        raise ValueError("chart fit has too few rows for its count and rank")
    mean = (weights[:, None] * contexts).sum(dim=0) / weights.sum()
    first_score = weights * (contexts - mean).square().sum(dim=-1)
    selected = [int(torch.argmax(first_score))]
    minimum_distance = (contexts - contexts[selected[0]]).square().sum(dim=-1)
    while len(selected) < chart_count:
        score = weights * minimum_distance
        score[selected] = -1.0
        candidate = int(torch.argmax(score))
        if float(score[candidate]) <= 0.0:
            raise ValueError("chart farthest-point initialization degenerated")
        selected.append(candidate)
        distance = (contexts - contexts[candidate]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    centers = contexts[selected].clone()

    assignments = torch.empty(contexts.shape[0], dtype=torch.int64)
    for _ in range(lloyd_iterations):
        distances = (
            contexts[:, None, :] - centers[None, :, :]
        ).square().sum(dim=-1)
        assignments = torch.argmin(distances, dim=-1)
        updated = torch.empty_like(centers)
        for chart in range(chart_count):
            mask = assignments == chart
            if not bool(mask.any()):
                raise ValueError("weighted Lloyd fit produced an empty chart")
            chart_weights = weights[mask]
            updated[chart] = (
                chart_weights[:, None] * contexts[mask]
            ).sum(dim=0) / chart_weights.sum()
        centers = updated

    order = sorted(
        range(chart_count),
        key=lambda index: tuple(float(value) for value in centers[index]),
    )
    centers = centers[order].contiguous()
    distances = (
        contexts[:, None, :] - centers[None, :, :]
    ).square().sum(dim=-1)
    assignments = torch.argmin(distances, dim=-1)
    bases = []
    normal_energy = torch.zeros((), dtype=torch.float64)
    for chart in range(chart_count):
        mask = assignments == chart
        if int(mask.sum()) < chart_rank + 1:
            raise ValueError("one chart has too few rows for its tangent rank")
        centered = contexts[mask] - centers[chart]
        chart_weights = weights[mask]
        covariance = (centered.T * chart_weights) @ centered
        covariance = covariance / chart_weights.sum()
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -chart_rank:].flip(dims=(1,))
        basis = _canonicalize_basis_signs(basis)
        bases.append(basis)
        tangent = centered @ basis
        residual = (
            centered.square().sum(dim=-1)
            - tangent.square().sum(dim=-1)
        ).clamp_min(0.0)
        normal_energy += (chart_weights * residual).sum()
    temperature = float(temperature_scale) * float(
        normal_energy / weights.sum()
    )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("chart fit produced a degenerate normal temperature")
    return FrozenAttentionChartAtlas(
        centers=centers,
        bases=torch.stack(bases),
        temperature=temperature,
    )


@dataclass(frozen=True, slots=True)
class SparseAttentionTransportOracle:
    coefficients: Tensor
    support: Tensor
    objective: Tensor
    active_count_limit: int
    solver_kind: str
    coefficient_ridge: float

    def __post_init__(self) -> None:
        coefficients = _cpu_float64(
            self.coefficients,
            label="oracle coefficients",
            ndim=2,
        )
        support = _cpu_bool(self.support, label="oracle support", ndim=2)
        objective = _cpu_float64(self.objective, label="oracle objective", ndim=1)
        if (
            support.shape != coefficients.shape
            or objective.shape != (coefficients.shape[0],)
            or type(self.active_count_limit) is not int
            or not 0 < self.active_count_limit <= coefficients.shape[1]
            or self.solver_kind != EXACT_SPARSE_ORACLE_SOLVER
            or isinstance(self.coefficient_ridge, bool)
            or not isinstance(self.coefficient_ridge, (int, float))
            or not math.isfinite(float(self.coefficient_ridge))
            or float(self.coefficient_ridge) < 0.0
            or bool((support.sum(dim=-1) > self.active_count_limit).any())
            or bool((coefficients.masked_select(~support) != 0.0).any())
            or bool((objective < 0.0).any())
        ):
            raise ValueError("sparse oracle tensors are inconsistent")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "objective", objective)


def solve_fisher_sparse_transport_oracle(
    candidate_outputs: Tensor,
    target_outputs: Tensor,
    score_gradients: Tensor,
    *,
    active_count: int,
    euclidean_floor: float,
    coefficient_ridge: float,
) -> SparseAttentionTransportOracle:
    """Exactly solve signed Fisher-ridge coalitions up to one support budget.

    Every support of size zero through ``active_count`` is enumerated.  The
    coefficient solve is minimum-norm for the ridge-free held-out oracle and a
    positive-ridge solve for the A-only teacher used to train the router.
    """

    candidates = _cpu_float64(candidate_outputs, label="candidate_outputs", ndim=3)
    targets = _cpu_float64(target_outputs, label="target_outputs", ndim=2)
    gradients = _cpu_float64(score_gradients, label="score_gradients", ndim=2)
    rows, generators, hidden = candidates.shape
    if targets.shape != (rows, hidden) or gradients.shape != targets.shape:
        raise ValueError("oracle candidates, targets, and gradients do not align")
    if type(active_count) is not int or not 0 < active_count <= generators:
        raise ValueError("active_count is outside the generator inventory")
    for label, value, positive in (
        ("euclidean_floor", euclidean_floor, True),
        ("coefficient_ridge", coefficient_ridge, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (positive and float(value) <= 0.0)
            or (not positive and float(value) < 0.0)
        ):
            raise ValueError(f"{label} is invalid")

    coefficients = torch.zeros(rows, generators, dtype=torch.float64)
    support = torch.zeros(rows, generators, dtype=torch.bool)
    identity = torch.eye(generators, dtype=torch.float64)
    projected_gradient = torch.einsum("nkh,nh->nk", candidates, gradients)
    gram = (
        float(euclidean_floor)
        * torch.einsum("nkh,njh->nkj", candidates, candidates)
        + projected_gradient[:, :, None] * projected_gradient[:, None, :]
        + float(coefficient_ridge) * identity[None]
    )
    right = (
        float(euclidean_floor) * torch.einsum("nkh,nh->nk", candidates, targets)
        + projected_gradient
        * torch.einsum("nh,nh->n", gradients, targets)[:, None]
    )
    target_energy = (
        float(euclidean_floor) * targets.square().sum(dim=-1)
        + torch.einsum("nh,nh->n", gradients, targets).square()
    )
    objectives = target_energy.clone()
    machine_epsilon = torch.finfo(torch.float64).eps
    for support_size in range(1, active_count + 1):
        for selected in combinations(range(generators), support_size):
            index = torch.tensor(selected, dtype=torch.long)
            block = gram.index_select(1, index).index_select(2, index)
            selected_right = right.index_select(1, index)
            beta = (
                torch.linalg.solve(block, selected_right.unsqueeze(-1)).squeeze(-1)
                if coefficient_ridge > 0.0
                else torch.einsum(
                    "nij,nj->ni",
                    torch.linalg.pinv(block, hermitian=True),
                    selected_right,
                )
            )
            candidate_objective = (
                target_energy - (selected_right * beta).sum(dim=-1)
            ).clamp_min(0.0)
            tolerance = (
                256.0
                * machine_epsilon
                * torch.maximum(torch.ones_like(objectives), objectives.abs())
            )
            improved = candidate_objective < objectives - tolerance
            if not bool(improved.any()):
                continue
            candidate_coefficients = torch.zeros_like(coefficients)
            candidate_coefficients[:, index] = beta
            candidate_support = torch.zeros_like(support)
            candidate_support[:, index] = True
            coefficients = torch.where(
                improved[:, None],
                candidate_coefficients,
                coefficients,
            )
            support = torch.where(improved[:, None], candidate_support, support)
            objectives = torch.where(
                improved,
                candidate_objective,
                objectives,
            )
    return SparseAttentionTransportOracle(
        coefficients=coefficients,
        support=support,
        objective=objectives,
        active_count_limit=active_count,
        solver_kind=EXACT_SPARSE_ORACLE_SOLVER,
        coefficient_ridge=float(coefficient_ridge),
    )


@dataclass(frozen=True, slots=True)
class ChartConditionedRouterConfig:
    active_count: int
    oracle_solver_kind: str = EXACT_SPARSE_ORACLE_SOLVER
    steps: int = 384
    learning_rate: float = 5.0e-2
    euclidean_floor: float = 1.0e-3
    oracle_ridge: float = 1.0e-8
    route_kl_coefficient: float = 0.1
    entropy_coefficient: float = 1.0e-3
    balance_coefficient: float = 1.0e-3
    coefficient_mse_coefficient: float = 0.1
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if type(self.active_count) is not int or self.active_count <= 0:
            raise ValueError("active_count must be positive")
        if self.oracle_solver_kind != EXACT_SPARSE_ORACLE_SOLVER:
            raise ValueError("router teacher oracle solver kind is invalid")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be nonnegative")
        for label, value, positive in (
            ("learning_rate", self.learning_rate, True),
            ("euclidean_floor", self.euclidean_floor, True),
            ("oracle_ridge", self.oracle_ridge, False),
            ("route_kl_coefficient", self.route_kl_coefficient, False),
            ("entropy_coefficient", self.entropy_coefficient, False),
            ("balance_coefficient", self.balance_coefficient, False),
            ("coefficient_mse_coefficient", self.coefficient_mse_coefficient, False),
            ("gradient_clip_norm", self.gradient_clip_norm, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (positive and float(value) <= 0.0)
                or (not positive and float(value) < 0.0)
            ):
                raise ValueError(f"{label} is invalid")


@dataclass(frozen=True, slots=True)
class ChartConditionedRouterExecution:
    chart_coordinates: Tensor
    chart_memberships: Tensor
    contextual_features: Tensor
    selection_probabilities: Tensor
    sparse_selection_probabilities: Tensor
    signed_amplitudes: Tensor
    active_mask: Tensor
    generator_coefficients: Tensor


@dataclass(frozen=True, slots=True)
class ChartConditionedAttentionTransportRouter:
    invocation_id: str
    generator_bank_sha256: str
    atlas: FrozenAttentionChartAtlas
    selection_weight: Tensor
    selection_bias: Tensor
    amplitude_weight: Tensor
    amplitude_bias: Tensor
    active_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ValueError("router invocation_id must be nonempty")
        if (
            not isinstance(self.generator_bank_sha256, str)
            or len(self.generator_bank_sha256) != 64
        ):
            raise ValueError("router generator bank hash is invalid")
        if not isinstance(self.atlas, FrozenAttentionChartAtlas):
            raise TypeError("atlas must be a FrozenAttentionChartAtlas")
        selection_weight = _cpu_float64(
            self.selection_weight,
            label="selection_weight",
            ndim=3,
        )
        selection_bias = _cpu_float64(
            self.selection_bias,
            label="selection_bias",
            ndim=2,
        )
        amplitude_weight = _cpu_float64(
            self.amplitude_weight,
            label="amplitude_weight",
            ndim=3,
        )
        amplitude_bias = _cpu_float64(
            self.amplitude_bias,
            label="amplitude_bias",
            ndim=2,
        )
        expected_prefix = (self.atlas.chart_count, selection_bias.shape[1])
        if (
            selection_bias.shape != expected_prefix
            or amplitude_bias.shape != expected_prefix
            or selection_weight.shape
            != expected_prefix + (self.atlas.chart_rank + selection_bias.shape[1],)
            or amplitude_weight.shape != selection_weight.shape
            or type(self.active_count) is not int
            or not 0 < self.active_count <= selection_bias.shape[1]
        ):
            raise ValueError("router parameter shapes are invalid")
        for name, value in (
            ("selection_weight", selection_weight),
            ("selection_bias", selection_bias),
            ("amplitude_weight", amplitude_weight),
            ("amplitude_bias", amplitude_bias),
        ):
            object.__setattr__(self, name, value)

    @property
    def generator_count(self) -> int:
        return int(self.selection_bias.shape[1])

    @property
    def stored_parameter_count(self) -> int:
        return int(
            self.selection_weight.numel()
            + self.selection_bias.numel()
            + self.amplitude_weight.numel()
            + self.amplitude_bias.numel()
        )

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "active_count": self.active_count,
            "amplitude_bias_sha256": _tensor_sha256(self.amplitude_bias),
            "amplitude_weight_sha256": _tensor_sha256(self.amplitude_weight),
            "atlas_sha256": self.atlas.artifact_sha256,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "generator_bank_sha256": self.generator_bank_sha256,
            "invocation_id": self.invocation_id,
            "selection_bias_sha256": _tensor_sha256(self.selection_bias),
            "selection_weight_sha256": _tensor_sha256(self.selection_weight),
        }
        return hashlib.sha256(
            _ROUTER_DOMAIN
            + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def state_dict(self) -> dict[str, object]:
        return {
            "active_count": self.active_count,
            "amplitude_bias": self.amplitude_bias.clone(),
            "amplitude_weight": self.amplitude_weight.clone(),
            "artifact_sha256": self.artifact_sha256,
            "atlas_sha256": self.atlas.artifact_sha256,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_teacher_outputs_or_gradients": False,
            "generator_bank_sha256": self.generator_bank_sha256,
            "invocation_id": self.invocation_id,
            "selection_bias": self.selection_bias.clone(),
            "selection_weight": self.selection_weight.clone(),
        }

    def execute(
        self,
        chart_inputs: Tensor,
        contextual_features: Tensor,
    ) -> ChartConditionedRouterExecution:
        coordinates, memberships = self.atlas.coordinates_and_memberships(chart_inputs)
        context = _cpu_float64(
            contextual_features,
            label="router contextual_features",
            ndim=2,
        )
        if context.shape != (coordinates.shape[0], self.generator_count):
            raise ValueError("router contextual features do not align with rows")
        features = torch.cat(
            (
                coordinates,
                context[:, None, :].expand(-1, self.atlas.chart_count, -1),
            ),
            dim=-1,
        )
        logits = (
            torch.einsum("ncf,ckf->nck", features, self.selection_weight)
            + self.selection_bias[None, :, :]
        )
        probabilities = torch.softmax(logits, dim=-1)
        amplitudes = (
            torch.einsum("ncf,ckf->nck", features, self.amplitude_weight)
            + self.amplitude_bias[None, :, :]
        )
        indices = torch.topk(
            probabilities,
            k=self.active_count,
            dim=-1,
            sorted=True,
        ).indices
        active = torch.zeros_like(probabilities, dtype=torch.bool)
        active.scatter_(-1, indices, True)
        sparse = probabilities.masked_fill(~active, 0.0)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        coefficients = torch.einsum(
            "nc,nck->nk",
            memberships,
            sparse * amplitudes,
        )
        return ChartConditionedRouterExecution(
            chart_coordinates=coordinates,
            chart_memberships=memberships,
            contextual_features=context,
            selection_probabilities=probabilities,
            sparse_selection_probabilities=sparse,
            signed_amplitudes=amplitudes,
            active_mask=active,
            generator_coefficients=coefficients,
        )

    def predict_operator_outputs(
        self,
        bank: FisherAttentionTransportGeneratorBank,
        *,
        head_contexts: Tensor,
        chart_inputs: Tensor,
    ) -> Tensor:
        if bank.artifact_sha256 != self.generator_bank_sha256:
            raise ValueError("router is not bound to this generator bank")
        contextual_features = bank.router_context_features_from_head_contexts(
            invocation_id=self.invocation_id,
            head_contexts=head_contexts,
        )
        execution = self.execute(chart_inputs, contextual_features)
        candidates = bank.candidate_outputs_from_head_contexts(
            invocation_id=self.invocation_id,
            head_contexts=head_contexts,
        )
        if candidates.shape[:2] != execution.generator_coefficients.shape:
            raise ValueError("router rows do not align with contextual candidates")
        return torch.einsum(
            "nk,nkh->nh",
            execution.generator_coefficients,
            candidates,
        ).contiguous()


def chart_conditioned_attention_transport_router_from_state_dict(
    state: Mapping[str, object],
    *,
    atlas: FrozenAttentionChartAtlas,
) -> ChartConditionedAttentionTransportRouter:
    expected = {
        "active_count",
        "amplitude_bias",
        "amplitude_weight",
        "artifact_sha256",
        "atlas_sha256",
        "contains_oracle_outputs_or_coefficients",
        "contains_teacher_outputs_or_gradients",
        "generator_bank_sha256",
        "invocation_id",
        "selection_bias",
        "selection_weight",
    }
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("chart-conditioned router state fields are invalid")
    if (
        state["atlas_sha256"] != atlas.artifact_sha256
        or state["contains_oracle_outputs_or_coefficients"] is not False
        or state["contains_teacher_outputs_or_gradients"] is not False
    ):
        raise ValueError("chart-conditioned router state lineage is invalid")
    router = ChartConditionedAttentionTransportRouter(
        invocation_id=state["invocation_id"],
        generator_bank_sha256=state["generator_bank_sha256"],
        atlas=atlas,
        selection_weight=state["selection_weight"],
        selection_bias=state["selection_bias"],
        amplitude_weight=state["amplitude_weight"],
        amplitude_bias=state["amplitude_bias"],
        active_count=state["active_count"],
    )
    if state["artifact_sha256"] != router.artifact_sha256:
        raise ValueError("chart-conditioned router artifact hash drifted")
    return router


@dataclass(frozen=True, slots=True)
class ChartConditionedRouterFit:
    router: ChartConditionedAttentionTransportRouter
    oracle: SparseAttentionTransportOracle
    oracle_output_metrics: AttentionTransportMetrics
    router_output_metrics: AttentionTransportMetrics


@dataclass(frozen=True, slots=True)
class AttentionTransportRouterEvaluation:
    output_metrics: AttentionTransportMetrics
    oracle_output_metrics: AttentionTransportMetrics
    oracle_solver_kind: str
    oracle_active_count_limit: int
    oracle_coefficient_ridge: float
    zero_predictor_fisher_loss: float
    relative_fisher_loss: float
    oracle_relative_fisher_loss: float
    oracle_router_fisher_gap: float
    active_count_minimum: int
    active_count_mean: float
    active_count_maximum: int
    effective_support_count_minimum: int
    effective_support_count_mean: float
    effective_support_count_maximum: int
    sparse_probability_renormalization_max_error: float
    generator_usage: Tensor
    dead_generator_count: int

    def __post_init__(self) -> None:
        usage = _cpu_float64(
            self.generator_usage,
            label="generator_usage",
            ndim=1,
        )
        if (
            self.oracle_solver_kind != EXACT_SPARSE_ORACLE_SOLVER
            or type(self.oracle_active_count_limit) is not int
            or not 0 < self.oracle_active_count_limit <= usage.shape[0]
            or type(self.oracle_coefficient_ridge) is not float
            or self.oracle_coefficient_ridge != HELDOUT_ORACLE_COEFFICIENT_RIDGE
            or type(self.active_count_minimum) is not int
            or type(self.active_count_maximum) is not int
            or not 0 < self.active_count_minimum <= self.active_count_maximum
            or self.active_count_minimum != self.oracle_active_count_limit
            or self.active_count_maximum != self.oracle_active_count_limit
            or type(self.effective_support_count_minimum) is not int
            or type(self.effective_support_count_maximum) is not int
            or not 0
            < self.effective_support_count_minimum
            <= self.effective_support_count_maximum
            <= usage.shape[0]
            or not math.isfinite(float(self.active_count_mean))
            or not math.isfinite(float(self.effective_support_count_mean))
            or not self.active_count_minimum
            <= float(self.active_count_mean)
            <= self.active_count_maximum
            or not self.effective_support_count_minimum
            <= float(self.effective_support_count_mean)
            <= self.effective_support_count_maximum
            or bool((usage < 0.0).any())
            or not torch.allclose(
                usage.sum(),
                torch.ones((), dtype=torch.float64),
                atol=1.0e-10,
                rtol=1.0e-10,
            )
            or self.dead_generator_count != int((usage == 0.0).sum())
        ):
            raise ValueError("router evaluation generator usage is invalid")
        object.__setattr__(self, "generator_usage", usage)


def _training_router_execution(
    features: Tensor,
    memberships: Tensor,
    selection_weight: Tensor,
    selection_bias: Tensor,
    amplitude_weight: Tensor,
    amplitude_bias: Tensor,
    *,
    active_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    logits = (
        torch.einsum("ncf,ckf->nck", features, selection_weight)
        + selection_bias[None]
    )
    probabilities = torch.softmax(logits, dim=-1)
    indices = torch.topk(probabilities, k=active_count, dim=-1).indices
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    hard = probabilities.masked_fill(~mask, 0.0)
    hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    straight_through = probabilities + (hard - probabilities).detach()
    amplitudes = (
        torch.einsum("ncf,ckf->nck", features, amplitude_weight)
        + amplitude_bias[None]
    )
    beta = torch.einsum(
        "nc,nck->nk",
        memberships.detach(),
        straight_through * amplitudes,
    )
    return beta, probabilities, amplitudes


def fit_chart_conditioned_attention_transport_router(
    bank: FisherAttentionTransportGeneratorBank,
    capture: AttentionTransportBatch | Sequence[AttentionTransportBatch],
    atlas: FrozenAttentionChartAtlas,
    *,
    config: ChartConditionedRouterConfig,
) -> ChartConditionedRouterFit:
    """Fit a deployable TopK router after freezing the shared generator bank."""

    if not isinstance(bank, FisherAttentionTransportGeneratorBank):
        raise TypeError("bank must be a FisherAttentionTransportGeneratorBank")
    capture_values = (
        (capture,)
        if isinstance(capture, (AttentionTransportCapture, AttentionTransportRowBatch))
        else tuple(capture)
        if isinstance(capture, Sequence)
        and not isinstance(capture, (str, bytes))
        else ()
    )
    if not capture_values or any(
        not isinstance(value, (AttentionTransportCapture, AttentionTransportRowBatch))
        for value in capture_values
    ):
        raise TypeError("capture must contain attention transport captures")
    captures, invocations, _, _ = _capture_catalog(capture_values)
    if len(invocations) != 1:
        raise ValueError("one router fit may cover only one invocation")
    representative = invocations[0]
    if not isinstance(atlas, FrozenAttentionChartAtlas):
        raise TypeError("atlas must be a FrozenAttentionChartAtlas")
    if not isinstance(config, ChartConditionedRouterConfig):
        raise TypeError("config must be a ChartConditionedRouterConfig")
    if (
        atlas.hidden_width != bank.hidden_width
        or config.active_count > bank.generator_count
    ):
        raise ValueError("atlas, router, and bank dimensions are inconsistent")

    candidates = torch.cat(
        tuple(bank.candidate_outputs(value) for value in captures)
    )
    targets = torch.cat(
        tuple(value.valid_operator_outputs for value in captures)
    )
    gradients = torch.cat(
        tuple(value.valid_score_gradients for value in captures)
    )
    chart_inputs = torch.cat(
        tuple(value.valid_chart_inputs for value in captures)
    )
    head_contexts = torch.cat(
        tuple(value.head_contexts for value in captures)
    )
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        targets,
        gradients,
        active_count=config.active_count,
        euclidean_floor=config.euclidean_floor,
        coefficient_ridge=config.oracle_ridge,
    )
    oracle_prediction = torch.einsum(
        "nk,nkh->nh",
        oracle.coefficients,
        candidates,
    )
    coordinates, memberships = atlas.coordinates_and_memberships(chart_inputs)
    contextual_features = bank.router_context_features_from_head_contexts(
        invocation_id=representative.invocation_id,
        head_contexts=head_contexts,
    )
    features = torch.cat(
        (
            coordinates,
            contextual_features[:, None, :].expand(-1, atlas.chart_count, -1),
        ),
        dim=-1,
    )
    charts = atlas.chart_count
    generators = bank.generator_count
    feature_count = atlas.chart_rank + generators
    target_mass = oracle.coefficients.abs()
    row_mass = target_mass.sum(dim=-1, keepdim=True)
    target_probabilities = target_mass / row_mass.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    zero_rows = row_mass.squeeze(-1) == 0.0
    if bool(zero_rows.any()):
        target_probabilities[zero_rows] = 1.0 / generators
    chart_mass = memberships.detach().sum(dim=0).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    target_chart_usage = torch.einsum(
        "nc,nk->ck",
        memberships.detach(),
        target_probabilities,
    ) / chart_mass[:, None]

    selection_weight = nn.Parameter(
        torch.zeros(charts, generators, feature_count, dtype=torch.float64)
    )
    selection_bias = nn.Parameter(
        target_chart_usage.clamp_min(1.0e-6).log().contiguous()
    )
    amplitude_weight = nn.Parameter(
        torch.zeros(charts, generators, feature_count, dtype=torch.float64)
    )
    mean_beta = torch.einsum(
        "nc,nk->ck",
        memberships.detach(),
        oracle.coefficients,
    ) / chart_mass[:, None]
    initial_indices = torch.topk(
        target_chart_usage,
        k=config.active_count,
        dim=-1,
    ).indices
    initial_active = torch.zeros_like(target_chart_usage, dtype=torch.bool)
    initial_active.scatter_(-1, initial_indices, True)
    initial_sparse = target_chart_usage.masked_fill(~initial_active, 0.0)
    initial_sparse = initial_sparse / initial_sparse.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(torch.float64).tiny)
    amplitude_bias = nn.Parameter(
        torch.where(
            initial_active,
            mean_beta / initial_sparse.clamp_min(1.0e-6),
            torch.zeros_like(mean_beta),
        ).contiguous()
    )
    optimizer = torch.optim.Adam(
        (selection_weight, selection_bias, amplitude_weight, amplitude_bias),
        lr=float(config.learning_rate),
    )
    tiny = torch.finfo(torch.float64).tiny
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        beta, probabilities, _ = _training_router_execution(
            features,
            memberships,
            selection_weight,
            selection_bias,
            amplitude_weight,
            amplitude_bias,
            active_count=config.active_count,
        )
        prediction = torch.einsum("nk,nkh->nh", beta, candidates)
        output_loss = samplewise_joint_fisher_loss(
            prediction,
            targets,
            gradients,
            euclidean_floor=config.euclidean_floor,
        )
        log_probabilities = probabilities.clamp_min(tiny).log()
        target_log = target_probabilities.clamp_min(tiny).log()
        per_chart_kl = torch.einsum(
            "nk,nck->nc",
            target_probabilities,
            target_log[:, None, :] - log_probabilities,
        )
        route_kl = (
            memberships.detach() * per_chart_kl
        ).sum(dim=-1).mean()
        entropy = (
            memberships.detach()
            * (-(probabilities * log_probabilities).sum(dim=-1))
        ).sum(dim=-1).mean()
        predicted_chart_usage = torch.einsum(
            "nc,nck->ck",
            memberships.detach(),
            probabilities,
        ) / chart_mass[:, None]
        balance = (predicted_chart_usage - target_chart_usage).square().mean()
        coefficient_mse = (beta - oracle.coefficients).square().mean()
        loss = (
            output_loss
            + float(config.route_kl_coefficient) * route_kl
            + float(config.entropy_coefficient) * entropy
            + float(config.balance_coefficient) * balance
            + float(config.coefficient_mse_coefficient) * coefficient_mse
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("chart-conditioned router fit became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (selection_weight, selection_bias, amplitude_weight, amplitude_bias),
            max_norm=float(config.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()

    router = ChartConditionedAttentionTransportRouter(
        invocation_id=representative.invocation_id,
        generator_bank_sha256=bank.artifact_sha256,
        atlas=atlas,
        selection_weight=selection_weight.detach(),
        selection_bias=selection_bias.detach(),
        amplitude_weight=amplitude_weight.detach(),
        amplitude_bias=amplitude_bias.detach(),
        active_count=config.active_count,
    )
    router_prediction = router.predict_operator_outputs(
        bank,
        head_contexts=head_contexts,
        chart_inputs=chart_inputs,
    )
    return ChartConditionedRouterFit(
        router=router,
        oracle=oracle,
        oracle_output_metrics=_metrics(
            oracle_prediction,
            targets,
            gradients,
            config.euclidean_floor,
        ),
        router_output_metrics=_metrics(
            router_prediction,
            targets,
            gradients,
            config.euclidean_floor,
        ),
    )


def evaluate_chart_conditioned_attention_transport_router(
    bank: FisherAttentionTransportGeneratorBank,
    router: ChartConditionedAttentionTransportRouter,
    captures: Sequence[AttentionTransportBatch],
    *,
    euclidean_floor: float,
) -> AttentionTransportRouterEvaluation:
    """Evaluate one frozen router against an exact ridge-free held-out oracle."""

    values, invocations, _, _ = _capture_catalog(captures)
    if len(invocations) != 1 or invocations[0].invocation_id != router.invocation_id:
        raise ValueError("router evaluation captures must cover its invocation")
    if bank.artifact_sha256 != router.generator_bank_sha256:
        raise ValueError("router evaluation bank lineage drifted")
    candidates = torch.cat(tuple(bank.candidate_outputs(value) for value in values))
    targets = torch.cat(tuple(value.valid_operator_outputs for value in values))
    gradients = torch.cat(tuple(value.valid_score_gradients for value in values))
    predictions = []
    active_counts = []
    effective_support_counts = []
    sparse_errors = []
    usage = torch.zeros(bank.generator_count, dtype=torch.float64)
    usage_denominator = 0.0
    for value in values:
        contextual = bank.router_context_features_from_head_contexts(
            invocation_id=router.invocation_id,
            head_contexts=value.head_contexts,
        )
        execution = router.execute(value.valid_chart_inputs, contextual)
        predictions.append(
            router.predict_operator_outputs(
                bank,
                head_contexts=value.head_contexts,
                chart_inputs=value.valid_chart_inputs,
            )
        )
        active_counts.append(execution.active_mask.sum(dim=-1).reshape(-1))
        effective_support_counts.append(
            execution.active_mask.any(dim=1).sum(dim=-1)
        )
        sparse_errors.append(
            (
                execution.sparse_selection_probabilities.sum(dim=-1) - 1.0
            ).abs().reshape(-1)
        )
        weighted_active = (
            execution.chart_memberships[:, :, None]
            * execution.active_mask.to(dtype=torch.float64)
        ).sum(dim=(0, 1))
        usage += weighted_active
        usage_denominator += float(execution.chart_memberships.sum())
    prediction = torch.cat(predictions)
    active = torch.cat(active_counts)
    effective_support = torch.cat(effective_support_counts)
    renormalization = torch.cat(sparse_errors)
    usage = usage / usage_denominator
    usage = usage / usage.sum().clamp_min(torch.finfo(torch.float64).tiny)
    output_metrics = _metrics(
        prediction,
        targets,
        gradients,
        euclidean_floor,
    )
    zero_loss = float(
        samplewise_joint_fisher_loss(
            torch.zeros_like(targets),
            targets,
            gradients,
            euclidean_floor=euclidean_floor,
        )
    )
    if not math.isfinite(zero_loss) or zero_loss <= 0.0:
        raise ValueError("zero-predictor Fisher loss is not evaluable")
    oracle = solve_fisher_sparse_transport_oracle(
        candidates,
        targets,
        gradients,
        active_count=router.active_count,
        euclidean_floor=euclidean_floor,
        coefficient_ridge=HELDOUT_ORACLE_COEFFICIENT_RIDGE,
    )
    oracle_prediction = torch.einsum(
        "nk,nkh->nh",
        oracle.coefficients,
        candidates,
    )
    oracle_metrics = _metrics(
        oracle_prediction,
        targets,
        gradients,
        euclidean_floor,
    )
    relative = output_metrics.fisher_loss / zero_loss
    oracle_relative = oracle_metrics.fisher_loss / zero_loss
    return AttentionTransportRouterEvaluation(
        output_metrics=output_metrics,
        oracle_output_metrics=oracle_metrics,
        oracle_solver_kind=oracle.solver_kind,
        oracle_active_count_limit=oracle.active_count_limit,
        oracle_coefficient_ridge=oracle.coefficient_ridge,
        zero_predictor_fisher_loss=zero_loss,
        relative_fisher_loss=float(relative),
        oracle_relative_fisher_loss=float(oracle_relative),
        oracle_router_fisher_gap=float(relative - oracle_relative),
        active_count_minimum=int(active.min()),
        active_count_mean=float(active.to(dtype=torch.float64).mean()),
        active_count_maximum=int(active.max()),
        effective_support_count_minimum=int(effective_support.min()),
        effective_support_count_mean=float(
            effective_support.to(dtype=torch.float64).mean()
        ),
        effective_support_count_maximum=int(effective_support.max()),
        sparse_probability_renormalization_max_error=float(
            renormalization.max()
        ),
        generator_usage=usage,
        dead_generator_count=int((usage == 0.0).sum()),
    )


@dataclass(frozen=True, slots=True)
class AttentionTransportResourceAccounting:
    native_v_o_parameters: int
    generator_bank_parameters: int
    chart_parameters: int
    router_parameters: int
    compiled_parameters: int
    parameter_reduction_fraction: float
    parameter_compression_eligible: bool
    native_qk_retained: bool = True
    native_output_rmsnorm_retained: bool = True
    logical_speedup_claimed: bool = False


def attention_transport_resource_accounting(
    bank: FisherAttentionTransportGeneratorBank,
    captures: Sequence[AttentionTransportBatch],
    routers: Sequence[ChartConditionedAttentionTransportRouter] = (),
) -> AttentionTransportResourceAccounting:
    """Close storage without converting parameter savings into a speed claim."""

    _, invocations, head_ids, _ = _capture_catalog(captures)
    if len(head_ids) != len(bank.head_ids) or set(head_ids) != set(bank.head_ids):
        raise ValueError("resource captures do not match the generator bank")
    router_values = tuple(routers)
    if any(
        not isinstance(value, ChartConditionedAttentionTransportRouter)
        for value in router_values
    ):
        raise TypeError("routers must contain chart-conditioned routers")
    if len({value.invocation_id for value in router_values}) != len(router_values):
        raise ValueError("router invocation ids must be unique")
    if any(
        value.generator_bank_sha256 != bank.artifact_sha256
        for value in router_values
    ):
        raise ValueError("one router is not bound to the generator bank")
    native = sum(value.native_v_o_parameter_count for value in invocations)
    bank_count = bank.stored_parameter_count
    unique_atlases = {
        value.atlas.artifact_sha256: value.atlas for value in router_values
    }
    chart_count = sum(
        value.stored_parameter_count for value in unique_atlases.values()
    )
    router_count = sum(value.stored_parameter_count for value in router_values)
    compiled = bank_count + chart_count + router_count
    return AttentionTransportResourceAccounting(
        native_v_o_parameters=native,
        generator_bank_parameters=bank_count,
        chart_parameters=chart_count,
        router_parameters=router_count,
        compiled_parameters=compiled,
        parameter_reduction_fraction=float(1.0 - compiled / native),
        parameter_compression_eligible=compiled < native,
    )


__all__ = [
    "AttentionTransportBankFit",
    "AttentionTransportBatch",
    "AttentionTransportCapture",
    "AttentionTransportFitConfig",
    "AttentionTransportHeadOperatorCatalog",
    "AttentionTransportMetrics",
    "AttentionTransportResourceAccounting",
    "AttentionTransportRouterEvaluation",
    "AttentionTransportRowBatch",
    "ChartConditionedAttentionTransportRouter",
    "ChartConditionedRouterConfig",
    "ChartConditionedRouterExecution",
    "ChartConditionedRouterFit",
    "EXACT_SPARSE_ORACLE_SOLVER",
    "FisherAttentionTransportGeneratorBank",
    "FrozenAttentionChartAtlas",
    "HELDOUT_ORACLE_COEFFICIENT_RIDGE",
    "SparseAttentionTransportOracle",
    "attention_transport_capture_from_exact_attention",
    "attention_transport_row_batch_from_capture",
    "attention_transport_resource_accounting",
    "chart_conditioned_attention_transport_router_from_state_dict",
    "evaluate_chart_conditioned_attention_transport_router",
    "fisher_attention_transport_generator_bank_from_state_dict",
    "fit_chart_conditioned_attention_transport_router",
    "fit_frozen_attention_chart_atlas",
    "fit_fisher_attention_transport_generator_bank",
    "frozen_attention_chart_atlas_from_state_dict",
    "samplewise_joint_fisher_loss",
    "solve_fisher_sparse_transport_oracle",
]
