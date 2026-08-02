"""Authenticated native reverse-mode execution of the Gemma post-H4 suffix."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields as dataclass_fields
import hashlib
import math

import torch
from torch import Tensor

from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _canonical_json_bytes,
    _require_sha256,
    _runtime_tensor_sha256,
)
from .gemma3_l3_l4_h4_suffix_jvp_runtime import (
    GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS,
    GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS,
    Gemma3L3L4H4SuffixJVPRuntime,
    _bitwise_equal,
    _finite_tensor,
    _materialized_tensor,
)

__all__ = [
    "GEMMA3_L3_L4_H4_SUFFIX_VJP_TOKEN_CHUNK_SIZE",
    "Gemma3L3L4H4SuffixVJP",
    "Gemma3L3L4H4SuffixVJPChunkReceipt",
    "Gemma3L3L4H4SuffixVJPReceipt",
    "Gemma3L3L4H4SuffixVJPRuntime",
    "gemma3_l3_l4_h4_suffix_vjp_resource_accounting",
    "require_gemma3_l3_l4_h4_suffix_vjp_complete_panel",
]

GEMMA3_L3_L4_H4_SUFFIX_VJP_TOKEN_CHUNK_SIZE = 8
_CHUNK_DOMAIN = b"fisher-graph:h4-suffix-vjp-chunk:v13\0"
_RECEIPT_DOMAIN = b"fisher-graph:h4-suffix-vjp-receipt:v13\0"

_COMPLETE = {
    "suffix_vjp_node_count": 64,
    "suffix_vjp_primal_vector_count": 64,
    "suffix_vjp_token_directional_derivative_count": 3_212,
    "suffix_segment_call_count": 832,
    "logit_projection_call_count": 64,
    "h4_dtype_cast_count": 64,
    "vjp_transform_count": 64,
    "vjp_pullback_chunk_call_count": 436,
    "vmap_pullback_call_count": 436,
    "canonical_token_cotangent_coverage_count": 3_212,
    "canonical_token_cotangent_nonzero_count": 3_212,
    "canonical_token_cotangent_element_observation_count": 174_292,
    "full_h4_row_observation_count": 3_788,
    "support_h4_row_observation_count": 3_276,
    "outside_support_h4_row_observation_count": 512,
    "direction_coordinate_validation_count": 2_424_320,
    "outside_support_direction_zero_validation_count": 327_680,
    "full_h4_cotangent_coordinate_observation_count": 130_048_000,
    "support_h4_cotangent_coordinate_observation_count": 113_602_560,
    "outside_support_h4_cotangent_coordinate_observation_count": 16_445_440,
    "direction_contraction_coordinate_product_count": 113_602_560,
    "full_h4_cotangent_sha256_count": 436,
    "support_h4_cotangent_sha256_count": 436,
    "contracted_directional_chunk_sha256_count": 436,
    "retained_full_h4_cotangent_count": 0,
    "serialized_full_h4_cotangent_count": 0,
    "resource_counts_are_not_FLOPs_or_total_model_compute": True,
}


def _artifact_sha256(domain: bytes, payload: Mapping[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(payload)).hexdigest()


def _shape(value: object, *, rank: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != rank
        or any(type(size) is not int or size <= 0 for size in value)
    ):
        raise ValueError(f"{label} must be a positive rank-{rank} shape")
    return value


def _support_indices(value: object, *, sequence_length: int) -> Tensor:
    indices = _materialized_tensor(value, label="suffix VJP support indices")
    if (
        indices.dtype != torch.int64
        or indices.device.type != "cpu"
        or indices.ndim != 1
        or indices.numel() <= 0
        or indices.requires_grad
        or not indices.is_contiguous()
    ):
        raise ValueError(
            "suffix VJP support indices must be nonempty contiguous CPU int64"
        )
    result = indices.detach().clone().contiguous()
    if (
        int(result[0]) < 0
        or int(result[-1]) >= sequence_length
        or (
            result.numel() > 1
            and not bool((result[1:] > result[:-1]).all())
        )
    ):
        raise ValueError("suffix VJP support indices escape or reorder H4")
    return result


def _canonical_token_cotangent_chunk(
    *, start: int, stop: int, token_count: int, device: torch.device
) -> Tensor:
    if (
        type(start) is not int
        or type(stop) is not int
        or type(token_count) is not int
        or not 0 <= start < stop <= token_count
        or stop - start > GEMMA3_L3_L4_H4_SUFFIX_VJP_TOKEN_CHUNK_SIZE
    ):
        raise ValueError("suffix VJP token cotangent chunk bounds are invalid")
    result = torch.zeros(
        (stop - start, token_count), dtype=torch.float64, device=device
    )
    local = torch.arange(stop - start, device=device)
    result[local, start + local] = 1.0
    return result.contiguous()


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4SuffixVJPChunkReceipt:
    """Hashes transient full/support cotangents without retaining tensors."""

    token_start: int
    token_stop: int
    token_count: int
    support_row_count: int
    token_cotangent_sha256: str
    full_h4_cotangent_sha256: str
    support_h4_cotangent_sha256: str
    contracted_directional_token_teacher_kl_sha256: str
    full_h4_cotangent_shape: tuple[int, int, int, int]
    support_h4_cotangent_shape: tuple[int, int, int, int]
    device: str
    pullback_mechanism: str = "torch.func.vmap(pullback)"
    canonical_one_hot_token_cotangents: bool = True
    vmap_pullback_call_count: int = 1
    token_cotangent_chunk_size: int = 8
    token_cotangent_dtype: str = "torch.float64"
    full_h4_cotangent_dtype: str = "torch.float64"
    support_h4_cotangent_dtype: str = "torch.float64"
    contraction_dtype: str = "torch.float64"
    token_cotangent_nonzero_count: int = field(init=False)
    token_cotangent_element_count: int = field(init=False)
    full_h4_cotangent_coordinate_count: int = field(init=False)
    support_h4_cotangent_coordinate_count: int = field(init=False)
    outside_support_h4_cotangent_coordinate_count: int = field(init=False)
    full_h4_cotangent_retained: bool = False
    full_h4_cotangent_hashed: bool = True
    full_h4_cotangent_serialized: bool = False
    support_h4_cotangent_retained: bool = False
    support_h4_cotangent_hashed: bool = True
    support_h4_cotangent_serialized: bool = False
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.token_start) is not int
            or type(self.token_stop) is not int
            or type(self.token_count) is not int
            or not 0 <= self.token_start < self.token_stop <= self.token_count
            or self.token_stop - self.token_start > 8
            or type(self.support_row_count) is not int
            or self.support_row_count <= 0
        ):
            raise ValueError("suffix VJP chunk coverage is invalid")
        for value, label in (
            (self.token_cotangent_sha256, "token cotangent"),
            (self.full_h4_cotangent_sha256, "full H4 cotangent"),
            (self.support_h4_cotangent_sha256, "support H4 cotangent"),
            (
                self.contracted_directional_token_teacher_kl_sha256,
                "contracted directional chunk",
            ),
        ):
            _require_sha256(value, label=f"suffix VJP {label}")
        full_shape = _shape(
            self.full_h4_cotangent_shape,
            rank=4,
            label="suffix VJP full H4 cotangent",
        )
        support_shape = _shape(
            self.support_h4_cotangent_shape,
            rank=4,
            label="suffix VJP support H4 cotangent",
        )
        width = self.token_stop - self.token_start
        if (
            full_shape[0] != width
            or support_shape[0] != width
            or support_shape[1] != full_shape[1]
            or support_shape[2] != self.support_row_count
            or support_shape[2] > full_shape[2]
            or support_shape[3] != full_shape[3]
        ):
            raise ValueError("suffix VJP full/support cotangent geometry differs")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("suffix VJP chunk device must be nonempty")
        if (
            self.pullback_mechanism != "torch.func.vmap(pullback)"
            or self.canonical_one_hot_token_cotangents is not True
            or self.vmap_pullback_call_count != 1
            or self.token_cotangent_chunk_size != 8
            or self.token_cotangent_dtype != "torch.float64"
            or self.full_h4_cotangent_dtype != "torch.float64"
            or self.support_h4_cotangent_dtype != "torch.float64"
            or self.contraction_dtype != "torch.float64"
            or self.full_h4_cotangent_retained is not False
            or self.full_h4_cotangent_hashed is not True
            or self.full_h4_cotangent_serialized is not False
            or self.support_h4_cotangent_retained is not False
            or self.support_h4_cotangent_hashed is not True
            or self.support_h4_cotangent_serialized is not False
        ):
            raise ValueError("suffix VJP chunk dtype or storage policy differs")
        object.__setattr__(self, "full_h4_cotangent_shape", full_shape)
        object.__setattr__(self, "support_h4_cotangent_shape", support_shape)
        object.__setattr__(self, "token_cotangent_nonzero_count", width)
        object.__setattr__(
            self, "token_cotangent_element_count", width * self.token_count
        )
        full_count = math.prod(full_shape)
        support_count = math.prod(support_shape)
        object.__setattr__(
            self, "full_h4_cotangent_coordinate_count", full_count
        )
        object.__setattr__(
            self, "support_h4_cotangent_coordinate_count", support_count
        )
        object.__setattr__(
            self,
            "outside_support_h4_cotangent_coordinate_count",
            full_count - support_count,
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _artifact_sha256(_CHUNK_DOMAIN, self.metadata(False)),
        )
        self.validate_integrity()

    def metadata(self, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "token_start": self.token_start,
            "token_stop": self.token_stop,
            "token_count": self.token_count,
            "support_row_count": self.support_row_count,
            "token_cotangent_sha256": self.token_cotangent_sha256,
            "full_h4_cotangent_sha256": self.full_h4_cotangent_sha256,
            "support_h4_cotangent_sha256": self.support_h4_cotangent_sha256,
            "contracted_directional_token_teacher_kl_sha256": (
                self.contracted_directional_token_teacher_kl_sha256
            ),
            "full_h4_cotangent_shape": self.full_h4_cotangent_shape,
            "support_h4_cotangent_shape": self.support_h4_cotangent_shape,
            "device": self.device,
            "pullback_mechanism": self.pullback_mechanism,
            "canonical_one_hot_token_cotangents": (
                self.canonical_one_hot_token_cotangents
            ),
            "vmap_pullback_call_count": self.vmap_pullback_call_count,
            "token_cotangent_chunk_size": self.token_cotangent_chunk_size,
            "token_cotangent_dtype": self.token_cotangent_dtype,
            "full_h4_cotangent_dtype": self.full_h4_cotangent_dtype,
            "support_h4_cotangent_dtype": self.support_h4_cotangent_dtype,
            "contraction_dtype": self.contraction_dtype,
            "token_cotangent_nonzero_count": self.token_cotangent_nonzero_count,
            "token_cotangent_element_count": self.token_cotangent_element_count,
            "full_h4_cotangent_coordinate_count": (
                self.full_h4_cotangent_coordinate_count
            ),
            "support_h4_cotangent_coordinate_count": (
                self.support_h4_cotangent_coordinate_count
            ),
            "outside_support_h4_cotangent_coordinate_count": (
                self.outside_support_h4_cotangent_coordinate_count
            ),
            "full_h4_cotangent_retained": self.full_h4_cotangent_retained,
            "full_h4_cotangent_hashed": self.full_h4_cotangent_hashed,
            "full_h4_cotangent_serialized": self.full_h4_cotangent_serialized,
            "support_h4_cotangent_retained": self.support_h4_cotangent_retained,
            "support_h4_cotangent_hashed": self.support_h4_cotangent_hashed,
            "support_h4_cotangent_serialized": (
                self.support_h4_cotangent_serialized
            ),
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        basis = _canonical_token_cotangent_chunk(
            start=self.token_start,
            stop=self.token_stop,
            token_count=self.token_count,
            device=torch.device(self.device),
        )
        if _runtime_tensor_sha256(basis) != self.token_cotangent_sha256:
            raise RuntimeError("suffix VJP canonical cotangent receipt drifted")
        expected = _artifact_sha256(_CHUNK_DOMAIN, self.metadata(False))
        if expected != _require_sha256(
            self.artifact_sha256, label="suffix VJP chunk artifact"
        ):
            raise RuntimeError("suffix VJP chunk receipt drifted")


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4SuffixVJPReceipt:
    """Immutable receipt for one exact suffix primal and streamed pullback."""

    adapter_semantic_sha256: str
    adapter_model_sha256: str
    adapter_execution_sha256: str
    suffix_segment_fingerprints: tuple[str, ...]
    sequence_sha256: str
    teacher_logits_sha256: str
    supervised_indices_sha256: str
    support_indices_sha256: str
    path_h4_sha256: str
    direction_h4_sha256: str
    support_direction_h4_sha256: str
    full_h4_sha256: str
    cast_h4_sha256: str
    full_logits_sha256: str
    suffix_logits_sha256: str
    full_token_teacher_kl_sha256: str
    primal_token_teacher_kl_sha256: str
    directional_token_teacher_kl_sha256: str
    chunk_receipts: tuple[Gemma3L3L4H4SuffixVJPChunkReceipt, ...]
    h4_shape: tuple[int, int, int]
    support_direction_h4_shape: tuple[int, int, int]
    logits_shape: tuple[int, int, int]
    token_count: int
    support_row_count: int
    outside_support_row_count: int
    live_dtype: str
    teacher_dtype: str
    device: str
    outside_direction_nonzero_count: int
    outside_direction_max_abs: float
    suffix_segment_ids: tuple[str, ...] = GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS
    suffix_segment_ordinals: tuple[int, ...] = (
        GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS
    )
    suffix_segment_call_count: int = 13
    logit_projection_call_count: int = 1
    h4_dtype_cast_count: int = 1
    vjp_transform_count: int = 1
    vjp_pullback_chunk_call_count: int = field(init=False)
    vmap_pullback_call_count: int = field(init=False)
    token_cotangent_coverage_count: int = field(init=False)
    token_cotangent_nonzero_count: int = field(init=False)
    token_cotangent_element_count: int = field(init=False)
    full_h4_cotangent_coordinate_count: int = field(init=False)
    support_h4_cotangent_coordinate_count: int = field(init=False)
    outside_support_h4_cotangent_coordinate_count: int = field(init=False)
    direction_contraction_coordinate_product_count: int = field(init=False)
    direction_coordinate_validation_count: int = field(init=False)
    outside_support_direction_zero_validation_count: int = field(init=False)
    input_dtype: str = "torch.float64"
    objective_dtype: str = "torch.float64"
    ad_mechanism: str = "torch.func.vjp.reverse_mode"
    vjp_has_aux: bool = True
    canonical_one_hot_token_cotangents: bool = True
    pullback_vectorization: str = (
        "torch.vmap_over_canonical_token_cotangents"
    )
    contraction_scope: str = "authenticated_support_rows_only"
    token_cotangent_chunk_size: int = 8
    direction_is_exactly_zero_outside_support: bool = True
    full_suffix_h4_bitwise_equal: bool = True
    full_suffix_logits_bitwise_equal: bool = True
    full_suffix_token_teacher_kl_bitwise_equal: bool = True
    full_h4_cotangents_retained: bool = False
    full_h4_cotangents_hashed: bool = True
    full_h4_cotangents_serialized: bool = False
    support_h4_cotangents_retained: bool = False
    support_h4_cotangents_hashed: bool = True
    support_h4_cotangents_serialized: bool = False
    fits_searches_selects_or_routes_candidates: bool = False
    authorizes_serving_compression_or_model_mutation: bool = False
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        hash_names = (
            "adapter_semantic_sha256",
            "adapter_model_sha256",
            "adapter_execution_sha256",
            "sequence_sha256",
            "teacher_logits_sha256",
            "supervised_indices_sha256",
            "support_indices_sha256",
            "path_h4_sha256",
            "direction_h4_sha256",
            "support_direction_h4_sha256",
            "full_h4_sha256",
            "cast_h4_sha256",
            "full_logits_sha256",
            "suffix_logits_sha256",
            "full_token_teacher_kl_sha256",
            "primal_token_teacher_kl_sha256",
            "directional_token_teacher_kl_sha256",
        )
        for name in hash_names:
            _require_sha256(getattr(self, name), label=f"suffix VJP {name}")
        if len(self.suffix_segment_fingerprints) != 13:
            raise ValueError("suffix VJP must bind exactly 13 segment hashes")
        for value in self.suffix_segment_fingerprints:
            _require_sha256(value, label="suffix VJP segment fingerprint")
        h4_shape = _shape(self.h4_shape, rank=3, label="suffix VJP H4")
        support_shape = _shape(
            self.support_direction_h4_shape,
            rank=3,
            label="suffix VJP support direction",
        )
        logits_shape = _shape(
            self.logits_shape, rank=3, label="suffix VJP logits"
        )
        chunks = tuple(self.chunk_receipts)
        if (
            self.suffix_segment_ids != GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS
            or self.suffix_segment_ordinals
            != GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS
            or h4_shape[:2] != logits_shape[:2]
            or type(self.token_count) is not int
            or self.token_count <= 0
            or type(self.support_row_count) is not int
            or not 0 < self.support_row_count <= h4_shape[1]
            or type(self.outside_support_row_count) is not int
            or self.outside_support_row_count
            != h4_shape[1] - self.support_row_count
            or support_shape
            != (h4_shape[0], self.support_row_count, h4_shape[2])
            or not chunks
            or any(
                not isinstance(value, Gemma3L3L4H4SuffixVJPChunkReceipt)
                for value in chunks
            )
        ):
            raise ValueError("suffix VJP receipt geometry differs")
        expected_start = 0
        for index, chunk in enumerate(chunks):
            chunk.validate_integrity()
            if (
                chunk.token_start != expected_start
                or chunk.token_count != self.token_count
                or chunk.support_row_count != self.support_row_count
                or chunk.full_h4_cotangent_shape[1:] != h4_shape
                or chunk.support_h4_cotangent_shape[1:] != support_shape
                or chunk.device != self.device
                or (
                    index + 1 < len(chunks)
                    and chunk.token_stop - chunk.token_start != 8
                )
            ):
                raise ValueError("suffix VJP chunk coverage is noncanonical")
            expected_start = chunk.token_stop
        if expected_start != self.token_count:
            raise ValueError("suffix VJP chunks do not cover every token")
        if self.live_dtype != "torch.float32" or self.teacher_dtype not in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
        }:
            raise ValueError("suffix VJP dtype is unsupported")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("suffix VJP device must be nonempty")
        if (
            type(self.outside_direction_nonzero_count) is not int
            or self.outside_direction_nonzero_count != 0
            or type(self.outside_direction_max_abs) is not float
            or self.outside_direction_max_abs != 0.0
            or self.suffix_segment_call_count != 13
            or self.logit_projection_call_count != 1
            or self.h4_dtype_cast_count != 1
            or self.vjp_transform_count != 1
            or self.input_dtype != "torch.float64"
            or self.objective_dtype != "torch.float64"
            or self.ad_mechanism != "torch.func.vjp.reverse_mode"
            or self.vjp_has_aux is not True
            or self.canonical_one_hot_token_cotangents is not True
            or self.pullback_vectorization
            != "torch.vmap_over_canonical_token_cotangents"
            or self.contraction_scope != "authenticated_support_rows_only"
            or self.token_cotangent_chunk_size != 8
            or self.direction_is_exactly_zero_outside_support is not True
            or not all(
                (
                    self.full_suffix_h4_bitwise_equal,
                    self.full_suffix_logits_bitwise_equal,
                    self.full_suffix_token_teacher_kl_bitwise_equal,
                )
            )
            or self.full_h4_cotangents_retained is not False
            or self.full_h4_cotangents_hashed is not True
            or self.full_h4_cotangents_serialized is not False
            or self.support_h4_cotangents_retained is not False
            or self.support_h4_cotangents_hashed is not True
            or self.support_h4_cotangents_serialized is not False
            or self.fits_searches_selects_or_routes_candidates is not False
            or self.authorizes_serving_compression_or_model_mutation is not False
        ):
            raise ValueError("suffix VJP execution or safety policy differs")
        if (
            self.cast_h4_sha256 != self.full_h4_sha256
            or self.suffix_logits_sha256 != self.full_logits_sha256
            or self.primal_token_teacher_kl_sha256
            != self.full_token_teacher_kl_sha256
        ):
            raise ValueError("suffix VJP exact primal parity differs")
        object.__setattr__(self, "h4_shape", h4_shape)
        object.__setattr__(self, "support_direction_h4_shape", support_shape)
        object.__setattr__(self, "logits_shape", logits_shape)
        object.__setattr__(self, "chunk_receipts", chunks)
        object.__setattr__(self, "vjp_pullback_chunk_call_count", len(chunks))
        object.__setattr__(self, "vmap_pullback_call_count", len(chunks))
        object.__setattr__(self, "token_cotangent_coverage_count", self.token_count)
        object.__setattr__(
            self,
            "token_cotangent_nonzero_count",
            sum(value.token_cotangent_nonzero_count for value in chunks),
        )
        object.__setattr__(
            self,
            "token_cotangent_element_count",
            sum(value.token_cotangent_element_count for value in chunks),
        )
        for target, source in (
            (
                "full_h4_cotangent_coordinate_count",
                "full_h4_cotangent_coordinate_count",
            ),
            (
                "support_h4_cotangent_coordinate_count",
                "support_h4_cotangent_coordinate_count",
            ),
            (
                "outside_support_h4_cotangent_coordinate_count",
                "outside_support_h4_cotangent_coordinate_count",
            ),
        ):
            object.__setattr__(
                self, target, sum(getattr(value, source) for value in chunks)
            )
        object.__setattr__(
            self,
            "direction_contraction_coordinate_product_count",
            self.token_count * math.prod(support_shape),
        )
        object.__setattr__(
            self, "direction_coordinate_validation_count", math.prod(h4_shape)
        )
        object.__setattr__(
            self,
            "outside_support_direction_zero_validation_count",
            h4_shape[0] * self.outside_support_row_count * h4_shape[2],
        )
        if (
            self.token_cotangent_nonzero_count != self.token_count
            or self.token_cotangent_element_count != self.token_count**2
            or self.full_h4_cotangent_coordinate_count
            != self.token_count * math.prod(h4_shape)
            or self.support_h4_cotangent_coordinate_count
            != self.direction_contraction_coordinate_product_count
            or self.outside_support_h4_cotangent_coordinate_count
            != self.full_h4_cotangent_coordinate_count
            - self.support_h4_cotangent_coordinate_count
        ):
            raise ValueError("suffix VJP coverage accounting differs")
        object.__setattr__(
            self,
            "artifact_sha256",
            _artifact_sha256(_RECEIPT_DOMAIN, self.metadata(False)),
        )
        self.validate_integrity()

    def metadata(self, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {}
        for descriptor in dataclass_fields(self):
            if descriptor.name == "artifact_sha256":
                continue
            value = getattr(self, descriptor.name)
            if descriptor.name == "chunk_receipts":
                value = tuple(chunk.metadata() for chunk in value)
            result[descriptor.name] = value
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        for chunk in self.chunk_receipts:
            chunk.validate_integrity()
        expected = _artifact_sha256(_RECEIPT_DOMAIN, self.metadata(False))
        if expected != _require_sha256(
            self.artifact_sha256, label="suffix VJP receipt artifact"
        ):
            raise RuntimeError("suffix VJP receipt drifted")


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4SuffixVJP:
    """Retains only token primal/directional vectors and typed receipts."""

    primal_token_teacher_kl: Tensor = field(repr=False)
    directional_token_teacher_kl: Tensor = field(repr=False)
    receipt: Gemma3L3L4H4SuffixVJPReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, Gemma3L3L4H4SuffixVJPReceipt):
            raise TypeError("suffix VJP receipt has the wrong type")
        self.receipt.validate_integrity()
        retained: list[Tensor] = []
        for value, label in (
            (self.primal_token_teacher_kl, "suffix VJP primal token KL"),
            (
                self.directional_token_teacher_kl,
                "suffix VJP directional token KL",
            ),
        ):
            tensor = _finite_tensor(value, label=label)
            if (
                tensor.dtype != torch.float64
                or tensor.ndim != 1
                or tensor.shape != (self.receipt.token_count,)
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or str(tensor.device) != self.receipt.device
            ):
                raise ValueError(f"{label} geometry or dtype differs")
            retained.append(tensor.detach().clone().contiguous())
        object.__setattr__(self, "primal_token_teacher_kl", retained[0])
        object.__setattr__(
            self, "directional_token_teacher_kl", retained[1]
        )
        self.validate_integrity()

    @property
    def artifact_sha256(self) -> str:
        return self.receipt.artifact_sha256

    @property
    def token_count(self) -> int:
        return self.receipt.token_count

    def metadata(self) -> dict[str, object]:
        return self.receipt.metadata()

    def validate_integrity(self) -> None:
        self.receipt.validate_integrity()
        if (
            _runtime_tensor_sha256(self.primal_token_teacher_kl)
            != self.receipt.primal_token_teacher_kl_sha256
            or _runtime_tensor_sha256(self.directional_token_teacher_kl)
            != self.receipt.directional_token_teacher_kl_sha256
        ):
            raise RuntimeError("suffix VJP tensor payload drifted")
        for chunk in self.receipt.chunk_receipts:
            contracted = self.directional_token_teacher_kl[
                chunk.token_start : chunk.token_stop
            ].contiguous()
            if (
                _runtime_tensor_sha256(contracted)
                != chunk.contracted_directional_token_teacher_kl_sha256
            ):
                raise RuntimeError("suffix VJP directional chunk payload drifted")


class Gemma3L3L4H4SuffixVJPRuntime(Gemma3L3L4H4SuffixJVPRuntime):
    """Exact V11 suffix function differentiated once in reverse mode."""

    def execute(
        self,
        path_h4_f64: Tensor,
        direction_h4_f64: Tensor,
        *,
        support_indices: Tensor,
        full_h4: Tensor,
        full_logits: Tensor,
    ) -> Gemma3L3L4H4SuffixVJP:
        self.validate_integrity()
        path = _finite_tensor(path_h4_f64, label="suffix VJP path H4")
        direction = _finite_tensor(
            direction_h4_f64, label="suffix VJP direction H4"
        )
        reference_h4 = _finite_tensor(full_h4, label="suffix VJP full H4")
        reference_logits = _finite_tensor(
            full_logits, label="suffix VJP full logits"
        )
        if (
            path.dtype != torch.float64
            or direction.dtype != torch.float64
            or path.ndim != 3
            or path.shape[0] != 1
            or direction.shape != path.shape
            or path.device != direction.device
            or path.requires_grad
            or direction.requires_grad
            or not path.is_contiguous()
            or not direction.is_contiguous()
        ):
            raise ValueError(
                "suffix VJP path and direction must be singleton-batch "
                "contiguous float64 peers"
            )
        if (
            reference_h4.ndim != 3
            or reference_h4.shape != path.shape
            or reference_h4.dtype != torch.float32
            or reference_h4.device != path.device
            or reference_h4.device != self._sequence.device
            or reference_h4.requires_grad
            or not reference_h4.is_contiguous()
        ):
            raise ValueError(
                "suffix VJP full H4 must be the contiguous live float32 H4"
            )
        expected_h4_shape = (
            self._sequence.batch_size,
            self._sequence.query_length,
            self._suffix_segments[0].input_width,
        )
        if tuple(reference_h4.shape) != expected_h4_shape:
            raise ValueError("suffix VJP full H4 differs from layer.5 input")
        if (
            reference_logits.ndim != 3
            or reference_logits.shape != self._teacher_logits.shape
            or reference_logits.dtype != torch.float32
            or reference_logits.device != reference_h4.device
            or reference_logits.requires_grad
            or not reference_logits.is_contiguous()
        ):
            raise ValueError(
                "suffix VJP full logits must be contiguous live float32 logits"
            )

        support = _support_indices(
            support_indices, sequence_length=reference_h4.shape[1]
        )
        support_on_device = support.to(device=path.device)
        support_mask = torch.zeros(reference_h4.shape[1], dtype=torch.bool)
        support_mask.index_fill_(0, support, True)
        outside_mask = ~support_mask
        outside_direction = direction.detach().to(device="cpu")[:, outside_mask]
        outside_nonzero_count = int(torch.count_nonzero(outside_direction))
        outside_max_abs = (
            0.0
            if outside_direction.numel() == 0
            else float(outside_direction.abs().max())
        )
        if outside_nonzero_count != 0 or outside_max_abs != 0.0:
            raise ValueError("suffix VJP direction is nonzero outside support")
        direction_support = (
            direction.index_select(1, support_on_device).detach().contiguous()
        )
        if (
            direction_support.dtype != torch.float64
            or direction_support.shape
            != (path.shape[0], support.numel(), path.shape[2])
        ):
            raise RuntimeError("suffix VJP support direction extraction differs")

        path_sha256 = _runtime_tensor_sha256(path)
        direction_sha256 = _runtime_tensor_sha256(direction)
        support_direction_sha256 = _runtime_tensor_sha256(direction_support)
        full_h4_sha256 = _runtime_tensor_sha256(reference_h4)
        segment_call_count = 0
        projection_call_count = 0

        def suffix_token_teacher_kl(
            candidate_path_f64: Tensor,
        ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
            nonlocal segment_call_count, projection_call_count
            hidden_states = candidate_path_f64.to(dtype=reference_h4.dtype)
            cast_h4_aux = hidden_states
            finite_flags: list[Tensor] = []
            for segment in self._suffix_segments:
                run = self._adapter.run_segment(
                    segment, hidden_states, self._sequence
                )
                segment_call_count += 1
                if run.sequence is not self._sequence:
                    raise RuntimeError("suffix VJP segment changed sequence ownership")
                hidden_states = run.hidden_states
                if (
                    hidden_states.shape
                    != (
                        self._sequence.batch_size,
                        self._sequence.query_length,
                        segment.output_width,
                    )
                    or hidden_states.dtype != reference_h4.dtype
                    or hidden_states.device != reference_h4.device
                ):
                    raise ValueError("suffix VJP segment output contract differs")
                finite_flags.append(torch.isfinite(hidden_states).all())
            logits = self._adapter.project_logits(hidden_states, self._sequence)
            projection_call_count += 1
            if (
                logits.shape != reference_logits.shape
                or logits.dtype != reference_logits.dtype
                or logits.device != reference_logits.device
            ):
                raise ValueError("suffix VJP logits contract differs")
            token_kl = self._token_teacher_kl(logits, validate_finite=False)
            return token_kl, (
                cast_h4_aux,
                logits,
                torch.stack(finite_flags),
            )

        primal, pullback, aux = torch.func.vjp(
            suffix_token_teacher_kl, path, has_aux=True
        )
        cast_h4, suffix_logits, finite_flags = (
            value.detach().contiguous() for value in aux
        )
        if segment_call_count != 13 or projection_call_count != 1:
            raise RuntimeError("suffix VJP suffix geometry changed")
        if (
            finite_flags.dtype != torch.bool
            or finite_flags.shape != (13,)
            or not bool(finite_flags.all())
        ):
            raise ValueError("one or more suffix VJP segment outputs are nonfinite")
        if not _bitwise_equal(cast_h4, reference_h4):
            raise ValueError("suffix VJP float64 path does not cast to full H4")
        cast_h4_sha256 = _runtime_tensor_sha256(cast_h4)
        if cast_h4_sha256 != full_h4_sha256:
            raise RuntimeError("suffix VJP cast and full H4 hashes differ")
        if not bool(torch.isfinite(suffix_logits).all()):
            raise ValueError("suffix VJP logits are nonfinite")
        if not _bitwise_equal(suffix_logits, reference_logits):
            raise RuntimeError(
                "suffix VJP logits are not bitwise equal to full logits"
            )
        full_logits_sha256 = _runtime_tensor_sha256(reference_logits)
        suffix_logits_sha256 = _runtime_tensor_sha256(suffix_logits)
        if suffix_logits_sha256 != full_logits_sha256:
            raise RuntimeError("suffix VJP suffix and full logits hashes differ")

        full_token_kl = (
            self._token_teacher_kl(reference_logits).detach().contiguous()
        )
        primal = primal.detach().contiguous()
        if (
            primal.dtype != torch.float64
            or primal.shape != (self.token_count,)
            or not bool(torch.isfinite(primal).all())
            or not _bitwise_equal(primal, full_token_kl)
        ):
            raise RuntimeError("suffix VJP token KL primal differs from full model")
        full_token_kl_sha256 = _runtime_tensor_sha256(full_token_kl)
        primal_sha256 = _runtime_tensor_sha256(primal)
        if primal_sha256 != full_token_kl_sha256:
            raise RuntimeError("suffix VJP token KL primal hash differs")

        directional = torch.empty_like(primal)
        chunks: list[Gemma3L3L4H4SuffixVJPChunkReceipt] = []
        for start in range(0, self.token_count, 8):
            stop = min(start + 8, self.token_count)
            token_cotangent = _canonical_token_cotangent_chunk(
                start=start,
                stop=stop,
                token_count=self.token_count,
                device=path.device,
            )
            (raw_full_h4_cotangent,) = torch.func.vmap(pullback)(
                token_cotangent
            )
            full_h4_cotangent = raw_full_h4_cotangent.detach().contiguous()
            del raw_full_h4_cotangent
            if (
                full_h4_cotangent.dtype != torch.float64
                or full_h4_cotangent.shape
                != (stop - start, *tuple(path.shape))
                or not bool(torch.isfinite(full_h4_cotangent).all())
            ):
                raise ValueError(
                    "suffix VJP pullback returned invalid H4 cotangents"
                )
            full_h4_cotangent_sha256 = _runtime_tensor_sha256(
                full_h4_cotangent
            )
            support_h4_cotangent = (
                full_h4_cotangent.index_select(2, support_on_device)
                .detach()
                .contiguous()
            )
            if (
                support_h4_cotangent.dtype != torch.float64
                or support_h4_cotangent.shape
                != (
                    stop - start,
                    path.shape[0],
                    support.numel(),
                    path.shape[2],
                )
                or not bool(torch.isfinite(support_h4_cotangent).all())
            ):
                raise ValueError(
                    "suffix VJP support cotangent extraction differs"
                )
            support_h4_cotangent_sha256 = _runtime_tensor_sha256(
                support_h4_cotangent
            )
            directional_chunk = torch.sum(
                support_h4_cotangent * direction_support.unsqueeze(0),
                dim=(1, 2, 3),
                dtype=torch.float64,
            ).detach().contiguous()
            if not bool(torch.isfinite(directional_chunk).all()):
                raise ValueError("suffix VJP directional contraction is nonfinite")
            directional_chunk_sha256 = _runtime_tensor_sha256(
                directional_chunk
            )
            directional[start:stop] = directional_chunk
            chunks.append(
                Gemma3L3L4H4SuffixVJPChunkReceipt(
                    token_start=start,
                    token_stop=stop,
                    token_count=self.token_count,
                    support_row_count=int(support.numel()),
                    token_cotangent_sha256=_runtime_tensor_sha256(
                        token_cotangent
                    ),
                    full_h4_cotangent_sha256=full_h4_cotangent_sha256,
                    support_h4_cotangent_sha256=(
                        support_h4_cotangent_sha256
                    ),
                    contracted_directional_token_teacher_kl_sha256=(
                        directional_chunk_sha256
                    ),
                    full_h4_cotangent_shape=tuple(
                        int(size) for size in full_h4_cotangent.shape
                    ),
                    support_h4_cotangent_shape=tuple(
                        int(size) for size in support_h4_cotangent.shape
                    ),
                    device=str(path.device),
                )
            )
            del (
                full_h4_cotangent,
                support_h4_cotangent,
                token_cotangent,
                directional_chunk,
            )
        del pullback
        directional = directional.detach().contiguous()
        if (
            directional.dtype != torch.float64
            or directional.shape != (self.token_count,)
            or not bool(torch.isfinite(directional).all())
        ):
            raise ValueError(
                "suffix VJP returned an invalid directional token vector"
            )

        self.validate_integrity()
        receipt = Gemma3L3L4H4SuffixVJPReceipt(
            adapter_semantic_sha256=self._adapter_semantic_sha256,
            adapter_model_sha256=self._adapter_model_sha256,
            adapter_execution_sha256=self._adapter_execution_sha256,
            suffix_segment_fingerprints=self._suffix_segment_fingerprints,
            sequence_sha256=self._sequence_sha256,
            teacher_logits_sha256=self._teacher_logits_sha256,
            supervised_indices_sha256=self._supervised_indices_sha256,
            support_indices_sha256=_runtime_tensor_sha256(support),
            path_h4_sha256=path_sha256,
            direction_h4_sha256=direction_sha256,
            support_direction_h4_sha256=support_direction_sha256,
            full_h4_sha256=full_h4_sha256,
            cast_h4_sha256=cast_h4_sha256,
            full_logits_sha256=full_logits_sha256,
            suffix_logits_sha256=suffix_logits_sha256,
            full_token_teacher_kl_sha256=full_token_kl_sha256,
            primal_token_teacher_kl_sha256=primal_sha256,
            directional_token_teacher_kl_sha256=(
                _runtime_tensor_sha256(directional)
            ),
            chunk_receipts=tuple(chunks),
            h4_shape=tuple(int(size) for size in reference_h4.shape),
            support_direction_h4_shape=tuple(
                int(size) for size in direction_support.shape
            ),
            logits_shape=tuple(int(size) for size in reference_logits.shape),
            token_count=self.token_count,
            support_row_count=int(support.numel()),
            outside_support_row_count=int(
                reference_h4.shape[1] - support.numel()
            ),
            live_dtype=str(reference_h4.dtype),
            teacher_dtype=str(self._teacher_logits.dtype),
            device=str(reference_h4.device),
            outside_direction_nonzero_count=outside_nonzero_count,
            outside_direction_max_abs=float(outside_max_abs),
            suffix_segment_call_count=segment_call_count,
            logit_projection_call_count=projection_call_count,
        )
        result = Gemma3L3L4H4SuffixVJP(
            primal_token_teacher_kl=primal,
            directional_token_teacher_kl=directional,
            receipt=receipt,
        )
        result.validate_integrity()
        return result


def gemma3_l3_l4_h4_suffix_vjp_resource_accounting(
    receipts: Iterable[Gemma3L3L4H4SuffixVJPReceipt],
) -> dict[str, int | bool]:
    values = tuple(receipts)
    if not values or any(
        not isinstance(value, Gemma3L3L4H4SuffixVJPReceipt)
        for value in values
    ):
        raise TypeError("suffix VJP resource accounting requires typed receipts")
    for value in values:
        value.validate_integrity()
    chunk_count = sum(
        value.vjp_pullback_chunk_call_count for value in values
    )
    return {
        "suffix_vjp_node_count": len(values),
        "suffix_vjp_primal_vector_count": len(values),
        "suffix_vjp_token_directional_derivative_count": sum(
            value.token_count for value in values
        ),
        "suffix_segment_call_count": sum(
            value.suffix_segment_call_count for value in values
        ),
        "logit_projection_call_count": sum(
            value.logit_projection_call_count for value in values
        ),
        "h4_dtype_cast_count": sum(
            value.h4_dtype_cast_count for value in values
        ),
        "vjp_transform_count": sum(
            value.vjp_transform_count for value in values
        ),
        "vjp_pullback_chunk_call_count": chunk_count,
        "vmap_pullback_call_count": sum(
            value.vmap_pullback_call_count for value in values
        ),
        "canonical_token_cotangent_coverage_count": sum(
            value.token_cotangent_coverage_count for value in values
        ),
        "canonical_token_cotangent_nonzero_count": sum(
            value.token_cotangent_nonzero_count for value in values
        ),
        "canonical_token_cotangent_element_observation_count": sum(
            value.token_cotangent_element_count for value in values
        ),
        "full_h4_row_observation_count": sum(
            value.h4_shape[0] * value.h4_shape[1] for value in values
        ),
        "support_h4_row_observation_count": sum(
            value.h4_shape[0] * value.support_row_count for value in values
        ),
        "outside_support_h4_row_observation_count": sum(
            value.h4_shape[0] * value.outside_support_row_count
            for value in values
        ),
        "direction_coordinate_validation_count": sum(
            value.direction_coordinate_validation_count for value in values
        ),
        "outside_support_direction_zero_validation_count": sum(
            value.outside_support_direction_zero_validation_count
            for value in values
        ),
        "full_h4_cotangent_coordinate_observation_count": sum(
            value.full_h4_cotangent_coordinate_count for value in values
        ),
        "support_h4_cotangent_coordinate_observation_count": sum(
            value.support_h4_cotangent_coordinate_count for value in values
        ),
        "outside_support_h4_cotangent_coordinate_observation_count": sum(
            value.outside_support_h4_cotangent_coordinate_count
            for value in values
        ),
        "direction_contraction_coordinate_product_count": sum(
            value.direction_contraction_coordinate_product_count
            for value in values
        ),
        "full_h4_cotangent_sha256_count": chunk_count,
        "support_h4_cotangent_sha256_count": chunk_count,
        "contracted_directional_chunk_sha256_count": chunk_count,
        "retained_full_h4_cotangent_count": 0,
        "serialized_full_h4_cotangent_count": 0,
        "resource_counts_are_not_FLOPs_or_total_model_compute": True,
    }


def require_gemma3_l3_l4_h4_suffix_vjp_complete_panel(
    resources: Mapping[str, object],
) -> None:
    if not isinstance(resources, Mapping):
        raise TypeError("suffix VJP resources must be a mapping")
    observed = dict(resources)
    if observed != _COMPLETE:
        raise RuntimeError(
            "suffix VJP complete-panel resource accounting differs: "
            f"observed={observed!r}"
        )
