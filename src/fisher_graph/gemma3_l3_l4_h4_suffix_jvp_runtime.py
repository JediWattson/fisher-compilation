"""Authenticated H4-suffix JVP execution for the 18-layer Gemma 3 stack.

The runtime accepts a float64 H4 path point and direction, casts the path
exactly once to the live ``layer.4.output`` dtype, replays native segments
5..17, and projects logits through the native final norm and LM head.  Token
``KL(teacher || candidate)`` arithmetic is performed in float64.  Every call
must also supply the corresponding full-model H4 and logits; suffix replay is
accepted only when both the cast H4 and the replayed logits are bitwise equal
to those full-model references.

Only compact hashes, geometry, counters, and scalar summaries are serialized.
The returned primal and directional token vectors remain tensors and are
tamper-detectable through their immutable receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters.base import SequenceContext
from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _canonical_json_bytes,
    _require_sha256,
    _runtime_tensor_sha256,
)


__all__ = [
    "GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS",
    "GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS",
    "Gemma3L3L4H4DiscreteCastIntervalStats",
    "Gemma3L3L4H4SuffixJVP",
    "Gemma3L3L4H4SuffixJVPReceipt",
    "Gemma3L3L4H4SuffixJVPRuntime",
    "gemma3_l3_l4_h4_discrete_cast_interval_stats",
]


GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS = tuple(range(5, 18))
GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS = tuple(
    f"layer.{ordinal}" for ordinal in GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS
)

_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l3-l4-h4-suffix-jvp-receipt:v11\0"
_INTERVAL_DOMAIN = b"fisher-graph:gemma3-l3-l4-h4-cast-interval:v11\0"
_SUPPORTED_LIVE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _artifact_sha256(domain: bytes, metadata: Mapping[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(metadata)).hexdigest()


def _shape(value: object, *, rank: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != rank
        or any(type(size) is not int or size <= 0 for size in value)
    ):
        raise ValueError(f"{label} must be a positive rank-{rank} shape")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _materialized_tensor(value: object, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError(f"{label} must use materialized strided storage")
    return value


def _finite_tensor(value: object, *, label: str) -> Tensor:
    tensor = _materialized_tensor(value, label=label)
    if not tensor.is_floating_point():
        raise ValueError(f"{label} must be floating-point")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    if (
        left.dtype != right.dtype
        or left.device != right.device
        or left.shape != right.shape
        or left.layout != torch.strided
        or right.layout != torch.strided
    ):
        return False
    left_bytes = left.detach().to(device="cpu").contiguous().view(torch.uint8)
    right_bytes = right.detach().to(device="cpu").contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _sequence_payload_value(value: object, *, label: str) -> object:
    if isinstance(value, Tensor):
        return {"tensor_sha256": _runtime_tensor_sha256(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return {"float_hex": value.hex()}
    if isinstance(value, Mapping):
        materialized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} mapping keys must be nonempty strings")
            materialized[key] = _sequence_payload_value(
                item,
                label=f"{label}.{key}",
            )
        return {key: materialized[key] for key in sorted(materialized)}
    if isinstance(value, (tuple, list)):
        return [
            _sequence_payload_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{label} contains unsupported execution state {type(value).__name__}"
    )


def _sequence_sha256(sequence: SequenceContext) -> str:
    origin = sequence.input_origin
    payload: dict[str, object] = {
        "query_valid_mask": _runtime_tensor_sha256(sequence.query_valid_mask),
        "key_valid_mask": _runtime_tensor_sha256(sequence.key_valid_mask),
        "logical_positions": _runtime_tensor_sha256(sequence.logical_positions),
        "key_logical_positions": _runtime_tensor_sha256(
            sequence.key_logical_positions
        ),
        "cache_positions": (
            None
            if sequence.cache_positions is None
            else _runtime_tensor_sha256(sequence.cache_positions)
        ),
        "phase": sequence.phase,
        "input_origin": {
            "attention_mask_supplied": origin.attention_mask_supplied,
            "position_ids_supplied": origin.position_ids_supplied,
            "cache_positions_supplied": origin.cache_positions_supplied,
        },
        "cache_state_is_none": sequence.cache_state is None,
        "adapter_payload": _sequence_payload_value(
            sequence.adapter_payload,
            label="sequence adapter_payload",
        ),
    }
    return hashlib.sha256(
        _RECEIPT_DOMAIN + b"sequence\0" + _canonical_json_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4SuffixJVPReceipt:
    """Immutable authentication receipt for one suffix JVP call."""

    adapter_semantic_sha256: str
    adapter_model_sha256: str
    adapter_execution_sha256: str
    suffix_segment_fingerprints: tuple[str, ...]
    sequence_sha256: str
    teacher_logits_sha256: str
    supervised_indices_sha256: str
    path_h4_sha256: str
    direction_h4_sha256: str
    full_h4_sha256: str
    cast_h4_sha256: str
    full_logits_sha256: str
    suffix_logits_sha256: str
    full_token_teacher_kl_sha256: str
    primal_token_teacher_kl_sha256: str
    directional_token_teacher_kl_sha256: str
    h4_shape: tuple[int, int, int]
    logits_shape: tuple[int, int, int]
    token_count: int
    live_dtype: str
    teacher_dtype: str
    device: str
    suffix_segment_ids: tuple[str, ...] = GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS
    suffix_segment_ordinals: tuple[int, ...] = (
        GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS
    )
    suffix_segment_call_count: int = 13
    logit_projection_call_count: int = 1
    h4_dtype_cast_count: int = 1
    input_dtype: str = "torch.float64"
    objective_dtype: str = "torch.float64"
    ad_mechanism: str = "torch.func.jvp.forward_mode"
    jvp_strict: bool = True
    jvp_has_aux: bool = True
    full_suffix_h4_bitwise_equal: bool = True
    full_suffix_logits_bitwise_equal: bool = True
    full_suffix_token_teacher_kl_bitwise_equal: bool = True
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for value, label in (
            (self.adapter_semantic_sha256, "adapter semantic"),
            (self.adapter_model_sha256, "adapter model"),
            (self.adapter_execution_sha256, "adapter execution"),
            (self.sequence_sha256, "sequence"),
            (self.teacher_logits_sha256, "teacher logits"),
            (self.supervised_indices_sha256, "supervised indices"),
            (self.path_h4_sha256, "path H4"),
            (self.direction_h4_sha256, "direction H4"),
            (self.full_h4_sha256, "full H4"),
            (self.cast_h4_sha256, "cast H4"),
            (self.full_logits_sha256, "full logits"),
            (self.suffix_logits_sha256, "suffix logits"),
            (self.full_token_teacher_kl_sha256, "full token teacher KL"),
            (self.primal_token_teacher_kl_sha256, "primal token teacher KL"),
            (
                self.directional_token_teacher_kl_sha256,
                "directional token teacher KL",
            ),
        ):
            _require_sha256(value, label=label)
        if len(self.suffix_segment_fingerprints) != 13:
            raise ValueError("suffix receipt must bind exactly 13 segment hashes")
        for fingerprint in self.suffix_segment_fingerprints:
            _require_sha256(fingerprint, label="suffix segment fingerprint")
        if self.suffix_segment_ids != GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS:
            raise ValueError("suffix segment ids must be exactly layer.5..layer.17")
        if self.suffix_segment_ordinals != (
            GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS
        ):
            raise ValueError("suffix segment ordinals must be exactly 5..17")
        h4_shape = _shape(self.h4_shape, rank=3, label="H4")
        logits_shape = _shape(self.logits_shape, rank=3, label="logits")
        if h4_shape[:2] != logits_shape[:2]:
            raise ValueError("H4 and logits grids differ")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("suffix JVP token count must be positive")
        if self.live_dtype not in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
        }:
            raise ValueError("suffix JVP live dtype is unsupported")
        if self.teacher_dtype not in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
        }:
            raise ValueError("suffix JVP teacher dtype is unsupported")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("suffix JVP device must be nonempty")
        if (
            self.suffix_segment_call_count != 13
            or self.logit_projection_call_count != 1
            or self.h4_dtype_cast_count != 1
            or self.input_dtype != "torch.float64"
            or self.objective_dtype != "torch.float64"
            or self.ad_mechanism != "torch.func.jvp.forward_mode"
            or self.jvp_strict is not True
            or self.jvp_has_aux is not True
        ):
            raise ValueError("suffix JVP execution counters or dtypes differ")
        if not all(
            (
                self.full_suffix_h4_bitwise_equal,
                self.full_suffix_logits_bitwise_equal,
                self.full_suffix_token_teacher_kl_bitwise_equal,
            )
        ):
            raise ValueError("suffix JVP exact primal parity was not established")
        if self.cast_h4_sha256 != self.full_h4_sha256:
            raise ValueError("cast H4 and full H4 hashes differ")
        if self.suffix_logits_sha256 != self.full_logits_sha256:
            raise ValueError("suffix and full logits hashes differ")
        if (
            self.primal_token_teacher_kl_sha256
            != self.full_token_teacher_kl_sha256
        ):
            raise ValueError("suffix and full token teacher-KL hashes differ")
        object.__setattr__(
            self,
            "artifact_sha256",
            _artifact_sha256(
                _RECEIPT_DOMAIN,
                self.metadata(include_artifact=False),
            ),
        )

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "adapter_semantic_sha256": self.adapter_semantic_sha256,
            "adapter_model_sha256": self.adapter_model_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "suffix_segment_fingerprints": self.suffix_segment_fingerprints,
            "sequence_sha256": self.sequence_sha256,
            "teacher_logits_sha256": self.teacher_logits_sha256,
            "supervised_indices_sha256": self.supervised_indices_sha256,
            "path_h4_sha256": self.path_h4_sha256,
            "direction_h4_sha256": self.direction_h4_sha256,
            "full_h4_sha256": self.full_h4_sha256,
            "cast_h4_sha256": self.cast_h4_sha256,
            "full_logits_sha256": self.full_logits_sha256,
            "suffix_logits_sha256": self.suffix_logits_sha256,
            "full_token_teacher_kl_sha256": (
                self.full_token_teacher_kl_sha256
            ),
            "primal_token_teacher_kl_sha256": (
                self.primal_token_teacher_kl_sha256
            ),
            "directional_token_teacher_kl_sha256": (
                self.directional_token_teacher_kl_sha256
            ),
            "h4_shape": self.h4_shape,
            "logits_shape": self.logits_shape,
            "token_count": self.token_count,
            "live_dtype": self.live_dtype,
            "teacher_dtype": self.teacher_dtype,
            "device": self.device,
            "suffix_segment_ids": self.suffix_segment_ids,
            "suffix_segment_ordinals": self.suffix_segment_ordinals,
            "suffix_segment_call_count": self.suffix_segment_call_count,
            "logit_projection_call_count": self.logit_projection_call_count,
            "h4_dtype_cast_count": self.h4_dtype_cast_count,
            "input_dtype": self.input_dtype,
            "objective_dtype": self.objective_dtype,
            "ad_mechanism": self.ad_mechanism,
            "jvp_strict": self.jvp_strict,
            "jvp_has_aux": self.jvp_has_aux,
            "full_suffix_h4_bitwise_equal": (
                self.full_suffix_h4_bitwise_equal
            ),
            "full_suffix_logits_bitwise_equal": (
                self.full_suffix_logits_bitwise_equal
            ),
            "full_suffix_token_teacher_kl_bitwise_equal": (
                self.full_suffix_token_teacher_kl_bitwise_equal
            ),
            "serialized_primal_or_directional_tensors": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        expected = _artifact_sha256(
            _RECEIPT_DOMAIN,
            self.metadata(include_artifact=False),
        )
        if expected != _require_sha256(
            self.artifact_sha256,
            label="suffix JVP receipt artifact",
        ):
            raise RuntimeError("suffix JVP receipt drifted")


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4SuffixJVP:
    """Float64 primal and directional token KL vectors with a receipt."""

    primal_token_teacher_kl: Tensor = field(repr=False)
    directional_token_teacher_kl: Tensor = field(repr=False)
    receipt: Gemma3L3L4H4SuffixJVPReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, Gemma3L3L4H4SuffixJVPReceipt):
            raise TypeError("suffix JVP receipt has the wrong type")
        self.receipt.validate_integrity()
        tensors: list[Tensor] = []
        for value, label in (
            (self.primal_token_teacher_kl, "primal token teacher KL"),
            (
                self.directional_token_teacher_kl,
                "directional token teacher KL",
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
                raise ValueError(f"{label} geometry or dtype is invalid")
            tensors.append(tensor.detach().clone().contiguous())
        object.__setattr__(self, "primal_token_teacher_kl", tensors[0])
        object.__setattr__(self, "directional_token_teacher_kl", tensors[1])
        self.validate_integrity()

    @property
    def primal_token_teacher_kl_sha256(self) -> str:
        return self.receipt.primal_token_teacher_kl_sha256

    @property
    def directional_token_teacher_kl_sha256(self) -> str:
        return self.receipt.directional_token_teacher_kl_sha256

    @property
    def path_h4_sha256(self) -> str:
        return self.receipt.path_h4_sha256

    @property
    def full_h4_sha256(self) -> str:
        return self.receipt.full_h4_sha256

    @property
    def full_logits_sha256(self) -> str:
        return self.receipt.full_logits_sha256

    @property
    def suffix_logits_sha256(self) -> str:
        return self.receipt.suffix_logits_sha256

    @property
    def full_token_teacher_kl_sha256(self) -> str:
        return self.receipt.full_token_teacher_kl_sha256

    @property
    def suffix_segment_call_count(self) -> int:
        return self.receipt.suffix_segment_call_count

    @property
    def logit_projection_call_count(self) -> int:
        return self.receipt.logit_projection_call_count

    @property
    def h4_dtype_cast_count(self) -> int:
        return self.receipt.h4_dtype_cast_count

    @property
    def artifact_sha256(self) -> str:
        return self.receipt.artifact_sha256

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
            raise RuntimeError("suffix JVP tensor payload drifted")


class Gemma3L3L4H4SuffixJVPRuntime:
    """Reusable authenticated native suffix from full ``layer.4.output``."""

    @staticmethod
    def _module_state_guard(
        adapter: Gemma3CausalLMAdapter,
    ) -> tuple[tuple[object, ...], ...]:
        """Cheaply detect ordinary tensor replacement or in-place mutation."""

        values: list[tuple[object, ...]] = []
        named_tensors = (
            ("parameter", name, tensor)
            for name, tensor in adapter.module.named_parameters()
        )
        named_buffers = (
            ("buffer", name, tensor)
            for name, tensor in adapter.module.named_buffers()
        )
        for kind, name, tensor in (*named_tensors, *named_buffers):
            values.append(
                (
                    kind,
                    name,
                    id(tensor),
                    tensor.data_ptr(),
                    tensor._version,
                    str(tensor.device),
                    str(tensor.dtype),
                    tuple(int(size) for size in tensor.shape),
                )
            )
        return tuple(values)

    def __init__(
        self,
        adapter: Gemma3CausalLMAdapter,
        sequence: SequenceContext,
        *,
        teacher_logits: Tensor,
        supervised_indices: Tensor,
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if not isinstance(sequence, SequenceContext):
            raise TypeError("sequence must be a SequenceContext")
        if sequence.phase != "prefill" or sequence.cache_state is not None:
            raise ValueError("suffix JVP requires cache-free prefill sequence state")
        if any(module.training for module in adapter.module.modules()):
            raise ValueError(
                "suffix JVP requires the complete Gemma module hierarchy "
                "in eval mode"
            )
        attention_implementation = getattr(
            getattr(adapter.module, "config", None),
            "_attn_implementation",
            None,
        )
        if attention_implementation != "eager":
            raise ValueError("suffix JVP requires eager attention")

        segments = tuple(adapter.segments)
        if len(segments) != 18:
            raise ValueError("suffix JVP requires the canonical 18-layer stack")
        suffix_segments = tuple(segments[5:18])
        ids = tuple(segment.id for segment in suffix_segments)
        ordinals = tuple(segment.ordinal for segment in suffix_segments)
        if ids != GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS or ordinals != (
            GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_ORDINALS
        ):
            raise ValueError("adapter suffix must be exactly segments 5..17")
        for expected_ordinal, segment in zip(ordinals, suffix_segments):
            expected_id = f"layer.{expected_ordinal}"
            if (
                segment.layer_ids != (expected_id,)
                or segment.input_site != f"{expected_id}.input"
                or segment.output_site != f"{expected_id}.output"
            ):
                raise ValueError("suffix segment semantics are noncanonical")

        teacher = _finite_tensor(teacher_logits, label="teacher logits")
        if (
            teacher.ndim != 3
            or teacher.shape[0] != sequence.batch_size
            or teacher.shape[1] != sequence.query_length
            or teacher.shape[2] <= 0
            or teacher.device != sequence.device
        ):
            raise ValueError("teacher logits do not match the sequence grid")
        indices = _materialized_tensor(
            supervised_indices,
            label="supervised indices",
        )
        if (
            indices.dtype != torch.int64
            or indices.device.type != "cpu"
            or indices.ndim != 2
            or indices.shape[1] != 2
            or indices.shape[0] <= 0
            or indices.requires_grad
            or not indices.is_contiguous()
        ):
            raise ValueError(
                "supervised indices must be nonempty contiguous CPU int64 [N, 2]"
            )
        batch_count, sequence_length = teacher.shape[:2]
        flattened = indices[:, 0] * sequence_length + indices[:, 1]
        if (
            bool((indices[:, 0] < 0).any())
            or bool((indices[:, 0] >= batch_count).any())
            or bool((indices[:, 1] < 0).any())
            or bool((indices[:, 1] >= sequence_length).any())
            or (
                indices.shape[0] > 1
                and not bool((flattened[1:] > flattened[:-1]).all())
            )
        ):
            raise ValueError("supervised indices escape or reorder the grid")
        indices_on_sequence = indices.to(device=sequence.device)
        if not bool(
            sequence.query_valid_mask[
                indices_on_sequence[:, 0],
                indices_on_sequence[:, 1],
            ].all()
        ):
            raise ValueError("supervised indices escape the valid sequence grid")

        self._adapter = adapter
        self._sequence = sequence
        self._suffix_segments = suffix_segments
        self._teacher_logits = teacher.detach().clone().contiguous()
        self._supervised_indices = indices.detach().clone().contiguous()
        self._adapter_semantic_sha256 = adapter.semantic_fingerprint()
        self._adapter_model_sha256 = adapter.model_fingerprint()
        self._adapter_execution_sha256 = adapter.execution_fingerprint()
        self._suffix_segment_fingerprints = tuple(
            adapter.segment_fingerprint(segment) for segment in suffix_segments
        )
        self._state_guard = self._module_state_guard(adapter)
        self._sequence_sha256 = _sequence_sha256(sequence)
        self._teacher_logits_sha256 = _runtime_tensor_sha256(
            self._teacher_logits
        )
        self._supervised_indices_sha256 = _runtime_tensor_sha256(
            self._supervised_indices
        )

    @property
    def token_count(self) -> int:
        return int(self._supervised_indices.shape[0])

    @property
    def suffix_segment_ids(self) -> tuple[str, ...]:
        return GEMMA3_L3_L4_H4_SUFFIX_SEGMENT_IDS

    def validate_integrity(self, *, deep: bool = False) -> None:
        if self._adapter.semantic_fingerprint() != self._adapter_semantic_sha256:
            raise RuntimeError("suffix JVP adapter semantics drifted")
        if self._adapter.execution_fingerprint() != self._adapter_execution_sha256:
            raise RuntimeError("suffix JVP adapter execution options drifted")
        if self._module_state_guard(self._adapter) != self._state_guard:
            raise RuntimeError("suffix JVP adapter model state drifted")
        if _sequence_sha256(self._sequence) != self._sequence_sha256:
            raise RuntimeError("suffix JVP sequence state drifted")
        if (
            _runtime_tensor_sha256(self._teacher_logits)
            != self._teacher_logits_sha256
            or _runtime_tensor_sha256(self._supervised_indices)
            != self._supervised_indices_sha256
        ):
            raise RuntimeError("suffix JVP teacher or supervised grid drifted")
        if deep:
            if self._adapter.model_fingerprint() != self._adapter_model_sha256:
                raise RuntimeError("suffix JVP adapter model state drifted")
            current = tuple(
                self._adapter.segment_fingerprint(segment)
                for segment in self._suffix_segments
            )
            if current != self._suffix_segment_fingerprints:
                raise RuntimeError("suffix JVP segment state drifted")

    def _token_teacher_kl(
        self,
        logits: Tensor,
        *,
        validate_finite: bool = True,
    ) -> Tensor:
        indices = self._supervised_indices.to(device=logits.device)
        candidate_f64 = logits[indices[:, 0], indices[:, 1]].to(
            dtype=torch.float64
        )
        teacher_f64 = self._teacher_logits[
            indices[:, 0], indices[:, 1]
        ].to(dtype=torch.float64)
        teacher_log_probabilities = F.log_softmax(teacher_f64, dim=-1)
        candidate_log_probabilities = F.log_softmax(candidate_f64, dim=-1)
        token_kl = (
            teacher_log_probabilities.exp()
            * (teacher_log_probabilities - candidate_log_probabilities)
        ).sum(dim=-1)
        if token_kl.dtype != torch.float64:
            raise ValueError("suffix JVP token teacher KL must be float64")
        if validate_finite and not bool(torch.isfinite(token_kl).all()):
            raise ValueError("suffix JVP token teacher KL is nonfinite")
        return token_kl

    def execute(
        self,
        path_h4_f64: Tensor,
        direction_h4_f64: Tensor,
        *,
        full_h4: Tensor,
        full_logits: Tensor,
    ) -> Gemma3L3L4H4SuffixJVP:
        """Run one exact suffix JVP at an authenticated full-model primal."""

        self.validate_integrity()
        path = _finite_tensor(path_h4_f64, label="path H4")
        direction = _finite_tensor(direction_h4_f64, label="direction H4")
        reference_h4 = _finite_tensor(full_h4, label="full H4")
        reference_logits = _finite_tensor(full_logits, label="full logits")
        if (
            path.dtype != torch.float64
            or direction.dtype != torch.float64
            or path.ndim != 3
            or direction.shape != path.shape
            or path.device != direction.device
            or path.requires_grad
            or direction.requires_grad
            or not path.is_contiguous()
            or not direction.is_contiguous()
        ):
            raise ValueError("path and direction H4 must be contiguous float64 peers")
        if (
            reference_h4.ndim != 3
            or reference_h4.shape != path.shape
            or reference_h4.dtype not in _SUPPORTED_LIVE_DTYPES
            or reference_h4.device != path.device
            or reference_h4.device != self._sequence.device
            or reference_h4.requires_grad
            or not reference_h4.is_contiguous()
        ):
            raise ValueError("full H4 geometry, dtype, or device is invalid")
        expected_h4_shape = (
            self._sequence.batch_size,
            self._sequence.query_length,
            self._suffix_segments[0].input_width,
        )
        if tuple(reference_h4.shape) != expected_h4_shape:
            raise ValueError("full H4 does not match the layer.5 input boundary")
        if (
            reference_logits.ndim != 3
            or reference_logits.shape != self._teacher_logits.shape
            or reference_logits.dtype != reference_h4.dtype
            or reference_logits.device != reference_h4.device
            or reference_logits.requires_grad
            or not reference_logits.is_contiguous()
        ):
            raise ValueError("full logits geometry, dtype, or device is invalid")

        path_sha256 = _runtime_tensor_sha256(path)
        direction_sha256 = _runtime_tensor_sha256(direction)
        full_h4_sha256 = _runtime_tensor_sha256(reference_h4)
        segment_call_count = 0
        projection_call_count = 0

        def suffix_token_teacher_kl(
            candidate_path_f64: Tensor,
        ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
            nonlocal segment_call_count, projection_call_count
            # This is the one and only H4 path dtype conversion in the call.
            hidden_states = candidate_path_f64.to(dtype=reference_h4.dtype)
            cast_h4_aux = hidden_states
            segment_finite_flags: list[Tensor] = []
            for segment in self._suffix_segments:
                run = self._adapter.run_segment(
                    segment,
                    hidden_states,
                    self._sequence,
                )
                segment_call_count += 1
                if run.sequence is not self._sequence:
                    raise RuntimeError("suffix segment changed sequence ownership")
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
                    raise ValueError("suffix segment output contract differs")
                segment_finite_flags.append(torch.isfinite(hidden_states).all())
            logits = self._adapter.project_logits(
                hidden_states,
                self._sequence,
            )
            projection_call_count += 1
            if (
                logits.shape != reference_logits.shape
                or logits.dtype != reference_logits.dtype
                or logits.device != reference_logits.device
            ):
                raise ValueError("suffix logits contract differs")
            return self._token_teacher_kl(
                logits,
                validate_finite=False,
            ), (cast_h4_aux, logits, torch.stack(segment_finite_flags))

        primal, directional, aux = torch.func.jvp(
            suffix_token_teacher_kl,
            (path,),
            (direction,),
            strict=True,
            has_aux=True,
        )
        cast_h4, suffix_logits, segment_finite_flags = (
            value.detach().contiguous() for value in aux
        )
        if (
            segment_call_count != 13
            or projection_call_count != 1
        ):
            raise RuntimeError("suffix JVP suffix geometry changed")
        if (
            segment_finite_flags.dtype != torch.bool
            or segment_finite_flags.shape != (13,)
            or not bool(segment_finite_flags.all())
        ):
            raise ValueError("one or more suffix segment outputs are nonfinite")
        if not _bitwise_equal(cast_h4, reference_h4):
            raise ValueError("float64 path does not cast bitwise to the full H4")
        cast_h4_sha256 = _runtime_tensor_sha256(cast_h4)
        if cast_h4_sha256 != full_h4_sha256:
            raise RuntimeError("cast and full H4 hashes differ despite byte parity")

        if not bool(torch.isfinite(suffix_logits).all()):
            raise ValueError("suffix logits are nonfinite")
        if not _bitwise_equal(suffix_logits, reference_logits):
            raise RuntimeError("suffix logits are not bitwise equal to full logits")
        full_logits_sha256 = _runtime_tensor_sha256(reference_logits)
        suffix_logits_sha256 = _runtime_tensor_sha256(suffix_logits)
        if suffix_logits_sha256 != full_logits_sha256:
            raise RuntimeError("suffix and full logits hashes differ")

        full_token_kl = self._token_teacher_kl(reference_logits).detach().contiguous()
        primal = primal.detach().contiguous()
        directional = directional.detach().contiguous()
        if (
            primal.dtype != torch.float64
            or directional.dtype != torch.float64
            or primal.shape != (self.token_count,)
            or directional.shape != (self.token_count,)
            or not bool(torch.isfinite(primal).all())
            or not bool(torch.isfinite(directional).all())
        ):
            raise ValueError("suffix JVP returned invalid token vectors")
        if not _bitwise_equal(primal, full_token_kl):
            raise RuntimeError("suffix and full token KL primals are not bitwise equal")
        full_token_kl_sha256 = _runtime_tensor_sha256(full_token_kl)
        primal_sha256 = _runtime_tensor_sha256(primal)
        if primal_sha256 != full_token_kl_sha256:
            raise RuntimeError("suffix and full token KL hashes differ")

        self.validate_integrity()
        receipt = Gemma3L3L4H4SuffixJVPReceipt(
            adapter_semantic_sha256=self._adapter_semantic_sha256,
            adapter_model_sha256=self._adapter_model_sha256,
            adapter_execution_sha256=self._adapter_execution_sha256,
            suffix_segment_fingerprints=self._suffix_segment_fingerprints,
            sequence_sha256=self._sequence_sha256,
            teacher_logits_sha256=self._teacher_logits_sha256,
            supervised_indices_sha256=self._supervised_indices_sha256,
            path_h4_sha256=path_sha256,
            direction_h4_sha256=direction_sha256,
            full_h4_sha256=full_h4_sha256,
            cast_h4_sha256=cast_h4_sha256,
            full_logits_sha256=full_logits_sha256,
            suffix_logits_sha256=suffix_logits_sha256,
            full_token_teacher_kl_sha256=full_token_kl_sha256,
            primal_token_teacher_kl_sha256=primal_sha256,
            directional_token_teacher_kl_sha256=(
                _runtime_tensor_sha256(directional)
            ),
            h4_shape=tuple(int(size) for size in reference_h4.shape),
            logits_shape=tuple(int(size) for size in reference_logits.shape),
            token_count=self.token_count,
            live_dtype=str(reference_h4.dtype),
            teacher_dtype=str(self._teacher_logits.dtype),
            device=str(reference_h4.device),
            suffix_segment_call_count=segment_call_count,
            logit_projection_call_count=projection_call_count,
        )
        result = Gemma3L3L4H4SuffixJVP(
            primal_token_teacher_kl=primal,
            directional_token_teacher_kl=directional,
            receipt=receipt,
        )
        result.validate_integrity()
        return result


@dataclass(frozen=True, slots=True)
class Gemma3L3L4H4DiscreteCastIntervalStats:
    """Authenticated exact statistics for two adjacent live-dtype H4 nodes."""

    left_h4_sha256: str
    right_h4_sha256: str
    ideal_left_h4_f64_sha256: str
    ideal_right_h4_f64_sha256: str
    ideal_h4_delta_f64_sha256: str
    live_h4_delta_f64_sha256: str
    h4_shape: tuple[int, int, int]
    live_dtype: str
    device: str
    left_path_fraction: float
    right_path_fraction: float
    interval_width: float
    coordinate_count: int
    ideal_changed_coordinate_count: int
    live_changed_coordinate_count: int
    preserved_change_coordinate_count: int
    cast_collision_coordinate_count: int
    static_coordinate_count: int
    unchanged_live_coordinate_count: int
    left_unique_value_count: int
    right_unique_value_count: int
    unique_bit_pair_count: int
    ideal_displacement_squared_l2: float
    ideal_displacement_l2: float
    ideal_displacement_maximum_abs: float
    live_displacement_squared_l2: float
    live_displacement_l2: float
    live_displacement_maximum_abs: float
    left_token_teacher_kl_sha256: str | None = None
    right_token_teacher_kl_sha256: str | None = None
    token_teacher_kl_delta_sha256: str | None = None
    token_teacher_kl_normalized_secant_sha256: str | None = None
    token_count: int | None = None
    token_teacher_kl_delta_mean: float | None = None
    token_teacher_kl_delta_squared_l2: float | None = None
    token_teacher_kl_delta_maximum_abs: float | None = None
    token_teacher_kl_normalized_secant_mean: float | None = None
    token_teacher_kl_normalized_secant_squared_l2: float | None = None
    token_teacher_kl_normalized_secant_maximum_abs: float | None = None
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.left_h4_sha256, label="left interval H4")
        _require_sha256(self.right_h4_sha256, label="right interval H4")
        for value, label in (
            (self.ideal_left_h4_f64_sha256, "ideal left interval H4"),
            (self.ideal_right_h4_f64_sha256, "ideal right interval H4"),
            (self.ideal_h4_delta_f64_sha256, "ideal interval H4 delta"),
            (self.live_h4_delta_f64_sha256, "live interval H4 delta"),
        ):
            _require_sha256(value, label=label)
        _shape(self.h4_shape, rank=3, label="interval H4")
        if self.live_dtype not in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
        }:
            raise ValueError("interval live dtype is unsupported")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("interval device must be nonempty")
        left_fraction = _finite_nonnegative(
            self.left_path_fraction,
            label="left path fraction",
        )
        right_fraction = _finite_nonnegative(
            self.right_path_fraction,
            label="right path fraction",
        )
        width = _finite_nonnegative(
            self.interval_width,
            label="path interval width",
        )
        if (
            not 0.0 <= left_fraction < right_fraction <= 1.0
            or width <= 0.0
            or width.hex() != (right_fraction - left_fraction).hex()
        ):
            raise ValueError("path fractions or interval width differ")
        if (
            type(self.coordinate_count) is not int
            or self.coordinate_count <= 0
        ):
            raise ValueError("interval coordinate counts are inconsistent")
        counts = (
            self.ideal_changed_coordinate_count,
            self.live_changed_coordinate_count,
            self.preserved_change_coordinate_count,
            self.cast_collision_coordinate_count,
            self.static_coordinate_count,
            self.unchanged_live_coordinate_count,
        )
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("interval coordinate counts are invalid")
        if (
            self.preserved_change_coordinate_count
            + self.cast_collision_coordinate_count
            != self.ideal_changed_coordinate_count
            or self.preserved_change_coordinate_count
            != self.live_changed_coordinate_count
            or self.cast_collision_coordinate_count
            + self.static_coordinate_count
            != self.unchanged_live_coordinate_count
            or self.live_changed_coordinate_count
            + self.unchanged_live_coordinate_count
            != self.coordinate_count
            or self.ideal_changed_coordinate_count
            + self.static_coordinate_count
            != self.coordinate_count
        ):
            raise ValueError("interval coordinate count partitions differ")
        for count, label in (
            (self.left_unique_value_count, "left unique values"),
            (self.right_unique_value_count, "right unique values"),
            (self.unique_bit_pair_count, "unique bit pairs"),
        ):
            if type(count) is not int or not 1 <= count <= self.coordinate_count:
                raise ValueError(f"{label} count is invalid")
        energies: dict[str, float] = {}
        for prefix, squared_value, norm_value, maximum_value in (
            (
                "ideal",
                self.ideal_displacement_squared_l2,
                self.ideal_displacement_l2,
                self.ideal_displacement_maximum_abs,
            ),
            (
                "live",
                self.live_displacement_squared_l2,
                self.live_displacement_l2,
                self.live_displacement_maximum_abs,
            ),
        ):
            squared = _finite_nonnegative(
                squared_value,
                label=f"{prefix} interval displacement squared L2",
            )
            norm = _finite_nonnegative(
                norm_value,
                label=f"{prefix} interval displacement L2",
            )
            maximum = _finite_nonnegative(
                maximum_value,
                label=f"{prefix} interval displacement maximum",
            )
            tolerance = (
                64.0 * torch.finfo(torch.float64).eps * max(squared, 1.0)
            )
            if abs(norm * norm - squared) > tolerance:
                raise ValueError(
                    f"{prefix} interval displacement norm and energy differ"
                )
            energies[f"{prefix}_squared"] = squared
            energies[f"{prefix}_norm"] = norm
            energies[f"{prefix}_maximum"] = maximum
        if self.ideal_changed_coordinate_count == 0 and (
            energies["ideal_squared"] != 0.0
            or energies["ideal_maximum"] != 0.0
        ):
            raise ValueError("static ideal interval has nonzero displacement")
        if self.live_changed_coordinate_count == 0 and (
            energies["live_squared"] != 0.0
            or energies["live_maximum"] != 0.0
        ):
            raise ValueError("static live interval has nonzero displacement")

        optional_hashes = (
            self.left_token_teacher_kl_sha256,
            self.right_token_teacher_kl_sha256,
            self.token_teacher_kl_delta_sha256,
            self.token_teacher_kl_normalized_secant_sha256,
        )
        optional_scalars = (
            self.token_count,
            self.token_teacher_kl_delta_mean,
            self.token_teacher_kl_delta_squared_l2,
            self.token_teacher_kl_delta_maximum_abs,
            self.token_teacher_kl_normalized_secant_mean,
            self.token_teacher_kl_normalized_secant_squared_l2,
            self.token_teacher_kl_normalized_secant_maximum_abs,
        )
        has_kl = optional_hashes[0] is not None
        if any((value is not None) != has_kl for value in optional_hashes):
            raise ValueError("interval KL hashes must be all present or all absent")
        if any((value is not None) != has_kl for value in optional_scalars):
            raise ValueError("interval KL summaries must be all present or all absent")
        if has_kl:
            for value in optional_hashes:
                _require_sha256(value, label="interval token teacher KL")
            if type(self.token_count) is not int or self.token_count <= 0:
                raise ValueError("interval KL token count must be positive")
            for value, label in (
                (self.token_teacher_kl_delta_mean, "interval KL delta mean"),
                (
                    self.token_teacher_kl_normalized_secant_mean,
                    "interval KL normalized secant mean",
                ),
            ):
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError(f"{label} must be a finite float")
            for value, label in (
                (
                    self.token_teacher_kl_delta_squared_l2,
                    "interval KL delta squared L2",
                ),
                (
                    self.token_teacher_kl_delta_maximum_abs,
                    "interval KL delta maximum",
                ),
                (
                    self.token_teacher_kl_normalized_secant_squared_l2,
                    "interval KL normalized secant squared L2",
                ),
                (
                    self.token_teacher_kl_normalized_secant_maximum_abs,
                    "interval KL normalized secant maximum",
                ),
            ):
                _finite_nonnegative(value, label=label)
        object.__setattr__(self, "left_path_fraction", left_fraction)
        object.__setattr__(self, "right_path_fraction", right_fraction)
        object.__setattr__(self, "interval_width", width)
        for field_name, energy in (
            ("ideal_displacement_squared_l2", energies["ideal_squared"]),
            ("ideal_displacement_l2", energies["ideal_norm"]),
            ("ideal_displacement_maximum_abs", energies["ideal_maximum"]),
            ("live_displacement_squared_l2", energies["live_squared"]),
            ("live_displacement_l2", energies["live_norm"]),
            ("live_displacement_maximum_abs", energies["live_maximum"]),
        ):
            object.__setattr__(self, field_name, energy)
        object.__setattr__(
            self,
            "artifact_sha256",
            _artifact_sha256(
                _INTERVAL_DOMAIN,
                self.metadata(include_artifact=False),
            ),
        )

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "left_h4_sha256": self.left_h4_sha256,
            "right_h4_sha256": self.right_h4_sha256,
            "ideal_left_h4_f64_sha256": self.ideal_left_h4_f64_sha256,
            "ideal_right_h4_f64_sha256": self.ideal_right_h4_f64_sha256,
            "ideal_h4_delta_f64_sha256": self.ideal_h4_delta_f64_sha256,
            "live_h4_delta_f64_sha256": self.live_h4_delta_f64_sha256,
            "h4_shape": self.h4_shape,
            "live_dtype": self.live_dtype,
            "device": self.device,
            "left_path_fraction_hex": self.left_path_fraction.hex(),
            "right_path_fraction_hex": self.right_path_fraction.hex(),
            "interval_width_hex": self.interval_width.hex(),
            "coordinate_count": self.coordinate_count,
            "ideal_changed_coordinate_count": (
                self.ideal_changed_coordinate_count
            ),
            "live_changed_coordinate_count": self.live_changed_coordinate_count,
            "preserved_change_coordinate_count": (
                self.preserved_change_coordinate_count
            ),
            "cast_collision_coordinate_count": (
                self.cast_collision_coordinate_count
            ),
            "static_coordinate_count": self.static_coordinate_count,
            "unchanged_live_coordinate_count": (
                self.unchanged_live_coordinate_count
            ),
            "left_unique_value_count": self.left_unique_value_count,
            "right_unique_value_count": self.right_unique_value_count,
            "unique_bit_pair_count": self.unique_bit_pair_count,
            "ideal_displacement_squared_l2_hex": (
                self.ideal_displacement_squared_l2.hex()
            ),
            "ideal_displacement_l2_hex": self.ideal_displacement_l2.hex(),
            "ideal_displacement_maximum_abs_hex": (
                self.ideal_displacement_maximum_abs.hex()
            ),
            "live_displacement_squared_l2_hex": (
                self.live_displacement_squared_l2.hex()
            ),
            "live_displacement_l2_hex": self.live_displacement_l2.hex(),
            "live_displacement_maximum_abs_hex": (
                self.live_displacement_maximum_abs.hex()
            ),
            "left_token_teacher_kl_sha256": (
                self.left_token_teacher_kl_sha256
            ),
            "right_token_teacher_kl_sha256": (
                self.right_token_teacher_kl_sha256
            ),
            "token_teacher_kl_delta_sha256": (
                self.token_teacher_kl_delta_sha256
            ),
            "token_teacher_kl_normalized_secant_sha256": (
                self.token_teacher_kl_normalized_secant_sha256
            ),
            "token_count": self.token_count,
            "token_teacher_kl_delta_mean_hex": (
                None
                if self.token_teacher_kl_delta_mean is None
                else self.token_teacher_kl_delta_mean.hex()
            ),
            "token_teacher_kl_delta_squared_l2_hex": (
                None
                if self.token_teacher_kl_delta_squared_l2 is None
                else self.token_teacher_kl_delta_squared_l2.hex()
            ),
            "token_teacher_kl_delta_maximum_abs_hex": (
                None
                if self.token_teacher_kl_delta_maximum_abs is None
                else self.token_teacher_kl_delta_maximum_abs.hex()
            ),
            "token_teacher_kl_normalized_secant_mean_hex": (
                None
                if self.token_teacher_kl_normalized_secant_mean is None
                else self.token_teacher_kl_normalized_secant_mean.hex()
            ),
            "token_teacher_kl_normalized_secant_squared_l2_hex": (
                None
                if self.token_teacher_kl_normalized_secant_squared_l2 is None
                else self.token_teacher_kl_normalized_secant_squared_l2.hex()
            ),
            "token_teacher_kl_normalized_secant_maximum_abs_hex": (
                None
                if self.token_teacher_kl_normalized_secant_maximum_abs is None
                else self.token_teacher_kl_normalized_secant_maximum_abs.hex()
            ),
            "serialized_h4_or_token_kl_tensors": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        expected = _artifact_sha256(
            _INTERVAL_DOMAIN,
            self.metadata(include_artifact=False),
        )
        if expected != _require_sha256(
            self.artifact_sha256,
            label="discrete cast interval artifact",
        ):
            raise RuntimeError("discrete cast interval receipt drifted")


def gemma3_l3_l4_h4_discrete_cast_interval_stats(
    left_h4: Tensor,
    right_h4: Tensor,
    *,
    ideal_left_h4_f64: Tensor,
    ideal_right_h4_f64: Tensor,
    left_path_fraction: float,
    right_path_fraction: float,
    left_token_teacher_kl: Tensor | None = None,
    right_token_teacher_kl: Tensor | None = None,
) -> Gemma3L3L4H4DiscreteCastIntervalStats:
    """Summarize exact representable changes across one cast-path interval."""

    left = _finite_tensor(left_h4, label="left interval H4")
    right = _finite_tensor(right_h4, label="right interval H4")
    ideal_left = _finite_tensor(
        ideal_left_h4_f64,
        label="ideal left interval H4",
    )
    ideal_right = _finite_tensor(
        ideal_right_h4_f64,
        label="ideal right interval H4",
    )
    if (
        left.ndim != 3
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.dtype not in _SUPPORTED_LIVE_DTYPES
        or left.device != right.device
        or left.requires_grad
        or right.requires_grad
    ):
        raise ValueError("interval H4 tensors must be same-shape live-dtype peers")
    if (
        ideal_left.dtype != torch.float64
        or ideal_right.dtype != torch.float64
        or ideal_left.shape != left.shape
        or ideal_right.shape != left.shape
        or ideal_left.device != left.device
        or ideal_right.device != left.device
        or ideal_left.requires_grad
        or ideal_right.requires_grad
        or not ideal_left.is_contiguous()
        or not ideal_right.is_contiguous()
    ):
        raise ValueError(
            "ideal interval H4 tensors must be contiguous float64 live-grid peers"
        )
    left_fraction = _finite_nonnegative(
        left_path_fraction,
        label="left path fraction",
    )
    right_fraction = _finite_nonnegative(
        right_path_fraction,
        label="right path fraction",
    )
    if not 0.0 <= left_fraction < right_fraction <= 1.0:
        raise ValueError("interval path fractions must increase within [0, 1]")
    interval_width = right_fraction - left_fraction
    cast_left = ideal_left.to(dtype=left.dtype)
    cast_right = ideal_right.to(dtype=right.dtype)
    if not _bitwise_equal(cast_left, left) or not _bitwise_equal(
        cast_right,
        right,
    ):
        raise ValueError("live interval endpoints differ from ideal endpoint casts")
    left_cpu = left.detach().to(device="cpu").contiguous()
    right_cpu = right.detach().to(device="cpu").contiguous()
    ideal_left_cpu = ideal_left.detach().to(device="cpu").contiguous()
    ideal_right_cpu = ideal_right.detach().to(device="cpu").contiguous()
    coordinate_count = left_cpu.numel()
    live_byte_width = left_cpu.element_size()
    left_bytes = left_cpu.view(torch.uint8).reshape(
        coordinate_count,
        live_byte_width,
    )
    right_bytes = right_cpu.view(torch.uint8).reshape(
        coordinate_count,
        live_byte_width,
    )
    live_changed = (left_bytes != right_bytes).any(dim=1)
    ideal_changed = ideal_left_cpu.reshape(-1) != ideal_right_cpu.reshape(-1)
    if bool((live_changed & ~ideal_changed).any()):
        raise RuntimeError("live cast changed a coordinate static in ideal f64")
    preserved_change = ideal_changed & live_changed
    cast_collision = ideal_changed & ~live_changed
    static = ~ideal_changed & ~live_changed
    ideal_changed_count = int(ideal_changed.count_nonzero())
    live_changed_count = int(live_changed.count_nonzero())
    preserved_change_count = int(preserved_change.count_nonzero())
    cast_collision_count = int(cast_collision.count_nonzero())
    static_count = int(static.count_nonzero())
    bit_pairs = torch.cat((left_bytes, right_bytes), dim=1)
    ideal_delta = (ideal_right - ideal_left).detach().contiguous()
    live_delta = (
        right.to(dtype=torch.float64) - left.to(dtype=torch.float64)
    ).detach().contiguous()
    ideal_squared_l2 = float(torch.sum(ideal_delta.square()))
    live_squared_l2 = float(torch.sum(live_delta.square()))
    ideal_displacement_l2 = math.sqrt(ideal_squared_l2)
    live_displacement_l2 = math.sqrt(live_squared_l2)
    ideal_maximum_abs = float(ideal_delta.abs().max())
    live_maximum_abs = float(live_delta.abs().max())

    optional = (left_token_teacher_kl, right_token_teacher_kl)
    if (optional[0] is None) != (optional[1] is None):
        raise ValueError("both interval token-KL endpoints must be supplied")
    kl_metadata: dict[str, Any] = {
        "left_token_teacher_kl_sha256": None,
        "right_token_teacher_kl_sha256": None,
        "token_teacher_kl_delta_sha256": None,
        "token_teacher_kl_normalized_secant_sha256": None,
        "token_count": None,
        "token_teacher_kl_delta_mean": None,
        "token_teacher_kl_delta_squared_l2": None,
        "token_teacher_kl_delta_maximum_abs": None,
        "token_teacher_kl_normalized_secant_mean": None,
        "token_teacher_kl_normalized_secant_squared_l2": None,
        "token_teacher_kl_normalized_secant_maximum_abs": None,
    }
    if optional[0] is not None and optional[1] is not None:
        left_kl = _finite_tensor(optional[0], label="left interval token KL")
        right_kl = _finite_tensor(optional[1], label="right interval token KL")
        if (
            left_kl.dtype != torch.float64
            or right_kl.dtype != torch.float64
            or left_kl.ndim != 1
            or left_kl.shape != right_kl.shape
            or left_kl.numel() <= 0
            or left_kl.device != right_kl.device
            or left_kl.requires_grad
            or right_kl.requires_grad
        ):
            raise ValueError("interval token KL endpoints must be float64 peers")
        delta = (right_kl - left_kl).detach().contiguous()
        normalized_secant = (delta / interval_width).detach().contiguous()
        kl_metadata = {
            "left_token_teacher_kl_sha256": _runtime_tensor_sha256(left_kl),
            "right_token_teacher_kl_sha256": _runtime_tensor_sha256(right_kl),
            "token_teacher_kl_delta_sha256": _runtime_tensor_sha256(delta),
            "token_teacher_kl_normalized_secant_sha256": (
                _runtime_tensor_sha256(normalized_secant)
            ),
            "token_count": int(delta.numel()),
            "token_teacher_kl_delta_mean": float(delta.mean()),
            "token_teacher_kl_delta_squared_l2": float(
                torch.sum(delta.square())
            ),
            "token_teacher_kl_delta_maximum_abs": float(delta.abs().max()),
            "token_teacher_kl_normalized_secant_mean": float(
                normalized_secant.mean()
            ),
            "token_teacher_kl_normalized_secant_squared_l2": float(
                torch.sum(normalized_secant.square())
            ),
            "token_teacher_kl_normalized_secant_maximum_abs": float(
                normalized_secant.abs().max()
            ),
        }

    receipt = Gemma3L3L4H4DiscreteCastIntervalStats(
        left_h4_sha256=_runtime_tensor_sha256(left),
        right_h4_sha256=_runtime_tensor_sha256(right),
        ideal_left_h4_f64_sha256=_runtime_tensor_sha256(ideal_left),
        ideal_right_h4_f64_sha256=_runtime_tensor_sha256(ideal_right),
        ideal_h4_delta_f64_sha256=_runtime_tensor_sha256(ideal_delta),
        live_h4_delta_f64_sha256=_runtime_tensor_sha256(live_delta),
        h4_shape=tuple(int(size) for size in left.shape),
        live_dtype=str(left.dtype),
        device=str(left.device),
        left_path_fraction=left_fraction,
        right_path_fraction=right_fraction,
        interval_width=interval_width,
        coordinate_count=coordinate_count,
        ideal_changed_coordinate_count=ideal_changed_count,
        live_changed_coordinate_count=live_changed_count,
        preserved_change_coordinate_count=preserved_change_count,
        cast_collision_coordinate_count=cast_collision_count,
        static_coordinate_count=static_count,
        unchanged_live_coordinate_count=coordinate_count - live_changed_count,
        left_unique_value_count=int(torch.unique(left_bytes, dim=0).shape[0]),
        right_unique_value_count=int(torch.unique(right_bytes, dim=0).shape[0]),
        unique_bit_pair_count=int(torch.unique(bit_pairs, dim=0).shape[0]),
        ideal_displacement_squared_l2=ideal_squared_l2,
        ideal_displacement_l2=ideal_displacement_l2,
        ideal_displacement_maximum_abs=ideal_maximum_abs,
        live_displacement_squared_l2=live_squared_l2,
        live_displacement_l2=live_displacement_l2,
        live_displacement_maximum_abs=live_maximum_abs,
        **kl_metadata,
    )
    receipt.validate_integrity()
    return receipt
