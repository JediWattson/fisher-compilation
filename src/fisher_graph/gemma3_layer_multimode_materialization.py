"""Typed capture-time authority for full-A MLP row materialization.

The production capture bridge stores every selected MLP parent mode as an
authenticated float32 little-endian spool and exposes its numerical semantics
as CPU float64 chunks.  Fitting consumes one materialized CPU float32
``LayerMLPFlowRows`` tensor pair at a time.  This module is the short-lived
authority at that cast boundary: while both representations exist, it streams
the spool a second time, checks exact bit equality after the frozen float32
cast for both ``x`` and ``v_T``, and emits tensor-free receipts.

It has no role loader, model access, panel reader, artifact writer, or fresh
role API.  Receipt parsers consume JSON-safe scalar state only and require a
separately trusted receipt hash pin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import sys
from weakref import WeakValueDictionary

import numpy as np
import torch
from torch import Tensor

from . import gemma3_chart_native_neighbor_experiment as _predecessor
from .gemma3_chart_native_neighbor_experiment import (
    RuntimeQualificationReceipt,
    derive_runtime_session_binding_sha256,
)
from .gemma3_layer_multimode_capture import (
    A_DEVELOPMENT_ROLE,
    GEMMA3_HIDDEN_WIDTH,
    GEMMA3_LAYER_COUNT,
    PRODUCTION_FULL_A_CAPTURE_SCOPE,
    ADevelopmentPartitionReceipt,
    FullALayerMLPSpoolBridgeReceipt,
    LayerMLPFlowRows,
    LayerMLPFlowSpoolInput,
    build_A_development_partition_receipt,
)


__all__ = [
    "LayerMLPFlowMaterializationInventoryReceipt",
    "LayerMLPFlowMaterializationReceipt",
    "LayerMLPFlowMaterializationResult",
    "build_layer_mlp_flow_materialization_inventory",
    "materialize_layer_mlp_flow_rows",
]


_LAYER_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.layer_flow_materialization.v1"
)
_INVENTORY_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.layer_flow_materialization_inventory.v1"
)
_FORMAT_VERSION = 1
_LAYER_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:layer-materialization:v1\0"
)
_INVENTORY_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:materialization-inventory:v1\0"
)
_CAPTURE_TENSOR_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:tensor:v1\0"
)
_FLOW_ROWS_DOMAIN = b"fisher_graph:gemma3-layer-multimode:flow-rows:v1\0"
_SHARED_CHART_TENSOR_DOMAIN = b"fisher_graph:shared-chart-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAST_ABI = (
    "authenticated_float32_le_spool_upcast_CPU_float64_then_"
    "torch_to_float32_exact_bit_equality.v1"
)
_FLOAT64_SEMANTIC_HASH_ABI = (
    "sha256(shared_chart_tensor_domain || canonical_JSON_dtype_float64_le_"
    "shape || canonical_float64_le_bytes).v1"
)
_FLOAT32_TENSOR_HASH_ABI = (
    "sha256(layer_multimode_tensor_domain || shape_NUL_torch.float32_NUL || "
    "contiguous_native_float32_bytes).v1"
)
_BUILDER_AUTHORITY = object()
_LIVE_LAYER_RECEIPTS: WeakValueDictionary[int, object] = WeakValueDictionary()
_LIVE_MATERIALIZATION_RESULTS: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)
_LIVE_MATERIALIZATION_INVENTORIES: WeakValueDictionary[int, object] = (
    WeakValueDictionary()
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _mode_id(layer_ordinal: int) -> str:
    _require_exact_int(layer_ordinal, label="layer_ordinal")
    if layer_ordinal >= GEMMA3_LAYER_COUNT:
        raise ValueError("layer_ordinal is outside the 18-layer model")
    return f"mode.mutation.{2 + 2 * layer_ordinal}"


def _capture_tensor_hasher(shape: tuple[int, int]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(_CAPTURE_TENSOR_DOMAIN)
    digest.update(f"{shape}\0torch.float32\0".encode("ascii"))
    return digest


def _float64_semantic_hasher(shape: tuple[int, int]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(_SHARED_CHART_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"dtype": "float64_le", "shape": list(shape)}
        )
    )
    return digest


def _float32_bytes(value: Tensor) -> bytes:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.requires_grad
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("float32 verification tensor is not canonical")
    return value.detach().numpy().tobytes(order="C")


def _float64_le_bytes(value: Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("materialization verification requires little endian")
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.requires_grad
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("float64 spool verification tensor is not canonical")
    array = value.detach().numpy().astype(np.dtype("<f8"), copy=False)
    return array.tobytes(order="C")


def _capture_tensor_sha256(value: Tensor) -> str:
    if value.ndim != 2:
        raise ValueError("capture tensor must be rank two")
    hasher = _capture_tensor_hasher((int(value.shape[0]), int(value.shape[1])))
    hasher.update(_float32_bytes(value))
    return hasher.hexdigest()


def _flow_rows_receipt_sha256(rows: LayerMLPFlowRows) -> str:
    return _digest(_FLOW_ROWS_DOMAIN, rows.payload_without_sha256())


def _strict_replay_bridge(
    receipt: FullALayerMLPSpoolBridgeReceipt,
) -> FullALayerMLPSpoolBridgeReceipt:
    state = json.loads(json.dumps(receipt.state_dict(), sort_keys=True))
    restored = FullALayerMLPSpoolBridgeReceipt.from_state_dict(
        state,
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    if restored != receipt:
        raise ValueError("full-A bridge receipt did not strictly replay")
    return restored


def _strict_replay_partition(
    receipt: ADevelopmentPartitionReceipt,
) -> ADevelopmentPartitionReceipt:
    state = json.loads(json.dumps(receipt.state_dict(), sort_keys=True))
    restored = ADevelopmentPartitionReceipt.from_state_dict(
        state,
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    if restored != receipt:
        raise ValueError("A development partition did not strictly replay")
    return restored


def _strict_replay_runtime_qualification(
    receipt: RuntimeQualificationReceipt,
) -> RuntimeQualificationReceipt:
    state = {
        **receipt.payload_without_sha256(),
        "qualification_receipt_sha256": receipt.qualification_receipt_sha256,
    }
    restored = _predecessor._parse_runtime_qualification_receipt_payload(
        json.loads(json.dumps(state, sort_keys=True))
    )
    if restored != receipt:
        raise ValueError("runtime qualification receipt did not strictly replay")
    return restored


def _validate_runtime_session_binding(
    *,
    qualification: RuntimeQualificationReceipt,
    preflight_receipt_sha256: str,
    expected_runtime_session_binding_sha256: str,
) -> None:
    _require_sha256(
        preflight_receipt_sha256,
        label="predecessor preflight receipt",
    )
    expected = derive_runtime_session_binding_sha256(
        preflight_receipt_sha256=preflight_receipt_sha256,
        runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        mode_lineage=qualification.mode_lineage,
    )
    if expected != expected_runtime_session_binding_sha256:
        raise ValueError(
            "runtime qualification/source fingerprint does not derive the "
            "full-A bridge session"
        )


def _validate_partition_bridge_binding(
    partition: ADevelopmentPartitionReceipt,
    bridge: FullALayerMLPSpoolBridgeReceipt,
) -> None:
    if (
        partition.source_full_A_bridge_receipt_sha256
        != bridge.receipt_sha256
        or partition.source_role_capture_receipt_sha256
        != bridge.source_role_capture_receipt_sha256
        or partition.source_runtime_session_binding_sha256
        != bridge.source_runtime_session_binding_sha256
        or partition.source_role_panel_binding_sha256
        != bridge.source_role_panel_binding_sha256
        or partition.source_batch_axis_catalog_sha256
        != bridge.source_batch_axis_catalog_sha256
        or partition.source_shared_spool_row_catalog_sha256
        != bridge.shared_spool_row_catalog_sha256
        or partition.source_ordered_objective_row_catalog_sha256
        != bridge.source_ordered_objective_row_catalog_sha256
        or partition.source_score_objective_receipt_sha256
        != bridge.source_score_objective_receipt_sha256
    ):
        raise ValueError("A partition and full-A bridge lineage differ")


def _rebuild_expected_A_partition(
    *,
    spool_input: LayerMLPFlowSpoolInput,
    bridge: FullALayerMLPSpoolBridgeReceipt,
    supplied: ADevelopmentPartitionReceipt,
) -> ADevelopmentPartitionReceipt:
    """Rebuild A's masks from the live spool's scalar example axis."""

    role_manifest: dict[str, object] = {
        "role": _predecessor.FIT_ROLE,
        "ordered_objective_row_catalog_sha256": (
            bridge.source_ordered_objective_row_catalog_sha256
        ),
        "score_objective_receipt_sha256": (
            bridge.source_score_objective_receipt_sha256
        ),
        "ordered_members": [
            {
                "family_id": family_id,
                "example_id": example_id,
                "target_token_count": token_count,
            }
            for family_id, example_id, token_count in zip(
                spool_input.family_ids,
                spool_input.example_ids,
                spool_input.token_counts,
                strict=True,
            )
        ],
    }
    expected = build_A_development_partition_receipt(
        bridge_receipt=bridge,
        role_manifest=role_manifest,
        expected_bridge_receipt_sha256=bridge.receipt_sha256,
    )
    if expected.state_dict() != supplied.state_dict():
        raise ValueError(
            "A development partition does not equal the live spool-derived "
            "partition"
        )
    return expected


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMLPFlowMaterializationReceipt:
    """Tensor-free proof for one exact production-A spool-to-row cast."""

    layer_ordinal: int
    mode_id: str
    source_full_A_bridge_receipt_sha256: str
    a_development_partition_receipt_sha256: str
    fit_partition_sha256: str
    validation_partition_sha256: str
    source_role_capture_receipt_sha256: str
    source_runtime_session_binding_sha256: str
    predecessor_preflight_receipt_sha256: str
    predecessor_runtime_qualification_receipt_sha256: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    compiled_execution_fingerprint_sha256: str
    tied_readout_binding_receipt_sha256: str
    source_role_panel_binding_sha256: str
    source_spool_artifact_sha256: str
    source_spool_row_catalog_sha256: str
    source_spool_tensor_catalog_sha256: str
    source_spool_kernel_receipt_sha256: str
    source_spool_context_float64_semantic_sha256: str
    source_spool_comparison_float64_semantic_sha256: str
    verified_context_float64_semantic_sha256: str
    verified_comparison_float64_semantic_sha256: str
    materialized_chart_state_float32_sha256: str
    materialized_teacher_velocity_float32_sha256: str
    verified_cast_chart_state_float32_sha256: str
    verified_cast_teacher_velocity_float32_sha256: str
    flow_rows_receipt_sha256: str
    row_count: int
    hidden_width: int
    verification_chunk_size: int
    verification_chunk_count: int
    receipt_sha256: str = ""
    schema: str = _LAYER_SCHEMA
    format_version: int = _FORMAT_VERSION
    source_role: str = A_DEVELOPMENT_ROLE
    source_capture_scope: str = PRODUCTION_FULL_A_CAPTURE_SCOPE
    source_spool_storage_dtype: str = "float32_le"
    streamed_spool_dtype: str = "torch.float64"
    materialized_row_dtype: str = "torch.float32"
    cast_abi: str = _CAST_ABI
    float64_semantic_hash_abi: str = _FLOAT64_SEMANTIC_HASH_ABI
    float32_tensor_hash_abi: str = _FLOAT32_TENSOR_HASH_ABI
    exact_x_cast_equality_verified: bool = True
    exact_teacher_velocity_cast_equality_verified: bool = True
    contains_raw_tensors: bool = False
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_logits: bool = False
    contains_gradients: bool = False
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self.schema != _LAYER_SCHEMA
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
            or self.source_role != A_DEVELOPMENT_ROLE
            or self.source_capture_scope != PRODUCTION_FULL_A_CAPTURE_SCOPE
        ):
            raise ValueError("layer materialization receipt header is invalid")
        if self.mode_id != _mode_id(self.layer_ordinal):
            raise ValueError("layer materialization mode/layer identity drifted")
        for label in (
            "source_full_A_bridge_receipt_sha256",
            "a_development_partition_receipt_sha256",
            "fit_partition_sha256",
            "validation_partition_sha256",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "predecessor_preflight_receipt_sha256",
            "predecessor_runtime_qualification_receipt_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "compiled_execution_fingerprint_sha256",
            "tied_readout_binding_receipt_sha256",
            "source_role_panel_binding_sha256",
            "source_spool_artifact_sha256",
            "source_spool_row_catalog_sha256",
            "source_spool_tensor_catalog_sha256",
            "source_spool_kernel_receipt_sha256",
            "source_spool_context_float64_semantic_sha256",
            "source_spool_comparison_float64_semantic_sha256",
            "verified_context_float64_semantic_sha256",
            "verified_comparison_float64_semantic_sha256",
            "materialized_chart_state_float32_sha256",
            "materialized_teacher_velocity_float32_sha256",
            "verified_cast_chart_state_float32_sha256",
            "verified_cast_teacher_velocity_float32_sha256",
            "flow_rows_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.fit_partition_sha256 == self.validation_partition_sha256:
            raise ValueError("fit and validation partition hashes collide")
        if (
            self.source_spool_context_float64_semantic_sha256
            != self.verified_context_float64_semantic_sha256
            or self.source_spool_comparison_float64_semantic_sha256
            != self.verified_comparison_float64_semantic_sha256
        ):
            raise ValueError("streamed float64 semantics do not match the spool")
        if (
            self.materialized_chart_state_float32_sha256
            != self.verified_cast_chart_state_float32_sha256
            or self.materialized_teacher_velocity_float32_sha256
            != self.verified_cast_teacher_velocity_float32_sha256
        ):
            raise ValueError("materialized float32 rows do not match verified casts")
        if (
            type(self.row_count) is not int
            or self.row_count != 24_732
            or type(self.hidden_width) is not int
            or self.hidden_width != GEMMA3_HIDDEN_WIDTH
            or type(self.verification_chunk_size) is not int
            or self.verification_chunk_size <= 0
            or type(self.verification_chunk_count) is not int
            or self.verification_chunk_count
            != math.ceil(self.row_count / self.verification_chunk_size)
        ):
            raise ValueError("layer materialization shape/chunk closure drifted")
        fixed_strings = {
            "source_spool_storage_dtype": "float32_le",
            "streamed_spool_dtype": "torch.float64",
            "materialized_row_dtype": "torch.float32",
            "cast_abi": _CAST_ABI,
            "float64_semantic_hash_abi": _FLOAT64_SEMANTIC_HASH_ABI,
            "float32_tensor_hash_abi": _FLOAT32_TENSOR_HASH_ABI,
        }
        if any(getattr(self, key) != value for key, value in fixed_strings.items()):
            raise ValueError("layer materialization numerical ABI drifted")
        fixed_bools = {
            "exact_x_cast_equality_verified": True,
            "exact_teacher_velocity_cast_equality_verified": True,
            "contains_raw_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_gradients": False,
        }
        for label, expected in fixed_bools.items():
            value = getattr(self, label)
            _require_exact_bool(value, label=label)
            if value is not expected:
                raise ValueError("layer materialization safety/proof flag drifted")
        computed = _digest(_LAYER_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="layer materialization receipt",
        ) != computed:
            raise ValueError("layer materialization receipt digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_role": self.source_role,
            "source_capture_scope": self.source_capture_scope,
            "layer_ordinal": self.layer_ordinal,
            "mode_id": self.mode_id,
            "source_full_A_bridge_receipt_sha256": (
                self.source_full_A_bridge_receipt_sha256
            ),
            "a_development_partition_receipt_sha256": (
                self.a_development_partition_receipt_sha256
            ),
            "fit_partition_sha256": self.fit_partition_sha256,
            "validation_partition_sha256": self.validation_partition_sha256,
            "source_role_capture_receipt_sha256": (
                self.source_role_capture_receipt_sha256
            ),
            "source_runtime_session_binding_sha256": (
                self.source_runtime_session_binding_sha256
            ),
            "predecessor_preflight_receipt_sha256": (
                self.predecessor_preflight_receipt_sha256
            ),
            "predecessor_runtime_qualification_receipt_sha256": (
                self.predecessor_runtime_qualification_receipt_sha256
            ),
            "source_model_fingerprint_sha256": (
                self.source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "compiled_execution_fingerprint_sha256": (
                self.compiled_execution_fingerprint_sha256
            ),
            "tied_readout_binding_receipt_sha256": (
                self.tied_readout_binding_receipt_sha256
            ),
            "source_role_panel_binding_sha256": (
                self.source_role_panel_binding_sha256
            ),
            "source_spool_artifact_sha256": self.source_spool_artifact_sha256,
            "source_spool_row_catalog_sha256": (
                self.source_spool_row_catalog_sha256
            ),
            "source_spool_tensor_catalog_sha256": (
                self.source_spool_tensor_catalog_sha256
            ),
            "source_spool_kernel_receipt_sha256": (
                self.source_spool_kernel_receipt_sha256
            ),
            "source_spool_context_float64_semantic_sha256": (
                self.source_spool_context_float64_semantic_sha256
            ),
            "source_spool_comparison_float64_semantic_sha256": (
                self.source_spool_comparison_float64_semantic_sha256
            ),
            "verified_context_float64_semantic_sha256": (
                self.verified_context_float64_semantic_sha256
            ),
            "verified_comparison_float64_semantic_sha256": (
                self.verified_comparison_float64_semantic_sha256
            ),
            "materialized_chart_state_float32_sha256": (
                self.materialized_chart_state_float32_sha256
            ),
            "materialized_teacher_velocity_float32_sha256": (
                self.materialized_teacher_velocity_float32_sha256
            ),
            "verified_cast_chart_state_float32_sha256": (
                self.verified_cast_chart_state_float32_sha256
            ),
            "verified_cast_teacher_velocity_float32_sha256": (
                self.verified_cast_teacher_velocity_float32_sha256
            ),
            "flow_rows_receipt_sha256": self.flow_rows_receipt_sha256,
            "row_count": self.row_count,
            "hidden_width": self.hidden_width,
            "verification_chunk_size": self.verification_chunk_size,
            "verification_chunk_count": self.verification_chunk_count,
            "source_spool_storage_dtype": self.source_spool_storage_dtype,
            "streamed_spool_dtype": self.streamed_spool_dtype,
            "materialized_row_dtype": self.materialized_row_dtype,
            "cast_abi": self.cast_abi,
            "float64_semantic_hash_abi": self.float64_semantic_hash_abi,
            "float32_tensor_hash_abi": self.float32_tensor_hash_abi,
            "exact_x_cast_equality_verified": (
                self.exact_x_cast_equality_verified
            ),
            "exact_teacher_velocity_cast_equality_verified": (
                self.exact_teacher_velocity_cast_equality_verified
            ),
            "contains_raw_tensors": self.contains_raw_tensors,
            "contains_prompt_text": self.contains_prompt_text,
            "contains_token_ids": self.contains_token_ids,
            "contains_logits": self.contains_logits,
            "contains_gradients": self.contains_gradients,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "receipt_sha256": self.receipt_sha256}

    @property
    def live_materialization_authorized(self) -> bool:
        """True only while this receipt retains capture-time builder authority."""

        return (
            self._construction_authority is _BUILDER_AUTHORITY
            and _LIVE_LAYER_RECEIPTS.get(id(self)) is self
        )

    def assert_integrity(self) -> None:
        if _digest(_LAYER_DOMAIN, self.payload_without_sha256()) != self.receipt_sha256:
            raise ValueError("layer materialization receipt drifted")

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        *,
        expected_receipt_sha256: str,
    ) -> "LayerMLPFlowMaterializationReceipt":
        _require_sha256(
            expected_receipt_sha256,
            label="expected layer materialization receipt",
        )
        if type(state) is not dict:
            raise TypeError("layer materialization receipt state must be an exact dict")
        expected_fields = set(cls.__dataclass_fields__) - {"_construction_authority"}
        if set(state) != expected_fields:
            raise ValueError("layer materialization receipt state fields differ")
        arguments = dict(state)
        receipt = cls(**arguments)
        if receipt.receipt_sha256 != expected_receipt_sha256:
            raise ValueError("layer materialization trusted hash pin differs")
        return receipt


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMLPFlowMaterializationResult:
    """Ephemeral typed pairing of materialized tensors and their receipt."""

    rows: LayerMLPFlowRows
    receipt: LayerMLPFlowMaterializationReceipt
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.rows) is not LayerMLPFlowRows:
            raise TypeError("materialization result rows have the wrong type")
        if type(self.receipt) is not LayerMLPFlowMaterializationReceipt:
            raise TypeError("materialization result receipt has the wrong type")
        self.receipt.assert_integrity()
        if (
            self.rows.layer_ordinal != self.receipt.layer_ordinal
            or self.rows.mode_id != self.receipt.mode_id
            or self.rows.receipt_sha256
            != self.receipt.flow_rows_receipt_sha256
            or self.rows.chart_state_sha256
            != self.receipt.materialized_chart_state_float32_sha256
            or self.rows.teacher_velocity_sha256
            != self.receipt.materialized_teacher_velocity_float32_sha256
        ):
            raise ValueError("materialization result rows/receipt binding drifted")

    @property
    def live_materialization_authorized(self) -> bool:
        """True only for the live result returned by the materializer."""

        return (
            self._construction_authority is _BUILDER_AUTHORITY
            and _LIVE_MATERIALIZATION_RESULTS.get(id(self)) is self
            and self.receipt.live_materialization_authorized
        )


def materialize_layer_mlp_flow_rows(
    spool_input: LayerMLPFlowSpoolInput,
    *,
    a_development_partition_receipt: ADevelopmentPartitionReceipt,
    runtime_qualification_receipt: RuntimeQualificationReceipt,
    predecessor_preflight_receipt_sha256: str,
    verification_chunk_size: int | None = None,
) -> LayerMLPFlowMaterializationResult:
    """Materialize one layer and prove the exact spool-to-row float cast."""

    if type(spool_input) is not LayerMLPFlowSpoolInput:
        raise TypeError("spool_input must be the exact LayerMLPFlowSpoolInput type")
    if spool_input.live_full_A_authorized is not True:
        raise ValueError(
            "spool_input must retain live strict full-A opener authority"
        )
    if type(a_development_partition_receipt) is not ADevelopmentPartitionReceipt:
        raise TypeError(
            "a_development_partition_receipt must be the exact typed receipt"
        )
    if type(runtime_qualification_receipt) is not RuntimeQualificationReceipt:
        raise TypeError(
            "runtime_qualification_receipt must be the exact typed receipt"
        )
    bridge = _strict_replay_bridge(spool_input.bridge_receipt)
    partition = _strict_replay_partition(a_development_partition_receipt)
    qualification = _strict_replay_runtime_qualification(
        runtime_qualification_receipt
    )
    _validate_partition_bridge_binding(partition, bridge)
    partition = _rebuild_expected_A_partition(
        spool_input=spool_input,
        bridge=bridge,
        supplied=partition,
    )
    _validate_runtime_session_binding(
        qualification=qualification,
        preflight_receipt_sha256=predecessor_preflight_receipt_sha256,
        expected_runtime_session_binding_sha256=(
            bridge.source_runtime_session_binding_sha256
        ),
    )
    resolved_chunk_size = spool_input.spool.row_chunk_size
    if verification_chunk_size is not None:
        _require_exact_int(
            verification_chunk_size,
            label="verification_chunk_size",
            minimum=1,
        )
        if verification_chunk_size != resolved_chunk_size:
            raise ValueError(
                "verification_chunk_size must equal the authenticated spool chunk size"
            )
    source = bridge.layer_sources[spool_input.layer_ordinal]
    semantic = spool_input.spool.field_semantic_sha256s
    if (
        source[0] != spool_input.layer_ordinal
        or source[1] != spool_input.mode_id
        or source[2] != spool_input.spool.artifact_sha256
        or source[3] != spool_input.spool.row_catalog_sha256
        or source[4] != spool_input.spool.tensor_catalog_sha256
        or source[5] != semantic["context_values"]
        or source[6] != semantic["comparison_values"]
        or partition.canonical_layer_mode_axis[spool_input.layer_ordinal]
        != (spool_input.layer_ordinal, spool_input.mode_id)
    ):
        raise ValueError("layer spool differs from bridge/partition authority")

    rows = spool_input.materialize_rows()
    if type(rows) is not LayerMLPFlowRows:
        raise TypeError("materialize_rows returned a non-exact flow-row type")
    if (
        rows.source_capture_scope != PRODUCTION_FULL_A_CAPTURE_SCOPE
        or rows.source_spool_artifact_sha256 != source[2]
        or rows.source_spool_row_catalog_sha256 != source[3]
        or rows.source_spool_tensor_catalog_sha256 != source[4]
        or rows.source_role_capture_receipt_sha256
        != bridge.source_role_capture_receipt_sha256
        or rows.source_runtime_session_binding_sha256
        != bridge.source_runtime_session_binding_sha256
        or rows.source_role_panel_binding_sha256
        != bridge.source_role_panel_binding_sha256
        or rows.row_keys != spool_input.row_keys
    ):
        raise ValueError("materialized flow rows lost spool lineage")
    x_sha256 = _capture_tensor_sha256(rows.chart_states)
    velocity_sha256 = _capture_tensor_sha256(rows.teacher_velocities)
    if (
        x_sha256 != rows.chart_state_sha256
        or velocity_sha256 != rows.teacher_velocity_sha256
        or _flow_rows_receipt_sha256(rows) != rows.receipt_sha256
    ):
        raise ValueError("materialized flow-row tensor/receipt integrity drifted")

    shape = (rows.row_count, GEMMA3_HIDDEN_WIDTH)
    context_float64 = _float64_semantic_hasher(shape)
    comparison_float64 = _float64_semantic_hasher(shape)
    cast_x = _capture_tensor_hasher(shape)
    cast_velocity = _capture_tensor_hasher(shape)
    cursor = 0
    chunk_count = 0
    for chunk in spool_input.iter_float64_chunks(
        chunk_size=resolved_chunk_size
    ):
        if chunk.start_row != cursor:
            raise ValueError("verification stream row order drifted")
        expected_x = chunk.chart_states.to(dtype=torch.float32).contiguous()
        expected_velocity = chunk.teacher_velocities.to(
            dtype=torch.float32
        ).contiguous()
        actual_x = rows.chart_states[chunk.start_row : chunk.end_row]
        actual_velocity = rows.teacher_velocities[
            chunk.start_row : chunk.end_row
        ]
        if not torch.equal(actual_x, expected_x):
            raise ValueError("materialized x differs from the exact spool cast")
        if not torch.equal(actual_velocity, expected_velocity):
            raise ValueError(
                "materialized teacher velocity differs from the exact spool cast"
            )
        context_float64.update(_float64_le_bytes(chunk.chart_states))
        comparison_float64.update(_float64_le_bytes(chunk.teacher_velocities))
        cast_x.update(_float32_bytes(expected_x))
        cast_velocity.update(_float32_bytes(expected_velocity))
        cursor = chunk.end_row
        chunk_count += 1
    if cursor != rows.row_count:
        raise ValueError("verification stream did not cover every materialized row")
    verified_context = context_float64.hexdigest()
    verified_comparison = comparison_float64.hexdigest()
    verified_cast_x = cast_x.hexdigest()
    verified_cast_velocity = cast_velocity.hexdigest()
    if (
        verified_context != semantic["context_values"]
        or verified_comparison != semantic["comparison_values"]
    ):
        raise ValueError("streamed float64 semantic hash differs from the spool")
    if verified_cast_x != x_sha256 or verified_cast_velocity != velocity_sha256:
        raise ValueError("streamed cast tensor hash differs from materialized rows")

    receipt = LayerMLPFlowMaterializationReceipt(
        layer_ordinal=spool_input.layer_ordinal,
        mode_id=spool_input.mode_id,
        source_full_A_bridge_receipt_sha256=bridge.receipt_sha256,
        a_development_partition_receipt_sha256=partition.receipt_sha256,
        fit_partition_sha256=partition.fit_partition_sha256,
        validation_partition_sha256=partition.validation_partition_sha256,
        source_role_capture_receipt_sha256=(
            bridge.source_role_capture_receipt_sha256
        ),
        source_runtime_session_binding_sha256=(
            bridge.source_runtime_session_binding_sha256
        ),
        predecessor_preflight_receipt_sha256=(
            predecessor_preflight_receipt_sha256
        ),
        predecessor_runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        source_role_panel_binding_sha256=(
            bridge.source_role_panel_binding_sha256
        ),
        source_spool_artifact_sha256=source[2],
        source_spool_row_catalog_sha256=source[3],
        source_spool_tensor_catalog_sha256=source[4],
        source_spool_kernel_receipt_sha256=(
            spool_input.spool.kernel_receipt_sha256
        ),
        source_spool_context_float64_semantic_sha256=source[5],
        source_spool_comparison_float64_semantic_sha256=source[6],
        verified_context_float64_semantic_sha256=verified_context,
        verified_comparison_float64_semantic_sha256=verified_comparison,
        materialized_chart_state_float32_sha256=x_sha256,
        materialized_teacher_velocity_float32_sha256=velocity_sha256,
        verified_cast_chart_state_float32_sha256=verified_cast_x,
        verified_cast_teacher_velocity_float32_sha256=(
            verified_cast_velocity
        ),
        flow_rows_receipt_sha256=rows.receipt_sha256,
        row_count=rows.row_count,
        hidden_width=GEMMA3_HIDDEN_WIDTH,
        verification_chunk_size=resolved_chunk_size,
        verification_chunk_count=chunk_count,
    )
    object.__setattr__(receipt, "_construction_authority", _BUILDER_AUTHORITY)
    _LIVE_LAYER_RECEIPTS[id(receipt)] = receipt
    result = LayerMLPFlowMaterializationResult(
        rows=rows,
        receipt=receipt,
    )
    object.__setattr__(result, "_construction_authority", _BUILDER_AUTHORITY)
    _LIVE_MATERIALIZATION_RESULTS[id(result)] = result
    return result


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMLPFlowMaterializationInventoryReceipt:
    """Tensor-free ordered authority covering all 18 materialized layers."""

    layer_receipts: tuple[LayerMLPFlowMaterializationReceipt, ...]
    a_development_partition_receipt_sha256: str
    source_full_A_bridge_receipt_sha256: str
    source_role_capture_receipt_sha256: str
    source_runtime_session_binding_sha256: str
    predecessor_preflight_receipt_sha256: str
    predecessor_runtime_qualification_receipt_sha256: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    compiled_execution_fingerprint_sha256: str
    tied_readout_binding_receipt_sha256: str
    source_role_panel_binding_sha256: str
    shared_spool_row_catalog_sha256: str
    fit_partition_sha256: str
    validation_partition_sha256: str
    receipt_sha256: str = ""
    schema: str = _INVENTORY_SCHEMA
    format_version: int = _FORMAT_VERSION
    source_role: str = A_DEVELOPMENT_ROLE
    source_capture_scope: str = PRODUCTION_FULL_A_CAPTURE_SCOPE
    layer_count: int = GEMMA3_LAYER_COUNT
    row_count_per_layer: int = 24_732
    hidden_width: int = GEMMA3_HIDDEN_WIDTH
    exact_cast_equality_all_layers: bool = True
    contains_raw_tensors: bool = False
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_logits: bool = False
    contains_gradients: bool = False
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self.schema != _INVENTORY_SCHEMA
            or type(self.format_version) is not int
            or self.format_version != _FORMAT_VERSION
            or self.source_role != A_DEVELOPMENT_ROLE
            or self.source_capture_scope != PRODUCTION_FULL_A_CAPTURE_SCOPE
        ):
            raise ValueError("materialization inventory header is invalid")
        if (
            type(self.layer_receipts) is not tuple
            or len(self.layer_receipts) != GEMMA3_LAYER_COUNT
            or any(
                type(value) is not LayerMLPFlowMaterializationReceipt
                for value in self.layer_receipts
            )
            or tuple(value.layer_ordinal for value in self.layer_receipts)
            != tuple(range(GEMMA3_LAYER_COUNT))
        ):
            raise ValueError("materialization inventory must cover layers 0..17")
        for value in self.layer_receipts:
            value.assert_integrity()
        for label in (
            "a_development_partition_receipt_sha256",
            "source_full_A_bridge_receipt_sha256",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "predecessor_preflight_receipt_sha256",
            "predecessor_runtime_qualification_receipt_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "compiled_execution_fingerprint_sha256",
            "tied_readout_binding_receipt_sha256",
            "source_role_panel_binding_sha256",
            "shared_spool_row_catalog_sha256",
            "fit_partition_sha256",
            "validation_partition_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.fit_partition_sha256 == self.validation_partition_sha256:
            raise ValueError("materialization inventory partition hashes collide")
        for receipt in self.layer_receipts:
            if (
                receipt.a_development_partition_receipt_sha256
                != self.a_development_partition_receipt_sha256
                or receipt.source_full_A_bridge_receipt_sha256
                != self.source_full_A_bridge_receipt_sha256
                or receipt.source_role_capture_receipt_sha256
                != self.source_role_capture_receipt_sha256
                or receipt.source_runtime_session_binding_sha256
                != self.source_runtime_session_binding_sha256
                or receipt.predecessor_preflight_receipt_sha256
                != self.predecessor_preflight_receipt_sha256
                or receipt.predecessor_runtime_qualification_receipt_sha256
                != self.predecessor_runtime_qualification_receipt_sha256
                or receipt.source_model_fingerprint_sha256
                != self.source_model_fingerprint_sha256
                or receipt.source_execution_fingerprint_sha256
                != self.source_execution_fingerprint_sha256
                or receipt.compiled_execution_fingerprint_sha256
                != self.compiled_execution_fingerprint_sha256
                or receipt.tied_readout_binding_receipt_sha256
                != self.tied_readout_binding_receipt_sha256
                or receipt.source_role_panel_binding_sha256
                != self.source_role_panel_binding_sha256
                or receipt.source_spool_row_catalog_sha256
                != self.shared_spool_row_catalog_sha256
                or receipt.fit_partition_sha256 != self.fit_partition_sha256
                or receipt.validation_partition_sha256
                != self.validation_partition_sha256
            ):
                raise ValueError("materialization inventory layer lineage differs")
        if (
            type(self.layer_count) is not int
            or self.layer_count != GEMMA3_LAYER_COUNT
            or type(self.row_count_per_layer) is not int
            or self.row_count_per_layer != 24_732
            or type(self.hidden_width) is not int
            or self.hidden_width != GEMMA3_HIDDEN_WIDTH
        ):
            raise ValueError("materialization inventory dimensions drifted")
        fixed_bools = {
            "exact_cast_equality_all_layers": True,
            "contains_raw_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_gradients": False,
        }
        for label, expected in fixed_bools.items():
            value = getattr(self, label)
            _require_exact_bool(value, label=label)
            if value is not expected:
                raise ValueError("materialization inventory safety/proof drifted")
        computed = _digest(_INVENTORY_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="materialization inventory receipt",
        ) != computed:
            raise ValueError("materialization inventory receipt digest drifted")

    @property
    def layer_materialization_receipt_sha256s(self) -> tuple[str, ...]:
        return tuple(value.receipt_sha256 for value in self.layer_receipts)

    @property
    def layer_flow_rows_receipt_sha256s(self) -> tuple[str, ...]:
        return tuple(
            value.flow_rows_receipt_sha256 for value in self.layer_receipts
        )

    @property
    def layer_chart_state_float32_sha256s(self) -> tuple[str, ...]:
        return tuple(
            value.materialized_chart_state_float32_sha256
            for value in self.layer_receipts
        )

    @property
    def layer_teacher_velocity_float32_sha256s(self) -> tuple[str, ...]:
        return tuple(
            value.materialized_teacher_velocity_float32_sha256
            for value in self.layer_receipts
        )

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_role": self.source_role,
            "source_capture_scope": self.source_capture_scope,
            "a_development_partition_receipt_sha256": (
                self.a_development_partition_receipt_sha256
            ),
            "source_full_A_bridge_receipt_sha256": (
                self.source_full_A_bridge_receipt_sha256
            ),
            "source_role_capture_receipt_sha256": (
                self.source_role_capture_receipt_sha256
            ),
            "source_runtime_session_binding_sha256": (
                self.source_runtime_session_binding_sha256
            ),
            "predecessor_preflight_receipt_sha256": (
                self.predecessor_preflight_receipt_sha256
            ),
            "predecessor_runtime_qualification_receipt_sha256": (
                self.predecessor_runtime_qualification_receipt_sha256
            ),
            "source_model_fingerprint_sha256": (
                self.source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "compiled_execution_fingerprint_sha256": (
                self.compiled_execution_fingerprint_sha256
            ),
            "tied_readout_binding_receipt_sha256": (
                self.tied_readout_binding_receipt_sha256
            ),
            "source_role_panel_binding_sha256": (
                self.source_role_panel_binding_sha256
            ),
            "shared_spool_row_catalog_sha256": (
                self.shared_spool_row_catalog_sha256
            ),
            "fit_partition_sha256": self.fit_partition_sha256,
            "validation_partition_sha256": self.validation_partition_sha256,
            "layer_receipts": [
                value.state_dict() for value in self.layer_receipts
            ],
            "layer_materialization_receipt_sha256s": list(
                self.layer_materialization_receipt_sha256s
            ),
            "layer_flow_rows_receipt_sha256s": list(
                self.layer_flow_rows_receipt_sha256s
            ),
            "layer_chart_state_float32_sha256s": list(
                self.layer_chart_state_float32_sha256s
            ),
            "layer_teacher_velocity_float32_sha256s": list(
                self.layer_teacher_velocity_float32_sha256s
            ),
            "layer_count": self.layer_count,
            "row_count_per_layer": self.row_count_per_layer,
            "hidden_width": self.hidden_width,
            "exact_cast_equality_all_layers": (
                self.exact_cast_equality_all_layers
            ),
            "contains_raw_tensors": self.contains_raw_tensors,
            "contains_prompt_text": self.contains_prompt_text,
            "contains_token_ids": self.contains_token_ids,
            "contains_logits": self.contains_logits,
            "contains_gradients": self.contains_gradients,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self.payload_without_sha256(), "receipt_sha256": self.receipt_sha256}

    @property
    def live_materialization_authorized(self) -> bool:
        """True only for the live ordered capture-time builder result."""

        return (
            self._construction_authority is _BUILDER_AUTHORITY
            and _LIVE_MATERIALIZATION_INVENTORIES.get(id(self)) is self
            and all(
                value.live_materialization_authorized
                for value in self.layer_receipts
            )
        )

    def assert_integrity(self) -> None:
        for receipt in self.layer_receipts:
            receipt.assert_integrity()
        if (
            _digest(_INVENTORY_DOMAIN, self.payload_without_sha256())
            != self.receipt_sha256
        ):
            raise ValueError("materialization inventory receipt drifted")

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        *,
        expected_receipt_sha256: str,
    ) -> "LayerMLPFlowMaterializationInventoryReceipt":
        _require_sha256(
            expected_receipt_sha256,
            label="expected materialization inventory receipt",
        )
        if type(state) is not dict:
            raise TypeError(
                "materialization inventory receipt state must be an exact dict"
            )
        expected_fields = {
            "schema",
            "format_version",
            "source_role",
            "source_capture_scope",
            "a_development_partition_receipt_sha256",
            "source_full_A_bridge_receipt_sha256",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "predecessor_preflight_receipt_sha256",
            "predecessor_runtime_qualification_receipt_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "compiled_execution_fingerprint_sha256",
            "tied_readout_binding_receipt_sha256",
            "source_role_panel_binding_sha256",
            "shared_spool_row_catalog_sha256",
            "fit_partition_sha256",
            "validation_partition_sha256",
            "layer_receipts",
            "layer_materialization_receipt_sha256s",
            "layer_flow_rows_receipt_sha256s",
            "layer_chart_state_float32_sha256s",
            "layer_teacher_velocity_float32_sha256s",
            "layer_count",
            "row_count_per_layer",
            "hidden_width",
            "exact_cast_equality_all_layers",
            "contains_raw_tensors",
            "contains_prompt_text",
            "contains_token_ids",
            "contains_logits",
            "contains_gradients",
            "receipt_sha256",
        }
        if set(state) != expected_fields:
            raise ValueError("materialization inventory state fields differ")
        layer_states = state["layer_receipts"]
        if type(layer_states) is not list or len(layer_states) != GEMMA3_LAYER_COUNT:
            raise TypeError("materialization inventory layers must be an 18-row list")
        layers: list[LayerMLPFlowMaterializationReceipt] = []
        for index, layer_state in enumerate(layer_states):
            if type(layer_state) is not dict:
                raise TypeError("materialization layer state must be an exact dict")
            layer_sha = layer_state.get("receipt_sha256")
            _require_sha256(
                layer_sha,
                label=f"layer {index} materialization receipt",
            )
            layers.append(
                LayerMLPFlowMaterializationReceipt.from_state_dict(
                    layer_state,
                    expected_receipt_sha256=layer_sha,
                )
            )
        list_fields = {
            "layer_materialization_receipt_sha256s": tuple(
                value.receipt_sha256 for value in layers
            ),
            "layer_flow_rows_receipt_sha256s": tuple(
                value.flow_rows_receipt_sha256 for value in layers
            ),
            "layer_chart_state_float32_sha256s": tuple(
                value.materialized_chart_state_float32_sha256
                for value in layers
            ),
            "layer_teacher_velocity_float32_sha256s": tuple(
                value.materialized_teacher_velocity_float32_sha256
                for value in layers
            ),
        }
        for name, expected in list_fields.items():
            raw = state[name]
            if type(raw) is not list:
                raise TypeError(f"{name} must be an exact JSON list")
            if tuple(raw) != expected:
                raise ValueError(f"{name} does not recompute from layer receipts")
        arguments = {
            key: value
            for key, value in state.items()
            if key not in {"layer_receipts", *list_fields}
        }
        arguments["layer_receipts"] = tuple(layers)
        receipt = cls(**arguments)
        if receipt.receipt_sha256 != expected_receipt_sha256:
            raise ValueError("materialization inventory trusted hash pin differs")
        return receipt


def build_layer_mlp_flow_materialization_inventory(
    layer_receipts: tuple[LayerMLPFlowMaterializationReceipt, ...],
    *,
    a_development_partition_receipt: ADevelopmentPartitionReceipt,
) -> LayerMLPFlowMaterializationInventoryReceipt:
    """Aggregate the exact ordered 18 per-layer tensor-free receipts."""

    if type(a_development_partition_receipt) is not ADevelopmentPartitionReceipt:
        raise TypeError(
            "a_development_partition_receipt must be the exact typed receipt"
        )
    partition = _strict_replay_partition(a_development_partition_receipt)
    if (
        type(layer_receipts) is not tuple
        or len(layer_receipts) != GEMMA3_LAYER_COUNT
        or any(
            type(value) is not LayerMLPFlowMaterializationReceipt
            for value in layer_receipts
        )
    ):
        raise TypeError("layer_receipts must be 18 exact ordered typed receipts")
    for receipt in layer_receipts:
        receipt.assert_integrity()
        if not receipt.live_materialization_authorized:
            raise ValueError(
                "production inventory requires live builder-authority layer receipts"
            )
    first = layer_receipts[0]
    if (
        first.a_development_partition_receipt_sha256
        != partition.receipt_sha256
        or first.source_full_A_bridge_receipt_sha256
        != partition.source_full_A_bridge_receipt_sha256
        or first.source_role_capture_receipt_sha256
        != partition.source_role_capture_receipt_sha256
        or first.source_runtime_session_binding_sha256
        != partition.source_runtime_session_binding_sha256
        or first.source_role_panel_binding_sha256
        != partition.source_role_panel_binding_sha256
        or first.source_spool_row_catalog_sha256
        != partition.source_shared_spool_row_catalog_sha256
        or first.fit_partition_sha256 != partition.fit_partition_sha256
        or first.validation_partition_sha256
        != partition.validation_partition_sha256
    ):
        raise ValueError("materialization receipts differ from the A partition")
    inventory = LayerMLPFlowMaterializationInventoryReceipt(
        layer_receipts=layer_receipts,
        a_development_partition_receipt_sha256=partition.receipt_sha256,
        source_full_A_bridge_receipt_sha256=(
            partition.source_full_A_bridge_receipt_sha256
        ),
        source_role_capture_receipt_sha256=(
            partition.source_role_capture_receipt_sha256
        ),
        source_runtime_session_binding_sha256=(
            partition.source_runtime_session_binding_sha256
        ),
        predecessor_preflight_receipt_sha256=(
            first.predecessor_preflight_receipt_sha256
        ),
        predecessor_runtime_qualification_receipt_sha256=(
            first.predecessor_runtime_qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            first.source_model_fingerprint_sha256
        ),
        source_execution_fingerprint_sha256=(
            first.source_execution_fingerprint_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            first.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            first.tied_readout_binding_receipt_sha256
        ),
        source_role_panel_binding_sha256=(
            partition.source_role_panel_binding_sha256
        ),
        shared_spool_row_catalog_sha256=(
            partition.source_shared_spool_row_catalog_sha256
        ),
        fit_partition_sha256=partition.fit_partition_sha256,
        validation_partition_sha256=partition.validation_partition_sha256,
    )
    object.__setattr__(
        inventory,
        "_construction_authority",
        _BUILDER_AUTHORITY,
    )
    _LIVE_MATERIALIZATION_INVENTORIES[id(inventory)] = inventory
    return inventory
