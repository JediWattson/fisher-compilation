"""Prompt-free MLP state/velocity rows for the layer-multimode compiler.

The chart-native runtime already captures exactly the two tensors needed by
the first multimode-flow rung for every Gemma MLP mutation:

* ``context_values`` are the output of the MLP input RMSNorm; and
* ``comparison_values`` are the post-output-RMSNorm residual delta.

This module lowers those existing, authenticated parent-mode rows into the
teacher pairs ``(x, v_T)`` used by the local flow compiler.  It neither loads
the model nor opens prompt data.  Attention remains native in this rung.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from weakref import WeakValueDictionary

import torch
from torch import Tensor

from . import gemma3_chart_native_neighbor_experiment as _predecessor_capture
from .gemma3_chart_native_neighbor_experiment import (
    FIT_ROLE as PREDECESSOR_A_ROLE,
    CapturedBatch,
    CapturedExampleRows,
    RoleCaptureReceipt,
)
from .shared_chart_energy_transport import ChartRowKey, row_catalog_sha256
from .streaming_shared_chart_energy_transport import (
    RawModeRowSpool,
    open_raw_mode_row_spool,
)


__all__ = [
    "A_FIT_TARGET_TOKEN_COUNTS",
    "A_VALIDATION_TARGET_TOKEN_COUNTS",
    "A_DEVELOPMENT_ROLE",
    "ADevelopmentPartitionReceipt",
    "B3_DISCOVERY_ROLE",
    "C3_CONFIRMATION_ROLE",
    "D3_FINAL_ROLE",
    "DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE",
    "GEMMA3_HIDDEN_WIDTH",
    "GEMMA3_LAYER_COUNT",
    "PRODUCTION_FULL_A_CAPTURE_SCOPE",
    "FreshRoleCaptureNotAuthorizedError",
    "FullALayerMLPSpoolBridge",
    "FullALayerMLPSpoolBridgeReceipt",
    "LayerMLPFlowSpoolChunk",
    "LayerMLPFlowSpoolInput",
    "LayerMLPFlowCaptureCatalog",
    "LayerMLPFlowRows",
    "build_layer_mlp_flow_rows",
    "build_layer_mlp_flow_rows_from_batch",
    "build_layer_multimode_capture_catalog",
    "build_A_development_partition_receipt",
    "family_example_balanced_row_weights",
    "fixed_A_development_split_masks",
    "open_full_A_layer_mlp_spool_bridge",
]


GEMMA3_LAYER_COUNT = 18
GEMMA3_HIDDEN_WIDTH = 640
A_DEVELOPMENT_ROLE = "A_exposed_development"
B3_DISCOVERY_ROLE = "B3_teacher_forced_layerwise"
C3_CONFIRMATION_ROLE = "C3_full_stack_source_free"
D3_FINAL_ROLE = "D3_final_heldout_validation"
A_FIT_TARGET_TOKEN_COUNTS = (96, 512, 514, 640)
A_VALIDATION_TARGET_TOKEN_COUNTS = (511, 513, 576, 768)
PRODUCTION_A_BATCH_COUNT = 45
PRODUCTION_A_EXAMPLE_COUNT = 48
PRODUCTION_A_ROW_COUNT = 24_732
DEFAULT_MAX_RAW_SPOOL_BYTES = 4 * 1024 * 1024 * 1024

_FLOW_ROWS_DOMAIN = b"fisher_graph:gemma3-layer-multimode:flow-rows:v1\0"
_FLOW_CATALOG_DOMAIN = b"fisher_graph:gemma3-layer-multimode:catalog:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:gemma3-layer-multimode:tensor:v1\0"
_WEIGHT_DOMAIN = b"fisher_graph:gemma3-layer-multimode:weights:v1\0"
_FULL_A_BRIDGE_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:full-A-spool-bridge:v1\0"
)
_FULL_A_BRIDGE_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.full_A_spool_bridge_receipt.v1"
)
_A_PARTITION_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.A_development_partition.v1"
)
_A_PARTITION_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:A-development-partition:v1\0"
)
_A_PARTITION_MASK_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:A-partition-mask:v1\0"
)
_A_PARTITION_EXAMPLE_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:A-partition-examples:v1\0"
)
_A_PARTITION_LAYER_AXIS_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:A-partition-layers:v1\0"
)
_A_FIT_PARTITION_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:A-fit-partition:v1\0"
)
_A_VALIDATION_PARTITION_DOMAIN = (
    b"fisher_graph:gemma3-layer-multimode:A-validation-partition:v1\0"
)
_A_MASK_HASH_ABI = (
    "sha256(domain || partition_utf8 || NUL || uint64_le(row_count) || "
    "uint8_mask_bytes_in_canonical_row_order)"
)
_FULL_A_BRIDGE_BUILDER_AUTHORITY = object()
_FULL_A_SPOOL_INPUT_AUTHORITY = object()
_LIVE_FULL_A_BRIDGES: WeakValueDictionary[int, object] = WeakValueDictionary()
_LIVE_FULL_A_SPOOL_INPUTS: WeakValueDictionary[int, object] = WeakValueDictionary()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXACT_PATH_TYPE = type(Path("."))
DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE = (
    "development_single_CapturedBatch_only"
)
PRODUCTION_FULL_A_CAPTURE_SCOPE = (
    "production_full_A_RoleCaptureReceipt_spools"
)
_DEVELOPMENT_SINGLE_BATCH_SCOPE = DEVELOPMENT_SINGLE_BATCH_CAPTURE_SCOPE
_PRODUCTION_FULL_A_SCOPE = PRODUCTION_FULL_A_CAPTURE_SCOPE
_ROLES = (
    A_DEVELOPMENT_ROLE,
    B3_DISCOVERY_ROLE,
    C3_CONFIRMATION_ROLE,
    D3_FINAL_ROLE,
)
_FRESH_ROLES = (B3_DISCOVERY_ROLE, C3_CONFIRMATION_ROLE, D3_FINAL_ROLE)


class FreshRoleCaptureNotAuthorizedError(RuntimeError):
    """Raised until a frozen, role-bound v3 panel manifest is implemented."""


def _require_currently_authorized_role(role: object) -> str:
    if role in _FRESH_ROLES:
        raise FreshRoleCaptureNotAuthorizedError(
            f"{role} capture is not authorized: no frozen role-bound fresh-"
            "panel manifest and strict batch validator exist"
        )
    if role != A_DEVELOPMENT_ROLE:
        raise ValueError("flow rows role is invalid")
    return A_DEVELOPMENT_ROLE


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


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype not in (torch.float32, torch.float64)
        or not value.is_contiguous()
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("tensor hash input is not canonical")
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(f"{tuple(value.shape)}\0{value.dtype}\0".encode("ascii"))
    digest.update(value.detach().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_capture_tensor(
    value: object,
    *,
    rows: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.shape != (rows, GEMMA3_HIDDEN_WIDTH)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.requires_grad
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be finite detached contiguous CPU float32 "
            f"[{rows},{GEMMA3_HIDDEN_WIDTH}]"
        )
    return value.detach().clone().contiguous()


def _expected_mode_ids() -> tuple[str, ...]:
    return (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )


def _mlp_mode_id(layer_ordinal: int) -> str:
    if type(layer_ordinal) is not int or not 0 <= layer_ordinal < 18:
        raise ValueError("layer_ordinal must be an integer in [0,17]")
    return f"mode.mutation.{2 + 2 * layer_ordinal}"


@dataclass(frozen=True, slots=True)
class LayerMLPFlowRows:
    """One layer's aligned, prompt-free MLP chart state and teacher velocity."""

    role: str
    layer_ordinal: int
    mode_id: str
    family_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    token_counts: tuple[int, ...]
    row_keys: tuple[tuple[str, str, int, int], ...]
    example_row_ranges: tuple[tuple[int, int], ...]
    chart_states: Tensor
    teacher_velocities: Tensor
    predecessor_source_role: str
    source_batch_lexical_nll_receipt_sha256: str | None
    source_batch_execution_receipt_sha256: str | None
    source_capture_receipt_sha256s: tuple[str, ...]
    chart_state_sha256: str
    teacher_velocity_sha256: str
    source_capture_scope: str = _DEVELOPMENT_SINGLE_BATCH_SCOPE
    source_role_capture_receipt_sha256: str | None = None
    source_runtime_session_binding_sha256: str | None = None
    source_role_panel_binding_sha256: str | None = None
    source_spool_artifact_sha256: str | None = None
    source_spool_row_catalog_sha256: str | None = None
    source_spool_tensor_catalog_sha256: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require_currently_authorized_role(self.role)
        if self.predecessor_source_role != PREDECESSOR_A_ROLE:
            raise ValueError("flow rows must retain the predecessor A role witness")
        production_lineage = (
            self.source_role_capture_receipt_sha256,
            self.source_runtime_session_binding_sha256,
            self.source_role_panel_binding_sha256,
            self.source_spool_artifact_sha256,
            self.source_spool_row_catalog_sha256,
            self.source_spool_tensor_catalog_sha256,
        )
        if self.source_capture_scope == _DEVELOPMENT_SINGLE_BATCH_SCOPE:
            _require_sha256(
                self.source_batch_lexical_nll_receipt_sha256,
                label="source batch lexical NLL receipt",
            )
            _require_sha256(
                self.source_batch_execution_receipt_sha256,
                label="source batch execution receipt",
            )
            if any(value is not None for value in production_lineage):
                raise ValueError(
                    "development single-batch rows cannot claim full-A authority"
                )
        elif self.source_capture_scope == _PRODUCTION_FULL_A_SCOPE:
            if (
                self.source_batch_lexical_nll_receipt_sha256 is not None
                or self.source_batch_execution_receipt_sha256 is not None
            ):
                raise ValueError(
                    "production full-A rows cannot claim one source batch"
                )
            for label, value in zip(
                (
                    "source role capture receipt",
                    "source runtime session binding",
                    "source role panel binding",
                    "source spool artifact",
                    "source spool row catalog",
                    "source spool tensor catalog",
                ),
                production_lineage,
                strict=True,
            ):
                _require_sha256(value, label=label)
        else:
            raise ValueError("flow rows source capture scope is invalid")
        expected_mode = _mlp_mode_id(self.layer_ordinal)
        if self.mode_id != expected_mode:
            raise ValueError("flow rows mode/layer identity drifted")
        axes = (
            self.family_ids,
            self.example_ids,
            self.token_counts,
            self.example_row_ranges,
            self.source_capture_receipt_sha256s,
        )
        if any(type(value) is not tuple for value in axes):
            raise TypeError("flow rows axes must be tuples")
        examples = len(self.example_ids)
        if (
            examples == 0
            or any(len(value) != examples for value in axes)
            or len(set(self.example_ids)) != examples
            or any(not isinstance(value, str) or not value for value in self.family_ids)
            or any(
                not isinstance(value, str) or not value
                for value in self.example_ids
            )
            or any(type(value) is not int or value < 2 for value in self.token_counts)
        ):
            raise ValueError("flow rows example axes are invalid")
        for value in self.source_capture_receipt_sha256s:
            _require_sha256(value, label="source capture receipt")
        if type(self.row_keys) is not tuple or not self.row_keys:
            raise ValueError("flow rows require a nonempty row-key tuple")
        cursor = 0
        expected_rows: list[tuple[str, str, int, int]] = []
        for family, example, tokens, bounds in zip(
            self.family_ids,
            self.example_ids,
            self.token_counts,
            self.example_row_ranges,
            strict=True,
        ):
            expected_bounds = (cursor, cursor + tokens - 1)
            if bounds != expected_bounds:
                raise ValueError("flow rows example ranges are not canonical")
            expected_rows.extend(
                (family, example, query, query + 1)
                for query in range(tokens - 1)
            )
            cursor = expected_bounds[1]
        if self.row_keys != tuple(expected_rows):
            raise ValueError("flow rows q/t keys are not canonical")
        states = _canonical_capture_tensor(
            self.chart_states,
            rows=len(self.row_keys),
            label="chart_states",
        )
        velocities = _canonical_capture_tensor(
            self.teacher_velocities,
            rows=len(self.row_keys),
            label="teacher_velocities",
        )
        object.__setattr__(self, "chart_states", states)
        object.__setattr__(self, "teacher_velocities", velocities)
        state_sha = _tensor_sha256(states)
        velocity_sha = _tensor_sha256(velocities)
        if self.chart_state_sha256 != state_sha:
            raise ValueError("chart-state tensor hash drifted")
        if self.teacher_velocity_sha256 != velocity_sha:
            raise ValueError("teacher-velocity tensor hash drifted")
        expected = _digest(_FLOW_ROWS_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="flow rows receipt",
        ) != expected:
            raise ValueError("flow rows receipt digest drifted")

    @property
    def row_count(self) -> int:
        return len(self.row_keys)

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "role": self.role,
            "layer_ordinal": self.layer_ordinal,
            "mode_id": self.mode_id,
            "family_ids": list(self.family_ids),
            "example_ids": list(self.example_ids),
            "token_counts": list(self.token_counts),
            "row_keys": [list(row) for row in self.row_keys],
            "example_row_ranges": [list(row) for row in self.example_row_ranges],
            "source_capture_receipt_sha256s": list(
                self.source_capture_receipt_sha256s
            ),
            "predecessor_source_role": self.predecessor_source_role,
            "source_batch_lexical_nll_receipt_sha256": (
                self.source_batch_lexical_nll_receipt_sha256
            ),
            "source_batch_execution_receipt_sha256": (
                self.source_batch_execution_receipt_sha256
            ),
            "source_capture_scope": self.source_capture_scope,
            "source_role_capture_receipt_sha256": (
                self.source_role_capture_receipt_sha256
            ),
            "source_runtime_session_binding_sha256": (
                self.source_runtime_session_binding_sha256
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
            "chart_state_sha256": self.chart_state_sha256,
            "teacher_velocity_sha256": self.teacher_velocity_sha256,
            "row_count": self.row_count,
            "hidden_width": GEMMA3_HIDDEN_WIDTH,
            "chart_state_semantics": "exact_mlp_input_rmsnorm_output",
            "teacher_velocity_semantics": (
                "exact_post_output_rmsnorm_mlp_residual_delta"
            ),
            "attention_replaced": False,
            "production_full_A_capture_authorized": (
                self.source_capture_scope == _PRODUCTION_FULL_A_SCOPE
            ),
            "single_batch_development_only": (
                self.source_capture_scope == _DEVELOPMENT_SINGLE_BATCH_SCOPE
            ),
            "captured_residual_carrier_h": False,
            "captured_output_or_h_plus_v_closure": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_gradient": False,
            "contains_logits": False,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class LayerMLPFlowCaptureCatalog:
    """Exact 18-layer axis for one role's multimode teacher rows."""

    role: str
    layers: tuple[LayerMLPFlowRows, ...]
    shared_row_catalog_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _require_currently_authorized_role(self.role)
        if (
            type(self.layers) is not tuple
            or len(self.layers) != GEMMA3_LAYER_COUNT
            or tuple(row.layer_ordinal for row in self.layers)
            != tuple(range(GEMMA3_LAYER_COUNT))
            or any(row.role != self.role for row in self.layers)
        ):
            raise ValueError("capture catalog must cover ordered layers 0..17")
        first = self.layers[0]
        if first.source_capture_scope != _DEVELOPMENT_SINGLE_BATCH_SCOPE:
            raise ValueError(
                "in-memory 18-layer catalogs are development-only; production "
                "must use the full-A spool bridge one layer at a time"
            )
        for row in self.layers[1:]:
            if (
                row.row_keys != first.row_keys
                or row.family_ids != first.family_ids
                or row.example_ids != first.example_ids
                or row.token_counts != first.token_counts
                or row.example_row_ranges != first.example_row_ranges
                or row.source_capture_receipt_sha256s
                != first.source_capture_receipt_sha256s
                or row.predecessor_source_role
                != first.predecessor_source_role
                or row.source_batch_lexical_nll_receipt_sha256
                != first.source_batch_lexical_nll_receipt_sha256
                or row.source_batch_execution_receipt_sha256
                != first.source_batch_execution_receipt_sha256
                or row.source_capture_scope != first.source_capture_scope
                or row.source_role_capture_receipt_sha256
                != first.source_role_capture_receipt_sha256
                or row.source_runtime_session_binding_sha256
                != first.source_runtime_session_binding_sha256
                or row.source_role_panel_binding_sha256
                != first.source_role_panel_binding_sha256
            ):
                raise ValueError("capture catalog layer axes are not aligned")
        expected_row_catalog = _digest(
            _FLOW_CATALOG_DOMAIN,
            {
                "role": self.role,
                "row_keys": [list(value) for value in first.row_keys],
                "source_capture_receipt_sha256s": list(
                    first.source_capture_receipt_sha256s
                ),
                "source_batch_lexical_nll_receipt_sha256": (
                    first.source_batch_lexical_nll_receipt_sha256
                ),
                "source_batch_execution_receipt_sha256": (
                    first.source_batch_execution_receipt_sha256
                ),
            },
        )
        if self.shared_row_catalog_sha256 != expected_row_catalog:
            raise ValueError("capture catalog shared row digest drifted")
        expected = _digest(_FLOW_CATALOG_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="capture catalog receipt",
        ) != expected:
            raise ValueError("capture catalog receipt digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "role": self.role,
            "layer_receipt_sha256s": [
                row.receipt_sha256 for row in self.layers
            ],
            "layer_mode_ids": [row.mode_id for row in self.layers],
            "shared_row_catalog_sha256": self.shared_row_catalog_sha256,
            "layer_count": GEMMA3_LAYER_COUNT,
            "attention_replaced": False,
            "source_runtime_parent_mode_count": 38,
            "source_capture_scope": self.layers[0].source_capture_scope,
            "production_full_A_capture_authorized": False,
            "single_batch_development_only": True,
            "captured_residual_carrier_h": False,
            "captured_output_or_h_plus_v_closure": False,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class FullALayerMLPSpoolBridgeReceipt:
    """Tensor-free lineage for the complete production A MLP spool bridge."""

    source_role_capture_receipt_sha256: str
    source_runtime_session_binding_sha256: str
    source_role_panel_binding_sha256: str
    source_batch_axis_catalog_sha256: str
    source_ordered_objective_row_catalog_sha256: str
    source_score_objective_receipt_sha256: str
    shared_spool_row_catalog_sha256: str
    layer_sources: tuple[tuple[int, str, str, str, str, str, str], ...]
    batch_count: int
    example_count: int
    row_count: int
    hidden_width: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        self._validate_structure()
        computed = _digest(_FULL_A_BRIDGE_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="full-A bridge receipt",
        ) != computed:
            raise ValueError("full-A bridge receipt digest drifted")

    def _validate_structure(self) -> None:
        for label in (
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "source_role_panel_binding_sha256",
            "source_batch_axis_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
            "source_score_objective_receipt_sha256",
            "shared_spool_row_catalog_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if (
            type(self.batch_count) is not int
            or self.batch_count != PRODUCTION_A_BATCH_COUNT
            or type(self.example_count) is not int
            or self.example_count != PRODUCTION_A_EXAMPLE_COUNT
            or type(self.row_count) is not int
            or self.row_count != PRODUCTION_A_ROW_COUNT
            or type(self.hidden_width) is not int
            or self.hidden_width != GEMMA3_HIDDEN_WIDTH
        ):
            raise ValueError("full-A bridge production inventory drifted")
        if (
            type(self.layer_sources) is not tuple
            or len(self.layer_sources) != GEMMA3_LAYER_COUNT
        ):
            raise ValueError("full-A bridge must bind exactly 18 MLP spools")
        for layer, source in enumerate(self.layer_sources):
            if type(source) is not tuple or len(source) != 7:
                raise ValueError("full-A bridge layer source row is invalid")
            (
                stored_layer,
                mode_id,
                artifact_sha256,
                row_catalog_digest,
                tensor_catalog_digest,
                context_semantic_sha256,
                comparison_semantic_sha256,
            ) = source
            if stored_layer != layer or type(stored_layer) is not int:
                raise ValueError("full-A bridge layer source order drifted")
            if type(mode_id) is not str or mode_id != _mlp_mode_id(layer):
                raise ValueError("full-A bridge even MLP mode inventory drifted")
            for label, digest in (
                ("spool artifact", artifact_sha256),
                ("spool row catalog", row_catalog_digest),
                ("spool tensor catalog", tensor_catalog_digest),
                ("spool context semantic tensor", context_semantic_sha256),
                ("spool comparison semantic tensor", comparison_semantic_sha256),
            ):
                _require_sha256(digest, label=label)
            if row_catalog_digest != self.shared_spool_row_catalog_sha256:
                raise ValueError("full-A bridge layer row catalog drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": _FULL_A_BRIDGE_SCHEMA,
            "format_version": 1,
            "source_role": PREDECESSOR_A_ROLE,
            "v3_role": A_DEVELOPMENT_ROLE,
            "source_role_capture_receipt_sha256": (
                self.source_role_capture_receipt_sha256
            ),
            "source_runtime_session_binding_sha256": (
                self.source_runtime_session_binding_sha256
            ),
            "source_role_panel_binding_sha256": (
                self.source_role_panel_binding_sha256
            ),
            "source_batch_axis_catalog_sha256": (
                self.source_batch_axis_catalog_sha256
            ),
            "source_ordered_objective_row_catalog_sha256": (
                self.source_ordered_objective_row_catalog_sha256
            ),
            "source_score_objective_receipt_sha256": (
                self.source_score_objective_receipt_sha256
            ),
            "shared_spool_row_catalog_sha256": (
                self.shared_spool_row_catalog_sha256
            ),
            "layer_sources": [list(value) for value in self.layer_sources],
            "batch_count": self.batch_count,
            "example_count": self.example_count,
            "row_count": self.row_count,
            "hidden_width": self.hidden_width,
            "source_parent_mode_count": 38,
            "selected_even_mlp_spool_count": GEMMA3_LAYER_COUNT,
            "chart_state_semantics": "exact_mlp_input_rmsnorm_output",
            "teacher_velocity_semantics": (
                "exact_post_output_rmsnorm_mlp_residual_delta"
            ),
            "captures_normalized_x": True,
            "captures_teacher_velocity_v_T": True,
            "captures_residual_carrier_h": False,
            "captures_output_or_h_plus_v_closure": False,
            "unchanged_decoder_layer_owns_residual_add": True,
            "attention_replaced": False,
            "production_full_A_capture_authorized": True,
            "single_batch_development_only": False,
            "contains_raw_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_gradients": False,
        }

    def metadata(self) -> dict[str, object]:
        self.assert_integrity()
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe, tensor-free receipt state."""

        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        *,
        expected_receipt_sha256: str,
    ) -> "FullALayerMLPSpoolBridgeReceipt":
        """Strictly replay a receipt against a separately trusted hash pin."""

        _require_sha256(
            expected_receipt_sha256,
            label="expected full-A bridge receipt",
        )
        if type(state) is not dict:
            raise TypeError("full-A bridge receipt state must be an exact dict")
        expected_fields = {
            "schema",
            "format_version",
            "source_role",
            "v3_role",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "source_role_panel_binding_sha256",
            "source_batch_axis_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
            "source_score_objective_receipt_sha256",
            "shared_spool_row_catalog_sha256",
            "layer_sources",
            "batch_count",
            "example_count",
            "row_count",
            "hidden_width",
            "source_parent_mode_count",
            "selected_even_mlp_spool_count",
            "chart_state_semantics",
            "teacher_velocity_semantics",
            "captures_normalized_x",
            "captures_teacher_velocity_v_T",
            "captures_residual_carrier_h",
            "captures_output_or_h_plus_v_closure",
            "unchanged_decoder_layer_owns_residual_add",
            "attention_replaced",
            "production_full_A_capture_authorized",
            "single_batch_development_only",
            "contains_raw_tensors",
            "contains_prompt_text",
            "contains_token_ids",
            "contains_gradients",
            "receipt_sha256",
        }
        if set(state) != expected_fields:
            raise ValueError("full-A bridge receipt state fields differ")
        fixed_values: dict[str, object] = {
            "schema": _FULL_A_BRIDGE_SCHEMA,
            "format_version": 1,
            "source_role": PREDECESSOR_A_ROLE,
            "v3_role": A_DEVELOPMENT_ROLE,
            "batch_count": PRODUCTION_A_BATCH_COUNT,
            "example_count": PRODUCTION_A_EXAMPLE_COUNT,
            "row_count": PRODUCTION_A_ROW_COUNT,
            "hidden_width": GEMMA3_HIDDEN_WIDTH,
            "source_parent_mode_count": 38,
            "selected_even_mlp_spool_count": GEMMA3_LAYER_COUNT,
            "chart_state_semantics": "exact_mlp_input_rmsnorm_output",
            "teacher_velocity_semantics": (
                "exact_post_output_rmsnorm_mlp_residual_delta"
            ),
            "captures_normalized_x": True,
            "captures_teacher_velocity_v_T": True,
            "captures_residual_carrier_h": False,
            "captures_output_or_h_plus_v_closure": False,
            "unchanged_decoder_layer_owns_residual_add": True,
            "attention_replaced": False,
            "production_full_A_capture_authorized": True,
            "single_batch_development_only": False,
            "contains_raw_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_gradients": False,
        }
        for name, expected in fixed_values.items():
            value = state[name]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"full-A bridge receipt fixed field {name} differs"
                )
        for name in (
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "source_role_panel_binding_sha256",
            "source_batch_axis_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
            "source_score_objective_receipt_sha256",
            "shared_spool_row_catalog_sha256",
            "receipt_sha256",
        ):
            _require_sha256(state[name], label=f"serialized {name}")
        if state["receipt_sha256"] != expected_receipt_sha256:
            raise ValueError("full-A bridge receipt differs from trusted hash pin")
        raw_sources = state["layer_sources"]
        if type(raw_sources) is not list or len(raw_sources) != GEMMA3_LAYER_COUNT:
            raise TypeError("serialized layer_sources must be an exact 18-row list")
        sources: list[tuple[int, str, str, str, str, str, str]] = []
        for layer, raw_source in enumerate(raw_sources):
            if type(raw_source) is not list or len(raw_source) != 7:
                raise TypeError(
                    "serialized layer source rows must be exact seven-item lists"
                )
            if type(raw_source[0]) is not int or raw_source[0] != layer:
                raise ValueError("serialized layer source order differs")
            if any(type(value) is not str for value in raw_source[1:]):
                raise TypeError("serialized layer source hashes must be strings")
            sources.append(tuple(raw_source))  # type: ignore[arg-type]
        return cls(
            source_role_capture_receipt_sha256=state[
                "source_role_capture_receipt_sha256"
            ],
            source_runtime_session_binding_sha256=state[
                "source_runtime_session_binding_sha256"
            ],
            source_role_panel_binding_sha256=state[
                "source_role_panel_binding_sha256"
            ],
            source_batch_axis_catalog_sha256=state[
                "source_batch_axis_catalog_sha256"
            ],
            source_ordered_objective_row_catalog_sha256=state[
                "source_ordered_objective_row_catalog_sha256"
            ],
            source_score_objective_receipt_sha256=state[
                "source_score_objective_receipt_sha256"
            ],
            shared_spool_row_catalog_sha256=state[
                "shared_spool_row_catalog_sha256"
            ],
            layer_sources=tuple(sources),
            batch_count=state["batch_count"],
            example_count=state["example_count"],
            row_count=state["row_count"],
            hidden_width=state["hidden_width"],
            receipt_sha256=state["receipt_sha256"],
        )  # type: ignore[arg-type]

    def assert_integrity(self) -> None:
        self._validate_structure()
        _require_sha256(
            self.receipt_sha256,
            label="full-A bridge receipt",
        )
        if (
            _digest(_FULL_A_BRIDGE_DOMAIN, self.payload_without_sha256())
            != self.receipt_sha256
        ):
            raise ValueError("full-A bridge receipt digest drifted")


def _A_partition_mask_sha256(partition: str, mask_bytes: bytes) -> str:
    if partition not in ("fit", "validation") or type(mask_bytes) is not bytes:
        raise TypeError("A partition mask digest input is invalid")
    digest = hashlib.sha256()
    digest.update(_A_PARTITION_MASK_DOMAIN)
    digest.update(partition.encode("ascii"))
    digest.update(b"\0")
    digest.update(len(mask_bytes).to_bytes(8, byteorder="little", signed=False))
    digest.update(mask_bytes)
    return digest.hexdigest()


def _A_partition_layer_axis_sha256(
    axis: tuple[tuple[int, str], ...],
) -> str:
    return _digest(
        _A_PARTITION_LAYER_AXIS_DOMAIN,
        {"canonical_layer_mode_axis": [list(value) for value in axis]},
    )


@dataclass(frozen=True, slots=True)
class ADevelopmentPartitionReceipt:
    """Shared tensor-free A fit/validation partition for every MLP layer."""

    source_full_A_bridge_receipt_sha256: str
    source_role_capture_receipt_sha256: str
    source_runtime_session_binding_sha256: str
    source_role_panel_binding_sha256: str
    source_batch_axis_catalog_sha256: str
    source_shared_spool_row_catalog_sha256: str
    source_ordered_objective_row_catalog_sha256: str
    source_score_objective_receipt_sha256: str
    canonical_layer_mode_axis: tuple[tuple[int, str], ...]
    canonical_layer_mode_axis_sha256: str
    fit_target_token_counts: tuple[int, ...]
    validation_target_token_counts: tuple[int, ...]
    mask_hash_abi: str
    fit_mask_sha256: str
    validation_mask_sha256: str
    fit_example_catalog_sha256: str
    validation_example_catalog_sha256: str
    fit_row_count: int
    validation_row_count: int
    row_count: int
    layer_count: int
    fit_partition_sha256: str = ""
    validation_partition_sha256: str = ""
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        self._validate_structure()
        expected_fit = self._derived_partition_sha256("fit")
        expected_validation = self._derived_partition_sha256("validation")
        if self.fit_partition_sha256 == "":
            object.__setattr__(self, "fit_partition_sha256", expected_fit)
        elif _require_sha256(
            self.fit_partition_sha256,
            label="fit partition",
        ) != expected_fit:
            raise ValueError("fit partition digest drifted")
        if self.validation_partition_sha256 == "":
            object.__setattr__(
                self,
                "validation_partition_sha256",
                expected_validation,
            )
        elif _require_sha256(
            self.validation_partition_sha256,
            label="validation partition",
        ) != expected_validation:
            raise ValueError("validation partition digest drifted")
        if self.fit_partition_sha256 == self.validation_partition_sha256:
            raise ValueError("A fit and validation partition digests collide")
        computed = _digest(_A_PARTITION_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256,
            label="A development partition receipt",
        ) != computed:
            raise ValueError("A development partition receipt digest drifted")

    def _validate_structure(self) -> None:
        for name in (
            "source_full_A_bridge_receipt_sha256",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "source_role_panel_binding_sha256",
            "source_batch_axis_catalog_sha256",
            "source_shared_spool_row_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
            "source_score_objective_receipt_sha256",
            "canonical_layer_mode_axis_sha256",
            "fit_mask_sha256",
            "validation_mask_sha256",
            "fit_example_catalog_sha256",
            "validation_example_catalog_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        expected_axis = tuple(
            (layer, _mlp_mode_id(layer))
            for layer in range(GEMMA3_LAYER_COUNT)
        )
        if (
            type(self.canonical_layer_mode_axis) is not tuple
            or any(
                type(value) is not tuple
                or len(value) != 2
                or type(value[0]) is not int
                or type(value[1]) is not str
                for value in self.canonical_layer_mode_axis
            )
            or self.canonical_layer_mode_axis != expected_axis
        ):
            raise ValueError("A partition canonical layer/mode axis drifted")
        if self.canonical_layer_mode_axis_sha256 != (
            _A_partition_layer_axis_sha256(self.canonical_layer_mode_axis)
        ):
            raise ValueError("A partition layer/mode axis hash drifted")
        if (
            type(self.fit_target_token_counts) is not tuple
            or self.fit_target_token_counts != A_FIT_TARGET_TOKEN_COUNTS
            or type(self.validation_target_token_counts) is not tuple
            or self.validation_target_token_counts
            != A_VALIDATION_TARGET_TOKEN_COUNTS
            or any(
                type(value) is not int
                for value in (
                    *self.fit_target_token_counts,
                    *self.validation_target_token_counts,
                )
            )
        ):
            raise ValueError("A partition target-token split drifted")
        if type(self.mask_hash_abi) is not str or self.mask_hash_abi != (
            _A_MASK_HASH_ABI
        ):
            raise ValueError("A partition mask hash ABI drifted")
        if (
            type(self.fit_row_count) is not int
            or self.fit_row_count != 10_548
            or type(self.validation_row_count) is not int
            or self.validation_row_count != 14_184
            or type(self.row_count) is not int
            or self.row_count != PRODUCTION_A_ROW_COUNT
            or self.fit_row_count + self.validation_row_count != self.row_count
            or type(self.layer_count) is not int
            or self.layer_count != GEMMA3_LAYER_COUNT
            or self.fit_mask_sha256 == self.validation_mask_sha256
            or (
                self.fit_partition_sha256 != ""
                and self.validation_partition_sha256 != ""
                and self.fit_partition_sha256
                == self.validation_partition_sha256
            )
        ):
            raise ValueError("A partition production counts/closure drifted")

    def _derived_partition_sha256(self, partition: str) -> str:
        if partition == "fit":
            domain = _A_FIT_PARTITION_DOMAIN
            mask_sha256 = self.fit_mask_sha256
            example_catalog_sha256 = self.fit_example_catalog_sha256
            target_token_counts = self.fit_target_token_counts
            row_count = self.fit_row_count
        elif partition == "validation":
            domain = _A_VALIDATION_PARTITION_DOMAIN
            mask_sha256 = self.validation_mask_sha256
            example_catalog_sha256 = self.validation_example_catalog_sha256
            target_token_counts = self.validation_target_token_counts
            row_count = self.validation_row_count
        else:
            raise ValueError("A partition identity is invalid")
        return _digest(
            domain,
            {
                "source_full_A_bridge_receipt_sha256": (
                    self.source_full_A_bridge_receipt_sha256
                ),
                "source_role_capture_receipt_sha256": (
                    self.source_role_capture_receipt_sha256
                ),
                "source_shared_spool_row_catalog_sha256": (
                    self.source_shared_spool_row_catalog_sha256
                ),
                "canonical_layer_mode_axis_sha256": (
                    self.canonical_layer_mode_axis_sha256
                ),
                "example_catalog_sha256": example_catalog_sha256,
                "mask_hash_abi": self.mask_hash_abi,
                "mask_sha256": mask_sha256,
                "partition": partition,
                "row_count": row_count,
                "target_token_counts": list(target_token_counts),
            },
        )

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": _A_PARTITION_SCHEMA,
            "format_version": 1,
            "source_role": PREDECESSOR_A_ROLE,
            "v3_role": A_DEVELOPMENT_ROLE,
            "source_capture_scope": PRODUCTION_FULL_A_CAPTURE_SCOPE,
            "source_full_A_bridge_receipt_sha256": (
                self.source_full_A_bridge_receipt_sha256
            ),
            "source_role_capture_receipt_sha256": (
                self.source_role_capture_receipt_sha256
            ),
            "source_runtime_session_binding_sha256": (
                self.source_runtime_session_binding_sha256
            ),
            "source_role_panel_binding_sha256": (
                self.source_role_panel_binding_sha256
            ),
            "source_batch_axis_catalog_sha256": (
                self.source_batch_axis_catalog_sha256
            ),
            "source_shared_spool_row_catalog_sha256": (
                self.source_shared_spool_row_catalog_sha256
            ),
            "source_ordered_objective_row_catalog_sha256": (
                self.source_ordered_objective_row_catalog_sha256
            ),
            "source_score_objective_receipt_sha256": (
                self.source_score_objective_receipt_sha256
            ),
            "canonical_layer_mode_axis": [
                list(value) for value in self.canonical_layer_mode_axis
            ],
            "canonical_layer_mode_axis_sha256": (
                self.canonical_layer_mode_axis_sha256
            ),
            "fit_target_token_counts": list(self.fit_target_token_counts),
            "validation_target_token_counts": list(
                self.validation_target_token_counts
            ),
            "mask_hash_abi": self.mask_hash_abi,
            "fit_mask_sha256": self.fit_mask_sha256,
            "validation_mask_sha256": self.validation_mask_sha256,
            "fit_example_catalog_sha256": self.fit_example_catalog_sha256,
            "validation_example_catalog_sha256": (
                self.validation_example_catalog_sha256
            ),
            "fit_row_count": self.fit_row_count,
            "validation_row_count": self.validation_row_count,
            "row_count": self.row_count,
            "layer_count": self.layer_count,
            "fit_partition_sha256": self.fit_partition_sha256,
            "validation_partition_sha256": self.validation_partition_sha256,
            "fit_validation_masks_disjoint": True,
            "fit_validation_masks_exhaustive": True,
            "layer_independent_partition": True,
            "masks_regenerated_without_tensor_reads": True,
            "contains_raw_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_gradients": False,
        }

    def metadata(self) -> dict[str, object]:
        self.assert_integrity()
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        *,
        expected_receipt_sha256: str,
    ) -> "ADevelopmentPartitionReceipt":
        _require_sha256(
            expected_receipt_sha256,
            label="expected A development partition receipt",
        )
        if type(state) is not dict:
            raise TypeError("A partition receipt state must be an exact dict")
        expected_fields = {
            "schema",
            "format_version",
            "source_role",
            "v3_role",
            "source_capture_scope",
            "source_full_A_bridge_receipt_sha256",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "source_role_panel_binding_sha256",
            "source_batch_axis_catalog_sha256",
            "source_shared_spool_row_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
            "source_score_objective_receipt_sha256",
            "canonical_layer_mode_axis",
            "canonical_layer_mode_axis_sha256",
            "fit_target_token_counts",
            "validation_target_token_counts",
            "mask_hash_abi",
            "fit_mask_sha256",
            "validation_mask_sha256",
            "fit_example_catalog_sha256",
            "validation_example_catalog_sha256",
            "fit_row_count",
            "validation_row_count",
            "row_count",
            "layer_count",
            "fit_partition_sha256",
            "validation_partition_sha256",
            "fit_validation_masks_disjoint",
            "fit_validation_masks_exhaustive",
            "layer_independent_partition",
            "masks_regenerated_without_tensor_reads",
            "contains_raw_tensors",
            "contains_prompt_text",
            "contains_token_ids",
            "contains_gradients",
            "receipt_sha256",
        }
        if set(state) != expected_fields:
            raise ValueError("A partition receipt state fields differ")
        fixed_values: dict[str, object] = {
            "schema": _A_PARTITION_SCHEMA,
            "format_version": 1,
            "source_role": PREDECESSOR_A_ROLE,
            "v3_role": A_DEVELOPMENT_ROLE,
            "source_capture_scope": PRODUCTION_FULL_A_CAPTURE_SCOPE,
            "mask_hash_abi": _A_MASK_HASH_ABI,
            "fit_row_count": 10_548,
            "validation_row_count": 14_184,
            "row_count": PRODUCTION_A_ROW_COUNT,
            "layer_count": GEMMA3_LAYER_COUNT,
            "fit_validation_masks_disjoint": True,
            "fit_validation_masks_exhaustive": True,
            "layer_independent_partition": True,
            "masks_regenerated_without_tensor_reads": True,
            "contains_raw_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_gradients": False,
        }
        for name, expected in fixed_values.items():
            value = state[name]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"A partition fixed field {name} differs")
        for name in (
            "source_full_A_bridge_receipt_sha256",
            "source_role_capture_receipt_sha256",
            "source_runtime_session_binding_sha256",
            "source_role_panel_binding_sha256",
            "source_batch_axis_catalog_sha256",
            "source_shared_spool_row_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
            "source_score_objective_receipt_sha256",
            "canonical_layer_mode_axis_sha256",
            "fit_mask_sha256",
            "validation_mask_sha256",
            "fit_example_catalog_sha256",
            "validation_example_catalog_sha256",
            "fit_partition_sha256",
            "validation_partition_sha256",
            "receipt_sha256",
        ):
            _require_sha256(state[name], label=f"serialized {name}")
        if state["receipt_sha256"] != expected_receipt_sha256:
            raise ValueError("A partition receipt differs from trusted hash pin")
        raw_axis = state["canonical_layer_mode_axis"]
        if type(raw_axis) is not list or len(raw_axis) != GEMMA3_LAYER_COUNT:
            raise TypeError("serialized canonical layer axis must be a list")
        axis: list[tuple[int, str]] = []
        for raw_value in raw_axis:
            if (
                type(raw_value) is not list
                or len(raw_value) != 2
                or type(raw_value[0]) is not int
                or type(raw_value[1]) is not str
            ):
                raise TypeError("serialized layer axis rows must be list pairs")
            axis.append((raw_value[0], raw_value[1]))
        for name, expected in (
            ("fit_target_token_counts", A_FIT_TARGET_TOKEN_COUNTS),
            (
                "validation_target_token_counts",
                A_VALIDATION_TARGET_TOKEN_COUNTS,
            ),
        ):
            raw_counts = state[name]
            if (
                type(raw_counts) is not list
                or any(type(value) is not int for value in raw_counts)
                or tuple(raw_counts) != expected
            ):
                raise TypeError(f"serialized {name} must be the exact JSON list")
        return cls(
            source_full_A_bridge_receipt_sha256=state[
                "source_full_A_bridge_receipt_sha256"
            ],
            source_role_capture_receipt_sha256=state[
                "source_role_capture_receipt_sha256"
            ],
            source_runtime_session_binding_sha256=state[
                "source_runtime_session_binding_sha256"
            ],
            source_role_panel_binding_sha256=state[
                "source_role_panel_binding_sha256"
            ],
            source_batch_axis_catalog_sha256=state[
                "source_batch_axis_catalog_sha256"
            ],
            source_shared_spool_row_catalog_sha256=state[
                "source_shared_spool_row_catalog_sha256"
            ],
            source_ordered_objective_row_catalog_sha256=state[
                "source_ordered_objective_row_catalog_sha256"
            ],
            source_score_objective_receipt_sha256=state[
                "source_score_objective_receipt_sha256"
            ],
            canonical_layer_mode_axis=tuple(axis),
            canonical_layer_mode_axis_sha256=state[
                "canonical_layer_mode_axis_sha256"
            ],
            fit_target_token_counts=tuple(state["fit_target_token_counts"]),
            validation_target_token_counts=tuple(
                state["validation_target_token_counts"]
            ),
            mask_hash_abi=state["mask_hash_abi"],
            fit_mask_sha256=state["fit_mask_sha256"],
            validation_mask_sha256=state["validation_mask_sha256"],
            fit_example_catalog_sha256=state[
                "fit_example_catalog_sha256"
            ],
            validation_example_catalog_sha256=state[
                "validation_example_catalog_sha256"
            ],
            fit_row_count=state["fit_row_count"],
            validation_row_count=state["validation_row_count"],
            row_count=state["row_count"],
            layer_count=state["layer_count"],
            fit_partition_sha256=state["fit_partition_sha256"],
            validation_partition_sha256=state[
                "validation_partition_sha256"
            ],
            receipt_sha256=state["receipt_sha256"],
        )  # type: ignore[arg-type]

    def assert_integrity(self) -> None:
        self._validate_structure()
        if (
            _require_sha256(
                self.fit_partition_sha256,
                label="fit partition",
            )
            != self._derived_partition_sha256("fit")
            or _require_sha256(
                self.validation_partition_sha256,
                label="validation partition",
            )
            != self._derived_partition_sha256("validation")
        ):
            raise ValueError("A derived partition digest drifted")
        if (
            _require_sha256(
                self.receipt_sha256,
                label="A development partition receipt",
            )
            != _digest(_A_PARTITION_DOMAIN, self.payload_without_sha256())
        ):
            raise ValueError("A development partition receipt digest drifted")


def build_A_development_partition_receipt(
    *,
    bridge_receipt: FullALayerMLPSpoolBridgeReceipt,
    role_manifest: Mapping[str, object],
    expected_bridge_receipt_sha256: str,
) -> ADevelopmentPartitionReceipt:
    """Regenerate the shared A masks from scalar manifest axes only."""

    if type(bridge_receipt) is not FullALayerMLPSpoolBridgeReceipt:
        raise TypeError("A partition requires the exact full-A bridge receipt")
    bridge_receipt.assert_integrity()
    _require_sha256(
        expected_bridge_receipt_sha256,
        label="expected full-A bridge receipt",
    )
    if bridge_receipt.receipt_sha256 != expected_bridge_receipt_sha256:
        raise ValueError("full-A bridge receipt differs from trusted hash pin")
    if not isinstance(role_manifest, Mapping):
        raise TypeError("role_manifest must be the frozen A role mapping")
    if (
        type(role_manifest.get("role")) is not str
        or role_manifest.get("role") != PREDECESSOR_A_ROLE
    ):
        raise ValueError("A partition requires the frozen A role manifest")
    for manifest_name, receipt_name in (
        (
            "ordered_objective_row_catalog_sha256",
            "source_ordered_objective_row_catalog_sha256",
        ),
        (
            "score_objective_receipt_sha256",
            "source_score_objective_receipt_sha256",
        ),
    ):
        manifest_value = role_manifest.get(manifest_name)
        if type(manifest_value) is not str:
            raise TypeError("A partition manifest lineage must be exact strings")
        _require_sha256(manifest_value, label=f"manifest {manifest_name}")
        if manifest_value != getattr(bridge_receipt, receipt_name):
            raise ValueError("A partition manifest/bridge lineage drifted")
    raw_members = role_manifest.get("ordered_members")
    if type(raw_members) is not list or len(raw_members) != (
        PRODUCTION_A_EXAMPLE_COUNT
    ):
        raise TypeError("A partition ordered_members must be the exact 48-row list")

    row_keys: list[ChartRowKey] = []
    fit_mask = bytearray(PRODUCTION_A_ROW_COUNT)
    validation_mask = bytearray(PRODUCTION_A_ROW_COUNT)
    fit_examples: list[list[object]] = []
    validation_examples: list[list[object]] = []
    by_family: dict[str, set[int]] = {}
    example_ids: set[str] = set()
    cursor = 0
    for member in raw_members:
        if type(member) is not dict:
            raise TypeError("A partition manifest members must be exact dicts")
        family = member.get("family_id")
        example = member.get("example_id")
        tokens = member.get("target_token_count")
        if (
            type(family) is not str
            or not family
            or type(example) is not str
            or not example
            or type(tokens) is not int
            or tokens < 2
            or example in example_ids
        ):
            raise ValueError("A partition manifest member axis drifted")
        example_ids.add(example)
        lengths = by_family.setdefault(family, set())
        if tokens in lengths:
            raise ValueError("A partition repeats a target length in one family")
        lengths.add(tokens)
        end = cursor + tokens - 1
        row_keys.extend(
            ChartRowKey(family, example, query, query + 1)
            for query in range(tokens - 1)
        )
        example_row = [family, example, tokens, cursor, end]
        if tokens in A_FIT_TARGET_TOKEN_COUNTS:
            fit_mask[cursor:end] = b"\x01" * (end - cursor)
            fit_examples.append(example_row)
        elif tokens in A_VALIDATION_TARGET_TOKEN_COUNTS:
            validation_mask[cursor:end] = b"\x01" * (end - cursor)
            validation_examples.append(example_row)
        else:
            raise ValueError("A partition encountered an unregistered target length")
        cursor = end
    expected_lengths = set(
        (*A_FIT_TARGET_TOKEN_COUNTS, *A_VALIDATION_TARGET_TOKEN_COUNTS)
    )
    if (
        cursor != PRODUCTION_A_ROW_COUNT
        or len(row_keys) != PRODUCTION_A_ROW_COUNT
        or len(by_family) != 6
        or any(lengths != expected_lengths for lengths in by_family.values())
    ):
        raise ValueError("A partition 6-family/48-example axis did not close")
    if row_catalog_sha256(
        tuple(row_keys),
        role=PREDECESSOR_A_ROLE,
    ) != bridge_receipt.shared_spool_row_catalog_sha256:
        raise ValueError("A partition row catalog differs from full-A bridge")
    if any(
        (fit_value + validation_value) != 1
        for fit_value, validation_value in zip(
            fit_mask,
            validation_mask,
            strict=True,
        )
    ):
        raise ValueError("A fit/validation masks are not disjoint and exhaustive")
    fit_count = sum(fit_mask)
    validation_count = sum(validation_mask)
    if (
        fit_count != 10_548
        or validation_count != 14_184
        or fit_count + validation_count != PRODUCTION_A_ROW_COUNT
    ):
        raise ValueError("A fit/validation mask counts drifted")
    axis = tuple(
        (source[0], source[1]) for source in bridge_receipt.layer_sources
    )
    return ADevelopmentPartitionReceipt(
        source_full_A_bridge_receipt_sha256=bridge_receipt.receipt_sha256,
        source_role_capture_receipt_sha256=(
            bridge_receipt.source_role_capture_receipt_sha256
        ),
        source_runtime_session_binding_sha256=(
            bridge_receipt.source_runtime_session_binding_sha256
        ),
        source_role_panel_binding_sha256=(
            bridge_receipt.source_role_panel_binding_sha256
        ),
        source_batch_axis_catalog_sha256=(
            bridge_receipt.source_batch_axis_catalog_sha256
        ),
        source_shared_spool_row_catalog_sha256=(
            bridge_receipt.shared_spool_row_catalog_sha256
        ),
        source_ordered_objective_row_catalog_sha256=(
            bridge_receipt.source_ordered_objective_row_catalog_sha256
        ),
        source_score_objective_receipt_sha256=(
            bridge_receipt.source_score_objective_receipt_sha256
        ),
        canonical_layer_mode_axis=axis,
        canonical_layer_mode_axis_sha256=(
            _A_partition_layer_axis_sha256(axis)
        ),
        fit_target_token_counts=A_FIT_TARGET_TOKEN_COUNTS,
        validation_target_token_counts=A_VALIDATION_TARGET_TOKEN_COUNTS,
        mask_hash_abi=_A_MASK_HASH_ABI,
        fit_mask_sha256=_A_partition_mask_sha256("fit", bytes(fit_mask)),
        validation_mask_sha256=_A_partition_mask_sha256(
            "validation",
            bytes(validation_mask),
        ),
        fit_example_catalog_sha256=_digest(
            _A_PARTITION_EXAMPLE_CATALOG_DOMAIN,
            {"examples": fit_examples, "partition": "fit"},
        ),
        validation_example_catalog_sha256=_digest(
            _A_PARTITION_EXAMPLE_CATALOG_DOMAIN,
            {"examples": validation_examples, "partition": "validation"},
        ),
        fit_row_count=fit_count,
        validation_row_count=validation_count,
        row_count=PRODUCTION_A_ROW_COUNT,
        layer_count=GEMMA3_LAYER_COUNT,
    )


def _row_key_tuple(value: ChartRowKey) -> tuple[str, str, int, int]:
    if type(value) is not ChartRowKey:
        raise TypeError("production spool row keys must be exact ChartRowKey values")
    return (
        value.family_id,
        value.example_id,
        value.prediction_query_position,
        value.target_label_position,
    )


def _validate_full_A_scalar_axes(
    *,
    family_ids: tuple[str, ...],
    example_ids: tuple[str, ...],
    token_counts: tuple[int, ...],
    row_keys: tuple[tuple[str, str, int, int], ...],
    chart_row_keys: tuple[ChartRowKey, ...],
    example_row_ranges: tuple[tuple[int, int], ...],
    source_capture_receipt_sha256s: tuple[str, ...],
    expected_row_catalog_sha256: str,
) -> None:
    axes = (
        family_ids,
        example_ids,
        token_counts,
        row_keys,
        chart_row_keys,
        example_row_ranges,
        source_capture_receipt_sha256s,
    )
    if any(type(value) is not tuple for value in axes):
        raise TypeError("full-A bridge scalar axes must be exact tuples")
    if (
        len(family_ids) != PRODUCTION_A_EXAMPLE_COUNT
        or len(example_ids) != PRODUCTION_A_EXAMPLE_COUNT
        or len(token_counts) != PRODUCTION_A_EXAMPLE_COUNT
        or len(example_row_ranges) != PRODUCTION_A_EXAMPLE_COUNT
        or len(source_capture_receipt_sha256s)
        != PRODUCTION_A_EXAMPLE_COUNT
        or len(set(example_ids)) != PRODUCTION_A_EXAMPLE_COUNT
        or any(type(value) is not str or not value for value in family_ids)
        or any(type(value) is not str or not value for value in example_ids)
        or any(type(value) is not int or value < 2 for value in token_counts)
    ):
        raise ValueError("full-A bridge example axes drifted")
    for digest in source_capture_receipt_sha256s:
        _require_sha256(digest, label="full-A example capture receipt")
    expected_rows: list[tuple[str, str, int, int]] = []
    cursor = 0
    by_family: dict[str, set[int]] = {}
    for family, example, tokens, bounds in zip(
        family_ids,
        example_ids,
        token_counts,
        example_row_ranges,
        strict=True,
    ):
        expected_bounds = (cursor, cursor + tokens - 1)
        if (
            type(bounds) is not tuple
            or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or bounds != expected_bounds
        ):
            raise ValueError("full-A bridge example ranges drifted")
        lengths = by_family.setdefault(family, set())
        if tokens in lengths:
            raise ValueError("full-A bridge repeats a family target length")
        lengths.add(tokens)
        expected_rows.extend(
            (family, example, query, query + 1)
            for query in range(tokens - 1)
        )
        cursor = expected_bounds[1]
    expected_lengths = set(
        (*A_FIT_TARGET_TOKEN_COUNTS, *A_VALIDATION_TARGET_TOKEN_COUNTS)
    )
    if (
        cursor != PRODUCTION_A_ROW_COUNT
        or len(by_family) != 6
        or any(lengths != expected_lengths for lengths in by_family.values())
        or row_keys != tuple(expected_rows)
        or len(chart_row_keys) != PRODUCTION_A_ROW_COUNT
        or tuple(_row_key_tuple(value) for value in chart_row_keys) != row_keys
        or row_catalog_sha256(
            chart_row_keys,
            role=PREDECESSOR_A_ROLE,
        )
        != expected_row_catalog_sha256
    ):
        raise ValueError("full-A bridge row/family catalog drifted")


def _validate_reopened_full_A_spool(
    spool: RawModeRowSpool,
    *,
    layer_ordinal: int,
    expected_path: Path,
    expected_artifact: tuple[str, str, str, str],
    expected_panel_binding_sha256: str,
    expected_row_keys: tuple[ChartRowKey, ...],
) -> tuple[str, str, str]:
    if type(spool) is not RawModeRowSpool:
        raise TypeError("full-A bridge strict reopen returned a non-spool value")
    mode_id, artifact, rows, tensors = expected_artifact
    if (
        spool.path != expected_path
        or spool.mode_id != mode_id
        or mode_id != _mlp_mode_id(layer_ordinal)
        or spool.artifact_sha256 != artifact
        or spool.row_catalog_sha256 != rows
        or spool.tensor_catalog_sha256 != tensors
    ):
        raise ValueError("full-A MLP spool differs from role capture receipt")
    if (
        type(spool.role) is not str
        or spool.role != PREDECESSOR_A_ROLE
        or type(spool.panel_binding_sha256) is not str
        or spool.panel_binding_sha256 != expected_panel_binding_sha256
        or type(spool.mode_id) is not str
        or type(spool.artifact_sha256) is not str
        or type(spool.row_catalog_sha256) is not str
        or type(spool.tensor_catalog_sha256) is not str
        or type(spool.hidden_width) is not int
        or spool.hidden_width != GEMMA3_HIDDEN_WIDTH
        or spool.row_count != PRODUCTION_A_ROW_COUNT
        or type(spool.row_keys) is not tuple
        or any(type(value) is not ChartRowKey for value in spool.row_keys)
        or spool.row_keys != expected_row_keys
    ):
        raise ValueError("full-A MLP spool role/panel/row lineage drifted")
    semantic = spool.field_semantic_sha256s
    if set(semantic) != {
        "comparison_values",
        "context_values",
        "generalized_contributions",
    }:
        raise ValueError("full-A MLP spool tensor field inventory drifted")
    context = _require_sha256(
        semantic["context_values"],
        label="spool context semantic tensor",
    )
    comparison = _require_sha256(
        semantic["comparison_values"],
        label="spool comparison semantic tensor",
    )
    generalized = _require_sha256(
        semantic["generalized_contributions"],
        label="spool generalized semantic tensor",
    )
    return context, comparison, generalized


@dataclass(frozen=True, slots=True)
class LayerMLPFlowSpoolChunk:
    start_row: int
    end_row: int
    row_keys: tuple[tuple[str, str, int, int], ...]
    chart_states: Tensor
    teacher_velocities: Tensor

    def __post_init__(self) -> None:
        if (
            type(self.start_row) is not int
            or type(self.end_row) is not int
            or not 0 <= self.start_row < self.end_row
            or type(self.row_keys) is not tuple
            or len(self.row_keys) != self.end_row - self.start_row
        ):
            raise ValueError("full-A MLP spool chunk row interval is invalid")
        for label, value in (
            ("chart_states", self.chart_states),
            ("teacher_velocities", self.teacher_velocities),
        ):
            if (
                not isinstance(value, Tensor)
                or value.shape
                != (self.end_row - self.start_row, GEMMA3_HIDDEN_WIDTH)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or value.requires_grad
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"{label} must be finite contiguous CPU float64 spool rows"
                )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMLPFlowSpoolInput:
    """One strictly reopened production A MLP spool and its full-role lineage."""

    spool: RawModeRowSpool
    bridge_receipt: FullALayerMLPSpoolBridgeReceipt
    layer_ordinal: int
    family_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    token_counts: tuple[int, ...]
    row_keys: tuple[tuple[str, str, int, int], ...]
    example_row_ranges: tuple[tuple[int, int], ...]
    source_capture_receipt_sha256s: tuple[str, ...]
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.spool) is not RawModeRowSpool:
            raise TypeError("spool input must hold one exact RawModeRowSpool")
        if type(self.bridge_receipt) is not FullALayerMLPSpoolBridgeReceipt:
            raise TypeError("spool input bridge receipt has the wrong type")
        self.bridge_receipt.assert_integrity()
        layer = _mlp_mode_id(self.layer_ordinal)
        source = self.bridge_receipt.layer_sources[self.layer_ordinal]
        context, comparison, _ = _validate_reopened_full_A_spool(
            self.spool,
            layer_ordinal=self.layer_ordinal,
            expected_path=self.spool.path,
            expected_artifact=(source[1], source[2], source[3], source[4]),
            expected_panel_binding_sha256=(
                self.bridge_receipt.source_role_panel_binding_sha256
            ),
            expected_row_keys=self.spool.row_keys,
        )
        if (
            self.spool.mode_id != layer
            or source[1] != layer
            or self.spool.artifact_sha256 != source[2]
            or self.spool.row_catalog_sha256 != source[3]
            or self.spool.tensor_catalog_sha256 != source[4]
            or tuple(_row_key_tuple(value) for value in self.spool.row_keys)
            != self.row_keys
            or context != source[5]
            or comparison != source[6]
        ):
            raise ValueError("one-layer spool input lineage drifted")
        _validate_full_A_scalar_axes(
            family_ids=self.family_ids,
            example_ids=self.example_ids,
            token_counts=self.token_counts,
            row_keys=self.row_keys,
            chart_row_keys=self.spool.row_keys,
            example_row_ranges=self.example_row_ranges,
            source_capture_receipt_sha256s=(
                self.source_capture_receipt_sha256s
            ),
            expected_row_catalog_sha256=(
                self.bridge_receipt.shared_spool_row_catalog_sha256
            ),
        )

    @property
    def mode_id(self) -> str:
        return _mlp_mode_id(self.layer_ordinal)

    @property
    def row_count(self) -> int:
        return len(self.row_keys)

    @property
    def live_full_A_authorized(self) -> bool:
        """True only for an input issued by a live strict bridge reopen."""

        return (
            self._construction_authority is _FULL_A_SPOOL_INPUT_AUTHORITY
            and _LIVE_FULL_A_SPOOL_INPUTS.get(id(self)) is self
        )

    def iter_float64_chunks(
        self,
        *,
        chunk_size: int | None = None,
    ) -> Iterator[LayerMLPFlowSpoolChunk]:
        """Stream only normalized x and teacher v_T; h is not in this artifact."""

        if not self.live_full_A_authorized:
            raise PermissionError(
                "only a live strict full-A bridge may stream production rows"
            )
        self.bridge_receipt.assert_integrity()
        cursor = 0
        for chunk in self.spool.iter_float64_chunks(
            fields=("context_values", "comparison_values"),
            chunk_size=chunk_size,
        ):
            keys = tuple(_row_key_tuple(value) for value in chunk.row_keys)
            if (
                chunk.start_row != cursor
                or chunk.end_row <= chunk.start_row
                or keys != self.row_keys[chunk.start_row : chunk.end_row]
            ):
                raise ValueError("full-A MLP spool chunk order drifted")
            result = LayerMLPFlowSpoolChunk(
                start_row=chunk.start_row,
                end_row=chunk.end_row,
                row_keys=keys,
                chart_states=chunk.tensors["context_values"].contiguous(),
                teacher_velocities=chunk.tensors[
                    "comparison_values"
                ].contiguous(),
            )
            cursor = chunk.end_row
            yield result
        if cursor != self.row_count:
            raise ValueError("full-A MLP spool chunk coverage is incomplete")

    def materialize_rows(self) -> LayerMLPFlowRows:
        """Materialize exactly one production layer as detached CPU float32 rows."""

        states = torch.empty(
            (self.row_count, GEMMA3_HIDDEN_WIDTH),
            dtype=torch.float32,
        )
        velocities = torch.empty_like(states)
        cursor = 0
        for chunk in self.iter_float64_chunks():
            states[chunk.start_row : chunk.end_row].copy_(chunk.chart_states)
            velocities[chunk.start_row : chunk.end_row].copy_(
                chunk.teacher_velocities
            )
            cursor = chunk.end_row
        if cursor != self.row_count:
            raise ValueError("one-layer full-A materialization is incomplete")
        source = self.bridge_receipt.layer_sources[self.layer_ordinal]
        return LayerMLPFlowRows(
            role=A_DEVELOPMENT_ROLE,
            layer_ordinal=self.layer_ordinal,
            mode_id=self.mode_id,
            family_ids=self.family_ids,
            example_ids=self.example_ids,
            token_counts=self.token_counts,
            row_keys=self.row_keys,
            example_row_ranges=self.example_row_ranges,
            chart_states=states.contiguous(),
            teacher_velocities=velocities.contiguous(),
            predecessor_source_role=PREDECESSOR_A_ROLE,
            source_batch_lexical_nll_receipt_sha256=None,
            source_batch_execution_receipt_sha256=None,
            source_capture_receipt_sha256s=(
                self.source_capture_receipt_sha256s
            ),
            chart_state_sha256=_tensor_sha256(states),
            teacher_velocity_sha256=_tensor_sha256(velocities),
            source_capture_scope=_PRODUCTION_FULL_A_SCOPE,
            source_role_capture_receipt_sha256=(
                self.bridge_receipt.source_role_capture_receipt_sha256
            ),
            source_runtime_session_binding_sha256=(
                self.bridge_receipt.source_runtime_session_binding_sha256
            ),
            source_role_panel_binding_sha256=(
                self.bridge_receipt.source_role_panel_binding_sha256
            ),
            source_spool_artifact_sha256=source[2],
            source_spool_row_catalog_sha256=source[3],
            source_spool_tensor_catalog_sha256=source[4],
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FullALayerMLPSpoolBridge:
    """Validated full-role bridge retaining paths, not eighteen layer tensors."""

    spool_paths: tuple[Path, ...]
    family_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    token_counts: tuple[int, ...]
    row_keys: tuple[tuple[str, str, int, int], ...]
    chart_row_keys: tuple[ChartRowKey, ...]
    example_row_ranges: tuple[tuple[int, int], ...]
    source_capture_receipt_sha256s: tuple[str, ...]
    max_artifact_bytes: int
    receipt: FullALayerMLPSpoolBridgeReceipt
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.spool_paths) is not tuple
            or len(self.spool_paths) != GEMMA3_LAYER_COUNT
            or any(
                type(value) is not _EXACT_PATH_TYPE
                for value in self.spool_paths
            )
            or len(set(self.spool_paths)) != GEMMA3_LAYER_COUNT
        ):
            raise ValueError("full-A bridge spool path inventory is invalid")
        if type(self.max_artifact_bytes) is not int or self.max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be a positive integer")
        if type(self.receipt) is not FullALayerMLPSpoolBridgeReceipt:
            raise TypeError("full-A bridge receipt has the wrong type")
        self.receipt.assert_integrity()
        _validate_full_A_scalar_axes(
            family_ids=self.family_ids,
            example_ids=self.example_ids,
            token_counts=self.token_counts,
            row_keys=self.row_keys,
            chart_row_keys=self.chart_row_keys,
            example_row_ranges=self.example_row_ranges,
            source_capture_receipt_sha256s=(
                self.source_capture_receipt_sha256s
            ),
            expected_row_catalog_sha256=(
                self.receipt.shared_spool_row_catalog_sha256
            ),
        )

    @property
    def live_full_A_authorized(self) -> bool:
        """True only for the original bridge returned by the strict opener."""

        return (
            self._construction_authority is _FULL_A_BRIDGE_BUILDER_AUTHORITY
            and _LIVE_FULL_A_BRIDGES.get(id(self)) is self
        )

    def open_layer(self, layer_ordinal: int) -> LayerMLPFlowSpoolInput:
        if not self.live_full_A_authorized:
            raise PermissionError(
                "only the live strict role opener may reopen production spools"
            )
        mode_id = _mlp_mode_id(layer_ordinal)
        source = self.receipt.layer_sources[layer_ordinal]
        expected = (source[1], source[2], source[3], source[4])
        path = self.spool_paths[layer_ordinal]
        reopened = open_raw_mode_row_spool(
            path,
            max_artifact_bytes=self.max_artifact_bytes,
        )
        context, comparison, _ = _validate_reopened_full_A_spool(
            reopened,
            layer_ordinal=layer_ordinal,
            expected_path=path,
            expected_artifact=expected,
            expected_panel_binding_sha256=(
                self.receipt.source_role_panel_binding_sha256
            ),
            expected_row_keys=self.chart_row_keys,
        )
        if context != source[5] or comparison != source[6] or reopened.mode_id != mode_id:
            raise ValueError("one-layer full-A semantic tensor lineage drifted")
        result = LayerMLPFlowSpoolInput(
            spool=reopened,
            bridge_receipt=self.receipt,
            layer_ordinal=layer_ordinal,
            family_ids=self.family_ids,
            example_ids=self.example_ids,
            token_counts=self.token_counts,
            row_keys=self.row_keys,
            example_row_ranges=self.example_row_ranges,
            source_capture_receipt_sha256s=(
                self.source_capture_receipt_sha256s
            ),
        )
        object.__setattr__(
            result,
            "_construction_authority",
            _FULL_A_SPOOL_INPUT_AUTHORITY,
        )
        _LIVE_FULL_A_SPOOL_INPUTS[id(result)] = result
        return result

    def materialize_layer_rows(self, layer_ordinal: int) -> LayerMLPFlowRows:
        return self.open_layer(layer_ordinal).materialize_rows()


def open_full_A_layer_mlp_spool_bridge(
    *,
    capture_receipt: RoleCaptureReceipt,
    mlp_spools: tuple[RawModeRowSpool, ...],
    role_manifest: Mapping[str, object],
    expected_capture_receipt_sha256: str,
    expected_runtime_session_binding_sha256: str,
    expected_role_panel_binding_sha256: str,
    max_artifact_bytes: int = DEFAULT_MAX_RAW_SPOOL_BYTES,
) -> FullALayerMLPSpoolBridge:
    """Strictly reopen the complete production A role's eighteen MLP spools."""

    if type(capture_receipt) is not RoleCaptureReceipt:
        raise TypeError("full-A bridge requires the exact RoleCaptureReceipt type")
    if type(max_artifact_bytes) is not int or max_artifact_bytes <= 0:
        raise ValueError("max_artifact_bytes must be a positive integer")
    for label, value in (
        ("expected capture receipt", expected_capture_receipt_sha256),
        ("expected runtime session binding", expected_runtime_session_binding_sha256),
        ("expected role panel binding", expected_role_panel_binding_sha256),
    ):
        _require_sha256(value, label=label)
    capture_state = {
        **capture_receipt.payload_without_sha256(),
        "receipt_sha256": capture_receipt.receipt_sha256,
    }
    parsed = _predecessor_capture._parse_role_capture_payload(capture_state)
    parsed_state = {
        **parsed.payload_without_sha256(),
        "receipt_sha256": parsed.receipt_sha256,
    }
    if _canonical_json_bytes(parsed_state) != _canonical_json_bytes(capture_state):
        raise ValueError("full-A role capture receipt canonical replay drifted")
    if (
        parsed.role != PREDECESSOR_A_ROLE
        or parsed.receipt_sha256 != expected_capture_receipt_sha256
        or parsed.runtime_session_binding_sha256
        != expected_runtime_session_binding_sha256
        or parsed.role_panel_binding_sha256
        != expected_role_panel_binding_sha256
    ):
        raise ValueError("full-A role/session/panel lineage drifted")
    if (
        parsed.batch_count != PRODUCTION_A_BATCH_COUNT
        or len(parsed.lexical_nll_receipt_sha256s) != PRODUCTION_A_BATCH_COUNT
        or len(parsed.execution_receipt_sha256s) != PRODUCTION_A_BATCH_COUNT
        or len(parsed.example_capture_receipt_sha256s)
        != PRODUCTION_A_EXAMPLE_COUNT
        or parsed.row_count != PRODUCTION_A_ROW_COUNT
    ):
        raise ValueError("full-A 45-batch/48-example capture is incomplete")
    if not isinstance(role_manifest, Mapping):
        raise TypeError("role_manifest must be the frozen A role mapping")
    manifest_role = role_manifest.get("role")
    if type(manifest_role) is not str or manifest_role != PREDECESSOR_A_ROLE:
        raise ValueError("full-A bridge requires the frozen A role manifest")
    manifest_ordered_rows = role_manifest.get(
        "ordered_objective_row_catalog_sha256"
    )
    manifest_score_objective = role_manifest.get(
        "score_objective_receipt_sha256"
    )
    if (
        type(manifest_ordered_rows) is not str
        or type(manifest_score_objective) is not str
    ):
        raise TypeError(
            "frozen manifest objective lineage fields must be exact strings"
        )
    _require_sha256(
        manifest_ordered_rows,
        label="manifest ordered objective row catalog",
    )
    _require_sha256(
        manifest_score_objective,
        label="manifest score objective receipt",
    )
    if (
        parsed.ordered_objective_row_catalog_sha256
        != manifest_ordered_rows
        or parsed.score_objective_receipt_sha256
        != manifest_score_objective
    ):
        raise ValueError(
            "full-A capture objective lineage differs from frozen manifest"
        )
    _predecessor_capture._validate_capture_manifest_axis(parsed, role_manifest)
    raw_keys = _predecessor_capture._canonical_role_row_keys(role_manifest)
    if len(raw_keys) != PRODUCTION_A_ROW_COUNT:
        raise ValueError("full-A role manifest row inventory drifted")
    chart_keys = tuple(ChartRowKey(*value) for value in raw_keys)
    expected_row_catalog = row_catalog_sha256(
        chart_keys,
        role=PREDECESSOR_A_ROLE,
    )
    members = tuple(role_manifest["ordered_members"])
    if len(members) != PRODUCTION_A_EXAMPLE_COUNT:
        raise ValueError("full-A role manifest example inventory drifted")
    family_ids: list[str] = []
    example_ids: list[str] = []
    token_counts: list[int] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for member in members:
        if not isinstance(member, Mapping):
            raise TypeError("full-A role manifest member is invalid")
        family = member.get("family_id")
        example = member.get("example_id")
        tokens = member.get("target_token_count")
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(example, str)
            or not example
            or type(tokens) is not int
            or tokens < 2
        ):
            raise ValueError("full-A role manifest member axis is invalid")
        family_ids.append(family)
        example_ids.append(example)
        token_counts.append(tokens)
        ranges.append((cursor, cursor + tokens - 1))
        cursor += tokens - 1
    if (
        cursor != PRODUCTION_A_ROW_COUNT
        or tuple(example_ids)
        != tuple(value[0] for value in parsed.example_capture_receipt_sha256s)
    ):
        raise ValueError("full-A role manifest/capture example order drifted")

    expected_modes = tuple(_mlp_mode_id(layer) for layer in range(18))
    expected_artifacts = tuple(
        value for value in parsed.mode_raw_artifacts if value[0] in expected_modes
    )
    if tuple(value[0] for value in expected_artifacts) != expected_modes:
        raise ValueError("full-A role receipt even MLP inventory drifted")
    if (
        type(mlp_spools) is not tuple
        or len(mlp_spools) != GEMMA3_LAYER_COUNT
        or any(type(value) is not RawModeRowSpool for value in mlp_spools)
    ):
        raise TypeError("full-A bridge requires exactly 18 typed MLP spools")
    if tuple(value.mode_id for value in mlp_spools) != expected_modes:
        raise ValueError("full-A MLP spools are incomplete or out of order")
    paths = tuple(value.path for value in mlp_spools)
    if (
        any(type(value) is not _EXACT_PATH_TYPE for value in paths)
        or len(set(paths)) != 18
    ):
        raise ValueError("full-A MLP spool paths must be unique exact paths")

    layer_sources: list[tuple[int, str, str, str, str, str, str]] = []
    for layer, (provided, expected, path) in enumerate(
        zip(mlp_spools, expected_artifacts, paths, strict=True)
    ):
        provided_artifact = (
            provided.mode_id,
            provided.artifact_sha256,
            provided.row_catalog_sha256,
            provided.tensor_catalog_sha256,
        )
        if provided_artifact != expected or expected[2] != expected_row_catalog:
            raise ValueError("provided full-A MLP spool lineage drifted")
        provided_semantic_triplet = _validate_reopened_full_A_spool(
            provided,
            layer_ordinal=layer,
            expected_path=path,
            expected_artifact=expected,
            expected_panel_binding_sha256=parsed.role_panel_binding_sha256,
            expected_row_keys=chart_keys,
        )
        reopened = open_raw_mode_row_spool(
            path,
            max_artifact_bytes=max_artifact_bytes,
        )
        context, comparison, generalized = _validate_reopened_full_A_spool(
            reopened,
            layer_ordinal=layer,
            expected_path=path,
            expected_artifact=expected,
            expected_panel_binding_sha256=parsed.role_panel_binding_sha256,
            expected_row_keys=chart_keys,
        )
        if (context, comparison, generalized) != provided_semantic_triplet:
            raise ValueError("reopened full-A MLP semantic lineage drifted")
        layer_sources.append(
            (
                layer,
                expected[0],
                expected[1],
                expected[2],
                expected[3],
                context,
                comparison,
            )
        )
        del reopened

    bridge_receipt = FullALayerMLPSpoolBridgeReceipt(
        source_role_capture_receipt_sha256=parsed.receipt_sha256,
        source_runtime_session_binding_sha256=(
            parsed.runtime_session_binding_sha256
        ),
        source_role_panel_binding_sha256=parsed.role_panel_binding_sha256,
        source_batch_axis_catalog_sha256=parsed.batch_axis_catalog_sha256,
        source_ordered_objective_row_catalog_sha256=(
            parsed.ordered_objective_row_catalog_sha256
        ),
        source_score_objective_receipt_sha256=(
            parsed.score_objective_receipt_sha256
        ),
        shared_spool_row_catalog_sha256=expected_row_catalog,
        layer_sources=tuple(layer_sources),
        batch_count=parsed.batch_count,
        example_count=len(parsed.example_capture_receipt_sha256s),
        row_count=parsed.row_count,
        hidden_width=GEMMA3_HIDDEN_WIDTH,
    )
    result = FullALayerMLPSpoolBridge(
        spool_paths=paths,
        family_ids=tuple(family_ids),
        example_ids=tuple(example_ids),
        token_counts=tuple(token_counts),
        row_keys=tuple(raw_keys),
        chart_row_keys=chart_keys,
        example_row_ranges=tuple(ranges),
        source_capture_receipt_sha256s=tuple(
            value[1] for value in parsed.example_capture_receipt_sha256s
        ),
        max_artifact_bytes=max_artifact_bytes,
        receipt=bridge_receipt,
    )
    object.__setattr__(
        result,
        "_construction_authority",
        _FULL_A_BRIDGE_BUILDER_AUTHORITY,
    )
    _LIVE_FULL_A_BRIDGES[id(result)] = result
    return result


def _lower_authenticated_A_examples(
    *,
    layer_ordinal: int,
    examples: Sequence[CapturedExampleRows],
    source_batch_lexical_nll_receipt_sha256: str,
    source_batch_execution_receipt_sha256: str,
) -> LayerMLPFlowRows:
    """Lower examples only after their enclosing A batch was authenticated."""

    mode_id = _mlp_mode_id(layer_ordinal)
    values = tuple(examples)
    if (
        not values
        or any(not isinstance(value, CapturedExampleRows) for value in values)
        or len({value.example_id for value in values}) != len(values)
    ):
        raise ValueError("examples must be unique CapturedExampleRows")
    family_ids: list[str] = []
    example_ids: list[str] = []
    token_counts: list[int] = []
    row_keys: list[tuple[str, str, int, int]] = []
    ranges: list[tuple[int, int]] = []
    states: list[Tensor] = []
    velocities: list[Tensor] = []
    receipts: list[str] = []
    cursor = 0
    expected_modes = _expected_mode_ids()
    for example in values:
        if tuple(mode.mode_id for mode in example.modes) != expected_modes:
            raise ValueError("captured example 38-mode inventory drifted")
        by_mode = {mode.mode_id: mode for mode in example.modes}
        parent = by_mode[mode_id]
        rows = example.token_count - 1
        state = _canonical_capture_tensor(
            parent.context_values,
            rows=rows,
            label=f"{example.example_id} chart states",
        )
        velocity = _canonical_capture_tensor(
            parent.comparison_values,
            rows=rows,
            label=f"{example.example_id} teacher velocities",
        )
        if tuple(example.row_keys) != tuple(
            (example.family_id, example.example_id, query, query + 1)
            for query in range(rows)
        ):
            raise ValueError("captured example row keys drifted")
        family_ids.append(example.family_id)
        example_ids.append(example.example_id)
        token_counts.append(example.token_count)
        row_keys.extend(example.row_keys)
        ranges.append((cursor, cursor + rows))
        cursor += rows
        states.append(state)
        velocities.append(velocity)
        receipts.append(example.capture_receipt_sha256)
    chart_states = torch.cat(states, dim=0).contiguous()
    teacher_velocities = torch.cat(velocities, dim=0).contiguous()
    return LayerMLPFlowRows(
        role=A_DEVELOPMENT_ROLE,
        layer_ordinal=layer_ordinal,
        mode_id=mode_id,
        family_ids=tuple(family_ids),
        example_ids=tuple(example_ids),
        token_counts=tuple(token_counts),
        row_keys=tuple(row_keys),
        example_row_ranges=tuple(ranges),
        chart_states=chart_states,
        teacher_velocities=teacher_velocities,
        predecessor_source_role=PREDECESSOR_A_ROLE,
        source_batch_lexical_nll_receipt_sha256=(
            source_batch_lexical_nll_receipt_sha256
        ),
        source_batch_execution_receipt_sha256=(
            source_batch_execution_receipt_sha256
        ),
        source_capture_receipt_sha256s=tuple(receipts),
        chart_state_sha256=_tensor_sha256(chart_states),
        teacher_velocity_sha256=_tensor_sha256(teacher_velocities),
    )


def build_layer_mlp_flow_rows(
    batch: CapturedBatch,
    *,
    layer_ordinal: int,
    development_only: bool = False,
) -> LayerMLPFlowRows:
    """Lower one development fixture batch; never grants production authority."""

    if type(development_only) is not bool or not development_only:
        raise ValueError(
            "single-CapturedBatch lowering is development-only and requires "
            "development_only=True"
        )
    if type(batch) is not CapturedBatch:
        raise TypeError("batch must be CapturedBatch")
    if batch.role != PREDECESSOR_A_ROLE:
        raise ValueError(
            "only the predecessor A capture may enter v3 development; "
            "spent predecessor B/C captures are forbidden"
        )
    return _lower_authenticated_A_examples(
        layer_ordinal=layer_ordinal,
        examples=batch.examples,
        source_batch_lexical_nll_receipt_sha256=(
            batch.lexical_nll_receipt_sha256
        ),
        source_batch_execution_receipt_sha256=(
            batch.execution_receipt_sha256
        ),
    )


def build_layer_mlp_flow_rows_from_batch(
    batch: CapturedBatch,
    *,
    layer_ordinal: int,
    development_only: bool = False,
) -> LayerMLPFlowRows:
    """Compatibility spelling for the explicit development-only constructor."""

    return build_layer_mlp_flow_rows(
        batch,
        layer_ordinal=layer_ordinal,
        development_only=development_only,
    )


def build_layer_multimode_capture_catalog(
    batch: CapturedBatch,
    *,
    development_only: bool = False,
) -> LayerMLPFlowCaptureCatalog:
    """Build a development-only in-memory catalog from one small A batch."""

    if type(development_only) is not bool or not development_only:
        raise ValueError(
            "single-CapturedBatch catalogs are development-only and require "
            "development_only=True"
        )
    if type(batch) is not CapturedBatch:
        raise TypeError("batch must be CapturedBatch")
    if batch.role != PREDECESSOR_A_ROLE:
        raise ValueError(
            "only the predecessor A capture may enter v3 development; "
            "spent predecessor B/C captures are forbidden"
        )
    layers = tuple(
        _lower_authenticated_A_examples(
            layer_ordinal=layer,
            examples=batch.examples,
            source_batch_lexical_nll_receipt_sha256=(
                batch.lexical_nll_receipt_sha256
            ),
            source_batch_execution_receipt_sha256=(
                batch.execution_receipt_sha256
            ),
        )
        for layer in range(GEMMA3_LAYER_COUNT)
    )
    first = layers[0]
    row_catalog = _digest(
        _FLOW_CATALOG_DOMAIN,
        {
            "role": A_DEVELOPMENT_ROLE,
            "row_keys": [list(value) for value in first.row_keys],
            "source_capture_receipt_sha256s": list(
                first.source_capture_receipt_sha256s
            ),
            "source_batch_lexical_nll_receipt_sha256": (
                first.source_batch_lexical_nll_receipt_sha256
            ),
            "source_batch_execution_receipt_sha256": (
                first.source_batch_execution_receipt_sha256
            ),
        },
    )
    return LayerMLPFlowCaptureCatalog(
        role=A_DEVELOPMENT_ROLE,
        layers=layers,
        shared_row_catalog_sha256=row_catalog,
    )


def family_example_balanced_row_weights(rows: LayerMLPFlowRows) -> Tensor:
    """Give every family, example, and row equal nested total mass."""

    if not isinstance(rows, LayerMLPFlowRows):
        raise TypeError("rows must be LayerMLPFlowRows")
    family_counts = Counter(rows.family_ids)
    family_count = len(family_counts)
    weights = torch.empty(rows.row_count, dtype=torch.float64)
    for family, bounds in zip(
        rows.family_ids,
        rows.example_row_ranges,
        strict=True,
    ):
        start, end = bounds
        row_count = end - start
        weights[start:end] = 1.0 / (
            family_count * family_counts[family] * row_count
        )
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights <= 0.0).any())
        or not math.isclose(
            float(weights.sum().item()),
            1.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise RuntimeError("balanced flow row weights did not close")
    return weights.contiguous()


def fixed_A_development_split_masks(
    rows: LayerMLPFlowRows,
) -> tuple[Tensor, Tensor, str]:
    """Return the frozen family-balanced A fit/validation row masks."""

    if (
        not isinstance(rows, LayerMLPFlowRows)
        or rows.role != A_DEVELOPMENT_ROLE
    ):
        raise ValueError("the fixed development split requires A flow rows")
    by_family: dict[str, set[int]] = {}
    fit = torch.zeros(rows.row_count, dtype=torch.bool)
    validation = torch.zeros_like(fit)
    for family, tokens, bounds in zip(
        rows.family_ids,
        rows.token_counts,
        rows.example_row_ranges,
        strict=True,
    ):
        seen = by_family.setdefault(family, set())
        if tokens in seen:
            raise ValueError("A split has duplicate target length in one family")
        seen.add(tokens)
        start, end = bounds
        if tokens in A_FIT_TARGET_TOKEN_COUNTS:
            fit[start:end] = True
        elif tokens in A_VALIDATION_TARGET_TOKEN_COUNTS:
            validation[start:end] = True
        else:
            raise ValueError("A split encountered an unregistered target length")
    expected_lengths = set(
        (*A_FIT_TARGET_TOKEN_COUNTS, *A_VALIDATION_TARGET_TOKEN_COUNTS)
    )
    if (
        len(by_family) != 6
        or any(value != expected_lengths for value in by_family.values())
        or bool((fit & validation).any())
        or not bool((fit | validation).all())
        or int(fit.sum()) == 0
        or int(validation.sum()) == 0
    ):
        raise ValueError("A fit/validation split did not close")
    receipt = _digest(
        _WEIGHT_DOMAIN,
        {
            "flow_rows_receipt_sha256": rows.receipt_sha256,
            "fit_target_token_counts": list(A_FIT_TARGET_TOKEN_COUNTS),
            "validation_target_token_counts": list(
                A_VALIDATION_TARGET_TOKEN_COUNTS
            ),
            "fit_row_count": int(fit.sum()),
            "validation_row_count": int(validation.sum()),
            "fit_mask_sha256": hashlib.sha256(
                fit.numpy().tobytes(order="C")
            ).hexdigest(),
            "validation_mask_sha256": hashlib.sha256(
                validation.numpy().tobytes(order="C")
            ).hexdigest(),
        },
    )
    return fit.contiguous(), validation.contiguous(), receipt
