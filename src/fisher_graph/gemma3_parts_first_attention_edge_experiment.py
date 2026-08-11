"""Retrospective A-only attention-edge atlas for the parts-first panel.

The atlas treats all sixty frozen forced-choice examples as one exploratory A
panel.  It does not reuse the old A/B/C role split and cannot confirm or
promote lexical modes.  For every exact Gemma attention mutation it retains
the eight largest coordinate-diagonal Fisher routes across all heads and keys
into each query, while reducing every allowed route into the fixed
18-layer x 4-head x 8-distance-bin motif table.

The model-facing path is deliberately source independent after its first-row
control.  It loads the pinned native model locally, compiles the exact 39-event
control, proves literal native/exact parity at the single supervised
correct-choice query, and then releases its native-model references.  The
committed exact executor's observer supplies normalized attention inputs and
its one-use suffix-gradient tape supplies gradients at the post-normalized
attention delta.  Route coordinates are evaluated by
``gemma3_parts_first_attention_edges``; no private native-attention hook or
uncommitted route-factor API is used.

Published JSON contains no prompts, token IDs, target IDs, raw activations, or
raw gradients.  It is a retrospective structure atlas, not confirmation,
compression, or an attention-promotion decision.  Import and preflight paths
do not import Transformers or load a model.

The compact artifact validates its stored algebra and pinned-model count
geometry, but it is not a proof that omitted full-route values equal a live
execution or that the retained routes are the true global top-k.  Those claims
require trusted live execution or an externally authenticated report hash.
The implementation bundle identifies the edge core and this runner's exact
source bytes; it is not a digest of every transitive Python or PyTorch runtime
dependency.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Final, Protocol
import weakref

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from . import gemma3_parts_first_attention_edges as attention_edge_core
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_parts_first_attention_edges import (
    ATTENTION_EDGE_PROBABILITY_FORMULA_ID,
    PartsFirstAttentionEdgeExtraction,
    compile_parts_first_attention_edges,
)
from .gemma3_parts_first_supermode_experiment import MODEL_SCORE_OBJECTIVE_ID
from .gemma3_parts_first_supermode_model_runner import (
    PINNED_MODEL_REVISION,
    PINNED_PANEL_FILE_SHA256,
    PINNED_PANEL_ID,
    PINNED_PANEL_PATH,
    PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256,
    build_correct_choice_nll_batch,
    load_pinned_parts_first_panel,
)
from .gemma3_parts_first_supermode_protocol import (
    MODEL_ID,
    SOURCE_MODEL_FINGERPRINT_SHA256,
)
from .gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    GEMMA3_LAYER_COUNT,
    HIDDEN_WIDTH,
    SOURCE_UNIQUE_PARAMETER_COUNT,
    VOCABULARY_SIZE,
)


ATTENTION_EDGE_ATLAS_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edge_atlas.v1"
)
ATTENTION_EDGE_ATLAS_FORMAT_VERSION: Final = 1
ATTENTION_EDGE_ATLAS_ABI: Final = (
    "full_allowed_coordinate_diagonal_fisher_top8_per_layer_query.v1"
)
ATTENTION_EDGE_ATLAS_ROLE: Final = "A_retrospective_atlas"
ATTENTION_EDGE_TOP_K: Final = 8
ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS: Final = 128
PINNED_ATTENTION_EDGE_VALID_TOKEN_COUNT: Final = 814
PINNED_ATTENTION_EDGE_MAXIMUM_RETAINED_ROUTE_COUNT: Final = 117_216
PINNED_ATTENTION_EDGE_SMOKE_VALID_TOKEN_COUNT: Final = 11
ATTENTION_DISTANCE_BINS: Final = (
    "0",
    "1",
    "2-3",
    "4-15",
    "16-63",
    "64-255",
    "256-511",
    "512+",
)
DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-attention-edge-atlas-A-retrospective-v1.json"
)
PINNED_ATTENTION_EDGE_ROW_CATALOG_SHA256: Final = (
    "c2ba01221dbace2c47614974f05049a19e305adde4b43989a5ff8c1dc7e49d62"
)
PINNED_ATTENTION_EDGE_SMOKE_ROW_CATALOG_SHA256: Final = (
    "60bc294635498ddac2ba82e47d27763c46b957ed27d1d5c52bc35f5cc37bd4d1"
)
# Filled from the pinned tokenizer/revision with the frozen 128-token policy.
# The value is independently reproducible without running the model.
PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256: Final = (
    "d86cb8c35ed284bbb151ddf10550ea72a369ac4bc5d36bcac07b7870db274fc5"
)
PINNED_ATTENTION_EDGE_SMOKE_TOKENIZATION_CATALOG_SHA256: Final = (
    "c68f62f83ac15d62a164a1ffab2c2ab1b4a2bf36b024408fd42d5da53eca6f87"
)

_REPORT_DOMAIN = b"fisher_graph:parts-first-attention-edge-atlas:v1\0"
_ROW_DOMAIN = b"fisher_graph:parts-first-attention-edge-row:v1\0"
_CONTROL_DOMAIN = b"fisher_graph:parts-first-attention-edge-control:v1\0"
_CONTROL_METADATA_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-control-metadata:v1\0"
)
_INDEPENDENCE_AUDIT_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-independence-audit:v1\0"
)
_CATALOG_DOMAIN = b"fisher_graph:parts-first-attention-edge-catalog:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:parts-first-attention-edge-tensor:v1\0"
_FIXTURE_DOMAIN = b"fisher_graph:parts-first-attention-edge-fixture:v1\0"
_IMPLEMENTATION_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-implementation:v1\0"
)
_TOKENIZATION_ROW_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-tokenization-row:v1\0"
)
_TOKENIZATION_CATALOG_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-tokenization-catalog:v1\0"
)
_ROW_ID_DOMAIN = b"fisher_graph:parts-first-attention-edge-row-id:v1\0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FLOAT32_UNIT_ROUNDOFF = 2.0**-24
_FLOAT64_UNIT_ROUNDOFF = 2.0**-53
_FLOAT32_MIN_NORMAL = 2.0**-126
_EXPECTED_EVENT_IDS: Final = (
    "lexical.ingress",
    *(f"mutation.{index}" for index in range(1, 37)),
    "terminal.rmsnorm",
    "lexical.egress",
)


__all__ = [
    "ATTENTION_DISTANCE_BINS",
    "ATTENTION_EDGE_ATLAS_ABI",
    "ATTENTION_EDGE_ATLAS_FORMAT_VERSION",
    "ATTENTION_EDGE_ATLAS_ROLE",
    "ATTENTION_EDGE_ATLAS_SCHEMA",
    "ATTENTION_EDGE_TOP_K",
    "ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS",
    "PINNED_ATTENTION_EDGE_VALID_TOKEN_COUNT",
    "PINNED_ATTENTION_EDGE_MAXIMUM_RETAINED_ROUTE_COUNT",
    "DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT",
    "PINNED_ATTENTION_EDGE_ROW_CATALOG_SHA256",
    "PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256",
    "AttentionDistanceMotif",
    "AttentionEdgeAtlasConfig",
    "AttentionEdgeAtlasLiveConfig",
    "AttentionEdgeAtlasReport",
    "AttentionEdgeAtlasRowCapture",
    "AttentionEdgeAtlasRowReceipt",
    "AttentionEdgeExactControlReceipt",
    "AttentionEdgeLiveRuntime",
    "RetainedAttentionRoute",
    "build_attention_edge_atlas_preflight",
    "build_deterministic_attention_edge_fixture",
    "compile_attention_edge_atlas",
    "load_attention_edge_atlas",
    "main",
    "parse_attention_edge_atlas_json",
    "run_attention_edge_atlas",
    "run_attention_edge_one_row_smoke",
    "validate_attention_edge_atlas",
    "write_attention_edge_atlas",
]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _nonnegative_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _f64_slack(*values: float, operations: int = 64) -> float:
    if type(operations) is not int or operations <= 0:
        raise ValueError("roundoff operation count must be positive")
    scale = max(1.0, *(abs(float(value)) for value in values))
    return float(operations * _FLOAT64_UNIT_ROUNDOFF * scale)


def _probability_forward_error_bound(support_count: int) -> float:
    """Mirror the core's frozen stored-float32 softmax error model."""

    if ATTENTION_EDGE_PROBABILITY_FORMULA_ID != (
        "stored_float32_softmax.gamma32_support_plus_2_gamma64_plus_ftz.v1"
    ):
        raise RuntimeError("attention probability error model drifted")
    if type(support_count) is not int or support_count <= 0:
        raise ValueError("softmax support_count must be positive")
    numerator32 = (support_count + 2) * _FLOAT32_UNIT_ROUNDOFF
    numerator64 = (support_count - 1) * _FLOAT64_UNIT_ROUNDOFF
    if numerator32 >= 1.0 or numerator64 >= 1.0:
        raise ValueError("softmax support exceeds the frozen error model")
    gamma32 = numerator32 / (1.0 - numerator32)
    gamma64 = numerator64 / (1.0 - numerator64)
    return float(
        gamma32
        + gamma64 * (1.0 + gamma32)
        + support_count * _FLOAT32_MIN_NORMAL
    )


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} field inventory is invalid")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_object(source: str | bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            source,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("tensor hash input must be a Tensor")
    canonical = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(canonical.dtype), "shape": list(canonical.shape)}
    )
    raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(_TENSOR_DOMAIN + header + b"\0" + raw).hexdigest()


def _tokenization_row_sha256(
    row_catalog_id: str,
    batch: CalibrationBatch,
) -> str:
    if not isinstance(row_catalog_id, str) or not row_catalog_id:
        raise ValueError("tokenization row catalog ID must be nonempty")
    input_ids = batch.model_inputs.get("input_ids")
    attention_mask = batch.model_inputs.get("attention_mask")
    if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
        raise ValueError("tokenization receipt requires input IDs and mask")
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape != attention_mask.shape
        or batch.targets.shape != input_ids.shape
        or batch.valid_positions.shape != input_ids.shape
    ):
        raise ValueError("tokenization receipt tensor geometry drifted")
    return _json_sha256(
        {
            "row_catalog_id_sha256": hashlib.sha256(
                _ROW_ID_DOMAIN + row_catalog_id.encode("utf-8")
            ).hexdigest(),
            "input_ids_sha256": _tensor_sha256(input_ids),
            "attention_mask_sha256": _tensor_sha256(attention_mask),
            "targets_sha256": _tensor_sha256(batch.targets),
            "valid_positions_sha256": _tensor_sha256(batch.valid_positions),
            "sequence_length": int(input_ids.shape[1]),
            "maximum_prompt_tokens": ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS,
        },
        domain=_TOKENIZATION_ROW_DOMAIN,
    )


def _tokenization_catalog_sha256(
    row_sha256s: Sequence[str],
) -> str:
    materialized = tuple(row_sha256s)
    if not materialized:
        raise ValueError("tokenization catalog cannot be empty")
    for value in materialized:
        _require_sha256(value, label="tokenization row")
    return _json_sha256(
        [
            {"row_ordinal": ordinal, "tokenization_sha256": value}
            for ordinal, value in enumerate(materialized)
        ],
        domain=_TOKENIZATION_CATALOG_DOMAIN,
    )


def _implementation_provenance() -> tuple[str, str, str]:
    core_path_value = getattr(attention_edge_core, "__file__", None)
    if not isinstance(core_path_value, str):
        raise RuntimeError("attention edge core implementation path is unavailable")
    core_path = Path(core_path_value)
    runner_path = Path(__file__)
    try:
        core_sha256 = hashlib.sha256(core_path.read_bytes()).hexdigest()
        runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            "attention edge implementation provenance is unavailable"
        ) from error
    bundle_sha256 = _json_sha256(
        {
            "core_implementation_sha256": core_sha256,
            "runner_implementation_sha256": runner_sha256,
        },
        domain=_IMPLEMENTATION_DOMAIN,
    )
    return core_sha256, runner_sha256, bundle_sha256


def _distance_bin(distance: int) -> str:
    if type(distance) is not int or distance < 0:
        raise ValueError("causal distance must be a nonnegative integer")
    if distance == 0:
        return "0"
    if distance == 1:
        return "1"
    if distance <= 3:
        return "2-3"
    if distance <= 15:
        return "4-15"
    if distance <= 63:
        return "16-63"
    if distance <= 255:
        return "64-255"
    if distance <= 511:
        return "256-511"
    return "512+"


@dataclass(frozen=True, slots=True)
class AttentionEdgeAtlasConfig:
    top_routes_per_target: int = ATTENTION_EDGE_TOP_K
    distance_bins: tuple[str, ...] = ATTENTION_DISTANCE_BINS
    selection_coordinate: str = "intrinsic_fisher_energy"
    selection_order: str = (
        "descending_energy_then_ascending_head_then_key_ordinal"
    )
    atlas_abi: str = ATTENTION_EDGE_ATLAS_ABI

    def __post_init__(self) -> None:
        if self.top_routes_per_target != ATTENTION_EDGE_TOP_K:
            raise ValueError("attention atlas top-k policy drifted")
        if self.distance_bins != ATTENTION_DISTANCE_BINS:
            raise ValueError("attention atlas distance bins drifted")
        if self.selection_coordinate != "intrinsic_fisher_energy":
            raise ValueError("attention atlas primary coordinate drifted")
        if self.selection_order != (
            "descending_energy_then_ascending_head_then_key_ordinal"
        ):
            raise ValueError("attention atlas tie-break policy drifted")
        if self.atlas_abi != ATTENTION_EDGE_ATLAS_ABI:
            raise ValueError("attention atlas ABI drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "top_routes_per_target": self.top_routes_per_target,
            "distance_bins": list(self.distance_bins),
            "selection_coordinate": self.selection_coordinate,
            "selection_order": self.selection_order,
            "atlas_abi": self.atlas_abi,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AttentionEdgeAtlasConfig":
        _exact_keys(
            value,
            {
                "top_routes_per_target",
                "distance_bins",
                "selection_coordinate",
                "selection_order",
                "atlas_abi",
            },
            label="attention atlas config",
        )
        raw_bins = value["distance_bins"]
        if not isinstance(raw_bins, list):
            raise ValueError("attention distance bins must be a list")
        return cls(
            top_routes_per_target=_require_int(
                value["top_routes_per_target"],
                label="top_routes_per_target",
            ),
            distance_bins=tuple(str(item) for item in raw_bins),
            selection_coordinate=str(value["selection_coordinate"]),
            selection_order=str(value["selection_order"]),
            atlas_abi=str(value["atlas_abi"]),
        )


@dataclass(frozen=True, slots=True)
class AttentionEdgeExactControlReceipt:
    ordered_event_ids: tuple[str, ...]
    compiler_metadata_sha256: str
    source_independence_audit_sha256: str
    native_valid_logits_sha256: str
    exact_valid_logits_sha256: str
    valid_logit_shape: tuple[int, int]
    literal_native_exact_parity_passed: bool
    source_model_fingerprint_before: str
    source_model_fingerprint_after_parity: str
    source_execution_fingerprint_before: str
    source_execution_fingerprint_after_parity: str
    source_parameter_count: int
    exact_parameter_inventory_parity: bool
    source_independence_passed: bool
    native_source_references_released_before_atlas: bool
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.ordered_event_ids != _EXPECTED_EVENT_IDS:
            raise ValueError("attention atlas exact event schedule drifted")
        if (
            type(self.valid_logit_shape) is not tuple
            or len(self.valid_logit_shape) != 2
            or self.valid_logit_shape[0] != 1
            or self.valid_logit_shape[1] != VOCABULARY_SIZE
        ):
            raise ValueError("attention atlas valid-logit shape is invalid")
        for label, value in (
            ("compiler metadata", self.compiler_metadata_sha256),
            (
                "source independence audit",
                self.source_independence_audit_sha256,
            ),
            ("native valid logits", self.native_valid_logits_sha256),
            ("exact valid logits", self.exact_valid_logits_sha256),
            ("source model before", self.source_model_fingerprint_before),
            (
                "source model after parity",
                self.source_model_fingerprint_after_parity,
            ),
            (
                "source execution before",
                self.source_execution_fingerprint_before,
            ),
            (
                "source execution after parity",
                self.source_execution_fingerprint_after_parity,
            ),
        ):
            _require_sha256(value, label=label)
        for label, value in (
            (
                "literal_native_exact_parity_passed",
                self.literal_native_exact_parity_passed,
            ),
            (
                "exact_parameter_inventory_parity",
                self.exact_parameter_inventory_parity,
            ),
            ("source_independence_passed", self.source_independence_passed),
            (
                "native_source_references_released_before_atlas",
                self.native_source_references_released_before_atlas,
            ),
        ):
            _require_bool(value, label=label)
        if self.source_parameter_count != SOURCE_UNIQUE_PARAMETER_COUNT:
            raise ValueError("attention atlas source inventory drifted")
        if self.literal_native_exact_parity_passed and (
            self.native_valid_logits_sha256 != self.exact_valid_logits_sha256
        ):
            raise ValueError("literal parity must bind equal value hashes")
        if self.passed and (
            self.source_model_fingerprint_before
            != self.source_model_fingerprint_after_parity
            or self.source_execution_fingerprint_before
            != self.source_execution_fingerprint_after_parity
        ):
            raise ValueError("native source changed during exact-control parity")
        computed = _json_sha256(self._payload(), domain=_CONTROL_DOMAIN)
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256, label="exact control receipt"
        ) != computed:
            raise ValueError("attention atlas exact-control hash mismatch")

    @property
    def passed(self) -> bool:
        return bool(
            self.literal_native_exact_parity_passed
            and self.exact_parameter_inventory_parity
            and self.source_independence_passed
            and self.native_source_references_released_before_atlas
        )

    def _payload(self) -> dict[str, object]:
        return {
            "ordered_event_ids": list(self.ordered_event_ids),
            "compiler_metadata_sha256": self.compiler_metadata_sha256,
            "source_independence_audit_sha256": (
                self.source_independence_audit_sha256
            ),
            "native_valid_logits_sha256": self.native_valid_logits_sha256,
            "exact_valid_logits_sha256": self.exact_valid_logits_sha256,
            "valid_logit_shape": list(self.valid_logit_shape),
            "literal_native_exact_parity_passed": (
                self.literal_native_exact_parity_passed
            ),
            "source_model_fingerprint_before": (
                self.source_model_fingerprint_before
            ),
            "source_model_fingerprint_after_parity": (
                self.source_model_fingerprint_after_parity
            ),
            "source_execution_fingerprint_before": (
                self.source_execution_fingerprint_before
            ),
            "source_execution_fingerprint_after_parity": (
                self.source_execution_fingerprint_after_parity
            ),
            "source_parameter_count": self.source_parameter_count,
            "exact_parameter_inventory_parity": (
                self.exact_parameter_inventory_parity
            ),
            "source_independence_passed": self.source_independence_passed,
            "native_source_references_released_before_atlas": (
                self.native_source_references_released_before_atlas
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeExactControlReceipt":
        _exact_keys(
            value,
            {
                "ordered_event_ids",
                "compiler_metadata_sha256",
                "source_independence_audit_sha256",
                "native_valid_logits_sha256",
                "exact_valid_logits_sha256",
                "valid_logit_shape",
                "literal_native_exact_parity_passed",
                "source_model_fingerprint_before",
                "source_model_fingerprint_after_parity",
                "source_execution_fingerprint_before",
                "source_execution_fingerprint_after_parity",
                "source_parameter_count",
                "exact_parameter_inventory_parity",
                "source_independence_passed",
                "native_source_references_released_before_atlas",
                "receipt_sha256",
            },
            label="attention atlas exact control receipt",
        )
        events = value["ordered_event_ids"]
        shape = value["valid_logit_shape"]
        if not isinstance(events, list) or not isinstance(shape, list):
            raise ValueError("attention atlas exact-control arrays are invalid")
        return cls(
            ordered_event_ids=tuple(str(item) for item in events),
            compiler_metadata_sha256=str(value["compiler_metadata_sha256"]),
            source_independence_audit_sha256=str(
                value["source_independence_audit_sha256"]
            ),
            native_valid_logits_sha256=str(value["native_valid_logits_sha256"]),
            exact_valid_logits_sha256=str(value["exact_valid_logits_sha256"]),
            valid_logit_shape=tuple(
                _require_int(item, label="valid_logit_shape", minimum=1)
                for item in shape
            ),  # type: ignore[arg-type]
            literal_native_exact_parity_passed=_require_bool(
                value["literal_native_exact_parity_passed"],
                label="literal_native_exact_parity_passed",
            ),
            source_model_fingerprint_before=str(
                value["source_model_fingerprint_before"]
            ),
            source_model_fingerprint_after_parity=str(
                value["source_model_fingerprint_after_parity"]
            ),
            source_execution_fingerprint_before=str(
                value["source_execution_fingerprint_before"]
            ),
            source_execution_fingerprint_after_parity=str(
                value["source_execution_fingerprint_after_parity"]
            ),
            source_parameter_count=_require_int(
                value["source_parameter_count"], label="source_parameter_count"
            ),
            exact_parameter_inventory_parity=_require_bool(
                value["exact_parameter_inventory_parity"],
                label="exact_parameter_inventory_parity",
            ),
            source_independence_passed=_require_bool(
                value["source_independence_passed"],
                label="source_independence_passed",
            ),
            native_source_references_released_before_atlas=_require_bool(
                value["native_source_references_released_before_atlas"],
                label="native_source_references_released_before_atlas",
            ),
            receipt_sha256=str(value["receipt_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class RetainedAttentionRoute:
    row_ordinal: int
    layer_ordinal: int
    query_head_ordinal: int
    query_ordinal: int
    key_ordinal: int
    causal_distance: int
    distance_bin: str
    target_rank: int
    probability: float
    signed_response: float
    intrinsic_fisher_energy: float
    softmax_logit_tangent: float

    def __post_init__(self) -> None:
        for label, value in (
            ("row_ordinal", self.row_ordinal),
            ("layer_ordinal", self.layer_ordinal),
            ("query_head_ordinal", self.query_head_ordinal),
            ("query_ordinal", self.query_ordinal),
            ("key_ordinal", self.key_ordinal),
            ("causal_distance", self.causal_distance),
            ("target_rank", self.target_rank),
        ):
            _require_int(value, label=label)
        if not 1 <= self.target_rank <= ATTENTION_EDGE_TOP_K:
            raise ValueError("retained route rank is outside top-k")
        if self.key_ordinal > self.query_ordinal:
            raise ValueError("retained attention route must be causal")
        if self.causal_distance != self.query_ordinal - self.key_ordinal:
            raise ValueError(
                "retained route causal distance does not match query/key ordinals"
            )
        if self.distance_bin != _distance_bin(self.causal_distance):
            raise ValueError("retained route distance bin drifted")
        probability = _nonnegative_float(
            self.probability, label="route probability"
        )
        if probability > 1.0 + 1e-6:
            raise ValueError("route probability exceeds one")
        response = _finite_float(
            self.signed_response, label="signed route response"
        )
        energy = _nonnegative_float(
            self.intrinsic_fisher_energy, label="route Fisher energy"
        )
        tangent = _finite_float(
            self.softmax_logit_tangent, label="softmax logit tangent"
        )
        if probability == 0.0 and any(
            value != 0.0 for value in (response, energy, tangent)
        ):
            raise ValueError("zero-probability route must have zero R/E/Z")
        cauchy_bound = HIDDEN_WIDTH * energy
        if response * response > cauchy_bound + _f64_slack(cauchy_bound):
            raise ValueError("retained route violates the Fisher Cauchy bound")

    def to_dict(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "layer_ordinal": self.layer_ordinal,
            "query_head_ordinal": self.query_head_ordinal,
            "query_ordinal": self.query_ordinal,
            "key_ordinal": self.key_ordinal,
            "causal_distance": self.causal_distance,
            "distance_bin": self.distance_bin,
            "target_rank": self.target_rank,
            "probability": self.probability,
            "signed_response": self.signed_response,
            "intrinsic_fisher_energy": self.intrinsic_fisher_energy,
            "softmax_logit_tangent": self.softmax_logit_tangent,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RetainedAttentionRoute":
        expected = {
            "row_ordinal",
            "layer_ordinal",
            "query_head_ordinal",
            "query_ordinal",
            "key_ordinal",
            "causal_distance",
            "distance_bin",
            "target_rank",
            "probability",
            "signed_response",
            "intrinsic_fisher_energy",
            "softmax_logit_tangent",
        }
        _exact_keys(value, expected, label="retained attention route")
        return cls(
            row_ordinal=_require_int(value["row_ordinal"], label="row_ordinal"),
            layer_ordinal=_require_int(
                value["layer_ordinal"], label="layer_ordinal"
            ),
            query_head_ordinal=_require_int(
                value["query_head_ordinal"], label="query_head_ordinal"
            ),
            query_ordinal=_require_int(
                value["query_ordinal"], label="query_ordinal"
            ),
            key_ordinal=_require_int(value["key_ordinal"], label="key_ordinal"),
            causal_distance=_require_int(
                value["causal_distance"], label="causal_distance"
            ),
            distance_bin=str(value["distance_bin"]),
            target_rank=_require_int(
                value["target_rank"], label="target_rank", minimum=1
            ),
            probability=_nonnegative_float(
                value["probability"], label="probability"
            ),
            signed_response=_finite_float(
                value["signed_response"], label="signed_response"
            ),
            intrinsic_fisher_energy=_nonnegative_float(
                value["intrinsic_fisher_energy"],
                label="intrinsic_fisher_energy",
            ),
            softmax_logit_tangent=_finite_float(
                value["softmax_logit_tangent"],
                label="softmax_logit_tangent",
            ),
        )


@dataclass(frozen=True, slots=True)
class AttentionDistanceMotif:
    layer_ordinal: int
    query_head_ordinal: int
    distance_bin: str
    full_target_count: int
    full_route_count: int
    retained_target_count: int
    retained_route_count: int
    probability_sum: float
    retained_probability_sum: float
    signed_response_sum: float
    absolute_signed_response_sum: float
    retained_signed_response_sum: float
    retained_absolute_signed_response_sum: float
    intrinsic_fisher_energy_sum: float
    retained_intrinsic_fisher_energy_sum: float
    maximum_intrinsic_fisher_energy: float
    softmax_logit_tangent_sum: float
    absolute_softmax_logit_tangent_sum: float
    retained_softmax_logit_tangent_sum: float
    retained_absolute_softmax_logit_tangent_sum: float

    def __post_init__(self) -> None:
        for label, value in (
            ("layer_ordinal", self.layer_ordinal),
            ("query_head_ordinal", self.query_head_ordinal),
            ("full_target_count", self.full_target_count),
            ("full_route_count", self.full_route_count),
            ("retained_target_count", self.retained_target_count),
            ("retained_route_count", self.retained_route_count),
        ):
            _require_int(value, label=label)
        if self.distance_bin not in ATTENTION_DISTANCE_BINS:
            raise ValueError("attention motif distance bin is invalid")
        if self.retained_route_count > self.full_route_count:
            raise ValueError("retained motif routes exceed full routes")
        if self.retained_target_count > self.full_target_count:
            raise ValueError("retained motif targets exceed full targets")
        if (
            self.full_target_count > self.full_route_count
            or self.retained_target_count > self.retained_route_count
        ):
            raise ValueError("attention motif targets exceed their route counts")
        for label, value in (
            ("probability_sum", self.probability_sum),
            ("retained_probability_sum", self.retained_probability_sum),
            ("absolute_signed_response_sum", self.absolute_signed_response_sum),
            (
                "retained_absolute_signed_response_sum",
                self.retained_absolute_signed_response_sum,
            ),
            ("intrinsic_fisher_energy_sum", self.intrinsic_fisher_energy_sum),
            (
                "retained_intrinsic_fisher_energy_sum",
                self.retained_intrinsic_fisher_energy_sum,
            ),
            (
                "maximum_intrinsic_fisher_energy",
                self.maximum_intrinsic_fisher_energy,
            ),
        ):
            _nonnegative_float(value, label=label)
        _finite_float(self.signed_response_sum, label="signed_response_sum")
        _finite_float(
            self.retained_signed_response_sum,
            label="retained_signed_response_sum",
        )
        for label, value in (
            ("softmax_logit_tangent_sum", self.softmax_logit_tangent_sum),
            (
                "retained_softmax_logit_tangent_sum",
                self.retained_softmax_logit_tangent_sum,
            ),
        ):
            _finite_float(value, label=label)
        for label, value in (
            (
                "absolute_softmax_logit_tangent_sum",
                self.absolute_softmax_logit_tangent_sum,
            ),
            (
                "retained_absolute_softmax_logit_tangent_sum",
                self.retained_absolute_softmax_logit_tangent_sum,
            ),
        ):
            _nonnegative_float(value, label=label)
        if (
            self.retained_intrinsic_fisher_energy_sum
            > self.intrinsic_fisher_energy_sum + 1e-12
        ):
            raise ValueError("retained motif energy exceeds full energy")
        if (
            self.retained_probability_sum
            > self.probability_sum
            + _f64_slack(self.probability_sum)
        ):
            raise ValueError("retained motif probability exceeds full probability")
        if (
            self.retained_absolute_signed_response_sum
            > self.absolute_signed_response_sum
            + _f64_slack(self.absolute_signed_response_sum)
        ):
            raise ValueError("retained motif response mass exceeds full mass")
        if (
            self.retained_absolute_softmax_logit_tangent_sum
            > self.absolute_softmax_logit_tangent_sum + 1e-12
        ):
            raise ValueError("retained motif tangent mass exceeds full mass")
        algebra = (
            (
                abs(self.signed_response_sum),
                self.absolute_signed_response_sum,
                "full response",
            ),
            (
                abs(self.retained_signed_response_sum),
                self.retained_absolute_signed_response_sum,
                "retained response",
            ),
            (
                abs(self.softmax_logit_tangent_sum),
                self.absolute_softmax_logit_tangent_sum,
                "full tangent",
            ),
            (
                abs(self.retained_softmax_logit_tangent_sum),
                self.retained_absolute_softmax_logit_tangent_sum,
                "retained tangent",
            ),
        )
        for signed_magnitude, absolute_mass, label in algebra:
            if signed_magnitude > absolute_mass + _f64_slack(absolute_mass):
                raise ValueError(f"attention motif {label} algebra drifted")
        if self.probability_sum > self.full_route_count + _f64_slack(
            float(self.full_route_count)
        ):
            raise ValueError("attention motif probability exceeds route count")
        if self.retained_probability_sum > (
            self.retained_route_count
            + _f64_slack(float(self.retained_route_count))
        ):
            raise ValueError("retained motif probability exceeds route count")
        if self.maximum_intrinsic_fisher_energy > (
            self.intrinsic_fisher_energy_sum
            + _f64_slack(self.intrinsic_fisher_energy_sum)
        ):
            raise ValueError("attention motif maximum energy exceeds its sum")
        aggregate_cauchy_bound = (
            self.full_route_count
            * HIDDEN_WIDTH
            * self.intrinsic_fisher_energy_sum
        )
        if self.absolute_signed_response_sum**2 > (
            aggregate_cauchy_bound + _f64_slack(aggregate_cauchy_bound)
        ):
            raise ValueError("attention motif violates aggregate Fisher Cauchy")
        full_scalars = (
            self.probability_sum,
            self.signed_response_sum,
            self.absolute_signed_response_sum,
            self.intrinsic_fisher_energy_sum,
            self.maximum_intrinsic_fisher_energy,
            self.softmax_logit_tangent_sum,
            self.absolute_softmax_logit_tangent_sum,
        )
        retained_scalars = (
            self.retained_probability_sum,
            self.retained_signed_response_sum,
            self.retained_absolute_signed_response_sum,
            self.retained_intrinsic_fisher_energy_sum,
            self.retained_softmax_logit_tangent_sum,
            self.retained_absolute_softmax_logit_tangent_sum,
        )
        if self.full_route_count == 0 and (
            self.full_target_count != 0
            or any(value != 0.0 for value in full_scalars)
        ):
            raise ValueError("empty full motif must have an exact zero payload")
        if self.full_route_count > 0 and self.full_target_count == 0:
            raise ValueError("nonempty full motif must have a target")
        if self.retained_route_count == 0 and (
            self.retained_target_count != 0
            or any(value != 0.0 for value in retained_scalars)
        ):
            raise ValueError("empty retained motif must have an exact zero payload")
        if self.retained_route_count > 0 and self.retained_target_count == 0:
            raise ValueError("nonempty retained motif must have a target")

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_ordinal": self.layer_ordinal,
            "query_head_ordinal": self.query_head_ordinal,
            "distance_bin": self.distance_bin,
            "full_target_count": self.full_target_count,
            "full_route_count": self.full_route_count,
            "retained_target_count": self.retained_target_count,
            "retained_route_count": self.retained_route_count,
            "probability_sum": self.probability_sum,
            "retained_probability_sum": self.retained_probability_sum,
            "signed_response_sum": self.signed_response_sum,
            "absolute_signed_response_sum": self.absolute_signed_response_sum,
            "retained_signed_response_sum": self.retained_signed_response_sum,
            "retained_absolute_signed_response_sum": (
                self.retained_absolute_signed_response_sum
            ),
            "intrinsic_fisher_energy_sum": self.intrinsic_fisher_energy_sum,
            "retained_intrinsic_fisher_energy_sum": (
                self.retained_intrinsic_fisher_energy_sum
            ),
            "maximum_intrinsic_fisher_energy": (
                self.maximum_intrinsic_fisher_energy
            ),
            "softmax_logit_tangent_sum": self.softmax_logit_tangent_sum,
            "absolute_softmax_logit_tangent_sum": (
                self.absolute_softmax_logit_tangent_sum
            ),
            "retained_softmax_logit_tangent_sum": (
                self.retained_softmax_logit_tangent_sum
            ),
            "retained_absolute_softmax_logit_tangent_sum": (
                self.retained_absolute_softmax_logit_tangent_sum
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AttentionDistanceMotif":
        expected = {
            "layer_ordinal",
            "query_head_ordinal",
            "distance_bin",
            "full_target_count",
            "full_route_count",
            "retained_target_count",
            "retained_route_count",
            "probability_sum",
            "retained_probability_sum",
            "signed_response_sum",
            "absolute_signed_response_sum",
            "retained_signed_response_sum",
            "retained_absolute_signed_response_sum",
            "intrinsic_fisher_energy_sum",
            "retained_intrinsic_fisher_energy_sum",
            "maximum_intrinsic_fisher_energy",
            "softmax_logit_tangent_sum",
            "absolute_softmax_logit_tangent_sum",
            "retained_softmax_logit_tangent_sum",
            "retained_absolute_softmax_logit_tangent_sum",
        }
        _exact_keys(value, expected, label="attention distance motif")
        ints = {
            name: _require_int(value[name], label=name)
            for name in (
                "layer_ordinal",
                "query_head_ordinal",
                "full_target_count",
                "full_route_count",
                "retained_target_count",
                "retained_route_count",
            )
        }
        return cls(
            distance_bin=str(value["distance_bin"]),
            probability_sum=_nonnegative_float(
                value["probability_sum"], label="probability_sum"
            ),
            retained_probability_sum=_nonnegative_float(
                value["retained_probability_sum"],
                label="retained_probability_sum",
            ),
            signed_response_sum=_finite_float(
                value["signed_response_sum"], label="signed_response_sum"
            ),
            absolute_signed_response_sum=_nonnegative_float(
                value["absolute_signed_response_sum"],
                label="absolute_signed_response_sum",
            ),
            retained_signed_response_sum=_finite_float(
                value["retained_signed_response_sum"],
                label="retained_signed_response_sum",
            ),
            retained_absolute_signed_response_sum=_nonnegative_float(
                value["retained_absolute_signed_response_sum"],
                label="retained_absolute_signed_response_sum",
            ),
            intrinsic_fisher_energy_sum=_nonnegative_float(
                value["intrinsic_fisher_energy_sum"],
                label="intrinsic_fisher_energy_sum",
            ),
            retained_intrinsic_fisher_energy_sum=_nonnegative_float(
                value["retained_intrinsic_fisher_energy_sum"],
                label="retained_intrinsic_fisher_energy_sum",
            ),
            maximum_intrinsic_fisher_energy=_nonnegative_float(
                value["maximum_intrinsic_fisher_energy"],
                label="maximum_intrinsic_fisher_energy",
            ),
            softmax_logit_tangent_sum=_finite_float(
                value["softmax_logit_tangent_sum"],
                label="softmax_logit_tangent_sum",
            ),
            absolute_softmax_logit_tangent_sum=_nonnegative_float(
                value["absolute_softmax_logit_tangent_sum"],
                label="absolute_softmax_logit_tangent_sum",
            ),
            retained_softmax_logit_tangent_sum=_finite_float(
                value["retained_softmax_logit_tangent_sum"],
                label="retained_softmax_logit_tangent_sum",
            ),
            retained_absolute_softmax_logit_tangent_sum=_nonnegative_float(
                value["retained_absolute_softmax_logit_tangent_sum"],
                label="retained_absolute_softmax_logit_tangent_sum",
            ),
            **ints,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AttentionEdgeAtlasRowReceipt:
    row_ordinal: int
    family_id: str
    sequence_length: int
    supervised_query_ordinal: int
    tokenization_sha256: str
    layer_extraction_sha256s: tuple[str, ...]
    all_layer_closures_passed: bool
    full_allowed_route_count: int
    retained_route_count: int
    full_intrinsic_fisher_energy: float
    retained_intrinsic_fisher_energy: float
    row_sha256: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("row_ordinal", self.row_ordinal),
            ("sequence_length", self.sequence_length),
            ("supervised_query_ordinal", self.supervised_query_ordinal),
            ("full_allowed_route_count", self.full_allowed_route_count),
            ("retained_route_count", self.retained_route_count),
        ):
            _require_int(value, label=label)
        if not self.family_id:
            raise ValueError("attention atlas family_id must be nonempty")
        if self.sequence_length <= 0 or not (
            0 <= self.supervised_query_ordinal < self.sequence_length
        ):
            raise ValueError("attention atlas query position is invalid")
        if not self.layer_extraction_sha256s:
            raise ValueError("attention atlas row omitted layer extractions")
        _require_sha256(self.tokenization_sha256, label="row tokenization")
        for value in self.layer_extraction_sha256s:
            _require_sha256(value, label="layer extraction")
        _require_bool(
            self.all_layer_closures_passed,
            label="all_layer_closures_passed",
        )
        if self.retained_route_count > self.full_allowed_route_count:
            raise ValueError("row retained route count exceeds full count")
        full = _nonnegative_float(
            self.full_intrinsic_fisher_energy,
            label="full_intrinsic_fisher_energy",
        )
        retained = _nonnegative_float(
            self.retained_intrinsic_fisher_energy,
            label="retained_intrinsic_fisher_energy",
        )
        if retained > full + 1e-12:
            raise ValueError("row retained Fisher energy exceeds full energy")
        computed = _json_sha256(self._payload(), domain=_ROW_DOMAIN)
        if self.row_sha256 == "":
            object.__setattr__(self, "row_sha256", computed)
        elif _require_sha256(self.row_sha256, label="row_sha256") != computed:
            raise ValueError("attention atlas row hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "family_id": self.family_id,
            "sequence_length": self.sequence_length,
            "supervised_query_ordinal": self.supervised_query_ordinal,
            "tokenization_sha256": self.tokenization_sha256,
            "layer_extraction_sha256s": list(self.layer_extraction_sha256s),
            "all_layer_closures_passed": self.all_layer_closures_passed,
            "full_allowed_route_count": self.full_allowed_route_count,
            "retained_route_count": self.retained_route_count,
            "full_intrinsic_fisher_energy": self.full_intrinsic_fisher_energy,
            "retained_intrinsic_fisher_energy": (
                self.retained_intrinsic_fisher_energy
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "row_sha256": self.row_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeAtlasRowReceipt":
        _exact_keys(
            value,
            {
                "row_ordinal",
                "family_id",
                "sequence_length",
                "supervised_query_ordinal",
                "tokenization_sha256",
                "layer_extraction_sha256s",
                "all_layer_closures_passed",
                "full_allowed_route_count",
                "retained_route_count",
                "full_intrinsic_fisher_energy",
                "retained_intrinsic_fisher_energy",
                "row_sha256",
            },
            label="attention atlas row receipt",
        )
        hashes = value["layer_extraction_sha256s"]
        if not isinstance(hashes, list):
            raise ValueError("layer extraction hashes must be a list")
        return cls(
            row_ordinal=_require_int(value["row_ordinal"], label="row_ordinal"),
            family_id=str(value["family_id"]),
            sequence_length=_require_int(
                value["sequence_length"], label="sequence_length", minimum=1
            ),
            supervised_query_ordinal=_require_int(
                value["supervised_query_ordinal"],
                label="supervised_query_ordinal",
            ),
            tokenization_sha256=str(value["tokenization_sha256"]),
            layer_extraction_sha256s=tuple(str(item) for item in hashes),
            all_layer_closures_passed=_require_bool(
                value["all_layer_closures_passed"],
                label="all_layer_closures_passed",
            ),
            full_allowed_route_count=_require_int(
                value["full_allowed_route_count"],
                label="full_allowed_route_count",
            ),
            retained_route_count=_require_int(
                value["retained_route_count"], label="retained_route_count"
            ),
            full_intrinsic_fisher_energy=_nonnegative_float(
                value["full_intrinsic_fisher_energy"],
                label="full_intrinsic_fisher_energy",
            ),
            retained_intrinsic_fisher_energy=_nonnegative_float(
                value["retained_intrinsic_fisher_energy"],
                label="retained_intrinsic_fisher_energy",
            ),
            row_sha256=str(value["row_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class AttentionEdgeAtlasRowCapture:
    """Ephemeral full-route inputs for one prompt; never serialized."""

    row_catalog_id: str
    family_id: str
    logical_positions: tuple[int, ...]
    supervised_query_ordinal: int
    tokenization_sha256: str
    layer_extractions: tuple[PartsFirstAttentionEdgeExtraction, ...]

    def __post_init__(self) -> None:
        if not self.row_catalog_id or not self.family_id:
            raise ValueError("attention edge capture identity is incomplete")
        if (
            not self.logical_positions
            or any(type(item) is not int for item in self.logical_positions)
            or len(set(self.logical_positions)) != len(self.logical_positions)
        ):
            raise ValueError("attention edge logical positions are invalid")
        if not 0 <= self.supervised_query_ordinal < len(self.logical_positions):
            raise ValueError("supervised query ordinal is invalid")
        _require_sha256(self.tokenization_sha256, label="capture tokenization")
        if not self.layer_extractions:
            raise ValueError("attention edge row omitted layer extractions")


class _MotifAccumulator:
    __slots__ = (
        "full_targets",
        "retained_targets",
        "full_route_count",
        "retained_route_count",
        "probabilities",
        "signed",
        "absolute_signed",
        "energies",
        "retained_energies",
        "maximum_energy",
        "tangents",
        "absolute_tangents",
        "retained_probabilities",
        "retained_signed",
        "retained_absolute_signed",
        "retained_tangents",
        "retained_absolute_tangents",
    )

    def __init__(self) -> None:
        self.full_targets = 0
        self.retained_targets = 0
        self.full_route_count = 0
        self.retained_route_count = 0
        self.probabilities: list[float] = []
        self.signed: list[float] = []
        self.absolute_signed: list[float] = []
        self.energies: list[float] = []
        self.retained_energies: list[float] = []
        self.maximum_energy = 0.0
        self.tangents: list[float] = []
        self.absolute_tangents: list[float] = []
        self.retained_probabilities: list[float] = []
        self.retained_signed: list[float] = []
        self.retained_absolute_signed: list[float] = []
        self.retained_tangents: list[float] = []
        self.retained_absolute_tangents: list[float] = []


@dataclass(frozen=True, slots=True)
class AttentionEdgeAtlasReport:
    source_kind: str
    model_id: str
    model_revision: str
    panel_id: str
    panel_file_sha256: str
    row_catalog_sha256: str
    tokenization_catalog_sha256: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    score_objective_id: str
    core_implementation_sha256: str
    runner_implementation_sha256: str
    implementation_bundle_sha256: str
    exact_control_receipt: AttentionEdgeExactControlReceipt | None
    config: AttentionEdgeAtlasConfig
    row_count: int
    family_counts: tuple[tuple[str, int], ...]
    layer_count: int
    query_head_count: int
    row_receipts: tuple[AttentionEdgeAtlasRowReceipt, ...]
    retained_routes: tuple[RetainedAttentionRoute, ...]
    distance_motifs: tuple[AttentionDistanceMotif, ...]
    full_allowed_route_count: int
    retained_route_count: int
    full_intrinsic_fisher_energy: float
    retained_intrinsic_fisher_energy: float
    old_roles_collapsed: bool = True
    roles_statistically_read: tuple[str, ...] = (ATTENTION_EDGE_ATLAS_ROLE,)
    retrospective_A_only: bool = True
    structure_only: bool = True
    B_confirmation_claimed: bool = False
    C_intervention_claimed: bool = False
    lexical_mode_confirmation_claimed: bool = False
    attention_edge_confirmation_claimed: bool = False
    attention_promotion_authorized: bool = False
    attention_promotion_performed: bool = False
    compression_attempted: bool = False
    compression_claimed: bool = False
    parameter_savings_claimed: bool = False
    mac_savings_claimed: bool = False
    latency_claimed: bool = False
    semantic_interpretation_claimed: bool = False
    prompts_persisted: bool = False
    token_ids_persisted: bool = False
    raw_activations_persisted: bool = False
    raw_gradients_persisted: bool = False
    schema: str = ATTENTION_EDGE_ATLAS_SCHEMA
    format_version: int = ATTENTION_EDGE_ATLAS_FORMAT_VERSION
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if self.source_kind not in {
            "model_capture",
            "model_free_fixture",
            "model_smoke",
        }:
            raise ValueError("attention atlas source kind is invalid")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.model_id,
                self.model_revision,
                self.panel_id,
                self.score_objective_id,
            )
        ):
            raise ValueError("attention atlas source metadata is incomplete")
        for label, value in (
            ("panel_file_sha256", self.panel_file_sha256),
            ("row_catalog_sha256", self.row_catalog_sha256),
            (
                "tokenization_catalog_sha256",
                self.tokenization_catalog_sha256,
            ),
            (
                "source_model_fingerprint_sha256",
                self.source_model_fingerprint_sha256,
            ),
            (
                "source_execution_fingerprint_sha256",
                self.source_execution_fingerprint_sha256,
            ),
            ("core_implementation_sha256", self.core_implementation_sha256),
            ("runner_implementation_sha256", self.runner_implementation_sha256),
            ("implementation_bundle_sha256", self.implementation_bundle_sha256),
        ):
            _require_sha256(value, label=label)
        expected_implementation_bundle = _json_sha256(
            {
                "core_implementation_sha256": self.core_implementation_sha256,
                "runner_implementation_sha256": self.runner_implementation_sha256,
            },
            domain=_IMPLEMENTATION_DOMAIN,
        )
        if self.implementation_bundle_sha256 != expected_implementation_bundle:
            raise ValueError("attention atlas implementation bundle drifted")
        if not isinstance(self.config, AttentionEdgeAtlasConfig):
            raise TypeError("attention atlas config is invalid")
        if self.source_kind in {"model_capture", "model_smoke"}:
            if (
                self.core_implementation_sha256,
                self.runner_implementation_sha256,
                self.implementation_bundle_sha256,
            ) != _implementation_provenance():
                raise ValueError(
                    "published model atlas implementation provenance drifted"
                )
            if (
                self.model_id != MODEL_ID
                or self.model_revision != PINNED_MODEL_REVISION
                or self.panel_id != PINNED_PANEL_ID
                or self.panel_file_sha256 != PINNED_PANEL_FILE_SHA256
                or self.source_model_fingerprint_sha256
                != SOURCE_MODEL_FINGERPRINT_SHA256
                or self.source_execution_fingerprint_sha256
                != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
                or self.score_objective_id != MODEL_SCORE_OBJECTIVE_ID
                or self.layer_count != GEMMA3_LAYER_COUNT
                or self.query_head_count != ATTENTION_HEAD_COUNT
                or self.exact_control_receipt is None
                or not self.exact_control_receipt.passed
                or self.exact_control_receipt.source_model_fingerprint_before
                != self.source_model_fingerprint_sha256
                or self.exact_control_receipt.source_execution_fingerprint_before
                != self.source_execution_fingerprint_sha256
            ):
                raise ValueError("published model atlas provenance drifted")
            if self.source_kind == "model_capture" and (
                self.row_count != 60
                or self.row_catalog_sha256
                != PINNED_ATTENTION_EDGE_ROW_CATALOG_SHA256
                or self.tokenization_catalog_sha256
                != PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256
                or self.family_counts
                != (
                    ("category_completion", 10),
                    ("integer_addition", 10),
                    ("integer_comparison", 10),
                    ("integer_subtraction", 10),
                    ("lexical_antonym", 10),
                    ("world_capitals", 10),
                )
            ):
                raise ValueError("published model atlas row catalog drifted")
            if self.source_kind == "model_smoke" and (
                self.row_count != 1
                or self.row_catalog_sha256
                != PINNED_ATTENTION_EDGE_SMOKE_ROW_CATALOG_SHA256
                or self.tokenization_catalog_sha256
                != PINNED_ATTENTION_EDGE_SMOKE_TOKENIZATION_CATALOG_SHA256
                or self.family_counts != (("integer_addition", 1),)
            ):
                raise ValueError("model smoke row catalog drifted")
        elif self.exact_control_receipt is not None:
            raise ValueError("model-free fixture cannot claim an exact control")
        for label, value in (
            ("row_count", self.row_count),
            ("layer_count", self.layer_count),
            ("query_head_count", self.query_head_count),
            ("full_allowed_route_count", self.full_allowed_route_count),
            ("retained_route_count", self.retained_route_count),
        ):
            _require_int(value, label=label, minimum=1)
        if len(self.row_receipts) != self.row_count or tuple(
            row.row_ordinal for row in self.row_receipts
        ) != tuple(range(self.row_count)):
            raise ValueError("attention atlas row receipt inventory drifted")
        if any(
            len(row.layer_extraction_sha256s) != self.layer_count
            for row in self.row_receipts
        ):
            raise ValueError("attention atlas row layer inventory drifted")
        if any(
            not row.all_layer_closures_passed
            or row.retained_route_count
            > row.sequence_length * self.layer_count * ATTENTION_EDGE_TOP_K
            for row in self.row_receipts
        ):
            raise ValueError("attention atlas row closure or sparse cap drifted")
        if self.source_kind in {"model_capture", "model_smoke"}:
            for row in self.row_receipts:
                length = row.sequence_length
                expected_full = (
                    self.layer_count
                    * self.query_head_count
                    * length
                    * (length + 1)
                    // 2
                )
                expected_retained = self.layer_count * sum(
                    min(
                        ATTENTION_EDGE_TOP_K,
                        self.query_head_count * (query + 1),
                    )
                    for query in range(length)
                )
                if (
                    length > ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS
                    or row.supervised_query_ordinal != length - 1
                    or row.full_allowed_route_count != expected_full
                    or row.retained_route_count != expected_retained
                ):
                    raise ValueError(
                        "published model atlas row causal geometry drifted"
                    )
        total_valid_tokens = sum(
            row.sequence_length for row in self.row_receipts
        )
        if self.source_kind == "model_capture" and (
            total_valid_tokens != PINNED_ATTENTION_EDGE_VALID_TOKEN_COUNT
            or self.retained_route_count
            > PINNED_ATTENTION_EDGE_MAXIMUM_RETAINED_ROUTE_COUNT
        ):
            raise ValueError("published model atlas token/sparse bound drifted")
        if (
            self.source_kind == "model_smoke"
            and total_valid_tokens
            != PINNED_ATTENTION_EDGE_SMOKE_VALID_TOKEN_COUNT
        ):
            raise ValueError("model smoke token count drifted")
        if self.tokenization_catalog_sha256 != _tokenization_catalog_sha256(
            tuple(row.tokenization_sha256 for row in self.row_receipts)
        ):
            raise ValueError("attention atlas tokenization catalog drifted")
        if sum(count for _, count in self.family_counts) != self.row_count:
            raise ValueError("attention atlas family counts do not close")
        if Counter(
            row.family_id for row in self.row_receipts
        ) != Counter(dict(self.family_counts)):
            raise ValueError("attention atlas row families do not close")
        if tuple(name for name, _ in self.family_counts) != tuple(
            sorted(name for name, _ in self.family_counts)
        ):
            raise ValueError("attention atlas family counts are not sorted")
        if len({name for name, _ in self.family_counts}) != len(self.family_counts):
            raise ValueError("attention atlas family count keys repeat")
        expected_motifs = tuple(
            (layer, head, bin_id)
            for layer in range(self.layer_count)
            for head in range(self.query_head_count)
            for bin_id in ATTENTION_DISTANCE_BINS
        )
        if tuple(
            (item.layer_ordinal, item.query_head_ordinal, item.distance_bin)
            for item in self.distance_motifs
        ) != expected_motifs:
            raise ValueError("attention atlas motif grid drifted")
        route_order = tuple(
            (
                route.row_ordinal,
                route.layer_ordinal,
                route.query_ordinal,
                route.target_rank,
            )
            for route in self.retained_routes
        )
        if route_order != tuple(sorted(route_order)):
            raise ValueError("retained attention routes are not canonical")
        targets: dict[tuple[int, int, int], list[RetainedAttentionRoute]] = {}
        retained_by_motif: dict[
            tuple[int, int, str], list[RetainedAttentionRoute]
        ] = {}
        retained_by_row: dict[int, list[RetainedAttentionRoute]] = {}
        retained_by_head_query: dict[
            tuple[int, int, int, int], list[RetainedAttentionRoute]
        ] = {}
        retained_coordinates: set[tuple[int, int, int, int, int]] = set()
        for route in self.retained_routes:
            if (
                route.row_ordinal >= self.row_count
                or route.layer_ordinal >= self.layer_count
                or route.query_head_ordinal >= self.query_head_count
                or route.query_ordinal
                >= self.row_receipts[route.row_ordinal].sequence_length
                or route.key_ordinal
                >= self.row_receipts[route.row_ordinal].sequence_length
            ):
                raise ValueError("retained attention route is out of bounds")
            key = (
                route.row_ordinal,
                route.layer_ordinal,
                route.query_ordinal,
            )
            coordinate = (
                *key,
                route.query_head_ordinal,
                route.key_ordinal,
            )
            if coordinate in retained_coordinates:
                raise ValueError("retained attention route coordinate repeats")
            retained_coordinates.add(coordinate)
            targets.setdefault(key, []).append(route)
            retained_by_motif.setdefault(
                (
                    route.layer_ordinal,
                    route.query_head_ordinal,
                    route.distance_bin,
                ),
                [],
            ).append(route)
            retained_by_row.setdefault(route.row_ordinal, []).append(route)
            retained_by_head_query.setdefault(
                (
                    route.row_ordinal,
                    route.layer_ordinal,
                    route.query_head_ordinal,
                    route.query_ordinal,
                ),
                [],
            ).append(route)
        for routes in targets.values():
            if len(routes) > ATTENTION_EDGE_TOP_K:
                raise ValueError("attention atlas per-target sparse cap drifted")
            if [item.target_rank for item in routes] != list(
                range(1, len(routes) + 1)
            ):
                raise ValueError("retained route ranks are not contiguous")
            ranking = [
                (
                    -item.intrinsic_fisher_energy,
                    item.query_head_ordinal,
                    item.key_ordinal,
                )
                for item in routes
            ]
            if ranking != sorted(ranking):
                raise ValueError("retained routes violate the frozen ranking")
        if self.source_kind in {"model_capture", "model_smoke"}:
            for row in self.row_receipts:
                for layer in range(self.layer_count):
                    for query in range(row.sequence_length):
                        expected = min(
                            ATTENTION_EDGE_TOP_K,
                            self.query_head_count * (query + 1),
                        )
                        if len(
                            targets.get((row.row_ordinal, layer, query), ())
                        ) != expected:
                            raise ValueError(
                                "published model atlas per-query top-k geometry "
                                "drifted"
                            )
        for key, routes in retained_by_head_query.items():
            probability_sum = math.fsum(route.probability for route in routes)
            row_length = self.row_receipts[key[0]].sequence_length
            probability_limit = (
                1.0
                + _probability_forward_error_bound(row_length)
                + len(routes) * _FLOAT64_UNIT_ROUNDOFF
            )
            if probability_sum > probability_limit:
                raise ValueError(
                    "retained head/query probability mass exceeds one"
                )
            positive = tuple(route for route in routes if route.probability > 0.0)
            if len(positive) >= 2:
                reference = max(positive, key=lambda route: route.probability)
                reference_response = (
                    reference.signed_response
                    - reference.softmax_logit_tangent
                )
                for route in positive:
                    lhs = (
                        route.signed_response
                        - route.softmax_logit_tangent
                    ) * reference.probability
                    rhs = reference_response * route.probability
                    if not math.isclose(
                        lhs,
                        rhs,
                        rel_tol=256 * _FLOAT64_UNIT_ROUNDOFF,
                        abs_tol=256 * _FLOAT64_UNIT_ROUNDOFF,
                    ):
                        raise ValueError(
                            "retained routes disagree on shared softmax response"
                        )
        for row in self.row_receipts:
            routes = retained_by_row.get(row.row_ordinal, [])
            if row.retained_route_count != len(routes) or not math.isclose(
                row.retained_intrinsic_fisher_energy,
                math.fsum(
                    route.intrinsic_fisher_energy for route in routes
                ),
                rel_tol=2e-13,
                abs_tol=1e-12,
            ):
                raise ValueError("retained row route ledger does not close")
        for motif in self.distance_motifs:
            routes = retained_by_motif.get(
                (
                    motif.layer_ordinal,
                    motif.query_head_ordinal,
                    motif.distance_bin,
                ),
                [],
            )
            target_count = len(
                {
                    (route.row_ordinal, route.query_ordinal)
                    for route in routes
                }
            )
            motif_closures = (
                (motif.retained_route_count, len(routes)),
                (motif.retained_target_count, target_count),
            )
            if any(actual != expected for actual, expected in motif_closures):
                raise ValueError("retained motif inventory does not close")
            scalar_closures = (
                (
                    motif.retained_probability_sum,
                    math.fsum(route.probability for route in routes),
                ),
                (
                    motif.retained_signed_response_sum,
                    math.fsum(route.signed_response for route in routes),
                ),
                (
                    motif.retained_absolute_signed_response_sum,
                    math.fsum(abs(route.signed_response) for route in routes),
                ),
                (
                    motif.retained_intrinsic_fisher_energy_sum,
                    math.fsum(
                        route.intrinsic_fisher_energy for route in routes
                    ),
                ),
                (
                    motif.retained_softmax_logit_tangent_sum,
                    math.fsum(route.softmax_logit_tangent for route in routes),
                ),
                (
                    motif.retained_absolute_softmax_logit_tangent_sum,
                    math.fsum(
                        abs(route.softmax_logit_tangent) for route in routes
                    ),
                ),
            )
            if any(
                not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-12)
                for actual, expected in scalar_closures
            ):
                raise ValueError("retained motif scalar ledgers do not close")
            maximum_retained_energy = max(
                (route.intrinsic_fisher_energy for route in routes),
                default=0.0,
            )
            if maximum_retained_energy > (
                motif.maximum_intrinsic_fisher_energy
                + _f64_slack(motif.maximum_intrinsic_fisher_energy)
            ):
                raise ValueError("retained route exceeds motif maximum energy")
        if self.source_kind in {"model_capture", "model_smoke"}:
            expected_full_routes = Counter(
                {
                    bin_id: 0
                    for bin_id in ATTENTION_DISTANCE_BINS
                }
            )
            expected_full_targets = Counter(
                {
                    bin_id: 0
                    for bin_id in ATTENTION_DISTANCE_BINS
                }
            )
            for row in self.row_receipts:
                for query in range(row.sequence_length):
                    seen_bins: set[str] = set()
                    for key_ordinal in range(query + 1):
                        bin_id = _distance_bin(query - key_ordinal)
                        expected_full_routes[bin_id] += 1
                        seen_bins.add(bin_id)
                    for bin_id in seen_bins:
                        expected_full_targets[bin_id] += 1
            for motif in self.distance_motifs:
                if (
                    motif.full_route_count
                    != expected_full_routes[motif.distance_bin]
                    or motif.full_target_count
                    != expected_full_targets[motif.distance_bin]
                ):
                    raise ValueError(
                        "published model atlas motif causal geometry drifted"
                    )
        if len(self.retained_routes) != self.retained_route_count:
            raise ValueError("retained attention route count does not close")
        if sum(row.full_allowed_route_count for row in self.row_receipts) != (
            self.full_allowed_route_count
        ):
            raise ValueError("full allowed route count does not close")
        if sum(row.retained_route_count for row in self.row_receipts) != (
            self.retained_route_count
        ):
            raise ValueError("retained row route count does not close")
        if sum(item.full_route_count for item in self.distance_motifs) != (
            self.full_allowed_route_count
        ):
            raise ValueError("full motif route count does not close")
        if sum(item.retained_route_count for item in self.distance_motifs) != (
            self.retained_route_count
        ):
            raise ValueError("retained motif route count does not close")
        expected_probability_mass = float(
            sum(row.sequence_length for row in self.row_receipts)
            * self.layer_count
            * self.query_head_count
        )
        maximum_sequence_length = max(
            row.sequence_length for row in self.row_receipts
        )
        probability_bound = _probability_forward_error_bound(
            maximum_sequence_length
        )
        for layer in range(self.layer_count):
            for head in range(self.query_head_count):
                group = tuple(
                    motif
                    for motif in self.distance_motifs
                    if motif.layer_ordinal == layer
                    and motif.query_head_ordinal == head
                )
                probability_sum = math.fsum(
                    motif.probability_sum for motif in group
                )
                probability_limit = (
                    total_valid_tokens * probability_bound
                    + 64
                    * _FLOAT64_UNIT_ROUNDOFF
                    * max(1.0, float(total_valid_tokens))
                )
                if abs(probability_sum - total_valid_tokens) > probability_limit:
                    raise ValueError(
                        "attention atlas head probability mass does not close"
                    )
                absolute_response = math.fsum(
                    motif.absolute_signed_response_sum for motif in group
                )
                signed_tangent = math.fsum(
                    motif.softmax_logit_tangent_sum for motif in group
                )
                absolute_tangent = math.fsum(
                    motif.absolute_softmax_logit_tangent_sum for motif in group
                )
                tangent_limit = (
                    probability_bound * absolute_response
                    + maximum_sequence_length
                    * _FLOAT64_UNIT_ROUNDOFF
                    * absolute_tangent
                    + total_valid_tokens * _FLOAT64_UNIT_ROUNDOFF
                    + 64
                    * _FLOAT64_UNIT_ROUNDOFF
                    * max(1.0, absolute_tangent)
                )
                if abs(signed_tangent) > tangent_limit:
                    raise ValueError(
                        "attention atlas head softmax tangent does not close"
                    )
                absolute_tangent_limit = (
                    (2.0 + probability_bound) * absolute_response
                    + tangent_limit
                    + _f64_slack(absolute_tangent, absolute_response)
                )
                if absolute_tangent > absolute_tangent_limit:
                    raise ValueError(
                        "attention atlas head tangent mass exceeds response bound"
                    )
        if not math.isclose(
            math.fsum(
                motif.probability_sum for motif in self.distance_motifs
            ),
            expected_probability_mass,
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError("attention atlas probability mass does not close")
        full_energy = _nonnegative_float(
            self.full_intrinsic_fisher_energy,
            label="full_intrinsic_fisher_energy",
        )
        retained_energy = _nonnegative_float(
            self.retained_intrinsic_fisher_energy,
            label="retained_intrinsic_fisher_energy",
        )
        if retained_energy > full_energy + 1e-12:
            raise ValueError("retained atlas energy exceeds full energy")
        energy_closures = (
            (
                math.fsum(
                    row.full_intrinsic_fisher_energy
                    for row in self.row_receipts
                ),
                full_energy,
            ),
            (
                math.fsum(
                    motif.intrinsic_fisher_energy_sum
                    for motif in self.distance_motifs
                ),
                full_energy,
            ),
            (
                math.fsum(
                    row.retained_intrinsic_fisher_energy
                    for row in self.row_receipts
                ),
                retained_energy,
            ),
            (
                math.fsum(
                    motif.retained_intrinsic_fisher_energy_sum
                    for motif in self.distance_motifs
                ),
                retained_energy,
            ),
            (
                math.fsum(
                    route.intrinsic_fisher_energy
                    for route in self.retained_routes
                ),
                retained_energy,
            ),
        )
        if any(
            not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-12)
            for actual, expected in energy_closures
        ):
            raise ValueError("attention atlas Fisher-energy ledgers do not close")
        bool_fields = (
            "old_roles_collapsed",
            "retrospective_A_only",
            "structure_only",
            "B_confirmation_claimed",
            "C_intervention_claimed",
            "lexical_mode_confirmation_claimed",
            "attention_edge_confirmation_claimed",
            "attention_promotion_authorized",
            "attention_promotion_performed",
            "compression_attempted",
            "compression_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
            "semantic_interpretation_claimed",
            "prompts_persisted",
            "token_ids_persisted",
            "raw_activations_persisted",
            "raw_gradients_persisted",
        )
        for name in bool_fields:
            _require_bool(getattr(self, name), label=name)
        if (
            not self.old_roles_collapsed
            or self.roles_statistically_read != (ATTENTION_EDGE_ATLAS_ROLE,)
            or not self.retrospective_A_only
            or not self.structure_only
            or any(
                getattr(self, name)
                for name in bool_fields[3:]
            )
            or self.schema != ATTENTION_EDGE_ATLAS_SCHEMA
            or self.format_version != ATTENTION_EDGE_ATLAS_FORMAT_VERSION
        ):
            raise ValueError("attention atlas fail-closed claim boundary drifted")
        computed = _json_sha256(self._payload(), domain=_REPORT_DOMAIN)
        if self.report_sha256 == "":
            object.__setattr__(self, "report_sha256", computed)
        elif _require_sha256(self.report_sha256, label="report_sha256") != computed:
            raise ValueError("attention atlas report hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "panel_id": self.panel_id,
            "panel_file_sha256": self.panel_file_sha256,
            "row_catalog_sha256": self.row_catalog_sha256,
            "tokenization_catalog_sha256": self.tokenization_catalog_sha256,
            "source_model_fingerprint_sha256": (
                self.source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "score_objective_id": self.score_objective_id,
            "core_implementation_sha256": self.core_implementation_sha256,
            "runner_implementation_sha256": self.runner_implementation_sha256,
            "implementation_bundle_sha256": self.implementation_bundle_sha256,
            "exact_control_receipt": (
                None
                if self.exact_control_receipt is None
                else self.exact_control_receipt.to_dict()
            ),
            "config": self.config.to_dict(),
            "row_count": self.row_count,
            "family_counts": [list(item) for item in self.family_counts],
            "layer_count": self.layer_count,
            "query_head_count": self.query_head_count,
            "row_receipts": [item.to_dict() for item in self.row_receipts],
            "retained_routes": [item.to_dict() for item in self.retained_routes],
            "distance_motifs": [item.to_dict() for item in self.distance_motifs],
            "full_allowed_route_count": self.full_allowed_route_count,
            "retained_route_count": self.retained_route_count,
            "full_intrinsic_fisher_energy": self.full_intrinsic_fisher_energy,
            "retained_intrinsic_fisher_energy": (
                self.retained_intrinsic_fisher_energy
            ),
            "old_roles_collapsed": self.old_roles_collapsed,
            "roles_statistically_read": list(self.roles_statistically_read),
            "retrospective_A_only": self.retrospective_A_only,
            "structure_only": self.structure_only,
            "B_confirmation_claimed": self.B_confirmation_claimed,
            "C_intervention_claimed": self.C_intervention_claimed,
            "lexical_mode_confirmation_claimed": (
                self.lexical_mode_confirmation_claimed
            ),
            "attention_edge_confirmation_claimed": (
                self.attention_edge_confirmation_claimed
            ),
            "attention_promotion_authorized": (
                self.attention_promotion_authorized
            ),
            "attention_promotion_performed": (
                self.attention_promotion_performed
            ),
            "compression_attempted": self.compression_attempted,
            "compression_claimed": self.compression_claimed,
            "parameter_savings_claimed": self.parameter_savings_claimed,
            "mac_savings_claimed": self.mac_savings_claimed,
            "latency_claimed": self.latency_claimed,
            "semantic_interpretation_claimed": (
                self.semantic_interpretation_claimed
            ),
            "prompts_persisted": self.prompts_persisted,
            "token_ids_persisted": self.token_ids_persisted,
            "raw_activations_persisted": self.raw_activations_persisted,
            "raw_gradients_persisted": self.raw_gradients_persisted,
            "schema": self.schema,
            "format_version": self.format_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "report_sha256": self.report_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AttentionEdgeAtlasReport":
        expected = {
            "source_kind",
            "model_id",
            "model_revision",
            "panel_id",
            "panel_file_sha256",
            "row_catalog_sha256",
            "tokenization_catalog_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "score_objective_id",
            "core_implementation_sha256",
            "runner_implementation_sha256",
            "implementation_bundle_sha256",
            "exact_control_receipt",
            "config",
            "row_count",
            "family_counts",
            "layer_count",
            "query_head_count",
            "row_receipts",
            "retained_routes",
            "distance_motifs",
            "full_allowed_route_count",
            "retained_route_count",
            "full_intrinsic_fisher_energy",
            "retained_intrinsic_fisher_energy",
            "old_roles_collapsed",
            "roles_statistically_read",
            "retrospective_A_only",
            "structure_only",
            "B_confirmation_claimed",
            "C_intervention_claimed",
            "lexical_mode_confirmation_claimed",
            "attention_edge_confirmation_claimed",
            "attention_promotion_authorized",
            "attention_promotion_performed",
            "compression_attempted",
            "compression_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
            "semantic_interpretation_claimed",
            "prompts_persisted",
            "token_ids_persisted",
            "raw_activations_persisted",
            "raw_gradients_persisted",
            "schema",
            "format_version",
            "report_sha256",
        }
        _exact_keys(value, expected, label="attention edge atlas report")
        config = value["config"]
        exact = value["exact_control_receipt"]
        rows = value["row_receipts"]
        routes = value["retained_routes"]
        motifs = value["distance_motifs"]
        families = value["family_counts"]
        roles = value["roles_statistically_read"]
        if (
            not isinstance(config, Mapping)
            or (exact is not None and not isinstance(exact, Mapping))
            or not isinstance(rows, list)
            or not all(isinstance(item, Mapping) for item in rows)
            or not isinstance(routes, list)
            or not all(isinstance(item, Mapping) for item in routes)
            or not isinstance(motifs, list)
            or not all(isinstance(item, Mapping) for item in motifs)
            or not isinstance(families, list)
            or not isinstance(roles, list)
        ):
            raise ValueError("attention atlas nested fields are invalid")
        family_rows: list[tuple[str, int]] = []
        for item in families:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("attention atlas family count row is invalid")
            family_rows.append(
                (str(item[0]), _require_int(item[1], label="family count"))
            )
        bool_names = (
            "old_roles_collapsed",
            "retrospective_A_only",
            "structure_only",
            "B_confirmation_claimed",
            "C_intervention_claimed",
            "lexical_mode_confirmation_claimed",
            "attention_edge_confirmation_claimed",
            "attention_promotion_authorized",
            "attention_promotion_performed",
            "compression_attempted",
            "compression_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
            "semantic_interpretation_claimed",
            "prompts_persisted",
            "token_ids_persisted",
            "raw_activations_persisted",
            "raw_gradients_persisted",
        )
        kwargs = {
            name: _require_bool(value[name], label=name) for name in bool_names
        }
        return cls(
            source_kind=str(value["source_kind"]),
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            panel_id=str(value["panel_id"]),
            panel_file_sha256=str(value["panel_file_sha256"]),
            row_catalog_sha256=str(value["row_catalog_sha256"]),
            tokenization_catalog_sha256=str(
                value["tokenization_catalog_sha256"]
            ),
            source_model_fingerprint_sha256=str(
                value["source_model_fingerprint_sha256"]
            ),
            source_execution_fingerprint_sha256=str(
                value["source_execution_fingerprint_sha256"]
            ),
            score_objective_id=str(value["score_objective_id"]),
            core_implementation_sha256=str(
                value["core_implementation_sha256"]
            ),
            runner_implementation_sha256=str(
                value["runner_implementation_sha256"]
            ),
            implementation_bundle_sha256=str(
                value["implementation_bundle_sha256"]
            ),
            exact_control_receipt=(
                None
                if exact is None
                else AttentionEdgeExactControlReceipt.from_dict(exact)
            ),
            config=AttentionEdgeAtlasConfig.from_dict(config),
            row_count=_require_int(value["row_count"], label="row_count"),
            family_counts=tuple(family_rows),
            layer_count=_require_int(value["layer_count"], label="layer_count"),
            query_head_count=_require_int(
                value["query_head_count"], label="query_head_count"
            ),
            row_receipts=tuple(
                AttentionEdgeAtlasRowReceipt.from_dict(item) for item in rows
            ),
            retained_routes=tuple(
                RetainedAttentionRoute.from_dict(item) for item in routes
            ),
            distance_motifs=tuple(
                AttentionDistanceMotif.from_dict(item) for item in motifs
            ),
            full_allowed_route_count=_require_int(
                value["full_allowed_route_count"],
                label="full_allowed_route_count",
            ),
            retained_route_count=_require_int(
                value["retained_route_count"], label="retained_route_count"
            ),
            full_intrinsic_fisher_energy=_nonnegative_float(
                value["full_intrinsic_fisher_energy"],
                label="full_intrinsic_fisher_energy",
            ),
            retained_intrinsic_fisher_energy=_nonnegative_float(
                value["retained_intrinsic_fisher_energy"],
                label="retained_intrinsic_fisher_energy",
            ),
            roles_statistically_read=tuple(str(item) for item in roles),
            schema=str(value["schema"]),
            format_version=_require_int(
                value["format_version"], label="format_version"
            ),
            report_sha256=str(value["report_sha256"]),
            **kwargs,  # type: ignore[arg-type]
        )


def _extraction_tensors(
    extraction: PartsFirstAttentionEdgeExtraction,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    probabilities = extraction.probabilities
    responses = extraction.signed_responses
    energies = extraction.intrinsic_fisher_energies
    tangents = extraction.softmax_tangents
    allowed = extraction.allowed_mask
    if (
        probabilities.ndim != 4
        or responses.shape != probabilities.shape
        or energies.shape != probabilities.shape
        or tangents.shape != probabilities.shape
        or allowed.shape
        != (probabilities.shape[0], probabilities.shape[2], probabilities.shape[3])
        or probabilities.shape[0] != 1
        or allowed.dtype is not torch.bool
        or not bool(torch.isfinite(probabilities).all())
        or not bool(torch.isfinite(responses).all())
        or not bool(torch.isfinite(energies).all())
        or not bool(torch.isfinite(tangents).all())
        or bool((energies < 0).any())
    ):
        raise ValueError("attention edge extraction tensors are invalid")
    return probabilities, responses, energies, tangents, allowed


def _closure_passed(extraction: PartsFirstAttentionEdgeExtraction) -> bool:
    closure = extraction.closure
    value = getattr(closure, "passed", None)
    if type(value) is bool:
        return value
    fields = (
        "probability_simplex_passed",
        "signed_response_closure_passed",
        "softmax_tangent_closure_passed",
        "intrinsic_fisher_energy_nonnegative_passed",
    )
    observed = tuple(getattr(closure, name, None) for name in fields)
    return bool(observed and all(value is True for value in observed))


def _extraction_sha256(extraction: PartsFirstAttentionEdgeExtraction) -> str:
    for name in ("sha256", "extraction_sha256", "receipt_sha256"):
        value = getattr(extraction, name, None)
        if isinstance(value, str) and _SHA256_RE.fullmatch(value):
            return value
    raise ValueError("attention edge extraction omitted its authenticated hash")


def compile_attention_edge_atlas(
    captures: Sequence[AttentionEdgeAtlasRowCapture],
    *,
    source_kind: str,
    model_id: str,
    model_revision: str,
    panel_id: str,
    panel_file_sha256: str,
    source_model_fingerprint_sha256: str,
    source_execution_fingerprint_sha256: str,
    score_objective_id: str,
    exact_control_receipt: AttentionEdgeExactControlReceipt | None,
    config: AttentionEdgeAtlasConfig = AttentionEdgeAtlasConfig(),
) -> AttentionEdgeAtlasReport:
    """Reduce full allowed route cubes into top-k edges and 72x8 motifs."""

    materialized = tuple(captures)
    if not materialized or any(
        not isinstance(item, AttentionEdgeAtlasRowCapture)
        for item in materialized
    ):
        raise ValueError("attention atlas requires typed row captures")
    layer_count = len(materialized[0].layer_extractions)
    if any(len(item.layer_extractions) != layer_count for item in materialized):
        raise ValueError("attention atlas rows have inconsistent layer counts")
    if len({item.row_catalog_id for item in materialized}) != len(materialized):
        raise ValueError("attention atlas row catalog IDs must be unique")
    row_catalog_sha256 = _json_sha256(
        [
            {
                "row_ordinal": ordinal,
                "row_catalog_id": row.row_catalog_id,
                "family_id": row.family_id,
            }
            for ordinal, row in enumerate(materialized)
        ],
        domain=_CATALOG_DOMAIN,
    )
    first_probabilities, _, _, _, _ = _extraction_tensors(
        materialized[0].layer_extractions[0]
    )
    head_count = int(first_probabilities.shape[1])
    accumulators = {
        (layer, head, bin_id): _MotifAccumulator()
        for layer in range(layer_count)
        for head in range(head_count)
        for bin_id in ATTENTION_DISTANCE_BINS
    }
    retained_routes: list[RetainedAttentionRoute] = []
    row_receipts: list[AttentionEdgeAtlasRowReceipt] = []
    for row_ordinal, capture in enumerate(materialized):
        sequence_length = len(capture.logical_positions)
        row_full_count = 0
        row_retained_count = 0
        row_full_energies: list[float] = []
        row_retained_energies: list[float] = []
        extraction_hashes: list[str] = []
        closures: list[bool] = []
        for layer_ordinal, extraction in enumerate(capture.layer_extractions):
            (
                probabilities,
                responses,
                energies,
                tangents,
                allowed,
            ) = _extraction_tensors(extraction)
            if (
                probabilities.shape[1] != head_count
                or probabilities.shape[2:] != (sequence_length, sequence_length)
            ):
                raise ValueError("attention extraction sequence geometry drifted")
            extraction_hashes.append(_extraction_sha256(extraction))
            closures.append(_closure_passed(extraction))
            allowed_cpu = allowed[0].to(device="cpu")
            probability_cpu = probabilities[0].to(device="cpu", dtype=torch.float64)
            response_cpu = responses[0].to(device="cpu", dtype=torch.float64)
            energy_cpu = energies[0].to(device="cpu", dtype=torch.float64)
            tangent_cpu = tangents[0].to(device="cpu", dtype=torch.float64)
            # The causal/window support is head-independent.  The sparse
            # publication cap is nevertheless global across (head, key) for
            # each (example, layer, query), while motifs below still consume
            # every allowed route and remain head-resolved.
            for query in range(sequence_length):
                query_allowed_keys = tuple(
                    int(item)
                    for item in torch.nonzero(
                        allowed_cpu[query], as_tuple=False
                    ).reshape(-1).tolist()
                )
                if not query_allowed_keys:
                    continue
                scored = sorted(
                    (
                        (
                            -float(energy_cpu[head, query, key].item()),
                            head,
                            key,
                        )
                        for head in range(head_count)
                        for key in query_allowed_keys
                    )
                )
                selected = tuple(
                    (head, key)
                    for _, head, key in scored[: config.top_routes_per_target]
                )
                full_bins_by_head: dict[int, set[str]] = {
                    head: set() for head in range(head_count)
                }
                retained_bins_by_head: dict[int, set[str]] = {
                    head: set() for head in range(head_count)
                }
                for head in range(head_count):
                    for key in query_allowed_keys:
                        # Match the exact generator/core visibility contract:
                        # distance is tensor ordinal, while logical positions
                        # affect RoPE and authenticated row identity only.
                        distance = query - key
                        bin_id = _distance_bin(distance)
                        full_bins_by_head[head].add(bin_id)
                        accumulator = accumulators[(layer_ordinal, head, bin_id)]
                        probability = float(probability_cpu[head, query, key].item())
                        response = float(response_cpu[head, query, key].item())
                        energy = float(energy_cpu[head, query, key].item())
                        tangent = float(tangent_cpu[head, query, key].item())
                        accumulator.full_route_count += 1
                        accumulator.probabilities.append(probability)
                        accumulator.signed.append(response)
                        accumulator.absolute_signed.append(abs(response))
                        accumulator.energies.append(energy)
                        accumulator.tangents.append(tangent)
                        accumulator.absolute_tangents.append(abs(tangent))
                        accumulator.maximum_energy = max(
                            accumulator.maximum_energy, energy
                        )
                        row_full_count += 1
                        row_full_energies.append(energy)
                for rank, (head, key) in enumerate(selected, start=1):
                    distance = query - key
                    bin_id = _distance_bin(distance)
                    retained_bins_by_head[head].add(bin_id)
                    probability = float(
                        probability_cpu[head, query, key].item()
                    )
                    response = float(response_cpu[head, query, key].item())
                    energy = float(energy_cpu[head, query, key].item())
                    tangent = float(tangent_cpu[head, query, key].item())
                    accumulator = accumulators[(layer_ordinal, head, bin_id)]
                    accumulator.retained_route_count += 1
                    accumulator.retained_probabilities.append(probability)
                    accumulator.retained_signed.append(response)
                    accumulator.retained_absolute_signed.append(abs(response))
                    accumulator.retained_energies.append(energy)
                    accumulator.retained_tangents.append(tangent)
                    accumulator.retained_absolute_tangents.append(abs(tangent))
                    row_retained_count += 1
                    row_retained_energies.append(energy)
                    retained_routes.append(
                        RetainedAttentionRoute(
                            row_ordinal=row_ordinal,
                            layer_ordinal=layer_ordinal,
                            query_head_ordinal=head,
                            query_ordinal=query,
                            key_ordinal=key,
                            causal_distance=distance,
                            distance_bin=bin_id,
                            target_rank=rank,
                            probability=probability,
                            signed_response=response,
                            intrinsic_fisher_energy=energy,
                            softmax_logit_tangent=tangent,
                        )
                    )
                for head in range(head_count):
                    for bin_id in full_bins_by_head[head]:
                        accumulators[
                            (layer_ordinal, head, bin_id)
                        ].full_targets += 1
                    for bin_id in retained_bins_by_head[head]:
                        accumulators[
                            (layer_ordinal, head, bin_id)
                        ].retained_targets += 1
        if not all(closures):
            raise RuntimeError("attention edge core closure failed")
        row_receipts.append(
            AttentionEdgeAtlasRowReceipt(
                row_ordinal=row_ordinal,
                family_id=capture.family_id,
                sequence_length=sequence_length,
                supervised_query_ordinal=capture.supervised_query_ordinal,
                tokenization_sha256=capture.tokenization_sha256,
                layer_extraction_sha256s=tuple(extraction_hashes),
                all_layer_closures_passed=True,
                full_allowed_route_count=row_full_count,
                retained_route_count=row_retained_count,
                full_intrinsic_fisher_energy=math.fsum(row_full_energies),
                retained_intrinsic_fisher_energy=math.fsum(
                    row_retained_energies
                ),
            )
        )
    motifs = tuple(
        AttentionDistanceMotif(
            layer_ordinal=layer,
            query_head_ordinal=head,
            distance_bin=bin_id,
            full_target_count=accumulators[(layer, head, bin_id)].full_targets,
            full_route_count=accumulators[(layer, head, bin_id)].full_route_count,
            retained_target_count=(
                accumulators[(layer, head, bin_id)].retained_targets
            ),
            retained_route_count=(
                accumulators[(layer, head, bin_id)].retained_route_count
            ),
            probability_sum=math.fsum(
                accumulators[(layer, head, bin_id)].probabilities
            ),
            retained_probability_sum=math.fsum(
                accumulators[(layer, head, bin_id)].retained_probabilities
            ),
            signed_response_sum=math.fsum(
                accumulators[(layer, head, bin_id)].signed
            ),
            absolute_signed_response_sum=math.fsum(
                accumulators[(layer, head, bin_id)].absolute_signed
            ),
            retained_signed_response_sum=math.fsum(
                accumulators[(layer, head, bin_id)].retained_signed
            ),
            retained_absolute_signed_response_sum=math.fsum(
                accumulators[(layer, head, bin_id)].retained_absolute_signed
            ),
            intrinsic_fisher_energy_sum=math.fsum(
                accumulators[(layer, head, bin_id)].energies
            ),
            retained_intrinsic_fisher_energy_sum=math.fsum(
                accumulators[(layer, head, bin_id)].retained_energies
            ),
            maximum_intrinsic_fisher_energy=(
                accumulators[(layer, head, bin_id)].maximum_energy
            ),
            softmax_logit_tangent_sum=math.fsum(
                accumulators[(layer, head, bin_id)].tangents
            ),
            absolute_softmax_logit_tangent_sum=math.fsum(
                accumulators[(layer, head, bin_id)].absolute_tangents
            ),
            retained_softmax_logit_tangent_sum=math.fsum(
                accumulators[(layer, head, bin_id)].retained_tangents
            ),
            retained_absolute_softmax_logit_tangent_sum=math.fsum(
                accumulators[
                    (layer, head, bin_id)
                ].retained_absolute_tangents
            ),
        )
        for layer in range(layer_count)
        for head in range(head_count)
        for bin_id in ATTENTION_DISTANCE_BINS
    )
    rows = tuple(row_receipts)
    routes = tuple(retained_routes)
    tokenization_catalog_sha256 = _tokenization_catalog_sha256(
        tuple(row.tokenization_sha256 for row in rows)
    )
    (
        core_implementation_sha256,
        runner_implementation_sha256,
        implementation_bundle_sha256,
    ) = _implementation_provenance()
    return AttentionEdgeAtlasReport(
        source_kind=source_kind,
        model_id=model_id,
        model_revision=model_revision,
        panel_id=panel_id,
        panel_file_sha256=panel_file_sha256,
        row_catalog_sha256=row_catalog_sha256,
        tokenization_catalog_sha256=tokenization_catalog_sha256,
        source_model_fingerprint_sha256=source_model_fingerprint_sha256,
        source_execution_fingerprint_sha256=(
            source_execution_fingerprint_sha256
        ),
        score_objective_id=score_objective_id,
        core_implementation_sha256=core_implementation_sha256,
        runner_implementation_sha256=runner_implementation_sha256,
        implementation_bundle_sha256=implementation_bundle_sha256,
        exact_control_receipt=exact_control_receipt,
        config=config,
        row_count=len(materialized),
        family_counts=tuple(sorted(Counter(
            item.family_id for item in materialized
        ).items())),
        layer_count=layer_count,
        query_head_count=head_count,
        row_receipts=rows,
        retained_routes=routes,
        distance_motifs=motifs,
        full_allowed_route_count=sum(
            item.full_allowed_route_count for item in rows
        ),
        retained_route_count=len(routes),
        full_intrinsic_fisher_energy=math.fsum(
            item.full_intrinsic_fisher_energy for item in rows
        ),
        retained_intrinsic_fisher_energy=math.fsum(
            item.retained_intrinsic_fisher_energy for item in rows
        ),
    )


class _FixtureClosure:
    passed = True


class _FixtureExtraction:
    def __init__(
        self,
        probabilities: Tensor,
        responses: Tensor,
        energies: Tensor,
        allowed: Tensor,
        *,
        label: str,
    ) -> None:
        self._probabilities = probabilities.clone()
        self._responses = responses.clone()
        self._energies = energies.clone()
        self._allowed = allowed.clone()
        allowed4 = allowed.unsqueeze(1).expand_as(responses)
        self._tangents = (
            responses
            - probabilities.to(responses.dtype)
            * responses.sum(dim=-1, keepdim=True)
        ).masked_fill(~allowed4, 0.0)
        self.closure = _FixtureClosure()
        self.sha256 = _json_sha256(
            {
                "label": label,
                "probabilities": _tensor_sha256(probabilities),
                "responses": _tensor_sha256(responses),
                "energies": _tensor_sha256(energies),
                "tangents": _tensor_sha256(self._tangents),
                "allowed": _tensor_sha256(allowed),
            },
            domain=_FIXTURE_DOMAIN,
        )

    @property
    def probabilities(self) -> Tensor:
        return self._probabilities.clone()

    @property
    def signed_responses(self) -> Tensor:
        return self._responses.clone()

    @property
    def intrinsic_fisher_energies(self) -> Tensor:
        return self._energies.clone()

    @property
    def softmax_tangents(self) -> Tensor:
        return self._tangents.clone()

    @property
    def allowed_mask(self) -> Tensor:
        return self._allowed.clone()


def build_deterministic_attention_edge_fixture() -> AttentionEdgeAtlasReport:
    """Build a small deterministic report without importing or loading Gemma."""

    captures: list[AttentionEdgeAtlasRowCapture] = []
    for row in range(2):
        sequence_length = 4 + row
        causal = torch.tril(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool)
        ).unsqueeze(0)
        layers: list[PartsFirstAttentionEdgeExtraction] = []
        for layer in range(2):
            base = torch.arange(
                1,
                1 + 2 * sequence_length * sequence_length,
                dtype=torch.float64,
            ).reshape(1, 2, sequence_length, sequence_length)
            allowed4 = causal.unsqueeze(1).expand(1, 2, -1, -1)
            weights = base.masked_fill(~allowed4, 0.0)
            probabilities = weights / weights.sum(dim=-1, keepdim=True)
            responses = (
                probabilities * (1.0 + 0.1 * row + 0.01 * layer)
            )
            energies = responses.square() * (
                1.0 + torch.arange(sequence_length, dtype=torch.float64)
            ).reshape(1, 1, 1, -1)
            layers.append(
                _FixtureExtraction(
                    probabilities.to(torch.float32),
                    responses,
                    energies,
                    causal,
                    label=f"fixture.{row}.{layer}",
                )  # type: ignore[arg-type]
            )
        captures.append(
            AttentionEdgeAtlasRowCapture(
                row_catalog_id=f"fixture-row-{row}",
                family_id="fixture_even" if row == 0 else "fixture_odd",
                logical_positions=tuple(range(sequence_length)),
                supervised_query_ordinal=sequence_length - 1,
                tokenization_sha256=_json_sha256(
                    {
                        "fixture_row": row,
                        "sequence_length": sequence_length,
                    },
                    domain=_TOKENIZATION_ROW_DOMAIN,
                ),
                layer_extractions=tuple(layers),
            )
        )
    fixture_hash = _json_sha256("model-free-fixture", domain=_FIXTURE_DOMAIN)
    return compile_attention_edge_atlas(
        tuple(captures),
        source_kind="model_free_fixture",
        model_id="model-free-fixture",
        model_revision="deterministic-v1",
        panel_id="attention-edge-fixture-v1",
        panel_file_sha256=fixture_hash,
        source_model_fingerprint_sha256=fixture_hash,
        source_execution_fingerprint_sha256=fixture_hash,
        score_objective_id="deterministic_fixture_scalar",
        exact_control_receipt=None,
    )


@dataclass(frozen=True, slots=True)
class _RuntimeRow:
    row_catalog_id: str
    family_id: str
    batch: CalibrationBatch
    sequence: SequenceContext
    tokenization_sha256: str


@dataclass(frozen=True, slots=True)
class AttentionEdgeAtlasLiveConfig:
    panel_path: Path = PINNED_PANEL_PATH
    cache_dir: Path | None = None
    local_files_only: bool = True
    maximum_prompt_tokens: int = ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS
    output: Path = DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_path", Path(self.panel_path))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "output", Path(self.output))
        if self.local_files_only is not True:
            raise ValueError("attention atlas requires a local-only checkpoint")
        if self.maximum_prompt_tokens != ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS:
            raise ValueError("attention atlas maximum_prompt_tokens is frozen at 128")
        if self.output.suffix != ".json":
            raise ValueError("attention atlas output must use .json")


class _ModelLoader(Protocol):
    def __call__(
        self,
        *,
        model_id: str,
        revision: str,
        cache_dir: Path,
        device: torch.device,
        dtype: str,
        local_files_only: bool,
    ) -> tuple[object, nn.Module]: ...


class _ExactCompiler(Protocol):
    def __call__(
        self,
        adapter: Gemma3CausalLMAdapter,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object: ...


def _default_model_loader() -> _ModelLoader:
    from .gemma3_experiment import load_gemma3

    return load_gemma3


def _default_exact_compiler() -> _ExactCompiler:
    from .gemma3_joint_cross_block_compiler import (
        compile_gemma3_joint_exact_control,
    )

    return compile_gemma3_joint_exact_control


def _resolve_cache_dir(value: Path | None) -> Path:
    from .gemma3_experiment import resolve_gemma3_huggingface_paths

    return resolve_gemma3_huggingface_paths(value)["hub_cache"]


def _valid_logits(logits: Tensor, batch: CalibrationBatch) -> Tensor:
    if logits.shape[:2] != batch.valid_positions.shape:
        raise ValueError("exact/native logits disagree with objective positions")
    if int(batch.valid_positions.sum().item()) != 1:
        raise ValueError("attention atlas requires one supervised query")
    return logits[batch.valid_positions.to(device=logits.device)].detach().clone()


def _attention_generators(executor: object) -> tuple[object, ...]:
    mutations = getattr(executor, "mutations", None)
    if not isinstance(mutations, nn.ModuleList) or len(mutations) != 36:
        raise ValueError("exact attention atlas executor body drifted")
    result = tuple(
        getattr(mutations[2 * layer], "compiled_operator", None)
        for layer in range(GEMMA3_LAYER_COUNT)
    )
    if any(item is None for item in result):
        raise ValueError("exact attention atlas omitted an attention generator")
    return result


@dataclass(slots=True)
class AttentionEdgeLiveRuntime:
    tokenizer: object
    executor: object
    rows: tuple[_RuntimeRow, ...]
    exact_control_receipt: AttentionEdgeExactControlReceipt
    model_id: str = MODEL_ID
    model_revision: str = PINNED_MODEL_REVISION
    panel_id: str = PINNED_PANEL_ID
    panel_file_sha256: str = PINNED_PANEL_FILE_SHA256
    source_model_fingerprint_sha256: str = SOURCE_MODEL_FINGERPRINT_SHA256
    source_execution_fingerprint_sha256: str = (
        PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
    )

    @classmethod
    def load(
        cls,
        config: AttentionEdgeAtlasLiveConfig = AttentionEdgeAtlasLiveConfig(),
        *,
        model_loader: _ModelLoader | None = None,
        exact_compiler: _ExactCompiler | None = None,
    ) -> "AttentionEdgeLiveRuntime":
        """Load, compile, prove first-row parity, then drop native references."""

        panel = load_pinned_parts_first_panel(config.panel_path)
        loader = _default_model_loader() if model_loader is None else model_loader
        device = torch.device("cpu")
        tokenizer, model = loader(
            model_id=MODEL_ID,
            revision=PINNED_MODEL_REVISION,
            cache_dir=_resolve_cache_dir(config.cache_dir),
            device=device,
            dtype="float32",
            local_files_only=config.local_files_only,
        )
        if not isinstance(model, nn.Module):
            raise TypeError("attention atlas model loader omitted a module")
        native_model_reference = weakref.ref(model)
        model.eval().requires_grad_(False)
        adapter = Gemma3CausalLMAdapter(model)
        model_before = adapter.model_fingerprint()
        execution_before = adapter.execution_fingerprint()
        if (
            model_before != SOURCE_MODEL_FINGERPRINT_SHA256
            or execution_before != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ):
            raise ValueError("loaded model differs from the pinned attention atlas")
        runtime_rows: list[_RuntimeRow] = []
        for example in panel.examples:
            batch = build_correct_choice_nll_batch(
                tokenizer,
                example,
                device=device,
                maximum_prompt_tokens=config.maximum_prompt_tokens,
            )
            runtime_rows.append(
                _RuntimeRow(
                    row_catalog_id=example.example_id,
                    family_id=example.family_id,
                    batch=batch,
                    sequence=adapter.prepare_sequence(batch.model_inputs),
                    tokenization_sha256=_tokenization_row_sha256(
                        example.example_id,
                        batch,
                    ),
                )
            )
        rows = tuple(runtime_rows)
        live_tokenization_catalog = _tokenization_catalog_sha256(
            tuple(row.tokenization_sha256 for row in rows)
        )
        if (
            live_tokenization_catalog
            != PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256
        ):
            raise ValueError("pinned attention atlas tokenization drifted")
        compiler = (
            _default_exact_compiler() if exact_compiler is None else exact_compiler
        )
        compilation = compiler(adapter, dtype=torch.float32, device=device)
        if getattr(compilation, "compilation_kind", None) != (
            "exact_39_event_control"
        ):
            raise ValueError("attention atlas compiler did not return exact control")
        metadata_callback = getattr(compilation, "metadata", None)
        metadata = metadata_callback() if callable(metadata_callback) else None
        independence = getattr(compilation, "source_independence_audit", None)
        executor = getattr(compilation, "executor", None)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("exact_parameter_inventory_parity") is not True
            or not isinstance(independence, Mapping)
            or independence.get("passed") is not True
            or tuple(getattr(executor, "ordered_event_ids", ()))
            != _EXPECTED_EVENT_IDS
        ):
            raise RuntimeError("attention atlas exact-control audit failed")
        _attention_generators(executor)
        first = rows[0]
        with torch.no_grad():
            native = adapter.forward(first.batch.model_inputs)
            native_valid = _valid_logits(native.logits, first.batch)
            exact_execution = executor.forward_components(
                first.batch.model_inputs["input_ids"],
                first.sequence,
            )
            exact_valid = _valid_logits(exact_execution.logits, first.batch)
        parity = bool(
            native_valid.shape == exact_valid.shape
            and torch.equal(native_valid, exact_valid)
        )
        model_after = adapter.model_fingerprint()
        execution_after = adapter.execution_fingerprint()
        native_hash = _tensor_sha256(native_valid)
        exact_hash = _tensor_sha256(exact_valid)
        valid_shape = tuple(int(item) for item in native_valid.shape)
        source_parameter_count = int(
            getattr(compilation, "source_model_parameter_count", 0)
        )
        compiler_metadata_sha256 = _json_sha256(
            dict(metadata), domain=_CONTROL_METADATA_DOMAIN
        )
        source_independence_audit_sha256 = _json_sha256(
            dict(independence), domain=_INDEPENDENCE_AUDIT_DOMAIN
        )
        del native_valid, exact_valid, exact_execution, native
        del adapter, model, compilation
        gc.collect()
        native_source_released = native_model_reference() is None
        receipt = AttentionEdgeExactControlReceipt(
            ordered_event_ids=_EXPECTED_EVENT_IDS,
            compiler_metadata_sha256=compiler_metadata_sha256,
            source_independence_audit_sha256=(
                source_independence_audit_sha256
            ),
            native_valid_logits_sha256=native_hash,
            exact_valid_logits_sha256=exact_hash,
            valid_logit_shape=valid_shape,
            literal_native_exact_parity_passed=parity,
            source_model_fingerprint_before=model_before,
            source_model_fingerprint_after_parity=model_after,
            source_execution_fingerprint_before=execution_before,
            source_execution_fingerprint_after_parity=execution_after,
            source_parameter_count=source_parameter_count,
            exact_parameter_inventory_parity=True,
            source_independence_passed=True,
            native_source_references_released_before_atlas=(
                native_source_released
            ),
        )
        if not receipt.passed:
            raise RuntimeError("attention atlas literal exact-control parity failed")
        return cls(
            tokenizer=tokenizer,
            executor=executor,
            rows=rows,
            exact_control_receipt=receipt,
        )

    def capture(self, *, row_limit: int | None = None) -> AttentionEdgeAtlasReport:
        if row_limit is None:
            selected = self.rows
            source_kind = "model_capture"
        else:
            if type(row_limit) is not int or row_limit != 1:
                raise ValueError("only the exact one-row smoke limit is supported")
            selected = self.rows[:1]
            source_kind = "model_smoke"
        generators = _attention_generators(self.executor)
        captures: list[AttentionEdgeAtlasRowCapture] = []
        objective = CausalLanguageModelNLL()
        for row in selected:
            snapshots: dict[str, Tensor] = {}

            def observer(*, event_id: str, role: str, values: Tensor) -> None:
                index = int(event_id.split(".")[-1]) if event_id.startswith(
                    "mutation."
                ) else -1
                if role == "input" and index in range(1, 37, 2):
                    snapshots[event_id] = values

            with torch.enable_grad():
                observed = self.executor.forward_components_with_observer(
                    row.batch.model_inputs["input_ids"],
                    row.sequence,
                    observer=observer,
                    retain_gradient_tape=True,
                )
                tape = observed.gradient_tape
                if tape is None:
                    raise RuntimeError("attention atlas exact execution omitted tape")
                score = objective(observed.execution, row.batch)
                gradients = tape.score_gradients(score)
            if len(snapshots) != GEMMA3_LAYER_COUNT:
                raise RuntimeError("attention atlas observer missed a layer")
            layer_extractions = tuple(
                compile_parts_first_attention_edges(
                    generators[layer],
                    snapshots[f"mutation.{1 + 2 * layer}"],
                    gradients[f"mutation.{1 + 2 * layer}.output"],
                    row.sequence,
                )
                for layer in range(GEMMA3_LAYER_COUNT)
            )
            logical = row.sequence.logical_positions[0]
            valid = row.sequence.query_valid_mask[0]
            valid_ordinals = tuple(
                int(item)
                for item in logical[valid].detach().to(device="cpu").tolist()
            )
            if len(valid_ordinals) != len(row.batch.valid_positions[0]):
                raise ValueError("attention atlas currently requires unpadded rows")
            supervised = row.batch.valid_positions[0].nonzero(as_tuple=False)
            if tuple(supervised.shape) != (1, 1):
                raise ValueError("attention atlas row must have one query target")
            captures.append(
                AttentionEdgeAtlasRowCapture(
                    row_catalog_id=row.row_catalog_id,
                    family_id=row.family_id,
                    logical_positions=valid_ordinals,
                    supervised_query_ordinal=int(supervised[0, 0].item()),
                    tokenization_sha256=row.tokenization_sha256,
                    layer_extractions=layer_extractions,
                )
            )
        return compile_attention_edge_atlas(
            tuple(captures),
            source_kind=source_kind,
            model_id=self.model_id,
            model_revision=self.model_revision,
            panel_id=self.panel_id,
            panel_file_sha256=self.panel_file_sha256,
            source_model_fingerprint_sha256=(
                self.source_model_fingerprint_sha256
            ),
            source_execution_fingerprint_sha256=(
                self.source_execution_fingerprint_sha256
            ),
            score_objective_id=MODEL_SCORE_OBJECTIVE_ID,
            exact_control_receipt=self.exact_control_receipt,
        )


def build_attention_edge_atlas_preflight(
    config: AttentionEdgeAtlasLiveConfig = AttentionEdgeAtlasLiveConfig(),
) -> dict[str, object]:
    panel = load_pinned_parts_first_panel(config.panel_path)
    row_catalog_sha256 = _json_sha256(
        [
            {
                "row_ordinal": ordinal,
                "row_catalog_id": example.example_id,
                "family_id": example.family_id,
            }
            for ordinal, example in enumerate(panel.examples)
        ],
        domain=_CATALOG_DOMAIN,
    )
    if row_catalog_sha256 != PINNED_ATTENTION_EDGE_ROW_CATALOG_SHA256:
        raise ValueError("attention atlas preflight row catalog drifted")
    fixture = build_deterministic_attention_edge_fixture()
    family_counts = Counter(example.family_id for example in panel.examples)
    core_sha256, runner_sha256, bundle_sha256 = _implementation_provenance()
    return {
        "schema": "fisher_graph.gemma3_parts_first.attention_edge_atlas_preflight.v1",
        "ready_to_load": True,
        "model_loaded": False,
        "transformers_imported_by_runner": False,
        "model_id": MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_dtype": "float32",
        "device": "cpu",
        "local_files_only": config.local_files_only,
        "panel_id": panel.panel_id,
        "panel_file_sha256": panel.file_sha256,
        "panel_row_count": len(panel.examples),
        "row_catalog_sha256": row_catalog_sha256,
        "family_counts": dict(sorted(family_counts.items())),
        "old_roles_collapsed": True,
        "roles_statistically_read": [ATTENTION_EDGE_ATLAS_ROLE],
        "score_objective_id": MODEL_SCORE_OBJECTIVE_ID,
        "maximum_prompt_tokens": ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS,
        "tokenization_receipt_abi": (
            "row_id_hash_plus_input_ids_mask_targets_valid_positions_hashes.v1"
        ),
        "tokenization_catalog_sha256": (
            PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256
        ),
        "supervised_targets_per_row": 1,
        "supervised_target_location": "final_valid_query",
        "full_suffix_query_key_routes_retained_during_reduction": True,
        "layer_count": GEMMA3_LAYER_COUNT,
        "query_head_count": ATTENTION_HEAD_COUNT,
        "distance_bin_count": len(ATTENTION_DISTANCE_BINS),
        "motif_cell_count": (
            GEMMA3_LAYER_COUNT
            * ATTENTION_HEAD_COUNT
            * len(ATTENTION_DISTANCE_BINS)
        ),
        "top_routes_per_target": ATTENTION_EDGE_TOP_K,
        "valid_token_count": PINNED_ATTENTION_EDGE_VALID_TOKEN_COUNT,
        "maximum_retained_route_count": (
            PINNED_ATTENTION_EDGE_MAXIMUM_RETAINED_ROUTE_COUNT
        ),
        "maximum_retained_route_count_formula": (
            "valid_token_count * layer_count * top_routes_per_target"
        ),
        "exact_event_count": len(_EXPECTED_EVENT_IDS),
        "native_exact_first_row_literal_parity_required": True,
        "native_source_release_before_atlas_required": True,
        "model_free_fixture_report_sha256": fixture.report_sha256,
        "core_implementation_sha256": core_sha256,
        "runner_implementation_sha256": runner_sha256,
        "implementation_bundle_sha256": bundle_sha256,
        "implementation_provenance_scope": (
            "attention_edge_core_and_atlas_runner_source_bytes_only.v1"
        ),
        "transitive_runtime_dependencies_in_implementation_bundle": False,
        "compact_artifact_self_proves_exact_full_bin_values": False,
        "compact_artifact_self_proves_top_k_optimality": False,
        "trusted_live_execution_or_external_report_hash_required": True,
        "prompts_persisted": False,
        "token_ids_persisted": False,
        "raw_activations_persisted": False,
        "raw_gradients_persisted": False,
        "retrospective_A_only": True,
        "structure_only": True,
        "confirmation_claimed": False,
        "promotion_authorized": False,
        "compression_claimed": False,
    }


def validate_attention_edge_atlas(
    report: AttentionEdgeAtlasReport,
) -> AttentionEdgeAtlasReport:
    if type(report) is not AttentionEdgeAtlasReport:
        raise TypeError("attention atlas must be an exact typed report")
    replay = AttentionEdgeAtlasReport.from_dict(report.to_dict())
    if replay != report:
        raise ValueError("attention atlas typed replay drifted")
    return report


def parse_attention_edge_atlas_json(source: str | bytes) -> AttentionEdgeAtlasReport:
    value = _strict_json_object(source, label="attention edge atlas")
    return validate_attention_edge_atlas(AttentionEdgeAtlasReport.from_dict(value))


def load_attention_edge_atlas(
    source: Path | str = DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT,
) -> AttentionEdgeAtlasReport:
    path = Path(source)
    if path.suffix != ".json":
        raise ValueError("attention atlas path must use .json")
    return _load_report_path(path)


def _load_report_path(path: Path) -> AttentionEdgeAtlasReport:
    descriptor: int | None = None
    primary: BaseException | None = None
    chunks: list[bytes] = []
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        initial = os.fstat(descriptor)
        named_initial = path.lstat()
        if (
            not os.path.samestat(initial, named_initial)
            or not stat.S_ISREG(initial.st_mode)
            or stat.S_IMODE(initial.st_mode) != 0o600
            or initial.st_nlink != 1
        ):
            raise ValueError("attention edge atlas file identity is unsafe")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(descriptor)
        named_final = path.lstat()
        if (
            not os.path.samestat(initial, final)
            or not os.path.samestat(initial, named_final)
            or not stat.S_ISREG(final.st_mode)
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_nlink != 1
        ):
            raise ValueError("attention edge atlas file changed during read")
    except OSError as error:
        primary = ValueError("attention edge atlas is unavailable or unsafe")
        raise primary from error
    except BaseException as error:
        primary = error
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "attention atlas descriptor close failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    encoded = b"".join(chunks)
    return parse_attention_edge_atlas_json(encoded)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_attention_edge_atlas(
    report: AttentionEdgeAtlasReport,
    destination: Path | str = DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT,
) -> AttentionEdgeAtlasReport:
    validate_attention_edge_atlas(report)
    target = Path(destination)
    if target.suffix != ".json":
        raise ValueError("attention atlas output must use .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    temporary_exists = True
    target_linked = False
    temporary_stat: os.stat_result | None = None
    primary: BaseException | None = None
    try:
        temporary_stat = os.fstat(descriptor)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not os.path.samestat(temporary_stat, os.fstat(handle.fileno())):
                raise RuntimeError("attention atlas descriptor identity drifted")
        loaded = _load_report_path(temporary)
        if loaded != report:
            raise RuntimeError("attention atlas changed on strict temp reload")
        current = temporary.lstat()
        if temporary_stat is None or not os.path.samestat(temporary_stat, current):
            raise RuntimeError("attention atlas temporary identity drifted")
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite an attention edge atlas"
            ) from error
        target_linked = True
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 2
        ):
            raise RuntimeError("attention atlas hardlink identity drifted")
        current = temporary.lstat()
        if not os.path.samestat(temporary_stat, current):
            raise RuntimeError("attention atlas temporary changed before unlink")
        temporary.unlink()
        temporary_exists = False
        _fsync_directory(target.parent)
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 1
        ):
            raise RuntimeError("attention atlas publication identity drifted")
        canonical = _load_report_path(target)
        if canonical != report:
            raise RuntimeError("attention atlas changed on canonical reload")
        final_stat = target.lstat()
        if (
            not os.path.samestat(temporary_stat, final_stat)
            or final_stat.st_nlink != 1
            or stat.S_IMODE(final_stat.st_mode) != 0o600
        ):
            raise RuntimeError("attention atlas changed after canonical reload")
        return canonical
    except BaseException as error:
        primary = error
        raise
    finally:
        if primary is not None and target_linked:
            try:
                current_target = target.lstat()
                if temporary_stat is not None and os.path.samestat(
                    temporary_stat, current_target
                ):
                    target.unlink()
                    _fsync_directory(target.parent)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                primary.add_note(
                    "attention atlas published-target rollback failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if temporary_exists:
            try:
                current = temporary.lstat()
                if temporary_stat is not None and os.path.samestat(
                    temporary_stat, current
                ):
                    temporary.unlink()
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "attention atlas temporary cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


def run_attention_edge_atlas(
    *,
    config: AttentionEdgeAtlasLiveConfig = AttentionEdgeAtlasLiveConfig(),
    write: bool = False,
) -> AttentionEdgeAtlasReport:
    runtime = AttentionEdgeLiveRuntime.load(config)
    report = runtime.capture()
    if write:
        write_attention_edge_atlas(report, config.output)
    return report


def run_attention_edge_one_row_smoke(
    *,
    config: AttentionEdgeAtlasLiveConfig = AttentionEdgeAtlasLiveConfig(),
) -> AttentionEdgeAtlasReport:
    return AttentionEdgeLiveRuntime.load(config).capture(row_limit=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--demo", action="store_true")
    action.add_argument("--smoke-one-row", action="store_true")
    action.add_argument("--run-live", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--panel", type=Path, default=PINNED_PANEL_PATH)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT,
    )
    return parser


def _compact_report_summary(
    report: AttentionEdgeAtlasReport,
    *,
    output: Path | None,
    published: bool,
) -> dict[str, object]:
    return {
        "schema": report.schema,
        "source_kind": report.source_kind,
        "report_sha256": report.report_sha256,
        "implementation_bundle_sha256": report.implementation_bundle_sha256,
        "tokenization_catalog_sha256": report.tokenization_catalog_sha256,
        "row_count": report.row_count,
        "motif_cell_count": len(report.distance_motifs),
        "full_allowed_route_count": report.full_allowed_route_count,
        "retained_route_count": report.retained_route_count,
        "output": None if output is None else output.as_posix(),
        "published": published,
        "retrospective_A_only": report.retrospective_A_only,
        "structure_only": report.structure_only,
        "attention_edge_confirmation_claimed": (
            report.attention_edge_confirmation_claimed
        ),
        "attention_promotion_authorized": (
            report.attention_promotion_authorized
        ),
        "compression_claimed": report.compression_claimed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = AttentionEdgeAtlasLiveConfig(
        panel_path=args.panel,
        cache_dir=args.cache_dir,
        local_files_only=True,
        maximum_prompt_tokens=ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS,
        output=args.output,
    )
    if args.write and not args.run_live:
        raise ValueError("only the complete live atlas may be published")
    if args.preflight_only:
        value: object = build_attention_edge_atlas_preflight(config)
    elif args.demo:
        report = build_deterministic_attention_edge_fixture()
        value = _compact_report_summary(
            report,
            output=None,
            published=False,
        )
    elif args.smoke_one_row:
        report = run_attention_edge_one_row_smoke(config=config)
        value = _compact_report_summary(
            report,
            output=None,
            published=False,
        )
    else:
        report = run_attention_edge_atlas(config=config, write=args.write)
        value = _compact_report_summary(
            report,
            output=config.output,
            published=bool(args.write),
        )
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
