"""Retrospective fidelity curve for the parts-first attention-edge atlas.

This experiment asks a narrower question than the atlas: if the globally
ranked token routes are retained at every attention layer, how faithfully do
they reproduce the exact attention operator and the model's supervised
logits?  The frozen curve uses K=(1, 2, 4, 8, 16) routes across all
``(query_head, key)`` coordinates for each ``(row, layer, query)``.  That
global convention is identical to the published atlas; it intentionally may
starve a head.

The primary matrix compares four rankings: coordinate-diagonal Fisher energy,
native probability, causal recency, and a deterministic same-support random
control.  A target-free route-output-energy ranking is a fifth diagnostic.
All five run across the full K/semantics curve; three additional deterministic
random salts run at the primary K8 zero-fill point, for 53 total conditions.
Each curve mask is evaluated both as an unnormalised contribution truncation
and as sparse attention renormalised independently within every head.  A
starved head has exactly zero output under the latter convention and is
reported.

The live path uses the committed exact executor only.  It first captures the
native graph and score gradient, then proves that this module's batch-one dense
replay is literally logit-identical to that executor.  Finally it evaluates a
pair of 53-row batches with identical shape and condition order: one
all-routes matched-control batch and one sparse-condition batch.  Each sparse
condition is compared with the all-routes control at the same batch ordinal,
eliminating both batch-size and row-ordinal kernel drift.  End-to-end fidelity
uses those ordinal-matched controls, while the local teacher-forced operator
and post-norm-delta diagnostics remain referenced to the native graph.  Q/K/V
and attention probabilities are recomputed from each evolving sparse state.
Thus the result is a genuine all-layer sparse-attention intervention, not a
linearised Fisher estimate or an unmeasured batch-shape artefact.

This remains a retrospective A-panel fidelity measurement.  It is not fresh
confirmation, semantic interpretation, compression qualification, or a
parameter/MAC/latency claim.  Import and preflight do not import Transformers
or load a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import SequenceContext
from .compiler.calibration import CausalLanguageModelNLL
from .gemma3_attention_generator import (
    AttentionCoordinatePacket,
    Gemma3AttentionGenerator,
)
from .gemma3_parts_first_attention_edge_experiment import (
    ATTENTION_EDGE_ATLAS_ABI,
    ATTENTION_EDGE_ATLAS_ROLE,
    ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS,
    ATTENTION_EDGE_TOP_K,
    DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT,
    PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256,
    AttentionEdgeAtlasLiveConfig,
    AttentionEdgeAtlasReport,
    AttentionEdgeLiveRuntime,
    parse_attention_edge_atlas_json,
)
from .gemma3_parts_first_attention_edges import (
    PartsFirstAttentionEdgeExtraction,
    compile_parts_first_attention_edges,
)
from . import gemma3_parts_first_attention_edge_fidelity as fidelity_kernel
from .gemma3_parts_first_supermode_experiment import MODEL_SCORE_OBJECTIVE_ID
from .gemma3_parts_first_supermode_model_runner import (
    PINNED_MODEL_REVISION,
    PINNED_PANEL_FILE_SHA256,
    PINNED_PANEL_ID,
    PINNED_PANEL_PATH,
    PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256,
)
from .gemma3_parts_first_supermode_protocol import (
    MODEL_ID,
    SOURCE_MODEL_FINGERPRINT_SHA256,
)
from .gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    GEMMA3_LAYER_COUNT,
)


ATTENTION_EDGE_FIDELITY_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edge_fidelity.v1"
)
ATTENTION_EDGE_FIDELITY_FORMAT_VERSION: Final = 1
ATTENTION_EDGE_FIDELITY_ABI: Final = (
    "baseline_global_head_key_topk_dynamic_all_layer_sparse_replay.v1"
)
ATTENTION_EDGE_FIDELITY_K_VALUES: Final = (
    fidelity_kernel.ATTENTION_EDGE_FIDELITY_K_VALUES
)
ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS: Final = (
    fidelity_kernel.ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS
)
ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS: Final = (
    fidelity_kernel.ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS
)
ATTENTION_EDGE_FIDELITY_RANKERS: Final = (
    fidelity_kernel.ATTENTION_EDGE_FIDELITY_RANKERS
)
ATTENTION_EDGE_FIDELITY_VARIANTS: Final = (
    fidelity_kernel.ATTENTION_EDGE_FIDELITY_VARIANTS
)
ATTENTION_EDGE_FIDELITY_RANDOM_SALTS: Final = (
    fidelity_kernel.ATTENTION_EDGE_FIDELITY_RANDOM_SALTS
)
ATTENTION_EDGE_FIDELITY_RANDOM_SEED_ID: Final = (
    "sha256_salt_row_layer_query_head_key.v1"
)
ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA: Final = 1e-4
ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE: Final = 1e-4
ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE: Final = 1e-5
ATTENTION_EDGE_FIDELITY_ROLE: Final = "A_retrospective_fidelity_diagnostic"
DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-attention-edge-fidelity-A-retrospective-v1.json"
)
PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256: Final = (
    "5e815075fae7575817a1fee4661950a0e45e8a36c75d86d7570c3e4ea4eb09c7"
)
PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256: Final = (
    "c5096391aa1d6600bcb5762f4528fa263f00bdc6539d7e2cb03b5476f1374a15"
)

_REPORT_DOMAIN = b"fisher_graph:parts-first-attention-edge-fidelity:v1\0"
_ROW_DOMAIN = b"fisher_graph:parts-first-attention-edge-fidelity-row:v1\0"
_SELECTION_DOMAIN = b"fisher_graph:parts-first-attention-edge-selection:v1\0"
_IMPLEMENTATION_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-fidelity-implementation:v1\0"
)
_ATLAS_CATALOG_DOMAIN = b"fisher_graph:parts-first-attention-edge-catalog:v1\0"
_TOKENIZATION_CATALOG_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-tokenization-catalog:v1\0"
)
_TENSOR_DOMAIN = b"fisher_graph:parts-first-attention-edge-fidelity-tensor:v1\0"
_MATCHED_SURFACE_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-fidelity-matched-surface:v1\0"
)
_MATCHED_DIAGNOSTIC_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-fidelity-matched-diagnostic:v1\0"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


__all__ = [
    "ATTENTION_EDGE_FIDELITY_ABI",
    "ATTENTION_EDGE_FIDELITY_FORMAT_VERSION",
    "ATTENTION_EDGE_FIDELITY_K_VALUES",
    "ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA",
    "ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE",
    "ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE",
    "ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS",
    "ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS",
    "ATTENTION_EDGE_FIDELITY_RANDOM_SEED_ID",
    "ATTENTION_EDGE_FIDELITY_RANDOM_SALTS",
    "ATTENTION_EDGE_FIDELITY_RANKERS",
    "ATTENTION_EDGE_FIDELITY_ROLE",
    "ATTENTION_EDGE_FIDELITY_SCHEMA",
    "ATTENTION_EDGE_FIDELITY_VARIANTS",
    "DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT",
    "PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256",
    "PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256",
    "AttentionFidelityVectorStats",
    "AttentionEdgeFidelityMetricLedger",
    "AttentionEdgeFidelityCell",
    "AttentionEdgeFidelityFamilyCell",
    "AttentionEdgeFidelityCondition",
    "AttentionEdgeFidelityLiveConfig",
    "AttentionEdgeFidelityMeasurement",
    "AttentionEdgeFidelityMatchedControlDiagnostic",
    "AttentionEdgeFidelityReport",
    "AttentionEdgeFidelityRuntime",
    "AttentionEdgeFidelityRowCapture",
    "AttentionEdgeFidelityRowReceipt",
    "AttentionRouteSelection",
    "build_attention_edge_fidelity_preflight",
    "build_deterministic_attention_edge_fidelity_fixture",
    "compile_attention_edge_fidelity",
    "attention_fidelity_vector_stats",
    "aggregate_attention_fidelity_vector_stats",
    "load_attention_edge_fidelity",
    "parse_attention_edge_fidelity_json",
    "run_attention_edge_fidelity",
    "select_attention_routes",
    "validate_attention_edge_fidelity",
    "write_attention_edge_fidelity",
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


def _finite(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _nonnegative(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_matched_control_bounds(
    *,
    maximum_absolute_nll_delta: float,
    maximum_centered_logit_nrmse: float,
    maximum_final_carrier_nrmse: float,
) -> None:
    observed = (
        (
            "matched control NLL drift",
            _nonnegative(
                maximum_absolute_nll_delta,
                label="maximum_absolute_nll_delta",
            ),
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA,
        ),
        (
            "matched control centered-logit NRMSE",
            _nonnegative(
                maximum_centered_logit_nrmse,
                label="maximum_centered_logit_nrmse",
            ),
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE,
        ),
        (
            "matched control final-carrier NRMSE",
            _nonnegative(
                maximum_final_carrier_nrmse,
                label="maximum_final_carrier_nrmse",
            ),
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE,
        ),
    )
    for label, value, bound in observed:
        if value > bound:
            raise ValueError(f"{label} exceeds frozen bound")


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
    canonical = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {"dtype": str(canonical.dtype), "shape": list(canonical.shape)}
    )
    raw = canonical.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(_TENSOR_DOMAIN + header + b"\0" + raw).hexdigest()


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_provenance() -> tuple[str, ...]:
    directory = Path(__file__).resolve().parent
    generator = _source_sha256(directory / "gemma3_attention_generator.py")
    executor = _source_sha256(
        directory / "gemma3_joint_cross_block_chart_executor.py"
    )
    compiler = _source_sha256(directory / "gemma3_joint_cross_block_compiler.py")
    core = _source_sha256(directory / "gemma3_parts_first_attention_edges.py")
    atlas = _source_sha256(
        directory / "gemma3_parts_first_attention_edge_experiment.py"
    )
    fidelity_kernel_hash = _source_sha256(
        directory / "gemma3_parts_first_attention_edge_fidelity.py"
    )
    fidelity_runner = _source_sha256(Path(__file__).resolve())
    bundle = _json_sha256(
        {
            "attention_generator_implementation_sha256": generator,
            "joint_executor_implementation_sha256": executor,
            "joint_compiler_implementation_sha256": compiler,
            "attention_edge_core_implementation_sha256": core,
            "attention_edge_atlas_implementation_sha256": atlas,
            "attention_edge_fidelity_kernel_implementation_sha256": (
                fidelity_kernel_hash
            ),
            "attention_edge_fidelity_implementation_sha256": fidelity_runner,
        },
        domain=_IMPLEMENTATION_DOMAIN,
    )
    return (
        generator,
        executor,
        compiler,
        core,
        atlas,
        fidelity_kernel_hash,
        fidelity_runner,
        bundle,
    )


_FLOAT64_UNIT_ROUNDOFF: Final = 2.0**-53


def _stats_algebra_tolerance(
    *,
    element_count: int,
    reference_sum_squares: float,
    sparse_sum_squares: float,
    error_sum_squares: float,
    dot_product: float,
    maximum_abs_error: float,
) -> float:
    operations = max(32, 12 * element_count + 32)
    numerator = operations * _FLOAT64_UNIT_ROUNDOFF
    if numerator >= 1.0:
        raise ValueError("fidelity statistic inventory exceeds float64 model")
    gamma = numerator / (1.0 - numerator)
    scale = max(
        1.0,
        reference_sum_squares,
        sparse_sum_squares,
        error_sum_squares,
        abs(2.0 * dot_product),
        element_count * maximum_abs_error * maximum_abs_error,
    )
    return float(gamma * scale + 32 * _FLOAT64_UNIT_ROUNDOFF * scale)


@dataclass(frozen=True, slots=True)
class AttentionFidelityVectorStats:
    element_count: int
    reference_sum_squares: float
    sparse_sum_squares: float
    error_sum_squares: float
    dot_product: float
    maximum_abs_error: float

    def __post_init__(self) -> None:
        count = _integer(self.element_count, label="element_count", minimum=1)
        reference = _nonnegative(
            self.reference_sum_squares, label="reference_sum_squares"
        )
        sparse = _nonnegative(
            self.sparse_sum_squares, label="sparse_sum_squares"
        )
        error = _nonnegative(self.error_sum_squares, label="error_sum_squares")
        dot = _finite(self.dot_product, label="dot_product")
        maximum = _nonnegative(
            self.maximum_abs_error, label="maximum_abs_error"
        )
        tolerance = _stats_algebra_tolerance(
            element_count=count,
            reference_sum_squares=reference,
            sparse_sum_squares=sparse,
            error_sum_squares=error,
            dot_product=dot,
            maximum_abs_error=maximum,
        )
        identity = reference + sparse - 2.0 * dot
        if abs(error - identity) > tolerance:
            raise ValueError("fidelity vector quadratic identity failed")
        cauchy_limit = math.sqrt(reference * sparse)
        if abs(dot) > cauchy_limit + tolerance:
            raise ValueError("fidelity vector Cauchy bound failed")
        maximum_square = maximum * maximum
        if maximum_square > error + tolerance:
            raise ValueError("fidelity vector maximum exceeds total error")
        if error > count * maximum_square + tolerance:
            raise ValueError("fidelity vector total error exceeds maximum bound")

    @property
    def nrmse(self) -> float:
        return _nrmse(self.error_sum_squares, self.reference_sum_squares)

    @property
    def rmse(self) -> float:
        return math.sqrt(self.error_sum_squares / self.element_count)

    @property
    def cosine(self) -> float:
        return _cosine(
            self.dot_product,
            self.reference_sum_squares,
            self.sparse_sum_squares,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "element_count": self.element_count,
            "reference_sum_squares": self.reference_sum_squares,
            "sparse_sum_squares": self.sparse_sum_squares,
            "error_sum_squares": self.error_sum_squares,
            "dot_product": self.dot_product,
            "maximum_abs_error": self.maximum_abs_error,
            "nrmse": self.nrmse,
            "rmse": self.rmse,
            "cosine": self.cosine,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionFidelityVectorStats":
        _exact_keys(
            value,
            {
                "element_count",
                "reference_sum_squares",
                "sparse_sum_squares",
                "error_sum_squares",
                "dot_product",
                "maximum_abs_error",
                "nrmse",
                "rmse",
                "cosine",
            },
            label="fidelity vector statistics",
        )
        result = cls(
            element_count=_integer(
                value["element_count"], label="element_count", minimum=1
            ),
            reference_sum_squares=_finite(
                value["reference_sum_squares"],
                label="reference_sum_squares",
            ),
            sparse_sum_squares=_finite(
                value["sparse_sum_squares"], label="sparse_sum_squares"
            ),
            error_sum_squares=_finite(
                value["error_sum_squares"], label="error_sum_squares"
            ),
            dot_product=_finite(value["dot_product"], label="dot_product"),
            maximum_abs_error=_finite(
                value["maximum_abs_error"], label="maximum_abs_error"
            ),
        )
        for label, expected in (
            ("nrmse", result.nrmse),
            ("rmse", result.rmse),
            ("cosine", result.cosine),
        ):
            if _finite(value[label], label=label) != expected:
                raise ValueError(f"fidelity vector derived {label} drifted")
        return result


def _aggregate_vector_stats(
    values: Sequence[AttentionFidelityVectorStats],
) -> AttentionFidelityVectorStats:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty fidelity vector ledger")
    return AttentionFidelityVectorStats(
        element_count=sum(item.element_count for item in materialized),
        reference_sum_squares=math.fsum(
            item.reference_sum_squares for item in materialized
        ),
        sparse_sum_squares=math.fsum(
            item.sparse_sum_squares for item in materialized
        ),
        error_sum_squares=math.fsum(
            item.error_sum_squares for item in materialized
        ),
        dot_product=math.fsum(item.dot_product for item in materialized),
        maximum_abs_error=max(item.maximum_abs_error for item in materialized),
    )


def aggregate_attention_fidelity_vector_stats(
    values: Sequence[AttentionFidelityVectorStats],
) -> AttentionFidelityVectorStats:
    """Public exact aggregation of authenticated vector sufficient stats."""

    return _aggregate_vector_stats(values)


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityMetricLedger:
    """Authenticated vector sufficient statistics for every fidelity surface."""

    local_operator: AttentionFidelityVectorStats
    local_postnorm_delta: AttentionFidelityVectorStats
    propagated_mutations: tuple[AttentionFidelityVectorStats, ...]
    final_valid_carrier: AttentionFidelityVectorStats
    supervised_carrier: AttentionFidelityVectorStats
    terminal_valid: AttentionFidelityVectorStats
    supervised_terminal: AttentionFidelityVectorStats
    valid_logits: AttentionFidelityVectorStats
    centered_valid_logits: AttentionFidelityVectorStats

    def __post_init__(self) -> None:
        scalar_surfaces = (
            self.local_operator,
            self.local_postnorm_delta,
            self.final_valid_carrier,
            self.supervised_carrier,
            self.terminal_valid,
            self.supervised_terminal,
            self.valid_logits,
            self.centered_valid_logits,
        )
        if any(
            not isinstance(item, AttentionFidelityVectorStats)
            for item in scalar_surfaces
        ):
            raise TypeError("fidelity metric surface is not typed")
        if (
            type(self.propagated_mutations) is not tuple
            or len(self.propagated_mutations) != 36
            or any(
                not isinstance(item, AttentionFidelityVectorStats)
                for item in self.propagated_mutations
            )
        ):
            raise ValueError(
                "fidelity metrics must persist all 36 propagated mutations"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "local_operator": self.local_operator.to_dict(),
            "local_postnorm_delta": self.local_postnorm_delta.to_dict(),
            "propagated_mutations": [
                item.to_dict() for item in self.propagated_mutations
            ],
            "final_valid_carrier": self.final_valid_carrier.to_dict(),
            "supervised_carrier": self.supervised_carrier.to_dict(),
            "terminal_valid": self.terminal_valid.to_dict(),
            "supervised_terminal": self.supervised_terminal.to_dict(),
            "valid_logits": self.valid_logits.to_dict(),
            "centered_valid_logits": self.centered_valid_logits.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityMetricLedger":
        expected = {
            "local_operator",
            "local_postnorm_delta",
            "propagated_mutations",
            "final_valid_carrier",
            "supervised_carrier",
            "terminal_valid",
            "supervised_terminal",
            "valid_logits",
            "centered_valid_logits",
        }
        _exact_keys(value, expected, label="fidelity metric ledger")

        def parsed(label: str) -> AttentionFidelityVectorStats:
            item = value[label]
            if not isinstance(item, Mapping):
                raise ValueError(f"{label} statistics must be an object")
            return AttentionFidelityVectorStats.from_dict(item)

        mutations = value["propagated_mutations"]
        if not isinstance(mutations, list):
            raise ValueError("propagated mutation statistics must be an array")
        parsed_mutations: list[AttentionFidelityVectorStats] = []
        for item in mutations:
            if not isinstance(item, Mapping):
                raise ValueError("propagated mutation statistic must be an object")
            parsed_mutations.append(AttentionFidelityVectorStats.from_dict(item))
        return cls(
            local_operator=parsed("local_operator"),
            local_postnorm_delta=parsed("local_postnorm_delta"),
            propagated_mutations=tuple(parsed_mutations),
            final_valid_carrier=parsed("final_valid_carrier"),
            supervised_carrier=parsed("supervised_carrier"),
            terminal_valid=parsed("terminal_valid"),
            supervised_terminal=parsed("supervised_terminal"),
            valid_logits=parsed("valid_logits"),
            centered_valid_logits=parsed("centered_valid_logits"),
        )


def _aggregate_metric_ledgers(
    values: Sequence[AttentionEdgeFidelityMetricLedger],
) -> AttentionEdgeFidelityMetricLedger:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty fidelity metric ledger")
    return AttentionEdgeFidelityMetricLedger(
        local_operator=_aggregate_vector_stats(
            [item.local_operator for item in materialized]
        ),
        local_postnorm_delta=_aggregate_vector_stats(
            [item.local_postnorm_delta for item in materialized]
        ),
        propagated_mutations=tuple(
            _aggregate_vector_stats(
                [item.propagated_mutations[index] for item in materialized]
            )
            for index in range(36)
        ),
        final_valid_carrier=_aggregate_vector_stats(
            [item.final_valid_carrier for item in materialized]
        ),
        supervised_carrier=_aggregate_vector_stats(
            [item.supervised_carrier for item in materialized]
        ),
        terminal_valid=_aggregate_vector_stats(
            [item.terminal_valid for item in materialized]
        ),
        supervised_terminal=_aggregate_vector_stats(
            [item.supervised_terminal for item in materialized]
        ),
        valid_logits=_aggregate_vector_stats(
            [item.valid_logits for item in materialized]
        ),
        centered_valid_logits=_aggregate_vector_stats(
            [item.centered_valid_logits for item in materialized]
        ),
    )


def _require_stats_match_legacy(
    stats: AttentionFidelityVectorStats,
    *,
    reference_sum_squares: float,
    sparse_sum_squares: float,
    error_sum_squares: float,
    dot_product: float,
    maximum_abs_error: float | None,
    label: str,
) -> None:
    expected = (
        stats.reference_sum_squares,
        stats.sparse_sum_squares,
        stats.error_sum_squares,
        stats.dot_product,
    )
    supplied = (
        reference_sum_squares,
        sparse_sum_squares,
        error_sum_squares,
        dot_product,
    )
    if supplied != expected or (
        maximum_abs_error is not None
        and maximum_abs_error != stats.maximum_abs_error
    ):
        raise ValueError(f"{label} scalar/statistic ledger drifted")


@dataclass(frozen=True, slots=True, order=True)
class AttentionEdgeFidelityCondition:
    ranker: str
    k: int
    variant: str
    random_salt: str = "primary"

    def __post_init__(self) -> None:
        if self.ranker not in ATTENTION_EDGE_FIDELITY_RANKERS:
            raise ValueError("attention fidelity ranker is invalid")
        if self.k not in ATTENTION_EDGE_FIDELITY_K_VALUES:
            raise ValueError("attention fidelity K is not frozen")
        if self.variant not in ATTENTION_EDGE_FIDELITY_VARIANTS:
            raise ValueError("attention fidelity variant is invalid")
        if self.random_salt not in ATTENTION_EDGE_FIDELITY_RANDOM_SALTS:
            raise ValueError("attention fidelity random salt is invalid")
        if self.ranker != "deterministic_random" and self.random_salt != "primary":
            raise ValueError("random replicates apply only to random ranking")
        if self.random_salt != "primary" and (
            self.k != ATTENTION_EDGE_TOP_K
            or self.variant != "unnormalized_contribution"
        ):
            raise ValueError("random replicates are frozen at K8 zero-fill")

    def to_dict(self) -> dict[str, object]:
        return {
            "ranker": self.ranker,
            "k": self.k,
            "variant": self.variant,
            "random_salt": self.random_salt,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityCondition":
        _exact_keys(
            value,
            {"ranker", "k", "variant", "random_salt"},
            label="condition",
        )
        return cls(
            ranker=str(value["ranker"]),
            k=_integer(value["k"], label="condition.k", minimum=1),
            variant=str(value["variant"]),
            random_salt=str(value["random_salt"]),
        )


def _conditions() -> tuple[AttentionEdgeFidelityCondition, ...]:
    primary_and_target_free = tuple(
        AttentionEdgeFidelityCondition(ranker, k, variant)
        for ranker in ATTENTION_EDGE_FIDELITY_RANKERS
        for k in ATTENTION_EDGE_FIDELITY_K_VALUES
        for variant in ATTENTION_EDGE_FIDELITY_VARIANTS
    )
    random_replicates = tuple(
        AttentionEdgeFidelityCondition(
            "deterministic_random",
            ATTENTION_EDGE_TOP_K,
            "unnormalized_contribution",
            random_salt,
        )
        for random_salt in ATTENTION_EDGE_FIDELITY_RANDOM_SALTS[1:]
    )
    return (*primary_and_target_free, *random_replicates)


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityMatchedControlDiagnostic:
    """Authenticated native-to-control diagnostic for one condition ordinal."""

    condition: AttentionEdgeFidelityCondition
    matched_control_receipt_sha256: str
    native_to_matched_valid_logit_max_abs_error: float
    absolute_nll_delta: float
    centered_logit_nrmse: float
    final_carrier_nrmse: float
    native_top1_parity_passed: bool
    diagnostic_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.condition, AttentionEdgeFidelityCondition):
            raise TypeError("matched control diagnostic condition is invalid")
        _require_sha256(
            self.matched_control_receipt_sha256,
            label="matched_control_receipt_sha256",
        )
        _nonnegative(
            self.native_to_matched_valid_logit_max_abs_error,
            label="native_to_matched_valid_logit_max_abs_error",
        )
        _nonnegative(self.absolute_nll_delta, label="absolute_nll_delta")
        _nonnegative(self.centered_logit_nrmse, label="centered_logit_nrmse")
        _nonnegative(self.final_carrier_nrmse, label="final_carrier_nrmse")
        if type(self.native_top1_parity_passed) is not bool:
            raise TypeError("native_top1_parity_passed must be bool")
        computed = _json_sha256(self._payload(), domain=_MATCHED_DIAGNOSTIC_DOMAIN)
        if self.diagnostic_sha256 == "":
            object.__setattr__(self, "diagnostic_sha256", computed)
        elif (
            _require_sha256(
                self.diagnostic_sha256, label="diagnostic_sha256"
            )
            != computed
        ):
            raise ValueError("matched control diagnostic hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "condition": self.condition.to_dict(),
            "matched_control_receipt_sha256": (
                self.matched_control_receipt_sha256
            ),
            "native_to_matched_valid_logit_max_abs_error": (
                self.native_to_matched_valid_logit_max_abs_error
            ),
            "absolute_nll_delta": self.absolute_nll_delta,
            "centered_logit_nrmse": self.centered_logit_nrmse,
            "final_carrier_nrmse": self.final_carrier_nrmse,
            "native_top1_parity_passed": self.native_top1_parity_passed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "diagnostic_sha256": self.diagnostic_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityMatchedControlDiagnostic":
        _exact_keys(
            value,
            {
                "condition",
                "matched_control_receipt_sha256",
                "native_to_matched_valid_logit_max_abs_error",
                "absolute_nll_delta",
                "centered_logit_nrmse",
                "final_carrier_nrmse",
                "native_top1_parity_passed",
                "diagnostic_sha256",
            },
            label="matched control diagnostic",
        )
        condition = value["condition"]
        if not isinstance(condition, Mapping):
            raise ValueError("matched control diagnostic condition must be an object")
        if type(value["native_top1_parity_passed"]) is not bool:
            raise TypeError("native_top1_parity_passed must be bool")
        return cls(
            condition=AttentionEdgeFidelityCondition.from_dict(condition),
            matched_control_receipt_sha256=str(
                value["matched_control_receipt_sha256"]
            ),
            native_to_matched_valid_logit_max_abs_error=_nonnegative(
                value["native_to_matched_valid_logit_max_abs_error"],
                label="native_to_matched_valid_logit_max_abs_error",
            ),
            absolute_nll_delta=_nonnegative(
                value["absolute_nll_delta"], label="absolute_nll_delta"
            ),
            centered_logit_nrmse=_nonnegative(
                value["centered_logit_nrmse"], label="centered_logit_nrmse"
            ),
            final_carrier_nrmse=_nonnegative(
                value["final_carrier_nrmse"], label="final_carrier_nrmse"
            ),
            native_top1_parity_passed=value["native_top1_parity_passed"],
            diagnostic_sha256=str(value["diagnostic_sha256"]),
        )


def _matched_control_diagnostic_catalog_sha256(
    values: Sequence[AttentionEdgeFidelityMatchedControlDiagnostic],
) -> str:
    return _json_sha256(
        {"diagnostics": [item.to_dict() for item in values]},
        domain=_MATCHED_DIAGNOSTIC_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class AttentionRouteSelection:
    ranker: str
    k: int
    random_salt: str
    mask: Tensor
    available_route_count: int
    selected_route_count: int
    active_head_query_count: int
    total_head_query_count: int
    selection_sha256: str

    def __post_init__(self) -> None:
        if self.ranker not in ATTENTION_EDGE_FIDELITY_RANKERS:
            raise ValueError("route-selection ranker is invalid")
        if self.k not in ATTENTION_EDGE_FIDELITY_K_VALUES:
            raise ValueError("route-selection K is invalid")
        if self.random_salt not in ATTENTION_EDGE_FIDELITY_RANDOM_SALTS:
            raise ValueError("route-selection random salt is invalid")
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype is not torch.bool
            or self.mask.device.type != "cpu"
            or self.mask.ndim != 4
            or self.mask.shape[0] != 1
        ):
            raise ValueError("route-selection mask must be CPU bool [1,H,Q,K]")
        available = _integer(
            self.available_route_count,
            label="available_route_count",
        )
        selected = _integer(
            self.selected_route_count,
            label="selected_route_count",
        )
        active = _integer(
            self.active_head_query_count,
            label="active_head_query_count",
        )
        total = _integer(
            self.total_head_query_count,
            label="total_head_query_count",
        )
        if selected != int(self.mask.sum().item()) or selected > available:
            raise ValueError("route-selection count closure failed")
        if active > total:
            raise ValueError("route-selection active-head count is invalid")
        computed = _json_sha256(
            {
                "ranker": self.ranker,
                "k": self.k,
                "random_salt": self.random_salt,
                "mask_sha256": _tensor_sha256(self.mask),
                "available_route_count": available,
                "selected_route_count": selected,
                "active_head_query_count": active,
                "total_head_query_count": total,
            },
            domain=_SELECTION_DOMAIN,
        )
        if _require_sha256(
            self.selection_sha256, label="selection_sha256"
        ) != computed:
            raise ValueError("route-selection hash mismatch")
        object.__setattr__(self, "mask", self.mask.detach().clone().contiguous())


def select_attention_routes(
    probabilities: Tensor,
    intrinsic_fisher_energies: Tensor,
    allowed_mask: Tensor,
    *,
    ranker: str,
    k: int,
    row_catalog_id: str,
    layer_ordinal: int,
    route_output_energies: Tensor | None = None,
    random_salt: str = "primary",
) -> AttentionRouteSelection:
    """Select global ``(head,key)`` routes per query with deterministic ties."""

    if (
        not isinstance(probabilities, Tensor)
        or probabilities.device.type != "cpu"
        or probabilities.ndim != 4
        or probabilities.shape[0] != 1
        or not probabilities.is_floating_point()
        or not bool(torch.isfinite(probabilities).all())
        or bool((probabilities < 0).any())
        or not isinstance(intrinsic_fisher_energies, Tensor)
        or intrinsic_fisher_energies.shape != probabilities.shape
        or intrinsic_fisher_energies.device.type != "cpu"
        or not intrinsic_fisher_energies.is_floating_point()
        or not bool(torch.isfinite(intrinsic_fisher_energies).all())
        or bool((intrinsic_fisher_energies < 0).any())
        or not isinstance(allowed_mask, Tensor)
        or allowed_mask.dtype is not torch.bool
        or allowed_mask.device.type != "cpu"
        or allowed_mask.shape
        != (1, probabilities.shape[2], probabilities.shape[3])
    ):
        raise ValueError("attention route-selection tensors are invalid")
    if route_output_energies is None:
        if ranker == "route_output_energy":
            raise ValueError("route-output ranking requires route energies")
        route_output_energies = torch.zeros_like(
            intrinsic_fisher_energies, dtype=torch.float64
        )
    mask = fidelity_kernel.select_global_attention_route_mask(
        probabilities,
        intrinsic_fisher_energies,
        route_output_energies,
        allowed_mask,
        ranker=ranker,
        k=k,
        row_catalog_id=row_catalog_id,
        layer_ordinal=layer_ordinal,
        random_salt=random_salt,
    )
    heads = int(probabilities.shape[1])
    queries = int(probabilities.shape[2])
    available = 0
    total_head_queries = 0
    for query in range(queries):
        keys = tuple(
            int(item)
            for item in torch.nonzero(
                allowed_mask[0, query], as_tuple=False
            ).reshape(-1).tolist()
        )
        if not keys:
            continue
        available += heads * len(keys)
        total_head_queries += heads
    selected = int(mask.sum().item())
    active = int(mask.any(dim=-1).sum().item())
    payload = {
        "ranker": ranker,
        "k": k,
        "random_salt": random_salt,
        "mask_sha256": _tensor_sha256(mask),
        "available_route_count": available,
        "selected_route_count": selected,
        "active_head_query_count": active,
        "total_head_query_count": total_head_queries,
    }
    return AttentionRouteSelection(
        ranker=ranker,
        k=k,
        random_salt=random_salt,
        mask=mask,
        available_route_count=available,
        selected_route_count=selected,
        active_head_query_count=active,
        total_head_query_count=total_head_queries,
        selection_sha256=_json_sha256(payload, domain=_SELECTION_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityMeasurement:
    """One row's sufficient statistics for one frozen curve condition."""

    condition: AttentionEdgeFidelityCondition
    selection_catalog_sha256: str
    matched_control_receipt_sha256: str
    available_route_count: int
    selected_route_count: int
    active_head_query_count: int
    total_head_query_count: int
    selected_probability_mass: float
    native_probability_mass: float
    selected_fisher_energy: float
    full_fisher_energy: float
    operator_reference_sum_squares: float
    operator_sparse_sum_squares: float
    operator_error_sum_squares: float
    operator_dot_product: float
    operator_max_abs_error: float
    final_carrier_reference_sum_squares: float
    final_carrier_sparse_sum_squares: float
    final_carrier_error_sum_squares: float
    final_carrier_dot_product: float
    final_carrier_max_abs_error: float
    terminal_reference_sum_squares: float
    terminal_sparse_sum_squares: float
    terminal_error_sum_squares: float
    terminal_dot_product: float
    terminal_max_abs_error: float
    baseline_valid_logit_sum_squares: float
    sparse_valid_logit_sum_squares: float
    valid_logit_error_sum_squares: float
    valid_logit_dot_product: float
    metrics: AttentionEdgeFidelityMetricLedger
    baseline_nll: float
    sparse_nll: float
    top1_retained: bool
    baseline_target_correct: bool
    sparse_target_correct: bool

    def __post_init__(self) -> None:
        if not isinstance(self.condition, AttentionEdgeFidelityCondition):
            raise TypeError("attention fidelity measurement condition is invalid")
        _require_sha256(
            self.selection_catalog_sha256,
            label="selection_catalog_sha256",
        )
        _require_sha256(
            self.matched_control_receipt_sha256,
            label="matched_control_receipt_sha256",
        )
        available = _integer(
            self.available_route_count, label="available_route_count"
        )
        selected = _integer(
            self.selected_route_count, label="selected_route_count"
        )
        active = _integer(
            self.active_head_query_count, label="active_head_query_count"
        )
        total = _integer(
            self.total_head_query_count, label="total_head_query_count"
        )
        if selected > available or active > total:
            raise ValueError("attention fidelity selection counts do not close")
        if available == 0 or total == 0:
            raise ValueError("attention fidelity selection support must be nonempty")
        for label, value in (
            ("selected_probability_mass", self.selected_probability_mass),
            ("native_probability_mass", self.native_probability_mass),
            ("selected_fisher_energy", self.selected_fisher_energy),
            ("full_fisher_energy", self.full_fisher_energy),
            (
                "operator_reference_sum_squares",
                self.operator_reference_sum_squares,
            ),
            ("operator_sparse_sum_squares", self.operator_sparse_sum_squares),
            ("operator_error_sum_squares", self.operator_error_sum_squares),
            ("operator_max_abs_error", self.operator_max_abs_error),
            (
                "final_carrier_reference_sum_squares",
                self.final_carrier_reference_sum_squares,
            ),
            (
                "final_carrier_sparse_sum_squares",
                self.final_carrier_sparse_sum_squares,
            ),
            (
                "final_carrier_error_sum_squares",
                self.final_carrier_error_sum_squares,
            ),
            ("final_carrier_max_abs_error", self.final_carrier_max_abs_error),
            (
                "terminal_reference_sum_squares",
                self.terminal_reference_sum_squares,
            ),
            ("terminal_sparse_sum_squares", self.terminal_sparse_sum_squares),
            ("terminal_error_sum_squares", self.terminal_error_sum_squares),
            ("terminal_max_abs_error", self.terminal_max_abs_error),
            (
                "baseline_valid_logit_sum_squares",
                self.baseline_valid_logit_sum_squares,
            ),
            (
                "sparse_valid_logit_sum_squares",
                self.sparse_valid_logit_sum_squares,
            ),
            (
                "valid_logit_error_sum_squares",
                self.valid_logit_error_sum_squares,
            ),
            ("baseline_nll", self.baseline_nll),
            ("sparse_nll", self.sparse_nll),
        ):
            _nonnegative(value, label=label)
        _finite(self.operator_dot_product, label="operator_dot_product")
        _finite(
            self.final_carrier_dot_product,
            label="final_carrier_dot_product",
        )
        _finite(self.terminal_dot_product, label="terminal_dot_product")
        _finite(self.valid_logit_dot_product, label="valid_logit_dot_product")
        if not isinstance(self.metrics, AttentionEdgeFidelityMetricLedger):
            raise TypeError("attention fidelity measurement metrics are invalid")
        _require_stats_match_legacy(
            self.metrics.local_operator,
            reference_sum_squares=self.operator_reference_sum_squares,
            sparse_sum_squares=self.operator_sparse_sum_squares,
            error_sum_squares=self.operator_error_sum_squares,
            dot_product=self.operator_dot_product,
            maximum_abs_error=self.operator_max_abs_error,
            label="local operator",
        )
        _require_stats_match_legacy(
            self.metrics.final_valid_carrier,
            reference_sum_squares=self.final_carrier_reference_sum_squares,
            sparse_sum_squares=self.final_carrier_sparse_sum_squares,
            error_sum_squares=self.final_carrier_error_sum_squares,
            dot_product=self.final_carrier_dot_product,
            maximum_abs_error=self.final_carrier_max_abs_error,
            label="final valid carrier",
        )
        _require_stats_match_legacy(
            self.metrics.terminal_valid,
            reference_sum_squares=self.terminal_reference_sum_squares,
            sparse_sum_squares=self.terminal_sparse_sum_squares,
            error_sum_squares=self.terminal_error_sum_squares,
            dot_product=self.terminal_dot_product,
            maximum_abs_error=self.terminal_max_abs_error,
            label="terminal valid",
        )
        _require_stats_match_legacy(
            self.metrics.valid_logits,
            reference_sum_squares=self.baseline_valid_logit_sum_squares,
            sparse_sum_squares=self.sparse_valid_logit_sum_squares,
            error_sum_squares=self.valid_logit_error_sum_squares,
            dot_product=self.valid_logit_dot_product,
            maximum_abs_error=None,
            label="valid logits",
        )
        for label, value in (
            ("top1_retained", self.top1_retained),
            ("baseline_target_correct", self.baseline_target_correct),
            ("sparse_target_correct", self.sparse_target_correct),
        ):
            if type(value) is not bool:
                raise TypeError(f"{label} must be bool")
        if self.selected_probability_mass > self.native_probability_mass + 1e-9:
            raise ValueError("selected probability exceeds native mass")
        if self.native_probability_mass <= 0.0:
            raise ValueError("native probability mass must be positive")
        if self.selected_fisher_energy > self.full_fisher_energy + 1e-9:
            raise ValueError("selected Fisher energy exceeds full energy")

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition.to_dict(),
            "selection_catalog_sha256": self.selection_catalog_sha256,
            "matched_control_receipt_sha256": (
                self.matched_control_receipt_sha256
            ),
            "available_route_count": self.available_route_count,
            "selected_route_count": self.selected_route_count,
            "active_head_query_count": self.active_head_query_count,
            "total_head_query_count": self.total_head_query_count,
            "selected_probability_mass": self.selected_probability_mass,
            "native_probability_mass": self.native_probability_mass,
            "selected_fisher_energy": self.selected_fisher_energy,
            "full_fisher_energy": self.full_fisher_energy,
            "operator_reference_sum_squares": (
                self.operator_reference_sum_squares
            ),
            "operator_sparse_sum_squares": self.operator_sparse_sum_squares,
            "operator_error_sum_squares": self.operator_error_sum_squares,
            "operator_dot_product": self.operator_dot_product,
            "operator_max_abs_error": self.operator_max_abs_error,
            "final_carrier_reference_sum_squares": (
                self.final_carrier_reference_sum_squares
            ),
            "final_carrier_sparse_sum_squares": (
                self.final_carrier_sparse_sum_squares
            ),
            "final_carrier_error_sum_squares": (
                self.final_carrier_error_sum_squares
            ),
            "final_carrier_dot_product": self.final_carrier_dot_product,
            "final_carrier_max_abs_error": self.final_carrier_max_abs_error,
            "terminal_reference_sum_squares": (
                self.terminal_reference_sum_squares
            ),
            "terminal_sparse_sum_squares": self.terminal_sparse_sum_squares,
            "terminal_error_sum_squares": self.terminal_error_sum_squares,
            "terminal_dot_product": self.terminal_dot_product,
            "terminal_max_abs_error": self.terminal_max_abs_error,
            "baseline_valid_logit_sum_squares": (
                self.baseline_valid_logit_sum_squares
            ),
            "sparse_valid_logit_sum_squares": (
                self.sparse_valid_logit_sum_squares
            ),
            "valid_logit_error_sum_squares": (
                self.valid_logit_error_sum_squares
            ),
            "valid_logit_dot_product": self.valid_logit_dot_product,
            "metrics": self.metrics.to_dict(),
            "baseline_nll": self.baseline_nll,
            "sparse_nll": self.sparse_nll,
            "top1_retained": self.top1_retained,
            "baseline_target_correct": self.baseline_target_correct,
            "sparse_target_correct": self.sparse_target_correct,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityMeasurement":
        expected = {
            "condition",
            "selection_catalog_sha256",
            "matched_control_receipt_sha256",
            "available_route_count",
            "selected_route_count",
            "active_head_query_count",
            "total_head_query_count",
            "selected_probability_mass",
            "native_probability_mass",
            "selected_fisher_energy",
            "full_fisher_energy",
            "operator_reference_sum_squares",
            "operator_sparse_sum_squares",
            "operator_error_sum_squares",
            "operator_dot_product",
            "operator_max_abs_error",
            "final_carrier_reference_sum_squares",
            "final_carrier_sparse_sum_squares",
            "final_carrier_error_sum_squares",
            "final_carrier_dot_product",
            "final_carrier_max_abs_error",
            "terminal_reference_sum_squares",
            "terminal_sparse_sum_squares",
            "terminal_error_sum_squares",
            "terminal_dot_product",
            "terminal_max_abs_error",
            "baseline_valid_logit_sum_squares",
            "sparse_valid_logit_sum_squares",
            "valid_logit_error_sum_squares",
            "valid_logit_dot_product",
            "metrics",
            "baseline_nll",
            "sparse_nll",
            "top1_retained",
            "baseline_target_correct",
            "sparse_target_correct",
        }
        _exact_keys(value, expected, label="attention fidelity measurement")
        condition = value["condition"]
        if not isinstance(condition, Mapping):
            raise ValueError("attention fidelity condition must be an object")
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError("attention fidelity metrics must be an object")
        for flag in (
            "top1_retained",
            "baseline_target_correct",
            "sparse_target_correct",
        ):
            if type(value[flag]) is not bool:
                raise TypeError(f"{flag} must be bool")
        return cls(
            condition=AttentionEdgeFidelityCondition.from_dict(condition),
            selection_catalog_sha256=str(value["selection_catalog_sha256"]),
            matched_control_receipt_sha256=str(
                value["matched_control_receipt_sha256"]
            ),
            available_route_count=_integer(
                value["available_route_count"], label="available_route_count"
            ),
            selected_route_count=_integer(
                value["selected_route_count"], label="selected_route_count"
            ),
            active_head_query_count=_integer(
                value["active_head_query_count"],
                label="active_head_query_count",
            ),
            total_head_query_count=_integer(
                value["total_head_query_count"],
                label="total_head_query_count",
            ),
            selected_probability_mass=_finite(
                value["selected_probability_mass"],
                label="selected_probability_mass",
            ),
            native_probability_mass=_finite(
                value["native_probability_mass"],
                label="native_probability_mass",
            ),
            selected_fisher_energy=_finite(
                value["selected_fisher_energy"],
                label="selected_fisher_energy",
            ),
            full_fisher_energy=_finite(
                value["full_fisher_energy"], label="full_fisher_energy"
            ),
            operator_reference_sum_squares=_finite(
                value["operator_reference_sum_squares"],
                label="operator_reference_sum_squares",
            ),
            operator_sparse_sum_squares=_finite(
                value["operator_sparse_sum_squares"],
                label="operator_sparse_sum_squares",
            ),
            operator_error_sum_squares=_finite(
                value["operator_error_sum_squares"],
                label="operator_error_sum_squares",
            ),
            operator_dot_product=_finite(
                value["operator_dot_product"], label="operator_dot_product"
            ),
            operator_max_abs_error=_finite(
                value["operator_max_abs_error"],
                label="operator_max_abs_error",
            ),
            final_carrier_reference_sum_squares=_finite(
                value["final_carrier_reference_sum_squares"],
                label="final_carrier_reference_sum_squares",
            ),
            final_carrier_sparse_sum_squares=_finite(
                value["final_carrier_sparse_sum_squares"],
                label="final_carrier_sparse_sum_squares",
            ),
            final_carrier_error_sum_squares=_finite(
                value["final_carrier_error_sum_squares"],
                label="final_carrier_error_sum_squares",
            ),
            final_carrier_dot_product=_finite(
                value["final_carrier_dot_product"],
                label="final_carrier_dot_product",
            ),
            final_carrier_max_abs_error=_finite(
                value["final_carrier_max_abs_error"],
                label="final_carrier_max_abs_error",
            ),
            terminal_reference_sum_squares=_finite(
                value["terminal_reference_sum_squares"],
                label="terminal_reference_sum_squares",
            ),
            terminal_sparse_sum_squares=_finite(
                value["terminal_sparse_sum_squares"],
                label="terminal_sparse_sum_squares",
            ),
            terminal_error_sum_squares=_finite(
                value["terminal_error_sum_squares"],
                label="terminal_error_sum_squares",
            ),
            terminal_dot_product=_finite(
                value["terminal_dot_product"],
                label="terminal_dot_product",
            ),
            terminal_max_abs_error=_finite(
                value["terminal_max_abs_error"],
                label="terminal_max_abs_error",
            ),
            baseline_valid_logit_sum_squares=_finite(
                value["baseline_valid_logit_sum_squares"],
                label="baseline_valid_logit_sum_squares",
            ),
            sparse_valid_logit_sum_squares=_finite(
                value["sparse_valid_logit_sum_squares"],
                label="sparse_valid_logit_sum_squares",
            ),
            valid_logit_error_sum_squares=_finite(
                value["valid_logit_error_sum_squares"],
                label="valid_logit_error_sum_squares",
            ),
            valid_logit_dot_product=_finite(
                value["valid_logit_dot_product"],
                label="valid_logit_dot_product",
            ),
            metrics=AttentionEdgeFidelityMetricLedger.from_dict(metrics),
            baseline_nll=_nonnegative(
                value["baseline_nll"], label="baseline_nll"
            ),
            sparse_nll=_nonnegative(value["sparse_nll"], label="sparse_nll"),
            top1_retained=value["top1_retained"],
            baseline_target_correct=value["baseline_target_correct"],
            sparse_target_correct=value["sparse_target_correct"],
        )


def _validate_matched_control_diagnostic_ledger(
    diagnostics: tuple[AttentionEdgeFidelityMatchedControlDiagnostic, ...],
    measurements: tuple[AttentionEdgeFidelityMeasurement, ...],
    *,
    catalog_sha256: str,
    native_baseline_nll: float,
) -> tuple[float, float, float, float, bool]:
    if (
        type(diagnostics) is not tuple
        or any(
            not isinstance(
                item, AttentionEdgeFidelityMatchedControlDiagnostic
            )
            for item in diagnostics
        )
    ):
        raise ValueError("matched control diagnostic inventory drifted")
    if tuple(item.condition for item in diagnostics) != _conditions():
        raise ValueError("matched control diagnostic inventory drifted")
    if len(measurements) != len(diagnostics):
        raise ValueError("matched control diagnostic/measurement count drifted")
    for diagnostic, measurement in zip(diagnostics, measurements, strict=True):
        if (
            diagnostic.condition != measurement.condition
            or diagnostic.matched_control_receipt_sha256
            != measurement.matched_control_receipt_sha256
        ):
            raise ValueError(
                "matched control diagnostic does not bind its measurement"
            )
        if diagnostic.absolute_nll_delta != abs(
            measurement.baseline_nll - native_baseline_nll
        ):
            raise ValueError("matched control diagnostic NLL delta drifted")
    expected_catalog = _matched_control_diagnostic_catalog_sha256(diagnostics)
    if catalog_sha256 != expected_catalog:
        raise ValueError("matched control diagnostic catalog drifted")
    return (
        max(
            item.native_to_matched_valid_logit_max_abs_error
            for item in diagnostics
        ),
        max(item.absolute_nll_delta for item in diagnostics),
        max(item.centered_logit_nrmse for item in diagnostics),
        max(item.final_carrier_nrmse for item in diagnostics),
        all(item.native_top1_parity_passed for item in diagnostics),
    )


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityRowCapture:
    row_catalog_id: str
    family_id: str
    sequence_length: int
    tokenization_sha256: str
    layer_extraction_sha256s: tuple[str, ...]
    dense_replay_valid_logits_sha256: str
    baseline_valid_logits_sha256: str
    matched_batch_size: int
    matched_control_catalog_sha256: str
    matched_control_diagnostics: tuple[
        AttentionEdgeFidelityMatchedControlDiagnostic, ...
    ]
    matched_control_diagnostic_catalog_sha256: str
    native_baseline_nll: float
    matched_controls_native_top1_parity_passed: bool
    native_to_matched_valid_logit_max_abs_error: float
    native_to_matched_nll_max_abs_delta: float
    native_to_matched_centered_logit_max_nrmse: float
    native_to_matched_final_carrier_max_nrmse: float
    matched_control_nll_max_abs_delta_bound: float
    matched_control_centered_logit_nrmse_bound: float
    matched_control_final_carrier_nrmse_bound: float
    attention_operator_catalog_sha256: str
    attention_delta_catalog_sha256: str
    mutation_delta_catalog_sha256: str
    mutation_carrier_catalog_sha256: str
    attention_operator_sha256s: tuple[str, ...]
    attention_delta_sha256s: tuple[str, ...]
    mutation_delta_sha256s: tuple[str, ...]
    mutation_carrier_sha256s: tuple[str, ...]
    final_carrier_sha256: str
    terminal_normalized_output_sha256: str
    dense_replay_literal_parity_passed: bool
    dense_replay_nll_literal_parity_passed: bool
    dense_replay_top1_parity_passed: bool
    all_routes_operator_delta_controls_passed: bool
    selection_prefix_nesting_passed: bool
    measurements: tuple[AttentionEdgeFidelityMeasurement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.row_catalog_id, str) or not self.row_catalog_id:
            raise ValueError("fidelity row_catalog_id must be nonempty")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("fidelity family_id must be nonempty")
        _integer(self.sequence_length, label="sequence_length", minimum=1)
        _require_sha256(self.tokenization_sha256, label="tokenization_sha256")
        if (
            type(self.layer_extraction_sha256s) is not tuple
            or len(self.layer_extraction_sha256s) != GEMMA3_LAYER_COUNT
        ):
            raise ValueError("fidelity row must bind all layer extractions")
        for digest in self.layer_extraction_sha256s:
            _require_sha256(digest, label="layer extraction")
        _require_sha256(
            self.dense_replay_valid_logits_sha256,
            label="dense_replay_valid_logits_sha256",
        )
        _require_sha256(
            self.baseline_valid_logits_sha256,
            label="baseline_valid_logits_sha256",
        )
        if self.matched_batch_size != _matched_batch_size():
            raise ValueError("matched fidelity batch size drifted")
        _require_sha256(
            self.matched_control_catalog_sha256,
            label="matched_control_catalog_sha256",
        )
        _require_sha256(
            self.matched_control_diagnostic_catalog_sha256,
            label="matched_control_diagnostic_catalog_sha256",
        )
        if self.matched_controls_native_top1_parity_passed is not True:
            raise ValueError("matched all-routes controls changed native top1")
        native_nll = _nonnegative(
            self.native_baseline_nll, label="native_baseline_nll"
        )
        _nonnegative(
            self.native_to_matched_valid_logit_max_abs_error,
            label="native_to_matched_valid_logit_max_abs_error",
        )
        maximum_nll_delta = _nonnegative(
            self.native_to_matched_nll_max_abs_delta,
            label="native_to_matched_nll_max_abs_delta",
        )
        maximum_centered_logit_nrmse = _nonnegative(
            self.native_to_matched_centered_logit_max_nrmse,
            label="native_to_matched_centered_logit_max_nrmse",
        )
        maximum_final_carrier_nrmse = _nonnegative(
            self.native_to_matched_final_carrier_max_nrmse,
            label="native_to_matched_final_carrier_max_nrmse",
        )
        if (
            self.matched_control_nll_max_abs_delta_bound
            != ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA
            or self.matched_control_centered_logit_nrmse_bound
            != ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE
            or self.matched_control_final_carrier_nrmse_bound
            != ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE
        ):
            raise ValueError("matched control validity bounds drifted")
        _validate_matched_control_bounds(
            maximum_absolute_nll_delta=maximum_nll_delta,
            maximum_centered_logit_nrmse=maximum_centered_logit_nrmse,
            maximum_final_carrier_nrmse=maximum_final_carrier_nrmse,
        )
        for label, digest in (
            ("attention operator catalog", self.attention_operator_catalog_sha256),
            ("attention delta catalog", self.attention_delta_catalog_sha256),
            ("mutation delta catalog", self.mutation_delta_catalog_sha256),
            ("mutation carrier catalog", self.mutation_carrier_catalog_sha256),
            ("final carrier", self.final_carrier_sha256),
            ("terminal normalized output", self.terminal_normalized_output_sha256),
        ):
            _require_sha256(digest, label=label)
        hash_ledgers = (
            (
                "attention_operator",
                self.attention_operator_sha256s,
                GEMMA3_LAYER_COUNT,
                self.attention_operator_catalog_sha256,
            ),
            (
                "attention_delta",
                self.attention_delta_sha256s,
                GEMMA3_LAYER_COUNT,
                self.attention_delta_catalog_sha256,
            ),
            (
                "mutation_delta",
                self.mutation_delta_sha256s,
                36,
                self.mutation_delta_catalog_sha256,
            ),
            (
                "mutation_carrier",
                self.mutation_carrier_sha256s,
                36,
                self.mutation_carrier_catalog_sha256,
            ),
        )
        for label, hashes, count, catalog in hash_ledgers:
            if type(hashes) is not tuple or len(hashes) != count:
                raise ValueError(f"{label} hash inventory drifted")
            for digest in hashes:
                _require_sha256(digest, label=f"{label} tensor")
            expected_catalog = _json_sha256(
                {"label": label, "tensor_sha256s": list(hashes)},
                domain=_TENSOR_DOMAIN,
            )
            if catalog != expected_catalog:
                raise ValueError(f"{label} hash catalog drifted")
        if self.attention_delta_sha256s != self.mutation_delta_sha256s[::2]:
            raise ValueError(
                "attention deltas do not bind the odd mutation deltas"
            )
        if self.final_carrier_sha256 != self.mutation_carrier_sha256s[-1]:
            raise ValueError("final carrier does not bind mutation 36")
        if self.dense_replay_literal_parity_passed is not True:
            raise ValueError("dense replay literal parity must pass")
        if (
            self.dense_replay_nll_literal_parity_passed is not True
            or self.dense_replay_top1_parity_passed is not True
        ):
            raise ValueError("dense replay task endpoint parity must pass")
        if self.all_routes_operator_delta_controls_passed is not True:
            raise ValueError("all-routes operator/delta controls must pass")
        if self.selection_prefix_nesting_passed is not True:
            raise ValueError("selection masks must form nested K prefixes")
        if (
            self.dense_replay_valid_logits_sha256
            != self.baseline_valid_logits_sha256
        ):
            raise ValueError("dense replay parity hashes disagree")
        if (
            type(self.measurements) is not tuple
            or tuple(item.condition for item in self.measurements) != _conditions()
        ):
            raise ValueError("fidelity row condition inventory drifted")
        expected_catalog = _json_sha256(
            [item.matched_control_receipt_sha256 for item in self.measurements],
            domain=_MATCHED_SURFACE_DOMAIN,
        )
        if self.matched_control_catalog_sha256 != expected_catalog:
            raise ValueError("matched control receipt catalog drifted")
        (
            observed_maximum_valid_logit_error,
            observed_maximum_nll_delta,
            observed_maximum_centered_logit_nrmse,
            observed_maximum_final_carrier_nrmse,
            observed_all_top1_parity,
        ) = _validate_matched_control_diagnostic_ledger(
            self.matched_control_diagnostics,
            self.measurements,
            catalog_sha256=self.matched_control_diagnostic_catalog_sha256,
            native_baseline_nll=native_nll,
        )
        if (
            self.native_to_matched_valid_logit_max_abs_error
            != observed_maximum_valid_logit_error
        ):
            raise ValueError(
                "native-to-matched valid-logit maximum drifted"
            )
        if maximum_nll_delta != observed_maximum_nll_delta:
            raise ValueError("native-to-matched NLL diagnostic drifted")
        if (
            maximum_centered_logit_nrmse
            != observed_maximum_centered_logit_nrmse
        ):
            raise ValueError("native-to-matched centered-logit maximum drifted")
        if (
            maximum_final_carrier_nrmse
            != observed_maximum_final_carrier_nrmse
        ):
            raise ValueError("native-to-matched final-carrier maximum drifted")
        if (
            self.matched_controls_native_top1_parity_passed
            != observed_all_top1_parity
        ):
            raise ValueError("matched-control all-top1 summary drifted")


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityRowReceipt:
    row_ordinal: int
    family_id: str
    sequence_length: int
    tokenization_sha256: str
    layer_extraction_sha256s: tuple[str, ...]
    dense_replay_valid_logits_sha256: str
    baseline_valid_logits_sha256: str
    matched_batch_size: int
    matched_control_catalog_sha256: str
    matched_control_diagnostics: tuple[
        AttentionEdgeFidelityMatchedControlDiagnostic, ...
    ]
    matched_control_diagnostic_catalog_sha256: str
    native_baseline_nll: float
    matched_controls_native_top1_parity_passed: bool
    native_to_matched_valid_logit_max_abs_error: float
    native_to_matched_nll_max_abs_delta: float
    native_to_matched_centered_logit_max_nrmse: float
    native_to_matched_final_carrier_max_nrmse: float
    matched_control_nll_max_abs_delta_bound: float
    matched_control_centered_logit_nrmse_bound: float
    matched_control_final_carrier_nrmse_bound: float
    attention_operator_catalog_sha256: str
    attention_delta_catalog_sha256: str
    mutation_delta_catalog_sha256: str
    mutation_carrier_catalog_sha256: str
    attention_operator_sha256s: tuple[str, ...]
    attention_delta_sha256s: tuple[str, ...]
    mutation_delta_sha256s: tuple[str, ...]
    mutation_carrier_sha256s: tuple[str, ...]
    final_carrier_sha256: str
    terminal_normalized_output_sha256: str
    dense_replay_literal_parity_passed: bool
    dense_replay_nll_literal_parity_passed: bool
    dense_replay_top1_parity_passed: bool
    all_routes_operator_delta_controls_passed: bool
    selection_prefix_nesting_passed: bool
    measurements: tuple[AttentionEdgeFidelityMeasurement, ...]
    measurement_catalog_sha256: str
    row_sha256: str = ""

    def __post_init__(self) -> None:
        _integer(self.row_ordinal, label="row_ordinal")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("row receipt family_id must be nonempty")
        _integer(self.sequence_length, label="sequence_length", minimum=1)
        _require_sha256(self.tokenization_sha256, label="tokenization_sha256")
        if (
            type(self.layer_extraction_sha256s) is not tuple
            or len(self.layer_extraction_sha256s) != GEMMA3_LAYER_COUNT
        ):
            raise ValueError("row receipt layer inventory drifted")
        for digest in self.layer_extraction_sha256s:
            _require_sha256(digest, label="layer extraction")
        for label, digest in (
            ("dense replay logits", self.dense_replay_valid_logits_sha256),
            ("baseline logits", self.baseline_valid_logits_sha256),
            ("matched control catalog", self.matched_control_catalog_sha256),
            (
                "matched control diagnostic catalog",
                self.matched_control_diagnostic_catalog_sha256,
            ),
            ("attention operator catalog", self.attention_operator_catalog_sha256),
            ("attention delta catalog", self.attention_delta_catalog_sha256),
            ("mutation delta catalog", self.mutation_delta_catalog_sha256),
            ("mutation carrier catalog", self.mutation_carrier_catalog_sha256),
            ("final carrier", self.final_carrier_sha256),
            ("terminal normalized output", self.terminal_normalized_output_sha256),
            ("measurement catalog", self.measurement_catalog_sha256),
        ):
            _require_sha256(digest, label=label)
        if self.matched_batch_size != _matched_batch_size():
            raise ValueError("row receipt matched batch size drifted")
        if self.matched_controls_native_top1_parity_passed is not True:
            raise ValueError("row receipt matched controls changed native top1")
        native_nll = _nonnegative(
            self.native_baseline_nll, label="native_baseline_nll"
        )
        _nonnegative(
            self.native_to_matched_valid_logit_max_abs_error,
            label="native_to_matched_valid_logit_max_abs_error",
        )
        maximum_nll_delta = _nonnegative(
            self.native_to_matched_nll_max_abs_delta,
            label="native_to_matched_nll_max_abs_delta",
        )
        maximum_centered_logit_nrmse = _nonnegative(
            self.native_to_matched_centered_logit_max_nrmse,
            label="native_to_matched_centered_logit_max_nrmse",
        )
        maximum_final_carrier_nrmse = _nonnegative(
            self.native_to_matched_final_carrier_max_nrmse,
            label="native_to_matched_final_carrier_max_nrmse",
        )
        if (
            self.matched_control_nll_max_abs_delta_bound
            != ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA
            or self.matched_control_centered_logit_nrmse_bound
            != ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE
            or self.matched_control_final_carrier_nrmse_bound
            != ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE
        ):
            raise ValueError("row receipt matched control validity bounds drifted")
        _validate_matched_control_bounds(
            maximum_absolute_nll_delta=maximum_nll_delta,
            maximum_centered_logit_nrmse=maximum_centered_logit_nrmse,
            maximum_final_carrier_nrmse=maximum_final_carrier_nrmse,
        )
        hash_ledgers = (
            (
                "attention_operator",
                self.attention_operator_sha256s,
                GEMMA3_LAYER_COUNT,
                self.attention_operator_catalog_sha256,
            ),
            (
                "attention_delta",
                self.attention_delta_sha256s,
                GEMMA3_LAYER_COUNT,
                self.attention_delta_catalog_sha256,
            ),
            (
                "mutation_delta",
                self.mutation_delta_sha256s,
                36,
                self.mutation_delta_catalog_sha256,
            ),
            (
                "mutation_carrier",
                self.mutation_carrier_sha256s,
                36,
                self.mutation_carrier_catalog_sha256,
            ),
        )
        for label, hashes, count, catalog in hash_ledgers:
            if type(hashes) is not tuple or len(hashes) != count:
                raise ValueError(f"row receipt {label} inventory drifted")
            for digest in hashes:
                _require_sha256(digest, label=f"row receipt {label}")
            if catalog != _json_sha256(
                {"label": label, "tensor_sha256s": list(hashes)},
                domain=_TENSOR_DOMAIN,
            ):
                raise ValueError(f"row receipt {label} catalog drifted")
        if self.attention_delta_sha256s != self.mutation_delta_sha256s[::2]:
            raise ValueError(
                "row receipt attention deltas do not bind odd mutations"
            )
        if self.final_carrier_sha256 != self.mutation_carrier_sha256s[-1]:
            raise ValueError("row receipt final carrier does not bind mutation 36")
        if (
            self.dense_replay_literal_parity_passed is not True
            or self.dense_replay_nll_literal_parity_passed is not True
            or self.dense_replay_top1_parity_passed is not True
            or self.all_routes_operator_delta_controls_passed is not True
            or self.selection_prefix_nesting_passed is not True
            or self.dense_replay_valid_logits_sha256
            != self.baseline_valid_logits_sha256
        ):
            raise ValueError("row receipt dense controls failed")
        if (
            type(self.measurements) is not tuple
            or tuple(item.condition for item in self.measurements) != _conditions()
        ):
            raise ValueError("row receipt measurement inventory drifted")
        expected_control_catalog = _json_sha256(
            [item.matched_control_receipt_sha256 for item in self.measurements],
            domain=_MATCHED_SURFACE_DOMAIN,
        )
        if self.matched_control_catalog_sha256 != expected_control_catalog:
            raise ValueError("row receipt matched control catalog drifted")
        (
            observed_maximum_valid_logit_error,
            observed_maximum_nll_delta,
            observed_maximum_centered_logit_nrmse,
            observed_maximum_final_carrier_nrmse,
            observed_all_top1_parity,
        ) = _validate_matched_control_diagnostic_ledger(
            self.matched_control_diagnostics,
            self.measurements,
            catalog_sha256=self.matched_control_diagnostic_catalog_sha256,
            native_baseline_nll=native_nll,
        )
        if (
            self.native_to_matched_valid_logit_max_abs_error
            != observed_maximum_valid_logit_error
        ):
            raise ValueError(
                "row receipt native-to-matched valid-logit maximum drifted"
            )
        if maximum_nll_delta != observed_maximum_nll_delta:
            raise ValueError("row receipt native-to-matched NLL diagnostic drifted")
        if (
            maximum_centered_logit_nrmse
            != observed_maximum_centered_logit_nrmse
        ):
            raise ValueError(
                "row receipt native-to-matched centered-logit maximum drifted"
            )
        if (
            maximum_final_carrier_nrmse
            != observed_maximum_final_carrier_nrmse
        ):
            raise ValueError(
                "row receipt native-to-matched final-carrier maximum drifted"
            )
        if (
            self.matched_controls_native_top1_parity_passed
            != observed_all_top1_parity
        ):
            raise ValueError("row receipt matched-control all-top1 summary drifted")
        computed_measurement_catalog = _json_sha256(
            [item.to_dict() for item in self.measurements],
            domain=_SELECTION_DOMAIN,
        )
        if self.measurement_catalog_sha256 != computed_measurement_catalog:
            raise ValueError("row receipt measurement catalog drifted")
        computed = _json_sha256(self._payload(), domain=_ROW_DOMAIN)
        if self.row_sha256 == "":
            object.__setattr__(self, "row_sha256", computed)
        elif _require_sha256(self.row_sha256, label="row_sha256") != computed:
            raise ValueError("attention fidelity row hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "family_id": self.family_id,
            "sequence_length": self.sequence_length,
            "tokenization_sha256": self.tokenization_sha256,
            "layer_extraction_sha256s": list(self.layer_extraction_sha256s),
            "dense_replay_valid_logits_sha256": (
                self.dense_replay_valid_logits_sha256
            ),
            "baseline_valid_logits_sha256": self.baseline_valid_logits_sha256,
            "matched_batch_size": self.matched_batch_size,
            "matched_control_catalog_sha256": (
                self.matched_control_catalog_sha256
            ),
            "matched_control_diagnostics": [
                item.to_dict() for item in self.matched_control_diagnostics
            ],
            "matched_control_diagnostic_catalog_sha256": (
                self.matched_control_diagnostic_catalog_sha256
            ),
            "native_baseline_nll": self.native_baseline_nll,
            "matched_controls_native_top1_parity_passed": (
                self.matched_controls_native_top1_parity_passed
            ),
            "native_to_matched_valid_logit_max_abs_error": (
                self.native_to_matched_valid_logit_max_abs_error
            ),
            "native_to_matched_nll_max_abs_delta": (
                self.native_to_matched_nll_max_abs_delta
            ),
            "native_to_matched_centered_logit_max_nrmse": (
                self.native_to_matched_centered_logit_max_nrmse
            ),
            "native_to_matched_final_carrier_max_nrmse": (
                self.native_to_matched_final_carrier_max_nrmse
            ),
            "matched_control_nll_max_abs_delta_bound": (
                self.matched_control_nll_max_abs_delta_bound
            ),
            "matched_control_centered_logit_nrmse_bound": (
                self.matched_control_centered_logit_nrmse_bound
            ),
            "matched_control_final_carrier_nrmse_bound": (
                self.matched_control_final_carrier_nrmse_bound
            ),
            "attention_operator_catalog_sha256": (
                self.attention_operator_catalog_sha256
            ),
            "attention_delta_catalog_sha256": self.attention_delta_catalog_sha256,
            "mutation_delta_catalog_sha256": (
                self.mutation_delta_catalog_sha256
            ),
            "mutation_carrier_catalog_sha256": (
                self.mutation_carrier_catalog_sha256
            ),
            "attention_operator_sha256s": list(
                self.attention_operator_sha256s
            ),
            "attention_delta_sha256s": list(self.attention_delta_sha256s),
            "mutation_delta_sha256s": list(self.mutation_delta_sha256s),
            "mutation_carrier_sha256s": list(self.mutation_carrier_sha256s),
            "final_carrier_sha256": self.final_carrier_sha256,
            "terminal_normalized_output_sha256": (
                self.terminal_normalized_output_sha256
            ),
            "dense_replay_literal_parity_passed": (
                self.dense_replay_literal_parity_passed
            ),
            "dense_replay_nll_literal_parity_passed": (
                self.dense_replay_nll_literal_parity_passed
            ),
            "dense_replay_top1_parity_passed": (
                self.dense_replay_top1_parity_passed
            ),
            "all_routes_operator_delta_controls_passed": (
                self.all_routes_operator_delta_controls_passed
            ),
            "selection_prefix_nesting_passed": (
                self.selection_prefix_nesting_passed
            ),
            "measurements": [item.to_dict() for item in self.measurements],
            "measurement_catalog_sha256": self.measurement_catalog_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "row_sha256": self.row_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityRowReceipt":
        expected = {
            "row_ordinal",
            "family_id",
            "sequence_length",
            "tokenization_sha256",
            "layer_extraction_sha256s",
            "dense_replay_valid_logits_sha256",
            "baseline_valid_logits_sha256",
            "matched_batch_size",
            "matched_control_catalog_sha256",
            "matched_control_diagnostics",
            "matched_control_diagnostic_catalog_sha256",
            "native_baseline_nll",
            "matched_controls_native_top1_parity_passed",
            "native_to_matched_valid_logit_max_abs_error",
            "native_to_matched_nll_max_abs_delta",
            "native_to_matched_centered_logit_max_nrmse",
            "native_to_matched_final_carrier_max_nrmse",
            "matched_control_nll_max_abs_delta_bound",
            "matched_control_centered_logit_nrmse_bound",
            "matched_control_final_carrier_nrmse_bound",
            "attention_operator_catalog_sha256",
            "attention_delta_catalog_sha256",
            "mutation_delta_catalog_sha256",
            "mutation_carrier_catalog_sha256",
            "attention_operator_sha256s",
            "attention_delta_sha256s",
            "mutation_delta_sha256s",
            "mutation_carrier_sha256s",
            "final_carrier_sha256",
            "terminal_normalized_output_sha256",
            "dense_replay_literal_parity_passed",
            "dense_replay_nll_literal_parity_passed",
            "dense_replay_top1_parity_passed",
            "all_routes_operator_delta_controls_passed",
            "selection_prefix_nesting_passed",
            "measurements",
            "measurement_catalog_sha256",
            "row_sha256",
        }
        _exact_keys(value, expected, label="attention fidelity row receipt")
        extraction_hashes = value["layer_extraction_sha256s"]
        measurements = value["measurements"]
        diagnostics = value["matched_control_diagnostics"]
        if not isinstance(extraction_hashes, list):
            raise ValueError("layer extraction hashes must be an array")
        if not isinstance(measurements, list):
            raise ValueError("row measurements must be an array")
        if not isinstance(diagnostics, list):
            raise ValueError("matched control diagnostics must be an array")
        parsed_measurements: list[AttentionEdgeFidelityMeasurement] = []
        for item in measurements:
            if not isinstance(item, Mapping):
                raise ValueError("row measurement must be an object")
            parsed_measurements.append(
                AttentionEdgeFidelityMeasurement.from_dict(item)
            )
        parsed_diagnostics: list[
            AttentionEdgeFidelityMatchedControlDiagnostic
        ] = []
        for item in diagnostics:
            if not isinstance(item, Mapping):
                raise ValueError("matched control diagnostic must be an object")
            parsed_diagnostics.append(
                AttentionEdgeFidelityMatchedControlDiagnostic.from_dict(item)
            )
        parsed_hash_ledgers: dict[str, tuple[str, ...]] = {}
        for label in (
            "attention_operator_sha256s",
            "attention_delta_sha256s",
            "mutation_delta_sha256s",
            "mutation_carrier_sha256s",
        ):
            items = value[label]
            if not isinstance(items, list):
                raise ValueError(f"{label} must be an array")
            parsed_hash_ledgers[label] = tuple(str(item) for item in items)
        for flag in (
            "dense_replay_literal_parity_passed",
            "dense_replay_nll_literal_parity_passed",
            "dense_replay_top1_parity_passed",
            "all_routes_operator_delta_controls_passed",
            "selection_prefix_nesting_passed",
            "matched_controls_native_top1_parity_passed",
        ):
            if type(value[flag]) is not bool:
                raise TypeError(f"{flag} must be bool")
        return cls(
            row_ordinal=_integer(value["row_ordinal"], label="row_ordinal"),
            family_id=str(value["family_id"]),
            sequence_length=_integer(
                value["sequence_length"], label="sequence_length", minimum=1
            ),
            tokenization_sha256=str(value["tokenization_sha256"]),
            layer_extraction_sha256s=tuple(str(item) for item in extraction_hashes),
            dense_replay_valid_logits_sha256=str(
                value["dense_replay_valid_logits_sha256"]
            ),
            baseline_valid_logits_sha256=str(
                value["baseline_valid_logits_sha256"]
            ),
            matched_batch_size=_integer(
                value["matched_batch_size"],
                label="matched_batch_size",
                minimum=1,
            ),
            matched_control_catalog_sha256=str(
                value["matched_control_catalog_sha256"]
            ),
            matched_control_diagnostics=tuple(parsed_diagnostics),
            matched_control_diagnostic_catalog_sha256=str(
                value["matched_control_diagnostic_catalog_sha256"]
            ),
            native_baseline_nll=_nonnegative(
                value["native_baseline_nll"], label="native_baseline_nll"
            ),
            matched_controls_native_top1_parity_passed=value[
                "matched_controls_native_top1_parity_passed"
            ],
            native_to_matched_valid_logit_max_abs_error=_nonnegative(
                value["native_to_matched_valid_logit_max_abs_error"],
                label="native_to_matched_valid_logit_max_abs_error",
            ),
            native_to_matched_nll_max_abs_delta=_nonnegative(
                value["native_to_matched_nll_max_abs_delta"],
                label="native_to_matched_nll_max_abs_delta",
            ),
            native_to_matched_centered_logit_max_nrmse=_nonnegative(
                value["native_to_matched_centered_logit_max_nrmse"],
                label="native_to_matched_centered_logit_max_nrmse",
            ),
            native_to_matched_final_carrier_max_nrmse=_nonnegative(
                value["native_to_matched_final_carrier_max_nrmse"],
                label="native_to_matched_final_carrier_max_nrmse",
            ),
            matched_control_nll_max_abs_delta_bound=_finite(
                value["matched_control_nll_max_abs_delta_bound"],
                label="matched_control_nll_max_abs_delta_bound",
            ),
            matched_control_centered_logit_nrmse_bound=_finite(
                value["matched_control_centered_logit_nrmse_bound"],
                label="matched_control_centered_logit_nrmse_bound",
            ),
            matched_control_final_carrier_nrmse_bound=_finite(
                value["matched_control_final_carrier_nrmse_bound"],
                label="matched_control_final_carrier_nrmse_bound",
            ),
            attention_operator_catalog_sha256=str(
                value["attention_operator_catalog_sha256"]
            ),
            attention_delta_catalog_sha256=str(
                value["attention_delta_catalog_sha256"]
            ),
            mutation_delta_catalog_sha256=str(
                value["mutation_delta_catalog_sha256"]
            ),
            mutation_carrier_catalog_sha256=str(
                value["mutation_carrier_catalog_sha256"]
            ),
            attention_operator_sha256s=parsed_hash_ledgers[
                "attention_operator_sha256s"
            ],
            attention_delta_sha256s=parsed_hash_ledgers[
                "attention_delta_sha256s"
            ],
            mutation_delta_sha256s=parsed_hash_ledgers[
                "mutation_delta_sha256s"
            ],
            mutation_carrier_sha256s=parsed_hash_ledgers[
                "mutation_carrier_sha256s"
            ],
            final_carrier_sha256=str(value["final_carrier_sha256"]),
            terminal_normalized_output_sha256=str(
                value["terminal_normalized_output_sha256"]
            ),
            dense_replay_literal_parity_passed=value[
                "dense_replay_literal_parity_passed"
            ],
            dense_replay_nll_literal_parity_passed=value[
                "dense_replay_nll_literal_parity_passed"
            ],
            dense_replay_top1_parity_passed=value[
                "dense_replay_top1_parity_passed"
            ],
            all_routes_operator_delta_controls_passed=value[
                "all_routes_operator_delta_controls_passed"
            ],
            selection_prefix_nesting_passed=value[
                "selection_prefix_nesting_passed"
            ],
            measurements=tuple(parsed_measurements),
            measurement_catalog_sha256=str(value["measurement_catalog_sha256"]),
            row_sha256=str(value["row_sha256"]),
        )


def _nrmse(error_sum_squares: float, reference_sum_squares: float) -> float:
    if reference_sum_squares == 0.0:
        return 0.0 if error_sum_squares == 0.0 else math.sqrt(error_sum_squares)
    return math.sqrt(error_sum_squares / reference_sum_squares)


def _cosine(dot: float, reference_sq: float, sparse_sq: float) -> float:
    if reference_sq == 0.0 and sparse_sq == 0.0:
        return 1.0
    if reference_sq == 0.0 or sparse_sq == 0.0:
        return 0.0
    result = dot / math.sqrt(reference_sq * sparse_sq)
    # Float64 accumulation can move an exact endpoint by a few ulps.
    return min(1.0, max(-1.0, result))


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityCell:
    condition: AttentionEdgeFidelityCondition
    selection_catalog_ledger_sha256: str
    row_count: int
    available_route_count: int
    selected_route_count: int
    active_head_query_count: int
    total_head_query_count: int
    selected_probability_mass: float
    native_probability_mass: float
    selected_fisher_energy: float
    full_fisher_energy: float
    operator_reference_sum_squares: float
    operator_sparse_sum_squares: float
    operator_error_sum_squares: float
    operator_dot_product: float
    operator_max_abs_error: float
    final_carrier_reference_sum_squares: float
    final_carrier_sparse_sum_squares: float
    final_carrier_error_sum_squares: float
    final_carrier_dot_product: float
    final_carrier_max_abs_error: float
    terminal_reference_sum_squares: float
    terminal_sparse_sum_squares: float
    terminal_error_sum_squares: float
    terminal_dot_product: float
    terminal_max_abs_error: float
    baseline_valid_logit_sum_squares: float
    sparse_valid_logit_sum_squares: float
    valid_logit_error_sum_squares: float
    valid_logit_dot_product: float
    metrics: AttentionEdgeFidelityMetricLedger
    baseline_nll_sum: float
    sparse_nll_sum: float
    top1_retained_count: int
    baseline_target_correct_count: int
    sparse_target_correct_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.condition, AttentionEdgeFidelityCondition):
            raise TypeError("attention fidelity cell condition is invalid")
        _require_sha256(
            self.selection_catalog_ledger_sha256,
            label="selection_catalog_ledger_sha256",
        )
        rows = _integer(self.row_count, label="row_count", minimum=1)
        available = _integer(
            self.available_route_count, label="available_route_count"
        )
        selected = _integer(
            self.selected_route_count, label="selected_route_count"
        )
        active = _integer(
            self.active_head_query_count, label="active_head_query_count"
        )
        total = _integer(
            self.total_head_query_count, label="total_head_query_count"
        )
        top1 = _integer(
            self.top1_retained_count, label="top1_retained_count"
        )
        baseline_correct = _integer(
            self.baseline_target_correct_count,
            label="baseline_target_correct_count",
        )
        sparse_correct = _integer(
            self.sparse_target_correct_count,
            label="sparse_target_correct_count",
        )
        if (
            selected > available
            or active > total
            or top1 > rows
            or baseline_correct > rows
            or sparse_correct > rows
        ):
            raise ValueError("attention fidelity cell counts do not close")
        if available == 0 or total == 0:
            raise ValueError("attention fidelity cell support must be nonempty")
        for label, value in (
            ("selected_probability_mass", self.selected_probability_mass),
            ("native_probability_mass", self.native_probability_mass),
            ("selected_fisher_energy", self.selected_fisher_energy),
            ("full_fisher_energy", self.full_fisher_energy),
            (
                "operator_reference_sum_squares",
                self.operator_reference_sum_squares,
            ),
            ("operator_sparse_sum_squares", self.operator_sparse_sum_squares),
            ("operator_error_sum_squares", self.operator_error_sum_squares),
            ("operator_max_abs_error", self.operator_max_abs_error),
            (
                "final_carrier_reference_sum_squares",
                self.final_carrier_reference_sum_squares,
            ),
            (
                "final_carrier_sparse_sum_squares",
                self.final_carrier_sparse_sum_squares,
            ),
            (
                "final_carrier_error_sum_squares",
                self.final_carrier_error_sum_squares,
            ),
            ("final_carrier_max_abs_error", self.final_carrier_max_abs_error),
            (
                "terminal_reference_sum_squares",
                self.terminal_reference_sum_squares,
            ),
            ("terminal_sparse_sum_squares", self.terminal_sparse_sum_squares),
            ("terminal_error_sum_squares", self.terminal_error_sum_squares),
            ("terminal_max_abs_error", self.terminal_max_abs_error),
            (
                "baseline_valid_logit_sum_squares",
                self.baseline_valid_logit_sum_squares,
            ),
            (
                "sparse_valid_logit_sum_squares",
                self.sparse_valid_logit_sum_squares,
            ),
            (
                "valid_logit_error_sum_squares",
                self.valid_logit_error_sum_squares,
            ),
            ("baseline_nll_sum", self.baseline_nll_sum),
            ("sparse_nll_sum", self.sparse_nll_sum),
        ):
            _nonnegative(value, label=label)
        _finite(self.operator_dot_product, label="operator_dot_product")
        _finite(self.final_carrier_dot_product, label="final_carrier_dot_product")
        _finite(self.terminal_dot_product, label="terminal_dot_product")
        _finite(self.valid_logit_dot_product, label="valid_logit_dot_product")
        if not isinstance(self.metrics, AttentionEdgeFidelityMetricLedger):
            raise TypeError("attention fidelity cell metrics are invalid")
        _require_stats_match_legacy(
            self.metrics.local_operator,
            reference_sum_squares=self.operator_reference_sum_squares,
            sparse_sum_squares=self.operator_sparse_sum_squares,
            error_sum_squares=self.operator_error_sum_squares,
            dot_product=self.operator_dot_product,
            maximum_abs_error=self.operator_max_abs_error,
            label="cell local operator",
        )
        _require_stats_match_legacy(
            self.metrics.final_valid_carrier,
            reference_sum_squares=self.final_carrier_reference_sum_squares,
            sparse_sum_squares=self.final_carrier_sparse_sum_squares,
            error_sum_squares=self.final_carrier_error_sum_squares,
            dot_product=self.final_carrier_dot_product,
            maximum_abs_error=self.final_carrier_max_abs_error,
            label="cell final valid carrier",
        )
        _require_stats_match_legacy(
            self.metrics.terminal_valid,
            reference_sum_squares=self.terminal_reference_sum_squares,
            sparse_sum_squares=self.terminal_sparse_sum_squares,
            error_sum_squares=self.terminal_error_sum_squares,
            dot_product=self.terminal_dot_product,
            maximum_abs_error=self.terminal_max_abs_error,
            label="cell terminal valid",
        )
        _require_stats_match_legacy(
            self.metrics.valid_logits,
            reference_sum_squares=self.baseline_valid_logit_sum_squares,
            sparse_sum_squares=self.sparse_valid_logit_sum_squares,
            error_sum_squares=self.valid_logit_error_sum_squares,
            dot_product=self.valid_logit_dot_product,
            maximum_abs_error=None,
            label="cell valid logits",
        )
        if self.selected_probability_mass > self.native_probability_mass + 1e-8:
            raise ValueError("cell selected probability exceeds native mass")
        if self.native_probability_mass <= 0.0:
            raise ValueError("cell native probability mass must be positive")
        if self.selected_fisher_energy > self.full_fisher_energy + 1e-8:
            raise ValueError("cell selected Fisher energy exceeds full energy")

    @property
    def selected_route_fraction(self) -> float:
        return self.selected_route_count / self.available_route_count

    @property
    def active_head_query_fraction(self) -> float:
        return self.active_head_query_count / self.total_head_query_count

    @property
    def starved_head_query_count(self) -> int:
        return self.total_head_query_count - self.active_head_query_count

    @property
    def retained_probability_mass_fraction(self) -> float:
        return self.selected_probability_mass / self.native_probability_mass

    @property
    def retained_fisher_energy_fraction(self) -> float:
        if self.full_fisher_energy == 0.0:
            return 1.0 if self.selected_fisher_energy == 0.0 else 0.0
        return self.selected_fisher_energy / self.full_fisher_energy

    @property
    def operator_nrmse(self) -> float:
        return _nrmse(
            self.operator_error_sum_squares,
            self.operator_reference_sum_squares,
        )

    @property
    def operator_cosine(self) -> float:
        return _cosine(
            self.operator_dot_product,
            self.operator_reference_sum_squares,
            self.operator_sparse_sum_squares,
        )

    @property
    def final_carrier_nrmse(self) -> float:
        return _nrmse(
            self.final_carrier_error_sum_squares,
            self.final_carrier_reference_sum_squares,
        )

    @property
    def final_carrier_cosine(self) -> float:
        return _cosine(
            self.final_carrier_dot_product,
            self.final_carrier_reference_sum_squares,
            self.final_carrier_sparse_sum_squares,
        )

    @property
    def terminal_nrmse(self) -> float:
        return _nrmse(
            self.terminal_error_sum_squares,
            self.terminal_reference_sum_squares,
        )

    @property
    def terminal_cosine(self) -> float:
        return _cosine(
            self.terminal_dot_product,
            self.terminal_reference_sum_squares,
            self.terminal_sparse_sum_squares,
        )

    @property
    def valid_logit_nrmse(self) -> float:
        return _nrmse(
            self.valid_logit_error_sum_squares,
            self.baseline_valid_logit_sum_squares,
        )

    @property
    def valid_logit_cosine(self) -> float:
        return _cosine(
            self.valid_logit_dot_product,
            self.baseline_valid_logit_sum_squares,
            self.sparse_valid_logit_sum_squares,
        )

    @property
    def mean_nll_delta(self) -> float:
        return (self.sparse_nll_sum - self.baseline_nll_sum) / self.row_count

    @property
    def top1_retention_rate(self) -> float:
        return self.top1_retained_count / self.row_count

    @property
    def baseline_target_accuracy(self) -> float:
        return self.baseline_target_correct_count / self.row_count

    @property
    def sparse_target_accuracy(self) -> float:
        return self.sparse_target_correct_count / self.row_count

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition.to_dict(),
            "selection_catalog_ledger_sha256": (
                self.selection_catalog_ledger_sha256
            ),
            "row_count": self.row_count,
            "available_route_count": self.available_route_count,
            "selected_route_count": self.selected_route_count,
            "selected_route_fraction": self.selected_route_fraction,
            "active_head_query_count": self.active_head_query_count,
            "starved_head_query_count": self.starved_head_query_count,
            "total_head_query_count": self.total_head_query_count,
            "active_head_query_fraction": self.active_head_query_fraction,
            "selected_probability_mass": self.selected_probability_mass,
            "native_probability_mass": self.native_probability_mass,
            "retained_probability_mass_fraction": (
                self.retained_probability_mass_fraction
            ),
            "selected_fisher_energy": self.selected_fisher_energy,
            "full_fisher_energy": self.full_fisher_energy,
            "retained_fisher_energy_fraction": (
                self.retained_fisher_energy_fraction
            ),
            "operator_reference_sum_squares": (
                self.operator_reference_sum_squares
            ),
            "operator_sparse_sum_squares": self.operator_sparse_sum_squares,
            "operator_error_sum_squares": self.operator_error_sum_squares,
            "operator_dot_product": self.operator_dot_product,
            "operator_max_abs_error": self.operator_max_abs_error,
            "operator_nrmse": self.operator_nrmse,
            "operator_cosine": self.operator_cosine,
            "final_carrier_reference_sum_squares": (
                self.final_carrier_reference_sum_squares
            ),
            "final_carrier_sparse_sum_squares": (
                self.final_carrier_sparse_sum_squares
            ),
            "final_carrier_error_sum_squares": (
                self.final_carrier_error_sum_squares
            ),
            "final_carrier_dot_product": self.final_carrier_dot_product,
            "final_carrier_max_abs_error": self.final_carrier_max_abs_error,
            "final_carrier_nrmse": self.final_carrier_nrmse,
            "final_carrier_cosine": self.final_carrier_cosine,
            "terminal_reference_sum_squares": (
                self.terminal_reference_sum_squares
            ),
            "terminal_sparse_sum_squares": self.terminal_sparse_sum_squares,
            "terminal_error_sum_squares": self.terminal_error_sum_squares,
            "terminal_dot_product": self.terminal_dot_product,
            "terminal_max_abs_error": self.terminal_max_abs_error,
            "terminal_nrmse": self.terminal_nrmse,
            "terminal_cosine": self.terminal_cosine,
            "baseline_valid_logit_sum_squares": (
                self.baseline_valid_logit_sum_squares
            ),
            "sparse_valid_logit_sum_squares": (
                self.sparse_valid_logit_sum_squares
            ),
            "valid_logit_error_sum_squares": (
                self.valid_logit_error_sum_squares
            ),
            "valid_logit_dot_product": self.valid_logit_dot_product,
            "valid_logit_nrmse": self.valid_logit_nrmse,
            "valid_logit_cosine": self.valid_logit_cosine,
            "metrics": self.metrics.to_dict(),
            "baseline_nll_sum": self.baseline_nll_sum,
            "sparse_nll_sum": self.sparse_nll_sum,
            "mean_nll_delta": self.mean_nll_delta,
            "top1_retained_count": self.top1_retained_count,
            "top1_retention_rate": self.top1_retention_rate,
            "baseline_target_correct_count": (
                self.baseline_target_correct_count
            ),
            "sparse_target_correct_count": self.sparse_target_correct_count,
            "baseline_target_accuracy": self.baseline_target_accuracy,
            "sparse_target_accuracy": self.sparse_target_accuracy,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityCell":
        expected = {
            "condition",
            "selection_catalog_ledger_sha256",
            "row_count",
            "available_route_count",
            "selected_route_count",
            "selected_route_fraction",
            "active_head_query_count",
            "starved_head_query_count",
            "total_head_query_count",
            "active_head_query_fraction",
            "selected_probability_mass",
            "native_probability_mass",
            "retained_probability_mass_fraction",
            "selected_fisher_energy",
            "full_fisher_energy",
            "retained_fisher_energy_fraction",
            "operator_reference_sum_squares",
            "operator_sparse_sum_squares",
            "operator_error_sum_squares",
            "operator_dot_product",
            "operator_max_abs_error",
            "operator_nrmse",
            "operator_cosine",
            "final_carrier_reference_sum_squares",
            "final_carrier_sparse_sum_squares",
            "final_carrier_error_sum_squares",
            "final_carrier_dot_product",
            "final_carrier_max_abs_error",
            "final_carrier_nrmse",
            "final_carrier_cosine",
            "terminal_reference_sum_squares",
            "terminal_sparse_sum_squares",
            "terminal_error_sum_squares",
            "terminal_dot_product",
            "terminal_max_abs_error",
            "terminal_nrmse",
            "terminal_cosine",
            "baseline_valid_logit_sum_squares",
            "sparse_valid_logit_sum_squares",
            "valid_logit_error_sum_squares",
            "valid_logit_dot_product",
            "valid_logit_nrmse",
            "valid_logit_cosine",
            "metrics",
            "baseline_nll_sum",
            "sparse_nll_sum",
            "mean_nll_delta",
            "top1_retained_count",
            "top1_retention_rate",
            "baseline_target_correct_count",
            "sparse_target_correct_count",
            "baseline_target_accuracy",
            "sparse_target_accuracy",
        }
        _exact_keys(value, expected, label="attention fidelity cell")
        condition_value = value["condition"]
        if not isinstance(condition_value, Mapping):
            raise ValueError("cell condition must be an object")
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError("cell metrics must be an object")
        result = cls(
            condition=AttentionEdgeFidelityCondition.from_dict(condition_value),
            selection_catalog_ledger_sha256=str(
                value["selection_catalog_ledger_sha256"]
            ),
            row_count=_integer(value["row_count"], label="row_count", minimum=1),
            available_route_count=_integer(
                value["available_route_count"], label="available_route_count"
            ),
            selected_route_count=_integer(
                value["selected_route_count"], label="selected_route_count"
            ),
            active_head_query_count=_integer(
                value["active_head_query_count"], label="active_head_query_count"
            ),
            total_head_query_count=_integer(
                value["total_head_query_count"], label="total_head_query_count"
            ),
            selected_probability_mass=_finite(
                value["selected_probability_mass"],
                label="selected_probability_mass",
            ),
            native_probability_mass=_finite(
                value["native_probability_mass"],
                label="native_probability_mass",
            ),
            selected_fisher_energy=_finite(
                value["selected_fisher_energy"],
                label="selected_fisher_energy",
            ),
            full_fisher_energy=_finite(
                value["full_fisher_energy"], label="full_fisher_energy"
            ),
            operator_reference_sum_squares=_finite(
                value["operator_reference_sum_squares"],
                label="operator_reference_sum_squares",
            ),
            operator_sparse_sum_squares=_finite(
                value["operator_sparse_sum_squares"],
                label="operator_sparse_sum_squares",
            ),
            operator_error_sum_squares=_finite(
                value["operator_error_sum_squares"],
                label="operator_error_sum_squares",
            ),
            operator_dot_product=_finite(
                value["operator_dot_product"], label="operator_dot_product"
            ),
            operator_max_abs_error=_finite(
                value["operator_max_abs_error"],
                label="operator_max_abs_error",
            ),
            final_carrier_reference_sum_squares=_finite(
                value["final_carrier_reference_sum_squares"],
                label="final_carrier_reference_sum_squares",
            ),
            final_carrier_sparse_sum_squares=_finite(
                value["final_carrier_sparse_sum_squares"],
                label="final_carrier_sparse_sum_squares",
            ),
            final_carrier_error_sum_squares=_finite(
                value["final_carrier_error_sum_squares"],
                label="final_carrier_error_sum_squares",
            ),
            final_carrier_dot_product=_finite(
                value["final_carrier_dot_product"],
                label="final_carrier_dot_product",
            ),
            final_carrier_max_abs_error=_finite(
                value["final_carrier_max_abs_error"],
                label="final_carrier_max_abs_error",
            ),
            terminal_reference_sum_squares=_finite(
                value["terminal_reference_sum_squares"],
                label="terminal_reference_sum_squares",
            ),
            terminal_sparse_sum_squares=_finite(
                value["terminal_sparse_sum_squares"],
                label="terminal_sparse_sum_squares",
            ),
            terminal_error_sum_squares=_finite(
                value["terminal_error_sum_squares"],
                label="terminal_error_sum_squares",
            ),
            terminal_dot_product=_finite(
                value["terminal_dot_product"],
                label="terminal_dot_product",
            ),
            terminal_max_abs_error=_finite(
                value["terminal_max_abs_error"],
                label="terminal_max_abs_error",
            ),
            baseline_valid_logit_sum_squares=_finite(
                value["baseline_valid_logit_sum_squares"],
                label="baseline_valid_logit_sum_squares",
            ),
            sparse_valid_logit_sum_squares=_finite(
                value["sparse_valid_logit_sum_squares"],
                label="sparse_valid_logit_sum_squares",
            ),
            valid_logit_error_sum_squares=_finite(
                value["valid_logit_error_sum_squares"],
                label="valid_logit_error_sum_squares",
            ),
            valid_logit_dot_product=_finite(
                value["valid_logit_dot_product"],
                label="valid_logit_dot_product",
            ),
            metrics=AttentionEdgeFidelityMetricLedger.from_dict(metrics),
            baseline_nll_sum=_nonnegative(
                value["baseline_nll_sum"], label="baseline_nll_sum"
            ),
            sparse_nll_sum=_nonnegative(
                value["sparse_nll_sum"], label="sparse_nll_sum"
            ),
            top1_retained_count=_integer(
                value["top1_retained_count"], label="top1_retained_count"
            ),
            baseline_target_correct_count=_integer(
                value["baseline_target_correct_count"],
                label="baseline_target_correct_count",
            ),
            sparse_target_correct_count=_integer(
                value["sparse_target_correct_count"],
                label="sparse_target_correct_count",
            ),
        )
        derived = {
            "selected_route_fraction": result.selected_route_fraction,
            "starved_head_query_count": result.starved_head_query_count,
            "active_head_query_fraction": result.active_head_query_fraction,
            "retained_probability_mass_fraction": (
                result.retained_probability_mass_fraction
            ),
            "retained_fisher_energy_fraction": (
                result.retained_fisher_energy_fraction
            ),
            "operator_nrmse": result.operator_nrmse,
            "operator_cosine": result.operator_cosine,
            "final_carrier_nrmse": result.final_carrier_nrmse,
            "final_carrier_cosine": result.final_carrier_cosine,
            "terminal_nrmse": result.terminal_nrmse,
            "terminal_cosine": result.terminal_cosine,
            "valid_logit_nrmse": result.valid_logit_nrmse,
            "valid_logit_cosine": result.valid_logit_cosine,
            "mean_nll_delta": result.mean_nll_delta,
            "top1_retention_rate": result.top1_retention_rate,
            "baseline_target_accuracy": result.baseline_target_accuracy,
            "sparse_target_accuracy": result.sparse_target_accuracy,
        }
        for label, expected_value in derived.items():
            supplied = _finite(value[label], label=label)
            if supplied != expected_value:
                raise ValueError(f"attention fidelity derived {label} drifted")
        return result


def _aggregate_fidelity_cell(
    condition: AttentionEdgeFidelityCondition,
    values: Sequence[AttentionEdgeFidelityMeasurement],
) -> AttentionEdgeFidelityCell:
    materialized = tuple(values)
    if not materialized or any(
        item.condition != condition for item in materialized
    ):
        raise ValueError("cannot aggregate a mismatched fidelity condition")
    return AttentionEdgeFidelityCell(
        condition=condition,
        selection_catalog_ledger_sha256=_json_sha256(
            [item.selection_catalog_sha256 for item in materialized],
            domain=_SELECTION_DOMAIN,
        ),
        row_count=len(materialized),
        available_route_count=sum(
            item.available_route_count for item in materialized
        ),
        selected_route_count=sum(
            item.selected_route_count for item in materialized
        ),
        active_head_query_count=sum(
            item.active_head_query_count for item in materialized
        ),
        total_head_query_count=sum(
            item.total_head_query_count for item in materialized
        ),
        selected_probability_mass=math.fsum(
            item.selected_probability_mass for item in materialized
        ),
        native_probability_mass=math.fsum(
            item.native_probability_mass for item in materialized
        ),
        selected_fisher_energy=math.fsum(
            item.selected_fisher_energy for item in materialized
        ),
        full_fisher_energy=math.fsum(
            item.full_fisher_energy for item in materialized
        ),
        operator_reference_sum_squares=math.fsum(
            item.operator_reference_sum_squares for item in materialized
        ),
        operator_sparse_sum_squares=math.fsum(
            item.operator_sparse_sum_squares for item in materialized
        ),
        operator_error_sum_squares=math.fsum(
            item.operator_error_sum_squares for item in materialized
        ),
        operator_dot_product=math.fsum(
            item.operator_dot_product for item in materialized
        ),
        operator_max_abs_error=max(
            item.operator_max_abs_error for item in materialized
        ),
        final_carrier_reference_sum_squares=math.fsum(
            item.final_carrier_reference_sum_squares for item in materialized
        ),
        final_carrier_sparse_sum_squares=math.fsum(
            item.final_carrier_sparse_sum_squares for item in materialized
        ),
        final_carrier_error_sum_squares=math.fsum(
            item.final_carrier_error_sum_squares for item in materialized
        ),
        final_carrier_dot_product=math.fsum(
            item.final_carrier_dot_product for item in materialized
        ),
        final_carrier_max_abs_error=max(
            item.final_carrier_max_abs_error for item in materialized
        ),
        terminal_reference_sum_squares=math.fsum(
            item.terminal_reference_sum_squares for item in materialized
        ),
        terminal_sparse_sum_squares=math.fsum(
            item.terminal_sparse_sum_squares for item in materialized
        ),
        terminal_error_sum_squares=math.fsum(
            item.terminal_error_sum_squares for item in materialized
        ),
        terminal_dot_product=math.fsum(
            item.terminal_dot_product for item in materialized
        ),
        terminal_max_abs_error=max(
            item.terminal_max_abs_error for item in materialized
        ),
        baseline_valid_logit_sum_squares=math.fsum(
            item.baseline_valid_logit_sum_squares for item in materialized
        ),
        sparse_valid_logit_sum_squares=math.fsum(
            item.sparse_valid_logit_sum_squares for item in materialized
        ),
        valid_logit_error_sum_squares=math.fsum(
            item.valid_logit_error_sum_squares for item in materialized
        ),
        valid_logit_dot_product=math.fsum(
            item.valid_logit_dot_product for item in materialized
        ),
        metrics=_aggregate_metric_ledgers(
            [item.metrics for item in materialized]
        ),
        baseline_nll_sum=math.fsum(
            item.baseline_nll for item in materialized
        ),
        sparse_nll_sum=math.fsum(
            item.sparse_nll for item in materialized
        ),
        top1_retained_count=sum(
            int(item.top1_retained) for item in materialized
        ),
        baseline_target_correct_count=sum(
            int(item.baseline_target_correct) for item in materialized
        ),
        sparse_target_correct_count=sum(
            int(item.sparse_target_correct) for item in materialized
        ),
    )


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityFamilyCell:
    family_id: str
    cell: AttentionEdgeFidelityCell

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("fidelity family cell ID must be nonempty")
        if not isinstance(self.cell, AttentionEdgeFidelityCell):
            raise TypeError("fidelity family cell is not typed")

    def to_dict(self) -> dict[str, object]:
        return {"family_id": self.family_id, "cell": self.cell.to_dict()}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityFamilyCell":
        _exact_keys(value, {"family_id", "cell"}, label="fidelity family cell")
        cell = value["cell"]
        if not isinstance(cell, Mapping):
            raise ValueError("fidelity family cell payload must be an object")
        return cls(
            family_id=str(value["family_id"]),
            cell=AttentionEdgeFidelityCell.from_dict(cell),
        )


_SELECTION_LEDGER_FIELDS: Final = (
    "selection_catalog_sha256",
    "available_route_count",
    "selected_route_count",
    "active_head_query_count",
    "total_head_query_count",
    "selected_probability_mass",
    "native_probability_mass",
    "selected_fisher_energy",
    "full_fisher_energy",
)


def _validate_measurement_selection_ledger(
    values: Sequence[AttentionEdgeFidelityMeasurement],
) -> None:
    materialized = tuple(values)
    groups: dict[
        tuple[str, int, str], list[AttentionEdgeFidelityMeasurement]
    ] = {}
    for item in materialized:
        groups.setdefault(
            (
                item.condition.ranker,
                item.condition.k,
                item.condition.random_salt,
            ),
            [],
        ).append(item)
    for selector, group in groups.items():
        if len(group) == 1:
            if selector[2] == "primary":
                raise ValueError("primary selector omitted a semantic variant")
            continue
        if len(group) != 2 or {
            item.condition.variant for item in group
        } != set(ATTENTION_EDGE_FIDELITY_VARIANTS):
            raise ValueError("selector semantic multiplicity drifted")
        left, right = group
        if any(
            getattr(left, field) != getattr(right, field)
            for field in _SELECTION_LEDGER_FIELDS
        ):
            raise ValueError(
                "zero-fill and renormalized cells changed the frozen selection"
            )


def _reference_signature(stats: AttentionFidelityVectorStats) -> tuple[int, float]:
    return (stats.element_count, stats.reference_sum_squares)


def _validate_measurement_baseline_ledger(
    values: Sequence[AttentionEdgeFidelityMeasurement],
) -> None:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("row measurement baseline ledger is empty")

    def signature(item: AttentionEdgeFidelityMeasurement) -> object:
        metrics = item.metrics
        return (
            item.available_route_count,
            item.total_head_query_count,
            item.native_probability_mass,
            item.full_fisher_energy,
            item.baseline_target_correct,
            _reference_signature(metrics.local_operator),
            _reference_signature(metrics.local_postnorm_delta),
            tuple(
                surface.element_count
                for surface in metrics.propagated_mutations
            ),
            metrics.final_valid_carrier.element_count,
            metrics.supervised_carrier.element_count,
            metrics.terminal_valid.element_count,
            metrics.supervised_terminal.element_count,
            metrics.valid_logits.element_count,
            metrics.centered_valid_logits.element_count,
        )

    expected = signature(materialized[0])
    if any(signature(item) != expected for item in materialized[1:]):
        raise ValueError("row-invariant native teacher baselines drifted")


def _validate_cell_selection_ledger(
    values: Sequence[AttentionEdgeFidelityCell],
) -> None:
    groups: dict[tuple[str, int, str], list[AttentionEdgeFidelityCell]] = {}
    for item in values:
        groups.setdefault(
            (
                item.condition.ranker,
                item.condition.k,
                item.condition.random_salt,
            ),
            [],
        ).append(item)
    fields = (
        "selection_catalog_ledger_sha256",
        "available_route_count",
        "selected_route_count",
        "active_head_query_count",
        "total_head_query_count",
        "selected_probability_mass",
        "native_probability_mass",
        "selected_fisher_energy",
        "full_fisher_energy",
    )
    for selector, group in groups.items():
        if len(group) == 1:
            if selector[2] == "primary":
                raise ValueError("primary cell selector omitted a semantic variant")
            continue
        if len(group) != 2 or any(
            getattr(group[0], field) != getattr(group[1], field)
            for field in fields
        ):
            raise ValueError(
                "zero-fill and renormalized aggregate selection ledgers drifted"
            )


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityReport:
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
    atlas_report_sha256: str
    atlas_file_sha256: str
    atlas_implementation_bundle_sha256: str
    exact_control_receipt_sha256: str
    attention_generator_implementation_sha256: str
    joint_executor_implementation_sha256: str
    joint_compiler_implementation_sha256: str
    attention_edge_core_implementation_sha256: str
    attention_edge_atlas_implementation_sha256: str
    attention_edge_fidelity_kernel_implementation_sha256: str
    attention_edge_fidelity_implementation_sha256: str
    implementation_bundle_sha256: str
    source_commit_sha: str
    pre_execution_source_state_sha256: str
    post_execution_source_state_sha256: str
    row_count: int
    family_counts: tuple[tuple[str, int], ...]
    layer_count: int
    query_head_count: int
    atlas_extraction_identity_count: int
    atlas_fisher_k8_coordinate_identity_count: int
    row_receipts: tuple[AttentionEdgeFidelityRowReceipt, ...]
    cells: tuple[AttentionEdgeFidelityCell, ...]
    family_cells: tuple[AttentionEdgeFidelityFamilyCell, ...]
    atlas_fisher_top8_coordinate_match_passed: bool = True
    full_atlas_1080_identity_passed: bool = False
    all_dense_replay_controls_passed: bool = True
    global_k_across_heads_and_keys: bool = True
    fisher_ranker_is_same_row_target_gradient_oracle: bool = True
    selection_masks_frozen_before_intervention: bool = True
    qkv_and_probabilities_recomputed_on_sparse_state: bool = True
    all_attention_layers_intervened_simultaneously: bool = True
    empty_renormalized_heads_zero_filled: bool = True
    selection_prefix_nesting_passed: bool = True
    clean_worktree_required_for_live_execution: bool = True
    clean_worktree_observed: bool = True
    committed_source_identity_passed: bool = True
    post_execution_source_recheck_passed: bool = True
    self_proves_live_fidelity: bool = False
    selection_optimality_claimed: bool = False
    retrospective_A_only: bool = True
    role: str = ATTENTION_EDGE_FIDELITY_ROLE
    fresh_panel_used: bool = False
    attention_edge_confirmation_claimed: bool = False
    semantic_interpretation_claimed: bool = False
    compression_qualification_claimed: bool = False
    parameter_savings_claimed: bool = False
    mac_savings_claimed: bool = False
    latency_claimed: bool = False
    prompts_persisted: bool = False
    token_ids_persisted: bool = False
    raw_activations_persisted: bool = False
    raw_gradients_persisted: bool = False
    schema: str = ATTENTION_EDGE_FIDELITY_SCHEMA
    format_version: int = ATTENTION_EDGE_FIDELITY_FORMAT_VERSION
    abi: str = ATTENTION_EDGE_FIDELITY_ABI
    report_sha256: str = ""

    def __post_init__(self) -> None:
        if self.source_kind not in {
            "model_capture",
            "model_smoke",
            "model_free_fixture",
        }:
            raise ValueError("attention fidelity source kind is invalid")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.model_id,
                self.model_revision,
                self.panel_id,
                self.score_objective_id,
            )
        ):
            raise ValueError("attention fidelity source metadata is incomplete")
        for label, digest in (
            ("panel file", self.panel_file_sha256),
            ("row catalog", self.row_catalog_sha256),
            ("tokenization catalog", self.tokenization_catalog_sha256),
            ("source model", self.source_model_fingerprint_sha256),
            ("source execution", self.source_execution_fingerprint_sha256),
            ("atlas report", self.atlas_report_sha256),
            ("atlas file", self.atlas_file_sha256),
            ("atlas implementation", self.atlas_implementation_bundle_sha256),
            ("exact control receipt", self.exact_control_receipt_sha256),
            ("attention generator implementation", self.attention_generator_implementation_sha256),
            ("joint executor implementation", self.joint_executor_implementation_sha256),
            ("joint compiler implementation", self.joint_compiler_implementation_sha256),
            ("edge core implementation", self.attention_edge_core_implementation_sha256),
            ("atlas runner implementation", self.attention_edge_atlas_implementation_sha256),
            ("fidelity kernel implementation", self.attention_edge_fidelity_kernel_implementation_sha256),
            ("fidelity implementation", self.attention_edge_fidelity_implementation_sha256),
            ("implementation bundle", self.implementation_bundle_sha256),
            ("pre-execution source state", self.pre_execution_source_state_sha256),
            ("post-execution source state", self.post_execution_source_state_sha256),
        ):
            _require_sha256(digest, label=label)
        expected_bundle = _json_sha256(
            {
                "attention_generator_implementation_sha256": (
                    self.attention_generator_implementation_sha256
                ),
                "joint_executor_implementation_sha256": (
                    self.joint_executor_implementation_sha256
                ),
                "joint_compiler_implementation_sha256": (
                    self.joint_compiler_implementation_sha256
                ),
                "attention_edge_core_implementation_sha256": (
                    self.attention_edge_core_implementation_sha256
                ),
                "attention_edge_atlas_implementation_sha256": (
                    self.attention_edge_atlas_implementation_sha256
                ),
                "attention_edge_fidelity_kernel_implementation_sha256": (
                    self.attention_edge_fidelity_kernel_implementation_sha256
                ),
                "attention_edge_fidelity_implementation_sha256": (
                    self.attention_edge_fidelity_implementation_sha256
                ),
            },
            domain=_IMPLEMENTATION_DOMAIN,
        )
        if self.implementation_bundle_sha256 != expected_bundle:
            raise ValueError("attention fidelity implementation bundle drifted")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit_sha) is None:
            raise ValueError("attention fidelity source commit is invalid")
        declared_implementation_hashes = (
            self.attention_generator_implementation_sha256,
            self.joint_executor_implementation_sha256,
            self.joint_compiler_implementation_sha256,
            self.attention_edge_core_implementation_sha256,
            self.attention_edge_atlas_implementation_sha256,
            self.attention_edge_fidelity_kernel_implementation_sha256,
            self.attention_edge_fidelity_implementation_sha256,
        )
        expected_source_state_sha256 = _json_sha256(
            {
                "commit_sha": self.source_commit_sha,
                "implementation_sha256s": [
                    [path, digest]
                    for path, digest in zip(
                        _COMMITTED_SOURCE_PATHS,
                        declared_implementation_hashes,
                        strict=True,
                    )
                ],
            },
            domain=_SOURCE_STATE_DOMAIN,
        )
        if self.source_kind != "model_free_fixture" and (
            self.atlas_report_sha256
            != PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256
            or self.atlas_file_sha256
            != PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256
            or self.pre_execution_source_state_sha256
            != self.post_execution_source_state_sha256
            or self.pre_execution_source_state_sha256
            != expected_source_state_sha256
            or self.post_execution_source_recheck_passed is not True
        ):
            raise ValueError("live fidelity source lifecycle identity drifted")
        rows = _integer(self.row_count, label="row_count", minimum=1)
        if (
            type(self.family_counts) is not tuple
            or sum(count for _, count in self.family_counts) != rows
            or tuple(sorted(self.family_counts)) != self.family_counts
            or any(
                not isinstance(family, str)
                or not family
                or type(count) is not int
                or count <= 0
                for family, count in self.family_counts
            )
        ):
            raise ValueError("attention fidelity family counts are invalid")
        if self.layer_count != GEMMA3_LAYER_COUNT:
            raise ValueError("attention fidelity layer count drifted")
        if self.query_head_count != ATTENTION_HEAD_COUNT:
            raise ValueError("attention fidelity head count drifted")
        expected_identity_count = rows * GEMMA3_LAYER_COUNT
        if (
            self.atlas_extraction_identity_count != expected_identity_count
            or self.atlas_fisher_k8_coordinate_identity_count
            != expected_identity_count
        ):
            raise ValueError("attention fidelity atlas identity count drifted")
        if self.source_kind == "model_capture":
            if (
                rows != 60
                or expected_identity_count != 1080
                or self.full_atlas_1080_identity_passed is not True
            ):
                raise ValueError("full A fidelity must bind all 1080 atlas layers")
        elif self.full_atlas_1080_identity_passed is not False:
            raise ValueError("partial/fixture fidelity cannot claim full atlas identity")
        if (
            type(self.row_receipts) is not tuple
            or len(self.row_receipts) != rows
            or tuple(item.row_ordinal for item in self.row_receipts)
            != tuple(range(rows))
        ):
            raise ValueError("attention fidelity row receipts are incomplete")
        observed_family_counts = tuple(
            sorted(Counter(item.family_id for item in self.row_receipts).items())
        )
        if observed_family_counts != self.family_counts:
            raise ValueError(
                "attention fidelity family counts do not close over row receipts"
            )
        if (
            type(self.cells) is not tuple
            or tuple(item.condition for item in self.cells) != _conditions()
            or any(item.row_count != rows for item in self.cells)
        ):
            raise ValueError("attention fidelity cell inventory drifted")
        expected_family_inventory = tuple(
            (family, condition)
            for family, _ in self.family_counts
            for condition in _conditions()
        )
        if (
            type(self.family_cells) is not tuple
            or tuple(
                (item.family_id, item.cell.condition)
                for item in self.family_cells
            )
            != expected_family_inventory
        ):
            raise ValueError("attention fidelity family cell inventory drifted")
        all_measurements = tuple(
            measurement
            for receipt in self.row_receipts
            for measurement in receipt.measurements
        )
        for receipt in self.row_receipts:
            _validate_measurement_selection_ledger(receipt.measurements)
            _validate_measurement_baseline_ledger(receipt.measurements)
        expected_cells = tuple(
            _aggregate_fidelity_cell(
                condition,
                [
                    item
                    for item in all_measurements
                    if item.condition == condition
                ],
            )
            for condition in _conditions()
        )
        if self.cells != expected_cells:
            raise ValueError("global fidelity cells do not close over row ledgers")
        _validate_cell_selection_ledger(self.cells)
        expected_family_cells = tuple(
            AttentionEdgeFidelityFamilyCell(
                family_id=family,
                cell=_aggregate_fidelity_cell(
                    condition,
                    [
                        measurement
                        for receipt in self.row_receipts
                        if receipt.family_id == family
                        for measurement in receipt.measurements
                        if measurement.condition == condition
                    ],
                ),
            )
            for family, _ in self.family_counts
            for condition in _conditions()
        )
        if self.family_cells != expected_family_cells:
            raise ValueError("family fidelity cells do not close over row ledgers")
        for family, _ in self.family_counts:
            _validate_cell_selection_ledger(
                [
                    item.cell
                    for item in self.family_cells
                    if item.family_id == family
                ]
            )
        required_true = (
            self.atlas_fisher_top8_coordinate_match_passed,
            self.all_dense_replay_controls_passed,
            self.global_k_across_heads_and_keys,
            self.fisher_ranker_is_same_row_target_gradient_oracle,
            self.selection_masks_frozen_before_intervention,
            self.qkv_and_probabilities_recomputed_on_sparse_state,
            self.all_attention_layers_intervened_simultaneously,
            self.empty_renormalized_heads_zero_filled,
            self.selection_prefix_nesting_passed,
            self.clean_worktree_required_for_live_execution,
            self.retrospective_A_only,
        )
        required_false = (
            self.fresh_panel_used,
            self.attention_edge_confirmation_claimed,
            self.semantic_interpretation_claimed,
            self.compression_qualification_claimed,
            self.parameter_savings_claimed,
            self.mac_savings_claimed,
            self.latency_claimed,
            self.prompts_persisted,
            self.token_ids_persisted,
            self.raw_activations_persisted,
            self.raw_gradients_persisted,
            self.self_proves_live_fidelity,
            self.selection_optimality_claimed,
        )
        if not all(value is True for value in required_true) or not all(
            value is False for value in required_false
        ):
            raise ValueError("attention fidelity claim boundary drifted")
        if self.source_kind == "model_free_fixture":
            if (
                self.clean_worktree_observed is not False
                or self.committed_source_identity_passed is not False
                or self.post_execution_source_recheck_passed is not False
            ):
                raise ValueError("model-free fixture cannot claim live source audit")
        elif (
            self.clean_worktree_observed is not True
            or self.committed_source_identity_passed is not True
            or self.post_execution_source_recheck_passed is not True
        ):
            raise ValueError("live fidelity requires clean committed sources")
        if (
            self.schema != ATTENTION_EDGE_FIDELITY_SCHEMA
            or self.format_version != ATTENTION_EDGE_FIDELITY_FORMAT_VERSION
            or self.abi != ATTENTION_EDGE_FIDELITY_ABI
            or self.role != ATTENTION_EDGE_FIDELITY_ROLE
        ):
            raise ValueError("attention fidelity schema/ABI drifted")
        computed = _json_sha256(self._payload(), domain=_REPORT_DOMAIN)
        if self.report_sha256 == "":
            object.__setattr__(self, "report_sha256", computed)
        elif _require_sha256(self.report_sha256, label="report_sha256") != computed:
            raise ValueError("attention fidelity report hash mismatch")

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
            "atlas_report_sha256": self.atlas_report_sha256,
            "atlas_file_sha256": self.atlas_file_sha256,
            "atlas_implementation_bundle_sha256": (
                self.atlas_implementation_bundle_sha256
            ),
            "exact_control_receipt_sha256": self.exact_control_receipt_sha256,
            "attention_generator_implementation_sha256": (
                self.attention_generator_implementation_sha256
            ),
            "joint_executor_implementation_sha256": (
                self.joint_executor_implementation_sha256
            ),
            "joint_compiler_implementation_sha256": (
                self.joint_compiler_implementation_sha256
            ),
            "attention_edge_core_implementation_sha256": (
                self.attention_edge_core_implementation_sha256
            ),
            "attention_edge_atlas_implementation_sha256": (
                self.attention_edge_atlas_implementation_sha256
            ),
            "attention_edge_fidelity_kernel_implementation_sha256": (
                self.attention_edge_fidelity_kernel_implementation_sha256
            ),
            "attention_edge_fidelity_implementation_sha256": (
                self.attention_edge_fidelity_implementation_sha256
            ),
            "implementation_bundle_sha256": self.implementation_bundle_sha256,
            "source_commit_sha": self.source_commit_sha,
            "pre_execution_source_state_sha256": (
                self.pre_execution_source_state_sha256
            ),
            "post_execution_source_state_sha256": (
                self.post_execution_source_state_sha256
            ),
            "row_count": self.row_count,
            "family_counts": [list(item) for item in self.family_counts],
            "layer_count": self.layer_count,
            "query_head_count": self.query_head_count,
            "atlas_extraction_identity_count": (
                self.atlas_extraction_identity_count
            ),
            "atlas_fisher_k8_coordinate_identity_count": (
                self.atlas_fisher_k8_coordinate_identity_count
            ),
            "row_receipts": [item.to_dict() for item in self.row_receipts],
            "cells": [item.to_dict() for item in self.cells],
            "family_cells": [item.to_dict() for item in self.family_cells],
            "atlas_fisher_top8_coordinate_match_passed": (
                self.atlas_fisher_top8_coordinate_match_passed
            ),
            "full_atlas_1080_identity_passed": (
                self.full_atlas_1080_identity_passed
            ),
            "all_dense_replay_controls_passed": (
                self.all_dense_replay_controls_passed
            ),
            "global_k_across_heads_and_keys": self.global_k_across_heads_and_keys,
            "fisher_ranker_is_same_row_target_gradient_oracle": (
                self.fisher_ranker_is_same_row_target_gradient_oracle
            ),
            "selection_masks_frozen_before_intervention": (
                self.selection_masks_frozen_before_intervention
            ),
            "qkv_and_probabilities_recomputed_on_sparse_state": (
                self.qkv_and_probabilities_recomputed_on_sparse_state
            ),
            "all_attention_layers_intervened_simultaneously": (
                self.all_attention_layers_intervened_simultaneously
            ),
            "empty_renormalized_heads_zero_filled": (
                self.empty_renormalized_heads_zero_filled
            ),
            "selection_prefix_nesting_passed": (
                self.selection_prefix_nesting_passed
            ),
            "clean_worktree_required_for_live_execution": (
                self.clean_worktree_required_for_live_execution
            ),
            "clean_worktree_observed": self.clean_worktree_observed,
            "committed_source_identity_passed": (
                self.committed_source_identity_passed
            ),
            "post_execution_source_recheck_passed": (
                self.post_execution_source_recheck_passed
            ),
            "self_proves_live_fidelity": self.self_proves_live_fidelity,
            "selection_optimality_claimed": self.selection_optimality_claimed,
            "retrospective_A_only": self.retrospective_A_only,
            "role": self.role,
            "fresh_panel_used": self.fresh_panel_used,
            "attention_edge_confirmation_claimed": (
                self.attention_edge_confirmation_claimed
            ),
            "semantic_interpretation_claimed": self.semantic_interpretation_claimed,
            "compression_qualification_claimed": (
                self.compression_qualification_claimed
            ),
            "parameter_savings_claimed": self.parameter_savings_claimed,
            "mac_savings_claimed": self.mac_savings_claimed,
            "latency_claimed": self.latency_claimed,
            "prompts_persisted": self.prompts_persisted,
            "token_ids_persisted": self.token_ids_persisted,
            "raw_activations_persisted": self.raw_activations_persisted,
            "raw_gradients_persisted": self.raw_gradients_persisted,
            "schema": self.schema,
            "format_version": self.format_version,
            "abi": self.abi,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "report_sha256": self.report_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AttentionEdgeFidelityReport":
        expected = {
            "source_kind", "model_id", "model_revision", "panel_id",
            "panel_file_sha256", "row_catalog_sha256",
            "tokenization_catalog_sha256", "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256", "score_objective_id",
            "atlas_report_sha256", "atlas_file_sha256",
            "atlas_implementation_bundle_sha256",
            "exact_control_receipt_sha256",
            "attention_generator_implementation_sha256",
            "joint_executor_implementation_sha256",
            "joint_compiler_implementation_sha256",
            "attention_edge_core_implementation_sha256",
            "attention_edge_atlas_implementation_sha256",
            "attention_edge_fidelity_kernel_implementation_sha256",
            "attention_edge_fidelity_implementation_sha256",
            "implementation_bundle_sha256", "source_commit_sha",
            "pre_execution_source_state_sha256",
            "post_execution_source_state_sha256",
            "row_count", "family_counts",
            "layer_count", "query_head_count",
            "atlas_extraction_identity_count",
            "atlas_fisher_k8_coordinate_identity_count",
            "row_receipts", "cells", "family_cells",
            "atlas_fisher_top8_coordinate_match_passed",
            "full_atlas_1080_identity_passed",
            "all_dense_replay_controls_passed",
            "global_k_across_heads_and_keys",
            "fisher_ranker_is_same_row_target_gradient_oracle",
            "selection_masks_frozen_before_intervention",
            "qkv_and_probabilities_recomputed_on_sparse_state",
            "all_attention_layers_intervened_simultaneously",
            "empty_renormalized_heads_zero_filled",
            "selection_prefix_nesting_passed",
            "clean_worktree_required_for_live_execution",
            "clean_worktree_observed", "committed_source_identity_passed",
            "post_execution_source_recheck_passed",
            "self_proves_live_fidelity", "selection_optimality_claimed",
            "retrospective_A_only", "role",
            "fresh_panel_used", "attention_edge_confirmation_claimed",
            "semantic_interpretation_claimed",
            "compression_qualification_claimed", "parameter_savings_claimed",
            "mac_savings_claimed", "latency_claimed", "prompts_persisted",
            "token_ids_persisted", "raw_activations_persisted",
            "raw_gradients_persisted", "schema", "format_version", "abi",
            "report_sha256",
        }
        _exact_keys(value, expected, label="attention fidelity report")
        families = value["family_counts"]
        rows = value["row_receipts"]
        cells = value["cells"]
        family_cells = value["family_cells"]
        if (
            not isinstance(families, list)
            or not isinstance(rows, list)
            or not isinstance(cells, list)
            or not isinstance(family_cells, list)
        ):
            raise ValueError("attention fidelity report arrays are invalid")
        parsed_families: list[tuple[str, int]] = []
        for item in families:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("family count entry is invalid")
            parsed_families.append(
                (str(item[0]), _integer(item[1], label="family count", minimum=1))
            )
        parsed_rows = []
        for item in rows:
            if not isinstance(item, Mapping):
                raise ValueError("row receipt must be an object")
            parsed_rows.append(AttentionEdgeFidelityRowReceipt.from_dict(item))
        parsed_cells = []
        for item in cells:
            if not isinstance(item, Mapping):
                raise ValueError("fidelity cell must be an object")
            parsed_cells.append(AttentionEdgeFidelityCell.from_dict(item))
        parsed_family_cells = []
        for item in family_cells:
            if not isinstance(item, Mapping):
                raise ValueError("fidelity family cell must be an object")
            parsed_family_cells.append(
                AttentionEdgeFidelityFamilyCell.from_dict(item)
            )
        bool_fields = expected - {
            "source_kind", "model_id", "model_revision", "panel_id",
            "panel_file_sha256", "row_catalog_sha256",
            "tokenization_catalog_sha256", "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256", "score_objective_id",
            "atlas_report_sha256", "atlas_file_sha256",
            "atlas_implementation_bundle_sha256",
            "exact_control_receipt_sha256",
            "attention_generator_implementation_sha256",
            "joint_executor_implementation_sha256",
            "joint_compiler_implementation_sha256",
            "attention_edge_core_implementation_sha256",
            "attention_edge_atlas_implementation_sha256",
            "attention_edge_fidelity_kernel_implementation_sha256",
            "attention_edge_fidelity_implementation_sha256",
            "implementation_bundle_sha256", "source_commit_sha",
            "pre_execution_source_state_sha256",
            "post_execution_source_state_sha256",
            "row_count", "family_counts",
            "layer_count", "query_head_count",
            "atlas_extraction_identity_count",
            "atlas_fisher_k8_coordinate_identity_count",
            "row_receipts", "cells", "family_cells",
            "role", "schema", "format_version", "abi", "report_sha256",
        }
        for field in bool_fields:
            if type(value[field]) is not bool:
                raise TypeError(f"{field} must be bool")
        return cls(
            source_kind=str(value["source_kind"]),
            model_id=str(value["model_id"]),
            model_revision=str(value["model_revision"]),
            panel_id=str(value["panel_id"]),
            panel_file_sha256=str(value["panel_file_sha256"]),
            row_catalog_sha256=str(value["row_catalog_sha256"]),
            tokenization_catalog_sha256=str(value["tokenization_catalog_sha256"]),
            source_model_fingerprint_sha256=str(value["source_model_fingerprint_sha256"]),
            source_execution_fingerprint_sha256=str(value["source_execution_fingerprint_sha256"]),
            score_objective_id=str(value["score_objective_id"]),
            atlas_report_sha256=str(value["atlas_report_sha256"]),
            atlas_file_sha256=str(value["atlas_file_sha256"]),
            atlas_implementation_bundle_sha256=str(value["atlas_implementation_bundle_sha256"]),
            exact_control_receipt_sha256=str(value["exact_control_receipt_sha256"]),
            attention_generator_implementation_sha256=str(value["attention_generator_implementation_sha256"]),
            joint_executor_implementation_sha256=str(value["joint_executor_implementation_sha256"]),
            joint_compiler_implementation_sha256=str(value["joint_compiler_implementation_sha256"]),
            attention_edge_core_implementation_sha256=str(value["attention_edge_core_implementation_sha256"]),
            attention_edge_atlas_implementation_sha256=str(value["attention_edge_atlas_implementation_sha256"]),
            attention_edge_fidelity_kernel_implementation_sha256=str(value["attention_edge_fidelity_kernel_implementation_sha256"]),
            attention_edge_fidelity_implementation_sha256=str(value["attention_edge_fidelity_implementation_sha256"]),
            implementation_bundle_sha256=str(value["implementation_bundle_sha256"]),
            source_commit_sha=str(value["source_commit_sha"]),
            pre_execution_source_state_sha256=str(
                value["pre_execution_source_state_sha256"]
            ),
            post_execution_source_state_sha256=str(
                value["post_execution_source_state_sha256"]
            ),
            row_count=_integer(value["row_count"], label="row_count", minimum=1),
            family_counts=tuple(parsed_families),
            layer_count=_integer(value["layer_count"], label="layer_count", minimum=1),
            query_head_count=_integer(value["query_head_count"], label="query_head_count", minimum=1),
            atlas_extraction_identity_count=_integer(
                value["atlas_extraction_identity_count"],
                label="atlas_extraction_identity_count",
                minimum=1,
            ),
            atlas_fisher_k8_coordinate_identity_count=_integer(
                value["atlas_fisher_k8_coordinate_identity_count"],
                label="atlas_fisher_k8_coordinate_identity_count",
                minimum=1,
            ),
            row_receipts=tuple(parsed_rows),
            cells=tuple(parsed_cells),
            family_cells=tuple(parsed_family_cells),
            atlas_fisher_top8_coordinate_match_passed=value["atlas_fisher_top8_coordinate_match_passed"],
            full_atlas_1080_identity_passed=value["full_atlas_1080_identity_passed"],
            all_dense_replay_controls_passed=value["all_dense_replay_controls_passed"],
            global_k_across_heads_and_keys=value["global_k_across_heads_and_keys"],
            fisher_ranker_is_same_row_target_gradient_oracle=value["fisher_ranker_is_same_row_target_gradient_oracle"],
            selection_masks_frozen_before_intervention=value["selection_masks_frozen_before_intervention"],
            qkv_and_probabilities_recomputed_on_sparse_state=value["qkv_and_probabilities_recomputed_on_sparse_state"],
            all_attention_layers_intervened_simultaneously=value["all_attention_layers_intervened_simultaneously"],
            empty_renormalized_heads_zero_filled=value["empty_renormalized_heads_zero_filled"],
            selection_prefix_nesting_passed=value["selection_prefix_nesting_passed"],
            clean_worktree_required_for_live_execution=value["clean_worktree_required_for_live_execution"],
            clean_worktree_observed=value["clean_worktree_observed"],
            committed_source_identity_passed=value["committed_source_identity_passed"],
            post_execution_source_recheck_passed=value[
                "post_execution_source_recheck_passed"
            ],
            self_proves_live_fidelity=value["self_proves_live_fidelity"],
            selection_optimality_claimed=value["selection_optimality_claimed"],
            retrospective_A_only=value["retrospective_A_only"],
            role=str(value["role"]),
            fresh_panel_used=value["fresh_panel_used"],
            attention_edge_confirmation_claimed=value["attention_edge_confirmation_claimed"],
            semantic_interpretation_claimed=value["semantic_interpretation_claimed"],
            compression_qualification_claimed=value["compression_qualification_claimed"],
            parameter_savings_claimed=value["parameter_savings_claimed"],
            mac_savings_claimed=value["mac_savings_claimed"],
            latency_claimed=value["latency_claimed"],
            prompts_persisted=value["prompts_persisted"],
            token_ids_persisted=value["token_ids_persisted"],
            raw_activations_persisted=value["raw_activations_persisted"],
            raw_gradients_persisted=value["raw_gradients_persisted"],
            schema=str(value["schema"]),
            format_version=_integer(value["format_version"], label="format_version", minimum=1),
            abi=str(value["abi"]),
            report_sha256=str(value["report_sha256"]),
        )


def compile_attention_edge_fidelity(
    captures: Sequence[AttentionEdgeFidelityRowCapture],
    *,
    source_kind: str,
    model_id: str,
    model_revision: str,
    panel_id: str,
    panel_file_sha256: str,
    row_catalog_sha256: str,
    tokenization_catalog_sha256: str,
    source_model_fingerprint_sha256: str,
    source_execution_fingerprint_sha256: str,
    atlas_report_sha256: str,
    atlas_file_sha256: str,
    atlas_implementation_bundle_sha256: str,
    exact_control_receipt_sha256: str,
    source_commit_sha: str,
    pre_execution_source_state_sha256: str,
    post_execution_source_state_sha256: str,
    clean_worktree_observed: bool,
    committed_source_identity_passed: bool,
    post_execution_source_recheck_passed: bool,
) -> AttentionEdgeFidelityReport:
    """Aggregate authenticated per-row sufficient statistics into the curve."""

    materialized = tuple(captures)
    if not materialized or any(
        not isinstance(item, AttentionEdgeFidelityRowCapture)
        for item in materialized
    ):
        raise ValueError("attention fidelity requires typed row captures")
    if len({item.row_catalog_id for item in materialized}) != len(materialized):
        raise ValueError("attention fidelity row catalog IDs must be unique")
    computed_row_catalog_sha256 = _json_sha256(
        [
            {
                "row_ordinal": ordinal,
                "row_catalog_id": row.row_catalog_id,
                "family_id": row.family_id,
            }
            for ordinal, row in enumerate(materialized)
        ],
        domain=_ATLAS_CATALOG_DOMAIN,
    )
    if row_catalog_sha256 != computed_row_catalog_sha256:
        raise ValueError("attention fidelity row catalog hash drifted")
    computed_tokenization_catalog_sha256 = _json_sha256(
        [
            {
                "row_ordinal": ordinal,
                "tokenization_sha256": row.tokenization_sha256,
            }
            for ordinal, row in enumerate(materialized)
        ],
        domain=_TOKENIZATION_CATALOG_DOMAIN,
    )
    if tokenization_catalog_sha256 != computed_tokenization_catalog_sha256:
        raise ValueError("attention fidelity tokenization catalog hash drifted")
    canonical_conditions = _conditions()
    rows: list[AttentionEdgeFidelityRowReceipt] = []
    by_condition: dict[
        AttentionEdgeFidelityCondition, list[AttentionEdgeFidelityMeasurement]
    ] = {condition: [] for condition in canonical_conditions}
    for row_ordinal, capture in enumerate(materialized):
        _validate_measurement_selection_ledger(capture.measurements)
        _validate_measurement_baseline_ledger(capture.measurements)
        first_measurement = capture.measurements[0]
        for measurement in capture.measurements[1:]:
            if (
                measurement.available_route_count
                != first_measurement.available_route_count
                or measurement.total_head_query_count
                != first_measurement.total_head_query_count
                or measurement.native_probability_mass
                != first_measurement.native_probability_mass
                or measurement.full_fisher_energy
                != first_measurement.full_fisher_energy
                or measurement.operator_reference_sum_squares
                != first_measurement.operator_reference_sum_squares
            ):
                raise ValueError("row-invariant native teacher baselines drifted")
        selector_groups: dict[
            tuple[str, int, str], list[AttentionEdgeFidelityMeasurement]
        ] = {}
        for measurement in capture.measurements:
            selector_groups.setdefault(
                (
                    measurement.condition.ranker,
                    measurement.condition.k,
                    measurement.condition.random_salt,
                ),
                [],
            ).append(measurement)
        for values in selector_groups.values():
            if len(values) == 2:
                left, right = values
                if any(
                    getattr(left, field) != getattr(right, field)
                    for field in (
                        "selection_catalog_sha256",
                        "available_route_count",
                        "selected_route_count",
                        "active_head_query_count",
                        "total_head_query_count",
                        "selected_probability_mass",
                        "native_probability_mass",
                        "selected_fisher_energy",
                        "full_fisher_energy",
                    )
                ):
                    raise ValueError(
                        "normalization variants must share one frozen selection"
                    )
            elif len(values) != 1:
                raise ValueError("selector condition multiplicity drifted")
        for measurement in capture.measurements:
            by_condition[measurement.condition].append(measurement)
        measurement_catalog = _json_sha256(
            [item.to_dict() for item in capture.measurements],
            domain=_SELECTION_DOMAIN,
        )
        rows.append(
            AttentionEdgeFidelityRowReceipt(
                row_ordinal=row_ordinal,
                family_id=capture.family_id,
                sequence_length=capture.sequence_length,
                tokenization_sha256=capture.tokenization_sha256,
                layer_extraction_sha256s=capture.layer_extraction_sha256s,
                dense_replay_valid_logits_sha256=(
                    capture.dense_replay_valid_logits_sha256
                ),
                baseline_valid_logits_sha256=(
                    capture.baseline_valid_logits_sha256
                ),
                matched_batch_size=capture.matched_batch_size,
                matched_control_catalog_sha256=(
                    capture.matched_control_catalog_sha256
                ),
                matched_control_diagnostics=capture.matched_control_diagnostics,
                matched_control_diagnostic_catalog_sha256=(
                    capture.matched_control_diagnostic_catalog_sha256
                ),
                native_baseline_nll=capture.native_baseline_nll,
                matched_controls_native_top1_parity_passed=(
                    capture.matched_controls_native_top1_parity_passed
                ),
                native_to_matched_valid_logit_max_abs_error=(
                    capture.native_to_matched_valid_logit_max_abs_error
                ),
                native_to_matched_nll_max_abs_delta=(
                    capture.native_to_matched_nll_max_abs_delta
                ),
                native_to_matched_centered_logit_max_nrmse=(
                    capture.native_to_matched_centered_logit_max_nrmse
                ),
                native_to_matched_final_carrier_max_nrmse=(
                    capture.native_to_matched_final_carrier_max_nrmse
                ),
                matched_control_nll_max_abs_delta_bound=(
                    capture.matched_control_nll_max_abs_delta_bound
                ),
                matched_control_centered_logit_nrmse_bound=(
                    capture.matched_control_centered_logit_nrmse_bound
                ),
                matched_control_final_carrier_nrmse_bound=(
                    capture.matched_control_final_carrier_nrmse_bound
                ),
                attention_operator_catalog_sha256=(
                    capture.attention_operator_catalog_sha256
                ),
                attention_delta_catalog_sha256=(
                    capture.attention_delta_catalog_sha256
                ),
                mutation_delta_catalog_sha256=(
                    capture.mutation_delta_catalog_sha256
                ),
                mutation_carrier_catalog_sha256=(
                    capture.mutation_carrier_catalog_sha256
                ),
                attention_operator_sha256s=(
                    capture.attention_operator_sha256s
                ),
                attention_delta_sha256s=capture.attention_delta_sha256s,
                mutation_delta_sha256s=capture.mutation_delta_sha256s,
                mutation_carrier_sha256s=capture.mutation_carrier_sha256s,
                final_carrier_sha256=capture.final_carrier_sha256,
                terminal_normalized_output_sha256=(
                    capture.terminal_normalized_output_sha256
                ),
                dense_replay_literal_parity_passed=(
                    capture.dense_replay_literal_parity_passed
                ),
                dense_replay_nll_literal_parity_passed=(
                    capture.dense_replay_nll_literal_parity_passed
                ),
                dense_replay_top1_parity_passed=(
                    capture.dense_replay_top1_parity_passed
                ),
                all_routes_operator_delta_controls_passed=(
                    capture.all_routes_operator_delta_controls_passed
                ),
                selection_prefix_nesting_passed=(
                    capture.selection_prefix_nesting_passed
                ),
                measurements=capture.measurements,
                measurement_catalog_sha256=measurement_catalog,
            )
        )
    cells: list[AttentionEdgeFidelityCell] = []
    for condition in canonical_conditions:
        values = by_condition[condition]
        if len(values) != len(materialized):
            raise ValueError("attention fidelity condition coverage drifted")
        cells.append(
            AttentionEdgeFidelityCell(
                condition=condition,
                selection_catalog_ledger_sha256=_json_sha256(
                    [item.selection_catalog_sha256 for item in values],
                    domain=_SELECTION_DOMAIN,
                ),
                row_count=len(values),
                available_route_count=sum(
                    item.available_route_count for item in values
                ),
                selected_route_count=sum(
                    item.selected_route_count for item in values
                ),
                active_head_query_count=sum(
                    item.active_head_query_count for item in values
                ),
                total_head_query_count=sum(
                    item.total_head_query_count for item in values
                ),
                selected_probability_mass=math.fsum(
                    item.selected_probability_mass for item in values
                ),
                native_probability_mass=math.fsum(
                    item.native_probability_mass for item in values
                ),
                selected_fisher_energy=math.fsum(
                    item.selected_fisher_energy for item in values
                ),
                full_fisher_energy=math.fsum(
                    item.full_fisher_energy for item in values
                ),
                operator_reference_sum_squares=math.fsum(
                    item.operator_reference_sum_squares for item in values
                ),
                operator_sparse_sum_squares=math.fsum(
                    item.operator_sparse_sum_squares for item in values
                ),
                operator_error_sum_squares=math.fsum(
                    item.operator_error_sum_squares for item in values
                ),
                operator_dot_product=math.fsum(
                    item.operator_dot_product for item in values
                ),
                operator_max_abs_error=max(
                    item.operator_max_abs_error for item in values
                ),
                final_carrier_reference_sum_squares=math.fsum(
                    item.final_carrier_reference_sum_squares for item in values
                ),
                final_carrier_sparse_sum_squares=math.fsum(
                    item.final_carrier_sparse_sum_squares for item in values
                ),
                final_carrier_error_sum_squares=math.fsum(
                    item.final_carrier_error_sum_squares for item in values
                ),
                final_carrier_dot_product=math.fsum(
                    item.final_carrier_dot_product for item in values
                ),
                final_carrier_max_abs_error=max(
                    item.final_carrier_max_abs_error for item in values
                ),
                terminal_reference_sum_squares=math.fsum(
                    item.terminal_reference_sum_squares for item in values
                ),
                terminal_sparse_sum_squares=math.fsum(
                    item.terminal_sparse_sum_squares for item in values
                ),
                terminal_error_sum_squares=math.fsum(
                    item.terminal_error_sum_squares for item in values
                ),
                terminal_dot_product=math.fsum(
                    item.terminal_dot_product for item in values
                ),
                terminal_max_abs_error=max(
                    item.terminal_max_abs_error for item in values
                ),
                baseline_valid_logit_sum_squares=math.fsum(
                    item.baseline_valid_logit_sum_squares for item in values
                ),
                sparse_valid_logit_sum_squares=math.fsum(
                    item.sparse_valid_logit_sum_squares for item in values
                ),
                valid_logit_error_sum_squares=math.fsum(
                    item.valid_logit_error_sum_squares for item in values
                ),
                valid_logit_dot_product=math.fsum(
                    item.valid_logit_dot_product for item in values
                ),
                metrics=_aggregate_metric_ledgers(
                    [item.metrics for item in values]
                ),
                baseline_nll_sum=math.fsum(
                    item.baseline_nll for item in values
                ),
                sparse_nll_sum=math.fsum(item.sparse_nll for item in values),
                top1_retained_count=sum(
                    int(item.top1_retained) for item in values
                ),
                baseline_target_correct_count=sum(
                    int(item.baseline_target_correct) for item in values
                ),
                sparse_target_correct_count=sum(
                    int(item.sparse_target_correct) for item in values
                ),
            )
        )
    family_cells = tuple(
        AttentionEdgeFidelityFamilyCell(
            family_id=family,
            cell=_aggregate_fidelity_cell(
                condition,
                [
                    measurement
                    for capture in materialized
                    if capture.family_id == family
                    for measurement in capture.measurements
                    if measurement.condition == condition
                ],
            ),
        )
        for family in sorted({item.family_id for item in materialized})
        for condition in canonical_conditions
    )
    (
        generator_hash,
        executor_hash,
        compiler_hash,
        core_hash,
        atlas_hash,
        fidelity_kernel_hash,
        fidelity_hash,
        bundle_hash,
    ) = _implementation_provenance()
    return AttentionEdgeFidelityReport(
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
        score_objective_id=MODEL_SCORE_OBJECTIVE_ID,
        atlas_report_sha256=atlas_report_sha256,
        atlas_file_sha256=atlas_file_sha256,
        atlas_implementation_bundle_sha256=(
            atlas_implementation_bundle_sha256
        ),
        exact_control_receipt_sha256=exact_control_receipt_sha256,
        attention_generator_implementation_sha256=generator_hash,
        joint_executor_implementation_sha256=executor_hash,
        joint_compiler_implementation_sha256=compiler_hash,
        attention_edge_core_implementation_sha256=core_hash,
        attention_edge_atlas_implementation_sha256=atlas_hash,
        attention_edge_fidelity_kernel_implementation_sha256=(
            fidelity_kernel_hash
        ),
        attention_edge_fidelity_implementation_sha256=fidelity_hash,
        implementation_bundle_sha256=bundle_hash,
        source_commit_sha=source_commit_sha,
        pre_execution_source_state_sha256=pre_execution_source_state_sha256,
        post_execution_source_state_sha256=post_execution_source_state_sha256,
        row_count=len(materialized),
        family_counts=tuple(
            sorted(Counter(item.family_id for item in materialized).items())
        ),
        layer_count=GEMMA3_LAYER_COUNT,
        query_head_count=ATTENTION_HEAD_COUNT,
        atlas_extraction_identity_count=(
            len(materialized) * GEMMA3_LAYER_COUNT
        ),
        atlas_fisher_k8_coordinate_identity_count=(
            len(materialized) * GEMMA3_LAYER_COUNT
        ),
        row_receipts=tuple(rows),
        cells=tuple(cells),
        family_cells=family_cells,
        full_atlas_1080_identity_passed=(source_kind == "model_capture"),
        clean_worktree_observed=clean_worktree_observed,
        committed_source_identity_passed=committed_source_identity_passed,
        post_execution_source_recheck_passed=(
            post_execution_source_recheck_passed
        ),
    )


def _attention_generators(executor: object) -> tuple[Gemma3AttentionGenerator, ...]:
    mutations = getattr(executor, "mutations", None)
    if not isinstance(mutations, nn.ModuleList) or len(mutations) != 36:
        raise ValueError("attention fidelity exact executor body drifted")
    result = tuple(
        getattr(mutations[2 * layer], "compiled_operator", None)
        for layer in range(GEMMA3_LAYER_COUNT)
    )
    if any(not isinstance(item, Gemma3AttentionGenerator) for item in result):
        raise ValueError("attention fidelity omitted an exact attention generator")
    return result  # type: ignore[return-value]


_COMMITTED_SOURCE_PATHS: Final = (
    "src/fisher_graph/gemma3_attention_generator.py",
    "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py",
    "src/fisher_graph/gemma3_joint_cross_block_compiler.py",
    "src/fisher_graph/gemma3_parts_first_attention_edges.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_experiment.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity_experiment.py",
)
_SOURCE_STATE_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-fidelity-source-state:v1\0"
)


@dataclass(frozen=True, slots=True)
class _CommittedSourceState:
    root: Path
    commit_sha: str
    implementation_sha256s: tuple[tuple[str, str], ...]
    state_sha256: str


def _capture_committed_source_state() -> _CommittedSourceState:
    module_path = Path(__file__).resolve()
    root_result = subprocess.run(
        ["git", "-C", str(module_path.parent), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    root = Path(root_result.stdout.strip()).resolve()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(
            "live attention fidelity requires a clean committed worktree"
        )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("attention fidelity could not authenticate HEAD")
    working_hashes: list[tuple[str, str]] = []
    for relative in _COMMITTED_SOURCE_PATHS:
        working_path = root / relative
        try:
            working_bytes = working_path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"attention fidelity source is unavailable: {relative}"
            ) from error
        head_result = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            check=False,
            capture_output=True,
        )
        if head_result.returncode != 0:
            raise RuntimeError(
                f"attention fidelity source is absent from HEAD: {relative}"
            )
        if working_bytes != head_result.stdout:
            raise RuntimeError(
                f"attention fidelity working bytes differ from HEAD: {relative}"
            )
        working_hashes.append(
            (relative, hashlib.sha256(working_bytes).hexdigest())
        )
    state_payload = {
        "commit_sha": commit,
        "implementation_sha256s": [list(item) for item in working_hashes],
    }
    return _CommittedSourceState(
        root=root,
        commit_sha=commit,
        implementation_sha256s=tuple(working_hashes),
        state_sha256=_json_sha256(state_payload, domain=_SOURCE_STATE_DOMAIN),
    )


def _descriptor_safe_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    descriptor: int | None = None
    primary: BaseException | None = None
    chunks: list[bytes] = []
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        initial = os.fstat(descriptor)
        named_initial = path.lstat()
        if (
            not os.path.samestat(initial, named_initial)
            or not stat.S_ISREG(initial.st_mode)
            or stat.S_IMODE(initial.st_mode) != 0o600
            or initial.st_nlink != 1
        ):
            raise ValueError(f"{label} file identity is unsafe")
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
            raise ValueError(f"{label} file changed during read")
    except OSError as error:
        primary = ValueError(f"{label} file is unavailable or unsafe")
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
                    f"{label} descriptor close failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    materialized = b"".join(chunks)
    return materialized, hashlib.sha256(materialized).hexdigest()


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityLiveConfig:
    atlas_path: Path = DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT
    panel_path: Path = PINNED_PANEL_PATH
    cache_dir: Path | None = None
    local_files_only: bool = True
    maximum_prompt_tokens: int = ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS
    output: Path = DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "atlas_path", Path(self.atlas_path))
        object.__setattr__(self, "panel_path", Path(self.panel_path))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "output", Path(self.output))
        if self.local_files_only is not True:
            raise ValueError("attention fidelity requires local-only checkpoint")
        if self.maximum_prompt_tokens != ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS:
            raise ValueError("attention fidelity maximum_prompt_tokens is frozen")
        if self.atlas_path.suffix != ".json" or self.output.suffix != ".json":
            raise ValueError("attention fidelity artifacts must use .json")


def _validate_source_atlas(atlas: AttentionEdgeAtlasReport) -> None:
    provenance = _implementation_provenance()
    current_core_hash = provenance[3]
    current_atlas_hash = provenance[4]
    if (
        atlas.report_sha256 != PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256
        or
        atlas.source_kind != "model_capture"
        or atlas.model_id != MODEL_ID
        or atlas.model_revision != PINNED_MODEL_REVISION
        or atlas.panel_id != PINNED_PANEL_ID
        or atlas.panel_file_sha256 != PINNED_PANEL_FILE_SHA256
        or atlas.tokenization_catalog_sha256
        != PINNED_ATTENTION_EDGE_TOKENIZATION_CATALOG_SHA256
        or atlas.source_model_fingerprint_sha256
        != SOURCE_MODEL_FINGERPRINT_SHA256
        or atlas.source_execution_fingerprint_sha256
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or atlas.config.atlas_abi != ATTENTION_EDGE_ATLAS_ABI
        or atlas.config.top_routes_per_target != ATTENTION_EDGE_TOP_K
        or atlas.roles_statistically_read != (ATTENTION_EDGE_ATLAS_ROLE,)
        or atlas.row_count != 60
        or atlas.layer_count != GEMMA3_LAYER_COUNT
        or atlas.query_head_count != ATTENTION_HEAD_COUNT
        or atlas.exact_control_receipt is None
        or not atlas.exact_control_receipt.passed
        or atlas.core_implementation_sha256 != current_core_hash
        or atlas.runner_implementation_sha256 != current_atlas_hash
    ):
        raise ValueError("published attention atlas is not the frozen live A atlas")


def _selection_catalog_sha256(
    masks: Sequence[Tensor], condition: AttentionEdgeFidelityCondition
) -> str:
    return _json_sha256(
        {
            "selector": {
                "ranker": condition.ranker,
                "k": condition.k,
                "random_salt": condition.random_salt,
            },
            "layer_mask_sha256s": [_tensor_sha256(mask) for mask in masks],
        },
        domain=_SELECTION_DOMAIN,
    )


def _catalog_sha256(values: Sequence[Tensor], *, label: str) -> str:
    return _json_sha256(
        {"label": label, "tensor_sha256s": [_tensor_sha256(item) for item in values]},
        domain=_TENSOR_DOMAIN,
    )


def _valid_logit_row(logits: Tensor, supervised_query: int) -> Tensor:
    if logits.ndim != 3 or not 0 <= supervised_query < logits.shape[1]:
        raise ValueError("supervised logit query is invalid")
    return logits[:, supervised_query, :].detach().clone()


def attention_fidelity_vector_stats(
    reference: Tensor, sparse: Tensor
) -> AttentionFidelityVectorStats:
    """Compute float64 sufficient statistics for two aligned tensors."""

    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    sparse64 = sparse.detach().to(device="cpu", dtype=torch.float64)
    if reference64.shape != sparse64.shape or not reference64.numel():
        raise ValueError("fidelity statistic tensors must align")
    error = sparse64 - reference64
    return AttentionFidelityVectorStats(
        element_count=int(reference64.numel()),
        reference_sum_squares=float(reference64.square().sum().item()),
        sparse_sum_squares=float(sparse64.square().sum().item()),
        error_sum_squares=float(error.square().sum().item()),
        dot_product=float((reference64 * sparse64).sum().item()),
        maximum_abs_error=float(error.abs().max().item()),
    )


def _stats(reference: Tensor, sparse: Tensor) -> AttentionFidelityVectorStats:
    return attention_fidelity_vector_stats(reference, sparse)


def _fisher_atlas_coordinates(
    atlas: AttentionEdgeAtlasReport,
) -> dict[tuple[int, int], frozenset[tuple[int, int, int]]]:
    result: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
    for route in atlas.retained_routes:
        result.setdefault((route.row_ordinal, route.layer_ordinal), set()).add(
            (
                route.query_head_ordinal,
                route.query_ordinal,
                route.key_ordinal,
            )
        )
    return {key: frozenset(value) for key, value in result.items()}


def _mask_coordinates(mask: Tensor) -> frozenset[tuple[int, int, int]]:
    indexes = torch.nonzero(mask[0], as_tuple=False).tolist()
    return frozenset((int(head), int(query), int(key)) for head, query, key in indexes)


def _run_batch1_all_routes_dense_replay(
    *,
    executor: object,
    input_ids: Tensor,
    sequence: SequenceContext,
    all_route_masks: tuple[Tensor, ...],
) -> fidelity_kernel.SparseWholeModelExecution:
    """Run the literal dense control with the baseline egress matrix shape."""

    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or sequence.batch_size != 1
        or len(all_route_masks) != GEMMA3_LAYER_COUNT
        or any(mask.shape[0] != 1 for mask in all_route_masks)
    ):
        raise ValueError("dense attention fidelity control must use batch size one")
    return fidelity_kernel.replay_exact_with_attention_masks(
        executor,
        input_ids,
        sequence,
        all_route_masks,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
    )


def _matched_batch_size() -> int:
    return len(_conditions())


def _run_batch53_attention_replay(
    *,
    executor: object,
    input_ids: Tensor,
    sequence: SequenceContext,
    condition_masks: Sequence[tuple[Tensor, ...]],
    condition_renormalize_rows: Tensor,
    supervised_query: int,
) -> fidelity_kernel.SparseWholeModelExecution:
    """Replay one row for every condition in the frozen ordinal order."""

    expected_conditions = len(_conditions())
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or sequence.batch_size != 1
        or len(condition_masks) != expected_conditions
        or any(
            len(masks) != GEMMA3_LAYER_COUNT
            or any(mask.shape[0] != 1 for mask in masks)
            for masks in condition_masks
        )
        or condition_renormalize_rows.dtype is not torch.bool
        or condition_renormalize_rows.device.type != "cpu"
        or tuple(condition_renormalize_rows.shape) != (expected_conditions,)
    ):
        raise ValueError("matched attention fidelity replay geometry drifted")
    batched_masks = tuple(
        torch.cat(
            [masks[layer] for masks in condition_masks],
            dim=0,
        )
        for layer in range(GEMMA3_LAYER_COUNT)
    )
    batch_size = _matched_batch_size()
    return fidelity_kernel.replay_exact_with_attention_masks(
        executor,
        input_ids.repeat(batch_size, 1),
        sequence,
        batched_masks,
        renormalize_rows=condition_renormalize_rows,
        egress_query_ordinal=supervised_query,
    )


def _matched_surface_tensors(
    execution: fidelity_kernel.SparseWholeModelExecution,
) -> tuple[tuple[str, tuple[Tensor, ...]], ...]:
    return (
        ("attention_operator", execution.attention_operator_outputs),
        ("attention_delta", execution.attention_residual_deltas),
        ("mutation_delta", execution.mutation_residual_deltas),
        ("mutation_carrier", execution.mutation_carriers),
        ("final_carrier", (execution.output,)),
        ("terminal_normalized_output", (execution.normalized_output,)),
        ("supervised_logits", (execution.logits,)),
    )


def _matched_surface_sha256(
    execution: fidelity_kernel.SparseWholeModelExecution,
    batch_ordinal: int,
) -> str:
    batch_size = _matched_batch_size()
    if type(batch_ordinal) is not int or not 0 <= batch_ordinal < batch_size:
        raise ValueError("matched replay batch ordinal is invalid")
    payload: list[dict[str, object]] = []
    for label, tensors in _matched_surface_tensors(execution):
        if any(tensor.shape[0] != batch_size for tensor in tensors):
            raise ValueError("matched replay surface batch geometry drifted")
        payload.append(
            {
                "label": label,
                "tensor_sha256s": [
                    _tensor_sha256(tensor[batch_ordinal : batch_ordinal + 1])
                    for tensor in tensors
                ],
            }
        )
    return _json_sha256(payload, domain=_MATCHED_SURFACE_DOMAIN)


def _matched_surface_rows_literal_equal(
    left: fidelity_kernel.SparseWholeModelExecution,
    left_ordinal: int,
    right: fidelity_kernel.SparseWholeModelExecution,
    right_ordinal: int,
) -> bool:
    batch_size = _matched_batch_size()
    if not all(
        type(ordinal) is int and 0 <= ordinal < batch_size
        for ordinal in (left_ordinal, right_ordinal)
    ):
        raise ValueError("matched replay comparison ordinal is invalid")
    left_surfaces = _matched_surface_tensors(left)
    right_surfaces = _matched_surface_tensors(right)
    for (left_label, left_tensors), (right_label, right_tensors) in zip(
        left_surfaces, right_surfaces, strict=True
    ):
        if left_label != right_label or len(left_tensors) != len(right_tensors):
            return False
        if any(
            left_tensor.shape[0] != batch_size
            or right_tensor.shape[0] != batch_size
            or not torch.equal(
                left_tensor[left_ordinal : left_ordinal + 1],
                right_tensor[right_ordinal : right_ordinal + 1],
            )
            for left_tensor, right_tensor in zip(
                left_tensors, right_tensors, strict=True
            )
        ):
            return False
    return True


@dataclass(slots=True)
class AttentionEdgeFidelityRuntime:
    atlas: AttentionEdgeAtlasReport
    edge_runtime: AttentionEdgeLiveRuntime
    atlas_coordinates: dict[
        tuple[int, int], frozenset[tuple[int, int, int]]
    ]
    source_commit_sha: str
    atlas_file_sha256: str
    pre_execution_source_state: _CommittedSourceState

    @classmethod
    def load(
        cls,
        config: AttentionEdgeFidelityLiveConfig = AttentionEdgeFidelityLiveConfig(),
    ) -> "AttentionEdgeFidelityRuntime":
        source_state = _capture_committed_source_state()
        atlas_bytes, atlas_file_sha256 = _descriptor_safe_bytes(
            config.atlas_path, label="published attention atlas"
        )
        if atlas_file_sha256 != PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256:
            raise ValueError("published attention atlas file SHA-256 drifted")
        atlas = parse_attention_edge_atlas_json(atlas_bytes)
        _validate_source_atlas(atlas)
        edge_runtime = AttentionEdgeLiveRuntime.load(
            AttentionEdgeAtlasLiveConfig(
                panel_path=config.panel_path,
                cache_dir=config.cache_dir,
                local_files_only=config.local_files_only,
                maximum_prompt_tokens=config.maximum_prompt_tokens,
                output=config.atlas_path,
            )
        )
        atlas_control = atlas.exact_control_receipt
        assert atlas_control is not None
        if (
            edge_runtime.exact_control_receipt.receipt_sha256
            != atlas_control.receipt_sha256
        ):
            raise RuntimeError("fidelity exact control differs from atlas control")
        if len(edge_runtime.rows) != atlas.row_count:
            raise RuntimeError("fidelity runtime row count differs from atlas")
        return cls(
            atlas=atlas,
            edge_runtime=edge_runtime,
            atlas_coordinates=_fisher_atlas_coordinates(atlas),
            source_commit_sha=source_state.commit_sha,
            atlas_file_sha256=atlas_file_sha256,
            pre_execution_source_state=source_state,
        )

    def _capture_row(self, row_ordinal: int) -> AttentionEdgeFidelityRowCapture:
        row = self.edge_runtime.rows[row_ordinal]
        executor = self.edge_runtime.executor
        generators = _attention_generators(executor)
        inputs: dict[str, Tensor] = {}
        deltas: dict[str, Tensor] = {}
        lexical_outputs: list[Tensor] = []

        def observer(*, event_id: str, role: str, values: Tensor) -> None:
            if event_id == "lexical.ingress" and role == "output":
                lexical_outputs.append(values)
                return
            if not event_id.startswith("mutation."):
                return
            index = int(event_id.split(".")[-1])
            if role == "input":
                inputs[event_id] = values
            elif role == "output":
                deltas[event_id] = values

        objective = CausalLanguageModelNLL()
        with torch.enable_grad():
            observed = executor.forward_components_with_observer(
                row.batch.model_inputs["input_ids"],
                row.sequence,
                observer=observer,
                retain_gradient_tape=True,
            )
            tape = observed.gradient_tape
            if tape is None:
                raise RuntimeError("attention fidelity exact execution omitted tape")
            baseline_nll_tensor = objective(observed.execution, row.batch)
            gradients = tape.score_gradients(baseline_nll_tensor)
        if len(inputs) != 36 or len(deltas) != 36 or len(lexical_outputs) != 1:
            raise RuntimeError("attention fidelity observer missed a mutation")
        baseline_nll = float(baseline_nll_tensor.detach().item())
        supervised = row.batch.valid_positions[0].nonzero(as_tuple=False)
        if tuple(supervised.shape) != (1, 1):
            raise ValueError("attention fidelity row requires one target query")
        supervised_query = int(supervised[0, 0].item())
        target = int(row.batch.targets[0, supervised_query].item())
        baseline_valid = _valid_logit_row(
            observed.execution.logits, supervised_query
        )
        extractions: list[PartsFirstAttentionEdgeExtraction] = []
        route_tensors: list[fidelity_kernel.AttentionRouteTensors] = []
        selections_by_layer: list[
            dict[tuple[str, int, str], AttentionRouteSelection]
        ] = []
        all_operator_controls = True
        nesting = True
        for layer, generator in enumerate(generators):
            event_id = f"mutation.{1 + 2 * layer}"
            extraction = compile_parts_first_attention_edges(
                generator,
                inputs[event_id],
                gradients[f"{event_id}.output"],
                row.sequence,
            )
            factors = fidelity_kernel.attention_route_tensors(
                generator, inputs[event_id], row.sequence
            )
            if not torch.equal(factors.probabilities, extraction.probabilities):
                raise RuntimeError("fidelity and atlas native probabilities differ")
            if not torch.equal(factors.reference_residual_delta, deltas[event_id]):
                raise RuntimeError("all-route attention delta differs from executor")
            all_operator_controls &= bool(
                factors.all_routes_operator_literal_parity_passed
                and factors.all_routes_delta_literal_parity_passed
            )
            extraction_hash = extraction.sha256
            if (
                extraction_hash
                != self.atlas.row_receipts[row_ordinal].layer_extraction_sha256s[
                    layer
                ]
            ):
                raise RuntimeError("live fidelity extraction differs from atlas")
            layer_selections: dict[
                tuple[str, int, str], AttentionRouteSelection
            ] = {}
            unique_specs = {
                (condition.ranker, condition.k, condition.random_salt)
                for condition in _conditions()
            }
            for ranker, k, random_salt in sorted(unique_specs):
                layer_selections[(ranker, k, random_salt)] = (
                    select_attention_routes(
                        extraction.probabilities,
                        extraction.intrinsic_fisher_energies,
                        extraction.allowed_mask,
                        ranker=ranker,
                        k=k,
                        row_catalog_id=row.row_catalog_id,
                        layer_ordinal=layer,
                        route_output_energies=factors.route_output_energies,
                        random_salt=random_salt,
                    )
                )
            for ranker in ATTENTION_EDGE_FIDELITY_RANKERS:
                previous: Tensor | None = None
                for k in ATTENTION_EDGE_FIDELITY_K_VALUES:
                    mask = layer_selections[(ranker, k, "primary")].mask
                    if previous is not None and bool((previous & ~mask).any()):
                        nesting = False
                    previous = mask
            fisher_k8 = layer_selections[("fisher", 8, "primary")].mask
            if _mask_coordinates(fisher_k8) != self.atlas_coordinates.get(
                (row_ordinal, layer), frozenset()
            ):
                raise RuntimeError("Fisher K8 coordinates differ from atlas")
            extractions.append(extraction)
            route_tensors.append(factors)
            selections_by_layer.append(layer_selections)
        if not all_operator_controls or not nesting:
            raise RuntimeError("attention fidelity local controls failed")

        all_route_masks = tuple(
            factors.allowed_mask[:, None, :, :].expand_as(factors.probabilities)
            for factors in route_tensors
        )
        dense = _run_batch1_all_routes_dense_replay(
            executor=executor,
            input_ids=row.batch.model_inputs["input_ids"],
            sequence=row.sequence,
            all_route_masks=all_route_masks,
        )
        operator_parity = all(
            torch.equal(actual, expected.reference_operator_output)
            for actual, expected in zip(
                dense.attention_operator_outputs, route_tensors, strict=True
            )
        )
        delta_parity = all(
            torch.equal(actual, deltas[f"mutation.{mutation + 1}"])
            for mutation, actual in enumerate(dense.mutation_residual_deltas)
        )
        valid_carrier = row.sequence.query_valid_mask.unsqueeze(-1)
        baseline_carrier = lexical_outputs[0]
        baseline_mutation_carrier_list: list[Tensor] = []
        for mutation in range(36):
            delta = deltas[f"mutation.{mutation + 1}"]
            baseline_carrier = torch.where(
                valid_carrier,
                baseline_carrier + delta.masked_fill(~valid_carrier, 0),
                lexical_outputs[0],
            )
            baseline_mutation_carrier_list.append(baseline_carrier)
        baseline_mutation_carriers = tuple(baseline_mutation_carrier_list)
        carrier_parity = all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                dense.mutation_carriers,
                baseline_mutation_carriers,
                strict=True,
            )
        )
        # Literal parity requires the same full-sequence tied egress shape as
        # the baseline executor.  Projecting only the supervised row selects
        # a different CPU GEMM kernel and changes otherwise-identical logits.
        dense_valid = dense.logits[:, supervised_query, :]
        dense_nll = float(
            F.cross_entropy(
                dense_valid,
                torch.tensor([target], dtype=torch.int64),
                reduction="sum",
            ).item()
        )
        final_parity = bool(
            torch.equal(dense.output, observed.execution.output)
            and torch.equal(
                dense.normalized_output, observed.execution.normalized_output
            )
            and torch.equal(dense_valid, baseline_valid)
        )
        nll_parity = dense_nll == baseline_nll
        top1_parity = int(dense_valid.argmax(dim=-1).item()) == int(
            baseline_valid.argmax(dim=-1).item()
        )
        if not (
            operator_parity
            and delta_parity
            and carrier_parity
            and final_parity
            and nll_parity
            and top1_parity
        ):
            raise RuntimeError("all-routes whole-stack replay failed literal parity")

        conditions = _conditions()
        condition_masks: list[tuple[Tensor, ...]] = []
        for condition in conditions:
            condition_masks.append(
                tuple(
                    selections_by_layer[layer][
                        (condition.ranker, condition.k, condition.random_salt)
                    ].mask
                    for layer in range(GEMMA3_LAYER_COUNT)
                )
            )
        random_replicate_catalogs = {
            _selection_catalog_sha256(condition_masks[index], condition)
            for index, condition in enumerate(conditions)
            if condition.ranker == "deterministic_random"
            and condition.k == ATTENTION_EDGE_TOP_K
            and condition.variant == "unnormalized_contribution"
        }
        if len(random_replicate_catalogs) != len(
            ATTENTION_EDGE_FIDELITY_RANDOM_SALTS
        ):
            raise RuntimeError("deterministic random replicate catalogs collided")
        condition_renormalize = torch.tensor(
            [
                condition.variant == "renormalized_sparse_attention"
                for condition in conditions
            ],
            dtype=torch.bool,
        )
        control_replay = _run_batch53_attention_replay(
            executor=executor,
            input_ids=row.batch.model_inputs["input_ids"],
            sequence=row.sequence,
            condition_masks=tuple(all_route_masks for _ in conditions),
            condition_renormalize_rows=torch.zeros(
                len(conditions), dtype=torch.bool
            ),
            supervised_query=supervised_query,
        )
        sparse_replay = _run_batch53_attention_replay(
            executor=executor,
            input_ids=row.batch.model_inputs["input_ids"],
            sequence=row.sequence,
            condition_masks=condition_masks,
            condition_renormalize_rows=condition_renormalize,
            supervised_query=supervised_query,
        )
        matched_control_receipts = tuple(
            _matched_surface_sha256(control_replay, index)
            for index in range(len(conditions))
        )
        matched_control_catalog_sha256 = _json_sha256(
            list(matched_control_receipts),
            domain=_MATCHED_SURFACE_DOMAIN,
        )
        control_valid = control_replay.logits[:, 0, :].detach()
        sparse_valid = sparse_replay.logits[:, 0, :].detach()
        if control_valid.shape != sparse_valid.shape or (
            control_valid.shape[0] != len(conditions)
        ):
            raise RuntimeError("paired batch53 replay omitted a condition")
        repeated_targets = torch.full(
            (len(conditions),), target, dtype=torch.int64
        )
        control_nlls = F.cross_entropy(
            control_valid,
            repeated_targets,
            reduction="none",
        ).to(torch.float64)
        sparse_nlls = F.cross_entropy(
            sparse_valid,
            repeated_targets,
            reduction="none",
        ).to(torch.float64)
        native_top1 = int(baseline_valid.argmax(dim=-1).item())
        control_top1s = control_valid.argmax(dim=-1)
        control_nll_values = tuple(
            float(control_nlls[index].item())
            for index in range(len(conditions))
        )
        sparse_nll_values = tuple(
            float(sparse_nlls[index].item())
            for index in range(len(conditions))
        )
        native_to_matched_absolute_nll_deltas = tuple(
            abs(value - baseline_nll) for value in control_nll_values
        )
        native_to_matched_valid_logit_max_abs_errors = tuple(
            float(
                (control_valid[index] - baseline_valid[0])
                .abs()
                .max()
                .item()
            )
            for index in range(len(conditions))
        )
        native_centered_logits = baseline_valid[0].to(torch.float64)
        native_centered_logits = (
            native_centered_logits - native_centered_logits.mean()
        )
        control_centered_logits = control_valid.to(torch.float64)
        control_centered_logits = control_centered_logits - (
            control_centered_logits.mean(dim=-1, keepdim=True)
        )
        native_to_matched_centered_logit_nrmses = tuple(
            _stats(native_centered_logits, control_centered_logits[index]).nrmse
            for index in range(len(conditions))
        )
        valid_hidden_mask = row.sequence.query_valid_mask[0].unsqueeze(-1)
        valid_hidden_mask = valid_hidden_mask.expand_as(
            observed.execution.output[0]
        )
        native_valid_carrier = observed.execution.output[0].masked_select(
            valid_hidden_mask
        )
        native_to_matched_final_carrier_nrmses = tuple(
            _stats(
                native_valid_carrier,
                control_replay.output[index].masked_select(valid_hidden_mask),
            ).nrmse
            for index in range(len(conditions))
        )
        matched_control_diagnostics = tuple(
            AttentionEdgeFidelityMatchedControlDiagnostic(
                condition=condition,
                matched_control_receipt_sha256=matched_control_receipts[index],
                native_to_matched_valid_logit_max_abs_error=(
                    native_to_matched_valid_logit_max_abs_errors[index]
                ),
                absolute_nll_delta=(
                    native_to_matched_absolute_nll_deltas[index]
                ),
                centered_logit_nrmse=(
                    native_to_matched_centered_logit_nrmses[index]
                ),
                final_carrier_nrmse=(
                    native_to_matched_final_carrier_nrmses[index]
                ),
                native_top1_parity_passed=(
                    int(control_top1s[index].item()) == native_top1
                ),
            )
            for index, condition in enumerate(conditions)
        )
        matched_control_diagnostic_catalog_sha256 = (
            _matched_control_diagnostic_catalog_sha256(
                matched_control_diagnostics
            )
        )
        native_to_matched_valid_logit_max_abs_error = max(
            item.native_to_matched_valid_logit_max_abs_error
            for item in matched_control_diagnostics
        )
        native_to_matched_nll_max_abs_delta = max(
            item.absolute_nll_delta for item in matched_control_diagnostics
        )
        native_to_matched_centered_logit_max_nrmse = max(
            item.centered_logit_nrmse for item in matched_control_diagnostics
        )
        native_to_matched_final_carrier_max_nrmse = max(
            item.final_carrier_nrmse for item in matched_control_diagnostics
        )
        matched_controls_native_top1_parity = all(
            item.native_top1_parity_passed
            for item in matched_control_diagnostics
        )
        if not matched_controls_native_top1_parity:
            raise RuntimeError("matched all-routes controls changed native top1")
        _validate_matched_control_bounds(
            maximum_absolute_nll_delta=native_to_matched_nll_max_abs_delta,
            maximum_centered_logit_nrmse=(
                native_to_matched_centered_logit_max_nrmse
            ),
            maximum_final_carrier_nrmse=(
                native_to_matched_final_carrier_max_nrmse
            ),
        )

        measurements: list[AttentionEdgeFidelityMeasurement] = []
        valid_operator = row.sequence.query_valid_mask.unsqueeze(-1)
        for condition_index, condition in enumerate(conditions):
            available_count = 0
            selected_count = 0
            active_count = 0
            total_head_queries = 0
            selected_probability: list[float] = []
            native_probability: list[float] = []
            selected_energy: list[float] = []
            full_energy: list[float] = []
            local_operator_stats: list[AttentionFidelityVectorStats] = []
            local_delta_stats: list[AttentionFidelityVectorStats] = []
            for layer, (generator, extraction, factors) in enumerate(
                zip(generators, extractions, route_tensors, strict=True)
            ):
                selection = selections_by_layer[layer][
                    (condition.ranker, condition.k, condition.random_salt)
                ]
                mask = selection.mask
                selected_native = factors.probabilities.masked_fill(~mask, 0)
                if condition.variant == "renormalized_sparse_attention":
                    mass = selected_native.sum(dim=-1, keepdim=True)
                    routed = torch.where(
                        mass > 0,
                        selected_native / mass.clamp_min(1e-45),
                        0,
                    )
                else:
                    routed = selected_native
                local_operator, local_delta = (
                    fidelity_kernel.operator_from_routed_probabilities(
                        generator,
                        routed,
                        factors.repeated_values,
                        row.sequence,
                    )
                )
                operator_mask = valid_operator.expand_as(local_operator)
                local_stats = _stats(
                    factors.reference_operator_output.masked_select(operator_mask),
                    local_operator.masked_select(operator_mask),
                )
                delta_stats = _stats(
                    factors.reference_residual_delta.masked_select(operator_mask),
                    local_delta.masked_select(operator_mask),
                )
                available_count += selection.available_route_count
                selected_count += selection.selected_route_count
                active_count += selection.active_head_query_count
                total_head_queries += selection.total_head_query_count
                selected_probability.append(
                    float(factors.probabilities.masked_select(mask).sum().item())
                )
                native_probability.append(float(factors.probabilities.sum().item()))
                selected_energy.append(
                    float(
                        extraction.intrinsic_fisher_energies.masked_select(mask)
                        .sum()
                        .item()
                    )
                )
                full_energy.append(
                    float(extraction.intrinsic_fisher_energies.sum().item())
                )
                local_operator_stats.append(local_stats)
                local_delta_stats.append(delta_stats)
            logits_stats = _stats(
                control_valid[condition_index], sparse_valid[condition_index]
            )
            baseline_centered = control_valid[condition_index].to(torch.float64)
            baseline_centered = baseline_centered - baseline_centered.mean(
                dim=-1, keepdim=True
            )
            sparse_centered = sparse_valid[condition_index].to(torch.float64)
            sparse_centered = sparse_centered - sparse_centered.mean(
                dim=-1, keepdim=True
            )
            centered_logits_stats = _stats(
                baseline_centered,
                sparse_centered,
            )
            carrier_stats = _stats(
                control_replay.output[condition_index].masked_select(
                    valid_hidden_mask
                ),
                sparse_replay.output[condition_index].masked_select(
                    valid_hidden_mask
                ),
            )
            supervised_carrier_stats = _stats(
                control_replay.output[condition_index, supervised_query],
                sparse_replay.output[condition_index, supervised_query],
            )
            terminal_stats = _stats(
                control_replay.normalized_output[condition_index].masked_select(
                    valid_hidden_mask
                ),
                sparse_replay.normalized_output[condition_index].masked_select(
                    valid_hidden_mask
                ),
            )
            supervised_terminal_stats = _stats(
                control_replay.normalized_output[
                    condition_index, supervised_query
                ],
                sparse_replay.normalized_output[
                    condition_index, supervised_query
                ],
            )
            propagated_stats = tuple(
                _stats(
                    control_replay.mutation_carriers[boundary][
                        condition_index
                    ].masked_select(valid_hidden_mask),
                    sparse_replay.mutation_carriers[boundary][
                        condition_index
                    ].masked_select(valid_hidden_mask),
                )
                for boundary in range(36)
            )
            operator_stats = _aggregate_vector_stats(local_operator_stats)
            postnorm_delta_stats = _aggregate_vector_stats(local_delta_stats)
            metric_ledger = AttentionEdgeFidelityMetricLedger(
                local_operator=operator_stats,
                local_postnorm_delta=postnorm_delta_stats,
                propagated_mutations=propagated_stats,
                final_valid_carrier=carrier_stats,
                supervised_carrier=supervised_carrier_stats,
                terminal_valid=terminal_stats,
                supervised_terminal=supervised_terminal_stats,
                valid_logits=logits_stats,
                centered_valid_logits=centered_logits_stats,
            )
            measurements.append(
                AttentionEdgeFidelityMeasurement(
                    condition=condition,
                    selection_catalog_sha256=_selection_catalog_sha256(
                        condition_masks[condition_index], condition
                    ),
                    matched_control_receipt_sha256=(
                        matched_control_receipts[condition_index]
                    ),
                    available_route_count=available_count,
                    selected_route_count=selected_count,
                    active_head_query_count=active_count,
                    total_head_query_count=total_head_queries,
                    selected_probability_mass=math.fsum(selected_probability),
                    native_probability_mass=math.fsum(native_probability),
                    selected_fisher_energy=math.fsum(selected_energy),
                    full_fisher_energy=math.fsum(full_energy),
                    operator_reference_sum_squares=(
                        operator_stats.reference_sum_squares
                    ),
                    operator_sparse_sum_squares=(
                        operator_stats.sparse_sum_squares
                    ),
                    operator_error_sum_squares=operator_stats.error_sum_squares,
                    operator_dot_product=operator_stats.dot_product,
                    operator_max_abs_error=operator_stats.maximum_abs_error,
                    final_carrier_reference_sum_squares=(
                        carrier_stats.reference_sum_squares
                    ),
                    final_carrier_sparse_sum_squares=(
                        carrier_stats.sparse_sum_squares
                    ),
                    final_carrier_error_sum_squares=(
                        carrier_stats.error_sum_squares
                    ),
                    final_carrier_dot_product=carrier_stats.dot_product,
                    final_carrier_max_abs_error=(
                        carrier_stats.maximum_abs_error
                    ),
                    terminal_reference_sum_squares=(
                        terminal_stats.reference_sum_squares
                    ),
                    terminal_sparse_sum_squares=(
                        terminal_stats.sparse_sum_squares
                    ),
                    terminal_error_sum_squares=terminal_stats.error_sum_squares,
                    terminal_dot_product=terminal_stats.dot_product,
                    terminal_max_abs_error=terminal_stats.maximum_abs_error,
                    baseline_valid_logit_sum_squares=(
                        logits_stats.reference_sum_squares
                    ),
                    sparse_valid_logit_sum_squares=(
                        logits_stats.sparse_sum_squares
                    ),
                    valid_logit_error_sum_squares=(
                        logits_stats.error_sum_squares
                    ),
                    valid_logit_dot_product=logits_stats.dot_product,
                    metrics=metric_ledger,
                    baseline_nll=control_nll_values[condition_index],
                    sparse_nll=sparse_nll_values[condition_index],
                    top1_retained=(
                        int(sparse_valid[condition_index].argmax().item())
                        == int(control_top1s[condition_index].item())
                    ),
                    baseline_target_correct=(
                        int(control_top1s[condition_index].item()) == target
                    ),
                    sparse_target_correct=(
                        int(sparse_valid[condition_index].argmax().item())
                        == target
                    ),
                )
            )
        operator_catalog = _catalog_sha256(
            dense.attention_operator_outputs, label="attention_operator"
        )
        delta_catalog = _catalog_sha256(
            dense.attention_residual_deltas, label="attention_delta"
        )
        mutation_delta_catalog = _catalog_sha256(
            dense.mutation_residual_deltas, label="mutation_delta"
        )
        mutation_carrier_catalog = _catalog_sha256(
            dense.mutation_carriers, label="mutation_carrier"
        )
        return AttentionEdgeFidelityRowCapture(
            row_catalog_id=row.row_catalog_id,
            family_id=row.family_id,
            sequence_length=row.sequence.query_length,
            tokenization_sha256=row.tokenization_sha256,
            layer_extraction_sha256s=tuple(item.sha256 for item in extractions),
            dense_replay_valid_logits_sha256=_tensor_sha256(dense_valid),
            baseline_valid_logits_sha256=_tensor_sha256(baseline_valid),
            matched_batch_size=_matched_batch_size(),
            matched_control_catalog_sha256=matched_control_catalog_sha256,
            matched_control_diagnostics=matched_control_diagnostics,
            matched_control_diagnostic_catalog_sha256=(
                matched_control_diagnostic_catalog_sha256
            ),
            native_baseline_nll=baseline_nll,
            matched_controls_native_top1_parity_passed=(
                matched_controls_native_top1_parity
            ),
            native_to_matched_valid_logit_max_abs_error=(
                native_to_matched_valid_logit_max_abs_error
            ),
            native_to_matched_nll_max_abs_delta=(
                native_to_matched_nll_max_abs_delta
            ),
            native_to_matched_centered_logit_max_nrmse=(
                native_to_matched_centered_logit_max_nrmse
            ),
            native_to_matched_final_carrier_max_nrmse=(
                native_to_matched_final_carrier_max_nrmse
            ),
            matched_control_nll_max_abs_delta_bound=(
                ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA
            ),
            matched_control_centered_logit_nrmse_bound=(
                ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE
            ),
            matched_control_final_carrier_nrmse_bound=(
                ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE
            ),
            attention_operator_catalog_sha256=operator_catalog,
            attention_delta_catalog_sha256=delta_catalog,
            mutation_delta_catalog_sha256=mutation_delta_catalog,
            mutation_carrier_catalog_sha256=mutation_carrier_catalog,
            attention_operator_sha256s=tuple(
                _tensor_sha256(item)
                for item in dense.attention_operator_outputs
            ),
            attention_delta_sha256s=tuple(
                _tensor_sha256(item)
                for item in dense.attention_residual_deltas
            ),
            mutation_delta_sha256s=tuple(
                _tensor_sha256(item) for item in dense.mutation_residual_deltas
            ),
            mutation_carrier_sha256s=tuple(
                _tensor_sha256(item) for item in dense.mutation_carriers
            ),
            final_carrier_sha256=_tensor_sha256(dense.output),
            terminal_normalized_output_sha256=_tensor_sha256(
                dense.normalized_output
            ),
            dense_replay_literal_parity_passed=final_parity,
            dense_replay_nll_literal_parity_passed=nll_parity,
            dense_replay_top1_parity_passed=top1_parity,
            all_routes_operator_delta_controls_passed=(
                operator_parity
                and delta_parity
                and carrier_parity
                and all_operator_controls
            ),
            selection_prefix_nesting_passed=nesting,
            measurements=tuple(measurements),
        )

    def capture(self, *, row_limit: int | None = None) -> AttentionEdgeFidelityReport:
        if row_limit is None:
            selected_count = len(self.edge_runtime.rows)
            source_kind = "model_capture"
        elif type(row_limit) is int and row_limit == 1:
            selected_count = 1
            source_kind = "model_smoke"
        else:
            raise ValueError("only the exact one-row fidelity smoke is supported")
        captures = tuple(self._capture_row(index) for index in range(selected_count))
        post_state = _capture_committed_source_state()
        if post_state != self.pre_execution_source_state:
            raise RuntimeError(
                "attention fidelity committed source state changed during execution"
            )
        atlas_control = self.atlas.exact_control_receipt
        assert atlas_control is not None
        return compile_attention_edge_fidelity(
            captures,
            source_kind=source_kind,
            model_id=self.atlas.model_id,
            model_revision=self.atlas.model_revision,
            panel_id=self.atlas.panel_id,
            panel_file_sha256=self.atlas.panel_file_sha256,
            row_catalog_sha256=(
                self.atlas.row_catalog_sha256
                if row_limit is None
                else _json_sha256(
                    [
                        {
                            "row_ordinal": 0,
                            "row_catalog_id": captures[0].row_catalog_id,
                            "family_id": captures[0].family_id,
                        }
                    ],
                    domain=_ATLAS_CATALOG_DOMAIN,
                )
            ),
            tokenization_catalog_sha256=(
                self.atlas.tokenization_catalog_sha256
                if row_limit is None
                else _json_sha256(
                    [
                        {
                            "row_ordinal": 0,
                            "tokenization_sha256": captures[0].tokenization_sha256,
                        }
                    ],
                    domain=_TOKENIZATION_CATALOG_DOMAIN,
                )
            ),
            source_model_fingerprint_sha256=(
                self.atlas.source_model_fingerprint_sha256
            ),
            source_execution_fingerprint_sha256=(
                self.atlas.source_execution_fingerprint_sha256
            ),
            atlas_report_sha256=self.atlas.report_sha256,
            atlas_file_sha256=self.atlas_file_sha256,
            atlas_implementation_bundle_sha256=(
                self.atlas.implementation_bundle_sha256
            ),
            exact_control_receipt_sha256=atlas_control.receipt_sha256,
            source_commit_sha=self.source_commit_sha,
            pre_execution_source_state_sha256=(
                self.pre_execution_source_state.state_sha256
            ),
            post_execution_source_state_sha256=post_state.state_sha256,
            clean_worktree_observed=True,
            committed_source_identity_passed=True,
            post_execution_source_recheck_passed=True,
        )


def build_deterministic_attention_edge_fidelity_fixture() -> AttentionEdgeFidelityReport:
    """Build a model-free, fully serializable schema/algebra fixture."""

    native_fixture_nll = 2.0
    measurements: list[AttentionEdgeFidelityMeasurement] = []
    for condition_index, condition in enumerate(_conditions()):
        matched_fixture_nll = native_fixture_nll + 1e-7 * (
            condition_index + 1
        )
        fraction = condition.k / max(ATTENTION_EDGE_FIDELITY_K_VALUES)
        variant_penalty = (
            0.8
            if condition.variant == "renormalized_sparse_attention"
            else 1.0
        )
        selector = {
            "ranker": condition.ranker,
            "k": condition.k,
            "random_salt": condition.random_salt,
        }
        reference = torch.arange(1, 17, dtype=torch.float64)
        sparse_values = reference * (
            1.0 - (0.4 / condition.k) * variant_penalty
        )
        vector_stats = _stats(reference, sparse_values)
        metrics = AttentionEdgeFidelityMetricLedger(
            local_operator=vector_stats,
            local_postnorm_delta=vector_stats,
            propagated_mutations=tuple(vector_stats for _ in range(36)),
            final_valid_carrier=vector_stats,
            supervised_carrier=vector_stats,
            terminal_valid=vector_stats,
            supervised_terminal=vector_stats,
            valid_logits=vector_stats,
            centered_valid_logits=vector_stats,
        )
        measurements.append(
            AttentionEdgeFidelityMeasurement(
                condition=condition,
                selection_catalog_sha256=_json_sha256(
                    selector, domain=_SELECTION_DOMAIN
                ),
                matched_control_receipt_sha256=_json_sha256(
                    {
                        "fixture_matched_control_ordinal": condition_index,
                        "condition": condition.to_dict(),
                    },
                    domain=_MATCHED_SURFACE_DOMAIN,
                ),
                available_route_count=1800,
                selected_route_count=18 * condition.k,
                active_head_query_count=min(720, 36 * condition.k),
                total_head_query_count=720,
                selected_probability_mass=720.0 * fraction,
                native_probability_mass=720.0,
                selected_fisher_energy=100.0 * fraction,
                full_fisher_energy=100.0,
                operator_reference_sum_squares=(
                    vector_stats.reference_sum_squares
                ),
                operator_sparse_sum_squares=vector_stats.sparse_sum_squares,
                operator_error_sum_squares=vector_stats.error_sum_squares,
                operator_dot_product=vector_stats.dot_product,
                operator_max_abs_error=vector_stats.maximum_abs_error,
                final_carrier_reference_sum_squares=(
                    vector_stats.reference_sum_squares
                ),
                final_carrier_sparse_sum_squares=(
                    vector_stats.sparse_sum_squares
                ),
                final_carrier_error_sum_squares=(
                    vector_stats.error_sum_squares
                ),
                final_carrier_dot_product=vector_stats.dot_product,
                final_carrier_max_abs_error=vector_stats.maximum_abs_error,
                terminal_reference_sum_squares=(
                    vector_stats.reference_sum_squares
                ),
                terminal_sparse_sum_squares=vector_stats.sparse_sum_squares,
                terminal_error_sum_squares=vector_stats.error_sum_squares,
                terminal_dot_product=vector_stats.dot_product,
                terminal_max_abs_error=vector_stats.maximum_abs_error,
                baseline_valid_logit_sum_squares=(
                    vector_stats.reference_sum_squares
                ),
                sparse_valid_logit_sum_squares=(
                    vector_stats.sparse_sum_squares
                ),
                valid_logit_error_sum_squares=(
                    vector_stats.error_sum_squares
                ),
                valid_logit_dot_product=vector_stats.dot_product,
                metrics=metrics,
                baseline_nll=matched_fixture_nll,
                sparse_nll=(
                    matched_fixture_nll
                    + (0.5 / condition.k) * variant_penalty
                ),
                top1_retained=condition.k >= 4,
                baseline_target_correct=True,
                sparse_target_correct=condition.k >= 4,
            )
        )
    matched_control_diagnostics = tuple(
        AttentionEdgeFidelityMatchedControlDiagnostic(
            condition=measurement.condition,
            matched_control_receipt_sha256=(
                measurement.matched_control_receipt_sha256
            ),
            native_to_matched_valid_logit_max_abs_error=(
                0.001 * (condition_index + 1) / len(measurements)
            ),
            absolute_nll_delta=abs(
                measurement.baseline_nll - native_fixture_nll
            ),
            centered_logit_nrmse=(
                2e-6 * (condition_index + 1) / len(measurements)
            ),
            final_carrier_nrmse=(
                4e-8 * (condition_index + 1) / len(measurements)
            ),
            native_top1_parity_passed=True,
        )
        for condition_index, measurement in enumerate(measurements)
    )
    digest = "1" * 64
    fixture_attention_operator_sha256s = tuple(
        hashlib.sha256(f"fixture-attention-operator-{index}".encode()).hexdigest()
        for index in range(GEMMA3_LAYER_COUNT)
    )
    fixture_mutation_delta_sha256s = tuple(
        hashlib.sha256(f"fixture-mutation-delta-{index}".encode()).hexdigest()
        for index in range(36)
    )
    fixture_attention_delta_sha256s = fixture_mutation_delta_sha256s[::2]
    fixture_mutation_carrier_sha256s = tuple(
        hashlib.sha256(f"fixture-mutation-carrier-{index}".encode()).hexdigest()
        for index in range(36)
    )
    capture = AttentionEdgeFidelityRowCapture(
        row_catalog_id="fixture-row",
        family_id="fixture-family",
        sequence_length=10,
        tokenization_sha256="2" * 64,
        layer_extraction_sha256s=tuple(
            hashlib.sha256(f"fixture-layer-{layer}".encode()).hexdigest()
            for layer in range(GEMMA3_LAYER_COUNT)
        ),
        dense_replay_valid_logits_sha256="3" * 64,
        baseline_valid_logits_sha256="3" * 64,
        matched_batch_size=_matched_batch_size(),
        matched_control_catalog_sha256=_json_sha256(
            [item.matched_control_receipt_sha256 for item in measurements],
            domain=_MATCHED_SURFACE_DOMAIN,
        ),
        matched_control_diagnostics=matched_control_diagnostics,
        matched_control_diagnostic_catalog_sha256=(
            _matched_control_diagnostic_catalog_sha256(
                matched_control_diagnostics
            )
        ),
        native_baseline_nll=native_fixture_nll,
        matched_controls_native_top1_parity_passed=True,
        native_to_matched_valid_logit_max_abs_error=max(
            item.native_to_matched_valid_logit_max_abs_error
            for item in matched_control_diagnostics
        ),
        native_to_matched_nll_max_abs_delta=max(
            item.absolute_nll_delta for item in matched_control_diagnostics
        ),
        native_to_matched_centered_logit_max_nrmse=max(
            item.centered_logit_nrmse for item in matched_control_diagnostics
        ),
        native_to_matched_final_carrier_max_nrmse=max(
            item.final_carrier_nrmse for item in matched_control_diagnostics
        ),
        matched_control_nll_max_abs_delta_bound=(
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA
        ),
        matched_control_centered_logit_nrmse_bound=(
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE
        ),
        matched_control_final_carrier_nrmse_bound=(
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE
        ),
        attention_operator_catalog_sha256=_json_sha256(
            {
                "label": "attention_operator",
                "tensor_sha256s": list(fixture_attention_operator_sha256s),
            },
            domain=_TENSOR_DOMAIN,
        ),
        attention_delta_catalog_sha256=_json_sha256(
            {
                "label": "attention_delta",
                "tensor_sha256s": list(fixture_attention_delta_sha256s),
            },
            domain=_TENSOR_DOMAIN,
        ),
        mutation_delta_catalog_sha256=_json_sha256(
            {
                "label": "mutation_delta",
                "tensor_sha256s": list(fixture_mutation_delta_sha256s),
            },
            domain=_TENSOR_DOMAIN,
        ),
        mutation_carrier_catalog_sha256=_json_sha256(
            {
                "label": "mutation_carrier",
                "tensor_sha256s": list(fixture_mutation_carrier_sha256s),
            },
            domain=_TENSOR_DOMAIN,
        ),
        attention_operator_sha256s=fixture_attention_operator_sha256s,
        attention_delta_sha256s=fixture_attention_delta_sha256s,
        mutation_delta_sha256s=fixture_mutation_delta_sha256s,
        mutation_carrier_sha256s=fixture_mutation_carrier_sha256s,
        final_carrier_sha256=fixture_mutation_carrier_sha256s[-1],
        terminal_normalized_output_sha256="7" * 64,
        dense_replay_literal_parity_passed=True,
        dense_replay_nll_literal_parity_passed=True,
        dense_replay_top1_parity_passed=True,
        all_routes_operator_delta_controls_passed=True,
        selection_prefix_nesting_passed=True,
        measurements=tuple(measurements),
    )
    return compile_attention_edge_fidelity(
        (capture,),
        source_kind="model_free_fixture",
        model_id=MODEL_ID,
        model_revision=PINNED_MODEL_REVISION,
        panel_id=PINNED_PANEL_ID,
        panel_file_sha256=PINNED_PANEL_FILE_SHA256,
        row_catalog_sha256=_json_sha256(
            [
                {
                    "row_ordinal": 0,
                    "row_catalog_id": capture.row_catalog_id,
                    "family_id": capture.family_id,
                }
            ],
            domain=_ATLAS_CATALOG_DOMAIN,
        ),
        tokenization_catalog_sha256=_json_sha256(
            [{"row_ordinal": 0, "tokenization_sha256": capture.tokenization_sha256}],
            domain=_TOKENIZATION_CATALOG_DOMAIN,
        ),
        source_model_fingerprint_sha256=SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_sha256=(
            PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        atlas_report_sha256="a" * 64,
        atlas_file_sha256="c" * 64,
        atlas_implementation_bundle_sha256="b" * 64,
        exact_control_receipt_sha256=digest,
        source_commit_sha="0" * 40,
        pre_execution_source_state_sha256="d" * 64,
        post_execution_source_state_sha256="e" * 64,
        clean_worktree_observed=False,
        committed_source_identity_passed=False,
        post_execution_source_recheck_passed=False,
    )


def validate_attention_edge_fidelity(
    report: AttentionEdgeFidelityReport,
) -> AttentionEdgeFidelityReport:
    if type(report) is not AttentionEdgeFidelityReport:
        raise TypeError("attention fidelity must be an exact typed report")
    replay = AttentionEdgeFidelityReport.from_dict(report.to_dict())
    if replay != report:
        raise ValueError("attention fidelity typed replay drifted")
    return report


def parse_attention_edge_fidelity_json(
    source: str | bytes,
) -> AttentionEdgeFidelityReport:
    value = _strict_json_object(source, label="attention edge fidelity")
    return validate_attention_edge_fidelity(
        AttentionEdgeFidelityReport.from_dict(value)
    )


def _load_report_path(path: Path) -> AttentionEdgeFidelityReport:
    descriptor: int | None = None
    primary: BaseException | None = None
    chunks: list[bytes] = []
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        initial = os.fstat(descriptor)
        named_initial = path.lstat()
        if (
            not os.path.samestat(initial, named_initial)
            or not stat.S_ISREG(initial.st_mode)
            or stat.S_IMODE(initial.st_mode) != 0o600
            or initial.st_nlink != 1
        ):
            raise ValueError("attention fidelity file identity is unsafe")
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
            raise ValueError("attention fidelity file changed during read")
    except OSError as error:
        primary = ValueError("attention fidelity file is unavailable or unsafe")
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
                    "attention fidelity descriptor close failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    return parse_attention_edge_fidelity_json(b"".join(chunks))


def load_attention_edge_fidelity(
    source: Path | str = DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT,
) -> AttentionEdgeFidelityReport:
    path = Path(source)
    if path.suffix != ".json":
        raise ValueError("attention fidelity path must use .json")
    return _load_report_path(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_canonical_fidelity_output(path: Path) -> bool:
    return path.resolve(strict=False) == (
        Path.cwd() / DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT
    ).resolve(strict=False)


def write_attention_edge_fidelity(
    report: AttentionEdgeFidelityReport,
    destination: Path | str = DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT,
) -> AttentionEdgeFidelityReport:
    validate_attention_edge_fidelity(report)
    target = Path(destination)
    if target.suffix != ".json":
        raise ValueError("attention fidelity output must use .json")
    if _is_canonical_fidelity_output(target) and (
        report.source_kind != "model_capture"
        or report.row_count != 60
        or report.full_atlas_1080_identity_passed is not True
    ):
        raise ValueError(
            "canonical A fidelity output accepts only the full 60-row "
            "model_capture"
        )
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
                raise RuntimeError("attention fidelity descriptor identity drifted")
        if _load_report_path(temporary) != report:
            raise RuntimeError("attention fidelity changed on strict temp reload")
        current = temporary.lstat()
        if temporary_stat is None or not os.path.samestat(temporary_stat, current):
            raise RuntimeError("attention fidelity temporary identity drifted")
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite an attention fidelity report"
            ) from error
        target_linked = True
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 2
        ):
            raise RuntimeError("attention fidelity hardlink identity drifted")
        temporary.unlink()
        temporary_exists = False
        _fsync_directory(target.parent)
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 1
        ):
            raise RuntimeError("attention fidelity publication identity drifted")
        canonical = _load_report_path(target)
        if canonical != report:
            raise RuntimeError("attention fidelity changed on canonical reload")
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
                    "attention fidelity target rollback failed: "
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
                    "attention fidelity temporary cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


def build_attention_edge_fidelity_preflight(
    config: AttentionEdgeFidelityLiveConfig = AttentionEdgeFidelityLiveConfig(),
) -> dict[str, object]:
    (
        generator_hash,
        executor_hash,
        compiler_hash,
        core_hash,
        atlas_hash,
        fidelity_kernel_hash,
        fidelity_hash,
        bundle_hash,
    ) = _implementation_provenance()
    return {
        "schema": ATTENTION_EDGE_FIDELITY_SCHEMA,
        "format_version": ATTENTION_EDGE_FIDELITY_FORMAT_VERSION,
        "abi": ATTENTION_EDGE_FIDELITY_ABI,
        "role": ATTENTION_EDGE_FIDELITY_ROLE,
        "model_id": MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "panel_id": PINNED_PANEL_ID,
        "panel_file_sha256": PINNED_PANEL_FILE_SHA256,
        "panel_path": str(config.panel_path),
        "atlas_path": str(config.atlas_path),
        "pinned_atlas_report_sha256": (
            PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256
        ),
        "pinned_atlas_file_sha256": PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256,
        "output": str(config.output),
        "k_values": list(ATTENTION_EDGE_FIDELITY_K_VALUES),
        "primary_rankers": list(ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS),
        "diagnostic_rankers": list(ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS),
        "variants": list(ATTENTION_EDGE_FIDELITY_VARIANTS),
        "random_salts": list(ATTENTION_EDGE_FIDELITY_RANDOM_SALTS),
        "random_seed_id": ATTENTION_EDGE_FIDELITY_RANDOM_SEED_ID,
        "condition_count": len(_conditions()),
        "matched_batch_size": _matched_batch_size(),
        "paired_batch53_all_routes_control_required": True,
        "ordinal_matched_control_metric_reference": (
            "propagated_and_endpoint_surfaces"
        ),
        "native_teacher_metric_reference": "local_operator_and_postnorm_delta",
        "matched_control_nll_max_abs_delta_bound": (
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_ABS_NLL_DELTA
        ),
        "matched_control_centered_logit_nrmse_bound": (
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_CENTERED_LOGIT_NRMSE
        ),
        "matched_control_final_carrier_nrmse_bound": (
            ATTENTION_EDGE_FIDELITY_MATCHED_MAX_FINAL_CARRIER_NRMSE
        ),
        "global_k_across_heads_and_keys": True,
        "fisher_ranker_is_same_row_target_gradient_oracle": True,
        "selection_masks_frozen_before_intervention": True,
        "qkv_and_probabilities_recomputed_on_sparse_state": True,
        "all_attention_layers_intervened_simultaneously": True,
        "empty_renormalized_heads_zero_filled": True,
        "all_routes_literal_control_required": True,
        "atlas_fisher_k8_and_extraction_identity_required": True,
        "clean_committed_worktree_required_for_live_execution": True,
        "post_execution_source_recheck_required": True,
        "attention_generator_implementation_sha256": generator_hash,
        "joint_executor_implementation_sha256": executor_hash,
        "joint_compiler_implementation_sha256": compiler_hash,
        "attention_edge_core_implementation_sha256": core_hash,
        "attention_edge_atlas_implementation_sha256": atlas_hash,
        "attention_edge_fidelity_kernel_implementation_sha256": (
            fidelity_kernel_hash
        ),
        "attention_edge_fidelity_implementation_sha256": fidelity_hash,
        "implementation_bundle_sha256": bundle_hash,
        "model_loaded": False,
        "atlas_opened": False,
        "panel_opened": False,
        "live_execution_performed": False,
        "fresh_panel_used": False,
        "compression_qualification_claimed": False,
    }


def run_attention_edge_fidelity(
    config: AttentionEdgeFidelityLiveConfig = AttentionEdgeFidelityLiveConfig(),
    *,
    row_limit: int | None = None,
) -> AttentionEdgeFidelityReport:
    return AttentionEdgeFidelityRuntime.load(config).capture(row_limit=row_limit)


def _compact_summary(report: AttentionEdgeFidelityReport) -> dict[str, object]:
    fisher_k8 = next(
        cell
        for cell in report.cells
        if cell.condition
        == AttentionEdgeFidelityCondition(
            "fisher", 8, "unnormalized_contribution"
        )
    )
    return {
        "schema": report.schema,
        "source_kind": report.source_kind,
        "role": report.role,
        "report_sha256": report.report_sha256,
        "row_count": report.row_count,
        "condition_count": len(report.cells),
        "matched_batch_control": {
            "batch_size": _matched_batch_size(),
            "maximum_native_to_matched_valid_logit_error": max(
                receipt.native_to_matched_valid_logit_max_abs_error
                for receipt in report.row_receipts
            ),
            "maximum_native_to_matched_nll_delta": max(
                receipt.native_to_matched_nll_max_abs_delta
                for receipt in report.row_receipts
            ),
            "maximum_native_to_matched_centered_logit_nrmse": max(
                receipt.native_to_matched_centered_logit_max_nrmse
                for receipt in report.row_receipts
            ),
            "maximum_native_to_matched_final_carrier_nrmse": max(
                receipt.native_to_matched_final_carrier_max_nrmse
                for receipt in report.row_receipts
            ),
        },
        "fisher_k8_unnormalized": {
            "operator_nrmse": fisher_k8.operator_nrmse,
            "operator_cosine": fisher_k8.operator_cosine,
            "final_carrier_nrmse": fisher_k8.final_carrier_nrmse,
            "final_carrier_cosine": fisher_k8.final_carrier_cosine,
            "terminal_nrmse": fisher_k8.terminal_nrmse,
            "terminal_cosine": fisher_k8.terminal_cosine,
            "valid_logit_nrmse": fisher_k8.valid_logit_nrmse,
            "valid_logit_cosine": fisher_k8.valid_logit_cosine,
            "mean_nll_delta": fisher_k8.mean_nll_delta,
            "top1_retention_rate": fisher_k8.top1_retention_rate,
            "active_head_query_fraction": (
                fisher_k8.active_head_query_fraction
            ),
        },
        "fresh_panel_used": report.fresh_panel_used,
        "compression_qualification_claimed": (
            report.compression_qualification_claimed
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the retrospective attention-edge fidelity curve"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true")
    modes.add_argument("--run-live", action="store_true")
    modes.add_argument("--smoke-one-row", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATTENTION_EDGE_ATLAS_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = AttentionEdgeFidelityLiveConfig(
        atlas_path=args.atlas,
        cache_dir=args.cache_dir,
        output=args.output,
    )
    if (
        args.write
        and _is_canonical_fidelity_output(config.output)
        and (args.demo or args.smoke_one_row)
    ):
        raise ValueError(
            "demo and one-row smoke writes require an explicit noncanonical "
            "output"
        )
    if args.write:
        try:
            config.output.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                "refusing to run before an existing fidelity output"
            )
    if args.demo:
        report = build_deterministic_attention_edge_fidelity_fixture()
    elif args.run_live or args.smoke_one_row:
        report = run_attention_edge_fidelity(
            config, row_limit=1 if args.smoke_one_row else None
        )
    else:
        print(
            json.dumps(
                build_attention_edge_fidelity_preflight(config),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.write:
        report = write_attention_edge_fidelity(report, config.output)
    print(json.dumps(_compact_summary(report), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
