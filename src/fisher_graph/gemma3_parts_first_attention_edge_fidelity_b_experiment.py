"""Prospective-B fidelity test for the frozen attention-edge selector.

The B runner is a consumer of an already-issued setup.  Its first operation is
``check_setup``; only after that succeeds may it import a model loader, create a
tokenizer batch, or execute the model.  The selection mask is determined only
by allowed route geometry and the authenticated retrospective-A motif means.
No B label, choice metadata, activation, probability, or gradient is accepted
by the selector API.

For each B prompt the runner freezes all eighteen masks, then performs a dense
teacher execution, a separate batch-one all-route replay, and a separate
batch-one sparse replay.  The all-route replay must be literally identical at
every mutation and endpoint.  The sparse replay uses a global K=8 cap across
``(query head, key)`` and zero-fills omitted routes without renormalization
while later Q/K/V and probabilities are recomputed on the evolving sparse
state.

The report is deliberately narrow.  Passing all six preregistered gates says
only that this frozen, gradient-free A-motif procedure preserved the measured
behavior on this one B panel.  It does not establish semantics, selection
optimality, compression, parameter/MAC/latency savings, or generalization.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Callable, Final, Protocol, TypeVar
import weakref

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, SequenceContext
from .compiler.calibration import CalibrationBatch
from . import gemma3_parts_first_attention_edge_fidelity as fidelity_kernel
from .gemma3_attention_generator import Gemma3AttentionGenerator
from .gemma3_parts_first_attention_edge_fidelity_b_setup import (
    A_MOTIF_SCORE_CATALOG_SHA256,
    BOOTSTRAP_L95_ZERO_BASED_INDEX as _SETUP_BOOTSTRAP_L95_INDEX,
    BOOTSTRAP_RESAMPLE_COUNT as _SETUP_BOOTSTRAP_RESAMPLE_COUNT,
    BOOTSTRAP_SEED as _SETUP_BOOTSTRAP_SEED,
    BOOTSTRAP_U95_ZERO_BASED_INDEX as _SETUP_BOOTSTRAP_U95_INDEX,
    B_PANEL_FILE_SHA256,
    B_PANEL_ID,
    B_PANEL_SEMANTIC_SHA256,
    B_METADATA_CATALOG_SHA256,
    DEFAULT_B_REPORT_OUTPUT,
    DEFAULT_OUTPUT as DEFAULT_B_SETUP,
    DISTANCE_BINS,
    EMPIRICAL_P90_ZERO_BASED_INDEX as _SETUP_EMPIRICAL_P90_INDEX,
    FAMILY_IDS,
    PRIMARY_K,
    SETUP_ID,
    WILSON_Z_TWO_SIDED_95 as _SETUP_WILSON_Z95,
    bootstrap_resample_indices as _setup_bootstrap_resample_indices,
    check_setup,
    load_fidelity_b_panel,
    validate_setup,
    wilson_top1_lower_bound as _setup_wilson_top1_lower_bound,
)
from .gemma3_parts_first_supermode_protocol import (
    MODEL_ID,
    SOURCE_MODEL_FINGERPRINT_SHA256,
)
from .gemma3_parts_first_supermodes import (
    ATTENTION_HEAD_COUNT,
    GEMMA3_LAYER_COUNT,
)


ATTENTION_EDGE_FIDELITY_B_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edge_fidelity_b.v1"
)
ATTENTION_EDGE_FIDELITY_B_FORMAT_VERSION: Final = 1
ATTENTION_EDGE_FIDELITY_B_ABI: Final = (
    "A_atlas_motif_mean_global_k8_cap_zero_fill_dynamic_all_layer_replay.v1"
)
ATTENTION_EDGE_FIDELITY_B_ROLE: Final = "B_fresh_execution_confirmation"
ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS: Final = 128
PINNED_MODEL_REVISION: Final = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)

BOOTSTRAP_SEED: Final = _SETUP_BOOTSTRAP_SEED
BOOTSTRAP_RESAMPLE_COUNT: Final = _SETUP_BOOTSTRAP_RESAMPLE_COUNT
BOOTSTRAP_PRNG_ID: Final = "sha256_counter_modulo.v1"
BOOTSTRAP_U95_INDEX: Final = _SETUP_BOOTSTRAP_U95_INDEX
BOOTSTRAP_L95_INDEX: Final = _SETUP_BOOTSTRAP_L95_INDEX
EMPIRICAL_P90_INDEX: Final = _SETUP_EMPIRICAL_P90_INDEX
WILSON_Z_TWO_SIDED_95: Final = _SETUP_WILSON_Z95
EXPECTED_B_ROW_FAMILIES: Final = tuple(
    family for family in FAMILY_IDS for _ in range(10)
)

GATE_THRESHOLDS: Final = {
    "macro_family_bootstrap_mean_absolute_delta_NLL_U95_max": 0.05,
    "empirical_absolute_delta_NLL_p90_max": 0.25,
    "full_vocab_top1_agreement_Wilson_L95_min": 0.90,
    "macro_family_bootstrap_terminal_cosine_L95_min": 0.995,
    "macro_family_bootstrap_centered_logit_NRMSE_U95_max": 0.05,
    "macro_family_bootstrap_teacher_operator_NRMSE_U95_max": 0.15,
}

_REPORT_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-report:v1\0"
_ROW_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-row:v1\0"
_SELECTION_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-selection:v1\0"
_TOKENIZATION_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-tokenization:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-tensor:v1\0"
_BOOTSTRAP_VALUE_DOMAIN = (
    b"fisher_graph:attention-edge-fidelity-b-bootstrap-values:v1\0"
)
_EXACT_CONTROL_DOMAIN = (
    b"fisher_graph:attention-edge-fidelity-b-exact-control:v1\0"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_EXPECTED_EVENT_IDS: Final = (
    "lexical.ingress",
    *(f"mutation.{index}" for index in range(1, 37)),
    "terminal.rmsnorm",
    "lexical.egress",
)


__all__ = [
    "ATTENTION_EDGE_FIDELITY_B_ABI",
    "ATTENTION_EDGE_FIDELITY_B_FORMAT_VERSION",
    "ATTENTION_EDGE_FIDELITY_B_ROLE",
    "ATTENTION_EDGE_FIDELITY_B_SCHEMA",
    "AttentionEdgeFidelityBLiveConfig",
    "AttentionEdgeFidelityBRowCapture",
    "AttentionEdgeFidelityBRuntime",
    "BOOTSTRAP_L95_INDEX",
    "BOOTSTRAP_PRNG_ID",
    "BOOTSTRAP_RESAMPLE_COUNT",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_U95_INDEX",
    "EMPIRICAL_P90_INDEX",
    "GATE_THRESHOLDS",
    "WILSON_Z_TWO_SIDED_95",
    "bootstrap_resample_indices",
    "build_attention_edge_fidelity_b_preflight",
    "build_attention_edge_fidelity_b_report",
    "build_deterministic_attention_edge_fidelity_b_fixture",
    "load_attention_edge_fidelity_b",
    "load_attention_edge_fidelity_b_content",
    "main",
    "parse_attention_edge_fidelity_b_json",
    "run_attention_edge_fidelity_b",
    "select_frozen_a_motif_mask",
    "validate_attention_edge_fidelity_b_content",
    "write_attention_edge_fidelity_b",
]


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


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} field inventory drifted: missing={missing}, extra={extra}")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_commit(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git object ID")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
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


def _strict_json_object(source: str | bytes, *, label: str) -> dict[str, object]:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        parsed = json.loads(
            source,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _lstat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _open_strict_parent_directory(
    path: Path,
    *,
    label: str,
    create: bool,
) -> tuple[Path, int, os.stat_result]:
    """Open every ancestor with ``O_NOFOLLOW`` and return the pinned parent."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("strict directory-relative path access is unavailable")
    absolute = Path(os.path.abspath(path))
    if not absolute.name or absolute.name in {".", ".."}:
        raise ValueError(f"{label} name is invalid")
    parent = absolute.parent
    anchor = parent.anchor
    if not anchor:
        raise ValueError(f"{label} path is not absolute after normalization")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_descriptor = os.open(anchor, flags)
    try:
        for component in parent.parts[1:]:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=directory_descriptor)
                except FileExistsError:
                    pass
            try:
                child_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{label} parent does not exist: {parent}"
                ) from None
            except OSError as error:
                raise ValueError(
                    f"{label} ancestors must be real non-symlink directories"
                ) from error
            try:
                named = _lstat_at(directory_descriptor, component)
                opened = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(named.st_mode)
                    or stat.S_ISLNK(named.st_mode)
                    or not stat.S_ISDIR(opened.st_mode)
                    or _inode_identity(named) != _inode_identity(opened)
                ):
                    raise ValueError(
                        f"{label} ancestor changed during directory traversal"
                    )
            except BaseException:
                os.close(child_descriptor)
                raise
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        opened_parent = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(opened_parent.st_mode):
            raise ValueError(f"{label} parent is not a directory")
        return absolute, directory_descriptor, opened_parent
    except BaseException:
        os.close(directory_descriptor)
        raise


def _reopen_same_strict_parent(
    path: Path,
    *,
    label: str,
    expected_parent: os.stat_result,
) -> None:
    absolute, descriptor, observed_parent = _open_strict_parent_directory(
        path,
        label=label,
        create=False,
    )
    try:
        if absolute != path or _inode_identity(observed_parent) != _inode_identity(
            expected_parent
        ):
            raise ValueError(f"{label} parent changed during access")
    finally:
        os.close(descriptor)


_RegularConsumerResult = TypeVar("_RegularConsumerResult")


def _regular_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _consume_regular(
    path: Path,
    *,
    label: str,
    byte_cap: int,
    consumer: Callable[[bytes], _RegularConsumerResult],
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> tuple[_RegularConsumerResult, bytes]:
    """Read and consume one retained inode, then revalidate its public name."""

    absolute, parent_descriptor, opened_parent = _open_strict_parent_directory(
        path,
        label=label,
        create=False,
    )
    try:
        try:
            before = _lstat_at(parent_descriptor, absolute.name)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{label} does not exist: {absolute}"
            ) from None
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if before.st_size <= 0 or before.st_size > byte_cap:
            raise ValueError(f"{label} violates its byte cap")
        if (
            required_mode is not None
            and stat.S_IMODE(before.st_mode) != required_mode
        ):
            raise PermissionError(f"{label} must have mode {required_mode:o}")
        if required_nlink is not None and before.st_nlink != required_nlink:
            raise ValueError(f"{label} must have link count {required_nlink}")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            opened = os.fstat(descriptor)
            identity = _regular_identity(before)
            if _regular_identity(opened) != identity:
                raise ValueError(f"{label} changed during open")
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"{label} was truncated during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError(f"{label} grew during read")
            raw = b"".join(chunks)
            opened_after_read = os.fstat(descriptor)
            if (
                required_mode is not None
                and stat.S_IMODE(opened_after_read.st_mode) != required_mode
            ):
                raise PermissionError(f"{label} mode changed while reading")
            if (
                required_nlink is not None
                and opened_after_read.st_nlink != required_nlink
            ):
                raise ValueError(f"{label} link count changed while reading")
            if _regular_identity(opened_after_read) != identity:
                raise ValueError(f"{label} changed while reading")
            result = consumer(raw)
            opened_after_consumer = os.fstat(descriptor)
            if _regular_identity(opened_after_consumer) != identity:
                raise ValueError(f"{label} changed while being consumed")
            os.lseek(descriptor, 0, os.SEEK_SET)
            verification_chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"{label} was truncated after consumption")
                verification_chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError(f"{label} grew after consumption")
            if b"".join(verification_chunks) != raw:
                raise ValueError(f"{label} bytes changed while being consumed")
            if _regular_identity(os.fstat(descriptor)) != identity:
                raise ValueError(f"{label} changed after consumption")
            after = _lstat_at(parent_descriptor, absolute.name)
            if _regular_identity(after) != identity:
                raise ValueError(f"{label} public path changed during consumption")
            _reopen_same_strict_parent(
                absolute,
                label=label,
                expected_parent=opened_parent,
            )
            return result, raw
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _read_regular(
    path: Path,
    *,
    label: str,
    byte_cap: int,
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> bytes:
    result, _ = _consume_regular(
        path,
        label=label,
        byte_cap=byte_cap,
        required_mode=required_mode,
        required_nlink=required_nlink,
        consumer=lambda raw: raw,
    )
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    metadata = {
        "dtype": str(tensor.dtype),
        "shape": [int(item) for item in tensor.shape],
    }
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + _canonical_json_bytes(metadata)
        + b"\0"
        + tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _distance_bin(distance: int) -> str:
    if type(distance) is not int or distance < 0:
        raise ValueError("causal route distance must be nonnegative")
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


def _row_family_sequence_sha256(values: Sequence[str]) -> str:
    return _digest(
        b"fisher_graph:attention-edge-fidelity-b-row-family-sequence:v1\0",
        [
            {"row_ordinal": ordinal, "family_id": family}
            for ordinal, family in enumerate(values)
        ],
    )


def _motif_scores(
    catalog: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int, str], float | None]:
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
        raise TypeError("A motif catalog must be a sequence")
    scores: dict[tuple[int, int, str], float | None] = {}
    for row in catalog:
        if not isinstance(row, Mapping):
            raise ValueError("A motif catalog row must be an object")
        layer = row.get("layer_ordinal")
        head = row.get("query_head_ordinal")
        distance_bin = row.get("distance_bin")
        supported = row.get("supported")
        mean_hex = row.get("mean_intrinsic_fisher_energy_hex")
        if (
            type(layer) is not int
            or not 0 <= layer < GEMMA3_LAYER_COUNT
            or type(head) is not int
            or not 0 <= head < ATTENTION_HEAD_COUNT
            or distance_bin not in DISTANCE_BINS
            or type(supported) is not bool
        ):
            raise ValueError("A motif catalog coordinate is invalid")
        key = (layer, head, str(distance_bin))
        if key in scores:
            raise ValueError("A motif catalog coordinate is duplicated")
        if supported:
            if not isinstance(mean_hex, str):
                raise ValueError("supported A motif omitted its mean score")
            try:
                score = float.fromhex(mean_hex)
            except ValueError as error:
                raise ValueError("A motif mean score encoding is invalid") from error
            if not math.isfinite(score) or score < 0.0:
                raise ValueError("A motif mean score must be finite and nonnegative")
            scores[key] = score
        else:
            if mean_hex is not None:
                raise ValueError("unsupported A motif unexpectedly has a score")
            scores[key] = None
    expected = {
        (layer, head, distance_bin)
        for layer in range(GEMMA3_LAYER_COUNT)
        for head in range(ATTENTION_HEAD_COUNT)
        for distance_bin in DISTANCE_BINS
    }
    if set(scores) != expected:
        raise ValueError("A motif score catalog is incomplete")
    return scores


def _allowed_route_geometry(
    sequence: SequenceContext, *, window_size: int | None
) -> Tensor:
    if sequence.batch_size != 1 or sequence.key_length != sequence.query_length:
        raise ValueError("prospective B selector requires one prefill self-attention row")
    query = torch.arange(sequence.query_length, dtype=torch.int64).view(
        1, sequence.query_length, 1
    )
    key = torch.arange(sequence.key_length, dtype=torch.int64).view(
        1, 1, sequence.key_length
    )
    allowed = (
        sequence.query_valid_mask.to(device="cpu")[:, :, None]
        & sequence.key_valid_mask.to(device="cpu")[:, None, :]
        & (key <= query)
    )
    if window_size is not None:
        if type(window_size) is not int or window_size <= 0:
            raise ValueError("attention window must be a positive integer")
        allowed &= query - key < window_size
    return allowed.contiguous()


def select_frozen_a_motif_mask(
    allowed_mask: Tensor,
    motif_catalog: Sequence[Mapping[str, object]],
    *,
    layer_ordinal: int,
    query_head_count: int = ATTENTION_HEAD_COUNT,
    k: int = PRIMARY_K,
) -> Tensor:
    """Select the B mask without accepting any B score or metadata input."""

    if (
        not isinstance(allowed_mask, Tensor)
        or allowed_mask.dtype is not torch.bool
        or allowed_mask.device.type != "cpu"
        or allowed_mask.ndim != 3
        or allowed_mask.shape[0] != 1
        or allowed_mask.shape[1] != allowed_mask.shape[2]
    ):
        raise ValueError("allowed route geometry must be CPU bool [1,Q,K]")
    if type(layer_ordinal) is not int or not 0 <= layer_ordinal < GEMMA3_LAYER_COUNT:
        raise ValueError("prospective B layer ordinal is invalid")
    if query_head_count != ATTENTION_HEAD_COUNT:
        raise ValueError("prospective B query-head count is frozen at four")
    if k != PRIMARY_K:
        raise ValueError("prospective B selector is frozen at global K=8")
    scores = _motif_scores(motif_catalog)
    queries = int(allowed_mask.shape[1])
    mask = torch.zeros(
        (1, query_head_count, queries, queries), dtype=torch.bool
    )
    for query in range(queries):
        keys = tuple(
            int(item)
            for item in torch.nonzero(
                allowed_mask[0, query], as_tuple=False
            ).reshape(-1).tolist()
        )
        candidates: list[tuple[float, int, int]] = []
        for head in range(query_head_count):
            for key in keys:
                score = scores[(layer_ordinal, head, _distance_bin(query - key))]
                if score is None:
                    raise ValueError(
                        "allowed B route requires an unsupported A motif cell"
                    )
                candidates.append((-score, head, key))
        candidates.sort()
        for _, head, key in candidates[: min(k, len(candidates))]:
            mask[0, head, query, key] = True
    if bool((mask & ~allowed_mask[:, None, :, :]).any()):
        raise AssertionError("prospective B selector emitted a forbidden route")
    return mask.contiguous()


def bootstrap_resample_indices(
    replicate_ordinal: int,
    family_ordinal: int,
    *,
    family_size: int = 10,
) -> tuple[int, ...]:
    """Return the preregistered paired within-family bootstrap indexes."""

    if (
        type(replicate_ordinal) is not int
        or not 0 <= replicate_ordinal < BOOTSTRAP_RESAMPLE_COUNT
        or type(family_ordinal) is not int
        or not 0 <= family_ordinal < len(FAMILY_IDS)
        or family_size != 10
    ):
        raise ValueError("bootstrap coordinates are outside the frozen design")
    return _setup_bootstrap_resample_indices(
        replicate_ordinal, family_ordinal
    )


def _wilson_lower(successes: int, total: int) -> float:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("Wilson counts are invalid")
    if total <= 0:
        raise ValueError("Wilson total must be positive")
    return _setup_wilson_top1_lower_bound(successes, total)


def _nearest_rank(values: Sequence[float], index: int) -> float:
    materialized = tuple(_finite(value, label="order statistic") for value in values)
    if type(index) is not int or not 0 <= index < len(materialized):
        raise ValueError("order-statistic index is invalid")
    return sorted(materialized)[index]


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityBRowCapture:
    row_ordinal: int
    family_id: str
    sequence_length: int
    tokenization_sha256: str
    selection_catalog_sha256: str
    available_route_count: int
    selected_route_count: int
    native_correct_target_nll: float
    sparse_correct_target_nll: float
    full_vocab_top1_agreement: bool
    terminal_supervised_vector_cosine: float
    centered_logit_nrmse: float
    dense_teacher_operator_nrmse: float
    dense_replay_literal_parity_passed: bool
    all_route_attention_operator_delta_parity_passed: bool
    mutation_boundary_literal_parity_passed: bool

    def __post_init__(self) -> None:
        _require_int(self.row_ordinal, label="row_ordinal")
        if self.family_id not in FAMILY_IDS:
            raise ValueError("B row family is outside the preregistered panel")
        _require_int(self.sequence_length, label="sequence_length", minimum=1)
        _require_sha256(self.tokenization_sha256, label="tokenization")
        _require_sha256(self.selection_catalog_sha256, label="selection catalog")
        available = _require_int(
            self.available_route_count, label="available_route_count", minimum=1
        )
        selected = _require_int(
            self.selected_route_count, label="selected_route_count", minimum=1
        )
        if selected > available:
            raise ValueError("selected B routes exceed available routes")
        _finite(self.native_correct_target_nll, label="native NLL")
        _finite(self.sparse_correct_target_nll, label="sparse NLL")
        terminal = _finite(
            self.terminal_supervised_vector_cosine, label="terminal cosine"
        )
        if not -1.0 <= terminal <= 1.0:
            raise ValueError("terminal cosine is outside [-1,1]")
        _nonnegative(self.centered_logit_nrmse, label="centered logit NRMSE")
        _nonnegative(
            self.dense_teacher_operator_nrmse,
            label="dense teacher operator NRMSE",
        )
        for label, flag in (
            ("full_vocab_top1_agreement", self.full_vocab_top1_agreement),
            ("dense replay parity", self.dense_replay_literal_parity_passed),
            (
                "all-route attention parity",
                self.all_route_attention_operator_delta_parity_passed,
            ),
            ("mutation parity", self.mutation_boundary_literal_parity_passed),
        ):
            if type(flag) is not bool:
                raise TypeError(f"{label} must be bool")
        if not (
            self.dense_replay_literal_parity_passed
            and self.all_route_attention_operator_delta_parity_passed
            and self.mutation_boundary_literal_parity_passed
        ):
            raise ValueError("B row all-route control did not close")

    @property
    def delta_nll(self) -> float:
        return self.sparse_correct_target_nll - self.native_correct_target_nll

    @property
    def absolute_delta_nll(self) -> float:
        return abs(self.delta_nll)

    def _payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "family_id": self.family_id,
            "sequence_length": self.sequence_length,
            "tokenization_sha256": self.tokenization_sha256,
            "selection_catalog_sha256": self.selection_catalog_sha256,
            "available_route_count": self.available_route_count,
            "selected_route_count": self.selected_route_count,
            "native_correct_target_nll": self.native_correct_target_nll,
            "sparse_correct_target_nll": self.sparse_correct_target_nll,
            "delta_nll": self.delta_nll,
            "absolute_delta_nll": self.absolute_delta_nll,
            "full_vocab_top1_agreement": self.full_vocab_top1_agreement,
            "terminal_supervised_vector_cosine": (
                self.terminal_supervised_vector_cosine
            ),
            "centered_logit_nrmse": self.centered_logit_nrmse,
            "dense_teacher_operator_nrmse": (
                self.dense_teacher_operator_nrmse
            ),
            "dense_replay_literal_parity_passed": (
                self.dense_replay_literal_parity_passed
            ),
            "all_route_attention_operator_delta_parity_passed": (
                self.all_route_attention_operator_delta_parity_passed
            ),
            "mutation_boundary_literal_parity_passed": (
                self.mutation_boundary_literal_parity_passed
            ),
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        return {**payload, "row_sha256": _digest(_ROW_DOMAIN, payload)}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "AttentionEdgeFidelityBRowCapture":
        expected = {
            "row_ordinal",
            "family_id",
            "sequence_length",
            "tokenization_sha256",
            "selection_catalog_sha256",
            "available_route_count",
            "selected_route_count",
            "native_correct_target_nll",
            "sparse_correct_target_nll",
            "delta_nll",
            "absolute_delta_nll",
            "full_vocab_top1_agreement",
            "terminal_supervised_vector_cosine",
            "centered_logit_nrmse",
            "dense_teacher_operator_nrmse",
            "dense_replay_literal_parity_passed",
            "all_route_attention_operator_delta_parity_passed",
            "mutation_boundary_literal_parity_passed",
            "row_sha256",
        }
        _exact_keys(value, expected, label="B fidelity row")
        result = cls(
            row_ordinal=_require_int(value["row_ordinal"], label="row_ordinal"),
            family_id=str(value["family_id"]),
            sequence_length=_require_int(
                value["sequence_length"], label="sequence_length", minimum=1
            ),
            tokenization_sha256=_require_sha256(
                value["tokenization_sha256"], label="tokenization"
            ),
            selection_catalog_sha256=_require_sha256(
                value["selection_catalog_sha256"], label="selection catalog"
            ),
            available_route_count=_require_int(
                value["available_route_count"],
                label="available_route_count",
                minimum=1,
            ),
            selected_route_count=_require_int(
                value["selected_route_count"],
                label="selected_route_count",
                minimum=1,
            ),
            native_correct_target_nll=_finite(
                value["native_correct_target_nll"], label="native NLL"
            ),
            sparse_correct_target_nll=_finite(
                value["sparse_correct_target_nll"], label="sparse NLL"
            ),
            full_vocab_top1_agreement=value["full_vocab_top1_agreement"],  # type: ignore[arg-type]
            terminal_supervised_vector_cosine=_finite(
                value["terminal_supervised_vector_cosine"],
                label="terminal cosine",
            ),
            centered_logit_nrmse=_nonnegative(
                value["centered_logit_nrmse"], label="centered logit NRMSE"
            ),
            dense_teacher_operator_nrmse=_nonnegative(
                value["dense_teacher_operator_nrmse"],
                label="teacher operator NRMSE",
            ),
            dense_replay_literal_parity_passed=value[
                "dense_replay_literal_parity_passed"
            ],  # type: ignore[arg-type]
            all_route_attention_operator_delta_parity_passed=value[
                "all_route_attention_operator_delta_parity_passed"
            ],  # type: ignore[arg-type]
            mutation_boundary_literal_parity_passed=value[
                "mutation_boundary_literal_parity_passed"
            ],  # type: ignore[arg-type]
        )
        if _finite(value["delta_nll"], label="delta NLL") != result.delta_nll:
            raise ValueError("B row delta NLL algebra drifted")
        if (
            _nonnegative(value["absolute_delta_nll"], label="absolute delta NLL")
            != result.absolute_delta_nll
        ):
            raise ValueError("B row absolute delta NLL algebra drifted")
        if value["row_sha256"] != _digest(_ROW_DOMAIN, result._payload()):
            raise ValueError("B row semantic digest mismatch")
        return result


def _family_rows(
    rows: Sequence[AttentionEdgeFidelityBRowCapture],
) -> dict[str, tuple[AttentionEdgeFidelityBRowCapture, ...]]:
    result: dict[str, tuple[AttentionEdgeFidelityBRowCapture, ...]] = {}
    for family in FAMILY_IDS:
        members = tuple(row for row in rows if row.family_id == family)
        if len(members) != 10:
            raise ValueError("B report must contain exactly ten rows per family")
        result[family] = members
    return result


def _family_ledgers(
    rows: Sequence[AttentionEdgeFidelityBRowCapture],
) -> list[dict[str, object]]:
    groups = _family_rows(rows)
    result: list[dict[str, object]] = []
    for family in FAMILY_IDS:
        values = groups[family]
        result.append(
            {
                "family_id": family,
                "row_count": len(values),
                "mean_absolute_delta_nll": math.fsum(
                    row.absolute_delta_nll for row in values
                )
                / len(values),
                "top1_agreement_count": sum(
                    int(row.full_vocab_top1_agreement) for row in values
                ),
                "top1_agreement_rate": sum(
                    int(row.full_vocab_top1_agreement) for row in values
                )
                / len(values),
                "mean_terminal_cosine": math.fsum(
                    row.terminal_supervised_vector_cosine for row in values
                )
                / len(values),
                "mean_centered_logit_nrmse": math.fsum(
                    row.centered_logit_nrmse for row in values
                )
                / len(values),
                "mean_dense_teacher_operator_nrmse": math.fsum(
                    row.dense_teacher_operator_nrmse for row in values
                )
                / len(values),
            }
        )
    return result


def _paired_bootstrap_catalog(
    rows: Sequence[AttentionEdgeFidelityBRowCapture],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Compute all four statistics from the same resampled whole examples."""

    groups = _family_rows(rows)
    absolute_values: list[float] = []
    terminal_values: list[float] = []
    centered_values: list[float] = []
    teacher_values: list[float] = []
    for replicate in range(BOOTSTRAP_RESAMPLE_COUNT):
        absolute_family: list[float] = []
        terminal_family: list[float] = []
        centered_family: list[float] = []
        teacher_family: list[float] = []
        for family_ordinal, family in enumerate(FAMILY_IDS):
            members = groups[family]
            sample = tuple(
                members[index]
                for index in bootstrap_resample_indices(
                    replicate, family_ordinal
                )
            )
            denominator = len(sample)
            absolute_family.append(
                math.fsum(row.absolute_delta_nll for row in sample) / denominator
            )
            terminal_family.append(
                math.fsum(
                    row.terminal_supervised_vector_cosine for row in sample
                )
                / denominator
            )
            centered_family.append(
                math.fsum(row.centered_logit_nrmse for row in sample)
                / denominator
            )
            teacher_family.append(
                math.fsum(row.dense_teacher_operator_nrmse for row in sample)
                / denominator
            )
        family_count = len(FAMILY_IDS)
        absolute_values.append(math.fsum(absolute_family) / family_count)
        terminal_values.append(math.fsum(terminal_family) / family_count)
        centered_values.append(math.fsum(centered_family) / family_count)
        teacher_values.append(math.fsum(teacher_family) / family_count)
    return (
        tuple(absolute_values),
        tuple(terminal_values),
        tuple(centered_values),
        tuple(teacher_values),
    )


def _aggregate_statistics(
    rows: Sequence[AttentionEdgeFidelityBRowCapture],
) -> dict[str, object]:
    if len(rows) != 60:
        raise ValueError("prospective B aggregation requires all sixty rows")
    absolute = tuple(row.absolute_delta_nll for row in rows)
    (
        absolute_bootstrap,
        terminal_bootstrap,
        centered_bootstrap,
        teacher_bootstrap,
    ) = _paired_bootstrap_catalog(
        rows
    )
    bootstrap_payload = {
        "absolute_delta_nll": [value.hex() for value in absolute_bootstrap],
        "terminal_cosine": [value.hex() for value in terminal_bootstrap],
        "centered_logit_nrmse": [value.hex() for value in centered_bootstrap],
        "teacher_operator_nrmse": [value.hex() for value in teacher_bootstrap],
    }
    agreements = sum(int(row.full_vocab_top1_agreement) for row in rows)
    return {
        "bootstrap_prng_id": BOOTSTRAP_PRNG_ID,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resample_count": BOOTSTRAP_RESAMPLE_COUNT,
        "bootstrap_resample_indexing": (
            "replicate_0_to_9999_family_in_FAMILY_IDS_order_draw_0_to_9"
        ),
        "bootstrap_U95_order_statistic_index": BOOTSTRAP_U95_INDEX,
        "bootstrap_L95_order_statistic_index": BOOTSTRAP_L95_INDEX,
        "empirical_p90_order_statistic_index": EMPIRICAL_P90_INDEX,
        "Wilson_confidence": "two_sided_95_percent_lower_endpoint",
        "Wilson_z": WILSON_Z_TWO_SIDED_95,
        "Wilson_continuity_correction": False,
        "Wilson_formula": (
            "(p+z^2/(2n)-z*sqrt(p*(1-p)/n+z^2/(4n^2)))"
            "/(1+z^2/n)"
        ),
        "bootstrap_value_catalog_sha256": _digest(
            _BOOTSTRAP_VALUE_DOMAIN, bootstrap_payload
        ),
        "macro_family_bootstrap_mean_absolute_delta_NLL_U95": _nearest_rank(
            absolute_bootstrap, BOOTSTRAP_U95_INDEX
        ),
        "empirical_absolute_delta_NLL_p90": _nearest_rank(
            absolute, EMPIRICAL_P90_INDEX
        ),
        "full_vocab_top1_agreement_count": agreements,
        "full_vocab_top1_agreement_rate": agreements / len(rows),
        "full_vocab_top1_agreement_Wilson_L95": _wilson_lower(
            agreements, len(rows)
        ),
        "macro_family_bootstrap_terminal_cosine_L95": _nearest_rank(
            terminal_bootstrap, BOOTSTRAP_L95_INDEX
        ),
        "macro_family_bootstrap_centered_logit_NRMSE_U95": _nearest_rank(
            centered_bootstrap, BOOTSTRAP_U95_INDEX
        ),
        "macro_family_bootstrap_teacher_operator_NRMSE_U95": _nearest_rank(
            teacher_bootstrap, BOOTSTRAP_U95_INDEX
        ),
    }


def _gate_outcomes(aggregates: Mapping[str, object]) -> list[dict[str, object]]:
    specifications = (
        (
            "macro_family_bootstrap_mean_absolute_delta_NLL_U95_max",
            "macro_family_bootstrap_mean_absolute_delta_NLL_U95",
            "max",
        ),
        (
            "empirical_absolute_delta_NLL_p90_max",
            "empirical_absolute_delta_NLL_p90",
            "max",
        ),
        (
            "full_vocab_top1_agreement_Wilson_L95_min",
            "full_vocab_top1_agreement_Wilson_L95",
            "min",
        ),
        (
            "macro_family_bootstrap_terminal_cosine_L95_min",
            "macro_family_bootstrap_terminal_cosine_L95",
            "min",
        ),
        (
            "macro_family_bootstrap_centered_logit_NRMSE_U95_max",
            "macro_family_bootstrap_centered_logit_NRMSE_U95",
            "max",
        ),
        (
            "macro_family_bootstrap_teacher_operator_NRMSE_U95_max",
            "macro_family_bootstrap_teacher_operator_NRMSE_U95",
            "max",
        ),
    )
    result: list[dict[str, object]] = []
    for gate_id, metric, comparison in specifications:
        observed = _finite(aggregates.get(metric), label=metric)
        threshold = GATE_THRESHOLDS[gate_id]
        passed = observed <= threshold if comparison == "max" else observed >= threshold
        result.append(
            {
                "gate_id": gate_id,
                "metric_id": metric,
                "comparison": comparison,
                "threshold": threshold,
                "observed": observed,
                "equality_at_threshold_passes": True,
                "passed": passed,
            }
        )
    return result


def _claim_boundary() -> dict[str, object]:
    return {
        "passing_supports": (
            "the_frozen_gradient_free_A_motif_global_K8_cap_zero_fill_procedure_"
            "preserved_behavior_on_this_B_panel"
        ),
        "bounded_to_this_B_panel": True,
        "semantic_interpretation_claimed": False,
        "selection_optimality_claimed": False,
        "compression_qualified": False,
        "parameter_savings_claimed": False,
        "MAC_savings_claimed": False,
        "latency_savings_claimed": False,
        "input_only_generalization_beyond_this_panel_claimed": False,
    }


def _protocol_receipt() -> dict[str, object]:
    return {
        "selector_id": "A_atlas_motif_mean_intrinsic_fisher_energy.v1",
        "score_coordinates": [
            "layer_ordinal",
            "query_head_ordinal",
            "causal_distance_bin",
        ],
        "global_across_query_head_and_key": True,
        "A_motif_score_catalog_sha256": A_MOTIF_SCORE_CATALOG_SHA256,
        "maximum_routes_per_layer_query": PRIMARY_K,
        "selected_route_count_formula": (
            "min(8,allowed_global_head_key_route_count)"
        ),
        "query_ordinal_zero_allowed_route_count": ATTENTION_HEAD_COUNT,
        "query_ordinal_zero_selected_route_count": ATTENTION_HEAD_COUNT,
        "tie_break": [
            "descending_A_motif_mean_score",
            "ascending_query_head_ordinal",
            "ascending_key_ordinal",
        ],
        "omitted_route_semantics": "zero_fill_without_renormalization",
        "later_QKV_and_probabilities_recomputed_on_sparse_state": True,
        "all_attention_layers_intervened_simultaneously": True,
        "dense_control_and_sparse_replay_use_separate_calls": True,
        "dense_control_execution_batch_size": 1,
        "sparse_execution_batch_size": 1,
        "cross_batch_literal_parity_assumed": False,
        "dense_teacher_operator_NRMSE_semantics": (
            "teacher_forced_local_dense_baseline_input_P_V_apply_frozen_mask_"
            "zero_fill_compare_operator_over_valid_queries"
        ),
        "dynamic_sparse_operator_drift_used_for_teacher_gate": False,
        "dynamic_sparse_propagation_measured_by": [
            "terminal_supervised_vector_cosine",
            "centered_logit_NRMSE",
            "delta_NLL",
            "full_vocab_top1_agreement",
        ],
        "centered_logit_NRMSE_semantics": (
            "norm((sparse-mean_sparse)-(native-mean_native))_divided_by_"
            "norm(native-mean_native)_at_supervised_query"
        ),
        "all_B_masks_frozen_before_correct_target_metric_access": True,
        "B_example_id_used_for_selection": False,
        "B_family_id_used_for_selection": False,
        "B_choice_order_used_for_selection": False,
        "B_distractors_used_for_selection": False,
        "B_correct_choice_or_target_used_for_selection": False,
        "B_gradients_computed": False,
        "B_probabilities_used_for_selection": False,
        "correct_target_used_only_after_mask_freeze_for_metrics": True,
        "smallest_vocab_ordinal_wins_argmax_ties": True,
    }


def build_attention_edge_fidelity_b_report(
    rows: Sequence[AttentionEdgeFidelityBRowCapture],
    *,
    source_kind: str,
    setup_sha256: str,
    setup_file_sha256: str,
    implementation_bundle_sha256: str,
    implementation_commit_sha: str,
    implementation_tree_sha: str,
    source_model_fingerprint_sha256: str,
    source_execution_fingerprint_sha256: str,
    exact_control_receipt: Mapping[str, object],
    model_loaded: bool,
    tokenizer_loaded: bool,
) -> dict[str, object]:
    materialized = tuple(rows)
    if source_kind not in {"model_capture", "model_free_fixture"}:
        raise ValueError("prospective B source kind is invalid")
    if len(materialized) != 60:
        raise ValueError("prospective B report requires all sixty rows")
    if tuple(row.row_ordinal for row in materialized) != tuple(range(60)):
        raise ValueError("prospective B row ordinals must be contiguous")
    if tuple(row.family_id for row in materialized) != EXPECTED_B_ROW_FAMILIES:
        raise ValueError("prospective B row-to-family sequence drifted")
    _require_sha256(setup_sha256, label="setup semantic")
    _require_sha256(setup_file_sha256, label="setup file")
    _require_sha256(implementation_bundle_sha256, label="implementation bundle")
    _require_commit(implementation_commit_sha, label="implementation commit")
    _require_commit(implementation_tree_sha, label="implementation tree")
    _require_sha256(source_model_fingerprint_sha256, label="model fingerprint")
    _require_sha256(
        source_execution_fingerprint_sha256, label="execution fingerprint"
    )
    if type(model_loaded) is not bool or type(tokenizer_loaded) is not bool:
        raise TypeError("model/tokenizer lifecycle receipts must be bool")
    exact = json.loads(_canonical_json_bytes(dict(exact_control_receipt)))
    if not isinstance(exact, dict):
        raise ValueError("exact-control receipt must be an object")
    aggregates = _aggregate_statistics(materialized)
    gates = _gate_outcomes(aggregates)
    state: dict[str, object] = {
        "schema": ATTENTION_EDGE_FIDELITY_B_SCHEMA,
        "format_version": ATTENTION_EDGE_FIDELITY_B_FORMAT_VERSION,
        "abi": ATTENTION_EDGE_FIDELITY_B_ABI,
        "role": ATTENTION_EDGE_FIDELITY_B_ROLE,
        "source_kind": source_kind,
        "setup_binding": {
            "setup_id": SETUP_ID,
            "setup_sha256": setup_sha256,
            "setup_file_sha256": setup_file_sha256,
            "check_setup_completed_before_tokenizer_or_model": True,
        },
        "implementation_binding": {
            "bundle_sha256": implementation_bundle_sha256,
            "commit_sha_before_execution": implementation_commit_sha,
            "commit_sha_after_execution": implementation_commit_sha,
            "tree_sha_before_execution": implementation_tree_sha,
            "tree_sha_after_execution": implementation_tree_sha,
            "clean_worktree_before_execution": True,
            "clean_worktree_after_execution": True,
            "all_setup_declared_HEAD_bytes_identical_before_execution": True,
            "all_setup_declared_HEAD_bytes_identical_after_execution": True,
        },
        "model_binding": {
            "model_id": MODEL_ID,
            "model_revision": PINNED_MODEL_REVISION,
            "model_dtype": "float32",
            "device": "cpu",
            "local_files_only": True,
            "source_model_fingerprint_sha256": (
                source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                source_execution_fingerprint_sha256
            ),
            "model_loaded": model_loaded,
            "tokenizer_loaded": tokenizer_loaded,
        },
        "panel_binding": {
            "panel_id": B_PANEL_ID,
            "panel_file_sha256": B_PANEL_FILE_SHA256,
            "panel_semantic_sha256": B_PANEL_SEMANTIC_SHA256,
            "panel_metadata_catalog_sha256": B_METADATA_CATALOG_SHA256,
            "row_family_sequence_sha256": _row_family_sequence_sha256(
                EXPECTED_B_ROW_FAMILIES
            ),
            "row_count": 60,
            "family_counts": [[family, 10] for family in FAMILY_IDS],
            "prompt_text_persisted": False,
            "choice_text_persisted": False,
            "correct_target_ordinal_persisted": False,
            "token_ids_persisted": False,
            "activations_or_probabilities_persisted": False,
        },
        "protocol_receipt": _protocol_receipt(),
        "exact_control_receipt": exact,
        "row_metrics": [row.to_dict() for row in materialized],
        "family_ledgers": _family_ledgers(materialized),
        "aggregate_statistics": aggregates,
        "gate_outcomes": gates,
        "all_six_preregistered_gates_passed": all(
            bool(gate["passed"]) for gate in gates
        ),
        "claim_boundary": _claim_boundary(),
    }
    state["report_sha256"] = _digest(_REPORT_DOMAIN, state)
    return validate_attention_edge_fidelity_b_content(state)


def _exact_control_receipt(
    *,
    control_kind: str,
    compiler_metadata_sha256: str,
    source_independence_audit_sha256: str,
    native_first_row_valid_logits_sha256: str,
    exact_first_row_valid_logits_sha256: str,
    source_model_fingerprint_before: str,
    source_model_fingerprint_after: str,
    source_execution_fingerprint_before: str,
    source_execution_fingerprint_after: str,
    source_model_references_released: bool,
    all_route_row_count: int,
) -> dict[str, object]:
    if control_kind not in {"live_exact", "deterministic_fixture"}:
        raise ValueError("B exact-control kind is invalid")
    state: dict[str, object] = {
        "schema": (
            "fisher_graph.gemma3_parts_first."
            "attention_edge_fidelity_b_exact_control.v1"
        ),
        "control_kind": control_kind,
        "ordered_event_ids": list(_EXPECTED_EVENT_IDS),
        "compiler_metadata_sha256": _require_sha256(
            compiler_metadata_sha256, label="compiler metadata"
        ),
        "source_independence_audit_sha256": _require_sha256(
            source_independence_audit_sha256,
            label="source independence audit",
        ),
        "native_first_row_valid_logits_sha256": _require_sha256(
            native_first_row_valid_logits_sha256,
            label="native first-row logits",
        ),
        "exact_first_row_valid_logits_sha256": _require_sha256(
            exact_first_row_valid_logits_sha256,
            label="exact first-row logits",
        ),
        "literal_native_exact_first_row_parity_passed": (
            native_first_row_valid_logits_sha256
            == exact_first_row_valid_logits_sha256
        ),
        "source_model_fingerprint_before": _require_sha256(
            source_model_fingerprint_before, label="model fingerprint before"
        ),
        "source_model_fingerprint_after": _require_sha256(
            source_model_fingerprint_after, label="model fingerprint after"
        ),
        "source_execution_fingerprint_before": _require_sha256(
            source_execution_fingerprint_before,
            label="execution fingerprint before",
        ),
        "source_execution_fingerprint_after": _require_sha256(
            source_execution_fingerprint_after,
            label="execution fingerprint after",
        ),
        "source_model_references_released_before_B_measurements": (
            source_model_references_released
        ),
        "all_route_row_count": _require_int(
            all_route_row_count, label="all-route row count", minimum=1
        ),
        "all_route_attention_operator_delta_parity_count": all_route_row_count,
        "all_route_mutation_boundary_parity_count": all_route_row_count,
        "all_route_final_endpoint_and_logit_parity_count": all_route_row_count,
        "dense_teacher_execution_count": all_route_row_count,
        "separate_batch1_all_route_replay_count": all_route_row_count,
        "separate_batch1_sparse_replay_count": all_route_row_count,
        "dense_and_sparse_logically_paired_row_count": all_route_row_count,
        "B_gradients_computed": False,
    }
    state["receipt_sha256"] = _digest(_EXACT_CONTROL_DOMAIN, state)
    return _validate_exact_control(state)


def _validate_exact_control(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema",
        "control_kind",
        "ordered_event_ids",
        "compiler_metadata_sha256",
        "source_independence_audit_sha256",
        "native_first_row_valid_logits_sha256",
        "exact_first_row_valid_logits_sha256",
        "literal_native_exact_first_row_parity_passed",
        "source_model_fingerprint_before",
        "source_model_fingerprint_after",
        "source_execution_fingerprint_before",
        "source_execution_fingerprint_after",
        "source_model_references_released_before_B_measurements",
        "all_route_row_count",
        "all_route_attention_operator_delta_parity_count",
        "all_route_mutation_boundary_parity_count",
        "all_route_final_endpoint_and_logit_parity_count",
        "dense_teacher_execution_count",
        "separate_batch1_all_route_replay_count",
        "separate_batch1_sparse_replay_count",
        "dense_and_sparse_logically_paired_row_count",
        "B_gradients_computed",
        "receipt_sha256",
    }
    _exact_keys(value, expected, label="B exact-control receipt")
    if (
        value["schema"]
        != (
            "fisher_graph.gemma3_parts_first."
            "attention_edge_fidelity_b_exact_control.v1"
        )
        or value["control_kind"] not in {"live_exact", "deterministic_fixture"}
        or value["ordered_event_ids"] != list(_EXPECTED_EVENT_IDS)
    ):
        raise ValueError("B exact-control header drifted")
    for field in (
        "compiler_metadata_sha256",
        "source_independence_audit_sha256",
        "native_first_row_valid_logits_sha256",
        "exact_first_row_valid_logits_sha256",
        "source_model_fingerprint_before",
        "source_model_fingerprint_after",
        "source_execution_fingerprint_before",
        "source_execution_fingerprint_after",
        "receipt_sha256",
    ):
        _require_sha256(value[field], label=f"exact-control {field}")
    count = _require_int(
        value["all_route_row_count"], label="all-route row count", minimum=1
    )
    for field in (
        "all_route_attention_operator_delta_parity_count",
        "all_route_mutation_boundary_parity_count",
        "all_route_final_endpoint_and_logit_parity_count",
        "dense_teacher_execution_count",
        "separate_batch1_all_route_replay_count",
        "separate_batch1_sparse_replay_count",
        "dense_and_sparse_logically_paired_row_count",
    ):
        if _require_int(value[field], label=field) != count:
            raise ValueError("B exact-control count ledger does not close")
    for field in (
        "literal_native_exact_first_row_parity_passed",
        "source_model_references_released_before_B_measurements",
    ):
        if value[field] is not True:
            raise ValueError(f"B exact-control required flag failed: {field}")
    if value["B_gradients_computed"] is not False:
        raise ValueError("prospective B must not compute gradients")
    if (
        value["native_first_row_valid_logits_sha256"]
        != value["exact_first_row_valid_logits_sha256"]
        or value["source_model_fingerprint_before"]
        != SOURCE_MODEL_FINGERPRINT_SHA256
        or value["source_model_fingerprint_after"]
        != SOURCE_MODEL_FINGERPRINT_SHA256
        or value["source_execution_fingerprint_before"]
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or value["source_execution_fingerprint_after"]
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
    ):
        raise ValueError("B exact-control pinned identity or literal parity failed")
    payload = dict(value)
    digest = payload.pop("receipt_sha256")
    if digest != _digest(_EXACT_CONTROL_DOMAIN, payload):
        raise ValueError("B exact-control semantic digest mismatch")
    return json.loads(_canonical_json_bytes(dict(value)))


def validate_attention_edge_fidelity_b_content(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild structure only; this does not authenticate live provenance."""

    if not isinstance(value, Mapping):
        raise TypeError("prospective B fidelity report must be an object")
    try:
        state = json.loads(_canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("prospective B fidelity report is not canonical JSON") from error
    if not isinstance(state, dict):
        raise ValueError("prospective B fidelity report must be an object")
    expected = {
        "schema",
        "format_version",
        "abi",
        "role",
        "source_kind",
        "setup_binding",
        "implementation_binding",
        "model_binding",
        "panel_binding",
        "protocol_receipt",
        "exact_control_receipt",
        "row_metrics",
        "family_ledgers",
        "aggregate_statistics",
        "gate_outcomes",
        "all_six_preregistered_gates_passed",
        "claim_boundary",
        "report_sha256",
    }
    _exact_keys(state, expected, label="prospective B fidelity report")
    source_kind = state["source_kind"]
    if (
        state["schema"] != ATTENTION_EDGE_FIDELITY_B_SCHEMA
        or state["format_version"] != ATTENTION_EDGE_FIDELITY_B_FORMAT_VERSION
        or state["abi"] != ATTENTION_EDGE_FIDELITY_B_ABI
        or state["role"] != ATTENTION_EDGE_FIDELITY_B_ROLE
        or source_kind not in {"model_capture", "model_free_fixture"}
    ):
        raise ValueError("prospective B fidelity header drifted")

    setup_binding = state["setup_binding"]
    if not isinstance(setup_binding, Mapping):
        raise ValueError("B setup binding must be an object")
    _exact_keys(
        setup_binding,
        {
            "setup_id",
            "setup_sha256",
            "setup_file_sha256",
            "check_setup_completed_before_tokenizer_or_model",
        },
        label="B setup binding",
    )
    if (
        setup_binding["setup_id"] != SETUP_ID
        or setup_binding["check_setup_completed_before_tokenizer_or_model"]
        is not True
    ):
        raise ValueError("B setup lifecycle binding drifted")
    _require_sha256(setup_binding["setup_sha256"], label="setup semantic")
    _require_sha256(setup_binding["setup_file_sha256"], label="setup file")

    implementation = state["implementation_binding"]
    if not isinstance(implementation, Mapping):
        raise ValueError("B implementation binding must be an object")
    _exact_keys(
        implementation,
        {
            "bundle_sha256",
            "commit_sha_before_execution",
            "commit_sha_after_execution",
            "tree_sha_before_execution",
            "tree_sha_after_execution",
            "clean_worktree_before_execution",
            "clean_worktree_after_execution",
            "all_setup_declared_HEAD_bytes_identical_before_execution",
            "all_setup_declared_HEAD_bytes_identical_after_execution",
        },
        label="B implementation binding",
    )
    _require_sha256(implementation["bundle_sha256"], label="source bundle")
    before_commit = _require_commit(
        implementation["commit_sha_before_execution"], label="commit before"
    )
    after_commit = _require_commit(
        implementation["commit_sha_after_execution"], label="commit after"
    )
    before_tree = _require_commit(
        implementation["tree_sha_before_execution"], label="tree before"
    )
    after_tree = _require_commit(
        implementation["tree_sha_after_execution"], label="tree after"
    )
    if before_commit != after_commit or before_tree != after_tree:
        raise ValueError("B source Git identity changed during execution")
    for field in (
        "clean_worktree_before_execution",
        "clean_worktree_after_execution",
        "all_setup_declared_HEAD_bytes_identical_before_execution",
        "all_setup_declared_HEAD_bytes_identical_after_execution",
    ):
        if implementation[field] is not True:
            raise ValueError(f"B source provenance flag failed: {field}")

    model = state["model_binding"]
    if not isinstance(model, Mapping):
        raise ValueError("B model binding must be an object")
    _exact_keys(
        model,
        {
            "model_id",
            "model_revision",
            "model_dtype",
            "device",
            "local_files_only",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "model_loaded",
            "tokenizer_loaded",
        },
        label="B model binding",
    )
    if (
        model["model_id"] != MODEL_ID
        or model["model_revision"] != PINNED_MODEL_REVISION
        or model["model_dtype"] != "float32"
        or model["device"] != "cpu"
        or model["local_files_only"] is not True
        or model["source_model_fingerprint_sha256"]
        != SOURCE_MODEL_FINGERPRINT_SHA256
        or model["source_execution_fingerprint_sha256"]
        != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or type(model["model_loaded"]) is not bool
        or type(model["tokenizer_loaded"]) is not bool
        or (
            source_kind == "model_capture"
            and (model["model_loaded"] is not True or model["tokenizer_loaded"] is not True)
        )
        or (
            source_kind == "model_free_fixture"
            and (model["model_loaded"] is not False or model["tokenizer_loaded"] is not False)
        )
    ):
        raise ValueError("B model binding drifted")

    panel = state["panel_binding"]
    expected_panel = {
        "panel_id": B_PANEL_ID,
        "panel_file_sha256": B_PANEL_FILE_SHA256,
        "panel_semantic_sha256": B_PANEL_SEMANTIC_SHA256,
        "panel_metadata_catalog_sha256": B_METADATA_CATALOG_SHA256,
        "row_family_sequence_sha256": _row_family_sequence_sha256(
            EXPECTED_B_ROW_FAMILIES
        ),
        "row_count": 60,
        "family_counts": [[family, 10] for family in FAMILY_IDS],
        "prompt_text_persisted": False,
        "choice_text_persisted": False,
        "correct_target_ordinal_persisted": False,
        "token_ids_persisted": False,
        "activations_or_probabilities_persisted": False,
    }
    if (
        not isinstance(panel, Mapping)
        or _canonical_json_bytes(panel) != _canonical_json_bytes(expected_panel)
    ):
        raise ValueError("B panel binding drifted")
    if (
        not isinstance(state["protocol_receipt"], Mapping)
        or _canonical_json_bytes(state["protocol_receipt"])
        != _canonical_json_bytes(_protocol_receipt())
    ):
        raise ValueError("B protocol receipt drifted")
    exact = state["exact_control_receipt"]
    if not isinstance(exact, Mapping):
        raise ValueError("B exact-control receipt must be an object")
    validated_exact = _validate_exact_control(exact)
    if validated_exact["all_route_row_count"] != 60:
        raise ValueError("B exact-control did not cover all sixty rows")
    if (
        source_kind == "model_capture"
        and validated_exact["control_kind"] != "live_exact"
    ) or (
        source_kind == "model_free_fixture"
        and validated_exact["control_kind"] != "deterministic_fixture"
    ):
        raise ValueError("B exact-control kind disagrees with source kind")

    raw_rows = state["row_metrics"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 60:
        raise ValueError("B report row metrics must contain sixty rows")
    rows = tuple(
        AttentionEdgeFidelityBRowCapture.from_dict(row)
        if isinstance(row, Mapping)
        else (_ for _ in ()).throw(ValueError("B row metric must be an object"))
        for row in raw_rows
    )
    if tuple(row.row_ordinal for row in rows) != tuple(range(60)):
        raise ValueError("B row metric order drifted")
    if tuple(row.family_id for row in rows) != EXPECTED_B_ROW_FAMILIES:
        raise ValueError("B row-to-family sequence drifted")
    if any(
        not (
            row.dense_replay_literal_parity_passed
            and row.all_route_attention_operator_delta_parity_passed
            and row.mutation_boundary_literal_parity_passed
        )
        for row in rows
    ):
        raise ValueError("B row exact-control ledger contains a failure")
    expected_families = _family_ledgers(rows)
    if _canonical_json_bytes(state["family_ledgers"]) != _canonical_json_bytes(
        expected_families
    ):
        raise ValueError("B family ledgers do not rebuild")
    expected_aggregates = _aggregate_statistics(rows)
    if _canonical_json_bytes(state["aggregate_statistics"]) != _canonical_json_bytes(
        expected_aggregates
    ):
        raise ValueError("B aggregate statistics do not rebuild")
    expected_gates = _gate_outcomes(expected_aggregates)
    if _canonical_json_bytes(state["gate_outcomes"]) != _canonical_json_bytes(
        expected_gates
    ):
        raise ValueError("B preregistered gate outcomes do not rebuild")
    passed = all(bool(gate["passed"]) for gate in expected_gates)
    if state["all_six_preregistered_gates_passed"] is not passed:
        raise ValueError("B conjunction outcome does not rebuild")
    if (
        not isinstance(state["claim_boundary"], Mapping)
        or _canonical_json_bytes(state["claim_boundary"])
        != _canonical_json_bytes(_claim_boundary())
    ):
        raise ValueError("B claim boundary drifted")
    digest = _require_sha256(state["report_sha256"], label="B report")
    payload = dict(state)
    payload.pop("report_sha256")
    if digest != _digest(_REPORT_DOMAIN, payload):
        raise ValueError("B report semantic digest mismatch")
    return state


def parse_attention_edge_fidelity_b_json(
    source: str | bytes,
) -> dict[str, object]:
    return validate_attention_edge_fidelity_b_content(
        _strict_json_object(source, label="prospective B fidelity report")
    )


def load_attention_edge_fidelity_b_content(
    source: Path | str,
) -> dict[str, object]:
    """Load any structurally valid report without claiming live provenance."""

    state, _ = _consume_regular(
        Path(source),
        label="prospective B fidelity report",
        byte_cap=_MAX_REPORT_BYTES,
        required_mode=0o600,
        required_nlink=1,
        consumer=parse_attention_edge_fidelity_b_json,
    )
    return state


def load_attention_edge_fidelity_b(
    source: Path | str = DEFAULT_B_REPORT_OUTPUT,
    *,
    setup_path: Path | str | None = None,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    """Authenticate a published live report against setup and clean HEAD."""

    active_setup_path = DEFAULT_B_SETUP if setup_path is None else Path(setup_path)
    config = AttentionEdgeFidelityBLiveConfig(
        setup_path=active_setup_path,
        output=DEFAULT_B_REPORT_OUTPUT,
        repository_root=(
            None if repository_root is None else Path(repository_root)
        ),
    )
    before_state, before_file_sha256 = _checked_setup(config)
    def authenticate(raw: bytes) -> dict[str, object]:
        state = parse_attention_edge_fidelity_b_json(raw)
        after_state, after_file_sha256 = _checked_setup(config)
        if (
            before_state != after_state
            or before_file_sha256 != after_file_sha256
        ):
            raise ValueError("B setup/source changed during report authentication")
        reproduced_state, reproduced_file_sha256 = _require_live_setup_binding(
            state,
            setup_path=active_setup_path,
            repository_root=repository_root,
        )
        if (
            reproduced_state != before_state
            or reproduced_file_sha256 != before_file_sha256
        ):
            raise ValueError("B setup/source authentication did not reproduce")
        return state

    state, _ = _consume_regular(
        Path(source),
        label="prospective B authenticated fidelity report",
        byte_cap=_MAX_REPORT_BYTES,
        required_mode=0o600,
        required_nlink=1,
        consumer=authenticate,
    )
    return state


def _is_canonical_report_destination(path: Path) -> bool:
    """Compare normalized spellings; strict traversal separately rejects aliases."""

    return Path(os.path.abspath(path)) == Path(
        os.path.abspath(DEFAULT_B_REPORT_OUTPUT)
    )


def _is_canonical_report_parent(opened_parent: os.stat_result) -> bool:
    """Match the opened inode so filesystem case/Unicode aliases cannot bypass auth."""

    try:
        _, descriptor, canonical_parent = _open_strict_parent_directory(
            DEFAULT_B_REPORT_OUTPUT,
            label="canonical B report destination",
            create=False,
        )
    except FileNotFoundError:
        return False
    try:
        return _inode_identity(opened_parent) == _inode_identity(canonical_parent)
    finally:
        os.close(descriptor)


def _unlink_owned_at(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> bool:
    try:
        observed = _lstat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    if _inode_identity(observed) != identity:
        return False
    os.unlink(name, dir_fd=directory_descriptor)
    return True


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("B report temporary write made no progress")
        view = view[written:]


def _read_exact_descriptor(descriptor: int, expected_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("B report publication inode was truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise RuntimeError("B report publication inode grew unexpectedly")
    return b"".join(chunks)


def _verify_publication_inode(
    value: os.stat_result,
    *,
    identity: tuple[int, int],
    expected_size: int,
    expected_nlink: int,
    label: str,
) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_size != expected_size
        or value.st_nlink != expected_nlink
        or _inode_identity(value) != identity
    ):
        raise RuntimeError(f"{label} identity, mode, size, or link count drifted")


def _require_live_setup_binding(
    state: Mapping[str, object],
    *,
    setup_path: Path | str,
    repository_root: Path | str | None,
) -> tuple[dict[str, object], str]:
    if state.get("source_kind") != "model_capture":
        raise ValueError("canonical B publication requires a live model_capture")
    exact = state.get("exact_control_receipt")
    model = state.get("model_binding")
    if (
        not isinstance(exact, Mapping)
        or exact.get("control_kind") != "live_exact"
        or not isinstance(model, Mapping)
        or model.get("model_loaded") is not True
        or model.get("tokenizer_loaded") is not True
    ):
        raise ValueError("canonical B publication requires live exact execution")
    config = AttentionEdgeFidelityBLiveConfig(
        setup_path=Path(setup_path),
        output=DEFAULT_B_REPORT_OUTPUT,
        repository_root=(
            None if repository_root is None else Path(repository_root)
        ),
    )
    setup_state, setup_file_sha256 = _checked_setup(config)
    setup_binding = state.get("setup_binding")
    implementation_binding = state.get("implementation_binding")
    implementation_receipt = setup_state.get("implementation_receipt")
    if not all(
        isinstance(item, Mapping)
        for item in (
            setup_binding,
            implementation_binding,
            implementation_receipt,
        )
    ):
        raise ValueError("canonical B publication binding is incomplete")
    assert isinstance(setup_binding, Mapping)
    assert isinstance(implementation_binding, Mapping)
    assert isinstance(implementation_receipt, Mapping)
    if (
        setup_binding.get("setup_sha256") != setup_state.get("setup_sha256")
        or setup_binding.get("setup_file_sha256") != setup_file_sha256
        or implementation_binding.get("bundle_sha256")
        != implementation_receipt.get("bundle_sha256")
        or implementation_binding.get("commit_sha_before_execution")
        != implementation_receipt.get("commit_sha")
        or implementation_binding.get("commit_sha_after_execution")
        != implementation_receipt.get("commit_sha")
        or implementation_binding.get("tree_sha_before_execution")
        != implementation_receipt.get("tree_sha")
        or implementation_binding.get("tree_sha_after_execution")
        != implementation_receipt.get("tree_sha")
    ):
        raise ValueError("canonical B setup/source binding does not reproduce")
    return setup_state, setup_file_sha256


def write_attention_edge_fidelity_b(
    value: Mapping[str, object],
    destination: Path | str = DEFAULT_B_REPORT_OUTPUT,
    *,
    setup_path: Path | str | None = None,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    state = validate_attention_edge_fidelity_b_content(value)
    path = Path(destination)
    canonical = _is_canonical_report_destination(path)
    authenticated_setup: tuple[dict[str, object], str] | None = None
    if canonical:
        authenticated_setup = _require_live_setup_binding(
            state,
            setup_path=DEFAULT_B_SETUP if setup_path is None else setup_path,
            repository_root=repository_root,
        )
    path, directory_descriptor, opened_parent = _open_strict_parent_directory(
        path,
        label="B report destination",
        create=True,
    )
    if not canonical and _is_canonical_report_parent(opened_parent):
        canonical = True
        try:
            authenticated_setup = _require_live_setup_binding(
                state,
                setup_path=(
                    DEFAULT_B_SETUP if setup_path is None else setup_path
                ),
                repository_root=repository_root,
            )
        except BaseException:
            os.close(directory_descriptor)
            raise
    try:
        _lstat_at(directory_descriptor, path.name)
    except FileNotFoundError:
        pass
    except BaseException:
        os.close(directory_descriptor)
        raise
    else:
        os.close(directory_descriptor)
        raise FileExistsError(path)
    payload = json.dumps(
        state,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = -1
    temporary_name: str | None = None
    owned_identity: tuple[int, int] | None = None
    try:
        for _ in range(128):
            candidate = f".{path.name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise FileExistsError("could not allocate unique B report temporary")
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        owned_identity = _inode_identity(initial)
        _verify_publication_inode(
            initial,
            identity=owned_identity,
            expected_size=0,
            expected_nlink=1,
            label="B report temporary",
        )
        _verify_publication_inode(
            _lstat_at(directory_descriptor, temporary_name),
            identity=owned_identity,
            expected_size=0,
            expected_nlink=1,
            label="B report named temporary",
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        for label, observed in (
            ("B report fsynced temporary", os.fstat(descriptor)),
            (
                "B report named fsynced temporary",
                _lstat_at(directory_descriptor, temporary_name),
            ),
        ):
            _verify_publication_inode(
                observed,
                identity=owned_identity,
                expected_size=len(payload),
                expected_nlink=1,
                label=label,
            )
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        for label, observed in (
            ("B report linked descriptor", os.fstat(descriptor)),
            (
                "B report linked temporary",
                _lstat_at(directory_descriptor, temporary_name),
            ),
            (
                "B report linked destination",
                _lstat_at(directory_descriptor, path.name),
            ),
        ):
            _verify_publication_inode(
                observed,
                identity=owned_identity,
                expected_size=len(payload),
                expected_nlink=2,
                label=label,
            )
        if not _unlink_owned_at(
            directory_descriptor, temporary_name, owned_identity
        ):
            raise RuntimeError("B report temporary identity changed before unlink")
        temporary_name = None
        os.fsync(directory_descriptor)
        for label, observed in (
            ("B report final descriptor", os.fstat(descriptor)),
            (
                "B report final destination",
                _lstat_at(directory_descriptor, path.name),
            ),
        ):
            _verify_publication_inode(
                observed,
                identity=owned_identity,
                expected_size=len(payload),
                expected_nlink=1,
                label=label,
            )
        if load_attention_edge_fidelity_b_content(path) != state:
            raise RuntimeError("B report strict reload drifted")
        if canonical:
            reproduced_setup = _require_live_setup_binding(
                state,
                setup_path=(
                    DEFAULT_B_SETUP if setup_path is None else setup_path
                ),
                repository_root=repository_root,
            )
            if reproduced_setup != authenticated_setup:
                raise RuntimeError(
                    "B setup or source changed during canonical publication"
                )
        _verify_publication_inode(
            _lstat_at(directory_descriptor, path.name),
            identity=owned_identity,
            expected_size=len(payload),
            expected_nlink=1,
            label="B report post-load destination",
        )
        if _read_exact_descriptor(descriptor, len(payload)) != payload:
            raise RuntimeError("B report publication bytes changed after reload")
        _reopen_same_strict_parent(
            path,
            label="B report destination",
            expected_parent=opened_parent,
        )
        os.fsync(directory_descriptor)
    except BaseException:
        removed_owned_inode = False
        if owned_identity is not None and temporary_name is not None:
            removed_owned_inode |= _unlink_owned_at(
                directory_descriptor,
                temporary_name,
                owned_identity,
            )
        if owned_identity is not None:
            removed_owned_inode |= _unlink_owned_at(
                directory_descriptor,
                path.name,
                owned_identity,
            )
        if removed_owned_inode:
            os.fsync(directory_descriptor)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)
    return state


def build_deterministic_attention_edge_fidelity_b_fixture() -> dict[str, object]:
    rows: list[AttentionEdgeFidelityBRowCapture] = []
    for row in range(60):
        family = EXPECTED_B_ROW_FAMILIES[row]
        delta = 0.008 + (row % 5) * 0.001
        sequence_length = 8 + row % 4
        rows.append(
            AttentionEdgeFidelityBRowCapture(
                row_ordinal=row,
                family_id=family,
                sequence_length=sequence_length,
                tokenization_sha256=hashlib.sha256(
                    f"B-fixture-tokenization-{row}".encode()
                ).hexdigest(),
                selection_catalog_sha256=hashlib.sha256(
                    f"B-fixture-selection-{row}".encode()
                ).hexdigest(),
                available_route_count=(
                    18 * 4 * sequence_length * (sequence_length + 1) // 2
                ),
                selected_route_count=(
                    18 * (4 + PRIMARY_K * (sequence_length - 1))
                ),
                native_correct_target_nll=2.0 + row * 0.001,
                sparse_correct_target_nll=2.0 + row * 0.001 + delta,
                full_vocab_top1_agreement=True,
                terminal_supervised_vector_cosine=0.9985 - (row % 3) * 0.0001,
                centered_logit_nrmse=0.018 + (row % 4) * 0.001,
                dense_teacher_operator_nrmse=0.08 + (row % 5) * 0.002,
                dense_replay_literal_parity_passed=True,
                all_route_attention_operator_delta_parity_passed=True,
                mutation_boundary_literal_parity_passed=True,
            )
        )
    fixed = hashlib.sha256(b"prospective-B-fixture").hexdigest()
    exact = _exact_control_receipt(
        control_kind="deterministic_fixture",
        compiler_metadata_sha256=fixed,
        source_independence_audit_sha256=fixed,
        native_first_row_valid_logits_sha256=fixed,
        exact_first_row_valid_logits_sha256=fixed,
        source_model_fingerprint_before=SOURCE_MODEL_FINGERPRINT_SHA256,
        source_model_fingerprint_after=SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_before=(
            PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_after=(
            PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_model_references_released=True,
        all_route_row_count=60,
    )
    return build_attention_edge_fidelity_b_report(
        rows,
        source_kind="model_free_fixture",
        setup_sha256=hashlib.sha256(b"fixture setup semantic").hexdigest(),
        setup_file_sha256=hashlib.sha256(b"fixture setup file").hexdigest(),
        implementation_bundle_sha256=hashlib.sha256(
            b"fixture implementation bundle"
        ).hexdigest(),
        implementation_commit_sha="1" * 40,
        implementation_tree_sha="2" * 40,
        source_model_fingerprint_sha256=SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_sha256=(
            PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        exact_control_receipt=exact,
        model_loaded=False,
        tokenizer_loaded=False,
    )


@dataclass(frozen=True, slots=True)
class AttentionEdgeFidelityBLiveConfig:
    setup_path: Path = DEFAULT_B_SETUP
    output: Path = DEFAULT_B_REPORT_OUTPUT
    cache_dir: Path | None = None
    repository_root: Path | None = None
    local_files_only: bool = True
    maximum_prompt_tokens: int = ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_path", Path(self.setup_path))
        object.__setattr__(self, "output", Path(self.output))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.repository_root is not None:
            object.__setattr__(self, "repository_root", Path(self.repository_root))
        if self.local_files_only is not True:
            raise ValueError("prospective B requires local-only model loading")
        if self.maximum_prompt_tokens != ATTENTION_EDGE_MAXIMUM_PROMPT_TOKENS:
            raise ValueError("prospective B prompt-token limit is frozen at 128")
        if self.setup_path.suffix != ".json" or self.output.suffix != ".json":
            raise ValueError("prospective B setup and report must be JSON")


def _checked_setup(
    config: AttentionEdgeFidelityBLiveConfig,
) -> tuple[dict[str, object], str]:
    """Authenticate setup bytes around ``check_setup`` without model access."""

    def checked(raw: bytes) -> dict[str, object]:
        state = check_setup(
            config.setup_path,
            repository_root=config.repository_root,
        )
        if not isinstance(state, dict):
            raise TypeError("check_setup did not return a strict setup object")
        raw_state = validate_setup(
            _strict_json_object(raw, label="prospective B setup")
        )
        if _canonical_json_bytes(raw_state) != _canonical_json_bytes(state):
            raise ValueError(
                "check_setup result does not match authenticated setup bytes"
            )
        return state

    state, raw = _consume_regular(
        config.setup_path,
        label="prospective B setup",
        byte_cap=8 * 1024 * 1024,
        required_mode=0o600,
        required_nlink=1,
        consumer=checked,
    )
    setup_sha = _require_sha256(state.get("setup_sha256"), label="setup semantic")
    output_contract = state.get("output_contract")
    if (
        not isinstance(output_contract, Mapping)
        or output_contract.get("default_setup_artifact_path")
        != str(DEFAULT_B_SETUP)
        or output_contract.get("planned_B_report_path")
        != str(DEFAULT_B_REPORT_OUTPUT)
        or output_contract.get("B_report_schema_and_writer_implemented")
        is not True
        or config.output != DEFAULT_B_REPORT_OUTPUT
    ):
        raise ValueError("prospective B output differs from the issued setup")
    if state.get("setup_id") != SETUP_ID:
        raise ValueError("prospective B setup ID drifted")
    # The semantic field is already validated by check_setup; retain this read
    # to make the ordering and byte binding explicit at the consumer boundary.
    if not setup_sha:
        raise AssertionError("unreachable empty setup semantic digest")
    return state, hashlib.sha256(raw).hexdigest()


def build_attention_edge_fidelity_b_preflight(
    config: AttentionEdgeFidelityBLiveConfig = AttentionEdgeFidelityBLiveConfig(),
) -> dict[str, object]:
    """Validate the issued setup without loading a tokenizer or model."""

    state, setup_file_sha256 = _checked_setup(config)
    implementation = state.get("implementation_receipt")
    if not isinstance(implementation, Mapping):
        raise ValueError("B setup omitted its implementation receipt")
    return {
        "schema": (
            "fisher_graph.gemma3_parts_first."
            "attention_edge_fidelity_b_preflight.v1"
        ),
        "ready_to_load": True,
        "setup_checked": True,
        "setup_sha256": state["setup_sha256"],
        "setup_file_sha256": setup_file_sha256,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "model_id": MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_dtype": "float32",
        "device": "cpu",
        "local_files_only": True,
        "panel_id": B_PANEL_ID,
        "panel_file_sha256": B_PANEL_FILE_SHA256,
        "selector_id": "A_atlas_motif_mean_intrinsic_fisher_energy.v1",
        "A_motif_score_catalog_sha256": A_MOTIF_SCORE_CATALOG_SHA256,
        "maximum_global_K": PRIMARY_K,
        "selected_route_count_formula": (
            "min(8,allowed_global_head_key_route_count)"
        ),
        "omitted_route_semantics": "zero_fill_without_renormalization",
        "tokenizer_loaded": False,
        "B_tokenization_performed": False,
        "model_loaded": False,
        "model_execution_performed": False,
        "B_gradients_computed": False,
        "B_result_present": False,
    }


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


def _tokenizer_tensor(encoded: object, name: str) -> Tensor:
    value = encoded.get(name) if isinstance(encoded, Mapping) else getattr(
        encoded, name, None
    )
    if not isinstance(value, Tensor):
        raise TypeError(f"tokenizer output omitted Tensor {name!r}")
    return value


def _tokenize_prompt_without_target(
    tokenizer: object,
    prompt: str,
    *,
    maximum_prompt_tokens: int,
) -> tuple[Tensor, Tensor]:
    """Tokenize only prompt text; this helper cannot read a choice or label."""

    if not callable(tokenizer) or not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("prospective B tokenizer lacks required interfaces")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prospective B prompt must be nonempty")
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("prospective B tokenizer lacks padding and EOS")
        setattr(tokenizer, "pad_token", eos_token)
    if hasattr(tokenizer, "padding_side"):
        setattr(tokenizer, "padding_side", "right")
    encoded = tokenizer(
        [prompt],
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=maximum_prompt_tokens,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = _tokenizer_tensor(encoded, "input_ids").to(
        device="cpu", dtype=torch.int64
    )
    attention_mask = _tokenizer_tensor(encoded, "attention_mask").to(
        device="cpu", dtype=torch.bool
    )
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape != attention_mask.shape
        or not bool(attention_mask.any())
    ):
        raise ValueError("prospective B prompt tokenization has invalid geometry")
    return input_ids.contiguous(), attention_mask.contiguous()


def _attach_correct_target_after_mask_freeze(
    tokenizer: object,
    example: object,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> CalibrationBatch:
    """Read the correct target only after the caller has frozen all masks."""

    choices = getattr(example, "choices", None)
    correct_choice = getattr(example, "correct_choice", None)
    example_id = getattr(example, "example_id", None)
    if (
        not isinstance(choices, tuple)
        or type(correct_choice) is not int
        or not 0 <= correct_choice < len(choices)
        or not isinstance(example_id, str)
    ):
        raise ValueError("prospective B example target metadata is invalid")
    correct = choices[correct_choice]
    choice_ids = tokenizer.encode(" " + correct, add_special_tokens=False)
    if not isinstance(choice_ids, list) or len(choice_ids) != 1:
        raise ValueError(
            "prospective B correct choice must be one leading-space token"
        )
    positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    final_position = positions.masked_fill(~attention_mask, -1).amax(dim=1)
    if int(final_position.item()) < 0:
        raise ValueError("prospective B prompt is empty")
    targets = torch.full_like(input_ids, -100)
    valid_positions = torch.zeros_like(attention_mask)
    query = int(final_position.item())
    targets[0, query] = int(choice_ids[0])
    valid_positions[0, query] = True
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        targets=targets,
        valid_positions=valid_positions,
        example_ids=(example_id,),
    )


def _tokenization_sha256(
    row_ordinal: int, input_ids: Tensor, attention_mask: Tensor
) -> str:
    return _digest(
        _TOKENIZATION_DOMAIN,
        {
            "row_ordinal": row_ordinal,
            "input_ids_sha256": _tensor_sha256(input_ids),
            "attention_mask_sha256": _tensor_sha256(attention_mask),
        },
    )


def _attention_generators(executor: object) -> tuple[Gemma3AttentionGenerator, ...]:
    mutations = getattr(executor, "mutations", None)
    transports = getattr(executor, "transports", None)
    if (
        not isinstance(mutations, nn.ModuleList)
        or len(mutations) != 36
        or not isinstance(transports, nn.ModuleList)
        or len(transports) != 0
    ):
        raise ValueError("prospective B requires the exact transport-free executor")
    result = tuple(
        getattr(mutations[2 * layer], "compiled_operator", None)
        for layer in range(GEMMA3_LAYER_COUNT)
    )
    if any(not isinstance(item, Gemma3AttentionGenerator) for item in result):
        raise ValueError("prospective B exact executor omitted attention generators")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _PreparedBRow:
    row_ordinal: int
    input_ids: Tensor
    attention_mask: Tensor
    sequence: SequenceContext
    layer_masks: tuple[Tensor, ...]
    tokenization_sha256: str
    selection_catalog_sha256: str

    def __post_init__(self) -> None:
        if len(self.layer_masks) != GEMMA3_LAYER_COUNT:
            raise ValueError("prepared B row omitted a layer mask")


def _prepare_prompt_masks(
    *,
    row_ordinal: int,
    prompt: str,
    tokenizer: object,
    adapter: Gemma3CausalLMAdapter,
    generators: Sequence[Gemma3AttentionGenerator],
    motif_catalog: Sequence[Mapping[str, object]],
    maximum_prompt_tokens: int,
) -> _PreparedBRow:
    input_ids, attention_mask = _tokenize_prompt_without_target(
        tokenizer,
        prompt,
        maximum_prompt_tokens=maximum_prompt_tokens,
    )
    sequence = adapter.prepare_sequence(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    masks: list[Tensor] = []
    for layer, generator in enumerate(generators):
        allowed = _allowed_route_geometry(
            sequence,
            window_size=generator.config.attention.window_size,
        )
        masks.append(
            select_frozen_a_motif_mask(
                allowed,
                motif_catalog,
                layer_ordinal=layer,
            )
        )
    selection_sha = _digest(
        _SELECTION_DOMAIN,
        {
            "selector_id": "A_atlas_motif_mean_intrinsic_fisher_energy.v1",
            "A_motif_score_catalog_sha256": A_MOTIF_SCORE_CATALOG_SHA256,
            "maximum_global_K": PRIMARY_K,
            "selected_route_count_formula": (
                "min(8,allowed_global_head_key_route_count)"
            ),
            "layer_mask_sha256s": [_tensor_sha256(mask) for mask in masks],
        },
    )
    return _PreparedBRow(
        row_ordinal=row_ordinal,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequence=sequence,
        layer_masks=tuple(masks),
        tokenization_sha256=_tokenization_sha256(
            row_ordinal, input_ids, attention_mask
        ),
        selection_catalog_sha256=selection_sha,
    )


def _valid_logits(logits: Tensor, batch: CalibrationBatch) -> Tensor:
    if logits.shape[:2] != batch.valid_positions.shape:
        raise ValueError("B logits disagree with objective positions")
    if int(batch.valid_positions.sum().item()) != 1:
        raise ValueError("prospective B row requires one supervised query")
    return logits[batch.valid_positions.to(device=logits.device)].detach().clone()


def _hash_mapping(domain: bytes, value: Mapping[str, object]) -> str:
    return _digest(domain, json.loads(_canonical_json_bytes(dict(value))))


def _smallest_index_argmax(values: Tensor) -> int:
    vector = values.detach().to(device="cpu")
    if vector.ndim != 1 or not vector.numel() or not bool(torch.isfinite(vector).all()):
        raise ValueError("full-vocabulary argmax vector is invalid")
    maximum = vector.max()
    ties = torch.nonzero(vector == maximum, as_tuple=False).reshape(-1)
    if not ties.numel():
        raise AssertionError("finite argmax unexpectedly has no maximizer")
    return int(ties.min().item())


def _strict_cosine(reference: Tensor, sparse: Tensor) -> float:
    left = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = sparse.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if left.shape != right.shape or not left.numel():
        raise ValueError("terminal cosine vectors do not align")
    left_sq = float(left.square().sum().item())
    right_sq = float(right.square().sum().item())
    if left_sq <= 0.0 or right_sq <= 0.0:
        raise ValueError("terminal cosine has a zero denominator")
    result = float((left * right).sum().item()) / math.sqrt(left_sq * right_sq)
    if not math.isfinite(result):
        raise ValueError("terminal cosine is nonfinite")
    return min(1.0, max(-1.0, result))


def _centered_logit_nrmse(reference: Tensor, sparse: Tensor) -> float:
    left = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = sparse.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if left.shape != right.shape or not left.numel():
        raise ValueError("centered-logit metric vectors do not align")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(left_centered.square().sum().item())
    numerator = float((right_centered - left_centered).square().sum().item())
    if denominator <= 0.0:
        raise ValueError("centered-logit NRMSE has a zero denominator")
    result = math.sqrt(numerator / denominator)
    if not math.isfinite(result):
        raise ValueError("centered-logit NRMSE is nonfinite")
    return result


def _reconstruct_dense_residual_carriers(
    lexical_output: Tensor,
    mutation_deltas: Sequence[Tensor],
    query_valid_mask: Tensor,
) -> tuple[Tensor, ...]:
    """Rebuild residual carriers; mutation chart inputs are not carriers."""

    if (
        not isinstance(lexical_output, Tensor)
        or lexical_output.ndim != 3
        or not lexical_output.is_floating_point()
        or not bool(torch.isfinite(lexical_output).all())
        or len(mutation_deltas) != 36
        or not isinstance(query_valid_mask, Tensor)
        or query_valid_mask.dtype is not torch.bool
        or tuple(query_valid_mask.shape) != tuple(lexical_output.shape[:2])
    ):
        raise ValueError("dense residual-carrier reconstruction inputs are invalid")
    valid = query_valid_mask.unsqueeze(-1)
    carrier = lexical_output
    result: list[Tensor] = []
    for delta in mutation_deltas:
        if (
            not isinstance(delta, Tensor)
            or delta.shape != lexical_output.shape
            or delta.dtype != lexical_output.dtype
            or delta.device != lexical_output.device
            or not bool(torch.isfinite(delta).all())
        ):
            raise ValueError("dense mutation delta is invalid")
        carrier = torch.where(
            valid,
            carrier + delta.masked_fill(~valid, 0),
            lexical_output,
        )
        result.append(carrier)
    return tuple(result)


def _run_separate_batch1_dense_and_sparse_replays(
    *,
    executor: object,
    input_ids: Tensor,
    sequence: SequenceContext,
    all_route_masks: tuple[Tensor, ...],
    sparse_masks: tuple[Tensor, ...],
    supervised_query: int,
) -> tuple[
    fidelity_kernel.SparseWholeModelExecution,
    fidelity_kernel.SparseWholeModelExecution,
]:
    """Keep literal dense control and sparse execution in batch-one calls."""

    if (
        not isinstance(input_ids, Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or sequence.batch_size != 1
        or len(all_route_masks) != GEMMA3_LAYER_COUNT
        or len(sparse_masks) != GEMMA3_LAYER_COUNT
        or any(mask.shape[0] != 1 for mask in (*all_route_masks, *sparse_masks))
    ):
        raise ValueError("prospective B replays must each use batch size one")
    dense = fidelity_kernel.replay_exact_with_attention_masks(
        executor,
        input_ids,
        sequence,
        all_route_masks,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
        egress_query_ordinal=supervised_query,
    )
    sparse = fidelity_kernel.replay_exact_with_attention_masks(
        executor,
        input_ids,
        sequence,
        sparse_masks,
        renormalize_rows=torch.zeros(1, dtype=torch.bool),
        egress_query_ordinal=supervised_query,
    )
    return dense, sparse


@dataclass(slots=True)
class AttentionEdgeFidelityBRuntime:
    setup_state: dict[str, object]
    setup_file_sha256: str
    executor: object
    generators: tuple[Gemma3AttentionGenerator, ...]
    prepared_rows: tuple[_PreparedBRow, ...]
    batches: tuple[CalibrationBatch, ...]
    family_ids: tuple[str, ...]
    compiler_metadata_sha256: str
    source_independence_audit_sha256: str
    native_first_row_valid_logits_sha256: str
    exact_first_row_valid_logits_sha256: str
    source_model_fingerprint_before: str
    source_model_fingerprint_after: str
    source_execution_fingerprint_before: str
    source_execution_fingerprint_after: str
    source_model_references_released: bool
    config: AttentionEdgeFidelityBLiveConfig

    @classmethod
    def load(
        cls,
        config: AttentionEdgeFidelityBLiveConfig = AttentionEdgeFidelityBLiveConfig(),
        *,
        model_loader: _ModelLoader | None = None,
        exact_compiler: _ExactCompiler | None = None,
    ) -> "AttentionEdgeFidelityBRuntime":
        # This call is intentionally the first action.  In particular, the B
        # panel dataclasses and model-loader modules are not touched above it.
        setup_state, setup_file_sha256 = _checked_setup(config)
        if config.output.exists() or config.output.is_symlink():
            raise FileExistsError(config.output)
        panel = load_fidelity_b_panel(repository_root=config.repository_root)
        examples = getattr(panel, "examples", None)
        if not isinstance(examples, tuple) or len(examples) != 60:
            raise ValueError("authenticated prospective B panel omitted rows")
        loader = _default_model_loader() if model_loader is None else model_loader
        compiler = (
            _default_exact_compiler() if exact_compiler is None else exact_compiler
        )
        device = torch.device("cpu")
        tokenizer, model = loader(
            model_id=MODEL_ID,
            revision=PINNED_MODEL_REVISION,
            cache_dir=_resolve_cache_dir(config.cache_dir),
            device=device,
            dtype="float32",
            local_files_only=True,
        )
        if not isinstance(model, nn.Module):
            raise TypeError("prospective B model loader omitted a module")
        model_reference = weakref.ref(model)
        model.eval().requires_grad_(False)
        adapter = Gemma3CausalLMAdapter(model)
        model_before = adapter.model_fingerprint()
        execution_before = adapter.execution_fingerprint()
        if (
            model_before != SOURCE_MODEL_FINGERPRINT_SHA256
            or execution_before != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ):
            raise ValueError("prospective B loaded model differs from setup pins")
        compilation = compiler(adapter, dtype=torch.float32, device=device)
        metadata_callback = getattr(compilation, "metadata", None)
        metadata = metadata_callback() if callable(metadata_callback) else None
        independence = getattr(compilation, "source_independence_audit", None)
        executor = getattr(compilation, "executor", None)
        if (
            getattr(compilation, "compilation_kind", None)
            != "exact_39_event_control"
            or not isinstance(metadata, Mapping)
            or metadata.get("exact_parameter_inventory_parity") is not True
            or not isinstance(independence, Mapping)
            or independence.get("passed") is not True
            or tuple(getattr(executor, "ordered_event_ids", ()))
            != _EXPECTED_EVENT_IDS
        ):
            raise RuntimeError("prospective B exact-control compilation failed")
        generators = _attention_generators(executor)
        catalog = setup_state.get("A_motif_score_catalog")
        if not isinstance(catalog, list):
            raise ValueError("prospective B setup omitted its A motif catalog")

        # Phase 1 reads only prompt text.  Every discrete mask for all sixty
        # rows is fixed before phase 2 reads a correct choice or target.
        prepared = tuple(
            _prepare_prompt_masks(
                row_ordinal=ordinal,
                prompt=getattr(example, "prompt"),
                tokenizer=tokenizer,
                adapter=adapter,
                generators=generators,
                motif_catalog=catalog,
                maximum_prompt_tokens=config.maximum_prompt_tokens,
            )
            for ordinal, example in enumerate(examples)
        )
        batches = tuple(
            _attach_correct_target_after_mask_freeze(
                tokenizer,
                example,
                row.input_ids,
                row.attention_mask,
            )
            for example, row in zip(examples, prepared, strict=True)
        )
        family_ids = tuple(str(getattr(example, "family_id")) for example in examples)
        if family_ids != EXPECTED_B_ROW_FAMILIES:
            raise ValueError("prospective B live row-to-family sequence drifted")

        first_batch = batches[0]
        with torch.no_grad():
            native = adapter.forward(first_batch.model_inputs)
            native_valid = _valid_logits(native.logits, first_batch)
            exact_execution = executor.forward_components(
                first_batch.model_inputs["input_ids"], prepared[0].sequence
            )
            exact_valid = _valid_logits(exact_execution.logits, first_batch)
        native_hash = _tensor_sha256(native_valid)
        exact_hash = _tensor_sha256(exact_valid)
        if not torch.equal(native_valid, exact_valid) or native_hash != exact_hash:
            raise RuntimeError("prospective B native/exact first-row parity failed")
        model_after = adapter.model_fingerprint()
        execution_after = adapter.execution_fingerprint()
        if model_after != model_before or execution_after != execution_before:
            raise RuntimeError("source model changed during prospective B control")
        compiler_metadata_sha256 = _hash_mapping(
            _EXACT_CONTROL_DOMAIN + b"metadata\0", metadata
        )
        independence_sha256 = _hash_mapping(
            _EXACT_CONTROL_DOMAIN + b"independence\0", independence
        )
        del native_valid, exact_valid, native, exact_execution
        del adapter, model, compilation, metadata_callback
        gc.collect()
        released = model_reference() is None
        if not released:
            raise RuntimeError(
                "prospective B native source references survived exact compilation"
            )
        return cls(
            setup_state=setup_state,
            setup_file_sha256=setup_file_sha256,
            executor=executor,
            generators=generators,
            prepared_rows=prepared,
            batches=batches,
            family_ids=family_ids,
            compiler_metadata_sha256=compiler_metadata_sha256,
            source_independence_audit_sha256=independence_sha256,
            native_first_row_valid_logits_sha256=native_hash,
            exact_first_row_valid_logits_sha256=exact_hash,
            source_model_fingerprint_before=model_before,
            source_model_fingerprint_after=model_after,
            source_execution_fingerprint_before=execution_before,
            source_execution_fingerprint_after=execution_after,
            source_model_references_released=released,
            config=config,
        )

    def _capture_row(self, row_ordinal: int) -> AttentionEdgeFidelityBRowCapture:
        prepared = self.prepared_rows[row_ordinal]
        batch = self.batches[row_ordinal]
        mutation_inputs: dict[str, Tensor] = {}
        mutation_outputs: dict[str, Tensor] = {}
        lexical_outputs: list[Tensor] = []

        def observer(*, event_id: str, role: str, values: Tensor) -> None:
            if event_id == "lexical.ingress" and role == "output":
                lexical_outputs.append(values)
                return
            if not event_id.startswith("mutation."):
                return
            if role == "input":
                mutation_inputs[event_id] = values
            elif role == "output":
                mutation_outputs[event_id] = values

        with torch.no_grad():
            observed = self.executor.forward_components_with_observer(
                batch.model_inputs["input_ids"],
                prepared.sequence,
                observer=observer,
                retain_gradient_tape=False,
            )
        if (
            len(mutation_inputs) != 36
            or len(mutation_outputs) != 36
            or len(lexical_outputs) != 1
        ):
            raise RuntimeError("prospective B dense observer missed a mutation")
        supervised = torch.nonzero(
            batch.valid_positions[0], as_tuple=False
        ).reshape(-1)
        if supervised.numel() != 1:
            raise ValueError("prospective B row must have one supervised query")
        supervised_query = int(supervised[0].item())

        factors: list[fidelity_kernel.AttentionRouteTensors] = []
        teacher_reference_sq = 0.0
        teacher_error_sq = 0.0
        available_route_count = 0
        selected_route_count = 0
        local_attention_parity = True
        valid_operator_mask = prepared.sequence.query_valid_mask.unsqueeze(-1)
        for layer, generator in enumerate(self.generators):
            event_id = f"mutation.{1 + 2 * layer}"
            route = fidelity_kernel.attention_route_tensors(
                generator,
                mutation_inputs[event_id],
                prepared.sequence,
            )
            expected_allowed = _allowed_route_geometry(
                prepared.sequence,
                window_size=generator.config.attention.window_size,
            )
            if not torch.equal(route.allowed_mask, expected_allowed):
                raise RuntimeError("B selector and fidelity visibility differ")
            selection = prepared.layer_masks[layer]
            if bool((selection & ~route.allowed_mask[:, None, :, :]).any()):
                raise RuntimeError("frozen B selection contains a forbidden route")
            routed = route.probabilities.masked_fill(~selection, 0)
            local_operator, _ = fidelity_kernel.operator_from_routed_probabilities(
                generator,
                routed,
                route.repeated_values,
                prepared.sequence,
            )
            expanded_valid = valid_operator_mask.expand_as(local_operator)
            reference64 = route.reference_operator_output.masked_select(
                expanded_valid
            ).to(torch.float64)
            sparse64 = local_operator.masked_select(expanded_valid).to(torch.float64)
            teacher_reference_sq += float(reference64.square().sum().item())
            teacher_error_sq += float((sparse64 - reference64).square().sum().item())
            available_route_count += int(
                route.allowed_mask[:, None, :, :]
                .expand_as(route.probabilities)
                .sum()
                .item()
            )
            selected_route_count += int(selection.sum().item())
            local_attention_parity &= bool(
                route.all_routes_operator_literal_parity_passed
                and route.all_routes_delta_literal_parity_passed
                and torch.equal(
                    route.reference_residual_delta,
                    mutation_outputs[event_id],
                )
            )
            factors.append(route)
        if teacher_reference_sq <= 0.0:
            raise ValueError("dense-teacher operator NRMSE has a zero denominator")
        teacher_nrmse = math.sqrt(teacher_error_sq / teacher_reference_sq)
        if not math.isfinite(teacher_nrmse):
            raise ValueError("dense-teacher operator NRMSE is nonfinite")

        all_route_masks = tuple(
            route.allowed_mask[:, None, :, :].expand_as(route.probabilities)
            for route in factors
        )
        with torch.no_grad():
            dense, sparse = _run_separate_batch1_dense_and_sparse_replays(
                executor=self.executor,
                input_ids=batch.model_inputs["input_ids"],
                sequence=prepared.sequence,
                all_route_masks=all_route_masks,
                sparse_masks=prepared.layer_masks,
                supervised_query=supervised_query,
            )
        dense_attention_parity = all(
            torch.equal(actual, route.reference_operator_output)
            for actual, route in zip(
                dense.attention_operator_outputs, factors, strict=True
            )
        ) and all(
            torch.equal(
                actual, mutation_outputs[f"mutation.{1 + 2 * layer}"]
            )
            for layer, actual in enumerate(dense.attention_residual_deltas)
        )
        mutation_delta_parity = all(
            torch.equal(
                actual, mutation_outputs[f"mutation.{index + 1}"]
            )
            for index, actual in enumerate(dense.mutation_residual_deltas)
        )
        baseline_mutation_carriers = _reconstruct_dense_residual_carriers(
            lexical_outputs[0],
            tuple(
                mutation_outputs[f"mutation.{index + 1}"]
                for index in range(36)
            ),
            prepared.sequence.query_valid_mask,
        )
        mutation_carrier_parity = all(
            torch.equal(
                actual,
                baseline_mutation_carriers[index],
            )
            for index, actual in enumerate(dense.mutation_carriers)
        )
        mutation_parity = mutation_delta_parity and mutation_carrier_parity
        baseline_valid = _valid_logits(observed.execution.logits, batch)[0]
        dense_valid = dense.logits[0, 0]
        sparse_valid = sparse.logits[0, 0]
        dense_endpoint_parity = bool(
            torch.equal(dense.output, observed.execution.output)
            and torch.equal(
                dense.normalized_output,
                observed.execution.normalized_output,
            )
            and torch.equal(dense_valid, baseline_valid)
        )
        target = int(batch.targets[0, supervised_query].item())
        target_tensor = torch.tensor([target], dtype=torch.int64)
        native_nll = float(
            F.cross_entropy(
                baseline_valid.unsqueeze(0), target_tensor, reduction="sum"
            ).item()
        )
        dense_nll = float(
            F.cross_entropy(
                dense_valid.unsqueeze(0), target_tensor, reduction="sum"
            ).item()
        )
        sparse_nll = float(
            F.cross_entropy(
                sparse_valid.unsqueeze(0), target_tensor, reduction="sum"
            ).item()
        )
        dense_replay_parity = bool(
            dense_endpoint_parity
            and dense_nll == native_nll
            and _smallest_index_argmax(dense_valid)
            == _smallest_index_argmax(baseline_valid)
        )
        if not (
            local_attention_parity
            and dense_attention_parity
            and mutation_parity
            and dense_replay_parity
        ):
            raise RuntimeError("prospective B all-route replay failed literal parity")
        terminal_cosine = _strict_cosine(
            observed.execution.normalized_output[0, supervised_query],
            sparse.normalized_output[0, supervised_query],
        )
        centered_nrmse = _centered_logit_nrmse(baseline_valid, sparse_valid)
        return AttentionEdgeFidelityBRowCapture(
            row_ordinal=row_ordinal,
            family_id=self.family_ids[row_ordinal],
            sequence_length=prepared.sequence.query_length,
            tokenization_sha256=prepared.tokenization_sha256,
            selection_catalog_sha256=prepared.selection_catalog_sha256,
            available_route_count=available_route_count,
            selected_route_count=selected_route_count,
            native_correct_target_nll=native_nll,
            sparse_correct_target_nll=sparse_nll,
            full_vocab_top1_agreement=(
                _smallest_index_argmax(sparse_valid)
                == _smallest_index_argmax(baseline_valid)
            ),
            terminal_supervised_vector_cosine=terminal_cosine,
            centered_logit_nrmse=centered_nrmse,
            dense_teacher_operator_nrmse=teacher_nrmse,
            dense_replay_literal_parity_passed=dense_replay_parity,
            all_route_attention_operator_delta_parity_passed=(
                local_attention_parity and dense_attention_parity
            ),
            mutation_boundary_literal_parity_passed=mutation_parity,
        )

    def capture(
        self,
    ) -> tuple[tuple[AttentionEdgeFidelityBRowCapture, ...], dict[str, object]]:
        rows = tuple(self._capture_row(row) for row in range(60))
        exact = _exact_control_receipt(
            control_kind="live_exact",
            compiler_metadata_sha256=self.compiler_metadata_sha256,
            source_independence_audit_sha256=(
                self.source_independence_audit_sha256
            ),
            native_first_row_valid_logits_sha256=(
                self.native_first_row_valid_logits_sha256
            ),
            exact_first_row_valid_logits_sha256=(
                self.exact_first_row_valid_logits_sha256
            ),
            source_model_fingerprint_before=(
                self.source_model_fingerprint_before
            ),
            source_model_fingerprint_after=self.source_model_fingerprint_after,
            source_execution_fingerprint_before=(
                self.source_execution_fingerprint_before
            ),
            source_execution_fingerprint_after=(
                self.source_execution_fingerprint_after
            ),
            source_model_references_released=(
                self.source_model_references_released
            ),
            all_route_row_count=len(rows),
        )
        return rows, exact


def run_attention_edge_fidelity_b(
    config: AttentionEdgeFidelityBLiveConfig = AttentionEdgeFidelityBLiveConfig(),
) -> dict[str, object]:
    """Run the single prospective B evaluation and publish it atomically."""

    runtime = AttentionEdgeFidelityBRuntime.load(config)
    rows, exact = runtime.capture()
    # Re-run the complete setup/source qualification after model execution and
    # before constructing or publishing any result.
    post_state, post_setup_file_sha256 = _checked_setup(config)
    if (
        post_state != runtime.setup_state
        or post_setup_file_sha256 != runtime.setup_file_sha256
    ):
        raise RuntimeError("B setup or source closure changed during execution")
    implementation = runtime.setup_state.get("implementation_receipt")
    if not isinstance(implementation, Mapping):
        raise ValueError("B setup implementation receipt is invalid")
    report = build_attention_edge_fidelity_b_report(
        rows,
        source_kind="model_capture",
        setup_sha256=str(runtime.setup_state["setup_sha256"]),
        setup_file_sha256=runtime.setup_file_sha256,
        implementation_bundle_sha256=str(implementation["bundle_sha256"]),
        implementation_commit_sha=str(implementation["commit_sha"]),
        implementation_tree_sha=str(implementation["tree_sha"]),
        source_model_fingerprint_sha256=SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_sha256=(
            PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        exact_control_receipt=exact,
        model_loaded=True,
        tokenizer_loaded=True,
    )
    return write_attention_edge_fidelity_b(
        report,
        config.output,
        setup_path=config.setup_path,
        repository_root=config.repository_root,
    )


def _compact_summary(report: Mapping[str, object]) -> dict[str, object]:
    aggregates = report["aggregate_statistics"]
    exact = report.get("exact_control_receipt")
    panel = report.get("panel_binding")
    gates = report.get("gate_outcomes")
    if (
        not isinstance(aggregates, Mapping)
        or not isinstance(exact, Mapping)
        or not isinstance(panel, Mapping)
        or not isinstance(gates, list)
    ):
        raise ValueError("prospective B report omitted aggregates")
    return {
        "schema": report["schema"],
        "role": report["role"],
        "source_kind": report["source_kind"],
        "row_count": panel["row_count"],
        "live_exact": bool(
            report["source_kind"] == "model_capture"
            and exact.get("control_kind") == "live_exact"
        ),
        "report_sha256": report["report_sha256"],
        "gates": gates,
        "outcome": (
            "pass"
            if report["all_six_preregistered_gates_passed"] is True
            else "fail"
        ),
        "absolute_delta_NLL_U95": aggregates[
            "macro_family_bootstrap_mean_absolute_delta_NLL_U95"
        ],
        "absolute_delta_NLL_p90": aggregates[
            "empirical_absolute_delta_NLL_p90"
        ],
        "top1_agreement_Wilson_L95": aggregates[
            "full_vocab_top1_agreement_Wilson_L95"
        ],
        "terminal_cosine_L95": aggregates[
            "macro_family_bootstrap_terminal_cosine_L95"
        ],
        "centered_logit_NRMSE_U95": aggregates[
            "macro_family_bootstrap_centered_logit_NRMSE_U95"
        ],
        "teacher_operator_NRMSE_U95": aggregates[
            "macro_family_bootstrap_teacher_operator_NRMSE_U95"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--validate", type=Path)
    parser.add_argument("--setup", type=Path, default=DEFAULT_B_SETUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_B_REPORT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.validate is not None:
        report = load_attention_edge_fidelity_b(
            arguments.validate,
            setup_path=arguments.setup,
        )
        print(json.dumps(_compact_summary(report), sort_keys=True))
        return 0
    config = AttentionEdgeFidelityBLiveConfig(
        setup_path=arguments.setup,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
    )
    if arguments.preflight:
        print(
            json.dumps(
                build_attention_edge_fidelity_b_preflight(config),
                sort_keys=True,
            )
        )
        return 0
    report = run_attention_edge_fidelity_b(config)
    print(json.dumps(_compact_summary(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
