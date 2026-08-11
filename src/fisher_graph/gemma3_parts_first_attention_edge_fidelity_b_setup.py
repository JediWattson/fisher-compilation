"""Issue the prospective-B attention-edge fidelity setup.

This module is deliberately standard-library-only.  It can authenticate and
qualify the frozen B panel, the retrospective A atlas, and the completed A
fidelity report without importing a tokenizer, Transformers, torch, or the
model-facing fidelity runner.  Issuance is possible only after an explicitly
authorized A report exists and the declared implementation closure is clean,
tracked, and byte-identical to ``HEAD``.

The B selector is fixed before B model execution.  It assigns every allowed
route the retrospective-A mean intrinsic Fisher energy for its
``(layer, query head, causal-distance bin)`` motif, ranks globally across
``(head, key)`` for each query, retains ``min(8, allowed routes)``, and
zero-fills every omitted route without renormalization.  B labels, target
gradients, example
or family IDs, choice order, and distractors cannot affect that mask.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Final, TypeVar


SETUP_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edge_fidelity_b_setup.v1"
)
SETUP_FORMAT_VERSION: Final = 1
SETUP_ID: Final = "gemma3-attention-edge-fidelity-B-setup-v1"
ISSUANCE_AUTHORIZATION: Final = "A_fidelity_reviewed_authorize_B_setup"

DEFAULT_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-attention-edge-fidelity-B-setup-v1.json"
)
DEFAULT_B_REPORT_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-attention-edge-fidelity-B-prospective-v1.json"
)

B_PANEL_PATH: Final = Path("examples/gemma3_attention_edge_fidelity_B_v1.json")
B_PANEL_ID: Final = "gemma3-attention-edge-fidelity-B-v1"
B_PANEL_FILE_SHA256: Final = (
    "1d8383f6c158fa2ea7e300c4722c5ae558667321fb091c79694d1e3918365364"
)
B_PANEL_SEMANTIC_SHA256: Final = (
    "167d9df0673c442316039265edd54fe1ba9b5906be0b84746683f244ed6df2f5"
)
B_METADATA_CATALOG_SHA256: Final = (
    "9d888eba31ffb2f7e613d86bb90722892c7fe0da3953589143825057f6e20ed1"
)
A_PANEL_PATH: Final = Path("examples/gemma3_downstream_retention_v1.json")
A_PANEL_ID: Final = "gemma3-single-token-forced-choice-v1"
A_PANEL_FILE_SHA256: Final = (
    "47d91ae000b4036d7744509bbbfec67aa0bd48bb1e1162fd6efa53974c23e6ff"
)
PANEL_SCHEMA: Final = "fisher_graph.gemma3_downstream_retention_panel"

A_ATLAS_PATH: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-attention-edge-atlas-A-retrospective-v1.json"
)
A_ATLAS_SCHEMA: Final = "fisher_graph.gemma3_parts_first.attention_edge_atlas.v1"
A_ATLAS_REPORT_SHA256: Final = (
    "5e815075fae7575817a1fee4661950a0e45e8a36c75d86d7570c3e4ea4eb09c7"
)
A_ATLAS_FILE_SHA256: Final = (
    "c5096391aa1d6600bcb5762f4528fa263f00bdc6539d7e2cb03b5476f1374a15"
)
A_ATLAS_IMPLEMENTATION_BUNDLE_SHA256: Final = (
    "0e5c0006a19399356fa8a37dbbeb52aca7c68b2452b24060b9ccbb38cbd288a1"
)
A_ATLAS_EXACT_CONTROL_RECEIPT_SHA256: Final = (
    "2761fff4205ec555693befc9c964faf323071b914861f00a7ebe0834f1c7f2c9"
)
A_MOTIF_SCORE_CATALOG_SHA256: Final = (
    "f179627a8a487515d98f033f7c0e9bdccdb0b3af4af021d1b9d74445687a2420"
)
A_ATLAS_FULL_ALLOWED_ROUTE_COUNT: Final = 473_328
A_ATLAS_MOTIF_CELL_COUNT: Final = 576
A_ATLAS_SUPPORTED_MOTIF_CELL_COUNT: Final = 360

A_FIDELITY_REPORT_PATH: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-attention-edge-fidelity-A-retrospective-v1.json"
)
A_FIDELITY_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edge_fidelity.v1"
)
A_FIDELITY_ABI: Final = (
    "baseline_global_head_key_topk_dynamic_all_layer_sparse_replay.v1"
)
A_FIDELITY_ROLE: Final = "A_retrospective_fidelity_diagnostic"
IMPLEMENTATION_RECEIPT_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.attention_edge_fidelity_b_sources.v1"
)

FAMILY_IDS: Final = (
    "category_completion",
    "integer_addition",
    "integer_comparison",
    "integer_subtraction",
    "lexical_antonym",
    "world_capitals",
)
DISTANCE_BINS: Final = (
    "0",
    "1",
    "2-3",
    "4-15",
    "16-63",
    "64-255",
    "256-511",
    "512+",
)
LAYER_COUNT: Final = 18
QUERY_HEAD_COUNT: Final = 4
PRIMARY_K: Final = 8
EXECUTION_CONSUMER_IMPLEMENTED: Final = True
BOOTSTRAP_SEED: Final = 20_260_811
BOOTSTRAP_RESAMPLE_COUNT: Final = 10_000
BOOTSTRAP_U95_ZERO_BASED_INDEX: Final = 9_499
BOOTSTRAP_L95_ZERO_BASED_INDEX: Final = 499
EMPIRICAL_P90_ZERO_BASED_INDEX: Final = 53
WILSON_Z_TWO_SIDED_95: Final = 1.959963984540054
MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL: Final = 0.0001
MATCHED_TARGET_ONLY_MAX_CENTERED_LOGIT_NRMSE: Final = 0.0001
_BOOTSTRAP_PRNG_DOMAIN = (
    b"fisher_graph:attention-edge-fidelity-b-bootstrap:sha256-counter-modulo:v1\0"
)

# These paths are the critical algorithmic and test closure.  The setup also
# pins the complete repository tree, so committed transitive helpers are bound
# even when they are not repeated in this human-auditable list.
IMPLEMENTATION_SOURCE_PATHS: Final = (
    "src/fisher_graph/gemma3_attention_generator.py",
    "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py",
    "src/fisher_graph/gemma3_joint_cross_block_compiler.py",
    "src/fisher_graph/gemma3_parts_first_attention_edges.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_experiment.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity_experiment.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity_b_setup.py",
    "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity_b_experiment.py",
)
IMPLEMENTATION_TEST_PATHS: Final = (
    "tests/test_gemma3_attention_generator.py",
    "tests/test_gemma3_joint_cross_block_chart_executor.py",
    "tests/test_gemma3_joint_cross_block_compiler.py",
    "tests/test_gemma3_parts_first_attention_edges.py",
    "tests/test_gemma3_parts_first_attention_edge_experiment.py",
    "tests/test_gemma3_parts_first_attention_edge_fidelity.py",
    "tests/test_gemma3_parts_first_attention_edge_fidelity_experiment.py",
    "tests/test_gemma3_attention_edge_fidelity_B_panel.py",
    "tests/test_gemma3_parts_first_attention_edge_fidelity_b_setup.py",
    "tests/test_gemma3_parts_first_attention_edge_fidelity_b_experiment.py",
)

_SETUP_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-setup:v1\0"
_PANEL_SEMANTIC_DOMAIN = b"fisher-graph:gemma3-downstream-panel:v1\0"
_PANEL_CATALOG_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-catalog:v1\0"
_MOTIF_CATALOG_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-motifs:v1\0"
_SOURCE_BUNDLE_DOMAIN = b"fisher_graph:attention-edge-fidelity-b-sources:v1\0"
_ATLAS_REPORT_DOMAIN = b"fisher_graph:parts-first-attention-edge-atlas:v1\0"
_A_FIDELITY_REPORT_DOMAIN = (
    b"fisher_graph:parts-first-attention-edge-fidelity:v1\0"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_MAX_PANEL_BYTES = 1 * 1024 * 1024
_MAX_SETUP_BYTES = 8 * 1024 * 1024
_MAX_ATLAS_BYTES = 64 * 1024 * 1024
_MAX_A_FIDELITY_BYTES = 256 * 1024 * 1024
_ReadResult = TypeVar("_ReadResult")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def primary_selected_route_count(allowed_global_route_count: int) -> int:
    """Return the frozen global support size for one layer/query target."""

    if type(allowed_global_route_count) is not int or allowed_global_route_count <= 0:
        raise ValueError("allowed_global_route_count must be a positive integer")
    return min(PRIMARY_K, allowed_global_route_count)


def bootstrap_resample_indices(
    replicate_ordinal: int,
    family_ordinal: int,
) -> tuple[int, ...]:
    """Return the ten frozen within-family bootstrap row ordinals."""

    if (
        type(replicate_ordinal) is not int
        or not 0 <= replicate_ordinal < BOOTSTRAP_RESAMPLE_COUNT
    ):
        raise ValueError("replicate_ordinal is outside the frozen bootstrap")
    if (
        type(family_ordinal) is not int
        or not 0 <= family_ordinal < len(FAMILY_IDS)
    ):
        raise ValueError("family_ordinal is outside FAMILY_IDS")
    prefix = (
        _BOOTSTRAP_PRNG_DOMAIN
        + BOOTSTRAP_SEED.to_bytes(8, "big")
        + replicate_ordinal.to_bytes(8, "big")
        + family_ordinal.to_bytes(4, "big")
    )
    return tuple(
        int.from_bytes(
            hashlib.sha256(prefix + draw_ordinal.to_bytes(4, "big")).digest()[:8],
            "big",
        )
        % 10
        for draw_ordinal in range(10)
    )


def wilson_top1_lower_bound(successes: int, observations: int = 60) -> float:
    """Frozen 95% Wilson lower endpoint without continuity correction."""

    if (
        type(successes) is not int
        or type(observations) is not int
        or observations <= 0
        or not 0 <= successes <= observations
    ):
        raise ValueError("Wilson success/observation counts are invalid")
    proportion = successes / observations
    z = WILSON_Z_TWO_SIDED_95
    denominator = 1.0 + z * z / observations
    numerator = (
        proportion
        + z * z / (2.0 * observations)
        - z
        * math.sqrt(
            proportion * (1.0 - proportion) / observations
            + z * z / (4.0 * observations * observations)
        )
    )
    return numerator / denominator


def _strict_json_loads(raw: bytes, *, label: str) -> object:
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
        return json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


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
    """Open every ancestor with ``O_NOFOLLOW`` and pin the parent inode."""

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


def _consume_regular(
    path: Path,
    *,
    label: str,
    byte_cap: int,
    consumer: Callable[[bytes], _ReadResult],
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> _ReadResult:
    """Read and consume one inode while its descriptor and parent stay pinned."""

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
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_mode,
                before.st_nlink,
            )
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_mode,
                opened.st_nlink,
            ) != identity:
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
            if (
                opened_after_read.st_dev,
                opened_after_read.st_ino,
                opened_after_read.st_size,
                opened_after_read.st_mtime_ns,
                opened_after_read.st_mode,
                opened_after_read.st_nlink,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_mode,
                opened.st_nlink,
            ):
                raise ValueError(f"{label} changed while reading")
            raw = b"".join(chunks)
            try:
                result = consumer(raw)
            finally:
                opened_after_consume = os.fstat(descriptor)
                if (
                    opened_after_consume.st_dev,
                    opened_after_consume.st_ino,
                    opened_after_consume.st_size,
                    opened_after_consume.st_mtime_ns,
                    opened_after_consume.st_mode,
                    opened_after_consume.st_nlink,
                ) != identity:
                    raise ValueError(
                        f"{label} changed while consuming parsed bytes"
                    )
                after = _lstat_at(parent_descriptor, absolute.name)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_mode,
                    after.st_nlink,
                ) != identity:
                    raise ValueError(
                        f"{label} path changed while consuming parsed bytes"
                    )
                _reopen_same_strict_parent(
                    absolute,
                    label=label,
                    expected_parent=opened_parent,
                )
            return result
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
    return _consume_regular(
        path,
        label=label,
        byte_cap=byte_cap,
        required_mode=required_mode,
        required_nlink=required_nlink,
        consumer=lambda raw: raw,
    )


def _parse_panel(raw: bytes, *, expected_id: str, label: str) -> dict[str, object]:
    value = _strict_json_loads(raw, label=label)
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "format_version",
        "panel_id",
        "choice_count",
        "examples",
    }:
        raise ValueError(f"{label} field inventory is invalid")
    if (
        value["schema"] != PANEL_SCHEMA
        or type(value["format_version"]) is not int
        or value["format_version"] != 1
        or value["panel_id"] != expected_id
        or type(value["choice_count"]) is not int
        or value["choice_count"] != 4
        or not isinstance(value["examples"], list)
    ):
        raise ValueError(f"{label} header is invalid")
    examples = value["examples"]
    for row in examples:
        if not isinstance(row, dict) or set(row) != {
            "example_id",
            "family_id",
            "prompt",
            "choices",
            "correct_choice",
        }:
            raise ValueError(f"{label} example fields are invalid")
        choices = row["choices"]
        if (
            not isinstance(row["example_id"], str)
            or not row["example_id"]
            or not isinstance(row["family_id"], str)
            or row["family_id"] not in FAMILY_IDS
            or not isinstance(row["prompt"], str)
            or not row["prompt"]
            or row["prompt"] != row["prompt"].strip()
            or not isinstance(choices, list)
            or len(choices) != 4
            or any(
                not isinstance(choice, str)
                or not choice
                or choice != choice.strip()
                for choice in choices
            )
            or len(set(choices)) != 4
            or type(row["correct_choice"]) is not int
            or not 0 <= row["correct_choice"] < 4
        ):
            raise ValueError(f"{label} example value is invalid")
    return value


def _content_key(row: Mapping[str, object]) -> tuple[object, ...]:
    family = row["family_id"]
    prompt = row["prompt"]
    if not isinstance(family, str) or not isinstance(prompt, str):
        raise ValueError("panel content row is invalid")
    numbers = tuple(int(item) for item in re.findall(r"\d+", prompt))
    if family == "integer_addition":
        if len(numbers) != 2:
            raise ValueError("addition prompt geometry is invalid")
        # Addition is commutative: 3+8 and 8+3 are the same content.
        return family, *sorted(numbers)
    if family == "integer_subtraction":
        if len(numbers) != 2:
            raise ValueError("subtraction prompt geometry is invalid")
        return family, *numbers
    if family == "integer_comparison":
        if len(numbers) != 4:
            raise ValueError("comparison prompt geometry is invalid")
        return family, *sorted(numbers)
    patterns = {
        "lexical_antonym": r"The opposite of ([A-Za-z]+) is$",
        "category_completion": r"category fact: (?:A |An )?([A-Za-z]+) is a kind of$",
        "world_capitals": r"The capital of ([A-Za-z]+) is$",
    }
    pattern = patterns.get(family)
    match = None if pattern is None else re.search(pattern, prompt)
    if match is None:
        raise ValueError(f"{family} prompt geometry is invalid")
    return family, match.group(1).casefold()


def _qualified_panel_bytes(
    *,
    b_bytes: bytes,
    a_bytes: bytes,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if hashlib.sha256(b_bytes).hexdigest() != B_PANEL_FILE_SHA256:
        raise ValueError("fidelity B panel bytes differ from the frozen panel")
    if hashlib.sha256(a_bytes).hexdigest() != A_PANEL_FILE_SHA256:
        raise ValueError("retrospective A panel bytes differ from the frozen panel")
    b_panel = _parse_panel(b_bytes, expected_id=B_PANEL_ID, label="fidelity B panel")
    a_panel = _parse_panel(
        a_bytes,
        expected_id=A_PANEL_ID,
        label="retrospective A panel",
    )
    b_rows = b_panel["examples"]
    a_rows = a_panel["examples"]
    assert isinstance(b_rows, list) and isinstance(a_rows, list)
    b_counts = Counter(str(row["family_id"]) for row in b_rows)
    a_counts = Counter(str(row["family_id"]) for row in a_rows)
    expected_counts = {family: 10 for family in FAMILY_IDS}
    if (
        len(b_rows) != 60
        or len(a_rows) != 60
        or b_counts != expected_counts
        or a_counts != expected_counts
    ):
        raise ValueError("A and B must each contain six families of ten rows")
    b_ids = tuple(str(row["example_id"]) for row in b_rows)
    a_ids = tuple(str(row["example_id"]) for row in a_rows)
    b_prompts = tuple(str(row["prompt"]) for row in b_rows)
    a_prompts = tuple(str(row["prompt"]) for row in a_rows)
    b_content = tuple(_content_key(row) for row in b_rows)
    a_content = tuple(_content_key(row) for row in a_rows)
    if len(set(b_ids)) != 60 or len(set(b_prompts)) != 60 or len(set(b_content)) != 60:
        raise ValueError("fidelity B row identities are not unique")
    if len(set(a_ids)) != 60 or len(set(a_prompts)) != 60 or len(set(a_content)) != 60:
        raise ValueError("retrospective A row identities are not unique")
    if not set(b_ids).isdisjoint(a_ids):
        raise ValueError("fidelity B example IDs overlap retrospective A")
    if not set(b_prompts).isdisjoint(a_prompts):
        raise ValueError("fidelity B prompts overlap retrospective A")
    if not set(b_content).isdisjoint(a_content):
        raise ValueError("fidelity B semantic content overlaps retrospective A")
    metadata_rows = [
        {
            "example_id": row["example_id"],
            "family_id": row["family_id"],
            "choice_count": len(row["choices"]),
        }
        for row in b_rows
    ]
    qualification = {
        "panel_schema": PANEL_SCHEMA,
        "b_panel_path": str(B_PANEL_PATH),
        "b_panel_id": B_PANEL_ID,
        "b_panel_file_sha256": B_PANEL_FILE_SHA256,
        "b_panel_semantic_sha256": _digest(
            _PANEL_SEMANTIC_DOMAIN,
            {
                "panel_id": B_PANEL_ID,
                "examples": b_rows,
            },
        ),
        "b_metadata_catalog_sha256": _digest(
            _PANEL_CATALOG_DOMAIN, metadata_rows
        ),
        "a_panel_path": str(A_PANEL_PATH),
        "a_panel_id": A_PANEL_ID,
        "a_panel_file_sha256": A_PANEL_FILE_SHA256,
        "example_count": 60,
        "choice_count": 4,
        "family_counts": [[family, b_counts[family]] for family in FAMILY_IDS],
        "example_ids_disjoint_from_A": True,
        "prompt_text_disjoint_from_A": True,
        "family_content_disjoint_from_A": True,
        "commutative_addition_content_normalized_before_disjointness": True,
        "prompt_text_persisted_in_setup": False,
        "choices_persisted_in_setup": False,
        "correct_choice_persisted_in_setup": False,
    }
    if (
        qualification["b_panel_semantic_sha256"] != B_PANEL_SEMANTIC_SHA256
        or qualification["b_metadata_catalog_sha256"]
        != B_METADATA_CATALOG_SHA256
    ):
        raise ValueError("fidelity B panel semantic hash drifted")
    return b_panel, a_panel, qualification


def _qualified_panel_pair(
    *,
    b_path: Path,
    a_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return _consume_regular(
        b_path,
        label="fidelity B panel",
        byte_cap=_MAX_PANEL_BYTES,
        consumer=lambda b_bytes: _consume_regular(
            a_path,
            label="retrospective A panel",
            byte_cap=_MAX_PANEL_BYTES,
            consumer=lambda a_bytes: _qualified_panel_bytes(
                b_bytes=b_bytes,
                a_bytes=a_bytes,
            ),
        ),
    )


def qualify_fidelity_b_panel(
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    """Authenticate B and prove its row/content disjointness from A."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    _, _, qualification = _qualified_panel_pair(
        b_path=root / B_PANEL_PATH,
        a_path=root / A_PANEL_PATH,
    )
    return qualification


def load_fidelity_b_panel(
    *,
    repository_root: Path | str | None = None,
) -> object:
    """Return a ``ForcedChoicePanel`` without changing legacy loader pins.

    The import is intentionally deferred.  Calling this function imports the
    existing dataclasses, but it does not call or monkeypatch the legacy
    model-facing panel loader and does not import the fidelity runner.
    """

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    b_panel, _, _ = _qualified_panel_pair(
        b_path=root / B_PANEL_PATH,
        a_path=root / A_PANEL_PATH,
    )
    from .gemma3_downstream_retention_experiment import (  # noqa: PLC0415
        ForcedChoiceExample,
        ForcedChoicePanel,
    )

    rows = b_panel["examples"]
    assert isinstance(rows, list)
    examples = tuple(
        ForcedChoiceExample(
            example_id=str(row["example_id"]),
            family_id=str(row["family_id"]),
            prompt=str(row["prompt"]),
            choices=tuple(str(choice) for choice in row["choices"]),
            correct_choice=int(row["correct_choice"]),
        )
        for row in rows
    )
    return ForcedChoicePanel(
        panel_id=B_PANEL_ID,
        examples=examples,
        file_sha256=B_PANEL_FILE_SHA256,
    )


def _validate_report_digest(
    value: Mapping[str, object], *, field: str, domain: bytes, label: str
) -> str:
    digest = _require_sha256(value.get(field), label=f"{label} semantic digest")
    payload = dict(value)
    payload.pop(field)
    if _digest(domain, payload) != digest:
        raise ValueError(f"{label} semantic digest mismatch")
    return digest


def _atlas_receipt_bytes(
    raw: bytes,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    file_digest = hashlib.sha256(raw).hexdigest()
    if file_digest != A_ATLAS_FILE_SHA256:
        raise ValueError("retrospective A atlas file hash drifted")
    value = _strict_json_loads(raw, label="retrospective A atlas")
    if not isinstance(value, dict):
        raise ValueError("retrospective A atlas must be a JSON object")
    semantic_digest = _validate_report_digest(
        value,
        field="report_sha256",
        domain=_ATLAS_REPORT_DOMAIN,
        label="retrospective A atlas",
    )
    exact = value.get("exact_control_receipt")
    config = value.get("config")
    motifs = value.get("distance_motifs")
    if (
        value.get("schema") != A_ATLAS_SCHEMA
        or type(value.get("format_version")) is not int
        or value.get("format_version") != 1
        or semantic_digest != A_ATLAS_REPORT_SHA256
        or value.get("implementation_bundle_sha256")
        != A_ATLAS_IMPLEMENTATION_BUNDLE_SHA256
        or not isinstance(exact, Mapping)
        or exact.get("receipt_sha256") != A_ATLAS_EXACT_CONTROL_RECEIPT_SHA256
        or value.get("retrospective_A_only") is not True
        or value.get("structure_only") is not True
        or value.get("attention_edge_confirmation_claimed") is not False
        or value.get("row_count") != 60
        or value.get("layer_count") != LAYER_COUNT
        or value.get("query_head_count") != QUERY_HEAD_COUNT
        or value.get("full_allowed_route_count")
        != A_ATLAS_FULL_ALLOWED_ROUTE_COUNT
        or not isinstance(config, Mapping)
        or config.get("top_routes_per_target") != 8
        or config.get("distance_bins") != list(DISTANCE_BINS)
        or not isinstance(motifs, list)
    ):
        raise ValueError("retrospective A atlas qualification failed")
    catalog: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for motif in motifs:
        if not isinstance(motif, Mapping):
            raise ValueError("retrospective A motif row is invalid")
        layer = motif.get("layer_ordinal")
        head = motif.get("query_head_ordinal")
        distance_bin = motif.get("distance_bin")
        route_count = motif.get("full_route_count")
        energy = motif.get("intrinsic_fisher_energy_sum")
        if (
            type(layer) is not int
            or not 0 <= layer < LAYER_COUNT
            or type(head) is not int
            or not 0 <= head < QUERY_HEAD_COUNT
            or distance_bin not in DISTANCE_BINS
            or type(route_count) is not int
            or route_count < 0
            or not isinstance(energy, (int, float))
            or isinstance(energy, bool)
            or not math.isfinite(float(energy))
            or float(energy) < 0.0
        ):
            raise ValueError("retrospective A motif score input is invalid")
        key = (layer, head, str(distance_bin))
        if key in seen:
            raise ValueError("retrospective A motif coordinates are duplicated")
        seen.add(key)
        if route_count == 0 and float(energy) != 0.0:
            raise ValueError("zero-count retrospective A motif has energy")
        score = None if route_count == 0 else float(energy) / route_count
        catalog.append(
            {
                "layer_ordinal": layer,
                "query_head_ordinal": head,
                "distance_bin": distance_bin,
                "full_route_count": route_count,
                "full_intrinsic_fisher_energy_hex": float(energy).hex(),
                "mean_intrinsic_fisher_energy_hex": (
                    None if score is None else score.hex()
                ),
                "supported": route_count > 0,
            }
        )
    expected_coordinates = {
        (layer, head, distance_bin)
        for layer in range(LAYER_COUNT)
        for head in range(QUERY_HEAD_COUNT)
        for distance_bin in DISTANCE_BINS
    }
    if seen != expected_coordinates or len(catalog) != A_ATLAS_MOTIF_CELL_COUNT:
        raise ValueError("retrospective A motif catalog is incomplete")
    catalog.sort(
        key=lambda row: (
            int(row["layer_ordinal"]),
            int(row["query_head_ordinal"]),
            DISTANCE_BINS.index(str(row["distance_bin"])),
        )
    )
    if (
        sum(int(row["full_route_count"]) for row in catalog)
        != A_ATLAS_FULL_ALLOWED_ROUTE_COUNT
    ):
        raise ValueError("retrospective A motif route ledger does not close")
    supported_count = sum(bool(row["supported"]) for row in catalog)
    if supported_count != A_ATLAS_SUPPORTED_MOTIF_CELL_COUNT:
        raise ValueError("retrospective A supported motif ledger drifted")
    catalog_digest = _digest(_MOTIF_CATALOG_DOMAIN, catalog)
    if catalog_digest != A_MOTIF_SCORE_CATALOG_SHA256:
        raise ValueError("retrospective A motif score catalog hash drifted")
    receipt = {
        "path": str(A_ATLAS_PATH),
        "schema": A_ATLAS_SCHEMA,
        "report_sha256": semantic_digest,
        "file_sha256": file_digest,
        "implementation_bundle_sha256": A_ATLAS_IMPLEMENTATION_BUNDLE_SHA256,
        "exact_control_receipt_sha256": A_ATLAS_EXACT_CONTROL_RECEIPT_SHA256,
        "row_count": 60,
        "layer_count": LAYER_COUNT,
        "query_head_count": QUERY_HEAD_COUNT,
        "full_allowed_route_count": A_ATLAS_FULL_ALLOWED_ROUTE_COUNT,
        "motif_score_catalog_sha256": catalog_digest,
        "motif_cell_count": len(catalog),
        "supported_motif_cell_count": supported_count,
    }
    return receipt, catalog


def _atlas_receipt(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    return _consume_regular(
        path,
        label="retrospective A atlas",
        byte_cap=_MAX_ATLAS_BYTES,
        required_mode=0o600,
        required_nlink=1,
        consumer=_atlas_receipt_bytes,
    )


def _a_fidelity_receipt_bytes(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, object]:
    expected = _require_sha256(
        expected_file_sha256, label="expected A fidelity report file"
    )
    file_digest = hashlib.sha256(raw).hexdigest()
    if file_digest != expected:
        raise ValueError("retrospective A fidelity report file hash drifted")
    value = _strict_json_loads(raw, label="retrospective A fidelity report")
    if not isinstance(value, dict):
        raise ValueError("retrospective A fidelity report must be an object")
    semantic_digest = _validate_report_digest(
        value,
        field="report_sha256",
        domain=_A_FIDELITY_REPORT_DOMAIN,
        label="retrospective A fidelity report",
    )
    family_counts = value.get("family_counts")
    expected_family_counts = [[family, 10] for family in FAMILY_IDS]
    required_true = (
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
        "clean_worktree_observed",
        "committed_source_identity_passed",
        "retrospective_A_only",
    )
    required_false = (
        "self_proves_live_fidelity",
        "selection_optimality_claimed",
        "fresh_panel_used",
        "attention_edge_confirmation_claimed",
        "semantic_interpretation_claimed",
        "compression_qualification_claimed",
        "parameter_savings_claimed",
        "mac_savings_claimed",
        "latency_claimed",
        "prompts_persisted",
        "token_ids_persisted",
        "raw_activations_persisted",
        "raw_gradients_persisted",
    )
    if (
        value.get("schema") != A_FIDELITY_SCHEMA
        or type(value.get("format_version")) is not int
        or value.get("format_version") != 1
        or value.get("abi") != A_FIDELITY_ABI
        or value.get("role") != A_FIDELITY_ROLE
        or value.get("source_kind") != "model_capture"
        or value.get("panel_id") != A_PANEL_ID
        or value.get("panel_file_sha256") != A_PANEL_FILE_SHA256
        or value.get("atlas_report_sha256") != A_ATLAS_REPORT_SHA256
        or value.get("atlas_implementation_bundle_sha256")
        != A_ATLAS_IMPLEMENTATION_BUNDLE_SHA256
        or value.get("exact_control_receipt_sha256")
        != A_ATLAS_EXACT_CONTROL_RECEIPT_SHA256
        or value.get("row_count") != 60
        or value.get("layer_count") != LAYER_COUNT
        or value.get("query_head_count") != QUERY_HEAD_COUNT
        or family_counts != expected_family_counts
        or not all(value.get(field) is True for field in required_true)
        or not all(value.get(field) is False for field in required_false)
    ):
        raise ValueError("retrospective A fidelity report qualification failed")
    commit = value.get("source_commit_sha")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("retrospective A fidelity source commit is invalid")
    source_field_map = {
        "src/fisher_graph/gemma3_attention_generator.py": (
            "attention_generator_implementation_sha256"
        ),
        "src/fisher_graph/gemma3_joint_cross_block_chart_executor.py": (
            "joint_executor_implementation_sha256"
        ),
        "src/fisher_graph/gemma3_joint_cross_block_compiler.py": (
            "joint_compiler_implementation_sha256"
        ),
        "src/fisher_graph/gemma3_parts_first_attention_edges.py": (
            "attention_edge_core_implementation_sha256"
        ),
        "src/fisher_graph/gemma3_parts_first_attention_edge_experiment.py": (
            "attention_edge_atlas_implementation_sha256"
        ),
        "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity.py": (
            "attention_edge_fidelity_kernel_implementation_sha256"
        ),
        "src/fisher_graph/gemma3_parts_first_attention_edge_fidelity_experiment.py": (
            "attention_edge_fidelity_implementation_sha256"
        ),
    }
    source_hashes: dict[str, str] = {}
    for source_path, field in source_field_map.items():
        source_hashes[source_path] = _require_sha256(
            value.get(field), label=f"A fidelity {field}"
        )
    return {
        "path": str(A_FIDELITY_REPORT_PATH),
        "schema": A_FIDELITY_SCHEMA,
        "abi": A_FIDELITY_ABI,
        "role": A_FIDELITY_ROLE,
        "report_sha256": semantic_digest,
        "file_sha256": file_digest,
        "source_commit_sha": commit,
        "source_file_sha256": source_hashes,
        "row_count": 60,
        "family_counts": [[family, 10] for family in FAMILY_IDS],
        "layer_count": LAYER_COUNT,
        "query_head_count": QUERY_HEAD_COUNT,
        "A_fidelity_curve_statistics_used_to_tune_B_selector_or_gates": False,
        "A_atlas_motif_statistics_are_frozen_B_selector_input": True,
        "A_report_is_issuance_prerequisite_and_engineering_evidence_only": True,
    }


def _a_fidelity_receipt(path: Path, *, expected_file_sha256: str) -> dict[str, object]:
    return _consume_regular(
        path,
        label="retrospective A fidelity report",
        byte_cap=_MAX_A_FIDELITY_BYTES,
        required_mode=0o600,
        required_nlink=1,
        consumer=lambda raw: _a_fidelity_receipt_bytes(
            raw,
            expected_file_sha256=expected_file_sha256,
        ),
    )


def _git_output(root: Path, arguments: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"git qualification failed for {' '.join(arguments)}"
        ) from error
    return result.stdout


def _source_receipt(root: Path) -> dict[str, object]:
    status_before = _git_output(
        root, ("status", "--porcelain", "--untracked-files=all")
    )
    if status_before:
        raise ValueError("B setup issuance requires a completely clean worktree")
    commit = _git_output(root, ("rev-parse", "HEAD")).decode("ascii").strip()
    tree = _git_output(
        root, ("rev-parse", f"{commit}^{{tree}}")
    ).decode("ascii").strip()
    if _COMMIT_RE.fullmatch(commit) is None or _COMMIT_RE.fullmatch(tree) is None:
        raise ValueError("B setup git identity is invalid")
    rows: list[dict[str, str]] = []
    for relative_text in (*IMPLEMENTATION_SOURCE_PATHS, *IMPLEMENTATION_TEST_PATHS):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("implementation source path is unsafe")
        working = _read_regular(
            root / relative,
            label=f"implementation source {relative_text}",
            byte_cap=8 * 1024 * 1024,
        )
        committed = _git_output(root, ("show", f"{commit}:{relative_text}"))
        if working != committed:
            raise ValueError(
                f"implementation source differs from HEAD: {relative_text}"
            )
        rows.append(
            {
                "path": relative_text,
                "file_sha256": hashlib.sha256(working).hexdigest(),
            }
        )
    for row in rows:
        observed = _read_regular(
            root / row["path"],
            label=f"implementation source recheck {row['path']}",
            byte_cap=8 * 1024 * 1024,
        )
        if hashlib.sha256(observed).hexdigest() != row["file_sha256"]:
            raise ValueError(
                f"implementation source changed after hashing: {row['path']}"
            )
    head_after = _git_output(root, ("rev-parse", "HEAD")).decode("ascii").strip()
    status_after = _git_output(
        root, ("status", "--porcelain", "--untracked-files=all")
    )
    if head_after != commit or status_after:
        raise ValueError("B setup source identity changed during qualification")
    bundle = _digest(
        _SOURCE_BUNDLE_DOMAIN,
        {"commit_sha": commit, "tree_sha": tree, "files": rows},
    )
    return {
        "schema": IMPLEMENTATION_RECEIPT_SCHEMA,
        "commit_sha": commit,
        "tree_sha": tree,
        "clean_worktree_observed": True,
        "all_declared_files_tracked_and_identical_to_HEAD": True,
        "repository_tree_pins_transitive_committed_dependencies": True,
        "HEAD_captured_once_then_rechecked": True,
        "tree_derived_from_captured_commit": True,
        "all_working_bytes_rechecked_after_hashing": True,
        "declared_file_count": len(rows),
        "files": rows,
        "bundle_sha256": bundle,
    }


def _frozen_contract() -> dict[str, object]:
    return {
        "prospective_role": "B_fresh_execution_protocol_only",
        "execution_consumer": {
            "implemented": EXECUTION_CONSUMER_IMPLEMENTED,
            "expected_source_path": (
                "src/fisher_graph/"
                "gemma3_parts_first_attention_edge_fidelity_b_experiment.py"
            ),
            "expected_test_path": (
                "tests/"
                "test_gemma3_parts_first_attention_edge_fidelity_b_experiment.py"
            ),
            "separate_consumer_audit_and_source_binding_required": True,
            "setup_or_issuance_alone_authorizes_model_execution": False,
        },
        "primary_condition": {
            "selector_id": "A_atlas_motif_mean_intrinsic_fisher_energy.v1",
            "score_formula": (
                "A_full_intrinsic_fisher_energy_sum/A_full_route_count"
            ),
            "score_coordinates": [
                "layer_ordinal",
                "query_head_ordinal",
                "causal_distance_bin",
            ],
            "A_motif_catalog_only": True,
            "B_target_gradient_used_for_selection": False,
            "B_probability_used_for_selection": False,
            "global_across_query_head_and_key": True,
            "maximum_routes_per_layer_query": PRIMARY_K,
            "selected_route_count_formula": (
                "min(8,allowed_global_head_key_route_count)"
            ),
            "sequence_length_domain": "integer_1_through_128_inclusive",
            "row_available_route_count_formula": (
                "18*4*sum_q_0_to_L_minus_1(q+1)=36*L*(L+1)"
            ),
            "row_selected_route_count_formula": (
                "18*sum_q_0_to_L_minus_1(min(8,4*(q+1)))=18*(8*L-4)"
            ),
            "row_route_count_formula_domain": "1<=L<=128",
            "query_ordinal_zero_allowed_route_count": QUERY_HEAD_COUNT,
            "query_ordinal_zero_selected_route_count": QUERY_HEAD_COUNT,
            "omitted_route_semantics": "zero_fill_without_renormalization",
            "fixed_order": [
                "descending_A_motif_mean_score",
                "ascending_query_head_ordinal",
                "ascending_key_ordinal",
            ],
            "zero_count_A_motif_policy": (
                "unsupported_and_abort_if_any_allowed_B_route_requires_cell"
            ),
            "B_rerank_forbidden": True,
            "B_refit_forbidden": True,
            "B_threshold_tuning_forbidden": True,
            "alternate_K_or_renormalized_primary_forbidden": True,
        },
        "metadata_firewall": {
            "selector_may_read": [
                "allowed_route_geometry",
                "layer_ordinal",
                "query_head_ordinal",
                "key_ordinal",
                "causal_distance_bin",
                "authenticated_A_motif_score_catalog",
            ],
            "B_example_id_excluded_from_selector": True,
            "B_family_id_excluded_from_selector": True,
            "B_choice_order_excluded_from_selector": True,
            "B_distractors_excluded_from_selector": True,
            "B_correct_choice_and_target_excluded_from_selector": True,
            "B_gradients_excluded_from_selector": True,
            "family_id_may_be_used_only_for_preregistered_aggregation": True,
            "correct_target_may_be_used_only_after_mask_freeze_for_metrics": True,
            "prompt_tokens_may_define_model_state_and_allowed_geometry_only": True,
        },
        "primary_metrics": {
            "delta_NLL": (
                "sparse_target_only_correct_target_full_vocab_NLL_minus_"
                "matched_dense_target_only_correct_target_full_vocab_NLL"
            ),
            "top1_agreement": (
                "sparse_target_only_full_vocab_argmax_equals_matched_dense_"
                "target_only_full_vocab_argmax"
            ),
            "argmax_tie_rule": (
                "lowest_vocabulary_token_ordinal_among_equal_maximum_logits"
            ),
            "terminal_cosine": (
                "cosine_at_the_single_supervised_terminal_hidden_vector"
            ),
            "centered_logit_NRMSE": (
                "RMS((sparse_target_only_logits-mean_sparse)-(matched_dense_"
                "target_only_logits-mean_matched_dense))/RMS(matched_dense_"
                "target_only_logits-mean_matched_dense)_at_the_single_"
                "supervised_query"
            ),
            "native_full_sequence_egress_role": (
                "literal_native_all_route_control_only"
            ),
            "gated_logit_metric_reference": (
                "matched_dense_target_only_tied_egress"
            ),
            "matched_target_only_egress_control": {
                "dense_input": (
                    "literal_dense_normalized_output_supervised_row"
                ),
                "sparse_input": (
                    "dynamic_sparse_normalized_output_supervised_row"
                ),
                "batch_size": 1,
                "query_length": 1,
                "dtype": "float32",
                "device": "cpu",
                "identical_tied_weights_and_final_softcap": True,
                "sparse_independent_reprojection_must_match": True,
                "all_row_validity_bounds": {
                    "maximum_absolute_native_to_matched_target_only_"
                    "delta_NLL_max": (
                        MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL
                    ),
                    "maximum_native_to_matched_target_only_centered_logit_"
                    "NRMSE_max": (
                        MATCHED_TARGET_ONLY_MAX_CENTERED_LOGIT_NRMSE
                    ),
                },
                "equality_at_validity_bound_passes": True,
                "validity_failure_semantics": (
                    "abort_before_interpreting_the_six_sparse_fidelity_gates"
                ),
            },
            "native_to_matched_target_only_drift_diagnostics": [
                "correct_target_delta_NLL",
                "centered_logit_NRMSE",
                "full_vocab_top1_agreement",
            ],
            "native_to_matched_target_only_argmax_mismatch_aborts": True,
            "native_to_matched_target_only_quantitative_bound_failure_aborts": (
                True
            ),
            "teacher_operator_NRMSE": (
                "sqrt(sum_over_all_18_layers_and_valid_queries_squared_L2_"
                "error_between_frozen_mask_zero_fill_operator_and_dense_"
                "teacher_operator_at_the_same_dense_baseline_normalized_"
                "input_P_V)/sum_over_all_18_layers_and_valid_queries_"
                "squared_L2_dense_teacher_operator)"
            ),
            "teacher_operator_NRMSE_is_teacher_forced_local_fidelity": True,
            "teacher_operator_NRMSE_state": (
                "dense_baseline_normalized_input_P_V_at_each_layer"
            ),
            "teacher_operator_NRMSE_mask": (
                "frozen_A_motif_mask_applied_at_that_same_teacher_state_"
                "with_zero_fill_without_renormalization"
            ),
            "teacher_operator_NRMSE_uses_dynamic_sparse_state": False,
            "dynamic_sparse_propagation_is_measured_separately_by": [
                "terminal_supervised_vector_cosine",
                "centered_logit_NRMSE",
                "delta_NLL",
                "full_vocab_top1_agreement",
            ],
            "zero_denominator_or_nonfinite_metric_aborts": True,
        },
        "aggregation": {
            "families": list(FAMILY_IDS),
            "examples_per_family": 10,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resample_count": BOOTSTRAP_RESAMPLE_COUNT,
            "bootstrap_unit": (
                "paired_whole_example_resampled_with_replacement_within_family"
            ),
            "bootstrap_statistic": (
                "equal_weight_macro_mean_of_six_within_family_means"
            ),
            "bootstrap_PRNG": {
                "algorithm": "sha256_counter_modulo.v1",
                "domain_hex": _BOOTSTRAP_PRNG_DOMAIN.hex(),
                "seed_encoding": "unsigned_64_bit_big_endian",
                "replicate_ordinals": "0_through_9999_inclusive",
                "family_ordinals": "FAMILY_IDS_order_0_through_5",
                "draw_ordinals": "0_through_9_inclusive",
                "digest_input": (
                    "domain||seed_u64_be||replicate_u64_be||family_u32_be||"
                    "draw_u32_be"
                ),
                "index_formula": (
                    "uint64_big_endian(first_8_SHA256_bytes)_modulo_10"
                ),
            },
            "U95_order_statistic": {
                "tail": "upper",
                "rule": "sorted_values[ceil(0.95*10000)-1]",
                "zero_based_index": BOOTSTRAP_U95_ZERO_BASED_INDEX,
            },
            "L95_order_statistic": {
                "tail": "lower",
                "rule": "sorted_values[ceil(0.05*10000)-1]",
                "zero_based_index": BOOTSTRAP_L95_ZERO_BASED_INDEX,
            },
            "empirical_p90_order_statistic": {
                "rule": "sorted_60_values[ceil(0.90*60)-1]",
                "zero_based_index": EMPIRICAL_P90_ZERO_BASED_INDEX,
            },
            "Wilson_top1_lower_bound": {
                "confidence": "two_sided_95_percent_lower_endpoint",
                "z": WILSON_Z_TWO_SIDED_95,
                "formula": (
                    "(p+z^2/(2n)-z*sqrt(p*(1-p)/n+z^2/(4n^2)))"
                    "/(1+z^2/n)"
                ),
                "continuity_correction": False,
                "n": 60,
                "success": (
                    "sparse_and_matched_dense_target_only_argmax_token_"
                    "ordinals_are_equal"
                ),
            },
            "all_six_gates_are_conjunctive": True,
            "equality_at_threshold_passes": True,
        },
        "gates": {
            "macro_family_bootstrap_mean_absolute_delta_NLL_U95_max": 0.05,
            "empirical_absolute_delta_NLL_p90_max": 0.25,
            "full_vocab_top1_agreement_Wilson_L95_min": 0.90,
            "macro_family_bootstrap_terminal_cosine_L95_min": 0.995,
            "macro_family_bootstrap_centered_logit_NRMSE_U95_max": 0.05,
            "macro_family_bootstrap_teacher_operator_NRMSE_U95_max": 0.15,
        },
        "claim_boundary": {
            "if_a_future_separately_audited_consumer_passes_all_gates_supports": (
                "the_frozen_gradient_free_A_motif_global_K8_cap_zero_fill_"
                "procedure_"
                "preserved_measured_behavior_under_the_matched_target_only_"
                "egress_control_on_this_B_panel"
            ),
            "setup_contains_B_result": False,
            "setup_claims_B_runtime_exists": EXECUTION_CONSUMER_IMPLEMENTED,
            "setup_or_issuance_authorizes_model_run": False,
            "matched_target_only_reference_validity_required": True,
            "semantic_interpretation_claimed": False,
            "selection_optimality_claimed": False,
            "compression_qualified": False,
            "parameter_savings_claimed": False,
            "MAC_savings_claimed": False,
            "latency_savings_claimed": False,
            "input_only_generalization_beyond_this_panel_claimed": False,
        },
    }


def setup_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("setup_sha256", None)
    return _digest(_SETUP_DOMAIN, payload)


def build_setup(
    *,
    panel_qualification: Mapping[str, object],
    atlas_receipt: Mapping[str, object],
    motif_score_catalog: Sequence[Mapping[str, object]],
    a_fidelity_receipt: Mapping[str, object],
    implementation_receipt: Mapping[str, object],
) -> dict[str, object]:
    state: dict[str, object] = {
        "schema": SETUP_SCHEMA,
        "format_version": SETUP_FORMAT_VERSION,
        "setup_id": SETUP_ID,
        "panel_qualification": json.loads(canonical_json_bytes(panel_qualification)),
        "A_atlas_receipt": json.loads(canonical_json_bytes(atlas_receipt)),
        "A_motif_score_catalog": json.loads(
            canonical_json_bytes(list(motif_score_catalog))
        ),
        "A_fidelity_receipt": json.loads(
            canonical_json_bytes(a_fidelity_receipt)
        ),
        "implementation_receipt": json.loads(
            canonical_json_bytes(implementation_receipt)
        ),
        "frozen_protocol": _frozen_contract(),
        "output_contract": {
            "default_setup_artifact_path": str(DEFAULT_OUTPUT),
            "planned_B_report_path": str(DEFAULT_B_REPORT_OUTPUT),
            "B_report_schema_and_writer_implemented": (
                EXECUTION_CONSUMER_IMPLEMENTED
            ),
            "future_B_report_must_bind_this_setup_sha256": True,
            "future_B_report_must_bind_this_setup_file_sha256": True,
            "future_B_report_must_bind_exact_implementation_bundle_sha256": True,
            "future_B_report_no_clobber_mode_0600_required": True,
        },
        "issuance_receipt": {
            "explicit_post_A_authorization_present": True,
            "B_panel_JSON_opened_for_schema_and_disjointness": True,
            "B_prompt_text_and_choices_persisted_in_setup": False,
            "B_tokenizer_loaded": False,
            "B_tokenization_performed": False,
            "tensor_library_used_by_issuer": False,
            "model_weights_loaded": False,
            "B_model_execution_performed": False,
            "B_gradients_computed": False,
            "B_selector_applied": False,
            "B_result_present": False,
            "execution_consumer_implemented": EXECUTION_CONSUMER_IMPLEMENTED,
            "issuance_alone_authorizes_B_model_run": False,
            "separately_audited_B_consumer_binding_required": True,
        },
    }
    state["setup_sha256"] = setup_sha256(state)
    return validate_setup(state)


def _validate_dynamic_state(value: Mapping[str, object]) -> None:
    if set(value) != {
        "schema",
        "format_version",
        "setup_id",
        "panel_qualification",
        "A_atlas_receipt",
        "A_motif_score_catalog",
        "A_fidelity_receipt",
        "implementation_receipt",
        "frozen_protocol",
        "output_contract",
        "issuance_receipt",
        "setup_sha256",
    }:
        raise ValueError("B setup field inventory is invalid")
    if (
        value["schema"] != SETUP_SCHEMA
        or type(value["format_version"]) is not int
        or value["format_version"] != SETUP_FORMAT_VERSION
        or value["setup_id"] != SETUP_ID
        or canonical_json_bytes(value["frozen_protocol"])
        != canonical_json_bytes(_frozen_contract())
    ):
        raise ValueError("B setup frozen protocol drifted")
    qualification = value["panel_qualification"]
    atlas = value["A_atlas_receipt"]
    fidelity = value["A_fidelity_receipt"]
    implementation = value["implementation_receipt"]
    catalog = value["A_motif_score_catalog"]
    if not all(
        isinstance(item, Mapping)
        for item in (qualification, atlas, fidelity, implementation)
    ) or not isinstance(catalog, list):
        raise ValueError("B setup dynamic receipts are invalid")
    if set(qualification) != {
        "panel_schema",
        "b_panel_path",
        "b_panel_id",
        "b_panel_file_sha256",
        "b_panel_semantic_sha256",
        "b_metadata_catalog_sha256",
        "a_panel_path",
        "a_panel_id",
        "a_panel_file_sha256",
        "example_count",
        "choice_count",
        "family_counts",
        "example_ids_disjoint_from_A",
        "prompt_text_disjoint_from_A",
        "family_content_disjoint_from_A",
        "commutative_addition_content_normalized_before_disjointness",
        "prompt_text_persisted_in_setup",
        "choices_persisted_in_setup",
        "correct_choice_persisted_in_setup",
    }:
        raise ValueError("B setup panel qualification fields drifted")
    if (
        qualification.get("panel_schema") != PANEL_SCHEMA
        or qualification.get("b_panel_path") != str(B_PANEL_PATH)
        or qualification.get("b_panel_id") != B_PANEL_ID
        or qualification.get("b_panel_file_sha256") != B_PANEL_FILE_SHA256
        or qualification.get("b_panel_semantic_sha256")
        != B_PANEL_SEMANTIC_SHA256
        or qualification.get("b_metadata_catalog_sha256")
        != B_METADATA_CATALOG_SHA256
        or qualification.get("a_panel_path") != str(A_PANEL_PATH)
        or qualification.get("a_panel_id") != A_PANEL_ID
        or qualification.get("a_panel_file_sha256") != A_PANEL_FILE_SHA256
        or type(qualification.get("example_count")) is not int
        or qualification.get("example_count") != 60
        or type(qualification.get("choice_count")) is not int
        or qualification.get("choice_count") != 4
        or qualification.get("family_counts")
        != [[family, 10] for family in FAMILY_IDS]
        or qualification.get("example_ids_disjoint_from_A") is not True
        or qualification.get("prompt_text_disjoint_from_A") is not True
        or qualification.get("family_content_disjoint_from_A") is not True
        or qualification.get(
            "commutative_addition_content_normalized_before_disjointness"
        )
        is not True
        or qualification.get("prompt_text_persisted_in_setup") is not False
        or qualification.get("choices_persisted_in_setup") is not False
        or qualification.get("correct_choice_persisted_in_setup") is not False
        or _SHA256_RE.fullmatch(
            str(qualification.get("b_panel_semantic_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(qualification.get("b_metadata_catalog_sha256", ""))
        )
        is None
    ):
        raise ValueError("B setup panel qualification drifted")
    if set(atlas) != {
        "path",
        "schema",
        "report_sha256",
        "file_sha256",
        "implementation_bundle_sha256",
        "exact_control_receipt_sha256",
        "row_count",
        "layer_count",
        "query_head_count",
        "full_allowed_route_count",
        "motif_score_catalog_sha256",
        "motif_cell_count",
        "supported_motif_cell_count",
    }:
        raise ValueError("B setup A-atlas receipt fields drifted")
    if (
        atlas.get("path") != str(A_ATLAS_PATH)
        or atlas.get("schema") != A_ATLAS_SCHEMA
        or atlas.get("report_sha256") != A_ATLAS_REPORT_SHA256
        or atlas.get("file_sha256") != A_ATLAS_FILE_SHA256
        or atlas.get("implementation_bundle_sha256")
        != A_ATLAS_IMPLEMENTATION_BUNDLE_SHA256
        or atlas.get("exact_control_receipt_sha256")
        != A_ATLAS_EXACT_CONTROL_RECEIPT_SHA256
        or type(atlas.get("row_count")) is not int
        or atlas.get("row_count") != 60
        or type(atlas.get("layer_count")) is not int
        or atlas.get("layer_count") != LAYER_COUNT
        or type(atlas.get("query_head_count")) is not int
        or atlas.get("query_head_count") != QUERY_HEAD_COUNT
        or type(atlas.get("full_allowed_route_count")) is not int
        or atlas.get("full_allowed_route_count")
        != A_ATLAS_FULL_ALLOWED_ROUTE_COUNT
        or type(atlas.get("motif_cell_count")) is not int
        or atlas.get("motif_cell_count") != A_ATLAS_MOTIF_CELL_COUNT
        or type(atlas.get("supported_motif_cell_count")) is not int
        or atlas.get("supported_motif_cell_count")
        != A_ATLAS_SUPPORTED_MOTIF_CELL_COUNT
        or atlas.get("motif_score_catalog_sha256")
        != A_MOTIF_SCORE_CATALOG_SHA256
        or _digest(_MOTIF_CATALOG_DOMAIN, catalog)
        != A_MOTIF_SCORE_CATALOG_SHA256
    ):
        raise ValueError("B setup A-atlas receipt drifted")
    if set(fidelity) != {
        "path",
        "schema",
        "abi",
        "role",
        "report_sha256",
        "file_sha256",
        "source_commit_sha",
        "source_file_sha256",
        "row_count",
        "family_counts",
        "layer_count",
        "query_head_count",
        "A_fidelity_curve_statistics_used_to_tune_B_selector_or_gates",
        "A_atlas_motif_statistics_are_frozen_B_selector_input",
        "A_report_is_issuance_prerequisite_and_engineering_evidence_only",
    }:
        raise ValueError("B setup A-fidelity receipt fields drifted")
    if (
        fidelity.get("path") != str(A_FIDELITY_REPORT_PATH)
        or fidelity.get("schema") != A_FIDELITY_SCHEMA
        or fidelity.get("abi") != A_FIDELITY_ABI
        or fidelity.get("role") != A_FIDELITY_ROLE
        or _SHA256_RE.fullmatch(str(fidelity.get("report_sha256", ""))) is None
        or _SHA256_RE.fullmatch(str(fidelity.get("file_sha256", ""))) is None
        or _COMMIT_RE.fullmatch(str(fidelity.get("source_commit_sha", ""))) is None
        or type(fidelity.get("row_count")) is not int
        or fidelity.get("row_count") != 60
        or fidelity.get("family_counts")
        != [[family, 10] for family in FAMILY_IDS]
        or type(fidelity.get("layer_count")) is not int
        or fidelity.get("layer_count") != LAYER_COUNT
        or type(fidelity.get("query_head_count")) is not int
        or fidelity.get("query_head_count") != QUERY_HEAD_COUNT
        or fidelity.get(
            "A_fidelity_curve_statistics_used_to_tune_B_selector_or_gates"
        )
        is not False
        or fidelity.get("A_atlas_motif_statistics_are_frozen_B_selector_input")
        is not True
        or fidelity.get(
            "A_report_is_issuance_prerequisite_and_engineering_evidence_only"
        )
        is not True
    ):
        raise ValueError("B setup A-fidelity receipt drifted")
    if set(implementation) != {
        "schema",
        "commit_sha",
        "tree_sha",
        "clean_worktree_observed",
        "all_declared_files_tracked_and_identical_to_HEAD",
        "repository_tree_pins_transitive_committed_dependencies",
        "HEAD_captured_once_then_rechecked",
        "tree_derived_from_captured_commit",
        "all_working_bytes_rechecked_after_hashing",
        "declared_file_count",
        "files",
        "bundle_sha256",
    }:
        raise ValueError("B setup implementation receipt fields drifted")
    files = implementation.get("files")
    expected_paths = [*IMPLEMENTATION_SOURCE_PATHS, *IMPLEMENTATION_TEST_PATHS]
    if (
        implementation.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA
        or implementation.get("clean_worktree_observed") is not True
        or implementation.get(
            "all_declared_files_tracked_and_identical_to_HEAD"
        )
        is not True
        or implementation.get(
            "repository_tree_pins_transitive_committed_dependencies"
        )
        is not True
        or implementation.get("HEAD_captured_once_then_rechecked") is not True
        or implementation.get("tree_derived_from_captured_commit") is not True
        or implementation.get(
            "all_working_bytes_rechecked_after_hashing"
        )
        is not True
        or type(implementation.get("declared_file_count")) is not int
        or implementation.get("declared_file_count") != len(expected_paths)
        or _COMMIT_RE.fullmatch(str(implementation.get("commit_sha", ""))) is None
        or _COMMIT_RE.fullmatch(str(implementation.get("tree_sha", ""))) is None
        or not isinstance(files, list)
        or [row.get("path") for row in files if isinstance(row, Mapping)]
        != expected_paths
        or len(files) != len(expected_paths)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "file_sha256"}
            or _SHA256_RE.fullmatch(str(row.get("file_sha256", ""))) is None
            for row in files
        )
    ):
        raise ValueError("B setup implementation receipt drifted")
    expected_bundle = _digest(
        _SOURCE_BUNDLE_DOMAIN,
        {
            "commit_sha": implementation["commit_sha"],
            "tree_sha": implementation["tree_sha"],
            "files": files,
        },
    )
    if implementation.get("bundle_sha256") != expected_bundle:
        raise ValueError("B setup implementation bundle drifted")
    if fidelity["source_commit_sha"] != implementation["commit_sha"]:
        raise ValueError("A fidelity and B setup commits differ")
    reported_source_hashes = fidelity.get("source_file_sha256")
    expected_reported_source_paths = set(IMPLEMENTATION_SOURCE_PATHS[:7])
    actual_source_hashes = {
        row["path"]: row["file_sha256"] for row in files if isinstance(row, Mapping)
    }
    if (
        not isinstance(reported_source_hashes, Mapping)
        or not reported_source_hashes
        or set(reported_source_hashes) != expected_reported_source_paths
        or any(
            _SHA256_RE.fullmatch(str(digest)) is None
            or actual_source_hashes.get(path) != digest
            for path, digest in reported_source_hashes.items()
        )
    ):
        raise ValueError("A fidelity source hashes differ from B setup sources")
    if len(catalog) != A_ATLAS_MOTIF_CELL_COUNT:
        raise ValueError("B setup motif catalog is incomplete")
    seen: set[tuple[int, int, str]] = set()
    expected_order = [
        (layer, head, distance_bin)
        for layer in range(LAYER_COUNT)
        for head in range(QUERY_HEAD_COUNT)
        for distance_bin in DISTANCE_BINS
    ]
    for index, row in enumerate(catalog):
        if not isinstance(row, Mapping):
            raise ValueError("B setup motif entry is invalid")
        if set(row) != {
            "layer_ordinal",
            "query_head_ordinal",
            "distance_bin",
            "full_route_count",
            "full_intrinsic_fisher_energy_hex",
            "mean_intrinsic_fisher_energy_hex",
            "supported",
        }:
            raise ValueError("B setup motif entry fields drifted")
        key = (
            row.get("layer_ordinal"),
            row.get("query_head_ordinal"),
            row.get("distance_bin"),
        )
        if (
            type(key[0]) is not int
            or not 0 <= key[0] < LAYER_COUNT
            or type(key[1]) is not int
            or not 0 <= key[1] < QUERY_HEAD_COUNT
            or key[2] not in DISTANCE_BINS
            or key in seen
            or key != expected_order[index]
        ):
            raise ValueError("B setup motif coordinate is invalid")
        seen.add(key)
        route_count = row.get("full_route_count")
        energy_hex = row.get("full_intrinsic_fisher_energy_hex")
        mean_hex = row.get("mean_intrinsic_fisher_energy_hex")
        try:
            energy = float.fromhex(energy_hex) if isinstance(energy_hex, str) else -1.0
        except ValueError as error:
            raise ValueError("B setup motif energy encoding is invalid") from error
        if (
            type(route_count) is not int
            or route_count < 0
            or not math.isfinite(energy)
            or energy < 0.0
            or row.get("supported") is not (route_count > 0)
        ):
            raise ValueError("B setup motif score inputs are invalid")
        if route_count == 0:
            if energy != 0.0 or mean_hex is not None:
                raise ValueError("B setup unsupported motif score drifted")
        elif (
            not isinstance(mean_hex, str)
            or mean_hex != (energy / route_count).hex()
        ):
            raise ValueError("B setup motif mean formula drifted")
    if (
        sum(int(row["full_route_count"]) for row in catalog)
        != A_ATLAS_FULL_ALLOWED_ROUTE_COUNT
        or sum(bool(row["supported"]) for row in catalog)
        != A_ATLAS_SUPPORTED_MOTIF_CELL_COUNT
    ):
        raise ValueError("B setup motif count ledger drifted")
    output = value["output_contract"]
    issuance = value["issuance_receipt"]
    if not isinstance(output, Mapping) or not isinstance(issuance, Mapping):
        raise ValueError("B setup lifecycle receipts are invalid")
    if canonical_json_bytes(output) != canonical_json_bytes({
        "default_setup_artifact_path": str(DEFAULT_OUTPUT),
        "planned_B_report_path": str(DEFAULT_B_REPORT_OUTPUT),
        "B_report_schema_and_writer_implemented": (
            EXECUTION_CONSUMER_IMPLEMENTED
        ),
        "future_B_report_must_bind_this_setup_sha256": True,
        "future_B_report_must_bind_this_setup_file_sha256": True,
        "future_B_report_must_bind_exact_implementation_bundle_sha256": True,
        "future_B_report_no_clobber_mode_0600_required": True,
    }):
        raise ValueError("B setup output contract drifted")
    if canonical_json_bytes(issuance) != canonical_json_bytes({
        "explicit_post_A_authorization_present": True,
        "B_panel_JSON_opened_for_schema_and_disjointness": True,
        "B_prompt_text_and_choices_persisted_in_setup": False,
        "B_tokenizer_loaded": False,
        "B_tokenization_performed": False,
        "tensor_library_used_by_issuer": False,
        "model_weights_loaded": False,
        "B_model_execution_performed": False,
        "B_gradients_computed": False,
        "B_selector_applied": False,
        "B_result_present": False,
        "execution_consumer_implemented": EXECUTION_CONSUMER_IMPLEMENTED,
        "issuance_alone_authorizes_B_model_run": False,
        "separately_audited_B_consumer_binding_required": True,
    }):
        raise ValueError("B setup issuance boundary drifted")
    if value["setup_sha256"] != setup_sha256(value):
        raise ValueError("B setup semantic digest mismatch")


def validate_setup(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("B setup must be a mapping")
    try:
        canonical = json.loads(canonical_json_bytes(dict(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("B setup is not canonical JSON") from error
    if not isinstance(canonical, dict):
        raise ValueError("B setup must be a JSON object")
    _validate_dynamic_state(canonical)
    return canonical


def _unlink_owned_at(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
) -> bool:
    """Unlink ``name`` only when it still denotes the issuer-owned inode."""

    try:
        observed = _lstat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(observed.st_mode) or _inode_identity(observed) != identity:
        return False
    os.unlink(name, dir_fd=directory_descriptor)
    return True


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("B setup temporary write made no progress")
        view = view[written:]


def _read_exact_descriptor(descriptor: int, expected_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("B setup publication inode was truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise RuntimeError("B setup publication inode grew unexpectedly")
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


def _is_exact_default_setup_destination(path: Path) -> bool:
    return Path(os.path.abspath(path)) == Path(os.path.abspath(DEFAULT_OUTPUT))


def _is_default_setup_parent(opened_parent: os.stat_result) -> bool:
    try:
        _, descriptor, canonical_parent = _open_strict_parent_directory(
            DEFAULT_OUTPUT,
            label="canonical B setup destination",
            create=False,
        )
    except FileNotFoundError:
        return False
    try:
        return _inode_identity(opened_parent) == _inode_identity(canonical_parent)
    finally:
        os.close(descriptor)


def _reject_default_setup_path_alias(path: Path, *, label: str) -> Path:
    absolute, descriptor, opened_parent = _open_strict_parent_directory(
        path,
        label=label,
        create=False,
    )
    try:
        if (
            not _is_exact_default_setup_destination(absolute)
            and _is_default_setup_parent(opened_parent)
        ):
            raise ValueError(
                "canonical B setup path must use its exact non-aliased spelling"
            )
        return absolute
    finally:
        os.close(descriptor)


def write_setup(
    value: Mapping[str, object], destination: Path | str
) -> dict[str, object]:
    state = validate_setup(value)
    exact_default = _is_exact_default_setup_destination(Path(destination))
    path, directory_descriptor, opened_parent = _open_strict_parent_directory(
        Path(destination),
        label="B setup destination",
        create=True,
    )
    if not exact_default and _is_default_setup_parent(opened_parent):
        os.close(directory_descriptor)
        raise ValueError(
            "canonical B setup path must use its exact non-aliased spelling"
        )
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
            raise FileExistsError("could not allocate unique B setup temporary")
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        owned_identity = _inode_identity(initial)
        _verify_publication_inode(
            initial,
            identity=owned_identity,
            expected_size=0,
            expected_nlink=1,
            label="B setup temporary",
        )
        named_initial = _lstat_at(directory_descriptor, temporary_name)
        _verify_publication_inode(
            named_initial,
            identity=owned_identity,
            expected_size=0,
            expected_nlink=1,
            label="B setup named temporary",
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        fsynced = os.fstat(descriptor)
        _verify_publication_inode(
            fsynced,
            identity=owned_identity,
            expected_size=len(payload),
            expected_nlink=1,
            label="B setup fsynced temporary",
        )
        named_fsynced = _lstat_at(directory_descriptor, temporary_name)
        _verify_publication_inode(
            named_fsynced,
            identity=owned_identity,
            expected_size=len(payload),
            expected_nlink=1,
            label="B setup named fsynced temporary",
        )
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        linked_descriptor = os.fstat(descriptor)
        linked_temporary = _lstat_at(directory_descriptor, temporary_name)
        linked_destination = _lstat_at(directory_descriptor, path.name)
        for label, observed in (
            ("B setup linked descriptor", linked_descriptor),
            ("B setup linked temporary", linked_temporary),
            ("B setup linked destination", linked_destination),
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
            raise RuntimeError("B setup temporary identity changed before unlink")
        temporary_name = None
        os.fsync(directory_descriptor)
        final_descriptor = os.fstat(descriptor)
        final_destination = _lstat_at(directory_descriptor, path.name)
        for label, observed in (
            ("B setup final descriptor", final_descriptor),
            ("B setup final destination", final_destination),
        ):
            _verify_publication_inode(
                observed,
                identity=owned_identity,
                expected_size=len(payload),
                expected_nlink=1,
                label=label,
            )
        if load_setup(path) != state:
            raise RuntimeError("B setup strict reload drifted")
        post_load = _lstat_at(directory_descriptor, path.name)
        _verify_publication_inode(
            post_load,
            identity=owned_identity,
            expected_size=len(payload),
            expected_nlink=1,
            label="B setup post-load destination",
        )
        if _read_exact_descriptor(descriptor, len(payload)) != payload:
            raise RuntimeError("B setup publication bytes changed after reload")
        _reopen_same_strict_parent(
            path,
            label="B setup destination",
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


def load_setup(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
    a_atlas_path: Path | str | None = None,
) -> dict[str, object]:
    path = _reject_default_setup_path_alias(
        Path(source),
        label="attention-edge fidelity B setup",
    )
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )

    def consume(raw: bytes) -> dict[str, object]:
        value = _strict_json_loads(raw, label="attention-edge fidelity B setup")
        if not isinstance(value, Mapping):
            raise ValueError("attention-edge fidelity B setup must be an object")
        state = validate_setup(value)
        atlas, catalog = _atlas_receipt(
            root / A_ATLAS_PATH if a_atlas_path is None else Path(a_atlas_path)
        )
        if atlas != state["A_atlas_receipt"] or catalog != state[
            "A_motif_score_catalog"
        ]:
            raise ValueError(
                "B setup does not match the exact retrospective A atlas"
            )
        return state

    return _consume_regular(
        path,
        label="attention-edge fidelity B setup",
        byte_cap=_MAX_SETUP_BYTES,
        required_mode=0o600,
        required_nlink=1,
        consumer=consume,
    )


def issue_setup(
    *,
    expected_a_fidelity_report_file_sha256: str,
    authorization: str,
    destination: Path | str = DEFAULT_OUTPUT,
    repository_root: Path | str | None = None,
    a_fidelity_report_path: Path | str | None = None,
    a_atlas_path: Path | str | None = None,
) -> dict[str, object]:
    """Issue the no-clobber setup after explicit post-A authorization."""

    if authorization != ISSUANCE_AUTHORIZATION:
        raise PermissionError("explicit post-A B-setup authorization is required")
    if not EXECUTION_CONSUMER_IMPLEMENTED:
        raise RuntimeError(
            "B setup issuance is deferred until its execution consumer is "
            "implemented, source-bound, and separately audited"
        )
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    panel = qualify_fidelity_b_panel(repository_root=root)
    atlas, catalog = _atlas_receipt(
        root / A_ATLAS_PATH if a_atlas_path is None else Path(a_atlas_path)
    )
    fidelity = _a_fidelity_receipt(
        root / A_FIDELITY_REPORT_PATH
        if a_fidelity_report_path is None
        else Path(a_fidelity_report_path),
        expected_file_sha256=expected_a_fidelity_report_file_sha256,
    )
    implementation = _source_receipt(root)
    state = build_setup(
        panel_qualification=panel,
        atlas_receipt=atlas,
        motif_score_catalog=catalog,
        a_fidelity_receipt=fidelity,
        implementation_receipt=implementation,
    )
    return write_setup(state, destination)


def check_setup(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
    a_fidelity_report_path: Path | str | None = None,
    a_atlas_path: Path | str | None = None,
) -> dict[str, object]:
    if not EXECUTION_CONSUMER_IMPLEMENTED:
        raise RuntimeError(
            "B setup check is deferred until its execution consumer is "
            "implemented, source-bound, and separately audited"
        )
    path = _reject_default_setup_path_alias(
        Path(source),
        label="attention-edge fidelity B setup check",
    )
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )

    def consume(raw: bytes) -> dict[str, object]:
        expected = _strict_json_loads(
            raw,
            label="attention-edge fidelity B setup check",
        )
        if not isinstance(expected, Mapping):
            raise ValueError("attention-edge fidelity B setup must be an object")
        expected_state = validate_setup(expected)
        state = load_setup(
            path,
            repository_root=root,
            a_atlas_path=a_atlas_path,
        )
        if state != expected_state:
            raise ValueError("B setup parsed bytes changed during authentication")
        if (
            qualify_fidelity_b_panel(repository_root=root)
            != state["panel_qualification"]
        ):
            raise ValueError("B setup panel qualification no longer reproduces")
        atlas, catalog = _atlas_receipt(
            root / A_ATLAS_PATH
            if a_atlas_path is None
            else Path(a_atlas_path)
        )
        if atlas != state["A_atlas_receipt"] or catalog != state[
            "A_motif_score_catalog"
        ]:
            raise ValueError("B setup A-atlas binding no longer reproduces")
        fidelity_state = state["A_fidelity_receipt"]
        assert isinstance(fidelity_state, Mapping)
        fidelity = _a_fidelity_receipt(
            root / A_FIDELITY_REPORT_PATH
            if a_fidelity_report_path is None
            else Path(a_fidelity_report_path),
            expected_file_sha256=str(fidelity_state["file_sha256"]),
        )
        if fidelity != fidelity_state:
            raise ValueError("B setup A-fidelity binding no longer reproduces")
        if _source_receipt(root) != state["implementation_receipt"]:
            raise ValueError("B setup implementation binding no longer reproduces")
        return state

    return _consume_regular(
        path,
        label="attention-edge fidelity B setup check",
        byte_cap=_MAX_SETUP_BYTES,
        required_mode=0o600,
        required_nlink=1,
        consumer=consume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--issue", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--a-fidelity-report",
        type=Path,
        default=A_FIDELITY_REPORT_PATH,
    )
    parser.add_argument("--a-fidelity-report-file-sha256")
    parser.add_argument("--authorization")
    arguments = parser.parse_args(argv)
    if arguments.check:
        check_setup(
            arguments.output,
            a_fidelity_report_path=arguments.a_fidelity_report,
        )
        return 0
    if arguments.a_fidelity_report_file_sha256 is None:
        parser.error("--issue requires --a-fidelity-report-file-sha256")
    if arguments.authorization is None:
        parser.error("--issue requires --authorization")
    issue_setup(
        expected_a_fidelity_report_file_sha256=(
            arguments.a_fidelity_report_file_sha256
        ),
        authorization=arguments.authorization,
        destination=arguments.output,
        a_fidelity_report_path=arguments.a_fidelity_report,
    )
    return 0


__all__ = [
    "A_ATLAS_FILE_SHA256",
    "A_ATLAS_FULL_ALLOWED_ROUTE_COUNT",
    "A_ATLAS_MOTIF_CELL_COUNT",
    "A_ATLAS_PATH",
    "A_ATLAS_REPORT_SHA256",
    "A_ATLAS_SUPPORTED_MOTIF_CELL_COUNT",
    "A_FIDELITY_REPORT_PATH",
    "A_MOTIF_SCORE_CATALOG_SHA256",
    "A_PANEL_PATH",
    "B_PANEL_FILE_SHA256",
    "B_PANEL_ID",
    "B_METADATA_CATALOG_SHA256",
    "B_PANEL_PATH",
    "B_PANEL_SEMANTIC_SHA256",
    "BOOTSTRAP_L95_ZERO_BASED_INDEX",
    "BOOTSTRAP_RESAMPLE_COUNT",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_U95_ZERO_BASED_INDEX",
    "DEFAULT_B_REPORT_OUTPUT",
    "DEFAULT_OUTPUT",
    "DISTANCE_BINS",
    "EMPIRICAL_P90_ZERO_BASED_INDEX",
    "EXECUTION_CONSUMER_IMPLEMENTED",
    "FAMILY_IDS",
    "IMPLEMENTATION_SOURCE_PATHS",
    "IMPLEMENTATION_TEST_PATHS",
    "IMPLEMENTATION_RECEIPT_SCHEMA",
    "ISSUANCE_AUTHORIZATION",
    "MATCHED_TARGET_ONLY_MAX_ABSOLUTE_DELTA_NLL",
    "MATCHED_TARGET_ONLY_MAX_CENTERED_LOGIT_NRMSE",
    "PRIMARY_K",
    "SETUP_FORMAT_VERSION",
    "SETUP_ID",
    "SETUP_SCHEMA",
    "WILSON_Z_TWO_SIDED_95",
    "build_setup",
    "bootstrap_resample_indices",
    "canonical_json_bytes",
    "check_setup",
    "issue_setup",
    "load_fidelity_b_panel",
    "load_setup",
    "main",
    "primary_selected_route_count",
    "qualify_fidelity_b_panel",
    "setup_sha256",
    "validate_setup",
    "wilson_top1_lower_bound",
    "write_setup",
]


if __name__ == "__main__":
    raise SystemExit(main())
