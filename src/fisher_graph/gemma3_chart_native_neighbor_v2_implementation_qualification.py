"""Build and authenticate the implementation qualification for v2.

This module is the explicitly reviewed bootstrap for the qualification
artifact.  Its own bytes are included in the artifact source closure, while no
claim is made that the bootstrap defeats coordinated source/artifact tampering.
The artifact authenticates the complete first-party
source closure used by the v2 worker, the immutable v2 scientific setup, and
the predecessor v1 setup.  Measured diagnostic values are deliberately not an
input to the qualification identity; they are authenticated separately by the
live v2 preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence

from . import gemma3_chart_native_neighbor_setup as v1_setup
from . import (
    gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup as diagnostic_setup,
)
from . import gemma3_chart_native_neighbor_v2_setup as v2_setup


SCHEMA = "fisher_graph.gemma3_chart_native_neighbor.v2_implementation_qualification.v1"
FORMAT_VERSION = 1
QUALIFICATION_ID = "gemma3-chart-native-neighbor-v2-implementation-v1"
DEFAULT_OUTPUT = v2_setup.FUTURE_IMPLEMENTATION_QUALIFICATION_OUTPUT
TRUST_ANCHOR_PATH = (
    "src/fisher_graph/"
    "gemma3_chart_native_neighbor_v2_implementation_qualification.py"
)
V2_WORKER_PATH = (
    "src/fisher_graph/gemma3_chart_native_neighbor_v2_experiment.py"
)
_DOMAIN = b"fisher_graph:chart-native-neighbor-v2:implementation-qualification:v1\0"
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024

_EXTRA_SOURCE_PATHS = {
    "chart_native_neighbor_v2_implementation_qualification": TRUST_ANCHOR_PATH,
    "chart_native_neighbor_v1_setup": (
        "src/fisher_graph/gemma3_chart_native_neighbor_setup.py"
    ),
    "chart_native_neighbor_v2_worker": V2_WORKER_PATH,
    "chart_native_neighbor_v2_report": (
        "src/fisher_graph/gemma3_chart_native_neighbor_v2_report.py"
    ),
    "chart_native_neighbor_v2_setup": (
        "src/fisher_graph/gemma3_chart_native_neighbor_v2_setup.py"
    ),
    "chart_native_neighbor_A_K8_diagnostic_runner": (
        "src/fisher_graph/"
        "gemma3_chart_native_neighbor_a_occupancy_diagnostic.py"
    ),
    "chart_native_neighbor_A_K8_diagnostic_setup": (
        "src/fisher_graph/"
        "gemma3_chart_native_neighbor_a_occupancy_diagnostic_setup.py"
    ),
    "streaming_shared_chart_atlas_selection_v1": (
        "src/fisher_graph/streaming_shared_chart_atlas_selection_v1.py"
    ),
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("qualification_sha256", None)
    return hashlib.sha256(_DOMAIN + canonical_json_bytes(unsigned)).hexdigest()


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("qualification JSON is invalid") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError("qualification JSON is not canonical")
    return value


def _regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"qualification input cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"qualification input is not a single-link regular file: {path}"
            )
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"qualification input size is invalid: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"qualification input was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"qualification input grew while reading: {path}")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"qualification input changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_inventory() -> dict[str, str]:
    inventory = {
        name: str(receipt["path"])
        for name, receipt in v1_setup.PINNED_WORKER_IMPLEMENTATION_SOURCES.items()
    }
    # The v1 worker is already present in the predecessor closure.  The names
    # below are new v2 entry points or additional direct dependencies.
    for name, path in _EXTRA_SOURCE_PATHS.items():
        if name in inventory:
            raise RuntimeError(f"v2 qualification source name collides: {name}")
        inventory[name] = path
    if len(set(inventory.values())) != len(inventory):
        # A duplicated path under two names weakens exact inventory checks.
        duplicates = sorted(
            path for path in set(inventory.values())
            if tuple(inventory.values()).count(path) > 1
        )
        raise RuntimeError(f"v2 qualification source paths collide: {duplicates}")
    for name, value in inventory.items():
        path = Path(value)
        if (
            path.is_absolute()
            or not value
            or path.parts != tuple(part for part in value.split("/") if part)
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != value
        ):
            raise ValueError(f"v2 qualification source path is unsafe: {name}")
    return dict(sorted(inventory.items()))


def build_qualification(
    *, repository_root: Path | str | None = None
) -> dict[str, object]:
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    v2_artifact_path = root / v2_setup.DEFAULT_OUTPUT
    v1_artifact_path = root / v1_setup.DEFAULT_OUTPUT
    diagnostic_artifact_path = root / diagnostic_setup.DEFAULT_OUTPUT
    v2_state = v2_setup.check_setup(v2_artifact_path)
    v1_state = v1_setup.check_setup(v1_artifact_path)
    diagnostic_state = diagnostic_setup.check_setup(diagnostic_artifact_path)
    v2_file_sha = hashlib.sha256(
        _regular_bytes(v2_artifact_path, maximum_bytes=_MAX_ARTIFACT_BYTES)
    ).hexdigest()
    v1_file_sha = hashlib.sha256(
        _regular_bytes(v1_artifact_path, maximum_bytes=_MAX_ARTIFACT_BYTES)
    ).hexdigest()
    diagnostic_file_sha = hashlib.sha256(
        _regular_bytes(
            diagnostic_artifact_path, maximum_bytes=_MAX_ARTIFACT_BYTES
        )
    ).hexdigest()
    if (
        v2_state["setup_sha256"] != v2_setup.PINNED_SETUP_SHA256
        or v2_file_sha != v2_setup.PINNED_SETUP_FILE_SHA256
        or v1_state["setup_sha256"] != v2_setup.V1_SETUP_SHA256
        or v1_file_sha != v2_setup.V1_SETUP_FILE_SHA256
        or diagnostic_state["setup_sha256"]
        != v2_setup.PINNED_A_DIAGNOSTIC_SETUP_SHA256
        or diagnostic_file_sha
        != v2_setup.PINNED_A_DIAGNOSTIC_SETUP_FILE_SHA256
    ):
        raise ValueError("frozen v1/v2/diagnostic setup lineage drifted")

    sources: list[dict[str, str]] = []
    for name, relative in _source_inventory().items():
        path = root / relative
        source_sha = hashlib.sha256(
            _regular_bytes(path, maximum_bytes=_MAX_SOURCE_BYTES)
        ).hexdigest()
        predecessor = v1_setup.PINNED_WORKER_IMPLEMENTATION_SOURCES.get(name)
        if predecessor is not None and source_sha != predecessor["file_sha256"]:
            raise ValueError(f"predecessor source {name} differs from v1 pin")
        if (
            name == "streaming_shared_chart_atlas_selection_v1"
            and source_sha != v2_setup.PINNED_SELECTION_SOURCE_SHA256
        ):
            raise ValueError("streaming selector differs from frozen v2 pin")
        sources.append(
            {"name": name, "path": relative, "file_sha256": source_sha}
        )

    state: dict[str, object] = {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "trust_anchor": {
            "path": TRUST_ANCHOR_PATH,
            "explicitly_reviewed_bootstrap": True,
            "included_in_source_closure": True,
            "coordinated_self_rehash_tamper_proof_claimed": False,
        },
        "frozen_v2_setup": {
            "path": str(v2_setup.DEFAULT_OUTPUT),
            "setup_sha256": v2_state["setup_sha256"],
            "file_sha256": v2_file_sha,
        },
        "predecessor_v1_setup": {
            "path": str(v1_setup.DEFAULT_OUTPUT),
            "setup_sha256": v1_state["setup_sha256"],
            "file_sha256": v1_file_sha,
        },
        "frozen_A_K8_diagnostic_setup": {
            "path": str(diagnostic_setup.DEFAULT_OUTPUT),
            "setup_sha256": diagnostic_state["setup_sha256"],
            "file_sha256": diagnostic_file_sha,
        },
        "source_closure": sources,
        "qualification_policy": {
            "source_inventory_is_exact": True,
            "v2_worker_and_all_invoked_first_party_dependencies_pinned": True,
            "frozen_v2_setup_and_selector_pinned": True,
            "diagnostic_result_file_sha256_excluded_from_identity": True,
            "diagnostic_result_sha256_excluded_from_identity": True,
            "diagnostic_measurements_excluded_from_identity": True,
            "canonical_result_path_bound_indirectly_by_v2_setup": True,
            "diagnostic_is_authenticated_and_bound_only_by_live_preflight": True,
        },
    }
    state["qualification_sha256"] = _digest(state)
    return state


def validate_qualification(
    value: Mapping[str, object],
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("qualification must be a mapping")
    expected = build_qualification(repository_root=repository_root)
    if (
        type(value) is not dict
        or canonical_json_bytes(value) != canonical_json_bytes(expected)
    ):
        raise ValueError("implementation qualification differs from live source closure")
    return dict(value)


def load_qualification(
    source: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    path = Path(source)
    if not path.is_absolute():
        path = root / path
    raw = _regular_bytes(path, maximum_bytes=_MAX_ARTIFACT_BYTES)
    value = _strict_json_loads(raw)
    if not isinstance(value, dict):
        raise ValueError("implementation qualification must be a JSON object")
    return validate_qualification(value, repository_root=root)


def write_qualification(
    destination: Path | str = DEFAULT_OUTPUT,
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    path = Path(destination)
    if not path.is_absolute():
        path = root / path
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("qualification destination parent must be a directory")
    state = build_qualification(repository_root=root)
    raw = canonical_json_bytes(state)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    temporary_owner = _inode_identity(os.fstat(descriptor))
    temporary_identity: tuple[int, int, int, int, int] | None = None
    linked_intended = False
    linked_owner = temporary_owner
    linked_identity: tuple[int, int, int, int, int] | None = None
    temporary_exists = True
    publication_failure: BaseException | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_after_write = temporary.lstat()
        if _inode_identity(temporary_after_write) != temporary_owner:
            raise RuntimeError("qualification temporary changed while writing")
        temporary_identity = _file_identity(temporary_after_write)
        if _file_identity(temporary.lstat()) != temporary_identity:
            raise RuntimeError("qualification temporary changed before link")
        # Register intended ownership before the syscall.  If the link succeeds
        # and execution is interrupted before the next statement, cleanup can
        # still authenticate the new name by the retained mkstemp inode.
        linked_intended = True
        os.link(temporary, path)
        linked_after = path.lstat()
        linked_identity = _file_identity(linked_after)
        temporary_after_link = temporary.lstat()
        if (
            _inode_identity(linked_after) != linked_owner
            or _inode_identity(temporary_after_link) != temporary_owner
        ):
            raise RuntimeError("qualification temporary changed while linking")
        temporary_identity = _file_identity(temporary_after_link)
        if _file_identity(temporary.lstat()) != temporary_identity:
            raise RuntimeError("qualification temporary changed before unlink")
        # Temp unlink changes the shared inode ctime.  Clear the stale full
        # output identity first; the owner inode remains a safe cleanup token.
        linked_identity = None
        os.unlink(temporary)
        temporary_exists = False
        temporary_identity = None
        linked_after_unlink = path.lstat()
        if _inode_identity(linked_after_unlink) != linked_owner:
            raise RuntimeError("qualification output changed after temp unlink")
        linked_identity = _file_identity(linked_after_unlink)
        _fsync_directory(path.parent)
        if load_qualification(path, repository_root=root) != state:
            raise RuntimeError("written implementation qualification reload drifted")
        return state
    except BaseException as error:
        publication_failure = error
        cleanup_errors: list[BaseException] = []
        if linked_intended:
            try:
                current = path.lstat()
                if (
                    not stat.S_ISREG(current.st_mode)
                    or _inode_identity(current) != linked_owner
                    or (
                        linked_identity is not None
                        and _file_identity(current) != linked_identity
                    )
                ):
                    raise RuntimeError(
                        "qualification rollback target changed before unlink"
                    )
                os.unlink(path)
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                _fsync_directory(path.parent)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            error.add_note(
                "qualification rollback had "
                f"{len(cleanup_errors)} cleanup failure(s)"
            )
            raise error from BaseExceptionGroup(
                "qualification rollback cleanup failures", cleanup_errors
            )
        raise
    finally:
        if temporary_exists:
            temporary_cleanup_errors: list[BaseException] = []
            removed_or_absent = False
            try:
                current = temporary.lstat()
                if _inode_identity(current) != temporary_owner:
                    raise RuntimeError(
                        "qualification temporary changed before cleanup"
                    )
                os.unlink(temporary)
                removed_or_absent = True
            except FileNotFoundError:
                removed_or_absent = True
            except BaseException as cleanup_error:
                temporary_cleanup_errors.append(cleanup_error)
            if removed_or_absent:
                try:
                    _fsync_directory(temporary.parent)
                except BaseException as cleanup_error:
                    temporary_cleanup_errors.append(cleanup_error)
            if temporary_cleanup_errors:
                if publication_failure is not None:
                    publication_failure.add_note(
                        "qualification temporary cleanup also failed: "
                        f"{len(temporary_cleanup_errors)} failure(s)"
                    )
                else:
                    raise BaseExceptionGroup(
                        "qualification temporary cleanup failures",
                        temporary_cleanup_errors,
                    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--print", action="store_true")
    action.add_argument("--write", action="store_true")
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        state = load_qualification(output, repository_root=root)
    elif args.write:
        state = write_qualification(output, repository_root=root)
    else:
        state = build_qualification(repository_root=root)
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
