"""Build the reviewed implementation qualification for multimode v3.

Identity flows in one direction::

    reviewed setup + frozen panel manifest + live source/test closure
        -> implementation qualification

The setup must not contain the qualification file digest.  Such a pin would
create an unresolvable setup/qualification hash fixed point because the setup
source and artifact are themselves members of this qualification's input
closure.  Production preflight instead strict-loads this artifact and requires
exact equality with :func:`build_qualification` over the live closure.

Measured experiment reports, bundles, summaries, and scalar result values are
not qualification inputs.  The trust anchor includes its own source bytes but
does not claim resistance to coordinated source-and-artifact rehashing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence

from . import gemma3_layer_multimode_setup as setup


SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.implementation_qualification.v2"
)
FORMAT_VERSION = 2
QUALIFICATION_ID = "gemma3-layer-multimode-implementation-v2"
DEFAULT_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-layer-multimode-implementation-qualification-v2.json"
)
SUPERSEDED_V1_OUTPUT = Path(
    "artifacts/research/"
    "gemma3-layer-multimode-implementation-qualification-v1.json"
)
SUPERSEDED_V1_SHA256 = (
    "be03acfd6530c89b2552ee5a913551dabbc9c25d506339b7ee20d122393ef075"
)
SUPERSEDED_V1_FILE_SHA256 = (
    "71c92e8739b6f968fb13a548dac515b19f023a10b04d5c1ef4b81c9b076e9855"
)
TRUST_ANCHOR_PATH = (
    "src/fisher_graph/"
    "gemma3_layer_multimode_implementation_qualification.py"
)

_DOMAIN = b"fisher_graph:layer-multimode:implementation-qualification:v2\0"
_SUPERSEDED_V1_DOMAIN = (
    b"fisher_graph:layer-multimode:implementation-qualification:v1\0"
)
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_PREDECESSOR_BYTES = 512 * 1024 * 1024

_SOURCE_ENTRYPOINTS = (
    "fisher_graph.gemma3_layer_multimode_implementation_qualification",
    "fisher_graph.gemma3_layer_multimode_report",
    "fisher_graph.gemma3_layer_multimode_experiment",
    "fisher_graph.gemma3_layer_multimode_runtime",
    "fisher_graph.gemma3_layer_multimode_fresh_panels",
    "fisher_graph.gemma3_layer_multimode_setup",
    "fisher_graph.gemma3_layer_multimode_flow",
    "fisher_graph.gemma3_layer_multimode_capture",
    "fisher_graph.gemma3_layer_multimode_materialization",
    "fisher_graph.gemma3_layer_multimode_fit",
    "fisher_graph.gemma3_layer_multimode_selection",
    "fisher_graph.gemma3_layer_multimode_compiler",
    "fisher_graph.gemma3_layer_multimode_evaluation",
)

_TEST_PATHS = (
    "tests/test_gemma3_layer_multimode_flow.py",
    "tests/test_gemma3_layer_multimode_capture.py",
    "tests/test_gemma3_layer_multimode_materialization.py",
    "tests/test_gemma3_layer_multimode_fit.py",
    "tests/test_gemma3_layer_multimode_selection.py",
    "tests/test_gemma3_layer_multimode_compiler.py",
    "tests/test_gemma3_layer_multimode_evaluation.py",
    "tests/test_gemma3_layer_multimode_setup.py",
    "tests/test_gemma3_layer_multimode_fresh_panels.py",
    "tests/test_gemma3_layer_multimode_experiment.py",
    "tests/test_gemma3_layer_multimode_runtime.py",
    "tests/test_gemma3_layer_multimode_report.py",
    "tests/test_gemma3_layer_multimode_implementation_qualification.py",
)


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
                raise ValueError(f"duplicate qualification JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite qualification JSON constant: {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("implementation qualification JSON is invalid") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError("implementation qualification JSON is not canonical")
    return value


def _read_regular(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    primary: BaseException | None = None
    payload: bytes | None = None
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"qualification input cannot be opened safely: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(
                f"qualification input must be a single-link regular file: {path}"
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
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError(f"qualification input changed while reading: {path}")
        payload = b"".join(chunks)
    except BaseException as error:
        primary = error
    finally:
        try:
            os.close(descriptor)
        except BaseException as close_error:
            if primary is None:
                primary = close_error
            else:
                primary.add_note(
                    "qualification input descriptor close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    if primary is not None:
        raise primary
    if payload is None:
        raise AssertionError("qualification input read produced no bytes")
    return payload


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
    primary: BaseException | None = None
    try:
        os.fsync(descriptor)
    except BaseException as error:
        primary = error
    finally:
        try:
            os.close(descriptor)
        except BaseException as close_error:
            if primary is None:
                primary = close_error
            else:
                primary.add_note(
                    "qualification directory close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    if primary is not None:
        raise primary


def _require_regular_directory(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except OSError as error:
            raise ValueError(
                "qualification parent ancestry must already exist"
            ) from error
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError(
                "qualification parent ancestry must contain only real directories"
            )


def _descriptor_stat(descriptor: int) -> os.stat_result:
    try:
        return os.fstat(descriptor)
    except OSError as first_error:
        try:
            return os.stat(descriptor)
        except OSError as second_error:
            first_error.add_note(
                "descriptor stat fallback also failed: "
                f"{type(second_error).__name__}: {second_error}"
            )
            raise first_error


def _close_descriptors(
    descriptors: Sequence[int],
    *,
    primary: BaseException | None,
    label: str,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except BaseException as error:
            errors.append(error)
            if primary is not None:
                primary.add_note(
                    f"{label} also failed: {type(error).__name__}: {error}"
                )
    return tuple(errors)


def _open_directory_chain(
    path: Path,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptors: list[int] = []
    identities: list[tuple[int, int]] = []
    try:
        descriptor = os.open(absolute.anchor, flags)
        descriptors.append(descriptor)
        identities.append(_inode_identity(_descriptor_stat(descriptor)))
        for component in absolute.parts[1:]:
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
            identities.append(_inode_identity(_descriptor_stat(descriptor)))
    except BaseException as error:
        _close_descriptors(
            descriptors,
            primary=error,
            label="qualification directory-chain cleanup",
        )
        raise
    return tuple(descriptors), tuple(identities)


def _verify_directory_chain(
    path: Path,
    expected: tuple[tuple[int, int], ...],
) -> None:
    descriptors, observed = _open_directory_chain(path)
    primary: BaseException | None = None
    if observed != expected:
        primary = RuntimeError("qualification publication directory chain changed")
    errors = _close_descriptors(
        descriptors,
        primary=primary,
        label="qualification directory-chain verification cleanup",
    )
    if primary is None and errors:
        primary = errors[0]
        for error in errors[1:]:
            primary.add_note(f"{type(error).__name__}: {error}")
    if primary is not None:
        raise primary


def _stat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
    expected_identity: tuple[int, int] | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    primary: BaseException | None = None
    payload: bytes | None = None
    try:
        before = os.fstat(descriptor)
        named_before = _stat_at(directory_descriptor, name)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
            or _inode_identity(named_before) != _inode_identity(before)
            or (
                expected_identity is not None
                and _inode_identity(before) != expected_identity
            )
        ):
            raise ValueError("qualification artifact identity or size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("qualification artifact was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("qualification artifact grew while reading")
        after = os.fstat(descriptor)
        named_after = _stat_at(directory_descriptor, name)
        if (
            _file_identity(before) != _file_identity(after)
            or _inode_identity(named_after) != _inode_identity(after)
            or (
                expected_identity is not None
                and _inode_identity(named_after) != expected_identity
            )
        ):
            raise RuntimeError("qualification artifact changed while reading")
        payload = b"".join(chunks)
    except BaseException as error:
        primary = error
    finally:
        try:
            os.close(descriptor)
        except BaseException as close_error:
            if primary is None:
                primary = close_error
            else:
                primary.add_note(
                    "qualification artifact descriptor close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    if primary is not None:
        raise primary
    if payload is None:
        raise AssertionError("qualification artifact read produced no bytes")
    return payload


def _guarded_unlink_at(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
    *,
    label: str,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    guard = os.open(name, flags, dir_fd=directory_descriptor)
    primary: BaseException | None = None
    try:
        guarded = os.fstat(guard)
        named = _stat_at(directory_descriptor, name)
        if (
            not stat.S_ISREG(guarded.st_mode)
            or _inode_identity(guarded) != identity
            or _inode_identity(named) != identity
            or guarded.st_nlink < 1
        ):
            raise RuntimeError(f"{label} was replaced")
        before_links = guarded.st_nlink
        os.unlink(name, dir_fd=directory_descriptor)
        after = os.fstat(guard)
        if (
            _inode_identity(after) != identity
            or after.st_nlink != before_links - 1
        ):
            raise RuntimeError(f"{label} unlink did not close")
    except BaseException as error:
        primary = error
    finally:
        try:
            os.close(guard)
        except BaseException as close_error:
            if primary is None:
                primary = close_error
            else:
                primary.add_note(
                    f"{label} guard close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    if primary is not None:
        raise primary


def _module_path(root: Path, module: str) -> Path | None:
    if not module.startswith("fisher_graph"):
        return None
    relative = Path("src", *module.split("."))
    module_file = root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = root / relative / "__init__.py"
    if package_file.is_file():
        return package_file
    return None


def _relative_import_module(
    current: str,
    *,
    level: int,
    module: str | None,
    current_is_package: bool,
) -> str:
    parts = current.split(".")
    package = parts if current_is_package else parts[:-1]
    if level > len(package):
        raise ValueError(f"relative import escapes fisher_graph: {current}")
    base = package[: len(package) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _first_party_imports(
    *,
    root: Path,
    module: str,
    raw: bytes,
) -> set[str]:
    try:
        tree = ast.parse(raw, filename=str(_module_path(root, module)))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"cannot parse qualification source: {module}") from error
    imports: set[str] = set()
    current_path = _module_path(root, module)
    current_is_package = (
        current_path is not None and current_path.name == "__init__.py"
    )
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _relative_import_module(
                    module,
                    level=node.level,
                    module=node.module,
                    current_is_package=current_is_package,
                )
            else:
                base = node.module or ""
            if node.module is None or base == "fisher_graph":
                candidates.extend(
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                )
            elif base:
                candidates.append(base)
        for candidate in candidates:
            if _module_path(root, candidate) is not None:
                imports.add(candidate)
    return imports


def _discover_source_closure(root: Path) -> dict[str, str]:
    pending = list(_SOURCE_ENTRYPOINTS)
    discovered: dict[str, str] = {}
    while pending:
        module = pending.pop()
        if module in discovered:
            continue
        path = _module_path(root, module)
        if path is None:
            raise FileNotFoundError(f"qualification source module is absent: {module}")
        relative = path.relative_to(root).as_posix()
        raw = _read_regular(path, maximum_bytes=_MAX_SOURCE_BYTES)
        discovered[module] = relative
        pending.extend(
            sorted(
                _first_party_imports(
                    root=root,
                    module=module,
                    raw=raw,
                )
                - discovered.keys()
            )
        )
    if len(set(discovered.values())) != len(discovered):
        raise RuntimeError("qualification source closure has duplicate paths")
    return dict(sorted(discovered.items()))


def _source_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for module, relative in _discover_source_closure(root).items():
        raw = _read_regular(root / relative, maximum_bytes=_MAX_SOURCE_BYTES)
        rows.append(
            {
                "module": module,
                "path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def _test_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if len(set(_TEST_PATHS)) != len(_TEST_PATHS):
        raise RuntimeError("qualification test closure contains duplicate paths")
    for relative in sorted(_TEST_PATHS):
        raw = _read_regular(root / relative, maximum_bytes=_MAX_SOURCE_BYTES)
        rows.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def _assert_acyclic_identity_boundary(
    *,
    setup_state: Mapping[str, object],
    panel_state: Mapping[str, object],
) -> None:
    readiness = setup_state.get("implementation_readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("setup implementation readiness is absent")
    if readiness.get("implementation_qualification_file_sha256") is not None:
        raise ValueError(
            "setup may not pin qualification bytes: setup/qualification hash cycle"
        )
    forbidden_panel_keys = {
        "implementation_qualification_sha256",
        "implementation_qualification_file_sha256",
        "report_sha256",
        "result_sha256",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, Mapping):
            result = set(value)
            for item in value.values():
                result.update(keys(item))
            return result
        if type(value) is list:
            result: set[str] = set()
            for item in value:
                result.update(keys(item))
            return result
        return set()

    collisions = forbidden_panel_keys.intersection(keys(panel_state))
    if collisions:
        raise ValueError(
            "panel manifest enters the qualification/result hash cycle: "
            f"{sorted(collisions)}"
        )


def build_qualification(
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    setup_path = root / setup.DEFAULT_OUTPUT
    setup_raw = _read_regular(setup_path, maximum_bytes=_MAX_ARTIFACT_BYTES)
    setup_value = setup._strict_json_loads(setup_raw)
    if not isinstance(setup_value, Mapping):
        raise ValueError("frozen setup must be a JSON object")
    setup_state = setup.check_setup(
        setup_path,
        repository_root=root,
        verify_predecessor=True,
    )
    if setup.render_setup_bytes(setup_state) != setup_raw:
        raise ValueError("frozen setup is not the live canonical setup")
    superseded_path = root / SUPERSEDED_V1_OUTPUT
    superseded_raw = _read_regular(
        superseded_path,
        maximum_bytes=_MAX_ARTIFACT_BYTES,
    )
    if hashlib.sha256(superseded_raw).hexdigest() != SUPERSEDED_V1_FILE_SHA256:
        raise ValueError("superseded v1 qualification file binding drifted")
    superseded_value = _strict_json_loads(superseded_raw)
    if type(superseded_value) is not dict:
        raise ValueError("superseded v1 qualification must be an exact object")
    superseded_unsigned = dict(superseded_value)
    superseded_semantic = superseded_unsigned.pop("qualification_sha256", None)
    recomputed_superseded = hashlib.sha256(
        _SUPERSEDED_V1_DOMAIN + canonical_json_bytes(superseded_unsigned)
    ).hexdigest()
    if (
        superseded_semantic != SUPERSEDED_V1_SHA256
        or recomputed_superseded != superseded_semantic
    ):
        raise ValueError("superseded v1 qualification semantic binding drifted")
    predecessor_setup_path = root / setup.PREDECESSOR_V2_SETUP
    predecessor_report_path = root / setup.PREDECESSOR_V2_REPORT
    predecessor_setup_raw = _read_regular(
        predecessor_setup_path,
        maximum_bytes=_MAX_ARTIFACT_BYTES,
    )
    provenance = setup_state.get("methodology_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen setup predecessor provenance is absent")
    expected_predecessor = {
        "predecessor_v2_setup_path": str(setup.PREDECESSOR_V2_SETUP),
        "predecessor_v2_setup_sha256": setup.PREDECESSOR_V2_SETUP_SHA256,
        "predecessor_v2_setup_file_sha256": (
            setup.PREDECESSOR_V2_SETUP_FILE_SHA256
        ),
        "predecessor_v2_report_path": str(setup.PREDECESSOR_V2_REPORT),
        "predecessor_v2_report_sha256": setup.PREDECESSOR_V2_REPORT_SHA256,
        "predecessor_v2_report_file_sha256": (
            setup.PREDECESSOR_V2_REPORT_FILE_SHA256
        ),
    }
    if any(
        provenance.get(name) != expected
        for name, expected in expected_predecessor.items()
    ):
        raise ValueError("frozen setup predecessor identity pins drifted")
    for name in (
        "predecessor_v2_preflight_receipt_sha256",
        "predecessor_v2_A_capture_receipt_sha256",
        "predecessor_v2_runtime_qualification_receipt_sha256",
        "predecessor_v2_runtime_execution_receipt_sha256",
        "predecessor_v2_runtime_session_binding_sha256",
        "predecessor_v2_source_model_fingerprint_sha256",
        "predecessor_v2_source_execution_fingerprint_sha256",
    ):
        value = provenance.get(name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"frozen setup {name} pin is invalid")

    from . import gemma3_layer_multimode_fresh_panels as panels

    panel_path = root / panels.DEFAULT_RECEIPT
    panel_raw = _read_regular(panel_path, maximum_bytes=_MAX_ARTIFACT_BYTES)
    panel_state = panels._strict_json_loads(
        panel_raw,
        label="frozen layer-multimode panel receipt",
    )
    if type(panel_state) is not dict:
        raise ValueError("frozen panel receipt must be a canonical object")
    validated_panel = panels.validate_receipt(panel_state)
    if panels.canonical_json_bytes(validated_panel) != panel_raw:
        raise ValueError("frozen panel receipt is not canonical")
    manifest_binding = validated_panel["manifest"]
    if not isinstance(manifest_binding, Mapping):
        raise ValueError("frozen panel manifest binding is invalid")
    panel_sha = manifest_binding.get("manifest_sha256")
    if (
        type(panel_sha) is not str
        or len(panel_sha) != 64
        or any(character not in "0123456789abcdef" for character in panel_sha)
    ):
        raise ValueError("panel manifest semantic digest is invalid")
    panel_file_sha = hashlib.sha256(panel_raw).hexdigest()
    panel_preflight = panels.load_frozen_preflight(
        repository_root=root,
        expected_receipt_file_sha256=panel_file_sha,
    )
    if (
        panel_preflight.receipt_sha256 != validated_panel["receipt_sha256"]
        or panel_preflight.bundle_sha256 != validated_panel["bundle_sha256"]
        or panel_preflight.manifest["manifest_sha256"] != panel_sha
    ):
        raise ValueError("frozen panel opaque preflight differs from receipt")
    readiness = setup_state["implementation_readiness"]
    pinned_panel = readiness["panel_receipt_file_sha256"]  # type: ignore[index]
    if pinned_panel is not None and pinned_panel != panel_file_sha:
        raise ValueError("panel receipt differs from frozen setup pin")
    _assert_acyclic_identity_boundary(
        setup_state=setup_state,
        panel_state=validated_panel,
    )

    state: dict[str, object] = {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "supersedes_qualification": {
            "path": str(SUPERSEDED_V1_OUTPUT),
            "qualification_sha256": SUPERSEDED_V1_SHA256,
            "file_sha256": SUPERSEDED_V1_FILE_SHA256,
            "reason": (
                "typed_preoptimizer_support_rejections_and_v2_experiment_"
                "receipt_schemas"
            ),
        },
        "trust_anchor": {
            "path": TRUST_ANCHOR_PATH,
            "explicitly_reviewed_bootstrap": True,
            "included_in_source_closure": True,
            "coordinated_source_and_artifact_rehash_tamper_proof_claimed": False,
        },
        "frozen_setup": {
            "path": str(setup.DEFAULT_OUTPUT),
            "setup_sha256": setup_state["setup_sha256"],
            "file_sha256": hashlib.sha256(setup_raw).hexdigest(),
        },
        "frozen_predecessor_v2": {
            "setup_path": str(setup.PREDECESSOR_V2_SETUP),
            "setup_sha256": setup.PREDECESSOR_V2_SETUP_SHA256,
            "setup_file_sha256": hashlib.sha256(
                predecessor_setup_raw
            ).hexdigest(),
            "report_path": str(setup.PREDECESSOR_V2_REPORT),
            "report_sha256": provenance["predecessor_v2_report_sha256"],
            "report_file_sha256": provenance[
                "predecessor_v2_report_file_sha256"
            ],
            "A_capture_receipt_sha256": provenance[
                "predecessor_v2_A_capture_receipt_sha256"
            ],
            "runtime_execution_receipt_sha256": provenance[
                "predecessor_v2_runtime_execution_receipt_sha256"
            ],
            "preflight_receipt_sha256": provenance[
                "predecessor_v2_preflight_receipt_sha256"
            ],
            "runtime_qualification_receipt_sha256": provenance[
                "predecessor_v2_runtime_qualification_receipt_sha256"
            ],
            "runtime_session_binding_sha256": provenance[
                "predecessor_v2_runtime_session_binding_sha256"
            ],
            "source_model_fingerprint_sha256": provenance[
                "predecessor_v2_source_model_fingerprint_sha256"
            ],
            "source_execution_fingerprint_sha256": provenance[
                "predecessor_v2_source_execution_fingerprint_sha256"
            ],
        },
        "frozen_panel_bundle": {
            "receipt_path": str(panels.DEFAULT_RECEIPT),
            "receipt_sha256": panel_preflight.receipt_sha256,
            "receipt_file_sha256": panel_preflight.receipt_file_sha256,
            "manifest_path": str(panels.DEFAULT_MANIFEST),
            "manifest_sha256": panel_sha,
            "manifest_file_sha256": panel_preflight.manifest_file_sha256,
            "role_files": [
                {
                    "role": role,
                    "path": str(panels.DEFAULT_ROLE_CORPORA[role]),
                    "file_sha256": file_sha256,
                }
                for role, file_sha256 in panel_preflight.role_file_sha256_items
            ],
            "bundle_sha256": panel_preflight.bundle_sha256,
        },
        "source_closure": _source_rows(root),
        "model_free_adversarial_test_closure": _test_rows(root),
        "identity_boundary": {
            "direction": (
                "reviewed_setup_and_panel_plus_live_source_and_test_closure_"
                "to_qualification"
            ),
            "setup_pins_qualification_file_sha256": False,
            "qualification_authenticated_by_exact_live_build": True,
            "qualification_artifact_bytes_excluded_from_own_identity": True,
            "experiment_report_bytes_excluded": True,
            "coefficient_bundle_bytes_excluded": True,
            "summary_sidecar_bytes_excluded": True,
            "measured_result_values_excluded": True,
        },
        "qualification_policy": {
            "source_inventory_is_transitive_AST_discovered_and_exact": True,
            "worker_report_panel_setup_and_core_entrypoints_required": True,
            "trust_anchor_source_is_included": True,
            "model_free_adversarial_tests_are_byte_authenticated": True,
            "strict_live_rebuild_required_by_preflight": True,
            "no_setup_qualification_digest_cycle": True,
            "no_claim_of_external_signature_or_coordinated_tamper_resistance": True,
            "scope": "exact_local_checkout_source_and_test_closure",
            "git_commit_is_self_contained_for_every_transitive_dependency_"
            "claimed": False,
        },
    }
    state["qualification_sha256"] = _digest(state)
    return state


def validate_qualification(
    value: Mapping[str, object],
    *,
    repository_root: Path | str | None = None,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("implementation qualification must be an exact dict")
    expected = build_qualification(repository_root=repository_root)
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ValueError("qualification differs from the live implementation closure")
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
    _require_regular_directory(path.parent)
    descriptors, identities = _open_directory_chain(path.parent)
    primary: BaseException | None = None
    raw: bytes | None = None
    try:
        raw = _read_regular_at(
            descriptors[-1],
            path.name,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
        )
        _verify_directory_chain(path.parent, identities)
    except BaseException as error:
        primary = error
    close_errors = _close_descriptors(
        descriptors,
        primary=primary,
        label="qualification loader directory descriptor cleanup",
    )
    if primary is None and close_errors:
        primary = close_errors[0]
        for error in close_errors[1:]:
            primary.add_note(f"{type(error).__name__}: {error}")
    if primary is not None:
        raise primary
    if raw is None:
        raise AssertionError("qualification loader produced no bytes")
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
    _require_regular_directory(path.parent)
    state = build_qualification(repository_root=root)
    raw = canonical_json_bytes(state)
    directory_descriptors, directory_identities = _open_directory_chain(
        path.parent
    )
    parent_descriptor = directory_descriptors[-1]
    descriptor = -1
    temporary_name = ""
    temporary_inode: tuple[int, int] | None = None
    temporary_present = False
    target_link_attempted = False
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []

    def rollback_target(directory_descriptor: int) -> None:
        if not target_link_attempted or temporary_inode is None:
            return
        unlink_error: BaseException | None = None
        try:
            _guarded_unlink_at(
                directory_descriptor,
                path.name,
                temporary_inode,
                label="qualification rollback target",
            )
        except FileNotFoundError:
            return
        except BaseException as error:
            unlink_error = error
        try:
            os.fsync(directory_descriptor)
        except BaseException as fsync_error:
            if unlink_error is None:
                unlink_error = fsync_error
            else:
                unlink_error.add_note(
                    "qualification rollback directory fsync also failed: "
                    f"{type(fsync_error).__name__}: {fsync_error}"
                )
        if unlink_error is not None:
            raise unlink_error

    def late_rollback_target() -> list[BaseException]:
        errors: list[BaseException] = []
        descriptors: tuple[int, ...] = ()
        try:
            descriptors, observed = _open_directory_chain(path.parent)
            if observed != directory_identities:
                raise RuntimeError(
                    "qualification directory changed before late rollback"
                )
            rollback_target(descriptors[-1])
        except BaseException as error:
            errors.append(error)
        if descriptors:
            errors.extend(
                _close_descriptors(
                    descriptors,
                    primary=errors[0] if errors else None,
                    label="late qualification rollback descriptor cleanup",
                )
            )
        return errors

    try:
        try:
            _stat_at(parent_descriptor, path.name)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(path)
        for _attempt in range(128):
            candidate = f".{path.name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            temporary_present = True
            break
        if descriptor < 0 or not temporary_name:
            raise RuntimeError("could not allocate qualification temporary")
        temporary_inode = _inode_identity(_descriptor_stat(descriptor))
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise RuntimeError("qualification temporary write stalled")
            written += count
        os.fsync(descriptor)
        current_descriptor = _descriptor_stat(descriptor)
        if (
            _inode_identity(current_descriptor) != temporary_inode
            or current_descriptor.st_size != len(raw)
        ):
            raise RuntimeError("qualification temporary changed while writing")
        os.close(descriptor)
        descriptor = -1
        current_temporary = _stat_at(parent_descriptor, temporary_name)
        if (
            not stat.S_ISREG(current_temporary.st_mode)
            or _inode_identity(current_temporary) != temporary_inode
            or current_temporary.st_nlink != 1
        ):
            raise RuntimeError("qualification temporary drifted before link")
        target_link_attempted = True
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published = _stat_at(parent_descriptor, path.name)
        if (
            not stat.S_ISREG(published.st_mode)
            or _inode_identity(published) != temporary_inode
            or published.st_nlink != 2
        ):
            raise RuntimeError("qualification identity drifted while linking")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        guard = os.open(temporary_name, flags, dir_fd=parent_descriptor)
        guard_error: BaseException | None = None
        try:
            guarded = _descriptor_stat(guard)
            named = _stat_at(parent_descriptor, temporary_name)
            if (
                _inode_identity(guarded) != temporary_inode
                or _inode_identity(named) != temporary_inode
                or guarded.st_nlink != 2
            ):
                raise RuntimeError(
                    "qualification temporary was replaced before unlink"
                )
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            after = _descriptor_stat(guard)
            if (
                _inode_identity(after) != temporary_inode
                or after.st_nlink != 1
            ):
                raise RuntimeError("qualification temporary unlink did not close")
        except BaseException as error:
            guard_error = error
        finally:
            try:
                os.close(guard)
            except BaseException as close_error:
                if guard_error is None:
                    guard_error = close_error
                else:
                    guard_error.add_note(
                        "qualification temporary guard close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
        if guard_error is not None:
            raise guard_error
        temporary_present = False
        os.fsync(parent_descriptor)
        observed_raw = _read_regular_at(
            parent_descriptor,
            path.name,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
            expected_identity=temporary_inode,
        )
        observed = _strict_json_loads(observed_raw)
        if (
            observed_raw != raw
            or not isinstance(observed, dict)
            or validate_qualification(observed, repository_root=root) != state
        ):
            raise RuntimeError("qualification strict reload drifted")
        _verify_directory_chain(path.parent, directory_identities)
    except BaseException as error:
        primary = error
    if descriptor >= 0:
        try:
            os.close(descriptor)
        except BaseException as close_error:
            cleanup_errors.append(close_error)
    if temporary_present and temporary_name:
        if temporary_inode is None:
            cleanup_errors.append(
                RuntimeError("qualification temporary identity was not established")
            )
        else:
            cleanup_error: BaseException | None = None
            try:
                _guarded_unlink_at(
                    parent_descriptor,
                    temporary_name,
                    temporary_inode,
                    label="qualification temporary",
                )
                temporary_present = False
            except FileNotFoundError:
                temporary_present = False
            except BaseException as error:
                cleanup_error = error
            try:
                os.fsync(parent_descriptor)
            except BaseException as fsync_error:
                if cleanup_error is None:
                    cleanup_error = fsync_error
                else:
                    cleanup_error.add_note(
                        "qualification temporary parent fsync also failed: "
                        f"{type(fsync_error).__name__}: {fsync_error}"
                    )
            if cleanup_error is not None:
                cleanup_errors.append(cleanup_error)
    if primary is not None or cleanup_errors:
        if primary is None:
            primary = cleanup_errors.pop(0)
        try:
            rollback_target(parent_descriptor)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
    try:
        _verify_directory_chain(path.parent, directory_identities)
    except BaseException as verify_error:
        if primary is None:
            primary = verify_error
            try:
                rollback_target(parent_descriptor)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        else:
            cleanup_errors.append(verify_error)
    close_errors = _close_descriptors(
        directory_descriptors,
        primary=primary,
        label="qualification publication directory descriptor cleanup",
    )
    if primary is None and close_errors:
        primary = close_errors[0]
        cleanup_errors.extend(close_errors[1:])
        cleanup_errors.extend(late_rollback_target())
    else:
        cleanup_errors.extend(close_errors)
    if primary is not None:
        if cleanup_errors:
            for cleanup_error in cleanup_errors:
                primary.add_note(
                    "qualification cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise primary from BaseExceptionGroup(
                "qualification rollback/temporary cleanup failures",
                cleanup_errors,
            )
        raise primary
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--print", action="store_true")
    action.add_argument("--write", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    output = (
        arguments.output
        if arguments.output.is_absolute()
        else root / arguments.output
    )
    if arguments.check:
        state = load_qualification(output, repository_root=root)
    elif arguments.write:
        state = write_qualification(output, repository_root=root)
    else:
        state = build_qualification(repository_root=root)
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "FORMAT_VERSION",
    "QUALIFICATION_ID",
    "SCHEMA",
    "SUPERSEDED_V1_FILE_SHA256",
    "SUPERSEDED_V1_OUTPUT",
    "SUPERSEDED_V1_SHA256",
    "TRUST_ANCHOR_PATH",
    "build_qualification",
    "load_qualification",
    "validate_qualification",
    "write_qualification",
]
