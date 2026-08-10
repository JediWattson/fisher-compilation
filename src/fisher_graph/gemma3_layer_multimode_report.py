"""Strict result replay and durable publication for layer-multimode v3.

The report is deliberately a scalar and receipt-only artifact.  It replays
the complete A materialization, the 324-candidate selection, the source-free
compiler receipt, and the typed B3/C3/D3 receipts.  It never accepts prompt
text, token ids, logits, weights, or activation rows.

Publication is an all-or-none transaction.  The coefficient bundle and the
deterministic summary are made durable first; the report is linked last and
is the commit marker.  Rollback authority is held only by the exact in-process
receipt returned by the publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Mapping, Sequence

from . import gemma3_layer_multimode_setup as setup


REPORT_SCHEMA = "fisher_graph.gemma3_layer_multimode.experiment_report.v1"
REPORT_FORMAT_VERSION = 1
PUBLICATION_SCHEMA = (
    "fisher_graph.gemma3_layer_multimode.publication_receipt.v1"
)
PUBLICATION_FORMAT_VERSION = 1

DEFAULT_REPORT_OUTPUT = setup.DEFAULT_REPORT_OUTPUT
DEFAULT_BUNDLE_OUTPUT = setup.DEFAULT_BUNDLE_OUTPUT
DEFAULT_SUMMARY_OUTPUT = setup.DEFAULT_SUMMARY_OUTPUT

TERMINAL_B3_FAILED = "B3_FAILED"
TERMINAL_C3_FAILED = "C3_FAILED"
TERMINAL_D3_COMPLETED = "D3_COMPLETED"
TERMINAL_STATUSES = (
    TERMINAL_B3_FAILED,
    TERMINAL_C3_FAILED,
    TERMINAL_D3_COMPLETED,
)

_REPORT_DOMAIN = b"fisher_graph:layer-multimode:report:v1\0"
_SUMMARY_DOMAIN = b"fisher_graph:layer-multimode:summary:v1\0"
_PUBLICATION_DOMAIN = b"fisher_graph:layer-multimode:publication:v1\0"
_MAX_REPORT_BYTES = 512 * 1024 * 1024
_MAX_SUMMARY_BYTES = 2 * 1024 * 1024
_MAX_BUNDLE_BYTES = 8 * 1024 * 1024 * 1024

_REPORT_PAYLOAD_FIELDS = {
    "terminal_status",
    "preflight_receipt",
    "A_materialization_receipt",
    "selection_receipt",
    "compilation_receipt",
    "bundle_receipt",
    "development_teardown_receipt",
    "experiment_counters_receipt",
    "fresh_panel_execution_receipt",
    "runtime_execution_receipt",
    "post_terminal_cleanup_receipt",
    "B3_evaluation",
    "C3_evaluation",
    "D3_evaluation",
    "reserve_disclosure",
}
_PRIVACY = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_tokens": False,
    "contains_raw_logits": False,
    "contains_model_weights": False,
    "contains_activation_rows": False,
    "contains_gradient_rows": False,
    "contains_scalar_example_aggregates": True,
}


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
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


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
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} JSON is invalid") from error


def _exact_dict(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} field inventory drifted")
    return dict(value)


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


def _read_regular(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened safely") from error
    primary: BaseException | None = None
    payload: bytes | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a single-link regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError(f"{label} changed while reading")
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
                    f"{label} descriptor close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    if primary is not None:
        raise primary
    if payload is None:
        raise AssertionError(f"{label} read completed without bytes")
    return payload


def _regular_parent(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    anchor = Path(absolute.anchor)
    current = anchor
    for component in absolute.parts[1:]:
        current = current / component
        try:
            before = current.lstat()
        except OSError as error:
            raise ValueError(
                "publication parent ancestry must already exist"
            ) from error
        if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise ValueError(
                "publication parent ancestry must contain only real directories"
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
    failures: list[BaseException] = []
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except BaseException as error:
            failures.append(error)
            if primary is not None:
                primary.add_note(
                    f"{label} also failed: {type(error).__name__}: {error}"
                )
    return tuple(failures)


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
            label="publication directory-chain cleanup",
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
        primary = RuntimeError("publication directory chain changed")
    failures = _close_descriptors(
        descriptors,
        primary=primary,
        label="publication directory-chain verification cleanup",
    )
    if primary is None and failures:
        primary = RuntimeError("publication directory-chain cleanup failed")
        for error in failures:
            primary.add_note(f"{type(error).__name__}: {error}")
    if primary is not None:
        raise primary


def _stat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    label: str,
    maximum_bytes: int,
    expected_identity: tuple[int, int, int, int, int] | None = None,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    primary: BaseException | None = None
    result: tuple[bytes, tuple[int, int, int, int, int]] | None = None
    try:
        before = os.fstat(descriptor)
        named_before = _stat_at(directory_descriptor, name)
        identity = _file_identity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
            or _inode_identity(named_before) != _inode_identity(before)
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise ValueError(f"{label} identity or size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        named_after = _stat_at(directory_descriptor, name)
        if (
            _file_identity(after) != identity
            or _inode_identity(named_after) != _inode_identity(after)
            or (
                expected_identity is not None
                and _file_identity(named_after) != expected_identity
            )
        ):
            raise RuntimeError(f"{label} changed while reading")
        result = b"".join(chunks), identity
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
                    f"{label} descriptor close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
    if primary is not None:
        raise primary
    if result is None:
        raise AssertionError(f"{label} read completed without bytes")
    return result


def _fsync_directories(paths: Sequence[Path]) -> None:
    for path in sorted(set(paths), key=str):
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
                        "directory descriptor close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
        if primary is not None:
            raise primary


def _reject_raw_payloads(value: object) -> None:
    forbidden = {
        "prompt_text",
        "prompt_texts",
        "token_ids",
        "raw_tokens",
        "raw_logits",
        "model_weights",
        "activation_rows",
        "gradient_rows",
        "teacher_velocities",
        "chart_states",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden and item not in (None, False, [], {}):
                raise ValueError(f"report contains forbidden raw field: {key}")
            _reject_raw_payloads(item)
    elif type(value) is list:
        for item in value:
            _reject_raw_payloads(item)


def _parse_bundle_receipt(value: object) -> dict[str, object]:
    fields = {
        "bundle_sha256",
        "byte_count",
        "manifest_sha256",
        "compilation_receipt_sha256",
    }
    row = _exact_dict(value, fields, label="bundle receipt")
    for name in fields - {"byte_count"}:
        _require_sha256(row[name], label=f"bundle {name}")
    if type(row["byte_count"]) is not int or row["byte_count"] <= 0:
        raise ValueError("bundle byte_count must be a positive integer")
    return row


def _parse_resource_preflight(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "workspace_parent_device",
        "workspace_parent_inode",
        "output_parent_devices",
        "validated_free_bytes_floor_by_device",
        "physical_memory_bytes",
        "A_recapture_payload_estimate_bytes",
        "A_materialization_payload_estimate_bytes",
        "full_grid_fit_peak_estimate_bytes",
        "model_and_atomic_staging_estimate_bytes",
        "required_free_disk_bytes",
        "required_physical_memory_bytes",
        "workspace_path_safe",
        "output_paths_safe",
        "test_only_nonproduction",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="resource preflight")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.resource_preflight.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
    ):
        raise ValueError("resource preflight header drifted")
    positive_integers = {
        "workspace_parent_device",
        "workspace_parent_inode",
        "physical_memory_bytes",
        "A_recapture_payload_estimate_bytes",
        "A_materialization_payload_estimate_bytes",
        "full_grid_fit_peak_estimate_bytes",
        "model_and_atomic_staging_estimate_bytes",
        "required_free_disk_bytes",
        "required_physical_memory_bytes",
    }
    if any(
        type(row[name]) is not int or row[name] <= 0
        for name in positive_integers
    ):
        raise ValueError("resource preflight integer inventory drifted")
    output_devices = row["output_parent_devices"]
    if (
        type(output_devices) is not list
        or len(output_devices) != 3
        or any(type(item) is not int or item <= 0 for item in output_devices)
    ):
        raise ValueError("resource preflight output devices drifted")
    free_rows = row["validated_free_bytes_floor_by_device"]
    if type(free_rows) is not list or not free_rows:
        raise ValueError("resource preflight free-byte ledger is empty")
    free_devices: list[int] = []
    for item in free_rows:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not int
            or item[0] <= 0
            or type(item[1]) is not int
            or item[1] != row["required_free_disk_bytes"]
        ):
            raise ValueError("resource preflight free-byte row drifted")
        free_devices.append(item[0])
    if free_devices != sorted(set(free_devices)):
        raise ValueError("resource preflight device order/collision drifted")
    gib = 1024**3
    expected_recapture = 2 * setup.LAYER_COUNT * 24_732 * 640 * 4
    if (
        row["physical_memory_bytes"] < row["required_physical_memory_bytes"]
        or row["required_free_disk_bytes"] < 24 * gib
        or row["required_physical_memory_bytes"] < 16 * gib
        or row["A_recapture_payload_estimate_bytes"] != expected_recapture
        or row["A_materialization_payload_estimate_bytes"]
        != expected_recapture
        or row["full_grid_fit_peak_estimate_bytes"] != 8 * gib
        or row["model_and_atomic_staging_estimate_bytes"] != 4 * gib
        or row["workspace_path_safe"] is not True
        or row["output_paths_safe"] is not True
        or type(row["test_only_nonproduction"]) is not bool
    ):
        raise ValueError("resource preflight capacity/firewall drifted")
    observed = _require_sha256(
        row["receipt_sha256"],
        label="resource preflight receipt_sha256",
    )
    unsigned = dict(row)
    unsigned.pop("receipt_sha256")
    domain = b"fisher_graph:layer-multimode:resource-preflight:v1\0"
    if observed != _digest(domain, unsigned):
        raise ValueError("resource preflight digest drifted")
    return row


def _parse_preflight(value: object) -> dict[str, object]:
    """Replay the worker-owned preflight envelope without importing it.

    The worker/report import graph must stay acyclic.  The exact preflight
    schema and digest are therefore validated here and mirrored by the worker.
    """

    fields = {
        "schema",
        "format_version",
        "setup_sha256",
        "setup_file_sha256",
        "panel_manifest_sha256",
        "panel_manifest_file_sha256",
        "panel_receipt_sha256",
        "panel_receipt_file_sha256",
        "panel_bundle_sha256",
        "implementation_qualification_sha256",
        "implementation_qualification_file_sha256",
        "source_model_sha256",
        "predecessor_v2_setup_sha256",
        "predecessor_v2_report_sha256",
        "predecessor_A_capture_receipt_sha256",
        "predecessor_runtime_execution_receipt_sha256",
        "evaluation_protocol_sha256",
        "resource_preflight_receipt",
        "test_only_nonproduction",
        "exposed_predecessor_A_recapture_only",
        "fresh_role_order",
        "full_grid_candidate_count",
        "production_runtime_ready",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="execution preflight")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.execution_preflight.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or type(row["test_only_nonproduction"]) is not bool
        or row["exposed_predecessor_A_recapture_only"] is not True
        or row["fresh_role_order"]
        != [
            setup.DISCOVERY_ROLE,
            setup.CONFIRMATION_ROLE,
            setup.FINAL_ROLE,
            setup.RESERVE_ROLE,
        ]
        or type(row["full_grid_candidate_count"]) is not int
        or row["full_grid_candidate_count"]
        != setup.LAYER_COUNT * setup.CANDIDATE_COUNT_PER_LAYER
        or type(row["production_runtime_ready"]) is not bool
    ):
        raise ValueError("execution preflight header/firewall drifted")
    resource = _parse_resource_preflight(row["resource_preflight_receipt"])
    if resource["test_only_nonproduction"] is not row["test_only_nonproduction"]:
        raise ValueError("execution/resource preflight production markers differ")
    for name in fields - {
        "schema",
        "format_version",
        "test_only_nonproduction",
        "exposed_predecessor_A_recapture_only",
        "fresh_role_order",
        "full_grid_candidate_count",
        "production_runtime_ready",
        "resource_preflight_receipt",
    }:
        _require_sha256(row[name], label=f"preflight {name}")
    unsigned = dict(row)
    observed = unsigned.pop("receipt_sha256")
    domain = b"fisher_graph:layer-multimode:preflight:v1\0"
    if observed != _digest(domain, unsigned):
        raise ValueError("execution preflight digest drifted")
    return row


def _parse_reserve(value: object) -> dict[str, object]:
    fields = {
        "role",
        "semantically_opened",
        "capture_receipt_sha256",
        "evaluation_receipt_sha256",
    }
    row = _exact_dict(value, fields, label="reserve disclosure")
    if (
        row["role"] != setup.RESERVE_ROLE
        or row["semantically_opened"] is not False
        or row["capture_receipt_sha256"] is not None
        or row["evaluation_receipt_sha256"] is not None
    ):
        raise ValueError("reserve must remain semantically unopened")
    return row


def _validate_worker_receipt_digest(
    row: Mapping[str, object],
    *,
    domain: bytes,
    label: str,
) -> None:
    observed = _require_sha256(
        row["receipt_sha256"],
        label=f"{label} receipt_sha256",
    )
    unsigned = dict(row)
    unsigned.pop("receipt_sha256")
    if observed != _digest(domain, unsigned):
        raise ValueError(f"{label} digest drifted")


def _expected_fresh_roles(terminal: str) -> tuple[str, ...]:
    roles = {
        TERMINAL_B3_FAILED: (setup.DISCOVERY_ROLE,),
        TERMINAL_C3_FAILED: (
            setup.DISCOVERY_ROLE,
            setup.CONFIRMATION_ROLE,
        ),
        TERMINAL_D3_COMPLETED: (
            setup.DISCOVERY_ROLE,
            setup.CONFIRMATION_ROLE,
            setup.FINAL_ROLE,
        ),
    }.get(terminal)
    if roles is None:
        raise ValueError("terminal status has no fresh-role schedule")
    return roles


def _parse_development_teardown(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "source_model_sha256",
        "source_execution_fingerprint_sha256",
        "compiled_candidate_present",
        "compilation_receipt_sha256",
        "compiled_model_fingerprint_sha256",
        "source_compilation_aliases_released",
        "native_baseline_retained_for_heldout_evaluation",
        "A_activation_rows_released",
        "A_fit_models_released",
        "A_optimizer_state_released",
        "source_compilation_native_mlp_alias_count",
        "native_baseline_instance_count",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="development teardown")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.development_teardown.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
    ):
        raise ValueError("development teardown header drifted")
    for name in (
        "source_model_sha256",
        "source_execution_fingerprint_sha256",
        "compilation_receipt_sha256",
        "compiled_model_fingerprint_sha256",
    ):
        _require_sha256(row[name], label=f"development teardown {name}")
    if row["compiled_candidate_present"] is not True:
        raise ValueError("published terminal teardown must retain a compiled candidate")
    for name in (
        "source_compilation_aliases_released",
        "native_baseline_retained_for_heldout_evaluation",
        "A_activation_rows_released",
        "A_fit_models_released",
        "A_optimizer_state_released",
    ):
        if row[name] is not True:
            raise ValueError(f"development teardown {name} must be true")
    if (
        type(row["source_compilation_native_mlp_alias_count"]) is not int
        or row["source_compilation_native_mlp_alias_count"] != 0
        or type(row["native_baseline_instance_count"]) is not int
        or row["native_baseline_instance_count"] != 1
    ):
        raise ValueError("development teardown alias/baseline count drifted")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:teardown:v1\0",
        label="development teardown",
    )
    return row


def _parse_experiment_counters(
    value: object,
    *,
    terminal: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "terminal_status",
        "execution_preflight_receipt_sha256",
        "selection_receipt_sha256",
        "strict_preflight_count",
        "runtime_factory_count",
        "exposed_predecessor_A_recapture_count",
        "fresh_v3_A_capture_count",
        "materialization_count",
        "atlas_fit_count",
        "candidate_fit_count",
        "split_refit_count",
        "candidate_evaluation_count",
        "selection_attempt_count",
        "compilation_count",
        "candidate_freeze_count",
        "development_teardown_count",
        "B3_open_count",
        "B3_evaluation_count",
        "C3_open_count",
        "C3_evaluation_count",
        "D3_open_count",
        "D3_evaluation_count",
        "reserve_open_count",
        "native_baseline_evaluator_count",
        "native_baseline_eval_mode",
        "native_baseline_parameter_mutation_count",
        "native_baseline_optimizer_reference_count",
        "compiled_candidate_native_mlp_module_count",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="experiment counters")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.experiment_counters.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or row["terminal_status"] != terminal
    ):
        raise ValueError("experiment counters header/status drifted")
    _require_sha256(
        row["execution_preflight_receipt_sha256"],
        label="counters execution preflight",
    )
    _require_sha256(
        row["selection_receipt_sha256"],
        label="counters selection",
    )
    fixed = {
        "strict_preflight_count": 1,
        "runtime_factory_count": 1,
        "exposed_predecessor_A_recapture_count": 1,
        "fresh_v3_A_capture_count": 0,
        "materialization_count": setup.LAYER_COUNT,
        "atlas_fit_count": 108,
        "candidate_fit_count": setup.LAYER_COUNT
        * setup.CANDIDATE_COUNT_PER_LAYER,
        "split_refit_count": 216,
        "candidate_evaluation_count": setup.LAYER_COUNT
        * setup.CANDIDATE_COUNT_PER_LAYER,
        "selection_attempt_count": 1,
        "compilation_count": 1,
        "candidate_freeze_count": 1,
        "development_teardown_count": 1,
        "B3_open_count": 1,
        "B3_evaluation_count": 1,
        "C3_open_count": 0 if terminal == TERMINAL_B3_FAILED else 1,
        "C3_evaluation_count": 0 if terminal == TERMINAL_B3_FAILED else 1,
        "D3_open_count": 1 if terminal == TERMINAL_D3_COMPLETED else 0,
        "D3_evaluation_count": 1 if terminal == TERMINAL_D3_COMPLETED else 0,
        "reserve_open_count": 0,
        "native_baseline_evaluator_count": 1,
        "native_baseline_parameter_mutation_count": 0,
        "native_baseline_optimizer_reference_count": 0,
        "compiled_candidate_native_mlp_module_count": 0,
    }
    for name, expected in fixed.items():
        if type(row[name]) is not int or row[name] != expected:
            raise ValueError(f"experiment counter {name} drifted")
    if row["native_baseline_eval_mode"] is not True:
        raise ValueError("experiment baseline must remain in eval mode")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:counters:v1\0",
        label="experiment counters",
    )
    return row


def _parse_fresh_panel_execution(
    value: object,
    *,
    terminal: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "terminal_status",
        "panel_manifest_sha256",
        "panel_manifest_file_sha256",
        "panel_receipt_sha256",
        "panel_receipt_file_sha256",
        "panel_bundle_sha256",
        "selection_receipt_sha256",
        "opened_roles",
        "ordered_freeze_receipts",
        "terminal_state",
        "reserve_open_count",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="fresh-panel execution")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.fresh_panel_execution.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or row["terminal_status"] != terminal
    ):
        raise ValueError("fresh-panel execution header/status drifted")
    for name in (
        "panel_manifest_sha256",
        "panel_manifest_file_sha256",
        "panel_receipt_sha256",
        "panel_receipt_file_sha256",
        "panel_bundle_sha256",
        "selection_receipt_sha256",
    ):
        _require_sha256(row[name], label=f"fresh-panel {name}")
    expected_roles = _expected_fresh_roles(terminal)
    if row["opened_roles"] != list(expected_roles):
        raise ValueError("fresh-panel opened-role order drifted")
    freezes = row["ordered_freeze_receipts"]
    if type(freezes) is not list or len(freezes) != len(expected_roles):
        raise ValueError("fresh-panel freeze receipt count drifted")
    stage_by_role = {
        setup.DISCOVERY_ROLE: "B3_teacher_forced_evaluation",
        setup.CONFIRMATION_ROLE: "C3_full_stack_source_free_evaluation",
        setup.FINAL_ROLE: "D3_final_heldout_evaluation",
    }
    passes: list[bool] = []
    for expected_role, freeze in zip(expected_roles, freezes, strict=True):
        freeze_row = _exact_dict(
            freeze,
            {
                "role",
                "stage_kind",
                "stage_receipt_sha256",
                "terminal_passed",
            },
            label="fresh-panel freeze receipt",
        )
        if (
            freeze_row["role"] != expected_role
            or freeze_row["stage_kind"] != stage_by_role[expected_role]
            or type(freeze_row["terminal_passed"]) is not bool
        ):
            raise ValueError("fresh-panel freeze role/stage drifted")
        _require_sha256(
            freeze_row["stage_receipt_sha256"],
            label=f"{expected_role} freeze stage",
        )
        passes.append(freeze_row["terminal_passed"])
    expected_prefix = {
        TERMINAL_B3_FAILED: [False],
        TERMINAL_C3_FAILED: [True, False],
    }.get(terminal)
    if expected_prefix is not None and passes != expected_prefix:
        raise ValueError("fresh-panel terminal pass vector drifted")
    if terminal == TERMINAL_D3_COMPLETED and passes[:2] != [True, True]:
        raise ValueError("fresh-panel D3 prerequisite pass vector drifted")
    expected_state = {
        TERMINAL_B3_FAILED: "terminal_B3_failed",
        TERMINAL_C3_FAILED: "terminal_C3_failed",
        TERMINAL_D3_COMPLETED: (
            "terminal_D3_passed" if passes[-1] else "terminal_D3_failed"
        ),
    }[terminal]
    if (
        row["terminal_state"] != expected_state
        or type(row["reserve_open_count"]) is not int
        or row["reserve_open_count"] != 0
    ):
        raise ValueError("fresh-panel terminal/reserve state drifted")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:fresh-panel-execution:v1\0",
        label="fresh-panel execution",
    )
    return row


def _parse_runtime_qualification(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "execution_preflight_receipt_sha256",
        "implementation_qualification_sha256",
        "implementation_qualification_file_sha256",
        "runtime_implementation_sha256",
        "qualified_runtime_source_sha256",
        "predecessor_preflight_receipt_sha256",
        "predecessor_runtime_qualification_receipt_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
        "tokenizer_asset_catalog_sha256",
        "evaluation_protocol_sha256",
        "source_model_mutated",
        "full_vocab_accumulator_dtype",
        "top1_tie_break",
        "predecessor_A_exact_control_batch_count",
        "predecessor_A_dense_parity_count",
        "fresh_batches_per_opened_role",
        "logits_persisted",
        "token_ids_persisted",
        "prompt_text_persisted",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="runtime qualification")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.runtime_qualification.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
    ):
        raise ValueError("runtime qualification header drifted")
    for name in (
        "execution_preflight_receipt_sha256",
        "implementation_qualification_sha256",
        "implementation_qualification_file_sha256",
        "runtime_implementation_sha256",
        "qualified_runtime_source_sha256",
        "predecessor_preflight_receipt_sha256",
        "predecessor_runtime_qualification_receipt_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
        "tokenizer_asset_catalog_sha256",
        "evaluation_protocol_sha256",
    ):
        _require_sha256(row[name], label=f"runtime qualification {name}")
    if (
        row["source_model_mutated"] is not False
        or row["qualified_runtime_source_sha256"]
        != row["runtime_implementation_sha256"]
        or row["source_model_fingerprint_before_sha256"]
        != row["source_model_fingerprint_after_sha256"]
        or row["source_execution_fingerprint_before_sha256"]
        != row["source_execution_fingerprint_after_sha256"]
        or row["full_vocab_accumulator_dtype"] != "float64"
        or row["top1_tie_break"] != "lowest_vocabulary_index"
        or type(row["predecessor_A_exact_control_batch_count"]) is not int
        or row["predecessor_A_exact_control_batch_count"] != 45
        or type(row["predecessor_A_dense_parity_count"]) is not int
        or row["predecessor_A_dense_parity_count"] != 1
        or type(row["fresh_batches_per_opened_role"]) is not int
        or row["fresh_batches_per_opened_role"] != 8
        or row["logits_persisted"] is not False
        or row["token_ids_persisted"] is not False
        or row["prompt_text_persisted"] is not False
    ):
        raise ValueError("runtime qualification policy/fingerprint drifted")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:runtime-qualification:v1\0",
        label="runtime qualification",
    )
    return row


def _parse_fresh_role_capture(
    value: object,
    *,
    expected_role: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "role",
        "stage_kind",
        "panel_manifest_sha256",
        "panel_manifest_file_sha256",
        "panel_receipt_sha256",
        "panel_receipt_file_sha256",
        "panel_bundle_sha256",
        "role_file_sha256",
        "selection_receipt_sha256",
        "frozen_candidate_identity_sha256",
        "runtime_session_binding_sha256",
        "ordered_tokenization_receipt_sha256",
        "ordered_example_identity_catalog_sha256",
        "ordered_row_identity_catalog_sha256",
        "family_axis_sha256",
        "batch_axis_sha256",
        "example_count",
        "batch_count",
        "predicted_next_token_position_count",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_logits",
        "contains_raw_activations",
        "contains_raw_gradients",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="fresh-role capture")
    stage_by_role = {
        setup.DISCOVERY_ROLE: "B3_teacher_forced_evaluation",
        setup.CONFIRMATION_ROLE: "C3_full_stack_source_free_evaluation",
        setup.FINAL_ROLE: "D3_final_heldout_evaluation",
    }
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.fresh_role_capture.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or row["role"] != expected_role
        or row["stage_kind"] != stage_by_role[expected_role]
    ):
        raise ValueError("fresh-role capture header/role/stage drifted")
    for name in fields - {
        "schema",
        "format_version",
        "role",
        "stage_kind",
        "example_count",
        "batch_count",
        "predicted_next_token_position_count",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_logits",
        "contains_raw_activations",
        "contains_raw_gradients",
    }:
        _require_sha256(row[name], label=f"fresh-role capture {name}")
    expected_counts = {
        "example_count": 48,
        "batch_count": 8,
        "predicted_next_token_position_count": 24_732,
    }
    if any(
        type(row[name]) is not int or row[name] != expected
        for name, expected in expected_counts.items()
    ):
        raise ValueError("fresh-role capture count axis drifted")
    for name in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_logits",
        "contains_raw_activations",
        "contains_raw_gradients",
    ):
        if row[name] is not False:
            raise ValueError("fresh-role capture retained raw evidence")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:fresh-role-capture:v1\0",
        label="fresh-role capture",
    )
    return row


def _parse_predecessor_A_runtime(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "execution_preflight_receipt_sha256",
        "predecessor_preflight_receipt_sha256",
        "predecessor_runtime_qualification_receipt_sha256",
        "source_capture_receipt_sha256",
        "runtime_session_binding_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
        "role_batch_counts",
        "exact_control_forward_count",
        "exact_control_vjp_count",
        "dense_parity_smoke_count",
        "dense_parity_forward_count",
        "dense_parity_vjp_count",
        "total_backbone_forward_count",
        "total_terminal_cotangent_vjp_count",
        "captured_example_count",
        "supplemental_route_receipt_count",
        "mode_raw_artifact_count",
        "lexical_nll_receipt_count",
        "execution_receipt_count",
        "example_capture_receipt_count",
        "canonical_gradient_targets_per_vjp",
        "auxiliary_gradient_targets_per_vjp",
        "gradient_targets_per_vjp",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations",
        "contains_raw_gradients",
        "contains_logits",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="predecessor A runtime")
    if (
        row["schema"]
        != (
            "fisher_graph.gemma3_layer_multimode."
            "predecessor_A_runtime_execution.v1"
        )
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
    ):
        raise ValueError("predecessor A runtime header drifted")
    digest_fields = (
        "execution_preflight_receipt_sha256",
        "predecessor_preflight_receipt_sha256",
        "predecessor_runtime_qualification_receipt_sha256",
        "source_capture_receipt_sha256",
        "runtime_session_binding_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
    )
    for name in digest_fields:
        _require_sha256(row[name], label=f"predecessor A runtime {name}")
    fixed: dict[str, object] = {
        "role_batch_counts": [
            ["A_chart_transport_fit", 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ],
        "exact_control_forward_count": 45,
        "exact_control_vjp_count": 45,
        "dense_parity_smoke_count": 1,
        "dense_parity_forward_count": 1,
        "dense_parity_vjp_count": 1,
        "total_backbone_forward_count": 46,
        "total_terminal_cotangent_vjp_count": 46,
        "captured_example_count": 48,
        "supplemental_route_receipt_count": 864,
        "mode_raw_artifact_count": 38,
        "lexical_nll_receipt_count": 45,
        "execution_receipt_count": 45,
        "example_capture_receipt_count": 48,
        "canonical_gradient_targets_per_vjp": 73,
        "auxiliary_gradient_targets_per_vjp": 19,
        "gradient_targets_per_vjp": 92,
    }
    if any(type(row[name]) is not type(expected) or row[name] != expected for name, expected in fixed.items()):
        raise ValueError("predecessor A runtime exact ledger drifted")
    if (
        row["source_model_fingerprint_before_sha256"]
        != row["source_model_fingerprint_after_sha256"]
        or row["source_execution_fingerprint_before_sha256"]
        != row["source_execution_fingerprint_after_sha256"]
    ):
        raise ValueError("predecessor A runtime source identity mutated")
    for name in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations",
        "contains_raw_gradients",
        "contains_logits",
    ):
        if row[name] is not False:
            raise ValueError("predecessor A runtime retained raw evidence")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:predecessor-A-runtime:v1\0",
        label="predecessor A runtime",
    )
    return row


def _parse_predecessor_A_comparison(value: object) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "predecessor_capture_receipt_sha256",
        "recaptured_capture_receipt_sha256",
        "predecessor_runtime_execution_receipt_sha256",
        "predecessor_runtime_session_binding_sha256",
        "recaptured_runtime_session_binding_sha256",
        "role",
        "batch_count",
        "row_count",
        "role_panel_binding_sha256",
        "ordered_objective_row_catalog_sha256",
        "score_objective_receipt_sha256",
        "dense_lexical_parity_receipt_sha256",
        "mode_raw_artifact_catalog_sha256",
        "mode_semantic_projection_catalog_sha256",
        "lexical_nll_receipt_catalog_sha256",
        "supplemental_route_projection_catalog_sha256",
        "compared_field_inventory",
        "intentionally_session_dependent_field_inventory",
        "runtime_sessions_expected_different",
        "all_comparable_fields_matched",
        "artifact_sha256",
    }
    row = _exact_dict(value, fields, label="predecessor A comparison")
    expected_compared = [
        "batch_count",
        "dense_lexical_parity_receipt_sha256",
        "lexical_nll_receipt_sha256s",
        "mode_raw_artifacts",
        "mode_semantic_projection",
        "ordered_objective_row_catalog_sha256",
        "role",
        "role_panel_binding_sha256",
        "row_count",
        "score_objective_receipt_sha256",
        "supplemental_route_projection",
    ]
    expected_dependent = [
        "batch_axis_catalog_sha256",
        "example_capture_receipt_sha256s",
        "execution_receipt_sha256s",
        "receipt_sha256",
        "runtime_session_binding_sha256",
        "supplemental_route_receipt_sha256s",
    ]
    if (
        row["schema"]
        != (
            "fisher_graph.gemma3_layer_multimode."
            "predecessor_A_recapture_comparison.v1"
        )
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or row["role"] != "A_chart_transport_fit"
        or type(row["batch_count"]) is not int
        or row["batch_count"] != 45
        or type(row["row_count"]) is not int
        or row["row_count"] != 24_732
        or row["compared_field_inventory"] != expected_compared
        or row["intentionally_session_dependent_field_inventory"]
        != expected_dependent
        or row["runtime_sessions_expected_different"] is not True
        or row["all_comparable_fields_matched"] is not True
        or row["predecessor_runtime_session_binding_sha256"]
        == row["recaptured_runtime_session_binding_sha256"]
    ):
        raise ValueError("predecessor A comparison exact replay drifted")
    for name in fields - {
        "schema",
        "format_version",
        "role",
        "batch_count",
        "row_count",
        "compared_field_inventory",
        "intentionally_session_dependent_field_inventory",
        "runtime_sessions_expected_different",
        "all_comparable_fields_matched",
    }:
        _require_sha256(row[name], label=f"predecessor A comparison {name}")
    observed = row["artifact_sha256"]
    payload = dict(row)
    del payload["artifact_sha256"]
    if observed != _digest(
        b"fisher_graph:layer-multimode:predecessor-A-comparison:v1\0",
        payload,
    ):
        raise ValueError("predecessor A comparison digest drifted")
    return row


def _parse_runtime_execution(
    value: object,
    *,
    terminal: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "terminal_status",
        "runtime_implementation_sha256",
        "implementation_qualification_sha256",
        "runtime_qualification_receipt",
        "runtime_qualification_receipt_sha256",
        "execution_environment_sha256",
        "tokenizer_fingerprint_sha256",
        "predecessor_A_capture_receipt_sha256",
        "predecessor_runtime_execution_receipt_sha256",
        "predecessor_A_runtime_receipt",
        "predecessor_A_runtime_receipt_sha256",
        "predecessor_A_lineage_comparison_receipt",
        "predecessor_A_lineage_comparison_receipt_sha256",
        "predecessor_A_recaptured_capture_receipt_sha256",
        "predecessor_A_current_runtime_session_binding_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_model_mutated",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
        "source_execution_mutated",
        "compiled_model_fingerprint_before_sha256",
        "compiled_model_fingerprint_after_sha256",
        "compiled_execution_fingerprint_before_sha256",
        "compiled_execution_fingerprint_after_sha256",
        "compiled_candidate_mutated",
        "runtime_session_binding_sha256",
        "fresh_role_capture_receipts",
        "predecessor_A_batch_count",
        "predecessor_A_exact_control_forward_count",
        "predecessor_A_exact_control_vjp_count",
        "predecessor_A_dense_parity_forward_count",
        "predecessor_A_dense_parity_vjp_count",
        "B3_batch_count",
        "B3_native_forward_count",
        "B3_candidate_layer_velocity_forward_count",
        "C3_batch_count",
        "C3_native_forward_count",
        "C3_compiled_forward_count",
        "D3_batch_count",
        "D3_native_forward_count",
        "D3_compiled_forward_count",
        "fresh_tokenization_authentication_count",
        "fresh_example_count",
        "fresh_next_token_position_count",
        "native_baseline_evaluator_count",
        "native_baseline_eval_mode",
        "native_baseline_parameter_mutation_count",
        "native_baseline_optimizer_reference_count",
        "compiled_candidate_native_mlp_module_count",
        "reserve_open_count",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="runtime execution")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.runtime_execution.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or row["terminal_status"] != terminal
    ):
        raise ValueError("runtime execution header/status drifted")
    digest_names = {
        "runtime_implementation_sha256",
        "implementation_qualification_sha256",
        "runtime_qualification_receipt_sha256",
        "execution_environment_sha256",
        "tokenizer_fingerprint_sha256",
        "predecessor_A_capture_receipt_sha256",
        "predecessor_runtime_execution_receipt_sha256",
        "predecessor_A_runtime_receipt_sha256",
        "predecessor_A_lineage_comparison_receipt_sha256",
        "predecessor_A_recaptured_capture_receipt_sha256",
        "predecessor_A_current_runtime_session_binding_sha256",
        "source_model_fingerprint_before_sha256",
        "source_model_fingerprint_after_sha256",
        "source_execution_fingerprint_before_sha256",
        "source_execution_fingerprint_after_sha256",
        "compiled_model_fingerprint_before_sha256",
        "compiled_model_fingerprint_after_sha256",
        "compiled_execution_fingerprint_before_sha256",
        "compiled_execution_fingerprint_after_sha256",
        "runtime_session_binding_sha256",
    }
    for name in digest_names:
        _require_sha256(row[name], label=f"runtime execution {name}")
    qualification = _parse_runtime_qualification(
        row["runtime_qualification_receipt"]
    )
    predecessor_A_runtime = _parse_predecessor_A_runtime(
        row["predecessor_A_runtime_receipt"]
    )
    predecessor_A_comparison = _parse_predecessor_A_comparison(
        row["predecessor_A_lineage_comparison_receipt"]
    )
    if (
        qualification["receipt_sha256"]
        != row["runtime_qualification_receipt_sha256"]
        or qualification["runtime_implementation_sha256"]
        != row["runtime_implementation_sha256"]
        or qualification["implementation_qualification_sha256"]
        != row["implementation_qualification_sha256"]
        or qualification["source_model_fingerprint_before_sha256"]
        != row["source_model_fingerprint_before_sha256"]
        or qualification["source_model_fingerprint_after_sha256"]
        != row["source_model_fingerprint_after_sha256"]
        or qualification["source_execution_fingerprint_before_sha256"]
        != row["source_execution_fingerprint_before_sha256"]
        or qualification["source_execution_fingerprint_after_sha256"]
        != row["source_execution_fingerprint_after_sha256"]
        or qualification["tokenizer_asset_catalog_sha256"]
        != row["tokenizer_fingerprint_sha256"]
        or predecessor_A_runtime["receipt_sha256"]
        != row["predecessor_A_runtime_receipt_sha256"]
        or predecessor_A_comparison["artifact_sha256"]
        != row["predecessor_A_lineage_comparison_receipt_sha256"]
    ):
        raise ValueError("runtime qualification/execution binding drifted")
    if (
        predecessor_A_runtime["source_capture_receipt_sha256"]
        != row["predecessor_A_recaptured_capture_receipt_sha256"]
        or predecessor_A_runtime["runtime_session_binding_sha256"]
        != row["predecessor_A_current_runtime_session_binding_sha256"]
        or predecessor_A_comparison["predecessor_capture_receipt_sha256"]
        != row["predecessor_A_capture_receipt_sha256"]
        or predecessor_A_comparison["recaptured_capture_receipt_sha256"]
        != row["predecessor_A_recaptured_capture_receipt_sha256"]
        or predecessor_A_comparison[
            "predecessor_runtime_execution_receipt_sha256"
        ]
        != row["predecessor_runtime_execution_receipt_sha256"]
        or predecessor_A_comparison[
            "recaptured_runtime_session_binding_sha256"
        ]
        != row["predecessor_A_current_runtime_session_binding_sha256"]
    ):
        raise ValueError("runtime predecessor A nested lineage drifted")
    if (
        row["source_model_mutated"] is not False
        or row["source_execution_mutated"] is not False
        or row["compiled_candidate_mutated"] is not False
        or row["source_model_fingerprint_before_sha256"]
        != row["source_model_fingerprint_after_sha256"]
        or row["source_execution_fingerprint_before_sha256"]
        != row["source_execution_fingerprint_after_sha256"]
        or row["compiled_model_fingerprint_before_sha256"]
        != row["compiled_model_fingerprint_after_sha256"]
        or row["compiled_execution_fingerprint_before_sha256"]
        != row["compiled_execution_fingerprint_after_sha256"]
    ):
        raise ValueError("runtime source model/execution identity mutated")
    session_payload = {
        name: row[name]
        for name in (
            "runtime_implementation_sha256",
            "implementation_qualification_sha256",
            "execution_environment_sha256",
            "tokenizer_fingerprint_sha256",
            "runtime_qualification_receipt_sha256",
            "source_model_fingerprint_before_sha256",
            "source_model_fingerprint_after_sha256",
            "source_execution_fingerprint_before_sha256",
            "source_execution_fingerprint_after_sha256",
            "compiled_model_fingerprint_before_sha256",
            "compiled_model_fingerprint_after_sha256",
            "compiled_execution_fingerprint_before_sha256",
            "compiled_execution_fingerprint_after_sha256",
            "compiled_candidate_mutated",
        )
    }
    if row["runtime_session_binding_sha256"] != _digest(
        b"fisher_graph:layer-multimode:runtime-session:v1\0",
        session_payload,
    ):
        raise ValueError("runtime session binding drifted")
    opened = len(_expected_fresh_roles(terminal))
    fresh_values = row["fresh_role_capture_receipts"]
    expected_roles = _expected_fresh_roles(terminal)
    if type(fresh_values) is not list or len(fresh_values) != opened:
        raise ValueError("fresh-role capture inventory count drifted")
    fresh_captures = [
        _parse_fresh_role_capture(value, expected_role=role)
        for role, value in zip(expected_roles, fresh_values, strict=True)
    ]
    common_fresh_fields = (
        "panel_manifest_sha256",
        "panel_manifest_file_sha256",
        "panel_receipt_sha256",
        "panel_receipt_file_sha256",
        "panel_bundle_sha256",
        "selection_receipt_sha256",
        "frozen_candidate_identity_sha256",
        "runtime_session_binding_sha256",
        "family_axis_sha256",
        "batch_axis_sha256",
    )
    if any(
        capture["runtime_session_binding_sha256"]
        != row["runtime_session_binding_sha256"]
        or any(
            capture[name] != fresh_captures[0][name]
            for name in common_fresh_fields
        )
        for capture in fresh_captures
    ):
        raise ValueError("fresh-role capture common/runtime lineage drifted")
    fixed = {
        "predecessor_A_batch_count": 45,
        "predecessor_A_exact_control_forward_count": 45,
        "predecessor_A_exact_control_vjp_count": 45,
        "predecessor_A_dense_parity_forward_count": 1,
        "predecessor_A_dense_parity_vjp_count": 1,
        "B3_batch_count": 8,
        "B3_native_forward_count": 8,
        "B3_candidate_layer_velocity_forward_count": 8 * setup.LAYER_COUNT,
        "C3_batch_count": 8 if opened >= 2 else 0,
        "C3_native_forward_count": 8 if opened >= 2 else 0,
        "C3_compiled_forward_count": 8 if opened >= 2 else 0,
        "D3_batch_count": 8 if opened >= 3 else 0,
        "D3_native_forward_count": 8 if opened >= 3 else 0,
        "D3_compiled_forward_count": 8 if opened >= 3 else 0,
        "fresh_tokenization_authentication_count": opened,
        "fresh_example_count": 48 * opened,
        "fresh_next_token_position_count": 24_732 * opened,
        "native_baseline_evaluator_count": 1,
        "native_baseline_parameter_mutation_count": 0,
        "native_baseline_optimizer_reference_count": 0,
        "compiled_candidate_native_mlp_module_count": 0,
        "reserve_open_count": 0,
    }
    for name, expected in fixed.items():
        if type(row[name]) is not int or row[name] != expected:
            raise ValueError(f"runtime execution {name} drifted")
    if row["native_baseline_eval_mode"] is not True:
        raise ValueError("runtime native baseline must remain in eval mode")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:runtime-execution:v1\0",
        label="runtime execution",
    )
    return row


def _parse_post_terminal_cleanup(
    value: object,
    *,
    terminal: str,
) -> dict[str, object]:
    fields = {
        "schema",
        "format_version",
        "terminal_status",
        "runtime_execution_receipt_sha256",
        "runtime_session_binding_sha256",
        "fresh_panel_execution_receipt_sha256",
        "runtime_closed",
        "native_baseline_released",
        "native_baseline_instance_count_after_close",
        "compiled_candidate_released",
        "model_reference_count_after_close",
        "tokenizer_reference_count_after_close",
        "temporary_A_spools_removed",
        "temporary_materializations_removed",
        "temporary_fit_state_removed",
        "frozen_bundle_retained_for_publication",
        "reserve_open_count",
        "receipt_sha256",
    }
    row = _exact_dict(value, fields, label="post-terminal cleanup")
    if (
        row["schema"]
        != "fisher_graph.gemma3_layer_multimode.post_terminal_cleanup.v1"
        or type(row["format_version"]) is not int
        or row["format_version"] != 1
        or row["terminal_status"] != terminal
    ):
        raise ValueError("post-terminal cleanup header/status drifted")
    for name in (
        "runtime_execution_receipt_sha256",
        "runtime_session_binding_sha256",
        "fresh_panel_execution_receipt_sha256",
    ):
        _require_sha256(row[name], label=f"post-terminal cleanup {name}")
    for name in (
        "runtime_closed",
        "native_baseline_released",
        "compiled_candidate_released",
        "temporary_A_spools_removed",
        "temporary_materializations_removed",
        "temporary_fit_state_removed",
        "frozen_bundle_retained_for_publication",
    ):
        if row[name] is not True:
            raise ValueError(f"post-terminal cleanup {name} must be true")
    for name in (
        "native_baseline_instance_count_after_close",
        "model_reference_count_after_close",
        "tokenizer_reference_count_after_close",
        "reserve_open_count",
    ):
        if type(row[name]) is not int or row[name] != 0:
            raise ValueError(f"post-terminal cleanup {name} must be zero")
    _validate_worker_receipt_digest(
        row,
        domain=b"fisher_graph:layer-multimode:post-terminal-cleanup:v1\0",
        label="post-terminal cleanup",
    )
    return row


@dataclass(frozen=True, slots=True)
class _Replay:
    preflight: dict[str, object]
    teardown: dict[str, object]
    counters: dict[str, object]
    fresh_panel: dict[str, object]
    runtime: dict[str, object]
    cleanup: dict[str, object]
    materialization: object
    selection: object
    compilation: object
    bundle: dict[str, object]
    B3: object
    C3: object | None
    D3: object | None


def _replay_payload(payload: Mapping[str, object]) -> _Replay:
    if type(payload) is not dict or set(payload) != _REPORT_PAYLOAD_FIELDS:
        raise ValueError("layer-multimode report payload fields drifted")
    terminal = payload["terminal_status"]
    if type(terminal) is not str or terminal not in TERMINAL_STATUSES:
        raise ValueError("layer-multimode terminal status is invalid")
    preflight = _parse_preflight(payload["preflight_receipt"])
    if (
        preflight["test_only_nonproduction"] is not False
        or preflight["production_runtime_ready"] is not True
    ):
        raise ValueError("canonical reports require a production-ready preflight")
    _parse_reserve(payload["reserve_disclosure"])
    teardown = _parse_development_teardown(
        payload["development_teardown_receipt"]
    )
    counters = _parse_experiment_counters(
        payload["experiment_counters_receipt"],
        terminal=terminal,
    )
    fresh_panel = _parse_fresh_panel_execution(
        payload["fresh_panel_execution_receipt"],
        terminal=terminal,
    )
    runtime = _parse_runtime_execution(
        payload["runtime_execution_receipt"],
        terminal=terminal,
    )
    cleanup = _parse_post_terminal_cleanup(
        payload["post_terminal_cleanup_receipt"],
        terminal=terminal,
    )

    from .gemma3_layer_multimode_materialization import (
        LayerMLPFlowMaterializationInventoryReceipt,
    )
    from .gemma3_layer_multimode_selection import (
        LayerMultimodeSelectionReceipt,
    )
    from .gemma3_layer_multimode_compiler import (
        Gemma3LayerMultimodeCompilationReceipt,
    )
    from .gemma3_layer_multimode_evaluation import (
        B3TeacherForcedEvaluationReceipt,
        C3FullStackEvaluationReceipt,
        CompiledCandidateIdentityReceipt,
        D3FinalEvaluationReceipt,
        heldout_evaluation_policy_sha256,
    )

    materialization_state = payload["A_materialization_receipt"]
    if type(materialization_state) is not dict:
        raise TypeError("A materialization receipt must be an exact dict")
    materialization_sha = _require_sha256(
        materialization_state.get("receipt_sha256"),
        label="A materialization receipt_sha256",
    )
    materialization = LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
        materialization_state,
        expected_receipt_sha256=materialization_sha,
    )
    materialization.assert_integrity()

    selection_state = payload["selection_receipt"]
    if not isinstance(selection_state, Mapping):
        raise TypeError("selection receipt must be an object")
    selection = LayerMultimodeSelectionReceipt.from_state_dict(selection_state)
    selection.validate_integrity()
    if (
        len(selection.candidates)
        != setup.LAYER_COUNT * setup.CANDIDATE_COUNT_PER_LAYER
        or selection.production_fit_authorized is not True
        or selection.fit_policy_test_only is not False
        or selection.all_layers_selected is not True
        or selection.latency_claimed is not False
    ):
        raise ValueError("selection is not the complete production A grid")

    compilation_state = payload["compilation_receipt"]
    if type(compilation_state) is not dict:
        raise TypeError("compilation receipt must be an exact dict")
    compilation = Gemma3LayerMultimodeCompilationReceipt.from_state_dict(
        compilation_state
    )
    if (
        compilation.production_provenance_authorized is not True
        or compilation.production_shape_enforced is not True
        or compilation.native_mlp_fallback_present is not False
        or compilation.source_parameter_alias_present is not False
    ):
        raise ValueError("compiler production/source-free proof drifted")
    bundle = _parse_bundle_receipt(payload["bundle_receipt"])

    B3_state = payload["B3_evaluation"]
    if not isinstance(B3_state, Mapping):
        raise TypeError("B3 evaluation must be an object")
    B3 = B3TeacherForcedEvaluationReceipt.from_state_dict(B3_state)
    C3_state = payload["C3_evaluation"]
    D3_state = payload["D3_evaluation"]
    C3 = (
        None
        if C3_state is None
        else C3FullStackEvaluationReceipt.from_state_dict(C3_state)
    )
    D3 = (
        None
        if D3_state is None
        else D3FinalEvaluationReceipt.from_state_dict(D3_state)
    )

    if terminal == TERMINAL_B3_FAILED:
        if B3.passed is not False or C3 is not None or D3 is not None:
            raise ValueError("B3_FAILED terminal receipt sequence is invalid")
    elif terminal == TERMINAL_C3_FAILED:
        if (
            B3.passed is not True
            or C3 is None
            or C3.passed is not False
            or D3 is not None
        ):
            raise ValueError("C3_FAILED terminal receipt sequence is invalid")
    elif (
        B3.passed is not True
        or C3 is None
        or C3.passed is not True
        or D3 is None
    ):
        raise ValueError("D3_COMPLETED terminal receipt sequence is invalid")

    source_sha = materialization.source_model_fingerprint_sha256
    selection_sha = selection.artifact_sha256
    if (
        source_sha != preflight["source_model_sha256"]
        or source_sha != selection.source_model_sha256
        or source_sha != compilation.source_model_sha256
        or source_sha != compilation.recomputed_source_model_sha256
        or source_sha != B3.source_model_sha256
        or compilation.source_execution_fingerprint_sha256
        != materialization.source_execution_fingerprint_sha256
        or compilation.recomputed_source_execution_fingerprint_sha256
        != materialization.source_execution_fingerprint_sha256
        or selection_sha != compilation.selection_receipt_sha256
        or selection_sha != B3.selection_receipt_sha256
        or bundle["compilation_receipt_sha256"]
        != compilation.receipt_sha256
    ):
        raise ValueError("A/selection/compiler/B3 provenance does not close")
    if (
        counters["execution_preflight_receipt_sha256"]
        != preflight["receipt_sha256"]
        or counters["selection_receipt_sha256"] != selection_sha
        or teardown["source_model_sha256"] != source_sha
        or teardown["source_execution_fingerprint_sha256"]
        != materialization.source_execution_fingerprint_sha256
        or teardown["compilation_receipt_sha256"]
        != compilation.receipt_sha256
        or teardown["compiled_model_fingerprint_sha256"]
        != compilation.compiled_state_sha256
    ):
        raise ValueError("teardown/counters provenance differs from this run")
    if (
        fresh_panel["panel_manifest_sha256"]
        != preflight["panel_manifest_sha256"]
        or fresh_panel["panel_manifest_file_sha256"]
        != preflight["panel_manifest_file_sha256"]
        or fresh_panel["panel_receipt_sha256"]
        != preflight["panel_receipt_sha256"]
        or fresh_panel["panel_receipt_file_sha256"]
        != preflight["panel_receipt_file_sha256"]
        or fresh_panel["panel_bundle_sha256"]
        != preflight["panel_bundle_sha256"]
        or fresh_panel["selection_receipt_sha256"] != selection_sha
    ):
        raise ValueError("fresh-panel execution differs from preflight/selection")
    if (
        runtime["implementation_qualification_sha256"]
        != preflight["implementation_qualification_sha256"]
        or runtime["predecessor_A_capture_receipt_sha256"]
        != preflight["predecessor_A_capture_receipt_sha256"]
        or runtime["predecessor_runtime_execution_receipt_sha256"]
        != preflight["predecessor_runtime_execution_receipt_sha256"]
        or runtime["source_model_fingerprint_before_sha256"] != source_sha
        or runtime["source_model_fingerprint_after_sha256"] != source_sha
        or runtime["source_execution_fingerprint_before_sha256"]
        != materialization.source_execution_fingerprint_sha256
        or runtime["source_execution_fingerprint_after_sha256"]
        != materialization.source_execution_fingerprint_sha256
        or runtime["compiled_model_fingerprint_before_sha256"]
        != compilation.compiled_state_sha256
        or runtime["compiled_model_fingerprint_after_sha256"]
        != compilation.compiled_state_sha256
        or runtime["compiled_execution_fingerprint_before_sha256"]
        != compilation.compiled_execution_fingerprint_sha256
        or runtime["compiled_execution_fingerprint_after_sha256"]
        != compilation.compiled_execution_fingerprint_sha256
    ):
        raise ValueError("runtime execution fingerprints differ from run lineage")
    runtime_qualification = runtime["runtime_qualification_receipt"]
    if not isinstance(runtime_qualification, Mapping):
        raise TypeError("runtime qualification receipt must be an object")
    if (
        runtime_qualification["execution_preflight_receipt_sha256"]
        != preflight["receipt_sha256"]
        or runtime_qualification["implementation_qualification_file_sha256"]
        != preflight["implementation_qualification_file_sha256"]
        or runtime_qualification["qualified_runtime_source_sha256"]
        != runtime["runtime_implementation_sha256"]
        or runtime_qualification["evaluation_protocol_sha256"]
        != preflight["evaluation_protocol_sha256"]
    ):
        raise ValueError("runtime qualification differs from run preflight")
    predecessor_A_runtime = runtime["predecessor_A_runtime_receipt"]
    predecessor_A_comparison = runtime[
        "predecessor_A_lineage_comparison_receipt"
    ]
    if not isinstance(predecessor_A_runtime, Mapping) or not isinstance(
        predecessor_A_comparison,
        Mapping,
    ):
        raise TypeError("runtime predecessor A evidence must be objects")
    if (
        predecessor_A_runtime["execution_preflight_receipt_sha256"]
        != preflight["receipt_sha256"]
        or predecessor_A_runtime["predecessor_preflight_receipt_sha256"]
        != runtime_qualification["predecessor_preflight_receipt_sha256"]
        or predecessor_A_runtime[
            "predecessor_runtime_qualification_receipt_sha256"
        ]
        != runtime_qualification[
            "predecessor_runtime_qualification_receipt_sha256"
        ]
        or predecessor_A_runtime["source_capture_receipt_sha256"]
        != materialization.source_role_capture_receipt_sha256
        or predecessor_A_runtime["runtime_session_binding_sha256"]
        != materialization.source_runtime_session_binding_sha256
        or predecessor_A_runtime["source_model_fingerprint_before_sha256"]
        != source_sha
        or predecessor_A_runtime["source_execution_fingerprint_before_sha256"]
        != materialization.source_execution_fingerprint_sha256
        or predecessor_A_comparison["predecessor_capture_receipt_sha256"]
        != preflight["predecessor_A_capture_receipt_sha256"]
        or predecessor_A_comparison["recaptured_capture_receipt_sha256"]
        != materialization.source_role_capture_receipt_sha256
        or predecessor_A_comparison[
            "predecessor_runtime_execution_receipt_sha256"
        ]
        != preflight["predecessor_runtime_execution_receipt_sha256"]
        or predecessor_A_comparison[
            "recaptured_runtime_session_binding_sha256"
        ]
        != materialization.source_runtime_session_binding_sha256
    ):
        raise ValueError("predecessor A runtime/materialization lineage differs")
    for name in (
        "native_baseline_evaluator_count",
        "native_baseline_eval_mode",
        "native_baseline_parameter_mutation_count",
        "native_baseline_optimizer_reference_count",
        "compiled_candidate_native_mlp_module_count",
    ):
        if counters[name] != runtime[name]:
            raise ValueError("counter/runtime native-baseline ledger differs")
    if (
        teardown["native_baseline_instance_count"]
        != runtime["native_baseline_evaluator_count"]
        or cleanup["native_baseline_instance_count_after_close"] != 0
        or cleanup["runtime_execution_receipt_sha256"]
        != runtime["receipt_sha256"]
        or cleanup["runtime_session_binding_sha256"]
        != runtime["runtime_session_binding_sha256"]
        or cleanup["fresh_panel_execution_receipt_sha256"]
        != fresh_panel["receipt_sha256"]
    ):
        raise ValueError("native baseline retain/release ledger drifted")
    panel_sha = preflight["panel_manifest_sha256"]
    policy_sha = heldout_evaluation_policy_sha256()
    if (
        B3.panel_manifest_sha256 != panel_sha
        or B3.evaluation_protocol_sha256 != policy_sha
        or preflight["evaluation_protocol_sha256"] != policy_sha
    ):
        raise ValueError("B3 panel/evaluation policy differs from preflight")
    selected_by_sha = {
        value.artifact_sha256: value for value in selection.candidates
    }
    for layer, evidence in zip(selection.layers, B3.layer_evidence, strict=True):
        selected = selected_by_sha[layer.selected_candidate_sha256]
        if (
            evidence.selected_candidate_sha256 != selected.artifact_sha256
            or evidence.chart_count != selected.chart_count
            or evidence.dynamic_mode_count != selected.dynamic_mode_count
            or evidence.tangent_rank != selected.tangent_rank
        ):
            raise ValueError("B3 evidence does not identify the selected layer map")
    from . import gemma3_layer_multimode_fresh_panels as panel_contract

    expected_families = tuple(sorted(panel_contract.FAMILY_IDS))
    predicted_schedule = tuple(
        value - 1 for value in panel_contract.TARGET_TOKEN_COUNTS
    )
    expected_rows_per_family = sum(predicted_schedule)
    expected_rows = len(expected_families) * expected_rows_per_family
    if expected_rows != 24_732:
        raise RuntimeError("fresh-panel row-axis constant drifted")
    for evidence in B3.layer_evidence:
        if (
            tuple(
                aggregate.family_id
                for aggregate in evidence.family_aggregates
            )
            != expected_families
            or any(
                aggregate.scalar_count
                != expected_rows_per_family * setup.HIDDEN_WIDTH
                for aggregate in evidence.family_aggregates
            )
            or evidence.row_weight_sum != float(expected_rows)
        ):
            raise ValueError("B3 manifest family/row axes drifted")
    frozen_identity = CompiledCandidateIdentityReceipt(
        selection_receipt_sha256=selection_sha,
        compiler_protocol_sha256=hashlib.sha256(
            b"fisher_graph:layer-multimode:compiler-source:v1\0"
            + Path(__file__)
            .with_name("gemma3_layer_multimode_compiler.py")
            .read_bytes()
        ).hexdigest(),
        compile_receipt_sha256=compilation.receipt_sha256,
        bundle_semantic_sha256=bundle["manifest_sha256"],
        compiled_model_state_sha256=compilation.compiled_state_sha256,
    )
    fresh_captures = runtime["fresh_role_capture_receipts"]
    if not isinstance(fresh_captures, list):
        raise TypeError("fresh-role capture inventory must be a list")
    evaluations = [B3, *(item for item in (C3, D3) if item is not None)]
    for capture, evaluation in zip(
        fresh_captures,
        evaluations,
        strict=True,
    ):
        if not isinstance(capture, Mapping):
            raise TypeError("fresh-role capture must be an object")
        role_capture_sha = (
            B3.layer_evidence[0].role_capture_receipt_sha256
            if evaluation is B3
            else evaluation.role_capture_receipt_sha256
        )
        if (
            capture["panel_manifest_sha256"]
            != preflight["panel_manifest_sha256"]
            or capture["panel_manifest_file_sha256"]
            != preflight["panel_manifest_file_sha256"]
            or capture["panel_receipt_sha256"]
            != preflight["panel_receipt_sha256"]
            or capture["panel_receipt_file_sha256"]
            != preflight["panel_receipt_file_sha256"]
            or capture["panel_bundle_sha256"]
            != preflight["panel_bundle_sha256"]
            or capture["selection_receipt_sha256"] != selection_sha
            or capture["frozen_candidate_identity_sha256"]
            != frozen_identity.artifact_sha256
            or capture["runtime_session_binding_sha256"]
            != runtime["runtime_session_binding_sha256"]
            or capture["receipt_sha256"] != role_capture_sha
        ):
            raise ValueError("fresh-role capture/evaluation lineage differs")

    def assert_full_stack_axes(evaluation: object, *, label: str) -> None:
        records = evaluation.example_aggregates  # type: ignore[attr-defined]
        if len(records) != 48:
            raise ValueError(f"{label} example axis drifted")
        for family in expected_families:
            members = [value for value in records if value.family_id == family]
            if (
                len(members) != 8
                or tuple(sorted(value.predicted_token_count for value in members))
                != predicted_schedule
                or any(
                    value.hard_long
                    is not (
                        value.predicted_token_count + 1
                        >= panel_contract.HARD_LONG_MINIMUM_TOKEN_COUNT
                    )
                    for value in members
                )
            ):
                raise ValueError(f"{label} manifest family/row axes drifted")

    if C3 is not None:
        assert_full_stack_axes(C3, label="C3")
    if D3 is not None:
        assert_full_stack_axes(D3, label="D3")
    if C3 is not None:
        identity = C3.candidate_identity
        if (
            C3.source_model_sha256 != source_sha
            or C3.b3_evaluation_sha256 != B3.artifact_sha256
            or C3.panel_manifest_sha256 != panel_sha
            or C3.evaluation_protocol_sha256 != policy_sha
            or identity.selection_receipt_sha256 != selection_sha
            or identity.compile_receipt_sha256 != compilation.receipt_sha256
            or identity.bundle_semantic_sha256 != bundle["manifest_sha256"]
            or identity.compiled_model_state_sha256
            != compilation.compiled_state_sha256
        ):
            raise ValueError("C3 candidate/provenance differs from compiled A result")
    if D3 is not None:
        if C3 is None:
            raise ValueError("D3 cannot exist without C3")
        if (
            D3.source_model_sha256 != source_sha
            or D3.c3_evaluation_sha256 != C3.artifact_sha256
            or D3.panel_manifest_sha256 != panel_sha
            or D3.evaluation_protocol_sha256 != policy_sha
            or D3.candidate_identity.artifact_sha256
            != C3.candidate_identity.artifact_sha256
        ):
            raise ValueError("D3 is not the unchanged C3 candidate")
        resources = D3.resource_comparison
        trace = D3.no_native_mlp_trace
        native_boundary = sum(
            row.native_boundary_parameters for row in compilation.layer_accounting
        )
        compiled_boundary = sum(
            row.compiled_boundary_parameters for row in compilation.layer_accounting
        )
        if (
            resources.native_full_model_parameter_count
            != compilation.source_parameter_count
            or resources.native_replaced_boundary_parameter_count
            != native_boundary
            or resources.compiled_replacement_boundary_parameter_count
            != compiled_boundary
            or resources.measured_compiled_full_model_parameter_count
            != compilation.compiled_parameter_count
        ):
            raise ValueError("D3 exact resource evidence differs from compiler")
        if (
            trace.traced_forward_call_count
            != runtime["D3_compiled_forward_count"]
            or trace.traced_example_count != len(D3.example_aggregates)
            or trace.traced_predicted_token_count
            != sum(
                value.predicted_token_count for value in D3.example_aggregates
            )
            or trace.role_capture_receipt_sha256
            != D3.role_capture_receipt_sha256
            or trace.candidate_identity_sha256
            != D3.candidate_identity.artifact_sha256
        ):
            raise ValueError("D3 replacement-hook trace differs from runtime")

    evaluations = [B3, *(item for item in (C3, D3) if item is not None)]
    freezes = fresh_panel["ordered_freeze_receipts"]
    if not isinstance(freezes, list):
        raise TypeError("fresh-panel freeze ledger must be a list")
    for freeze, evaluation in zip(freezes, evaluations, strict=True):
        if not isinstance(freeze, Mapping):
            raise TypeError("fresh-panel freeze receipt must be an object")
        if (
            freeze["stage_receipt_sha256"] != evaluation.artifact_sha256
            or freeze["terminal_passed"] is not evaluation.passed
        ):
            raise ValueError("fresh-panel freeze does not bind its evaluation")

    compiler_selected_parameters = sum(
        row.compiled_boundary_parameters for row in compilation.layer_accounting
    )
    compiler_selected_macs = sum(
        setup.replacement_parameter_accounting(
            row.chart_count,
            row.modes_per_chart,
            row.tangent_rank,
        )["candidate_matrix_and_mixture_macs_per_token"]
        for row in compilation.layer_accounting
    )
    if (
        compiler_selected_parameters
        != selection.selected_serialized_parameter_count
        or compiler_selected_macs != selection.selected_matrix_macs_per_token
        or sum(
            row.native_boundary_parameters for row in compilation.layer_accounting
        )
        != selection.native_boundary_parameter_count
        or setup.LAYER_COUNT
        * setup.NATIVE_MLP_MATRIX_PARAMETER_COUNT_PER_LAYER
        != selection.native_matrix_macs_per_token
    ):
        raise ValueError("selection/compiler parameter or MAC accounting differs")

    _reject_raw_payloads(payload)
    return _Replay(
        preflight=preflight,
        teardown=teardown,
        counters=counters,
        fresh_panel=fresh_panel,
        runtime=runtime,
        cleanup=cleanup,
        materialization=materialization,
        selection=selection,
        compilation=compilation,
        bundle=bundle,
        B3=B3,
        C3=C3,
        D3=D3,
    )


def _claims(replay: _Replay, *, terminal_status: str) -> dict[str, object]:
    selection = replay.selection
    compilation = replay.compilation
    d3 = replay.D3
    d3_passed = d3 is not None and d3.passed is True
    return {
        "scientific_result_is_negative": (
            terminal_status != TERMINAL_D3_COMPLETED or not d3_passed
        ),
        "all_18_token_local_MLP_maps_replaced_source_free": d3_passed,
        "heldout_behavior_preserved_under_frozen_thresholds": d3_passed,
        "exact_parameter_reduction_claimed": d3_passed,
        "serialized_byte_reduction_claimed": d3_passed,
        "MAC_reduction_claimed": False,
        "generic_operation_reduction_claimed": False,
        "matrix_MAC_formula_accounting_reported": True,
        "latency_improvement_claimed": False,
        "latency_measurement_receipt_sha256": None,
        "native_baseline_retained_immutable_during_heldout_evaluation": True,
        "native_baseline_evaluator_count_during_heldout": (
            replay.runtime["native_baseline_evaluator_count"]
        ),
        "native_baseline_parameter_mutation_count": (
            replay.runtime["native_baseline_parameter_mutation_count"]
        ),
        "native_baseline_optimizer_reference_count": (
            replay.runtime["native_baseline_optimizer_reference_count"]
        ),
        "source_compilation_native_MLP_alias_count_before_B3": (
            replay.teardown["source_compilation_native_mlp_alias_count"]
        ),
        "compiled_candidate_native_MLP_module_count": (
            replay.runtime["compiled_candidate_native_mlp_module_count"]
        ),
        "native_baseline_released_after_terminal": (
            replay.cleanup["native_baseline_released"]
        ),
        "runtime_session_binding_sha256": (
            replay.runtime["runtime_session_binding_sha256"]
        ),
        "source_full_model_parameter_count": compilation.source_parameter_count,
        "compiled_full_model_parameter_count": compilation.compiled_parameter_count,
        "source_parameter_storage_bytes": (
            compilation.source_parameter_storage_bytes
        ),
        "compiled_parameter_storage_bytes": (
            compilation.compiled_parameter_storage_bytes
        ),
        "source_persistent_storage_bytes": (
            compilation.source_persistent_storage_bytes
        ),
        "compiled_persistent_storage_bytes": (
            compilation.compiled_persistent_storage_bytes
        ),
        "source_state_tensor_payload_bytes": (
            compilation.source_state_tensor_payload_bytes
        ),
        "compiled_state_tensor_payload_bytes": (
            compilation.compiled_state_tensor_payload_bytes
        ),
        "source_replaced_boundary_parameter_count": (
            selection.native_boundary_parameter_count
        ),
        "compiled_replacement_boundary_parameter_count": (
            selection.selected_serialized_parameter_count
        ),
        "replacement_boundary_parameter_savings": (
            selection.serialized_parameter_savings
        ),
        "source_matrix_MACs_per_token": selection.native_matrix_macs_per_token,
        "compiled_matrix_and_mixture_MACs_per_token": (
            selection.selected_matrix_macs_per_token
        ),
        "matrix_MAC_savings_per_token": selection.matrix_mac_savings_per_token,
        "coefficient_bundle_byte_count": replay.bundle["byte_count"],
        "coefficient_bundle_file_sha256": replay.bundle["bundle_sha256"],
    }


def _validate_completed_report(report: Mapping[str, object]) -> _Replay:
    expected = {
        "schema",
        "format_version",
        *_REPORT_PAYLOAD_FIELDS,
        "claims",
        "privacy",
        "summary_sha256",
        "report_sha256",
    }
    if type(report) is not dict or set(report) != expected:
        raise ValueError("completed report field inventory drifted")
    if (
        report["schema"] != REPORT_SCHEMA
        or type(report["format_version"]) is not int
        or report["format_version"] != REPORT_FORMAT_VERSION
    ):
        raise ValueError("completed report header drifted")
    unsigned = dict(report)
    observed = unsigned.pop("report_sha256")
    _require_sha256(observed, label="report_sha256")
    if observed != _digest(_REPORT_DOMAIN, unsigned):
        raise ValueError("completed report digest drifted")
    payload = {name: report[name] for name in _REPORT_PAYLOAD_FIELDS}
    replay = _replay_payload(payload)
    expected_claims = _claims(
        replay,
        terminal_status=report["terminal_status"],  # type: ignore[arg-type]
    )
    if canonical_json_bytes(report["claims"]) != canonical_json_bytes(
        expected_claims
    ):
        raise ValueError("report claims do not recompute")
    if report["privacy"] != _PRIVACY:
        raise ValueError("report privacy boundary drifted")
    if report["summary_sha256"] != _digest(
        _SUMMARY_DOMAIN,
        _summary_binding_payload(report),
    ):
        raise ValueError("report summary binding drifted")
    return replay


def _summary_payload(report: Mapping[str, object]) -> dict[str, object]:
    selection = report["selection_receipt"]
    compilation = report["compilation_receipt"]
    bundle = report["bundle_receipt"]
    B3 = report["B3_evaluation"]
    C3 = report["C3_evaluation"]
    D3 = report["D3_evaluation"]
    teardown = report["development_teardown_receipt"]
    counters = report["experiment_counters_receipt"]
    fresh_panel = report["fresh_panel_execution_receipt"]
    runtime = report["runtime_execution_receipt"]
    cleanup = report["post_terminal_cleanup_receipt"]
    summary_sources = (
        selection,
        compilation,
        bundle,
        teardown,
        counters,
        fresh_panel,
        runtime,
        cleanup,
        B3,
    )
    if not all(isinstance(value, Mapping) for value in summary_sources):
        raise TypeError("summary source receipts must be mappings")
    return {
        "schema": "fisher_graph.gemma3_layer_multimode.summary.v1",
        "format_version": 1,
        "report_sha256": report["report_sha256"],
        "terminal_status": report["terminal_status"],
        "selection_receipt_sha256": selection["artifact_sha256"],
        "compilation_receipt_sha256": compilation["receipt_sha256"],
        "bundle_file_sha256": bundle["bundle_sha256"],
        "bundle_byte_count": bundle["byte_count"],
        "development_teardown_receipt_sha256": teardown["receipt_sha256"],
        "experiment_counters_receipt_sha256": counters["receipt_sha256"],
        "fresh_panel_execution_receipt_sha256": fresh_panel["receipt_sha256"],
        "runtime_execution_receipt_sha256": runtime["receipt_sha256"],
        "post_terminal_cleanup_receipt_sha256": cleanup["receipt_sha256"],
        "source_model_fingerprint_sha256": (
            runtime["source_model_fingerprint_after_sha256"]
        ),
        "source_execution_fingerprint_sha256": (
            runtime["source_execution_fingerprint_after_sha256"]
        ),
        "compiled_model_fingerprint_sha256": (
            runtime["compiled_model_fingerprint_after_sha256"]
        ),
        "compiled_execution_fingerprint_sha256": (
            runtime["compiled_execution_fingerprint_after_sha256"]
        ),
        "runtime_session_binding_sha256": (
            runtime["runtime_session_binding_sha256"]
        ),
        "B3_evaluation_sha256": B3["artifact_sha256"],
        "B3_passed": B3["passed"],
        "C3_evaluation_sha256": (
            None if C3 is None else C3["artifact_sha256"]
        ),
        "C3_passed": None if C3 is None else C3["passed"],
        "D3_evaluation_sha256": (
            None if D3 is None else D3["artifact_sha256"]
        ),
        "D3_passed": None if D3 is None else D3["passed"],
        "claims": report["claims"],
        "privacy": report["privacy"],
    }


def _summary_binding_payload(report: Mapping[str, object]) -> dict[str, object]:
    payload = _summary_payload(report)
    # report_sha256 is added to the rendered sidecar for navigation, but is
    # excluded from the semantic sidecar identity to avoid a report/sidecar
    # digest fixed-point cycle.  The strict loader separately requires it to
    # equal the containing report digest.
    payload["report_sha256"] = None
    return payload


def seal_layer_multimode_experiment_report(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != _REPORT_PAYLOAD_FIELDS:
        raise ValueError("layer-multimode report payload fields drifted")
    replay = _replay_payload(payload)
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        **dict(payload),
        "claims": _claims(
            replay,
            terminal_status=payload["terminal_status"],  # type: ignore[arg-type]
        ),
        "privacy": dict(_PRIVACY),
        "summary_sha256": "",
    }
    report["summary_sha256"] = _digest(
        _SUMMARY_DOMAIN,
        _summary_binding_payload({**report, "report_sha256": ""}),
    )
    report["report_sha256"] = _digest(_REPORT_DOMAIN, report)
    _validate_completed_report(report)
    return report


def strict_load_layer_multimode_experiment_report_bytes(
    raw: bytes,
) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_REPORT_BYTES:
        raise ValueError("layer-multimode report byte size is invalid")
    value = _strict_json_loads(raw, label="layer-multimode report")
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError("layer-multimode report must be canonical JSON")
    _validate_completed_report(value)
    return value


def render_layer_multimode_summary_bytes(
    report: Mapping[str, object],
) -> bytes:
    _validate_completed_report(report)
    payload = _summary_payload(report)
    payload["summary_sha256"] = _digest(
        _SUMMARY_DOMAIN,
        _summary_binding_payload(report),
    )
    return canonical_json_bytes(payload)


def _strict_load_summary_bytes(
    raw: bytes,
    *,
    report: Mapping[str, object],
) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_SUMMARY_BYTES:
        raise ValueError("layer-multimode summary byte size is invalid")
    value = _strict_json_loads(raw, label="layer-multimode summary")
    expected_fields = set(_summary_payload(report)) | {"summary_sha256"}
    row = _exact_dict(value, expected_fields, label="layer-multimode summary")
    if canonical_json_bytes(row) != raw:
        raise ValueError("layer-multimode summary must be canonical JSON")
    observed = _require_sha256(
        row.pop("summary_sha256"),
        label="summary summary_sha256",
    )
    if (
        row != _summary_payload(report)
        or row["report_sha256"] != report["report_sha256"]
        or observed != report["summary_sha256"]
        or observed != _digest(_SUMMARY_DOMAIN, _summary_binding_payload(report))
    ):
        raise ValueError("layer-multimode summary differs from report")
    return {**row, "summary_sha256": observed}


def _strict_live_rehash(
    report: Mapping[str, object],
    *,
    repository_root: Path,
) -> None:
    root = repository_root.resolve()
    preflight = _parse_preflight(report["preflight_receipt"])
    runtime_execution = _parse_runtime_execution(
        report["runtime_execution_receipt"],
        terminal=report["terminal_status"],  # type: ignore[arg-type]
    )
    runtime_qualification = runtime_execution["runtime_qualification_receipt"]
    if not isinstance(runtime_qualification, Mapping):
        raise ValueError("runtime qualification receipt is absent")
    setup_path = root / setup.DEFAULT_OUTPUT
    setup_raw = _read_regular(
        setup_path,
        label="live layer-multimode setup",
        maximum_bytes=8 * 1024 * 1024,
    )
    setup_value = _strict_json_loads(setup_raw, label="live layer-multimode setup")
    if not isinstance(setup_value, Mapping):
        raise ValueError("live layer-multimode setup must be an object")
    setup_state = setup.check_setup(
        setup_path,
        repository_root=root,
        verify_predecessor=True,
    )
    if (
        setup_state["setup_sha256"] != preflight["setup_sha256"]
        or hashlib.sha256(setup_raw).hexdigest()
        != preflight["setup_file_sha256"]
    ):
        raise ValueError("live layer-multimode setup differs from preflight")
    provenance = setup_state.get("methodology_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("live setup predecessor provenance is absent")
    if (
        provenance.get("predecessor_v2_setup_sha256")
        != preflight["predecessor_v2_setup_sha256"]
        or provenance.get("predecessor_v2_report_sha256")
        != preflight["predecessor_v2_report_sha256"]
        or provenance.get("predecessor_v2_A_capture_receipt_sha256")
        != preflight["predecessor_A_capture_receipt_sha256"]
        or provenance.get(
            "predecessor_v2_runtime_execution_receipt_sha256"
        )
        != preflight["predecessor_runtime_execution_receipt_sha256"]
        or runtime_qualification.get("predecessor_preflight_receipt_sha256")
        != provenance.get("predecessor_v2_preflight_receipt_sha256")
        or runtime_qualification.get(
            "predecessor_runtime_qualification_receipt_sha256"
        )
        != provenance.get(
            "predecessor_v2_runtime_qualification_receipt_sha256"
        )
        or runtime_execution["source_model_fingerprint_before_sha256"]
        != provenance.get("predecessor_v2_source_model_fingerprint_sha256")
        or runtime_execution["source_execution_fingerprint_before_sha256"]
        != provenance.get(
            "predecessor_v2_source_execution_fingerprint_sha256"
        )
    ):
        raise ValueError(
            "live predecessor v2/preflight/A/qualification/runtime lineage differs"
        )
    predecessor_comparison = runtime_execution[
        "predecessor_A_lineage_comparison_receipt"
    ]
    if (
        not isinstance(predecessor_comparison, Mapping)
        or predecessor_comparison.get(
            "predecessor_runtime_session_binding_sha256"
        )
        != provenance.get("predecessor_v2_runtime_session_binding_sha256")
    ):
        raise ValueError("live predecessor runtime-session lineage differs")

    from . import gemma3_layer_multimode_implementation_qualification as qualified

    qualification_path = root / qualified.DEFAULT_OUTPUT
    qualification_raw = _read_regular(
        qualification_path,
        label="live layer-multimode implementation qualification",
        maximum_bytes=16 * 1024 * 1024,
    )
    qualification_state = qualified.load_qualification(
        qualification_path,
        repository_root=root,
    )
    if (
        qualification_state["qualification_sha256"]
        != preflight["implementation_qualification_sha256"]
        or hashlib.sha256(qualification_raw).hexdigest()
        != preflight["implementation_qualification_file_sha256"]
    ):
        raise ValueError("live implementation qualification differs from preflight")
    source_closure = qualification_state.get("source_closure")
    if type(source_closure) is not list:
        raise ValueError("live implementation source closure is absent")
    runtime_source_rows = [
        row
        for row in source_closure
        if isinstance(row, Mapping)
        and row.get("module")
        == "fisher_graph.gemma3_layer_multimode_runtime"
    ]
    if (
        len(runtime_source_rows) != 1
        or runtime_source_rows[0].get("file_sha256")
        != runtime_execution["runtime_implementation_sha256"]
        or runtime_qualification.get("execution_preflight_receipt_sha256")
        != preflight["receipt_sha256"]
        or runtime_qualification.get("implementation_qualification_sha256")
        != preflight["implementation_qualification_sha256"]
        or runtime_qualification.get("evaluation_protocol_sha256")
        != preflight["evaluation_protocol_sha256"]
    ):
        raise ValueError(
            "live runtime source/qualification/preflight binding differs"
        )

    from . import gemma3_layer_multimode_fresh_panels as panels

    panel_path = root / panels.DEFAULT_RECEIPT
    panel_raw = _read_regular(
        panel_path,
        label="live layer-multimode panel receipt",
        maximum_bytes=16 * 1024 * 1024,
    )
    panel_value = panels._strict_json_loads(
        panel_raw,
        label="live layer-multimode panel receipt",
    )
    if type(panel_value) is not dict:
        raise ValueError("live panel receipt must be a canonical object")
    validated_panel = panels.validate_receipt(panel_value)
    if panels.canonical_json_bytes(validated_panel) != panel_raw:
        raise ValueError("live panel receipt is not canonical")
    manifest_binding = validated_panel["manifest"]
    if not isinstance(manifest_binding, Mapping):
        raise ValueError("live panel manifest binding is invalid")
    panel_sha = manifest_binding.get("manifest_sha256")
    panel_preflight = panels.load_frozen_preflight(
        repository_root=root,
        expected_receipt_file_sha256=preflight["panel_receipt_file_sha256"],
    )
    if (
        panel_sha != preflight["panel_manifest_sha256"]
        or panel_preflight.manifest_file_sha256
        != preflight["panel_manifest_file_sha256"]
        or validated_panel["receipt_sha256"]
        != preflight["panel_receipt_sha256"]
        or panel_preflight.receipt_sha256
        != preflight["panel_receipt_sha256"]
        or hashlib.sha256(panel_raw).hexdigest()
        != preflight["panel_receipt_file_sha256"]
        or panel_preflight.receipt_file_sha256
        != preflight["panel_receipt_file_sha256"]
        or validated_panel["bundle_sha256"] != preflight["panel_bundle_sha256"]
        or panel_preflight.bundle_sha256 != preflight["panel_bundle_sha256"]
    ):
        raise ValueError("live panel bundle differs from preflight")

    from . import gemma3_layer_multimode_runtime as runtime_contract

    live_manifest = panel_preflight.manifest
    live_roles = live_manifest.get("roles")
    if not isinstance(live_roles, Mapping):
        raise ValueError("live panel role manifest is absent")
    fresh_captures = runtime_execution["fresh_role_capture_receipts"]
    if type(fresh_captures) is not list:
        raise ValueError("runtime fresh-role capture inventory is absent")
    expected_roles = _expected_fresh_roles(report["terminal_status"])
    for expected_role, capture in zip(
        expected_roles,
        fresh_captures,
        strict=True,
    ):
        if not isinstance(capture, Mapping):
            raise ValueError("runtime fresh-role capture is invalid")
        descriptor = live_roles.get(expected_role)
        if not isinstance(descriptor, Mapping):
            raise ValueError("live panel opened-role descriptor is absent")
        (
            example_catalog_sha256,
            row_catalog_sha256,
            family_axis_sha256,
            batch_axis_sha256,
        ) = runtime_contract._fresh_role_catalog_sha256s(
            manifest=live_manifest,
            role=expected_role,
        )
        if (
            capture["role_file_sha256"] != descriptor.get("file_sha256")
            or capture["ordered_example_identity_catalog_sha256"]
            != example_catalog_sha256
            or capture["ordered_row_identity_catalog_sha256"]
            != row_catalog_sha256
            or capture["family_axis_sha256"] != family_axis_sha256
            or capture["batch_axis_sha256"] != batch_axis_sha256
        ):
            raise ValueError(
                "fresh-role capture differs from live frozen panel axes"
            )

        evaluation = {
            setup.CONFIRMATION_ROLE: report["C3_evaluation"],
            setup.FINAL_ROLE: report["D3_evaluation"],
        }.get(expected_role)
        if evaluation is None:
            continue
        if not isinstance(evaluation, Mapping):
            raise ValueError("live full-stack evaluation is absent")
        aggregates = evaluation.get("example_aggregates")
        members = descriptor.get("ordered_members")
        if type(aggregates) is not list or type(members) is not list:
            raise ValueError("live full-stack example axis is absent")
        expected_rows: list[tuple[str, str, bool, int]] = []
        for member in members:
            if not isinstance(member, Mapping):
                raise ValueError("live panel member is invalid")
            identity = _digest(
                runtime_contract._FRESH_EXAMPLE_IDENTITY_DOMAIN,
                {
                    "role": expected_role,
                    "example_id": member.get("example_id"),
                    "family_id": member.get("family_id"),
                    "hard_long": member.get("hard_long"),
                    "target_token_count": member.get("target_token_count"),
                    "predicted_token_count": member.get(
                        "predicted_token_count"
                    ),
                    "token_ids_sha256": member.get("token_ids_sha256"),
                    "semantic_lineage_sha256": member.get(
                        "semantic_lineage_sha256"
                    ),
                },
            )
            expected_rows.append(
                (
                    identity,
                    member.get("family_id"),
                    member.get("hard_long"),
                    member.get("predicted_token_count"),
                )
            )
        observed_rows = []
        for aggregate in aggregates:
            if not isinstance(aggregate, Mapping):
                raise ValueError("live full-stack aggregate is invalid")
            observed_rows.append(
                (
                    aggregate.get("example_identity_sha256"),
                    aggregate.get("family_id"),
                    aggregate.get("hard_long"),
                    aggregate.get("predicted_token_count"),
                )
            )
        if sorted(observed_rows) != sorted(expected_rows):
            raise ValueError(
                "full-stack evaluation differs from live frozen example axis"
            )


def strict_load_layer_multimode_experiment_report(
    report_path: Path | str,
    *,
    bundle_path: Path | str | None = None,
    summary_path: Path | str | None = None,
    repository_root: Path | str | None = None,
    verify_live_authorization: bool = True,
) -> dict[str, object]:
    report_source = Path(report_path)
    bundle_source = (
        report_source.parent / DEFAULT_BUNDLE_OUTPUT.name
        if bundle_path is None
        else Path(bundle_path)
    )
    summary_source = (
        report_source.parent / DEFAULT_SUMMARY_OUTPUT.name
        if summary_path is None
        else Path(summary_path)
    )
    paths = (report_source, bundle_source, summary_source)
    if (
        len(set(paths)) != 3
        or {path.parent for path in paths} != {report_source.parent}
        or report_source.name != DEFAULT_REPORT_OUTPUT.name
        or bundle_source.name != DEFAULT_BUNDLE_OUTPUT.name
        or summary_source.name != DEFAULT_SUMMARY_OUTPUT.name
    ):
        raise ValueError("published report/bundle/summary names or parents drifted")
    _regular_parent(report_source.parent)
    directory_descriptors, directory_identities = _open_directory_chain(
        report_source.parent
    )
    parent_descriptor = directory_descriptors[-1]
    primary: BaseException | None = None
    report: dict[str, object] | None = None
    try:
        report_raw, _ = _read_regular_at(
            parent_descriptor,
            report_source.name,
            label="published layer-multimode report",
            maximum_bytes=_MAX_REPORT_BYTES,
        )
        report = strict_load_layer_multimode_experiment_report_bytes(report_raw)
        bundle_raw, _ = _read_regular_at(
            parent_descriptor,
            bundle_source.name,
            label="published layer-multimode bundle",
            maximum_bytes=_MAX_BUNDLE_BYTES,
        )
        bundle = _parse_bundle_receipt(report["bundle_receipt"])
        if (
            len(bundle_raw) != bundle["byte_count"]
            or hashlib.sha256(bundle_raw).hexdigest()
            != bundle["bundle_sha256"]
        ):
            raise ValueError("published bundle byte identity differs from report")
        summary_raw, _ = _read_regular_at(
            parent_descriptor,
            summary_source.name,
            label="published layer-multimode summary",
            maximum_bytes=_MAX_SUMMARY_BYTES,
        )
        _strict_load_summary_bytes(summary_raw, report=report)
        _verify_directory_chain(report_source.parent, directory_identities)
    except BaseException as error:
        primary = error
    close_failures = _close_descriptors(
        directory_descriptors,
        primary=primary,
        label="strict-loader directory descriptor cleanup",
    )
    if primary is None and close_failures:
        primary = RuntimeError("strict-loader descriptor cleanup failed")
        for error in close_failures:
            primary.add_note(f"{type(error).__name__}: {error}")
    if primary is not None:
        raise primary
    if report is None:
        raise AssertionError("strict loader completed without a report")
    if verify_live_authorization:
        root = (
            Path(__file__).resolve().parents[2]
            if repository_root is None
            else Path(repository_root)
        )
        _strict_live_rehash(report, repository_root=root)
    return report


@dataclass(frozen=True, slots=True)
class PublishedLayerMultimodeExperimentReceipt:
    bundle_path: str
    summary_path: str
    report_path: str
    bundle_file_sha256: str
    bundle_byte_count: int
    summary_file_sha256: str
    report_file_sha256: str
    report_sha256: str
    publication_receipt_sha256: str

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": PUBLICATION_SCHEMA,
            "format_version": PUBLICATION_FORMAT_VERSION,
            "bundle_path": self.bundle_path,
            "summary_path": self.summary_path,
            "report_path": self.report_path,
            "bundle_file_sha256": self.bundle_file_sha256,
            "bundle_byte_count": self.bundle_byte_count,
            "summary_file_sha256": self.summary_file_sha256,
            "report_file_sha256": self.report_file_sha256,
            "report_sha256": self.report_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "publication_receipt_sha256": self.publication_receipt_sha256,
        }

    def validate_receipt(self) -> None:
        for name in (
            "bundle_file_sha256",
            "summary_file_sha256",
            "report_file_sha256",
            "report_sha256",
            "publication_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"publication {name}")
        if type(self.bundle_byte_count) is not int or self.bundle_byte_count <= 0:
            raise ValueError("publication bundle byte count is invalid")
        paths = tuple(
            Path(value)
            for value in (self.bundle_path, self.summary_path, self.report_path)
        )
        if (
            len(set(paths)) != 3
            or {path.parent for path in paths} != {paths[0].parent}
            or paths[0].name != DEFAULT_BUNDLE_OUTPUT.name
            or paths[1].name != DEFAULT_SUMMARY_OUTPUT.name
            or paths[2].name != DEFAULT_REPORT_OUTPUT.name
        ):
            raise ValueError("publication path contract drifted")
        expected = _digest(_PUBLICATION_DOMAIN, self.payload_without_sha256())
        if self.publication_receipt_sha256 != expected:
            raise ValueError("publication receipt digest drifted")


@dataclass(frozen=True, slots=True)
class _PublicationAuthority:
    publication: PublishedLayerMultimodeExperimentReceipt
    identities: tuple[tuple[Path, tuple[int, int, int, int, int]], ...]
    directory_identity_chain: tuple[tuple[int, int], ...]


_PUBLICATION_AUTHORITIES: dict[int, _PublicationAuthority] = {}
_PUBLICATION_LOCK = threading.Lock()


def _target_specs(
    publication: PublishedLayerMultimodeExperimentReceipt,
) -> tuple[tuple[Path, str, int, str], ...]:
    return (
        (
            Path(publication.bundle_path),
            publication.bundle_file_sha256,
            _MAX_BUNDLE_BYTES,
            "bundle",
        ),
        (
            Path(publication.summary_path),
            publication.summary_file_sha256,
            _MAX_SUMMARY_BYTES,
            "summary",
        ),
        (
            Path(publication.report_path),
            publication.report_file_sha256,
            _MAX_REPORT_BYTES,
            "report",
        ),
    )


def _authenticate_targets_at(
    publication: PublishedLayerMultimodeExperimentReceipt,
    authority: _PublicationAuthority,
    *,
    parent_descriptor: int,
    allow_missing: bool,
) -> dict[Path, tuple[int, int, int, int, int]]:
    issued = dict(authority.identities)
    verified: dict[Path, tuple[int, int, int, int, int]] = {}
    for path, expected_sha, maximum, label in _target_specs(publication):
        try:
            raw, identity = _read_regular_at(
                parent_descriptor,
                path.name,
                label=f"published {label}",
                maximum_bytes=maximum,
                expected_identity=issued.get(path),
            )
        except FileNotFoundError:
            if allow_missing:
                continue
            raise RuntimeError(f"published {label} disappeared")
        except ValueError as error:
            raise RuntimeError(
                f"published {label} identity drifted"
            ) from error
        if (
            issued.get(path) != identity
            or hashlib.sha256(raw).hexdigest() != expected_sha
        ):
            raise RuntimeError(f"published {label} identity drifted")
        verified[path] = identity
    return verified


def _authenticate_targets(
    publication: PublishedLayerMultimodeExperimentReceipt,
    authority: _PublicationAuthority,
    *,
    allow_missing: bool,
) -> dict[Path, tuple[int, int, int, int, int]]:
    parent = Path(publication.report_path).parent
    _regular_parent(parent)
    descriptors, identities = _open_directory_chain(parent)
    primary: BaseException | None = None
    verified: dict[Path, tuple[int, int, int, int, int]] | None = None
    try:
        if identities != authority.directory_identity_chain:
            raise RuntimeError("published output directory identity drifted")
        verified = _authenticate_targets_at(
            publication,
            authority,
            parent_descriptor=descriptors[-1],
            allow_missing=allow_missing,
        )
        _verify_directory_chain(parent, identities)
    except BaseException as error:
        primary = error
    close_errors = _close_descriptors(
        descriptors,
        primary=primary,
        label="publication target-authentication descriptor cleanup",
    )
    if primary is None and close_errors:
        primary = close_errors[0]
        for error in close_errors[1:]:
            primary.add_note(f"{type(error).__name__}: {error}")
    if primary is not None:
        raise primary
    if verified is None:
        raise AssertionError("publication authentication produced no result")
    return verified


def publish_layer_multimode_experiment_outputs(
    *,
    report: Mapping[str, object],
    bundle_source_path: Path | str,
    report_path: Path | str,
    bundle_path: Path | str,
    summary_path: Path | str,
    repository_root: Path | str | None = None,
    verify_live_authorization: bool = True,
) -> PublishedLayerMultimodeExperimentReceipt:
    report_destination = Path(report_path)
    bundle_destination = Path(bundle_path)
    summary_destination = Path(summary_path)
    destinations = (bundle_destination, summary_destination, report_destination)
    if (
        len(set(destinations)) != 3
        or {path.parent for path in destinations} != {report_destination.parent}
        or report_destination.name != DEFAULT_REPORT_OUTPUT.name
        or bundle_destination.name != DEFAULT_BUNDLE_OUTPUT.name
        or summary_destination.name != DEFAULT_SUMMARY_OUTPUT.name
    ):
        raise ValueError("publication destinations differ from frozen contract")
    _regular_parent(report_destination.parent)
    source = Path(bundle_source_path)
    if source in destinations:
        raise ValueError("bundle source must be a distinct staging file")
    _regular_parent(source.parent)

    report_raw = canonical_json_bytes(dict(report))
    loaded = strict_load_layer_multimode_experiment_report_bytes(report_raw)
    summary_raw = render_layer_multimode_summary_bytes(loaded)
    source_before = source.lstat()
    bundle_raw = _read_regular(
        source,
        label="staged layer-multimode bundle",
        maximum_bytes=_MAX_BUNDLE_BYTES,
    )
    if _file_identity(source_before) != _file_identity(source.lstat()):
        raise RuntimeError("staged bundle changed across read")
    bundle_receipt = _parse_bundle_receipt(loaded["bundle_receipt"])
    bundle_sha = hashlib.sha256(bundle_raw).hexdigest()
    if (
        bundle_sha != bundle_receipt["bundle_sha256"]
        or len(bundle_raw) != bundle_receipt["byte_count"]
    ):
        raise ValueError("staged bundle differs from report receipt")
    payload = {
        "bundle_path": str(bundle_destination),
        "summary_path": str(summary_destination),
        "report_path": str(report_destination),
        "bundle_file_sha256": bundle_sha,
        "bundle_byte_count": len(bundle_raw),
        "summary_file_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "report_file_sha256": hashlib.sha256(report_raw).hexdigest(),
        "report_sha256": loaded["report_sha256"],
    }
    publication = PublishedLayerMultimodeExperimentReceipt(
        **payload,
        publication_receipt_sha256=_digest(
            _PUBLICATION_DOMAIN,
            {
                "schema": PUBLICATION_SCHEMA,
                "format_version": PUBLICATION_FORMAT_VERSION,
                **payload,
            },
        ),
    )
    publication.validate_receipt()

    raw_rows = (
        (bundle_destination, bundle_raw),
        (summary_destination, summary_raw),
        (report_destination, report_raw),
    )
    directory_descriptors, directory_identities = _open_directory_chain(
        report_destination.parent
    )
    parent_descriptor = directory_descriptors[-1]
    temporaries: list[str | None] = []
    temporary_inodes: list[tuple[int, int] | None] = []
    linked_inodes: dict[Path, tuple[int, int]] = {}
    published_identities: dict[Path, tuple[int, int, int, int, int]] = {}
    primary: BaseException | None = None

    def guarded_unlink_at(
        directory_descriptor: int,
        name: str,
        identity: tuple[int, int],
        *,
        label: str,
    ) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        guard = os.open(name, flags, dir_fd=directory_descriptor)
        unlink_error: BaseException | None = None
        try:
            guarded = _descriptor_stat(guard)
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
            after = _descriptor_stat(guard)
            if (
                _inode_identity(after) != identity
                or after.st_nlink != before_links - 1
            ):
                raise RuntimeError(f"{label} unlink did not close")
        except BaseException as error:
            unlink_error = error
        finally:
            try:
                os.close(guard)
            except BaseException as close_error:
                if unlink_error is None:
                    unlink_error = close_error
                else:
                    unlink_error.add_note(
                        f"{label} guard close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
        if unlink_error is not None:
            raise unlink_error

    def cleanup_temporaries() -> list[BaseException]:
        errors: list[BaseException] = []
        removed = False
        for index, temporary in enumerate(temporaries):
            if temporary is None:
                continue
            identity = temporary_inodes[index]
            if identity is None:
                errors.append(
                    RuntimeError(
                        "publication temporary identity was not established"
                    )
                )
                continue
            try:
                guarded_unlink_at(
                    parent_descriptor,
                    temporary,
                    identity,
                    label="publication temporary",
                )
                temporaries[index] = None
                temporary_inodes[index] = None
                removed = True
            except FileNotFoundError:
                temporaries[index] = None
                temporary_inodes[index] = None
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
        if removed:
            try:
                os.fsync(parent_descriptor)
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
        return errors

    def rollback_targets() -> list[BaseException]:
        errors: list[BaseException] = []

        def unlink_owned(destination: Path) -> bool:
            identity = linked_inodes.get(destination)
            if identity is None:
                return False
            try:
                guarded_unlink_at(
                    parent_descriptor,
                    destination.name,
                    identity,
                    label="publication rollback target",
                )
                return True
            except FileNotFoundError:
                return False
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
                return False

        unlink_owned(report_destination)
        # A fault may be injected after unlink has taken effect.  Always make
        # the commit-marker rollback phase durable, even when the unlink path
        # reported an error.
        try:
            os.fsync(parent_descriptor)
        except BaseException as cleanup_error:
            errors.append(cleanup_error)
        unlink_owned(summary_destination)
        unlink_owned(bundle_destination)
        # Sidecars form a second rollback phase and receive their own durable
        # directory commit regardless of an individual unlink result.
        try:
            os.fsync(parent_descriptor)
        except BaseException as cleanup_error:
            errors.append(cleanup_error)
        return errors

    def late_rollback_targets() -> list[BaseException]:
        errors: list[BaseException] = []
        descriptors: tuple[int, ...] = ()
        try:
            descriptors, observed = _open_directory_chain(
                report_destination.parent
            )
            if observed != directory_identities:
                raise RuntimeError(
                    "publication directory changed before late rollback"
                )
            late_parent = descriptors[-1]

            def unlink_owned(destination: Path) -> bool:
                identity = linked_inodes.get(destination)
                if identity is None:
                    return False
                try:
                    guarded_unlink_at(
                        late_parent,
                        destination.name,
                        identity,
                        label="late publication rollback target",
                    )
                    return True
                except FileNotFoundError:
                    return False

            try:
                unlink_owned(report_destination)
            except BaseException as error:
                errors.append(error)
            try:
                os.fsync(late_parent)
            except BaseException as error:
                errors.append(error)
            for sidecar in (summary_destination, bundle_destination):
                try:
                    unlink_owned(sidecar)
                except BaseException as error:
                    errors.append(error)
            try:
                os.fsync(late_parent)
            except BaseException as error:
                errors.append(error)
        except BaseException as error:
            errors.append(error)
        if descriptors:
            errors.extend(
                _close_descriptors(
                    descriptors,
                    primary=errors[0] if errors else None,
                    label="late publication rollback descriptor cleanup",
                )
            )
        return errors

    def allocate_temporary(destination: Path, raw: bytes) -> None:
        descriptor = -1
        temporary = ""
        index = -1
        allocation_error: BaseException | None = None
        try:
            for _attempt in range(128):
                candidate = f".{destination.name}.{secrets.token_hex(16)}.tmp"
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
                temporary = candidate
                break
            if descriptor < 0 or not temporary:
                raise RuntimeError("could not allocate publication temporary")
            temporaries.append(temporary)
            temporary_inodes.append(None)
            index = len(temporaries) - 1
            identity = _inode_identity(_descriptor_stat(descriptor))
            temporary_inodes[index] = identity
            os.fchmod(descriptor, 0o600)
            view = memoryview(raw)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise RuntimeError("publication temporary write stalled")
                written += count
            os.fsync(descriptor)
            current = _descriptor_stat(descriptor)
            if (
                not stat.S_ISREG(current.st_mode)
                or _inode_identity(current) != identity
                or current.st_size != len(raw)
            ):
                raise RuntimeError("publication temporary changed while writing")
        except BaseException as error:
            allocation_error = error
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException as close_error:
                    if allocation_error is None:
                        allocation_error = close_error
                    else:
                        allocation_error.add_note(
                            "temporary descriptor close also failed: "
                            f"{type(close_error).__name__}: {close_error}"
                        )
        if allocation_error is not None:
            raise allocation_error
        if index < 0:
            raise AssertionError("temporary allocation did not register state")

    def link_and_unlink_temporary(index: int, destination: Path) -> None:
        temporary = temporaries[index]
        identity = temporary_inodes[index]
        if temporary is None or identity is None:
            raise RuntimeError("publication temporary was lost")
        current = _stat_at(parent_descriptor, temporary)
        if (
            not stat.S_ISREG(current.st_mode)
            or _inode_identity(current) != identity
            or current.st_nlink != 1
        ):
            raise RuntimeError("publication temporary identity drifted before link")
        # Register before link.  A fault raised after the link side effect is
        # therefore rolled back, while an unrelated raced target is preserved
        # because rollback requires this exact inode.
        linked_inodes[destination] = identity
        os.link(
            temporary,
            destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published = _stat_at(parent_descriptor, destination.name)
        if (
            not stat.S_ISREG(published.st_mode)
            or _inode_identity(published) != identity
            or published.st_nlink != 2
        ):
            raise RuntimeError("published target identity drifted while linking")
        guard_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        guard = os.open(temporary, guard_flags, dir_fd=parent_descriptor)
        guard_error: BaseException | None = None
        try:
            guarded = _descriptor_stat(guard)
            named = _stat_at(parent_descriptor, temporary)
            if (
                _inode_identity(guarded) != identity
                or _inode_identity(named) != identity
                or guarded.st_nlink != 2
            ):
                raise RuntimeError(
                    "publication temporary was replaced before unlink"
                )
            os.unlink(temporary, dir_fd=parent_descriptor)
            after = _descriptor_stat(guard)
            if _inode_identity(after) != identity or after.st_nlink != 1:
                raise RuntimeError("publication temporary unlink did not close")
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
                        "temporary guard close also failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
        if guard_error is not None:
            raise guard_error
        temporaries[index] = None
        temporary_inodes[index] = None
        final = _stat_at(parent_descriptor, destination.name)
        published_identities[destination] = _file_identity(final)

    try:
        for destination in destinations:
            try:
                _stat_at(parent_descriptor, destination.name)
            except FileNotFoundError:
                continue
            raise FileExistsError(destination)
        for destination, raw in raw_rows:
            allocate_temporary(destination, raw)

        link_and_unlink_temporary(0, bundle_destination)
        link_and_unlink_temporary(1, summary_destination)
        os.fsync(parent_descriptor)

        if verify_live_authorization:
            root = (
                Path(__file__).resolve().parents[2]
                if repository_root is None
                else Path(repository_root)
            )
            _strict_live_rehash(loaded, repository_root=root)

        link_and_unlink_temporary(2, report_destination)
        os.fsync(parent_descriptor)
        reloaded = strict_load_layer_multimode_experiment_report(
            report_destination,
            bundle_path=bundle_destination,
            summary_path=summary_destination,
            repository_root=repository_root,
            verify_live_authorization=verify_live_authorization,
        )
        if canonical_json_bytes(reloaded) != report_raw:
            raise RuntimeError("published report strict reload drifted")
        # The public loader authenticates bytes and canonical semantics.  The
        # publisher additionally owns exact inode identities and must prove
        # that none of the three names was replaced during that reload.
        for destination, raw in raw_rows:
            expected = published_identities.get(destination)
            observed_raw, _ = _read_regular_at(
                parent_descriptor,
                destination.name,
                label="published layer-multimode output",
                maximum_bytes={
                    bundle_destination: _MAX_BUNDLE_BYTES,
                    summary_destination: _MAX_SUMMARY_BYTES,
                    report_destination: _MAX_REPORT_BYTES,
                }[destination],
                expected_identity=expected,
            )
            if observed_raw != raw:
                raise RuntimeError("published output changed during strict reload")
    except BaseException as error:
        primary = error
    cleanup_errors = cleanup_temporaries()
    if primary is not None or cleanup_errors:
        if primary is None:
            primary = cleanup_errors.pop(0)
        cleanup_errors.extend(rollback_targets())
    try:
        _verify_directory_chain(
            report_destination.parent,
            directory_identities,
        )
    except BaseException as error:
        if primary is None:
            primary = error
            cleanup_errors.extend(rollback_targets())
        else:
            cleanup_errors.append(error)
    close_errors = _close_descriptors(
        directory_descriptors,
        primary=primary,
        label="publication directory descriptor cleanup",
    )
    if primary is None and close_errors:
        primary = close_errors[0]
        cleanup_errors.extend(close_errors[1:])
        cleanup_errors.extend(late_rollback_targets())
    else:
        cleanup_errors.extend(close_errors)
    if primary is not None:
        if cleanup_errors:
            for cleanup_error in cleanup_errors:
                primary.add_note(
                    "publication cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise primary from BaseExceptionGroup(
                "layer-multimode publication rollback/temporary failures",
                cleanup_errors,
            )
        raise primary
    authority = _PublicationAuthority(
        publication=publication,
        identities=tuple(published_identities.items()),
        directory_identity_chain=directory_identities,
    )
    try:
        with _PUBLICATION_LOCK:
            if id(publication) in _PUBLICATION_AUTHORITIES:
                raise RuntimeError("publication authority already exists")
            _PUBLICATION_AUTHORITIES[id(publication)] = authority
    except BaseException as error:
        cleanup_errors = late_rollback_targets()
        if cleanup_errors:
            for cleanup_error in cleanup_errors:
                error.add_note(
                    "publication cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise error from BaseExceptionGroup(
                "late publication authority rollback failures",
                cleanup_errors,
            )
        raise
    return publication


def rollback_published_layer_multimode_experiment(
    publication: PublishedLayerMultimodeExperimentReceipt,
) -> None:
    if type(publication) is not PublishedLayerMultimodeExperimentReceipt:
        raise TypeError("publication must be the exact typed receipt")
    publication.validate_receipt()
    with _PUBLICATION_LOCK:
        authority = _PUBLICATION_AUTHORITIES.get(id(publication))
        if authority is None or authority.publication is not publication:
            raise PermissionError("publication receipt has no live rollback authority")
        parent = Path(publication.report_path).parent
        _regular_parent(parent)
        descriptors, identities = _open_directory_chain(parent)
        parent_descriptor = descriptors[-1]
        primary: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        report = Path(publication.report_path)
        sidecars = (Path(publication.summary_path), Path(publication.bundle_path))

        def unlink_verified(
            path: Path,
            verified: Mapping[Path, tuple[int, int, int, int, int]],
        ) -> bool:
            expected = verified.get(path)
            if expected is None:
                return False
            guard = -1
            unlink_error: BaseException | None = None
            try:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                guard = os.open(path.name, flags, dir_fd=parent_descriptor)
                guarded = os.fstat(guard)
                named = _stat_at(parent_descriptor, path.name)
                if (
                    _file_identity(guarded) != expected
                    or _file_identity(named) != expected
                    or guarded.st_nlink != 1
                ):
                    raise RuntimeError("publication target changed before rollback")
                os.unlink(path.name, dir_fd=parent_descriptor)
                after = os.fstat(guard)
                if _inode_identity(after) != _inode_identity(guarded):
                    raise RuntimeError("publication rollback guard inode changed")
                if after.st_nlink != 0:
                    raise RuntimeError("publication rollback unlink did not close")
                return True
            except FileNotFoundError:
                return False
            except BaseException as error:
                unlink_error = error
                raise
            finally:
                if guard >= 0:
                    try:
                        os.close(guard)
                    except BaseException as close_error:
                        if unlink_error is None:
                            raise
                        unlink_error.add_note(
                            "rollback target guard close also failed: "
                            f"{type(close_error).__name__}: {close_error}"
                        )

        verified: dict[Path, tuple[int, int, int, int, int]] | None = None
        try:
            if identities != authority.directory_identity_chain:
                raise RuntimeError("published output directory identity drifted")
            verified = _authenticate_targets_at(
                publication,
                authority,
                parent_descriptor=parent_descriptor,
                allow_missing=True,
            )
        except BaseException as error:
            primary = error
        if verified is not None:
            operation_errors: list[BaseException] = []
            try:
                unlink_verified(report, verified)
            except BaseException as error:
                operation_errors.append(error)
            try:
                os.fsync(parent_descriptor)
            except BaseException as error:
                operation_errors.append(error)
            for sidecar in sidecars:
                try:
                    unlink_verified(sidecar, verified)
                except BaseException as error:
                    operation_errors.append(error)
            try:
                os.fsync(parent_descriptor)
            except BaseException as error:
                operation_errors.append(error)
            try:
                _verify_directory_chain(parent, identities)
            except BaseException as error:
                operation_errors.append(error)
            if operation_errors:
                primary = operation_errors[0]
                cleanup_errors.extend(operation_errors[1:])
        cleanup_errors.extend(
            _close_descriptors(
                descriptors,
                primary=primary,
                label="publication rollback descriptor cleanup",
            )
        )
        if primary is not None:
            if cleanup_errors:
                raise primary from BaseExceptionGroup(
                    "publication rollback cleanup failures",
                    cleanup_errors,
                )
            raise primary
        if cleanup_errors:
            raise BaseExceptionGroup(
                "publication rollback descriptor failures",
                cleanup_errors,
            )
        del _PUBLICATION_AUTHORITIES[id(publication)]


def finalize_published_layer_multimode_experiment(
    publication: PublishedLayerMultimodeExperimentReceipt,
) -> None:
    if type(publication) is not PublishedLayerMultimodeExperimentReceipt:
        raise TypeError("publication must be the exact typed receipt")
    publication.validate_receipt()
    with _PUBLICATION_LOCK:
        authority = _PUBLICATION_AUTHORITIES.get(id(publication))
        if authority is None or authority.publication is not publication:
            raise PermissionError("publication receipt has no live authority")
        _authenticate_targets(publication, authority, allow_missing=False)
        del _PUBLICATION_AUTHORITIES[id(publication)]


__all__ = [
    "DEFAULT_BUNDLE_OUTPUT",
    "DEFAULT_REPORT_OUTPUT",
    "DEFAULT_SUMMARY_OUTPUT",
    "PUBLICATION_FORMAT_VERSION",
    "PUBLICATION_SCHEMA",
    "PublishedLayerMultimodeExperimentReceipt",
    "REPORT_FORMAT_VERSION",
    "REPORT_SCHEMA",
    "TERMINAL_B3_FAILED",
    "TERMINAL_C3_FAILED",
    "TERMINAL_D3_COMPLETED",
    "finalize_published_layer_multimode_experiment",
    "publish_layer_multimode_experiment_outputs",
    "render_layer_multimode_summary_bytes",
    "rollback_published_layer_multimode_experiment",
    "seal_layer_multimode_experiment_report",
    "strict_load_layer_multimode_experiment_report",
    "strict_load_layer_multimode_experiment_report_bytes",
]
