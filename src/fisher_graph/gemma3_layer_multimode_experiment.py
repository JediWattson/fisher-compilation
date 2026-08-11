"""Fail-closed orchestration for the Gemma-3 layer-multimode experiment.

The numerical implementation lives in the sibling capture, materialization,
fit, selection, compiler, and evaluation modules.  This module owns only the
experiment state machine:

* strict static preflight happens before a runtime is constructed;
* the already exposed predecessor A corpus is re-executed exactly once into
  new temporary spools because the predecessor run deleted its raw spools;
* every one of the 18 layers materializes once and evaluates the complete
  18-candidate ``C x M x R`` grid;
* one chart atlas is cached for each ``(layer, C, R)`` and reused across M;
* the selected source-free candidate is frozen and development state is torn
  down before any fresh B3 example is opened;
* fresh roles advance exactly B3 -> C3 -> D3, while reserve is never opened;
* all terminal outcomes are reportable, including no-candidate and failed
  fidelity outcomes; and
* publication is atomic from the worker's point of view and is rolled back if
  finalization or cleanup fails.

This source has no tensor-library or model-runtime import.  The package's
legacy eager ``__init__`` may import torch before submodule dispatch, so the
strong preflight claim is narrower and observable: no model/tensor is created,
no model loader or model-facing runtime is imported, and no panel role is
semantically opened.  The qualified runtime lives in a sibling module and its
model-facing methods are reached only after strict preflight.
The predecessor chart-native capture runtime is deliberately incompatible:
its A/B/C role ledger may not be relabelled as fresh B3/C3/D3 evidence.

The production runtime implementation is complete, but execution still fails
closed unless the frozen setup, panel receipt, and exact live implementation
qualification all authenticate.  The test seam exercises orchestration only
and can never emit a production claim or write canonical experiment outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Callable, Final, Mapping, Sequence
import weakref


if __name__ == "__main__":
    sys.modules.setdefault(
        "fisher_graph.gemma3_layer_multimode_experiment",
        sys.modules[__name__],
    )


__all__ = [
    "ADevelopmentRecapture",
    "B3FidelityFailureOutcome",
    "C3FidelityFailureOutcome",
    "D3CompletedOutcome",
    "LayerMaterialization",
    "LayerMultimodeDevelopmentTeardownReceipt",
    "LayerMultimodeExecutionPreflightReceipt",
    "LayerMultimodeExperimentCountersReceipt",
    "LayerMultimodeExperimentRunResult",
    "LayerMultimodeFrozenCandidate",
    "LayerMultimodeFrozenPublicationEvidence",
    "LayerMultimodeFreshRoleCaptureReceipt",
    "LayerMultimodeNoAdmissibleSelectionReceipt",
    "LayerMultimodeFreshPanelExecutionReceipt",
    "LayerMultimodePostTerminalCleanupReceipt",
    "LayerMultimodePreoptimizerAttempt",
    "LayerMultimodePredecessorARuntimeReceipt",
    "LayerMultimodeResourcePreflightReceipt",
    "LayerMultimodeRuntimeExecutionReceipt",
    "LayerMultimodeV3Runtime",
    "NoAdmissibleCandidateOutcome",
    "PRODUCTION_RUNTIME_IMPLEMENTATION_READY",
    "StaticLayerMultimodeExperimentAuthority",
    "make_layer_multimode_test_preflight",
    "layer_multimode_runtime_session_binding_sha256",
    "preflight_layer_multimode_resources",
    "revalidate_layer_multimode_resources",
    "run_layer_multimode_experiment",
    "run_production_layer_multimode_experiment",
    "static_preflight_layer_multimode_experiment",
]


LAYER_COUNT: Final = 18
CHART_COUNT_CANDIDATES: Final = (2, 1)
MODE_COUNT_CANDIDATES: Final = (4, 2, 1)
TANGENT_RANK_CANDIDATES: Final = (64, 32, 16)
CANDIDATES_PER_LAYER: Final = 18
TOTAL_CANDIDATE_COUNT: Final = 324
ATLAS_FIT_COUNT: Final = 108
SPLIT_REFIT_COUNT: Final = 216

A_ROLE: Final = "A_exposed_development"
B3_ROLE: Final = "B3_teacher_forced_layerwise"
C3_ROLE: Final = "C3_full_stack_source_free"
D3_ROLE: Final = "D3_final_heldout_validation"
RESERVE_ROLE: Final = "reserve_v3_unopened"
FRESH_STAGE_KIND_BY_ROLE: Final = {
    B3_ROLE: "B3_teacher_forced_evaluation",
    C3_ROLE: "C3_full_stack_source_free_evaluation",
    D3_ROLE: "D3_final_heldout_evaluation",
}

TERMINAL_A_NO_CANDIDATE: Final = "A_NO_ADMISSIBLE_CANDIDATE"
TERMINAL_B3_FAILED: Final = "B3_FAILED"
TERMINAL_C3_FAILED: Final = "C3_FAILED"
TERMINAL_D3_COMPLETED: Final = "D3_COMPLETED"

PREFLIGHT_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.execution_preflight.v1"
)
NO_SELECTION_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.no_admissible_selection.v2"
)
COUNTERS_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.experiment_counters.v2"
)
TEARDOWN_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.development_teardown.v1"
)
RESOURCE_PREFLIGHT_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.resource_preflight.v1"
)
FRESH_PANEL_EXECUTION_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.fresh_panel_execution.v1"
)
FRESH_ROLE_CAPTURE_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.fresh_role_capture.v1"
)
PREDECESSOR_A_RUNTIME_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.predecessor_A_runtime_execution.v1"
)
PREDECESSOR_A_COMPARISON_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.predecessor_A_recapture_comparison.v1"
)
RUNTIME_EXECUTION_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.runtime_execution.v1"
)
POST_TERMINAL_CLEANUP_SCHEMA: Final = (
    "fisher_graph.gemma3_layer_multimode.post_terminal_cleanup.v1"
)

PRODUCTION_RUNTIME_IMPLEMENTATION_READY: Final = True
"""The qualified model-facing A/B3/C3/D3 implementation is complete."""

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PREFLIGHT_DOMAIN = b"fisher_graph:layer-multimode:preflight:v1\0"
_NO_SELECTION_DOMAIN = b"fisher_graph:layer-multimode:no-selection:v2\0"
_COUNTERS_DOMAIN = b"fisher_graph:layer-multimode:counters:v2\0"
_TEARDOWN_DOMAIN = b"fisher_graph:layer-multimode:teardown:v1\0"
_RESOURCE_PREFLIGHT_DOMAIN = (
    b"fisher_graph:layer-multimode:resource-preflight:v1\0"
)
_FRESH_PANEL_EXECUTION_DOMAIN = (
    b"fisher_graph:layer-multimode:fresh-panel-execution:v1\0"
)
_FRESH_ROLE_CAPTURE_DOMAIN = (
    b"fisher_graph:layer-multimode:fresh-role-capture:v1\0"
)
_PREDECESSOR_A_RUNTIME_DOMAIN = (
    b"fisher_graph:layer-multimode:predecessor-A-runtime:v1\0"
)
_PREDECESSOR_A_COMPARISON_DOMAIN = (
    b"fisher_graph:layer-multimode:predecessor-A-comparison:v1\0"
)
_RUNTIME_EXECUTION_DOMAIN = (
    b"fisher_graph:layer-multimode:runtime-execution:v1\0"
)
_RUNTIME_SESSION_DOMAIN = (
    b"fisher_graph:layer-multimode:runtime-session:v1\0"
)
_POST_TERMINAL_CLEANUP_DOMAIN = (
    b"fisher_graph:layer-multimode:post-terminal-cleanup:v1\0"
)
_TEST_AUTHORITY = object()
_PRODUCTION_AUTHORITY = object()
_PRODUCTION_RUN_ENTRY = object()
_LIVE_PREFLIGHTS: weakref.WeakValueDictionary[
    int, LayerMultimodeExecutionPreflightReceipt
] = weakref.WeakValueDictionary()

_GIB: Final = 1024**3
_PREDICTED_ROWS_PER_ROLE: Final = 24_732
_A_RECAPTURE_PAYLOAD_ESTIMATE_BYTES: Final = (
    2 * LAYER_COUNT * _PREDICTED_ROWS_PER_ROLE * 640 * 4
)
_A_MATERIALIZATION_PAYLOAD_ESTIMATE_BYTES: Final = (
    2 * LAYER_COUNT * _PREDICTED_ROWS_PER_ROLE * 640 * 4
)
# This deliberately exceeds the retained 324-candidate coefficient payload.
# It covers sequential optimizer workspaces, split refits, and allocator
# fragmentation without pretending that coefficient bytes are the peak.
_FULL_GRID_FIT_PEAK_ESTIMATE_BYTES: Final = 8 * _GIB
_MODEL_AND_ATOMIC_STAGING_ESTIMATE_BYTES: Final = 4 * _GIB
_MINIMUM_FREE_DISK_BYTES: Final = 24 * _GIB
_MINIMUM_PHYSICAL_MEMORY_BYTES: Final = 16 * _GIB


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _read_regular_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one immutable regular file without following its final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    primary: BaseException | None = None
    raw: bytes | None = None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} is not a bounded single-link file")
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
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError(f"{label} changed while reading")
        raw = b"".join(chunks)
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
    if raw is None:
        raise AssertionError(f"{label} read produced no bytes")
    return raw


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _artifact_sha256(value: object, *, label: str) -> str:
    for name in (
        "artifact_sha256",
        "receipt_sha256",
        "inventory_sha256",
        "coefficient_sha256",
    ):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and _SHA256.fullmatch(candidate):
            return candidate
    raise TypeError(f"{label} does not expose a typed artifact digest")


def _state_dict(value: object, *, label: str) -> dict[str, object]:
    method = getattr(value, "state_dict", None)
    if not callable(method):
        method = getattr(value, "metadata", None)
    if not callable(method):
        raise TypeError(f"{label} does not expose state_dict/metadata")
    state = method()
    if type(state) is not dict:
        raise TypeError(f"{label} state must be an exact dict")
    # Prove that it is JSON-safe and finite before it crosses the report ABI.
    parsed = json.loads(_canonical_json(state))
    if type(parsed) is not dict:
        raise TypeError(f"{label} state must remain an object after replay")
    return parsed


def layer_multimode_runtime_session_binding_sha256(
    *,
    runtime_implementation_sha256: str,
    implementation_qualification_sha256: str,
    execution_environment_sha256: str,
    tokenizer_fingerprint_sha256: str,
    runtime_qualification_receipt_sha256: str,
    source_model_fingerprint_before_sha256: str,
    source_model_fingerprint_after_sha256: str,
    source_execution_fingerprint_before_sha256: str,
    source_execution_fingerprint_after_sha256: str,
    compiled_model_fingerprint_before_sha256: str,
    compiled_model_fingerprint_after_sha256: str,
    compiled_execution_fingerprint_before_sha256: str,
    compiled_execution_fingerprint_after_sha256: str,
    compiled_candidate_mutated: bool,
) -> str:
    values = {
        "runtime_implementation_sha256": runtime_implementation_sha256,
        "implementation_qualification_sha256": (
            implementation_qualification_sha256
        ),
        "execution_environment_sha256": execution_environment_sha256,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        "runtime_qualification_receipt_sha256": (
            runtime_qualification_receipt_sha256
        ),
        "source_model_fingerprint_before_sha256": (
            source_model_fingerprint_before_sha256
        ),
        "source_model_fingerprint_after_sha256": (
            source_model_fingerprint_after_sha256
        ),
        "source_execution_fingerprint_before_sha256": (
            source_execution_fingerprint_before_sha256
        ),
        "source_execution_fingerprint_after_sha256": (
            source_execution_fingerprint_after_sha256
        ),
        "compiled_model_fingerprint_before_sha256": (
            compiled_model_fingerprint_before_sha256
        ),
        "compiled_model_fingerprint_after_sha256": (
            compiled_model_fingerprint_after_sha256
        ),
        "compiled_execution_fingerprint_before_sha256": (
            compiled_execution_fingerprint_before_sha256
        ),
        "compiled_execution_fingerprint_after_sha256": (
            compiled_execution_fingerprint_after_sha256
        ),
    }
    for name, value in values.items():
        _require_sha256(value, label=name)
    if _require_exact_bool(
        compiled_candidate_mutated,
        label="compiled_candidate_mutated",
    ) is not False:
        raise ValueError("compiled candidate must remain immutable")
    values["compiled_candidate_mutated"] = compiled_candidate_mutated
    return _digest(_RUNTIME_SESSION_DOMAIN, values)


def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if (
        type(pages) is int
        and type(page_size) is int
        and pages > 0
        and page_size > 0
    ):
        return pages * page_size
    return None


@dataclass(frozen=True, slots=True)
class LayerMultimodeResourcePreflightReceipt:
    """Model-free, path-bound capacity receipt issued before runtime import."""

    workspace_parent_path: str
    workspace_path: str
    report_path: str
    bundle_path: str
    summary_path: str
    workspace_parent_device: int
    workspace_parent_inode: int
    output_parent_identities: tuple[tuple[int, int], ...]
    observed_free_bytes_by_device: tuple[tuple[int, int], ...]
    physical_memory_bytes: int
    A_recapture_payload_estimate_bytes: int
    A_materialization_payload_estimate_bytes: int
    full_grid_fit_peak_estimate_bytes: int
    model_and_atomic_staging_estimate_bytes: int
    required_free_disk_bytes: int
    required_physical_memory_bytes: int
    workspace_path_safe: bool
    output_paths_safe: bool
    test_only_nonproduction: bool
    receipt_sha256: str = ""
    schema: str = RESOURCE_PREFLIGHT_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.schema != RESOURCE_PREFLIGHT_SCHEMA or self.format_version != 1:
            raise ValueError("resource preflight header is invalid")
        for name in (
            "workspace_parent_path",
            "workspace_path",
            "report_path",
            "bundle_path",
            "summary_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute canonical path")
        if Path(self.workspace_path).parent != Path(self.workspace_parent_path):
            raise ValueError("workspace path is not under its frozen parent")
        if len({self.report_path, self.bundle_path, self.summary_path}) != 3:
            raise ValueError("frozen output paths must be distinct")
        for name in (
            "workspace_parent_device",
            "workspace_parent_inode",
            "physical_memory_bytes",
            "A_recapture_payload_estimate_bytes",
            "A_materialization_payload_estimate_bytes",
            "full_grid_fit_peak_estimate_bytes",
            "model_and_atomic_staging_estimate_bytes",
            "required_free_disk_bytes",
            "required_physical_memory_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            type(self.output_parent_identities) is not tuple
            or len(self.output_parent_identities) != 3
            or any(
                type(value) is not tuple
                or len(value) != 2
                or any(type(item) is not int or item <= 0 for item in value)
                for value in self.output_parent_identities
            )
        ):
            raise ValueError("output parent identity inventory is invalid")
        if (
            type(self.observed_free_bytes_by_device) is not tuple
            or not self.observed_free_bytes_by_device
            or tuple(value[0] for value in self.observed_free_bytes_by_device)
            != tuple(
                sorted(
                    {value[0] for value in self.observed_free_bytes_by_device}
                )
            )
        ):
            raise ValueError("free-byte device inventory is invalid")
        for row in self.observed_free_bytes_by_device:
            if (
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not int
                or type(row[1]) is not int
                or row[0] <= 0
                or row[1] < self.required_free_disk_bytes
            ):
                raise ValueError("resource preflight has insufficient free disk")
        if self.physical_memory_bytes < self.required_physical_memory_bytes:
            raise ValueError("resource preflight has insufficient physical memory")
        if self.required_free_disk_bytes < _MINIMUM_FREE_DISK_BYTES:
            raise ValueError("resource preflight disk floor drifted below 24 GiB")
        if self.required_physical_memory_bytes < _MINIMUM_PHYSICAL_MEMORY_BYTES:
            raise ValueError("resource preflight memory floor drifted below 16 GiB")
        if self.A_recapture_payload_estimate_bytes != (
            _A_RECAPTURE_PAYLOAD_ESTIMATE_BYTES
        ):
            raise ValueError("A recapture byte estimate drifted")
        if self.A_materialization_payload_estimate_bytes != (
            _A_MATERIALIZATION_PAYLOAD_ESTIMATE_BYTES
        ):
            raise ValueError("A materialization byte estimate drifted")
        if self.full_grid_fit_peak_estimate_bytes != (
            _FULL_GRID_FIT_PEAK_ESTIMATE_BYTES
        ):
            raise ValueError("full-grid fit byte estimate drifted")
        if self.model_and_atomic_staging_estimate_bytes != (
            _MODEL_AND_ATOMIC_STAGING_ESTIMATE_BYTES
        ):
            raise ValueError("model/atomic staging byte estimate drifted")
        for name in (
            "workspace_path_safe",
            "output_paths_safe",
        ):
            if _require_exact_bool(getattr(self, name), label=name) is not True:
                raise ValueError(f"{name} must be true")
        _require_exact_bool(
            self.test_only_nonproduction,
            label="test_only_nonproduction",
        )
        expected = _digest(
            _RESOURCE_PREFLIGHT_DOMAIN,
            self.payload_without_sha256(),
        )
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif self.receipt_sha256 != expected:
            raise ValueError("resource preflight digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "workspace_parent_path": self.workspace_parent_path,
            "workspace_path": self.workspace_path,
            "report_path": self.report_path,
            "bundle_path": self.bundle_path,
            "summary_path": self.summary_path,
            "workspace_parent_device": self.workspace_parent_device,
            "workspace_parent_inode": self.workspace_parent_inode,
            "output_parent_identities": [
                list(value) for value in self.output_parent_identities
            ],
            "observed_free_bytes_by_device": [
                list(value) for value in self.observed_free_bytes_by_device
            ],
            "physical_memory_bytes": self.physical_memory_bytes,
            "A_recapture_payload_estimate_bytes": (
                self.A_recapture_payload_estimate_bytes
            ),
            "A_materialization_payload_estimate_bytes": (
                self.A_materialization_payload_estimate_bytes
            ),
            "full_grid_fit_peak_estimate_bytes": (
                self.full_grid_fit_peak_estimate_bytes
            ),
            "model_and_atomic_staging_estimate_bytes": (
                self.model_and_atomic_staging_estimate_bytes
            ),
            "required_free_disk_bytes": self.required_free_disk_bytes,
            "required_physical_memory_bytes": (
                self.required_physical_memory_bytes
            ),
            "workspace_path_safe": self.workspace_path_safe,
            "output_paths_safe": self.output_paths_safe,
            "test_only_nonproduction": self.test_only_nonproduction,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


def _test_resource_preflight() -> LayerMultimodeResourcePreflightReceipt:
    return LayerMultimodeResourcePreflightReceipt(
        workspace_parent_path="/TEST_ONLY/workspace-parent",
        workspace_path=(
            "/TEST_ONLY/workspace-parent/layer-multimode-experiment-v2.work"
        ),
        report_path="/TEST_ONLY/report.json",
        bundle_path="/TEST_ONLY/candidate.bundle",
        summary_path="/TEST_ONLY/summary.json",
        workspace_parent_device=1,
        workspace_parent_inode=1,
        output_parent_identities=((1, 1), (1, 1), (1, 1)),
        observed_free_bytes_by_device=((1, _MINIMUM_FREE_DISK_BYTES),),
        physical_memory_bytes=_MINIMUM_PHYSICAL_MEMORY_BYTES,
        A_recapture_payload_estimate_bytes=_A_RECAPTURE_PAYLOAD_ESTIMATE_BYTES,
        A_materialization_payload_estimate_bytes=(
            _A_MATERIALIZATION_PAYLOAD_ESTIMATE_BYTES
        ),
        full_grid_fit_peak_estimate_bytes=_FULL_GRID_FIT_PEAK_ESTIMATE_BYTES,
        model_and_atomic_staging_estimate_bytes=(
            _MODEL_AND_ATOMIC_STAGING_ESTIMATE_BYTES
        ),
        required_free_disk_bytes=_MINIMUM_FREE_DISK_BYTES,
        required_physical_memory_bytes=_MINIMUM_PHYSICAL_MEMORY_BYTES,
        workspace_path_safe=True,
        output_paths_safe=True,
        test_only_nonproduction=True,
    )


def preflight_layer_multimode_resources(
    *,
    workspace_parent: Path | str,
    report_path: Path | str,
    bundle_path: Path | str,
    summary_path: Path | str,
) -> LayerMultimodeResourcePreflightReceipt:
    """Freeze capacity/path evidence before panels, tensors, or model access."""

    parent_input = Path(workspace_parent)
    if not parent_input.is_dir() or parent_input.is_symlink():
        raise ValueError("workspace_parent must be a non-symlink directory")
    parent = parent_input.resolve(strict=True)
    workspace = parent / "layer-multimode-experiment-v2.work"
    if os.path.lexists(workspace):
        raise FileExistsError(workspace)
    destinations = tuple(
        Path(value).resolve(strict=False)
        for value in (report_path, bundle_path, summary_path)
    )
    if len(set(destinations)) != 3:
        raise ValueError("report, bundle, and summary paths must differ")
    output_stats: list[os.stat_result] = []
    for destination in destinations:
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        output_parent = destination.parent
        if not output_parent.is_dir() or output_parent.is_symlink():
            raise ValueError("output parents must be non-symlink directories")
        output_stats.append(output_parent.stat())
    parent_stat = parent.stat()
    free_by_device: dict[int, int] = {
        parent_stat.st_dev: shutil.disk_usage(parent).free,
    }
    for destination, info in zip(destinations, output_stats, strict=True):
        free_by_device[info.st_dev] = min(
            free_by_device.get(info.st_dev, 2**63 - 1),
            shutil.disk_usage(destination.parent).free,
        )
    if min(free_by_device.values()) < _MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("static preflight has insufficient free disk")
    physical = _physical_memory_bytes()
    if physical is None or physical < _MINIMUM_PHYSICAL_MEMORY_BYTES:
        raise RuntimeError("static preflight has insufficient physical memory")
    return LayerMultimodeResourcePreflightReceipt(
        workspace_parent_path=str(parent),
        workspace_path=str(workspace),
        report_path=str(destinations[0]),
        bundle_path=str(destinations[1]),
        summary_path=str(destinations[2]),
        workspace_parent_device=parent_stat.st_dev,
        workspace_parent_inode=parent_stat.st_ino,
        output_parent_identities=tuple(
            (info.st_dev, info.st_ino) for info in output_stats
        ),
        observed_free_bytes_by_device=tuple(sorted(free_by_device.items())),
        physical_memory_bytes=physical,
        A_recapture_payload_estimate_bytes=_A_RECAPTURE_PAYLOAD_ESTIMATE_BYTES,
        A_materialization_payload_estimate_bytes=(
            _A_MATERIALIZATION_PAYLOAD_ESTIMATE_BYTES
        ),
        full_grid_fit_peak_estimate_bytes=_FULL_GRID_FIT_PEAK_ESTIMATE_BYTES,
        model_and_atomic_staging_estimate_bytes=(
            _MODEL_AND_ATOMIC_STAGING_ESTIMATE_BYTES
        ),
        required_free_disk_bytes=_MINIMUM_FREE_DISK_BYTES,
        required_physical_memory_bytes=_MINIMUM_PHYSICAL_MEMORY_BYTES,
        workspace_path_safe=True,
        output_paths_safe=True,
        test_only_nonproduction=False,
    )


def revalidate_layer_multimode_resources(
    receipt: LayerMultimodeResourcePreflightReceipt,
    *,
    workspace_parent: Path | str,
    report_path: Path | str,
    bundle_path: Path | str,
    summary_path: Path | str,
    workspace_must_exist: bool,
) -> None:
    """Re-stat the exact frozen paths and capacity without issuing authority.

    The initial resource receipt is an immutable observation, so later checks
    do not mutate or replace it.  They require the caller's resolved paths,
    parent identities, device inventory, and current capacity to remain within
    the already-authorized envelope.
    """

    if type(receipt) is not LayerMultimodeResourcePreflightReceipt:
        raise TypeError("resource receipt must have the exact preflight type")
    parent_input = Path(workspace_parent)
    if parent_input.is_symlink():
        raise RuntimeError("workspace parent became a symlink")
    parent = parent_input.resolve(strict=True)
    destinations = tuple(
        Path(value).resolve(strict=False)
        for value in (report_path, bundle_path, summary_path)
    )
    expected_paths = (
        receipt.workspace_parent_path,
        receipt.report_path,
        receipt.bundle_path,
        receipt.summary_path,
    )
    observed_paths = (str(parent), *(str(value) for value in destinations))
    if observed_paths != expected_paths:
        raise RuntimeError("runtime paths differ from resource preflight")
    workspace = parent / "layer-multimode-experiment-v2.work"
    if str(workspace) != receipt.workspace_path:
        raise RuntimeError("workspace name differs from resource preflight")
    if workspace_must_exist:
        if not workspace.is_dir() or workspace.is_symlink():
            raise RuntimeError("authorized workspace is missing or unsafe")
    elif os.path.lexists(workspace):
        raise FileExistsError(workspace)

    parent_stat = parent.stat()
    if (
        parent_stat.st_dev != receipt.workspace_parent_device
        or parent_stat.st_ino != receipt.workspace_parent_inode
    ):
        raise RuntimeError("workspace parent identity changed")
    output_stats: list[os.stat_result] = []
    for destination in destinations:
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        output_parent = destination.parent
        if not output_parent.is_dir() or output_parent.is_symlink():
            raise RuntimeError("output parent is missing or unsafe")
        output_stats.append(output_parent.stat())
    identities = tuple((value.st_dev, value.st_ino) for value in output_stats)
    if identities != receipt.output_parent_identities:
        raise RuntimeError("output parent identity changed")

    free_by_device: dict[int, int] = {
        parent_stat.st_dev: shutil.disk_usage(parent).free,
    }
    for destination, info in zip(destinations, output_stats, strict=True):
        free_by_device[info.st_dev] = min(
            free_by_device.get(info.st_dev, 2**63 - 1),
            shutil.disk_usage(destination.parent).free,
        )
    if tuple(sorted(free_by_device)) != tuple(
        value[0] for value in receipt.observed_free_bytes_by_device
    ):
        raise RuntimeError("resource device inventory changed")
    if min(free_by_device.values()) < receipt.required_free_disk_bytes:
        raise RuntimeError("runtime resource revalidation has insufficient disk")
    physical = _physical_memory_bytes()
    if physical is None or physical < receipt.required_physical_memory_bytes:
        raise RuntimeError("runtime resource revalidation has insufficient memory")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class LayerMultimodeExecutionPreflightReceipt:
    """Static authority that must exist before a runtime or panel source."""

    setup_sha256: str
    setup_file_sha256: str
    panel_manifest_sha256: str
    panel_manifest_file_sha256: str
    panel_receipt_sha256: str
    panel_receipt_file_sha256: str
    panel_bundle_sha256: str
    implementation_qualification_sha256: str
    implementation_qualification_file_sha256: str
    predecessor_v2_setup_sha256: str
    predecessor_v2_report_sha256: str
    predecessor_A_capture_receipt_sha256: str
    predecessor_runtime_execution_receipt_sha256: str
    source_model_sha256: str
    evaluation_protocol_sha256: str
    resource_preflight_receipt: LayerMultimodeResourcePreflightReceipt
    test_only_nonproduction: bool
    receipt_sha256: str = ""
    schema: str = PREFLIGHT_SCHEMA
    format_version: int = 1
    exposed_predecessor_A_recapture_only: bool = True
    fresh_role_order: tuple[str, str, str, str] = (
        B3_ROLE,
        C3_ROLE,
        D3_ROLE,
        RESERVE_ROLE,
    )
    full_grid_candidate_count: int = TOTAL_CANDIDATE_COUNT
    production_runtime_ready: bool = PRODUCTION_RUNTIME_IMPLEMENTATION_READY
    _construction_authority: object = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self.schema != PREFLIGHT_SCHEMA
            or type(self.format_version) is not int
            or self.format_version != 1
        ):
            raise ValueError("execution preflight header is invalid")
        for name in (
            "setup_sha256",
            "setup_file_sha256",
            "panel_manifest_sha256",
            "panel_manifest_file_sha256",
            "panel_receipt_sha256",
            "panel_receipt_file_sha256",
            "panel_bundle_sha256",
            "implementation_qualification_sha256",
            "implementation_qualification_file_sha256",
            "predecessor_v2_setup_sha256",
            "predecessor_v2_report_sha256",
            "predecessor_A_capture_receipt_sha256",
            "predecessor_runtime_execution_receipt_sha256",
            "source_model_sha256",
            "evaluation_protocol_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_exact_bool(
            self.test_only_nonproduction,
            label="test_only_nonproduction",
        )
        if type(self.resource_preflight_receipt) is not (
            LayerMultimodeResourcePreflightReceipt
        ):
            raise TypeError("resource preflight receipt type is invalid")
        if (
            self.resource_preflight_receipt.test_only_nonproduction
            is not self.test_only_nonproduction
        ):
            raise ValueError("resource/preflight production markers differ")
        if self.exposed_predecessor_A_recapture_only is not True:
            raise ValueError(
                "v3 may recapture only the predecessor exposed A role"
            )
        if self.fresh_role_order != (B3_ROLE, C3_ROLE, D3_ROLE, RESERVE_ROLE):
            raise ValueError("fresh-role order drifted")
        if (
            type(self.full_grid_candidate_count) is not int
            or self.full_grid_candidate_count != TOTAL_CANDIDATE_COUNT
        ):
            raise ValueError("full-grid candidate count drifted")
        _require_exact_bool(
            self.production_runtime_ready,
            label="production_runtime_ready",
        )
        if self.test_only_nonproduction:
            if self.production_runtime_ready is not False:
                raise ValueError("test preflight cannot authorize production")
        elif self.production_runtime_ready is not True:
            raise ValueError("production preflight requires the production runtime")
        expected = _digest(_PREFLIGHT_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="preflight receipt_sha256",
        ) != expected:
            raise ValueError("preflight receipt hash mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "setup_sha256": self.setup_sha256,
            "setup_file_sha256": self.setup_file_sha256,
            "panel_manifest_sha256": self.panel_manifest_sha256,
            "panel_manifest_file_sha256": self.panel_manifest_file_sha256,
            "panel_receipt_sha256": self.panel_receipt_sha256,
            "panel_receipt_file_sha256": self.panel_receipt_file_sha256,
            "panel_bundle_sha256": self.panel_bundle_sha256,
            "implementation_qualification_sha256": (
                self.implementation_qualification_sha256
            ),
            "implementation_qualification_file_sha256": (
                self.implementation_qualification_file_sha256
            ),
            "predecessor_v2_setup_sha256": self.predecessor_v2_setup_sha256,
            "predecessor_v2_report_sha256": self.predecessor_v2_report_sha256,
            "predecessor_A_capture_receipt_sha256": (
                self.predecessor_A_capture_receipt_sha256
            ),
            "predecessor_runtime_execution_receipt_sha256": (
                self.predecessor_runtime_execution_receipt_sha256
            ),
            "source_model_sha256": self.source_model_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "resource_preflight_receipt": (
                self.resource_preflight_receipt.state_dict()
            ),
            "test_only_nonproduction": self.test_only_nonproduction,
            "exposed_predecessor_A_recapture_only": (
                self.exposed_predecessor_A_recapture_only
            ),
            "fresh_role_order": list(self.fresh_role_order),
            "full_grid_candidate_count": self.full_grid_candidate_count,
            "production_runtime_ready": self.production_runtime_ready,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }

    @property
    def live_preflight_authorized(self) -> bool:
        return (
            _LIVE_PREFLIGHTS.get(id(self)) is self
            and self._construction_authority
            in (_TEST_AUTHORITY, _PRODUCTION_AUTHORITY)
        )


def make_layer_multimode_test_preflight(
    *, seed_sha256: str | None = None
) -> LayerMultimodeExecutionPreflightReceipt:
    """Issue an unmistakably nonproduction receipt for model-free tests."""

    seed = seed_sha256 or hashlib.sha256(
        b"fisher_graph:layer-multimode:orchestration-test"
    ).hexdigest()
    _require_sha256(seed, label="seed_sha256")
    receipt = LayerMultimodeExecutionPreflightReceipt(
        setup_sha256=seed,
        setup_file_sha256=seed,
        panel_manifest_sha256=seed,
        panel_manifest_file_sha256=seed,
        panel_receipt_sha256=seed,
        panel_receipt_file_sha256=seed,
        panel_bundle_sha256=seed,
        implementation_qualification_sha256=seed,
        implementation_qualification_file_sha256=seed,
        predecessor_v2_setup_sha256=seed,
        predecessor_v2_report_sha256=seed,
        predecessor_A_capture_receipt_sha256=seed,
        predecessor_runtime_execution_receipt_sha256=seed,
        source_model_sha256=seed,
        evaluation_protocol_sha256=seed,
        resource_preflight_receipt=_test_resource_preflight(),
        test_only_nonproduction=True,
        production_runtime_ready=False,
    )
    object.__setattr__(receipt, "_construction_authority", _TEST_AUTHORITY)
    _LIVE_PREFLIGHTS[id(receipt)] = receipt
    return receipt


@dataclass(frozen=True, slots=True)
class _ProductionPreflightObservation:
    receipt: LayerMultimodeExecutionPreflightReceipt
    panel_preflight: object


def _panel_preflight_state(value: object) -> dict[str, object]:
    manifest = getattr(value, "manifest", None)
    role_items = getattr(value, "role_file_sha256_items", None)
    if not isinstance(manifest, Mapping) or type(role_items) is not tuple:
        raise TypeError("opaque panel preflight shape drifted")
    return {
        "manifest": json.loads(_canonical_json(dict(manifest))),
        "manifest_file_sha256": getattr(
            value,
            "manifest_file_sha256",
            None,
        ),
        "receipt_file_sha256": getattr(
            value,
            "receipt_file_sha256",
            None,
        ),
        "receipt_sha256": getattr(value, "receipt_sha256", None),
        "bundle_sha256": getattr(value, "bundle_sha256", None),
        "role_file_sha256_items": [list(row) for row in role_items],
        "reserve_semantically_decoded": getattr(
            value,
            "reserve_semantically_decoded",
            None,
        ),
    }


def _strict_production_preflight_observation(
    *,
    repository_root: Path,
    setup_path: Path,
    workspace_parent: Path,
    report_path: Path,
    bundle_path: Path,
    summary_path: Path,
    resource_receipt: LayerMultimodeResourcePreflightReceipt | None,
    workspace_must_exist: bool,
    grant_live_authority: bool,
) -> _ProductionPreflightObservation:
    """Rebuild every model-free production authority from canonical bytes."""

    from . import gemma3_layer_multimode_setup as setup

    root_input = Path(repository_root)
    if root_input.is_symlink():
        raise ValueError("repository root cannot be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository root must be a directory")
    canonical_setup = (root / setup.DEFAULT_OUTPUT).resolve(strict=False)
    source = Path(setup_path)
    if not source.is_absolute():
        source = root / source
    if (
        source.is_symlink()
        or Path(os.path.abspath(source)) != canonical_setup
    ):
        raise ValueError("production setup path is not canonical")
    canonical_workspace_parent = (
        root / setup.DEFAULT_REPORT_OUTPUT.parent
    ).resolve(strict=False)
    canonical_destinations = tuple(
        (root / value).resolve(strict=False)
        for value in (
            setup.DEFAULT_REPORT_OUTPUT,
            setup.DEFAULT_BUNDLE_OUTPUT,
            setup.DEFAULT_SUMMARY_OUTPUT,
        )
    )
    observed_workspace_parent = Path(workspace_parent).resolve(strict=False)
    observed_destinations = tuple(
        Path(value).resolve(strict=False)
        for value in (report_path, bundle_path, summary_path)
    )
    if (
        observed_workspace_parent != canonical_workspace_parent
        or observed_destinations != canonical_destinations
    ):
        raise ValueError("production workspace/output paths are not canonical")

    if resource_receipt is None:
        resources = preflight_layer_multimode_resources(
            workspace_parent=workspace_parent,
            report_path=report_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
        )
    else:
        revalidate_layer_multimode_resources(
            resource_receipt,
            workspace_parent=workspace_parent,
            report_path=report_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
            workspace_must_exist=workspace_must_exist,
        )
        resources = resource_receipt

    setup_state = setup.authorize_run(
        canonical_setup,
        repository_root=root,
        verify_predecessor=True,
    )
    if not PRODUCTION_RUNTIME_IMPLEMENTATION_READY:
        raise setup.ExecutionNotAuthorizedError(
            "layer-multimode v3 runtime implementation is not release-ready"
        )
    setup_raw = _read_regular_bytes(
        canonical_setup,
        maximum_bytes=8 * 1024 * 1024,
        label="canonical layer-multimode setup",
    )
    if setup_raw != setup.render_setup_bytes(setup_state):
        raise ValueError("canonical setup bytes differ from live source")

    from . import gemma3_layer_multimode_fresh_panels as panels
    from . import gemma3_layer_multimode_implementation_qualification as iq

    readiness = setup_state["implementation_readiness"]
    expected_panel_file_sha256 = readiness[  # type: ignore[index]
        "panel_receipt_file_sha256"
    ]
    _require_sha256(
        expected_panel_file_sha256,
        label="setup panel receipt file",
    )
    panel_preflight = panels.load_frozen_preflight(
        repository_root=root,
        expected_receipt_file_sha256=expected_panel_file_sha256,
    )
    qualification_path = root / iq.DEFAULT_OUTPUT
    qualification = iq.load_qualification(
        qualification_path,
        repository_root=root,
    )
    qualification_raw = _read_regular_bytes(
        qualification_path,
        maximum_bytes=32 * 1024 * 1024,
        label="implementation qualification",
    )
    if qualification_raw != iq.canonical_json_bytes(qualification):
        raise RuntimeError("implementation qualification changed across replay")

    from . import gemma3_chart_native_neighbor_v2_report as predecessor
    from . import gemma3_layer_multimode_evaluation as evaluation

    predecessor_path = root / setup.PREDECESSOR_V2_REPORT
    predecessor_raw = _read_regular_bytes(
        predecessor_path,
        maximum_bytes=512 * 1024 * 1024,
        label="predecessor v2 report",
    )
    if hashlib.sha256(predecessor_raw).hexdigest() != (
        setup.PREDECESSOR_V2_REPORT_FILE_SHA256
    ):
        raise ValueError("predecessor v2 report file pin drifted")
    predecessor_report = predecessor.strict_load_v2_experiment_report_bytes(
        predecessor_raw
    )
    A_fit = predecessor_report.get("A_fit")
    runtime_state = predecessor_report.get("runtime_execution_receipt")
    if not isinstance(A_fit, Mapping) or not isinstance(
        runtime_state,
        Mapping,
    ):
        raise ValueError("predecessor A/runtime evidence is absent")
    capture_state = A_fit.get("capture_receipt")
    if not isinstance(capture_state, Mapping):
        raise ValueError("predecessor A capture evidence is absent")
    predecessor_preflight = predecessor_report.get("v2_preflight")
    provenance = setup_state.get("methodology_provenance")
    if not isinstance(predecessor_preflight, Mapping) or not isinstance(
        provenance,
        Mapping,
    ):
        raise ValueError("predecessor v2 setup lineage is absent")

    frozen_setup = qualification.get("frozen_setup")
    frozen_predecessor = qualification.get("frozen_predecessor_v2")
    frozen_panel = qualification.get("frozen_panel_bundle")
    if not all(
        isinstance(value, Mapping)
        for value in (frozen_setup, frozen_predecessor, frozen_panel)
    ):
        raise ValueError("implementation qualification lineage is incomplete")
    setup_file_sha256 = hashlib.sha256(setup_raw).hexdigest()
    qualification_file_sha256 = hashlib.sha256(
        qualification_raw
    ).hexdigest()
    source_model_sha256 = runtime_state.get(
        "source_model_fingerprint_before_sha256"
    )
    evaluation_sha256 = evaluation.heldout_evaluation_policy_sha256()
    if (
        frozen_setup.get("setup_sha256") != setup_state["setup_sha256"]
        or frozen_setup.get("file_sha256") != setup_file_sha256
        or frozen_predecessor.get("setup_sha256")
        != setup.PREDECESSOR_V2_SETUP_SHA256
        or frozen_predecessor.get("report_sha256")
        != predecessor_report["report_sha256"]
        or frozen_predecessor.get("A_capture_receipt_sha256")
        != capture_state.get("receipt_sha256")
        or frozen_predecessor.get("runtime_execution_receipt_sha256")
        != runtime_state.get("runtime_receipt_sha256")
        or predecessor_report.get("report_sha256")
        != setup.PREDECESSOR_V2_REPORT_SHA256
        or predecessor_preflight.get("v2_setup_sha256")
        != setup.PREDECESSOR_V2_SETUP_SHA256
        or capture_state.get("receipt_sha256")
        != setup.PREDECESSOR_V2_A_CAPTURE_RECEIPT_SHA256
        or runtime_state.get("runtime_receipt_sha256")
        != setup.PREDECESSOR_V2_RUNTIME_EXECUTION_RECEIPT_SHA256
        or runtime_state.get("source_model_fingerprint_before_sha256")
        != setup.PREDECESSOR_V2_SOURCE_MODEL_FINGERPRINT_SHA256
        or runtime_state.get("source_model_fingerprint_after_sha256")
        != setup.PREDECESSOR_V2_SOURCE_MODEL_FINGERPRINT_SHA256
        or runtime_state.get("source_execution_fingerprint_before_sha256")
        != setup.PREDECESSOR_V2_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or runtime_state.get("source_execution_fingerprint_after_sha256")
        != setup.PREDECESSOR_V2_SOURCE_EXECUTION_FINGERPRINT_SHA256
        or provenance.get("predecessor_v2_setup_sha256")
        != setup.PREDECESSOR_V2_SETUP_SHA256
        or provenance.get("predecessor_v2_report_sha256")
        != predecessor_report.get("report_sha256")
        or provenance.get("predecessor_v2_A_capture_receipt_sha256")
        != capture_state.get("receipt_sha256")
        or provenance.get(
            "predecessor_v2_runtime_execution_receipt_sha256"
        )
        != runtime_state.get("runtime_receipt_sha256")
        or provenance.get(
            "predecessor_v2_source_model_fingerprint_sha256"
        )
        != runtime_state.get("source_model_fingerprint_before_sha256")
        or provenance.get(
            "predecessor_v2_source_execution_fingerprint_sha256"
        )
        != runtime_state.get("source_execution_fingerprint_before_sha256")
        or frozen_panel.get("receipt_sha256")
        != panel_preflight.receipt_sha256
        or frozen_panel.get("receipt_file_sha256")
        != panel_preflight.receipt_file_sha256
        or frozen_panel.get("manifest_sha256")
        != panel_preflight.manifest["manifest_sha256"]
        or frozen_panel.get("manifest_file_sha256")
        != panel_preflight.manifest_file_sha256
        or frozen_panel.get("bundle_sha256")
        != panel_preflight.bundle_sha256
        or runtime_state.get("source_model_fingerprint_after_sha256")
        != source_model_sha256
        or any(
            setup_state[name]["heldout_evaluation_policy_sha256"]
            != evaluation_sha256
            for name in (
                "B3_teacher_forced_layerwise",
                "C3_full_stack_source_free",
                "D3_final_validation",
            )
        )
    ):
        raise ValueError("production preflight lineage cross-binding drifted")
    _require_sha256(source_model_sha256, label="source model fingerprint")

    receipt = LayerMultimodeExecutionPreflightReceipt(
        setup_sha256=str(setup_state["setup_sha256"]),
        setup_file_sha256=setup_file_sha256,
        panel_manifest_sha256=str(
            panel_preflight.manifest["manifest_sha256"]
        ),
        panel_manifest_file_sha256=(
            panel_preflight.manifest_file_sha256
        ),
        panel_receipt_sha256=panel_preflight.receipt_sha256,
        panel_receipt_file_sha256=panel_preflight.receipt_file_sha256,
        panel_bundle_sha256=panel_preflight.bundle_sha256,
        implementation_qualification_sha256=str(
            qualification["qualification_sha256"]
        ),
        implementation_qualification_file_sha256=(
            qualification_file_sha256
        ),
        predecessor_v2_setup_sha256=setup.PREDECESSOR_V2_SETUP_SHA256,
        predecessor_v2_report_sha256=str(
            predecessor_report["report_sha256"]
        ),
        predecessor_A_capture_receipt_sha256=str(
            capture_state["receipt_sha256"]
        ),
        predecessor_runtime_execution_receipt_sha256=str(
            runtime_state["runtime_receipt_sha256"]
        ),
        source_model_sha256=str(source_model_sha256),
        evaluation_protocol_sha256=evaluation_sha256,
        resource_preflight_receipt=resources,
        test_only_nonproduction=False,
        production_runtime_ready=True,
    )
    if grant_live_authority:
        object.__setattr__(
            receipt,
            "_construction_authority",
            _PRODUCTION_AUTHORITY,
        )
        _LIVE_PREFLIGHTS[id(receipt)] = receipt
    return _ProductionPreflightObservation(
        receipt=receipt,
        panel_preflight=panel_preflight,
    )


def static_preflight_layer_multimode_experiment(
    *,
    repository_root: Path | str,
    setup_path: Path | str,
    workspace_parent: Path | str | None = None,
    report_path: Path | str | None = None,
    bundle_path: Path | str | None = None,
    summary_path: Path | str | None = None,
) -> LayerMultimodeExecutionPreflightReceipt:
    """Issue one strict model-free production receipt from canonical bytes."""

    from . import gemma3_layer_multimode_setup as setup

    root = Path(repository_root).resolve(strict=True)
    output_parent = root / setup.DEFAULT_REPORT_OUTPUT.parent
    observation = _strict_production_preflight_observation(
        repository_root=root,
        setup_path=Path(setup_path),
        workspace_parent=(
            output_parent if workspace_parent is None else Path(workspace_parent)
        ),
        report_path=(
            root / setup.DEFAULT_REPORT_OUTPUT
            if report_path is None
            else Path(report_path)
        ),
        bundle_path=(
            root / setup.DEFAULT_BUNDLE_OUTPUT
            if bundle_path is None
            else Path(bundle_path)
        ),
        summary_path=(
            root / setup.DEFAULT_SUMMARY_OUTPUT
            if summary_path is None
            else Path(summary_path)
        ),
        resource_receipt=None,
        workspace_must_exist=False,
        grant_live_authority=True,
    )
    return observation.receipt


class StaticLayerMultimodeExperimentAuthority:
    """One-shot production authority owning the original live byte receipts."""

    def __init__(
        self,
        *,
        repository_root: Path | str,
        setup_path: Path | str,
        workspace_parent: Path | str | None = None,
        report_path: Path | str | None = None,
        bundle_path: Path | str | None = None,
        summary_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        from . import gemma3_layer_multimode_setup as setup

        self.repository_root = Path(repository_root).resolve(strict=True)
        self.setup_path = Path(setup_path)
        output_parent = self.repository_root / setup.DEFAULT_REPORT_OUTPUT.parent
        self.workspace_parent = (
            output_parent
            if workspace_parent is None
            else Path(workspace_parent)
        )
        self.report_path = (
            self.repository_root / setup.DEFAULT_REPORT_OUTPUT
            if report_path is None
            else Path(report_path)
        )
        self.bundle_path = (
            self.repository_root / setup.DEFAULT_BUNDLE_OUTPUT
            if bundle_path is None
            else Path(bundle_path)
        )
        self.summary_path = (
            self.repository_root / setup.DEFAULT_SUMMARY_OUTPUT
            if summary_path is None
            else Path(summary_path)
        )
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self._receipt: LayerMultimodeExecutionPreflightReceipt | None = None
        self._panel_preflight: object | None = None
        self._runtime_built = False
        self._panel_source_built = False
        self._workspace_seen = False

    def strict_preflight(self) -> LayerMultimodeExecutionPreflightReceipt:
        if self._receipt is not None:
            raise RuntimeError("production preflight may be issued exactly once")
        observation = _strict_production_preflight_observation(
            repository_root=self.repository_root,
            setup_path=self.setup_path,
            workspace_parent=self.workspace_parent,
            report_path=self.report_path,
            bundle_path=self.bundle_path,
            summary_path=self.summary_path,
            resource_receipt=None,
            workspace_must_exist=False,
            grant_live_authority=True,
        )
        self._receipt = observation.receipt
        self._panel_preflight = observation.panel_preflight
        return observation.receipt

    def revalidate(
        self,
        receipt: LayerMultimodeExecutionPreflightReceipt,
    ) -> None:
        if (
            type(receipt) is not LayerMultimodeExecutionPreflightReceipt
            or receipt is not self._receipt
        ):
            raise TypeError("revalidation requires the original preflight receipt")
        if not receipt.live_preflight_authorized:
            raise RuntimeError("original production preflight is no longer live")
        workspace = Path(receipt.resource_preflight_receipt.workspace_path)
        if os.path.lexists(workspace):
            self._workspace_seen = True
        observation = _strict_production_preflight_observation(
            repository_root=self.repository_root,
            setup_path=self.setup_path,
            workspace_parent=self.workspace_parent,
            report_path=self.report_path,
            bundle_path=self.bundle_path,
            summary_path=self.summary_path,
            resource_receipt=receipt.resource_preflight_receipt,
            workspace_must_exist=self._workspace_seen,
            grant_live_authority=False,
        )
        if observation.receipt.state_dict() != receipt.state_dict():
            raise RuntimeError("production preflight state changed")
        if self._panel_preflight is None or _panel_preflight_state(
            observation.panel_preflight
        ) != _panel_preflight_state(self._panel_preflight):
            raise RuntimeError("original panel preflight state changed")

    def build_runtime(
        self,
        receipt: LayerMultimodeExecutionPreflightReceipt,
    ) -> LayerMultimodeV3Runtime:
        if self._runtime_built:
            raise RuntimeError("production runtime may be built exactly once")
        self._runtime_built = True
        self.revalidate(receipt)
        from .gemma3_layer_multimode_runtime import (
            Gemma3LayerMultimodeRuntime,
            build_production_layer_multimode_runtime,
        )

        value = build_production_layer_multimode_runtime(
            receipt,
            repository_root=self.repository_root,
            cache_dir=self.cache_dir,
        )
        if type(value) is not Gemma3LayerMultimodeRuntime:
            raise TypeError("production runtime factory returned a foreign type")
        return value

    def build_panel_source(
        self,
        receipt: LayerMultimodeExecutionPreflightReceipt,
        selection_receipt_sha256: str,
    ) -> object:
        if self._panel_source_built:
            raise RuntimeError("fresh panel source may be built exactly once")
        self._panel_source_built = True
        self.revalidate(receipt)
        from . import gemma3_layer_multimode_fresh_panels as panels

        if not panels._is_live_preflight(self._panel_preflight):
            raise RuntimeError("original opaque panel preflight is no longer live")
        role_bytes = {
            role: panels._read_regular(
                self.repository_root / panels.DEFAULT_ROLE_CORPORA[role]
            )
            for role in panels.ORDERED_ROLES
        }
        value = panels.SequentialPanelSource.from_frozen_a_selection(
            self._panel_preflight,
            role_bytes,
            selection_receipt_sha256=selection_receipt_sha256,
            selected_layer_count=LAYER_COUNT,
        )
        if type(value) is not panels.SequentialPanelSource:
            raise TypeError("production panel factory returned a foreign type")
        return value


@dataclass(frozen=True, slots=True)
class ADevelopmentRecapture:
    """One new temporary capture of the already-exposed predecessor A role.

    V2 deleted its raw spools after publication.  V3 therefore re-executes
    that frozen, already-exposed corpus exactly once and requires its new
    temporary spools to close against the predecessor scalar lineage.
    """

    bridge: object
    partition_receipt: object
    source_model: object
    source_model_sha256: str
    source_capture_receipt_sha256: str
    predecessor_capture_receipt_sha256: str
    predecessor_lineage_comparison_receipt: object
    predecessor_A_runtime_receipt: object

    def __post_init__(self) -> None:
        _require_sha256(self.source_model_sha256, label="source_model_sha256")
        _require_sha256(
            self.source_capture_receipt_sha256,
            label="source_capture_receipt_sha256",
        )
        _require_sha256(
            self.predecessor_capture_receipt_sha256,
            label="predecessor_capture_receipt_sha256",
        )
        if self.bridge is None or self.partition_receipt is None:
            raise ValueError(
                "A recapture must retain bridge and partition authority"
            )
        if self.source_model is None:
            raise ValueError(
                "A recapture must retain the source through compilation"
            )
        _artifact_sha256(
            self.predecessor_lineage_comparison_receipt,
            label="predecessor A comparison",
        )
        _artifact_sha256(
            self.predecessor_A_runtime_receipt,
            label="predecessor A runtime receipt",
        )


@dataclass(frozen=True, slots=True)
class LayerMaterialization:
    layer_ordinal: int
    rows: object
    receipt: object

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or not 0 <= self.layer_ordinal < 18:
            raise ValueError("materialized layer ordinal is invalid")
        if self.rows is None or self.receipt is None:
            raise ValueError("materialization rows/receipt are required")


@dataclass(frozen=True, slots=True)
class LayerMultimodePreoptimizerAttempt:
    """Ephemeral result of one support-aware preoptimizer operation.

    Production runtimes may place only the exact structured atlas-support
    exception in ``support_error``.  The envelope keeps the orchestrator from
    catching a broad exception class while making an expected support failure
    an explicit branch of the runtime protocol.
    """

    result: object | None
    support_error: BaseException | None

    def __post_init__(self) -> None:
        if (self.result is None) is (self.support_error is None):
            raise ValueError(
                "preoptimizer attempt requires exactly one result or support error"
            )
        if self.support_error is not None and not isinstance(
            self.support_error,
            BaseException,
        ):
            raise TypeError("preoptimizer support_error must be an exception")

    @property
    def support_rejected(self) -> bool:
        return self.support_error is not None


@dataclass(frozen=True, slots=True)
class LayerMultimodeFrozenCandidate:
    """Source-free candidate reloaded from its temporary frozen bundle."""

    compiled_model: object
    candidate_identity: object
    bundle_receipt: object
    compilation_receipt: object
    bundle_path: str
    bundle_file_sha256: str
    source_free: bool = True
    native_mlp_module_count: int = 0

    def __post_init__(self) -> None:
        if self.compiled_model is None:
            raise ValueError("frozen candidate requires the reloaded model")
        if not self.bundle_path or Path(self.bundle_path).name in ("", ".", ".."):
            raise ValueError("frozen candidate bundle path is invalid")
        _require_sha256(
            self.bundle_file_sha256,
            label="bundle_file_sha256",
        )
        if self.source_free is not True or self.native_mlp_module_count != 0:
            raise ValueError("fresh roles require a source-free no-native-MLP model")


@dataclass(frozen=True, slots=True)
class LayerMultimodeFrozenPublicationEvidence:
    """Scalar bundle/report evidence retained after the compiled model dies."""

    candidate_identity: object
    bundle_receipt: object
    compilation_receipt: object
    bundle_path: str
    bundle_file_sha256: str

    def __post_init__(self) -> None:
        _artifact_sha256(self.candidate_identity, label="candidate identity")
        _artifact_sha256(self.compilation_receipt, label="compilation receipt")
        if not self.bundle_path or Path(self.bundle_path).name in ("", ".", ".."):
            raise ValueError("frozen publication bundle path is invalid")
        _require_sha256(self.bundle_file_sha256, label="bundle_file_sha256")
        if getattr(self.bundle_receipt, "bundle_sha256", None) != (
            self.bundle_file_sha256
        ):
            raise ValueError("publication evidence bundle bytes drifted")


def _frozen_publication_evidence(
    frozen: LayerMultimodeFrozenCandidate,
) -> LayerMultimodeFrozenPublicationEvidence:
    if type(frozen) is not LayerMultimodeFrozenCandidate:
        raise TypeError("publication evidence requires the exact frozen candidate")
    return LayerMultimodeFrozenPublicationEvidence(
        candidate_identity=frozen.candidate_identity,
        bundle_receipt=frozen.bundle_receipt,
        compilation_receipt=frozen.compilation_receipt,
        bundle_path=frozen.bundle_path,
        bundle_file_sha256=frozen.bundle_file_sha256,
    )


@dataclass(frozen=True, slots=True)
class LayerMultimodeDevelopmentTeardownReceipt:
    source_model_sha256: str
    source_execution_fingerprint_sha256: str
    compiled_candidate_present: bool
    compilation_receipt_sha256: str | None
    compiled_model_fingerprint_sha256: str | None
    source_compilation_aliases_released: bool
    native_baseline_retained_for_heldout_evaluation: bool
    A_activation_rows_released: bool
    A_fit_models_released: bool
    A_optimizer_state_released: bool
    source_compilation_native_mlp_alias_count: int
    native_baseline_instance_count: int = 1
    receipt_sha256: str = ""
    schema: str = TEARDOWN_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.schema != TEARDOWN_SCHEMA or self.format_version != 1:
            raise ValueError("development teardown header is invalid")
        _require_sha256(self.source_model_sha256, label="source_model_sha256")
        _require_sha256(
            self.source_execution_fingerprint_sha256,
            label="source_execution_fingerprint_sha256",
        )
        _require_exact_bool(
            self.compiled_candidate_present,
            label="compiled_candidate_present",
        )
        if self.compiled_candidate_present:
            _require_sha256(
                self.compilation_receipt_sha256,
                label="compilation_receipt_sha256",
            )
            _require_sha256(
                self.compiled_model_fingerprint_sha256,
                label="compiled_model_fingerprint_sha256",
            )
        elif (
            self.compilation_receipt_sha256 is not None
            or self.compiled_model_fingerprint_sha256 is not None
        ):
            raise ValueError("A-negative teardown cannot claim a compiled model")
        for name in (
            "source_compilation_aliases_released",
            "native_baseline_retained_for_heldout_evaluation",
            "A_activation_rows_released",
            "A_fit_models_released",
            "A_optimizer_state_released",
        ):
            if _require_exact_bool(getattr(self, name), label=name) is not True:
                raise ValueError(f"{name} must be true before B3")
        if (
            type(self.source_compilation_native_mlp_alias_count) is not int
            or self.source_compilation_native_mlp_alias_count != 0
        ):
            raise ValueError("source/compiler native MLP aliases remain before B3")
        if (
            type(self.native_baseline_instance_count) is not int
            or self.native_baseline_instance_count != 1
        ):
            raise ValueError("exactly one immutable native baseline is required")
        expected = _digest(_TEARDOWN_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif self.receipt_sha256 != expected:
            raise ValueError("development teardown digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_model_sha256": self.source_model_sha256,
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "compiled_candidate_present": self.compiled_candidate_present,
            "compilation_receipt_sha256": self.compilation_receipt_sha256,
            "compiled_model_fingerprint_sha256": (
                self.compiled_model_fingerprint_sha256
            ),
            "source_compilation_aliases_released": (
                self.source_compilation_aliases_released
            ),
            "native_baseline_retained_for_heldout_evaluation": (
                self.native_baseline_retained_for_heldout_evaluation
            ),
            "A_activation_rows_released": self.A_activation_rows_released,
            "A_fit_models_released": self.A_fit_models_released,
            "A_optimizer_state_released": self.A_optimizer_state_released,
            "source_compilation_native_mlp_alias_count": (
                self.source_compilation_native_mlp_alias_count
            ),
            "native_baseline_instance_count": self.native_baseline_instance_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class LayerMultimodeFreshPanelExecutionReceipt:
    terminal_status: str
    panel_manifest_sha256: str
    panel_manifest_file_sha256: str
    panel_receipt_sha256: str
    panel_receipt_file_sha256: str
    panel_bundle_sha256: str
    selection_receipt_sha256: str
    opened_roles: tuple[str, ...]
    ordered_freeze_receipts: tuple[tuple[str, str, str, bool], ...]
    terminal_state: str
    reserve_open_count: int
    receipt_sha256: str = ""
    schema: str = FRESH_PANEL_EXECUTION_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.schema != FRESH_PANEL_EXECUTION_SCHEMA or self.format_version != 1:
            raise ValueError("fresh-panel execution header is invalid")
        for name in (
            "panel_manifest_sha256",
            "panel_manifest_file_sha256",
            "panel_receipt_sha256",
            "panel_receipt_file_sha256",
            "panel_bundle_sha256",
            "selection_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        expected_roles = {
            TERMINAL_B3_FAILED: (B3_ROLE,),
            TERMINAL_C3_FAILED: (B3_ROLE, C3_ROLE),
            TERMINAL_D3_COMPLETED: (B3_ROLE, C3_ROLE, D3_ROLE),
        }.get(self.terminal_status)
        if expected_roles is None or self.opened_roles != expected_roles:
            raise ValueError("fresh-panel opened-role ledger is invalid")
        if len(self.ordered_freeze_receipts) != len(expected_roles):
            raise ValueError("fresh-panel freeze receipt count drifted")
        for expected_role, row in zip(
            expected_roles,
            self.ordered_freeze_receipts,
            strict=True,
        ):
            if type(row) is not tuple or len(row) != 4:
                raise TypeError("fresh-panel freeze rows must be exact tuples")
            role, stage_kind, stage_receipt_sha256, terminal_passed = row
            if role != expected_role or stage_kind != FRESH_STAGE_KIND_BY_ROLE[role]:
                raise ValueError("fresh-panel freeze role/stage order drifted")
            _require_sha256(
                stage_receipt_sha256,
                label=f"{role} stage receipt",
            )
            _require_exact_bool(
                terminal_passed,
                label=f"{role} terminal_passed",
            )
        pass_vector = tuple(row[3] for row in self.ordered_freeze_receipts)
        expected_pass_vector = {
            TERMINAL_B3_FAILED: (False,),
            TERMINAL_C3_FAILED: (True, False),
            TERMINAL_D3_COMPLETED: (True, True, pass_vector[-1]),
        }[self.terminal_status]
        if pass_vector != expected_pass_vector:
            raise ValueError("fresh-panel terminal pass vector drifted")
        expected_terminal_prefix = {
            TERMINAL_B3_FAILED: "terminal_B3_failed",
            TERMINAL_C3_FAILED: "terminal_C3_failed",
            TERMINAL_D3_COMPLETED: (
                "terminal_D3_passed" if pass_vector[-1] else "terminal_D3_failed"
            ),
        }[self.terminal_status]
        if self.terminal_state != expected_terminal_prefix:
            raise ValueError("fresh-panel terminal state drifted")
        if type(self.reserve_open_count) is not int or self.reserve_open_count != 0:
            raise ValueError("fresh-panel reserve role was opened")
        expected = _digest(
            _FRESH_PANEL_EXECUTION_DOMAIN,
            self.payload_without_sha256(),
        )
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif self.receipt_sha256 != expected:
            raise ValueError("fresh-panel execution digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "terminal_status": self.terminal_status,
            "panel_manifest_sha256": self.panel_manifest_sha256,
            "panel_manifest_file_sha256": self.panel_manifest_file_sha256,
            "panel_receipt_sha256": self.panel_receipt_sha256,
            "panel_receipt_file_sha256": self.panel_receipt_file_sha256,
            "panel_bundle_sha256": self.panel_bundle_sha256,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "opened_roles": list(self.opened_roles),
            "ordered_freeze_receipts": [
                {
                    "role": role,
                    "stage_kind": stage_kind,
                    "stage_receipt_sha256": stage_receipt_sha256,
                    "terminal_passed": terminal_passed,
                }
                for role, stage_kind, stage_receipt_sha256, terminal_passed
                in self.ordered_freeze_receipts
            ],
            "terminal_state": self.terminal_state,
            "reserve_open_count": self.reserve_open_count,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class LayerMultimodeFreshRoleCaptureReceipt:
    """Scalar-only proof that one frozen fresh role was tokenized and scored."""

    role: str
    stage_kind: str
    panel_manifest_sha256: str
    panel_manifest_file_sha256: str
    panel_receipt_sha256: str
    panel_receipt_file_sha256: str
    panel_bundle_sha256: str
    role_file_sha256: str
    selection_receipt_sha256: str
    frozen_candidate_identity_sha256: str
    runtime_session_binding_sha256: str
    ordered_tokenization_receipt_sha256: str
    ordered_example_identity_catalog_sha256: str
    ordered_row_identity_catalog_sha256: str
    family_axis_sha256: str
    batch_axis_sha256: str
    example_count: int
    batch_count: int
    predicted_next_token_position_count: int
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_logits: bool = False
    contains_raw_activations: bool = False
    contains_raw_gradients: bool = False
    receipt_sha256: str = ""
    schema: str = FRESH_ROLE_CAPTURE_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if (
            self.schema != FRESH_ROLE_CAPTURE_SCHEMA
            or type(self.format_version) is not int
            or self.format_version != 1
        ):
            raise ValueError("fresh-role capture header is invalid")
        if self.role not in FRESH_STAGE_KIND_BY_ROLE:
            raise ValueError("fresh-role capture role is invalid")
        if self.stage_kind != FRESH_STAGE_KIND_BY_ROLE[self.role]:
            raise ValueError("fresh-role capture stage kind drifted")
        for name in (
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
        ):
            _require_sha256(getattr(self, name), label=name)
        expected_counts = {
            "example_count": 48,
            "batch_count": 8,
            "predicted_next_token_position_count": _PREDICTED_ROWS_PER_ROLE,
        }
        for name, expected in expected_counts.items():
            observed = getattr(self, name)
            if type(observed) is not int or observed != expected:
                raise ValueError(f"fresh-role capture {name} must equal {expected}")
        for name in (
            "contains_prompt_text",
            "contains_token_ids",
            "contains_logits",
            "contains_raw_activations",
            "contains_raw_gradients",
        ):
            if _require_exact_bool(getattr(self, name), label=name) is not False:
                raise ValueError("fresh-role capture cannot retain raw evidence")
        expected = _digest(
            _FRESH_ROLE_CAPTURE_DOMAIN,
            self.payload_without_sha256(),
        )
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="fresh-role capture receipt_sha256",
        ) != expected:
            raise ValueError("fresh-role capture digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "receipt_sha256"
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class LayerMultimodePredecessorARuntimeReceipt:
    """Exact observed A-only predecessor execution ledger for this run."""

    execution_preflight_receipt_sha256: str
    predecessor_preflight_receipt_sha256: str
    predecessor_runtime_qualification_receipt_sha256: str
    source_capture_receipt_sha256: str
    runtime_session_binding_sha256: str
    source_model_fingerprint_before_sha256: str
    source_model_fingerprint_after_sha256: str
    source_execution_fingerprint_before_sha256: str
    source_execution_fingerprint_after_sha256: str
    role_batch_counts: tuple[tuple[str, int], ...]
    exact_control_forward_count: int
    exact_control_vjp_count: int
    dense_parity_smoke_count: int
    dense_parity_forward_count: int
    dense_parity_vjp_count: int
    total_backbone_forward_count: int
    total_terminal_cotangent_vjp_count: int
    captured_example_count: int
    supplemental_route_receipt_count: int
    mode_raw_artifact_count: int
    lexical_nll_receipt_count: int
    execution_receipt_count: int
    example_capture_receipt_count: int
    canonical_gradient_targets_per_vjp: int
    auxiliary_gradient_targets_per_vjp: int
    gradient_targets_per_vjp: int
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_raw_activations: bool = False
    contains_raw_gradients: bool = False
    contains_logits: bool = False
    receipt_sha256: str = ""
    schema: str = PREDECESSOR_A_RUNTIME_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if (
            self.schema != PREDECESSOR_A_RUNTIME_SCHEMA
            or type(self.format_version) is not int
            or self.format_version != 1
        ):
            raise ValueError("predecessor A-only runtime header is invalid")
        for name in (
            "execution_preflight_receipt_sha256",
            "predecessor_preflight_receipt_sha256",
            "predecessor_runtime_qualification_receipt_sha256",
            "source_capture_receipt_sha256",
            "runtime_session_binding_sha256",
            "source_model_fingerprint_before_sha256",
            "source_model_fingerprint_after_sha256",
            "source_execution_fingerprint_before_sha256",
            "source_execution_fingerprint_after_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.source_model_fingerprint_before_sha256
            != self.source_model_fingerprint_after_sha256
            or self.source_execution_fingerprint_before_sha256
            != self.source_execution_fingerprint_after_sha256
        ):
            raise ValueError("predecessor A-only runtime mutated its source")
        expected_values: dict[str, object] = {
            "role_batch_counts": (
                ("A_chart_transport_fit", 45),
                ("B_neighbor_discovery", 0),
                ("C_neighbor_confirmation", 0),
            ),
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
        for name, expected_value in expected_values.items():
            observed = getattr(self, name)
            if observed != expected_value or type(observed) is not type(
                expected_value
            ):
                raise ValueError(f"predecessor A-only ledger {name} drifted")
        for name in (
            "contains_prompt_text",
            "contains_token_ids",
            "contains_raw_activations",
            "contains_raw_gradients",
            "contains_logits",
        ):
            if _require_exact_bool(getattr(self, name), label=name) is not False:
                raise ValueError("predecessor A-only receipt leaked raw evidence")
        expected = _digest(
            _PREDECESSOR_A_RUNTIME_DOMAIN,
            self.payload_without_sha256(),
        )
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif _require_sha256(
            self.receipt_sha256,
            label="predecessor A-only receipt_sha256",
        ) != expected:
            raise ValueError("predecessor A-only runtime digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "receipt_sha256"
        }
        payload["role_batch_counts"] = [list(row) for row in self.role_batch_counts]
        return payload

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


def _predecessor_A_comparison_state(value: object) -> dict[str, object]:
    state = _state_dict(value, label="predecessor A lineage comparison")
    required = {
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
    if set(state) != required:
        raise ValueError("predecessor A comparison schema drifted")
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
    expected_session_dependent = [
        "batch_axis_catalog_sha256",
        "example_capture_receipt_sha256s",
        "execution_receipt_sha256s",
        "receipt_sha256",
        "runtime_session_binding_sha256",
        "supplemental_route_receipt_sha256s",
    ]
    if (
        state["schema"] != PREDECESSOR_A_COMPARISON_SCHEMA
        or type(state["format_version"]) is not int
        or state["format_version"] != 1
        or state["role"] != "A_chart_transport_fit"
        or type(state["batch_count"]) is not int
        or state["batch_count"] != 45
        or type(state["row_count"]) is not int
        or state["row_count"] != _PREDICTED_ROWS_PER_ROLE
        or state["compared_field_inventory"] != expected_compared
        or state["intentionally_session_dependent_field_inventory"]
        != expected_session_dependent
        or _require_exact_bool(
            state["runtime_sessions_expected_different"],
            label="runtime_sessions_expected_different",
        )
        is not True
        or _require_exact_bool(
            state["all_comparable_fields_matched"],
            label="all_comparable_fields_matched",
        )
        is not True
        or state["predecessor_runtime_session_binding_sha256"]
        == state["recaptured_runtime_session_binding_sha256"]
    ):
        raise ValueError("predecessor A comparison did not close")
    for name in required - {
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
        _require_sha256(state[name], label=f"predecessor comparison {name}")
    artifact_sha256 = state["artifact_sha256"]
    payload = dict(state)
    del payload["artifact_sha256"]
    if artifact_sha256 != _digest(_PREDECESSOR_A_COMPARISON_DOMAIN, payload):
        raise ValueError("predecessor A comparison digest mismatch")
    return state


@dataclass(frozen=True, slots=True)
class LayerMultimodeRuntimeExecutionReceipt:
    terminal_status: str
    runtime_implementation_sha256: str
    implementation_qualification_sha256: str
    runtime_qualification_receipt: object
    runtime_qualification_receipt_sha256: str
    execution_environment_sha256: str
    tokenizer_fingerprint_sha256: str
    predecessor_A_capture_receipt_sha256: str
    predecessor_runtime_execution_receipt_sha256: str
    predecessor_A_runtime_receipt: LayerMultimodePredecessorARuntimeReceipt
    predecessor_A_runtime_receipt_sha256: str
    predecessor_A_lineage_comparison_receipt: object
    predecessor_A_lineage_comparison_receipt_sha256: str
    predecessor_A_recaptured_capture_receipt_sha256: str
    predecessor_A_current_runtime_session_binding_sha256: str
    source_model_fingerprint_before_sha256: str
    source_model_fingerprint_after_sha256: str
    source_model_mutated: bool
    source_execution_fingerprint_before_sha256: str
    source_execution_fingerprint_after_sha256: str
    source_execution_mutated: bool
    compiled_model_fingerprint_before_sha256: str
    compiled_model_fingerprint_after_sha256: str
    compiled_execution_fingerprint_before_sha256: str
    compiled_execution_fingerprint_after_sha256: str
    compiled_candidate_mutated: bool
    runtime_session_binding_sha256: str
    fresh_role_capture_receipts: tuple[
        LayerMultimodeFreshRoleCaptureReceipt, ...
    ]
    predecessor_A_batch_count: int
    predecessor_A_exact_control_forward_count: int
    predecessor_A_exact_control_vjp_count: int
    predecessor_A_dense_parity_forward_count: int
    predecessor_A_dense_parity_vjp_count: int
    B3_batch_count: int
    B3_native_forward_count: int
    B3_candidate_layer_velocity_forward_count: int
    C3_batch_count: int
    C3_native_forward_count: int
    C3_compiled_forward_count: int
    D3_batch_count: int
    D3_native_forward_count: int
    D3_compiled_forward_count: int
    fresh_tokenization_authentication_count: int
    fresh_example_count: int
    fresh_next_token_position_count: int
    native_baseline_evaluator_count: int
    native_baseline_eval_mode: bool
    native_baseline_parameter_mutation_count: int
    native_baseline_optimizer_reference_count: int
    compiled_candidate_native_mlp_module_count: int
    reserve_open_count: int
    receipt_sha256: str = ""
    schema: str = RUNTIME_EXECUTION_SCHEMA
    format_version: int = 1
    _runtime_qualification_state_json: bytes = field(
        default=b"",
        init=False,
        repr=False,
        compare=False,
    )
    _predecessor_A_comparison_state_json: bytes = field(
        default=b"",
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            self.schema != RUNTIME_EXECUTION_SCHEMA
            or type(self.format_version) is not int
            or self.format_version != 1
        ):
            raise ValueError("runtime execution header is invalid")
        qualification_state = _state_dict(
            self.runtime_qualification_receipt,
            label="runtime qualification receipt",
        )
        if (
            _artifact_sha256(
                self.runtime_qualification_receipt,
                label="runtime qualification receipt",
            )
            != self.runtime_qualification_receipt_sha256
            or qualification_state.get("receipt_sha256")
            != self.runtime_qualification_receipt_sha256
        ):
            raise ValueError("runtime qualification receipt binding drifted")
        object.__setattr__(
            self,
            "_runtime_qualification_state_json",
            _canonical_json(qualification_state),
        )
        if type(self.predecessor_A_runtime_receipt) is not (
            LayerMultimodePredecessorARuntimeReceipt
        ):
            raise TypeError("predecessor A runtime receipt has the wrong type")
        if (
            self.predecessor_A_runtime_receipt.receipt_sha256
            != self.predecessor_A_runtime_receipt_sha256
        ):
            raise ValueError("predecessor A runtime receipt binding drifted")
        comparison_state = _predecessor_A_comparison_state(
            self.predecessor_A_lineage_comparison_receipt
        )
        object.__setattr__(
            self,
            "_predecessor_A_comparison_state_json",
            _canonical_json(comparison_state),
        )
        if (
            comparison_state["artifact_sha256"]
            != self.predecessor_A_lineage_comparison_receipt_sha256
        ):
            raise ValueError("predecessor A comparison binding drifted")
        for name in (
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
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            self.predecessor_A_runtime_receipt.source_capture_receipt_sha256
            != self.predecessor_A_recaptured_capture_receipt_sha256
            or self.predecessor_A_runtime_receipt.runtime_session_binding_sha256
            != self.predecessor_A_current_runtime_session_binding_sha256
            or comparison_state["predecessor_capture_receipt_sha256"]
            != self.predecessor_A_capture_receipt_sha256
            or comparison_state["recaptured_capture_receipt_sha256"]
            != self.predecessor_A_recaptured_capture_receipt_sha256
            or comparison_state[
                "predecessor_runtime_execution_receipt_sha256"
            ]
            != self.predecessor_runtime_execution_receipt_sha256
            or comparison_state["recaptured_runtime_session_binding_sha256"]
            != self.predecessor_A_current_runtime_session_binding_sha256
        ):
            raise ValueError("predecessor A nested lineage drifted")
        expected_session = layer_multimode_runtime_session_binding_sha256(
            runtime_implementation_sha256=self.runtime_implementation_sha256,
            implementation_qualification_sha256=(
                self.implementation_qualification_sha256
            ),
            execution_environment_sha256=self.execution_environment_sha256,
            tokenizer_fingerprint_sha256=self.tokenizer_fingerprint_sha256,
            runtime_qualification_receipt_sha256=(
                self.runtime_qualification_receipt_sha256
            ),
            source_model_fingerprint_before_sha256=(
                self.source_model_fingerprint_before_sha256
            ),
            source_model_fingerprint_after_sha256=(
                self.source_model_fingerprint_after_sha256
            ),
            source_execution_fingerprint_before_sha256=(
                self.source_execution_fingerprint_before_sha256
            ),
            source_execution_fingerprint_after_sha256=(
                self.source_execution_fingerprint_after_sha256
            ),
            compiled_model_fingerprint_before_sha256=(
                self.compiled_model_fingerprint_before_sha256
            ),
            compiled_model_fingerprint_after_sha256=(
                self.compiled_model_fingerprint_after_sha256
            ),
            compiled_execution_fingerprint_before_sha256=(
                self.compiled_execution_fingerprint_before_sha256
            ),
            compiled_execution_fingerprint_after_sha256=(
                self.compiled_execution_fingerprint_after_sha256
            ),
            compiled_candidate_mutated=self.compiled_candidate_mutated,
        )
        if _require_sha256(
            self.runtime_session_binding_sha256,
            label="runtime_session_binding_sha256",
        ) != expected_session:
            raise ValueError("runtime session binding digest mismatch")
        for name in (
            "source_model_mutated",
            "source_execution_mutated",
            "compiled_candidate_mutated",
        ):
            if _require_exact_bool(getattr(self, name), label=name) is not False:
                raise ValueError(f"{name} must remain false")
        if (
            self.source_model_fingerprint_before_sha256
            != self.source_model_fingerprint_after_sha256
            or self.source_execution_fingerprint_before_sha256
            != self.source_execution_fingerprint_after_sha256
            or self.compiled_model_fingerprint_before_sha256
            != self.compiled_model_fingerprint_after_sha256
            or self.compiled_execution_fingerprint_before_sha256
            != self.compiled_execution_fingerprint_after_sha256
        ):
            raise ValueError("source or compiled model/execution fingerprint mutated")
        opened_count = {
            TERMINAL_B3_FAILED: 1,
            TERMINAL_C3_FAILED: 2,
            TERMINAL_D3_COMPLETED: 3,
        }.get(self.terminal_status)
        if opened_count is None:
            raise ValueError("runtime execution terminal status is invalid")
        if type(self.fresh_role_capture_receipts) is not tuple:
            raise TypeError("fresh role capture inventory must be an exact tuple")
        expected_roles = (B3_ROLE, C3_ROLE, D3_ROLE)[:opened_count]
        if (
            len(self.fresh_role_capture_receipts) != opened_count
            or tuple(
                type(value) for value in self.fresh_role_capture_receipts
            )
            != (LayerMultimodeFreshRoleCaptureReceipt,) * opened_count
            or tuple(value.role for value in self.fresh_role_capture_receipts)
            != expected_roles
        ):
            raise ValueError("fresh role capture order/count drifted")
        first_fresh = self.fresh_role_capture_receipts[0]
        common_fields = (
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
        for value in self.fresh_role_capture_receipts:
            if any(
                getattr(value, name) != getattr(first_fresh, name)
                for name in common_fields
            ):
                raise ValueError("fresh role capture common lineage drifted")
            if value.runtime_session_binding_sha256 != (
                self.runtime_session_binding_sha256
            ):
                raise ValueError("fresh role capture/runtime session drifted")
        expected = {
            "predecessor_A_batch_count": 45,
            "predecessor_A_exact_control_forward_count": 45,
            "predecessor_A_exact_control_vjp_count": 45,
            "predecessor_A_dense_parity_forward_count": 1,
            "predecessor_A_dense_parity_vjp_count": 1,
            "B3_batch_count": 8,
            "B3_native_forward_count": 8,
            "B3_candidate_layer_velocity_forward_count": 8 * LAYER_COUNT,
            "C3_batch_count": 8 if opened_count >= 2 else 0,
            "C3_native_forward_count": 8 if opened_count >= 2 else 0,
            "C3_compiled_forward_count": 8 if opened_count >= 2 else 0,
            "D3_batch_count": 8 if opened_count >= 3 else 0,
            "D3_native_forward_count": 8 if opened_count >= 3 else 0,
            "D3_compiled_forward_count": 8 if opened_count >= 3 else 0,
            "fresh_tokenization_authentication_count": opened_count,
            "fresh_example_count": 48 * opened_count,
            "fresh_next_token_position_count": (
                _PREDICTED_ROWS_PER_ROLE * opened_count
            ),
            "native_baseline_evaluator_count": 1,
            "native_baseline_parameter_mutation_count": 0,
            "native_baseline_optimizer_reference_count": 0,
            "compiled_candidate_native_mlp_module_count": 0,
            "reserve_open_count": 0,
        }
        for name, value in expected.items():
            observed = getattr(self, name)
            if type(observed) is not int or observed != value:
                raise ValueError(f"runtime ledger {name} must equal {value}")
        A_runtime = self.predecessor_A_runtime_receipt
        if (
            self.predecessor_A_batch_count != A_runtime.role_batch_counts[0][1]
            or self.predecessor_A_exact_control_forward_count
            != A_runtime.exact_control_forward_count
            or self.predecessor_A_exact_control_vjp_count
            != A_runtime.exact_control_vjp_count
            or self.predecessor_A_dense_parity_forward_count
            != A_runtime.dense_parity_forward_count
            or self.predecessor_A_dense_parity_vjp_count
            != A_runtime.dense_parity_vjp_count
        ):
            raise ValueError("outer predecessor A ledger differs from nested receipt")
        if _require_exact_bool(
            self.native_baseline_eval_mode,
            label="native_baseline_eval_mode",
        ) is not True:
            raise ValueError("native baseline must remain in eval mode")
        digest = _digest(_RUNTIME_EXECUTION_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", digest)
        elif self.receipt_sha256 != digest:
            raise ValueError("runtime execution digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        fields = (
            "schema",
            "format_version",
            "terminal_status",
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
        )
        payload = {name: getattr(self, name) for name in fields}
        payload["runtime_qualification_receipt"] = json.loads(
            self._runtime_qualification_state_json
        )
        payload["predecessor_A_runtime_receipt"] = (
            self.predecessor_A_runtime_receipt.state_dict()
        )
        payload["predecessor_A_lineage_comparison_receipt"] = json.loads(
            self._predecessor_A_comparison_state_json
        )
        payload["fresh_role_capture_receipts"] = [
            value.state_dict() for value in self.fresh_role_capture_receipts
        ]
        return payload

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class LayerMultimodePostTerminalCleanupReceipt:
    terminal_status: str
    runtime_execution_receipt_sha256: str | None
    runtime_session_binding_sha256: str | None
    fresh_panel_execution_receipt_sha256: str | None
    runtime_closed: bool
    native_baseline_released: bool
    native_baseline_instance_count_after_close: int
    compiled_candidate_released: bool
    model_reference_count_after_close: int
    tokenizer_reference_count_after_close: int
    temporary_A_spools_removed: bool
    temporary_materializations_removed: bool
    temporary_fit_state_removed: bool
    frozen_bundle_retained_for_publication: bool
    reserve_open_count: int
    receipt_sha256: str = ""
    schema: str = POST_TERMINAL_CLEANUP_SCHEMA
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.schema != POST_TERMINAL_CLEANUP_SCHEMA or self.format_version != 1:
            raise ValueError("post-terminal cleanup header is invalid")
        if self.terminal_status not in (
            TERMINAL_A_NO_CANDIDATE,
            TERMINAL_B3_FAILED,
            TERMINAL_C3_FAILED,
            TERMINAL_D3_COMPLETED,
        ):
            raise ValueError("post-terminal cleanup status is invalid")
        if self.terminal_status == TERMINAL_A_NO_CANDIDATE:
            if any(
                value is not None
                for value in (
                    self.runtime_execution_receipt_sha256,
                    self.runtime_session_binding_sha256,
                    self.fresh_panel_execution_receipt_sha256,
                )
            ):
                raise ValueError("A-negative cleanup cannot claim fresh execution")
        else:
            for name in (
                "runtime_execution_receipt_sha256",
                "runtime_session_binding_sha256",
                "fresh_panel_execution_receipt_sha256",
            ):
                _require_sha256(getattr(self, name), label=name)
        for name in (
            "runtime_closed",
            "native_baseline_released",
            "compiled_candidate_released",
            "temporary_A_spools_removed",
            "temporary_materializations_removed",
            "temporary_fit_state_removed",
        ):
            if _require_exact_bool(getattr(self, name), label=name) is not True:
                raise ValueError(f"cleanup invariant {name} must be true")
        expected_bundle_retained = self.terminal_status != TERMINAL_A_NO_CANDIDATE
        if (
            _require_exact_bool(
                self.frozen_bundle_retained_for_publication,
                label="frozen_bundle_retained_for_publication",
            )
            is not expected_bundle_retained
        ):
            raise ValueError("cleanup frozen-bundle state is inconsistent")
        for name in (
            "native_baseline_instance_count_after_close",
            "model_reference_count_after_close",
            "tokenizer_reference_count_after_close",
            "reserve_open_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value != 0:
                raise ValueError(f"cleanup {name} must equal zero")
        digest = _digest(
            _POST_TERMINAL_CLEANUP_DOMAIN,
            self.payload_without_sha256(),
        )
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", digest)
        elif self.receipt_sha256 != digest:
            raise ValueError("post-terminal cleanup digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        fields = (
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
        )
        return {name: getattr(self, name) for name in fields}

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


class LayerMultimodeV3Runtime(ABC):
    """Distinct v3 runtime seam; old generic A/B/C capture is not accepted."""

    test_only_nonproduction: bool = False

    @abstractmethod
    def recapture_exposed_predecessor_A(
        self,
        *,
        preflight: LayerMultimodeExecutionPreflightReceipt,
        workspace: Path,
    ) -> ADevelopmentRecapture:
        raise NotImplementedError

    @abstractmethod
    def materialize_A_layer(
        self,
        recapture: ADevelopmentRecapture,
        layer_ordinal: int,
    ) -> LayerMaterialization:
        raise NotImplementedError

    @abstractmethod
    def build_materialization_inventory(
        self,
        recapture: ADevelopmentRecapture,
        layers: tuple[LayerMaterialization, ...],
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def fit_chart_atlas(
        self,
        materialization: LayerMaterialization,
        *,
        chart_count: int,
        tangent_rank: int,
    ) -> LayerMultimodePreoptimizerAttempt:
        raise NotImplementedError

    @abstractmethod
    def fit_candidate(
        self,
        materialization: LayerMaterialization,
        *,
        chart_count: int,
        mode_count: int,
        tangent_rank: int,
        atlas: object,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def evaluate_split_refit(
        self,
        materialization: LayerMaterialization,
        *,
        chart_count: int,
        mode_count: int,
        tangent_rank: int,
    ) -> LayerMultimodePreoptimizerAttempt:
        raise NotImplementedError

    @abstractmethod
    def lower_candidate_support_rejection(
        self,
        *,
        materialization: LayerMaterialization,
        chart_count: int,
        mode_count: int,
        tangent_rank: int,
        candidate_fit: object | None,
        support_error: BaseException,
        recapture: ADevelopmentRecapture,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def lower_candidate_evaluation(
        self,
        *,
        candidate_fit: object,
        split_refit: object | None,
        recapture: ADevelopmentRecapture,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def select_full_grid(self, evaluations: tuple[object, ...]) -> object:
        raise NotImplementedError

    @abstractmethod
    def build_selected_inventory(
        self,
        *,
        recapture: ADevelopmentRecapture,
        materialization_inventory: object,
        selection: object,
        selected_fits: tuple[object, ...],
        selected_splits: tuple[object | None, ...],
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def compile_source_free(
        self,
        *,
        source_model: object,
        selected_inventory: object,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def freeze_compiled_candidate(
        self,
        *,
        compiled: object,
        selection: object,
        workspace: Path,
    ) -> LayerMultimodeFrozenCandidate:
        raise NotImplementedError

    @abstractmethod
    def release_development_state(
        self,
        *,
        recapture: ADevelopmentRecapture,
        materializations: tuple[LayerMaterialization, ...],
        atlases: tuple[object, ...],
        candidate_fits: tuple[object | None, ...],
        split_refits: tuple[object | None, ...],
        compiled: object | None,
    ) -> object:
        """Drop runtime-owned A/compiler aliases and issue a one-shot token.

        This phase cannot itself attest caller-local release because all
        development objects are necessarily still live as call arguments.
        """

        raise NotImplementedError

    @abstractmethod
    def confirm_development_teardown(
        self,
        prepared_release: object,
    ) -> LayerMultimodeDevelopmentTeardownReceipt:
        """Attest teardown only after the worker has dropped caller aliases."""

        raise NotImplementedError

    @abstractmethod
    def evaluate_b3_teacher_forced(
        self,
        *,
        panel: object,
        candidate: LayerMultimodeFrozenCandidate,
        fresh_role_capture_receipt: LayerMultimodeFreshRoleCaptureReceipt,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def authenticate_fresh_panel_tokenization(
        self,
        *,
        panel_source: object,
        panel: object,
        role: str,
        selection_receipt_sha256: str,
        frozen_candidate_identity_sha256: str,
    ) -> LayerMultimodeFreshRoleCaptureReceipt:
        """Bind one decoded role to the tokenizer, panel, and candidate."""

        raise NotImplementedError

    @abstractmethod
    def evaluate_c3_full_stack(
        self,
        *,
        panel: object,
        candidate: LayerMultimodeFrozenCandidate,
        b3_evaluation: object,
        fresh_role_capture_receipt: LayerMultimodeFreshRoleCaptureReceipt,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def evaluate_d3_final(
        self,
        *,
        panel: object,
        candidate: LayerMultimodeFrozenCandidate,
        c3_evaluation: object,
        fresh_role_capture_receipt: LayerMultimodeFreshRoleCaptureReceipt,
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def build_runtime_execution_receipt(
        self,
        *,
        terminal_status: str,
    ) -> LayerMultimodeRuntimeExecutionReceipt:
        raise NotImplementedError

    @abstractmethod
    def close(
        self,
        *,
        terminal_status: str,
        runtime_execution_receipt_sha256: str | None,
        runtime_session_binding_sha256: str | None,
        fresh_panel_execution_receipt_sha256: str | None,
    ) -> LayerMultimodePostTerminalCleanupReceipt:
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> None:
        """Best-effort non-reporting release for engineering failures."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LayerMultimodeNoAdmissibleSelectionReceipt:
    """Authenticated full-grid terminal A receipt.

    It records all 324 candidate and gate-decision digests, not merely the
    first selector exception string.  It therefore remains a scientific
    negative result that can be published without opening B3.
    """

    source_model_sha256: str
    selection_policy_sha256: str
    fit_policy_sha256: str
    fit_a_sha256: str
    validation_a_sha256: str
    evaluation_protocol_sha256: str
    candidate_sha256s: tuple[str, ...]
    decision_sha256s: tuple[str, ...]
    failed_layer_ordinals: tuple[int, ...]
    first_failed_layer_ordinal: int
    receipt_sha256: str = ""
    schema: str = NO_SELECTION_SCHEMA
    format_version: int = 2
    candidate_count: int = TOTAL_CANDIDATE_COUNT
    fresh_roles_opened: tuple[str, ...] = ()
    reserve_opened: bool = False

    def __post_init__(self) -> None:
        if self.schema != NO_SELECTION_SCHEMA or self.format_version != 2:
            raise ValueError("no-selection receipt header is invalid")
        for name in (
            "source_model_sha256",
            "selection_policy_sha256",
            "fit_policy_sha256",
            "fit_a_sha256",
            "validation_a_sha256",
            "evaluation_protocol_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            len(self.candidate_sha256s) != TOTAL_CANDIDATE_COUNT
            or len(set(self.candidate_sha256s)) != TOTAL_CANDIDATE_COUNT
            or len(self.decision_sha256s) != TOTAL_CANDIDATE_COUNT
            or len(set(self.decision_sha256s)) != TOTAL_CANDIDATE_COUNT
        ):
            raise ValueError("no-selection receipt must cover 324 unique rows")
        for value in (*self.candidate_sha256s, *self.decision_sha256s):
            _require_sha256(value, label="candidate/decision digest")
        if (
            not self.failed_layer_ordinals
            or self.failed_layer_ordinals
            != tuple(sorted(set(self.failed_layer_ordinals)))
            or any(
                type(value) is not int or not 0 <= value < 18
                for value in self.failed_layer_ordinals
            )
            or self.first_failed_layer_ordinal != self.failed_layer_ordinals[0]
        ):
            raise ValueError("failed-layer inventory is invalid")
        if self.candidate_count != TOTAL_CANDIDATE_COUNT:
            raise ValueError("candidate_count drifted")
        if self.fresh_roles_opened != () or self.reserve_opened is not False:
            raise ValueError("fresh roles must remain unopened after A failure")
        expected = _digest(_NO_SELECTION_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif self.receipt_sha256 != expected:
            raise ValueError("no-selection receipt digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_model_sha256": self.source_model_sha256,
            "selection_policy_sha256": self.selection_policy_sha256,
            "fit_policy_sha256": self.fit_policy_sha256,
            "fit_a_sha256": self.fit_a_sha256,
            "validation_a_sha256": self.validation_a_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "candidate_sha256s": list(self.candidate_sha256s),
            "decision_sha256s": list(self.decision_sha256s),
            "failed_layer_ordinals": list(self.failed_layer_ordinals),
            "first_failed_layer_ordinal": self.first_failed_layer_ordinal,
            "candidate_count": self.candidate_count,
            "fresh_roles_opened": [],
            "reserve_opened": self.reserve_opened,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class _MutableCounterSnapshot:
    materialization_count: int
    atlas_fit_attempt_count: int
    atlas_fit_success_count: int
    atlas_support_rejection_count: int
    candidate_fit_count: int
    split_refit_attempt_count: int
    split_refit_count: int
    split_refit_support_rejection_count: int
    candidate_evaluation_receipt_count: int
    candidate_support_rejection_count: int
    candidate_inventory_count: int
    selection_attempt_count: int
    compilation_count: int
    candidate_freeze_count: int
    development_teardown_count: int
    B3_open_count: int
    B3_evaluation_count: int
    C3_open_count: int
    C3_evaluation_count: int
    D3_open_count: int
    D3_evaluation_count: int


@dataclass(frozen=True, slots=True)
class LayerMultimodeExperimentCountersReceipt:
    terminal_status: str
    execution_preflight_receipt_sha256: str
    selection_receipt_sha256: str
    materialization_count: int
    atlas_fit_attempt_count: int
    atlas_fit_success_count: int
    atlas_support_rejection_count: int
    candidate_fit_count: int
    split_refit_attempt_count: int
    split_refit_count: int
    split_refit_support_rejection_count: int
    candidate_evaluation_receipt_count: int
    candidate_support_rejection_count: int
    candidate_inventory_count: int
    selection_attempt_count: int
    compilation_count: int
    candidate_freeze_count: int
    development_teardown_count: int
    B3_open_count: int
    B3_evaluation_count: int
    C3_open_count: int
    C3_evaluation_count: int
    D3_open_count: int
    D3_evaluation_count: int
    receipt_sha256: str = ""
    schema: str = COUNTERS_SCHEMA
    format_version: int = 2
    strict_preflight_count: int = 1
    runtime_factory_count: int = 1
    exposed_predecessor_A_recapture_count: int = 1
    fresh_v3_A_capture_count: int = 0
    reserve_open_count: int = 0
    native_baseline_evaluator_count: int = 1
    native_baseline_eval_mode: bool = True
    native_baseline_parameter_mutation_count: int = 0
    native_baseline_optimizer_reference_count: int = 0
    compiled_candidate_native_mlp_module_count: int = 0

    def __post_init__(self) -> None:
        if self.schema != COUNTERS_SCHEMA or self.format_version != 2:
            raise ValueError("experiment counters header is invalid")
        _require_sha256(
            self.execution_preflight_receipt_sha256,
            label="execution_preflight_receipt_sha256",
        )
        _require_sha256(
            self.selection_receipt_sha256,
            label="selection_receipt_sha256",
        )
        terminal_expectations = {
            TERMINAL_A_NO_CANDIDATE: (0, 0, 1, 0, 0, 0, 0, 0),
            TERMINAL_B3_FAILED: (1, 1, 1, 1, 1, 0, 0, 0),
            TERMINAL_C3_FAILED: (1, 1, 1, 1, 1, 1, 1, 0),
            TERMINAL_D3_COMPLETED: (1, 1, 1, 1, 1, 1, 1, 1),
        }
        if self.terminal_status not in terminal_expectations:
            raise ValueError("unknown experiment terminal status")
        fixed = {
            "strict_preflight_count": 1,
            "runtime_factory_count": 1,
            "exposed_predecessor_A_recapture_count": 1,
            "fresh_v3_A_capture_count": 0,
            "materialization_count": LAYER_COUNT,
            "atlas_fit_attempt_count": ATLAS_FIT_COUNT,
            "candidate_inventory_count": TOTAL_CANDIDATE_COUNT,
            "selection_attempt_count": 1,
            "reserve_open_count": 0,
            "native_baseline_evaluator_count": 1,
            "native_baseline_parameter_mutation_count": 0,
            "native_baseline_optimizer_reference_count": 0,
            "compiled_candidate_native_mlp_module_count": 0,
        }
        for name, expected in fixed.items():
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f"{name} must equal {expected}")
        for name in (
            "atlas_fit_success_count",
            "atlas_support_rejection_count",
            "candidate_fit_count",
            "split_refit_attempt_count",
            "split_refit_count",
            "split_refit_support_rejection_count",
            "candidate_evaluation_receipt_count",
            "candidate_support_rejection_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")
        if (
            self.atlas_fit_success_count + self.atlas_support_rejection_count
            != self.atlas_fit_attempt_count
        ):
            raise ValueError("atlas fit attempts do not close")
        if self.candidate_fit_count != 3 * self.atlas_fit_success_count:
            raise ValueError("candidate fits do not close against supported atlases")
        if self.split_refit_attempt_count != 2 * self.atlas_fit_success_count:
            raise ValueError("split-refit attempts do not close against atlases")
        if (
            self.split_refit_count + self.split_refit_support_rejection_count
            != self.split_refit_attempt_count
        ):
            raise ValueError("split-refit attempts do not close")
        if self.candidate_support_rejection_count != (
            3 * self.atlas_support_rejection_count
            + self.split_refit_support_rejection_count
        ):
            raise ValueError("candidate support rejections do not close")
        if self.candidate_evaluation_receipt_count != (
            self.candidate_fit_count - self.split_refit_support_rejection_count
        ):
            raise ValueError("candidate evaluation receipts do not close")
        if (
            self.candidate_evaluation_receipt_count
            + self.candidate_support_rejection_count
            != self.candidate_inventory_count
        ):
            raise ValueError("candidate inventory does not close")
        if _require_exact_bool(
            self.native_baseline_eval_mode,
            label="native_baseline_eval_mode",
        ) is not True:
            raise ValueError("native baseline evaluator must remain in eval mode")
        expected_tail = terminal_expectations[self.terminal_status]
        observed_tail = (
            self.compilation_count,
            self.candidate_freeze_count,
            self.development_teardown_count,
            self.B3_open_count,
            self.B3_evaluation_count,
            self.C3_open_count,
            self.C3_evaluation_count,
            self.D3_open_count,
        )
        if observed_tail != expected_tail:
            raise ValueError("terminal phase counters are inconsistent")
        expected_d3_eval = 1 if self.terminal_status == TERMINAL_D3_COMPLETED else 0
        if self.D3_evaluation_count != expected_d3_eval:
            raise ValueError("D3 evaluation count is inconsistent")
        expected = _digest(_COUNTERS_DOMAIN, self.payload_without_sha256())
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", expected)
        elif self.receipt_sha256 != expected:
            raise ValueError("experiment counters digest mismatch")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
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
                "atlas_fit_attempt_count",
                "atlas_fit_success_count",
                "atlas_support_rejection_count",
                "candidate_fit_count",
                "split_refit_attempt_count",
                "split_refit_count",
                "split_refit_support_rejection_count",
                "candidate_evaluation_receipt_count",
                "candidate_support_rejection_count",
                "candidate_inventory_count",
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
            )
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self.payload_without_sha256(),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class NoAdmissibleCandidateOutcome:
    no_selection_receipt: LayerMultimodeNoAdmissibleSelectionReceipt
    counters: LayerMultimodeExperimentCountersReceipt
    terminal_status: str = TERMINAL_A_NO_CANDIDATE


@dataclass(frozen=True, slots=True)
class B3FidelityFailureOutcome:
    b3_evaluation: object
    counters: LayerMultimodeExperimentCountersReceipt
    terminal_status: str = TERMINAL_B3_FAILED


@dataclass(frozen=True, slots=True)
class C3FidelityFailureOutcome:
    b3_evaluation: object
    c3_evaluation: object
    counters: LayerMultimodeExperimentCountersReceipt
    terminal_status: str = TERMINAL_C3_FAILED


@dataclass(frozen=True, slots=True)
class D3CompletedOutcome:
    b3_evaluation: object
    c3_evaluation: object
    d3_evaluation: object
    counters: LayerMultimodeExperimentCountersReceipt
    terminal_status: str = TERMINAL_D3_COMPLETED


TerminalOutcome = (
    NoAdmissibleCandidateOutcome
    | B3FidelityFailureOutcome
    | C3FidelityFailureOutcome
    | D3CompletedOutcome
)


@dataclass(frozen=True, slots=True)
class LayerMultimodeExperimentRunResult:
    outcome: TerminalOutcome
    publication: object | None
    workspace_removed: bool
    runtime_closed: bool
    reserve_unopened: bool = True


@dataclass(slots=True)
class _Counters:
    materialization_count: int = 0
    atlas_fit_attempt_count: int = 0
    atlas_fit_success_count: int = 0
    atlas_support_rejection_count: int = 0
    candidate_fit_count: int = 0
    split_refit_attempt_count: int = 0
    split_refit_count: int = 0
    split_refit_support_rejection_count: int = 0
    candidate_evaluation_receipt_count: int = 0
    candidate_support_rejection_count: int = 0
    candidate_inventory_count: int = 0
    selection_attempt_count: int = 0
    compilation_count: int = 0
    candidate_freeze_count: int = 0
    development_teardown_count: int = 0
    B3_open_count: int = 0
    B3_evaluation_count: int = 0
    C3_open_count: int = 0
    C3_evaluation_count: int = 0
    D3_open_count: int = 0
    D3_evaluation_count: int = 0

    def freeze(
        self,
        terminal_status: str,
        *,
        execution_preflight_receipt_sha256: str,
        selection_receipt_sha256: str,
    ) -> LayerMultimodeExperimentCountersReceipt:
        return LayerMultimodeExperimentCountersReceipt(
            terminal_status=terminal_status,
            execution_preflight_receipt_sha256=(
                execution_preflight_receipt_sha256
            ),
            selection_receipt_sha256=selection_receipt_sha256,
            **{
                name: getattr(self, name)
                for name in _Counters.__dataclass_fields__
            },
        )


def _candidate_key(value: object) -> tuple[int, int, int, int]:
    try:
        key = (
            getattr(value, "layer_ordinal"),
            getattr(value, "chart_count"),
            getattr(value, "dynamic_mode_count"),
            getattr(value, "tangent_rank"),
        )
    except AttributeError as error:
        raise TypeError("candidate evaluation lacks frozen coordinates") from error
    if any(type(item) is not int for item in key):
        raise TypeError("candidate coordinates must be exact integers")
    return key


def _build_no_selection_receipt(
    evaluations: tuple[object, ...],
    error: BaseException,
) -> LayerMultimodeNoAdmissibleSelectionReceipt:
    from . import gemma3_layer_multimode_selection as selection_module

    expected_error = selection_module.NoAdmissibleLayerMultimodeCandidateError
    if type(error) is not expected_error:
        raise TypeError("selector failure is not the exact typed no-candidate error")
    canonical = tuple(sorted(evaluations, key=_candidate_key))
    if len(canonical) != TOTAL_CANDIDATE_COUNT:
        raise ValueError("no-selection receipt requires the complete grid")
    first = canonical[0]
    decisions: list[object] = []
    failed_layers: list[int] = []
    for layer in range(LAYER_COUNT):
        layer_candidates = tuple(
            value for value in canonical if _candidate_key(value)[0] == layer
        )
        by_coordinates = {
            (
                _candidate_key(value)[1],
                _candidate_key(value)[2],
                _candidate_key(value)[3],
            ): value
            for value in layer_candidates
        }
        layer_decisions: list[object] = []
        for candidate in layer_candidates:
            _, chart_count, _, tangent_rank = _candidate_key(candidate)
            decision = selection_module._decision_for_candidate(
                candidate,
                by_coordinates[(chart_count, 1, tangent_rank)],
            )
            layer_decisions.append(decision)
        decisions.extend(layer_decisions)
        if not any(getattr(value, "passed") is True for value in layer_decisions):
            failed_layers.append(layer)
    if not failed_layers or error.layer_ordinal != failed_layers[0]:
        raise RuntimeError("typed selector failure does not recompute from full grid")
    error_decisions = tuple(
        _artifact_sha256(value, label="selector error decision")
        for value in error.decisions
    )
    recomputed_error_decisions = tuple(
        _artifact_sha256(value, label="recomputed decision")
        for value in decisions
        if getattr(value, "layer_ordinal") == error.layer_ordinal
    )
    if error_decisions != recomputed_error_decisions:
        raise RuntimeError("selector error decisions differ from recomputation")
    return LayerMultimodeNoAdmissibleSelectionReceipt(
        source_model_sha256=getattr(first, "source_model_sha256"),
        selection_policy_sha256=(
            selection_module.layer_multimode_selection_policy_sha256()
        ),
        fit_policy_sha256=getattr(first, "fit_policy_sha256"),
        fit_a_sha256=getattr(first, "fit_a_sha256"),
        validation_a_sha256=getattr(first, "validation_a_sha256"),
        evaluation_protocol_sha256=getattr(
            first,
            "evaluation_protocol_sha256",
        ),
        candidate_sha256s=tuple(
            _artifact_sha256(value, label="candidate evaluation")
            for value in canonical
        ),
        decision_sha256s=tuple(
            _artifact_sha256(value, label="candidate gate decision")
            for value in decisions
        ),
        failed_layer_ordinals=tuple(failed_layers),
        first_failed_layer_ordinal=failed_layers[0],
    )


def _selected_objects(
    selection: object,
    evaluations: tuple[object, ...],
    fits: tuple[object | None, ...],
    splits: tuple[object | None, ...],
) -> tuple[tuple[object, ...], tuple[object | None, ...]]:
    layers = getattr(selection, "layers", None)
    if type(layers) is not tuple or len(layers) != LAYER_COUNT:
        raise TypeError("selection must expose 18 ordered layer receipts")
    by_hash = {
        _artifact_sha256(value, label="candidate evaluation"): index
        for index, value in enumerate(evaluations)
    }
    selected_fits: list[object] = []
    selected_splits: list[object | None] = []
    for layer_ordinal, layer in enumerate(layers):
        selected_hash = getattr(layer, "selected_candidate_sha256", None)
        if selected_hash not in by_hash:
            raise ValueError("selected candidate is absent from the full grid")
        index = by_hash[selected_hash]
        candidate_key = _candidate_key(evaluations[index])
        if candidate_key[0] != layer_ordinal:
            raise ValueError("selected layer/candidate order drifted")
        selected_fit = fits[index]
        selected_split = splits[index]
        if selected_fit is None or (
            candidate_key[2] > 1 and selected_split is None
        ):
            raise ValueError("selector chose a support-rejected candidate")
        selected_fits.append(selected_fit)
        selected_splits.append(selected_split)
    return tuple(selected_fits), tuple(selected_splits)


def _require_passed(value: object, *, label: str) -> bool:
    passed = getattr(value, "passed", None)
    if type(passed) is not bool:
        raise TypeError(f"{label} must expose an exact bool passed field")
    return passed


def _panel_open(panel_source: object, role: str) -> object:
    method = getattr(panel_source, "open_role", None)
    if not callable(method):
        raise TypeError("panel source does not implement open_role")
    return method(role)


def _require_fresh_role_capture(
    value: object,
    *,
    role: str,
    preflight: LayerMultimodeExecutionPreflightReceipt,
    selection_receipt_sha256: str,
    frozen_candidate_identity_sha256: str,
) -> LayerMultimodeFreshRoleCaptureReceipt:
    if type(value) is not LayerMultimodeFreshRoleCaptureReceipt:
        raise TypeError("runtime returned the wrong fresh-role capture type")
    expected = {
        "role": role,
        "stage_kind": FRESH_STAGE_KIND_BY_ROLE[role],
        "panel_manifest_sha256": preflight.panel_manifest_sha256,
        "panel_manifest_file_sha256": preflight.panel_manifest_file_sha256,
        "panel_receipt_sha256": preflight.panel_receipt_sha256,
        "panel_receipt_file_sha256": preflight.panel_receipt_file_sha256,
        "panel_bundle_sha256": preflight.panel_bundle_sha256,
        "selection_receipt_sha256": selection_receipt_sha256,
        "frozen_candidate_identity_sha256": frozen_candidate_identity_sha256,
    }
    if any(
        getattr(value, name) != expected_value
        for name, expected_value in expected.items()
    ):
        raise RuntimeError("fresh-role capture lineage differs from preflight")
    return value


def _panel_freeze(
    panel_source: object,
    *,
    role: str,
    receipt_sha256: str,
    terminal_passed: bool,
) -> None:
    method = getattr(panel_source, "freeze_current", None)
    if not callable(method):
        raise TypeError("panel source does not implement freeze_current")
    try:
        expected_stage_kind = FRESH_STAGE_KIND_BY_ROLE[role]
    except KeyError as error:
        raise ValueError("unknown executable fresh role") from error
    advertised = getattr(
        panel_source,
        "required_freeze_stage_kind",
        expected_stage_kind,
    )
    if advertised != expected_stage_kind:
        raise RuntimeError("panel source freeze-stage contract drifted")
    method(
        expected_stage_kind,
        receipt_sha256,
        terminal_passed=terminal_passed,
    )


def _assert_panel_closure(panel_source: object, terminal_status: str) -> None:
    expected_by_status = {
        TERMINAL_B3_FAILED: (B3_ROLE,),
        TERMINAL_C3_FAILED: (B3_ROLE, C3_ROLE),
        TERMINAL_D3_COMPLETED: (B3_ROLE, C3_ROLE, D3_ROLE),
    }
    opened_roles = getattr(panel_source, "opened_roles", None)
    if opened_roles != expected_by_status[terminal_status]:
        raise RuntimeError("fresh panel source opened-role ledger drifted")
    if getattr(panel_source, "reserve_open_count", None) != 0:
        raise RuntimeError("reserve role was semantically opened")
    if getattr(panel_source, "current_role", None) is not None:
        raise RuntimeError("fresh panel source retained an unfrozen role")


def _build_fresh_panel_execution_receipt(
    *,
    panel_source: object,
    preflight: LayerMultimodeExecutionPreflightReceipt,
    selection_receipt_sha256: str,
    terminal_status: str,
) -> LayerMultimodeFreshPanelExecutionReceipt:
    _assert_panel_closure(panel_source, terminal_status)
    opened_roles = getattr(panel_source, "opened_roles", None)
    if type(opened_roles) is not tuple:
        raise TypeError("panel source opened_roles must be an exact tuple")
    if getattr(panel_source, "panel_manifest_sha256", None) not in (
        None,
        preflight.panel_manifest_sha256,
    ):
        raise RuntimeError("fresh panel manifest differs from preflight")
    raw_freezes = getattr(panel_source, "freeze_receipts", None)
    if not isinstance(raw_freezes, Mapping):
        raise TypeError("panel source lacks an ordered freeze receipt ledger")
    rows: list[tuple[str, str, str, bool]] = []
    if tuple(raw_freezes) != opened_roles:
        raise RuntimeError("fresh panel freeze receipt order drifted")
    for role in opened_roles:
        state = raw_freezes[role]
        if not isinstance(state, Mapping) or set(state) != {
            "stage_kind",
            "stage_receipt_sha256",
            "terminal_passed",
        }:
            raise ValueError("fresh panel freeze receipt schema drifted")
        stage_kind = state["stage_kind"]
        stage_sha = state["stage_receipt_sha256"]
        passed = state["terminal_passed"]
        if not isinstance(stage_kind, str):
            raise TypeError("fresh panel stage kind must be a string")
        _require_sha256(stage_sha, label=f"{role} freeze receipt")
        _require_exact_bool(passed, label=f"{role} freeze passed")
        rows.append((role, stage_kind, stage_sha, passed))
    terminal_state = getattr(panel_source, "terminal_state", None)
    if not isinstance(terminal_state, str):
        raise TypeError("panel source terminal_state must be a string")
    return LayerMultimodeFreshPanelExecutionReceipt(
        terminal_status=terminal_status,
        panel_manifest_sha256=preflight.panel_manifest_sha256,
        panel_manifest_file_sha256=preflight.panel_manifest_file_sha256,
        panel_receipt_sha256=preflight.panel_receipt_sha256,
        panel_receipt_file_sha256=preflight.panel_receipt_file_sha256,
        panel_bundle_sha256=preflight.panel_bundle_sha256,
        selection_receipt_sha256=selection_receipt_sha256,
        opened_roles=opened_roles,
        ordered_freeze_receipts=tuple(rows),
        terminal_state=terminal_state,
        reserve_open_count=getattr(panel_source, "reserve_open_count", None),
    )


def _report_payload(
    *,
    preflight: LayerMultimodeExecutionPreflightReceipt,
    terminal_status: str,
    materialization_inventory: object,
    selection: object,
    frozen_publication: LayerMultimodeFrozenPublicationEvidence,
    development_teardown: LayerMultimodeDevelopmentTeardownReceipt,
    experiment_counters: LayerMultimodeExperimentCountersReceipt,
    fresh_panel_execution: LayerMultimodeFreshPanelExecutionReceipt,
    runtime_execution: LayerMultimodeRuntimeExecutionReceipt,
    post_terminal_cleanup: LayerMultimodePostTerminalCleanupReceipt,
    b3: object | None,
    c3: object | None,
    d3: object | None,
) -> dict[str, object]:
    if type(frozen_publication) is not LayerMultimodeFrozenPublicationEvidence:
        raise TypeError("report requires scalar frozen publication evidence")
    bundle = frozen_publication.bundle_receipt
    bundle_state = {
        "bundle_sha256": getattr(bundle, "bundle_sha256", None),
        "byte_count": getattr(bundle, "byte_count", None),
        "manifest_sha256": getattr(bundle, "manifest_sha256", None),
        "compilation_receipt_sha256": getattr(
            bundle,
            "compilation_receipt_sha256",
            None,
        ),
    }
    for name in (
        "bundle_sha256",
        "manifest_sha256",
        "compilation_receipt_sha256",
    ):
        _require_sha256(bundle_state[name], label=f"bundle receipt {name}")
    if (
        type(bundle_state["byte_count"]) is not int
        or bundle_state["byte_count"] <= 0
    ):
        raise ValueError("bundle receipt byte_count is invalid")
    if bundle_state["bundle_sha256"] != frozen_publication.bundle_file_sha256:
        raise ValueError("frozen bundle bytes differ from compiler receipt")
    return {
        "terminal_status": terminal_status,
        "preflight_receipt": preflight.state_dict(),
        "A_materialization_receipt": _state_dict(
            materialization_inventory,
            label="A materialization inventory",
        ),
        "selection_receipt": _state_dict(selection, label="selection"),
        "compilation_receipt": _state_dict(
            frozen_publication.compilation_receipt,
            label="compilation",
        ),
        "bundle_receipt": bundle_state,
        "development_teardown_receipt": development_teardown.state_dict(),
        "experiment_counters_receipt": experiment_counters.state_dict(),
        "fresh_panel_execution_receipt": fresh_panel_execution.state_dict(),
        "runtime_execution_receipt": runtime_execution.state_dict(),
        "post_terminal_cleanup_receipt": post_terminal_cleanup.state_dict(),
        "B3_evaluation": (
            None if b3 is None else _state_dict(b3, label="B3 evaluation")
        ),
        "C3_evaluation": (
            None if c3 is None else _state_dict(c3, label="C3 evaluation")
        ),
        "D3_evaluation": (
            None if d3 is None else _state_dict(d3, label="D3 evaluation")
        ),
        "reserve_disclosure": {
            "role": RESERVE_ROLE,
            "semantically_opened": False,
            "capture_receipt_sha256": None,
            "evaluation_receipt_sha256": None,
        },
    }


def _publish(
    publisher: object,
    *,
    payload: Mapping[str, object],
    bundle_source_path: Path,
    report_path: Path,
    bundle_path: Path,
    summary_path: Path,
) -> object:
    seal = getattr(publisher, "seal_layer_multimode_experiment_report", None)
    publish = getattr(
        publisher,
        "publish_layer_multimode_experiment_outputs",
        None,
    )
    if not callable(seal) or not callable(publish):
        raise TypeError("publisher lacks atomic multimode publication")
    report = seal(payload)
    return publish(
        report=report,
        bundle_source_path=bundle_source_path,
        report_path=report_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
    )


def _rollback(publisher: object, publication: object) -> None:
    method = getattr(publisher, "rollback_published_layer_multimode_experiment", None)
    if not callable(method):
        raise TypeError("publisher lacks multimode rollback")
    method(publication)


def _finalize(publisher: object, publication: object) -> None:
    method = getattr(publisher, "finalize_published_layer_multimode_experiment", None)
    if not callable(method):
        raise TypeError("publisher lacks multimode finalization")
    method(publication)


def run_layer_multimode_experiment(
    *,
    authority: object,
    runtime_factory: Callable[
        [LayerMultimodeExecutionPreflightReceipt], LayerMultimodeV3Runtime
    ],
    panel_source_factory: Callable[
        [LayerMultimodeExecutionPreflightReceipt, str], object
    ],
    publisher: object,
    workspace_parent: Path | str,
    report_path: Path | str,
    bundle_path: Path | str,
    summary_path: Path | str,
    _production_entry_token: object | None = None,
) -> LayerMultimodeExperimentRunResult:
    """Run the exact A -> B3 -> C3 -> D3 state machine."""

    strict_preflight = getattr(authority, "strict_preflight", None)
    revalidate = getattr(authority, "revalidate", None)
    if not callable(strict_preflight) or not callable(revalidate):
        raise TypeError("authority must provide strict_preflight and revalidate")
    preflight = strict_preflight()
    if type(preflight) is not LayerMultimodeExecutionPreflightReceipt:
        raise TypeError("authority returned the wrong preflight receipt type")
    if not preflight.live_preflight_authorized:
        raise RuntimeError("preflight is not the original live authority result")
    revalidate(preflight)
    if not preflight.test_only_nonproduction:
        from . import gemma3_layer_multimode_report as canonical_report

        if (
            _production_entry_token is not _PRODUCTION_RUN_ENTRY
            or type(authority) is not StaticLayerMultimodeExperimentAuthority
            or getattr(runtime_factory, "__self__", None) is not authority
            or getattr(runtime_factory, "__func__", None)
            is not StaticLayerMultimodeExperimentAuthority.build_runtime
            or getattr(panel_source_factory, "__self__", None) is not authority
            or getattr(panel_source_factory, "__func__", None)
            is not StaticLayerMultimodeExperimentAuthority.build_panel_source
            or publisher is not canonical_report
        ):
            raise RuntimeError(
                "production execution forbids injected runtime, panel, or publisher"
            )

    parent = Path(workspace_parent).resolve()
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("workspace_parent must be a real directory")
    report_destination = Path(report_path).resolve()
    bundle_destination = Path(bundle_path).resolve()
    summary_destination = Path(summary_path).resolve()
    if len({report_destination, bundle_destination, summary_destination}) != 3:
        raise ValueError("report, bundle, and summary paths must differ")
    for destination in (
        report_destination,
        bundle_destination,
        summary_destination,
    ):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)

    if not preflight.test_only_nonproduction:
        revalidate_layer_multimode_resources(
            preflight.resource_preflight_receipt,
            workspace_parent=parent,
            report_path=report_destination,
            bundle_path=bundle_destination,
            summary_path=summary_destination,
            workspace_must_exist=False,
        )

    workspace = parent / "layer-multimode-experiment-v2.work"
    workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
    runtime: LayerMultimodeV3Runtime | None = None
    publication: object | None = None
    published = False
    counters = _Counters()
    try:
        runtime = runtime_factory(preflight)
        if not isinstance(runtime, LayerMultimodeV3Runtime):
            raise TypeError("runtime_factory must return a v3 runtime")
        if (
            _require_exact_bool(
                runtime.test_only_nonproduction,
                label="runtime.test_only_nonproduction",
            )
            is not preflight.test_only_nonproduction
        ):
            raise RuntimeError("preflight/runtime production markers differ")

        recapture = runtime.recapture_exposed_predecessor_A(
            preflight=preflight,
            workspace=workspace,
        )
        if type(recapture) is not ADevelopmentRecapture:
            raise TypeError("runtime returned the wrong A recapture type")
        if (
            recapture.source_model_sha256 != preflight.source_model_sha256
            or recapture.predecessor_capture_receipt_sha256
            != preflight.predecessor_A_capture_receipt_sha256
        ):
            raise RuntimeError("A recapture lineage differs from preflight")
        if type(recapture.predecessor_A_runtime_receipt) is not (
            LayerMultimodePredecessorARuntimeReceipt
        ):
            raise TypeError("runtime returned the wrong predecessor A ledger type")
        predecessor_A_runtime_receipt = recapture.predecessor_A_runtime_receipt
        predecessor_A_comparison_receipt = (
            recapture.predecessor_lineage_comparison_receipt
        )
        predecessor_A_comparison_state = _predecessor_A_comparison_state(
            predecessor_A_comparison_receipt
        )
        if (
            predecessor_A_runtime_receipt.source_capture_receipt_sha256
            != recapture.source_capture_receipt_sha256
            or predecessor_A_comparison_state[
                "predecessor_capture_receipt_sha256"
            ]
            != recapture.predecessor_capture_receipt_sha256
            or predecessor_A_comparison_state[
                "recaptured_capture_receipt_sha256"
            ]
            != recapture.source_capture_receipt_sha256
            or predecessor_A_comparison_state[
                "recaptured_runtime_session_binding_sha256"
            ]
            != predecessor_A_runtime_receipt.runtime_session_binding_sha256
        ):
            raise RuntimeError("A recapture nested evidence differs from capture")

        materializations: list[LayerMaterialization] = []
        for layer_ordinal in range(LAYER_COUNT):
            materialized = runtime.materialize_A_layer(
                recapture,
                layer_ordinal,
            )
            if (
                type(materialized) is not LayerMaterialization
                or materialized.layer_ordinal != layer_ordinal
            ):
                raise RuntimeError("A materialization layer order drifted")
            materializations.append(materialized)
            counters.materialization_count += 1
        materialization_tuple = tuple(materializations)
        materialization_inventory = runtime.build_materialization_inventory(
            recapture,
            materialization_tuple,
        )

        atlas_attempts_by_key: dict[
            tuple[int, int, int], LayerMultimodePreoptimizerAttempt
        ] = {}
        fits: list[object | None] = []
        splits: list[object | None] = []
        evaluations: list[object] = []
        for materialized in materialization_tuple:
            for chart_count in CHART_COUNT_CANDIDATES:
                for mode_count in MODE_COUNT_CANDIDATES:
                    for tangent_rank in TANGENT_RANK_CANDIDATES:
                        atlas_key = (
                            materialized.layer_ordinal,
                            chart_count,
                            tangent_rank,
                        )
                        atlas_attempt = atlas_attempts_by_key.get(atlas_key)
                        if atlas_attempt is None:
                            atlas_attempt = runtime.fit_chart_atlas(
                                materialized,
                                chart_count=chart_count,
                                tangent_rank=tangent_rank,
                            )
                            if type(atlas_attempt) is not (
                                LayerMultimodePreoptimizerAttempt
                            ):
                                raise TypeError(
                                    "runtime returned the wrong atlas-attempt type"
                                )
                            atlas_attempts_by_key[atlas_key] = atlas_attempt
                            counters.atlas_fit_attempt_count += 1
                            if atlas_attempt.support_rejected:
                                counters.atlas_support_rejection_count += 1
                            else:
                                counters.atlas_fit_success_count += 1

                        fit: object | None = None
                        split: object | None = None
                        split_attempt: LayerMultimodePreoptimizerAttempt | None = None
                        if atlas_attempt.support_rejected:
                            support_error = atlas_attempt.support_error
                            if support_error is None:
                                raise AssertionError(
                                    "rejected atlas attempt lacks support error"
                                )
                            evaluation = runtime.lower_candidate_support_rejection(
                                materialization=materialized,
                                chart_count=chart_count,
                                mode_count=mode_count,
                                tangent_rank=tangent_rank,
                                candidate_fit=None,
                                support_error=support_error,
                                recapture=recapture,
                            )
                            counters.candidate_support_rejection_count += 1
                        else:
                            if atlas_attempt.result is None:
                                raise AssertionError(
                                    "successful atlas attempt lacks an atlas"
                                )
                            fit = runtime.fit_candidate(
                                materialized,
                                chart_count=chart_count,
                                mode_count=mode_count,
                                tangent_rank=tangent_rank,
                                atlas=atlas_attempt.result,
                            )
                            counters.candidate_fit_count += 1
                            split_rejected = False
                            if mode_count > 1:
                                split_attempt = runtime.evaluate_split_refit(
                                    materialized,
                                    chart_count=chart_count,
                                    mode_count=mode_count,
                                    tangent_rank=tangent_rank,
                                )
                                if type(split_attempt) is not (
                                    LayerMultimodePreoptimizerAttempt
                                ):
                                    raise TypeError(
                                        "runtime returned the wrong split-attempt type"
                                    )
                                counters.split_refit_attempt_count += 1
                                if split_attempt.support_rejected:
                                    support_error = split_attempt.support_error
                                    if support_error is None:
                                        raise AssertionError(
                                            "rejected split attempt lacks support error"
                                        )
                                    evaluation = (
                                        runtime.lower_candidate_support_rejection(
                                            materialization=materialized,
                                            chart_count=chart_count,
                                            mode_count=mode_count,
                                            tangent_rank=tangent_rank,
                                            candidate_fit=fit,
                                            support_error=support_error,
                                            recapture=recapture,
                                        )
                                    )
                                    counters.split_refit_support_rejection_count += 1
                                    counters.candidate_support_rejection_count += 1
                                    split_rejected = True
                                else:
                                    split = split_attempt.result
                                    if split is None:
                                        raise AssertionError(
                                            "successful split attempt lacks evidence"
                                        )
                                    counters.split_refit_count += 1
                            if not split_rejected:
                                evaluation = runtime.lower_candidate_evaluation(
                                    candidate_fit=fit,
                                    split_refit=split,
                                    recapture=recapture,
                                )
                                counters.candidate_evaluation_receipt_count += 1
                        expected_key = (
                            materialized.layer_ordinal,
                            chart_count,
                            mode_count,
                            tangent_rank,
                        )
                        if _candidate_key(evaluation) != expected_key:
                            raise RuntimeError(
                                "candidate evaluation grid order drifted"
                            )
                        fits.append(fit)
                        splits.append(split)
                        evaluations.append(evaluation)
                        counters.candidate_inventory_count += 1
        if (
            len(atlas_attempts_by_key) != ATLAS_FIT_COUNT
            or len(evaluations) != TOTAL_CANDIDATE_COUNT
        ):
            raise RuntimeError("A fit did not cover the exact cached full grid")

        evaluation_tuple = tuple(evaluations)
        counters.selection_attempt_count += 1
        no_selection: LayerMultimodeNoAdmissibleSelectionReceipt | None = None
        selection: object | None = None
        frozen: LayerMultimodeFrozenCandidate | None = None
        teardown: LayerMultimodeDevelopmentTeardownReceipt | None = None
        b3: object | None = None
        c3: object | None = None
        d3: object | None = None
        fresh_role_captures: list[LayerMultimodeFreshRoleCaptureReceipt] = []
        from .gemma3_layer_multimode_selection import (
            NoAdmissibleLayerMultimodeCandidateError,
        )

        try:
            selection = runtime.select_full_grid(evaluation_tuple)
        except NoAdmissibleLayerMultimodeCandidateError as selection_error:
            no_selection = _build_no_selection_receipt(
                evaluation_tuple,
                selection_error,
            )
            prepared_release = runtime.release_development_state(
                recapture=recapture,
                materializations=materialization_tuple,
                atlases=tuple(
                    attempt.result
                    for attempt in atlas_attempts_by_key.values()
                    if attempt.result is not None
                ),
                candidate_fits=tuple(fits),
                split_refits=tuple(splits),
                compiled=None,
            )
            materializations.clear()
            atlas_attempts_by_key.clear()
            fits.clear()
            splits.clear()
            evaluations.clear()
            del (
                recapture,
                materialization_tuple,
                materialization_inventory,
                evaluation_tuple,
                materialized,
                atlas_attempt,
                fit,
                split,
                split_attempt,
                evaluation,
            )
            gc.collect()
            teardown = runtime.confirm_development_teardown(prepared_release)
            if type(teardown) is not LayerMultimodeDevelopmentTeardownReceipt:
                raise TypeError("runtime returned the wrong teardown receipt type")
            counters.development_teardown_count += 1
            terminal_status = TERMINAL_A_NO_CANDIDATE
            outcome: TerminalOutcome = NoAdmissibleCandidateOutcome(
                no_selection_receipt=no_selection,
                counters=counters.freeze(
                    terminal_status,
                    execution_preflight_receipt_sha256=(
                        preflight.receipt_sha256
                    ),
                    selection_receipt_sha256=no_selection.receipt_sha256,
                ),
            )
            cleanup = runtime.close(
                terminal_status=terminal_status,
                runtime_execution_receipt_sha256=None,
                runtime_session_binding_sha256=None,
                fresh_panel_execution_receipt_sha256=None,
            )
            if type(cleanup) is not LayerMultimodePostTerminalCleanupReceipt:
                raise TypeError("runtime returned the wrong cleanup receipt type")
            runtime = None
            shutil.rmtree(workspace)
            return LayerMultimodeExperimentRunResult(
                outcome=outcome,
                publication=None,
                workspace_removed=True,
                runtime_closed=True,
            )
        else:
            selected_fits, selected_splits = _selected_objects(
                selection,
                evaluation_tuple,
                tuple(fits),
                tuple(splits),
            )
            selected_inventory = runtime.build_selected_inventory(
                recapture=recapture,
                materialization_inventory=materialization_inventory,
                selection=selection,
                selected_fits=selected_fits,
                selected_splits=selected_splits,
            )
            compiled = runtime.compile_source_free(
                source_model=recapture.source_model,
                selected_inventory=selected_inventory,
            )
            counters.compilation_count += 1
            frozen = runtime.freeze_compiled_candidate(
                compiled=compiled,
                selection=selection,
                workspace=workspace,
            )
            if type(frozen) is not LayerMultimodeFrozenCandidate:
                raise TypeError("runtime returned the wrong frozen candidate type")
            counters.candidate_freeze_count += 1
            prepared_release = runtime.release_development_state(
                recapture=recapture,
                materializations=materialization_tuple,
                atlases=tuple(
                    attempt.result
                    for attempt in atlas_attempts_by_key.values()
                    if attempt.result is not None
                ),
                candidate_fits=tuple(fits),
                split_refits=tuple(splits),
                compiled=compiled,
            )
            materializations.clear()
            atlas_attempts_by_key.clear()
            fits.clear()
            splits.clear()
            evaluations.clear()
            del (
                recapture,
                materialization_tuple,
                evaluation_tuple,
                materialized,
                atlas_attempt,
                fit,
                split,
                split_attempt,
                evaluation,
                selected_fits,
                selected_splits,
                selected_inventory,
                compiled,
            )
            gc.collect()
            teardown = runtime.confirm_development_teardown(prepared_release)
            if type(teardown) is not LayerMultimodeDevelopmentTeardownReceipt:
                raise TypeError("runtime returned the wrong teardown receipt type")
            counters.development_teardown_count += 1

            selection_sha = _artifact_sha256(selection, label="selection")
            frozen_candidate_identity_sha = _artifact_sha256(
                frozen.candidate_identity,
                label="frozen candidate identity",
            )
            revalidate(preflight)
            panel_source = panel_source_factory(preflight, selection_sha)
            revalidate(preflight)
            b3_panel = _panel_open(panel_source, B3_ROLE)
            counters.B3_open_count += 1
            b3_capture = _require_fresh_role_capture(
                runtime.authenticate_fresh_panel_tokenization(
                    panel_source=panel_source,
                    panel=b3_panel,
                    role=B3_ROLE,
                    selection_receipt_sha256=selection_sha,
                    frozen_candidate_identity_sha256=(
                        frozen_candidate_identity_sha
                    ),
                ),
                role=B3_ROLE,
                preflight=preflight,
                selection_receipt_sha256=selection_sha,
                frozen_candidate_identity_sha256=frozen_candidate_identity_sha,
            )
            fresh_role_captures.append(b3_capture)
            b3 = runtime.evaluate_b3_teacher_forced(
                panel=b3_panel,
                candidate=frozen,
                fresh_role_capture_receipt=b3_capture,
            )
            counters.B3_evaluation_count += 1
            b3_passed = _require_passed(b3, label="B3 evaluation")
            _panel_freeze(
                panel_source,
                role=B3_ROLE,
                receipt_sha256=_artifact_sha256(b3, label="B3 evaluation"),
                terminal_passed=b3_passed,
            )
            if not b3_passed:
                terminal_status = TERMINAL_B3_FAILED
                outcome = B3FidelityFailureOutcome(
                    b3_evaluation=b3,
                    counters=counters.freeze(
                        terminal_status,
                        execution_preflight_receipt_sha256=(
                            preflight.receipt_sha256
                        ),
                        selection_receipt_sha256=selection_sha,
                    ),
                )
            else:
                revalidate(preflight)
                c3_panel = _panel_open(panel_source, C3_ROLE)
                counters.C3_open_count += 1
                c3_capture = _require_fresh_role_capture(
                    runtime.authenticate_fresh_panel_tokenization(
                        panel_source=panel_source,
                        panel=c3_panel,
                        role=C3_ROLE,
                        selection_receipt_sha256=selection_sha,
                        frozen_candidate_identity_sha256=(
                            frozen_candidate_identity_sha
                        ),
                    ),
                    role=C3_ROLE,
                    preflight=preflight,
                    selection_receipt_sha256=selection_sha,
                    frozen_candidate_identity_sha256=(
                        frozen_candidate_identity_sha
                    ),
                )
                fresh_role_captures.append(c3_capture)
                c3 = runtime.evaluate_c3_full_stack(
                    panel=c3_panel,
                    candidate=frozen,
                    b3_evaluation=b3,
                    fresh_role_capture_receipt=c3_capture,
                )
                counters.C3_evaluation_count += 1
                c3_passed = _require_passed(c3, label="C3 evaluation")
                _panel_freeze(
                    panel_source,
                    role=C3_ROLE,
                    receipt_sha256=_artifact_sha256(c3, label="C3 evaluation"),
                    terminal_passed=c3_passed,
                )
                if not c3_passed:
                    terminal_status = TERMINAL_C3_FAILED
                    outcome = C3FidelityFailureOutcome(
                        b3_evaluation=b3,
                        c3_evaluation=c3,
                        counters=counters.freeze(
                            terminal_status,
                            execution_preflight_receipt_sha256=(
                                preflight.receipt_sha256
                            ),
                            selection_receipt_sha256=selection_sha,
                        ),
                    )
                else:
                    revalidate(preflight)
                    d3_panel = _panel_open(panel_source, D3_ROLE)
                    counters.D3_open_count += 1
                    d3_capture = _require_fresh_role_capture(
                        runtime.authenticate_fresh_panel_tokenization(
                            panel_source=panel_source,
                            panel=d3_panel,
                            role=D3_ROLE,
                            selection_receipt_sha256=selection_sha,
                            frozen_candidate_identity_sha256=(
                                frozen_candidate_identity_sha
                            ),
                        ),
                        role=D3_ROLE,
                        preflight=preflight,
                        selection_receipt_sha256=selection_sha,
                        frozen_candidate_identity_sha256=(
                            frozen_candidate_identity_sha
                        ),
                    )
                    fresh_role_captures.append(d3_capture)
                    d3 = runtime.evaluate_d3_final(
                        panel=d3_panel,
                        candidate=frozen,
                        c3_evaluation=c3,
                        fresh_role_capture_receipt=d3_capture,
                    )
                    counters.D3_evaluation_count += 1
                    _panel_freeze(
                        panel_source,
                        role=D3_ROLE,
                        receipt_sha256=_artifact_sha256(
                            d3,
                            label="D3 evaluation",
                        ),
                        terminal_passed=_require_passed(
                            d3,
                            label="D3 evaluation",
                        ),
                    )
                    terminal_status = TERMINAL_D3_COMPLETED
                    outcome = D3CompletedOutcome(
                        b3_evaluation=b3,
                        c3_evaluation=c3,
                        d3_evaluation=d3,
                        counters=counters.freeze(
                            terminal_status,
                            execution_preflight_receipt_sha256=(
                                preflight.receipt_sha256
                            ),
                            selection_receipt_sha256=selection_sha,
                        ),
                    )

            _assert_panel_closure(panel_source, outcome.terminal_status)

        fresh_panel_execution = _build_fresh_panel_execution_receipt(
            panel_source=panel_source,
            preflight=preflight,
            selection_receipt_sha256=selection_sha,
            terminal_status=outcome.terminal_status,
        )
        runtime_execution = runtime.build_runtime_execution_receipt(
            terminal_status=outcome.terminal_status,
        )
        if type(runtime_execution) is not LayerMultimodeRuntimeExecutionReceipt:
            raise TypeError("runtime returned the wrong execution receipt type")
        if (
            runtime_execution.implementation_qualification_sha256
            != preflight.implementation_qualification_sha256
            or runtime_execution.predecessor_A_capture_receipt_sha256
            != preflight.predecessor_A_capture_receipt_sha256
            or runtime_execution.predecessor_runtime_execution_receipt_sha256
            != preflight.predecessor_runtime_execution_receipt_sha256
            or runtime_execution.source_model_fingerprint_before_sha256
            != preflight.source_model_sha256
            or runtime_execution.source_model_fingerprint_after_sha256
            != preflight.source_model_sha256
            or runtime_execution.predecessor_A_runtime_receipt
            is not predecessor_A_runtime_receipt
            or runtime_execution.predecessor_A_lineage_comparison_receipt
            is not predecessor_A_comparison_receipt
            or predecessor_A_runtime_receipt.execution_preflight_receipt_sha256
            != preflight.receipt_sha256
            or predecessor_A_runtime_receipt.source_model_fingerprint_before_sha256
            != preflight.source_model_sha256
            or predecessor_A_runtime_receipt.source_model_fingerprint_after_sha256
            != preflight.source_model_sha256
            or runtime_execution.predecessor_A_recaptured_capture_receipt_sha256
            != predecessor_A_runtime_receipt.source_capture_receipt_sha256
            or len(runtime_execution.fresh_role_capture_receipts)
            != len(fresh_role_captures)
            or any(
                observed is not expected
                for observed, expected in zip(
                    runtime_execution.fresh_role_capture_receipts,
                    fresh_role_captures,
                    strict=True,
                )
            )
            or any(
                value.panel_manifest_sha256 != preflight.panel_manifest_sha256
                or value.panel_manifest_file_sha256
                != preflight.panel_manifest_file_sha256
                or value.panel_receipt_sha256 != preflight.panel_receipt_sha256
                or value.panel_receipt_file_sha256
                != preflight.panel_receipt_file_sha256
                or value.panel_bundle_sha256 != preflight.panel_bundle_sha256
                or value.selection_receipt_sha256 != selection_sha
                or value.frozen_candidate_identity_sha256
                != frozen_candidate_identity_sha
                for value in runtime_execution.fresh_role_capture_receipts
            )
        ):
            raise RuntimeError("runtime execution identity differs from preflight")
        frozen_publication = _frozen_publication_evidence(frozen)
        del frozen
        gc.collect()
        post_terminal_cleanup = runtime.close(
            terminal_status=outcome.terminal_status,
            runtime_execution_receipt_sha256=runtime_execution.receipt_sha256,
            runtime_session_binding_sha256=(
                runtime_execution.runtime_session_binding_sha256
            ),
            fresh_panel_execution_receipt_sha256=(
                fresh_panel_execution.receipt_sha256
            ),
        )
        if type(post_terminal_cleanup) is not (
            LayerMultimodePostTerminalCleanupReceipt
        ):
            raise TypeError("runtime returned the wrong cleanup receipt type")
        runtime = None
        if not preflight.test_only_nonproduction:
            revalidate_layer_multimode_resources(
                preflight.resource_preflight_receipt,
                workspace_parent=parent,
                report_path=report_destination,
                bundle_path=bundle_destination,
                summary_path=summary_destination,
                workspace_must_exist=True,
            )
        payload = _report_payload(
            preflight=preflight,
            terminal_status=outcome.terminal_status,
            materialization_inventory=materialization_inventory,
            selection=selection,
            frozen_publication=frozen_publication,
            development_teardown=teardown,
            experiment_counters=outcome.counters,
            fresh_panel_execution=fresh_panel_execution,
            runtime_execution=runtime_execution,
            post_terminal_cleanup=post_terminal_cleanup,
            b3=b3,
            c3=c3,
            d3=d3,
        )
        revalidate(preflight)
        publication = _publish(
            publisher,
            payload=payload,
            bundle_source_path=Path(frozen_publication.bundle_path),
            report_path=report_destination,
            bundle_path=bundle_destination,
            summary_path=summary_destination,
        )
        published = True
        shutil.rmtree(workspace)
        _finalize(publisher, publication)
        published = False
        return LayerMultimodeExperimentRunResult(
            outcome=outcome,
            publication=publication,
            workspace_removed=True,
            runtime_closed=True,
        )
    except BaseException as primary_error:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if published and publication is not None:
            try:
                _rollback(publisher, publication)
            except BaseException as cleanup_error:
                cleanup_errors.append(("published output rollback", cleanup_error))
        if runtime is not None:
            try:
                runtime.abort()
            except BaseException as cleanup_error:
                cleanup_errors.append(("runtime abort", cleanup_error))
        try:
            if workspace.exists() and not workspace.is_symlink():
                shutil.rmtree(workspace)
        except BaseException as cleanup_error:
            cleanup_errors.append(("workspace cleanup", cleanup_error))
        for label, cleanup_error in cleanup_errors:
            primary_error.add_note(
                f"{label} also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def run_production_layer_multimode_experiment(
    *,
    authority: StaticLayerMultimodeExperimentAuthority,
) -> LayerMultimodeExperimentRunResult:
    """Run production only through the canonical one-shot authority."""

    if type(authority) is not StaticLayerMultimodeExperimentAuthority:
        raise TypeError("production requires the exact static authority")
    from . import gemma3_layer_multimode_report as report

    return run_layer_multimode_experiment(
        authority=authority,
        runtime_factory=authority.build_runtime,
        panel_source_factory=authority.build_panel_source,
        publisher=report,
        workspace_parent=authority.workspace_parent,
        report_path=authority.report_path,
        bundle_path=authority.bundle_path,
        summary_path=authority.summary_path,
        _production_entry_token=_PRODUCTION_RUN_ENTRY,
    )


def _cli_run_result_state(
    result: LayerMultimodeExperimentRunResult,
    *,
    preflight: LayerMultimodeExecutionPreflightReceipt,
) -> tuple[dict[str, object], bool]:
    outcome = result.outcome
    passed = type(outcome) is D3CompletedOutcome and _require_passed(
        outcome.d3_evaluation,
        label="D3 evaluation",
    )
    publication_sha256 = None
    if result.publication is not None:
        publication_sha256 = getattr(
            result.publication,
            "publication_receipt_sha256",
            None,
        )
        _require_sha256(
            publication_sha256,
            label="publication receipt_sha256",
        )
    return (
        {
            "terminal_status": outcome.terminal_status,
            "passed": passed,
            "execution_preflight_receipt_sha256": preflight.receipt_sha256,
            "experiment_counters_receipt_sha256": (
                outcome.counters.receipt_sha256
            ),
            "publication_receipt_sha256": publication_sha256,
            "workspace_removed": result.workspace_removed,
            "runtime_closed": result.runtime_closed,
            "reserve_unopened": result.reserve_unopened,
        },
        passed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run strict model-free preflight or one explicit production v3 run."""

    from . import gemma3_layer_multimode_setup as setup

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate all frozen inputs and print the canonical receipt",
    )
    action.add_argument(
        "--run-v3",
        action="store_true",
        help="perform exactly one production A -> B3 -> C3 -> D3 run",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--setup-path", type=Path, default=None)
    parser.add_argument("--workspace-parent", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--bundle-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.repository_root.resolve(strict=True)
    authority = StaticLayerMultimodeExperimentAuthority(
        repository_root=root,
        setup_path=(
            root / setup.DEFAULT_OUTPUT
            if args.setup_path is None
            else args.setup_path
        ),
        workspace_parent=args.workspace_parent,
        report_path=args.report_path,
        bundle_path=args.bundle_path,
        summary_path=args.summary_path,
        cache_dir=args.cache_dir,
    )
    if args.preflight_only:
        receipt = authority.strict_preflight()
        print(_canonical_json(receipt.state_dict()).decode("utf-8"))
        return 0

    result = run_production_layer_multimode_experiment(authority=authority)
    preflight = authority._receipt
    if preflight is None:
        raise AssertionError("production run completed without a preflight")
    state, passed = _cli_run_result_state(result, preflight=preflight)
    print(_canonical_json(state).decode("utf-8"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
