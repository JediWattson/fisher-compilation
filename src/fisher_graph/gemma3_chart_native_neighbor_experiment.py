"""Authenticated A/B/C/D worker for chart-native Gemma mode neighbors.

This module itself has only standard-library and inert setup imports; normal
package import eagerly executes the root package before this module and may
initialize its tensor backend.  That eager import is recorded but lies outside
the pre-execution authentication boundary.  The frozen setup, conservative
first-party functional dependency closure, prompt-free fresh-panel
manifest/receipt, and opaque bytes of all five role corpora are authenticated
before Transformers, the tokenizer, or model weights may be touched by a
runtime factory.  This is a scientific reproducibility boundary, not a claim
that the whole Python process or every imported executable is tamper-proof.
Semantic access is delegated to the authoritative ``SequentialPanelSource``
state machine; this module defines no second panel manifest.

The live path is intentionally fail-closed until the final setup artifact pins
this worker and its five fresh corpora.  Model-free tests may exercise the
same orchestration with an explicitly marked small-width panel source/runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import html
import importlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import xml.etree.ElementTree as ET

from .gemma3_chart_native_neighbor_setup import (
    PINNED_SETUP_FILE_SHA256,
    PINNED_SETUP_SHA256,
    PINNED_WORKER_IMPLEMENTATION_SOURCES as _SETUP_IMPLEMENTATION_SOURCES,
)

# ``python -m`` executes this file as ``__main__``.  The concrete runtime
# imports the canonical package name, so alias it before any dataclass is
# defined to keep receipt type identity singular across the lazy CLI boundary.
if __name__ == "__main__":
    sys.modules.setdefault(
        "fisher_graph.gemma3_chart_native_neighbor_experiment",
        sys.modules[__name__],
    )


EXPERIMENT_REPORT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.experiment_report.v1"
)
EXPERIMENT_REPORT_FORMAT_VERSION = 1
EXECUTION_PREFLIGHT_SCHEMA = (
    "fisher_graph.gemma3_chart_native_neighbor.execution_preflight.v1"
)

FIT_ROLE = "A_chart_transport_fit"
DISCOVERY_ROLE = "B_neighbor_discovery"
CONFIRMATION_ROLE = "C_neighbor_confirmation"
RESPONSE_ROLE = "D_response_validation"
RESERVE_ROLE = "reserve_unopened"
ORDERED_ROLES = (
    FIT_ROLE,
    DISCOVERY_ROLE,
    CONFIRMATION_ROLE,
    RESPONSE_ROLE,
    RESERVE_ROLE,
)

DEFAULT_REPORT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v1.json"
)
DEFAULT_SVG_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v1.svg"
)
DEFAULT_ATLAS_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "chart-native-neighbor-experiment-v1.atlas.bundle"
)
DEFAULT_SETUP_PATH = Path(
    "artifacts/research/gemma3-chart-native-neighbor-setup-v1.json"
)
DEFAULT_RUNTIME_QUALIFICATION_PATH = Path(
    "artifacts/research/"
    "gemma3-chart-native-neighbor-runtime-qualification-v1.json"
)

# These values come from the setup module by stable symbol.  Updating the
# final setup pins therefore never changes this worker's source bytes, while
# the setup artifact can authenticate both this worker and the runtime.
PINNED_SOURCE_SHA256S = {
    name: str(receipt["file_sha256"])
    for name, receipt in _SETUP_IMPLEMENTATION_SOURCES.items()
}
_REQUIRED_SOURCE_NAMES = frozenset(PINNED_SOURCE_SHA256S)
_FORBIDDEN_BEFORE_PREFLIGHT = (
    "transformers",
    "mlx",
    "mlx_lm",
    "fisher_graph.gemma3_chart_native_neighbor_runtime",
    "fisher_graph.gemma3_experiment",
)
_EXPECTED_PRODUCTION_SCHEDULE = (
    (96, 2),
    (511, 1),
    (512, 1),
    (513, 1),
    (514, 1),
    (576, 1),
    (640, 1),
    (768, 1),
)

_BINDING_DOMAIN = b"fisher_graph:chart-native-execution-binding:v1\0"
_EXECUTION_CONTRACT_DOMAIN = (
    b"fisher_graph:chart-native-execution-contract:v1\0"
)
_PREFLIGHT_DOMAIN = b"fisher_graph:chart-native-execution-preflight:v1\0"
_PANEL_BINDING_DOMAIN = b"fisher_graph:chart-native-role-panel:v1\0"
_CAPTURE_DOMAIN = b"fisher_graph:chart-native-role-capture:v1\0"
_CAPTURE_BATCH_AXIS_DOMAIN = (
    b"fisher_graph:chart-native-capture-batch-axis-catalog:v1\0"
)
_MODE_LINEAGE_DOMAIN = b"fisher_graph:chart-native-mode-lineage:v1\0"
_RUNTIME_QUALIFICATION_DOMAIN = (
    b"fisher_graph:chart-native-runtime:qualification:v1\0"
)
_RUNTIME_RECEIPT_DOMAIN = b"fisher_graph:chart-native-runtime-receipt:v1\0"
_FIRST_A_CANARY_DOMAIN = (
    b"fisher_graph:chart-native-first-A-capture-canary:v1\0"
)
_RUNTIME_PLACEMENT_DOMAIN = (
    b"fisher_graph:chart-native-runtime:placement:v1\0"
)
_RUNTIME_ADAPTER_DOMAIN = b"fisher_graph:chart-native-runtime:adapter:v1\0"
_RUNTIME_SESSION_DOMAIN = b"fisher_graph:chart-native-runtime:session:v1\0"
_ENERGY_VECTOR_DOMAIN = b"fisher_graph:chart-native-energy-vector:v1\0"
_EXAMPLE_ROW_DOMAIN = b"fisher_graph:chart-native-example-q-t-rows:v1\0"
_ROLE_EVALUATION_DOMAIN = b"fisher_graph:chart-native-role-evaluation:v1\0"
_TRANSPORT_CATALOG_DOMAIN = b"fisher_graph:chart-native-transport-catalog:v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph:chart-native-B-candidate-catalog:v1\0"
_GRAPH_DOMAIN = b"fisher_graph:chart-native-C-graph-catalog:v1\0"
_D_SKIP_DOMAIN = b"fisher_graph:chart-native-D-skip:v1\0"
_D_RESPONSE_DOMAIN = b"fisher_graph:chart-native-D-response:v1\0"
_REPORT_DOMAIN = b"fisher_graph:chart-native-report:v1\0"
_FROZEN_ANALYTIC_NLL_CONFIG_SHA256 = (
    "10f62c57a49d5d3250321b28ae939701d1064951986bdc89a1940178007a964e"
)

_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_SETUP_BYTES = 16 * 1024 * 1024
_MAX_RUNTIME_QUALIFICATION_BYTES = 4 * 1024 * 1024
_MAX_REPORT_BYTES = 512 * 1024 * 1024
_MAX_SVG_BYTES = 16 * 1024 * 1024
_MAX_ATLAS_BYTES = 64 * 1024 * 1024
_PRODUCTION_TRANSPORT_BYTES = 979_025_920
_PRODUCTION_TEMP_BYTES = 20 * 1024**3
_PRODUCTION_ATOMIC_HEADROOM_BYTES = 1 * 1024**3
_PRODUCTION_MINIMUM_FREE_BYTES = 24 * 1024**3
_PRODUCTION_MAXIMUM_RESIDENT_MEMORY_BYTES = 8 * 1024**3
_PRODUCTION_MINIMUM_PHYSICAL_MEMORY_BYTES = 12 * 1024**3

_FORBIDDEN_REPORT_FIELDS = frozenset(
    {
        "prompt",
        "prompts",
        "input_ids",
        "token_ids",
        "context_values",
        "comparison_values",
        "generalized_contributions",
        "carrier_values",
        "carrier_score_gradients",
        "score_gradients",
        "attention_probabilities",
        "repeated_values",
        "projected_values",
        "route_coordinates",
        "transport_coefficient",
        "coefficient",
    }
)
_STAGE_KIND_BY_ROLE = {
    FIT_ROLE: "chart_atlas_transport_fit",
    DISCOVERY_ROLE: "neighbor_candidate_catalog",
    CONFIRMATION_ROLE: "chart_neighbor_graph_and_gates",
    RESPONSE_ROLE: "response_validation",
}
_STAGE_DOMAIN = b"fisher_graph:chart-native-frozen-stage:v1\0"
_RESOURCE_DOMAIN = b"fisher_graph:chart-native-resource-observation:v1\0"
_PUBLICATION_DOMAIN = b"fisher_graph:chart-native-publication:v1\0"
_ELIGIBILITY_DOMAIN = b"fisher_graph:chart-native-role-eligibility:v1\0"
_CAPTURE_LEDGER_DOMAIN = b"fisher_graph:chart-native-capture-ledger:v1\0"

_PRIMARY_ELIGIBLE_LENGTHS = (96, 511, 512, 513, 576, 640, 768)
_HARD_LONG_LENGTHS = (576, 640, 768)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_loads(raw: bytes, *, require_canonical: bool) -> object:
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
        raise ValueError("strict JSON is invalid") from error
    if require_canonical and _canonical_json_bytes(value) != raw:
        raise ValueError("JSON bytes are not canonical")
    return value


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _read_regular(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is inaccessible") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise ValueError(f"{label} has an invalid byte size")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError(f"{label} changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{label} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed during authentication")
    return b"".join(chunks)


def _regular_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is inaccessible") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be an existing non-symlink directory")
    return info


def _relative_path(root: Path, value: str, *, label: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    absolute = Path(os.path.abspath(candidate))
    if absolute != candidate.resolve(strict=False):
        raise ValueError(f"{label} may not traverse symlinked path components")
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside repository_root") from error
    return absolute


@dataclass(frozen=True, slots=True)
class ExecutionSourcePin:
    name: str
    path: str
    file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("source pin name must be nonempty")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("source pin path must be nonempty")
        _require_sha256(self.file_sha256, label="source pin file_sha256")

    def payload(self) -> dict[str, str]:
        return {
            "file_sha256": self.file_sha256,
            "name": self.name,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class LengthBucketBatchSchedule:
    buckets: tuple[tuple[int, int], ...] = _EXPECTED_PRODUCTION_SCHEDULE

    def __post_init__(self) -> None:
        if not isinstance(self.buckets, tuple) or not self.buckets:
            raise TypeError("batch schedule buckets must be a nonempty tuple")
        lengths: list[int] = []
        for length, batch_size in self.buckets:
            _require_positive_int(length, label="schedule token length")
            _require_positive_int(batch_size, label="schedule batch size")
            lengths.append(length)
        if lengths != sorted(set(lengths)):
            raise ValueError("batch schedule token lengths must be sorted unique")

    def payload(self) -> list[list[int]]:
        return [[length, batch] for length, batch in self.buckets]

    @property
    def sha256(self) -> str:
        return _digest(
            b"fisher_graph:chart-native-batch-schedule:v1\0",
            self.payload(),
        )

    def batch_size_for(self, token_length: int) -> int:
        try:
            return dict(self.buckets)[token_length]
        except KeyError as error:
            raise ValueError("token length is outside frozen schedule") from error


@dataclass(frozen=True, slots=True)
class CaptureCallLedger:
    schedule_sha256: str
    analytic_nll_vjp_sha256: str
    per_role_length_batch_counts: tuple[tuple[int, int], ...]
    chart_capture_roles: tuple[str, ...]
    chart_batches_per_role: int
    total_chart_forward_count: int
    total_summed_shifted_nll_backward_count: int
    gradient_target_count_per_backward: int
    canonical_carrier_site_gradient_count: int
    auxiliary_gradient_count: int
    D_chart_capture_count: int
    ledger_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.schedule_sha256, label="schedule_sha256")
        _require_sha256(
            self.analytic_nll_vjp_sha256,
            label="analytic_nll_vjp_sha256",
        )
        if (
            self.chart_capture_roles
            != (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE)
            or self.chart_batches_per_role != 45
            or self.total_chart_forward_count != 135
            or self.total_summed_shifted_nll_backward_count != 135
            or self.gradient_target_count_per_backward != 92
            or self.canonical_carrier_site_gradient_count != 73
            or self.auxiliary_gradient_count != 19
            or self.D_chart_capture_count != 0
        ):
            raise ValueError("capture call ledger drifted")
        _require_sha256(self.ledger_sha256, label="ledger_sha256")
        if self.ledger_sha256 != _digest(
            _CAPTURE_LEDGER_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("capture call ledger digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "D_chart_capture_count": self.D_chart_capture_count,
            "analytic_nll_vjp_sha256": self.analytic_nll_vjp_sha256,
            "auxiliary_gradient_count": self.auxiliary_gradient_count,
            "canonical_carrier_site_gradient_count": (
                self.canonical_carrier_site_gradient_count
            ),
            "chart_batches_per_role": self.chart_batches_per_role,
            "chart_capture_roles": list(self.chart_capture_roles),
            "gradient_target_count_per_backward": (
                self.gradient_target_count_per_backward
            ),
            "one_backbone_forward_per_chart_batch": True,
            "one_summed_shifted_nll_backward_per_chart_batch": True,
            "per_role_length_batch_counts": [
                [length, count]
                for length, count in self.per_role_length_batch_counts
            ],
            "schedule_sha256": self.schedule_sha256,
            "total_chart_forward_count": self.total_chart_forward_count,
            "total_summed_shifted_nll_backward_count": (
                self.total_summed_shifted_nll_backward_count
            ),
        }


def build_capture_call_ledger(
    *,
    schedule: LengthBucketBatchSchedule = LengthBucketBatchSchedule(),
    analytic_nll_vjp_sha256: str = _FROZEN_ANALYTIC_NLL_CONFIG_SHA256,
) -> CaptureCallLedger:
    _require_sha256(
        analytic_nll_vjp_sha256,
        label="analytic_nll_vjp_sha256",
    )
    examples_per_length = 6
    per_length = tuple(
        (
            length,
            math.ceil(examples_per_length / batch_size),
        )
        for length, batch_size in schedule.buckets
    )
    per_role = sum(count for _, count in per_length)
    roles = (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE)
    total = per_role * len(roles)
    values = {
        "schedule_sha256": schedule.sha256,
        "analytic_nll_vjp_sha256": analytic_nll_vjp_sha256,
        "per_role_length_batch_counts": per_length,
        "chart_capture_roles": roles,
        "chart_batches_per_role": per_role,
        "total_chart_forward_count": total,
        "total_summed_shifted_nll_backward_count": total,
        "gradient_target_count_per_backward": 92,
        "canonical_carrier_site_gradient_count": 73,
        "auxiliary_gradient_count": 19,
        "D_chart_capture_count": 0,
    }
    provisional = object.__new__(CaptureCallLedger)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "ledger_sha256", "0" * 64)
    return CaptureCallLedger(
        **values,
        ledger_sha256=_digest(
            _CAPTURE_LEDGER_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


@dataclass(frozen=True, slots=True)
class ExecutionStorageBudget:
    transport_artifact_bytes: int = _PRODUCTION_TRANSPORT_BYTES
    fit_tensor_temp_bytes: int = _PRODUCTION_TEMP_BYTES
    atomic_publication_headroom_bytes: int = _PRODUCTION_ATOMIC_HEADROOM_BYTES
    minimum_free_bytes: int = _PRODUCTION_MINIMUM_FREE_BYTES
    maximum_resident_memory_bytes: int = (
        _PRODUCTION_MAXIMUM_RESIDENT_MEMORY_BYTES
    )
    minimum_physical_memory_bytes: int = (
        _PRODUCTION_MINIMUM_PHYSICAL_MEMORY_BYTES
    )

    def __post_init__(self) -> None:
        for label in (
            "transport_artifact_bytes",
            "fit_tensor_temp_bytes",
            "atomic_publication_headroom_bytes",
            "minimum_free_bytes",
            "maximum_resident_memory_bytes",
            "minimum_physical_memory_bytes",
        ):
            _require_positive_int(getattr(self, label), label=label)
        if self.minimum_free_bytes < (
            self.transport_artifact_bytes
            + self.fit_tensor_temp_bytes
            + self.atomic_publication_headroom_bytes
        ):
            raise ValueError("minimum_free_bytes does not cover declared storage")

    def payload(self) -> dict[str, int]:
        return {
            "atomic_publication_headroom_bytes": (
                self.atomic_publication_headroom_bytes
            ),
            "fit_tensor_temp_bytes": self.fit_tensor_temp_bytes,
            "maximum_resident_memory_bytes": (
                self.maximum_resident_memory_bytes
            ),
            "minimum_physical_memory_bytes": (
                self.minimum_physical_memory_bytes
            ),
            "minimum_free_bytes": self.minimum_free_bytes,
            "transport_artifact_bytes": self.transport_artifact_bytes,
        }


@dataclass(frozen=True, slots=True)
class ChartNativeExecutionBinding:
    repository_root: Path
    setup_path: str
    source_pins: tuple[ExecutionSourcePin, ...]
    schedule: LengthBucketBatchSchedule
    storage_budget: ExecutionStorageBudget
    output_directory: Path
    temp_directory: Path
    test_only_allow_nonproduction_shape: bool = False

    def __post_init__(self) -> None:
        root = Path(os.path.abspath(self.repository_root))
        output = Path(os.path.abspath(self.output_directory))
        temporary = Path(os.path.abspath(self.temp_directory))
        object.__setattr__(self, "repository_root", root)
        object.__setattr__(self, "output_directory", output)
        object.__setattr__(self, "temp_directory", temporary)
        if not isinstance(self.setup_path, str) or not self.setup_path:
            raise ValueError("setup_path must be nonempty")
        if not isinstance(self.source_pins, tuple):
            raise TypeError("source_pins must be a tuple")
        names = tuple(pin.name for pin in self.source_pins)
        if len(set(names)) != len(names) or set(names) != _REQUIRED_SOURCE_NAMES:
            raise ValueError("execution source pin inventory is incomplete")
        if type(self.test_only_allow_nonproduction_shape) is not bool:
            raise TypeError("test_only_allow_nonproduction_shape must be bool")
        if not self.test_only_allow_nonproduction_shape:
            if self.schedule.buckets != _EXPECTED_PRODUCTION_SCHEDULE:
                raise ValueError("live execution requires frozen length schedule")
            if (
                self.storage_budget.transport_artifact_bytes
                < _PRODUCTION_TRANSPORT_BYTES
                or self.storage_budget.fit_tensor_temp_bytes
                < _PRODUCTION_TEMP_BYTES
                or self.storage_budget.atomic_publication_headroom_bytes
                < _PRODUCTION_ATOMIC_HEADROOM_BYTES
                or self.storage_budget.minimum_free_bytes
                < _PRODUCTION_MINIMUM_FREE_BYTES
                or self.storage_budget.maximum_resident_memory_bytes
                != _PRODUCTION_MAXIMUM_RESIDENT_MEMORY_BYTES
                or self.storage_budget.minimum_physical_memory_bytes
                < _PRODUCTION_MINIMUM_PHYSICAL_MEMORY_BYTES
            ):
                raise ValueError("live execution storage/memory budget is too small")

    def payload(self) -> dict[str, object]:
        return {
            "output_directory": str(self.output_directory),
            "repository_root": str(self.repository_root),
            "schedule": self.schedule.payload(),
            "schedule_sha256": self.schedule.sha256,
            "setup_path": self.setup_path,
            "source_pins": [
                pin.payload() for pin in sorted(self.source_pins, key=lambda x: x.name)
            ],
            "storage_budget": self.storage_budget.payload(),
            "temp_directory": str(self.temp_directory),
            "test_only_allow_nonproduction_shape": (
                self.test_only_allow_nonproduction_shape
            ),
        }

    def execution_contract_payload(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "schedule": self.schedule.payload(),
            "schedule_sha256": self.schedule.sha256,
            "schema": "fisher_graph.chart_native_execution_contract.v1",
            "setup_path": self.setup_path,
            "source_pins": [
                pin.payload()
                for pin in sorted(self.source_pins, key=lambda item: item.name)
            ],
            "storage_budget": self.storage_budget.payload(),
            "test_only_allow_nonproduction_shape": (
                self.test_only_allow_nonproduction_shape
            ),
        }

    @property
    def execution_contract_sha256(self) -> str:
        return _digest(
            _EXECUTION_CONTRACT_DOMAIN,
            self.execution_contract_payload(),
        )

    @property
    def sha256(self) -> str:
        return _digest(_BINDING_DOMAIN, self.payload())


def build_default_execution_binding(
    *,
    repository_root: Path | str | None = None,
    output_directory: Path | str | None = None,
    temp_directory: Path | str | None = None,
) -> ChartNativeExecutionBinding:
    """Build the exact production binding from setup-owned source pins."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    output = (
        root / DEFAULT_REPORT_OUTPUT.parent
        if output_directory is None
        else Path(output_directory)
    )
    temporary = (
        output / ".chart-native-neighbor-temp"
        if temp_directory is None
        else Path(temp_directory)
    )
    pins = tuple(
        ExecutionSourcePin(
            name=name,
            path=str(receipt["path"]),
            file_sha256=str(receipt["file_sha256"]),
        )
        for name, receipt in sorted(_SETUP_IMPLEMENTATION_SOURCES.items())
    )
    return ChartNativeExecutionBinding(
        repository_root=root,
        setup_path=str(DEFAULT_SETUP_PATH),
        source_pins=pins,
        schedule=LengthBucketBatchSchedule(),
        storage_budget=ExecutionStorageBudget(),
        output_directory=output,
        temp_directory=temporary,
        test_only_allow_nonproduction_shape=False,
    )


def production_execution_contract_payload() -> dict[str, object]:
    schedule = LengthBucketBatchSchedule()
    storage = ExecutionStorageBudget()
    pins = tuple(
        ExecutionSourcePin(
            name=name,
            path=str(receipt["path"]),
            file_sha256=str(receipt["file_sha256"]),
        )
        for name, receipt in sorted(_SETUP_IMPLEMENTATION_SOURCES.items())
    )
    return {
        "format_version": 1,
        "schedule": schedule.payload(),
        "schedule_sha256": schedule.sha256,
        "schema": "fisher_graph.chart_native_execution_contract.v1",
        "setup_path": str(DEFAULT_SETUP_PATH),
        "source_pins": [pin.payload() for pin in pins],
        "storage_budget": storage.payload(),
        "test_only_allow_nonproduction_shape": False,
    }


@dataclass(frozen=True, slots=True)
class ExecutionPreflightReceipt:
    binding: ChartNativeExecutionBinding
    setup: Mapping[str, object]
    fresh_manifest: Mapping[str, object]
    fresh_receipt: Mapping[str, object]
    opaque_panel_receipt: object
    source_file_sha256s: Mapping[str, str]
    free_bytes_by_device: tuple[tuple[int, int], ...]
    physical_memory_bytes: int | None
    receipt_sha256: str

    def safe_payload_without_sha256(self) -> dict[str, object]:
        return {
            "binding_sha256": self.binding.sha256,
            "execution_contract": self.binding.execution_contract_payload(),
            "execution_contract_sha256": (
                self.binding.execution_contract_sha256
            ),
            "format_version": 1,
            "free_bytes_by_device": [
                [device, free] for device, free in self.free_bytes_by_device
            ],
            "fresh_manifest_sha256": self.fresh_manifest["manifest_sha256"],
            "fresh_receipt_sha256": self.fresh_receipt["receipt_sha256"],
            "opaque_role_file_sha256s": dict(
                getattr(self.opaque_panel_receipt, "role_file_sha256s")
            ),
            "physical_memory_bytes": self.physical_memory_bytes,
            "reserve_semantically_decoded": bool(
                getattr(
                    self.opaque_panel_receipt,
                    "reserve_semantically_decoded",
                )
            ),
            "schema": EXECUTION_PREFLIGHT_SCHEMA,
            "setup_sha256": self.setup["setup_sha256"],
            "source_file_sha256s": dict(self.source_file_sha256s),
            "transformers_or_model_runtime_touched_before_preflight": False,
        }

    def validate_receipt(self) -> None:
        _require_sha256(self.receipt_sha256, label="preflight receipt_sha256")
        if self.receipt_sha256 != _digest(
            _PREFLIGHT_DOMAIN,
            self.safe_payload_without_sha256(),
        ):
            raise ValueError("preflight receipt digest drifted")



def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if type(pages) is int and type(page_size) is int and pages > 0 and page_size > 0:
        return pages * page_size
    return None


def role_panel_binding_sha256(
    fresh_manifest: Mapping[str, object],
    role: str,
) -> str:
    role_manifest = fresh_manifest["roles"][role]
    return _digest(
        _PANEL_BINDING_DOMAIN,
        {
            "fresh_manifest_sha256": fresh_manifest["manifest_sha256"],
            "ordered_objective_row_catalog_sha256": role_manifest[
                "ordered_objective_row_catalog_sha256"
            ],
            "role": role,
            "role_file_sha256": role_manifest["file_sha256"],
            "score_objective_receipt_sha256": role_manifest[
                "score_objective_receipt_sha256"
            ],
        },
    )


@dataclass(frozen=True, slots=True)
class RoleEligibilityCatalog:
    role: str
    role_panel_binding_sha256: str
    all_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    fit_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    primary_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    response_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    short_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    boundary_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    transition_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    hard_long_family_example_ids: tuple[tuple[str, tuple[str, ...]], ...]
    catalog_sha256: str

    def __post_init__(self) -> None:
        if self.role not in (
            FIT_ROLE,
            DISCOVERY_ROLE,
            CONFIRMATION_ROLE,
            RESPONSE_ROLE,
        ):
            raise ValueError("eligibility catalog role is unsupported")
        _require_sha256(
            self.role_panel_binding_sha256,
            label="role_panel_binding_sha256",
        )
        primary_count = 7 if self.role in (DISCOVERY_ROLE, CONFIRMATION_ROLE) else 0
        fit_count = 8 if self.role == FIT_ROLE else 0
        response_count = 8 if self.role == RESPONSE_ROLE else 0
        fields = (
            ("all_family_example_ids", 8),
            ("fit_family_example_ids", fit_count),
            ("primary_family_example_ids", primary_count),
            ("response_family_example_ids", response_count),
            ("short_family_example_ids", 1),
            ("boundary_family_example_ids", 3),
            ("transition_family_example_ids", 1),
            ("hard_long_family_example_ids", 3),
        )
        all_mapping = dict(self.all_family_example_ids)
        if len(all_mapping) != 6:
            raise ValueError("eligibility catalog must contain six families")
        for field_name, count in fields:
            value = getattr(self, field_name)
            if (
                not isinstance(value, tuple)
                or tuple(family for family, _ in value)
                != tuple(all_mapping)
            ):
                raise ValueError("eligibility subset family order drifted")
            for family, examples in value:
                if (
                    not isinstance(family, str)
                    or not family
                    or not isinstance(examples, tuple)
                    or len(examples) != count
                    or len(set(examples)) != len(examples)
                    or any(
                        not isinstance(example, str) or not example
                        for example in examples
                    )
                    or not set(examples).issubset(all_mapping[family])
                ):
                    raise ValueError("eligibility subset examples drifted")
        _require_sha256(self.catalog_sha256, label="catalog_sha256")
        if self.catalog_sha256 != _digest(
            _ELIGIBILITY_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("eligibility catalog digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        def materialize(
            values: tuple[tuple[str, tuple[str, ...]], ...],
        ) -> dict[str, list[str]]:
            return {family: list(examples) for family, examples in values}

        return {
            "all_family_example_ids": materialize(
                self.all_family_example_ids
            ),
            "boundary_family_example_ids": materialize(
                self.boundary_family_example_ids
            ),
            "fit_family_example_ids": materialize(
                self.fit_family_example_ids
            ),
            "hard_long_family_example_ids": materialize(
                self.hard_long_family_example_ids
            ),
            "primary_family_example_ids": materialize(
                self.primary_family_example_ids
            ),
            "role": self.role,
            "role_panel_binding_sha256": self.role_panel_binding_sha256,
            "response_family_example_ids": materialize(
                self.response_family_example_ids
            ),
            "short_family_example_ids": materialize(
                self.short_family_example_ids
            ),
            "transition_family_example_ids": materialize(
                self.transition_family_example_ids
            ),
        }

    def mapping(self, subset: str) -> dict[str, tuple[str, ...]]:
        field_name = f"{subset}_family_example_ids"
        if field_name not in {
            "all_family_example_ids",
            "fit_family_example_ids",
            "primary_family_example_ids",
            "response_family_example_ids",
            "short_family_example_ids",
            "boundary_family_example_ids",
            "transition_family_example_ids",
            "hard_long_family_example_ids",
        }:
            raise ValueError("unknown eligibility subset")
        return dict(getattr(self, field_name))


def build_role_eligibility_catalog(
    fresh_manifest: Mapping[str, object],
    *,
    role: str,
) -> RoleEligibilityCatalog:
    if role not in (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE, RESPONSE_ROLE):
        raise ValueError("eligibility catalog role is not executable")
    role_manifest = fresh_manifest["roles"][role]
    members = role_manifest["ordered_members"]
    family_order = tuple(fresh_manifest["family_ids"])
    if len(family_order) != 6 or len(members) != 48:
        raise ValueError("fresh role does not contain exact 6x8 examples")
    by_family: dict[str, list[Mapping[str, object]]] = {
        family: [] for family in family_order
    }
    for member in members:
        family = member["family_id"]
        if family not in by_family:
            raise ValueError("fresh role contains an unknown family")
        by_family[family].append(member)
    for family, rows in by_family.items():
        if len(rows) != 8 or tuple(
            int(row["target_token_count"]) for row in rows
        ) != tuple(length for length, _ in _EXPECTED_PRODUCTION_SCHEDULE):
            raise ValueError(f"fresh family {family} length schedule drifted")

    def select(predicate: Callable[[Mapping[str, object]], bool]) -> tuple[
        tuple[str, tuple[str, ...]], ...
    ]:
        return tuple(
            (
                family,
                tuple(
                    str(row["example_id"])
                    for row in by_family[family]
                    if predicate(row)
                ),
            )
            for family in family_order
        )

    all_rows = select(lambda _: True)
    fit = select(lambda _: role == FIT_ROLE)
    primary = select(
        lambda row: role in (DISCOVERY_ROLE, CONFIRMATION_ROLE)
        and int(row["target_token_count"]) in _PRIMARY_ELIGIBLE_LENGTHS
    )
    response = select(lambda _: role == RESPONSE_ROLE)
    short = select(lambda row: int(row["target_token_count"]) == 96)
    boundary = select(
        lambda row: int(row["target_token_count"]) in (511, 512, 513)
    )
    transition = select(
        lambda row: int(row["target_token_count"]) == 514
    )
    hard_long = select(
        lambda row: int(row["target_token_count"]) in _HARD_LONG_LENGTHS
    )
    if role in (DISCOVERY_ROLE, CONFIRMATION_ROLE):
        if any(len(examples) != 7 for _, examples in primary):
            raise ValueError("B/C primary eligibility must be exact 7 per family")
        for row in members:
            target = int(row["target_token_count"])
            expected_primary = target in _PRIMARY_ELIGIBLE_LENGTHS
            if any(
                bool(row[field]) != expected_primary
                for field in (
                    "eligible_for_pooled_point_topology",
                    "eligible_for_bootstrap",
                    "eligible_for_LOFO",
                    "may_change_chart_neighbor_graph",
                )
            ):
                raise ValueError("B/C manifest primary eligibility flags drifted")
            if bool(row["boundary_transition_diagnostic_only"]) != (
                target == 514
            ):
                raise ValueError("514 transition diagnostic flag drifted")
            if bool(row["eligible_for_hard_long_veto"]) != (
                target in _HARD_LONG_LENGTHS
            ):
                raise ValueError("hard-long veto eligibility drifted")
    panel_sha = role_panel_binding_sha256(fresh_manifest, role)
    values = {
        "role": role,
        "role_panel_binding_sha256": panel_sha,
        "all_family_example_ids": all_rows,
        "fit_family_example_ids": fit,
        "primary_family_example_ids": primary,
        "response_family_example_ids": response,
        "short_family_example_ids": short,
        "boundary_family_example_ids": boundary,
        "transition_family_example_ids": transition,
        "hard_long_family_example_ids": hard_long,
    }
    provisional = object.__new__(RoleEligibilityCatalog)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "catalog_sha256", "0" * 64)
    return RoleEligibilityCatalog(
        **values,
        catalog_sha256=_digest(
            _ELIGIBILITY_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


def static_preflight(binding: ChartNativeExecutionBinding) -> ExecutionPreflightReceipt:
    """Authenticate all sources and opaque role bytes before runtime import."""

    if not isinstance(binding, ChartNativeExecutionBinding):
        raise TypeError("binding must be ChartNativeExecutionBinding")
    already_loaded = tuple(
        sorted(
            loaded
            for loaded in sys.modules
            if any(
                loaded == forbidden or loaded.startswith(forbidden + ".")
                for forbidden in _FORBIDDEN_BEFORE_PREFLIGHT
            )
        )
    )
    if already_loaded:
        raise RuntimeError(
            "model runtime modules were imported before static preflight: "
            + ", ".join(already_loaded)
        )
    root = binding.repository_root
    _regular_directory(root, label="repository_root")
    output_info = _regular_directory(
        binding.output_directory,
        label="output_directory",
    )
    temp_info = _regular_directory(binding.temp_directory, label="temp_directory")

    observed_sources: dict[str, str] = {}
    for pin in sorted(binding.source_pins, key=lambda item: item.name):
        path = _relative_path(root, pin.path, label=f"source {pin.name}")
        raw = _read_regular(
            path,
            label=f"source {pin.name}",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != pin.file_sha256:
            raise ValueError(f"source {pin.name} differs from execution pin")
        expected = PINNED_SOURCE_SHA256S.get(pin.name)
        if expected is not None and digest != expected:
            raise ValueError(f"source {pin.name} differs from worker pin")
        observed_sources[pin.name] = digest

    setup_module = importlib.import_module(
        "fisher_graph.gemma3_chart_native_neighbor_setup"
    )
    setup_path = _relative_path(root, binding.setup_path, label="setup artifact")
    setup_raw = _read_regular(
        setup_path,
        label="setup artifact",
        maximum_bytes=_MAX_SETUP_BYTES,
    )
    if hashlib.sha256(setup_raw).hexdigest() != PINNED_SETUP_FILE_SHA256:
        raise ValueError("setup artifact bytes differ from worker pin")
    setup = setup_module.check_setup(setup_path)
    if setup["setup_sha256"] != PINNED_SETUP_SHA256:
        raise ValueError("setup semantic digest differs from worker pin")
    predecessor = setup["frozen_predecessor_lineage"]
    expected_dynamic = {
        "event_executor": predecessor["event_executor_source"]["file_sha256"],
    }
    for name, expected in expected_dynamic.items():
        if observed_sources[name] != expected:
            raise ValueError(f"{name} differs from setup lineage")

    # This module is standard-library-only.  It is imported only after its
    # source bytes have been authenticated above.
    fresh = importlib.import_module(
        "fisher_graph.gemma3_chart_native_neighbor_fresh_panels"
    )
    fresh_manifest = fresh.load_manifest(root / fresh.DEFAULT_MANIFEST)
    fresh_receipt = fresh.load_receipt(root / fresh.DEFAULT_RECEIPT)
    frozen_fresh = setup["frozen_fresh_panel_lineage"]
    if (
        observed_sources["gemma3_chart_native_neighbor_fresh_panels"]
        != frozen_fresh["panel_source"]["file_sha256"]
        or str(fresh.DEFAULT_MANIFEST) != frozen_fresh["manifest"]["path"]
        or fresh_manifest["manifest_sha256"]
        != frozen_fresh["manifest"]["manifest_sha256"]
        or str(fresh.DEFAULT_RECEIPT) != frozen_fresh["receipt"]["path"]
        or fresh_receipt["receipt_sha256"]
        != frozen_fresh["receipt"]["receipt_sha256"]
    ):
        raise ValueError("fresh-panel source/manifest/receipt differs from setup")
    for role in ORDERED_ROLES:
        if (
            fresh.PINNED_ROLE_FILE_SHA256S[role]
            != frozen_fresh["role_files"][role]["file_sha256"]
        ):
            raise ValueError(f"fresh-panel role {role} differs from setup")
    opaque = fresh.opaque_preflight(
        repository_root=root,
        manifest=fresh_manifest,
        receipt=fresh_receipt,
    )
    if bool(getattr(opaque, "reserve_semantically_decoded")):
        raise RuntimeError("reserve semantic access invalidated preflight")

    free_by_device: dict[int, int] = {}
    for path, info in (
        (binding.output_directory, output_info),
        (binding.temp_directory, temp_info),
    ):
        free_by_device[info.st_dev] = min(
            free_by_device.get(info.st_dev, 2**63 - 1),
            shutil.disk_usage(path).free,
        )
    if not binding.test_only_allow_nonproduction_shape:
        if min(free_by_device.values()) < binding.storage_budget.minimum_free_bytes:
            raise RuntimeError("static preflight has insufficient free disk")
    physical = _physical_memory_bytes()
    if (
        not binding.test_only_allow_nonproduction_shape
        and (
            physical is None
            or physical < binding.storage_budget.minimum_physical_memory_bytes
        )
    ):
        raise RuntimeError("static preflight has insufficient physical memory")

    provisional = ExecutionPreflightReceipt(
        binding=binding,
        setup=setup,
        fresh_manifest=fresh_manifest,
        fresh_receipt=fresh_receipt,
        opaque_panel_receipt=opaque,
        source_file_sha256s=observed_sources,
        free_bytes_by_device=tuple(sorted(free_by_device.items())),
        physical_memory_bytes=physical,
        receipt_sha256="0" * 64,
    )
    receipt_sha256 = _digest(_PREFLIGHT_DOMAIN, provisional.safe_payload_without_sha256())
    return ExecutionPreflightReceipt(
        binding=binding,
        setup=setup,
        fresh_manifest=fresh_manifest,
        fresh_receipt=fresh_receipt,
        opaque_panel_receipt=opaque,
        source_file_sha256s=observed_sources,
        free_bytes_by_device=tuple(sorted(free_by_device.items())),
        physical_memory_bytes=physical,
        receipt_sha256=receipt_sha256,
    )


def _peak_resident_set_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if not isinstance(observed, (int, float)) or not math.isfinite(observed):
        raise RuntimeError("peak resident-set measurement is unavailable")
    # Darwin reports bytes; Linux and the BSDs exposed by Python report KiB.
    multiplier = 1 if sys.platform == "darwin" else 1024
    result = int(observed * multiplier)
    if result <= 0:
        raise RuntimeError("peak resident-set measurement is invalid")
    return result


def _unique_regular_bytes(root: Path) -> int:
    """Return exact logical bytes, counting hard-linked files only once."""

    _regular_directory(root, label="resource ledger root")
    pending = [root]
    seen: set[tuple[int, int]] = set()
    total = 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise RuntimeError("resource ledger refuses symlinked entries")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    identity = (info.st_dev, info.st_ino)
                    if identity not in seen:
                        seen.add(identity)
                        total += info.st_size
                else:
                    raise RuntimeError("resource ledger found a non-regular entry")
    return total


@dataclass(frozen=True, slots=True)
class PhaseResourceObservation:
    phase_id: str
    peak_resident_set_bytes: int
    temp_unique_regular_bytes: int
    temp_free_bytes: int
    completed_pair_count: int
    total_pair_count: int
    observed_monotonic_ns: int
    receipt_sha256: str

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "completed_pair_count": self.completed_pair_count,
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "peak_resident_set_bytes": self.peak_resident_set_bytes,
            "phase_id": self.phase_id,
            "temp_free_bytes": self.temp_free_bytes,
            "temp_unique_regular_bytes": self.temp_unique_regular_bytes,
            "total_pair_count": self.total_pair_count,
        }


def observe_phase_resources(
    binding: ChartNativeExecutionBinding,
    *,
    phase_id: str,
    completed_pair_count: int = 0,
    total_pair_count: int = 0,
) -> PhaseResourceObservation:
    if not isinstance(phase_id, str) or not phase_id:
        raise ValueError("phase_id must be nonempty")
    if (
        type(completed_pair_count) is not int
        or type(total_pair_count) is not int
        or completed_pair_count < 0
        or total_pair_count < 0
        or completed_pair_count > total_pair_count
    ):
        raise ValueError("pair progress counters are invalid")
    rss = _peak_resident_set_bytes()
    temporary = _unique_regular_bytes(binding.temp_directory)
    free = shutil.disk_usage(binding.temp_directory).free
    if not binding.test_only_allow_nonproduction_shape:
        if rss > binding.storage_budget.maximum_resident_memory_bytes:
            raise RuntimeError("live phase exceeded the frozen resident-set cap")
        if free < binding.storage_budget.atomic_publication_headroom_bytes:
            raise RuntimeError("live phase lost atomic-publication disk headroom")
        if temporary > binding.storage_budget.fit_tensor_temp_bytes:
            raise RuntimeError("live phase exceeded the frozen temp-byte cap")
    values = {
        "phase_id": phase_id,
        "peak_resident_set_bytes": rss,
        "temp_unique_regular_bytes": temporary,
        "temp_free_bytes": free,
        "completed_pair_count": completed_pair_count,
        "total_pair_count": total_pair_count,
        "observed_monotonic_ns": time.monotonic_ns(),
    }
    provisional = PhaseResourceObservation(**values, receipt_sha256="0" * 64)
    return PhaseResourceObservation(
        **values,
        receipt_sha256=_digest(
            _RESOURCE_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


@dataclass(frozen=True, slots=True)
class FrozenStageReceipt:
    role: str
    stage_kind: str
    preflight_receipt_sha256: str
    role_panel_binding_sha256: str
    ordered_objective_row_catalog_sha256: str
    score_objective_receipt_sha256: str
    prior_stage_receipt_sha256: str | None
    artifact_sha256s: tuple[tuple[str, str], ...]
    primary_eligible_example_catalog_sha256: str | None
    stage_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.role not in _STAGE_KIND_BY_ROLE:
            raise ValueError("frozen stage role is unsupported")
        if self.stage_kind != _STAGE_KIND_BY_ROLE[self.role]:
            raise ValueError("frozen stage kind does not match role")
        for label in (
            "preflight_receipt_sha256",
            "role_panel_binding_sha256",
            "ordered_objective_row_catalog_sha256",
            "score_objective_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.prior_stage_receipt_sha256 is not None:
            _require_sha256(
                self.prior_stage_receipt_sha256,
                label="prior_stage_receipt_sha256",
            )
        if self.primary_eligible_example_catalog_sha256 is not None:
            _require_sha256(
                self.primary_eligible_example_catalog_sha256,
                label="primary_eligible_example_catalog_sha256",
            )
        if (
            not isinstance(self.artifact_sha256s, tuple)
            or not self.artifact_sha256s
            or tuple(sorted(self.artifact_sha256s)) != self.artifact_sha256s
            or len({name for name, _ in self.artifact_sha256s})
            != len(self.artifact_sha256s)
        ):
            raise ValueError("stage artifacts must be a sorted nonempty tuple")
        for name, digest in self.artifact_sha256s:
            if not isinstance(name, str) or not name:
                raise ValueError("stage artifact names must be nonempty")
            _require_sha256(digest, label=f"stage artifact {name}")
        _require_sha256(self.stage_receipt_sha256, label="stage_receipt_sha256")
        if self.stage_receipt_sha256 != _digest(
            _STAGE_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("stage receipt digest does not authenticate stage")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "artifact_sha256s": dict(self.artifact_sha256s),
            "ordered_objective_row_catalog_sha256": (
                self.ordered_objective_row_catalog_sha256
            ),
            "preflight_receipt_sha256": self.preflight_receipt_sha256,
            "primary_eligible_example_catalog_sha256": (
                self.primary_eligible_example_catalog_sha256
            ),
            "prior_stage_receipt_sha256": self.prior_stage_receipt_sha256,
            "role": self.role,
            "role_panel_binding_sha256": self.role_panel_binding_sha256,
            "score_objective_receipt_sha256": (
                self.score_objective_receipt_sha256
            ),
            "stage_kind": self.stage_kind,
        }


def build_frozen_stage_receipt(
    *,
    role: str,
    preflight_receipt_sha256: str,
    role_panel_binding_sha256_value: str,
    ordered_objective_row_catalog_sha256: str,
    score_objective_receipt_sha256: str,
    prior_stage_receipt_sha256: str | None,
    artifact_sha256s: Mapping[str, str],
    primary_eligible_example_catalog_sha256: str | None,
) -> FrozenStageReceipt:
    values = {
        "role": role,
        "stage_kind": _STAGE_KIND_BY_ROLE.get(role, ""),
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "role_panel_binding_sha256": role_panel_binding_sha256_value,
        "ordered_objective_row_catalog_sha256": (
            ordered_objective_row_catalog_sha256
        ),
        "score_objective_receipt_sha256": score_objective_receipt_sha256,
        "prior_stage_receipt_sha256": prior_stage_receipt_sha256,
        "artifact_sha256s": tuple(sorted(artifact_sha256s.items())),
        "primary_eligible_example_catalog_sha256": (
            primary_eligible_example_catalog_sha256
        ),
    }
    provisional = object.__new__(FrozenStageReceipt)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "stage_receipt_sha256", "0" * 64)
    return FrozenStageReceipt(
        **values,
        stage_receipt_sha256=_digest(
            _STAGE_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


def _assert_summary_safe(value: object, *, path: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} has a non-string or empty key")
            if key.lower() in _FORBIDDEN_REPORT_FIELDS:
                raise ValueError(f"{path}.{key} would leak private row data")
            _assert_summary_safe(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_summary_safe(child, path=f"{path}[{index}]")
        return
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 4096:
            raise ValueError(f"{path} contains an oversized string")
        return
    raise ValueError(f"{path} contains a non-JSON or raw tensor value")


def seal_experiment_report(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("report payload must be a mapping")
    if {"schema", "format_version", "report_sha256"} & set(payload):
        raise ValueError("report envelope fields are worker-owned")
    report = {
        "schema": EXPERIMENT_REPORT_SCHEMA,
        "format_version": EXPERIMENT_REPORT_FORMAT_VERSION,
        **dict(payload),
    }
    _assert_summary_safe(report)
    report["report_sha256"] = _digest(_REPORT_DOMAIN, report)
    return report


def _strict_load_experiment_report_envelope_bytes(
    raw: bytes,
) -> dict[str, object]:
    """Load only the canonical hash envelope.

    This private seam exists for focused byte-I/O tests.  It deliberately does
    not authorize a scientific result; every public load goes through the full
    completed-report replay below.
    """

    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_REPORT_BYTES:
        raise ValueError("experiment report byte size is invalid")
    parsed = _strict_json_loads(raw, require_canonical=True)
    if not isinstance(parsed, dict):
        raise ValueError("experiment report must be a JSON object")
    if (
        parsed.get("schema") != EXPERIMENT_REPORT_SCHEMA
        or parsed.get("format_version") != EXPERIMENT_REPORT_FORMAT_VERSION
    ):
        raise ValueError("experiment report envelope drifted")
    observed = parsed.get("report_sha256")
    _require_sha256(observed, label="report_sha256")
    unsigned = dict(parsed)
    unsigned.pop("report_sha256")
    if observed != _digest(_REPORT_DOMAIN, unsigned):
        raise ValueError("experiment report digest drifted")
    _assert_summary_safe(parsed)
    return parsed


def strict_load_experiment_report_bytes(raw: bytes) -> dict[str, object]:
    parsed = _strict_load_experiment_report_envelope_bytes(raw)
    _validate_completed_experiment_report(parsed)
    return parsed


def _validate_svg_bytes(raw: bytes) -> None:
    if not raw or len(raw) > _MAX_SVG_BYTES:
        raise ValueError("neighbor SVG byte size is invalid")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError("neighbor SVG is not valid XML") from error
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("neighbor visualization root must be SVG")
    for element in root.iter():
        local = element.tag.rsplit("}", 1)[-1]
        if local in {"script", "foreignObject", "image", "use"}:
            raise ValueError("neighbor SVG contains an unsafe element")
        if any(key.rsplit("}", 1)[-1] == "href" for key in element.attrib):
            raise ValueError("neighbor SVG may not reference external content")


def render_neighbor_svg(
    *,
    mode_ids: Sequence[str],
    confirmed_edges: Sequence[Sequence[str]],
) -> bytes:
    modes = tuple(mode_ids)
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(not isinstance(mode, str) or not mode for mode in modes)
    ):
        raise ValueError("mode_ids must be unique nonempty strings")
    mode_set = set(modes)
    edges: list[tuple[str, str]] = []
    for edge in confirmed_edges:
        if (
            isinstance(edge, (str, bytes))
            or len(edge) != 2
            or edge[0] == edge[1]
            or edge[0] not in mode_set
            or edge[1] not in mode_set
        ):
            raise ValueError("confirmed edge endpoints are invalid")
        edges.append(tuple(sorted((str(edge[0]), str(edge[1])))))
    if len(set(edges)) != len(edges):
        raise ValueError("confirmed edges must be unique")
    ordered_edges = sorted(edges)
    width = 1200
    center = width / 2
    radius = 480
    coordinates = {
        mode: (
            center + radius * math.cos(2 * math.pi * index / len(modes)),
            center + radius * math.sin(2 * math.pi * index / len(modes)),
        )
        for index, mode in enumerate(modes)
    }
    rows = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" '
        'height="1200" viewBox="0 0 1200 1200">',
        '<rect width="1200" height="1200" fill="#fbfaf7"/>',
        '<g stroke="#607d8b" stroke-width="2" opacity="0.75">',
    ]
    for left, right in ordered_edges:
        x1, y1 = coordinates[left]
        x2, y2 = coordinates[right]
        rows.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" '
            f'x2="{x2:.3f}" y2="{y2:.3f}"/>'
        )
    rows.append("</g><g>")
    for mode in modes:
        x, y = coordinates[mode]
        label = html.escape(mode)
        rows.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="14" '
            'fill="#263238"/>'
            f'<text x="{x + 18:.3f}" y="{y + 4:.3f}" '
            'font-family="sans-serif" font-size="12" '
            f'fill="#263238">{label}</text>'
        )
    rows.append("</g></svg>")
    encoded = "".join(rows).encode("utf-8")
    _validate_svg_bytes(encoded)
    return encoded


def _strict_rehash_frozen_repository_inputs(
    *,
    report: Mapping[str, object],
    repository_root: Path,
) -> None:
    """Re-read every frozen local input used to authorize a public report."""

    root = Path(os.path.abspath(repository_root))
    _regular_directory(root, label="strict report repository root")
    setup_path = _relative_path(
        root,
        str(DEFAULT_SETUP_PATH),
        label="canonical setup artifact",
    )
    setup_raw = _read_regular(
        setup_path,
        label="canonical setup artifact",
        maximum_bytes=_MAX_SETUP_BYTES,
    )
    if hashlib.sha256(setup_raw).hexdigest() != PINNED_SETUP_FILE_SHA256:
        raise ValueError("live setup artifact file digest drifted")
    from fisher_graph.gemma3_chart_native_neighbor_setup import check_setup

    setup = check_setup(setup_path)
    if setup["setup_sha256"] != report["preflight"]["setup_sha256"]:
        raise ValueError("live setup semantic digest differs from report")
    implementations = setup["frozen_worker_implementation_sources"]
    reported_sources = report["preflight"]["source_file_sha256s"]
    if (
        not isinstance(implementations, Mapping)
        or not isinstance(reported_sources, Mapping)
        or set(implementations) != set(reported_sources)
    ):
        raise ValueError("live implementation source catalog drifted")
    for name, receipt in implementations.items():
        if not isinstance(receipt, Mapping):
            raise ValueError(f"implementation source {name} receipt drifted")
        source = _relative_path(
            root,
            str(receipt["path"]),
            label=f"implementation source {name}",
        )
        digest = hashlib.sha256(
            _read_regular(
                source,
                label=f"implementation source {name}",
                maximum_bytes=_MAX_SOURCE_BYTES,
            )
        ).hexdigest()
        if (
            digest != receipt["file_sha256"]
            or digest != reported_sources[name]
        ):
            raise ValueError(f"live implementation source {name} drifted")
    fresh = setup["frozen_fresh_panel_lineage"]
    frozen_files = {
        "fresh manifest": fresh["manifest"],
        "fresh receipt": fresh["receipt"],
        **{
            f"fresh role {role}": receipt
            for role, receipt in fresh["role_files"].items()
        },
    }
    for label, receipt in frozen_files.items():
        source = _relative_path(
            root,
            str(receipt["path"]),
            label=label,
        )
        digest = hashlib.sha256(
            _read_regular(
                source,
                label=label,
                maximum_bytes=_MAX_SOURCE_BYTES,
            )
        ).hexdigest()
        if digest != receipt["file_sha256"]:
            raise ValueError(f"live {label} bytes drifted")
    # The qualification is model-derived but prompt-free.  Its canonical bytes
    # are also part of the setup and must remain present at public load time.
    _load_frozen_runtime_qualification_from_setup(
        setup=setup,
        repository_root=root,
    )


def strict_load_experiment_report(
    report_path: Path | str,
    *,
    svg_path: Path | str | None = None,
    atlas_path: Path | str | None = None,
) -> dict[str, object]:
    """Strictly load a completed report, SVG, and shared chart atlas."""

    report_source = Path(report_path)
    svg_source = (
        report_source.with_suffix(".svg") if svg_path is None else Path(svg_path)
    )
    report = strict_load_experiment_report_bytes(
        _read_regular(
            report_source,
            label="completed experiment report",
            maximum_bytes=_MAX_REPORT_BYTES,
        )
    )
    _strict_rehash_frozen_repository_inputs(
        report=report,
        repository_root=Path(__file__).resolve().parents[2],
    )
    observed_svg = _read_regular(
        svg_source,
        label="completed neighbor SVG",
        maximum_bytes=_MAX_SVG_BYTES,
    )
    _validate_svg_bytes(observed_svg)
    expected_svg = render_neighbor_svg(
        mode_ids=report["mode_axis"],
        confirmed_edges=report["confirmed_edges"],
    )
    if observed_svg != expected_svg:
        raise ValueError("published neighbor SVG differs from report render")
    if report["neighbor_svg_sha256"] != hashlib.sha256(observed_svg).hexdigest():
        raise ValueError("published neighbor SVG digest differs from report")
    atlas_name = report["A_fit"]["published_atlas_file_name"]
    atlas_source = (
        report_source.parent / atlas_name
        if atlas_path is None
        else Path(atlas_path)
    )
    if atlas_source.name != atlas_name:
        raise ValueError("published atlas path differs from report name")
    observed_atlas = _read_regular(
        atlas_source,
        label="published shared chart atlas",
        maximum_bytes=_MAX_ATLAS_BYTES,
    )
    if (
        hashlib.sha256(observed_atlas).hexdigest()
        != report["A_fit"]["atlas_artifact_file_sha256"]
    ):
        raise ValueError("published shared chart atlas file digest drifted")
    from fisher_graph.shared_chart_energy_transport import (
        load_shared_chart_atlas,
    )

    atlas = load_shared_chart_atlas(atlas_source)
    if atlas.atlas_sha256 != report["A_fit"]["atlas_sha256"]:
        raise ValueError("published shared chart atlas semantic digest drifted")
    return report


@dataclass(frozen=True, slots=True)
class PublishedExperimentReceipt:
    atlas_path: str
    report_path: str
    svg_path: str
    atlas_file_sha256: str
    atlas_sha256: str
    report_file_sha256: str
    svg_file_sha256: str
    report_sha256: str
    publication_receipt_sha256: str


def publish_experiment_outputs(
    *,
    report: Mapping[str, object],
    svg_bytes: bytes,
    atlas_source_path: Path | str,
    report_path: Path | str,
    svg_path: Path | str,
    atlas_path: Path | str,
) -> PublishedExperimentReceipt:
    report_destination = Path(report_path)
    svg_destination = Path(svg_path)
    atlas_source = Path(atlas_source_path)
    atlas_destination = Path(atlas_path)
    if len({report_destination, svg_destination, atlas_destination}) != 3:
        raise ValueError("report, SVG, and atlas destinations must differ")
    for path in (report_destination, svg_destination, atlas_destination):
        _regular_directory(path.parent, label="publication parent")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    report_raw = _canonical_json_bytes(dict(report))
    loaded = strict_load_experiment_report_bytes(report_raw)
    _validate_svg_bytes(svg_bytes)
    expected_svg = render_neighbor_svg(
        mode_ids=loaded["mode_axis"],
        confirmed_edges=loaded["confirmed_edges"],
    )
    if svg_bytes != expected_svg:
        raise ValueError(
            "neighbor SVG does not equal the deterministic report render"
        )
    expected_svg_sha256 = hashlib.sha256(expected_svg).hexdigest()
    if (
        "neighbor_svg_sha256" in loaded
        and loaded["neighbor_svg_sha256"] != expected_svg_sha256
    ):
        raise ValueError("neighbor visualization digest drifted")
    expected_atlas_name = loaded["A_fit"]["published_atlas_file_name"]
    if (
        expected_atlas_name
        != "chart-native-neighbor-experiment-v1.atlas.bundle"
        or atlas_destination.name != expected_atlas_name
    ):
        raise ValueError("published shared chart atlas name drifted")
    atlas_raw = _read_regular(
        atlas_source,
        label="source shared chart atlas",
        maximum_bytes=_MAX_ATLAS_BYTES,
    )
    atlas_file_sha256 = hashlib.sha256(atlas_raw).hexdigest()
    if atlas_file_sha256 != loaded["A_fit"]["atlas_artifact_file_sha256"]:
        raise ValueError("source shared chart atlas file digest drifted")
    from fisher_graph.shared_chart_energy_transport import (
        load_shared_chart_atlas,
    )

    source_atlas = load_shared_chart_atlas(atlas_source)
    if source_atlas.atlas_sha256 != loaded["A_fit"]["atlas_sha256"]:
        raise ValueError("source shared chart atlas semantic digest drifted")
    temporaries: list[Path] = []
    published: list[Path] = []
    try:
        for destination, raw in (
            (svg_destination, svg_bytes),
            (atlas_destination, atlas_raw),
            # The report is the commit marker and is always linked last.
            (report_destination, report_raw),
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            temporaries.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        # Sidecars become durable before the report commit marker is linked.
        for temporary, destination in zip(
            temporaries[:2],
            (svg_destination, atlas_destination),
        ):
            os.link(temporary, destination)
            published.append(destination)
        for directory in sorted(
            {
                svg_destination.parent,
                atlas_destination.parent,
            },
            key=str,
        ):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.link(temporaries[2], report_destination)
        published.append(report_destination)
        descriptor = os.open(report_destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if strict_load_experiment_report(
            report_destination,
            svg_path=svg_destination,
            atlas_path=atlas_destination,
        ) != loaded:
            raise RuntimeError("published report strict reload drifted")
    except BaseException:
        for path in reversed(published):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        for directory in {
            report_destination.parent,
            svg_destination.parent,
            atlas_destination.parent,
        }:
            try:
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
        raise
    finally:
        for temporary in temporaries:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    report_file_sha256 = hashlib.sha256(report_raw).hexdigest()
    svg_file_sha256 = hashlib.sha256(svg_bytes).hexdigest()
    payload = {
        "atlas_file_sha256": atlas_file_sha256,
        "atlas_path": str(atlas_destination),
        "atlas_sha256": source_atlas.atlas_sha256,
        "report_file_sha256": report_file_sha256,
        "report_path": str(report_destination),
        "report_sha256": loaded["report_sha256"],
        "svg_file_sha256": svg_file_sha256,
        "svg_path": str(svg_destination),
    }
    return PublishedExperimentReceipt(
        **payload,
        publication_receipt_sha256=_digest(_PUBLICATION_DOMAIN, payload),
    )


def _rollback_published_experiment(
    publication: PublishedExperimentReceipt,
) -> None:
    """Durably remove the report commit marker before either sidecar."""

    report = Path(publication.report_path)
    sidecars = (Path(publication.svg_path), Path(publication.atlas_path))
    try:
        os.unlink(report)
    except FileNotFoundError:
        pass
    try:
        descriptor = os.open(report.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
    for path in sidecars:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    for directory in sorted({path.parent for path in sidecars}, key=str):
        try:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Disk-backed chart capture and phase orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeModeLineage:
    """Prompt-free runtime lineage needed to construct one core mode spec."""

    mode_id: str
    state_fingerprint_sha256: str
    site_placement_sha256: str
    comparison_adapter_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or not self.mode_id:
            raise ValueError("runtime mode_id must be nonempty")
        for label in (
            "state_fingerprint_sha256",
            "site_placement_sha256",
            "comparison_adapter_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)


def _runtime_mode_lineage_payload(
    rows: Sequence[RuntimeModeLineage],
) -> list[dict[str, str]]:
    return [
        {
            "comparison_adapter_receipt_sha256": (
                row.comparison_adapter_receipt_sha256
            ),
            "mode_id": row.mode_id,
            "site_placement_sha256": row.site_placement_sha256,
            "state_fingerprint_sha256": row.state_fingerprint_sha256,
        }
        for row in rows
    ]


def runtime_mode_lineage_catalog_sha256(
    rows: Sequence[RuntimeModeLineage],
) -> str:
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("runtime mode lineage catalog is empty")
    return _digest(_MODE_LINEAGE_DOMAIN, _runtime_mode_lineage_payload(rows))


def derive_runtime_mode_lineage(
    *,
    setup_modes: Sequence[Mapping[str, object]],
    state_fingerprints_by_mode: Mapping[str, str],
    hidden_width: int,
) -> tuple[RuntimeModeLineage, ...]:
    """Derive placement and adapter receipts from the frozen mode ontology."""

    if type(hidden_width) is not int or hidden_width <= 0:
        raise ValueError("runtime lineage hidden_width must be positive")
    if not isinstance(setup_modes, Sequence) or isinstance(
        setup_modes, (str, bytes, bytearray)
    ):
        raise TypeError("setup_modes must be an ordered sequence")
    if not isinstance(state_fingerprints_by_mode, Mapping):
        raise TypeError("state_fingerprints_by_mode must be a mapping")
    from fisher_graph.chart_energy_neighbor_graph import (
        CAUSAL_SEQUENCE_NATIVE_ABI,
        DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        EMBEDDING_CARRIER_H640_ADAPTER,
        LEXICAL_INGRESS_NATIVE_ABI,
        LEXICAL_READOUT_NATIVE_ABI,
        SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
        TOKEN_LOCAL_NATIVE_ABI,
        NativeChartABI,
        canonical_carrier_comparison_abi,
    )

    kind_contract = {
        "attention": (
            CAUSAL_SEQUENCE_NATIVE_ABI,
            DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        ),
        "mlp": (
            TOKEN_LOCAL_NATIVE_ABI,
            DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        ),
        "lexical_ingress": (
            LEXICAL_INGRESS_NATIVE_ABI,
            EMBEDDING_CARRIER_H640_ADAPTER,
        ),
        "lexical_egress": (
            LEXICAL_READOUT_NATIVE_ABI,
            SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
        ),
    }
    mode_ids = tuple(str(row.get("mode_id")) for row in setup_modes)
    if set(state_fingerprints_by_mode) != set(mode_ids):
        raise ValueError("runtime state fingerprint axis differs from setup")
    comparison = canonical_carrier_comparison_abi()
    result: list[RuntimeModeLineage] = []
    for ordinal, mode in enumerate(setup_modes):
        required = {
            "comparison_ABI",
            "context_site_id",
            "contribution_rule",
            "contribution_site_id",
            "event_id",
            "event_index",
            "mode_id",
            "mode_index",
            "mode_kind",
            "native_ABI",
            "score_gradient_site_id",
            "value_site_id",
        }
        if not required.issubset(mode):
            raise ValueError("setup mode omits runtime lineage fields")
        if mode["mode_index"] != ordinal:
            raise ValueError("setup mode order differs from mode_index")
        mode_id = str(mode["mode_id"])
        mode_kind = str(mode["mode_kind"])
        if mode_kind not in kind_contract:
            raise ValueError("setup mode_kind has no runtime ABI contract")
        abi_kind, adapter_kind = kind_contract[mode_kind]
        native = NativeChartABI(
            abi_kind=abi_kind,
            abi_name=str(mode["native_ABI"]),
            abi_version=1,
            input_signature=f"{abi_kind}.input",
            output_signature=f"{abi_kind}.output",
            context_signature=f"{abi_kind}.context",
        )
        if str(mode["comparison_ABI"]) != comparison.abi_name:
            raise ValueError("setup common comparison ABI drifted")
        state_sha = state_fingerprints_by_mode[mode_id]
        _require_sha256(state_sha, label=f"state fingerprint {mode_id}")
        placement_sha = _digest(
            _RUNTIME_PLACEMENT_DOMAIN,
            {
                "context_site_id": str(mode["context_site_id"]),
                "contribution_site_id": str(mode["contribution_site_id"]),
                "event_id": str(mode["event_id"]),
                "event_index": int(mode["event_index"]),
                "mode_id": mode_id,
                "score_gradient_site_id": str(
                    mode["score_gradient_site_id"]
                ),
                "value_site_id": str(mode["value_site_id"]),
            },
        )
        adapter_sha = _digest(
            _RUNTIME_ADAPTER_DOMAIN,
            {
                "adapter_kind": adapter_kind,
                "carrier_width": hidden_width,
                "comparison_abi_sha256": comparison.sha256,
                "mapping": str(mode["contribution_rule"]),
                "native_abi_sha256": native.sha256,
            },
        )
        result.append(
            RuntimeModeLineage(
                mode_id=mode_id,
                state_fingerprint_sha256=state_sha,
                site_placement_sha256=placement_sha,
                comparison_adapter_receipt_sha256=adapter_sha,
            )
        )
    if len(set(mode_ids)) != len(mode_ids):
        raise ValueError("setup mode_id axis collides")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RuntimeQualificationReceipt:
    """Prompt-free, pre-panel anchor for the compiled runtime lineage."""

    schema: str
    format_version: int
    model_id: str
    model_revision: str
    source_model_fingerprint_before_compilation_sha256: str
    source_model_fingerprint_after_compilation_sha256: str
    source_execution_fingerprint_before_compilation_sha256: str
    source_execution_fingerprint_after_compilation_sha256: str
    source_model_mutated_during_compilation: bool
    source_parameter_count: int
    hidden_width: int
    compiled_execution_fingerprint_sha256: str
    compiler_receipt_sha256: str
    tied_readout_binding_receipt_sha256: str
    tied_lexical_lineage_sha256: str
    final_logit_softcap: float | None
    mode_lineage: tuple[RuntimeModeLineage, ...]
    mode_lineage_catalog_sha256: str
    lexical_kernel_receipt_sha256: str
    streaming_kernel_receipt_sha256: str
    factorized_route_kernel_receipt_sha256: str
    contains_prompt_text: bool
    contains_token_ids: bool
    contains_raw_activations: bool
    contains_raw_gradients: bool
    contains_logits: bool
    qualification_receipt_sha256: str

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "compiled_execution_fingerprint_sha256": (
                self.compiled_execution_fingerprint_sha256
            ),
            "compiler_receipt_sha256": self.compiler_receipt_sha256,
            "contains_logits": self.contains_logits,
            "contains_prompt_text": self.contains_prompt_text,
            "contains_raw_activations": self.contains_raw_activations,
            "contains_raw_gradients": self.contains_raw_gradients,
            "contains_token_ids": self.contains_token_ids,
            "factorized_route_kernel_receipt_sha256": (
                self.factorized_route_kernel_receipt_sha256
            ),
            "final_logit_softcap": self.final_logit_softcap,
            "format_version": self.format_version,
            "hidden_width": self.hidden_width,
            "lexical_kernel_receipt_sha256": (
                self.lexical_kernel_receipt_sha256
            ),
            "mode_lineage": _runtime_mode_lineage_payload(self.mode_lineage),
            "mode_lineage_catalog_sha256": (
                self.mode_lineage_catalog_sha256
            ),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "schema": self.schema,
            "source_execution_fingerprint_after_compilation_sha256": (
                self.source_execution_fingerprint_after_compilation_sha256
            ),
            "source_execution_fingerprint_before_compilation_sha256": (
                self.source_execution_fingerprint_before_compilation_sha256
            ),
            "source_model_fingerprint_after_compilation_sha256": (
                self.source_model_fingerprint_after_compilation_sha256
            ),
            "source_model_fingerprint_before_compilation_sha256": (
                self.source_model_fingerprint_before_compilation_sha256
            ),
            "source_model_mutated_during_compilation": (
                self.source_model_mutated_during_compilation
            ),
            "source_parameter_count": self.source_parameter_count,
            "streaming_kernel_receipt_sha256": (
                self.streaming_kernel_receipt_sha256
            ),
            "tied_lexical_lineage_sha256": self.tied_lexical_lineage_sha256,
            "tied_readout_binding_receipt_sha256": (
                self.tied_readout_binding_receipt_sha256
            ),
        }

    def __post_init__(self) -> None:
        if self.schema != (
            "fisher_graph.gemma3_chart_native_neighbor."
            "runtime_qualification.v1"
        ) or self.format_version != 1:
            raise ValueError("runtime qualification schema drifted")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("runtime qualification model_id is invalid")
        if not isinstance(self.model_revision, str) or not self.model_revision:
            raise ValueError("runtime qualification revision is invalid")
        for label in (
            "source_model_fingerprint_before_compilation_sha256",
            "source_model_fingerprint_after_compilation_sha256",
            "source_execution_fingerprint_before_compilation_sha256",
            "source_execution_fingerprint_after_compilation_sha256",
            "compiled_execution_fingerprint_sha256",
            "compiler_receipt_sha256",
            "tied_readout_binding_receipt_sha256",
            "tied_lexical_lineage_sha256",
            "lexical_kernel_receipt_sha256",
            "streaming_kernel_receipt_sha256",
            "factorized_route_kernel_receipt_sha256",
            "mode_lineage_catalog_sha256",
            "qualification_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if (
            type(self.source_parameter_count) is not int
            or self.source_parameter_count <= 0
            or type(self.hidden_width) is not int
            or self.hidden_width <= 0
            or not isinstance(self.mode_lineage, tuple)
            or not self.mode_lineage
            or len({row.mode_id for row in self.mode_lineage})
            != len(self.mode_lineage)
            or self.mode_lineage_catalog_sha256
            != runtime_mode_lineage_catalog_sha256(self.mode_lineage)
        ):
            raise ValueError("runtime qualification inventory drifted")
        if self.final_logit_softcap is not None and (
            type(self.final_logit_softcap) is not float
            or not math.isfinite(self.final_logit_softcap)
            or self.final_logit_softcap <= 0.0
        ):
            raise ValueError("runtime qualification softcap drifted")
        if (
            self.source_model_mutated_during_compilation is not False
            or self.source_model_fingerprint_before_compilation_sha256
            != self.source_model_fingerprint_after_compilation_sha256
            or self.source_execution_fingerprint_before_compilation_sha256
            != self.source_execution_fingerprint_after_compilation_sha256
            or self.contains_prompt_text is not False
            or self.contains_token_ids is not False
            or self.contains_raw_activations is not False
            or self.contains_raw_gradients is not False
            or self.contains_logits is not False
        ):
            raise ValueError("runtime qualification violates no-mutation/no-raw contract")
        if self.qualification_receipt_sha256 != _digest(
            _RUNTIME_QUALIFICATION_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("runtime qualification digest drifted")


def build_runtime_qualification_receipt(
    **values: object,
) -> RuntimeQualificationReceipt:
    expected = {
        name
        for name in RuntimeQualificationReceipt.__dataclass_fields__
        if name != "qualification_receipt_sha256"
    }
    if set(values) != expected:
        raise ValueError("runtime qualification builder fields drifted")
    provisional = object.__new__(RuntimeQualificationReceipt)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "qualification_receipt_sha256", "0" * 64)
    return RuntimeQualificationReceipt(
        **values,
        qualification_receipt_sha256=_digest(
            _RUNTIME_QUALIFICATION_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


def _parse_runtime_qualification_receipt_payload(
    value: object,
) -> RuntimeQualificationReceipt:
    expected = set(RuntimeQualificationReceipt.__dataclass_fields__)
    raw = _exact_mapping_fields(
        value,
        expected,
        label="runtime qualification receipt",
    )
    mode_rows = raw["mode_lineage"]
    if not isinstance(mode_rows, list):
        raise ValueError("runtime qualification mode lineage drifted")
    modes: list[RuntimeModeLineage] = []
    for item in mode_rows:
        row = _exact_mapping_fields(
            item,
            {
                "comparison_adapter_receipt_sha256",
                "mode_id",
                "site_placement_sha256",
                "state_fingerprint_sha256",
            },
            label="runtime qualification mode lineage",
        )
        modes.append(RuntimeModeLineage(**row))
    values = dict(raw)
    values["mode_lineage"] = tuple(modes)
    return RuntimeQualificationReceipt(**values)


def load_runtime_qualification_receipt(
    source: Path | str = DEFAULT_RUNTIME_QUALIFICATION_PATH,
) -> RuntimeQualificationReceipt:
    path = Path(source)
    raw = _read_regular(
        path,
        label="runtime qualification artifact",
        maximum_bytes=_MAX_RUNTIME_QUALIFICATION_BYTES,
    )
    parsed = _strict_json_loads(raw, require_canonical=True)
    return _parse_runtime_qualification_receipt_payload(parsed)


def write_runtime_qualification_receipt(
    receipt: RuntimeQualificationReceipt,
    destination: Path | str = DEFAULT_RUNTIME_QUALIFICATION_PATH,
) -> Path:
    """Publish one canonical qualification without overwrite or partials."""

    if not isinstance(receipt, RuntimeQualificationReceipt):
        raise TypeError("receipt must be RuntimeQualificationReceipt")
    path = Path(destination)
    _regular_directory(path.parent, label="qualification parent directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = {
        **receipt.payload_without_sha256(),
        "qualification_receipt_sha256": receipt.qualification_receipt_sha256,
    }
    raw = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if load_runtime_qualification_receipt(path) != receipt:
            raise RuntimeError("runtime qualification strict reload drifted")
    except BaseException:
        if published:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            try:
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def _validate_runtime_qualification_semantics(
    *,
    setup: Mapping[str, object],
    receipt: RuntimeQualificationReceipt,
    maximum_resident_memory_bytes: int,
) -> None:
    if not isinstance(receipt, RuntimeQualificationReceipt):
        raise TypeError("runtime qualification receipt is not typed")
    source_model = setup["source_model"]
    setup_modes = tuple(
        setup["logical_mode_ontology"]["ordered_modes"]
    )
    states = {
        row.mode_id: row.state_fingerprint_sha256
        for row in receipt.mode_lineage
    }
    expected_lineage = derive_runtime_mode_lineage(
        setup_modes=setup_modes,
        state_fingerprints_by_mode=states,
        hidden_width=receipt.hidden_width,
    )
    from fisher_graph.chunked_tied_lexical_nll import (
        chunked_tied_lexical_nll_kernel_receipt,
    )
    from fisher_graph.gemma3_chart_mode_surfaces import (
        factorized_attention_route_kernel_receipt,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        streaming_chart_kernel_receipt,
    )

    streaming_resources = StreamingChartResourceConfig(
        max_resident_bytes=maximum_resident_memory_bytes
    )
    if (
        receipt.model_id != source_model["model_id"]
        or receipt.model_revision != source_model["model_revision"]
        or receipt.source_parameter_count != source_model["parameter_count"]
        or receipt.hidden_width != source_model["hidden_width"]
        or receipt.source_model_fingerprint_before_compilation_sha256
        != source_model["source_model_fingerprint_sha256"]
        or receipt.source_execution_fingerprint_before_compilation_sha256
        != source_model["source_execution_fingerprint_sha256"]
        or receipt.mode_lineage != expected_lineage
        or receipt.lexical_kernel_receipt_sha256
        != chunked_tied_lexical_nll_kernel_receipt()["receipt_sha256"]
        or receipt.streaming_kernel_receipt_sha256
        != streaming_chart_kernel_receipt(streaming_resources)[
            "receipt_sha256"
        ]
        or receipt.factorized_route_kernel_receipt_sha256
        != factorized_attention_route_kernel_receipt()["receipt_sha256"]
    ):
        raise ValueError("runtime qualification differs from frozen contract")


def _load_frozen_runtime_qualification_from_setup(
    *,
    setup: Mapping[str, object],
    repository_root: Path,
) -> RuntimeQualificationReceipt:
    frozen = setup.get("frozen_runtime_qualification")
    if not isinstance(frozen, Mapping):
        raise RuntimeError(
            "setup lacks a frozen runtime qualification; live run is disabled"
        )
    required = {
        "authenticated_present",
        "file_sha256",
        "path",
        "qualification_receipt_sha256",
    }
    if (
        not required.issubset(frozen)
        or frozen["authenticated_present"] is not True
    ):
        raise ValueError("setup runtime qualification pin drifted")
    qualification_path = _relative_path(
        repository_root,
        str(frozen["path"]),
        label="runtime qualification artifact",
    )
    raw = _read_regular(
        qualification_path,
        label="runtime qualification artifact",
        maximum_bytes=_MAX_RUNTIME_QUALIFICATION_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != frozen["file_sha256"]:
        raise ValueError("runtime qualification artifact bytes drifted")
    frozen_receipt = _parse_runtime_qualification_receipt_payload(
        _strict_json_loads(raw, require_canonical=True)
    )
    if (
        frozen_receipt.qualification_receipt_sha256
        != frozen["qualification_receipt_sha256"]
    ):
        raise ValueError("runtime qualification semantic pin drifted")
    return frozen_receipt


def validate_runtime_qualification(
    *,
    preflight: ExecutionPreflightReceipt,
    receipt: RuntimeQualificationReceipt,
) -> None:
    """Authenticate the prompt-free compiled lineage before opening A."""

    preflight.validate_receipt()
    _validate_runtime_qualification_semantics(
        setup=preflight.setup,
        receipt=receipt,
        maximum_resident_memory_bytes=(
            preflight.binding.storage_budget.maximum_resident_memory_bytes
        ),
    )
    if preflight.binding.test_only_allow_nonproduction_shape:
        return
    frozen_receipt = _load_frozen_runtime_qualification_from_setup(
        setup=preflight.setup,
        repository_root=preflight.binding.repository_root,
    )
    if frozen_receipt != receipt:
        raise ValueError("live runtime differs from frozen qualification")


def derive_runtime_session_binding_sha256(
    *,
    preflight_receipt_sha256: str,
    runtime_qualification_receipt_sha256: str,
    source_model_fingerprint_sha256: str,
    source_execution_fingerprint_sha256: str,
    compiled_execution_fingerprint_sha256: str,
    tied_readout_binding_receipt_sha256: str,
    mode_lineage: Sequence[RuntimeModeLineage],
) -> str:
    """Bind one qualified immutable runtime to one static preflight."""

    for label, value in (
        ("preflight_receipt_sha256", preflight_receipt_sha256),
        (
            "runtime_qualification_receipt_sha256",
            runtime_qualification_receipt_sha256,
        ),
        ("source_model_fingerprint_sha256", source_model_fingerprint_sha256),
        (
            "source_execution_fingerprint_sha256",
            source_execution_fingerprint_sha256,
        ),
        (
            "compiled_execution_fingerprint_sha256",
            compiled_execution_fingerprint_sha256,
        ),
        (
            "tied_readout_binding_receipt_sha256",
            tied_readout_binding_receipt_sha256,
        ),
    ):
        _require_sha256(value, label=label)
    if not isinstance(mode_lineage, Sequence) or not mode_lineage:
        raise ValueError("runtime session mode lineage is empty")
    return _digest(
        _RUNTIME_SESSION_DOMAIN,
        {
            "compiled_execution_fingerprint_sha256": (
                compiled_execution_fingerprint_sha256
            ),
            "mode_lineage": _runtime_mode_lineage_payload(mode_lineage),
            "preflight_receipt_sha256": preflight_receipt_sha256,
            "runtime_qualification_receipt_sha256": (
                runtime_qualification_receipt_sha256
            ),
            "source_execution_fingerprint_sha256": (
                source_execution_fingerprint_sha256
            ),
            "source_model_fingerprint_sha256": (
                source_model_fingerprint_sha256
            ),
            "tied_readout_binding_receipt_sha256": (
                tied_readout_binding_receipt_sha256
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class RuntimeExecutionReceipt:
    model_id: str
    model_revision: str
    runtime_qualification_receipt_sha256: str
    runtime_session_binding_sha256: str
    source_model_fingerprint_before_sha256: str
    source_model_fingerprint_after_sha256: str
    source_execution_fingerprint_before_sha256: str
    source_execution_fingerprint_after_sha256: str
    compiled_execution_fingerprint_sha256: str
    source_model_mutated: bool
    source_parameter_count: int
    hidden_width: int
    mode_lineage: tuple[RuntimeModeLineage, ...]
    tied_readout_binding_receipt_sha256: str
    tied_lexical_lineage_sha256: str
    final_logit_softcap: float | None
    compiler_receipt_sha256: str
    lexical_kernel_receipt_sha256: str
    streaming_kernel_receipt_sha256: str
    factorized_route_kernel_receipt_sha256: str
    expected_chart_role_count: int
    expected_batches_per_chart_role: int
    expected_chart_capture_backbone_forward_count: int
    expected_chart_capture_terminal_cotangent_VJP_count: int
    expected_dense_parity_backbone_forward_count: int
    expected_dense_parity_terminal_cotangent_VJP_count: int
    expected_total_backbone_forward_count: int
    expected_total_terminal_cotangent_VJP_count: int
    expected_gradient_target_count_per_VJP: int
    observed_role_batch_counts: tuple[tuple[str, int], ...]
    observed_chart_capture_backbone_forward_count: int
    observed_chart_capture_terminal_cotangent_VJP_count: int
    observed_dense_parity_backbone_forward_count: int
    observed_dense_parity_terminal_cotangent_VJP_count: int
    observed_total_backbone_forward_count: int
    observed_total_terminal_cotangent_VJP_count: int
    observed_dense_lexical_parity_smoke_count: int
    observed_captured_example_count: int
    observed_supplemental_route_receipt_count: int
    contains_prompt_text: bool
    contains_token_ids: bool
    contains_raw_activations: bool
    contains_raw_gradients: bool
    contains_logits: bool
    runtime_receipt_sha256: str

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "compiler_receipt_sha256": self.compiler_receipt_sha256,
            "compiled_execution_fingerprint_sha256": (
                self.compiled_execution_fingerprint_sha256
            ),
            "contains_logits": self.contains_logits,
            "contains_prompt_text": self.contains_prompt_text,
            "contains_raw_activations": self.contains_raw_activations,
            "contains_raw_gradients": self.contains_raw_gradients,
            "contains_token_ids": self.contains_token_ids,
            "expected_batches_per_chart_role": (
                self.expected_batches_per_chart_role
            ),
            "expected_chart_role_count": self.expected_chart_role_count,
            "expected_chart_capture_backbone_forward_count": (
                self.expected_chart_capture_backbone_forward_count
            ),
            "expected_chart_capture_terminal_cotangent_VJP_count": (
                self.expected_chart_capture_terminal_cotangent_VJP_count
            ),
            "expected_dense_parity_backbone_forward_count": (
                self.expected_dense_parity_backbone_forward_count
            ),
            "expected_dense_parity_terminal_cotangent_VJP_count": (
                self.expected_dense_parity_terminal_cotangent_VJP_count
            ),
            "expected_gradient_target_count_per_VJP": (
                self.expected_gradient_target_count_per_VJP
            ),
            "expected_total_backbone_forward_count": (
                self.expected_total_backbone_forward_count
            ),
            "expected_total_terminal_cotangent_VJP_count": (
                self.expected_total_terminal_cotangent_VJP_count
            ),
            "factorized_route_kernel_receipt_sha256": (
                self.factorized_route_kernel_receipt_sha256
            ),
            "final_logit_softcap": self.final_logit_softcap,
            "hidden_width": self.hidden_width,
            "lexical_kernel_receipt_sha256": (
                self.lexical_kernel_receipt_sha256
            ),
            "mode_lineage": [
                {
                    "comparison_adapter_receipt_sha256": (
                        row.comparison_adapter_receipt_sha256
                    ),
                    "mode_id": row.mode_id,
                    "site_placement_sha256": row.site_placement_sha256,
                    "state_fingerprint_sha256": (
                        row.state_fingerprint_sha256
                    ),
                }
                for row in self.mode_lineage
            ],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "observed_captured_example_count": (
                self.observed_captured_example_count
            ),
            "observed_chart_capture_backbone_forward_count": (
                self.observed_chart_capture_backbone_forward_count
            ),
            "observed_chart_capture_terminal_cotangent_VJP_count": (
                self.observed_chart_capture_terminal_cotangent_VJP_count
            ),
            "observed_dense_parity_backbone_forward_count": (
                self.observed_dense_parity_backbone_forward_count
            ),
            "observed_dense_parity_terminal_cotangent_VJP_count": (
                self.observed_dense_parity_terminal_cotangent_VJP_count
            ),
            "observed_dense_lexical_parity_smoke_count": (
                self.observed_dense_lexical_parity_smoke_count
            ),
            "observed_role_batch_counts": [
                [role, count]
                for role, count in self.observed_role_batch_counts
            ],
            "observed_supplemental_route_receipt_count": (
                self.observed_supplemental_route_receipt_count
            ),
            "observed_total_backbone_forward_count": (
                self.observed_total_backbone_forward_count
            ),
            "observed_total_terminal_cotangent_VJP_count": (
                self.observed_total_terminal_cotangent_VJP_count
            ),
            "runtime_session_binding_sha256": (
                self.runtime_session_binding_sha256
            ),
            "runtime_qualification_receipt_sha256": (
                self.runtime_qualification_receipt_sha256
            ),
            "source_execution_fingerprint_after_sha256": (
                self.source_execution_fingerprint_after_sha256
            ),
            "source_execution_fingerprint_before_sha256": (
                self.source_execution_fingerprint_before_sha256
            ),
            "source_model_fingerprint_after_sha256": (
                self.source_model_fingerprint_after_sha256
            ),
            "source_model_fingerprint_before_sha256": (
                self.source_model_fingerprint_before_sha256
            ),
            "source_model_mutated": self.source_model_mutated,
            "source_parameter_count": self.source_parameter_count,
            "streaming_kernel_receipt_sha256": (
                self.streaming_kernel_receipt_sha256
            ),
            "tied_lexical_lineage_sha256": self.tied_lexical_lineage_sha256,
            "tied_readout_binding_receipt_sha256": (
                self.tied_readout_binding_receipt_sha256
            ),
        }

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("runtime model_id must be nonempty")
        if not isinstance(self.model_revision, str) or not self.model_revision:
            raise ValueError("runtime model_revision must be nonempty")
        for label in (
            "runtime_qualification_receipt_sha256",
            "runtime_session_binding_sha256",
            "source_model_fingerprint_before_sha256",
            "source_model_fingerprint_after_sha256",
            "source_execution_fingerprint_before_sha256",
            "source_execution_fingerprint_after_sha256",
            "compiled_execution_fingerprint_sha256",
            "tied_readout_binding_receipt_sha256",
            "tied_lexical_lineage_sha256",
            "compiler_receipt_sha256",
            "lexical_kernel_receipt_sha256",
            "streaming_kernel_receipt_sha256",
            "factorized_route_kernel_receipt_sha256",
            "runtime_session_binding_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        integer_fields = (
            "source_parameter_count",
            "hidden_width",
            "expected_chart_role_count",
            "expected_batches_per_chart_role",
            "expected_chart_capture_backbone_forward_count",
            "expected_chart_capture_terminal_cotangent_VJP_count",
            "expected_dense_parity_backbone_forward_count",
            "expected_dense_parity_terminal_cotangent_VJP_count",
            "expected_total_backbone_forward_count",
            "expected_total_terminal_cotangent_VJP_count",
            "expected_gradient_target_count_per_VJP",
            "observed_chart_capture_backbone_forward_count",
            "observed_chart_capture_terminal_cotangent_VJP_count",
            "observed_dense_parity_backbone_forward_count",
            "observed_dense_parity_terminal_cotangent_VJP_count",
            "observed_total_backbone_forward_count",
            "observed_total_terminal_cotangent_VJP_count",
            "observed_dense_lexical_parity_smoke_count",
            "observed_captured_example_count",
            "observed_supplemental_route_receipt_count",
        )
        if any(type(getattr(self, field)) is not int for field in integer_fields):
            raise TypeError("runtime execution integer inventory drifted")
        if (
            not isinstance(self.mode_lineage, tuple)
            or not isinstance(self.observed_role_batch_counts, tuple)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
                for row in self.observed_role_batch_counts
            )
        ):
            raise TypeError("runtime execution tuple inventory drifted")
        if (
            self.source_parameter_count != 268_098_176
            or self.hidden_width != 640
            or len(self.mode_lineage) != 38
            or len({row.mode_id for row in self.mode_lineage}) != 38
            or self.expected_chart_role_count != 3
            or self.expected_batches_per_chart_role != 45
            or self.expected_chart_capture_backbone_forward_count != 135
            or self.expected_chart_capture_terminal_cotangent_VJP_count != 135
            or self.expected_dense_parity_backbone_forward_count != 1
            or self.expected_dense_parity_terminal_cotangent_VJP_count != 1
            or self.expected_total_backbone_forward_count != 136
            or self.expected_total_terminal_cotangent_VJP_count != 136
            or self.expected_gradient_target_count_per_VJP != 92
            or self.observed_role_batch_counts
            != (
                (FIT_ROLE, 45),
                (DISCOVERY_ROLE, 45),
                (CONFIRMATION_ROLE, 45),
            )
            or self.observed_chart_capture_backbone_forward_count != 135
            or self.observed_chart_capture_terminal_cotangent_VJP_count != 135
            or self.observed_dense_parity_backbone_forward_count != 1
            or self.observed_dense_parity_terminal_cotangent_VJP_count != 1
            or self.observed_total_backbone_forward_count != 136
            or self.observed_total_terminal_cotangent_VJP_count != 136
            or self.observed_dense_lexical_parity_smoke_count != 1
            or self.observed_captured_example_count != 144
            or self.observed_supplemental_route_receipt_count != 3 * 18 * 48
            or self.contains_prompt_text is not False
            or self.contains_token_ids is not False
            or self.contains_raw_activations is not False
            or self.contains_raw_gradients is not False
            or self.contains_logits is not False
            or self.source_model_mutated is not False
            or self.source_model_fingerprint_before_sha256
            != self.source_model_fingerprint_after_sha256
            or self.source_execution_fingerprint_before_sha256
            != self.source_execution_fingerprint_after_sha256
        ):
            raise ValueError("runtime execution inventory drifted")
        if self.final_logit_softcap is not None and (
            not isinstance(self.final_logit_softcap, float)
            or not math.isfinite(self.final_logit_softcap)
            or self.final_logit_softcap <= 0.0
        ):
            raise ValueError("runtime final logit softcap drifted")
        if self.runtime_receipt_sha256 != _digest(
            _RUNTIME_RECEIPT_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("runtime execution receipt digest drifted")


def build_runtime_execution_receipt(
    **values: object,
) -> RuntimeExecutionReceipt:
    expected = {
        field_name
        for field_name in RuntimeExecutionReceipt.__dataclass_fields__
        if field_name != "runtime_receipt_sha256"
    }
    if set(values) != expected:
        raise ValueError("runtime receipt builder fields drifted")
    provisional = object.__new__(RuntimeExecutionReceipt)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "runtime_receipt_sha256", "0" * 64)
    return RuntimeExecutionReceipt(
        **values,
        runtime_receipt_sha256=_digest(
            _RUNTIME_RECEIPT_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


def _parse_runtime_execution_receipt_payload(
    value: object,
) -> RuntimeExecutionReceipt:
    expected = {
        field_name
        for field_name in RuntimeExecutionReceipt.__dataclass_fields__
    }
    raw = _exact_mapping_fields(
        value,
        expected,
        label="runtime execution receipt",
    )
    mode_rows = raw["mode_lineage"]
    role_counts = raw["observed_role_batch_counts"]
    if not isinstance(mode_rows, list) or not isinstance(role_counts, list):
        raise ValueError("runtime execution receipt collections drifted")
    modes: list[RuntimeModeLineage] = []
    for item in mode_rows:
        row = _exact_mapping_fields(
            item,
            {
                "comparison_adapter_receipt_sha256",
                "mode_id",
                "site_placement_sha256",
                "state_fingerprint_sha256",
            },
            label="runtime mode lineage",
        )
        modes.append(RuntimeModeLineage(**row))
    parsed_counts: list[tuple[str, int]] = []
    for item in role_counts:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or type(item[1]) is not int
        ):
            raise ValueError("runtime observed role counts drifted")
        parsed_counts.append((item[0], item[1]))
    values = dict(raw)
    values["mode_lineage"] = tuple(modes)
    values["observed_role_batch_counts"] = tuple(parsed_counts)
    return RuntimeExecutionReceipt(**values)


@dataclass(frozen=True, slots=True)
class CapturedModeRows:
    """One example's three chart tensors for one logical mode.

    Tensor values are deliberately opaque at module import time.  The guarded
    capture path validates them with torch only after static preflight.
    """

    mode_id: str
    context_values: Any
    comparison_values: Any
    generalized_contributions: Any
    supplemental_route_catalog_sha256: str | None = None
    supplemental_route_receipt_sha256: str | None = None
    supplemental_route_simplex_audit_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode_id, str) or not self.mode_id:
            raise ValueError("captured mode_id must be nonempty")
        route_values = (
            self.supplemental_route_catalog_sha256,
            self.supplemental_route_receipt_sha256,
        )
        if any(value is not None for value in route_values):
            if any(value is None for value in route_values):
                raise ValueError("supplemental route evidence must be paired")
            _require_sha256(route_values[0], label="route catalog")
            _require_sha256(route_values[1], label="route receipt")
        if self.supplemental_route_simplex_audit_sha256 is not None:
            if any(value is None for value in route_values):
                raise ValueError("route simplex audit requires route evidence")
            _require_sha256(
                self.supplemental_route_simplex_audit_sha256,
                label="route simplex audit",
            )


@dataclass(frozen=True, slots=True)
class CapturedExampleRows:
    family_id: str
    example_id: str
    token_count: int
    row_keys: tuple[tuple[str, str, int, int], ...]
    modes: tuple[CapturedModeRows, ...]
    context_fingerprint_sha256: str
    capture_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("captured family_id must be nonempty")
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("captured example_id must be nonempty")
        _require_positive_int(self.token_count, label="captured token_count")
        if self.token_count < 2 or len(self.row_keys) != self.token_count - 1:
            raise ValueError("captured example q/t row count drifted")
        expected = tuple(
            (self.family_id, self.example_id, query, query + 1)
            for query in range(self.token_count - 1)
        )
        if self.row_keys != expected:
            raise ValueError("captured example must bind exact q/t row keys")
        if (
            not isinstance(self.modes, tuple)
            or not self.modes
            or len({mode.mode_id for mode in self.modes}) != len(self.modes)
        ):
            raise ValueError("captured example mode inventory drifted")
        _require_sha256(
            self.context_fingerprint_sha256,
            label="context_fingerprint_sha256",
        )
        _require_sha256(
            self.capture_receipt_sha256,
            label="capture_receipt_sha256",
        )


@dataclass(frozen=True, slots=True)
class CapturedBatch:
    role: str
    examples: tuple[CapturedExampleRows, ...]
    lexical_nll_receipt_sha256: str
    execution_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.role not in (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE):
            raise ValueError("captured batch role is not a chart role")
        if (
            not isinstance(self.examples, tuple)
            or not self.examples
            or len({row.example_id for row in self.examples})
            != len(self.examples)
        ):
            raise ValueError("captured batch examples drifted")
        _require_sha256(
            self.lexical_nll_receipt_sha256,
            label="lexical_nll_receipt_sha256",
        )
        _require_sha256(
            self.execution_receipt_sha256,
            label="execution_receipt_sha256",
        )


@dataclass(frozen=True, slots=True)
class FirstACaptureCanaryReceipt:
    """Prompt-free hard-stop receipt for parity plus the first A batch only."""

    preflight_receipt_sha256: str
    runtime_qualification_receipt_sha256: str
    runtime_session_binding_sha256: str
    A_role_panel_binding_sha256: str
    dense_lexical_parity_receipt_sha256: str
    lexical_nll_receipt_sha256: str
    execution_receipt_sha256: str
    runtime_counters_receipt_sha256: str
    filesystem_before_catalog_sha256: str
    filesystem_after_catalog_sha256: str
    examples: tuple[tuple[str, str, int, int, str], ...]
    mode_ids: tuple[str, ...]
    route_evidence: tuple[tuple[str, str, str, str, str], ...]
    observed_role_batch_counts: tuple[tuple[str, int], ...]
    hidden_width: int
    gradient_target_count_per_VJP: int
    dense_parity_backbone_forward_count: int
    dense_parity_terminal_cotangent_VJP_count: int
    chart_backbone_forward_count: int
    chart_terminal_cotangent_VJP_count: int
    total_backbone_forward_count: int
    total_terminal_cotangent_VJP_count: int
    captured_example_count: int
    supplemental_route_receipt_count: int
    A_semantically_opened: bool
    B_semantically_opened: bool
    C_semantically_opened: bool
    D_semantically_opened: bool
    reserve_semantically_opened: bool
    runtime_closed_before_receipt: bool
    filesystem_entry_sets_unchanged: bool
    spool_artifact_count: int
    atlas_artifact_count: int
    transport_artifact_count: int
    graph_artifact_count: int
    published_output_count: int
    contains_prompt_text: bool
    contains_token_ids: bool
    contains_raw_tensors: bool
    canary_receipt_sha256: str

    def __post_init__(self) -> None:
        for label in (
            "preflight_receipt_sha256",
            "runtime_qualification_receipt_sha256",
            "runtime_session_binding_sha256",
            "A_role_panel_binding_sha256",
            "dense_lexical_parity_receipt_sha256",
            "lexical_nll_receipt_sha256",
            "execution_receipt_sha256",
            "runtime_counters_receipt_sha256",
            "filesystem_before_catalog_sha256",
            "filesystem_after_catalog_sha256",
            "canary_receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        expected_modes = (
            "mode.lexical.ingress",
            *(f"mode.mutation.{index}" for index in range(1, 37)),
            "mode.lexical.egress",
        )
        if self.mode_ids != expected_modes:
            raise ValueError("first-A canary mode inventory drifted")
        if (
            type(self.examples) is not tuple
            or any(type(row) is not tuple or len(row) != 5 for row in self.examples)
        ):
            raise TypeError("first-A canary examples must be exact tuples")
        if (
            len(self.examples) != 2
            or len({row[1] for row in self.examples}) != 2
            or len({row[4] for row in self.examples}) != 2
            or any(
                type(family_id) is not str
                or not family_id
                or type(example_id) is not str
                or not example_id
                or type(token_count) is not int
                or type(row_count) is not int
                or token_count != 96
                or row_count != 95
                for family_id, example_id, token_count, row_count, _ in self.examples
            )
        ):
            raise ValueError("first-A canary example inventory drifted")
        for *_, digest in self.examples:
            _require_sha256(digest, label="canary example capture receipt")
        if (
            type(self.route_evidence) is not tuple
            or any(
                type(row) is not tuple or len(row) != 5
                for row in self.route_evidence
            )
        ):
            raise TypeError("first-A canary routes must be exact tuples")
        expected_route_cells = tuple(
            (f"mode.mutation.{index}", example_id)
            for _, example_id, _, _, _ in self.examples
            for index in range(1, 37, 2)
        )
        observed_route_cells = tuple(
            (mode_id, example_id)
            for mode_id, example_id, _, _, _ in self.route_evidence
        )
        global_attention_modes = {
            "mode.mutation.11",
            "mode.mutation.23",
            "mode.mutation.35",
        }
        catalogs_by_example_and_kind: dict[
            tuple[str, str], list[str]
        ] = {}
        for mode_id, example_id, catalog, _, _ in self.route_evidence:
            attention_kind = (
                "global" if mode_id in global_attention_modes else "sliding_512"
            )
            catalogs_by_example_and_kind.setdefault(
                (example_id, attention_kind), []
            ).append(catalog)
        expected_catalog_groups = {
            (example_id, attention_kind)
            for _, example_id, _, _, _ in self.examples
            for attention_kind in ("sliding_512", "global")
        }
        if (
            len(self.route_evidence) != 36
            or observed_route_cells != expected_route_cells
            or len({row[3] for row in self.route_evidence}) != 36
            or set(catalogs_by_example_and_kind) != expected_catalog_groups
            or any(
                len(catalogs) != (3 if kind == "global" else 15)
                or len(set(catalogs)) != 1
                for (_, kind), catalogs in catalogs_by_example_and_kind.items()
            )
            or len(
                {
                    catalogs[0]
                    for catalogs in catalogs_by_example_and_kind.values()
                }
            )
            != 4
        ):
            raise ValueError("first-A canary route inventory drifted")
        for _, _, catalog, receipt, audit in self.route_evidence:
            _require_sha256(catalog, label="canary route catalog")
            _require_sha256(receipt, label="canary route receipt")
            _require_sha256(audit, label="canary route simplex audit")
        integer_fields = (
            "hidden_width",
            "gradient_target_count_per_VJP",
            "dense_parity_backbone_forward_count",
            "dense_parity_terminal_cotangent_VJP_count",
            "chart_backbone_forward_count",
            "chart_terminal_cotangent_VJP_count",
            "total_backbone_forward_count",
            "total_terminal_cotangent_VJP_count",
            "captured_example_count",
            "supplemental_route_receipt_count",
            "spool_artifact_count",
            "atlas_artifact_count",
            "transport_artifact_count",
            "graph_artifact_count",
            "published_output_count",
        )
        if any(type(getattr(self, field)) is not int for field in integer_fields):
            raise TypeError("first-A canary counters must be exact integers")
        if (
            type(self.observed_role_batch_counts) is not tuple
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or type(row[1]) is not int
                for row in self.observed_role_batch_counts
            )
        ):
            raise TypeError("first-A canary role counts must be exact tuples")
        if (
            self.observed_role_batch_counts
            != (
                (FIT_ROLE, 1),
                (DISCOVERY_ROLE, 0),
                (CONFIRMATION_ROLE, 0),
            )
            or self.hidden_width != 640
            or self.gradient_target_count_per_VJP != 92
            or self.dense_parity_backbone_forward_count != 1
            or self.dense_parity_terminal_cotangent_VJP_count != 1
            or self.chart_backbone_forward_count != 1
            or self.chart_terminal_cotangent_VJP_count != 1
            or self.total_backbone_forward_count != 2
            or self.total_terminal_cotangent_VJP_count != 2
            or self.captured_example_count != 2
            or self.supplemental_route_receipt_count != 36
        ):
            raise ValueError("first-A canary execution ledger drifted")
        expected_counter_payload = {
            "role_batch_counts": [
                [role, count] for role, count in self.observed_role_batch_counts
            ],
            "chart_backbone_forward_count": (
                self.chart_backbone_forward_count
            ),
            "chart_terminal_cotangent_vjp_count": (
                self.chart_terminal_cotangent_VJP_count
            ),
            "dense_parity_smoke_count": 1,
            "parity_backbone_forward_count": (
                self.dense_parity_backbone_forward_count
            ),
            "parity_terminal_cotangent_vjp_count": (
                self.dense_parity_terminal_cotangent_VJP_count
            ),
            "captured_example_count": self.captured_example_count,
            "supplemental_route_receipt_count": (
                self.supplemental_route_receipt_count
            ),
            "canonical_gradient_targets_per_vjp": 73,
            "auxiliary_gradient_targets_per_vjp": 19,
            "gradient_targets_per_vjp": self.gradient_target_count_per_VJP,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_activations": False,
            "contains_raw_gradients": False,
            "contains_logits": False,
        }
        if self.runtime_counters_receipt_sha256 != _digest(
            _FIRST_A_CANARY_DOMAIN,
            {"runtime_counters": expected_counter_payload},
        ):
            raise ValueError("first-A canary runtime counter binding drifted")
        if (
            self.A_semantically_opened is not True
            or self.B_semantically_opened is not False
            or self.C_semantically_opened is not False
            or self.D_semantically_opened is not False
            or self.reserve_semantically_opened is not False
            or self.runtime_closed_before_receipt is not True
            or self.filesystem_entry_sets_unchanged is not True
            or self.filesystem_before_catalog_sha256
            != self.filesystem_after_catalog_sha256
            or any(
                value != 0
                for value in (
                    self.spool_artifact_count,
                    self.atlas_artifact_count,
                    self.transport_artifact_count,
                    self.graph_artifact_count,
                    self.published_output_count,
                )
            )
            or self.contains_prompt_text is not False
            or self.contains_token_ids is not False
            or self.contains_raw_tensors is not False
        ):
            raise ValueError("first-A canary hard-stop/no-raw contract drifted")
        if self.canary_receipt_sha256 != _digest(
            _FIRST_A_CANARY_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("first-A canary receipt digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            field_name: (
                [
                    {
                        "family_id": family_id,
                        "example_id": example_id,
                        "token_count": token_count,
                        "row_count": row_count,
                        "capture_receipt_sha256": capture_sha,
                    }
                    for (
                        family_id,
                        example_id,
                        token_count,
                        row_count,
                        capture_sha,
                    ) in self.examples
                ]
                if field_name == "examples"
                else [
                    {
                        "mode_id": mode_id,
                        "example_id": example_id,
                        "route_catalog_sha256": catalog,
                        "route_receipt_sha256": receipt,
                        "probability_simplex_audit_sha256": audit,
                    }
                    for mode_id, example_id, catalog, receipt, audit in (
                        self.route_evidence
                    )
                ]
                if field_name == "route_evidence"
                else list(self.mode_ids)
                if field_name == "mode_ids"
                else getattr(self, field_name)
            )
            for field_name in self.__dataclass_fields__
            if field_name != "canary_receipt_sha256"
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.chart_native_first_A_capture_canary.v1",
            "format_version": 1,
            **self.payload_without_sha256(),
            "canary_receipt_sha256": self.canary_receipt_sha256,
        }


def build_first_A_capture_canary_receipt(
    **values: object,
) -> FirstACaptureCanaryReceipt:
    expected = {
        field_name
        for field_name in FirstACaptureCanaryReceipt.__dataclass_fields__
        if field_name != "canary_receipt_sha256"
    }
    if set(values) != expected:
        raise ValueError("first-A canary receipt builder fields drifted")
    provisional = object.__new__(FirstACaptureCanaryReceipt)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "canary_receipt_sha256", "0" * 64)
    return FirstACaptureCanaryReceipt(
        **values,
        canary_receipt_sha256=_digest(
            _FIRST_A_CANARY_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


class ChartNativeCaptureRuntime(Protocol):
    """Model-facing seam used after the static firewall has completed."""

    hidden_width: int
    mode_lineage: tuple[RuntimeModeLineage, ...]
    runtime_session_binding_sha256: str

    @property
    def counters(self) -> Any: ...

    def runtime_qualification_receipt(
        self,
    ) -> RuntimeQualificationReceipt: ...

    def runtime_execution_receipt(self) -> RuntimeExecutionReceipt: ...

    def run_dense_lexical_parity_smoke(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        raw_example: Mapping[str, object],
        role_manifest: Mapping[str, object],
    ) -> str: ...

    def capture_batch(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        role: str,
        raw_examples: Sequence[Mapping[str, object]],
        role_manifest: Mapping[str, object],
    ) -> CapturedBatch: ...

    def response_validate(
        self,
        *,
        preflight: ExecutionPreflightReceipt,
        raw_corpus: Mapping[str, object],
        role_manifest: Mapping[str, object],
        confirmed_edges: Sequence[Sequence[str]],
    ) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RoleCaptureReceipt:
    role: str
    runtime_session_binding_sha256: str
    role_panel_binding_sha256: str
    ordered_objective_row_catalog_sha256: str
    score_objective_receipt_sha256: str
    batch_axis_catalog_sha256: str
    mode_raw_artifacts: tuple[tuple[str, str, str, str], ...]
    lexical_nll_receipt_sha256s: tuple[str, ...]
    execution_receipt_sha256s: tuple[str, ...]
    example_capture_receipt_sha256s: tuple[tuple[str, str], ...]
    supplemental_route_evidence: tuple[tuple[str, str, str, str], ...]
    dense_lexical_parity_receipt_sha256: str | None
    batch_count: int
    row_count: int
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.role not in (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE):
            raise ValueError("role capture receipt has an invalid role")
        for label in (
            "runtime_session_binding_sha256",
            "role_panel_binding_sha256",
            "ordered_objective_row_catalog_sha256",
            "score_objective_receipt_sha256",
            "batch_axis_catalog_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if self.role == FIT_ROLE:
            _require_sha256(
                self.dense_lexical_parity_receipt_sha256,
                label="dense lexical parity receipt",
            )
        elif self.dense_lexical_parity_receipt_sha256 is not None:
            raise ValueError("only A may bind the dense lexical parity smoke")
        if (
            self.batch_count != 45
            or self.row_count != 24_732
            or len(self.mode_raw_artifacts) != 38
            or len(self.lexical_nll_receipt_sha256s) != 45
            or len(self.execution_receipt_sha256s) != 45
            or len(self.example_capture_receipt_sha256s) != 48
            or len(self.supplemental_route_evidence) != 18 * 48
        ):
            raise ValueError("role capture receipt production ledger drifted")
        mode_ids = tuple(item[0] for item in self.mode_raw_artifacts)
        expected_mode_ids = (
            "mode.lexical.ingress",
            *(f"mode.mutation.{index}" for index in range(1, 37)),
            "mode.lexical.egress",
        )
        if mode_ids != expected_mode_ids:
            raise ValueError("role capture receipt mode inventory drifted")
        example_ids = tuple(
            item[0] for item in self.example_capture_receipt_sha256s
        )
        if len(set(example_ids)) != 48:
            raise ValueError("role capture receipt example inventory drifted")
        for values in (
            self.lexical_nll_receipt_sha256s,
            self.execution_receipt_sha256s,
        ):
            for digest in values:
                _require_sha256(digest, label="capture batch receipt")
        for _, digest in self.example_capture_receipt_sha256s:
            _require_sha256(digest, label="example capture receipt")
        for _, artifact, rows, tensors in self.mode_raw_artifacts:
            _require_sha256(artifact, label="raw mode artifact")
            _require_sha256(rows, label="raw mode row catalog")
            _require_sha256(tensors, label="raw mode tensor catalog")
        for _, _, catalog, receipt in self.supplemental_route_evidence:
            _require_sha256(catalog, label="supplemental route catalog")
            _require_sha256(receipt, label="supplemental route receipt")
        route_cells = tuple(
            (mode_id, example_id)
            for mode_id, example_id, _, _ in self.supplemental_route_evidence
        )
        expected_route_cells = {
            (f"mode.mutation.{index}", example_id)
            for index in range(1, 37, 2)
            for example_id in example_ids
        }
        if (
            len(set(route_cells)) != 18 * 48
            or set(route_cells) != expected_route_cells
        ):
            raise ValueError(
                "supplemental route evidence inventory drifted"
            )
        if self.receipt_sha256 != _digest(
            _CAPTURE_DOMAIN,
            self.payload_without_sha256(),
        ):
            raise ValueError("role capture receipt digest drifted")

    def payload_without_sha256(self) -> dict[str, object]:
        return {
            "batch_count": self.batch_count,
            "batch_axis_catalog_sha256": self.batch_axis_catalog_sha256,
            "dense_lexical_parity_receipt_sha256": (
                self.dense_lexical_parity_receipt_sha256
            ),
            "example_capture_receipt_sha256s": dict(
                self.example_capture_receipt_sha256s
            ),
            "execution_receipt_sha256s": list(
                self.execution_receipt_sha256s
            ),
            "lexical_nll_receipt_sha256s": list(
                self.lexical_nll_receipt_sha256s
            ),
            "mode_raw_artifacts": [
                {
                    "artifact_sha256": artifact_sha256,
                    "mode_id": mode_id,
                    "row_catalog_sha256": row_catalog_sha256,
                    "tensor_catalog_sha256": tensor_catalog_sha256,
                }
                for (
                    mode_id,
                    artifact_sha256,
                    row_catalog_sha256,
                    tensor_catalog_sha256,
                ) in self.mode_raw_artifacts
            ],
            "ordered_objective_row_catalog_sha256": (
                self.ordered_objective_row_catalog_sha256
            ),
            "role": self.role,
            "runtime_session_binding_sha256": (
                self.runtime_session_binding_sha256
            ),
            "role_panel_binding_sha256": self.role_panel_binding_sha256,
            "row_count": self.row_count,
            "score_objective_receipt_sha256": (
                self.score_objective_receipt_sha256
            ),
            "supplemental_route_evidence": [
                {
                    "example_id": example_id,
                    "mode_id": mode_id,
                    "route_catalog_sha256": catalog_sha256,
                    "route_receipt_sha256": receipt_sha256,
                }
                for (
                    mode_id,
                    example_id,
                    catalog_sha256,
                    receipt_sha256,
                ) in self.supplemental_route_evidence
            ],
        }


@dataclass(frozen=True, slots=True)
class CapturedRole:
    raw_spools: tuple[Any, ...]
    receipt: RoleCaptureReceipt


def _canonical_role_row_keys(
    role_manifest: Mapping[str, object],
) -> tuple[tuple[str, str, int, int], ...]:
    rows: list[tuple[str, str, int, int]] = []
    for member in role_manifest["ordered_members"]:
        family_id = str(member["family_id"])
        example_id = str(member["example_id"])
        token_count = int(member["target_token_count"])
        rows.extend(
            (family_id, example_id, query, query + 1)
            for query in range(token_count - 1)
        )
    return tuple(rows)


def _capture_batch_plan(
    raw_corpus: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Exact 45-call order: three short pairs, then 42 long singletons."""

    examples = tuple(raw_corpus["examples"])
    by_length: dict[int, list[Mapping[str, object]]] = {
        length: [] for length, _ in _EXPECTED_PRODUCTION_SCHEDULE
    }
    for example in examples:
        by_length[int(example["target_token_count"])].append(example)
    if any(len(rows) != 6 for rows in by_length.values()):
        raise ValueError("opened role length buckets are not six-way balanced")
    plan: list[tuple[Mapping[str, object], ...]] = []
    short = by_length[96]
    plan.extend(tuple(short[index : index + 2]) for index in range(0, 6, 2))
    for length, _ in _EXPECTED_PRODUCTION_SCHEDULE[1:]:
        plan.extend((example,) for example in by_length[length])
    if len(plan) != 45:
        raise RuntimeError("capture batch plan did not produce exact 45 calls")
    return tuple(plan)


def _capture_batch_axis_catalog_sha256(
    *,
    role: str,
    batch_example_ids: Sequence[Sequence[str]],
    lexical_nll_receipt_sha256s: Sequence[str],
    execution_receipt_sha256s: Sequence[str],
    ordered_objective_row_catalog_sha256: str,
    score_objective_receipt_sha256: str,
) -> str:
    batches = tuple(tuple(batch) for batch in batch_example_ids)
    lexical = tuple(lexical_nll_receipt_sha256s)
    executions = tuple(execution_receipt_sha256s)
    if (
        role not in (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE)
        or len(batches) != 45
        or len(lexical) != 45
        or len(executions) != 45
        or any(not batch or len(batch) > 2 for batch in batches)
    ):
        raise ValueError("capture batch-axis inventory drifted")
    for digest in (
        *lexical,
        *executions,
        ordered_objective_row_catalog_sha256,
        score_objective_receipt_sha256,
    ):
        _require_sha256(digest, label="capture batch-axis receipt")
    return _digest(
        _CAPTURE_BATCH_AXIS_DOMAIN,
        {
            "batches": [
                {
                    "batch_ordinal": ordinal,
                    "example_ids": list(example_ids),
                    "execution_receipt_sha256": executions[ordinal],
                    "lexical_nll_receipt_sha256": lexical[ordinal],
                }
                for ordinal, example_ids in enumerate(batches)
            ],
            "ordered_objective_row_catalog_sha256": (
                ordered_objective_row_catalog_sha256
            ),
            "role": role,
            "score_objective_receipt_sha256": (
                score_objective_receipt_sha256
            ),
        },
    )


def _validate_capture_manifest_axis(
    capture: RoleCaptureReceipt,
    role_manifest: Mapping[str, object],
) -> None:
    expected_example_ids = tuple(
        str(member["example_id"])
        for member in role_manifest["ordered_members"]
    )
    observed_example_ids = tuple(
        example_id
        for example_id, _ in capture.example_capture_receipt_sha256s
    )
    if observed_example_ids != expected_example_ids:
        raise ValueError("capture example order differs from frozen manifest")
    batch_example_ids = tuple(
        tuple(str(row["example_id"]) for row in batch)
        for batch in _capture_batch_plan(
            {"examples": tuple(role_manifest["ordered_members"])}
        )
    )
    expected_catalog = _capture_batch_axis_catalog_sha256(
        role=capture.role,
        batch_example_ids=batch_example_ids,
        lexical_nll_receipt_sha256s=(
            capture.lexical_nll_receipt_sha256s
        ),
        execution_receipt_sha256s=capture.execution_receipt_sha256s,
        ordered_objective_row_catalog_sha256=(
            capture.ordered_objective_row_catalog_sha256
        ),
        score_objective_receipt_sha256=(
            capture.score_objective_receipt_sha256
        ),
    )
    if capture.batch_axis_catalog_sha256 != expected_catalog:
        raise ValueError("capture batch-axis catalog differs from manifest")


def _validate_captured_mode_tensor(
    value: Any,
    *,
    row_count: int,
    hidden_width: int,
    label: str,
) -> None:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.ndim != 2
        or tuple(value.shape) != (row_count, hidden_width)
        or not value.is_floating_point()
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be finite CPU [{row_count},{hidden_width}]"
        )


@dataclass(frozen=True, slots=True)
class _CapturedExampleFragment:
    family_id: str
    example_id: str
    token_count: int
    capture_receipt_sha256: str
    mode_artifacts: tuple[
        tuple[str, str, str, str | None, str | None], ...
    ]


def _append_raw_rows_in_fixed_chunks(
    writer: Any,
    *,
    row_keys: Sequence[Any],
    context_values: Any,
    comparison_values: Any,
    generalized_contributions: Any,
    row_chunk_size: int,
) -> None:
    for start in range(0, len(row_keys), row_chunk_size):
        end = min(len(row_keys), start + row_chunk_size)
        writer.append(
            row_keys[start:end],
            context_values[start:end],
            comparison_values[start:end],
            generalized_contributions[start:end],
        )


def _build_role_capture_receipt(
    *,
    role: str,
    runtime_session_binding_sha256: str,
    panel_binding_sha256: str,
    role_manifest: Mapping[str, object],
    spools: Sequence[Any],
    batch_example_ids: Sequence[Sequence[str]],
    lexical_receipts: Sequence[str],
    execution_receipts: Sequence[str],
    example_receipts: Sequence[tuple[str, str]],
    route_evidence: Sequence[tuple[str, str, str, str]],
    dense_lexical_parity_receipt_sha256: str | None,
    row_count: int,
) -> RoleCaptureReceipt:
    artifacts = tuple(
        (
            spool.mode_id,
            spool.artifact_sha256,
            spool.row_catalog_sha256,
            spool.tensor_catalog_sha256,
        )
        for spool in spools
    )
    ordered_rows = role_manifest["ordered_objective_row_catalog_sha256"]
    objective = role_manifest["score_objective_receipt_sha256"]
    values = {
        "role": role,
        "runtime_session_binding_sha256": runtime_session_binding_sha256,
        "role_panel_binding_sha256": panel_binding_sha256,
        "ordered_objective_row_catalog_sha256": ordered_rows,
        "score_objective_receipt_sha256": objective,
        "batch_axis_catalog_sha256": _capture_batch_axis_catalog_sha256(
            role=role,
            batch_example_ids=batch_example_ids,
            lexical_nll_receipt_sha256s=lexical_receipts,
            execution_receipt_sha256s=execution_receipts,
            ordered_objective_row_catalog_sha256=ordered_rows,
            score_objective_receipt_sha256=objective,
        ),
        "mode_raw_artifacts": artifacts,
        "lexical_nll_receipt_sha256s": tuple(lexical_receipts),
        "execution_receipt_sha256s": tuple(execution_receipts),
        "example_capture_receipt_sha256s": tuple(example_receipts),
        "supplemental_route_evidence": tuple(sorted(route_evidence)),
        "dense_lexical_parity_receipt_sha256": (
            dense_lexical_parity_receipt_sha256
        ),
        "batch_count": len(lexical_receipts),
        "row_count": row_count,
    }
    provisional = object.__new__(RoleCaptureReceipt)
    for field_name, field_value in values.items():
        object.__setattr__(provisional, field_name, field_value)
    object.__setattr__(provisional, "receipt_sha256", "0" * 64)
    return RoleCaptureReceipt(
        **values,
        receipt_sha256=_digest(
            _CAPTURE_DOMAIN,
            provisional.payload_without_sha256(),
        ),
    )


def capture_role_to_raw_spools(
    *,
    preflight: ExecutionPreflightReceipt,
    runtime: ChartNativeCaptureRuntime,
    role: str,
    raw_corpus: Mapping[str, object],
    destination_directory: Path | str,
    dense_lexical_parity_receipt_sha256: str | None,
) -> CapturedRole:
    """Capture one role into 38 strict streaming spools.

    Runtime calls follow the frozen 45-batch schedule.  The six short examples
    are buffered only until their three batch-two calls complete; all rows are
    then appended in the manifest's canonical family-major q/t order.
    """

    preflight.validate_receipt()
    if role not in (FIT_ROLE, DISCOVERY_ROLE, CONFIRMATION_ROLE):
        raise ValueError("only A/B/C roles have chart captures")
    if role == FIT_ROLE:
        _require_sha256(
            dense_lexical_parity_receipt_sha256,
            label="dense lexical parity receipt",
        )
    elif dense_lexical_parity_receipt_sha256 is not None:
        raise ValueError("dense lexical parity smoke belongs only to A")

    import torch
    from fisher_graph.shared_chart_energy_transport import (
        ChartRowKey,
        build_within_example_row_shuffle,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        RawModeRowSpoolWriter,
        StreamingChartResourceConfig,
        open_raw_mode_row_spool,
    )

    if not isinstance(runtime.hidden_width, int) or runtime.hidden_width <= 0:
        raise ValueError("runtime hidden width is invalid")
    setup_modes = tuple(
        preflight.setup["logical_mode_ontology"]["ordered_modes"]
    )
    mode_ids = tuple(str(item["mode_id"]) for item in setup_modes)
    if tuple(item.mode_id for item in runtime.mode_lineage) != mode_ids:
        raise ValueError("runtime mode lineage differs from frozen setup")
    if (
        not preflight.binding.test_only_allow_nonproduction_shape
        and runtime.hidden_width != 640
    ):
        raise ValueError("production runtime hidden width must be 640")

    role_manifest = preflight.fresh_manifest["roles"][role]
    panel_binding = role_panel_binding_sha256(preflight.fresh_manifest, role)
    canonical_keys = _canonical_role_row_keys(role_manifest)
    if len(canonical_keys) != int(role_manifest["prediction_row_count"]):
        raise ValueError("canonical role row count differs from manifest")
    core_keys = tuple(ChartRowKey(*key) for key in canonical_keys)
    destination = Path(destination_directory)
    _regular_directory(destination, label="role capture destination")
    resources = StreamingChartResourceConfig(
        max_resident_bytes=(
            preflight.binding.storage_budget.maximum_resident_memory_bytes
        )
    )
    fragment_directory = destination / "sealed-example-fragments"
    fragment_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    writers: dict[str, Any] = {}
    sealed_final_paths: list[Path] = []
    try:
        captured: dict[str, _CapturedExampleFragment] = {}
        lexical_receipts: list[str] = []
        execution_receipts: list[str] = []
        batch_example_ids: list[tuple[str, ...]] = []
        capture_plan = _capture_batch_plan(raw_corpus)
        for batch_examples in capture_plan:
            observed = runtime.capture_batch(
                preflight=preflight,
                role=role,
                raw_examples=batch_examples,
                role_manifest=role_manifest,
            )
            if not isinstance(observed, CapturedBatch) or observed.role != role:
                raise TypeError("runtime returned an invalid captured batch")
            expected_ids = tuple(
                str(example["example_id"]) for example in batch_examples
            )
            batch_example_ids.append(expected_ids)
            if tuple(row.example_id for row in observed.examples) != expected_ids:
                raise ValueError("runtime captured examples out of batch order")
            lexical_receipts.append(observed.lexical_nll_receipt_sha256)
            execution_receipts.append(observed.execution_receipt_sha256)
            for row in observed.examples:
                if row.example_id in captured:
                    raise ValueError("runtime captured an example twice")
                if tuple(item.mode_id for item in row.modes) != mode_ids:
                    raise ValueError("captured example mode order drifted")
                row_keys = tuple(ChartRowKey(*key) for key in row.row_keys)
                mode_artifacts: list[
                    tuple[str, str, str, str | None, str | None]
                ] = []
                capture_ordinal = len(captured)
                for mode_ordinal, mode in enumerate(row.modes):
                    for label, tensor in (
                        ("context", mode.context_values),
                        ("comparison", mode.comparison_values),
                        ("contribution", mode.generalized_contributions),
                    ):
                        _validate_captured_mode_tensor(
                            tensor,
                            row_count=row.token_count - 1,
                            hidden_width=runtime.hidden_width,
                            label=f"{mode.mode_id} {label}",
                        )
                    fragment_path = fragment_directory / (
                        f"example-{capture_ordinal:02d}-mode-"
                        f"{mode_ordinal:02d}.raw-chart"
                    )
                    fragment_writer = RawModeRowSpoolWriter.create(
                        fragment_path,
                        role=role,
                        panel_binding_sha256=panel_binding,
                        mode_id=mode.mode_id,
                        row_keys=row_keys,
                        hidden_width=runtime.hidden_width,
                        resources=resources,
                    )
                    try:
                        _append_raw_rows_in_fixed_chunks(
                            fragment_writer,
                            row_keys=row_keys,
                            context_values=mode.context_values,
                            comparison_values=mode.comparison_values,
                            generalized_contributions=(
                                mode.generalized_contributions
                            ),
                            row_chunk_size=resources.row_chunk_size,
                        )
                        fragment = fragment_writer.seal()
                    except BaseException:
                        fragment_writer.abort()
                        raise
                    if (
                        fragment.mode_id != mode.mode_id
                        or fragment.row_keys != row_keys
                        or fragment.role != role
                        or fragment.panel_binding_sha256 != panel_binding
                    ):
                        raise RuntimeError(
                            "sealed example fragment strict reload drifted"
                        )
                    mode_artifacts.append(
                        (
                            mode.mode_id,
                            str(fragment.path),
                            fragment.artifact_sha256,
                            mode.supplemental_route_catalog_sha256,
                            mode.supplemental_route_receipt_sha256,
                        )
                    )
                captured[row.example_id] = _CapturedExampleFragment(
                    family_id=row.family_id,
                    example_id=row.example_id,
                    token_count=row.token_count,
                    capture_receipt_sha256=row.capture_receipt_sha256,
                    mode_artifacts=tuple(mode_artifacts),
                )
            # Python loop locals otherwise keep the final example's three
            # h640 tensors alive across the next model call.
            del row, mode, tensor
            del observed
            if (
                _peak_resident_set_bytes()
                > preflight.binding.storage_budget.maximum_resident_memory_bytes
                or _unique_regular_bytes(fragment_directory)
                > preflight.binding.storage_budget.fit_tensor_temp_bytes
            ):
                raise RuntimeError(
                    "capture fragment phase exceeded frozen resource budget"
                )

        canonical_example_ids = tuple(
            str(member["example_id"])
            for member in role_manifest["ordered_members"]
        )
        if set(captured) != set(canonical_example_ids):
            raise ValueError("role capture did not cover the exact manifest")
        for mode_id in mode_ids:
            writers[mode_id] = RawModeRowSpoolWriter.create(
                destination / f"{mode_id.replace('.', '--')}.raw-chart",
                role=role,
                panel_binding_sha256=panel_binding,
                mode_id=mode_id,
                row_keys=core_keys,
                hidden_width=runtime.hidden_width,
                resources=resources,
            )
        route_evidence: list[tuple[str, str, str, str]] = []
        example_receipts: list[tuple[str, str]] = []
        for example_id in canonical_example_ids:
            row = captured.pop(example_id)
            example_receipts.append(
                (example_id, row.capture_receipt_sha256)
            )
            expected_row_keys = tuple(
                ChartRowKey(
                    row.family_id,
                    row.example_id,
                    query,
                    query + 1,
                )
                for query in range(row.token_count - 1)
            )
            if tuple(item[0] for item in row.mode_artifacts) != mode_ids:
                raise RuntimeError("captured fragment mode order drifted")
            for (
                mode_id,
                fragment_path_text,
                fragment_artifact_sha256,
                route_catalog_sha256,
                route_receipt_sha256,
            ) in row.mode_artifacts:
                fragment_path = Path(fragment_path_text)
                fragment = open_raw_mode_row_spool(
                    fragment_path,
                    max_artifact_bytes=resources.max_artifact_bytes,
                )
                if (
                    fragment.artifact_sha256 != fragment_artifact_sha256
                    or fragment.mode_id != mode_id
                    or fragment.row_keys != expected_row_keys
                    or fragment.role != role
                    or fragment.panel_binding_sha256 != panel_binding
                ):
                    raise RuntimeError(
                        "captured fragment changed before canonical assembly"
                    )
                for chunk in fragment.iter_float64_chunks(
                    chunk_size=resources.row_chunk_size
                ):
                    _append_raw_rows_in_fixed_chunks(
                        writers[mode_id],
                        row_keys=chunk.row_keys,
                        context_values=chunk.tensors["context_values"],
                        comparison_values=chunk.tensors["comparison_values"],
                        generalized_contributions=chunk.tensors[
                            "generalized_contributions"
                        ],
                        row_chunk_size=resources.row_chunk_size,
                    )
                fragment_stat = os.lstat(fragment_path)
                if not stat.S_ISREG(fragment_stat.st_mode):
                    raise RuntimeError("captured fragment is not a regular file")
                os.unlink(fragment_path)
                if route_catalog_sha256 is not None:
                    route_evidence.append(
                        (
                            mode_id,
                            example_id,
                            route_catalog_sha256,
                            route_receipt_sha256,
                        )
                    )
            del row
        os.rmdir(fragment_directory)
        descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        sealed_spools: list[Any] = []
        for mode_id in mode_ids:
            spool = writers[mode_id].seal()
            sealed_final_paths.append(Path(spool.path))
            sealed_spools.append(spool)
        spools = tuple(sealed_spools)
        receipt = _build_role_capture_receipt(
            role=role,
            runtime_session_binding_sha256=(
                runtime.runtime_session_binding_sha256
            ),
            panel_binding_sha256=panel_binding,
            role_manifest=role_manifest,
            spools=spools,
            batch_example_ids=batch_example_ids,
            lexical_receipts=lexical_receipts,
            execution_receipts=execution_receipts,
            example_receipts=example_receipts,
            route_evidence=route_evidence,
            dense_lexical_parity_receipt_sha256=(
                dense_lexical_parity_receipt_sha256
            ),
            row_count=len(canonical_keys),
        )
        return CapturedRole(raw_spools=spools, receipt=receipt)
    except BaseException:
        for writer in writers.values():
            try:
                writer.abort()
            except BaseException:
                pass
        for path in reversed(sealed_final_paths):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        if fragment_directory.exists() and not fragment_directory.is_symlink():
            _safe_remove_tree(fragment_directory, allowed_parent=destination)
        descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise

@dataclass(frozen=True, slots=True)
class FitPhaseArtifacts:
    atlas: Any
    applied_stores: tuple[Any, ...]
    transport_artifacts: tuple[Any, ...]
    stage_receipt: FrozenStageReceipt
    atlas_artifact_path: str
    atlas_artifact_file_sha256: str
    transport_catalog_sha256: str
    mode_lineage_catalog_sha256: str
    streaming_kernel_receipt_sha256: str
    resource_observations: tuple[PhaseResourceObservation, ...]


def _streaming_kernel_receipt_sha256(receipt: Mapping[str, object]) -> str:
    observed = receipt.get("receipt_sha256")
    _require_sha256(observed, label="streaming kernel receipt_sha256")
    return observed


def _core_mode_lineage_catalog_sha256(mode_specs: Sequence[Any]) -> str:
    if len(mode_specs) != 38:
        raise ValueError("mode lineage catalog must contain exactly 38 modes")
    payloads = [mode.canonical_payload() for mode in mode_specs]
    if len({payload["mode_id"] for payload in payloads}) != 38:
        raise ValueError("mode lineage catalog mode IDs collide")
    return _digest(_MODE_LINEAGE_DOMAIN, payloads)


def fit_A_chart_atlas_and_transports(
    *,
    preflight: ExecutionPreflightReceipt,
    captured: CapturedRole,
    mode_specs: Sequence[Any],
    destination_directory: Path | str,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> FitPhaseArtifacts:
    """Fit the shared A atlas and every ordered directional transport."""

    preflight.validate_receipt()
    if captured.receipt.role != FIT_ROLE:
        raise ValueError("atlas and transports may be fit only on A")
    if captured.receipt.role_panel_binding_sha256 != role_panel_binding_sha256(
        preflight.fresh_manifest,
        FIT_ROLE,
    ):
        raise ValueError("A capture panel binding drifted")
    mode_lineage_catalog_sha = _core_mode_lineage_catalog_sha256(mode_specs)

    from fisher_graph.shared_chart_energy_transport import (
        save_shared_chart_atlas,
        transport_coefficient_storage_bytes,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        apply_shared_chart_atlas_streaming,
        fit_all_ordered_directed_chart_transports_streaming,
        fit_shared_chart_atlas_streaming,
        open_streaming_applied_chart_store,
        streaming_chart_kernel_receipt,
    )

    destination = Path(destination_directory)
    _regular_directory(destination, label="A fit destination")
    applied_directory = destination / "applied"
    transport_directory = destination / "transports"
    for path in (applied_directory, transport_directory):
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    resources = StreamingChartResourceConfig(
        max_resident_bytes=(
            preflight.binding.storage_budget.maximum_resident_memory_bytes
        )
    )
    kernel = streaming_chart_kernel_receipt(resources)
    kernel_sha = _streaming_kernel_receipt_sha256(kernel)
    observations: list[PhaseResourceObservation] = [
        observe_phase_resources(
            preflight.binding,
            phase_id="A_raw_capture_sealed",
        )
    ]
    atlas = fit_shared_chart_atlas_streaming(
        captured.raw_spools,
        resources=resources,
    )
    atlas_path = destination / "shared-chart-atlas.bundle"
    save_shared_chart_atlas(atlas, atlas_path)
    atlas_file_sha = hashlib.sha256(
        _read_regular(
            atlas_path,
            label="saved shared chart atlas",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
    ).hexdigest()
    observations.append(
        observe_phase_resources(
            preflight.binding,
            phase_id="A_shared_atlas_frozen",
        )
    )
    stores: list[Any] = []
    for spool in captured.raw_spools:
        store = apply_shared_chart_atlas_streaming(
            atlas,
            spool,
            applied_directory / f"{spool.mode_id.replace('.', '--')}.applied-chart",
            resources=resources,
        )
        # Reopen through the strict loader before it is admitted downstream.
        reloaded = open_streaming_applied_chart_store(
            store.path,
            max_artifact_bytes=resources.max_artifact_bytes,
        )
        if (
            reloaded.artifact_sha256 != store.artifact_sha256
            or reloaded.vector_catalog_sha256 != store.vector_catalog_sha256
        ):
            raise RuntimeError("strict applied chart store reload drifted")
        stores.append(reloaded)
    if tuple(store.mode_id for store in stores) != tuple(
        spool.mode_id for spool in captured.raw_spools
    ):
        raise RuntimeError("A applied store mode order drifted")
    observations.append(
        observe_phase_resources(
            preflight.binding,
            phase_id="A_applied_chart_stores_frozen",
        )
    )
    coefficient_bytes = transport_coefficient_storage_bytes(
        atlas.config,
        mode_count=len(stores),
    )
    if coefficient_bytes != _PRODUCTION_TRANSPORT_BYTES:
        raise RuntimeError("ordered transport coefficient byte count drifted")
    transports = fit_all_ordered_directed_chart_transports_streaming(
        atlas,
        stores,
        transport_directory,
        resources=resources,
        progress=progress,
    )
    if len(transports) != 38 * 37:
        raise RuntimeError("A fit did not produce all 1,406 transports")
    transport_payload = [
        {
            "byte_count": item.byte_count,
            "source_mode_id": item.source_mode_id,
            "target_mode_id": item.target_mode_id,
            "transport_sha256": item.transport_sha256,
        }
        for item in transports
    ]
    transport_catalog_sha = _digest(
        _TRANSPORT_CATALOG_DOMAIN,
        transport_payload,
    )
    observations.append(
        observe_phase_resources(
            preflight.binding,
            phase_id="A_all_directional_transports_frozen",
            completed_pair_count=len(transports),
            total_pair_count=38 * 37,
        )
    )
    stage = build_frozen_stage_receipt(
        role=FIT_ROLE,
        preflight_receipt_sha256=preflight.receipt_sha256,
        role_panel_binding_sha256_value=(
            captured.receipt.role_panel_binding_sha256
        ),
        ordered_objective_row_catalog_sha256=(
            captured.receipt.ordered_objective_row_catalog_sha256
        ),
        score_objective_receipt_sha256=(
            captured.receipt.score_objective_receipt_sha256
        ),
        prior_stage_receipt_sha256=None,
        artifact_sha256s={
            "A_capture_receipt": captured.receipt.receipt_sha256,
            "A_shared_chart_atlas": atlas.atlas_sha256,
            "A_shared_chart_atlas_file": atlas_file_sha,
            "A_streaming_kernel": kernel_sha,
            "A_transport_catalog": transport_catalog_sha,
            "A_mode_lineage_catalog": mode_lineage_catalog_sha,
        },
        primary_eligible_example_catalog_sha256=None,
    )
    return FitPhaseArtifacts(
        atlas=atlas,
        applied_stores=tuple(stores),
        transport_artifacts=tuple(transports),
        stage_receipt=stage,
        atlas_artifact_path=str(atlas_path),
        atlas_artifact_file_sha256=atlas_file_sha,
        transport_catalog_sha256=transport_catalog_sha,
        mode_lineage_catalog_sha256=mode_lineage_catalog_sha,
        streaming_kernel_receipt_sha256=kernel_sha,
        resource_observations=tuple(observations),
    )


@dataclass(frozen=True, slots=True)
class RoleEvaluationArtifacts:
    role: str
    applied_stores: tuple[Any, ...]
    example_stats: tuple[Any, ...]
    directed_transport_stats: tuple[Any, ...]
    row_shuffle_catalog_sha256: str
    evaluation_catalog_sha256: str
    resource_observations: tuple[PhaseResourceObservation, ...]


def _contiguous_example_spans(
    row_keys: Sequence[Any],
) -> tuple[tuple[str, str, int, int], ...]:
    spans: list[tuple[str, str, int, int]] = []
    start = 0
    while start < len(row_keys):
        first = row_keys[start]
        family_id = first.family_id
        example_id = first.example_id
        end = start + 1
        while end < len(row_keys) and (
            row_keys[end].family_id,
            row_keys[end].example_id,
        ) == (family_id, example_id):
            end += 1
        expected = tuple(range(end - start))
        if tuple(
            row.prediction_query_position for row in row_keys[start:end]
        ) != expected or tuple(
            row.target_label_position for row in row_keys[start:end]
        ) != tuple(query + 1 for query in expected):
            raise ValueError("streaming store example q/t rows drifted")
        spans.append((family_id, example_id, start, end))
        start = end
    identities = tuple((family, example) for family, example, _, _ in spans)
    if len(set(identities)) != len(identities):
        raise ValueError("streaming store splits one example into multiple spans")
    return tuple(spans)


def _example_row_catalog_sha256(
    *,
    role: str,
    row_keys: Sequence[Any],
) -> str:
    return _digest(
        _EXAMPLE_ROW_DOMAIN,
        {
            "ordered_row_key_fields": [
                "role",
                "family_id",
                "example_id",
                "prediction_query_position",
                "target_label_position",
            ],
            "rows": [
                [
                    role,
                    row.family_id,
                    row.example_id,
                    row.prediction_query_position,
                    row.target_label_position,
                ]
                for row in row_keys
            ],
        },
    )


def _example_chart_stats_from_store(
    *,
    store: Any,
    atlas_sha256: str,
    comparison_abi_sha256: str,
    supplemental_route_by_example: Mapping[str, tuple[str, str]],
) -> tuple[Any, ...]:
    import torch
    from fisher_graph.chart_energy_neighbor_graph import ChartExampleStats
    from fisher_graph.chunked_tied_lexical_nll import tensor_stream_sha256

    fields = (
        "features",
        "generalized_contributions",
        "memberships",
        "weighted_activation_energy",
        "weighted_chart_tangent_energy",
        "weighted_chart_orthogonal_energy",
        "weighted_fisher_energy",
    )
    rows: list[Any] = []
    for family_id, example_id, start, end in _contiguous_example_spans(
        store.row_keys
    ):
        tensors = {
            field: store.read_slice(field, start, end, float64=True)
            for field in fields
        }
        energy = tensors["weighted_fisher_energy"]
        tangent = tensors["weighted_chart_tangent_energy"]
        orthogonal = tensors["weighted_chart_orthogonal_energy"]
        if not torch.allclose(
            tangent + orthogonal,
            energy,
            atol=1.0e-9,
            rtol=1.0e-9,
        ):
            raise RuntimeError("example tangent/orthogonal Fisher energy drifted")
        example_row_keys = store.row_keys[start:end]
        row_catalog = _example_row_catalog_sha256(
            role=store.role,
            row_keys=example_row_keys,
        )
        tensor_sha256s = {
            field: tensor_stream_sha256(tensor)
            for field, tensor in tensors.items()
        }
        vector_catalog = _digest(
            _ENERGY_VECTOR_DOMAIN,
            {
                "atlas_sha256": atlas_sha256,
                "example_id": example_id,
                "family_id": family_id,
                "mode_id": store.mode_id,
                "role": store.role,
                "row_catalog_sha256": row_catalog,
                "source_raw_artifact_sha256": (
                    store.source_raw_artifact_sha256
                ),
                "source_tensor_catalog_sha256": (
                    store.source_tensor_catalog_sha256
                ),
                "store_artifact_sha256": store.artifact_sha256,
                "store_vector_catalog_sha256": store.vector_catalog_sha256,
                "tensor_sha256s": tensor_sha256s,
            },
        )
        route = supplemental_route_by_example.get(example_id)
        rows.append(
            ChartExampleStats(
                family_id=family_id,
                example_id=example_id,
                mode_id=store.mode_id,
                comparison_abi_sha256=comparison_abi_sha256,
                chart_atlas_sha256=atlas_sha256,
                aligned_row_catalog_sha256=row_catalog,
                chart_energy_vector_catalog_sha256=vector_catalog,
                row_count=end - start,
                occupied_row_count=int((energy > 0.0).sum()),
                minimum_row_fisher_energy=float(energy.min()),
                weighted_activation_energy=float(
                    tensors["weighted_activation_energy"].sum()
                ),
                weighted_chart_tangent_energy=float(tangent.sum()),
                weighted_chart_orthogonal_energy=float(orthogonal.sum()),
                weighted_fisher_energy=float(energy.sum()),
                supplemental_route_catalog_sha256=(
                    None if route is None else route[0]
                ),
                supplemental_route_receipt_sha256=(
                    None if route is None else route[1]
                ),
            )
        )
    return tuple(rows)


def _role_evaluation_catalog_sha256(
    *,
    role: str,
    atlas_sha256: str,
    capture_receipt_sha256: str,
    fit_transport_catalog_sha256: str,
    row_shuffle_catalog_sha256: str,
    example_stats: Sequence[Any],
    directed_transport_stats: Sequence[Any],
) -> str:
    """Bind one held-out role using the core's canonical row order."""

    example_rows = tuple(
        sorted(
            example_stats,
            key=lambda row: (row.family_id, row.example_id, row.mode_id),
        )
    )
    directed_rows = tuple(
        sorted(
            directed_transport_stats,
            key=lambda row: (
                row.family_id,
                row.example_id,
                row.source_mode_id,
                row.target_mode_id,
            ),
        )
    )
    example_keys = tuple(
        (row.family_id, row.example_id, row.mode_id) for row in example_rows
    )
    directed_keys = tuple(
        (
            row.family_id,
            row.example_id,
            row.source_mode_id,
            row.target_mode_id,
        )
        for row in directed_rows
    )
    if (
        len(example_rows) != 38 * 48
        or len(set(example_keys)) != len(example_keys)
        or len(directed_rows) != 38 * 37 * 48
        or len(set(directed_keys)) != len(directed_keys)
    ):
        raise ValueError("role evaluation row catalog is incomplete")
    return _digest(
        _ROLE_EVALUATION_DOMAIN,
        {
            "atlas_sha256": atlas_sha256,
            "capture_receipt_sha256": capture_receipt_sha256,
            "directed_stats_sha256s": [row.sha256 for row in directed_rows],
            "example_stats_sha256s": [row.sha256 for row in example_rows],
            "fit_transport_catalog_sha256": fit_transport_catalog_sha256,
            "role": role,
            "row_shuffle_catalog_sha256": row_shuffle_catalog_sha256,
        },
    )


def evaluate_heldout_role_transports(
    *,
    preflight: ExecutionPreflightReceipt,
    captured: CapturedRole,
    fit: FitPhaseArtifacts,
    destination_directory: Path | str,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> RoleEvaluationArtifacts:
    """Apply the frozen A atlas and transports to all eight B or C examples."""

    preflight.validate_receipt()
    role = captured.receipt.role
    if role not in (DISCOVERY_ROLE, CONFIRMATION_ROLE):
        raise ValueError("heldout transport evaluation role must be B or C")
    if fit.stage_receipt.role != FIT_ROLE:
        raise ValueError("heldout evaluation requires a frozen A fit")

    from fisher_graph.chart_energy_neighbor_graph import (
        DirectedChartTransportStats,
        FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL,
        canonical_carrier_comparison_abi,
    )
    from fisher_graph.shared_chart_energy_transport import (
        build_within_example_row_shuffle,
        load_directed_chart_transport,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        StreamingChartResourceConfig,
        apply_shared_chart_atlas_streaming,
        evaluate_directed_chart_transport_streaming,
        open_streaming_applied_chart_store,
    )

    destination = Path(destination_directory)
    _regular_directory(destination, label=f"{role} evaluation destination")
    resources = StreamingChartResourceConfig(
        max_resident_bytes=(
            preflight.binding.storage_budget.maximum_resident_memory_bytes
        )
    )
    stores: list[Any] = []
    for spool in captured.raw_spools:
        store = apply_shared_chart_atlas_streaming(
            fit.atlas,
            spool,
            destination / f"{spool.mode_id.replace('.', '--')}.applied-chart",
            resources=resources,
        )
        strict = open_streaming_applied_chart_store(
            store.path,
            max_artifact_bytes=resources.max_artifact_bytes,
        )
        if strict.artifact_sha256 != store.artifact_sha256:
            raise RuntimeError("heldout applied store strict reload drifted")
        stores.append(strict)
    if len(stores) != 38:
        raise RuntimeError("heldout role did not produce 38 applied stores")
    observations = [
        observe_phase_resources(
            preflight.binding,
            phase_id=f"{role}_applied_chart_stores_frozen",
        )
    ]
    comparison_abi_sha = canonical_carrier_comparison_abi().sha256
    route_by_mode: dict[str, dict[str, tuple[str, str]]] = {
        store.mode_id: {} for store in stores
    }
    for mode_id, example_id, catalog, receipt in (
        captured.receipt.supplemental_route_evidence
    ):
        route_by_mode[mode_id][example_id] = (catalog, receipt)
    example_stats: list[Any] = []
    for store in stores:
        example_stats.extend(
            _example_chart_stats_from_store(
                store=store,
                atlas_sha256=fit.atlas.atlas_sha256,
                comparison_abi_sha256=comparison_abi_sha,
                supplemental_route_by_example=route_by_mode[store.mode_id],
            )
        )
    if len(example_stats) != 38 * 48:
        raise RuntimeError("heldout example chart statistics are incomplete")
    stats_by_key = {
        (row.family_id, row.example_id, row.mode_id): row
        for row in example_stats
    }
    if len(stats_by_key) != len(example_stats):
        raise RuntimeError("heldout example chart statistic keys collide")
    row_shuffle = build_within_example_row_shuffle(
        stores[0].row_keys,
        role=role,
    )
    if any(
        store.row_catalog_sha256 != stores[0].row_catalog_sha256
        or store.row_keys != stores[0].row_keys
        for store in stores[1:]
    ):
        raise RuntimeError("heldout mode stores are not row-aligned")
    store_by_mode = {store.mode_id: store for store in stores}
    directed_stats: list[Any] = []
    for ordinal, artifact in enumerate(fit.transport_artifacts, start=1):
        transport = load_directed_chart_transport(artifact.path)
        if (
            transport.transport_sha256 != artifact.transport_sha256
            or transport.source_mode_id != artifact.source_mode_id
            or transport.target_mode_id != artifact.target_mode_id
        ):
            raise RuntimeError("A transport strict reload endpoint drifted")
        evaluation = evaluate_directed_chart_transport_streaming(
            fit.atlas,
            transport,
            store_by_mode[transport.source_mode_id],
            store_by_mode[transport.target_mode_id],
            row_shuffle=row_shuffle,
            resources=resources,
        )
        for example in evaluation.examples:
            source_stats = stats_by_key[
                (example.family_id, example.example_id, transport.source_mode_id)
            ]
            target_stats = stats_by_key[
                (example.family_id, example.example_id, transport.target_mode_id)
            ]
            if (
                source_stats.aligned_row_catalog_sha256
                != target_stats.aligned_row_catalog_sha256
                or source_stats.row_count != example.row_count
            ):
                raise RuntimeError("directed example rows are not chart-aligned")
            directed_stats.append(
                DirectedChartTransportStats(
                    family_id=example.family_id,
                    example_id=example.example_id,
                    source_mode_id=transport.source_mode_id,
                    target_mode_id=transport.target_mode_id,
                    comparison_abi_sha256=comparison_abi_sha,
                    aligned_row_catalog_sha256=(
                        source_stats.aligned_row_catalog_sha256
                    ),
                    source_chart_stats_sha256=source_stats.sha256,
                    target_chart_stats_sha256=target_stats.sha256,
                    row_count=example.row_count,
                    transported_row_count=example.row_count,
                    transport_fit_role=FIT_ROLE,
                    transport_evaluation_role=role,
                    transport_fit_catalog_sha256=(
                        fit.transport_catalog_sha256
                    ),
                    transport_evaluation_catalog_sha256=(
                        source_stats.aligned_row_catalog_sha256
                    ),
                    transport_evaluation_panel_sha256=(
                        captured.receipt.role_panel_binding_sha256
                    ),
                    transport_split_receipt_sha256=(
                        evaluation.split_receipt_sha256
                    ),
                    transport_fit_receipt_sha256=transport.transport_sha256,
                    row_shuffle_catalog_sha256=(
                        evaluation.row_shuffle_catalog_sha256
                    ),
                    transport_protocol=(
                        FROZEN_DIRECTED_CHART_TRANSPORT_PROTOCOL
                    ),
                    weighted_transport_residual_energy=(
                        example.weighted_transport_residual_energy
                    ),
                    weighted_predicted_fisher_energy=(
                        example.weighted_predicted_fisher_energy
                    ),
                    weighted_target_fisher_energy=(
                        example.weighted_target_fisher_energy
                    ),
                    weighted_predicted_target_cross_inner_product=(
                        example.weighted_predicted_target_cross_inner_product
                    ),
                    weighted_zero_prediction_residual_energy=(
                        example.weighted_zero_prediction_residual_energy
                    ),
                    weighted_row_shuffled_transport_residual_energy=(
                        example.weighted_row_shuffled_transport_residual_energy
                    ),
                )
            )
        if progress is not None:
            progress(
                ordinal,
                len(fit.transport_artifacts),
                transport.source_mode_id,
                transport.target_mode_id,
            )
    if len(directed_stats) != 38 * 37 * 48:
        raise RuntimeError("heldout directed example evaluations are incomplete")
    evaluation_catalog = _role_evaluation_catalog_sha256(
        role=role,
        atlas_sha256=fit.atlas.atlas_sha256,
        capture_receipt_sha256=captured.receipt.receipt_sha256,
        fit_transport_catalog_sha256=fit.transport_catalog_sha256,
        row_shuffle_catalog_sha256=row_shuffle.catalog_sha256,
        example_stats=example_stats,
        directed_transport_stats=directed_stats,
    )
    observations.append(
        observe_phase_resources(
            preflight.binding,
            phase_id=f"{role}_all_directional_evaluations_frozen",
            completed_pair_count=38 * 37,
            total_pair_count=38 * 37,
        )
    )
    return RoleEvaluationArtifacts(
        role=role,
        applied_stores=tuple(stores),
        example_stats=tuple(example_stats),
        directed_transport_stats=tuple(directed_stats),
        row_shuffle_catalog_sha256=row_shuffle.catalog_sha256,
        evaluation_catalog_sha256=evaluation_catalog,
        resource_observations=tuple(observations),
    )


def derive_core_mode_specs(
    *,
    setup: Mapping[str, object],
    mode_lineage: Sequence[RuntimeModeLineage],
) -> tuple[Any, ...]:
    """Derive the complete core ModeSpec catalog from setup + qualification."""

    from fisher_graph.chart_energy_neighbor_graph import (
        CAUSAL_SEQUENCE_NATIVE_ABI,
        DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        EMBEDDING_CARRIER_H640_ADAPTER,
        LEXICAL_INGRESS_NATIVE_ABI,
        LEXICAL_READOUT_NATIVE_ABI,
        SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
        TOKEN_LOCAL_NATIVE_ABI,
        ChartModeSpec,
        NativeChartABI,
        canonical_carrier_comparison_abi,
    )

    kind_contract = {
        "attention": (
            CAUSAL_SEQUENCE_NATIVE_ABI,
            DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        ),
        "mlp": (
            TOKEN_LOCAL_NATIVE_ABI,
            DIRECT_H640_MUTATION_RESIDUAL_ADAPTER,
        ),
        "lexical_ingress": (
            LEXICAL_INGRESS_NATIVE_ABI,
            EMBEDDING_CARRIER_H640_ADAPTER,
        ),
        "lexical_egress": (
            LEXICAL_READOUT_NATIVE_ABI,
            SOFTCAP_READOUT_PULLBACK_H640_ADAPTER,
        ),
    }
    runtime_by_mode = {row.mode_id: row for row in mode_lineage}
    setup_modes = tuple(setup["logical_mode_ontology"]["ordered_modes"])
    if tuple(runtime_by_mode) != tuple(row["mode_id"] for row in setup_modes):
        raise ValueError("runtime mode lineage order differs from setup")
    comparison = canonical_carrier_comparison_abi()
    result: list[Any] = []
    for row in setup_modes:
        mode_id = str(row["mode_id"])
        mode_kind = str(row["mode_kind"])
        abi_kind, adapter_kind = kind_contract[mode_kind]
        runtime_row = runtime_by_mode[mode_id]
        native = NativeChartABI(
            abi_kind=abi_kind,
            abi_name=str(row["native_ABI"]),
            abi_version=1,
            input_signature=f"{abi_kind}.input",
            output_signature=f"{abi_kind}.output",
            context_signature=f"{abi_kind}.context",
        )
        result.append(
            ChartModeSpec(
                mode_id=mode_id,
                mode_kind=mode_kind,
                state_fingerprint_sha256=(
                    runtime_row.state_fingerprint_sha256
                ),
                event_id=str(row["event_id"]),
                event_index=int(row["event_index"]),
                context_site_id=str(row["context_site_id"]),
                contribution_site_id=str(row["contribution_site_id"]),
                site_placement_sha256=(
                    runtime_row.site_placement_sha256
                ),
                native_abi=native,
                comparison_abi=comparison,
                comparison_adapter_kind=adapter_kind,
                comparison_adapter_receipt_sha256=(
                    runtime_row.comparison_adapter_receipt_sha256
                ),
            )
        )
    return tuple(result)


def build_core_mode_specs(
    *,
    preflight: ExecutionPreflightReceipt,
    runtime: ChartNativeCaptureRuntime,
) -> tuple[Any, ...]:
    """Cross-bind frozen placement with runtime state and adapter receipts."""

    preflight.validate_receipt()
    return derive_core_mode_specs(
        setup=preflight.setup,
        mode_lineage=runtime.mode_lineage,
    )


@dataclass(frozen=True, slots=True)
class RoleGraphInputs:
    all_eight_diagnostic_source: Any
    primary_seven: Any
    eligibility: RoleEligibilityCatalog


def _frozen_chart_neighbor_config(setup: Mapping[str, object]) -> Any:
    from fisher_graph.chart_energy_neighbor_graph import (
        ChartEnergyNeighborConfig,
    )

    graph = setup["neighbor_graph_protocol"]
    gates = setup["fresh_confirmation_gates"]
    bootstrap = gates["within_family_bootstrap"]
    lofo = gates["leave_one_family_out"]
    return ChartEnergyNeighborConfig(
        k=int(graph["k"]),
        rank_boundary_tie_absolute_tolerance=float(
            graph["rank_boundary_tie_absolute_tolerance"]
        ),
        rank_boundary_tie_policy=str(graph["rank_boundary_tie_policy"]),
        fisher_energy_epsilon=float(graph["fisher_energy_epsilon"]),
        maximum_directional_normalized_residual=float(
            graph["maximum_each_directional_normalized_residual"]
        ),
        bootstrap_minimum_mutual_frequency=float(
            bootstrap["minimum_mutual_frequency"]
        ),
        lofo_minimum_mutual_count=int(lofo["minimum_mutual_count"]),
        lofo_expected_replicate_count=int(lofo["replicate_count"]),
    )


def _frozen_chart_bootstrap_config(
    setup: Mapping[str, object],
    *,
    family_example_ids: tuple[tuple[str, tuple[str, ...]], ...],
) -> Any:
    from fisher_graph.chart_energy_neighbor_graph import (
        ChartWithinFamilyBootstrapConfig,
    )

    frozen = setup["fresh_confirmation_gates"]["within_family_bootstrap"]
    return ChartWithinFamilyBootstrapConfig(
        seed=int(frozen["seed"]),
        resample_count=int(frozen["resample_count"]),
        draws_per_family=len(family_example_ids[0][1]),
        maximum_degenerate_fraction=float(
            frozen["maximum_degenerate_fraction"]
        ),
        family_example_ids=family_example_ids,
        draw_protocol_domain=bytes.fromhex(
            str(frozen["draw_protocol_domain_hex"])
        ),
    )


def _validate_role_graph_input_contract(
    *,
    setup: Mapping[str, object],
    role_manifest: Mapping[str, object],
    eligibility: RoleEligibilityCatalog,
    all_eight_source: Any,
    primary_source: Any,
    capture_row_count: int | None,
) -> None:
    """Bind graph policy, bootstrap, and q/t rows to the frozen setup."""

    from fisher_graph.shared_chart_energy_transport import (
        ChartRowKey,
        build_within_example_row_shuffle,
    )

    expected_config = _frozen_chart_neighbor_config(setup)
    expected_all_bootstrap = _frozen_chart_bootstrap_config(
        setup,
        family_example_ids=eligibility.all_family_example_ids,
    )
    expected_primary_bootstrap = _frozen_chart_bootstrap_config(
        setup,
        family_example_ids=eligibility.primary_family_example_ids,
    )
    frozen_bootstrap = setup["fresh_confirmation_gates"][
        "within_family_bootstrap"
    ]
    if (
        int(frozen_bootstrap["all_eight_diagnostic_source_draws_per_family"])
        != 8
        or int(frozen_bootstrap["primary_authorizing_draws_per_family"])
        != 7
        or all_eight_source.config != expected_config
        or primary_source.config != expected_config
        or all_eight_source.within_family_bootstrap_config
        != expected_all_bootstrap
        or primary_source.within_family_bootstrap_config
        != expected_primary_bootstrap
    ):
        raise ValueError("role graph policy/bootstrap differs from frozen setup")

    members = tuple(role_manifest["ordered_members"])
    member_by_id = {str(row["example_id"]): row for row in members}
    if len(member_by_id) != 48:
        raise ValueError("role manifest example catalog drifted")
    expected_rows_by_example: dict[str, tuple[str, int, str]] = {}
    all_role_row_keys: list[ChartRowKey] = []
    for example_id, member in member_by_id.items():
        family_id = str(member["family_id"])
        row_count = int(member["prediction_row_count"])
        if (
            row_count != int(member["target_token_count"]) - 1
            or int(member["prediction_query_position_start"]) != 0
            or int(member["prediction_query_position_stop_exclusive"])
            != row_count
            or int(member["target_label_position_start"]) != 1
            or int(member["target_label_position_stop_exclusive"])
            != row_count + 1
        ):
            raise ValueError("role manifest q/t row interval drifted")
        row_keys = tuple(
            ChartRowKey(family_id, example_id, query, query + 1)
            for query in range(row_count)
        )
        all_role_row_keys.extend(row_keys)
        expected_rows_by_example[example_id] = (
            family_id,
            row_count,
            _example_row_catalog_sha256(role=eligibility.role, row_keys=row_keys),
        )
    expected_shuffle_catalog = build_within_example_row_shuffle(
        all_role_row_keys,
        role=eligibility.role,
    ).catalog_sha256

    def validate_rows(source: Any, *, expected_examples: set[str]) -> None:
        observed_examples = {row.example_id for row in source.example_stats}
        if observed_examples != expected_examples:
            raise ValueError("role graph example set differs from eligibility")
        mode_row_totals: dict[str, int] = {}
        for row in source.example_stats:
            family_id, row_count, row_catalog = expected_rows_by_example[
                row.example_id
            ]
            if (
                row.family_id != family_id
                or row.row_count != row_count
                or row.aligned_row_catalog_sha256 != row_catalog
            ):
                raise ValueError("role graph chart q/t row lineage drifted")
            mode_row_totals[row.mode_id] = (
                mode_row_totals.get(row.mode_id, 0) + row.row_count
            )
        expected_total = sum(
            expected_rows_by_example[example_id][1]
            for example_id in expected_examples
        )
        if set(mode_row_totals.values()) != {expected_total}:
            raise ValueError("role graph per-mode row total drifted")
        for row in source.directed_transport_stats:
            family_id, row_count, row_catalog = expected_rows_by_example[
                row.example_id
            ]
            if (
                row.example_id not in expected_examples
                or row.family_id != family_id
                or row.row_count != row_count
                or row.transported_row_count != row_count
                or row.aligned_row_catalog_sha256 != row_catalog
                or row.transport_evaluation_catalog_sha256 != row_catalog
            ):
                raise ValueError("role graph transport q/t row lineage drifted")
        if {
            row.row_shuffle_catalog_sha256
            for row in source.directed_transport_stats
        } != {expected_shuffle_catalog}:
            raise ValueError(
                "role graph row-shuffle catalog differs from frozen v2 control"
            )

    all_ids = {
        example
        for _, examples in eligibility.all_family_example_ids
        for example in examples
    }
    primary_ids = {
        example
        for _, examples in eligibility.primary_family_example_ids
        for example in examples
    }
    validate_rows(all_eight_source, expected_examples=all_ids)
    validate_rows(primary_source, expected_examples=primary_ids)
    expected_capture_rows = int(role_manifest["prediction_row_count"])
    if expected_capture_rows != 24_732 or (
        capture_row_count is not None
        and capture_row_count != expected_capture_rows
    ):
        raise ValueError("role graph/capture total q/t row count drifted")


def build_role_graph_inputs(
    *,
    preflight: ExecutionPreflightReceipt,
    runtime: ChartNativeCaptureRuntime,
    evaluation: RoleEvaluationArtifacts,
) -> RoleGraphInputs:
    """Build separate all-eight diagnostic and exact-seven primary inputs."""

    from fisher_graph.chart_energy_neighbor_graph import (
        ChartEnergyNeighborInput,
        canonicalize_chart_energy_neighbor_input,
    )

    role = evaluation.role
    if role not in (DISCOVERY_ROLE, CONFIRMATION_ROLE):
        raise ValueError("graph input role must be B or C")
    eligibility = build_role_eligibility_catalog(
        preflight.fresh_manifest,
        role=role,
    )
    modes = build_core_mode_specs(preflight=preflight, runtime=runtime)
    frozen_config = _frozen_chart_neighbor_config(preflight.setup)

    def build_for(
        family_examples: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> Any:
        selected = {
            (family_id, example_id)
            for family_id, example_ids in family_examples
            for example_id in example_ids
        }
        stats = tuple(
            row
            for row in evaluation.example_stats
            if (row.family_id, row.example_id) in selected
        )
        transports = tuple(
            row
            for row in evaluation.directed_transport_stats
            if (row.family_id, row.example_id) in selected
        )
        expected_examples = sum(len(rows) for _, rows in family_examples)
        if (
            len(stats) != expected_examples * 38
            or len(transports) != expected_examples * 38 * 37
        ):
            raise RuntimeError("graph input subset is incomplete")
        return canonicalize_chart_energy_neighbor_input(
            ChartEnergyNeighborInput(
                panel_role=role,
                panel_binding_sha256=role_panel_binding_sha256(
                    preflight.fresh_manifest,
                    role,
                ),
                mode_specs=modes,
                example_stats=stats,
                directed_transport_stats=transports,
                within_family_bootstrap_config=(
                    _frozen_chart_bootstrap_config(
                        preflight.setup,
                        family_example_ids=family_examples,
                    )
                ),
                config=frozen_config,
            )
        )

    all_eight = eligibility.all_family_example_ids
    primary = eligibility.primary_family_example_ids
    if any(len(rows) != 8 for _, rows in all_eight) or any(
        len(rows) != 7 for _, rows in primary
    ):
        raise RuntimeError("B/C graph eligibility count drifted")
    result = RoleGraphInputs(
        all_eight_diagnostic_source=build_for(all_eight),
        primary_seven=build_for(primary),
        eligibility=eligibility,
    )
    _validate_role_graph_input_contract(
        setup=preflight.setup,
        role_manifest=preflight.fresh_manifest["roles"][role],
        eligibility=eligibility,
        all_eight_source=result.all_eight_diagnostic_source,
        primary_source=result.primary_seven,
        capture_row_count=None,
    )
    return result


@dataclass(frozen=True, slots=True)
class RoleGraphPhase:
    role: str
    all_eight_diagnostic_source: Any
    primary_receipt: Any
    diagnostic_receipts: tuple[tuple[str, Any], ...]
    candidate_or_graph_catalog_sha256: str
    frozen_candidate_edges: tuple[tuple[str, str], ...]
    confirmation_core_edges: tuple[tuple[str, str], ...]
    hard_long_veto_edges: tuple[tuple[str, str], ...]
    confirmed_edges: tuple[tuple[str, str], ...]
    exploratory_C_edges: tuple[tuple[str, str], ...]
    stage_receipt: FrozenStageReceipt
    capture_receipt: RoleCaptureReceipt
    evaluation_catalog_sha256: str
    evaluation_resource_observations: tuple[PhaseResourceObservation, ...]


def _canonical_edges(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError("edge catalog must be a list")
    result: list[tuple[str, str]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(item, str) and item for item in row)
            or row[0] >= row[1]
        ):
            raise ValueError("edge catalog row is not canonical")
        result.append((row[0], row[1]))
    if tuple(sorted(set(result))) != tuple(result):
        raise ValueError("edge catalog must be sorted and unique")
    return tuple(result)


def freeze_C_confirmed_edge_intersection(
    *,
    frozen_B_candidates: Sequence[Sequence[str]],
    C_confirmation_core: Sequence[Sequence[str]],
    C_hard_long_veto_pass: Sequence[Sequence[str]],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Return authorizing B∩C∩long edges and nonauthorizing new-C edges."""

    def canonical(value: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
        materialized = [list(edge) for edge in value]
        materialized.sort()
        return _canonical_edges(materialized)

    candidates = canonical(frozen_B_candidates)
    core = canonical(C_confirmation_core)
    long = canonical(C_hard_long_veto_pass)
    confirmed = tuple(sorted(set(candidates) & set(core) & set(long)))
    exploratory = tuple(sorted(set(core) - set(candidates)))
    return confirmed, exploratory


def build_role_graph_phase(
    *,
    preflight: ExecutionPreflightReceipt,
    captured: CapturedRole,
    evaluation: RoleEvaluationArtifacts,
    graph_inputs: RoleGraphInputs,
    prior_stage_receipt_sha256: str,
    frozen_B_phase: RoleGraphPhase | None = None,
) -> RoleGraphPhase:
    """Freeze B candidates or C confirmations with nonauthorizing strata."""

    from fisher_graph.chart_energy_neighbor_graph import (
        build_chart_energy_diagnostic_subset_receipt,
        build_chart_energy_neighbor_receipt,
    )

    role = evaluation.role
    if role != captured.receipt.role or role not in (
        DISCOVERY_ROLE,
        CONFIRMATION_ROLE,
    ):
        raise ValueError("graph phase role lineage drifted")
    _require_sha256(
        prior_stage_receipt_sha256,
        label="prior_stage_receipt_sha256",
    )
    primary = build_chart_energy_neighbor_receipt(graph_inputs.primary_seven)
    eligibility = graph_inputs.eligibility
    diagnostic_specs = (
        ("short_96", eligibility.short_family_example_ids),
        ("boundary_511_512_513", eligibility.boundary_family_example_ids),
        (
            "boundary_transition_514_diagnostic_only",
            eligibility.transition_family_example_ids,
        ),
        ("hard_long_576_640_768_veto", eligibility.hard_long_family_example_ids),
    )
    diagnostics = tuple(
        (
            subset_id,
            build_chart_energy_diagnostic_subset_receipt(
                graph_inputs.all_eight_diagnostic_source,
                subset_id=subset_id,
                family_example_ids=family_examples,
            ),
        )
        for subset_id, family_examples in diagnostic_specs
    )
    diagnostic_by_id = dict(diagnostics)
    candidates: tuple[tuple[str, str], ...] = ()
    core_edges: tuple[tuple[str, str], ...] = ()
    hard_long_edges: tuple[tuple[str, str], ...] = ()
    confirmed: tuple[tuple[str, str], ...] = ()
    exploratory: tuple[tuple[str, str], ...] = ()
    if role == DISCOVERY_ROLE:
        if frozen_B_phase is not None:
            raise ValueError("B phase cannot consume a prior B catalog")
        candidates = _canonical_edges(
            primary.analysis["point_gated_candidate_edges"]
        )
        catalog = _digest(
            _CANDIDATE_DOMAIN,
            {
                "B_primary_input_sha256": primary.input_sha256,
                "B_primary_receipt_sha256": primary.receipt_sha256,
                "candidate_edges": [list(edge) for edge in candidates],
                "eligibility_catalog_sha256": eligibility.catalog_sha256,
            },
        )
    else:
        if frozen_B_phase is None or frozen_B_phase.role != DISCOVERY_ROLE:
            raise ValueError("C phase requires the frozen B phase")
        candidates = frozen_B_phase.frozen_candidate_edges
        core_edges = _canonical_edges(primary.analysis["confirmation_core_edges"])
        hard_long_edges = _canonical_edges(
            diagnostic_by_id["hard_long_576_640_768_veto"].analysis[
                "point_gated_edges"
            ]
        )
        confirmed, exploratory = freeze_C_confirmed_edge_intersection(
            frozen_B_candidates=candidates,
            C_confirmation_core=core_edges,
            C_hard_long_veto_pass=hard_long_edges,
        )
        catalog = _digest(
            _GRAPH_DOMAIN,
            {
                "C_confirmation_core_edges": [
                    list(edge) for edge in core_edges
                ],
                "C_exploratory_edges_not_in_B": [
                    list(edge) for edge in exploratory
                ],
                "C_hard_long_veto_pass_edges": [
                    list(edge) for edge in hard_long_edges
                ],
                "C_primary_input_sha256": primary.input_sha256,
                "C_primary_receipt_sha256": primary.receipt_sha256,
                "confirmed_edges": [list(edge) for edge in confirmed],
                "frozen_B_candidate_catalog_sha256": (
                    frozen_B_phase.candidate_or_graph_catalog_sha256
                ),
                "frozen_B_candidate_edges": [
                    list(edge) for edge in candidates
                ],
                "hard_long_diagnostic_receipt_sha256": (
                    diagnostic_by_id[
                        "hard_long_576_640_768_veto"
                    ].receipt_sha256
                ),
            },
        )
    artifact_sha256s = {
        f"{role}_capture_receipt": captured.receipt.receipt_sha256,
        f"{role}_evaluation_catalog": evaluation.evaluation_catalog_sha256,
        f"{role}_primary_graph_input": primary.input_sha256,
        f"{role}_primary_graph_receipt": primary.receipt_sha256,
        f"{role}_candidate_or_graph_catalog": catalog,
        **{
            f"{role}_diagnostic_{subset_id}": receipt.receipt_sha256
            for subset_id, receipt in diagnostics
        },
    }
    stage = build_frozen_stage_receipt(
        role=role,
        preflight_receipt_sha256=preflight.receipt_sha256,
        role_panel_binding_sha256_value=(
            captured.receipt.role_panel_binding_sha256
        ),
        ordered_objective_row_catalog_sha256=(
            captured.receipt.ordered_objective_row_catalog_sha256
        ),
        score_objective_receipt_sha256=(
            captured.receipt.score_objective_receipt_sha256
        ),
        prior_stage_receipt_sha256=prior_stage_receipt_sha256,
        artifact_sha256s=artifact_sha256s,
        primary_eligible_example_catalog_sha256=eligibility.catalog_sha256,
    )
    return RoleGraphPhase(
        role=role,
        all_eight_diagnostic_source=graph_inputs.all_eight_diagnostic_source,
        primary_receipt=primary,
        diagnostic_receipts=diagnostics,
        candidate_or_graph_catalog_sha256=catalog,
        frozen_candidate_edges=candidates,
        confirmation_core_edges=core_edges,
        hard_long_veto_edges=hard_long_edges,
        confirmed_edges=confirmed,
        exploratory_C_edges=exploratory,
        stage_receipt=stage,
        capture_receipt=captured.receipt,
        evaluation_catalog_sha256=evaluation.evaluation_catalog_sha256,
        evaluation_resource_observations=evaluation.resource_observations,
    )


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    publication: PublishedExperimentReceipt
    A_stage_receipt_sha256: str
    B_stage_receipt_sha256: str
    C_stage_receipt_sha256: str
    D_stage_receipt_sha256: str | None
    confirmed_edges: tuple[tuple[str, str], ...]
    temp_workspace_removed: bool


def _safe_remove_tree(path: Path, *, allowed_parent: Path) -> None:
    """Remove one explicit run workspace without following any symlinks."""

    absolute = Path(os.path.abspath(path))
    parent = Path(os.path.abspath(allowed_parent))
    if absolute.parent != parent or absolute == parent:
        raise ValueError("cleanup target is not one direct run workspace")
    info = absolute.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("cleanup target must be a regular directory")

    def remove(directory: Path) -> None:
        with os.scandir(directory) as entries:
            materialized = sorted(entries, key=lambda item: item.name)
        for entry in materialized:
            entry_path = Path(entry.path)
            item = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(item.st_mode):
                raise RuntimeError("cleanup refuses symlinked workspace entries")
            if stat.S_ISDIR(item.st_mode):
                remove(entry_path)
                os.rmdir(entry_path)
            elif stat.S_ISREG(item.st_mode):
                os.unlink(entry_path)
            else:
                raise RuntimeError("cleanup found a non-regular workspace entry")

    remove(absolute)
    os.rmdir(absolute)
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _role_phase_safe_payload(phase: RoleGraphPhase) -> dict[str, object]:
    return {
        "all_eight_diagnostic_source": (
            phase.all_eight_diagnostic_source.canonical_payload()
        ),
        "candidate_or_graph_catalog_sha256": (
            phase.candidate_or_graph_catalog_sha256
        ),
        "capture_receipt": {
            **phase.capture_receipt.payload_without_sha256(),
            "receipt_sha256": phase.capture_receipt.receipt_sha256,
        },
        "confirmation_core_edges": [
            list(edge) for edge in phase.confirmation_core_edges
        ],
        "confirmed_edges": [list(edge) for edge in phase.confirmed_edges],
        "diagnostic_receipts": {
            subset_id: receipt.canonical_payload()
            for subset_id, receipt in phase.diagnostic_receipts
        },
        "exploratory_C_edges": [
            list(edge) for edge in phase.exploratory_C_edges
        ],
        "frozen_candidate_edges": [
            list(edge) for edge in phase.frozen_candidate_edges
        ],
        "hard_long_veto_edges": [
            list(edge) for edge in phase.hard_long_veto_edges
        ],
        "primary_receipt": {
            "analysis": phase.primary_receipt.analysis,
            "input_sha256": phase.primary_receipt.input_sha256,
            "receipt_sha256": phase.primary_receipt.receipt_sha256,
        },
        "evaluation_catalog_sha256": phase.evaluation_catalog_sha256,
        "evaluation_resource_observations": [
            {
                **item.payload_without_sha256(),
                "receipt_sha256": item.receipt_sha256,
            }
            for item in phase.evaluation_resource_observations
        ],
        "role": phase.role,
        "stage_receipt": {
            **phase.stage_receipt.payload_without_sha256(),
            "stage_receipt_sha256": phase.stage_receipt.stage_receipt_sha256,
        },
    }


def run_chart_native_neighbor_experiment(
    *,
    binding: ChartNativeExecutionBinding,
    runtime_factory: Callable[
        [ExecutionPreflightReceipt], ChartNativeCaptureRuntime
    ],
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
    svg_path: Path | str = DEFAULT_SVG_OUTPUT,
    atlas_path: Path | str = DEFAULT_ATLAS_OUTPUT,
    run_response_validation: bool = False,
    progress: Callable[[str, int, int, str, str], None] | None = None,
) -> ExperimentRunResult:
    """Execute the frozen A -> B -> C pipeline and optionally report D.

    The only model-facing dependency is ``runtime_factory``.  It is invoked
    strictly after source/setup/panel byte authentication.  A concrete Gemma
    factory is supplied by the CLI below; tests can provide a small fake.
    """

    if run_response_validation:
        raise ValueError(
            "v1 does not authorize D response validation until its strict "
            "measurement protocol is frozen"
        )
    if not binding.test_only_allow_nonproduction_shape:
        expected_outputs = (
            binding.output_directory / DEFAULT_REPORT_OUTPUT.name,
            binding.output_directory / DEFAULT_SVG_OUTPUT.name,
            binding.output_directory / DEFAULT_ATLAS_OUTPUT.name,
        )
        observed_outputs = tuple(
            Path(os.path.abspath(path))
            for path in (report_path, svg_path, atlas_path)
        )
        if observed_outputs != expected_outputs:
            raise ValueError(
                "live output paths must be the frozen names on the preflighted "
                "output device"
            )
    preflight = static_preflight(binding)
    preflight.validate_receipt()
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    panel_source = SequentialPanelSource(
        repository_root=binding.repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    workspace = binding.temp_directory / "chart-native-neighbor-work-v1"
    workspace.mkdir(mode=0o700, parents=False, exist_ok=False)
    runtime: ChartNativeCaptureRuntime | None = None
    published = False
    publication: PublishedExperimentReceipt | None = None
    try:
        runtime = runtime_factory(preflight)
        if not isinstance(runtime.mode_lineage, tuple):
            raise TypeError("runtime mode lineage must be a tuple")
        runtime_qualification = runtime.runtime_qualification_receipt()
        validate_runtime_qualification(
            preflight=preflight,
            receipt=runtime_qualification,
        )
        expected_runtime_session = derive_runtime_session_binding_sha256(
            preflight_receipt_sha256=preflight.receipt_sha256,
            runtime_qualification_receipt_sha256=(
                runtime_qualification.qualification_receipt_sha256
            ),
            source_model_fingerprint_sha256=(
                runtime_qualification
                .source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_sha256=(
                runtime_qualification
                .source_execution_fingerprint_after_compilation_sha256
            ),
            compiled_execution_fingerprint_sha256=(
                runtime_qualification.compiled_execution_fingerprint_sha256
            ),
            tied_readout_binding_receipt_sha256=(
                runtime_qualification.tied_readout_binding_receipt_sha256
            ),
            mode_lineage=runtime_qualification.mode_lineage,
        )
        if (
            runtime.mode_lineage != runtime_qualification.mode_lineage
            or runtime.runtime_session_binding_sha256
            != expected_runtime_session
        ):
            raise RuntimeError("runtime differs from its qualified lineage")
        frozen_mode_specs = build_core_mode_specs(
            preflight=preflight,
            runtime=runtime,
        )

        def phase_progress(
            phase: str,
        ) -> Callable[[int, int, str, str], None]:
            def emit(
                completed: int,
                total: int,
                source: str,
                target: str,
            ) -> None:
                if progress is not None:
                    progress(phase, completed, total, source, target)

            return emit

        # A may now be parsed/tokenized/modeled.  Later roles remain unopened.
        A_corpus = panel_source.open_role(FIT_ROLE)
        A_manifest = preflight.fresh_manifest["roles"][FIT_ROLE]
        dense_parity_sha = runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=A_corpus["examples"][0],
            role_manifest=A_manifest,
        )
        _require_sha256(dense_parity_sha, label="dense lexical parity receipt")
        A_raw_directory = workspace / "A-raw"
        A_fit_directory = workspace / "A-fit"
        A_raw_directory.mkdir(mode=0o700)
        A_fit_directory.mkdir(mode=0o700)
        A_capture = capture_role_to_raw_spools(
            preflight=preflight,
            runtime=runtime,
            role=FIT_ROLE,
            raw_corpus=A_corpus,
            destination_directory=A_raw_directory,
            dense_lexical_parity_receipt_sha256=dense_parity_sha,
        )
        A_fit = fit_A_chart_atlas_and_transports(
            preflight=preflight,
            captured=A_capture,
            mode_specs=frozen_mode_specs,
            destination_directory=A_fit_directory,
            progress=phase_progress("A_transport_fit"),
        )
        panel_source.freeze_current(
            A_fit.stage_receipt.stage_kind,
            A_fit.stage_receipt.stage_receipt_sha256,
        )
        A_cleanup_before = observe_phase_resources(
            binding,
            phase_id="A_before_raw_and_applied_cleanup",
        )
        _safe_remove_tree(A_raw_directory, allowed_parent=workspace)
        _safe_remove_tree(
            A_fit_directory / "applied",
            allowed_parent=A_fit_directory,
        )
        A_cleanup_after = observe_phase_resources(
            binding,
            phase_id="A_after_raw_and_applied_cleanup",
        )
        del A_corpus

        # B discovery is semantically unavailable until A is frozen.
        B_corpus = panel_source.open_role(DISCOVERY_ROLE)
        B_raw_directory = workspace / "B-raw"
        B_eval_directory = workspace / "B-eval"
        B_raw_directory.mkdir(mode=0o700)
        B_eval_directory.mkdir(mode=0o700)
        B_capture = capture_role_to_raw_spools(
            preflight=preflight,
            runtime=runtime,
            role=DISCOVERY_ROLE,
            raw_corpus=B_corpus,
            destination_directory=B_raw_directory,
            dense_lexical_parity_receipt_sha256=None,
        )
        B_evaluation = evaluate_heldout_role_transports(
            preflight=preflight,
            captured=B_capture,
            fit=A_fit,
            destination_directory=B_eval_directory,
            progress=phase_progress("B_transport_evaluation"),
        )
        B_inputs = build_role_graph_inputs(
            preflight=preflight,
            runtime=runtime,
            evaluation=B_evaluation,
        )
        B_phase = build_role_graph_phase(
            preflight=preflight,
            captured=B_capture,
            evaluation=B_evaluation,
            graph_inputs=B_inputs,
            prior_stage_receipt_sha256=(
                A_fit.stage_receipt.stage_receipt_sha256
            ),
        )
        panel_source.freeze_current(
            B_phase.stage_receipt.stage_kind,
            B_phase.stage_receipt.stage_receipt_sha256,
        )
        B_cleanup_before = observe_phase_resources(
            binding,
            phase_id="B_before_raw_and_applied_cleanup",
        )
        _safe_remove_tree(B_raw_directory, allowed_parent=workspace)
        _safe_remove_tree(B_eval_directory, allowed_parent=workspace)
        B_cleanup_after = observe_phase_resources(
            binding,
            phase_id="B_after_raw_and_applied_cleanup",
        )
        del B_corpus, B_inputs, B_evaluation

        # C is parsed only after the exact B candidate catalog is immutable.
        C_corpus = panel_source.open_role(CONFIRMATION_ROLE)
        C_raw_directory = workspace / "C-raw"
        C_eval_directory = workspace / "C-eval"
        C_raw_directory.mkdir(mode=0o700)
        C_eval_directory.mkdir(mode=0o700)
        C_capture = capture_role_to_raw_spools(
            preflight=preflight,
            runtime=runtime,
            role=CONFIRMATION_ROLE,
            raw_corpus=C_corpus,
            destination_directory=C_raw_directory,
            dense_lexical_parity_receipt_sha256=None,
        )
        C_evaluation = evaluate_heldout_role_transports(
            preflight=preflight,
            captured=C_capture,
            fit=A_fit,
            destination_directory=C_eval_directory,
            progress=phase_progress("C_transport_evaluation"),
        )
        C_inputs = build_role_graph_inputs(
            preflight=preflight,
            runtime=runtime,
            evaluation=C_evaluation,
        )
        C_phase = build_role_graph_phase(
            preflight=preflight,
            captured=C_capture,
            evaluation=C_evaluation,
            graph_inputs=C_inputs,
            prior_stage_receipt_sha256=(
                B_phase.stage_receipt.stage_receipt_sha256
            ),
            frozen_B_phase=B_phase,
        )
        panel_source.freeze_current(
            C_phase.stage_receipt.stage_kind,
            C_phase.stage_receipt.stage_receipt_sha256,
        )
        C_cleanup_before = observe_phase_resources(
            binding,
            phase_id="C_before_raw_and_applied_cleanup",
        )
        _safe_remove_tree(C_raw_directory, allowed_parent=workspace)
        _safe_remove_tree(C_eval_directory, allowed_parent=workspace)
        C_cleanup_after = observe_phase_resources(
            binding,
            phase_id="C_after_raw_and_applied_cleanup",
        )
        del C_corpus, C_inputs, C_evaluation

        D_stage: FrozenStageReceipt | None = None
        response_payload: Mapping[str, object] = {
            "D_semantically_opened": False,
            "D_chart_capture_count": 0,
            "D_may_change_graph": False,
            "skip_receipt_sha256": _digest(
                _D_SKIP_DOMAIN,
                {
                    "C_graph_catalog_sha256": (
                        C_phase.candidate_or_graph_catalog_sha256
                    ),
                    "reason": "response_validation_not_requested",
                },
            ),
        }

        ledger = build_capture_call_ledger()
        observed_capture_count = sum(
            receipt.batch_count
            for receipt in (
                A_capture.receipt,
                B_capture.receipt,
                C_capture.receipt,
            )
        )
        if (
            observed_capture_count != ledger.total_chart_forward_count
            or observed_capture_count != 135
        ):
            raise RuntimeError("live A/B/C capture counter drifted")
        runtime_receipt = runtime.runtime_execution_receipt()
        if not isinstance(runtime_receipt, RuntimeExecutionReceipt):
            raise TypeError("runtime did not return a typed execution receipt")
        if (
            runtime_receipt.runtime_qualification_receipt_sha256
            != runtime_qualification.qualification_receipt_sha256
            or runtime_receipt.compiled_execution_fingerprint_sha256
            != runtime_qualification.compiled_execution_fingerprint_sha256
            or runtime_receipt.compiler_receipt_sha256
            != runtime_qualification.compiler_receipt_sha256
            or runtime_receipt.tied_readout_binding_receipt_sha256
            != runtime_qualification.tied_readout_binding_receipt_sha256
            or runtime_receipt.tied_lexical_lineage_sha256
            != runtime_qualification.tied_lexical_lineage_sha256
            or runtime_receipt.final_logit_softcap
            != runtime_qualification.final_logit_softcap
            or runtime_receipt.runtime_session_binding_sha256
            != runtime.runtime_session_binding_sha256
            or runtime_receipt.mode_lineage != runtime.mode_lineage
            or any(
                capture.runtime_session_binding_sha256
                != runtime_receipt.runtime_session_binding_sha256
                for capture in (
                    A_capture.receipt,
                    B_capture.receipt,
                    C_capture.receipt,
                )
            )
        ):
            raise RuntimeError("runtime execution/capture lineage drifted")
        mode_axis = tuple(B_phase.primary_receipt.analysis["mode_axis"])
        C_mode_axis = tuple(C_phase.primary_receipt.analysis["mode_axis"])
        qualified_graph_axis = tuple(
            sorted(row.mode_id for row in runtime_qualification.mode_lineage)
        )
        if (
            mode_axis != C_mode_axis
            or mode_axis != qualified_graph_axis
            or mode_axis
            != tuple(
                mode.mode_id
                for mode in B_phase.all_eight_diagnostic_source.mode_specs
            )
            or mode_axis
            != tuple(
                mode.mode_id
                for mode in C_phase.all_eight_diagnostic_source.mode_specs
            )
        ):
            raise RuntimeError("B/C mode axis drifted before publication")
        # Final source guards and model teardown must succeed before either
        # report file can become visible.  A close failure therefore leaves no
        # publication to roll back.
        closing_runtime = runtime
        runtime = None
        closing_runtime.close()
        final_resources = observe_phase_resources(
            binding,
            phase_id="C_graph_frozen_before_publication",
            completed_pair_count=38 * 37 * 3,
            total_pair_count=38 * 37 * 3,
        )
        svg = render_neighbor_svg(
            mode_ids=mode_axis,
            confirmed_edges=C_phase.confirmed_edges,
        )
        report = seal_experiment_report(
            {
                "A_fit": {
                    "atlas_artifact_file_sha256": (
                        A_fit.atlas_artifact_file_sha256
                    ),
                    "atlas_sha256": A_fit.atlas.atlas_sha256,
                    "capture_receipt": {
                        **A_capture.receipt.payload_without_sha256(),
                        "receipt_sha256": A_capture.receipt.receipt_sha256,
                    },
                    "resource_observations": [
                        {
                            **item.payload_without_sha256(),
                            "receipt_sha256": item.receipt_sha256,
                        }
                        for item in A_fit.resource_observations
                    ],
                    "stage_receipt": {
                        **A_fit.stage_receipt.payload_without_sha256(),
                        "stage_receipt_sha256": (
                            A_fit.stage_receipt.stage_receipt_sha256
                        ),
                    },
                    "streaming_kernel_receipt_sha256": (
                        A_fit.streaming_kernel_receipt_sha256
                    ),
                    "mode_lineage_catalog_sha256": (
                        A_fit.mode_lineage_catalog_sha256
                    ),
                    "published_atlas_file_name": DEFAULT_ATLAS_OUTPUT.name,
                    "transport_catalog_sha256": (
                        A_fit.transport_catalog_sha256
                    ),
                    "transport_artifacts": [
                        {
                            "byte_count": item.byte_count,
                            "source_mode_id": item.source_mode_id,
                            "target_mode_id": item.target_mode_id,
                            "transport_sha256": item.transport_sha256,
                        }
                        for item in A_fit.transport_artifacts
                    ],
                },
                "B_discovery": _role_phase_safe_payload(B_phase),
                "C_confirmation": _role_phase_safe_payload(C_phase),
                "D_response_validation": {
                    "measurement": dict(response_payload),
                    "stage_receipt": (
                        None
                        if D_stage is None
                        else {
                            **D_stage.payload_without_sha256(),
                            "stage_receipt_sha256": (
                                D_stage.stage_receipt_sha256
                            ),
                        }
                    ),
                },
                "capture_call_ledger": {
                    **ledger.payload_without_sha256(),
                    "ledger_sha256": ledger.ledger_sha256,
                    "observed_live_capture_count": observed_capture_count,
                },
                "claims": {
                    "chart_native_neighbor_measured": True,
                    "compression_ready": False,
                    "D_response_validation_measured": False,
                    "fresh_reserve_opened": False,
                    "new_C_edges_authorize_neighbors": False,
                    "response_measurements_changed_graph": False,
                },
                "confirmed_edges": [
                    list(edge) for edge in C_phase.confirmed_edges
                ],
                "neighbor_svg_sha256": hashlib.sha256(svg).hexdigest(),
                "final_resource_observation": {
                    **final_resources.payload_without_sha256(),
                    "receipt_sha256": final_resources.receipt_sha256,
                },
                "phase_cleanup_resource_observations": [
                    {
                        **item.payload_without_sha256(),
                        "receipt_sha256": item.receipt_sha256,
                    }
                    for item in (
                        A_cleanup_before,
                        A_cleanup_after,
                        B_cleanup_before,
                        B_cleanup_after,
                        C_cleanup_before,
                        C_cleanup_after,
                    )
                ],
                "mode_axis": list(mode_axis),
                "preflight": {
                    **preflight.safe_payload_without_sha256(),
                    "receipt_sha256": preflight.receipt_sha256,
                },
                "protocol": {
                    "B_candidates_are_point_gated_only": True,
                    "C_confirmation_is_frozen_B_intersect_C_core_intersect_"
                    "hard_long_veto": True,
                    "common_comparison_ABI": "carrier_chart_energy.h640.v1",
                    "response_validation_is_non_authorizing": True,
                },
                "runtime_execution_receipt": {
                    **runtime_receipt.payload_without_sha256(),
                    "runtime_receipt_sha256": (
                        runtime_receipt.runtime_receipt_sha256
                    ),
                },
                "runtime_qualification_receipt": {
                    **runtime_qualification.payload_without_sha256(),
                    "qualification_receipt_sha256": (
                        runtime_qualification.qualification_receipt_sha256
                    ),
                },
            }
        )
        publication = publish_experiment_outputs(
            report=report,
            svg_bytes=svg,
            atlas_source_path=A_fit.atlas_artifact_path,
            report_path=report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
        published = True
        try:
            _safe_remove_tree(workspace, allowed_parent=binding.temp_directory)
        except BaseException:
            _rollback_published_experiment(publication)
            published = False
            raise
        return ExperimentRunResult(
            publication=publication,
            A_stage_receipt_sha256=(
                A_fit.stage_receipt.stage_receipt_sha256
            ),
            B_stage_receipt_sha256=(
                B_phase.stage_receipt.stage_receipt_sha256
            ),
            C_stage_receipt_sha256=(
                C_phase.stage_receipt.stage_receipt_sha256
            ),
            D_stage_receipt_sha256=(
                None if D_stage is None else D_stage.stage_receipt_sha256
            ),
            confirmed_edges=C_phase.confirmed_edges,
            temp_workspace_removed=True,
        )
    except BaseException:
        # A failed run cannot leave a partial workspace that a retry might
        # accidentally inherit.  Published outputs are rolled back by their
        # own atomic publisher before control reaches here.
        if workspace.exists() and not workspace.is_symlink():
            _safe_remove_tree(workspace, allowed_parent=binding.temp_directory)
        if published and publication is not None:
            _rollback_published_experiment(publication)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if published and workspace.exists():
            raise RuntimeError("published run left its temp workspace behind")


def _exact_mapping_fields(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields drifted")
    return value


def _parse_phase_resource_payload(value: object) -> PhaseResourceObservation:
    raw = _exact_mapping_fields(
        value,
        {
            "completed_pair_count",
            "observed_monotonic_ns",
            "peak_resident_set_bytes",
            "phase_id",
            "receipt_sha256",
            "temp_free_bytes",
            "temp_unique_regular_bytes",
            "total_pair_count",
        },
        label="phase resource observation",
    )
    receipt = PhaseResourceObservation(**raw)
    for label in (
        "peak_resident_set_bytes",
        "temp_unique_regular_bytes",
        "temp_free_bytes",
        "completed_pair_count",
        "total_pair_count",
        "observed_monotonic_ns",
    ):
        if type(getattr(receipt, label)) is not int or getattr(receipt, label) < 0:
            raise ValueError(f"phase resource {label} drifted")
    if (
        not isinstance(receipt.phase_id, str)
        or not receipt.phase_id
        or receipt.completed_pair_count > receipt.total_pair_count
    ):
        raise ValueError("phase resource identity or progress drifted")
    _require_sha256(receipt.receipt_sha256, label="resource receipt_sha256")
    if receipt.receipt_sha256 != _digest(
        _RESOURCE_DOMAIN,
        receipt.payload_without_sha256(),
    ):
        raise ValueError("phase resource receipt digest drifted")
    return receipt


def _validate_phase_resource_identity(
    receipt: PhaseResourceObservation,
    *,
    phase_id: str,
    completed_pair_count: int = 0,
    total_pair_count: int = 0,
) -> None:
    if (
        receipt.phase_id != phase_id
        or receipt.completed_pair_count != completed_pair_count
        or receipt.total_pair_count != total_pair_count
        or receipt.peak_resident_set_bytes
        > _PRODUCTION_MAXIMUM_RESIDENT_MEMORY_BYTES
        or receipt.temp_unique_regular_bytes > _PRODUCTION_TEMP_BYTES
        or receipt.temp_free_bytes < _PRODUCTION_ATOMIC_HEADROOM_BYTES
        or receipt.observed_monotonic_ns <= 0
    ):
        raise ValueError(f"resource phase {phase_id} lifecycle drifted")


def _parse_frozen_stage_payload(value: object) -> FrozenStageReceipt:
    raw = _exact_mapping_fields(
        value,
        {
            "artifact_sha256s",
            "ordered_objective_row_catalog_sha256",
            "preflight_receipt_sha256",
            "primary_eligible_example_catalog_sha256",
            "prior_stage_receipt_sha256",
            "role",
            "role_panel_binding_sha256",
            "score_objective_receipt_sha256",
            "stage_kind",
            "stage_receipt_sha256",
        },
        label="frozen stage receipt",
    )
    artifacts = raw["artifact_sha256s"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("frozen stage artifacts must be a mapping")
    return FrozenStageReceipt(
        role=raw["role"],
        stage_kind=raw["stage_kind"],
        preflight_receipt_sha256=raw["preflight_receipt_sha256"],
        role_panel_binding_sha256=raw["role_panel_binding_sha256"],
        ordered_objective_row_catalog_sha256=raw[
            "ordered_objective_row_catalog_sha256"
        ],
        score_objective_receipt_sha256=raw[
            "score_objective_receipt_sha256"
        ],
        prior_stage_receipt_sha256=raw["prior_stage_receipt_sha256"],
        artifact_sha256s=tuple(sorted(artifacts.items())),
        primary_eligible_example_catalog_sha256=raw[
            "primary_eligible_example_catalog_sha256"
        ],
        stage_receipt_sha256=raw["stage_receipt_sha256"],
    )


def _parse_role_capture_payload(value: object) -> RoleCaptureReceipt:
    raw = _exact_mapping_fields(
        value,
        {
            "batch_count",
            "batch_axis_catalog_sha256",
            "dense_lexical_parity_receipt_sha256",
            "example_capture_receipt_sha256s",
            "execution_receipt_sha256s",
            "lexical_nll_receipt_sha256s",
            "mode_raw_artifacts",
            "ordered_objective_row_catalog_sha256",
            "receipt_sha256",
            "role",
            "role_panel_binding_sha256",
            "runtime_session_binding_sha256",
            "row_count",
            "score_objective_receipt_sha256",
            "supplemental_route_evidence",
        },
        label="role capture receipt",
    )
    artifacts = raw["mode_raw_artifacts"]
    routes = raw["supplemental_route_evidence"]
    examples = raw["example_capture_receipt_sha256s"]
    if (
        not isinstance(artifacts, list)
        or not isinstance(routes, list)
        or not isinstance(examples, Mapping)
        or not isinstance(raw["lexical_nll_receipt_sha256s"], list)
        or not isinstance(raw["execution_receipt_sha256s"], list)
    ):
        raise ValueError("role capture receipt collections drifted")
    parsed_artifacts = []
    for item in artifacts:
        row = _exact_mapping_fields(
            item,
            {
                "artifact_sha256",
                "mode_id",
                "row_catalog_sha256",
                "tensor_catalog_sha256",
            },
            label="mode raw artifact",
        )
        parsed_artifacts.append(
            (
                row["mode_id"],
                row["artifact_sha256"],
                row["row_catalog_sha256"],
                row["tensor_catalog_sha256"],
            )
        )
    parsed_routes = []
    for item in routes:
        row = _exact_mapping_fields(
            item,
            {
                "example_id",
                "mode_id",
                "route_catalog_sha256",
                "route_receipt_sha256",
            },
            label="supplemental route evidence",
        )
        parsed_routes.append(
            (
                row["mode_id"],
                row["example_id"],
                row["route_catalog_sha256"],
                row["route_receipt_sha256"],
            )
        )
    return RoleCaptureReceipt(
        role=raw["role"],
        batch_axis_catalog_sha256=raw["batch_axis_catalog_sha256"],
        runtime_session_binding_sha256=raw[
            "runtime_session_binding_sha256"
        ],
        role_panel_binding_sha256=raw["role_panel_binding_sha256"],
        ordered_objective_row_catalog_sha256=raw[
            "ordered_objective_row_catalog_sha256"
        ],
        score_objective_receipt_sha256=raw[
            "score_objective_receipt_sha256"
        ],
        mode_raw_artifacts=tuple(parsed_artifacts),
        lexical_nll_receipt_sha256s=tuple(
            raw["lexical_nll_receipt_sha256s"]
        ),
        execution_receipt_sha256s=tuple(raw["execution_receipt_sha256s"]),
        example_capture_receipt_sha256s=tuple(examples.items()),
        supplemental_route_evidence=tuple(parsed_routes),
        dense_lexical_parity_receipt_sha256=raw[
            "dense_lexical_parity_receipt_sha256"
        ],
        batch_count=raw["batch_count"],
        row_count=raw["row_count"],
        receipt_sha256=raw["receipt_sha256"],
    )


def _primary_input_from_all_eight(
    source: Any,
    eligibility: RoleEligibilityCatalog,
) -> Any:
    from fisher_graph.chart_energy_neighbor_graph import (
        ChartEnergyNeighborInput,
        ChartWithinFamilyBootstrapConfig,
        canonicalize_chart_energy_neighbor_input,
    )

    family_examples = eligibility.primary_family_example_ids
    selected = {
        (family, example)
        for family, examples in family_examples
        for example in examples
    }
    bootstrap = source.within_family_bootstrap_config
    return canonicalize_chart_energy_neighbor_input(
        ChartEnergyNeighborInput(
            panel_role=source.panel_role,
            panel_binding_sha256=source.panel_binding_sha256,
            mode_specs=source.mode_specs,
            example_stats=tuple(
                row
                for row in source.example_stats
                if (row.family_id, row.example_id) in selected
            ),
            directed_transport_stats=tuple(
                row
                for row in source.directed_transport_stats
                if (row.family_id, row.example_id) in selected
            ),
            within_family_bootstrap_config=ChartWithinFamilyBootstrapConfig(
                seed=bootstrap.seed,
                resample_count=bootstrap.resample_count,
                draws_per_family=7,
                maximum_degenerate_fraction=(
                    bootstrap.maximum_degenerate_fraction
                ),
                family_example_ids=family_examples,
                draw_protocol_domain=bootstrap.draw_protocol_domain,
            ),
            config=source.config,
        )
    )


def _recompute_role_evaluation_catalog(
    *,
    source: Any,
    capture: RoleCaptureReceipt,
) -> str:
    fit_catalogs = {
        row.transport_fit_catalog_sha256
        for row in source.directed_transport_stats
    }
    shuffle_catalogs = {
        row.row_shuffle_catalog_sha256
        for row in source.directed_transport_stats
    }
    atlases = {row.chart_atlas_sha256 for row in source.example_stats}
    if len(fit_catalogs) != 1 or len(shuffle_catalogs) != 1 or len(atlases) != 1:
        raise ValueError("role evaluation lineage is not singular")
    return _role_evaluation_catalog_sha256(
        role=source.panel_role,
        atlas_sha256=next(iter(atlases)),
        capture_receipt_sha256=capture.receipt_sha256,
        fit_transport_catalog_sha256=next(iter(fit_catalogs)),
        row_shuffle_catalog_sha256=next(iter(shuffle_catalogs)),
        example_stats=source.example_stats,
        directed_transport_stats=source.directed_transport_stats,
    )


def _validate_completed_mode_lineage(
    *,
    mode_axis: Sequence[str],
    B_source: Any,
    C_source: Any,
    setup: Mapping[str, object],
    runtime_qualification: RuntimeQualificationReceipt,
    runtime_receipt: RuntimeExecutionReceipt,
    A_mode_lineage_catalog_sha256: str,
) -> tuple[Any, ...]:
    """Close canonical graph order against setup-order runtime lineage."""

    B_mode_specs = tuple(B_source.mode_specs)
    C_mode_specs = tuple(C_source.mode_specs)
    B_mode_payloads = {
        mode.mode_id: mode.canonical_payload() for mode in B_mode_specs
    }
    C_mode_payloads = {
        mode.mode_id: mode.canonical_payload() for mode in C_mode_specs
    }
    qualified_mode_specs = derive_core_mode_specs(
        setup=setup,
        mode_lineage=runtime_qualification.mode_lineage,
    )
    qualified_mode_payloads = {
        mode.mode_id: mode.canonical_payload() for mode in qualified_mode_specs
    }
    expected_graph_axis = tuple(sorted(qualified_mode_payloads))
    if (
        tuple(mode_axis) != expected_graph_axis
        or tuple(mode.mode_id for mode in B_mode_specs) != expected_graph_axis
        or tuple(mode.mode_id for mode in C_mode_specs) != expected_graph_axis
        or B_mode_payloads != C_mode_payloads
        or B_mode_payloads != qualified_mode_payloads
        or len(expected_graph_axis) != 38
    ):
        raise ValueError("completed report mode axis drifted")
    expected_lineage_catalog = _core_mode_lineage_catalog_sha256(
        qualified_mode_specs
    )
    expected_runtime_lineage = tuple(
        RuntimeModeLineage(
            mode_id=mode.mode_id,
            state_fingerprint_sha256=mode.state_fingerprint_sha256,
            site_placement_sha256=mode.site_placement_sha256,
            comparison_adapter_receipt_sha256=(
                mode.comparison_adapter_receipt_sha256
            ),
        )
        for mode in qualified_mode_specs
    )
    if (
        A_mode_lineage_catalog_sha256 != expected_lineage_catalog
        or runtime_receipt.mode_lineage != expected_runtime_lineage
    ):
        raise ValueError("A/B/C/runtime mode lineage catalog drifted")
    setup_modes = tuple(setup["logical_mode_ontology"]["ordered_modes"])
    if len(setup_modes) != 38:
        raise ValueError("setup mode ontology cardinality drifted")
    for ordinal, (mode, frozen) in enumerate(
        zip(qualified_mode_specs, setup_modes)
    ):
        if (
            frozen["mode_index"] != ordinal
            or mode.mode_id != frozen["mode_id"]
            or mode.mode_kind != frozen["mode_kind"]
            or mode.event_id != frozen["event_id"]
            or mode.event_index != frozen["event_index"]
            or mode.context_site_id != frozen["context_site_id"]
            or mode.contribution_site_id != frozen["contribution_site_id"]
            or mode.native_abi.abi_name != frozen["native_ABI"]
            or mode.comparison_abi.abi_name != frozen["comparison_ABI"]
            or (mode.mode_kind == "attention")
            is not bool(frozen["requires_cross_token_route_receipt"])
        ):
            raise ValueError("mode specification differs from setup ontology")
    return qualified_mode_specs


def _validate_completed_experiment_report(report: Mapping[str, object]) -> None:
    """Replay every authorizing receipt in a completed scientific report."""

    expected_top = {
        "A_fit",
        "B_discovery",
        "C_confirmation",
        "D_response_validation",
        "capture_call_ledger",
        "claims",
        "confirmed_edges",
        "final_resource_observation",
        "format_version",
        "mode_axis",
        "neighbor_svg_sha256",
        "phase_cleanup_resource_observations",
        "preflight",
        "protocol",
        "report_sha256",
        "runtime_execution_receipt",
        "runtime_qualification_receipt",
        "schema",
    }
    if set(report) != expected_top:
        raise ValueError("completed report top-level fields drifted")
    preflight = _exact_mapping_fields(
        report["preflight"],
        {
            "binding_sha256",
            "execution_contract",
            "execution_contract_sha256",
            "format_version",
            "free_bytes_by_device",
            "fresh_manifest_sha256",
            "fresh_receipt_sha256",
            "opaque_role_file_sha256s",
            "physical_memory_bytes",
            "receipt_sha256",
            "reserve_semantically_decoded",
            "schema",
            "setup_sha256",
            "source_file_sha256s",
            "transformers_or_model_runtime_touched_before_preflight",
        },
        label="completed preflight receipt",
    )
    preflight_sha = preflight["receipt_sha256"]
    _require_sha256(preflight_sha, label="preflight receipt_sha256")
    unsigned_preflight = dict(preflight)
    unsigned_preflight.pop("receipt_sha256")
    if preflight_sha != _digest(_PREFLIGHT_DOMAIN, unsigned_preflight):
        raise ValueError("completed preflight receipt digest drifted")
    expected_execution_contract = production_execution_contract_payload()
    if (
        preflight["execution_contract"] != expected_execution_contract
        or preflight["execution_contract_sha256"]
        != _digest(_EXECUTION_CONTRACT_DOMAIN, expected_execution_contract)
    ):
        raise ValueError("completed execution contract drifted")
    free_rows = preflight["free_bytes_by_device"]
    if (
        not isinstance(free_rows, list)
        or not free_rows
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] < 0
            or type(row[1]) is not int
            or row[1] < _PRODUCTION_MINIMUM_FREE_BYTES
            for row in free_rows
        )
        or free_rows != sorted(free_rows)
        or len({row[0] for row in free_rows}) != len(free_rows)
        or type(preflight["physical_memory_bytes"]) is not int
        or preflight["physical_memory_bytes"]
        < _PRODUCTION_MINIMUM_PHYSICAL_MEMORY_BYTES
    ):
        raise ValueError("completed preflight resource contract drifted")
    if (
        preflight["schema"] != EXECUTION_PREFLIGHT_SCHEMA
        or preflight["setup_sha256"] != PINNED_SETUP_SHA256
        or preflight["fresh_manifest_sha256"]
        != "fe045222c8d4f7ec21b14d158da63878c3462a1c38c6b9f9dcecf2b4d1d5e57f"
        or preflight["fresh_receipt_sha256"]
        != "a1586d6601323390573e750e9732bc7db4e415eefc0f15bf0a2fa8ef820cf1f0"
        or preflight["reserve_semantically_decoded"] is not False
        or preflight["transformers_or_model_runtime_touched_before_preflight"]
        is not False
    ):
        raise ValueError("completed preflight frozen lineage drifted")
    source_pins = preflight["source_file_sha256s"]
    if not isinstance(source_pins, Mapping) or set(source_pins) != set(
        _REQUIRED_SOURCE_NAMES
    ):
        raise ValueError("completed source pin catalog drifted")
    if any(
        source_pins.get(name) != digest
        for name, digest in PINNED_SOURCE_SHA256S.items()
    ):
        raise ValueError("completed source bytes differ from worker pins")
    for name, digest in source_pins.items():
        _require_sha256(digest, label=f"completed source pin {name}")
    live_worker_sha256 = hashlib.sha256(
        _read_regular(
            Path(__file__),
            label="live chart-native neighbor worker",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
    ).hexdigest()
    if source_pins["chart_native_neighbor_worker"] != live_worker_sha256:
        raise ValueError("completed worker source pin differs from live bytes")

    from fisher_graph import chart_energy_neighbor_graph as core
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        load_manifest,
    )

    parse_input = core.parse_chart_energy_neighbor_input_payload
    manifest = load_manifest()
    expected_opaque_role_files = {
        role: manifest["roles"][role]["file_sha256"] for role in ORDERED_ROLES
    }
    if preflight["opaque_role_file_sha256s"] != expected_opaque_role_files:
        raise ValueError("completed opaque role-file catalog drifted")

    runtime_qualification = _parse_runtime_qualification_receipt_payload(
        report["runtime_qualification_receipt"]
    )
    runtime_receipt = _parse_runtime_execution_receipt_payload(
        report["runtime_execution_receipt"]
    )
    # The setup is already authenticated by preflight; use its frozen model
    # identity rather than trusting values selected inside the report.
    from fisher_graph.gemma3_chart_native_neighbor_setup import build_setup

    setup = build_setup()
    source_model = setup["source_model"]
    from fisher_graph.chunked_tied_lexical_nll import (
        chunked_tied_lexical_nll_kernel_receipt,
    )
    from fisher_graph.gemma3_chart_mode_surfaces import (
        factorized_attention_route_kernel_receipt,
    )
    from fisher_graph.streaming_shared_chart_energy_transport import (
        streaming_chart_kernel_receipt,
    )

    expected_lexical_kernel = chunked_tied_lexical_nll_kernel_receipt()[
        "receipt_sha256"
    ]
    expected_streaming_kernel = streaming_chart_kernel_receipt()[
        "receipt_sha256"
    ]
    expected_route_kernel = factorized_attention_route_kernel_receipt()[
        "receipt_sha256"
    ]
    _validate_runtime_qualification_semantics(
        setup=setup,
        receipt=runtime_qualification,
        maximum_resident_memory_bytes=(
            _PRODUCTION_MAXIMUM_RESIDENT_MEMORY_BYTES
        ),
    )
    frozen_runtime_qualification = (
        _load_frozen_runtime_qualification_from_setup(
            setup=setup,
            repository_root=Path(__file__).resolve().parents[2],
        )
    )
    if (
        runtime_qualification != frozen_runtime_qualification
        or runtime_receipt.runtime_qualification_receipt_sha256
        != runtime_qualification.qualification_receipt_sha256
        or runtime_receipt.runtime_session_binding_sha256
        != derive_runtime_session_binding_sha256(
            preflight_receipt_sha256=preflight_sha,
            runtime_qualification_receipt_sha256=(
                runtime_qualification.qualification_receipt_sha256
            ),
            source_model_fingerprint_sha256=(
                runtime_qualification
                .source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_sha256=(
                runtime_qualification
                .source_execution_fingerprint_after_compilation_sha256
            ),
            compiled_execution_fingerprint_sha256=(
                runtime_qualification.compiled_execution_fingerprint_sha256
            ),
            tied_readout_binding_receipt_sha256=(
                runtime_qualification.tied_readout_binding_receipt_sha256
            ),
            mode_lineage=runtime_qualification.mode_lineage,
        )
        or runtime_receipt.mode_lineage != runtime_qualification.mode_lineage
        or runtime_receipt.compiled_execution_fingerprint_sha256
        != runtime_qualification.compiled_execution_fingerprint_sha256
        or runtime_receipt.compiler_receipt_sha256
        != runtime_qualification.compiler_receipt_sha256
        or runtime_receipt.tied_readout_binding_receipt_sha256
        != runtime_qualification.tied_readout_binding_receipt_sha256
        or runtime_receipt.tied_lexical_lineage_sha256
        != runtime_qualification.tied_lexical_lineage_sha256
        or runtime_receipt.final_logit_softcap
        != runtime_qualification.final_logit_softcap
        or runtime_receipt.model_id != source_model["model_id"]
        or runtime_receipt.model_revision != source_model["model_revision"]
        or runtime_receipt.source_parameter_count
        != source_model["parameter_count"]
        or runtime_receipt.hidden_width != source_model["hidden_width"]
        or runtime_receipt.source_model_fingerprint_before_sha256
        != source_model["source_model_fingerprint_sha256"]
        or runtime_receipt.source_execution_fingerprint_before_sha256
        != source_model["source_execution_fingerprint_sha256"]
        or runtime_receipt.lexical_kernel_receipt_sha256
        != expected_lexical_kernel
        or runtime_receipt.streaming_kernel_receipt_sha256
        != expected_streaming_kernel
        or runtime_receipt.factorized_route_kernel_receipt_sha256
        != expected_route_kernel
    ):
        raise ValueError("runtime execution receipt frozen lineage drifted")

    A = _exact_mapping_fields(
        report["A_fit"],
        {
            "atlas_artifact_file_sha256",
            "atlas_sha256",
            "capture_receipt",
            "mode_lineage_catalog_sha256",
            "published_atlas_file_name",
            "resource_observations",
            "stage_receipt",
            "streaming_kernel_receipt_sha256",
            "transport_artifacts",
            "transport_catalog_sha256",
        },
        label="A fit report",
    )
    A_capture = _parse_role_capture_payload(A["capture_receipt"])
    if (
        A["published_atlas_file_name"]
        != "chart-native-neighbor-experiment-v1.atlas.bundle"
    ):
        raise ValueError("published shared chart atlas name drifted")
    A_manifest = manifest["roles"][FIT_ROLE]
    A_panel_binding = role_panel_binding_sha256(manifest, FIT_ROLE)
    if (
        A_capture.role != FIT_ROLE
        or A_capture.runtime_session_binding_sha256
        != runtime_receipt.runtime_session_binding_sha256
        or A_capture.role_panel_binding_sha256 != A_panel_binding
        or A_capture.ordered_objective_row_catalog_sha256
        != A_manifest["ordered_objective_row_catalog_sha256"]
        or A_capture.score_objective_receipt_sha256
        != A_manifest["score_objective_receipt_sha256"]
    ):
        raise ValueError("A capture frozen role lineage drifted")
    _validate_capture_manifest_axis(A_capture, A_manifest)
    A_resources = A["resource_observations"]
    if not isinstance(A_resources, list) or len(A_resources) != 4:
        raise ValueError("A resource observations drifted")
    parsed_A_resources = tuple(
        _parse_phase_resource_payload(item) for item in A_resources
    )
    for receipt, phase_id, completed, total in zip(
        parsed_A_resources,
        (
            "A_raw_capture_sealed",
            "A_shared_atlas_frozen",
            "A_applied_chart_stores_frozen",
            "A_all_directional_transports_frozen",
        ),
        (0, 0, 0, 38 * 37),
        (0, 0, 0, 38 * 37),
    ):
        _validate_phase_resource_identity(
            receipt,
            phase_id=phase_id,
            completed_pair_count=completed,
            total_pair_count=total,
        )
    A_transports = A["transport_artifacts"]
    if not isinstance(A_transports, list) or len(A_transports) != 38 * 37:
        raise ValueError("A transport artifact count drifted")
    transport_rows: list[dict[str, object]] = []
    for item in A_transports:
        row = dict(
            _exact_mapping_fields(
                item,
                {
                    "byte_count",
                    "source_mode_id",
                    "target_mode_id",
                    "transport_sha256",
                },
                label="A transport artifact",
            )
        )
        if type(row["byte_count"]) is not int or row["byte_count"] <= 0:
            raise ValueError("A transport artifact byte count drifted")
        _require_sha256(row["transport_sha256"], label="transport_sha256")
        transport_rows.append(row)
    A_transport_catalog = _digest(_TRANSPORT_CATALOG_DOMAIN, transport_rows)
    if A_transport_catalog != A["transport_catalog_sha256"]:
        raise ValueError("A transport catalog digest drifted")
    A_transport_by_endpoint = {
        (row["source_mode_id"], row["target_mode_id"]): row[
            "transport_sha256"
        ]
        for row in transport_rows
    }
    if len(A_transport_by_endpoint) != 38 * 37:
        raise ValueError("A transport endpoint catalog collides")
    if (
        A["streaming_kernel_receipt_sha256"] != expected_streaming_kernel
        or runtime_receipt.streaming_kernel_receipt_sha256
        != A["streaming_kernel_receipt_sha256"]
    ):
        raise ValueError("A streaming kernel receipt drifted")
    A_stage = _parse_frozen_stage_payload(A["stage_receipt"])
    if (
        A_stage.role != FIT_ROLE
        or A_stage.prior_stage_receipt_sha256 is not None
        or A_stage.preflight_receipt_sha256 != preflight_sha
        or A_stage.role_panel_binding_sha256 != A_panel_binding
        or A_stage.ordered_objective_row_catalog_sha256
        != A_manifest["ordered_objective_row_catalog_sha256"]
        or A_stage.score_objective_receipt_sha256
        != A_manifest["score_objective_receipt_sha256"]
        or A_stage.primary_eligible_example_catalog_sha256 is not None
        or dict(A_stage.artifact_sha256s)
        != {
            "A_capture_receipt": A_capture.receipt_sha256,
            "A_mode_lineage_catalog": A["mode_lineage_catalog_sha256"],
            "A_shared_chart_atlas": A["atlas_sha256"],
            "A_shared_chart_atlas_file": A["atlas_artifact_file_sha256"],
            "A_streaming_kernel": A["streaming_kernel_receipt_sha256"],
            "A_transport_catalog": A_transport_catalog,
        }
    ):
        raise ValueError("A frozen stage lineage drifted")

    parsed_roles: dict[str, dict[str, object]] = {}
    for role, report_key, prior_stage in (
        (DISCOVERY_ROLE, "B_discovery", A_stage.stage_receipt_sha256),
        (CONFIRMATION_ROLE, "C_confirmation", None),
    ):
        raw = _exact_mapping_fields(
            report[report_key],
            {
                "all_eight_diagnostic_source",
                "candidate_or_graph_catalog_sha256",
                "capture_receipt",
                "confirmation_core_edges",
                "confirmed_edges",
                "diagnostic_receipts",
                "evaluation_catalog_sha256",
                "evaluation_resource_observations",
                "exploratory_C_edges",
                "frozen_candidate_edges",
                "hard_long_veto_edges",
                "primary_receipt",
                "role",
                "stage_receipt",
            },
            label=f"{role} report",
        )
        if raw["role"] != role:
            raise ValueError(f"{role} report role drifted")
        source = parse_input(raw["all_eight_diagnostic_source"])
        role_manifest = manifest["roles"][role]
        expected_panel_binding = role_panel_binding_sha256(manifest, role)
        if (
            source.panel_role != role
            or source.panel_binding_sha256 != expected_panel_binding
            or len(source.mode_specs) != 38
            or len(source.example_stats) != 38 * 48
            or len(source.directed_transport_stats) != 38 * 37 * 48
        ):
            raise ValueError(f"{role} all-eight source cardinality drifted")
        eligibility = build_role_eligibility_catalog(manifest, role=role)
        primary_input = _primary_input_from_all_eight(source, eligibility)
        if (
            len(primary_input.example_stats) != 38 * 42
            or len(primary_input.directed_transport_stats) != 38 * 37 * 42
        ):
            raise ValueError(f"{role} primary-seven cardinality drifted")
        capture = _parse_role_capture_payload(raw["capture_receipt"])
        _validate_role_graph_input_contract(
            setup=setup,
            role_manifest=role_manifest,
            eligibility=eligibility,
            all_eight_source=source,
            primary_source=primary_input,
            capture_row_count=capture.row_count,
        )
        rebuilt_primary = core.build_chart_energy_neighbor_receipt(primary_input)
        summary = _exact_mapping_fields(
            raw["primary_receipt"],
            {"analysis", "input_sha256", "receipt_sha256"},
            label=f"{role} primary receipt summary",
        )
        if summary != {
            "analysis": rebuilt_primary.analysis,
            "input_sha256": rebuilt_primary.input_sha256,
            "receipt_sha256": rebuilt_primary.receipt_sha256,
        }:
            raise ValueError(f"{role} primary core receipt replay drifted")
        if (
            capture.role != role
            or capture.runtime_session_binding_sha256
            != runtime_receipt.runtime_session_binding_sha256
            or capture.role_panel_binding_sha256 != source.panel_binding_sha256
            or capture.role_panel_binding_sha256 != expected_panel_binding
            or capture.ordered_objective_row_catalog_sha256
            != role_manifest["ordered_objective_row_catalog_sha256"]
            or capture.score_objective_receipt_sha256
            != role_manifest["score_objective_receipt_sha256"]
        ):
            raise ValueError(f"{role} capture-to-core lineage drifted")
        _validate_capture_manifest_axis(capture, role_manifest)
        route_by_cell = {
            (mode_id, example_id): (catalog, receipt)
            for mode_id, example_id, catalog, receipt in (
                capture.supplemental_route_evidence
            )
        }
        kind_by_mode = {
            mode.mode_id: mode.mode_kind for mode in source.mode_specs
        }
        for stats in source.example_stats:
            observed_route = (
                stats.supplemental_route_catalog_sha256,
                stats.supplemental_route_receipt_sha256,
            )
            expected_route = route_by_cell.get(
                (stats.mode_id, stats.example_id)
            )
            if kind_by_mode[stats.mode_id] == "attention":
                if expected_route is None or observed_route != expected_route:
                    raise ValueError(
                        f"{role} attention route evidence lineage drifted"
                    )
            elif observed_route != (None, None):
                raise ValueError(
                    f"{role} non-attention mode carries route evidence"
                )
        for directed in source.directed_transport_stats:
            endpoint = (directed.source_mode_id, directed.target_mode_id)
            if (
                directed.transport_fit_catalog_sha256
                != A_transport_catalog
                or directed.transport_fit_receipt_sha256
                != A_transport_by_endpoint.get(endpoint)
            ):
                raise ValueError(
                    f"{role} directed transport does not bind exact A fit"
                )
        evaluation_catalog = _recompute_role_evaluation_catalog(
            source=source,
            capture=capture,
        )
        if evaluation_catalog != raw["evaluation_catalog_sha256"]:
            raise ValueError(f"{role} evaluation catalog replay drifted")
        resource_rows = raw["evaluation_resource_observations"]
        if not isinstance(resource_rows, list) or len(resource_rows) != 2:
            raise ValueError(f"{role} evaluation resource ledger drifted")
        parsed_role_resources = tuple(
            _parse_phase_resource_payload(item) for item in resource_rows
        )
        _validate_phase_resource_identity(
            parsed_role_resources[0],
            phase_id=f"{role}_applied_chart_stores_frozen",
        )
        _validate_phase_resource_identity(
            parsed_role_resources[1],
            phase_id=f"{role}_all_directional_evaluations_frozen",
            completed_pair_count=38 * 37,
            total_pair_count=38 * 37,
        )
        diagnostics_raw = raw["diagnostic_receipts"]
        expected_diagnostics = {
            "short_96": eligibility.short_family_example_ids,
            "boundary_511_512_513": eligibility.boundary_family_example_ids,
            "boundary_transition_514_diagnostic_only": (
                eligibility.transition_family_example_ids
            ),
            "hard_long_576_640_768_veto": (
                eligibility.hard_long_family_example_ids
            ),
        }
        if not isinstance(diagnostics_raw, Mapping) or set(
            diagnostics_raw
        ) != set(expected_diagnostics):
            raise ValueError(f"{role} diagnostic receipt catalog drifted")
        diagnostics: dict[str, Any] = {}
        for subset_id, families in expected_diagnostics.items():
            parsed = core.parse_chart_energy_diagnostic_subset_receipt_payload(
                diagnostics_raw[subset_id],
                source_graph_input=source,
            )
            rebuilt = core.build_chart_energy_diagnostic_subset_receipt(
                source,
                subset_id=subset_id,
                family_example_ids=families,
            )
            if parsed.canonical_payload() != rebuilt.canonical_payload():
                raise ValueError(f"{role} diagnostic receipt replay drifted")
            diagnostics[subset_id] = parsed
        stage = _parse_frozen_stage_payload(raw["stage_receipt"])
        expected_prior = (
            prior_stage
            if role == DISCOVERY_ROLE
            else parsed_roles[DISCOVERY_ROLE]["stage"].stage_receipt_sha256
        )
        if (
            stage.role != role
            or stage.prior_stage_receipt_sha256 != expected_prior
            or stage.preflight_receipt_sha256 != preflight_sha
            or stage.role_panel_binding_sha256 != expected_panel_binding
            or stage.ordered_objective_row_catalog_sha256
            != role_manifest["ordered_objective_row_catalog_sha256"]
            or stage.score_objective_receipt_sha256
            != role_manifest["score_objective_receipt_sha256"]
            or stage.primary_eligible_example_catalog_sha256
            != eligibility.catalog_sha256
        ):
            raise ValueError(f"{role} stage chain or eligibility drifted")
        expected_artifacts = {
            f"{role}_capture_receipt": capture.receipt_sha256,
            f"{role}_evaluation_catalog": evaluation_catalog,
            f"{role}_primary_graph_input": rebuilt_primary.input_sha256,
            f"{role}_primary_graph_receipt": rebuilt_primary.receipt_sha256,
            f"{role}_candidate_or_graph_catalog": raw[
                "candidate_or_graph_catalog_sha256"
            ],
            **{
                f"{role}_diagnostic_{subset_id}": receipt.receipt_sha256
                for subset_id, receipt in diagnostics.items()
            },
        }
        if dict(stage.artifact_sha256s) != expected_artifacts:
            raise ValueError(f"{role} stage artifact catalog drifted")
        parsed_roles[role] = {
            "capture": capture,
            "diagnostics": diagnostics,
            "eligibility": eligibility,
            "primary": rebuilt_primary,
            "raw": raw,
            "resource_receipts": parsed_role_resources,
            "source": source,
            "stage": stage,
        }

    B = parsed_roles[DISCOVERY_ROLE]
    C = parsed_roles[CONFIRMATION_ROLE]
    B_candidates = _canonical_edges(
        B["primary"].analysis["point_gated_candidate_edges"]
    )
    B_catalog = _digest(
        _CANDIDATE_DOMAIN,
        {
            "B_primary_input_sha256": B["primary"].input_sha256,
            "B_primary_receipt_sha256": B["primary"].receipt_sha256,
            "candidate_edges": [list(edge) for edge in B_candidates],
            "eligibility_catalog_sha256": B["eligibility"].catalog_sha256,
        },
    )
    if (
        B_catalog != B["raw"]["candidate_or_graph_catalog_sha256"]
        or tuple(map(tuple, B["raw"]["frozen_candidate_edges"]))
        != B_candidates
        or B["raw"]["confirmation_core_edges"] != []
        or B["raw"]["hard_long_veto_edges"] != []
        or B["raw"]["confirmed_edges"] != []
    ):
        raise ValueError("B candidate catalog replay drifted")
    C_core = _canonical_edges(C["primary"].analysis["confirmation_core_edges"])
    C_long = _canonical_edges(
        C["diagnostics"]["hard_long_576_640_768_veto"].analysis[
            "point_gated_edges"
        ]
    )
    confirmed, exploratory = freeze_C_confirmed_edge_intersection(
        frozen_B_candidates=B_candidates,
        C_confirmation_core=C_core,
        C_hard_long_veto_pass=C_long,
    )
    C_catalog = _digest(
        _GRAPH_DOMAIN,
        {
            "C_confirmation_core_edges": [list(edge) for edge in C_core],
            "C_exploratory_edges_not_in_B": [
                list(edge) for edge in exploratory
            ],
            "C_hard_long_veto_pass_edges": [
                list(edge) for edge in C_long
            ],
            "C_primary_input_sha256": C["primary"].input_sha256,
            "C_primary_receipt_sha256": C["primary"].receipt_sha256,
            "confirmed_edges": [list(edge) for edge in confirmed],
            "frozen_B_candidate_catalog_sha256": B_catalog,
            "frozen_B_candidate_edges": [list(edge) for edge in B_candidates],
            "hard_long_diagnostic_receipt_sha256": C["diagnostics"][
                "hard_long_576_640_768_veto"
            ].receipt_sha256,
        },
    )
    if (
        C_catalog != C["raw"]["candidate_or_graph_catalog_sha256"]
        or tuple(map(tuple, C["raw"]["frozen_candidate_edges"]))
        != B_candidates
        or tuple(map(tuple, C["raw"]["confirmation_core_edges"])) != C_core
        or tuple(map(tuple, C["raw"]["hard_long_veto_edges"])) != C_long
        or tuple(map(tuple, C["raw"]["confirmed_edges"])) != confirmed
        or tuple(map(tuple, C["raw"]["exploratory_C_edges"])) != exploratory
        or tuple(map(tuple, report["confirmed_edges"])) != confirmed
    ):
        raise ValueError("C confirmed graph replay drifted")

    mode_axis = tuple(report["mode_axis"])
    qualified_mode_specs = _validate_completed_mode_lineage(
        mode_axis=mode_axis,
        B_source=B["source"],
        C_source=C["source"],
        setup=setup,
        runtime_qualification=runtime_qualification,
        runtime_receipt=runtime_receipt,
        A_mode_lineage_catalog_sha256=A["mode_lineage_catalog_sha256"],
    )
    expected_transport_order = tuple(
        (source, target)
        for target in sorted(mode_axis)
        for source in sorted(mode_axis)
        if source != target
    )
    if tuple(
        (row["source_mode_id"], row["target_mode_id"])
        for row in transport_rows
    ) != expected_transport_order:
        raise ValueError("A transport endpoint order drifted")
    if any(
        row.chart_atlas_sha256 != A["atlas_sha256"]
        for parsed in (B, C)
        for row in parsed["source"].example_stats
    ):
        raise ValueError("A atlas is not shared by B and C")

    ledger_raw = _exact_mapping_fields(
        report["capture_call_ledger"],
        set(build_capture_call_ledger().payload_without_sha256())
        | {"ledger_sha256", "observed_live_capture_count"},
        label="capture call ledger",
    )
    ledger = build_capture_call_ledger()
    if ledger_raw != {
        **ledger.payload_without_sha256(),
        "ledger_sha256": ledger.ledger_sha256,
        "observed_live_capture_count": 135,
    }:
        raise ValueError("completed capture call ledger drifted")
    cleanup = report["phase_cleanup_resource_observations"]
    if not isinstance(cleanup, list) or len(cleanup) != 6:
        raise ValueError("phase cleanup resource ledger drifted")
    parsed_cleanup = tuple(_parse_phase_resource_payload(item) for item in cleanup)
    cleanup_phase_ids = (
        "A_before_raw_and_applied_cleanup",
        "A_after_raw_and_applied_cleanup",
        "B_before_raw_and_applied_cleanup",
        "B_after_raw_and_applied_cleanup",
        "C_before_raw_and_applied_cleanup",
        "C_after_raw_and_applied_cleanup",
    )
    for receipt, phase_id in zip(parsed_cleanup, cleanup_phase_ids):
        _validate_phase_resource_identity(receipt, phase_id=phase_id)
    final_resource = _parse_phase_resource_payload(
        report["final_resource_observation"]
    )
    _validate_phase_resource_identity(
        final_resource,
        phase_id="C_graph_frozen_before_publication",
        completed_pair_count=38 * 37 * 3,
        total_pair_count=38 * 37 * 3,
    )
    all_resource_receipts = (
        *parsed_A_resources,
        *parsed_cleanup[0:2],
        *parsed_roles[DISCOVERY_ROLE]["resource_receipts"],
        *parsed_cleanup[2:4],
        *parsed_roles[CONFIRMATION_ROLE]["resource_receipts"],
        *parsed_cleanup[4:6],
        final_resource,
    )
    if any(
        left.observed_monotonic_ns >= right.observed_monotonic_ns
        for left, right in zip(all_resource_receipts, all_resource_receipts[1:])
    ):
        raise ValueError("resource phase monotonic lifecycle order drifted")

    claims = report["claims"]
    if claims != {
        "D_response_validation_measured": False,
        "chart_native_neighbor_measured": True,
        "compression_ready": False,
        "fresh_reserve_opened": False,
        "new_C_edges_authorize_neighbors": False,
        "response_measurements_changed_graph": False,
    }:
        raise ValueError("completed report claims drifted")
    protocol = report["protocol"]
    if protocol != {
        "B_candidates_are_point_gated_only": True,
        "C_confirmation_is_frozen_B_intersect_C_core_intersect_hard_long_veto": (
            True
        ),
        "common_comparison_ABI": "carrier_chart_energy.h640.v1",
        "response_validation_is_non_authorizing": True,
    }:
        raise ValueError("completed report protocol drifted")

    D = _exact_mapping_fields(
        report["D_response_validation"],
        {"measurement", "stage_receipt"},
        label="D response validation",
    )
    if not isinstance(D["measurement"], Mapping):
        raise ValueError("D measurement payload drifted")
    if D["stage_receipt"] is not None or D["measurement"] != {
        "D_chart_capture_count": 0,
        "D_may_change_graph": False,
        "D_semantically_opened": False,
        "skip_receipt_sha256": _digest(
            _D_SKIP_DOMAIN,
            {
                "C_graph_catalog_sha256": C_catalog,
                "reason": "response_validation_not_requested",
            },
        ),
    }:
        raise ValueError("D skip receipt drifted")

    expected_svg = render_neighbor_svg(
        mode_ids=mode_axis,
        confirmed_edges=confirmed,
    )
    expected_svg_sha256 = hashlib.sha256(expected_svg).hexdigest()
    _require_sha256(
        report["neighbor_svg_sha256"],
        label="neighbor_svg_sha256",
    )
    if report["neighbor_svg_sha256"] != expected_svg_sha256:
        raise ValueError("neighbor visualization binding drifted")


def _build_default_runtime_after_preflight(
    preflight: ExecutionPreflightReceipt,
    *,
    cache_dir: Path | str | None,
) -> ChartNativeCaptureRuntime:
    """Import the concrete model runtime only after static preflight."""

    from fisher_graph.gemma3_chart_native_neighbor_runtime import (
        build_pinned_gemma3_chart_native_capture_runtime,
    )

    return build_pinned_gemma3_chart_native_capture_runtime(
        preflight,
        cache_dir=cache_dir,
    )


def _canary_filesystem_catalog(
    binding: ChartNativeExecutionBinding,
) -> dict[str, object]:
    roots: list[dict[str, object]] = []
    for label, path in (
        ("output_directory", binding.output_directory),
        ("temp_directory", binding.temp_directory),
    ):
        root = Path(path)
        root_stat = root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
            raise ValueError(f"canary {label} must be a regular directory")
        entries: list[dict[str, object]] = []
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            child_stat = child.lstat()
            entries.append(
                {
                    "name": child.name,
                    "mode": stat.S_IFMT(child_stat.st_mode),
                    "size": child_stat.st_size,
                    "mtime_ns": child_stat.st_mtime_ns,
                }
            )
        roots.append(
            {
                "label": label,
                "directory_mtime_ns": root_stat.st_mtime_ns,
                "entries": entries,
            }
        )
    return {"roots": roots}


def _validate_first_A_canary_runtime_counters(runtime: object) -> dict[str, object]:
    counters = getattr(runtime, "counters", None)
    payload_method = getattr(counters, "payload", None)
    if not callable(payload_method):
        raise TypeError("canary runtime must expose prompt-free counters")
    payload = payload_method()
    expected_fields = {
        "role_batch_counts",
        "chart_backbone_forward_count",
        "chart_terminal_cotangent_vjp_count",
        "dense_parity_smoke_count",
        "parity_backbone_forward_count",
        "parity_terminal_cotangent_vjp_count",
        "captured_example_count",
        "supplemental_route_receipt_count",
        "canonical_gradient_targets_per_vjp",
        "auxiliary_gradient_targets_per_vjp",
        "gradient_targets_per_vjp",
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_activations",
        "contains_raw_gradients",
        "contains_logits",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("canary runtime counter fields drifted")
    expected = {
        "role_batch_counts": [
            [FIT_ROLE, 1],
            [DISCOVERY_ROLE, 0],
            [CONFIRMATION_ROLE, 0],
        ],
        "chart_backbone_forward_count": 1,
        "chart_terminal_cotangent_vjp_count": 1,
        "dense_parity_smoke_count": 1,
        "parity_backbone_forward_count": 1,
        "parity_terminal_cotangent_vjp_count": 1,
        "captured_example_count": 2,
        "supplemental_route_receipt_count": 36,
        "canonical_gradient_targets_per_vjp": 73,
        "auxiliary_gradient_targets_per_vjp": 19,
        "gradient_targets_per_vjp": 92,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_activations": False,
        "contains_raw_gradients": False,
        "contains_logits": False,
    }
    if dict(payload) != expected:
        raise ValueError("first-A canary observed runtime ledger drifted")
    return dict(payload)


def run_first_A_capture_canary(
    *,
    binding: ChartNativeExecutionBinding,
    runtime_factory: Callable[
        [ExecutionPreflightReceipt], ChartNativeCaptureRuntime
    ],
) -> FirstACaptureCanaryReceipt:
    """Run qualification, A0 parity, and exactly the first scheduled A batch.

    This hard-stop path never creates a workspace, spool, chart, transport,
    graph, visualization, or result artifact.  It semantically opens only A,
    closes the guarded runtime before sealing the prompt-free receipt, and has
    no code path that can open B, C, D, or the reserve.
    """

    preflight = static_preflight(binding)
    preflight.validate_receipt()
    filesystem_before = _canary_filesystem_catalog(binding)
    forbidden_paths = {
        "report": binding.output_directory / DEFAULT_REPORT_OUTPUT.name,
        "svg": binding.output_directory / DEFAULT_SVG_OUTPUT.name,
        "atlas": binding.output_directory / DEFAULT_ATLAS_OUTPUT.name,
        "workspace": binding.temp_directory / "chart-native-neighbor-work-v1",
    }
    forbidden_before = {
        label: path.exists() or path.is_symlink()
        for label, path in forbidden_paths.items()
    }
    if any(forbidden_before.values()):
        raise FileExistsError(
            "canary requires absent report/SVG/atlas/workspace sentinels"
        )
    filesystem_before_sha = _digest(
        _FIRST_A_CANARY_DOMAIN,
        {"filesystem_catalog": filesystem_before},
    )
    from fisher_graph.gemma3_chart_native_neighbor_fresh_panels import (
        SequentialPanelSource,
    )

    panel_source = SequentialPanelSource(
        repository_root=binding.repository_root,
        manifest=preflight.fresh_manifest,
        receipt=preflight.fresh_receipt,
        preflight=preflight.opaque_panel_receipt,
    )
    runtime: ChartNativeCaptureRuntime | None = None
    try:
        runtime = runtime_factory(preflight)
        qualification = runtime.runtime_qualification_receipt()
        validate_runtime_qualification(
            preflight=preflight,
            receipt=qualification,
        )
        expected_session = derive_runtime_session_binding_sha256(
            preflight_receipt_sha256=preflight.receipt_sha256,
            runtime_qualification_receipt_sha256=(
                qualification.qualification_receipt_sha256
            ),
            source_model_fingerprint_sha256=(
                qualification
                .source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_sha256=(
                qualification
                .source_execution_fingerprint_after_compilation_sha256
            ),
            compiled_execution_fingerprint_sha256=(
                qualification.compiled_execution_fingerprint_sha256
            ),
            tied_readout_binding_receipt_sha256=(
                qualification.tied_readout_binding_receipt_sha256
            ),
            mode_lineage=qualification.mode_lineage,
        )
        if (
            runtime.mode_lineage != qualification.mode_lineage
            or runtime.runtime_session_binding_sha256 != expected_session
            or runtime.hidden_width != 640
        ):
            raise RuntimeError("canary runtime differs from qualified lineage")
        observed_hidden_width = runtime.hidden_width
        mode_specs = build_core_mode_specs(preflight=preflight, runtime=runtime)
        mode_ids = tuple(row.mode_id for row in mode_specs)
        if mode_ids != tuple(row.mode_id for row in runtime.mode_lineage):
            raise RuntimeError("canary mode catalog differs from runtime lineage")

        A_corpus = panel_source.open_role(FIT_ROLE)
        A_manifest = preflight.fresh_manifest["roles"][FIT_ROLE]
        dense_parity_sha = runtime.run_dense_lexical_parity_smoke(
            preflight=preflight,
            raw_example=A_corpus["examples"][0],
            role_manifest=A_manifest,
        )
        _require_sha256(dense_parity_sha, label="canary dense parity receipt")
        first_batch = _capture_batch_plan(A_corpus)[0]
        if (
            len(first_batch) != 2
            or any(int(row["target_token_count"]) != 96 for row in first_batch)
        ):
            raise RuntimeError("first A canary batch differs from 96:b2")
        captured = runtime.capture_batch(
            preflight=preflight,
            role=FIT_ROLE,
            raw_examples=first_batch,
            role_manifest=A_manifest,
        )
        if (
            not isinstance(captured, CapturedBatch)
            or captured.role != FIT_ROLE
            or len(captured.examples) != 2
        ):
            raise RuntimeError("first A canary capture inventory drifted")

        import torch

        expected_raw = tuple(
            (str(row["family_id"]), str(row["example_id"]))
            for row in first_batch
        )
        observed_raw = tuple(
            (row.family_id, row.example_id) for row in captured.examples
        )
        if observed_raw != expected_raw:
            raise RuntimeError("first A canary example order drifted")
        examples: list[tuple[str, str, int, int, str]] = []
        route_evidence: list[tuple[str, str, str, str, str]] = []
        attention_modes = {
            f"mode.mutation.{index}" for index in range(1, 37, 2)
        }
        for example in captured.examples:
            if not isinstance(example, CapturedExampleRows):
                raise TypeError("first A canary example row type drifted")
            expected_row_keys = tuple(
                (example.family_id, example.example_id, query, query + 1)
                for query in range(95)
            )
            if (
                example.token_count != 96
                or example.row_keys != expected_row_keys
                or tuple(row.mode_id for row in example.modes) != mode_ids
            ):
                raise RuntimeError("first A canary mode rows drifted")
            examples.append(
                (
                    example.family_id,
                    example.example_id,
                    example.token_count,
                    len(example.row_keys),
                    example.capture_receipt_sha256,
                )
            )
            for mode in example.modes:
                if not isinstance(mode, CapturedModeRows):
                    raise TypeError("first A canary mode row type drifted")
                for label, tensor in (
                    ("context", mode.context_values),
                    ("comparison", mode.comparison_values),
                    ("contribution", mode.generalized_contributions),
                ):
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.shape != (95, 640)
                        or tensor.dtype != torch.float32
                        or tensor.device.type != "cpu"
                        or tensor.requires_grad
                        or not tensor.is_contiguous()
                        or not bool(torch.isfinite(tensor).all())
                    ):
                        raise ValueError(
                            f"first A canary {label} tensor inventory drifted"
                        )
                route = (
                    mode.supplemental_route_catalog_sha256,
                    mode.supplemental_route_receipt_sha256,
                    mode.supplemental_route_simplex_audit_sha256,
                )
                if mode.mode_id in attention_modes:
                    if any(value is None for value in route):
                        raise RuntimeError(
                            "first A canary lacks a factorized route audit"
                        )
                    route_evidence.append(
                        (
                            mode.mode_id,
                            example.example_id,
                            str(route[0]),
                            str(route[1]),
                            str(route[2]),
                        )
                    )
                elif any(value is not None for value in route):
                    raise RuntimeError(
                        "non-attention canary mode carries route evidence"
                    )
        if (
            panel_source.opened_roles != (FIT_ROLE,)
            or panel_source.current_role != FIT_ROLE
            or panel_source.freeze_receipts
            or panel_source.next_role != FIT_ROLE
        ):
            raise RuntimeError("first-A canary panel firewall state drifted")
        counter_payload = _validate_first_A_canary_runtime_counters(runtime)
        counter_receipt_sha = _digest(
            _FIRST_A_CANARY_DOMAIN,
            {"runtime_counters": counter_payload},
        )
        lexical_nll_sha = captured.lexical_nll_receipt_sha256
        execution_sha = captured.execution_receipt_sha256
        del captured
        del tensor, mode, example
        del first_batch, A_corpus

        runtime.close()
        runtime = None
        filesystem_after = _canary_filesystem_catalog(binding)
        filesystem_after_sha = _digest(
            _FIRST_A_CANARY_DOMAIN,
            {"filesystem_catalog": filesystem_after},
        )
        if (
            filesystem_after != filesystem_before
            or filesystem_after_sha != filesystem_before_sha
            or any(
                path.exists() or path.is_symlink()
                for path in forbidden_paths.values()
            )
        ):
            raise RuntimeError("first-A canary created or changed an artifact path")
        forbidden_after = {
            label: path.exists() or path.is_symlink()
            for label, path in forbidden_paths.items()
        }
        report_created = int(forbidden_after["report"])
        svg_created = int(forbidden_after["svg"])
        atlas_created = int(forbidden_after["atlas"])
        workspace_created = int(forbidden_after["workspace"])
        opened_roles = panel_source.opened_roles
        return build_first_A_capture_canary_receipt(
            preflight_receipt_sha256=preflight.receipt_sha256,
            runtime_qualification_receipt_sha256=(
                qualification.qualification_receipt_sha256
            ),
            runtime_session_binding_sha256=expected_session,
            A_role_panel_binding_sha256=role_panel_binding_sha256(
                preflight.fresh_manifest,
                FIT_ROLE,
            ),
            dense_lexical_parity_receipt_sha256=dense_parity_sha,
            lexical_nll_receipt_sha256=lexical_nll_sha,
            execution_receipt_sha256=execution_sha,
            runtime_counters_receipt_sha256=counter_receipt_sha,
            filesystem_before_catalog_sha256=filesystem_before_sha,
            filesystem_after_catalog_sha256=filesystem_after_sha,
            examples=tuple(examples),
            mode_ids=mode_ids,
            route_evidence=tuple(route_evidence),
            observed_role_batch_counts=tuple(
                (str(role), int(count))
                for role, count in counter_payload["role_batch_counts"]
            ),
            hidden_width=observed_hidden_width,
            gradient_target_count_per_VJP=int(
                counter_payload["gradient_targets_per_vjp"]
            ),
            dense_parity_backbone_forward_count=int(
                counter_payload["parity_backbone_forward_count"]
            ),
            dense_parity_terminal_cotangent_VJP_count=int(
                counter_payload["parity_terminal_cotangent_vjp_count"]
            ),
            chart_backbone_forward_count=int(
                counter_payload["chart_backbone_forward_count"]
            ),
            chart_terminal_cotangent_VJP_count=int(
                counter_payload["chart_terminal_cotangent_vjp_count"]
            ),
            total_backbone_forward_count=int(
                counter_payload["chart_backbone_forward_count"]
            )
            + int(counter_payload["parity_backbone_forward_count"]),
            total_terminal_cotangent_VJP_count=int(
                counter_payload["chart_terminal_cotangent_vjp_count"]
            )
            + int(counter_payload["parity_terminal_cotangent_vjp_count"]),
            captured_example_count=int(counter_payload["captured_example_count"]),
            supplemental_route_receipt_count=int(
                counter_payload["supplemental_route_receipt_count"]
            ),
            A_semantically_opened=FIT_ROLE in opened_roles,
            B_semantically_opened=DISCOVERY_ROLE in opened_roles,
            C_semantically_opened=CONFIRMATION_ROLE in opened_roles,
            D_semantically_opened=RESPONSE_ROLE in opened_roles,
            reserve_semantically_opened=RESERVE_ROLE in opened_roles,
            runtime_closed_before_receipt=True,
            filesystem_entry_sets_unchanged=(
                filesystem_before_sha == filesystem_after_sha
            ),
            spool_artifact_count=workspace_created,
            atlas_artifact_count=atlas_created,
            transport_artifact_count=workspace_created,
            graph_artifact_count=workspace_created,
            published_output_count=(report_created + svg_created + atlas_created),
            contains_prompt_text=False,
            contains_token_ids=False,
            contains_raw_tensors=False,
        )
    finally:
        try:
            if runtime is not None:
                runtime.close()
                runtime = None
        finally:
            failure_filesystem_after = _canary_filesystem_catalog(binding)
            if (
                failure_filesystem_after != filesystem_before
                or any(
                    path.exists() or path.is_symlink()
                    for path in forbidden_paths.values()
                )
            ):
                raise RuntimeError(
                    "first-A canary failure path changed an artifact path"
                )


def qualify_pinned_runtime(
    *,
    binding: ChartNativeExecutionBinding,
    destination: Path | str = DEFAULT_RUNTIME_QUALIFICATION_PATH,
    cache_dir: Path | str | None = None,
) -> RuntimeQualificationReceipt:
    """Generate the prompt-free runtime anchor without opening any panel role."""

    preflight = static_preflight(binding)
    runtime: ChartNativeCaptureRuntime | None = None
    try:
        runtime = _build_default_runtime_after_preflight(
            preflight,
            cache_dir=cache_dir,
        )
        receipt = runtime.runtime_qualification_receipt()
        _validate_runtime_qualification_semantics(
            setup=preflight.setup,
            receipt=receipt,
            maximum_resident_memory_bytes=(
                binding.storage_budget.maximum_resident_memory_bytes
            ),
        )
        expected_session = derive_runtime_session_binding_sha256(
            preflight_receipt_sha256=preflight.receipt_sha256,
            runtime_qualification_receipt_sha256=(
                receipt.qualification_receipt_sha256
            ),
            source_model_fingerprint_sha256=(
                receipt.source_model_fingerprint_after_compilation_sha256
            ),
            source_execution_fingerprint_sha256=(
                receipt.source_execution_fingerprint_after_compilation_sha256
            ),
            compiled_execution_fingerprint_sha256=(
                receipt.compiled_execution_fingerprint_sha256
            ),
            tied_readout_binding_receipt_sha256=(
                receipt.tied_readout_binding_receipt_sha256
            ),
            mode_lineage=receipt.mode_lineage,
        )
        if (
            runtime.mode_lineage != receipt.mode_lineage
            or runtime.runtime_session_binding_sha256 != expected_session
        ):
            raise RuntimeError("qualified runtime session binding drifted")
        closing = runtime
        runtime = None
        closing.close()
        write_runtime_qualification_receipt(receipt, destination)
        return receipt
    finally:
        if runtime is not None:
            runtime.close()


def _resolve_cli_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="strictly reload an existing JSON/SVG/atlas result",
    )
    action.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate setup, sources, and opaque panels without a model",
    )
    action.add_argument(
        "--qualify-runtime",
        action="store_true",
        help="load the model but open no panel; write its prompt-free lineage",
    )
    action.add_argument(
        "--canary-first-a-capture",
        action="store_true",
        help=(
            "validate qualification, A0 parity, and only the first 96:b2 A "
            "capture; print a no-raw receipt and hard-stop"
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG_OUTPUT)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS_OUTPUT)
    parser.add_argument(
        "--qualification-output",
        type=Path,
        default=DEFAULT_RUNTIME_QUALIFICATION_PATH,
    )
    parser.add_argument("--temp-directory", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    report_path = _resolve_cli_path(root, args.report)
    svg_path = _resolve_cli_path(root, args.svg)
    atlas_path = _resolve_cli_path(root, args.atlas)
    qualification_path = _resolve_cli_path(root, args.qualification_output)
    if args.check:
        report = strict_load_experiment_report(
            report_path,
            svg_path=svg_path,
            atlas_path=atlas_path,
        )
        print(
            json.dumps(
                {
                    "atlas_sha256": report["A_fit"]["atlas_sha256"],
                    "confirmed_edge_count": len(report["confirmed_edges"]),
                    "report_sha256": report["report_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not (
        report_path.parent == svg_path.parent == atlas_path.parent
    ):
        parser.error("report, SVG, and atlas must share one output directory")
    output_directory = report_path.parent
    temp_directory = (
        _resolve_cli_path(root, args.temp_directory)
        if args.temp_directory is not None
        else output_directory / ".chart-native-neighbor-temp"
    )
    output_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    binding = build_default_execution_binding(
        repository_root=root,
        output_directory=output_directory,
        temp_directory=temp_directory,
    )
    if args.preflight_only:
        preflight = static_preflight(binding)
        print(
            json.dumps(
                {
                    "execution_contract_sha256": (
                        binding.execution_contract_sha256
                    ),
                    "preflight_receipt_sha256": preflight.receipt_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.qualify_runtime:
        receipt = qualify_pinned_runtime(
            binding=binding,
            destination=qualification_path,
            cache_dir=args.cache_dir,
        )
        print(
            json.dumps(
                {
                    "qualification_path": str(qualification_path),
                    "qualification_receipt_sha256": (
                        receipt.qualification_receipt_sha256
                    ),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.canary_first_a_capture:
        receipt = run_first_A_capture_canary(
            binding=binding,
            runtime_factory=lambda preflight: (
                _build_default_runtime_after_preflight(
                    preflight,
                    cache_dir=args.cache_dir,
                )
            ),
        )
        print(json.dumps(receipt.state_dict(), sort_keys=True))
        return 0

    def runtime_factory(
        preflight: ExecutionPreflightReceipt,
    ) -> ChartNativeCaptureRuntime:
        return _build_default_runtime_after_preflight(
            preflight,
            cache_dir=args.cache_dir,
        )

    result = run_chart_native_neighbor_experiment(
        binding=binding,
        runtime_factory=runtime_factory,
        report_path=report_path,
        svg_path=svg_path,
        atlas_path=atlas_path,
    )
    print(
        json.dumps(
            {
                "atlas_path": result.publication.atlas_path,
                "confirmed_edge_count": len(result.confirmed_edges),
                "report_path": result.publication.report_path,
                "report_sha256": result.publication.report_sha256,
                "svg_path": result.publication.svg_path,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
