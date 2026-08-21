"""Production-scale, disk-backed shared-chart geometry and transport.

The in-memory reference implementation in :mod:`shared_chart_energy_transport`
is the numerical specification.  This module provides a fixed-order streaming
implementation for the production ``K=8, H=640`` experiment.  Raw carrier rows
are sealed as float32, every numerical reduction is performed on CPU float64,
and no operation materializes all modes or all rows in memory.

The artifact format is intentionally private to this module.  It is strict,
canonical, no-clobber, symlink rejecting, and self-authenticating down to each
fixed row chunk.  Raw spools contain context, comparison value, and signed
generalized contribution.  Applied stores deliberately omit raw context and
comparison value; they retain only the compact design/membership coordinates,
the signed generalized contribution required as a transport target, and row
energy diagnostics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from . import shared_chart_energy_transport as _reference
from .shared_chart_energy_transport import (
    COMPARISON_ABI_NAME,
    FIT_ROLE,
    ROW_SHUFFLE_SEED,
    ChartRowKey,
    DirectedChartTransport,
    DirectedExampleEnergy,
    DirectedTransportEvaluation,
    RowShuffleCatalog,
    SharedChartAtlas,
    SharedChartNumericsConfig,
    SharedChartRowBatch,
    build_within_example_row_shuffle,
    row_catalog_sha256,
    save_directed_chart_transport,
    transport_coefficient_storage_bytes,
)


STREAMING_STORE_SCHEMA = "fisher_graph.streaming_chart_array_store.v1"
RAW_MODE_ROW_SPOOL_KIND = "fisher_graph.streaming_raw_mode_rows.v1"
STREAMING_APPLIED_CHART_STORE_KIND = (
    "fisher_graph.streaming_applied_chart_rows.v1"
)
STREAMING_CHART_KERNEL_SCHEMA = (
    "fisher_graph.streaming_shared_chart_kernel.v1"
)
STREAMING_PREDICTION_BACKEND_ID = (
    "torch_mm_cpu_float64_streamed_fixed_rows_v1"
)

_STORE_MAGIC = b"FGCSTRM1\n"
_STORE_VERSION = 1
_MAX_HEADER_BYTES = 32 * 1024 * 1024
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_ABSOLUTE_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
_DEFAULT_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
_STORE_ARTIFACT_DOMAIN = b"fisher_graph:streaming-chart-store:v1\0"
_STREAMING_KERNEL_DOMAIN = b"fisher_graph:streaming-chart-kernel:v1\0"
_STREAMING_APPLIED_DOMAIN = b"fisher_graph:streaming-chart-applied:v1\0"
_STREAMING_RESOURCE_DOMAIN = b"fisher_graph:streaming-chart-resource:v1\0"

_RAW_FIELDS: Mapping[str, tuple[str, tuple[int, ...] | None]] = {
    "comparison_values": ("float32_le", None),
    "context_values": ("float32_le", None),
    "generalized_contributions": ("float32_le", None),
}
_APPLIED_FIELD_NAMES = frozenset(
    {
        "features",
        "generalized_contributions",
        "memberships",
        "weighted_activation_energy",
        "weighted_chart_orthogonal_energy",
        "weighted_chart_tangent_energy",
        "weighted_fisher_energy",
    }
)
_DTYPE_BY_NAME = {
    "float32_le": torch.float32,
    "float64_le": torch.float64,
}
_NUMPY_DTYPE_BY_NAME = {
    "float32_le": np.dtype("<f4"),
    "float64_le": np.dtype("<f8"),
}


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


def _checked_product(values: Sequence[int], *, label: str) -> int:
    result = 1
    for value in values:
        _require_positive_int(value, label=label)
        result *= value
        if result > _ABSOLUTE_MAX_ARTIFACT_BYTES:
            raise ValueError(f"{label} exceeds the strict artifact bound")
    return result


@dataclass(frozen=True, slots=True)
class StreamingChartResourceConfig:
    """Executable memory/artifact limits and the authenticated chunk schedule."""

    row_chunk_size: int = 256
    atlas_chunk_size: int = 256
    max_resident_bytes: int = 8 * 1024 * 1024 * 1024
    max_artifact_bytes: int = 4 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for label in (
            "row_chunk_size",
            "atlas_chunk_size",
            "max_resident_bytes",
            "max_artifact_bytes",
        ):
            _require_positive_int(getattr(self, label), label=label)
        if self.max_artifact_bytes > _ABSOLUTE_MAX_ARTIFACT_BYTES:
            raise ValueError("max_artifact_bytes exceeds the absolute safety cap")

    def payload(self) -> dict[str, int]:
        return {
            "atlas_chunk_size": self.atlas_chunk_size,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_resident_bytes": self.max_resident_bytes,
            "row_chunk_size": self.row_chunk_size,
        }

    @property
    def receipt_sha256(self) -> str:
        return _digest(_STREAMING_RESOURCE_DOMAIN, self.payload())


@dataclass(frozen=True, slots=True)
class StreamingChartResourceEstimate:
    row_count_per_mode: int
    mode_count: int
    hidden_width: int
    chart_count: int
    tangent_rank: int
    source_feature_width: int
    raw_spool_payload_bytes_per_mode: int
    raw_spool_payload_bytes_per_role: int
    raw_spool_atomic_publish_peak_bytes_per_mode: int
    applied_store_payload_bytes_per_mode: int
    applied_store_payload_bytes_per_role: int
    applied_store_atomic_publish_peak_bytes_per_mode: int
    raw_plus_applied_role_payload_bytes: int
    maximum_phase_payload_and_atomic_temp_bytes: int
    atlas_peak_resident_bytes: int
    apply_peak_resident_bytes: int
    transport_fit_peak_resident_bytes: int
    transport_eval_peak_resident_bytes: int
    all_ordered_coefficient_bytes: int
    receipt_sha256: str

    def payload(self) -> dict[str, int | str]:
        return {
            "all_ordered_coefficient_bytes": self.all_ordered_coefficient_bytes,
            "applied_store_payload_bytes_per_mode": (
                self.applied_store_payload_bytes_per_mode
            ),
            "applied_store_payload_bytes_per_role": (
                self.applied_store_payload_bytes_per_role
            ),
            "applied_store_atomic_publish_peak_bytes_per_mode": (
                self.applied_store_atomic_publish_peak_bytes_per_mode
            ),
            "apply_peak_resident_bytes": self.apply_peak_resident_bytes,
            "atlas_peak_resident_bytes": self.atlas_peak_resident_bytes,
            "chart_count": self.chart_count,
            "hidden_width": self.hidden_width,
            "mode_count": self.mode_count,
            "maximum_phase_payload_and_atomic_temp_bytes": (
                self.maximum_phase_payload_and_atomic_temp_bytes
            ),
            "raw_spool_payload_bytes_per_mode": (
                self.raw_spool_payload_bytes_per_mode
            ),
            "raw_spool_payload_bytes_per_role": (
                self.raw_spool_payload_bytes_per_role
            ),
            "raw_spool_atomic_publish_peak_bytes_per_mode": (
                self.raw_spool_atomic_publish_peak_bytes_per_mode
            ),
            "raw_plus_applied_role_payload_bytes": (
                self.raw_plus_applied_role_payload_bytes
            ),
            "receipt_sha256": self.receipt_sha256,
            "row_count_per_mode": self.row_count_per_mode,
            "source_feature_width": self.source_feature_width,
            "tangent_rank": self.tangent_rank,
            "transport_eval_peak_resident_bytes": (
                self.transport_eval_peak_resident_bytes
            ),
            "transport_fit_peak_resident_bytes": (
                self.transport_fit_peak_resident_bytes
            ),
        }


def estimate_streaming_chart_resources(
    *,
    row_count_per_mode: int,
    mode_count: int,
    config: SharedChartNumericsConfig = SharedChartNumericsConfig(),
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> StreamingChartResourceEstimate:
    """Return allocation and disk estimates without allocating production rows."""

    _require_positive_int(row_count_per_mode, label="row_count_per_mode")
    if type(mode_count) is not int or mode_count < 2:
        raise ValueError("mode_count must be at least two")
    if not isinstance(config, SharedChartNumericsConfig):
        raise TypeError("config must be SharedChartNumericsConfig")
    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")
    n = row_count_per_mode
    h = config.hidden_width
    k = config.chart_count
    r = config.tangent_rank
    f = config.source_feature_width
    raw_per_mode = n * 3 * h * 4
    applied_per_mode = n * (f * 8 + k * 8 + h * 4 + 4 * 8)
    atlas_peak = (
        (k + 12) * h * h * 8
        + resources.atlas_chunk_size * h * 8 * 5
        + resources.atlas_chunk_size * k * 8 * 3
        + k * h * r * 8
    )
    apply_peak = (
        resources.row_chunk_size * h * 8 * 8
        + resources.row_chunk_size * (f + k) * 8
        + k * h * r * 8
    )
    fit_peak = (
        f * f * 8 * 3
        + f * h * 8 * 3
        + resources.row_chunk_size * (f * 3 + h * 2) * 8
        + n * 8 * 2
    )
    eval_peak = (
        resources.row_chunk_size * (f * 2 + k * 3 + h * 5) * 8
        + n * 6 * 40
    )
    raw_role = raw_per_mode * mode_count
    applied_role = applied_per_mode * mode_count
    values = {
        "row_count_per_mode": n,
        "mode_count": mode_count,
        "hidden_width": h,
        "chart_count": k,
        "tangent_rank": r,
        "source_feature_width": f,
        "raw_spool_payload_bytes_per_mode": raw_per_mode,
        "raw_spool_payload_bytes_per_role": raw_role,
        "raw_spool_atomic_publish_peak_bytes_per_mode": raw_per_mode * 2,
        "applied_store_payload_bytes_per_mode": applied_per_mode,
        "applied_store_payload_bytes_per_role": applied_role,
        "applied_store_atomic_publish_peak_bytes_per_mode": applied_per_mode * 2,
        "raw_plus_applied_role_payload_bytes": raw_role + applied_role,
        "maximum_phase_payload_and_atomic_temp_bytes": (
            raw_role + applied_role + applied_per_mode
        ),
        "atlas_peak_resident_bytes": atlas_peak,
        "apply_peak_resident_bytes": apply_peak,
        "transport_fit_peak_resident_bytes": fit_peak,
        "transport_eval_peak_resident_bytes": eval_peak,
        "all_ordered_coefficient_bytes": transport_coefficient_storage_bytes(
            config, mode_count=mode_count
        ),
    }
    receipt = _digest(
        _STREAMING_RESOURCE_DOMAIN,
        {**values, "resource_config": resources.payload()},
    )
    return StreamingChartResourceEstimate(**values, receipt_sha256=receipt)


def _enforce_resident_budget(
    resources: StreamingChartResourceConfig,
    *,
    required_bytes: int,
    operation: str,
) -> None:
    if required_bytes > resources.max_resident_bytes:
        raise MemoryError(
            f"{operation} requires an estimated {required_bytes} resident bytes, "
            f"above max_resident_bytes={resources.max_resident_bytes}"
        )


def _resource_config_from_payload(
    value: object,
) -> StreamingChartResourceConfig:
    expected = set(StreamingChartResourceConfig().payload())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("streaming resource config payload drifted")
    try:
        return StreamingChartResourceConfig(**dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("streaming resource config is invalid") from error


def _validate_fixed_block_chunks(
    store: "_OpenedArrayStore", *, row_chunk_size: int
) -> None:
    for block in store.blocks.values():
        for chunk in block["chunk_sha256s"]:
            expected = min(
                row_chunk_size, len(store.row_keys) - chunk["start_row"]
            )
            if chunk["row_count"] != expected:
                raise ValueError(
                    "streaming artifact does not use the fixed row-chunk schedule"
                )


def streaming_chart_kernel_receipt(
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> dict[str, object]:
    """Authenticate the production reduction order, kernels, and environment."""

    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")
    payload: dict[str, object] = {
        "atlas_covariance_kernel": (
            "mode_then_fixed_row_chunk_soft_membership_outer_sum_float64_v1"
        ),
        "atlas_farthest_kernel": (
            "mode_then_fixed_row_chunk_weighted_norm_identity_first_occurrence_v1"
        ),
        "atlas_lloyd_kernel": (
            "twelve_mode_then_fixed_row_chunk_weighted_norm_identity_passes_v1"
        ),
        "atlas_orientation_kernel": (
            "torch_linalg_eigh_descending_tie_gate_largest_abs_positive_v1"
        ),
        "compute_device": "cpu",
        "compute_dtype": "float64",
        "prediction_backend_id": STREAMING_PREDICTION_BACKEND_ID,
        "raw_spool_dtype": "float32_le",
        "resources": resources.payload(),
        "schema": STREAMING_CHART_KERNEL_SCHEMA,
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "transport_fit_kernel": (
            "target_fisher_balanced_streamed_primal_normal_equations_v1"
        ),
    }
    return {
        **payload,
        "receipt_sha256": _digest(_STREAMING_KERNEL_DOMAIN, payload),
    }


def _tensor_semantic_hasher(shape: Sequence[int]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(_reference._TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {"dtype": "float64_le", "shape": list(shape)}
        )
    )
    return digest


def _tensor_raw_bytes(value: Tensor, *, dtype_name: str) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("streaming chart artifacts require a little-endian host")
    dtype = _DTYPE_BY_NAME[dtype_name]
    converted = value.detach().to(device="cpu", dtype=dtype).contiguous()
    if not bool(torch.isfinite(converted).all()):
        raise ValueError("streaming chart tensors must remain finite after casting")
    return converted.numpy().tobytes(order="C")


@dataclass(slots=True)
class _TemporaryField:
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    path: Path
    handle: object
    rows_written: int = 0

    @property
    def row_width(self) -> int:
        return math.prod(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def row_bytes(self) -> int:
        return self.row_width * _NUMPY_DTYPE_BY_NAME[self.dtype_name].itemsize

    def append(self, value: Tensor) -> None:
        expected_tail = self.shape[1:]
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.ndim != len(self.shape)
            or tuple(value.shape[1:]) != expected_tail
            or not value.shape[0]
        ):
            raise ValueError(
                f"{self.name} must be a nonempty floating tensor with tail "
                f"shape {expected_tail}"
            )
        if self.rows_written + int(value.shape[0]) > self.shape[0]:
            raise ValueError(f"{self.name} exceeds its declared row count")
        raw = _tensor_raw_bytes(value, dtype_name=self.dtype_name)
        self.handle.write(raw)
        self.rows_written += int(value.shape[0])

    def finish(self) -> None:
        if self.rows_written != self.shape[0]:
            raise ValueError(
                f"{self.name} has {self.rows_written} rows, expected {self.shape[0]}"
            )
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def abort(self) -> None:
        try:
            if not self.handle.closed:
                self.handle.close()
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _create_temporary_field(
    parent: Path,
    *,
    name: str,
    dtype_name: str,
    shape: tuple[int, ...],
) -> _TemporaryField:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".stream-{name}-", suffix=".rows", dir=parent
    )
    return _TemporaryField(
        name=name,
        dtype_name=dtype_name,
        shape=shape,
        path=Path(temporary),
        handle=os.fdopen(descriptor, "wb"),
    )


def _read_exact(descriptor: int, offset: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        chunk = os.pread(descriptor, min(4 * 1024 * 1024, remaining), position)
        if not chunk:
            raise ValueError("streaming chart artifact truncated during read")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _field_catalog(
    field: _TemporaryField,
    *,
    row_chunk_size: int,
    payload_offset: int,
) -> dict[str, object]:
    raw_digest = hashlib.sha256()
    semantic = _tensor_semantic_hasher(field.shape)
    chunk_entries: list[dict[str, object]] = []
    descriptor = os.open(field.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        expected_size = field.shape[0] * field.row_bytes
        if os.fstat(descriptor).st_size != expected_size:
            raise ValueError(f"temporary field {field.name} size drifted")
        for start in range(0, field.shape[0], row_chunk_size):
            count = min(row_chunk_size, field.shape[0] - start)
            raw = _read_exact(
                descriptor,
                start * field.row_bytes,
                count * field.row_bytes,
            )
            raw_digest.update(raw)
            array = np.frombuffer(raw, dtype=_NUMPY_DTYPE_BY_NAME[field.dtype_name])
            if not bool(np.isfinite(array).all()):
                raise ValueError(f"temporary field {field.name} is non-finite")
            float64 = np.asarray(array, dtype="<f8")
            semantic.update(float64.tobytes(order="C"))
            chunk_entries.append(
                {
                    "row_count": count,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "start_row": start,
                }
            )
    finally:
        os.close(descriptor)
    catalog = {
        "byte_count": field.shape[0] * field.row_bytes,
        "chunk_sha256s": chunk_entries,
        "dtype": field.dtype_name,
        "name": field.name,
        "offset": payload_offset,
        "semantic_float64_tensor_sha256": semantic.hexdigest(),
        "sha256": raw_digest.hexdigest(),
        "shape": list(field.shape),
    }
    return catalog


def _parse_row_keys(value: object) -> tuple[ChartRowKey, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("streaming chart row catalog must be a nonempty list")
    keys: list[ChartRowKey] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 4
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
            or type(row[2]) is not int
            or type(row[3]) is not int
        ):
            raise ValueError("streaming chart row key encoding drifted")
        keys.append(ChartRowKey(row[0], row[1], row[2], row[3]))
    result = tuple(keys)
    if len(set(result)) != len(result):
        raise ValueError("streaming chart row catalog repeats a row")
    return result


def _ensure_safe_destination(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_state = destination.parent.lstat()
    if not stat.S_ISDIR(parent_state.st_mode) or stat.S_ISLNK(parent_state.st_mode):
        raise ValueError("streaming artifact parent must be a real directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)


def _publish_fields(
    destination: Path,
    *,
    kind: str,
    metadata: Mapping[str, object],
    row_keys: tuple[ChartRowKey, ...],
    fields: Sequence[_TemporaryField],
    resources: StreamingChartResourceConfig,
) -> None:
    _ensure_safe_destination(destination)
    ordered = tuple(sorted(fields, key=lambda item: item.name))
    catalogs: list[dict[str, object]] = []
    offset = 0
    for field in ordered:
        catalog = _field_catalog(
            field,
            row_chunk_size=resources.row_chunk_size,
            payload_offset=offset,
        )
        catalogs.append(catalog)
        offset += int(catalog["byte_count"])
    if offset > resources.max_artifact_bytes:
        raise ValueError("streaming artifact payload exceeds max_artifact_bytes")
    payload_digest = hashlib.sha256()
    for field in ordered:
        descriptor = os.open(
            field.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            remaining = int(os.fstat(descriptor).st_size)
            position = 0
            while remaining:
                raw = os.pread(
                    descriptor, min(4 * 1024 * 1024, remaining), position
                )
                if not raw:
                    raise ValueError("temporary streaming field truncated")
                payload_digest.update(raw)
                position += len(raw)
                remaining -= len(raw)
        finally:
            os.close(descriptor)
    payload_sha256 = payload_digest.hexdigest()
    unsigned_header: dict[str, object] = {
        "blocks": catalogs,
        "kind": kind,
        "metadata": dict(metadata),
        "payload_byte_count": offset,
        "payload_sha256": payload_sha256,
        "row_keys": [key.payload() for key in row_keys],
        "schema": STREAMING_STORE_SCHEMA,
        "version": _STORE_VERSION,
    }
    header = {
        **unsigned_header,
        "artifact_sha256": _digest(_STORE_ARTIFACT_DOMAIN, unsigned_header),
    }
    header_bytes = _canonical_json_bytes(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("streaming artifact header exceeds the strict cap")
    total_size = len(_STORE_MAGIC) + 8 + len(header_bytes) + offset
    if total_size > resources.max_artifact_bytes:
        raise ValueError("streaming artifact exceeds max_artifact_bytes")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_STORE_MAGIC)
            output.write(struct.pack(">Q", len(header_bytes)))
            output.write(header_bytes)
            for field in ordered:
                input_descriptor = os.open(
                    field.path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    remaining = int(os.fstat(input_descriptor).st_size)
                    position = 0
                    while remaining:
                        raw = os.pread(
                            input_descriptor,
                            min(4 * 1024 * 1024, remaining),
                            position,
                        )
                        if not raw:
                            raise ValueError(
                                "temporary streaming field truncated during publish"
                            )
                        output.write(raw)
                        position += len(raw)
                        remaining -= len(raw)
                finally:
                    os.close(input_descriptor)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination)
        published = True
        temporary.unlink()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        _open_array_store(
            destination,
            expected_kind=kind,
            max_artifact_bytes=resources.max_artifact_bytes,
        )
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if published:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            try:
                directory_descriptor = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        raise


@dataclass(frozen=True, slots=True)
class _OpenedArrayStore:
    path: Path
    kind: str
    metadata: Mapping[str, object]
    row_keys: tuple[ChartRowKey, ...]
    blocks: Mapping[str, Mapping[str, object]]
    artifact_sha256: str
    payload_start: int
    device: int
    inode: int
    size: int
    mtime_ns: int

    @contextmanager
    def descriptor(self) -> Iterator[int]:
        before = self.path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_dev != self.device
            or before.st_ino != self.inode
            or before.st_size != self.size
            or before.st_mtime_ns != self.mtime_ns
        ):
            raise ValueError("streaming chart artifact changed after authentication")
        descriptor = os.open(
            self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != self.device
                or opened.st_ino != self.inode
                or opened.st_size != self.size
                or opened.st_mtime_ns != self.mtime_ns
            ):
                raise ValueError("streaming chart artifact changed during open")
            yield descriptor
        finally:
            os.close(descriptor)

    def read_slice(self, field: str, start: int, end: int) -> Tensor:
        if field not in self.blocks:
            raise KeyError(field)
        if type(start) is not int or type(end) is not int or not (0 <= start < end <= len(self.row_keys)):
            raise ValueError("streaming row slice is invalid")
        block = self.blocks[field]
        shape = tuple(block["shape"])
        itemsize = _NUMPY_DTYPE_BY_NAME[block["dtype"]].itemsize
        row_width = math.prod(shape[1:]) if len(shape) > 1 else 1
        row_bytes = row_width * itemsize
        with self.descriptor() as descriptor:
            raw = _read_exact(
                descriptor,
                self.payload_start + block["offset"] + start * row_bytes,
                (end - start) * row_bytes,
            )
        array = np.frombuffer(
            raw, dtype=_NUMPY_DTYPE_BY_NAME[block["dtype"]]
        )
        if not bool(np.isfinite(array).all()):
            raise ValueError("streaming chart row slice became non-finite")
        tensor = torch.from_numpy(array.copy())
        return tensor.reshape((end - start, *shape[1:]))

    def read_rows(self, field: str, indices: Sequence[int] | Tensor) -> Tensor:
        if isinstance(indices, Tensor):
            if indices.ndim != 1:
                raise ValueError("streaming gather indices must be rank one")
            materialized = [int(value) for value in indices.tolist()]
        else:
            materialized = list(indices)
        if not materialized:
            raise ValueError("streaming gather indices must be nonempty")
        if any(type(index) is not int or not 0 <= index < len(self.row_keys) for index in materialized):
            raise ValueError("streaming gather index is out of range")
        block = self.blocks[field]
        shape = tuple(block["shape"])
        result = torch.empty(
            (len(materialized), *shape[1:]),
            dtype=_DTYPE_BY_NAME[block["dtype"]],
        )
        ordered = sorted((source, output) for output, source in enumerate(materialized))
        cursor = 0
        while cursor < len(ordered):
            run_end = cursor + 1
            while (
                run_end < len(ordered)
                and ordered[run_end][0] == ordered[run_end - 1][0] + 1
            ):
                run_end += 1
            start_source = ordered[cursor][0]
            end_source = ordered[run_end - 1][0] + 1
            values = self.read_slice(field, start_source, end_source)
            for source, output in ordered[cursor:run_end]:
                result[output] = values[source - start_source]
            cursor = run_end
        return result


def _open_array_store(
    source: Path | str,
    *,
    expected_kind: str,
    max_artifact_bytes: int = _ABSOLUTE_MAX_ARTIFACT_BYTES,
) -> _OpenedArrayStore:
    path = Path(source)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError("streaming chart artifact must be a regular file")
    if before.st_size > min(max_artifact_bytes, _ABSOLUTE_MAX_ARTIFACT_BYTES):
        raise ValueError("streaming chart artifact exceeds the byte cap")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ValueError("streaming chart artifact changed during open")
        prefix = _read_exact(descriptor, 0, len(_STORE_MAGIC) + 8)
        if not prefix.startswith(_STORE_MAGIC):
            raise ValueError("streaming chart artifact magic is invalid")
        header_length = struct.unpack(">Q", prefix[len(_STORE_MAGIC) :])[0]
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError("streaming chart artifact header exceeds the cap")
        header_start = len(_STORE_MAGIC) + 8
        if header_start + header_length > opened.st_size:
            raise ValueError("streaming chart artifact header is truncated")
        header_raw = _read_exact(descriptor, header_start, header_length)
        header = _reference._strict_json_loads(header_raw)
        if not isinstance(header, Mapping) or set(header) != {
            "artifact_sha256",
            "blocks",
            "kind",
            "metadata",
            "payload_byte_count",
            "payload_sha256",
            "row_keys",
            "schema",
            "version",
        }:
            raise ValueError("streaming chart artifact header contract drifted")
        if (
            header["schema"] != STREAMING_STORE_SCHEMA
            or header["version"] != _STORE_VERSION
            or header["kind"] != expected_kind
            or not isinstance(header["metadata"], Mapping)
            or not isinstance(header["blocks"], list)
            or not header["blocks"]
            or type(header["payload_byte_count"]) is not int
            or header["payload_byte_count"] <= 0
        ):
            raise ValueError("streaming chart artifact identity drifted")
        _require_sha256(header["artifact_sha256"], label="artifact_sha256")
        _require_sha256(header["payload_sha256"], label="payload_sha256")
        unsigned = dict(header)
        del unsigned["artifact_sha256"]
        if _digest(_STORE_ARTIFACT_DOMAIN, unsigned) != header["artifact_sha256"]:
            raise ValueError("streaming chart artifact header digest drifted")
        row_keys = _parse_row_keys(header["row_keys"])
        declared_row_chunk_size = _require_positive_int(
            header["metadata"].get("row_chunk_size"),
            label="metadata.row_chunk_size",
        )
        blocks: dict[str, Mapping[str, object]] = {}
        expected_offset = 0
        previous_name: str | None = None
        payload_digest = hashlib.sha256()
        payload_start = header_start + header_length
        for entry in header["blocks"]:
            if not isinstance(entry, Mapping) or set(entry) != {
                "byte_count",
                "chunk_sha256s",
                "dtype",
                "name",
                "offset",
                "semantic_float64_tensor_sha256",
                "sha256",
                "shape",
            }:
                raise ValueError("streaming chart block contract drifted")
            if (
                not isinstance(entry["name"], str)
                or not entry["name"]
                or entry["dtype"] not in _DTYPE_BY_NAME
                or type(entry["offset"]) is not int
                or entry["offset"] != expected_offset
                or type(entry["byte_count"]) is not int
                or entry["byte_count"] <= 0
                or not isinstance(entry["shape"], list)
                or not entry["shape"]
                or any(type(size) is not int or size <= 0 for size in entry["shape"])
                or entry["shape"][0] != len(row_keys)
                or not isinstance(entry["chunk_sha256s"], list)
                or not entry["chunk_sha256s"]
            ):
                raise ValueError("streaming chart block metadata is invalid")
            for label in ("sha256", "semantic_float64_tensor_sha256"):
                _require_sha256(entry[label], label=label)
            if previous_name is not None and entry["name"] <= previous_name:
                raise ValueError("streaming chart block names are not canonical")
            if entry["name"] in blocks:
                raise ValueError("streaming chart artifact repeats a block")
            shape = tuple(entry["shape"])
            itemsize = _NUMPY_DTYPE_BY_NAME[entry["dtype"]].itemsize
            expected_bytes = _checked_product(shape, label="block shape") * itemsize
            if entry["byte_count"] != expected_bytes:
                raise ValueError("streaming chart block shape/size drifted")
            raw_digest = hashlib.sha256()
            semantic = _tensor_semantic_hasher(shape)
            expected_start = 0
            chunk_total_bytes = 0
            for chunk in entry["chunk_sha256s"]:
                if not isinstance(chunk, Mapping) or set(chunk) != {
                    "row_count",
                    "sha256",
                    "start_row",
                }:
                    raise ValueError("streaming chart chunk catalog drifted")
                if (
                    type(chunk["start_row"]) is not int
                    or chunk["start_row"] != expected_start
                    or type(chunk["row_count"]) is not int
                    or chunk["row_count"] <= 0
                    or expected_start + chunk["row_count"] > shape[0]
                ):
                    raise ValueError("streaming chart chunk schedule drifted")
                _require_sha256(chunk["sha256"], label="chunk sha256")
                row_width = math.prod(shape[1:]) if len(shape) > 1 else 1
                size = chunk["row_count"] * row_width * itemsize
                expected_chunk_rows = min(
                    declared_row_chunk_size, shape[0] - chunk["start_row"]
                )
                if (
                    chunk["row_count"] != expected_chunk_rows
                    or size > _MAX_CHUNK_BYTES
                ):
                    raise ValueError(
                        "streaming chart chunk violates the fixed bounded schedule"
                    )
                raw = _read_exact(
                    descriptor,
                    payload_start + entry["offset"] + chunk_total_bytes,
                    size,
                )
                if hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
                    raise ValueError("streaming chart chunk digest drifted")
                array = np.frombuffer(raw, dtype=_NUMPY_DTYPE_BY_NAME[entry["dtype"]])
                if not bool(np.isfinite(array).all()):
                    raise ValueError("streaming chart artifact contains non-finite rows")
                raw_digest.update(raw)
                payload_digest.update(raw)
                semantic.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
                expected_start += chunk["row_count"]
                chunk_total_bytes += size
            if expected_start != shape[0] or chunk_total_bytes != entry["byte_count"]:
                raise ValueError("streaming chart chunk coverage drifted")
            if raw_digest.hexdigest() != entry["sha256"]:
                raise ValueError("streaming chart block digest drifted")
            if semantic.hexdigest() != entry["semantic_float64_tensor_sha256"]:
                raise ValueError("streaming chart semantic tensor digest drifted")
            blocks[entry["name"]] = dict(entry)
            expected_offset += entry["byte_count"]
            previous_name = entry["name"]
        if (
            expected_offset != header["payload_byte_count"]
            or payload_start + expected_offset != opened.st_size
            or payload_digest.hexdigest() != header["payload_sha256"]
        ):
            raise ValueError("streaming chart payload catalog drifted")
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError("streaming chart artifact changed during authentication")
    return _OpenedArrayStore(
        path=path,
        kind=expected_kind,
        metadata=dict(header["metadata"]),
        row_keys=row_keys,
        blocks=blocks,
        artifact_sha256=header["artifact_sha256"],
        payload_start=payload_start,
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


@dataclass(frozen=True, slots=True)
class StreamingRowChunk:
    start_row: int
    end_row: int
    row_keys: tuple[ChartRowKey, ...]
    tensors: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class RawModeRowSpool:
    _store: _OpenedArrayStore
    role: str
    panel_binding_sha256: str
    mode_id: str
    row_catalog_sha256: str
    tensor_catalog_sha256: str
    hidden_width: int
    row_chunk_size: int
    kernel_receipt_sha256: str

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def artifact_sha256(self) -> str:
        return self._store.artifact_sha256

    @property
    def row_keys(self) -> tuple[ChartRowKey, ...]:
        return self._store.row_keys

    @property
    def row_count(self) -> int:
        return len(self.row_keys)

    @property
    def field_semantic_sha256s(self) -> dict[str, str]:
        return {
            name: str(block["semantic_float64_tensor_sha256"])
            for name, block in self._store.blocks.items()
        }

    def iter_float64_chunks(
        self,
        *,
        fields: Sequence[str] = (
            "context_values",
            "comparison_values",
            "generalized_contributions",
        ),
        chunk_size: int | None = None,
    ) -> Iterator[StreamingRowChunk]:
        names = tuple(fields)
        if not names or len(set(names)) != len(names) or any(name not in _RAW_FIELDS for name in names):
            raise ValueError("raw spool fields are invalid")
        resolved = self.row_chunk_size if chunk_size is None else chunk_size
        _require_positive_int(resolved, label="chunk_size")
        for start in range(0, self.row_count, resolved):
            end = min(self.row_count, start + resolved)
            yield StreamingRowChunk(
                start_row=start,
                end_row=end,
                row_keys=self.row_keys[start:end],
                tensors={
                    name: self._store.read_slice(name, start, end).to(
                        dtype=torch.float64
                    )
                    for name in names
                },
            )


class RawModeRowSpoolWriter:
    """Append fixed catalog rows to private field files, then atomically seal."""

    def __init__(
        self,
        destination: Path,
        *,
        role: str,
        panel_binding_sha256: str,
        mode_id: str,
        row_keys: tuple[ChartRowKey, ...],
        hidden_width: int,
        resources: StreamingChartResourceConfig,
    ) -> None:
        if not role or not mode_id:
            raise ValueError("raw spool role and mode_id must be nonempty")
        _require_sha256(panel_binding_sha256, label="panel_binding_sha256")
        _require_positive_int(hidden_width, label="hidden_width")
        if not isinstance(resources, StreamingChartResourceConfig):
            raise TypeError("resources must be StreamingChartResourceConfig")
        catalog = row_catalog_sha256(row_keys, role=role)
        _ensure_safe_destination(destination)
        _enforce_resident_budget(
            resources,
            required_bytes=resources.row_chunk_size * hidden_width * 3 * 8,
            operation="raw spool append",
        )
        self.destination = destination
        self.role = role
        self.panel_binding_sha256 = panel_binding_sha256
        self.mode_id = mode_id
        self.row_keys = row_keys
        self.hidden_width = hidden_width
        self.resources = resources
        self.row_catalog_sha256 = catalog
        self._next_row = 0
        self._closed = False
        self._fields = {
            name: _create_temporary_field(
                destination.parent,
                name=name,
                dtype_name="float32_le",
                shape=(len(row_keys), hidden_width),
            )
            for name in sorted(_RAW_FIELDS)
        }

    @classmethod
    def create(
        cls,
        destination: Path | str,
        *,
        role: str,
        panel_binding_sha256: str,
        mode_id: str,
        row_keys: Sequence[ChartRowKey],
        hidden_width: int,
        resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
    ) -> "RawModeRowSpoolWriter":
        keys = tuple(row_keys)
        return cls(
            Path(destination),
            role=role,
            panel_binding_sha256=panel_binding_sha256,
            mode_id=mode_id,
            row_keys=keys,
            hidden_width=hidden_width,
            resources=resources,
        )

    def append(
        self,
        row_keys: Sequence[ChartRowKey],
        context_values: Tensor,
        comparison_values: Tensor,
        generalized_contributions: Tensor,
    ) -> None:
        if self._closed:
            raise RuntimeError("raw spool writer is closed")
        keys = tuple(row_keys)
        if not keys:
            raise ValueError("raw spool append must be nonempty")
        if len(keys) > self.resources.row_chunk_size:
            raise ValueError("raw spool append exceeds the fixed row chunk size")
        expected = self.row_keys[self._next_row : self._next_row + len(keys)]
        if keys != expected:
            raise ValueError("raw spool append row catalog/order drifted")
        values = {
            "comparison_values": comparison_values,
            "context_values": context_values,
            "generalized_contributions": generalized_contributions,
        }
        for name, value in values.items():
            if tuple(value.shape) != (len(keys), self.hidden_width):
                raise ValueError(f"{name} shape disagrees with appended row keys")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating rows")
            converted = value.detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f"{name} overflows the float32 raw spool")
        for name, value in values.items():
            self._fields[name].append(value)
        self._next_row += len(keys)

    def seal(self) -> RawModeRowSpool:
        if self._closed:
            raise RuntimeError("raw spool writer is closed")
        self._closed = True
        try:
            if self._next_row != len(self.row_keys):
                raise ValueError("raw spool is incomplete")
            for field in self._fields.values():
                field.finish()
            semantic = {
                name: _field_catalog(
                    field,
                    row_chunk_size=self.resources.row_chunk_size,
                    payload_offset=0,
                )["semantic_float64_tensor_sha256"]
                for name, field in self._fields.items()
            }
            tensor_catalog = _reference._digest(
                _reference._ROW_BATCH_DOMAIN,
                {
                    "mode_id": self.mode_id,
                    "panel_binding_sha256": self.panel_binding_sha256,
                    "role": self.role,
                    "row_catalog_sha256": self.row_catalog_sha256,
                    "tensor_sha256s": semantic,
                },
            )
            kernel = streaming_chart_kernel_receipt(self.resources)
            metadata = {
                "hidden_width": self.hidden_width,
                "kernel_receipt_sha256": kernel["receipt_sha256"],
                "mode_id": self.mode_id,
                "panel_binding_sha256": self.panel_binding_sha256,
                "resource_config": self.resources.payload(),
                "role": self.role,
                "row_catalog_sha256": self.row_catalog_sha256,
                "row_chunk_size": self.resources.row_chunk_size,
                "tensor_catalog_sha256": tensor_catalog,
            }
            _publish_fields(
                self.destination,
                kind=RAW_MODE_ROW_SPOOL_KIND,
                metadata=metadata,
                row_keys=self.row_keys,
                fields=tuple(self._fields.values()),
                resources=self.resources,
            )
            return open_raw_mode_row_spool(
                self.destination,
                max_artifact_bytes=self.resources.max_artifact_bytes,
            )
        finally:
            for field in self._fields.values():
                field.abort()

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            for field in self._fields.values():
                field.abort()

    def __enter__(self) -> "RawModeRowSpoolWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            self.abort()


def open_raw_mode_row_spool(
    source: Path | str,
    *,
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
) -> RawModeRowSpool:
    store = _open_array_store(
        source,
        expected_kind=RAW_MODE_ROW_SPOOL_KIND,
        max_artifact_bytes=max_artifact_bytes,
    )
    metadata = store.metadata
    expected = {
        "hidden_width",
        "kernel_receipt_sha256",
        "mode_id",
        "panel_binding_sha256",
        "resource_config",
        "role",
        "row_catalog_sha256",
        "row_chunk_size",
        "tensor_catalog_sha256",
    }
    if set(metadata) != expected or set(store.blocks) != set(_RAW_FIELDS):
        raise ValueError("raw mode spool metadata/field contract drifted")
    if not isinstance(metadata["role"], str) or not metadata["role"]:
        raise ValueError("raw mode spool role is invalid")
    if not isinstance(metadata["mode_id"], str) or not metadata["mode_id"]:
        raise ValueError("raw mode spool mode_id is invalid")
    _require_sha256(metadata["panel_binding_sha256"], label="panel_binding_sha256")
    _require_sha256(metadata["row_catalog_sha256"], label="row_catalog_sha256")
    _require_sha256(metadata["tensor_catalog_sha256"], label="tensor_catalog_sha256")
    _require_sha256(metadata["kernel_receipt_sha256"], label="kernel_receipt_sha256")
    width = _require_positive_int(metadata["hidden_width"], label="hidden_width")
    chunk = _require_positive_int(metadata["row_chunk_size"], label="row_chunk_size")
    resource_config = _resource_config_from_payload(metadata["resource_config"])
    if (
        resource_config.row_chunk_size != chunk
        or store.size > resource_config.max_artifact_bytes
        or metadata["kernel_receipt_sha256"]
        != streaming_chart_kernel_receipt(resource_config)["receipt_sha256"]
    ):
        raise ValueError("raw mode spool resource/kernel receipt drifted")
    _validate_fixed_block_chunks(store, row_chunk_size=chunk)
    if metadata["row_catalog_sha256"] != row_catalog_sha256(
        store.row_keys, role=metadata["role"]
    ):
        raise ValueError("raw mode spool row catalog digest drifted")
    semantic: dict[str, str] = {}
    for name, block in store.blocks.items():
        if block["dtype"] != "float32_le" or tuple(block["shape"]) != (
            len(store.row_keys),
            width,
        ):
            raise ValueError("raw mode spool field shape/dtype drifted")
        semantic[name] = str(block["semantic_float64_tensor_sha256"])
    expected_tensor_catalog = _reference._digest(
        _reference._ROW_BATCH_DOMAIN,
        {
            "mode_id": metadata["mode_id"],
            "panel_binding_sha256": metadata["panel_binding_sha256"],
            "role": metadata["role"],
            "row_catalog_sha256": metadata["row_catalog_sha256"],
            "tensor_sha256s": semantic,
        },
    )
    if expected_tensor_catalog != metadata["tensor_catalog_sha256"]:
        raise ValueError("raw mode spool tensor catalog digest drifted")
    return RawModeRowSpool(
        _store=store,
        role=metadata["role"],
        panel_binding_sha256=metadata["panel_binding_sha256"],
        mode_id=metadata["mode_id"],
        row_catalog_sha256=metadata["row_catalog_sha256"],
        tensor_catalog_sha256=metadata["tensor_catalog_sha256"],
        hidden_width=width,
        row_chunk_size=chunk,
        kernel_receipt_sha256=metadata["kernel_receipt_sha256"],
    )


def write_raw_mode_row_spool(
    batch: SharedChartRowBatch,
    destination: Path | str,
    *,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> RawModeRowSpool:
    """Fixture/conversion helper; production capture should append directly."""

    writer = RawModeRowSpoolWriter.create(
        destination,
        role=batch.role,
        panel_binding_sha256=batch.panel_binding_sha256,
        mode_id=batch.mode_id,
        row_keys=batch.row_keys,
        hidden_width=batch.width,
        resources=resources,
    )
    try:
        for start in range(0, len(batch.row_keys), resources.row_chunk_size):
            end = min(len(batch.row_keys), start + resources.row_chunk_size)
            writer.append(
                batch.row_keys[start:end],
                batch.context_values[start:end],
                batch.comparison_values[start:end],
                batch.generalized_contributions[start:end],
            )
        return writer.seal()
    except BaseException:
        writer.abort()
        raise


def _aligned_raw_spools(
    spools: Sequence[RawModeRowSpool],
    *,
    resources: StreamingChartResourceConfig,
    expected_role: str | None = None,
) -> tuple[RawModeRowSpool, ...]:
    aligned = tuple(sorted(spools, key=lambda item: item.mode_id))
    if not aligned or len({item.mode_id for item in aligned}) != len(aligned):
        raise ValueError("raw spools must contain unique modes")
    first = aligned[0]
    if expected_role is not None and first.role != expected_role:
        raise ValueError(f"raw spools must use role {expected_role}")
    expected_kernel = streaming_chart_kernel_receipt(resources)["receipt_sha256"]
    for item in aligned:
        if (
            item.role != first.role
            or item.panel_binding_sha256 != first.panel_binding_sha256
            or item.row_catalog_sha256 != first.row_catalog_sha256
            or item.row_keys != first.row_keys
            or item.hidden_width != first.hidden_width
            or item.row_chunk_size != resources.row_chunk_size
            or item.kernel_receipt_sha256 != expected_kernel
        ):
            raise ValueError(
                "all raw spools must share one authenticated role/panel/row/kernel catalog"
            )
    return aligned


def _balanced_row_weights(
    row_keys: tuple[ChartRowKey, ...], *, mode_count: int
) -> Tensor:
    families = tuple(sorted({key.family_id for key in row_keys}))
    examples_by_family = {
        family: tuple(
            sorted(
                {
                    key.example_id
                    for key in row_keys
                    if key.family_id == family
                }
            )
        )
        for family in families
    }
    row_counts = {
        (family, example): sum(
            key.family_id == family and key.example_id == example
            for key in row_keys
        )
        for family in families
        for example in examples_by_family[family]
    }
    result = torch.empty(len(row_keys), dtype=torch.float64)
    for index, key in enumerate(row_keys):
        result[index] = 1.0 / (
            mode_count
            * len(families)
            * len(examples_by_family[key.family_id])
            * row_counts[(key.family_id, key.example_id)]
        )
    if not math.isclose(
        float(result.sum()) * mode_count,
        1.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("streaming balanced row weights did not close to one")
    return result


def _iter_context_chunks(
    spools: tuple[RawModeRowSpool, ...],
    *,
    base_weights: Tensor,
    chunk_size: int,
) -> Iterator[tuple[int, Tensor, Tensor]]:
    rows_per_mode = len(spools[0].row_keys)
    for mode_ordinal, spool in enumerate(spools):
        for chunk in spool.iter_float64_chunks(
            fields=("context_values",), chunk_size=chunk_size
        ):
            yield (
                mode_ordinal * rows_per_mode + chunk.start_row,
                chunk.tensors["context_values"],
                base_weights[chunk.start_row : chunk.end_row],
            )


def _norm_identity_distances(values: Tensor, centers: Tensor) -> Tensor:
    if values.dtype != torch.float64 or centers.dtype != torch.float64:
        raise ValueError("norm-identity distances require float64 inputs")
    squared = (
        values.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1)[None, :]
        - 2.0 * torch.mm(values, centers.T)
    )
    return squared.clamp_min_(0.0)


def _streaming_memberships(
    *,
    contexts: Tensor,
    centers: Tensor,
    radius_squared: Tensor,
    temperature: float,
) -> Tensor:
    distances = _norm_identity_distances(contexts, centers)
    logits = -0.5 * distances / (radius_squared[None, :] * temperature)
    return torch.softmax(logits, dim=1)


def fit_shared_chart_atlas_streaming(
    raw_spools: Sequence[RawModeRowSpool],
    *,
    config: SharedChartNumericsConfig = SharedChartNumericsConfig(),
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> SharedChartAtlas:
    """Fit the reference chart geometry in fixed-order, bounded-memory passes.

    Float32-representable fixtures retain the reference topology and agree at
    strict float64 tolerance.  The artifact receipt authenticates this
    implementation's chunked reduction order rather than claiming byte-exact
    equality with the in-memory reduction tree.
    """

    if not isinstance(config, SharedChartNumericsConfig):
        raise TypeError("config must be SharedChartNumericsConfig")
    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")
    if config.lloyd_iteration_count != 12:
        raise ValueError("production streaming atlas requires exactly 12 Lloyd passes")
    aligned = _aligned_raw_spools(
        raw_spools, resources=resources, expected_role=FIT_ROLE
    )
    if aligned[0].hidden_width != config.hidden_width:
        raise ValueError("raw spool width disagrees with atlas config")
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=len(aligned[0].row_keys),
        mode_count=len(aligned),
        config=config,
        resources=resources,
    )
    _enforce_resident_budget(
        resources,
        required_bytes=estimate.atlas_peak_resident_bytes,
        operation="streaming atlas fit",
    )
    total_rows = len(aligned) * len(aligned[0].row_keys)
    if total_rows < config.chart_count * (config.tangent_rank + 1):
        raise ValueError("atlas fit has too few rows for chart/rank occupancy")
    base_weights = _balanced_row_weights(
        aligned[0].row_keys, mode_count=len(aligned)
    )

    weighted_sum = torch.zeros(config.hidden_width, dtype=torch.float64)
    weight_sum = torch.zeros((), dtype=torch.float64)
    for _, contexts, weights in _iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        weighted_sum += (weights[:, None] * contexts).sum(dim=0)
        weight_sum += weights.sum()
    global_mean = weighted_sum / weight_sum

    selected_indices: list[int] = []
    selected_centers: list[Tensor] = []
    best_score = -math.inf
    best_index = -1
    best_center: Tensor | None = None
    mean_center = global_mean[None, :]
    for global_start, contexts, weights in _iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        scores = weights * _norm_identity_distances(
            contexts, mean_center
        )[:, 0]
        local = int(torch.argmax(scores).item())
        score = float(scores[local])
        if score > best_score:
            best_score = score
            best_index = global_start + local
            best_center = contexts[local].clone()
    if best_center is None:
        raise RuntimeError("streaming farthest initialization found no row")
    selected_indices.append(best_index)
    selected_centers.append(best_center)
    while len(selected_centers) < config.chart_count:
        centers_now = torch.stack(selected_centers)
        best_score = -math.inf
        best_index = -1
        best_center = None
        selected_set = set(selected_indices)
        for global_start, contexts, weights in _iter_context_chunks(
            aligned,
            base_weights=base_weights,
            chunk_size=resources.atlas_chunk_size,
        ):
            distances = _norm_identity_distances(contexts, centers_now)
            scores = weights * distances.min(dim=1).values
            for selected_index in selected_set:
                local = selected_index - global_start
                if 0 <= local < len(scores):
                    scores[local] = -1.0
            local = int(torch.argmax(scores).item())
            score = float(scores[local])
            if score > best_score:
                best_score = score
                best_index = global_start + local
                best_center = contexts[local].clone()
        if best_center is None or best_score <= 0.0:
            raise ValueError("farthest-point chart initialization degenerated")
        selected_indices.append(best_index)
        selected_centers.append(best_center)
    centers = torch.stack(selected_centers)

    for _ in range(config.lloyd_iteration_count):
        sums = torch.zeros_like(centers)
        occupancy = torch.zeros(config.chart_count, dtype=torch.float64)
        for _, contexts, weights in _iter_context_chunks(
            aligned,
            base_weights=base_weights,
            chunk_size=resources.atlas_chunk_size,
        ):
            assignments = torch.argmin(
                _norm_identity_distances(contexts, centers), dim=1
            )
            for chart in range(config.chart_count):
                mask = assignments == chart
                if bool(mask.any()):
                    chart_weights = weights[mask]
                    sums[chart] += (
                        chart_weights[:, None] * contexts[mask]
                    ).sum(dim=0)
                    occupancy[chart] += chart_weights.sum()
        if bool((occupancy <= 0.0).any()):
            raise ValueError("weighted Lloyd iteration produced an empty chart")
        centers = sums / occupancy[:, None]

    order = sorted(
        range(config.chart_count),
        key=lambda index: tuple(float(value) for value in centers[index]),
    )
    centers = centers[order].contiguous()
    counts = torch.zeros(config.chart_count, dtype=torch.int64)
    occupancy_weights = torch.zeros(config.chart_count, dtype=torch.float64)
    radius_numerators = torch.zeros(config.chart_count, dtype=torch.float64)
    context_reference_sum = torch.zeros((), dtype=torch.float64)
    global_variance_sum = torch.zeros((), dtype=torch.float64)
    for _, contexts, weights in _iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        distances = _norm_identity_distances(contexts, centers)
        assignments = torch.argmin(distances, dim=1)
        counts += torch.bincount(assignments, minlength=config.chart_count)
        for chart in range(config.chart_count):
            mask = assignments == chart
            if bool(mask.any()):
                chart_weights = weights[mask]
                occupancy_weights[chart] += chart_weights.sum()
                radius_numerators[chart] += (
                    chart_weights * distances[mask, chart]
                ).sum()
        context_reference_sum += (
            weights * contexts.square().sum(dim=1)
        ).sum()
        global_variance_sum += (
            weights
            * _norm_identity_distances(contexts, global_mean[None, :])[:, 0]
        ).sum()
    minimum_rows = config.tangent_rank + 1
    if bool((counts < minimum_rows).any()):
        raise ValueError("a chart has fewer than tangent_rank+1 hard rows")
    if bool(
        (occupancy_weights < config.minimum_hard_occupancy_fraction).any()
    ):
        raise ValueError("a chart failed the minimum balanced hard occupancy gate")
    context_reference = max(float(context_reference_sum), 1.0e-30)
    radius_floor = config.radius_floor_fraction * context_reference
    radii = torch.maximum(
        radius_numerators / occupancy_weights,
        torch.full(
            (config.chart_count,), radius_floor, dtype=torch.float64
        ),
    )

    covariance_numerators = torch.zeros(
        (config.chart_count, config.hidden_width, config.hidden_width),
        dtype=torch.float64,
    )
    soft_occupancy = torch.zeros(config.chart_count, dtype=torch.float64)
    for _, contexts, weights in _iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        memberships = _streaming_memberships(
            contexts=contexts,
            centers=centers,
            radius_squared=radii,
            temperature=config.membership_temperature,
        )
        for chart in range(config.chart_count):
            chart_weights = weights * memberships[:, chart]
            centered = contexts - centers[chart]
            covariance_numerators[chart] += centered.T @ (
                chart_weights[:, None] * centered
            )
            soft_occupancy[chart] += chart_weights.sum()
    covariances = covariance_numerators / soft_occupancy[:, None, None]
    context_variance_reference = max(
        float(global_variance_sum) / config.hidden_width,
        1.0e-30,
    )
    variance_floor = (
        config.variance_floor_fraction * context_variance_reference
    )
    bases = torch.empty(
        (config.chart_count, config.hidden_width, config.tangent_rank),
        dtype=torch.float64,
    )
    tangent_variance = torch.empty(
        (config.chart_count, config.tangent_rank), dtype=torch.float64
    )
    for chart in range(config.chart_count):
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances[chart])
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        for ordinal in range(config.tangent_rank):
            following = eigenvalues[ordinal + 1]
            scale = max(
                1.0,
                abs(float(eigenvalues[ordinal])),
                abs(float(following)),
            )
            if abs(float(eigenvalues[ordinal] - following)) <= (
                config.eigenvalue_tie_relative_tolerance * scale
            ):
                raise ValueError("chart tangent eigenvalue orientation is undefined")
        basis = _reference._canonicalize_vector_signs(
            eigenvectors[:, : config.tangent_rank]
        )
        bases[chart] = basis
        tangent_variance[chart] = torch.clamp(
            eigenvalues[: config.tangent_rank], min=variance_floor
        )

    orthogonal_numerators = torch.zeros(
        config.chart_count, dtype=torch.float64
    )
    for _, contexts, weights in _iter_context_chunks(
        aligned,
        base_weights=base_weights,
        chunk_size=resources.atlas_chunk_size,
    ):
        memberships = _streaming_memberships(
            contexts=contexts,
            centers=centers,
            radius_squared=radii,
            temperature=config.membership_temperature,
        )
        for chart in range(config.chart_count):
            centered = contexts - centers[chart]
            projection = centered @ bases[chart]
            residual_square = (
                centered.square().sum(dim=1)
                - projection.square().sum(dim=1)
            ).clamp_min(0.0)
            orthogonal_numerators[chart] += (
                weights * memberships[:, chart] * residual_square
            ).sum()
    alpha = torch.empty(config.chart_count, dtype=torch.float64)
    for chart in range(config.chart_count):
        orthogonal_variance = float(
            orthogonal_numerators[chart]
            / (
                soft_occupancy[chart]
                * (config.hidden_width - config.tangent_rank)
            )
        )
        alpha[chart] = 1.0 / max(orthogonal_variance, variance_floor)

    fit_context_catalog_sha256 = _reference._digest(
        _reference._ATLAS_CONTEXT_CATALOG_DOMAIN,
        {
            "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
            "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
            "modes": [
                {
                    "context_values_sha256": (
                        item.field_semantic_sha256s["context_values"]
                    ),
                    "mode_id": item.mode_id,
                }
                for item in aligned
            ],
        },
    )
    weight_sha256 = _reference._tensor_sha256(base_weights)
    fit_balanced_row_weight_catalog_sha256 = _reference._digest(
        _reference._ATLAS_WEIGHT_CATALOG_DOMAIN,
        {
            "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
            "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
            "mode_weights": [
                {"mode_id": item.mode_id, "weight_sha256": weight_sha256}
                for item in aligned
            ],
        },
    )
    kernel = streaming_chart_kernel_receipt(resources)
    fit_solver_receipt_sha256 = _reference._digest(
        _reference._ATLAS_SOLVER_RECEIPT_DOMAIN,
        {
            "config": config.payload(),
            "fit_balanced_row_weight_catalog_sha256": (
                fit_balanced_row_weight_catalog_sha256
            ),
            "fit_context_catalog_sha256": fit_context_catalog_sha256,
            "kernel_receipt_sha256": kernel["receipt_sha256"],
            "resource_estimate_receipt_sha256": estimate.receipt_sha256,
            "solver_id": (
                "streamed_balanced_norm_identity_farthest_"
                "weighted_lloyd12_soft_covariance_eigh_v1"
            ),
        },
    )
    values = {
        "config": config,
        "fit_role": FIT_ROLE,
        "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
        "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
        "fit_mode_ids": tuple(item.mode_id for item in aligned),
        "fit_context_catalog_sha256": fit_context_catalog_sha256,
        "fit_balanced_row_weight_catalog_sha256": (
            fit_balanced_row_weight_catalog_sha256
        ),
        "fit_solver_receipt_sha256": fit_solver_receipt_sha256,
        "centers": centers,
        "radius_squared": radii,
        "tangent_bases": bases,
        "tangent_variance": tangent_variance,
        "orthogonal_alpha": alpha,
        "hard_occupancy_count": counts,
        "hard_occupancy_weight": occupancy_weights,
    }
    provisional = object.__new__(SharedChartAtlas)
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    object.__setattr__(provisional, "atlas_sha256", "0" * 64)
    return SharedChartAtlas(
        **values,
        atlas_sha256=_reference._atlas_sha256(provisional),
    )


def _config_from_payload(value: object) -> SharedChartNumericsConfig:
    expected = set(SharedChartNumericsConfig().payload())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("streaming chart numerics config payload drifted")
    try:
        return SharedChartNumericsConfig(**dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("streaming chart numerics config is invalid") from error


@dataclass(frozen=True, slots=True)
class StreamingAppliedChartStore:
    _store: _OpenedArrayStore
    config: SharedChartNumericsConfig
    role: str
    panel_binding_sha256: str
    mode_id: str
    row_catalog_sha256: str
    atlas_sha256: str
    source_raw_artifact_sha256: str
    source_tensor_catalog_sha256: str
    vector_catalog_sha256: str
    row_chunk_size: int
    kernel_receipt_sha256: str

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def artifact_sha256(self) -> str:
        return self._store.artifact_sha256

    @property
    def row_keys(self) -> tuple[ChartRowKey, ...]:
        return self._store.row_keys

    @property
    def row_count(self) -> int:
        return len(self.row_keys)

    @property
    def field_semantic_sha256s(self) -> dict[str, str]:
        return {
            name: str(block["semantic_float64_tensor_sha256"])
            for name, block in self._store.blocks.items()
        }

    def read_slice(
        self, field: str, start: int, end: int, *, float64: bool = True
    ) -> Tensor:
        if field not in _APPLIED_FIELD_NAMES:
            raise KeyError(field)
        value = self._store.read_slice(field, start, end)
        return value.to(dtype=torch.float64) if float64 else value

    def read_rows(
        self,
        field: str,
        indices: Sequence[int] | Tensor,
        *,
        float64: bool = True,
    ) -> Tensor:
        if field not in _APPLIED_FIELD_NAMES:
            raise KeyError(field)
        value = self._store.read_rows(field, indices)
        return value.to(dtype=torch.float64) if float64 else value

    def iter_float64_chunks(
        self,
        *,
        fields: Sequence[str],
        chunk_size: int | None = None,
    ) -> Iterator[StreamingRowChunk]:
        names = tuple(fields)
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in _APPLIED_FIELD_NAMES for name in names)
        ):
            raise ValueError("applied store fields are invalid")
        resolved = self.row_chunk_size if chunk_size is None else chunk_size
        _require_positive_int(resolved, label="chunk_size")
        for start in range(0, self.row_count, resolved):
            end = min(self.row_count, start + resolved)
            yield StreamingRowChunk(
                start_row=start,
                end_row=end,
                row_keys=self.row_keys[start:end],
                tensors={
                    name: self.read_slice(name, start, end)
                    for name in names
                },
            )


def apply_shared_chart_atlas_streaming(
    atlas: SharedChartAtlas,
    raw_spool: RawModeRowSpool,
    destination: Path | str,
    *,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> StreamingAppliedChartStore:
    """Apply one frozen atlas and seal only transport/evaluation sufficients."""

    if not isinstance(atlas, SharedChartAtlas):
        raise TypeError("atlas must be SharedChartAtlas")
    if not isinstance(raw_spool, RawModeRowSpool):
        raise TypeError("raw_spool must be RawModeRowSpool")
    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")
    if raw_spool.hidden_width != atlas.config.hidden_width:
        raise ValueError("raw spool width disagrees with shared atlas")
    expected_kernel = streaming_chart_kernel_receipt(resources)["receipt_sha256"]
    if (
        raw_spool.row_chunk_size != resources.row_chunk_size
        or raw_spool.kernel_receipt_sha256 != expected_kernel
    ):
        raise ValueError("raw spool uses a different authenticated chunk kernel")
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=raw_spool.row_count,
        mode_count=max(2, len(atlas.fit_mode_ids)),
        config=atlas.config,
        resources=resources,
    )
    _enforce_resident_budget(
        resources,
        required_bytes=estimate.apply_peak_resident_bytes,
        operation="streaming atlas apply",
    )
    destination_path = Path(destination)
    _ensure_safe_destination(destination_path)
    shapes = {
        "features": (
            raw_spool.row_count,
            atlas.config.source_feature_width,
        ),
        "generalized_contributions": (
            raw_spool.row_count,
            atlas.config.hidden_width,
        ),
        "memberships": (raw_spool.row_count, atlas.config.chart_count),
        "weighted_activation_energy": (raw_spool.row_count,),
        "weighted_chart_orthogonal_energy": (raw_spool.row_count,),
        "weighted_chart_tangent_energy": (raw_spool.row_count,),
        "weighted_fisher_energy": (raw_spool.row_count,),
    }
    dtype_names = {
        name: (
            "float32_le" if name == "generalized_contributions" else "float64_le"
        )
        for name in shapes
    }
    fields = {
        name: _create_temporary_field(
            destination_path.parent,
            name=name,
            dtype_name=dtype_names[name],
            shape=shape,
        )
        for name, shape in shapes.items()
    }
    try:
        for chunk in raw_spool.iter_float64_chunks(
            chunk_size=resources.row_chunk_size
        ):
            context = chunk.tensors["context_values"]
            values = chunk.tensors["comparison_values"]
            contribution = chunk.tensors["generalized_contributions"]
            memberships = _streaming_memberships(
                contexts=context,
                centers=atlas.centers,
                radius_squared=atlas.radius_squared,
                temperature=atlas.config.membership_temperature,
            )
            coordinates = torch.empty(
                (
                    len(context),
                    atlas.config.chart_count,
                    atlas.config.tangent_rank,
                ),
                dtype=torch.float64,
            )
            for chart in range(atlas.config.chart_count):
                coordinates[:, chart, :] = (
                    (context - atlas.centers[chart])
                    @ atlas.tangent_bases[chart]
                ) / torch.sqrt(atlas.tangent_variance[chart])
            ones = torch.ones(
                (len(context), atlas.config.chart_count, 1),
                dtype=torch.float64,
            )
            features = (
                memberships[:, :, None]
                * torch.cat((ones, coordinates), dim=2)
            ).reshape(len(context), atlas.config.source_feature_width)
            activation = _reference._spd_energy_parts(
                atlas, memberships, values
            )[2]
            tangent, orthogonal, fisher = _reference._spd_energy_parts(
                atlas, memberships, contribution
            )
            if not torch.allclose(
                tangent + orthogonal,
                fisher,
                atol=1.0e-9,
                rtol=1.0e-9,
            ):
                raise RuntimeError("streaming applied Fisher energy did not close")
            payload = {
                "features": features,
                "generalized_contributions": contribution,
                "memberships": memberships,
                "weighted_activation_energy": activation,
                "weighted_chart_orthogonal_energy": orthogonal,
                "weighted_chart_tangent_energy": tangent,
                "weighted_fisher_energy": fisher,
            }
            for name, value in payload.items():
                fields[name].append(value)
        for field in fields.values():
            field.finish()
        catalogs = {
            name: _field_catalog(
                field,
                row_chunk_size=resources.row_chunk_size,
                payload_offset=0,
            )
            for name, field in fields.items()
        }
        semantic = {
            name: str(catalog["semantic_float64_tensor_sha256"])
            for name, catalog in catalogs.items()
        }
        if (
            semantic["generalized_contributions"]
            != raw_spool.field_semantic_sha256s["generalized_contributions"]
        ):
            raise RuntimeError("applied contribution stream drifted from raw spool")
        vector_catalog_sha256 = _digest(
            _STREAMING_APPLIED_DOMAIN,
            {
                "atlas_sha256": atlas.atlas_sha256,
                "mode_id": raw_spool.mode_id,
                "panel_binding_sha256": raw_spool.panel_binding_sha256,
                "role": raw_spool.role,
                "row_catalog_sha256": raw_spool.row_catalog_sha256,
                "source_raw_artifact_sha256": raw_spool.artifact_sha256,
                "source_tensor_catalog_sha256": raw_spool.tensor_catalog_sha256,
                "tensor_sha256s": semantic,
            },
        )
        metadata = {
            "atlas_sha256": atlas.atlas_sha256,
            "config": atlas.config.payload(),
            "kernel_receipt_sha256": expected_kernel,
            "mode_id": raw_spool.mode_id,
            "panel_binding_sha256": raw_spool.panel_binding_sha256,
            "resource_config": resources.payload(),
            "role": raw_spool.role,
            "row_catalog_sha256": raw_spool.row_catalog_sha256,
            "row_chunk_size": resources.row_chunk_size,
            "source_raw_artifact_sha256": raw_spool.artifact_sha256,
            "source_tensor_catalog_sha256": raw_spool.tensor_catalog_sha256,
            "vector_catalog_sha256": vector_catalog_sha256,
        }
        _publish_fields(
            destination_path,
            kind=STREAMING_APPLIED_CHART_STORE_KIND,
            metadata=metadata,
            row_keys=raw_spool.row_keys,
            fields=tuple(fields.values()),
            resources=resources,
        )
        return open_streaming_applied_chart_store(
            destination_path,
            max_artifact_bytes=resources.max_artifact_bytes,
        )
    finally:
        for field in fields.values():
            field.abort()


def open_streaming_applied_chart_store(
    source: Path | str,
    *,
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
) -> StreamingAppliedChartStore:
    store = _open_array_store(
        source,
        expected_kind=STREAMING_APPLIED_CHART_STORE_KIND,
        max_artifact_bytes=max_artifact_bytes,
    )
    metadata = store.metadata
    expected = {
        "atlas_sha256",
        "config",
        "kernel_receipt_sha256",
        "mode_id",
        "panel_binding_sha256",
        "resource_config",
        "role",
        "row_catalog_sha256",
        "row_chunk_size",
        "source_raw_artifact_sha256",
        "source_tensor_catalog_sha256",
        "vector_catalog_sha256",
    }
    if set(metadata) != expected or set(store.blocks) != _APPLIED_FIELD_NAMES:
        raise ValueError("applied chart store metadata/field contract drifted")
    config = _config_from_payload(metadata["config"])
    if not isinstance(metadata["role"], str) or not metadata["role"]:
        raise ValueError("applied chart store role is invalid")
    if not isinstance(metadata["mode_id"], str) or not metadata["mode_id"]:
        raise ValueError("applied chart store mode_id is invalid")
    for label in (
        "atlas_sha256",
        "kernel_receipt_sha256",
        "panel_binding_sha256",
        "row_catalog_sha256",
        "source_raw_artifact_sha256",
        "source_tensor_catalog_sha256",
        "vector_catalog_sha256",
    ):
        _require_sha256(metadata[label], label=label)
    chunk = _require_positive_int(metadata["row_chunk_size"], label="row_chunk_size")
    resource_config = _resource_config_from_payload(metadata["resource_config"])
    if (
        resource_config.row_chunk_size != chunk
        or store.size > resource_config.max_artifact_bytes
        or metadata["kernel_receipt_sha256"]
        != streaming_chart_kernel_receipt(resource_config)["receipt_sha256"]
    ):
        raise ValueError("applied chart store resource/kernel receipt drifted")
    _validate_fixed_block_chunks(store, row_chunk_size=chunk)
    if metadata["row_catalog_sha256"] != row_catalog_sha256(
        store.row_keys, role=metadata["role"]
    ):
        raise ValueError("applied chart store row catalog digest drifted")
    shapes = {
        "features": (len(store.row_keys), config.source_feature_width),
        "generalized_contributions": (
            len(store.row_keys),
            config.hidden_width,
        ),
        "memberships": (len(store.row_keys), config.chart_count),
        "weighted_activation_energy": (len(store.row_keys),),
        "weighted_chart_orthogonal_energy": (len(store.row_keys),),
        "weighted_chart_tangent_energy": (len(store.row_keys),),
        "weighted_fisher_energy": (len(store.row_keys),),
    }
    for name, expected_shape in shapes.items():
        block = store.blocks[name]
        expected_dtype = (
            "float32_le"
            if name == "generalized_contributions"
            else "float64_le"
        )
        if tuple(block["shape"]) != expected_shape or block["dtype"] != expected_dtype:
            raise ValueError("applied chart store field shape/dtype drifted")
    semantic = {
        name: str(block["semantic_float64_tensor_sha256"])
        for name, block in store.blocks.items()
    }
    expected_vector_catalog = _digest(
        _STREAMING_APPLIED_DOMAIN,
        {
            "atlas_sha256": metadata["atlas_sha256"],
            "mode_id": metadata["mode_id"],
            "panel_binding_sha256": metadata["panel_binding_sha256"],
            "role": metadata["role"],
            "row_catalog_sha256": metadata["row_catalog_sha256"],
            "source_raw_artifact_sha256": metadata[
                "source_raw_artifact_sha256"
            ],
            "source_tensor_catalog_sha256": metadata[
                "source_tensor_catalog_sha256"
            ],
            "tensor_sha256s": semantic,
        },
    )
    if expected_vector_catalog != metadata["vector_catalog_sha256"]:
        raise ValueError("applied chart store vector catalog digest drifted")
    result = StreamingAppliedChartStore(
        _store=store,
        config=config,
        role=metadata["role"],
        panel_binding_sha256=metadata["panel_binding_sha256"],
        mode_id=metadata["mode_id"],
        row_catalog_sha256=metadata["row_catalog_sha256"],
        atlas_sha256=metadata["atlas_sha256"],
        source_raw_artifact_sha256=metadata["source_raw_artifact_sha256"],
        source_tensor_catalog_sha256=metadata["source_tensor_catalog_sha256"],
        vector_catalog_sha256=metadata["vector_catalog_sha256"],
        row_chunk_size=chunk,
        kernel_receipt_sha256=metadata["kernel_receipt_sha256"],
    )
    energy_fields = (
        "weighted_activation_energy",
        "weighted_chart_orthogonal_energy",
        "weighted_chart_tangent_energy",
        "weighted_fisher_energy",
    )
    for start in range(0, result.row_count, chunk):
        end = min(result.row_count, start + chunk)
        memberships = result.read_slice("memberships", start, end)
        features = result.read_slice("features", start, end)
        energies = {
            name: result.read_slice(name, start, end)
            for name in energy_fields
        }
        if not torch.allclose(
            memberships.sum(dim=1),
            torch.ones(end - start, dtype=torch.float64),
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise ValueError("applied chart memberships do not form a simplex")
        membership_columns = features[
            :, :: config.tangent_rank + 1
        ]
        if not torch.equal(membership_columns, memberships):
            raise ValueError("applied chart feature/membership binding drifted")
        if any(bool((value < 0.0).any()) for value in energies.values()):
            raise ValueError("applied chart energies must be nonnegative")
        if not torch.allclose(
            energies["weighted_chart_tangent_energy"]
            + energies["weighted_chart_orthogonal_energy"],
            energies["weighted_fisher_energy"],
            atol=1.0e-9,
            rtol=1.0e-9,
        ):
            raise ValueError("applied chart Fisher energy closure drifted")
    return result


def _validate_applied_pair(
    atlas: SharedChartAtlas,
    source: StreamingAppliedChartStore,
    target: StreamingAppliedChartStore,
    *,
    resources: StreamingChartResourceConfig,
    fit: bool,
) -> None:
    if source.mode_id == target.mode_id:
        raise ValueError("transport endpoints must differ")
    expected_kernel = streaming_chart_kernel_receipt(resources)["receipt_sha256"]
    if (
        source.config != atlas.config
        or target.config != atlas.config
        or source.atlas_sha256 != atlas.atlas_sha256
        or target.atlas_sha256 != atlas.atlas_sha256
    ):
        raise ValueError("streaming transport endpoints use the wrong atlas/config")
    if (
        source.role != target.role
        or source.panel_binding_sha256 != target.panel_binding_sha256
        or source.row_catalog_sha256 != target.row_catalog_sha256
        or source.row_keys != target.row_keys
    ):
        raise ValueError("streaming transport endpoint rows are not aligned")
    if (
        source.row_chunk_size != resources.row_chunk_size
        or target.row_chunk_size != resources.row_chunk_size
        or source.kernel_receipt_sha256 != expected_kernel
        or target.kernel_receipt_sha256 != expected_kernel
    ):
        raise ValueError("streaming transport endpoints use a different chunk kernel")
    if fit:
        if source.role != FIT_ROLE:
            raise ValueError("streaming transports may fit only on role A")
        if (
            source.panel_binding_sha256 != atlas.fit_panel_binding_sha256
            or source.row_catalog_sha256 != atlas.fit_row_catalog_sha256
            or source.mode_id not in atlas.fit_mode_ids
            or target.mode_id not in atlas.fit_mode_ids
        ):
            raise ValueError("streaming fit rows are not the exact atlas fit catalog")
    else:
        if source.role == FIT_ROLE:
            raise ValueError("streaming transport evaluation must be disjoint from A")
        if source.panel_binding_sha256 == atlas.fit_panel_binding_sha256:
            raise ValueError("streaming evaluation panel must differ from fit panel")


def _read_full_vector(
    store: StreamingAppliedChartStore,
    field: str,
    *,
    chunk_size: int,
) -> Tensor:
    result = torch.empty(store.row_count, dtype=torch.float64)
    for start in range(0, store.row_count, chunk_size):
        end = min(store.row_count, start + chunk_size)
        value = store.read_slice(field, start, end)
        if value.ndim != 1:
            raise ValueError(f"{field} is not a row vector")
        result[start:end] = value
    return result


def fit_directed_chart_transport_streaming(
    atlas: SharedChartAtlas,
    source: StreamingAppliedChartStore,
    target: StreamingAppliedChartStore,
    *,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> DirectedChartTransport:
    """Fit one directed WLS map from compact stores using normal equations."""

    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")
    _validate_applied_pair(
        atlas, source, target, resources=resources, fit=True
    )
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=source.row_count,
        mode_count=len(atlas.fit_mode_ids),
        config=atlas.config,
        resources=resources,
    )
    _enforce_resident_budget(
        resources,
        required_bytes=estimate.transport_fit_peak_resident_bytes,
        operation="streaming directed transport fit",
    )
    balanced = _balanced_row_weights(
        source.row_keys, mode_count=len(atlas.fit_mode_ids)
    )
    fisher = _read_full_vector(
        target,
        "weighted_fisher_energy",
        chunk_size=resources.row_chunk_size,
    )
    weights = balanced * torch.clamp(
        fisher, min=atlas.config.transport_fisher_energy_floor
    )
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("streaming target-Fisher row weights degenerated")
    weights = weights / total
    feature_width = atlas.config.source_feature_width
    hidden_width = atlas.config.hidden_width
    gram = torch.zeros(
        (feature_width, feature_width), dtype=torch.float64
    )
    right = torch.zeros((feature_width, hidden_width), dtype=torch.float64)
    for start in range(0, source.row_count, resources.row_chunk_size):
        end = min(source.row_count, start + resources.row_chunk_size)
        design = source.read_slice("features", start, end)
        target_values = target.read_slice(
            "generalized_contributions", start, end
        )
        chunk_weights = weights[start:end]
        gram += design.T @ (chunk_weights[:, None] * design)
        right += design.T @ (chunk_weights[:, None] * target_values)
    scale = float(torch.trace(gram) / gram.shape[0])
    absolute_ridge = atlas.config.transport_relative_ridge * max(
        scale, atlas.config.ridge_scale_floor
    )
    coefficient = torch.linalg.solve(
        gram
        + absolute_ridge
        * torch.eye(feature_width, dtype=torch.float64),
        right,
    )
    fit_catalog = _reference._digest(
        _reference._FIT_CATALOG_DOMAIN,
        {
            "atlas_sha256": atlas.atlas_sha256,
            "fit_panel_binding_sha256": source.panel_binding_sha256,
            "fit_row_catalog_sha256": source.row_catalog_sha256,
        },
    )
    split = _reference._digest(
        _reference._SPLIT_DOMAIN,
        {"fit_catalog_sha256": fit_catalog, "fit_role": FIT_ROLE},
    )
    values = {
        "config": atlas.config,
        "source_mode_id": source.mode_id,
        "target_mode_id": target.mode_id,
        "atlas_sha256": atlas.atlas_sha256,
        "fit_role": FIT_ROLE,
        "fit_panel_binding_sha256": source.panel_binding_sha256,
        "fit_row_catalog_sha256": source.row_catalog_sha256,
        "fit_catalog_sha256": fit_catalog,
        "split_receipt_sha256": split,
        "fit_design_sha256": source.field_semantic_sha256s["features"],
        "fit_target_sha256": target.field_semantic_sha256s[
            "generalized_contributions"
        ],
        "fit_row_weights_sha256": _reference._tensor_sha256(weights),
        "solver_id": "balanced_float64_primal_ridge_solve_v1",
        "absolute_ridge": absolute_ridge,
        "coefficient": coefficient,
    }
    provisional = object.__new__(DirectedChartTransport)
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    object.__setattr__(provisional, "transport_sha256", "0" * 64)
    return DirectedChartTransport(
        **values,
        transport_sha256=_reference._transport_sha256(provisional),
    )


@dataclass(frozen=True, slots=True)
class StreamingTransportArtifactReceipt:
    path: Path
    source_mode_id: str
    target_mode_id: str
    transport_sha256: str
    byte_count: int


def fit_and_save_directed_chart_transport_streaming(
    atlas: SharedChartAtlas,
    source: StreamingAppliedChartStore,
    target: StreamingAppliedChartStore,
    destination: Path | str,
    *,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> StreamingTransportArtifactReceipt:
    """Fit and immediately persist one coefficient; do not retain a catalog."""

    destination_path = Path(destination)
    transport = fit_directed_chart_transport_streaming(
        atlas, source, target, resources=resources
    )
    save_directed_chart_transport(transport, destination_path)
    state = destination_path.lstat()
    if not stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise RuntimeError("saved transport artifact is not a regular file")
    return StreamingTransportArtifactReceipt(
        path=destination_path,
        source_mode_id=source.mode_id,
        target_mode_id=target.mode_id,
        transport_sha256=transport.transport_sha256,
        byte_count=state.st_size,
    )


def fit_all_ordered_directed_chart_transports_streaming(
    atlas: SharedChartAtlas,
    stores: Sequence[StreamingAppliedChartStore],
    destination_directory: Path | str,
    *,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
    progress: Callable[[int, int, str, str], None] | None = None,
) -> tuple[StreamingTransportArtifactReceipt, ...]:
    """Persist all ordered maps one at a time; at most one coefficient is live."""

    ordered = tuple(sorted(stores, key=lambda item: item.mode_id))
    if (
        len(ordered) != len(atlas.fit_mode_ids)
        or tuple(item.mode_id for item in ordered) != atlas.fit_mode_ids
    ):
        raise ValueError("ordered fit stores do not equal the atlas mode catalog")
    directory = Path(destination_directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = directory.lstat()
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise ValueError("transport destination must be a real directory")
    total = len(ordered) * (len(ordered) - 1)
    receipts: list[StreamingTransportArtifactReceipt] = []
    ordinal = 0
    for target in ordered:
        for source in ordered:
            if source.mode_id == target.mode_id:
                continue
            digest = hashlib.sha256(
                (source.mode_id + "\0" + target.mode_id).encode("utf-8")
            ).hexdigest()[:20]
            destination = directory / f"{ordinal:04d}-{digest}.fgchart"
            receipt = fit_and_save_directed_chart_transport_streaming(
                atlas,
                source,
                target,
                destination,
                resources=resources,
            )
            receipts.append(receipt)
            ordinal += 1
            if progress is not None:
                progress(ordinal, total, source.mode_id, target.mode_id)
    return tuple(receipts)


def evaluate_directed_chart_transport_streaming(
    atlas: SharedChartAtlas,
    transport: DirectedChartTransport,
    source: StreamingAppliedChartStore,
    target: StreamingAppliedChartStore,
    *,
    row_shuffle: RowShuffleCatalog,
    resources: StreamingChartResourceConfig = StreamingChartResourceConfig(),
) -> DirectedTransportEvaluation:
    """Evaluate real/zero/shuffle controls with bounded prediction chunks."""

    if not isinstance(resources, StreamingChartResourceConfig):
        raise TypeError("resources must be StreamingChartResourceConfig")
    _validate_applied_pair(
        atlas, source, target, resources=resources, fit=False
    )
    if (
        transport.source_mode_id != source.mode_id
        or transport.target_mode_id != target.mode_id
    ):
        raise ValueError("streaming evaluation endpoints do not match transport")
    if (
        transport.atlas_sha256 != atlas.atlas_sha256
        or transport.config != atlas.config
    ):
        raise ValueError("streaming evaluation transport uses the wrong atlas")
    if row_shuffle.role != source.role or row_shuffle.row_catalog_sha256 != source.row_catalog_sha256:
        raise ValueError("row-shuffle catalog does not match streaming rows")
    if row_shuffle.seed != ROW_SHUFFLE_SEED:
        raise ValueError("row-shuffle seed does not match the frozen protocol")
    expected_shuffle = build_within_example_row_shuffle(
        source.row_keys, role=source.role, seed=row_shuffle.seed
    )
    if (
        expected_shuffle.catalog_sha256 != row_shuffle.catalog_sha256
        or not torch.equal(expected_shuffle.permutation, row_shuffle.permutation)
    ):
        raise ValueError(
            "row-shuffle is not the deterministic within-example control"
        )
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=source.row_count,
        mode_count=len(atlas.fit_mode_ids),
        config=atlas.config,
        resources=resources,
    )
    _enforce_resident_budget(
        resources,
        required_bytes=estimate.transport_eval_peak_resident_bytes,
        operation="streaming directed transport evaluation",
    )
    buckets: dict[tuple[str, str], list[list[float]]] = {}
    permutation = row_shuffle.permutation
    for start in range(0, source.row_count, resources.row_chunk_size):
        end = min(source.row_count, start + resources.row_chunk_size)
        source_features = source.read_slice("features", start, end)
        source_memberships = source.read_slice("memberships", start, end)
        shuffled_indices = permutation[start:end]
        shuffled_features = source.read_rows("features", shuffled_indices)
        shuffled_memberships = source.read_rows(
            "memberships", shuffled_indices
        )
        target_values = target.read_slice(
            "generalized_contributions", start, end
        )
        target_memberships = target.read_slice("memberships", start, end)
        predicted = torch.mm(source_features, transport.coefficient)
        shuffled_predicted = torch.mm(
            shuffled_features, transport.coefficient
        )
        if not bool(torch.isfinite(predicted).all()) or not bool(
            torch.isfinite(shuffled_predicted).all()
        ):
            raise ValueError("streaming transport prediction became non-finite")
        p_energy, t_energy, cross, residual = _reference._pair_energy_rows(
            atlas,
            predicted,
            source_memberships,
            target_values,
            target_memberships,
        )
        _, _, _, shuffled_residual = _reference._pair_energy_rows(
            atlas,
            shuffled_predicted,
            shuffled_memberships,
            target_values,
            target_memberships,
        )
        stored_target_energy = target.read_slice(
            "weighted_fisher_energy", start, end
        )
        if not torch.allclose(
            t_energy,
            stored_target_energy,
            atol=1.0e-9,
            rtol=1.0e-9,
        ):
            raise ValueError("streaming target Fisher energy receipt drifted")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                p_energy,
                t_energy,
                cross,
                residual,
                shuffled_residual,
            )
        ):
            raise ValueError("streaming transport energy became non-finite")
        for local, row_key in enumerate(source.row_keys[start:end]):
            bucket = buckets.setdefault(
                (row_key.family_id, row_key.example_id),
                [[], [], [], [], [], []],
            )
            bucket[0].append(float(residual[local]))
            bucket[1].append(float(p_energy[local]))
            bucket[2].append(float(t_energy[local]))
            bucket[3].append(float(cross[local]))
            bucket[4].append(float(t_energy[local]))
            bucket[5].append(float(shuffled_residual[local]))
    examples: list[DirectedExampleEnergy] = []
    for family_id, example_id in sorted(buckets):
        values = buckets[(family_id, example_id)]
        residual_energy = math.fsum(values[0])
        zero_energy = math.fsum(values[4])
        shuffle_energy = math.fsum(values[5])
        examples.append(
            DirectedExampleEnergy(
                family_id=family_id,
                example_id=example_id,
                row_count=len(values[0]),
                weighted_transport_residual_energy=residual_energy,
                weighted_predicted_fisher_energy=math.fsum(values[1]),
                weighted_target_fisher_energy=math.fsum(values[2]),
                weighted_predicted_target_cross_inner_product=math.fsum(
                    values[3]
                ),
                weighted_zero_prediction_residual_energy=zero_energy,
                weighted_row_shuffled_transport_residual_energy=shuffle_energy,
                real_gain_over_zero=(
                    1.0 - residual_energy / zero_energy
                    if zero_energy > 0.0
                    else None
                ),
                real_gain_over_row_shuffle=(
                    1.0 - residual_energy / shuffle_energy
                    if shuffle_energy > 0.0
                    else None
                ),
            )
        )
    kernel = streaming_chart_kernel_receipt(resources)
    return DirectedTransportEvaluation(
        comparison_abi_name=COMPARISON_ABI_NAME,
        source_mode_id=source.mode_id,
        target_mode_id=target.mode_id,
        evaluation_role=source.role,
        evaluation_panel_binding_sha256=source.panel_binding_sha256,
        evaluation_row_catalog_sha256=source.row_catalog_sha256,
        row_shuffle_catalog_sha256=row_shuffle.catalog_sha256,
        split_receipt_sha256=_reference._digest(
            _reference._SPLIT_DOMAIN,
            {
                "evaluation_panel_binding_sha256": source.panel_binding_sha256,
                "evaluation_role": source.role,
                "evaluation_row_catalog_sha256": source.row_catalog_sha256,
                "fit_catalog_sha256": transport.fit_catalog_sha256,
                "fit_role": FIT_ROLE,
            },
        ),
        transport_sha256=transport.transport_sha256,
        prediction_backend_id=STREAMING_PREDICTION_BACKEND_ID,
        prediction_row_chunk_size=resources.row_chunk_size,
        prediction_backend_receipt_sha256=kernel["receipt_sha256"],
        examples=tuple(examples),
    )


__all__ = [
    "RAW_MODE_ROW_SPOOL_KIND",
    "STREAMING_APPLIED_CHART_STORE_KIND",
    "STREAMING_CHART_KERNEL_SCHEMA",
    "STREAMING_PREDICTION_BACKEND_ID",
    "STREAMING_STORE_SCHEMA",
    "RawModeRowSpool",
    "RawModeRowSpoolWriter",
    "StreamingAppliedChartStore",
    "StreamingChartResourceConfig",
    "StreamingChartResourceEstimate",
    "StreamingRowChunk",
    "StreamingTransportArtifactReceipt",
    "apply_shared_chart_atlas_streaming",
    "estimate_streaming_chart_resources",
    "evaluate_directed_chart_transport_streaming",
    "fit_all_ordered_directed_chart_transports_streaming",
    "fit_and_save_directed_chart_transport_streaming",
    "fit_directed_chart_transport_streaming",
    "fit_shared_chart_atlas_streaming",
    "open_raw_mode_row_spool",
    "open_streaming_applied_chart_store",
    "streaming_chart_kernel_receipt",
    "write_raw_mode_row_spool",
]
