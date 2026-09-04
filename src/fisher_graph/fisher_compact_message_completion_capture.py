"""Deterministic safetensors publication for compact-message captures.

The K-series protocol verifies row identity from a physical
``capture_row_id_raw32`` tensor inside every finalized shard.  This module is
the corresponding producer.  It intentionally has no model or corpus access:
callers supply already-captured NumPy arrays and the committed row IDs.

Publication is fail-closed and no-clobber.  Bytes are written and synced to a
private temporary file in the destination directory, then installed with a
hard link.  A pre-existing destination is never replaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import secrets
import stat
import struct
from typing import Final

import numpy as np

from .fisher_compact_message_completion_protocol import (
    ProtocolSchemaError,
    ShardPlanEntry,
)


__all__ = [
    "CAPTURE_ROW_ID_TENSOR",
    "SafetensorsPublicationError",
    "decode_safetensors",
    "encode_safetensors",
    "publish_bytes_exclusive",
    "write_capture_shard",
]


CAPTURE_ROW_ID_TENSOR: Final = "capture_row_id_raw32"
_HEADER_LENGTH = struct.Struct("<Q")
_SHA256_HEX = frozenset("0123456789abcdef")
_NUMPY_TO_SAFETENSORS: Final[dict[str, str]] = {
    "?": "BOOL",
    "u1": "U8",
    "i1": "I8",
    "u2": "U16",
    "i2": "I16",
    "f2": "F16",
    "u4": "U32",
    "i4": "I32",
    "f4": "F32",
    "u8": "U64",
    "i8": "I64",
    "f8": "F64",
}
_SAFETENSORS_TO_NUMPY: Final[dict[str, str]] = {
    value: key for key, value in _NUMPY_TO_SAFETENSORS.items()
}


class SafetensorsPublicationError(RuntimeError):
    """A shard cannot be encoded or published without weakening integrity."""


def _normalize_array(name: str, value: np.ndarray) -> tuple[np.ndarray, str]:
    if not isinstance(name, str) or not name or name == "__metadata__":
        raise ProtocolSchemaError("tensor names must be nonempty and nonreserved")
    if not isinstance(value, np.ndarray):
        raise ProtocolSchemaError(f"tensor {name!r} must be a NumPy array")
    if value.ndim == 0:
        raise ProtocolSchemaError(f"tensor {name!r} must have a leading row axis")
    if any(int(dimension) < 0 for dimension in value.shape):
        raise ProtocolSchemaError(f"tensor {name!r} has an invalid shape")

    dtype = value.dtype
    kind_width = dtype.kind + str(dtype.itemsize)
    if dtype.kind == "b":
        kind_width = "?"
    safetensors_dtype = _NUMPY_TO_SAFETENSORS.get(kind_width)
    if safetensors_dtype is None:
        raise ProtocolSchemaError(
            f"tensor {name!r} has unsupported dtype {dtype.str!r}"
        )

    # Safetensors numeric payloads are little-endian.  Byte-order-neutral
    # one-byte dtypes need no conversion.
    if dtype.itemsize > 1 and dtype.byteorder not in ("<", "="):
        value = value.astype(dtype.newbyteorder("<"), copy=False)
    elif dtype.itemsize > 1 and dtype.byteorder == "=" and not np.little_endian:
        value = value.astype(dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(value), safetensors_dtype


def _row_id_array(row_ids: Sequence[str]) -> np.ndarray:
    if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes)):
        raise ProtocolSchemaError("row_ids must be a sequence of lowercase hashes")
    if not row_ids:
        raise ProtocolSchemaError("a capture shard must contain at least one row")
    raw: list[bytes] = []
    for index, row_id in enumerate(row_ids):
        if (
            not isinstance(row_id, str)
            or len(row_id) != 64
            or any(character not in _SHA256_HEX for character in row_id)
        ):
            raise ProtocolSchemaError(
                f"row_ids[{index}] must be a lowercase SHA-256"
            )
        raw.append(bytes.fromhex(row_id))
    if len(set(raw)) != len(raw):
        raise ProtocolSchemaError("capture row IDs must be unique")
    return np.frombuffer(b"".join(raw), dtype=np.uint8).reshape(len(raw), 32)


def encode_safetensors(
    tensors: Mapping[str, np.ndarray],
    *,
    row_ids: Sequence[str] | None = None,
) -> bytes:
    """Encode deterministic metadata-free safetensors bytes.

    When ``row_ids`` is provided, the mandatory physical row-identity tensor
    is inserted and every supplied tensor must have the same leading extent.
    Mapping iteration order is ignored; tensor names determine byte layout.
    """

    if not isinstance(tensors, Mapping) or not tensors:
        raise ProtocolSchemaError("tensors must be a nonempty mapping")
    if CAPTURE_ROW_ID_TENSOR in tensors:
        raise ProtocolSchemaError(
            "capture_row_id_raw32 is derived from committed row IDs, not caller data"
        )

    materialized: dict[str, np.ndarray] = dict(tensors)
    expected_rows: int | None = None
    if row_ids is not None:
        row_tensor = _row_id_array(row_ids)
        expected_rows = int(row_tensor.shape[0])
        materialized[CAPTURE_ROW_ID_TENSOR] = row_tensor

    normalized: list[tuple[str, np.ndarray, str]] = []
    for name in sorted(materialized):
        array, dtype_name = _normalize_array(name, materialized[name])
        if expected_rows is not None and int(array.shape[0]) != expected_rows:
            raise ProtocolSchemaError(
                f"tensor {name!r} leading dimension differs from row count"
            )
        normalized.append((name, array, dtype_name))

    descriptors: dict[str, object] = {}
    chunks: list[bytes] = []
    cursor = 0
    for name, array, dtype_name in normalized:
        content = array.tobytes(order="C")
        end = cursor + len(content)
        descriptors[name] = {
            "data_offsets": [cursor, end],
            "dtype": dtype_name,
            "shape": [int(dimension) for dimension in array.shape],
        }
        chunks.append(content)
        cursor = end

    header = json.dumps(
        descriptors,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    header += b" " * (-len(header) % 8)
    return _HEADER_LENGTH.pack(len(header)) + header + b"".join(chunks)


def decode_safetensors(content: bytes) -> dict[str, np.ndarray]:
    """Decode the strict metadata-free subset emitted by this module."""

    if not isinstance(content, bytes) or len(content) < _HEADER_LENGTH.size:
        raise ProtocolSchemaError("safetensors content is truncated")
    (header_length,) = _HEADER_LENGTH.unpack(content[: _HEADER_LENGTH.size])
    header_end = _HEADER_LENGTH.size + header_length
    if header_length == 0 or header_end > len(content):
        raise ProtocolSchemaError("safetensors header length is invalid")
    try:
        header = json.loads(content[_HEADER_LENGTH.size : header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolSchemaError("safetensors header is not valid JSON") from error
    if not isinstance(header, dict) or not header or "__metadata__" in header:
        raise ProtocolSchemaError("safetensors header must contain only tensors")
    payload = memoryview(content)[header_end:]
    occupied: list[tuple[int, int]] = []
    result: dict[str, np.ndarray] = {}
    for name in sorted(header):
        descriptor = header[name]
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise ProtocolSchemaError("safetensors tensor descriptor is invalid")
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        kind_width = _SAFETENSORS_TO_NUMPY.get(dtype_name)
        if (
            kind_width is None
            or not isinstance(shape, list)
            or not shape
            or any(type(value) is not int or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in offsets)
        ):
            raise ProtocolSchemaError(f"safetensors tensor {name!r} is malformed")
        start, end = offsets
        if start > end or end > len(payload):
            raise ProtocolSchemaError(f"safetensors tensor {name!r} has invalid offsets")
        if any(start < other_end and other_start < end for other_start, other_end in occupied):
            raise ProtocolSchemaError("safetensors tensor payloads overlap")
        dtype = np.dtype(np.bool_) if kind_width == "?" else np.dtype("<" + kind_width)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if end - start != expected:
            raise ProtocolSchemaError(f"safetensors tensor {name!r} byte size is invalid")
        occupied.append((start, end))
        result[name] = np.frombuffer(payload[start:end], dtype=dtype).reshape(shape).copy()
    cursor = 0
    for start, end in sorted(occupied):
        if start != cursor:
            raise ProtocolSchemaError("safetensors payload contains an unreferenced gap")
        cursor = end
    if cursor != len(payload):
        raise ProtocolSchemaError("safetensors payload contains unreferenced trailing bytes")
    return result


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    cursor = path
    while True:
        info = os.lstat(cursor)
        if stat.S_ISLNK(info.st_mode):
            raise SafetensorsPublicationError(
                f"publication path contains symlink component {cursor}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise SafetensorsPublicationError(
                f"publication path component is not a directory: {cursor}"
            )
        if cursor.parent == cursor:
            break
        cursor = cursor.parent


def publish_bytes_exclusive(path: str | os.PathLike[str], content: bytes) -> Path:
    """Atomically publish exact bytes without ever replacing a destination."""

    destination = Path(path)
    if not isinstance(content, bytes) or not content:
        raise ProtocolSchemaError("published safetensors content must be bytes")
    _ensure_private_directory(destination.parent)
    temporary = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        view = memoryview(content)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise SafetensorsPublicationError("short safetensors write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise SafetensorsPublicationError(
                f"refusing to overwrite existing artifact {destination}"
            ) from error
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def write_capture_shard(
    capture_root: str | os.PathLike[str],
    entry: ShardPlanEntry,
    tensors: Mapping[str, np.ndarray],
) -> Path:
    """Encode and exclusively publish one exact planned capture shard."""

    if not isinstance(entry, ShardPlanEntry):
        raise ProtocolSchemaError("entry must be ShardPlanEntry")
    content = encode_safetensors(tensors, row_ids=entry.row_ids)
    return publish_bytes_exclusive(Path(capture_root) / entry.path, content)
