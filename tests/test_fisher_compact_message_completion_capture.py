from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from fisher_graph.fisher_compact_message_completion_capture import (
    SafetensorsPublicationError,
    decode_safetensors,
    encode_safetensors,
    publish_bytes_exclusive,
    write_capture_shard,
)
from fisher_graph.fisher_compact_message_completion_protocol import (
    EvaluationRow,
    ProtocolSchemaError,
    build_row_index_payload,
    build_shard_plan,
    verify_capture_shard_files,
)


def _plan():
    rows = tuple(
        EvaluationRow(
            analysis_unit_commitment=hashlib.sha256(
                f"unit-{index}".encode()
            ).hexdigest(),
            canonical_sequence_hash=hashlib.sha256(
                f"sequence-{index}".encode()
            ).hexdigest(),
            source_token_position=index,
            target_token_position=index + 1,
        )
        for index in range(3)
    )
    payload = build_row_index_payload("fit", "12" * 32, rows)
    return build_shard_plan(payload, 2)


def test_deterministic_encoder_embeds_physical_row_ids() -> None:
    plan = _plan()
    entry = plan.entries[0]
    tensors = {
        "h": np.arange(entry.row_count * 4, dtype=np.float32).reshape(
            entry.row_count, 4
        ),
        "head_indexed": np.ones((entry.row_count, 4, 3), dtype=np.float32),
    }
    first = encode_safetensors(tensors, row_ids=entry.row_ids)
    second = encode_safetensors(dict(reversed(tuple(tensors.items()))), row_ids=entry.row_ids)
    assert first == second

    from fisher_graph.fisher_compact_message_completion_protocol import (
        PhysicalShardEvidence,
        verify_physical_shard,
    )

    record = verify_physical_shard(
        entry,
        PhysicalShardEvidence(
            first,
            {
                "capture_row_id_raw32": entry.row_count,
                "h": entry.row_count,
                "head_indexed": entry.row_count,
            },
        ),
        shard_plan_sha256=plan.plan_sha256,
    )
    assert record.row_ids == entry.row_ids


def test_encoder_rejects_detached_or_misaligned_identity() -> None:
    entry = _plan().entries[0]
    with pytest.raises(ProtocolSchemaError, match="derived from committed"):
        encode_safetensors(
            {"capture_row_id_raw32": np.zeros((2, 32), dtype=np.uint8)},
            row_ids=entry.row_ids,
        )
    with pytest.raises(ProtocolSchemaError, match="leading dimension"):
        encode_safetensors(
            {"h": np.zeros((entry.row_count + 1, 4), dtype=np.float32)},
            row_ids=entry.row_ids,
        )
    with pytest.raises(ProtocolSchemaError, match="lowercase SHA-256"):
        encode_safetensors(
            {"h": np.zeros((entry.row_count, 4), dtype=np.float32)},
            row_ids=("AA" * 32,) * entry.row_count,
        )


def test_metadata_free_noncapture_bundle_allows_heterogeneous_tensor_shapes() -> None:
    first = encode_safetensors(
        {
            "weight": np.zeros((3, 5), dtype=np.float32),
            "bias": np.zeros((5,), dtype=np.float32),
            "catalog": np.arange(2, dtype=np.int64),
        }
    )
    second = encode_safetensors(
        {
            "catalog": np.arange(2, dtype=np.int64),
            "bias": np.zeros((5,), dtype=np.float32),
            "weight": np.zeros((3, 5), dtype=np.float32),
        }
    )
    assert first == second


def test_decoder_exactly_round_trips_heterogeneous_tensors() -> None:
    tensors = {
        "bools": np.array([[True, False], [False, True]], dtype=np.bool_),
        "floats": np.array([[1.25, -2.5, 0.0]], dtype=np.float32),
        "signed": np.array([[-7, 0], [12, 99]], dtype=np.int32),
        "unsigned": np.array([0, 1, 65535], dtype=np.uint16),
    }

    encoded = encode_safetensors(tensors)
    decoded = decode_safetensors(encoded)

    assert tuple(decoded) == tuple(sorted(tensors))
    for name, expected in tensors.items():
        actual = decoded[name]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected)


def _replace_safetensors_header(
    content: bytes,
    mutate: Callable[[dict[str, dict[str, object]]], None],
) -> bytes:
    header_length = struct.unpack("<Q", content[:8])[0]
    payload = content[8 + header_length :]
    header = json.loads(content[8 : 8 + header_length].decode("utf-8"))
    mutate(header)
    replacement = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    replacement += b" " * (-len(replacement) % 8)
    return struct.pack("<Q", len(replacement)) + replacement + payload


def test_decoder_rejects_malformed_trailing_and_overlapping_data() -> None:
    encoded = encode_safetensors(
        {
            "first": np.arange(4, dtype=np.int32),
            "second": np.arange(3, dtype=np.float32),
        }
    )

    with pytest.raises(ProtocolSchemaError, match="truncated"):
        decode_safetensors(b"short")
    with pytest.raises(ProtocolSchemaError, match="trailing bytes"):
        decode_safetensors(encoded + b"unreferenced")

    malformed = _replace_safetensors_header(
        encoded,
        lambda header: header["first"].__setitem__("shape", [5]),
    )
    with pytest.raises(ProtocolSchemaError, match="byte size is invalid"):
        decode_safetensors(malformed)

    overlapping = _replace_safetensors_header(
        encoded,
        lambda header: header["second"].__setitem__("data_offsets", [8, 20]),
    )
    with pytest.raises(ProtocolSchemaError, match="payloads overlap"):
        decode_safetensors(overlapping)


def test_capture_shards_publish_no_clobber_and_reopen(tmp_path: Path) -> None:
    plan = _plan()
    for entry in plan.entries:
        write_capture_shard(
            tmp_path,
            entry,
            {"h": np.zeros((entry.row_count, 8), dtype=np.float32)},
        )
    verification = verify_capture_shard_files(plan, tmp_path)
    assert verification.row_count == plan.row_count
    assert verification.captured_ordered_row_sequence_sha256 == (
        plan.ordered_row_sequence_sha256
    )

    first = plan.entries[0]
    with pytest.raises(SafetensorsPublicationError, match="overwrite"):
        write_capture_shard(
            tmp_path,
            first,
            {"h": np.ones((first.row_count, 8), dtype=np.float32)},
        )


def test_exclusive_publisher_never_replaces_existing_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "bundle.safetensors"
    publish_bytes_exclusive(destination, b"first")
    assert destination.read_bytes() == b"first"
    assert destination.stat().st_mode & 0o777 == 0o600
    with pytest.raises(SafetensorsPublicationError, match="overwrite"):
        publish_bytes_exclusive(destination, b"second")
    assert destination.read_bytes() == b"first"
