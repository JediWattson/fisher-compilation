from __future__ import annotations

import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
import runpy
import sqlite3
import struct
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "src"
    / "fisher_graph"
    / "fisher_compact_message_completion_protocol.py"
)

# Load the standard-library-only kernel without executing fisher_graph.__init__.
protocol = SimpleNamespace(
    **runpy.run_path(str(SOURCE), run_name="_future_d2_protocol_under_test")
)

CAMPAIGN_ID = "future-d2-synthetic-campaign"
ROW_PROTOCOL_HASH = "33" * 32
CAPTURE_SCHEMA_HASH = "44" * 32
CAPTURE_IMPLEMENTATION_HASH = "55" * 32

ROW_INPUTS = (
    ("00" * 32, "11" * 32, 0, 1),
    ("ff" * 32, "22" * 32, 257, 258),
    (
        "0123456789abcdef" * 4,
        "abcdef0123456789" * 4,
        9,
        10,
    ),
)
ROW_IDS_IN_INPUT_ORDER = (
    "5695b25763f3056cbf9c92d5389d44e1de1d3ca03e75ebacd293ee87ca5268da",
    "59ba20c40eb17c4e45863816a1a2b21c10e18b7db9a735a0c3aca07b2a965558",
    "55c9a956a7cbd600f0dad76ea9158d2321c939137194e108ec2085ba4cd04745",
)
CANONICAL_ROW_IDS = (
    "55c9a956a7cbd600f0dad76ea9158d2321c939137194e108ec2085ba4cd04745",
    "5695b25763f3056cbf9c92d5389d44e1de1d3ca03e75ebacd293ee87ca5268da",
    "59ba20c40eb17c4e45863816a1a2b21c10e18b7db9a735a0c3aca07b2a965558",
)
ROW_SET_SHA256 = (
    "1231f82eaa524a0ad21ba6d3341a67f477ff791f85ae136974b01fbaffbf4e4d"
)
ORDERED_ROW_SEQUENCE_SHA256 = (
    "1081544c222c0caf354ef0542a73548ff2faf9dc17c352616c18b77d2bd01817"
)
REVERSED_ROW_SEQUENCE_SHA256 = (
    "c3e531338bd8ee226b608fab59a4083a54774d41116dbbc0305dd0b64bb95e36"
)
ROW_INDEX_PAYLOAD_SHA256 = (
    "8650ade784237af3b33775b5ef0c787eadfa8d39fd64ff372024c42f738edefb"
)
SHARD_PLAN_SHA256_L2 = (
    "b01969a4b6bcd2b1ce39fccbf47b4415c5a3bcd31bc68c7fbdb78f4e124bdb8c"
)

COMPLETED_PREFIX_ARTIFACTS = {
    "P0": (
        "semantic_canonicalization_spec.json",
        "reserve_retired_exposed.json",
        "local_development_registry_status.json",
        "corpus_role_classification.json",
        "preflight_decision.json",
        "commands.md",
    ),
    "P1": (
        "independence_protocol.json",
        "split_manifest.json",
        "o1_calibration_assignment.json",
    ),
    "P2": (
        "model_identity.json",
        "fisher_carrier_identity.json",
        "runtime_identity.json",
        "execution_budget.json",
        "source_inventory.json",
        "row_derivation_protocol.json",
    ),
}
FIT_P3_AFTER_AUTH_OMISSIONS = {
    "row_index_manifest_fit.json": "not_reached_support_terminal",
    "fit_row_index_committed.json": "not_reached_support_terminal",
    "capture_shard_plan_fit.json": "not_reached_support_terminal",
    "fit_tensor_capture_started.json": "not_reached_support_terminal",
    "fit_capture_transition_consumed_success.json": (
        "not_reached_support_terminal"
    ),
    "capture_shard_manifest_fit.json": "not_reached_support_terminal",
    "capture_invariants.json": "not_reached_support_terminal",
}
FIT_P3_AFTER_CAPTURE_FAILURE_OMISSIONS = {
    "fit_capture_transition_consumed_success.json": "terminal_failure_branch",
    "capture_shard_manifest_fit.json": "terminal_failure_branch",
    "capture_invariants.json": "terminal_failure_branch",
}
FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS = {
    name: "not_reached_support_terminal"
    for name in protocol._PREFIX_MANDATORY_ARTIFACTS[protocol.CampaignPrefix.P4]
}

_DTYPE_WIDTH = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F16": 2,
    "F32": 4,
}


def _rows() -> tuple[Any, ...]:
    return tuple(protocol.EvaluationRow(*values) for values in ROW_INPUTS)


def _row_index(partition_role: str = "fit") -> Any:
    return protocol.build_row_index_payload(
        partition_role,
        ROW_PROTOCOL_HASH,
        _rows(),
    )


def _authorization(
    payload: Any,
    mode: Any,
    *,
    campaign_id: str = CAMPAIGN_ID,
    overrides: Mapping[str, object] | None = None,
) -> Any:
    precomputed = mode is protocol.AuthorizationMode.PRECOMPUTED
    values: dict[str, object] = {
        "partition_role": payload.partition_role,
        "campaign_id": campaign_id,
        "expected_row_set_mode": mode,
        "row_derivation_protocol_hash": payload.row_derivation_protocol_hash,
        "authorization_nonce": "synthetic-authorization-nonce",
        "expected_row_set_sha256": (
            payload.row_set_sha256 if precomputed else None
        ),
        "expected_ordered_row_sequence_sha256": (
            payload.ordered_row_sequence_sha256 if precomputed else None
        ),
        "expected_row_count": payload.row_count if precomputed else None,
        "expected_row_index_payload_sha256": (
            payload.payload_sha256 if precomputed else None
        ),
        "frozen_bindings": {
            "capture_protocol": "synthetic_v1",
            "row_identity_only": True,
        },
    }
    if overrides:
        values.update(overrides)
    return protocol.AuthorizationSpec(**values)


def _encode_safetensors(
    tensors: Sequence[tuple[str, str, Sequence[int], bytes]],
) -> bytes:
    """Encode a small, valid safetensors file without third-party packages."""

    descriptors: dict[str, object] = {}
    chunks: list[bytes] = []
    cursor = 0
    for name, dtype, shape_value, content in tensors:
        shape = list(shape_value)
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        assert len(content) == element_count * _DTYPE_WIDTH[dtype]
        end = cursor + len(content)
        descriptors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, end],
        }
        chunks.append(content)
        cursor = end
    header = json.dumps(
        descriptors,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + b"".join(chunks)


def _shard_bytes(
    row_ids: Sequence[str],
    *,
    include_row_tensor: bool = True,
    row_dtype: str = "U8",
    row_shape: Sequence[int] | None = None,
    row_content: bytes | None = None,
    activation_shape: Sequence[int] | None = None,
) -> bytes:
    n = len(row_ids)
    activation = list(activation_shape or (n, 2))
    activation_elements = 1
    for dimension in activation:
        activation_elements *= dimension
    tensors: list[tuple[str, str, Sequence[int], bytes]] = [
        (
            "activation",
            "F32",
            activation,
            b"\x00" * (activation_elements * 4),
        )
    ]
    if include_row_tensor:
        shape = list(row_shape or (n, 32))
        if row_content is None:
            row_content = b"".join(bytes.fromhex(row_id) for row_id in row_ids)
        tensors.append(
            ("capture_row_id_raw32", row_dtype, shape, row_content)
        )
    return _encode_safetensors(tensors)


def _evidence(entry: Any, row_ids: Sequence[str] | None = None) -> Any:
    physical_ids = tuple(row_ids or entry.row_ids)
    return protocol.PhysicalShardEvidence(
        _shard_bytes(physical_ids),
        {
            "activation": len(physical_ids),
            "capture_row_id_raw32": len(physical_ids),
        },
    )


def _write_shards(
    capture_root: Path,
    plan: Any,
    *,
    row_ids_by_shard: Mapping[int, Sequence[str]] | None = None,
) -> None:
    replacements = row_ids_by_shard or {}
    for entry in plan.entries:
        path = capture_root / entry.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            _shard_bytes(tuple(replacements.get(entry.shard_index, entry.row_ids)))
        )


def _assert_reason(error: pytest.ExceptionInfo[BaseException], reason: str) -> None:
    assert getattr(error.value, "reason_code", None) == reason


def _assert_transaction_inventory(transaction: Any) -> None:
    receipts = [
        artifact
        for artifact in transaction.artifacts
        if artifact.kind == "transaction_commit_receipt"
    ]
    assert len(receipts) == 1
    receipt = receipts[0]
    expected = [
        {"byte_sha256": artifact.byte_sha256, "name": artifact.name}
        for artifact in sorted(
            (
                artifact
                for artifact in transaction.artifacts
                if artifact is not receipt
            ),
            key=lambda artifact: artifact.name,
        )
    ]
    assert receipt.json_value()["artifacts"] == expected
    assert receipt.name not in {item["name"] for item in expected}
    assert all(
        artifact.transaction_id == transaction.transaction_id
        for artifact in transaction.artifacts
    )


def _coherently_rewrite_transaction_artifact(
    ledger_path: Path,
    artifact_name: str,
    mutate: Any,
) -> None:
    """Rewrite one JSON artifact and every hash-only envelope around it."""

    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.row_factory = sqlite3.Row
        artifact_row = connection.execute(
            "SELECT content, transaction_id FROM artifacts WHERE name = ?",
            (artifact_name,),
        ).fetchone()
        assert artifact_row is not None
        transaction_id = artifact_row["transaction_id"]
        body = json.loads(bytes(artifact_row["content"]).decode("utf-8"))
        mutate(body)
        rewritten = protocol.canonical_json_bytes(body)
        rewritten_hash = hashlib.sha256(rewritten).hexdigest()
        connection.execute(
            "UPDATE artifacts SET content = ?, byte_sha256 = ? WHERE name = ?",
            (rewritten, rewritten_hash, artifact_name),
        )
        connection.execute(
            "UPDATE campaign_artifact_inventory SET byte_sha256 = ?, "
            "byte_count = ? WHERE name = ?",
            (rewritten_hash, len(rewritten), artifact_name),
        )
        receipt_row = connection.execute(
            "SELECT name, content FROM artifacts WHERE transaction_id = ? "
            "AND kind = 'transaction_commit_receipt'",
            (transaction_id,),
        ).fetchone()
        assert receipt_row is not None
        receipt = json.loads(bytes(receipt_row["content"]).decode("utf-8"))
        target = next(
            item for item in receipt["artifacts"] if item["name"] == artifact_name
        )
        target["byte_sha256"] = rewritten_hash
        receipt_bytes = protocol.canonical_json_bytes(receipt)
        receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
        connection.execute(
            "UPDATE artifacts SET content = ?, byte_sha256 = ? WHERE name = ?",
            (receipt_bytes, receipt_hash, receipt_row["name"]),
        )
        connection.execute(
            "UPDATE ledger_transactions SET commit_receipt = ?, "
            "commit_receipt_sha256 = ? WHERE transaction_id = ?",
            (receipt_bytes, receipt_hash, transaction_id),
        )
        connection.execute(
            "UPDATE campaign_artifact_inventory SET byte_sha256 = ?, "
            "byte_count = ? WHERE name = ?",
            (receipt_hash, len(receipt_bytes), receipt_row["name"]),
        )


def _start_capture(
    ledger: Any,
    *,
    maximum_rows_per_shard: int = 2,
    capture_root: Path | None = None,
) -> tuple[Any, Any, str]:
    _register_completed_prefixes(ledger)
    root = (
        Path(ledger.path).resolve().parent / "capture"
        if capture_root is None
        else capture_root
    )
    root.mkdir(parents=True, exist_ok=True)
    payload = _row_index()
    ledger.authorize(
        _authorization(payload, protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION)
    )
    ledger.commit_row_index(payload)
    plan = protocol.build_shard_plan(payload, maximum_rows_per_shard)
    ledger.commit_shard_plan(plan)
    start = ledger.start_capture(
        plan,
        root,
        capture_schema_hash=CAPTURE_SCHEMA_HASH,
        capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
    )
    snapshot = ledger.snapshot("fit")
    assert snapshot is not None and snapshot.capture_attempt_id is not None
    return plan, start, snapshot.capture_attempt_id


def _failure_context() -> dict[str, str]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "last_completed_prefix": "P2",
        "active_prefix": "P3",
        "failure_stage": "tensor_capture",
    }


def _zero_fit_support_spec() -> Any:
    return protocol.SupportFailureSpec(
        campaign_id=CAMPAIGN_ID,
        partition_role="fit",
        last_completed_prefix="P3",
        active_prefix="P4",
        support_stage="fit_monitor_chart_support",
        reason_code="zero_fit_monitor_chart_support",
        active_chart_ids=("chart_a", "chart_empty"),
        expected_head_ids=("head_a",),
        chart_counts={"chart_a": 1, "chart_empty": 0},
        chart_head_validation_counts={
            "chart_a__head_a": 1,
            "chart_empty__head_a": 0,
        },
    )


def _register_completed_prefixes(ledger: Any) -> None:
    for prefix, names in COMPLETED_PREFIX_ARTIFACTS.items():
        for name in names:
            content = (
                b"# Synthetic protocol artifact\n"
                if name.endswith(".md")
                else b'{"synthetic":true}'
            )
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix=prefix,
                name=name,
                content=content,
            )


def _complete_fit_capture(ledger: Any, capture_root: Path) -> tuple[Any, Any]:
    plan, _, attempt = _start_capture(ledger, capture_root=capture_root)
    _write_shards(capture_root, plan)
    verification = protocol.verify_capture_shard_files(plan, capture_root)
    finalized = ledger.finalize_capture_success(
        plan,
        capture_root,
        capture_attempt_id=attempt,
        expected_verification=verification,
    )
    ledger.register_reached_artifact(
        campaign_id=CAMPAIGN_ID,
        prefix="P3",
        name="capture_invariants.json",
        content=b'{"synthetic":true}',
    )
    return plan, finalized


def _complete_partition_capture(
    ledger: Any,
    partition_role: str,
    capture_root: Path,
) -> tuple[Any, Any]:
    capture_root.mkdir(parents=True, exist_ok=True)
    payload = _row_index(partition_role)
    ledger.authorize(
        _authorization(
            payload,
            protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
        )
    )
    ledger.commit_row_index(payload)
    plan = protocol.build_shard_plan(payload, 2)
    ledger.commit_shard_plan(plan)
    ledger.start_capture(
        plan,
        capture_root,
        capture_schema_hash=CAPTURE_SCHEMA_HASH,
        capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
    )
    attempt = ledger.snapshot(partition_role).capture_attempt_id
    assert attempt is not None
    _write_shards(capture_root, plan)
    verification = protocol.verify_capture_shard_files(plan, capture_root)
    finalized = ledger.finalize_capture_success(
        plan,
        capture_root,
        capture_attempt_id=attempt,
        expected_verification=verification,
    )
    return plan, finalized


def _register_missing_mandatory_artifacts(
    ledger: Any,
    prefix: Any,
) -> None:
    present = {entry.name for entry in ledger.campaign_inventory()}
    for name in sorted(protocol._PREFIX_MANDATORY_ARTIFACTS[prefix]):
        if name in present or protocol._is_ledger_owned_artifact_name(name):
            continue
        ledger.register_reached_artifact(
            campaign_id=CAMPAIGN_ID,
            prefix=prefix,
            name=name,
            content=(
                b"# Synthetic protocol artifact\n"
                if name.endswith(".md")
                else b'{"synthetic":true}'
            ),
        )


def test_module_load_is_model_tokenizer_corpus_and_side_effect_free(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        f"""
        import pathlib
        import runpy
        import sys

        before = set(pathlib.Path('.').iterdir())
        namespace = runpy.run_path({str(SOURCE)!r}, run_name='_isolated_protocol')
        after = set(pathlib.Path('.').iterdir())
        assert before == after
        assert namespace['CURRENT_P0_TERMINAL_STATE'] == (
            'development_complete_independent_data_unavailable'
        )
        forbidden = {{'fisher_graph', 'mlx', 'numpy', 'tokenizers', 'torch', 'transformers'}}
        loaded = sorted(
            name for name in sys.modules
            if name.partition('.')[0] in forbidden
        )
        if loaded:
            raise SystemExit('forbidden imports: ' + ','.join(loaded))
        """
    )
    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_exact_row_id_set_order_and_payload_hash_vectors() -> None:
    rows = _rows()
    assert tuple(row.row_id for row in rows) == ROW_IDS_IN_INPUT_ORDER

    payload = _row_index()
    assert payload.row_ids == CANONICAL_ROW_IDS
    assert payload.row_set_sha256 == ROW_SET_SHA256
    assert payload.ordered_row_sequence_sha256 == ORDERED_ROW_SEQUENCE_SHA256
    assert payload.payload_sha256 == ROW_INDEX_PAYLOAD_SHA256
    assert protocol.row_index_payload_sha256(payload) == ROW_INDEX_PAYLOAD_SHA256
    assert protocol.RowIndexPayload.from_canonical_bytes(
        payload.canonical_bytes
    ) == payload


def test_current_p0_campaign_id_cannot_authorize_future_d2() -> None:
    assert protocol.CURRENT_P0_CAMPAIGN_ID == (
        "fisher-compact-message-completion-v2-current-abcd-p0"
    )
    with pytest.raises(
        protocol.ProtocolSchemaError,
        match="current P0 campaign",
    ):
        _authorization(
            _row_index(),
            protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            campaign_id=protocol.CURRENT_P0_CAMPAIGN_ID,
        )


def test_raw_byte_sorting_and_duplicate_rejection() -> None:
    raw = tuple(bytes.fromhex(row_id) for row_id in CANONICAL_ROW_IDS)
    assert protocol.row_set_sha256(tuple(reversed(raw))) == ROW_SET_SHA256
    assert protocol.ordered_row_sequence_sha256(
        tuple(reversed(raw))
    ) == REVERSED_ROW_SEQUENCE_SHA256

    with pytest.raises(protocol.IntegrityViolation) as duplicate_set:
        protocol.row_set_sha256((raw[0], CANONICAL_ROW_IDS[0]))
    _assert_reason(duplicate_set, "duplicate_row_id")
    with pytest.raises(protocol.IntegrityViolation) as duplicate_rows:
        protocol.build_row_index_payload(
            "fit", ROW_PROTOCOL_HASH, (_rows()[0], _rows()[0])
        )
    _assert_reason(duplicate_rows, "duplicate_row_id")
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.row_set_sha256((b"short",))


@pytest.mark.parametrize(
    "mode",
    [
        protocol.AuthorizationMode.PRECOMPUTED,
        protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
    ],
)
def test_both_authorization_modes_commit_identically_bound_row_index(
    tmp_path: Path,
    mode: Any,
) -> None:
    payload = _row_index()
    with protocol.ProtocolLedger(tmp_path / f"{mode.value}.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        authorization = ledger.authorize(_authorization(payload, mode))
        assert authorization.state_after is protocol.CaptureState.AUTHORIZED
        _assert_transaction_inventory(authorization)

        committed = ledger.commit_row_index(payload)
        assert committed.state_after is protocol.CaptureState.ROW_INDEX_COMMITTED
        _assert_transaction_inventory(committed)
        manifest = committed.artifact("row_index_manifest_fit.json").json_value()
        transition = committed.artifact("fit_row_index_committed.json").json_value()
        for key, expected in {
            "authorization_hash": authorization.identity_sha256,
            "authorization_transaction_id": authorization.transaction_id,
            "derived_ordered_row_sequence_sha256": ORDERED_ROW_SEQUENCE_SHA256,
            "derived_row_count": 3,
            "derived_row_set_sha256": ROW_SET_SHA256,
            "partition_role": "fit",
            "row_derivation_protocol_hash": ROW_PROTOCOL_HASH,
            "row_index_payload_sha256": ROW_INDEX_PAYLOAD_SHA256,
            "row_index_transaction_id": committed.transaction_id,
        }.items():
            assert manifest[key] == expected
            assert transition[key] == expected

        auth_body = authorization.artifact(
            "fit_capture_authorization.json"
        ).json_value()
        expected = None if mode is protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION else ROW_INDEX_PAYLOAD_SHA256
        assert auth_body["expected_row_index_payload_sha256"] == expected
        retry = ledger.commit_row_index(payload)
        assert retry.idempotent_recovery
        assert retry.transaction_id == committed.transaction_id


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("expected_row_set_sha256", "aa" * 32),
        ("expected_ordered_row_sequence_sha256", "bb" * 32),
        ("expected_row_count", 4),
        ("expected_row_index_payload_sha256", "cc" * 32),
    ],
)
def test_mode_a_rejects_every_precomputed_payload_binding_mismatch(
    tmp_path: Path,
    field: str,
    wrong_value: object,
) -> None:
    payload = _row_index()
    with protocol.ProtocolLedger(tmp_path / f"{field}.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.PRECOMPUTED,
                overrides={field: wrong_value},
            )
        )
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.commit_row_index(payload)
        _assert_reason(caught, "precomputed_row_index_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.AUTHORIZED


def test_mode_b_requires_null_expected_bindings_and_pre_capture_commits(
    tmp_path: Path,
) -> None:
    payload = _row_index()
    with pytest.raises(protocol.ProtocolSchemaError):
        _authorization(
            payload,
            protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            overrides={"expected_row_count": payload.row_count},
        )

    plan = protocol.build_shard_plan(payload, 2)
    with protocol.ProtocolLedger(tmp_path / "pre-capture.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            )
        )
        with pytest.raises(protocol.ProtocolStateError):
            ledger.start_capture(
                plan,
                tmp_path,
                capture_schema_hash=CAPTURE_SCHEMA_HASH,
                capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
            )
        ledger.commit_row_index(payload)
        with pytest.raises(protocol.ProtocolStateError):
            ledger.start_capture(
                plan,
                tmp_path,
                capture_schema_hash=CAPTURE_SCHEMA_HASH,
                capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
            )


def test_exact_k3_shard_ranges_paths_domain_and_limit() -> None:
    payload = _row_index()
    plan = protocol.build_shard_plan(payload, 2)
    assert [
        (
            entry.shard_index,
            entry.start_row_ordinal,
            entry.end_row_ordinal_exclusive,
            entry.path,
        )
        for entry in plan.entries
    ] == [
        (0, 0, 2, "capture_shards/fit/shard-000000.safetensors"),
        (1, 2, 3, "capture_shards/fit/shard-000001.safetensors"),
    ]
    assert plan.as_dict()["shard_path_template"] == (
        "capture_shards/{partition}/shard-{shard_index:06d}.safetensors"
    )
    assert plan.plan_sha256 == SHARD_PLAN_SHA256_L2
    assert protocol.build_shard_plan(payload, 16_384).maximum_rows_per_shard == 16_384
    for invalid in (0, 16_385, True):
        with pytest.raises(protocol.ProtocolSchemaError):
            protocol.build_shard_plan(payload, invalid)


def test_physical_safetensors_extracts_u8_row_ids_and_checks_all_dimensions() -> None:
    entry = protocol.build_shard_plan(_row_index(), 2).entries[0]
    shard_bytes = _shard_bytes(entry.row_ids)
    record = protocol.verify_physical_shard(
        entry,
        protocol.PhysicalShardEvidence(
            shard_bytes,
            {"activation": 2, "capture_row_id_raw32": 2},
        ),
        shard_plan_sha256=protocol.build_shard_plan(
            _row_index(), 2
        ).plan_sha256,
    )
    assert record.row_ids == entry.row_ids
    assert record.shard_byte_sha256 == hashlib.sha256(shard_bytes).hexdigest()
    assert record.start_row_ordinal == 0
    assert record.end_row_ordinal_exclusive == 2

    bad = _shard_bytes(entry.row_ids, activation_shape=(1, 2))
    with pytest.raises(protocol.MeasurementContractViolation) as caught:
        protocol.verify_physical_shard(
            entry,
            protocol.PhysicalShardEvidence(bad),
            shard_plan_sha256=protocol.build_shard_plan(
                _row_index(), 2
            ).plan_sha256,
        )
    _assert_reason(caught, "tensor_row_identity_alignment_mismatch")


@pytest.mark.parametrize(
    ("case", "exception_type", "reason_code"),
    [
        ("missing", "measurement", "tensor_row_identity_alignment_mismatch"),
        ("reordered", "integrity", "capture_row_order_mismatch"),
        ("wrong_set", "integrity", "row_set_mismatch"),
        ("duplicate", "integrity", "duplicate_row_id"),
        ("dtype", "measurement", "invalid_tensor_dtype"),
        ("shape", "measurement", "invalid_tensor_shape"),
    ],
)
def test_missing_reordered_duplicate_or_malformed_row_tensor_fails_exactly(
    case: str,
    exception_type: str,
    reason_code: str,
) -> None:
    entry = protocol.build_shard_plan(_row_index(), 2).entries[0]
    if case == "missing":
        content = _shard_bytes(entry.row_ids, include_row_tensor=False)
    elif case == "reordered":
        content = _shard_bytes(tuple(reversed(entry.row_ids)))
    elif case == "wrong_set":
        content = _shard_bytes((entry.row_ids[0], "ee" * 32))
    elif case == "duplicate":
        content = _shard_bytes((entry.row_ids[0], entry.row_ids[0]))
    elif case == "dtype":
        content = _shard_bytes(entry.row_ids, row_dtype="I8")
    else:
        raw = b"".join(bytes.fromhex(row_id) for row_id in entry.row_ids)
        content = _shard_bytes(entry.row_ids, row_shape=(4, 16), row_content=raw)
    expected_exception = (
        protocol.IntegrityViolation
        if exception_type == "integrity"
        else protocol.MeasurementContractViolation
    )
    with pytest.raises(expected_exception) as caught:
        protocol.verify_physical_shard(
            entry,
            protocol.PhysicalShardEvidence(content),
            shard_plan_sha256=protocol.build_shard_plan(
                _row_index(), 2
            ).plan_sha256,
        )
    _assert_reason(caught, reason_code)


def test_complete_shard_verification_uses_frozen_plan_order() -> None:
    plan = protocol.build_shard_plan(_row_index(), 2)
    verification = protocol.verify_capture_shards(
        plan,
        {1: _evidence(plan.entries[1]), 0: _evidence(plan.entries[0])},
    )
    assert verification.row_count == 3
    assert verification.captured_row_set_sha256 == ROW_SET_SHA256
    assert (
        verification.captured_ordered_row_sequence_sha256
        == ORDERED_ROW_SEQUENCE_SHA256
    )
    assert tuple(shard.shard_index for shard in verification.shards) == (0, 1)
    with pytest.raises(protocol.IntegrityViolation) as caught:
        protocol.verify_capture_shards(plan, {0: _evidence(plan.entries[0])})
    _assert_reason(caught, "row_set_mismatch")


def test_exact_on_disk_paths_are_reopened_and_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    plan = protocol.build_shard_plan(_row_index(), 2)
    capture_root = tmp_path / "capture"
    _write_shards(capture_root, plan)
    verification = protocol.verify_capture_shard_files(plan, capture_root)
    assert verification.verification_source == "finalized_files_reopened"
    assert tuple(shard.path for shard in verification.shards) == tuple(
        entry.path for entry in plan.entries
    )
    assert all(
        shard.shard_plan_sha256 == plan.plan_sha256
        for shard in verification.shards
    )

    external = tmp_path / "outside.safetensors"
    external.write_bytes(_shard_bytes(plan.entries[0].row_ids))
    planned = capture_root / plan.entries[0].path
    planned.unlink()
    planned.symlink_to(external)
    with pytest.raises(protocol.IntegrityViolation) as caught:
        protocol.verify_physical_shard_file(plan, 0, capture_root)
    _assert_reason(caught, "shard_symlink_forbidden")


def test_unplanned_extra_safetensors_shard_is_rejected(
    tmp_path: Path,
) -> None:
    plan = protocol.build_shard_plan(_row_index(), 2)
    capture_root = tmp_path / "capture"
    _write_shards(capture_root, plan)
    extra = (
        capture_root
        / "capture_shards"
        / "fit"
        / "shard-999999.safetensors"
    )
    extra.write_bytes(_shard_bytes(plan.entries[0].row_ids))
    with pytest.raises(protocol.IntegrityViolation) as caught:
        protocol.verify_capture_shard_files(plan, capture_root)
    _assert_reason(caught, "shard_plan_mismatch")


def test_cross_shard_permutation_is_exact_row_order_mismatch(
    tmp_path: Path,
) -> None:
    fourth = protocol.EvaluationRow("77" * 32, "88" * 32, 20, 21)
    payload = protocol.build_row_index_payload(
        "fit", ROW_PROTOCOL_HASH, (*_rows(), fourth)
    )
    plan = protocol.build_shard_plan(payload, 2)
    assert len(plan.entries) == 2
    capture_root = tmp_path / "capture"
    _write_shards(
        capture_root,
        plan,
        row_ids_by_shard={
            0: plan.entries[1].row_ids,
            1: plan.entries[0].row_ids,
        },
    )
    with pytest.raises(protocol.IntegrityViolation) as caught:
        protocol.verify_capture_shard_files(plan, capture_root)
    _assert_reason(caught, "capture_row_order_mismatch")


def test_direct_construction_cannot_forge_physical_verification() -> None:
    plan = protocol.build_shard_plan(_row_index(), 2)
    entry = plan.entries[0]
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.VerifiedShardRecord(
            partition_role="fit",
            shard_index=0,
            start_row_ordinal=0,
            end_row_ordinal_exclusive=2,
            path=entry.path,
            row_ids=entry.row_ids,
            shard_byte_sha256="00" * 32,
            shard_plan_sha256=plan.plan_sha256,
            _construction_token=object(),
        )
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.CaptureVerification(
            partition_role="fit",
            shard_plan_sha256=plan.plan_sha256,
            committed_row_set_sha256=plan.row_set_sha256,
            committed_ordered_row_sequence_sha256=(
                plan.ordered_row_sequence_sha256
            ),
            captured_row_set_sha256=plan.row_set_sha256,
            captured_ordered_row_sequence_sha256=(
                plan.ordered_row_sequence_sha256
            ),
            row_count=plan.row_count,
            shards=(),
            verification_source="finalized_files_reopened",
            _construction_token=object(),
        )
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.PartialShardInventory(
            partition_role="fit",
            shard_plan_sha256=plan.plan_sha256,
            shards=(),
            verification_source="finalized_files_reopened",
            _construction_token=object(),
        )


def test_in_memory_verification_cannot_be_used_as_finalization_evidence(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "ledger.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger)
        _write_shards(capture_root, plan)
        in_memory = protocol.verify_capture_shards(
            plan,
            {
                entry.shard_index: _evidence(entry)
                for entry in plan.entries
            },
        )
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_success(
                plan,
                capture_root,
                capture_attempt_id=attempt,
                expected_verification=in_memory,
            )
        _assert_reason(caught, "shard_hash_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED
        assert ledger.get_artifact("capture_shard_manifest_fit.json") is None


def test_mutation_after_file_verification_is_rejected_before_success(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "ledger.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger)
        _write_shards(capture_root, plan)
        before = protocol.verify_capture_shard_files(plan, capture_root)
        first_path = capture_root / plan.entries[0].path
        mutated = bytearray(first_path.read_bytes())
        data_offset = 8 + struct.unpack_from("<Q", mutated, 0)[0]
        mutated[data_offset] ^= 1
        first_path.write_bytes(mutated)

        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_success(
                plan,
                capture_root,
                capture_attempt_id=attempt,
                expected_verification=before,
            )
        _assert_reason(caught, "shard_hash_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED
        assert ledger.get_artifact("capture_shard_manifest_fit.json") is None


def test_finalized_shards_are_file_backed_and_deletion_blocks_terminal(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "shard-deletion.sqlite3"
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        plan, _ = _complete_fit_capture(ledger, capture_root)
        shard_entries = {
            entry.name: entry for entry in ledger.campaign_inventory()
            if entry.name.startswith("capture_shards/fit/")
        }
        assert set(shard_entries) == {entry.path for entry in plan.entries}
        for entry in plan.entries:
            (capture_root / entry.path).unlink()
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_support_failure(
                _zero_fit_support_spec(),
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        _assert_reason(caught, "artifact_hash_mismatch")
        assert ledger.get_artifact("final_experiment_manifest.json") is None


def test_partial_shard_mutation_after_verification_is_rejected_before_failure(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "ledger.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger)
        _write_shards(capture_root, plan)
        (capture_root / plan.entries[1].path).unlink()
        before = protocol.verify_partial_capture_shard_files(
            plan, capture_root, (0,)
        )
        first_path = capture_root / plan.entries[0].path
        mutated = bytearray(first_path.read_bytes())
        data_offset = 8 + struct.unpack_from("<Q", mutated, 0)[0]
        mutated[data_offset] ^= 1
        first_path.write_bytes(mutated)

        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_failure(
                "fit",
                capture_attempt_id=attempt,
                reason_code="memory_resource_limit",
                plan=plan,
                capture_root=capture_root,
                finalized_partial_shard_indexes=(0,),
                expected_partial_shard_inventory=before,
                **_failure_context(),
            )
        _assert_reason(caught, "shard_hash_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED
        assert ledger.get_artifact("capture_failure_fit.json") is None


def test_durable_authorize_index_plan_start_success_and_exact_hash_closure(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "success.sqlite3"
    capture_root = tmp_path / "success-capture"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        plan, start, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        assert start.state_after is protocol.CaptureState.CAPTURE_STARTED
        started = start.artifact("fit_tensor_capture_started.json").json_value()
        assert started["capture_attempt_id"] == attempt
        assert started["shard_plan_sha256"] == plan.plan_sha256
        assert started["tensor_capture_started"] is True
        _assert_transaction_inventory(start)

        _write_shards(capture_root, plan)
        verification = protocol.verify_capture_shard_files(plan, capture_root)
        finalized = ledger.finalize_capture_success(
            plan,
            capture_root,
            capture_attempt_id=attempt,
            expected_verification=verification,
        )
        assert finalized.state_after is protocol.CaptureState.CONSUMED_SUCCESS
        assert {
            artifact.name for artifact in finalized.artifacts
        } == {
            "capture_shard_manifest_fit.json",
            "fit_capture_transition_consumed_success.json",
            "fit_capture_success_transaction_commit.json",
        }
        _assert_transaction_inventory(finalized)
        manifest_body = finalized.artifact(
            "capture_shard_manifest_fit.json"
        ).json_value()
        transition_body = finalized.artifact(
            "fit_capture_transition_consumed_success.json"
        ).json_value()
        assert transition_body["shard_plan_sha256"] == plan.plan_sha256
        assert all(
            shard["shard_plan_sha256"] == plan.plan_sha256
            for shard in manifest_body["shards"]
        )
        assert ledger.get_artifact("capture_failure_fit.json") is None
        assert ledger.get_artifact("invalid_campaign_receipt.json") is None
        assert ledger.get_artifact("resource_limit_failure.json") is None
        assert ledger.verify_artifact_closure("fit").valid

    with protocol.ProtocolLedger(ledger_path) as reopened:
        assert reopened.snapshot("fit").state is protocol.CaptureState.CONSUMED_SUCCESS
        assert reopened.verify_artifact_closure("fit").valid

    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        original = connection.execute(
            "SELECT content FROM artifacts WHERE name = ?",
            ("capture_shard_manifest_fit.json",),
        ).fetchone()[0]
        connection.execute(
            "UPDATE artifacts SET content = ? WHERE name = ?",
            (bytes(original) + b" ", "capture_shard_manifest_fit.json"),
        )
    with protocol.ProtocolLedger(ledger_path) as tampered:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            tampered.get_artifact("capture_shard_manifest_fit.json")
        _assert_reason(caught, "artifact_hash_mismatch")


def test_row_index_failure_never_starts_capture_and_gets_one_primary_receipt(
    tmp_path: Path,
) -> None:
    payload = _row_index()
    context = {
        **_failure_context(),
        "failure_stage": "row_derivation",
    }
    with protocol.ProtocolLedger(tmp_path / "row-failure.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            )
        )
        failed = ledger.record_row_index_failure(
            "fit",
            reason_code="unable_to_derive_expected_panel_rows",
            sanitized_failure_details={"stage": "row_derivation"},
            **context,
        )
        assert failed.state_after is protocol.CaptureState.ROW_INDEX_FAILED
        assert ledger.get_artifact("invalid_campaign_receipt.json") is None
        row_failure = failed.artifact("row_index_failure_fit.json").json_value()
        assert row_failure["tensor_capture_started"] is False
        assert ledger.snapshot("fit").capture_attempt_id is None

        finalized = ledger.finalize_row_index_failure("fit", **context)
        assert finalized.state_after is protocol.CaptureState.CONSUMED_FAILED
        assert {
            artifact.name for artifact in finalized.artifacts
        } == {
            "fit_capture_transition_consumed_failed.json",
            "fit_row_index_failure_finalization_commit.json",
            "invalid_campaign_receipt.json",
        }
        _assert_transaction_inventory(finalized)
        primary = finalized.artifact("invalid_campaign_receipt.json").json_value()
        assert primary["failure_class"] == "measurement_contract"
        assert primary["failure_stage"] == "row_derivation"
        assert primary["tensor_capture_started"] is False
        assert ledger.get_artifact("resource_limit_failure.json") is None
        names = {artifact.name for artifact in ledger.list_artifacts("fit")}
        assert not any("shard" in name for name in names)
        assert "fit_tensor_capture_started.json" not in names
        assert ledger.verify_artifact_closure("fit").valid


def test_recoverable_interruption_preserves_started_state_and_attempt_id(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "interrupted-capture"
    capture_root.mkdir()
    with protocol.ProtocolLedger(tmp_path / "interruption.sqlite3") as ledger:
        plan, start, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        interruption = ledger.record_recoverable_interruption(
            "fit",
            capture_attempt_id=attempt,
            sanitized_failure_details={
                "exception_type": "InterruptedError",
                "stage": "tensor_capture",
            },
        )
        assert interruption.state_before is protocol.CaptureState.CAPTURE_STARTED
        assert interruption.state_after is protocol.CaptureState.CAPTURE_STARTED
        assert ledger.snapshot("fit").capture_attempt_id == attempt
        resumed = ledger.resume_capture(
            plan,
            capture_attempt_id=attempt,
            capture_root=capture_root,
            capture_schema_hash=CAPTURE_SCHEMA_HASH,
            capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
        )
        assert resumed.start_transaction.idempotent_recovery
        assert resumed.start_transaction.transaction_id == start.transaction_id
        assert resumed.verified_existing_shards is None
        assert resumed.missing_shard_indexes == (0, 1)
        restarted = ledger.start_capture(
            plan,
            capture_root,
            capture_schema_hash=CAPTURE_SCHEMA_HASH,
            capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
        )
        assert restarted.idempotent_recovery
        assert ledger.snapshot("fit").capture_attempt_id == attempt
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED
        assert ledger.get_artifact("capture_failure_fit.json") is None


def test_resume_checkpoint_detects_deleted_pin_row_before_finalization(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "resume-delete.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger, capture_root=capture_root)
        first = plan.entries[0]
        first_path = capture_root / first.path
        first_path.parent.mkdir(parents=True, exist_ok=True)
        first_path.write_bytes(_shard_bytes(first.row_ids))
        resumed = ledger.resume_capture(
            plan,
            capture_attempt_id=attempt,
            capture_root=capture_root,
            capture_schema_hash=CAPTURE_SCHEMA_HASH,
            capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
        )
        assert resumed.verified_existing_shards is not None
        assert ledger.snapshot("fit").resume_checkpoint_count == 1
        ledger._connection.execute(
            "DELETE FROM capture_resume_shards WHERE partition_role = 'fit'"
        )
        mutated = bytearray(first_path.read_bytes())
        data_offset = 8 + struct.unpack_from("<Q", mutated, 0)[0]
        mutated[data_offset] ^= 1
        first_path.write_bytes(mutated)
        second = plan.entries[1]
        second_path = capture_root / second.path
        second_path.write_bytes(_shard_bytes(second.row_ids))
        fresh = protocol.verify_capture_shard_files(plan, capture_root)
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_success(
                plan,
                capture_root,
                capture_attempt_id=attempt,
                expected_verification=fresh,
            )
        _assert_reason(caught, "shard_hash_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED


@pytest.mark.parametrize(
    ("reason", "failure_class", "terminal"),
    [
        ("row_set_mismatch", "integrity", "invalid_campaign_integrity"),
        (
            "invalid_tensor_shape",
            "measurement_contract",
            "invalid_campaign_measurement_contract",
        ),
        (
            "memory_resource_limit",
            "resource_limit",
            "development_complete_resource_limited",
        ),
    ],
)
def test_failure_reason_mapping_is_exact_and_exclusive(
    reason: str,
    failure_class: str,
    terminal: str,
) -> None:
    disposition = protocol.classify_failure(reason)
    assert disposition.reason_code == reason
    assert disposition.failure_class.value == failure_class
    assert disposition.terminal_state.value == terminal


@pytest.mark.parametrize(
    "details",
    [
        {"raw_text": "secret"},
        {"message": "secret"},
        {"error_code": b"secret"},
        {"available": 1.5},
        {"expected_shape": [2, True]},
    ],
)
def test_failure_details_reject_raw_free_form_and_ambiguous_values(
    details: Mapping[str, object],
) -> None:
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.validate_sanitized_failure_details(details)
    assert protocol.validate_sanitized_failure_details(
        {
            "exception_type": "MemoryError",
            "expected_shape": [2, 32],
            "observed_count": 1,
            "stage": "tensor_capture",
        }
    ) == {
        "exception_type": "MemoryError",
        "expected_shape": [2, 32],
        "observed_count": 1,
        "stage": "tensor_capture",
    }


@pytest.mark.parametrize(
    ("reason", "failure_class", "terminal", "primary_name"),
    [
        (
            "row_set_mismatch",
            "integrity",
            "invalid_campaign_integrity",
            "invalid_campaign_receipt.json",
        ),
        (
            "invalid_tensor_shape",
            "measurement_contract",
            "invalid_campaign_measurement_contract",
            "invalid_campaign_receipt.json",
        ),
        (
            "memory_resource_limit",
            "resource_limit",
            "development_complete_resource_limited",
            "resource_limit_failure.json",
        ),
    ],
)
def test_capture_failure_atomically_closes_with_one_k5_primary_receipt(
    tmp_path: Path,
    reason: str,
    failure_class: str,
    terminal: str,
    primary_name: str,
) -> None:
    capture_root = tmp_path / f"{failure_class}-capture"
    with protocol.ProtocolLedger(tmp_path / f"{failure_class}.sqlite3") as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        _write_shards(
            capture_root,
            plan,
            row_ids_by_shard={0: plan.entries[0].row_ids},
        )
        (capture_root / plan.entries[1].path).unlink()
        expected_partial = protocol.verify_partial_capture_shard_files(
            plan, capture_root, (0,)
        )
        finalized = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code=reason,
            sanitized_failure_details={
                "expected_count": 3,
                "observed_count": 2,
                "stage": "tensor_capture",
            },
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(0,),
            expected_partial_shard_inventory=expected_partial,
            **_failure_context(),
        )
        assert finalized.state_after is protocol.CaptureState.CONSUMED_FAILED
        expected_names = {
            "capture_failure_fit.json",
            "fit_capture_transition_consumed_failed.json",
            "fit_capture_failure_transaction_commit.json",
            "partial_shard_inventory_fit.json",
            primary_name,
        }
        assert {artifact.name for artifact in finalized.artifacts} == expected_names
        _assert_transaction_inventory(finalized)
        other_primary = (
            "resource_limit_failure.json"
            if primary_name == "invalid_campaign_receipt.json"
            else "invalid_campaign_receipt.json"
        )
        assert ledger.get_artifact(other_primary) is None
        assert ledger.get_artifact("capture_shard_manifest_fit.json") is None

        primary = finalized.artifact(primary_name).json_value()
        assert primary["campaign_id"] == CAMPAIGN_ID
        assert primary["failure_class"] == failure_class
        assert primary["terminal_state"] == terminal
        assert primary["last_completed_prefix"] == "P2"
        assert primary["active_prefix"] == "P3"
        assert primary["failure_stage"] == "tensor_capture"
        assert primary["capture_attempt_id"] == attempt
        excluded = {
            primary_name,
            "fit_capture_failure_transaction_commit.json",
        }
        expected_reached = [
            entry.as_dict()
            for entry in ledger.campaign_inventory()
            if entry.name not in excluded
        ]
        assert primary["durably_reached_artifacts"] == expected_reached
        assert ledger.verify_artifact_closure("fit").valid

        retry = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code=reason,
            sanitized_failure_details={
                "expected_count": 3,
                "observed_count": 2,
                "stage": "tensor_capture",
            },
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(0,),
            expected_partial_shard_inventory=expected_partial,
            **_failure_context(),
        )
        assert retry.idempotent_recovery
        assert retry.transaction_id == finalized.transaction_id


def test_capture_failure_without_complete_partial_omits_partial_inventory(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "no-partial-capture"
    capture_root.mkdir()
    with protocol.ProtocolLedger(tmp_path / "no-partial.sqlite3") as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        finalized = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="storage_resource_limit",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(),
            **_failure_context(),
        )
        assert "partial_shard_inventory_fit.json" not in {
            artifact.name for artifact in finalized.artifacts
        }
        assert ledger.verify_artifact_closure("fit").valid


def test_malformed_finalized_shard_can_commit_measurement_terminal(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "malformed-capture"
    with protocol.ProtocolLedger(tmp_path / "malformed.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger, capture_root=capture_root)
        first = plan.entries[0]
        first_path = capture_root / first.path
        first_path.parent.mkdir(parents=True, exist_ok=True)
        first_path.write_bytes(
            _shard_bytes(first.row_ids, activation_shape=(1, 2))
        )
        finalized = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="tensor_row_identity_alignment_mismatch",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(0,),
            expected_partial_shard_inventory=None,
            sanitized_failure_details={
                "shard_index": 0,
                "stage": "tensor_capture",
            },
            **_failure_context(),
        )
        assert finalized.state_after is protocol.CaptureState.CONSUMED_FAILED
        assert ledger.get_artifact("partial_shard_inventory_fit.json") is None
        failure = finalized.artifact("capture_failure_fit.json").json_value()
        assert failure["unverified_finalized_shards"] == [
            {
                "artifact_byte_sha256": hashlib.sha256(
                    first_path.read_bytes()
                ).hexdigest(),
                "byte_count": first_path.stat().st_size,
                "path": first.path,
                "path_entry_type": "regular",
                "reason_code": "tensor_row_identity_alignment_mismatch",
                "shard_index": 0,
            }
        ]
        terminal = ledger.finalize_campaign_terminal(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P2",
            active_prefix="P3",
            omitted_active_artifacts=FIT_P3_AFTER_CAPTURE_FAILURE_OMISSIONS,
        )
        assert terminal.terminal_state is (
            protocol.TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT
        )
        assert terminal.prefix_closure.valid


def test_symlink_shard_failure_is_bound_without_stranding_started_capture(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "symlink-capture"
    with protocol.ProtocolLedger(tmp_path / "symlink.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger, capture_root=capture_root)
        first = plan.entries[0]
        first_path = capture_root / first.path
        first_path.parent.mkdir(parents=True, exist_ok=True)
        target = tmp_path / "untrusted.safetensors"
        target.write_bytes(_shard_bytes(first.row_ids))
        first_path.symlink_to(target)
        finalized = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="shard_symlink_forbidden",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(0,),
            expected_partial_shard_inventory=None,
            sanitized_failure_details={
                "shard_index": 0,
                "stage": "tensor_capture",
            },
            **_failure_context(),
        )
        failure = finalized.artifact("capture_failure_fit.json").json_value()
        assert failure["unverified_finalized_shards"][0]["path_entry_type"] == (
            "symlink"
        )
        assert ledger.snapshot("fit").state is protocol.CaptureState.CONSUMED_FAILED


@pytest.mark.parametrize(
    "tamper_kind",
    [
        "missing_manifest",
        "missing_commit_receipt",
        "rewritten_transition",
        "rewritten_commit_inventory",
    ],
)
def test_idempotent_success_detects_missing_or_rewritten_artifacts(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    ledger_path = tmp_path / f"{tamper_kind}.sqlite3"
    capture_root = tmp_path / f"{tamper_kind}-capture"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        _write_shards(capture_root, plan)
        expected = protocol.verify_capture_shard_files(plan, capture_root)
        ledger.finalize_capture_success(
            plan,
            capture_root,
            capture_attempt_id=attempt,
            expected_verification=expected,
        )

    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        if tamper_kind == "missing_manifest":
            connection.execute(
                "DELETE FROM artifacts WHERE name = ?",
                ("capture_shard_manifest_fit.json",),
            )
        elif tamper_kind == "missing_commit_receipt":
            connection.execute(
                "DELETE FROM artifacts WHERE name = ?",
                ("fit_capture_success_transaction_commit.json",),
            )
        elif tamper_kind == "rewritten_transition":
            name = "fit_capture_transition_consumed_success.json"
            original = bytes(
                connection.execute(
                    "SELECT content FROM artifacts WHERE name = ?", (name,)
                ).fetchone()[0]
            )
            rewritten = original + b" "
            connection.execute(
                "UPDATE artifacts SET content = ?, byte_sha256 = ? WHERE name = ?",
                (rewritten, hashlib.sha256(rewritten).hexdigest(), name),
            )
        else:
            name = "fit_capture_success_transaction_commit.json"
            row = connection.execute(
                "SELECT content, transaction_id FROM artifacts WHERE name = ?",
                (name,),
            ).fetchone()
            body = json.loads(bytes(row[0]).decode("utf-8"))
            body["artifacts"][0]["byte_sha256"] = "00" * 32
            rewritten = protocol.canonical_json_bytes(body)
            digest = hashlib.sha256(rewritten).hexdigest()
            connection.execute(
                "UPDATE artifacts SET content = ?, byte_sha256 = ? WHERE name = ?",
                (rewritten, digest, name),
            )
            connection.execute(
                "UPDATE ledger_transactions SET commit_receipt = ?, "
                "commit_receipt_sha256 = ? WHERE transaction_id = ?",
                (rewritten, digest, row[1]),
            )

    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_success(
                plan,
                capture_root,
                capture_attempt_id=attempt,
                expected_verification=expected,
            )
        _assert_reason(caught, "artifact_hash_mismatch")


@pytest.mark.parametrize(
    ("artifact_name", "field", "replacement"),
    [
        (
            "fit_capture_transition_consumed_success.json",
            "state_after",
            "consumed_failed",
        ),
        (
            "capture_shard_manifest_fit.json",
            "captured_row_set_sha256",
            "aa" * 32,
        ),
    ],
)
def test_semantic_recovery_rejects_coherent_success_transaction_rewrite(
    tmp_path: Path,
    artifact_name: str,
    field: str,
    replacement: str,
) -> None:
    ledger_path = tmp_path / f"coherent-{field}.sqlite3"
    capture_root = tmp_path / f"coherent-{field}-capture"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _complete_fit_capture(ledger, capture_root)
    _coherently_rewrite_transaction_artifact(
        ledger_path,
        artifact_name,
        lambda body: body.__setitem__(field, replacement),
    )
    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_support_failure(
                _zero_fit_support_spec(),
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        _assert_reason(caught, "artifact_hash_mismatch")
        assert ledger.get_artifact("final_experiment_manifest.json") is None


@pytest.mark.parametrize("tamper", ["kind", "state", "name"])
def test_transaction_recovery_rejects_row_kind_state_and_name_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    ledger_path = tmp_path / f"row-{tamper}.sqlite3"
    capture_root = tmp_path / f"row-{tamper}-capture"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _complete_fit_capture(ledger, capture_root)
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        if tamper == "kind":
            connection.execute(
                "UPDATE artifacts SET kind = 'capture_shard_plan' WHERE name = ?",
                ("row_index_manifest_fit.json",),
            )
        elif tamper == "state":
            transaction_id = connection.execute(
                "SELECT transaction_id FROM artifacts WHERE name = ?",
                ("row_index_manifest_fit.json",),
            ).fetchone()[0]
            connection.execute(
                "UPDATE ledger_transactions SET state_after = 'authorized' "
                "WHERE transaction_id = ?",
                (transaction_id,),
            )
        else:
            connection.execute(
                "UPDATE artifacts SET name = ? WHERE name = ?",
                ("row_index_manifest_fit_alias.json", "row_index_manifest_fit.json"),
            )
    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_support_failure(
                _zero_fit_support_spec(),
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        _assert_reason(caught, "artifact_hash_mismatch")


def test_primary_failure_globally_stops_other_panels_and_is_sticky(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "global-terminal.sqlite3"
    capture_root = tmp_path / "empty-capture"
    capture_root.mkdir()
    fit_payload = _row_index("fit")
    tune_payload = _row_index("tune")
    with protocol.ProtocolLedger(ledger_path) as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        terminal = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="memory_resource_limit",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(),
            **_failure_context(),
        )
        with pytest.raises(protocol.ProtocolStateError, match="globally terminal"):
            ledger.authorize(
                _authorization(
                    tune_payload,
                    protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
                )
            )
        assert ledger.snapshot("tune") is None
        retry = ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="memory_resource_limit",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(),
            **_failure_context(),
        )
        assert retry.idempotent_recovery
        assert retry.transaction_id == terminal.transaction_id

        dev_payload = _row_index("development_confirm")
        with pytest.raises(protocol.ProtocolStateError, match="globally terminal"):
            ledger.authorize(
                _authorization(
                    dev_payload,
                    protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
                )
            )

    with protocol.ProtocolLedger(ledger_path) as reopened:
        with pytest.raises(protocol.ProtocolStateError, match="globally terminal"):
            reopened.authorize(
                _authorization(
                    tune_payload,
                    protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
                )
            )


@pytest.mark.parametrize(
    "reason_code",
    [
        "zero_fit_monitor_chart_support",
        "insufficient_independent_chart_support",
    ],
)
def test_support_failure_has_separate_receipt_and_manifest_last_terminal(
    tmp_path: Path,
    reason_code: str,
) -> None:
    ledger_path = tmp_path / f"support-{reason_code}.sqlite3"
    payload = _row_index()
    specification = protocol.SupportFailureSpec(
        campaign_id=CAMPAIGN_ID,
        partition_role="fit",
        last_completed_prefix="P3",
        active_prefix="P4",
        support_stage="fit_monitor_chart_support",
        reason_code=reason_code,
        attempted_locator_K_values=(64, 32),
        attempted_representation_or_codec_ids=("codec-a", "codec-b"),
        analysis_unit_counts={"fit": 3},
        sequence_counts={"fit": 3},
        active_chart_ids=("chart_a", "chart_empty"),
        expected_head_ids=("head_a",),
        chart_counts={
            "chart_a": 1,
            "chart_empty": 0 if reason_code.startswith("zero_") else 1,
        },
        chart_head_validation_counts={
            "chart_a__head_a": 1,
            "chart_empty__head_a": (
                0 if reason_code.startswith("zero_") else 1
            ),
        },
        fallback_attempts=("lower_K",),
    )
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _complete_fit_capture(ledger, tmp_path / "capture")
        terminal = ledger.finalize_support_failure(
            specification,
            omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
        )
        assert terminal.terminal_state is (
            protocol.TerminalState.DEVELOPMENT_COMPLETE_INSUFFICIENT_SUPPORT
        )
        assert terminal.prefix_closure.valid
        assert {
            artifact.name for artifact in terminal.artifacts
        } == {
            "support_failure_receipt.json",
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        }
        assert ledger.get_artifact("invalid_campaign_receipt.json") is None
        assert ledger.get_artifact("resource_limit_failure.json") is None
        support = terminal.transaction.artifact(
            "support_failure_receipt.json"
        ).json_value()
        assert support["reason_code"] == reason_code
        assert support["terminal_state"] == (
            "development_complete_insufficient_support"
        )
        inventory = ledger.campaign_inventory()
        assert [entry.name for entry in inventory[-4:]] == [
            "support_failure_receipt.json",
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]
        manifest = terminal.transaction.artifact(
            "final_experiment_manifest.json"
        ).json_value()
        assert manifest["materialized_last"] is True
        assert manifest["excludes_only_self"] is True
        assert manifest["artifacts"] == [
            entry.as_dict() for entry in inventory[:-1]
        ]
        assert "final_experiment_manifest.json" not in {
            item["name"] for item in manifest["artifacts"]
        }

        retry = ledger.finalize_support_failure(
            specification,
            omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
        )
        assert retry.idempotent_recovery
        assert (
            retry.transaction.transaction_id
            == terminal.transaction.transaction_id
        )


def test_missing_head_cannot_be_misreported_as_support_failure() -> None:
    with pytest.raises(
        protocol.ProtocolSchemaError,
        match="registered support-failure reason",
    ):
        protocol.SupportFailureSpec(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P3",
            active_prefix="P4",
            support_stage="fit_monitor_chart_support",
            reason_code="missing_head_in_validation_rows",
        )


@pytest.mark.parametrize(
    "artifact_name",
    ["terminal_state.json", "final_report.md", "final_experiment_manifest.json"],
)
def test_terminal_tail_tamper_is_rejected_on_idempotent_recovery(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    ledger_path = tmp_path / f"terminal-tamper-{artifact_name}.sqlite3"
    specification = _zero_fit_support_spec()
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _complete_fit_capture(ledger, tmp_path / "capture")
        ledger.finalize_support_failure(
            specification,
            omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
        )

    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        original = bytes(
            connection.execute(
                "SELECT content FROM artifacts WHERE name = ?",
                (artifact_name,),
            ).fetchone()[0]
        )
        connection.execute(
            "UPDATE artifacts SET content = ? WHERE name = ?",
            (original + b" ", artifact_name),
        )

    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_support_failure(
                specification,
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        _assert_reason(caught, "artifact_hash_mismatch")


def test_terminal_retry_binds_exact_omission_reasons(
    tmp_path: Path,
) -> None:
    specification = _zero_fit_support_spec()
    with protocol.ProtocolLedger(tmp_path / "omission-binding.sqlite3") as ledger:
        _complete_fit_capture(ledger, tmp_path / "capture")
        ledger.finalize_support_failure(
            specification,
            omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
        )
        changed = dict(FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS)
        changed["o1_validation_results.json"] = "different_reason"
        with pytest.raises(protocol.ProtocolStateError):
            ledger.finalize_support_failure(
                specification,
                omitted_active_artifacts=changed,
            )


def test_mechanical_terminal_closure_keeps_primary_and_materializes_manifest_last(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "mechanical-terminal.sqlite3"
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _register_completed_prefixes(ledger)
        plan, _, attempt = _start_capture(ledger)
        ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="memory_resource_limit",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(),
            **_failure_context(),
        )
        terminal = ledger.finalize_campaign_terminal(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P2",
            active_prefix="P3",
            omitted_active_artifacts=FIT_P3_AFTER_CAPTURE_FAILURE_OMISSIONS,
        )
        assert terminal.terminal_state is (
            protocol.TerminalState.DEVELOPMENT_COMPLETE_RESOURCE_LIMITED
        )
        assert terminal.prefix_closure.valid
        assert {
            artifact.name for artifact in terminal.artifacts
        } == {
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        }
        assert ledger.get_artifact("resource_limit_failure.json") is not None
        assert ledger.get_artifact("invalid_campaign_receipt.json") is None
        assert ledger.get_artifact("support_failure_receipt.json") is None
        inventory = ledger.campaign_inventory()
        assert [entry.name for entry in inventory[-3:]] == [
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]
        manifest = terminal.transaction.artifact(
            "final_experiment_manifest.json"
        ).json_value()
        assert manifest["artifacts"] == [
            entry.as_dict() for entry in inventory[:-1]
        ]
        retry = ledger.finalize_campaign_terminal(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P2",
            active_prefix="P3",
            omitted_active_artifacts=FIT_P3_AFTER_CAPTURE_FAILURE_OMISSIONS,
        )
        assert retry.idempotent_recovery
        assert (
            retry.transaction.transaction_id
            == terminal.transaction.transaction_id
        )
        before = tuple(entry.as_dict() for entry in ledger.campaign_inventory())
        with pytest.raises(protocol.ProtocolStateError, match="globally terminal"):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P4",
                name="o1_validation_results.json",
                content=b'{"synthetic":true}',
            )
        assert tuple(
            entry.as_dict() for entry in ledger.campaign_inventory()
        ) == before


def test_deleted_chart_head_support_reason_is_unreachable() -> None:
    with pytest.raises(
        protocol.ProtocolSchemaError,
        match="zero_fit_monitor_chart_head_support is forbidden",
    ):
        protocol.classify_failure("zero_fit_monitor_chart_head_support")

    decisions = (
        protocol.classify_chart_head_support(
            [
                protocol.ChartHeadObservation(
                    "unit-a",
                    "chart-a",
                    None,
                    otherwise_valid=True,
                    row_id="00" * 32,
                )
            ],
            active_chart_ids=("chart-a",),
            expected_head_ids=("head-a",),
            expected_rows=(
                protocol.ExpectedChartHeadRow(
                    "00" * 32, "unit-a", "chart-a", "head-a"
                ),
            ),
        ),
        protocol.classify_chart_head_support(
            (
                protocol.ChartHeadObservation(
                    "unit-a",
                    "chart-a",
                    "head-a",
                    otherwise_valid=True,
                    row_id="00" * 32,
                ),
            ),
            active_chart_ids=("chart-a", "chart-b"),
            expected_head_ids=("head-a",),
            expected_rows=(
                protocol.ExpectedChartHeadRow(
                    "00" * 32, "unit-a", "chart-a", "head-a"
                ),
            ),
        ),
        protocol.classify_chart_head_support(
            [
                protocol.ChartHeadObservation(
                    "unit-a",
                    "chart-a",
                    "head-a",
                    otherwise_valid=True,
                    row_id="00" * 32,
                )
            ],
            active_chart_ids=("chart-a",),
            expected_head_ids=("head-a",),
            expected_rows=(
                protocol.ExpectedChartHeadRow(
                    "00" * 32, "unit-a", "chart-a", "head-a"
                ),
            ),
        ),
    )
    assert [decision.reason_code for decision in decisions] == [
        "missing_head_in_validation_rows",
        "zero_fit_monitor_chart_support",
        "insufficient_independent_chart_support",
    ]
    assert all(
        decision.reason_code != "zero_fit_monitor_chart_head_support"
        for decision in decisions
    )


@pytest.mark.parametrize(
    ("last_completed", "active"),
    [
        ("P99", "P3"),
        ("p2", "P3"),
        ("P5", "P3"),
        ("P3", "P3"),
        ("P2", "P5"),
        (None, "P1"),
    ],
)
def test_prefix_closure_rejects_unknown_noncanonical_or_nonadjacent_pairs(
    last_completed: str | None,
    active: str,
) -> None:
    assert tuple(prefix.name for prefix in protocol.CampaignPrefix) == tuple(
        f"P{index}" for index in range(13)
    )
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.validate_reached_prefix_closure(
            (),
            last_completed_prefix=last_completed,
            active_prefix=active,
            omitted_active_artifacts={},
        )


@pytest.mark.parametrize(
    ("last_completed", "active"),
    [
        ("P99", "P3"),
        ("p2", "P3"),
        ("P5", "P3"),
        ("P3", "P3"),
        ("P1", "P3"),
        ("P4", "P5"),
    ],
)
def test_row_index_failure_rejects_bad_prefix_before_any_artifact(
    tmp_path: Path,
    last_completed: str,
    active: str,
) -> None:
    payload = _row_index()
    with protocol.ProtocolLedger(tmp_path / "row-prefix.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            )
        )
        before = {
            artifact.name for artifact in ledger.list_artifacts("fit")
        }
        with pytest.raises((protocol.ProtocolSchemaError, protocol.ProtocolStateError)):
            ledger.record_row_index_failure(
                "fit",
                campaign_id=CAMPAIGN_ID,
                reason_code="unable_to_derive_expected_panel_rows",
                last_completed_prefix=last_completed,
                active_prefix=active,
                failure_stage="row_derivation",
            )
        assert ledger.snapshot("fit").state is protocol.CaptureState.AUTHORIZED
        assert {
            artifact.name for artifact in ledger.list_artifacts("fit")
        } == before


@pytest.mark.parametrize(
    ("last_completed", "active"),
    [
        ("P99", "P3"),
        ("p2", "P3"),
        ("P5", "P3"),
        ("P3", "P3"),
        ("P1", "P3"),
        ("P4", "P5"),
    ],
)
def test_post_start_failure_rejects_bad_prefix_before_any_artifact(
    tmp_path: Path,
    last_completed: str,
    active: str,
) -> None:
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    with protocol.ProtocolLedger(tmp_path / "capture-prefix.sqlite3") as ledger:
        plan, _, attempt = _start_capture(ledger)
        before = {
            artifact.name for artifact in ledger.list_artifacts("fit")
        }
        with pytest.raises((protocol.ProtocolSchemaError, protocol.ProtocolStateError)):
            ledger.finalize_capture_failure(
                "fit",
                campaign_id=CAMPAIGN_ID,
                capture_attempt_id=attempt,
                reason_code="memory_resource_limit",
                last_completed_prefix=last_completed,
                active_prefix=active,
                failure_stage="tensor_capture",
                plan=plan,
                capture_root=capture_root,
                finalized_partial_shard_indexes=(),
            )
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED
        assert {
            artifact.name for artifact in ledger.list_artifacts("fit")
        } == before


def test_authorization_requires_completed_predecessors_and_prefixes_never_regress(
    tmp_path: Path,
) -> None:
    payload = _row_index()
    with protocol.ProtocolLedger(tmp_path / "prefix-order.sqlite3") as ledger:
        with pytest.raises(protocol.ProtocolStateError, match="mandatory predecessor"):
            ledger.authorize(
                _authorization(
                    payload,
                    protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
                )
            )
        assert ledger.snapshot("fit") is None
        assert ledger.campaign_inventory() == ()

        _register_completed_prefixes(ledger)
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            )
        )
        before = ledger.campaign_inventory()
        with pytest.raises(protocol.ProtocolStateError, match="prefix regression"):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P0",
                name="late_protocol_note.json",
                content=b'{"synthetic":true}',
            )
        assert ledger.campaign_inventory() == before


@pytest.mark.parametrize(
    "reserved_name",
    [
        "fit_capture_transition_consumed_success.json",
        "capture_shard_manifest_fit.json",
        "capture_failure_fit.json",
        "fit_capture_success_transaction_commit.json",
        "terminal_state.json",
        "final_experiment_manifest.json",
    ],
)
def test_external_registration_rejects_every_ledger_owned_branch_name(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "reserved.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        with pytest.raises(protocol.ProtocolSchemaError, match="reserved"):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P3",
                name=reserved_name,
                content=b'{"synthetic":true}',
            )


@pytest.mark.parametrize(
    "alias",
    [
        "./invalid_campaign_receipt.json",
        "nested//artifact.json",
        "terminal_state.json/",
    ],
)
def test_artifact_paths_must_be_canonical_before_reserved_name_checks(
    tmp_path: Path,
    alias: str,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "path-alias.sqlite3") as ledger:
        with pytest.raises(protocol.ProtocolSchemaError, match="safe relative path"):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P0",
                name=alias,
                content=b'{"synthetic":true}',
            )


@pytest.mark.parametrize(
    "alias",
    [
        "bad\x00invalid_campaign_receipt.json",
        "bad\ninvalid_campaign_receipt.json",
        "INVALID_CAMPAIGN_RECEIPT.JSON",
        "Frozen_B_star.json",
    ],
)
def test_artifact_paths_reject_controls_and_casefold_reserved_aliases(
    tmp_path: Path,
    alias: str,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "path-fold.sqlite3") as ledger:
        with pytest.raises(protocol.ProtocolSchemaError):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P0",
                name=alias,
                content=b'{"synthetic":true}',
            )


def test_campaign_inventory_rejects_casefold_duplicate_names(
    tmp_path: Path,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "casefold.sqlite3") as ledger:
        ledger.register_reached_artifact(
            campaign_id=CAMPAIGN_ID,
            prefix="P0",
            name="custom_artifact.json",
            content=b'{"synthetic":true}',
        )
        with pytest.raises(protocol.ProtocolSchemaError, match="casefold"):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P0",
                name="CUSTOM_ARTIFACT.JSON",
                content=b'{"synthetic":true}',
            )


def test_external_byte_and_file_evidence_remain_durable_and_reopenable(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "external.sqlite3"
    source = tmp_path / "commands.md"
    source.write_bytes(b"# Frozen commands\n")
    with protocol.ProtocolLedger(ledger_path) as ledger:
        first = ledger.register_reached_artifact(
            campaign_id=CAMPAIGN_ID,
            prefix="P0",
            name="semantic_canonicalization_spec.json",
            content=b'{"synthetic":true}',
        )
        second = ledger.register_reached_artifact_file(
            campaign_id=CAMPAIGN_ID,
            prefix="P0",
            name="commands.md",
            path=source,
        )
        assert ledger.campaign_inventory() == (first, second)
        assert ledger.get_artifact(first.name) is None

    source.unlink()
    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.campaign_inventory()
        _assert_reason(caught, "artifact_hash_mismatch")

    source.write_bytes(b"# Frozen commands\n")
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.execute(
            "UPDATE campaign_artifact_inventory SET persisted_content = ? "
            "WHERE name = ?",
            (b'{"synthetic":false}', first.name),
        )
    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.campaign_inventory()
        _assert_reason(caught, "artifact_hash_mismatch")


def test_public_artifact_sanitizer_allows_commitments_but_rejects_raw_values(
    tmp_path: Path,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "public-fields.sqlite3") as ledger:
        accepted = ledger.register_reached_artifact(
            campaign_id=CAMPAIGN_ID,
            prefix="P0",
            name="safe_commitment.json",
            content=protocol.canonical_json_bytes(
                {
                    "activation_shape": [1, 2],
                    "message_sketch_sha256": "aa" * 32,
                    "prompt_count": 1,
                    "tokenization_protocol": "frozen_v1",
                }
            ),
        )
        assert accepted.name == "safe_commitment.json"
        for index, content in enumerate(
            (
                {"message": "protected"},
                {"prompt": "protected"},
                {"response": "protected"},
                {"completion": "protected"},
                {"token_ids": [1, 2]},
                {"activation_values": [1, 2]},
            )
        ):
            with pytest.raises(protocol.ProtocolSchemaError, match="protected field"):
                ledger.register_reached_artifact(
                    campaign_id=CAMPAIGN_ID,
                    prefix="P0",
                    name=f"raw-{index}.json",
                    content=protocol.canonical_json_bytes(content),
                )

def test_start_capture_rejects_preexisting_shards_and_binds_one_root(
    tmp_path: Path,
) -> None:
    payload = _row_index()
    with protocol.ProtocolLedger(tmp_path / "preexisting.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            )
        )
        ledger.commit_row_index(payload)
        plan = protocol.build_shard_plan(payload, 2)
        ledger.commit_shard_plan(plan)
        dirty_root = tmp_path / "dirty"
        _write_shards(dirty_root, plan)
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.start_capture(
                plan,
                dirty_root,
                capture_schema_hash=CAPTURE_SCHEMA_HASH,
                capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
            )
        _assert_reason(caught, "shard_plan_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.ROW_INDEX_COMMITTED

        clean_root = tmp_path / "clean"
        clean_root.mkdir()
        ledger.start_capture(
            plan,
            clean_root,
            capture_schema_hash=CAPTURE_SCHEMA_HASH,
            capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
        )
        other_root = tmp_path / "other"
        other_root.mkdir()
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.start_capture(
                plan,
                other_root,
                capture_schema_hash=CAPTURE_SCHEMA_HASH,
                capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
            )
        _assert_reason(caught, "capture_attempt_identity_mismatch")
        _write_shards(other_root, plan)
        other_verification = protocol.verify_capture_shard_files(
            plan, other_root
        )
        attempt = ledger.snapshot("fit").capture_attempt_id
        assert attempt is not None
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_success(
                plan,
                other_root,
                capture_attempt_id=attempt,
                expected_verification=other_verification,
            )
        _assert_reason(caught, "capture_root_identity_mismatch")


def test_resume_pins_existing_shard_bytes_against_later_rewrite(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "resume-pinning.sqlite3") as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        first_path = capture_root / plan.entries[0].path
        first_path.parent.mkdir(parents=True, exist_ok=True)
        first_path.write_bytes(_shard_bytes(plan.entries[0].row_ids))
        resumed = ledger.resume_capture(
            plan,
            capture_attempt_id=attempt,
            capture_root=capture_root,
            capture_schema_hash=CAPTURE_SCHEMA_HASH,
            capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
        )
        assert resumed.missing_shard_indexes == (1,)
        assert resumed.verified_existing_shards is not None
        old_hash = resumed.verified_existing_shards.shards[0].shard_byte_sha256

        mutated = bytearray(first_path.read_bytes())
        data_offset = 8 + struct.unpack_from("<Q", mutated, 0)[0]
        mutated[data_offset] ^= 1
        first_path.write_bytes(mutated)
        second_path = capture_root / plan.entries[1].path
        second_path.write_bytes(_shard_bytes(plan.entries[1].row_ids))
        fresh = protocol.verify_capture_shard_files(plan, capture_root)
        assert fresh.shards[0].shard_byte_sha256 != old_hash
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_capture_success(
                plan,
                capture_root,
                capture_attempt_id=attempt,
                expected_verification=fresh,
            )
        _assert_reason(caught, "shard_hash_mismatch")
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED


def test_prefix_closure_rejects_every_unknown_future_tagged_artifact() -> None:
    entries: list[Any] = []
    ordinal = 1
    for prefix in (protocol.CampaignPrefix.P0, protocol.CampaignPrefix.P1, protocol.CampaignPrefix.P2):
        for name in sorted(protocol._PREFIX_MANDATORY_ARTIFACTS[prefix]):
            entries.append(
                protocol.ArtifactInventoryEntry(
                    name=name,
                    artifact_byte_sha256="aa" * 32,
                    byte_count=1,
                    prefix=prefix,
                    materialization_ordinal=ordinal,
                )
            )
            ordinal += 1
    entries.append(
        protocol.ArtifactInventoryEntry(
            name="unknown_future_artifact.json",
            artifact_byte_sha256="bb" * 32,
            byte_count=1,
            prefix=protocol.CampaignPrefix.P12,
            materialization_ordinal=ordinal,
        )
    )
    omissions = {
        name: "not_reached"
        for name in protocol._PREFIX_MANDATORY_ARTIFACTS[protocol.CampaignPrefix.P3]
    }
    closure = protocol.validate_reached_prefix_closure(
        entries,
        last_completed_prefix="P2",
        active_prefix="P3",
        omitted_active_artifacts=omissions,
    )
    assert not closure.valid
    assert closure.forbidden_future_artifacts_present == frozenset(
        {"unknown_future_artifact.json"}
    )


def test_prefix_closure_independently_rejects_nonmonotonic_materialization() -> None:
    entries: list[Any] = [
        protocol.ArtifactInventoryEntry(
            name="fit_capture_authorization.json",
            artifact_byte_sha256="aa" * 32,
            byte_count=1,
            prefix=protocol.CampaignPrefix.P3,
            materialization_ordinal=1,
        )
    ]
    ordinal = 2
    for prefix in (
        protocol.CampaignPrefix.P0,
        protocol.CampaignPrefix.P1,
        protocol.CampaignPrefix.P2,
    ):
        for name in sorted(protocol._PREFIX_MANDATORY_ARTIFACTS[prefix]):
            entries.append(
                protocol.ArtifactInventoryEntry(
                    name=name,
                    artifact_byte_sha256="bb" * 32,
                    byte_count=1,
                    prefix=prefix,
                    materialization_ordinal=ordinal,
                )
            )
            ordinal += 1
    omissions = {
        name: "not_reached"
        for name in (
            protocol._PREFIX_MANDATORY_ARTIFACTS[protocol.CampaignPrefix.P3]
            - {"fit_capture_authorization.json"}
        )
    }
    closure = protocol.validate_reached_prefix_closure(
        entries,
        last_completed_prefix="P2",
        active_prefix="P3",
        omitted_active_artifacts=omissions,
    )
    assert not closure.valid
    assert closure.prefix_mismatches

    with pytest.raises(protocol.ProtocolSchemaError, match="contiguous"):
        protocol.validate_reached_prefix_closure(
            [
                protocol.ArtifactInventoryEntry(
                    name="semantic_canonicalization_spec.json",
                    artifact_byte_sha256="cc" * 32,
                    byte_count=1,
                    prefix=protocol.CampaignPrefix.P0,
                    materialization_ordinal=2,
                )
            ],
            last_completed_prefix=None,
            active_prefix="P0",
            omitted_active_artifacts={
                name: "not_reached"
                for name in (
                    protocol._PREFIX_MANDATORY_ARTIFACTS[protocol.CampaignPrefix.P0]
                    - {"semantic_canonicalization_spec.json"}
                )
            },
        )


def test_chart_head_classifier_requires_exact_frozen_row_unit_key_universe() -> None:
    expected = (
        protocol.ExpectedChartHeadRow("00" * 32, "unit-a", "chart-a", "head-a"),
        protocol.ExpectedChartHeadRow("00" * 32, "unit-a", "chart-a", "head-b"),
        protocol.ExpectedChartHeadRow("11" * 32, "unit-b", "chart-a", "head-a"),
        protocol.ExpectedChartHeadRow("11" * 32, "unit-b", "chart-a", "head-b"),
    )
    observations = tuple(
        protocol.ChartHeadObservation(
            item.analysis_unit_id,
            item.chart_id,
            item.head_id,
            row_id=item.row_id,
        )
        for item in expected
    )
    sufficient = protocol.classify_chart_head_support(
        observations,
        active_chart_ids=("chart-a",),
        expected_head_ids=("head-a", "head-b"),
        expected_rows=expected,
    )
    assert sufficient.status is protocol.SupportStatus.SUFFICIENT
    assert sufficient.qualifying_pair_count == 2

    missing = protocol.classify_chart_head_support(
        observations[:-1],
        active_chart_ids=("chart-a",),
        expected_head_ids=("head-a", "head-b"),
        expected_rows=expected,
    )
    assert missing.status is protocol.SupportStatus.INVALID_MEASUREMENT_CONTRACT
    assert missing.reason_code == "missing_head_in_validation_rows"

    with pytest.raises(protocol.IntegrityViolation) as duplicate:
        protocol.classify_chart_head_support(
            (*observations, observations[0]),
            active_chart_ids=("chart-a",),
            expected_head_ids=("head-a", "head-b"),
            expected_rows=expected,
        )
    _assert_reason(duplicate, "duplicate_row_id")

    swapped = list(observations)
    swapped[0] = protocol.ChartHeadObservation(
        "unit-b", "chart-a", "head-a", row_id="00" * 32
    )
    with pytest.raises(protocol.IntegrityViolation) as changed_unit:
        protocol.classify_chart_head_support(
            swapped,
            active_chart_ids=("chart-a",),
            expected_head_ids=("head-a", "head-b"),
            expected_rows=expected,
        )
    _assert_reason(changed_unit, "malformed_sequence_identity")

    incomplete_frozen = protocol.classify_chart_head_support(
        (),
        active_chart_ids=("chart-a",),
        expected_head_ids=("head-a", "head-b"),
        expected_rows=(expected[0],),
    )
    assert incomplete_frozen.status is protocol.SupportStatus.INVALID_MEASUREMENT_CONTRACT
    assert incomplete_frozen.reason_code == "missing_head_in_validation_rows"

    missing_observed_before_empty_chart = protocol.classify_chart_head_support(
        (),
        active_chart_ids=("chart-a", "chart-empty"),
        expected_head_ids=("head-a",),
        expected_rows=(expected[0],),
    )
    assert missing_observed_before_empty_chart.status is (
        protocol.SupportStatus.INVALID_MEASUREMENT_CONTRACT
    )
    assert (
        missing_observed_before_empty_chart.reason_code
        == "missing_head_in_validation_rows"
    )

    zero_chart = protocol.classify_chart_head_support(
        (
            protocol.ChartHeadObservation(
                "unit-a", "chart-a", "head-a", row_id="00" * 32
            ),
        ),
        active_chart_ids=("chart-a", "chart-empty"),
        expected_head_ids=("head-a",),
        expected_rows=(expected[0],),
    )
    assert zero_chart.status is protocol.SupportStatus.INSUFFICIENT_SUPPORT
    assert zero_chart.reason_code == "zero_fit_monitor_chart_support"

    with pytest.raises(protocol.IntegrityViolation) as extra_in_empty_chart:
        protocol.classify_chart_head_support(
            (
                protocol.ChartHeadObservation(
                    "unit-c", "chart-b", "head-a", row_id="22" * 32
                ),
            ),
            active_chart_ids=("chart-a", "chart-b"),
            expected_head_ids=("head-a",),
            expected_rows=(expected[0],),
        )
    _assert_reason(extra_in_empty_chart, "row_set_mismatch")


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "last_completed_prefix": "P2",
            "active_prefix": "P3",
            "chart_counts": {"fit": 0},
            "chart_head_validation_counts": {"fit": 0},
        },
        {
            "last_completed_prefix": "P3",
            "active_prefix": "P4",
            "chart_counts": {"fit": 1},
            "chart_head_validation_counts": {"fit": 0},
        },
    ],
)
def test_support_failure_reason_stage_and_evidence_are_cross_validated(
    kwargs: Mapping[str, object],
) -> None:
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.SupportFailureSpec(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            support_stage="fit_monitor_chart_support",
            reason_code="zero_fit_monitor_chart_support",
            active_chart_ids=("chart-a",),
            expected_head_ids=("head-a",),
            **kwargs,
        )


def test_zero_chart_support_receipt_requires_explicit_frozen_empty_chart() -> None:
    with pytest.raises(protocol.ProtocolSchemaError, match="frozen chart/head"):
        protocol.SupportFailureSpec(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P3",
            active_prefix="P4",
            support_stage="fit_monitor_chart_support",
            reason_code="zero_fit_monitor_chart_support",
        )
    specification = _zero_fit_support_spec()
    assert specification.as_dict()["active_chart_ids"] == [
        "chart_a",
        "chart_empty",
    ]
    assert specification.as_dict()["expected_head_ids"] == ["head_a"]


def test_support_terminal_cannot_strand_a_started_capture(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    specification = _zero_fit_support_spec()
    with protocol.ProtocolLedger(tmp_path / "live-support.sqlite3") as ledger:
        _start_capture(ledger, capture_root=capture_root)
        with pytest.raises(protocol.ProtocolStateError, match="live capture"):
            ledger.finalize_support_failure(
                specification,
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED
        assert ledger.get_artifact("support_failure_receipt.json") is None


@pytest.mark.parametrize(
    "details",
    [
        {"expected_hash": "banana"},
        {"row_id": 7},
        {"expected_count": "banana"},
        {"observed_shape": [2, True]},
    ],
)
def test_failure_detail_fields_have_exact_types(
    details: Mapping[str, object],
) -> None:
    with pytest.raises(protocol.ProtocolSchemaError):
        protocol.validate_sanitized_failure_details(details)


def test_failure_details_cannot_claim_a_different_partition_or_stage(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "failure-context.sqlite3") as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        with pytest.raises(protocol.ProtocolSchemaError, match="receipt context"):
            ledger.finalize_capture_failure(
                "fit",
                campaign_id=CAMPAIGN_ID,
                capture_attempt_id=attempt,
                reason_code="memory_resource_limit",
                last_completed_prefix="P2",
                active_prefix="P3",
                failure_stage="tensor_capture",
                plan=plan,
                capture_root=capture_root,
                finalized_partial_shard_indexes=(),
                sanitized_failure_details={
                    "partition_role": "tune",
                    "stage": "row_derivation",
                },
            )
        assert ledger.snapshot("fit").state is protocol.CaptureState.CAPTURE_STARTED


def test_post_capture_stage_failure_supports_p4_measurement_terminal(
    tmp_path: Path,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "stage-failure.sqlite3") as ledger:
        _complete_fit_capture(ledger, tmp_path / "capture")
        primary = ledger.record_stage_failure(
            "fit",
            campaign_id=CAMPAIGN_ID,
            reason_code="missing_head_in_validation_rows",
            last_completed_prefix="P3",
            active_prefix="P4",
            failure_stage="fit_monitor_chart_validation",
            sanitized_failure_details={
                "partition_role": "fit",
                "reason_code": "missing_head_in_validation_rows",
                "last_completed_prefix": "P3",
                "active_prefix": "P4",
                "stage": "fit_monitor_chart_validation",
            },
        )
        assert primary.transaction_kind == "post_capture_stage_failure"
        assert primary.artifact("invalid_campaign_receipt.json").json_value()[
            "terminal_state"
        ] == "invalid_campaign_measurement_contract"
        assert ledger.verify_artifact_closure("fit").valid
        terminal = ledger.finalize_campaign_terminal(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P3",
            active_prefix="P4",
            omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
        )
        assert terminal.terminal_state is (
            protocol.TerminalState.INVALID_CAMPAIGN_MEASUREMENT_CONTRACT
        )
        assert terminal.prefix_closure.valid


def test_post_capture_failure_can_close_inside_the_capture_prefix(
    tmp_path: Path,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "same-prefix-stage.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        _complete_partition_capture(ledger, "fit", tmp_path / "capture")
        primary = ledger.record_stage_failure(
            "fit",
            campaign_id=CAMPAIGN_ID,
            reason_code="protocol_dimension_mismatch",
            last_completed_prefix="P2",
            active_prefix="P3",
            failure_stage="capture_invariant_validation",
        )
        assert primary.transaction_kind == "post_capture_stage_failure"
        assert ledger.verify_artifact_closure("fit").valid
        terminal = ledger.finalize_campaign_terminal(
            campaign_id=CAMPAIGN_ID,
            partition_role="fit",
            last_completed_prefix="P2",
            active_prefix="P3",
            omitted_active_artifacts={
                "capture_invariants.json": "terminal_failure_branch"
            },
        )
        assert terminal.terminal_state is (
            protocol.TerminalState.INVALID_CAMPAIGN_INTEGRITY
        )
        assert terminal.prefix_closure.valid


def test_branch_closure_defense_rejects_synthetic_success_with_real_failure(
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(tmp_path / "branch-defense.sqlite3") as ledger:
        plan, _, attempt = _start_capture(
            ledger, capture_root=capture_root
        )
        ledger.finalize_capture_failure(
            "fit",
            capture_attempt_id=attempt,
            reason_code="memory_resource_limit",
            plan=plan,
            capture_root=capture_root,
            finalized_partial_shard_indexes=(),
            **_failure_context(),
        )
        inventory = ledger.campaign_inventory()
        synthetic = protocol.ArtifactInventoryEntry(
            name="fit_capture_transition_consumed_success.json",
            artifact_byte_sha256="cc" * 32,
            byte_count=1,
            prefix=protocol.CampaignPrefix.P3,
            materialization_ordinal=inventory[-1].materialization_ordinal + 1,
        )
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger._validate_campaign_branch_exclusivity(
                ledger._connection, (*inventory, synthetic)
            )
        _assert_reason(caught, "artifact_hash_mismatch")


def test_o2_artifacts_require_tune_success_and_are_forbidden_on_tune_failure(
    tmp_path: Path,
) -> None:
    tune_root = tmp_path / "tune-capture"
    tune_root.mkdir()
    with protocol.ProtocolLedger(tmp_path / "o2-branch.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        _complete_partition_capture(ledger, "fit", tmp_path / "fit-capture")
        _register_missing_mandatory_artifacts(ledger, protocol.CampaignPrefix.P3)
        _register_missing_mandatory_artifacts(ledger, protocol.CampaignPrefix.P4)

        payload = _row_index("tune")
        ledger.authorize(
            _authorization(
                payload,
                protocol.AuthorizationMode.DERIVE_AFTER_AUTHORIZATION,
            )
        )
        ledger.commit_row_index(payload)
        plan = protocol.build_shard_plan(payload, 2)
        ledger.commit_shard_plan(plan)
        ledger.start_capture(
            plan,
            tune_root,
            capture_schema_hash=CAPTURE_SCHEMA_HASH,
            capture_implementation_hash=CAPTURE_IMPLEMENTATION_HASH,
        )
        with pytest.raises(protocol.ProtocolStateError, match="successful tune"):
            ledger.register_reached_artifact(
                campaign_id=CAMPAIGN_ID,
                prefix="P5",
                name="o2q_results.json",
                content=b'{"synthetic":true}',
            )
        attempt = ledger.snapshot("tune").capture_attempt_id
        assert attempt is not None
        ledger.finalize_capture_failure(
            "tune",
            campaign_id=CAMPAIGN_ID,
            capture_attempt_id=attempt,
            reason_code="memory_resource_limit",
            last_completed_prefix="P4",
            active_prefix="P5",
            failure_stage="tensor_capture",
            plan=plan,
            capture_root=tune_root,
            finalized_partial_shard_indexes=(),
        )
        inventory = ledger.campaign_inventory()
        synthetic = protocol.ArtifactInventoryEntry(
            name="o2q_results.json",
            artifact_byte_sha256="dd" * 32,
            byte_count=1,
            prefix=protocol.CampaignPrefix.P5,
            materialization_ordinal=inventory[-1].materialization_ordinal + 1,
        )
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger._validate_campaign_branch_exclusivity(
                ledger._connection, (*inventory, synthetic)
            )
        _assert_reason(caught, "artifact_hash_mismatch")


def test_terminal_closure_revalidates_every_prior_transaction_receipt(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "prior-receipt.sqlite3"
    capture_root = tmp_path / "capture"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _complete_fit_capture(ledger, capture_root)

    receipt_name = "fit_authorization_transaction_commit.json"
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        content, transaction_id = connection.execute(
            "SELECT content, transaction_id FROM artifacts WHERE name = ?",
            (receipt_name,),
        ).fetchone()
        body = json.loads(bytes(content).decode("utf-8"))
        body["transaction_kind"] = "tampered_authorization"
        rewritten = protocol.canonical_json_bytes(body)
        digest = hashlib.sha256(rewritten).hexdigest()
        connection.execute(
            "UPDATE artifacts SET content = ?, byte_sha256 = ? WHERE name = ?",
            (rewritten, digest, receipt_name),
        )
        connection.execute(
            "UPDATE ledger_transactions SET commit_receipt = ?, "
            "commit_receipt_sha256 = ? WHERE transaction_id = ?",
            (rewritten, digest, transaction_id),
        )
        connection.execute(
            "UPDATE campaign_artifact_inventory SET byte_sha256 = ?, "
            "byte_count = ? WHERE name = ?",
            (digest, len(rewritten), receipt_name),
        )

    specification = _zero_fit_support_spec()
    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_support_failure(
                specification,
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        _assert_reason(caught, "artifact_hash_mismatch")
        assert ledger.get_artifact("final_experiment_manifest.json") is None


def test_terminal_closure_requires_every_ledger_artifact_in_inventory(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "missing-inventory-row.sqlite3"
    with protocol.ProtocolLedger(ledger_path) as ledger:
        _complete_fit_capture(ledger, tmp_path / "capture")
    with closing(sqlite3.connect(ledger_path)) as connection, connection:
        connection.execute(
            "DELETE FROM campaign_artifact_inventory WHERE name = ?",
            ("fit_capture_start_transaction_commit.json",),
        )
    specification = _zero_fit_support_spec()
    with protocol.ProtocolLedger(ledger_path) as ledger:
        with pytest.raises(protocol.IntegrityViolation) as caught:
            ledger.finalize_support_failure(
                specification,
                omitted_active_artifacts=FIT_P4_AFTER_FIT_CAPTURE_OMISSIONS,
            )
        _assert_reason(caught, "artifact_hash_mismatch")


def test_successful_p12_campaign_materializes_manifest_last_without_guard_claim(
    tmp_path: Path,
) -> None:
    with protocol.ProtocolLedger(tmp_path / "successful-p12.sqlite3") as ledger:
        _register_completed_prefixes(ledger)
        _complete_partition_capture(ledger, "fit", tmp_path / "fit-capture")
        _register_missing_mandatory_artifacts(ledger, protocol.CampaignPrefix.P3)
        _register_missing_mandatory_artifacts(ledger, protocol.CampaignPrefix.P4)

        _complete_partition_capture(ledger, "tune", tmp_path / "tune-capture")
        _register_missing_mandatory_artifacts(ledger, protocol.CampaignPrefix.P5)
        for prefix in (
            protocol.CampaignPrefix.P6,
            protocol.CampaignPrefix.P7,
            protocol.CampaignPrefix.P8,
            protocol.CampaignPrefix.P9,
        ):
            _register_missing_mandatory_artifacts(ledger, prefix)

        _complete_partition_capture(
            ledger,
            "development_confirm",
            tmp_path / "development-capture",
        )
        for prefix in (
            protocol.CampaignPrefix.P10,
            protocol.CampaignPrefix.P11,
            protocol.CampaignPrefix.P12,
        ):
            _register_missing_mandatory_artifacts(ledger, prefix)

        terminal = ledger.finalize_successful_campaign(campaign_id=CAMPAIGN_ID)
        assert terminal.terminal_state is (
            protocol.TerminalState.DEVELOPMENT_COMPLETE_NO_ELIGIBLE_GUARD
        )
        assert terminal.prefix_closure.valid
        inventory = ledger.campaign_inventory()
        assert [entry.name for entry in inventory[-3:]] == [
            "terminal_state.json",
            "final_report.md",
            "final_experiment_manifest.json",
        ]
        report = terminal.transaction.artifact("final_report.md").content
        assert b"not a guard claim" in report
        assert ledger.get_artifact("invalid_campaign_receipt.json") is None
        assert ledger.get_artifact("resource_limit_failure.json") is None
        assert ledger.get_artifact("support_failure_receipt.json") is None

        retry = ledger.finalize_successful_campaign(campaign_id=CAMPAIGN_ID)
        assert retry.idempotent_recovery
        assert retry.transaction.transaction_id == terminal.transaction.transaction_id
