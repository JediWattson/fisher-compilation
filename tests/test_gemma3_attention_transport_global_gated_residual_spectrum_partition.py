from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_attention_transport_global_gated_residual_spectrum_partition as partitioner,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _members() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "canonical_prompt_sha256": _sha(
                f"context-family-{family}-length-{target}"
            ),
            "family_id": f"family.{family}",
            "objective_row_stream_sha256": _sha(
                f"objective-family-{family}-length-{target}"
            ),
            "target_token_count": target,
        }
        for family in range(6)
        for target in partitioner.TARGET_TOKEN_COUNTS
    )


def _catalog() -> partitioner.StableABatchCatalog:
    return partitioner.build_stable_A_batch_catalog(
        role_panel_binding_sha256=_sha("A-panel"),
        ordered_members=_members(),
    )


def _records(*, receipt_prefix: str = "execution") -> tuple[object, ...]:
    return tuple(
        SimpleNamespace(
            role="A",
            base_capture_receipt_sha256=_sha(
                f"{receipt_prefix}-batch-{batch_ordinal}"
            ),
            capture=SimpleNamespace(
                invocation_id=invocation_id,
                row_count=partitioner.ROWS_PER_INVOCATION,
            ),
        )
        for batch_ordinal in range(partitioner.BATCH_COUNT)
        for invocation_id in reversed(partitioner.INVOCATION_IDS)
    )


def test_import_is_standard_library_only() -> None:
    code = """
import sys
import fisher_graph
baseline = set(sys.modules)
import fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum_partition
new_modules = set(sys.modules) - baseline
assert 'torch' not in new_modules
assert 'transformers' not in new_modules
assert 'mlx' not in new_modules
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_stable_catalog_uses_exact_45_batch_schedule_and_content_hashes() -> None:
    catalog = _catalog()
    assert len(catalog.batches) == 45
    assert tuple(batch.batch_ordinal for batch in catalog.batches) == tuple(
        range(45)
    )
    assert all(
        len(batch.ordered_context_fingerprint_sha256s) == 2
        and len(batch.ordered_objective_row_stream_sha256s) == 2
        for batch in catalog.batches[:3]
    )
    assert all(
        len(batch.ordered_context_fingerprint_sha256s) == 1
        and len(batch.ordered_objective_row_stream_sha256s) == 1
        for batch in catalog.batches[3:]
    )
    assert len(
        {batch.batch_identity_sha256 for batch in catalog.batches}
    ) == 45
    assert catalog == _catalog()

    changed_members = list(deepcopy(_members()))
    changed_members[0]["canonical_prompt_sha256"] = _sha("changed-context")
    changed = partitioner.build_stable_A_batch_catalog(
        role_panel_binding_sha256=catalog.role_panel_binding_sha256,
        ordered_members=changed_members,
    )
    assert changed.batch_plan_artifact_sha256 != catalog.batch_plan_artifact_sha256
    assert (
        changed.batches[0].batch_identity_sha256
        != catalog.batches[0].batch_identity_sha256
    )


def test_fixed_partitions_are_nested_balanced_and_exact() -> None:
    assert partitioner.OUTER_DEV_BATCH_ORDINALS == (
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    )
    assert partitioner.ALLOCATION_VALIDATION_BATCH_ORDINALS == (
        1,
        3,
        14,
        19,
        34,
        42,
    )
    assert len(partitioner.OUTER_FIT_BATCH_ORDINALS) == 36
    assert len(partitioner.OUTER_DEV_BATCH_ORDINALS) == 9
    assert len(partitioner.DIRECTION_TRAIN_BATCH_ORDINALS) == 30
    assert len(partitioner.ALLOCATION_VALIDATION_BATCH_ORDINALS) == 6
    assert set(partitioner.OUTER_FIT_BATCH_ORDINALS).isdisjoint(
        partitioner.OUTER_DEV_BATCH_ORDINALS
    )
    assert set(partitioner.DIRECTION_TRAIN_BATCH_ORDINALS).isdisjoint(
        partitioner.ALLOCATION_VALIDATION_BATCH_ORDINALS
    )
    assert set(partitioner.DIRECTION_TRAIN_BATCH_ORDINALS) | set(
        partitioner.ALLOCATION_VALIDATION_BATCH_ORDINALS
    ) == set(partitioner.OUTER_FIT_BATCH_ORDINALS)

    members = _members()
    by_length = {
        target: [
            member
            for member in members
            if member["target_token_count"] == target
        ]
        for target in partitioner.TARGET_TOKEN_COUNTS
    }
    plan = [
        tuple(by_length[96][index : index + 2])
        for index in range(0, 6, 2)
    ]
    plan.extend(
        (member,)
        for target in partitioner.TARGET_TOKEN_COUNTS[1:]
        for member in by_length[target]
    )

    def composition(ordinals: tuple[int, ...]) -> tuple[set[int], set[str]]:
        selected = [member for ordinal in ordinals for member in plan[ordinal]]
        return (
            {int(member["target_token_count"]) for member in selected},
            {str(member["family_id"]) for member in selected},
        )

    outer_lengths, outer_families = composition(
        partitioner.OUTER_DEV_BATCH_ORDINALS
    )
    inner_lengths, inner_families = composition(
        partitioner.ALLOCATION_VALIDATION_BATCH_ORDINALS
    )
    assert outer_lengths == set(partitioner.TARGET_TOKEN_COUNTS)
    assert outer_families == {f"family.{index}" for index in range(6)}
    assert inner_lengths == {96, 511, 512, 513, 640, 768}
    assert inner_families == {f"family.{index}" for index in range(6)}


def test_partition_binds_runtime_separately_from_stable_split() -> None:
    catalog = _catalog()
    first = partitioner.partition_stable_A_capture_records(
        _records(receipt_prefix="first"),
        catalog=catalog,
    )
    second = partitioner.partition_stable_A_capture_records(
        _records(receipt_prefix="second"),
        catalog=catalog,
    )
    assert first.outer_split_artifact_sha256 == second.outer_split_artifact_sha256
    assert first.inner_split_artifact_sha256 == second.inner_split_artifact_sha256
    assert first.catalog.batches == second.catalog.batches
    assert (
        first.outer_fit_batch_identity_sha256s
        == second.outer_fit_batch_identity_sha256s
    )
    assert (
        first.outer_dev_batch_identity_sha256s
        == second.outer_dev_batch_identity_sha256s
    )
    assert first.runtime_capture_binding_sha256 != second.runtime_capture_binding_sha256
    assert first.outer_split_artifact_sha256 == (
        partitioner.outer_split_artifact_sha256(catalog)
    )
    assert first.inner_split_artifact_sha256 == (
        partitioner.inner_split_artifact_sha256(catalog)
    )
    assert len(first.outer_fit_batches) == 36
    assert len(first.outer_dev_batches) == 9
    assert len(first.direction_train_batches) == 30
    assert len(first.allocation_validation_batches) == 6
    assert first.outer_fit_batch_ordinals == (
        partitioner.OUTER_FIT_BATCH_ORDINALS
    )
    assert first.outer_dev_batch_ordinals == (
        partitioner.OUTER_DEV_BATCH_ORDINALS
    )
    assert first.direction_train_batch_ordinals == (
        partitioner.DIRECTION_TRAIN_BATCH_ORDINALS
    )
    assert first.allocation_validation_batch_ordinals == (
        partitioner.ALLOCATION_VALIDATION_BATCH_ORDINALS
    )
    assert len(first.outer_fit_records) == 36 * 18
    assert len(first.outer_dev_records) == 9 * 18
    assert len(first.direction_train_records) == 30 * 18
    assert len(first.allocation_validation_records) == 6 * 18
    assert sum(record.capture.row_count for record in first.outer_fit_records) == (
        36 * partitioner.ROWS_PER_BATCH
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_record",
        "duplicate_invocation",
        "wrong_row_count",
        "mixed_receipt",
        "reused_receipt",
        "wrong_role",
    ),
)
def test_partition_rejects_capture_chunk_drift(mutation: str) -> None:
    records = list(_records())
    if mutation == "missing_record":
        records.pop()
    elif mutation == "duplicate_invocation":
        records[1].capture.invocation_id = records[0].capture.invocation_id
    elif mutation == "wrong_row_count":
        records[0].capture.row_count = 15
    elif mutation == "mixed_receipt":
        records[1].base_capture_receipt_sha256 = _sha("mixed")
    elif mutation == "reused_receipt":
        records[18].base_capture_receipt_sha256 = (
            records[0].base_capture_receipt_sha256
        )
        for index in range(19, 36):
            records[index].base_capture_receipt_sha256 = (
                records[0].base_capture_receipt_sha256
            )
    elif mutation == "wrong_role":
        records[0].role = "B"
    with pytest.raises(ValueError):
        partitioner.partition_stable_A_capture_records(
            records,
            catalog=_catalog(),
        )


def test_public_bound_batch_value_object_fails_closed() -> None:
    catalog = _catalog()
    records = _records()[:18]
    with pytest.raises(ValueError):
        partitioner.StableABoundCaptureBatch(
            identity=catalog.batches[0],
            execution_receipt_sha256=records[0].base_capture_receipt_sha256,
            records=records[:-1],
        )
    with pytest.raises(ValueError):
        partitioner.StableABoundCaptureBatch(
            identity=catalog.batches[0],
            execution_receipt_sha256=_sha("wrong-runtime-receipt"),
            records=records,
        )


@pytest.mark.parametrize(
    "mutation",
    ("wrong_count", "bad_length", "bad_context_sha", "unbalanced_length"),
)
def test_catalog_rejects_manifest_schedule_drift(mutation: str) -> None:
    members = list(deepcopy(_members()))
    if mutation == "wrong_count":
        members.pop()
    elif mutation == "bad_length":
        members[0]["target_token_count"] = 97
    elif mutation == "bad_context_sha":
        members[0]["canonical_prompt_sha256"] = "not-a-sha"
    elif mutation == "unbalanced_length":
        members[0]["target_token_count"] = 511
    with pytest.raises(ValueError):
        partitioner.build_stable_A_batch_catalog(
            role_panel_binding_sha256=_sha("A-panel"),
            ordered_members=members,
        )
