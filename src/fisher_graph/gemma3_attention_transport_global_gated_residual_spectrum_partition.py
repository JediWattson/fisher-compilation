"""Stable A-batch identity and nested partition for the global-gated rung.

The runtime execution receipts authenticate a particular execution, but they
are intentionally unsuitable as cross-run sample identities.  This module
derives 45 stable batch identities solely from the sealed A panel, its fixed
capture schedule, and prompt-free manifest hashes.  It then binds a particular
runtime capture separately, without allowing execution receipts to influence
either split.

Importing this module is standard-library-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json


A_ROLE = "A_chart_transport_fit"
INVOCATION_IDS = tuple(f"mutation.{2 * index + 1}" for index in range(18))
TARGET_TOKEN_COUNTS = (96, 511, 512, 513, 514, 576, 640, 768)
BATCH_COUNT = 45
INVOCATIONS_PER_BATCH = 18
ROWS_PER_INVOCATION = 16
ROWS_PER_BATCH = INVOCATIONS_PER_BATCH * ROWS_PER_INVOCATION

OUTER_DEV_BATCH_ORDINALS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
OUTER_FIT_BATCH_ORDINALS = tuple(
    ordinal
    for ordinal in range(BATCH_COUNT)
    if ordinal not in frozenset(OUTER_DEV_BATCH_ORDINALS)
)
ALLOCATION_VALIDATION_BATCH_ORDINALS = (1, 3, 14, 19, 34, 42)
DIRECTION_TRAIN_BATCH_ORDINALS = tuple(
    ordinal
    for ordinal in OUTER_FIT_BATCH_ORDINALS
    if ordinal not in frozenset(ALLOCATION_VALIDATION_BATCH_ORDINALS)
)

_BATCH_PLAN_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-a-batch-plan:v1\0"
)
_BATCH_IDENTITY_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-a-batch-identity:v1\0"
)
_OUTER_SPLIT_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-a-outer-split:v1\0"
)
_INNER_SPLIT_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-a-inner-split:v1\0"
)
_RUNTIME_CAPTURE_BINDING_DOMAIN = (
    b"fisher_graph:global-gated-residual-spectrum-a-runtime-capture-binding:v1\0"
)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _batch_payload(
    *,
    batch_ordinal: int,
    ordered_context_fingerprint_sha256s: Sequence[str],
    ordered_objective_row_stream_sha256s: Sequence[str],
) -> dict[str, object]:
    return {
        "batch_ordinal": batch_ordinal,
        "ordered_context_fingerprint_sha256s": list(
            ordered_context_fingerprint_sha256s
        ),
        "ordered_objective_row_stream_sha256s": list(
            ordered_objective_row_stream_sha256s
        ),
    }


@dataclass(frozen=True, slots=True)
class StableABatchIdentity:
    batch_ordinal: int
    batch_identity_sha256: str
    ordered_context_fingerprint_sha256s: tuple[str, ...]
    ordered_objective_row_stream_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.batch_ordinal) is not int
            or not 0 <= self.batch_ordinal < BATCH_COUNT
        ):
            raise ValueError("stable A batch ordinal is invalid")
        _require_sha256(
            self.batch_identity_sha256,
            label="stable A batch identity",
        )
        contexts = self.ordered_context_fingerprint_sha256s
        objectives = self.ordered_objective_row_stream_sha256s
        expected_examples = 2 if self.batch_ordinal < 3 else 1
        if (
            type(contexts) is not tuple
            or type(objectives) is not tuple
            or len(contexts) != expected_examples
            or len(objectives) != expected_examples
            or any(not _is_sha256(value) for value in (*contexts, *objectives))
        ):
            raise ValueError("stable A batch manifest identity drifted")

    def payload_without_identity(self) -> dict[str, object]:
        return _batch_payload(
            batch_ordinal=self.batch_ordinal,
            ordered_context_fingerprint_sha256s=(
                self.ordered_context_fingerprint_sha256s
            ),
            ordered_objective_row_stream_sha256s=(
                self.ordered_objective_row_stream_sha256s
            ),
        )


def _batch_plan_artifact_sha256(
    role_panel_binding_sha256: str,
    batches: Sequence[StableABatchIdentity],
) -> str:
    return _digest(
        _BATCH_PLAN_DOMAIN,
        {
            "batch_count": BATCH_COUNT,
            "batches": [batch.payload_without_identity() for batch in batches],
            "role": A_ROLE,
            "role_panel_binding_sha256": role_panel_binding_sha256,
            "schedule_policy": "three_short_pairs_then_42_long_singletons_v1",
        },
    )


def _batch_identity_sha256(
    *,
    batch_plan_artifact_sha256: str,
    batch_payload: Mapping[str, object],
) -> str:
    return _digest(
        _BATCH_IDENTITY_DOMAIN,
        {
            "batch": dict(batch_payload),
            "batch_plan_artifact_sha256": batch_plan_artifact_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class StableABatchCatalog:
    role_panel_binding_sha256: str
    batch_plan_artifact_sha256: str
    batches: tuple[StableABatchIdentity, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.role_panel_binding_sha256,
            label="A role panel binding",
        )
        _require_sha256(
            self.batch_plan_artifact_sha256,
            label="stable A batch plan artifact",
        )
        if (
            type(self.batches) is not tuple
            or len(self.batches) != BATCH_COUNT
            or tuple(batch.batch_ordinal for batch in self.batches)
            != tuple(range(BATCH_COUNT))
        ):
            raise ValueError("stable A batch catalog inventory drifted")
        expected_plan = _batch_plan_artifact_sha256(
            self.role_panel_binding_sha256,
            self.batches,
        )
        if self.batch_plan_artifact_sha256 != expected_plan:
            raise ValueError("stable A batch plan digest drifted")
        for batch in self.batches:
            expected_identity = _batch_identity_sha256(
                batch_plan_artifact_sha256=expected_plan,
                batch_payload=batch.payload_without_identity(),
            )
            if batch.batch_identity_sha256 != expected_identity:
                raise ValueError("stable A batch identity digest drifted")


def build_stable_A_batch_catalog(
    *,
    role_panel_binding_sha256: str,
    ordered_members: Sequence[Mapping[str, object]],
) -> StableABatchCatalog:
    """Build the stable, prompt-free identity catalog for the 45 A batches."""

    panel_sha = _require_sha256(
        role_panel_binding_sha256,
        label="A role panel binding",
    )
    members = tuple(ordered_members)
    if len(members) != 48 or any(
        not isinstance(member, Mapping) for member in members
    ):
        raise ValueError("stable A identity requires exactly 48 manifest members")
    by_length: dict[int, list[Mapping[str, object]]] = {
        length: [] for length in TARGET_TOKEN_COUNTS
    }
    for member in members:
        target = member.get("target_token_count")
        if type(target) is not int or target not in by_length:
            raise ValueError("stable A member token-count schedule drifted")
        _require_sha256(
            member.get("canonical_prompt_sha256"),
            label="A context fingerprint",
        )
        _require_sha256(
            member.get("objective_row_stream_sha256"),
            label="A objective row stream",
        )
        by_length[target].append(member)
    if any(len(group) != 6 for group in by_length.values()):
        raise ValueError("stable A length buckets must be six-way balanced")

    plan: list[tuple[Mapping[str, object], ...]] = []
    short = by_length[96]
    plan.extend(tuple(short[index : index + 2]) for index in range(0, 6, 2))
    for target in TARGET_TOKEN_COUNTS[1:]:
        plan.extend((member,) for member in by_length[target])
    if len(plan) != BATCH_COUNT:
        raise RuntimeError("stable A batch plan did not produce 45 batches")

    payloads = tuple(
        _batch_payload(
            batch_ordinal=ordinal,
            ordered_context_fingerprint_sha256s=tuple(
                str(member["canonical_prompt_sha256"]) for member in batch
            ),
            ordered_objective_row_stream_sha256s=tuple(
                str(member["objective_row_stream_sha256"]) for member in batch
            ),
        )
        for ordinal, batch in enumerate(plan)
    )
    provisional = tuple(
        StableABatchIdentity(
            batch_ordinal=ordinal,
            batch_identity_sha256="0" * 64,
            ordered_context_fingerprint_sha256s=tuple(
                payload["ordered_context_fingerprint_sha256s"]
            ),
            ordered_objective_row_stream_sha256s=tuple(
                payload["ordered_objective_row_stream_sha256s"]
            ),
        )
        for ordinal, payload in enumerate(payloads)
    )
    plan_sha = _batch_plan_artifact_sha256(panel_sha, provisional)
    batches = tuple(
        StableABatchIdentity(
            batch_ordinal=ordinal,
            batch_identity_sha256=_batch_identity_sha256(
                batch_plan_artifact_sha256=plan_sha,
                batch_payload=payload,
            ),
            ordered_context_fingerprint_sha256s=tuple(
                payload["ordered_context_fingerprint_sha256s"]
            ),
            ordered_objective_row_stream_sha256s=tuple(
                payload["ordered_objective_row_stream_sha256s"]
            ),
        )
        for ordinal, payload in enumerate(payloads)
    )
    return StableABatchCatalog(
        role_panel_binding_sha256=panel_sha,
        batch_plan_artifact_sha256=plan_sha,
        batches=batches,
    )


def _split_artifact_sha256(
    *,
    domain: bytes,
    catalog: StableABatchCatalog,
    first_label: str,
    first_ordinals: Sequence[int],
    second_label: str,
    second_ordinals: Sequence[int],
) -> str:
    by_ordinal = {batch.batch_ordinal: batch for batch in catalog.batches}

    def materialize(ordinals: Sequence[int]) -> list[dict[str, object]]:
        return [
            {
                "batch_identity_sha256": by_ordinal[ordinal].batch_identity_sha256,
                "batch_ordinal": ordinal,
            }
            for ordinal in ordinals
        ]

    return _digest(
        domain,
        {
            "batch_plan_artifact_sha256": catalog.batch_plan_artifact_sha256,
            first_label: materialize(first_ordinals),
            second_label: materialize(second_ordinals),
        },
    )


def outer_split_artifact_sha256(catalog: StableABatchCatalog) -> str:
    return _split_artifact_sha256(
        domain=_OUTER_SPLIT_DOMAIN,
        catalog=catalog,
        first_label="outer_fit_batches",
        first_ordinals=OUTER_FIT_BATCH_ORDINALS,
        second_label="outer_dev_batches",
        second_ordinals=OUTER_DEV_BATCH_ORDINALS,
    )


def inner_split_artifact_sha256(catalog: StableABatchCatalog) -> str:
    return _split_artifact_sha256(
        domain=_INNER_SPLIT_DOMAIN,
        catalog=catalog,
        first_label="direction_train_batches",
        first_ordinals=DIRECTION_TRAIN_BATCH_ORDINALS,
        second_label="allocation_validation_batches",
        second_ordinals=ALLOCATION_VALIDATION_BATCH_ORDINALS,
    )


@dataclass(frozen=True, slots=True)
class StableABoundCaptureBatch:
    identity: StableABatchIdentity
    execution_receipt_sha256: str
    records: tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, StableABatchIdentity):
            raise TypeError("stable A bound batch identity is invalid")
        receipt = _require_sha256(
            self.execution_receipt_sha256,
            label="A runtime execution receipt",
        )
        records = self.records
        if type(records) is not tuple or len(records) != INVOCATIONS_PER_BATCH:
            raise ValueError("stable A bound batch needs exactly 18 records")
        invocation_ids = tuple(
            str(getattr(getattr(record, "capture", None), "invocation_id", ""))
            for record in records
        )
        if (
            len(set(invocation_ids)) != INVOCATIONS_PER_BATCH
            or frozenset(invocation_ids) != frozenset(INVOCATION_IDS)
            or any(getattr(record, "role", None) != "A" for record in records)
            or any(
                getattr(getattr(record, "capture", None), "row_count", None)
                != ROWS_PER_INVOCATION
                for record in records
            )
            or any(
                getattr(record, "base_capture_receipt_sha256", None) != receipt
                for record in records
            )
        ):
            raise ValueError("stable A bound batch record inventory drifted")


def _runtime_capture_binding_sha256(
    *,
    catalog: StableABatchCatalog,
    batches: Sequence[StableABoundCaptureBatch],
) -> str:
    return _digest(
        _RUNTIME_CAPTURE_BINDING_DOMAIN,
        {
            "batch_plan_artifact_sha256": catalog.batch_plan_artifact_sha256,
            "batches": [
                {
                    "batch_identity_sha256": batch.identity.batch_identity_sha256,
                    "batch_ordinal": batch.identity.batch_ordinal,
                    "execution_receipt_sha256": batch.execution_receipt_sha256,
                }
                for batch in batches
            ],
        },
    )


@dataclass(frozen=True, slots=True)
class StableAPartition:
    catalog: StableABatchCatalog
    capture_batches: tuple[StableABoundCaptureBatch, ...]
    runtime_capture_binding_sha256: str
    outer_split_artifact_sha256: str
    inner_split_artifact_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.catalog, StableABatchCatalog)
            or type(self.capture_batches) is not tuple
            or len(self.capture_batches) != BATCH_COUNT
            or tuple(batch.identity for batch in self.capture_batches)
            != self.catalog.batches
            or len(
                {
                    batch.execution_receipt_sha256
                    for batch in self.capture_batches
                }
            )
            != BATCH_COUNT
        ):
            raise ValueError("stable A bound capture inventory drifted")
        if self.runtime_capture_binding_sha256 != _runtime_capture_binding_sha256(
            catalog=self.catalog,
            batches=self.capture_batches,
        ):
            raise ValueError("stable A runtime capture binding drifted")
        if self.outer_split_artifact_sha256 != outer_split_artifact_sha256(
            self.catalog
        ):
            raise ValueError("stable A outer split digest drifted")
        if self.inner_split_artifact_sha256 != inner_split_artifact_sha256(
            self.catalog
        ):
            raise ValueError("stable A inner split digest drifted")

    def _selected_batches(
        self,
        ordinals: Sequence[int],
    ) -> tuple[StableABoundCaptureBatch, ...]:
        by_ordinal = {
            batch.identity.batch_ordinal: batch for batch in self.capture_batches
        }
        return tuple(by_ordinal[ordinal] for ordinal in ordinals)

    @staticmethod
    def _flatten(
        batches: Sequence[StableABoundCaptureBatch],
    ) -> tuple[object, ...]:
        return tuple(record for batch in batches for record in batch.records)

    @staticmethod
    def _identities(
        batches: Sequence[StableABoundCaptureBatch],
    ) -> tuple[str, ...]:
        return tuple(batch.identity.batch_identity_sha256 for batch in batches)

    @property
    def outer_fit_batches(self) -> tuple[StableABoundCaptureBatch, ...]:
        return self._selected_batches(OUTER_FIT_BATCH_ORDINALS)

    @property
    def outer_dev_batches(self) -> tuple[StableABoundCaptureBatch, ...]:
        return self._selected_batches(OUTER_DEV_BATCH_ORDINALS)

    @property
    def direction_train_batches(self) -> tuple[StableABoundCaptureBatch, ...]:
        return self._selected_batches(DIRECTION_TRAIN_BATCH_ORDINALS)

    @property
    def allocation_validation_batches(
        self,
    ) -> tuple[StableABoundCaptureBatch, ...]:
        return self._selected_batches(ALLOCATION_VALIDATION_BATCH_ORDINALS)

    @property
    def outer_fit_batch_ordinals(self) -> tuple[int, ...]:
        return tuple(batch.identity.batch_ordinal for batch in self.outer_fit_batches)

    @property
    def outer_dev_batch_ordinals(self) -> tuple[int, ...]:
        return tuple(batch.identity.batch_ordinal for batch in self.outer_dev_batches)

    @property
    def direction_train_batch_ordinals(self) -> tuple[int, ...]:
        return tuple(
            batch.identity.batch_ordinal for batch in self.direction_train_batches
        )

    @property
    def allocation_validation_batch_ordinals(self) -> tuple[int, ...]:
        return tuple(
            batch.identity.batch_ordinal
            for batch in self.allocation_validation_batches
        )

    @property
    def outer_fit_records(self) -> tuple[object, ...]:
        return self._flatten(self.outer_fit_batches)

    @property
    def outer_dev_records(self) -> tuple[object, ...]:
        return self._flatten(self.outer_dev_batches)

    @property
    def direction_train_records(self) -> tuple[object, ...]:
        return self._flatten(self.direction_train_batches)

    @property
    def allocation_validation_records(self) -> tuple[object, ...]:
        return self._flatten(self.allocation_validation_batches)

    @property
    def outer_fit_batch_identity_sha256s(self) -> tuple[str, ...]:
        return self._identities(self.outer_fit_batches)

    @property
    def outer_dev_batch_identity_sha256s(self) -> tuple[str, ...]:
        return self._identities(self.outer_dev_batches)

    @property
    def direction_train_batch_identity_sha256s(self) -> tuple[str, ...]:
        return self._identities(self.direction_train_batches)

    @property
    def allocation_validation_batch_identity_sha256s(self) -> tuple[str, ...]:
        return self._identities(self.allocation_validation_batches)


def partition_stable_A_capture_records(
    records: Sequence[object],
    *,
    catalog: StableABatchCatalog,
) -> StableAPartition:
    """Authenticate capture order, bind runtime receipts, and apply fixed splits."""

    captured = tuple(records)
    expected_record_count = BATCH_COUNT * INVOCATIONS_PER_BATCH
    if len(captured) != expected_record_count:
        raise ValueError("stable A capture must contain exactly 45x18 records")
    expected_invocations = frozenset(INVOCATION_IDS)
    observed_receipts: set[str] = set()
    bound: list[StableABoundCaptureBatch] = []
    for ordinal, identity in enumerate(catalog.batches):
        start = ordinal * INVOCATIONS_PER_BATCH
        chunk = captured[start : start + INVOCATIONS_PER_BATCH]
        invocation_ids = tuple(
            str(getattr(getattr(record, "capture", None), "invocation_id", ""))
            for record in chunk
        )
        receipts = {
            getattr(record, "base_capture_receipt_sha256", None)
            for record in chunk
        }
        if (
            len(invocation_ids) != INVOCATIONS_PER_BATCH
            or len(set(invocation_ids)) != INVOCATIONS_PER_BATCH
            or frozenset(invocation_ids) != expected_invocations
            or any(getattr(record, "role", None) != "A" for record in chunk)
            or any(
                getattr(getattr(record, "capture", None), "row_count", None)
                != ROWS_PER_INVOCATION
                for record in chunk
            )
            or len(receipts) != 1
        ):
            raise ValueError("stable A contiguous capture chunk drifted")
        receipt = next(iter(receipts))
        receipt_sha = _require_sha256(
            receipt,
            label="A runtime execution receipt",
        )
        if receipt_sha in observed_receipts:
            raise ValueError("stable A execution receipt was reused across batches")
        observed_receipts.add(receipt_sha)
        bound.append(
            StableABoundCaptureBatch(
                identity=identity,
                execution_receipt_sha256=receipt_sha,
                records=chunk,
            )
        )
    batches = tuple(bound)
    return StableAPartition(
        catalog=catalog,
        capture_batches=batches,
        runtime_capture_binding_sha256=_runtime_capture_binding_sha256(
            catalog=catalog,
            batches=batches,
        ),
        outer_split_artifact_sha256=outer_split_artifact_sha256(catalog),
        inner_split_artifact_sha256=inner_split_artifact_sha256(catalog),
    )


__all__ = [
    "ALLOCATION_VALIDATION_BATCH_ORDINALS",
    "BATCH_COUNT",
    "DIRECTION_TRAIN_BATCH_ORDINALS",
    "INVOCATIONS_PER_BATCH",
    "INVOCATION_IDS",
    "OUTER_DEV_BATCH_ORDINALS",
    "OUTER_FIT_BATCH_ORDINALS",
    "ROWS_PER_BATCH",
    "ROWS_PER_INVOCATION",
    "StableABatchCatalog",
    "StableABatchIdentity",
    "StableABoundCaptureBatch",
    "StableAPartition",
    "build_stable_A_batch_catalog",
    "inner_split_artifact_sha256",
    "outer_split_artifact_sha256",
    "partition_stable_A_capture_records",
]
