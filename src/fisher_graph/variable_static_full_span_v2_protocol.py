"""Frozen data protocol for the second static full-span experiment.

The V2 source task expands the associative vocabulary from eight keys and
values to ten.  A context is eligible for V2 only when its canonical semantic
hash did not occur anywhere in the original 8-by-8 dataset.  Whole contexts
are then assigned to development and calibration roles by a declared salted
SHA-256 rank; the fresh validation and test partitions remain outside that
allocation.

This module only defines data provenance and novelty metrics.  It does not
train a source model, fit an executor, or open an evaluation split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import torch
from torch import Tensor

from .variable_associative import (
    VariableAssociativeRecallSplit,
    VariableAssociativeRecallTaskConfig,
    build_variable_associative_recall_splits,
    subset_variable_associative_recall_split,
)


V2_BASELINE_TASK_CONFIG = VariableAssociativeRecallTaskConfig(
    n_keys=8,
    n_values=8,
    split_seed=26_071,
)
V2_TASK_CONFIG = VariableAssociativeRecallTaskConfig(
    n_keys=10,
    n_values=10,
    split_seed=26_071,
)
V2_ROLE_SALT = "fisher_graph.variable_static_full_span.roles.v2.n10"
V2_ROLE_SIZES: dict[str, int] = {
    "basis_fit_a": 128,
    "graph_fit_a": 1_024,
    "graph_stop_a": 192,
    "graph_select_a": 192,
    "calibration_b": 256,
}
V2_ROLE_NAMES = tuple(V2_ROLE_SIZES)
V2_NEW_KEY_START = V2_BASELINE_TASK_CONFIG.n_keys
V2_NEW_VALUE_START = V2_BASELINE_TASK_CONFIG.n_values

# Preserve the original allocator's domain and change only the declared salt.
# This makes the V2 ordering independently reproducible with the established
# whole-context ranking rule.
_ROLE_HASH_DOMAIN = b"fisher_graph.variable_static_full_span.role.v1\0"
_MANIFEST_SCHEMA = "fisher_graph.variable_static_full_span_v2_protocol"
_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class VariableStaticFullSpanV2ProtocolAudit:
    """Exact context counts and fail-closed overlap checks."""

    baseline_contexts: int
    source_contexts: int
    source_train_contexts: int
    source_validation_contexts: int
    source_test_contexts: int
    fresh_train_contexts: int
    fresh_validation_contexts: int
    fresh_test_contexts: int
    excluded_train_contexts: int
    excluded_validation_contexts: int
    excluded_test_contexts: int
    allocated_role_contexts: int
    reserve_contexts: int
    role_context_counts: tuple[tuple[str, int], ...]
    role_pairwise_overlap: bool
    roles_baseline_overlap: bool
    roles_validation_overlap: bool
    roles_test_overlap: bool
    reserve_role_overlap: bool
    reserve_baseline_overlap: bool
    reserve_validation_overlap: bool
    reserve_test_overlap: bool
    fresh_validation_baseline_overlap: bool
    fresh_test_baseline_overlap: bool
    fresh_validation_test_overlap: bool

    @property
    def all_overlap_checks_pass(self) -> bool:
        return not any(
            (
                self.role_pairwise_overlap,
                self.roles_baseline_overlap,
                self.roles_validation_overlap,
                self.roles_test_overlap,
                self.reserve_role_overlap,
                self.reserve_baseline_overlap,
                self.reserve_validation_overlap,
                self.reserve_test_overlap,
                self.fresh_validation_baseline_overlap,
                self.fresh_test_baseline_overlap,
                self.fresh_validation_test_overlap,
            )
        )


@dataclass(frozen=True, slots=True)
class VariableStaticFullSpanV2Protocol:
    """Materialized fresh-context roles and their provenance receipt."""

    task_config: VariableAssociativeRecallTaskConfig
    baseline_task_config: VariableAssociativeRecallTaskConfig
    source_dataset_sha256: str
    baseline_dataset_sha256: str
    baseline_context_hashes: tuple[str, ...]
    fresh_train: VariableAssociativeRecallSplit
    basis_fit_a: VariableAssociativeRecallSplit
    graph_fit_a: VariableAssociativeRecallSplit
    graph_stop_a: VariableAssociativeRecallSplit
    graph_select_a: VariableAssociativeRecallSplit
    calibration_b: VariableAssociativeRecallSplit
    reserve: VariableAssociativeRecallSplit
    fresh_validation: VariableAssociativeRecallSplit
    fresh_test: VariableAssociativeRecallSplit
    audit: VariableStaticFullSpanV2ProtocolAudit

    @property
    def roles(self) -> dict[str, VariableAssociativeRecallSplit]:
        return {
            "basis_fit_a": self.basis_fit_a,
            "graph_fit_a": self.graph_fit_a,
            "graph_stop_a": self.graph_stop_a,
            "graph_select_a": self.graph_select_a,
            "calibration_b": self.calibration_b,
        }

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible, hash-complete protocol manifest."""

        context_set_sha256 = {
            "excluded_baseline": _context_set_sha256(
                self.baseline_context_hashes
            ),
            **{
                name: _context_set_sha256(split.semantic_context_hashes)
                for name, split in self.roles.items()
            },
            "reserve": _context_set_sha256(
                self.reserve.semantic_context_hashes
            ),
            "fresh_validation": _context_set_sha256(
                self.fresh_validation.semantic_context_hashes
            ),
            "fresh_test": _context_set_sha256(
                self.fresh_test.semantic_context_hashes
            ),
        }
        return {
            "schema": _MANIFEST_SCHEMA,
            "format_version": _MANIFEST_VERSION,
            "task_config": asdict(self.task_config),
            "baseline_task_config": asdict(self.baseline_task_config),
            "source_dataset_sha256": self.source_dataset_sha256,
            "baseline_dataset_sha256": self.baseline_dataset_sha256,
            "role_allocation": (
                "salted SHA-256 rank over whole fresh V2 train contexts"
            ),
            "role_salt": V2_ROLE_SALT,
            "role_sizes": dict(V2_ROLE_SIZES),
            "role_context_hashes": {
                name: split.semantic_context_hashes
                for name, split in self.roles.items()
            },
            "reserve_context_hashes": self.reserve.semantic_context_hashes,
            "fresh_validation_context_hashes": (
                self.fresh_validation.semantic_context_hashes
            ),
            "fresh_test_context_hashes": (
                self.fresh_test.semantic_context_hashes
            ),
            "excluded_baseline_context_hashes": self.baseline_context_hashes,
            "context_set_sha256": context_set_sha256,
            "audit": {
                **asdict(self.audit),
                "all_overlap_checks_pass": (
                    self.audit.all_overlap_checks_pass
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class VariableStaticFullSpanV2NoveltyStratum:
    """Answer accuracy for one semantic novelty category."""

    samples: int
    contexts: int
    correct_samples: int
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class VariableStaticFullSpanV2NoveltyAccuracy:
    """Novel-key/value accuracy strata plus their declared joint gate."""

    new_key: VariableStaticFullSpanV2NoveltyStratum
    new_value: VariableStaticFullSpanV2NoveltyStratum
    key_only: VariableStaticFullSpanV2NoveltyStratum
    value_only: VariableStaticFullSpanV2NoveltyStratum
    both: VariableStaticFullSpanV2NoveltyStratum
    minimum_accuracy: float
    primary_strata_nonempty: bool
    new_key_pass: bool
    new_value_pass: bool
    both_pass: bool
    passes: bool


def _context_set_sha256(hashes: tuple[str, ...]) -> str:
    encoded = json.dumps(
        sorted(hashes),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _fresh_context_rows(
    split: VariableAssociativeRecallSplit,
    *,
    excluded_hashes: set[str],
) -> Tensor:
    return torch.tensor(
        [
            index
            for index, semantic_hash in enumerate(
                split.semantic_context_hashes
            )
            if semantic_hash not in excluded_hashes
        ],
        dtype=torch.int64,
    )


def _fresh_subset(
    split: VariableAssociativeRecallSplit,
    *,
    excluded_hashes: set[str],
    name: str,
) -> VariableAssociativeRecallSplit:
    rows = _fresh_context_rows(split, excluded_hashes=excluded_hashes)
    if rows.numel() == 0:
        raise ValueError(f"{name} contains no contexts fresh to the baseline")
    return subset_variable_associative_recall_split(
        split,
        context_rows=rows,
        name=name,
    )


def _salted_rank(
    split: VariableAssociativeRecallSplit,
    *,
    salt: str,
) -> tuple[int, ...]:
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a nonempty string")

    def rank(index: int) -> tuple[str, str]:
        semantic_hash = split.semantic_context_hashes[index]
        digest = hashlib.sha256()
        digest.update(_ROLE_HASH_DOMAIN)
        digest.update(salt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(semantic_hash.encode("ascii"))
        return digest.hexdigest(), semantic_hash

    return tuple(sorted(range(split.contexts), key=rank))


def _has_pairwise_overlap(values: tuple[set[str], ...]) -> bool:
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] & values[right]:
                return True
    return False


def build_variable_static_full_span_v2_protocol(
) -> VariableStaticFullSpanV2Protocol:
    """Build the frozen 10-by-10 V2 role allocation and audit it."""

    baseline = build_variable_associative_recall_splits(
        V2_BASELINE_TASK_CONFIG
    )
    source = build_variable_associative_recall_splits(V2_TASK_CONFIG)
    baseline_hash_set = (
        set(baseline.train.semantic_context_hashes)
        | set(baseline.validation.semantic_context_hashes)
        | set(baseline.test.semantic_context_hashes)
    )
    baseline_hashes = tuple(sorted(baseline_hash_set))
    if len(baseline_hashes) != V2_BASELINE_TASK_CONFIG.semantic_context_count:
        raise RuntimeError("baseline semantic-context hashes are not unique")

    fresh_train = _fresh_subset(
        source.train,
        excluded_hashes=baseline_hash_set,
        name="v2_fresh_train",
    )
    fresh_validation = _fresh_subset(
        source.validation,
        excluded_hashes=baseline_hash_set,
        name="v2_fresh_validation",
    )
    fresh_test = _fresh_subset(
        source.test,
        excluded_hashes=baseline_hash_set,
        name="v2_fresh_test",
    )

    ranked_rows = _salted_rank(fresh_train, salt=V2_ROLE_SALT)
    required = sum(V2_ROLE_SIZES.values())
    if len(ranked_rows) <= required:
        raise ValueError("too few fresh train contexts for roles and reserve")

    roles: dict[str, VariableAssociativeRecallSplit] = {}
    cursor = 0
    for name, size in V2_ROLE_SIZES.items():
        rows = torch.tensor(
            ranked_rows[cursor : cursor + size],
            dtype=torch.int64,
        )
        roles[name] = subset_variable_associative_recall_split(
            fresh_train,
            context_rows=rows,
            name=name,
        )
        cursor += size
    reserve = subset_variable_associative_recall_split(
        fresh_train,
        context_rows=torch.tensor(ranked_rows[cursor:], dtype=torch.int64),
        name="reserve",
    )

    role_sets = tuple(
        set(roles[name].semantic_context_hashes) for name in V2_ROLE_NAMES
    )
    all_roles = set().union(*role_sets)
    reserve_hashes = set(reserve.semantic_context_hashes)
    validation_hashes = set(fresh_validation.semantic_context_hashes)
    test_hashes = set(fresh_test.semantic_context_hashes)
    audit = VariableStaticFullSpanV2ProtocolAudit(
        baseline_contexts=len(baseline_hashes),
        source_contexts=V2_TASK_CONFIG.semantic_context_count,
        source_train_contexts=source.train.contexts,
        source_validation_contexts=source.validation.contexts,
        source_test_contexts=source.test.contexts,
        fresh_train_contexts=fresh_train.contexts,
        fresh_validation_contexts=fresh_validation.contexts,
        fresh_test_contexts=fresh_test.contexts,
        excluded_train_contexts=source.train.contexts - fresh_train.contexts,
        excluded_validation_contexts=(
            source.validation.contexts - fresh_validation.contexts
        ),
        excluded_test_contexts=source.test.contexts - fresh_test.contexts,
        allocated_role_contexts=len(all_roles),
        reserve_contexts=reserve.contexts,
        role_context_counts=tuple(
            (name, roles[name].contexts) for name in V2_ROLE_NAMES
        ),
        role_pairwise_overlap=_has_pairwise_overlap(role_sets),
        roles_baseline_overlap=bool(all_roles & baseline_hash_set),
        roles_validation_overlap=bool(all_roles & validation_hashes),
        roles_test_overlap=bool(all_roles & test_hashes),
        reserve_role_overlap=bool(reserve_hashes & all_roles),
        reserve_baseline_overlap=bool(reserve_hashes & baseline_hash_set),
        reserve_validation_overlap=bool(
            reserve_hashes & validation_hashes
        ),
        reserve_test_overlap=bool(reserve_hashes & test_hashes),
        fresh_validation_baseline_overlap=bool(
            validation_hashes & baseline_hash_set
        ),
        fresh_test_baseline_overlap=bool(test_hashes & baseline_hash_set),
        fresh_validation_test_overlap=bool(validation_hashes & test_hashes),
    )
    if not audit.all_overlap_checks_pass:
        raise RuntimeError("V2 protocol context roles overlap")
    if audit.allocated_role_contexts != required:
        raise RuntimeError("V2 protocol did not allocate every declared role")
    if audit.allocated_role_contexts + audit.reserve_contexts != (
        audit.fresh_train_contexts
    ):
        raise RuntimeError("V2 roles and reserve do not exhaust fresh train")

    return VariableStaticFullSpanV2Protocol(
        task_config=V2_TASK_CONFIG,
        baseline_task_config=V2_BASELINE_TASK_CONFIG,
        source_dataset_sha256=source.dataset_sha256,
        baseline_dataset_sha256=baseline.dataset_sha256,
        baseline_context_hashes=baseline_hashes,
        fresh_train=fresh_train,
        basis_fit_a=roles["basis_fit_a"],
        graph_fit_a=roles["graph_fit_a"],
        graph_stop_a=roles["graph_stop_a"],
        graph_select_a=roles["graph_select_a"],
        calibration_b=roles["calibration_b"],
        reserve=reserve,
        fresh_validation=fresh_validation,
        fresh_test=fresh_test,
        audit=audit,
    )


def _novelty_stratum(
    *,
    context_mask: Tensor,
    split: VariableAssociativeRecallSplit,
    correct: Tensor,
) -> VariableStaticFullSpanV2NoveltyStratum:
    context_mask = context_mask.to(device=correct.device, dtype=torch.bool)
    example_context_indices = split.example_context_indices.to(
        device=correct.device
    )
    sample_mask = context_mask.index_select(0, example_context_indices)
    samples = int(sample_mask.sum().item())
    contexts = int(context_mask.sum().item())
    correct_samples = int(correct[sample_mask].sum().item())
    return VariableStaticFullSpanV2NoveltyStratum(
        samples=samples,
        contexts=contexts,
        correct_samples=correct_samples,
        accuracy=(correct_samples / samples if samples else None),
    )


def variable_static_full_span_v2_novelty_accuracy(
    split: VariableAssociativeRecallSplit,
    answer_logits: Tensor,
    *,
    minimum_accuracy: float = 1.0,
) -> VariableStaticFullSpanV2NoveltyAccuracy:
    """Measure and gate accuracy on key/value novelty intersections."""

    if not isinstance(split, VariableAssociativeRecallSplit):
        raise TypeError("split must be a VariableAssociativeRecallSplit")
    if (
        not isinstance(answer_logits, Tensor)
        or answer_logits.ndim != 2
        or answer_logits.shape[0] != split.samples
    ):
        raise ValueError("answer_logits must have shape [split samples, vocab]")
    if answer_logits.shape[1] <= int(split.answer_token_ids.max().item()):
        raise ValueError("answer_logits do not cover the task vocabulary")
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("minimum_accuracy must be in [0, 1]")

    device = answer_logits.device
    contexts = split.semantic_contexts.to(device=device)
    has_new_key = (contexts[:, :2] >= V2_NEW_KEY_START).any(dim=1)
    has_new_value = (contexts[:, 2:] >= V2_NEW_VALUE_START).any(dim=1)
    correct = answer_logits.argmax(dim=-1).eq(
        split.answer_token_ids.to(device=device)
    )

    new_key = _novelty_stratum(
        context_mask=has_new_key,
        split=split,
        correct=correct,
    )
    new_value = _novelty_stratum(
        context_mask=has_new_value,
        split=split,
        correct=correct,
    )
    key_only = _novelty_stratum(
        context_mask=has_new_key & ~has_new_value,
        split=split,
        correct=correct,
    )
    value_only = _novelty_stratum(
        context_mask=~has_new_key & has_new_value,
        split=split,
        correct=correct,
    )
    both = _novelty_stratum(
        context_mask=has_new_key & has_new_value,
        split=split,
        correct=correct,
    )
    primary_nonempty = all(
        stratum.samples > 0 and stratum.contexts > 0
        for stratum in (new_key, new_value, both)
    )

    def passes(stratum: VariableStaticFullSpanV2NoveltyStratum) -> bool:
        return (
            stratum.accuracy is not None
            and stratum.accuracy >= minimum_accuracy
        )

    new_key_pass = passes(new_key)
    new_value_pass = passes(new_value)
    both_pass = passes(both)
    return VariableStaticFullSpanV2NoveltyAccuracy(
        new_key=new_key,
        new_value=new_value,
        key_only=key_only,
        value_only=value_only,
        both=both,
        minimum_accuracy=minimum_accuracy,
        primary_strata_nonempty=primary_nonempty,
        new_key_pass=new_key_pass,
        new_value_pass=new_value_pass,
        both_pass=both_pass,
        passes=(
            primary_nonempty
            and new_key_pass
            and new_value_pass
            and both_pass
        ),
    )


__all__ = [
    "V2_BASELINE_TASK_CONFIG",
    "V2_NEW_KEY_START",
    "V2_NEW_VALUE_START",
    "V2_ROLE_NAMES",
    "V2_ROLE_SALT",
    "V2_ROLE_SIZES",
    "V2_TASK_CONFIG",
    "VariableStaticFullSpanV2NoveltyAccuracy",
    "VariableStaticFullSpanV2NoveltyStratum",
    "VariableStaticFullSpanV2Protocol",
    "VariableStaticFullSpanV2ProtocolAudit",
    "build_variable_static_full_span_v2_protocol",
    "variable_static_full_span_v2_novelty_accuracy",
]
