"""Leakage-aware development partitioning and unified graph-rung evaluation.

This module is deliberately independent of the Gemma development runner.  It
provides two small pieces needed by a multi-node modal-generator rung:

* a deterministic, source-safe split of one 40-prompt development export into
  disjoint interaction-selection and open-assessment memberships; and
* one evaluator that compares native inference, an interacting graph, the
  node-identical edgeless graph, matched deletion, and an optional nodewise
  dense-fused control.

The assessment partition remains open development.  It is not a closed guard
or test split, even though its declared prompt and source-index memberships are
disjoint from the selection partition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from .compiler.calibration import CalibrationBatch


__all__ = [
    "DevelopmentInteractionPartition",
    "DevelopmentInteractionPartitionPlan",
    "evaluate_modal_graph_rung_conditions",
    "partition_development_export_for_interactions",
]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_PARTITION_DOMAIN = b"fisher_graph.modal_graph_rung.partition.v1\0"
_PARTITION_PLAN_DOMAIN = b"fisher_graph.modal_graph_rung.partition_plan.v1\0"
_MEMBERSHIP_SCORE_DOMAIN = (
    b"fisher_graph.modal_graph_rung.membership_score.v1\0"
)
_PARTITION_ROLES = frozenset(
    {"interaction_selection", "open_development_assessment"}
)
_DEFAULT_PARTITION_SALT = "four-node-fanin-rung-v1"
_ASSESSMENT_ROLES = frozenset(
    {
        "open_development_assessment",
        "claimed_closed_guard_assessment",
    }
)


class _DevelopmentExportLike(Protocol):
    prompts: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]
    family_ids: tuple[str, ...]
    fit_positions: tuple[int, ...]
    source_prompt_indices: tuple[int, ...]
    source_corpus_id: str
    source_fit_prompt_index_sha256: str
    artifact_sha256: str


class _AdapterLike(Protocol):
    module: object


class _GraphExecutorLike(Protocol):
    graph_plan: object

    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str,
    ) -> object: ...


class _DenseExecutorLike(Protocol):
    def run(
        self,
        model_inputs: Mapping[str, Tensor],
        *,
        condition: str,
    ) -> object: ...


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _raw_text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_canonical_name(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _CANONICAL_NAME.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True, slots=True)
class DevelopmentInteractionPartition:
    """One ephemeral prompt-bearing development partition.

    ``metadata()`` intentionally omits prompt text.  The in-memory object keeps
    prompts only so the caller can tokenize the selected membership.
    """

    role: str
    prompts: tuple[str, ...] = field(repr=False)
    prompt_sha256s: tuple[str, ...]
    family_ids: tuple[str, ...]
    fit_positions: tuple[int, ...]
    source_prompt_indices: tuple[int, ...]
    source_export_sha256: str
    source_corpus_id: str
    source_fit_prompt_index_sha256: str
    partition_rule: str
    partition_salt: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role not in _PARTITION_ROLES:
            raise ValueError("development interaction partition role is invalid")
        count = len(self.prompts)
        if count <= 0:
            raise ValueError("development interaction partition cannot be empty")
        if (
            type(self.prompts) is not tuple
            or type(self.prompt_sha256s) is not tuple
            or type(self.family_ids) is not tuple
            or type(self.fit_positions) is not tuple
            or type(self.source_prompt_indices) is not tuple
            or any(
                len(values) != count
                for values in (
                    self.prompt_sha256s,
                    self.family_ids,
                    self.fit_positions,
                    self.source_prompt_indices,
                )
            )
        ):
            raise ValueError("development partition columns are inconsistent")
        if any(
            not isinstance(prompt, str) or not prompt.strip()
            for prompt in self.prompts
        ):
            raise ValueError("development partition prompts must be nonempty")
        if tuple(
            _raw_text_sha256(prompt) for prompt in self.prompts
        ) != self.prompt_sha256s:
            raise ValueError("development partition prompt hashes drifted")
        if len(set(self.prompt_sha256s)) != count:
            raise ValueError("development partition prompt hashes repeat")
        for index, digest in enumerate(self.prompt_sha256s):
            _require_sha256(digest, label=f"prompt_sha256s[{index}]")
        if any(
            not isinstance(family, str) or not family
            for family in self.family_ids
        ):
            raise ValueError("development partition family ids are invalid")
        for label, values in (
            ("fit_positions", self.fit_positions),
            ("source_prompt_indices", self.source_prompt_indices),
        ):
            if (
                any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != count
            ):
                raise ValueError(
                    f"development partition {label} must be unique "
                    "nonnegative integers"
                )
        _require_sha256(
            self.source_export_sha256,
            label="source_export_sha256",
        )
        _require_sha256(
            self.source_fit_prompt_index_sha256,
            label="source_fit_prompt_index_sha256",
        )
        _require_canonical_name(
            self.source_corpus_id,
            label="source_corpus_id",
        )
        _require_canonical_name(
            self.partition_rule,
            label="partition_rule",
        )
        _require_canonical_name(
            self.partition_salt,
            label="partition_salt",
        )
        computed = _json_sha256(
            self._safe_payload(),
            domain=_PARTITION_DOMAIN,
        )
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            != computed
        ):
            raise ValueError("development partition artifact hash mismatch")

    @property
    def prompt_count(self) -> int:
        return len(self.prompts)

    def _safe_payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.development_interaction_partition",
            "format_version": 1,
            "role": self.role,
            "source_export_sha256": self.source_export_sha256,
            "source_corpus_id": self.source_corpus_id,
            "source_fit_prompt_index_sha256": (
                self.source_fit_prompt_index_sha256
            ),
            "partition_rule": self.partition_rule,
            "partition_salt": self.partition_salt,
            "prompt_sha256s": self.prompt_sha256s,
            "family_ids": self.family_ids,
            "fit_positions": self.fit_positions,
            "source_prompt_indices": self.source_prompt_indices,
            "assessment_status": "open_development_not_closed_guard",
            "serialized_contains_prompt_text": False,
            "membership_provenance": "caller_declared_self_attested",
            "membership_externally_authenticated": False,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._safe_payload(),
            "prompt_count": self.prompt_count,
            "family_count": len(set(self.family_ids)),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentInteractionPartitionPlan:
    """Authenticated declaration that selection and assessment are disjoint."""

    source_export_sha256: str
    selection: DevelopmentInteractionPartition
    assessment: DevelopmentInteractionPartition
    expected_prompt_count: int
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_export_sha256,
            label="source_export_sha256",
        )
        if (
            not isinstance(self.selection, DevelopmentInteractionPartition)
            or not isinstance(
                self.assessment,
                DevelopmentInteractionPartition,
            )
        ):
            raise TypeError("partition plan members have invalid types")
        if (
            self.selection.role != "interaction_selection"
            or self.assessment.role != "open_development_assessment"
            or self.selection.source_export_sha256
            != self.source_export_sha256
            or self.assessment.source_export_sha256
            != self.source_export_sha256
            or self.selection.source_corpus_id
            != self.assessment.source_corpus_id
            or self.selection.source_fit_prompt_index_sha256
            != self.assessment.source_fit_prompt_index_sha256
            or self.selection.partition_rule
            != self.assessment.partition_rule
            or self.selection.partition_salt
            != self.assessment.partition_salt
        ):
            raise ValueError("partition plan member provenance is invalid")
        if (
            type(self.expected_prompt_count) is not int
            or self.expected_prompt_count <= 1
            or self.selection.prompt_count + self.assessment.prompt_count
            != self.expected_prompt_count
        ):
            raise ValueError("partition plan prompt accounting is invalid")
        prompt_overlap = set(self.selection.prompt_sha256s) & set(
            self.assessment.prompt_sha256s
        )
        source_overlap = set(self.selection.source_prompt_indices) & set(
            self.assessment.source_prompt_indices
        )
        position_overlap = set(self.selection.fit_positions) & set(
            self.assessment.fit_positions
        )
        if prompt_overlap or source_overlap or position_overlap:
            raise ValueError("selection and assessment memberships overlap")
        computed = _json_sha256(
            self._safe_payload(),
            domain=_PARTITION_PLAN_DOMAIN,
        )
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(
                self.artifact_sha256,
                label="artifact_sha256",
            )
            != computed
        ):
            raise ValueError("development partition plan hash mismatch")

    def _safe_payload(self) -> dict[str, object]:
        family_overlap = set(self.selection.family_ids) & set(
            self.assessment.family_ids
        )
        return {
            "schema": "fisher_graph.development_interaction_partition_plan",
            "format_version": 1,
            "source_export_sha256": self.source_export_sha256,
            "selection_partition_sha256": self.selection.artifact_sha256,
            "assessment_partition_sha256": self.assessment.artifact_sha256,
            "expected_prompt_count": self.expected_prompt_count,
            "selection_prompt_count": self.selection.prompt_count,
            "assessment_prompt_count": self.assessment.prompt_count,
            "prompt_membership_disjoint": True,
            "source_prompt_index_membership_disjoint": True,
            "fit_position_membership_disjoint": True,
            "family_disjoint": not family_overlap,
            "overlapping_family_count": len(family_overlap),
            "assessment_status": "open_development_not_closed_guard",
            "membership_provenance": "caller_declared_self_attested",
            "membership_externally_authenticated": False,
            "serialized_contains_prompt_text": False,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._safe_payload(),
            "artifact_sha256": self.artifact_sha256,
            "selection": self.selection.metadata(),
            "assessment": self.assessment.metadata(),
        }


def _canonical_export_columns(
    export: _DevelopmentExportLike,
    *,
    expected_prompt_count: int,
) -> tuple[
    tuple[str, str, str, int, int],
    ...,
]:
    required = (
        "prompts",
        "prompt_sha256s",
        "family_ids",
        "fit_positions",
        "source_prompt_indices",
        "source_corpus_id",
        "source_fit_prompt_index_sha256",
        "artifact_sha256",
    )
    if any(not hasattr(export, name) for name in required):
        raise TypeError("development export is missing required columns")
    columns = (
        export.prompts,
        export.prompt_sha256s,
        export.family_ids,
        export.fit_positions,
        export.source_prompt_indices,
    )
    if any(type(values) is not tuple for values in columns):
        raise TypeError("development export columns must be tuples")
    if (
        type(expected_prompt_count) is not int
        or expected_prompt_count <= 1
        or any(len(values) != expected_prompt_count for values in columns)
    ):
        raise ValueError(
            f"development export must contain exactly "
            f"{expected_prompt_count} prompts"
        )
    rows = tuple(zip(*columns, strict=True))
    if tuple(_raw_text_sha256(row[0]) for row in rows) != tuple(
        row[1] for row in rows
    ):
        raise ValueError("development export prompt hashes drifted")
    for digest in export.prompt_sha256s:
        _require_sha256(digest, label="prompt_sha256")
    for value, label in (
        (export.artifact_sha256, "source export artifact_sha256"),
        (
            export.source_fit_prompt_index_sha256,
            "source_fit_prompt_index_sha256",
        ),
    ):
        _require_sha256(value, label=label)
    _require_canonical_name(
        export.source_corpus_id,
        label="source_corpus_id",
    )
    if (
        len(set(export.prompt_sha256s)) != expected_prompt_count
        or len(set(export.fit_positions)) != expected_prompt_count
        or len(set(export.source_prompt_indices)) != expected_prompt_count
    ):
        raise ValueError("development export membership columns repeat")
    return rows


def partition_development_export_for_interactions(
    export: _DevelopmentExportLike,
    *,
    selection_count: int = 20,
    expected_prompt_count: int = 40,
    partition_salt: str = _DEFAULT_PARTITION_SALT,
) -> DevelopmentInteractionPartitionPlan:
    """Deterministically reserve disjoint selection and assessment prompts."""

    rows = _canonical_export_columns(
        export,
        expected_prompt_count=expected_prompt_count,
    )
    if (
        type(selection_count) is not int
        or not 0 < selection_count < expected_prompt_count
    ):
        raise ValueError(
            "selection_count must leave nonempty selection and assessment"
        )
    salt = _require_canonical_name(
        partition_salt,
        label="partition_salt",
    )
    partition_rule = (
        f"sha256_ordered_{selection_count}_of_{expected_prompt_count}_v1"
    )
    scored: list[
        tuple[str, str, tuple[str, str, str, int, int]]
    ] = []
    for row in rows:
        prompt_sha256 = row[1]
        score = _json_sha256(
            {
                "source_export_sha256": export.artifact_sha256,
                "partition_salt": salt,
                "prompt_sha256": prompt_sha256,
                "source_prompt_index": row[4],
            },
            domain=_MEMBERSHIP_SCORE_DOMAIN,
        )
        scored.append((score, prompt_sha256, row))
    ordered = tuple(value[2] for value in sorted(scored))
    selected_rows = ordered[:selection_count]
    assessment_rows = ordered[selection_count:]

    def build(
        role: str,
        selected: tuple[tuple[str, str, str, int, int], ...],
    ) -> DevelopmentInteractionPartition:
        # Restore declared fit order inside each deterministically chosen set.
        canonical = tuple(sorted(selected, key=lambda row: (row[3], row[1])))
        return DevelopmentInteractionPartition(
            role=role,
            prompts=tuple(row[0] for row in canonical),
            prompt_sha256s=tuple(row[1] for row in canonical),
            family_ids=tuple(row[2] for row in canonical),
            fit_positions=tuple(row[3] for row in canonical),
            source_prompt_indices=tuple(row[4] for row in canonical),
            source_export_sha256=export.artifact_sha256,
            source_corpus_id=export.source_corpus_id,
            source_fit_prompt_index_sha256=(
                export.source_fit_prompt_index_sha256
            ),
            partition_rule=partition_rule,
            partition_salt=salt,
        )

    return DevelopmentInteractionPartitionPlan(
        source_export_sha256=export.artifact_sha256,
        selection=build("interaction_selection", selected_rows),
        assessment=build(
            "open_development_assessment",
            assessment_rows,
        ),
        expected_prompt_count=expected_prompt_count,
    )


def _model_logits(output: object) -> Tensor:
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise TypeError("model output must expose [batch, sequence, vocab] logits")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("model logits must be finite floating values")
    return logits


def _selected_logits_and_targets(
    logits: Tensor,
    batch: CalibrationBatch,
) -> tuple[Tensor, Tensor]:
    targets = batch.targets.to(device=logits.device)
    if targets.shape != logits.shape[:2]:
        raise ValueError("evaluation targets and logits positions differ")
    selected = targets != -100
    valid = batch.valid_positions.to(device=logits.device)
    if valid.shape != selected.shape or bool((selected & ~valid).any()):
        raise ValueError(
            "supervised targets must be a subset of valid positions"
        )
    if not bool(selected.any()):
        raise ValueError("evaluation batch has no supervised tokens")
    return (
        logits[selected].detach().to(device="cpu", dtype=torch.float32),
        targets[selected].detach().to(device="cpu", dtype=torch.long),
    )


def _native_nll(logits: Tensor, targets: Tensor) -> float:
    return float(
        F.cross_entropy(logits, targets, reduction="sum").double().item()
    )


def _candidate_comparison(
    native_logits: Tensor,
    candidate_logits: Tensor,
    targets: Tensor,
    *,
    vocabulary_chunk_size: int,
) -> dict[str, float | int]:
    if candidate_logits.shape != native_logits.shape:
        raise ValueError("native and candidate supervised logits differ")
    native_lse = torch.logsumexp(native_logits, dim=-1)
    candidate_lse = torch.logsumexp(candidate_logits, dim=-1)
    row = torch.arange(targets.shape[0])
    nll = -(
        candidate_logits[row, targets] - candidate_lse
    ).double().sum()
    top1_matches = int(
        (
            candidate_logits.argmax(dim=-1)
            == native_logits.argmax(dim=-1)
        ).sum().item()
    )
    kl_sum = 0.0
    for start in range(0, native_logits.shape[1], vocabulary_chunk_size):
        stop = min(start + vocabulary_chunk_size, native_logits.shape[1])
        native_log_probability = (
            native_logits[:, start:stop] - native_lse[:, None]
        ).double()
        candidate_log_probability = (
            candidate_logits[:, start:stop] - candidate_lse[:, None]
        ).double()
        kl_sum += float(
            (
                native_log_probability.exp()
                * (native_log_probability - candidate_log_probability)
            ).sum().item()
        )
    return {
        "nll_sum": float(nll.item()),
        "native_to_candidate_kl_sum": max(kl_sum, 0.0),
        "top1_matches": top1_matches,
    }


def _plan_signature(plan: object) -> tuple[object, ...]:
    nodes = getattr(plan, "nodes", None)
    interactions = getattr(plan, "interactions", None)
    traversal = getattr(plan, "traversal_order", None)
    if (
        type(nodes) is not tuple
        or not nodes
        or type(interactions) is not tuple
        or type(traversal) is not tuple
    ):
        raise TypeError("graph executor plan is missing canonical graph fields")
    node_hashes = tuple(
        _require_sha256(
            getattr(node, "artifact_sha256", None),
            label="graph node artifact_sha256",
        )
        for node in nodes
    )
    return (
        _require_sha256(
            getattr(plan, "model_fingerprint", None),
            label="graph model_fingerprint",
        ),
        _require_sha256(
            getattr(plan, "parameter_cluster_plan_sha256", None),
            label="graph parameter_cluster_plan_sha256",
        ),
        node_hashes,
        traversal,
    )


_GRAPH_STATIC_FIELDS = (
    "replacement_scope",
    "replaced_layer_count",
    "graph_node_count",
    "fragment_count",
    "removed_mode_count",
    "source_whole_model_learned_parameters",
    "candidate_whole_model_learned_parameters",
    "native_removed_learned_parameters",
    "modal_graph_learned_parameters",
    "net_stored_parameter_savings",
    "graph_runtime_storage",
)
_GRAPH_LOGICAL_FIELDS = (
    "logical_linear_macs_native_removed",
    "logical_modal_graph_macs",
    "logical_executed_modal_graph_macs",
    "logical_modal_graph_additions",
    "logical_executed_modal_graph_additions",
    "net_logical_macs_saved",
)
_DENSE_STATIC_FIELDS = (
    "replacement_scope",
    "replaced_layer_count",
    "removed_mode_count",
    "source_whole_model_learned_parameters",
    "candidate_whole_model_learned_parameters",
    "native_removed_learned_parameters",
    "modal_generator_learned_parameters",
    "net_stored_parameter_savings",
)
_DENSE_LOGICAL_FIELDS = (
    "logical_linear_macs_native_removed",
    "logical_modal_generator_macs",
    "logical_executed_modal_generator_macs",
    "logical_modal_generator_bias_additions",
    "logical_executed_modal_generator_bias_additions",
    "net_logical_macs_saved",
)


def _execution_fields(
    execution: object,
    names: tuple[str, ...],
    *,
    label: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in names:
        if not hasattr(execution, name):
            raise TypeError(f"{label} execution is missing {name}")
        value = getattr(execution, name)
        if name == "replacement_scope" or name == "graph_runtime_storage":
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} {name} must be a nonempty string")
        else:
            if type(value) is not int:
                raise ValueError(f"{label} {name} must be an integer")
        result[name] = value
    return result


def _accumulate_logical(
    totals: dict[str, int],
    execution: object,
    names: tuple[str, ...],
    *,
    label: str,
) -> None:
    for name in names:
        value = getattr(execution, name, None)
        if type(value) is not int:
            raise ValueError(f"{label} {name} must be an integer")
        totals[name] += value


def _assert_close_logits(
    left: Tensor,
    right: Tensor,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> float:
    if left.shape != right.shape:
        raise ValueError(f"{label} supervised logit shapes differ")
    difference = float((left - right).abs().max().item())
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise ValueError(
            f"{label} supervised logits exceed declared tolerance: "
            f"max_abs={difference}, atol={atol}, rtol={rtol}"
        )
    return difference


def _validate_graph_execution(
    execution: object,
    plan: object,
    *,
    condition: str,
    label: str,
) -> None:
    if getattr(execution, "condition", None) != condition:
        raise RuntimeError(f"{label} execution condition drifted")
    graph_execution = getattr(execution, "graph_execution", None)
    traversal = getattr(graph_execution, "traversal_order", None)
    if type(traversal) is not tuple:
        raise TypeError(f"{label} execution lacks graph traversal evidence")
    expected = (
        getattr(plan, "traversal_order")
        if condition == "generated"
        else ()
    )
    if traversal != expected:
        raise RuntimeError(f"{label} graph traversal order drifted")


def evaluate_modal_graph_rung_conditions(
    adapter: _AdapterLike,
    interacting_executor: _GraphExecutorLike,
    edgeless_executor: _GraphExecutorLike,
    batches: Sequence[CalibrationBatch],
    *,
    nodewise_dense_executor: _DenseExecutorLike | None = None,
    dense_equivalence_atol: float = 1e-5,
    dense_equivalence_rtol: float = 1e-5,
    deletion_equivalence_atol: float = 0.0,
    deletion_equivalence_rtol: float = 0.0,
    vocabulary_chunk_size: int = 16384,
    assessment_role: str = "open_development_assessment",
    expected_example_ids: Sequence[str],
) -> dict[str, object]:
    """Evaluate every honest control for one frozen modal-graph rung.

    The interacting and edgeless graph plans must contain exactly the same
    authenticated nodes; the latter must contain no interactions.  Deletion is
    executed through both plans and required to agree within its declared
    tolerance.  When supplied, the nodewise dense-fused executor must agree
    with the edgeless graph on supervised logits within the dense tolerance.
    """

    materialized = tuple(batches)
    if (
        not materialized
        or any(not isinstance(batch, CalibrationBatch) for batch in materialized)
    ):
        raise ValueError("batches must contain CalibrationBatch values")
    if (
        isinstance(expected_example_ids, (str, bytes))
        or not isinstance(expected_example_ids, Sequence)
    ):
        raise TypeError("expected_example_ids must be a sequence")
    expected_ids = tuple(expected_example_ids)
    observed_ids = tuple(
        example_id
        for batch in materialized
        for example_id in (
            batch.example_ids
            if batch.example_ids is not None
            else ()
        )
    )
    if (
        not expected_ids
        or len(expected_ids) != len(set(expected_ids))
        or any(not isinstance(value, str) or not value for value in expected_ids)
        or any(batch.example_ids is None for batch in materialized)
        or observed_ids != expected_ids
    ):
        raise ValueError(
            "assessment batches do not match the declared example membership"
        )
    for value, label in (
        (dense_equivalence_atol, "dense_equivalence_atol"),
        (dense_equivalence_rtol, "dense_equivalence_rtol"),
        (deletion_equivalence_atol, "deletion_equivalence_atol"),
        (deletion_equivalence_rtol, "deletion_equivalence_rtol"),
    ):
        _finite_nonnegative(value, label=label)
    if type(vocabulary_chunk_size) is not int or vocabulary_chunk_size <= 0:
        raise ValueError("vocabulary_chunk_size must be positive")
    if assessment_role not in _ASSESSMENT_ROLES:
        raise ValueError(
            "assessment_role must identify open development or one "
            "claim-first closed guard"
        )
    native_model = getattr(adapter, "module", None)
    if not callable(native_model):
        raise TypeError("adapter must expose a callable native module")
    interacting_plan = getattr(interacting_executor, "graph_plan", None)
    edgeless_plan = getattr(edgeless_executor, "graph_plan", None)
    if _plan_signature(interacting_plan) != _plan_signature(edgeless_plan):
        raise ValueError(
            "interacting and edgeless plans must contain identical modal nodes"
        )
    interacting_edges = getattr(interacting_plan, "interactions")
    edgeless_edges = getattr(edgeless_plan, "interactions")
    if edgeless_edges:
        raise ValueError("edgeless control plan contains interactions")

    condition_names = [
        "interacting_graph",
        "edgeless_graph",
        "matched_deletion",
    ]
    if nodewise_dense_executor is not None:
        condition_names.append("nodewise_dense_fused")
    metric_totals = {
        name: {
            "nll_sum": 0.0,
            "native_to_candidate_kl_sum": 0.0,
            "top1_matches": 0,
        }
        for name in condition_names
    }
    graph_logical_totals = {
        name: {field: 0 for field in _GRAPH_LOGICAL_FIELDS}
        for name in (
            "interacting_graph",
            "edgeless_graph",
            "matched_deletion",
        )
    }
    dense_logical_totals = {
        field: 0 for field in _DENSE_LOGICAL_FIELDS
    }
    graph_static: dict[str, dict[str, object]] = {}
    graph_peak_live_width = {
        "interacting_graph": 0,
        "edgeless_graph": 0,
        "matched_deletion": 0,
    }
    dense_static: dict[str, object] | None = None
    native_nll_sum = 0.0
    supervised_tokens = 0
    logical_valid_tokens = 0
    deletion_max_abs = 0.0
    dense_max_abs = 0.0

    for batch in materialized:
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        with torch.no_grad():
            native_output = native_model(**call_inputs)
            interacting = interacting_executor.run(
                batch.model_inputs,
                condition="generated",
            )
            edgeless = edgeless_executor.run(
                batch.model_inputs,
                condition="generated",
            )
            interacting_deletion = interacting_executor.run(
                batch.model_inputs,
                condition="deletion",
            )
            edgeless_deletion = edgeless_executor.run(
                batch.model_inputs,
                condition="deletion",
            )
            dense = (
                None
                if nodewise_dense_executor is None
                else nodewise_dense_executor.run(
                    batch.model_inputs,
                    condition="generated",
                )
            )
        _validate_graph_execution(
            interacting,
            interacting_plan,
            condition="generated",
            label="interacting",
        )
        _validate_graph_execution(
            edgeless,
            edgeless_plan,
            condition="generated",
            label="edgeless",
        )
        _validate_graph_execution(
            interacting_deletion,
            interacting_plan,
            condition="deletion",
            label="interacting deletion",
        )
        _validate_graph_execution(
            edgeless_deletion,
            edgeless_plan,
            condition="deletion",
            label="edgeless deletion",
        )

        native_logits, targets = _selected_logits_and_targets(
            _model_logits(native_output),
            batch,
        )
        native_nll_sum += _native_nll(native_logits, targets)
        supervised_tokens += targets.numel()
        executions = {
            "interacting_graph": interacting,
            "edgeless_graph": edgeless,
            "matched_deletion": interacting_deletion,
        }
        selected_logits: dict[str, Tensor] = {}
        for name, execution in executions.items():
            logits, candidate_targets = _selected_logits_and_targets(
                _model_logits(getattr(execution, "model_output", None)),
                batch,
            )
            if not torch.equal(targets, candidate_targets):
                raise RuntimeError(f"{name} evaluation targets drifted")
            selected_logits[name] = logits
            comparison = _candidate_comparison(
                native_logits,
                logits,
                targets,
                vocabulary_chunk_size=vocabulary_chunk_size,
            )
            for metric, value in comparison.items():
                metric_totals[name][metric] += value

            static = _execution_fields(
                execution,
                _GRAPH_STATIC_FIELDS,
                label=name,
            )
            prior_static = graph_static.setdefault(name, static)
            if prior_static != static:
                raise RuntimeError(f"{name} static accounting changed by batch")
            _accumulate_logical(
                graph_logical_totals[name],
                execution,
                _GRAPH_LOGICAL_FIELDS,
                label=name,
            )
            peak_width = _exact_nonnegative_int(
                getattr(execution, "peak_live_modal_width", None),
                label=f"{name} peak_live_modal_width",
            )
            graph_peak_live_width[name] = max(
                graph_peak_live_width[name],
                peak_width,
            )

        other_deletion_logits, other_deletion_targets = (
            _selected_logits_and_targets(
                _model_logits(
                    getattr(edgeless_deletion, "model_output", None)
                ),
                batch,
            )
        )
        if not torch.equal(targets, other_deletion_targets):
            raise RuntimeError("edgeless deletion targets drifted")
        deletion_max_abs = max(
            deletion_max_abs,
            _assert_close_logits(
                selected_logits["matched_deletion"],
                other_deletion_logits,
                atol=deletion_equivalence_atol,
                rtol=deletion_equivalence_rtol,
                label="interacting/edgeless deletion",
            ),
        )
        other_deletion_static = _execution_fields(
            edgeless_deletion,
            _GRAPH_STATIC_FIELDS,
            label="edgeless deletion",
        )
        if other_deletion_static != graph_static["edgeless_graph"]:
            raise RuntimeError(
                "edgeless generated/deletion static accounting differs"
            )
        if any(
            getattr(edgeless_deletion, field) != 0
            for field in (
                "logical_executed_modal_graph_macs",
                "logical_executed_modal_graph_additions",
                "peak_live_modal_width",
            )
        ):
            raise RuntimeError("edgeless deletion executed modal graph work")

        valid_token_values = {
            getattr(execution, "valid_tokens", None)
            for execution in (
                interacting,
                edgeless,
                interacting_deletion,
                edgeless_deletion,
            )
        }
        if (
            len(valid_token_values) != 1
            or any(type(value) is not int for value in valid_token_values)
        ):
            raise RuntimeError("graph conditions disagree on valid-token count")
        valid_tokens = next(iter(valid_token_values))
        if valid_tokens != int(batch.valid_positions.sum().item()):
            raise RuntimeError(
                "graph valid-token accounting differs from calibration mask"
            )
        logical_valid_tokens += valid_tokens

        if dense is not None:
            dense_logits, dense_targets = _selected_logits_and_targets(
                _model_logits(getattr(dense, "model_output", None)),
                batch,
            )
            if not torch.equal(targets, dense_targets):
                raise RuntimeError("nodewise dense evaluation targets drifted")
            dense_max_abs = max(
                dense_max_abs,
                _assert_close_logits(
                    selected_logits["edgeless_graph"],
                    dense_logits,
                    atol=dense_equivalence_atol,
                    rtol=dense_equivalence_rtol,
                    label="edgeless graph/nodewise dense",
                ),
            )
            comparison = _candidate_comparison(
                native_logits,
                dense_logits,
                targets,
                vocabulary_chunk_size=vocabulary_chunk_size,
            )
            for metric, value in comparison.items():
                metric_totals["nodewise_dense_fused"][metric] += value
            current_dense_static = _execution_fields(
                dense,
                _DENSE_STATIC_FIELDS,
                label="nodewise dense",
            )
            if dense_static is None:
                dense_static = current_dense_static
            elif dense_static != current_dense_static:
                raise RuntimeError(
                    "nodewise dense static accounting changed by batch"
                )
            _accumulate_logical(
                dense_logical_totals,
                dense,
                _DENSE_LOGICAL_FIELDS,
                label="nodewise dense",
            )
            if getattr(dense, "valid_tokens", None) != valid_tokens:
                raise RuntimeError(
                    "nodewise dense and graph valid-token counts differ"
                )

    if supervised_tokens <= 0:
        raise ValueError("evaluation stream has no supervised tokens")
    if (
        graph_static["interacting_graph"]["graph_node_count"]
        != graph_static["edgeless_graph"]["graph_node_count"]
        or graph_static["interacting_graph"]["fragment_count"]
        != graph_static["edgeless_graph"]["fragment_count"]
    ):
        raise RuntimeError("interacting and edgeless physical scopes differ")
    if graph_static["matched_deletion"] != graph_static["interacting_graph"]:
        raise RuntimeError(
            "interacting generated/deletion static accounting differs"
        )
    if (
        graph_logical_totals["matched_deletion"][
            "logical_executed_modal_graph_macs"
        ]
        != 0
        or graph_logical_totals["matched_deletion"][
            "logical_executed_modal_graph_additions"
        ]
        != 0
        or graph_peak_live_width["matched_deletion"] != 0
    ):
        raise RuntimeError("matched deletion executed modal graph work")
    for field in (
        "replacement_scope",
        "replaced_layer_count",
        "fragment_count",
        "removed_mode_count",
        "source_whole_model_learned_parameters",
        "native_removed_learned_parameters",
    ):
        if (
            graph_static["interacting_graph"][field]
            != graph_static["edgeless_graph"][field]
        ):
            raise RuntimeError(
                "interacting and edgeless replacement accounting differs"
            )
    graph_parameter_delta = int(
        graph_static["interacting_graph"]["modal_graph_learned_parameters"]
    ) - int(
        graph_static["edgeless_graph"]["modal_graph_learned_parameters"]
    )
    candidate_parameter_delta = int(
        graph_static["interacting_graph"][
            "candidate_whole_model_learned_parameters"
        ]
    ) - int(
        graph_static["edgeless_graph"][
            "candidate_whole_model_learned_parameters"
        ]
    )
    if graph_parameter_delta < 0 or (
        graph_parameter_delta != candidate_parameter_delta
    ):
        raise RuntimeError("interaction parameter accounting is inconsistent")
    if dense_static is not None:
        for field in (
            "replacement_scope",
            "replaced_layer_count",
            "removed_mode_count",
            "source_whole_model_learned_parameters",
            "native_removed_learned_parameters",
        ):
            if dense_static[field] != graph_static["edgeless_graph"][field]:
                raise RuntimeError(
                    "nodewise dense and edgeless replacement scopes differ"
                )

    native_nll = native_nll_sum / supervised_tokens
    conditions: dict[str, object] = {}
    for name in condition_names:
        totals = metric_totals[name]
        nll = float(totals["nll_sum"]) / supervised_tokens
        conditions[name] = {
            "nll_per_token": nll,
            "delta_nll_per_token": nll - native_nll,
            "native_to_candidate_kl_per_token": (
                float(totals["native_to_candidate_kl_sum"])
                / supervised_tokens
            ),
            "top1_agreement_to_native": (
                int(totals["top1_matches"]) / supervised_tokens
            ),
        }

    resources: dict[str, object] = {}
    for name in (
        "interacting_graph",
        "edgeless_graph",
        "matched_deletion",
    ):
        resources[name] = {
            **graph_static[name],
            **graph_logical_totals[name],
            "executed_peak_live_modal_width": graph_peak_live_width[name],
            "storage_scope": (
                "runtime_branch_interacting_graph_parameters_still_stored"
                if name == "matched_deletion"
                else "standalone_compiled_candidate"
            ),
        }
    if dense_static is not None:
        resources["nodewise_dense_fused"] = {
            **dense_static,
            **dense_logical_totals,
        }
    return {
        "execution_path": "unified_modal_generator_graph_rung",
        "assessment_role": assessment_role,
        "heldout_confirmation": (
            assessment_role == "claimed_closed_guard_assessment"
        ),
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "native": {"nll_per_token": native_nll},
        "conditions": conditions,
        "graph_comparison": {
            "node_count": len(getattr(interacting_plan, "nodes")),
            "interacting_edge_count": len(interacting_edges),
            "edgeless_edge_count": 0,
            "node_artifacts_identical": True,
            "interaction_parameter_delta": graph_parameter_delta,
            "deletion_paths_agree": True,
            "matched_deletion_resource_scope": (
                "runtime_branch_interacting_graph_parameters_still_stored"
            ),
            "deletion_equivalence_scope": "supervised_logits",
            "deletion_equivalence_atol": deletion_equivalence_atol,
            "deletion_equivalence_rtol": deletion_equivalence_rtol,
            "deletion_max_abs_logit_difference": deletion_max_abs,
            "nodewise_dense_supplied": nodewise_dense_executor is not None,
            "nodewise_dense_agrees_with_edgeless": (
                None if dense_static is None else True
            ),
            "nodewise_dense_equivalence_scope": (
                None if dense_static is None else "supervised_logits"
            ),
            "nodewise_dense_equivalence_atol": (
                None if dense_static is None else dense_equivalence_atol
            ),
            "nodewise_dense_equivalence_rtol": (
                None if dense_static is None else dense_equivalence_rtol
            ),
            "nodewise_dense_max_abs_logit_difference": (
                None if dense_static is None else dense_max_abs
            ),
        },
        "resource_accounting": resources,
        "latency_or_kernel_speed_claim": False,
    }
