"""Runnable capture-driven pilot for parts-first ordered Gemma supermodes.

The live-model boundary is deliberately narrow.  A model-facing bridge emits
an authenticated :class:`PartsFirstCaptureBundle` containing per-row response
coordinates for a finite subset of physical atom *occurrences*.  This module
then owns the scientific role firewall:

* A alone fits a deterministic chart atlas, same-boundary atom groups, stage
  thresholds, and an ordered candidate catalog;
* B evaluates that frozen catalog against membership- and row-shuffled
  controls without refitting or re-ranking on B; and
* exactly one C finite intervention is requested for the first A-ranked path
  that passes B.

The capture axis is validated against the exact 37,576-atom catalog.  A whole
layer cannot appear as a response coordinate.  Atom emissions are summed only
inside a same-boundary stage; a path activation is an ordered conjunction of
stage activations, never a cross-boundary vector sum.

Importing this module does not load Gemma, open prompt data, or run a model.
The CLI accepts either a frozen capture JSON or ``--demo``.  The demo is a
deterministic orchestration smoke test and is never reported as model evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tempfile
from typing import Final

from .gemma3_parts_first_supermode_protocol import (
    MODEL_ID,
    SOURCE_MODEL_FINGERPRINT_SHA256,
    build_parts_first_supermode_protocol,
)
from .gemma3_parts_first_supermodes import (
    AtomGroupDraft,
    BoundaryKind,
    ExecutionBoundary,
    Gemma3PartsCatalog,
    PhysicalAtomFamily,
    SupermodePathSpec,
    SupermodeStepSpec,
    build_atom_group_sidecar,
    build_gemma3_parts_catalog,
    validate_supermode_path,
)


CAPTURE_SCHEMA: Final = "fisher_graph.gemma3_parts_first.capture.v1"
REPORT_SCHEMA: Final = "fisher_graph.gemma3_parts_first.pilot_report.v1"
FORMAT_VERSION: Final = 1
CHART_COUNT_SEARCH_POLICY: Final = (
    "largest_supported_k_descending_with_frozen_support_gates"
)
ROLE_ORDER: Final = ("A_discovery", "B_confirmation", "C_intervention")
FAMILY_ORDER: Final = (
    PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
    PhysicalAtomFamily.ATTENTION_QO_HEAD,
    PhysicalAtomFamily.MLP_CHANNEL,
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/parts-first-supermode-pilot-v1.json"
)

_CAPTURE_HASH_DOMAIN = b"fisher_graph:parts-first-capture:v1\0"
_ROLE_HASH_DOMAIN = b"fisher_graph:parts-first-capture-role:v1\0"
_CHART_HASH_DOMAIN = b"fisher_graph:parts-first-chart-atlas:v1\0"
_CANDIDATE_HASH_DOMAIN = b"fisher_graph:parts-first-candidate:v1\0"
_CATALOG_HASH_DOMAIN = b"fisher_graph:parts-first-candidate-catalog:v1\0"
_EVIDENCE_HASH_DOMAIN = b"fisher_graph:parts-first-intervention:v1\0"
_REPORT_HASH_DOMAIN = b"fisher_graph:parts-first-pilot-report:v1\0"
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


__all__ = [
    "CAPTURE_SCHEMA",
    "CHART_COUNT_SEARCH_POLICY",
    "DEFAULT_OUTPUT",
    "FORMAT_VERSION",
    "REPORT_SCHEMA",
    "AtomResponseAxisEntry",
    "CaptureRole",
    "CaptureRow",
    "DEMO_EXACT_CONTROL_EVIDENCE_SHA256",
    "DEMO_NATIVE_GATE_RECEIPT_SHA256",
    "DEMO_SCORE_OBJECTIVE_ID",
    "DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256",
    "DEMO_SOURCE_MODEL_FINGERPRINT_SHA256",
    "DEMO_CHART_SURFACE_ID",
    "ExactLiftCaptureReceipt",
    "FiniteInterventionEvidence",
    "FiniteInterventionRunner",
    "FrozenChartAtlas",
    "FrozenSupermodeArtifact",
    "ModelCaptureLoader",
    "MODEL_CHART_SURFACE_ID",
    "MODEL_SCORE_OBJECTIVE_ID",
    "PINNED_MODEL_FAMILY_IDS",
    "PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256",
    "PartsFirstCaptureBundle",
    "PartsFirstPilotConfig",
    "PartsFirstPilotReport",
    "build_demo_capture_bundle",
    "load_parts_first_capture_bundle",
    "main",
    "preflight_parts_first_capture_bundle",
    "run_gemma3_parts_first_supermode_experiment",
    "run_parts_first_supermode_from_captures",
    "validate_parts_first_supermode_report",
    "write_parts_first_capture_bundle",
    "write_parts_first_supermode_report",
]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes = b"") -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


DEMO_SOURCE_MODEL_FINGERPRINT_SHA256: Final = _json_sha256(
    "deterministic_parts_first_demo_model_v1"
)
DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256: Final = _json_sha256(
    "deterministic_parts_first_demo_execution_v1"
)
DEMO_EXACT_CONTROL_EVIDENCE_SHA256: Final = _json_sha256(
    "deterministic_parts_first_demo_exact_control_evidence_v1"
)
DEMO_NATIVE_GATE_RECEIPT_SHA256: Final = _json_sha256(
    "deterministic_parts_first_demo_native_gate_receipt_v1"
)
PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256: Final = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
MODEL_SCORE_OBJECTIVE_ID: Final = (
    "correct_choice_next_token_nll_sum_per_independent_example"
)
MODEL_CHART_SURFACE_ID: Final = (
    "terminal.rmsnorm.output_at_objective_query"
)
DEMO_SCORE_OBJECTIVE_ID: Final = "deterministic_demo_scalar_score"
DEMO_CHART_SURFACE_ID: Final = "deterministic_demo_terminal_state_coordinates"
PINNED_MODEL_FAMILY_IDS: Final = (
    "category_completion",
    "integer_addition",
    "integer_comparison",
    "integer_subtraction",
    "lexical_antonym",
    "world_capitals",
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} field inventory is invalid")


def _boundary_from_execution_ordinal(ordinal: int) -> ExecutionBoundary:
    _require_int(ordinal, label="execution_ordinal")
    if ordinal == 0:
        return ExecutionBoundary.lexical_ingress()
    if ordinal == 37:
        return ExecutionBoundary.terminal_norm()
    if ordinal == 38:
        return ExecutionBoundary.lexical_readout()
    if ordinal >= 39:
        raise ValueError("execution ordinal is outside the 39-event schedule")
    if ordinal % 2:
        return ExecutionBoundary.attention((ordinal - 1) // 2)
    return ExecutionBoundary.mlp((ordinal - 2) // 2)


def _row_ids_sha256(rows: Sequence["CaptureRow"]) -> str:
    return _json_sha256(
        tuple(row.row_id for row in rows),
        domain=_ROLE_HASH_DOMAIN,
    )


@lru_cache(maxsize=1)
def _canonical_catalog() -> Gemma3PartsCatalog:
    return build_gemma3_parts_catalog()


@lru_cache(maxsize=1)
def _canonical_complete_occurrence_coordinates() -> tuple[tuple[int, int], ...]:
    """Return all 38,216 atom occurrences in native event/global order."""

    catalog = _canonical_catalog()
    lexical = catalog.family_spec(
        PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN
    )
    attention = catalog.family_spec(PhysicalAtomFamily.ATTENTION_QO_HEAD)
    mlp = catalog.family_spec(PhysicalAtomFamily.MLP_CHANNEL)
    coordinates: list[tuple[int, int]] = [
        (ordinal, 0)
        for ordinal in range(
            lexical.global_ordinal_start,
            lexical.global_ordinal_stop,
        )
    ]
    for layer in range(18):
        attention_start = attention.global_ordinal_start + 4 * layer
        coordinates.extend(
            (ordinal, 1 + 2 * layer)
            for ordinal in range(attention_start, attention_start + 4)
        )
        mlp_start = mlp.global_ordinal_start + 2_048 * layer
        coordinates.extend(
            (ordinal, 2 + 2 * layer)
            for ordinal in range(mlp_start, mlp_start + 2_048)
        )
    coordinates.extend(
        (ordinal, 38)
        for ordinal in range(
            lexical.global_ordinal_start,
            lexical.global_ordinal_stop,
        )
    )
    if len(coordinates) != 38_216:
        raise AssertionError("canonical parts-first occurrence inventory drifted")
    return tuple(coordinates)


@dataclass(frozen=True, slots=True)
class AtomResponseAxisEntry:
    """One physical atom invoked at one exact additive boundary."""

    global_atom_ordinal: int
    execution_ordinal: int

    def __post_init__(self) -> None:
        _require_int(self.global_atom_ordinal, label="global_atom_ordinal")
        boundary = _boundary_from_execution_ordinal(self.execution_ordinal)
        if boundary.boundary_kind is BoundaryKind.TERMINAL_NORM:
            raise ValueError("terminal norm is shared substrate, not an atom axis")
        catalog = _canonical_catalog()
        atom = catalog.atom_id_from_global_ordinal(self.global_atom_ordinal)
        catalog.occurrence_at(atom, boundary)

    @property
    def boundary(self) -> ExecutionBoundary:
        return _boundary_from_execution_ordinal(self.execution_ordinal)

    @property
    def family(self) -> PhysicalAtomFamily:
        return _canonical_catalog().atom_id_from_global_ordinal(
            self.global_atom_ordinal
        ).family

    def to_dict(self) -> dict[str, object]:
        return {
            "global_atom_ordinal": self.global_atom_ordinal,
            "execution_ordinal": self.execution_ordinal,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AtomResponseAxisEntry":
        _exact_keys(
            value,
            {"global_atom_ordinal", "execution_ordinal"},
            label="response axis entry",
        )
        return cls(
            global_atom_ordinal=_require_int(
                value["global_atom_ordinal"],
                label="global_atom_ordinal",
            ),
            execution_ordinal=_require_int(
                value["execution_ordinal"],
                label="execution_ordinal",
            ),
        )


@dataclass(frozen=True, slots=True)
class CaptureRow:
    """A prompt-position row with chart features and atom score responses."""

    row_id: str
    family_id: str
    chart_coordinates: tuple[float, ...]
    target_score: float
    atom_responses: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_id(self.row_id, label="row_id")
        _require_id(self.family_id, label="family_id")
        if not self.chart_coordinates or not self.atom_responses:
            raise ValueError("capture rows require chart and atom coordinates")
        object.__setattr__(
            self,
            "chart_coordinates",
            tuple(
                _finite_float(item, label="chart coordinate")
                for item in self.chart_coordinates
            ),
        )
        object.__setattr__(
            self,
            "target_score",
            _finite_float(self.target_score, label="target_score"),
        )
        object.__setattr__(
            self,
            "atom_responses",
            tuple(
                _finite_float(item, label="atom response")
                for item in self.atom_responses
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "family_id": self.family_id,
            "chart_coordinates": list(self.chart_coordinates),
            "target_score": self.target_score,
            "atom_responses": list(self.atom_responses),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CaptureRow":
        _exact_keys(
            value,
            {
                "row_id",
                "family_id",
                "chart_coordinates",
                "target_score",
                "atom_responses",
            },
            label="capture row",
        )
        chart = value["chart_coordinates"]
        responses = value["atom_responses"]
        if not isinstance(chart, list) or not isinstance(responses, list):
            raise ValueError("capture row coordinate arrays are invalid")
        return cls(
            row_id=_require_id(value["row_id"], label="row_id"),
            family_id=_require_id(value["family_id"], label="family_id"),
            chart_coordinates=tuple(
                _finite_float(item, label="chart coordinate") for item in chart
            ),
            target_score=_finite_float(
                value["target_score"],
                label="target_score",
            ),
            atom_responses=tuple(
                _finite_float(item, label="atom response") for item in responses
            ),
        )


@dataclass(frozen=True, slots=True)
class CaptureRole:
    role: str
    rows: tuple[CaptureRow, ...]
    exact_observer_passed: bool

    def __post_init__(self) -> None:
        if self.role not in ROLE_ORDER:
            raise ValueError("capture role is not preregistered")
        if len(self.rows) < 4 or any(
            not isinstance(row, CaptureRow) for row in self.rows
        ):
            raise ValueError("each capture role requires at least four rows")
        row_ids = tuple(row.row_id for row in self.rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("capture role row IDs must be unique")
        chart_widths = {len(row.chart_coordinates) for row in self.rows}
        response_widths = {len(row.atom_responses) for row in self.rows}
        if len(chart_widths) != 1 or len(response_widths) != 1:
            raise ValueError("capture role row widths must be constant")
        if type(self.exact_observer_passed) is not bool:
            raise TypeError("exact_observer_passed must be bool")

    @property
    def row_ids_sha256(self) -> str:
        return _row_ids_sha256(self.rows)

    @property
    def family_row_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.family_id] = counts.get(row.family_id, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def role_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_ROLE_HASH_DOMAIN)

    def _payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "rows": [row.to_dict() for row in self.rows],
            "exact_observer_passed": self.exact_observer_passed,
            "row_ids_sha256": self.row_ids_sha256,
            "family_row_counts": self.family_row_counts,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "role_sha256": self.role_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CaptureRole":
        _exact_keys(
            value,
            {
                "role",
                "rows",
                "exact_observer_passed",
                "row_ids_sha256",
                "family_row_counts",
                "role_sha256",
            },
            label="capture role",
        )
        rows_value = value["rows"]
        if not isinstance(rows_value, list) or not all(
            isinstance(item, Mapping) for item in rows_value
        ):
            raise ValueError("capture role rows are invalid")
        role = cls(
            role=str(value["role"]),
            rows=tuple(CaptureRow.from_dict(item) for item in rows_value),
            exact_observer_passed=value["exact_observer_passed"],  # type: ignore[arg-type]
        )
        if (
            value["row_ids_sha256"] != role.row_ids_sha256
            or value["family_row_counts"] != role.family_row_counts
            or value["role_sha256"] != role.role_sha256
        ):
            raise ValueError("capture role hash does not close")
        return role


@dataclass(frozen=True, slots=True)
class ExactLiftCaptureReceipt:
    """Bridge assertion that the response sidecar observed the exact lift."""

    model_facing: bool
    exact_39_event_order_passed: bool
    all_37576_physical_atoms_closed: bool
    all_shared_substrate_slices_closed: bool
    tied_lexical_storage_identity_passed: bool
    source_state_unchanged: bool
    exact_control_evidence_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "model_facing",
            "exact_39_event_order_passed",
            "all_37576_physical_atoms_closed",
            "all_shared_substrate_slices_closed",
            "tied_lexical_storage_identity_passed",
            "source_state_unchanged",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        _require_sha256(
            self.exact_control_evidence_sha256,
            label="exact_control_evidence_sha256",
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.model_facing
            and self.exact_39_event_order_passed
            and self.all_37576_physical_atoms_closed
            and self.all_shared_substrate_slices_closed
            and self.tied_lexical_storage_identity_passed
            and self.source_state_unchanged
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_facing": self.model_facing,
            "exact_39_event_order_passed": self.exact_39_event_order_passed,
            "all_37576_physical_atoms_closed": (
                self.all_37576_physical_atoms_closed
            ),
            "all_shared_substrate_slices_closed": (
                self.all_shared_substrate_slices_closed
            ),
            "tied_lexical_storage_identity_passed": (
                self.tied_lexical_storage_identity_passed
            ),
            "source_state_unchanged": self.source_state_unchanged,
            "exact_control_evidence_sha256": (
                self.exact_control_evidence_sha256
            ),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> "ExactLiftCaptureReceipt":
        fields = {
            "model_facing",
            "exact_39_event_order_passed",
            "all_37576_physical_atoms_closed",
            "all_shared_substrate_slices_closed",
            "tied_lexical_storage_identity_passed",
            "source_state_unchanged",
            "exact_control_evidence_sha256",
            "passed",
        }
        _exact_keys(value, fields, label="exact lift capture receipt")
        receipt = cls(
            model_facing=value["model_facing"],  # type: ignore[arg-type]
            exact_39_event_order_passed=value[  # type: ignore[arg-type]
                "exact_39_event_order_passed"
            ],
            all_37576_physical_atoms_closed=value[  # type: ignore[arg-type]
                "all_37576_physical_atoms_closed"
            ],
            all_shared_substrate_slices_closed=value[  # type: ignore[arg-type]
                "all_shared_substrate_slices_closed"
            ],
            tied_lexical_storage_identity_passed=value[  # type: ignore[arg-type]
                "tied_lexical_storage_identity_passed"
            ],
            source_state_unchanged=value[  # type: ignore[arg-type]
                "source_state_unchanged"
            ],
            exact_control_evidence_sha256=str(
                value["exact_control_evidence_sha256"]
            ),
        )
        if value["passed"] is not receipt.passed:
            raise ValueError("exact lift capture decision does not close")
        return receipt


@dataclass(frozen=True, slots=True)
class FiniteInterventionEvidence:
    """One isolated finite gate result for one already-frozen path."""

    candidate_sha256: str
    c_row_ids_sha256: str
    gate_alpha: float
    baseline_scores: tuple[float, ...]
    intervened_scores: tuple[float, ...]
    first_order_predicted_score_deltas: tuple[float, ...]
    baseline_chart_ids: tuple[str, ...]
    online_chart_ids_after: tuple[str, ...]
    native_noop_max_abs_error: float
    finite_intervention_candidate_count: int
    finite_intervention_forward_count: int
    executor_kind: str
    native_gate_execution_receipt_sha256: str
    source_model_fingerprint_before: str
    source_model_fingerprint_after: str
    source_execution_fingerprint_before: str
    source_execution_fingerprint_after: str
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_sha256, label="candidate_sha256")
        _require_sha256(self.c_row_ids_sha256, label="c_row_ids_sha256")
        alpha = _finite_float(self.gate_alpha, label="gate_alpha")
        if alpha == 1.0:
            raise ValueError("finite intervention gate alpha cannot be one")
        object.__setattr__(self, "gate_alpha", alpha)
        arrays = (
            self.baseline_scores,
            self.intervened_scores,
            self.first_order_predicted_score_deltas,
        )
        if not arrays[0] or len({len(value) for value in arrays}) != 1:
            raise ValueError("finite intervention score arrays do not align")
        for name in (
            "baseline_scores",
            "intervened_scores",
            "first_order_predicted_score_deltas",
        ):
            object.__setattr__(
                self,
                name,
                tuple(
                    _finite_float(item, label=name)
                    for item in getattr(self, name)
                ),
            )
        if (
            len(self.baseline_chart_ids) != len(arrays[0])
            or len(self.online_chart_ids_after) != len(arrays[0])
        ):
            raise ValueError("finite intervention chart arrays do not align")
        for chart_id in (*self.baseline_chart_ids, *self.online_chart_ids_after):
            _require_id(chart_id, label="chart_id")
        noop = _finite_float(
            self.native_noop_max_abs_error,
            label="native_noop_max_abs_error",
        )
        if noop < 0:
            raise ValueError("native no-op error cannot be negative")
        object.__setattr__(self, "native_noop_max_abs_error", noop)
        if self.finite_intervention_candidate_count != 1:
            raise ValueError("exactly one finite intervention candidate is required")
        _require_int(
            self.finite_intervention_forward_count,
            label="finite_intervention_forward_count",
            minimum=1,
        )
        if self.finite_intervention_forward_count != len(self.baseline_scores):
            raise ValueError(
                "finite-intervention forward count must equal baseline row count"
            )
        if self.executor_kind not in {
            "live_model_bridge",
            "frozen_capture_replay",
            "deterministic_demo",
        }:
            raise ValueError("finite intervention executor kind is invalid")
        for name in (
            "native_gate_execution_receipt_sha256",
            "source_model_fingerprint_before",
            "source_model_fingerprint_after",
            "source_execution_fingerprint_before",
            "source_execution_fingerprint_after",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.executor_kind in {"live_model_bridge", "frozen_capture_replay"}:
            if (
                self.source_model_fingerprint_before
                != self.source_model_fingerprint_after
                or self.source_execution_fingerprint_before
                != self.source_execution_fingerprint_after
            ):
                raise ValueError(
                    "finite gate execution changed source model or execution state"
                )
        elif (
            self.native_gate_execution_receipt_sha256
            != DEMO_NATIVE_GATE_RECEIPT_SHA256
            or self.source_model_fingerprint_before
            != DEMO_SOURCE_MODEL_FINGERPRINT_SHA256
            or self.source_model_fingerprint_after
            != DEMO_SOURCE_MODEL_FINGERPRINT_SHA256
            or self.source_execution_fingerprint_before
            != DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
            or self.source_execution_fingerprint_after
            != DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ):
            raise ValueError("deterministic demo evidence sentinels drifted")
        computed = _json_sha256(self._payload(), domain=_EVIDENCE_HASH_DOMAIN)
        if self.evidence_sha256 == "":
            object.__setattr__(self, "evidence_sha256", computed)
        elif self.evidence_sha256 != computed:
            raise ValueError("finite intervention evidence hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "c_row_ids_sha256": self.c_row_ids_sha256,
            "gate_alpha": self.gate_alpha,
            "baseline_scores": list(self.baseline_scores),
            "intervened_scores": list(self.intervened_scores),
            "first_order_predicted_score_deltas": list(
                self.first_order_predicted_score_deltas
            ),
            "baseline_chart_ids": list(self.baseline_chart_ids),
            "online_chart_ids_after": list(self.online_chart_ids_after),
            "native_noop_max_abs_error": self.native_noop_max_abs_error,
            "finite_intervention_candidate_count": (
                self.finite_intervention_candidate_count
            ),
            "finite_intervention_forward_count": (
                self.finite_intervention_forward_count
            ),
            "executor_kind": self.executor_kind,
            "native_gate_execution_receipt_sha256": (
                self.native_gate_execution_receipt_sha256
            ),
            "source_model_fingerprint_before": (
                self.source_model_fingerprint_before
            ),
            "source_model_fingerprint_after": (
                self.source_model_fingerprint_after
            ),
            "source_execution_fingerprint_before": (
                self.source_execution_fingerprint_before
            ),
            "source_execution_fingerprint_after": (
                self.source_execution_fingerprint_after
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "evidence_sha256": self.evidence_sha256}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> "FiniteInterventionEvidence":
        fields = {
            "candidate_sha256",
            "c_row_ids_sha256",
            "gate_alpha",
            "baseline_scores",
            "intervened_scores",
            "first_order_predicted_score_deltas",
            "baseline_chart_ids",
            "online_chart_ids_after",
            "native_noop_max_abs_error",
            "finite_intervention_candidate_count",
            "finite_intervention_forward_count",
            "executor_kind",
            "native_gate_execution_receipt_sha256",
            "source_model_fingerprint_before",
            "source_model_fingerprint_after",
            "source_execution_fingerprint_before",
            "source_execution_fingerprint_after",
            "evidence_sha256",
        }
        _exact_keys(value, fields, label="finite intervention evidence")
        sequence_fields = (
            "baseline_scores",
            "intervened_scores",
            "first_order_predicted_score_deltas",
            "baseline_chart_ids",
            "online_chart_ids_after",
        )
        if any(not isinstance(value[name], list) for name in sequence_fields):
            raise ValueError("finite intervention arrays are invalid")
        return cls(
            candidate_sha256=str(value["candidate_sha256"]),
            c_row_ids_sha256=str(value["c_row_ids_sha256"]),
            gate_alpha=_finite_float(value["gate_alpha"], label="gate_alpha"),
            baseline_scores=tuple(value["baseline_scores"]),  # type: ignore[arg-type]
            intervened_scores=tuple(value["intervened_scores"]),  # type: ignore[arg-type]
            first_order_predicted_score_deltas=tuple(  # type: ignore[arg-type]
                value["first_order_predicted_score_deltas"]
            ),
            baseline_chart_ids=tuple(value["baseline_chart_ids"]),  # type: ignore[arg-type]
            online_chart_ids_after=tuple(  # type: ignore[arg-type]
                value["online_chart_ids_after"]
            ),
            native_noop_max_abs_error=_finite_float(
                value["native_noop_max_abs_error"],
                label="native_noop_max_abs_error",
            ),
            finite_intervention_candidate_count=_require_int(
                value["finite_intervention_candidate_count"],
                label="finite_intervention_candidate_count",
                minimum=1,
            ),
            finite_intervention_forward_count=_require_int(
                value["finite_intervention_forward_count"],
                label="finite_intervention_forward_count",
                minimum=1,
            ),
            executor_kind=str(value["executor_kind"]),
            native_gate_execution_receipt_sha256=str(
                value["native_gate_execution_receipt_sha256"]
            ),
            source_model_fingerprint_before=str(
                value["source_model_fingerprint_before"]
            ),
            source_model_fingerprint_after=str(
                value["source_model_fingerprint_after"]
            ),
            source_execution_fingerprint_before=str(
                value["source_execution_fingerprint_before"]
            ),
            source_execution_fingerprint_after=str(
                value["source_execution_fingerprint_after"]
            ),
            evidence_sha256=str(value["evidence_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PartsFirstCaptureBundle:
    source_kind: str
    model_id: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    score_objective_id: str
    chart_surface_id: str
    protocol_sha256: str
    atom_universe_sha256: str
    response_axis: tuple[AtomResponseAxisEntry, ...]
    roles: tuple[CaptureRole, ...]
    exact_lift_receipt: ExactLiftCaptureReceipt
    intervention_evidence: FiniteInterventionEvidence | None = None
    schema: str = CAPTURE_SCHEMA
    format_version: int = FORMAT_VERSION
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    capture_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != CAPTURE_SCHEMA or self.format_version != FORMAT_VERSION:
            raise ValueError("parts-first capture header is invalid")
        if self.source_kind not in {"model_capture", "deterministic_demo"}:
            raise ValueError("capture source kind is invalid")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("capture model_id must be nonempty")
        _require_sha256(
            self.source_model_fingerprint_sha256,
            label="source_model_fingerprint_sha256",
        )
        _require_id(self.score_objective_id, label="score_objective_id")
        _require_id(self.chart_surface_id, label="chart_surface_id")
        _require_sha256(
            self.source_execution_fingerprint_sha256,
            label="source_execution_fingerprint_sha256",
        )
        protocol = build_parts_first_supermode_protocol()
        catalog = _canonical_catalog()
        if self.protocol_sha256 != protocol["protocol_sha256"]:
            raise ValueError("capture protocol binding drifted")
        if self.atom_universe_sha256 != catalog.catalog_sha256:
            raise ValueError("capture atom-universe binding drifted")
        if not self.response_axis or any(
            not isinstance(item, AtomResponseAxisEntry)
            for item in self.response_axis
        ):
            raise ValueError("capture response axis is empty or invalid")
        coordinates = tuple(
            (item.global_atom_ordinal, item.execution_ordinal)
            for item in self.response_axis
        )
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("capture response axis contains duplicate occurrences")
        if tuple(role.role for role in self.roles) != ROLE_ORDER:
            raise ValueError("capture roles must be complete and ordered")
        if any(not isinstance(role, CaptureRole) for role in self.roles):
            raise TypeError("capture roles are invalid")
        row_ids = [row.row_id for role in self.roles for row in role.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("A, B, and C row identities must be disjoint")
        family_inventories = tuple(
            frozenset(row.family_id for row in role.rows) for role in self.roles
        )
        if (
            len(set(family_inventories)) != 1
            or len(family_inventories[0]) != 6
        ):
            raise ValueError(
                "A, B, and C must share the frozen six-family panel inventory"
            )
        chart_widths = {
            len(row.chart_coordinates)
            for role in self.roles
            for row in role.rows
        }
        if len(chart_widths) != 1:
            raise ValueError("chart-coordinate ABI differs across roles")
        if any(
            len(row.atom_responses) != len(self.response_axis)
            for role in self.roles
            for row in role.rows
        ):
            raise ValueError("atom response width does not match its axis")
        if not isinstance(self.exact_lift_receipt, ExactLiftCaptureReceipt):
            raise TypeError("exact lift capture receipt is invalid")
        if self.source_kind == "model_capture":
            if (
                self.model_id != MODEL_ID
                or self.source_model_fingerprint_sha256
                != SOURCE_MODEL_FINGERPRINT_SHA256
                or self.source_execution_fingerprint_sha256
                != PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
                or self.score_objective_id != MODEL_SCORE_OBJECTIVE_ID
                or self.chart_surface_id != MODEL_CHART_SURFACE_ID
                or not self.exact_lift_receipt.passed
            ):
                raise ValueError(
                    "model captures require the pinned model and exact lift gate"
                )
            if not all(role.exact_observer_passed for role in self.roles):
                raise ValueError("model capture observer parity did not pass")
            if (
                self.exact_lift_receipt.exact_control_evidence_sha256
                == DEMO_EXACT_CONTROL_EVIDENCE_SHA256
            ):
                raise ValueError(
                    "model capture requires live exact-control evidence"
                )
            expected_per_family = {
                "A_discovery": 6,
                "B_confirmation": 2,
                "C_intervention": 2,
            }
            if family_inventories[0] != frozenset(PINNED_MODEL_FAMILY_IDS):
                raise ValueError("model capture family identity inventory drifted")
            if any(
                set(role.family_row_counts.values())
                != {expected_per_family[role.role]}
                for role in self.roles
            ):
                raise ValueError(
                    "model capture must bind the six-family 6/2/2 panel split"
                )
            expected_axis = _canonical_complete_occurrence_coordinates()
            if coordinates != expected_axis:
                raise ValueError(
                    "model capture requires the complete 38216-occurrence axis"
                )
        elif self.exact_lift_receipt.model_facing or any(
            role.exact_observer_passed for role in self.roles
        ):
            raise ValueError("demo captures may not impersonate model evidence")
        elif self.score_objective_id != DEMO_SCORE_OBJECTIVE_ID:
            raise ValueError("demo score objective sentinel drifted")
        elif self.chart_surface_id != DEMO_CHART_SURFACE_ID:
            raise ValueError("demo chart surface sentinel drifted")
        elif (
            self.exact_lift_receipt.exact_control_evidence_sha256
            != DEMO_EXACT_CONTROL_EVIDENCE_SHA256
        ):
            raise ValueError("demo exact-control evidence sentinel drifted")
        if self.intervention_evidence is not None and not isinstance(
            self.intervention_evidence,
            FiniteInterventionEvidence,
        ):
            raise TypeError("embedded intervention evidence is invalid")
        if self.contains_prompt_text is not False or self.contains_token_ids is not False:
            raise ValueError("capture bundles may not serialize prompts or token IDs")
        computed = _json_sha256(self._payload(), domain=_CAPTURE_HASH_DOMAIN)
        if self.capture_sha256 == "":
            object.__setattr__(self, "capture_sha256", computed)
        elif self.capture_sha256 != computed:
            raise ValueError("parts-first capture hash mismatch")

    @property
    def role_by_id(self) -> dict[str, CaptureRole]:
        return {role.role: role for role in self.roles}

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_kind": self.source_kind,
            "model_id": self.model_id,
            "source_model_fingerprint_sha256": (
                self.source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "score_objective_id": self.score_objective_id,
            "chart_surface_id": self.chart_surface_id,
            "protocol_sha256": self.protocol_sha256,
            "atom_universe_sha256": self.atom_universe_sha256,
            "response_axis": [item.to_dict() for item in self.response_axis],
            "roles": [role.to_dict() for role in self.roles],
            "exact_lift_receipt": self.exact_lift_receipt.to_dict(),
            "intervention_evidence": (
                None
                if self.intervention_evidence is None
                else self.intervention_evidence.to_dict()
            ),
            "contains_prompt_text": self.contains_prompt_text,
            "contains_token_ids": self.contains_token_ids,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "capture_sha256": self.capture_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PartsFirstCaptureBundle":
        fields = {
            "schema",
            "format_version",
            "source_kind",
            "model_id",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "score_objective_id",
            "chart_surface_id",
            "protocol_sha256",
            "atom_universe_sha256",
            "response_axis",
            "roles",
            "exact_lift_receipt",
            "intervention_evidence",
            "contains_prompt_text",
            "contains_token_ids",
            "capture_sha256",
        }
        _exact_keys(value, fields, label="parts-first capture bundle")
        axis = value["response_axis"]
        roles = value["roles"]
        exact = value["exact_lift_receipt"]
        evidence = value["intervention_evidence"]
        if (
            not isinstance(axis, list)
            or not all(isinstance(item, Mapping) for item in axis)
            or not isinstance(roles, list)
            or not all(isinstance(item, Mapping) for item in roles)
            or not isinstance(exact, Mapping)
            or (evidence is not None and not isinstance(evidence, Mapping))
        ):
            raise ValueError("parts-first capture bundle has invalid objects")
        return cls(
            schema=str(value["schema"]),
            format_version=_require_int(
                value["format_version"],
                label="format_version",
                minimum=1,
            ),
            source_kind=str(value["source_kind"]),
            model_id=str(value["model_id"]),
            source_model_fingerprint_sha256=str(
                value["source_model_fingerprint_sha256"]
            ),
            source_execution_fingerprint_sha256=str(
                value["source_execution_fingerprint_sha256"]
            ),
            score_objective_id=str(value["score_objective_id"]),
            chart_surface_id=str(value["chart_surface_id"]),
            protocol_sha256=str(value["protocol_sha256"]),
            atom_universe_sha256=str(value["atom_universe_sha256"]),
            response_axis=tuple(
                AtomResponseAxisEntry.from_dict(item) for item in axis
            ),
            roles=tuple(CaptureRole.from_dict(item) for item in roles),
            exact_lift_receipt=ExactLiftCaptureReceipt.from_dict(exact),
            intervention_evidence=(
                None
                if evidence is None
                else FiniteInterventionEvidence.from_dict(evidence)
            ),
            contains_prompt_text=value["contains_prompt_text"],  # type: ignore[arg-type]
            contains_token_ids=value["contains_token_ids"],  # type: ignore[arg-type]
            capture_sha256=str(value["capture_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class PartsFirstPilotConfig:
    deterministic_seed: int = 1_729
    chart_count: int = 2
    chart_lloyd_iterations: int = 16
    group_size: int = 2
    boundary_limit_per_family: int = 3
    maximum_A_candidates: int = 32
    minimum_rows_per_chart: int = 4
    minimum_families_per_chart: int = 3
    minimum_active_rows: int = 2
    minimum_inactive_rows: int = 2
    minimum_absolute_A_effect: float = 0.5
    minimum_absolute_B_effect: float = 0.4
    minimum_B_control_margin: float = 0.05
    gate_alpha: float = 0.5
    maximum_C_first_order_relative_error: float = 0.05
    minimum_C_mean_absolute_delta: float = 0.01
    minimum_C_router_stability: float = 0.75
    maximum_native_noop_error: float = 0.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("deterministic_seed", 0),
            ("chart_count", 1),
            ("chart_lloyd_iterations", 1),
            ("group_size", 1),
            ("boundary_limit_per_family", 1),
            ("maximum_A_candidates", 1),
            ("minimum_rows_per_chart", 2),
            ("minimum_families_per_chart", 1),
            ("minimum_active_rows", 1),
            ("minimum_inactive_rows", 1),
        ):
            _require_int(getattr(self, name), label=name, minimum=minimum)
        for name in (
            "minimum_absolute_A_effect",
            "minimum_absolute_B_effect",
            "minimum_B_control_margin",
            "maximum_C_first_order_relative_error",
            "minimum_C_mean_absolute_delta",
            "minimum_C_router_stability",
            "maximum_native_noop_error",
        ):
            value = _finite_float(getattr(self, name), label=name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if self.minimum_C_router_stability > 1:
            raise ValueError("minimum_C_router_stability cannot exceed one")
        if self.minimum_families_per_chart > 6:
            raise ValueError("minimum_families_per_chart cannot exceed six")
        alpha = _finite_float(self.gate_alpha, label="gate_alpha")
        if alpha == 1:
            raise ValueError("gate_alpha must perform a finite intervention")
        object.__setattr__(self, "gate_alpha", alpha)

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> "PartsFirstPilotConfig":
        fields = set(cls.__dataclass_fields__)
        _exact_keys(value, fields, label="parts-first pilot configuration")
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FrozenChartAtlas:
    centers: tuple[tuple[float, ...], ...]
    A_role_sha256: str
    configured_maximum_chart_count: int
    selected_chart_count: int
    minimum_rows_per_chart: int
    minimum_families_per_chart: int
    chart_count_search_policy: str
    attempted_chart_counts: tuple[int, ...]
    atlas_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.centers or len({len(item) for item in self.centers}) != 1:
            raise ValueError("chart centers must be nonempty and rectangular")
        for center in self.centers:
            for value in center:
                _finite_float(value, label="chart center")
        _require_sha256(self.A_role_sha256, label="A_role_sha256")
        _require_int(
            self.configured_maximum_chart_count,
            label="configured_maximum_chart_count",
            minimum=1,
        )
        _require_int(
            self.selected_chart_count,
            label="selected_chart_count",
            minimum=1,
        )
        _require_int(
            self.minimum_rows_per_chart,
            label="minimum_rows_per_chart",
            minimum=2,
        )
        _require_int(
            self.minimum_families_per_chart,
            label="minimum_families_per_chart",
            minimum=1,
        )
        if self.selected_chart_count != len(self.centers):
            raise ValueError("selected chart count does not match atlas centers")
        if self.selected_chart_count > self.configured_maximum_chart_count:
            raise ValueError("selected chart count exceeds configured maximum")
        if self.chart_count_search_policy != CHART_COUNT_SEARCH_POLICY:
            raise ValueError("chart count search policy drifted")
        expected_attempts = tuple(
            range(
                self.configured_maximum_chart_count,
                self.selected_chart_count - 1,
                -1,
            )
        )
        if self.attempted_chart_counts != expected_attempts:
            raise ValueError("chart count search attempts do not close")
        computed = _json_sha256(self._payload(), domain=_CHART_HASH_DOMAIN)
        if self.atlas_sha256 == "":
            object.__setattr__(self, "atlas_sha256", computed)
        elif self.atlas_sha256 != computed:
            raise ValueError("frozen chart atlas hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "centers": [list(center) for center in self.centers],
            "A_role_sha256": self.A_role_sha256,
            **self.selection_dict(),
        }

    def selection_dict(self) -> dict[str, object]:
        return {
            "configured_maximum_chart_count": (
                self.configured_maximum_chart_count
            ),
            "selected_chart_count": self.selected_chart_count,
            "minimum_rows_per_chart": self.minimum_rows_per_chart,
            "minimum_families_per_chart": self.minimum_families_per_chart,
            "chart_count_search_policy": self.chart_count_search_policy,
            "attempted_chart_counts": list(self.attempted_chart_counts),
        }

    def chart_index(self, coordinates: Sequence[float]) -> int:
        if len(coordinates) != len(self.centers[0]):
            raise ValueError("chart coordinates do not match frozen atlas")
        distances = tuple(
            sum((float(a) - float(b)) ** 2 for a, b in zip(coordinates, center))
            for center in self.centers
        )
        return min(range(len(distances)), key=lambda index: (distances[index], index))

    def chart_id(self, coordinates: Sequence[float]) -> str:
        return f"chart.{self.chart_index(coordinates)}"

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "atlas_sha256": self.atlas_sha256}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> "FrozenChartAtlas":
        selection_fields = {
            "configured_maximum_chart_count",
            "selected_chart_count",
            "minimum_rows_per_chart",
            "minimum_families_per_chart",
            "chart_count_search_policy",
            "attempted_chart_counts",
        }
        _exact_keys(
            value,
            {
                "centers",
                "A_role_sha256",
                "atlas_sha256",
                *selection_fields,
            },
            label="frozen chart atlas",
        )
        centers = value["centers"]
        attempts = value["attempted_chart_counts"]
        if (
            not isinstance(centers, list)
            or not all(isinstance(center, list) for center in centers)
            or not isinstance(attempts, list)
        ):
            raise ValueError("frozen chart atlas arrays are invalid")
        return cls(
            centers=tuple(
                tuple(
                    _finite_float(item, label="chart center")
                    for item in center
                )
                for center in centers
            ),
            A_role_sha256=str(value["A_role_sha256"]),
            configured_maximum_chart_count=_require_int(
                value["configured_maximum_chart_count"],
                label="configured_maximum_chart_count",
                minimum=1,
            ),
            selected_chart_count=_require_int(
                value["selected_chart_count"],
                label="selected_chart_count",
                minimum=1,
            ),
            minimum_rows_per_chart=_require_int(
                value["minimum_rows_per_chart"],
                label="minimum_rows_per_chart",
                minimum=2,
            ),
            minimum_families_per_chart=_require_int(
                value["minimum_families_per_chart"],
                label="minimum_families_per_chart",
                minimum=1,
            ),
            chart_count_search_policy=str(
                value["chart_count_search_policy"]
            ),
            attempted_chart_counts=tuple(
                _require_int(item, label="attempted chart count", minimum=1)
                for item in attempts
            ),
            atlas_sha256=str(value["atlas_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class FrozenStage:
    family: PhysicalAtomFamily
    execution_ordinal: int
    atom_ordinals: tuple[int, ...]
    orientation: int
    activation_threshold: float
    A_stage_correlation: float

    def __post_init__(self) -> None:
        if not isinstance(self.family, PhysicalAtomFamily):
            raise TypeError("frozen stage family is invalid")
        boundary = _boundary_from_execution_ordinal(self.execution_ordinal)
        if not self.atom_ordinals or self.atom_ordinals != tuple(
            sorted(set(self.atom_ordinals))
        ):
            raise ValueError("frozen stage atoms must be sorted and unique")
        catalog = _canonical_catalog()
        for ordinal in self.atom_ordinals:
            atom = catalog.atom_id_from_global_ordinal(ordinal)
            if atom.family is not self.family:
                raise ValueError("frozen stage is not family homogeneous")
            catalog.occurrence_at(atom, boundary)
        if self.orientation not in {-1, 1}:
            raise ValueError("frozen stage orientation must be -1 or +1")
        _finite_float(self.activation_threshold, label="activation_threshold")
        _finite_float(self.A_stage_correlation, label="A_stage_correlation")

    @property
    def boundary(self) -> ExecutionBoundary:
        return _boundary_from_execution_ordinal(self.execution_ordinal)

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family.value,
            "execution_ordinal": self.execution_ordinal,
            "boundary_id": self.boundary.boundary_id,
            "atom_ordinals": list(self.atom_ordinals),
            "orientation": self.orientation,
            "activation_threshold": self.activation_threshold,
            "A_stage_correlation": self.A_stage_correlation,
            "within_boundary_operator": "additive_coalition",
        }


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    chart_id: str
    stages: tuple[FrozenStage, ...]
    A_effect: float
    A_active_rows: int
    A_inactive_rows: int
    A_rank_score: float
    candidate_sha256: str = ""

    def __post_init__(self) -> None:
        _require_id(self.chart_id, label="chart_id")
        expected_families = (
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
            PhysicalAtomFamily.ATTENTION_QO_HEAD,
            PhysicalAtomFamily.MLP_CHANNEL,
            PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
        )
        if tuple(stage.family for stage in self.stages) != expected_families:
            raise ValueError(
                "pilot paths require lexical ingress, attention, MLP, and "
                "the tied lexical readout occurrence"
            )
        ordinals = tuple(stage.execution_ordinal for stage in self.stages)
        if ordinals != tuple(sorted(set(ordinals))):
            raise ValueError("frozen candidate stages are not strictly ordered")
        ingress, attention, mlp, readout = self.stages
        if (
            ingress.boundary.boundary_kind is not BoundaryKind.LEXICAL_INGRESS
            or readout.boundary.boundary_kind is not BoundaryKind.LEXICAL_READOUT
            or attention.boundary.boundary_kind is not BoundaryKind.ATTENTION
            or mlp.boundary.boundary_kind is not BoundaryKind.MLP
            or attention.boundary.layer_ordinal is None
            or mlp.boundary.layer_ordinal is None
            or attention.boundary.layer_ordinal >= mlp.boundary.layer_ordinal
        ):
            raise ValueError(
                "pilot path requires attention before a strictly later-layer MLP"
            )
        for name in ("A_effect", "A_rank_score"):
            _finite_float(getattr(self, name), label=name)
        _require_int(self.A_active_rows, label="A_active_rows", minimum=1)
        _require_int(self.A_inactive_rows, label="A_inactive_rows", minimum=1)
        computed = _json_sha256(self._identity_payload(), domain=_CANDIDATE_HASH_DOMAIN)
        if self.candidate_sha256 == "":
            object.__setattr__(self, "candidate_sha256", computed)
        elif self.candidate_sha256 != computed:
            raise ValueError("frozen candidate hash mismatch")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "chart_id": self.chart_id,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "A_effect": self.A_effect,
            "A_active_rows": self.A_active_rows,
            "A_inactive_rows": self.A_inactive_rows,
            "A_rank_score": self.A_rank_score,
            "candidate_sha256": self.candidate_sha256,
            "cross_boundary_operator": "ordered_conjunction_not_vector_sum",
        }


@dataclass(frozen=True, slots=True)
class FrozenSupermodeArtifact:
    """Core-authenticated same-boundary groups plus their ordered path."""

    candidate: FrozenCandidate
    group_blob: bytes = field(repr=False)
    group_blob_sha256: str
    path: SupermodePathSpec

    @property
    def candidate_sha256(self) -> str:
        return self.candidate.candidate_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            **self.candidate.to_dict(),
            "group_blob_sha256": self.group_blob_sha256,
            "group_blob_byte_count": len(self.group_blob),
            "path_id": self.path.path_id,
            "atom_universe_sha256": self.path.atom_universe_sha256,
            "core_path_validated": True,
            "group_storage_credit_scalar_count": 0,
        }


@dataclass(frozen=True, slots=True)
class DiscoveryReceipt:
    A_role_sha256: str
    chart_atlas: FrozenChartAtlas
    A_family_row_counts_by_chart: tuple[
        tuple[str, tuple[tuple[str, int], ...]], ...
    ]
    candidates: tuple[FrozenSupermodeArtifact, ...]
    candidate_catalog_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "A_discovery",
            "A_role_sha256": self.A_role_sha256,
            "chart_atlas": self.chart_atlas.to_dict(),
            "chart_selection": self.chart_atlas.selection_dict(),
            "A_family_row_counts_by_chart": {
                chart_id: dict(counts)
                for chart_id, counts in self.A_family_row_counts_by_chart
            },
            "candidate_count": len(self.candidates),
            "candidates": [item.to_dict() for item in self.candidates],
            "candidate_catalog_sha256": self.candidate_catalog_sha256,
            "roles_read": ["A_discovery"],
            "B_rows_read": 0,
            "C_rows_read": 0,
            "frozen_before_B": True,
            "layers_used_as_generator_ids": False,
            "cross_boundary_vector_sum_used": False,
        }


@dataclass(frozen=True, slots=True)
class CandidateConfirmation:
    candidate_sha256: str
    B_effect: float
    B_active_rows: int
    B_inactive_rows: int
    B_family_row_counts: tuple[tuple[str, int], ...]
    shuffled_membership_effect: float
    shuffled_row_effect: float
    shuffled_row_cross_chart_source_count: int
    zero_control_effect: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "B_effect": self.B_effect,
            "B_active_rows": self.B_active_rows,
            "B_inactive_rows": self.B_inactive_rows,
            "B_family_row_counts": dict(self.B_family_row_counts),
            "shuffled_membership_effect": self.shuffled_membership_effect,
            "shuffled_row_effect": self.shuffled_row_effect,
            "shuffled_row_cross_chart_source_count": (
                self.shuffled_row_cross_chart_source_count
            ),
            "shuffled_row_control": (
                "deterministic_permutation_within_semantic_family"
            ),
            "zero_control_effect": self.zero_control_effect,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class ConfirmationReceipt:
    B_role_sha256: str
    frozen_candidate_catalog_sha256: str
    evaluations: tuple[CandidateConfirmation, ...]
    selected_candidate_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "B_confirmation",
            "B_role_sha256": self.B_role_sha256,
            "frozen_candidate_catalog_sha256": (
                self.frozen_candidate_catalog_sha256
            ),
            "candidate_evaluation_count": len(self.evaluations),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "selection_rule": "first_A_ranked_candidate_passing_frozen_B_gates",
            "A_artifact_mutated": False,
            "fit_or_refit_performed": False,
            "B_reranking_performed": False,
            "C_rows_read": 0,
        }


@dataclass(frozen=True, slots=True)
class InterventionReceipt:
    candidate_sha256: str
    evidence_sha256: str
    C_role_sha256: str
    finite_intervention_candidate_count: int
    finite_intervention_forward_count: int
    selected_chart_row_count: int
    mean_absolute_score_delta_on_selected_chart: float
    maximum_first_order_relative_error: float
    router_stability_fraction: float
    native_noop_max_abs_error: float
    executor_kind: str
    native_gate_execution_receipt_sha256: str
    source_model_fingerprint_before: str
    source_model_fingerprint_after: str
    source_execution_fingerprint_before: str
    source_execution_fingerprint_after: str
    passed: bool
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "role": "C_intervention",
            "candidate_sha256": self.candidate_sha256,
            "evidence_sha256": self.evidence_sha256,
            "C_role_sha256": self.C_role_sha256,
            "finite_intervention_candidate_count": (
                self.finite_intervention_candidate_count
            ),
            "finite_intervention_forward_count": (
                self.finite_intervention_forward_count
            ),
            "selected_chart_row_count": self.selected_chart_row_count,
            "mean_absolute_score_delta_on_selected_chart": (
                self.mean_absolute_score_delta_on_selected_chart
            ),
            "maximum_first_order_relative_error": (
                self.maximum_first_order_relative_error
            ),
            "router_stability_fraction": self.router_stability_fraction,
            "native_noop_max_abs_error": self.native_noop_max_abs_error,
            "executor_kind": self.executor_kind,
            "native_gate_execution_receipt_sha256": (
                self.native_gate_execution_receipt_sha256
            ),
            "source_model_fingerprint_before": (
                self.source_model_fingerprint_before
            ),
            "source_model_fingerprint_after": self.source_model_fingerprint_after,
            "source_execution_fingerprint_before": (
                self.source_execution_fingerprint_before
            ),
            "source_execution_fingerprint_after": (
                self.source_execution_fingerprint_after
            ),
            "primary_router_semantics": "frozen_baseline_chart_mask",
            "later_native_computation_reran_authenticated": (
                self.executor_kind == "live_model_bridge"
            ),
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class PartsFirstPilotReport:
    payload: Mapping[str, object]
    report_sha256: str = ""

    def __post_init__(self) -> None:
        materialized = json.loads(json.dumps(self.payload, allow_nan=False))
        if not isinstance(materialized, dict):
            raise TypeError("parts-first report payload must be a mapping")
        object.__setattr__(self, "payload", materialized)
        computed = _json_sha256(materialized, domain=_REPORT_HASH_DOMAIN)
        if self.report_sha256 == "":
            object.__setattr__(self, "report_sha256", computed)
        elif self.report_sha256 != computed:
            raise ValueError("parts-first report hash mismatch")
        validate_parts_first_supermode_report(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {**dict(self.payload), "report_sha256": self.report_sha256}


ModelCaptureLoader = Callable[[], PartsFirstCaptureBundle | Mapping[str, object]]
FiniteInterventionRunner = Callable[
    [FrozenSupermodeArtifact, CaptureRole, FrozenChartAtlas, PartsFirstPilotConfig],
    FiniteInterventionEvidence,
]


class _ChartCountRejected(ValueError):
    pass


def _fit_chart_atlas_at_count(
    role: CaptureRole,
    config: PartsFirstPilotConfig,
    *,
    chart_count: int,
    attempted_chart_counts: tuple[int, ...],
) -> FrozenChartAtlas:
    rows = tuple(sorted(role.rows, key=lambda row: row.row_id))
    if chart_count > len(rows):
        raise _ChartCountRejected("chart count exceeds A row count")
    centers: list[tuple[float, ...]] = [rows[0].chart_coordinates]
    while len(centers) < chart_count:
        candidates = []
        for row in rows:
            nearest = min(
                sum((a - b) ** 2 for a, b in zip(row.chart_coordinates, center))
                for center in centers
            )
            candidates.append((nearest, row.row_id, row.chart_coordinates))
        _, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
        if selected in centers:
            raise _ChartCountRejected(
                "A chart coordinates cannot support requested charts"
            )
        centers.append(selected)
    for _ in range(config.chart_lloyd_iterations):
        assignments = [
            min(
                range(len(centers)),
                key=lambda index: (
                    sum(
                        (a - b) ** 2
                        for a, b in zip(row.chart_coordinates, centers[index])
                    ),
                    index,
                ),
            )
            for row in rows
        ]
        buckets = [
            [row.chart_coordinates for row, assigned in zip(rows, assignments) if assigned == index]
            for index in range(len(centers))
        ]
        if any(not bucket for bucket in buckets):
            raise _ChartCountRejected(
                "deterministic A chart fit produced an empty chart"
            )
        centers = [
            tuple(
                sum(vector[dimension] for vector in bucket) / len(bucket)
                for dimension in range(len(bucket[0]))
            )
            for bucket in buckets
        ]
    atlas = FrozenChartAtlas(
        centers=tuple(centers),
        A_role_sha256=role.role_sha256,
        configured_maximum_chart_count=config.chart_count,
        selected_chart_count=chart_count,
        minimum_rows_per_chart=config.minimum_rows_per_chart,
        minimum_families_per_chart=config.minimum_families_per_chart,
        chart_count_search_policy=CHART_COUNT_SEARCH_POLICY,
        attempted_chart_counts=attempted_chart_counts,
    )
    counts = [0] * len(atlas.centers)
    family_support = [set() for _ in atlas.centers]
    for row in role.rows:
        chart_index = atlas.chart_index(row.chart_coordinates)
        counts[chart_index] += 1
        family_support[chart_index].add(row.family_id)
    if min(counts) < config.minimum_rows_per_chart:
        raise _ChartCountRejected(
            "one or more fitted A charts have insufficient support"
        )
    if (
        min(len(values) for values in family_support)
        < config.minimum_families_per_chart
    ):
        raise _ChartCountRejected(
            "one or more fitted A charts have insufficient family support"
        )
    return atlas


def _fit_chart_atlas(
    role: CaptureRole,
    config: PartsFirstPilotConfig,
) -> FrozenChartAtlas:
    rejected: list[tuple[int, str]] = []
    for chart_count in range(config.chart_count, 0, -1):
        attempted = tuple(range(config.chart_count, chart_count - 1, -1))
        try:
            return _fit_chart_atlas_at_count(
                role,
                config,
                chart_count=chart_count,
                attempted_chart_counts=attempted,
            )
        except _ChartCountRejected as error:
            rejected.append((chart_count, str(error)))
    attempted_counts = ", ".join(str(count) for count, _ in rejected)
    raise ValueError(
        "no deterministic A chart count satisfies the frozen row and family "
        f"support gates; attempted K={attempted_counts}"
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss * right_ss)
    return 0.0 if denominator == 0 else numerator / denominator


def _standardized_effect(
    targets: Sequence[float],
    active: Sequence[bool],
) -> tuple[float, int, int]:
    selected = [value for value, enabled in zip(targets, active) if enabled]
    other = [value for value, enabled in zip(targets, active) if not enabled]
    if not selected or not other:
        return 0.0, len(selected), len(other)
    spread = statistics.pstdev(targets)
    if spread == 0:
        return 0.0, len(selected), len(other)
    effect = (sum(selected) / len(selected) - sum(other) / len(other)) / spread
    return effect, len(selected), len(other)


def _axis_indexes(
    axis: Sequence[AtomResponseAxisEntry],
) -> dict[tuple[PhysicalAtomFamily, int], tuple[int, ...]]:
    values: dict[tuple[PhysicalAtomFamily, int], list[int]] = {}
    for index, item in enumerate(axis):
        values.setdefault((item.family, item.execution_ordinal), []).append(index)
    return {key: tuple(indexes) for key, indexes in values.items()}


def _stage_response(
    row: CaptureRow,
    stage: FrozenStage,
    axis_index_by_coordinate: Mapping[tuple[int, int], int],
) -> float:
    return sum(
        row.atom_responses[
            axis_index_by_coordinate[(ordinal, stage.execution_ordinal)]
        ]
        for ordinal in stage.atom_ordinals
    )


def _stage_active(
    row: CaptureRow,
    stage: FrozenStage,
    axis_index_by_coordinate: Mapping[tuple[int, int], int],
) -> bool:
    return (
        stage.orientation
        * _stage_response(row, stage, axis_index_by_coordinate)
        >= stage.activation_threshold
    )


def _path_active(
    row: CaptureRow,
    candidate: FrozenCandidate,
    atlas: FrozenChartAtlas,
    axis_index_by_coordinate: Mapping[tuple[int, int], int],
) -> bool:
    if atlas.chart_id(row.chart_coordinates) != candidate.chart_id:
        return False
    # This conjunction preserves event order.  It does not add emissions from
    # different residual-stream boundaries.
    return all(
        _stage_active(row, stage, axis_index_by_coordinate)
        for stage in candidate.stages
    )


def _materialize_artifact(
    candidate: FrozenCandidate,
    catalog: Gemma3PartsCatalog,
) -> FrozenSupermodeArtifact:
    short = candidate.candidate_sha256[:16]
    drafts = tuple(
        AtomGroupDraft(
            group_id=f"group.{short}.{index}",
            boundary=stage.boundary,
            atom_ids=tuple(
                catalog.atom_id_from_global_ordinal(ordinal)
                for ordinal in stage.atom_ordinals
            ),
        )
        for index, stage in enumerate(candidate.stages)
    )
    blob, refs = build_atom_group_sidecar(catalog, drafts)
    steps = tuple(
        SupermodeStepSpec(
            step_id=f"step.{short}.{index}",
            atom_group=group,
            chart_id=candidate.chart_id,
            input_carrier_abi=(
                "carrier.h640"
                if group.boundary.execution_ordinal < 38
                else "terminal.h640"
            ),
            additive_output_abi=(
                "raw_logits.v262144"
                if group.boundary.execution_ordinal == 38
                else "operator_output.h640"
            ),
            post_boundary_abi="native_suffix.recomputed",
        )
        for index, group in enumerate(refs)
    )
    path = SupermodePathSpec(
        path_id=f"supermode.{short}",
        atom_universe_sha256=catalog.catalog_sha256,
        steps=steps,
        cross_layer_claimed=True,
    )
    validate_supermode_path(path, catalog, blob)
    return FrozenSupermodeArtifact(
        candidate=candidate,
        group_blob=blob,
        group_blob_sha256=hashlib.sha256(blob).hexdigest(),
        path=path,
    )


def _discover_on_A(
    *,
    role: CaptureRole,
    axis: Sequence[AtomResponseAxisEntry],
    catalog: Gemma3PartsCatalog,
    config: PartsFirstPilotConfig,
) -> DiscoveryReceipt:
    if role.role != "A_discovery":
        raise ValueError("discovery accepts A only")
    atlas = _fit_chart_atlas(role, config)
    boundary_indexes = _axis_indexes(axis)
    coordinate_index = {
        (item.global_atom_ordinal, item.execution_ordinal): index
        for index, item in enumerate(axis)
    }
    candidates: list[FrozenCandidate] = []
    for chart_index in range(len(atlas.centers)):
        chart_id = f"chart.{chart_index}"
        rows = tuple(
            row
            for row in role.rows
            if atlas.chart_id(row.chart_coordinates) == chart_id
        )
        targets = tuple(row.target_score for row in rows)
        stage_pool: dict[str, list[FrozenStage]] = {
            role_id: []
            for role_id in (
                "lexical_ingress",
                "attention",
                "mlp",
                "lexical_readout",
            )
        }
        for (family, event_ordinal), indexes in boundary_indexes.items():
            per_atom = []
            for axis_index in indexes:
                responses = tuple(row.atom_responses[axis_index] for row in rows)
                correlation = _pearson(responses, targets)
                per_atom.append(
                    (
                        abs(correlation),
                        axis[axis_index].global_atom_ordinal,
                        axis_index,
                    )
                )
            selected = sorted(per_atom, key=lambda item: (-item[0], item[1]))[
                : min(config.group_size, len(per_atom))
            ]
            atom_ordinals = tuple(sorted(item[1] for item in selected))
            group_responses = tuple(
                sum(row.atom_responses[item[2]] for item in selected)
                for row in rows
            )
            correlation = _pearson(group_responses, targets)
            orientation = 1 if correlation >= 0 else -1
            threshold = float(
                statistics.median(
                    tuple(orientation * value for value in group_responses)
                )
            )
            boundary_kind = _boundary_from_execution_ordinal(
                event_ordinal
            ).boundary_kind
            role_id = {
                BoundaryKind.LEXICAL_INGRESS: "lexical_ingress",
                BoundaryKind.ATTENTION: "attention",
                BoundaryKind.MLP: "mlp",
                BoundaryKind.LEXICAL_READOUT: "lexical_readout",
            }.get(boundary_kind)
            if role_id is None:
                continue
            stage_pool[role_id].append(
                FrozenStage(
                    family=family,
                    execution_ordinal=event_ordinal,
                    atom_ordinals=atom_ordinals,
                    orientation=orientation,
                    activation_threshold=threshold,
                    A_stage_correlation=correlation,
                )
            )
        for role_id in stage_pool:
            stage_pool[role_id] = sorted(
                stage_pool[role_id],
                key=lambda stage: (
                    -abs(stage.A_stage_correlation),
                    stage.execution_ordinal,
                    stage.atom_ordinals,
                ),
            )[: config.boundary_limit_per_family]
        if any(not values for values in stage_pool.values()):
            continue
        for combination in itertools.product(
            stage_pool["lexical_ingress"],
            stage_pool["attention"],
            stage_pool["mlp"],
            stage_pool["lexical_readout"],
        ):
            stages = tuple(combination)
            attention_layer = stages[1].boundary.layer_ordinal
            mlp_layer = stages[2].boundary.layer_ordinal
            if (
                attention_layer is None
                or mlp_layer is None
                or attention_layer >= mlp_layer
            ):
                continue
            provisional = FrozenCandidate(
                chart_id=chart_id,
                stages=stages,
                A_effect=1.0,
                A_active_rows=1,
                A_inactive_rows=1,
                A_rank_score=1.0,
            )
            active = tuple(
                _path_active(row, provisional, atlas, coordinate_index)
                for row in rows
            )
            effect, active_count, inactive_count = _standardized_effect(
                targets,
                active,
            )
            if (
                active_count < config.minimum_active_rows
                or inactive_count < config.minimum_inactive_rows
                or abs(effect) < config.minimum_absolute_A_effect
            ):
                continue
            rank_score = abs(effect) * math.sqrt(min(active_count, inactive_count))
            candidates.append(
                FrozenCandidate(
                    chart_id=chart_id,
                    stages=stages,
                    A_effect=effect,
                    A_active_rows=active_count,
                    A_inactive_rows=inactive_count,
                    A_rank_score=rank_score,
                )
            )
    ranked = sorted(
        candidates,
        key=lambda item: (-item.A_rank_score, item.candidate_sha256),
    )[: config.maximum_A_candidates]
    artifacts = tuple(_materialize_artifact(item, catalog) for item in ranked)
    catalog_sha = _json_sha256(
        tuple(item.to_dict() for item in artifacts),
        domain=_CATALOG_HASH_DOMAIN,
    )
    return DiscoveryReceipt(
        A_role_sha256=role.role_sha256,
        chart_atlas=atlas,
        A_family_row_counts_by_chart=tuple(
            (
                f"chart.{chart_index}",
                tuple(
                    sorted(
                        {
                            family_id: sum(
                                row.family_id == family_id
                                and atlas.chart_index(row.chart_coordinates)
                                == chart_index
                                for row in role.rows
                            )
                            for family_id in role.family_row_counts
                        }.items()
                    )
                ),
            )
            for chart_index in range(len(atlas.centers))
        ),
        candidates=artifacts,
        candidate_catalog_sha256=catalog_sha,
    )


def _shuffled_stage(
    stage: FrozenStage,
    axis: Sequence[AtomResponseAxisEntry],
) -> FrozenStage:
    universe = tuple(
        sorted(
            item.global_atom_ordinal
            for item in axis
            if item.family is stage.family
            and item.execution_ordinal == stage.execution_ordinal
        )
    )
    if len(universe) <= len(stage.atom_ordinals):
        return stage
    start = (
        universe.index(stage.atom_ordinals[0]) + len(stage.atom_ordinals)
    ) % len(universe)
    members = tuple(
        sorted(
            universe[(start + offset) % len(universe)]
            for offset in range(len(stage.atom_ordinals))
        )
    )
    if members == stage.atom_ordinals:
        members = tuple(sorted(universe[-len(stage.atom_ordinals) :]))
    return replace(stage, atom_ordinals=members)


def _confirm_on_B(
    *,
    role: CaptureRole,
    axis: Sequence[AtomResponseAxisEntry],
    discovery: DiscoveryReceipt,
    config: PartsFirstPilotConfig,
) -> ConfirmationReceipt:
    if role.role != "B_confirmation":
        raise ValueError("confirmation accepts B only")
    coordinate_index = {
        (item.global_atom_ordinal, item.execution_ordinal): index
        for index, item in enumerate(axis)
    }
    evaluations: list[CandidateConfirmation] = []
    selected: str | None = None
    for artifact in discovery.candidates:
        candidate = artifact.candidate
        rows = tuple(
            row
            for row in role.rows
            if discovery.chart_atlas.chart_id(row.chart_coordinates)
            == candidate.chart_id
        )
        targets = tuple(row.target_score for row in rows)
        active = tuple(
            _path_active(
                row,
                candidate,
                discovery.chart_atlas,
                coordinate_index,
            )
            for row in rows
        )
        effect, active_count, inactive_count = _standardized_effect(targets, active)
        shuffled_candidate = replace(
            candidate,
            stages=tuple(_shuffled_stage(stage, axis) for stage in candidate.stages),
            candidate_sha256="",
        )
        shuffled_active = tuple(
            _path_active(
                row,
                shuffled_candidate,
                discovery.chart_atlas,
                coordinate_index,
            )
            for row in rows
        )
        shuffled_effect, _, _ = _standardized_effect(targets, shuffled_active)
        full_family_positions: dict[str, list[int]] = {}
        for index, row in enumerate(role.rows):
            full_family_positions.setdefault(row.family_id, []).append(index)
        permuted_by_row_id: dict[str, float] = {}
        source_by_row_id: dict[str, CaptureRow] = {}
        for family_id, positions in sorted(full_family_positions.items()):
            sources = list(positions)
            generator = random.Random(
                int(candidate.candidate_sha256[:16], 16)
                ^ config.deterministic_seed
                ^ int(hashlib.sha256(family_id.encode("utf-8")).hexdigest()[:16], 16)
            )
            generator.shuffle(sources)
            if sources == positions and len(sources) > 1:
                sources = sources[1:] + sources[:1]
            for destination, source in zip(positions, sources):
                destination_row = role.rows[destination]
                source_row = role.rows[source]
                permuted_by_row_id[destination_row.row_id] = (
                    source_row.target_score
                )
                source_by_row_id[destination_row.row_id] = source_row
        permuted_targets = tuple(
            permuted_by_row_id[row.row_id] for row in rows
        )
        cross_chart_sources = sum(
            discovery.chart_atlas.chart_id(
                source_by_row_id[row.row_id].chart_coordinates
            )
            != candidate.chart_id
            for row in rows
        )
        row_effect, _, _ = _standardized_effect(permuted_targets, active)
        chart_family_counts: dict[str, int] = {}
        for row in rows:
            chart_family_counts[row.family_id] = (
                chart_family_counts.get(row.family_id, 0) + 1
            )
        reasons: list[str] = []
        if active_count < config.minimum_active_rows:
            reasons.append("insufficient_active_support")
        if inactive_count < config.minimum_inactive_rows:
            reasons.append("insufficient_inactive_support")
        if abs(effect) < config.minimum_absolute_B_effect:
            reasons.append("effect_below_frozen_threshold")
        if effect * candidate.A_effect <= 0:
            reasons.append("effect_sign_did_not_replicate")
        strongest_control = max(abs(shuffled_effect), abs(row_effect), 0.0)
        if abs(effect) < strongest_control + config.minimum_B_control_margin:
            reasons.append("matched_control_margin_failed")
        passed = not reasons
        evaluations.append(
            CandidateConfirmation(
                candidate_sha256=candidate.candidate_sha256,
                B_effect=effect,
                B_active_rows=active_count,
                B_inactive_rows=inactive_count,
                B_family_row_counts=tuple(
                    sorted(chart_family_counts.items())
                ),
                shuffled_membership_effect=shuffled_effect,
                shuffled_row_effect=row_effect,
                shuffled_row_cross_chart_source_count=cross_chart_sources,
                zero_control_effect=0.0,
                passed=passed,
                failure_reasons=tuple(reasons),
            )
        )
        if selected is None and passed:
            selected = candidate.candidate_sha256
    return ConfirmationReceipt(
        B_role_sha256=role.role_sha256,
        frozen_candidate_catalog_sha256=discovery.candidate_catalog_sha256,
        evaluations=tuple(evaluations),
        selected_candidate_sha256=selected,
    )


def _evaluate_C_intervention(
    *,
    role: CaptureRole,
    artifact: FrozenSupermodeArtifact,
    atlas: FrozenChartAtlas,
    evidence: FiniteInterventionEvidence,
    config: PartsFirstPilotConfig,
    expected_source_model_fingerprint: str,
    expected_source_execution_fingerprint: str,
) -> InterventionReceipt:
    if role.role != "C_intervention":
        raise ValueError("intervention evaluation accepts C only")
    if evidence.candidate_sha256 != artifact.candidate_sha256:
        raise ValueError("C evidence does not bind the B-selected candidate")
    if evidence.c_row_ids_sha256 != role.row_ids_sha256:
        raise ValueError("C evidence row identity does not match fresh C")
    if evidence.gate_alpha != config.gate_alpha:
        raise ValueError("C evidence gate alpha differs from frozen config")
    if (
        evidence.source_model_fingerprint_before
        != expected_source_model_fingerprint
        or evidence.source_execution_fingerprint_before
        != expected_source_execution_fingerprint
    ):
        raise ValueError("C evidence source fingerprint binding drifted")
    expected_charts = tuple(
        atlas.chart_id(row.chart_coordinates) for row in role.rows
    )
    if evidence.baseline_chart_ids != expected_charts:
        raise ValueError("C evidence baseline chart routing drifted")
    if evidence.baseline_scores != tuple(row.target_score for row in role.rows):
        raise ValueError("C evidence baseline scores do not match capture")
    if len(evidence.baseline_scores) != len(role.rows):
        raise ValueError("C evidence row count does not close")
    if evidence.finite_intervention_forward_count != len(role.rows):
        raise ValueError(
            "C finite-intervention forward count must equal fresh C row count"
        )
    selected_mask = tuple(
        chart_id == artifact.candidate.chart_id for chart_id in expected_charts
    )
    if not any(selected_mask):
        raise ValueError("selected path has no fresh C chart support")
    actual = tuple(
        after - before
        for before, after in zip(
            evidence.baseline_scores,
            evidence.intervened_scores,
        )
    )
    selected_actual = tuple(
        value for value, selected in zip(actual, selected_mask) if selected
    )
    selected_count = len(selected_actual)
    mean_abs_delta = sum(abs(value) for value in selected_actual) / len(
        selected_actual
    )
    relative_errors = tuple(
        abs(observed - predicted)
        / max(abs(observed), abs(predicted), 1.0e-12)
        for observed, predicted, selected in zip(
            actual,
            evidence.first_order_predicted_score_deltas,
            selected_mask,
        )
        if selected
    )
    maximum_relative_error = max(relative_errors)
    stability = sum(
        before == after
        for before, after, selected in zip(
            evidence.baseline_chart_ids,
            evidence.online_chart_ids_after,
            selected_mask,
        )
        if selected
    ) / len(selected_actual)
    reasons: list[str] = []
    if selected_count < config.minimum_active_rows:
        reasons.append("insufficient_selected_chart_support")
    if mean_abs_delta < config.minimum_C_mean_absolute_delta:
        reasons.append("finite_effect_below_threshold")
    if maximum_relative_error > config.maximum_C_first_order_relative_error:
        reasons.append("first_order_finite_agreement_failed")
    if stability < config.minimum_C_router_stability:
        reasons.append("online_router_stability_failed")
    if evidence.native_noop_max_abs_error > config.maximum_native_noop_error:
        reasons.append("native_noop_parity_failed")
    return InterventionReceipt(
        candidate_sha256=artifact.candidate_sha256,
        evidence_sha256=evidence.evidence_sha256,
        C_role_sha256=role.role_sha256,
        finite_intervention_candidate_count=(
            evidence.finite_intervention_candidate_count
        ),
        finite_intervention_forward_count=(
            evidence.finite_intervention_forward_count
        ),
        selected_chart_row_count=selected_count,
        mean_absolute_score_delta_on_selected_chart=mean_abs_delta,
        maximum_first_order_relative_error=maximum_relative_error,
        router_stability_fraction=stability,
        native_noop_max_abs_error=evidence.native_noop_max_abs_error,
        executor_kind=evidence.executor_kind,
        native_gate_execution_receipt_sha256=(
            evidence.native_gate_execution_receipt_sha256
        ),
        source_model_fingerprint_before=(
            evidence.source_model_fingerprint_before
        ),
        source_model_fingerprint_after=evidence.source_model_fingerprint_after,
        source_execution_fingerprint_before=(
            evidence.source_execution_fingerprint_before
        ),
        source_execution_fingerprint_after=(
            evidence.source_execution_fingerprint_after
        ),
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def preflight_parts_first_capture_bundle(
    captures: PartsFirstCaptureBundle | Mapping[str, object],
    *,
    config: PartsFirstPilotConfig = PartsFirstPilotConfig(),
) -> dict[str, object]:
    bundle = (
        captures
        if isinstance(captures, PartsFirstCaptureBundle)
        else PartsFirstCaptureBundle.from_dict(captures)
    )
    if not isinstance(config, PartsFirstPilotConfig):
        raise TypeError("config must be PartsFirstPilotConfig")
    family_counts = {
        family.value: sum(item.family is family for item in bundle.response_axis)
        for family in FAMILY_ORDER
    }
    if any(count == 0 for count in family_counts.values()):
        raise ValueError("pilot capture must expose lexical, attention, and MLP parts")
    return {
        "capture_sha256": bundle.capture_sha256,
        "source_kind": bundle.source_kind,
        "model_id": bundle.model_id,
        "source_model_fingerprint_sha256": (
            bundle.source_model_fingerprint_sha256
        ),
        "source_execution_fingerprint_sha256": (
            bundle.source_execution_fingerprint_sha256
        ),
        "score_objective_id": bundle.score_objective_id,
        "chart_surface_id": bundle.chart_surface_id,
        "protocol_sha256": bundle.protocol_sha256,
        "atom_universe_sha256": bundle.atom_universe_sha256,
        "response_axis_occurrence_count": len(bundle.response_axis),
        "response_axis_family_counts": family_counts,
        "role_row_counts": {
            role.role: len(role.rows) for role in bundle.roles
        },
        "semantic_family_row_counts_by_role": {
            role.role: role.family_row_counts for role in bundle.roles
        },
        "role_sha256s": {
            role.role: role.role_sha256 for role in bundle.roles
        },
        "role_rows_disjoint": True,
        "whole_layer_generator_axis_present": False,
        "exact_lift_gate_passed": bundle.exact_lift_receipt.passed,
        "scientific_evidence_eligible": (
            bundle.source_kind == "model_capture"
            and bundle.exact_lift_receipt.passed
        ),
        "demo_only": bundle.source_kind == "deterministic_demo",
        "configuration_sha256": _json_sha256(config.to_dict()),
        "passed": True,
    }


def run_parts_first_supermode_from_captures(
    captures: PartsFirstCaptureBundle | Mapping[str, object],
    *,
    config: PartsFirstPilotConfig = PartsFirstPilotConfig(),
    intervention_runner: FiniteInterventionRunner | None = None,
) -> PartsFirstPilotReport:
    """Run A discovery, frozen B confirmation, and at most one selected C gate."""

    bundle = (
        captures
        if isinstance(captures, PartsFirstCaptureBundle)
        else PartsFirstCaptureBundle.from_dict(captures)
    )
    preflight = preflight_parts_first_capture_bundle(bundle, config=config)
    roles = bundle.role_by_id
    catalog = _canonical_catalog()
    discovery = _discover_on_A(
        role=roles["A_discovery"],
        axis=bundle.response_axis,
        catalog=catalog,
        config=config,
    )
    confirmation = _confirm_on_B(
        role=roles["B_confirmation"],
        axis=bundle.response_axis,
        discovery=discovery,
        config=config,
    )
    intervention: InterventionReceipt | None = None
    selected_sha = confirmation.selected_candidate_sha256
    callback_count = 0
    if selected_sha is not None:
        artifact = next(
            item for item in discovery.candidates if item.candidate_sha256 == selected_sha
        )
        if intervention_runner is not None and bundle.intervention_evidence is not None:
            raise ValueError(
                "provide either an intervention runner or embedded evidence, not both"
            )
        if intervention_runner is not None:
            callback_count += 1
            evidence = intervention_runner(
                artifact,
                roles["C_intervention"],
                discovery.chart_atlas,
                config,
            )
            if not isinstance(evidence, FiniteInterventionEvidence):
                raise TypeError("intervention runner returned an invalid receipt")
        elif bundle.intervention_evidence is not None:
            evidence = bundle.intervention_evidence
        else:
            raise RuntimeError(
                "B selected a path but no C finite-intervention evidence is available"
            )
        intervention = _evaluate_C_intervention(
            role=roles["C_intervention"],
            artifact=artifact,
            atlas=discovery.chart_atlas,
            evidence=evidence,
            config=config,
            expected_source_model_fingerprint=(
                bundle.source_model_fingerprint_sha256
            ),
            expected_source_execution_fingerprint=(
                bundle.source_execution_fingerprint_sha256
            ),
        )
    elif bundle.intervention_evidence is not None:
        raise ValueError("embedded C evidence exists without a B-selected path")
    outcome = (
        "no_A_candidate"
        if not discovery.candidates
        else (
            "no_frozen_candidate_confirmed_on_B"
            if selected_sha is None
            else (
                "selected_path_failed_C_finite_gate"
                if intervention is not None and not intervention.passed
                else "parts_first_supermode_pilot_completed"
            )
        )
    )
    model_facing_intervention = bool(
        intervention is not None
        and intervention.executor_kind == "live_model_bridge"
        and bundle.source_kind == "model_capture"
    )
    payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "format_version": FORMAT_VERSION,
        "diagnostic_outcome": outcome,
        "protocol": {
            "protocol_sha256": bundle.protocol_sha256,
            "atom_universe_sha256": bundle.atom_universe_sha256,
        },
        "capture": {
            "capture_sha256": bundle.capture_sha256,
            "source_kind": bundle.source_kind,
            "source_model_fingerprint_sha256": (
                bundle.source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                bundle.source_execution_fingerprint_sha256
            ),
            "score_objective_id": bundle.score_objective_id,
            "chart_surface_id": bundle.chart_surface_id,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "exact_lift_receipt": bundle.exact_lift_receipt.to_dict(),
        },
        "configuration": config.to_dict(),
        "preflight": preflight,
        "A_discovery": discovery.to_dict(),
        "B_confirmation": confirmation.to_dict(),
        "C_intervention": (
            None if intervention is None else intervention.to_dict()
        ),
        "execution_firewall": {
            "A_roles_read": ["A_discovery"],
            "B_roles_read": ["B_confirmation"],
            "C_roles_read": (
                [] if intervention is None else ["C_intervention"]
            ),
            "A_frozen_before_B": True,
            "B_fit_or_refit_count": 0,
            "C_selection_count": 0 if selected_sha is None else 1,
            "C_finite_intervention_candidate_count": (
                0
                if intervention is None
                else intervention.finite_intervention_candidate_count
            ),
            "C_intervention_callback_count": callback_count,
            "C_finite_intervention_forward_count": (
                0
                if intervention is None
                else intervention.finite_intervention_forward_count
            ),
        },
        "scientific_status": {
            "pilot_only": True,
            "full_preregistered_protocol_completed": False,
            "model_facing_capture_consumed": (
                bundle.source_kind == "model_capture"
            ),
            "model_facing_finite_intervention_executed": (
                model_facing_intervention
            ),
            "deterministic_demo_only": (
                bundle.source_kind == "deterministic_demo"
            ),
            "ordered_parts_path_tested": intervention is not None,
            "heldout_C_gate_passed": (
                None if intervention is None else intervention.passed
            ),
            "compression_attempted": False,
            "compression_success_claimed": False,
            "parameter_savings_claimed": False,
            "MAC_savings_claimed": False,
            "latency_claimed": False,
        },
        "claims": {
            "whole_layer_is_a_generator": False,
            "cross_boundary_emissions_are_additive": False,
            "full_protocol_success": False,
            "compression_success": False,
            "parameter_savings": False,
            "MAC_savings": False,
            "latency_improvement": False,
            "invariant_circuit_ontology": False,
        },
    }
    return PartsFirstPilotReport(payload)


def validate_parts_first_supermode_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("parts-first report must be a mapping")
    materialized = json.loads(json.dumps(value, allow_nan=False))
    if not isinstance(materialized, dict):
        raise TypeError("parts-first report must be an object")
    expected_fields = {
        "schema",
        "format_version",
        "diagnostic_outcome",
        "protocol",
        "capture",
        "configuration",
        "preflight",
        "A_discovery",
        "B_confirmation",
        "C_intervention",
        "execution_firewall",
        "scientific_status",
        "claims",
        "report_sha256",
    }
    _exact_keys(materialized, expected_fields, label="parts-first report")
    observed_hash = materialized.pop("report_sha256")
    if (
        not isinstance(observed_hash, str)
        or _json_sha256(materialized, domain=_REPORT_HASH_DOMAIN) != observed_hash
    ):
        raise ValueError("parts-first report hash is invalid")
    if (
        materialized["schema"] != REPORT_SCHEMA
        or materialized["format_version"] != FORMAT_VERSION
    ):
        raise ValueError("parts-first report header is invalid")
    claims = materialized["claims"]
    status = materialized["scientific_status"]
    firewall = materialized["execution_firewall"]
    configuration = materialized["configuration"]
    capture_section = materialized["capture"]
    preflight_section = materialized["preflight"]
    discovery_section = materialized["A_discovery"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            claims,
            status,
            firewall,
            configuration,
            capture_section,
            preflight_section,
            discovery_section,
        )
    ):
        raise ValueError("parts-first report semantic sections are invalid")
    chart_surface_id = capture_section.get("chart_surface_id")
    source_kind = capture_section.get("source_kind")
    expected_chart_surface_id = (
        {
            "model_capture": MODEL_CHART_SURFACE_ID,
            "deterministic_demo": DEMO_CHART_SURFACE_ID,
        }.get(source_kind)
        if isinstance(source_kind, str)
        else None
    )
    if (
        chart_surface_id != expected_chart_surface_id
        or preflight_section.get("chart_surface_id") != chart_surface_id
    ):
        raise ValueError("parts-first report chart surface binding drifted")
    chart_atlas = discovery_section.get("chart_atlas")
    chart_selection = discovery_section.get("chart_selection")
    if not isinstance(chart_atlas, Mapping) or not isinstance(
        chart_selection,
        Mapping,
    ):
        raise ValueError("parts-first report chart selection is invalid")
    try:
        typed_configuration = PartsFirstPilotConfig.from_dict(configuration)
        typed_atlas = FrozenChartAtlas.from_dict(chart_atlas)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "parts-first report typed configuration or atlas is invalid"
        ) from error
    expected_selection = typed_atlas.selection_dict()
    if (
        _canonical_json_bytes(typed_configuration.to_dict())
        != _canonical_json_bytes(configuration)
        or _canonical_json_bytes(typed_atlas.to_dict())
        != _canonical_json_bytes(chart_atlas)
        or _canonical_json_bytes(expected_selection)
        != _canonical_json_bytes(chart_selection)
        or typed_atlas.configured_maximum_chart_count
        != typed_configuration.chart_count
        or typed_atlas.minimum_rows_per_chart
        != typed_configuration.minimum_rows_per_chart
        or typed_atlas.minimum_families_per_chart
        != typed_configuration.minimum_families_per_chart
        or preflight_section.get("configuration_sha256")
        != _json_sha256(typed_configuration.to_dict())
    ):
        raise ValueError("parts-first report chart selection does not close")
    preflight_role_sha256s = preflight_section.get("role_sha256s")
    discovery_A_role_sha256 = discovery_section.get("A_role_sha256")
    if (
        not isinstance(preflight_role_sha256s, Mapping)
        or not isinstance(discovery_A_role_sha256, str)
        or _SHA256_RE.fullmatch(discovery_A_role_sha256) is None
        or typed_atlas.A_role_sha256 != discovery_A_role_sha256
        or preflight_role_sha256s.get("A_discovery")
        != discovery_A_role_sha256
    ):
        raise ValueError("parts-first report atlas A-role binding drifted")
    if any(value is not False for value in claims.values()):
        raise ValueError("parts-first pilot report contains an unsupported claim")
    for field_name in (
        "compression_attempted",
        "compression_success_claimed",
        "parameter_savings_claimed",
        "MAC_savings_claimed",
        "latency_claimed",
    ):
        if status.get(field_name) is not False:
            raise ValueError("parts-first pilot crossed its claim boundary")
    selected = materialized["B_confirmation"].get(  # type: ignore[union-attr]
        "selected_candidate_sha256"
    )
    c_receipt = materialized["C_intervention"]
    finite_count = firewall.get("C_finite_intervention_forward_count")
    candidate_count = firewall.get("C_finite_intervention_candidate_count")
    if selected is None:
        if c_receipt is not None or finite_count != 0 or candidate_count != 0:
            raise ValueError("C ran without a B-selected path")
    else:
        expected_C_forwards = materialized["preflight"][  # type: ignore[index]
            "role_row_counts"
        ]["C_intervention"]
        if (
            not isinstance(c_receipt, Mapping)
            or c_receipt.get("candidate_sha256") != selected
            or c_receipt.get("finite_intervention_candidate_count") != 1
            or candidate_count != 1
            or c_receipt.get("finite_intervention_forward_count")
            != expected_C_forwards
            or finite_count != expected_C_forwards
        ):
            raise ValueError(
                "selected path does not bind one C arm over every fresh C row"
            )
        selected_count = c_receipt.get("selected_chart_row_count")
        minimum_selected = configuration.get("minimum_active_rows")
        failure_reasons = c_receipt.get("failure_reasons")
        if (
            type(selected_count) is not int
            or type(expected_C_forwards) is not int
            or selected_count < 1
            or selected_count > expected_C_forwards
            or type(minimum_selected) is not int
            or minimum_selected < 1
            or not isinstance(failure_reasons, list)
            or (
                selected_count < minimum_selected
                and "insufficient_selected_chart_support"
                not in failure_reasons
            )
            or (
                selected_count >= minimum_selected
                and "insufficient_selected_chart_support"
                in failure_reasons
            )
            or (
                selected_count < minimum_selected
                and c_receipt.get("passed") is not False
            )
        ):
            raise ValueError("C selected-chart support does not close")
    materialized["report_sha256"] = observed_hash
    return materialized


def load_parts_first_capture_bundle(
    source: Path | str,
) -> PartsFirstCaptureBundle:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("parts-first capture bundle is unavailable") from error
    if not isinstance(value, Mapping):
        raise ValueError("parts-first capture bundle must be an object")
    return PartsFirstCaptureBundle.from_dict(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_validated_json_no_clobber(
    value: object,
    target: Path,
    *,
    label: str,
    loader: Callable[[Path], object],
    expected: object,
) -> object:
    """Validate a durable temp file, then atomically link it into place."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    temporary_exists = True
    published_stat: os.stat_result | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded = loader(temporary)
        if loaded != expected:
            raise RuntimeError(f"{label} changed across strict temp reload")
        published_stat = temporary.stat()
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite a {label}") from error
        temporary.unlink()
        temporary_exists = False
        try:
            _fsync_directory(target.parent)
        except BaseException:
            try:
                target_stat = target.stat()
                if os.path.samestat(published_stat, target_stat):
                    target.unlink()
                    _fsync_directory(target.parent)
            except OSError:
                pass
            raise
        return loaded
    finally:
        if temporary_exists:
            temporary.unlink(missing_ok=True)


def write_parts_first_capture_bundle(
    bundle: PartsFirstCaptureBundle,
    destination: Path | str,
) -> PartsFirstCaptureBundle:
    if not isinstance(bundle, PartsFirstCaptureBundle):
        raise TypeError("bundle must be PartsFirstCaptureBundle")
    target = Path(destination)
    if target.suffix != ".json":
        raise ValueError("parts-first capture output must use .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    loaded = _publish_validated_json_no_clobber(
        bundle.to_dict(),
        target,
        label="parts-first capture",
        loader=load_parts_first_capture_bundle,
        expected=bundle,
    )
    if not isinstance(loaded, PartsFirstCaptureBundle):
        raise RuntimeError("parts-first capture loader returned an invalid type")
    return loaded


def write_parts_first_supermode_report(
    report: PartsFirstPilotReport | Mapping[str, object],
    destination: Path | str,
) -> dict[str, object]:
    value = report.to_dict() if isinstance(report, PartsFirstPilotReport) else report
    validated = validate_parts_first_supermode_report(value)
    target = Path(destination)
    if target.suffix != ".json":
        raise ValueError("parts-first report output must use .json")
    target.parent.mkdir(parents=True, exist_ok=True)

    def load_report(path: Path) -> dict[str, object]:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("parts-first report is unavailable") from error
        if not isinstance(loaded, Mapping):
            raise ValueError("parts-first report must be an object")
        return validate_parts_first_supermode_report(loaded)

    loaded = _publish_validated_json_no_clobber(
        validated,
        target,
        label="parts-first report",
        loader=load_report,
        expected=validated,
    )
    if not isinstance(loaded, dict):
        raise RuntimeError("parts-first report loader returned an invalid type")
    return validated


def _demo_rows(
    role: str,
    *,
    row_count_per_chart: int = 24,
) -> tuple[CaptureRow, ...]:
    rows: list[CaptureRow] = []
    role_offset = {"A_discovery": 0.00, "B_confirmation": 0.03, "C_intervention": -0.02}[role]
    for chart in range(2):
        for local in range(row_count_per_chart):
            centered = float(local) - (row_count_per_chart - 1) / 2
            target = centered + 0.15 * chart + role_offset
            alternating = 1.0 if local % 2 == 0 else -1.0
            distract = alternating * (0.7 + 0.05 * chart)
            responses = (
                target + 0.01 * alternating,
                0.82 * target - 0.02 * alternating,
                distract,
                -0.8 * distract,
                1.15 * target + 0.01 * alternating,
                0.73 * target - 0.01 * alternating,
                -distract,
                0.6 * distract,
                0.91 * target + 0.02 * alternating,
                1.08 * target - 0.01 * alternating,
                0.5 * distract,
                -0.4 * distract,
                1.03 * target - 0.01 * alternating,
                0.88 * target + 0.02 * alternating,
                -0.65 * distract,
                0.45 * distract,
            )
            rows.append(
                CaptureRow(
                    row_id=f"{role.lower()}.{chart}.{local}",
                    family_id=f"family.{local % 6}",
                    chart_coordinates=(
                        chart * 10.0 + centered * 0.05,
                        chart * -8.0 + centered * 0.03,
                    ),
                    target_score=target,
                    atom_responses=responses,
                )
            )
    return tuple(rows)


def build_demo_capture_bundle() -> PartsFirstCaptureBundle:
    """Build a deterministic non-model bundle that exercises all three roles."""

    protocol = build_parts_first_supermode_protocol()
    catalog = _canonical_catalog()
    mlp_start = 640 + 18 * 4 + 2_048  # layer 1, channels 0..3
    axis = (
        *(AtomResponseAxisEntry(index, 0) for index in range(4)),
        *(AtomResponseAxisEntry(640 + index, 1) for index in range(4)),
        *(AtomResponseAxisEntry(mlp_start + index, 4) for index in range(4)),
        *(AtomResponseAxisEntry(index, 38) for index in range(4)),
    )
    roles = tuple(
        CaptureRole(
            role=role,
            rows=_demo_rows(role),
            exact_observer_passed=False,
        )
        for role in ROLE_ORDER
    )
    return PartsFirstCaptureBundle(
        source_kind="deterministic_demo",
        model_id=MODEL_ID,
        source_model_fingerprint_sha256=DEMO_SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_sha256=(
            DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        score_objective_id=DEMO_SCORE_OBJECTIVE_ID,
        chart_surface_id=DEMO_CHART_SURFACE_ID,
        protocol_sha256=str(protocol["protocol_sha256"]),
        atom_universe_sha256=catalog.catalog_sha256,
        response_axis=tuple(axis),
        roles=roles,
        exact_lift_receipt=ExactLiftCaptureReceipt(
            model_facing=False,
            exact_39_event_order_passed=False,
            all_37576_physical_atoms_closed=False,
            all_shared_substrate_slices_closed=False,
            tied_lexical_storage_identity_passed=False,
            source_state_unchanged=True,
            exact_control_evidence_sha256=(
                DEMO_EXACT_CONTROL_EVIDENCE_SHA256
            ),
        ),
    )


def _demo_intervention_runner(
    artifact: FrozenSupermodeArtifact,
    role: CaptureRole,
    atlas: FrozenChartAtlas,
    config: PartsFirstPilotConfig,
) -> FiniteInterventionEvidence:
    chart_ids = tuple(atlas.chart_id(row.chart_coordinates) for row in role.rows)
    baseline = tuple(row.target_score for row in role.rows)
    displacement = tuple(
        (
            (config.gate_alpha - 1.0) * 0.4
            if chart_id == artifact.candidate.chart_id
            else 0.0
        )
        for chart_id in chart_ids
    )
    return FiniteInterventionEvidence(
        candidate_sha256=artifact.candidate_sha256,
        c_row_ids_sha256=role.row_ids_sha256,
        gate_alpha=config.gate_alpha,
        baseline_scores=baseline,
        intervened_scores=tuple(
            value + delta for value, delta in zip(baseline, displacement)
        ),
        first_order_predicted_score_deltas=displacement,
        baseline_chart_ids=chart_ids,
        online_chart_ids_after=chart_ids,
        native_noop_max_abs_error=0.0,
        finite_intervention_candidate_count=1,
        finite_intervention_forward_count=len(role.rows),
        executor_kind="deterministic_demo",
        native_gate_execution_receipt_sha256=(
            DEMO_NATIVE_GATE_RECEIPT_SHA256
        ),
        source_model_fingerprint_before=(
            DEMO_SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_model_fingerprint_after=(
            DEMO_SOURCE_MODEL_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_before=(
            DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_after=(
            DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
    )


def run_gemma3_parts_first_supermode_experiment(
    *,
    captures: PartsFirstCaptureBundle | Mapping[str, object] | None = None,
    captures_path: Path | str | None = None,
    model_capture_loader: ModelCaptureLoader | None = None,
    intervention_runner: FiniteInterventionRunner | None = None,
    demo: bool = False,
    dry_run: bool = False,
    config: PartsFirstPilotConfig = PartsFirstPilotConfig(),
    output: Path | str | None = None,
) -> PartsFirstPilotReport | dict[str, object]:
    """Resolve one capture source and execute or preflight the pilot.

    ``model_capture_loader`` is the injectable live-model boundary.  It must
    return the same authenticated capture type as the JSON path; this runner
    never imports a model framework or silently falls back to synthetic data.
    """

    source_count = sum(
        value is not None
        for value in (captures, captures_path, model_capture_loader)
    ) + int(demo)
    if source_count != 1:
        raise ValueError(
            "exactly one of captures, captures_path, model_capture_loader, or demo is required"
        )
    if demo:
        bundle = build_demo_capture_bundle()
        if intervention_runner is None:
            intervention_runner = _demo_intervention_runner
    elif captures_path is not None:
        bundle = load_parts_first_capture_bundle(captures_path)
    elif model_capture_loader is not None:
        loaded = model_capture_loader()
        bundle = (
            loaded
            if isinstance(loaded, PartsFirstCaptureBundle)
            else PartsFirstCaptureBundle.from_dict(loaded)
        )
        if bundle.source_kind != "model_capture":
            raise ValueError("model_capture_loader returned a non-model bundle")
    else:
        assert captures is not None
        bundle = (
            captures
            if isinstance(captures, PartsFirstCaptureBundle)
            else PartsFirstCaptureBundle.from_dict(captures)
        )
    if dry_run:
        if output is not None:
            raise ValueError("dry-run does not write a report")
        return preflight_parts_first_capture_bundle(bundle, config=config)
    report = run_parts_first_supermode_from_captures(
        bundle,
        config=config,
        intervention_runner=intervention_runner,
    )
    if output is not None:
        write_parts_first_supermode_report(report, output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--captures", type=Path)
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_gemma3_parts_first_supermode_experiment(
        captures_path=args.captures,
        demo=args.demo,
        dry_run=args.dry_run,
        output=args.output,
    )
    if args.output is None:
        value = result.to_dict() if isinstance(result, PartsFirstPilotReport) else result
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
