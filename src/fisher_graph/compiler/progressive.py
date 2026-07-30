"""Residual-guided progressive compilation on development-only evidence.

This module is a model-independent control plane for growing a compiled graph
until it reaches a fidelity target and then compacting it while it remains
inside that target.  It deliberately does not execute a model, load prompts,
or know how a particular mutation is implemented.  Model-specific workers
provide four narrowly scoped operations:

``map``
    Analyze fit-only residuals and return ranked, hash-bound repair targets.
``propose``
    Describe candidate mutations and their complete deployment resources.
``build``
    Materialize one proposed immutable candidate.
``validate``
    Measure that candidate on the family-disjoint development guard split.

The controller verifies every binding, applies a deterministic acceptance
policy, records an immutable transcript, and emits a candidate handoff only
after fidelity and resource gates pass.  Held-out assessment manifests are
registered only as forbidden identities.  They are never supplied to any
development callback and this module has no held-out execution hook.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Literal, Protocol


PROGRESSIVE_COMPILATION_SCHEMA = (
    "fisher_graph.compiler.progressive_compilation"
)
PROGRESSIVE_COMPILATION_FORMAT_VERSION = 1

ProgressivePhase = Literal["repair", "compact"]
MutationKind = Literal[
    "seed",
    "refit_edges",
    "add_residual_edge",
    "split_generator",
    "widen_carrier",
    "retain_source_island",
    "merge_generators",
    "prune_generator",
    "factorize_edges",
    "remove_source_island",
]
ProgressiveStatus = Literal[
    "ready_for_candidate_binding",
    "rejected_by_guard",
    "stalled_fidelity",
    "stalled_budget",
    "max_iterations",
]

_REPAIR_MUTATIONS = frozenset(
    {
        "refit_edges",
        "add_residual_edge",
        "split_generator",
        "widen_carrier",
        "retain_source_island",
    }
)
_COMPACT_MUTATIONS = frozenset(
    {
        "merge_generators",
        "prune_generator",
        "factorize_edges",
        "remove_source_island",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_HASH_DOMAIN = b"fisher-graph:progressive-compilation:v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _payload_sha256(kind: str, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable nonempty identifier")
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


def _finite_positive(value: object, *, label: str) -> float:
    result = _finite_nonnegative(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _unit_interval(value: object, *, label: str) -> float:
    result = _finite_nonnegative(value, label=label)
    if result > 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _canonical_identifiers(
    values: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{label} must be a nonempty tuple")
    parsed = tuple(
        _identifier(value, label=f"{label}[]") for value in values
    )
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(
            f"{label} must be sorted and contain no duplicates"
        )
    return parsed


def _canonical_sha256s(
    values: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if type(values) is not tuple or not values:
        raise ValueError(f"{label} must be a nonempty tuple")
    parsed = tuple(
        _require_sha256(value, label=f"{label}[]") for value in values
    )
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError(
            f"{label} must be sorted and contain no duplicates"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class DevelopmentCorpus:
    """Pairwise family-disjoint fit, selection, and final-guard manifests."""

    corpus_id: str
    fit_manifest_sha256: str
    selection_manifest_sha256: str
    guard_manifest_sha256: str
    fit_example_count: int
    selection_example_count: int
    guard_example_count: int
    fit_family_ids: tuple[str, ...]
    selection_family_ids: tuple[str, ...]
    guard_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.corpus_id, label="corpus_id")
        fit_manifest = _require_sha256(
            self.fit_manifest_sha256,
            label="fit_manifest_sha256",
        )
        selection_manifest = _require_sha256(
            self.selection_manifest_sha256,
            label="selection_manifest_sha256",
        )
        guard_manifest = _require_sha256(
            self.guard_manifest_sha256,
            label="guard_manifest_sha256",
        )
        if len({fit_manifest, selection_manifest, guard_manifest}) != 3:
            raise ValueError(
                "fit, selection, and guard manifests must be distinct"
            )
        for name in (
            "fit_example_count",
            "selection_example_count",
            "guard_example_count",
        ):
            _positive_integer(getattr(self, name), label=name)
        fit_families = _canonical_identifiers(
            self.fit_family_ids,
            label="fit_family_ids",
        )
        selection_families = _canonical_identifiers(
            self.selection_family_ids,
            label="selection_family_ids",
        )
        guard_families = _canonical_identifiers(
            self.guard_family_ids,
            label="guard_family_ids",
        )
        family_sets = (
            set(fit_families),
            set(selection_families),
            set(guard_families),
        )
        if any(
            left.intersection(right)
            for index, left in enumerate(family_sets)
            for right in family_sets[index + 1 :]
        ):
            raise ValueError(
                "fit, selection, and guard families must be pairwise "
                "disjoint"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "role": "calibration_a_development",
            "fit": {
                "manifest_sha256": self.fit_manifest_sha256,
                "example_count": self.fit_example_count,
                "family_ids": self.fit_family_ids,
            },
            "selection": {
                "manifest_sha256": self.selection_manifest_sha256,
                "example_count": self.selection_example_count,
                "family_ids": self.selection_family_ids,
            },
            "guard": {
                "manifest_sha256": self.guard_manifest_sha256,
                "example_count": self.guard_example_count,
                "family_ids": self.guard_family_ids,
            },
            "pairwise_family_disjoint": True,
            "selection_reusable_during_loop": True,
            "guard_opened_only_after_challenger_freeze": True,
        }


@dataclass(frozen=True, slots=True)
class FitDevelopmentView:
    """The only corpus view supplied to the residual mapper."""

    protocol_sha256: str
    manifest_sha256: str
    example_count: int
    family_ids: tuple[str, ...]
    role: str = "calibration_a_fit"

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="protocol_sha256")
        _require_sha256(self.manifest_sha256, label="manifest_sha256")
        _positive_integer(self.example_count, label="example_count")
        _canonical_identifiers(self.family_ids, label="family_ids")
        if self.role != "calibration_a_fit":
            raise ValueError("fit development role is immutable")


@dataclass(frozen=True, slots=True)
class SelectionDevelopmentView:
    """Reusable selection evidence for choosing among fit proposals."""

    protocol_sha256: str
    manifest_sha256: str
    example_count: int
    family_ids: tuple[str, ...]
    role: str = "calibration_a_selection"

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="protocol_sha256")
        _require_sha256(self.manifest_sha256, label="manifest_sha256")
        _positive_integer(self.example_count, label="example_count")
        _canonical_identifiers(self.family_ids, label="family_ids")
        if self.role != "calibration_a_selection":
            raise ValueError("selection development role is immutable")


@dataclass(frozen=True, slots=True)
class GuardDevelopmentView:
    """The once-only final veto view for a frozen challenger."""

    protocol_sha256: str
    manifest_sha256: str
    example_count: int
    family_ids: tuple[str, ...]
    role: str = "calibration_a_guard"

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="protocol_sha256")
        _require_sha256(self.manifest_sha256, label="manifest_sha256")
        _positive_integer(self.example_count, label="example_count")
        _canonical_identifiers(self.family_ids, label="family_ids")
        if self.role != "calibration_a_guard":
            raise ValueError("guard development role is immutable")


@dataclass(frozen=True, slots=True)
class ProgressiveResourceFootprint:
    """Complete deployable resource accounting for one candidate.

    ``support`` includes carriers, routers, normalization, lookup tables, and
    any other work that is neither part of the compiled graph nor retained
    source execution.  Keeping it explicit prevents those costs from being
    hidden while the loop grows.
    """

    candidate_execution_sha256: str
    accounting_artifact_sha256: str
    parameter_scope: str
    compute_scope: str
    runtime_id: str
    runtime_dtype: str
    sequence_scope_sha256: str
    compiled_learned_parameters: int
    retained_source_learned_parameters: int
    support_learned_parameters: int
    compiled_runtime_parameter_bytes: int
    retained_source_runtime_parameter_bytes: int
    support_runtime_parameter_bytes: int
    compiled_logical_macs_per_token: int
    retained_source_logical_macs_per_token: int
    support_logical_macs_per_token: int
    cost_complete: bool
    incomplete_cost_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_execution_sha256",
            "accounting_artifact_sha256",
            "sequence_scope_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name in (
            "parameter_scope",
            "compute_scope",
            "runtime_id",
            "runtime_dtype",
        ):
            _identifier(getattr(self, name), label=name)
        for name in (
            "compiled_learned_parameters",
            "retained_source_learned_parameters",
            "support_learned_parameters",
            "compiled_runtime_parameter_bytes",
            "retained_source_runtime_parameter_bytes",
            "support_runtime_parameter_bytes",
            "compiled_logical_macs_per_token",
            "retained_source_logical_macs_per_token",
            "support_logical_macs_per_token",
        ):
            _nonnegative_integer(getattr(self, name), label=name)
        if (
            self.total_learned_parameters <= 0
            or self.total_runtime_parameter_bytes <= 0
            or self.total_logical_macs_per_token <= 0
        ):
            raise ValueError(
                "candidate totals must be positive on every resource axis"
            )
        if type(self.cost_complete) is not bool:
            raise TypeError("cost_complete must be boolean")
        if type(self.incomplete_cost_reasons) is not tuple:
            raise TypeError("incomplete_cost_reasons must be a tuple")
        reasons = tuple(
            _identifier(reason, label="incomplete_cost_reasons[]")
            for reason in self.incomplete_cost_reasons
        )
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError(
                "incomplete_cost_reasons must be sorted and unique"
            )
        if self.cost_complete == bool(reasons):
            raise ValueError(
                "complete cost requires no reasons and incomplete cost "
                "requires at least one reason"
            )

    @property
    def total_learned_parameters(self) -> int:
        return (
            self.compiled_learned_parameters
            + self.retained_source_learned_parameters
            + self.support_learned_parameters
        )

    @property
    def total_runtime_parameter_bytes(self) -> int:
        return (
            self.compiled_runtime_parameter_bytes
            + self.retained_source_runtime_parameter_bytes
            + self.support_runtime_parameter_bytes
        )

    @property
    def total_logical_macs_per_token(self) -> int:
        return (
            self.compiled_logical_macs_per_token
            + self.retained_source_logical_macs_per_token
            + self.support_logical_macs_per_token
        )

    @property
    def receipt_sha256(self) -> str:
        """Bind every claimed cost and execution scope into one receipt."""

        return _payload_sha256("resource-footprint", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_execution_sha256": (
                self.candidate_execution_sha256
            ),
            "accounting_artifact_sha256": (
                self.accounting_artifact_sha256
            ),
            "scope": {
                "parameter": self.parameter_scope,
                "compute": self.compute_scope,
                "runtime_id": self.runtime_id,
                "runtime_dtype": self.runtime_dtype,
                "sequence_sha256": self.sequence_scope_sha256,
            },
            "learned_parameters": {
                "compiled": self.compiled_learned_parameters,
                "retained_source": (
                    self.retained_source_learned_parameters
                ),
                "support": self.support_learned_parameters,
                "total": self.total_learned_parameters,
            },
            "runtime_parameter_bytes": {
                "compiled": self.compiled_runtime_parameter_bytes,
                "retained_source": (
                    self.retained_source_runtime_parameter_bytes
                ),
                "support": self.support_runtime_parameter_bytes,
                "total": self.total_runtime_parameter_bytes,
            },
            "logical_macs_per_token": {
                "compiled": self.compiled_logical_macs_per_token,
                "retained_source": (
                    self.retained_source_logical_macs_per_token
                ),
                "support": self.support_logical_macs_per_token,
                "total": self.total_logical_macs_per_token,
            },
            "source_fallback_charged": True,
            "support_work_charged": True,
            "cost_complete": self.cost_complete,
            "incomplete_cost_reasons": self.incomplete_cost_reasons,
        }


@dataclass(frozen=True, slots=True)
class ProgressiveResourceBudget:
    """Hard fractions relative to the complete source-model baseline."""

    parameter_scope: str
    compute_scope: str
    runtime_id: str
    runtime_dtype: str
    sequence_scope_sha256: str
    source_learned_parameters: int
    source_runtime_parameter_bytes: int
    source_logical_macs_per_token: int
    max_total_parameter_fraction: float
    max_total_parameter_byte_fraction: float
    max_total_mac_fraction: float
    max_retained_source_parameter_fraction: float
    max_retained_source_parameter_byte_fraction: float
    max_retained_source_mac_fraction: float

    def __post_init__(self) -> None:
        for name in (
            "parameter_scope",
            "compute_scope",
            "runtime_id",
            "runtime_dtype",
        ):
            _identifier(getattr(self, name), label=name)
        _require_sha256(
            self.sequence_scope_sha256,
            label="sequence_scope_sha256",
        )
        for name in (
            "source_learned_parameters",
            "source_runtime_parameter_bytes",
            "source_logical_macs_per_token",
        ):
            _positive_integer(getattr(self, name), label=name)
        for name in (
            "max_total_parameter_fraction",
            "max_total_parameter_byte_fraction",
            "max_total_mac_fraction",
        ):
            _finite_positive(getattr(self, name), label=name)
        for name in (
            "max_retained_source_parameter_fraction",
            "max_retained_source_parameter_byte_fraction",
            "max_retained_source_mac_fraction",
        ):
            _unit_interval(getattr(self, name), label=name)

    def violations(
        self,
        resources: ProgressiveResourceFootprint,
    ) -> tuple[str, ...]:
        if not isinstance(resources, ProgressiveResourceFootprint):
            raise TypeError(
                "resources must be ProgressiveResourceFootprint"
            )
        if not resources.cost_complete:
            return ("cost_incomplete",)
        scope_checks = (
            ("parameter_scope", resources.parameter_scope, self.parameter_scope),
            ("compute_scope", resources.compute_scope, self.compute_scope),
            ("runtime_id", resources.runtime_id, self.runtime_id),
            ("runtime_dtype", resources.runtime_dtype, self.runtime_dtype),
            (
                "sequence_scope_sha256",
                resources.sequence_scope_sha256,
                self.sequence_scope_sha256,
            ),
        )
        scope_violations = tuple(
            f"incomparable_{label}"
            for label, observed, expected in scope_checks
            if observed != expected
        )
        if scope_violations:
            return scope_violations
        checks = (
            (
                "total_learned_parameters",
                resources.total_learned_parameters,
                self.source_learned_parameters
                * self.max_total_parameter_fraction,
            ),
            (
                "total_runtime_parameter_bytes",
                resources.total_runtime_parameter_bytes,
                self.source_runtime_parameter_bytes
                * self.max_total_parameter_byte_fraction,
            ),
            (
                "total_logical_macs_per_token",
                resources.total_logical_macs_per_token,
                self.source_logical_macs_per_token
                * self.max_total_mac_fraction,
            ),
            (
                "retained_source_learned_parameters",
                resources.retained_source_learned_parameters,
                self.source_learned_parameters
                * self.max_retained_source_parameter_fraction,
            ),
            (
                "retained_source_runtime_parameter_bytes",
                resources.retained_source_runtime_parameter_bytes,
                self.source_runtime_parameter_bytes
                * self.max_retained_source_parameter_byte_fraction,
            ),
            (
                "retained_source_logical_macs_per_token",
                resources.retained_source_logical_macs_per_token,
                self.source_logical_macs_per_token
                * self.max_retained_source_mac_fraction,
            ),
        )
        return tuple(
            label for label, observed, maximum in checks
            if observed > maximum
        )

    def allows(self, resources: ProgressiveResourceFootprint) -> bool:
        return not self.violations(resources)

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": {
                "parameter": self.parameter_scope,
                "compute": self.compute_scope,
                "runtime_id": self.runtime_id,
                "runtime_dtype": self.runtime_dtype,
                "sequence_sha256": self.sequence_scope_sha256,
            },
            "source_baseline": {
                "learned_parameters": self.source_learned_parameters,
                "runtime_parameter_bytes": (
                    self.source_runtime_parameter_bytes
                ),
                "logical_macs_per_token": (
                    self.source_logical_macs_per_token
                ),
            },
            "max_total_fraction": {
                "learned_parameters": (
                    self.max_total_parameter_fraction
                ),
                "runtime_parameter_bytes": (
                    self.max_total_parameter_byte_fraction
                ),
                "logical_macs_per_token": self.max_total_mac_fraction,
            },
            "max_retained_source_fraction": {
                "learned_parameters": (
                    self.max_retained_source_parameter_fraction
                ),
                "runtime_parameter_bytes": (
                    self.max_retained_source_parameter_byte_fraction
                ),
                "logical_macs_per_token": (
                    self.max_retained_source_mac_fraction
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ProgressiveBehavioralFidelity:
    """One source-authoritative NLL/KL/agreement measurement family."""

    absolute_delta_nll_per_token: float
    source_to_candidate_kl_per_token: float
    top1_agreement_to_source: float
    per_prompt_p90_absolute_delta_nll_per_token: float
    per_prompt_p10_top1_agreement_to_source: float

    def __post_init__(self) -> None:
        for name in (
            "absolute_delta_nll_per_token",
            "source_to_candidate_kl_per_token",
            "per_prompt_p90_absolute_delta_nll_per_token",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), label=name),
            )
        for name in (
            "top1_agreement_to_source",
            "per_prompt_p10_top1_agreement_to_source",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), label=name),
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "absolute_delta_nll_per_token": (
                self.absolute_delta_nll_per_token
            ),
            "source_to_candidate_kl_per_token": (
                self.source_to_candidate_kl_per_token
            ),
            "top1_agreement_to_source": (
                self.top1_agreement_to_source
            ),
            "per_prompt_p90_absolute_delta_nll_per_token": (
                self.per_prompt_p90_absolute_delta_nll_per_token
            ),
            "per_prompt_p10_top1_agreement_to_source": (
                self.per_prompt_p10_top1_agreement_to_source
            ),
        }


@dataclass(frozen=True, slots=True)
class ProgressiveBehavioralTargets:
    """Thresholds shared by one behavioral measurement family."""

    absolute_delta_nll_per_token_max: float
    source_to_candidate_kl_per_token_max: float
    top1_agreement_to_source_min: float
    per_prompt_p90_absolute_delta_nll_per_token_max: float
    per_prompt_p10_top1_agreement_to_source_min: float

    def __post_init__(self) -> None:
        for name in (
            "absolute_delta_nll_per_token_max",
            "source_to_candidate_kl_per_token_max",
            "per_prompt_p90_absolute_delta_nll_per_token_max",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), label=name),
            )
        for name in (
            "top1_agreement_to_source_min",
            "per_prompt_p10_top1_agreement_to_source_min",
        ):
            value = _unit_interval(getattr(self, name), label=name)
            if value >= 1.0:
                raise ValueError(
                    f"{name} must be below 1 for normalized repair scoring"
                )
            object.__setattr__(self, name, value)

    def normalized_ratios(
        self,
        fidelity: ProgressiveBehavioralFidelity,
        *,
        prefix: str,
    ) -> dict[str, float]:
        if not isinstance(fidelity, ProgressiveBehavioralFidelity):
            raise TypeError(
                "fidelity must be ProgressiveBehavioralFidelity"
            )
        return {
            f"{prefix}.absolute_delta_nll_per_token": (
                fidelity.absolute_delta_nll_per_token
                / self.absolute_delta_nll_per_token_max
            ),
            f"{prefix}.source_to_candidate_kl_per_token": (
                fidelity.source_to_candidate_kl_per_token
                / self.source_to_candidate_kl_per_token_max
            ),
            f"{prefix}.top1_agreement_to_source": (
                (1.0 - fidelity.top1_agreement_to_source)
                / (1.0 - self.top1_agreement_to_source_min)
            ),
            (
                f"{prefix}."
                "per_prompt_p90_absolute_delta_nll_per_token"
            ): (
                fidelity.per_prompt_p90_absolute_delta_nll_per_token
                / self.per_prompt_p90_absolute_delta_nll_per_token_max
            ),
            (
                f"{prefix}."
                "per_prompt_p10_top1_agreement_to_source"
            ): (
                (
                    1.0
                    - fidelity.per_prompt_p10_top1_agreement_to_source
                )
                / (
                    1.0
                    - self.per_prompt_p10_top1_agreement_to_source_min
                )
            ),
        }

    def to_dict(self) -> dict[str, float]:
        return {
            "absolute_delta_nll_per_token_max": (
                self.absolute_delta_nll_per_token_max
            ),
            "source_to_candidate_kl_per_token_max": (
                self.source_to_candidate_kl_per_token_max
            ),
            "top1_agreement_to_source_min": (
                self.top1_agreement_to_source_min
            ),
            "per_prompt_p90_absolute_delta_nll_per_token_max": (
                self.per_prompt_p90_absolute_delta_nll_per_token_max
            ),
            "per_prompt_p10_top1_agreement_to_source_min": (
                self.per_prompt_p10_top1_agreement_to_source_min
            ),
        }


@dataclass(frozen=True, slots=True)
class ProgressiveFidelity:
    """Complete candidate, projection, and carrier guard measurements."""

    candidate_behavior: ProgressiveBehavioralFidelity
    projection_oracle_behavior: ProgressiveBehavioralFidelity
    carrier_oracle_behavior: ProgressiveBehavioralFidelity
    operator_nrmse: float
    boundary_relative_error: float
    boundary_cosine: float
    valid_target_coverage: float
    worst_family_boundary_relative_error: float
    worst_family_boundary_cosine: float
    minimum_family_source_modal_signal_l2_norm: float
    projection_full_width_relative_error: float
    projection_full_width_cosine: float
    worst_family_projection_relative_error: float
    worst_family_projection_cosine: float
    minimum_family_source_full_width_signal_l2_norm: float

    def __post_init__(self) -> None:
        for name in (
            "candidate_behavior",
            "projection_oracle_behavior",
            "carrier_oracle_behavior",
        ):
            if not isinstance(
                getattr(self, name),
                ProgressiveBehavioralFidelity,
            ):
                raise TypeError(
                    f"{name} must be ProgressiveBehavioralFidelity"
                )
        for name in (
            "operator_nrmse",
            "boundary_relative_error",
            "worst_family_boundary_relative_error",
            "minimum_family_source_modal_signal_l2_norm",
            "projection_full_width_relative_error",
            "worst_family_projection_relative_error",
            "minimum_family_source_full_width_signal_l2_norm",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), label=name),
            )
        for name in (
            "boundary_cosine",
            "valid_target_coverage",
            "worst_family_boundary_cosine",
            "projection_full_width_cosine",
            "worst_family_projection_cosine",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), label=name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_behavior": self.candidate_behavior.to_dict(),
            "projection_oracle_behavior": (
                self.projection_oracle_behavior.to_dict()
            ),
            "carrier_oracle_behavior": (
                self.carrier_oracle_behavior.to_dict()
            ),
            "operator_nrmse": self.operator_nrmse,
            "boundary": {
                "relative_error": self.boundary_relative_error,
                "cosine": self.boundary_cosine,
                "valid_target_coverage": self.valid_target_coverage,
                "worst_family_relative_error": (
                    self.worst_family_boundary_relative_error
                ),
                "worst_family_cosine": (
                    self.worst_family_boundary_cosine
                ),
                "minimum_family_source_signal_l2_norm": (
                    self.minimum_family_source_modal_signal_l2_norm
                ),
            },
            "projection_capacity": {
                "full_width_relative_error": (
                    self.projection_full_width_relative_error
                ),
                "full_width_cosine": (
                    self.projection_full_width_cosine
                ),
                "worst_family_relative_error": (
                    self.worst_family_projection_relative_error
                ),
                "worst_family_cosine": (
                    self.worst_family_projection_cosine
                ),
                "minimum_family_source_signal_l2_norm": (
                    self.minimum_family_source_full_width_signal_l2_norm
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ProgressiveFidelityTargets:
    """Complete gates and the normalized burden used during repair."""

    candidate_behavior: ProgressiveBehavioralTargets
    projection_oracle_behavior: ProgressiveBehavioralTargets
    carrier_oracle_behavior: ProgressiveBehavioralTargets
    operator_nrmse_max: float
    boundary_relative_error_max: float
    boundary_cosine_min: float
    valid_target_coverage_min: float
    worst_family_boundary_relative_error_max: float
    worst_family_boundary_cosine_min: float
    minimum_family_source_modal_signal_l2_norm: float
    projection_full_width_relative_error_max: float
    projection_full_width_cosine_min: float
    worst_family_projection_relative_error_max: float
    worst_family_projection_cosine_min: float
    minimum_family_source_full_width_signal_l2_norm: float

    def __post_init__(self) -> None:
        for name in (
            "candidate_behavior",
            "projection_oracle_behavior",
            "carrier_oracle_behavior",
        ):
            if not isinstance(
                getattr(self, name),
                ProgressiveBehavioralTargets,
            ):
                raise TypeError(
                    f"{name} must be ProgressiveBehavioralTargets"
                )
        for name in (
            "operator_nrmse_max",
            "boundary_relative_error_max",
            "worst_family_boundary_relative_error_max",
            "minimum_family_source_modal_signal_l2_norm",
            "projection_full_width_relative_error_max",
            "worst_family_projection_relative_error_max",
            "minimum_family_source_full_width_signal_l2_norm",
        ):
            object.__setattr__(
                self,
                name,
                _finite_positive(getattr(self, name), label=name),
            )
        for name in (
            "boundary_cosine_min",
            "valid_target_coverage_min",
            "worst_family_boundary_cosine_min",
            "projection_full_width_cosine_min",
            "worst_family_projection_cosine_min",
        ):
            value = _unit_interval(getattr(self, name), label=name)
            if value >= 1.0:
                raise ValueError(
                    f"{name} must be below 1 for normalized repair scoring"
                )
            object.__setattr__(self, name, value)

    @staticmethod
    def _minimum_ratio(observed: float, minimum: float) -> float:
        return 2.0 - min(observed / minimum, 1.0)

    def normalized_ratios(
        self,
        fidelity: ProgressiveFidelity,
    ) -> dict[str, float]:
        if not isinstance(fidelity, ProgressiveFidelity):
            raise TypeError("fidelity must be ProgressiveFidelity")
        ratios = {}
        ratios.update(
            self.candidate_behavior.normalized_ratios(
                fidelity.candidate_behavior,
                prefix="candidate_behavior",
            )
        )
        ratios.update(
            self.projection_oracle_behavior.normalized_ratios(
                fidelity.projection_oracle_behavior,
                prefix="projection_oracle_behavior",
            )
        )
        ratios.update(
            self.carrier_oracle_behavior.normalized_ratios(
                fidelity.carrier_oracle_behavior,
                prefix="carrier_oracle_behavior",
            )
        )
        ratios.update(
            {
                "operator_nrmse": (
                    fidelity.operator_nrmse / self.operator_nrmse_max
                ),
                "boundary.relative_error": (
                    fidelity.boundary_relative_error
                    / self.boundary_relative_error_max
                ),
                "boundary.cosine": (
                    (1.0 - fidelity.boundary_cosine)
                    / (1.0 - self.boundary_cosine_min)
                ),
                "boundary.valid_target_coverage": (
                    (1.0 - fidelity.valid_target_coverage)
                    / (1.0 - self.valid_target_coverage_min)
                ),
                "boundary.worst_family_relative_error": (
                    fidelity.worst_family_boundary_relative_error
                    / self.worst_family_boundary_relative_error_max
                ),
                "boundary.worst_family_cosine": (
                    (1.0 - fidelity.worst_family_boundary_cosine)
                    / (1.0 - self.worst_family_boundary_cosine_min)
                ),
                "boundary.minimum_family_source_signal": (
                    self._minimum_ratio(
                        fidelity.minimum_family_source_modal_signal_l2_norm,
                        self.minimum_family_source_modal_signal_l2_norm,
                    )
                ),
                "projection.full_width_relative_error": (
                    fidelity.projection_full_width_relative_error
                    / self.projection_full_width_relative_error_max
                ),
                "projection.full_width_cosine": (
                    (1.0 - fidelity.projection_full_width_cosine)
                    / (1.0 - self.projection_full_width_cosine_min)
                ),
                "projection.worst_family_relative_error": (
                    fidelity.worst_family_projection_relative_error
                    / self.worst_family_projection_relative_error_max
                ),
                "projection.worst_family_cosine": (
                    (1.0 - fidelity.worst_family_projection_cosine)
                    / (1.0 - self.worst_family_projection_cosine_min)
                ),
                "projection.minimum_family_source_signal": (
                    self._minimum_ratio(
                        fidelity.minimum_family_source_full_width_signal_l2_norm,
                        self.minimum_family_source_full_width_signal_l2_norm,
                    )
                ),
            }
        )
        return ratios

    def burden(self, fidelity: ProgressiveFidelity) -> float:
        return max(self.normalized_ratios(fidelity).values())

    def passes(self, fidelity: ProgressiveFidelity) -> bool:
        return all(
            value <= 1.0
            for value in self.normalized_ratios(fidelity).values()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_behavior": self.candidate_behavior.to_dict(),
            "projection_oracle_behavior": (
                self.projection_oracle_behavior.to_dict()
            ),
            "carrier_oracle_behavior": (
                self.carrier_oracle_behavior.to_dict()
            ),
            "operator_nrmse_max": self.operator_nrmse_max,
            "boundary": {
                "relative_error_max": self.boundary_relative_error_max,
                "cosine_min": self.boundary_cosine_min,
                "valid_target_coverage_min": (
                    self.valid_target_coverage_min
                ),
                "worst_family_relative_error_max": (
                    self.worst_family_boundary_relative_error_max
                ),
                "worst_family_cosine_min": (
                    self.worst_family_boundary_cosine_min
                ),
                "minimum_family_source_signal_l2_norm": (
                    self.minimum_family_source_modal_signal_l2_norm
                ),
            },
            "projection_capacity": {
                "full_width_relative_error_max": (
                    self.projection_full_width_relative_error_max
                ),
                "full_width_cosine_min": (
                    self.projection_full_width_cosine_min
                ),
                "worst_family_relative_error_max": (
                    self.worst_family_projection_relative_error_max
                ),
                "worst_family_cosine_min": (
                    self.worst_family_projection_cosine_min
                ),
                "minimum_family_source_signal_l2_norm": (
                    self.minimum_family_source_full_width_signal_l2_norm
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ProgressiveCompilationProtocol:
    """Frozen policy for one repeated development campaign."""

    protocol_id: str
    source_model_sha256: str
    seed_candidate_artifact_sha256: str
    seed_candidate_execution_sha256: str
    seed_runtime_binding_sha256: str
    seed_resource_receipt_sha256: str
    seed_lineage_sha256s: tuple[str, ...]
    corpus: DevelopmentCorpus
    forbidden_assessment_manifest_sha256s: tuple[str, ...]
    fidelity_targets: ProgressiveFidelityTargets
    resource_budget: ProgressiveResourceBudget
    max_iterations: int = 16
    max_proposals_per_iteration: int = 8
    minimum_repair_relative_burden_reduction: float = 0.02
    maximum_repair_axis_regression_fraction: float = 0.25
    compact_after_fidelity: bool = True
    artifact_sha256: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        _identifier(self.protocol_id, label="protocol_id")
        _require_sha256(
            self.source_model_sha256,
            label="source_model_sha256",
        )
        _require_sha256(
            self.seed_candidate_artifact_sha256,
            label="seed_candidate_artifact_sha256",
        )
        _require_sha256(
            self.seed_candidate_execution_sha256,
            label="seed_candidate_execution_sha256",
        )
        _require_sha256(
            self.seed_runtime_binding_sha256,
            label="seed_runtime_binding_sha256",
        )
        _require_sha256(
            self.seed_resource_receipt_sha256,
            label="seed_resource_receipt_sha256",
        )
        _canonical_sha256s(
            self.seed_lineage_sha256s,
            label="seed_lineage_sha256s",
        )
        if not isinstance(self.corpus, DevelopmentCorpus):
            raise TypeError("corpus must be DevelopmentCorpus")
        forbidden = _canonical_sha256s(
            self.forbidden_assessment_manifest_sha256s,
            label="forbidden_assessment_manifest_sha256s",
        )
        if (
            self.corpus.fit_manifest_sha256 in forbidden
            or self.corpus.selection_manifest_sha256 in forbidden
            or self.corpus.guard_manifest_sha256 in forbidden
        ):
            raise ValueError(
                "development manifests cannot reuse an assessment manifest"
            )
        if not isinstance(
            self.fidelity_targets,
            ProgressiveFidelityTargets,
        ):
            raise TypeError(
                "fidelity_targets must be ProgressiveFidelityTargets"
            )
        if not isinstance(
            self.resource_budget,
            ProgressiveResourceBudget,
        ):
            raise TypeError(
                "resource_budget must be ProgressiveResourceBudget"
            )
        _positive_integer(self.max_iterations, label="max_iterations")
        _positive_integer(
            self.max_proposals_per_iteration,
            label="max_proposals_per_iteration",
        )
        reduction = _unit_interval(
            self.minimum_repair_relative_burden_reduction,
            label="minimum_repair_relative_burden_reduction",
        )
        if reduction >= 1.0:
            raise ValueError(
                "minimum repair reduction must be below one"
            )
        _finite_nonnegative(
            self.maximum_repair_axis_regression_fraction,
            label="maximum_repair_axis_regression_fraction",
        )
        if type(self.compact_after_fidelity) is not bool:
            raise TypeError("compact_after_fidelity must be boolean")
        computed = _payload_sha256(
            "protocol",
            self._payload(),
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="protocol artifact_sha256",
                )
                != computed
            ):
                raise ValueError("progressive protocol hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": PROGRESSIVE_COMPILATION_SCHEMA,
            "format_version": PROGRESSIVE_COMPILATION_FORMAT_VERSION,
            "protocol_id": self.protocol_id,
            "source_model_sha256": self.source_model_sha256,
            "seed_candidate": {
                "artifact_sha256": (
                    self.seed_candidate_artifact_sha256
                ),
                "execution_sha256": (
                    self.seed_candidate_execution_sha256
                ),
                "runtime_binding_sha256": (
                    self.seed_runtime_binding_sha256
                ),
                "resource_receipt_sha256": (
                    self.seed_resource_receipt_sha256
                ),
                "lineage_sha256s": self.seed_lineage_sha256s,
            },
            "corpus": self.corpus.to_dict(),
            "forbidden_assessment_manifest_sha256s": (
                self.forbidden_assessment_manifest_sha256s
            ),
            "fidelity_targets": self.fidelity_targets.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "loop": {
                "max_iterations": self.max_iterations,
                "max_proposals_per_iteration": (
                    self.max_proposals_per_iteration
                ),
                "minimum_repair_relative_burden_reduction": (
                    self.minimum_repair_relative_burden_reduction
                ),
                "maximum_repair_axis_regression_fraction": (
                    self.maximum_repair_axis_regression_fraction
                ),
                "compact_after_fidelity": self.compact_after_fidelity,
                "failing_phase": "map_repair_selection_validate",
                "passing_phase": "map_compact_selection_validate",
                "terminal_phase": "freeze_then_single_guard_veto",
            },
            "assessment_access": {
                "authorized": False,
                "identities_registered_for_rejection_only": True,
            },
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    def fit_view(self) -> FitDevelopmentView:
        return FitDevelopmentView(
            protocol_sha256=self.artifact_sha256,
            manifest_sha256=self.corpus.fit_manifest_sha256,
            example_count=self.corpus.fit_example_count,
            family_ids=self.corpus.fit_family_ids,
        )

    def guard_view(self) -> GuardDevelopmentView:
        return GuardDevelopmentView(
            protocol_sha256=self.artifact_sha256,
            manifest_sha256=self.corpus.guard_manifest_sha256,
            example_count=self.corpus.guard_example_count,
            family_ids=self.corpus.guard_family_ids,
        )

    def selection_view(self) -> SelectionDevelopmentView:
        return SelectionDevelopmentView(
            protocol_sha256=self.artifact_sha256,
            manifest_sha256=self.corpus.selection_manifest_sha256,
            example_count=self.corpus.selection_example_count,
            family_ids=self.corpus.selection_family_ids,
        )

    def validate_integrity(self) -> None:
        if (
            _payload_sha256("protocol", self._payload())
            != self.artifact_sha256
        ):
            raise ValueError("progressive protocol hash mismatch")


@dataclass(frozen=True, slots=True)
class ProgressiveCandidate:
    """One immutable executable candidate in a parent-bound lineage."""

    candidate_id: str
    iteration: int
    artifact_sha256: str
    execution_sha256: str
    runtime_binding_sha256: str
    resources: ProgressiveResourceFootprint
    mutation_kind: MutationKind
    parent_artifact_sha256: str | None = None
    proposal_sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, label="candidate_id")
        _nonnegative_integer(self.iteration, label="iteration")
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        _require_sha256(self.execution_sha256, label="execution_sha256")
        _require_sha256(
            self.runtime_binding_sha256,
            label="runtime_binding_sha256",
        )
        if not isinstance(self.resources, ProgressiveResourceFootprint):
            raise TypeError(
                "resources must be ProgressiveResourceFootprint"
            )
        if (
            self.resources.candidate_execution_sha256
            != self.execution_sha256
        ):
            raise ValueError(
                "resource accounting does not bind candidate execution"
            )
        if self.mutation_kind == "seed":
            if (
                self.iteration != 0
                or self.parent_artifact_sha256 is not None
                or self.proposal_sha256 is not None
            ):
                raise ValueError(
                    "seed candidate must be iteration zero without a parent"
                )
        else:
            if self.iteration <= 0:
                raise ValueError(
                    "mutated candidates require a positive iteration"
                )
            _require_sha256(
                self.parent_artifact_sha256,
                label="parent_artifact_sha256",
            )
            _require_sha256(
                self.proposal_sha256,
                label="proposal_sha256",
            )

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("candidate", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "iteration": self.iteration,
            "artifact_sha256": self.artifact_sha256,
            "execution_sha256": self.execution_sha256,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "mutation_kind": self.mutation_kind,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "proposal_sha256": self.proposal_sha256,
            "resources": self.resources.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ResidualTarget:
    """One ranked fit-only direction offered to the mutation planner."""

    rank: int
    location: str
    direction_sha256: str
    residual_energy_fraction: float
    loss_coupling: float
    jvp_gain: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.rank, label="rank")
        _identifier(self.location, label="location")
        _require_sha256(
            self.direction_sha256,
            label="direction_sha256",
        )
        object.__setattr__(
            self,
            "residual_energy_fraction",
            _unit_interval(
                self.residual_energy_fraction,
                label="residual_energy_fraction",
            ),
        )
        for name in ("loss_coupling", "jvp_gain"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), label=name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "location": self.location,
            "direction_sha256": self.direction_sha256,
            "residual_energy_fraction": self.residual_energy_fraction,
            "loss_coupling": self.loss_coupling,
            "jvp_gain": self.jvp_gain,
        }


@dataclass(frozen=True, slots=True)
class ResidualMap:
    """Scalar metadata for one fit-only residual analysis."""

    protocol_sha256: str
    fit_manifest_sha256: str
    candidate_artifact_sha256: str
    candidate_receipt_sha256: str
    iteration: int
    mapper_id: str
    mapper_version: int
    analysis_artifact_sha256: str
    targets: tuple[ResidualTarget, ...]

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "fit_manifest_sha256",
            "candidate_artifact_sha256",
            "candidate_receipt_sha256",
            "analysis_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _nonnegative_integer(self.iteration, label="iteration")
        _identifier(self.mapper_id, label="mapper_id")
        _positive_integer(self.mapper_version, label="mapper_version")
        if (
            type(self.targets) is not tuple
            or not self.targets
            or any(
                not isinstance(target, ResidualTarget)
                for target in self.targets
            )
        ):
            raise ValueError(
                "targets must be a nonempty tuple of ResidualTarget"
            )
        ranks = tuple(target.rank for target in self.targets)
        if ranks != tuple(range(len(self.targets))):
            raise ValueError("residual target ranks must be contiguous")
        identities = tuple(
            (target.location, target.direction_sha256)
            for target in self.targets
        )
        if len(set(identities)) != len(identities):
            raise ValueError("residual targets cannot contain duplicates")

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("residual-map", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "role": "calibration_a_fit_residual_map",
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "candidate_artifact_sha256": (
                self.candidate_artifact_sha256
            ),
            "candidate_receipt_sha256": (
                self.candidate_receipt_sha256
            ),
            "iteration": self.iteration,
            "mapper": {
                "id": self.mapper_id,
                "version": self.mapper_version,
            },
            "analysis_artifact_sha256": (
                self.analysis_artifact_sha256
            ),
            "targets": tuple(target.to_dict() for target in self.targets),
            "raw_prompt_or_activation_payload_in_receipt": False,
        }


@dataclass(frozen=True, slots=True)
class MutationProposal:
    """One exact, budgetable mutation proposed from a residual map."""

    proposal_id: str
    phase: ProgressivePhase
    mutation_kind: MutationKind
    parent_artifact_sha256: str
    parent_receipt_sha256: str
    residual_map_sha256: str
    recipe_sha256: str
    target_ranks: tuple[int, ...]
    resources: ProgressiveResourceFootprint

    def __post_init__(self) -> None:
        _identifier(self.proposal_id, label="proposal_id")
        if self.phase not in ("repair", "compact"):
            raise ValueError("unsupported progressive phase")
        allowed = (
            _REPAIR_MUTATIONS
            if self.phase == "repair"
            else _COMPACT_MUTATIONS
        )
        if self.mutation_kind not in allowed:
            raise ValueError(
                f"{self.mutation_kind!r} is not a {self.phase} mutation"
            )
        for name in (
            "parent_artifact_sha256",
            "parent_receipt_sha256",
            "residual_map_sha256",
            "recipe_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            type(self.target_ranks) is not tuple
            or not self.target_ranks
            or any(
                type(rank) is not int or rank < 0
                for rank in self.target_ranks
            )
            or self.target_ranks
            != tuple(sorted(set(self.target_ranks)))
        ):
            raise ValueError(
                "target_ranks must be a sorted nonempty unique tuple"
            )
        if not isinstance(self.resources, ProgressiveResourceFootprint):
            raise TypeError(
                "resources must be ProgressiveResourceFootprint"
            )

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("mutation-proposal", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "phase": self.phase,
            "mutation_kind": self.mutation_kind,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "parent_receipt_sha256": self.parent_receipt_sha256,
            "residual_map_sha256": self.residual_map_sha256,
            "recipe_sha256": self.recipe_sha256,
            "target_ranks": self.target_ranks,
            "resources": self.resources.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationCoverage:
    """Manifest membership and completeness for one scalar evaluation."""

    manifest_sha256: str
    expected_example_count: int
    observed_example_count: int
    expected_family_ids: tuple[str, ...]
    observed_family_ids: tuple[str, ...]
    supervised_token_count: int
    membership_receipt_sha256: str
    model_inputs_receipt_sha256: str
    complete: bool

    def __post_init__(self) -> None:
        for name in (
            "manifest_sha256",
            "membership_receipt_sha256",
            "model_inputs_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name in (
            "expected_example_count",
            "observed_example_count",
            "supervised_token_count",
        ):
            _positive_integer(getattr(self, name), label=name)
        expected = _canonical_identifiers(
            self.expected_family_ids,
            label="expected_family_ids",
        )
        observed = _canonical_identifiers(
            self.observed_family_ids,
            label="observed_family_ids",
        )
        if not set(observed).issubset(expected):
            raise ValueError(
                "observed families must belong to the expected manifest"
            )
        if type(self.complete) is not bool:
            raise TypeError("complete must be boolean")
        derived_complete = (
            self.observed_example_count == self.expected_example_count
            and observed == expected
        )
        if self.complete != derived_complete:
            raise ValueError(
                "coverage completeness differs from counts and families"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "expected_example_count": self.expected_example_count,
            "observed_example_count": self.observed_example_count,
            "expected_family_ids": self.expected_family_ids,
            "observed_family_ids": self.observed_family_ids,
            "supervised_token_count": self.supervised_token_count,
            "membership_receipt_sha256": (
                self.membership_receipt_sha256
            ),
            "model_inputs_receipt_sha256": (
                self.model_inputs_receipt_sha256
            ),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One scalar-only family-disjoint selection or guard evaluation."""

    protocol_sha256: str
    development_role: Literal[
        "calibration_a_selection",
        "calibration_a_guard",
    ]
    manifest_sha256: str
    candidate_artifact_sha256: str
    candidate_receipt_sha256: str
    evaluation_artifact_sha256: str
    coverage: DevelopmentEvaluationCoverage
    fidelity: ProgressiveFidelity
    resources: ProgressiveResourceFootprint
    challenger_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "manifest_sha256",
            "candidate_artifact_sha256",
            "candidate_receipt_sha256",
            "evaluation_artifact_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.development_role not in (
            "calibration_a_selection",
            "calibration_a_guard",
        ):
            raise ValueError("unsupported candidate evaluation role")
        if not isinstance(
            self.coverage,
            DevelopmentEvaluationCoverage,
        ):
            raise TypeError(
                "coverage must be DevelopmentEvaluationCoverage"
            )
        if self.coverage.manifest_sha256 != self.manifest_sha256:
            raise ValueError(
                "evaluation coverage manifest binding differs"
            )
        if self.development_role == "calibration_a_selection":
            if self.challenger_receipt_sha256 is not None:
                raise ValueError(
                    "selection evaluation cannot bind a frozen challenger"
                )
        else:
            _require_sha256(
                self.challenger_receipt_sha256,
                label="challenger_receipt_sha256",
            )
        if not isinstance(self.fidelity, ProgressiveFidelity):
            raise TypeError("fidelity must be ProgressiveFidelity")
        if not isinstance(self.resources, ProgressiveResourceFootprint):
            raise TypeError(
                "resources must be ProgressiveResourceFootprint"
            )

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("candidate-evaluation", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "role": self.development_role,
            "manifest_sha256": self.manifest_sha256,
            "candidate_artifact_sha256": (
                self.candidate_artifact_sha256
            ),
            "candidate_receipt_sha256": (
                self.candidate_receipt_sha256
            ),
            "evaluation_artifact_sha256": (
                self.evaluation_artifact_sha256
            ),
            "coverage": self.coverage.to_dict(),
            "challenger_receipt_sha256": (
                self.challenger_receipt_sha256
            ),
            "fidelity": self.fidelity.to_dict(),
            "resources": self.resources.to_dict(),
            "tensor_payload_exposed": False,
        }


class ResidualMapper(Protocol):
    def __call__(
        self,
        candidate: ProgressiveCandidate,
        fit: FitDevelopmentView,
    ) -> ResidualMap: ...


class MutationProposer(Protocol):
    def __call__(
        self,
        candidate: ProgressiveCandidate,
        residual_map: ResidualMap,
        phase: ProgressivePhase,
    ) -> Sequence[MutationProposal]: ...


class CandidateBuilder(Protocol):
    def __call__(
        self,
        parent: ProgressiveCandidate,
        proposal: MutationProposal,
    ) -> ProgressiveCandidate: ...


class GuardEvaluator(Protocol):
    def __call__(
        self,
        challenger: FrozenCalibrationAChallenger,
        guard: GuardDevelopmentView,
    ) -> CandidateEvaluation: ...


class SelectionEvaluator(Protocol):
    def __call__(
        self,
        candidate: ProgressiveCandidate,
        selection: SelectionDevelopmentView,
    ) -> CandidateEvaluation: ...


@dataclass(frozen=True, slots=True)
class ProgressiveIterationReceipt:
    """One deterministic map/mutate/guard-selection transaction."""

    iteration: int
    phase: ProgressivePhase
    parent_candidate_receipt_sha256: str
    residual_map_receipt_sha256: str
    proposal_receipt_sha256s: tuple[str, ...]
    evaluation_receipt_sha256s: tuple[str, ...]
    accepted_candidate_receipt_sha256: str | None
    accepted_evaluation_receipt_sha256: str | None
    decision: str

    def __post_init__(self) -> None:
        _positive_integer(self.iteration, label="iteration")
        if self.phase not in ("repair", "compact"):
            raise ValueError("unsupported progressive phase")
        _require_sha256(
            self.parent_candidate_receipt_sha256,
            label="parent_candidate_receipt_sha256",
        )
        _require_sha256(
            self.residual_map_receipt_sha256,
            label="residual_map_receipt_sha256",
        )
        for label, values in (
            (
                "proposal_receipt_sha256s",
                self.proposal_receipt_sha256s,
            ),
            (
                "evaluation_receipt_sha256s",
                self.evaluation_receipt_sha256s,
            ),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{label} must be a tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} cannot contain duplicates")
            for value in values:
                _require_sha256(value, label=f"{label}[]")
        if self.accepted_candidate_receipt_sha256 is not None:
            _require_sha256(
                self.accepted_candidate_receipt_sha256,
                label="accepted_candidate_receipt_sha256",
            )
        if self.accepted_evaluation_receipt_sha256 is not None:
            _require_sha256(
                self.accepted_evaluation_receipt_sha256,
                label="accepted_evaluation_receipt_sha256",
            )
        if (
            self.accepted_candidate_receipt_sha256 is None
        ) != (
            self.accepted_evaluation_receipt_sha256 is None
        ):
            raise ValueError(
                "accepted candidate and evaluation receipts must appear "
                "together"
            )
        if (
            self.accepted_evaluation_receipt_sha256 is not None
            and self.accepted_evaluation_receipt_sha256
            not in self.evaluation_receipt_sha256s
        ):
            raise ValueError(
                "accepted evaluation must belong to this iteration"
            )
        _identifier(self.decision, label="decision")

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("iteration", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "phase": self.phase,
            "parent_candidate_receipt_sha256": (
                self.parent_candidate_receipt_sha256
            ),
            "residual_map_receipt_sha256": (
                self.residual_map_receipt_sha256
            ),
            "proposal_receipt_sha256s": self.proposal_receipt_sha256s,
            "evaluation_receipt_sha256s": (
                self.evaluation_receipt_sha256s
            ),
            "accepted_candidate_receipt_sha256": (
                self.accepted_candidate_receipt_sha256
            ),
            "accepted_evaluation_receipt_sha256": (
                self.accepted_evaluation_receipt_sha256
            ),
            "decision": self.decision,
        }


@dataclass(frozen=True, slots=True)
class FrozenCalibrationAChallenger:
    """The complete immutable transcript prefix exposed to the A guard."""

    protocol_sha256: str
    seed_candidate_receipt_sha256: str
    seed_selection_evaluation_receipt_sha256: str
    candidate: ProgressiveCandidate
    selection_evaluation: CandidateEvaluation
    iteration_receipt_sha256s: tuple[str, ...]
    residual_map_receipt_sha256s: tuple[str, ...]
    proposal_receipt_sha256s: tuple[str, ...]
    candidate_archive_receipt_sha256s: tuple[str, ...]
    selection_archive_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "seed_candidate_receipt_sha256",
            "seed_selection_evaluation_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.candidate, ProgressiveCandidate):
            raise TypeError("candidate must be ProgressiveCandidate")
        if not isinstance(
            self.selection_evaluation,
            CandidateEvaluation,
        ):
            raise TypeError(
                "selection_evaluation must be CandidateEvaluation"
            )
        if (
            self.selection_evaluation.development_role
            != "calibration_a_selection"
            or self.selection_evaluation.protocol_sha256
            != self.protocol_sha256
            or self.selection_evaluation.candidate_artifact_sha256
            != self.candidate.artifact_sha256
            or self.selection_evaluation.candidate_receipt_sha256
            != self.candidate.receipt_sha256
            or self.selection_evaluation.resources
            != self.candidate.resources
        ):
            raise ValueError(
                "challenger selection evaluation binding differs"
            )
        for label, values in (
            (
                "iteration_receipt_sha256s",
                self.iteration_receipt_sha256s,
            ),
            (
                "residual_map_receipt_sha256s",
                self.residual_map_receipt_sha256s,
            ),
            (
                "proposal_receipt_sha256s",
                self.proposal_receipt_sha256s,
            ),
            (
                "candidate_archive_receipt_sha256s",
                self.candidate_archive_receipt_sha256s,
            ),
            (
                "selection_archive_receipt_sha256s",
                self.selection_archive_receipt_sha256s,
            ),
        ):
            if type(values) is not tuple:
                raise TypeError(f"{label} must be a tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} cannot contain duplicates")
            for value in values:
                _require_sha256(value, label=f"{label}[]")
        if (
            not self.candidate_archive_receipt_sha256s
            or not self.selection_archive_receipt_sha256s
            or self.candidate_archive_receipt_sha256s[0]
            != self.seed_candidate_receipt_sha256
            or self.selection_archive_receipt_sha256s[0]
            != self.seed_selection_evaluation_receipt_sha256
            or self.candidate.receipt_sha256
            not in self.candidate_archive_receipt_sha256s
            or self.selection_evaluation.receipt_sha256
            not in self.selection_archive_receipt_sha256s
        ):
            raise ValueError(
                "challenger does not bind its seed and active head archives"
            )

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("frozen-a-challenger", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "seed_candidate_receipt_sha256": (
                self.seed_candidate_receipt_sha256
            ),
            "seed_selection_evaluation_receipt_sha256": (
                self.seed_selection_evaluation_receipt_sha256
            ),
            "candidate": self.candidate.to_dict(),
            "selection_evaluation": (
                self.selection_evaluation.to_dict()
            ),
            "iteration_receipt_sha256s": (
                self.iteration_receipt_sha256s
            ),
            "residual_map_receipt_sha256s": (
                self.residual_map_receipt_sha256s
            ),
            "proposal_receipt_sha256s": self.proposal_receipt_sha256s,
            "candidate_archive_receipt_sha256s": (
                self.candidate_archive_receipt_sha256s
            ),
            "selection_archive_receipt_sha256s": (
                self.selection_archive_receipt_sha256s
            ),
            "guard_opened": False,
            "assessment_opened": False,
        }


@dataclass(frozen=True, slots=True)
class ProgressiveCompilationResult:
    """Terminal result of one bounded repeated development campaign."""

    protocol_sha256: str
    seed_candidate_receipt_sha256: str
    seed_selection_evaluation_receipt_sha256: str
    final_candidate: ProgressiveCandidate
    final_selection_evaluation: CandidateEvaluation
    frozen_challenger: FrozenCalibrationAChallenger | None
    guard_evaluation: CandidateEvaluation | None
    iterations: tuple[ProgressiveIterationReceipt, ...]
    residual_map_archive: tuple[ResidualMap, ...]
    proposal_archive: tuple[MutationProposal, ...]
    candidate_archive: tuple[ProgressiveCandidate, ...]
    selection_evaluation_archive: tuple[CandidateEvaluation, ...]
    status: ProgressiveStatus

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "seed_candidate_receipt_sha256",
            "seed_selection_evaluation_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.final_candidate, ProgressiveCandidate):
            raise TypeError(
                "final_candidate must be ProgressiveCandidate"
            )
        if not isinstance(
            self.final_selection_evaluation,
            CandidateEvaluation,
        ):
            raise TypeError(
                "final_selection_evaluation must be CandidateEvaluation"
            )
        if (
            self.frozen_challenger is not None
            and not isinstance(
                self.frozen_challenger,
                FrozenCalibrationAChallenger,
            )
        ):
            raise TypeError(
                "frozen_challenger must be "
                "FrozenCalibrationAChallenger or None"
            )
        if (
            self.guard_evaluation is not None
            and not isinstance(self.guard_evaluation, CandidateEvaluation)
        ):
            raise TypeError(
                "guard_evaluation must be CandidateEvaluation or None"
            )
        if (
            type(self.iterations) is not tuple
            or any(
                not isinstance(item, ProgressiveIterationReceipt)
                for item in self.iterations
            )
        ):
            raise TypeError(
                "iterations must be a tuple of iteration receipts"
            )
        if (
            type(self.proposal_archive) is not tuple
            or any(
                not isinstance(item, MutationProposal)
                for item in self.proposal_archive
            )
        ):
            raise TypeError(
                "proposal_archive must be a tuple of mutation proposals"
            )
        if (
            type(self.residual_map_archive) is not tuple
            or any(
                not isinstance(item, ResidualMap)
                for item in self.residual_map_archive
            )
        ):
            raise TypeError(
                "residual_map_archive must be a tuple of residual maps"
            )
        if (
            type(self.candidate_archive) is not tuple
            or not self.candidate_archive
            or any(
                not isinstance(item, ProgressiveCandidate)
                for item in self.candidate_archive
            )
        ):
            raise TypeError(
                "candidate_archive must be a nonempty candidate tuple"
            )
        if (
            type(self.selection_evaluation_archive) is not tuple
            or any(
                not isinstance(item, CandidateEvaluation)
                for item in self.selection_evaluation_archive
            )
            or len(self.selection_evaluation_archive)
            != len(self.candidate_archive)
        ):
            raise TypeError(
                "selection archive must contain one evaluation per "
                "candidate"
            )
        if self.status not in (
            "ready_for_candidate_binding",
            "rejected_by_guard",
            "stalled_fidelity",
            "stalled_budget",
            "max_iterations",
        ):
            raise ValueError("unsupported progressive result status")
        final_receipt = self.final_candidate.receipt_sha256
        selection = self.final_selection_evaluation
        if (
            selection.protocol_sha256 != self.protocol_sha256
            or selection.development_role
            != "calibration_a_selection"
            or selection.candidate_artifact_sha256
            != self.final_candidate.artifact_sha256
            or selection.candidate_receipt_sha256 != final_receipt
            or selection.resources != self.final_candidate.resources
        ):
            raise ValueError(
                "final selection evaluation binding differs"
            )
        if self.guard_evaluation is not None:
            guard = self.guard_evaluation
            if (
                guard.protocol_sha256 != self.protocol_sha256
                or guard.development_role != "calibration_a_guard"
                or guard.candidate_artifact_sha256
                != self.final_candidate.artifact_sha256
                or guard.candidate_receipt_sha256 != final_receipt
                or guard.resources != self.final_candidate.resources
            ):
                raise ValueError("final guard evaluation binding differs")
        guard_terminal = self.status in (
            "ready_for_candidate_binding",
            "rejected_by_guard",
        )
        if (
            guard_terminal
            != (self.guard_evaluation is not None)
            or guard_terminal
            != (self.frozen_challenger is not None)
        ):
            raise ValueError(
                "challenger or guard presence differs from terminal status"
            )
        if self.frozen_challenger is not None:
            if (
                self.frozen_challenger.protocol_sha256
                != self.protocol_sha256
                or self.frozen_challenger.candidate.receipt_sha256
                != final_receipt
                or self.frozen_challenger.selection_evaluation.receipt_sha256
                != self.final_selection_evaluation.receipt_sha256
                or self.guard_evaluation is None
                or self.guard_evaluation.challenger_receipt_sha256
                != self.frozen_challenger.receipt_sha256
            ):
                raise ValueError(
                    "frozen challenger or guard binding differs"
                )

        proposal_receipts = tuple(
            proposal.receipt_sha256
            for proposal in self.proposal_archive
        )
        residual_map_receipts = tuple(
            residual_map.receipt_sha256
            for residual_map in self.residual_map_archive
        )
        candidate_receipts = tuple(
            candidate.receipt_sha256
            for candidate in self.candidate_archive
        )
        evaluation_receipts = tuple(
            evaluation.receipt_sha256
            for evaluation in self.selection_evaluation_archive
        )
        for label, values in (
            ("residual map archive", residual_map_receipts),
            ("proposal archive", proposal_receipts),
            ("candidate archive", candidate_receipts),
            ("evaluation archive", evaluation_receipts),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} contains duplicate receipts")
        if (
            candidate_receipts[0]
            != self.seed_candidate_receipt_sha256
            or evaluation_receipts[0]
            != self.seed_selection_evaluation_receipt_sha256
        ):
            raise ValueError("archive does not begin at the frozen seed")
        for candidate, evaluation in zip(
            self.candidate_archive,
            self.selection_evaluation_archive,
            strict=True,
        ):
            if (
                evaluation.protocol_sha256 != self.protocol_sha256
                or evaluation.development_role
                != "calibration_a_selection"
                or evaluation.candidate_artifact_sha256
                != candidate.artifact_sha256
                or evaluation.candidate_receipt_sha256
                != candidate.receipt_sha256
                or evaluation.resources != candidate.resources
            ):
                raise ValueError(
                    "selection archive candidate binding differs"
                )
        proposal_receipt_set = set(proposal_receipts)
        evaluation_receipt_set = set(evaluation_receipts)
        for receipt in self.iterations:
            if not set(
                receipt.proposal_receipt_sha256s
            ).issubset(proposal_receipt_set):
                raise ValueError(
                    "iteration references a proposal outside the archive"
                )
            if not set(
                receipt.evaluation_receipt_sha256s
            ).issubset(evaluation_receipt_set):
                raise ValueError(
                    "iteration references an evaluation outside the archive"
                )

        active_head = self.seed_candidate_receipt_sha256
        accepted_count = 0
        for index, receipt in enumerate(self.iterations):
            if receipt.parent_candidate_receipt_sha256 != active_head:
                raise ValueError(
                    "iteration does not bind the active candidate head"
                )
            if receipt.iteration != accepted_count + 1:
                raise ValueError(
                    "iteration receipt sequence is not contiguous"
                )
            accepted = receipt.accepted_candidate_receipt_sha256
            if accepted is None:
                if index != len(self.iterations) - 1:
                    raise ValueError(
                        "a rejected iteration must terminate the loop"
                    )
            else:
                active_head = accepted
                accepted_count += 1
        if (
            active_head != final_receipt
            or self.final_candidate.iteration != accepted_count
        ):
            raise ValueError(
                "final candidate does not match the accepted receipt chain"
            )

    @property
    def transcript_sha256(self) -> str:
        return _payload_sha256("transcript", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PROGRESSIVE_COMPILATION_SCHEMA,
            "format_version": PROGRESSIVE_COMPILATION_FORMAT_VERSION,
            "protocol_sha256": self.protocol_sha256,
            "seed_candidate_receipt_sha256": (
                self.seed_candidate_receipt_sha256
            ),
            "seed_selection_evaluation_receipt_sha256": (
                self.seed_selection_evaluation_receipt_sha256
            ),
            "final_candidate": self.final_candidate.to_dict(),
            "final_selection_evaluation": (
                self.final_selection_evaluation.to_dict()
            ),
            "frozen_challenger": (
                None
                if self.frozen_challenger is None
                else self.frozen_challenger.to_dict()
            ),
            "guard_evaluation": (
                None
                if self.guard_evaluation is None
                else self.guard_evaluation.to_dict()
            ),
            "iterations": tuple(
                item.to_dict() for item in self.iterations
            ),
            "archive": {
                "residual_maps": tuple(
                    item.to_dict() for item in self.residual_map_archive
                ),
                "proposals": tuple(
                    item.to_dict() for item in self.proposal_archive
                ),
                "candidates": tuple(
                    item.to_dict() for item in self.candidate_archive
                ),
                "selection_evaluations": tuple(
                    item.to_dict()
                    for item in self.selection_evaluation_archive
                ),
                "raw_and_dominated_points_retained": True,
            },
            "status": self.status,
            "assessment_manifest_opened": False,
            "guard_opened_once": self.guard_evaluation is not None,
        }

    def validate_against(
        self,
        protocol: ProgressiveCompilationProtocol,
    ) -> None:
        """Recompute the complete active-head and selection decision chain."""

        if not isinstance(protocol, ProgressiveCompilationProtocol):
            raise TypeError(
                "protocol must be ProgressiveCompilationProtocol"
            )
        protocol.validate_integrity()
        if self.protocol_sha256 != protocol.artifact_sha256:
            raise ValueError("result protocol binding differs")

        seed = self.candidate_archive[0]
        seed_evaluation = self.selection_evaluation_archive[0]
        if (
            seed.mutation_kind != "seed"
            or seed.artifact_sha256
            != protocol.seed_candidate_artifact_sha256
            or seed.execution_sha256
            != protocol.seed_candidate_execution_sha256
            or seed.runtime_binding_sha256
            != protocol.seed_runtime_binding_sha256
            or seed.resources.receipt_sha256
            != protocol.seed_resource_receipt_sha256
            or seed.receipt_sha256
            != self.seed_candidate_receipt_sha256
        ):
            raise ValueError("result seed differs from the frozen protocol")
        _validate_evaluation(
            protocol=protocol,
            candidate=seed,
            evaluation=seed_evaluation,
            expected_role="calibration_a_selection",
        )

        maps_by_receipt = {
            item.receipt_sha256: item
            for item in self.residual_map_archive
        }
        proposals_by_receipt = {
            item.receipt_sha256: item
            for item in self.proposal_archive
        }
        candidates_by_proposal: dict[
            str,
            list[ProgressiveCandidate],
        ] = {}
        for candidate in self.candidate_archive[1:]:
            if candidate.proposal_sha256 is None:
                raise ValueError(
                    "nonseed archive candidate lacks a proposal"
                )
            candidates_by_proposal.setdefault(
                candidate.proposal_sha256,
                [],
            ).append(candidate)
        evaluations_by_candidate: dict[
            str,
            list[CandidateEvaluation],
        ] = {}
        for evaluation in self.selection_evaluation_archive[1:]:
            evaluations_by_candidate.setdefault(
                evaluation.candidate_receipt_sha256,
                [],
            ).append(evaluation)

        if len(
            {
                candidate.candidate_id
                for candidate in self.candidate_archive
            }
        ) != len(self.candidate_archive):
            raise ValueError("candidate archive IDs must be unique")
        if len(
            {
                candidate.artifact_sha256
                for candidate in self.candidate_archive
            }
        ) != len(self.candidate_archive):
            raise ValueError("candidate archive artifacts must be unique")

        used_maps: set[str] = set()
        used_proposals: set[str] = set()
        used_candidates = {seed.receipt_sha256}
        used_evaluations = {seed_evaluation.receipt_sha256}
        active_candidate = seed
        active_evaluation = seed_evaluation
        accepted_count = 0
        replayed_terminal_status: ProgressiveStatus | None = None

        if len(self.iterations) > protocol.max_iterations:
            raise ValueError(
                "iteration transcript exceeds the protocol maximum"
            )

        for receipt_index, receipt in enumerate(self.iterations):
            fidelity_before = protocol.fidelity_targets.passes(
                active_evaluation.fidelity
            )
            resources_before = protocol.resource_budget.allows(
                active_candidate.resources
            )
            if (
                fidelity_before
                and resources_before
                and not protocol.compact_after_fidelity
            ):
                raise ValueError(
                    "iteration appears after the loop should have "
                    "terminated"
                )
            expected_phase: ProgressivePhase = (
                "compact" if fidelity_before else "repair"
            )
            if receipt.phase != expected_phase:
                raise ValueError(
                    "iteration phase differs from the active-head "
                    "fidelity state"
                )

            residual_map = maps_by_receipt.get(
                receipt.residual_map_receipt_sha256
            )
            if residual_map is None:
                raise ValueError(
                    "iteration residual map is absent from the archive"
                )
            _validate_residual_map(
                protocol=protocol,
                candidate=active_candidate,
                residual_map=residual_map,
            )
            if residual_map.receipt_sha256 in used_maps:
                raise ValueError("residual map is replayed")
            used_maps.add(residual_map.receipt_sha256)

            proposals = tuple(
                proposal
                for proposal in self.proposal_archive
                if proposal.residual_map_sha256
                == residual_map.receipt_sha256
            )
            if tuple(
                proposal.receipt_sha256 for proposal in proposals
            ) != receipt.proposal_receipt_sha256s:
                raise ValueError(
                    "iteration proposal archive membership differs"
                )
            if len(proposals) > protocol.max_proposals_per_iteration:
                raise ValueError(
                    "archived proposal count exceeds the protocol"
                )

            evaluated: list[
                tuple[
                    MutationProposal,
                    ProgressiveCandidate,
                    CandidateEvaluation,
                ]
            ] = []
            expected_evaluation_receipts: list[str] = []
            budget_blocked = False
            for proposal in proposals:
                _validate_proposal(
                    candidate=active_candidate,
                    residual_map=residual_map,
                    phase=receipt.phase,
                    proposal=proposal,
                )
                proposal_sha = proposal.receipt_sha256
                if proposal_sha in used_proposals:
                    raise ValueError("mutation proposal is replayed")
                used_proposals.add(proposal_sha)
                children = candidates_by_proposal.get(proposal_sha, [])
                if not protocol.resource_budget.allows(
                    proposal.resources
                ):
                    budget_blocked = True
                    if children:
                        raise ValueError(
                            "over-budget proposal has a built candidate"
                        )
                    continue
                if len(children) != 1:
                    raise ValueError(
                        "budget-eligible proposal must have one candidate"
                    )
                child = children[0]
                _validate_built_candidate(
                    parent=active_candidate,
                    proposal=proposal,
                    candidate=child,
                )
                child_receipt = child.receipt_sha256
                child_evaluations = evaluations_by_candidate.get(
                    child_receipt,
                    [],
                )
                if len(child_evaluations) != 1:
                    raise ValueError(
                        "built candidate must have one selection evaluation"
                    )
                evaluation = child_evaluations[0]
                _validate_evaluation(
                    protocol=protocol,
                    candidate=child,
                    evaluation=evaluation,
                    expected_role="calibration_a_selection",
                )
                used_candidates.add(child_receipt)
                used_evaluations.add(evaluation.receipt_sha256)
                expected_evaluation_receipts.append(
                    evaluation.receipt_sha256
                )
                evaluated.append((proposal, child, evaluation))

            if (
                tuple(expected_evaluation_receipts)
                != receipt.evaluation_receipt_sha256s
            ):
                raise ValueError(
                    "iteration evaluation archive membership differs"
                )
            selected = _select_evaluation(
                protocol=protocol,
                phase=receipt.phase,
                parent=active_evaluation,
                candidates=evaluated,
            )
            if selected is None:
                if (
                    receipt.accepted_candidate_receipt_sha256 is not None
                    or receipt.accepted_evaluation_receipt_sha256
                    is not None
                ):
                    raise ValueError(
                        "iteration claims an ineligible accepted child"
                    )
                expected_decision = (
                    "no_budget_eligible_candidate"
                    if budget_blocked and not evaluated
                    else "no_quality_eligible_candidate"
                )
                if receipt.decision != expected_decision:
                    raise ValueError("iteration rejection decision differs")
                if receipt_index != len(self.iterations) - 1:
                    raise ValueError(
                        "rejected iteration is not terminal"
                    )
                if fidelity_before and resources_before:
                    replayed_terminal_status = (
                        "ready_for_candidate_binding"
                    )
                elif budget_blocked and not evaluated:
                    replayed_terminal_status = "stalled_budget"
                else:
                    replayed_terminal_status = "stalled_fidelity"
            else:
                _, child, evaluation = selected
                if (
                    receipt.accepted_candidate_receipt_sha256
                    != child.receipt_sha256
                    or receipt.accepted_evaluation_receipt_sha256
                    != evaluation.receipt_sha256
                ):
                    raise ValueError(
                        "accepted child differs from deterministic selection"
                    )
                expected_decision = (
                    "accepted_fidelity_repair"
                    if receipt.phase == "repair"
                    else "accepted_pareto_compaction"
                )
                if receipt.decision != expected_decision:
                    raise ValueError("iteration acceptance decision differs")
                active_candidate = child
                active_evaluation = evaluation
                accepted_count += 1

        if set(maps_by_receipt) != used_maps:
            raise ValueError("residual map archive contains orphan entries")
        if set(proposals_by_receipt) != used_proposals:
            raise ValueError("proposal archive contains orphan entries")
        if {
            item.receipt_sha256 for item in self.candidate_archive
        } != used_candidates:
            raise ValueError("candidate archive contains orphan entries")
        if {
            item.receipt_sha256
            for item in self.selection_evaluation_archive
        } != used_evaluations:
            raise ValueError("evaluation archive contains orphan entries")
        if (
            active_candidate.receipt_sha256
            != self.final_candidate.receipt_sha256
            or active_evaluation.receipt_sha256
            != self.final_selection_evaluation.receipt_sha256
            or active_candidate.iteration != accepted_count
        ):
            raise ValueError(
                "final result differs from the recomputed active head"
            )

        selection_passed = protocol.fidelity_targets.passes(
            active_evaluation.fidelity
        )
        resource_passed = protocol.resource_budget.allows(
            active_candidate.resources
        )
        if replayed_terminal_status is None:
            if len(self.iterations) == protocol.max_iterations:
                replayed_terminal_status = (
                    "ready_for_candidate_binding"
                    if selection_passed and resource_passed
                    else "max_iterations"
                )
            elif (
                selection_passed
                and resource_passed
                and not protocol.compact_after_fidelity
            ):
                replayed_terminal_status = (
                    "ready_for_candidate_binding"
                )
            else:
                raise ValueError(
                    "iteration transcript terminates before the loop "
                    "policy permits"
                )

        if self.guard_evaluation is not None:
            if (
                replayed_terminal_status
                != "ready_for_candidate_binding"
            ):
                raise ValueError(
                    "guard was opened before legal loop termination"
                )
            expected_challenger = FrozenCalibrationAChallenger(
                protocol_sha256=protocol.artifact_sha256,
                seed_candidate_receipt_sha256=(
                    self.seed_candidate_receipt_sha256
                ),
                seed_selection_evaluation_receipt_sha256=(
                    self.seed_selection_evaluation_receipt_sha256
                ),
                candidate=active_candidate,
                selection_evaluation=active_evaluation,
                iteration_receipt_sha256s=tuple(
                    receipt.receipt_sha256
                    for receipt in self.iterations
                ),
                residual_map_receipt_sha256s=tuple(
                    residual_map.receipt_sha256
                    for residual_map in self.residual_map_archive
                ),
                proposal_receipt_sha256s=tuple(
                    proposal.receipt_sha256
                    for proposal in self.proposal_archive
                ),
                candidate_archive_receipt_sha256s=tuple(
                    candidate.receipt_sha256
                    for candidate in self.candidate_archive
                ),
                selection_archive_receipt_sha256s=tuple(
                    evaluation.receipt_sha256
                    for evaluation
                    in self.selection_evaluation_archive
                ),
            )
            if (
                self.frozen_challenger is None
                or self.frozen_challenger.receipt_sha256
                != expected_challenger.receipt_sha256
            ):
                raise ValueError(
                    "guard did not bind the recomputed frozen challenger"
                )
            _validate_evaluation(
                protocol=protocol,
                candidate=active_candidate,
                evaluation=self.guard_evaluation,
                expected_role="calibration_a_guard",
                expected_challenger_receipt_sha256=(
                    expected_challenger.receipt_sha256
                ),
            )
            expected_status = (
                "ready_for_candidate_binding"
                if protocol.fidelity_targets.passes(
                    self.guard_evaluation.fidelity
                )
                else "rejected_by_guard"
            )
            if self.status != expected_status:
                raise ValueError("guard outcome and result status differ")
        else:
            if (
                replayed_terminal_status
                == "ready_for_candidate_binding"
            ):
                raise ValueError(
                    "legally ready result lacks its terminal guard"
                )
            if self.status != replayed_terminal_status:
                raise ValueError(
                    "unguarded result status differs from loop replay"
                )


@dataclass(frozen=True, slots=True)
class FrozenProgressiveCandidateHandoff:
    """Development-only freeze offered to a separate one-shot binder."""

    protocol_sha256: str
    transcript_sha256: str
    candidate_id: str
    candidate_artifact_sha256: str
    candidate_execution_sha256: str
    candidate_runtime_binding_sha256: str
    candidate_receipt_sha256: str
    calibration_a_challenger_receipt_sha256: str
    selection_evaluation_receipt_sha256: str
    guard_evaluation_receipt_sha256: str
    resources: ProgressiveResourceFootprint
    fidelity: ProgressiveFidelity
    assessment_accessed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "transcript_sha256",
            "candidate_artifact_sha256",
            "candidate_execution_sha256",
            "candidate_runtime_binding_sha256",
            "candidate_receipt_sha256",
            "calibration_a_challenger_receipt_sha256",
            "selection_evaluation_receipt_sha256",
            "guard_evaluation_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _identifier(self.candidate_id, label="candidate_id")
        if not isinstance(self.resources, ProgressiveResourceFootprint):
            raise TypeError(
                "resources must be ProgressiveResourceFootprint"
            )
        if (
            self.resources.candidate_execution_sha256
            != self.candidate_execution_sha256
        ):
            raise ValueError(
                "handoff resources do not bind candidate execution"
            )
        if not isinstance(self.fidelity, ProgressiveFidelity):
            raise TypeError("fidelity must be ProgressiveFidelity")
        if self.assessment_accessed is not False:
            raise ValueError(
                "progressive handoff cannot claim assessment access"
            )

    @property
    def receipt_sha256(self) -> str:
        return _payload_sha256("frozen-handoff", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "transcript_sha256": self.transcript_sha256,
            "candidate_id": self.candidate_id,
            "candidate_artifact_sha256": (
                self.candidate_artifact_sha256
            ),
            "candidate_execution_sha256": (
                self.candidate_execution_sha256
            ),
            "candidate_runtime_binding_sha256": (
                self.candidate_runtime_binding_sha256
            ),
            "candidate_receipt_sha256": (
                self.candidate_receipt_sha256
            ),
            "calibration_a_challenger_receipt_sha256": (
                self.calibration_a_challenger_receipt_sha256
            ),
            "selection_evaluation_receipt_sha256": (
                self.selection_evaluation_receipt_sha256
            ),
            "guard_evaluation_receipt_sha256": (
                self.guard_evaluation_receipt_sha256
            ),
            "resources": self.resources.to_dict(),
            "fidelity": self.fidelity.to_dict(),
            "assessment_accessed": False,
            "next_authority": (
                "separate_model_specific_one_shot_protocol_binder"
            ),
        }


def _validate_evaluation(
    *,
    protocol: ProgressiveCompilationProtocol,
    candidate: ProgressiveCandidate,
    evaluation: CandidateEvaluation,
    expected_role: Literal[
        "calibration_a_selection",
        "calibration_a_guard",
    ],
    expected_challenger_receipt_sha256: str | None = None,
) -> None:
    if not isinstance(evaluation, CandidateEvaluation):
        raise TypeError("evaluator must return CandidateEvaluation")
    if evaluation.protocol_sha256 != protocol.artifact_sha256:
        raise ValueError("evaluation protocol binding differs")
    if evaluation.development_role != expected_role:
        raise ValueError("evaluation development role differs")
    if expected_role == "calibration_a_selection":
        if (
            expected_challenger_receipt_sha256 is not None
            or evaluation.challenger_receipt_sha256 is not None
        ):
            raise ValueError(
                "selection evaluation cannot bind a challenger"
            )
    else:
        expected_challenger = _require_sha256(
            expected_challenger_receipt_sha256,
            label="expected_challenger_receipt_sha256",
        )
        if evaluation.challenger_receipt_sha256 != expected_challenger:
            raise ValueError("guard challenger binding differs")
    expected_manifest = (
        protocol.corpus.selection_manifest_sha256
        if expected_role == "calibration_a_selection"
        else protocol.corpus.guard_manifest_sha256
    )
    expected_example_count = (
        protocol.corpus.selection_example_count
        if expected_role == "calibration_a_selection"
        else protocol.corpus.guard_example_count
    )
    expected_family_ids = (
        protocol.corpus.selection_family_ids
        if expected_role == "calibration_a_selection"
        else protocol.corpus.guard_family_ids
    )
    if evaluation.manifest_sha256 != expected_manifest:
        raise ValueError(
            "evaluation did not use the frozen role manifest"
        )
    coverage = evaluation.coverage
    if (
        not coverage.complete
        or coverage.manifest_sha256 != expected_manifest
        or coverage.expected_example_count != expected_example_count
        or coverage.observed_example_count != expected_example_count
        or coverage.expected_family_ids != expected_family_ids
        or coverage.observed_family_ids != expected_family_ids
    ):
        raise ValueError(
            "evaluation does not cover the complete frozen role"
        )
    if (
        evaluation.manifest_sha256
        in protocol.forbidden_assessment_manifest_sha256s
    ):
        raise ValueError("evaluation attempted to use assessment evidence")
    if (
        evaluation.candidate_artifact_sha256
        != candidate.artifact_sha256
        or evaluation.candidate_receipt_sha256
        != candidate.receipt_sha256
    ):
        raise ValueError("evaluation candidate binding differs")
    if evaluation.resources != candidate.resources:
        raise ValueError(
            "evaluation resource accounting differs from the candidate"
        )


def _validate_residual_map(
    *,
    protocol: ProgressiveCompilationProtocol,
    candidate: ProgressiveCandidate,
    residual_map: ResidualMap,
) -> None:
    if not isinstance(residual_map, ResidualMap):
        raise TypeError("mapper must return ResidualMap")
    if residual_map.protocol_sha256 != protocol.artifact_sha256:
        raise ValueError("residual map protocol binding differs")
    if (
        residual_map.fit_manifest_sha256
        != protocol.corpus.fit_manifest_sha256
    ):
        raise ValueError("residual map did not use the frozen fit manifest")
    if (
        residual_map.fit_manifest_sha256
        in protocol.forbidden_assessment_manifest_sha256s
    ):
        raise ValueError("residual map attempted to use assessment evidence")
    if (
        residual_map.candidate_artifact_sha256
        != candidate.artifact_sha256
        or residual_map.candidate_receipt_sha256
        != candidate.receipt_sha256
        or residual_map.iteration != candidate.iteration
    ):
        raise ValueError("residual map candidate binding differs")


def _validate_proposal(
    *,
    candidate: ProgressiveCandidate,
    residual_map: ResidualMap,
    phase: ProgressivePhase,
    proposal: MutationProposal,
) -> None:
    if not isinstance(proposal, MutationProposal):
        raise TypeError("proposer must return MutationProposal objects")
    if proposal.phase != phase:
        raise ValueError("proposal phase differs from the active phase")
    if (
        proposal.parent_artifact_sha256
        != candidate.artifact_sha256
        or proposal.parent_receipt_sha256
        != candidate.receipt_sha256
    ):
        raise ValueError("proposal parent binding differs")
    if proposal.residual_map_sha256 != residual_map.receipt_sha256:
        raise ValueError("proposal residual-map binding differs")
    if any(rank >= len(residual_map.targets) for rank in proposal.target_ranks):
        raise ValueError("proposal references an unknown residual target")


def _validate_built_candidate(
    *,
    parent: ProgressiveCandidate,
    proposal: MutationProposal,
    candidate: ProgressiveCandidate,
) -> None:
    if not isinstance(candidate, ProgressiveCandidate):
        raise TypeError("builder must return ProgressiveCandidate")
    if candidate.iteration != parent.iteration + 1:
        raise ValueError("built candidate iteration is not contiguous")
    if candidate.parent_artifact_sha256 != parent.artifact_sha256:
        raise ValueError("built candidate parent binding differs")
    if candidate.proposal_sha256 != proposal.receipt_sha256:
        raise ValueError("built candidate proposal binding differs")
    if candidate.mutation_kind != proposal.mutation_kind:
        raise ValueError("built candidate mutation kind differs")
    if candidate.resources != proposal.resources:
        raise ValueError(
            "built candidate resources differ from the proposal"
        )
    if candidate.artifact_sha256 == parent.artifact_sha256:
        raise ValueError("mutation must produce a new candidate artifact")


def _repair_is_eligible(
    *,
    protocol: ProgressiveCompilationProtocol,
    parent: CandidateEvaluation,
    child: CandidateEvaluation,
) -> bool:
    targets = protocol.fidelity_targets
    parent_ratios = targets.normalized_ratios(parent.fidelity)
    child_ratios = targets.normalized_ratios(child.fidelity)
    parent_burden = max(parent_ratios.values())
    child_burden = max(child_ratios.values())
    required = parent_burden * (
        1.0 - protocol.minimum_repair_relative_burden_reduction
    )
    if child_burden > required:
        return False
    regression = protocol.maximum_repair_axis_regression_fraction
    for name, child_ratio in child_ratios.items():
        parent_ratio = parent_ratios[name]
        allowed = parent_ratio * (1.0 + regression)
        if child_ratio > allowed:
            return False
    return True


def _compact_is_eligible(
    *,
    protocol: ProgressiveCompilationProtocol,
    parent: CandidateEvaluation,
    child: CandidateEvaluation,
) -> bool:
    if not protocol.fidelity_targets.passes(child.fidelity):
        return False
    before = parent.resources
    after = child.resources
    pairs = (
        (
            after.total_learned_parameters,
            before.total_learned_parameters,
        ),
        (
            after.total_runtime_parameter_bytes,
            before.total_runtime_parameter_bytes,
        ),
        (
            after.total_logical_macs_per_token,
            before.total_logical_macs_per_token,
        ),
    )
    return all(
        child_value <= parent_value
        for child_value, parent_value in pairs
    ) and any(
        child_value < parent_value
        for child_value, parent_value in pairs
    )


def _select_evaluation(
    *,
    protocol: ProgressiveCompilationProtocol,
    phase: ProgressivePhase,
    parent: CandidateEvaluation,
    candidates: Sequence[
        tuple[MutationProposal, ProgressiveCandidate, CandidateEvaluation]
    ],
) -> tuple[
    MutationProposal,
    ProgressiveCandidate,
    CandidateEvaluation,
] | None:
    eligible = []
    for item in candidates:
        _, _, evaluation = item
        accepted = (
            _repair_is_eligible(
                protocol=protocol,
                parent=parent,
                child=evaluation,
            )
            if phase == "repair"
            else _compact_is_eligible(
                protocol=protocol,
                parent=parent,
                child=evaluation,
            )
        )
        if accepted:
            eligible.append(item)
    if not eligible:
        return None
    targets = protocol.fidelity_targets
    if phase == "repair":
        key: Callable[
            [
                tuple[
                    MutationProposal,
                    ProgressiveCandidate,
                    CandidateEvaluation,
                ]
            ],
            object,
        ] = lambda item: (
            targets.burden(item[2].fidelity),
            item[2].resources.total_logical_macs_per_token,
            item[2].resources.total_learned_parameters,
            item[2].resources.total_runtime_parameter_bytes,
            item[1].candidate_id,
            item[1].artifact_sha256,
            item[1].receipt_sha256,
        )
    else:
        key = lambda item: (
            item[2].resources.total_logical_macs_per_token,
            item[2].resources.total_learned_parameters,
            item[2].resources.total_runtime_parameter_bytes,
            targets.burden(item[2].fidelity),
            item[1].candidate_id,
            item[1].artifact_sha256,
            item[1].receipt_sha256,
        )
    return min(eligible, key=key)


def run_progressive_compilation(
    *,
    protocol: ProgressiveCompilationProtocol,
    seed_candidate: ProgressiveCandidate,
    seed_selection_evaluation: CandidateEvaluation,
    map_residual: ResidualMapper,
    propose_mutations: MutationProposer,
    build_candidate: CandidateBuilder,
    evaluate_selection: SelectionEvaluator,
    evaluate_guard: GuardEvaluator,
) -> ProgressiveCompilationResult:
    """Run repeated fit/selection loops and one frozen final-guard veto."""

    if not isinstance(protocol, ProgressiveCompilationProtocol):
        raise TypeError(
            "protocol must be ProgressiveCompilationProtocol"
        )
    protocol.validate_integrity()
    if not isinstance(seed_candidate, ProgressiveCandidate):
        raise TypeError("seed_candidate must be ProgressiveCandidate")
    if seed_candidate.mutation_kind != "seed":
        raise ValueError("seed_candidate must have mutation_kind='seed'")
    if (
        seed_candidate.artifact_sha256
        != protocol.seed_candidate_artifact_sha256
        or seed_candidate.execution_sha256
        != protocol.seed_candidate_execution_sha256
        or seed_candidate.runtime_binding_sha256
        != protocol.seed_runtime_binding_sha256
        or seed_candidate.resources.receipt_sha256
        != protocol.seed_resource_receipt_sha256
    ):
        raise ValueError("seed candidate differs from the frozen protocol")
    _validate_evaluation(
        protocol=protocol,
        candidate=seed_candidate,
        evaluation=seed_selection_evaluation,
        expected_role="calibration_a_selection",
    )
    for label, callback in (
        ("map_residual", map_residual),
        ("propose_mutations", propose_mutations),
        ("build_candidate", build_candidate),
        ("evaluate_selection", evaluate_selection),
        ("evaluate_guard", evaluate_guard),
    ):
        if not callable(callback):
            raise TypeError(f"{label} must be callable")

    current_candidate = seed_candidate
    current_evaluation = seed_selection_evaluation
    receipts: list[ProgressiveIterationReceipt] = []
    residual_map_archive: list[ResidualMap] = []
    proposal_archive: list[MutationProposal] = []
    candidate_archive = [seed_candidate]
    selection_archive = [seed_selection_evaluation]
    budget_blocked = False
    status: ProgressiveStatus = "max_iterations"

    for _ in range(protocol.max_iterations):
        fidelity_passed = protocol.fidelity_targets.passes(
            current_evaluation.fidelity
        )
        resource_passed = protocol.resource_budget.allows(
            current_candidate.resources
        )
        if (
            fidelity_passed
            and resource_passed
            and not protocol.compact_after_fidelity
        ):
            status = "ready_for_candidate_binding"
            break

        phase: ProgressivePhase = (
            "compact" if fidelity_passed else "repair"
        )
        residual_map = map_residual(
            current_candidate,
            protocol.fit_view(),
        )
        _validate_residual_map(
            protocol=protocol,
            candidate=current_candidate,
            residual_map=residual_map,
        )
        residual_map_archive.append(residual_map)
        raw_proposals = propose_mutations(
            current_candidate,
            residual_map,
            phase,
        )
        if isinstance(raw_proposals, (str, bytes)) or not isinstance(
            raw_proposals,
            Sequence,
        ):
            raise TypeError("proposer must return a proposal sequence")
        proposals = tuple(raw_proposals)
        if len(proposals) > protocol.max_proposals_per_iteration:
            raise ValueError("proposal count exceeds the protocol maximum")
        for proposal in proposals:
            _validate_proposal(
                candidate=current_candidate,
                residual_map=residual_map,
                phase=phase,
                proposal=proposal,
            )
        proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
        proposal_receipts = tuple(
            proposal.receipt_sha256 for proposal in proposals
        )
        if (
            len(set(proposal_ids)) != len(proposal_ids)
            or len(set(proposal_receipts)) != len(proposal_receipts)
        ):
            raise ValueError("proposals must have unique identities")
        proposal_archive.extend(proposals)

        evaluated: list[
            tuple[
                MutationProposal,
                ProgressiveCandidate,
                CandidateEvaluation,
            ]
        ] = []
        iteration_budget_blocked = False
        for proposal in proposals:
            if not protocol.resource_budget.allows(proposal.resources):
                iteration_budget_blocked = True
                continue
            child = build_candidate(current_candidate, proposal)
            _validate_built_candidate(
                parent=current_candidate,
                proposal=proposal,
                candidate=child,
            )
            if any(
                archived.candidate_id == child.candidate_id
                or archived.artifact_sha256 == child.artifact_sha256
                for archived in candidate_archive
            ):
                raise ValueError(
                    "built candidate identity must be globally unique"
                )
            evaluation = evaluate_selection(
                child,
                protocol.selection_view(),
            )
            _validate_evaluation(
                protocol=protocol,
                candidate=child,
                evaluation=evaluation,
                expected_role="calibration_a_selection",
            )
            candidate_archive.append(child)
            selection_archive.append(evaluation)
            evaluated.append((proposal, child, evaluation))

        selected = _select_evaluation(
            protocol=protocol,
            phase=phase,
            parent=current_evaluation,
            candidates=evaluated,
        )
        if selected is None:
            budget_blocked = (
                iteration_budget_blocked and not evaluated
            )
            receipts.append(
                ProgressiveIterationReceipt(
                    iteration=current_candidate.iteration + 1,
                    phase=phase,
                    parent_candidate_receipt_sha256=(
                        current_candidate.receipt_sha256
                    ),
                    residual_map_receipt_sha256=(
                        residual_map.receipt_sha256
                    ),
                    proposal_receipt_sha256s=proposal_receipts,
                    evaluation_receipt_sha256s=tuple(
                        item[2].receipt_sha256 for item in evaluated
                    ),
                    accepted_candidate_receipt_sha256=None,
                    accepted_evaluation_receipt_sha256=None,
                    decision=(
                        "no_budget_eligible_candidate"
                        if budget_blocked
                        else "no_quality_eligible_candidate"
                    ),
                )
            )
            if fidelity_passed and resource_passed:
                status = "ready_for_candidate_binding"
            elif budget_blocked:
                status = "stalled_budget"
            else:
                status = "stalled_fidelity"
            break

        _, accepted_candidate, accepted_evaluation = selected
        receipts.append(
            ProgressiveIterationReceipt(
                iteration=accepted_candidate.iteration,
                phase=phase,
                parent_candidate_receipt_sha256=(
                    current_candidate.receipt_sha256
                ),
                residual_map_receipt_sha256=residual_map.receipt_sha256,
                proposal_receipt_sha256s=proposal_receipts,
                evaluation_receipt_sha256s=tuple(
                    item[2].receipt_sha256 for item in evaluated
                ),
                accepted_candidate_receipt_sha256=(
                    accepted_candidate.receipt_sha256
                ),
                accepted_evaluation_receipt_sha256=(
                    accepted_evaluation.receipt_sha256
                ),
                decision=(
                    "accepted_fidelity_repair"
                    if phase == "repair"
                    else "accepted_pareto_compaction"
                ),
            )
        )
        current_candidate = accepted_candidate
        current_evaluation = accepted_evaluation
    else:
        if (
            protocol.fidelity_targets.passes(
                current_evaluation.fidelity
            )
            and protocol.resource_budget.allows(
                current_candidate.resources
            )
        ):
            status = "ready_for_candidate_binding"
        else:
            status = "max_iterations"

    frozen_challenger: FrozenCalibrationAChallenger | None = None
    guard_evaluation: CandidateEvaluation | None = None
    if status == "ready_for_candidate_binding":
        frozen_challenger = FrozenCalibrationAChallenger(
            protocol_sha256=protocol.artifact_sha256,
            seed_candidate_receipt_sha256=(
                seed_candidate.receipt_sha256
            ),
            seed_selection_evaluation_receipt_sha256=(
                seed_selection_evaluation.receipt_sha256
            ),
            candidate=current_candidate,
            selection_evaluation=current_evaluation,
            iteration_receipt_sha256s=tuple(
                receipt.receipt_sha256 for receipt in receipts
            ),
            residual_map_receipt_sha256s=tuple(
                residual_map.receipt_sha256
                for residual_map in residual_map_archive
            ),
            proposal_receipt_sha256s=tuple(
                proposal.receipt_sha256
                for proposal in proposal_archive
            ),
            candidate_archive_receipt_sha256s=tuple(
                candidate.receipt_sha256
                for candidate in candidate_archive
            ),
            selection_archive_receipt_sha256s=tuple(
                evaluation.receipt_sha256
                for evaluation in selection_archive
            ),
        )
        guard_evaluation = evaluate_guard(
            frozen_challenger,
            protocol.guard_view(),
        )
        _validate_evaluation(
            protocol=protocol,
            candidate=current_candidate,
            evaluation=guard_evaluation,
            expected_role="calibration_a_guard",
            expected_challenger_receipt_sha256=(
                frozen_challenger.receipt_sha256
            ),
        )
        if not protocol.fidelity_targets.passes(
            guard_evaluation.fidelity
        ):
            status = "rejected_by_guard"

    return ProgressiveCompilationResult(
        protocol_sha256=protocol.artifact_sha256,
        seed_candidate_receipt_sha256=seed_candidate.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=(
            seed_selection_evaluation.receipt_sha256
        ),
        final_candidate=current_candidate,
        final_selection_evaluation=current_evaluation,
        frozen_challenger=frozen_challenger,
        guard_evaluation=guard_evaluation,
        iterations=tuple(receipts),
        residual_map_archive=tuple(residual_map_archive),
        proposal_archive=tuple(proposal_archive),
        candidate_archive=tuple(candidate_archive),
        selection_evaluation_archive=tuple(selection_archive),
        status=status,
    )


def freeze_progressive_candidate(
    *,
    protocol: ProgressiveCompilationProtocol,
    result: ProgressiveCompilationResult,
) -> FrozenProgressiveCandidateHandoff:
    """Freeze a passing development result without touching held-out data."""

    if not isinstance(protocol, ProgressiveCompilationProtocol):
        raise TypeError(
            "protocol must be ProgressiveCompilationProtocol"
        )
    protocol.validate_integrity()
    if not isinstance(result, ProgressiveCompilationResult):
        raise TypeError(
            "result must be ProgressiveCompilationResult"
        )
    if result.protocol_sha256 != protocol.artifact_sha256:
        raise ValueError("result protocol binding differs")
    result.validate_against(protocol)
    if result.status != "ready_for_candidate_binding":
        raise ValueError(
            "only a ready_for_candidate_binding result can be frozen"
        )
    _validate_evaluation(
        protocol=protocol,
        candidate=result.final_candidate,
        evaluation=result.final_selection_evaluation,
        expected_role="calibration_a_selection",
    )
    if result.guard_evaluation is None:
        raise ValueError("ready result lacks its final guard evaluation")
    if result.frozen_challenger is None:
        raise ValueError("ready result lacks its frozen A challenger")
    _validate_evaluation(
        protocol=protocol,
        candidate=result.final_candidate,
        evaluation=result.guard_evaluation,
        expected_role="calibration_a_guard",
        expected_challenger_receipt_sha256=(
            result.frozen_challenger.receipt_sha256
        ),
    )
    if not protocol.fidelity_targets.passes(
        result.final_selection_evaluation.fidelity
    ):
        raise ValueError(
            "final candidate does not pass selection fidelity gates"
        )
    if not protocol.fidelity_targets.passes(
        result.guard_evaluation.fidelity
    ):
        raise ValueError("final candidate does not pass guard fidelity gates")
    if not protocol.resource_budget.allows(
        result.final_candidate.resources
    ):
        raise ValueError("final candidate does not pass resource gates")
    return FrozenProgressiveCandidateHandoff(
        protocol_sha256=protocol.artifact_sha256,
        transcript_sha256=result.transcript_sha256,
        candidate_id=result.final_candidate.candidate_id,
        candidate_artifact_sha256=(
            result.final_candidate.artifact_sha256
        ),
        candidate_execution_sha256=(
            result.final_candidate.execution_sha256
        ),
        candidate_runtime_binding_sha256=(
            result.final_candidate.runtime_binding_sha256
        ),
        candidate_receipt_sha256=(
            result.final_candidate.receipt_sha256
        ),
        calibration_a_challenger_receipt_sha256=(
            result.frozen_challenger.receipt_sha256
        ),
        selection_evaluation_receipt_sha256=(
            result.final_selection_evaluation.receipt_sha256
        ),
        guard_evaluation_receipt_sha256=(
            result.guard_evaluation.receipt_sha256
        ),
        resources=result.final_candidate.resources,
        fidelity=result.guard_evaluation.fidelity,
    )


__all__ = [
    "CandidateBuilder",
    "CandidateEvaluation",
    "DevelopmentCorpus",
    "DevelopmentEvaluationCoverage",
    "FitDevelopmentView",
    "FrozenCalibrationAChallenger",
    "FrozenProgressiveCandidateHandoff",
    "GuardDevelopmentView",
    "GuardEvaluator",
    "MutationKind",
    "MutationProposal",
    "MutationProposer",
    "PROGRESSIVE_COMPILATION_FORMAT_VERSION",
    "PROGRESSIVE_COMPILATION_SCHEMA",
    "ProgressiveBehavioralFidelity",
    "ProgressiveBehavioralTargets",
    "ProgressiveCandidate",
    "ProgressiveCompilationProtocol",
    "ProgressiveCompilationResult",
    "ProgressiveFidelity",
    "ProgressiveFidelityTargets",
    "ProgressiveIterationReceipt",
    "ProgressivePhase",
    "ProgressiveResourceBudget",
    "ProgressiveResourceFootprint",
    "ProgressiveStatus",
    "ResidualMap",
    "ResidualMapper",
    "ResidualTarget",
    "SelectionDevelopmentView",
    "SelectionEvaluator",
    "freeze_progressive_candidate",
    "run_progressive_compilation",
]
