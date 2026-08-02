"""Nested family-LOFO comparison for adaptive generator-innovation banks.

The collector owns temporal windows, target-blind scale calibration, feature
health, and the construction of exact token-loss directional scores.  This
module starts only after those score rows have been reduced to
``TokenLossFisherPromptRecord`` sufficient statistics.

Every declared candidate and conditional ridge is scored on every outer held
family.  Inner whole-family splits select ridges for individual candidates and
jointly select candidates for plan-declared portfolios.  Portfolio selection
may use only candidates admitted by an activation-only eligibility receipt;
the exact static-U control is always available and represented once.

No token is ever an independent train/validation unit.  The serialized report
contains prompt-local sufficient statistics, fitted scalar receipts, aggregate
feature-health hashes, and family-level scores, but no prompt text, token ids,
token score rows, activations, gradients, or logits.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .gemma3_l3_l4_iterative_generator_innovation import (
    GENERATOR_INNOVATION_TANGENT_ORDER,
)
from .gemma3_l3_l4_iterative_generator_innovation_edges import (
    GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER,
)
from .token_loss_fisher import (
    TokenLossFisherPromptRecord,
    _canonical_records,
    _family_moments,
    _mean_moments,
    _relative_improvement,
    _residual_rmse,
    token_loss_fisher_prompt_record_from_dict,
)
from .token_loss_fisher_generator_innovation import (
    GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS,
    _basis_payload,
    _basis_tensor,
    _fit_generator,
    _fit_legacy_shared,
    _mean_and_standard_error,
    _same_number,
)


__all__ = [
    "ADAPTIVE_GENERATOR_INNOVATION_V2_SCHEMA",
    "AdaptiveGeneratorInnovationCandidateSpec",
    "AdaptiveGeneratorInnovationEligibilityReceipt",
    "AdaptiveGeneratorInnovationPortfolioSpec",
    "AdaptiveGeneratorInnovationV2Protocol",
    "adaptive_generator_innovation_fit_candidate_id",
    "build_generator_innovation_adaptive_v2_report",
    "replay_generator_innovation_adaptive_v2_report",
    "validate_generator_innovation_adaptive_v2_report",
]


ADAPTIVE_GENERATOR_INNOVATION_V2_SCHEMA = (
    "fisher_graph.token_loss_fisher_generator_innovation_adaptive_v2.v1"
)
_PROTOCOL_SCHEMA = (
    "fisher_graph.generator_innovation_adaptive_v2_protocol.v1"
)
_ELIGIBILITY_SCHEMA = (
    "fisher_graph.generator_innovation_adaptive_v2_eligibility.v1"
)
_STATIC_RIDGE = "inf"
_PROTOCOL_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-protocol:v1\0"
)
_ELIGIBILITY_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-eligibility:v1\0"
)
_INNER_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-inner:v1\0"
)
_OUTER_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-outer:v1\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-selection:v1\0"
)
_FOLD_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-fold:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:generator-innovation-adaptive-v2-report:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_WIDTH = 4
_NUMBER_TOLERANCE = 1.0e-12


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_bytes(left) == _canonical_bytes(right)


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ordered_identifiers(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (tuple, list)
    ):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(
        _identifier(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if (not allow_empty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique and nonempty")
    return result


def _metadata(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, str | int | float | bool | None], ...]:
    if isinstance(value, Mapping):
        items = tuple(sorted(value.items()))
    elif isinstance(value, (tuple, list)):
        items = tuple(value)
    else:
        raise TypeError(f"{label} must be a mapping or key-value sequence")
    result: list[tuple[str, str | int | float | bool | None]] = []
    for index, item in enumerate(items):
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
        ):
            raise ValueError(f"{label}[{index}] must be a key-value pair")
        key = _identifier(item[0], label=f"{label}[{index}] key")
        raw = item[1]
        if raw is not None and not isinstance(
            raw, (str, int, float, bool)
        ):
            raise TypeError(f"{label}[{index}] value must be a JSON scalar")
        if isinstance(raw, float) and not math.isfinite(raw):
            raise ValueError(f"{label}[{index}] value must be finite")
        result.append((key, raw))
    canonical = tuple(sorted(result))
    if len({key for key, _value in canonical}) != len(canonical):
        raise ValueError(f"{label} keys must be unique")
    return canonical


def adaptive_generator_innovation_fit_candidate_id(
    candidate_id: str,
    ridge_label: str,
) -> str:
    """Return the stable ID for one named feature candidate and ridge."""

    name = _identifier(candidate_id, label="candidate_id")
    ridge = _identifier(ridge_label, label="ridge_label")
    if ridge not in GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS:
        raise ValueError("adaptive conditional ridge label is unsupported")
    encoded = ridge.replace(".", "p")
    return f"{name}__conditional_ridge_{encoded}"


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorInnovationCandidateSpec:
    """One collector-defined feature arm; its math remains outside this file."""

    candidate_id: str
    family: str
    metadata: tuple[
        tuple[str, str | int | float | bool | None], ...
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, label="candidate ID"),
        )
        object.__setattr__(
            self,
            "family",
            _identifier(self.family, label="candidate family"),
        )
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, label="candidate metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> AdaptiveGeneratorInnovationCandidateSpec:
        if set(value) != {"candidate_id", "family", "metadata"}:
            raise ValueError("adaptive candidate-spec fields differ")
        return cls(
            candidate_id=str(value["candidate_id"]),
            family=str(value["family"]),
            metadata=_metadata(
                value["metadata"], label="serialized candidate metadata"
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorInnovationPortfolioSpec:
    """A plan-declared subset with one explicit simplicity order."""

    portfolio_id: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "portfolio_id",
            _identifier(self.portfolio_id, label="portfolio ID"),
        )
        object.__setattr__(
            self,
            "candidate_ids",
            _ordered_identifiers(
                self.candidate_ids,
                label=f"{self.portfolio_id} candidate IDs",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "portfolio_id": self.portfolio_id,
            "candidate_ids": self.candidate_ids,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> AdaptiveGeneratorInnovationPortfolioSpec:
        if set(value) != {"portfolio_id", "candidate_ids"}:
            raise ValueError("adaptive portfolio-spec fields differ")
        return cls(
            portfolio_id=str(value["portfolio_id"]),
            candidate_ids=_ordered_identifiers(
                value["candidate_ids"],
                label="serialized portfolio candidate IDs",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorInnovationV2Protocol:
    """Serializable, injectable comparison and selection policy."""

    candidate_specs: tuple[AdaptiveGeneratorInnovationCandidateSpec, ...]
    candidate_simplicity_order: tuple[str, ...]
    portfolio_specs: tuple[AdaptiveGeneratorInnovationPortfolioSpec, ...]
    static_reference_candidate_id: str
    v1_candidate_id: str
    conditional_ridge_labels: tuple[str, ...] = (
        GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS
    )
    ridge_simplicity_order: tuple[str, ...] = ("10", "1", "0.1")
    static_candidate_id: str = "static_u"
    minimum_family_count: int = 4
    required_prompts_per_family: int | None = None
    protocol_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        specs = tuple(self.candidate_specs)
        if (
            not specs
            or any(
                not isinstance(
                    spec, AdaptiveGeneratorInnovationCandidateSpec
                )
                for spec in specs
            )
        ):
            raise TypeError("adaptive protocol candidate specs differ")
        ids = tuple(spec.candidate_id for spec in specs)
        if len(set(ids)) != len(ids):
            raise ValueError("adaptive protocol candidate IDs must be unique")
        simplicity = _ordered_identifiers(
            self.candidate_simplicity_order,
            label="candidate simplicity order",
        )
        if set(simplicity) != set(ids):
            raise ValueError(
                "candidate simplicity order must cover every candidate once"
            )
        portfolios = tuple(self.portfolio_specs)
        if (
            not portfolios
            or any(
                not isinstance(
                    value, AdaptiveGeneratorInnovationPortfolioSpec
                )
                for value in portfolios
            )
            or len({value.portfolio_id for value in portfolios})
            != len(portfolios)
        ):
            raise ValueError("adaptive portfolio declarations differ")
        position = {name: index for index, name in enumerate(simplicity)}
        for portfolio in portfolios:
            if (
                not set(portfolio.candidate_ids).issubset(ids)
                or tuple(
                    sorted(
                        portfolio.candidate_ids,
                        key=position.__getitem__,
                    )
                )
                != portfolio.candidate_ids
            ):
                raise ValueError(
                    f"{portfolio.portfolio_id} must follow global simplicity"
                )
        reference = _identifier(
            self.static_reference_candidate_id,
            label="static reference candidate ID",
        )
        v1 = _identifier(self.v1_candidate_id, label="v1 candidate ID")
        if reference not in ids or v1 not in ids:
            raise ValueError("adaptive static reference or v1 is undeclared")
        ridges = _ordered_identifiers(
            self.conditional_ridge_labels,
            label="conditional ridge labels",
        )
        if (
            set(ridges)
            - set(GENERATOR_INNOVATION_CONDITIONAL_RIDGE_LABELS)
            or _STATIC_RIDGE not in ridges
        ):
            raise ValueError("adaptive ridge grid is unsupported")
        finite = tuple(label for label in ridges if label != _STATIC_RIDGE)
        ridge_order = _ordered_identifiers(
            self.ridge_simplicity_order,
            label="ridge simplicity order",
            allow_empty=not finite,
        )
        if set(ridge_order) != set(finite):
            raise ValueError(
                "ridge simplicity order must cover every finite ridge"
            )
        static = _identifier(
            self.static_candidate_id,
            label="static candidate ID",
        )
        fit_ids = {
            adaptive_generator_innovation_fit_candidate_id(candidate, ridge)
            for candidate in ids
            for ridge in finite
        }
        if static in ids or static in fit_ids:
            raise ValueError("static candidate ID collides with feature arms")
        if (
            type(self.minimum_family_count) is not int
            or self.minimum_family_count < 4
            or (
                self.required_prompts_per_family is not None
                and (
                    type(self.required_prompts_per_family) is not int
                    or self.required_prompts_per_family <= 0
                )
            )
        ):
            raise ValueError("adaptive protocol family geometry is invalid")
        object.__setattr__(self, "candidate_specs", specs)
        object.__setattr__(self, "candidate_simplicity_order", simplicity)
        object.__setattr__(self, "portfolio_specs", portfolios)
        object.__setattr__(
            self, "static_reference_candidate_id", reference
        )
        object.__setattr__(self, "v1_candidate_id", v1)
        object.__setattr__(self, "conditional_ridge_labels", ridges)
        object.__setattr__(self, "ridge_simplicity_order", ridge_order)
        object.__setattr__(self, "static_candidate_id", static)
        object.__setattr__(
            self,
            "protocol_sha256",
            _sha256(_PROTOCOL_DOMAIN, self._payload()),
        )

    @property
    def finite_ridge_labels(self) -> tuple[str, ...]:
        return tuple(
            label
            for label in self.conditional_ridge_labels
            if label != _STATIC_RIDGE
        )

    @property
    def ordered_fit_candidate_ids(self) -> tuple[str, ...]:
        return (
            self.static_candidate_id,
            *(
                adaptive_generator_innovation_fit_candidate_id(
                    candidate_id,
                    ridge,
                )
                for ridge in self.ridge_simplicity_order
                for candidate_id in self.candidate_simplicity_order
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _PROTOCOL_SCHEMA,
            "candidate_specs": tuple(
                spec.to_dict() for spec in self.candidate_specs
            ),
            "candidate_simplicity_order": (
                self.candidate_simplicity_order
            ),
            "portfolio_specs": tuple(
                spec.to_dict() for spec in self.portfolio_specs
            ),
            "static_reference_candidate_id": (
                self.static_reference_candidate_id
            ),
            "v1_candidate_id": self.v1_candidate_id,
            "conditional_ridge_labels": self.conditional_ridge_labels,
            "ridge_simplicity_order": self.ridge_simplicity_order,
            "static_candidate_id": self.static_candidate_id,
            "minimum_family_count": self.minimum_family_count,
            "required_prompts_per_family": (
                self.required_prompts_per_family
            ),
            "selection_rule": (
                "simplest_candidate_within_one_standard_error_of_"
                "minimum_inner_family_macro_rmse_ratio"
            ),
            "family_weighting": (
                "equal_family_then_equal_prompt_then_equal_token"
            ),
            "ordered_fit_candidate_ids": (
                self.ordered_fit_candidate_ids
            ),
            "static_represented_once": True,
            "tokens_are_split_units": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "protocol_sha256": self.protocol_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> AdaptiveGeneratorInnovationV2Protocol:
        expected = {
            "schema",
            "candidate_specs",
            "candidate_simplicity_order",
            "portfolio_specs",
            "static_reference_candidate_id",
            "v1_candidate_id",
            "conditional_ridge_labels",
            "ridge_simplicity_order",
            "static_candidate_id",
            "minimum_family_count",
            "required_prompts_per_family",
            "selection_rule",
            "family_weighting",
            "ordered_fit_candidate_ids",
            "static_represented_once",
            "tokens_are_split_units",
            "protocol_sha256",
        }
        if set(value) != expected or value.get("schema") != _PROTOCOL_SCHEMA:
            raise ValueError("adaptive protocol fields differ")
        candidates_raw = value["candidate_specs"]
        portfolios_raw = value["portfolio_specs"]
        if not isinstance(candidates_raw, (tuple, list)) or not isinstance(
            portfolios_raw, (tuple, list)
        ):
            raise TypeError("serialized adaptive protocol lists differ")
        result = cls(
            candidate_specs=tuple(
                AdaptiveGeneratorInnovationCandidateSpec.from_dict(
                    item  # type: ignore[arg-type]
                )
                for item in candidates_raw
            ),
            candidate_simplicity_order=_ordered_identifiers(
                value["candidate_simplicity_order"],
                label="serialized candidate simplicity",
            ),
            portfolio_specs=tuple(
                AdaptiveGeneratorInnovationPortfolioSpec.from_dict(
                    item  # type: ignore[arg-type]
                )
                for item in portfolios_raw
            ),
            static_reference_candidate_id=str(
                value["static_reference_candidate_id"]
            ),
            v1_candidate_id=str(value["v1_candidate_id"]),
            conditional_ridge_labels=_ordered_identifiers(
                value["conditional_ridge_labels"],
                label="serialized ridge labels",
            ),
            ridge_simplicity_order=_ordered_identifiers(
                value["ridge_simplicity_order"],
                label="serialized ridge simplicity",
                allow_empty=True,
            ),
            static_candidate_id=str(value["static_candidate_id"]),
            minimum_family_count=int(value["minimum_family_count"]),
            required_prompts_per_family=(
                None
                if value["required_prompts_per_family"] is None
                else int(value["required_prompts_per_family"])
            ),
        )
        if not _canonical_equal(result.to_dict(), value):
            raise ValueError("serialized adaptive protocol receipt differs")
        return result


@dataclass(frozen=True, slots=True)
class AdaptiveGeneratorInnovationEligibilityReceipt:
    """Target-blind activation-only admission decisions from the scale pass."""

    protocol_sha256: str
    scale_receipt_sha256: str
    eligible_candidate_ids: tuple[str, ...]
    feature_health_receipt_sha256_by_candidate: tuple[
        tuple[str, str], ...
    ]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        protocol = _require_sha256(
            self.protocol_sha256,
            label="eligibility protocol",
        )
        scale = _require_sha256(
            self.scale_receipt_sha256,
            label="activation-only scale receipt",
        )
        eligible = _ordered_identifiers(
            self.eligible_candidate_ids,
            label="eligible candidate IDs",
            allow_empty=True,
        )
        raw = tuple(self.feature_health_receipt_sha256_by_candidate)
        health: list[tuple[str, str]] = []
        for index, row in enumerate(raw):
            if (
                not isinstance(row, (tuple, list))
                or len(row) != 2
            ):
                raise ValueError(
                    f"feature-health receipt {index} must be a pair"
                )
            health.append(
                (
                    _identifier(
                        row[0],
                        label=f"feature-health candidate {index}",
                    ),
                    _require_sha256(
                        row[1],
                        label=f"feature-health receipt {index}",
                    ),
                )
            )
        canonical = tuple(sorted(health))
        if len({name for name, _value in canonical}) != len(canonical):
            raise ValueError("feature-health candidate IDs must be unique")
        object.__setattr__(self, "protocol_sha256", protocol)
        object.__setattr__(self, "scale_receipt_sha256", scale)
        object.__setattr__(self, "eligible_candidate_ids", eligible)
        object.__setattr__(
            self,
            "feature_health_receipt_sha256_by_candidate",
            canonical,
        )
        object.__setattr__(
            self,
            "receipt_sha256",
            _sha256(_ELIGIBILITY_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _ELIGIBILITY_SCHEMA,
            "protocol_sha256": self.protocol_sha256,
            "scale_receipt_sha256": self.scale_receipt_sha256,
            "eligible_candidate_ids": self.eligible_candidate_ids,
            "feature_health_receipt_sha256_by_candidate": dict(
                self.feature_health_receipt_sha256_by_candidate
            ),
            "source": (
                "activation_only_target_blind_scale_and_feature_health_pass"
            ),
            "feature_health_rule": (
                "both_channels_q90_abs_h_below_0p95_and_"
                "central_fraction_abs_h_between_0p1_and_0p9_at_least_0p5"
            ),
            "target_loss_or_vjp_opened": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> AdaptiveGeneratorInnovationEligibilityReceipt:
        expected = {
            "schema",
            "protocol_sha256",
            "scale_receipt_sha256",
            "eligible_candidate_ids",
            "feature_health_receipt_sha256_by_candidate",
            "source",
            "feature_health_rule",
            "target_loss_or_vjp_opened",
            "receipt_sha256",
        }
        if set(value) != expected or value.get("schema") != _ELIGIBILITY_SCHEMA:
            raise ValueError("adaptive eligibility fields differ")
        health = value["feature_health_receipt_sha256_by_candidate"]
        if not isinstance(health, Mapping):
            raise TypeError("serialized feature-health receipts differ")
        result = cls(
            protocol_sha256=str(value["protocol_sha256"]),
            scale_receipt_sha256=str(value["scale_receipt_sha256"]),
            eligible_candidate_ids=_ordered_identifiers(
                value["eligible_candidate_ids"],
                label="serialized eligible candidates",
                allow_empty=True,
            ),
            feature_health_receipt_sha256_by_candidate=tuple(
                sorted((str(key), str(item)) for key, item in health.items())
            ),
        )
        if not _canonical_equal(result.to_dict(), value):
            raise ValueError("serialized eligibility receipt differs")
        return result


@dataclass(frozen=True, slots=True)
class _FitCandidate:
    fit_candidate_id: str
    feature_candidate_id: str
    ridge_label: str
    static: bool


def _bind_eligibility(
    protocol: AdaptiveGeneratorInnovationV2Protocol,
    eligibility: AdaptiveGeneratorInnovationEligibilityReceipt,
) -> None:
    candidate_ids = set(protocol.candidate_simplicity_order)
    health_ids = {
        name
        for name, _receipt in (
            eligibility.feature_health_receipt_sha256_by_candidate
        )
    }
    expected_eligible = tuple(
        name
        for name in protocol.candidate_simplicity_order
        if name in set(eligibility.eligible_candidate_ids)
    )
    if (
        eligibility.protocol_sha256 != protocol.protocol_sha256
        or health_ids != candidate_ids
        or not set(eligibility.eligible_candidate_ids).issubset(
            candidate_ids
        )
        or eligibility.eligible_candidate_ids != expected_eligible
    ):
        raise ValueError(
            "activation-only eligibility does not bind the exact protocol"
        )


def _fit_candidates(
    protocol: AdaptiveGeneratorInnovationV2Protocol,
) -> tuple[_FitCandidate, ...]:
    finite = set(protocol.finite_ridge_labels)
    result = [
        _FitCandidate(
            fit_candidate_id=protocol.static_candidate_id,
            feature_candidate_id=protocol.static_reference_candidate_id,
            ridge_label=_STATIC_RIDGE,
            static=True,
        )
    ]
    for ridge in protocol.ridge_simplicity_order:
        for candidate_id in protocol.candidate_simplicity_order:
            if ridge not in finite:
                raise RuntimeError("adaptive ridge registry drifted")
            result.append(
                _FitCandidate(
                    fit_candidate_id=(
                        adaptive_generator_innovation_fit_candidate_id(
                            candidate_id,
                            ridge,
                        )
                    ),
                    feature_candidate_id=candidate_id,
                    ridge_label=ridge,
                    static=False,
                )
            )
    if tuple(row.fit_candidate_id for row in result) != (
        protocol.ordered_fit_candidate_ids
    ):
        raise RuntimeError("adaptive fit candidate order drifted")
    return tuple(result)


def _records_from_bank(
    record_bank: Mapping[str, Sequence[object]],
    *,
    legacy_records: Sequence[object],
    protocol: AdaptiveGeneratorInnovationV2Protocol,
    eligibility: AdaptiveGeneratorInnovationEligibilityReceipt,
) -> tuple[
    dict[str, tuple[TokenLossFisherPromptRecord, ...]],
    tuple[TokenLossFisherPromptRecord, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(record_bank, Mapping) or set(record_bank) != set(
        protocol.candidate_simplicity_order
    ):
        raise ValueError("adaptive record-bank candidate IDs differ")
    bank = {
        candidate_id: _canonical_records(record_bank[candidate_id])
        for candidate_id in protocol.candidate_simplicity_order
    }
    reference = bank[protocol.static_reference_candidate_id]
    if (
        len(reference[0].coordinate_names) != _FIT_WIDTH
        or reference[0].coordinate_names
        != GENERATOR_INNOVATION_TANGENT_ORDER
    ):
        raise ValueError("adaptive R4 coordinate order differs")
    reference_by_example = {row.example_id: row for row in reference}
    example_ids = tuple(sorted(reference_by_example))
    legacy = _canonical_records(legacy_records)
    legacy_by_example = {row.example_id: row for row in legacy}
    if (
        len(legacy[0].coordinate_names) != 6
        or legacy[0].coordinate_names
        != GENERATOR_INNOVATION_SOURCE_COORDINATE_ORDER
        or tuple(sorted(legacy_by_example)) != example_ids
    ):
        raise ValueError("adaptive legacy Q6 panel is not aligned")
    for candidate_id, records in bank.items():
        if (
            records[0].coordinate_names
            != GENERATOR_INNOVATION_TANGENT_ORDER
            or tuple(sorted(row.example_id for row in records))
            != example_ids
        ):
            raise ValueError(
                f"adaptive record bank is not aligned for {candidate_id}"
            )
        by_example = {row.example_id: row for row in records}
        for example_id in example_ids:
            base = reference_by_example[example_id]
            row = by_example[example_id]
            base_fisher = tuple(
                tuple(base.fisher_second_moment[left][right] for right in (0, 1))
                for left in (0, 1)
            )
            row_fisher = tuple(
                tuple(row.fisher_second_moment[left][right] for right in (0, 1))
                for left in (0, 1)
            )
            if (
                row.family_id != base.family_id
                or row.supervised_tokens != base.supervised_tokens
                or row.compensation_target_sha256
                != base.compensation_target_sha256
                or not _same_number(
                    row.target_second_moment,
                    base.target_second_moment,
                )
                or row_fisher != base_fisher
                or row.target_cross_moment[:2]
                != base.target_cross_moment[:2]
                or row.mean_score[:2] != base.mean_score[:2]
            ):
                raise ValueError(
                    "adaptive targets or exact static-U shared columns differ "
                    f"for {candidate_id}/{example_id}"
                )
    for example_id in example_ids:
        base = reference_by_example[example_id]
        row = legacy_by_example[example_id]
        if (
            row.family_id != base.family_id
            or row.supervised_tokens != base.supervised_tokens
            or row.compensation_target_sha256
            != base.compensation_target_sha256
            or not _same_number(
                row.target_second_moment,
                base.target_second_moment,
            )
        ):
            raise ValueError(
                "adaptive legacy-shared Q6 target binding differs for "
                f"{example_id}"
            )
    counts = Counter(row.family_id for row in reference)
    family_ids = tuple(sorted(counts))
    if len(family_ids) < protocol.minimum_family_count:
        raise ValueError("adaptive record bank has too few families")
    if (
        protocol.required_prompts_per_family is not None
        and set(counts.values())
        != {protocol.required_prompts_per_family}
    ):
        raise ValueError("adaptive prompts-per-family geometry differs")
    _bind_eligibility(protocol, eligibility)
    return bank, legacy, family_ids, example_ids


def _family_bank(
    bank: Mapping[str, Sequence[TokenLossFisherPromptRecord]],
) -> dict[str, Mapping[str, object]]:
    return {
        candidate_id: _family_moments(
            records,
            tuple(range(_FIT_WIDTH)),
        )
        for candidate_id, records in bank.items()
    }


def _candidate_moments(
    candidate: _FitCandidate,
    family_bank: Mapping[str, Mapping[str, object]],
    family_ids: Sequence[str],
) -> object:
    moments = family_bank[candidate.feature_candidate_id]
    return _mean_moments(tuple(moments[family] for family in family_ids))


def _fit(
    candidate: _FitCandidate,
    family_bank: Mapping[str, Mapping[str, object]],
    family_ids: Sequence[str],
    *,
    basis: Tensor,
) -> dict[str, object]:
    return _fit_generator(
        _candidate_moments(candidate, family_bank, family_ids),
        basis=basis,
        conditional_ridge_label=candidate.ridge_label,
    )


def _rmse(
    candidate: _FitCandidate,
    family_bank: Mapping[str, Mapping[str, object]],
    family_id: str,
    coefficients: Sequence[float],
) -> tuple[float, float]:
    held = family_bank[candidate.feature_candidate_id][family_id]
    coefficient_tensor = torch.tensor(
        tuple(coefficients),
        dtype=torch.float64,
    )
    zero = torch.zeros(_FIT_WIDTH, dtype=torch.float64)
    return (
        _residual_rmse(held, zero),
        _residual_rmse(held, coefficient_tensor),
    )


def _inner_receipt(
    candidate: _FitCandidate,
    *,
    outer_train_family_ids: tuple[str, ...],
    family_bank: Mapping[str, Mapping[str, object]],
    basis: Tensor,
) -> dict[str, object]:
    ratios: list[float] = []
    held_rmse: list[float] = []
    for inner_held in outer_train_family_ids:
        inner_train = tuple(
            family
            for family in outer_train_family_ids
            if family != inner_held
        )
        fit = _fit(candidate, family_bank, inner_train, basis=basis)
        before, after = _rmse(
            candidate,
            family_bank,
            inner_held,
            fit["coefficients"],  # type: ignore[arg-type]
        )
        ratios.append(0.0 if before == 0.0 else after / before)
        held_rmse.append(after)
    mean, standard_error = _mean_and_standard_error(ratios)
    payload = {
        "fit_candidate_id": candidate.fit_candidate_id,
        "feature_candidate_id": candidate.feature_candidate_id,
        "ridge_label": candidate.ridge_label,
        "static": candidate.static,
        "inner_held_family_ids": outer_train_family_ids,
        "inner_held_rmse_ratio_by_family": {
            family: ratio
            for family, ratio in zip(
                outer_train_family_ids,
                ratios,
                strict=True,
            )
        },
        "inner_held_rmse_by_family": {
            family: value
            for family, value in zip(
                outer_train_family_ids,
                held_rmse,
                strict=True,
            )
        },
        "mean_inner_held_rmse_ratio": mean,
        "standard_error_of_mean_inner_held_rmse_ratio": (
            standard_error
        ),
    }
    return {**payload, "inner_receipt_sha256": _sha256(_INNER_DOMAIN, payload)}


def _outer_receipt(
    candidate: _FitCandidate,
    *,
    held_family_id: str,
    train_family_ids: tuple[str, ...],
    family_bank: Mapping[str, Mapping[str, object]],
    bank: Mapping[str, Sequence[TokenLossFisherPromptRecord]],
    basis: Tensor,
) -> dict[str, object]:
    fit = _fit(candidate, family_bank, train_family_ids, basis=basis)
    before, after = _rmse(
        candidate,
        family_bank,
        held_family_id,
        fit["coefficients"],  # type: ignore[arg-type]
    )
    records = bank[candidate.feature_candidate_id]
    payload = {
        "fit_candidate_id": candidate.fit_candidate_id,
        "feature_candidate_id": candidate.feature_candidate_id,
        "ridge_label": candidate.ridge_label,
        "static": candidate.static,
        "train_prompt_record_sha256s": tuple(
            sorted(
                row.prompt_record_sha256
                for row in records
                if row.family_id in train_family_ids
            )
        ),
        "held_prompt_record_sha256s": tuple(
            sorted(
                row.prompt_record_sha256
                for row in records
                if row.family_id == held_family_id
            )
        ),
        "fit": fit,
        "held_parent_rmse": before,
        "held_rmse": after,
        "held_rmse_ratio": 0.0 if before == 0.0 else after / before,
        "held_relative_rmse_improvement_vs_parent": (
            _relative_improvement(before, after)
        ),
    }
    return {**payload, "outer_receipt_sha256": _sha256(_OUTER_DOMAIN, payload)}


def _legacy_outer_receipt(
    *,
    held_family_id: str,
    train_family_ids: tuple[str, ...],
    legacy_family_moments: Mapping[str, object],
    legacy_records: Sequence[TokenLossFisherPromptRecord],
) -> dict[str, object]:
    train = _mean_moments(
        tuple(
            legacy_family_moments[family]
            for family in train_family_ids
        )
    )
    held = legacy_family_moments[held_family_id]
    fit = _fit_legacy_shared(train)
    coefficients = torch.tensor(
        fit["coefficients"],
        dtype=torch.float64,
    )
    zero = torch.zeros(2, dtype=torch.float64)
    before = _residual_rmse(held, zero)
    after = _residual_rmse(held, coefficients)
    payload = {
        "train_prompt_record_sha256s": tuple(
            sorted(
                row.prompt_record_sha256
                for row in legacy_records
                if row.family_id in train_family_ids
            )
        ),
        "held_prompt_record_sha256s": tuple(
            sorted(
                row.prompt_record_sha256
                for row in legacy_records
                if row.family_id == held_family_id
            )
        ),
        "fit": fit,
        "held_parent_rmse": before,
        "held_rmse": after,
        "held_rmse_ratio": 0.0 if before == 0.0 else after / before,
        "held_relative_rmse_improvement_vs_parent": (
            _relative_improvement(before, after)
        ),
    }
    return {
        **payload,
        "legacy_outer_receipt_sha256": _sha256(
            _OUTER_DOMAIN,
            {"legacy_shared": payload},
        ),
    }


def _selection(
    *,
    scope_id: str,
    ordered_candidate_ids: Sequence[str],
    inner_by_id: Mapping[str, Mapping[str, object]],
    outer_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    ordered = tuple(ordered_candidate_ids)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError(f"{scope_id} selection candidates differ")
    rows = tuple(inner_by_id[candidate_id] for candidate_id in ordered)
    best = min(
        rows,
        key=lambda row: (
            float(row["mean_inner_held_rmse_ratio"]),
            ordered.index(str(row["fit_candidate_id"])),
        ),
    )
    threshold = (
        float(best["mean_inner_held_rmse_ratio"])
        + float(best["standard_error_of_mean_inner_held_rmse_ratio"])
    )
    eligible = tuple(
        candidate_id
        for candidate_id in ordered
        if float(
            inner_by_id[candidate_id]["mean_inner_held_rmse_ratio"]
        )
        <= threshold + _NUMBER_TOLERANCE
    )
    selected = eligible[0]
    held = outer_by_id[selected]
    payload = {
        "scope_id": scope_id,
        "ordered_candidate_ids": ordered,
        "best_mean_candidate_id": str(best["fit_candidate_id"]),
        "one_standard_error_threshold": threshold,
        "within_one_standard_error_candidate_ids": eligible,
        "selected_fit_candidate_id": selected,
        "selected_feature_candidate_id": str(
            held["feature_candidate_id"]
        ),
        "selected_ridge_label": str(held["ridge_label"]),
        "selected_static": bool(held["static"]),
        "held_parent_rmse": float(held["held_parent_rmse"]),
        "held_rmse": float(held["held_rmse"]),
        "held_rmse_ratio": float(held["held_rmse_ratio"]),
        "held_relative_rmse_improvement_vs_parent": float(
            held["held_relative_rmse_improvement_vs_parent"]
        ),
    }
    return {
        **payload,
        "selection_receipt_sha256": _sha256(
            _SELECTION_DOMAIN,
            payload,
        ),
    }


def _fold(
    *,
    held_family_id: str,
    family_ids: tuple[str, ...],
    bank: Mapping[str, Sequence[TokenLossFisherPromptRecord]],
    family_bank: Mapping[str, Mapping[str, object]],
    legacy_records: Sequence[TokenLossFisherPromptRecord],
    legacy_family_moments: Mapping[str, object],
    basis: Tensor,
    protocol: AdaptiveGeneratorInnovationV2Protocol,
    eligibility: AdaptiveGeneratorInnovationEligibilityReceipt,
) -> dict[str, object]:
    train_families = tuple(
        family for family in family_ids if family != held_family_id
    )
    candidates = _fit_candidates(protocol)
    inner_rows = tuple(
        _inner_receipt(
            candidate,
            outer_train_family_ids=train_families,
            family_bank=family_bank,
            basis=basis,
        )
        for candidate in candidates
    )
    outer_rows = tuple(
        _outer_receipt(
            candidate,
            held_family_id=held_family_id,
            train_family_ids=train_families,
            family_bank=family_bank,
            bank=bank,
            basis=basis,
        )
        for candidate in candidates
    )
    legacy_outer = _legacy_outer_receipt(
        held_family_id=held_family_id,
        train_family_ids=train_families,
        legacy_family_moments=legacy_family_moments,
        legacy_records=legacy_records,
    )
    inner = {str(row["fit_candidate_id"]): row for row in inner_rows}
    outer = {str(row["fit_candidate_id"]): row for row in outer_rows}
    static_id = protocol.static_candidate_id
    finite_by_feature = {
        feature: tuple(
            adaptive_generator_innovation_fit_candidate_id(
                feature,
                ridge,
            )
            for ridge in protocol.ridge_simplicity_order
        )
        for feature in protocol.candidate_simplicity_order
    }
    variant_selections = tuple(
        {
            "feature_candidate_id": feature,
            "ridge_candidate_id_by_label": {
                **{
                    ridge: adaptive_generator_innovation_fit_candidate_id(
                        feature,
                        ridge,
                    )
                    for ridge in protocol.finite_ridge_labels
                },
                _STATIC_RIDGE: static_id,
            },
            "selection": _selection(
                scope_id=f"feature:{feature}",
                ordered_candidate_ids=(
                    static_id,
                    *finite_by_feature[feature],
                ),
                inner_by_id=inner,
                outer_by_id=outer,
            ),
        }
        for feature in protocol.candidate_simplicity_order
    )
    eligible = set(eligibility.eligible_candidate_ids)
    portfolio_selections = []
    for portfolio in protocol.portfolio_specs:
        fit_ids = (
            static_id,
            *(
                adaptive_generator_innovation_fit_candidate_id(
                    feature,
                    ridge,
                )
                for ridge in protocol.ridge_simplicity_order
                for feature in portfolio.candidate_ids
                if feature in eligible
            ),
        )
        portfolio_selections.append(
            {
                "portfolio_id": portfolio.portfolio_id,
                "declared_feature_candidate_ids": (
                    portfolio.candidate_ids
                ),
                "eligible_feature_candidate_ids": tuple(
                    feature
                    for feature in portfolio.candidate_ids
                    if feature in eligible
                ),
                "selection": _selection(
                    scope_id=f"portfolio:{portfolio.portfolio_id}",
                    ordered_candidate_ids=fit_ids,
                    inner_by_id=inner,
                    outer_by_id=outer,
                ),
            }
        )
    reference_outer = outer[static_id]
    for row in outer_rows:
        if not _same_number(
            float(row["held_parent_rmse"]),
            float(reference_outer["held_parent_rmse"]),
        ):
            raise RuntimeError(
                "adaptive aligned candidate parent RMSE differs"
            )
    if not _same_number(
        float(reference_outer["held_parent_rmse"]),
        float(legacy_outer["held_parent_rmse"]),
    ):
        raise RuntimeError("adaptive legacy parent RMSE differs")
    payload = {
        "held_family_id": held_family_id,
        "train_family_ids": train_families,
        "train_example_ids": tuple(
            sorted(
                row.example_id
                for row in bank[protocol.static_reference_candidate_id]
                if row.family_id in train_families
            )
        ),
        "held_example_ids": tuple(
            sorted(
                row.example_id
                for row in bank[protocol.static_reference_candidate_id]
                if row.family_id == held_family_id
            )
        ),
        "ordered_fit_candidate_ids": (
            protocol.ordered_fit_candidate_ids
        ),
        "inner_candidate_receipts": inner_rows,
        "outer_candidate_receipts": outer_rows,
        "legacy_shared_outer_receipt": legacy_outer,
        "variant_ridge_selections": variant_selections,
        "portfolio_selections": tuple(portfolio_selections),
    }
    return {**payload, "fold_sha256": _sha256(_FOLD_DOMAIN, payload)}


def _selected_metric(
    *,
    family_ids: tuple[str, ...],
    selected_by_family: Mapping[str, Mapping[str, object]],
    parent_by_family: Mapping[str, float],
) -> dict[str, object]:
    held = {
        family: float(selected_by_family[family]["held_rmse"])
        for family in family_ids
    }
    relative = {
        family: _relative_improvement(parent_by_family[family], held[family])
        for family in family_ids
    }
    parent_macro = math.fsum(parent_by_family.values()) / len(family_ids)
    macro = math.fsum(held.values()) / len(family_ids)
    return {
        "family_macro_rmse": macro,
        "family_macro_relative_rmse_improvement_vs_parent": (
            _relative_improvement(parent_macro, macro)
        ),
        "family_win_count_vs_parent": sum(
            value > 0.0 for value in relative.values()
        ),
        "worst_family_relative_rmse_improvement_vs_parent": min(
            relative.values()
        ),
        "held_rmse_by_family": held,
        "held_relative_rmse_improvement_vs_parent_by_family": relative,
        "selected_fit_candidate_id_by_family": {
            family: str(
                selected_by_family[family]["selected_fit_candidate_id"]
            )
            for family in family_ids
        },
    }


def _metrics(
    folds: Sequence[Mapping[str, object]],
    *,
    family_ids: tuple[str, ...],
    protocol: AdaptiveGeneratorInnovationV2Protocol,
) -> dict[str, object]:
    fold_by_family = {
        str(fold["held_family_id"]): fold for fold in folds
    }
    outer_by_family = {
        family: {
            str(row["fit_candidate_id"]): row
            for row in fold_by_family[family]["outer_candidate_receipts"]
        }
        for family in family_ids
    }
    parent = {
        family: float(
            outer_by_family[family][protocol.static_candidate_id][
                "held_parent_rmse"
            ]
        )
        for family in family_ids
    }
    parent_macro = math.fsum(parent.values()) / len(parent)
    legacy_selected = {
        family: {
            **fold_by_family[family]["legacy_shared_outer_receipt"],
            "selected_fit_candidate_id": "legacy_shared_q6_first_two",
        }
        for family in family_ids
    }
    legacy_metric = _selected_metric(
        family_ids=family_ids,
        selected_by_family=legacy_selected,
        parent_by_family=parent,
    )
    fixed = {}
    for candidate_id in protocol.ordered_fit_candidate_ids:
        selected = {
            family: {
                **outer_by_family[family][candidate_id],
                "selected_fit_candidate_id": candidate_id,
            }
            for family in family_ids
        }
        fixed[candidate_id] = _selected_metric(
            family_ids=family_ids,
            selected_by_family=selected,
            parent_by_family=parent,
        )
    variant = {}
    for feature in protocol.candidate_simplicity_order:
        selected_by_family = {}
        for family in family_ids:
            rows = fold_by_family[family]["variant_ridge_selections"]
            match = next(
                row
                for row in rows
                if row["feature_candidate_id"] == feature
            )
            selected_by_family[family] = match["selection"]
        variant[feature] = _selected_metric(
            family_ids=family_ids,
            selected_by_family=selected_by_family,
            parent_by_family=parent,
        )
    portfolios = {}
    for spec in protocol.portfolio_specs:
        selected_by_family = {}
        for family in family_ids:
            rows = fold_by_family[family]["portfolio_selections"]
            match = next(
                row
                for row in rows
                if row["portfolio_id"] == spec.portfolio_id
            )
            selected_by_family[family] = match["selection"]
        metric = _selected_metric(
            family_ids=family_ids,
            selected_by_family=selected_by_family,
            parent_by_family=parent,
        )
        v1 = variant[protocol.v1_candidate_id]
        v1_by_family = v1["held_rmse_by_family"]
        selected_rmse = metric["held_rmse_by_family"]
        relative_v1 = {
            family: _relative_improvement(
                float(v1_by_family[family]),
                float(selected_rmse[family]),
            )
            for family in family_ids
        }
        metric["family_macro_relative_rmse_improvement_vs_v1"] = (
            _relative_improvement(
                float(v1["family_macro_rmse"]),
                float(metric["family_macro_rmse"]),
            )
        )
        metric["family_win_count_vs_v1"] = sum(
            value > 0.0 for value in relative_v1.values()
        )
        metric[
            "worst_family_relative_rmse_improvement_vs_v1"
        ] = min(relative_v1.values())
        metric[
            "held_relative_rmse_improvement_vs_v1_by_family"
        ] = relative_v1
        legacy_by_family = legacy_metric["held_rmse_by_family"]
        relative_legacy = {
            family: _relative_improvement(
                float(legacy_by_family[family]),
                float(selected_rmse[family]),
            )
            for family in family_ids
        }
        metric[
            "family_macro_relative_rmse_improvement_vs_legacy_shared"
        ] = _relative_improvement(
            float(legacy_metric["family_macro_rmse"]),
            float(metric["family_macro_rmse"]),
        )
        metric["family_win_count_vs_legacy_shared"] = sum(
            value > 0.0 for value in relative_legacy.values()
        )
        metric[
            "worst_family_relative_rmse_improvement_vs_legacy_shared"
        ] = min(relative_legacy.values())
        metric[
            "held_relative_rmse_improvement_vs_legacy_shared_by_family"
        ] = relative_legacy
        portfolios[spec.portfolio_id] = metric
    v1_metric = variant[protocol.v1_candidate_id]
    v1_macro_vs_legacy = _relative_improvement(
        float(legacy_metric["family_macro_rmse"]),
        float(v1_metric["family_macro_rmse"]),
    )
    return {
        "family_macro_parent_rmse": parent_macro,
        "held_parent_rmse_by_family": parent,
        "fixed_fit_candidate_metrics": fixed,
        "ridge_selected_variant_metrics": variant,
        "static_u_metrics": fixed[protocol.static_candidate_id],
        "legacy_shared_metrics": legacy_metric,
        "v1_metrics": v1_metric,
        "v1_family_macro_relative_rmse_improvement_vs_legacy_shared": (
            v1_macro_vs_legacy
        ),
        "v1_structural_gate_macro_not_worse_than_legacy_shared": (
            v1_macro_vs_legacy >= 0.0
        ),
        "portfolio_metrics": portfolios,
    }


def build_generator_innovation_adaptive_v2_report(
    record_bank: Mapping[str, Sequence[object]],
    *,
    legacy_records: Sequence[object],
    fixed_basis: Sequence[Sequence[float]],
    protocol: AdaptiveGeneratorInnovationV2Protocol,
    eligibility: AdaptiveGeneratorInnovationEligibilityReceipt,
) -> dict[str, object]:
    """Build every fixed arm and honest adaptive portfolio from prompt moments."""

    if not isinstance(protocol, AdaptiveGeneratorInnovationV2Protocol):
        raise TypeError("adaptive v2 protocol has the wrong type")
    if not isinstance(
        eligibility,
        AdaptiveGeneratorInnovationEligibilityReceipt,
    ):
        raise TypeError("adaptive v2 eligibility has the wrong type")
    basis = _basis_tensor(fixed_basis)
    bank, legacy, family_ids, example_ids = _records_from_bank(
        record_bank,
        legacy_records=legacy_records,
        protocol=protocol,
        eligibility=eligibility,
    )
    moments = _family_bank(bank)
    legacy_moments = _family_moments(legacy, (0, 1))
    folds = tuple(
        _fold(
            held_family_id=family,
            family_ids=family_ids,
            bank=bank,
            family_bank=moments,
            legacy_records=legacy,
            legacy_family_moments=legacy_moments,
            basis=basis,
            protocol=protocol,
            eligibility=eligibility,
        )
        for family in family_ids
    )
    metrics = _metrics(folds, family_ids=family_ids, protocol=protocol)
    reference = {
        row.example_id: row
        for row in bank[protocol.static_reference_candidate_id]
    }
    payload: dict[str, object] = {
        "schema": ADAPTIVE_GENERATOR_INNOVATION_V2_SCHEMA,
        "protocol": protocol.to_dict(),
        "eligibility": eligibility.to_dict(),
        "fixed_basis": _basis_payload(basis),
        "ordered_candidate_ids": protocol.candidate_simplicity_order,
        "ordered_fit_candidate_ids": (
            protocol.ordered_fit_candidate_ids
        ),
        "family_ids": family_ids,
        "example_ids": example_ids,
        "family_id_by_example_id": {
            example_id: reference[example_id].family_id
            for example_id in example_ids
        },
        "target_sha256_by_example_id": {
            example_id: reference[
                example_id
            ].compensation_target_sha256
            for example_id in example_ids
        },
        "record_bank": {
            candidate_id: tuple(
                row.to_dict() for row in bank[candidate_id]
            )
            for candidate_id in protocol.candidate_simplicity_order
        },
        "legacy_prompt_fisher_records": tuple(
            row.to_dict() for row in legacy
        ),
        "folds": folds,
        "metrics": metrics,
        "decision": {
            "adaptive_development_only": True,
            "every_predeclared_arm_scored": True,
            "portfolio_selection_family_nested": True,
            "held_family_used_for_selection": False,
            "ineligible_control_scores_reported": True,
            "ineligible_candidate_selectable_by_portfolio": False,
            "v1_reported_even_when_ineligible": True,
            "v1_structural_gate_macro_not_worse_than_legacy_shared": (
                metrics[
                    (
                        "v1_structural_gate_macro_not_worse_than_"
                        "legacy_shared"
                    )
                ]
            ),
            "finite_displacement_or_provider_authorized": False,
            "compression_or_runtime_claim_authorized": False,
            "next_step": (
                "inspect_family_nested_rate_curves_then_preregister_"
                "any_finite_validation"
            ),
        },
        "audit": {
            "input_level": "prompt_sufficient_statistics",
            "outer_split_unit": "family",
            "inner_split_unit": "family",
            "token_used_as_independent_split_unit": False,
            "family_prompt_token_weighting": (
                "equal_family_then_equal_prompt_then_equal_token"
            ),
            "eligibility_computed_before_target_or_vjp": True,
            "exact_static_u_control_represented_once": True,
            "candidate_grid_defined_by_protocol_not_analyzer": True,
            "raw_prompt_text_retained": False,
            "raw_token_ids_retained": False,
            "raw_token_score_rows_retained": False,
            "raw_activations_retained": False,
            "raw_gradients_retained": False,
            "raw_logits_retained": False,
        },
    }
    return {
        **payload,
        "report_sha256": _sha256(_REPORT_DOMAIN, payload),
    }


def validate_generator_innovation_adaptive_v2_report(
    report: object,
) -> None:
    """Validate report structure, receipts, partitions, and canonical hash."""

    if not isinstance(report, Mapping):
        raise TypeError("adaptive v2 report must be a mapping")
    expected = {
        "schema",
        "protocol",
        "eligibility",
        "fixed_basis",
        "ordered_candidate_ids",
        "ordered_fit_candidate_ids",
        "family_ids",
        "example_ids",
        "family_id_by_example_id",
        "target_sha256_by_example_id",
        "record_bank",
        "legacy_prompt_fisher_records",
        "folds",
        "metrics",
        "decision",
        "audit",
        "report_sha256",
    }
    if set(report) != expected or report.get(
        "schema"
    ) != ADAPTIVE_GENERATOR_INNOVATION_V2_SCHEMA:
        raise ValueError("adaptive v2 report fields differ")
    protocol_raw = report["protocol"]
    eligibility_raw = report["eligibility"]
    if not isinstance(protocol_raw, Mapping) or not isinstance(
        eligibility_raw, Mapping
    ):
        raise TypeError("adaptive protocol/eligibility receipts differ")
    protocol = AdaptiveGeneratorInnovationV2Protocol.from_dict(protocol_raw)
    eligibility = (
        AdaptiveGeneratorInnovationEligibilityReceipt.from_dict(
            eligibility_raw
        )
    )
    _bind_eligibility(protocol, eligibility)
    basis_raw = report["fixed_basis"]
    if not isinstance(basis_raw, Mapping):
        raise TypeError("adaptive fixed basis receipt differs")
    basis = _basis_tensor(basis_raw.get("rows"))  # type: ignore[arg-type]
    if not _canonical_equal(_basis_payload(basis), basis_raw):
        raise ValueError("adaptive fixed basis receipt differs")
    if (
        tuple(report["ordered_candidate_ids"])
        != protocol.candidate_simplicity_order
        or tuple(report["ordered_fit_candidate_ids"])
        != protocol.ordered_fit_candidate_ids
    ):
        raise ValueError("adaptive ordered candidate IDs differ")
    family_ids = tuple(report["family_ids"])
    example_ids = tuple(report["example_ids"])
    if (
        family_ids != tuple(sorted(set(family_ids)))
        or len(family_ids) < protocol.minimum_family_count
        or example_ids != tuple(sorted(set(example_ids)))
        or not example_ids
    ):
        raise ValueError("adaptive family/example IDs differ")
    bank_raw = report["record_bank"]
    if not isinstance(bank_raw, Mapping) or set(bank_raw) != set(
        protocol.candidate_simplicity_order
    ):
        raise ValueError("adaptive serialized record bank differs")
    for candidate_id in protocol.candidate_simplicity_order:
        rows = bank_raw[candidate_id]
        if not isinstance(rows, (tuple, list)):
            raise TypeError("adaptive serialized prompt records differ")
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("adaptive prompt record must be a mapping")
            token_loss_fisher_prompt_record_from_dict(row)
    legacy_raw = report["legacy_prompt_fisher_records"]
    if not isinstance(legacy_raw, (tuple, list)):
        raise TypeError("adaptive serialized legacy records differ")
    for row in legacy_raw:
        if not isinstance(row, Mapping):
            raise TypeError("adaptive legacy record must be a mapping")
        token_loss_fisher_prompt_record_from_dict(row)
    folds = report["folds"]
    if (
        not isinstance(folds, (tuple, list))
        or len(folds) != len(family_ids)
        or tuple(row["held_family_id"] for row in folds) != family_ids
    ):
        raise ValueError("adaptive outer folds differ")
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise TypeError("adaptive fold must be a mapping")
        payload = dict(fold)
        receipt = payload.pop("fold_sha256", None)
        if receipt != _sha256(_FOLD_DOMAIN, payload):
            raise ValueError("adaptive fold receipt differs")
        if (
            fold["held_family_id"] in tuple(fold["train_family_ids"])
            or tuple(fold["ordered_fit_candidate_ids"])
            != protocol.ordered_fit_candidate_ids
        ):
            raise ValueError("adaptive held family leaked into training")
        for field_name, domain, receipt_name in (
            (
                "inner_candidate_receipts",
                _INNER_DOMAIN,
                "inner_receipt_sha256",
            ),
            (
                "outer_candidate_receipts",
                _OUTER_DOMAIN,
                "outer_receipt_sha256",
            ),
        ):
            rows = fold[field_name]
            if tuple(row["fit_candidate_id"] for row in rows) != (
                protocol.ordered_fit_candidate_ids
            ):
                raise ValueError(
                    f"adaptive {field_name} order differs"
                )
            for row in rows:
                candidate_payload = dict(row)
                candidate_receipt = candidate_payload.pop(
                    receipt_name,
                    None,
                )
                if candidate_receipt != _sha256(
                    domain,
                    candidate_payload,
                ):
                    raise ValueError(
                        f"adaptive {field_name} receipt differs"
                    )
        legacy_receipt = fold["legacy_shared_outer_receipt"]
        legacy_payload = dict(legacy_receipt)
        legacy_hash = legacy_payload.pop(
            "legacy_outer_receipt_sha256",
            None,
        )
        if legacy_hash != _sha256(
            _OUTER_DOMAIN,
            {"legacy_shared": legacy_payload},
        ):
            raise ValueError("adaptive legacy outer receipt differs")
        for group_name in (
            "variant_ridge_selections",
            "portfolio_selections",
        ):
            for group in fold[group_name]:
                selection = group["selection"]
                selection_payload = dict(selection)
                selection_receipt = selection_payload.pop(
                    "selection_receipt_sha256",
                    None,
                )
                if selection_receipt != _sha256(
                    _SELECTION_DOMAIN,
                    selection_payload,
                ):
                    raise ValueError("adaptive selection receipt differs")
    decision = report["decision"]
    audit = report["audit"]
    if (
        not isinstance(decision, Mapping)
        or decision.get("adaptive_development_only") is not True
        or decision.get("held_family_used_for_selection") is not False
        or decision.get("finite_displacement_or_provider_authorized")
        is not False
        or not isinstance(audit, Mapping)
        or audit.get("outer_split_unit") != "family"
        or audit.get("inner_split_unit") != "family"
        or audit.get("token_used_as_independent_split_unit") is not False
        or audit.get("eligibility_computed_before_target_or_vjp") is not True
        or audit.get("raw_token_score_rows_retained") is not False
    ):
        raise ValueError("adaptive decision/audit differs")
    payload = dict(report)
    receipt = payload.pop("report_sha256", None)
    if receipt != _sha256(_REPORT_DOMAIN, payload):
        raise ValueError("adaptive v2 report hash mismatch")


def replay_generator_innovation_adaptive_v2_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the complete report from its retained prompt statistics."""

    validate_generator_innovation_adaptive_v2_report(report)
    protocol = AdaptiveGeneratorInnovationV2Protocol.from_dict(
        report["protocol"]  # type: ignore[arg-type]
    )
    eligibility = AdaptiveGeneratorInnovationEligibilityReceipt.from_dict(
        report["eligibility"]  # type: ignore[arg-type]
    )
    basis = report["fixed_basis"]
    bank = report["record_bank"]
    rebuilt = build_generator_innovation_adaptive_v2_report(
        bank,  # type: ignore[arg-type]
        legacy_records=report["legacy_prompt_fisher_records"],  # type: ignore[arg-type]
        fixed_basis=basis["rows"],  # type: ignore[index]
        protocol=protocol,
        eligibility=eligibility,
    )
    if not _canonical_equal(rebuilt, report):
        raise ValueError("adaptive v2 report replay differs")
    return rebuilt
