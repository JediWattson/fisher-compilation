"""Frozen-B confirmation for A-discovered tied lexical mode candidates.

This module never discovers, refits, filters, or reranks lexical membership.
It consumes the authenticated A dictionary, redacts the capture to the frozen
``B_confirmation`` role, and evaluates every candidate only at its already
declared ingress/readout role.  The evidence is a signed within-family-centered
response cosine, an exact 64-state test per pair, and the complete 4,096-state
independent familywise correction.

Because B was visible while this exact policy was finalized, the result is
retrospective and cannot authorize attention promotion; a fresh untouched
panel is required.  It is not semantic confirmation, a compression result,
or an attention chart.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Final

import numpy as np

from . import gemma3_parts_first_lexical_modes as lexical_modes
from .gemma3_parts_first_supermode_experiment import (
    AtomResponseAxisEntry,
    CaptureRole,
    ExactLiftCaptureReceipt,
    PartsFirstCaptureBundle,
)
from .gemma3_parts_first_supermodes import HIDDEN_WIDTH


__all__ = [
    "DEFAULT_LEXICAL_MODE_B_OUTPUT",
    "LEXICAL_MODE_B_CONFIRMATION_ROLE",
    "LEXICAL_MODE_B_FORMAT_VERSION",
    "LEXICAL_MODE_B_SCHEMA",
    "PINNED_LEXICAL_B_ROLE_SHA256",
    "PINNED_LEXICAL_B_ROW_IDS_SHA256",
    "PINNED_LEXICAL_B_SOURCE_SHA256",
    "PINNED_LEXICAL_B_CONFIRMATION_RECEIPT_SHA256",
    "FrozenBFamilyBlockPermutationEvidence",
    "FrozenBRoleQualification",
    "FrozenBCandidateQualification",
    "LexicalModeBConfirmationPolicy",
    "LexicalModeBConfirmationReceipt",
    "LexicalModeBSource",
    "build_lexical_mode_B_source",
    "compile_frozen_B_lexical_mode_confirmation",
    "compile_published_frozen_B_lexical_mode_confirmation",
    "load_lexical_mode_B_confirmation",
    "main",
    "parse_lexical_mode_B_confirmation_json",
    "validate_lexical_mode_B_confirmation",
    "write_lexical_mode_B_confirmation",
]


LEXICAL_MODE_B_SCHEMA: Final = (
    "fisher_graph.gemma3_parts_first.lexical_mode_B_confirmation.v1"
)
LEXICAL_MODE_B_FORMAT_VERSION: Final = 1
LEXICAL_MODE_B_CONFIRMATION_ROLE: Final = "B_confirmation"
LEXICAL_MODE_B_PROFILE_ABI: Final = (
    "role_scoped_within_family_centered_nminus1_signed_cosine.v1"
)
LEXICAL_MODE_B_PERMUTATION_ABI: Final = (
    "exact_independent_2hypothesis_64x64_family_block_maxT_one_sided.v1"
)
DEFAULT_LEXICAL_MODE_B_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/"
    "parts-first-lexical-mode-B-confirmation-v1.json"
)
PINNED_LEXICAL_B_ROLE_SHA256: Final = (
    "bfe4d8618a5c784d81b13ba915198352ab0baf92a19eecb3ded42ea6a8adcff6"
)
PINNED_LEXICAL_B_ROW_IDS_SHA256: Final = (
    "4e536180fd43bea6dcdf302e9730f526565f591ecc581fc895597a4fe2115aff"
)
# Filled from the canonical redacted B envelope; unlike the full capture hash,
# this binds axis order, static metadata, exact-lift evidence, and B rows only.
PINNED_LEXICAL_B_SOURCE_SHA256: Final = (
    "de0752688113899766367bc47cea43f1666d8016a6099137f07e15feb9032674"
)
PINNED_LEXICAL_B_CONFIRMATION_RECEIPT_SHA256: Final = (
    "69598504b77ff062f7e519205f5a673f79285aa57d679021fc8700d06d156275"
)

_SOURCE_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-source:v1\0"
_POLICY_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-policy:v1\0"
_PAIR_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-pair:v1\0"
_ROLE_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-role:v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-candidate:v1\0"
_RECEIPT_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-receipt:v1\0"
_AXIS_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-axis:v1\0"
_INVENTORY_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-B-inventory:v1\0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool")
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
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} field inventory is invalid")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_object(source: str | bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            source,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_json_path(value: Path | str, *, label: str) -> Path:
    path = Path(value)
    if path.suffix != ".json":
        raise ValueError(f"{label} must use a .json suffix")
    return path


@dataclass(frozen=True, slots=True)
class LexicalModeBConfirmationPolicy:
    orientation_preserving_cosine_floor: float = 0.75
    maximum_exact_permutation_p_value: float = 0.05
    family_count: int = 6
    rows_per_family: int = 2
    exact_permutation_count: int = 64
    frozen_hypothesis_count: int = 2
    familywise_permutation_count: int = 4_096
    profile_abi: str = LEXICAL_MODE_B_PROFILE_ABI
    permutation_abi: str = LEXICAL_MODE_B_PERMUTATION_ABI
    policy_sha256: str = ""

    def __post_init__(self) -> None:
        if self.orientation_preserving_cosine_floor != 0.75:
            raise ValueError("frozen B cosine floor must remain 0.75")
        if self.maximum_exact_permutation_p_value != 0.05:
            raise ValueError("frozen B exact-test level must remain 0.05")
        if (
            self.family_count != 6
            or self.rows_per_family != 2
            or self.exact_permutation_count != 2**6
            or self.frozen_hypothesis_count != 2
            or self.familywise_permutation_count != 64**2
        ):
            raise ValueError("frozen B family-block geometry drifted")
        if (
            self.profile_abi != LEXICAL_MODE_B_PROFILE_ABI
            or self.permutation_abi != LEXICAL_MODE_B_PERMUTATION_ABI
        ):
            raise ValueError("frozen B qualification ABI drifted")
        computed = _json_sha256(self._payload(), domain=_POLICY_DOMAIN)
        if self.policy_sha256 == "":
            object.__setattr__(self, "policy_sha256", computed)
        elif _require_sha256(
            self.policy_sha256, label="policy_sha256"
        ) != computed:
            raise ValueError("frozen B policy hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "orientation_preserving_cosine_floor": (
                self.orientation_preserving_cosine_floor
            ),
            "maximum_exact_permutation_p_value": (
                self.maximum_exact_permutation_p_value
            ),
            "family_count": self.family_count,
            "rows_per_family": self.rows_per_family,
            "exact_permutation_count": self.exact_permutation_count,
            "frozen_hypothesis_count": self.frozen_hypothesis_count,
            "familywise_permutation_count": self.familywise_permutation_count,
            "profile_abi": self.profile_abi,
            "permutation_abi": self.permutation_abi,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "policy_sha256": self.policy_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "LexicalModeBConfirmationPolicy":
        _exact_keys(
            value,
            {
                "orientation_preserving_cosine_floor",
                "maximum_exact_permutation_p_value",
                "family_count",
                "rows_per_family",
                "exact_permutation_count",
                "frozen_hypothesis_count",
                "familywise_permutation_count",
                "profile_abi",
                "permutation_abi",
                "policy_sha256",
            },
            label="frozen B policy",
        )
        return cls(
            orientation_preserving_cosine_floor=_finite_float(
                value["orientation_preserving_cosine_floor"],
                label="orientation_preserving_cosine_floor",
            ),
            maximum_exact_permutation_p_value=_finite_float(
                value["maximum_exact_permutation_p_value"],
                label="maximum_exact_permutation_p_value",
            ),
            family_count=_require_int(value["family_count"], label="family_count"),
            rows_per_family=_require_int(
                value["rows_per_family"], label="rows_per_family"
            ),
            exact_permutation_count=_require_int(
                value["exact_permutation_count"],
                label="exact_permutation_count",
            ),
            frozen_hypothesis_count=_require_int(
                value["frozen_hypothesis_count"],
                label="frozen_hypothesis_count",
            ),
            familywise_permutation_count=_require_int(
                value["familywise_permutation_count"],
                label="familywise_permutation_count",
            ),
            profile_abi=str(value["profile_abi"]),
            permutation_abi=str(value["permutation_abi"]),
            policy_sha256=str(value["policy_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class LexicalModeBSource:
    source_kind: str
    model_id: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    score_objective_id: str
    chart_surface_id: str
    protocol_sha256: str
    atom_universe_sha256: str
    response_axis: tuple[AtomResponseAxisEntry, ...]
    B_role: CaptureRole
    exact_lift_receipt: ExactLiftCaptureReceipt
    origin_capture_sha256: str
    published_capture_authenticated: bool = False
    B_source_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "protocol_sha256",
            "atom_universe_sha256",
            "origin_capture_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not self.source_kind or not self.model_id:
            raise ValueError("frozen B source metadata is incomplete")
        if not self.response_axis or any(
            type(item) is not AtomResponseAxisEntry for item in self.response_axis
        ):
            raise TypeError("frozen B response axis is invalid")
        if (
            type(self.B_role) is not CaptureRole
            or self.B_role.role != LEXICAL_MODE_B_CONFIRMATION_ROLE
        ):
            raise TypeError("frozen B source must contain exactly B_confirmation")
        if type(self.exact_lift_receipt) is not ExactLiftCaptureReceipt:
            raise TypeError("frozen B exact-lift receipt is invalid")
        _require_bool(
            self.published_capture_authenticated,
            label="published_capture_authenticated",
        )
        computed = _json_sha256(self._payload(), domain=_SOURCE_DOMAIN)
        if self.published_capture_authenticated and (
            self.origin_capture_sha256
            != lexical_modes.PINNED_PARTS_FIRST_CAPTURE_SHA256
            or self.B_role.role_sha256 != PINNED_LEXICAL_B_ROLE_SHA256
            or self.B_role.row_ids_sha256 != PINNED_LEXICAL_B_ROW_IDS_SHA256
            or computed != PINNED_LEXICAL_B_SOURCE_SHA256
        ):
            raise ValueError("published lexical B source does not match its pins")
        if self.B_source_sha256 == "":
            object.__setattr__(self, "B_source_sha256", computed)
        elif _require_sha256(
            self.B_source_sha256, label="B_source_sha256"
        ) != computed:
            raise ValueError("lexical B source hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
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
            "B_role": self.B_role.to_dict(),
            "exact_lift_receipt": self.exact_lift_receipt.to_dict(),
        }


def build_lexical_mode_B_source(
    captures: PartsFirstCaptureBundle,
    *,
    authenticate_published_capture: bool = False,
) -> LexicalModeBSource:
    if type(captures) is not PartsFirstCaptureBundle:
        raise TypeError("B-source redaction requires an exact capture bundle")
    B_roles = tuple(
        role
        for role in captures.roles
        if role.role == LEXICAL_MODE_B_CONFIRMATION_ROLE
    )
    if len(B_roles) != 1:
        raise ValueError("capture must contain exactly one B confirmation role")
    return LexicalModeBSource(
        source_kind=captures.source_kind,
        model_id=captures.model_id,
        source_model_fingerprint_sha256=(
            captures.source_model_fingerprint_sha256
        ),
        source_execution_fingerprint_sha256=(
            captures.source_execution_fingerprint_sha256
        ),
        score_objective_id=captures.score_objective_id,
        chart_surface_id=captures.chart_surface_id,
        protocol_sha256=captures.protocol_sha256,
        atom_universe_sha256=captures.atom_universe_sha256,
        response_axis=captures.response_axis,
        B_role=B_roles[0],
        exact_lift_receipt=captures.exact_lift_receipt,
        origin_capture_sha256=captures.capture_sha256,
        published_capture_authenticated=authenticate_published_capture,
    )


@dataclass(frozen=True, slots=True)
class FrozenBFamilyBlockPermutationEvidence:
    role_scope: str
    left_column: int
    right_column: int
    family_ids: tuple[str, ...]
    family_block_cross_products: tuple[float, ...]
    observed_signed_cosine: float
    canonical_sign_pattern_cosines: tuple[float, ...]
    familywise_max_pattern_cosines: tuple[float, ...]
    raw_exceedance_count: int
    raw_exact_p_value: float
    familywise_exceedance_count: int
    familywise_exact_p_value: float
    frozen_hypothesis_count: int
    orientation_preserved: bool
    exact_familywise_evidence_passed: bool
    permutation_abi: str = LEXICAL_MODE_B_PERMUTATION_ABI
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role_scope not in {"lexical_ingress", "lexical_readout"}:
            raise ValueError("frozen B pair role is invalid")
        for name in ("left_column", "right_column"):
            value = _require_int(getattr(self, name), label=name)
            if value >= HIDDEN_WIDTH:
                raise ValueError("frozen B pair column is outside hidden width")
        if self.left_column >= self.right_column:
            raise ValueError("frozen B pair columns are not canonical")
        if len(self.family_ids) != 6 or self.family_ids != tuple(
            sorted(set(self.family_ids))
        ):
            raise ValueError("frozen B family inventory is not canonical")
        cross = tuple(
            _finite_float(value, label="family block cross product")
            for value in self.family_block_cross_products
        )
        null = tuple(
            _finite_float(value, label="sign-pattern cosine")
            for value in self.canonical_sign_pattern_cosines
        )
        max_null = tuple(
            _finite_float(value, label="familywise max cosine")
            for value in self.familywise_max_pattern_cosines
        )
        policy = LexicalModeBConfirmationPolicy()
        if (
            len(cross) != 6
            or len(null) != policy.exact_permutation_count
            or len(max_null) != policy.familywise_permutation_count
        ):
            raise ValueError("frozen B exact permutation inventory is incomplete")
        observed = _finite_float(
            self.observed_signed_cosine,
            label="observed_signed_cosine",
        )
        if not -1.000000000001 <= observed <= 1.000000000001:
            raise ValueError("frozen B observed cosine is out of range")
        if any(not -1.000000000001 <= value <= 1.000000000001 for value in null):
            raise ValueError("frozen B null cosine is out of range")
        if any(
            not -1.000000000001 <= value <= 1.000000000001
            for value in max_null
        ):
            raise ValueError("frozen B familywise cosine is out of range")
        raw_count = sum(value >= observed for value in null)
        familywise_count = sum(value >= observed for value in max_null)
        if self.raw_exceedance_count != raw_count or self.raw_exact_p_value != (
            raw_count / float(policy.exact_permutation_count)
        ):
            raise ValueError("frozen B raw exact p-value does not close")
        if (
            self.familywise_exceedance_count != familywise_count
            or self.familywise_exact_p_value
            != familywise_count / float(policy.familywise_permutation_count)
        ):
            raise ValueError("frozen B familywise exact p-value does not close")
        _require_int(
            self.frozen_hypothesis_count,
            label="frozen_hypothesis_count",
            minimum=1,
        )
        if self.frozen_hypothesis_count != policy.frozen_hypothesis_count:
            raise ValueError("frozen B hypothesis count drifted")
        if self.orientation_preserved is not (
            observed >= policy.orientation_preserving_cosine_floor
        ):
            raise ValueError("frozen B orientation decision does not close")
        if self.exact_familywise_evidence_passed is not (
            self.familywise_exact_p_value
            <= policy.maximum_exact_permutation_p_value
        ):
            raise ValueError("frozen B exact-test decision does not close")
        if self.permutation_abi != LEXICAL_MODE_B_PERMUTATION_ABI:
            raise ValueError("frozen B permutation ABI drifted")
        computed = _json_sha256(self._payload(), domain=_PAIR_DOMAIN)
        if self.evidence_sha256 == "":
            object.__setattr__(self, "evidence_sha256", computed)
        elif _require_sha256(
            self.evidence_sha256, label="evidence_sha256"
        ) != computed:
            raise ValueError("frozen B pair evidence hash mismatch")

    @property
    def passed(self) -> bool:
        return self.orientation_preserved and self.exact_familywise_evidence_passed

    def _payload(self) -> dict[str, object]:
        return {
            "role_scope": self.role_scope,
            "left_column": self.left_column,
            "right_column": self.right_column,
            "family_ids": list(self.family_ids),
            "family_block_cross_products": list(
                self.family_block_cross_products
            ),
            "observed_signed_cosine": self.observed_signed_cosine,
            "canonical_sign_pattern_cosines": list(
                self.canonical_sign_pattern_cosines
            ),
            "familywise_max_pattern_cosines": list(
                self.familywise_max_pattern_cosines
            ),
            "raw_exceedance_count": self.raw_exceedance_count,
            "raw_exact_p_value": self.raw_exact_p_value,
            "familywise_exceedance_count": self.familywise_exceedance_count,
            "familywise_exact_p_value": self.familywise_exact_p_value,
            "frozen_hypothesis_count": self.frozen_hypothesis_count,
            "orientation_preserved": self.orientation_preserved,
            "exact_familywise_evidence_passed": (
                self.exact_familywise_evidence_passed
            ),
            "passed": self.passed,
            "permutation_abi": self.permutation_abi,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "evidence_sha256": self.evidence_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "FrozenBFamilyBlockPermutationEvidence":
        fields = {
            "role_scope",
            "left_column",
            "right_column",
            "family_ids",
            "family_block_cross_products",
            "observed_signed_cosine",
            "canonical_sign_pattern_cosines",
            "familywise_max_pattern_cosines",
            "raw_exceedance_count",
            "raw_exact_p_value",
            "familywise_exceedance_count",
            "familywise_exact_p_value",
            "frozen_hypothesis_count",
            "orientation_preserved",
            "exact_familywise_evidence_passed",
            "passed",
            "permutation_abi",
            "evidence_sha256",
        }
        _exact_keys(value, fields, label="frozen B pair evidence")
        arrays = (
            "family_ids",
            "family_block_cross_products",
            "canonical_sign_pattern_cosines",
            "familywise_max_pattern_cosines",
        )
        if any(not isinstance(value[name], list) for name in arrays):
            raise ValueError("frozen B pair arrays are invalid")
        receipt = cls(
            role_scope=str(value["role_scope"]),
            left_column=_require_int(value["left_column"], label="left_column"),
            right_column=_require_int(value["right_column"], label="right_column"),
            family_ids=tuple(str(item) for item in value["family_ids"]),
            family_block_cross_products=tuple(
                _finite_float(item, label="family block cross product")
                for item in value["family_block_cross_products"]
            ),
            observed_signed_cosine=_finite_float(
                value["observed_signed_cosine"], label="observed_signed_cosine"
            ),
            canonical_sign_pattern_cosines=tuple(
                _finite_float(item, label="sign-pattern cosine")
                for item in value["canonical_sign_pattern_cosines"]
            ),
            familywise_max_pattern_cosines=tuple(
                _finite_float(item, label="familywise max cosine")
                for item in value["familywise_max_pattern_cosines"]
            ),
            raw_exceedance_count=_require_int(
                value["raw_exceedance_count"], label="raw_exceedance_count"
            ),
            raw_exact_p_value=_finite_float(
                value["raw_exact_p_value"], label="raw_exact_p_value"
            ),
            familywise_exceedance_count=_require_int(
                value["familywise_exceedance_count"],
                label="familywise_exceedance_count",
            ),
            familywise_exact_p_value=_finite_float(
                value["familywise_exact_p_value"],
                label="familywise_exact_p_value",
            ),
            frozen_hypothesis_count=_require_int(
                value["frozen_hypothesis_count"],
                label="frozen_hypothesis_count",
                minimum=1,
            ),
            orientation_preserved=_require_bool(
                value["orientation_preserved"], label="orientation_preserved"
            ),
            exact_familywise_evidence_passed=_require_bool(
                value["exact_familywise_evidence_passed"],
                label="exact_familywise_evidence_passed",
            ),
            permutation_abi=str(value["permutation_abi"]),
            evidence_sha256=str(value["evidence_sha256"]),
        )
        if value["passed"] is not receipt.passed:
            raise ValueError("frozen B pair pass decision does not close")
        return receipt


@dataclass(frozen=True, slots=True)
class FrozenBRoleQualification:
    role_scope: str
    minimum_signed_cosine: float
    maximum_raw_exact_p_value: float
    maximum_familywise_exact_p_value: float
    family_centered_energy: tuple[tuple[str, float], ...]
    active_family_relative_energy_floor: float
    active_family_count: int
    minimum_active_family_count: int
    pair_evidence: tuple[FrozenBFamilyBlockPermutationEvidence, ...]
    passed: bool
    failure_reasons: tuple[str, ...]
    role_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role_scope not in {"lexical_ingress", "lexical_readout"}:
            raise ValueError("frozen B role qualification scope is invalid")
        if not self.pair_evidence or any(
            not isinstance(item, FrozenBFamilyBlockPermutationEvidence)
            for item in self.pair_evidence
        ):
            raise ValueError("frozen B role pair evidence is empty")
        if any(item.role_scope != self.role_scope for item in self.pair_evidence):
            raise ValueError("frozen B pair evidence crossed role scope")
        minimum = min(item.observed_signed_cosine for item in self.pair_evidence)
        raw = max(item.raw_exact_p_value for item in self.pair_evidence)
        adjusted = max(
            item.familywise_exact_p_value for item in self.pair_evidence
        )
        if (
            self.minimum_signed_cosine != minimum
            or self.maximum_raw_exact_p_value != raw
            or self.maximum_familywise_exact_p_value != adjusted
        ):
            raise ValueError("frozen B role aggregates do not close")
        energy_rows = tuple(
            (str(family), _finite_float(energy, label="family centered energy"))
            for family, energy in self.family_centered_energy
        )
        if (
            len(energy_rows) != 6
            or tuple(family for family, _ in energy_rows)
            != tuple(sorted({family for family, _ in energy_rows}))
            or any(energy < 0 for _, energy in energy_rows)
        ):
            raise ValueError("frozen B family energy inventory is invalid")
        relative_floor = _finite_float(
            self.active_family_relative_energy_floor,
            label="active_family_relative_energy_floor",
        )
        maximum_energy = max(energy for _, energy in energy_rows)
        expected_active = sum(
            energy >= relative_floor * maximum_energy
            for _, energy in energy_rows
        )
        if self.active_family_count != expected_active:
            raise ValueError("frozen B active-family count does not close")
        _require_int(
            self.minimum_active_family_count,
            label="minimum_active_family_count",
            minimum=1,
        )
        reasons: list[str] = []
        policy = LexicalModeBConfirmationPolicy()
        if minimum < policy.orientation_preserving_cosine_floor:
            reasons.append("orientation_preserving_cosine_below_frozen_floor")
        if adjusted > policy.maximum_exact_permutation_p_value:
            reasons.append("familywise_exact_sign_evidence_failed")
        if self.active_family_count < self.minimum_active_family_count:
            reasons.append("active_family_support_below_frozen_minimum")
        if tuple(self.failure_reasons) != tuple(reasons) or self.passed is not (
            not reasons
        ):
            raise ValueError("frozen B role decision does not close")
        computed = _json_sha256(self._payload(), domain=_ROLE_DOMAIN)
        if self.role_sha256 == "":
            object.__setattr__(self, "role_sha256", computed)
        elif _require_sha256(self.role_sha256, label="role_sha256") != computed:
            raise ValueError("frozen B role hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "role_scope": self.role_scope,
            "minimum_signed_cosine": self.minimum_signed_cosine,
            "maximum_raw_exact_p_value": self.maximum_raw_exact_p_value,
            "maximum_familywise_exact_p_value": (
                self.maximum_familywise_exact_p_value
            ),
            "family_centered_energy": [
                [family, energy] for family, energy in self.family_centered_energy
            ],
            "active_family_relative_energy_floor": (
                self.active_family_relative_energy_floor
            ),
            "active_family_count": self.active_family_count,
            "minimum_active_family_count": self.minimum_active_family_count,
            "pair_evidence": [item.to_dict() for item in self.pair_evidence],
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "role_sha256": self.role_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FrozenBRoleQualification":
        _exact_keys(
            value,
            {
                "role_scope",
                "minimum_signed_cosine",
                "maximum_raw_exact_p_value",
                "maximum_familywise_exact_p_value",
                "family_centered_energy",
                "active_family_relative_energy_floor",
                "active_family_count",
                "minimum_active_family_count",
                "pair_evidence",
                "passed",
                "failure_reasons",
                "role_sha256",
            },
            label="frozen B role qualification",
        )
        pairs = value["pair_evidence"]
        energies = value["family_centered_energy"]
        reasons = value["failure_reasons"]
        if (
            not isinstance(pairs, list)
            or not all(isinstance(item, Mapping) for item in pairs)
            or not isinstance(energies, list)
            or not all(
                isinstance(item, list) and len(item) == 2 for item in energies
            )
            or not isinstance(reasons, list)
        ):
            raise ValueError("frozen B role nested fields are invalid")
        return cls(
            role_scope=str(value["role_scope"]),
            minimum_signed_cosine=_finite_float(
                value["minimum_signed_cosine"], label="minimum_signed_cosine"
            ),
            maximum_raw_exact_p_value=_finite_float(
                value["maximum_raw_exact_p_value"],
                label="maximum_raw_exact_p_value",
            ),
            maximum_familywise_exact_p_value=_finite_float(
                value["maximum_familywise_exact_p_value"],
                label="maximum_familywise_exact_p_value",
            ),
            family_centered_energy=tuple(
                (
                    str(item[0]),
                    _finite_float(item[1], label="family centered energy"),
                )
                for item in energies
            ),
            active_family_relative_energy_floor=_finite_float(
                value["active_family_relative_energy_floor"],
                label="active_family_relative_energy_floor",
            ),
            active_family_count=_require_int(
                value["active_family_count"], label="active_family_count"
            ),
            minimum_active_family_count=_require_int(
                value["minimum_active_family_count"],
                label="minimum_active_family_count",
                minimum=1,
            ),
            pair_evidence=tuple(
                FrozenBFamilyBlockPermutationEvidence.from_dict(item)
                for item in pairs
            ),
            passed=_require_bool(value["passed"], label="passed"),
            failure_reasons=tuple(str(item) for item in reasons),
            role_sha256=str(value["role_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class FrozenBCandidateQualification:
    candidate_id: str
    candidate_sha256: str
    member_columns: tuple[int, ...]
    frozen_role_scope: str
    role_qualifications: tuple[FrozenBRoleQualification, ...]
    passed: bool
    failure_reasons: tuple[str, ...]
    membership_refit_performed: bool = False
    threshold_refit_performed: bool = False
    B_reranking_performed: bool = False
    candidate_sha256_B: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("frozen B candidate ID is empty")
        _require_sha256(self.candidate_sha256, label="candidate_sha256")
        members = tuple(self.member_columns)
        if len(members) < 2 or members != tuple(sorted(set(members))):
            raise ValueError("frozen B candidate members are invalid")
        expected_roles = {
            "lexical_ingress": ("lexical_ingress",),
            "lexical_readout": ("lexical_readout",),
            "dual_role": ("lexical_ingress", "lexical_readout"),
        }
        if self.frozen_role_scope not in expected_roles:
            raise ValueError("frozen B candidate role scope is invalid")
        if tuple(item.role_scope for item in self.role_qualifications) != (
            expected_roles[self.frozen_role_scope]
        ):
            raise ValueError("frozen B candidate role inventory drifted")
        reasons = tuple(
            f"{role.role_scope}:{reason}"
            for role in self.role_qualifications
            for reason in role.failure_reasons
        )
        if self.failure_reasons != reasons or self.passed is not (not reasons):
            raise ValueError("frozen B candidate decision does not close")
        for name in (
            "membership_refit_performed",
            "threshold_refit_performed",
            "B_reranking_performed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must remain false")
        computed = _json_sha256(self._payload(), domain=_CANDIDATE_DOMAIN)
        if self.candidate_sha256_B == "":
            object.__setattr__(self, "candidate_sha256_B", computed)
        elif _require_sha256(
            self.candidate_sha256_B, label="candidate_sha256_B"
        ) != computed:
            raise ValueError("frozen B candidate qualification hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "member_columns": list(self.member_columns),
            "frozen_role_scope": self.frozen_role_scope,
            "role_qualifications": [
                item.to_dict() for item in self.role_qualifications
            ],
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "membership_refit_performed": self.membership_refit_performed,
            "threshold_refit_performed": self.threshold_refit_performed,
            "B_reranking_performed": self.B_reranking_performed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "candidate_sha256_B": self.candidate_sha256_B}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "FrozenBCandidateQualification":
        _exact_keys(
            value,
            {
                "candidate_id",
                "candidate_sha256",
                "member_columns",
                "frozen_role_scope",
                "role_qualifications",
                "passed",
                "failure_reasons",
                "membership_refit_performed",
                "threshold_refit_performed",
                "B_reranking_performed",
                "candidate_sha256_B",
            },
            label="frozen B candidate qualification",
        )
        members = value["member_columns"]
        roles = value["role_qualifications"]
        reasons = value["failure_reasons"]
        if (
            not isinstance(members, list)
            or not isinstance(roles, list)
            or not all(isinstance(item, Mapping) for item in roles)
            or not isinstance(reasons, list)
        ):
            raise ValueError("frozen B candidate nested fields are invalid")
        return cls(
            candidate_id=str(value["candidate_id"]),
            candidate_sha256=str(value["candidate_sha256"]),
            member_columns=tuple(
                _require_int(item, label="member column") for item in members
            ),
            frozen_role_scope=str(value["frozen_role_scope"]),
            role_qualifications=tuple(
                FrozenBRoleQualification.from_dict(item) for item in roles
            ),
            passed=_require_bool(value["passed"], label="passed"),
            failure_reasons=tuple(str(item) for item in reasons),
            membership_refit_performed=_require_bool(
                value["membership_refit_performed"],
                label="membership_refit_performed",
            ),
            threshold_refit_performed=_require_bool(
                value["threshold_refit_performed"],
                label="threshold_refit_performed",
            ),
            B_reranking_performed=_require_bool(
                value["B_reranking_performed"], label="B_reranking_performed"
            ),
            candidate_sha256_B=str(value["candidate_sha256_B"]),
        )


@dataclass(frozen=True, slots=True)
class LexicalModeBConfirmationReceipt:
    capture_sha256: str
    dictionary_artifact_sha256: str
    dictionary_A_source_sha256: str
    dictionary_configuration_sha256: str
    dictionary_A_role_sha256: str
    B_source_sha256: str
    B_role_sha256: str
    B_row_ids_sha256: str
    protocol_sha256: str
    atom_universe_sha256: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    exact_control_evidence_sha256: str
    lexical_axis_sha256: str
    frozen_candidate_inventory_sha256: str
    policy: LexicalModeBConfirmationPolicy
    candidate_qualifications: tuple[FrozenBCandidateQualification, ...]
    dictionary_candidate_count: int
    evaluated_candidate_count: int
    orientation_threshold_pass_count: int
    exact_familywise_evidence_pass_count: int
    active_family_support_pass_count: int
    confirmed_candidate_count: int
    all_candidates_rejected: bool
    published_inputs_authenticated: bool
    roles_statistically_read: tuple[str, ...] = (LEXICAL_MODE_B_CONFIRMATION_ROLE,)
    A_rows_read_for_dictionary_provenance_replay: int = 0
    A_rows_read_for_qualification: int = 0
    C_rows_read: int = 0
    membership_refit_performed: bool = False
    threshold_refit_performed: bool = False
    A_candidate_reselection_performed: bool = False
    B_reranking_performed: bool = False
    attention_promotion_performed: bool = False
    attention_promotion_candidate_count: int = 0
    B_was_visible_during_protocol_finalization: bool = True
    clean_prospective_confirmation: bool = False
    attention_promotion_authorized: bool = False
    fresh_untouched_panel_required: bool = True
    confirmation_scope: str = (
        "B_only_frozen_role_signed_response_replication_not_semantics"
    )
    narrow_B_confirmation_completed: bool = True
    semantic_mode_confirmed: bool = False
    attention_chart_structure_claimed: bool = False
    compression_attempted: bool = False
    compression_claimed: bool = False
    parameter_savings_claimed: bool = False
    mac_savings_claimed: bool = False
    latency_claimed: bool = False
    schema: str = LEXICAL_MODE_B_SCHEMA
    format_version: int = LEXICAL_MODE_B_FORMAT_VERSION
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != LEXICAL_MODE_B_SCHEMA or self.format_version != 1:
            raise ValueError("frozen B receipt header is invalid")
        for name in (
            "capture_sha256",
            "dictionary_artifact_sha256",
            "dictionary_A_source_sha256",
            "dictionary_configuration_sha256",
            "dictionary_A_role_sha256",
            "B_source_sha256",
            "B_role_sha256",
            "B_row_ids_sha256",
            "protocol_sha256",
            "atom_universe_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "exact_control_evidence_sha256",
            "lexical_axis_sha256",
            "frozen_candidate_inventory_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.policy, LexicalModeBConfirmationPolicy):
            raise TypeError("frozen B receipt policy is invalid")
        if any(
            not isinstance(item, FrozenBCandidateQualification)
            for item in self.candidate_qualifications
        ):
            raise TypeError("frozen B candidate qualifications are invalid")
        ids = tuple(item.candidate_id for item in self.candidate_qualifications)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("frozen B candidate order is not canonical")
        inventory = _json_sha256(
            [
                {
                    "candidate_id": item.candidate_id,
                    "candidate_sha256": item.candidate_sha256,
                    "member_columns": list(item.member_columns),
                    "frozen_role_scope": item.frozen_role_scope,
                }
                for item in self.candidate_qualifications
            ],
            domain=_INVENTORY_DOMAIN,
        )
        if self.frozen_candidate_inventory_sha256 != inventory:
            raise ValueError("frozen B candidate inventory hash drifted")
        candidate_count = len(self.candidate_qualifications)
        if (
            self.dictionary_candidate_count != candidate_count
            or self.evaluated_candidate_count != candidate_count
        ):
            raise ValueError("frozen B candidate count does not close")
        roles = tuple(
            role
            for candidate in self.candidate_qualifications
            for role in candidate.role_qualifications
        )
        orientation_count = sum(
            role.minimum_signed_cosine
            >= self.policy.orientation_preserving_cosine_floor
            for role in roles
        )
        exact_count = sum(
            role.maximum_familywise_exact_p_value
            <= self.policy.maximum_exact_permutation_p_value
            for role in roles
        )
        active_count = sum(
            role.active_family_count >= role.minimum_active_family_count
            for role in roles
        )
        confirmed = sum(item.passed for item in self.candidate_qualifications)
        if (
            self.orientation_threshold_pass_count != orientation_count
            or self.exact_familywise_evidence_pass_count != exact_count
            or self.active_family_support_pass_count != active_count
            or self.confirmed_candidate_count != confirmed
            or self.all_candidates_rejected is not (confirmed == 0)
        ):
            raise ValueError("frozen B aggregate decisions do not close")
        _require_bool(
            self.published_inputs_authenticated,
            label="published_inputs_authenticated",
        )
        if self.published_inputs_authenticated and (
            self.capture_sha256
            != lexical_modes.PINNED_PARTS_FIRST_CAPTURE_SHA256
            or self.dictionary_artifact_sha256
            != lexical_modes.PINNED_LEXICAL_MODE_ARTIFACT_SHA256
            or self.dictionary_A_source_sha256
            != lexical_modes.PINNED_LEXICAL_A_SOURCE_SHA256
            or self.dictionary_configuration_sha256
            != lexical_modes.PINNED_LEXICAL_MODE_CONFIG_SHA256
            or self.dictionary_A_role_sha256
            != lexical_modes.PINNED_LEXICAL_A_ROLE_SHA256
            or self.B_source_sha256 != PINNED_LEXICAL_B_SOURCE_SHA256
            or self.B_role_sha256 != PINNED_LEXICAL_B_ROLE_SHA256
            or self.B_row_ids_sha256 != PINNED_LEXICAL_B_ROW_IDS_SHA256
        ):
            raise ValueError("published frozen B receipt pins drifted")
        if self.roles_statistically_read != (LEXICAL_MODE_B_CONFIRMATION_ROLE,):
            raise ValueError("frozen B qualifier may statistically read only B")
        for name in (
            "A_rows_read_for_qualification",
            "C_rows_read",
            "attention_promotion_candidate_count",
        ):
            if getattr(self, name) != 0:
                raise ValueError(f"{name} must remain zero")
        if self.A_rows_read_for_dictionary_provenance_replay < 0:
            raise ValueError("A provenance replay row count cannot be negative")
        for name in (
            "membership_refit_performed",
            "threshold_refit_performed",
            "A_candidate_reselection_performed",
            "B_reranking_performed",
            "attention_promotion_performed",
            "clean_prospective_confirmation",
            "attention_promotion_authorized",
            "semantic_mode_confirmed",
            "attention_chart_structure_claimed",
            "compression_attempted",
            "compression_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must remain false")
        if (
            self.B_was_visible_during_protocol_finalization is not True
            or self.fresh_untouched_panel_required is not True
        ):
            raise ValueError("frozen B retrospective status drifted")
        if self.confirmation_scope != (
            "B_only_frozen_role_signed_response_replication_not_semantics"
        ):
            raise ValueError("frozen B confirmation scope drifted")
        if self.narrow_B_confirmation_completed is not True:
            raise ValueError("frozen B confirmation completion flag drifted")
        computed = _json_sha256(self._payload(), domain=_RECEIPT_DOMAIN)
        if self.receipt_sha256 == "":
            object.__setattr__(self, "receipt_sha256", computed)
        elif _require_sha256(
            self.receipt_sha256, label="receipt_sha256"
        ) != computed:
            raise ValueError("frozen B receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "capture_sha256": self.capture_sha256,
            "dictionary_artifact_sha256": self.dictionary_artifact_sha256,
            "dictionary_A_source_sha256": self.dictionary_A_source_sha256,
            "dictionary_configuration_sha256": (
                self.dictionary_configuration_sha256
            ),
            "dictionary_A_role_sha256": self.dictionary_A_role_sha256,
            "B_source_sha256": self.B_source_sha256,
            "B_role_sha256": self.B_role_sha256,
            "B_row_ids_sha256": self.B_row_ids_sha256,
            "protocol_sha256": self.protocol_sha256,
            "atom_universe_sha256": self.atom_universe_sha256,
            "source_model_fingerprint_sha256": (
                self.source_model_fingerprint_sha256
            ),
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "exact_control_evidence_sha256": (
                self.exact_control_evidence_sha256
            ),
            "lexical_axis_sha256": self.lexical_axis_sha256,
            "frozen_candidate_inventory_sha256": (
                self.frozen_candidate_inventory_sha256
            ),
            "policy": self.policy.to_dict(),
            "candidate_qualifications": [
                item.to_dict() for item in self.candidate_qualifications
            ],
            "dictionary_candidate_count": self.dictionary_candidate_count,
            "evaluated_candidate_count": self.evaluated_candidate_count,
            "orientation_threshold_pass_count": (
                self.orientation_threshold_pass_count
            ),
            "exact_familywise_evidence_pass_count": (
                self.exact_familywise_evidence_pass_count
            ),
            "active_family_support_pass_count": (
                self.active_family_support_pass_count
            ),
            "confirmed_candidate_count": self.confirmed_candidate_count,
            "all_candidates_rejected": self.all_candidates_rejected,
            "published_inputs_authenticated": self.published_inputs_authenticated,
            "roles_statistically_read": list(self.roles_statistically_read),
            "A_rows_read_for_dictionary_provenance_replay": (
                self.A_rows_read_for_dictionary_provenance_replay
            ),
            "A_rows_read_for_qualification": self.A_rows_read_for_qualification,
            "C_rows_read": self.C_rows_read,
            "membership_refit_performed": self.membership_refit_performed,
            "threshold_refit_performed": self.threshold_refit_performed,
            "A_candidate_reselection_performed": (
                self.A_candidate_reselection_performed
            ),
            "B_reranking_performed": self.B_reranking_performed,
            "attention_promotion_performed": self.attention_promotion_performed,
            "attention_promotion_candidate_count": (
                self.attention_promotion_candidate_count
            ),
            "B_was_visible_during_protocol_finalization": (
                self.B_was_visible_during_protocol_finalization
            ),
            "clean_prospective_confirmation": (
                self.clean_prospective_confirmation
            ),
            "attention_promotion_authorized": (
                self.attention_promotion_authorized
            ),
            "fresh_untouched_panel_required": self.fresh_untouched_panel_required,
            "confirmation_scope": self.confirmation_scope,
            "narrow_B_confirmation_completed": (
                self.narrow_B_confirmation_completed
            ),
            "semantic_mode_confirmed": self.semantic_mode_confirmed,
            "attention_chart_structure_claimed": (
                self.attention_chart_structure_claimed
            ),
            "compression_attempted": self.compression_attempted,
            "compression_claimed": self.compression_claimed,
            "parameter_savings_claimed": self.parameter_savings_claimed,
            "mac_savings_claimed": self.mac_savings_claimed,
            "latency_claimed": self.latency_claimed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object]
    ) -> "LexicalModeBConfirmationReceipt":
        fields = {
            "schema",
            "format_version",
            "capture_sha256",
            "dictionary_artifact_sha256",
            "dictionary_A_source_sha256",
            "dictionary_configuration_sha256",
            "dictionary_A_role_sha256",
            "B_source_sha256",
            "B_role_sha256",
            "B_row_ids_sha256",
            "protocol_sha256",
            "atom_universe_sha256",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "exact_control_evidence_sha256",
            "lexical_axis_sha256",
            "frozen_candidate_inventory_sha256",
            "policy",
            "candidate_qualifications",
            "dictionary_candidate_count",
            "evaluated_candidate_count",
            "orientation_threshold_pass_count",
            "exact_familywise_evidence_pass_count",
            "active_family_support_pass_count",
            "confirmed_candidate_count",
            "all_candidates_rejected",
            "published_inputs_authenticated",
            "roles_statistically_read",
            "A_rows_read_for_dictionary_provenance_replay",
            "A_rows_read_for_qualification",
            "C_rows_read",
            "membership_refit_performed",
            "threshold_refit_performed",
            "A_candidate_reselection_performed",
            "B_reranking_performed",
            "attention_promotion_performed",
            "attention_promotion_candidate_count",
            "B_was_visible_during_protocol_finalization",
            "clean_prospective_confirmation",
            "attention_promotion_authorized",
            "fresh_untouched_panel_required",
            "confirmation_scope",
            "narrow_B_confirmation_completed",
            "semantic_mode_confirmed",
            "attention_chart_structure_claimed",
            "compression_attempted",
            "compression_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
            "receipt_sha256",
        }
        _exact_keys(value, fields, label="frozen B confirmation receipt")
        policy = value["policy"]
        candidates = value["candidate_qualifications"]
        roles = value["roles_statistically_read"]
        if (
            not isinstance(policy, Mapping)
            or not isinstance(candidates, list)
            or not all(isinstance(item, Mapping) for item in candidates)
            or not isinstance(roles, list)
        ):
            raise ValueError("frozen B receipt nested fields are invalid")
        bool_fields = (
            "all_candidates_rejected",
            "published_inputs_authenticated",
            "membership_refit_performed",
            "threshold_refit_performed",
            "A_candidate_reselection_performed",
            "B_reranking_performed",
            "attention_promotion_performed",
            "B_was_visible_during_protocol_finalization",
            "clean_prospective_confirmation",
            "attention_promotion_authorized",
            "fresh_untouched_panel_required",
            "narrow_B_confirmation_completed",
            "semantic_mode_confirmed",
            "attention_chart_structure_claimed",
            "compression_attempted",
            "compression_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
        )
        kwargs: dict[str, object] = {
            name: _require_bool(value[name], label=name) for name in bool_fields
        }
        return cls(
            capture_sha256=str(value["capture_sha256"]),
            dictionary_artifact_sha256=str(value["dictionary_artifact_sha256"]),
            dictionary_A_source_sha256=str(value["dictionary_A_source_sha256"]),
            dictionary_configuration_sha256=str(
                value["dictionary_configuration_sha256"]
            ),
            dictionary_A_role_sha256=str(value["dictionary_A_role_sha256"]),
            B_source_sha256=str(value["B_source_sha256"]),
            B_role_sha256=str(value["B_role_sha256"]),
            B_row_ids_sha256=str(value["B_row_ids_sha256"]),
            protocol_sha256=str(value["protocol_sha256"]),
            atom_universe_sha256=str(value["atom_universe_sha256"]),
            source_model_fingerprint_sha256=str(
                value["source_model_fingerprint_sha256"]
            ),
            source_execution_fingerprint_sha256=str(
                value["source_execution_fingerprint_sha256"]
            ),
            exact_control_evidence_sha256=str(
                value["exact_control_evidence_sha256"]
            ),
            lexical_axis_sha256=str(value["lexical_axis_sha256"]),
            frozen_candidate_inventory_sha256=str(
                value["frozen_candidate_inventory_sha256"]
            ),
            policy=LexicalModeBConfirmationPolicy.from_dict(policy),
            candidate_qualifications=tuple(
                FrozenBCandidateQualification.from_dict(item)
                for item in candidates
            ),
            dictionary_candidate_count=_require_int(
                value["dictionary_candidate_count"],
                label="dictionary_candidate_count",
            ),
            evaluated_candidate_count=_require_int(
                value["evaluated_candidate_count"],
                label="evaluated_candidate_count",
            ),
            orientation_threshold_pass_count=_require_int(
                value["orientation_threshold_pass_count"],
                label="orientation_threshold_pass_count",
            ),
            exact_familywise_evidence_pass_count=_require_int(
                value["exact_familywise_evidence_pass_count"],
                label="exact_familywise_evidence_pass_count",
            ),
            active_family_support_pass_count=_require_int(
                value["active_family_support_pass_count"],
                label="active_family_support_pass_count",
            ),
            confirmed_candidate_count=_require_int(
                value["confirmed_candidate_count"],
                label="confirmed_candidate_count",
            ),
            roles_statistically_read=tuple(str(item) for item in roles),
            A_rows_read_for_dictionary_provenance_replay=_require_int(
                value["A_rows_read_for_dictionary_provenance_replay"],
                label="A_rows_read_for_dictionary_provenance_replay",
            ),
            A_rows_read_for_qualification=_require_int(
                value["A_rows_read_for_qualification"],
                label="A_rows_read_for_qualification",
            ),
            C_rows_read=_require_int(value["C_rows_read"], label="C_rows_read"),
            attention_promotion_candidate_count=_require_int(
                value["attention_promotion_candidate_count"],
                label="attention_promotion_candidate_count",
            ),
            confirmation_scope=str(value["confirmation_scope"]),
            schema=str(value["schema"]),
            format_version=_require_int(
                value["format_version"], label="format_version"
            ),
            receipt_sha256=str(value["receipt_sha256"]),
            **kwargs,  # type: ignore[arg-type]
        )


def _lexical_axis_indices(
    source: LexicalModeBSource,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    coordinates: dict[tuple[int, int], int] = {}
    for index, entry in enumerate(source.response_axis):
        coordinate = (entry.global_atom_ordinal, entry.execution_ordinal)
        if coordinate in coordinates:
            raise ValueError("frozen B response axis repeats an occurrence")
        coordinates[coordinate] = index
    try:
        ingress = tuple(coordinates[(column, 0)] for column in range(HIDDEN_WIDTH))
        readout = tuple(
            coordinates[(column, 38)] for column in range(HIDDEN_WIDTH)
        )
    except KeyError as error:
        raise ValueError(
            "frozen B source lacks all 640 lexical columns at both roles"
        ) from error
    return ingress, readout


def _center_B_role(
    raw: np.ndarray,
    family_labels: np.ndarray,
    family_ids: tuple[str, ...],
) -> np.ndarray:
    centered = np.array(raw, dtype=np.float64, copy=True)
    for family in family_ids:
        indices = np.flatnonzero(family_labels == family)
        if len(indices) != 2:
            raise ValueError("frozen B requires exactly two rows per family")
        values = centered[indices]
        values -= values.mean(axis=0, keepdims=True)
        # sqrt(n_f - 1) is exactly one for the frozen B panel, retained here
        # to bind the profile ABI to the A discovery convention.
        values /= math.sqrt(len(indices) - 1)
        centered[indices] = values
    return centered


def _pair_raw_evidence(
    matrix: np.ndarray,
    family_labels: np.ndarray,
    family_ids: tuple[str, ...],
    left: int,
    right: int,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    left_values = matrix[:, left]
    right_values = matrix[:, right]
    denominator = float(
        math.sqrt(
            float(np.dot(left_values, left_values))
            * float(np.dot(right_values, right_values))
        )
    )
    cross = tuple(
        float(
            np.dot(
                left_values[np.flatnonzero(family_labels == family)],
                right_values[np.flatnonzero(family_labels == family)],
            )
        )
        for family in family_ids
    )
    if denominator <= 1e-30:
        observed = 0.0
        null = (0.0,) * 64
    else:
        observed = float(math.fsum(cross) / denominator)
        null = tuple(
            float(
                math.fsum(
                    sign * value for sign, value in zip(pattern, cross)
                )
                / denominator
            )
            for pattern in itertools.product((-1.0, 1.0), repeat=6)
        )
    return observed, cross, null


def _role_scopes(candidate: lexical_modes.LexicalModeCandidate) -> tuple[str, ...]:
    if candidate.role_scope == "dual_role":
        return ("lexical_ingress", "lexical_readout")
    return (candidate.role_scope,)


def _frozen_inventory_sha256(
    candidates: Sequence[lexical_modes.LexicalModeCandidate],
) -> str:
    return _json_sha256(
        [
            {
                "candidate_id": item.candidate_id,
                "candidate_sha256": item.candidate_sha256,
                "member_columns": list(item.member_columns),
                "frozen_role_scope": item.role_scope,
            }
            for item in candidates
        ],
        domain=_INVENTORY_DOMAIN,
    )


def compile_frozen_B_lexical_mode_confirmation(
    dictionary: lexical_modes.LexicalModeDictionary,
    source: LexicalModeBSource,
) -> LexicalModeBConfirmationReceipt:
    """Evaluate every frozen A candidate on B without selection or refit."""

    if type(dictionary) is not lexical_modes.LexicalModeDictionary:
        raise TypeError("frozen B qualifier requires an exact lexical dictionary")
    if type(source) is not LexicalModeBSource:
        raise TypeError("frozen B qualifier requires an exact redacted B source")
    policy = LexicalModeBConfirmationPolicy()
    if dictionary.configuration.role_stability_floor != (
        policy.orientation_preserving_cosine_floor
    ):
        raise ValueError("frozen B cosine floor diverged from A configuration")
    if (
        dictionary.protocol_sha256 != source.protocol_sha256
        or dictionary.atom_universe_sha256 != source.atom_universe_sha256
        or dictionary.source_model_fingerprint_sha256
        != source.source_model_fingerprint_sha256
        or dictionary.source_execution_fingerprint_sha256
        != source.source_execution_fingerprint_sha256
        or dictionary.exact_control_evidence_sha256
        != source.exact_lift_receipt.exact_control_evidence_sha256
    ):
        raise ValueError("frozen B source and A dictionary bindings diverged")
    ingress_indices, readout_indices = _lexical_axis_indices(source)
    rows = tuple(sorted(source.B_role.rows, key=lambda row: row.row_id))
    family_ids = tuple(sorted({row.family_id for row in rows}))
    if len(family_ids) != policy.family_count:
        raise ValueError("frozen B requires the six-family panel")
    family_labels = np.asarray([row.family_id for row in rows], dtype=object)
    if any(
        int(np.sum(family_labels == family)) != policy.rows_per_family
        for family in family_ids
    ):
        raise ValueError("frozen B family row counts drifted")
    responses = np.asarray([row.atom_responses for row in rows], dtype=np.float64)
    if not bool(np.isfinite(responses).all()):
        raise ValueError("frozen B responses must be finite")
    matrices = {
        "lexical_ingress": _center_B_role(
            responses[:, ingress_indices], family_labels, family_ids
        ),
        "lexical_readout": _center_B_role(
            responses[:, readout_indices], family_labels, family_ids
        ),
    }

    hypothesis_rows: list[
        tuple[int, str, int, int, float, tuple[float, ...], tuple[float, ...]]
    ] = []
    for candidate_index, candidate in enumerate(dictionary.candidates):
        for role_scope in _role_scopes(candidate):
            matrix = matrices[role_scope]
            for left, right in itertools.combinations(candidate.member_columns, 2):
                observed, cross, null = _pair_raw_evidence(
                    matrix,
                    family_labels,
                    family_ids,
                    left,
                    right,
                )
                hypothesis_rows.append(
                    (
                        candidate_index,
                        role_scope,
                        left,
                        right,
                        observed,
                        cross,
                        null,
                    )
                )
    if not hypothesis_rows:
        raise ValueError("frozen A dictionary contains no B-testable candidates")
    if len(hypothesis_rows) != policy.frozen_hypothesis_count:
        raise ValueError(
            "frozen B familywise test requires exactly two A hypotheses"
        )
    seen_columns: set[int] = set()
    for hypothesis in hypothesis_rows:
        pair = {hypothesis[2], hypothesis[3]}
        if seen_columns & pair:
            raise ValueError(
                "independent frozen B maxT requires disjoint column pairs"
            )
        seen_columns.update(pair)
    # Each disjoint hypothesis receives its own exact six-family swap action.
    # The Cartesian product is therefore 64 x 64, not a coordinated 64-state
    # sign sensitivity.  Taking max T on every product state controls the two
    # frozen hypotheses without using B to choose a correction.
    familywise_max = tuple(
        max(values)
        for values in itertools.product(
            *(row[6] for row in hypothesis_rows)
        )
    )
    pair_by_key: dict[
        tuple[int, str], list[FrozenBFamilyBlockPermutationEvidence]
    ] = {}
    for (
        candidate_index,
        role_scope,
        left,
        right,
        observed,
        cross,
        null,
    ) in hypothesis_rows:
        raw_count = sum(value >= observed for value in null)
        familywise_count = sum(value >= observed for value in familywise_max)
        evidence = FrozenBFamilyBlockPermutationEvidence(
            role_scope=role_scope,
            left_column=left,
            right_column=right,
            family_ids=family_ids,
            family_block_cross_products=cross,
            observed_signed_cosine=observed,
            canonical_sign_pattern_cosines=null,
            familywise_max_pattern_cosines=familywise_max,
            raw_exceedance_count=raw_count,
            raw_exact_p_value=(
                raw_count / float(policy.exact_permutation_count)
            ),
            familywise_exceedance_count=familywise_count,
            familywise_exact_p_value=(
                familywise_count / float(policy.familywise_permutation_count)
            ),
            frozen_hypothesis_count=len(hypothesis_rows),
            orientation_preserved=(
                observed >= policy.orientation_preserving_cosine_floor
            ),
            exact_familywise_evidence_passed=(
                familywise_count / float(policy.familywise_permutation_count)
                <= policy.maximum_exact_permutation_p_value
            ),
        )
        pair_by_key.setdefault((candidate_index, role_scope), []).append(evidence)

    candidate_qualifications: list[FrozenBCandidateQualification] = []
    for candidate_index, candidate in enumerate(dictionary.candidates):
        role_qualifications: list[FrozenBRoleQualification] = []
        for role_scope in _role_scopes(candidate):
            pairs = tuple(pair_by_key[(candidate_index, role_scope)])
            matrix = matrices[role_scope]
            energy_rows = tuple(
                (
                    family,
                    float(
                        np.sum(
                            np.square(
                                matrices["lexical_ingress"][np.ix_(
                                    np.flatnonzero(family_labels == family),
                                    candidate.member_columns,
                                )]
                            )
                        )
                        + np.sum(
                            np.square(
                                matrices["lexical_readout"][np.ix_(
                                    np.flatnonzero(family_labels == family),
                                    candidate.member_columns,
                                )]
                            )
                        )
                    ),
                )
                for family in family_ids
            )
            maximum_energy = max(value for _, value in energy_rows)
            active = sum(
                value
                >= dictionary.configuration.active_family_relative_energy_floor
                * maximum_energy
                for _, value in energy_rows
            )
            minimum_cosine = min(item.observed_signed_cosine for item in pairs)
            maximum_raw_p = max(item.raw_exact_p_value for item in pairs)
            maximum_adjusted_p = max(
                item.familywise_exact_p_value for item in pairs
            )
            reasons: list[str] = []
            if minimum_cosine < policy.orientation_preserving_cosine_floor:
                reasons.append("orientation_preserving_cosine_below_frozen_floor")
            if maximum_adjusted_p > policy.maximum_exact_permutation_p_value:
                reasons.append("familywise_exact_sign_evidence_failed")
            if active < dictionary.configuration.minimum_active_family_count:
                reasons.append("active_family_support_below_frozen_minimum")
            role_qualifications.append(
                FrozenBRoleQualification(
                    role_scope=role_scope,
                    minimum_signed_cosine=minimum_cosine,
                    maximum_raw_exact_p_value=maximum_raw_p,
                    maximum_familywise_exact_p_value=maximum_adjusted_p,
                    family_centered_energy=energy_rows,
                    active_family_relative_energy_floor=(
                        dictionary.configuration.active_family_relative_energy_floor
                    ),
                    active_family_count=active,
                    minimum_active_family_count=(
                        dictionary.configuration.minimum_active_family_count
                    ),
                    pair_evidence=pairs,
                    passed=not reasons,
                    failure_reasons=tuple(reasons),
                )
            )
        candidate_reasons = tuple(
            f"{role.role_scope}:{reason}"
            for role in role_qualifications
            for reason in role.failure_reasons
        )
        candidate_qualifications.append(
            FrozenBCandidateQualification(
                candidate_id=candidate.candidate_id,
                candidate_sha256=candidate.candidate_sha256,
                member_columns=candidate.member_columns,
                frozen_role_scope=candidate.role_scope,
                role_qualifications=tuple(role_qualifications),
                passed=not candidate_reasons,
                failure_reasons=candidate_reasons,
            )
        )
    qualifications = tuple(candidate_qualifications)
    role_receipts = tuple(
        role for item in qualifications for role in item.role_qualifications
    )
    confirmed = sum(item.passed for item in qualifications)
    return LexicalModeBConfirmationReceipt(
        capture_sha256=source.origin_capture_sha256,
        dictionary_artifact_sha256=dictionary.artifact_sha256,
        dictionary_A_source_sha256=dictionary.source_envelope_sha256,
        dictionary_configuration_sha256=dictionary.configuration.config_sha256,
        dictionary_A_role_sha256=dictionary.A_role_sha256,
        B_source_sha256=source.B_source_sha256,
        B_role_sha256=source.B_role.role_sha256,
        B_row_ids_sha256=source.B_role.row_ids_sha256,
        protocol_sha256=source.protocol_sha256,
        atom_universe_sha256=source.atom_universe_sha256,
        source_model_fingerprint_sha256=source.source_model_fingerprint_sha256,
        source_execution_fingerprint_sha256=(
            source.source_execution_fingerprint_sha256
        ),
        exact_control_evidence_sha256=(
            source.exact_lift_receipt.exact_control_evidence_sha256
        ),
        lexical_axis_sha256=dictionary.lexical_axis_sha256,
        frozen_candidate_inventory_sha256=_frozen_inventory_sha256(
            dictionary.candidates
        ),
        policy=policy,
        candidate_qualifications=qualifications,
        dictionary_candidate_count=len(dictionary.candidates),
        evaluated_candidate_count=len(qualifications),
        orientation_threshold_pass_count=sum(
            role.minimum_signed_cosine
            >= policy.orientation_preserving_cosine_floor
            for role in role_receipts
        ),
        exact_familywise_evidence_pass_count=sum(
            role.maximum_familywise_exact_p_value
            <= policy.maximum_exact_permutation_p_value
            for role in role_receipts
        ),
        active_family_support_pass_count=sum(
            role.active_family_count >= role.minimum_active_family_count
            for role in role_receipts
        ),
        confirmed_candidate_count=confirmed,
        all_candidates_rejected=(confirmed == 0),
        published_inputs_authenticated=bool(
            dictionary.published_A_anchor_authenticated
            and source.published_capture_authenticated
        ),
        A_rows_read_for_dictionary_provenance_replay=(
            dictionary.profile_receipt.row_count
        ),
    )


def _require_published_dictionary(
    dictionary: lexical_modes.LexicalModeDictionary,
) -> lexical_modes.LexicalModeDictionary:
    if dictionary.published_A_anchor_authenticated is not True:
        raise ValueError("frozen B requires the authenticated A dictionary")
    if (
        dictionary.artifact_sha256
        != lexical_modes.PINNED_LEXICAL_MODE_ARTIFACT_SHA256
        or dictionary.configuration.config_sha256
        != lexical_modes.PINNED_LEXICAL_MODE_CONFIG_SHA256
        or dictionary.source_envelope_sha256
        != lexical_modes.PINNED_LEXICAL_A_SOURCE_SHA256
    ):
        raise ValueError("published A dictionary pins drifted before frozen B")
    return dictionary


def _load_published_inputs(
    *,
    capture: Path | str,
    dictionary: Path | str,
) -> tuple[lexical_modes.LexicalModeDictionary, LexicalModeBSource]:
    capture_path = _require_json_path(capture, label="frozen B capture")
    dictionary_path = _require_json_path(dictionary, label="frozen A dictionary")
    if capture_path.resolve(strict=False) == dictionary_path.resolve(strict=False):
        raise ValueError("frozen B capture and dictionary paths must differ")
    captures = lexical_modes.load_lexical_mode_capture_strict(capture_path)
    A_source = lexical_modes.build_lexical_mode_A_source(
        captures,
        authenticate_published_A=True,
    )
    frozen_dictionary = lexical_modes.load_lexical_mode_dictionary(
        dictionary_path,
        A_source=A_source,
    )
    _require_published_dictionary(frozen_dictionary)
    B_source = build_lexical_mode_B_source(
        captures,
        authenticate_published_capture=True,
    )
    return frozen_dictionary, B_source


def compile_published_frozen_B_lexical_mode_confirmation(
    *,
    capture: Path | str = lexical_modes.DEFAULT_LEXICAL_MODE_CAPTURE,
    dictionary: Path | str = lexical_modes.DEFAULT_LEXICAL_MODE_OUTPUT,
) -> LexicalModeBConfirmationReceipt:
    frozen_dictionary, B_source = _load_published_inputs(
        capture=capture,
        dictionary=dictionary,
    )
    receipt = compile_frozen_B_lexical_mode_confirmation(
        frozen_dictionary,
        B_source,
    )
    if receipt.published_inputs_authenticated is not True:
        raise ValueError("frozen B published input authentication did not close")
    return _require_published_receipt(receipt)


def _require_published_receipt(
    receipt: LexicalModeBConfirmationReceipt,
) -> LexicalModeBConfirmationReceipt:
    if receipt.published_inputs_authenticated is not True:
        raise ValueError("published frozen B receipt is unauthenticated")
    if receipt.receipt_sha256 != PINNED_LEXICAL_B_CONFIRMATION_RECEIPT_SHA256:
        raise ValueError("published frozen B confirmation receipt pin drifted")
    return receipt


def validate_lexical_mode_B_confirmation(
    receipt: LexicalModeBConfirmationReceipt,
    *,
    dictionary: lexical_modes.LexicalModeDictionary,
    B_source: LexicalModeBSource,
) -> LexicalModeBConfirmationReceipt:
    if type(receipt) is not LexicalModeBConfirmationReceipt:
        raise TypeError("frozen B confirmation must be an exact typed receipt")
    replay = compile_frozen_B_lexical_mode_confirmation(dictionary, B_source)
    if replay.to_dict() != receipt.to_dict():
        raise ValueError("frozen B confirmation does not replay from its inputs")
    return receipt


def parse_lexical_mode_B_confirmation_json(
    source: str | bytes,
    *,
    dictionary: lexical_modes.LexicalModeDictionary,
    B_source: LexicalModeBSource,
) -> LexicalModeBConfirmationReceipt:
    value = _strict_json_object(source, label="frozen B confirmation")
    receipt = LexicalModeBConfirmationReceipt.from_dict(value)
    return validate_lexical_mode_B_confirmation(
        receipt,
        dictionary=dictionary,
        B_source=B_source,
    )


def _load_receipt_path(
    source: Path,
    *,
    dictionary: lexical_modes.LexicalModeDictionary,
    B_source: LexicalModeBSource,
) -> LexicalModeBConfirmationReceipt:
    try:
        encoded = source.read_bytes()
    except OSError as error:
        raise ValueError("frozen B confirmation is unavailable") from error
    return parse_lexical_mode_B_confirmation_json(
        encoded,
        dictionary=dictionary,
        B_source=B_source,
    )


def load_lexical_mode_B_confirmation(
    source: Path | str = DEFAULT_LEXICAL_MODE_B_OUTPUT,
    *,
    capture: Path | str = lexical_modes.DEFAULT_LEXICAL_MODE_CAPTURE,
    dictionary: Path | str = lexical_modes.DEFAULT_LEXICAL_MODE_OUTPUT,
) -> LexicalModeBConfirmationReceipt:
    path = _require_json_path(source, label="frozen B confirmation")
    frozen_dictionary, B_source = _load_published_inputs(
        capture=capture,
        dictionary=dictionary,
    )
    return _require_published_receipt(
        _load_receipt_path(
            path,
            dictionary=frozen_dictionary,
            B_source=B_source,
        )
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_paths(
    *,
    capture: Path | str,
    dictionary: Path | str,
    output: Path | str,
) -> tuple[Path, Path, Path]:
    capture_path = _require_json_path(capture, label="frozen B capture")
    dictionary_path = _require_json_path(dictionary, label="frozen A dictionary")
    output_path = _require_json_path(output, label="frozen B output")
    resolved = (
        capture_path.resolve(strict=False),
        dictionary_path.resolve(strict=False),
        output_path.resolve(strict=False),
    )
    if len(set(resolved)) != 3:
        raise ValueError("frozen B capture, dictionary, and output paths must differ")
    return capture_path, dictionary_path, output_path


def write_lexical_mode_B_confirmation(
    receipt: LexicalModeBConfirmationReceipt,
    destination: Path | str = DEFAULT_LEXICAL_MODE_B_OUTPUT,
    *,
    capture: Path | str = lexical_modes.DEFAULT_LEXICAL_MODE_CAPTURE,
    dictionary: Path | str = lexical_modes.DEFAULT_LEXICAL_MODE_OUTPUT,
) -> LexicalModeBConfirmationReceipt:
    if type(receipt) is not LexicalModeBConfirmationReceipt:
        raise TypeError("frozen B writer requires an exact typed receipt")
    capture_path, dictionary_path, target = _validate_paths(
        capture=capture,
        dictionary=dictionary,
        output=destination,
    )
    frozen_dictionary, B_source = _load_published_inputs(
        capture=capture_path,
        dictionary=dictionary_path,
    )
    validate_lexical_mode_B_confirmation(
        receipt,
        dictionary=frozen_dictionary,
        B_source=B_source,
    )
    _require_published_receipt(receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    temporary_exists = True
    temporary_stat: os.stat_result | None = None
    primary: BaseException | None = None
    try:
        temporary_stat = os.fstat(descriptor)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                receipt.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not os.path.samestat(
                temporary_stat,
                os.fstat(handle.fileno()),
            ):
                raise RuntimeError(
                    "frozen B confirmation descriptor identity drifted"
                )
        loaded = _load_receipt_path(
            temporary,
            dictionary=frozen_dictionary,
            B_source=B_source,
        )
        if loaded != receipt:
            raise RuntimeError("frozen B receipt changed on strict temp reload")
        current_temporary = temporary.lstat()
        if temporary_stat is None or not os.path.samestat(
            temporary_stat,
            current_temporary,
        ):
            raise RuntimeError("frozen B confirmation temporary identity drifted")
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite a frozen B confirmation"
            ) from error
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 2
        ):
            raise RuntimeError("frozen B hardlink identity drifted")
        current_temporary = temporary.lstat()
        if not os.path.samestat(temporary_stat, current_temporary):
            raise RuntimeError("frozen B temporary changed before unlink")
        temporary.unlink()
        temporary_exists = False
        _fsync_directory(target.parent)
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 1
        ):
            raise RuntimeError("frozen B publication identity drifted")
        canonical = _load_receipt_path(
            target,
            dictionary=frozen_dictionary,
            B_source=B_source,
        )
        if canonical != receipt:
            raise RuntimeError("frozen B receipt changed on canonical reload")
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 1
        ):
            raise RuntimeError("frozen B receipt changed after canonical reload")
        return loaded
    except BaseException as error:
        primary = error
        raise
    finally:
        if temporary_exists:
            try:
                current_temporary = temporary.lstat()
                if temporary_stat is not None and os.path.samestat(
                    temporary_stat,
                    current_temporary,
                ):
                    temporary.unlink()
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "frozen B temporary cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--write", action="store_true")
    parser.add_argument(
        "--capture",
        type=Path,
        default=lexical_modes.DEFAULT_LEXICAL_MODE_CAPTURE,
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=lexical_modes.DEFAULT_LEXICAL_MODE_OUTPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_LEXICAL_MODE_B_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    capture, dictionary, output = _validate_paths(
        capture=args.capture,
        dictionary=args.dictionary,
        output=args.output,
    )
    receipt = compile_published_frozen_B_lexical_mode_confirmation(
        capture=capture,
        dictionary=dictionary,
    )
    if args.write:
        write_lexical_mode_B_confirmation(
            receipt,
            output,
            capture=capture,
            dictionary=dictionary,
        )
    print(
        json.dumps(
            {
                "all_candidates_rejected": receipt.all_candidates_rejected,
                "candidate_count": receipt.evaluated_candidate_count,
                "confirmed_candidate_count": receipt.confirmed_candidate_count,
                "output": str(output),
                "published": bool(args.write),
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
