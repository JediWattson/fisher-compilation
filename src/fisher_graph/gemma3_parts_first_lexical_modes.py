"""A-only Fisher discovery of tied lexical-column mode candidates.

The 640 tied hidden columns remain the exact physical generator basis.  This
module discovers response-similar *logical* coalitions on the frozen
``A_discovery`` rows; it neither merges parameters nor reads B or C.  Every
candidate is accompanied by unit-coefficient subset/complement partitions at
both lexical occurrences so its additive claim can be replayed exactly.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Final

import numpy as np

from .gemma3_parts_first_supermode_experiment import (
    AtomResponseAxisEntry,
    CaptureRole,
    ExactLiftCaptureReceipt,
    PartsFirstCaptureBundle,
)
from .gemma3_parts_first_supermodes import (
    HIDDEN_WIDTH,
    AtomGroupDraft,
    ExecutionBoundary,
    PhysicalAtomFamily,
    PhysicalAtomId,
    build_atom_group_sidecar,
    build_gemma3_parts_catalog,
    validate_exact_atom_partition,
)


__all__ = [
    "LEXICAL_MODE_DISCOVERY_ROLE",
    "LEXICAL_MODE_FORMAT_VERSION",
    "LEXICAL_MODE_GENERATOR_ABI",
    "LEXICAL_MODE_NULL_ABI",
    "LEXICAL_MODE_PROFILE_ABI",
    "LEXICAL_MODE_SCHEMA",
    "DEFAULT_LEXICAL_MODE_CAPTURE",
    "DEFAULT_LEXICAL_MODE_OUTPUT",
    "PINNED_LEXICAL_A_ROLE_SHA256",
    "PINNED_LEXICAL_A_ROW_IDS_SHA256",
    "PINNED_LEXICAL_A_SOURCE_SHA256",
    "PINNED_LEXICAL_MODE_ARTIFACT_SHA256",
    "PINNED_LEXICAL_MODE_CONFIG_SHA256",
    "PINNED_PARTS_FIRST_CAPTURE_SHA256",
    "ExactLexicalRolePartition",
    "LexicalModeASource",
    "LexicalModeCandidate",
    "LexicalModeCompilerConfig",
    "LexicalModeDictionary",
    "LexicalModeNullReceipt",
    "LexicalModeProfileReceipt",
    "build_lexical_mode_A_source",
    "compile_lexical_mode_dictionary",
    "compile_lexical_mode_dictionary_from_capture",
    "compile_published_lexical_mode_dictionary",
    "load_lexical_mode_capture_strict",
    "load_lexical_mode_dictionary",
    "load_published_lexical_mode_dictionary",
    "load_published_lexical_mode_A_source",
    "main",
    "parse_lexical_mode_dictionary_json",
    "validate_lexical_mode_dictionary",
    "write_published_lexical_mode_dictionary",
]


LEXICAL_MODE_SCHEMA: Final = "fisher_graph.gemma3_parts_first.lexical_modes.v1"
LEXICAL_MODE_FORMAT_VERSION: Final = 1
LEXICAL_MODE_DISCOVERY_ROLE: Final = "A_discovery"
LEXICAL_MODE_PROFILE_ABI: Final = (
    "within_family_centered_nminus1_weighted_equal_role_frobenius.v1"
)
LEXICAL_MODE_NULL_ABI: Final = (
    "pcg64_raw_u64_rank_keys_c_order_column_family_row_paired_role_"
    "within_family_max_abs_cosine.v1"
)
LEXICAL_MODE_GENERATOR_ABI: Final = (
    "tied_hidden_column_unit_sum_subset_plus_complement.v1"
)
DEFAULT_LEXICAL_MODE_CAPTURE: Final = Path(
    ".local-runs/google--gemma-3-270m/parts-first-supermode-captures-v1.json"
)
DEFAULT_LEXICAL_MODE_OUTPUT: Final = Path(
    ".local-runs/google--gemma-3-270m/parts-first-lexical-mode-dictionary-v1.json"
)
PINNED_PARTS_FIRST_CAPTURE_SHA256: Final = (
    "0598ddf5ab0454d023f19e20803a671b253ea35984aabe45bf15256ee9565b72"
)
PINNED_LEXICAL_A_ROLE_SHA256: Final = (
    "ec02fbeb09bffc2cb01379b742555eb23f62629f273278f8a5839948684d73e6"
)
PINNED_LEXICAL_A_ROW_IDS_SHA256: Final = (
    "c2b1caf232241027bbe6e3b36a236c605c73de2eeb0a3085819c396fd4d64de4"
)
PINNED_LEXICAL_A_SOURCE_SHA256: Final = (
    "ab1d8bb304c2c73c3c227bfcba5148454531c22af135fcb9d93b0c04aaa8b689"
)
PINNED_LEXICAL_MODE_CONFIG_SHA256: Final = (
    "85e37d4acb4acd07ad844024a624e7c2518de67faebb20b52bad25c4d6e1bca9"
)
PINNED_LEXICAL_MODE_ARTIFACT_SHA256: Final = (
    "f4bde6e69c723361c657882b39081639eddbfb2908e3d2f1b6483fd7ef8482b5"
)

_ARTIFACT_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-modes:v1\0"
_CONFIG_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-mode-config:v1\0"
_PROFILE_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-profile:v1\0"
_NULL_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-null:v1\0"
_CANDIDATE_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-candidate:v1\0"
_PARTITION_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-partition:v1\0"
_AXIS_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-axis:v1\0"
_SOURCE_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-source:v1\0"
_EXACT_LIFT_DOMAIN = b"fisher_graph:gemma3-parts-first-lexical-exact-lift:v1\0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool")
    return value


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    header = _canonical_json_bytes(
        {"label": label, "shape": list(array.shape), "dtype": "float64_le"}
    )
    return hashlib.sha256(_PROFILE_DOMAIN + header + array.tobytes()).hexdigest()


def _numpy_build_config_sha256() -> str:
    configuration = np.__config__.show(mode="dicts")
    if not isinstance(configuration, Mapping):
        raise RuntimeError("NumPy build configuration is unavailable")
    return _json_sha256(configuration, domain=_NULL_DOMAIN)


def _nearest_rank_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("a null quantile requires at least one replicate")
    ordered = sorted(_finite_float(item, label="null maximum") for item in values)
    rank = max(1, math.ceil(quantile * (len(ordered) + 1)))
    if rank > len(ordered):
        raise ValueError("null replicate count is too small for its quantile")
    return ordered[rank - 1]


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


def _strict_json_object(
    source: str | bytes,
    *,
    label: str,
) -> Mapping[str, object]:
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
class LexicalModeASource:
    """Redacted capability containing static capture metadata and A only."""

    source_kind: str
    model_id: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    score_objective_id: str
    chart_surface_id: str
    protocol_sha256: str
    atom_universe_sha256: str
    response_axis: tuple[AtomResponseAxisEntry, ...]
    A_role: CaptureRole
    exact_lift_receipt: ExactLiftCaptureReceipt
    origin_capture_sha256: str
    published_A_anchor_authenticated: bool = False
    A_source_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.source_kind, label="source_kind")
        _require_text(self.model_id, label="model_id")
        _require_text(self.score_objective_id, label="score_objective_id")
        _require_text(self.chart_surface_id, label="chart_surface_id")
        for name in (
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "protocol_sha256",
            "atom_universe_sha256",
            "origin_capture_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not self.response_axis or any(
            type(item) is not AtomResponseAxisEntry for item in self.response_axis
        ):
            raise TypeError("A source response axis is invalid")
        if type(self.A_role) is not CaptureRole or self.A_role.role != (
            LEXICAL_MODE_DISCOVERY_ROLE
        ):
            raise TypeError("A source must contain exactly the A discovery role")
        if type(self.exact_lift_receipt) is not ExactLiftCaptureReceipt:
            raise TypeError("A source exact-lift receipt is invalid")
        _require_bool(
            self.published_A_anchor_authenticated,
            label="published_A_anchor_authenticated",
        )
        computed = _json_sha256(self._payload(), domain=_SOURCE_DOMAIN)
        if self.published_A_anchor_authenticated and (
            self.origin_capture_sha256 != PINNED_PARTS_FIRST_CAPTURE_SHA256
            or self.A_role.role_sha256 != PINNED_LEXICAL_A_ROLE_SHA256
            or self.A_role.row_ids_sha256 != PINNED_LEXICAL_A_ROW_IDS_SHA256
            or computed != PINNED_LEXICAL_A_SOURCE_SHA256
        ):
            raise ValueError("published lexical A anchor does not match its pins")
        if self.A_source_sha256 == "":
            object.__setattr__(self, "A_source_sha256", computed)
        elif _require_sha256(
            self.A_source_sha256, label="A_source_sha256"
        ) != computed:
            raise ValueError("lexical A source hash mismatch")

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
            "A_role": self.A_role.to_dict(),
            "exact_lift_receipt": self.exact_lift_receipt.to_dict(),
        }


def build_lexical_mode_A_source(
    captures: PartsFirstCaptureBundle,
    *,
    authenticate_published_A: bool = False,
) -> LexicalModeASource:
    """Redact B/C before the compiler receives a capability."""

    if type(captures) is not PartsFirstCaptureBundle:
        raise TypeError("A-source redaction requires an exact capture bundle")
    A_roles = tuple(
        role for role in captures.roles if role.role == LEXICAL_MODE_DISCOVERY_ROLE
    )
    if len(A_roles) != 1:
        raise ValueError("capture must contain exactly one A discovery role")
    if authenticate_published_A and captures.capture_sha256 != (
        PINNED_PARTS_FIRST_CAPTURE_SHA256
    ):
        raise ValueError("published parts-first capture hash drifted")
    return LexicalModeASource(
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
        A_role=A_roles[0],
        exact_lift_receipt=captures.exact_lift_receipt,
        origin_capture_sha256=captures.capture_sha256,
        published_A_anchor_authenticated=authenticate_published_A,
    )


def load_lexical_mode_capture_strict(
    source: Path | str,
) -> PartsFirstCaptureBundle:
    """Load a capture with duplicate-key and non-finite JSON rejection."""

    path = _require_json_path(source, label="lexical mode capture")
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError("lexical mode capture is unavailable") from error
    value = _strict_json_object(encoded, label="lexical mode capture")
    return PartsFirstCaptureBundle.from_dict(value)


def load_published_lexical_mode_A_source(
    source: Path | str = DEFAULT_LEXICAL_MODE_CAPTURE,
) -> LexicalModeASource:
    """Strict-load and authenticate the frozen model capture before redaction."""

    captures = load_lexical_mode_capture_strict(source)
    return build_lexical_mode_A_source(
        captures,
        authenticate_published_A=True,
    )


@dataclass(frozen=True, slots=True)
class LexicalModeCompilerConfig:
    """Preregistered A-only graph and deterministic null calibration."""

    signed_similarity_floor: float = 0.80
    role_stability_floor: float = 0.75
    permutation_count: int = 1_000
    permutation_seed: int = 31_415_926
    null_quantile: float = 0.95
    minimum_lofo_pass_count: int = 5
    minimum_active_family_count: int = 3
    active_family_relative_energy_floor: float = 0.05
    maximum_clique_size: int = 8
    maximum_candidate_count: int = 256

    def __post_init__(self) -> None:
        for name in (
            "signed_similarity_floor",
            "role_stability_floor",
            "null_quantile",
            "active_family_relative_energy_floor",
        ):
            value = _finite_float(getattr(self, name), label=name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        for name, minimum in (
            ("permutation_count", 19),
            ("permutation_seed", 0),
            ("minimum_lofo_pass_count", 1),
            ("minimum_active_family_count", 1),
            ("maximum_clique_size", 2),
            ("maximum_candidate_count", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.minimum_lofo_pass_count > 6:
            raise ValueError("minimum_lofo_pass_count cannot exceed six")
        if self.minimum_active_family_count > 6:
            raise ValueError("minimum_active_family_count cannot exceed six")
        if self.null_quantile != 0.95:
            raise ValueError("v1 freezes the familywise null quantile at 0.95")
        if math.ceil(self.null_quantile * (self.permutation_count + 1)) > (
            self.permutation_count
        ):
            raise ValueError("permutation_count is too small for null_quantile")

    def to_dict(self) -> dict[str, object]:
        return {
            "signed_similarity_floor": self.signed_similarity_floor,
            "role_stability_floor": self.role_stability_floor,
            "permutation_count": self.permutation_count,
            "permutation_seed": self.permutation_seed,
            "null_quantile": self.null_quantile,
            "minimum_lofo_pass_count": self.minimum_lofo_pass_count,
            "minimum_active_family_count": self.minimum_active_family_count,
            "active_family_relative_energy_floor": (
                self.active_family_relative_energy_floor
            ),
            "maximum_clique_size": self.maximum_clique_size,
            "maximum_candidate_count": self.maximum_candidate_count,
        }

    @property
    def config_sha256(self) -> str:
        return _json_sha256(self.to_dict(), domain=_CONFIG_DOMAIN)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LexicalModeCompilerConfig":
        _exact_keys(
            value,
            {
                "signed_similarity_floor",
                "role_stability_floor",
                "permutation_count",
                "permutation_seed",
                "null_quantile",
                "minimum_lofo_pass_count",
                "minimum_active_family_count",
                "active_family_relative_energy_floor",
                "maximum_clique_size",
                "maximum_candidate_count",
            },
            label="lexical mode compiler configuration",
        )
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ExactLexicalRolePartition:
    """One exact unit-coefficient subset plus complement at one occurrence."""

    boundary_id: str
    execution_ordinal: int
    subset_columns: tuple[int, ...]
    complement_columns: tuple[int, ...]
    membership_blob_sha256: str
    subset_slice_sha256: str
    complement_slice_sha256: str
    generator_abi: str = LEXICAL_MODE_GENERATOR_ABI
    subset_coefficient: float = 1.0
    complement_coefficient: float = 1.0
    exact_partition_validated: bool = True
    storage_credit_scalar_count: int = 0
    partition_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.boundary_id, label="boundary_id")
        _require_int(self.execution_ordinal, label="execution_ordinal")
        expected = {
            ExecutionBoundary.lexical_ingress().boundary_id: 0,
            ExecutionBoundary.lexical_readout().boundary_id: 38,
        }
        if expected.get(self.boundary_id) != self.execution_ordinal:
            raise ValueError("lexical partition boundary is not canonical")
        subset = tuple(self.subset_columns)
        complement = tuple(self.complement_columns)
        for label, values in (("subset", subset), ("complement", complement)):
            if not values or values != tuple(sorted(set(values))):
                raise ValueError(f"{label} columns must be sorted and unique")
            if values[0] < 0 or values[-1] >= HIDDEN_WIDTH:
                raise ValueError(f"{label} column is out of range")
        if set(subset).intersection(complement) or set(subset).union(complement) != set(
            range(HIDDEN_WIDTH)
        ):
            raise ValueError("lexical subset/complement partition does not close")
        for name in (
            "membership_blob_sha256",
            "subset_slice_sha256",
            "complement_slice_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.generator_abi != LEXICAL_MODE_GENERATOR_ABI:
            raise ValueError("lexical generator ABI drifted")
        if self.subset_coefficient != 1.0 or self.complement_coefficient != 1.0:
            raise ValueError("native lexical generators require unit coefficients")
        if self.exact_partition_validated is not True:
            raise ValueError("lexical subset/complement partition is not exact")
        if self.storage_credit_scalar_count != 0:
            raise ValueError("logical lexical groups receive no storage credit")
        computed = _json_sha256(self._payload(), domain=_PARTITION_DOMAIN)
        if self.partition_sha256 == "":
            object.__setattr__(self, "partition_sha256", computed)
        elif _require_sha256(
            self.partition_sha256, label="partition_sha256"
        ) != computed:
            raise ValueError("lexical partition hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "boundary_id": self.boundary_id,
            "execution_ordinal": self.execution_ordinal,
            "subset_columns": list(self.subset_columns),
            "complement_columns": list(self.complement_columns),
            "membership_blob_sha256": self.membership_blob_sha256,
            "subset_slice_sha256": self.subset_slice_sha256,
            "complement_slice_sha256": self.complement_slice_sha256,
            "generator_abi": self.generator_abi,
            "subset_coefficient": self.subset_coefficient,
            "complement_coefficient": self.complement_coefficient,
            "exact_partition_validated": self.exact_partition_validated,
            "storage_credit_scalar_count": self.storage_credit_scalar_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "partition_sha256": self.partition_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExactLexicalRolePartition":
        _exact_keys(
            value,
            {
                "boundary_id",
                "execution_ordinal",
                "subset_columns",
                "complement_columns",
                "membership_blob_sha256",
                "subset_slice_sha256",
                "complement_slice_sha256",
                "generator_abi",
                "subset_coefficient",
                "complement_coefficient",
                "exact_partition_validated",
                "storage_credit_scalar_count",
                "partition_sha256",
            },
            label="exact lexical role partition",
        )
        subset = value["subset_columns"]
        complement = value["complement_columns"]
        if not isinstance(subset, list) or not isinstance(complement, list):
            raise ValueError("lexical partition columns must be arrays")
        return cls(
            boundary_id=str(value["boundary_id"]),
            execution_ordinal=_require_int(
                value["execution_ordinal"], label="execution_ordinal"
            ),
            subset_columns=tuple(
                _require_int(item, label="subset column") for item in subset
            ),
            complement_columns=tuple(
                _require_int(item, label="complement column") for item in complement
            ),
            membership_blob_sha256=str(value["membership_blob_sha256"]),
            subset_slice_sha256=str(value["subset_slice_sha256"]),
            complement_slice_sha256=str(value["complement_slice_sha256"]),
            generator_abi=str(value["generator_abi"]),
            subset_coefficient=_finite_float(
                value["subset_coefficient"], label="subset_coefficient"
            ),
            complement_coefficient=_finite_float(
                value["complement_coefficient"], label="complement_coefficient"
            ),
            exact_partition_validated=_require_bool(
                value["exact_partition_validated"],
                label="exact_partition_validated",
            ),
            storage_credit_scalar_count=_require_int(
                value["storage_credit_scalar_count"],
                label="storage_credit_scalar_count",
            ),
            partition_sha256=str(value["partition_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class LexicalModeProfileReceipt:
    row_count: int
    family_ids: tuple[str, ...]
    family_row_counts: tuple[tuple[str, int], ...]
    raw_ingress_empirical_fisher_mass: float
    raw_readout_empirical_fisher_mass: float
    centered_ingress_frobenius_energy: float
    centered_readout_frobenius_energy: float
    family_energy_rows: tuple[tuple[str, float, float, float, float], ...]
    ingress_profile_sha256: str
    readout_profile_sha256: str
    joint_profile_sha256: str
    profile_abi: str = LEXICAL_MODE_PROFILE_ABI
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        _require_int(self.row_count, label="row_count", minimum=1)
        if len(self.family_ids) != 6 or self.family_ids != tuple(
            sorted(set(self.family_ids))
        ):
            raise ValueError("lexical profile requires six canonical families")
        if tuple(item[0] for item in self.family_row_counts) != self.family_ids:
            raise ValueError("lexical profile family counts are not aligned")
        if any(
            _require_int(item[1], label="family row count", minimum=2) < 2
            for item in self.family_row_counts
        ):
            raise ValueError("each A family requires at least two rows")
        if sum(item[1] for item in self.family_row_counts) != self.row_count:
            raise ValueError("lexical profile family counts do not close")
        for name in (
            "raw_ingress_empirical_fisher_mass",
            "raw_readout_empirical_fisher_mass",
            "centered_ingress_frobenius_energy",
            "centered_readout_frobenius_energy",
        ):
            value = _finite_float(getattr(self, name), label=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if tuple(item[0] for item in self.family_energy_rows) != self.family_ids:
            raise ValueError("lexical profile family energy rows are not aligned")
        for row in self.family_energy_rows:
            if len(row) != 5 or any(
                _finite_float(item, label="family energy") < 0 for item in row[1:]
            ):
                raise ValueError("lexical profile family energy is invalid")
        for name in (
            "ingress_profile_sha256",
            "readout_profile_sha256",
            "joint_profile_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if self.profile_abi != LEXICAL_MODE_PROFILE_ABI:
            raise ValueError("lexical profile ABI drifted")
        computed = _json_sha256(self._payload(), domain=_PROFILE_DOMAIN)
        if self.profile_sha256 == "":
            object.__setattr__(self, "profile_sha256", computed)
        elif _require_sha256(
            self.profile_sha256, label="profile_sha256"
        ) != computed:
            raise ValueError("lexical profile receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "family_ids": list(self.family_ids),
            "family_row_counts": [list(item) for item in self.family_row_counts],
            "raw_ingress_empirical_fisher_mass": (
                self.raw_ingress_empirical_fisher_mass
            ),
            "raw_readout_empirical_fisher_mass": (
                self.raw_readout_empirical_fisher_mass
            ),
            "centered_ingress_frobenius_energy": (
                self.centered_ingress_frobenius_energy
            ),
            "centered_readout_frobenius_energy": (
                self.centered_readout_frobenius_energy
            ),
            "family_energy_rows": [list(item) for item in self.family_energy_rows],
            "ingress_profile_sha256": self.ingress_profile_sha256,
            "readout_profile_sha256": self.readout_profile_sha256,
            "joint_profile_sha256": self.joint_profile_sha256,
            "profile_abi": self.profile_abi,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "profile_sha256": self.profile_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LexicalModeProfileReceipt":
        expected = {
            "row_count",
            "family_ids",
            "family_row_counts",
            "raw_ingress_empirical_fisher_mass",
            "raw_readout_empirical_fisher_mass",
            "centered_ingress_frobenius_energy",
            "centered_readout_frobenius_energy",
            "family_energy_rows",
            "ingress_profile_sha256",
            "readout_profile_sha256",
            "joint_profile_sha256",
            "profile_abi",
            "profile_sha256",
        }
        _exact_keys(value, expected, label="lexical profile receipt")
        family_ids = value["family_ids"]
        counts = value["family_row_counts"]
        energies = value["family_energy_rows"]
        if (
            not isinstance(family_ids, list)
            or not isinstance(counts, list)
            or not isinstance(energies, list)
        ):
            raise ValueError("lexical profile arrays are invalid")
        parsed_counts: list[tuple[str, int]] = []
        for item in counts:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("lexical family count row is invalid")
            parsed_counts.append(
                (str(item[0]), _require_int(item[1], label="family row count"))
            )
        parsed_energies: list[tuple[str, float, float, float, float]] = []
        for item in energies:
            if not isinstance(item, list) or len(item) != 5:
                raise ValueError("lexical family energy row is invalid")
            parsed_energies.append(
                (
                    str(item[0]),
                    *(
                        _finite_float(part, label="family energy")
                        for part in item[1:]
                    ),
                )
            )
        return cls(
            row_count=_require_int(value["row_count"], label="row_count"),
            family_ids=tuple(str(item) for item in family_ids),
            family_row_counts=tuple(parsed_counts),
            raw_ingress_empirical_fisher_mass=_finite_float(
                value["raw_ingress_empirical_fisher_mass"],
                label="raw_ingress_empirical_fisher_mass",
            ),
            raw_readout_empirical_fisher_mass=_finite_float(
                value["raw_readout_empirical_fisher_mass"],
                label="raw_readout_empirical_fisher_mass",
            ),
            centered_ingress_frobenius_energy=_finite_float(
                value["centered_ingress_frobenius_energy"],
                label="centered_ingress_frobenius_energy",
            ),
            centered_readout_frobenius_energy=_finite_float(
                value["centered_readout_frobenius_energy"],
                label="centered_readout_frobenius_energy",
            ),
            family_energy_rows=tuple(parsed_energies),
            ingress_profile_sha256=str(value["ingress_profile_sha256"]),
            readout_profile_sha256=str(value["readout_profile_sha256"]),
            joint_profile_sha256=str(value["joint_profile_sha256"]),
            profile_abi=str(value["profile_abi"]),
            profile_sha256=str(value["profile_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class LexicalModeNullReceipt:
    permutation_count: int
    permutation_seed: int
    null_quantile: float
    sorted_max_absolute_cosines: tuple[float, ...]
    max_absolute_cosine_q95: float
    signed_similarity_floor: float
    effective_joint_similarity_floor: float
    numpy_version: str
    numpy_build_config_sha256: str
    null_abi: str = LEXICAL_MODE_NULL_ABI
    null_sha256: str = ""

    def __post_init__(self) -> None:
        _require_int(self.permutation_count, label="permutation_count", minimum=1)
        _require_int(self.permutation_seed, label="permutation_seed")
        quantile = _finite_float(self.null_quantile, label="null_quantile")
        if not 0 < quantile <= 1:
            raise ValueError("null_quantile must be in (0, 1]")
        maxima = tuple(
            _finite_float(item, label="null maximum")
            for item in self.sorted_max_absolute_cosines
        )
        if len(maxima) != self.permutation_count or maxima != tuple(sorted(maxima)):
            raise ValueError("null maxima must be complete and sorted")
        if maxima[0] < 0 or maxima[-1] > 1 + 1e-12:
            raise ValueError("null maximum cosine is out of range")
        expected_q = _nearest_rank_quantile(maxima, quantile)
        if self.max_absolute_cosine_q95 != expected_q:
            raise ValueError("null quantile does not close")
        floor = _finite_float(
            self.signed_similarity_floor,
            label="signed_similarity_floor",
        )
        effective = _finite_float(
            self.effective_joint_similarity_floor,
            label="effective_joint_similarity_floor",
        )
        if effective != max(floor, expected_q):
            raise ValueError("effective lexical similarity floor does not close")
        if self.numpy_version != np.__version__:
            raise ValueError("lexical null NumPy version drifted")
        if self.numpy_build_config_sha256 != _numpy_build_config_sha256():
            raise ValueError("lexical null NumPy build configuration drifted")
        if self.null_abi != LEXICAL_MODE_NULL_ABI:
            raise ValueError("lexical null ABI drifted")
        computed = _json_sha256(self._payload(), domain=_NULL_DOMAIN)
        if self.null_sha256 == "":
            object.__setattr__(self, "null_sha256", computed)
        elif _require_sha256(self.null_sha256, label="null_sha256") != computed:
            raise ValueError("lexical null receipt hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "permutation_count": self.permutation_count,
            "permutation_seed": self.permutation_seed,
            "null_quantile": self.null_quantile,
            "sorted_max_absolute_cosines": list(self.sorted_max_absolute_cosines),
            "max_absolute_cosine_q95": self.max_absolute_cosine_q95,
            "signed_similarity_floor": self.signed_similarity_floor,
            "effective_joint_similarity_floor": (
                self.effective_joint_similarity_floor
            ),
            "numpy_version": self.numpy_version,
            "numpy_build_config_sha256": self.numpy_build_config_sha256,
            "null_abi": self.null_abi,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "null_sha256": self.null_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LexicalModeNullReceipt":
        _exact_keys(
            value,
            {
                "permutation_count",
                "permutation_seed",
                "null_quantile",
                "sorted_max_absolute_cosines",
                "max_absolute_cosine_q95",
                "signed_similarity_floor",
                "effective_joint_similarity_floor",
                "numpy_version",
                "numpy_build_config_sha256",
                "null_abi",
                "null_sha256",
            },
            label="lexical mode null receipt",
        )
        maxima = value["sorted_max_absolute_cosines"]
        if not isinstance(maxima, list):
            raise ValueError("lexical null maxima must be an array")
        return cls(
            permutation_count=_require_int(
                value["permutation_count"], label="permutation_count"
            ),
            permutation_seed=_require_int(
                value["permutation_seed"], label="permutation_seed"
            ),
            null_quantile=_finite_float(value["null_quantile"], label="null_quantile"),
            sorted_max_absolute_cosines=tuple(
                _finite_float(item, label="null maximum") for item in maxima
            ),
            max_absolute_cosine_q95=_finite_float(
                value["max_absolute_cosine_q95"],
                label="max_absolute_cosine_q95",
            ),
            signed_similarity_floor=_finite_float(
                value["signed_similarity_floor"], label="signed_similarity_floor"
            ),
            effective_joint_similarity_floor=_finite_float(
                value["effective_joint_similarity_floor"],
                label="effective_joint_similarity_floor",
            ),
            numpy_version=str(value["numpy_version"]),
            numpy_build_config_sha256=str(value["numpy_build_config_sha256"]),
            null_abi=str(value["null_abi"]),
            null_sha256=str(value["null_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class LexicalModeCandidate:
    """A stable positive response clique; never a new physical parameter."""

    candidate_id: str
    member_columns: tuple[int, ...]
    joint_min_signed_cosine: float
    ingress_min_signed_cosine: float
    readout_min_signed_cosine: float
    balanced_empirical_fisher_mass: float
    role_scope: str
    parity_joint_signed_cosines: tuple[float, float]
    lofo_joint_signed_cosines: tuple[float, ...]
    ingress_parity_signed_cosines: tuple[float, float]
    readout_parity_signed_cosines: tuple[float, float]
    ingress_lofo_signed_cosines: tuple[float, ...]
    readout_lofo_signed_cosines: tuple[float, ...]
    lofo_pass_count: int
    active_family_count: int
    ingress_partition: ExactLexicalRolePartition
    readout_partition: ExactLexicalRolePartition
    A_discovery_only: bool = True
    native_coefficients_all_one: bool = True
    storage_credit_scalar_count: int = 0
    candidate_sha256: str = ""

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, label="candidate_id")
        members = tuple(self.member_columns)
        if (
            len(members) < 2
            or members != tuple(sorted(set(members)))
            or members[0] < 0
            or members[-1] >= HIDDEN_WIDTH
        ):
            raise ValueError("lexical candidate members are invalid")
        for name in (
            "joint_min_signed_cosine",
            "ingress_min_signed_cosine",
            "readout_min_signed_cosine",
        ):
            value = _finite_float(getattr(self, name), label=name)
            if not -1.000000000001 <= value <= 1.000000000001:
                raise ValueError(f"{name} is outside cosine range")
        if _finite_float(
            self.balanced_empirical_fisher_mass,
            label="balanced_empirical_fisher_mass",
        ) <= 0:
            raise ValueError("candidate Fisher mass must be positive")
        if self.role_scope not in {"lexical_ingress", "lexical_readout", "dual_role"}:
            raise ValueError("lexical candidate role scope is invalid")
        parity_rows = (
            self.parity_joint_signed_cosines,
            self.ingress_parity_signed_cosines,
            self.readout_parity_signed_cosines,
        )
        lofo_rows = (
            self.lofo_joint_signed_cosines,
            self.ingress_lofo_signed_cosines,
            self.readout_lofo_signed_cosines,
        )
        if any(len(row) != 2 for row in parity_rows) or any(
            len(row) != 6 for row in lofo_rows
        ):
            raise ValueError("lexical stability evidence is incomplete")
        for value in (
            *(item for row in parity_rows for item in row),
            *(item for row in lofo_rows for item in row),
        ):
            cosine = _finite_float(value, label="stability cosine")
            if not -1.000000000001 <= cosine <= 1.000000000001:
                raise ValueError("stability cosine is out of range")
        _require_int(self.lofo_pass_count, label="lofo_pass_count")
        _require_int(self.active_family_count, label="active_family_count")
        if not isinstance(
            self.ingress_partition, ExactLexicalRolePartition
        ) or not isinstance(self.readout_partition, ExactLexicalRolePartition):
            raise TypeError("lexical candidate partitions are invalid")
        if (
            self.ingress_partition.subset_columns != members
            or self.readout_partition.subset_columns != members
        ):
            raise ValueError("lexical candidate role memberships diverged")
        if (
            self.A_discovery_only is not True
            or self.native_coefficients_all_one is not True
        ):
            raise ValueError("lexical candidates must remain A-only native sums")
        if self.storage_credit_scalar_count != 0:
            raise ValueError("lexical candidates receive no storage credit")
        computed = _json_sha256(self._payload(), domain=_CANDIDATE_DOMAIN)
        if self.candidate_sha256 == "":
            object.__setattr__(self, "candidate_sha256", computed)
        elif _require_sha256(
            self.candidate_sha256, label="candidate_sha256"
        ) != computed:
            raise ValueError("lexical candidate hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "member_columns": list(self.member_columns),
            "joint_min_signed_cosine": self.joint_min_signed_cosine,
            "ingress_min_signed_cosine": self.ingress_min_signed_cosine,
            "readout_min_signed_cosine": self.readout_min_signed_cosine,
            "balanced_empirical_fisher_mass": self.balanced_empirical_fisher_mass,
            "role_scope": self.role_scope,
            "parity_joint_signed_cosines": list(
                self.parity_joint_signed_cosines
            ),
            "lofo_joint_signed_cosines": list(self.lofo_joint_signed_cosines),
            "ingress_parity_signed_cosines": list(
                self.ingress_parity_signed_cosines
            ),
            "readout_parity_signed_cosines": list(
                self.readout_parity_signed_cosines
            ),
            "ingress_lofo_signed_cosines": list(
                self.ingress_lofo_signed_cosines
            ),
            "readout_lofo_signed_cosines": list(
                self.readout_lofo_signed_cosines
            ),
            "lofo_pass_count": self.lofo_pass_count,
            "active_family_count": self.active_family_count,
            "ingress_partition": self.ingress_partition.to_dict(),
            "readout_partition": self.readout_partition.to_dict(),
            "A_discovery_only": self.A_discovery_only,
            "native_coefficients_all_one": self.native_coefficients_all_one,
            "storage_credit_scalar_count": self.storage_credit_scalar_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "candidate_sha256": self.candidate_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LexicalModeCandidate":
        _exact_keys(
            value,
            {
                "candidate_id",
                "member_columns",
                "joint_min_signed_cosine",
                "ingress_min_signed_cosine",
                "readout_min_signed_cosine",
                "balanced_empirical_fisher_mass",
                "role_scope",
                "parity_joint_signed_cosines",
                "lofo_joint_signed_cosines",
                "ingress_parity_signed_cosines",
                "readout_parity_signed_cosines",
                "ingress_lofo_signed_cosines",
                "readout_lofo_signed_cosines",
                "lofo_pass_count",
                "active_family_count",
                "ingress_partition",
                "readout_partition",
                "A_discovery_only",
                "native_coefficients_all_one",
                "storage_credit_scalar_count",
                "candidate_sha256",
            },
            label="lexical mode candidate",
        )
        members = value["member_columns"]
        parity = value["parity_joint_signed_cosines"]
        lofo = value["lofo_joint_signed_cosines"]
        ingress_parity = value["ingress_parity_signed_cosines"]
        readout_parity = value["readout_parity_signed_cosines"]
        ingress_lofo = value["ingress_lofo_signed_cosines"]
        readout_lofo = value["readout_lofo_signed_cosines"]
        ingress = value["ingress_partition"]
        readout = value["readout_partition"]
        if (
            not isinstance(members, list)
            or not isinstance(parity, list)
            or not isinstance(lofo, list)
            or not isinstance(ingress_parity, list)
            or not isinstance(readout_parity, list)
            or not isinstance(ingress_lofo, list)
            or not isinstance(readout_lofo, list)
            or not isinstance(ingress, Mapping)
            or not isinstance(readout, Mapping)
        ):
            raise ValueError("lexical candidate nested fields are invalid")
        return cls(
            candidate_id=str(value["candidate_id"]),
            member_columns=tuple(
                _require_int(item, label="member column") for item in members
            ),
            joint_min_signed_cosine=_finite_float(
                value["joint_min_signed_cosine"],
                label="joint_min_signed_cosine",
            ),
            ingress_min_signed_cosine=_finite_float(
                value["ingress_min_signed_cosine"],
                label="ingress_min_signed_cosine",
            ),
            readout_min_signed_cosine=_finite_float(
                value["readout_min_signed_cosine"],
                label="readout_min_signed_cosine",
            ),
            balanced_empirical_fisher_mass=_finite_float(
                value["balanced_empirical_fisher_mass"],
                label="balanced_empirical_fisher_mass",
            ),
            role_scope=str(value["role_scope"]),
            parity_joint_signed_cosines=tuple(
                _finite_float(item, label="parity cosine") for item in parity
            ),  # type: ignore[arg-type]
            lofo_joint_signed_cosines=tuple(
                _finite_float(item, label="LOFO cosine") for item in lofo
            ),
            ingress_parity_signed_cosines=tuple(
                _finite_float(item, label="ingress parity cosine")
                for item in ingress_parity
            ),  # type: ignore[arg-type]
            readout_parity_signed_cosines=tuple(
                _finite_float(item, label="readout parity cosine")
                for item in readout_parity
            ),  # type: ignore[arg-type]
            ingress_lofo_signed_cosines=tuple(
                _finite_float(item, label="ingress LOFO cosine")
                for item in ingress_lofo
            ),
            readout_lofo_signed_cosines=tuple(
                _finite_float(item, label="readout LOFO cosine")
                for item in readout_lofo
            ),
            lofo_pass_count=_require_int(
                value["lofo_pass_count"], label="lofo_pass_count"
            ),
            active_family_count=_require_int(
                value["active_family_count"], label="active_family_count"
            ),
            ingress_partition=ExactLexicalRolePartition.from_dict(ingress),
            readout_partition=ExactLexicalRolePartition.from_dict(readout),
            A_discovery_only=_require_bool(
                value["A_discovery_only"], label="A_discovery_only"
            ),
            native_coefficients_all_one=_require_bool(
                value["native_coefficients_all_one"],
                label="native_coefficients_all_one",
            ),
            storage_credit_scalar_count=_require_int(
                value["storage_credit_scalar_count"],
                label="storage_credit_scalar_count",
            ),
            candidate_sha256=str(value["candidate_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class LexicalModeDictionary:
    """Frozen structural artifact derived only from ``A_discovery``."""

    source_kind: str
    model_id: str
    source_model_fingerprint_sha256: str
    source_execution_fingerprint_sha256: str
    score_objective_id: str
    chart_surface_id: str
    protocol_sha256: str
    exact_control_evidence_sha256: str
    exact_lift_receipt_sha256: str
    source_envelope_sha256: str
    atom_universe_sha256: str
    A_role_sha256: str
    A_row_ids_sha256: str
    lexical_axis_sha256: str
    configuration: LexicalModeCompilerConfig
    profile_receipt: LexicalModeProfileReceipt
    null_receipt: LexicalModeNullReceipt
    candidates: tuple[LexicalModeCandidate, ...]
    roles_read: tuple[str, ...] = (LEXICAL_MODE_DISCOVERY_ROLE,)
    B_rows_read: int = 0
    C_rows_read: int = 0
    physical_basis_kind: str = "640_tied_hidden_column_singletons"
    physical_basis_count: int = HIDDEN_WIDTH
    physical_basis_sha256: str = ""
    candidate_semantics: str = "logical_overlap_no_storage_credit"
    profile_abi: str = LEXICAL_MODE_PROFILE_ABI
    null_abi: str = LEXICAL_MODE_NULL_ABI
    compression_attempted: bool = False
    compression_claimed: bool = False
    semantic_confirmation_claimed: bool = False
    parameter_savings_claimed: bool = False
    mac_savings_claimed: bool = False
    latency_claimed: bool = False
    attention_chart_structure_claimed: bool = False
    basis_invariant_semantics_claimed: bool = False
    full_capture_sha256_intentionally_omitted_for_heldout_firewall: bool = True
    published_A_anchor_authenticated: bool = False
    fisher_quantity_semantics: str = (
        "mean_squared_signed_atom_response_not_full_parameter_fisher_matrix"
    )
    grouping_statistic: str = "positive_signed_response_cosine_not_squared_mass"
    tied_physical_storage_counted_once: bool = True
    ingress_and_readout_role_atlases_are_separate: bool = True
    schema: str = LEXICAL_MODE_SCHEMA
    format_version: int = LEXICAL_MODE_FORMAT_VERSION
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != LEXICAL_MODE_SCHEMA or self.format_version != 1:
            raise ValueError("lexical mode dictionary header is invalid")
        _require_text(self.source_kind, label="source_kind")
        _require_text(self.model_id, label="model_id")
        _require_text(self.score_objective_id, label="score_objective_id")
        _require_text(self.chart_surface_id, label="chart_surface_id")
        for name in (
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "protocol_sha256",
            "exact_control_evidence_sha256",
            "exact_lift_receipt_sha256",
            "source_envelope_sha256",
            "atom_universe_sha256",
            "A_role_sha256",
            "A_row_ids_sha256",
            "lexical_axis_sha256",
            "physical_basis_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.configuration, LexicalModeCompilerConfig):
            raise TypeError("lexical mode configuration is invalid")
        if not isinstance(self.profile_receipt, LexicalModeProfileReceipt):
            raise TypeError("lexical profile receipt is invalid")
        if not isinstance(self.null_receipt, LexicalModeNullReceipt):
            raise TypeError("lexical null receipt is invalid")
        if self.null_receipt.permutation_count != self.configuration.permutation_count:
            raise ValueError("lexical null/configuration replicate count drifted")
        if self.null_receipt.permutation_seed != self.configuration.permutation_seed:
            raise ValueError("lexical null/configuration seed drifted")
        if self.null_receipt.null_quantile != self.configuration.null_quantile:
            raise ValueError("lexical null/configuration quantile drifted")
        if self.null_receipt.signed_similarity_floor != (
            self.configuration.signed_similarity_floor
        ):
            raise ValueError("lexical null/configuration similarity floor drifted")
        if any(not isinstance(item, LexicalModeCandidate) for item in self.candidates):
            raise TypeError("lexical candidate inventory is invalid")
        if len(self.candidates) > self.configuration.maximum_candidate_count:
            raise ValueError("lexical candidate inventory exceeds its bound")
        for candidate in self.candidates:
            if len(candidate.member_columns) > self.configuration.maximum_clique_size:
                raise ValueError("lexical candidate exceeds its clique-size bound")
            if candidate.joint_min_signed_cosine < (
                self.null_receipt.effective_joint_similarity_floor
            ):
                raise ValueError("lexical candidate misses the calibrated A gate")
            if min(candidate.parity_joint_signed_cosines) < (
                self.configuration.role_stability_floor
            ):
                raise ValueError("lexical candidate misses an A parity gate")
            expected_joint_lofo_pass = sum(
                value >= self.configuration.role_stability_floor
                for value in candidate.lofo_joint_signed_cosines
            )
            if (
                candidate.lofo_pass_count != expected_joint_lofo_pass
                or expected_joint_lofo_pass
                < self.configuration.minimum_lofo_pass_count
            ):
                raise ValueError("lexical candidate LOFO evidence does not close")
            if candidate.active_family_count < (
                self.configuration.minimum_active_family_count
            ):
                raise ValueError("lexical candidate has insufficient family support")
            if not 0 < candidate.balanced_empirical_fisher_mass <= 1:
                raise ValueError("lexical candidate Fisher mass is out of range")

            def role_passed(role: str) -> bool:
                if role == "ingress":
                    full = candidate.ingress_min_signed_cosine
                    parity = candidate.ingress_parity_signed_cosines
                    lofo = candidate.ingress_lofo_signed_cosines
                else:
                    full = candidate.readout_min_signed_cosine
                    parity = candidate.readout_parity_signed_cosines
                    lofo = candidate.readout_lofo_signed_cosines
                return (
                    full >= self.configuration.role_stability_floor
                    and min(parity) >= self.configuration.role_stability_floor
                    and sum(
                        value >= self.configuration.role_stability_floor
                        for value in lofo
                    )
                    >= self.configuration.minimum_lofo_pass_count
                )

            ingress_passed = role_passed("ingress")
            readout_passed = role_passed("readout")
            expected_scope = (
                "dual_role"
                if ingress_passed and readout_passed
                else "lexical_ingress"
                if ingress_passed
                else "lexical_readout"
                if readout_passed
                else None
            )
            if candidate.role_scope != expected_scope:
                raise ValueError("lexical candidate role scope does not close")
            ingress_partition = candidate.ingress_partition
            readout_partition = candidate.readout_partition
            if (
                ingress_partition.membership_blob_sha256
                != readout_partition.membership_blob_sha256
                or ingress_partition.subset_slice_sha256
                != readout_partition.subset_slice_sha256
                or ingress_partition.complement_slice_sha256
                != readout_partition.complement_slice_sha256
            ):
                raise ValueError("tied lexical role partition membership diverged")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("lexical candidate IDs are not canonical")
        if self.roles_read != (LEXICAL_MODE_DISCOVERY_ROLE,):
            raise ValueError("lexical discovery may read only A")
        if self.B_rows_read != 0 or self.C_rows_read != 0:
            raise ValueError("lexical discovery opened a held-out role")
        if self.physical_basis_kind != "640_tied_hidden_column_singletons":
            raise ValueError("lexical physical basis kind drifted")
        if self.physical_basis_count != HIDDEN_WIDTH:
            raise ValueError("lexical physical basis does not contain 640 columns")
        expected_basis = _json_sha256(
            list(range(HIDDEN_WIDTH)), domain=_PARTITION_DOMAIN
        )
        if self.physical_basis_sha256 != expected_basis:
            raise ValueError("lexical physical singleton basis hash drifted")
        if self.candidate_semantics != "logical_overlap_no_storage_credit":
            raise ValueError("lexical candidate semantics drifted")
        if (
            self.profile_abi != LEXICAL_MODE_PROFILE_ABI
            or self.null_abi != LEXICAL_MODE_NULL_ABI
        ):
            raise ValueError("lexical mode dictionary ABI drifted")
        for name in (
            "compression_attempted",
            "compression_claimed",
            "semantic_confirmation_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
            "attention_chart_structure_claimed",
            "basis_invariant_semantics_claimed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} must remain false")
        if (
            self.full_capture_sha256_intentionally_omitted_for_heldout_firewall
            is not True
        ):
            raise ValueError("full capture binding would violate the B/C firewall")
        _require_bool(
            self.published_A_anchor_authenticated,
            label="published_A_anchor_authenticated",
        )
        if self.published_A_anchor_authenticated and (
            self.A_role_sha256 != PINNED_LEXICAL_A_ROLE_SHA256
            or self.A_row_ids_sha256 != PINNED_LEXICAL_A_ROW_IDS_SHA256
            or self.source_envelope_sha256 != PINNED_LEXICAL_A_SOURCE_SHA256
            or self.configuration.config_sha256
            != PINNED_LEXICAL_MODE_CONFIG_SHA256
        ):
            raise ValueError("published lexical A dictionary pins drifted")
        if self.fisher_quantity_semantics != (
            "mean_squared_signed_atom_response_not_full_parameter_fisher_matrix"
        ):
            raise ValueError("lexical Fisher quantity semantics drifted")
        if self.grouping_statistic != (
            "positive_signed_response_cosine_not_squared_mass"
        ):
            raise ValueError("lexical grouping statistic drifted")
        if self.tied_physical_storage_counted_once is not True:
            raise ValueError("tied lexical storage must be counted once")
        if self.ingress_and_readout_role_atlases_are_separate is not True:
            raise ValueError("lexical role atlases must remain separate")
        computed = _json_sha256(self._payload(), domain=_ARTIFACT_DOMAIN)
        if (
            self.published_A_anchor_authenticated
            and computed != PINNED_LEXICAL_MODE_ARTIFACT_SHA256
        ):
            raise ValueError("published lexical mode artifact pin drifted")
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif _require_sha256(
            self.artifact_sha256, label="artifact_sha256"
        ) != computed:
            raise ValueError("lexical mode dictionary hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "source_kind": self.source_kind,
            "model_id": self.model_id,
            "source_model_fingerprint_sha256": self.source_model_fingerprint_sha256,
            "source_execution_fingerprint_sha256": (
                self.source_execution_fingerprint_sha256
            ),
            "score_objective_id": self.score_objective_id,
            "chart_surface_id": self.chart_surface_id,
            "protocol_sha256": self.protocol_sha256,
            "exact_control_evidence_sha256": self.exact_control_evidence_sha256,
            "exact_lift_receipt_sha256": self.exact_lift_receipt_sha256,
            "source_envelope_sha256": self.source_envelope_sha256,
            "atom_universe_sha256": self.atom_universe_sha256,
            "A_role_sha256": self.A_role_sha256,
            "A_row_ids_sha256": self.A_row_ids_sha256,
            "lexical_axis_sha256": self.lexical_axis_sha256,
            "configuration": self.configuration.to_dict(),
            "configuration_sha256": self.configuration.config_sha256,
            "profile_receipt": self.profile_receipt.to_dict(),
            "null_receipt": self.null_receipt.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "roles_read": list(self.roles_read),
            "B_rows_read": self.B_rows_read,
            "C_rows_read": self.C_rows_read,
            "physical_basis_kind": self.physical_basis_kind,
            "physical_basis_count": self.physical_basis_count,
            "physical_basis_sha256": self.physical_basis_sha256,
            "candidate_semantics": self.candidate_semantics,
            "profile_abi": self.profile_abi,
            "null_abi": self.null_abi,
            "compression_attempted": self.compression_attempted,
            "compression_claimed": self.compression_claimed,
            "semantic_confirmation_claimed": self.semantic_confirmation_claimed,
            "parameter_savings_claimed": self.parameter_savings_claimed,
            "mac_savings_claimed": self.mac_savings_claimed,
            "latency_claimed": self.latency_claimed,
            "attention_chart_structure_claimed": (
                self.attention_chart_structure_claimed
            ),
            "basis_invariant_semantics_claimed": (
                self.basis_invariant_semantics_claimed
            ),
            "full_capture_sha256_intentionally_omitted_for_heldout_firewall": (
                self.full_capture_sha256_intentionally_omitted_for_heldout_firewall
            ),
            "published_A_anchor_authenticated": (
                self.published_A_anchor_authenticated
            ),
            "fisher_quantity_semantics": self.fisher_quantity_semantics,
            "grouping_statistic": self.grouping_statistic,
            "tied_physical_storage_counted_once": (
                self.tied_physical_storage_counted_once
            ),
            "ingress_and_readout_role_atlases_are_separate": (
                self.ingress_and_readout_role_atlases_are_separate
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LexicalModeDictionary":
        expected = {
            "schema",
            "format_version",
            "source_kind",
            "model_id",
            "source_model_fingerprint_sha256",
            "source_execution_fingerprint_sha256",
            "score_objective_id",
            "chart_surface_id",
            "protocol_sha256",
            "exact_control_evidence_sha256",
            "exact_lift_receipt_sha256",
            "source_envelope_sha256",
            "atom_universe_sha256",
            "A_role_sha256",
            "A_row_ids_sha256",
            "lexical_axis_sha256",
            "configuration",
            "configuration_sha256",
            "profile_receipt",
            "null_receipt",
            "candidates",
            "roles_read",
            "B_rows_read",
            "C_rows_read",
            "physical_basis_kind",
            "physical_basis_count",
            "physical_basis_sha256",
            "candidate_semantics",
            "profile_abi",
            "null_abi",
            "compression_attempted",
            "compression_claimed",
            "semantic_confirmation_claimed",
            "parameter_savings_claimed",
            "mac_savings_claimed",
            "latency_claimed",
            "attention_chart_structure_claimed",
            "basis_invariant_semantics_claimed",
            "full_capture_sha256_intentionally_omitted_for_heldout_firewall",
            "published_A_anchor_authenticated",
            "fisher_quantity_semantics",
            "grouping_statistic",
            "tied_physical_storage_counted_once",
            "ingress_and_readout_role_atlases_are_separate",
            "artifact_sha256",
        }
        _exact_keys(value, expected, label="lexical mode dictionary")
        configuration = value["configuration"]
        profile = value["profile_receipt"]
        null = value["null_receipt"]
        candidates = value["candidates"]
        roles = value["roles_read"]
        if (
            not isinstance(configuration, Mapping)
            or not isinstance(profile, Mapping)
            or not isinstance(null, Mapping)
            or not isinstance(candidates, list)
            or not all(isinstance(item, Mapping) for item in candidates)
            or not isinstance(roles, list)
        ):
            raise ValueError("lexical mode dictionary nested fields are invalid")
        parsed_config = LexicalModeCompilerConfig.from_dict(configuration)
        if value["configuration_sha256"] != parsed_config.config_sha256:
            raise ValueError("lexical mode configuration hash mismatch")
        return cls(
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
            exact_control_evidence_sha256=str(
                value["exact_control_evidence_sha256"]
            ),
            exact_lift_receipt_sha256=str(value["exact_lift_receipt_sha256"]),
            source_envelope_sha256=str(value["source_envelope_sha256"]),
            atom_universe_sha256=str(value["atom_universe_sha256"]),
            A_role_sha256=str(value["A_role_sha256"]),
            A_row_ids_sha256=str(value["A_row_ids_sha256"]),
            lexical_axis_sha256=str(value["lexical_axis_sha256"]),
            configuration=parsed_config,
            profile_receipt=LexicalModeProfileReceipt.from_dict(profile),
            null_receipt=LexicalModeNullReceipt.from_dict(null),
            candidates=tuple(
                LexicalModeCandidate.from_dict(item) for item in candidates
            ),
            roles_read=tuple(str(item) for item in roles),
            B_rows_read=_require_int(value["B_rows_read"], label="B_rows_read"),
            C_rows_read=_require_int(value["C_rows_read"], label="C_rows_read"),
            physical_basis_kind=str(value["physical_basis_kind"]),
            physical_basis_count=_require_int(
                value["physical_basis_count"], label="physical_basis_count"
            ),
            physical_basis_sha256=str(value["physical_basis_sha256"]),
            candidate_semantics=str(value["candidate_semantics"]),
            profile_abi=str(value["profile_abi"]),
            null_abi=str(value["null_abi"]),
            compression_attempted=_require_bool(
                value["compression_attempted"], label="compression_attempted"
            ),
            compression_claimed=_require_bool(
                value["compression_claimed"], label="compression_claimed"
            ),
            semantic_confirmation_claimed=_require_bool(
                value["semantic_confirmation_claimed"],
                label="semantic_confirmation_claimed",
            ),
            parameter_savings_claimed=_require_bool(
                value["parameter_savings_claimed"],
                label="parameter_savings_claimed",
            ),
            mac_savings_claimed=_require_bool(
                value["mac_savings_claimed"], label="mac_savings_claimed"
            ),
            latency_claimed=_require_bool(
                value["latency_claimed"], label="latency_claimed"
            ),
            attention_chart_structure_claimed=_require_bool(
                value["attention_chart_structure_claimed"],
                label="attention_chart_structure_claimed",
            ),
            basis_invariant_semantics_claimed=_require_bool(
                value["basis_invariant_semantics_claimed"],
                label="basis_invariant_semantics_claimed",
            ),
            full_capture_sha256_intentionally_omitted_for_heldout_firewall=(
                _require_bool(
                    value[
                        "full_capture_sha256_intentionally_omitted_for_heldout_firewall"
                    ],
                    label=(
                        "full_capture_sha256_intentionally_omitted_for_heldout_firewall"
                    ),
                )
            ),
            published_A_anchor_authenticated=_require_bool(
                value["published_A_anchor_authenticated"],
                label="published_A_anchor_authenticated",
            ),
            fisher_quantity_semantics=str(value["fisher_quantity_semantics"]),
            grouping_statistic=str(value["grouping_statistic"]),
            tied_physical_storage_counted_once=_require_bool(
                value["tied_physical_storage_counted_once"],
                label="tied_physical_storage_counted_once",
            ),
            ingress_and_readout_role_atlases_are_separate=_require_bool(
                value["ingress_and_readout_role_atlases_are_separate"],
                label="ingress_and_readout_role_atlases_are_separate",
            ),
            schema=str(value["schema"]),
            format_version=_require_int(
                value["format_version"], label="format_version"
            ),
            artifact_sha256=str(value["artifact_sha256"]),
        )


def _lexical_axis_indices(
    source: LexicalModeASource,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    coordinate_to_index: dict[tuple[int, int], int] = {}
    for index, entry in enumerate(source.response_axis):
        coordinate = (entry.global_atom_ordinal, entry.execution_ordinal)
        if coordinate in coordinate_to_index:
            raise ValueError("capture response axis repeats a lexical occurrence")
        coordinate_to_index[coordinate] = index
    ingress_coordinates = tuple((column, 0) for column in range(HIDDEN_WIDTH))
    readout_coordinates = tuple((column, 38) for column in range(HIDDEN_WIDTH))
    try:
        ingress = tuple(coordinate_to_index[item] for item in ingress_coordinates)
        readout = tuple(coordinate_to_index[item] for item in readout_coordinates)
    except KeyError as error:
        raise ValueError(
            "capture does not expose all 640 tied lexical columns at both roles"
        ) from error
    axis_sha256 = _json_sha256(
        {
            "ingress": [list(item) for item in ingress_coordinates],
            "readout": [list(item) for item in readout_coordinates],
        },
        domain=_AXIS_DOMAIN,
    )
    return ingress, readout, axis_sha256


def _A_rows(source: LexicalModeASource) -> tuple[object, ...]:
    return tuple(sorted(source.A_role.rows, key=lambda row: row.row_id))


def _center_and_weight(
    raw: np.ndarray,
    family_labels: np.ndarray,
    family_ids: tuple[str, ...],
) -> np.ndarray:
    centered = np.array(raw, dtype=np.float64, copy=True)
    for family in family_ids:
        indices = np.flatnonzero(family_labels == family)
        if len(indices) < 2:
            raise ValueError("each A family requires at least two rows")
        values = centered[indices]
        values -= values.mean(axis=0, keepdims=True)
        values /= math.sqrt(len(indices) - 1)
        centered[indices] = values
    return centered


def _normalize_joint_profile(
    ingress_centered: np.ndarray,
    readout_centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    ingress_energy = float(np.sum(np.square(ingress_centered)))
    readout_energy = float(np.sum(np.square(readout_centered)))
    if ingress_energy <= 1e-30 or readout_energy <= 1e-30:
        raise ValueError("a centered lexical role has zero response energy")
    ingress_profile = ingress_centered / math.sqrt(ingress_energy)
    readout_profile = readout_centered / math.sqrt(readout_energy)
    joint = np.concatenate((ingress_profile, readout_profile), axis=0) / math.sqrt(2.0)
    column_norms = np.linalg.norm(joint, axis=0)
    nonzero = column_norms > 1e-15
    joint_normalized = np.zeros_like(joint)
    joint_normalized[:, nonzero] = joint[:, nonzero] / column_norms[nonzero]
    return (
        ingress_profile,
        readout_profile,
        joint_normalized,
        ingress_energy,
        readout_energy,
    )


def _role_cosine_matrix(profile: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(profile, axis=0)
    normalized = np.zeros_like(profile)
    nonzero = norms > 1e-15
    normalized[:, nonzero] = profile[:, nonzero] / norms[nonzero]
    cosine = normalized.T @ normalized
    return np.clip(cosine, -1.0, 1.0)


def _profiles_for_indices(
    ingress_raw: np.ndarray,
    readout_raw: np.ndarray,
    family_labels: np.ndarray,
    family_ids: tuple[str, ...],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_families = family_labels[indices]
    present = tuple(family for family in family_ids if family in set(sub_families))
    ingress = _center_and_weight(ingress_raw[indices], sub_families, present)
    readout = _center_and_weight(readout_raw[indices], sub_families, present)
    ingress_profile, readout_profile, joint, *_ = _normalize_joint_profile(
        ingress, readout
    )
    return ingress_profile, readout_profile, joint


def _null_maxima(
    ingress_profile: np.ndarray,
    readout_profile: np.ndarray,
    family_labels: np.ndarray,
    family_ids: tuple[str, ...],
    *,
    count: int,
    seed: int,
) -> tuple[float, ...]:
    """Column-independent within-family shuffles, paired across lexical roles."""

    bit_generator = np.random.PCG64(seed)
    columns = np.arange(HIDDEN_WIDTH)[None, :]
    family_indices = tuple(
        np.flatnonzero(family_labels == family) for family in family_ids
    )
    if any(len(indices) != 6 for indices in family_indices):
        raise ValueError("the frozen A null requires six rows per family")
    maxima: list[float] = []
    for _ in range(count):
        keys = bit_generator.random_raw(
            (HIDDEN_WIDTH, len(family_ids), 6)
        )
        ordered_keys = np.sort(keys, axis=-1)
        if np.any(np.diff(ordered_keys, axis=-1) == 0):
            raise ValueError("lexical null random-key collision")
        orders = np.argsort(keys, axis=-1, kind="stable")
        shuffled_ingress = np.empty_like(ingress_profile)
        shuffled_readout = np.empty_like(readout_profile)
        for family_ordinal, indices in enumerate(family_indices):
            source_rows = indices[orders[:, family_ordinal, :]].T
            shuffled_ingress[indices, :] = ingress_profile[source_rows, columns]
            shuffled_readout[indices, :] = readout_profile[source_rows, columns]
        joint = np.concatenate(
            (shuffled_ingress, shuffled_readout), axis=0
        ) / math.sqrt(2.0)
        norms = np.linalg.norm(joint, axis=0)
        nonzero = norms > 1e-15
        normalized = np.zeros_like(joint)
        normalized[:, nonzero] = joint[:, nonzero] / norms[nonzero]
        cosine = normalized.T @ normalized
        upper = np.triu_indices(HIDDEN_WIDTH, k=1)
        maxima.append(float(np.max(np.abs(cosine[upper]))))
    return tuple(sorted(maxima))


def _partition_for_members(
    members: tuple[int, ...],
    boundary: ExecutionBoundary,
) -> ExactLexicalRolePartition:
    catalog = build_gemma3_parts_catalog()
    member_set = set(members)
    complement = tuple(
        column for column in range(HIDDEN_WIDTH) if column not in member_set
    )
    member_atoms = tuple(
        PhysicalAtomId(PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN, column)
        for column in members
    )
    complement_atoms = tuple(
        PhysicalAtomId(PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN, column)
        for column in complement
    )
    blob, refs = build_atom_group_sidecar(
        catalog,
        (
            AtomGroupDraft(
                group_id=f"lexical_subset.{boundary.execution_ordinal}",
                boundary=boundary,
                atom_ids=member_atoms,
                group_semantics="exact_partition",
            ),
            AtomGroupDraft(
                group_id=f"lexical_complement.{boundary.execution_ordinal}",
                boundary=boundary,
                atom_ids=complement_atoms,
                group_semantics="exact_partition",
            ),
        ),
    )
    validate_exact_atom_partition(
        catalog,
        refs,
        blob,
        atom_family=PhysicalAtomFamily.TIED_LEXICAL_HIDDEN_COLUMN,
        boundary=boundary,
    )
    return ExactLexicalRolePartition(
        boundary_id=boundary.boundary_id,
        execution_ordinal=boundary.execution_ordinal,
        subset_columns=members,
        complement_columns=complement,
        membership_blob_sha256=refs[0].blob_sha256,
        subset_slice_sha256=refs[0].slice_sha256,
        complement_slice_sha256=refs[1].slice_sha256,
    )


def _minimum_pair_cosine(matrix: np.ndarray, members: tuple[int, ...]) -> float:
    values = [
        float(matrix[left, right])
        for offset, left in enumerate(members)
        for right in members[offset + 1 :]
    ]
    if not values:
        raise ValueError("a candidate requires at least one pair")
    return min(values)


def _stable_pair_candidates(
    *,
    joint_cosine: np.ndarray,
    ingress_cosine: np.ndarray,
    readout_cosine: np.ndarray,
    parity_cosines: tuple[np.ndarray, np.ndarray],
    lofo_cosines: tuple[np.ndarray, ...],
    ingress_raw: np.ndarray,
    readout_raw: np.ndarray,
    family_labels: np.ndarray,
    family_ids: tuple[str, ...],
    null_receipt: LexicalModeNullReceipt,
    config: LexicalModeCompilerConfig,
) -> tuple[tuple[int, int, tuple[float, float], tuple[float, ...], int], ...]:
    results: list[tuple[int, int, tuple[float, float], tuple[float, ...], int]] = []
    threshold = null_receipt.effective_joint_similarity_floor
    for left in range(HIDDEN_WIDTH):
        for right in range(left + 1, HIDDEN_WIDTH):
            if float(joint_cosine[left, right]) < threshold:
                continue
            parity = tuple(
                float(matrix[left, right]) for matrix in parity_cosines
            )
            if min(parity) < config.role_stability_floor:
                continue
            lofo = tuple(float(matrix[left, right]) for matrix in lofo_cosines)
            if sum(value >= config.role_stability_floor for value in lofo) < (
                config.minimum_lofo_pass_count
            ):
                continue
            family_energies = []
            for family in family_ids:
                indices = np.flatnonzero(family_labels == family)
                energy = float(
                    np.sum(np.square(ingress_raw[np.ix_(indices, (left, right))]))
                    + np.sum(np.square(readout_raw[np.ix_(indices, (left, right))]))
                )
                family_energies.append(energy)
            maximum = max(family_energies)
            active = sum(
                energy >= config.active_family_relative_energy_floor * maximum
                for energy in family_energies
            )
            if active < config.minimum_active_family_count:
                continue
            if max(
                float(ingress_cosine[left, right]),
                float(readout_cosine[left, right]),
            ) < config.role_stability_floor:
                continue
            results.append((left, right, parity, lofo, active))
    return tuple(results)


def _all_stable_cliques(
    stable_pairs: Sequence[
        tuple[int, int, tuple[float, float], tuple[float, ...], int]
    ],
    *,
    maximum_size: int,
    maximum_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every stable clique of size 2..``maximum_size``.

    Keeping only maximal cliques is unsafe here: a larger clique can fail an
    aggregate LOFO or family-support gate even though one or more of its
    proper subcliques pass.  Lexicographic extension emits each subclique once
    and makes candidate identity independent of set iteration order.
    """

    adjacency: dict[int, set[int]] = {}
    for left, right, *_ in stable_pairs:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    cliques: list[tuple[int, ...]] = []

    def extend(prefix: tuple[int, ...], prospective: tuple[int, ...]) -> None:
        for offset, vertex in enumerate(prospective):
            candidate = (*prefix, vertex)
            if len(candidate) >= 2:
                cliques.append(candidate)
                if len(cliques) > maximum_count:
                    raise ValueError(
                        "stable lexical clique inventory exceeds "
                        "maximum_candidate_count"
                    )
            if len(candidate) == maximum_size:
                continue
            neighbors = adjacency.get(vertex, set())
            following = tuple(
                item
                for item in prospective[offset + 1 :]
                if item in neighbors
            )
            if following:
                extend(candidate, following)

    extend((), tuple(sorted(adjacency)))
    return tuple(cliques)


def _role_edge_passes(
    left: int,
    right: int,
    *,
    full_cosine: np.ndarray,
    parity_cosines: Sequence[np.ndarray],
    lofo_cosines: Sequence[np.ndarray],
    config: LexicalModeCompilerConfig,
) -> bool:
    return (
        float(full_cosine[left, right]) >= config.role_stability_floor
        and min(float(matrix[left, right]) for matrix in parity_cosines)
        >= config.role_stability_floor
        and sum(
            float(matrix[left, right]) >= config.role_stability_floor
            for matrix in lofo_cosines
        )
        >= config.minimum_lofo_pass_count
    )


def compile_lexical_mode_dictionary(
    source: LexicalModeASource,
    config: LexicalModeCompilerConfig | None = None,
) -> LexicalModeDictionary:
    """Compile a frozen lexical-mode dictionary without reading B or C."""

    if type(source) is not LexicalModeASource:
        raise TypeError("lexical discovery requires an exact redacted A source")
    configuration = LexicalModeCompilerConfig() if config is None else config
    if type(configuration) is not LexicalModeCompilerConfig:
        raise TypeError("lexical discovery configuration is invalid")
    ingress_indices, readout_indices, axis_sha256 = _lexical_axis_indices(source)
    rows = _A_rows(source)
    A_role = source.A_role
    family_ids = tuple(sorted({row.family_id for row in rows}))
    if len(family_ids) != 6:
        raise ValueError("lexical discovery requires the frozen six-family A panel")
    family_labels = np.asarray([row.family_id for row in rows], dtype=object)
    responses = np.asarray([row.atom_responses for row in rows], dtype=np.float64)
    ingress_raw = responses[:, ingress_indices]
    readout_raw = responses[:, readout_indices]
    ingress_centered = _center_and_weight(ingress_raw, family_labels, family_ids)
    readout_centered = _center_and_weight(readout_raw, family_labels, family_ids)
    (
        ingress_profile,
        readout_profile,
        joint_profile,
        ingress_centered_energy,
        readout_centered_energy,
    ) = _normalize_joint_profile(ingress_centered, readout_centered)
    family_counts = tuple(
        (family, int(np.sum(family_labels == family))) for family in family_ids
    )
    family_energy_rows = []
    for family in family_ids:
        indices = np.flatnonzero(family_labels == family)
        family_energy_rows.append(
            (
                family,
                float(np.sum(np.square(ingress_raw[indices]))),
                float(np.sum(np.square(readout_raw[indices]))),
                float(np.sum(np.square(ingress_centered[indices]))),
                float(np.sum(np.square(readout_centered[indices]))),
            )
        )
    profile_receipt = LexicalModeProfileReceipt(
        row_count=len(rows),
        family_ids=family_ids,
        family_row_counts=family_counts,
        raw_ingress_empirical_fisher_mass=float(np.mean(np.square(ingress_raw))),
        raw_readout_empirical_fisher_mass=float(np.mean(np.square(readout_raw))),
        centered_ingress_frobenius_energy=ingress_centered_energy,
        centered_readout_frobenius_energy=readout_centered_energy,
        family_energy_rows=tuple(family_energy_rows),
        ingress_profile_sha256=_array_sha256(
            ingress_profile, label="ingress_profile"
        ),
        readout_profile_sha256=_array_sha256(
            readout_profile, label="readout_profile"
        ),
        joint_profile_sha256=_array_sha256(joint_profile, label="joint_profile"),
    )
    maxima = _null_maxima(
        ingress_profile,
        readout_profile,
        family_labels,
        family_ids,
        count=configuration.permutation_count,
        seed=configuration.permutation_seed,
    )
    q95 = _nearest_rank_quantile(maxima, configuration.null_quantile)
    null_receipt = LexicalModeNullReceipt(
        permutation_count=configuration.permutation_count,
        permutation_seed=configuration.permutation_seed,
        null_quantile=configuration.null_quantile,
        sorted_max_absolute_cosines=maxima,
        max_absolute_cosine_q95=q95,
        signed_similarity_floor=configuration.signed_similarity_floor,
        effective_joint_similarity_floor=max(
            configuration.signed_similarity_floor, q95
        ),
        numpy_version=np.__version__,
        numpy_build_config_sha256=_numpy_build_config_sha256(),
    )
    joint_cosine = np.clip(joint_profile.T @ joint_profile, -1.0, 1.0)
    ingress_cosine = _role_cosine_matrix(ingress_profile)
    readout_cosine = _role_cosine_matrix(readout_profile)
    parity_indices: list[np.ndarray] = []
    for parity in (0, 1):
        selected: list[int] = []
        for family in family_ids:
            indices = np.flatnonzero(family_labels == family)
            selected.extend(indices[parity::2].tolist())
        parity_indices.append(np.asarray(selected, dtype=np.int64))
    parity_profiles = tuple(
        _profiles_for_indices(
            ingress_raw,
            readout_raw,
            family_labels,
            family_ids,
            indices,
        )
        for indices in parity_indices
    )
    parity_ingress_cosines = tuple(
        _role_cosine_matrix(profiles[0]) for profiles in parity_profiles
    )
    parity_readout_cosines = tuple(
        _role_cosine_matrix(profiles[1]) for profiles in parity_profiles
    )
    parity_cosines = tuple(
        np.clip(profiles[2].T @ profiles[2], -1.0, 1.0)
        for profiles in parity_profiles
    )
    lofo_profiles = tuple(
        _profiles_for_indices(
            ingress_raw,
            readout_raw,
            family_labels,
            family_ids,
            np.flatnonzero(family_labels != held_family),
        )
        for held_family in family_ids
    )
    lofo_ingress_cosines = tuple(
        _role_cosine_matrix(profiles[0]) for profiles in lofo_profiles
    )
    lofo_readout_cosines = tuple(
        _role_cosine_matrix(profiles[1]) for profiles in lofo_profiles
    )
    lofo_cosines = tuple(
        np.clip(profiles[2].T @ profiles[2], -1.0, 1.0)
        for profiles in lofo_profiles
    )
    stable_pairs = _stable_pair_candidates(
        joint_cosine=joint_cosine,
        ingress_cosine=ingress_cosine,
        readout_cosine=readout_cosine,
        parity_cosines=parity_cosines,  # type: ignore[arg-type]
        lofo_cosines=lofo_cosines,
        ingress_raw=ingress_centered,
        readout_raw=readout_centered,
        family_labels=family_labels,
        family_ids=family_ids,
        null_receipt=null_receipt,
        config=configuration,
    )
    ingress_pairs = tuple(
        pair
        for pair in stable_pairs
        if _role_edge_passes(
            pair[0],
            pair[1],
            full_cosine=ingress_cosine,
            parity_cosines=parity_ingress_cosines,
            lofo_cosines=lofo_ingress_cosines,
            config=configuration,
        )
    )
    readout_pairs = tuple(
        pair
        for pair in stable_pairs
        if _role_edge_passes(
            pair[0],
            pair[1],
            full_cosine=readout_cosine,
            parity_cosines=parity_readout_cosines,
            lofo_cosines=lofo_readout_cosines,
            config=configuration,
        )
    )
    ingress_cliques = _all_stable_cliques(
        ingress_pairs,
        maximum_size=configuration.maximum_clique_size,
        maximum_count=configuration.maximum_candidate_count,
    )
    readout_cliques = _all_stable_cliques(
        readout_pairs,
        maximum_size=configuration.maximum_clique_size,
        maximum_count=configuration.maximum_candidate_count,
    )
    cliques = tuple(sorted(set(ingress_cliques).union(readout_cliques)))
    if len(cliques) > configuration.maximum_candidate_count:
        raise ValueError("lexical candidate count exceeds the preregistered bound")
    raw_ingress_column_mass = np.mean(np.square(ingress_raw), axis=0)
    raw_readout_column_mass = np.mean(np.square(readout_raw), axis=0)
    ingress_total = float(np.sum(raw_ingress_column_mass))
    readout_total = float(np.sum(raw_readout_column_mass))
    candidates = []
    for ordinal, members in enumerate(cliques):
        ingress_value = _minimum_pair_cosine(ingress_cosine, members)
        readout_value = _minimum_pair_cosine(readout_cosine, members)
        ingress_parity = tuple(
            _minimum_pair_cosine(matrix, members)
            for matrix in parity_ingress_cosines
        )
        readout_parity = tuple(
            _minimum_pair_cosine(matrix, members)
            for matrix in parity_readout_cosines
        )
        ingress_lofo = tuple(
            _minimum_pair_cosine(matrix, members)
            for matrix in lofo_ingress_cosines
        )
        readout_lofo = tuple(
            _minimum_pair_cosine(matrix, members)
            for matrix in lofo_readout_cosines
        )
        ingress_supported = (
            ingress_value >= configuration.role_stability_floor
            and min(ingress_parity) >= configuration.role_stability_floor
            and sum(
                value >= configuration.role_stability_floor
                for value in ingress_lofo
            )
            >= configuration.minimum_lofo_pass_count
        )
        readout_supported = (
            readout_value >= configuration.role_stability_floor
            and min(readout_parity) >= configuration.role_stability_floor
            and sum(
                value >= configuration.role_stability_floor
                for value in readout_lofo
            )
            >= configuration.minimum_lofo_pass_count
        )
        if ingress_supported and readout_supported:
            role_scope = "dual_role"
        elif ingress_supported:
            role_scope = "lexical_ingress"
        elif readout_supported:
            role_scope = "lexical_readout"
        else:
            continue
        parity = tuple(
            _minimum_pair_cosine(matrix, members) for matrix in parity_cosines
        )
        lofo = tuple(
            _minimum_pair_cosine(matrix, members) for matrix in lofo_cosines
        )
        family_energies = []
        for family in family_ids:
            indices = np.flatnonzero(family_labels == family)
            family_energies.append(
                float(
                    np.sum(
                        np.square(
                            ingress_centered[np.ix_(indices, members)]
                        )
                    )
                    + np.sum(
                        np.square(
                            readout_centered[np.ix_(indices, members)]
                        )
                    )
                )
            )
        maximum_family_energy = max(family_energies)
        active = sum(
            energy
            >= configuration.active_family_relative_energy_floor
            * maximum_family_energy
            for energy in family_energies
        )
        if active < configuration.minimum_active_family_count:
            continue
        balanced_mass = 0.5 * float(
            np.sum(raw_ingress_column_mass[list(members)]) / ingress_total
            + np.sum(raw_readout_column_mass[list(members)]) / readout_total
        )
        candidates.append(
            LexicalModeCandidate(
                candidate_id=f"lexical.mode_candidate.{ordinal:04d}",
                member_columns=members,
                joint_min_signed_cosine=_minimum_pair_cosine(
                    joint_cosine, members
                ),
                ingress_min_signed_cosine=ingress_value,
                readout_min_signed_cosine=readout_value,
                balanced_empirical_fisher_mass=balanced_mass,
                role_scope=role_scope,
                parity_joint_signed_cosines=parity,
                lofo_joint_signed_cosines=lofo,
                ingress_parity_signed_cosines=ingress_parity,
                readout_parity_signed_cosines=readout_parity,
                ingress_lofo_signed_cosines=ingress_lofo,
                readout_lofo_signed_cosines=readout_lofo,
                lofo_pass_count=sum(
                    value >= configuration.role_stability_floor for value in lofo
                ),
                active_family_count=active,
                ingress_partition=_partition_for_members(
                    members, ExecutionBoundary.lexical_ingress()
                ),
                readout_partition=_partition_for_members(
                    members, ExecutionBoundary.lexical_readout()
                ),
            )
        )
    candidates.sort(key=lambda item: item.candidate_id)
    return LexicalModeDictionary(
        source_kind=source.source_kind,
        model_id=source.model_id,
        source_model_fingerprint_sha256=source.source_model_fingerprint_sha256,
        source_execution_fingerprint_sha256=(
            source.source_execution_fingerprint_sha256
        ),
        score_objective_id=source.score_objective_id,
        chart_surface_id=source.chart_surface_id,
        protocol_sha256=source.protocol_sha256,
        exact_control_evidence_sha256=(
            source.exact_lift_receipt.exact_control_evidence_sha256
        ),
        exact_lift_receipt_sha256=_json_sha256(
            source.exact_lift_receipt.to_dict(), domain=_EXACT_LIFT_DOMAIN
        ),
        source_envelope_sha256=source.A_source_sha256,
        atom_universe_sha256=source.atom_universe_sha256,
        A_role_sha256=A_role.role_sha256,
        A_row_ids_sha256=A_role.row_ids_sha256,
        lexical_axis_sha256=axis_sha256,
        configuration=configuration,
        profile_receipt=profile_receipt,
        null_receipt=null_receipt,
        candidates=tuple(candidates),
        physical_basis_sha256=_json_sha256(
            list(range(HIDDEN_WIDTH)), domain=_PARTITION_DOMAIN
        ),
        published_A_anchor_authenticated=(
            source.published_A_anchor_authenticated
        ),
    )


def validate_lexical_mode_dictionary(
    dictionary: LexicalModeDictionary,
    source: LexicalModeASource,
) -> LexicalModeDictionary:
    """Replay A exactly and require byte-equivalent structured output."""

    if type(dictionary) is not LexicalModeDictionary:
        raise TypeError("lexical dictionary must be an exact typed receipt")
    if dictionary.published_A_anchor_authenticated != (
        source.published_A_anchor_authenticated
    ):
        raise ValueError("lexical dictionary A-anchor status drifted")
    replay = compile_lexical_mode_dictionary(source, dictionary.configuration)
    if dictionary.to_dict() != replay.to_dict():
        raise ValueError("lexical mode dictionary does not replay from source A")
    return dictionary


def compile_lexical_mode_dictionary_from_capture(
    captures: PartsFirstCaptureBundle,
    config: LexicalModeCompilerConfig | None = None,
    *,
    authenticate_published_A: bool = False,
) -> LexicalModeDictionary:
    source = build_lexical_mode_A_source(
        captures,
        authenticate_published_A=authenticate_published_A,
    )
    return compile_lexical_mode_dictionary(source, config)


def parse_lexical_mode_dictionary_json(
    source: str | bytes,
    *,
    A_source: LexicalModeASource,
) -> LexicalModeDictionary:
    value = _strict_json_object(source, label="lexical mode dictionary")
    dictionary = LexicalModeDictionary.from_dict(value)
    return validate_lexical_mode_dictionary(dictionary, A_source)


def _published_config() -> LexicalModeCompilerConfig:
    configuration = LexicalModeCompilerConfig()
    if configuration.config_sha256 != PINNED_LEXICAL_MODE_CONFIG_SHA256:
        raise ValueError("published lexical mode configuration pin drifted")
    return configuration


def _require_published_dictionary(
    dictionary: LexicalModeDictionary,
) -> LexicalModeDictionary:
    if dictionary.published_A_anchor_authenticated is not True:
        raise ValueError("published lexical mode dictionary is unauthenticated")
    if dictionary.configuration.config_sha256 != (
        PINNED_LEXICAL_MODE_CONFIG_SHA256
    ):
        raise ValueError("published lexical mode configuration pin drifted")
    if dictionary.artifact_sha256 != PINNED_LEXICAL_MODE_ARTIFACT_SHA256:
        raise ValueError("published lexical mode artifact pin drifted")
    return dictionary


def compile_published_lexical_mode_dictionary(
    capture: Path | str = DEFAULT_LEXICAL_MODE_CAPTURE,
) -> LexicalModeDictionary:
    """Compile only the canonical authenticated A source and frozen config."""

    source = load_published_lexical_mode_A_source(capture)
    dictionary = compile_lexical_mode_dictionary(source, _published_config())
    return _require_published_dictionary(dictionary)


def _load_lexical_mode_dictionary_path(
    source: Path,
    *,
    A_source: LexicalModeASource,
) -> LexicalModeDictionary:
    try:
        encoded = source.read_bytes()
    except OSError as error:
        raise ValueError("lexical mode dictionary is unavailable") from error
    return parse_lexical_mode_dictionary_json(encoded, A_source=A_source)


def load_lexical_mode_dictionary(
    source: Path | str,
    *,
    A_source: LexicalModeASource,
) -> LexicalModeDictionary:
    """Strict-load a dictionary and replay it from the supplied redacted A."""

    path = _require_json_path(source, label="lexical mode dictionary")
    return _load_lexical_mode_dictionary_path(path, A_source=A_source)


def load_published_lexical_mode_dictionary(
    source: Path | str = DEFAULT_LEXICAL_MODE_OUTPUT,
    *,
    capture: Path | str = DEFAULT_LEXICAL_MODE_CAPTURE,
) -> LexicalModeDictionary:
    """Strict-load the pinned artifact and replay it from a fresh canonical A."""

    A_source = load_published_lexical_mode_A_source(capture)
    dictionary = load_lexical_mode_dictionary(source, A_source=A_source)
    return _require_published_dictionary(dictionary)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_publication_paths(
    capture: Path | str,
    output: Path | str,
) -> tuple[Path, Path]:
    capture_path = _require_json_path(capture, label="lexical mode capture")
    output_path = _require_json_path(output, label="lexical mode output")
    if capture_path.resolve(strict=False) == output_path.resolve(strict=False):
        raise ValueError("lexical mode capture and output paths must differ")
    return capture_path, output_path


def write_published_lexical_mode_dictionary(
    dictionary: LexicalModeDictionary,
    destination: Path | str = DEFAULT_LEXICAL_MODE_OUTPUT,
    *,
    capture: Path | str = DEFAULT_LEXICAL_MODE_CAPTURE,
) -> LexicalModeDictionary:
    """Publish with a strict same-directory, no-clobber durability protocol."""

    if type(dictionary) is not LexicalModeDictionary:
        raise TypeError("published lexical writer requires an exact dictionary")
    capture_path, target = _validate_publication_paths(capture, destination)
    A_source = load_published_lexical_mode_A_source(capture_path)
    validate_lexical_mode_dictionary(dictionary, A_source)
    _require_published_dictionary(dictionary)
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
                dictionary.to_dict(),
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
                    "published lexical dictionary descriptor identity drifted"
                )
        loaded = _load_lexical_mode_dictionary_path(
            temporary,
            A_source=A_source,
        )
        if loaded != dictionary:
            raise RuntimeError("published lexical dictionary changed on temp reload")
        current_temporary = temporary.lstat()
        if temporary_stat is None or not os.path.samestat(
            temporary_stat,
            current_temporary,
        ):
            raise RuntimeError(
                "published lexical dictionary temporary identity drifted"
            )
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite a lexical mode dictionary"
            ) from error
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 2
        ):
            raise RuntimeError("published lexical dictionary hardlink identity drifted")
        current_temporary = temporary.lstat()
        if not os.path.samestat(temporary_stat, current_temporary):
            raise RuntimeError(
                "published lexical dictionary temporary changed before unlink"
            )
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
            raise RuntimeError("published lexical dictionary identity drifted")
        canonical = _load_lexical_mode_dictionary_path(
            target,
            A_source=A_source,
        )
        if canonical != dictionary:
            raise RuntimeError(
                "published lexical dictionary changed on canonical reload"
            )
        published = target.lstat()
        if (
            not os.path.samestat(temporary_stat, published)
            or not stat.S_ISREG(published.st_mode)
            or stat.S_IMODE(published.st_mode) != 0o600
            or published.st_nlink != 1
        ):
            raise RuntimeError(
                "published lexical dictionary changed after canonical reload"
            )
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
                    "lexical dictionary temporary cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--write", action="store_true")
    parser.add_argument("--capture", type=Path, default=DEFAULT_LEXICAL_MODE_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_LEXICAL_MODE_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    capture, output = _validate_publication_paths(args.capture, args.output)
    dictionary = compile_published_lexical_mode_dictionary(capture)
    if args.write:
        write_published_lexical_mode_dictionary(
            dictionary,
            output,
            capture=capture,
        )
    print(
        json.dumps(
            {
                "artifact_sha256": dictionary.artifact_sha256,
                "candidate_count": len(dictionary.candidates),
                "capture": str(capture),
                "configuration_sha256": dictionary.configuration.config_sha256,
                "output": str(output),
                "published": bool(args.write),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
