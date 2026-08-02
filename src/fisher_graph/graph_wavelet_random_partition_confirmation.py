"""Fresh size-matched random controls for grouped graph-wavelet claims.

The controls in this module are deliberately *not* graph-topology
partitions.  Each control directly partitions the 64 frozen parent-mode IDs
into eight groups of eight by sorting domain-separated SHA-256 keys.  The
partition can be passed to the local-SVD grouped-basis fitter through its
small runtime interface, while graph-Fourier use is rejected explicitly.

The confirmation panel fixes 63 controls.  With the conservative plus-one
empirical p-value used below, that gives a denominator of 64 and makes a
pre-registered ``p <= 0.05`` gate possible without asymptotic assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from statistics import median
from typing import Protocol, Sequence, runtime_checkable

from torch import Tensor

from .graph_wavelet_grouped_basis import _tensor_sha256


__all__ = [
    "CONFIRMATION_FAMILY_COUNT",
    "CONFIRMATION_GROUP_COUNT",
    "CONFIRMATION_GROUP_SIZE",
    "CONFIRMATION_PARENT_RANK",
    "PRIMARY_NULL_GATES",
    "RANDOM_CONTROL_COUNT",
    "BalancedRandomModePartition",
    "BalancedRandomPartitionPanel",
    "LocalSVDPartition",
    "RandomPartitionNullGates",
    "RandomPartitionNullPanelStatistics",
    "derive_balanced_random_mode_partition",
    "derive_balanced_random_partition_panel",
    "evaluate_random_partition_null_panel",
    "grouped_parent_basis_sha256",
]


CONFIRMATION_PARENT_RANK = 64
CONFIRMATION_GROUP_COUNT = 8
CONFIRMATION_GROUP_SIZE = 8
CONFIRMATION_FAMILY_COUNT = 8
RANDOM_CONTROL_COUNT = 63

_FORMAT_VERSION = 1
_CONTROL_KIND = "fisher_graph.balanced_random_mode_partition_control"
_PANEL_KIND = "fisher_graph.balanced_random_partition_confirmation_panel"
_STATISTICS_KIND = "fisher_graph.random_partition_null_panel_statistics"
_CONTROL_DOMAIN = b"fisher-graph:balanced-random-mode-partition:v1\0"
_MODE_KEY_DOMAIN = b"fisher-graph:balanced-random-mode-key:v1\0"
_PANEL_DOMAIN = b"fisher-graph:balanced-random-partition-panel:v1\0"
_STATISTICS_DOMAIN = b"fisher-graph:random-partition-null-statistics:v1\0"
_CONTROL_ALGORITHM = (
    "domain_separated_sha256_key_sort_of_mode_ids_then_balanced_chunks"
)
_CONTROL_ROLE = "size_matched_random_control_not_graph_topology"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AGGREGATE_RELATIVE_TOLERANCE = 1.0e-12
_AGGREGATE_ABSOLUTE_TOLERANCE = 1.0e-12


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_finite_nonnegative(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _canonical_groups(
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    try:
        return tuple(
            sorted(tuple(sorted(group)) for group in groups)
        )
    except TypeError as error:
        raise ValueError("groups contain non-orderable mode IDs") from error


def _validate_balanced_groups(
    groups: tuple[tuple[int, ...], ...],
    *,
    label: str,
) -> None:
    if (
        len(groups) != CONFIRMATION_GROUP_COUNT
        or any(len(group) != CONFIRMATION_GROUP_SIZE for group in groups)
        or any(
            isinstance(member, bool) or not isinstance(member, int)
            for group in groups
            for member in group
        )
    ):
        raise ValueError(
            f"{label} must be eight canonical groups partitioning modes 0..63"
        )
    if (
        any(tuple(sorted(group)) != group for group in groups)
        or tuple(sorted(groups)) != groups
        or tuple(sorted(member for group in groups for member in group))
        != tuple(range(CONFIRMATION_PARENT_RANK))
    ):
        raise ValueError(
            f"{label} must be eight canonical groups partitioning modes 0..63"
        )


def grouped_parent_basis_sha256(parent_basis: Tensor) -> str:
    """Return the exact parent-basis binding used by the grouped fitter."""

    return _tensor_sha256(parent_basis)


def _mode_sort_key(
    *,
    candidate_artifact_sha256: str,
    parent_basis_sha256: str,
    control_ordinal: int,
    mode_id: int,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(_MODE_KEY_DOMAIN)
    digest.update(bytes.fromhex(candidate_artifact_sha256))
    digest.update(bytes.fromhex(parent_basis_sha256))
    digest.update(control_ordinal.to_bytes(8, byteorder="big", signed=False))
    digest.update(mode_id.to_bytes(8, byteorder="big", signed=False))
    return digest.digest()


def _derive_groups(
    *,
    candidate_artifact_sha256: str,
    parent_basis_sha256: str,
    control_ordinal: int,
) -> tuple[tuple[int, ...], ...]:
    ordered_modes = sorted(
        range(CONFIRMATION_PARENT_RANK),
        key=lambda mode_id: (
            _mode_sort_key(
                candidate_artifact_sha256=candidate_artifact_sha256,
                parent_basis_sha256=parent_basis_sha256,
                control_ordinal=control_ordinal,
                mode_id=mode_id,
            ),
            mode_id,
        ),
    )
    return _canonical_groups(
        tuple(
            tuple(ordered_modes[start : start + CONFIRMATION_GROUP_SIZE])
            for start in range(
                0,
                CONFIRMATION_PARENT_RANK,
                CONFIRMATION_GROUP_SIZE,
            )
        )
    )


@runtime_checkable
class LocalSVDPartition(Protocol):
    """Small partition interface consumed by local grouped-SVD fitting."""

    groups: tuple[tuple[int, ...], ...]
    parent_rank: int
    group_count: int
    parent_basis_sha256: str
    artifact_sha256: str

    def validate_integrity(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BalancedRandomModePartition:
    """One direct balanced random partition, with no topology claim."""

    control_ordinal: int
    groups: tuple[tuple[int, ...], ...]
    candidate_artifact_sha256: str
    parent_basis_sha256: str
    artifact_sha256: str
    parent_rank: int = CONFIRMATION_PARENT_RANK
    group_count: int = CONFIRMATION_GROUP_COUNT
    group_size: int = CONFIRMATION_GROUP_SIZE
    random_control: bool = True
    topology_partition: bool = False
    topology_used: bool = False
    response_values_used: bool = False
    partition_role: str = _CONTROL_ROLE
    algorithm: str = _CONTROL_ALGORITHM
    artifact_kind: str = _CONTROL_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        self.validate_integrity()

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.groups)

    @property
    def projected_laplacian(self) -> Tensor:
        raise ValueError(
            "random partition controls have no graph Laplacian; "
            "use method='wavelet_local_svd'"
        )

    def _payload(self) -> dict[str, object]:
        return {
            "control_ordinal": self.control_ordinal,
            "groups": self.groups,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "parent_basis_sha256": self.parent_basis_sha256,
            "parent_rank": self.parent_rank,
            "group_count": self.group_count,
            "group_size": self.group_size,
            "random_control": self.random_control,
            "topology_partition": self.topology_partition,
            "topology_used": self.topology_used,
            "response_values_used": self.response_values_used,
            "partition_role": self.partition_role,
            "algorithm": self.algorithm,
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    def validate_integrity(self) -> None:
        _require_sha256(
            self.candidate_artifact_sha256,
            label="candidate_artifact_sha256",
        )
        _require_sha256(
            self.parent_basis_sha256,
            label="parent_basis_sha256",
        )
        if (
            isinstance(self.control_ordinal, bool)
            or not isinstance(self.control_ordinal, int)
            or not 0 <= self.control_ordinal < 2**64
            or self.parent_rank != CONFIRMATION_PARENT_RANK
            or self.group_count != CONFIRMATION_GROUP_COUNT
            or self.group_size != CONFIRMATION_GROUP_SIZE
            or self.random_control is not True
            or self.topology_partition is not False
            or self.topology_used is not False
            or self.response_values_used is not False
            or self.partition_role != _CONTROL_ROLE
            or self.algorithm != _CONTROL_ALGORITHM
            or self.artifact_kind != _CONTROL_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("random partition control fields are invalid")
        _validate_balanced_groups(self.groups, label="groups")
        expected = _derive_groups(
            candidate_artifact_sha256=self.candidate_artifact_sha256,
            parent_basis_sha256=self.parent_basis_sha256,
            control_ordinal=self.control_ordinal,
        )
        if self.groups != expected:
            raise ValueError("random partition differs from deterministic replay")
        if self.artifact_sha256 != _json_sha256(
            self._payload(),
            domain=_CONTROL_DOMAIN,
        ):
            raise ValueError("random partition artifact hash differs")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "group_sizes": self.group_sizes,
            "artifact_sha256": self.artifact_sha256,
            "graph_topology_claim": False,
        }


def derive_balanced_random_mode_partition(
    *,
    candidate_artifact_sha256: str,
    parent_basis_sha256: str,
    control_ordinal: int,
) -> BalancedRandomModePartition:
    """Derive one fresh partition without topology or response values."""

    _require_sha256(
        candidate_artifact_sha256,
        label="candidate_artifact_sha256",
    )
    _require_sha256(parent_basis_sha256, label="parent_basis_sha256")
    if (
        isinstance(control_ordinal, bool)
        or not isinstance(control_ordinal, int)
        or not 0 <= control_ordinal < 2**64
    ):
        raise ValueError("control_ordinal must be an unsigned 64-bit integer")
    groups = _derive_groups(
        candidate_artifact_sha256=candidate_artifact_sha256,
        parent_basis_sha256=parent_basis_sha256,
        control_ordinal=control_ordinal,
    )
    payload = {
        "control_ordinal": control_ordinal,
        "groups": groups,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "parent_basis_sha256": parent_basis_sha256,
        "parent_rank": CONFIRMATION_PARENT_RANK,
        "group_count": CONFIRMATION_GROUP_COUNT,
        "group_size": CONFIRMATION_GROUP_SIZE,
        "random_control": True,
        "topology_partition": False,
        "topology_used": False,
        "response_values_used": False,
        "partition_role": _CONTROL_ROLE,
        "algorithm": _CONTROL_ALGORITHM,
        "artifact_kind": _CONTROL_KIND,
        "format_version": _FORMAT_VERSION,
    }
    return BalancedRandomModePartition(
        control_ordinal=control_ordinal,
        groups=groups,
        candidate_artifact_sha256=candidate_artifact_sha256,
        parent_basis_sha256=parent_basis_sha256,
        artifact_sha256=_json_sha256(payload, domain=_CONTROL_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class BalancedRandomPartitionPanel:
    """A frozen 63-control panel bound to one native candidate."""

    candidate_artifact_sha256: str
    parent_basis_sha256: str
    native_partition_artifact_sha256: str
    native_groups: tuple[tuple[int, ...], ...]
    controls: tuple[BalancedRandomModePartition, ...]
    artifact_sha256: str
    control_count: int = RANDOM_CONTROL_COUNT
    controls_frozen_before_confirmation_values: bool = True
    controls_use_confirmation_values: bool = False
    controls_use_native_topology: bool = False
    artifact_kind: str = _PANEL_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "parent_basis_sha256": self.parent_basis_sha256,
            "native_partition_artifact_sha256": (
                self.native_partition_artifact_sha256
            ),
            "native_groups": self.native_groups,
            "control_artifact_sha256s": tuple(
                control.artifact_sha256 for control in self.controls
            ),
            "control_count": self.control_count,
            "controls_frozen_before_confirmation_values": (
                self.controls_frozen_before_confirmation_values
            ),
            "controls_use_confirmation_values": (
                self.controls_use_confirmation_values
            ),
            "controls_use_native_topology": self.controls_use_native_topology,
            "partition_geometry": {
                "parent_rank": CONFIRMATION_PARENT_RANK,
                "group_count": CONFIRMATION_GROUP_COUNT,
                "group_size": CONFIRMATION_GROUP_SIZE,
            },
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    def validate_integrity(self) -> None:
        _require_sha256(
            self.candidate_artifact_sha256,
            label="candidate_artifact_sha256",
        )
        _require_sha256(
            self.parent_basis_sha256,
            label="parent_basis_sha256",
        )
        _require_sha256(
            self.native_partition_artifact_sha256,
            label="native_partition_artifact_sha256",
        )
        _validate_balanced_groups(self.native_groups, label="native_groups")
        if (
            self.control_count != RANDOM_CONTROL_COUNT
            or len(self.controls) != RANDOM_CONTROL_COUNT
            or self.controls_frozen_before_confirmation_values is not True
            or self.controls_use_confirmation_values is not False
            or self.controls_use_native_topology is not False
            or self.artifact_kind != _PANEL_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("random partition panel fields are invalid")
        for ordinal, control in enumerate(self.controls):
            control.validate_integrity()
            if (
                control.control_ordinal != ordinal
                or control.candidate_artifact_sha256
                != self.candidate_artifact_sha256
                or control.parent_basis_sha256 != self.parent_basis_sha256
            ):
                raise ValueError("random control provenance or ordering differs")
        control_hashes = tuple(
            control.artifact_sha256 for control in self.controls
        )
        control_groups = tuple(control.groups for control in self.controls)
        if (
            len(set(control_hashes)) != RANDOM_CONTROL_COUNT
            or len(set(control_groups)) != RANDOM_CONTROL_COUNT
            or self.native_groups in control_groups
        ):
            raise ValueError(
                "random controls must be unique and distinct from native groups"
            )
        if self.artifact_sha256 != _json_sha256(
            self._payload(),
            domain=_PANEL_DOMAIN,
        ):
            raise ValueError("random partition panel artifact hash differs")

    @property
    def control_artifact_sha256s(self) -> tuple[str, ...]:
        return tuple(control.artifact_sha256 for control in self.controls)

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
            "random_controls_are_graph_topology_partitions": False,
            "native_topology_validation_is_modified": False,
        }


def derive_balanced_random_partition_panel(
    *,
    candidate_artifact_sha256: str,
    parent_basis_sha256: str,
    native_partition_artifact_sha256: str,
    native_groups: Sequence[Sequence[int]],
) -> BalancedRandomPartitionPanel:
    """Freeze the complete size-matched null panel before reading results."""

    canonical_native_groups = _canonical_groups(native_groups)
    _validate_balanced_groups(
        canonical_native_groups,
        label="native_groups",
    )
    controls = tuple(
        derive_balanced_random_mode_partition(
            candidate_artifact_sha256=candidate_artifact_sha256,
            parent_basis_sha256=parent_basis_sha256,
            control_ordinal=ordinal,
        )
        for ordinal in range(RANDOM_CONTROL_COUNT)
    )
    payload = {
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "parent_basis_sha256": parent_basis_sha256,
        "native_partition_artifact_sha256": (
            native_partition_artifact_sha256
        ),
        "native_groups": canonical_native_groups,
        "control_artifact_sha256s": tuple(
            control.artifact_sha256 for control in controls
        ),
        "control_count": RANDOM_CONTROL_COUNT,
        "controls_frozen_before_confirmation_values": True,
        "controls_use_confirmation_values": False,
        "controls_use_native_topology": False,
        "partition_geometry": {
            "parent_rank": CONFIRMATION_PARENT_RANK,
            "group_count": CONFIRMATION_GROUP_COUNT,
            "group_size": CONFIRMATION_GROUP_SIZE,
        },
        "artifact_kind": _PANEL_KIND,
        "format_version": _FORMAT_VERSION,
    }
    return BalancedRandomPartitionPanel(
        candidate_artifact_sha256=candidate_artifact_sha256,
        parent_basis_sha256=parent_basis_sha256,
        native_partition_artifact_sha256=(
            native_partition_artifact_sha256
        ),
        native_groups=canonical_native_groups,
        controls=controls,
        artifact_sha256=_json_sha256(payload, domain=_PANEL_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class RandomPartitionNullGates:
    """Pre-registered primary gates for the 63-control null panel."""

    maximum_empirical_p_value: float = 0.05
    minimum_median_sse_recovery_fraction: float = 0.05
    minimum_family_win_count: int = 7
    expected_control_count: int = RANDOM_CONTROL_COUNT
    expected_family_count: int = CONFIRMATION_FAMILY_COUNT

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_empirical_p_value)
            or not 0.0 < self.maximum_empirical_p_value <= 1.0
            or not math.isfinite(self.minimum_median_sse_recovery_fraction)
            or not 0.0 <= self.minimum_median_sse_recovery_fraction <= 1.0
            or isinstance(self.minimum_family_win_count, bool)
            or not isinstance(self.minimum_family_win_count, int)
            or not 0
            <= self.minimum_family_win_count
            <= CONFIRMATION_FAMILY_COUNT
            or self.expected_control_count != RANDOM_CONTROL_COUNT
            or self.expected_family_count != CONFIRMATION_FAMILY_COUNT
        ):
            raise ValueError("random partition null gates are invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "maximum_empirical_p_value": self.maximum_empirical_p_value,
            "minimum_median_sse_recovery_fraction": (
                self.minimum_median_sse_recovery_fraction
            ),
            "minimum_family_win_count": self.minimum_family_win_count,
            "expected_control_count": self.expected_control_count,
            "expected_family_count": self.expected_family_count,
            "empirical_p_tie_policy": "control_sse_less_than_or_equal_native",
            "family_win_tie_policy": "strict_native_sse_less_than_median_control",
        }


PRIMARY_NULL_GATES = RandomPartitionNullGates()


def _statistics_payload(
    *,
    panel_artifact_sha256: str,
    control_artifact_sha256s: tuple[str, ...],
    family_ids: tuple[str, ...],
    native_pooled_sse: float,
    control_pooled_sses: tuple[float, ...],
    native_family_sses: tuple[float, ...],
    control_family_sses: tuple[tuple[float, ...], ...],
    better_or_tied_control_count: int,
    empirical_p_value: float,
    median_control_pooled_sse: float,
    median_sse_recovery_fraction: float,
    median_control_family_sses: tuple[float, ...],
    family_wins: tuple[bool, ...],
    family_win_count: int,
    empirical_p_gate_passed: bool,
    median_sse_recovery_gate_passed: bool,
    family_win_gate_passed: bool,
    passed: bool,
) -> dict[str, object]:
    return {
        "panel_artifact_sha256": panel_artifact_sha256,
        "control_artifact_sha256s": control_artifact_sha256s,
        "family_ids": family_ids,
        "native_pooled_sse": native_pooled_sse,
        "control_pooled_sses": control_pooled_sses,
        "native_family_sses": native_family_sses,
        "control_family_sses": control_family_sses,
        "better_or_tied_control_count": better_or_tied_control_count,
        "empirical_p_value": empirical_p_value,
        "median_control_pooled_sse": median_control_pooled_sse,
        "median_sse_recovery_fraction": median_sse_recovery_fraction,
        "median_control_family_sses": median_control_family_sses,
        "family_wins": family_wins,
        "family_win_count": family_win_count,
        "primary_gates": PRIMARY_NULL_GATES.metadata(),
        "empirical_p_gate_passed": empirical_p_gate_passed,
        "median_sse_recovery_gate_passed": (
            median_sse_recovery_gate_passed
        ),
        "family_win_gate_passed": family_win_gate_passed,
        "passed": passed,
        "artifact_kind": _STATISTICS_KIND,
        "format_version": _FORMAT_VERSION,
    }


def _assert_aggregate_matches(
    pooled: float,
    family_values: Sequence[float],
    *,
    label: str,
) -> None:
    if not math.isclose(
        pooled,
        math.fsum(family_values),
        rel_tol=_AGGREGATE_RELATIVE_TOLERANCE,
        abs_tol=_AGGREGATE_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{label} does not equal the sum of family SSEs")


@dataclass(frozen=True, slots=True)
class RandomPartitionNullPanelStatistics:
    """Authenticated primary statistics for a frozen 63-control panel."""

    panel_artifact_sha256: str
    control_artifact_sha256s: tuple[str, ...]
    family_ids: tuple[str, ...]
    native_pooled_sse: float
    control_pooled_sses: tuple[float, ...]
    native_family_sses: tuple[float, ...]
    control_family_sses: tuple[tuple[float, ...], ...]
    better_or_tied_control_count: int
    empirical_p_value: float
    median_control_pooled_sse: float
    median_sse_recovery_fraction: float
    median_control_family_sses: tuple[float, ...]
    family_wins: tuple[bool, ...]
    family_win_count: int
    empirical_p_gate_passed: bool
    median_sse_recovery_gate_passed: bool
    family_win_gate_passed: bool
    passed: bool
    artifact_sha256: str
    artifact_kind: str = _STATISTICS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return _statistics_payload(
            panel_artifact_sha256=self.panel_artifact_sha256,
            control_artifact_sha256s=self.control_artifact_sha256s,
            family_ids=self.family_ids,
            native_pooled_sse=self.native_pooled_sse,
            control_pooled_sses=self.control_pooled_sses,
            native_family_sses=self.native_family_sses,
            control_family_sses=self.control_family_sses,
            better_or_tied_control_count=self.better_or_tied_control_count,
            empirical_p_value=self.empirical_p_value,
            median_control_pooled_sse=self.median_control_pooled_sse,
            median_sse_recovery_fraction=(
                self.median_sse_recovery_fraction
            ),
            median_control_family_sses=self.median_control_family_sses,
            family_wins=self.family_wins,
            family_win_count=self.family_win_count,
            empirical_p_gate_passed=self.empirical_p_gate_passed,
            median_sse_recovery_gate_passed=(
                self.median_sse_recovery_gate_passed
            ),
            family_win_gate_passed=self.family_win_gate_passed,
            passed=self.passed,
        )

    def validate_integrity(self) -> None:
        _require_sha256(
            self.panel_artifact_sha256,
            label="panel_artifact_sha256",
        )
        if (
            len(self.control_artifact_sha256s) != RANDOM_CONTROL_COUNT
            or len(set(self.control_artifact_sha256s))
            != RANDOM_CONTROL_COUNT
        ):
            raise ValueError("control artifact bindings are invalid")
        for digest in self.control_artifact_sha256s:
            _require_sha256(digest, label="control_artifact_sha256")
        if (
            len(self.family_ids) != CONFIRMATION_FAMILY_COUNT
            or len(set(self.family_ids)) != CONFIRMATION_FAMILY_COUNT
            or any(not isinstance(value, str) or not value for value in self.family_ids)
            or len(self.control_pooled_sses) != RANDOM_CONTROL_COUNT
            or len(self.native_family_sses) != CONFIRMATION_FAMILY_COUNT
            or len(self.control_family_sses) != RANDOM_CONTROL_COUNT
            or len(self.median_control_family_sses)
            != CONFIRMATION_FAMILY_COUNT
            or len(self.family_wins) != CONFIRMATION_FAMILY_COUNT
            or any(
                len(values) != CONFIRMATION_FAMILY_COUNT
                for values in self.control_family_sses
            )
        ):
            raise ValueError("null-panel SSE geometry is invalid")
        native_pooled = _require_finite_nonnegative(
            self.native_pooled_sse,
            label="native_pooled_sse",
        )
        control_pooled = tuple(
            _require_finite_nonnegative(value, label="control_pooled_sse")
            for value in self.control_pooled_sses
        )
        native_families = tuple(
            _require_finite_nonnegative(value, label="native_family_sse")
            for value in self.native_family_sses
        )
        control_families = tuple(
            tuple(
                _require_finite_nonnegative(
                    value,
                    label="control_family_sse",
                )
                for value in row
            )
            for row in self.control_family_sses
        )
        _assert_aggregate_matches(
            native_pooled,
            native_families,
            label="native_pooled_sse",
        )
        for ordinal, (pooled, families) in enumerate(
            zip(control_pooled, control_families, strict=True)
        ):
            _assert_aggregate_matches(
                pooled,
                families,
                label=f"control_pooled_sses[{ordinal}]",
            )
        measured_median = float(median(control_pooled))
        if measured_median <= 0.0:
            raise ValueError("median control pooled SSE must be positive")
        measured_better_or_tied = sum(
            control <= native_pooled for control in control_pooled
        )
        measured_p = (1.0 + measured_better_or_tied) / (
            1.0 + RANDOM_CONTROL_COUNT
        )
        measured_recovery = (
            measured_median - native_pooled
        ) / measured_median
        measured_family_medians = tuple(
            float(
                median(
                    tuple(
                        control_families[control][family]
                        for control in range(RANDOM_CONTROL_COUNT)
                    )
                )
            )
            for family in range(CONFIRMATION_FAMILY_COUNT)
        )
        measured_family_wins = tuple(
            native < control_median
            for native, control_median in zip(
                native_families,
                measured_family_medians,
                strict=True,
            )
        )
        measured_family_win_count = sum(measured_family_wins)
        measured_p_gate = measured_p <= (
            PRIMARY_NULL_GATES.maximum_empirical_p_value
        )
        measured_recovery_gate = measured_recovery >= (
            PRIMARY_NULL_GATES.minimum_median_sse_recovery_fraction
        )
        measured_family_gate = measured_family_win_count >= (
            PRIMARY_NULL_GATES.minimum_family_win_count
        )
        measured_pass = (
            measured_p_gate
            and measured_recovery_gate
            and measured_family_gate
        )
        if (
            isinstance(self.better_or_tied_control_count, bool)
            or self.better_or_tied_control_count != measured_better_or_tied
            or self.empirical_p_value != measured_p
            or self.median_control_pooled_sse != measured_median
            or self.median_sse_recovery_fraction != measured_recovery
            or self.median_control_family_sses != measured_family_medians
            or self.family_wins != measured_family_wins
            or isinstance(self.family_win_count, bool)
            or self.family_win_count != measured_family_win_count
            or self.empirical_p_gate_passed is not measured_p_gate
            or self.median_sse_recovery_gate_passed
            is not measured_recovery_gate
            or self.family_win_gate_passed is not measured_family_gate
            or self.passed is not measured_pass
            or self.artifact_kind != _STATISTICS_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("null-panel statistics differ from replay")
        if self.artifact_sha256 != _json_sha256(
            self._payload(),
            domain=_STATISTICS_DOMAIN,
        ):
            raise ValueError("null-panel statistics artifact hash differs")

    @property
    def gate_results(self) -> dict[str, bool]:
        return {
            "empirical_p_value_at_most_0_05": self.empirical_p_gate_passed,
            "median_sse_recovery_at_least_0_05": (
                self.median_sse_recovery_gate_passed
            ),
            "strict_family_wins_at_least_7_of_8": (
                self.family_win_gate_passed
            ),
            "all_primary_null_gates_passed": self.passed,
        }

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "gate_results": self.gate_results,
            "artifact_sha256": self.artifact_sha256,
        }


def evaluate_random_partition_null_panel(
    panel: BalancedRandomPartitionPanel,
    *,
    family_ids: Sequence[str],
    native_pooled_sse: float,
    control_pooled_sses: Sequence[float],
    native_family_sses: Sequence[float],
    control_family_sses: Sequence[Sequence[float]],
) -> RandomPartitionNullPanelStatistics:
    """Evaluate the frozen primary null gates without tunable thresholds."""

    panel.validate_integrity()
    frozen_family_ids = tuple(family_ids)
    frozen_native_pooled = _require_finite_nonnegative(
        native_pooled_sse,
        label="native_pooled_sse",
    )
    frozen_control_pooled = tuple(
        _require_finite_nonnegative(value, label="control_pooled_sse")
        for value in control_pooled_sses
    )
    frozen_native_families = tuple(
        _require_finite_nonnegative(value, label="native_family_sse")
        for value in native_family_sses
    )
    frozen_control_families = tuple(
        tuple(
            _require_finite_nonnegative(
                value,
                label="control_family_sse",
            )
            for value in row
        )
        for row in control_family_sses
    )
    if (
        len(frozen_family_ids) != CONFIRMATION_FAMILY_COUNT
        or len(set(frozen_family_ids)) != CONFIRMATION_FAMILY_COUNT
        or any(not isinstance(value, str) or not value for value in frozen_family_ids)
        or len(frozen_control_pooled) != RANDOM_CONTROL_COUNT
        or len(frozen_native_families) != CONFIRMATION_FAMILY_COUNT
        or len(frozen_control_families) != RANDOM_CONTROL_COUNT
        or any(
            len(values) != CONFIRMATION_FAMILY_COUNT
            for values in frozen_control_families
        )
    ):
        raise ValueError("null-panel SSE geometry is invalid")
    _assert_aggregate_matches(
        frozen_native_pooled,
        frozen_native_families,
        label="native_pooled_sse",
    )
    for ordinal, (pooled, families) in enumerate(
        zip(frozen_control_pooled, frozen_control_families, strict=True)
    ):
        _assert_aggregate_matches(
            pooled,
            families,
            label=f"control_pooled_sses[{ordinal}]",
        )
    median_control_pooled = float(median(frozen_control_pooled))
    if median_control_pooled <= 0.0:
        raise ValueError("median control pooled SSE must be positive")
    better_or_tied = sum(
        control <= frozen_native_pooled for control in frozen_control_pooled
    )
    empirical_p = (1.0 + better_or_tied) / (1.0 + RANDOM_CONTROL_COUNT)
    median_recovery = (
        median_control_pooled - frozen_native_pooled
    ) / median_control_pooled
    family_medians = tuple(
        float(
            median(
                tuple(
                    frozen_control_families[control][family]
                    for control in range(RANDOM_CONTROL_COUNT)
                )
            )
        )
        for family in range(CONFIRMATION_FAMILY_COUNT)
    )
    family_wins = tuple(
        native < control_median
        for native, control_median in zip(
            frozen_native_families,
            family_medians,
            strict=True,
        )
    )
    family_win_count = sum(family_wins)
    empirical_p_gate = (
        empirical_p <= PRIMARY_NULL_GATES.maximum_empirical_p_value
    )
    median_recovery_gate = median_recovery >= (
        PRIMARY_NULL_GATES.minimum_median_sse_recovery_fraction
    )
    family_win_gate = (
        family_win_count >= PRIMARY_NULL_GATES.minimum_family_win_count
    )
    passed = empirical_p_gate and median_recovery_gate and family_win_gate
    payload = _statistics_payload(
        panel_artifact_sha256=panel.artifact_sha256,
        control_artifact_sha256s=panel.control_artifact_sha256s,
        family_ids=frozen_family_ids,
        native_pooled_sse=frozen_native_pooled,
        control_pooled_sses=frozen_control_pooled,
        native_family_sses=frozen_native_families,
        control_family_sses=frozen_control_families,
        better_or_tied_control_count=better_or_tied,
        empirical_p_value=empirical_p,
        median_control_pooled_sse=median_control_pooled,
        median_sse_recovery_fraction=median_recovery,
        median_control_family_sses=family_medians,
        family_wins=family_wins,
        family_win_count=family_win_count,
        empirical_p_gate_passed=empirical_p_gate,
        median_sse_recovery_gate_passed=median_recovery_gate,
        family_win_gate_passed=family_win_gate,
        passed=passed,
    )
    return RandomPartitionNullPanelStatistics(
        panel_artifact_sha256=panel.artifact_sha256,
        control_artifact_sha256s=panel.control_artifact_sha256s,
        family_ids=frozen_family_ids,
        native_pooled_sse=frozen_native_pooled,
        control_pooled_sses=frozen_control_pooled,
        native_family_sses=frozen_native_families,
        control_family_sses=frozen_control_families,
        better_or_tied_control_count=better_or_tied,
        empirical_p_value=empirical_p,
        median_control_pooled_sse=median_control_pooled,
        median_sse_recovery_fraction=median_recovery,
        median_control_family_sses=family_medians,
        family_wins=family_wins,
        family_win_count=family_win_count,
        empirical_p_gate_passed=empirical_p_gate,
        median_sse_recovery_gate_passed=median_recovery_gate,
        family_win_gate_passed=family_win_gate,
        passed=passed,
        artifact_sha256=_json_sha256(payload, domain=_STATISTICS_DOMAIN),
    )
