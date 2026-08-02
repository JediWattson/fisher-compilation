"""Pure fresh-response confirmation for grouped graph-wavelet plans.

This module does not load artifacts, fit bases, or write reports.  It accepts
already-frozen conditional spectral plans and evaluates each plan directly on
one fresh central-response tensor.  The 63 random-partition controls remain in
their frozen panel order, and residual SSE is split by the eight *native*
source-mode groups for every plan.  That makes the random-panel family gate a
comparison of like with like rather than a comparison of control-specific
partitions.

The returned object contains authenticated metadata and scalar measurements
only.  It never retains response, scale, prediction, or residual tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .conditional_spectral_generator import ConditionalSpectralGeneratorPlan
from .graph_wavelet_random_partition_confirmation import (
    CONFIRMATION_FAMILY_COUNT,
    CONFIRMATION_GROUP_COUNT,
    CONFIRMATION_GROUP_SIZE,
    CONFIRMATION_PARENT_RANK,
    RANDOM_CONTROL_COUNT,
    BalancedRandomPartitionPanel,
    RandomPartitionNullPanelStatistics,
    evaluate_random_partition_null_panel,
)


__all__ = [
    "MAXIMUM_NATIVE_RELATIVE_ERROR",
    "MINIMUM_NATIVE_COSINE",
    "GraphWaveletStructuralConfirmation",
    "StructuralOriginMetrics",
    "StructuralPlanMetrics",
    "evaluate_graph_wavelet_structural_confirmation",
]


MAXIMUM_NATIVE_RELATIVE_ERROR = 0.20
MINIMUM_NATIVE_COSINE = 0.98

_FORMAT_VERSION = 1
_ARTIFACT_KIND = "fisher_graph.graph_wavelet_structural_confirmation"
_ARTIFACT_DOMAIN = (
    b"fisher-graph:graph-wavelet-structural-confirmation:v1\0"
)
_TENSOR_DOMAIN = (
    b"fisher-graph:graph-wavelet-structural-confirmation-tensor:v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_RELATIVE_TOLERANCE = 1.0e-12
_FLOAT_ABSOLUTE_TOLERANCE = 1.0e-12

_NATIVE_ROLE = "native_signed_g8"
_CONTROL_ROLE = "random_partition_control"
_SIGNED_GFA_ROLE = "signed_gfa_reference"
_GLOBAL_SVD_ROLE = "global_svd_descriptive_ceiling"
_ROLES = {
    _NATIVE_ROLE,
    _CONTROL_ROLE,
    _SIGNED_GFA_ROLE,
    _GLOBAL_SVD_ROLE,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_ARTIFACT_DOMAIN + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_cosine(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [-1, 1]")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_FLOAT_RELATIVE_TOLERANCE,
        abs_tol=_FLOAT_ABSOLUTE_TOLERANCE,
    )


def _canonical_tensor(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim or any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(int(width) for width in canonical.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_origins(origins: Sequence[int]) -> tuple[int, ...]:
    if isinstance(origins, (str, bytes)) or not isinstance(origins, Sequence):
        raise TypeError("origins must be a sequence of integers")
    result = tuple(origins)
    if (
        not result
        or any(type(origin) is not int or origin < 0 for origin in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(
            "origins must be a nonempty strictly increasing sequence of "
            "nonnegative integers"
        )
    return result


def _canonical_groups(
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError("native_groups must be a sequence of groups")
    try:
        result = tuple(tuple(group) for group in groups)
    except TypeError as error:
        raise TypeError("native_groups must contain sequences") from error
    if (
        len(result) != CONFIRMATION_GROUP_COUNT
        or any(len(group) != CONFIRMATION_GROUP_SIZE for group in result)
        or any(
            type(member) is not int
            for group in result
            for member in group
        )
        or any(tuple(sorted(group)) != group for group in result)
        or tuple(sorted(result)) != result
        or tuple(sorted(member for group in result for member in group))
        != tuple(range(CONFIRMATION_PARENT_RANK))
    ):
        raise ValueError(
            "native_groups must be eight canonical groups partitioning "
            "source modes 0..63"
        )
    return result


def _cosine(left: Tensor, right: Tensor) -> float:
    first = left.reshape(-1)
    second = right.reshape(-1)
    first_norm = float(torch.linalg.vector_norm(first))
    second_norm = float(torch.linalg.vector_norm(second))
    epsilon = torch.finfo(torch.float64).eps
    if first_norm <= epsilon:
        return 1.0 if second_norm <= epsilon else 0.0
    if second_norm <= epsilon:
        return 0.0
    return max(
        -1.0,
        min(1.0, float(torch.dot(first, second)) / (first_norm * second_norm)),
    )


@dataclass(frozen=True, slots=True)
class StructuralOriginMetrics:
    """Scalar geometry for one fresh source origin."""

    origin: int
    target_sse: float
    residual_sse: float
    target_frobenius: float
    residual_frobenius: float
    relative_error: float
    cosine: float

    def __post_init__(self) -> None:
        if type(self.origin) is not int or self.origin < 0:
            raise ValueError("origin must be a nonnegative integer")
        target_sse = _finite_nonnegative(self.target_sse, label="target_sse")
        residual_sse = _finite_nonnegative(
            self.residual_sse,
            label="residual_sse",
        )
        target_frobenius = _finite_nonnegative(
            self.target_frobenius,
            label="target_frobenius",
        )
        residual_frobenius = _finite_nonnegative(
            self.residual_frobenius,
            label="residual_frobenius",
        )
        relative_error = _finite_nonnegative(
            self.relative_error,
            label="relative_error",
        )
        cosine = _finite_cosine(self.cosine, label="cosine")
        if target_sse <= torch.finfo(torch.float64).eps:
            raise ValueError("fresh target energy must be positive at every origin")
        if (
            not _close(target_frobenius, math.sqrt(target_sse))
            or not _close(residual_frobenius, math.sqrt(residual_sse))
            or not _close(relative_error, math.sqrt(residual_sse / target_sse))
        ):
            raise ValueError("per-origin metric scalars are inconsistent")
        for name, value in (
            ("target_sse", target_sse),
            ("residual_sse", residual_sse),
            ("target_frobenius", target_frobenius),
            ("residual_frobenius", residual_frobenius),
            ("relative_error", relative_error),
            ("cosine", cosine),
        ):
            object.__setattr__(self, name, value)

    def metadata(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "target_sse": self.target_sse,
            "residual_sse": self.residual_sse,
            "target_frobenius": self.target_frobenius,
            "residual_frobenius": self.residual_frobenius,
            "relative_error": self.relative_error,
            "cosine": self.cosine,
        }


@dataclass(frozen=True, slots=True)
class StructuralPlanMetrics:
    """Tensor-free structural metrics for one frozen plan."""

    role: str
    plan_artifact_sha256: str
    partition_artifact_sha256: str | None
    control_ordinal: int | None
    pooled_target_sse: float
    pooled_residual_sse: float
    pooled_target_frobenius: float
    pooled_residual_frobenius: float
    pooled_relative_error: float
    pooled_cosine: float
    per_origin: tuple[StructuralOriginMetrics, ...]
    native_group_residual_sses: tuple[float, ...]
    fit_was_not_recomputed: bool = True

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError("structural plan role is invalid")
        _require_sha256(
            self.plan_artifact_sha256,
            label="plan_artifact_sha256",
        )
        if self.role in {_NATIVE_ROLE, _CONTROL_ROLE}:
            _require_sha256(
                self.partition_artifact_sha256,
                label="partition_artifact_sha256",
            )
        elif self.partition_artifact_sha256 is not None:
            raise ValueError("reference plans cannot claim a partition")
        if self.role == _CONTROL_ROLE:
            if (
                type(self.control_ordinal) is not int
                or not 0 <= self.control_ordinal < RANDOM_CONTROL_COUNT
            ):
                raise ValueError("random control ordinal is invalid")
        elif self.control_ordinal is not None:
            raise ValueError("only random controls have a control ordinal")
        if not self.per_origin:
            raise ValueError("per_origin cannot be empty")
        origins = tuple(metric.origin for metric in self.per_origin)
        if tuple(sorted(set(origins))) != origins:
            raise ValueError("per-origin metrics must be strictly ordered")
        if len(self.native_group_residual_sses) != CONFIRMATION_FAMILY_COUNT:
            raise ValueError("exactly eight native-group SSEs are required")
        pooled_target_sse = _finite_nonnegative(
            self.pooled_target_sse,
            label="pooled_target_sse",
        )
        pooled_residual_sse = _finite_nonnegative(
            self.pooled_residual_sse,
            label="pooled_residual_sse",
        )
        pooled_target_frobenius = _finite_nonnegative(
            self.pooled_target_frobenius,
            label="pooled_target_frobenius",
        )
        pooled_residual_frobenius = _finite_nonnegative(
            self.pooled_residual_frobenius,
            label="pooled_residual_frobenius",
        )
        pooled_relative_error = _finite_nonnegative(
            self.pooled_relative_error,
            label="pooled_relative_error",
        )
        pooled_cosine = _finite_cosine(
            self.pooled_cosine,
            label="pooled_cosine",
        )
        group_sses = tuple(
            _finite_nonnegative(value, label="native_group_residual_sse")
            for value in self.native_group_residual_sses
        )
        target_sum = math.fsum(metric.target_sse for metric in self.per_origin)
        residual_sum = math.fsum(
            metric.residual_sse for metric in self.per_origin
        )
        if pooled_target_sse <= torch.finfo(torch.float64).eps:
            raise ValueError("fresh pooled target energy must be positive")
        if (
            not _close(pooled_target_sse, target_sum)
            or not _close(pooled_residual_sse, residual_sum)
            or not _close(pooled_residual_sse, math.fsum(group_sses))
            or not _close(
                pooled_target_frobenius,
                math.sqrt(pooled_target_sse),
            )
            or not _close(
                pooled_residual_frobenius,
                math.sqrt(pooled_residual_sse),
            )
            or not _close(
                pooled_relative_error,
                math.sqrt(pooled_residual_sse / pooled_target_sse),
            )
        ):
            raise ValueError("pooled metric scalars are inconsistent")
        if self.fit_was_not_recomputed is not True:
            raise ValueError("structural confirmation cannot recompute a fit")
        for name, value in (
            ("pooled_target_sse", pooled_target_sse),
            ("pooled_residual_sse", pooled_residual_sse),
            ("pooled_target_frobenius", pooled_target_frobenius),
            ("pooled_residual_frobenius", pooled_residual_frobenius),
            ("pooled_relative_error", pooled_relative_error),
            ("pooled_cosine", pooled_cosine),
            ("native_group_residual_sses", group_sses),
        ):
            object.__setattr__(self, name, value)

    def metadata(self) -> dict[str, object]:
        return {
            "role": self.role,
            "plan_artifact_sha256": self.plan_artifact_sha256,
            "partition_artifact_sha256": self.partition_artifact_sha256,
            "control_ordinal": self.control_ordinal,
            "pooled": {
                "target_sse": self.pooled_target_sse,
                "residual_sse": self.pooled_residual_sse,
                "target_frobenius": self.pooled_target_frobenius,
                "residual_frobenius": self.pooled_residual_frobenius,
                "relative_error": self.pooled_relative_error,
                "cosine": self.pooled_cosine,
            },
            "per_origin": tuple(metric.metadata() for metric in self.per_origin),
            "native_group_residual_sses": self.native_group_residual_sses,
            "fit_was_not_recomputed": self.fit_was_not_recomputed,
        }


def _same_targets(
    first: StructuralPlanMetrics,
    second: StructuralPlanMetrics,
) -> bool:
    return (
        tuple(metric.origin for metric in first.per_origin)
        == tuple(metric.origin for metric in second.per_origin)
        and _close(first.pooled_target_sse, second.pooled_target_sse)
        and all(
            _close(left.target_sse, right.target_sse)
            for left, right in zip(first.per_origin, second.per_origin, strict=True)
        )
    )


@dataclass(frozen=True, slots=True)
class GraphWaveletStructuralConfirmation:
    """Authenticated scalar-only result for one frozen confirmation panel."""

    panel_artifact_sha256: str
    fresh_response_sha256: str
    source_scales_sha256: str
    origins: tuple[int, ...]
    native_groups: tuple[tuple[int, ...], ...]
    native: StructuralPlanMetrics
    controls: tuple[StructuralPlanMetrics, ...]
    signed_gfa_reference: StructuralPlanMetrics
    global_svd_ceiling: StructuralPlanMetrics
    random_null_statistics: RandomPartitionNullPanelStatistics
    native_relative_error_gate_passed: bool
    native_cosine_gate_passed: bool
    signed_gfa_sse_gate_passed: bool
    random_null_gate_passed: bool
    passed: bool
    artifact_sha256: str
    fit_was_not_recomputed: bool = True
    global_svd_is_descriptive_only: bool = True
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        _require_sha256(
            self.panel_artifact_sha256,
            label="panel_artifact_sha256",
        )
        _require_sha256(
            self.fresh_response_sha256,
            label="fresh_response_sha256",
        )
        _require_sha256(
            self.source_scales_sha256,
            label="source_scales_sha256",
        )
        origins = _canonical_origins(self.origins)
        groups = _canonical_groups(self.native_groups)
        if self.native.role != _NATIVE_ROLE:
            raise ValueError("native metrics role is invalid")
        if (
            len(self.controls) != RANDOM_CONTROL_COUNT
            or any(
                metric.role != _CONTROL_ROLE
                or metric.control_ordinal != ordinal
                for ordinal, metric in enumerate(self.controls)
            )
        ):
            raise ValueError("random control metrics are not in panel order")
        if self.signed_gfa_reference.role != _SIGNED_GFA_ROLE:
            raise ValueError("signed-GFA reference role is invalid")
        if self.global_svd_ceiling.role != _GLOBAL_SVD_ROLE:
            raise ValueError("global-SVD ceiling role is invalid")
        all_metrics = (
            self.native,
            *self.controls,
            self.signed_gfa_reference,
            self.global_svd_ceiling,
        )
        if any(
            tuple(item.origin for item in metric.per_origin) != origins
            for metric in all_metrics
        ):
            raise ValueError("plan metrics do not match confirmation origins")
        if any(not _same_targets(self.native, metric) for metric in all_metrics):
            raise ValueError("plans were not evaluated against one common target")
        statistics = self.random_null_statistics
        statistics.validate_integrity()
        expected_family_ids = tuple(
            f"native_group_{ordinal:02d}"
            for ordinal in range(CONFIRMATION_FAMILY_COUNT)
        )
        if (
            statistics.panel_artifact_sha256 != self.panel_artifact_sha256
            or statistics.family_ids != expected_family_ids
            or statistics.control_artifact_sha256s
            != tuple(
                metric.partition_artifact_sha256 for metric in self.controls
            )
            or not _close(
                statistics.native_pooled_sse,
                self.native.pooled_residual_sse,
            )
            or any(
                not _close(left, right)
                for left, right in zip(
                    statistics.control_pooled_sses,
                    tuple(
                        metric.pooled_residual_sse for metric in self.controls
                    ),
                    strict=True,
                )
            )
            or any(
                not _close(left, right)
                for left, right in zip(
                    statistics.native_family_sses,
                    self.native.native_group_residual_sses,
                    strict=True,
                )
            )
            or any(
                any(
                    not _close(left, right)
                    for left, right in zip(
                        statistics_row,
                        metric.native_group_residual_sses,
                        strict=True,
                    )
                )
                for statistics_row, metric in zip(
                    statistics.control_family_sses,
                    self.controls,
                    strict=True,
                )
            )
        ):
            raise ValueError("random-null statistics differ from plan metrics")
        expected_relative_gate = (
            self.native.pooled_relative_error
            <= MAXIMUM_NATIVE_RELATIVE_ERROR
            and all(
                metric.relative_error <= MAXIMUM_NATIVE_RELATIVE_ERROR
                for metric in self.native.per_origin
            )
        )
        expected_cosine_gate = (
            self.native.pooled_cosine >= MINIMUM_NATIVE_COSINE
            and all(
                metric.cosine >= MINIMUM_NATIVE_COSINE
                for metric in self.native.per_origin
            )
        )
        expected_signed_gate = (
            self.native.pooled_residual_sse
            <= self.signed_gfa_reference.pooled_residual_sse
        )
        expected_random_gate = statistics.passed
        expected_pass = (
            expected_relative_gate
            and expected_cosine_gate
            and expected_signed_gate
            and expected_random_gate
        )
        if (
            self.native_relative_error_gate_passed is not expected_relative_gate
            or self.native_cosine_gate_passed is not expected_cosine_gate
            or self.signed_gfa_sse_gate_passed is not expected_signed_gate
            or self.random_null_gate_passed is not expected_random_gate
            or self.passed is not expected_pass
            or self.fit_was_not_recomputed is not True
            or self.global_svd_is_descriptive_only is not True
            or self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("structural confirmation gates or claims differ")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "native_groups", groups)
        if self.artifact_sha256 != _json_sha256(self._payload()):
            raise ValueError("structural confirmation artifact hash differs")

    @property
    def gate_results(self) -> dict[str, bool]:
        return {
            "native_pooled_and_every_origin_relative_error_at_most_0_20": (
                self.native_relative_error_gate_passed
            ),
            "native_pooled_and_every_origin_cosine_at_least_0_98": (
                self.native_cosine_gate_passed
            ),
            "native_pooled_sse_at_most_signed_gfa": (
                self.signed_gfa_sse_gate_passed
            ),
            "all_random_partition_null_gates_passed": (
                self.random_null_gate_passed
            ),
            "all_primary_structural_gates_passed": self.passed,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "panel_artifact_sha256": self.panel_artifact_sha256,
            "fresh_input": {
                "response_sha256": self.fresh_response_sha256,
                "source_scales_sha256": self.source_scales_sha256,
                "response_shape": (
                    CONFIRMATION_PARENT_RANK,
                    len(self.origins),
                    32,
                    CONFIRMATION_PARENT_RANK,
                ),
                "origins": self.origins,
            },
            "native_groups": self.native_groups,
            "native": self.native.metadata(),
            "controls": tuple(metric.metadata() for metric in self.controls),
            "signed_gfa_reference": self.signed_gfa_reference.metadata(),
            "global_svd_ceiling": self.global_svd_ceiling.metadata(),
            "random_null_statistics": self.random_null_statistics.metadata(),
            "primary_gate_thresholds": {
                "maximum_native_relative_error": (
                    MAXIMUM_NATIVE_RELATIVE_ERROR
                ),
                "minimum_native_cosine": MINIMUM_NATIVE_COSINE,
                "native_sse_must_not_exceed_signed_gfa": True,
            },
            "gate_results": self.gate_results,
            "fit_was_not_recomputed": self.fit_was_not_recomputed,
            "global_svd_is_descriptive_only": (
                self.global_svd_is_descriptive_only
            ),
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _evaluate_plan(
    plan: ConditionalSpectralGeneratorPlan,
    *,
    role: str,
    partition_artifact_sha256: str | None,
    control_ordinal: int | None,
    targets: Tensor,
    origins: tuple[int, ...],
    native_groups: tuple[tuple[int, ...], ...],
) -> StructuralPlanMetrics:
    if not isinstance(plan, ConditionalSpectralGeneratorPlan):
        raise TypeError("every plan must be a ConditionalSpectralGeneratorPlan")
    plan.validate_integrity()
    predictions = torch.stack(
        tuple(plan.weighted_kernel_at_origin(origin) for origin in origins),
        dim=0,
    )
    residuals = predictions - targets
    pooled_target_sse = float(targets.square().sum())
    pooled_residual_sse = float(residuals.square().sum())
    per_origin = tuple(
        StructuralOriginMetrics(
            origin=origin,
            target_sse=(target_sse := float(target.square().sum())),
            residual_sse=(residual_sse := float(residual.square().sum())),
            target_frobenius=math.sqrt(target_sse),
            residual_frobenius=math.sqrt(residual_sse),
            relative_error=math.sqrt(residual_sse / target_sse),
            cosine=_cosine(prediction, target),
        )
        for origin, prediction, target, residual in zip(
            origins,
            predictions,
            targets,
            residuals,
            strict=True,
        )
    )
    group_sses = tuple(
        float(
            residuals.index_select(
                1,
                torch.tensor(group, dtype=torch.int64),
            )
            .square()
            .sum()
        )
        for group in native_groups
    )
    return StructuralPlanMetrics(
        role=role,
        plan_artifact_sha256=plan.artifact_sha256,
        partition_artifact_sha256=partition_artifact_sha256,
        control_ordinal=control_ordinal,
        pooled_target_sse=pooled_target_sse,
        pooled_residual_sse=pooled_residual_sse,
        pooled_target_frobenius=math.sqrt(pooled_target_sse),
        pooled_residual_frobenius=math.sqrt(pooled_residual_sse),
        pooled_relative_error=math.sqrt(
            pooled_residual_sse / pooled_target_sse
        ),
        pooled_cosine=_cosine(predictions, targets),
        per_origin=per_origin,
        native_group_residual_sses=group_sses,
    )


def evaluate_graph_wavelet_structural_confirmation(
    *,
    panel: BalancedRandomPartitionPanel,
    native_plan: ConditionalSpectralGeneratorPlan,
    control_plans: Sequence[ConditionalSpectralGeneratorPlan],
    signed_gfa_reference_plan: ConditionalSpectralGeneratorPlan,
    global_svd_ceiling_plan: ConditionalSpectralGeneratorPlan,
    fresh_central_responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    native_groups: Sequence[Sequence[int]],
) -> GraphWaveletStructuralConfirmation:
    """Evaluate frozen plans on fresh responses without fitting anything.

    ``control_plans[i]`` is interpreted as the compiled plan for
    ``panel.controls[i]``.  The null-panel artifact or its loader is
    responsible for authenticating that correspondence before this pure
    evaluator is called.
    """

    if not isinstance(panel, BalancedRandomPartitionPanel):
        raise TypeError("panel must be a BalancedRandomPartitionPanel")
    panel.validate_integrity()
    frozen_origins = _canonical_origins(origins)
    frozen_groups = _canonical_groups(native_groups)
    if frozen_groups != panel.native_groups:
        raise ValueError("native_groups do not match the frozen null panel")
    controls = tuple(control_plans)
    if len(controls) != RANDOM_CONTROL_COUNT:
        raise ValueError("exactly 63 control plans are required")
    responses = _canonical_tensor(
        fresh_central_responses,
        label="fresh_central_responses",
        ndim=4,
    )
    scales = _canonical_tensor(source_scales, label="source_scales", ndim=1)
    expected_shape = (
        CONFIRMATION_PARENT_RANK,
        len(frozen_origins),
        32,
        CONFIRMATION_PARENT_RANK,
    )
    if tuple(responses.shape) != expected_shape:
        raise ValueError(
            "fresh_central_responses must have shape "
            f"{expected_shape}"
        )
    if tuple(scales.shape) != (CONFIRMATION_PARENT_RANK,):
        raise ValueError("source_scales must have shape (64,)")
    if bool((scales <= 0.0).any()):
        raise ValueError("source_scales must be strictly positive")
    all_plans = (
        native_plan,
        *controls,
        signed_gfa_reference_plan,
        global_svd_ceiling_plan,
    )
    for plan in all_plans:
        if not isinstance(plan, ConditionalSpectralGeneratorPlan):
            raise TypeError(
                "every plan must be a ConditionalSpectralGeneratorPlan"
            )
        plan.validate_integrity()
        if (
            plan.source_modes != CONFIRMATION_PARENT_RANK
            or plan.target_modes != CONFIRMATION_PARENT_RANK
            or plan.lag_count != 32
        ):
            raise ValueError("plan geometry must be [64, 32, 64]")
        if not torch.equal(plan.source_scales, scales):
            raise ValueError("plan source scales differ from confirmation scales")
    targets = (
        responses * scales.view(-1, 1, 1, 1)
    ).permute(1, 0, 2, 3).contiguous()
    if any(
        float(target.square().sum()) <= torch.finfo(torch.float64).eps
        for target in targets
    ):
        raise ValueError("fresh target energy must be positive at every origin")
    native_metrics = _evaluate_plan(
        native_plan,
        role=_NATIVE_ROLE,
        partition_artifact_sha256=(
            panel.native_partition_artifact_sha256
        ),
        control_ordinal=None,
        targets=targets,
        origins=frozen_origins,
        native_groups=frozen_groups,
    )
    control_metrics = tuple(
        _evaluate_plan(
            plan,
            role=_CONTROL_ROLE,
            partition_artifact_sha256=(
                panel.controls[ordinal].artifact_sha256
            ),
            control_ordinal=ordinal,
            targets=targets,
            origins=frozen_origins,
            native_groups=frozen_groups,
        )
        for ordinal, plan in enumerate(controls)
    )
    signed_gfa_metrics = _evaluate_plan(
        signed_gfa_reference_plan,
        role=_SIGNED_GFA_ROLE,
        partition_artifact_sha256=None,
        control_ordinal=None,
        targets=targets,
        origins=frozen_origins,
        native_groups=frozen_groups,
    )
    global_svd_metrics = _evaluate_plan(
        global_svd_ceiling_plan,
        role=_GLOBAL_SVD_ROLE,
        partition_artifact_sha256=None,
        control_ordinal=None,
        targets=targets,
        origins=frozen_origins,
        native_groups=frozen_groups,
    )
    family_ids = tuple(
        f"native_group_{ordinal:02d}"
        for ordinal in range(CONFIRMATION_FAMILY_COUNT)
    )
    null_statistics = evaluate_random_partition_null_panel(
        panel,
        family_ids=family_ids,
        native_pooled_sse=native_metrics.pooled_residual_sse,
        control_pooled_sses=tuple(
            metric.pooled_residual_sse for metric in control_metrics
        ),
        native_family_sses=native_metrics.native_group_residual_sses,
        control_family_sses=tuple(
            metric.native_group_residual_sses for metric in control_metrics
        ),
    )
    relative_gate = (
        native_metrics.pooled_relative_error
        <= MAXIMUM_NATIVE_RELATIVE_ERROR
        and all(
            metric.relative_error <= MAXIMUM_NATIVE_RELATIVE_ERROR
            for metric in native_metrics.per_origin
        )
    )
    cosine_gate = (
        native_metrics.pooled_cosine >= MINIMUM_NATIVE_COSINE
        and all(
            metric.cosine >= MINIMUM_NATIVE_COSINE
            for metric in native_metrics.per_origin
        )
    )
    signed_gate = (
        native_metrics.pooled_residual_sse
        <= signed_gfa_metrics.pooled_residual_sse
    )
    random_gate = null_statistics.passed
    passed = relative_gate and cosine_gate and signed_gate and random_gate
    payload: dict[str, object] = {
        "panel_artifact_sha256": panel.artifact_sha256,
        "fresh_input": {
            "response_sha256": _tensor_sha256(responses),
            "source_scales_sha256": _tensor_sha256(scales),
            "response_shape": expected_shape,
            "origins": frozen_origins,
        },
        "native_groups": frozen_groups,
        "native": native_metrics.metadata(),
        "controls": tuple(metric.metadata() for metric in control_metrics),
        "signed_gfa_reference": signed_gfa_metrics.metadata(),
        "global_svd_ceiling": global_svd_metrics.metadata(),
        "random_null_statistics": null_statistics.metadata(),
        "primary_gate_thresholds": {
            "maximum_native_relative_error": MAXIMUM_NATIVE_RELATIVE_ERROR,
            "minimum_native_cosine": MINIMUM_NATIVE_COSINE,
            "native_sse_must_not_exceed_signed_gfa": True,
        },
        "gate_results": {
            "native_pooled_and_every_origin_relative_error_at_most_0_20": (
                relative_gate
            ),
            "native_pooled_and_every_origin_cosine_at_least_0_98": (
                cosine_gate
            ),
            "native_pooled_sse_at_most_signed_gfa": signed_gate,
            "all_random_partition_null_gates_passed": random_gate,
            "all_primary_structural_gates_passed": passed,
        },
        "fit_was_not_recomputed": True,
        "global_svd_is_descriptive_only": True,
        "artifact_kind": _ARTIFACT_KIND,
        "format_version": _FORMAT_VERSION,
    }
    return GraphWaveletStructuralConfirmation(
        panel_artifact_sha256=panel.artifact_sha256,
        fresh_response_sha256=_tensor_sha256(responses),
        source_scales_sha256=_tensor_sha256(scales),
        origins=frozen_origins,
        native_groups=frozen_groups,
        native=native_metrics,
        controls=control_metrics,
        signed_gfa_reference=signed_gfa_metrics,
        global_svd_ceiling=global_svd_metrics,
        random_null_statistics=null_statistics,
        native_relative_error_gate_passed=relative_gate,
        native_cosine_gate_passed=cosine_gate,
        signed_gfa_sse_gate_passed=signed_gate,
        random_null_gate_passed=random_gate,
        passed=passed,
        artifact_sha256=_json_sha256(payload),
    )
