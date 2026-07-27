"""Raw compression rate-distortion points and deterministic Pareto views."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .structured_mlp_dense_supermode_pipeline import (
    STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_SCHEMA,
    StructuredMLPDenseSupermodeCandidate,
)


_RATE_AXES = {
    "learned_parameters",
    "runtime_parameter_bytes",
    "logical_macs_per_token",
    "measured_latency_ms",
}
_QUALITY_DIRECTIONS = {
    "downstream_score": "maximize",
    "top1_agreement": "maximize",
    "nll": "minimize",
    "teacher_kl": "minimize",
    "operator_nrmse": "minimize",
}
_DENSE_SUPERMODE_PARAMETER_SCOPE = (
    "compiled_structured_transformer_layer_all_learned_parameters"
)
_DENSE_SUPERMODE_COMPUTE_SCOPE = (
    "compiled_structured_transformer_layer_mlp_gate_up_down_"
    "linear_weight_matmuls_only"
)
_DENSE_SUPERMODE_REPORT_COMPUTE_SCOPE = (
    "gate_up_down_linear_weight_matmuls_only; "
    "attention_and_norm_unchanged"
)


def _finite_float(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (float, int))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class CompressionRatePoint:
    """One raw quality/resource observation for a frozen candidate."""

    candidate_id: str
    method: str
    evaluation_id: str
    evaluation_split_sha256: str
    task_suite: str
    candidate_execution_fingerprint: str
    candidate_report_sha256: str
    parameter_scope: str
    compute_scope: str
    runtime_dtype: str
    runtime: str
    learned_parameters: int
    runtime_parameter_bytes: int
    logical_macs_per_token: int
    downstream_score: float
    nll: float
    teacher_kl: float
    top1_agreement: float
    operator_nrmse: float
    measured_latency_ms: float | None = None
    hardware_id: str | None = None
    benchmark_protocol: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "method",
            "evaluation_id",
            "task_suite",
            "parameter_scope",
            "compute_scope",
            "runtime_dtype",
            "runtime",
        ):
            _nonempty_string(getattr(self, name), label=name)
        for name in (
            "evaluation_split_sha256",
            "candidate_execution_fingerprint",
            "candidate_report_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name in (
            "learned_parameters",
            "runtime_parameter_bytes",
            "logical_macs_per_token",
        ):
            _positive_integer(getattr(self, name), label=name)
        for name in (
            "downstream_score",
            "nll",
            "teacher_kl",
            "top1_agreement",
            "operator_nrmse",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), label=name),
            )
        if (
            self.nll < 0.0
            or self.teacher_kl < 0.0
            or not 0.0 <= self.top1_agreement <= 1.0
            or self.operator_nrmse < 0.0
        ):
            raise ValueError("compression quality metrics are out of range")
        if self.measured_latency_ms is not None:
            latency = _finite_float(
                self.measured_latency_ms,
                label="measured_latency_ms",
            )
            if latency <= 0.0:
                raise ValueError(
                    "measured_latency_ms must be positive when supplied"
                )
            object.__setattr__(self, "measured_latency_ms", latency)
        for name in ("hardware_id", "benchmark_protocol"):
            value = getattr(self, name)
            if value is not None:
                _nonempty_string(value, label=name)
        if (
            self.measured_latency_ms is not None
            and (
                self.hardware_id is None
                or self.benchmark_protocol is None
            )
        ):
            raise ValueError(
                "measured latency requires hardware_id and "
                "benchmark_protocol"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": {
                "candidate_id": self.candidate_id,
                "method": self.method,
                "execution_fingerprint": (
                    self.candidate_execution_fingerprint
                ),
                "report_sha256": self.candidate_report_sha256,
            },
            "evaluation": {
                "evaluation_id": self.evaluation_id,
                "split_sha256": self.evaluation_split_sha256,
                "task_suite": self.task_suite,
            },
            "resources": {
                "parameter_scope": self.parameter_scope,
                "compute_scope": self.compute_scope,
                "runtime_dtype": self.runtime_dtype,
                "runtime": self.runtime,
                "learned_parameters": self.learned_parameters,
                "runtime_parameter_bytes": self.runtime_parameter_bytes,
                "logical_macs_per_token": self.logical_macs_per_token,
                "measured_latency_ms": self.measured_latency_ms,
                "hardware_id": self.hardware_id,
                "benchmark_protocol": self.benchmark_protocol,
            },
            "quality": {
                "downstream_score": self.downstream_score,
                "nll": self.nll,
                "teacher_kl": self.teacher_kl,
                "top1_agreement": self.top1_agreement,
                "operator_nrmse": self.operator_nrmse,
            },
        }


def _axis_value(point: CompressionRatePoint, axis: str) -> float:
    value = getattr(point, axis)
    if value is None:
        raise ValueError(
            f"candidate {point.candidate_id!r} has no value for {axis}"
        )
    return float(value)


def _single_comparability_value(
    points: Sequence[CompressionRatePoint],
    attribute: str,
    *,
    label: str,
) -> object:
    values = {getattr(point, attribute) for point in points}
    if len(values) != 1:
        raise ValueError(
            f"Pareto points have incomparable {label}: "
            f"{sorted(repr(value) for value in values)}"
        )
    return next(iter(values))


def _validate_pareto_comparability(
    points: Sequence[CompressionRatePoint],
    *,
    rate_axis: str,
) -> None:
    for attribute, label in (
        ("evaluation_id", "evaluation ids"),
        ("evaluation_split_sha256", "evaluation splits"),
        ("task_suite", "task suites"),
    ):
        _single_comparability_value(
            points,
            attribute,
            label=label,
        )
    if rate_axis in {"learned_parameters", "runtime_parameter_bytes"}:
        _single_comparability_value(
            points,
            "parameter_scope",
            label="parameter scopes",
        )
    elif rate_axis == "logical_macs_per_token":
        _single_comparability_value(
            points,
            "compute_scope",
            label="compute scopes",
        )
    elif rate_axis == "measured_latency_ms":
        _single_comparability_value(
            points,
            "runtime",
            label="runtimes",
        )
        hardware = _single_comparability_value(
            points,
            "hardware_id",
            label="hardware ids",
        )
        protocol = _single_comparability_value(
            points,
            "benchmark_protocol",
            label="benchmark protocols",
        )
        if hardware is None or protocol is None:
            raise ValueError(
                "latency Pareto points require hardware and benchmark "
                "protocol identities"
            )


def compression_pareto_frontier(
    points: Sequence[CompressionRatePoint],
    *,
    rate_axis: str = "runtime_parameter_bytes",
    quality_axis: str = "downstream_score",
) -> tuple[CompressionRatePoint, ...]:
    """Return nondominated points for one rate and one quality axis."""

    values = tuple(points)
    if not values or any(
        not isinstance(point, CompressionRatePoint)
        for point in values
    ):
        raise ValueError(
            "points must be a nonempty CompressionRatePoint sequence"
        )
    if rate_axis not in _RATE_AXES:
        raise ValueError(f"unsupported rate axis {rate_axis!r}")
    if quality_axis not in _QUALITY_DIRECTIONS:
        raise ValueError(f"unsupported quality axis {quality_axis!r}")
    for point in values:
        _axis_value(point, rate_axis)
    _validate_pareto_comparability(values, rate_axis=rate_axis)
    direction = _QUALITY_DIRECTIONS[quality_axis]
    result = []
    for candidate in values:
        candidate_rate = _axis_value(candidate, rate_axis)
        candidate_quality = _axis_value(candidate, quality_axis)
        dominated = False
        for challenger in values:
            if challenger is candidate:
                continue
            challenger_rate = _axis_value(challenger, rate_axis)
            challenger_quality = _axis_value(challenger, quality_axis)
            quality_no_worse = (
                challenger_quality >= candidate_quality
                if direction == "maximize"
                else challenger_quality <= candidate_quality
            )
            quality_better = (
                challenger_quality > candidate_quality
                if direction == "maximize"
                else challenger_quality < candidate_quality
            )
            if (
                challenger_rate <= candidate_rate
                and quality_no_worse
                and (
                    challenger_rate < candidate_rate
                    or quality_better
                )
            ):
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    quality_sign = -1.0 if direction == "maximize" else 1.0
    return tuple(
        sorted(
            result,
            key=lambda point: (
                _axis_value(point, rate_axis),
                quality_sign * _axis_value(point, quality_axis),
                point.candidate_id,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class CompressionRateDistortionCurve:
    """All raw points plus reproducible on-demand Pareto projections."""

    points: tuple[CompressionRatePoint, ...]

    def __post_init__(self) -> None:
        if (
            type(self.points) is not tuple
            or not self.points
            or any(
                not isinstance(point, CompressionRatePoint)
                for point in self.points
            )
        ):
            raise ValueError(
                "points must be a nonempty tuple of compression points"
            )
        ids = tuple(point.candidate_id for point in self.points)
        if len(set(ids)) != len(ids):
            raise ValueError("compression candidate ids must be unique")

    def frontier(
        self,
        *,
        rate_axis: str = "runtime_parameter_bytes",
        quality_axis: str = "downstream_score",
    ) -> tuple[CompressionRatePoint, ...]:
        return compression_pareto_frontier(
            self.points,
            rate_axis=rate_axis,
            quality_axis=quality_axis,
        )

    def to_dict(
        self,
        *,
        rate_axis: str = "runtime_parameter_bytes",
        quality_axis: str = "downstream_score",
    ) -> dict[str, object]:
        frontier = self.frontier(
            rate_axis=rate_axis,
            quality_axis=quality_axis,
        )
        return {
            "points": tuple(point.to_dict() for point in self.points),
            "pareto": {
                "rate_axis": rate_axis,
                "rate_direction": "minimize",
                "quality_axis": quality_axis,
                "quality_direction": _QUALITY_DIRECTIONS[quality_axis],
                "candidate_ids": tuple(
                    point.candidate_id for point in frontier
                ),
            },
            "raw_points_retained_even_when_dominated": True,
        }


def _required_mapping(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _validated_dense_supermode_resources(
    candidate: StructuredMLPDenseSupermodeCandidate,
) -> tuple[int, int, int, str, str, str]:
    if not isinstance(candidate, StructuredMLPDenseSupermodeCandidate):
        raise TypeError(
            "candidate must be a StructuredMLPDenseSupermodeCandidate"
        )
    candidate.validate_integrity()
    report = _required_mapping(
        candidate.report,
        label="candidate report",
    )
    if report.get("schema") != (
        STRUCTURED_MLP_DENSE_SUPERMODE_PIPELINE_SCHEMA
    ):
        raise ValueError("candidate has the wrong dense-supermode schema")
    claimed_report_sha256 = _require_sha256(
        report.get("report_sha256"),
        label="candidate report_sha256",
    )
    execution_fingerprint = _require_sha256(
        candidate.executor.execution_fingerprint(),
        label="candidate execution fingerprint",
    )

    resources = _required_mapping(
        report.get("resources"),
        label="candidate resources",
    )
    parameters = _required_mapping(
        resources.get("parameters"),
        label="candidate parameter accounting",
    )
    parameter_bytes = _required_mapping(
        resources.get("parameter_bytes_in_executor_dtype"),
        label="candidate byte accounting",
    )
    compute = _required_mapping(
        resources.get("compute_per_valid_token"),
        label="candidate compute accounting",
    )
    macs = _required_mapping(
        compute.get("macs"),
        label="candidate MAC accounting",
    )
    learned_parameters = _positive_integer(
        parameters.get("candidate_full_layer"),
        label="candidate_full_layer parameters",
    )
    runtime_parameter_bytes = _positive_integer(
        parameter_bytes.get("candidate_full_layer"),
        label="candidate_full_layer parameter bytes",
    )
    logical_macs_per_token = _positive_integer(
        macs.get("candidate"),
        label="candidate logical MACs/token",
    )
    if learned_parameters != candidate.executor.learned_parameter_count:
        raise ValueError(
            "candidate parameter accounting does not match its executor"
        )
    actual_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in candidate.executor.parameters()
    )
    if runtime_parameter_bytes != actual_parameter_bytes:
        raise ValueError(
            "candidate byte accounting does not match its executor"
        )
    if (
        parameters.get("scope") != _DENSE_SUPERMODE_PARAMETER_SCOPE
        or parameter_bytes.get("scope")
        != _DENSE_SUPERMODE_PARAMETER_SCOPE
    ):
        raise ValueError("candidate parameter scope is not canonical")
    if parameter_bytes.get("executor_dtype") != str(
        candidate.executor.dtype
    ):
        raise ValueError(
            "candidate byte-accounting dtype does not match its executor"
        )
    if compute.get("scope") != _DENSE_SUPERMODE_REPORT_COMPUTE_SCOPE:
        raise ValueError("candidate compute scope is not canonical")
    rung = _required_mapping(
        report.get("rung"),
        label="candidate rung",
    )
    runtime_width = _positive_integer(
        rung.get("runtime_intermediate_width"),
        label="runtime_intermediate_width",
    )
    configured_width = (
        candidate.executor.config.transformer.feed_forward.intermediate_width
    )
    expected_macs = 3 * candidate.executor.width * runtime_width
    if (
        runtime_width != configured_width
        or logical_macs_per_token != expected_macs
    ):
        raise ValueError(
            "candidate MAC accounting does not match its executor shape"
        )
    parameter_dtypes = {
        str(parameter.dtype)
        for parameter in candidate.executor.parameters()
    }
    runtime_dtype = (
        next(iter(parameter_dtypes))
        if len(parameter_dtypes) == 1
        else "mixed[" + ",".join(sorted(parameter_dtypes)) + "]"
    )
    return (
        learned_parameters,
        runtime_parameter_bytes,
        logical_macs_per_token,
        execution_fingerprint,
        claimed_report_sha256,
        runtime_dtype,
    )


def dense_supermode_rate_point_from_candidate(
    candidate: StructuredMLPDenseSupermodeCandidate,
    *,
    candidate_id: str,
    evaluation_id: str,
    evaluation_split_sha256: str,
    task_suite: str,
    runtime: str,
    downstream_score: float,
    nll: float,
    teacher_kl: float,
    top1_agreement: float,
    operator_nrmse: float,
    measured_latency_ms: float | None = None,
    hardware_id: str | None = None,
    benchmark_protocol: str | None = None,
) -> CompressionRatePoint:
    """Bind quality to an integrity-checked dense-supermode candidate bundle."""

    (
        learned_parameters,
        runtime_parameter_bytes,
        logical_macs_per_token,
        execution_fingerprint,
        report_sha256,
        runtime_dtype,
    ) = _validated_dense_supermode_resources(candidate)
    return CompressionRatePoint(
        candidate_id=candidate_id,
        method="dense_supermode",
        evaluation_id=evaluation_id,
        evaluation_split_sha256=evaluation_split_sha256,
        task_suite=task_suite,
        candidate_execution_fingerprint=execution_fingerprint,
        candidate_report_sha256=report_sha256,
        parameter_scope=_DENSE_SUPERMODE_PARAMETER_SCOPE,
        compute_scope=_DENSE_SUPERMODE_COMPUTE_SCOPE,
        runtime_dtype=runtime_dtype,
        runtime=runtime,
        learned_parameters=learned_parameters,
        runtime_parameter_bytes=runtime_parameter_bytes,
        logical_macs_per_token=logical_macs_per_token,
        downstream_score=downstream_score,
        nll=nll,
        teacher_kl=teacher_kl,
        top1_agreement=top1_agreement,
        operator_nrmse=operator_nrmse,
        measured_latency_ms=measured_latency_ms,
        hardware_id=hardware_id,
        benchmark_protocol=benchmark_protocol,
    )


__all__ = [
    "CompressionRateDistortionCurve",
    "CompressionRatePoint",
    "compression_pareto_frontier",
    "dense_supermode_rate_point_from_candidate",
]
