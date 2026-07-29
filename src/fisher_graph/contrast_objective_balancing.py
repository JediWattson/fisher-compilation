"""Scale-stable Fisher gauges and contrast-objective diagnostics.

The contrast-aware fitter deliberately accepts an explicit Fisher metric.
Its pointwise and intended-null components are absolute squared errors in
that metric, while relative-delta, direction-cosine, and relative-JVP terms
are invariant to a common rescaling only while their fixed numerical floors
are inactive.  Passing a raw Fisher square-root gauge can therefore make the
absolute terms dominate because of the arbitrary global units of the Fisher
spectrum, but unit-RMS normalization alone does not prove that only absolute
terms changed in a floor-active run.

This additive module removes only that global scale.  It preserves every
mode's relative Fisher weight and leaves the artifact-bound fitter unchanged.
Call :meth:`UnitRmsFisherGauge.from_metric_weight`, retain the returned state
as provenance, and pass ``gauge.metric_weight`` to the existing fitter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import Tensor

from .state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastTrainingMetrics,
)


__all__ = [
    "ObjectiveContributionAudit",
    "UnitRmsFisherGauge",
    "audit_objective_contributions",
]


_FORMAT_VERSION = 1
_SHA256 = frozenset("0123456789abcdef")
_TENSOR_DOMAIN = b"fisher_graph.unit_rms_fisher_gauge.tensor.v1\0"
_GAUGE_DOMAIN = b"fisher_graph.unit_rms_fisher_gauge.v1\0"
_UNIT_RMS_RTOL = 1e-12
_UNIT_RMS_ATOL = 1e-12


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(canonical.dtype),
            "shape": tuple(int(width) for width in canonical.shape),
        }
    )
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + header
        + b"\0"
        + canonical.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_metric_weight(
    value: Tensor,
    *,
    label: str,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != 1 or value.numel() <= 0:
        raise ValueError(f"{label} must be a nonempty rank-one Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must use a floating dtype")
    canonical = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError(f"{label} must be finite")
    if bool((canonical <= 0.0).any()):
        raise ValueError(f"{label} must be strictly positive")
    return canonical


def _metric_rms(value: Tensor) -> float:
    return float(torch.sqrt(value.square().mean()))


@dataclass(frozen=True, slots=True)
class UnitRmsFisherGauge:
    """Authenticated global normalization of a Fisher square-root gauge.

    The stored weights are always detached CPU ``float64`` values with
    ``mean(weight**2) == 1``.  ``raw_metric_weight_sha256`` and ``raw_rms``
    bind the source gauge without retaining a second copy of it.
    """

    metric_weight: Tensor
    raw_metric_weight_sha256: str
    raw_rms: float
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        metric = _canonical_metric_weight(
            self.metric_weight,
            label="normalized Fisher metric weight",
        )
        normalized_rms = _metric_rms(metric)
        if not math.isclose(
            normalized_rms,
            1.0,
            rel_tol=_UNIT_RMS_RTOL,
            abs_tol=_UNIT_RMS_ATOL,
        ):
            raise ValueError("normalized Fisher metric must have unit RMS")
        raw_sha = _require_sha256(
            self.raw_metric_weight_sha256,
            label="raw Fisher metric binding",
        )
        raw_rms = float(self.raw_rms)
        if not math.isfinite(raw_rms) or raw_rms <= 0.0:
            raise ValueError("raw_rms must be finite and positive")
        object.__setattr__(self, "metric_weight", metric)
        object.__setattr__(self, "raw_metric_weight_sha256", raw_sha)
        object.__setattr__(self, "raw_rms", raw_rms)
        payload = self._payload()
        expected = _json_sha256(payload, domain=_GAUGE_DOMAIN)
        if self.artifact_sha256:
            supplied = _require_sha256(
                self.artifact_sha256,
                label="unit-RMS Fisher gauge artifact",
            )
            if supplied != expected:
                raise ValueError("unit-RMS Fisher gauge artifact is invalid")
        object.__setattr__(self, "artifact_sha256", expected)

    @classmethod
    def from_metric_weight(cls, metric_weight: Tensor) -> UnitRmsFisherGauge:
        """Normalize one positive Fisher square-root gauge to unit RMS."""

        if not isinstance(metric_weight, Tensor):
            raise TypeError("Fisher metric weight must be a Tensor")
        raw_sha = _tensor_sha256(metric_weight)
        canonical = _canonical_metric_weight(
            metric_weight,
            label="Fisher metric weight",
        )
        raw_rms = _metric_rms(canonical)
        normalized = (canonical / raw_rms).contiguous()
        return cls(
            metric_weight=normalized,
            raw_metric_weight_sha256=raw_sha,
            raw_rms=raw_rms,
        )

    @property
    def raw_mean_square(self) -> float:
        """Return the absolute-loss divisor induced by normalization."""

        return self.raw_rms**2

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.unit_rms_fisher_gauge",
            "format_version": _FORMAT_VERSION,
            "normalization_semantics": (
                "positive_sqrt_fisher_weights_divided_by_global_rms_"
                "relative_mode_weights_preserved"
            ),
            "width": int(self.metric_weight.numel()),
            "raw_metric_weight_sha256": self.raw_metric_weight_sha256,
            "raw_rms": self.raw_rms,
            "raw_mean_square": self.raw_mean_square,
            "normalized_metric_weight_sha256": _tensor_sha256(
                self.metric_weight
            ),
            "normalized_rms": _metric_rms(self.metric_weight),
        }

    def state_dict(self) -> dict[str, object]:
        """Return an authenticated, round-trippable gauge state."""

        return {
            **self._payload(),
            "metric_weight": self.metric_weight.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> UnitRmsFisherGauge:
        if not isinstance(raw, Mapping):
            raise TypeError("unit-RMS Fisher gauge state must be a mapping")
        expected = {
            "artifact_kind",
            "format_version",
            "normalization_semantics",
            "width",
            "raw_metric_weight_sha256",
            "raw_rms",
            "raw_mean_square",
            "normalized_metric_weight_sha256",
            "normalized_rms",
            "metric_weight",
            "artifact_sha256",
        }
        if set(raw) != expected:
            raise ValueError("unit-RMS Fisher gauge state keys are invalid")
        metric = raw["metric_weight"]
        if not isinstance(metric, Tensor):
            raise TypeError("stored normalized Fisher metric must be a Tensor")
        result = cls(
            metric_weight=metric,
            raw_metric_weight_sha256=str(
                raw["raw_metric_weight_sha256"]
            ),
            raw_rms=raw["raw_rms"],  # type: ignore[arg-type]
            artifact_sha256=str(raw["artifact_sha256"]),
        )
        payload = result._payload()
        for name in expected - {"metric_weight", "artifact_sha256"}:
            if raw[name] != payload[name]:
                raise ValueError(
                    f"stored unit-RMS Fisher gauge field {name!r} is invalid"
                )
        return result

    def validate_source(self, metric_weight: Tensor) -> None:
        """Fail closed unless ``metric_weight`` is the bound raw source."""

        if not isinstance(metric_weight, Tensor):
            raise TypeError("Fisher metric weight must be a Tensor")
        if _tensor_sha256(metric_weight) != self.raw_metric_weight_sha256:
            raise ValueError("raw Fisher metric binding differs")
        canonical = _canonical_metric_weight(
            metric_weight,
            label="Fisher metric weight",
        )
        if not math.isclose(
            _metric_rms(canonical),
            self.raw_rms,
            rel_tol=1e-13,
            abs_tol=1e-13,
        ):
            raise ValueError("raw Fisher metric RMS differs")
        expected = canonical / self.raw_rms
        if not torch.equal(expected, self.metric_weight):
            raise ValueError("normalized Fisher metric differs from source")


@dataclass(frozen=True, slots=True)
class ObjectiveContributionAudit:
    """Scalar contribution and fraction report for one frozen objective."""

    pointwise: float
    sensitivity_relative_delta: float
    sensitivity_direction: float
    midpoint_jvp: float
    intended_null: float
    total: float
    pointwise_fraction: float
    contrast_fraction: float
    intended_null_fraction: float
    reported_total: float
    reported_total_matches: bool
    objective_sha256: str
    metrics_sha256: str

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.objective_contribution_audit",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in (
                    "pointwise",
                    "sensitivity_relative_delta",
                    "sensitivity_direction",
                    "midpoint_jvp",
                    "intended_null",
                    "total",
                    "pointwise_fraction",
                    "contrast_fraction",
                    "intended_null_fraction",
                    "reported_total",
                    "reported_total_matches",
                    "objective_sha256",
                    "metrics_sha256",
                )
            },
        }


def audit_objective_contributions(
    metrics: ContrastTrainingMetrics,
    objective: ContrastAwareObjective,
) -> ObjectiveContributionAudit:
    """Explain the scalar loss budget under ``objective``.

    Relative-delta, direction, and midpoint-JVP terms are grouped as contrast
    contributions.  A zero total has defined zero fractions rather than a
    division-by-zero sentinel.
    """

    if not isinstance(metrics, ContrastTrainingMetrics):
        raise TypeError("metrics must be ContrastTrainingMetrics")
    if not isinstance(objective, ContrastAwareObjective):
        raise TypeError("objective must be ContrastAwareObjective")
    pointwise = objective.pointwise_weight * metrics.pointwise_mse
    sensitivity_delta = (
        objective.sensitivity_relative_delta_weight
        * metrics.sensitivity_relative_delta_mse
    )
    sensitivity_direction = (
        objective.sensitivity_direction_weight
        * metrics.sensitivity_direction_loss
    )
    midpoint_jvp = (
        objective.midpoint_jvp_weight
        * metrics.midpoint_jvp_relative_mse
    )
    intended_null = (
        objective.intended_null_weight
        * metrics.intended_null_absolute_mse
    )
    contrast = sensitivity_delta + sensitivity_direction + midpoint_jvp
    total = pointwise + contrast + intended_null
    if total > 0.0:
        pointwise_fraction = pointwise / total
        contrast_fraction = contrast / total
        intended_null_fraction = intended_null / total
    else:
        pointwise_fraction = 0.0
        contrast_fraction = 0.0
        intended_null_fraction = 0.0
    return ObjectiveContributionAudit(
        pointwise=pointwise,
        sensitivity_relative_delta=sensitivity_delta,
        sensitivity_direction=sensitivity_direction,
        midpoint_jvp=midpoint_jvp,
        intended_null=intended_null,
        total=total,
        pointwise_fraction=pointwise_fraction,
        contrast_fraction=contrast_fraction,
        intended_null_fraction=intended_null_fraction,
        reported_total=metrics.weighted_total,
        reported_total_matches=math.isclose(
            total,
            metrics.weighted_total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        objective_sha256=objective.artifact_sha256,
        metrics_sha256=metrics.artifact_sha256,
    )
