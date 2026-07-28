"""Contrast-aware fitting for a packed, full-width reference provider.

This module is intentionally additive.  It does not alter the hash-bound V2
reference-provider implementation.  A packed provider observes all 64 source
modes, projects them into a learned latent rank, executes the existing causal
modal core at that rank, and decodes back to all 64 target modes.

The exact RMS-null coordinate is not an executor feature.  It is used only to
remove its analytically known contribution from row RMS before constructing a
nonnull gain feature.  Consequently two physically consistent states that
differ only in the exact null component have identical prepared features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal

import torch
from torch import Tensor, nn

from .gated_executor import (
    GatedCausalExecutionAccounting,
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)
from .state_conditioned_reference_provider import SyntheticReferenceBatch


__all__ = [
    "ContrastAwareExecutionAccounting",
    "ContrastAwareObjective",
    "ContrastAwareReferenceProviderAccounting",
    "ContrastAwareReferenceProviderPlan",
    "ContrastTrainingMetrics",
    "IndexedReferenceBatch",
    "PreparedContrastAwareReferenceProvider",
    "ReferenceProviderContrastPair",
    "evaluate_contrast_aware_reference_provider",
    "fit_contrast_aware_reference_provider",
]


_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_TARGET_WIDTH = 64
_SHA256 = frozenset("0123456789abcdef")
_INDEXED_BATCH_DOMAIN = b"fisher_graph.contrast_fit.indexed_batch.v1\0"
_PAIR_DOMAIN = b"fisher_graph.contrast_fit.pair.v1\0"
_OBJECTIVE_DOMAIN = b"fisher_graph.contrast_fit.objective.v1\0"
_METRICS_DOMAIN = b"fisher_graph.contrast_fit.metrics.v1\0"
_PLAN_DOMAIN = b"fisher_graph.contrast_fit.packed_plan.v1\0"
_ENDPOINT_DOMAIN = b"fisher_graph.contrast_fit.endpoint.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.contrast_fit.tensor.v1\0"
_RUNTIME_DOMAIN = b"fisher_graph.contrast_fit.runtime.v1\0"

ContrastRole = Literal["expected_sensitivity", "intended_null"]
_PROVIDER_CHART_FIELDS = (
    "provider_chart_modal_primal",
    "provider_chart_null_primal",
    "provider_chart_row_rms_primal",
    "provider_chart_modal_tangent",
    "provider_chart_null_tangent",
    "provider_chart_row_rms_tangent",
)


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


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


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


def _canonical_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point() or value.ndim != ndim:
        raise TypeError(f"{label} must be a rank-{ndim} floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if any(int(width) <= 0 for width in result.shape):
        raise ValueError(f"{label} must be nonempty")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _strict_keys(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(state) != expected:
        raise ValueError(f"{label} fields do not match the frozen format")


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest_or_validate(
    supplied: str,
    payload: object,
    *,
    domain: bytes,
    label: str,
) -> str:
    expected = _json_sha256(payload, domain=domain)
    if supplied and _require_sha256(supplied, label=label) != expected:
        raise ValueError(f"{label} hash mismatch")
    return expected


def _sha_tuple(
    values: object,
    *,
    label: str,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_require_sha256(value, label=label) for value in values)
    if nonempty and not result:
        raise ValueError(f"{label} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class IndexedReferenceBatch:
    """A synthetic batch with stable endpoint identities for pair lookup."""

    batch: SyntheticReferenceBatch
    endpoint_ids: tuple[str, ...]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.batch, SyntheticReferenceBatch):
            raise TypeError("batch must be a SyntheticReferenceBatch")
        self.batch.validate_integrity()
        if (
            type(self.endpoint_ids) is not tuple
            or len(self.endpoint_ids) != self.batch.batch_size
            or len(set(self.endpoint_ids)) != len(self.endpoint_ids)
        ):
            raise ValueError(
                "endpoint_ids must be unique and match batch_size"
            )
        for value in self.endpoint_ids:
            _nonempty_string(value, label="endpoint id")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                self.artifact_sha256,
                self._payload(),
                domain=_INDEXED_BATCH_DOMAIN,
                label="indexed batch",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.indexed_reference_batch",
            "format_version": _FORMAT_VERSION,
            "batch_sha256": self.batch.artifact_sha256,
            "batch_content_sha256": self.batch.content_sha256,
            "endpoint_ids": self.endpoint_ids,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "batch_state": self.batch.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> IndexedReferenceBatch:
        expected = {
            "artifact_kind",
            "format_version",
            "batch_sha256",
            "batch_content_sha256",
            "endpoint_ids",
            "batch_state",
            "artifact_sha256",
        }
        _strict_keys(raw, expected=expected, label="indexed batch")
        if (
            raw["artifact_kind"] != "fisher_graph.indexed_reference_batch"
            or raw["format_version"] != _FORMAT_VERSION
            or not isinstance(raw["batch_state"], Mapping)
        ):
            raise ValueError("indexed batch envelope is invalid")
        batch = SyntheticReferenceBatch.from_state_dict(raw["batch_state"])
        if (
            raw["batch_sha256"] != batch.artifact_sha256
            or raw["batch_content_sha256"] != batch.content_sha256
            or not isinstance(raw["endpoint_ids"], tuple)
        ):
            raise ValueError("indexed batch binding is invalid")
        return cls(
            batch=batch,
            endpoint_ids=raw["endpoint_ids"],  # type: ignore[arg-type]
            artifact_sha256=str(raw["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ReferenceProviderContrastPair:
    """One declared finite-displacement or intended-null comparison."""

    pair_id: str
    family: str
    role: ContrastRole
    left_endpoint_id: str
    right_endpoint_id: str
    rank_stratum: str
    teacher_midpoint_jvp: Tensor | None = None
    provider_chart_modal_primal: Tensor | None = None
    provider_chart_null_primal: Tensor | None = None
    provider_chart_row_rms_primal: Tensor | None = None
    provider_chart_modal_tangent: Tensor | None = None
    provider_chart_null_tangent: Tensor | None = None
    provider_chart_row_rms_tangent: Tensor | None = None
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "family",
            "left_endpoint_id",
            "right_endpoint_id",
            "rank_stratum",
        ):
            _nonempty_string(getattr(self, name), label=name)
        if self.left_endpoint_id == self.right_endpoint_id:
            raise ValueError("contrast pair endpoints must differ")
        if self.role not in ("expected_sensitivity", "intended_null"):
            raise ValueError("contrast pair role is invalid")
        teacher = self.teacher_midpoint_jvp
        chart_values = {
            name: getattr(self, name) for name in _PROVIDER_CHART_FIELDS
        }
        chart_presence = tuple(
            value is not None for value in chart_values.values()
        )
        if teacher is None:
            if any(chart_presence):
                raise ValueError(
                    "provider-chart JVP tensors require a teacher JVP"
                )
        elif not all(chart_presence):
            raise ValueError(
                "teacher JVP requires the complete provider-chart primal "
                "and push-forward tangent"
            )
        if teacher is not None:
            if self.role != "expected_sensitivity":
                raise ValueError(
                    "only expected-sensitivity pairs may carry a teacher JVP"
                )
            teacher = _canonical_float_tensor(
                teacher,
                label="teacher_midpoint_jvp",
                ndim=2,
            )
            if teacher.shape[1] != _TARGET_WIDTH:
                raise ValueError(
                    "teacher_midpoint_jvp must have 64 target modes"
                )
            canonical_chart = {
                "provider_chart_modal_primal": _canonical_float_tensor(
                    chart_values["provider_chart_modal_primal"],
                    label="provider_chart_modal_primal",
                    ndim=2,
                ),
                "provider_chart_null_primal": _canonical_float_tensor(
                    chart_values["provider_chart_null_primal"],
                    label="provider_chart_null_primal",
                    ndim=2,
                ),
                "provider_chart_row_rms_primal": _canonical_float_tensor(
                    chart_values["provider_chart_row_rms_primal"],
                    label="provider_chart_row_rms_primal",
                    ndim=1,
                ),
                "provider_chart_modal_tangent": _canonical_float_tensor(
                    chart_values["provider_chart_modal_tangent"],
                    label="provider_chart_modal_tangent",
                    ndim=2,
                ),
                "provider_chart_null_tangent": _canonical_float_tensor(
                    chart_values["provider_chart_null_tangent"],
                    label="provider_chart_null_tangent",
                    ndim=2,
                ),
                "provider_chart_row_rms_tangent": _canonical_float_tensor(
                    chart_values["provider_chart_row_rms_tangent"],
                    label="provider_chart_row_rms_tangent",
                    ndim=1,
                ),
            }
            sequence_length = int(teacher.shape[0])
            expected_shapes = {
                "provider_chart_modal_primal": (
                    sequence_length,
                    _MODAL_WIDTH,
                ),
                "provider_chart_null_primal": (sequence_length, 1),
                "provider_chart_row_rms_primal": (sequence_length,),
                "provider_chart_modal_tangent": (
                    sequence_length,
                    _MODAL_WIDTH,
                ),
                "provider_chart_null_tangent": (sequence_length, 1),
                "provider_chart_row_rms_tangent": (sequence_length,),
            }
            for name, expected_shape in expected_shapes.items():
                if canonical_chart[name].shape != expected_shape:
                    raise ValueError(
                        f"{name} must have shape {expected_shape}"
                    )
                object.__setattr__(self, name, canonical_chart[name])
            if bool(
                (
                    canonical_chart["provider_chart_row_rms_primal"]
                    <= 0.0
                ).any()
            ):
                raise ValueError(
                    "provider_chart_row_rms_primal must be positive"
                )
        object.__setattr__(self, "teacher_midpoint_jvp", teacher)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                self.artifact_sha256,
                self._payload(),
                domain=_PAIR_DOMAIN,
                label="contrast pair",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.reference_provider_contrast_pair",
            "format_version": _FORMAT_VERSION,
            "pair_id": self.pair_id,
            "family": self.family,
            "role": self.role,
            "left_endpoint_id": self.left_endpoint_id,
            "right_endpoint_id": self.right_endpoint_id,
            "rank_stratum": self.rank_stratum,
            "teacher_midpoint_jvp_sha256": (
                None
                if self.teacher_midpoint_jvp is None
                else _tensor_sha256(self.teacher_midpoint_jvp)
            ),
            **{
                f"{name}_sha256": (
                    None
                    if getattr(self, name) is None
                    else _tensor_sha256(getattr(self, name))
                )
                for name in _PROVIDER_CHART_FIELDS
            },
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "teacher_midpoint_jvp": (
                None
                if self.teacher_midpoint_jvp is None
                else self.teacher_midpoint_jvp.clone()
            ),
            **{
                name: (
                    None
                    if getattr(self, name) is None
                    else getattr(self, name).clone()
                )
                for name in _PROVIDER_CHART_FIELDS
            },
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> ReferenceProviderContrastPair:
        expected = {
            "artifact_kind",
            "format_version",
            "pair_id",
            "family",
            "role",
            "left_endpoint_id",
            "right_endpoint_id",
            "rank_stratum",
            "teacher_midpoint_jvp_sha256",
            "teacher_midpoint_jvp",
            *(
                f"{name}_sha256" for name in _PROVIDER_CHART_FIELDS
            ),
            *_PROVIDER_CHART_FIELDS,
            "artifact_sha256",
        }
        _strict_keys(raw, expected=expected, label="contrast pair")
        if (
            raw["artifact_kind"]
            != "fisher_graph.reference_provider_contrast_pair"
            or raw["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("contrast pair envelope is invalid")
        teacher = raw["teacher_midpoint_jvp"]
        if teacher is not None and not isinstance(teacher, Tensor):
            raise TypeError("stored teacher_midpoint_jvp must be a Tensor")
        for name in _PROVIDER_CHART_FIELDS:
            value = raw[name]
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"stored {name} must be a Tensor")
        result = cls(
            pair_id=str(raw["pair_id"]),
            family=str(raw["family"]),
            role=str(raw["role"]),  # type: ignore[arg-type]
            left_endpoint_id=str(raw["left_endpoint_id"]),
            right_endpoint_id=str(raw["right_endpoint_id"]),
            rank_stratum=str(raw["rank_stratum"]),
            teacher_midpoint_jvp=teacher,
            **{name: raw[name] for name in _PROVIDER_CHART_FIELDS},
            artifact_sha256=str(raw["artifact_sha256"]),
        )
        expected_teacher_sha = (
            None
            if result.teacher_midpoint_jvp is None
            else _tensor_sha256(result.teacher_midpoint_jvp)
        )
        if raw["teacher_midpoint_jvp_sha256"] != expected_teacher_sha:
            raise ValueError("stored teacher JVP binding is invalid")
        for name in _PROVIDER_CHART_FIELDS:
            value = getattr(result, name)
            expected_sha = (
                None if value is None else _tensor_sha256(value)
            )
            if raw[f"{name}_sha256"] != expected_sha:
                raise ValueError(
                    f"stored {name} binding is invalid"
                )
        return result


@dataclass(frozen=True, slots=True)
class ContrastAwareObjective:
    """Frozen component weights and numerical floors for one fit."""

    pointwise_weight: float = 1.0
    sensitivity_relative_delta_weight: float = 1.0
    sensitivity_direction_weight: float = 0.25
    midpoint_jvp_weight: float = 1.0
    intended_null_weight: float = 1.0
    sensitivity_relative_floor: float = 1e-6
    direction_norm_floor: float = 1e-8
    jvp_relative_floor: float = 1e-6
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "pointwise_weight",
            "sensitivity_relative_delta_weight",
            "sensitivity_direction_weight",
            "midpoint_jvp_weight",
            "intended_null_weight",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), label=name),
            )
        if not any(
            getattr(self, name) > 0.0
            for name in (
                "pointwise_weight",
                "sensitivity_relative_delta_weight",
                "sensitivity_direction_weight",
                "midpoint_jvp_weight",
                "intended_null_weight",
            )
        ):
            raise ValueError("at least one objective weight must be positive")
        for name in (
            "sensitivity_relative_floor",
            "direction_norm_floor",
            "jvp_relative_floor",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                self.artifact_sha256,
                self._payload(),
                domain=_OBJECTIVE_DOMAIN,
                label="contrast objective",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.contrast_aware_objective",
            "format_version": _FORMAT_VERSION,
            "pointwise_weight": self.pointwise_weight,
            "sensitivity_relative_delta_weight": (
                self.sensitivity_relative_delta_weight
            ),
            "sensitivity_direction_weight": (
                self.sensitivity_direction_weight
            ),
            "midpoint_jvp_weight": self.midpoint_jvp_weight,
            "intended_null_weight": self.intended_null_weight,
            "sensitivity_relative_floor": self.sensitivity_relative_floor,
            "direction_norm_floor": self.direction_norm_floor,
            "jvp_relative_floor": self.jvp_relative_floor,
            "pointwise_semantics": (
                "mean_squared_standardized_error_times_optional_fisher_weight"
            ),
            "sensitivity_semantics": (
                "pair_relative_delta_mse_plus_one_minus_direction_cosine"
            ),
            "jvp_semantics": (
                "torch_func_jvp_of_candidate_at_supplied_hidden_midpoint_"
                "provider_chart_primal_along_supplied_push_forward_tangent_"
                "against_teacher_midpoint_jvp_divided_by_target_scale_"
                "endpoint_arithmetic_forbidden"
            ),
            "null_semantics": (
                "absolute_standardized_candidate_delta_mse_never_divided_by_"
                "teacher_null_delta"
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> ContrastAwareObjective:
        expected = set(cls().state_dict())
        _strict_keys(raw, expected=expected, label="contrast objective")
        defaults = cls()._payload()
        for name in (
            "artifact_kind",
            "format_version",
            "pointwise_semantics",
            "sensitivity_semantics",
            "jvp_semantics",
            "null_semantics",
        ):
            if raw[name] != defaults[name]:
                raise ValueError("contrast objective semantics drifted")
        return cls(
            pointwise_weight=raw["pointwise_weight"],  # type: ignore[arg-type]
            sensitivity_relative_delta_weight=raw[
                "sensitivity_relative_delta_weight"
            ],  # type: ignore[arg-type]
            sensitivity_direction_weight=raw[
                "sensitivity_direction_weight"
            ],  # type: ignore[arg-type]
            midpoint_jvp_weight=raw[
                "midpoint_jvp_weight"
            ],  # type: ignore[arg-type]
            intended_null_weight=raw[
                "intended_null_weight"
            ],  # type: ignore[arg-type]
            sensitivity_relative_floor=raw[
                "sensitivity_relative_floor"
            ],  # type: ignore[arg-type]
            direction_norm_floor=raw[
                "direction_norm_floor"
            ],  # type: ignore[arg-type]
            jvp_relative_floor=raw[
                "jvp_relative_floor"
            ],  # type: ignore[arg-type]
            artifact_sha256=str(raw["artifact_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ContrastTrainingMetrics:
    """Exact scalar components from one full-batch objective evaluation."""

    pointwise_mse: float
    sensitivity_relative_delta_mse: float
    sensitivity_direction_loss: float
    midpoint_jvp_relative_mse: float
    intended_null_absolute_mse: float
    weighted_total: float
    endpoint_count: int
    sensitivity_pair_count: int
    jvp_pair_count: int
    intended_null_pair_count: int
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "pointwise_mse",
            "sensitivity_relative_delta_mse",
            "sensitivity_direction_loss",
            "midpoint_jvp_relative_mse",
            "intended_null_absolute_mse",
            "weighted_total",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), label=name),
            )
        for name in (
            "endpoint_count",
            "sensitivity_pair_count",
            "jvp_pair_count",
            "intended_null_pair_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), label=name),
            )
        if self.endpoint_count <= 0:
            raise ValueError("training metrics must cover at least one endpoint")
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                self.artifact_sha256,
                self._payload(),
                domain=_METRICS_DOMAIN,
                label="contrast training metrics",
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.contrast_training_metrics",
            "format_version": _FORMAT_VERSION,
            **{
                name: getattr(self, name)
                for name in (
                    "pointwise_mse",
                    "sensitivity_relative_delta_mse",
                    "sensitivity_direction_loss",
                    "midpoint_jvp_relative_mse",
                    "intended_null_absolute_mse",
                    "weighted_total",
                    "endpoint_count",
                    "sensitivity_pair_count",
                    "jvp_pair_count",
                    "intended_null_pair_count",
                )
            },
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> ContrastTrainingMetrics:
        expected = set(
            cls(
                pointwise_mse=0.0,
                sensitivity_relative_delta_mse=0.0,
                sensitivity_direction_loss=0.0,
                midpoint_jvp_relative_mse=0.0,
                intended_null_absolute_mse=0.0,
                weighted_total=0.0,
                endpoint_count=1,
                sensitivity_pair_count=0,
                jvp_pair_count=0,
                intended_null_pair_count=0,
            ).state_dict()
        )
        _strict_keys(raw, expected=expected, label="training metrics")
        if (
            raw["artifact_kind"] != "fisher_graph.contrast_training_metrics"
            or raw["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("training metrics envelope is invalid")
        return cls(
            **{
                name: raw[name]
                for name in expected
                if name
                not in {"artifact_kind", "format_version"}
            }  # type: ignore[arg-type]
        )


def _clone_executor_artifact(
    raw: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("executor artifact must be a mapping")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        restored = ResidualGatedCausalModalExecutor.from_artifact_state_dict(
            raw
        )
    result = restored.artifact_state_dict()
    state = result.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("executor artifact lacks a model state")
    return {
        **result,
        "config": dict(result["config"]),  # type: ignore[arg-type]
        "model_state_dict": {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in sorted(state.items())
            if isinstance(value, Tensor)
        },
    }


def _executor_artifact_sha256(raw: Mapping[str, object]) -> str:
    artifact = _clone_executor_artifact(raw)
    state = artifact["model_state_dict"]
    assert isinstance(state, Mapping)
    return _json_sha256(
        {
            "artifact_kind": artifact["artifact_kind"],
            "format_version": artifact["format_version"],
            "config": dict(artifact["config"]),  # type: ignore[arg-type]
            "dtype": artifact["dtype"],
            "model_state_dict_sha256": {
                name: _tensor_sha256(value)
                for name, value in sorted(state.items())
                if isinstance(value, Tensor)
            },
        },
        domain=_PLAN_DOMAIN,
    )


@dataclass(frozen=True, slots=True)
class ContrastAwareReferenceProviderAccounting:
    modal_modes: int
    latent_rank: int
    target_modes: int
    feature_codec_scalar_count: int
    target_standardization_scalar_count: int
    encoder_parameter_count: int
    executor_parameter_count: int
    decoder_parameter_count: int
    total_stored_scalar_count: int
    training_metric_scalar_count: int


@dataclass(frozen=True, slots=True)
class ContrastAwareExecutionAccounting:
    core: GatedCausalExecutionAccounting
    valid_rows: int
    encoder_mac_count: int
    decoder_mac_count: int
    target_destandardization_mac_count: int
    total_mac_count: int


@dataclass(frozen=True, slots=True)
class ContrastAwareReferenceProviderPlan:
    """Hash-authenticated packed 64→rank→64 provider state."""

    modal_center: Tensor
    gain_log_center: float
    gain_log_scale: float
    residual_width: int
    rms_epsilon: float
    target_center: Tensor
    target_scale: Tensor
    encoder_weight: Tensor
    executor_artifact: Mapping[str, object]
    decoder_weight: Tensor
    fisher_metric_weight: Tensor
    fisher_metric_supplied: bool
    synthetic_binding_sha256: str
    fit_batch_sha256s: tuple[str, ...]
    fit_batch_content_sha256s: tuple[str, ...]
    fit_indexed_batch_sha256s: tuple[str, ...]
    fit_endpoint_sha256s: tuple[str, ...]
    fit_pair_sha256s: tuple[str, ...]
    objective: ContrastAwareObjective
    training_steps: int
    learning_rate: float
    seed: int
    initial_metrics: ContrastTrainingMetrics
    final_metrics: ContrastTrainingMetrics
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        modal_center = _canonical_float_tensor(
            self.modal_center,
            label="modal_center",
            ndim=1,
        )
        target_center = _canonical_float_tensor(
            self.target_center,
            label="target_center",
            ndim=1,
        )
        target_scale = _canonical_float_tensor(
            self.target_scale,
            label="target_scale",
            ndim=1,
        )
        encoder = _canonical_float_tensor(
            self.encoder_weight,
            label="encoder_weight",
            ndim=2,
        )
        decoder = _canonical_float_tensor(
            self.decoder_weight,
            label="decoder_weight",
            ndim=2,
        )
        metric = _canonical_float_tensor(
            self.fisher_metric_weight,
            label="fisher_metric_weight",
            ndim=1,
        )
        if modal_center.shape != (_MODAL_WIDTH,):
            raise ValueError("modal_center must have width 64")
        if (
            target_center.shape != (_TARGET_WIDTH,)
            or target_scale.shape != (_TARGET_WIDTH,)
            or metric.shape != (_TARGET_WIDTH,)
        ):
            raise ValueError("target gauge tensors must have width 64")
        if bool((target_scale <= 0).any()) or bool((metric <= 0).any()):
            raise ValueError("target scale and Fisher weights must be positive")
        rank = int(encoder.shape[1])
        if not 1 <= rank <= _MODAL_WIDTH:
            raise ValueError("latent rank must be between 1 and 64")
        if (
            encoder.shape != (_MODAL_WIDTH, rank)
            or decoder.shape != (rank, _TARGET_WIDTH)
        ):
            raise ValueError("packed encoder/decoder shapes are incompatible")
        artifact = _clone_executor_artifact(self.executor_artifact)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            executor = (
                ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                    artifact
                )
            )
        if (
            executor.input_modes != rank + 2
            or executor.output_modes != rank
            or executor.config.same_position_skip
        ):
            raise ValueError("latent executor geometry is incompatible")
        if not isinstance(self.objective, ContrastAwareObjective):
            raise TypeError("objective must be ContrastAwareObjective")
        if not isinstance(self.initial_metrics, ContrastTrainingMetrics) or not (
            isinstance(self.final_metrics, ContrastTrainingMetrics)
        ):
            raise TypeError("plan metrics have invalid types")
        if type(self.fisher_metric_supplied) is not bool:
            raise TypeError("fisher_metric_supplied must be boolean")
        if not self.fisher_metric_supplied and not torch.equal(
            metric,
            torch.ones_like(metric),
        ):
            raise ValueError(
                "unsupplied Fisher metric state must contain exact unit weights"
            )
        object.__setattr__(
            self,
            "gain_log_center",
            _finite_float(self.gain_log_center, label="gain_log_center"),
        )
        object.__setattr__(
            self,
            "gain_log_scale",
            _positive_float(self.gain_log_scale, label="gain_log_scale"),
        )
        object.__setattr__(
            self,
            "residual_width",
            _positive_int(self.residual_width, label="residual_width"),
        )
        object.__setattr__(
            self,
            "rms_epsilon",
            _positive_float(self.rms_epsilon, label="rms_epsilon"),
        )
        object.__setattr__(
            self,
            "training_steps",
            _positive_int(self.training_steps, label="training_steps"),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_float(self.learning_rate, label="learning_rate"),
        )
        object.__setattr__(
            self,
            "seed",
            _nonnegative_int(self.seed, label="seed"),
        )
        object.__setattr__(
            self,
            "synthetic_binding_sha256",
            _require_sha256(
                self.synthetic_binding_sha256,
                label="synthetic binding",
            ),
        )
        for name in (
            "fit_batch_sha256s",
            "fit_batch_content_sha256s",
            "fit_indexed_batch_sha256s",
            "fit_endpoint_sha256s",
            "fit_pair_sha256s",
        ):
            object.__setattr__(
                self,
                name,
                _sha_tuple(getattr(self, name), label=name),
            )
        object.__setattr__(self, "modal_center", modal_center)
        object.__setattr__(self, "target_center", target_center)
        object.__setattr__(self, "target_scale", target_scale)
        object.__setattr__(self, "encoder_weight", encoder)
        object.__setattr__(self, "decoder_weight", decoder)
        object.__setattr__(self, "fisher_metric_weight", metric)
        object.__setattr__(self, "executor_artifact", artifact)
        object.__setattr__(
            self,
            "artifact_sha256",
            _digest_or_validate(
                self.artifact_sha256,
                self._hash_payload(),
                domain=_PLAN_DOMAIN,
                label="packed provider plan",
            ),
        )

    @property
    def latent_rank(self) -> int:
        return int(self.encoder_weight.shape[1])

    @property
    def rank(self) -> int:
        """Short alias used by rank-ladder orchestration."""

        return self.latent_rank

    @property
    def executor_config(self) -> GatedCausalModalExecutorConfig:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return (
                ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                    self.executor_artifact
                ).config
            )

    @property
    def active_encoder_source_modes(self) -> int:
        return int(
            (torch.linalg.vector_norm(self.encoder_weight, dim=1) > 0)
            .sum()
            .item()
        )

    @property
    def active_decoder_target_modes(self) -> int:
        return int(
            (torch.linalg.vector_norm(self.decoder_weight, dim=0) > 0)
            .sum()
            .item()
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": "fisher_graph.packed_contrast_reference_provider",
            "format_version": _FORMAT_VERSION,
            "modal_center_sha256": _tensor_sha256(self.modal_center),
            "gain_log_center": self.gain_log_center,
            "gain_log_scale": self.gain_log_scale,
            "residual_width": self.residual_width,
            "rms_epsilon": self.rms_epsilon,
            "gain_semantics": (
                "h_null=normalized_null*sqrt(row_rms^2+epsilon);"
                "nonnull_rms=sqrt(max(row_rms^2-h_null^2/residual_width,"
                "epsilon));standardized_log_nonnull_rms"
            ),
            "exact_null_executor_feature": False,
            "target_center_sha256": _tensor_sha256(self.target_center),
            "target_scale_sha256": _tensor_sha256(self.target_scale),
            "encoder_weight_sha256": _tensor_sha256(self.encoder_weight),
            "executor_sha256": _executor_artifact_sha256(
                self.executor_artifact
            ),
            "decoder_weight_sha256": _tensor_sha256(self.decoder_weight),
            "fisher_metric_weight_sha256": _tensor_sha256(
                self.fisher_metric_weight
            ),
            "fisher_metric_supplied": self.fisher_metric_supplied,
            "synthetic_binding_sha256": self.synthetic_binding_sha256,
            "fit_batch_sha256s": self.fit_batch_sha256s,
            "fit_batch_content_sha256s": self.fit_batch_content_sha256s,
            "fit_indexed_batch_sha256s": self.fit_indexed_batch_sha256s,
            "fit_endpoint_sha256s": self.fit_endpoint_sha256s,
            "fit_pair_sha256s": self.fit_pair_sha256s,
            "objective_sha256": self.objective.artifact_sha256,
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "initial_metrics_sha256": self.initial_metrics.artifact_sha256,
            "final_metrics_sha256": self.final_metrics.artifact_sha256,
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._hash_payload(), domain=_PLAN_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("packed provider plan hash mismatch")

    def accounting(self) -> ContrastAwareReferenceProviderAccounting:
        self.validate_integrity()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            executor = (
                ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                    self.executor_artifact
                )
            )
        encoder_count = int(self.encoder_weight.numel())
        decoder_count = int(self.decoder_weight.numel())
        feature_count = int(self.modal_center.numel()) + 2
        target_count = 2 * _TARGET_WIDTH
        total = (
            feature_count
            + target_count
            + encoder_count
            + executor.learned_parameter_count
            + decoder_count
        )
        return ContrastAwareReferenceProviderAccounting(
            modal_modes=_MODAL_WIDTH,
            latent_rank=self.latent_rank,
            target_modes=_TARGET_WIDTH,
            feature_codec_scalar_count=feature_count,
            target_standardization_scalar_count=target_count,
            encoder_parameter_count=encoder_count,
            executor_parameter_count=executor.learned_parameter_count,
            decoder_parameter_count=decoder_count,
            total_stored_scalar_count=total,
            training_metric_scalar_count=_TARGET_WIDTH,
        )

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "modal_center": self.modal_center.clone(),
            "target_center": self.target_center.clone(),
            "target_scale": self.target_scale.clone(),
            "encoder_weight": self.encoder_weight.clone(),
            "executor_artifact": _clone_executor_artifact(
                self.executor_artifact
            ),
            "decoder_weight": self.decoder_weight.clone(),
            "fisher_metric_weight": self.fisher_metric_weight.clone(),
            "objective_state": self.objective.state_dict(),
            "initial_metrics_state": self.initial_metrics.state_dict(),
            "final_metrics_state": self.final_metrics.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        raw: Mapping[str, object],
    ) -> ContrastAwareReferenceProviderPlan:
        payload_fields = {
            "artifact_kind",
            "format_version",
            "modal_center_sha256",
            "gain_log_center",
            "gain_log_scale",
            "residual_width",
            "rms_epsilon",
            "gain_semantics",
            "exact_null_executor_feature",
            "target_center_sha256",
            "target_scale_sha256",
            "encoder_weight_sha256",
            "executor_sha256",
            "decoder_weight_sha256",
            "fisher_metric_weight_sha256",
            "fisher_metric_supplied",
            "synthetic_binding_sha256",
            "fit_batch_sha256s",
            "fit_batch_content_sha256s",
            "fit_indexed_batch_sha256s",
            "fit_endpoint_sha256s",
            "fit_pair_sha256s",
            "objective_sha256",
            "training_steps",
            "learning_rate",
            "seed",
            "initial_metrics_sha256",
            "final_metrics_sha256",
        }
        tensor_fields = {
            "modal_center",
            "target_center",
            "target_scale",
            "encoder_weight",
            "executor_artifact",
            "decoder_weight",
            "fisher_metric_weight",
            "objective_state",
            "initial_metrics_state",
            "final_metrics_state",
            "artifact_sha256",
        }
        _strict_keys(
            raw,
            expected=payload_fields | tensor_fields,
            label="packed provider plan",
        )
        for name in (
            "executor_artifact",
            "objective_state",
            "initial_metrics_state",
            "final_metrics_state",
        ):
            if not isinstance(raw[name], Mapping):
                raise TypeError(f"{name} must be a mapping")
        result = cls(
            modal_center=raw["modal_center"],  # type: ignore[arg-type]
            gain_log_center=raw["gain_log_center"],  # type: ignore[arg-type]
            gain_log_scale=raw["gain_log_scale"],  # type: ignore[arg-type]
            residual_width=raw["residual_width"],  # type: ignore[arg-type]
            rms_epsilon=raw["rms_epsilon"],  # type: ignore[arg-type]
            target_center=raw["target_center"],  # type: ignore[arg-type]
            target_scale=raw["target_scale"],  # type: ignore[arg-type]
            encoder_weight=raw["encoder_weight"],  # type: ignore[arg-type]
            executor_artifact=raw["executor_artifact"],  # type: ignore[arg-type]
            decoder_weight=raw["decoder_weight"],  # type: ignore[arg-type]
            fisher_metric_weight=raw[
                "fisher_metric_weight"
            ],  # type: ignore[arg-type]
            fisher_metric_supplied=raw[
                "fisher_metric_supplied"
            ],  # type: ignore[arg-type]
            synthetic_binding_sha256=str(raw["synthetic_binding_sha256"]),
            fit_batch_sha256s=raw[
                "fit_batch_sha256s"
            ],  # type: ignore[arg-type]
            fit_batch_content_sha256s=raw[
                "fit_batch_content_sha256s"
            ],  # type: ignore[arg-type]
            fit_indexed_batch_sha256s=raw[
                "fit_indexed_batch_sha256s"
            ],  # type: ignore[arg-type]
            fit_endpoint_sha256s=raw[
                "fit_endpoint_sha256s"
            ],  # type: ignore[arg-type]
            fit_pair_sha256s=raw[
                "fit_pair_sha256s"
            ],  # type: ignore[arg-type]
            objective=ContrastAwareObjective.from_state_dict(
                raw["objective_state"]  # type: ignore[arg-type]
            ),
            training_steps=raw["training_steps"],  # type: ignore[arg-type]
            learning_rate=raw["learning_rate"],  # type: ignore[arg-type]
            seed=raw["seed"],  # type: ignore[arg-type]
            initial_metrics=ContrastTrainingMetrics.from_state_dict(
                raw["initial_metrics_state"]  # type: ignore[arg-type]
            ),
            final_metrics=ContrastTrainingMetrics.from_state_dict(
                raw["final_metrics_state"]  # type: ignore[arg-type]
            ),
            artifact_sha256=str(raw["artifact_sha256"]),
        )
        canonical = result._hash_payload()
        for name in payload_fields:
            if raw[name] != canonical[name]:
                raise ValueError(f"packed plan field {name!r} is inconsistent")
        return result

    def prepare(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> PreparedContrastAwareReferenceProvider:
        return PreparedContrastAwareReferenceProvider(
            self,
            dtype=dtype,
            device=device,
        )


def _packed_features(
    *,
    modal_coordinates: Tensor,
    null_coordinates: Tensor,
    row_rms: Tensor,
    valid_mask: Tensor,
    modal_center: Tensor,
    encoder_weight: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
) -> Tensor:
    shape = modal_coordinates.shape[:2]
    if (
        modal_coordinates.ndim != 3
        or modal_coordinates.shape[-1] != _MODAL_WIDTH
        or null_coordinates.shape != (*shape, 1)
        or row_rms.shape != shape
        or valid_mask.shape != shape
        or valid_mask.dtype is not torch.bool
    ):
        raise ValueError("packed feature tensors have incompatible geometry")
    if (
        modal_coordinates.device != modal_center.device
        or null_coordinates.device != modal_center.device
        or row_rms.device != modal_center.device
        or valid_mask.device != modal_center.device
        or modal_coordinates.dtype != modal_center.dtype
        or null_coordinates.dtype != modal_center.dtype
        or row_rms.dtype != modal_center.dtype
    ):
        raise ValueError("packed feature tensors must share dtype and device")
    if not bool(torch.isfinite(modal_coordinates).all()) or not bool(
        torch.isfinite(null_coordinates).all()
    ) or not bool(torch.isfinite(row_rms).all()):
        raise ValueError("packed feature tensors must be finite")
    if bool((row_rms[valid_mask] <= 0).any()):
        raise ValueError("row_rms must be positive on valid rows")
    safe_rms = torch.where(valid_mask, row_rms, torch.ones_like(row_rms))
    h_null = null_coordinates[..., 0] * torch.sqrt(
        safe_rms.square() + rms_epsilon
    )
    nonnull_squared = (
        safe_rms.square() - h_null.square() / float(residual_width)
    )
    nonnull_rms = torch.sqrt(
        torch.clamp(nonnull_squared, min=rms_epsilon)
    )
    gain = (
        torch.log(nonnull_rms) - gain_log_center
    ) / gain_log_scale
    latent = (
        modal_coordinates - modal_center.view(1, 1, -1)
    ) @ encoder_weight
    features = torch.cat(
        (
            torch.ones(
                (*shape, 1),
                dtype=modal_coordinates.dtype,
                device=modal_coordinates.device,
            ),
            latent,
            gain.unsqueeze(-1),
        ),
        dim=-1,
    )
    return torch.where(
        valid_mask.unsqueeze(-1),
        features,
        torch.zeros_like(features),
    )


class PreparedContrastAwareReferenceProvider(nn.Module):
    """Prepared packed runtime returning raw 64-mode target coordinates."""

    def __init__(
        self,
        plan: ContrastAwareReferenceProviderPlan,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("prepared dtype must be floating point")
        plan.validate_integrity()
        target_device = torch.device("cpu" if device is None else device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            executor = (
                ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                    plan.executor_artifact
                )
            )
        executor.to(device=target_device, dtype=dtype)
        executor.eval()
        executor.requires_grad_(False)
        self.executor = executor
        for name, tensor in (
            ("modal_center", plan.modal_center),
            ("encoder_weight", plan.encoder_weight),
            ("decoder_weight", plan.decoder_weight),
            ("target_center", plan.target_center),
            ("target_scale", plan.target_scale),
        ):
            self.register_buffer(
                name,
                tensor.to(device=target_device, dtype=dtype),
                persistent=False,
            )
        self.gain_log_center = plan.gain_log_center
        self.gain_log_scale = plan.gain_log_scale
        self.residual_width = plan.residual_width
        self.rms_epsilon = plan.rms_epsilon
        self.plan_sha256 = plan.artifact_sha256
        self._expected_runtime_sha256 = self._runtime_sha256()

    @property
    def dtype(self) -> torch.dtype:
        return self.modal_center.dtype

    @property
    def device(self) -> torch.device:
        return self.modal_center.device

    @property
    def latent_rank(self) -> int:
        return int(self.encoder_weight.shape[1])

    @property
    def rank(self) -> int:
        return self.latent_rank

    @property
    def active_encoder_source_modes(self) -> int:
        return int(
            (torch.linalg.vector_norm(self.encoder_weight, dim=1) > 0)
            .sum()
            .item()
        )

    @property
    def active_decoder_target_modes(self) -> int:
        return int(
            (torch.linalg.vector_norm(self.decoder_weight, dim=0) > 0)
            .sum()
            .item()
        )

    def _runtime_sha256(self) -> str:
        return _json_sha256(
            {
                "plan_sha256": self.plan_sha256,
                "dtype": str(self.dtype),
                "device": self.device.type,
                "modal_center": _tensor_sha256(self.modal_center),
                "encoder": _tensor_sha256(self.encoder_weight),
                "executor": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(
                        self.executor.state_dict().items()
                    )
                },
                "decoder": _tensor_sha256(self.decoder_weight),
                "target_center": _tensor_sha256(self.target_center),
                "target_scale": _tensor_sha256(self.target_scale),
                "gain_log_center": self.gain_log_center,
                "gain_log_scale": self.gain_log_scale,
                "residual_width": self.residual_width,
                "rms_epsilon": self.rms_epsilon,
            },
            domain=_RUNTIME_DOMAIN,
        )

    def validate_integrity(self) -> None:
        if self._runtime_sha256() != self._expected_runtime_sha256:
            raise ValueError("packed provider runtime integrity mismatch")

    def encode_features(
        self,
        modal_coordinates: Tensor,
        null_coordinates: Tensor,
        row_rms: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        return _packed_features(
            modal_coordinates=modal_coordinates,
            null_coordinates=null_coordinates,
            row_rms=row_rms,
            valid_mask=valid_mask,
            modal_center=self.modal_center,
            encoder_weight=self.encoder_weight,
            gain_log_center=self.gain_log_center,
            gain_log_scale=self.gain_log_scale,
            residual_width=self.residual_width,
            rms_epsilon=self.rms_epsilon,
        )

    def forward_standardized(
        self,
        modal_coordinates: Tensor,
        null_coordinates: Tensor,
        row_rms: Tensor,
        *,
        valid_mask: Tensor,
        logical_positions: Tensor,
    ) -> Tensor:
        features = self.encode_features(
            modal_coordinates,
            null_coordinates,
            row_rms,
            valid_mask,
        )
        latent_output = self.executor(
            features,
            query_valid_mask=valid_mask,
            key_valid_mask=valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=logical_positions,
        )
        standardized = latent_output @ self.decoder_weight
        return torch.where(
            valid_mask.unsqueeze(-1),
            standardized,
            torch.zeros_like(standardized),
        )

    def forward(
        self,
        modal_coordinates: Tensor,
        null_coordinates: Tensor,
        row_rms: Tensor,
        *,
        valid_mask: Tensor,
        logical_positions: Tensor,
    ) -> Tensor:
        standardized = self.forward_standardized(
            modal_coordinates,
            null_coordinates,
            row_rms,
            valid_mask=valid_mask,
            logical_positions=logical_positions,
        )
        raw = standardized * self.target_scale + self.target_center
        return torch.where(
            valid_mask.unsqueeze(-1),
            raw,
            torch.zeros_like(raw),
        )

    def execution_accounting(
        self,
        *,
        valid_mask: Tensor,
        logical_positions: Tensor,
    ) -> ContrastAwareExecutionAccounting:
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.ndim != 2
            or valid_mask.dtype is not torch.bool
            or valid_mask.device != self.device
            or not isinstance(logical_positions, Tensor)
            or logical_positions.shape != valid_mask.shape
            or logical_positions.device != self.device
        ):
            raise ValueError("accounting masks and positions are invalid")
        core = self.executor.execution_accounting(
            int(valid_mask.shape[1]),
            batch_size=int(valid_mask.shape[0]),
            query_valid_mask=valid_mask,
            key_valid_mask=valid_mask,
            logical_positions=logical_positions,
            key_logical_positions=logical_positions,
        )
        rows = int(valid_mask.sum().item())
        encoder_macs = rows * _MODAL_WIDTH * self.latent_rank
        decoder_macs = rows * self.latent_rank * _TARGET_WIDTH
        target_macs = rows * _TARGET_WIDTH
        return ContrastAwareExecutionAccounting(
            core=core,
            valid_rows=rows,
            encoder_mac_count=encoder_macs,
            decoder_mac_count=decoder_macs,
            target_destandardization_mac_count=target_macs,
            total_mac_count=(
                core.total_mac_count
                + encoder_macs
                + decoder_macs
                + target_macs
            ),
        )


class _PackedTrainingModule(nn.Module):
    def __init__(
        self,
        *,
        modal_center: Tensor,
        gain_log_center: float,
        gain_log_scale: float,
        residual_width: int,
        rms_epsilon: float,
        target_center: Tensor,
        target_scale: Tensor,
        executor_config: GatedCausalModalExecutorConfig,
        seed: int,
    ) -> None:
        super().__init__()
        rank = executor_config.output_modes
        encoder = torch.zeros(_MODAL_WIDTH, rank, dtype=torch.float64)
        decoder = torch.zeros(rank, _TARGET_WIDTH, dtype=torch.float64)
        diagonal = min(rank, _MODAL_WIDTH)
        encoder[torch.arange(diagonal), torch.arange(diagonal)] = 1.0
        decoder[torch.arange(diagonal), torch.arange(diagonal)] = 1.0
        self.encoder_weight = nn.Parameter(encoder)
        self.decoder_weight = nn.Parameter(decoder)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.executor = ResidualGatedCausalModalExecutor(
                executor_config,
                dtype=torch.float64,
                device="cpu",
            )
        self.register_buffer("modal_center", modal_center, persistent=False)
        self.register_buffer("target_center", target_center, persistent=False)
        self.register_buffer("target_scale", target_scale, persistent=False)
        self.gain_log_center = gain_log_center
        self.gain_log_scale = gain_log_scale
        self.residual_width = residual_width
        self.rms_epsilon = rms_epsilon

    def encode_features(
        self,
        modal: Tensor,
        null: Tensor,
        rms: Tensor,
        mask: Tensor,
    ) -> Tensor:
        return _packed_features(
            modal_coordinates=modal,
            null_coordinates=null,
            row_rms=rms,
            valid_mask=mask,
            modal_center=self.modal_center,
            encoder_weight=self.encoder_weight,
            gain_log_center=self.gain_log_center,
            gain_log_scale=self.gain_log_scale,
            residual_width=self.residual_width,
            rms_epsilon=self.rms_epsilon,
        )

    def forward_standardized(
        self,
        modal: Tensor,
        null: Tensor,
        rms: Tensor,
        mask: Tensor,
        positions: Tensor,
    ) -> Tensor:
        features = self.encode_features(modal, null, rms, mask)
        latent = self.executor(
            features,
            query_valid_mask=mask,
            key_valid_mask=mask,
            logical_positions=positions,
            key_logical_positions=positions,
        )
        result = latent @ self.decoder_weight
        return torch.where(
            mask.unsqueeze(-1),
            result,
            torch.zeros_like(result),
        )


@dataclass(frozen=True, slots=True)
class _EndpointLocation:
    batch_index: int
    row_index: int
    endpoint_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedFitData:
    batches: tuple[IndexedReferenceBatch, ...]
    pairs: tuple[ReferenceProviderContrastPair, ...]
    endpoints: Mapping[str, _EndpointLocation]


def _prepare_fit_data(
    *,
    fit_batches: Sequence[IndexedReferenceBatch],
    contrast_pairs: Sequence[ReferenceProviderContrastPair],
    require_fit_split: bool,
) -> _PreparedFitData:
    if isinstance(fit_batches, (str, bytes)) or not isinstance(
        fit_batches, Sequence
    ):
        raise TypeError("fit_batches must be a sequence")
    batches = tuple(fit_batches)
    if not batches or any(
        not isinstance(value, IndexedReferenceBatch) for value in batches
    ):
        raise TypeError(
            "fit_batches must contain IndexedReferenceBatch values"
        )
    batches = tuple(sorted(batches, key=lambda value: value.artifact_sha256))
    bindings = {value.batch.synthetic_binding_sha256 for value in batches}
    if len(bindings) != 1:
        raise ValueError("all fit batches must share one synthetic binding")
    if require_fit_split and any(
        value.batch.split != "fit" for value in batches
    ):
        raise ValueError("contrast fitting accepts only split='fit'")
    endpoints: dict[str, _EndpointLocation] = {}
    for batch_index, indexed in enumerate(batches):
        batch = indexed.batch
        if (
            batch.modal_modes != _MODAL_WIDTH
            or batch.target_mode_count != _TARGET_WIDTH
            or batch.null_modes != 1
        ):
            raise ValueError(
                "packed fitting requires 64 source modes, 64 target modes, "
                "and one exact-null coordinate"
            )
        for row, endpoint_id in enumerate(indexed.endpoint_ids):
            if endpoint_id in endpoints:
                raise ValueError("endpoint ids must be globally unique")
            endpoint_sha = _json_sha256(
                {
                    "endpoint_id": endpoint_id,
                    "indexed_batch_sha256": indexed.artifact_sha256,
                    "batch_content_sha256": batch.content_sha256,
                    "row_index": row,
                },
                domain=_ENDPOINT_DOMAIN,
            )
            endpoints[endpoint_id] = _EndpointLocation(
                batch_index=batch_index,
                row_index=row,
                endpoint_sha256=endpoint_sha,
            )
    if isinstance(contrast_pairs, (str, bytes)) or not isinstance(
        contrast_pairs, Sequence
    ):
        raise TypeError("contrast_pairs must be a sequence")
    pairs = tuple(contrast_pairs)
    if not pairs or any(
        not isinstance(value, ReferenceProviderContrastPair)
        for value in pairs
    ):
        raise TypeError(
            "contrast_pairs must contain ReferenceProviderContrastPair values"
        )
    pairs = tuple(sorted(pairs, key=lambda value: value.pair_id))
    if len({value.pair_id for value in pairs}) != len(pairs):
        raise ValueError("contrast pair ids must be unique")
    for pair in pairs:
        if (
            pair.left_endpoint_id not in endpoints
            or pair.right_endpoint_id not in endpoints
        ):
            raise ValueError("contrast pair references an unknown endpoint")
        left_location = endpoints[pair.left_endpoint_id]
        right_location = endpoints[pair.right_endpoint_id]
        left_batch = batches[left_location.batch_index].batch
        right_batch = batches[right_location.batch_index].batch
        left_row = left_location.row_index
        right_row = right_location.row_index
        if (
            left_batch.sequence_length != right_batch.sequence_length
            or not torch.equal(
                left_batch.valid_mask[left_row],
                right_batch.valid_mask[right_row],
            )
            or not torch.equal(
                left_batch.logical_positions[left_row],
                right_batch.logical_positions[right_row],
            )
        ):
            raise ValueError(
                "contrast pair shapes, masks, and positions must align"
            )
        if pair.teacher_midpoint_jvp is not None and (
            pair.teacher_midpoint_jvp.shape
            != (left_batch.sequence_length, _TARGET_WIDTH)
        ):
            raise ValueError("teacher midpoint JVP shape is incompatible")
    return _PreparedFitData(
        batches=batches,
        pairs=pairs,
        endpoints=endpoints,
    )


@dataclass(frozen=True, slots=True)
class _TensorMetrics:
    pointwise_mse: Tensor
    sensitivity_relative_delta_mse: Tensor
    sensitivity_direction_loss: Tensor
    midpoint_jvp_relative_mse: Tensor
    intended_null_absolute_mse: Tensor
    weighted_total: Tensor


def _mean_or_zero(values: list[Tensor], *, like: Tensor) -> Tensor:
    if not values:
        return like.new_zeros(())
    return torch.stack(values).mean()


def _pair_row(
    data: _PreparedFitData,
    endpoint_id: str,
) -> tuple[IndexedReferenceBatch, int]:
    location = data.endpoints[endpoint_id]
    return data.batches[location.batch_index], location.row_index


def _midpoint_jvp_losses(
    model: _PackedTrainingModule,
    *,
    data: _PreparedFitData,
    target_scale: Tensor,
    metric_weight: Tensor,
    objective: ContrastAwareObjective,
    batch_same_length: bool,
) -> list[Tensor]:
    """Evaluate supplied chart JVPs, optionally batching equal-length pairs.

    The provider-chart primal and tangent are measured at, and pushed forward
    from, the teacher's hidden-space midpoint.  Endpoint arithmetic is
    intentionally absent from this path.
    """

    work: list[
        tuple[
            int,
            ReferenceProviderContrastPair,
            IndexedReferenceBatch,
            int,
        ]
    ] = []
    for pair_index, pair in enumerate(data.pairs):
        if pair.teacher_midpoint_jvp is None:
            continue
        left_indexed, left_row = _pair_row(
            data,
            pair.left_endpoint_id,
        )
        work.append((pair_index, pair, left_indexed, left_row))

    grouped: dict[
        tuple[int, int],
        list[
            tuple[
                int,
                ReferenceProviderContrastPair,
                IndexedReferenceBatch,
                int,
            ]
        ],
    ] = {}
    for item in work:
        pair_index, pair, left_indexed, _ = item
        sequence_length = left_indexed.batch.sequence_length
        key = (
            sequence_length,
            0 if batch_same_length else pair_index,
        )
        grouped.setdefault(key, []).append(item)

    by_pair_index: dict[int, Tensor] = {}
    for group in grouped.values():
        modal_primals: list[Tensor] = []
        null_primals: list[Tensor] = []
        row_rms_primals: list[Tensor] = []
        modal_tangents: list[Tensor] = []
        null_tangents: list[Tensor] = []
        row_rms_tangents: list[Tensor] = []
        teacher_jvps: list[Tensor] = []
        masks: list[Tensor] = []
        positions: list[Tensor] = []
        for _, pair, left_indexed, left_row in group:
            chart_tensors = tuple(
                getattr(pair, name) for name in _PROVIDER_CHART_FIELDS
            )
            if any(value is None for value in chart_tensors):
                raise RuntimeError(
                    "validated teacher JVP lost its provider-chart contract"
                )
            (
                modal_primal,
                null_primal,
                row_rms_primal,
                modal_tangent,
                null_tangent,
                row_rms_tangent,
            ) = chart_tensors
            teacher_jvp = pair.teacher_midpoint_jvp
            assert isinstance(modal_primal, Tensor)
            assert isinstance(null_primal, Tensor)
            assert isinstance(row_rms_primal, Tensor)
            assert isinstance(modal_tangent, Tensor)
            assert isinstance(null_tangent, Tensor)
            assert isinstance(row_rms_tangent, Tensor)
            assert isinstance(teacher_jvp, Tensor)
            modal_primals.append(modal_primal)
            null_primals.append(null_primal)
            row_rms_primals.append(row_rms_primal)
            modal_tangents.append(modal_tangent)
            null_tangents.append(null_tangent)
            row_rms_tangents.append(row_rms_tangent)
            teacher_jvps.append(teacher_jvp)
            masks.append(left_indexed.batch.valid_mask[left_row])
            positions.append(
                left_indexed.batch.logical_positions[left_row]
            )

        modal_primal_batch = torch.stack(modal_primals, dim=0)
        null_primal_batch = torch.stack(null_primals, dim=0)
        row_rms_primal_batch = torch.stack(row_rms_primals, dim=0)
        modal_tangent_batch = torch.stack(modal_tangents, dim=0)
        null_tangent_batch = torch.stack(null_tangents, dim=0)
        row_rms_tangent_batch = torch.stack(row_rms_tangents, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        position_batch = torch.stack(positions, dim=0)

        def candidate(
            modal: Tensor,
            null: Tensor,
            rms: Tensor,
        ) -> Tensor:
            return model.forward_standardized(
                modal,
                null,
                rms,
                mask_batch,
                position_batch,
            )

        _, candidate_jvp = torch.func.jvp(
            candidate,
            (
                modal_primal_batch,
                null_primal_batch,
                row_rms_primal_batch,
            ),
            (
                modal_tangent_batch,
                null_tangent_batch,
                row_rms_tangent_batch,
            ),
        )
        teacher_jvp = torch.stack(teacher_jvps, dim=0) / target_scale
        candidate_weighted = candidate_jvp * metric_weight
        teacher_weighted = teacher_jvp * metric_weight
        for row, (pair_index, _, _, _) in enumerate(group):
            expanded_mask = mask_batch[row].unsqueeze(-1).expand(
                -1,
                _TARGET_WIDTH,
            )
            candidate_values = candidate_weighted[row][expanded_mask]
            teacher_values = teacher_weighted[row][expanded_mask]
            error = (candidate_values - teacher_values).square().mean()
            teacher_mse = teacher_values.square().mean()
            by_pair_index[pair_index] = error / torch.clamp(
                teacher_mse,
                min=objective.jvp_relative_floor**2,
            )

    return [
        by_pair_index[pair_index]
        for pair_index, _, _, _ in work
    ]


def _loss_components(
    model: _PackedTrainingModule,
    *,
    data: _PreparedFitData,
    target_center: Tensor,
    target_scale: Tensor,
    metric_weight: Tensor,
    objective: ContrastAwareObjective,
) -> _TensorMetrics:
    predictions: dict[str, Tensor] = {}
    standardized_targets: dict[str, Tensor] = {}
    pointwise_squared = model.encoder_weight.new_zeros(())
    pointwise_count = 0
    for indexed in data.batches:
        batch = indexed.batch
        prediction = model.forward_standardized(
            batch.modal_coordinates,
            batch.null_coordinates,
            batch.row_rms,
            batch.valid_mask,
            batch.logical_positions,
        )
        target = (batch.target_modes - target_center) / target_scale
        target = torch.where(
            batch.valid_mask.unsqueeze(-1),
            target,
            torch.zeros_like(target),
        )
        weighted_error = (prediction - target) * metric_weight
        pointwise_squared = pointwise_squared + (
            weighted_error.square() * batch.valid_mask.unsqueeze(-1)
        ).sum()
        pointwise_count += batch.valid_row_count * _TARGET_WIDTH
        for row, endpoint_id in enumerate(indexed.endpoint_ids):
            predictions[endpoint_id] = prediction[row : row + 1]
            standardized_targets[endpoint_id] = target[row : row + 1]
    if pointwise_count <= 0:
        raise ValueError("fit data contains no valid target scalars")
    pointwise = pointwise_squared / pointwise_count

    sensitivity_delta: list[Tensor] = []
    sensitivity_direction: list[Tensor] = []
    null_losses: list[Tensor] = []
    for pair in data.pairs:
        left_indexed, left_row = _pair_row(data, pair.left_endpoint_id)
        right_indexed, right_row = _pair_row(data, pair.right_endpoint_id)
        left_batch = left_indexed.batch
        right_batch = right_indexed.batch
        mask = left_batch.valid_mask[left_row : left_row + 1]
        expanded_mask = mask.unsqueeze(-1).expand(
            -1, -1, _TARGET_WIDTH
        )
        predicted_delta = (
            predictions[pair.right_endpoint_id]
            - predictions[pair.left_endpoint_id]
        ) * metric_weight
        teacher_delta = (
            standardized_targets[pair.right_endpoint_id]
            - standardized_targets[pair.left_endpoint_id]
        ) * metric_weight
        predicted_values = predicted_delta[expanded_mask]
        teacher_values = teacher_delta[expanded_mask]
        if pair.role == "expected_sensitivity":
            delta_error_mse = (predicted_values - teacher_values).square().mean()
            teacher_mse = teacher_values.square().mean()
            sensitivity_delta.append(
                delta_error_mse
                / torch.clamp(
                    teacher_mse,
                    min=objective.sensitivity_relative_floor**2,
                )
            )
            dot = torch.dot(predicted_values, teacher_values)
            predicted_norm = torch.linalg.vector_norm(predicted_values)
            teacher_norm = torch.linalg.vector_norm(teacher_values)
            denominator = torch.clamp(
                predicted_norm,
                min=objective.direction_norm_floor,
            ) * torch.clamp(
                teacher_norm,
                min=objective.direction_norm_floor,
            )
            cosine = torch.clamp(dot / denominator, min=-1.0, max=1.0)
            sensitivity_direction.append(1.0 - cosine)
        else:
            # Deliberately absolute: no tiny teacher-null denominator appears.
            null_losses.append(predicted_values.square().mean())

    jvp_losses = _midpoint_jvp_losses(
        model,
        data=data,
        target_scale=target_scale,
        metric_weight=metric_weight,
        objective=objective,
        batch_same_length=True,
    )
    zero = model.encoder_weight.new_zeros(())
    sensitivity_delta_mean = _mean_or_zero(sensitivity_delta, like=zero)
    sensitivity_direction_mean = _mean_or_zero(
        sensitivity_direction,
        like=zero,
    )
    jvp_mean = _mean_or_zero(jvp_losses, like=zero)
    null_mean = _mean_or_zero(null_losses, like=zero)
    total = (
        objective.pointwise_weight * pointwise
        + objective.sensitivity_relative_delta_weight
        * sensitivity_delta_mean
        + objective.sensitivity_direction_weight
        * sensitivity_direction_mean
        + objective.midpoint_jvp_weight * jvp_mean
        + objective.intended_null_weight * null_mean
    )
    return _TensorMetrics(
        pointwise_mse=pointwise,
        sensitivity_relative_delta_mse=sensitivity_delta_mean,
        sensitivity_direction_loss=sensitivity_direction_mean,
        midpoint_jvp_relative_mse=jvp_mean,
        intended_null_absolute_mse=null_mean,
        weighted_total=total,
    )


def _materialize_metrics(
    values: _TensorMetrics,
    *,
    data: _PreparedFitData,
) -> ContrastTrainingMetrics:
    return ContrastTrainingMetrics(
        pointwise_mse=float(values.pointwise_mse.detach()),
        sensitivity_relative_delta_mse=float(
            values.sensitivity_relative_delta_mse.detach()
        ),
        sensitivity_direction_loss=float(
            values.sensitivity_direction_loss.detach()
        ),
        midpoint_jvp_relative_mse=float(
            values.midpoint_jvp_relative_mse.detach()
        ),
        intended_null_absolute_mse=float(
            values.intended_null_absolute_mse.detach()
        ),
        weighted_total=float(values.weighted_total.detach()),
        endpoint_count=len(data.endpoints),
        sensitivity_pair_count=sum(
            value.role == "expected_sensitivity" for value in data.pairs
        ),
        jvp_pair_count=sum(
            value.teacher_midpoint_jvp is not None for value in data.pairs
        ),
        intended_null_pair_count=sum(
            value.role == "intended_null" for value in data.pairs
        ),
    )


def _validate_fit_geometry(
    *,
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
    target_center: Tensor,
    target_scale: Tensor,
    executor_config: GatedCausalModalExecutorConfig,
    fisher_metric_weight: Tensor | None,
) -> tuple[Tensor, float, float, int, float, Tensor, Tensor, Tensor, bool]:
    modal = _canonical_float_tensor(
        modal_center,
        label="modal_center",
        ndim=1,
    )
    center = _canonical_float_tensor(
        target_center,
        label="target_center",
        ndim=1,
    )
    scale = _canonical_float_tensor(
        target_scale,
        label="target_scale",
        ndim=1,
    )
    if (
        modal.shape != (_MODAL_WIDTH,)
        or center.shape != (_TARGET_WIDTH,)
        or scale.shape != (_TARGET_WIDTH,)
    ):
        raise ValueError("packed provider gauges must have width 64")
    if bool((scale <= 0).any()):
        raise ValueError("target_scale must be strictly positive")
    if not isinstance(executor_config, GatedCausalModalExecutorConfig):
        raise TypeError("executor_config must be a gated executor config")
    rank = executor_config.output_modes
    if not 1 <= rank <= _MODAL_WIDTH:
        raise ValueError("latent rank must be between 1 and 64")
    if (
        executor_config.input_modes != rank + 2
        or executor_config.same_position_skip
    ):
        raise ValueError(
            "latent executor must map [constant, latent, gain] to latent "
            "without same-position skip"
        )
    supplied = fisher_metric_weight is not None
    metric = (
        torch.ones(_TARGET_WIDTH, dtype=torch.float64)
        if fisher_metric_weight is None
        else _canonical_float_tensor(
            fisher_metric_weight,
            label="fisher_metric_weight",
            ndim=1,
        )
    )
    if metric.shape != (_TARGET_WIDTH,) or bool((metric <= 0).any()):
        raise ValueError("Fisher metric weights must be positive width-64")
    return (
        modal,
        _finite_float(gain_log_center, label="gain_log_center"),
        _positive_float(gain_log_scale, label="gain_log_scale"),
        _positive_int(residual_width, label="residual_width"),
        _positive_float(rms_epsilon, label="rms_epsilon"),
        center,
        scale,
        metric,
        supplied,
    )


def fit_contrast_aware_reference_provider(
    *,
    modal_center: Tensor,
    gain_log_center: float,
    gain_log_scale: float,
    residual_width: int,
    rms_epsilon: float,
    target_center: Tensor,
    target_scale: Tensor,
    fit_batches: Sequence[IndexedReferenceBatch],
    contrast_pairs: Sequence[ReferenceProviderContrastPair],
    executor_config: GatedCausalModalExecutorConfig,
    objective: ContrastAwareObjective,
    fisher_metric_weight: Tensor | None = None,
    steps: int,
    learning_rate: float,
    seed: int,
) -> ContrastAwareReferenceProviderPlan:
    """Fit one deterministic full-width packed provider on fit-only data."""

    if not isinstance(objective, ContrastAwareObjective):
        raise TypeError("objective must be ContrastAwareObjective")
    (
        modal,
        gain_center,
        gain_scale,
        width,
        epsilon,
        target_center64,
        target_scale64,
        metric,
        metric_supplied,
    ) = _validate_fit_geometry(
        modal_center=modal_center,
        gain_log_center=gain_log_center,
        gain_log_scale=gain_log_scale,
        residual_width=residual_width,
        rms_epsilon=rms_epsilon,
        target_center=target_center,
        target_scale=target_scale,
        executor_config=executor_config,
        fisher_metric_weight=fisher_metric_weight,
    )
    training_steps = _positive_int(steps, label="steps")
    rate = _positive_float(learning_rate, label="learning_rate")
    fit_seed = _nonnegative_int(seed, label="seed")
    data = _prepare_fit_data(
        fit_batches=fit_batches,
        contrast_pairs=contrast_pairs,
        require_fit_split=True,
    )
    model = _PackedTrainingModule(
        modal_center=modal,
        gain_log_center=gain_center,
        gain_log_scale=gain_scale,
        residual_width=width,
        rms_epsilon=epsilon,
        target_center=target_center64,
        target_scale=target_scale64,
        executor_config=executor_config,
        seed=fit_seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    model.train()
    initial = _materialize_metrics(
        _loss_components(
            model,
            data=data,
            target_center=target_center64,
            target_scale=target_scale64,
            metric_weight=metric,
            objective=objective,
        ),
        data=data,
    )
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        components = _loss_components(
            model,
            data=data,
            target_center=target_center64,
            target_scale=target_scale64,
            metric_weight=metric,
            objective=objective,
        )
        if not bool(torch.isfinite(components.weighted_total)):
            raise ValueError("contrast-aware fit produced a nonfinite loss")
        components.weighted_total.backward()
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise ValueError("contrast-aware fit produced a nonfinite gradient")
        optimizer.step()
    model.eval()
    final = _materialize_metrics(
        _loss_components(
            model,
            data=data,
            target_center=target_center64,
            target_scale=target_scale64,
            metric_weight=metric,
            objective=objective,
        ),
        data=data,
    )
    bindings = {
        value.batch.synthetic_binding_sha256 for value in data.batches
    }
    assert len(bindings) == 1
    return ContrastAwareReferenceProviderPlan(
        modal_center=modal,
        gain_log_center=gain_center,
        gain_log_scale=gain_scale,
        residual_width=width,
        rms_epsilon=epsilon,
        target_center=target_center64,
        target_scale=target_scale64,
        encoder_weight=model.encoder_weight.detach(),
        executor_artifact=model.executor.artifact_state_dict(),
        decoder_weight=model.decoder_weight.detach(),
        fisher_metric_weight=metric,
        fisher_metric_supplied=metric_supplied,
        synthetic_binding_sha256=next(iter(bindings)),
        fit_batch_sha256s=tuple(
            value.batch.artifact_sha256 for value in data.batches
        ),
        fit_batch_content_sha256s=tuple(
            value.batch.content_sha256 for value in data.batches
        ),
        fit_indexed_batch_sha256s=tuple(
            value.artifact_sha256 for value in data.batches
        ),
        fit_endpoint_sha256s=tuple(
            location.endpoint_sha256
            for _, location in sorted(data.endpoints.items())
        ),
        fit_pair_sha256s=tuple(
            value.artifact_sha256 for value in data.pairs
        ),
        objective=objective,
        training_steps=training_steps,
        learning_rate=rate,
        seed=fit_seed,
        initial_metrics=initial,
        final_metrics=final,
    )


def evaluate_contrast_aware_reference_provider(
    plan: ContrastAwareReferenceProviderPlan,
    *,
    batches: Sequence[IndexedReferenceBatch],
    contrast_pairs: Sequence[ReferenceProviderContrastPair],
) -> ContrastTrainingMetrics:
    """Evaluate the frozen objective on aligned endpoint and pair tensors."""

    if not isinstance(plan, ContrastAwareReferenceProviderPlan):
        raise TypeError("plan must be ContrastAwareReferenceProviderPlan")
    plan.validate_integrity()
    data = _prepare_fit_data(
        fit_batches=batches,
        contrast_pairs=contrast_pairs,
        require_fit_split=False,
    )
    bindings = {
        value.batch.synthetic_binding_sha256 for value in data.batches
    }
    if bindings != {plan.synthetic_binding_sha256}:
        raise ValueError("evaluation binding differs from the provider plan")
    config = plan.executor_config
    model = _PackedTrainingModule(
        modal_center=plan.modal_center,
        gain_log_center=plan.gain_log_center,
        gain_log_scale=plan.gain_log_scale,
        residual_width=plan.residual_width,
        rms_epsilon=plan.rms_epsilon,
        target_center=plan.target_center,
        target_scale=plan.target_scale,
        executor_config=config,
        seed=plan.seed,
    )
    model.encoder_weight.data.copy_(plan.encoder_weight)
    model.decoder_weight.data.copy_(plan.decoder_weight)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        restored = (
            ResidualGatedCausalModalExecutor.from_artifact_state_dict(
                plan.executor_artifact
            )
        )
    model.executor.load_state_dict(restored.state_dict(), strict=True)
    model.eval()
    values = _loss_components(
        model,
        data=data,
        target_center=plan.target_center,
        target_scale=plan.target_scale,
        metric_weight=plan.fisher_metric_weight,
        objective=plan.objective,
    )
    return _materialize_metrics(values, data=data)
