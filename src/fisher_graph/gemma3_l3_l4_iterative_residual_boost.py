"""Minimal behavior-gradient boost for the Gemma L3/L4 compiled parent.

This module implements one deliberately small compiler iteration.  The
accepted X4 plus lag-B parent remains frozen.  Four learned, causal scalars
modulate the lag-B correction at fixed logical-position buckets:

``[0, 3]``, ``[4, 7]``, ``[8, 15]``, and ``[16, +inf)``.

The scalars are fit from per-prompt downstream-NLL linearizations,

``d_i(theta) ~= d_i(0) + J_i theta``,

with equal family mass and leave-one-family-out fitting owned by the live
campaign.  No native activations, logits, gradients, token IDs, or prompts
are retained by the records below.
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

from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
)
from .gemma3_l3_l4_h4_damping_selection_runtime import (
    GemmaH4DampingFiniteNLLObservation,
)


__all__ = [
    "CAUSAL_POSITION_BIN_COUNT",
    "CAUSAL_POSITION_BIN_EDGES",
    "GemmaCausalPositionScaleH4Provider",
    "GemmaIterativeResidualFitRecord",
    "GemmaIterativeResidualFoldFit",
    "build_gemma_iterative_residual_fit_record",
    "causal_position_bin_indices",
    "fit_gemma_iterative_residual_fold",
    "fit_gemma_iterative_residual_fold_provider",
    "fit_gemma_iterative_residual_full_provider",
    "gemma_causal_position_scale_provider_artifact_sha256",
]


CAUSAL_POSITION_BIN_EDGES: tuple[int, int, int] = (4, 8, 16)
CAUSAL_POSITION_BIN_COUNT = 4
RIDGE = 1.0e-6
TRUST_BOUND = 0.5
_COLUMN_SUPPORT_EPSILON = 1.0e-12
_H4_SITE = "layer.4.output"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIT_RECORD_DOMAIN = b"fisher-graph:gemma-iterative-fit-record:v1\0"
_FOLD_FIT_DOMAIN = b"fisher-graph:gemma-iterative-fold-fit:v1\0"
_PROVIDER_DOMAIN = b"fisher-graph:gemma-position-scale-provider:v1\0"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty stripped string")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float4(value: object, *, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four scalars")
    result = tuple(
        _finite(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _int4(value: object, *, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(
            f"{label} must contain exactly four nonnegative integers"
        )
    return tuple(value)  # type: ignore[return-value]


def causal_position_bin_indices(logical_positions: Tensor) -> Tensor:
    """Return fixed causal bucket IDs using only each row's own position."""

    if (
        not isinstance(logical_positions, Tensor)
        or logical_positions.dtype not in (torch.int32, torch.int64)
    ):
        raise TypeError("logical_positions must be an integer Tensor")
    result = torch.zeros_like(logical_positions, dtype=torch.int64)
    result = result + (logical_positions >= 4).to(torch.int64)
    result = result + (logical_positions >= 8).to(torch.int64)
    result = result + (logical_positions >= 16).to(torch.int64)
    return result


def gemma_causal_position_scale_provider_artifact_sha256(
    *,
    parent_artifact_sha256: str,
    parent_h4_artifact_sha256: str,
    bridge_binding_sha256: str,
    fold_receipt_sha256: str,
    coefficients_by_bin: Sequence[float],
) -> str:
    """Replay a provider identity without loading its frozen parent tensors."""

    payload = {
        "semantics": (
            "lag_b_times_one_plus_causal_logical_position_theta_v1"
        ),
        "site": _H4_SITE,
        "parent_artifact_sha256": _require_sha256(
            parent_artifact_sha256,
            label="parent artifact",
        ),
        "parent_h4_artifact_sha256": _require_sha256(
            parent_h4_artifact_sha256,
            label="parent H4 artifact",
        ),
        "bridge_binding_sha256": _require_sha256(
            bridge_binding_sha256,
            label="bridge binding",
        ),
        "causal_position_bin_edges": CAUSAL_POSITION_BIN_EDGES,
        "fold_receipt_sha256": _require_sha256(
            fold_receipt_sha256,
            label="fold receipt",
        ),
        "coefficients_by_bin": _float4(
            coefficients_by_bin,
            label="coefficients_by_bin",
        ),
    }
    return _sha256(_PROVIDER_DOMAIN, payload)


@dataclass(frozen=True, slots=True)
class GemmaIterativeResidualFitRecord:
    """One prompt's tensor-free parent-point behavioral linearization."""

    example_id: str
    family_id: str
    model_inputs_sha256: str
    parent_execution_sha256: str
    parent_observation_sha256: str
    supervised_tokens: int
    parent_signed_delta_nll_per_token: float
    jacobian_by_bin: tuple[float, float, float, float]
    active_rows_by_bin: tuple[int, int, int, int]
    fit_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="fit record example_id")
        _identifier(self.family_id, label="fit record family_id")
        for name in (
            "model_inputs_sha256",
            "parent_execution_sha256",
            "parent_observation_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"fit record {name}")
        if type(self.supervised_tokens) is not int or self.supervised_tokens <= 0:
            raise ValueError("fit record supervised_tokens must be positive")
        object.__setattr__(
            self,
            "parent_signed_delta_nll_per_token",
            _finite(
                self.parent_signed_delta_nll_per_token,
                label="parent signed delta NLL/token",
            ),
        )
        object.__setattr__(
            self,
            "jacobian_by_bin",
            _float4(self.jacobian_by_bin, label="jacobian_by_bin"),
        )
        object.__setattr__(
            self,
            "active_rows_by_bin",
            _int4(self.active_rows_by_bin, label="active_rows_by_bin"),
        )
        object.__setattr__(
            self,
            "fit_record_sha256",
            _sha256(_FIT_RECORD_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "model_inputs_sha256": self.model_inputs_sha256,
            "parent_execution_sha256": self.parent_execution_sha256,
            "parent_observation_sha256": self.parent_observation_sha256,
            "supervised_tokens": self.supervised_tokens,
            "parent_signed_delta_nll_per_token": (
                self.parent_signed_delta_nll_per_token
            ),
            "jacobian_by_bin": self.jacobian_by_bin,
            "active_rows_by_bin": self.active_rows_by_bin,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fit_record_sha256": self.fit_record_sha256}


def build_gemma_iterative_residual_fit_record(
    *,
    example: object,
    parent_execution: object,
    gradient: Tensor,
    lag_b_correction: Tensor,
    parent_observation: GemmaH4DampingFiniteNLLObservation,
) -> GemmaIterativeResidualFitRecord:
    """Reduce one exact parent NLL VJP to four causal scalar derivatives."""

    if not isinstance(
        parent_observation,
        GemmaH4DampingFiniteNLLObservation,
    ):
        raise TypeError("parent_observation has the wrong type")
    validate = getattr(parent_execution, "validate_integrity", None)
    if not callable(validate):
        raise TypeError("parent execution lacks integrity validation")
    validate()
    prefix = getattr(parent_execution, "prefix", None)
    if not isinstance(prefix, Gemma3L3L4OnePassPrefix):
        raise TypeError("parent execution omitted its authenticated prefix")
    prefix.validate_integrity()
    if (
        not isinstance(gradient, Tensor)
        or not isinstance(lag_b_correction, Tensor)
        or gradient.shape != lag_b_correction.shape
        or gradient.shape[:2] != prefix.logical_positions.shape
        or gradient.ndim != 3
        or not gradient.is_floating_point()
        or not lag_b_correction.is_floating_point()
    ):
        raise ValueError("behavior-gradient fit geometry differs")
    active = prefix.target_affected_mask
    if (
        bool(active.any())
        and (
            not bool(torch.isfinite(gradient[active.to(gradient.device)]).all())
            or not bool(
                torch.isfinite(
                    lag_b_correction[active.to(lag_b_correction.device)]
                ).all()
            )
        )
    ):
        raise ValueError("behavior-gradient fit tensors are nonfinite")
    inactive = ~active
    if bool(inactive.any()) and not bool(
        (lag_b_correction[inactive.to(lag_b_correction.device)] == 0).all()
    ):
        raise ValueError("lag-B correction wrote outside target support")

    example_id = _identifier(
        getattr(example, "example_id", None),
        label="example_id",
    )
    family_id = _identifier(
        getattr(example, "family_id", None),
        label="family_id",
    )
    model_inputs_sha256 = _require_sha256(
        getattr(example, "model_inputs_sha256", None),
        label="example model inputs",
    )
    if (
        parent_observation.example_id != example_id
        or parent_observation.family_id != family_id
        or getattr(parent_execution, "model_inputs_sha256", None)
        != model_inputs_sha256
    ):
        raise ValueError("fit record identities differ")

    bins = causal_position_bin_indices(prefix.logical_positions)
    product = (
        gradient.detach().to(device="cpu", dtype=torch.float64)
        * lag_b_correction.detach().to(device="cpu", dtype=torch.float64)
    )
    active_cpu = active.detach().to(device="cpu")
    bins_cpu = bins.detach().to(device="cpu")
    derivatives: list[float] = []
    counts: list[int] = []
    for index in range(CAUSAL_POSITION_BIN_COUNT):
        selected = active_cpu & (bins_cpu == index)
        counts.append(int(selected.sum()))
        derivatives.append(
            float(product[selected].sum()) / parent_observation.supervised_tokens
        )
    signed = (
        parent_observation.candidate_summed_nll
        - parent_observation.source_summed_nll
    ) / parent_observation.supervised_tokens
    return GemmaIterativeResidualFitRecord(
        example_id=example_id,
        family_id=family_id,
        model_inputs_sha256=model_inputs_sha256,
        parent_execution_sha256=_require_sha256(
            getattr(parent_execution, "artifact_sha256", None),
            label="parent execution",
        ),
        parent_observation_sha256=parent_observation.observation_sha256,
        supervised_tokens=parent_observation.supervised_tokens,
        parent_signed_delta_nll_per_token=signed,
        jacobian_by_bin=tuple(derivatives),  # type: ignore[arg-type]
        active_rows_by_bin=tuple(counts),  # type: ignore[arg-type]
    )


def _record(value: object) -> GemmaIterativeResidualFitRecord:
    if isinstance(value, GemmaIterativeResidualFitRecord):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("fit records must be mappings or strict records")
    expected = {
        "example_id",
        "family_id",
        "model_inputs_sha256",
        "parent_execution_sha256",
        "parent_observation_sha256",
        "supervised_tokens",
        "parent_signed_delta_nll_per_token",
        "jacobian_by_bin",
        "active_rows_by_bin",
        "fit_record_sha256",
    }
    if set(value) != expected:
        raise ValueError("serialized fit-record fields differ")
    result = GemmaIterativeResidualFitRecord(
        example_id=value["example_id"],  # type: ignore[arg-type]
        family_id=value["family_id"],  # type: ignore[arg-type]
        model_inputs_sha256=value["model_inputs_sha256"],  # type: ignore[arg-type]
        parent_execution_sha256=value["parent_execution_sha256"],  # type: ignore[arg-type]
        parent_observation_sha256=value["parent_observation_sha256"],  # type: ignore[arg-type]
        supervised_tokens=value["supervised_tokens"],  # type: ignore[arg-type]
        parent_signed_delta_nll_per_token=value[
            "parent_signed_delta_nll_per_token"
        ],  # type: ignore[arg-type]
        jacobian_by_bin=value["jacobian_by_bin"],  # type: ignore[arg-type]
        active_rows_by_bin=value["active_rows_by_bin"],  # type: ignore[arg-type]
    )
    if result.fit_record_sha256 != value["fit_record_sha256"]:
        raise ValueError("fit record hash mismatch")
    return result


@dataclass(frozen=True, slots=True)
class GemmaIterativeResidualFoldFit:
    """Replayable result of one fixed weighted-ridge fit."""

    held_family_id: str
    train_example_ids: tuple[str, ...]
    train_family_ids: tuple[str, ...]
    train_fit_record_sha256s: tuple[str, ...]
    coefficients_by_bin: tuple[float, float, float, float]
    unsupported_bin_indices: tuple[int, ...]
    active_rows_by_bin: tuple[int, int, int, int]
    weighted_column_norm_by_bin: tuple[float, float, float, float]
    normal_condition_number: float
    linearized_rmse_before: float
    linearized_rmse_after: float
    linearization_extrapolation: bool
    ridge: float = RIDGE
    trust_bound: float = TRUST_BOUND
    fold_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.held_family_id != "__full_fit__":
            _identifier(self.held_family_id, label="held_family_id")
        for name in (
            "train_example_ids",
            "train_family_ids",
            "train_fit_record_sha256s",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be canonical and unique")
        for value in self.train_fit_record_sha256s:
            _require_sha256(value, label="training fit-record receipt")
        object.__setattr__(
            self,
            "coefficients_by_bin",
            _float4(self.coefficients_by_bin, label="coefficients_by_bin"),
        )
        if any(abs(value) > self.trust_bound for value in self.coefficients_by_bin):
            raise ValueError("a fitted coefficient exceeds the trust bound")
        if (
            type(self.unsupported_bin_indices) is not tuple
            or self.unsupported_bin_indices
            != tuple(sorted(set(self.unsupported_bin_indices)))
            or any(
                type(value) is not int or not 0 <= value < 4
                for value in self.unsupported_bin_indices
            )
        ):
            raise ValueError("unsupported bins must be canonical indices")
        object.__setattr__(
            self,
            "active_rows_by_bin",
            _int4(self.active_rows_by_bin, label="active_rows_by_bin"),
        )
        object.__setattr__(
            self,
            "weighted_column_norm_by_bin",
            _float4(
                self.weighted_column_norm_by_bin,
                label="weighted_column_norm_by_bin",
            ),
        )
        for name in (
            "normal_condition_number",
            "linearized_rmse_before",
            "linearized_rmse_after",
            "ridge",
            "trust_bound",
        ):
            value = _finite(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.ridge != RIDGE or self.trust_bound != TRUST_BOUND:
            raise ValueError("the frozen fit recipe cannot be retuned")
        if type(self.linearization_extrapolation) is not bool:
            raise TypeError("linearization_extrapolation must be boolean")
        object.__setattr__(
            self,
            "fold_receipt_sha256",
            _sha256(_FOLD_FIT_DOMAIN, self._payload()),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "held_family_id": self.held_family_id,
            "train_example_ids": self.train_example_ids,
            "train_family_ids": self.train_family_ids,
            "train_fit_record_sha256s": self.train_fit_record_sha256s,
            "coefficients_by_bin": self.coefficients_by_bin,
            "unsupported_bin_indices": self.unsupported_bin_indices,
            "active_rows_by_bin": self.active_rows_by_bin,
            "weighted_column_norm_by_bin": self.weighted_column_norm_by_bin,
            "normal_condition_number": self.normal_condition_number,
            "linearized_rmse_before": self.linearized_rmse_before,
            "linearized_rmse_after": self.linearized_rmse_after,
            "linearization_extrapolation": self.linearization_extrapolation,
            "ridge": self.ridge,
            "trust_bound": self.trust_bound,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fold_receipt_sha256": self.fold_receipt_sha256,
        }


def fit_gemma_iterative_residual_fold(
    records: Sequence[object],
    *,
    held_family_id: str,
) -> GemmaIterativeResidualFoldFit:
    """Fit four bounded scales with equal family and prompt mass."""

    selected = tuple(
        sorted((_record(value) for value in records), key=lambda row: row.example_id)
    )
    if not selected or len({row.example_id for row in selected}) != len(selected):
        raise ValueError("fit records must be nonempty and unique")
    family_counts = Counter(row.family_id for row in selected)
    if held_family_id != "__full_fit__" and held_family_id in family_counts:
        raise ValueError("the held family leaked into its training records")
    if any(count <= 0 for count in family_counts.values()):
        raise ValueError("fit record family geometry differs")

    design = torch.tensor(
        [row.jacobian_by_bin for row in selected],
        dtype=torch.float64,
    )
    target = -torch.tensor(
        [row.parent_signed_delta_nll_per_token for row in selected],
        dtype=torch.float64,
    )
    family_mass = 1.0 / len(family_counts)
    weights = torch.tensor(
        [
            family_mass / family_counts[row.family_id]
            for row in selected
        ],
        dtype=torch.float64,
    )
    column_squares = (weights[:, None] * design.square()).sum(dim=0)
    column_norms = torch.sqrt(column_squares)
    active_counts = tuple(
        sum(row.active_rows_by_bin[index] for row in selected)
        for index in range(4)
    )
    supported = tuple(
        index
        for index in range(4)
        if active_counts[index] > 0
        and float(column_norms[index]) > _COLUMN_SUPPORT_EPSILON
    )
    coefficients = torch.zeros(4, dtype=torch.float64)
    condition = 0.0
    if supported:
        indices = torch.tensor(supported, dtype=torch.int64)
        x = design.index_select(1, indices)
        normal = x.T @ (weights[:, None] * x)
        eigenvalues = torch.linalg.eigvalsh(normal)
        positive = eigenvalues[eigenvalues > _COLUMN_SUPPORT_EPSILON]
        condition = (
            float(eigenvalues.max() / positive.min())
            if positive.numel()
            else 0.0
        )
        regularized = normal + RIDGE * torch.eye(
            len(supported),
            dtype=torch.float64,
        )
        right = x.T @ (weights * target)
        solved = torch.linalg.solve(regularized, right)
        if not bool(torch.isfinite(solved).all()):
            raise RuntimeError("iterative residual ridge fit became nonfinite")
        coefficients[indices] = solved
    clipped = coefficients.clamp(-TRUST_BOUND, TRUST_BOUND)
    extrapolation = bool((clipped != coefficients).any())
    coefficients = clipped
    prediction_before = torch.zeros_like(target)
    prediction_after = design @ coefficients
    before = float(
        torch.sqrt((weights * (prediction_before - target).square()).sum())
    )
    after = float(
        torch.sqrt((weights * (prediction_after - target).square()).sum())
    )
    return GemmaIterativeResidualFoldFit(
        held_family_id=held_family_id,
        train_example_ids=tuple(row.example_id for row in selected),
        train_family_ids=tuple(sorted(family_counts)),
        train_fit_record_sha256s=tuple(
            sorted(row.fit_record_sha256 for row in selected)
        ),
        coefficients_by_bin=tuple(float(value) for value in coefficients),  # type: ignore[arg-type]
        unsupported_bin_indices=tuple(
            index for index in range(4) if index not in supported
        ),
        active_rows_by_bin=active_counts,  # type: ignore[arg-type]
        weighted_column_norm_by_bin=tuple(
            float(value) for value in column_norms
        ),  # type: ignore[arg-type]
        normal_condition_number=condition,
        linearized_rmse_before=before,
        linearized_rmse_after=after,
        linearization_extrapolation=extrapolation,
    )


@dataclass(frozen=True, slots=True)
class GemmaCausalPositionScaleH4Provider(Gemma3L3L4CorrectionProvider):
    """Authenticated lag-B plus four-scalar causal amplitude gate."""

    parent_h4: Gemma3L3L4CorrectionProvider
    parent_artifact_sha256: str
    fold_fit: GemmaIterativeResidualFoldFit
    site: str = field(init=False, default=_H4_SITE)
    artifact_sha256: str = field(init=False)
    _parent_h4_sha256: str = field(init=False, repr=False)
    _bridge_binding_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parent_h4, Gemma3L3L4CorrectionProvider):
            raise TypeError("parent_h4 must be an authenticated provider")
        self.parent_h4.validate_integrity()
        if self.parent_h4.site != _H4_SITE:
            raise ValueError("position gate requires an H4 parent")
        parent_h4_sha256 = _require_sha256(
            self.parent_h4.artifact_sha256,
            label="parent H4 artifact",
        )
        parent_artifact_sha256 = _require_sha256(
            self.parent_artifact_sha256,
            label="parent artifact",
        )
        bridge_binding = _require_sha256(
            getattr(self.parent_h4, "bridge_binding_sha256", None),
            label="parent H4 bridge binding",
        )
        if not isinstance(self.fold_fit, GemmaIterativeResidualFoldFit):
            raise TypeError("fold_fit must be a strict fold fit")
        object.__setattr__(self, "_parent_h4_sha256", parent_h4_sha256)
        object.__setattr__(self, "_bridge_binding_sha256", bridge_binding)
        object.__setattr__(
            self,
            "artifact_sha256",
            gemma_causal_position_scale_provider_artifact_sha256(
                parent_artifact_sha256=self.parent_artifact_sha256,
                parent_h4_artifact_sha256=parent_h4_sha256,
                bridge_binding_sha256=bridge_binding,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                coefficients_by_bin=self.coefficients_by_bin,
            ),
        )
        self.validate_integrity()

    @property
    def bridge_binding_sha256(self) -> str:
        return self._bridge_binding_sha256

    @property
    def coefficients_by_bin(self) -> tuple[float, float, float, float]:
        return self.fold_fit.coefficients_by_bin

    @property
    def width(self) -> int:
        width = getattr(self.parent_h4, "width", None)
        if type(width) is not int or width <= 0:
            raise ValueError("parent H4 provider omitted its residual width")
        return width

    @property
    def marginal_prepared_float_scalar_count(self) -> int:
        return CAUSAL_POSITION_BIN_COUNT

    @property
    def marginal_logical_macs_per_token_upper_bound(self) -> int:
        return self.width

    @property
    def prepared_float_scalar_count(self) -> int:
        return int(
            getattr(self.parent_h4, "prepared_float_scalar_count")
        ) + self.marginal_prepared_float_scalar_count

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return int(
            getattr(self.parent_h4, "logical_macs_per_token_upper_bound")
        ) + self.marginal_logical_macs_per_token_upper_bound

    def _payload(self) -> dict[str, object]:
        return {
            "semantics": (
                "lag_b_times_one_plus_causal_logical_position_theta_v1"
            ),
            "site": self.site,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "parent_h4_artifact_sha256": self._parent_h4_sha256,
            "bridge_binding_sha256": self._bridge_binding_sha256,
            "causal_position_bin_edges": CAUSAL_POSITION_BIN_EDGES,
            "fold_receipt_sha256": self.fold_fit.fold_receipt_sha256,
            "coefficients_by_bin": self.coefficients_by_bin,
        }

    def validate_integrity(self) -> None:
        self.parent_h4.validate_integrity()
        if (
            self.parent_h4.artifact_sha256 != self._parent_h4_sha256
            or getattr(self.parent_h4, "bridge_binding_sha256", None)
            != self._bridge_binding_sha256
            or gemma_causal_position_scale_provider_artifact_sha256(
                parent_artifact_sha256=self.parent_artifact_sha256,
                parent_h4_artifact_sha256=self._parent_h4_sha256,
                bridge_binding_sha256=self._bridge_binding_sha256,
                fold_receipt_sha256=self.fold_fit.fold_receipt_sha256,
                coefficients_by_bin=self.coefficients_by_bin,
            )
            != self.artifact_sha256
        ):
            raise RuntimeError("position-scale provider integrity drifted")

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        self.validate_integrity()
        prefix.validate_integrity()
        if prefix.bridge_binding_sha256 != self.bridge_binding_sha256:
            raise ValueError("position-scale provider belongs to another bridge")
        prefix_sha256 = prefix.artifact_sha256
        base = self.parent_h4.correction(prefix, realized_state)
        self.parent_h4.validate_integrity()
        prefix.validate_integrity()
        if prefix.artifact_sha256 != prefix_sha256:
            raise RuntimeError("parent H4 provider mutated the prefix")
        if (
            not isinstance(base, Tensor)
            or base.shape != realized_state.shape
            or not base.is_floating_point()
        ):
            raise ValueError("parent H4 correction geometry differs")
        bins = causal_position_bin_indices(
            prefix.logical_positions.to(base.device)
        )
        theta = torch.tensor(
            self.coefficients_by_bin,
            device=base.device,
            dtype=base.dtype,
        )
        multiplier = 1.0 + theta[bins]
        result = base * multiplier.unsqueeze(-1)
        inactive = ~prefix.target_affected_mask.to(result.device)
        if bool(inactive.any()) and not bool((result[inactive] == 0).all()):
            raise RuntimeError("position-scale provider wrote off support")
        active = prefix.target_affected_mask.to(result.device)
        if bool(active.any()) and not bool(torch.isfinite(result[active]).all()):
            raise ValueError("position-scale correction became nonfinite")
        self.validate_integrity()
        return result


def fit_gemma_iterative_residual_fold_provider(
    *,
    records: Sequence[object],
    held_family: str,
    parent_h4: Gemma3L3L4CorrectionProvider,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalPositionScaleH4Provider:
    """Fit one OOF realization of the single frozen position-gate recipe."""

    fit = fit_gemma_iterative_residual_fold(
        records,
        held_family_id=held_family,
    )
    return GemmaCausalPositionScaleH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )


def fit_gemma_iterative_residual_full_provider(
    *,
    records: Sequence[object],
    parent_h4: Gemma3L3L4CorrectionProvider,
    parent_artifact_sha256: str | None = None,
) -> GemmaCausalPositionScaleH4Provider:
    """Refit the frozen recipe on all development rows after OOF retention."""

    fit = fit_gemma_iterative_residual_fold(
        records,
        held_family_id="__full_fit__",
    )
    return GemmaCausalPositionScaleH4Provider(
        parent_h4=parent_h4,
        parent_artifact_sha256=(
            parent_h4.artifact_sha256
            if parent_artifact_sha256 is None
            else parent_artifact_sha256
        ),
        fold_fit=fit,
    )
