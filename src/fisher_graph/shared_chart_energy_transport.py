"""Deterministic shared-chart geometry and directed transport numerics.

This module is model-independent.  It accepts already-authenticated h-width
context, value, and generalized-contribution rows; it never sees prompt text,
token IDs, model weights, or a tokenizer.  All fitting and energy accumulation
uses finite CPU float64 tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from torch import Tensor


SHARED_CHART_ATLAS_SCHEMA = "fisher_graph.shared_chart_energy_atlas.v1"
DIRECTED_CHART_TRANSPORT_SCHEMA = (
    "fisher_graph.shared_directed_chart_transport.v1"
)
COMPARISON_ABI_NAME = "carrier_chart_energy.h640.v1"
FIT_ROLE = "A_chart_transport_fit"
ROW_SHUFFLE_SEED = 6_607
ROW_SHUFFLE_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-row-shuffle-per-example:v2\0"
)
ROW_SHUFFLE_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-chart-native-neighbor-row-shuffle-per-example-catalog:v2\0"
)

_ROW_CATALOG_DOMAIN = b"fisher_graph:shared-chart-row-catalog:v1\0"
_ROW_BATCH_DOMAIN = b"fisher_graph:shared-chart-row-batch:v1\0"
_ATLAS_DOMAIN = b"fisher_graph:shared-chart-atlas:v1\0"
_ATLAS_CONTEXT_CATALOG_DOMAIN = (
    b"fisher_graph:shared-chart-atlas-context-catalog:v1\0"
)
_ATLAS_WEIGHT_CATALOG_DOMAIN = (
    b"fisher_graph:shared-chart-atlas-weight-catalog:v1\0"
)
_ATLAS_SOLVER_RECEIPT_DOMAIN = (
    b"fisher_graph:shared-chart-atlas-solver-receipt:v1\0"
)
_APPLIED_DOMAIN = b"fisher_graph:shared-chart-applied:v1\0"
_TRANSPORT_DOMAIN = b"fisher_graph:shared-chart-transport:v1\0"
_FIT_CATALOG_DOMAIN = b"fisher_graph:shared-chart-fit-catalog:v1\0"
_SPLIT_DOMAIN = b"fisher_graph:shared-chart-split:v1\0"
_TENSOR_DOMAIN = b"fisher_graph:shared-chart-tensor:v1\0"
_BUNDLE_MAGIC = b"FGCHART1\n"
_BUNDLE_SCHEMA = "fisher_graph.strict_float64_tensor_bundle.v1"
_MAX_BUNDLE_BYTES = 4 * 1024 * 1024 * 1024
PREDICTION_ROW_CHUNK_SIZE = 256
PREDICTION_BACKEND_ID = "torch_mm_cpu_float64_fixed_rows_v1"
_PREDICTION_BACKEND_DOMAIN = (
    b"fisher_graph:shared-chart-prediction-backend:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_loads(raw: bytes) -> object:
    def pairs(entries: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        result = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("strict canonical JSON is invalid") from error
    if _canonical_json_bytes(result) != raw:
        raise ValueError("JSON bytes are not in canonical form")
    return result


def _digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _float64_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not value.shape[0]
        or not value.shape[1]
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating matrix")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


def _float64_tensor(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or any(not size for size in value.shape)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite rank-{ndim} tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


def _tensor_bytes(value: Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("strict tensor artifacts require little-endian host")
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return tensor.numpy().tobytes(order="C")


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(_canonical_json_bytes({"dtype": "float64_le", "shape": list(tensor.shape)}))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ChartRowKey:
    family_id: str
    example_id: str
    prediction_query_position: int
    target_label_position: int

    def __post_init__(self) -> None:
        if not self.family_id or not self.example_id:
            raise ValueError("chart row family/example IDs must be nonempty")
        if (
            type(self.prediction_query_position) is not int
            or self.prediction_query_position < 0
            or type(self.target_label_position) is not int
            or self.target_label_position != self.prediction_query_position + 1
        ):
            raise ValueError(
                "chart rows require q>=0 and target label position t=q+1"
            )

    def payload(self) -> list[object]:
        return [
            self.family_id,
            self.example_id,
            self.prediction_query_position,
            self.target_label_position,
        ]


def row_catalog_sha256(
    row_keys: Sequence[ChartRowKey],
    *,
    role: str,
) -> str:
    keys = tuple(row_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("chart row keys must be nonempty and unique")
    if not isinstance(role, str) or not role:
        raise ValueError("chart row catalog role must be nonempty")
    return _digest(
        _ROW_CATALOG_DOMAIN,
        {"role": role, "rows": [key.payload() for key in keys]},
    )


@dataclass(frozen=True, slots=True)
class SharedChartRowBatch:
    """One mode's aligned rows on the common carrier comparison surface."""

    role: str
    panel_binding_sha256: str
    mode_id: str
    row_keys: tuple[ChartRowKey, ...]
    context_values: Tensor
    comparison_values: Tensor
    generalized_contributions: Tensor

    def __post_init__(self) -> None:
        if not self.role or not self.mode_id:
            raise ValueError("row batch role and mode_id must be nonempty")
        _require_sha256(self.panel_binding_sha256, label="panel_binding_sha256")
        if not isinstance(self.row_keys, tuple):
            raise TypeError("row_keys must be a tuple")
        row_catalog_sha256(self.row_keys, role=self.role)
        context = _float64_matrix(self.context_values, label="context_values")
        values = _float64_matrix(self.comparison_values, label="comparison_values")
        contribution = _float64_matrix(
            self.generalized_contributions,
            label="generalized_contributions",
        )
        if context.shape != values.shape or context.shape != contribution.shape:
            raise ValueError("context, value, and contribution shapes must match")
        if context.shape[0] != len(self.row_keys):
            raise ValueError("tensor rows must match row_keys")
        object.__setattr__(self, "context_values", context)
        object.__setattr__(self, "comparison_values", values)
        object.__setattr__(self, "generalized_contributions", contribution)

    @property
    def width(self) -> int:
        return int(self.context_values.shape[1])

    @property
    def row_catalog_sha256(self) -> str:
        return row_catalog_sha256(self.row_keys, role=self.role)

    @property
    def tensor_catalog_sha256(self) -> str:
        """Sign-complete receipt for every full-width tensor in this batch."""

        return _digest(
            _ROW_BATCH_DOMAIN,
            {
                "mode_id": self.mode_id,
                "panel_binding_sha256": self.panel_binding_sha256,
                "role": self.role,
                "row_catalog_sha256": self.row_catalog_sha256,
                "tensor_sha256s": {
                    "comparison_values": _tensor_sha256(
                        self.comparison_values
                    ),
                    "context_values": _tensor_sha256(self.context_values),
                    "generalized_contributions": _tensor_sha256(
                        self.generalized_contributions
                    ),
                },
            },
        )


@dataclass(frozen=True, slots=True)
class SharedChartNumericsConfig:
    hidden_width: int = 640
    chart_count: int = 8
    tangent_rank: int = 16
    lloyd_iteration_count: int = 12
    membership_temperature: float = 1.0
    radius_floor_fraction: float = 1.0e-8
    variance_floor_fraction: float = 1.0e-8
    minimum_hard_occupancy_fraction: float = 0.005
    eigenvalue_tie_relative_tolerance: float = 1.0e-12
    transport_relative_ridge: float = 1.0e-3
    ridge_scale_floor: float = 1.0e-12
    transport_fisher_energy_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        for label in ("hidden_width", "chart_count", "tangent_rank", "lloyd_iteration_count"):
            value = getattr(self, label)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.tangent_rank >= self.hidden_width:
            raise ValueError("tangent_rank must be smaller than hidden_width")
        for label in (
            "membership_temperature",
            "radius_floor_fraction",
            "variance_floor_fraction",
            "minimum_hard_occupancy_fraction",
            "eigenvalue_tie_relative_tolerance",
            "transport_relative_ridge",
            "ridge_scale_floor",
            "transport_fisher_energy_floor",
        ):
            value = getattr(self, label)
            if not isinstance(value, float) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        if self.minimum_hard_occupancy_fraction >= 1.0:
            raise ValueError("minimum_hard_occupancy_fraction must be below one")

    @property
    def source_feature_width(self) -> int:
        return self.chart_count * (self.tangent_rank + 1)

    def payload(self) -> dict[str, object]:
        return {
            "chart_count": self.chart_count,
            "eigenvalue_tie_relative_tolerance": self.eigenvalue_tie_relative_tolerance,
            "hidden_width": self.hidden_width,
            "lloyd_iteration_count": self.lloyd_iteration_count,
            "membership_temperature": self.membership_temperature,
            "minimum_hard_occupancy_fraction": self.minimum_hard_occupancy_fraction,
            "radius_floor_fraction": self.radius_floor_fraction,
            "ridge_scale_floor": self.ridge_scale_floor,
            "tangent_rank": self.tangent_rank,
            "transport_relative_ridge": self.transport_relative_ridge,
            "transport_fisher_energy_floor": (
                self.transport_fisher_energy_floor
            ),
            "variance_floor_fraction": self.variance_floor_fraction,
        }


def _validate_aligned_batches(
    batches: Sequence[SharedChartRowBatch],
    *,
    expected_role: str | None = None,
) -> tuple[SharedChartRowBatch, ...]:
    materialized = tuple(sorted(batches, key=lambda item: item.mode_id))
    if not materialized or len({item.mode_id for item in materialized}) != len(materialized):
        raise ValueError("row batches must contain unique modes")
    first = materialized[0]
    if expected_role is not None and first.role != expected_role:
        raise ValueError(f"row batches must use role {expected_role}")
    for item in materialized:
        if (
            item.role != first.role
            or item.panel_binding_sha256 != first.panel_binding_sha256
            or item.row_keys != first.row_keys
            or item.width != first.width
        ):
            raise ValueError("all mode batches must share one aligned role/panel/row catalog")
    return materialized


def balanced_atlas_row_weights(
    batches: Sequence[SharedChartRowBatch],
) -> dict[str, Tensor]:
    """Equal mode, family, example, then row weights; total weight is one."""

    aligned = _validate_aligned_batches(batches)
    keys = aligned[0].row_keys
    families = tuple(sorted({key.family_id for key in keys}))
    examples_by_family = {
        family: tuple(sorted({key.example_id for key in keys if key.family_id == family}))
        for family in families
    }
    row_count = {
        (family, example): sum(
            key.family_id == family and key.example_id == example for key in keys
        )
        for family in families
        for example in examples_by_family[family]
    }
    mode_count = len(aligned)
    family_count = len(families)
    base = torch.empty(len(keys), dtype=torch.float64)
    for index, key in enumerate(keys):
        base[index] = 1.0 / (
            mode_count
            * family_count
            * len(examples_by_family[key.family_id])
            * row_count[(key.family_id, key.example_id)]
        )
    if not math.isclose(float(base.sum()) * mode_count, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("balanced chart row weights did not close to one")
    return {batch.mode_id: base.clone() for batch in aligned}


def _canonicalize_vector_signs(basis: Tensor) -> Tensor:
    result = basis.clone()
    for column in range(result.shape[1]):
        vector = result[:, column]
        absolute = vector.abs()
        pivot = int(torch.argmax(absolute).item())
        if float(vector[pivot]) < 0.0:
            result[:, column] = -vector
    return result


@dataclass(frozen=True, slots=True)
class SharedChartAtlas:
    config: SharedChartNumericsConfig
    fit_role: str
    fit_panel_binding_sha256: str
    fit_row_catalog_sha256: str
    fit_mode_ids: tuple[str, ...]
    fit_context_catalog_sha256: str
    fit_balanced_row_weight_catalog_sha256: str
    fit_solver_receipt_sha256: str
    centers: Tensor
    radius_squared: Tensor
    tangent_bases: Tensor
    tangent_variance: Tensor
    orthogonal_alpha: Tensor
    hard_occupancy_count: Tensor
    hard_occupancy_weight: Tensor
    atlas_sha256: str

    def __post_init__(self) -> None:
        if self.fit_role != FIT_ROLE:
            raise ValueError(f"fit_role must be {FIT_ROLE}")
        _require_sha256(self.fit_panel_binding_sha256, label="fit_panel_binding_sha256")
        _require_sha256(self.fit_row_catalog_sha256, label="fit_row_catalog_sha256")
        _require_sha256(
            self.fit_context_catalog_sha256,
            label="fit_context_catalog_sha256",
        )
        _require_sha256(
            self.fit_balanced_row_weight_catalog_sha256,
            label="fit_balanced_row_weight_catalog_sha256",
        )
        _require_sha256(
            self.fit_solver_receipt_sha256,
            label="fit_solver_receipt_sha256",
        )
        if not self.fit_mode_ids or len(set(self.fit_mode_ids)) != len(self.fit_mode_ids):
            raise ValueError("fit_mode_ids must be nonempty and unique")
        centers = _float64_matrix(self.centers, label="centers")
        radii = _float64_tensor(self.radius_squared, label="radius_squared", ndim=1)
        bases = _float64_tensor(self.tangent_bases, label="tangent_bases", ndim=3)
        variance = _float64_matrix(self.tangent_variance, label="tangent_variance")
        alpha = _float64_tensor(self.orthogonal_alpha, label="orthogonal_alpha", ndim=1)
        counts = self.hard_occupancy_count.detach().to(device="cpu", dtype=torch.int64).clone().contiguous()
        weights = _float64_tensor(self.hard_occupancy_weight, label="hard_occupancy_weight", ndim=1)
        expected = (self.config.chart_count, self.config.hidden_width)
        if tuple(centers.shape) != expected:
            raise ValueError("center shape disagrees with config")
        if tuple(bases.shape) != (
            self.config.chart_count,
            self.config.hidden_width,
            self.config.tangent_rank,
        ):
            raise ValueError("basis shape disagrees with config")
        if tuple(variance.shape) != (self.config.chart_count, self.config.tangent_rank):
            raise ValueError("variance shape disagrees with config")
        if any(tuple(value.shape) != (self.config.chart_count,) for value in (radii, alpha, counts, weights)):
            raise ValueError("per-chart vector shape disagrees with config")
        if bool((radii <= 0).any()) or bool((variance <= 0).any()) or bool((alpha <= 0).any()):
            raise ValueError("atlas scales and alpha must be positive")
        identity = torch.eye(self.config.tangent_rank, dtype=torch.float64)
        for chart in range(self.config.chart_count):
            if not torch.allclose(bases[chart].T @ bases[chart], identity, atol=1e-9, rtol=1e-9):
                raise ValueError("tangent bases must have orthonormal columns")
        for field, resolved in (
            ("centers", centers),
            ("radius_squared", radii),
            ("tangent_bases", bases),
            ("tangent_variance", variance),
            ("orthogonal_alpha", alpha),
            ("hard_occupancy_count", counts),
            ("hard_occupancy_weight", weights),
        ):
            object.__setattr__(self, field, resolved)
        _require_sha256(self.atlas_sha256, label="atlas_sha256")
        if self.atlas_sha256 != _atlas_sha256(self):
            raise ValueError("atlas_sha256 does not authenticate atlas state")

    def receipt_payload(self) -> dict[str, object]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "config": self.config.payload(),
            "fit_mode_ids": list(self.fit_mode_ids),
            "fit_balanced_row_weight_catalog_sha256": (
                self.fit_balanced_row_weight_catalog_sha256
            ),
            "fit_context_catalog_sha256": self.fit_context_catalog_sha256,
            "fit_panel_binding_sha256": self.fit_panel_binding_sha256,
            "fit_role": self.fit_role,
            "fit_row_catalog_sha256": self.fit_row_catalog_sha256,
            "fit_solver_receipt_sha256": self.fit_solver_receipt_sha256,
            "hard_occupancy_count": self.hard_occupancy_count.tolist(),
            "hard_occupancy_weight": self.hard_occupancy_weight.tolist(),
            "schema": SHARED_CHART_ATLAS_SCHEMA,
            "tensor_sha256s": _atlas_tensor_sha256s(self),
        }


def _atlas_tensor_sha256s(atlas: SharedChartAtlas) -> dict[str, str]:
    return {
        name: _tensor_sha256(getattr(atlas, name))
        for name in (
            "centers",
            "radius_squared",
            "tangent_bases",
            "tangent_variance",
            "orthogonal_alpha",
        )
    }


def _atlas_sha256(atlas: SharedChartAtlas) -> str:
    return _digest(
        _ATLAS_DOMAIN,
        {
            "config": atlas.config.payload(),
            "fit_mode_ids": list(atlas.fit_mode_ids),
            "fit_balanced_row_weight_catalog_sha256": (
                atlas.fit_balanced_row_weight_catalog_sha256
            ),
            "fit_context_catalog_sha256": atlas.fit_context_catalog_sha256,
            "fit_panel_binding_sha256": atlas.fit_panel_binding_sha256,
            "fit_role": atlas.fit_role,
            "fit_row_catalog_sha256": atlas.fit_row_catalog_sha256,
            "fit_solver_receipt_sha256": atlas.fit_solver_receipt_sha256,
            "hard_occupancy_count": atlas.hard_occupancy_count.tolist(),
            "hard_occupancy_weight": atlas.hard_occupancy_weight.tolist(),
            "tensor_sha256s": _atlas_tensor_sha256s(atlas),
        },
    )


def _memberships(atlas: SharedChartAtlas, contexts: Tensor) -> Tensor:
    squared = (contexts[:, None, :] - atlas.centers[None, :, :]).square().sum(dim=2)
    logits = -0.5 * squared / (
        atlas.radius_squared[None, :] * atlas.config.membership_temperature
    )
    return torch.softmax(logits, dim=1)


def fit_shared_chart_atlas(
    batches: Sequence[SharedChartRowBatch],
    *,
    config: SharedChartNumericsConfig = SharedChartNumericsConfig(),
) -> SharedChartAtlas:
    """Fit one deterministic shared router and full-rank SPD chart metric."""

    aligned = _validate_aligned_batches(batches, expected_role=FIT_ROLE)
    if aligned[0].width != config.hidden_width:
        raise ValueError("fit rows disagree with configured hidden width")
    weights_by_mode = balanced_atlas_row_weights(aligned)
    fit_context_catalog_sha256 = _digest(
        _ATLAS_CONTEXT_CATALOG_DOMAIN,
        {
            "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
            "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
            "modes": [
                {
                    "context_values_sha256": _tensor_sha256(
                        item.context_values
                    ),
                    "mode_id": item.mode_id,
                }
                for item in aligned
            ],
        },
    )
    fit_balanced_row_weight_catalog_sha256 = _digest(
        _ATLAS_WEIGHT_CATALOG_DOMAIN,
        {
            "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
            "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
            "mode_weights": [
                {
                    "mode_id": item.mode_id,
                    "weight_sha256": _tensor_sha256(
                        weights_by_mode[item.mode_id]
                    ),
                }
                for item in aligned
            ],
        },
    )
    fit_solver_receipt_sha256 = _digest(
        _ATLAS_SOLVER_RECEIPT_DOMAIN,
        {
            "config": config.payload(),
            "fit_balanced_row_weight_catalog_sha256": (
                fit_balanced_row_weight_catalog_sha256
            ),
            "fit_context_catalog_sha256": fit_context_catalog_sha256,
            "solver_id": (
                "balanced_farthest_point_weighted_lloyd12_"
                "soft_router_context_covariance_eigh_v1"
            ),
        },
    )
    contexts = torch.cat([item.context_values for item in aligned], dim=0)
    weights = torch.cat([weights_by_mode[item.mode_id] for item in aligned], dim=0)
    if contexts.shape[0] < config.chart_count * (config.tangent_rank + 1):
        raise ValueError("atlas fit has too few rows for chart/rank occupancy")

    global_mean = (weights[:, None] * contexts).sum(dim=0) / weights.sum()
    first_score = weights * (contexts - global_mean).square().sum(dim=1)
    selected = [int(torch.argmax(first_score).item())]
    minimum_distance = (contexts - contexts[selected[0]]).square().sum(dim=1)
    while len(selected) < config.chart_count:
        score = weights * minimum_distance
        score[selected] = -1.0
        next_index = int(torch.argmax(score).item())
        if float(score[next_index]) <= 0.0:
            raise ValueError("farthest-point chart initialization degenerated")
        selected.append(next_index)
        distance = (contexts - contexts[next_index]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    centers = contexts[selected].clone()

    assignments = torch.zeros(contexts.shape[0], dtype=torch.int64)
    for _ in range(config.lloyd_iteration_count):
        distances = (contexts[:, None, :] - centers[None, :, :]).square().sum(dim=2)
        assignments = torch.argmin(distances, dim=1)
        updated = torch.empty_like(centers)
        for chart in range(config.chart_count):
            mask = assignments == chart
            if not bool(mask.any()):
                raise ValueError("weighted Lloyd iteration produced an empty chart")
            chart_weights = weights[mask]
            updated[chart] = (
                chart_weights[:, None] * contexts[mask]
            ).sum(dim=0) / chart_weights.sum()
        centers = updated

    order = sorted(range(config.chart_count), key=lambda index: tuple(float(x) for x in centers[index]))
    centers = centers[order].contiguous()
    distances = (contexts[:, None, :] - centers[None, :, :]).square().sum(dim=2)
    assignments = torch.argmin(distances, dim=1)
    counts = torch.bincount(assignments, minlength=config.chart_count)
    occupancy_weights = torch.stack(
        [weights[assignments == chart].sum() for chart in range(config.chart_count)]
    )
    minimum_rows = config.tangent_rank + 1
    if bool((counts < minimum_rows).any()):
        raise ValueError("a chart has fewer than tangent_rank+1 hard rows")
    if bool((occupancy_weights < config.minimum_hard_occupancy_fraction).any()):
        raise ValueError("a chart failed the minimum balanced hard occupancy gate")

    context_reference = max(float((weights * contexts.square().sum(dim=1)).sum()), 1.0e-30)
    radius_floor = config.radius_floor_fraction * context_reference
    radii = torch.empty(config.chart_count, dtype=torch.float64)
    for chart in range(config.chart_count):
        mask = assignments == chart
        chart_weights = weights[mask]
        mean_squared = float((chart_weights * distances[mask, chart]).sum() / chart_weights.sum())
        radii[chart] = max(mean_squared, radius_floor)

    provisional = object.__new__(SharedChartAtlas)
    object.__setattr__(provisional, "config", config)
    object.__setattr__(provisional, "centers", centers)
    object.__setattr__(provisional, "radius_squared", radii)
    memberships = _memberships(provisional, contexts)
    centered_global_context = contexts - global_mean
    context_variance_reference = max(
        float((weights * centered_global_context.square().sum(dim=1)).sum())
        / config.hidden_width,
        1.0e-30,
    )
    variance_floor = (
        config.variance_floor_fraction * context_variance_reference
    )
    bases = torch.empty(
        (config.chart_count, config.hidden_width, config.tangent_rank),
        dtype=torch.float64,
    )
    tangent_variance = torch.empty(
        (config.chart_count, config.tangent_rank), dtype=torch.float64
    )
    alpha = torch.empty(config.chart_count, dtype=torch.float64)
    for chart in range(config.chart_count):
        chart_weights = weights * memberships[:, chart]
        occupancy = chart_weights.sum()
        centered_context = contexts - centers[chart]
        covariance = (
            centered_context.T
            @ (chart_weights[:, None] * centered_context)
            / occupancy
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        for ordinal in range(config.tangent_rank):
            following = eigenvalues[ordinal + 1]
            scale = max(1.0, abs(float(eigenvalues[ordinal])), abs(float(following)))
            if abs(float(eigenvalues[ordinal] - following)) <= (
                config.eigenvalue_tie_relative_tolerance * scale
            ):
                raise ValueError("chart tangent eigenvalue orientation is undefined")
        basis = _canonicalize_vector_signs(eigenvectors[:, : config.tangent_rank])
        bases[chart] = basis
        tangent_variance[chart] = torch.clamp(
            eigenvalues[: config.tangent_rank], min=variance_floor
        )
        projection = centered_context @ basis
        residual_square = (
            centered_context.square().sum(dim=1)
            - projection.square().sum(dim=1)
        )
        orthogonal_variance = float(
            (chart_weights * residual_square.clamp_min(0.0)).sum()
            / (occupancy * (config.hidden_width - config.tangent_rank))
        )
        alpha[chart] = 1.0 / max(orthogonal_variance, variance_floor)

    provisional = object.__new__(SharedChartAtlas)
    values = {
        "config": config,
        "fit_role": FIT_ROLE,
        "fit_panel_binding_sha256": aligned[0].panel_binding_sha256,
        "fit_row_catalog_sha256": aligned[0].row_catalog_sha256,
        "fit_mode_ids": tuple(item.mode_id for item in aligned),
        "fit_context_catalog_sha256": fit_context_catalog_sha256,
        "fit_balanced_row_weight_catalog_sha256": (
            fit_balanced_row_weight_catalog_sha256
        ),
        "fit_solver_receipt_sha256": fit_solver_receipt_sha256,
        "centers": centers,
        "radius_squared": radii,
        "tangent_bases": bases,
        "tangent_variance": tangent_variance,
        "orthogonal_alpha": alpha,
        "hard_occupancy_count": counts,
        "hard_occupancy_weight": occupancy_weights,
    }
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    object.__setattr__(provisional, "atlas_sha256", "0" * 64)
    return SharedChartAtlas(
        **values,
        atlas_sha256=_atlas_sha256(provisional),
    )


@dataclass(frozen=True, slots=True)
class AppliedChartRows:
    batch: SharedChartRowBatch
    atlas_sha256: str
    batch_tensor_catalog_sha256: str
    memberships: Tensor
    tangent_coordinates: Tensor
    weighted_activation_energy: Tensor
    weighted_chart_tangent_energy: Tensor
    weighted_chart_orthogonal_energy: Tensor
    weighted_fisher_energy: Tensor
    vector_catalog_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.atlas_sha256, label="atlas_sha256")
        _require_sha256(
            self.batch_tensor_catalog_sha256,
            label="batch_tensor_catalog_sha256",
        )
        if self.batch_tensor_catalog_sha256 != self.batch.tensor_catalog_sha256:
            raise ValueError(
                "batch_tensor_catalog_sha256 does not authenticate source rows"
            )
        rows = len(self.batch.row_keys)
        memberships = _float64_matrix(self.memberships, label="memberships")
        coordinates = _float64_tensor(self.tangent_coordinates, label="tangent_coordinates", ndim=3)
        vectors = []
        for label in (
            "weighted_activation_energy",
            "weighted_chart_tangent_energy",
            "weighted_chart_orthogonal_energy",
            "weighted_fisher_energy",
        ):
            value = _float64_tensor(getattr(self, label), label=label, ndim=1)
            if tuple(value.shape) != (rows,) or bool((value < 0).any()):
                raise ValueError(f"{label} must be nonnegative with one value per row")
            object.__setattr__(self, label, value)
            vectors.append(value)
        if memberships.shape[0] != rows or coordinates.shape[0] != rows:
            raise ValueError("applied chart tensors must match batch rows")
        if not torch.allclose(memberships.sum(dim=1), torch.ones(rows, dtype=torch.float64), atol=1e-12, rtol=1e-12):
            raise ValueError("memberships must form a row simplex")
        if not torch.allclose(vectors[1] + vectors[2], vectors[3], atol=1e-9, rtol=1e-9):
            raise ValueError("tangent plus orthogonal energy must close to Fisher energy")
        object.__setattr__(self, "memberships", memberships)
        object.__setattr__(self, "tangent_coordinates", coordinates)
        _require_sha256(self.vector_catalog_sha256, label="vector_catalog_sha256")
        if self.vector_catalog_sha256 != _applied_vector_catalog_sha256(self):
            raise ValueError(
                "vector_catalog_sha256 does not authenticate applied chart rows"
            )

    def feature_matrix(self) -> Tensor:
        ones = torch.ones(
            (self.memberships.shape[0], self.memberships.shape[1], 1),
            dtype=torch.float64,
        )
        return (self.memberships[:, :, None] * torch.cat((ones, self.tangent_coordinates), dim=2)).reshape(self.memberships.shape[0], -1)


def _applied_vector_catalog_sha256(rows: AppliedChartRows) -> str:
    return _digest(
        _APPLIED_DOMAIN,
        {
            "atlas_sha256": rows.atlas_sha256,
            "batch_tensor_catalog_sha256": rows.batch_tensor_catalog_sha256,
            "mode_id": rows.batch.mode_id,
            "panel_binding_sha256": rows.batch.panel_binding_sha256,
            "role": rows.batch.role,
            "row_catalog_sha256": rows.batch.row_catalog_sha256,
            "tensor_sha256s": {
                "memberships": _tensor_sha256(rows.memberships),
                "tangent_coordinates": _tensor_sha256(
                    rows.tangent_coordinates
                ),
                "weighted_activation_energy": _tensor_sha256(
                    rows.weighted_activation_energy
                ),
                "weighted_chart_orthogonal_energy": _tensor_sha256(
                    rows.weighted_chart_orthogonal_energy
                ),
                "weighted_chart_tangent_energy": _tensor_sha256(
                    rows.weighted_chart_tangent_energy
                ),
                "weighted_fisher_energy": _tensor_sha256(
                    rows.weighted_fisher_energy
                ),
            },
        },
    )


def _spd_energy_parts(atlas: SharedChartAtlas, memberships: Tensor, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    tangent = torch.zeros(values.shape[0], dtype=torch.float64)
    orthogonal = torch.zeros_like(tangent)
    for chart in range(atlas.config.chart_count):
        basis = atlas.tangent_bases[chart]
        projection = values @ basis
        chart_tangent = (projection.square() / atlas.tangent_variance[chart]).sum(dim=1)
        residual = values - projection @ basis.T
        chart_orthogonal = atlas.orthogonal_alpha[chart] * residual.square().sum(dim=1)
        tangent += memberships[:, chart] * chart_tangent
        orthogonal += memberships[:, chart] * chart_orthogonal
    return tangent, orthogonal, tangent + orthogonal


def apply_shared_chart_atlas(
    atlas: SharedChartAtlas,
    batch: SharedChartRowBatch,
) -> AppliedChartRows:
    if batch.width != atlas.config.hidden_width:
        raise ValueError("batch width disagrees with shared atlas")
    memberships = _memberships(atlas, batch.context_values)
    coordinates = torch.empty(
        (len(batch.row_keys), atlas.config.chart_count, atlas.config.tangent_rank),
        dtype=torch.float64,
    )
    for chart in range(atlas.config.chart_count):
        centered = batch.context_values - atlas.centers[chart]
        coordinates[:, chart, :] = (
            centered @ atlas.tangent_bases[chart]
        ) / torch.sqrt(atlas.tangent_variance[chart])
    activation = _spd_energy_parts(atlas, memberships, batch.comparison_values)[2]
    tangent, orthogonal, fisher = _spd_energy_parts(
        atlas, memberships, batch.generalized_contributions
    )
    provisional = object.__new__(AppliedChartRows)
    values = {
        "batch": batch,
        "atlas_sha256": atlas.atlas_sha256,
        "batch_tensor_catalog_sha256": batch.tensor_catalog_sha256,
        "memberships": memberships,
        "tangent_coordinates": coordinates,
        "weighted_activation_energy": activation,
        "weighted_chart_tangent_energy": tangent,
        "weighted_chart_orthogonal_energy": orthogonal,
        "weighted_fisher_energy": fisher,
    }
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    object.__setattr__(provisional, "vector_catalog_sha256", "0" * 64)
    return AppliedChartRows(
        **values,
        vector_catalog_sha256=_applied_vector_catalog_sha256(provisional),
    )


@dataclass(frozen=True, slots=True)
class DirectedChartTransport:
    config: SharedChartNumericsConfig
    source_mode_id: str
    target_mode_id: str
    atlas_sha256: str
    fit_role: str
    fit_panel_binding_sha256: str
    fit_row_catalog_sha256: str
    fit_catalog_sha256: str
    split_receipt_sha256: str
    fit_design_sha256: str
    fit_target_sha256: str
    fit_row_weights_sha256: str
    solver_id: str
    absolute_ridge: float
    coefficient: Tensor
    transport_sha256: str

    def __post_init__(self) -> None:
        if self.source_mode_id == self.target_mode_id:
            raise ValueError("transport endpoints must differ")
        if self.fit_role != FIT_ROLE:
            raise ValueError(f"transport fit role must be {FIT_ROLE}")
        for label in (
            "atlas_sha256",
            "fit_panel_binding_sha256",
            "fit_row_catalog_sha256",
            "fit_catalog_sha256",
            "split_receipt_sha256",
            "fit_design_sha256",
            "fit_target_sha256",
            "fit_row_weights_sha256",
            "transport_sha256",
        ):
            _require_sha256(getattr(self, label), label=label)
        if not math.isfinite(self.absolute_ridge) or self.absolute_ridge <= 0.0:
            raise ValueError("absolute_ridge must be positive")
        if self.solver_id != "balanced_float64_primal_ridge_solve_v1":
            raise ValueError("directed transport solver_id is unsupported")
        coefficient = _float64_matrix(self.coefficient, label="coefficient")
        if tuple(coefficient.shape) != (
            self.config.source_feature_width,
            self.config.hidden_width,
        ):
            raise ValueError("transport coefficient shape disagrees with config")
        object.__setattr__(self, "coefficient", coefficient)
        if self.transport_sha256 != _transport_sha256(self):
            raise ValueError("transport_sha256 does not authenticate state")

    def predict(self, source: AppliedChartRows) -> Tensor:
        if source.batch.mode_id != self.source_mode_id or source.atlas_sha256 != self.atlas_sha256:
            raise ValueError("source rows do not match frozen transport")
        return _fixed_chunk_linear(source.feature_matrix(), self.coefficient)

    def receipt_payload(self) -> dict[str, object]:
        return {
            "absolute_ridge": self.absolute_ridge,
            "atlas_sha256": self.atlas_sha256,
            "coefficient_sha256": _tensor_sha256(self.coefficient),
            "config": self.config.payload(),
            "fit_catalog_sha256": self.fit_catalog_sha256,
            "fit_panel_binding_sha256": self.fit_panel_binding_sha256,
            "fit_role": self.fit_role,
            "fit_row_catalog_sha256": self.fit_row_catalog_sha256,
            "fit_design_sha256": self.fit_design_sha256,
            "fit_target_sha256": self.fit_target_sha256,
            "fit_row_weights_sha256": self.fit_row_weights_sha256,
            "schema": DIRECTED_CHART_TRANSPORT_SCHEMA,
            "source_mode_id": self.source_mode_id,
            "split_receipt_sha256": self.split_receipt_sha256,
            "solver_id": self.solver_id,
            "target_mode_id": self.target_mode_id,
            "transport_sha256": self.transport_sha256,
        }


def _transport_sha256(transport: DirectedChartTransport) -> str:
    return _digest(
        _TRANSPORT_DOMAIN,
        {
            "absolute_ridge": transport.absolute_ridge,
            "atlas_sha256": transport.atlas_sha256,
            "coefficient_sha256": _tensor_sha256(transport.coefficient),
            "config": transport.config.payload(),
            "fit_catalog_sha256": transport.fit_catalog_sha256,
            "fit_panel_binding_sha256": transport.fit_panel_binding_sha256,
            "fit_role": transport.fit_role,
            "fit_row_catalog_sha256": transport.fit_row_catalog_sha256,
            "fit_design_sha256": transport.fit_design_sha256,
            "fit_target_sha256": transport.fit_target_sha256,
            "fit_row_weights_sha256": transport.fit_row_weights_sha256,
            "source_mode_id": transport.source_mode_id,
            "split_receipt_sha256": transport.split_receipt_sha256,
            "solver_id": transport.solver_id,
            "target_mode_id": transport.target_mode_id,
        },
    )


def fit_directed_chart_transport(
    atlas: SharedChartAtlas,
    source: AppliedChartRows,
    target: AppliedChartRows,
    *,
    row_weights: Tensor,
) -> DirectedChartTransport:
    if source.batch.role != FIT_ROLE or target.batch.role != FIT_ROLE:
        raise ValueError("directed transports may fit only on A_chart_transport_fit")
    if source.batch.row_keys != target.batch.row_keys:
        raise ValueError("transport fit source/target rows must align exactly")
    if (
        source.batch.panel_binding_sha256
        != atlas.fit_panel_binding_sha256
        or target.batch.panel_binding_sha256
        != atlas.fit_panel_binding_sha256
        or source.batch.row_catalog_sha256 != atlas.fit_row_catalog_sha256
        or target.batch.row_catalog_sha256 != atlas.fit_row_catalog_sha256
    ):
        raise ValueError(
            "transport fit rows must be the exact panel/catalog used by atlas"
        )
    if (
        source.batch.mode_id not in atlas.fit_mode_ids
        or target.batch.mode_id not in atlas.fit_mode_ids
    ):
        raise ValueError("transport endpoints are absent from atlas fit modes")
    if source.atlas_sha256 != target.atlas_sha256:
        raise ValueError("transport endpoints must use one shared atlas")
    if source.atlas_sha256 != atlas.atlas_sha256:
        raise ValueError("transport endpoints do not use the supplied atlas")
    expected_balanced = _canonical_fit_balanced_row_weights(atlas, target.batch)
    supplied_balanced = row_weights.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    if not torch.equal(supplied_balanced, expected_balanced):
        raise ValueError(
            "transport row_weights do not equal the frozen balanced formula"
        )
    weights = target_fisher_balanced_transport_row_weights(
        atlas,
        target,
        balanced_row_weights=expected_balanced,
    )
    design = source.feature_matrix()
    target_values = target.batch.generalized_contributions
    gram = design.T @ (weights[:, None] * design)
    scale = float(torch.trace(gram) / gram.shape[0])
    absolute_ridge_value = atlas.config.transport_relative_ridge * max(
        scale, atlas.config.ridge_scale_floor
    )
    identity = torch.eye(gram.shape[0], dtype=torch.float64)
    right = design.T @ (weights[:, None] * target_values)
    coefficient = torch.linalg.solve(gram + absolute_ridge_value * identity, right)
    fit_catalog = _digest(
        _FIT_CATALOG_DOMAIN,
        {
            "atlas_sha256": source.atlas_sha256,
            "fit_panel_binding_sha256": source.batch.panel_binding_sha256,
            "fit_row_catalog_sha256": source.batch.row_catalog_sha256,
        },
    )
    split = _digest(
        _SPLIT_DOMAIN,
        {"fit_catalog_sha256": fit_catalog, "fit_role": FIT_ROLE},
    )
    values = {
        "config": atlas.config,
        "source_mode_id": source.batch.mode_id,
        "target_mode_id": target.batch.mode_id,
        "atlas_sha256": source.atlas_sha256,
        "fit_role": FIT_ROLE,
        "fit_panel_binding_sha256": source.batch.panel_binding_sha256,
        "fit_row_catalog_sha256": source.batch.row_catalog_sha256,
        "fit_catalog_sha256": fit_catalog,
        "split_receipt_sha256": split,
        "fit_design_sha256": _tensor_sha256(design),
        "fit_target_sha256": _tensor_sha256(target_values),
        "fit_row_weights_sha256": _tensor_sha256(weights),
        "solver_id": "balanced_float64_primal_ridge_solve_v1",
        "absolute_ridge": absolute_ridge_value,
        "coefficient": coefficient,
    }
    provisional = object.__new__(DirectedChartTransport)
    for field, value in values.items():
        object.__setattr__(provisional, field, value)
    object.__setattr__(provisional, "transport_sha256", "0" * 64)
    digest = _transport_sha256(provisional)
    return DirectedChartTransport(
        **values,
        transport_sha256=digest,
    )


def _canonical_fit_balanced_row_weights(
    atlas: SharedChartAtlas,
    batch: SharedChartRowBatch,
) -> Tensor:
    keys = batch.row_keys
    families = tuple(sorted({key.family_id for key in keys}))
    examples_by_family = {
        family: tuple(
            sorted(
                {
                    key.example_id
                    for key in keys
                    if key.family_id == family
                }
            )
        )
        for family in families
    }
    row_counts = {
        (family, example): sum(
            key.family_id == family and key.example_id == example
            for key in keys
        )
        for family in families
        for example in examples_by_family[family]
    }
    result = torch.empty(len(keys), dtype=torch.float64)
    for index, key in enumerate(keys):
        result[index] = 1.0 / (
            len(atlas.fit_mode_ids)
            * len(families)
            * len(examples_by_family[key.family_id])
            * row_counts[(key.family_id, key.example_id)]
        )
    return result


def target_fisher_balanced_transport_row_weights(
    atlas: SharedChartAtlas,
    target: AppliedChartRows,
    *,
    balanced_row_weights: Tensor,
) -> Tensor:
    """Balanced WLS weights times floored target Fisher energy, normalized."""

    weights = (
        balanced_row_weights.detach()
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .contiguous()
    )
    if (
        tuple(weights.shape) != (len(target.batch.row_keys),)
        or not bool(torch.isfinite(weights).all())
        or bool((weights <= 0).any())
    ):
        raise ValueError(
            "balanced transport row weights must be finite positive and aligned"
        )
    if target.atlas_sha256 != atlas.atlas_sha256:
        raise ValueError("target rows do not use the supplied shared atlas")
    fisher = torch.clamp(
        target.weighted_fisher_energy,
        min=atlas.config.transport_fisher_energy_floor,
    )
    result = weights * fisher
    total = result.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("target-Fisher transport row weights degenerated")
    return result / total


@dataclass(frozen=True, slots=True)
class RowShuffleCatalog:
    role: str
    row_catalog_sha256: str
    seed: int
    permutation: Tensor
    catalog_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.row_catalog_sha256, label="row_catalog_sha256")
        _require_sha256(self.catalog_sha256, label="catalog_sha256")
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**64:
            raise ValueError("row-shuffle seed must be unsigned u64")
        permutation = self.permutation.detach().to(device="cpu", dtype=torch.int64).clone().contiguous()
        if permutation.ndim != 1 or not permutation.numel() or sorted(permutation.tolist()) != list(range(permutation.numel())):
            raise ValueError("row-shuffle permutation must be a complete permutation")
        object.__setattr__(self, "permutation", permutation)


def build_within_example_row_shuffle(
    row_keys: Sequence[ChartRowKey],
    *,
    role: str,
    seed: int = ROW_SHUFFLE_SEED,
) -> RowShuffleCatalog:
    keys = tuple(row_keys)
    catalog_sha = row_catalog_sha256(keys, role=role)
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise ValueError("row-shuffle seed must be unsigned u64")
    permutation = list(range(len(keys)))
    payload_rows: list[dict[str, object]] = []
    example_groups = sorted({(key.family_id, key.example_id) for key in keys})
    for family_id, example_id in example_groups:
        positions = [
            index
            for index, key in enumerate(keys)
            if key.family_id == family_id and key.example_id == example_id
        ]
        if len(positions) < 2:
            raise ValueError(
                "row shuffle requires at least two rows per family/example"
            )
        role_bytes = role.encode("utf-8")
        family_bytes = family_id.encode("utf-8")
        example_bytes = example_id.encode("utf-8")
        digest = hashlib.sha256(
            ROW_SHUFFLE_DOMAIN
            + seed.to_bytes(8, "big")
            + len(role_bytes).to_bytes(4, "big")
            + role_bytes
            + len(family_bytes).to_bytes(4, "big")
            + family_bytes
            + len(example_bytes).to_bytes(4, "big")
            + example_bytes
        ).digest()
        offset = 1 + int.from_bytes(digest[:8], "big") % (len(positions) - 1)
        for ordinal, target_position in enumerate(positions):
            source_position = positions[(ordinal + offset) % len(positions)]
            permutation[target_position] = source_position
            payload_rows.append(
                {
                    "example_id": example_id,
                    "family_id": family_id,
                    "permuted_source_row": keys[source_position].payload(),
                    "target_row": keys[target_position].payload(),
                }
            )
    payload = {
        "role": role,
        "row_catalog_sha256": catalog_sha,
        "rows": payload_rows,
        "seed": seed,
    }
    return RowShuffleCatalog(
        role=role,
        row_catalog_sha256=catalog_sha,
        seed=seed,
        permutation=torch.tensor(permutation, dtype=torch.int64),
        catalog_sha256=_digest(ROW_SHUFFLE_CATALOG_DOMAIN, payload),
    )


def build_within_family_row_shuffle(
    row_keys: Sequence[ChartRowKey],
    *,
    role: str,
    seed: int = ROW_SHUFFLE_SEED,
) -> RowShuffleCatalog:
    """Compatibility spelling for the frozen per-example shuffle control."""

    return build_within_example_row_shuffle(
        row_keys,
        role=role,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class DirectedExampleEnergy:
    family_id: str
    example_id: str
    row_count: int
    weighted_transport_residual_energy: float
    weighted_predicted_fisher_energy: float
    weighted_target_fisher_energy: float
    weighted_predicted_target_cross_inner_product: float
    weighted_zero_prediction_residual_energy: float
    weighted_row_shuffled_transport_residual_energy: float
    real_gain_over_zero: float | None
    real_gain_over_row_shuffle: float | None


@dataclass(frozen=True, slots=True)
class DirectedTransportEvaluation:
    comparison_abi_name: str
    source_mode_id: str
    target_mode_id: str
    evaluation_role: str
    evaluation_panel_binding_sha256: str
    evaluation_row_catalog_sha256: str
    row_shuffle_catalog_sha256: str
    split_receipt_sha256: str
    transport_sha256: str
    prediction_backend_id: str
    prediction_row_chunk_size: int
    prediction_backend_receipt_sha256: str
    examples: tuple[DirectedExampleEnergy, ...]


def transport_coefficient_storage_bytes(
    config: SharedChartNumericsConfig,
    *,
    mode_count: int,
) -> int:
    """Exact unique float64 coefficient bytes for all ordered mode pairs."""

    if not isinstance(config, SharedChartNumericsConfig):
        raise TypeError("config must be SharedChartNumericsConfig")
    if type(mode_count) is not int or mode_count < 2:
        raise ValueError("mode_count must be at least two")
    return (
        mode_count
        * (mode_count - 1)
        * config.source_feature_width
        * config.hidden_width
        * 8
    )


def _pair_energy_rows(
    atlas: SharedChartAtlas,
    predicted: Tensor,
    predicted_memberships: Tensor,
    target: Tensor,
    target_memberships: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    predicted_energy = torch.zeros(predicted.shape[0], dtype=torch.float64)
    target_energy = torch.zeros_like(predicted_energy)
    cross = torch.zeros_like(predicted_energy)
    residual = torch.zeros_like(predicted_energy)
    for chart in range(atlas.config.chart_count):
        basis = atlas.tangent_bases[chart]
        variance = atlas.tangent_variance[chart]
        alpha = atlas.orthogonal_alpha[chart]
        pp = predicted @ basis
        tp = target @ basis
        po = predicted - pp @ basis.T
        to = target - tp @ basis.T
        p_metric = (pp.square() / variance).sum(dim=1) + alpha * po.square().sum(dim=1)
        t_metric = (tp.square() / variance).sum(dim=1) + alpha * to.square().sum(dim=1)
        cross_metric = (pp * tp / variance).sum(dim=1) + alpha * (po * to).sum(dim=1)
        qp = predicted_memberships[:, chart]
        qt = target_memberships[:, chart]
        weight_cross = torch.sqrt(qp * qt)
        predicted_energy += qp * p_metric
        target_energy += qt * t_metric
        cross += weight_cross * cross_metric
        residual += qp * p_metric + qt * t_metric - 2.0 * weight_cross * cross_metric
    return predicted_energy, target_energy, cross, residual.clamp_min(0.0)


def _rowwise_linear_reference(design: Tensor, coefficient: Tensor) -> Tensor:
    """Fixture-only scalar-row oracle for the production ``torch.mm`` path."""

    return torch.stack(
        [torch.mv(coefficient.T, design[row]) for row in range(design.shape[0])]
    )


def prediction_backend_receipt() -> dict[str, object]:
    """Authenticate the fixed production prediction kernel and environment."""

    payload = {
        "backend_id": PREDICTION_BACKEND_ID,
        "device": "cpu",
        "dtype": "float64",
        "row_chunk_size": PREDICTION_ROW_CHUNK_SIZE,
        "torch_build_config_sha256": hashlib.sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "torch_version": str(torch.__version__),
    }
    return {
        **payload,
        "receipt_sha256": _digest(_PREDICTION_BACKEND_DOMAIN, payload),
    }


def _fixed_chunk_linear(design: Tensor, coefficient: Tensor) -> Tensor:
    """Apply one fixed-size vectorized GEMM schedule on CPU float64 rows."""

    if (
        design.ndim != 2
        or coefficient.ndim != 2
        or design.shape[1] != coefficient.shape[0]
        or design.device.type != "cpu"
        or coefficient.device.type != "cpu"
        or design.dtype != torch.float64
        or coefficient.dtype != torch.float64
    ):
        raise ValueError(
            "fixed chart prediction requires aligned CPU float64 matrices"
        )
    output = torch.empty(
        (design.shape[0], coefficient.shape[1]),
        dtype=torch.float64,
    )
    for start in range(0, design.shape[0], PREDICTION_ROW_CHUNK_SIZE):
        end = min(design.shape[0], start + PREDICTION_ROW_CHUNK_SIZE)
        output[start:end] = torch.mm(design[start:end], coefficient)
    return output


def evaluate_directed_chart_transport(
    atlas: SharedChartAtlas,
    transport: DirectedChartTransport,
    source: AppliedChartRows,
    target: AppliedChartRows,
    *,
    row_shuffle: RowShuffleCatalog,
    chunk_size: int = 256,
) -> DirectedTransportEvaluation:
    if transport.source_mode_id != source.batch.mode_id:
        raise ValueError("transport source endpoint does not match source rows")
    if transport.target_mode_id != target.batch.mode_id:
        raise ValueError("transport target endpoint does not match target rows")
    if (
        transport.atlas_sha256 != atlas.atlas_sha256
        or source.atlas_sha256 != atlas.atlas_sha256
        or target.atlas_sha256 != atlas.atlas_sha256
        or transport.config != atlas.config
    ):
        raise ValueError("transport evaluation atlas/config lineage is invalid")
    if source.batch.role == FIT_ROLE or target.batch.role == FIT_ROLE:
        raise ValueError("transport evaluation must be disjoint from fit role A")
    if source.batch.role != target.batch.role or source.batch.panel_binding_sha256 != target.batch.panel_binding_sha256 or source.batch.row_keys != target.batch.row_keys:
        raise ValueError("evaluation source and target must share aligned heldout rows")
    if source.batch.panel_binding_sha256 == transport.fit_panel_binding_sha256:
        raise ValueError("transport evaluation panel must differ from fit panel")
    if row_shuffle.role != source.batch.role or row_shuffle.row_catalog_sha256 != source.batch.row_catalog_sha256:
        raise ValueError("row-shuffle catalog does not match evaluation rows")
    if row_shuffle.seed != ROW_SHUFFLE_SEED:
        raise ValueError("row-shuffle seed does not match the frozen protocol")
    expected_shuffle = build_within_example_row_shuffle(
        source.batch.row_keys,
        role=source.batch.role,
        seed=row_shuffle.seed,
    )
    if (
        row_shuffle.catalog_sha256 != expected_shuffle.catalog_sha256
        or not torch.equal(row_shuffle.permutation, expected_shuffle.permutation)
    ):
        raise ValueError(
            "row-shuffle catalog is not the deterministic within-example control"
        )
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    predicted_all = transport.predict(source)
    if not bool(torch.isfinite(predicted_all).all()):
        raise ValueError("transport prediction produced non-finite energy inputs")
    target_values = target.batch.generalized_contributions
    permutation = row_shuffle.permutation
    fields: dict[tuple[str, str], list[list[float]]] = {}
    for start in range(0, len(source.batch.row_keys), chunk_size):
        end = min(len(source.batch.row_keys), start + chunk_size)
        predicted = predicted_all[start:end]
        shuffled_indices = permutation[start:end]
        shuffled_predicted = predicted_all[shuffled_indices]
        p_energy, t_energy, cross, residual = _pair_energy_rows(
            atlas,
            predicted,
            source.memberships[start:end],
            target_values[start:end],
            target.memberships[start:end],
        )
        _, _, _, shuffled_residual = _pair_energy_rows(
            atlas,
            shuffled_predicted,
            source.memberships[shuffled_indices],
            target_values[start:end],
            target.memberships[start:end],
        )
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                p_energy,
                t_energy,
                cross,
                residual,
                shuffled_residual,
            )
        ):
            raise ValueError("transport evaluation produced non-finite energy")
        for local, row_key in enumerate(source.batch.row_keys[start:end]):
            bucket = fields.setdefault((row_key.family_id, row_key.example_id), [[], [], [], [], [], []])
            bucket[0].append(float(residual[local]))
            bucket[1].append(float(p_energy[local]))
            bucket[2].append(float(t_energy[local]))
            bucket[3].append(float(cross[local]))
            bucket[4].append(float(t_energy[local]))
            bucket[5].append(float(shuffled_residual[local]))
    examples = []
    for family_id, example_id in sorted(fields):
        values = fields[(family_id, example_id)]
        residual_energy = math.fsum(values[0])
        zero_energy = math.fsum(values[4])
        shuffle_energy = math.fsum(values[5])
        examples.append(
            DirectedExampleEnergy(
                family_id=family_id,
                example_id=example_id,
                row_count=len(values[0]),
                weighted_transport_residual_energy=residual_energy,
                weighted_predicted_fisher_energy=math.fsum(values[1]),
                weighted_target_fisher_energy=math.fsum(values[2]),
                weighted_predicted_target_cross_inner_product=math.fsum(values[3]),
                weighted_zero_prediction_residual_energy=zero_energy,
                weighted_row_shuffled_transport_residual_energy=shuffle_energy,
                real_gain_over_zero=(
                    1.0 - residual_energy / zero_energy
                    if zero_energy > 0.0
                    else None
                ),
                real_gain_over_row_shuffle=(
                    1.0 - residual_energy / shuffle_energy
                    if shuffle_energy > 0.0
                    else None
                ),
            )
        )
    return DirectedTransportEvaluation(
        comparison_abi_name=COMPARISON_ABI_NAME,
        source_mode_id=source.batch.mode_id,
        target_mode_id=target.batch.mode_id,
        evaluation_role=source.batch.role,
        evaluation_panel_binding_sha256=source.batch.panel_binding_sha256,
        evaluation_row_catalog_sha256=source.batch.row_catalog_sha256,
        row_shuffle_catalog_sha256=row_shuffle.catalog_sha256,
        split_receipt_sha256=_digest(
            _SPLIT_DOMAIN,
            {
                "evaluation_panel_binding_sha256": (
                    source.batch.panel_binding_sha256
                ),
                "evaluation_role": source.batch.role,
                "evaluation_row_catalog_sha256": (
                    source.batch.row_catalog_sha256
                ),
                "fit_catalog_sha256": transport.fit_catalog_sha256,
                "fit_role": FIT_ROLE,
            },
        ),
        transport_sha256=transport.transport_sha256,
        prediction_backend_id=PREDICTION_BACKEND_ID,
        prediction_row_chunk_size=PREDICTION_ROW_CHUNK_SIZE,
        prediction_backend_receipt_sha256=prediction_backend_receipt()[
            "receipt_sha256"
        ],
        examples=tuple(examples),
    )


def _save_bundle(
    destination: Path | str,
    *,
    kind: str,
    metadata: Mapping[str, object],
    tensors: Mapping[str, Tensor],
) -> None:
    ordered = []
    payload = bytearray()
    for name in sorted(tensors):
        value = tensors[name].detach().to(device="cpu", dtype=torch.float64).contiguous()
        raw = _tensor_bytes(value)
        ordered.append(
            {
                "byte_count": len(raw),
                "name": name,
                "offset": len(payload),
                "sha256": _tensor_sha256(value),
                "shape": list(value.shape),
            }
        )
        payload.extend(raw)
    header = {
        "kind": kind,
        "metadata": dict(metadata),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "schema": _BUNDLE_SCHEMA,
        "tensors": ordered,
    }
    header_bytes = _canonical_json_bytes(header)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_BUNDLE_MAGIC)
            handle.write(struct.pack(">Q", len(header_bytes)))
            handle.write(header_bytes)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        os.unlink(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        observed_metadata, observed_tensors = _load_bundle(
            path,
            expected_kind=kind,
        )
        if observed_metadata != dict(metadata) or set(observed_tensors) != set(tensors):
            raise RuntimeError("strict chart tensor bundle reload drifted")
        for name, supplied in tensors.items():
            expected = supplied.detach().to(
                device="cpu", dtype=torch.float64
            ).contiguous()
            if (
                _tensor_sha256(observed_tensors[name])
                != _tensor_sha256(expected)
                or not torch.equal(observed_tensors[name], expected)
            ):
                raise RuntimeError(
                    "strict chart tensor bundle tensor reload drifted"
                )
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        if published:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            try:
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        raise


def _load_bundle(source: Path | str, *, expected_kind: str) -> tuple[dict[str, object], dict[str, Tensor]]:
    path = Path(source)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError("strict chart tensor bundle must be a regular file")
    if before.st_size > _MAX_BUNDLE_BYTES:
        raise ValueError("strict chart tensor bundle exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ValueError("strict chart tensor bundle changed during open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("strict chart tensor bundle truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise ValueError("strict chart tensor bundle changed during read")
    if not raw.startswith(_BUNDLE_MAGIC):
        raise ValueError("strict chart tensor bundle is invalid")
    position = len(_BUNDLE_MAGIC)
    if len(raw) < position + 8:
        raise ValueError("strict chart tensor bundle is truncated")
    header_length = struct.unpack(">Q", raw[position : position + 8])[0]
    position += 8
    if header_length > len(raw) - position:
        raise ValueError("strict chart tensor bundle header is truncated")
    header_raw = raw[position : position + header_length]
    header = _strict_json_loads(header_raw)
    position += header_length
    payload = raw[position:]
    if not isinstance(header, Mapping):
        raise ValueError("strict chart tensor bundle header must be an object")
    if (
        set(header)
        != {"kind", "metadata", "payload_sha256", "schema", "tensors"}
        or type(header["schema"]) is not str
        or type(header["kind"]) is not str
        or type(header["payload_sha256"]) is not str
        or header["schema"] != _BUNDLE_SCHEMA
        or header["kind"] != expected_kind
    ):
        raise ValueError("strict chart tensor bundle header contract drifted")
    _require_sha256(header["payload_sha256"], label="payload_sha256")
    if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise ValueError("strict chart tensor bundle payload digest drifted")
    if not isinstance(header["metadata"], Mapping) or not isinstance(header["tensors"], list):
        raise ValueError("strict chart tensor bundle catalogs are invalid")
    tensors: dict[str, Tensor] = {}
    expected_offset = 0
    previous_name: str | None = None
    for entry in header["tensors"]:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {"byte_count", "name", "offset", "sha256", "shape"}
            or type(entry["name"]) is not str
            or not entry["name"]
            or type(entry["sha256"]) is not str
            or type(entry["offset"]) is not int
            or entry["offset"] < 0
            or type(entry["byte_count"]) is not int
            or entry["byte_count"] < 0
            or not isinstance(entry["shape"], list)
            or not entry["shape"]
            or any(type(size) is not int or size <= 0 for size in entry["shape"])
            or entry["offset"] != expected_offset
        ):
            raise ValueError("strict chart tensor bundle tensor catalog drifted")
        _require_sha256(entry["sha256"], label="tensor sha256")
        if previous_name is not None and entry["name"] <= previous_name:
            raise ValueError("strict chart tensor names are not canonical")
        if entry["name"] in tensors:
            raise ValueError("strict chart tensor bundle repeats a tensor name")
        end = expected_offset + entry["byte_count"]
        if end > len(payload):
            raise ValueError("strict chart tensor bundle tensor exceeds payload")
        chunk = payload[expected_offset:end]
        shape = tuple(entry["shape"])
        if math.prod(shape) * 8 != len(chunk):
            raise ValueError("strict chart tensor bundle tensor shape drifted")
        tensor = torch.frombuffer(bytearray(chunk), dtype=torch.float64).reshape(shape).clone()
        if _tensor_sha256(tensor) != entry["sha256"]:
            raise ValueError("strict chart tensor bundle tensor digest drifted")
        tensors[entry["name"]] = tensor
        expected_offset = end
        previous_name = entry["name"]
    if expected_offset != len(payload):
        raise ValueError("strict chart tensor bundle has trailing bytes")
    return dict(header["metadata"]), tensors


def save_shared_chart_atlas(atlas: SharedChartAtlas, destination: Path | str) -> None:
    _save_bundle(
        destination,
        kind=SHARED_CHART_ATLAS_SCHEMA,
        metadata=atlas.receipt_payload(),
        tensors={
            "centers": atlas.centers,
            "hard_occupancy_weight": atlas.hard_occupancy_weight,
            "orthogonal_alpha": atlas.orthogonal_alpha,
            "radius_squared": atlas.radius_squared,
            "tangent_bases": atlas.tangent_bases,
            "tangent_variance": atlas.tangent_variance,
        },
    )


def load_shared_chart_atlas(source: Path | str) -> SharedChartAtlas:
    metadata, tensors = _load_bundle(source, expected_kind=SHARED_CHART_ATLAS_SCHEMA)
    expected_metadata = {
        "atlas_sha256",
        "config",
        "fit_balanced_row_weight_catalog_sha256",
        "fit_context_catalog_sha256",
        "fit_mode_ids",
        "fit_panel_binding_sha256",
        "fit_role",
        "fit_row_catalog_sha256",
        "fit_solver_receipt_sha256",
        "hard_occupancy_count",
        "hard_occupancy_weight",
        "schema",
        "tensor_sha256s",
    }
    expected_tensors = {
        "centers",
        "hard_occupancy_weight",
        "orthogonal_alpha",
        "radius_squared",
        "tangent_bases",
        "tangent_variance",
    }
    if set(metadata) != expected_metadata or set(tensors) != expected_tensors:
        raise ValueError("shared chart atlas bundle catalog drifted")
    if metadata["schema"] != SHARED_CHART_ATLAS_SCHEMA:
        raise ValueError("shared chart atlas metadata schema drifted")
    counts = metadata["hard_occupancy_count"]
    if (
        not isinstance(counts, list)
        or any(type(value) is not int or value <= 0 for value in counts)
    ):
        raise ValueError("shared chart hard occupancy counts are invalid")
    config = SharedChartNumericsConfig(**metadata["config"])
    atlas = SharedChartAtlas(
        config=config,
        fit_role=metadata["fit_role"],
        fit_panel_binding_sha256=metadata["fit_panel_binding_sha256"],
        fit_row_catalog_sha256=metadata["fit_row_catalog_sha256"],
        fit_mode_ids=tuple(metadata["fit_mode_ids"]),
        fit_context_catalog_sha256=metadata[
            "fit_context_catalog_sha256"
        ],
        fit_balanced_row_weight_catalog_sha256=metadata[
            "fit_balanced_row_weight_catalog_sha256"
        ],
        fit_solver_receipt_sha256=metadata["fit_solver_receipt_sha256"],
        centers=tensors["centers"],
        radius_squared=tensors["radius_squared"],
        tangent_bases=tensors["tangent_bases"],
        tangent_variance=tensors["tangent_variance"],
        orthogonal_alpha=tensors["orthogonal_alpha"],
        hard_occupancy_count=torch.tensor(counts, dtype=torch.int64),
        hard_occupancy_weight=tensors["hard_occupancy_weight"],
        atlas_sha256=metadata["atlas_sha256"],
    )
    if metadata["tensor_sha256s"] != _atlas_tensor_sha256s(atlas):
        raise ValueError("shared chart atlas tensor receipt drifted")
    if metadata["hard_occupancy_weight"] != atlas.hard_occupancy_weight.tolist():
        raise ValueError("shared chart atlas occupancy receipt drifted")
    return atlas


def save_directed_chart_transport(transport: DirectedChartTransport, destination: Path | str) -> None:
    _save_bundle(
        destination,
        kind=DIRECTED_CHART_TRANSPORT_SCHEMA,
        metadata=transport.receipt_payload(),
        tensors={"coefficient": transport.coefficient},
    )


def load_directed_chart_transport(source: Path | str) -> DirectedChartTransport:
    metadata, tensors = _load_bundle(source, expected_kind=DIRECTED_CHART_TRANSPORT_SCHEMA)
    expected_metadata = {
        "absolute_ridge",
        "atlas_sha256",
        "coefficient_sha256",
        "config",
        "fit_catalog_sha256",
        "fit_design_sha256",
        "fit_panel_binding_sha256",
        "fit_role",
        "fit_row_catalog_sha256",
        "fit_row_weights_sha256",
        "fit_target_sha256",
        "schema",
        "solver_id",
        "source_mode_id",
        "split_receipt_sha256",
        "target_mode_id",
        "transport_sha256",
    }
    if set(metadata) != expected_metadata or set(tensors) != {"coefficient"}:
        raise ValueError("directed chart transport bundle catalog drifted")
    if metadata["schema"] != DIRECTED_CHART_TRANSPORT_SCHEMA:
        raise ValueError("directed chart transport metadata schema drifted")
    transport = DirectedChartTransport(
        config=SharedChartNumericsConfig(**metadata["config"]),
        source_mode_id=metadata["source_mode_id"],
        target_mode_id=metadata["target_mode_id"],
        atlas_sha256=metadata["atlas_sha256"],
        fit_role=metadata["fit_role"],
        fit_panel_binding_sha256=metadata["fit_panel_binding_sha256"],
        fit_row_catalog_sha256=metadata["fit_row_catalog_sha256"],
        fit_catalog_sha256=metadata["fit_catalog_sha256"],
        split_receipt_sha256=metadata["split_receipt_sha256"],
        fit_design_sha256=metadata["fit_design_sha256"],
        fit_target_sha256=metadata["fit_target_sha256"],
        fit_row_weights_sha256=metadata["fit_row_weights_sha256"],
        solver_id=metadata["solver_id"],
        absolute_ridge=metadata["absolute_ridge"],
        coefficient=tensors["coefficient"],
        transport_sha256=metadata["transport_sha256"],
    )
    if metadata["coefficient_sha256"] != _tensor_sha256(transport.coefficient):
        raise ValueError("directed chart transport coefficient receipt drifted")
    return transport


__all__ = [
    "COMPARISON_ABI_NAME",
    "DIRECTED_CHART_TRANSPORT_SCHEMA",
    "FIT_ROLE",
    "ROW_SHUFFLE_CATALOG_DOMAIN",
    "ROW_SHUFFLE_DOMAIN",
    "ROW_SHUFFLE_SEED",
    "SHARED_CHART_ATLAS_SCHEMA",
    "AppliedChartRows",
    "ChartRowKey",
    "DirectedChartTransport",
    "DirectedExampleEnergy",
    "DirectedTransportEvaluation",
    "RowShuffleCatalog",
    "SharedChartAtlas",
    "SharedChartNumericsConfig",
    "SharedChartRowBatch",
    "apply_shared_chart_atlas",
    "balanced_atlas_row_weights",
    "build_within_family_row_shuffle",
    "build_within_example_row_shuffle",
    "evaluate_directed_chart_transport",
    "fit_directed_chart_transport",
    "fit_shared_chart_atlas",
    "load_directed_chart_transport",
    "load_shared_chart_atlas",
    "row_catalog_sha256",
    "save_directed_chart_transport",
    "save_shared_chart_atlas",
    "target_fisher_balanced_transport_row_weights",
    "transport_coefficient_storage_bytes",
]
