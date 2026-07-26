"""Model-agnostic conditional routing for Fisher modal coordinates.

This module separates three questions that are easy to conflate:

* Which modal coordinates does each calibration token need?
* Can those need patterns be summarized by a small set of route masks?
* Can a router choose the route from the block *input*, without observing the
  native block output or a Fisher profile at inference time?

``cluster_fisher_need_profiles`` answers the first two questions with
deterministic, L1-normalized k-means.  ``build_conditional_mode_table`` turns
the resulting route centroids into inspectable mode masks with explicit,
possibly route-specific budgets.  ``fit_pointwise_causal_router`` fits a
ridge-regularized linear classifier using only each token's current block
input.  Since prediction is pointwise over the leading dimensions, appending
future tokens cannot change an earlier routing decision.

The reference implementation reports an *active-mode ratio*.  This is useful
logical accounting, but it is not a FLOP, latency, or sparse-kernel claim.
Actual savings require an executor that avoids computing inactive modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


_TABLE_KIND = "fisher_graph.conditional_mode_table"
_ROUTER_KIND = "fisher_graph.pointwise_causal_router"
_PLAN_KIND = "fisher_graph.conditional_modal_routing_plan"
_TOTAL_NEED_TEACHER_KIND = "fisher_graph.total_need_route_teacher"
_FORMAT_VERSION = 1


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_float(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    if positive and converted <= 0:
        raise ValueError(f"{label} must be positive")
    return converted


def _finite_float64_rows(
    values: Tensor,
    *,
    label: str,
    nonnegative: bool = False,
) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if values.ndim < 2 or values.shape[-1] == 0:
        raise ValueError(
            f"{label} must have shape [..., nonzero features]"
        )
    if not values.is_floating_point():
        raise ValueError(f"{label} must be floating point")
    converted = values.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(converted).all():
        raise ValueError(f"{label} must be finite")
    if nonnegative and (converted < 0).any():
        raise ValueError(f"{label} must be nonnegative")
    return converted.reshape(-1, converted.shape[-1])


def _selection_mask(
    leading_shape: torch.Size,
    valid_mask: Tensor | None,
    *,
    label: str,
) -> Tensor:
    observations = math.prod(leading_shape)
    if valid_mask is None:
        return torch.ones(observations, dtype=torch.bool)
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError(f"{label} must be a boolean Tensor")
    if valid_mask.shape != leading_shape:
        raise ValueError(
            f"{label} must match the values' leading dimensions"
        )
    return valid_mask.detach().to(device="cpu").reshape(-1)


def _selected_float64_rows(
    values: Tensor,
    *,
    label: str,
    valid_mask: Tensor | None,
    nonnegative: bool = False,
) -> tuple[Tensor, Tensor]:
    rows = _finite_float64_rows(
        values,
        label=label,
        nonnegative=nonnegative,
    )
    mask = _selection_mask(
        values.shape[:-1],
        valid_mask,
        label="valid_mask",
    )
    selected = rows[mask]
    if selected.shape[0] == 0:
        raise ValueError(f"{label} has no selected token rows")
    return selected, mask.nonzero(as_tuple=False).flatten()


def _normalized_need_rows(rows: Tensor) -> Tensor:
    totals = rows.sum(dim=1, keepdim=True)
    return torch.where(totals > 0, rows / totals.clamp_min(1e-300), rows)


def linear_codec_fisher_damage_profiles(
    activation_deltas: Tensor,
    output_score_gradients: Tensor,
    *,
    encoder: Tensor,
    decoder: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Estimate per-mode first-order damage for a linear activation codec.

    A linear codec represents an activation delta ``delta`` with coordinates
    ``c = delta @ encoder`` and decodes them with
    ``c @ decoder.T``.  Suppressing mode ``j`` therefore changes the decoded
    activation by ``-c_j * decoder[:, j]``.  For output-score gradient ``g``,
    this function reports the corresponding squared first-order score change

    ``(c_j * (g @ decoder[:, j])) ** 2``.

    ``encoder`` and ``decoder`` both have shape
    ``[activation width, modes]``.  They need not be equal, orthogonal, or
    square, which makes this definition suitable for generalized and oblique
    codecs.  Without ``valid_mask``, the result preserves the input leading
    dimensions and has shape ``[..., modes]``.  With a boolean
    ``valid_mask`` matching those leading dimensions, selected rows are
    returned flattened in deterministic row-major order with shape
    ``[selected rows, modes]``.

    This is an independent-mode, first-order removal estimate.  It does not
    include interactions between simultaneously removed modes.
    """

    if not isinstance(activation_deltas, Tensor):
        raise TypeError("activation_deltas must be a Tensor")
    if not isinstance(output_score_gradients, Tensor):
        raise TypeError("output_score_gradients must be a Tensor")
    if (
        activation_deltas.shape != output_score_gradients.shape
        or activation_deltas.ndim < 2
        or activation_deltas.shape[-1] == 0
    ):
        raise ValueError(
            "activation deltas and score gradients must share shape "
            "[..., nonzero width]"
        )
    for label, value in (
        ("activation_deltas", activation_deltas),
        ("output_score_gradients", output_score_gradients),
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{label} must be a floating Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} must be finite")
    if (
        activation_deltas.device != output_score_gradients.device
        or activation_deltas.dtype != output_score_gradients.dtype
    ):
        raise ValueError(
            "activation deltas and score gradients must share dtype and device"
        )

    width = activation_deltas.shape[-1]
    if (
        encoder.ndim != 2
        or encoder.shape[0] != width
        or encoder.shape[1] == 0
    ):
        raise ValueError(
            "encoder must have shape [activation width, nonzero modes]"
        )
    if decoder.shape != encoder.shape:
        raise ValueError("decoder must have the same shape as encoder")

    compute_dtype = (
        torch.float32
        if activation_deltas.dtype in (torch.float16, torch.bfloat16)
        else activation_deltas.dtype
    )
    deltas = activation_deltas.to(dtype=compute_dtype)
    gradients = output_score_gradients.to(dtype=compute_dtype)
    encoding = encoder.to(device=deltas.device, dtype=compute_dtype)
    decoding = decoder.to(device=deltas.device, dtype=compute_dtype)
    coordinates = deltas @ encoding
    output_sensitivities = gradients @ decoding
    profiles = (coordinates * output_sensitivities).square()

    if valid_mask is None:
        return profiles
    selection = _selection_mask(
        activation_deltas.shape[:-1],
        valid_mask,
        label="valid_mask",
    )
    if not selection.any():
        raise ValueError("activation_deltas has no selected token rows")
    return profiles.reshape(-1, profiles.shape[-1])[
        selection.to(device=profiles.device)
    ]


def fisher_projection_damage_profiles(
    output_activations: Tensor,
    output_score_gradients: Tensor,
    *,
    center: Tensor,
    basis_vectors: Tensor,
) -> Tensor:
    """Compute per-token first-order damage estimates in modal coordinates.

    For activation displacement ``a`` and score gradient ``g``, mode vector
    ``v_j`` receives need

    ``((a - center) @ v_j * (g @ v_j)) ** 2``.

    The result is nonnegative and has shape ``[..., modes]``.  It estimates
    the squared first-order score change from suppressing each mode
    independently; it is not itself a full Hessian or interaction model.
    """

    if (
        not isinstance(output_activations, Tensor)
        or not isinstance(output_score_gradients, Tensor)
        or output_activations.shape != output_score_gradients.shape
        or output_activations.ndim < 2
        or output_activations.shape[-1] == 0
    ):
        raise ValueError(
            "output activations and score gradients must share shape "
            "[..., nonzero width]"
        )
    for label, value in (
        ("output_activations", output_activations),
        ("output_score_gradients", output_score_gradients),
        ("center", center),
        ("basis_vectors", basis_vectors),
    ):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{label} must be a floating Tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"{label} must be finite")
    width = output_activations.shape[-1]
    if center.shape != (width,):
        raise ValueError("center must match the activation width")
    if (
        basis_vectors.ndim != 2
        or basis_vectors.shape[0] != width
        or basis_vectors.shape[1] == 0
    ):
        raise ValueError(
            "basis_vectors must have shape [activation width, nonzero modes]"
        )
    if (
        output_activations.device != output_score_gradients.device
        or output_activations.dtype != output_score_gradients.dtype
    ):
        raise ValueError(
            "output activations and score gradients must share dtype and device"
        )
    compute_dtype = (
        torch.float32
        if output_activations.dtype in (torch.float16, torch.bfloat16)
        else output_activations.dtype
    )
    activations = output_activations.to(dtype=compute_dtype)
    gradients = output_score_gradients.to(dtype=compute_dtype)
    modal_vectors = basis_vectors.to(
        device=activations.device,
        dtype=compute_dtype,
    )
    modal_center = center.to(
        device=activations.device,
        dtype=compute_dtype,
    )
    return linear_codec_fisher_damage_profiles(
        activations - modal_center,
        gradients,
        encoder=modal_vectors,
        decoder=modal_vectors,
    )


def _stable_descending_indices(values: Tensor) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(values.numel()),
            key=lambda index: (-float(values[index].item()), index),
        )
    )


def _deterministic_maximum_row(
    scores: Tensor,
    profiles: Tensor,
    *,
    eligible: Tensor,
) -> int:
    eligible_indices = eligible.nonzero(as_tuple=False).flatten().tolist()
    if not eligible_indices:
        raise ValueError("no eligible row remains")
    maximum = max(float(scores[index].item()) for index in eligible_indices)
    candidates = [
        index
        for index in eligible_indices
        if float(scores[index].item()) == maximum
    ]
    return min(
        candidates,
        key=lambda index: (
            tuple(float(value) for value in profiles[index].tolist()),
            index,
        ),
    )


@dataclass(frozen=True, slots=True)
class FisherNeedClustering:
    """Deterministic route discovery over normalized per-token need profiles."""

    assignments: Tensor
    centroids: Tensor
    selected_flat_indices: Tensor
    route_counts: tuple[int, ...]
    iterations: int
    objective: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.assignments, Tensor)
            or self.assignments.device.type != "cpu"
            or self.assignments.dtype != torch.int64
            or self.assignments.ndim != 1
            or self.assignments.numel() == 0
        ):
            raise ValueError("assignments must be a nonempty CPU int64 vector")
        if (
            not isinstance(self.centroids, Tensor)
            or self.centroids.device.type != "cpu"
            or self.centroids.dtype != torch.float64
            or self.centroids.ndim != 2
            or self.centroids.shape[0] == 0
            or self.centroids.shape[1] == 0
            or not torch.isfinite(self.centroids).all()
            or (self.centroids < 0).any()
        ):
            raise ValueError(
                "centroids must be a finite nonnegative CPU float64 matrix"
            )
        routes = self.centroids.shape[0]
        if (
            type(self.route_counts) is not tuple
            or len(self.route_counts) != routes
            or any(type(count) is not int or count <= 0 for count in self.route_counts)
            or sum(self.route_counts) != self.assignments.numel()
        ):
            raise ValueError("route_counts must match nonempty assignments")
        if (
            int(self.assignments.min().item()) < 0
            or int(self.assignments.max().item()) >= routes
        ):
            raise ValueError("assignment exceeds the centroid route range")
        if tuple(torch.bincount(self.assignments, minlength=routes).tolist()) != (
            self.route_counts
        ):
            raise ValueError("route_counts disagree with assignments")
        if (
            not isinstance(self.selected_flat_indices, Tensor)
            or self.selected_flat_indices.device.type != "cpu"
            or self.selected_flat_indices.dtype != torch.int64
            or self.selected_flat_indices.shape != self.assignments.shape
            or (self.selected_flat_indices < 0).any()
        ):
            raise ValueError(
                "selected_flat_indices must align with assignments"
            )
        if self.selected_flat_indices.unique().numel() != self.assignments.numel():
            raise ValueError("selected_flat_indices must be unique")
        _require_positive_int(self.iterations, label="iterations")
        _require_finite_float(self.objective, label="objective")
        if self.objective < 0:
            raise ValueError("objective must be nonnegative")

        sums = self.centroids.sum(dim=1)
        if not torch.all((sums == 0) | torch.isclose(sums, torch.ones_like(sums))):
            raise ValueError("centroids must be zero or L1-normalized")
        object.__setattr__(
            self,
            "assignments",
            self.assignments.detach().clone(),
        )
        object.__setattr__(self, "centroids", self.centroids.detach().clone())
        object.__setattr__(
            self,
            "selected_flat_indices",
            self.selected_flat_indices.detach().clone(),
        )

    @property
    def routes(self) -> int:
        return self.centroids.shape[0]

    @property
    def modes(self) -> int:
        return self.centroids.shape[1]

    @property
    def observations(self) -> int:
        return self.assignments.numel()


def cluster_fisher_need_profiles(
    need_profiles: Tensor,
    *,
    route_count: int,
    valid_mask: Tensor | None = None,
    max_iterations: int = 100,
) -> FisherNeedClustering:
    """Cluster nonnegative per-token Fisher need profiles deterministically.

    Profiles are normalized to unit L1 mass before clustering, so routes
    describe *which* modes matter rather than merely separating tokens by
    gradient magnitude.  Zero-need rows remain zero.  Initialization uses
    deterministic farthest-first traversal and empty clusters are repaired
    deterministically.
    """

    routes = _require_positive_int(route_count, label="route_count")
    iteration_limit = _require_positive_int(
        max_iterations,
        label="max_iterations",
    )
    rows, selected_indices = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    if routes > rows.shape[0]:
        raise ValueError("route_count cannot exceed selected token rows")
    profiles = _normalized_need_rows(rows)
    if torch.unique(profiles, dim=0).shape[0] < routes:
        raise ValueError(
            "route_count cannot exceed distinct normalized need profiles"
        )

    chosen: list[int] = []
    eligible = torch.ones(rows.shape[0], dtype=torch.bool)
    first_scores = profiles.square().sum(dim=1)
    first = _deterministic_maximum_row(
        first_scores,
        profiles,
        eligible=eligible,
    )
    chosen.append(first)
    eligible[first] = False
    while len(chosen) < routes:
        centroids_so_far = profiles[torch.tensor(chosen)]
        distances = (
            profiles[:, None, :] - centroids_so_far[None, :, :]
        ).square().sum(dim=2)
        minimum_distances = distances.min(dim=1).values
        # Exclude every duplicate of an already chosen profile.
        for index in chosen:
            eligible &= ~(profiles == profiles[index]).all(dim=1)
        next_index = _deterministic_maximum_row(
            minimum_distances,
            profiles,
            eligible=eligible,
        )
        chosen.append(next_index)
        eligible[next_index] = False

    centroids = profiles[torch.tensor(chosen)].clone()
    previous: Tensor | None = None
    assignments = torch.empty(rows.shape[0], dtype=torch.int64)
    completed_iterations = 0

    for iteration in range(1, iteration_limit + 1):
        distances = (
            profiles[:, None, :] - centroids[None, :, :]
        ).square().sum(dim=2)
        assignments = distances.argmin(dim=1)
        counts = torch.bincount(assignments, minlength=routes)
        for empty_route in (counts == 0).nonzero(as_tuple=False).flatten().tolist():
            movable = counts[assignments] > 1
            assigned_distances = distances[
                torch.arange(rows.shape[0]),
                assignments,
            ]
            moved_index = _deterministic_maximum_row(
                assigned_distances,
                profiles,
                eligible=movable,
            )
            donor = int(assignments[moved_index].item())
            assignments[moved_index] = empty_route
            counts[donor] -= 1
            counts[empty_route] += 1

        completed_iterations = iteration
        if previous is not None and torch.equal(assignments, previous):
            break
        centroids = torch.stack(
            [
                _normalized_need_rows(
                    profiles[assignments == route].mean(
                        dim=0,
                        keepdim=True,
                    )
                )[0]
                for route in range(routes)
            ]
        )
        previous = assignments.clone()

    final_distances = (
        profiles - centroids[assignments]
    ).square().sum(dim=1)
    route_counts = tuple(
        int(count)
        for count in torch.bincount(assignments, minlength=routes).tolist()
    )
    return FisherNeedClustering(
        assignments=assignments,
        centroids=centroids,
        selected_flat_indices=selected_indices,
        route_counts=route_counts,
        iterations=completed_iterations,
        objective=float(final_distances.sum().item()),
    )


def partition_fisher_need_profiles_by_total_need(
    need_profiles: Tensor,
    *,
    route_count: int,
    valid_mask: Tensor | None = None,
) -> FisherNeedClustering:
    """Partition tokens into deterministic equal-frequency total-need bins.

    Routes are ordered from lowest to highest total Fisher need.  Every route
    is nonempty and route sizes differ by at most one.  Ties are resolved by
    normalized profile values and then original flattened row index.
    """

    routes = _require_positive_int(route_count, label="route_count")
    rows, selected_indices = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    if routes > rows.shape[0]:
        raise ValueError("route_count cannot exceed selected token rows")
    normalized = _normalized_need_rows(rows)
    totals = rows.sum(dim=1)
    ordered = sorted(
        range(rows.shape[0]),
        key=lambda index: (
            float(totals[index].item()),
            tuple(float(value) for value in normalized[index].tolist()),
            int(selected_indices[index].item()),
        ),
    )
    base, remainder = divmod(rows.shape[0], routes)
    assignments = torch.empty(rows.shape[0], dtype=torch.int64)
    cursor = 0
    counts: list[int] = []
    for route in range(routes):
        count = base + (1 if route < remainder else 0)
        assignments[torch.tensor(ordered[cursor : cursor + count])] = route
        counts.append(count)
        cursor += count
    centroids = torch.stack(
        [
            _normalized_need_rows(
                normalized[assignments == route].mean(
                    dim=0,
                    keepdim=True,
                )
            )[0]
            for route in range(routes)
        ]
    )
    objective = float(
        (normalized - centroids[assignments]).square().sum().item()
    )
    return FisherNeedClustering(
        assignments=assignments,
        centroids=centroids,
        selected_flat_indices=selected_indices,
        route_counts=tuple(counts),
        iterations=1,
        objective=objective,
    )


@dataclass(frozen=True, slots=True)
class TotalNeedRouteTeacher:
    """Portable A-fitted cutpoints for total Fisher-need route labels.

    The teacher is useful as an unavailable-at-inference ceiling on fresh
    data.  It sees each token's Fisher need, sums over modes, and applies only
    the thresholds fitted on calibration A.  It never refits quantiles on the
    evaluation split.
    """

    thresholds: Tensor
    fit_observations: int
    fit_quantiles: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.thresholds, Tensor)
            or self.thresholds.device.type != "cpu"
            or self.thresholds.dtype != torch.float64
            or self.thresholds.ndim != 1
            or not torch.isfinite(self.thresholds).all()
        ):
            raise ValueError(
                "thresholds must be a finite CPU float64 vector"
            )
        if (
            self.thresholds.numel() > 1
            and (self.thresholds[1:] < self.thresholds[:-1]).any()
        ):
            raise ValueError("thresholds must be nondecreasing")
        _require_positive_int(
            self.fit_observations,
            label="fit_observations",
        )
        if (
            type(self.fit_quantiles) is not tuple
            or len(self.fit_quantiles) != self.thresholds.numel()
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.fit_quantiles
            )
            or tuple(sorted(set(float(value) for value in self.fit_quantiles)))
            != tuple(float(value) for value in self.fit_quantiles)
            or any(not 0 < float(value) < 1 for value in self.fit_quantiles)
        ):
            raise ValueError(
                "fit_quantiles must be unique ascending values in (0, 1)"
            )
        object.__setattr__(
            self,
            "thresholds",
            self.thresholds.detach().clone(),
        )
        object.__setattr__(
            self,
            "fit_quantiles",
            tuple(float(value) for value in self.fit_quantiles),
        )

    @property
    def routes(self) -> int:
        return self.thresholds.numel() + 1

    def assign(
        self,
        need_profiles: Tensor,
        *,
        valid_mask: Tensor | None = None,
        invalid_route: int = 0,
    ) -> Tensor:
        """Apply frozen thresholds and return a leading-shape route grid."""

        if (
            type(invalid_route) is not int
            or not 0 <= invalid_route < self.routes
        ):
            raise ValueError("invalid_route exceeds the teacher route range")
        rows = _finite_float64_rows(
            need_profiles,
            label="need_profiles",
            nonnegative=True,
        )
        selection = _selection_mask(
            need_profiles.shape[:-1],
            valid_mask,
            label="valid_mask",
        )
        output = torch.full(
            (rows.shape[0],),
            invalid_route,
            dtype=torch.int64,
        )
        selected = rows[selection]
        if selected.shape[0] == 0:
            raise ValueError("need_profiles has no selected token rows")
        routes = torch.bucketize(
            selected.sum(dim=1),
            self.thresholds,
            right=True,
        )
        output[selection] = routes
        return output.reshape(need_profiles.shape[:-1])

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _TOTAL_NEED_TEACHER_KIND,
            "format_version": _FORMAT_VERSION,
            "thresholds": self.thresholds.detach().clone(),
            "fit_observations": self.fit_observations,
            "fit_quantiles": list(self.fit_quantiles),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> TotalNeedRouteTeacher:
        expected = {
            "artifact_kind",
            "format_version",
            "thresholds",
            "fit_observations",
            "fit_quantiles",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("total-need route teacher fields are invalid")
        if state["artifact_kind"] != _TOTAL_NEED_TEACHER_KIND:
            raise ValueError("unsupported total-need route teacher kind")
        if (
            type(state["format_version"]) is not int
            or state["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("unsupported total-need route teacher version")
        return cls(
            thresholds=state["thresholds"],  # type: ignore[arg-type]
            fit_observations=state["fit_observations"],  # type: ignore[arg-type]
            fit_quantiles=tuple(state["fit_quantiles"]),  # type: ignore[arg-type]
        )


def fit_total_need_route_teacher(
    need_profiles: Tensor,
    *,
    route_count: int,
    valid_mask: Tensor | None = None,
    quantiles: Sequence[float] | None = None,
) -> TotalNeedRouteTeacher:
    """Fit deterministic caller-declared total-need cutpoints on A only."""

    routes = _require_positive_int(route_count, label="route_count")
    rows, _ = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    if routes > rows.shape[0]:
        raise ValueError("route_count cannot exceed selected token rows")
    if quantiles is None:
        fitted_quantiles = tuple(
            index / routes for index in range(1, routes)
        )
    else:
        if not isinstance(quantiles, Sequence) or isinstance(
            quantiles,
            (str, bytes),
        ):
            raise TypeError("quantiles must be a numeric sequence")
        fitted_quantiles = tuple(float(value) for value in quantiles)
        if (
            len(fitted_quantiles) != routes - 1
            or tuple(sorted(set(fitted_quantiles))) != fitted_quantiles
            or any(not math.isfinite(value) or not 0 < value < 1 for value in fitted_quantiles)
        ):
            raise ValueError(
                "quantiles must contain route_count - 1 unique ascending "
                "values in (0, 1)"
            )
    if routes == 1:
        thresholds = torch.empty(0, dtype=torch.float64)
    else:
        totals = rows.sum(dim=1)
        thresholds = torch.quantile(
            totals,
            torch.tensor(fitted_quantiles, dtype=torch.float64),
            interpolation="midpoint",
        )
    return TotalNeedRouteTeacher(
        thresholds=thresholds,
        fit_observations=rows.shape[0],
        fit_quantiles=fitted_quantiles,
    )


def partition_fisher_need_profiles_by_teacher(
    need_profiles: Tensor,
    teacher: TotalNeedRouteTeacher,
    *,
    valid_mask: Tensor | None = None,
) -> FisherNeedClustering:
    """Build a deterministic A partition from frozen total-need cutpoints."""

    if not isinstance(teacher, TotalNeedRouteTeacher):
        raise TypeError("teacher must be a TotalNeedRouteTeacher")
    rows, selected_indices = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    assignments = torch.bucketize(
        rows.sum(dim=1),
        teacher.thresholds,
        right=True,
    )
    counts = torch.bincount(assignments, minlength=teacher.routes)
    if (counts == 0).any():
        raise ValueError(
            "teacher cutpoints leave an empty route on the partition data"
        )
    normalized = _normalized_need_rows(rows)
    centroids = torch.stack(
        [
            _normalized_need_rows(
                normalized[assignments == route].mean(
                    dim=0,
                    keepdim=True,
                )
            )[0]
            for route in range(teacher.routes)
        ]
    )
    objective = float(
        (normalized - centroids[assignments]).square().sum().item()
    )
    return FisherNeedClustering(
        assignments=assignments,
        centroids=centroids,
        selected_flat_indices=selected_indices,
        route_counts=tuple(int(value) for value in counts.tolist()),
        iterations=1,
        objective=objective,
    )


@dataclass(frozen=True, slots=True)
class ConditionalModeTable:
    """Inspectable route-specific modal masks and their Fisher centroids."""

    mode_masks: Tensor
    common_mask: Tensor
    need_centroids: Tensor
    route_budgets: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mode_masks, Tensor)
            or self.mode_masks.device.type != "cpu"
            or self.mode_masks.dtype != torch.bool
            or self.mode_masks.ndim != 2
            or self.mode_masks.shape[0] == 0
            or self.mode_masks.shape[1] == 0
        ):
            raise ValueError("mode_masks must be a nonempty CPU boolean matrix")
        routes, modes = self.mode_masks.shape
        if (
            not isinstance(self.common_mask, Tensor)
            or self.common_mask.device.type != "cpu"
            or self.common_mask.dtype != torch.bool
            or self.common_mask.shape != (modes,)
        ):
            raise ValueError("common_mask must match the modal width")
        if (
            not isinstance(self.need_centroids, Tensor)
            or self.need_centroids.device.type != "cpu"
            or self.need_centroids.dtype != torch.float64
            or self.need_centroids.shape != (routes, modes)
            or not torch.isfinite(self.need_centroids).all()
            or (self.need_centroids < 0).any()
        ):
            raise ValueError(
                "need_centroids must be a finite nonnegative CPU float64 matrix"
            )
        if (
            type(self.route_budgets) is not tuple
            or len(self.route_budgets) != routes
            or any(
                type(budget) is not int or not 0 <= budget <= modes
                for budget in self.route_budgets
            )
        ):
            raise ValueError("route_budgets must be valid for every route")
        actual = tuple(int(mask.sum().item()) for mask in self.mode_masks)
        if actual != self.route_budgets:
            raise ValueError("route masks must exactly match route_budgets")
        if not (self.mode_masks[:, self.common_mask]).all():
            raise ValueError("every route must contain every common mode")
        sums = self.need_centroids.sum(dim=1)
        if not torch.all((sums == 0) | torch.isclose(sums, torch.ones_like(sums))):
            raise ValueError("need_centroids must be zero or L1-normalized")
        object.__setattr__(
            self,
            "mode_masks",
            self.mode_masks.detach().clone(),
        )
        object.__setattr__(
            self,
            "common_mask",
            self.common_mask.detach().clone(),
        )
        object.__setattr__(
            self,
            "need_centroids",
            self.need_centroids.detach().clone(),
        )

    @property
    def routes(self) -> int:
        return self.mode_masks.shape[0]

    @property
    def modes(self) -> int:
        return self.mode_masks.shape[1]

    @property
    def common_modes(self) -> int:
        return int(self.common_mask.sum().item())

    @classmethod
    def from_masks(
        cls,
        mode_masks: Tensor,
        *,
        need_centroids: Tensor | None = None,
        common_mask: Tensor | None = None,
    ) -> ConditionalModeTable:
        """Build a table from arbitrary caller-specified route masks."""

        if (
            not isinstance(mode_masks, Tensor)
            or mode_masks.dtype != torch.bool
            or mode_masks.ndim != 2
            or mode_masks.shape[0] == 0
            or mode_masks.shape[1] == 0
        ):
            raise ValueError("mode_masks must be a nonempty boolean matrix")
        masks = mode_masks.detach().to(device="cpu").clone()
        if common_mask is None:
            common = masks.all(dim=0)
        else:
            if (
                not isinstance(common_mask, Tensor)
                or common_mask.dtype != torch.bool
                or common_mask.shape != (masks.shape[1],)
            ):
                raise ValueError("common_mask must match the modal width")
            common = common_mask.detach().to(device="cpu").clone()
        if need_centroids is None:
            centroids = torch.zeros(
                masks.shape,
                dtype=torch.float64,
            )
        else:
            centroids = need_centroids.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
        return cls(
            mode_masks=masks,
            common_mask=common,
            need_centroids=centroids,
            route_budgets=tuple(
                int(mask.sum().item()) for mask in masks
            ),
        )

    def masks_for(self, route_ids: Tensor) -> Tensor:
        if not isinstance(route_ids, Tensor):
            raise TypeError("route_ids must be a Tensor")
        if route_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("route_ids must use an integer dtype")
        if route_ids.numel() == 0:
            raise ValueError("route_ids cannot be empty")
        if int(route_ids.min().item()) < 0 or int(route_ids.max().item()) >= self.routes:
            raise ValueError("route_ids exceed the route table")
        masks = self.mode_masks.to(device=route_ids.device)
        return masks[route_ids.to(dtype=torch.int64)]

    def mask_coordinates(
        self,
        coordinates: Tensor,
        route_ids: Tensor,
    ) -> Tensor:
        """Zero inactive modal coordinates without changing tensor shape."""

        if not isinstance(coordinates, Tensor) or not coordinates.is_floating_point():
            raise TypeError("coordinates must be a floating Tensor")
        if coordinates.ndim < 2 or coordinates.shape[-1] != self.modes:
            raise ValueError(
                "coordinates must have shape [..., table modes]"
            )
        if route_ids.shape != coordinates.shape[:-1]:
            raise ValueError(
                "route_ids must match the coordinate leading dimensions"
            )
        return coordinates * self.masks_for(route_ids).to(
            dtype=coordinates.dtype
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _TABLE_KIND,
            "format_version": _FORMAT_VERSION,
            "mode_masks": self.mode_masks.detach().clone(),
            "common_mask": self.common_mask.detach().clone(),
            "need_centroids": self.need_centroids.detach().clone(),
            "route_budgets": list(self.route_budgets),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> ConditionalModeTable:
        expected = {
            "artifact_kind",
            "format_version",
            "mode_masks",
            "common_mask",
            "need_centroids",
            "route_budgets",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("conditional mode table fields are invalid")
        if state["artifact_kind"] != _TABLE_KIND:
            raise ValueError("unsupported conditional mode table kind")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("unsupported conditional mode table version")
        raw_budgets = state["route_budgets"]
        if not isinstance(raw_budgets, list):
            raise TypeError("route_budgets must be a list")
        return cls(
            mode_masks=state["mode_masks"],  # type: ignore[arg-type]
            common_mask=state["common_mask"],  # type: ignore[arg-type]
            need_centroids=state["need_centroids"],  # type: ignore[arg-type]
            route_budgets=tuple(raw_budgets),
        )


def _normalize_budgets(
    route_budgets: int | Sequence[int],
    *,
    routes: int,
    modes: int,
) -> tuple[int, ...]:
    if type(route_budgets) is int:
        budgets = (route_budgets,) * routes
    elif isinstance(route_budgets, Sequence) and not isinstance(
        route_budgets,
        (str, bytes),
    ):
        budgets = tuple(route_budgets)
    else:
        raise TypeError("route_budgets must be an integer or integer sequence")
    if len(budgets) != routes:
        raise ValueError("route_budgets must contain one budget per route")
    if any(type(budget) is not int or not 0 <= budget <= modes for budget in budgets):
        raise ValueError(f"route budgets must be between 0 and {modes}")
    return budgets


def build_conditional_mode_table(
    need_profiles: Tensor,
    clustering: FisherNeedClustering,
    *,
    route_budgets: int | Sequence[int],
    common_modes: int = 0,
    valid_mask: Tensor | None = None,
) -> ConditionalModeTable:
    """Build exact-budget masks from clustered Fisher need.

    Common modes are the globally highest-need modes and appear in every
    route.  Remaining slots are filled from that route's mean normalized need.
    Ties always prefer the lower modal index.
    """

    if not isinstance(clustering, FisherNeedClustering):
        raise TypeError("clustering must be a FisherNeedClustering")
    rows, selected = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    if (
        not torch.equal(selected, clustering.selected_flat_indices)
        or rows.shape[0] != clustering.observations
        or rows.shape[1] != clustering.modes
    ):
        raise ValueError(
            "need_profiles selection must exactly match the clustering"
        )
    budgets = _normalize_budgets(
        route_budgets,
        routes=clustering.routes,
        modes=clustering.modes,
    )
    common_count = _require_nonnegative_int(
        common_modes,
        label="common_modes",
    )
    if common_count > min(budgets):
        raise ValueError("common_modes cannot exceed the smallest route budget")

    normalized = _normalized_need_rows(rows)
    global_need = rows.mean(dim=0)
    common_indices = _stable_descending_indices(global_need)[:common_count]
    common_mask = torch.zeros(clustering.modes, dtype=torch.bool)
    if common_indices:
        common_mask[list(common_indices)] = True

    masks = torch.zeros(
        clustering.routes,
        clustering.modes,
        dtype=torch.bool,
    )
    centroids = torch.empty_like(clustering.centroids)
    for route in range(clustering.routes):
        route_rows = normalized[clustering.assignments == route]
        centroid = _normalized_need_rows(
            route_rows.mean(dim=0, keepdim=True)
        )[0]
        centroids[route] = centroid
        masks[route] = common_mask
        route_mean_need = rows[clustering.assignments == route].mean(dim=0)
        specialist_order = [
            index
            for index in _stable_descending_indices(route_mean_need)
            if not common_mask[index]
        ]
        needed = budgets[route] - common_count
        if needed:
            masks[route, specialist_order[:needed]] = True

    return ConditionalModeTable(
        mode_masks=masks,
        common_mask=common_mask,
        need_centroids=centroids,
        route_budgets=budgets,
    )


def assign_need_profiles_to_routes(
    need_profiles: Tensor,
    mode_table: ConditionalModeTable,
    *,
    method: str = "maximum_capture",
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return deterministic oracle routes for observed Fisher need.

    ``maximum_capture`` selects the mask capturing the most need mass.
    ``nearest_centroid`` selects the closest L1-normalized route centroid.
    Both resolve ties in favor of the lower route index.
    """

    if not isinstance(mode_table, ConditionalModeTable):
        raise TypeError("mode_table must be a ConditionalModeTable")
    rows, _ = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    if rows.shape[1] != mode_table.modes:
        raise ValueError("need profile width must match the route table")
    if method == "maximum_capture":
        scores = rows @ mode_table.mode_masks.to(dtype=torch.float64).T
        return scores.argmax(dim=1)
    if method == "nearest_centroid":
        normalized = _normalized_need_rows(rows)
        distances = (
            normalized[:, None, :]
            - mode_table.need_centroids[None, :, :]
        ).square().sum(dim=2)
        return distances.argmin(dim=1)
    raise ValueError(
        "method must be 'maximum_capture' or 'nearest_centroid'"
    )


@dataclass(frozen=True, slots=True)
class RouterClassificationMetrics:
    observations: int
    accuracy: float
    cross_entropy: float
    macro_recall: float
    target_route_counts: tuple[int, ...]
    predicted_route_counts: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class PointwiseCausalRouter:
    """A serialized pointwise router fitted on current block-input features."""

    feature_mean: Tensor
    feature_scale: Tensor
    weight: Tensor
    bias: Tensor
    ridge: float
    observations: int

    def __post_init__(self) -> None:
        for name, value, ndim in (
            ("feature_mean", self.feature_mean, 1),
            ("feature_scale", self.feature_scale, 1),
            ("weight", self.weight, 2),
            ("bias", self.bias, 1),
        ):
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float64
                or value.ndim != ndim
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} must be a finite CPU float64 Tensor"
                )
        features = self.feature_mean.numel()
        routes = self.bias.numel()
        if features == 0 or routes == 0:
            raise ValueError("router dimensions must be nonzero")
        if self.feature_scale.shape != (features,) or self.weight.shape != (
            features,
            routes,
        ):
            raise ValueError("router tensor shapes are inconsistent")
        if (self.feature_scale <= 0).any():
            raise ValueError("feature_scale must be positive")
        _require_finite_float(self.ridge, label="ridge", positive=True)
        _require_positive_int(self.observations, label="observations")
        for name in ("feature_mean", "feature_scale", "weight", "bias"):
            object.__setattr__(
                self,
                name,
                getattr(self, name).detach().clone(),
            )

    @property
    def input_features(self) -> int:
        return self.feature_mean.numel()

    @property
    def routes(self) -> int:
        return self.bias.numel()

    def logits(self, block_inputs: Tensor) -> Tensor:
        """Compute route logits independently at every tensor position."""

        if not isinstance(block_inputs, Tensor):
            raise TypeError("block_inputs must be a Tensor")
        if (
            block_inputs.ndim < 2
            or block_inputs.shape[-1] != self.input_features
        ):
            raise ValueError(
                "block_inputs must have shape [..., router input features]"
            )
        if not block_inputs.is_floating_point():
            raise ValueError("block_inputs must be floating point")
        if not torch.isfinite(block_inputs).all():
            raise ValueError("block_inputs must be finite")
        compute_dtype = (
            torch.float32
            if block_inputs.dtype in (torch.float16, torch.bfloat16)
            else block_inputs.dtype
        )
        values = block_inputs.to(dtype=compute_dtype)
        mean = self.feature_mean.to(
            device=values.device,
            dtype=compute_dtype,
        )
        scale = self.feature_scale.to(
            device=values.device,
            dtype=compute_dtype,
        )
        weight = self.weight.to(
            device=values.device,
            dtype=compute_dtype,
        )
        bias = self.bias.to(
            device=values.device,
            dtype=compute_dtype,
        )
        return ((values - mean) / scale) @ weight + bias

    def probabilities(self, block_inputs: Tensor) -> Tensor:
        return F.softmax(self.logits(block_inputs), dim=-1)

    def predict(self, block_inputs: Tensor) -> Tensor:
        return self.logits(block_inputs).argmax(dim=-1)

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _ROUTER_KIND,
            "format_version": _FORMAT_VERSION,
            "feature_mean": self.feature_mean.detach().clone(),
            "feature_scale": self.feature_scale.detach().clone(),
            "weight": self.weight.detach().clone(),
            "bias": self.bias.detach().clone(),
            "ridge": self.ridge,
            "observations": self.observations,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> PointwiseCausalRouter:
        expected = {
            "artifact_kind",
            "format_version",
            "feature_mean",
            "feature_scale",
            "weight",
            "bias",
            "ridge",
            "observations",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("pointwise causal router fields are invalid")
        if state["artifact_kind"] != _ROUTER_KIND:
            raise ValueError("unsupported pointwise causal router kind")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("unsupported pointwise causal router version")
        return cls(
            feature_mean=state["feature_mean"],  # type: ignore[arg-type]
            feature_scale=state["feature_scale"],  # type: ignore[arg-type]
            weight=state["weight"],  # type: ignore[arg-type]
            bias=state["bias"],  # type: ignore[arg-type]
            ridge=state["ridge"],  # type: ignore[arg-type]
            observations=state["observations"],  # type: ignore[arg-type]
        )


def _selected_route_labels(
    route_labels: Tensor,
    *,
    leading_shape: torch.Size,
    valid_mask: Tensor | None,
    route_count: int,
) -> Tensor:
    if not isinstance(route_labels, Tensor):
        raise TypeError("route_labels must be a Tensor")
    if route_labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("route_labels must use an integer dtype")
    if route_labels.shape != leading_shape:
        raise ValueError(
            "route_labels must match the feature leading dimensions"
        )
    mask = _selection_mask(
        leading_shape,
        valid_mask,
        label="valid_mask",
    )
    labels = route_labels.detach().to(device="cpu", dtype=torch.int64).reshape(-1)[
        mask
    ]
    if labels.numel() == 0:
        raise ValueError("route_labels has no selected token rows")
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= route_count:
        raise ValueError("route_labels exceed route_count")
    return labels


def _router_metrics(
    router: PointwiseCausalRouter,
    features: Tensor,
    targets: Tensor,
) -> RouterClassificationMetrics:
    logits = router.logits(features)
    predicted = logits.argmax(dim=-1)
    routes = router.routes
    confusion = torch.zeros(routes, routes, dtype=torch.int64)
    for target, prediction in zip(targets, predicted, strict=True):
        confusion[int(target.item()), int(prediction.item())] += 1
    target_counts = torch.bincount(targets, minlength=routes)
    predicted_counts = torch.bincount(predicted, minlength=routes)
    observed = target_counts > 0
    recalls = (
        confusion.diagonal()[observed].to(dtype=torch.float64)
        / target_counts[observed]
    )
    return RouterClassificationMetrics(
        observations=targets.numel(),
        accuracy=float((predicted == targets).double().mean().item()),
        cross_entropy=float(F.cross_entropy(logits, targets).item()),
        macro_recall=float(recalls.mean().item()),
        target_route_counts=tuple(int(value) for value in target_counts.tolist()),
        predicted_route_counts=tuple(
            int(value) for value in predicted_counts.tolist()
        ),
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in confusion.tolist()
        ),
    )


def fit_pointwise_causal_router(
    block_inputs: Tensor,
    route_labels: Tensor,
    *,
    route_count: int,
    valid_mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    ridge: float = 1e-3,
) -> tuple[PointwiseCausalRouter, RouterClassificationMetrics]:
    """Fit a deterministic ridge classifier from current-token inputs.

    The fit is a closed-form, weighted least-squares classification problem.
    It intentionally contains no temporal mixing: a transformer's incoming
    hidden state may summarize its causal prefix, but this router never reads
    another tensor position itself.
    """

    routes = _require_positive_int(route_count, label="route_count")
    regularization = _require_finite_float(
        ridge,
        label="ridge",
        positive=True,
    )
    rows, _ = _selected_float64_rows(
        block_inputs,
        label="block_inputs",
        valid_mask=valid_mask,
    )
    targets = _selected_route_labels(
        route_labels,
        leading_shape=block_inputs.shape[:-1],
        valid_mask=valid_mask,
        route_count=routes,
    )
    if rows.shape[0] != targets.numel():
        raise ValueError("block inputs and route labels must align")

    if sample_weights is None:
        weights = torch.ones(rows.shape[0], dtype=torch.float64)
    else:
        if (
            not isinstance(sample_weights, Tensor)
            or not sample_weights.is_floating_point()
            or sample_weights.shape != block_inputs.shape[:-1]
        ):
            raise ValueError(
                "sample_weights must be floating and match leading dimensions"
            )
        mask = _selection_mask(
            block_inputs.shape[:-1],
            valid_mask,
            label="valid_mask",
        )
        weights = (
            sample_weights.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(-1)[mask]
        )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weights must be finite and nonnegative")
        if float(weights.sum().item()) <= 0:
            raise ValueError("sample_weights must contain positive mass")

    total_weight = weights.sum()
    feature_mean = (weights[:, None] * rows).sum(dim=0) / total_weight
    centered = rows - feature_mean
    variance = (weights[:, None] * centered.square()).sum(dim=0) / total_weight
    feature_scale = variance.sqrt()
    feature_scale = torch.where(
        feature_scale > 64 * torch.finfo(torch.float64).eps,
        feature_scale,
        torch.ones_like(feature_scale),
    )
    normalized = centered / feature_scale

    one_hot = F.one_hot(targets, num_classes=routes).to(dtype=torch.float64)
    target_mean = (weights[:, None] * one_hot).sum(dim=0) / total_weight
    centered_targets = one_hot - target_mean
    weighted_features = normalized * weights[:, None]
    gram = normalized.T @ weighted_features
    gram = gram + regularization * torch.eye(
        rows.shape[1],
        dtype=torch.float64,
    )
    rhs = normalized.T @ (weights[:, None] * centered_targets)
    weight = torch.linalg.solve(gram, rhs)

    router = PointwiseCausalRouter(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weight=weight,
        bias=target_mean,
        ridge=regularization,
        observations=rows.shape[0],
    )
    metrics = _router_metrics(router, rows, targets)
    return router, metrics


@dataclass(frozen=True, slots=True)
class ConditionalComputeMetrics:
    """Logical route quality and mode-activity accounting."""

    observations: int
    total_need: float
    captured_need_fraction: float
    mean_token_captured_fraction: float
    p10_token_captured_fraction: float
    minimum_token_captured_fraction: float
    oracle_capture_fraction: float
    routing_regret_fraction: float
    average_active_modes: float
    active_mode_ratio: float
    ideal_mode_activation_reduction_fraction: float
    route_counts: tuple[int, ...]
    route_utilization: tuple[float, ...]
    target_route_accuracy: float | None
    zero_need_tokens: int


def evaluate_conditional_compute(
    need_profiles: Tensor,
    route_ids: Tensor,
    mode_table: ConditionalModeTable,
    *,
    valid_mask: Tensor | None = None,
    target_route_ids: Tensor | None = None,
) -> ConditionalComputeMetrics:
    """Measure need capture and logical active-mode usage for selected routes."""

    if not isinstance(mode_table, ConditionalModeTable):
        raise TypeError("mode_table must be a ConditionalModeTable")
    rows, _ = _selected_float64_rows(
        need_profiles,
        label="need_profiles",
        valid_mask=valid_mask,
        nonnegative=True,
    )
    selected_routes = _selected_route_labels(
        route_ids,
        leading_shape=need_profiles.shape[:-1],
        valid_mask=valid_mask,
        route_count=mode_table.routes,
    )
    if rows.shape[1] != mode_table.modes:
        raise ValueError("need profile width must match the route table")
    masks = mode_table.mode_masks[selected_routes].to(dtype=torch.float64)
    token_need = rows.sum(dim=1)
    captured = (rows * masks).sum(dim=1)
    fractions = torch.where(
        token_need > 0,
        captured / token_need.clamp_min(1e-300),
        torch.ones_like(token_need),
    )
    total_need = float(token_need.sum().item())
    aggregate_capture = (
        float(captured.sum().item()) / total_need
        if total_need > 0
        else 1.0
    )

    oracle_routes = assign_need_profiles_to_routes(rows, mode_table)
    oracle_masks = mode_table.mode_masks[oracle_routes].to(dtype=torch.float64)
    oracle_captured = float((rows * oracle_masks).sum().item())
    oracle_fraction = oracle_captured / total_need if total_need > 0 else 1.0

    active_counts = torch.tensor(
        mode_table.route_budgets,
        dtype=torch.float64,
    )[selected_routes]
    route_counts = torch.bincount(
        selected_routes,
        minlength=mode_table.routes,
    )
    target_accuracy: float | None = None
    if target_route_ids is not None:
        targets = _selected_route_labels(
            target_route_ids,
            leading_shape=need_profiles.shape[:-1],
            valid_mask=valid_mask,
            route_count=mode_table.routes,
        )
        target_accuracy = float(
            (selected_routes == targets).double().mean().item()
        )

    average_active = float(active_counts.mean().item())
    active_ratio = average_active / mode_table.modes
    return ConditionalComputeMetrics(
        observations=rows.shape[0],
        total_need=total_need,
        captured_need_fraction=aggregate_capture,
        mean_token_captured_fraction=float(fractions.mean().item()),
        p10_token_captured_fraction=float(
            torch.quantile(fractions, 0.1).item()
        ),
        minimum_token_captured_fraction=float(fractions.min().item()),
        oracle_capture_fraction=oracle_fraction,
        routing_regret_fraction=max(0.0, oracle_fraction - aggregate_capture),
        average_active_modes=average_active,
        active_mode_ratio=active_ratio,
        ideal_mode_activation_reduction_fraction=1.0 - active_ratio,
        route_counts=tuple(int(value) for value in route_counts.tolist()),
        route_utilization=tuple(
            float(value) / rows.shape[0] for value in route_counts.tolist()
        ),
        target_route_accuracy=target_accuracy,
        zero_need_tokens=int((token_need == 0).sum().item()),
    )


@dataclass(frozen=True, slots=True)
class ConditionalModalRoutingPlan:
    """Serializable table plus a runtime-safe causal route predictor."""

    mode_table: ConditionalModeTable
    router: PointwiseCausalRouter
    profile_semantics: str = "per_token_nonnegative_fisher_need"

    def __post_init__(self) -> None:
        if not isinstance(self.mode_table, ConditionalModeTable):
            raise TypeError("mode_table must be a ConditionalModeTable")
        if not isinstance(self.router, PointwiseCausalRouter):
            raise TypeError("router must be a PointwiseCausalRouter")
        if self.mode_table.routes != self.router.routes:
            raise ValueError("mode table and router route counts must match")
        if not isinstance(self.profile_semantics, str) or not self.profile_semantics:
            raise ValueError("profile_semantics must be a nonempty string")

    def route(self, block_inputs: Tensor) -> Tensor:
        return self.router.predict(block_inputs)

    def mask_coordinates(
        self,
        coordinates: Tensor,
        block_inputs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if coordinates.shape[:-1] != block_inputs.shape[:-1]:
            raise ValueError(
                "coordinates and block_inputs must share leading dimensions"
            )
        routes = self.route(block_inputs)
        return self.mode_table.mask_coordinates(coordinates, routes), routes

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _PLAN_KIND,
            "format_version": _FORMAT_VERSION,
            "profile_semantics": self.profile_semantics,
            "mode_table": self.mode_table.state_dict(),
            "router": self.router.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ConditionalModalRoutingPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "profile_semantics",
            "mode_table",
            "router",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("conditional routing plan fields are invalid")
        if state["artifact_kind"] != _PLAN_KIND:
            raise ValueError("unsupported conditional routing plan kind")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("unsupported conditional routing plan version")
        raw_table = state["mode_table"]
        raw_router = state["router"]
        if not isinstance(raw_table, Mapping) or not isinstance(raw_router, Mapping):
            raise TypeError("conditional routing plan components must be mappings")
        return cls(
            mode_table=ConditionalModeTable.from_state_dict(raw_table),
            router=PointwiseCausalRouter.from_state_dict(raw_router),
            profile_semantics=state["profile_semantics"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ConditionalRoutingFit:
    """Inspectable result of route discovery and causal-router fitting."""

    plan: ConditionalModalRoutingPlan
    clustering: FisherNeedClustering
    router_metrics: RouterClassificationMetrics
    teacher_metrics: ConditionalComputeMetrics
    routed_metrics: ConditionalComputeMetrics
    teacher_route_ids: Tensor


def fit_conditional_modal_routing(
    need_profiles: Tensor,
    block_inputs: Tensor,
    *,
    route_count: int,
    route_budgets: int | Sequence[int],
    common_modes: int = 0,
    valid_mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    max_iterations: int = 100,
    ridge: float = 1e-3,
    route_assignment: str = "pattern_clusters",
    profile_semantics: str = "per_token_nonnegative_fisher_need",
) -> ConditionalRoutingFit:
    """Discover routes, fit the causal router, and report both quality levels."""

    if (
        not isinstance(need_profiles, Tensor)
        or not isinstance(block_inputs, Tensor)
        or need_profiles.shape[:-1] != block_inputs.shape[:-1]
    ):
        raise ValueError(
            "need_profiles and block_inputs must share leading dimensions"
        )
    if route_assignment == "pattern_clusters":
        clustering = cluster_fisher_need_profiles(
            need_profiles,
            route_count=route_count,
            valid_mask=valid_mask,
            max_iterations=max_iterations,
        )
    elif route_assignment == "total_need_bins":
        clustering = partition_fisher_need_profiles_by_total_need(
            need_profiles,
            route_count=route_count,
            valid_mask=valid_mask,
        )
    else:
        raise ValueError(
            "route_assignment must be 'pattern_clusters' or "
            "'total_need_bins'"
        )
    table = build_conditional_mode_table(
        need_profiles,
        clustering,
        route_budgets=route_budgets,
        common_modes=common_modes,
        valid_mask=valid_mask,
    )

    # Expand selected cluster labels back to the caller's leading shape so the
    # same validity mask can be used by the generic router fitter.
    teacher_full = torch.zeros(
        math.prod(need_profiles.shape[:-1]),
        dtype=torch.int64,
        device=need_profiles.device,
    )
    teacher_full[
        clustering.selected_flat_indices.to(device=need_profiles.device)
    ] = clustering.assignments.to(device=need_profiles.device)
    teacher_full = teacher_full.reshape(need_profiles.shape[:-1])
    router, router_metrics = fit_pointwise_causal_router(
        block_inputs,
        teacher_full,
        route_count=route_count,
        valid_mask=valid_mask,
        sample_weights=sample_weights,
        ridge=ridge,
    )
    plan = ConditionalModalRoutingPlan(
        mode_table=table,
        router=router,
        profile_semantics=profile_semantics,
    )
    predicted_full = plan.route(block_inputs)
    teacher_metrics = evaluate_conditional_compute(
        need_profiles,
        teacher_full,
        table,
        valid_mask=valid_mask,
        target_route_ids=teacher_full,
    )
    routed_metrics = evaluate_conditional_compute(
        need_profiles,
        predicted_full,
        table,
        valid_mask=valid_mask,
        target_route_ids=teacher_full,
    )
    return ConditionalRoutingFit(
        plan=plan,
        clustering=clustering,
        router_metrics=router_metrics,
        teacher_metrics=teacher_metrics,
        routed_metrics=routed_metrics,
        teacher_route_ids=teacher_full.detach().to(device="cpu").clone(),
    )


__all__ = [
    "ConditionalComputeMetrics",
    "ConditionalModalRoutingPlan",
    "ConditionalModeTable",
    "ConditionalRoutingFit",
    "FisherNeedClustering",
    "PointwiseCausalRouter",
    "RouterClassificationMetrics",
    "TotalNeedRouteTeacher",
    "assign_need_profiles_to_routes",
    "build_conditional_mode_table",
    "cluster_fisher_need_profiles",
    "evaluate_conditional_compute",
    "fisher_projection_damage_profiles",
    "fit_conditional_modal_routing",
    "fit_pointwise_causal_router",
    "fit_total_need_route_teacher",
    "linear_codec_fisher_damage_profiles",
    "partition_fisher_need_profiles_by_total_need",
    "partition_fisher_need_profiles_by_teacher",
]
