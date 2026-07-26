"""Deterministic metadata controls for conditional-routing experiments.

These controls answer a narrower question than a learned hidden-state router:
could the same route choices have been recovered from categorical metadata
such as logical position, valid sequence length, or current token ID?

The implementation is intentionally simple and inspectable.  Every lookup
cell predicts its majority A-route, ties prefer the lower route ID, and unseen
cells fall through a caller-declared hierarchy before using the global
majority route.  The stratified shuffle preserves the exact route histogram
inside every metadata cell while breaking the association with hidden state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


_CONTROL_KIND = "fisher_graph.hierarchical_categorical_route_control"
_CONTROL_VERSION = 1


def _validate_route_grid(
    route_ids: Tensor,
    *,
    route_count: int,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if type(route_count) is not int or route_count <= 0:
        raise ValueError("route_count must be a positive integer")
    if (
        not isinstance(route_ids, Tensor)
        or route_ids.dtype not in (torch.int32, torch.int64)
        or route_ids.ndim == 0
    ):
        raise ValueError("route_ids must be a non-scalar integer Tensor")
    if (
        not isinstance(valid_mask, Tensor)
        or valid_mask.dtype != torch.bool
        or valid_mask.shape != route_ids.shape
    ):
        raise ValueError("valid_mask must be boolean and match route_ids")
    routes = route_ids.detach().to(device="cpu", dtype=torch.int64)
    valid = valid_mask.detach().to(device="cpu")
    selected = routes[valid]
    if selected.numel() == 0:
        raise ValueError("valid_mask must select at least one route")
    if int(selected.min().item()) < 0 or int(selected.max().item()) >= route_count:
        raise ValueError("valid route IDs exceed route_count")
    return routes, valid


def _validate_features(
    features: Mapping[str, Tensor],
    *,
    leading_shape: torch.Size,
) -> dict[str, Tensor]:
    if not isinstance(features, Mapping) or not features:
        raise ValueError("features must be a nonempty mapping")
    normalized: dict[str, Tensor] = {}
    for name, values in features.items():
        if not isinstance(name, str) or not name:
            raise ValueError("feature names must be nonempty strings")
        if (
            not isinstance(values, Tensor)
            or values.dtype not in (torch.int32, torch.int64)
            or values.shape != leading_shape
        ):
            raise ValueError(
                f"feature {name!r} must be an integer Tensor matching route_ids"
            )
        normalized[name] = values.detach().to(device="cpu", dtype=torch.int64)
    return normalized


def _majority_route(
    route_ids: Tensor,
    *,
    route_count: int,
) -> int:
    counts = torch.bincount(route_ids, minlength=route_count)
    return int(counts.argmax().item())


@dataclass(frozen=True, slots=True)
class CategoricalRouteTable:
    """One exact categorical lookup table."""

    feature_names: tuple[str, ...]
    keys: Tensor
    routes: Tensor

    def __post_init__(self) -> None:
        if (
            type(self.feature_names) is not tuple
            or not self.feature_names
            or any(not isinstance(name, str) or not name for name in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError(
                "feature_names must be unique nonempty strings"
            )
        if (
            not isinstance(self.keys, Tensor)
            or self.keys.device.type != "cpu"
            or self.keys.dtype != torch.int64
            or self.keys.ndim != 2
            or self.keys.shape[0] == 0
            or self.keys.shape[1] != len(self.feature_names)
        ):
            raise ValueError("keys must be a nonempty CPU int64 matrix")
        if (
            not isinstance(self.routes, Tensor)
            or self.routes.device.type != "cpu"
            or self.routes.dtype != torch.int64
            or self.routes.shape != (self.keys.shape[0],)
        ):
            raise ValueError("routes must align with lookup keys")
        if self.keys.unique(dim=0).shape[0] != self.keys.shape[0]:
            raise ValueError("lookup keys must be unique")
        object.__setattr__(self, "keys", self.keys.detach().clone())
        object.__setattr__(self, "routes", self.routes.detach().clone())

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "keys": self.keys.detach().clone(),
            "routes": self.routes.detach().clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CategoricalRouteTable:
        expected = {"feature_names", "keys", "routes"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("categorical route table fields are invalid")
        feature_names = state["feature_names"]
        if not isinstance(feature_names, list):
            raise TypeError("feature_names must be a list")
        return cls(
            feature_names=tuple(feature_names),
            keys=state["keys"],  # type: ignore[arg-type]
            routes=state["routes"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class HierarchicalCategoricalRouteControl:
    """A-fitted categorical route policy with deterministic fallbacks."""

    route_count: int
    tables: tuple[CategoricalRouteTable, ...]
    global_route: int

    def __post_init__(self) -> None:
        if type(self.route_count) is not int or self.route_count <= 0:
            raise ValueError("route_count must be a positive integer")
        if (
            type(self.tables) is not tuple
            or not self.tables
            or any(not isinstance(table, CategoricalRouteTable) for table in self.tables)
        ):
            raise ValueError("tables must contain at least one route table")
        if (
            type(self.global_route) is not int
            or not 0 <= self.global_route < self.route_count
        ):
            raise ValueError("global_route exceeds route_count")
        if any(
            int(table.routes.min().item()) < 0
            or int(table.routes.max().item()) >= self.route_count
            for table in self.tables
        ):
            raise ValueError("a lookup route exceeds route_count")

    def predict(
        self,
        features: Mapping[str, Tensor],
        *,
        valid_mask: Tensor,
        invalid_route: int = 0,
    ) -> Tensor:
        """Predict routes, falling through tables from specific to broad."""

        if type(invalid_route) is not int or not 0 <= invalid_route < self.route_count:
            raise ValueError("invalid_route exceeds route_count")
        if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean Tensor")
        normalized = _validate_features(
            features,
            leading_shape=valid_mask.shape,
        )
        valid = valid_mask.detach().to(device="cpu")
        output = torch.full(
            valid.shape,
            invalid_route,
            dtype=torch.int64,
        )
        flat_valid = valid.reshape(-1)
        selected_indices = flat_valid.nonzero(as_tuple=False).flatten()
        unresolved = torch.ones(selected_indices.numel(), dtype=torch.bool)
        selected_features = {
            name: values.reshape(-1).index_select(0, selected_indices)
            for name, values in normalized.items()
        }

        selected_output = torch.full(
            (selected_indices.numel(),),
            self.global_route,
            dtype=torch.int64,
        )
        for table in self.tables:
            missing = [
                name
                for name in table.feature_names
                if name not in selected_features
            ]
            if missing:
                raise ValueError(
                    "prediction features omit fitted fields: "
                    + ", ".join(missing)
                )
            row_keys = torch.stack(
                [selected_features[name] for name in table.feature_names],
                dim=1,
            )
            for key, route in zip(table.keys, table.routes, strict=True):
                matched = unresolved & (row_keys == key).all(dim=1)
                selected_output[matched] = route
                unresolved[matched] = False
            if not unresolved.any():
                break

        output.reshape(-1).index_copy_(
            0,
            selected_indices,
            selected_output,
        )
        return output

    def state_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": _CONTROL_KIND,
            "format_version": _CONTROL_VERSION,
            "route_count": self.route_count,
            "global_route": self.global_route,
            "tables": [table.state_dict() for table in self.tables],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> HierarchicalCategoricalRouteControl:
        expected = {
            "artifact_kind",
            "format_version",
            "route_count",
            "global_route",
            "tables",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "hierarchical categorical control fields are invalid"
            )
        if (
            state["artifact_kind"] != _CONTROL_KIND
            or state["format_version"] != _CONTROL_VERSION
        ):
            raise ValueError(
                "unsupported hierarchical categorical control"
            )
        tables = state["tables"]
        if not isinstance(tables, list):
            raise TypeError("control tables must be a list")
        return cls(
            route_count=state["route_count"],  # type: ignore[arg-type]
            global_route=state["global_route"],  # type: ignore[arg-type]
            tables=tuple(
                CategoricalRouteTable.from_state_dict(table)
                for table in tables
            ),
        )


def fit_hierarchical_categorical_route_control(
    route_ids: Tensor,
    features: Mapping[str, Tensor],
    *,
    valid_mask: Tensor,
    route_count: int,
    levels: Sequence[Sequence[str]],
) -> HierarchicalCategoricalRouteControl:
    """Fit majority-route tables for a declared metadata hierarchy.

    ``levels`` must be ordered from most specific to least specific.  The
    empty/global fallback is fitted automatically and must not be listed.
    """

    routes, valid = _validate_route_grid(
        route_ids,
        route_count=route_count,
        valid_mask=valid_mask,
    )
    normalized = _validate_features(
        features,
        leading_shape=routes.shape,
    )
    normalized_levels = tuple(tuple(level) for level in levels)
    if not normalized_levels:
        raise ValueError("levels must contain at least one feature selection")
    if any(
        not level
        or len(set(level)) != len(level)
        or any(name not in normalized for name in level)
        for level in normalized_levels
    ):
        raise ValueError(
            "each level must contain unique names present in features"
        )
    if len(set(normalized_levels)) != len(normalized_levels):
        raise ValueError("levels cannot contain duplicates")

    flat_valid = valid.reshape(-1)
    selected_routes = routes.reshape(-1)[flat_valid]
    selected_features = {
        name: values.reshape(-1)[flat_valid]
        for name, values in normalized.items()
    }
    tables: list[CategoricalRouteTable] = []
    for level in normalized_levels:
        cell_rows = torch.stack(
            [selected_features[name] for name in level],
            dim=1,
        )
        keys = cell_rows.unique(dim=0, sorted=True)
        cell_routes = torch.empty(keys.shape[0], dtype=torch.int64)
        for index, key in enumerate(keys):
            members = (cell_rows == key).all(dim=1)
            cell_routes[index] = _majority_route(
                selected_routes[members],
                route_count=route_count,
            )
        tables.append(
            CategoricalRouteTable(
                feature_names=level,
                keys=keys,
                routes=cell_routes,
            )
        )

    return HierarchicalCategoricalRouteControl(
        route_count=route_count,
        tables=tuple(tables),
        global_route=_majority_route(
            selected_routes,
            route_count=route_count,
        ),
    )


def stratified_shuffle_routes(
    route_ids: Tensor,
    strata: Mapping[str, Tensor],
    *,
    valid_mask: Tensor,
    route_count: int,
    seed: int,
) -> Tensor:
    """Shuffle valid routes only within exact categorical metadata cells."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    routes, valid = _validate_route_grid(
        route_ids,
        route_count=route_count,
        valid_mask=valid_mask,
    )
    normalized = _validate_features(
        strata,
        leading_shape=routes.shape,
    )
    names = tuple(sorted(normalized))
    flat_valid = valid.reshape(-1)
    selected_indices = flat_valid.nonzero(as_tuple=False).flatten()
    selected_cells = torch.stack(
        [
            normalized[name].reshape(-1).index_select(0, selected_indices)
            for name in names
        ],
        dim=1,
    )
    output = routes.clone().reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for key in selected_cells.unique(dim=0, sorted=True):
        in_cell = (selected_cells == key).all(dim=1)
        cell_indices = selected_indices[in_cell]
        if cell_indices.numel() <= 1:
            continue
        cell_routes = output.index_select(0, cell_indices)
        permutation = torch.randperm(
            cell_routes.numel(),
            generator=generator,
        )
        output.index_copy_(
            0,
            cell_indices,
            cell_routes.index_select(0, permutation),
        )
    return output.reshape(routes.shape)


def route_histograms_by_stratum(
    route_ids: Tensor,
    strata: Mapping[str, Tensor],
    *,
    valid_mask: Tensor,
    route_count: int,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    """Return exact route counts for each sorted-feature metadata cell."""

    routes, valid = _validate_route_grid(
        route_ids,
        route_count=route_count,
        valid_mask=valid_mask,
    )
    normalized = _validate_features(
        strata,
        leading_shape=routes.shape,
    )
    names = tuple(sorted(normalized))
    flat_valid = valid.reshape(-1)
    selected_routes = routes.reshape(-1)[flat_valid]
    selected_cells = torch.stack(
        [normalized[name].reshape(-1)[flat_valid] for name in names],
        dim=1,
    )
    result: dict[tuple[int, ...], tuple[int, ...]] = {}
    for key in selected_cells.unique(dim=0, sorted=True):
        members = (selected_cells == key).all(dim=1)
        counts = torch.bincount(
            selected_routes[members],
            minlength=route_count,
        )
        result[tuple(int(value) for value in key.tolist())] = tuple(
            int(value) for value in counts.tolist()
        )
    if sum(sum(counts) for counts in result.values()) != int(valid.sum().item()):
        raise RuntimeError("route histogram accounting is incomplete")
    return result


__all__ = [
    "CategoricalRouteTable",
    "HierarchicalCategoricalRouteControl",
    "fit_hierarchical_categorical_route_control",
    "route_histograms_by_stratum",
    "stratified_shuffle_routes",
]
