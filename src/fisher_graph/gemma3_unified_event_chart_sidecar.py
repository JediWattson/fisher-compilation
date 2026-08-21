"""Read-only event-chart capture and edge fitting for the Gemma 39-event graph.

The objects in this module are compiler sidecars.  A recorder may observe an
exact execution, but it only stores detached clones and its callback returns
``None``.  No object here is an ``nn.Module`` and no fitted edge is installed
in the exact runtime.

The numerical edge design contains chart coordinates only.  Event indexes,
block ordinals, and component families authenticate ordering and eligibility;
they are never appended as regression features.  An edge is fitted against
the target mutation flow left over after a frozen target-local chart baseline.
The held-out evaluator scores the real edge alongside exact zero-edge and
deterministically shuffled-source controls without refitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .gemma3_joint_cross_block_chart_executor import (
    PINNED_GEMMA3_EVENT_COUNT,
    PINNED_GEMMA3_MUTATION_COUNT,
    JointGraphEventSpec,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLES = frozenset({"input", "output"})
_MUTATION_KINDS = frozenset({"attention_mutation", "mlp_mutation"})


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _row_identity_sha256(row_keys: Sequence[tuple[str, int]]) -> str:
    return hashlib.sha256(
        b"fisher_graph:unified-event-chart-rows:v1\0"
        + _canonical_json_bytes(tuple(row_keys))
    ).hexdigest()


def _canonical_row_keys(
    row_keys: Sequence[tuple[str, int]],
    *,
    expected_count: int,
) -> tuple[tuple[str, int], ...]:
    if isinstance(row_keys, (str, bytes)) or not isinstance(row_keys, Sequence):
        raise TypeError("row_keys must be a materialized sequence")
    materialized = tuple(row_keys)
    if len(materialized) != expected_count:
        raise ValueError("row_keys do not match valid observation rows")
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        or type(item[1]) is not int
        or item[1] < 0
        for item in materialized
    ):
        raise ValueError("row_keys must contain (example_id, position) pairs")
    if len(set(materialized)) != len(materialized):
        raise ValueError("row_keys must be unique")
    return materialized


def _canonical_schedule(
    schedule: Sequence[JointGraphEventSpec],
) -> tuple[JointGraphEventSpec, ...]:
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise TypeError("schedule must be a materialized event sequence")
    result = tuple(schedule)
    if (
        len(result) != PINNED_GEMMA3_EVENT_COUNT
        or any(not isinstance(item, JointGraphEventSpec) for item in result)
        or tuple(item.event_index for item in result)
        != tuple(range(PINNED_GEMMA3_EVENT_COUNT))
        or len({item.event_id for item in result}) != len(result)
        or result[0].event_kind != "lexical_ingress"
        or tuple(item.event_kind for item in result[1:-2]).count(
            "attention_mutation"
        )
        != PINNED_GEMMA3_MUTATION_COUNT // 2
        or tuple(item.event_kind for item in result[1:-2]).count("mlp_mutation")
        != PINNED_GEMMA3_MUTATION_COUNT // 2
        or result[-2].event_kind != "terminal_rmsnorm"
        or result[-1].event_kind != "lexical_egress"
    ):
        raise ValueError("schedule is not the exact ordered 39-event contract")
    return result


def _site_semantics(event: JointGraphEventSpec, role: str) -> str:
    if role not in _ROLES:
        raise ValueError("site role must be input or output")
    if event.event_kind == "lexical_ingress":
        if role != "output":
            raise ValueError("lexical ingress exposes only its output site")
        return "lexical_hidden_state"
    if event.event_kind not in _MUTATION_KINDS:
        raise ValueError("only lexical ingress and mutations are chart sites")
    return (
        "normalized_mutation_input"
        if role == "input"
        else "mutation_residual_delta"
    )


def _expected_site_catalog(
    schedule: Sequence[JointGraphEventSpec],
) -> tuple[tuple[str, str, str], ...]:
    sites: list[tuple[str, str, str]] = []
    for event in schedule:
        if event.event_kind == "lexical_ingress":
            sites.append((f"{event.event_id}.output", event.event_id, "output"))
        elif event.event_kind in _MUTATION_KINDS:
            sites.extend(
                (
                    (f"{event.event_id}.input", event.event_id, "input"),
                    (f"{event.event_id}.output", event.event_id, "output"),
                )
            )
    expected = 1 + 2 * PINNED_GEMMA3_MUTATION_COUNT
    if len(sites) != expected:
        raise RuntimeError("unified event site catalog did not close")
    return tuple(sites)


def _finite_matrix(value: Tensor, *, label: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or not value.shape[0]
        or not value.shape[1]
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating matrix")


def _finite_packet(value: Tensor, *, label: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 3
        or not value.shape[0]
        or not value.shape[1]
        or not value.shape[2]
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite [N,C,R] tensor")


def _normalized_row_weights(value: Tensor, *, row_count: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.shape[0] != row_count
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value <= 0).any())
    ):
        raise ValueError("row_fisher_weights must be finite and positive")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    return result / result.mean().clamp_min(1.0e-12)


def _weighted_sse(residual: Tensor, weights: Tensor) -> Tensor:
    return (weights[:, None] * residual.square()).sum()


@dataclass(frozen=True, slots=True)
class UnifiedEventChartSiteObservation:
    """One detached, full-width site captured beside an exact execution."""

    site_id: str
    event_id: str
    event_index: int
    event_kind: str
    component_family: str
    block_ordinal: int | None
    role: str
    value_semantics: str
    values: Tensor
    valid_mask: Tensor
    row_keys: tuple[tuple[str, int], ...]
    source_fingerprint: str
    execution_fingerprint: str

    def __post_init__(self) -> None:
        if self.site_id != f"{self.event_id}.{self.role}":
            raise ValueError("site_id must be event_id.role")
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be nonnegative")
        if self.role not in _ROLES:
            raise ValueError("role must be input or output")
        if self.value_semantics not in {
            "lexical_hidden_state",
            "normalized_mutation_input",
            "mutation_residual_delta",
        }:
            raise ValueError("value_semantics is unsupported")
        if (
            not isinstance(self.values, Tensor)
            or self.values.ndim != 3
            or not self.values.shape[0]
            or not self.values.shape[1]
            or not self.values.shape[2]
            or not self.values.is_floating_point()
            or not bool(torch.isfinite(self.values).all())
            or self.values.device.type != "cpu"
            or self.values.requires_grad
        ):
            raise ValueError("values must be detached finite CPU [B,S,H]")
        if (
            not isinstance(self.valid_mask, Tensor)
            or self.valid_mask.shape != self.values.shape[:2]
            or self.valid_mask.dtype != torch.bool
            or self.valid_mask.device.type != "cpu"
        ):
            raise ValueError("valid_mask must be a CPU bool [B,S] tensor")
        canonical = _canonical_row_keys(
            self.row_keys,
            expected_count=int(self.valid_mask.sum().item()),
        )
        if canonical != self.row_keys:
            raise ValueError("row_keys must already be canonical")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.execution_fingerprint,
            label="execution_fingerprint",
        )

    @property
    def hidden_width(self) -> int:
        return int(self.values.shape[2])

    @property
    def valid_row_count(self) -> int:
        return len(self.row_keys)

    def valid_rows(self) -> Tensor:
        return self.values[self.valid_mask].contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "event_id": self.event_id,
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "component_family": self.component_family,
            "block_ordinal": self.block_ordinal,
            "role": self.role,
            "value_semantics": self.value_semantics,
            "batch_size": int(self.values.shape[0]),
            "sequence_width": int(self.values.shape[1]),
            "hidden_width": self.hidden_width,
            "valid_row_count": self.valid_row_count,
            "row_identity_sha256": _row_identity_sha256(self.row_keys),
            "source_fingerprint": self.source_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "detached_cpu_clone": True,
        }


@dataclass(frozen=True, slots=True)
class UnifiedEventChartCapture:
    """Closed inventory of the 73 observation sites for one exact run."""

    schedule: tuple[JointGraphEventSpec, ...]
    observations: tuple[UnifiedEventChartSiteObservation, ...]
    valid_mask: Tensor
    row_keys: tuple[tuple[str, int], ...]
    source_fingerprint: str
    execution_fingerprint: str

    def __post_init__(self) -> None:
        schedule = _canonical_schedule(self.schedule)
        if schedule != self.schedule:
            raise ValueError("schedule must already be canonical")
        catalog = _expected_site_catalog(schedule)
        if tuple(item.site_id for item in self.observations) != tuple(
            item[0] for item in catalog
        ):
            raise ValueError("capture does not contain the ordered 73 sites")
        if len({item.site_id for item in self.observations}) != len(
            self.observations
        ):
            raise ValueError("capture site ids must be unique")
        if not isinstance(self.valid_mask, Tensor) or self.valid_mask.dtype != torch.bool:
            raise ValueError("capture valid_mask must be boolean")
        canonical_rows = _canonical_row_keys(
            self.row_keys,
            expected_count=int(self.valid_mask.sum().item()),
        )
        if canonical_rows != self.row_keys:
            raise ValueError("capture row_keys must already be canonical")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.execution_fingerprint,
            label="execution_fingerprint",
        )
        widths = {item.hidden_width for item in self.observations}
        if len(widths) != 1:
            raise ValueError("all unified event sites must share hidden width")
        for item in self.observations:
            if (
                item.valid_mask.shape != self.valid_mask.shape
                or not torch.equal(item.valid_mask, self.valid_mask)
                or item.row_keys != self.row_keys
                or item.source_fingerprint != self.source_fingerprint
                or item.execution_fingerprint != self.execution_fingerprint
            ):
                raise ValueError("capture sites do not share one row/execution identity")

    @property
    def hidden_width(self) -> int:
        return self.observations[0].hidden_width

    @property
    def row_identity_sha256(self) -> str:
        return _row_identity_sha256(self.row_keys)

    def site(self, site_id: str) -> UnifiedEventChartSiteObservation:
        for observation in self.observations:
            if observation.site_id == site_id:
                return observation
        raise KeyError(site_id)

    def event_site(
        self,
        event_id: str,
        role: str,
    ) -> UnifiedEventChartSiteObservation:
        return self.site(f"{event_id}.{role}")

    def metadata(self) -> dict[str, object]:
        return {
            "site_count": len(self.observations),
            "hidden_width": self.hidden_width,
            "valid_row_count": len(self.row_keys),
            "row_identity_sha256": self.row_identity_sha256,
            "source_fingerprint": self.source_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "observational_sidecar_only": True,
            "callback_result_enters_runtime": False,
            "all_values_detached_cpu_clones": True,
        }


def merge_unified_event_chart_captures(
    captures: Sequence[UnifiedEventChartCapture],
) -> UnifiedEventChartCapture:
    """Merge padded exact-run batches into one canonical all-valid capture."""

    if isinstance(captures, (str, bytes)) or not isinstance(captures, Sequence):
        raise TypeError("captures must be a materialized sequence")
    materialized = tuple(captures)
    if not materialized or any(
        not isinstance(item, UnifiedEventChartCapture) for item in materialized
    ):
        raise ValueError("captures must contain a UnifiedEventChartCapture")
    first = materialized[0]
    if any(
        item.schedule != first.schedule
        or item.source_fingerprint != first.source_fingerprint
        or item.execution_fingerprint != first.execution_fingerprint
        or item.hidden_width != first.hidden_width
        or tuple(site.site_id for site in item.observations)
        != tuple(site.site_id for site in first.observations)
        for item in materialized[1:]
    ):
        raise ValueError("captures do not share one schedule/source/execution role")
    row_keys = tuple(key for item in materialized for key in item.row_keys)
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("merged capture row keys must remain unique")
    valid_mask = torch.ones((1, len(row_keys)), dtype=torch.bool)
    observations: list[UnifiedEventChartSiteObservation] = []
    for site_index, template in enumerate(first.observations):
        rows = torch.cat(
            tuple(item.observations[site_index].valid_rows() for item in materialized),
            dim=0,
        ).unsqueeze(0).contiguous()
        observations.append(
            UnifiedEventChartSiteObservation(
                site_id=template.site_id,
                event_id=template.event_id,
                event_index=template.event_index,
                event_kind=template.event_kind,
                component_family=template.component_family,
                block_ordinal=template.block_ordinal,
                role=template.role,
                value_semantics=template.value_semantics,
                values=rows,
                valid_mask=valid_mask.clone(),
                row_keys=row_keys,
                source_fingerprint=first.source_fingerprint,
                execution_fingerprint=first.execution_fingerprint,
            )
        )
    return UnifiedEventChartCapture(
        schedule=first.schedule,
        observations=tuple(observations),
        valid_mask=valid_mask,
        row_keys=row_keys,
        source_fingerprint=first.source_fingerprint,
        execution_fingerprint=first.execution_fingerprint,
    )


class UnifiedEventChartRecorder:
    """Strict callback sink for one exact 39-event execution.

    The runtime owns when this callback is invoked.  ``record`` deliberately
    returns ``None`` so there is no value that can be fed back into execution.
    """

    def __init__(
        self,
        *,
        schedule: Sequence[JointGraphEventSpec],
        source_fingerprint: str,
        execution_fingerprint: str,
        valid_mask: Tensor,
        row_keys: Sequence[tuple[str, int]],
    ) -> None:
        self._schedule = _canonical_schedule(schedule)
        self._event_by_id = {item.event_id: item for item in self._schedule}
        self._source_fingerprint = _require_sha256(
            source_fingerprint,
            label="source_fingerprint",
        )
        self._execution_fingerprint = _require_sha256(
            execution_fingerprint,
            label="execution_fingerprint",
        )
        if (
            not isinstance(valid_mask, Tensor)
            or valid_mask.ndim != 2
            or valid_mask.dtype != torch.bool
            or not valid_mask.numel()
            or not bool(valid_mask.any())
        ):
            raise ValueError("valid_mask must contain a valid [B,S] row")
        self._valid_mask = valid_mask.detach().to(device="cpu").clone().contiguous()
        self._row_keys = _canonical_row_keys(
            row_keys,
            expected_count=int(self._valid_mask.sum().item()),
        )
        self._catalog = _expected_site_catalog(self._schedule)
        self._observations: dict[str, UnifiedEventChartSiteObservation] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def record(self, *, event_id: str, role: str, values: Tensor) -> None:
        if self._closed:
            raise RuntimeError("unified event chart recorder is closed")
        event = self._event_by_id.get(event_id)
        if event is None:
            raise ValueError("event_id is outside the exact schedule")
        semantics = _site_semantics(event, role)
        site_id = f"{event_id}.{role}"
        if site_id in self._observations:
            raise RuntimeError("unified event site was captured twice")
        if (
            not isinstance(values, Tensor)
            or values.ndim != 3
            or values.shape[:2] != self._valid_mask.shape
            or not values.shape[2]
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("site values must be finite floating [B,S,H]")
        detached = values.detach().to(device="cpu").clone().contiguous()
        self._observations[site_id] = UnifiedEventChartSiteObservation(
            site_id=site_id,
            event_id=event.event_id,
            event_index=event.event_index,
            event_kind=event.event_kind,
            component_family=event.component_family,
            block_ordinal=event.block_ordinal,
            role=role,
            value_semantics=semantics,
            values=detached,
            valid_mask=self._valid_mask.clone(),
            row_keys=self._row_keys,
            source_fingerprint=self._source_fingerprint,
            execution_fingerprint=self._execution_fingerprint,
        )
        return None

    __call__ = record

    def finish(self) -> UnifiedEventChartCapture:
        if self._closed:
            raise RuntimeError("unified event chart recorder is closed")
        expected = tuple(item[0] for item in self._catalog)
        missing = tuple(site_id for site_id in expected if site_id not in self._observations)
        unexpected = tuple(sorted(set(self._observations) - set(expected)))
        if missing or unexpected:
            raise RuntimeError(
                "unified event capture is incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self._closed = True
        return UnifiedEventChartCapture(
            schedule=self._schedule,
            observations=tuple(self._observations[site_id] for site_id in expected),
            valid_mask=self._valid_mask.clone(),
            row_keys=self._row_keys,
            source_fingerprint=self._source_fingerprint,
            execution_fingerprint=self._execution_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class UnifiedEventChartPacket:
    """Event-keyed chart coordinates used as an edge source."""

    event_id: str
    event_index: int
    event_kind: str
    component_family: str
    block_ordinal: int | None
    site_id: str
    site_role: str
    row_keys: tuple[tuple[str, int], ...]
    coordinates: Tensor
    source_fingerprint: str
    execution_fingerprint: str

    def __post_init__(self) -> None:
        _finite_packet(self.coordinates, label="coordinates")
        if self.coordinates.device.type != "cpu" or self.coordinates.requires_grad:
            raise ValueError("coordinates must be detached CPU values")
        if self.site_role not in _ROLES:
            raise ValueError("packet site_role must be input or output")
        if self.site_id != f"{self.event_id}.{self.site_role}":
            raise ValueError("packet site_id must bind its event and role")
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("event_index must be nonnegative")
        canonical = _canonical_row_keys(
            self.row_keys,
            expected_count=int(self.coordinates.shape[0]),
        )
        if canonical != self.row_keys:
            raise ValueError("packet row_keys must already be canonical")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.execution_fingerprint,
            label="execution_fingerprint",
        )

    @property
    def chart_count(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def coordinate_rank(self) -> int:
        return int(self.coordinates.shape[2])


def packet_from_unified_event_site(
    capture: UnifiedEventChartCapture,
    *,
    event_id: str,
    role: str,
    coordinates: Tensor | None = None,
) -> UnifiedEventChartPacket:
    """Bind chart coordinates to any eligible capture source site.

    Without an external chart encoder, the exact full-width output is exposed
    as one chart.  Supplying ``coordinates`` allows a shared or event-specific
    atlas to provide any finite ``[N,C,R]`` coordinate packet.
    """

    if not isinstance(capture, UnifiedEventChartCapture):
        raise TypeError("capture must be a UnifiedEventChartCapture")
    site = capture.event_site(event_id, role)
    if coordinates is None:
        resolved = site.valid_rows()[:, None, :]
    else:
        _finite_packet(coordinates, label="coordinates")
        if coordinates.shape[0] != len(capture.row_keys):
            raise ValueError("coordinates do not match capture rows")
        resolved = coordinates.detach().to(device="cpu").clone().contiguous()
    return UnifiedEventChartPacket(
        event_id=site.event_id,
        event_index=site.event_index,
        event_kind=site.event_kind,
        component_family=site.component_family,
        block_ordinal=site.block_ordinal,
        site_id=site.site_id,
        site_role=site.role,
        row_keys=site.row_keys,
        coordinates=resolved,
        source_fingerprint=site.source_fingerprint,
        execution_fingerprint=site.execution_fingerprint,
    )


def packet_from_unified_event_output(
    capture: UnifiedEventChartCapture,
    *,
    event_id: str,
    coordinates: Tensor | None = None,
) -> UnifiedEventChartPacket:
    """Compatibility shorthand for an event output source packet."""

    return packet_from_unified_event_site(
        capture,
        event_id=event_id,
        role="output",
        coordinates=coordinates,
    )


@dataclass(frozen=True, slots=True)
class UnifiedEventLocalBaseline:
    """Frozen target-local flow and the residual available to graph edges."""

    event_id: str
    event_index: int
    event_kind: str
    component_family: str
    block_ordinal: int | None
    row_keys: tuple[tuple[str, int], ...]
    target_flow: Tensor
    local_predicted_flow: Tensor
    target_memberships: Tensor
    row_fisher_weights: Tensor
    source_fingerprint: str
    execution_fingerprint: str

    def __post_init__(self) -> None:
        if self.event_kind not in _MUTATION_KINDS:
            raise ValueError("edge targets must be mutation events")
        _finite_matrix(self.target_flow, label="target_flow")
        _finite_matrix(self.local_predicted_flow, label="local_predicted_flow")
        if self.local_predicted_flow.shape != self.target_flow.shape:
            raise ValueError("local baseline must match target mutation flow")
        rows = int(self.target_flow.shape[0])
        if (
            not isinstance(self.target_memberships, Tensor)
            or self.target_memberships.ndim != 2
            or self.target_memberships.shape[0] != rows
            or not self.target_memberships.shape[1]
            or not self.target_memberships.is_floating_point()
            or not bool(torch.isfinite(self.target_memberships).all())
            or bool((self.target_memberships < 0).any())
            or not torch.allclose(
                self.target_memberships.sum(dim=1),
                torch.ones(rows, dtype=self.target_memberships.dtype),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise ValueError("target_memberships must be a finite row simplex")
        _normalized_row_weights(self.row_fisher_weights, row_count=rows)
        canonical = _canonical_row_keys(self.row_keys, expected_count=rows)
        if canonical != self.row_keys:
            raise ValueError("baseline row_keys must already be canonical")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.execution_fingerprint,
            label="execution_fingerprint",
        )

    @property
    def flow_width(self) -> int:
        return int(self.target_flow.shape[1])

    @property
    def chart_count(self) -> int:
        return int(self.target_memberships.shape[1])

    @property
    def residual_flow(self) -> Tensor:
        return self.target_flow - self.local_predicted_flow


def local_baseline_from_unified_event_output(
    capture: UnifiedEventChartCapture,
    *,
    event_id: str,
    local_predicted_flow: Tensor,
    target_memberships: Tensor | None = None,
    row_fisher_weights: Tensor | None = None,
) -> UnifiedEventLocalBaseline:
    """Bind a frozen local-chart baseline to an exact mutation output."""

    if not isinstance(capture, UnifiedEventChartCapture):
        raise TypeError("capture must be a UnifiedEventChartCapture")
    site = capture.event_site(event_id, "output")
    if site.event_kind not in _MUTATION_KINDS:
        raise ValueError("local baselines require a mutation output")
    target = site.valid_rows()
    _finite_matrix(local_predicted_flow, label="local_predicted_flow")
    if local_predicted_flow.shape != target.shape:
        raise ValueError("local_predicted_flow does not match exact target flow")
    rows = target.shape[0]
    memberships = (
        torch.ones((rows, 1), dtype=target.dtype)
        if target_memberships is None
        else target_memberships
    )
    weights = (
        torch.ones(rows, dtype=target.dtype)
        if row_fisher_weights is None
        else row_fisher_weights
    )
    return UnifiedEventLocalBaseline(
        event_id=site.event_id,
        event_index=site.event_index,
        event_kind=site.event_kind,
        component_family=site.component_family,
        block_ordinal=site.block_ordinal,
        row_keys=site.row_keys,
        target_flow=target,
        local_predicted_flow=(
            local_predicted_flow.detach().to(device="cpu").clone().contiguous()
        ),
        target_memberships=(
            memberships.detach().to(device="cpu").clone().contiguous()
        ),
        row_fisher_weights=(
            weights.detach().to(device="cpu").clone().contiguous()
        ),
        source_fingerprint=site.source_fingerprint,
        execution_fingerprint=site.execution_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class FittedUnifiedEventLocalBaseline:
    """Frozen full-width affine chart baseline for one mutation event."""

    event_id: str
    event_index: int
    event_kind: str
    component_family: str
    block_ordinal: int
    input_site_id: str
    output_site_id: str
    weight: Tensor
    bias: Tensor
    relative_ridge: float
    absolute_ridge: float
    fit_zero_weighted_sse: float
    fit_weighted_sse: float
    fit_row_keys: tuple[tuple[str, int], ...]
    fit_row_identity_sha256: str
    source_fingerprint: str
    fit_execution_fingerprint: str

    def __post_init__(self) -> None:
        if self.event_kind not in _MUTATION_KINDS:
            raise ValueError("local baseline must bind a mutation event")
        if self.input_site_id != f"{self.event_id}.input":
            raise ValueError("input_site_id does not bind the mutation")
        if self.output_site_id != f"{self.event_id}.output":
            raise ValueError("output_site_id does not bind the mutation")
        _finite_matrix(self.weight, label="weight")
        if (
            not isinstance(self.bias, Tensor)
            or self.bias.ndim != 1
            or self.bias.shape[0] != self.weight.shape[0]
            or not self.bias.is_floating_point()
            or not bool(torch.isfinite(self.bias).all())
        ):
            raise ValueError("bias must match the local flow width")
        if (
            not isinstance(self.relative_ridge, (int, float))
            or isinstance(self.relative_ridge, bool)
            or not math.isfinite(float(self.relative_ridge))
            or float(self.relative_ridge) <= 0.0
            or not math.isfinite(float(self.absolute_ridge))
            or float(self.absolute_ridge) <= 0.0
        ):
            raise ValueError("local baseline ridge must be finite and positive")
        if (
            not math.isfinite(float(self.fit_zero_weighted_sse))
            or float(self.fit_zero_weighted_sse) <= 0.0
            or not math.isfinite(float(self.fit_weighted_sse))
            or float(self.fit_weighted_sse) < 0.0
            or float(self.fit_weighted_sse)
            > float(self.fit_zero_weighted_sse)
        ):
            raise ValueError("local baseline fit loss is invalid")
        canonical_rows = _canonical_row_keys(
            self.fit_row_keys,
            expected_count=len(self.fit_row_keys),
        )
        if (
            canonical_rows != self.fit_row_keys
            or _row_identity_sha256(self.fit_row_keys)
            != self.fit_row_identity_sha256
        ):
            raise ValueError("local baseline fit row identity does not close")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.fit_execution_fingerprint,
            label="fit_execution_fingerprint",
        )

    @property
    def input_width(self) -> int:
        return int(self.weight.shape[1])

    @property
    def flow_width(self) -> int:
        return int(self.weight.shape[0])

    @property
    def stored_scalar_count(self) -> int:
        return int(self.weight.numel() + self.bias.numel())

    @property
    def fit_explained_fraction(self) -> float:
        return 1.0 - float(self.fit_weighted_sse) / float(
            self.fit_zero_weighted_sse
        )

    def predict(self, capture: UnifiedEventChartCapture) -> Tensor:
        if not isinstance(capture, UnifiedEventChartCapture):
            raise TypeError("capture must be a UnifiedEventChartCapture")
        if capture.source_fingerprint != self.source_fingerprint:
            raise ValueError("capture source differs from fitted local baseline")
        input_rows = capture.site(self.input_site_id).valid_rows()
        if input_rows.shape[1] != self.input_width:
            raise ValueError("capture input width differs from local baseline")
        weight = self.weight.to(device=input_rows.device, dtype=input_rows.dtype)
        bias = self.bias.to(device=input_rows.device, dtype=input_rows.dtype)
        return input_rows @ weight.T + bias

    def metadata(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "component_family": self.component_family,
            "block_ordinal": self.block_ordinal,
            "input_site_id": self.input_site_id,
            "output_site_id": self.output_site_id,
            "fit_algorithm": (
                "fisher_weighted_full_width_affine_ridge_"
                "unpenalized_intercept_v1"
            ),
            "input_width": self.input_width,
            "flow_width": self.flow_width,
            "stored_scalar_count": self.stored_scalar_count,
            "relative_ridge": float(self.relative_ridge),
            "absolute_ridge": float(self.absolute_ridge),
            "bias_penalized": False,
            "fit_zero_weighted_sse": float(self.fit_zero_weighted_sse),
            "fit_weighted_sse": float(self.fit_weighted_sse),
            "fit_explained_fraction": self.fit_explained_fraction,
            "fit_row_identity_sha256": self.fit_row_identity_sha256,
            "source_fingerprint": self.source_fingerprint,
            "fit_execution_fingerprint": self.fit_execution_fingerprint,
            "sidecar_only": True,
            "installed_in_exact_runtime": False,
        }


def fit_fisher_weighted_unified_event_local_baseline(
    *,
    capture: UnifiedEventChartCapture,
    event_id: str,
    row_fisher_weights: Tensor,
    relative_ridge: float,
) -> FittedUnifiedEventLocalBaseline:
    """Fit paired same-event normalized-input to exact residual-delta flow."""

    if not isinstance(capture, UnifiedEventChartCapture):
        raise TypeError("capture must be a UnifiedEventChartCapture")
    if (
        not isinstance(relative_ridge, (int, float))
        or isinstance(relative_ridge, bool)
        or not math.isfinite(float(relative_ridge))
        or float(relative_ridge) <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")
    input_site = capture.event_site(event_id, "input")
    output_site = capture.event_site(event_id, "output")
    if input_site.event_kind not in _MUTATION_KINDS:
        raise ValueError("local baseline event must be a mutation")
    inputs = input_site.valid_rows().double()
    target = output_site.valid_rows().double()
    if inputs.shape[0] != target.shape[0]:
        raise RuntimeError("paired local baseline rows drifted")
    weights = _normalized_row_weights(
        row_fisher_weights,
        row_count=inputs.shape[0],
    )
    weighted_inputs = weights[:, None] * inputs
    feature_gram = inputs.T @ weighted_inputs
    scale = float(torch.trace(feature_gram) / max(1, inputs.shape[1]))
    absolute_ridge = float(relative_ridge) * max(scale, 1.0e-12)
    design = torch.cat(
        (inputs, torch.ones((inputs.shape[0], 1), dtype=torch.float64)),
        dim=1,
    )
    gram = design.T @ (weights[:, None] * design)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * absolute_ridge
    penalty[-1, -1] = 0.0
    right_hand = design.T @ (weights[:, None] * target)
    coefficient = torch.linalg.solve(gram + penalty, right_hand)
    prediction = design @ coefficient
    zero_sse = float(_weighted_sse(target, weights))
    fitted_sse = float(_weighted_sse(target - prediction, weights))
    tolerance = 256.0 * torch.finfo(torch.float64).eps * max(1.0, zero_sse)
    if fitted_sse > zero_sse + tolerance:
        raise RuntimeError("local affine ridge increased fit loss")
    event = next(item for item in capture.schedule if item.event_id == event_id)
    assert event.block_ordinal is not None
    return FittedUnifiedEventLocalBaseline(
        event_id=event.event_id,
        event_index=event.event_index,
        event_kind=event.event_kind,
        component_family=event.component_family,
        block_ordinal=event.block_ordinal,
        input_site_id=input_site.site_id,
        output_site_id=output_site.site_id,
        weight=coefficient[:-1].T.to(dtype=torch.float32).contiguous(),
        bias=coefficient[-1].to(dtype=torch.float32).contiguous(),
        relative_ridge=float(relative_ridge),
        absolute_ridge=absolute_ridge,
        fit_zero_weighted_sse=zero_sse,
        fit_weighted_sse=fitted_sse,
        fit_row_keys=capture.row_keys,
        fit_row_identity_sha256=capture.row_identity_sha256,
        source_fingerprint=capture.source_fingerprint,
        fit_execution_fingerprint=capture.execution_fingerprint,
    )


def apply_fitted_unified_event_local_baseline(
    fit: FittedUnifiedEventLocalBaseline,
    *,
    capture: UnifiedEventChartCapture,
    row_fisher_weights: Tensor | None = None,
    target_memberships: Tensor | None = None,
) -> UnifiedEventLocalBaseline:
    """Apply a frozen local fit and bind its residual; never refits."""

    if not isinstance(fit, FittedUnifiedEventLocalBaseline):
        raise TypeError("fit must be a FittedUnifiedEventLocalBaseline")
    prediction = fit.predict(capture)
    return local_baseline_from_unified_event_output(
        capture,
        event_id=fit.event_id,
        local_predicted_flow=prediction,
        target_memberships=target_memberships,
        row_fisher_weights=row_fisher_weights,
    )


@dataclass(frozen=True, slots=True)
class UnifiedEventLocalBaselineEvaluation:
    """Held-out receipt for a frozen same-event local affine baseline."""

    event_id: str
    heldout_row_count: int
    heldout_row_identity_sha256: str
    zero_flow_weighted_sse: float
    local_baseline_weighted_sse: float
    local_baseline_nrmse: float
    explained_fraction: float
    fit_rows_disjoint: bool
    fit_execution_disjoint: bool

    def metadata(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "heldout_row_count": self.heldout_row_count,
            "heldout_row_identity_sha256": self.heldout_row_identity_sha256,
            "zero_flow_weighted_sse": self.zero_flow_weighted_sse,
            "local_baseline_weighted_sse": self.local_baseline_weighted_sse,
            "local_baseline_nrmse": self.local_baseline_nrmse,
            "explained_fraction": self.explained_fraction,
            "fit_rows_disjoint": self.fit_rows_disjoint,
            "fit_execution_disjoint": self.fit_execution_disjoint,
            "heldout_refit_performed": False,
        }


def evaluate_fitted_unified_event_local_baseline(
    fit: FittedUnifiedEventLocalBaseline,
    *,
    capture: UnifiedEventChartCapture,
    row_fisher_weights: Tensor | None = None,
    require_disjoint_from_fit: bool = True,
) -> UnifiedEventLocalBaselineEvaluation:
    """Evaluate a frozen local baseline on held-out exact captures."""

    if not isinstance(fit, FittedUnifiedEventLocalBaseline):
        raise TypeError("fit must be a FittedUnifiedEventLocalBaseline")
    if not isinstance(capture, UnifiedEventChartCapture):
        raise TypeError("capture must be a UnifiedEventChartCapture")
    if type(require_disjoint_from_fit) is not bool:
        raise TypeError("require_disjoint_from_fit must be boolean")
    fit_rows_disjoint = set(capture.row_keys).isdisjoint(fit.fit_row_keys)
    fit_execution_disjoint = (
        capture.execution_fingerprint != fit.fit_execution_fingerprint
    )
    if require_disjoint_from_fit and (
        not fit_rows_disjoint or not fit_execution_disjoint
    ):
        raise ValueError("held-out local evaluation reused fit rows or execution")
    target = capture.site(fit.output_site_id).valid_rows().double()
    prediction = fit.predict(capture).double()
    weights = _normalized_row_weights(
        (
            torch.ones(target.shape[0], dtype=torch.float32)
            if row_fisher_weights is None
            else row_fisher_weights
        ),
        row_count=target.shape[0],
    )
    zero_sse = float(_weighted_sse(target, weights))
    fitted_sse = float(_weighted_sse(target - prediction, weights))
    denominator = max(zero_sse, 1.0e-30)
    return UnifiedEventLocalBaselineEvaluation(
        event_id=fit.event_id,
        heldout_row_count=len(capture.row_keys),
        heldout_row_identity_sha256=capture.row_identity_sha256,
        zero_flow_weighted_sse=zero_sse,
        local_baseline_weighted_sse=fitted_sse,
        local_baseline_nrmse=math.sqrt(fitted_sse / denominator),
        explained_fraction=(zero_sse - fitted_sse) / denominator,
        fit_rows_disjoint=fit_rows_disjoint,
        fit_execution_disjoint=fit_execution_disjoint,
    )


@dataclass(frozen=True, slots=True)
class UnifiedEventEdgeSpec:
    """Authenticated causal endpoints for one chart edge."""

    source_event_id: str
    source_event_index: int
    source_event_kind: str
    source_component_family: str
    source_block_ordinal: int | None
    source_site_id: str
    source_site_role: str
    source_chart_index: int
    target_event_id: str
    target_event_index: int
    target_event_kind: str
    target_component_family: str
    target_block_ordinal: int | None
    target_chart_index: int

    def __post_init__(self) -> None:
        if self.source_event_index >= self.target_event_index:
            raise ValueError("unified event edges must be strictly forward")
        if type(self.source_chart_index) is not int or self.source_chart_index < 0:
            raise ValueError("source_chart_index must be nonnegative")
        if self.source_site_role not in _ROLES:
            raise ValueError("source_site_role must be input or output")
        if self.source_site_id != (
            f"{self.source_event_id}.{self.source_site_role}"
        ):
            raise ValueError("source_site_id does not bind its event and role")
        if type(self.target_chart_index) is not int or self.target_chart_index < 0:
            raise ValueError("target_chart_index must be nonnegative")
        if self.target_event_kind not in _MUTATION_KINDS:
            raise ValueError("unified event edge target must be a mutation")

    @property
    def cross_block(self) -> bool:
        source_block = (
            -1
            if self.source_event_kind == "lexical_ingress"
            else self.source_block_ordinal
        )
        return (
            source_block is not None
            and self.target_block_ordinal is not None
            and source_block != self.target_block_ordinal
        )

    @property
    def cross_family(self) -> bool:
        return self.source_component_family != self.target_component_family

    def metadata(self) -> dict[str, object]:
        return {
            "source_event_id": self.source_event_id,
            "source_event_index": self.source_event_index,
            "source_event_kind": self.source_event_kind,
            "source_component_family": self.source_component_family,
            "source_block_ordinal": self.source_block_ordinal,
            "source_site_id": self.source_site_id,
            "source_site_role": self.source_site_role,
            "source_chart_index": self.source_chart_index,
            "target_event_id": self.target_event_id,
            "target_event_index": self.target_event_index,
            "target_event_kind": self.target_event_kind,
            "target_component_family": self.target_component_family,
            "target_block_ordinal": self.target_block_ordinal,
            "target_chart_index": self.target_chart_index,
            "strictly_forward": True,
            "cross_block": self.cross_block,
            "lexical_ingress_virtual_block_ordinal": (
                -1 if self.source_event_kind == "lexical_ingress" else None
            ),
            "cross_family": self.cross_family,
            "event_or_layer_features_in_numerical_design": False,
        }


def _resolve_edge_spec(
    *,
    schedule: Sequence[JointGraphEventSpec],
    source: UnifiedEventChartPacket,
    target: UnifiedEventLocalBaseline,
    source_chart_index: int,
    target_chart_index: int,
    require_cross_block: bool,
    require_cross_family: bool,
) -> UnifiedEventEdgeSpec:
    canonical = _canonical_schedule(schedule)
    event_by_id = {item.event_id: item for item in canonical}
    source_event = event_by_id.get(source.event_id)
    target_event = event_by_id.get(target.event_id)
    if source_event is None or target_event is None:
        raise ValueError("edge endpoint is outside the exact event schedule")
    if (
        source.event_index != source_event.event_index
        or source.event_kind != source_event.event_kind
        or source.component_family != source_event.component_family
        or source.block_ordinal != source_event.block_ordinal
        or target.event_index != target_event.event_index
        or target.event_kind != target_event.event_kind
        or target.component_family != target_event.component_family
        or target.block_ordinal != target_event.block_ordinal
    ):
        raise ValueError("edge endpoint metadata disagrees with the schedule")
    if source_chart_index >= source.chart_count:
        raise ValueError("source_chart_index is outside the packet")
    if target_chart_index >= target.chart_count:
        raise ValueError("target_chart_index is outside the local baseline")
    spec = UnifiedEventEdgeSpec(
        source_event_id=source_event.event_id,
        source_event_index=source_event.event_index,
        source_event_kind=source_event.event_kind,
        source_component_family=source_event.component_family,
        source_block_ordinal=source_event.block_ordinal,
        source_site_id=source.site_id,
        source_site_role=source.site_role,
        source_chart_index=source_chart_index,
        target_event_id=target_event.event_id,
        target_event_index=target_event.event_index,
        target_event_kind=target_event.event_kind,
        target_component_family=target_event.component_family,
        target_block_ordinal=target_event.block_ordinal,
        target_chart_index=target_chart_index,
    )
    if require_cross_block and not spec.cross_block:
        raise ValueError("edge is not cross-block")
    if require_cross_family and not spec.cross_family:
        raise ValueError("edge is not cross-family")
    return spec


@dataclass(frozen=True, slots=True)
class FittedUnifiedEventEdge:
    """Source-independent low-rank predictor fitted outside exact execution."""

    spec: UnifiedEventEdgeSpec
    left_factor: Tensor
    right_factor: Tensor
    relative_ridge: float
    absolute_ridge: float
    fit_baseline_weighted_sse: float
    fit_edge_weighted_sse: float
    fit_row_keys: tuple[tuple[str, int], ...]
    fit_row_identity_sha256: str
    source_fingerprint: str
    fit_execution_fingerprint: str

    def __post_init__(self) -> None:
        _finite_matrix(self.left_factor, label="left_factor")
        _finite_matrix(self.right_factor, label="right_factor")
        if self.left_factor.shape[1] != self.right_factor.shape[0]:
            raise ValueError("low-rank edge factors do not contract")
        if (
            not isinstance(self.relative_ridge, (int, float))
            or isinstance(self.relative_ridge, bool)
            or not math.isfinite(float(self.relative_ridge))
            or float(self.relative_ridge) <= 0.0
            or not math.isfinite(float(self.absolute_ridge))
            or float(self.absolute_ridge) <= 0.0
        ):
            raise ValueError("edge ridge values must be finite and positive")
        if (
            not math.isfinite(float(self.fit_baseline_weighted_sse))
            or float(self.fit_baseline_weighted_sse) <= 0.0
            or not math.isfinite(float(self.fit_edge_weighted_sse))
            or float(self.fit_edge_weighted_sse) < 0.0
            or float(self.fit_edge_weighted_sse)
            >= float(self.fit_baseline_weighted_sse)
        ):
            raise ValueError("fitted edge must improve the local baseline")
        _require_sha256(
            self.fit_row_identity_sha256,
            label="fit_row_identity_sha256",
        )
        canonical_rows = _canonical_row_keys(
            self.fit_row_keys,
            expected_count=len(self.fit_row_keys),
        )
        if (
            canonical_rows != self.fit_row_keys
            or _row_identity_sha256(self.fit_row_keys)
            != self.fit_row_identity_sha256
        ):
            raise ValueError("fit row identity does not close")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.fit_execution_fingerprint,
            label="fit_execution_fingerprint",
        )

    @property
    def source_rank(self) -> int:
        return int(self.left_factor.shape[0])

    @property
    def edge_rank(self) -> int:
        return int(self.left_factor.shape[1])

    @property
    def target_flow_width(self) -> int:
        return int(self.right_factor.shape[1])

    @property
    def stored_scalar_count(self) -> int:
        return int(self.left_factor.numel() + self.right_factor.numel())

    @property
    def fit_explained_fraction(self) -> float:
        return 1.0 - (
            float(self.fit_edge_weighted_sse)
            / float(self.fit_baseline_weighted_sse)
        )

    def predict(
        self,
        source: UnifiedEventChartPacket,
        target_memberships: Tensor,
    ) -> Tensor:
        if (
            source.event_id != self.spec.source_event_id
            or source.site_id != self.spec.source_site_id
        ):
            raise ValueError("source packet site does not match fitted edge")
        if source.coordinate_rank != self.source_rank:
            raise ValueError("source packet rank does not match fitted edge")
        if self.spec.source_chart_index >= source.chart_count:
            raise ValueError("source chart is absent from packet")
        rows = source.coordinates.shape[0]
        if (
            not isinstance(target_memberships, Tensor)
            or target_memberships.ndim != 2
            or target_memberships.shape[0] != rows
            or self.spec.target_chart_index >= target_memberships.shape[1]
        ):
            raise ValueError("target memberships do not match fitted edge")
        design = (
            source.coordinates[:, self.spec.source_chart_index, :]
            * target_memberships[:, self.spec.target_chart_index, None]
        )
        left = self.left_factor.to(device=design.device, dtype=design.dtype)
        right = self.right_factor.to(device=design.device, dtype=design.dtype)
        return design @ left @ right

    def metadata(self) -> dict[str, object]:
        return {
            **self.spec.metadata(),
            "fit_algorithm": "fisher_weighted_ridge_then_low_rank_svd_v1",
            "edge_low_rank_factorization_used": True,
            "source_rank": self.source_rank,
            "edge_rank": self.edge_rank,
            "target_flow_width": self.target_flow_width,
            "stored_scalar_count": self.stored_scalar_count,
            "relative_ridge": float(self.relative_ridge),
            "absolute_ridge": float(self.absolute_ridge),
            "fit_baseline_weighted_sse": float(
                self.fit_baseline_weighted_sse
            ),
            "fit_edge_weighted_sse": float(self.fit_edge_weighted_sse),
            "fit_explained_fraction": self.fit_explained_fraction,
            "fit_row_identity_sha256": self.fit_row_identity_sha256,
            "source_fingerprint": self.source_fingerprint,
            "fit_execution_fingerprint": self.fit_execution_fingerprint,
            "sidecar_only": True,
            "installed_in_exact_runtime": False,
        }


@dataclass(frozen=True, slots=True)
class FittedUnifiedEventDenseEdge:
    """One dense rectangular ridge map for the frozen full-width protocol."""

    spec: UnifiedEventEdgeSpec
    coefficient: Tensor
    relative_ridge: float
    absolute_ridge: float
    fit_baseline_weighted_sse: float
    fit_edge_weighted_sse: float
    fit_row_keys: tuple[tuple[str, int], ...]
    fit_row_identity_sha256: str
    source_fingerprint: str
    fit_execution_fingerprint: str

    def __post_init__(self) -> None:
        _finite_matrix(self.coefficient, label="coefficient")
        if self.coefficient.device.type != "cpu" or self.coefficient.requires_grad:
            raise ValueError("dense edge coefficient must be detached CPU data")
        if (
            not isinstance(self.relative_ridge, (int, float))
            or isinstance(self.relative_ridge, bool)
            or not math.isfinite(float(self.relative_ridge))
            or float(self.relative_ridge) <= 0.0
            or not math.isfinite(float(self.absolute_ridge))
            or float(self.absolute_ridge) <= 0.0
        ):
            raise ValueError("dense edge ridge values must be finite and positive")
        if (
            not math.isfinite(float(self.fit_baseline_weighted_sse))
            or float(self.fit_baseline_weighted_sse) <= 0.0
            or not math.isfinite(float(self.fit_edge_weighted_sse))
            or float(self.fit_edge_weighted_sse) < 0.0
            or float(self.fit_edge_weighted_sse)
            >= float(self.fit_baseline_weighted_sse)
        ):
            raise ValueError("fitted dense edge must improve the local baseline")
        canonical_rows = _canonical_row_keys(
            self.fit_row_keys,
            expected_count=len(self.fit_row_keys),
        )
        if (
            canonical_rows != self.fit_row_keys
            or _row_identity_sha256(self.fit_row_keys)
            != self.fit_row_identity_sha256
        ):
            raise ValueError("dense edge fit row identity does not close")
        _require_sha256(self.source_fingerprint, label="source_fingerprint")
        _require_sha256(
            self.fit_execution_fingerprint,
            label="fit_execution_fingerprint",
        )

    @property
    def source_rank(self) -> int:
        return int(self.coefficient.shape[0])

    @property
    def target_flow_width(self) -> int:
        return int(self.coefficient.shape[1])

    @property
    def edge_rank(self) -> int:
        return min(self.source_rank, self.target_flow_width)

    def _numerical_rank_receipt(self) -> tuple[int, float]:
        singular = torch.linalg.svdvals(self.coefficient.double())
        largest = float(singular[0]) if singular.numel() else 0.0
        tolerance = (
            max(self.coefficient.shape)
            * torch.finfo(self.coefficient.dtype).eps
            * largest
        )
        return int((singular > tolerance).sum().item()), tolerance

    @property
    def numerical_rank_tolerance(self) -> float:
        return self._numerical_rank_receipt()[1]

    @property
    def realized_numerical_rank(self) -> int:
        return self._numerical_rank_receipt()[0]

    @property
    def stored_scalar_count(self) -> int:
        return int(self.coefficient.numel())

    @property
    def fit_explained_fraction(self) -> float:
        return 1.0 - float(self.fit_edge_weighted_sse) / float(
            self.fit_baseline_weighted_sse
        )

    def predict(
        self,
        source: UnifiedEventChartPacket,
        target_memberships: Tensor,
    ) -> Tensor:
        if (
            source.event_id != self.spec.source_event_id
            or source.site_id != self.spec.source_site_id
        ):
            raise ValueError("source packet site does not match fitted edge")
        if source.coordinate_rank != self.source_rank:
            raise ValueError("source packet rank does not match fitted edge")
        if self.spec.source_chart_index >= source.chart_count:
            raise ValueError("source chart is absent from packet")
        rows = source.coordinates.shape[0]
        if (
            not isinstance(target_memberships, Tensor)
            or target_memberships.ndim != 2
            or target_memberships.shape[0] != rows
            or self.spec.target_chart_index >= target_memberships.shape[1]
        ):
            raise ValueError("target memberships do not match fitted edge")
        design = (
            source.coordinates[:, self.spec.source_chart_index, :]
            * target_memberships[:, self.spec.target_chart_index, None]
        )
        coefficient = self.coefficient.to(
            device=design.device,
            dtype=design.dtype,
        )
        return design @ coefficient

    def metadata(self) -> dict[str, object]:
        realized_rank, rank_tolerance = self._numerical_rank_receipt()
        return {
            **self.spec.metadata(),
            "fit_algorithm": "fisher_weighted_dense_rectangular_ridge_v1",
            "edge_low_rank_factorization_used": False,
            "source_rank": self.source_rank,
            "edge_rank": self.edge_rank,
            "edge_capacity_rank": self.edge_rank,
            "realized_numerical_rank": realized_rank,
            "numerical_rank_tolerance": rank_tolerance,
            "numerical_rank_tolerance_rule": (
                "max(matrix_shape)*stored_dtype_epsilon*largest_singular_value"
            ),
            "target_flow_width": self.target_flow_width,
            "coefficient_shape": [self.source_rank, self.target_flow_width],
            "stored_scalar_count": self.stored_scalar_count,
            "relative_ridge": float(self.relative_ridge),
            "absolute_ridge": float(self.absolute_ridge),
            "fit_baseline_weighted_sse": float(
                self.fit_baseline_weighted_sse
            ),
            "fit_edge_weighted_sse": float(self.fit_edge_weighted_sse),
            "fit_explained_fraction": self.fit_explained_fraction,
            "fit_row_identity_sha256": self.fit_row_identity_sha256,
            "source_fingerprint": self.source_fingerprint,
            "fit_execution_fingerprint": self.fit_execution_fingerprint,
            "sidecar_only": True,
            "installed_in_exact_runtime": False,
        }


def _deterministic_factor_signs_(left: Tensor, right: Tensor) -> None:
    for component in range(left.shape[1]):
        vector = left[:, component]
        pivot = int(vector.abs().argmax())
        if float(vector[pivot]) < 0.0:
            left[:, component].mul_(-1.0)
            right[component].mul_(-1.0)


def fit_forward_unified_event_edge(
    *,
    schedule: Sequence[JointGraphEventSpec],
    source: UnifiedEventChartPacket,
    target: UnifiedEventLocalBaseline,
    edge_rank: int,
    relative_ridge: float,
    source_chart_index: int = 0,
    target_chart_index: int = 0,
    require_cross_block: bool = True,
    require_cross_family: bool = True,
) -> FittedUnifiedEventEdge:
    """Fit one causal chart edge to target flow beyond its local baseline."""

    if not isinstance(source, UnifiedEventChartPacket):
        raise TypeError("source must be a UnifiedEventChartPacket")
    if not isinstance(target, UnifiedEventLocalBaseline):
        raise TypeError("target must be a UnifiedEventLocalBaseline")
    if type(require_cross_block) is not bool or type(require_cross_family) is not bool:
        raise TypeError("edge eligibility flags must be boolean")
    if source.row_keys != target.row_keys:
        raise ValueError("source and target rows are not aligned")
    if (
        source.source_fingerprint != target.source_fingerprint
        or source.execution_fingerprint != target.execution_fingerprint
    ):
        raise ValueError("source and target do not belong to one exact execution")
    if (
        type(edge_rank) is not int
        or edge_rank <= 0
        or edge_rank > min(source.coordinate_rank, target.flow_width)
    ):
        raise ValueError("edge_rank is outside rectangular edge capacity")
    if (
        not isinstance(relative_ridge, (int, float))
        or isinstance(relative_ridge, bool)
        or not math.isfinite(float(relative_ridge))
        or float(relative_ridge) <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")
    spec = _resolve_edge_spec(
        schedule=schedule,
        source=source,
        target=target,
        source_chart_index=source_chart_index,
        target_chart_index=target_chart_index,
        require_cross_block=require_cross_block,
        require_cross_family=require_cross_family,
    )

    weights = _normalized_row_weights(
        target.row_fisher_weights,
        row_count=len(target.row_keys),
    )
    design = (
        source.coordinates[:, source_chart_index, :].double()
        * target.target_memberships[:, target_chart_index, None].double()
    )
    residual = target.residual_flow.double()
    baseline_sse = float(_weighted_sse(residual, weights))
    if not math.isfinite(baseline_sse) or baseline_sse <= 0.0:
        raise ValueError("target local baseline leaves no finite flow residual")
    gram = design.T @ (weights[:, None] * design)
    scale = float(torch.trace(gram) / max(1, gram.shape[0]))
    absolute_ridge = float(relative_ridge) * max(scale, 1.0e-12)
    right_hand = design.T @ (weights[:, None] * residual)
    coefficient = torch.linalg.solve(
        gram + absolute_ridge * torch.eye(gram.shape[0], dtype=torch.float64),
        right_hand,
    )
    left_u, singular, right_vh = torch.linalg.svd(
        coefficient,
        full_matrices=False,
    )
    root = singular[:edge_rank].clamp_min(0.0).sqrt()
    left = left_u[:, :edge_rank] * root.unsqueeze(0)
    right = root.unsqueeze(1) * right_vh[:edge_rank]
    _deterministic_factor_signs_(left, right)
    prediction = design @ left @ right
    fitted_sse = float(_weighted_sse(residual - prediction, weights))
    tolerance = 256.0 * torch.finfo(torch.float64).eps * max(1.0, baseline_sse)
    if fitted_sse >= baseline_sse - tolerance:
        raise RuntimeError("low-rank edge does not improve the target-local baseline")
    return FittedUnifiedEventEdge(
        spec=spec,
        left_factor=left.to(dtype=torch.float32).contiguous(),
        right_factor=right.to(dtype=torch.float32).contiguous(),
        relative_ridge=float(relative_ridge),
        absolute_ridge=absolute_ridge,
        fit_baseline_weighted_sse=baseline_sse,
        fit_edge_weighted_sse=fitted_sse,
        fit_row_keys=source.row_keys,
        fit_row_identity_sha256=_row_identity_sha256(source.row_keys),
        source_fingerprint=source.source_fingerprint,
        fit_execution_fingerprint=source.execution_fingerprint,
    )


def fit_forward_unified_event_dense_edge(
    *,
    schedule: Sequence[JointGraphEventSpec],
    source: UnifiedEventChartPacket,
    target: UnifiedEventLocalBaseline,
    relative_ridge: float,
    source_chart_index: int = 0,
    target_chart_index: int = 0,
    require_cross_block: bool = True,
    require_cross_family: bool = True,
) -> FittedUnifiedEventDenseEdge:
    """Fit one dense causal ridge map without any low-rank factorization."""

    if not isinstance(source, UnifiedEventChartPacket):
        raise TypeError("source must be a UnifiedEventChartPacket")
    if not isinstance(target, UnifiedEventLocalBaseline):
        raise TypeError("target must be a UnifiedEventLocalBaseline")
    if (
        type(require_cross_block) is not bool
        or type(require_cross_family) is not bool
    ):
        raise TypeError("edge eligibility flags must be boolean")
    if source.row_keys != target.row_keys:
        raise ValueError("source and target rows are not aligned")
    if (
        source.source_fingerprint != target.source_fingerprint
        or source.execution_fingerprint != target.execution_fingerprint
    ):
        raise ValueError("source and target do not belong to one exact execution")
    if (
        not isinstance(relative_ridge, (int, float))
        or isinstance(relative_ridge, bool)
        or not math.isfinite(float(relative_ridge))
        or float(relative_ridge) <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")
    spec = _resolve_edge_spec(
        schedule=schedule,
        source=source,
        target=target,
        source_chart_index=source_chart_index,
        target_chart_index=target_chart_index,
        require_cross_block=require_cross_block,
        require_cross_family=require_cross_family,
    )
    weights = _normalized_row_weights(
        target.row_fisher_weights,
        row_count=len(target.row_keys),
    )
    design = (
        source.coordinates[:, source_chart_index, :].double()
        * target.target_memberships[:, target_chart_index, None].double()
    )
    residual = target.residual_flow.double()
    baseline_sse = float(_weighted_sse(residual, weights))
    if not math.isfinite(baseline_sse) or baseline_sse <= 0.0:
        raise ValueError("target local baseline leaves no finite flow residual")
    gram = design.T @ (weights[:, None] * design)
    scale = float(torch.trace(gram) / max(1, gram.shape[0]))
    absolute_ridge = float(relative_ridge) * max(scale, 1.0e-12)
    right_hand = design.T @ (weights[:, None] * residual)
    coefficient = torch.linalg.solve(
        gram + absolute_ridge * torch.eye(gram.shape[0], dtype=torch.float64),
        right_hand,
    )
    prediction = design @ coefficient
    fitted_sse = float(_weighted_sse(residual - prediction, weights))
    tolerance = 256.0 * torch.finfo(torch.float64).eps * max(1.0, baseline_sse)
    if fitted_sse >= baseline_sse - tolerance:
        raise RuntimeError("dense edge does not improve the target-local baseline")
    return FittedUnifiedEventDenseEdge(
        spec=spec,
        coefficient=coefficient.to(dtype=torch.float32).contiguous(),
        relative_ridge=float(relative_ridge),
        absolute_ridge=absolute_ridge,
        fit_baseline_weighted_sse=baseline_sse,
        fit_edge_weighted_sse=fitted_sse,
        fit_row_keys=source.row_keys,
        fit_row_identity_sha256=_row_identity_sha256(source.row_keys),
        source_fingerprint=source.source_fingerprint,
        fit_execution_fingerprint=source.execution_fingerprint,
    )


def fit_forward_unified_event_edges(
    *,
    schedule: Sequence[JointGraphEventSpec],
    sources: Mapping[str, UnifiedEventChartPacket],
    targets: Mapping[str, UnifiedEventLocalBaseline],
    edge_rank: int,
    relative_ridge: float,
    require_cross_block: bool = True,
    require_cross_family: bool = True,
) -> tuple[FittedUnifiedEventEdge, ...]:
    """Fit every eligible positive-gain source-chart/target-chart edge.

    This is a candidate catalog, not a held-out selector.  Callers must use the
    held-out evaluator before choosing or installing any edge elsewhere.
    """

    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must contain an event packet")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must contain a local baseline")
    canonical = _canonical_schedule(schedule)
    order = {item.event_id: item.event_index for item in canonical}
    if any(key != value.site_id for key, value in sources.items()):
        raise ValueError("source mapping keys must be site ids")
    if any(key != value.event_id for key, value in targets.items()):
        raise ValueError("target mapping keys must be event ids")
    fitted: list[FittedUnifiedEventEdge] = []
    for target_id in sorted(targets, key=lambda item: order.get(item, -1)):
        target = targets[target_id]
        for source_id in sorted(
            sources,
            key=lambda item: (
                order.get(sources[item].event_id, -1),
                sources[item].site_role,
            ),
        ):
            source = sources[source_id]
            for source_chart in range(source.chart_count):
                for target_chart in range(target.chart_count):
                    try:
                        edge = fit_forward_unified_event_edge(
                            schedule=canonical,
                            source=source,
                            target=target,
                            edge_rank=edge_rank,
                            relative_ridge=relative_ridge,
                            source_chart_index=source_chart,
                            target_chart_index=target_chart,
                            require_cross_block=require_cross_block,
                            require_cross_family=require_cross_family,
                        )
                    except ValueError as error:
                        if str(error) in {
                            "unified event edges must be strictly forward",
                            "edge is not cross-block",
                            "edge is not cross-family",
                        }:
                            continue
                        raise
                    except RuntimeError as error:
                        if str(error) == (
                            "low-rank edge does not improve the "
                            "target-local baseline"
                        ):
                            continue
                        raise
                    fitted.append(edge)
    return tuple(
        sorted(
            fitted,
            key=lambda item: (
                item.spec.target_event_index,
                item.spec.target_chart_index,
                item.spec.source_event_index,
                item.spec.source_chart_index,
            ),
        )
    )


def fit_forward_unified_event_dense_edges(
    *,
    schedule: Sequence[JointGraphEventSpec],
    sources: Mapping[str, UnifiedEventChartPacket],
    targets: Mapping[str, UnifiedEventLocalBaseline],
    relative_ridge: float,
    require_cross_block: bool = True,
    require_cross_family: bool = True,
) -> tuple[FittedUnifiedEventDenseEdge, ...]:
    """Catalog every positive eligible dense full-map edge candidate."""

    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must contain an event packet")
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must contain a local baseline")
    canonical = _canonical_schedule(schedule)
    order = {item.event_id: item.event_index for item in canonical}
    if any(key != value.site_id for key, value in sources.items()):
        raise ValueError("source mapping keys must be site ids")
    if any(key != value.event_id for key, value in targets.items()):
        raise ValueError("target mapping keys must be event ids")
    fitted: list[FittedUnifiedEventDenseEdge] = []
    for target_id in sorted(targets, key=lambda item: order.get(item, -1)):
        target = targets[target_id]
        for source_id in sorted(
            sources,
            key=lambda item: (
                order.get(sources[item].event_id, -1),
                sources[item].site_role,
            ),
        ):
            source = sources[source_id]
            for source_chart in range(source.chart_count):
                for target_chart in range(target.chart_count):
                    try:
                        edge = fit_forward_unified_event_dense_edge(
                            schedule=canonical,
                            source=source,
                            target=target,
                            relative_ridge=relative_ridge,
                            source_chart_index=source_chart,
                            target_chart_index=target_chart,
                            require_cross_block=require_cross_block,
                            require_cross_family=require_cross_family,
                        )
                    except ValueError as error:
                        if str(error) in {
                            "unified event edges must be strictly forward",
                            "edge is not cross-block",
                            "edge is not cross-family",
                        }:
                            continue
                        raise
                    except RuntimeError as error:
                        if str(error) == (
                            "dense edge does not improve the "
                            "target-local baseline"
                        ):
                            continue
                        raise
                    fitted.append(edge)
    return tuple(
        sorted(
            fitted,
            key=lambda item: (
                item.spec.target_event_index,
                item.spec.target_chart_index,
                item.spec.source_event_index,
                item.spec.source_site_role,
                item.spec.source_chart_index,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class UnifiedEventEdgeEvaluation:
    """Held-out real/zero/shuffled receipt for one frozen edge."""

    spec: UnifiedEventEdgeSpec
    heldout_row_count: int
    heldout_row_identity_sha256: str
    baseline_zero_edge_weighted_sse: float
    real_edge_weighted_sse: float
    shuffled_source_weighted_sse: float
    real_edge_residual_nrmse: float
    shuffled_source_residual_nrmse: float
    real_edge_gain_fraction: float
    shuffled_source_gain_fraction: float
    shuffle_permutation_sha256: str
    fit_rows_disjoint: bool
    fit_execution_disjoint: bool

    @property
    def real_beats_zero(self) -> bool:
        return self.real_edge_weighted_sse < self.baseline_zero_edge_weighted_sse

    @property
    def real_beats_shuffle(self) -> bool:
        return self.real_edge_weighted_sse < self.shuffled_source_weighted_sse

    def metadata(self) -> dict[str, object]:
        return {
            **self.spec.metadata(),
            "heldout_row_count": self.heldout_row_count,
            "heldout_row_identity_sha256": self.heldout_row_identity_sha256,
            "baseline_zero_edge_weighted_sse": (
                self.baseline_zero_edge_weighted_sse
            ),
            "real_edge_weighted_sse": self.real_edge_weighted_sse,
            "shuffled_source_weighted_sse": self.shuffled_source_weighted_sse,
            "real_edge_residual_nrmse": self.real_edge_residual_nrmse,
            "shuffled_source_residual_nrmse": (
                self.shuffled_source_residual_nrmse
            ),
            "real_edge_gain_fraction": self.real_edge_gain_fraction,
            "shuffled_source_gain_fraction": self.shuffled_source_gain_fraction,
            "shuffle_permutation_sha256": self.shuffle_permutation_sha256,
            "fit_rows_disjoint": self.fit_rows_disjoint,
            "fit_execution_disjoint": self.fit_execution_disjoint,
            "real_beats_zero": self.real_beats_zero,
            "real_beats_shuffle": self.real_beats_shuffle,
            "heldout_refit_performed": False,
        }


def evaluate_unified_event_edge(
    edge: FittedUnifiedEventEdge | FittedUnifiedEventDenseEdge,
    *,
    source: UnifiedEventChartPacket,
    target: UnifiedEventLocalBaseline,
    shuffle_permutation: Tensor | None = None,
    require_disjoint_from_fit: bool = True,
) -> UnifiedEventEdgeEvaluation:
    """Score one frozen edge on held-out real, zero, and shuffled controls."""

    if not isinstance(
        edge,
        (FittedUnifiedEventEdge, FittedUnifiedEventDenseEdge),
    ):
        raise TypeError("edge must be a fitted unified event edge")
    if not isinstance(source, UnifiedEventChartPacket):
        raise TypeError("source must be a UnifiedEventChartPacket")
    if not isinstance(target, UnifiedEventLocalBaseline):
        raise TypeError("target must be a UnifiedEventLocalBaseline")
    if type(require_disjoint_from_fit) is not bool:
        raise TypeError("require_disjoint_from_fit must be boolean")
    if source.row_keys != target.row_keys:
        raise ValueError("held-out source and target rows are not aligned")
    if source.source_fingerprint != edge.source_fingerprint:
        raise ValueError("held-out source model differs from fitted edge")
    if (
        target.source_fingerprint != edge.source_fingerprint
        or source.execution_fingerprint != target.execution_fingerprint
    ):
        raise ValueError("held-out endpoints do not share source/execution identity")
    heldout_sha = _row_identity_sha256(source.row_keys)
    fit_rows_disjoint = set(source.row_keys).isdisjoint(edge.fit_row_keys)
    fit_execution_disjoint = (
        source.execution_fingerprint != edge.fit_execution_fingerprint
    )
    if require_disjoint_from_fit and (
        not fit_rows_disjoint or not fit_execution_disjoint
    ):
        raise ValueError("held-out evaluation reused fit rows or execution")
    if (
        source.event_id != edge.spec.source_event_id
        or source.site_id != edge.spec.source_site_id
    ):
        raise ValueError("held-out source site differs from fitted edge")
    if target.event_id != edge.spec.target_event_id:
        raise ValueError("held-out target event differs from fitted edge")
    if target.flow_width != edge.target_flow_width:
        raise ValueError("held-out target flow width drifted")
    rows = len(source.row_keys)
    if rows < 2:
        raise ValueError("shuffled control requires at least two held-out rows")
    if shuffle_permutation is None:
        permutation = torch.roll(torch.arange(rows), shifts=1)
    else:
        if (
            not isinstance(shuffle_permutation, Tensor)
            or shuffle_permutation.ndim != 1
            or shuffle_permutation.shape[0] != rows
            or shuffle_permutation.dtype != torch.long
            or tuple(sorted(int(item) for item in shuffle_permutation.tolist()))
            != tuple(range(rows))
        ):
            raise ValueError("shuffle_permutation must permute all held-out rows")
        permutation = shuffle_permutation.detach().cpu().clone().contiguous()
    weights = _normalized_row_weights(
        target.row_fisher_weights,
        row_count=rows,
    )
    residual = target.residual_flow.double()
    real_prediction = edge.predict(source, target.target_memberships).double()
    shuffled_design = (
        source.coordinates[permutation, edge.spec.source_chart_index, :]
        * target.target_memberships[:, edge.spec.target_chart_index, None]
    )
    if isinstance(edge, FittedUnifiedEventDenseEdge):
        shuffled_prediction = (
            shuffled_design
            @ edge.coefficient.to(dtype=shuffled_design.dtype)
        ).double()
    else:
        shuffled_prediction = (
            shuffled_design
            @ edge.left_factor.to(dtype=shuffled_design.dtype)
            @ edge.right_factor.to(dtype=shuffled_design.dtype)
        ).double()
    zero_sse = float(_weighted_sse(residual, weights))
    real_sse = float(_weighted_sse(residual - real_prediction, weights))
    shuffled_sse = float(_weighted_sse(residual - shuffled_prediction, weights))
    denominator = max(zero_sse, 1.0e-30)
    permutation_sha = hashlib.sha256(
        b"fisher_graph:unified-event-edge-shuffle:v1\0"
        + permutation.numpy().tobytes()
    ).hexdigest()
    return UnifiedEventEdgeEvaluation(
        spec=edge.spec,
        heldout_row_count=rows,
        heldout_row_identity_sha256=heldout_sha,
        baseline_zero_edge_weighted_sse=zero_sse,
        real_edge_weighted_sse=real_sse,
        shuffled_source_weighted_sse=shuffled_sse,
        real_edge_residual_nrmse=math.sqrt(real_sse / denominator),
        shuffled_source_residual_nrmse=math.sqrt(shuffled_sse / denominator),
        real_edge_gain_fraction=(zero_sse - real_sse) / denominator,
        shuffled_source_gain_fraction=(zero_sse - shuffled_sse) / denominator,
        shuffle_permutation_sha256=permutation_sha,
        fit_rows_disjoint=fit_rows_disjoint,
        fit_execution_disjoint=fit_execution_disjoint,
    )


__all__ = [
    "FittedUnifiedEventDenseEdge",
    "FittedUnifiedEventLocalBaseline",
    "FittedUnifiedEventEdge",
    "UnifiedEventChartCapture",
    "UnifiedEventChartPacket",
    "UnifiedEventChartRecorder",
    "UnifiedEventChartSiteObservation",
    "UnifiedEventEdgeEvaluation",
    "UnifiedEventEdgeSpec",
    "UnifiedEventLocalBaselineEvaluation",
    "UnifiedEventLocalBaseline",
    "apply_fitted_unified_event_local_baseline",
    "evaluate_unified_event_edge",
    "evaluate_fitted_unified_event_local_baseline",
    "fit_fisher_weighted_unified_event_local_baseline",
    "fit_forward_unified_event_edge",
    "fit_forward_unified_event_dense_edge",
    "fit_forward_unified_event_dense_edges",
    "fit_forward_unified_event_edges",
    "local_baseline_from_unified_event_output",
    "merge_unified_event_chart_captures",
    "packet_from_unified_event_site",
    "packet_from_unified_event_output",
]
