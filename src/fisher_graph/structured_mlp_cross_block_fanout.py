"""Authenticated grouped fan-out compilation for cross-block MLP modes.

A fan-out group replaces the *total residual contribution* of ``m`` native
consumer coordinates in one later MLP layer.  It reads ``k`` retained scalar
coordinates from strictly earlier layers and applies one fused decoder
``[residual_width, k]``.  This is deliberately more expressive than a set of
independent scalar carries: several retained modes can jointly reconstruct the
removed output and one retained mode can absorb any number of consumers.

The accounting boundary is exact.  Removing one native gated-MLP coordinate
removes one gate row, one up row, and one down column, or ``3 * d``
coefficients and multiply-accumulates.  A grouped decoder costs ``d * k``, so a
group with ``m`` consumers has net savings ``d * (3 * m - k)``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
    ModeKey,
)
from .structured_mlp_global_cross_block_merge import (
    DirectedCrossBlockMerge,
    GlobalCrossBlockMergePlan,
)


_GROUP_ARTIFACT_KIND = "fisher_graph.cross_block_fanout_group"
_PLAN_ARTIFACT_KIND = "fisher_graph.global_cross_block_fanout_plan"
_FORMAT_VERSION = 1
_GROUP_HASH_DOMAIN = b"fisher_graph.cross_block_fanout_group.v1\0"
_PLAN_HASH_DOMAIN = b"fisher_graph.global_cross_block_fanout_plan.v1\0"
_TENSOR_HASH_DOMAIN = b"fisher_graph.cross_block_fanout_tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_optional_sha256(
    value: object,
    *,
    label: str,
) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label=label)


def _json_sha256(value: object, *, domain: bytes) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(payload)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be a finite CPU float64 Tensor")
    canonical = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_HASH_DOMAIN)
    digest.update(f"{tuple(canonical.shape)}\0float64\0".encode("utf-8"))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _as_float64_matrix(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{label} must be a nonempty floating matrix")
    canonical = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(canonical).all():
        raise ValueError(f"{label} must contain only finite values")
    return canonical.clone()


def _mode_coordinate(value: ModeKey) -> tuple[int, int]:
    return value.layer_ordinal, value.mode_index


def _validate_modes(
    values: Sequence[ModeKey],
    *,
    label: str,
) -> tuple[ModeKey, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{label} must be a tuple")
    modes = tuple(values)
    if not modes or any(not isinstance(value, ModeKey) for value in modes):
        raise ValueError(f"{label} must contain at least one ModeKey")
    if modes != tuple(sorted(modes)):
        raise ValueError(f"{label} must be canonical")
    coordinates = tuple(_mode_coordinate(value) for value in modes)
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"{label} coordinates must be unique")
    return modes


def _validate_layer_specs(
    values: Sequence[CrossBlockLayerSpec],
) -> tuple[CrossBlockLayerSpec, ...]:
    if type(values) is not tuple:
        raise TypeError("layer_specs must be a tuple")
    specs = tuple(values)
    if len(specs) < 2 or any(
        not isinstance(spec, CrossBlockLayerSpec) for spec in specs
    ):
        raise ValueError("at least two layer specs are required")
    if specs != tuple(sorted(specs, key=lambda spec: spec.layer_ordinal)):
        raise ValueError("layer_specs must be canonical")
    if len({spec.layer_ordinal for spec in specs}) != len(specs):
        raise ValueError("layer ordinals must be unique")
    if len({spec.layer_id for spec in specs}) != len(specs):
        raise ValueError("layer ids must be unique")
    return specs


def _group_payload(
    *,
    anchors: tuple[ModeKey, ...],
    consumers: tuple[ModeKey, ...],
    residual_width: int,
    fused_decoder_sha256: str,
    artifact_kind: str,
    format_version: int,
) -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        "format_version": format_version,
        "anchors": tuple(value.metadata() for value in anchors),
        "consumers": tuple(value.metadata() for value in consumers),
        "residual_width": residual_width,
        "fused_decoder_sha256": fused_decoder_sha256,
    }


@dataclass(frozen=True, slots=True)
class CrossBlockFanoutGroup:
    """One fused multi-source reconstruction at a later MLP boundary."""

    anchors: tuple[ModeKey, ...]
    consumers: tuple[ModeKey, ...]
    fused_decoder: Tensor
    fused_decoder_sha256: str
    artifact_sha256: str
    artifact_kind: str = _GROUP_ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        anchors = _validate_modes(self.anchors, label="anchors")
        consumers = _validate_modes(self.consumers, label="consumers")
        if (
            len(
                {
                    (
                        value.layer_ordinal,
                        value.layer_id,
                        value.activation_site,
                    )
                    for value in consumers
                }
            )
            != 1
        ):
            raise ValueError("fan-out consumers must occupy one target layer")
        target_ordinal = consumers[0].layer_ordinal
        if any(value.layer_ordinal >= target_ordinal for value in anchors):
            raise ValueError("fan-out edges must point strictly forward")
        if {
            _mode_coordinate(value) for value in anchors
        } & {
            _mode_coordinate(value) for value in consumers
        }:
            raise ValueError("a fan-out consumer cannot be its own anchor")
        if (
            not isinstance(self.fused_decoder, Tensor)
            or self.fused_decoder.device.type != "cpu"
            or self.fused_decoder.dtype != torch.float64
            or self.fused_decoder.ndim != 2
            or self.fused_decoder.shape[0] <= 0
            or self.fused_decoder.shape[1] != len(anchors)
            or not torch.isfinite(self.fused_decoder).all()
        ):
            raise ValueError(
                "fused_decoder must be a finite CPU float64 "
                "[residual_width, anchor_count] Tensor"
            )
        object.__setattr__(
            self,
            "fused_decoder",
            self.fused_decoder.detach().contiguous().clone(),
        )
        _require_sha256(
            self.fused_decoder_sha256,
            label="fused_decoder_sha256",
        )
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if (
            self.artifact_kind != _GROUP_ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("fan-out group artifact header is invalid")
        if self.net_parameter_savings <= 0:
            raise ValueError("fan-out group must have positive net compression")
        self.validate_integrity()

    @property
    def target_layer_ordinal(self) -> int:
        return self.consumers[0].layer_ordinal

    @property
    def target_layer_id(self) -> str:
        return self.consumers[0].layer_id

    @property
    def target_activation_site(self) -> str:
        return self.consumers[0].activation_site

    @property
    def residual_width(self) -> int:
        return int(self.fused_decoder.shape[0])

    @property
    def anchor_count(self) -> int:
        return len(self.anchors)

    @property
    def consumer_count(self) -> int:
        return len(self.consumers)

    @property
    def native_removed_parameter_count(self) -> int:
        return 3 * self.residual_width * self.consumer_count

    @property
    def fused_decoder_parameter_count(self) -> int:
        return self.residual_width * self.anchor_count

    @property
    def net_parameter_savings(self) -> int:
        return (
            self.native_removed_parameter_count
            - self.fused_decoder_parameter_count
        )

    @property
    def native_removed_macs_per_token(self) -> int:
        return self.native_removed_parameter_count

    @property
    def fused_decoder_macs_per_token(self) -> int:
        return self.fused_decoder_parameter_count

    @property
    def net_mac_savings_per_token(self) -> int:
        return self.net_parameter_savings

    def _payload(self) -> dict[str, object]:
        return _group_payload(
            anchors=self.anchors,
            consumers=self.consumers,
            residual_width=self.residual_width,
            fused_decoder_sha256=self.fused_decoder_sha256,
            artifact_kind=self.artifact_kind,
            format_version=self.format_version,
        )

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_GROUP_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        if (
            _tensor_sha256(
                self.fused_decoder,
                label="fused_decoder",
            )
            != self.fused_decoder_sha256
        ):
            raise ValueError("fan-out fused decoder hash mismatch")
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("fan-out group artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "target_layer_ordinal": self.target_layer_ordinal,
            "target_layer_id": self.target_layer_id,
            "target_activation_site": self.target_activation_site,
            "anchor_count": self.anchor_count,
            "consumer_count": self.consumer_count,
            "native_removed_parameter_count": (
                self.native_removed_parameter_count
            ),
            "fused_decoder_parameter_count": (
                self.fused_decoder_parameter_count
            ),
            "net_parameter_savings": self.net_parameter_savings,
            "native_removed_macs_per_token": (
                self.native_removed_macs_per_token
            ),
            "fused_decoder_macs_per_token": (
                self.fused_decoder_macs_per_token
            ),
            "net_mac_savings_per_token": self.net_mac_savings_per_token,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "fused_decoder": self.fused_decoder.detach().clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> CrossBlockFanoutGroup:
        expected = {
            "artifact_kind",
            "format_version",
            "anchors",
            "consumers",
            "residual_width",
            "fused_decoder",
            "fused_decoder_sha256",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("fan-out group state fields are invalid")
        if not isinstance(state["anchors"], tuple) or not isinstance(
            state["consumers"],
            tuple,
        ):
            raise TypeError("fan-out group mode sequences must be tuples")
        if not isinstance(state["fused_decoder"], Tensor):
            raise TypeError("fan-out group decoder state must be a Tensor")
        decoder = state["fused_decoder"]
        residual_width = state["residual_width"]
        if (
            type(residual_width) is not int
            or residual_width <= 0
            or decoder.ndim != 2
            or decoder.shape[0] != residual_width
        ):
            raise ValueError("fan-out group residual width is invalid")
        return cls(
            anchors=tuple(
                ModeKey.from_state_dict(value)
                for value in state["anchors"]
            ),
            consumers=tuple(
                ModeKey.from_state_dict(value)
                for value in state["consumers"]
            ),
            fused_decoder=decoder,
            fused_decoder_sha256=str(state["fused_decoder_sha256"]),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
        )


def create_cross_block_fanout_group(
    *,
    anchors: Sequence[ModeKey],
    consumers: Sequence[ModeKey],
    fused_decoder: Tensor,
) -> CrossBlockFanoutGroup:
    """Create a canonical, tensor-authenticated target-layer group."""

    supplied_anchors = tuple(anchors)
    anchor_order = tuple(
        sorted(
            range(len(supplied_anchors)),
            key=supplied_anchors.__getitem__,
        )
    )
    canonical_anchors = tuple(
        supplied_anchors[index] for index in anchor_order
    )
    canonical_consumers = tuple(sorted(consumers))
    decoder = _as_float64_matrix(
        fused_decoder,
        label="fused_decoder",
    )
    if decoder.shape[1] != len(supplied_anchors):
        raise ValueError(
            "fused_decoder column count must match the supplied anchors"
        )
    decoder = decoder.index_select(
        1,
        torch.tensor(anchor_order, dtype=torch.long),
    )
    decoder_sha256 = _tensor_sha256(decoder, label="fused_decoder")
    payload = _group_payload(
        anchors=canonical_anchors,
        consumers=canonical_consumers,
        residual_width=int(decoder.shape[0]),
        fused_decoder_sha256=decoder_sha256,
        artifact_kind=_GROUP_ARTIFACT_KIND,
        format_version=_FORMAT_VERSION,
    )
    return CrossBlockFanoutGroup(
        anchors=canonical_anchors,
        consumers=canonical_consumers,
        fused_decoder=decoder,
        fused_decoder_sha256=decoder_sha256,
        artifact_sha256=_json_sha256(
            payload,
            domain=_GROUP_HASH_DOMAIN,
        ),
    )


def _plan_payload(
    *,
    source_discovery_artifact_sha256: str,
    source_model_fingerprint: str,
    source_merge_plan_artifact_sha256: str | None,
    layer_specs: tuple[CrossBlockLayerSpec, ...],
    groups: tuple[CrossBlockFanoutGroup, ...],
    artifact_kind: str,
    format_version: int,
    strict_forward_edges: bool,
    removed_consumers_may_anchor: bool,
    positive_net_compression_required: bool,
) -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        "format_version": format_version,
        "source_discovery_artifact_sha256": (
            source_discovery_artifact_sha256
        ),
        "source_model_fingerprint": source_model_fingerprint,
        "source_merge_plan_artifact_sha256": (
            source_merge_plan_artifact_sha256
        ),
        "layer_specs": tuple(spec.metadata() for spec in layer_specs),
        "groups": tuple(group.metadata() for group in groups),
        "strict_forward_edges": strict_forward_edges,
        "removed_consumers_may_anchor": removed_consumers_may_anchor,
        "positive_net_compression_required": (
            positive_net_compression_required
        ),
    }


@dataclass(frozen=True, slots=True)
class GlobalCrossBlockFanoutPlan:
    """Authenticated full-model DAG of grouped, strictly forward fan-outs."""

    source_discovery_artifact_sha256: str
    source_model_fingerprint: str
    source_merge_plan_artifact_sha256: str | None
    layer_specs: tuple[CrossBlockLayerSpec, ...]
    groups: tuple[CrossBlockFanoutGroup, ...]
    artifact_sha256: str
    artifact_kind: str = _PLAN_ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION
    strict_forward_edges: bool = True
    removed_consumers_may_anchor: bool = False
    positive_net_compression_required: bool = True

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_discovery_artifact_sha256,
            label="source_discovery_artifact_sha256",
        )
        _require_sha256(
            self.source_model_fingerprint,
            label="source_model_fingerprint",
        )
        _require_optional_sha256(
            self.source_merge_plan_artifact_sha256,
            label="source_merge_plan_artifact_sha256",
        )
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        specs = _validate_layer_specs(self.layer_specs)
        if (
            type(self.groups) is not tuple
            or not self.groups
            or any(
                not isinstance(group, CrossBlockFanoutGroup)
                for group in self.groups
            )
        ):
            raise ValueError("fan-out plan groups are invalid")
        expected_groups = tuple(
            sorted(
                self.groups,
                key=lambda group: (
                    group.target_layer_ordinal,
                    group.anchors,
                    group.consumers,
                ),
            )
        )
        if self.groups != expected_groups:
            raise ValueError("fan-out plan groups are not canonical")
        targets = tuple(
            group.target_layer_ordinal for group in self.groups
        )
        if len(targets) != len(set(targets)):
            raise ValueError("a target layer may have only one fan-out group")
        if (
            self.artifact_kind != _PLAN_ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
            or self.strict_forward_edges is not True
            or self.removed_consumers_may_anchor is not False
            or self.positive_net_compression_required is not True
        ):
            raise ValueError("fan-out plan policy fields are invalid")

        spec_by_ordinal = {spec.layer_ordinal: spec for spec in specs}
        consumers = tuple(
            _mode_coordinate(consumer)
            for group in self.groups
            for consumer in group.consumers
        )
        if len(consumers) != len(set(consumers)):
            raise ValueError("fan-out consumers must be globally unique")
        removed = set(consumers)
        anchors = tuple(
            anchor for group in self.groups for anchor in group.anchors
        )
        if any(_mode_coordinate(anchor) in removed for anchor in anchors):
            raise ValueError("a removed consumer cannot be a fan-out anchor")
        for group in self.groups:
            group.validate_integrity()
            for endpoint in (*group.anchors, *group.consumers):
                try:
                    spec = spec_by_ordinal[endpoint.layer_ordinal]
                except KeyError as error:
                    raise ValueError(
                        "fan-out endpoint is outside the layer catalog"
                    ) from error
                if (
                    endpoint.layer_id != spec.layer_id
                    or endpoint.activation_site != spec.activation_site
                    or endpoint.mode_index >= spec.width
                ):
                    raise ValueError(
                        "fan-out endpoint does not match its layer spec"
                    )
        if self.net_parameter_savings <= 0:
            raise ValueError("fan-out plan must have positive net compression")
        self.validate_integrity()

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def anchor_count(self) -> int:
        return sum(group.anchor_count for group in self.groups)

    @property
    def unique_anchor_count(self) -> int:
        return len(
            {
                _mode_coordinate(anchor)
                for group in self.groups
                for anchor in group.anchors
            }
        )

    @property
    def consumer_count(self) -> int:
        return sum(group.consumer_count for group in self.groups)

    @property
    def native_removed_parameter_count(self) -> int:
        return sum(
            group.native_removed_parameter_count for group in self.groups
        )

    @property
    def fused_decoder_parameter_count(self) -> int:
        return sum(
            group.fused_decoder_parameter_count for group in self.groups
        )

    @property
    def net_parameter_savings(self) -> int:
        return (
            self.native_removed_parameter_count
            - self.fused_decoder_parameter_count
        )

    @property
    def native_removed_macs_per_token(self) -> int:
        return self.native_removed_parameter_count

    @property
    def fused_decoder_macs_per_token(self) -> int:
        return self.fused_decoder_parameter_count

    @property
    def net_mac_savings_per_token(self) -> int:
        return self.net_parameter_savings

    @property
    def maximum_anchor_fanout_observed(self) -> int:
        counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for group in self.groups:
            for anchor in group.anchors:
                counts[_mode_coordinate(anchor)] += group.consumer_count
        return max(counts.values(), default=0)

    def _payload(self) -> dict[str, object]:
        return _plan_payload(
            source_discovery_artifact_sha256=(
                self.source_discovery_artifact_sha256
            ),
            source_model_fingerprint=self.source_model_fingerprint,
            source_merge_plan_artifact_sha256=(
                self.source_merge_plan_artifact_sha256
            ),
            layer_specs=self.layer_specs,
            groups=self.groups,
            artifact_kind=self.artifact_kind,
            format_version=self.format_version,
            strict_forward_edges=self.strict_forward_edges,
            removed_consumers_may_anchor=(
                self.removed_consumers_may_anchor
            ),
            positive_net_compression_required=(
                self.positive_net_compression_required
            ),
        )

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload(), domain=_PLAN_HASH_DOMAIN)

    def validate_integrity(self) -> None:
        for group in self.groups:
            group.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise ValueError("fan-out plan artifact hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "group_count": self.group_count,
            "anchor_count": self.anchor_count,
            "unique_anchor_count": self.unique_anchor_count,
            "consumer_count": self.consumer_count,
            "native_removed_parameter_count": (
                self.native_removed_parameter_count
            ),
            "fused_decoder_parameter_count": (
                self.fused_decoder_parameter_count
            ),
            "net_parameter_savings": self.net_parameter_savings,
            "native_removed_macs_per_token": (
                self.native_removed_macs_per_token
            ),
            "fused_decoder_macs_per_token": (
                self.fused_decoder_macs_per_token
            ),
            "net_mac_savings_per_token": self.net_mac_savings_per_token,
            "maximum_anchor_fanout_observed": (
                self.maximum_anchor_fanout_observed
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        payload = self._payload()
        payload["groups"] = tuple(
            group.state_dict() for group in self.groups
        )
        return {**payload, "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GlobalCrossBlockFanoutPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "source_discovery_artifact_sha256",
            "source_model_fingerprint",
            "source_merge_plan_artifact_sha256",
            "layer_specs",
            "groups",
            "strict_forward_edges",
            "removed_consumers_may_anchor",
            "positive_net_compression_required",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("fan-out plan state fields are invalid")
        if not isinstance(state["layer_specs"], tuple) or not isinstance(
            state["groups"],
            tuple,
        ):
            raise TypeError("fan-out plan sequences must be tuples")
        return cls(
            source_discovery_artifact_sha256=str(
                state["source_discovery_artifact_sha256"]
            ),
            source_model_fingerprint=str(state["source_model_fingerprint"]),
            source_merge_plan_artifact_sha256=(
                None
                if state["source_merge_plan_artifact_sha256"] is None
                else str(state["source_merge_plan_artifact_sha256"])
            ),
            layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in state["layer_specs"]
            ),
            groups=tuple(
                CrossBlockFanoutGroup.from_state_dict(value)
                for value in state["groups"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
            strict_forward_edges=state["strict_forward_edges"],
            removed_consumers_may_anchor=state[
                "removed_consumers_may_anchor"
            ],
            positive_net_compression_required=state[
                "positive_net_compression_required"
            ],
        )


def create_global_cross_block_fanout_plan(
    *,
    source_discovery_artifact_sha256: str,
    source_model_fingerprint: str,
    layer_specs: Sequence[CrossBlockLayerSpec],
    groups: Sequence[CrossBlockFanoutGroup],
    source_merge_plan_artifact_sha256: str | None = None,
) -> GlobalCrossBlockFanoutPlan:
    """Create a canonical authenticated global grouped-fan-out plan."""

    canonical_specs = tuple(
        sorted(layer_specs, key=lambda spec: spec.layer_ordinal)
    )
    canonical_groups = tuple(
        sorted(
            groups,
            key=lambda group: (
                group.target_layer_ordinal,
                group.anchors,
                group.consumers,
            ),
        )
    )
    payload = _plan_payload(
        source_discovery_artifact_sha256=(
            source_discovery_artifact_sha256
        ),
        source_model_fingerprint=source_model_fingerprint,
        source_merge_plan_artifact_sha256=(
            source_merge_plan_artifact_sha256
        ),
        layer_specs=canonical_specs,
        groups=canonical_groups,
        artifact_kind=_PLAN_ARTIFACT_KIND,
        format_version=_FORMAT_VERSION,
        strict_forward_edges=True,
        removed_consumers_may_anchor=False,
        positive_net_compression_required=True,
    )
    return GlobalCrossBlockFanoutPlan(
        source_discovery_artifact_sha256=(
            source_discovery_artifact_sha256
        ),
        source_model_fingerprint=source_model_fingerprint,
        source_merge_plan_artifact_sha256=(
            source_merge_plan_artifact_sha256
        ),
        layer_specs=canonical_specs,
        groups=canonical_groups,
        artifact_sha256=_json_sha256(payload, domain=_PLAN_HASH_DOMAIN),
    )


@dataclass(frozen=True, slots=True)
class FanoutFoldMetric:
    """Weighted reconstruction quality for one supplied calibration fold."""

    fold_id: int
    row_count: int
    weight_sum: float
    fit_nrmse: float | None
    deletion_nrmse: float | None
    recovery_fraction_vs_deletion: float | None

    def __post_init__(self) -> None:
        if type(self.fold_id) is not int:
            raise TypeError("fold_id must be an integer")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("fold row_count must be positive")
        if (
            not isinstance(self.weight_sum, float)
            or not math.isfinite(self.weight_sum)
            or self.weight_sum < 0.0
        ):
            raise ValueError("fold weight_sum is invalid")
        for label, value in (
            ("fit_nrmse", self.fit_nrmse),
            ("deletion_nrmse", self.deletion_nrmse),
            (
                "recovery_fraction_vs_deletion",
                self.recovery_fraction_vs_deletion,
            ),
        ):
            if value is not None and (
                not isinstance(value, float) or not math.isfinite(value)
            ):
                raise ValueError(f"{label} must be finite or None")

    def metadata(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "row_count": self.row_count,
            "weight_sum": self.weight_sum,
            "fit_nrmse": self.fit_nrmse,
            "deletion_nrmse": self.deletion_nrmse,
            "recovery_fraction_vs_deletion": (
                self.recovery_fraction_vs_deletion
            ),
        }


@dataclass(frozen=True, slots=True)
class GroupedFanoutDecoderFit:
    """Deterministic weighted ridge fit at the target residual boundary."""

    fused_decoder: Tensor
    ridge: float
    row_count: int
    anchor_count: int
    consumer_count: int
    residual_width: int
    weight_sum: float
    fit_nrmse: float
    deletion_nrmse: float
    recovery_fraction_vs_deletion: float
    fold_metrics: tuple[FanoutFoldMetric, ...]
    fused_decoder_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fused_decoder, Tensor)
            or self.fused_decoder.device.type != "cpu"
            or self.fused_decoder.dtype != torch.float64
            or self.fused_decoder.shape
            != (self.residual_width, self.anchor_count)
            or not torch.isfinite(self.fused_decoder).all()
        ):
            raise ValueError("fit fused decoder shape or tensor type is invalid")
        object.__setattr__(
            self,
            "fused_decoder",
            self.fused_decoder.detach().contiguous().clone(),
        )
        for label, value in (
            ("ridge", self.ridge),
            ("weight_sum", self.weight_sum),
            ("fit_nrmse", self.fit_nrmse),
            ("deletion_nrmse", self.deletion_nrmse),
            (
                "recovery_fraction_vs_deletion",
                self.recovery_fraction_vs_deletion,
            ),
        ):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.ridge < 0.0 or self.weight_sum <= 0.0:
            raise ValueError("ridge or fit weight sum is invalid")
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.anchor_count) is not int
            or self.anchor_count <= 0
            or type(self.consumer_count) is not int
            or self.consumer_count <= 0
            or type(self.residual_width) is not int
            or self.residual_width <= 0
        ):
            raise ValueError("fit dimensions are invalid")
        if (
            type(self.fold_metrics) is not tuple
            or any(
                not isinstance(value, FanoutFoldMetric)
                for value in self.fold_metrics
            )
            or tuple(
                metric.fold_id for metric in self.fold_metrics
            )
            != tuple(
                sorted(metric.fold_id for metric in self.fold_metrics)
            )
        ):
            raise ValueError("fit fold metrics are not canonical")
        _require_sha256(
            self.fused_decoder_sha256,
            label="fused_decoder_sha256",
        )
        if (
            _tensor_sha256(
                self.fused_decoder,
                label="fused_decoder",
            )
            != self.fused_decoder_sha256
        ):
            raise ValueError("fit fused decoder hash mismatch")

    def predict(self, anchor_activations: Tensor) -> Tensor:
        anchors = _as_float64_matrix(
            anchor_activations,
            label="anchor_activations",
        )
        if anchors.shape[1] != self.anchor_count:
            raise ValueError("anchor activation width does not match the fit")
        return anchors @ self.fused_decoder.T

    def metadata(self) -> dict[str, object]:
        return {
            "ridge": self.ridge,
            "row_count": self.row_count,
            "anchor_count": self.anchor_count,
            "consumer_count": self.consumer_count,
            "residual_width": self.residual_width,
            "weight_sum": self.weight_sum,
            "fit_nrmse": self.fit_nrmse,
            "deletion_nrmse": self.deletion_nrmse,
            "recovery_fraction_vs_deletion": (
                self.recovery_fraction_vs_deletion
            ),
            "fold_metrics": tuple(
                value.metadata() for value in self.fold_metrics
            ),
            "fused_decoder_sha256": self.fused_decoder_sha256,
        }


def _metric_triplet(
    target: Tensor,
    prediction: Tensor,
    weights: Tensor,
) -> tuple[float | None, float | None, float | None]:
    weighted_target_energy = float(
        (weights[:, None] * target.square()).sum().item()
    )
    if weighted_target_energy <= torch.finfo(torch.float64).tiny:
        return None, None, None
    weighted_fit_error = float(
        (weights[:, None] * (prediction - target).square()).sum().item()
    )
    fit_nrmse = math.sqrt(
        max(weighted_fit_error, 0.0) / weighted_target_energy
    )
    deletion_nrmse = 1.0
    recovery = 1.0 - weighted_fit_error / weighted_target_energy
    return float(fit_nrmse), deletion_nrmse, float(recovery)


def fit_grouped_fanout_decoder(
    anchor_activations: Tensor,
    consumer_activations: Tensor,
    native_down_columns: Tensor,
    *,
    row_weights: Tensor | None = None,
    fold_ids: Tensor | None = None,
    ridge: float = 1e-8,
) -> GroupedFanoutDecoderFit:
    """Fit a fused decoder directly against removed residual output.

    ``anchor_activations`` has shape ``[rows, k]``,
    ``consumer_activations`` has shape ``[rows, m]``, and
    ``native_down_columns`` has shape ``[d, m]``.  The regression target is
    ``consumer_activations @ native_down_columns.T`` rather than the individual
    consumer scalars.  Optional nonnegative row weights affect both the ridge
    solve and all reported errors.
    """

    anchors = _as_float64_matrix(
        anchor_activations,
        label="anchor_activations",
    )
    consumers = _as_float64_matrix(
        consumer_activations,
        label="consumer_activations",
    )
    down = _as_float64_matrix(
        native_down_columns,
        label="native_down_columns",
    )
    if anchors.shape[0] != consumers.shape[0]:
        raise ValueError("anchor and consumer row counts must match")
    if consumers.shape[1] != down.shape[1]:
        raise ValueError(
            "consumer width must match the selected native down columns"
        )
    if (
        not isinstance(ridge, (float, int))
        or isinstance(ridge, bool)
        or not math.isfinite(float(ridge))
        or float(ridge) < 0.0
    ):
        raise ValueError("ridge must be finite and nonnegative")
    ridge_value = float(ridge)
    row_count = int(anchors.shape[0])
    if row_weights is None:
        weights = torch.ones(row_count, dtype=torch.float64)
    else:
        if (
            not isinstance(row_weights, Tensor)
            or row_weights.ndim != 1
            or row_weights.shape[0] != row_count
            or not row_weights.is_floating_point()
        ):
            raise ValueError("row_weights must be a floating [rows] Tensor")
        weights = (
            row_weights.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        if not torch.isfinite(weights).all() or bool((weights < 0.0).any()):
            raise ValueError("row_weights must be finite and nonnegative")
    weight_sum = float(weights.sum().item())
    if weight_sum <= 0.0:
        raise ValueError("at least one row weight must be positive")

    canonical_fold_ids: Tensor | None
    if fold_ids is None:
        canonical_fold_ids = None
    else:
        if (
            not isinstance(fold_ids, Tensor)
            or fold_ids.ndim != 1
            or fold_ids.shape[0] != row_count
            or fold_ids.dtype
            not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
        ):
            raise ValueError("fold_ids must be an integer [rows] Tensor")
        canonical_fold_ids = (
            fold_ids.detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
            .clone()
        )

    target = consumers @ down.T
    target_energy = float(
        (weights[:, None] * target.square()).sum().item()
    )
    if target_energy <= torch.finfo(torch.float64).tiny:
        raise ValueError("removed native output must have positive energy")
    square_root_weights = torch.sqrt(weights)
    weighted_anchors = anchors * square_root_weights[:, None]
    weighted_target = target * square_root_weights[:, None]
    gram = weighted_anchors.T @ weighted_anchors
    right_hand_side = weighted_anchors.T @ weighted_target
    if ridge_value > 0.0:
        gram = gram + ridge_value * torch.eye(
            anchors.shape[1],
            dtype=torch.float64,
        )
        coefficients = torch.linalg.solve(gram, right_hand_side)
    elif int(torch.linalg.matrix_rank(gram).item()) == anchors.shape[1]:
        coefficients = torch.linalg.solve(gram, right_hand_side)
    else:
        coefficients = torch.linalg.lstsq(
            weighted_anchors,
            weighted_target,
            driver="gelsd",
        ).solution
    decoder = coefficients.T.contiguous()
    prediction = anchors @ decoder.T
    overall = _metric_triplet(target, prediction, weights)
    if any(value is None for value in overall):
        raise RuntimeError("positive-energy fit produced undefined metrics")
    fit_nrmse, deletion_nrmse, recovery = overall
    assert fit_nrmse is not None
    assert deletion_nrmse is not None
    assert recovery is not None

    fold_metrics: list[FanoutFoldMetric] = []
    if canonical_fold_ids is not None:
        for fold_id_tensor in torch.unique(
            canonical_fold_ids,
            sorted=True,
        ):
            fold_id = int(fold_id_tensor.item())
            selected = canonical_fold_ids == fold_id
            metric = _metric_triplet(
                target[selected],
                prediction[selected],
                weights[selected],
            )
            fold_metrics.append(
                FanoutFoldMetric(
                    fold_id=fold_id,
                    row_count=int(selected.sum().item()),
                    weight_sum=float(weights[selected].sum().item()),
                    fit_nrmse=metric[0],
                    deletion_nrmse=metric[1],
                    recovery_fraction_vs_deletion=metric[2],
                )
            )

    decoder_sha256 = _tensor_sha256(decoder, label="fused_decoder")
    return GroupedFanoutDecoderFit(
        fused_decoder=decoder,
        ridge=ridge_value,
        row_count=row_count,
        anchor_count=int(anchors.shape[1]),
        consumer_count=int(consumers.shape[1]),
        residual_width=int(down.shape[0]),
        weight_sum=weight_sum,
        fit_nrmse=fit_nrmse,
        deletion_nrmse=deletion_nrmse,
        recovery_fraction_vs_deletion=recovery,
        fold_metrics=tuple(fold_metrics),
        fused_decoder_sha256=decoder_sha256,
    )


def _native_down_for_layer(
    native_down_weights: Mapping[int | str, Tensor],
    spec: CrossBlockLayerSpec,
) -> Tensor:
    by_ordinal = native_down_weights.get(spec.layer_ordinal)
    by_id = native_down_weights.get(spec.layer_id)
    if by_ordinal is not None and by_id is not None:
        raise ValueError(
            f"native down weight for layer {spec.layer_id!r} is ambiguous"
        )
    value = by_ordinal if by_ordinal is not None else by_id
    if value is None:
        raise ValueError(
            f"native down weight is missing for layer {spec.layer_id!r}"
        )
    down = _as_float64_matrix(
        value,
        label=f"native_down_weights[{spec.layer_id!r}]",
    )
    if down.shape[1] != spec.width:
        raise ValueError(
            f"native down weight for {spec.layer_id!r} must have "
            f"shape [residual_width, {spec.width}]"
        )
    return down


def build_fanout_plan_from_global_merges(
    merge_plan: GlobalCrossBlockMergePlan,
    *,
    native_down_weights: Mapping[int | str, Tensor],
) -> GlobalCrossBlockFanoutPlan:
    """Fuse scalar merge carries into one decoder per consumer layer.

    For every scalar relation ``z_consumer ~= scale * z_anchor``, its native
    down column is folded into the anchor's decoder column as
    ``scale * down[:, consumer]``.  Relations sharing an anchor and target
    layer therefore become one-root-many-consumer fan-out without retaining
    the individual consumer decoder columns.
    """

    if not isinstance(merge_plan, GlobalCrossBlockMergePlan):
        raise TypeError("merge_plan must be a GlobalCrossBlockMergePlan")
    if not isinstance(native_down_weights, Mapping):
        raise TypeError("native_down_weights must be a mapping")
    spec_by_ordinal = {
        spec.layer_ordinal: spec for spec in merge_plan.layer_specs
    }
    by_target: defaultdict[
        int,
        list[DirectedCrossBlockMerge],
    ] = defaultdict(list)
    for merge in merge_plan.merges:
        by_target[merge.consumer.layer_ordinal].append(merge)
    if not by_target:
        raise ValueError("merge_plan contains no compiled merges")

    groups: list[CrossBlockFanoutGroup] = []
    for target_ordinal in sorted(by_target):
        spec = spec_by_ordinal[target_ordinal]
        down = _native_down_for_layer(native_down_weights, spec)
        merges = tuple(
            sorted(
                by_target[target_ordinal],
                key=lambda value: (value.consumer, value.anchor),
            )
        )
        anchors = tuple(sorted({value.anchor for value in merges}))
        consumers = tuple(sorted(value.consumer for value in merges))
        anchor_indices = {
            anchor: index for index, anchor in enumerate(anchors)
        }
        decoder = torch.zeros(
            down.shape[0],
            len(anchors),
            dtype=torch.float64,
        )
        for merge in merges:
            decoder[:, anchor_indices[merge.anchor]].add_(
                float(merge.activation_scale)
                * down[:, merge.consumer.mode_index]
            )
        groups.append(
            create_cross_block_fanout_group(
                anchors=anchors,
                consumers=consumers,
                fused_decoder=decoder,
            )
        )

    return create_global_cross_block_fanout_plan(
        source_discovery_artifact_sha256=(
            merge_plan.source_discovery_artifact_sha256
        ),
        source_model_fingerprint=merge_plan.source_model_fingerprint,
        source_merge_plan_artifact_sha256=merge_plan.artifact_sha256,
        layer_specs=merge_plan.layer_specs,
        groups=groups,
    )


__all__ = [
    "CrossBlockFanoutGroup",
    "FanoutFoldMetric",
    "GlobalCrossBlockFanoutPlan",
    "GroupedFanoutDecoderFit",
    "build_fanout_plan_from_global_merges",
    "create_cross_block_fanout_group",
    "create_global_cross_block_fanout_plan",
    "fit_grouped_fanout_decoder",
]
