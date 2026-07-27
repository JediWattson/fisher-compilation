"""Uncapped fan-out planning for directed cross-block MLP supermodes.

The discovery layer deliberately uses a sparse proxy search, but this planner
does not impose a per-layer quota or a maximum accepted-merge count.  Every
exactly qualified consumer is assigned its best available earlier native root.
Roots may fan out to arbitrarily many later consumers.  A removed consumer is
never reused as a root, so every runtime carry remains a native feature rather
than a chain of progressively approximated features.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryResult,
    CrossBlockLayerSpec,
    CrossBlockPairEvidence,
    ModeKey,
)


_ARTIFACT_KIND = "fisher_graph.global_cross_block_merge_plan"
_FORMAT_VERSION = 1
_HASH_DOMAIN = b"fisher_graph.global_cross_block_merge_plan.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DirectedCrossBlockMerge:
    """One native earlier feature replacing one later generator coordinate."""

    anchor: ModeKey
    consumer: ModeKey
    activation_scale: float
    activation_residual_nrmse: float
    priority_max_loss: float
    priority_sum_loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, ModeKey) or not isinstance(
            self.consumer,
            ModeKey,
        ):
            raise TypeError("merge endpoints must be ModeKey values")
        if self.anchor.layer_ordinal >= self.consumer.layer_ordinal:
            raise ValueError("a directed merge must point strictly forward")
        for label, value in (
            ("activation_scale", self.activation_scale),
            ("activation_residual_nrmse", self.activation_residual_nrmse),
            ("priority_max_loss", self.priority_max_loss),
            ("priority_sum_loss", self.priority_sum_loss),
        ):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if (
            not 0.0 <= self.activation_residual_nrmse <= 1.0 + 1e-12
            or self.priority_max_loss < 0.0
            or self.priority_sum_loss < 0.0
        ):
            raise ValueError("merge quality metrics are invalid")

    @property
    def anchor_coordinate(self) -> tuple[int, int]:
        return self.anchor.layer_ordinal, self.anchor.mode_index

    @property
    def consumer_coordinate(self) -> tuple[int, int]:
        return self.consumer.layer_ordinal, self.consumer.mode_index

    def metadata(self) -> dict[str, object]:
        return {
            "anchor": self.anchor.metadata(),
            "consumer": self.consumer.metadata(),
            "activation_scale": self.activation_scale,
            "activation_residual_nrmse": self.activation_residual_nrmse,
            "priority_max_loss": self.priority_max_loss,
            "priority_sum_loss": self.priority_sum_loss,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> DirectedCrossBlockMerge:
        expected = {
            "anchor",
            "consumer",
            "activation_scale",
            "activation_residual_nrmse",
            "priority_max_loss",
            "priority_sum_loss",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("directed merge state fields are invalid")
        if not isinstance(state["anchor"], Mapping) or not isinstance(
            state["consumer"],
            Mapping,
        ):
            raise TypeError("directed merge endpoints must be mappings")
        return cls(
            anchor=ModeKey.from_state_dict(state["anchor"]),
            consumer=ModeKey.from_state_dict(state["consumer"]),
            activation_scale=float(state["activation_scale"]),
            activation_residual_nrmse=float(
                state["activation_residual_nrmse"]
            ),
            priority_max_loss=float(state["priority_max_loss"]),
            priority_sum_loss=float(state["priority_sum_loss"]),
        )


def _merge_from_evidence(
    evidence: CrossBlockPairEvidence,
) -> DirectedCrossBlockMerge:
    if not evidence.is_static_merge_hypothesis:
        raise ValueError("only exact static-merge hypotheses may be compiled")
    gram = evidence.activation_gram
    denominator = float(gram[0, 0].item())
    target_square = float(gram[1, 1].item())
    cross = float(gram[0, 1].item())
    if denominator <= 0.0 or target_square <= 0.0:
        raise ValueError("qualified merge activation energy must be positive")
    scale = cross / denominator
    residual_square = max(
        target_square - cross * cross / denominator,
        0.0,
    )
    residual_nrmse = math.sqrt(residual_square / target_square)
    return DirectedCrossBlockMerge(
        anchor=evidence.first,
        consumer=evidence.second,
        activation_scale=float(scale),
        activation_residual_nrmse=float(residual_nrmse),
        priority_max_loss=evidence.priority_max_loss,
        priority_sum_loss=evidence.priority_sum_loss,
    )


def _merge_quality_key(
    merge: DirectedCrossBlockMerge,
) -> tuple[float, float, float, ModeKey]:
    return (
        merge.priority_max_loss,
        merge.priority_sum_loss,
        merge.activation_residual_nrmse,
        merge.anchor,
    )


@dataclass(frozen=True, slots=True)
class GlobalCrossBlockMergePlan:
    """Authenticated full-model fan-out forest with no merge-count quota."""

    source_discovery_artifact_sha256: str
    source_model_fingerprint: str
    layer_specs: tuple[CrossBlockLayerSpec, ...]
    merges: tuple[DirectedCrossBlockMerge, ...]
    qualified_hypothesis_count: int
    artifact_sha256: str
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION
    all_discovered_layers_eligible: bool = True
    anchor_fanout_unbounded: bool = True
    maximum_accepted_merges: None = None

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_discovery_artifact_sha256,
            label="source_discovery_artifact_sha256",
        )
        _require_sha256(
            self.source_model_fingerprint,
            label="source_model_fingerprint",
        )
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        if (
            type(self.layer_specs) is not tuple
            or len(self.layer_specs) < 2
            or any(
                not isinstance(spec, CrossBlockLayerSpec)
                for spec in self.layer_specs
            )
        ):
            raise ValueError("global merge layer specs are invalid")
        if (
            type(self.qualified_hypothesis_count) is not int
            or self.qualified_hypothesis_count < len(self.merges)
        ):
            raise ValueError("qualified hypothesis count is invalid")
        expected_order = tuple(
            sorted(
                self.merges,
                key=lambda merge: (
                    merge.consumer,
                    merge.anchor,
                ),
            )
        )
        if type(self.merges) is not tuple or self.merges != expected_order:
            raise ValueError("global merges are not canonical")
        consumers = tuple(
            merge.consumer_coordinate for merge in self.merges
        )
        if len(consumers) != len(set(consumers)):
            raise ValueError("a consumer may have only one incoming merge")
        removed = set(consumers)
        if any(merge.anchor_coordinate in removed for merge in self.merges):
            raise ValueError("a removed consumer cannot be a merge anchor")
        spec_by_ordinal = {
            spec.layer_ordinal: spec for spec in self.layer_specs
        }
        for merge in self.merges:
            for endpoint in (merge.anchor, merge.consumer):
                try:
                    spec = spec_by_ordinal[endpoint.layer_ordinal]
                except KeyError as error:
                    raise ValueError(
                        "merge endpoint is outside the layer catalog"
                    ) from error
                if (
                    endpoint.layer_id != spec.layer_id
                    or endpoint.activation_site != spec.activation_site
                    or endpoint.mode_index >= spec.width
                ):
                    raise ValueError(
                        "merge endpoint does not match its layer spec"
                    )
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
            or self.all_discovered_layers_eligible is not True
            or self.anchor_fanout_unbounded is not True
            or self.maximum_accepted_merges is not None
        ):
            raise ValueError("global merge plan policy fields are invalid")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("global merge plan hash mismatch")

    @property
    def merge_count(self) -> int:
        return len(self.merges)

    @property
    def affected_layer_ordinals(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    endpoint.layer_ordinal
                    for merge in self.merges
                    for endpoint in (merge.anchor, merge.consumer)
                }
            )
        )

    @property
    def maximum_anchor_fanout_observed(self) -> int:
        counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for merge in self.merges:
            counts[merge.anchor_coordinate] += 1
        return max(counts.values(), default=0)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_discovery_artifact_sha256": (
                self.source_discovery_artifact_sha256
            ),
            "source_model_fingerprint": self.source_model_fingerprint,
            "layer_specs": tuple(
                spec.metadata() for spec in self.layer_specs
            ),
            "merges": tuple(merge.metadata() for merge in self.merges),
            "qualified_hypothesis_count": (
                self.qualified_hypothesis_count
            ),
            "all_discovered_layers_eligible": (
                self.all_discovered_layers_eligible
            ),
            "anchor_fanout_unbounded": self.anchor_fanout_unbounded,
            "maximum_accepted_merges": self.maximum_accepted_merges,
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "merge_count": self.merge_count,
            "affected_layer_ordinals": self.affected_layer_ordinals,
            "maximum_anchor_fanout_observed": (
                self.maximum_anchor_fanout_observed
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> GlobalCrossBlockMergePlan:
        expected = {
            "artifact_kind",
            "format_version",
            "source_discovery_artifact_sha256",
            "source_model_fingerprint",
            "layer_specs",
            "merges",
            "qualified_hypothesis_count",
            "all_discovered_layers_eligible",
            "anchor_fanout_unbounded",
            "maximum_accepted_merges",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("global merge plan state fields are invalid")
        if not isinstance(state["layer_specs"], tuple) or not isinstance(
            state["merges"],
            tuple,
        ):
            raise TypeError("global merge plan sequences must be tuples")
        return cls(
            source_discovery_artifact_sha256=str(
                state["source_discovery_artifact_sha256"]
            ),
            source_model_fingerprint=str(state["source_model_fingerprint"]),
            layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in state["layer_specs"]
            ),
            merges=tuple(
                DirectedCrossBlockMerge.from_state_dict(value)
                for value in state["merges"]
            ),
            qualified_hypothesis_count=int(
                state["qualified_hypothesis_count"]
            ),
            artifact_sha256=str(state["artifact_sha256"]),
            artifact_kind=str(state["artifact_kind"]),
            format_version=int(state["format_version"]),
            all_discovered_layers_eligible=bool(
                state["all_discovered_layers_eligible"]
            ),
            anchor_fanout_unbounded=bool(
                state["anchor_fanout_unbounded"]
            ),
            maximum_accepted_merges=state["maximum_accepted_merges"],
        )


def plan_global_cross_block_merges(
    discovery: CrossBlockDiscoveryResult,
) -> GlobalCrossBlockMergePlan:
    """Select every feasible exact hypothesis without an accepted-edge cap.

    Consumers are visited in model order.  For each consumer, the best exact
    candidate whose anchor remains native is selected.  A retained anchor can
    be reused without limit, giving true star-shaped supermodes across blocks.
    """

    if not isinstance(discovery, CrossBlockDiscoveryResult):
        raise TypeError("discovery must be a CrossBlockDiscoveryResult")
    candidates_by_consumer: defaultdict[
        ModeKey,
        list[DirectedCrossBlockMerge],
    ] = defaultdict(list)
    qualified = 0
    for evidence in discovery.evidence:
        if evidence.is_static_merge_hypothesis:
            qualified += 1
            merge = _merge_from_evidence(evidence)
            candidates_by_consumer[merge.consumer].append(merge)

    removed: set[tuple[int, int]] = set()
    selected: list[DirectedCrossBlockMerge] = []
    for consumer in sorted(candidates_by_consumer):
        candidates = sorted(
            candidates_by_consumer[consumer],
            key=_merge_quality_key,
        )
        chosen = next(
            (
                merge
                for merge in candidates
                if merge.anchor_coordinate not in removed
            ),
            None,
        )
        if chosen is None:
            continue
        selected.append(chosen)
        removed.add(chosen.consumer_coordinate)

    merges = tuple(
        sorted(
            selected,
            key=lambda merge: (merge.consumer, merge.anchor),
        )
    )
    temporary = {
        "artifact_kind": _ARTIFACT_KIND,
        "format_version": _FORMAT_VERSION,
        "source_discovery_artifact_sha256": discovery.artifact_sha256,
        "source_model_fingerprint": discovery.provenance.model_fingerprint,
        "layer_specs": tuple(
            spec.metadata() for spec in discovery.layer_specs
        ),
        "merges": tuple(merge.metadata() for merge in merges),
        "qualified_hypothesis_count": qualified,
        "all_discovered_layers_eligible": True,
        "anchor_fanout_unbounded": True,
        "maximum_accepted_merges": None,
    }
    return GlobalCrossBlockMergePlan(
        source_discovery_artifact_sha256=discovery.artifact_sha256,
        source_model_fingerprint=discovery.provenance.model_fingerprint,
        layer_specs=discovery.layer_specs,
        merges=merges,
        qualified_hypothesis_count=qualified,
        artifact_sha256=_json_sha256(temporary),
    )


__all__ = [
    "DirectedCrossBlockMerge",
    "GlobalCrossBlockMergePlan",
    "plan_global_cross_block_merges",
]
