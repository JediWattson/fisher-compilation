"""Proposal-only planning for cross-block native MLP coordinates.

This module is a topology boundary, not a compiler boundary.  It converts the
endpoint-disjoint hypotheses in a validated
``CrossBlockDiscoveryResult`` into deterministic, unresolved carry proposals
and groups overlapping inclusive layer intervals into minimal windows.

``ModeKey.mode_index`` is the native MLP source coordinate.  Fisher rank is
retained as discovery metadata and is never substituted for that coordinate.
No decoder scale is inferred here, and no proposal or window authorizes an
intervention, compilation, execution, held-out guard, or calibration-B run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re

from .structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryResult,
    CrossBlockLayerSpec,
    ModeKey,
)


_ARTIFACT_KIND = "fisher_graph.structured_mlp_cross_block_proposal_plan"
_FORMAT_VERSION = 1
_HASH_DOMAIN = b"fisher_graph.structured_mlp_cross_block_plan.v1\0"
_ID_DOMAIN = b"fisher_graph.structured_mlp_cross_block_plan.id.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes = _HASH_DOMAIN) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_json_bytes(value))
    return digest.hexdigest()


def _native_endpoint_metadata(key: ModeKey) -> dict[str, object]:
    return {
        "layer_ordinal": key.layer_ordinal,
        "layer_id": key.layer_id,
        "activation_site": key.activation_site,
        "source_index": key.mode_index,
    }


def _proposal_id(anchor: ModeKey, consumer: ModeKey) -> str:
    digest = _json_sha256(
        {
            "kind": "unresolved_native_mlp_carry",
            "anchor": _native_endpoint_metadata(anchor),
            "consumer": _native_endpoint_metadata(consumer),
        },
        domain=_ID_DOMAIN,
    )
    return f"unresolved-carry-{digest}"


def _window_id(
    *,
    start_layer_ordinal: int,
    end_layer_ordinal: int,
    layer_ids: tuple[str, ...],
    proposal_ids: tuple[str, ...],
) -> str:
    digest = _json_sha256(
        {
            "kind": "inclusive_cross_block_window",
            "start_layer_ordinal": start_layer_ordinal,
            "end_layer_ordinal": end_layer_ordinal,
            "layer_ids": layer_ids,
            "proposal_ids": proposal_ids,
        },
        domain=_ID_DOMAIN,
    )
    return f"proposal-window-{digest}"


@dataclass(frozen=True, slots=True)
class UnresolvedCrossBlockCarryProposal:
    """One native-coordinate carry hypothesis awaiting intervention."""

    proposal_id: str
    anchor: ModeKey
    consumer: ModeKey
    anchor_source_index: int
    consumer_source_index: int
    consumer_decoder_scale: None = None
    intervention_required: bool = True
    discovery_only: bool = True
    authorizes_static_merge: bool = False
    authorizes_intervention: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_guard: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, ModeKey) or not isinstance(
            self.consumer,
            ModeKey,
        ):
            raise TypeError("carry endpoints must be ModeKey values")
        if self.anchor.layer_ordinal >= self.consumer.layer_ordinal:
            raise ValueError(
                "an unresolved carry must point strictly forward"
            )
        if (
            type(self.anchor_source_index) is not int
            or self.anchor_source_index != self.anchor.mode_index
            or type(self.consumer_source_index) is not int
            or self.consumer_source_index != self.consumer.mode_index
        ):
            raise ValueError(
                "carry source indices must equal native ModeKey.mode_index"
            )
        if self.proposal_id != _proposal_id(self.anchor, self.consumer):
            raise ValueError("unresolved carry proposal id mismatch")
        if self.consumer_decoder_scale is not None:
            raise ValueError(
                "proposal-only carries cannot resolve a decoder scale"
            )
        if (
            self.intervention_required is not True
            or self.discovery_only is not True
            or self.authorizes_static_merge is not False
            or self.authorizes_intervention is not False
            or self.authorizes_compilation is not False
            or self.authorizes_execution is not False
            or self.authorizes_guard is not False
            or self.authorizes_b is not False
        ):
            raise ValueError(
                "unresolved carry proposal safety metadata is invalid"
            )

    @property
    def inclusive_interval(self) -> tuple[int, int]:
        return self.anchor.layer_ordinal, self.consumer.layer_ordinal

    def metadata(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "anchor": self.anchor.metadata(),
            "consumer": self.consumer.metadata(),
            "anchor_source_index": self.anchor_source_index,
            "consumer_source_index": self.consumer_source_index,
            "consumer_decoder_scale": self.consumer_decoder_scale,
            "intervention_required": self.intervention_required,
            "discovery_only": self.discovery_only,
            "authorizes_static_merge": self.authorizes_static_merge,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_mode_keys(
        cls,
        anchor: ModeKey,
        consumer: ModeKey,
    ) -> UnresolvedCrossBlockCarryProposal:
        """Create an unresolved proposal from native discovery endpoints."""

        if not isinstance(anchor, ModeKey) or not isinstance(
            consumer,
            ModeKey,
        ):
            raise TypeError("carry endpoints must be ModeKey values")
        return cls(
            proposal_id=_proposal_id(anchor, consumer),
            anchor=anchor,
            consumer=consumer,
            anchor_source_index=anchor.mode_index,
            consumer_source_index=consumer.mode_index,
        )

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> UnresolvedCrossBlockCarryProposal:
        fields = {
            "proposal_id",
            "anchor",
            "consumer",
            "anchor_source_index",
            "consumer_source_index",
            "consumer_decoder_scale",
            "intervention_required",
            "discovery_only",
            "authorizes_static_merge",
            "authorizes_intervention",
            "authorizes_compilation",
            "authorizes_execution",
            "authorizes_guard",
            "authorizes_b",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError(
                "unresolved carry proposal state fields are invalid"
            )
        if not isinstance(state["anchor"], Mapping) or not isinstance(
            state["consumer"],
            Mapping,
        ):
            raise TypeError("carry endpoint states must be mappings")
        if type(state["anchor_source_index"]) is not int or type(
            state["consumer_source_index"]
        ) is not int:
            raise TypeError("carry source indices must be integers")
        return cls(
            proposal_id=state["proposal_id"],  # type: ignore[arg-type]
            anchor=ModeKey.from_state_dict(state["anchor"]),
            consumer=ModeKey.from_state_dict(state["consumer"]),
            anchor_source_index=state["anchor_source_index"],
            consumer_source_index=state["consumer_source_index"],
            consumer_decoder_scale=state["consumer_decoder_scale"],
            intervention_required=state[  # type: ignore[arg-type]
                "intervention_required"
            ],
            discovery_only=state["discovery_only"],  # type: ignore[arg-type]
            authorizes_static_merge=state[  # type: ignore[arg-type]
                "authorizes_static_merge"
            ],
            authorizes_intervention=state[  # type: ignore[arg-type]
                "authorizes_intervention"
            ],
            authorizes_compilation=state[  # type: ignore[arg-type]
                "authorizes_compilation"
            ],
            authorizes_execution=state[  # type: ignore[arg-type]
                "authorizes_execution"
            ],
            authorizes_guard=state[  # type: ignore[arg-type]
                "authorizes_guard"
            ],
            authorizes_b=state["authorizes_b"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StructuredMLPCrossBlockWindow:
    """Minimal inclusive layer window for overlapping proposals."""

    window_id: str
    start_layer_ordinal: int
    end_layer_ordinal: int
    layer_ids: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    intervention_required: bool = True
    discovery_only: bool = True
    authorizes_intervention: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_guard: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.start_layer_ordinal) is not int
            or type(self.end_layer_ordinal) is not int
            or self.start_layer_ordinal < 0
            or self.start_layer_ordinal >= self.end_layer_ordinal
        ):
            raise ValueError("proposal window ordinals are invalid")
        if (
            type(self.layer_ids) is not tuple
            or len(self.layer_ids)
            != self.end_layer_ordinal - self.start_layer_ordinal + 1
            or any(
                not isinstance(layer_id, str) or not layer_id
                for layer_id in self.layer_ids
            )
            or len(set(self.layer_ids)) != len(self.layer_ids)
        ):
            raise ValueError(
                "proposal window must contain every inclusive layer id"
            )
        if (
            type(self.proposal_ids) is not tuple
            or not self.proposal_ids
            or self.proposal_ids != tuple(sorted(self.proposal_ids))
            or len(set(self.proposal_ids)) != len(self.proposal_ids)
            or any(
                not isinstance(proposal_id, str) or not proposal_id
                for proposal_id in self.proposal_ids
            )
        ):
            raise ValueError(
                "proposal window ids must be a canonical nonempty tuple"
            )
        if self.window_id != _window_id(
            start_layer_ordinal=self.start_layer_ordinal,
            end_layer_ordinal=self.end_layer_ordinal,
            layer_ids=self.layer_ids,
            proposal_ids=self.proposal_ids,
        ):
            raise ValueError("proposal window id mismatch")
        if (
            self.intervention_required is not True
            or self.discovery_only is not True
            or self.authorizes_intervention is not False
            or self.authorizes_compilation is not False
            or self.authorizes_execution is not False
            or self.authorizes_guard is not False
            or self.authorizes_b is not False
        ):
            raise ValueError(
                "proposal window safety metadata is invalid"
            )

    @property
    def inclusive_interval(self) -> tuple[int, int]:
        return self.start_layer_ordinal, self.end_layer_ordinal

    def metadata(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "start_layer_ordinal": self.start_layer_ordinal,
            "end_layer_ordinal": self.end_layer_ordinal,
            "layer_ids": self.layer_ids,
            "proposal_ids": self.proposal_ids,
            "intervention_required": self.intervention_required,
            "discovery_only": self.discovery_only,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
        }

    def state_dict(self) -> dict[str, object]:
        return self.metadata()

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StructuredMLPCrossBlockWindow:
        fields = {
            "window_id",
            "start_layer_ordinal",
            "end_layer_ordinal",
            "layer_ids",
            "proposal_ids",
            "intervention_required",
            "discovery_only",
            "authorizes_intervention",
            "authorizes_compilation",
            "authorizes_execution",
            "authorizes_guard",
            "authorizes_b",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError(
                "cross-block proposal window state fields are invalid"
            )
        if (
            type(state["start_layer_ordinal"]) is not int
            or type(state["end_layer_ordinal"]) is not int
            or not isinstance(state["layer_ids"], tuple)
            or not isinstance(state["proposal_ids"], tuple)
        ):
            raise TypeError("proposal window state types are invalid")
        return cls(
            window_id=state["window_id"],  # type: ignore[arg-type]
            start_layer_ordinal=state["start_layer_ordinal"],
            end_layer_ordinal=state["end_layer_ordinal"],
            layer_ids=state["layer_ids"],  # type: ignore[arg-type]
            proposal_ids=state["proposal_ids"],  # type: ignore[arg-type]
            intervention_required=state[  # type: ignore[arg-type]
                "intervention_required"
            ],
            discovery_only=state["discovery_only"],  # type: ignore[arg-type]
            authorizes_intervention=state[  # type: ignore[arg-type]
                "authorizes_intervention"
            ],
            authorizes_compilation=state[  # type: ignore[arg-type]
                "authorizes_compilation"
            ],
            authorizes_execution=state[  # type: ignore[arg-type]
                "authorizes_execution"
            ],
            authorizes_guard=state[  # type: ignore[arg-type]
                "authorizes_guard"
            ],
            authorizes_b=state["authorizes_b"],  # type: ignore[arg-type]
        )


def _canonical_layer_specs(
    values: Sequence[CrossBlockLayerSpec],
) -> tuple[CrossBlockLayerSpec, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("source_layer_specs must be a sequence")
    specs = tuple(values)
    if len(specs) < 2 or any(
        not isinstance(spec, CrossBlockLayerSpec) for spec in specs
    ):
        raise ValueError(
            "cross-block plan requires at least two layer specs"
        )
    ordinals = tuple(spec.layer_ordinal for spec in specs)
    if ordinals != tuple(range(ordinals[0], ordinals[-1] + 1)):
        raise ValueError(
            "cross-block plan layer specs must be contiguous and ordered"
        )
    if (
        len({spec.layer_id for spec in specs}) != len(specs)
        or len({spec.activation_site for spec in specs}) != len(specs)
    ):
        raise ValueError("cross-block plan layer identities must be unique")
    return specs


def _windows_from_proposals(
    proposals: tuple[UnresolvedCrossBlockCarryProposal, ...],
    layer_specs: tuple[CrossBlockLayerSpec, ...],
) -> tuple[StructuredMLPCrossBlockWindow, ...]:
    if not proposals:
        return ()
    spec_by_ordinal = {
        spec.layer_ordinal: spec for spec in layer_specs
    }
    ordered = tuple(
        sorted(
            proposals,
            key=lambda proposal: (
                proposal.inclusive_interval,
                proposal.proposal_id,
            ),
        )
    )
    groups: list[
        tuple[int, int, list[UnresolvedCrossBlockCarryProposal]]
    ] = []
    start, end = ordered[0].inclusive_interval
    members = [ordered[0]]
    for proposal in ordered[1:]:
        next_start, next_end = proposal.inclusive_interval
        if next_start <= end:
            end = max(end, next_end)
            members.append(proposal)
        else:
            groups.append((start, end, members))
            start, end, members = next_start, next_end, [proposal]
    groups.append((start, end, members))

    windows = []
    for start, end, members in groups:
        try:
            layer_ids = tuple(
                spec_by_ordinal[ordinal].layer_id
                for ordinal in range(start, end + 1)
            )
        except KeyError as error:
            raise ValueError(
                "proposal interval is not covered by the layer catalog"
            ) from error
        proposal_ids = tuple(
            sorted(proposal.proposal_id for proposal in members)
        )
        windows.append(
            StructuredMLPCrossBlockWindow(
                window_id=_window_id(
                    start_layer_ordinal=start,
                    end_layer_ordinal=end,
                    layer_ids=layer_ids,
                    proposal_ids=proposal_ids,
                ),
                start_layer_ordinal=start,
                end_layer_ordinal=end,
                layer_ids=layer_ids,
                proposal_ids=proposal_ids,
            )
        )
    return tuple(windows)


@dataclass(frozen=True, slots=True)
class StructuredMLPCrossBlockPlan:
    """Hashed discovery-only carry topology with no execution authority."""

    source_discovery_artifact_sha256: str
    source_sketch_artifact_sha256: str
    source_model_fingerprint: str
    source_layer_specs: tuple[CrossBlockLayerSpec, ...]
    proposals: tuple[UnresolvedCrossBlockCarryProposal, ...]
    windows: tuple[StructuredMLPCrossBlockWindow, ...]
    artifact_sha256: str
    artifact_kind: str = _ARTIFACT_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_corpus_rows: bool = False
    consumer_decoder_scales_resolved: bool = False
    intervention_required: bool = True
    discovery_only: bool = True
    authorizes_static_merge: bool = False
    authorizes_intervention: bool = False
    authorizes_compilation: bool = False
    authorizes_execution: bool = False
    authorizes_guard: bool = False
    authorizes_b: bool = False

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_discovery_artifact_sha256,
            label="source_discovery_artifact_sha256",
        )
        _require_sha256(
            self.source_sketch_artifact_sha256,
            label="source_sketch_artifact_sha256",
        )
        _require_sha256(
            self.source_model_fingerprint,
            label="source_model_fingerprint",
        )
        _require_sha256(self.artifact_sha256, label="artifact_sha256")
        specs = _canonical_layer_specs(self.source_layer_specs)
        if specs != self.source_layer_specs:
            raise ValueError("source layer specs are not canonical")
        if (
            type(self.proposals) is not tuple
            or any(
                not isinstance(
                    proposal,
                    UnresolvedCrossBlockCarryProposal,
                )
                for proposal in self.proposals
            )
            or self.proposals
            != tuple(
                sorted(
                    self.proposals,
                    key=lambda proposal: (
                        proposal.anchor,
                        proposal.consumer,
                    ),
                )
            )
            or len(
                {proposal.proposal_id for proposal in self.proposals}
            )
            != len(self.proposals)
        ):
            raise ValueError("cross-block proposals are not canonical")

        spec_by_ordinal = {
            spec.layer_ordinal: spec for spec in specs
        }
        native_endpoints: list[tuple[int, int]] = []
        for proposal in self.proposals:
            for endpoint in (proposal.anchor, proposal.consumer):
                try:
                    spec = spec_by_ordinal[endpoint.layer_ordinal]
                except KeyError as error:
                    raise ValueError(
                        "proposal endpoint is outside the layer catalog"
                    ) from error
                if (
                    endpoint.layer_id != spec.layer_id
                    or endpoint.activation_site != spec.activation_site
                    or endpoint.mode_index >= spec.width
                ):
                    raise ValueError(
                        "proposal endpoint does not match its native layer "
                        "coordinate"
                    )
                native_endpoints.append(
                    (endpoint.layer_ordinal, endpoint.mode_index)
                )
        if len(native_endpoints) != len(set(native_endpoints)):
            raise ValueError(
                "cross-block proposals must remain endpoint-disjoint by "
                "native coordinate"
            )

        expected_windows = _windows_from_proposals(
            self.proposals,
            specs,
        )
        if (
            type(self.windows) is not tuple
            or len(self.windows) != len(expected_windows)
            or any(
                left.metadata() != right.metadata()
                for left, right in zip(
                    self.windows,
                    expected_windows,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "proposal windows do not match deterministic interval merge"
            )
        if (
            self.artifact_kind != _ARTIFACT_KIND
            or self.format_version != _FORMAT_VERSION
            or self.contains_source_model_weights is not False
            or self.contains_corpus_rows is not False
            or self.consumer_decoder_scales_resolved is not False
            or self.intervention_required is not True
            or self.discovery_only is not True
            or self.authorizes_static_merge is not False
            or self.authorizes_intervention is not False
            or self.authorizes_compilation is not False
            or self.authorizes_execution is not False
            or self.authorizes_guard is not False
            or self.authorizes_b is not False
        ):
            raise ValueError("cross-block proposal plan safety metadata invalid")
        if self.artifact_sha256 != self._computed_sha256():
            raise ValueError("cross-block proposal plan hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "contains_source_model_weights": (
                self.contains_source_model_weights
            ),
            "contains_corpus_rows": self.contains_corpus_rows,
            "consumer_decoder_scales_resolved": (
                self.consumer_decoder_scales_resolved
            ),
            "intervention_required": self.intervention_required,
            "discovery_only": self.discovery_only,
            "authorizes_static_merge": self.authorizes_static_merge,
            "authorizes_intervention": self.authorizes_intervention,
            "authorizes_compilation": self.authorizes_compilation,
            "authorizes_execution": self.authorizes_execution,
            "authorizes_guard": self.authorizes_guard,
            "authorizes_b": self.authorizes_b,
            "source_discovery_artifact_sha256": (
                self.source_discovery_artifact_sha256
            ),
            "source_sketch_artifact_sha256": (
                self.source_sketch_artifact_sha256
            ),
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_layer_specs": tuple(
                spec.metadata() for spec in self.source_layer_specs
            ),
            "proposals": tuple(
                proposal.metadata() for proposal in self.proposals
            ),
            "windows": tuple(
                window.metadata() for window in self.windows
            ),
        }

    def _computed_sha256(self) -> str:
        return _json_sha256(self._payload())

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "proposal_count": len(self.proposals),
            "window_count": len(self.windows),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> StructuredMLPCrossBlockPlan:
        fields = {
            "artifact_kind",
            "format_version",
            "contains_source_model_weights",
            "contains_corpus_rows",
            "consumer_decoder_scales_resolved",
            "intervention_required",
            "discovery_only",
            "authorizes_static_merge",
            "authorizes_intervention",
            "authorizes_compilation",
            "authorizes_execution",
            "authorizes_guard",
            "authorizes_b",
            "source_discovery_artifact_sha256",
            "source_sketch_artifact_sha256",
            "source_model_fingerprint",
            "source_layer_specs",
            "proposals",
            "windows",
            "artifact_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != fields:
            raise ValueError(
                "cross-block proposal plan state fields are invalid"
            )
        if (
            type(state["format_version"]) is not int
            or not isinstance(state["source_layer_specs"], tuple)
            or not isinstance(state["proposals"], tuple)
            or not isinstance(state["windows"], tuple)
        ):
            raise TypeError("cross-block proposal plan state types invalid")
        return cls(
            source_discovery_artifact_sha256=state[  # type: ignore[arg-type]
                "source_discovery_artifact_sha256"
            ],
            source_sketch_artifact_sha256=state[  # type: ignore[arg-type]
                "source_sketch_artifact_sha256"
            ],
            source_model_fingerprint=state[  # type: ignore[arg-type]
                "source_model_fingerprint"
            ],
            source_layer_specs=tuple(
                CrossBlockLayerSpec.from_state_dict(value)
                for value in state["source_layer_specs"]  # type: ignore[union-attr]
            ),
            proposals=tuple(
                UnresolvedCrossBlockCarryProposal.from_state_dict(value)
                for value in state["proposals"]  # type: ignore[union-attr]
            ),
            windows=tuple(
                StructuredMLPCrossBlockWindow.from_state_dict(value)
                for value in state["windows"]  # type: ignore[union-attr]
            ),
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],
            contains_source_model_weights=state[  # type: ignore[arg-type]
                "contains_source_model_weights"
            ],
            contains_corpus_rows=state[  # type: ignore[arg-type]
                "contains_corpus_rows"
            ],
            consumer_decoder_scales_resolved=state[  # type: ignore[arg-type]
                "consumer_decoder_scales_resolved"
            ],
            intervention_required=state[  # type: ignore[arg-type]
                "intervention_required"
            ],
            discovery_only=state["discovery_only"],  # type: ignore[arg-type]
            authorizes_static_merge=state[  # type: ignore[arg-type]
                "authorizes_static_merge"
            ],
            authorizes_intervention=state[  # type: ignore[arg-type]
                "authorizes_intervention"
            ],
            authorizes_compilation=state[  # type: ignore[arg-type]
                "authorizes_compilation"
            ],
            authorizes_execution=state[  # type: ignore[arg-type]
                "authorizes_execution"
            ],
            authorizes_guard=state[  # type: ignore[arg-type]
                "authorizes_guard"
            ],
            authorizes_b=state["authorizes_b"],  # type: ignore[arg-type]
        )


def _temporary_plan_payload(
    *,
    result: CrossBlockDiscoveryResult,
    layer_specs: tuple[CrossBlockLayerSpec, ...],
    proposals: tuple[UnresolvedCrossBlockCarryProposal, ...],
    windows: tuple[StructuredMLPCrossBlockWindow, ...],
) -> dict[str, object]:
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "format_version": _FORMAT_VERSION,
        "contains_source_model_weights": False,
        "contains_corpus_rows": False,
        "consumer_decoder_scales_resolved": False,
        "intervention_required": True,
        "discovery_only": True,
        "authorizes_static_merge": False,
        "authorizes_intervention": False,
        "authorizes_compilation": False,
        "authorizes_execution": False,
        "authorizes_guard": False,
        "authorizes_b": False,
        "source_discovery_artifact_sha256": result.artifact_sha256,
        "source_sketch_artifact_sha256": (
            result.sketch_artifact_sha256
        ),
        "source_model_fingerprint": result.provenance.model_fingerprint,
        "source_layer_specs": tuple(
            spec.metadata() for spec in layer_specs
        ),
        "proposals": tuple(
            proposal.metadata() for proposal in proposals
        ),
        "windows": tuple(window.metadata() for window in windows),
    }


def plan_structured_mlp_cross_block_carries(
    result: CrossBlockDiscoveryResult,
) -> StructuredMLPCrossBlockPlan:
    """Create unresolved carry topology from validated discovery hypotheses."""

    if not isinstance(result, CrossBlockDiscoveryResult):
        raise TypeError("result must be a CrossBlockDiscoveryResult")
    if (
        not result.discovery_only
        or result.authorizes_static_merge
        or result.authorizes_execution
        or result.authorizes_b
    ):
        raise ValueError(
            "source discovery result unexpectedly authorizes execution"
        )
    layer_specs = _canonical_layer_specs(result.layer_specs)
    proposals = tuple(
        sorted(
            (
                UnresolvedCrossBlockCarryProposal.from_mode_keys(
                    first,
                    second,
                )
                for first, second in result.selected_pairs
            ),
            key=lambda proposal: (
                proposal.anchor,
                proposal.consumer,
            ),
        )
    )
    windows = _windows_from_proposals(proposals, layer_specs)
    payload = _temporary_plan_payload(
        result=result,
        layer_specs=layer_specs,
        proposals=proposals,
        windows=windows,
    )
    return StructuredMLPCrossBlockPlan(
        source_discovery_artifact_sha256=result.artifact_sha256,
        source_sketch_artifact_sha256=result.sketch_artifact_sha256,
        source_model_fingerprint=result.provenance.model_fingerprint,
        source_layer_specs=layer_specs,
        proposals=proposals,
        windows=windows,
        artifact_sha256=_json_sha256(payload),
    )


__all__ = [
    "StructuredMLPCrossBlockPlan",
    "StructuredMLPCrossBlockWindow",
    "UnresolvedCrossBlockCarryProposal",
    "plan_structured_mlp_cross_block_carries",
]
