"""Authenticated whole-layer views of parameter-cluster fragments.

Prompt-Fisher clusters may cross native transformer-layer boundaries.  The
cluster-fragment artifact lowers those clusters into native-layer pieces.  This
module performs the complementary aggregation:

``cluster/layer fragments -> exactly one exhaustive superfragment per layer``.

The resulting artifact is deliberately analysis-only.  It binds every source
fragment by SHA-256, proves disjoint and exhaustive group/channel coverage, and
accounts for the exact native MLP parameters represented by each layer.  It
contains no model weights, activations, gradients, or prompt text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)


__all__ = [
    "ParameterLayerSuperfragment",
    "ParameterLayerSuperfragmentPlan",
    "build_parameter_layer_superfragments",
]


_FORMAT_VERSION = 1
_SUPERFRAGMENT_KIND = "fisher_graph.parameter_layer_superfragment"
_PLAN_KIND = "fisher_graph.parameter_layer_superfragment_plan"
_SUPERFRAGMENT_DOMAIN = b"fisher_graph.parameter_layer_superfragment.v1\0"
_PLAN_DOMAIN = b"fisher_graph.parameter_layer_superfragment_plan.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SAFETY_METADATA: dict[str, bool] = {
    "contains_source_model_weights": False,
    "contains_prompt_text": False,
    "contains_activation_rows": False,
    "contains_gradient_rows": False,
    "contains_parameter_values": False,
    "analysis_only": True,
    "authorizes_execution": False,
}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _strict_fields(
    state: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


@dataclass(frozen=True, slots=True)
class ParameterLayerSuperfragment:
    """All authenticated parameter-cluster fragments for one native layer."""

    layer_ordinal: int
    layer_id: str
    activation_site: str
    input_site: str
    output_site: str
    input_catalog_sha256: str
    input_width: int
    output_width: int
    member_fragment_sha256s: tuple[str, ...]
    group_indices: tuple[int, ...]
    channel_indices: tuple[int, ...]
    native_parameter_count: int
    fisher_mass: float
    source_fragment_plan_sha256: str
    source_cluster_plan_sha256: str
    source_fisher_coupling_sha256: str
    parameter_catalog_sha256: str
    source_model_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _SUPERFRAGMENT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be a nonnegative integer")
        for name in (
            "layer_id",
            "activation_site",
            "input_site",
            "output_site",
        ):
            _require_nonempty(getattr(self, name), label=name)
        for name in (
            "input_catalog_sha256",
            "source_fragment_plan_sha256",
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name in ("input_width", "output_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.member_fragment_sha256s) is not tuple
            or not self.member_fragment_sha256s
            or self.member_fragment_sha256s
            != tuple(sorted(set(self.member_fragment_sha256s)))
        ):
            raise ValueError(
                "member_fragment_sha256s must be a nonempty canonical tuple"
            )
        for index, value in enumerate(self.member_fragment_sha256s):
            _require_sha256(value, label=f"member_fragment_sha256s[{index}]")
        if (
            type(self.group_indices) is not tuple
            or not self.group_indices
            or self.group_indices != tuple(sorted(set(self.group_indices)))
            or any(
                type(value) is not int or value < 0
                for value in self.group_indices
            )
        ):
            raise ValueError("group_indices must be a nonempty canonical tuple")
        if (
            type(self.channel_indices) is not tuple
            or self.channel_indices != tuple(range(len(self.channel_indices)))
            or len(self.channel_indices) != len(self.group_indices)
        ):
            raise ValueError(
                "channel_indices must exhaustively cover one zero-based layer"
            )
        if (
            type(self.native_parameter_count) is not int
            or self.native_parameter_count <= 0
        ):
            raise ValueError("native_parameter_count must be positive")
        expected_native_parameter_count = self.mode_count * (
            2 * self.input_width + self.output_width
        )
        if self.native_parameter_count != expected_native_parameter_count:
            raise ValueError(
                "native_parameter_count does not match layer widths "
                "and exhaustive mode count"
            )
        if (
            isinstance(self.fisher_mass, bool)
            or not isinstance(self.fisher_mass, (int, float))
            or not math.isfinite(float(self.fisher_mass))
            or float(self.fisher_mass) <= 0.0
        ):
            raise ValueError("fisher_mass must be finite and positive")
        object.__setattr__(self, "fisher_mass", float(self.fisher_mass))
        if (
            self.artifact_kind != _SUPERFRAGMENT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("parameter-layer superfragment header is invalid")
        computed = _json_sha256(
            self._payload(),
            domain=_SUPERFRAGMENT_DOMAIN,
        )
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("parameter-layer superfragment hash mismatch")

    @property
    def superfragment_id(self) -> str:
        return f"layer.{self.layer_ordinal}"

    @property
    def member_fragment_count(self) -> int:
        return len(self.member_fragment_sha256s)

    @property
    def mode_count(self) -> int:
        return len(self.channel_indices)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "layer_ordinal": self.layer_ordinal,
            "layer_id": self.layer_id,
            "activation_site": self.activation_site,
            "input_site": self.input_site,
            "output_site": self.output_site,
            "input_catalog_sha256": self.input_catalog_sha256,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "member_fragment_sha256s": self.member_fragment_sha256s,
            "group_indices": self.group_indices,
            "channel_indices": self.channel_indices,
            "native_parameter_count": self.native_parameter_count,
            "fisher_mass": self.fisher_mass,
            "source_fragment_plan_sha256": (
                self.source_fragment_plan_sha256
            ),
            "source_cluster_plan_sha256": self.source_cluster_plan_sha256,
            "source_fisher_coupling_sha256": (
                self.source_fisher_coupling_sha256
            ),
            "parameter_catalog_sha256": self.parameter_catalog_sha256,
            "source_model_sha256": self.source_model_sha256,
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_SUPERFRAGMENT_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("parameter-layer superfragment hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "superfragment_id": self.superfragment_id,
            "member_fragment_count": self.member_fragment_count,
            "mode_count": self.mode_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ParameterLayerSuperfragment:
        expected = {
            "artifact_kind",
            "format_version",
            "layer_ordinal",
            "layer_id",
            "activation_site",
            "input_site",
            "output_site",
            "input_catalog_sha256",
            "input_width",
            "output_width",
            "member_fragment_sha256s",
            "group_indices",
            "channel_indices",
            "native_parameter_count",
            "fisher_mass",
            "source_fragment_plan_sha256",
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
            "artifact_sha256",
        }
        _strict_fields(
            state,
            expected=expected,
            label="parameter-layer superfragment",
        )
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ParameterLayerSuperfragmentPlan:
    """Authenticated, disjoint, exhaustive whole-layer aggregation."""

    source_fragment_plan: ParameterClusterLayerFragmentPlan
    superfragments: tuple[ParameterLayerSuperfragment, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _PLAN_KIND
    format_version: int = _FORMAT_VERSION
    contains_source_model_weights: bool = False
    contains_prompt_text: bool = False
    contains_activation_rows: bool = False
    contains_gradient_rows: bool = False
    contains_parameter_values: bool = False
    analysis_only: bool = True
    authorizes_execution: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_fragment_plan,
            ParameterClusterLayerFragmentPlan,
        ):
            raise TypeError(
                "source_fragment_plan must be a "
                "ParameterClusterLayerFragmentPlan"
            )
        self.source_fragment_plan.validate_integrity()
        if (
            self.source_fragment_plan.assigned_group_count
            != self.source_fragment_plan.source_group_count
        ):
            raise ValueError(
                "source fragment plan must exhaustively assign every group"
            )
        if (
            type(self.superfragments) is not tuple
            or not self.superfragments
            or any(
                not isinstance(value, ParameterLayerSuperfragment)
                for value in self.superfragments
            )
        ):
            raise ValueError("superfragments must be a nonempty tuple")
        canonical = tuple(
            sorted(
                self.superfragments,
                key=lambda value: (
                    value.layer_ordinal,
                    value.layer_id,
                    value.activation_site,
                ),
            )
        )
        if self.superfragments != canonical:
            raise ValueError(
                "superfragments must be in canonical native-layer order"
            )
        if len(
            {value.layer_ordinal for value in self.superfragments}
        ) != len(self.superfragments):
            raise ValueError("there must be exactly one superfragment per layer")

        source_fragments = self.source_fragment_plan.fragments
        source_by_sha256 = {
            value.artifact_sha256: value for value in source_fragments
        }
        if len(source_by_sha256) != len(source_fragments):
            raise ValueError("source fragment hashes must be unique")
        expected_source_sha256s = tuple(sorted(source_by_sha256))
        member_sha256s = tuple(
            digest
            for superfragment in self.superfragments
            for digest in superfragment.member_fragment_sha256s
        )
        if (
            len(member_sha256s) != len(set(member_sha256s))
            or tuple(sorted(member_sha256s)) != expected_source_sha256s
        ):
            raise ValueError(
                "superfragment members must disjointly and exhaustively "
                "authenticate the source fragments"
            )

        layer_ordinals = {
            value.layer_ordinal for value in source_fragments
        }
        if {
            value.layer_ordinal for value in self.superfragments
        } != layer_ordinals:
            raise ValueError(
                "superfragments must cover every source native layer exactly"
            )

        all_group_indices: list[int] = []
        for superfragment in self.superfragments:
            superfragment.validate_integrity()
            members = tuple(
                source_by_sha256[digest]
                for digest in superfragment.member_fragment_sha256s
            )
            self._validate_superfragment_members(superfragment, members)
            all_group_indices.extend(superfragment.group_indices)

        expected_groups = tuple(
            range(self.source_fragment_plan.source_group_count)
        )
        if (
            len(all_group_indices) != len(set(all_group_indices))
            or tuple(sorted(all_group_indices)) != expected_groups
        ):
            raise ValueError(
                "superfragment groups must be globally disjoint and exhaustive"
            )
        if (
            sum(
                value.native_parameter_count
                for value in self.superfragments
            )
            != self.source_fragment_plan.assigned_native_parameter_count
        ):
            raise ValueError(
                "superfragment native parameter accounting drifted"
            )
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError(
                    "parameter-layer superfragment safety metadata is invalid"
                )
        if (
            self.artifact_kind != _PLAN_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError(
                "parameter-layer superfragment plan header is invalid"
            )
        computed = _json_sha256(self._payload(), domain=_PLAN_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError(
                "parameter-layer superfragment plan hash mismatch"
            )

    def _validate_superfragment_members(
        self,
        superfragment: ParameterLayerSuperfragment,
        members: tuple[ParameterClusterLayerFragment, ...],
    ) -> None:
        expected_sha256s = tuple(
            sorted(value.artifact_sha256 for value in members)
        )
        if superfragment.member_fragment_sha256s != expected_sha256s:
            raise ValueError("superfragment member hashes are not canonical")
        expected_groups = tuple(
            sorted(
                group
                for fragment in members
                for group in fragment.group_indices
            )
        )
        expected_channels = tuple(
            sorted(
                channel
                for fragment in members
                for channel in fragment.channel_indices
            )
        )
        if (
            len(expected_groups) != len(set(expected_groups))
            or superfragment.group_indices != expected_groups
        ):
            raise ValueError(
                "superfragment group coverage does not match its members"
            )
        if (
            len(expected_channels) != len(set(expected_channels))
            or superfragment.channel_indices != expected_channels
        ):
            raise ValueError(
                "superfragment channel coverage does not match its members"
            )
        if any(
            (
                fragment.layer_ordinal != superfragment.layer_ordinal
                or fragment.layer_id != superfragment.layer_id
                or fragment.activation_site != superfragment.activation_site
                or fragment.input_site != superfragment.input_site
                or fragment.output_site != superfragment.output_site
                or fragment.input_catalog_sha256
                != superfragment.input_catalog_sha256
                or fragment.input_width != superfragment.input_width
                or fragment.output_width != superfragment.output_width
            )
            for fragment in members
        ):
            raise ValueError(
                "superfragment members do not share one native-layer schema"
            )
        source = self.source_fragment_plan
        if (
            superfragment.source_fragment_plan_sha256
            != source.artifact_sha256
            or superfragment.source_cluster_plan_sha256
            != source.source_cluster_plan_sha256
            or superfragment.source_fisher_coupling_sha256
            != source.source_fisher_coupling_sha256
            or superfragment.parameter_catalog_sha256
            != source.parameter_catalog_sha256
            or superfragment.source_model_sha256
            != source.source_model_sha256
        ):
            raise ValueError(
                "superfragment provenance does not match its source plan"
            )
        if superfragment.native_parameter_count != sum(
            value.native_parameter_count for value in members
        ):
            raise ValueError(
                "superfragment native parameter total does not match members"
            )
        if superfragment.fisher_mass != math.fsum(
            value.fisher_mass for value in members
        ):
            raise ValueError(
                "superfragment Fisher mass does not match its members"
            )

    @property
    def source_fragment_plan_sha256(self) -> str:
        return self.source_fragment_plan.artifact_sha256

    @property
    def source_cluster_plan_sha256(self) -> str:
        return self.source_fragment_plan.source_cluster_plan_sha256

    @property
    def source_fisher_coupling_sha256(self) -> str:
        return self.source_fragment_plan.source_fisher_coupling_sha256

    @property
    def parameter_catalog_sha256(self) -> str:
        return self.source_fragment_plan.parameter_catalog_sha256

    @property
    def source_model_sha256(self) -> str:
        return self.source_fragment_plan.source_model_sha256

    @property
    def source_fragment_sha256s(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.artifact_sha256
                for value in self.source_fragment_plan.fragments
            )
        )

    @property
    def source_fragment_count(self) -> int:
        return self.source_fragment_plan.fragment_count

    @property
    def layer_count(self) -> int:
        return len(self.superfragments)

    @property
    def superfragment_count(self) -> int:
        return len(self.superfragments)

    @property
    def source_group_count(self) -> int:
        return self.source_fragment_plan.source_group_count

    @property
    def assigned_group_count(self) -> int:
        return sum(value.mode_count for value in self.superfragments)

    @property
    def assigned_native_parameter_count(self) -> int:
        return sum(
            value.native_parameter_count for value in self.superfragments
        )

    def for_layer(
        self,
        layer_ordinal: int,
    ) -> ParameterLayerSuperfragment:
        if type(layer_ordinal) is not int or layer_ordinal < 0:
            raise ValueError("layer_ordinal must be a nonnegative integer")
        matches = tuple(
            value
            for value in self.superfragments
            if value.layer_ordinal == layer_ordinal
        )
        if len(matches) != 1:
            raise KeyError(f"no superfragment for layer {layer_ordinal}")
        return matches[0]

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_fragment_plan_sha256": (
                self.source_fragment_plan_sha256
            ),
            "source_cluster_plan_sha256": self.source_cluster_plan_sha256,
            "source_fisher_coupling_sha256": (
                self.source_fisher_coupling_sha256
            ),
            "parameter_catalog_sha256": self.parameter_catalog_sha256,
            "source_model_sha256": self.source_model_sha256,
            "source_fragment_count": self.source_fragment_count,
            "layer_count": self.layer_count,
            "source_group_count": self.source_group_count,
            "assigned_group_count": self.assigned_group_count,
            "assigned_native_parameter_count": (
                self.assigned_native_parameter_count
            ),
            "source_fragment_sha256s": self.source_fragment_sha256s,
            "superfragment_sha256s": tuple(
                value.artifact_sha256 for value in self.superfragments
            ),
            **_SAFETY_METADATA,
        }

    def validate_integrity(self) -> None:
        self.source_fragment_plan.validate_integrity()
        for superfragment in self.superfragments:
            superfragment.validate_integrity()
        if _json_sha256(self._payload(), domain=_PLAN_DOMAIN) != self.artifact_sha256:
            raise ValueError(
                "parameter-layer superfragment plan hash mismatch"
            )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "superfragment_count": self.superfragment_count,
            "superfragments": tuple(
                value.metadata() for value in self.superfragments
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "source_fragment_plan": self.source_fragment_plan.state_dict(),
            "superfragments": tuple(
                value.state_dict() for value in self.superfragments
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ParameterLayerSuperfragmentPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "source_fragment_plan_sha256",
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
            "source_fragment_count",
            "layer_count",
            "source_group_count",
            "assigned_group_count",
            "assigned_native_parameter_count",
            "source_fragment_sha256s",
            "superfragment_sha256s",
            *set(_SAFETY_METADATA),
            "source_fragment_plan",
            "superfragments",
            "artifact_sha256",
        }
        _strict_fields(
            state,
            expected=expected,
            label="parameter-layer superfragment plan",
        )
        if not isinstance(state["source_fragment_plan"], Mapping):
            raise TypeError("source_fragment_plan state must be a mapping")
        if type(state["superfragments"]) is not tuple:
            raise TypeError("superfragment collection must be a tuple")
        source_fragment_plan = (
            ParameterClusterLayerFragmentPlan.from_state_dict(
                state["source_fragment_plan"],
            )
        )
        superfragments = tuple(
            ParameterLayerSuperfragment.from_state_dict(value)
            for value in state["superfragments"]
        )
        if state["source_fragment_sha256s"] != tuple(
            sorted(
                value.artifact_sha256
                for value in source_fragment_plan.fragments
            )
        ):
            raise ValueError(
                "source fragment hash catalog does not match nested plan"
            )
        if state["superfragment_sha256s"] != tuple(
            value.artifact_sha256 for value in superfragments
        ):
            raise ValueError(
                "superfragment hash catalog does not match nested records"
            )
        restored = cls(
            source_fragment_plan=source_fragment_plan,
            superfragments=superfragments,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name] for name in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )
        if restored.state_dict() != dict(state):
            raise ValueError(
                "parameter-layer superfragment plan summaries are invalid"
            )
        return restored


def build_parameter_layer_superfragments(
    fragment_plan: ParameterClusterLayerFragmentPlan,
) -> ParameterLayerSuperfragmentPlan:
    """Aggregate an exhaustive cluster-fragment plan once per native layer."""

    if not isinstance(fragment_plan, ParameterClusterLayerFragmentPlan):
        raise TypeError(
            "fragment_plan must be a ParameterClusterLayerFragmentPlan"
        )
    fragment_plan.validate_integrity()
    if fragment_plan.assigned_group_count != fragment_plan.source_group_count:
        raise ValueError(
            "fragment_plan must exhaustively assign every parameter group"
        )

    grouped: dict[int, list[ParameterClusterLayerFragment]] = {}
    for fragment in fragment_plan.fragments:
        grouped.setdefault(fragment.layer_ordinal, []).append(fragment)

    superfragments: list[ParameterLayerSuperfragment] = []
    for layer_ordinal, unsorted_members in sorted(grouped.items()):
        members = tuple(
            sorted(
                unsorted_members,
                key=lambda value: value.artifact_sha256,
            )
        )
        representative = members[0]
        group_indices = tuple(
            sorted(
                group
                for fragment in members
                for group in fragment.group_indices
            )
        )
        channel_indices = tuple(
            sorted(
                channel
                for fragment in members
                for channel in fragment.channel_indices
            )
        )
        superfragments.append(
            ParameterLayerSuperfragment(
                layer_ordinal=layer_ordinal,
                layer_id=representative.layer_id,
                activation_site=representative.activation_site,
                input_site=representative.input_site,
                output_site=representative.output_site,
                input_catalog_sha256=(
                    representative.input_catalog_sha256
                ),
                input_width=representative.input_width,
                output_width=representative.output_width,
                member_fragment_sha256s=tuple(
                    value.artifact_sha256 for value in members
                ),
                group_indices=group_indices,
                channel_indices=channel_indices,
                native_parameter_count=sum(
                    value.native_parameter_count for value in members
                ),
                fisher_mass=math.fsum(
                    value.fisher_mass for value in members
                ),
                source_fragment_plan_sha256=fragment_plan.artifact_sha256,
                source_cluster_plan_sha256=(
                    fragment_plan.source_cluster_plan_sha256
                ),
                source_fisher_coupling_sha256=(
                    fragment_plan.source_fisher_coupling_sha256
                ),
                parameter_catalog_sha256=(
                    fragment_plan.parameter_catalog_sha256
                ),
                source_model_sha256=fragment_plan.source_model_sha256,
            )
        )
    return ParameterLayerSuperfragmentPlan(
        source_fragment_plan=fragment_plan,
        superfragments=tuple(superfragments),
    )
