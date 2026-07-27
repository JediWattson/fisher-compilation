"""Executable layer fragments of prompt-Fisher parameter clusters.

Prompt-conditioned Fisher clustering is allowed to discover one cluster that
contains natural MLP parameter groups from several transformer layers.  A
native MLP replacement, however, removes rows and columns at one concrete
layer.  This module makes that lowering boundary explicit:

``global Fisher cluster -> one or more authenticated layer fragments``.

Each fragment contains only parameter-group identities, native channel
indices, axial orientations, aggregate Fisher mass, and exact native parameter
counts.  It contains neither model weights nor prompt/activation rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .fisher_prompt_clustering import (
    FisherPromptClusterPlan,
    fisher_prompt_effects_sha256,
)
from .parameter_fisher_coupling import GroupedVirtualGateFisher
from .parameter_fisher_coupling import natural_mlp_input_catalog_sha256


__all__ = [
    "ParameterClusterLayerFragment",
    "ParameterClusterLayerFragmentPlan",
    "build_parameter_cluster_layer_fragments",
]


_FORMAT_VERSION = 2
_FRAGMENT_KIND = "fisher_graph.parameter_cluster_layer_fragment"
_PLAN_KIND = "fisher_graph.parameter_cluster_layer_fragment_plan"
_FRAGMENT_DOMAIN = b"fisher_graph.parameter_cluster_fragment.v2\0"
_PLAN_DOMAIN = b"fisher_graph.parameter_cluster_fragment_plan.v2\0"
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
class ParameterClusterLayerFragment:
    """One global Fisher cluster restricted to one native MLP layer."""

    cluster_id: int
    layer_ordinal: int
    layer_id: str
    activation_site: str
    input_site: str
    output_site: str
    input_catalog_sha256: str
    input_width: int
    output_width: int
    group_indices: tuple[int, ...]
    channel_indices: tuple[int, ...]
    fisher_ranks: tuple[int, ...]
    axial_orientations: tuple[int, ...]
    native_parameter_count: int
    fisher_mass: float
    source_cluster_plan_sha256: str
    source_fisher_coupling_sha256: str
    parameter_catalog_sha256: str
    source_model_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _FRAGMENT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if type(self.cluster_id) is not int or self.cluster_id < 0:
            raise ValueError("cluster_id must be a nonnegative integer")
        if type(self.layer_ordinal) is not int or self.layer_ordinal < 0:
            raise ValueError("layer_ordinal must be a nonnegative integer")
        _require_nonempty(self.layer_id, label="layer_id")
        _require_nonempty(self.activation_site, label="activation_site")
        _require_nonempty(self.input_site, label="input_site")
        _require_nonempty(self.output_site, label="output_site")
        _require_sha256(
            self.input_catalog_sha256,
            label="input_catalog_sha256",
        )
        for name in ("input_width", "output_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        sequences = (
            self.group_indices,
            self.channel_indices,
            self.fisher_ranks,
            self.axial_orientations,
        )
        if any(type(value) is not tuple for value in sequences):
            raise TypeError("fragment member collections must be tuples")
        count = len(self.group_indices)
        if count == 0 or any(len(value) != count for value in sequences):
            raise ValueError(
                "fragment member collections must be nonempty and aligned"
            )
        if (
            self.group_indices != tuple(sorted(set(self.group_indices)))
            or self.channel_indices != tuple(sorted(set(self.channel_indices)))
            or any(type(value) is not int or value < 0 for value in self.group_indices)
            or any(
                type(value) is not int or value < 0
                for value in self.channel_indices
            )
            or any(type(value) is not int or value < 0 for value in self.fisher_ranks)
            or any(value not in (-1, 1) for value in self.axial_orientations)
        ):
            raise ValueError("fragment members are not canonical")
        if (
            type(self.native_parameter_count) is not int
            or self.native_parameter_count <= 0
        ):
            raise ValueError("native_parameter_count must be positive")
        expected_native_parameter_count = count * (
            2 * self.input_width + self.output_width
        )
        if self.native_parameter_count != expected_native_parameter_count:
            raise ValueError(
                "native_parameter_count does not match fragment widths "
                "and mode count"
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
            self.artifact_kind != _FRAGMENT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("parameter-cluster fragment header is invalid")
        computed = _json_sha256(self._payload(), domain=_FRAGMENT_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("parameter-cluster fragment hash mismatch")

    @property
    def fragment_id(self) -> str:
        return f"cluster.{self.cluster_id}/layer.{self.layer_ordinal}"

    @property
    def mode_count(self) -> int:
        return len(self.group_indices)

    @property
    def removed_mode_indices(self) -> tuple[int, ...]:
        """Native channel indices suitable for a physical MLP replacement."""

        return self.channel_indices

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "cluster_id": self.cluster_id,
            "layer_ordinal": self.layer_ordinal,
            "layer_id": self.layer_id,
            "activation_site": self.activation_site,
            "input_site": self.input_site,
            "output_site": self.output_site,
            "input_catalog_sha256": self.input_catalog_sha256,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "group_indices": self.group_indices,
            "channel_indices": self.channel_indices,
            "fisher_ranks": self.fisher_ranks,
            "axial_orientations": self.axial_orientations,
            "native_parameter_count": self.native_parameter_count,
            "fisher_mass": self.fisher_mass,
            "source_cluster_plan_sha256": self.source_cluster_plan_sha256,
            "source_fisher_coupling_sha256": (
                self.source_fisher_coupling_sha256
            ),
            "parameter_catalog_sha256": self.parameter_catalog_sha256,
            "source_model_sha256": self.source_model_sha256,
        }

    def validate_integrity(self) -> None:
        if (
            _json_sha256(self._payload(), domain=_FRAGMENT_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("parameter-cluster fragment hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "fragment_id": self.fragment_id,
            "mode_count": self.mode_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ParameterClusterLayerFragment:
        expected = {
            "artifact_kind",
            "format_version",
            "cluster_id",
            "layer_ordinal",
            "layer_id",
            "activation_site",
            "input_site",
            "output_site",
            "input_catalog_sha256",
            "input_width",
            "output_width",
            "group_indices",
            "channel_indices",
            "fisher_ranks",
            "axial_orientations",
            "native_parameter_count",
            "fisher_mass",
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="cluster fragment")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ParameterClusterLayerFragmentPlan:
    """Authenticated exhaustive partition of assigned groups by cluster/layer."""

    source_cluster_plan_sha256: str
    source_fisher_coupling_sha256: str
    parameter_catalog_sha256: str
    source_model_sha256: str
    cluster_count: int
    source_group_count: int
    assigned_group_count: int
    assigned_native_parameter_count: int
    fragments: tuple[ParameterClusterLayerFragment, ...]
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
        for name in (
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        for name, minimum in (
            ("cluster_count", 1),
            ("source_group_count", 1),
            ("assigned_group_count", 1),
            ("assigned_native_parameter_count", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.assigned_group_count > self.source_group_count:
            raise ValueError("assigned_group_count exceeds source_group_count")
        if (
            type(self.fragments) is not tuple
            or not self.fragments
            or any(
                not isinstance(value, ParameterClusterLayerFragment)
                for value in self.fragments
            )
        ):
            raise ValueError("fragments must be a nonempty tuple")
        canonical = tuple(
            sorted(
                self.fragments,
                key=lambda value: (
                    value.cluster_id,
                    value.layer_ordinal,
                    value.layer_id,
                    value.activation_site,
                ),
            )
        )
        if self.fragments != canonical:
            raise ValueError("fragments must be in canonical cluster/layer order")
        if len({value.fragment_id for value in self.fragments}) != len(
            self.fragments
        ):
            raise ValueError("fragment identities must be unique")
        all_groups = tuple(
            group
            for fragment in self.fragments
            for group in fragment.group_indices
        )
        if (
            len(all_groups) != len(set(all_groups))
            or len(all_groups) != self.assigned_group_count
            or any(
                group < 0 or group >= self.source_group_count
                for group in all_groups
            )
        ):
            raise ValueError("fragment group coverage is invalid")
        if (
            sum(value.native_parameter_count for value in self.fragments)
            != self.assigned_native_parameter_count
        ):
            raise ValueError("assigned native parameter accounting drifted")
        for fragment in self.fragments:
            fragment.validate_integrity()
            if (
                fragment.cluster_id >= self.cluster_count
                or fragment.source_cluster_plan_sha256
                != self.source_cluster_plan_sha256
                or fragment.source_fisher_coupling_sha256
                != self.source_fisher_coupling_sha256
                or fragment.parameter_catalog_sha256
                != self.parameter_catalog_sha256
                or fragment.source_model_sha256 != self.source_model_sha256
            ):
                raise ValueError("fragment provenance does not match its plan")
        for name, expected in _SAFETY_METADATA.items():
            if getattr(self, name) is not expected:
                raise ValueError("cluster-fragment safety metadata is invalid")
        if (
            self.artifact_kind != _PLAN_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("parameter-cluster fragment plan header is invalid")
        computed = _json_sha256(self._payload(), domain=_PLAN_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("parameter-cluster fragment plan hash mismatch")

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)

    @property
    def assigned_group_fraction(self) -> float:
        return float(self.assigned_group_count) / float(self.source_group_count)

    def for_cluster(
        self,
        cluster_id: int,
    ) -> tuple[ParameterClusterLayerFragment, ...]:
        return tuple(
            value for value in self.fragments if value.cluster_id == cluster_id
        )

    def for_layer(
        self,
        layer_ordinal: int,
    ) -> tuple[ParameterClusterLayerFragment, ...]:
        return tuple(
            value
            for value in self.fragments
            if value.layer_ordinal == layer_ordinal
        )

    def top_by_fisher_mass(
        self,
        count: int = 1,
    ) -> tuple[ParameterClusterLayerFragment, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(
            sorted(
                self.fragments,
                key=lambda value: (
                    -value.fisher_mass,
                    value.cluster_id,
                    value.layer_ordinal,
                ),
            )[:count]
        )

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_cluster_plan_sha256": self.source_cluster_plan_sha256,
            "source_fisher_coupling_sha256": (
                self.source_fisher_coupling_sha256
            ),
            "parameter_catalog_sha256": self.parameter_catalog_sha256,
            "source_model_sha256": self.source_model_sha256,
            "cluster_count": self.cluster_count,
            "source_group_count": self.source_group_count,
            "assigned_group_count": self.assigned_group_count,
            "assigned_native_parameter_count": (
                self.assigned_native_parameter_count
            ),
            "fragment_sha256s": tuple(
                value.artifact_sha256 for value in self.fragments
            ),
            **_SAFETY_METADATA,
        }

    def validate_integrity(self) -> None:
        for fragment in self.fragments:
            fragment.validate_integrity()
        if _json_sha256(self._payload(), domain=_PLAN_DOMAIN) != self.artifact_sha256:
            raise ValueError("parameter-cluster fragment plan hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "fragment_count": self.fragment_count,
            "fragments": tuple(value.metadata() for value in self.fragments),
            "artifact_sha256": self.artifact_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "fragments": tuple(
                value.state_dict() for value in self.fragments
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ParameterClusterLayerFragmentPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "source_cluster_plan_sha256",
            "source_fisher_coupling_sha256",
            "parameter_catalog_sha256",
            "source_model_sha256",
            "cluster_count",
            "source_group_count",
            "assigned_group_count",
            "assigned_native_parameter_count",
            "fragment_sha256s",
            *set(_SAFETY_METADATA),
            "fragments",
            "artifact_sha256",
        }
        _strict_fields(state, expected=expected, label="cluster fragment plan")
        if type(state["fragments"]) is not tuple:
            raise TypeError("cluster fragment collection must be a tuple")
        fragments = tuple(
            ParameterClusterLayerFragment.from_state_dict(value)
            for value in state["fragments"]
        )
        if state["fragment_sha256s"] != tuple(
            value.artifact_sha256 for value in fragments
        ):
            raise ValueError("fragment hash catalog does not match nested fragments")
        return cls(
            source_cluster_plan_sha256=state[
                "source_cluster_plan_sha256"
            ],  # type: ignore[arg-type]
            source_fisher_coupling_sha256=state[
                "source_fisher_coupling_sha256"
            ],  # type: ignore[arg-type]
            parameter_catalog_sha256=state[
                "parameter_catalog_sha256"
            ],  # type: ignore[arg-type]
            source_model_sha256=state[
                "source_model_sha256"
            ],  # type: ignore[arg-type]
            cluster_count=state["cluster_count"],  # type: ignore[arg-type]
            source_group_count=state[
                "source_group_count"
            ],  # type: ignore[arg-type]
            assigned_group_count=state[
                "assigned_group_count"
            ],  # type: ignore[arg-type]
            assigned_native_parameter_count=state[
                "assigned_native_parameter_count"
            ],  # type: ignore[arg-type]
            fragments=fragments,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                name: state[name] for name in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )


def build_parameter_cluster_layer_fragments(
    cluster_plan: FisherPromptClusterPlan,
    fisher_coupling: GroupedVirtualGateFisher,
) -> ParameterClusterLayerFragmentPlan:
    """Split every assigned global cluster into exact native-layer fragments."""

    if not isinstance(cluster_plan, FisherPromptClusterPlan):
        raise TypeError("cluster_plan must be a FisherPromptClusterPlan")
    if not isinstance(fisher_coupling, GroupedVirtualGateFisher):
        raise TypeError(
            "fisher_coupling must be a GroupedVirtualGateFisher"
        )
    cluster_plan.validate_integrity()
    fisher_coupling.validate_integrity()
    config = cluster_plan.config
    catalog = fisher_coupling.catalog
    if (
        config.source_fisher_coupling_sha256
        != fisher_coupling.artifact_sha256
        or config.model_fingerprint != catalog.model_fingerprint
        or config.calibration_split_sha256
        != fisher_coupling.calibration_split_sha256
        or config.objective_sha256 != fisher_coupling.objective_sha256
        or config.mode_catalog != fisher_coupling.fisher_ranked_mode_catalog()
        or cluster_plan.source_effects_sha256
        != fisher_prompt_effects_sha256(fisher_coupling.score_factor)
    ):
        raise ValueError(
            "cluster plan does not bind the supplied Fisher/catalog provenance"
        )

    grouped: dict[
        tuple[int, int, str, str],
        list[tuple[int, int, int, int, int, float]],
    ] = {}
    for group_index, (assignment, orientation, mode, group) in enumerate(
        zip(
            cluster_plan.assignments.tolist(),
            cluster_plan.orientations.tolist(),
            cluster_plan.mode_catalog,
            catalog.groups,
            strict=True,
        )
    ):
        if assignment < 0:
            continue
        if (
            group.group_index != group_index
            or group.key.layer_ordinal != mode.layer_ordinal
            or group.key.layer_id != mode.layer_id
            or group.key.activation_site != mode.activation_site
            or group.key.channel_index != mode.mode_index
        ):
            raise ValueError(
                "cluster mode catalog is not aligned with parameter groups"
            )
        key = (
            int(assignment),
            mode.layer_ordinal,
            mode.layer_id,
            mode.activation_site,
        )
        grouped.setdefault(key, []).append(
            (
                group_index,
                mode.mode_index,
                mode.fisher_rank,
                int(orientation),
                group.parameter_count,
                float(fisher_coupling.fisher_mass[group_index].item()),
            )
        )
    if not grouped:
        raise ValueError("cluster plan has no assigned nonzero-Fisher groups")

    layer_specs = {
        value.layer_ordinal: value for value in catalog.layer_specs
    }
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=key[0],
            layer_ordinal=key[1],
            layer_id=key[2],
            activation_site=key[3],
            input_site=layer_specs[key[1]].input_site,
            output_site=layer_specs[key[1]].output_site,
            input_catalog_sha256=natural_mlp_input_catalog_sha256(
                source_model_sha256=catalog.model_fingerprint,
                input_site=layer_specs[key[1]].input_site,
                input_width=layer_specs[key[1]].input_width,
            ),
            input_width=layer_specs[key[1]].input_width,
            output_width=layer_specs[key[1]].output_width,
            group_indices=tuple(value[0] for value in members),
            channel_indices=tuple(value[1] for value in members),
            fisher_ranks=tuple(value[2] for value in members),
            axial_orientations=tuple(value[3] for value in members),
            native_parameter_count=sum(value[4] for value in members),
            fisher_mass=sum(value[5] for value in members),
            source_cluster_plan_sha256=cluster_plan.artifact_sha256,
            source_fisher_coupling_sha256=fisher_coupling.artifact_sha256,
            parameter_catalog_sha256=catalog.artifact_sha256,
            source_model_sha256=catalog.model_fingerprint,
        )
        for key, members in sorted(grouped.items())
    )
    return ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=cluster_plan.artifact_sha256,
        source_fisher_coupling_sha256=fisher_coupling.artifact_sha256,
        parameter_catalog_sha256=catalog.artifact_sha256,
        source_model_sha256=catalog.model_fingerprint,
        cluster_count=cluster_plan.cluster_count,
        source_group_count=catalog.group_count,
        assigned_group_count=cluster_plan.assigned_mode_count,
        assigned_native_parameter_count=sum(
            value.native_parameter_count for value in fragments
        ),
        fragments=fragments,
    )
