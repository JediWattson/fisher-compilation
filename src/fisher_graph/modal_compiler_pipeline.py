"""Authenticated end-to-end manifest for the modal-generator compiler.

The manifest makes the complete compilation lineage machine-checkable:

``parameter catalog -> grouped Fisher -> Fisher clusters -> layer fragments``
``-> computational-mode bases -> coordinate generators -> lowered graph``.

Large analysis artifacts are accepted and authenticated at build time, but the
manifest retains only their source-safe metadata references.  Executable
artifacts needed to audit or traverse the compiled graph are copied into the
manifest.  Prompt rows, activation/gradient rows, grouped score rows, cluster
centroids, source parameter values, and source model weights are never stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import torch
from torch import Tensor

from .computational_modes import ComputationalModeBasis
from .fisher_prompt_clustering import FisherPromptClusterPlan
from .modal_generator_graph import (
    ModalGeneratorGraphPlan,
    ModalGeneratorNode,
)
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_generators import ModalGeneratorPlan
from .modal_interaction_fitting import ModalInteractionSelection
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
    build_parameter_cluster_layer_fragments,
)
from .parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPParameterGroupCatalog,
)
from .prompt_mode_tracing import PromptModeTrace


__all__ = [
    "AuthenticatedArtifactReference",
    "ModalCompilerNodeArtifact",
    "ModalCompilerPipeline",
    "ModalSourceReplacementAccounting",
    "build_modal_compiler_pipeline",
    "build_modal_source_replacement_accounting",
]


_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")

_REFERENCE_KIND = "fisher_graph.authenticated_artifact_reference"
_NODE_KIND = "fisher_graph.modal_compiler_node_artifact"
_SOURCE_ACCOUNTING_KIND = "fisher_graph.modal_source_replacement_accounting"
_PIPELINE_KIND = "fisher_graph.modal_compiler_pipeline"

_REFERENCE_DOMAIN = b"fisher_graph.modal_compiler.reference.v1\0"
_NODE_DOMAIN = b"fisher_graph.modal_compiler.node.v1\0"
_SOURCE_ACCOUNTING_DOMAIN = (
    b"fisher_graph.modal_compiler.source_accounting.v1\0"
)
_PIPELINE_DOMAIN = b"fisher_graph.modal_compiler.pipeline.v1\0"

_SOURCE_MAC_DEFINITION = (
    "one_token_native_gate_up_down_weight_matrix_macs"
)
_SAFETY_METADATA: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_grouped_fisher_score_rows": False,
    "contains_cluster_centroids": False,
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_executable_modal_bases": True,
    "contains_executable_generator_weights": True,
    "contains_executable_graph": True,
    "executable": True,
}
_FORBIDDEN_REFERENCE_KEYS = frozenset(
    {
        "prompt_text",
        "prompts",
        "token_ids",
        "activation_rows",
        "gradient_rows",
        "score_factor",
        "assignments",
        "orientations",
        "similarities",
        "centroids",
        "source_model_weights",
        "source_parameter_values",
        "parameter_values",
    }
)


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical source-safe name")
    return value


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_fields(
    state: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError(f"{label} fields are invalid")


def _authenticate_copy(value: object, expected_type: type[Any]) -> Any:
    """Fail closed and isolate a concrete authenticated artifact."""

    if not isinstance(value, expected_type):
        raise TypeError(f"artifact must be {expected_type.__name__}")
    validator = getattr(value, "validate_integrity", None)
    if callable(validator):
        validator()
    state_builder = getattr(value, "state_dict", None)
    state_loader = getattr(type(value), "from_state_dict", None)
    if not callable(state_builder) or not callable(state_loader):
        raise TypeError(
            f"{type(value).__name__} must expose an authenticated state "
            "roundtrip"
        )
    restored = state_loader(state_builder())
    restored_validator = getattr(restored, "validate_integrity", None)
    if callable(restored_validator):
        restored_validator()
    if getattr(restored, "artifact_sha256", None) != getattr(
        value,
        "artifact_sha256",
        None,
    ):
        raise ValueError("authenticated artifact roundtrip changed its hash")
    return restored


def _visit_safe_metadata(
    value: object,
    *,
    path: str = "metadata",
) -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} may not retain Tensor rows or weights")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            if key in _FORBIDDEN_REFERENCE_KEYS:
                raise ValueError(
                    f"{path} contains forbidden analysis field {key!r}"
                )
            _visit_safe_metadata(child, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _visit_safe_metadata(child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{path} contains a non-JSON-safe value")


def _artifact_metadata(value: object) -> Mapping[str, object]:
    metadata_builder = getattr(value, "metadata", None)
    if not callable(metadata_builder):
        raise TypeError(
            f"{type(value).__name__} must expose source-safe metadata()"
        )
    metadata = metadata_builder()
    if not isinstance(metadata, Mapping):
        raise TypeError("artifact metadata must be a mapping")
    _visit_safe_metadata(metadata)
    return metadata


def _metadata_json(value: Mapping[str, object]) -> str:
    # The JSON roundtrip also removes caller-owned mutable containers.
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _all_sha256_values(value: object) -> frozenset[str]:
    result: set[str] = set()

    def visit(child: object) -> None:
        if isinstance(child, Mapping):
            for nested in child.values():
                visit(nested)
        elif isinstance(child, (tuple, list)):
            for nested in child:
                visit(nested)
        elif isinstance(child, str) and _SHA256.fullmatch(child):
            result.add(child)

    visit(value)
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class AuthenticatedArtifactReference:
    """Metadata-only reference to a validated non-runtime artifact."""

    referenced_artifact_kind: str
    referenced_artifact_sha256: str
    metadata_json: str
    metadata_sha256: str = ""
    artifact_sha256: str = ""
    artifact_kind: str = _REFERENCE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(
            self.referenced_artifact_kind,
            label="referenced_artifact_kind",
        )
        _require_sha256(
            self.referenced_artifact_sha256,
            label="referenced_artifact_sha256",
        )
        if not isinstance(self.metadata_json, str) or not self.metadata_json:
            raise ValueError("metadata_json must be a nonempty JSON object")
        try:
            metadata = json.loads(self.metadata_json)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata_json is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode an object")
        _visit_safe_metadata(metadata)
        if metadata.get("artifact_kind") != self.referenced_artifact_kind:
            raise ValueError("referenced artifact kind does not match metadata")
        if (
            metadata.get("artifact_sha256")
            != self.referenced_artifact_sha256
        ):
            raise ValueError("referenced artifact hash does not match metadata")
        canonical_json = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if canonical_json != self.metadata_json:
            raise ValueError("metadata_json must use canonical JSON encoding")
        computed_metadata = hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()
        if self.metadata_sha256 == "":
            object.__setattr__(
                self,
                "metadata_sha256",
                computed_metadata,
            )
        elif (
            _require_sha256(
                self.metadata_sha256,
                label="metadata_sha256",
            )
            != computed_metadata
        ):
            raise ValueError("artifact-reference metadata hash mismatch")
        if (
            self.artifact_kind != _REFERENCE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("authenticated artifact reference header is invalid")
        computed = _json_sha256(self._payload(), domain=_REFERENCE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("authenticated artifact reference hash mismatch")

    @classmethod
    def from_artifact(
        cls,
        value: object,
    ) -> AuthenticatedArtifactReference:
        validator = getattr(value, "validate_integrity", None)
        if callable(validator):
            validator()
        state_builder = getattr(value, "state_dict", None)
        state_loader = getattr(type(value), "from_state_dict", None)
        if not callable(state_builder) or not callable(state_loader):
            raise TypeError(
                "referenced artifact must expose state_dict/from_state_dict"
            )
        restored = state_loader(state_builder())
        restored_validator = getattr(restored, "validate_integrity", None)
        if callable(restored_validator):
            restored_validator()
        artifact_sha256 = _require_sha256(
            getattr(restored, "artifact_sha256", None),
            label="referenced artifact_sha256",
        )
        artifact_kind = _require_name(
            getattr(restored, "artifact_kind", None),
            label="referenced artifact_kind",
        )
        metadata = _artifact_metadata(restored)
        return cls(
            referenced_artifact_kind=artifact_kind,
            referenced_artifact_sha256=artifact_sha256,
            metadata_json=_metadata_json(metadata),
        )

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self.metadata_json)

    @property
    def sha256_values(self) -> frozenset[str]:
        return _all_sha256_values(self.metadata)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "referenced_artifact_kind": self.referenced_artifact_kind,
            "referenced_artifact_sha256": (
                self.referenced_artifact_sha256
            ),
            "metadata_sha256": self.metadata_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "metadata_json": self.metadata_json,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> AuthenticatedArtifactReference:
        fields = {
            "artifact_kind",
            "format_version",
            "referenced_artifact_kind",
            "referenced_artifact_sha256",
            "metadata_sha256",
            "metadata_json",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="authenticated artifact reference")
        return cls(**state)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ModalSourceReplacementAccounting:
    """Exact native parameter groups physically replaced by the graph."""

    parameter_catalog_sha256: str
    parameter_cluster_fragment_plan_sha256: str
    fragment_ids: tuple[str, ...]
    fragment_sha256s: tuple[str, ...]
    group_indices: tuple[int, ...]
    source_parameter_count: int
    source_macs_per_token: int
    source_mac_definition: str = _SOURCE_MAC_DEFINITION
    artifact_sha256: str = ""
    artifact_kind: str = _SOURCE_ACCOUNTING_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.parameter_catalog_sha256,
            label="parameter_catalog_sha256",
        )
        _require_sha256(
            self.parameter_cluster_fragment_plan_sha256,
            label="parameter_cluster_fragment_plan_sha256",
        )
        if (
            type(self.fragment_ids) is not tuple
            or not self.fragment_ids
            or self.fragment_ids != tuple(sorted(set(self.fragment_ids)))
        ):
            raise ValueError("fragment_ids must be nonempty, sorted, and unique")
        for value in self.fragment_ids:
            _require_name(value, label="fragment_id")
        if (
            type(self.fragment_sha256s) is not tuple
            or len(self.fragment_sha256s) != len(self.fragment_ids)
        ):
            raise ValueError("fragment hash catalog is not aligned")
        for value in self.fragment_sha256s:
            _require_sha256(value, label="fragment_sha256")
        if (
            type(self.group_indices) is not tuple
            or not self.group_indices
            or self.group_indices
            != tuple(sorted(set(self.group_indices)))
        ):
            raise ValueError("group_indices must be nonempty, sorted, and unique")
        for value in self.group_indices:
            _require_int(value, label="group_index", minimum=0)
        _require_int(
            self.source_parameter_count,
            label="source_parameter_count",
            minimum=1,
        )
        _require_int(
            self.source_macs_per_token,
            label="source_macs_per_token",
            minimum=1,
        )
        if self.source_mac_definition != _SOURCE_MAC_DEFINITION:
            raise ValueError("source MAC accounting definition is invalid")
        if self.source_macs_per_token != self.source_parameter_count:
            raise ValueError(
                "natural MLP weight-matrix MACs must equal replaced parameters"
            )
        if (
            self.artifact_kind != _SOURCE_ACCOUNTING_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("source replacement accounting header is invalid")
        computed = _json_sha256(
            self._payload(),
            domain=_SOURCE_ACCOUNTING_DOMAIN,
        )
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("source replacement accounting hash mismatch")

    @property
    def replaced_group_count(self) -> int:
        return len(self.group_indices)

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "parameter_catalog_sha256": self.parameter_catalog_sha256,
            "parameter_cluster_fragment_plan_sha256": (
                self.parameter_cluster_fragment_plan_sha256
            ),
            "fragment_ids": self.fragment_ids,
            "fragment_sha256s": self.fragment_sha256s,
            "group_indices": self.group_indices,
            "replaced_group_count": self.replaced_group_count,
            "source_parameter_count": self.source_parameter_count,
            "source_macs_per_token": self.source_macs_per_token,
            "source_mac_definition": self.source_mac_definition,
            "contains_source_model_weights": False,
            "contains_source_parameter_values": False,
        }

    def validate_against(
        self,
        catalog: NaturalMLPParameterGroupCatalog,
        fragments: ParameterClusterLayerFragmentPlan,
    ) -> None:
        catalog.validate_integrity()
        fragments.validate_integrity()
        if (
            self.parameter_catalog_sha256 != catalog.artifact_sha256
            or self.parameter_cluster_fragment_plan_sha256
            != fragments.artifact_sha256
        ):
            raise ValueError("source accounting provenance does not match")
        by_id = {value.fragment_id: value for value in fragments.fragments}
        try:
            selected = tuple(by_id[value] for value in self.fragment_ids)
        except KeyError as error:
            raise ValueError(
                "source accounting references an unknown fragment"
            ) from error
        if self.fragment_sha256s != tuple(
            value.artifact_sha256 for value in selected
        ):
            raise ValueError("source accounting fragment hashes drifted")
        expected_groups = tuple(
            sorted(
                group
                for fragment in selected
                for group in fragment.group_indices
            )
        )
        if self.group_indices != expected_groups:
            raise ValueError("source accounting group coverage drifted")
        expected_parameters = sum(
            catalog.groups[index].parameter_count for index in expected_groups
        )
        if (
            self.source_parameter_count != expected_parameters
            or self.source_macs_per_token != expected_parameters
        ):
            raise ValueError("source replacement accounting is not exact")

    def state_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalSourceReplacementAccounting:
        fields = {
            "artifact_kind",
            "format_version",
            "parameter_catalog_sha256",
            "parameter_cluster_fragment_plan_sha256",
            "fragment_ids",
            "fragment_sha256s",
            "group_indices",
            "replaced_group_count",
            "source_parameter_count",
            "source_macs_per_token",
            "source_mac_definition",
            "contains_source_model_weights",
            "contains_source_parameter_values",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="source replacement accounting")
        if (
            state["contains_source_model_weights"] is not False
            or state["contains_source_parameter_values"] is not False
        ):
            raise ValueError("source accounting safety metadata is invalid")
        result = cls(
            parameter_catalog_sha256=state[
                "parameter_catalog_sha256"
            ],  # type: ignore[arg-type]
            parameter_cluster_fragment_plan_sha256=state[
                "parameter_cluster_fragment_plan_sha256"
            ],  # type: ignore[arg-type]
            fragment_ids=state["fragment_ids"],  # type: ignore[arg-type]
            fragment_sha256s=state[
                "fragment_sha256s"
            ],  # type: ignore[arg-type]
            group_indices=state["group_indices"],  # type: ignore[arg-type]
            source_parameter_count=state[
                "source_parameter_count"
            ],  # type: ignore[arg-type]
            source_macs_per_token=state[
                "source_macs_per_token"
            ],  # type: ignore[arg-type]
            source_mac_definition=state[
                "source_mac_definition"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )
        if state["replaced_group_count"] != result.replaced_group_count:
            raise ValueError("serialized replaced_group_count is inconsistent")
        return result


def build_modal_source_replacement_accounting(
    catalog: NaturalMLPParameterGroupCatalog,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    fragment_ids: Sequence[str],
) -> ModalSourceReplacementAccounting:
    """Authenticate exact native groups replaced by selected graph nodes."""

    authenticated_catalog = _authenticate_copy(
        catalog,
        NaturalMLPParameterGroupCatalog,
    )
    authenticated_fragments = _authenticate_copy(
        fragment_plan,
        ParameterClusterLayerFragmentPlan,
    )
    if isinstance(fragment_ids, (str, bytes)) or not isinstance(
        fragment_ids,
        Sequence,
    ):
        raise TypeError("fragment_ids must be a sequence of fragment names")
    supplied_ids = tuple(fragment_ids)
    ids = tuple(sorted(set(supplied_ids)))
    if not ids or len(ids) != len(supplied_ids):
        raise ValueError("fragment_ids must be nonempty and unique")
    if (
        authenticated_fragments.parameter_catalog_sha256
        != authenticated_catalog.artifact_sha256
    ):
        raise ValueError(
            "fragment plan does not bind the supplied parameter catalog"
        )
    by_id = {
        value.fragment_id: value for value in authenticated_fragments.fragments
    }
    try:
        selected = tuple(by_id[value] for value in ids)
    except KeyError as error:
        raise ValueError("source accounting references an unknown fragment") from error
    groups = tuple(
        sorted(
            group
            for fragment in selected
            for group in fragment.group_indices
        )
    )
    if len(groups) != len(set(groups)):
        raise ValueError("source replacement fragments overlap")
    parameters = sum(
        authenticated_catalog.groups[index].parameter_count for index in groups
    )
    result = ModalSourceReplacementAccounting(
        parameter_catalog_sha256=authenticated_catalog.artifact_sha256,
        parameter_cluster_fragment_plan_sha256=(
            authenticated_fragments.artifact_sha256
        ),
        fragment_ids=ids,
        fragment_sha256s=tuple(
            value.artifact_sha256 for value in selected
        ),
        group_indices=groups,
        source_parameter_count=parameters,
        source_macs_per_token=parameters,
    )
    result.validate_against(authenticated_catalog, authenticated_fragments)
    return result


@dataclass(frozen=True, slots=True)
class ModalCompilerNodeArtifact:
    """One selected fragment, mode basis, generator, and graph-node lowering."""

    node_name: str
    lowering: ModalGeneratorLowering
    graph_node_artifact_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _NODE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.node_name, label="node_name")
        authenticated = _authenticate_copy(
            self.lowering,
            ModalGeneratorLowering,
        )
        object.__setattr__(self, "lowering", authenticated)
        _require_sha256(
            self.graph_node_artifact_sha256,
            label="graph_node_artifact_sha256",
        )
        if (
            self.artifact_kind != _NODE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal compiler node artifact header is invalid")
        computed = _json_sha256(self._payload(), domain=_NODE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal compiler node artifact hash mismatch")

    @property
    def mode_basis(self) -> ComputationalModeBasis:
        return self.lowering.computational_mode_basis

    @property
    def coordinate_generator(self) -> ModalGeneratorPlan:
        return self.lowering.coordinate_generator_plan

    @property
    def mode_set_id(self) -> str:
        return self.lowering.mode_set_id

    @property
    def generator_id(self) -> str:
        return self.lowering.generator_id

    @property
    def selected_fragment_sha256(self) -> str:
        return self.lowering.selected_fragment_sha256

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "node_name": self.node_name,
            "mode_set_id": self.mode_set_id,
            "generator_id": self.generator_id,
            "selected_fragment_sha256": (
                self.selected_fragment_sha256
            ),
            "computational_mode_basis_sha256": (
                self.mode_basis.artifact_sha256
            ),
            "coordinate_generator_sha256": (
                self.coordinate_generator.artifact_sha256
            ),
            "lowering_sha256": self.lowering.artifact_sha256,
            "graph_weights_sha256": (
                self.lowering.graph_weights.artifact_sha256
            ),
            "fused_residual_plan_sha256": (
                self.lowering.fused_residual_plan.artifact_sha256
            ),
            "graph_node_artifact_sha256": (
                self.graph_node_artifact_sha256
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "lowering": self.lowering.state_dict(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalCompilerNodeArtifact:
        fields = {
            "artifact_kind",
            "format_version",
            "node_name",
            "mode_set_id",
            "generator_id",
            "selected_fragment_sha256",
            "computational_mode_basis_sha256",
            "coordinate_generator_sha256",
            "lowering_sha256",
            "graph_weights_sha256",
            "fused_residual_plan_sha256",
            "graph_node_artifact_sha256",
            "lowering",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal compiler node artifact")
        lowering = ModalGeneratorLowering.from_state_dict(
            state["lowering"]  # type: ignore[arg-type]
        )
        expected = {
            "mode_set_id": lowering.mode_set_id,
            "generator_id": lowering.generator_id,
            "selected_fragment_sha256": (
                lowering.selected_fragment_sha256
            ),
            "computational_mode_basis_sha256": (
                lowering.computational_mode_basis.artifact_sha256
            ),
            "coordinate_generator_sha256": (
                lowering.coordinate_generator_plan.artifact_sha256
            ),
            "lowering_sha256": lowering.artifact_sha256,
            "graph_weights_sha256": lowering.graph_weights.artifact_sha256,
            "fused_residual_plan_sha256": (
                lowering.fused_residual_plan.artifact_sha256
            ),
        }
        for field, actual in expected.items():
            if state[field] != actual:
                raise ValueError(f"{field} does not match nested lowering")
        return cls(
            node_name=state["node_name"],  # type: ignore[arg-type]
            lowering=lowering,
            graph_node_artifact_sha256=state[
                "graph_node_artifact_sha256"
            ],  # type: ignore[arg-type]
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
        )


def _metadata_path(
    value: Mapping[str, object],
    *keys: str,
) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(
                "upstream artifact metadata is missing required provenance"
            )
        current = current[key]
    return current


def _validate_upstream_lineage(
    catalog: NaturalMLPParameterGroupCatalog,
    fisher: AuthenticatedArtifactReference,
    clusters: AuthenticatedArtifactReference,
    fragments: ParameterClusterLayerFragmentPlan,
) -> None:
    fisher_metadata = fisher.metadata
    cluster_metadata = clusters.metadata
    source_prompt_trace_sha256 = fisher_metadata.get(
        "source_prompt_trace_sha256"
    )
    _require_sha256(
        source_prompt_trace_sha256,
        label="grouped Fisher source prompt trace",
    )
    if fisher_metadata.get("source_trace_authenticated") is not True:
        raise ValueError(
            "executable compiler requires prompt-trace-bound grouped Fisher"
        )
    if (
        _metadata_path(
            fisher_metadata,
            "catalog_artifact_sha256",
        )
        != catalog.artifact_sha256
        or fragments.parameter_catalog_sha256 != catalog.artifact_sha256
        or fragments.source_model_sha256 != catalog.model_fingerprint
        or fragments.source_fisher_coupling_sha256
        != fisher.referenced_artifact_sha256
        or fragments.source_cluster_plan_sha256
        != clusters.referenced_artifact_sha256
    ):
        raise ValueError(
            "parameter catalog, Fisher, clusters, and fragment plan "
            "provenance differ"
        )
    config = _metadata_path(cluster_metadata, "config")
    if not isinstance(config, Mapping):
        raise ValueError("cluster metadata config is invalid")
    checks = (
        (
            config.get("model_fingerprint"),
            catalog.model_fingerprint,
            "cluster model",
        ),
        (
            config.get("source_fisher_coupling_sha256"),
            fisher.referenced_artifact_sha256,
            "cluster Fisher",
        ),
        (
            config.get("calibration_split_sha256"),
            fisher_metadata.get("calibration_split_sha256"),
            "cluster calibration split",
        ),
        (
            config.get("objective_sha256"),
            fisher_metadata.get("objective_sha256"),
            "cluster objective",
        ),
    )
    for first, second, label in checks:
        if first != second:
            raise ValueError(f"{label} provenance mismatch")


def _fragment_for_node(
    entry: ModalCompilerNodeArtifact,
    fragment_plan: ParameterClusterLayerFragmentPlan,
) -> ParameterClusterLayerFragment:
    matches = tuple(
        value
        for value in fragment_plan.fragments
        if value.artifact_sha256 == entry.selected_fragment_sha256
    )
    if len(matches) != 1:
        raise ValueError(
            "compiler node does not bind exactly one parameter fragment"
        )
    fragment = matches[0]
    if entry.mode_set_id != fragment.fragment_id:
        raise ValueError("mode_set_id does not match fragment_id")
    return fragment


def _validate_node_chain(
    entry: ModalCompilerNodeArtifact,
    graph_node: ModalGeneratorNode,
    *,
    catalog: NaturalMLPParameterGroupCatalog,
    fisher: AuthenticatedArtifactReference,
    fragment_plan: ParameterClusterLayerFragmentPlan,
) -> None:
    lowering = entry.lowering
    basis = entry.mode_basis
    generator = entry.coordinate_generator
    fragment = _fragment_for_node(entry, fragment_plan)
    basis_binding = basis.binding
    generator_binding = generator.binding
    layer_specs = tuple(
        value
        for value in catalog.layer_specs
        if value.layer_ordinal == fragment.layer_ordinal
    )
    if len(layer_specs) != 1:
        raise ValueError(
            "compiler fragment does not select exactly one catalog layer"
        )
    layer_spec = layer_specs[0]
    checks = (
        (
            lowering.fragment_plan.artifact_sha256,
            fragment_plan.artifact_sha256,
            "lowering fragment plan",
        ),
        (
            basis_binding.source_model_sha256,
            catalog.model_fingerprint,
            "basis source model",
        ),
        (
            basis_binding.parameter_catalog_sha256,
            catalog.artifact_sha256,
            "basis parameter catalog",
        ),
        (
            basis_binding.fisher_coupling_sha256,
            fisher.referenced_artifact_sha256,
            "basis Fisher",
        ),
        (
            basis_binding.parameter_cluster_sha256,
            fragment.artifact_sha256,
            "basis fragment",
        ),
        (
            generator_binding.source_model_sha256,
            catalog.model_fingerprint,
            "generator source model",
        ),
        (
            generator_binding.output_catalog_sha256,
            basis.artifact_sha256,
            "generator coordinate catalog",
        ),
        (
            generator_binding.cluster_plan_sha256,
            fragment_plan.artifact_sha256,
            "generator fragment plan",
        ),
        (
            generator_binding.parameter_cluster_fragment_sha256,
            fragment.artifact_sha256,
            "generator fragment",
        ),
        (
            generator_binding.fisher_coupling_sha256,
            fisher.referenced_artifact_sha256,
            "generator Fisher",
        ),
        (
            generator_binding.computational_mode_basis_sha256,
            basis.artifact_sha256,
            "generator basis",
        ),
        (
            generator_binding.fit_split_sha256,
            basis_binding.fit_split_sha256,
            "generator fit split",
        ),
        (
            generator_binding.eval_split_sha256,
            basis_binding.eval_split_sha256,
            "generator evaluation split",
        ),
        (
            generator_binding.output_site,
            basis_binding.output_site,
            "generator output site",
        ),
        (
            graph_node.artifact_sha256,
            entry.graph_node_artifact_sha256,
            "graph node",
        ),
        (
            graph_node.weights.artifact_sha256,
            lowering.graph_weights.artifact_sha256,
            "graph weights",
        ),
        (
            graph_node.weights.generator_artifact_sha256,
            generator.artifact_sha256,
            "graph coordinate generator",
        ),
        (
            graph_node.weights.computational_mode_basis_sha256,
            basis.artifact_sha256,
            "graph computational-mode basis",
        ),
        (
            graph_node.weights.parameter_cluster_plan_sha256,
            fragment_plan.artifact_sha256,
            "graph fragment plan",
        ),
        (
            graph_node.weights.source_model_sha256,
            catalog.model_fingerprint,
            "graph source model",
        ),
        (
            graph_node.input_boundary,
            generator_binding.input_site,
            "graph input boundary",
        ),
        (
            graph_node.output_boundary,
            basis_binding.output_site,
            "graph output boundary",
        ),
        (
            fragment.input_width,
            layer_spec.input_width,
            "fragment and catalog input width",
        ),
        (
            fragment.output_width,
            layer_spec.output_width,
            "fragment and catalog output width",
        ),
        (
            generator.input_width,
            layer_spec.input_width,
            "generator and catalog input width",
        ),
        (
            basis.residual_width,
            layer_spec.output_width,
            "basis and catalog output width",
        ),
    )
    for first, second, label in checks:
        if first != second:
            raise ValueError(f"{label} mismatch")
    if generator_binding.target_kind != "computational_mode_coordinates":
        raise ValueError("manifest generators must target mode coordinates")
    if generator.output_width != basis.rank:
        raise ValueError("generator output width does not equal mode rank")
    if graph_node.name != entry.node_name:
        raise ValueError("graph node name does not match compiler node")
    if graph_node.weights.state_kind != "computational_mode_coordinates":
        raise ValueError("graph node state is not mode coordinates")


@dataclass(frozen=True, slots=True)
class ModalCompilerPipeline:
    """Executable compiler manifest with authenticated upstream lineage."""

    parameter_catalog: NaturalMLPParameterGroupCatalog
    grouped_fisher: AuthenticatedArtifactReference
    fisher_clusters: AuthenticatedArtifactReference
    parameter_cluster_fragments: ParameterClusterLayerFragmentPlan
    nodes: tuple[ModalCompilerNodeArtifact, ...]
    graph_plan: ModalGeneratorGraphPlan
    interaction_selection: ModalInteractionSelection | None = None
    source_replacement_accounting: (
        ModalSourceReplacementAccounting | None
    ) = None
    artifact_sha256: str = ""
    artifact_kind: str = _PIPELINE_KIND
    format_version: int = _FORMAT_VERSION
    contains_prompt_text: bool = False
    contains_token_ids: bool = False
    contains_raw_prompt_rows: bool = False
    contains_raw_activation_rows: bool = False
    contains_raw_gradient_rows: bool = False
    contains_grouped_fisher_score_rows: bool = False
    contains_cluster_centroids: bool = False
    contains_source_model_weights: bool = False
    contains_source_parameter_values: bool = False
    contains_executable_modal_bases: bool = True
    contains_executable_generator_weights: bool = True
    contains_executable_graph: bool = True
    executable: bool = True

    def __post_init__(self) -> None:
        catalog = _authenticate_copy(
            self.parameter_catalog,
            NaturalMLPParameterGroupCatalog,
        )
        fragments = _authenticate_copy(
            self.parameter_cluster_fragments,
            ParameterClusterLayerFragmentPlan,
        )
        graph = _authenticate_copy(
            self.graph_plan,
            ModalGeneratorGraphPlan,
        )
        object.__setattr__(self, "parameter_catalog", catalog)
        object.__setattr__(
            self,
            "parameter_cluster_fragments",
            fragments,
        )
        object.__setattr__(self, "graph_plan", graph)
        if not isinstance(
            self.grouped_fisher,
            AuthenticatedArtifactReference,
        ) or not isinstance(
            self.fisher_clusters,
            AuthenticatedArtifactReference,
        ):
            raise TypeError("Fisher and cluster lineage must be references")
        if (
            type(self.nodes) is not tuple
            or not self.nodes
            or any(
                not isinstance(value, ModalCompilerNodeArtifact)
                for value in self.nodes
            )
        ):
            raise ValueError("nodes must be a nonempty compiler-node tuple")
        graph_by_name = {value.name: value for value in graph.nodes}
        expected_nodes = tuple(
            sorted(
                self.nodes,
                key=lambda value: (
                    graph_by_name.get(value.node_name).causal_order
                    if value.node_name in graph_by_name
                    else 2**63,
                    value.node_name,
                ),
            )
        )
        if self.nodes != expected_nodes:
            raise ValueError("compiler nodes must be in graph causal order")
        for values, label in (
            (
                tuple(value.node_name for value in self.nodes),
                "graph node names",
            ),
            (
                tuple(value.mode_set_id for value in self.nodes),
                "mode-set ids",
            ),
            (
                tuple(value.generator_id for value in self.nodes),
                "generator ids",
            ),
            (
                tuple(
                    value.mode_basis.artifact_sha256 for value in self.nodes
                ),
                "mode bases",
            ),
            (
                tuple(
                    value.coordinate_generator.artifact_sha256
                    for value in self.nodes
                ),
                "coordinate generators",
            ),
            (
                tuple(
                    value.selected_fragment_sha256 for value in self.nodes
                ),
                "selected fragments",
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"compiler {label} must be one-to-one")
        if set(graph_by_name) != {
            value.node_name for value in self.nodes
        }:
            raise ValueError("graph nodes and compiler nodes differ")

        _validate_upstream_lineage(
            catalog,
            self.grouped_fisher,
            self.fisher_clusters,
            fragments,
        )
        fit_splits: set[str] = set()
        eval_splits: set[str] = set()
        graph_layer_order: list[int] = []
        for entry in self.nodes:
            _validate_node_chain(
                entry,
                graph_by_name[entry.node_name],
                catalog=catalog,
                fisher=self.grouped_fisher,
                fragment_plan=fragments,
            )
            graph_layer_order.append(
                _fragment_for_node(entry, fragments).layer_ordinal
            )
            fit_splits.add(entry.mode_basis.binding.fit_split_sha256)
            eval_splits.add(entry.mode_basis.binding.eval_split_sha256)
        if graph_layer_order != sorted(graph_layer_order):
            raise ValueError(
                "graph causal order cannot move backward across model layers"
            )
        if len(fit_splits) != 1 or len(eval_splits) != 1:
            raise ValueError("all modal nodes must share fit/evaluation splits")
        fit_split = next(iter(fit_splits))
        eval_split = next(iter(eval_splits))
        fisher_fit_split = self.grouped_fisher.metadata.get(
            "calibration_split_sha256"
        )
        if fit_split != fisher_fit_split:
            raise ValueError("modal fit split does not match grouped Fisher")
        if fit_split == eval_split:
            raise ValueError("modal fit and evaluation splits must differ")
        if (
            graph.model_fingerprint != catalog.model_fingerprint
            or graph.parameter_cluster_plan_sha256
            != fragments.artifact_sha256
        ):
            raise ValueError("graph model or fragment plan provenance differs")

        selection = self.interaction_selection
        if selection is None:
            if graph.interactions:
                raise ValueError(
                    "graph interactions require authenticated selection"
                )
        else:
            if not isinstance(selection, ModalInteractionSelection):
                raise TypeError(
                    "interaction_selection must be ModalInteractionSelection"
                )
            selection.validate_integrity()
            restored_selection = ModalInteractionSelection.from_state_dict(
                selection.state_dict()
            )
            object.__setattr__(
                self,
                "interaction_selection",
                restored_selection,
            )
            selection = restored_selection
            if (
                selection.source_model_sha256 != catalog.model_fingerprint
                or selection.parameter_cluster_plan_sha256
                != fragments.artifact_sha256
                or selection.fit_split_sha256 != fit_split
                or selection.eval_split_sha256 != eval_split
            ):
                raise ValueError(
                    "interaction-selection provenance differs from graph"
                )
            if tuple(
                value.artifact_sha256 for value in selection.interactions
            ) != tuple(
                value.artifact_sha256 for value in graph.interactions
            ):
                raise ValueError(
                    "graph interactions do not equal selected interactions"
                )
            expected_generator_hashes = {
                value.node_name: value.coordinate_generator.artifact_sha256
                for value in self.nodes
            }
            if {
                name for name, _, _, _ in selection.node_catalog
            } != set(graph_by_name):
                raise ValueError(
                    "interaction node catalog does not cover the graph"
                )
            for (
                name,
                causal_order,
                width,
                generator_hash,
            ) in selection.node_catalog:
                if (
                    name not in graph_by_name
                    or causal_order != graph_by_name[name].causal_order
                    or generator_hash != expected_generator_hashes[name]
                    or width != graph_by_name[name].latent_width
                ):
                    raise ValueError(
                        "interaction node catalog differs from modal nodes"
                    )

        accounting = self.source_replacement_accounting
        if accounting is not None:
            if not isinstance(accounting, ModalSourceReplacementAccounting):
                raise TypeError(
                    "source_replacement_accounting has an invalid type"
                )
            accounting.validate_against(catalog, fragments)
            if set(accounting.fragment_sha256s) != {
                value.selected_fragment_sha256 for value in self.nodes
            }:
                raise ValueError(
                    "exact source accounting must cover every compiled node"
                )
        for field, expected in _SAFETY_METADATA.items():
            if getattr(self, field) is not expected:
                raise ValueError("modal compiler safety metadata is invalid")
        if (
            self.artifact_kind != _PIPELINE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("modal compiler pipeline header is invalid")
        computed = _json_sha256(self._payload(), domain=_PIPELINE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif (
            _require_sha256(self.artifact_sha256, label="artifact_sha256")
            != computed
        ):
            raise ValueError("modal compiler pipeline hash mismatch")

    @property
    def model_fingerprint(self) -> str:
        return self.parameter_catalog.model_fingerprint

    @property
    def fit_split_sha256(self) -> str:
        return self.nodes[0].mode_basis.binding.fit_split_sha256

    @property
    def eval_split_sha256(self) -> str:
        return self.nodes[0].mode_basis.binding.eval_split_sha256

    @property
    def graph_parameter_count(self) -> int:
        return self.graph_plan.parameter_count

    @property
    def graph_macs_per_token(self) -> int:
        return self.graph_plan.macs_per_token

    @property
    def replaced_parameter_group_count(self) -> int | None:
        if self.source_replacement_accounting is None:
            return None
        return self.source_replacement_accounting.replaced_group_count

    @property
    def source_parameter_count(self) -> int | None:
        if self.source_replacement_accounting is None:
            return None
        return self.source_replacement_accounting.source_parameter_count

    @property
    def source_macs_per_token(self) -> int | None:
        if self.source_replacement_accounting is None:
            return None
        return self.source_replacement_accounting.source_macs_per_token

    @property
    def replaced_parameter_group_indices(self) -> tuple[int, ...] | None:
        if self.source_replacement_accounting is None:
            return None
        return self.source_replacement_accounting.group_indices

    @property
    def replaced_fragment_ids(self) -> tuple[str, ...] | None:
        if self.source_replacement_accounting is None:
            return None
        return self.source_replacement_accounting.fragment_ids

    @property
    def net_parameter_savings(self) -> int | None:
        source = self.source_parameter_count
        if source is None:
            return None
        return source - self.graph_parameter_count

    @property
    def net_macs_saved_per_token(self) -> int | None:
        source = self.source_macs_per_token
        if source is None:
            return None
        return source - self.graph_macs_per_token

    def _payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            **_SAFETY_METADATA,
            "model_fingerprint": self.model_fingerprint,
            "parameter_catalog_sha256": (
                self.parameter_catalog.artifact_sha256
            ),
            "grouped_fisher_sha256": (
                self.grouped_fisher.referenced_artifact_sha256
            ),
            "grouped_fisher_reference_sha256": (
                self.grouped_fisher.artifact_sha256
            ),
            "fisher_cluster_plan_sha256": (
                self.fisher_clusters.referenced_artifact_sha256
            ),
            "fisher_cluster_reference_sha256": (
                self.fisher_clusters.artifact_sha256
            ),
            "parameter_cluster_fragment_plan_sha256": (
                self.parameter_cluster_fragments.artifact_sha256
            ),
            "fit_split_sha256": self.fit_split_sha256,
            "eval_split_sha256": self.eval_split_sha256,
            "node_sha256s": tuple(
                value.artifact_sha256 for value in self.nodes
            ),
            "graph_plan_sha256": self.graph_plan.artifact_sha256,
            "interaction_selection_sha256": (
                None
                if self.interaction_selection is None
                else self.interaction_selection.artifact_sha256
            ),
            "source_replacement_accounting_sha256": (
                None
                if self.source_replacement_accounting is None
                else self.source_replacement_accounting.artifact_sha256
            ),
            "graph_parameter_count": self.graph_parameter_count,
            "graph_macs_per_token": self.graph_macs_per_token,
            "has_exact_source_accounting": (
                self.source_replacement_accounting is not None
            ),
            "replaced_parameter_group_count": (
                self.replaced_parameter_group_count
            ),
            "replaced_parameter_group_indices": (
                self.replaced_parameter_group_indices
            ),
            "replaced_fragment_ids": self.replaced_fragment_ids,
            "source_parameter_count": self.source_parameter_count,
            "source_macs_per_token": self.source_macs_per_token,
            "net_parameter_savings": self.net_parameter_savings,
            "net_macs_saved_per_token": self.net_macs_saved_per_token,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "parameter_catalog": self.parameter_catalog.state_dict(),
            "grouped_fisher": self.grouped_fisher.state_dict(),
            "fisher_clusters": self.fisher_clusters.state_dict(),
            "parameter_cluster_fragments": (
                self.parameter_cluster_fragments.state_dict()
            ),
            "nodes": tuple(value.state_dict() for value in self.nodes),
            "graph_plan": self.graph_plan.state_dict(),
            "interaction_selection": (
                None
                if self.interaction_selection is None
                else self.interaction_selection.state_dict()
            ),
            "source_replacement_accounting": (
                None
                if self.source_replacement_accounting is None
                else self.source_replacement_accounting.state_dict()
            ),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> ModalCompilerPipeline:
        fields = {
            "artifact_kind",
            "format_version",
            *_SAFETY_METADATA,
            "model_fingerprint",
            "parameter_catalog_sha256",
            "grouped_fisher_sha256",
            "grouped_fisher_reference_sha256",
            "fisher_cluster_plan_sha256",
            "fisher_cluster_reference_sha256",
            "parameter_cluster_fragment_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "node_sha256s",
            "graph_plan_sha256",
            "interaction_selection_sha256",
            "source_replacement_accounting_sha256",
            "graph_parameter_count",
            "graph_macs_per_token",
            "has_exact_source_accounting",
            "replaced_parameter_group_count",
            "replaced_parameter_group_indices",
            "replaced_fragment_ids",
            "source_parameter_count",
            "source_macs_per_token",
            "net_parameter_savings",
            "net_macs_saved_per_token",
            "parameter_catalog",
            "grouped_fisher",
            "fisher_clusters",
            "parameter_cluster_fragments",
            "nodes",
            "graph_plan",
            "interaction_selection",
            "source_replacement_accounting",
            "artifact_sha256",
        }
        _strict_fields(state, fields, label="modal compiler pipeline")
        for field, expected in _SAFETY_METADATA.items():
            if state[field] is not expected:
                raise ValueError("modal compiler safety metadata is invalid")
        raw_nodes = state["nodes"]
        if type(raw_nodes) is not tuple:
            raise TypeError("serialized compiler nodes must be a tuple")
        nodes = tuple(
            ModalCompilerNodeArtifact.from_state_dict(value)
            for value in raw_nodes  # type: ignore[arg-type]
        )
        interaction_state = state["interaction_selection"]
        interaction = (
            None
            if interaction_state is None
            else ModalInteractionSelection.from_state_dict(
                interaction_state  # type: ignore[arg-type]
            )
        )
        accounting_state = state["source_replacement_accounting"]
        accounting = (
            None
            if accounting_state is None
            else ModalSourceReplacementAccounting.from_state_dict(
                accounting_state  # type: ignore[arg-type]
            )
        )
        result = cls(
            parameter_catalog=NaturalMLPParameterGroupCatalog.from_state_dict(
                state["parameter_catalog"]  # type: ignore[arg-type]
            ),
            grouped_fisher=AuthenticatedArtifactReference.from_state_dict(
                state["grouped_fisher"]  # type: ignore[arg-type]
            ),
            fisher_clusters=AuthenticatedArtifactReference.from_state_dict(
                state["fisher_clusters"]  # type: ignore[arg-type]
            ),
            parameter_cluster_fragments=(
                ParameterClusterLayerFragmentPlan.from_state_dict(
                    state[
                        "parameter_cluster_fragments"
                    ]  # type: ignore[arg-type]
                )
            ),
            nodes=nodes,
            graph_plan=ModalGeneratorGraphPlan.from_state_dict(
                state["graph_plan"]  # type: ignore[arg-type]
            ),
            interaction_selection=interaction,
            source_replacement_accounting=accounting,
            artifact_sha256=state["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=state["artifact_kind"],  # type: ignore[arg-type]
            format_version=state["format_version"],  # type: ignore[arg-type]
            **{
                field: state[field] for field in _SAFETY_METADATA
            },  # type: ignore[arg-type]
        )
        expected = result._payload()
        for field in (
            "model_fingerprint",
            "parameter_catalog_sha256",
            "grouped_fisher_sha256",
            "grouped_fisher_reference_sha256",
            "fisher_cluster_plan_sha256",
            "fisher_cluster_reference_sha256",
            "parameter_cluster_fragment_plan_sha256",
            "fit_split_sha256",
            "eval_split_sha256",
            "node_sha256s",
            "graph_plan_sha256",
            "interaction_selection_sha256",
            "source_replacement_accounting_sha256",
            "graph_parameter_count",
            "graph_macs_per_token",
            "has_exact_source_accounting",
            "replaced_parameter_group_count",
            "replaced_parameter_group_indices",
            "replaced_fragment_ids",
            "source_parameter_count",
            "source_macs_per_token",
            "net_parameter_savings",
            "net_macs_saved_per_token",
        ):
            if state[field] != expected[field]:
                raise ValueError(f"serialized {field} is inconsistent")
        return result


def build_modal_compiler_pipeline(
    *,
    source_prompt_trace: PromptModeTrace,
    parameter_catalog: NaturalMLPParameterGroupCatalog,
    grouped_fisher: GroupedVirtualGateFisher,
    fisher_clusters: FisherPromptClusterPlan,
    parameter_cluster_fragments: ParameterClusterLayerFragmentPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    graph_plan: ModalGeneratorGraphPlan,
    interaction_selection: ModalInteractionSelection | None = None,
    source_replacement_accounting: (
        ModalSourceReplacementAccounting | None
    ) = None,
) -> ModalCompilerPipeline:
    """Authenticate every stage and emit a source-independent manifest."""

    catalog = _authenticate_copy(
        parameter_catalog,
        NaturalMLPParameterGroupCatalog,
    )
    trace = _authenticate_copy(source_prompt_trace, PromptModeTrace)
    fisher = _authenticate_copy(
        grouped_fisher,
        GroupedVirtualGateFisher,
    )
    if (
        fisher.source_prompt_trace_sha256 != trace.artifact_sha256
        or trace.provenance.source_model_fingerprint
        != catalog.model_fingerprint
        or trace.provenance.calibration_split_sha256
        != fisher.calibration_split_sha256
        or trace.provenance.objective_sha256 != fisher.objective_sha256
    ):
        raise ValueError(
            "grouped Fisher is not bound to the supplied authenticated "
            "prompt trace"
        )
    clusters = _authenticate_copy(
        fisher_clusters,
        FisherPromptClusterPlan,
    )
    fragments = _authenticate_copy(
        parameter_cluster_fragments,
        ParameterClusterLayerFragmentPlan,
    )
    derived_fragments = build_parameter_cluster_layer_fragments(
        clusters,
        fisher,
    )
    if derived_fragments.artifact_sha256 != fragments.artifact_sha256:
        raise ValueError(
            "fragment plan is not the authenticated lowering of the "
            "supplied Fisher clusters"
        )
    graph = _authenticate_copy(graph_plan, ModalGeneratorGraphPlan)
    if not isinstance(lowerings_by_node, Mapping) or not lowerings_by_node:
        raise ValueError("lowerings_by_node must be a nonempty mapping")
    graph_by_name = {value.name: value for value in graph.nodes}
    if set(lowerings_by_node) != set(graph_by_name):
        raise ValueError("lowering names must exactly match graph nodes")
    nodes = tuple(
        ModalCompilerNodeArtifact(
            node_name=graph_node.name,
            lowering=lowerings_by_node[graph_node.name],
            graph_node_artifact_sha256=graph_node.artifact_sha256,
        )
        for graph_node in graph.nodes
    )
    return ModalCompilerPipeline(
        parameter_catalog=catalog,
        grouped_fisher=AuthenticatedArtifactReference.from_artifact(fisher),
        fisher_clusters=AuthenticatedArtifactReference.from_artifact(
            clusters
        ),
        parameter_cluster_fragments=fragments,
        nodes=nodes,
        graph_plan=graph,
        interaction_selection=interaction_selection,
        source_replacement_accounting=source_replacement_accounting,
    )
