"""Source-safe composition of the frozen Gemma layer-10 and layer-17 graphs.

The two input candidates were fitted and evaluated independently, but both
were lowered from the same authenticated global parameter-fragment plan.  This
module composes their disjoint nodes and edges into one graph that can be
passed directly to :class:`Gemma3ModalGeneratorGraphExecutor`.

Composition deliberately does *not* manufacture a combined compiler pipeline
or claim a new evaluation.  Each complete parent candidate remains nested and
strictly authenticated, while its independent evaluation, pipeline, and
caller-supplied guard-evidence lineage is projected into the bundle lineage.
The guard evidence is a source-safe reference only: this module binds its file
hash, logical hash, and passed status without opening the guarded corpus or
copying any guard rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

from .gemma3_state_conditioned_modal_graph_artifact import (
    GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
    _validate_and_restore_payload as _validate_parent_candidate_payload,
)
from .gemma3_layer17_v8_all_family_refit import (
    GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
    restore_gemma3_layer17_v8_all_family_refit_runtime,
    validate_gemma3_layer17_v8_all_family_refit_candidate,
)
from .modal_compiler_pipeline import ModalCompilerPipeline
from .modal_generator_graph import (
    ModalGeneratorGraphPlan,
    StateConditionedModalGeneratorInteraction,
)
from .modal_generator_lowering import ModalGeneratorLowering
from .parameter_cluster_fragments import ParameterClusterLayerFragment


__all__ = [
    "GEMMA3_LAYER10_LAYER17_COMPOSITION_FORMAT_VERSION",
    "GEMMA3_LAYER10_LAYER17_COMPOSITION_SCHEMA",
    "SourceSafeGuardEvidenceRecord",
    "build_gemma3_layer10_layer17_composition_bundle",
    "build_gemma3_layer10_layer17_composition_report",
    "load_gemma3_layer10_layer17_composition_bundle",
    "restore_gemma3_layer10_layer17_composition_runtime",
    "save_gemma3_layer10_layer17_composition_bundle",
]


GEMMA3_LAYER10_LAYER17_COMPOSITION_SCHEMA = (
    "fisher_graph.gemma3_layer10_layer17_modal_graph_composition_bundle"
)
GEMMA3_LAYER10_LAYER17_COMPOSITION_FORMAT_VERSION = 1

_BUNDLE_DOMAIN = (
    b"fisher_graph.gemma3_layer10_layer17_composition.bundle.v1\0"
)
_GROUP_CATALOG_DOMAIN = (
    b"fisher_graph.gemma3_layer10_layer17_composition.groups.v1\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_layer10_layer17_composition.report.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MACHINE_STRING = re.compile(r"^[^\s\x00-\x1f\x7f]{1,512}$")

_ROOT_FIELDS = {
    "schema",
    "format_version",
    "parents",
    "combined_edgeless_graph",
    "combined_dynamic_graph",
    "lineage",
    "safety",
    "composition_payload_sha256",
}
_PARENT_FIELDS = {
    "role",
    "layer_ordinals",
    "candidate_tensor_file",
    "candidate_tensor_file_sha256",
    "candidate_scientific_payload_sha256",
    "eval_split_sha256",
    "compiler_pipeline_sha256",
    "interaction_promotion_sha256",
    "guard_evidence",
    "candidate",
}
_GUARD_FIELDS = {
    "evidence_file_sha256",
    "logical_sha256",
    "status",
    "assessment_role",
    "heldout_confirmation",
    "fresh_validation",
}
_PARENT_SPECS = (("layer10", 10), ("layer17", 17))
_GUARD_PROFILE_BY_LAYER: dict[int, tuple[str, bool, bool]] = {
    10: ("claimed_closed_guard_assessment", True, False),
    17: ("open_development_assessment", False, False),
}

_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_raw_prompt_rows": False,
    "contains_raw_token_rows": False,
    "contains_raw_activation_rows": False,
    "contains_raw_gradient_rows": False,
    "contains_raw_latent_rows": False,
    "contains_target_residual_rows": False,
    "contains_teacher_outputs": False,
    "contains_source_model_weights": False,
    "contains_source_parameter_values": False,
    "contains_guard_evidence_payloads": False,
    "contains_guard_evidence_bindings": True,
    "contains_nested_parent_candidates": True,
    "contains_executable_lowerings": True,
    "contains_executable_graphs": True,
    "source_safe": True,
}


def _strict_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
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


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_filename(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _MACHINE_STRING.fullmatch(value) is None
        or Path(value).name != value
        or Path(value).suffix != ".pt"
    ):
        raise ValueError(f"{label} must be a source-safe .pt basename")
    return value


@dataclass(frozen=True, slots=True)
class SourceSafeGuardEvidenceRecord:
    """Hash-only evidence that one frozen parent passed its guard.

    ``evidence_file_sha256`` authenticates the evidence file as supplied by
    the caller. ``logical_sha256`` binds the evidence's canonical scientific
    or assessment payload.  The evidence file itself is intentionally not
    opened or nested here.
    """

    evidence_file_sha256: str
    logical_sha256: str
    status: str
    assessment_role: str
    heldout_confirmation: bool
    fresh_validation: bool

    def __post_init__(self) -> None:
        _require_sha256(
            self.evidence_file_sha256,
            label="guard evidence_file_sha256",
        )
        _require_sha256(
            self.logical_sha256,
            label="guard logical_sha256",
        )
        if (
            not isinstance(self.status, str)
            or _MACHINE_STRING.fullmatch(self.status) is None
        ):
            raise ValueError("guard evidence status must be canonical")
        if type(self.heldout_confirmation) is not bool:
            raise TypeError("guard heldout_confirmation must be boolean")
        if type(self.fresh_validation) is not bool:
            raise TypeError("guard fresh_validation must be boolean")
        profiles = {
            "claimed_closed_guard_assessment": (True, False),
            "open_development_assessment": (False, False),
        }
        expected = profiles.get(self.assessment_role)
        if expected is None:
            raise ValueError("guard assessment_role is unsupported")
        if (
            (self.heldout_confirmation, self.fresh_validation) != expected
        ):
            raise ValueError(
                "guard assessment role/confirmation/freshness combination "
                "is invalid"
            )
        if self.assessment_role == "claimed_closed_guard_assessment":
            if self.status != "passed":
                raise ValueError("closed guard evidence status must be 'passed'")
        elif self.status not in {
            "passed",
            "primary_nll_transfer_passed_mixed_secondary_fidelity",
        }:
            raise ValueError(
                "open-development guard evidence status is not qualified"
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "evidence_file_sha256": self.evidence_file_sha256,
            "logical_sha256": self.logical_sha256,
            "status": self.status,
            "assessment_role": self.assessment_role,
            "heldout_confirmation": self.heldout_confirmation,
            "fresh_validation": self.fresh_validation,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> SourceSafeGuardEvidenceRecord:
        _strict_fields(state, _GUARD_FIELDS, label="guard evidence")
        return cls(
            evidence_file_sha256=state[
                "evidence_file_sha256"
            ],  # type: ignore[arg-type]
            logical_sha256=state["logical_sha256"],  # type: ignore[arg-type]
            status=state["status"],  # type: ignore[arg-type]
            assessment_role=state[
                "assessment_role"
            ],  # type: ignore[arg-type]
            heldout_confirmation=state["heldout_confirmation"],  # type: ignore[arg-type]
            fresh_validation=state["fresh_validation"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class _ParentRuntime:
    role: str
    expected_layer_ordinal: int
    candidate: Mapping[str, object]
    lowerings_by_node: Mapping[str, ModalGeneratorLowering]
    edgeless_graph: ModalGeneratorGraphPlan
    dynamic_graph: ModalGeneratorGraphPlan
    compiler_pipeline: ModalCompilerPipeline | None
    fragments_by_node: Mapping[str, ParameterClusterLayerFragment]
    guard_evidence: SourceSafeGuardEvidenceRecord

    @property
    def layer_ordinals(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    fragment.layer_ordinal
                    for fragment in self.fragments_by_node.values()
                }
            )
        )

    @property
    def fragments(self) -> tuple[ParameterClusterLayerFragment, ...]:
        return tuple(
            self.fragments_by_node[node.name]
            for node in self.dynamic_graph.nodes
        )


def _selected_fragment(
    lowering: ModalGeneratorLowering,
) -> ParameterClusterLayerFragment:
    matches = tuple(
        fragment
        for fragment in lowering.fragment_plan.fragments
        if fragment.artifact_sha256 == lowering.selected_fragment_sha256
    )
    if len(matches) != 1:
        raise ValueError("lowering must bind exactly one selected fragment")
    return matches[0]


def _load_parent_candidate(path: Path) -> dict[str, object]:
    """Load one supported executable parent without weakening its validator.

    The original composition bundle accepted only the state-conditioned
    candidate envelope.  Post-LOFO layer-17 refits intentionally use a
    stricter, fit-authority-bearing envelope while exposing the same graph,
    lowering, and compiler-pipeline runtime contract.  Dispatch on the
    authenticated schema and then run that schema's complete validator.
    """

    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("composition parent artifact must contain one dict")
    schema = raw.get("schema")
    if schema == GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA:
        _validate_parent_candidate_payload(raw)
        return raw
    if schema == GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA:
        return validate_gemma3_layer17_v8_all_family_refit_candidate(raw)
    raise ValueError("composition parent candidate schema is unsupported")


def _restore_parent_runtime_contract(
    candidate: Mapping[str, object],
) -> tuple[
    tuple[tuple[str, ModalGeneratorLowering], ...],
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    ModalCompilerPipeline | None,
]:
    """Normalize supported parent envelopes to the executor contract."""

    schema = candidate.get("schema")
    if schema == GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA:
        return _validate_parent_candidate_payload(candidate)
    if schema == GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA:
        graph, lowerings_by_node, pipeline = (
            restore_gemma3_layer17_v8_all_family_refit_runtime(candidate)
        )
        lowerings = tuple(
            (node.name, lowerings_by_node[node.name]) for node in graph.nodes
        )
        return lowerings, graph, graph, pipeline
    raise ValueError("composition parent candidate schema is unsupported")


def _parent_eval_split_sha256(parent: _ParentRuntime) -> str:
    """Return the runtime lowering's authenticated evaluation-row binding.

    Legacy candidates call this field ``eval_split_sha256``.  The all-family
    refit names the same lowering-side role ``diagnostic_split_sha256`` to
    make clear that it is a within-fit rate-curve subset, not an assessment.
    The external assessment remains separately bound by guard evidence.
    """

    lineage = parent.candidate.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("parent candidate lineage is invalid")
    field = (
        "diagnostic_split_sha256"
        if parent.candidate.get("schema")
        == GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
        else "eval_split_sha256"
    )
    value = lineage.get(field)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("parent candidate evaluation split is invalid")
    bound = {
        lowering.coordinate_generator_plan.binding.eval_split_sha256
        for lowering in parent.lowerings_by_node.values()
    }
    if bound != {value}:
        raise ValueError(
            "parent candidate evaluation split differs from its lowerings"
        )
    return value


def _restore_parent_candidate(
    candidate: Mapping[str, object],
    *,
    role: str,
    expected_layer_ordinal: int,
    guard_evidence: SourceSafeGuardEvidenceRecord,
) -> _ParentRuntime:
    lowerings, edgeless, dynamic, pipeline = (
        _restore_parent_runtime_contract(candidate)
    )
    lowerings_by_node = dict(lowerings)
    fragments_by_node = {
        name: _selected_fragment(lowering)
        for name, lowering in lowerings
    }
    layers = tuple(
        sorted(
            {
                fragment.layer_ordinal
                for fragment in fragments_by_node.values()
            }
        )
    )
    if layers != (expected_layer_ordinal,):
        raise ValueError(
            f"{role} candidate must replace only layer "
            f"{expected_layer_ordinal}"
        )
    expected_guard_profile = _GUARD_PROFILE_BY_LAYER[expected_layer_ordinal]
    actual_guard_profile = (
        guard_evidence.assessment_role,
        guard_evidence.heldout_confirmation,
        guard_evidence.fresh_validation,
    )
    if actual_guard_profile != expected_guard_profile:
        raise ValueError(
            f"{role} guard evidence has the wrong scientific assessment "
            "profile"
        )
    fragment_hashes = tuple(
        fragment.artifact_sha256 for fragment in fragments_by_node.values()
    )
    if len(fragment_hashes) != len(set(fragment_hashes)):
        raise ValueError(f"{role} candidate repeats a selected fragment")
    group_indices = tuple(
        group
        for fragment in fragments_by_node.values()
        for group in fragment.group_indices
    )
    if len(group_indices) != len(set(group_indices)):
        raise ValueError(f"{role} candidate repeats a parameter group")
    if any(
        lowering.fragment_plan.artifact_sha256
        != dynamic.parameter_cluster_plan_sha256
        for lowering in lowerings_by_node.values()
    ):
        raise ValueError(f"{role} lowering fragment plan differs from graph")
    return _ParentRuntime(
        role=role,
        expected_layer_ordinal=expected_layer_ordinal,
        candidate=candidate,
        lowerings_by_node=lowerings_by_node,
        edgeless_graph=edgeless,
        dynamic_graph=dynamic,
        compiler_pipeline=pipeline,
        fragments_by_node=fragments_by_node,
        guard_evidence=guard_evidence,
    )


def _conditional_routing_groups(
    graph: ModalGeneratorGraphPlan,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                edge.routing_group_key
                for edge in graph.interactions
                if isinstance(
                    edge,
                    StateConditionedModalGeneratorInteraction,
                )
            }
        )
    )


def _validate_parent_pair(
    parents: tuple[_ParentRuntime, _ParentRuntime],
) -> None:
    first, second = parents
    if (
        first.dynamic_graph.model_fingerprint
        != second.dynamic_graph.model_fingerprint
    ):
        raise ValueError("parent candidates bind different source models")
    if (
        first.dynamic_graph.parameter_cluster_plan_sha256
        != second.dynamic_graph.parameter_cluster_plan_sha256
    ):
        raise ValueError("parent candidates bind different fragment plans")

    for metadata_key in ("model_id", "requested_revision"):
        values: list[object] = []
        for parent in parents:
            experiment = parent.candidate.get("experiment")
            if isinstance(experiment, Mapping) and metadata_key in experiment:
                values.append(experiment[metadata_key])
        if len(values) == 2 and values[0] != values[1]:
            raise ValueError(
                f"parent candidates disagree on {metadata_key}"
            )

    node_sets = tuple(
        {node.name for node in parent.dynamic_graph.nodes}
        for parent in parents
    )
    if node_sets[0] & node_sets[1]:
        raise ValueError("parent candidates contain overlapping graph nodes")

    fragment_id_sets = tuple(
        {fragment.fragment_id for fragment in parent.fragments}
        for parent in parents
    )
    fragment_hash_sets = tuple(
        {fragment.artifact_sha256 for fragment in parent.fragments}
        for parent in parents
    )
    if (
        fragment_id_sets[0] & fragment_id_sets[1]
        or fragment_hash_sets[0] & fragment_hash_sets[1]
    ):
        raise ValueError("parent candidates contain overlapping fragments")

    group_sets = tuple(
        {
            group
            for fragment in parent.fragments
            for group in fragment.group_indices
        }
        for parent in parents
    )
    if group_sets[0] & group_sets[1]:
        raise ValueError(
            "parent candidates contain overlapping parameter groups"
        )

    layer_sets = tuple(set(parent.layer_ordinals) for parent in parents)
    if layer_sets[0] & layer_sets[1]:
        raise ValueError("parent candidates contain overlapping layers")

    routing_sets = tuple(
        set(_conditional_routing_groups(parent.dynamic_graph))
        for parent in parents
    )
    if routing_sets[0] & routing_sets[1]:
        raise ValueError(
            "parent candidates contain overlapping conditional routing groups"
        )


def _compose_graphs(
    parents: tuple[_ParentRuntime, _ParentRuntime],
) -> tuple[
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
]:
    _validate_parent_pair(parents)
    nodes = tuple(
        sorted(
            (
                node
                for parent in parents
                for node in parent.dynamic_graph.nodes
            ),
            key=lambda node: (node.causal_order, node.name),
        )
    )
    interactions = tuple(
        sorted(
            (
                edge
                for parent in parents
                for edge in parent.dynamic_graph.interactions
            ),
            key=lambda edge: (edge.source_node, edge.target_node),
        )
    )
    model_fingerprint = parents[0].dynamic_graph.model_fingerprint
    fragment_plan_sha256 = (
        parents[0].dynamic_graph.parameter_cluster_plan_sha256
    )
    edgeless = ModalGeneratorGraphPlan(
        model_fingerprint=model_fingerprint,
        parameter_cluster_plan_sha256=fragment_plan_sha256,
        nodes=nodes,
        interactions=(),
    )
    dynamic = ModalGeneratorGraphPlan(
        model_fingerprint=model_fingerprint,
        parameter_cluster_plan_sha256=fragment_plan_sha256,
        nodes=nodes,
        interactions=interactions,
    )
    lowerings = {
        node.name: next(
            parent.lowerings_by_node[node.name]
            for parent in parents
            if node.name in parent.lowerings_by_node
        )
        for node in nodes
    }
    return edgeless, dynamic, lowerings


def _parent_lineage(parent: _ParentRuntime) -> dict[str, object]:
    lineage = parent.candidate.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("parent candidate lineage is invalid")
    promotion = (
        None
        if parent.compiler_pipeline is None
        else parent.compiler_pipeline.interaction_selection
    )
    return {
        "role": parent.role,
        "layer_ordinals": parent.layer_ordinals,
        "candidate_scientific_payload_sha256": parent.candidate[
            "scientific_payload_sha256"
        ],
        "eval_split_sha256": _parent_eval_split_sha256(parent),
        "compiler_pipeline_sha256": (
            None
            if parent.compiler_pipeline is None
            else parent.compiler_pipeline.artifact_sha256
        ),
        "interaction_promotion_sha256": (
            None
            if promotion is None
            else getattr(promotion, "artifact_sha256", None)
        ),
        "guard_evidence_file_sha256": (
            parent.guard_evidence.evidence_file_sha256
        ),
        "guard_evidence_logical_sha256": (
            parent.guard_evidence.logical_sha256
        ),
        "guard_status": parent.guard_evidence.status,
        "guard_assessment_role": parent.guard_evidence.assessment_role,
        "guard_heldout_confirmation": (
            parent.guard_evidence.heldout_confirmation
        ),
        "guard_fresh_validation": parent.guard_evidence.fresh_validation,
    }


def _composition_lineage(
    parents: tuple[_ParentRuntime, _ParentRuntime],
    *,
    edgeless: ModalGeneratorGraphPlan,
    dynamic: ModalGeneratorGraphPlan,
) -> dict[str, object]:
    fragments = tuple(
        parent.fragments_by_node[node.name]
        for node in dynamic.nodes
        for parent in parents
        if node.name in parent.fragments_by_node
    )
    group_catalog = tuple(
        sorted(
            group
            for fragment in fragments
            for group in fragment.group_indices
        )
    )
    parent_lineage = tuple(_parent_lineage(parent) for parent in parents)
    return {
        "model_fingerprint": dynamic.model_fingerprint,
        "parameter_cluster_plan_sha256": (
            dynamic.parameter_cluster_plan_sha256
        ),
        "parent_lineage": parent_lineage,
        "layer_ordinals": tuple(
            fragment.layer_ordinal for fragment in fragments
        ),
        "node_names": dynamic.traversal_order,
        "graph_node_sha256s": tuple(
            node.artifact_sha256 for node in dynamic.nodes
        ),
        "lowering_sha256s": tuple(
            parent.lowerings_by_node[node.name].artifact_sha256
            for node in dynamic.nodes
            for parent in parents
            if node.name in parent.lowerings_by_node
        ),
        "fragment_ids": tuple(
            fragment.fragment_id for fragment in fragments
        ),
        "fragment_sha256s": tuple(
            fragment.artifact_sha256 for fragment in fragments
        ),
        "parameter_group_count": len(group_catalog),
        "parameter_group_catalog_sha256": _json_sha256(
            group_catalog,
            domain=_GROUP_CATALOG_DOMAIN,
        ),
        "conditional_routing_groups": tuple(
            _conditional_routing_groups(parent.dynamic_graph)
            for parent in parents
        ),
        "combined_edgeless_graph_sha256": edgeless.artifact_sha256,
        "combined_dynamic_graph_sha256": dynamic.artifact_sha256,
        "node_count": len(dynamic.nodes),
        "dynamic_interaction_count": len(dynamic.interactions),
        "edgeless_parameter_count": edgeless.parameter_count,
        "dynamic_parameter_count": dynamic.parameter_count,
        "edgeless_dense_macs_per_token": edgeless.macs_per_token,
        "dynamic_dense_macs_per_token": dynamic.macs_per_token,
        "conditional_routing_macs_per_token": (
            dynamic.conditional_routing_macs_per_token
        ),
        "conditional_selected_message_macs_per_token_upper_bound": (
            dynamic.conditional_selected_message_macs_per_token_upper_bound
        ),
    }


def _parent_binding_projection(
    record: Mapping[str, object],
) -> dict[str, object]:
    return {key: record[key] for key in sorted(_PARENT_FIELDS - {"candidate"})}


def _bundle_projection(value: Mapping[str, object]) -> dict[str, object]:
    parents = value["parents"]
    if type(parents) is not tuple:
        raise TypeError("composition parents must be a tuple")
    return {
        "schema": value["schema"],
        "format_version": value["format_version"],
        "parents": tuple(
            _parent_binding_projection(record)  # type: ignore[arg-type]
            for record in parents
        ),
        "combined_edgeless_graph_sha256": value[
            "lineage"
        ]["combined_edgeless_graph_sha256"],  # type: ignore[index]
        "combined_dynamic_graph_sha256": value[
            "lineage"
        ]["combined_dynamic_graph_sha256"],  # type: ignore[index]
        "lineage": value["lineage"],
        "safety": value["safety"],
    }


def _composition_sha256(value: Mapping[str, object]) -> str:
    return _json_sha256(_bundle_projection(value), domain=_BUNDLE_DOMAIN)


def _build_parent_record(
    *,
    path: Path,
    candidate: Mapping[str, object],
    parent: _ParentRuntime,
) -> dict[str, object]:
    lineage = _parent_lineage(parent)
    return {
        "role": parent.role,
        "layer_ordinals": parent.layer_ordinals,
        "candidate_tensor_file": _tensor_filename(
            path.name,
            label="candidate tensor file",
        ),
        "candidate_tensor_file_sha256": _file_sha256(path),
        "candidate_scientific_payload_sha256": candidate[
            "scientific_payload_sha256"
        ],
        "eval_split_sha256": lineage["eval_split_sha256"],
        "compiler_pipeline_sha256": lineage[
            "compiler_pipeline_sha256"
        ],
        "interaction_promotion_sha256": lineage[
            "interaction_promotion_sha256"
        ],
        "guard_evidence": parent.guard_evidence.state_dict(),
        "candidate": candidate,
    }


def _restore_parent_record(
    record: Mapping[str, object],
    *,
    expected_role: str,
    expected_layer_ordinal: int,
) -> _ParentRuntime:
    _strict_fields(record, _PARENT_FIELDS, label=f"{expected_role} parent")
    if record["role"] != expected_role:
        raise ValueError("composition parent roles are not canonical")
    _tensor_filename(
        record["candidate_tensor_file"],
        label=f"{expected_role} candidate tensor file",
    )
    _require_sha256(
        record["candidate_tensor_file_sha256"],
        label=f"{expected_role} candidate tensor file hash",
    )
    guard = SourceSafeGuardEvidenceRecord.from_state_dict(
        record["guard_evidence"],  # type: ignore[arg-type]
    )
    candidate = record["candidate"]
    if not isinstance(candidate, Mapping):
        raise TypeError("nested parent candidate must be a mapping")
    parent = _restore_parent_candidate(
        candidate,
        role=expected_role,
        expected_layer_ordinal=expected_layer_ordinal,
        guard_evidence=guard,
    )
    lineage = _parent_lineage(parent)
    expected = {
        "layer_ordinals": parent.layer_ordinals,
        "candidate_scientific_payload_sha256": candidate[
            "scientific_payload_sha256"
        ],
        "eval_split_sha256": lineage["eval_split_sha256"],
        "compiler_pipeline_sha256": lineage[
            "compiler_pipeline_sha256"
        ],
        "interaction_promotion_sha256": lineage[
            "interaction_promotion_sha256"
        ],
    }
    for key, value in expected.items():
        if record[key] != value:
            raise ValueError(f"{expected_role} parent {key} drifted")
    return parent


def _validate_and_restore_bundle(
    raw: Mapping[str, object],
) -> tuple[
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    tuple[ModalGeneratorLowering, ...],
]:
    _strict_fields(raw, _ROOT_FIELDS, label="composition bundle")
    if (
        raw["schema"] != GEMMA3_LAYER10_LAYER17_COMPOSITION_SCHEMA
        or raw["format_version"]
        != GEMMA3_LAYER10_LAYER17_COMPOSITION_FORMAT_VERSION
    ):
        raise ValueError("unsupported layer10+17 composition bundle")
    if raw["safety"] != _SAFETY:
        raise ValueError("composition bundle safety flags are invalid")
    parents_state = raw["parents"]
    if type(parents_state) is not tuple or len(parents_state) != 2:
        raise ValueError("composition bundle requires two canonical parents")
    parents = tuple(
        _restore_parent_record(
            record,  # type: ignore[arg-type]
            expected_role=role,
            expected_layer_ordinal=layer,
        )
        for record, (role, layer) in zip(
            parents_state,
            _PARENT_SPECS,
            strict=True,
        )
    )
    expected_edgeless, expected_dynamic, lowerings_by_node = _compose_graphs(
        parents  # type: ignore[arg-type]
    )
    serialized_edgeless = raw["combined_edgeless_graph"]
    serialized_dynamic = raw["combined_dynamic_graph"]
    if not isinstance(serialized_edgeless, Mapping) or not isinstance(
        serialized_dynamic,
        Mapping,
    ):
        raise TypeError("combined graph states must be mappings")
    edgeless = ModalGeneratorGraphPlan.from_state_dict(serialized_edgeless)
    dynamic = ModalGeneratorGraphPlan.from_state_dict(serialized_dynamic)
    if (
        edgeless.artifact_sha256 != expected_edgeless.artifact_sha256
        or dynamic.artifact_sha256 != expected_dynamic.artifact_sha256
    ):
        raise ValueError("combined graphs are not the canonical parent union")
    expected_lineage = _composition_lineage(
        parents,  # type: ignore[arg-type]
        edgeless=edgeless,
        dynamic=dynamic,
    )
    if raw["lineage"] != expected_lineage:
        raise ValueError("composition bundle lineage is inconsistent")
    _require_sha256(
        raw["composition_payload_sha256"],
        label="composition_payload_sha256",
    )
    if raw["composition_payload_sha256"] != _composition_sha256(raw):
        raise ValueError("composition bundle payload hash mismatch")
    lowerings = tuple(
        lowerings_by_node[node.name] for node in dynamic.nodes
    )
    return edgeless, dynamic, lowerings


def build_gemma3_layer10_layer17_composition_bundle(
    *,
    layer10_candidate_path: Path | str,
    layer10_guard_evidence: SourceSafeGuardEvidenceRecord,
    layer17_candidate_path: Path | str,
    layer17_guard_evidence: SourceSafeGuardEvidenceRecord,
) -> dict[str, object]:
    """Strict-load both parents and build their canonical executable union."""

    paths = (Path(layer10_candidate_path), Path(layer17_candidate_path))
    evidence = (layer10_guard_evidence, layer17_guard_evidence)
    if any(not isinstance(value, SourceSafeGuardEvidenceRecord) for value in evidence):
        raise TypeError(
            "guard evidence must be SourceSafeGuardEvidenceRecord values"
        )
    candidates = tuple(_load_parent_candidate(path) for path in paths)
    parents = tuple(
        _restore_parent_candidate(
            candidate,
            role=role,
            expected_layer_ordinal=layer,
            guard_evidence=guard,
        )
        for candidate, guard, (role, layer) in zip(
            candidates,
            evidence,
            _PARENT_SPECS,
            strict=True,
        )
    )
    edgeless, dynamic, _ = _compose_graphs(parents)  # type: ignore[arg-type]
    parent_records = tuple(
        _build_parent_record(
            path=path,
            candidate=candidate,
            parent=parent,
        )
        for path, candidate, parent in zip(
            paths,
            candidates,
            parents,
            strict=True,
        )
    )
    lineage = _composition_lineage(
        parents,  # type: ignore[arg-type]
        edgeless=edgeless,
        dynamic=dynamic,
    )
    without_digest: dict[str, object] = {
        "schema": GEMMA3_LAYER10_LAYER17_COMPOSITION_SCHEMA,
        "format_version": GEMMA3_LAYER10_LAYER17_COMPOSITION_FORMAT_VERSION,
        "parents": parent_records,
        "combined_edgeless_graph": edgeless.state_dict(),
        "combined_dynamic_graph": dynamic.state_dict(),
        "lineage": lineage,
        "safety": dict(_SAFETY),
    }
    payload = {
        **without_digest,
        "composition_payload_sha256": _composition_sha256(without_digest),
    }
    _validate_and_restore_bundle(payload)
    return payload


def restore_gemma3_layer10_layer17_composition_runtime(
    raw: Mapping[str, object],
) -> tuple[
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    tuple[ModalGeneratorLowering, ...],
]:
    """Return ``(edgeless, dynamic, lowerings)`` for one Gemma executor.

    For example, the dynamic runtime is constructed as
    ``Gemma3ModalGeneratorGraphExecutor(adapter, dynamic, lowerings)``.
    """

    return _validate_and_restore_bundle(raw)


def _json_native(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_native(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_native(child) for child in value]
    return value


def build_gemma3_layer10_layer17_composition_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    """Build a compact tensor-free JSON report for a strict bundle."""

    _validate_and_restore_bundle(payload)
    filename = _tensor_filename(tensor_file, label="tensor_file")
    parents = payload["parents"]
    if type(parents) is not tuple:
        raise TypeError("composition parents must be a tuple")
    without_digest = {
        "schema": payload["schema"],
        "format_version": payload["format_version"],
        "parents": tuple(
            _parent_binding_projection(record)  # type: ignore[arg-type]
            for record in parents
        ),
        "lineage": payload["lineage"],
        "safety": payload["safety"],
        "artifact": {
            "tensor_file": filename,
            "composition_payload_sha256": payload[
                "composition_payload_sha256"
            ],
        },
    }
    report = {
        **without_digest,
        "report_sha256": _json_sha256(without_digest, domain=_REPORT_DOMAIN),
    }
    native = _json_native(report)
    if not isinstance(native, dict):
        raise AssertionError("composition report root is not a dict")
    return native


def save_gemma3_layer10_layer17_composition_bundle(
    output: Path | str,
    **build_arguments: object,
) -> dict[str, object]:
    """Save the strict bundle and compact report, refusing overwrite."""

    path = Path(output)
    if path.suffix != ".pt":
        raise ValueError("composition bundle output must use .pt")
    report_path = path.with_suffix(".json")
    if path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite composition bundle")
    payload = build_gemma3_layer10_layer17_composition_bundle(
        **build_arguments,  # type: ignore[arg-type]
    )
    report = build_gemma3_layer10_layer17_composition_report(
        payload,
        tensor_file=path.name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    created_tensor = False
    created_report = False
    try:
        with path.open("xb") as handle:
            created_tensor = True
            torch.save(payload, handle)
        with report_path.open("x", encoding="utf-8") as handle:
            created_report = True
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        if created_report and report_path.exists():
            report_path.unlink()
        if created_tensor and path.exists():
            path.unlink()
        raise
    return report


def load_gemma3_layer10_layer17_composition_bundle(
    input_path: Path | str,
) -> dict[str, object]:
    """Strict-load and authenticate both parents and their graph union."""

    path = Path(input_path)
    if path.suffix != ".pt":
        raise ValueError("composition bundle input must use .pt")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("composition bundle artifact must be a dict")
    _validate_and_restore_bundle(raw)
    return raw
