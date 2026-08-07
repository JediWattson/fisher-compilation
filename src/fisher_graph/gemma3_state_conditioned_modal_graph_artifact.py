"""Strict source-safe artifacts for Gemma state-conditioned modal graphs.

This module owns the serialization boundary for one frozen candidate.  The
candidate contains authenticated modal-generator lowerings, an exact edgeless
control graph, an exact state-conditioned graph, and optionally the promoted
compiler pipeline.  Experiment-facing metadata is canonical JSON data only;
prompt text, token ids, raw fit/evaluation rows, teacher outputs, and source
weights are rejected.

The outer scientific digest authenticates metadata and the hashes of nested
artifacts rather than duplicating their tensor payloads in the digest input.
Every nested artifact is still strict-loaded and independently authenticated
before the candidate is accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor

from .modal_compiler_pipeline import ModalCompilerPipeline
from .modal_generator_graph import (
    ModalGeneratorGraphPlan,
    StateConditionedModalGeneratorInteraction,
)
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_interaction_promotion import ModalInteractionGraphPromotion


__all__ = [
    "GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_FORMAT_VERSION",
    "GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA",
    "build_gemma3_state_conditioned_modal_graph_candidate",
    "build_gemma3_state_conditioned_modal_graph_report",
    "load_gemma3_state_conditioned_modal_graph_candidate",
    "save_gemma3_state_conditioned_modal_graph_candidate",
]


GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA = (
    "fisher_graph.gemma3_state_conditioned_modal_graph_candidate"
)
GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_FORMAT_VERSION = 1

_SCIENTIFIC_DOMAIN = (
    b"fisher_graph.gemma3_state_conditioned_modal_graph.scientific.v1\0"
)
_REPORT_DOMAIN = (
    b"fisher_graph.gemma3_state_conditioned_modal_graph.report.v1\0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MACHINE_STRING = re.compile(r"^[^\s\x00-\x1f\x7f]{1,512}$")

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
    "contains_executable_lowerings": True,
    "contains_executable_graphs": True,
    "contains_promoted_compiler_pipeline": False,
    "source_safe": True,
}

_ROOT_FIELDS = {
    "schema",
    "format_version",
    "experiment",
    "config",
    "splits",
    "selection",
    "resources",
    "lowering_records",
    "edgeless_graph",
    "dynamic_graph",
    "compiler_pipeline",
    "lineage",
    "safety",
    "scientific_payload_sha256",
}
_LOWERING_RECORD_FIELDS = {
    "node_name",
    "causal_order",
    "input_boundary",
    "output_boundary",
    "lowering_sha256",
    "graph_node_sha256",
    "lowering",
}
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "input_ids",
        "token_ids",
        "tokens",
        "tokenizer",
        "tokenizer_state",
        "raw_rows",
        "raw_fit_rows",
        "raw_eval_rows",
        "raw_prompt_rows",
        "raw_token_rows",
        "activation_rows",
        "gradient_rows",
        "latent_rows",
        "target_residual_rows",
        "target_messages",
        "teacher_outputs",
        "teacher_logits",
        "source_model_weights",
        "source_parameter_values",
        "model_state_dict",
        "source_state_dict",
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


def _strict_fields(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _is_forbidden_metadata_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized in _FORBIDDEN_METADATA_KEYS:
        return True
    if normalized.startswith("raw_") and normalized.endswith(
        ("rows", "tokens", "activations", "gradients", "latents")
    ):
        return True
    return normalized.endswith(
        (
            "_source_model_weights",
            "_source_parameter_values",
            "_teacher_outputs",
            "_teacher_logits",
            "_token_ids",
            "_input_ids",
        )
    )


def _canonical_json_value(value: object, *, path: str) -> object:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} may not contain Tensor rows or weights")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains an invalid metadata key")
        for key in sorted(value):  # type: ignore[type-var]
            if (
                not _MACHINE_STRING.fullmatch(key)
            ):
                raise ValueError(f"{path} contains an invalid metadata key")
            if _is_forbidden_metadata_key(key):
                raise ValueError(f"{path} contains forbidden field {key!r}")
            result[key] = _canonical_json_value(
                value[key],
                path=f"{path}.{key}",
            )
        return result
    if isinstance(value, (tuple, list)):
        return tuple(
            _canonical_json_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return float(value)
    if isinstance(value, str) and _MACHINE_STRING.fullmatch(value):
        return value
    raise ValueError(f"{path} contains a non-JSON-safe value")


def _canonical_metadata(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    canonical = _canonical_json_value(value, path=label)
    if not isinstance(canonical, dict):
        raise AssertionError("canonical metadata root is not a dict")
    return canonical


def _authenticated_copy(value: object, expected_type: type[Any]) -> Any:
    if not isinstance(value, expected_type):
        raise TypeError(f"artifact must be {expected_type.__name__}")
    validator = getattr(value, "validate_integrity", None)
    if callable(validator):
        validator()
    state_builder = getattr(value, "state_dict", None)
    state_loader = getattr(type(value), "from_state_dict", None)
    if not callable(state_builder) or not callable(state_loader):
        raise TypeError("artifact lacks a strict authenticated state roundtrip")
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


def _graph_node_hashes(graph: ModalGeneratorGraphPlan) -> tuple[str, ...]:
    return tuple(node.artifact_sha256 for node in graph.nodes)


def _validate_graph_pair(
    edgeless_graph: ModalGeneratorGraphPlan,
    dynamic_graph: ModalGeneratorGraphPlan,
) -> None:
    if edgeless_graph.interactions:
        raise ValueError("edgeless control graph must not contain interactions")
    conditional_count = sum(
        isinstance(edge, StateConditionedModalGeneratorInteraction)
        for edge in dynamic_graph.interactions
    )
    if not dynamic_graph.interactions or conditional_count <= 0:
        raise ValueError(
            "dynamic graph requires state-conditioned interactions"
        )
    if (
        edgeless_graph.model_fingerprint
        != dynamic_graph.model_fingerprint
        or edgeless_graph.parameter_cluster_plan_sha256
        != dynamic_graph.parameter_cluster_plan_sha256
        or _graph_node_hashes(edgeless_graph)
        != _graph_node_hashes(dynamic_graph)
    ):
        raise ValueError(
            "edgeless and dynamic graphs must contain identical nodes"
        )
    if edgeless_graph.artifact_sha256 == dynamic_graph.artifact_sha256:
        raise ValueError("edgeless and dynamic graph hashes must differ")


def _coerce_lowerings(
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    graph: ModalGeneratorGraphPlan,
) -> tuple[dict[str, object], ...]:
    if not isinstance(lowerings_by_node, Mapping):
        raise TypeError("lowerings_by_node must be a mapping")
    names = tuple(node.name for node in graph.nodes)
    if set(lowerings_by_node) != set(names):
        raise ValueError("lowering names must exactly match graph nodes")
    records: list[dict[str, object]] = []
    for node in graph.nodes:
        lowering = _authenticated_copy(
            lowerings_by_node[node.name],
            ModalGeneratorLowering,
        )
        reconstructed = lowering.to_graph_node(
            name=node.name,
            causal_order=node.causal_order,
            input_boundary=node.input_boundary,
            output_boundary=node.output_boundary,
        )
        if reconstructed.artifact_sha256 != node.artifact_sha256:
            raise ValueError(
                "lowering does not reconstruct the exact graph node"
            )
        records.append(
            {
                "node_name": node.name,
                "causal_order": node.causal_order,
                "input_boundary": node.input_boundary,
                "output_boundary": node.output_boundary,
                "lowering_sha256": lowering.artifact_sha256,
                "graph_node_sha256": node.artifact_sha256,
                "lowering": lowering.state_dict(),
            }
        )
    return tuple(records)


def _restore_lowering_records(
    raw_records: object,
    graph: ModalGeneratorGraphPlan,
) -> tuple[tuple[str, ModalGeneratorLowering], ...]:
    if type(raw_records) is not tuple or not raw_records:
        raise TypeError("serialized lowering_records must be a nonempty tuple")
    if len(raw_records) != len(graph.nodes):
        raise ValueError("lowering record count differs from graph nodes")
    restored: list[tuple[str, ModalGeneratorLowering]] = []
    for index, (raw, node) in enumerate(
        zip(raw_records, graph.nodes, strict=True)
    ):
        _strict_fields(
            raw,  # type: ignore[arg-type]
            _LOWERING_RECORD_FIELDS,
            label=f"lowering_records[{index}]",
        )
        lowering = ModalGeneratorLowering.from_state_dict(
            raw["lowering"]  # type: ignore[index,arg-type]
        )
        reconstructed = lowering.to_graph_node(
            name=raw["node_name"],  # type: ignore[index,arg-type]
            causal_order=raw["causal_order"],  # type: ignore[index,arg-type]
            input_boundary=raw["input_boundary"],  # type: ignore[index,arg-type]
            output_boundary=raw["output_boundary"],  # type: ignore[index,arg-type]
        )
        if (
            raw["node_name"] != node.name  # type: ignore[index]
            or raw["causal_order"] != node.causal_order  # type: ignore[index]
            or raw["input_boundary"] != node.input_boundary  # type: ignore[index]
            or raw["output_boundary"] != node.output_boundary  # type: ignore[index]
            or raw["lowering_sha256"] != lowering.artifact_sha256  # type: ignore[index]
            or raw["graph_node_sha256"] != node.artifact_sha256  # type: ignore[index]
            or reconstructed.artifact_sha256 != node.artifact_sha256
        ):
            raise ValueError(
                "serialized lowering does not reconstruct its exact graph node"
            )
        restored.append((node.name, lowering))
    return tuple(restored)


def _shared_split_hashes(
    lowerings: Sequence[tuple[str, ModalGeneratorLowering]],
) -> tuple[str, str]:
    fit_hashes: set[str] = set()
    eval_hashes: set[str] = set()
    for _, lowering in lowerings:
        generator_binding = lowering.coordinate_generator_plan.binding
        basis_binding = lowering.computational_mode_basis.binding
        if (
            generator_binding.fit_split_sha256
            != basis_binding.fit_split_sha256
            or generator_binding.eval_split_sha256
            != basis_binding.eval_split_sha256
        ):
            raise ValueError("lowering generator and basis split lineage differs")
        fit_hashes.add(generator_binding.fit_split_sha256)
        eval_hashes.add(generator_binding.eval_split_sha256)
    if len(fit_hashes) != 1 or len(eval_hashes) != 1:
        raise ValueError("all candidate lowerings must share fit/eval splits")
    fit = next(iter(fit_hashes))
    evaluation = next(iter(eval_hashes))
    if fit == evaluation:
        raise ValueError("candidate fit and evaluation splits must differ")
    return fit, evaluation


def _validate_optional_pipeline(
    pipeline: ModalCompilerPipeline | None,
    *,
    graph: ModalGeneratorGraphPlan,
    lowerings: Sequence[tuple[str, ModalGeneratorLowering]],
) -> None:
    if pipeline is None:
        return
    if pipeline.graph_plan.artifact_sha256 != graph.artifact_sha256:
        raise ValueError("promoted pipeline does not contain the dynamic graph")
    if not isinstance(
        pipeline.interaction_selection,
        ModalInteractionGraphPromotion,
    ):
        raise ValueError(
            "candidate pipeline must contain conditional graph promotion"
        )
    pipeline.interaction_selection.validate_against_graph(graph)
    lowering_hashes = {
        name: lowering.artifact_sha256 for name, lowering in lowerings
    }
    if tuple(node.node_name for node in pipeline.nodes) != tuple(
        node.name for node in graph.nodes
    ):
        raise ValueError("pipeline nodes differ from dynamic graph traversal")
    for node in pipeline.nodes:
        if lowering_hashes.get(node.node_name) != node.lowering.artifact_sha256:
            raise ValueError("pipeline lowering differs from candidate lowering")


def _lineage(
    *,
    lowerings: Sequence[tuple[str, ModalGeneratorLowering]],
    edgeless_graph: ModalGeneratorGraphPlan,
    dynamic_graph: ModalGeneratorGraphPlan,
    compiler_pipeline: ModalCompilerPipeline | None,
) -> dict[str, object]:
    fit_split, eval_split = _shared_split_hashes(lowerings)
    conditional_count = sum(
        isinstance(edge, StateConditionedModalGeneratorInteraction)
        for edge in dynamic_graph.interactions
    )
    promotion = (
        None
        if compiler_pipeline is None
        else compiler_pipeline.interaction_selection
    )
    if promotion is not None and not isinstance(
        promotion,
        ModalInteractionGraphPromotion,
    ):
        raise ValueError("compiler pipeline promotion type is invalid")
    selected_message_macs = (
        dynamic_graph.conditional_selected_message_macs_per_token_upper_bound
    )
    return {
        "model_fingerprint": dynamic_graph.model_fingerprint,
        "parameter_cluster_plan_sha256": (
            dynamic_graph.parameter_cluster_plan_sha256
        ),
        "fit_split_sha256": fit_split,
        "eval_split_sha256": eval_split,
        "node_names": tuple(node.name for node in dynamic_graph.nodes),
        "graph_node_sha256s": _graph_node_hashes(dynamic_graph),
        "lowering_sha256s": tuple(
            lowering.artifact_sha256 for _, lowering in lowerings
        ),
        "edgeless_graph_sha256": edgeless_graph.artifact_sha256,
        "dynamic_graph_sha256": dynamic_graph.artifact_sha256,
        "compiler_pipeline_sha256": (
            None
            if compiler_pipeline is None
            else compiler_pipeline.artifact_sha256
        ),
        "interaction_promotion_sha256": (
            None if promotion is None else promotion.artifact_sha256
        ),
        "node_count": len(dynamic_graph.nodes),
        "dynamic_interaction_count": len(dynamic_graph.interactions),
        "state_conditioned_interaction_count": conditional_count,
        "edgeless_parameter_count": edgeless_graph.parameter_count,
        "dynamic_parameter_count": dynamic_graph.parameter_count,
        "edgeless_dense_macs_per_token": edgeless_graph.macs_per_token,
        "dynamic_dense_macs_per_token": dynamic_graph.macs_per_token,
        "conditional_routing_macs_per_token": (
            dynamic_graph.conditional_routing_macs_per_token
        ),
        "conditional_dense_message_macs_per_token": (
            dynamic_graph.conditional_dense_message_macs_per_token
        ),
        "conditional_selected_message_macs_per_token_upper_bound": (
            selected_message_macs
        ),
        "source_parameter_count": (
            None
            if compiler_pipeline is None
            else compiler_pipeline.source_parameter_count
        ),
        "source_macs_per_token": (
            None
            if compiler_pipeline is None
            else compiler_pipeline.source_macs_per_token
        ),
        "net_parameter_savings": (
            None
            if compiler_pipeline is None
            else compiler_pipeline.net_parameter_savings
        ),
        "net_dense_macs_saved_per_token": (
            None
            if compiler_pipeline is None
            else compiler_pipeline.net_macs_saved_per_token
        ),
    }


def _validate_declared_metadata_bindings(
    *,
    splits: Mapping[str, object],
    selection: Mapping[str, object],
    resources: Mapping[str, object],
    lineage: Mapping[str, object],
) -> None:
    expected_by_key = {
        "fit_split_sha256": lineage["fit_split_sha256"],
        "eval_split_sha256": lineage["eval_split_sha256"],
        "selection_split_sha256": lineage["eval_split_sha256"],
        "dynamic_graph_sha256": lineage["dynamic_graph_sha256"],
        "edgeless_graph_sha256": lineage["edgeless_graph_sha256"],
        "compiler_pipeline_sha256": lineage["compiler_pipeline_sha256"],
        "interaction_promotion_sha256": lineage[
            "interaction_promotion_sha256"
        ],
        "edgeless_parameter_count": lineage["edgeless_parameter_count"],
        "dynamic_parameter_count": lineage["dynamic_parameter_count"],
        "edgeless_dense_macs_per_token": lineage[
            "edgeless_dense_macs_per_token"
        ],
        "dynamic_dense_macs_per_token": lineage[
            "dynamic_dense_macs_per_token"
        ],
        "conditional_routing_macs_per_token": lineage[
            "conditional_routing_macs_per_token"
        ],
        "conditional_dense_message_macs_per_token": lineage[
            "conditional_dense_message_macs_per_token"
        ],
        "conditional_selected_message_macs_per_token_upper_bound": lineage[
            "conditional_selected_message_macs_per_token_upper_bound"
        ],
        "source_parameter_count": lineage["source_parameter_count"],
        "source_macs_per_token": lineage["source_macs_per_token"],
        "net_parameter_savings": lineage["net_parameter_savings"],
        "net_dense_macs_saved_per_token": lineage[
            "net_dense_macs_saved_per_token"
        ],
    }
    for metadata, label in (
        (splits, "splits"),
        (selection, "selection"),
        (resources, "resources"),
    ):
        for key, expected in expected_by_key.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"{label}.{key} contradicts nested artifacts")

    promotion_hash_keys = {
        "compiler_pipeline_sha256",
        "interaction_promotion_sha256",
    }
    declared_promotion_hash_keys = promotion_hash_keys & set(selection)
    if declared_promotion_hash_keys and (
        declared_promotion_hash_keys != promotion_hash_keys
    ):
        raise ValueError(
            "selection promotion hashes must be declared together"
        )
    if "promotion_passed" in selection:
        if type(selection["promotion_passed"]) is not bool:
            raise ValueError("selection.promotion_passed must be boolean")
        if declared_promotion_hash_keys != promotion_hash_keys:
            raise ValueError(
                "selection.promotion_passed requires both promotion hashes"
            )
        pipeline_present = lineage["compiler_pipeline_sha256"] is not None
        if selection["promotion_passed"] is not pipeline_present:
            raise ValueError(
                "selection.promotion_passed contradicts compiler pipeline"
            )


def _scientific_projection(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": value["schema"],
        "format_version": value["format_version"],
        "experiment": value["experiment"],
        "config": value["config"],
        "splits": value["splits"],
        "selection": value["selection"],
        "resources": value["resources"],
        "lineage": value["lineage"],
        "safety": value["safety"],
    }


def _scientific_sha256(value: Mapping[str, object]) -> str:
    return _json_sha256(
        _scientific_projection(value),
        domain=_SCIENTIFIC_DOMAIN,
    )


def _validate_and_restore_payload(
    raw: Mapping[str, object],
) -> tuple[
    tuple[tuple[str, ModalGeneratorLowering], ...],
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    ModalCompilerPipeline | None,
]:
    _strict_fields(raw, _ROOT_FIELDS, label="state-conditioned candidate")
    if (
        raw["schema"] != GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA
        or raw["format_version"]
        != GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_FORMAT_VERSION
    ):
        raise ValueError("unsupported state-conditioned candidate artifact")
    expected_safety = {
        **_SAFETY,
        "contains_promoted_compiler_pipeline": (
            raw["compiler_pipeline"] is not None
        ),
    }
    if raw["safety"] != expected_safety:
        raise ValueError("state-conditioned candidate safety flags are invalid")
    metadata: dict[str, dict[str, object]] = {}
    for field in ("experiment", "config", "splits", "selection", "resources"):
        canonical = _canonical_metadata(
            raw[field],  # type: ignore[arg-type]
            label=field,
        )
        if raw[field] != canonical:
            raise ValueError(f"serialized {field} metadata is not canonical")
        metadata[field] = canonical
    _require_sha256(
        raw["scientific_payload_sha256"],
        label="scientific_payload_sha256",
    )
    if _scientific_sha256(raw) != raw["scientific_payload_sha256"]:
        raise ValueError("state-conditioned scientific payload hash mismatch")

    edgeless = ModalGeneratorGraphPlan.from_state_dict(
        raw["edgeless_graph"]  # type: ignore[arg-type]
    )
    dynamic = ModalGeneratorGraphPlan.from_state_dict(
        raw["dynamic_graph"]  # type: ignore[arg-type]
    )
    _validate_graph_pair(edgeless, dynamic)
    lowerings = _restore_lowering_records(raw["lowering_records"], dynamic)
    pipeline_state = raw["compiler_pipeline"]
    pipeline = (
        None
        if pipeline_state is None
        else ModalCompilerPipeline.from_state_dict(
            pipeline_state  # type: ignore[arg-type]
        )
    )
    _validate_optional_pipeline(
        pipeline,
        graph=dynamic,
        lowerings=lowerings,
    )
    expected_lineage = _lineage(
        lowerings=lowerings,
        edgeless_graph=edgeless,
        dynamic_graph=dynamic,
        compiler_pipeline=pipeline,
    )
    if raw["lineage"] != expected_lineage:
        raise ValueError("state-conditioned candidate lineage is inconsistent")
    _validate_declared_metadata_bindings(
        splits=metadata["splits"],
        selection=metadata["selection"],
        resources=metadata["resources"],
        lineage=expected_lineage,
    )
    return lowerings, edgeless, dynamic, pipeline


def build_gemma3_state_conditioned_modal_graph_candidate(
    *,
    experiment: Mapping[str, object],
    config: Mapping[str, object],
    splits: Mapping[str, object],
    selection: Mapping[str, object],
    resources: Mapping[str, object],
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    edgeless_graph: ModalGeneratorGraphPlan,
    dynamic_graph: ModalGeneratorGraphPlan,
    compiler_pipeline: ModalCompilerPipeline | None = None,
) -> dict[str, object]:
    """Build and fully validate one canonical source-safe candidate payload."""

    edgeless = _authenticated_copy(
        edgeless_graph,
        ModalGeneratorGraphPlan,
    )
    dynamic = _authenticated_copy(dynamic_graph, ModalGeneratorGraphPlan)
    _validate_graph_pair(edgeless, dynamic)
    records = _coerce_lowerings(lowerings_by_node, dynamic)
    lowerings = tuple(
        (
            record["node_name"],
            ModalGeneratorLowering.from_state_dict(record["lowering"]),
        )
        for record in records
    )
    pipeline = (
        None
        if compiler_pipeline is None
        else _authenticated_copy(compiler_pipeline, ModalCompilerPipeline)
    )
    _validate_optional_pipeline(
        pipeline,
        graph=dynamic,
        lowerings=lowerings,  # type: ignore[arg-type]
    )
    metadata = {
        field: _canonical_metadata(value, label=field)
        for field, value in (
            ("experiment", experiment),
            ("config", config),
            ("splits", splits),
            ("selection", selection),
            ("resources", resources),
        )
    }
    lineage = _lineage(
        lowerings=lowerings,  # type: ignore[arg-type]
        edgeless_graph=edgeless,
        dynamic_graph=dynamic,
        compiler_pipeline=pipeline,
    )
    _validate_declared_metadata_bindings(
        splits=metadata["splits"],
        selection=metadata["selection"],
        resources=metadata["resources"],
        lineage=lineage,
    )
    safety = {
        **_SAFETY,
        "contains_promoted_compiler_pipeline": pipeline is not None,
    }
    without_digest: dict[str, object] = {
        "schema": GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
        "format_version": (
            GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_FORMAT_VERSION
        ),
        **metadata,
        "lowering_records": records,
        "edgeless_graph": edgeless.state_dict(),
        "dynamic_graph": dynamic.state_dict(),
        "compiler_pipeline": (
            None if pipeline is None else pipeline.state_dict()
        ),
        "lineage": lineage,
        "safety": safety,
    }
    payload = {
        **without_digest,
        "scientific_payload_sha256": _scientific_sha256(without_digest),
    }
    _validate_and_restore_payload(payload)
    return payload


def _validate_tensor_file(value: object) -> str:
    if (
        not isinstance(value, str)
        or not _MACHINE_STRING.fullmatch(value)
        or Path(value).name != value
        or Path(value).suffix != ".pt"
    ):
        raise ValueError("tensor_file must be a source-safe .pt basename")
    return value


def _json_native(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_native(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_native(child) for child in value]
    return value


def build_gemma3_state_conditioned_modal_graph_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    """Build a compact tensor-free JSON report from a strict payload."""

    _validate_and_restore_payload(payload)
    filename = _validate_tensor_file(tensor_file)
    without_digest = {
        "schema": payload["schema"],
        "format_version": payload["format_version"],
        "experiment": payload["experiment"],
        "config": payload["config"],
        "splits": payload["splits"],
        "selection": payload["selection"],
        "resources": payload["resources"],
        "lineage": payload["lineage"],
        "safety": payload["safety"],
        "artifact": {
            "tensor_file": filename,
            "scientific_payload_sha256": payload[
                "scientific_payload_sha256"
            ],
        },
    }
    report = {
        **without_digest,
        "report_sha256": _json_sha256(without_digest, domain=_REPORT_DOMAIN),
    }
    native = _json_native(report)
    if not isinstance(native, dict):
        raise AssertionError("JSON report root is not a dict")
    return native


def save_gemma3_state_conditioned_modal_graph_candidate(
    output: Path | str,
    **build_arguments: object,
) -> dict[str, object]:
    """Build, save as ``.pt`` plus compact ``.json``, and refuse overwrite."""

    path = Path(output)
    if path.suffix != ".pt":
        raise ValueError("state-conditioned candidate output must use .pt")
    report_path = path.with_suffix(".json")
    if path.exists() or report_path.exists():
        raise FileExistsError(
            "refusing to overwrite state-conditioned candidate artifact"
        )
    payload = build_gemma3_state_conditioned_modal_graph_candidate(
        **build_arguments,  # type: ignore[arg-type]
    )
    report = build_gemma3_state_conditioned_modal_graph_report(
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


def load_gemma3_state_conditioned_modal_graph_candidate(
    input_path: Path | str,
) -> dict[str, object]:
    """Strict-load and cross-check every nested candidate artifact."""

    path = Path(input_path)
    if path.suffix != ".pt":
        raise ValueError("state-conditioned candidate input must use .pt")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("state-conditioned candidate artifact must be a dict")
    _validate_and_restore_payload(raw)
    return raw
