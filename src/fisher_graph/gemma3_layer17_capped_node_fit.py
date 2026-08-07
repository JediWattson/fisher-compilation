"""Fit an edgeless capped-rank modal graph on Gemma layer 17.

The runner is intentionally development-only.  It accepts exactly two strict
Calibration-A fit-only exports: one fit partition and one prompt-disjoint open
selection partition.  There is no guard, Calibration-B, validation, or test
input in the API or CLI.

Unlike the original same-layer experiment, a requested mode rank is resolved
per frozen fragment as ``min(cap, fragment.mode_count)``.  This lets cap 64
realize ranks ``(54, 38, 64, 53)`` without changing the selected parameter
clusters.  The output is an executable edgeless graph plus a compact,
tensor-free JSON report; it does not reuse the state-conditioned candidate
schema, whose nonempty dynamic-graph invariant remains intact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_NATIVE_MODE_COUNTS,
    LAYER17_TOPOLOGY_SHA256,
    Layer17NodeRankResourceRow,
    build_layer17_node_rank_resource_row,
    resolve_layer17_node_ranks,
)
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_FIT_EXPORT,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    FittedModalGeneratorPilot,
    _safe_tokenized_stream_metadata,
    evaluate_modal_generator_graph_conditions,
    fit_layer_cluster_modal_generator,
    load_gemma3_modal_generator_dev_artifact,
    load_development_prompt_export,
    validate_development_split_pair,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    DEFAULT_BASE_ARTIFACT,
    _bind_batch_example_ids,
    _restore_upstream_analysis,
    _validate_upstream_bindings,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    build_edgeless_same_layer_graph,
    select_top_fisher_same_layer_fragments,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _collect_same_layer_native_rows,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_compiler_pipeline import (
    ModalCompilerPipeline,
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering


__all__ = [
    "DEFAULT_GENERATOR_RANK",
    "DEFAULT_MODE_RANK_CAP",
    "DEFAULT_OUTPUT",
    "GEMMA3_LAYER17_CAPPED_NODE_FORMAT_VERSION",
    "GEMMA3_LAYER17_CAPPED_NODE_SCHEMA",
    "build_gemma3_layer17_capped_node_candidate",
    "build_gemma3_layer17_capped_node_report",
    "build_parser",
    "default_gemma3_layer17_capped_node_output",
    "fit_gemma3_layer17_capped_node_candidate",
    "fit_layer17_capped_node_pilots",
    "load_gemma3_layer17_capped_node_candidate",
    "main",
    "restore_gemma3_layer17_capped_node_runtime",
    "save_gemma3_layer17_capped_node_candidate",
]


GEMMA3_LAYER17_CAPPED_NODE_SCHEMA = (
    "fisher_graph.gemma3_layer17_capped_node_edgeless_candidate"
)
GEMMA3_LAYER17_CAPPED_NODE_FORMAT_VERSION = 1
_REPORT_SCHEMA = "fisher_graph.gemma3_layer17_capped_node_edgeless_report"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_MODE_RANK_CAP = 32
DEFAULT_GENERATOR_RANK = 32

_SCIENTIFIC_DOMAIN = b"fisher-graph:gemma3-layer17-capped-node:scientific:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-layer17-capped-node:report:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_MACHINE_STRING = re.compile(r"^[^\s\x00-\x1f\x7f]{1,512}$")

_ROOT_FIELDS = {
    "schema",
    "format_version",
    "experiment",
    "config",
    "splits",
    "fragment_selection",
    "resources",
    "evaluation",
    "lowering_records",
    "edgeless_graph",
    "compiler_pipeline",
    "lineage",
    "safety",
    "scientific_payload_sha256",
}
_LOWERING_RECORD_FIELDS = {
    "node_name",
    "lowering_sha256",
    "graph_node_sha256",
    "selected_fragment_sha256",
    "lowering",
}
_FORBIDDEN_METADATA_KEYS = {
    "prompt",
    "prompts",
    "prompt_text",
    "prompt_texts",
    "input_ids",
    "token_ids",
    "tokens",
    "raw_rows",
    "activation_rows",
    "gradient_rows",
    "teacher_logits",
    "source_model_weights",
    "source_parameter_values",
    "model_state_dict",
    "tokenizer_state",
}
_SPLIT_FIELDS = {
    "policy",
    "fit_export",
    "selection_export",
    "fit_tokenized",
    "selection_tokenized",
}
_TOKENIZED_FIELDS = {
    "schema",
    "format_version",
    "split",
    "batches",
    "sequences",
    "serialized_sha256",
    "source_prompt_sha256",
    "content_sha256",
    "valid_tokens",
    "supervised_positions",
    "contains_prompt_text",
    "contains_token_ids",
}
_EVALUATION_FIELDS = {
    "execution_path",
    "supervised_tokens",
    "logical_valid_tokens",
    "native",
    "conditions",
    "graph",
    "resource_accounting",
}
_CONDITION_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}
_RESOURCE_FIELDS = {
    "replacement_scope",
    "replaced_layer_count",
    "graph_node_count",
    "fragment_count",
    "removed_mode_count",
    "source_whole_model_learned_parameters",
    "candidate_whole_model_learned_parameters",
    "native_removed_learned_parameters",
    "modal_graph_learned_parameters",
    "net_stored_parameter_savings",
    "graph_runtime_storage",
    "planned_peak_live_modal_width",
    "generated",
    "deletion",
    "parameter_savings_positive",
    "logical_mac_savings_positive_generated",
    "latency_or_kernel_speed_claim",
}
_CONDITION_RESOURCE_FIELDS = {
    "logical_linear_macs_native_removed",
    "logical_modal_graph_macs",
    "logical_executed_modal_graph_macs",
    "logical_modal_graph_additions",
    "logical_executed_modal_graph_additions",
    "executed_peak_live_modal_width",
    "net_logical_macs_saved",
}


def default_gemma3_layer17_capped_node_output(
    mode_rank_cap: int,
    generator_rank: int,
) -> Path:
    ranks = resolve_layer17_node_ranks(mode_rank_cap)
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError("generator_rank must be a positive integer")
    if generator_rank > min(ranks):
        raise ValueError("generator_rank exceeds a resolved node rank")
    return _LOCAL_ROOT / (
        f"layer17-capped-node-c{mode_rank_cap}-r{generator_rank}-"
        "edgeless-dev-v1.pt"
    )


DEFAULT_OUTPUT = default_gemma3_layer17_capped_node_output(
    DEFAULT_MODE_RANK_CAP,
    DEFAULT_GENERATOR_RANK,
)


def _progress(message: str) -> None:
    print(f"[layer17-capped-node] {message}", file=sys.stderr, flush=True)


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


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _canonical_json_value(value: object, *, path: str) -> object:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} may not contain tensors")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string key")
        result: dict[str, object] = {}
        for key in sorted(value):  # type: ignore[type-var]
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_METADATA_KEYS or (
                normalized.startswith("raw_")
                and normalized.endswith(
                    ("rows", "tokens", "activations", "gradients")
                )
            ):
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
    raise ValueError(f"{path} contains a non-source-safe metadata value")


def _canonical_metadata(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = _canonical_json_value(value, path=label)
    if not isinstance(result, dict):
        raise AssertionError("metadata root did not canonicalize to dict")
    return result


def _json_native(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def _positive_exact_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _validate_split_lineage(
    raw: object,
    *,
    fit_split_sha256: str,
    selection_split_sha256: str,
) -> None:
    if not isinstance(raw, Mapping) or set(raw) != _SPLIT_FIELDS:
        raise ValueError("capped-node split fields are invalid")
    for label in ("policy", "fit_export", "selection_export"):
        value = raw[label]
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"capped-node {label} metadata is empty")
    policy = raw["policy"]
    if (
        policy.get("fit_export_sha256")
        != raw["fit_export"].get("artifact_sha256")
        or policy.get("eval_export_sha256")
        != raw["selection_export"].get("artifact_sha256")
        or policy.get("prompt_disjoint") is not True
        or policy.get("source_prompt_index_disjoint") is not True
        or policy.get("heldout_guard_used") is not False
        or policy.get("calibration_b_used") is not False
        or policy.get("validation_used") is not False
        or policy.get("test_used") is not False
    ):
        raise ValueError("capped-node split policy is not open fit-only")
    for label, expected in (
        ("fit_tokenized", fit_split_sha256),
        ("selection_tokenized", selection_split_sha256),
    ):
        tokenized = raw[label]
        if not isinstance(tokenized, Mapping) or set(tokenized) != _TOKENIZED_FIELDS:
            raise ValueError(f"capped-node {label} fields are invalid")
        if (
            tokenized.get("serialized_sha256") != expected
            or tokenized.get("contains_prompt_text") is not False
            or tokenized.get("contains_token_ids") is not False
        ):
            raise ValueError(f"capped-node {label} lineage drifted")


def _validate_evaluation(
    raw: object,
    *,
    graph: ModalGeneratorGraphPlan,
    resource_row: Layer17NodeRankResourceRow,
) -> None:
    if not isinstance(raw, Mapping) or set(raw) != _EVALUATION_FIELDS:
        raise ValueError("capped-node evaluation fields are invalid")
    if raw["execution_path"] != "incremental_modal_generator_graph_traversal":
        raise ValueError("capped-node evaluation path drifted")
    _positive_exact_int(raw["supervised_tokens"], label="supervised_tokens")
    valid_tokens = _positive_exact_int(
        raw["logical_valid_tokens"],
        label="logical_valid_tokens",
    )
    native = raw["native"]
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError("capped-node native evaluation is invalid")
    _finite_number(native["nll_per_token"], label="native nll", minimum=0.0)
    conditions = raw["conditions"]
    if not isinstance(conditions, Mapping) or set(conditions) != {
        "generated",
        "deletion",
    }:
        raise ValueError("capped-node evaluation conditions are invalid")
    for label in ("generated", "deletion"):
        condition = conditions[label]
        if not isinstance(condition, Mapping) or set(condition) != _CONDITION_FIELDS:
            raise ValueError(f"capped-node {label} metrics are invalid")
        _finite_number(
            condition["nll_per_token"],
            label=f"{label} nll",
            minimum=0.0,
        )
        _finite_number(
            condition["delta_nll_per_token"],
            label=f"{label} delta nll",
        )
        _finite_number(
            condition["native_to_candidate_kl_per_token"],
            label=f"{label} kl",
            minimum=0.0,
        )
        _finite_number(
            condition["top1_agreement_to_native"],
            label=f"{label} agreement",
            minimum=0.0,
            maximum=1.0,
        )
    graph_record = raw["graph"]
    expected_graph = {
        "node_count": len(graph.nodes),
        "interaction_count": 0,
        "traversal_order": graph.traversal_order,
    }
    if graph_record != expected_graph:
        raise ValueError("capped-node evaluation graph lineage drifted")
    resources = raw["resource_accounting"]
    if not isinstance(resources, Mapping) or set(resources) != _RESOURCE_FIELDS:
        raise ValueError("capped-node evaluation resources are invalid")
    source_whole = _positive_exact_int(
        resources["source_whole_model_learned_parameters"],
        label="source whole-model parameters",
    )
    candidate_whole = _positive_exact_int(
        resources["candidate_whole_model_learned_parameters"],
        label="candidate whole-model parameters",
    )
    base_expected = {
        "replacement_scope": "partial_native_mlp_mode_replacement",
        "replaced_layer_count": 1,
        "graph_node_count": len(graph.nodes),
        "fragment_count": len(graph.nodes),
        "removed_mode_count": sum(LAYER17_NATIVE_MODE_COUNTS),
        "native_removed_learned_parameters": resource_row.source_parameter_count,
        "modal_graph_learned_parameters": resource_row.graph_parameter_count,
        "net_stored_parameter_savings": resource_row.net_parameter_savings,
        "graph_runtime_storage": "registered_copied_device_local_graph_parameters",
        "planned_peak_live_modal_width": max(resource_row.node_ranks),
        "parameter_savings_positive": resource_row.net_parameter_savings > 0,
        "logical_mac_savings_positive_generated": (
            resource_row.net_dense_macs_saved_per_token > 0
        ),
        "latency_or_kernel_speed_claim": False,
    }
    if any(resources.get(key) != value for key, value in base_expected.items()):
        raise ValueError("capped-node evaluation base resources drifted")
    if candidate_whole != (
        source_whole
        - resource_row.source_parameter_count
        + resource_row.graph_parameter_count
    ):
        raise ValueError("capped-node whole-model parameter accounting drifted")
    graph_macs = resource_row.graph_dense_macs_per_token * valid_tokens
    removed_macs = resource_row.source_macs_per_token * valid_tokens
    graph_additions = (
        graph.accounting.elementwise_additions_per_token * valid_tokens
    )
    for label in ("generated", "deletion"):
        actual = resources[label]
        if not isinstance(actual, Mapping) or set(actual) != _CONDITION_RESOURCE_FIELDS:
            raise ValueError(f"capped-node {label} resources are invalid")
        expected = {
            "logical_linear_macs_native_removed": removed_macs,
            "logical_modal_graph_macs": graph_macs,
            "logical_executed_modal_graph_macs": (
                graph_macs if label == "generated" else 0
            ),
            "logical_modal_graph_additions": graph_additions,
            "logical_executed_modal_graph_additions": (
                graph_additions if label == "generated" else 0
            ),
            "executed_peak_live_modal_width": (
                max(resource_row.node_ranks) if label == "generated" else 0
            ),
            "net_logical_macs_saved": (
                removed_macs - graph_macs
                if label == "generated"
                else removed_macs
            ),
        }
        if actual != expected:
            raise ValueError(f"capped-node {label} resource totals drifted")


def _validate_frozen_selection(
    selection: SameLayerFragmentSelection,
) -> None:
    if not isinstance(selection, SameLayerFragmentSelection):
        raise TypeError("selection must be SameLayerFragmentSelection")
    selection.validate_integrity()
    ids = tuple(fragment.fragment_id for fragment in selection.execution_order)
    counts = tuple(fragment.mode_count for fragment in selection.execution_order)
    if (
        selection.layer_ordinal != 17
        or ids != LAYER17_FRAGMENT_IDS
        or counts != LAYER17_NATIVE_MODE_COUNTS
        or selection.fragment_ids != LAYER17_FRAGMENT_IDS
    ):
        raise ValueError("layer-17 capped fit fragment topology drifted")


def _coerce_lowering_records(
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    graph: ModalGeneratorGraphPlan,
) -> tuple[dict[str, object], ...]:
    if not isinstance(lowerings_by_node, Mapping) or set(lowerings_by_node) != {
        node.name for node in graph.nodes
    }:
        raise ValueError("lowerings must exactly cover the edgeless graph")
    records = []
    for node in graph.nodes:
        lowering = lowerings_by_node[node.name]
        if not isinstance(lowering, ModalGeneratorLowering):
            raise TypeError("candidate lowering has an invalid type")
        lowering = ModalGeneratorLowering.from_state_dict(
            lowering.state_dict()
        )
        reconstructed = lowering.to_graph_node(
            name=node.name,
            causal_order=node.causal_order,
            input_boundary=node.input_boundary,
            output_boundary=node.output_boundary,
        )
        if reconstructed.artifact_sha256 != node.artifact_sha256:
            raise ValueError("lowering does not reconstruct its graph node")
        records.append(
            {
                "node_name": node.name,
                "lowering_sha256": lowering.artifact_sha256,
                "graph_node_sha256": node.artifact_sha256,
                "selected_fragment_sha256": lowering.selected_fragment_sha256,
                "lowering": lowering.state_dict(),
            }
        )
    return tuple(records)


def _restore_lowering_records(
    raw_records: object,
    graph: ModalGeneratorGraphPlan,
) -> dict[str, ModalGeneratorLowering]:
    if type(raw_records) is not tuple or len(raw_records) != len(graph.nodes):
        raise ValueError("serialized lowering records do not cover graph nodes")
    result: dict[str, ModalGeneratorLowering] = {}
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
            causal_order=node.causal_order,
            input_boundary=node.input_boundary,
            output_boundary=node.output_boundary,
        )
        if (
            raw["node_name"] != node.name  # type: ignore[index]
            or raw["lowering_sha256"] != lowering.artifact_sha256  # type: ignore[index]
            or raw["graph_node_sha256"] != node.artifact_sha256  # type: ignore[index]
            or raw["selected_fragment_sha256"]  # type: ignore[index]
            != lowering.selected_fragment_sha256
            or reconstructed.artifact_sha256 != node.artifact_sha256
        ):
            raise ValueError("serialized lowering record lineage drifted")
        result[node.name] = lowering
    return result


def _selection_from_lowerings(
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
) -> SameLayerFragmentSelection:
    values = tuple(lowerings_by_node.values())
    if not values:
        raise ValueError("candidate has no lowerings")
    first_plan = values[0].fragment_plan
    if any(
        lowering.fragment_plan.artifact_sha256 != first_plan.artifact_sha256
        for lowering in values
    ):
        raise ValueError("candidate lowerings bind different fragment plans")
    selection = select_top_fisher_same_layer_fragments(
        first_plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    _validate_frozen_selection(selection)
    return selection


def _validate_pipeline(
    pipeline: ModalCompilerPipeline | None,
    *,
    graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    resource_row: Layer17NodeRankResourceRow,
) -> None:
    if pipeline is None:
        return
    pipeline = ModalCompilerPipeline.from_state_dict(pipeline.state_dict())
    if (
        pipeline.graph_plan.artifact_sha256 != graph.artifact_sha256
        or pipeline.interaction_selection is not None
        or pipeline.replaced_fragment_ids != LAYER17_FRAGMENT_IDS
        or pipeline.source_parameter_count != resource_row.source_parameter_count
        or pipeline.source_macs_per_token != resource_row.source_macs_per_token
        or pipeline.graph_parameter_count != resource_row.graph_parameter_count
        or pipeline.net_parameter_savings != resource_row.net_parameter_savings
        or tuple(node.node_name for node in pipeline.nodes)
        != graph.traversal_order
    ):
        raise ValueError("capped-node compiler pipeline lineage drifted")
    expected = {
        name: lowering.artifact_sha256
        for name, lowering in lowerings_by_node.items()
    }
    if any(
        expected.get(node.node_name) != node.lowering.artifact_sha256
        for node in pipeline.nodes
    ):
        raise ValueError("pipeline lowerings differ from candidate graph")


def _lineage(
    *,
    graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    pipeline: ModalCompilerPipeline | None,
    selection: SameLayerFragmentSelection,
    resource_row: Layer17NodeRankResourceRow,
) -> dict[str, object]:
    fit_hashes = {
        lowering.coordinate_generator_plan.binding.fit_split_sha256
        for lowering in lowerings_by_node.values()
    }
    eval_hashes = {
        lowering.coordinate_generator_plan.binding.eval_split_sha256
        for lowering in lowerings_by_node.values()
    }
    if len(fit_hashes) != 1 or len(eval_hashes) != 1:
        raise ValueError("candidate lowerings do not share split lineage")
    fit_split = next(iter(fit_hashes))
    eval_split = next(iter(eval_hashes))
    if fit_split == eval_split:
        raise ValueError("candidate fit and selection splits must differ")
    return {
        "topology_sha256": LAYER17_TOPOLOGY_SHA256,
        "model_fingerprint": graph.model_fingerprint,
        "fragment_plan_sha256": graph.parameter_cluster_plan_sha256,
        "fragment_selection_sha256": selection.artifact_sha256,
        "fragment_ids": LAYER17_FRAGMENT_IDS,
        "native_mode_counts": LAYER17_NATIVE_MODE_COUNTS,
        "fit_split_sha256": fit_split,
        "selection_split_sha256": eval_split,
        "lowering_sha256s": tuple(
            lowerings_by_node[name].artifact_sha256
            for name in graph.traversal_order
        ),
        "graph_sha256": graph.artifact_sha256,
        "compiler_pipeline_sha256": (
            None if pipeline is None else pipeline.artifact_sha256
        ),
        "resource_row_sha256": resource_row.artifact_sha256,
    }


def _safety(pipeline_present: bool) -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_prompt_rows": False,
        "contains_raw_token_rows": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_source_model_weights": False,
        "contains_source_parameter_values": False,
        "contains_executable_generator_weights": True,
        "contains_compiler_pipeline": pipeline_present,
        "source_safe": True,
        "heldout_confirmation": False,
        "calibration_b_used": False,
        "guard_used": False,
        "validation_used": False,
        "test_used": False,
    }


def _scientific_projection(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value[key]
        for key in (
            "schema",
            "format_version",
            "experiment",
            "config",
            "splits",
            "fragment_selection",
            "resources",
            "evaluation",
            "lineage",
            "safety",
        )
    }


def build_gemma3_layer17_capped_node_candidate(
    *,
    experiment: Mapping[str, object],
    splits: Mapping[str, object],
    evaluation: Mapping[str, object],
    mode_rank_cap: int,
    generator_rank: int,
    selection: SameLayerFragmentSelection,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    edgeless_graph: ModalGeneratorGraphPlan,
    compiler_pipeline: ModalCompilerPipeline | None = None,
) -> dict[str, object]:
    """Build one strict executable edgeless candidate payload."""

    _validate_frozen_selection(selection)
    edgeless_graph.validate_integrity()
    if edgeless_graph.interactions:
        raise ValueError("capped-node output must remain edgeless")
    resolved_ranks = resolve_layer17_node_ranks(mode_rank_cap)
    resource_row = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=mode_rank_cap,
        generator_rank=generator_rank,
        edge_policy="edgeless",
    )
    records = _coerce_lowering_records(lowerings_by_node, edgeless_graph)
    actual_ranks = tuple(
        lowerings_by_node[node.name].computational_mode_basis.rank
        for node in edgeless_graph.nodes
    )
    actual_generator_ranks = tuple(
        lowerings_by_node[node.name].coordinate_generator_plan.rank
        for node in edgeless_graph.nodes
    )
    if (
        actual_ranks != resolved_ranks
        or actual_generator_ranks != (generator_rank,) * len(resolved_ranks)
        or edgeless_graph.parameter_count != resource_row.graph_parameter_count
        or edgeless_graph.macs_per_token != resource_row.graph_dense_macs_per_token
    ):
        raise ValueError("candidate graph does not realize the declared ranks")
    _validate_pipeline(
        compiler_pipeline,
        graph=edgeless_graph,
        lowerings_by_node=lowerings_by_node,
        resource_row=resource_row,
    )
    lineage = _lineage(
        graph=edgeless_graph,
        lowerings_by_node=lowerings_by_node,
        pipeline=compiler_pipeline,
        selection=selection,
        resource_row=resource_row,
    )
    config = {
        "mode_rank_cap": mode_rank_cap,
        "resolved_node_ranks": resolved_ranks,
        "generator_rank": generator_rank,
        "edge_policy": "edgeless",
        "fragment_ids": LAYER17_FRAGMENT_IDS,
        "native_mode_counts": LAYER17_NATIVE_MODE_COUNTS,
        "rank_resolution_rule": "min_cap_and_native_fragment_mode_count",
    }
    payload: dict[str, object] = {
        "schema": GEMMA3_LAYER17_CAPPED_NODE_SCHEMA,
        "format_version": GEMMA3_LAYER17_CAPPED_NODE_FORMAT_VERSION,
        "experiment": _canonical_metadata(experiment, label="experiment"),
        "config": config,
        "splits": _canonical_metadata(splits, label="splits"),
        "fragment_selection": selection.metadata(),
        "resources": resource_row.state_dict(),
        "evaluation": _canonical_metadata(evaluation, label="evaluation"),
        "lowering_records": records,
        "edgeless_graph": edgeless_graph.state_dict(),
        "compiler_pipeline": (
            None if compiler_pipeline is None else compiler_pipeline.state_dict()
        ),
        "lineage": lineage,
        "safety": _safety(compiler_pipeline is not None),
    }
    payload["scientific_payload_sha256"] = _json_sha256(
        _scientific_projection(payload),
        domain=_SCIENTIFIC_DOMAIN,
    )
    _validate_and_restore_payload(payload)
    return payload


def _validate_and_restore_payload(
    raw: Mapping[str, object],
) -> tuple[
    SameLayerFragmentSelection,
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
    ModalCompilerPipeline | None,
    Layer17NodeRankResourceRow,
]:
    _strict_fields(raw, _ROOT_FIELDS, label="layer-17 capped-node candidate")
    if (
        raw["schema"] != GEMMA3_LAYER17_CAPPED_NODE_SCHEMA
        or raw["format_version"] != GEMMA3_LAYER17_CAPPED_NODE_FORMAT_VERSION
    ):
        raise ValueError("unsupported layer-17 capped-node candidate")
    for field in ("experiment", "splits", "evaluation"):
        canonical = _canonical_metadata(
            raw[field],  # type: ignore[arg-type]
            label=field,
        )
        if raw[field] != canonical:
            raise ValueError(f"serialized {field} metadata is not canonical")
    graph = ModalGeneratorGraphPlan.from_state_dict(
        raw["edgeless_graph"]  # type: ignore[arg-type]
    )
    if graph.interactions:
        raise ValueError("serialized capped-node graph is not edgeless")
    lowerings = _restore_lowering_records(raw["lowering_records"], graph)
    selection = _selection_from_lowerings(lowerings)
    if raw["fragment_selection"] != selection.metadata():
        raise ValueError("serialized frozen fragment selection drifted")
    config = raw["config"]
    if not isinstance(config, Mapping) or set(config) != {
        "mode_rank_cap",
        "resolved_node_ranks",
        "generator_rank",
        "edge_policy",
        "fragment_ids",
        "native_mode_counts",
        "rank_resolution_rule",
    }:
        raise ValueError("capped-node config fields are invalid")
    cap = config["mode_rank_cap"]
    generator_rank = config["generator_rank"]
    if type(cap) is not int or type(generator_rank) is not int:
        raise ValueError("capped-node ranks must be integers")
    resolved = resolve_layer17_node_ranks(cap)
    if (
        config["resolved_node_ranks"] != resolved
        or config["edge_policy"] != "edgeless"
        or config["fragment_ids"] != LAYER17_FRAGMENT_IDS
        or config["native_mode_counts"] != LAYER17_NATIVE_MODE_COUNTS
        or config["rank_resolution_rule"]
        != "min_cap_and_native_fragment_mode_count"
    ):
        raise ValueError("capped-node config topology drifted")
    resource_state = raw["resources"]
    if not isinstance(resource_state, Mapping):
        raise TypeError("capped-node resource row must be a mapping")
    resource_row = Layer17NodeRankResourceRow.from_state_dict(resource_state)
    if (
        resource_row.spec.mode_rank_cap != cap
        or resource_row.spec.generator_rank != generator_rank
        or resource_row.spec.edge_policy != "edgeless"
        or resource_row.spec.label != "candidate"
        or graph.parameter_count != resource_row.graph_parameter_count
        or graph.macs_per_token != resource_row.graph_dense_macs_per_token
    ):
        raise ValueError("capped-node resource row contradicts graph")
    actual_ranks = tuple(
        lowerings[node.name].computational_mode_basis.rank for node in graph.nodes
    )
    actual_private = tuple(
        lowerings[node.name].coordinate_generator_plan.rank for node in graph.nodes
    )
    if actual_ranks != resolved or actual_private != (generator_rank,) * 4:
        raise ValueError("serialized lowerings do not realize declared ranks")
    lowering_by_fragment_sha256: dict[str, ModalGeneratorLowering] = {}
    for lowering in lowerings.values():
        selected_sha256 = lowering.selected_fragment_sha256
        if selected_sha256 in lowering_by_fragment_sha256:
            raise ValueError(
                "serialized lowerings bind one fragment more than once"
            )
        lowering_by_fragment_sha256[selected_sha256] = lowering
    selected_sha256s = {
        fragment.artifact_sha256 for fragment in selection.execution_order
    }
    if set(lowering_by_fragment_sha256) != selected_sha256s:
        raise ValueError(
            "serialized lowerings do not bind every selected fragment once"
        )
    rebuilt = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=next(iter(lowerings.values())).fragment_plan,
        lowerings_by_fragment={
            fragment.fragment_id: lowering_by_fragment_sha256[
                fragment.artifact_sha256
            ]
            for fragment in selection.execution_order
        },
    )
    if rebuilt.graph_plan.artifact_sha256 != graph.artifact_sha256:
        raise ValueError("capped-node graph cannot be rebuilt from lowerings")
    pipeline_state = raw["compiler_pipeline"]
    pipeline = (
        None
        if pipeline_state is None
        else ModalCompilerPipeline.from_state_dict(
            pipeline_state  # type: ignore[arg-type]
        )
    )
    _validate_pipeline(
        pipeline,
        graph=graph,
        lowerings_by_node=lowerings,
        resource_row=resource_row,
    )
    expected_lineage = _lineage(
        graph=graph,
        lowerings_by_node=lowerings,
        pipeline=pipeline,
        selection=selection,
        resource_row=resource_row,
    )
    _validate_split_lineage(
        raw["splits"],
        fit_split_sha256=expected_lineage["fit_split_sha256"],  # type: ignore[arg-type]
        selection_split_sha256=expected_lineage[
            "selection_split_sha256"
        ],  # type: ignore[arg-type]
    )
    _validate_evaluation(
        raw["evaluation"],
        graph=graph,
        resource_row=resource_row,
    )
    if raw["lineage"] != expected_lineage:
        raise ValueError("capped-node candidate lineage drifted")
    if raw["safety"] != _safety(pipeline is not None):
        raise ValueError("capped-node candidate safety flags drifted")
    _require_sha256(
        raw["scientific_payload_sha256"],
        label="scientific_payload_sha256",
    )
    if _json_sha256(
        _scientific_projection(raw),
        domain=_SCIENTIFIC_DOMAIN,
    ) != raw["scientific_payload_sha256"]:
        raise ValueError("capped-node scientific payload hash mismatch")
    return selection, graph, lowerings, pipeline, resource_row


def build_gemma3_layer17_capped_node_report(
    payload: Mapping[str, object],
    *,
    tensor_file: str,
) -> dict[str, object]:
    """Build a compact tensor-free report from a strict candidate."""

    _, graph, _, pipeline, resource_row = _validate_and_restore_payload(payload)
    if (
        not isinstance(tensor_file, str)
        or Path(tensor_file).name != tensor_file
        or Path(tensor_file).suffix != ".pt"
    ):
        raise ValueError("tensor_file must be a source-safe .pt basename")
    without_digest = {
        "schema": _REPORT_SCHEMA,
        "format_version": GEMMA3_LAYER17_CAPPED_NODE_FORMAT_VERSION,
        "experiment": payload["experiment"],
        "config": payload["config"],
        "splits": payload["splits"],
        "fragment_selection": payload["fragment_selection"],
        "resources": resource_row.state_dict(),
        "evaluation": payload["evaluation"],
        "lineage": payload["lineage"],
        "graph": {
            "artifact_sha256": graph.artifact_sha256,
            "node_count": len(graph.nodes),
            "interaction_count": len(graph.interactions),
            "traversal_order": graph.traversal_order,
            "parameter_count": graph.parameter_count,
            "macs_per_token": graph.macs_per_token,
        },
        "compiler_pipeline_sha256": (
            None if pipeline is None else pipeline.artifact_sha256
        ),
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_tensors": False,
            "contains_generator_weights": False,
            "source_safe": True,
        },
        "artifact": {
            "tensor_file": tensor_file,
            "scientific_payload_sha256": payload["scientific_payload_sha256"],
        },
    }
    report = {
        **without_digest,
        "report_sha256": _json_sha256(without_digest, domain=_REPORT_DOMAIN),
    }
    native = _json_native(report)
    if not isinstance(native, dict):
        raise AssertionError("capped-node report root is not a dict")
    return native


def save_gemma3_layer17_capped_node_candidate(
    output: Path | str,
    **build_arguments: object,
) -> dict[str, object]:
    """Save an exclusive ``.pt`` candidate and tensor-free ``.json`` report."""

    path = Path(output)
    if path.suffix != ".pt":
        raise ValueError("capped-node candidate output must use .pt")
    report_path = path.with_suffix(".json")
    if path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite capped-node candidate")
    payload = build_gemma3_layer17_capped_node_candidate(
        **build_arguments,  # type: ignore[arg-type]
    )
    report = build_gemma3_layer17_capped_node_report(
        payload,
        tensor_file=path.name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_created = False
    report_created = False
    try:
        with path.open("xb") as handle:
            tensor_created = True
            torch.save(payload, handle)
        with report_path.open("x", encoding="utf-8") as handle:
            report_created = True
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except BaseException:
        if report_created and report_path.exists():
            report_path.unlink()
        if tensor_created and path.exists():
            path.unlink()
        raise
    return report


def load_gemma3_layer17_capped_node_candidate(
    path: Path | str,
) -> dict[str, object]:
    """Strict-load every nested executable artifact on CPU."""

    source = Path(path)
    if source.suffix != ".pt":
        raise ValueError("capped-node candidate input must use .pt")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("capped-node candidate must contain a dict")
    _validate_and_restore_payload(raw)
    return raw


def restore_gemma3_layer17_capped_node_runtime(
    value: Mapping[str, object] | Path | str,
) -> tuple[
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
    ModalCompilerPipeline | None,
]:
    """Restore the graph, ordered lowering map, and optional pipeline."""

    raw = (
        load_gemma3_layer17_capped_node_candidate(value)
        if isinstance(value, (Path, str))
        else value
    )
    if not isinstance(raw, Mapping):
        raise TypeError("capped-node runtime source must be a mapping or path")
    _, graph, lowerings, pipeline, _ = _validate_and_restore_payload(raw)
    return graph, lowerings, pipeline


def fit_layer17_capped_node_pilots(
    fit_rows: object,
    selection_rows: object,
    *,
    selection: SameLayerFragmentSelection,
    source_model_sha256: str,
    parameter_catalog_sha256: str,
    fisher_coupling_sha256: str,
    fragment_plan: object,
    fit_split_sha256: str,
    selection_split_sha256: str,
    mode_rank_cap: int,
    generator_rank: int,
    ridge: float = 0.0,
) -> dict[str, FittedModalGeneratorPilot]:
    """Fit each frozen fragment at its independently resolved rank."""

    _validate_frozen_selection(selection)
    resolved = resolve_layer17_node_ranks(mode_rank_cap)
    if type(generator_rank) is not int or generator_rank <= 0:
        raise ValueError("generator_rank must be a positive integer")
    if generator_rank > min(resolved):
        raise ValueError("generator_rank exceeds a resolved node rank")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    fit_catalog = getattr(fit_rows, "rows_by_fragment", None)
    selection_catalog = getattr(selection_rows, "rows_by_fragment", None)
    if not isinstance(fit_catalog, Mapping) or not isinstance(
        selection_catalog,
        Mapping,
    ):
        raise TypeError("aligned rows must expose rows_by_fragment mappings")
    expected_ids = set(LAYER17_FRAGMENT_IDS)
    if set(fit_catalog) != expected_ids or set(selection_catalog) != expected_ids:
        raise ValueError("aligned rows do not exactly cover frozen fragments")
    pilots: dict[str, FittedModalGeneratorPilot] = {}
    for fragment, resolved_rank in zip(
        selection.execution_order,
        resolved,
        strict=True,
    ):
        pilots[fragment.fragment_id] = fit_layer_cluster_modal_generator(
            fit_catalog[fragment.fragment_id],
            selection_catalog[fragment.fragment_id],
            selection=fragment,
            source_model_sha256=source_model_sha256,
            parameter_catalog_sha256=parameter_catalog_sha256,
            fisher_coupling_sha256=fisher_coupling_sha256,
            fragment_plan=fragment_plan,  # type: ignore[arg-type]
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=selection_split_sha256,
            input_site=fragment.input_site,
            output_site=fragment.output_site,
            mode_ranks=(resolved_rank,),
            selected_mode_rank=resolved_rank,
            generator_ranks=(generator_rank,),
            selected_generator_rank=generator_rank,
            ridge=ridge,
        )
    return pilots


def fit_gemma3_layer17_capped_node_candidate(
    *,
    revision: str,
    output: Path | str = DEFAULT_OUTPUT,
    fit_export_path: Path | str = DEFAULT_FIT_EXPORT,
    selection_export_path: Path | str = DEFAULT_EVAL_EXPORT,
    base_artifact_path: Path | str = DEFAULT_BASE_ARTIFACT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    mode_rank_cap: int = DEFAULT_MODE_RANK_CAP,
    generator_rank: int = DEFAULT_GENERATOR_RANK,
    ridge: float = 0.0,
) -> dict[str, object]:
    """Fit and assess one edgeless candidate on open development only."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.suffix != ".pt":
        raise ValueError("capped-node candidate output must use .pt")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite capped-node candidate")
    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    resource_row = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=mode_rank_cap,
        generator_rank=generator_rank,
        edge_policy="edgeless",
    )

    _progress("preflight: strict-load two open fit-only exports")
    fit_export = load_development_prompt_export(fit_export_path)
    selection_export = load_development_prompt_export(selection_export_path)
    split_policy = validate_development_split_pair(fit_export, selection_export)
    upstream = load_gemma3_modal_generator_dev_artifact(base_artifact_path)
    upstream_splits = upstream.get("splits")
    if not isinstance(upstream_splits, Mapping) or (
        upstream_splits.get("fit_export") != fit_export.metadata()
        or upstream_splits.get("eval_export") != selection_export.metadata()
    ):
        raise ValueError("open exports do not match the source analysis")
    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    selection = select_top_fisher_same_layer_fragments(
        fragment_plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    _validate_frozen_selection(selection)

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma checkpoint from local cache")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    fingerprint = adapter.model_fingerprint()
    _validate_upstream_bindings(
        upstream,
        model_id=model_id,
        revision=revision,
        model_fingerprint=fingerprint,
    )

    _progress("tokenize: materialize fit and prompt-disjoint open selection")
    fit_batches, fit_stream = _materialize_split(
        tokenizer,
        fit_export.prompts,
        split_name="modal_generator_development_fit",
        max_length=256,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        selection_export.prompts,
        split_name="modal_generator_development_eval",
        max_length=256,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fit_batches = _bind_batch_example_ids(fit_batches, fit_export.prompt_sha256s)
    selection_batches = _bind_batch_example_ids(
        selection_batches,
        selection_export.prompt_sha256s,
    )
    fit_safe = _safe_tokenized_stream_metadata(fit_stream)
    selection_safe = _safe_tokenized_stream_metadata(selection_stream)
    if (
        fit_safe != upstream_splits.get("fit_tokenized")
        or selection_safe != upstream_splits.get("eval_tokenized")
    ):
        raise ValueError("live tokenization differs from source analysis")
    fit_split_sha256 = _require_sha256(
        fit_safe.get("serialized_sha256"),
        label="fit split",
    )
    selection_split_sha256 = _require_sha256(
        selection_safe.get("serialized_sha256"),
        label="selection split",
    )
    if fit_split_sha256 == selection_split_sha256:
        raise ValueError("fit and selection token streams overlap")
    live_specs, leaf_site, _ = _whole_model_layer_specs(adapter)
    if tuple(spec.layer_id for spec in live_specs) != tuple(
        spec.layer_id for spec in fit_trace.layer_specs
    ):
        raise ValueError("live layer catalog differs from source analysis")

    _progress("rows: collect four frozen layer-17 fragment streams once")
    fit_rows = _collect_same_layer_native_rows(
        adapter,
        fit_batches,
        selection=selection,
        leaf_activation_site=leaf_site,
    )
    selection_rows = _collect_same_layer_native_rows(
        adapter,
        selection_batches,
        selection=selection,
        leaf_activation_site=leaf_site,
    )
    _progress(
        f"fit: cap {mode_rank_cap} -> {resource_row.node_ranks}; "
        f"generator rank {generator_rank}"
    )
    pilots = fit_layer17_capped_node_pilots(
        fit_rows,
        selection_rows,
        selection=selection,
        source_model_sha256=fingerprint,
        parameter_catalog_sha256=catalog.artifact_sha256,
        fisher_coupling_sha256=fisher.artifact_sha256,
        fragment_plan=fragment_plan,
        fit_split_sha256=fit_split_sha256,
        selection_split_sha256=selection_split_sha256,
        mode_rank_cap=mode_rank_cap,
        generator_rank=generator_rank,
        ridge=ridge,
    )
    edgeless = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=fragment_plan,
        lowerings_by_fragment={
            fragment_id: pilot.lowering for fragment_id, pilot in pilots.items()
        },
    )
    accounting = build_modal_source_replacement_accounting(
        catalog,
        fragment_plan,
        LAYER17_FRAGMENT_IDS,
    )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=fit_trace,
        parameter_catalog=catalog,
        grouped_fisher=fisher,
        fisher_clusters=clusters,
        parameter_cluster_fragments=fragment_plan,
        lowerings_by_node=edgeless.lowerings_by_node,
        graph_plan=edgeless.graph_plan,
        interaction_selection=None,
        source_replacement_accounting=accounting,
    )
    if (
        pipeline.source_parameter_count != resource_row.source_parameter_count
        or pipeline.graph_parameter_count != resource_row.graph_parameter_count
        or pipeline.graph_plan.macs_per_token
        != resource_row.graph_dense_macs_per_token
    ):
        raise RuntimeError("live compiler resources differ from analytic plan")
    executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless.graph_plan,
        edgeless.lowerings,
    )
    _progress("selection: evaluate edgeless generation and matched deletion")
    evaluation = evaluate_modal_generator_graph_conditions(
        adapter,
        executor,
        selection_batches,
    )
    execution_resources = evaluation.get("resource_accounting")
    if not isinstance(execution_resources, Mapping) or (
        execution_resources.get("native_removed_learned_parameters")
        != resource_row.source_parameter_count
        or execution_resources.get("modal_graph_learned_parameters")
        != resource_row.graph_parameter_count
        or execution_resources.get("net_stored_parameter_savings")
        != resource_row.net_parameter_savings
    ):
        raise RuntimeError("live executor resources differ from analytic plan")
    if adapter.model_fingerprint() != fingerprint:
        raise RuntimeError("capped-node fit mutated the source model")
    return save_gemma3_layer17_capped_node_candidate(
        destination,
        experiment={
            "experiment_kind": "gemma3_layer17_capped_node_edgeless_v1",
            "scientific_role": "open_development_fit_and_selection",
            "model_id": model_id,
            "requested_revision": revision,
            "adapter_model_fingerprint": fingerprint,
            "source_model_unchanged": True,
            "heldout_confirmation": False,
        },
        splits={
            "policy": split_policy,
            "fit_export": fit_export.metadata(),
            "selection_export": selection_export.metadata(),
            "fit_tokenized": fit_safe,
            "selection_tokenized": selection_safe,
        },
        evaluation=evaluation,
        mode_rank_cap=mode_rank_cap,
        generator_rank=generator_rank,
        selection=selection,
        lowerings_by_node=edgeless.lowerings_by_node,
        edgeless_graph=edgeless.graph_plan,
        compiler_pipeline=pipeline,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a frozen-topology layer-17 capped-node edgeless candidate "
            "using open Calibration-A fit-only exports."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    parser.add_argument(
        "--selection-export",
        type=Path,
        default=DEFAULT_EVAL_EXPORT,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_BASE_ARTIFACT,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--mode-rank-cap",
        type=int,
        default=DEFAULT_MODE_RANK_CAP,
    )
    parser.add_argument(
        "--generator-rank",
        type=int,
        default=DEFAULT_GENERATOR_RANK,
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = fit_gemma3_layer17_capped_node_candidate(
        revision=arguments.revision,
        output=arguments.output,
        fit_export_path=arguments.fit_export,
        selection_export_path=arguments.selection_export,
        base_artifact_path=arguments.base_artifact,
        model_id=arguments.model_id,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        tokenization_batch_size=arguments.tokenization_batch_size,
        mode_rank_cap=arguments.mode_rank_cap,
        generator_rank=arguments.generator_rank,
        ridge=arguments.ridge,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
