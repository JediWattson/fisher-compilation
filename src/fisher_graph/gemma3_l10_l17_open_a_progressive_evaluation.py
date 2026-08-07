"""Leakage-safe open-development evaluation of the Gemma layer-10+17 graph.

This rung composes the already frozen, guard-qualified layer-10 dynamic graph
with the all-eight-family layer-17 refit.  It deliberately reuses only the
already-open ``calibration_a_selection`` role.  The executable bundle, the
layer-17 LOFO receipt, and the prior adaptive-selection receipt are strictly
authenticated *before* the one function that can open selection prompts is
called.

The output is an adaptive-development receipt.  It is not heldout evidence,
does not authorize serving, does not claim latency, and does not claim that
the whole model has been compiled.  Full-model logits are scored while only
the declared layer-10 and layer-17 MLP fragments are replaced.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_layer17_open_a_capacity_evaluation import (
    _add_comparison,
    _add_native,
    _candidate_comparison,
    _equal_family_macro,
    _file_sha256,
    _finalize_metric_accumulator,
    _load_open_selection_authority,
    _materialize_selection_families,
    _metric_identity_within_ulps,
    _model_logits,
    _native_nll,
    _new_metric_accumulator,
    _selected_logits_and_targets,
    load_gemma3_layer17_open_a_capacity_result,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
    DEFAULT_SELECTION_OUTPUT,
)
from .gemma3_layer17_v8_fit_lofo import (
    load_gemma3_layer17_v8_fit_lofo_report,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_graph_composition_bundle import (
    load_gemma3_layer10_layer17_composition_bundle,
    restore_gemma3_layer10_layer17_composition_runtime,
)
from .gemma3_state_conditioned_shape_flow_experiment import (
    _tokenizer_contract,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_graph_rung_evaluation import (
    _GRAPH_LOGICAL_FIELDS,
    _GRAPH_STATIC_FIELDS,
    _assert_close_logits,
    _execution_fields,
    _validate_graph_execution,
)


__all__ = [
    "DEFAULT_ADAPTIVE_RESULT_PATH",
    "DEFAULT_COMPOSITION_BUNDLE_PATH",
    "DEFAULT_LOFO_REPORT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "evaluate_gemma3_l10_l17_open_a_progressive",
    "finalize_gemma3_l10_l17_open_a_prevalidation_checkpoint",
    "load_gemma3_l10_l17_open_a_progressive_result",
    "load_gemma3_l10_l17_open_a_prevalidation_checkpoint",
    "progressive_composition_decision",
    "validate_gemma3_l10_l17_open_a_progressive_result",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_COMPOSITION_BUNDLE_PATH = (
    _LOCAL_ROOT / "layer10-layer17-adaptive-composition-open-a-v2.pt"
)
DEFAULT_LOFO_REPORT_PATH = (
    _LOCAL_ROOT / "layer17-v8-fit-lofo-cap48-r16-edgeless-v1.json"
)
DEFAULT_ADAPTIVE_RESULT_PATH = (
    _LOCAL_ROOT / "layer17-open-a-adaptive-refit-c48-r16-dev-v1.json"
)
DEFAULT_OUTPUT_PATH = (
    _LOCAL_ROOT / "layer10-layer17-open-a-progressive-adaptive-v1.json"
)

_SCHEMA = "fisher_graph.gemma3_l10_l17_open_a_progressive_evaluation"
_FORMAT_VERSION = 1
_RESULT_DOMAIN = b"fisher-graph:gemma3-l10-l17-open-a-progressive:v1\0"
_CHECKPOINT_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-open-a-prevalidation-checkpoint:v1\0"
)
_CHECKPOINT_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-open-a-unvalidated-result:v1\0"
)
_POLICY_DOMAIN = b"fisher-graph:gemma3-l10-l17-progressive-policy:v1\0"
_AUTHORITY_DOMAIN = b"fisher-graph:gemma3-l10-l17-open-a-authority:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_EXAMPLES = 128
_EXPECTED_FAMILIES = 4
_VOCABULARY_CHUNK_SIZE = 16384
_CONDITIONS = (
    "layer10_dynamic",
    "layer17_adaptive_edgeless",
    "composed_edgeless",
    "composed_primary",
    "matched_double_deletion",
)

# This is an immutable experiment policy, not a post-outcome choice.  Its
# canonical hash is emitted and replayed by the strict result validator.
_POLICY: dict[str, object] = {
    "policy_id": "gemma3_l10_l17_progressive_open_a_v1",
    "maximum_micro_delta_nll_per_token": 0.08,
    "maximum_equal_family_macro_delta_nll_per_token": 0.08,
    "maximum_equal_family_macro_native_kl_per_token": 0.09,
    "minimum_equal_family_macro_top1_agreement": 0.84,
    "maximum_family_delta_nll_per_token": 0.10,
    "minimum_passing_family_count": 3,
    "required_family_count": 4,
    "maximum_macro_interaction_excess_nll": 0.01,
    "minimum_macro_deletion_recovery_fraction": 0.60,
    "minimum_worst_family_deletion_recovery_fraction": 0.40,
    "maximum_primary_nll_regression_to_composed_edgeless": 0.001,
    "require_exact_cumulative_resources": True,
    "require_positive_incremental_parameter_savings_vs_layer17": True,
    "require_positive_incremental_executed_mac_savings_vs_layer17": True,
    "decision_role": "adaptive_open_development_composition",
    "heldout_confirmation": False,
}

# Exact live candidate/evidence authorities.  Changing any candidate or
# evidence artifact requires a new evaluator policy/version rather than a
# silent replay on the already-open role.
_EXPECTED_AUTHORITIES: dict[str, object] = {
    "composition_bundle_file_sha256": (
        "394906f8e84a50e18922de0dc8c114be1ea9889f0995ccca180b9f6a8d303d8d"
    ),
    "composition_payload_sha256": (
        "2f7c2179656fc16c614cd84b7a0b29d3250443a5d8c80db221b220e3d3f082bf"
    ),
    "combined_edgeless_graph_sha256": (
        "76e6ca06124a542e0f1ce4b26315f5892abdb1057d07d097a849dca312dc3f6c"
    ),
    "combined_primary_graph_sha256": (
        "35d35f2318e0728bb649f2825601d2edbe06e13307a412c4d81c66e8e387c4ca"
    ),
    "layer10_candidate_tensor_file_sha256": (
        "feffc023ba37aee10591cc4313238dd6936181a5e77c5a61d12cfe6be04b8a1b"
    ),
    "layer10_candidate_scientific_payload_sha256": (
        "eae90f334b34dc76d7ef38585e394f77825e1514189cecc3e38e36ec3842fcbb"
    ),
    "layer10_guard_evidence_file_sha256": (
        "9245d5b1ff435bc9b96c618b117f1884de36cb65bad734d927e4096163108d86"
    ),
    "layer10_guard_evidence_logical_sha256": (
        "db0e191e2e92e3eedb1f6a6fd955257a3e9e2c2dd7a94d29a829f1900ec6f6a7"
    ),
    "layer17_candidate_tensor_file_sha256": (
        "fc989138da2c190c848fe64460752711b19144b68b20fedb047f6352e9aeea17"
    ),
    "layer17_candidate_scientific_payload_sha256": (
        "e0969e90e78c714dc27bc1ee80d925e4dddc02a6e3fff2ea610bd46815c7231e"
    ),
    "layer17_edgeless_graph_sha256": (
        "4b81283db0df73b3be06d67ed61be4733190824687b0d34a5d9b3662a26d1607"
    ),
    "layer17_lofo_report_file_sha256": (
        "f06a7ad0a761bf8504f6e9d89602e2fb662b0145fd9cdee875b006ae62527106"
    ),
    "layer17_lofo_report_sha256": (
        "89787c438072062aec7b3b07d75fc38e9a5b24b9e799ca8c0c333b36f22b1ed8"
    ),
    "layer17_adaptive_result_file_sha256": (
        "f770f20aa03894c8f71d4d6f8879415846b799c2483c726ca2ec9cd72d7fb708"
    ),
    "layer17_adaptive_result_sha256": (
        "dfe7bda2242c9f2c1b138b87ef49e4f61fe18df3e19befba7c540d5ef73cef8f"
    ),
}

_EXPECTED_RESOURCES: dict[str, int] = {
    "source_whole_model_learned_parameters": 268_098_176,
    "replaced_layer_count": 2,
    "graph_node_count": 8,
    "dynamic_interaction_count": 3,
    "native_removed_parameters": 1_082_880,
    "primary_graph_parameters": 295_129,
    "net_stored_parameter_savings": 787_751,
    "candidate_whole_model_learned_parameters": 267_310_425,
    "dense_graph_macs_per_token": 289_600,
    "executed_graph_macs_per_token": 286_784,
    "native_removed_macs_per_token": 1_082_880,
    "net_executed_macs_saved_per_token": 796_096,
    "layer17_only_net_parameter_savings": 278_506,
    "layer17_only_net_executed_macs_saved_per_token": 281_248,
    "incremental_parameter_savings_vs_layer17_only": 509_245,
    "incremental_executed_macs_saved_vs_layer17_only": 514_848,
}

_EXPECTED_CONDITION_RESOURCES: dict[str, dict[str, int]] = {
    "layer10_dynamic": {
        "replaced_layer_count": 1,
        "graph_node_count": 4,
        "interaction_count": 3,
        "native_removed_parameters": 641_280,
        "graph_parameters": 132_035,
        "net_parameter_savings": 509_245,
        "dense_graph_macs_per_token": 129_248,
        "executed_graph_macs_per_token": 126_432,
        "net_executed_macs_saved_per_token": 514_848,
    },
    "layer17_adaptive_edgeless": {
        "replaced_layer_count": 1,
        "graph_node_count": 4,
        "interaction_count": 0,
        "native_removed_parameters": 441_600,
        "graph_parameters": 163_094,
        "net_parameter_savings": 278_506,
        "dense_graph_macs_per_token": 160_352,
        "executed_graph_macs_per_token": 160_352,
        "net_executed_macs_saved_per_token": 281_248,
    },
    "composed_edgeless": {
        "replaced_layer_count": 2,
        "graph_node_count": 8,
        "interaction_count": 0,
        "native_removed_parameters": 1_082_880,
        "graph_parameters": 290_710,
        "net_parameter_savings": 792_170,
        "dense_graph_macs_per_token": 285_280,
        "executed_graph_macs_per_token": 285_280,
        "net_executed_macs_saved_per_token": 797_600,
    },
    "composed_primary": {
        "replaced_layer_count": 2,
        "graph_node_count": 8,
        "interaction_count": 3,
        "native_removed_parameters": 1_082_880,
        "graph_parameters": 295_129,
        "net_parameter_savings": 787_751,
        "dense_graph_macs_per_token": 289_600,
        "executed_graph_macs_per_token": 286_784,
        "net_executed_macs_saved_per_token": 796_096,
    },
    "matched_double_deletion": {
        "replaced_layer_count": 2,
        "graph_node_count": 8,
        "interaction_count": 3,
        "native_removed_parameters": 1_082_880,
        "graph_parameters": 295_129,
        "net_parameter_savings": 787_751,
        "dense_graph_macs_per_token": 289_600,
        "executed_graph_macs_per_token": 0,
        "net_executed_macs_saved_per_token": 1_082_880,
    },
}

_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_family_ids": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_model_or_candidate_weights": False,
    "fit_opened": False,
    "guard_opened": False,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "candidate_weights_mutated": False,
    "model_weights_mutated": False,
    "local_files_only": True,
    "source_safe": True,
}
_CHECKPOINT_SAFETY: dict[str, bool] = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_family_ids": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_model_or_candidate_weights": False,
    "source_safe": True,
    "selection_scoring_completed": True,
    "strict_result_validation_completed": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_equal(left: object, right: object) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_atomic(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite {path.name}") from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _BundleAuthority:
    path: Path
    file_sha256: str
    raw: dict[str, object]
    edgeless: ModalGeneratorGraphPlan
    primary: ModalGeneratorGraphPlan
    lowerings: tuple[object, ...]
    binding: dict[str, object]
    model_id: str
    requested_revision: str


def _parent_by_role(
    bundle: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    parents = bundle.get("parents")
    if type(parents) is not tuple or len(parents) != 2:
        raise ValueError("composition parent catalog is invalid")
    result: dict[str, Mapping[str, object]] = {}
    for parent in parents:
        if not isinstance(parent, Mapping) or parent.get("role") not in {
            "layer10",
            "layer17",
        }:
            raise ValueError("composition parent record is invalid")
        result[str(parent["role"])] = parent
    if set(result) != {"layer10", "layer17"}:
        raise ValueError("composition parent roles differ")
    return result


def _subgraph(
    graph: ModalGeneratorGraphPlan,
    *,
    layer_ordinal: int,
    include_interactions: bool,
) -> ModalGeneratorGraphPlan:
    boundary = f"layer.{layer_ordinal}."
    nodes = tuple(
        node
        for node in graph.nodes
        if node.input_boundary.startswith(boundary)
        and node.output_boundary.startswith(boundary)
    )
    names = {node.name for node in nodes}
    if len(nodes) != 4:
        raise ValueError(f"layer {layer_ordinal} graph must contain four nodes")
    interactions = (
        tuple(
            edge
            for edge in graph.interactions
            if edge.source_node in names and edge.target_node in names
        )
        if include_interactions
        else ()
    )
    return ModalGeneratorGraphPlan(
        model_fingerprint=graph.model_fingerprint,
        parameter_cluster_plan_sha256=graph.parameter_cluster_plan_sha256,
        nodes=nodes,
        interactions=interactions,
    )


def _resource_formula(plan: ModalGeneratorGraphPlan) -> int:
    """Exact selected-execution MAC bound for the current conditional DAG."""

    return (
        plan.accounting.node_macs_per_token
        + sum(
            edge.macs_per_token
            for edge in plan.interactions
            if edge.__class__.__name__ == "ModalGeneratorInteraction"
        )
        + plan.conditional_routing_macs_per_token
        + plan.conditional_selected_message_macs_per_token_upper_bound
    )


def _bundle_authority(path: Path | str) -> _BundleAuthority:
    source = Path(path)
    file_sha256 = _file_sha256(source)
    raw = load_gemma3_layer10_layer17_composition_bundle(source)
    edgeless, primary, lowerings = (
        restore_gemma3_layer10_layer17_composition_runtime(raw)
    )
    parents = _parent_by_role(raw)
    if (
        file_sha256
        != _EXPECTED_AUTHORITIES["composition_bundle_file_sha256"]
        or raw.get("composition_payload_sha256")
        != _EXPECTED_AUTHORITIES["composition_payload_sha256"]
        or edgeless.artifact_sha256
        != _EXPECTED_AUTHORITIES["combined_edgeless_graph_sha256"]
        or primary.artifact_sha256
        != _EXPECTED_AUTHORITIES["combined_primary_graph_sha256"]
    ):
        raise ValueError("composition bundle differs from frozen authority")
    layer10 = parents["layer10"]
    layer17 = parents["layer17"]
    expected_l10 = {
        "candidate_tensor_file_sha256": _EXPECTED_AUTHORITIES[
            "layer10_candidate_tensor_file_sha256"
        ],
        "candidate_scientific_payload_sha256": _EXPECTED_AUTHORITIES[
            "layer10_candidate_scientific_payload_sha256"
        ],
    }
    expected_l17 = {
        "candidate_tensor_file_sha256": _EXPECTED_AUTHORITIES[
            "layer17_candidate_tensor_file_sha256"
        ],
        "candidate_scientific_payload_sha256": _EXPECTED_AUTHORITIES[
            "layer17_candidate_scientific_payload_sha256"
        ],
    }
    for field, expected in expected_l10.items():
        if layer10.get(field) != expected:
            raise ValueError(f"layer10 {field} differs from frozen authority")
    for field, expected in expected_l17.items():
        if layer17.get(field) != expected:
            raise ValueError(f"layer17 {field} differs from frozen authority")
    l10_guard = layer10.get("guard_evidence")
    l17_guard = layer17.get("guard_evidence")
    if not isinstance(l10_guard, Mapping) or not isinstance(l17_guard, Mapping):
        raise TypeError("composition guard evidence is unavailable")
    if (
        l10_guard.get("evidence_file_sha256")
        != _EXPECTED_AUTHORITIES["layer10_guard_evidence_file_sha256"]
        or l10_guard.get("logical_sha256")
        != _EXPECTED_AUTHORITIES["layer10_guard_evidence_logical_sha256"]
        or l10_guard.get("status") != "passed"
        or l10_guard.get("assessment_role")
        != "claimed_closed_guard_assessment"
        or l10_guard.get("heldout_confirmation") is not True
        or l10_guard.get("fresh_validation") is not False
    ):
        raise ValueError("layer10 guard evidence differs from frozen authority")
    if (
        l17_guard.get("evidence_file_sha256")
        != _EXPECTED_AUTHORITIES["layer17_adaptive_result_file_sha256"]
        or l17_guard.get("logical_sha256")
        != _EXPECTED_AUTHORITIES["layer17_adaptive_result_sha256"]
        or l17_guard.get("status") != "passed"
        or l17_guard.get("assessment_role") != "open_development_assessment"
        or l17_guard.get("heldout_confirmation") is not False
        or l17_guard.get("fresh_validation") is not False
    ):
        raise ValueError("layer17 adaptive evidence differs from frozen authority")

    layer10_plan = _subgraph(primary, layer_ordinal=10, include_interactions=True)
    layer17_plan = _subgraph(edgeless, layer_ordinal=17, include_interactions=False)
    if (
        len(primary.nodes) != 8
        or len(primary.interactions) != 3
        or any("layer-10" not in edge.source_node for edge in primary.interactions)
        or edgeless.interactions
        or layer17_plan.artifact_sha256
        != _EXPECTED_AUTHORITIES["layer17_edgeless_graph_sha256"]
        or layer10_plan.parameter_count != 132_035
        or layer17_plan.parameter_count != 163_094
        or edgeless.parameter_count != 290_710
        or primary.parameter_count != 295_129
        or layer10_plan.macs_per_token != 129_248
        or layer17_plan.macs_per_token != 160_352
        or edgeless.macs_per_token != 285_280
        or primary.macs_per_token != 289_600
        or _resource_formula(layer10_plan) != 126_432
        or _resource_formula(layer17_plan) != 160_352
        or _resource_formula(edgeless) != 285_280
        or _resource_formula(primary) != 286_784
    ):
        raise ValueError("composition bundle resources/topology differ from policy")

    experiment_values: list[Mapping[str, object]] = []
    for parent in (layer10, layer17):
        candidate = parent.get("candidate")
        experiment = candidate.get("experiment") if isinstance(candidate, Mapping) else None
        if not isinstance(experiment, Mapping):
            raise TypeError("composition parent experiment is unavailable")
        experiment_values.append(experiment)
    model_ids = {value.get("model_id") for value in experiment_values}
    revisions = {value.get("requested_revision") for value in experiment_values}
    if (
        len(model_ids) != 1
        or len(revisions) != 1
        or not isinstance(next(iter(model_ids)), str)
        or not isinstance(next(iter(revisions)), str)
    ):
        raise ValueError("composition parent model identities differ")
    lineage = raw.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("composition lineage is unavailable")
    binding = {
        "bundle_file": source.name,
        "bundle_file_sha256": file_sha256,
        "composition_payload_sha256": _require_sha256(
            raw.get("composition_payload_sha256"), label="composition payload"
        ),
        "combined_edgeless_graph_sha256": edgeless.artifact_sha256,
        "combined_primary_graph_sha256": primary.artifact_sha256,
        "model_fingerprint": primary.model_fingerprint,
        "parameter_cluster_plan_sha256": primary.parameter_cluster_plan_sha256,
        "layer10_candidate_tensor_file_sha256": layer10[
            "candidate_tensor_file_sha256"
        ],
        "layer10_candidate_scientific_payload_sha256": layer10[
            "candidate_scientific_payload_sha256"
        ],
        "layer10_guard_evidence_file_sha256": l10_guard[
            "evidence_file_sha256"
        ],
        "layer10_guard_evidence_logical_sha256": l10_guard["logical_sha256"],
        "layer17_candidate_tensor_file_sha256": layer17[
            "candidate_tensor_file_sha256"
        ],
        "layer17_candidate_scientific_payload_sha256": layer17[
            "candidate_scientific_payload_sha256"
        ],
        "layer17_edgeless_graph_sha256": layer17_plan.artifact_sha256,
        "layer17_adaptive_evidence_file_sha256": l17_guard[
            "evidence_file_sha256"
        ],
        "layer17_adaptive_evidence_logical_sha256": l17_guard["logical_sha256"],
        "resources": dict(_EXPECTED_RESOURCES),
    }
    return _BundleAuthority(
        path=source,
        file_sha256=file_sha256,
        raw=raw,
        edgeless=edgeless,
        primary=primary,
        lowerings=lowerings,
        binding=binding,
        model_id=next(iter(model_ids)),  # type: ignore[arg-type]
        requested_revision=next(iter(revisions)),  # type: ignore[arg-type]
    )


def _authorize_before_selection(
    *,
    bundle_path: Path | str,
    lofo_report_path: Path | str,
    adaptive_result_path: Path | str,
) -> tuple[_BundleAuthority, dict[str, object], dict[str, object]]:
    """Authenticate all executable/evidence lineage without opening prompts."""

    bundle = _bundle_authority(bundle_path)
    lofo_path = Path(lofo_report_path)
    adaptive_path = Path(adaptive_result_path)
    lofo_file_sha256 = _file_sha256(lofo_path)
    adaptive_file_sha256 = _file_sha256(adaptive_path)
    if (
        lofo_file_sha256
        != _EXPECTED_AUTHORITIES["layer17_lofo_report_file_sha256"]
        or adaptive_file_sha256
        != _EXPECTED_AUTHORITIES["layer17_adaptive_result_file_sha256"]
    ):
        raise ValueError("layer17 evidence file differs from frozen authority")
    lofo = load_gemma3_layer17_v8_fit_lofo_report(lofo_path)
    adaptive = load_gemma3_layer17_open_a_capacity_result(adaptive_path)
    lofo_decision = lofo.get("decision")
    adaptive_decision = adaptive.get("adaptive_selection")
    adaptive_candidates = adaptive.get("candidates")
    adaptive_authorization = adaptive.get("authorization")
    adaptive_corpus = adaptive.get("corpus")
    if not all(
        isinstance(value, Mapping)
        for value in (
            lofo_decision,
            adaptive_decision,
            adaptive_candidates,
            adaptive_authorization,
            adaptive_corpus,
        )
    ):
        raise TypeError("layer17 source-safe authorization sections are unavailable")
    assert isinstance(lofo_decision, Mapping)
    assert isinstance(adaptive_decision, Mapping)
    assert isinstance(adaptive_candidates, Mapping)
    assert isinstance(adaptive_authorization, Mapping)
    challenger = adaptive_candidates.get("adaptive_a_fit")
    if not isinstance(challenger, Mapping):
        raise TypeError("adaptive layer17 challenger binding is unavailable")
    if (
        lofo.get("report_sha256")
        != _EXPECTED_AUTHORITIES["layer17_lofo_report_sha256"]
        or lofo_decision.get("all_required_gates_pass") is not True
        or adaptive.get("result_sha256")
        != _EXPECTED_AUTHORITIES["layer17_adaptive_result_sha256"]
        or adaptive.get("scientific_role")
        != "already_open_adaptive_development_fixed_capacity_refit"
        or adaptive_decision.get("all_required_gates_pass") is not True
        or adaptive_decision.get("adaptive_candidate_selected") is not True
        or adaptive_authorization.get("authorization_completed_before_selection_open")
        is not True
        or adaptive_authorization.get("lofo_report_file_sha256")
        != lofo_file_sha256
        or challenger.get("tensor_file_sha256")
        != bundle.binding["layer17_candidate_tensor_file_sha256"]
        or challenger.get("scientific_payload_sha256")
        != bundle.binding["layer17_candidate_scientific_payload_sha256"]
        or challenger.get("edgeless_graph_sha256")
        != bundle.binding["layer17_edgeless_graph_sha256"]
    ):
        raise ValueError("layer17 adaptive/LOFO lineage is not authorized")
    authority_without_digest: dict[str, object] = {
        "authorization_kind": (
            "frozen_layer10_guard_plus_passing_layer17_lofo_adaptive_selection"
        ),
        "authorization_completed_before_selection_open": True,
        "selection_access_authorized": True,
        "bundle": bundle.binding,
        "layer17_lofo_report_file": lofo_path.name,
        "layer17_lofo_report_file_sha256": lofo_file_sha256,
        "layer17_lofo_report_sha256": lofo["report_sha256"],
        "layer17_adaptive_result_file": adaptive_path.name,
        "layer17_adaptive_result_file_sha256": adaptive_file_sha256,
        "layer17_adaptive_result_sha256": adaptive["result_sha256"],
        "prior_selection_binding": dict(adaptive_corpus),
        "claim_role": "already_open_adaptive_development_selection",
        "fit_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "full_model_compiled": False,
        "source_safe": True,
    }
    authority = {
        **authority_without_digest,
        "authority_sha256": _domain_sha256(
            _AUTHORITY_DOMAIN, authority_without_digest
        ),
    }
    return bundle, authority, dict(adaptive_corpus)


def _record_execution(
    static_by_condition: dict[str, dict[str, object]],
    totals_by_condition: dict[str, dict[str, int]],
    peak_by_condition: dict[str, int],
    *,
    condition: str,
    execution: object,
) -> None:
    static = _execution_fields(
        execution, _GRAPH_STATIC_FIELDS, label=condition
    )
    prior = static_by_condition.setdefault(condition, static)
    if prior != static:
        raise RuntimeError(f"{condition} static accounting changed by batch")
    totals = totals_by_condition.setdefault(
        condition, {field: 0 for field in _GRAPH_LOGICAL_FIELDS}
    )
    for field in _GRAPH_LOGICAL_FIELDS:
        value = getattr(execution, field, None)
        if type(value) is not int:
            raise ValueError(f"{condition} {field} must be an integer")
        totals[field] += value
    peak = getattr(execution, "peak_live_modal_width", None)
    if type(peak) is not int or peak < 0:
        raise ValueError(f"{condition} peak modal width is invalid")
    peak_by_condition[condition] = max(
        peak_by_condition.get(condition, 0), peak
    )


def _compact_resource_record(
    *,
    condition: str,
    plan: ModalGeneratorGraphPlan,
    static: Mapping[str, object],
    totals: Mapping[str, int],
    logical_valid_tokens: int,
    peak_live_modal_width: int,
) -> dict[str, int]:
    if logical_valid_tokens <= 0:
        raise ValueError("logical token total must be positive")
    for field in _GRAPH_LOGICAL_FIELDS:
        if totals[field] % logical_valid_tokens:
            raise RuntimeError(f"{condition} {field} is not per-token exact")
    executed = totals["logical_executed_modal_graph_macs"] // logical_valid_tokens
    net_executed = totals["net_logical_macs_saved"] // logical_valid_tokens
    observed = {
        "replaced_layer_count": int(static["replaced_layer_count"]),
        "graph_node_count": int(static["graph_node_count"]),
        "interaction_count": len(plan.interactions),
        "native_removed_parameters": int(
            static["native_removed_learned_parameters"]
        ),
        "graph_parameters": int(static["modal_graph_learned_parameters"]),
        "net_parameter_savings": int(static["net_stored_parameter_savings"]),
        "dense_graph_macs_per_token": plan.macs_per_token,
        "executed_graph_macs_per_token": executed,
        "net_executed_macs_saved_per_token": net_executed,
    }
    expected = _EXPECTED_CONDITION_RESOURCES[condition]
    if observed != expected:
        raise RuntimeError(
            f"{condition} resources differ from progressive policy: "
            f"observed={observed!r}"
        )
    if (
        int(static["source_whole_model_learned_parameters"])
        != _EXPECTED_RESOURCES["source_whole_model_learned_parameters"]
        or int(static["candidate_whole_model_learned_parameters"])
        != int(static["source_whole_model_learned_parameters"])
        - observed["native_removed_parameters"]
        + observed["graph_parameters"]
    ):
        raise RuntimeError(f"{condition} whole-model parameter accounting drifted")
    if (
        totals["logical_linear_macs_native_removed"]
        != observed["native_removed_parameters"] * logical_valid_tokens
        or totals["logical_modal_graph_macs"]
        != observed["dense_graph_macs_per_token"] * logical_valid_tokens
        or peak_live_modal_width < 0
    ):
        raise RuntimeError(f"{condition} logical accounting drifted")
    return {**observed, "executed_peak_live_modal_width": peak_live_modal_width}


def _score_progressive_panel_in_transaction(
    *,
    adapter: Gemma3CausalLMAdapter,
    layer10_executor: Gemma3ModalGeneratorGraphExecutor,
    layer17_executor: Gemma3ModalGeneratorGraphExecutor,
    edgeless_executor: Gemma3ModalGeneratorGraphExecutor,
    primary_executor: Gemma3ModalGeneratorGraphExecutor,
    family_batches: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
) -> dict[str, object]:
    native_model = adapter.module
    if not callable(native_model):
        raise TypeError("adapter does not expose a callable native model")
    executors = {
        "layer10_dynamic": layer10_executor,
        "layer17_adaptive_edgeless": layer17_executor,
        "composed_edgeless": edgeless_executor,
        "composed_primary": primary_executor,
    }
    plans = {name: executor.graph_plan for name, executor in executors.items()}
    if (
        plans["layer17_adaptive_edgeless"].interactions
        or plans["composed_edgeless"].interactions
        or len(plans["layer10_dynamic"].interactions) != 3
        or len(plans["composed_primary"].interactions) != 3
    ):
        raise ValueError("progressive execution graph topology differs")
    if tuple(name for name, _ in family_batches) != tuple(
        f"family_{index:02d}" for index in range(_EXPECTED_FAMILIES)
    ):
        raise ValueError("progressive family batch catalog is invalid")
    example_ids = tuple(
        example_id
        for _, batches in family_batches
        for batch in batches
        for example_id in (
            batch.example_ids if batch.example_ids is not None else ()
        )
    )
    if (
        any(
            batch.example_ids is None
            for _, batches in family_batches
            for batch in batches
        )
        or len(example_ids) != _EXPECTED_EXAMPLES
        or len(set(example_ids)) != _EXPECTED_EXAMPLES
    ):
        raise ValueError("progressive batches must contain 128 unique examples")

    aggregate = _new_metric_accumulator(_CONDITIONS)
    family_accumulators = {
        family: _new_metric_accumulator(_CONDITIONS)
        for family, _ in family_batches
    }
    static_by_condition: dict[str, dict[str, object]] = {}
    totals_by_condition: dict[str, dict[str, int]] = {}
    peak_by_condition: dict[str, int] = {}
    logical_valid_tokens = 0
    deletion_max_abs = 0.0

    for family_index, (family, batches) in enumerate(family_batches):
        family_accumulator = family_accumulators[family]
        print(
            f"open-a composition: opaque family {family_index + 1}/4 "
            f"({len(batches)} batches)",
            flush=True,
        )
        for batch_index, batch in enumerate(batches):
            print(
                f"open-a composition: family {family_index + 1}/4, "
                f"batch {batch_index + 1}/{len(batches)}",
                flush=True,
            )
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output), batch
            )
            del native_output
            token_count = targets.numel()
            native_nll_sum = _native_nll(native_logits, targets)
            _add_native(
                aggregate, nll_sum=native_nll_sum, token_count=token_count
            )
            _add_native(
                family_accumulator,
                nll_sum=native_nll_sum,
                token_count=token_count,
            )

            valid_counts: list[int] = []
            current_static: dict[str, dict[str, object]] = {}
            for condition, executor in executors.items():
                plan = plans[condition]
                with torch.no_grad():
                    execution = executor.run(
                        batch.model_inputs, condition="generated"
                    )
                _validate_graph_execution(
                    execution,
                    plan,
                    condition="generated",
                    label=condition,
                )
                logits, candidate_targets = _selected_logits_and_targets(
                    _model_logits(execution.model_output), batch
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{condition} targets drifted")
                comparison = _candidate_comparison(
                    native_logits,
                    logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                )
                _add_comparison(aggregate, condition, comparison)
                _add_comparison(family_accumulator, condition, comparison)
                _record_execution(
                    static_by_condition,
                    totals_by_condition,
                    peak_by_condition,
                    condition=condition,
                    execution=execution,
                )
                current_static[condition] = _execution_fields(
                    execution, _GRAPH_STATIC_FIELDS, label=condition
                )
                valid = getattr(execution, "valid_tokens", None)
                if type(valid) is not int:
                    raise RuntimeError(f"{condition} valid-token count is invalid")
                valid_counts.append(valid)
                del logits, execution

            with torch.no_grad():
                deletion = primary_executor.run(
                    batch.model_inputs, condition="deletion"
                )
            _validate_graph_execution(
                deletion,
                plans["composed_primary"],
                condition="deletion",
                label="matched double deletion",
            )
            deletion_logits, deletion_targets = _selected_logits_and_targets(
                _model_logits(deletion.model_output), batch
            )
            if not torch.equal(targets, deletion_targets):
                raise RuntimeError("matched double-deletion targets drifted")
            comparison = _candidate_comparison(
                native_logits,
                deletion_logits,
                targets,
                vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
            )
            _add_comparison(aggregate, "matched_double_deletion", comparison)
            _add_comparison(
                family_accumulator, "matched_double_deletion", comparison
            )
            _record_execution(
                static_by_condition,
                totals_by_condition,
                peak_by_condition,
                condition="matched_double_deletion",
                execution=deletion,
            )
            if any(
                getattr(deletion, field, None) != 0
                for field in (
                    "logical_executed_modal_graph_macs",
                    "logical_executed_modal_graph_additions",
                    "peak_live_modal_width",
                )
            ):
                raise RuntimeError("matched double deletion executed graph work")
            deletion_static = _execution_fields(
                deletion,
                _GRAPH_STATIC_FIELDS,
                label="matched double deletion",
            )
            if deletion_static != current_static["composed_primary"]:
                raise RuntimeError("generated/deletion static accounting differs")
            deletion_valid = getattr(deletion, "valid_tokens", None)
            if type(deletion_valid) is not int:
                raise RuntimeError("deletion valid-token count is invalid")
            valid_counts.append(deletion_valid)

            with torch.no_grad():
                other_deletion = edgeless_executor.run(
                    batch.model_inputs, condition="deletion"
                )
            _validate_graph_execution(
                other_deletion,
                plans["composed_edgeless"],
                condition="deletion",
                label="edgeless double deletion",
            )
            other_logits, other_targets = _selected_logits_and_targets(
                _model_logits(other_deletion.model_output), batch
            )
            if not torch.equal(targets, other_targets):
                raise RuntimeError("edgeless deletion targets drifted")
            deletion_max_abs = max(
                deletion_max_abs,
                _assert_close_logits(
                    deletion_logits,
                    other_logits,
                    atol=0.0,
                    rtol=0.0,
                    label="primary/edgeless double deletion",
                ),
            )
            if any(
                getattr(other_deletion, field, None) != 0
                for field in (
                    "logical_executed_modal_graph_macs",
                    "logical_executed_modal_graph_additions",
                    "peak_live_modal_width",
                )
            ):
                raise RuntimeError("edgeless deletion executed graph work")
            if _execution_fields(
                other_deletion,
                _GRAPH_STATIC_FIELDS,
                label="edgeless double deletion",
            ) != current_static["composed_edgeless"]:
                raise RuntimeError("edgeless generated/deletion accounting differs")
            other_valid = getattr(other_deletion, "valid_tokens", None)
            if type(other_valid) is not int:
                raise RuntimeError("edgeless deletion valid-token count is invalid")
            valid_counts.append(other_valid)
            expected_valid = int(batch.valid_positions.sum().item())
            if set(valid_counts) != {expected_valid}:
                raise RuntimeError("progressive conditions disagree on valid tokens")
            logical_valid_tokens += expected_valid
            del (
                native_logits,
                deletion_logits,
                other_logits,
                deletion,
                other_deletion,
            )

    micro = _finalize_metric_accumulator(aggregate, conditions=_CONDITIONS)
    families = {
        family: _finalize_metric_accumulator(
            accumulator, conditions=_CONDITIONS
        )
        for family, accumulator in family_accumulators.items()
    }
    macro = _equal_family_macro(families, conditions=_CONDITIONS)
    plan_by_condition = {
        **plans,
        "matched_double_deletion": plans["composed_primary"],
    }
    resources = {
        condition: _compact_resource_record(
            condition=condition,
            plan=plan_by_condition[condition],
            static=static_by_condition[condition],
            totals=totals_by_condition[condition],
            logical_valid_tokens=logical_valid_tokens,
            peak_live_modal_width=peak_by_condition[condition],
        )
        for condition in _CONDITIONS
    }
    return {
        "execution_path": "heterogeneous_layer10_dynamic_layer17_adaptive_graph",
        "assessment_role": "adaptive_open_development_composition",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "example_count": _EXPECTED_EXAMPLES,
        "family_count": _EXPECTED_FAMILIES,
        "supervised_tokens": micro["supervised_tokens"],
        "logical_valid_tokens": logical_valid_tokens,
        "native": micro["native"],
        "conditions": micro["conditions"],
        "equal_family_macro": macro,
        "families": families,
        "graph_comparison": {
            "node_count": 8,
            "primary_interaction_count": 3,
            "edgeless_interaction_count": 0,
            "layer17_interaction_count": 0,
            "primary_edges_are_layer10_only": True,
            "node_artifacts_identical_between_composed_arms": True,
            "double_deletion_paths_agree": True,
            "deletion_equivalence_atol": 0.0,
            "deletion_equivalence_rtol": 0.0,
            "deletion_max_abs_logit_difference": deletion_max_abs,
        },
        "resource_accounting": resources,
        "observed_resources": dict(_EXPECTED_RESOURCES),
        "latency_or_kernel_speed_claim": False,
    }


def _score_progressive_panel(
    *,
    adapter: Gemma3CausalLMAdapter,
    layer10_executor: Gemma3ModalGeneratorGraphExecutor,
    layer17_executor: Gemma3ModalGeneratorGraphExecutor,
    edgeless_executor: Gemma3ModalGeneratorGraphExecutor,
    primary_executor: Gemma3ModalGeneratorGraphExecutor,
    family_batches: Sequence[tuple[str, tuple[CalibrationBatch, ...]]],
) -> dict[str, object]:
    executor_ids = {
        id(layer10_executor),
        id(layer17_executor),
        id(edgeless_executor),
        id(primary_executor),
    }
    if len(executor_ids) != 4:
        raise ValueError("progressive executors must be distinct")
    with ExitStack() as stack:
        for executor in (
            layer10_executor,
            layer17_executor,
            edgeless_executor,
            primary_executor,
        ):
            stack.enter_context(executor.validated_transaction())
        return _score_progressive_panel_in_transaction(
            adapter=adapter,
            layer10_executor=layer10_executor,
            layer17_executor=layer17_executor,
            edgeless_executor=edgeless_executor,
            primary_executor=primary_executor,
            family_batches=family_batches,
        )


_METRIC_FIELDS = {
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
}


def _native_metric(container: Mapping[str, object], *, label: str) -> float:
    native = container.get("native")
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError(f"{label} native metric is invalid")
    value = _finite(native.get("nll_per_token"), label=f"{label} native NLL")
    if value < 0.0:
        raise ValueError(f"{label} native NLL must be nonnegative")
    return value


def _condition_metric(
    container: Mapping[str, object], condition: str, *, label: str
) -> dict[str, float]:
    conditions = container.get("conditions")
    record = conditions.get(condition) if isinstance(conditions, Mapping) else None
    if not isinstance(record, Mapping) or set(record) != _METRIC_FIELDS:
        raise ValueError(f"{label} {condition} metrics are invalid")
    metrics = {
        key: _finite(value, label=f"{label} {condition} {key}")
        for key, value in record.items()
    }
    if (
        metrics["nll_per_token"] < 0.0
        or metrics["native_to_candidate_kl_per_token"] < 0.0
        or not 0.0 <= metrics["top1_agreement_to_native"] <= 1.0
    ):
        raise ValueError(f"{label} {condition} metrics are out of range")
    return metrics


def _require_metric_identity(
    metric: Mapping[str, float],
    *,
    native_nll: float,
    label: str,
    maximum_ulps: int,
) -> None:
    valid, _ = _metric_identity_within_ulps(
        metric["delta_nll_per_token"],
        metric["nll_per_token"] - native_nll,
        operands=(metric["nll_per_token"], native_nll),
        maximum_ulps=maximum_ulps,
    )
    if not valid:
        raise ValueError(f"{label} NLL/delta identity is invalid")


def _scalar_identity(
    actual: float,
    expected: float,
    *,
    operands: Sequence[float],
    maximum_ulps: int,
    label: str,
) -> None:
    valid, _ = _metric_identity_within_ulps(
        actual,
        expected,
        operands=operands,
        maximum_ulps=maximum_ulps,
    )
    if not valid:
        raise ValueError(f"{label} does not reproduce")


def _require_macro_reproduction(
    supplied: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    if set(supplied) != {"native", "conditions"} or set(expected) != {
        "native",
        "conditions",
    }:
        raise ValueError("equal-family macro fields are invalid")
    supplied_native = _native_metric(supplied, label="supplied macro")
    expected_native = _native_metric(expected, label="expected macro")
    _scalar_identity(
        supplied_native,
        expected_native,
        operands=(supplied_native, expected_native),
        maximum_ulps=32,
        label="equal-family macro native NLL",
    )
    for condition in _CONDITIONS:
        left = _condition_metric(supplied, condition, label="supplied macro")
        right = _condition_metric(expected, condition, label="expected macro")
        for field in _METRIC_FIELDS:
            _scalar_identity(
                left[field],
                right[field],
                operands=(left[field], right[field]),
                maximum_ulps=32,
                label=f"equal-family macro {condition} {field}",
            )


def _gate(
    gate_id: str,
    *,
    observed: bool | int | float,
    operator: str,
    threshold: bool | int | float,
    passed: bool,
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "required": True,
        "passed": bool(passed),
    }


def _deletion_recovery(
    *, primary_delta: float, deletion_delta: float
) -> tuple[bool, float]:
    if deletion_delta <= 0.0:
        return False, 0.0
    return True, (deletion_delta - primary_delta) / deletion_delta


def _validate_resource_accounting(
    assessment: Mapping[str, object],
) -> tuple[bool, dict[str, int]]:
    accounting = assessment.get("resource_accounting")
    observed = assessment.get("observed_resources")
    if not isinstance(accounting, Mapping) or set(accounting) != set(_CONDITIONS):
        raise ValueError("assessment resource condition catalog is invalid")
    for condition in _CONDITIONS:
        record = accounting[condition]
        expected = {
            **_EXPECTED_CONDITION_RESOURCES[condition],
            "executed_peak_live_modal_width": None,
        }
        if not isinstance(record, Mapping) or set(record) != set(expected):
            raise ValueError(f"{condition} resource fields are invalid")
        peak = record.get("executed_peak_live_modal_width")
        if type(peak) is not int or peak < 0:
            raise ValueError(f"{condition} peak width is invalid")
        if {
            key: record[key]
            for key in _EXPECTED_CONDITION_RESOURCES[condition]
        } != _EXPECTED_CONDITION_RESOURCES[condition]:
            return False, {}
    if not isinstance(observed, Mapping):
        raise TypeError("assessment cumulative resources are unavailable")
    reproduced = {
        "source_whole_model_learned_parameters": _EXPECTED_RESOURCES[
            "source_whole_model_learned_parameters"
        ],
        "replaced_layer_count": accounting["composed_primary"][  # type: ignore[index]
            "replaced_layer_count"
        ],
        "graph_node_count": accounting["composed_primary"][  # type: ignore[index]
            "graph_node_count"
        ],
        "dynamic_interaction_count": accounting["composed_primary"][  # type: ignore[index]
            "interaction_count"
        ],
        "native_removed_parameters": accounting["composed_primary"][  # type: ignore[index]
            "native_removed_parameters"
        ],
        "primary_graph_parameters": accounting["composed_primary"][  # type: ignore[index]
            "graph_parameters"
        ],
        "net_stored_parameter_savings": accounting["composed_primary"][  # type: ignore[index]
            "net_parameter_savings"
        ],
        "candidate_whole_model_learned_parameters": (
            _EXPECTED_RESOURCES["source_whole_model_learned_parameters"]
            - int(
                accounting["composed_primary"][  # type: ignore[index]
                    "native_removed_parameters"
                ]
            )
            + int(
                accounting["composed_primary"][  # type: ignore[index]
                    "graph_parameters"
                ]
            )
        ),
        "dense_graph_macs_per_token": accounting["composed_primary"][  # type: ignore[index]
            "dense_graph_macs_per_token"
        ],
        "executed_graph_macs_per_token": accounting["composed_primary"][  # type: ignore[index]
            "executed_graph_macs_per_token"
        ],
        "native_removed_macs_per_token": accounting["composed_primary"][  # type: ignore[index]
            "native_removed_parameters"
        ],
        "net_executed_macs_saved_per_token": accounting[
            "composed_primary"
        ]["net_executed_macs_saved_per_token"],  # type: ignore[index]
        "layer17_only_net_parameter_savings": accounting[
            "layer17_adaptive_edgeless"
        ]["net_parameter_savings"],  # type: ignore[index]
        "layer17_only_net_executed_macs_saved_per_token": accounting[
            "layer17_adaptive_edgeless"
        ]["net_executed_macs_saved_per_token"],  # type: ignore[index]
        "incremental_parameter_savings_vs_layer17_only": (
            int(
                accounting["composed_primary"][  # type: ignore[index]
                    "net_parameter_savings"
                ]
            )
            - int(
                accounting["layer17_adaptive_edgeless"][  # type: ignore[index]
                    "net_parameter_savings"
                ]
            )
        ),
        "incremental_executed_macs_saved_vs_layer17_only": (
            int(
                accounting["composed_primary"][  # type: ignore[index]
                    "net_executed_macs_saved_per_token"
                ]
            )
            - int(
                accounting["layer17_adaptive_edgeless"][  # type: ignore[index]
                    "net_executed_macs_saved_per_token"
                ]
            )
        ),
    }
    exact = _canonical_equal(observed, reproduced) and reproduced == _EXPECTED_RESOURCES
    return exact, {key: int(value) for key, value in reproduced.items()}


def progressive_composition_decision(
    assessment: Mapping[str, object],
) -> dict[str, object]:
    """Replay the immutable progressive-development gates from scalar data."""

    if (
        assessment.get("assessment_role")
        != "adaptive_open_development_composition"
        or assessment.get("heldout_confirmation") is not False
        or assessment.get("full_model_logits_scored") is not True
        or assessment.get("full_model_compiled") is not False
        or assessment.get("example_count") != _EXPECTED_EXAMPLES
        or assessment.get("family_count") != _EXPECTED_FAMILIES
        or assessment.get("latency_or_kernel_speed_claim") is not False
    ):
        raise ValueError("progressive assessment identity is invalid")
    supervised = assessment.get("supervised_tokens")
    logical = assessment.get("logical_valid_tokens")
    if (
        type(supervised) is not int
        or supervised <= 0
        or type(logical) is not int
        or logical <= 0
    ):
        raise ValueError("progressive assessment token counts are invalid")
    native_nll = _native_metric(assessment, label="micro")
    micro = {
        condition: _condition_metric(assessment, condition, label="micro")
        for condition in _CONDITIONS
    }
    for condition, metric in micro.items():
        _require_metric_identity(
            metric,
            native_nll=native_nll,
            label=f"micro {condition}",
            maximum_ulps=2,
        )

    macro_container = assessment.get("equal_family_macro")
    families = assessment.get("families")
    expected_slots = {f"family_{index:02d}" for index in range(4)}
    if not isinstance(macro_container, Mapping):
        raise TypeError("equal-family macro is unavailable")
    if not isinstance(families, Mapping) or set(families) != expected_slots:
        raise ValueError("family metric slots are invalid")
    macro_native = _native_metric(macro_container, label="macro")
    macro = {
        condition: _condition_metric(
            macro_container, condition, label="macro"
        )
        for condition in _CONDITIONS
    }
    for condition, metric in macro.items():
        _require_metric_identity(
            metric,
            native_nll=macro_native,
            label=f"macro {condition}",
            maximum_ulps=32,
        )

    family_metrics: dict[str, dict[str, dict[str, float]]] = {}
    family_tokens = 0
    for slot in sorted(expected_slots):
        family = families[slot]
        if not isinstance(family, Mapping):
            raise TypeError(f"{slot} metrics are invalid")
        count = family.get("supervised_tokens")
        if type(count) is not int or count <= 0:
            raise ValueError(f"{slot} token count is invalid")
        family_tokens += count
        family_native = _native_metric(family, label=slot)
        family_metrics[slot] = {}
        for condition in _CONDITIONS:
            metric = _condition_metric(family, condition, label=slot)
            _require_metric_identity(
                metric,
                native_nll=family_native,
                label=f"{slot} {condition}",
                maximum_ulps=2,
            )
            family_metrics[slot][condition] = metric
    if family_tokens != supervised:
        raise ValueError("family tokens do not sum to the micro total")
    recomputed_macro = _equal_family_macro(
        families, conditions=_CONDITIONS  # type: ignore[arg-type]
    )
    _require_macro_reproduction(macro_container, recomputed_macro)

    # Reproduce the micro averages from the source-safe per-family scalars.
    reproduced_native = sum(
        int(families[slot]["supervised_tokens"])  # type: ignore[index]
        * _native_metric(families[slot], label=slot)  # type: ignore[arg-type]
        for slot in sorted(expected_slots)
    ) / supervised
    _scalar_identity(
        native_nll,
        reproduced_native,
        operands=(native_nll, reproduced_native),
        maximum_ulps=32,
        label="micro native NLL",
    )
    for condition in _CONDITIONS:
        for metric_name in (
            "nll_per_token",
            "native_to_candidate_kl_per_token",
            "top1_agreement_to_native",
        ):
            reproduced = sum(
                int(families[slot]["supervised_tokens"])  # type: ignore[index]
                * family_metrics[slot][condition][metric_name]
                for slot in sorted(expected_slots)
            ) / supervised
            _scalar_identity(
                micro[condition][metric_name],
                reproduced,
                operands=(micro[condition][metric_name], reproduced),
                maximum_ulps=32,
                label=f"micro {condition} {metric_name}",
            )

    primary = micro["composed_primary"]
    macro_primary = macro["composed_primary"]
    macro_interaction_excess = (
        macro_primary["delta_nll_per_token"]
        - macro["layer10_dynamic"]["delta_nll_per_token"]
        - macro["layer17_adaptive_edgeless"]["delta_nll_per_token"]
    )
    primary_regression_to_edgeless = (
        primary["nll_per_token"]
        - micro["composed_edgeless"]["nll_per_token"]
    )
    macro_denominator_valid, macro_recovery = _deletion_recovery(
        primary_delta=macro_primary["delta_nll_per_token"],
        deletion_delta=macro["matched_double_deletion"][
            "delta_nll_per_token"
        ],
    )
    family_recovery: dict[str, float] = {}
    invalid_denominator_count = 0
    passing_family_count = 0
    for slot in sorted(expected_slots):
        family_primary = family_metrics[slot]["composed_primary"]
        valid, recovery = _deletion_recovery(
            primary_delta=family_primary["delta_nll_per_token"],
            deletion_delta=family_metrics[slot]["matched_double_deletion"][
                "delta_nll_per_token"
            ],
        )
        invalid_denominator_count += int(not valid)
        family_recovery[slot] = recovery
        passing_family_count += int(
            family_primary["delta_nll_per_token"]
            <= float(_POLICY["maximum_family_delta_nll_per_token"])
        )
    worst_family_recovery = min(family_recovery.values())
    resources_exact, resources = _validate_resource_accounting(assessment)
    graph = assessment.get("graph_comparison")
    graph_exact = bool(
        isinstance(graph, Mapping)
        and graph.get("node_count") == 8
        and graph.get("primary_interaction_count") == 3
        and graph.get("edgeless_interaction_count") == 0
        and graph.get("layer17_interaction_count") == 0
        and graph.get("primary_edges_are_layer10_only") is True
        and graph.get("node_artifacts_identical_between_composed_arms") is True
        and graph.get("double_deletion_paths_agree") is True
        and graph.get("deletion_equivalence_atol") == 0.0
        and graph.get("deletion_equivalence_rtol") == 0.0
        and graph.get("deletion_max_abs_logit_difference") == 0.0
    )

    gates = (
        _gate(
            "micro_delta_nll_per_token",
            observed=primary["delta_nll_per_token"],
            operator="<=",
            threshold=float(_POLICY["maximum_micro_delta_nll_per_token"]),
            passed=primary["delta_nll_per_token"]
            <= float(_POLICY["maximum_micro_delta_nll_per_token"]),
        ),
        _gate(
            "equal_family_macro_delta_nll_per_token",
            observed=macro_primary["delta_nll_per_token"],
            operator="<=",
            threshold=float(
                _POLICY["maximum_equal_family_macro_delta_nll_per_token"]
            ),
            passed=macro_primary["delta_nll_per_token"]
            <= float(_POLICY["maximum_equal_family_macro_delta_nll_per_token"]),
        ),
        _gate(
            "equal_family_macro_native_kl_per_token",
            observed=macro_primary["native_to_candidate_kl_per_token"],
            operator="<=",
            threshold=float(
                _POLICY["maximum_equal_family_macro_native_kl_per_token"]
            ),
            passed=macro_primary["native_to_candidate_kl_per_token"]
            <= float(_POLICY["maximum_equal_family_macro_native_kl_per_token"]),
        ),
        _gate(
            "equal_family_macro_top1_agreement",
            observed=macro_primary["top1_agreement_to_native"],
            operator=">=",
            threshold=float(
                _POLICY["minimum_equal_family_macro_top1_agreement"]
            ),
            passed=macro_primary["top1_agreement_to_native"]
            >= float(_POLICY["minimum_equal_family_macro_top1_agreement"]),
        ),
        _gate(
            "family_delta_nll_pass_count",
            observed=passing_family_count,
            operator=">=",
            threshold=int(_POLICY["minimum_passing_family_count"]),
            passed=passing_family_count
            >= int(_POLICY["minimum_passing_family_count"]),
        ),
        _gate(
            "macro_interaction_excess_nll",
            observed=macro_interaction_excess,
            operator="<=",
            threshold=float(_POLICY["maximum_macro_interaction_excess_nll"]),
            passed=macro_interaction_excess
            <= float(_POLICY["maximum_macro_interaction_excess_nll"]),
        ),
        _gate(
            "macro_deletion_recovery_fraction",
            observed=macro_recovery,
            operator=">=",
            threshold=float(
                _POLICY["minimum_macro_deletion_recovery_fraction"]
            ),
            passed=macro_denominator_valid
            and macro_recovery
            >= float(_POLICY["minimum_macro_deletion_recovery_fraction"]),
        ),
        _gate(
            "worst_family_deletion_recovery_fraction",
            observed=worst_family_recovery,
            operator=">=",
            threshold=float(
                _POLICY["minimum_worst_family_deletion_recovery_fraction"]
            ),
            passed=invalid_denominator_count == 0
            and worst_family_recovery
            >= float(
                _POLICY["minimum_worst_family_deletion_recovery_fraction"]
            ),
        ),
        _gate(
            "primary_nll_regression_to_composed_edgeless",
            observed=primary_regression_to_edgeless,
            operator="<=",
            threshold=float(
                _POLICY["maximum_primary_nll_regression_to_composed_edgeless"]
            ),
            passed=primary_regression_to_edgeless
            <= float(
                _POLICY["maximum_primary_nll_regression_to_composed_edgeless"]
            ),
        ),
        _gate(
            "exact_cumulative_resources",
            observed=resources_exact,
            operator="==",
            threshold=True,
            passed=resources_exact,
        ),
        _gate(
            "positive_incremental_parameter_savings_vs_layer17_only",
            observed=resources.get(
                "incremental_parameter_savings_vs_layer17_only", 0
            ),
            operator=">",
            threshold=0,
            passed=resources.get(
                "incremental_parameter_savings_vs_layer17_only", 0
            )
            > 0,
        ),
        _gate(
            "positive_incremental_executed_mac_savings_vs_layer17_only",
            observed=resources.get(
                "incremental_executed_macs_saved_vs_layer17_only", 0
            ),
            operator=">",
            threshold=0,
            passed=resources.get(
                "incremental_executed_macs_saved_vs_layer17_only", 0
            )
            > 0,
        ),
        _gate(
            "exact_graph_and_double_deletion_controls",
            observed=graph_exact,
            operator="==",
            threshold=True,
            passed=graph_exact,
        ),
    )
    all_pass = all(row["passed"] is True for row in gates)
    policy = {
        **_POLICY,
        "artifact_sha256": _domain_sha256(_POLICY_DOMAIN, _POLICY),
    }
    return {
        "eligible_condition": "composed_primary",
        "controls_eligible": False,
        "assessment_role": "adaptive_open_development_composition",
        "heldout_confirmation": False,
        "heldout_or_serving_authorized": False,
        "all_required_gates_pass": all_pass,
        "gate_table": gates,
        "policy": policy,
        "derived_metrics": {
            "passing_family_count": passing_family_count,
            "macro_interaction_excess_nll": macro_interaction_excess,
            "primary_nll_regression_to_composed_edgeless": (
                primary_regression_to_edgeless
            ),
            "macro_deletion_recovery_denominator_valid": (
                macro_denominator_valid
            ),
            "macro_deletion_recovery_fraction": macro_recovery,
            "family_deletion_recovery_invalid_denominator_count": (
                invalid_denominator_count
            ),
            "worst_family_deletion_recovery_fraction": worst_family_recovery,
            "family_deletion_recovery_fraction_by_alias": family_recovery,
            "incremental_parameter_savings_vs_layer17_only": resources.get(
                "incremental_parameter_savings_vs_layer17_only", 0
            ),
            "incremental_executed_macs_saved_vs_layer17_only": resources.get(
                "incremental_executed_macs_saved_vs_layer17_only", 0
            ),
        },
        "next_action": (
            "retain_two_layer_composition_for_open_development_only"
            if all_pass
            else "reject_two_layer_composition_keep_layer17_only_candidate"
        ),
    }


_ROOT_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "heldout_confirmation",
    "authorization",
    "corpus",
    "runtime",
    "tokenization",
    "assessment",
    "decision",
    "bundle_changed",
    "evidence_changed",
    "selection_opened",
    "fit_opened",
    "guard_opened",
    "calibration_b_opened",
    "validation_opened",
    "test_opened",
    "full_model_logits_scored",
    "full_model_compiled",
    "serving_authorized",
    "latency_or_kernel_speed_claim",
    "safety",
    "result_sha256",
}
_ASSESSMENT_FIELDS = {
    "execution_path",
    "assessment_role",
    "heldout_confirmation",
    "full_model_logits_scored",
    "full_model_compiled",
    "example_count",
    "family_count",
    "supervised_tokens",
    "logical_valid_tokens",
    "native",
    "conditions",
    "equal_family_macro",
    "families",
    "graph_comparison",
    "resource_accounting",
    "observed_resources",
    "latency_or_kernel_speed_claim",
}
_CORPUS_FIELDS = {
    "corpus_artifact_file",
    "corpus_artifact_file_sha256",
    "corpus_artifact_sha256",
    "receipt_file",
    "receipt_file_sha256",
    "receipt_sha256",
    "selection_role_file",
    "selection_role_file_sha256",
    "selection_manifest_sha256",
    "ordered_membership_sha256",
    "tokenizer_contract_sha256",
    "example_count",
    "family_count",
    "assessment_role",
}
_TOKENIZATION_FIELDS = {
    "family_stream_count",
    "family_stream_catalog_sha256",
    "example_count",
    "logical_valid_tokens",
    "supervised_tokens",
    "max_length",
    "tokenization_batch_size",
    "contains_prompt_text",
    "contains_prompt_identities",
    "contains_token_ids",
}
_RUNTIME_FIELDS = {
    "model_id",
    "requested_revision",
    "model_fingerprint",
    "device",
    "dtype",
    "tokenization_batch_size",
    "max_length",
    "vocabulary_chunk_size",
    "local_files_only",
}
_AUTHORIZATION_FIELDS = {
    "authorization_kind",
    "authorization_completed_before_selection_open",
    "selection_access_authorized",
    "bundle",
    "layer17_lofo_report_file",
    "layer17_lofo_report_file_sha256",
    "layer17_lofo_report_sha256",
    "layer17_adaptive_result_file",
    "layer17_adaptive_result_file_sha256",
    "layer17_adaptive_result_sha256",
    "prior_selection_binding",
    "claim_role",
    "fit_opened",
    "guard_opened",
    "calibration_b_opened",
    "validation_opened",
    "test_opened",
    "heldout_confirmation",
    "serving_authorized",
    "full_model_compiled",
    "source_safe",
    "authority_sha256",
}
_BUNDLE_BINDING_FIELDS = {
    "bundle_file",
    "bundle_file_sha256",
    "composition_payload_sha256",
    "combined_edgeless_graph_sha256",
    "combined_primary_graph_sha256",
    "model_fingerprint",
    "parameter_cluster_plan_sha256",
    "layer10_candidate_tensor_file_sha256",
    "layer10_candidate_scientific_payload_sha256",
    "layer10_guard_evidence_file_sha256",
    "layer10_guard_evidence_logical_sha256",
    "layer17_candidate_tensor_file_sha256",
    "layer17_candidate_scientific_payload_sha256",
    "layer17_edgeless_graph_sha256",
    "layer17_adaptive_evidence_file_sha256",
    "layer17_adaptive_evidence_logical_sha256",
    "resources",
}
_FORBIDDEN_KEYS = {
    "prompt",
    "prompts",
    "prompt_text",
    "ordered_prompt_sha256s",
    "prompt_sha256s",
    "family_id",
    "family_ids",
    "input_ids",
    "token_ids",
    "logits",
    "weights",
    "model_weights",
    "candidate_weights",
    "activation_rows",
    "gradient_rows",
    "latent_rows",
    "target_residual_rows",
}
_DECISION_FIELDS = {
    "eligible_condition",
    "controls_eligible",
    "assessment_role",
    "heldout_confirmation",
    "heldout_or_serving_authorized",
    "all_required_gates_pass",
    "gate_table",
    "policy",
    "derived_metrics",
    "next_action",
}
_GATE_FIELDS = {
    "gate_id",
    "observed",
    "operator",
    "threshold",
    "required",
    "passed",
}
_GRAPH_COMPARISON_FIELDS = {
    "node_count",
    "primary_interaction_count",
    "edgeless_interaction_count",
    "layer17_interaction_count",
    "primary_edges_are_layer10_only",
    "node_artifacts_identical_between_composed_arms",
    "double_deletion_paths_agree",
    "deletion_equivalence_atol",
    "deletion_equivalence_rtol",
    "deletion_max_abs_logit_difference",
}
_RESOURCE_RECORD_FIELDS = {
    "replaced_layer_count",
    "graph_node_count",
    "interaction_count",
    "native_removed_parameters",
    "graph_parameters",
    "net_parameter_savings",
    "dense_graph_macs_per_token",
    "executed_graph_macs_per_token",
    "net_executed_macs_saved_per_token",
    "executed_peak_live_modal_width",
}
_CHECKPOINT_FIELDS = {
    "schema",
    "format_version",
    "status",
    "intended_final_output_file",
    "unvalidated_result",
    "unvalidated_result_sha256",
    "safety",
    "checkpoint_sha256",
}
_CHECKPOINT_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_open_a_prevalidation_checkpoint"
)


def _reject_forbidden_fields(value: object, *, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"{path}.{key} is a forbidden source field")
            _reject_forbidden_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=f"{path}[{index}]")


def _validate_metric_shape(container: Mapping[str, object], *, label: str) -> None:
    if set(container) not in (
        {"native", "conditions"},
        {"supervised_tokens", "native", "conditions"},
    ):
        raise ValueError(f"{label} metric container fields are invalid")
    native = container.get("native")
    conditions = container.get("conditions")
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError(f"{label} native metric fields are invalid")
    if not isinstance(conditions, Mapping) or set(conditions) != set(_CONDITIONS):
        raise ValueError(f"{label} condition catalog is invalid")
    for condition in _CONDITIONS:
        record = conditions[condition]
        if not isinstance(record, Mapping) or set(record) != _METRIC_FIELDS:
            raise ValueError(f"{label} {condition} metric fields are invalid")
        for field, scalar in record.items():
            _finite(scalar, label=f"{label} {condition} {field}")
    _finite(native["nll_per_token"], label=f"{label} native NLL")


def _validate_unvalidated_result_shape(raw: Mapping[str, object]) -> None:
    """Whitelist a scored scalar envelope without applying metric identities."""

    if set(raw) != _ROOT_FIELDS:
        raise ValueError("unvalidated progressive result fields are invalid")
    if raw.get("schema") != _SCHEMA or raw.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unvalidated progressive result header is invalid")
    if raw.get("safety") != _SAFETY:
        raise ValueError("unvalidated progressive safety flags are invalid")
    corpus = raw.get("corpus")
    tokenization = raw.get("tokenization")
    runtime = raw.get("runtime")
    assessment = raw.get("assessment")
    decision = raw.get("decision")
    authorization = raw.get("authorization")
    if not isinstance(corpus, Mapping) or set(corpus) != _CORPUS_FIELDS:
        raise ValueError("unvalidated corpus fields are invalid")
    if not isinstance(tokenization, Mapping) or set(tokenization) != _TOKENIZATION_FIELDS:
        raise ValueError("unvalidated tokenization fields are invalid")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("unvalidated runtime fields are invalid")
    if not isinstance(assessment, Mapping) or set(assessment) != _ASSESSMENT_FIELDS:
        raise ValueError("unvalidated assessment fields are invalid")
    if not isinstance(decision, Mapping) or set(decision) != _DECISION_FIELDS:
        raise ValueError("unvalidated decision fields are invalid")
    if not isinstance(authorization, Mapping):
        raise TypeError("unvalidated authorization is unavailable")
    _validate_authorization(authorization, corpus)
    _validate_metric_shape(
        {
            "supervised_tokens": assessment["supervised_tokens"],
            "native": assessment["native"],
            "conditions": assessment["conditions"],
        },
        label="unvalidated micro",
    )
    macro = assessment.get("equal_family_macro")
    families = assessment.get("families")
    if not isinstance(macro, Mapping):
        raise TypeError("unvalidated macro is unavailable")
    _validate_metric_shape(macro, label="unvalidated macro")
    expected_slots = {f"family_{index:02d}" for index in range(4)}
    if not isinstance(families, Mapping) or set(families) != expected_slots:
        raise ValueError("unvalidated family slots are invalid")
    for slot, family in families.items():
        if not isinstance(family, Mapping):
            raise TypeError(f"unvalidated {slot} is invalid")
        _validate_metric_shape(family, label=f"unvalidated {slot}")
    graph = assessment.get("graph_comparison")
    resources = assessment.get("resource_accounting")
    if not isinstance(graph, Mapping) or set(graph) != _GRAPH_COMPARISON_FIELDS:
        raise ValueError("unvalidated graph-comparison fields are invalid")
    if not isinstance(resources, Mapping) or set(resources) != set(_CONDITIONS):
        raise ValueError("unvalidated resource catalog is invalid")
    for condition, record in resources.items():
        if not isinstance(record, Mapping) or set(record) != _RESOURCE_RECORD_FIELDS:
            raise ValueError(f"unvalidated {condition} resource fields are invalid")
    gates = decision.get("gate_table")
    if (
        not isinstance(gates, (tuple, list))
        or len(gates) != 13
        or any(not isinstance(row, Mapping) or set(row) != _GATE_FIELDS for row in gates)
    ):
        raise ValueError("unvalidated gate table is invalid")
    policy = decision.get("policy")
    if not isinstance(policy, Mapping) or set(policy) != set(_POLICY) | {
        "artifact_sha256"
    }:
        raise ValueError("unvalidated policy fields are invalid")
    _reject_forbidden_fields(raw)
    supplied = _require_sha256(raw.get("result_sha256"), label="unvalidated result")
    payload = {key: child for key, child in raw.items() if key != "result_sha256"}
    if supplied != _domain_sha256(_RESULT_DOMAIN, payload):
        raise ValueError("unvalidated result hash mismatch")


def _prevalidation_checkpoint_path(final_output: Path | str) -> Path:
    destination = Path(final_output)
    if destination.suffix != ".json" or not destination.name:
        raise ValueError("progressive output must have a JSON basename")
    return destination.with_name(
        f"{destination.stem}.prevalidation-checkpoint.json"
    )


def _build_prevalidation_checkpoint(
    result: Mapping[str, object], *, final_output: Path
) -> dict[str, object]:
    _validate_unvalidated_result_shape(result)
    if final_output.suffix != ".json" or final_output.name != str(final_output.name):
        raise ValueError("checkpoint final output must have a JSON basename")
    without_digest: dict[str, object] = {
        "schema": _CHECKPOINT_SCHEMA,
        "format_version": 1,
        "status": "unvalidated",
        "intended_final_output_file": final_output.name,
        "unvalidated_result": dict(result),
        "unvalidated_result_sha256": _domain_sha256(
            _CHECKPOINT_RESULT_DOMAIN, result
        ),
        "safety": dict(_CHECKPOINT_SAFETY),
    }
    return {
        **without_digest,
        "checkpoint_sha256": _domain_sha256(_CHECKPOINT_DOMAIN, without_digest),
    }


def _validate_prevalidation_checkpoint(
    value: Mapping[str, object], *, source_path: Path | None = None
) -> dict[str, object]:
    if set(value) != _CHECKPOINT_FIELDS:
        raise ValueError("prevalidation checkpoint fields are invalid")
    if (
        value.get("schema") != _CHECKPOINT_SCHEMA
        or value.get("format_version") != 1
        or value.get("status") != "unvalidated"
        or value.get("safety") != _CHECKPOINT_SAFETY
    ):
        raise ValueError("prevalidation checkpoint header/safety is invalid")
    intended = value.get("intended_final_output_file")
    if (
        not isinstance(intended, str)
        or Path(intended).name != intended
        or Path(intended).suffix != ".json"
    ):
        raise ValueError("checkpoint intended output is invalid")
    if source_path is not None:
        intended_path = source_path.with_name(intended)
        if source_path != _prevalidation_checkpoint_path(intended_path):
            raise ValueError("checkpoint path differs from intended output")
    result = value.get("unvalidated_result")
    if not isinstance(result, Mapping):
        raise TypeError("checkpoint unvalidated result is unavailable")
    _validate_unvalidated_result_shape(result)
    if value.get("unvalidated_result_sha256") != _domain_sha256(
        _CHECKPOINT_RESULT_DOMAIN, result
    ):
        raise ValueError("checkpoint unvalidated-result hash mismatch")
    without_digest = {
        key: child for key, child in value.items() if key != "checkpoint_sha256"
    }
    if value.get("checkpoint_sha256") != _domain_sha256(
        _CHECKPOINT_DOMAIN, without_digest
    ):
        raise ValueError("prevalidation checkpoint hash mismatch")
    return dict(value)


def load_gemma3_l10_l17_open_a_prevalidation_checkpoint(
    path: Path | str,
) -> dict[str, object]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prevalidation checkpoint is not strict JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("prevalidation checkpoint must contain one JSON object")
    return _validate_prevalidation_checkpoint(raw, source_path=source)


def _publish_with_prevalidation_checkpoint(
    result: Mapping[str, object], *, output: Path | str
) -> dict[str, object]:
    destination = Path(output)
    checkpoint_path = _prevalidation_checkpoint_path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination.name}")
    if checkpoint_path.exists():
        raise FileExistsError(
            f"surviving prevalidation checkpoint requires recovery: "
            f"{checkpoint_path.name}"
        )
    checkpoint = _build_prevalidation_checkpoint(
        result, final_output=destination
    )
    _write_exclusive_atomic(checkpoint_path, checkpoint)
    try:
        validated = validate_gemma3_l10_l17_open_a_progressive_result(result)
        _write_exclusive_atomic(destination, validated)
    except BaseException:
        # Preserve the source-safe scalar envelope across validator defects.
        raise
    checkpoint_path.unlink()
    _fsync_directory(checkpoint_path.parent)
    return validated


def finalize_gemma3_l10_l17_open_a_prevalidation_checkpoint(
    checkpoint_path: Path | str,
    *,
    output: Path | str | None = None,
) -> dict[str, object]:
    """Validate and publish a surviving checkpoint without model/prompt I/O."""

    source = Path(checkpoint_path)
    checkpoint = load_gemma3_l10_l17_open_a_prevalidation_checkpoint(source)
    intended = source.with_name(str(checkpoint["intended_final_output_file"]))
    destination = intended if output is None else Path(output)
    if destination != intended:
        raise ValueError("recovery output path differs from checkpoint")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination.name}")
    result = checkpoint["unvalidated_result"]
    assert isinstance(result, Mapping)
    validated = validate_gemma3_l10_l17_open_a_progressive_result(result)
    _write_exclusive_atomic(destination, validated)
    source.unlink()
    _fsync_directory(source.parent)
    return validated


def _validate_authorization(
    authority: Mapping[str, object], corpus: Mapping[str, object]
) -> None:
    if set(authority) != _AUTHORIZATION_FIELDS:
        raise ValueError("authorization fields are invalid")
    bundle = authority.get("bundle")
    prior = authority.get("prior_selection_binding")
    if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_BINDING_FIELDS:
        raise ValueError("authorization bundle binding is invalid")
    if not isinstance(prior, Mapping) or set(prior) != _CORPUS_FIELDS:
        raise ValueError("authorization prior-selection binding is invalid")
    if not _canonical_equal(prior, corpus):
        raise ValueError("current selection differs from prior adaptive binding")
    bundle_authority_fields = {
        "bundle_file_sha256": "composition_bundle_file_sha256",
        "composition_payload_sha256": "composition_payload_sha256",
        "combined_edgeless_graph_sha256": "combined_edgeless_graph_sha256",
        "combined_primary_graph_sha256": "combined_primary_graph_sha256",
    }
    for bundle_field, expected_field in bundle_authority_fields.items():
        if bundle.get(bundle_field) != _EXPECTED_AUTHORITIES[expected_field]:
            raise ValueError(
                f"authorization {bundle_field} differs from frozen policy"
            )
    for key in (
        "layer10_candidate_tensor_file_sha256",
        "layer10_candidate_scientific_payload_sha256",
        "layer10_guard_evidence_file_sha256",
        "layer10_guard_evidence_logical_sha256",
        "layer17_candidate_tensor_file_sha256",
        "layer17_candidate_scientific_payload_sha256",
        "layer17_edgeless_graph_sha256",
    ):
        if bundle.get(key) != _EXPECTED_AUTHORITIES[key]:
            raise ValueError(f"authorization {key} differs from frozen policy")
    for key in (
        "layer17_lofo_report_file_sha256",
        "layer17_lofo_report_sha256",
        "layer17_adaptive_result_file_sha256",
        "layer17_adaptive_result_sha256",
    ):
        if authority.get(key) != _EXPECTED_AUTHORITIES[key]:
            raise ValueError(f"authorization {key} differs from frozen policy")
    if (
        bundle.get("layer17_adaptive_evidence_file_sha256")
        != _EXPECTED_AUTHORITIES["layer17_adaptive_result_file_sha256"]
        or bundle.get("layer17_adaptive_evidence_logical_sha256")
        != _EXPECTED_AUTHORITIES["layer17_adaptive_result_sha256"]
    ):
        raise ValueError("bundle layer17 adaptive evidence differs from policy")
    for key in (
        "composition_payload_sha256",
        "combined_edgeless_graph_sha256",
        "combined_primary_graph_sha256",
        "model_fingerprint",
        "parameter_cluster_plan_sha256",
        "bundle_file_sha256",
    ):
        _require_sha256(bundle.get(key), label=f"bundle {key}")
    if bundle.get("resources") != _EXPECTED_RESOURCES:
        raise ValueError("authorization bundle resources differ")
    if (
        authority.get("authorization_kind")
        != "frozen_layer10_guard_plus_passing_layer17_lofo_adaptive_selection"
        or authority.get("authorization_completed_before_selection_open") is not True
        or authority.get("selection_access_authorized") is not True
        or authority.get("claim_role")
        != "already_open_adaptive_development_selection"
        or any(
            authority.get(field) is not False
            for field in (
                "fit_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
                "heldout_confirmation",
                "serving_authorized",
                "full_model_compiled",
            )
        )
        or authority.get("source_safe") is not True
    ):
        raise ValueError("authorization scientific boundary is invalid")
    without_digest = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    if authority.get("authority_sha256") != _domain_sha256(
        _AUTHORITY_DOMAIN, without_digest
    ):
        raise ValueError("authorization hash mismatch")


def validate_gemma3_l10_l17_open_a_progressive_result(
    value: Mapping[str, object] | Path | str,
) -> dict[str, object]:
    """Strictly validate a source-safe progressive result or JSON path."""

    if isinstance(value, (str, Path)):
        try:
            raw = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("progressive result is not strict JSON") from error
        if not isinstance(raw, dict):
            raise TypeError("progressive result must contain one JSON object")
    else:
        raw = dict(value)
    if set(raw) != _ROOT_FIELDS:
        raise ValueError("progressive result fields are invalid")
    if raw.get("schema") != _SCHEMA or raw.get("format_version") != _FORMAT_VERSION:
        raise ValueError("progressive result header is invalid")
    if raw.get("safety") != _SAFETY:
        raise ValueError("progressive result safety flags are invalid")
    if (
        raw.get("scientific_role") != "adaptive_open_development_composition"
        or raw.get("heldout_confirmation") is not False
        or raw.get("selection_opened") is not True
        or any(
            raw.get(field) is not False
            for field in (
                "fit_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
                "bundle_changed",
                "evidence_changed",
                "full_model_compiled",
                "serving_authorized",
                "latency_or_kernel_speed_claim",
            )
        )
        or raw.get("full_model_logits_scored") is not True
    ):
        raise ValueError("progressive result scientific boundary is invalid")
    corpus = raw.get("corpus")
    tokenization = raw.get("tokenization")
    runtime = raw.get("runtime")
    assessment = raw.get("assessment")
    decision = raw.get("decision")
    authorization = raw.get("authorization")
    if not isinstance(corpus, Mapping) or set(corpus) != _CORPUS_FIELDS:
        raise ValueError("progressive corpus binding fields are invalid")
    if not isinstance(tokenization, Mapping) or set(tokenization) != _TOKENIZATION_FIELDS:
        raise ValueError("progressive tokenization fields are invalid")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("progressive runtime fields are invalid")
    if not isinstance(assessment, Mapping) or set(assessment) != _ASSESSMENT_FIELDS:
        raise ValueError("progressive assessment fields are invalid")
    if not isinstance(decision, Mapping) or not isinstance(authorization, Mapping):
        raise TypeError("progressive decision/authorization are unavailable")
    _validate_authorization(authorization, corpus)
    authorization_bundle = authorization["bundle"]
    assert isinstance(authorization_bundle, Mapping)
    if (
        corpus.get("assessment_role") != "already_open_calibration_a_selection"
        or corpus.get("example_count") != _EXPECTED_EXAMPLES
        or corpus.get("family_count") != _EXPECTED_FAMILIES
        or tokenization.get("example_count") != _EXPECTED_EXAMPLES
        or tokenization.get("family_stream_count") != _EXPECTED_FAMILIES
        or tokenization.get("contains_prompt_text") is not False
        or tokenization.get("contains_prompt_identities") is not False
        or tokenization.get("contains_token_ids") is not False
        or tokenization.get("logical_valid_tokens")
        != assessment.get("logical_valid_tokens")
        or tokenization.get("supervised_tokens")
        != assessment.get("supervised_tokens")
        or runtime.get("model_fingerprint")
        != authorization_bundle.get("model_fingerprint")
        or runtime.get("local_files_only") is not True
        or runtime.get("vocabulary_chunk_size") != _VOCABULARY_CHUNK_SIZE
    ):
        raise ValueError("progressive corpus/runtime/tokenization identity is invalid")
    replayed = progressive_composition_decision(assessment)
    if not _canonical_equal(decision, replayed):
        raise ValueError("progressive decision is not reproducible")
    _reject_forbidden_fields(raw)
    supplied = _require_sha256(raw.get("result_sha256"), label="result")
    payload = {key: child for key, child in raw.items() if key != "result_sha256"}
    if supplied != _domain_sha256(_RESULT_DOMAIN, payload):
        raise ValueError("progressive result hash mismatch")
    return raw


def load_gemma3_l10_l17_open_a_progressive_result(
    path: Path | str,
) -> dict[str, object]:
    return validate_gemma3_l10_l17_open_a_progressive_result(path)


def evaluate_gemma3_l10_l17_open_a_progressive(
    *,
    bundle_path: Path | str = DEFAULT_COMPOSITION_BUNDLE_PATH,
    lofo_report_path: Path | str = DEFAULT_LOFO_REPORT_PATH,
    adaptive_result_path: Path | str = DEFAULT_ADAPTIVE_RESULT_PATH,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    selection_path: Path | str = DEFAULT_SELECTION_OUTPUT,
    receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = 4,
    output: Path | str = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    """Score the frozen two-layer composition on already-open selection."""

    if type(tokenization_batch_size) is not int or tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    destination = Path(output)
    if destination.suffix != ".json" or not destination.name:
        raise ValueError("progressive output must be a JSON basename")
    checkpoint_path = _prevalidation_checkpoint_path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination.name}")
    if checkpoint_path.exists():
        raise FileExistsError(
            f"surviving prevalidation checkpoint requires recovery: "
            f"{checkpoint_path.name}"
        )

    # The following completes every executable/evidence check before the only
    # call in this module that can read selection prompt text.
    bundle, authorization, prior_selection = _authorize_before_selection(
        bundle_path=bundle_path,
        lofo_report_path=lofo_report_path,
        adaptive_result_path=adaptive_result_path,
    )
    selection = _load_open_selection_authority(
        corpus_artifact_path=corpus_artifact_path,
        selection_path=selection_path,
        receipt_path=receipt_path,
    )
    if not _canonical_equal(selection.binding, prior_selection):
        raise ValueError("selection authority differs from prior adaptive run")

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    tokenizer, model = load_gemma3(
        model_id=bundle.model_id,
        revision=bundle.requested_revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != bundle.primary.model_fingerprint:
        raise ValueError("live Gemma fingerprint differs from frozen bundle")
    family_batches, tokenization = _materialize_selection_families(
        tokenizer,
        selection,
        device=device,
        tokenization_batch_size=tokenization_batch_size,
    )
    lowerings = {
        name: lowering
        for name, lowering in zip(
            bundle.primary.traversal_order, bundle.lowerings, strict=True
        )
    }
    layer10_plan = _subgraph(
        bundle.primary, layer_ordinal=10, include_interactions=True
    )
    layer17_plan = _subgraph(
        bundle.edgeless, layer_ordinal=17, include_interactions=False
    )

    def executor(plan: ModalGeneratorGraphPlan) -> Gemma3ModalGeneratorGraphExecutor:
        return Gemma3ModalGeneratorGraphExecutor(
            adapter,
            plan,
            tuple(lowerings[name] for name in plan.traversal_order),
        )

    assessment = _score_progressive_panel(
        adapter=adapter,
        layer10_executor=executor(layer10_plan),
        layer17_executor=executor(layer17_plan),
        edgeless_executor=executor(bundle.edgeless),
        primary_executor=executor(bundle.primary),
        family_batches=family_batches,
    )
    if (
        tokenization["logical_valid_tokens"] != assessment["logical_valid_tokens"]
        or tokenization["supervised_tokens"] != assessment["supervised_tokens"]
    ):
        raise RuntimeError("tokenization and scoring totals disagree")
    decision = progressive_composition_decision(assessment)
    after = {
        "bundle_file_sha256": _file_sha256(bundle.path),
        "lofo_report_file_sha256": _file_sha256(lofo_report_path),
        "adaptive_result_file_sha256": _file_sha256(adaptive_result_path),
    }
    if (
        after["bundle_file_sha256"] != bundle.file_sha256
        or after["lofo_report_file_sha256"]
        != authorization["layer17_lofo_report_file_sha256"]
        or after["adaptive_result_file_sha256"]
        != authorization["layer17_adaptive_result_file_sha256"]
        or adapter.model_fingerprint() != bundle.primary.model_fingerprint
    ):
        raise RuntimeError("bundle, evidence, or source model changed during scoring")

    payload: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "scientific_role": "adaptive_open_development_composition",
        "heldout_confirmation": False,
        "authorization": authorization,
        "corpus": selection.binding,
        "runtime": {
            "model_id": bundle.model_id,
            "requested_revision": bundle.requested_revision,
            "model_fingerprint": bundle.primary.model_fingerprint,
            "device": str(device),
            "dtype": dtype,
            "tokenization_batch_size": tokenization_batch_size,
            "max_length": int(_tokenizer_contract()["max_length"]),
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
            "local_files_only": True,
        },
        "tokenization": tokenization,
        "assessment": assessment,
        "decision": decision,
        "bundle_changed": False,
        "evidence_changed": False,
        "selection_opened": True,
        "fit_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "serving_authorized": False,
        "latency_or_kernel_speed_claim": False,
        "safety": dict(_SAFETY),
    }
    result = {
        **payload,
        "result_sha256": _domain_sha256(_RESULT_DOMAIN, payload),
    }
    return _publish_with_prevalidation_checkpoint(result, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "score the frozen heterogeneous layer10+17 composition on the "
            "already-open v8 Calibration-A selection role"
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_COMPOSITION_BUNDLE_PATH)
    parser.add_argument("--lofo-report", type=Path, default=DEFAULT_LOFO_REPORT_PATH)
    parser.add_argument(
        "--adaptive-result", type=Path, default=DEFAULT_ADAPTIVE_RESULT_PATH
    )
    parser.add_argument("--corpus-artifact", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tokenization-batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = evaluate_gemma3_l10_l17_open_a_progressive(
        bundle_path=arguments.bundle,
        lofo_report_path=arguments.lofo_report,
        adaptive_result_path=arguments.adaptive_result,
        corpus_artifact_path=arguments.corpus_artifact,
        selection_path=arguments.selection,
        receipt_path=arguments.receipt,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        tokenization_batch_size=arguments.tokenization_batch_size,
        output=arguments.output,
    )
    decision = result["decision"]
    assert isinstance(decision, Mapping)
    print(f"result: {arguments.output}")
    print(f"all gates pass: {decision['all_required_gates_pass']}")
    print(f"next action: {decision['next_action']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
