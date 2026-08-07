"""Fit-only outer-LOFO evaluation of the Gemma L10 -> L17 A3 correction.

The runner in this module is intentionally narrower than the ordinary open-A
evaluation path.  It authenticates the frozen two-layer composition before it
opens the sole permitted role (``calibration_a_fit``), captures paired native
and frozen-Layer10 trajectories once, and performs eight train-seven/hold-one
generator refits.  Layer10, all four Layer17 decoder tensors, graph ranks, and
the graph resource envelope remain frozen.

Only scalar metrics, structural counts, and cryptographic commitments are
serialized.  Activation rows, prompts, token ids, logits, and fitted tensors
remain ephemeral.  This is adaptive fit-only evidence: it neither opens nor
authorizes selection, guard, Calibration-B, validation, test, or serving.
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
import sys
import tempfile
from types import MappingProxyType

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
    _bundle_authority,
    _record_execution,
    _subgraph,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    build_a3_raw_mlp_target,
    build_projected_correction_rows,
    fit_frozen_basis_coordinate_generators,
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_protocol import (
    FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256,
    build_default_gemma3_l10_l17_trajectory_correction_protocol,
    validate_gemma3_l10_l17_trajectory_correction_protocol,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_capped_node_fit import _validate_frozen_selection
from .gemma3_layer17_family_lofo_authority import (
    Gemma3Layer17FamilyLOFOAuthority,
    load_gemma3_layer17_family_lofo_authority,
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_authority_metadata,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
)
from .gemma3_layer17_trajectory_row_capture import (
    GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_FORMAT_VERSION,
    GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_SCHEMA,
    Gemma3Layer17TrajectoryRowPair,
    capture_gemma3_layer17_native_and_layer10_rows,
)
from .gemma3_layer17_v8_fit_lofo import (
    _authority_metadata,
    _blocks_to_device,
    _family_blocks,
    _load_authenticated_protocol,
    _ordered_restored_lowerings,
    _reject_forbidden_output_fields as _reject_lofo_authority_fields,
)
from .gemma3_layer17_open_a_capacity_evaluation import (
    _add_comparison,
    _add_native,
    _candidate_comparison,
    _finalize_metric_accumulator,
    _model_logits,
    _native_nll,
    _new_metric_accumulator,
    _selected_logits_and_targets,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
    _row_key_sha256,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    select_top_fisher_same_layer_fragments,
)
from .modal_generator_graph import ModalGeneratorGraphPlan
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_graph_rung_evaluation import (
    _GRAPH_LOGICAL_FIELDS,
    _GRAPH_STATIC_FIELDS,
    _execution_fields,
    _validate_graph_execution,
)


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_OUTPUT",
    "GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_FORMAT_VERSION",
    "GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_SCHEMA",
    "aggregate_trajectory_correction_lofo_folds",
    "build_trajectory_correction_fold_rows",
    "build_trajectory_correction_lofo_report",
    "evaluate_trajectory_correction_lofo_gates",
    "load_gemma3_l10_l17_trajectory_correction_lofo_report",
    "run_gemma3_l10_l17_trajectory_correction_lofo",
    "save_gemma3_l10_l17_trajectory_correction_lofo_report",
    "score_trajectory_correction_fold",
    "validate_gemma3_l10_l17_trajectory_correction_lofo_report",
]


GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_trajectory_correction_lofo"
)
GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_FORMAT_VERSION = 1

_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_OUTPUT = (
    _LOCAL_ROOT / "layer10-layer17-a3-trajectory-correction-a-fit-lofo-v1.json"
)
_EXPECTED_FAMILIES = 8
_EXPECTED_EXAMPLES = 256
_GENERATOR_RANK = 16
_RIDGE = 0.0
_VOCABULARY_CHUNK_SIZE = 16384
_MINIMUM_NUMERICAL_KL_PER_TOKEN = -1.0e-12
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TRAJECTORY_PROTOCOL_SHA256 = (
    "ab3794c3cf6660738db6b24c66db02383a72d932e0b540462cec8fa41aff55e3"
)
_EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256 = (
    "84b80b3cbabc3b8ff8bcf9f63e1f97a620fb38e2f6940b196f30906d9dfcb1b7"
)
_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a3-lofo-report:v1\0"
_SPLIT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a3-lofo-split:v1\0"
_CAPTURE_METADATA_DOMAIN = (
    b"fisher-graph:gemma3-layer17-native-layer10-trajectory-rows:v2\0"
)
_RANDOMNESS_RECIPE_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a3-lofo-randomness:v1\0"
)
_SOURCE_RUNTIME_CATALOG_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a3-lofo-source-runtime-catalog:v1\0"
)
_COMPOSITION_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l10-l17-a3-lofo-composition-receipt:v1\0"
)
_RANDOM_SEED = 170_117
_RANDOMNESS_RECIPE = {
    "recipe_id": "gemma3_l10_l17_a3_lofo_deterministic_v1",
    "torch_seed": _RANDOM_SEED,
    "seed_applied_before_model_load_and_fitting": True,
    "fixed_family_fold_order": True,
    "fixed_rank_generator_fit": True,
    "stochastic_rank_or_arm_selection": False,
    "stochastic_scoring": False,
}

# ``native`` is stored separately.  This order exactly follows the frozen
# protocol and is used for every metric accumulator and report validator.
_CONDITIONS = (
    "layer10_only",
    "trajectory_corrected_layer17_only",
    "frozen_uncorrected_composition",
    "trajectory_corrected_composition",
    "matched_double_deletion",
)

_EXPECTED_EXACT_RESOURCES = {
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
}

_EXPECTED_CONDITION_RESOURCES = {
    "layer10_only": {
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
    "trajectory_corrected_layer17_only": {
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
    "frozen_uncorrected_composition": {
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
    "trajectory_corrected_composition": {
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

_COMPOSITION_REPLACEMENT_POLICY = {
    "operation": "replace_exact_layer_nodes_in_authenticated_composition",
    "replaced_layer_ordinal": 17,
    "preserved_layer_ordinals": [10],
    "layer10_graph_and_lowerings_unchanged": True,
    "layer17_decoder_bases_and_means_unchanged": True,
    "layer17_generators_refit_per_fold": True,
    "interactions_preserved_exactly": True,
    "traversal_order_preserved_exactly": True,
    "lowering_merge_policy": (
        "frozen_layer10_then_fold_refit_layer17_in_result_traversal_order"
    ),
}

_PROJECTION_METADATA_FIELDS = {
    "projection_method",
    "node_order",
    "basis_sha256_by_node",
    "mean_bias_sha256_by_node",
    "decoder_basis_sha256_by_node",
    "affine_offset_sha256",
    "combined_basis_rank",
    "observation_count",
    "residual_width",
    "target_sha256",
    "prediction_sha256",
    "coordinate_sha256_by_node",
    "contribution_sha256_by_node",
    "rmse",
    "target_rms",
    "nrmse",
    "max_abs_error",
    "offline_projection_only",
    "runtime_parameter_count",
    "runtime_macs_per_token",
}

_AUTHORIZATION_FIELDS = {
    "authorization_kind",
    "authorization_completed_before_fit_open",
    "protocol_sha256",
    "bundle",
    "source_runtime_catalog",
    "fit_authority",
    "fit_authority_sha256",
    "fit_opened",
    "selection_opened",
    "guard_opened",
    "calibration_b_opened",
    "validation_opened",
    "test_opened",
    "heldout_confirmation",
    "serving_authorized",
    "source_safe",
}

_RUNTIME_FIELDS = {
    "model_id",
    "requested_revision",
    "model_fingerprint",
    "device",
    "dtype",
    "local_files_only",
    "vocabulary_chunk_size",
    "randomness",
}

_FIT_COLLECTION_FIELDS = {
    "materialization",
    "authenticated_stream_sha256",
    "capture",
    "compiled_keep_replay_audit",
    "capture_count",
    "captured_examples",
    "captured_sequences",
    "captured_observations",
    "model_rows_recollected_per_fold",
    "a3_target_construction",
    "decoder_span_sha256",
    "summed_mean_sha256",
}

_ROW_RECEIPT_FIELDS = {
    "capture_sha256",
    "protocol_fold_sha256",
    "authenticated_stream_sha256",
    "fit_split_sha256",
    "held_split_sha256",
    "fit_family_aliases",
    "held_family_alias",
    "fit_row_key_sha256",
    "held_row_key_sha256",
    "fit_observations",
    "held_observations",
    "fit_sequences",
    "held_sequences",
    "held_family_excluded_from_projection_fit_and_generator_fit",
    "held_fisher_weights_preserved_without_held_family_normalization",
    "held_projection_used_only_as_fixed_basis_generator_evaluation_target",
    "fit_projection",
    "held_projection",
}

_CORRECTION_FIT_FIELDS = {
    "graph_sha256",
    "node_order",
    "parameter_count",
    "macs_per_token",
    "interaction_count",
    "source_mean_bias_sha256_by_node",
    "source_decoder_basis_sha256_by_node",
    "lowering_sha256_by_node",
    "generator_plan_sha256_by_node",
}

_REPORT_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "heldout_confirmation",
    "protocol",
    "authorization",
    "runtime",
    "fit_collection",
    "folds",
    "aggregate",
    "resources",
    "decision",
    "source_model_unchanged",
    "layer10_unchanged",
    "selection_opened",
    "guard_opened",
    "calibration_b_opened",
    "validation_opened",
    "test_opened",
    "full_model_logits_scored",
    "full_model_compiled",
    "serving_authorized",
    "latency_or_kernel_speed_claim",
    "safety",
    "report_sha256",
}


def _progress(message: str) -> None:
    print(f"[l10-l17-a3-lofo] {message}", file=sys.stderr, flush=True)


def _canonical_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("strict JSON cannot contain non-finite numbers")
        return value
    if isinstance(value, Tensor):
        raise TypeError("source-safe reports cannot contain tensors")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("strict JSON mappings require string keys")
        return {
            key: _canonical_json_value(child)
            for key, child in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_canonical_json_value(child) for child in value]
    raise TypeError(f"value of type {type(value).__name__} is not strict JSON")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _capture_metadata_sha256(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping) or "capture_sha256" not in value:
        raise ValueError("trajectory capture metadata is incomplete")
    payload = {key: child for key, child in value.items() if key != "capture_sha256"}
    return hashlib.sha256(
        _CAPTURE_METADATA_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()


def _randomness_recipe_receipt(device: torch.device) -> dict[str, object]:
    if not isinstance(device, torch.device):
        raise TypeError("randomness recipe device must be torch.device")
    torch.manual_seed(_RANDOM_SEED)
    cuda_seed_applied = device.type == "cuda"
    if cuda_seed_applied:
        torch.cuda.manual_seed_all(_RANDOM_SEED)
    return {
        **_RANDOMNESS_RECIPE,
        "recipe_sha256": _domain_sha256(
            _RANDOMNESS_RECIPE_DOMAIN,
            _RANDOMNESS_RECIPE,
        ),
        "torch_manual_seed_applied": True,
        "torch_cuda_manual_seed_all_applied": cuda_seed_applied,
    }


def _forbidden_output_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized.startswith("contains_"):
        return False
    return normalized in {
        "prompt",
        "prompts",
        "prompt_text",
        "prompt_texts",
        "example_id",
        "example_ids",
        "input_ids",
        "token_id",
        "token_ids",
        "tokens",
        "logit",
        "logits",
        "activation",
        "activations",
        "gradient",
        "gradients",
        "row_key",
        "row_keys",
        "weights",
        "state_dict",
        "coordinates",
        "contributions",
        "targets",
    }


def _reject_forbidden_output_fields(value: object, *, path: str = "report") -> None:
    if isinstance(value, Tensor):
        raise ValueError(f"{path} contains a Tensor")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            if _forbidden_output_key(key):
                raise ValueError(f"{path}.{key} is a forbidden source field")
            _reject_forbidden_output_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_forbidden_output_fields(child, path=f"{path}[{index}]")


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _fit_access_binding(protocol: Mapping[str, object]) -> Mapping[str, object]:
    source = protocol.get("source_authority")
    if not isinstance(source, Mapping):
        raise TypeError("trajectory protocol source authority is unavailable")
    fit = source.get("calibration_a_fit")
    if not isinstance(fit, Mapping):
        raise TypeError("trajectory protocol A-fit authority is unavailable")
    return fit


def _validate_bundle_against_protocol(
    bundle: object,
    protocol: Mapping[str, object],
) -> None:
    source = protocol.get("source_authority")
    candidate = protocol.get("candidate_contract")
    if not isinstance(source, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("trajectory protocol source/candidate contract is missing")
    composition = source.get("composition_bundle")
    layer10 = source.get("layer10")
    layer17 = source.get("layer17_decoder_source")
    binding = getattr(bundle, "binding", None)
    primary = getattr(bundle, "primary", None)
    edgeless = getattr(bundle, "edgeless", None)
    if not all(
        isinstance(value, Mapping)
        for value in (composition, layer10, layer17, binding)
    ) or not isinstance(primary, ModalGeneratorGraphPlan) or not isinstance(
        edgeless, ModalGeneratorGraphPlan
    ):
        raise TypeError("authenticated composition binding is incomplete")
    assert isinstance(composition, Mapping)
    assert isinstance(layer10, Mapping)
    assert isinstance(layer17, Mapping)
    assert isinstance(binding, Mapping)
    if (
        binding.get("bundle_file_sha256") != composition.get("tensor_file_sha256")
        or binding.get("composition_payload_sha256")
        != composition.get("composition_payload_sha256")
        or primary.artifact_sha256
        != composition.get("combined_primary_graph_sha256")
        or edgeless.artifact_sha256
        != composition.get("combined_edgeless_graph_sha256")
        or binding.get("layer10_candidate_tensor_file_sha256")
        != layer10.get("candidate_tensor_file_sha256")
        or binding.get("layer10_candidate_scientific_payload_sha256")
        != layer10.get("candidate_scientific_payload_sha256")
        or binding.get("layer17_candidate_tensor_file_sha256")
        != layer17.get("candidate_tensor_file_sha256")
        or binding.get("layer17_candidate_scientific_payload_sha256")
        != layer17.get("candidate_scientific_payload_sha256")
        or binding.get("layer17_edgeless_graph_sha256")
        != layer17.get("source_edgeless_graph_sha256")
    ):
        raise ValueError("composition bundle differs from trajectory protocol")
    exact = candidate.get("exact_resources")
    if _canonical_json_bytes(exact) != _canonical_json_bytes(
        _EXPECTED_EXACT_RESOURCES
    ):
        raise ValueError("trajectory protocol exact resources drifted")


def _authenticate_before_fit_access(
    *,
    bundle_path: Path | str,
    corpus_receipt_path: Path | str,
    corpus_artifact_path: Path | str,
    fit_input_path: Path | str,
) -> tuple[object, Gemma3Layer17FamilyLOFOAuthority, dict[str, object], dict[str, object]]:
    """Authenticate the frozen executable before the only role-opening call."""

    protocol = validate_gemma3_l10_l17_trajectory_correction_protocol(
        build_default_gemma3_l10_l17_trajectory_correction_protocol()
    )
    if protocol.get("artifact_sha256") != (
        FROZEN_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_PROTOCOL_SHA256
    ) or protocol.get("artifact_sha256") != _EXPECTED_TRAJECTORY_PROTOCOL_SHA256:
        raise ValueError("trajectory correction protocol identity is not frozen")
    evaluation_contract = protocol.get("evaluation_contract")
    if not isinstance(evaluation_contract, Mapping) or tuple(
        evaluation_contract.get("conditions", ())
    ) != ("native", *_CONDITIONS):
        raise ValueError("trajectory correction condition catalog drifted")

    # Ordering is a safety property: `_bundle_authority` must complete before
    # `load_gemma3_layer17_family_lofo_authority` can open A-fit.
    bundle = _bundle_authority(bundle_path)
    _validate_bundle_against_protocol(bundle, protocol)

    authority = load_gemma3_layer17_family_lofo_authority(
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    authority_sha256, authority_safe = _authority_metadata(authority)
    validate_gemma3_layer17_family_lofo_authority_metadata(authority_safe)
    v8_protocol = _load_authenticated_protocol(
        corpus_artifact_path,
        authority_metadata=authority_safe,
    )
    fit_binding = _fit_access_binding(protocol)
    if (
        v8_protocol.get("artifact_sha256")
        != FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
        or fit_binding.get("source_lofo_protocol_sha256")
        != v8_protocol.get("artifact_sha256")
        or fit_binding.get("fit_manifest_sha256")
        != authority_safe["corpus"]["fit_manifest_sha256"]  # type: ignore[index]
        or fit_binding.get("fit_source_file_sha256")
        != authority_safe["corpus"]["fit_role_file_sha256"]  # type: ignore[index]
        or fit_binding.get("family_aliases")
        != list(V8_FAMILY_LOFO_FAMILY_ALIASES)
        or fit_binding.get("exclusive_role_use") != "calibration_a_fit"
    ):
        raise ValueError("A-fit authority differs from trajectory protocol")
    authorization = {
        "authorization_kind": "frozen_composition_then_fit_only_family_lofo",
        "authorization_completed_before_fit_open": True,
        "protocol_sha256": protocol["artifact_sha256"],
        "bundle": dict(getattr(bundle, "binding")),
        "fit_authority": authority_safe,
        "fit_authority_sha256": authority_sha256,
        "fit_opened": True,
        "selection_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "source_safe": True,
    }
    _reject_forbidden_output_fields(authorization, path="authorization")
    return bundle, authority, protocol, authorization


def _source_lowering_maps(
    bundle: object,
) -> tuple[
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
    dict[str, ModalGeneratorLowering],
]:
    primary = getattr(bundle, "primary", None)
    lowerings = getattr(bundle, "lowerings", None)
    if not isinstance(primary, ModalGeneratorGraphPlan) or not isinstance(
        lowerings, tuple
    ):
        raise TypeError("composition runtime is unavailable")
    if len(lowerings) != len(primary.nodes) or any(
        not isinstance(value, ModalGeneratorLowering) for value in lowerings
    ):
        raise ValueError("composition lowering catalog is invalid")
    by_name = {
        node.name: lowering
        for node, lowering in zip(primary.nodes, lowerings, strict=True)
    }
    layer10 = _subgraph(primary, layer_ordinal=10, include_interactions=True)
    layer17 = _subgraph(primary, layer_ordinal=17, include_interactions=False)
    l10_lowerings = {name: by_name[name] for name in layer10.traversal_order}
    l17_lowerings = {name: by_name[name] for name in layer17.traversal_order}
    return layer10, layer17, l10_lowerings, l17_lowerings


def _selected_fragment_id(lowering: ModalGeneratorLowering) -> str:
    matches = tuple(
        fragment.fragment_id
        for fragment in lowering.fragment_plan.fragments
        if fragment.artifact_sha256 == lowering.selected_fragment_sha256
    )
    if len(matches) != 1:
        raise ValueError("lowering does not bind exactly one fragment")
    return matches[0]


def _validate_source_decoder_contract(
    source_graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    projection = protocol.get("projection_contract")
    candidate = protocol.get("candidate_contract")
    if not isinstance(projection, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("projection/candidate contracts are unavailable")
    raw_records = projection.get("ordered_decoders")
    if isinstance(raw_records, (str, bytes)) or not isinstance(
        raw_records, Sequence
    ):
        raise TypeError("ordered decoder contract is unavailable")
    records = tuple(raw_records)
    if len(records) != len(source_graph.nodes) or set(lowerings_by_node) != set(
        source_graph.traversal_order
    ):
        raise ValueError("source Layer17 graph/lowerings differ")
    bases: dict[str, object] = {}
    fragments: dict[str, str] = {}
    for node, raw in zip(source_graph.nodes, records, strict=True):
        if not isinstance(raw, Mapping):
            raise TypeError("decoder record must be a mapping")
        lowering = lowerings_by_node[node.name]
        basis = lowering.computational_mode_basis
        basis.validate_integrity()
        fragment_id = _selected_fragment_id(lowering)
        if (
            raw.get("node_name") != node.name
            or raw.get("fragment_id") != fragment_id
            or raw.get("node_rank") != basis.rank
            or raw.get("computational_mode_basis_sha256")
            != basis.artifact_sha256
            or raw.get("mean_bias_sha256") != basis.mean_bias_sha256
            or raw.get("decoder_basis_sha256")
            != basis.decoder_basis_sha256
            or int(raw.get("coordinate_stop", -1))
            - int(raw.get("coordinate_start", -1))
            != basis.rank
        ):
            raise ValueError("frozen Layer17 decoder contract drifted")
        bases[node.name] = basis
        fragments[node.name] = fragment_id
    if (
        tuple(fragments.values())
        != tuple(candidate.get("layer17_fragment_ids_in_execution_order", ()))
        or tuple(node.latent_width for node in source_graph.nodes)
        != tuple(candidate.get("layer17_node_ranks_in_execution_order", ()))
        or source_graph.parameter_count != 163_094
        or source_graph.macs_per_token != 160_352
        or source_graph.interactions
    ):
        raise ValueError("fixed Layer17 topology/resources drifted")
    return bases, fragments


def _lowering_sha256_catalog(
    graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
) -> dict[str, str]:
    graph.validate_integrity()
    if set(lowerings_by_node) != set(graph.traversal_order):
        raise ValueError("runtime lowering catalog differs from graph")
    result: dict[str, str] = {}
    for name in graph.traversal_order:
        lowering = lowerings_by_node[name]
        if not isinstance(lowering, ModalGeneratorLowering):
            raise TypeError("runtime lowering catalog contains a non-lowering")
        result[name] = _require_sha256(
            lowering.artifact_sha256,
            label=f"{name} lowering",
        )
    return result


def _build_source_runtime_catalog(
    *,
    bundle_binding: Mapping[str, object],
    primary_graph: ModalGeneratorGraphPlan,
    layer10_graph: ModalGeneratorGraphPlan,
    layer17_graph: ModalGeneratorGraphPlan,
    layer10_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    selection: SameLayerFragmentSelection,
) -> dict[str, object]:
    """Commit the exact authenticated graph/lowering catalog used at runtime."""

    for graph in (primary_graph, layer10_graph, layer17_graph):
        graph.validate_integrity()
    if tuple(primary_graph.traversal_order) != (
        *layer10_graph.traversal_order,
        *layer17_graph.traversal_order,
    ):
        raise ValueError("source graph layer traversal order is not compositional")
    layer10_lowerings = _lowering_sha256_catalog(
        layer10_graph,
        layer10_lowerings_by_node,
    )
    layer17_lowerings = _lowering_sha256_catalog(
        layer17_graph,
        layer17_lowerings_by_node,
    )
    selection_sha256 = _require_sha256(
        selection.artifact_sha256,
        label="Layer17 selection",
    )
    fragment_ids = tuple(selection.fragment_ids)
    if (
        not fragment_ids
        or len(fragment_ids) != len(set(fragment_ids))
        or len(fragment_ids) != len(layer17_graph.traversal_order)
    ):
        raise ValueError("Layer17 runtime fragment catalog is invalid")
    payload = {
        "authenticated_bundle_file_sha256": _require_sha256(
            bundle_binding.get("bundle_file_sha256"),
            label="composition bundle",
        ),
        "frozen_primary_graph_sha256": _require_sha256(
            primary_graph.artifact_sha256,
            label="frozen primary graph",
        ),
        "frozen_primary_traversal_order": tuple(primary_graph.traversal_order),
        "layer10": {
            "graph_sha256": _require_sha256(
                layer10_graph.artifact_sha256,
                label="Layer10 graph",
            ),
            "traversal_order": tuple(layer10_graph.traversal_order),
            "lowering_sha256_by_node": layer10_lowerings,
        },
        "layer17": {
            "graph_sha256": _require_sha256(
                layer17_graph.artifact_sha256,
                label="Layer17 graph",
            ),
            "traversal_order": tuple(layer17_graph.traversal_order),
            "lowering_sha256_by_node": layer17_lowerings,
            "fragment_ids": fragment_ids,
            "selection_sha256": selection_sha256,
        },
    }
    return {
        **payload,
        "catalog_sha256": _domain_sha256(
            _SOURCE_RUNTIME_CATALOG_DOMAIN,
            payload,
        ),
    }


def _validate_source_runtime_catalog(
    value: object,
    *,
    protocol: Mapping[str, object],
    bundle_binding: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "authenticated_bundle_file_sha256",
        "frozen_primary_graph_sha256",
        "frozen_primary_traversal_order",
        "layer10",
        "layer17",
        "catalog_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("source runtime catalog fields are invalid")
    layer10 = value.get("layer10")
    layer17 = value.get("layer17")
    layer_fields = {
        "graph_sha256",
        "traversal_order",
        "lowering_sha256_by_node",
    }
    if (
        not isinstance(layer10, Mapping)
        or set(layer10) != layer_fields
        or not isinstance(layer17, Mapping)
        or set(layer17) != {
            *layer_fields,
            "fragment_ids",
            "selection_sha256",
        }
    ):
        raise ValueError("source runtime layer catalogs are invalid")
    source = protocol.get("source_authority")
    projection = protocol.get("projection_contract")
    candidate = protocol.get("candidate_contract")
    if not all(isinstance(item, Mapping) for item in (source, projection, candidate)):
        raise TypeError("source runtime protocol contracts are unavailable")
    assert isinstance(source, Mapping)
    assert isinstance(projection, Mapping)
    assert isinstance(candidate, Mapping)
    composition = source.get("composition_bundle")
    layer10_contract = source.get("layer10")
    layer17_contract = source.get("layer17_decoder_source")
    records = projection.get("ordered_decoders")
    if (
        not isinstance(composition, Mapping)
        or not isinstance(layer10_contract, Mapping)
        or not isinstance(layer17_contract, Mapping)
        or isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
    ):
        raise TypeError("source runtime frozen contracts are unavailable")
    expected_layer17_order = tuple(
        row.get("node_name") for row in records if isinstance(row, Mapping)
    )
    expected_fragments = tuple(
        candidate.get("layer17_fragment_ids_in_execution_order", ())
    )
    layer10_order = tuple(layer10.get("traversal_order", ()))
    layer17_order = tuple(layer17.get("traversal_order", ()))
    primary_order = tuple(value.get("frozen_primary_traversal_order", ()))
    layer10_lowerings = layer10.get("lowering_sha256_by_node")
    layer17_lowerings = layer17.get("lowering_sha256_by_node")
    if (
        not layer10_order
        or len(layer10_order) != len(set(layer10_order))
        or layer17_order != expected_layer17_order
        or not expected_layer17_order
        or primary_order != (*layer10_order, *layer17_order)
        or len(primary_order) != int(_EXPECTED_EXACT_RESOURCES["graph_node_count"])
        or not isinstance(layer10_lowerings, Mapping)
        or set(layer10_lowerings) != set(layer10_order)
        or not isinstance(layer17_lowerings, Mapping)
        or set(layer17_lowerings) != set(layer17_order)
        or any(
            _SHA256.fullmatch(str(digest)) is None
            for digest in (*layer10_lowerings.values(), *layer17_lowerings.values())
        )
        or tuple(layer17.get("fragment_ids", ())) != expected_fragments
        or _SHA256.fullmatch(str(layer17.get("selection_sha256"))) is None
        or value.get("authenticated_bundle_file_sha256")
        != bundle_binding.get("bundle_file_sha256")
        or value.get("authenticated_bundle_file_sha256")
        != composition.get("tensor_file_sha256")
        or value.get("frozen_primary_graph_sha256")
        != composition.get("combined_primary_graph_sha256")
        or value.get("frozen_primary_graph_sha256")
        != bundle_binding.get("combined_primary_graph_sha256")
        or layer10.get("graph_sha256")
        != layer10_contract.get("primary_graph_sha256")
        or layer17.get("graph_sha256")
        != layer17_contract.get("source_edgeless_graph_sha256")
    ):
        raise ValueError("source runtime catalog differs from frozen authority")
    supplied_sha256 = _require_sha256(
        value.get("catalog_sha256"),
        label="source runtime catalog",
    )
    payload = {key: value[key] for key in fields if key != "catalog_sha256"}
    if (
        supplied_sha256
        != _domain_sha256(_SOURCE_RUNTIME_CATALOG_DOMAIN, payload)
        or supplied_sha256 != _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256
    ):
        raise ValueError("source runtime catalog hash mismatch")
    return value


def _build_corrected_composition_receipt(
    *,
    source_runtime_catalog: Mapping[str, object],
    corrected_layer17_graph_sha256: str,
    corrected_layer17_lowering_sha256_by_node: Mapping[str, str],
    corrected_composition_graph_sha256: str,
) -> dict[str, object]:
    """Bind one fold's corrected graph to the exact frozen composition inputs."""

    layer10 = source_runtime_catalog.get("layer10")
    layer17 = source_runtime_catalog.get("layer17")
    if not isinstance(layer10, Mapping) or not isinstance(layer17, Mapping):
        raise TypeError("source runtime layer catalogs are unavailable")
    primary_order = tuple(
        source_runtime_catalog.get("frozen_primary_traversal_order", ())
    )
    layer10_order = tuple(layer10.get("traversal_order", ()))
    layer17_order = tuple(layer17.get("traversal_order", ()))
    frozen_layer10_lowerings = layer10.get("lowering_sha256_by_node")
    if (
        primary_order != (*layer10_order, *layer17_order)
        or not isinstance(frozen_layer10_lowerings, Mapping)
        or set(frozen_layer10_lowerings) != set(layer10_order)
        or set(corrected_layer17_lowering_sha256_by_node) != set(layer17_order)
    ):
        raise ValueError("corrected composition lowering catalogs are incomplete")
    for digest in (
        *frozen_layer10_lowerings.values(),
        *corrected_layer17_lowering_sha256_by_node.values(),
    ):
        _require_sha256(digest, label="composition lowering")
    result_lowerings = {
        name: (
            frozen_layer10_lowerings[name]
            if name in frozen_layer10_lowerings
            else corrected_layer17_lowering_sha256_by_node[name]
        )
        for name in primary_order
    }
    payload = {
        "source_primary_graph_sha256": _require_sha256(
            source_runtime_catalog.get("frozen_primary_graph_sha256"),
            label="source primary graph",
        ),
        "source_runtime_catalog_sha256": _require_sha256(
            source_runtime_catalog.get("catalog_sha256"),
            label="source runtime catalog",
        ),
        "frozen_layer10": {
            "graph_sha256": _require_sha256(
                layer10.get("graph_sha256"),
                label="frozen Layer10 graph",
            ),
            "traversal_order": layer10_order,
            "lowering_sha256_by_node": dict(frozen_layer10_lowerings),
        },
        "corrected_layer17": {
            "graph_sha256": _require_sha256(
                corrected_layer17_graph_sha256,
                label="corrected Layer17 graph",
            ),
            "traversal_order": layer17_order,
            "lowering_sha256_by_node": dict(
                corrected_layer17_lowering_sha256_by_node
            ),
        },
        "result": {
            "graph_sha256": _require_sha256(
                corrected_composition_graph_sha256,
                label="corrected composition graph",
            ),
            "traversal_order": primary_order,
            "lowering_sha256_by_node": result_lowerings,
        },
        "replacement_policy": dict(_COMPOSITION_REPLACEMENT_POLICY),
        "exact_resources": dict(
            _EXPECTED_CONDITION_RESOURCES["trajectory_corrected_composition"]
        ),
    }
    return {
        **payload,
        "receipt_sha256": _domain_sha256(_COMPOSITION_RECEIPT_DOMAIN, payload),
    }


@dataclass(frozen=True, slots=True)
class _TrajectoryCorrectionFitView:
    """The only capture tensors retained across the eight LOFO folds."""

    compiled_input: Tensor
    a3_target: Tensor
    native_fisher_weights_by_fragment: Mapping[str, Tensor]
    row_keys: tuple[tuple[str, int], ...]
    row_key_sha256: str
    sequences: int
    fragment_ids: tuple[str, ...]
    capture_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.fragment_ids) is not tuple
            or len(self.fragment_ids) != 4
            or len(self.fragment_ids) != len(set(self.fragment_ids))
            or any(
                not isinstance(fragment_id, str) or not fragment_id
                for fragment_id in self.fragment_ids
            )
        ):
            raise ValueError("fit view requires exactly four unique fragments")
        if (
            not isinstance(self.compiled_input, Tensor)
            or not isinstance(self.a3_target, Tensor)
            or self.compiled_input.ndim != 2
            or self.a3_target.ndim != 2
            or self.compiled_input.shape != self.a3_target.shape
            or self.compiled_input.shape[0] != len(self.row_keys)
            or self.compiled_input.device.type != "cpu"
            or self.a3_target.device.type != "cpu"
            or self.compiled_input.dtype != torch.float64
            or self.a3_target.dtype != torch.float64
            or not bool(torch.isfinite(self.compiled_input).all())
            or not bool(torch.isfinite(self.a3_target).all())
        ):
            raise ValueError("fit view compiled input/A3 target are invalid")
        if (
            type(self.row_keys) is not tuple
            or not self.row_keys
            or len(self.row_keys) != len(set(self.row_keys))
            or any(
                type(key) is not tuple
                or len(key) != 2
                or not isinstance(key[0], str)
                or not key[0]
                or type(key[1]) is not int
                or key[1] < 0
                for key in self.row_keys
            )
            or self.row_key_sha256 != _row_key_sha256(self.row_keys)
        ):
            raise ValueError("fit view row-key lineage is invalid")
        if (
            type(self.sequences) is not int
            or self.sequences <= 0
            or self.sequences
            != len({example_id for example_id, _ in self.row_keys})
        ):
            raise ValueError("fit view sequence accounting is invalid")
        _require_sha256(self.capture_sha256, label="fit view capture")
        if (
            not isinstance(self.native_fisher_weights_by_fragment, Mapping)
            or tuple(self.native_fisher_weights_by_fragment) != self.fragment_ids
        ):
            raise ValueError("fit view native Fisher catalog is invalid")
        fisher = dict(self.native_fisher_weights_by_fragment)
        for fragment_id in self.fragment_ids:
            weights = fisher[fragment_id]
            if (
                not isinstance(weights, Tensor)
                or weights.ndim != 1
                or weights.shape[0] != len(self.row_keys)
                or weights.device.type != "cpu"
                or weights.dtype != torch.float64
                or not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any())
                or float(weights.sum().item()) <= 0.0
            ):
                raise ValueError("fit view native Fisher vector is invalid")
        object.__setattr__(
            self,
            "native_fisher_weights_by_fragment",
            MappingProxyType(fisher),
        )

    @property
    def observations(self) -> int:
        return len(self.row_keys)


def _family_indices(
    row_keys: tuple[tuple[str, int], ...],
    family_alias_by_example: Mapping[str, str],
) -> dict[str, Tensor]:
    examples = {example_id for example_id, _ in row_keys}
    if set(family_alias_by_example) != examples:
        raise ValueError("family ownership must exactly cover trajectory rows")
    aliases = tuple(sorted(set(family_alias_by_example.values())))
    if aliases != V8_FAMILY_LOFO_FAMILY_ALIASES:
        raise ValueError("trajectory rows require the exact eight opaque families")
    result = {
        alias: torch.tensor(
            [
                index
                for index, (example_id, _) in enumerate(row_keys)
                if family_alias_by_example[example_id] == alias
            ],
            dtype=torch.long,
        )
        for alias in aliases
    }
    if any(indices.numel() <= 0 for indices in result.values()):
        raise ValueError("each trajectory family must contain rows")
    return result


def _ordered_family_indices(
    family_indices: Mapping[str, Tensor],
    aliases: Sequence[str],
) -> Tensor:
    selected = tuple(aliases)
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(family_indices)
    ):
        raise ValueError("family index aliases are invalid")
    parts = tuple(family_indices[alias] for alias in selected)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _selected_native_fisher(
    weights: Tensor,
    family_indices: Mapping[str, Tensor],
    aliases: Sequence[str],
    *,
    normalize_by_family: bool,
) -> Tensor:
    selected = tuple(aliases)
    if (
        not isinstance(weights, Tensor)
        or weights.ndim != 1
        or not selected
        or not set(selected).issubset(family_indices)
    ):
        raise ValueError("native Fisher selection is invalid")
    parts: list[Tensor] = []
    for alias in selected:
        current = weights.index_select(0, family_indices[alias])
        total = current.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            raise ValueError("family Fisher mass must be positive")
        parts.append(
            current
            if not normalize_by_family
            else current / total / len(selected)
        )
    result = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    if normalize_by_family:
        expected = torch.tensor(1.0, dtype=result.dtype)
        if not torch.allclose(result.sum(), expected, rtol=0.0, atol=2e-12):
            raise RuntimeError("equal-family Fisher normalization drifted")
    return result


def _shared_compiled_input(
    rows: AlignedFragmentRows,
    fragment_ids: Sequence[str],
) -> Tensor:
    ordered = tuple(fragment_ids)
    values = tuple(rows.rows_by_fragment[value].inputs for value in ordered)
    if not values or any(
        value.shape != values[0].shape or not torch.equal(value, values[0])
        for value in values[1:]
    ):
        raise ValueError("selected Layer17 fragments do not share one input")
    return values[0]


def _build_trajectory_correction_fit_view(
    row_pair: Gemma3Layer17TrajectoryRowPair,
) -> _TrajectoryCorrectionFitView:
    """Retain only the tensors needed by the fixed eight-fold refit."""

    if not isinstance(row_pair, Gemma3Layer17TrajectoryRowPair):
        raise TypeError("row_pair must be a trajectory capture")
    fragment_ids = tuple(row_pair.fragment_ids)
    compiled_input = _shared_compiled_input(
        row_pair.compiled_rows,
        fragment_ids,
    )
    target = build_a3_raw_mlp_target(
        row_pair.native_full_mlp_output,
        row_pair.compiled_compact_retained_mlp_output,
    )
    return _TrajectoryCorrectionFitView(
        compiled_input=compiled_input,
        a3_target=target,
        native_fisher_weights_by_fragment={
            fragment_id: row_pair.native_rows.rows_by_fragment[
                fragment_id
            ].fisher_weights
            for fragment_id in fragment_ids
        },
        row_keys=row_pair.native_rows.row_keys,
        row_key_sha256=row_pair.native_rows.row_key_sha256,
        sequences=row_pair.native_rows.sequences,
        fragment_ids=fragment_ids,
        capture_sha256=row_pair.capture_sha256,
    )


def _split_sha256(
    *,
    protocol_sha256: str,
    capture_sha256: str,
    fold_sha256: str,
    authenticated_stream_sha256: str,
    role: str,
    aliases: Sequence[str],
    rows: AlignedFragmentRows,
) -> str:
    if role not in {"fit", "held"}:
        raise ValueError("trajectory split role must be fit or held")
    return _domain_sha256(
        _SPLIT_DOMAIN,
        {
            "protocol_sha256": _require_sha256(
                protocol_sha256, label="protocol"
            ),
            "capture_sha256": _require_sha256(capture_sha256, label="capture"),
            "fold_sha256": _require_sha256(fold_sha256, label="fold"),
            "authenticated_stream_sha256": _require_sha256(
                authenticated_stream_sha256,
                label="authenticated tokenized stream",
            ),
            "role": role,
            "family_aliases": tuple(aliases),
            "row_key_sha256": rows.row_key_sha256,
            "observation_count": rows.observations,
            "sequence_count": rows.sequences,
            "fisher_normalization": (
                "equal_total_mass_per_training_family_from_native_virtual_gate_fisher"
                if role == "fit"
                else "native_virtual_gate_fisher_preserved_unnormalized"
            ),
        },
    )


def _build_trajectory_correction_fold_rows_from_fit_view(
    fit_view: _TrajectoryCorrectionFitView,
    *,
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: Sequence[str],
    held_family_alias: str,
    fold_sha256: str,
    protocol_sha256: str,
    authenticated_stream_sha256: str,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
) -> tuple[AlignedFragmentRows, AlignedFragmentRows, dict[str, object]]:
    """Create one fold by directly indexing the capture-bound lean fit view."""

    if not isinstance(fit_view, _TrajectoryCorrectionFitView):
        raise TypeError("fit_view must be a trajectory correction fit view")
    training = tuple(training_family_aliases)
    if (
        held_family_alias not in V8_FAMILY_LOFO_FAMILY_ALIASES
        or len(training) != _EXPECTED_FAMILIES - 1
        or held_family_alias in training
        or tuple(sorted((*training, held_family_alias)))
        != V8_FAMILY_LOFO_FAMILY_ALIASES
    ):
        raise ValueError("trajectory fold must be exact train-seven/hold-one")
    source_graph.validate_integrity()
    node_order = source_graph.traversal_order
    if set(source_lowerings_by_node) != set(node_order):
        raise ValueError("source lowering catalog differs from graph")
    fragment_by_node = {
        name: _selected_fragment_id(source_lowerings_by_node[name])
        for name in node_order
    }
    bases_by_node = {
        name: source_lowerings_by_node[name].computational_mode_basis
        for name in node_order
    }
    if tuple(fragment_by_node.values()) != fit_view.fragment_ids:
        raise ValueError("fit view fragments differ from source lowerings")
    family_indices = _family_indices(
        fit_view.row_keys,
        family_alias_by_example,
    )

    def build(
        aliases: tuple[str, ...],
    ) -> tuple[AlignedFragmentRows, dict[str, object]]:
        is_training = len(aliases) == _EXPECTED_FAMILIES - 1
        indices = _ordered_family_indices(family_indices, aliases)
        row_keys = tuple(
            fit_view.row_keys[int(index)] for index in indices.tolist()
        )
        sequences = len({example_id for example_id, _ in row_keys})
        projected, projection = build_projected_correction_rows(
            inputs=fit_view.compiled_input.index_select(0, indices),
            target=fit_view.a3_target.index_select(0, indices),
            fisher_weights_by_node={
                name: _selected_native_fisher(
                    fit_view.native_fisher_weights_by_fragment[
                        fragment_by_node[name]
                    ],
                    family_indices,
                    aliases,
                    normalize_by_family=is_training,
                )
                for name in node_order
            },
            fragment_id_by_node=fragment_by_node,
            bases_by_node=bases_by_node,  # type: ignore[arg-type]
            node_order=node_order,
            row_keys=row_keys,
            sequences=sequences,
        )
        projection_metadata = projection.metadata()
        del projection
        return projected, projection_metadata

    fit_rows, fit_projection_metadata = build(training)
    held_rows, held_projection_metadata = build((held_family_alias,))
    if set(fit_rows.row_keys) & set(held_rows.row_keys):
        raise RuntimeError("trajectory fit and held row sets overlap")
    if set(fit_rows.row_keys) | set(held_rows.row_keys) != set(
        fit_view.row_keys
    ) or len(fit_rows.row_keys) + len(held_rows.row_keys) != len(
        fit_view.row_keys
    ):
        raise RuntimeError("trajectory fit and held rows do not cover capture")
    fit_split_sha256 = _split_sha256(
        protocol_sha256=protocol_sha256,
        capture_sha256=fit_view.capture_sha256,
        fold_sha256=fold_sha256,
        authenticated_stream_sha256=authenticated_stream_sha256,
        role="fit",
        aliases=training,
        rows=fit_rows,
    )
    held_split_sha256 = _split_sha256(
        protocol_sha256=protocol_sha256,
        capture_sha256=fit_view.capture_sha256,
        fold_sha256=fold_sha256,
        authenticated_stream_sha256=authenticated_stream_sha256,
        role="held",
        aliases=(held_family_alias,),
        rows=held_rows,
    )
    receipt = {
        "capture_sha256": fit_view.capture_sha256,
        "protocol_fold_sha256": fold_sha256,
        "authenticated_stream_sha256": authenticated_stream_sha256,
        "fit_split_sha256": fit_split_sha256,
        "held_split_sha256": held_split_sha256,
        "fit_family_aliases": list(training),
        "held_family_alias": held_family_alias,
        "fit_row_key_sha256": fit_rows.row_key_sha256,
        "held_row_key_sha256": held_rows.row_key_sha256,
        "fit_observations": fit_rows.observations,
        "held_observations": held_rows.observations,
        "fit_sequences": fit_rows.sequences,
        "held_sequences": held_rows.sequences,
        "held_family_excluded_from_projection_fit_and_generator_fit": True,
        "held_fisher_weights_preserved_without_held_family_normalization": True,
        "held_projection_used_only_as_fixed_basis_generator_evaluation_target": True,
        "fit_projection": fit_projection_metadata,
        "held_projection": held_projection_metadata,
    }
    _reject_forbidden_output_fields(receipt, path="fold row receipt")
    return fit_rows, held_rows, receipt


def build_trajectory_correction_fold_rows(
    row_pair: Gemma3Layer17TrajectoryRowPair,
    *,
    family_alias_by_example: Mapping[str, str],
    training_family_aliases: Sequence[str],
    held_family_alias: str,
    fold_sha256: str,
    protocol_sha256: str,
    authenticated_stream_sha256: str,
    source_graph: ModalGeneratorGraphPlan,
    source_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
) -> tuple[AlignedFragmentRows, AlignedFragmentRows, dict[str, object]]:
    """Compatibility wrapper using the same exact-target lean fit view."""

    fit_view = _build_trajectory_correction_fit_view(row_pair)
    return _build_trajectory_correction_fold_rows_from_fit_view(
        fit_view,
        family_alias_by_example=family_alias_by_example,
        training_family_aliases=training_family_aliases,
        held_family_alias=held_family_alias,
        fold_sha256=fold_sha256,
        protocol_sha256=protocol_sha256,
        authenticated_stream_sha256=authenticated_stream_sha256,
        source_graph=source_graph,
        source_lowerings_by_node=source_lowerings_by_node,
    )


def _compact_condition_resources(
    *,
    condition: str,
    plan: ModalGeneratorGraphPlan,
    static: Mapping[str, object],
    totals: Mapping[str, int],
    logical_valid_tokens: int,
    peak_live_modal_width: int,
) -> dict[str, int]:
    if condition not in _EXPECTED_CONDITION_RESOURCES:
        raise ValueError("unknown trajectory evaluation condition")
    if logical_valid_tokens <= 0:
        raise ValueError("logical valid-token total must be positive")
    for field in _GRAPH_LOGICAL_FIELDS:
        if type(totals.get(field)) is not int:
            raise ValueError(f"{condition} logical accounting is incomplete")
        if totals[field] % logical_valid_tokens:
            raise RuntimeError(f"{condition} logical accounting is not exact")
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
        "executed_graph_macs_per_token": (
            totals["logical_executed_modal_graph_macs"] // logical_valid_tokens
        ),
        "net_executed_macs_saved_per_token": (
            totals["net_logical_macs_saved"] // logical_valid_tokens
        ),
    }
    expected = _EXPECTED_CONDITION_RESOURCES[condition]
    if observed != expected:
        raise RuntimeError(
            f"{condition} resources differ from frozen protocol: {observed!r}"
        )
    source_parameters = int(static["source_whole_model_learned_parameters"])
    if (
        source_parameters
        != _EXPECTED_EXACT_RESOURCES["source_whole_model_learned_parameters"]
        or int(static["candidate_whole_model_learned_parameters"])
        != source_parameters
        - observed["native_removed_parameters"]
        + observed["graph_parameters"]
        or totals["logical_linear_macs_native_removed"]
        != observed["native_removed_parameters"] * logical_valid_tokens
        or totals["logical_modal_graph_macs"]
        != observed["dense_graph_macs_per_token"] * logical_valid_tokens
    ):
        raise RuntimeError(f"{condition} exact accounting identities drifted")
    return {
        **observed,
        "executed_peak_live_modal_width": peak_live_modal_width,
    }


def score_trajectory_correction_fold(
    *,
    adapter: Gemma3CausalLMAdapter,
    layer10_executor: Gemma3ModalGeneratorGraphExecutor,
    corrected_layer17_executor: Gemma3ModalGeneratorGraphExecutor,
    frozen_composition_executor: Gemma3ModalGeneratorGraphExecutor,
    corrected_composition_executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    """Score the exact frozen six-condition panel on one held family."""

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("held fold batches must contain CalibrationBatch values")
    ids = tuple(
        example_id
        for batch in materialized
        for example_id in (batch.example_ids or ())
    )
    if (
        any(batch.example_ids is None for batch in materialized)
        or not ids
        or len(ids) != len(set(ids))
    ):
        raise ValueError("held fold batches require unique example identities")
    executors = {
        "layer10_only": layer10_executor,
        "trajectory_corrected_layer17_only": corrected_layer17_executor,
        "frozen_uncorrected_composition": frozen_composition_executor,
        "trajectory_corrected_composition": corrected_composition_executor,
    }
    if len({id(value) for value in executors.values()}) != len(executors):
        raise ValueError("trajectory evaluation executors must be distinct")
    plans = {name: value.graph_plan for name, value in executors.items()}
    if (
        layer10_executor.affected_layer_ordinals != (10,)
        or corrected_layer17_executor.affected_layer_ordinals != (17,)
        or frozen_composition_executor.affected_layer_ordinals != (10, 17)
        or corrected_composition_executor.affected_layer_ordinals != (10, 17)
        or len(plans["layer10_only"].interactions) != 3
        or plans["trajectory_corrected_layer17_only"].interactions
        or len(plans["frozen_uncorrected_composition"].interactions) != 3
        or len(plans["trajectory_corrected_composition"].interactions) != 3
    ):
        raise ValueError("trajectory evaluation topology differs from protocol")

    aggregate = _new_metric_accumulator(_CONDITIONS)
    static_by_condition: dict[str, dict[str, object]] = {}
    totals_by_condition: dict[str, dict[str, int]] = {}
    peak_by_condition: dict[str, int] = {}
    logical_valid_tokens = 0
    native_model = adapter.module
    with ExitStack() as stack:
        for executor in executors.values():
            stack.enter_context(executor.validated_transaction())
        for batch in materialized:
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = native_model(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output), batch
            )
            token_count = targets.numel()
            _add_native(
                aggregate,
                nll_sum=_native_nll(native_logits, targets),
                token_count=token_count,
            )
            expected_valid = int(batch.valid_positions.sum().item())
            valid_counts: list[int] = []

            for condition, executor in executors.items():
                with torch.no_grad():
                    execution = executor.run(
                        batch.model_inputs,
                        condition="generated",
                    )
                _validate_graph_execution(
                    execution,
                    plans[condition],
                    condition="generated",
                    label=condition,
                )
                logits, candidate_targets = _selected_logits_and_targets(
                    _model_logits(execution.model_output), batch
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{condition} held targets drifted")
                comparison = _candidate_comparison(
                    native_logits,
                    logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                )
                _add_comparison(aggregate, condition, comparison)
                _record_execution(
                    static_by_condition,
                    totals_by_condition,
                    peak_by_condition,
                    condition=condition,
                    execution=execution,
                )
                valid = getattr(execution, "valid_tokens", None)
                if type(valid) is not int:
                    raise RuntimeError(f"{condition} valid-token count is invalid")
                valid_counts.append(valid)

            with torch.no_grad():
                deletion = corrected_composition_executor.run(
                    batch.model_inputs,
                    condition="deletion",
                )
            deletion_plan = plans["trajectory_corrected_composition"]
            _validate_graph_execution(
                deletion,
                deletion_plan,
                condition="deletion",
                label="matched_double_deletion",
            )
            deletion_logits, deletion_targets = _selected_logits_and_targets(
                _model_logits(deletion.model_output), batch
            )
            if not torch.equal(targets, deletion_targets):
                raise RuntimeError("matched deletion targets drifted")
            _add_comparison(
                aggregate,
                "matched_double_deletion",
                _candidate_comparison(
                    native_logits,
                    deletion_logits,
                    targets,
                    vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                ),
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
                label="matched_double_deletion",
            )
            corrected_static = static_by_condition[
                "trajectory_corrected_composition"
            ]
            if deletion_static != corrected_static:
                raise RuntimeError("corrected generated/deletion scopes differ")
            deletion_valid = getattr(deletion, "valid_tokens", None)
            if type(deletion_valid) is not int:
                raise RuntimeError("matched deletion valid-token count is invalid")
            valid_counts.append(deletion_valid)
            if set(valid_counts) != {expected_valid}:
                raise RuntimeError("trajectory conditions disagree on valid tokens")
            logical_valid_tokens += expected_valid

    metrics = _finalize_metric_accumulator(aggregate, conditions=_CONDITIONS)
    plan_by_condition = {
        **plans,
        "matched_double_deletion": plans[
            "trajectory_corrected_composition"
        ],
    }
    resources = {
        condition: _compact_condition_resources(
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
        "execution_path": "full_model_logits_fixed_capacity_a3_trajectory_lofo",
        "assessment_role": "calibration_a_fit_family_blocked_development",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": metrics["supervised_tokens"],
        "logical_valid_tokens": logical_valid_tokens,
        "native": metrics["native"],
        "conditions": metrics["conditions"],
        "resource_accounting": resources,
        "exact_resources_match_protocol": True,
        "latency_or_kernel_speed_claim": False,
    }


def _validate_metric_row(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }:
        raise ValueError(f"{label} metric fields are invalid")
    nll = _finite(value["nll_per_token"], label=f"{label} NLL")
    delta = _finite(
        value["delta_nll_per_token"], label=f"{label} delta NLL"
    )
    kl = _finite(
        value["native_to_candidate_kl_per_token"], label=f"{label} KL"
    )
    top1 = _finite(
        value["top1_agreement_to_native"], label=f"{label} top1"
    )
    if (
        nll < 0.0
        or kl < _MINIMUM_NUMERICAL_KL_PER_TOKEN
        or not 0.0 <= top1 <= 1.0
    ):
        raise ValueError(f"{label} metrics are outside valid ranges")
    if not math.isclose(nll, delta + (nll - delta), rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{label} metrics are non-finite")
    return value


def _validate_fold_evaluation(
    value: object,
    *,
    label: str,
) -> Mapping[str, object]:
    required = {
        "execution_path",
        "assessment_role",
        "heldout_confirmation",
        "full_model_logits_scored",
        "full_model_compiled",
        "supervised_tokens",
        "logical_valid_tokens",
        "native",
        "conditions",
        "resource_accounting",
        "exact_resources_match_protocol",
        "latency_or_kernel_speed_claim",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} evaluation fields are invalid")
    if (
        value["execution_path"]
        != "full_model_logits_fixed_capacity_a3_trajectory_lofo"
        or value["assessment_role"]
        != "calibration_a_fit_family_blocked_development"
        or value["heldout_confirmation"] is not False
        or value["full_model_logits_scored"] is not True
        or value["full_model_compiled"] is not False
        or value["exact_resources_match_protocol"] is not True
        or value["latency_or_kernel_speed_claim"] is not False
        or type(value["supervised_tokens"]) is not int
        or int(value["supervised_tokens"]) <= 0
        or type(value["logical_valid_tokens"]) is not int
        or int(value["logical_valid_tokens"]) <= 0
    ):
        raise ValueError(f"{label} evaluation boundary is invalid")
    native = value.get("native")
    conditions = value.get("conditions")
    resources = value.get("resource_accounting")
    if (
        not isinstance(native, Mapping)
        or set(native) != {"nll_per_token"}
        or _finite(native["nll_per_token"], label=f"{label} native NLL") < 0.0
        or not isinstance(conditions, Mapping)
        or set(conditions) != set(_CONDITIONS)
        or not isinstance(resources, Mapping)
        or set(resources) != set(_CONDITIONS)
    ):
        raise ValueError(f"{label} metric/resource catalog is invalid")
    native_nll = float(native["nll_per_token"])
    for condition in _CONDITIONS:
        row = _validate_metric_row(
            conditions[condition], label=f"{label}/{condition}"
        )
        if not math.isclose(
            float(row["nll_per_token"]),
            native_nll + float(row["delta_nll_per_token"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label}/{condition} NLL identity drifted")
        expected = {
            **_EXPECTED_CONDITION_RESOURCES[condition],
            "executed_peak_live_modal_width": (
                resources[condition].get("executed_peak_live_modal_width")
                if isinstance(resources[condition], Mapping)
                else None
            ),
        }
        if (
            not isinstance(resources[condition], Mapping)
            or _canonical_json_bytes(resources[condition])
            != _canonical_json_bytes(expected)
            or type(expected["executed_peak_live_modal_width"]) is not int
            or int(expected["executed_peak_live_modal_width"]) < 0
            or (
                condition == "matched_double_deletion"
                and expected["executed_peak_live_modal_width"] != 0
            )
        ):
            raise ValueError(f"{label}/{condition} resources drifted")
    return value


def aggregate_trajectory_correction_lofo_folds(
    fold_evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return micro and equal-family summaries for exactly eight folds."""

    folds = tuple(
        _validate_fold_evaluation(value, label=f"fold {index}")
        for index, value in enumerate(fold_evaluations)
    )
    if len(folds) != _EXPECTED_FAMILIES:
        raise ValueError("trajectory LOFO requires exactly eight fold evaluations")
    totals = _new_metric_accumulator(_CONDITIONS)
    families: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(folds):
        tokens = int(value["supervised_tokens"])
        native = value["native"]
        conditions = value["conditions"]
        assert isinstance(native, Mapping)
        assert isinstance(conditions, Mapping)
        _add_native(
            totals,
            nll_sum=float(native["nll_per_token"]) * tokens,
            token_count=tokens,
        )
        for condition in _CONDITIONS:
            row = conditions[condition]
            assert isinstance(row, Mapping)
            _add_comparison(
                totals,
                condition,
                {
                    "nll_sum": float(row["nll_per_token"]) * tokens,
                    "native_to_candidate_kl_sum": float(
                        row["native_to_candidate_kl_per_token"]
                    )
                    * tokens,
                    "top1_matches": round(
                        float(row["top1_agreement_to_native"]) * tokens
                    ),
                },
            )
        families[f"family_{index:02d}"] = {
            "supervised_tokens": tokens,
            "native": dict(native),
            "conditions": {
                condition: dict(conditions[condition])  # type: ignore[arg-type]
                for condition in _CONDITIONS
            },
        }
    micro = _finalize_metric_accumulator(totals, conditions=_CONDITIONS)
    macro = {
        "native": {
            "nll_per_token": math.fsum(
                float(value["native"]["nll_per_token"])  # type: ignore[index]
                for value in families.values()
            )
            / _EXPECTED_FAMILIES
        },
        "conditions": {
            condition: {
                metric: math.fsum(
                    float(value["conditions"][condition][metric])  # type: ignore[index]
                    for value in families.values()
                )
                / _EXPECTED_FAMILIES
                for metric in (
                    "nll_per_token",
                    "delta_nll_per_token",
                    "native_to_candidate_kl_per_token",
                    "top1_agreement_to_native",
                )
            }
            for condition in _CONDITIONS
        },
    }
    return {
        "micro": micro,
        "equal_family_macro": macro,
        "family_count": _EXPECTED_FAMILIES,
        "completed_fold_count": len(folds),
        "failed_fold_count": 0,
    }


def _gate(
    gate_id: str,
    *,
    observed: int | float | bool | None,
    operator: str,
    threshold: int | float | bool,
) -> dict[str, object]:
    if observed is None:
        passed = False
    elif operator == "<=":
        passed = float(observed) <= float(threshold)
    elif operator == ">=":
        passed = float(observed) >= float(threshold)
    elif operator == "<":
        passed = float(observed) < float(threshold)
    elif operator == "==":
        passed = observed == threshold
    else:
        raise ValueError("unsupported trajectory gate operator")
    return {
        "gate_id": gate_id,
        "required": True,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }


def _condition_metric(
    evaluation: Mapping[str, object],
    condition: str,
    metric: str,
) -> float:
    conditions = evaluation["conditions"]
    assert isinstance(conditions, Mapping)
    row = conditions[condition]
    assert isinstance(row, Mapping)
    return float(row[metric])


def evaluate_trajectory_correction_lofo_gates(
    *,
    protocol: Mapping[str, object],
    fold_evaluations: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    exact_resources_match: bool,
    exact_projection_metadata_match: bool,
    compact_replay_algebraic_equivalence_audit: bool,
    source_model_unchanged: bool,
    layer10_unchanged: bool,
) -> dict[str, object]:
    """Replay every gate predeclared by the frozen trajectory protocol."""

    frozen = validate_gemma3_l10_l17_trajectory_correction_protocol(protocol)
    folds = tuple(
        _validate_fold_evaluation(value, label=f"gate fold {index}")
        for index, value in enumerate(fold_evaluations)
    )
    recomputed = aggregate_trajectory_correction_lofo_folds(folds)
    if _canonical_json_bytes(aggregate) != _canonical_json_bytes(recomputed):
        raise ValueError("trajectory aggregate differs from fold metrics")
    gates = frozen.get("gates")
    if not isinstance(gates, Mapping):
        raise TypeError("trajectory protocol gates are unavailable")
    macro = recomputed["equal_family_macro"]
    assert isinstance(macro, Mapping)
    macro_conditions = macro["conditions"]
    assert isinstance(macro_conditions, Mapping)
    corrected = macro_conditions["trajectory_corrected_composition"]
    frozen_composition = macro_conditions["frozen_uncorrected_composition"]
    assert isinstance(corrected, Mapping)
    assert isinstance(frozen_composition, Mapping)

    corrected_delta = float(corrected["delta_nll_per_token"])
    corrected_kl = float(corrected["native_to_candidate_kl_per_token"])
    corrected_top1 = float(corrected["top1_agreement_to_native"])
    frozen_delta = float(frozen_composition["delta_nll_per_token"])
    frozen_kl = float(frozen_composition["native_to_candidate_kl_per_token"])
    frozen_top1 = float(frozen_composition["top1_agreement_to_native"])
    worst_delta = max(
        _condition_metric(
            value,
            "trajectory_corrected_composition",
            "delta_nll_per_token",
        )
        for value in folds
    )
    interaction_by_family = tuple(
        _condition_metric(
            value,
            "trajectory_corrected_composition",
            "delta_nll_per_token",
        )
        - _condition_metric(value, "layer10_only", "delta_nll_per_token")
        - _condition_metric(
            value,
            "trajectory_corrected_layer17_only",
            "delta_nll_per_token",
        )
        for value in folds
    )
    macro_interaction = math.fsum(interaction_by_family) / len(
        interaction_by_family
    )
    kl_improvement_count = sum(
        _condition_metric(
            value,
            "trajectory_corrected_composition",
            "native_to_candidate_kl_per_token",
        )
        < _condition_metric(
            value,
            "frozen_uncorrected_composition",
            "native_to_candidate_kl_per_token",
        )
        for value in folds
    )
    recovery: list[float | None] = []
    for value in folds:
        generated = _condition_metric(
            value,
            "trajectory_corrected_composition",
            "delta_nll_per_token",
        )
        deletion = _condition_metric(
            value,
            "matched_double_deletion",
            "delta_nll_per_token",
        )
        recovery.append(
            None if deletion <= 0.0 else (deletion - generated) / deletion
        )
    valid_recovery = tuple(value for value in recovery if value is not None)
    invalid_recovery_count = len(recovery) - len(valid_recovery)
    macro_recovery = (
        None
        if invalid_recovery_count
        else math.fsum(valid_recovery) / len(valid_recovery)
    )
    worst_recovery = (
        None if invalid_recovery_count else min(valid_recovery)
    )

    gate_rows = (
        _gate(
            "completed_fold_count",
            observed=len(folds),
            operator=">=",
            threshold=int(gates["required_completed_fold_count"]),
        ),
        _gate(
            "failed_fold_count",
            observed=0,
            operator="<=",
            threshold=int(gates["maximum_failed_fold_count"]),
        ),
        _gate(
            "family_macro_delta_nll_per_token",
            observed=corrected_delta,
            operator="<=",
            threshold=float(gates["maximum_family_macro_delta_nll_per_token"]),
        ),
        _gate(
            "worst_family_delta_nll_per_token",
            observed=worst_delta,
            operator="<=",
            threshold=float(gates["maximum_worst_family_delta_nll_per_token"]),
        ),
        _gate(
            "family_macro_native_to_candidate_kl_per_token",
            observed=corrected_kl,
            operator="<=",
            threshold=float(
                gates[
                    "maximum_family_macro_native_to_candidate_kl_per_token"
                ]
            ),
        ),
        _gate(
            "family_macro_top1_agreement_to_native",
            observed=corrected_top1,
            operator=">=",
            threshold=float(
                gates["minimum_family_macro_top1_agreement_to_native"]
            ),
        ),
        _gate(
            "family_macro_interaction_excess_nll",
            observed=macro_interaction,
            operator="<=",
            threshold=float(
                gates["maximum_family_macro_interaction_excess_nll"]
            ),
        ),
        _gate(
            "strict_family_macro_kl_improvement_vs_frozen",
            observed=corrected_kl,
            operator="<",
            threshold=frozen_kl,
        ),
        _gate(
            "held_family_kl_improvement_count",
            observed=kl_improvement_count,
            operator=">=",
            threshold=int(gates["minimum_held_family_kl_improvement_count"]),
        ),
        _gate(
            "held_family_count",
            observed=len(folds),
            operator="==",
            threshold=int(gates["required_held_family_count"]),
        ),
        _gate(
            "family_macro_nll_regression_vs_frozen",
            observed=corrected_delta - frozen_delta,
            operator="<=",
            threshold=float(
                gates["maximum_family_macro_nll_regression_vs_frozen"]
            ),
        ),
        _gate(
            "family_macro_top1_regression_vs_frozen",
            observed=frozen_top1 - corrected_top1,
            operator="<=",
            threshold=float(
                gates["maximum_family_macro_top1_regression_vs_frozen"]
            ),
        ),
        _gate(
            "family_macro_deletion_nll_recovery_fraction",
            observed=macro_recovery,
            operator=">=",
            threshold=float(
                gates[
                    "minimum_family_macro_deletion_nll_recovery_fraction"
                ]
            ),
        ),
        _gate(
            "worst_family_deletion_nll_recovery_fraction",
            observed=worst_recovery,
            operator=">=",
            threshold=float(
                gates[
                    "minimum_worst_family_deletion_nll_recovery_fraction"
                ]
            ),
        ),
        _gate(
            "exact_resources",
            observed=exact_resources_match,
            operator="==",
            threshold=bool(gates["require_exact_resources"]),
        ),
        _gate(
            "exact_projection_metadata",
            observed=exact_projection_metadata_match,
            operator="==",
            threshold=bool(gates["require_exact_projection_metadata"]),
        ),
        _gate(
            "compact_replay_algebraic_equivalence_audit",
            observed=compact_replay_algebraic_equivalence_audit,
            operator="==",
            threshold=bool(
                gates[
                    "require_compact_replay_algebraic_equivalence_audit"
                ]
            ),
        ),
        _gate(
            "source_model_unchanged",
            observed=source_model_unchanged,
            operator="==",
            threshold=bool(gates["require_source_model_unchanged"]),
        ),
        _gate(
            "layer10_unchanged",
            observed=layer10_unchanged,
            operator="==",
            threshold=bool(gates["require_layer10_unchanged"]),
        ),
    )
    passed = all(bool(row["passed"]) for row in gate_rows)
    return {
        "protocol_sha256": frozen["artifact_sha256"],
        "decision_policy": gates["decision_policy"],
        "derived_metrics": {
            "family_macro_interaction_excess_nll": macro_interaction,
            "family_macro_kl_improvement_vs_frozen": frozen_kl - corrected_kl,
            "held_family_kl_improvement_count": kl_improvement_count,
            "family_macro_nll_regression_vs_frozen": corrected_delta
            - frozen_delta,
            "family_macro_top1_regression_vs_frozen": frozen_top1
            - corrected_top1,
            "deletion_nll_recovery_fraction_by_family_alias": {
                f"family_{index:02d}": value
                for index, value in enumerate(recovery)
            },
            "deletion_nll_recovery_denominator_valid": (
                invalid_recovery_count == 0
            ),
            "deletion_nll_recovery_invalid_denominator_count": (
                invalid_recovery_count
            ),
            "family_macro_deletion_nll_recovery_fraction": macro_recovery,
            "worst_family_deletion_nll_recovery_fraction": worst_recovery,
        },
        "gate_table": list(gate_rows),
        "all_required_gates_pass": passed,
        "next_action": (
            "fit_all_eight_families_then_replay_exactly_one_open_a_selection"
            if passed
            else "stop_keep_other_roles_closed_and_revise_a_fit_recipe"
        ),
    }


def _projection_receipt_matches_protocol(
    receipt: Mapping[str, object],
    protocol: Mapping[str, object],
) -> bool:
    projection = protocol.get("projection_contract")
    if not isinstance(projection, Mapping):
        return False
    records = projection.get("ordered_decoders")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        return False
    expected_order = tuple(
        raw.get("node_name") for raw in records if isinstance(raw, Mapping)
    )
    expected_bases = {
        str(raw["node_name"]): raw["computational_mode_basis_sha256"]
        for raw in records
        if isinstance(raw, Mapping)
    }
    expected_means = {
        str(raw["node_name"]): raw["mean_bias_sha256"]
        for raw in records
        if isinstance(raw, Mapping)
    }
    expected_decoders = {
        str(raw["node_name"]): raw["decoder_basis_sha256"]
        for raw in records
        if isinstance(raw, Mapping)
    }
    if (
        len(expected_order) != 4
        or len(expected_bases) != len(expected_order)
        or len(expected_means) != len(expected_order)
        or len(expected_decoders) != len(expected_order)
    ):
        return False
    expected_rank = projection.get("concatenated_coordinate_width")
    decoder_shape = projection.get("concatenated_decoder_shape")
    if (
        expected_rank != 182
        or isinstance(decoder_shape, (str, bytes))
        or not isinstance(decoder_shape, Sequence)
        or tuple(decoder_shape) != (182, 640)
    ):
        return False
    for role in ("fit_projection", "held_projection"):
        value = receipt.get(role)
        if not isinstance(value, Mapping) or set(value) != _PROJECTION_METADATA_FIELDS:
            return False
        coordinate_hashes = value.get("coordinate_sha256_by_node")
        contribution_hashes = value.get("contribution_sha256_by_node")
        if (
            not isinstance(coordinate_hashes, Mapping)
            or set(coordinate_hashes) != set(expected_order)
            or not isinstance(contribution_hashes, Mapping)
            or set(contribution_hashes) != set(expected_order)
            or any(
                _SHA256.fullmatch(str(digest)) is None
                for digest in (
                    *coordinate_hashes.values(),
                    *contribution_hashes.values(),
                )
            )
            or _SHA256.fullmatch(str(value.get("target_sha256"))) is None
            or _SHA256.fullmatch(str(value.get("prediction_sha256"))) is None
            or _SHA256.fullmatch(str(value.get("affine_offset_sha256"))) is None
        ):
            return False
        diagnostics: list[float] = []
        for field in ("rmse", "target_rms", "nrmse", "max_abs_error"):
            raw = value.get(field)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0.0
            ):
                return False
            diagnostics.append(float(raw))
        rmse, target_rms, nrmse, _ = diagnostics
        if not math.isclose(
            nrmse,
            rmse / max(target_rms, 1e-30),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            return False
        if (
            value.get("projection_method")
            != "float64_affine_sum_svd_pseudoinverse_minimum_norm"
            or tuple(value.get("node_order", ())) != expected_order
            or value.get("basis_sha256_by_node") != expected_bases
            or value.get("mean_bias_sha256_by_node") != expected_means
            or value.get("decoder_basis_sha256_by_node")
            != expected_decoders
            or value.get("affine_offset_sha256")
            != projection.get("summed_mean_sha256")
            or value.get("combined_basis_rank") != expected_rank
            or type(value.get("observation_count")) is not int
            or int(value.get("observation_count", 0)) <= 0
            or value.get("residual_width") != decoder_shape[1]
            or value.get("offline_projection_only") is not True
            or value.get("runtime_parameter_count") != 0
            or value.get("runtime_macs_per_token") != 0
        ):
            return False
    return True


def _reported_split_sha256(
    receipt: Mapping[str, object],
    *,
    protocol_sha256: str,
    role: str,
) -> str:
    if role == "fit":
        aliases = receipt.get("fit_family_aliases")
        row_hash = receipt.get("fit_row_key_sha256")
        observations = receipt.get("fit_observations")
        sequences = receipt.get("fit_sequences")
    elif role == "held":
        aliases = (receipt.get("held_family_alias"),)
        row_hash = receipt.get("held_row_key_sha256")
        observations = receipt.get("held_observations")
        sequences = receipt.get("held_sequences")
    else:
        raise ValueError("reported split role must be fit or held")
    if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
        raise ValueError("reported split aliases are invalid")
    return _domain_sha256(
        _SPLIT_DOMAIN,
        {
            "protocol_sha256": _require_sha256(
                protocol_sha256, label="protocol"
            ),
            "capture_sha256": _require_sha256(
                receipt.get("capture_sha256"), label="capture"
            ),
            "fold_sha256": _require_sha256(
                receipt.get("protocol_fold_sha256"), label="fold"
            ),
            "authenticated_stream_sha256": _require_sha256(
                receipt.get("authenticated_stream_sha256"),
                label="authenticated tokenized stream",
            ),
            "role": role,
            "family_aliases": tuple(aliases),
            "row_key_sha256": _require_sha256(row_hash, label=f"{role} rows"),
            "observation_count": observations,
            "sequence_count": sequences,
            "fisher_normalization": (
                "equal_total_mass_per_training_family_from_native_virtual_gate_fisher"
                if role == "fit"
                else "native_virtual_gate_fisher_preserved_unnormalized"
            ),
        },
    )


def build_trajectory_correction_lofo_report(
    *,
    protocol: Mapping[str, object],
    authorization: Mapping[str, object],
    runtime: Mapping[str, object],
    fit_collection: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    resources: Mapping[str, object],
    decision: Mapping[str, object],
    source_model_unchanged: bool,
    layer10_unchanged: bool,
) -> dict[str, object]:
    """Build and validate one tensor-free hash-authenticated LOFO report."""

    frozen = validate_gemma3_l10_l17_trajectory_correction_protocol(protocol)
    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_SCHEMA,
        "format_version": (
            GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_FORMAT_VERSION
        ),
        "scientific_role": (
            "calibration_a_fit_adaptive_trajectory_correction_development"
        ),
        "heldout_confirmation": False,
        "protocol": frozen,
        "authorization": dict(authorization),
        "runtime": dict(runtime),
        "fit_collection": dict(fit_collection),
        "folds": [dict(value) for value in folds],
        "aggregate": dict(aggregate),
        "resources": dict(resources),
        "decision": dict(decision),
        "source_model_unchanged": source_model_unchanged,
        "layer10_unchanged": layer10_unchanged,
        "selection_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "serving_authorized": False,
        "latency_or_kernel_speed_claim": False,
        "safety": {
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_gradient_or_projection_tensors": False,
            "contains_model_or_candidate_weights": False,
            "source_safe": True,
        },
    }
    _reject_forbidden_output_fields(payload)
    report = {
        **payload,
        "report_sha256": _domain_sha256(_REPORT_DOMAIN, payload),
    }
    return validate_gemma3_l10_l17_trajectory_correction_lofo_report(report)


def _validate_fold_report(
    value: object,
    *,
    index: int,
    protocol: Mapping[str, object],
    source_runtime_catalog: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "fold_index",
        "fold_id",
        "held_family_alias",
        "training_family_aliases",
        "protocol_fold_sha256",
        "row_receipt",
        "correction_fit",
        "corrected_layer17_graph_sha256",
        "corrected_composition_graph_sha256",
        "corrected_lowering_sha256_by_node",
        "composition_receipt",
        "evaluation",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"fold {index} report fields are invalid")
    protocol_folds = protocol.get("folds")
    if isinstance(protocol_folds, (str, bytes)) or not isinstance(
        protocol_folds, Sequence
    ):
        raise TypeError("trajectory protocol folds are unavailable")
    expected = protocol_folds[index]
    if not isinstance(expected, Mapping):
        raise TypeError("trajectory protocol fold is invalid")
    if (
        value["fold_index"] != expected["fold_index"]
        or value["fold_id"] != expected["fold_id"]
        or value["held_family_alias"] != expected["held_family_alias"]
        or value["training_family_aliases"]
        != expected["training_family_aliases"]
        or value["protocol_fold_sha256"] != expected["artifact_sha256"]
    ):
        raise ValueError(f"fold {index} differs from frozen split")
    receipt = value.get("row_receipt")
    fit = value.get("correction_fit")
    lowerings = value.get("corrected_lowering_sha256_by_node")
    fit_lowerings = fit.get("lowering_sha256_by_node") if isinstance(fit, Mapping) else None
    fit_generators = fit.get("generator_plan_sha256_by_node") if isinstance(fit, Mapping) else None
    projection_contract = protocol.get("projection_contract")
    if not isinstance(projection_contract, Mapping):
        raise TypeError("trajectory projection contract is unavailable")
    decoder_records = projection_contract.get("ordered_decoders")
    if isinstance(decoder_records, (str, bytes)) or not isinstance(
        decoder_records, Sequence
    ):
        raise TypeError("trajectory decoder records are unavailable")
    expected_node_order = tuple(
        raw["node_name"] for raw in decoder_records if isinstance(raw, Mapping)
    )
    expected_decoder_hashes = {
        str(raw["node_name"]): raw["decoder_basis_sha256"]
        for raw in decoder_records
        if isinstance(raw, Mapping)
    }
    expected_mean_hashes = {
        str(raw["node_name"]): raw["mean_bias_sha256"]
        for raw in decoder_records
        if isinstance(raw, Mapping)
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != _ROW_RECEIPT_FIELDS
        or not _projection_receipt_matches_protocol(receipt, protocol)
        or receipt.get("held_family_excluded_from_projection_fit_and_generator_fit")
        is not True
        or receipt.get("fit_sequences") != 224
        or receipt.get("held_sequences") != 32
        or type(receipt.get("fit_observations")) is not int
        or int(receipt.get("fit_observations", 0)) <= 0
        or type(receipt.get("held_observations")) is not int
        or int(receipt.get("held_observations", 0)) <= 0
        or not isinstance(receipt.get("fit_projection"), Mapping)
        or not isinstance(receipt.get("held_projection"), Mapping)
        or receipt["fit_projection"].get("observation_count")
        != receipt.get("fit_observations")
        or receipt["held_projection"].get("observation_count")
        != receipt.get("held_observations")
        or not isinstance(fit, Mapping)
        or set(fit) != _CORRECTION_FIT_FIELDS
        or fit.get("parameter_count") != 163_094
        or fit.get("macs_per_token") != 160_352
        or fit.get("interaction_count") != 0
        or tuple(fit.get("node_order", ())) != expected_node_order
        or fit.get("graph_sha256")
        != value.get("corrected_layer17_graph_sha256")
        or fit.get("source_decoder_basis_sha256_by_node")
        != expected_decoder_hashes
        or fit.get("source_mean_bias_sha256_by_node")
        != expected_mean_hashes
        or not isinstance(lowerings, Mapping)
        or not isinstance(fit_lowerings, Mapping)
        or not isinstance(fit_generators, Mapping)
        or set(lowerings) != set(fit.get("node_order", ()))
        or set(fit_lowerings) != set(expected_node_order)
        or set(fit_generators) != set(expected_node_order)
        or fit_lowerings != lowerings
        or any(
            _SHA256.fullmatch(str(digest)) is None
            for digest in lowerings.values()
        )
        or any(
            _SHA256.fullmatch(str(digest)) is None
            for digest in fit_generators.values()
        )
    ):
        raise ValueError(f"fold {index} projection/refit receipt is invalid")
    if (
        receipt.get("protocol_fold_sha256") != expected["artifact_sha256"]
        or receipt.get("held_family_alias") != expected["held_family_alias"]
        or tuple(receipt.get("fit_family_aliases", ()))
        != tuple(expected["training_family_aliases"])
        or receipt.get("fit_split_sha256")
        != _reported_split_sha256(
            receipt,
            protocol_sha256=str(protocol["artifact_sha256"]),
            role="fit",
        )
        or receipt.get("held_split_sha256")
        != _reported_split_sha256(
            receipt,
            protocol_sha256=str(protocol["artifact_sha256"]),
            role="held",
        )
    ):
        raise ValueError(f"fold {index} row membership receipt drifted")
    _require_sha256(
        value["corrected_layer17_graph_sha256"],
        label=f"fold {index} Layer17 graph",
    )
    _require_sha256(
        value["corrected_composition_graph_sha256"],
        label=f"fold {index} composition graph",
    )
    expected_composition_receipt = _build_corrected_composition_receipt(
        source_runtime_catalog=source_runtime_catalog,
        corrected_layer17_graph_sha256=str(
            value["corrected_layer17_graph_sha256"]
        ),
        corrected_layer17_lowering_sha256_by_node={
            str(name): str(digest) for name, digest in lowerings.items()
        },
        corrected_composition_graph_sha256=str(
            value["corrected_composition_graph_sha256"]
        ),
    )
    if _canonical_json_bytes(value.get("composition_receipt")) != (
        _canonical_json_bytes(expected_composition_receipt)
    ):
        raise ValueError(f"fold {index} corrected composition lineage drifted")
    _validate_fold_evaluation(value["evaluation"], label=f"fold {index}")
    return value


def validate_gemma3_l10_l17_trajectory_correction_lofo_report(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on source leakage, lineage drift, or metric tampering."""

    if not isinstance(raw, Mapping) or set(raw) != _REPORT_FIELDS:
        raise ValueError("trajectory LOFO report fields are invalid")
    _reject_forbidden_output_fields(raw)
    if (
        raw["schema"] != GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_SCHEMA
        or raw["format_version"]
        != GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_FORMAT_VERSION
        or raw["scientific_role"]
        != "calibration_a_fit_adaptive_trajectory_correction_development"
        or raw["heldout_confirmation"] is not False
        or raw["source_model_unchanged"] is not True
        or raw["layer10_unchanged"] is not True
        or any(
            raw[field] is not False
            for field in (
                "selection_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
                "full_model_compiled",
                "serving_authorized",
                "latency_or_kernel_speed_claim",
            )
        )
        or raw["full_model_logits_scored"] is not True
    ):
        raise ValueError("trajectory LOFO claim boundary is invalid")
    protocol_raw = raw.get("protocol")
    if not isinstance(protocol_raw, Mapping):
        raise TypeError("trajectory LOFO protocol is unavailable")
    protocol = validate_gemma3_l10_l17_trajectory_correction_protocol(
        protocol_raw
    )
    if protocol.get("artifact_sha256") != _EXPECTED_TRAJECTORY_PROTOCOL_SHA256:
        raise ValueError("trajectory LOFO protocol is not final frozen v2")
    authorization = raw.get("authorization")
    if not isinstance(authorization, Mapping) or set(authorization) != (
        _AUTHORIZATION_FIELDS
    ):
        raise ValueError("trajectory LOFO authorization fields are invalid")
    fit_authority = authorization.get("fit_authority")
    if not isinstance(fit_authority, Mapping):
        raise TypeError("trajectory LOFO fit authority is unavailable")
    validate_gemma3_layer17_family_lofo_authority_metadata(fit_authority)
    source_authority = protocol.get("source_authority")
    runtime = raw.get("runtime")
    bundle_binding = authorization.get("bundle")
    if (
        not isinstance(source_authority, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(bundle_binding, Mapping)
    ):
        raise TypeError("trajectory report source/runtime binding is unavailable")
    if set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("trajectory LOFO runtime fields are invalid")
    model_contract = source_authority.get("model")
    composition_contract = source_authority.get("composition_bundle")
    layer10_contract = source_authority.get("layer10")
    layer17_contract = source_authority.get("layer17_decoder_source")
    fit_contract = source_authority.get("calibration_a_fit")
    if not all(
        isinstance(value, Mapping)
        for value in (
            model_contract,
            composition_contract,
            layer10_contract,
            layer17_contract,
            fit_contract,
        )
    ):
        raise TypeError("trajectory report frozen authority is incomplete")
    assert isinstance(model_contract, Mapping)
    assert isinstance(composition_contract, Mapping)
    assert isinstance(layer10_contract, Mapping)
    assert isinstance(layer17_contract, Mapping)
    assert isinstance(fit_contract, Mapping)
    fit_corpus = fit_authority.get("corpus")
    fit_protocol = fit_authority.get("protocol")
    if not isinstance(fit_corpus, Mapping) or not isinstance(
        fit_protocol, Mapping
    ):
        raise TypeError("trajectory report fit authority binding is incomplete")
    evaluation_contract = protocol.get("evaluation_contract")
    randomness = runtime.get("randomness")
    expected_recipe_sha256 = _domain_sha256(
        _RANDOMNESS_RECIPE_DOMAIN,
        _RANDOMNESS_RECIPE,
    )
    if (
        not isinstance(evaluation_contract, Mapping)
        or evaluation_contract.get("randomness_policy")
        != "seed_and_recipe_committed_before_execution"
        or not isinstance(randomness, Mapping)
        or any(
            randomness.get(key) != value
            for key, value in _RANDOMNESS_RECIPE.items()
        )
        or randomness.get("recipe_sha256") != expected_recipe_sha256
        or randomness.get("torch_manual_seed_applied") is not True
        or randomness.get("torch_cuda_manual_seed_all_applied")
        != str(runtime.get("device", "")).startswith("cuda")
    ):
        raise ValueError("trajectory deterministic randomness recipe drifted")
    if (
        authorization.get("authorization_kind")
        != "frozen_composition_then_fit_only_family_lofo"
        or authorization.get("authorization_completed_before_fit_open") is not True
        or authorization.get("protocol_sha256") != protocol["artifact_sha256"]
        or authorization.get("fit_authority_sha256")
        != fit_authority.get("authority_sha256")
        or authorization.get("fit_opened") is not True
        or authorization.get("heldout_confirmation") is not False
        or authorization.get("serving_authorized") is not False
        or authorization.get("source_safe") is not True
        or any(
            authorization.get(field) is not False
            for field in (
                "selection_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
            )
        )
        or runtime.get("model_id") != model_contract.get("model_id")
        or runtime.get("local_files_only") is not True
        or runtime.get("requested_revision")
        != model_contract.get("requested_revision")
        or runtime.get("model_fingerprint")
        != model_contract.get("adapter_model_fingerprint")
        or bundle_binding.get("model_fingerprint")
        != model_contract.get("adapter_model_fingerprint")
        or bundle_binding.get("bundle_file_sha256")
        != composition_contract.get("tensor_file_sha256")
        or bundle_binding.get("composition_payload_sha256")
        != composition_contract.get("composition_payload_sha256")
        or bundle_binding.get("combined_edgeless_graph_sha256")
        != composition_contract.get("combined_edgeless_graph_sha256")
        or bundle_binding.get("combined_primary_graph_sha256")
        != composition_contract.get("combined_primary_graph_sha256")
        or bundle_binding.get("layer10_candidate_tensor_file_sha256")
        != layer10_contract.get("candidate_tensor_file_sha256")
        or bundle_binding.get("layer10_candidate_scientific_payload_sha256")
        != layer10_contract.get("candidate_scientific_payload_sha256")
        or bundle_binding.get("layer17_candidate_tensor_file_sha256")
        != layer17_contract.get("candidate_tensor_file_sha256")
        or bundle_binding.get("layer17_candidate_scientific_payload_sha256")
        != layer17_contract.get("candidate_scientific_payload_sha256")
        or bundle_binding.get("layer17_edgeless_graph_sha256")
        != layer17_contract.get("source_edgeless_graph_sha256")
        or fit_protocol.get("protocol_artifact_sha256")
        != fit_contract.get("source_lofo_protocol_sha256")
        or fit_protocol.get("fit_membership_sha256")
        != fit_contract.get("fit_membership_sha256")
        or fit_protocol.get("family_alias_mapping_sha256")
        != fit_contract.get("family_alias_mapping_sha256")
        or fit_corpus.get("corpus_artifact_sha256")
        != fit_contract.get("corpus_artifact_sha256")
        or fit_corpus.get("tokenizer_contract_sha256")
        != fit_contract.get("tokenizer_contract_sha256")
        or fit_corpus.get("fit_manifest_sha256")
        != fit_contract.get("fit_manifest_sha256")
        or fit_corpus.get("fit_role_file_sha256")
        != fit_contract.get("fit_source_file_sha256")
        or tuple(fit_corpus.get("block_labels", ()))
        != tuple(fit_contract.get("family_aliases", ()))
    ):
        raise ValueError("trajectory LOFO authorization boundary drifted")
    source_runtime_catalog = _validate_source_runtime_catalog(
        authorization.get("source_runtime_catalog"),
        protocol=protocol,
        bundle_binding=bundle_binding,
    )
    source_runtime_layer10 = source_runtime_catalog.get("layer10")
    source_runtime_layer17 = source_runtime_catalog.get("layer17")
    if not isinstance(source_runtime_layer10, Mapping) or not isinstance(
        source_runtime_layer17,
        Mapping,
    ):
        raise TypeError("source runtime layer catalogs are unavailable")
    fit_collection = raw.get("fit_collection")
    if not isinstance(fit_collection, Mapping) or set(fit_collection) != (
        _FIT_COLLECTION_FIELDS
    ):
        raise ValueError("trajectory LOFO fit collection fields are invalid")
    materialization = fit_collection.get("materialization")
    if not isinstance(materialization, Mapping):
        raise TypeError("trajectory LOFO materialization is unavailable")
    validate_gemma3_layer17_family_lofo_materialization_metadata(
        materialization
    )
    if materialization.get("authority_sha256") != authorization.get(
        "fit_authority_sha256"
    ):
        raise ValueError("trajectory materialization authority drifted")
    tokenization = materialization.get("tokenization")
    capture = fit_collection.get("capture")
    compiled_keep_audit = fit_collection.get("compiled_keep_replay_audit")
    target_contract = protocol.get("target_contract")
    if (
        not isinstance(tokenization, Mapping)
        or fit_collection.get("authenticated_stream_sha256")
        != tokenization.get("stream_catalog_sha256")
        or fit_collection.get("capture_count") != 1
        or fit_collection.get("captured_examples") != _EXPECTED_EXAMPLES
        or fit_collection.get("captured_sequences") != _EXPECTED_EXAMPLES
        or fit_collection.get("model_rows_recollected_per_fold") is not False
        or not isinstance(capture, Mapping)
        or capture.get("capture_sha256") is None
        or not isinstance(compiled_keep_audit, Mapping)
        or not isinstance(target_contract, Mapping)
        or compiled_keep_audit.get("construction")
        != target_contract.get("compiled_keep_pass")
        or compiled_keep_audit.get("capture")
        != target_contract.get("compiled_keep_capture")
        or compiled_keep_audit.get("exact_compact_replay_used_as_target")
        is not True
        or compiled_keep_audit.get("algebraic_reference_used_as_target")
        is not False
        or compiled_keep_audit.get("runtime_deletion_operator_equivalent")
        is not True
        or compiled_keep_audit.get("maximum_algebraic_equivalence_rmse")
        != target_contract.get("maximum_algebraic_equivalence_rmse")
        or compiled_keep_audit.get(
            "maximum_algebraic_equivalence_max_abs_difference"
        )
        != target_contract.get(
            "maximum_algebraic_equivalence_max_abs_difference"
        )
    ):
        raise ValueError("trajectory capture/compiled-keep provenance drifted")
    capture_layer10 = capture.get("layer10")
    capture_layer17 = capture.get("layer17")
    capture_audit = capture.get("compact_retained_numerical_audit")
    capture_alignment = capture.get("alignment")
    compact_executor = (
        capture_layer17.get("compact_executor")
        if isinstance(capture_layer17, Mapping)
        else None
    )
    if (
        not isinstance(capture_layer10, Mapping)
        or not isinstance(compact_executor, Mapping)
        or not isinstance(capture_audit, Mapping)
        or not isinstance(capture_alignment, Mapping)
        or capture.get("schema")
        != GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_SCHEMA
        or capture.get("format_version")
        != GEMMA3_LAYER17_TRAJECTORY_ROW_CAPTURE_FORMAT_VERSION
        or capture.get("scientific_role")
        != "paired_native_and_layer10_compiled_layer17_rows"
        or capture.get("source_safe") is not True
        or capture.get("contains_tensors") is not False
        or capture.get("contains_prompt_text") is not False
        or capture.get("contains_prompt_identities") is not False
        or capture.get("contains_token_ids") is not False
        or capture.get("capture_sha256") != _capture_metadata_sha256(capture)
        or capture.get("condition") != "generated"
        or tuple(capture.get("affected_layer_ordinals", ())) != (10,)
        or capture.get("model_fingerprint")
        != runtime.get("model_fingerprint")
        or capture_alignment.get("sequences") != _EXPECTED_EXAMPLES
        or capture_alignment.get("observations")
        != fit_collection.get("captured_observations")
        or capture_layer10.get("graph_sha256")
        != source_runtime_layer10.get("graph_sha256")
        or tuple(capture_layer10.get("traversal_order", ()))
        != tuple(source_runtime_layer10.get("traversal_order", ()))
        or tuple(capture_layer10.get("ordered_lowering_sha256s", ()))
        != tuple(
            source_runtime_layer10["lowering_sha256_by_node"][name]  # type: ignore[index]
            for name in source_runtime_layer10.get("traversal_order", ())
        )
        or compact_executor.get("graph_sha256")
        != source_runtime_layer17.get("graph_sha256")
        or tuple(compact_executor.get("traversal_order", ()))
        != tuple(source_runtime_layer17.get("traversal_order", ()))
        or tuple(compact_executor.get("ordered_lowering_sha256s", ()))
        != tuple(
            source_runtime_layer17["lowering_sha256_by_node"][name]  # type: ignore[index]
            for name in source_runtime_layer17.get("traversal_order", ())
        )
        or compact_executor.get("interaction_count") != 0
        or tuple(compact_executor.get("affected_layer_ordinals", ()))
        != (17,)
        or capture_layer17.get("selection_sha256")
        != source_runtime_layer17.get("selection_sha256")
        or tuple(capture_layer17.get("fragment_ids", ()))
        != tuple(source_runtime_layer17.get("fragment_ids", ()))
        or capture_alignment.get("fragment_count")
        != len(tuple(source_runtime_layer17.get("fragment_ids", ())))
        or capture_layer17.get("teacher_role")
        != "native_layer17_fragment_residual_contribution_on_each_trajectory"
        or capture_layer17.get("input_roles")
        != {
            "native": "native_model_layer17_normalized_mlp_input",
            "compiled": (
                "layer17_normalized_mlp_input_after_frozen_layer10_generated_graph"
            ),
        }
        or capture_layer17.get("full_output_roles")
        != {
            "native": "native_model_layer17_full_mlp_output",
            "compiled": (
                "native_layer17_full_mlp_output_after_frozen_layer10_generated_graph"
            ),
        }
        or capture_layer17.get("compact_retained_output_role")
        != "authenticated_layer17_compact_mlp_output_on_layer10_compiled_input"
        or capture_layer17.get("algebraic_compact_retained_audit_role")
        != (
            "compiled_full_minus_selected_compiled_contributions_"
            "numerical_audit_only"
        )
        or capture_layer17.get("a3_target_role")
        != "native_full_minus_exact_authenticated_compact_retained_output"
        or capture_audit.get("role")
        != (
            "compiled_full_minus_selected_compiled_contributions_"
            "numerical_audit_only"
        )
        or capture_audit.get("difference_definition")
        != (
            "exact_compact_retained_minus_compiled_full_minus_"
            "selected_compiled_contributions"
        )
        or compiled_keep_audit.get("layer17_compact_graph_sha256")
        != compact_executor.get("graph_sha256")
        or compiled_keep_audit.get("ordered_layer17_compact_lowering_sha256s")
        != compact_executor.get("ordered_lowering_sha256s")
        or compiled_keep_audit.get("algebraic_equivalence_rmse")
        != capture_audit.get("rms_difference")
        or compiled_keep_audit.get(
            "algebraic_equivalence_max_abs_difference"
        )
        != capture_audit.get("max_abs_difference")
        or fit_collection.get("a3_target_construction")
        != target_contract.get("raw_target_formula")
    ):
        raise ValueError("trajectory capture executable lineage drifted")
    projection_contract = protocol.get("projection_contract")
    if (
        not isinstance(projection_contract, Mapping)
        or fit_collection.get("decoder_span_sha256")
        != projection_contract.get("decoder_span_sha256")
        or fit_collection.get("summed_mean_sha256")
        != projection_contract.get("summed_mean_sha256")
    ):
        raise ValueError("trajectory affine decoder-span lineage drifted")
    audit_pass = (
        _finite(
            compiled_keep_audit.get("algebraic_equivalence_rmse"),
            label="compiled-keep algebraic RMSE",
        )
        <= float(target_contract["maximum_algebraic_equivalence_rmse"])
        and _finite(
            compiled_keep_audit.get(
                "algebraic_equivalence_max_abs_difference"
            ),
            label="compiled-keep algebraic max abs",
        )
        <= float(
            target_contract[
                "maximum_algebraic_equivalence_max_abs_difference"
            ]
        )
    )
    folds_raw = raw.get("folds")
    if isinstance(folds_raw, (str, bytes)) or not isinstance(
        folds_raw, Sequence
    ) or len(folds_raw) != _EXPECTED_FAMILIES:
        raise ValueError("trajectory LOFO requires exactly eight folds")
    folds = tuple(
        _validate_fold_report(
            value,
            index=index,
            protocol=protocol,
            source_runtime_catalog=source_runtime_catalog,
        )
        for index, value in enumerate(folds_raw)
    )
    captured_observations = fit_collection.get("captured_observations")
    if (
        type(captured_observations) is not int
        or captured_observations <= 0
        or any(
            value["row_receipt"].get("fit_observations")
            + value["row_receipt"].get("held_observations")
            != captured_observations
            for value in folds
        )
        or sum(
            int(value["row_receipt"]["held_observations"])
            for value in folds
        )
        != captured_observations
    ):
        raise ValueError("trajectory fold observation coverage drifted")
    if any(
        value["row_receipt"].get("capture_sha256")
        != capture.get("capture_sha256")
        or value["row_receipt"].get("authenticated_stream_sha256")
        != fit_collection.get("authenticated_stream_sha256")
        for value in folds
    ):
        raise ValueError("trajectory fold capture/stream binding drifted")
    evaluations = tuple(value["evaluation"] for value in folds)
    aggregate = aggregate_trajectory_correction_lofo_folds(evaluations)  # type: ignore[arg-type]
    if _canonical_json_bytes(raw["aggregate"]) != _canonical_json_bytes(
        aggregate
    ):
        raise ValueError("trajectory LOFO aggregate was not reproduced")
    resources = raw.get("resources")
    candidate = protocol.get("candidate_contract")
    if (
        not isinstance(resources, Mapping)
        or not isinstance(candidate, Mapping)
        or _canonical_json_bytes(resources)
        != _canonical_json_bytes(candidate.get("exact_resources"))
        or _canonical_json_bytes(resources)
        != _canonical_json_bytes(_EXPECTED_EXACT_RESOURCES)
    ):
        raise ValueError("trajectory LOFO exact resources drifted")
    decision = evaluate_trajectory_correction_lofo_gates(
        protocol=protocol,
        fold_evaluations=evaluations,  # type: ignore[arg-type]
        aggregate=aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=True,
        compact_replay_algebraic_equivalence_audit=audit_pass,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )
    if _canonical_json_bytes(raw["decision"]) != _canonical_json_bytes(
        decision
    ):
        raise ValueError("trajectory LOFO decision was not reproduced")
    safety = raw.get("safety")
    if not isinstance(safety, Mapping) or safety != {
        "contains_prompt_text": False,
        "contains_prompt_identities": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_gradient_or_projection_tensors": False,
        "contains_model_or_candidate_weights": False,
        "source_safe": True,
    }:
        raise ValueError("trajectory LOFO safety receipt is invalid")
    supplied = _require_sha256(raw["report_sha256"], label="LOFO report")
    payload = {key: raw[key] for key in _REPORT_FIELDS if key != "report_sha256"}
    if supplied != _domain_sha256(_REPORT_DOMAIN, payload):
        raise ValueError("trajectory LOFO report hash mismatch")
    return json.loads(_canonical_json_bytes(raw).decode("utf-8"))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_gemma3_l10_l17_trajectory_correction_lofo_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Exclusively publish one validated report; never overwrite evidence."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite trajectory LOFO report")
    validated = validate_gemma3_l10_l17_trajectory_correction_lofo_report(
        report
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(validated) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                "refusing to overwrite trajectory LOFO report"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_gemma3_l10_l17_trajectory_correction_lofo_report(
    path: Path | str,
) -> dict[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("trajectory LOFO report is not strict JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("trajectory LOFO report must contain one object")
    return validate_gemma3_l10_l17_trajectory_correction_lofo_report(raw)


def _merge_corrected_composition_lowerings(
    composed_graph: ModalGeneratorGraphPlan,
    *,
    layer10_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    corrected_layer17_lowerings_by_node: Mapping[str, ModalGeneratorLowering],
) -> tuple[ModalGeneratorLowering, ...]:
    """Merge new L17 generators with byte-identical L10 lowerings in order."""

    composed_graph.validate_integrity()
    supplied = {
        **dict(layer10_lowerings_by_node),
        **dict(corrected_layer17_lowerings_by_node),
    }
    if set(supplied) != set(composed_graph.traversal_order):
        raise ValueError("corrected composition lowering catalog is incomplete")
    ordered = tuple(supplied[name] for name in composed_graph.traversal_order)
    if any(
        node.weights.artifact_sha256 != lowering.graph_weights.artifact_sha256
        for node, lowering in zip(composed_graph.nodes, ordered, strict=True)
    ):
        raise ValueError("corrected graph nodes do not bind merged lowerings")
    return ordered


def _fold_catalog(protocol: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = protocol.get("folds")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("trajectory protocol fold catalog is unavailable")
    folds = tuple(raw)
    if len(folds) != _EXPECTED_FAMILIES or any(
        not isinstance(value, Mapping) for value in folds
    ):
        raise ValueError("trajectory protocol requires exactly eight folds")
    return folds  # type: ignore[return-value]


def run_gemma3_l10_l17_trajectory_correction_lofo(
    *,
    revision: str,
    output: Path | str = (
        DEFAULT_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_OUTPUT
    ),
    composition_bundle_path: Path | str = DEFAULT_COMPOSITION_BUNDLE_PATH,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run the sealed A-fit-only eight-family trajectory-correction LOFO."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite trajectory LOFO report")

    _progress("preflight: authenticate frozen composition before A-fit access")
    bundle, authority, protocol, authorization = (
        _authenticate_before_fit_access(
            bundle_path=composition_bundle_path,
            corpus_receipt_path=corpus_receipt_path,
            corpus_artifact_path=corpus_artifact_path,
            fit_input_path=fit_input_path,
        )
    )
    if (
        getattr(bundle, "model_id", None) != model_id
        or getattr(bundle, "requested_revision", None) != revision
    ):
        raise ValueError("requested model identity differs from frozen bundle")
    device = resolve_torch_device(device_name)
    randomness_receipt = _randomness_recipe_receipt(device)
    layer10_graph, layer17_graph, layer10_lowerings, layer17_lowerings = (
        _source_lowering_maps(bundle)
    )
    _, fragment_by_node = _validate_source_decoder_contract(
        layer17_graph,
        layer17_lowerings,
        protocol,
    )
    layer10_graph_sha256 = layer10_graph.artifact_sha256
    layer10_lowering_sha256s = tuple(
        layer10_lowerings[name].artifact_sha256
        for name in layer10_graph.traversal_order
    )

    fragment_plans = {
        lowering.fragment_plan.artifact_sha256: lowering.fragment_plan
        for lowering in layer17_lowerings.values()
    }
    if len(fragment_plans) != 1:
        raise ValueError("Layer17 source lowerings use different fragment plans")
    fragment_plan = next(iter(fragment_plans.values()))
    selection: SameLayerFragmentSelection = (
        select_top_fisher_same_layer_fragments(
            fragment_plan,
            count=4,
            minimum_fragment_modes=32,
            layer_ordinal=17,
        )
    )
    _validate_frozen_selection(selection)
    if tuple(selection.fragment_ids) != tuple(fragment_by_node.values()):
        raise ValueError("Layer17 selected fragment order differs from protocol")
    bundle_binding = getattr(bundle, "binding", None)
    primary_graph = getattr(bundle, "primary", None)
    if not isinstance(bundle_binding, Mapping) or not isinstance(
        primary_graph,
        ModalGeneratorGraphPlan,
    ):
        raise TypeError("authenticated composition runtime is unavailable")
    source_runtime_catalog = _build_source_runtime_catalog(
        bundle_binding=bundle_binding,
        primary_graph=primary_graph,
        layer10_graph=layer10_graph,
        layer17_graph=layer17_graph,
        layer10_lowerings_by_node=layer10_lowerings,
        layer17_lowerings_by_node=layer17_lowerings,
        selection=selection,
    )
    if source_runtime_catalog.get("catalog_sha256") != (
        _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256
    ):
        raise ValueError("authenticated source runtime catalog is not frozen")
    authorization = {
        **authorization,
        "source_runtime_catalog": source_runtime_catalog,
    }
    _reject_forbidden_output_fields(authorization, path="authorization")
    leaf_site = selection.execution_order[0].input_site

    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned local Gemma checkpoint")
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
    model_fingerprint = adapter.model_fingerprint()
    if (
        model_fingerprint != getattr(bundle, "primary").model_fingerprint
        or model_fingerprint != selection.source_model_sha256
    ):
        raise ValueError("live Gemma fingerprint differs from frozen composition")

    _progress("tokenize: materialize calibration_a_fit only")
    raw_blocks, materialization = materialize_gemma3_layer17_family_lofo(
        authority,
        tokenizer,
    )
    validate_gemma3_layer17_family_lofo_materialization_metadata(
        materialization
    )
    _reject_lofo_authority_fields(materialization, path="materialization")
    authority_blocks = _family_blocks(raw_blocks)
    blocks = _blocks_to_device(authority_blocks, device)
    all_batches = tuple(batch for _, values in blocks for batch in values)
    family_alias_by_example = {
        example_id: alias
        for alias, values in blocks
        for batch in values
        for example_id in batch.example_ids or ()
    }
    tokenization = materialization.get("tokenization")
    if not isinstance(tokenization, Mapping):
        raise TypeError("A-fit tokenization receipt is unavailable")
    stream_sha256 = _require_sha256(
        tokenization.get("stream_catalog_sha256"),
        label="A-fit tokenized stream",
    )

    layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        tuple(
            layer10_lowerings[name] for name in layer10_graph.traversal_order
        ),
    )
    layer17_source_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer17_graph,
        tuple(
            layer17_lowerings[name] for name in layer17_graph.traversal_order
        ),
    )
    frozen_composition_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        getattr(bundle, "primary"),
        _ordered_restored_lowerings(
            getattr(bundle, "primary"),
            {
                node.name: lowering
                for node, lowering in zip(
                    getattr(bundle, "primary").nodes,
                    getattr(bundle, "lowerings"),
                    strict=True,
                )
            },
        ),
    )
    _progress("rows: capture native and frozen-Layer10 Layer17 trajectories")
    row_pair = capture_gemma3_layer17_native_and_layer10_rows(
        adapter,
        all_batches,
        selection=selection,
        leaf_activation_site=leaf_site,
        layer10_executor=layer10_executor,
        layer17_executor=layer17_source_executor,
    )
    expected_sequences = sum(batch.batch_size for batch in all_batches)
    if (
        row_pair.native_rows.sequences != expected_sequences
        or row_pair.compiled_rows.sequences != expected_sequences
    ):
        raise RuntimeError("trajectory capture sequence accounting drifted")
    capture_metadata = row_pair.metadata()
    if capture_metadata.get("contains_tensors") is not False:
        raise RuntimeError("trajectory capture metadata is not source safe")
    raw_compact_audit = capture_metadata.get(
        "compact_retained_numerical_audit"
    )
    target_contract = protocol.get("target_contract")
    if not isinstance(raw_compact_audit, Mapping) or not isinstance(
        target_contract, Mapping
    ):
        raise TypeError("compact-retained capture/protocol audit is unavailable")
    compiled_keep_audit = {
        "construction": target_contract["compiled_keep_pass"],
        "capture": target_contract["compiled_keep_capture"],
        "runtime_deletion_operator_equivalent": target_contract[
            "compiled_keep_replay_matches_runtime_deletion_operator"
        ],
        "layer17_compact_graph_sha256": (
            row_pair.layer17_compact_graph_sha256
        ),
        "ordered_layer17_compact_lowering_sha256s": (
            row_pair.ordered_layer17_compact_lowering_sha256s
        ),
        "algebraic_equivalence_rmse": raw_compact_audit["rms_difference"],
        "algebraic_equivalence_max_abs_difference": raw_compact_audit[
            "max_abs_difference"
        ],
        "maximum_algebraic_equivalence_rmse": target_contract[
            "maximum_algebraic_equivalence_rmse"
        ],
        "maximum_algebraic_equivalence_max_abs_difference": target_contract[
            "maximum_algebraic_equivalence_max_abs_difference"
        ],
        "algebraic_reference_used_as_target": False,
        "exact_compact_replay_used_as_target": True,
        "contains_tensors": False,
    }
    fit_view = _build_trajectory_correction_fit_view(row_pair)
    captured_sequences = fit_view.sequences
    captured_observations = fit_view.observations
    del row_pair

    fold_reports: list[dict[str, object]] = []
    fold_evaluations: list[Mapping[str, object]] = []
    held_batches_by_alias = dict(blocks)
    for index, fold in enumerate(_fold_catalog(protocol)):
        held = str(fold["held_family_alias"])
        training = tuple(fold["training_family_aliases"])  # type: ignore[arg-type]
        _progress(
            f"fold {index + 1}/{_EXPECTED_FAMILIES}: fit seven, hold {held}"
        )
        fit_rows, held_rows, row_receipt = (
            _build_trajectory_correction_fold_rows_from_fit_view(
                fit_view,
                family_alias_by_example=family_alias_by_example,
                training_family_aliases=training,
                held_family_alias=held,
                fold_sha256=str(fold["artifact_sha256"]),
                protocol_sha256=str(protocol["artifact_sha256"]),
                authenticated_stream_sha256=stream_sha256,
                source_graph=layer17_graph,
                source_lowerings_by_node=layer17_lowerings,
            )
        )
        correction = fit_frozen_basis_coordinate_generators(
            fit_rows,
            held_rows,
            source_graph=layer17_graph,
            source_lowerings_by_node=layer17_lowerings,
            fit_split_sha256=str(row_receipt["fit_split_sha256"]),
            eval_split_sha256=str(row_receipt["held_split_sha256"]),
            generator_rank=_GENERATOR_RANK,
            ridge=_RIDGE,
        )
        if (
            correction.graph_plan.parameter_count != layer17_graph.parameter_count
            or correction.graph_plan.macs_per_token != layer17_graph.macs_per_token
            or correction.source_decoder_basis_sha256_by_node
            != {
                name: layer17_lowerings[
                    name
                ].computational_mode_basis.decoder_basis_sha256
                for name in layer17_graph.traversal_order
            }
            or correction.source_mean_bias_sha256_by_node
            != {
                name: layer17_lowerings[
                    name
                ].computational_mode_basis.mean_bias_sha256
                for name in layer17_graph.traversal_order
            }
        ):
            raise RuntimeError("fold correction escaped fixed Layer17 capacity")
        corrected_composition = replace_layer_nodes_in_composed_graph(
            getattr(bundle, "primary"),
            correction.graph_plan,
            layer_ordinal=17,
        )
        if (
            corrected_composition.parameter_count != 295_129
            or corrected_composition.macs_per_token != 289_600
            or corrected_composition.interactions
            != getattr(bundle, "primary").interactions
        ):
            raise RuntimeError("corrected composition resources/topology drifted")
        corrected_composition_lowerings = (
            _merge_corrected_composition_lowerings(
                corrected_composition,
                layer10_lowerings_by_node=layer10_lowerings,
                corrected_layer17_lowerings_by_node=(
                    correction.lowerings_by_node
                ),
            )
        )
        corrected_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            correction.graph_plan,
            tuple(
                correction.lowerings_by_node[name]
                for name in correction.graph_plan.traversal_order
            ),
        )
        corrected_composition_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            corrected_composition,
            corrected_composition_lowerings,
        )
        _progress(f"fold {index + 1}/{_EXPECTED_FAMILIES}: score held logits")
        evaluation = score_trajectory_correction_fold(
            adapter=adapter,
            layer10_executor=layer10_executor,
            corrected_layer17_executor=corrected_layer17_executor,
            frozen_composition_executor=frozen_composition_executor,
            corrected_composition_executor=corrected_composition_executor,
            batches=held_batches_by_alias[held],
        )
        fold_evaluations.append(evaluation)
        corrected_lowering_sha256_by_node = {
            name: correction.lowerings_by_node[name].artifact_sha256
            for name in correction.graph_plan.traversal_order
        }
        composition_receipt = _build_corrected_composition_receipt(
            source_runtime_catalog=source_runtime_catalog,
            corrected_layer17_graph_sha256=(
                correction.graph_plan.artifact_sha256
            ),
            corrected_layer17_lowering_sha256_by_node=(
                corrected_lowering_sha256_by_node
            ),
            corrected_composition_graph_sha256=(
                corrected_composition.artifact_sha256
            ),
        )
        fold_reports.append(
            {
                "fold_index": fold["fold_index"],
                "fold_id": fold["fold_id"],
                "held_family_alias": held,
                "training_family_aliases": list(training),
                "protocol_fold_sha256": fold["artifact_sha256"],
                "row_receipt": row_receipt,
                "correction_fit": correction.metadata(),
                "corrected_layer17_graph_sha256": (
                    correction.graph_plan.artifact_sha256
                ),
                "corrected_composition_graph_sha256": (
                    corrected_composition.artifact_sha256
                ),
                "corrected_lowering_sha256_by_node": (
                    corrected_lowering_sha256_by_node
                ),
                "composition_receipt": composition_receipt,
                "evaluation": evaluation,
            }
        )
        del (
            fit_rows,
            held_rows,
            correction,
            corrected_composition,
            corrected_composition_lowerings,
            corrected_layer17_executor,
            corrected_composition_executor,
        )

    source_model_unchanged = adapter.model_fingerprint() == model_fingerprint
    layer10_unchanged = (
        layer10_graph.artifact_sha256 == layer10_graph_sha256
        and tuple(
            layer10_lowerings[name].artifact_sha256
            for name in layer10_graph.traversal_order
        )
        == layer10_lowering_sha256s
    )
    if not source_model_unchanged or not layer10_unchanged:
        raise RuntimeError("trajectory LOFO mutated frozen source state")
    aggregate = aggregate_trajectory_correction_lofo_folds(fold_evaluations)
    decision = evaluate_trajectory_correction_lofo_gates(
        protocol=protocol,
        fold_evaluations=fold_evaluations,
        aggregate=aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=all(
            _projection_receipt_matches_protocol(
                value["row_receipt"], protocol  # type: ignore[arg-type]
            )
            for value in fold_reports
        ),
        compact_replay_algebraic_equivalence_audit=(
            compiled_keep_audit["exact_compact_replay_used_as_target"] is True
            and float(compiled_keep_audit["algebraic_equivalence_rmse"])
            <= float(
                compiled_keep_audit["maximum_algebraic_equivalence_rmse"]
            )
            and float(
                compiled_keep_audit[
                    "algebraic_equivalence_max_abs_difference"
                ]
            )
            <= float(
                compiled_keep_audit[
                    "maximum_algebraic_equivalence_max_abs_difference"
                ]
            )
        ),
        source_model_unchanged=source_model_unchanged,
        layer10_unchanged=layer10_unchanged,
    )
    projection_contract = protocol["projection_contract"]
    assert isinstance(projection_contract, Mapping)
    report = build_trajectory_correction_lofo_report(
        protocol=protocol,
        authorization=authorization,
        runtime={
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": model_fingerprint,
            "device": str(device),
            "dtype": dtype,
            "local_files_only": True,
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
            "randomness": randomness_receipt,
        },
        fit_collection={
            "materialization": materialization,
            "authenticated_stream_sha256": stream_sha256,
            "capture": capture_metadata,
            "compiled_keep_replay_audit": compiled_keep_audit,
            "capture_count": 1,
            "captured_examples": _EXPECTED_EXAMPLES,
            "captured_sequences": captured_sequences,
            "captured_observations": captured_observations,
            "model_rows_recollected_per_fold": False,
            "a3_target_construction": (
                protocol["target_contract"]["raw_target_formula"]  # type: ignore[index]
            ),
            "decoder_span_sha256": projection_contract[
                "decoder_span_sha256"
            ],
            "summed_mean_sha256": projection_contract[
                "summed_mean_sha256"
            ],
        },
        folds=fold_reports,
        aggregate=aggregate,
        resources=_EXPECTED_EXACT_RESOURCES,
        decision=decision,
        source_model_unchanged=source_model_unchanged,
        layer10_unchanged=layer10_unchanged,
    )
    _progress("report: write tensor-free strict JSON without overwrite")
    return save_gemma3_l10_l17_trajectory_correction_lofo_report(
        destination,
        report,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed fit-only eight-family Gemma L10+L17 A3 "
            "trajectory-correction LOFO diagnostic."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--composition-bundle",
        type=Path,
        default=DEFAULT_COMPOSITION_BUNDLE_PATH,
    )
    parser.add_argument("--corpus-receipt", type=Path, default=DEFAULT_RECEIPT_OUTPUT)
    parser.add_argument("--corpus-artifact", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l10_l17_trajectory_correction_lofo(
        revision=args.revision,
        output=args.output,
        composition_bundle_path=args.composition_bundle,
        corpus_receipt_path=args.corpus_receipt,
        corpus_artifact_path=args.corpus_artifact,
        fit_input_path=args.fit_input,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
