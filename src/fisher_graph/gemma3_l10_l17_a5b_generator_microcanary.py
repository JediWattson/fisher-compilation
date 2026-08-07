"""Bounded A5b distillation canary for the frozen Layer-17 affine image.

This runner is the first learned-generator rung after A5a proves that the
frozen rank-182 affine image has enough downstream capacity.  It solves a
small, authenticated coordinate target bank on seven outer-training
families, keeps an example-disjoint inner audit split, refits only the four
existing rank-16 coordinate generators, freezes the resulting graph hashes,
and then scores one example from the untouched eighth family through the
normal full-model graph executor.

The canary is intentionally small.  It is Calibration-A development evidence,
not independent held-out confirmation and not a whole-model compilation.
All activation and coordinate tensors are ephemeral; the JSON report contains
only scalars, resource counts, booleans, and cryptographic hashes.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
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
from .computational_modes import ComputationalModeBasis
from .downstream_affine_coordinate_solver import DownstreamAffineSolverConfig
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_a4_oracle_attribution import (
    DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
)
from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT,
    _authenticate_a4_oracle_chain,
    _contains_tensor,
    _file_sha256,
    _take_first_examples,
    build_frozen_affine_image,
    load_a5_frozen_affine_capacity_report,
)
from .gemma3_l10_l17_a5b_batched_capacity import (
    solve_batched_frozen_affine_capacity_rows,
)
from .gemma3_l10_l17_a5b_downstream_coordinate_targets import (
    a5b_tensor_sha256,
    build_a5b_downstream_coordinate_target_bridge,
)
from .gemma3_l10_l17_full_block_closure_bundle import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
)
from .gemma3_l10_l17_full_block_closure_lofo import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256,
    _POST_DELTA_BOUNDARY,
    _build_source_runtime_catalog,
    _canonical_json_bytes,
    _fold_catalog,
    _full_block_capture_audit_receipt,
    _ordered_restored_lowerings,
    _source_lowering_maps,
    _validate_frozen_selection,
    _validate_source_decoder_contract,
    _validate_source_runtime_catalog,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    fit_frozen_basis_coordinate_generators,
    project_joint_target_to_frozen_bases,
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _authenticate_before_fit_access,
    _merge_corrected_composition_lowerings,
    _shared_compiled_input,
    _validate_fold_evaluation,
    score_trajectory_correction_fold,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_family_lofo_authority import (
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_full_block_closure_capture import (
    capture_gemma3_layer17_full_block_closure,
)
from .gemma3_layer17_v8_fit_lofo import _blocks_to_device, _family_blocks
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    select_top_fisher_same_layer_fragments,
)
from .modal_generator_graph import ModalGeneratorGraphPlan


GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a5b_generator_microcanary"
)
GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a5b-generator-microcanary-v1.json"
)

_EXPECTED_A5A_FILE_SHA256 = (
    "bace02fa1a290a5a076c6c6a723f9590513fa0290cccf5a7fd8b3a2117584390"
)
_EXPECTED_A5A_REPORT_SHA256 = (
    "c46d7c587962c64fca48da5fb54525fc1630d04e14cf1b87830457732224948e"
)
_EXPECTED_SOURCE_BINDINGS = {
    "a5a_file_sha256": _EXPECTED_A5A_FILE_SHA256,
    "a5a_report_sha256": _EXPECTED_A5A_REPORT_SHA256,
    "a4_oracle_file_sha256": (
        "9669aca95cf81eb33e8c0ac941e31279e8f8e484fb0352f6a04e868cf1bc72a6"
    ),
    "a4_oracle_report_sha256": (
        "f38b55bcc65d76d6eba1daeeea2e04dbd57401e58d33659163cea2731d5546eb"
    ),
    "a4_report_file_sha256": (
        "78222a62eee08bc58a92aa70613018de2f5870e3df7ca24e5a187b60956ed80d"
    ),
    "a4_report_sha256": (
        "db0e5d938c9f71f457a8de5b535c659abd6a85a509b2a3aa4261c79a9de6f702"
    ),
    "composition_bundle_file_sha256": (
        "394906f8e84a50e18922de0dc8c114be1ea9889f0995ccca180b9f6a8d303d8d"
    ),
    "composition_payload_sha256": (
        "2f7c2179656fc16c614cd84b7a0b29d3250443a5d8c80db221b220e3d3f082bf"
    ),
    "fold_bundle_file_sha256": (
        "cd4d0621e5b4fce44430d5bfc2c680fd29373e53788c01437f61580829bda162"
    ),
    "fold_bundle_payload_sha256": (
        "6d0dd667ccf9a34c15fdd3deda35795d4ada56344e3e6408134262da687958d2"
    ),
    "protocol_sha256": (
        "9adefa7d75d11343d8ab103ac7c683aaea269f65b894b11039eaa508c08fa3dc"
    ),
    "source_runtime_catalog_sha256": (
        "84b80b3cbabc3b8ff8bcf9f63e1f97a620fb38e2f6940b196f30906d9dfcb1b7"
    ),
}
_EXPECTED_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_EXPECTED_MODEL_FINGERPRINT = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_REPORT_DOMAIN = b"fisher-graph:a5b-generator-microcanary-report:v1\0"
_TARGET_RECEIPT_DOMAIN = b"fisher-graph:a5b-target-solve-report-receipt:v1\0"
_BRIDGE_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a5b-receipt:v1\0"
_INNER_SPLIT_DOMAIN = b"fisher-graph:a5b-generator-inner-split:v1\0"
_FITTER_SPLIT_DOMAIN = b"fisher-graph:a5b-generator-fitter-split:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")

_OUTER_FOLD_INDEX = 0
_TRAIN_EXAMPLES_PER_FAMILY = 2
_ROWS_PER_EXAMPLE = 4
_INNER_AUDIT_EXAMPLES_PER_FAMILY = 1
_TARGET_SOLVER_STEPS = 64
_TARGET_BATCH_ROWS = 8
_TARGET_LEARNING_RATE_FRACTION = 1.0e-2
_TOKEN_LOCALITY_ATOL = 1.0e-6
_TOKEN_LOCALITY_RTOL = 2.0e-6
_GENERATOR_RANK = 16
_RIDGE = 0.0
_EXPECTED_TRAINING_FAMILIES = 7
_EXPECTED_CAPTURE_EXAMPLES = 14
_EXPECTED_TARGET_ROWS = 56

_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_or_coordinate_tensors": False,
    "source_safe": True,
}


def _sha256_value(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _exact_mapping(
    value: object,
    *,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _assert_no_sensitive_payload_keys(value: object, *, path: str = "report") -> None:
    forbidden = {
        "prompt",
        "prompts",
        "prompt_texts",
        "raw_prompt",
        "raw_prompts",
        "prompt_id",
        "prompt_ids",
        "prompt_identity",
        "prompt_identities",
        "token_ids",
        "input_ids",
        "target_ids",
        "labels",
        "logits",
        "activations",
        "activation_tensor",
        "activation_tensors",
        "coordinates",
        "coordinate_tensor",
        "coordinate_tensors",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if key.casefold() in forbidden:
                raise ValueError(f"{path}.{key} contains prohibited source data")
            _assert_no_sensitive_payload_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_no_sensitive_payload_keys(child, path=f"{path}[{index}]")


def _validate_summary(value: object, *, label: str) -> Mapping[str, object]:
    summary = _exact_mapping(
        value,
        fields={"sum", "mean", "minimum", "median", "maximum"},
        label=label,
    )
    parsed = {
        name: _finite(summary[name], label=f"{label} {name}")
        for name in summary
    }
    if not (
        parsed["minimum"]
        <= parsed["median"]
        <= parsed["maximum"]
        and parsed["minimum"] <= parsed["mean"] <= parsed["maximum"]
    ):
        raise ValueError(f"{label} ordering is contradictory")
    return summary


def _validate_error_summary(value: object, *, label: str) -> Mapping[str, object]:
    error = _exact_mapping(
        value,
        fields={"rmse", "reference_rms", "nrmse", "max_abs_error"},
        label=label,
    )
    parsed = {
        name: _finite(error[name], label=f"{label} {name}") for name in error
    }
    if any(number < 0.0 for number in parsed.values()):
        raise ValueError(f"{label} values must be nonnegative")
    expected = parsed["rmse"] / max(parsed["reference_rms"], 1.0e-30)
    if not math.isclose(parsed["nrmse"], expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"{label} NRMSE is contradictory")
    return error


def _validate_target_solve_receipt(value: object) -> Mapping[str, object]:
    fields = {
        "schema",
        "objective",
        "scientific_method",
        "throughput_change_only",
        "teacher_boundary",
        "candidate_formula",
        "initialization",
        "canonical_target_dtype",
        "affine_arithmetic_dtype",
        "coordinate_layout",
        "runtime_correction_dtype",
        "runtime_correction_cast_count_per_materialization",
        "initial_correction_bit_identical_to_a4_float64_one_cast",
        "row_count",
        "row_chunk_size",
        "chunk_count",
        "batching",
        "solver",
        "initial_kl",
        "selected_kl",
        "absolute_mean_kl_improvement",
        "selected_not_worse_than_initial_for_every_token",
        "selected_step",
        "trust_projection_count",
        "initial_state_error",
        "selected_state_error",
        "hashes",
        "chunk_receipts",
        "frozen_affine_membership_by_construction",
        "basis_mean_or_decoder_changed",
        "deployable_generator_fitted",
        "contains_tensor_payloads",
        "receipt_sha256",
    }
    target = _exact_mapping(value, fields=fields, label="A5b target solve")
    if (
        target["schema"]
        != "fisher_graph.gemma3_l10_l17_a5b_batched_capacity.v1"
        or target["scientific_method"] != "a5a_frozen_affine_capacity_oracle"
        or target["throughput_change_only"] is not True
        or target["teacher_boundary"] != "captured_native_layer17_output"
        or target["canonical_target_dtype"] != "torch.float64"
        or target["affine_arithmetic_dtype"] != "torch.float64"
        or target["coordinate_layout"]
        != "joint_concatenated_four_node_rank_182"
        or target["runtime_correction_dtype"] != "torch.float32"
        or target["runtime_correction_cast_count_per_materialization"] != 1
        or target["initial_correction_bit_identical_to_a4_float64_one_cast"]
        is not True
        or target["row_count"] != _EXPECTED_TARGET_ROWS
        or target["row_chunk_size"] != _TARGET_BATCH_ROWS
        or target["chunk_count"] != 7
        or target["selected_not_worse_than_initial_for_every_token"] is not True
        or target["frozen_affine_membership_by_construction"] is not True
        or target["basis_mean_or_decoder_changed"] is not False
        or target["deployable_generator_fitted"] is not False
        or target["contains_tensor_payloads"] is not False
    ):
        raise ValueError("A5b target-solve contract drifted")

    batching = _exact_mapping(
        target["batching"],
        fields={
            "one_batched_head_callback_per_optimizer_evaluation",
            "independent_adam_parameter_group_per_token",
            "independent_kl_best_checkpoint_per_token",
            "token_locality_audited_on_native_and_a4_states",
            "token_locality_absolute_tolerance",
            "token_locality_relative_tolerance",
        },
        label="A5b target batching",
    )
    if any(
        batching[name] is not True
        for name in (
            "one_batched_head_callback_per_optimizer_evaluation",
            "independent_adam_parameter_group_per_token",
            "independent_kl_best_checkpoint_per_token",
            "token_locality_audited_on_native_and_a4_states",
        )
    ):
        raise ValueError("A5b target batching independence drifted")
    for name in (
        "token_locality_absolute_tolerance",
        "token_locality_relative_tolerance",
    ):
        if _finite(batching[name], label=f"A5b target {name}") < 0.0:
            raise ValueError("A5b target token-locality tolerance is invalid")
    if (
        batching["token_locality_absolute_tolerance"] != _TOKEN_LOCALITY_ATOL
        or batching["token_locality_relative_tolerance"] != _TOKEN_LOCALITY_RTOL
    ):
        raise ValueError("A5b target token-locality tolerance drifted")

    solver = _exact_mapping(
        target["solver"],
        fields={
            "steps",
            "learning_rate_fraction_of_per_token_initial_coefficient_rms",
            "minimum_scale_for_zero_rms",
            "initial_coefficient_rms",
            "effective_learning_rate",
            "scale_is_independent_for_each_token",
            "ridge",
            "trust_radius",
            "initial_point_evaluated_as_safe_abstention",
        },
        label="A5b target solver",
    )
    if (
        solver["steps"] != _TARGET_SOLVER_STEPS
        or solver[
            "learning_rate_fraction_of_per_token_initial_coefficient_rms"
        ]
        != _TARGET_LEARNING_RATE_FRACTION
        or _finite(
            solver["minimum_scale_for_zero_rms"],
            label="A5b target minimum scale",
        )
        <= 0.0
        or solver["scale_is_independent_for_each_token"] is not True
        or solver["ridge"] != 0.0
        or solver["trust_radius"] is not None
        or solver["initial_point_evaluated_as_safe_abstention"] is not True
    ):
        raise ValueError("A5b target solver parameters drifted")
    _validate_summary(
        solver["initial_coefficient_rms"],
        label="A5b initial coefficient RMS",
    )
    _validate_summary(
        solver["effective_learning_rate"],
        label="A5b effective learning rate",
    )

    initial_kl = _validate_summary(target["initial_kl"], label="A5b initial KL")
    selected_kl = _validate_summary(target["selected_kl"], label="A5b selected KL")
    _validate_summary(target["selected_step"], label="A5b selected step")
    _validate_summary(
        target["trust_projection_count"],
        label="A5b trust projection count",
    )
    if (
        any(
            _finite(summary[name], label=f"A5b KL {name}") < 0.0
            for summary in (initial_kl, selected_kl)
            for name in summary
        )
        or not math.isclose(
            _finite(
                target["absolute_mean_kl_improvement"],
                label="A5b KL improvement",
            ),
            float(initial_kl["mean"]) - float(selected_kl["mean"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("A5b target KL summaries are contradictory")
    _validate_error_summary(target["initial_state_error"], label="A5b initial state")
    _validate_error_summary(target["selected_state_error"], label="A5b selected state")

    hashes = _exact_mapping(
        target["hashes"],
        fields={
            "initial_coefficient_sha256",
            "selected_coefficient_sha256",
            "initial_correction_sha256",
            "selected_correction_sha256",
            "initial_state_sha256",
            "selected_state_sha256",
        },
        label="A5b target hashes",
    )
    for name, digest in hashes.items():
        _require_sha256(digest, label=f"A5b target hash {name}")

    chunks = target["chunk_receipts"]
    if not isinstance(chunks, list) or len(chunks) != 7:
        raise ValueError("A5b target chunk catalog is invalid")
    for index, raw_chunk in enumerate(chunks):
        chunk = _exact_mapping(
            raw_chunk,
            fields={
                "chunk_index",
                "row_start",
                "row_stop",
                "row_count",
                "initial_kl",
                "selected_kl",
                "selected_step",
                "trust_projection_count",
                "token_locality",
                "full_solver_receipt_sha256",
            },
            label=f"A5b target chunk {index}",
        )
        if (
            chunk["chunk_index"] != index
            or chunk["row_start"] != index * _TARGET_BATCH_ROWS
            or chunk["row_stop"] != (index + 1) * _TARGET_BATCH_ROWS
            or chunk["row_count"] != _TARGET_BATCH_ROWS
        ):
            raise ValueError("A5b target chunk accounting drifted")
        for name in (
            "initial_kl",
            "selected_kl",
            "selected_step",
            "trust_projection_count",
        ):
            _validate_summary(chunk[name], label=f"A5b chunk {index} {name}")
        locality = _exact_mapping(
            chunk["token_locality"],
            fields={
                "method",
                "probe_states",
                "row_count",
                "nontrivial_multirow_probe",
                "absolute_tolerance",
                "relative_tolerance",
                "teacher",
                "a4_baseline",
                "passed",
            },
            label=f"A5b chunk {index} token locality",
        )
        if (
            locality["row_count"] != _TARGET_BATCH_ROWS
            or locality["nontrivial_multirow_probe"] is not True
            or locality["passed"] is not True
            or locality["absolute_tolerance"] != _TOKEN_LOCALITY_ATOL
            or locality["relative_tolerance"] != _TOKEN_LOCALITY_RTOL
        ):
            raise ValueError("A5b target token-locality audit drifted")
        for name in ("teacher", "a4_baseline"):
            error = _exact_mapping(
                locality[name],
                fields={"max_abs", "rms", "reference_max_abs"},
                label=f"A5b chunk {index} locality {name}",
            )
            if any(
                _finite(number, label=f"A5b locality {name} {field}") < 0.0
                for field, number in error.items()
            ):
                raise ValueError("A5b locality error is invalid")
        _require_sha256(
            chunk["full_solver_receipt_sha256"],
            label=f"A5b chunk {index} solver receipt",
        )

    supplied = _require_sha256(target["receipt_sha256"], label="A5b target receipt")
    payload = dict(target)
    payload.pop("receipt_sha256")
    if supplied != _sha256_value(_TARGET_RECEIPT_DOMAIN, payload):
        raise ValueError("A5b target-solve receipt hash mismatch")
    return target


def _validate_bridge_receipt(
    value: object,
    *,
    selected_coefficient_sha256: str,
) -> Mapping[str, object]:
    bridge = _exact_mapping(
        value,
        fields={
            "schema",
            "format_version",
            "scientific_role",
            "source_safe",
            "contains_tensors",
            "contains_prompt_text",
            "contains_prompt_identities",
            "contains_token_ids",
            "heldout_confirmation",
            "outer_split",
            "authentication",
            "frozen_affine_image",
            "joint_roundtrip_audit",
            "inner_split",
            "fisher_normalization",
            "row_accounting",
            "consumer_contract",
            "receipt_sha256",
        },
        label="A5b target bridge",
    )
    if (
        bridge["schema"]
        != "fisher_graph.gemma3_l10_l17_a5b_downstream_coordinate_targets"
        or bridge["format_version"] != 1
        or bridge["scientific_role"]
        != "calibration_a_fit_outer_training_downstream_coordinate_targets"
        or bridge["source_safe"] is not True
        or bridge["contains_tensors"] is not False
        or bridge["contains_prompt_text"] is not False
        or bridge["contains_prompt_identities"] is not False
        or bridge["contains_token_ids"] is not False
        or bridge["heldout_confirmation"] is not False
    ):
        raise ValueError("A5b target bridge safety boundary drifted")

    outer = _exact_mapping(
        bridge["outer_split"],
        fields={
            "training_family_aliases",
            "held_family_alias",
            "held_family_rows_accepted",
            "held_family_used_for_fit_or_audit",
        },
        label="A5b bridge outer split",
    )
    aliases = outer["training_family_aliases"]
    if (
        not isinstance(aliases, Sequence)
        or isinstance(aliases, (str, bytes))
        or len(aliases) != _EXPECTED_TRAINING_FAMILIES
        or len(set(aliases)) != _EXPECTED_TRAINING_FAMILIES
        or outer["held_family_alias"] in aliases
        or outer["held_family_rows_accepted"] is not False
        or outer["held_family_used_for_fit_or_audit"] is not False
    ):
        raise ValueError("A5b bridge outer split leaked the held family")

    authentication = _exact_mapping(
        bridge["authentication"],
        fields={
            "compiled_inputs_sha256",
            "selected_joint_coordinates_sha256",
            "joint_coordinate_width",
        },
        label="A5b bridge authentication",
    )
    _require_sha256(
        authentication["compiled_inputs_sha256"],
        label="A5b bridge compiled inputs",
    )
    if (
        _require_sha256(
            authentication["selected_joint_coordinates_sha256"],
            label="A5b bridge selected coordinates",
        )
        != selected_coefficient_sha256
        or authentication["joint_coordinate_width"] != 182
    ):
        raise ValueError("A5b bridge does not bind the target coordinates")

    image = _exact_mapping(
        bridge["frozen_affine_image"],
        fields={
            "node_order",
            "rank_by_node",
            "coordinate_slices",
            "fragment_id_by_node",
            "basis_sha256_by_node",
            "mean_sha256_by_node",
            "encoder_sha256_by_node",
            "decoder_sha256_by_node",
            "basis_artifacts_unchanged_after_decode",
            "mean_tensors_byte_identical_after_decode",
            "encoder_tensors_byte_identical_after_decode",
            "decoder_tensors_byte_identical_after_decode",
        },
        label="A5b bridge frozen affine image",
    )
    node_order = image["node_order"]
    ranks = image["rank_by_node"]
    if (
        not isinstance(node_order, Sequence)
        or isinstance(node_order, (str, bytes))
        or len(node_order) != 4
        or len(set(node_order)) != 4
        or not isinstance(ranks, Sequence)
        or isinstance(ranks, (str, bytes))
        or len(ranks) != 4
        or any(type(rank) is not int or rank <= 0 for rank in ranks)
        or sum(ranks) != 182
        or any(
            image[name] is not True
            for name in (
                "basis_artifacts_unchanged_after_decode",
                "mean_tensors_byte_identical_after_decode",
                "encoder_tensors_byte_identical_after_decode",
                "decoder_tensors_byte_identical_after_decode",
            )
        )
    ):
        raise ValueError("A5b bridge frozen image drifted")
    nodes = tuple(str(name) for name in node_order)
    fragment_map = _exact_mapping(
        image["fragment_id_by_node"], fields=set(nodes), label="A5b fragments"
    )
    if len(set(fragment_map.values())) != 4:
        raise ValueError("A5b bridge fragments must be distinct")
    for map_name in (
        "basis_sha256_by_node",
        "mean_sha256_by_node",
        "encoder_sha256_by_node",
        "decoder_sha256_by_node",
    ):
        digests = _exact_mapping(
            image[map_name], fields=set(nodes), label=f"A5b bridge {map_name}"
        )
        for name, digest in digests.items():
            _require_sha256(digest, label=f"A5b bridge {map_name} {name}")
    slices = _exact_mapping(
        image["coordinate_slices"],
        fields=set(nodes),
        label="A5b coordinate slices",
    )
    start = 0
    for name, rank in zip(nodes, ranks, strict=True):
        row = _exact_mapping(
            slices[name],
            fields={"start", "stop", "rank"},
            label=f"A5b coordinate slice {name}",
        )
        if row != {"start": start, "stop": start + rank, "rank": rank}:
            raise ValueError("A5b bridge coordinate slices drifted")
        start += rank

    roundtrip = _exact_mapping(
        bridge["joint_roundtrip_audit"],
        fields={
            "definition",
            "joint_roundtrip_sha256",
            "summed_decoded_contribution_sha256",
            "max_abs_difference",
            "rms_difference",
            "relative_tolerance",
            "absolute_tolerance",
            "passed",
        },
        label="A5b bridge roundtrip",
    )
    for name in ("joint_roundtrip_sha256", "summed_decoded_contribution_sha256"):
        _require_sha256(roundtrip[name], label=f"A5b bridge {name}")
    if (
        _finite(roundtrip["max_abs_difference"], label="A5b roundtrip max") < 0.0
        or _finite(roundtrip["rms_difference"], label="A5b roundtrip RMS") < 0.0
        or roundtrip["relative_tolerance"] != 1.0e-11
        or roundtrip["absolute_tolerance"] != 1.0e-11
        or roundtrip["passed"] is not True
    ):
        raise ValueError("A5b bridge roundtrip audit drifted")

    inner = _exact_mapping(
        bridge["inner_split"],
        fields={
            "method",
            "inner_split_binding_sha256",
            "inner_audit_examples_per_family",
            "fit_example_count",
            "audit_example_count",
            "fit_example_membership_sha256",
            "audit_example_membership_sha256",
            "row_overlap_count",
            "example_overlap_count",
            "rows_exactly_partitioned",
            "examples_exactly_partitioned",
        },
        label="A5b bridge inner split",
    )
    for name in (
        "inner_split_binding_sha256",
        "fit_example_membership_sha256",
        "audit_example_membership_sha256",
    ):
        _require_sha256(inner[name], label=f"A5b bridge {name}")
    if (
        inner["inner_audit_examples_per_family"] != 1
        or inner["fit_example_count"] != 7
        or inner["audit_example_count"] != 7
        or inner["row_overlap_count"] != 0
        or inner["example_overlap_count"] != 0
        or inner["rows_exactly_partitioned"] is not True
        or inner["examples_exactly_partitioned"] is not True
    ):
        raise ValueError("A5b bridge inner split drifted")

    fisher = _exact_mapping(
        bridge["fisher_normalization"],
        fields={
            "all_rows_preserve_raw_authenticated_fisher_weights",
            "fit_and_audit_normalized_independently",
            "audit_weights_influence_fit_normalization",
            "policy",
            "training_family_count",
            "target_total_mass_per_role_and_node",
            "target_mass_per_family",
            "fit_by_node",
            "audit_by_node",
        },
        label="A5b bridge Fisher normalization",
    )
    if (
        fisher["all_rows_preserve_raw_authenticated_fisher_weights"] is not True
        or fisher["fit_and_audit_normalized_independently"] is not True
        or fisher["audit_weights_influence_fit_normalization"] is not False
        or fisher["training_family_count"] != 7
        or fisher["target_total_mass_per_role_and_node"] != 1.0
        or not math.isclose(
            _finite(fisher["target_mass_per_family"], label="A5b Fisher mass"),
            1.0 / 7.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError("A5b bridge Fisher normalization drifted")
    for role in ("fit_by_node", "audit_by_node"):
        by_node = _exact_mapping(
            fisher[role], fields=set(nodes), label=f"A5b Fisher {role}"
        )
        for name, raw_row in by_node.items():
            row = _exact_mapping(
                raw_row,
                fields={
                    "fragment_id",
                    "total_mass",
                    "family_mass_by_alias",
                    "unit_total_mass",
                    "equal_family_mass",
                },
                label=f"A5b Fisher {role} {name}",
            )
            masses = row["family_mass_by_alias"]
            if (
                row["fragment_id"] != fragment_map[name]
                or not math.isclose(
                    _finite(row["total_mass"], label="A5b Fisher total"),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=2.0e-12,
                )
                or not isinstance(masses, Mapping)
                or set(masses) != set(aliases)
                or any(
                    not math.isclose(
                        _finite(mass, label="A5b Fisher family mass"),
                        1.0 / 7.0,
                        rel_tol=0.0,
                        abs_tol=2.0e-12,
                    )
                    for mass in masses.values()
                )
                or row["unit_total_mass"] is not True
                or row["equal_family_mass"] is not True
            ):
                raise ValueError("A5b per-family Fisher normalization drifted")

    accounting = _exact_mapping(
        bridge["row_accounting"],
        fields={
            "all",
            "fit",
            "audit",
            "all_observations_equal_fit_plus_audit",
            "all_examples_equal_fit_plus_audit",
        },
        label="A5b bridge row accounting",
    )
    expected_counts = {"all": (56, 14), "fit": (28, 7), "audit": (28, 7)}
    fragment_ids = set(fragment_map.values())
    for role, (observations, sequences) in expected_counts.items():
        rows = _exact_mapping(
            accounting[role],
            fields={
                "row_key_sha256",
                "observations",
                "sequences",
                "fragment_tensor_sha256s",
            },
            label=f"A5b bridge rows {role}",
        )
        _require_sha256(rows["row_key_sha256"], label=f"A5b {role} row keys")
        tensor_hashes = _exact_mapping(
            rows["fragment_tensor_sha256s"],
            fields=fragment_ids,
            label=f"A5b {role} fragment tensors",
        )
        if rows["observations"] != observations or rows["sequences"] != sequences:
            raise ValueError("A5b bridge row counts drifted")
        for fragment_id, raw_hashes in tensor_hashes.items():
            hashes = _exact_mapping(
                raw_hashes,
                fields={
                    "inputs_sha256",
                    "contributions_sha256",
                    "fisher_weights_sha256",
                },
                label=f"A5b {role} tensors {fragment_id}",
            )
            for name, digest in hashes.items():
                _require_sha256(digest, label=f"A5b {role} {fragment_id} {name}")
    if (
        accounting["all_observations_equal_fit_plus_audit"] is not True
        or accounting["all_examples_equal_fit_plus_audit"] is not True
    ):
        raise ValueError("A5b bridge partition accounting drifted")

    consumer = _exact_mapping(
        bridge["consumer_contract"],
        fields={"compatible_with", "contribution_target", "generator_fit_performed"},
        label="A5b bridge consumer",
    )
    if (
        consumer["compatible_with"] != "fit_frozen_basis_coordinate_generators"
        or consumer["generator_fit_performed"] is not False
    ):
        raise ValueError("A5b bridge consumer contract drifted")

    supplied = _require_sha256(bridge["receipt_sha256"], label="A5b bridge receipt")
    payload = dict(bridge)
    payload.pop("receipt_sha256")
    if supplied != _sha256_value(_BRIDGE_RECEIPT_DOMAIN, payload):
        raise ValueError("A5b target bridge receipt hash mismatch")
    return bridge


def _select_first_rows_per_example(
    row_keys: tuple[tuple[str, int], ...],
    *,
    expected_examples: Sequence[str],
    rows_per_example: int,
) -> tuple[Tensor, tuple[tuple[str, int], ...]]:
    """Select a bounded prefix per example while preserving capture order."""

    examples = tuple(expected_examples)
    if (
        not examples
        or len(examples) != len(set(examples))
        or any(not isinstance(value, str) or not value for value in examples)
        or type(rows_per_example) is not int
        or rows_per_example <= 0
    ):
        raise ValueError("bounded row selection configuration is invalid")
    expected = set(examples)
    if {example_id for example_id, _ in row_keys} != expected:
        raise ValueError("capture row examples differ from the bounded selection")
    counts = {example_id: 0 for example_id in examples}
    indices: list[int] = []
    selected: list[tuple[str, int]] = []
    for index, key in enumerate(row_keys):
        example_id = key[0]
        if counts[example_id] < rows_per_example:
            counts[example_id] += 1
            indices.append(index)
            selected.append(key)
    if set(counts.values()) != {rows_per_example}:
        raise ValueError("a training example has too few captured rows")
    if len(indices) != len(examples) * rows_per_example:
        raise RuntimeError("bounded row accounting drifted")
    return torch.tensor(indices, dtype=torch.long), tuple(selected)


def _metric_row(value: object, *, label: str) -> dict[str, float]:
    fields = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} metric fields are invalid")
    parsed = {name: _finite(value[name], label=f"{label} {name}") for name in fields}
    if (
        parsed["nll_per_token"] < 0.0
        or parsed["native_to_candidate_kl_per_token"] < 0.0
        or not 0.0 <= parsed["top1_agreement_to_native"] <= 1.0
    ):
        raise ValueError(f"{label} metric range is invalid")
    return parsed


def _conclusion_from_evaluation(evaluation: Mapping[str, object]) -> dict[str, object]:
    conditions = evaluation.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("A5b evaluation conditions are unavailable")
    frozen = _metric_row(
        conditions.get("frozen_uncorrected_composition"),
        label="A5b frozen composition",
    )
    learned = _metric_row(
        conditions.get("trajectory_corrected_composition"),
        label="A5b learned composition",
    )
    return {
        "bounded_held_example_frozen_kl": frozen[
            "native_to_candidate_kl_per_token"
        ],
        "bounded_held_example_learned_kl": learned[
            "native_to_candidate_kl_per_token"
        ],
        "bounded_held_example_frozen_delta_nll": frozen[
            "delta_nll_per_token"
        ],
        "bounded_held_example_learned_delta_nll": learned[
            "delta_nll_per_token"
        ],
        "bounded_held_example_frozen_top1": frozen[
            "top1_agreement_to_native"
        ],
        "bounded_held_example_learned_top1": learned[
            "top1_agreement_to_native"
        ],
        "learned_generator_improves_frozen_kl": (
            learned["native_to_candidate_kl_per_token"]
            < frozen["native_to_candidate_kl_per_token"]
        ),
        "learned_generator_improves_frozen_delta_nll": (
            learned["delta_nll_per_token"] < frozen["delta_nll_per_token"]
        ),
        "learned_generator_improves_frozen_top1": (
            learned["top1_agreement_to_native"]
            > frozen["top1_agreement_to_native"]
        ),
        "microcanary_only": True,
        "does_not_establish_eight_fold_competitive_compilation": True,
    }


def validate_a5b_generator_microcanary_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("A5b microcanary report must be a mapping")
    raw = dict(value)
    expected = {
        "schema",
        "format_version",
        "scientific_role",
        "source_bindings",
        "runtime",
        "configuration",
        "capture",
        "target_solve",
        "target_bridge",
        "generator_fit",
        "frozen_executable",
        "evaluation",
        "conclusion",
        "full_model_forward_evaluated",
        "whole_model_compiled",
        "heldout_confirmation",
        "serving_authorized",
        "latency_or_kernel_speed_claim",
        "safety",
        "report_sha256",
    }
    if set(raw) != expected:
        raise ValueError("A5b microcanary report fields are invalid")
    if (
        raw["schema"] != GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_SCHEMA
        or raw["format_version"]
        != GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_FORMAT_VERSION
        or raw["scientific_role"]
        != "calibration_a_one_fold_downstream_coordinate_distillation_microcanary"
    ):
        raise ValueError("A5b microcanary report header is invalid")
    if (
        raw["full_model_forward_evaluated"] is not True
        or raw["whole_model_compiled"] is not False
        or raw["heldout_confirmation"] is not False
        or raw["serving_authorized"] is not False
        or raw["latency_or_kernel_speed_claim"] is not False
        or raw["safety"] != _SAFETY
        or _contains_tensor(raw)
    ):
        raise ValueError("A5b microcanary scope or safety contract drifted")
    _assert_no_sensitive_payload_keys(raw)

    bindings = raw["source_bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        _EXPECTED_SOURCE_BINDINGS
    ):
        raise ValueError("A5b source binding fields are invalid")
    for name in _EXPECTED_SOURCE_BINDINGS:
        _require_sha256(bindings[name], label=f"A5b source binding {name}")
    if dict(bindings) != _EXPECTED_SOURCE_BINDINGS:
        raise ValueError("A5b source chain differs from the canonical A5a lineage")

    runtime = _exact_mapping(
        raw["runtime"],
        fields={
            "model_id",
            "requested_revision",
            "model_fingerprint",
            "device",
            "dtype",
            "local_files_only",
        },
        label="A5b runtime",
    )
    if (
        runtime["model_id"] != DEFAULT_MODEL_ID
        or runtime["requested_revision"] != _EXPECTED_MODEL_REVISION
        or runtime["model_fingerprint"] != _EXPECTED_MODEL_FINGERPRINT
        or runtime["device"] != "cpu"
        or runtime["dtype"] != "float32"
        or runtime["local_files_only"] is not True
    ):
        raise ValueError("A5b runtime differs from the canonical CPU replay")
    _require_sha256(runtime["model_fingerprint"], label="A5b model fingerprint")

    config = raw["configuration"]
    expected_config = {
        "outer_fold_index": _OUTER_FOLD_INDEX,
        "training_family_count": _EXPECTED_TRAINING_FAMILIES,
        "training_examples_per_family": _TRAIN_EXAMPLES_PER_FAMILY,
        "rows_per_example": _ROWS_PER_EXAMPLE,
        "inner_audit_examples_per_family": _INNER_AUDIT_EXAMPLES_PER_FAMILY,
        "target_solver_steps": _TARGET_SOLVER_STEPS,
        "target_batch_rows": _TARGET_BATCH_ROWS,
        "target_learning_rate_fraction": _TARGET_LEARNING_RATE_FRACTION,
        "token_locality_absolute_tolerance": _TOKEN_LOCALITY_ATOL,
        "token_locality_relative_tolerance": _TOKEN_LOCALITY_RTOL,
        "target_ridge": 0.0,
        "target_trust_radius": None,
        "generator_rank": _GENERATOR_RANK,
        "generator_ridge": _RIDGE,
        "held_examples_scored": 1,
    }
    if config != expected_config:
        raise ValueError("A5b microcanary configuration drifted")

    capture = raw["capture"]
    if not isinstance(capture, Mapping) or set(capture) != {
        "capture_sha256",
        "capture_audit_sha256",
        "source_row_catalog_sha256",
        "bounded_row_catalog_sha256",
        "training_examples",
        "captured_observations",
        "bounded_target_rows",
        "held_family_rows_present",
        "all_required_capture_audits_pass",
    }:
        raise ValueError("A5b capture fields are invalid")
    for name in (
        "capture_sha256",
        "capture_audit_sha256",
        "source_row_catalog_sha256",
        "bounded_row_catalog_sha256",
    ):
        _require_sha256(capture[name], label=f"A5b capture {name}")
    if (
        capture["training_examples"] != _EXPECTED_CAPTURE_EXAMPLES
        or type(capture["captured_observations"]) is not int
        or capture["captured_observations"] < _EXPECTED_TARGET_ROWS
        or capture["bounded_target_rows"] != _EXPECTED_TARGET_ROWS
        or capture["held_family_rows_present"] is not False
        or capture["all_required_capture_audits_pass"] is not True
    ):
        raise ValueError("A5b capture accounting drifted")

    target = _validate_target_solve_receipt(raw["target_solve"])
    target_hashes = target["hashes"]
    assert isinstance(target_hashes, Mapping)
    bridge = _validate_bridge_receipt(
        raw["target_bridge"],
        selected_coefficient_sha256=str(
            target_hashes["selected_coefficient_sha256"]
        ),
    )

    generator = raw["generator_fit"]
    if (
        not isinstance(generator, Mapping)
        or set(generator)
        != {
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
        or generator.get("parameter_count") != 163_094
        or generator.get("macs_per_token") != 160_352
        or generator.get("interaction_count") != 0
    ):
        raise ValueError("A5b generator resources drifted")
    node_order = generator.get("node_order")
    if (
        not isinstance(node_order, Sequence)
        or isinstance(node_order, (str, bytes))
        or len(node_order) != 4
        or len(set(node_order)) != 4
    ):
        raise ValueError("A5b generator node order is invalid")
    generator_nodes = set(node_order)
    for name in (
        "graph_sha256",
        "source_mean_bias_sha256_by_node",
        "source_decoder_basis_sha256_by_node",
        "lowering_sha256_by_node",
        "generator_plan_sha256_by_node",
    ):
        if name == "graph_sha256":
            _require_sha256(generator.get(name), label=f"A5b generator {name}")
        else:
            values = _exact_mapping(
                generator.get(name),
                fields=generator_nodes,
                label=f"A5b generator {name}",
            )
            for node_name, digest in values.items():
                _require_sha256(
                    digest,
                    label=f"A5b generator {name} {node_name}",
                )

    frozen = raw["frozen_executable"]
    if (
        not isinstance(frozen, Mapping)
        or set(frozen)
        != {
            "corrected_layer17_graph_sha256",
            "corrected_layer17_lowering_sha256_by_node",
            "corrected_composition_graph_sha256",
            "bridge_receipt_sha256",
            "fit_split_sha256",
            "audit_split_sha256",
            "executable_freeze_sha256",
            "corrected_layer17_parameters",
            "corrected_layer17_macs_per_token",
            "corrected_composition_parameters",
            "corrected_composition_macs_per_token",
            (
                "generator_and_composition_hashes_frozen_before_held_"
                "example_selection_or_model_evaluation"
            ),
        }
        or frozen.get(
            "generator_and_composition_hashes_frozen_before_held_"
            "example_selection_or_model_evaluation"
        )
        is not True
        or frozen.get("corrected_layer17_parameters") != 163_094
        or frozen.get("corrected_layer17_macs_per_token") != 160_352
        or frozen.get("corrected_composition_parameters") != 295_129
        or frozen.get("corrected_composition_macs_per_token") != 289_600
    ):
        raise ValueError("A5b frozen executable contract drifted")
    for name in (
        "corrected_layer17_graph_sha256",
        "corrected_composition_graph_sha256",
        "executable_freeze_sha256",
    ):
        _require_sha256(frozen.get(name), label=f"A5b executable {name}")
    lowering_hashes = _exact_mapping(
        frozen["corrected_layer17_lowering_sha256_by_node"],
        fields=generator_nodes,
        label="A5b frozen Layer17 lowerings",
    )
    for node_name, digest in lowering_hashes.items():
        _require_sha256(digest, label=f"A5b frozen lowering {node_name}")
    if (
        frozen["corrected_layer17_graph_sha256"] != generator["graph_sha256"]
        or dict(lowering_hashes) != dict(generator["lowering_sha256_by_node"])
        or frozen["bridge_receipt_sha256"] != bridge["receipt_sha256"]
    ):
        raise ValueError("A5b frozen executable does not bind the fitted generator")
    for name in ("bridge_receipt_sha256", "fit_split_sha256", "audit_split_sha256"):
        _require_sha256(frozen[name], label=f"A5b executable {name}")
    accounting = bridge["row_accounting"]
    assert isinstance(accounting, Mapping)
    expected_fit_split = _sha256_value(
        _FITTER_SPLIT_DOMAIN,
        {
            "role": "fit",
            "bridge_receipt_sha256": bridge["receipt_sha256"],
            "row_key_sha256": accounting["fit"]["row_key_sha256"],
            "observations": accounting["fit"]["observations"],
        },
    )
    expected_audit_split = _sha256_value(
        _FITTER_SPLIT_DOMAIN,
        {
            "role": "audit",
            "bridge_receipt_sha256": bridge["receipt_sha256"],
            "row_key_sha256": accounting["audit"]["row_key_sha256"],
            "observations": accounting["audit"]["observations"],
        },
    )
    if (
        frozen["fit_split_sha256"] != expected_fit_split
        or frozen["audit_split_sha256"] != expected_audit_split
    ):
        raise ValueError("A5b frozen executable split lineage drifted")
    expected_freeze = _sha256_value(
        b"fisher-graph:a5b-executable-freeze:v1\0",
        {
            "layer17_graph_sha256": frozen["corrected_layer17_graph_sha256"],
            "lowering_sha256_by_node": dict(lowering_hashes),
            "composition_graph_sha256": frozen[
                "corrected_composition_graph_sha256"
            ],
            "bridge_receipt_sha256": frozen["bridge_receipt_sha256"],
            "fit_split_sha256": frozen["fit_split_sha256"],
            "audit_split_sha256": frozen["audit_split_sha256"],
        },
    )
    if frozen["executable_freeze_sha256"] != expected_freeze:
        raise ValueError("A5b executable freeze hash is contradictory")

    evaluation = raw["evaluation"]
    _validate_fold_evaluation(evaluation, label="A5b held microcanary")
    assert isinstance(evaluation, Mapping)
    conditions = evaluation.get("conditions")
    required_conditions = {
        "layer10_only",
        "trajectory_corrected_layer17_only",
        "frozen_uncorrected_composition",
        "trajectory_corrected_composition",
        "matched_double_deletion",
    }
    if not isinstance(conditions, Mapping) or set(conditions) != required_conditions:
        raise ValueError("A5b evaluation condition catalog drifted")
    for name in required_conditions:
        _metric_row(conditions[name], label=f"A5b evaluation {name}")
    native = evaluation.get("native")
    if not isinstance(native, Mapping) or set(native) != {"nll_per_token"}:
        raise ValueError("A5b native evaluation metric is invalid")
    native_nll = _finite(native["nll_per_token"], label="A5b native NLL")
    if native_nll < 0.0:
        raise ValueError("A5b native NLL is invalid")
    for name in required_conditions:
        metric = conditions[name]
        if not math.isclose(
            float(metric["delta_nll_per_token"]),
            float(metric["nll_per_token"]) - native_nll,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("A5b evaluation NLL delta is contradictory")

    expected_conclusion = _conclusion_from_evaluation(evaluation)
    if raw["conclusion"] != expected_conclusion:
        raise ValueError("A5b conclusion contradicts held evaluation")
    supplied = _require_sha256(raw.pop("report_sha256"), label="A5b report")
    if supplied != _sha256_value(_REPORT_DOMAIN, raw):
        raise ValueError("A5b microcanary report hash mismatch")
    raw["report_sha256"] = supplied
    return raw


def build_a5b_generator_microcanary_report(
    *,
    source_bindings: Mapping[str, str],
    runtime: Mapping[str, object],
    capture: Mapping[str, object],
    target_solve: Mapping[str, object],
    target_bridge: Mapping[str, object],
    generator_fit: Mapping[str, object],
    frozen_executable: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_SCHEMA,
        "format_version": GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_FORMAT_VERSION,
        "scientific_role": (
            "calibration_a_one_fold_downstream_coordinate_distillation_microcanary"
        ),
        "source_bindings": dict(source_bindings),
        "runtime": dict(runtime),
        "configuration": {
            "outer_fold_index": _OUTER_FOLD_INDEX,
            "training_family_count": _EXPECTED_TRAINING_FAMILIES,
            "training_examples_per_family": _TRAIN_EXAMPLES_PER_FAMILY,
            "rows_per_example": _ROWS_PER_EXAMPLE,
            "inner_audit_examples_per_family": _INNER_AUDIT_EXAMPLES_PER_FAMILY,
            "target_solver_steps": _TARGET_SOLVER_STEPS,
            "target_batch_rows": _TARGET_BATCH_ROWS,
            "target_learning_rate_fraction": _TARGET_LEARNING_RATE_FRACTION,
            "token_locality_absolute_tolerance": _TOKEN_LOCALITY_ATOL,
            "token_locality_relative_tolerance": _TOKEN_LOCALITY_RTOL,
            "target_ridge": 0.0,
            "target_trust_radius": None,
            "generator_rank": _GENERATOR_RANK,
            "generator_ridge": _RIDGE,
            "held_examples_scored": 1,
        },
        "capture": dict(capture),
        "target_solve": dict(target_solve),
        "target_bridge": dict(target_bridge),
        "generator_fit": dict(generator_fit),
        "frozen_executable": dict(frozen_executable),
        "evaluation": dict(evaluation),
        "conclusion": _conclusion_from_evaluation(evaluation),
        "full_model_forward_evaluated": True,
        "whole_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "latency_or_kernel_speed_claim": False,
        "safety": dict(_SAFETY),
    }
    payload["report_sha256"] = _sha256_value(_REPORT_DOMAIN, payload)
    return validate_a5b_generator_microcanary_report(payload)


def save_a5b_generator_microcanary_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5b microcanary report")
    validated = validate_a5b_generator_microcanary_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        validated,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if destination.exists():
            raise FileExistsError("refusing to overwrite A5b microcanary report")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_a5b_generator_microcanary_report(
    path: Path | str,
) -> dict[str, object]:
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("A5b microcanary report is not strict JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("A5b microcanary report must contain one object")
    return validate_a5b_generator_microcanary_report(raw)


def _progress(message: str) -> None:
    print(f"[a5b-generator] {message}", flush=True)


def run_gemma3_l10_l17_a5b_generator_microcanary(
    *,
    revision: str,
    output: Path | str = DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT,
    a5a_path: Path | str = DEFAULT_GEMMA3_L10_L17_A5_FROZEN_AFFINE_CAPACITY_ORACLE_OUTPUT,
    a4_oracle_path: Path | str = DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
    source_a4_report_path: Path | str = DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    fold_bundle_path: Path | str = DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    composition_bundle_path: Path | str = DEFAULT_COMPOSITION_BUNDLE_PATH,
    corpus_receipt_path: Path | str = DEFAULT_RECEIPT_OUTPUT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_OUTPUT,
    fit_input_path: Path | str = DEFAULT_FIT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Run the fixed one-fold A5b learned-generator microcanary."""

    if (
        not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or revision != _EXPECTED_MODEL_REVISION
        or model_id != DEFAULT_MODEL_ID
        or device_name != "cpu"
        or dtype != "float32"
    ):
        raise ValueError("A5b must replay the canonical pinned CPU float32 runtime")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A5b microcanary report")
    a5a_file = Path(a5a_path)
    a4_oracle_file = Path(a4_oracle_path)
    a4_file = Path(source_a4_report_path)
    fold_file = Path(fold_bundle_path)
    composition_file = Path(composition_bundle_path)

    _progress("preflight: authenticate A5a capacity and frozen A4 sources")
    a5a = load_a5_frozen_affine_capacity_report(a5a_file)
    if (
        _file_sha256(a5a_file) != _EXPECTED_A5A_FILE_SHA256
        or a5a.get("report_sha256") != _EXPECTED_A5A_REPORT_SHA256
        or a5a.get("conclusion", {}).get("bounded_canary_resolves_affine_capacity")
        is not True
    ):
        raise ValueError("A5b requires the canonical capacity-passing A5a result")
    a4_oracle, source_report, fold_bundle = _authenticate_a4_oracle_chain(
        a4_oracle_path=a4_oracle_file,
        a4_report_path=a4_file,
        fold_bundle_path=fold_file,
        composition_bundle_path=composition_file,
    )
    source_runtime = source_report.get("runtime")
    source_authorization = source_report.get("authorization")
    protocol = source_report.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (source_runtime, source_authorization, protocol)
    ):
        raise TypeError("published A4 runtime/authorization/protocol is unavailable")
    assert isinstance(source_runtime, Mapping)
    assert isinstance(source_authorization, Mapping)
    assert isinstance(protocol, Mapping)
    if (
        source_runtime.get("model_id") != model_id
        or source_runtime.get("requested_revision") != revision
        or source_runtime.get("device") != device_name
        or source_runtime.get("dtype") != dtype
    ):
        raise ValueError("A5b runtime must exactly replay published A4")

    bundle, authority, _, fit_authorization = _authenticate_before_fit_access(
        bundle_path=composition_file,
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    if (
        _canonical_json_bytes(source_authorization.get("bundle"))
        != _canonical_json_bytes(fit_authorization.get("bundle"))
        or _canonical_json_bytes(source_authorization.get("fit_authority"))
        != _canonical_json_bytes(fit_authorization.get("fit_authority"))
        or source_authorization.get("fit_authority_sha256")
        != fit_authorization.get("fit_authority_sha256")
    ):
        raise ValueError("live A-fit authority differs from published A4")
    bundle_binding = getattr(bundle, "binding", None)
    primary_graph = getattr(bundle, "primary", None)
    bundle_lowerings = getattr(bundle, "lowerings", None)
    if (
        not isinstance(bundle_binding, Mapping)
        or not isinstance(primary_graph, ModalGeneratorGraphPlan)
        or not isinstance(bundle_lowerings, tuple)
    ):
        raise TypeError("authenticated composition runtime is unavailable")
    layer10_graph, layer17_graph, layer10_lowerings, layer17_lowerings = (
        _source_lowering_maps(bundle)
    )
    _, fragment_by_node = _validate_source_decoder_contract(
        layer17_graph, layer17_lowerings, protocol
    )
    fragment_plans = {
        lowering.fragment_plan.artifact_sha256: lowering.fragment_plan
        for lowering in layer17_lowerings.values()
    }
    if len(fragment_plans) != 1:
        raise ValueError("Layer17 source lowerings use different fragment plans")
    selection: SameLayerFragmentSelection = select_top_fisher_same_layer_fragments(
        next(iter(fragment_plans.values())),
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    _validate_frozen_selection(selection)
    if tuple(selection.fragment_ids) != tuple(fragment_by_node.values()):
        raise ValueError("Layer17 selected fragment order differs from A4")
    catalog = _build_source_runtime_catalog(
        bundle_binding=bundle_binding,
        primary_graph=primary_graph,
        layer10_graph=layer10_graph,
        layer17_graph=layer17_graph,
        layer10_lowerings_by_node=layer10_lowerings,
        layer17_lowerings_by_node=layer17_lowerings,
        selection=selection,
    )
    if catalog.get("catalog_sha256") != _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256:
        raise ValueError("live source runtime catalog is not frozen")
    _validate_source_runtime_catalog(
        catalog, protocol=protocol, bundle_binding=bundle_binding
    )
    if (
        _canonical_json_bytes(source_authorization.get("source_runtime_catalog"))
        != _canonical_json_bytes(catalog)
        or a4_oracle.get("source_bindings", {}).get(
            "source_runtime_catalog_sha256"
        )
        != catalog.get("catalog_sha256")
    ):
        raise ValueError("live source runtime catalog differs from A4 oracle")

    bases_by_node: dict[str, ComputationalModeBasis] = {
        name: layer17_lowerings[name].computational_mode_basis
        for name in layer17_graph.traversal_order
    }
    image = build_frozen_affine_image(
        bases_by_node, node_order=layer17_graph.traversal_order
    )
    randomness = source_runtime.get("randomness")
    inherited = (
        randomness.get("inherits_seed_and_execution_recipe_from")
        if isinstance(randomness, Mapping)
        else None
    )
    seed = inherited.get("torch_seed") if isinstance(inherited, Mapping) else None
    if type(seed) is not int:
        raise ValueError("published A4 deterministic seed is unavailable")
    torch.manual_seed(seed)
    device = resolve_torch_device(device_name)

    _progress("model: load pinned local Gemma checkpoint")
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
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
    if adapter.model_fingerprint() != source_runtime.get("model_fingerprint"):
        raise ValueError("live Gemma fingerprint differs from published A4")

    _progress("tokenize: replay sealed family authority")
    raw_blocks, materialization = materialize_gemma3_layer17_family_lofo(
        authority, tokenizer
    )
    validate_gemma3_layer17_family_lofo_materialization_metadata(materialization)
    fit_collection = source_report.get("fit_collection")
    if (
        not isinstance(fit_collection, Mapping)
        or _canonical_json_bytes(materialization)
        != _canonical_json_bytes(fit_collection.get("materialization"))
    ):
        raise ValueError("live A-fit materialization differs from published A4")
    blocks = dict(_blocks_to_device(_family_blocks(raw_blocks), device))
    fold = _fold_catalog(protocol)[_OUTER_FOLD_INDEX]
    held_alias = str(fold["held_family_alias"])
    training_aliases = tuple(str(value) for value in fold["training_family_aliases"])
    if len(training_aliases) != _EXPECTED_TRAINING_FAMILIES:
        raise ValueError("A5b outer fold training-family count drifted")

    training_batches = []
    family_alias_by_example: dict[str, str] = {}
    training_example_ids: list[str] = []
    for alias in training_aliases:
        selected = _take_first_examples(
            blocks[alias], _TRAIN_EXAMPLES_PER_FAMILY
        )
        training_batches.extend(selected)
        for batch in selected:
            if batch.example_ids is None or len(batch.example_ids) != 1:
                raise ValueError("A5b bounded training examples require identities")
            example_id = batch.example_ids[0]
            if example_id in family_alias_by_example:
                raise ValueError("A5b training example identity is duplicated")
            family_alias_by_example[example_id] = alias
            training_example_ids.append(example_id)
    if (
        len(training_batches) != _EXPECTED_CAPTURE_EXAMPLES
        or held_alias in family_alias_by_example.values()
    ):
        raise RuntimeError("A5b bounded training selection leaked the held family")

    layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        tuple(layer10_lowerings[name] for name in layer10_graph.traversal_order),
    )
    source_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer17_graph,
        tuple(layer17_lowerings[name] for name in layer17_graph.traversal_order),
    )
    source_lowering_by_name = {
        node.name: lowering
        for node, lowering in zip(primary_graph.nodes, bundle_lowerings, strict=True)
    }
    frozen_primary_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        primary_graph,
        _ordered_restored_lowerings(primary_graph, source_lowering_by_name),
    )

    _progress("capture: seven-family bounded Layer17 trajectories")
    capture = capture_gemma3_layer17_full_block_closure(
        adapter,
        tuple(training_batches),
        selection=selection,
        leaf_activation_site=selection.execution_order[0].input_site,
        layer10_executor=layer10_executor,
        layer17_executor=source_layer17_executor,
    )
    capture_metadata = capture.metadata()
    capture_audit = _full_block_capture_audit_receipt(capture_metadata, protocol)
    if capture_audit.get("all_required_capture_audits_pass") is not True:
        raise RuntimeError("A5b bounded capture audits failed")
    indices, bounded_row_keys = _select_first_rows_per_example(
        capture.native_rows.row_keys,
        expected_examples=training_example_ids,
        rows_per_example=_ROWS_PER_EXAMPLE,
    )
    fragment_ids = tuple(capture.trajectory_rows.fragment_ids)
    compiled_inputs = _shared_compiled_input(
        capture.compiled_rows, fragment_ids
    ).index_select(0, indices).contiguous()
    native_state = capture.native_block_output.index_select(0, indices).to(
        dtype=torch.float32
    ).contiguous()
    post_attention = capture.compiled_post_attention_residual.index_select(
        0, indices
    ).to(dtype=torch.float32).contiguous()
    retained_delta = (
        capture.compiled_compact_retained_post_feedforward_delta.index_select(
            0, indices
        ).to(dtype=torch.float32).contiguous()
    )
    target = capture.a4_full_block_closure_target.index_select(
        0, indices
    ).contiguous()
    a4_projection = project_joint_target_to_frozen_bases(
        target,
        bases_by_node,
        node_order=layer17_graph.traversal_order,
    )
    a4_initial_correction = a4_projection.prediction.to(
        dtype=torch.float32
    ).contiguous()
    del a4_projection

    _progress("solve: batched per-token downstream coordinate targets")
    target_solution = solve_batched_frozen_affine_capacity_rows(
        adapter=adapter,
        image=image,
        native_state=native_state,
        compiled_post_attention_residual=post_attention,
        compiled_compact_retained_delta=retained_delta,
        target_correction=target,
        a4_float64_projection_correction=a4_initial_correction,
        row_chunk_size=_TARGET_BATCH_ROWS,
        token_locality_atol=_TOKEN_LOCALITY_ATOL,
        token_locality_rtol=_TOKEN_LOCALITY_RTOL,
        solver_config=DownstreamAffineSolverConfig(
            steps=_TARGET_SOLVER_STEPS,
            learning_rate=_TARGET_LEARNING_RATE_FRACTION,
            ridge=0.0,
            trust_radius=None,
        ),
    )
    target_receipt = target_solution.receipt
    target_hashes = target_receipt.get("hashes")
    if not isinstance(target_hashes, Mapping):
        raise RuntimeError("A5b target solver hashes are unavailable")
    target_report_receipt = {
        **target_receipt,
        "receipt_sha256": _sha256_value(
            _TARGET_RECEIPT_DOMAIN,
            target_receipt,
        ),
    }

    inner_split_binding = _sha256_value(
        _INNER_SPLIT_DOMAIN,
        {
            "a5a_report_sha256": a5a["report_sha256"],
            "protocol_fold_sha256": fold["artifact_sha256"],
            "capture_sha256": capture.capture_sha256,
            "bounded_row_catalog_sha256": _sha256_value(
                b"a5b:bounded-row-catalog:\0", bounded_row_keys
            ),
            "configuration": {
                "examples_per_family": _TRAIN_EXAMPLES_PER_FAMILY,
                "rows_per_example": _ROWS_PER_EXAMPLE,
                "audit_examples_per_family": _INNER_AUDIT_EXAMPLES_PER_FAMILY,
            },
        },
    )
    fisher_by_node = {
        name: capture.native_rows.rows_by_fragment[
            fragment_by_node[name]
        ].fisher_weights.index_select(0, indices).contiguous()
        for name in layer17_graph.traversal_order
    }
    bridge = build_a5b_downstream_coordinate_target_bridge(
        compiled_inputs=compiled_inputs,
        authenticated_compiled_inputs_sha256=a5b_tensor_sha256(compiled_inputs),
        selected_joint_coordinates=target_solution.selected_coefficients,
        authenticated_joint_coordinates_sha256=str(
            target_hashes["selected_coefficient_sha256"]
        ),
        bases_by_node=bases_by_node,
        node_order=layer17_graph.traversal_order,
        fisher_weights_by_node=fisher_by_node,
        fragment_id_by_node=fragment_by_node,
        row_keys=bounded_row_keys,
        family_alias_by_example=family_alias_by_example,
        training_family_aliases=training_aliases,
        held_family_alias=held_alias,
        inner_split_binding_sha256=inner_split_binding,
        inner_audit_examples_per_family=_INNER_AUDIT_EXAMPLES_PER_FAMILY,
    )
    bridge_receipt = bridge.receipt()
    fit_split_sha = _sha256_value(
        _FITTER_SPLIT_DOMAIN,
        {
            "role": "fit",
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "row_key_sha256": bridge.fit_rows.row_key_sha256,
            "observations": bridge.fit_rows.observations,
        },
    )
    audit_split_sha = _sha256_value(
        _FITTER_SPLIT_DOMAIN,
        {
            "role": "audit",
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "row_key_sha256": bridge.audit_rows.row_key_sha256,
            "observations": bridge.audit_rows.observations,
        },
    )

    _progress("fit: distill coordinates into frozen-basis rank-16 generators")
    correction = fit_frozen_basis_coordinate_generators(
        bridge.fit_rows,
        bridge.audit_rows,
        source_graph=layer17_graph,
        source_lowerings_by_node=layer17_lowerings,
        fit_split_sha256=fit_split_sha,
        eval_split_sha256=audit_split_sha,
        generator_rank=_GENERATOR_RANK,
        ridge=_RIDGE,
        output_boundary=_POST_DELTA_BOUNDARY,
    )
    fit_diagnostics = {
        name: {
            "fit_weighted_nrmse": curve.point_for_rank(
                _GENERATOR_RANK
            ).fit_metrics.weighted_nrmse,
            "audit_weighted_nrmse": curve.point_for_rank(
                _GENERATOR_RANK
            ).eval_metrics.weighted_nrmse,
            "fit_weighted_cosine": curve.point_for_rank(
                _GENERATOR_RANK
            ).fit_metrics.weighted_cosine_similarity,
            "audit_weighted_cosine": curve.point_for_rank(
                _GENERATOR_RANK
            ).eval_metrics.weighted_cosine_similarity,
        }
        for name, curve in correction.rate_curves_by_node.items()
    }
    _progress(
        "fit audit: "
        + json.dumps(fit_diagnostics, sort_keys=True, separators=(",", ":"))
    )
    correction_metadata = correction.metadata()
    if (
        correction.graph_plan.parameter_count != 163_094
        or correction.graph_plan.macs_per_token != 160_352
        or correction.graph_plan.interactions
        or any(
            node.output_boundary != _POST_DELTA_BOUNDARY
            for node in correction.graph_plan.nodes
        )
    ):
        raise RuntimeError("A5b generator escaped the frozen Layer17 resources")
    corrected_primary = replace_layer_nodes_in_composed_graph(
        primary_graph, correction.graph_plan, layer_ordinal=17
    )
    if (
        corrected_primary.parameter_count != 295_129
        or corrected_primary.macs_per_token != 289_600
        or corrected_primary.interactions != primary_graph.interactions
    ):
        raise RuntimeError("A5b corrected composition resources drifted")
    corrected_lowerings = _merge_corrected_composition_lowerings(
        corrected_primary,
        layer10_lowerings_by_node=layer10_lowerings,
        corrected_layer17_lowerings_by_node=correction.lowerings_by_node,
    )
    corrected_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        correction.graph_plan,
        tuple(
            correction.lowerings_by_node[name]
            for name in correction.graph_plan.traversal_order
        ),
        post_feedforward_delta_layer_ordinals=(17,),
    )
    corrected_primary_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        corrected_primary,
        corrected_lowerings,
        post_feedforward_delta_layer_ordinals=(17,),
    )
    lowering_hashes = {
        name: correction.lowerings_by_node[name].artifact_sha256
        for name in correction.graph_plan.traversal_order
    }
    executable_freeze_sha = _sha256_value(
        b"fisher-graph:a5b-executable-freeze:v1\0",
        {
            "layer17_graph_sha256": correction.graph_plan.artifact_sha256,
            "lowering_sha256_by_node": lowering_hashes,
            "composition_graph_sha256": corrected_primary.artifact_sha256,
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "fit_split_sha256": fit_split_sha,
            "audit_split_sha256": audit_split_sha,
        },
    )

    # The held-family batch is selected only after every fitted/executable
    # artifact identity above is frozen.
    held_batches = _take_first_examples(blocks[held_alias], 1)
    _progress("score: first untouched outer-family example through full model")
    evaluation = score_trajectory_correction_fold(
        adapter=adapter,
        layer10_executor=layer10_executor,
        corrected_layer17_executor=corrected_layer17_executor,
        frozen_composition_executor=frozen_primary_executor,
        corrected_composition_executor=corrected_primary_executor,
        batches=held_batches,
    )

    bounded_row_catalog_sha = _sha256_value(
        b"a5b:bounded-row-catalog:\0", bounded_row_keys
    )
    report = build_a5b_generator_microcanary_report(
        source_bindings={
            "a5a_file_sha256": _file_sha256(a5a_file),
            "a5a_report_sha256": str(a5a["report_sha256"]),
            "a4_oracle_file_sha256": _file_sha256(a4_oracle_file),
            "a4_oracle_report_sha256": str(a4_oracle["report_sha256"]),
            "a4_report_file_sha256": _file_sha256(a4_file),
            "a4_report_sha256": str(source_report["report_sha256"]),
            "composition_bundle_file_sha256": _file_sha256(composition_file),
            "composition_payload_sha256": str(
                bundle_binding["composition_payload_sha256"]
            ),
            "fold_bundle_file_sha256": _file_sha256(fold_file),
            "fold_bundle_payload_sha256": str(
                fold_bundle["scientific_payload_sha256"]
            ),
            "protocol_sha256": str(protocol["artifact_sha256"]),
            "source_runtime_catalog_sha256": str(catalog["catalog_sha256"]),
        },
        runtime={
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "device": device_name,
            "dtype": dtype,
            "local_files_only": True,
        },
        capture={
            "capture_sha256": capture.capture_sha256,
            "capture_audit_sha256": _sha256_value(
                b"a5b:capture-audit:\0", capture_audit
            ),
            "source_row_catalog_sha256": capture.native_rows.row_key_sha256,
            "bounded_row_catalog_sha256": bounded_row_catalog_sha,
            "training_examples": len(training_example_ids),
            "captured_observations": capture.native_rows.observations,
            "bounded_target_rows": len(bounded_row_keys),
            "held_family_rows_present": False,
            "all_required_capture_audits_pass": True,
        },
        target_solve=target_report_receipt,
        target_bridge=bridge_receipt,
        generator_fit=correction_metadata,
        frozen_executable={
            "corrected_layer17_graph_sha256": correction.graph_plan.artifact_sha256,
            "corrected_layer17_lowering_sha256_by_node": lowering_hashes,
            "corrected_composition_graph_sha256": corrected_primary.artifact_sha256,
            "bridge_receipt_sha256": bridge.receipt_sha256,
            "fit_split_sha256": fit_split_sha,
            "audit_split_sha256": audit_split_sha,
            "executable_freeze_sha256": executable_freeze_sha,
            "corrected_layer17_parameters": correction.graph_plan.parameter_count,
            "corrected_layer17_macs_per_token": correction.graph_plan.macs_per_token,
            "corrected_composition_parameters": corrected_primary.parameter_count,
            "corrected_composition_macs_per_token": corrected_primary.macs_per_token,
            (
                "generator_and_composition_hashes_frozen_before_held_"
                "example_selection_or_model_evaluation"
            ): True,
        },
        evaluation=evaluation,
    )
    saved = save_a5b_generator_microcanary_report(destination, report)
    _progress(f"published: {destination}")
    del capture, target_solution, bridge, correction
    gc.collect()
    return saved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT,
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_l10_l17_a5b_generator_microcanary(
        revision=args.revision,
        output=args.output,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_OUTPUT",
    "GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_FORMAT_VERSION",
    "GEMMA3_L10_L17_A5B_GENERATOR_MICROCANARY_SCHEMA",
    "build_a5b_generator_microcanary_report",
    "load_a5b_generator_microcanary_report",
    "run_gemma3_l10_l17_a5b_generator_microcanary",
    "save_a5b_generator_microcanary_report",
    "validate_a5b_generator_microcanary_report",
]
