"""Eight-family full-logit evaluation of Gemma A4 full-block closure.

This runner is an A-fit-only successor to the failed A3 trajectory correction.
It changes the Layer-17 target and application boundary, not the corpus,
Fisher weighting, decoder tensors, ranks, Layer-10 graph, or capacity.  The
generated Layer-17 contribution is added after Gemma's post-feed-forward
RMSNorm so that the target has the exact block-space meaning frozen by the A4
protocol.

All activation rows and fitted projection tensors are ephemeral.  The JSON
report contains scalar metrics and cryptographic lineage only.  A companion
``.pt`` artifact retains the eight fold-local graph/lowering executables so
later causal diagnostics do not require refitting.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

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
from .gemma3_l10_l17_full_block_closure_bundle import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    build_gemma3_l10_l17_full_block_closure_fold_bundle,
    save_gemma3_l10_l17_full_block_closure_fold_bundle,
)
from .gemma3_l10_l17_full_block_closure_protocol import (
    FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256,
    build_default_gemma3_l10_l17_full_block_closure_protocol,
    validate_gemma3_l10_l17_full_block_closure_protocol,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
    _record_execution,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    fit_frozen_basis_coordinate_generators,
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_protocol import (
    build_default_gemma3_l10_l17_trajectory_correction_protocol,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    DEFAULT_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_OUTPUT,
    _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256,
    _TrajectoryCorrectionFitView,
    _authenticate_before_fit_access,
    _build_source_runtime_catalog,
    _build_trajectory_correction_fold_rows_from_fit_view,
    _capture_metadata_sha256,
    _fold_catalog,
    _merge_corrected_composition_lowerings,
    _projection_receipt_matches_protocol,
    _randomness_recipe_receipt,
    _reject_forbidden_output_fields,
    _shared_compiled_input,
    _source_lowering_maps,
    _validate_metric_row,
    _validate_source_decoder_contract,
    _validate_source_runtime_catalog,
    aggregate_trajectory_correction_lofo_folds,
    evaluate_trajectory_correction_lofo_gates,
    load_gemma3_l10_l17_trajectory_correction_lofo_report,
    score_trajectory_correction_fold,
)
from .gemma3_layer10_v8_corpus import (
    DEFAULT_CORPUS_OUTPUT,
    DEFAULT_FIT_OUTPUT,
    DEFAULT_RECEIPT_OUTPUT,
)
from .gemma3_layer17_capped_node_fit import _validate_frozen_selection
from .gemma3_layer17_family_lofo_authority import (
    materialize_gemma3_layer17_family_lofo,
    validate_gemma3_layer17_family_lofo_materialization_metadata,
)
from .gemma3_layer17_full_block_closure_capture import (
    Gemma3Layer17FullBlockClosureCapture,
    capture_gemma3_layer17_full_block_closure,
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
from .gemma3_layer17_v8_fit_lofo import (
    _blocks_to_device,
    _family_blocks,
    _ordered_restored_lowerings,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
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
    "DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT",
    "GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_FORMAT_VERSION",
    "GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_SCHEMA",
    "aggregate_full_block_closure_folds",
    "load_gemma3_l10_l17_full_block_closure_lofo_report",
    "run_gemma3_l10_l17_full_block_closure_lofo",
    "save_gemma3_l10_l17_full_block_closure_lofo_report",
    "validate_gemma3_l10_l17_full_block_closure_lofo_report",
]


GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_full_block_closure_lofo"
)
GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a4-full-block-a-fit-lofo-v1.json"
)

_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-lofo-report:v1\0"
_COMPOSITION_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-composition:v1\0"
_RANDOMNESS_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-randomness:v1\0"
_SPLIT_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-split-receipt:v1\0"
_A4_CAPTURE_DOMAIN = b"fisher-graph:gemma3-layer17-full-block-closure:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FAMILIES = 8
_GENERATOR_RANK = 16
_RIDGE = 0.0
_POST_DELTA_BOUNDARY = "layer.17.mlp.delta"
_VOCABULARY_CHUNK_SIZE = 16_384

_CONDITIONS = (
    "layer10_only",
    "source_layer17_only",
    "a4_full_block_layer17_only",
    "frozen_uncorrected_composition",
    "a4_full_block_corrected_composition",
    "l10_edgeless_frozen_l17_composition",
    "l10_edgeless_a4_composition",
    "matched_double_deletion",
)

_PRIMARY_RESOURCES = {
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

_AUX_RESOURCE_EXPECTATIONS = {
    "source_layer17_only": {
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
    "l10_edgeless_frozen_l17_composition": {
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
    "l10_edgeless_a4_composition": {
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
    "source_layer17_only": dict(
        _AUX_RESOURCE_EXPECTATIONS["source_layer17_only"]
    ),
    "a4_full_block_layer17_only": dict(
        _AUX_RESOURCE_EXPECTATIONS["source_layer17_only"]
    ),
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
    "a4_full_block_corrected_composition": {
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
    "l10_edgeless_frozen_l17_composition": dict(
        _AUX_RESOURCE_EXPECTATIONS[
            "l10_edgeless_frozen_l17_composition"
        ]
    ),
    "l10_edgeless_a4_composition": dict(
        _AUX_RESOURCE_EXPECTATIONS["l10_edgeless_a4_composition"]
    ),
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

_SAFETY = {
    "contains_prompt_text": False,
    "contains_prompt_identities": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_gradient_or_projection_tensors": False,
    "contains_model_or_candidate_weights": False,
    "source_safe": True,
}

_REPORT_FIELDS = {
    "schema",
    "format_version",
    "scientific_role",
    "heldout_confirmation",
    "protocol",
    "authorization",
    "prior_a3_comparison",
    "runtime",
    "fit_collection",
    "fold_executable_bundle",
    "folds",
    "aggregate",
    "resources",
    "interaction_factorial",
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
        value,
        (str, bytes, bytearray),
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(message: str) -> None:
    print(f"[a4-full-block] {message}", file=os.sys.stderr, flush=True)


def _metric_close(left: object, right: object, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def _compact_resources(
    *,
    condition: str,
    plan: ModalGeneratorGraphPlan,
    static: Mapping[str, object],
    totals: Mapping[str, int],
    logical_valid_tokens: int,
    peak_live_modal_width: int,
) -> dict[str, int]:
    expected = _AUX_RESOURCE_EXPECTATIONS[condition]
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
    if observed != expected:
        raise RuntimeError(f"{condition} resource accounting drifted: {observed}")
    return {**observed, "executed_peak_live_modal_width": peak_live_modal_width}


def _score_auxiliary_conditions(
    *,
    adapter: Gemma3CausalLMAdapter,
    executors: Mapping[str, Gemma3ModalGeneratorGraphExecutor],
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    names = tuple(executors)
    if names != tuple(_AUX_RESOURCE_EXPECTATIONS):
        raise ValueError("A4 auxiliary condition order is not frozen")
    if len({id(value) for value in executors.values()}) != len(executors):
        raise ValueError("A4 auxiliary executors must be distinct")
    plans = {name: executor.graph_plan for name, executor in executors.items()}
    if (
        executors["source_layer17_only"].affected_layer_ordinals != (17,)
        or executors[
            "l10_edgeless_frozen_l17_composition"
        ].affected_layer_ordinals
        != (10, 17)
        or executors["l10_edgeless_a4_composition"].affected_layer_ordinals
        != (10, 17)
        or any(plan.interactions for plan in plans.values())
        or executors["source_layer17_only"].post_feedforward_delta_layer_ordinals
        or executors[
            "l10_edgeless_frozen_l17_composition"
        ].post_feedforward_delta_layer_ordinals
        or executors[
            "l10_edgeless_a4_composition"
        ].post_feedforward_delta_layer_ordinals
        != (17,)
    ):
        raise ValueError("A4 auxiliary executor topology/boundary drifted")

    aggregate = _new_metric_accumulator(names)
    static_by_condition: dict[str, dict[str, object]] = {}
    totals_by_condition: dict[str, dict[str, int]] = {}
    peak_by_condition: dict[str, int] = {}
    logical_valid_tokens = 0
    with ExitStack() as stack:
        for executor in executors.values():
            stack.enter_context(executor.validated_transaction())
        for batch in batches:
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            with torch.no_grad():
                native_output = adapter.module(**call_inputs)
            native_logits, targets = _selected_logits_and_targets(
                _model_logits(native_output),
                batch,
            )
            _add_native(
                aggregate,
                nll_sum=_native_nll(native_logits, targets),
                token_count=targets.numel(),
            )
            expected_valid = int(batch.valid_positions.sum().item())
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
                    _model_logits(execution.model_output),
                    batch,
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{condition} targets drifted")
                _add_comparison(
                    aggregate,
                    condition,
                    _candidate_comparison(
                        native_logits,
                        logits,
                        targets,
                        vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                    ),
                )
                _record_execution(
                    static_by_condition,
                    totals_by_condition,
                    peak_by_condition,
                    condition=condition,
                    execution=execution,
                )
                if getattr(execution, "valid_tokens", None) != expected_valid:
                    raise RuntimeError(f"{condition} valid-token count drifted")
            logical_valid_tokens += expected_valid
    metrics = _finalize_metric_accumulator(aggregate, conditions=names)
    resources = {
        condition: _compact_resources(
            condition=condition,
            plan=plans[condition],
            static=static_by_condition[condition],
            totals=totals_by_condition[condition],
            logical_valid_tokens=logical_valid_tokens,
            peak_live_modal_width=peak_by_condition[condition],
        )
        for condition in names
    }
    return {
        "supervised_tokens": metrics["supervised_tokens"],
        "logical_valid_tokens": logical_valid_tokens,
        "native": metrics["native"],
        "conditions": metrics["conditions"],
        "resource_accounting": resources,
    }


def _combine_fold_evaluations(
    core: Mapping[str, object],
    auxiliary: Mapping[str, object],
) -> dict[str, object]:
    core_conditions = core.get("conditions")
    auxiliary_conditions = auxiliary.get("conditions")
    core_resources = core.get("resource_accounting")
    auxiliary_resources = auxiliary.get("resource_accounting")
    if not all(
        isinstance(value, Mapping)
        for value in (
            core_conditions,
            auxiliary_conditions,
            core_resources,
            auxiliary_resources,
        )
    ):
        raise TypeError("A4 fold metric/resource panels are incomplete")
    assert isinstance(core_conditions, Mapping)
    assert isinstance(auxiliary_conditions, Mapping)
    assert isinstance(core_resources, Mapping)
    assert isinstance(auxiliary_resources, Mapping)
    if (
        core.get("supervised_tokens") != auxiliary.get("supervised_tokens")
        or core.get("logical_valid_tokens")
        != auxiliary.get("logical_valid_tokens")
        or not _metric_close(
            core["native"]["nll_per_token"],  # type: ignore[index]
            auxiliary["native"]["nll_per_token"],  # type: ignore[index]
        )
    ):
        raise RuntimeError("A4 core/auxiliary native scoring disagreed")
    conditions = {
        "layer10_only": core_conditions["layer10_only"],
        "source_layer17_only": auxiliary_conditions["source_layer17_only"],
        "a4_full_block_layer17_only": core_conditions[
            "trajectory_corrected_layer17_only"
        ],
        "frozen_uncorrected_composition": core_conditions[
            "frozen_uncorrected_composition"
        ],
        "a4_full_block_corrected_composition": core_conditions[
            "trajectory_corrected_composition"
        ],
        "l10_edgeless_frozen_l17_composition": auxiliary_conditions[
            "l10_edgeless_frozen_l17_composition"
        ],
        "l10_edgeless_a4_composition": auxiliary_conditions[
            "l10_edgeless_a4_composition"
        ],
        "matched_double_deletion": core_conditions[
            "matched_double_deletion"
        ],
    }
    resources = {
        "layer10_only": core_resources["layer10_only"],
        "source_layer17_only": auxiliary_resources["source_layer17_only"],
        "a4_full_block_layer17_only": core_resources[
            "trajectory_corrected_layer17_only"
        ],
        "frozen_uncorrected_composition": core_resources[
            "frozen_uncorrected_composition"
        ],
        "a4_full_block_corrected_composition": core_resources[
            "trajectory_corrected_composition"
        ],
        "l10_edgeless_frozen_l17_composition": auxiliary_resources[
            "l10_edgeless_frozen_l17_composition"
        ],
        "l10_edgeless_a4_composition": auxiliary_resources[
            "l10_edgeless_a4_composition"
        ],
        "matched_double_deletion": core_resources["matched_double_deletion"],
    }
    if tuple(conditions) != _CONDITIONS or tuple(resources) != _CONDITIONS:
        raise RuntimeError("A4 condition panel order drifted")
    return {
        "execution_path": "full_model_logits_a4_full_block_closure_lofo",
        "assessment_role": "calibration_a_fit_family_blocked_development",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": core["supervised_tokens"],
        "logical_valid_tokens": core["logical_valid_tokens"],
        "native": dict(core["native"]),  # type: ignore[arg-type]
        "conditions": conditions,
        "resource_accounting": resources,
        "application_boundary": _POST_DELTA_BOUNDARY,
        "latency_or_kernel_speed_claim": False,
    }


def _validate_full_block_evaluation(
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
        "application_boundary",
        "latency_or_kernel_speed_claim",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} evaluation fields are invalid")
    if (
        value.get("execution_path")
        != "full_model_logits_a4_full_block_closure_lofo"
        or value.get("assessment_role")
        != "calibration_a_fit_family_blocked_development"
        or value.get("heldout_confirmation") is not False
        or value.get("full_model_logits_scored") is not True
        or value.get("full_model_compiled") is not False
        or value.get("application_boundary") != _POST_DELTA_BOUNDARY
        or value.get("latency_or_kernel_speed_claim") is not False
        or type(value.get("supervised_tokens")) is not int
        or int(value.get("supervised_tokens", 0)) <= 0
        or type(value.get("logical_valid_tokens")) is not int
        or int(value.get("logical_valid_tokens", 0)) <= 0
    ):
        raise ValueError(f"{label} evaluation boundary is invalid")
    native = value.get("native")
    conditions = value.get("conditions")
    resources = value.get("resource_accounting")
    if (
        not isinstance(native, Mapping)
        or set(native) != {"nll_per_token"}
        or not math.isfinite(float(native["nll_per_token"]))
        or float(native["nll_per_token"]) < 0.0
        or not isinstance(conditions, Mapping)
        or set(conditions) != set(_CONDITIONS)
        or not isinstance(resources, Mapping)
        or set(resources) != set(_CONDITIONS)
    ):
        raise ValueError(f"{label} metric/resource catalog is invalid")
    native_nll = float(native["nll_per_token"])
    for condition in _CONDITIONS:
        metric = _validate_metric_row(
            conditions[condition],
            label=f"{label}/{condition}",
        )
        if not _metric_close(
            metric["nll_per_token"],
            native_nll + float(metric["delta_nll_per_token"]),
        ):
            raise ValueError(f"{label}/{condition} NLL identity drifted")
        resource = resources[condition]
        expected = _EXPECTED_CONDITION_RESOURCES[condition]
        if (
            not isinstance(resource, Mapping)
            or set(resource) != {*expected, "executed_peak_live_modal_width"}
            or any(resource.get(key) != expected_value for key, expected_value in expected.items())
            or type(resource.get("executed_peak_live_modal_width")) is not int
            or int(resource.get("executed_peak_live_modal_width", -1)) < 0
            or (
                condition == "matched_double_deletion"
                and resource.get("executed_peak_live_modal_width") != 0
            )
        ):
            raise ValueError(f"{label}/{condition} resources drifted")
    return value


def aggregate_full_block_closure_folds(
    fold_evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return micro and equal-family summaries for exactly eight A4 folds."""

    folds = tuple(
        _validate_full_block_evaluation(value, label=f"fold {index}")
        for index, value in enumerate(fold_evaluations)
    )
    if len(folds) != _FAMILIES:
        raise ValueError("A4 full-block LOFO requires exactly eight folds")
    totals = _new_metric_accumulator(_CONDITIONS)
    for value in folds:
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
    micro = _finalize_metric_accumulator(totals, conditions=_CONDITIONS)
    macro = {
        "native": {
            "nll_per_token": math.fsum(
                float(value["native"]["nll_per_token"])  # type: ignore[index]
                for value in folds
            )
            / _FAMILIES
        },
        "conditions": {
            condition: {
                metric: math.fsum(
                    float(value["conditions"][condition][metric])  # type: ignore[index]
                    for value in folds
                )
                / _FAMILIES
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
        "family_count": _FAMILIES,
        "completed_fold_count": len(folds),
        "failed_fold_count": 0,
    }


def _as_a3_gate_evaluation(
    value: Mapping[str, object],
) -> dict[str, object]:
    checked = _validate_full_block_evaluation(value, label="A4 gate fold")
    conditions = checked["conditions"]
    resources = checked["resource_accounting"]
    assert isinstance(conditions, Mapping)
    assert isinstance(resources, Mapping)
    return {
        "execution_path": "full_model_logits_fixed_capacity_a3_trajectory_lofo",
        "assessment_role": checked["assessment_role"],
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "supervised_tokens": checked["supervised_tokens"],
        "logical_valid_tokens": checked["logical_valid_tokens"],
        "native": dict(checked["native"]),  # type: ignore[arg-type]
        "conditions": {
            "layer10_only": dict(conditions["layer10_only"]),  # type: ignore[arg-type]
            "trajectory_corrected_layer17_only": dict(
                conditions["a4_full_block_layer17_only"]  # type: ignore[arg-type]
            ),
            "frozen_uncorrected_composition": dict(
                conditions["frozen_uncorrected_composition"]  # type: ignore[arg-type]
            ),
            "trajectory_corrected_composition": dict(
                conditions["a4_full_block_corrected_composition"]  # type: ignore[arg-type]
            ),
            "matched_double_deletion": dict(
                conditions["matched_double_deletion"]  # type: ignore[arg-type]
            ),
        },
        "resource_accounting": {
            "layer10_only": dict(resources["layer10_only"]),  # type: ignore[arg-type]
            "trajectory_corrected_layer17_only": dict(
                resources["a4_full_block_layer17_only"]  # type: ignore[arg-type]
            ),
            "frozen_uncorrected_composition": dict(
                resources["frozen_uncorrected_composition"]  # type: ignore[arg-type]
            ),
            "trajectory_corrected_composition": dict(
                resources["a4_full_block_corrected_composition"]  # type: ignore[arg-type]
            ),
            "matched_double_deletion": dict(
                resources["matched_double_deletion"]  # type: ignore[arg-type]
            ),
        },
        "exact_resources_match_protocol": True,
        "latency_or_kernel_speed_claim": False,
    }


def _shared_control_receipt(
    current: Mapping[str, object],
    prior: Mapping[str, object],
) -> dict[str, object]:
    """Require every unchanged condition to reproduce the sealed A3 run."""

    current_checked = _validate_full_block_evaluation(
        current,
        label="current shared controls",
    )
    prior_conditions = prior.get("conditions")
    prior_native = prior.get("native")
    current_conditions = current_checked["conditions"]
    current_native = current_checked["native"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            prior_conditions,
            prior_native,
            current_conditions,
            current_native,
        )
    ):
        raise TypeError("shared-control metric panels are incomplete")
    assert isinstance(prior_conditions, Mapping)
    assert isinstance(prior_native, Mapping)
    assert isinstance(current_conditions, Mapping)
    assert isinstance(current_native, Mapping)
    pairs = {
        "native": (
            float(current_native["nll_per_token"]),
            float(prior_native["nll_per_token"]),
        )
    }
    for current_name, prior_name in (
        ("layer10_only", "layer10_only"),
        (
            "frozen_uncorrected_composition",
            "frozen_uncorrected_composition",
        ),
        ("matched_double_deletion", "matched_double_deletion"),
    ):
        current_row = current_conditions[current_name]
        prior_row = prior_conditions[prior_name]
        if not isinstance(current_row, Mapping) or not isinstance(
            prior_row,
            Mapping,
        ):
            raise TypeError("shared-control condition row is unavailable")
        for metric in (
            "nll_per_token",
            "delta_nll_per_token",
            "native_to_candidate_kl_per_token",
            "top1_agreement_to_native",
        ):
            pairs[f"{current_name}.{metric}"] = (
                float(current_row[metric]),
                float(prior_row[metric]),
            )
    differences = {name: abs(left - right) for name, (left, right) in pairs.items()}
    maximum = max(differences.values(), default=0.0)
    counts_match = (
        current_checked.get("supervised_tokens")
        == prior.get("supervised_tokens")
        and current_checked.get("logical_valid_tokens")
        == prior.get("logical_valid_tokens")
    )
    matched = counts_match and maximum <= 1e-12
    if not matched:
        raise RuntimeError(
            "A4 shared controls do not reproduce sealed A3 evidence: "
            f"counts_match={counts_match}, max_abs={maximum:.3e}"
        )
    return {
        "conditions": [
            "native",
            "layer10_only",
            "frozen_uncorrected_composition",
            "matched_double_deletion",
        ],
        "metric_count": len(pairs),
        "maximum_absolute_difference": maximum,
        "maximum_allowed_absolute_difference": 1e-12,
        "token_counts_match": counts_match,
        "matched": True,
    }


def _full_block_capture_audit_receipt(
    metadata: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    target = protocol.get("target_contract")
    observed_target = metadata.get("target")
    alignment = metadata.get("alignment")
    audits = metadata.get("audits")
    activation_capture = metadata.get("activation_only_capture")
    trajectory_capture = metadata.get("trajectory_capture")
    tensor_sha256s = metadata.get("tensor_sha256s")
    if not all(
        isinstance(value, Mapping)
        for value in (
            target,
            observed_target,
            alignment,
            audits,
            activation_capture,
            trajectory_capture,
            tensor_sha256s,
        )
    ):
        raise TypeError("A4 capture audit metadata is incomplete")
    assert isinstance(target, Mapping)
    assert isinstance(observed_target, Mapping)
    assert isinstance(alignment, Mapping)
    assert isinstance(audits, Mapping)
    assert isinstance(activation_capture, Mapping)
    assert isinstance(trajectory_capture, Mapping)
    assert isinstance(tensor_sha256s, Mapping)
    decomposition_names = (
        "native_block_decomposition",
        "compiled_block_decomposition",
        "a4_reconstruction",
        "a4_equivalent_formula",
        "a4_minus_delta_only_closure_offset_identity",
    )
    maximum_decomposition_max = float(
        target["maximum_decomposition_max_abs_difference"]
    )
    maximum_decomposition_rmse = float(
        target["maximum_decomposition_rmse"]
    )
    rows: dict[str, dict[str, object]] = {}
    for name in (*decomposition_names, "compact_post_feedforward_replay"):
        raw = audits.get(name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"A4 capture audit {name!r} is unavailable")
        maximum_abs = float(raw.get("max_abs_difference", float("nan")))
        rmse = float(raw.get("rms_difference", float("nan")))
        reference_rms = float(raw.get("reference_rms", float("nan")))
        normalized_rmse = float(
            raw.get("normalized_rms_difference", float("nan"))
        )
        if not all(
            math.isfinite(value)
            for value in (
                maximum_abs,
                rmse,
                reference_rms,
                normalized_rmse,
            )
        ) or reference_rms < 0.0 or normalized_rmse < 0.0:
            raise ValueError(f"A4 capture audit {name!r} is non-finite")
        if name == "compact_post_feedforward_replay":
            max_threshold = float(
                target["maximum_compact_equivalence_max_abs_difference"]
            )
            rmse_threshold = float(
                target["maximum_compact_equivalence_rmse"]
            )
            normalized_rmse_threshold = float(
                target["maximum_compact_equivalence_normalized_rmse"]
            )
        else:
            max_threshold = maximum_decomposition_max
            rmse_threshold = maximum_decomposition_rmse
            normalized_rmse_threshold = float(
                target["maximum_decomposition_normalized_rmse"]
            )
        rows[name] = {
            "max_abs_difference": maximum_abs,
            "maximum_max_abs_difference": max_threshold,
            "rms_difference": rmse,
            "maximum_rms_difference": rmse_threshold,
            "reference_rms": reference_rms,
            "normalized_rms_difference": normalized_rmse,
            "maximum_normalized_rms_difference": normalized_rmse_threshold,
            "passed": (
                maximum_abs <= max_threshold
                and rmse <= rmse_threshold
                and normalized_rmse <= normalized_rmse_threshold
            ),
        }
    sites = activation_capture.get("sites")
    nested_alignment = trajectory_capture.get("alignment")
    expected_tensor_hashes = {
        "native_post_attention_residual",
        "native_post_feedforward_delta",
        "native_block_output",
        "compiled_post_attention_residual",
        "compiled_post_feedforward_delta",
        "compiled_block_output",
        "exact_compact_retained_post_feedforward_delta",
        "algebraic_compact_retained_post_feedforward_delta",
        "a4_full_block_closure_target",
        "native_delta_only_closure",
        "residual_stream_closure_offset",
    }
    supplied_capture_sha256 = _require_sha256(
        metadata.get("capture_sha256"),
        label="A4 capture",
    )
    recomputed_capture_sha256 = hashlib.sha256(
        _A4_CAPTURE_DOMAIN
        + _canonical_json_bytes(
            {
                key: value
                for key, value in metadata.items()
                if key != "capture_sha256"
            }
        )
    ).hexdigest()
    structural = (
        metadata.get("scientific_role")
        == "paired_native_and_layer10_compiled_layer17_full_block_closure"
        and metadata.get("source_safe") is True
        and metadata.get("contains_tensors") is False
        and metadata.get("condition") == "generated"
        and tuple(metadata.get("affected_layer_ordinals", ())) == (10,)
        and observed_target.get("variant") == "A4_full_block_closure"
        and observed_target.get("application_boundary") == _POST_DELTA_BOUNDARY
        and observed_target.get("uses_exact_compact_post_feedforward_delta")
        is True
        and observed_target.get("uses_raw_compact_mlp_output") is False
        and observed_target.get("formula")
        == (
            "native_layer17_block_output-"
            "compiled_layer17_post_attention_residual-"
            "exact_compact_retained_layer17_post_feedforward_delta"
        )
        and isinstance(sites, Mapping)
        and sites
        == {
            "post_attention_residual": "layer.17.post_attention",
            "post_feedforward_delta": "layer.17.mlp.delta",
            "block_output": "layer.17.output",
        }
        and activation_capture.get("uses_autograd_grad") is False
        and activation_capture.get("pre_leaf_capture_uses_auxiliary_forward")
        is True
        and activation_capture.get(
            "post_attention_derived_from_block_output_subtraction"
        )
        is False
        and alignment.get(
            "activation_fisher_and_activation_only_row_keys_equal"
        )
        is True
        and alignment.get("native_and_compiled_row_keys_equal") is True
        and type(alignment.get("observations")) is int
        and int(alignment.get("observations", 0)) > 0
        and type(alignment.get("sequences")) is int
        and int(alignment.get("sequences", 0)) > 0
        and alignment.get("fragment_count") == 4
        and isinstance(nested_alignment, Mapping)
        and alignment.get("observations")
        == nested_alignment.get("observations")
        and alignment.get("sequences") == nested_alignment.get("sequences")
        and alignment.get("fragment_count")
        == nested_alignment.get("fragment_count")
        and alignment.get("row_key_sha256")
        == nested_alignment.get("row_key_sha256")
        and trajectory_capture.get("capture_sha256")
        == _capture_metadata_sha256(trajectory_capture)
        and set(tensor_sha256s) == expected_tensor_hashes
        and all(
            isinstance(value, str) and _SHA256.fullmatch(value) is not None
            for value in tensor_sha256s.values()
        )
        and supplied_capture_sha256 == recomputed_capture_sha256
    )
    passed = structural and all(bool(value["passed"]) for value in rows.values())
    return {
        "capture_sha256": supplied_capture_sha256,
        "capture_hash_recomputed": supplied_capture_sha256
        == recomputed_capture_sha256,
        "application_boundary": observed_target.get("application_boundary"),
        "structural_contract_match": structural,
        "numerical_audits": rows,
        "all_required_capture_audits_pass": passed,
    }


def _prior_a3_comparison(
    *,
    protocol: Mapping[str, object],
    prior_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    source = protocol.get("source_authority")
    if not isinstance(source, Mapping):
        raise TypeError("A4 source authority is unavailable")
    expected = source.get("prior_a3_evidence")
    if not isinstance(expected, Mapping):
        raise TypeError("A4 prior-A3 evidence contract is unavailable")
    prior = load_gemma3_l10_l17_trajectory_correction_lofo_report(prior_path)
    if (
        prior_path.name != expected.get("report_file")
        or _file_sha256(prior_path) != expected.get("report_file_sha256")
        or prior.get("report_sha256") != expected.get("report_sha256")
        or prior.get("protocol", {}).get("artifact_sha256")
        != expected.get("protocol_sha256")
        or prior.get("decision", {}).get("all_required_gates_pass")
        != expected.get("all_required_gates_pass")
        or prior.get("decision", {}).get("next_action")
        != expected.get("next_action")
    ):
        raise ValueError("sealed prior A3 evidence differs from A4 protocol")
    prior_folds = prior.get("folds")
    if isinstance(prior_folds, (str, bytes)) or not isinstance(
        prior_folds,
        Sequence,
    ) or len(prior_folds) != _FAMILIES:
        raise ValueError("prior A3 evidence does not contain eight folds")
    kl_by_family: dict[str, float] = {}
    for raw in prior_folds:
        if not isinstance(raw, Mapping):
            raise TypeError("prior A3 fold is invalid")
        alias = raw.get("held_family_alias")
        evaluation = raw.get("evaluation")
        if not isinstance(alias, str) or not isinstance(evaluation, Mapping):
            raise TypeError("prior A3 fold identity/evaluation is unavailable")
        conditions = evaluation.get("conditions")
        if not isinstance(conditions, Mapping):
            raise TypeError("prior A3 fold condition panel is unavailable")
        row = conditions.get("trajectory_corrected_composition")
        if not isinstance(row, Mapping):
            raise TypeError("prior A3 corrected composition metric is unavailable")
        kl_by_family[alias] = float(
            row["native_to_candidate_kl_per_token"]
        )
    if len(kl_by_family) != _FAMILIES:
        raise ValueError("prior A3 held-family aliases are not unique")
    aggregate = prior.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise TypeError("prior A3 aggregate is unavailable")
    macro = aggregate.get("equal_family_macro")
    if not isinstance(macro, Mapping):
        raise TypeError("prior A3 macro aggregate is unavailable")
    conditions = macro.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("prior A3 macro conditions are unavailable")
    corrected = conditions.get("trajectory_corrected_composition")
    if not isinstance(corrected, Mapping):
        raise TypeError("prior A3 corrected macro is unavailable")
    comparison = {
        "report_file": prior_path.name,
        "report_file_sha256": expected["report_file_sha256"],
        "report_sha256": expected["report_sha256"],
        "protocol_sha256": expected["protocol_sha256"],
        "eligible_condition": "trajectory_corrected_composition",
        "equal_family_macro": dict(corrected),
        "native_to_candidate_kl_per_token_by_family_alias": kl_by_family,
        "all_required_gates_pass": False,
        "source_safe": True,
    }
    if not _metric_close(
        corrected["native_to_candidate_kl_per_token"],
        expected["family_macro_native_to_candidate_kl_per_token"],
    ):
        raise ValueError("prior A3 macro KL differs from frozen evidence")
    return prior, comparison


def _interaction_factorial(
    fold_evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    folds = tuple(
        _validate_full_block_evaluation(value, label=f"factorial fold {index}")
        for index, value in enumerate(fold_evaluations)
    )
    if len(folds) != _FAMILIES:
        raise ValueError("interaction factorial requires exactly eight folds")
    metric_specs = {
        "delta_nll_per_token": (
            "delta_nll_per_token",
            lambda value: value,
        ),
        "native_to_candidate_kl_per_token": (
            "native_to_candidate_kl_per_token",
            lambda value: value,
        ),
        "top1_error_to_native": (
            "top1_agreement_to_native",
            lambda value: 1.0 - value,
        ),
    }
    by_metric: dict[str, dict[str, object]] = {}
    for output_name, (source_name, transform) in metric_specs.items():
        family_rows = []
        for index, fold in enumerate(folds):
            conditions = fold["conditions"]
            assert isinstance(conditions, Mapping)

            def cost(condition: str) -> float:
                row = conditions[condition]
                assert isinstance(row, Mapping)
                return float(transform(float(row[source_name])))

            frozen_off = cost("l10_edgeless_frozen_l17_composition")
            frozen_on = cost("frozen_uncorrected_composition")
            a4_off = cost("l10_edgeless_a4_composition")
            a4_on = cost("a4_full_block_corrected_composition")
            frozen_effect = frozen_on - frozen_off
            a4_effect = a4_on - a4_off
            difference_in_differences = a4_effect - frozen_effect
            family_rows.append(
                {
                    "family_index": index,
                    "frozen_edges_off": frozen_off,
                    "frozen_edges_on": frozen_on,
                    "a4_edges_off": a4_off,
                    "a4_edges_on": a4_on,
                    "frozen_edge_effect": frozen_effect,
                    "a4_edge_effect": a4_effect,
                    "difference_in_differences": difference_in_differences,
                }
            )
        macro = {
            key: math.fsum(float(row[key]) for row in family_rows) / _FAMILIES
            for key in (
                "frozen_edges_off",
                "frozen_edges_on",
                "a4_edges_off",
                "a4_edges_on",
                "frozen_edge_effect",
                "a4_edge_effect",
                "difference_in_differences",
            )
        }
        a4_vs_frozen_gap = macro["a4_edges_on"] - macro["frozen_edges_on"]
        fraction = (
            None
            if a4_vs_frozen_gap <= 0.0
            else macro["difference_in_differences"] / a4_vs_frozen_gap
        )
        by_metric[output_name] = {
            "by_family": family_rows,
            "equal_family_macro": macro,
            "a4_vs_frozen_edges_on_gap": a4_vs_frozen_gap,
            "difference_in_differences_fraction_of_gap": fraction,
            "positive_difference_in_differences_family_count": sum(
                float(row["difference_in_differences"]) > 0.0
                for row in family_rows
            ),
            "a4_edges_harmful_family_count": sum(
                float(row["a4_edge_effect"]) > 0.0 for row in family_rows
            ),
        }
    nll = by_metric["delta_nll_per_token"]
    nll_macro = nll["equal_family_macro"]
    assert isinstance(nll_macro, Mapping)
    nll_did = float(nll_macro["difference_in_differences"])
    fraction = nll["difference_in_differences_fraction_of_gap"]
    fraction_value = None if fraction is None else float(fraction)
    positive_count = int(nll["positive_difference_in_differences_family_count"])
    harmful_count = int(nll["a4_edges_harmful_family_count"])
    corroborating_count = max(
        int(
            by_metric[name][
                "positive_difference_in_differences_family_count"
            ]
        )
        for name in (
            "native_to_candidate_kl_per_token",
            "top1_error_to_native",
        )
    )
    primary = (
        nll_did >= 0.01
        and fraction_value is not None
        and fraction_value >= 0.50
        and positive_count >= 6
        and harmful_count >= 6
        and corroborating_count >= 5
    )
    material = (
        nll_did >= 0.01
        and fraction_value is not None
        and fraction_value >= 0.20
        and positive_count >= 5
        and harmful_count >= 5
        and corroborating_count >= 5
    )
    if primary:
        classification = "primary_explanation"
    elif material:
        classification = "material_partial_explanation"
    elif nll_did <= 0.0 or (
        fraction_value is not None and fraction_value <= 0.10
    ):
        classification = "not_explanatory"
    else:
        classification = "mixed_or_inconclusive"
    return {
        "design": "two_by_two_layer10_edges_by_layer17_generator_recipe",
        "difference_in_differences_formula": (
            "(a4_edges_on-a4_edges_off)-"
            "(frozen_edges_on-frozen_edges_off)"
        ),
        "metrics": by_metric,
        "classification": classification,
        "diagnostic_only": True,
        "selected_or_mutated_topology": False,
    }


def _a4_randomness_receipt(device: torch.device) -> dict[str, object]:
    inherited = _randomness_recipe_receipt(device)
    payload = {
        "recipe_id": "gemma3_l10_l17_a4_full_block_lofo_deterministic_v1",
        "inherits_seed_and_execution_recipe_from": inherited,
        "inheritance_reason": (
            "preserve_bitwise_shared_controls_against_sealed_a3"
        ),
        "target_or_boundary_selection_is_stochastic": False,
    }
    return {
        **payload,
        "recipe_sha256": _domain_sha256(_RANDOMNESS_DOMAIN, payload),
    }


def _a4_split_receipt(
    *,
    row_receipt: Mapping[str, object],
    protocol_fold_sha256: str,
    protocol_sha256: str,
) -> dict[str, object]:
    payload = {
        "protocol_sha256": _require_sha256(
            protocol_sha256,
            label="A4 split protocol",
        ),
        "protocol_fold_sha256": _require_sha256(
            protocol_fold_sha256,
            label="A4 split fold",
        ),
        "capture_sha256": _require_sha256(
            row_receipt.get("capture_sha256"),
            label="A4 split capture",
        ),
        "fit_split_sha256": _require_sha256(
            row_receipt.get("fit_split_sha256"),
            label="A4 fit split",
        ),
        "held_split_sha256": _require_sha256(
            row_receipt.get("held_split_sha256"),
            label="A4 held split",
        ),
        "underlying_splitter_implementation": (
            "sealed_a3_equal_family_fisher_splitter"
        ),
        "underlying_hash_domain_is_inherited": True,
        "target_tensor": "a4_full_block_closure_target",
        "held_family_excluded_from_fit": row_receipt.get(
            "held_family_excluded_from_projection_fit_and_generator_fit"
        ),
    }
    return {
        **payload,
        "receipt_sha256": _domain_sha256(_SPLIT_RECEIPT_DOMAIN, payload),
    }


def _a4_composition_receipt(
    *,
    source_runtime_catalog: Mapping[str, object],
    source_layer17_lowerings: Mapping[str, ModalGeneratorLowering],
    corrected_layer17_graph: ModalGeneratorGraphPlan,
    corrected_layer17_lowerings: Mapping[str, ModalGeneratorLowering],
    corrected_primary_graph: ModalGeneratorGraphPlan,
    corrected_edgeless_graph: ModalGeneratorGraphPlan,
) -> dict[str, object]:
    corrected_layer17_graph.validate_integrity()
    corrected_primary_graph.validate_integrity()
    corrected_edgeless_graph.validate_integrity()
    names = corrected_layer17_graph.traversal_order
    if (
        set(source_layer17_lowerings) != set(names)
        or set(corrected_layer17_lowerings) != set(names)
        or any(
            node.output_boundary != _POST_DELTA_BOUNDARY
            for node in corrected_layer17_graph.nodes
        )
        or corrected_layer17_graph.interactions
        or len(corrected_primary_graph.interactions) != 3
        or corrected_edgeless_graph.interactions
        or corrected_primary_graph.parameter_count
        != _PRIMARY_RESOURCES["primary_graph_parameters"]
        or corrected_edgeless_graph.parameter_count != 290_710
    ):
        raise ValueError("A4 corrected composition topology/boundary drifted")
    binding_rows = []
    for name in names:
        source = source_layer17_lowerings[name]
        corrected = corrected_layer17_lowerings[name]
        source_basis = source.computational_mode_basis
        corrected_basis = corrected.computational_mode_basis
        if (
            source_basis.mean_bias_sha256 != corrected_basis.mean_bias_sha256
            or source_basis.encoder_basis_sha256
            != corrected_basis.encoder_basis_sha256
            or source_basis.decoder_basis_sha256
            != corrected_basis.decoder_basis_sha256
            or source_basis.binding.output_site
            != "layer.17.mlp.operator_output"
            or corrected_basis.binding.output_site != _POST_DELTA_BOUNDARY
            or corrected.coordinate_generator_plan.binding.output_site
            != _POST_DELTA_BOUNDARY
        ):
            raise ValueError("A4 relocation changed decoder tensors or boundary")
        binding_rows.append(
            {
                "node_name": name,
                "source_basis_sha256": source_basis.artifact_sha256,
                "relocated_basis_sha256": corrected_basis.artifact_sha256,
                "mean_bias_sha256": source_basis.mean_bias_sha256,
                "encoder_basis_sha256": source_basis.encoder_basis_sha256,
                "decoder_basis_sha256": source_basis.decoder_basis_sha256,
                "relocated_generator_plan_sha256": (
                    corrected.coordinate_generator_plan.artifact_sha256
                ),
                "relocated_lowering_sha256": corrected.artifact_sha256,
                "source_output_boundary": "layer.17.mlp.operator_output",
                "runtime_output_boundary": _POST_DELTA_BOUNDARY,
            }
        )
    payload = {
        "source_runtime_catalog_sha256": _require_sha256(
            source_runtime_catalog.get("catalog_sha256"),
            label="source runtime catalog",
        ),
        "source_primary_graph_sha256": _require_sha256(
            source_runtime_catalog.get("frozen_primary_graph_sha256"),
            label="source primary graph",
        ),
        "corrected_layer17_graph_sha256": corrected_layer17_graph.artifact_sha256,
        "corrected_primary_graph_sha256": corrected_primary_graph.artifact_sha256,
        "corrected_edgeless_graph_sha256": corrected_edgeless_graph.artifact_sha256,
        "ordered_relocated_bindings": binding_rows,
        "application_policy": {
            "layer10_generated_output_boundary": (
                "layer.10.mlp.operator_output"
            ),
            "layer17_compact_path": (
                "compact_raw_mlp_then_live_post_feedforward_rmsnorm"
            ),
            "layer17_generated_output_boundary": _POST_DELTA_BOUNDARY,
            "generated_correction_is_post_feedforward_rmsnorm": True,
            "layer10_interactions_preserved_in_primary": True,
            "layer10_interactions_removed_only_in_diagnostic": True,
        },
        "exact_primary_resources": dict(_PRIMARY_RESOURCES),
        "exact_edgeless_resources": {
            **_AUX_RESOURCE_EXPECTATIONS[
                "l10_edgeless_a4_composition"
            ],
            "candidate_whole_model_learned_parameters": 267_306_006,
        },
    }
    return {
        **payload,
        "receipt_sha256": _domain_sha256(_COMPOSITION_DOMAIN, payload),
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
        raise ValueError("unsupported A4 gate operator")
    return {
        "gate_id": gate_id,
        "required": True,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }


def evaluate_full_block_closure_lofo_gates(
    *,
    protocol: Mapping[str, object],
    fold_aliases: Sequence[str],
    fold_evaluations: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    prior_a3_comparison: Mapping[str, object],
    exact_projection_metadata_match: bool,
    capture_audit: Mapping[str, object],
    fold_bundle_receipt: Mapping[str, object],
    source_model_unchanged: bool,
    layer10_unchanged: bool,
) -> dict[str, object]:
    frozen = validate_gemma3_l10_l17_full_block_closure_protocol(protocol)
    evaluations = tuple(
        _validate_full_block_evaluation(value, label=f"gate fold {index}")
        for index, value in enumerate(fold_evaluations)
    )
    recomputed = aggregate_full_block_closure_folds(evaluations)
    if _canonical_json_bytes(aggregate) != _canonical_json_bytes(recomputed):
        raise ValueError("A4 aggregate differs from fold evaluations")
    aliases = tuple(fold_aliases)
    prior_kl = prior_a3_comparison.get(
        "native_to_candidate_kl_per_token_by_family_alias"
    )
    if (
        len(aliases) != _FAMILIES
        or len(set(aliases)) != _FAMILIES
        or not isinstance(prior_kl, Mapping)
        or set(prior_kl) != set(aliases)
    ):
        raise ValueError("A4/prior family comparison catalog is invalid")
    a3_protocol = build_default_gemma3_l10_l17_trajectory_correction_protocol()
    a3_evaluations = tuple(_as_a3_gate_evaluation(value) for value in evaluations)
    a3_aggregate = aggregate_trajectory_correction_lofo_folds(a3_evaluations)
    base = evaluate_trajectory_correction_lofo_gates(
        protocol=a3_protocol,
        fold_evaluations=a3_evaluations,
        aggregate=a3_aggregate,
        exact_resources_match=True,
        exact_projection_metadata_match=exact_projection_metadata_match,
        compact_replay_algebraic_equivalence_audit=bool(
            capture_audit.get("all_required_capture_audits_pass")
        ),
        source_model_unchanged=source_model_unchanged,
        layer10_unchanged=layer10_unchanged,
    )
    macro = recomputed["equal_family_macro"]
    assert isinstance(macro, Mapping)
    macro_conditions = macro["conditions"]
    assert isinstance(macro_conditions, Mapping)
    current = macro_conditions["a4_full_block_corrected_composition"]
    prior_macro = prior_a3_comparison.get("equal_family_macro")
    if not isinstance(current, Mapping) or not isinstance(prior_macro, Mapping):
        raise TypeError("A4/prior macro comparison is unavailable")
    current_kl = float(current["native_to_candidate_kl_per_token"])
    prior_macro_kl = float(
        prior_macro["native_to_candidate_kl_per_token"]
    )
    improvement_by_alias: dict[str, bool] = {}
    for alias, evaluation in zip(aliases, evaluations, strict=True):
        conditions = evaluation["conditions"]
        assert isinstance(conditions, Mapping)
        row = conditions["a4_full_block_corrected_composition"]
        assert isinstance(row, Mapping)
        improvement_by_alias[alias] = float(
            row["native_to_candidate_kl_per_token"]
        ) < float(prior_kl[alias])
    improvement_count = sum(improvement_by_alias.values())
    gates = frozen.get("gates")
    if not isinstance(gates, Mapping):
        raise TypeError("A4 gate contract is unavailable")
    additional = (
        _gate(
            "strict_family_macro_kl_improvement_vs_prior_a3",
            observed=current_kl,
            operator="<",
            threshold=prior_macro_kl,
        ),
        _gate(
            "held_family_kl_improvement_count_vs_prior_a3",
            observed=improvement_count,
            operator=">=",
            threshold=int(
                gates[
                    "minimum_held_family_kl_improvement_count_vs_prior_a3"
                ]
            ),
        ),
        _gate(
            "post_feedforward_application_boundary",
            observed=all(
                value.get("application_boundary") == _POST_DELTA_BOUNDARY
                for value in evaluations
            ),
            operator="==",
            threshold=bool(
                gates["require_post_feedforward_application_boundary"]
            ),
        ),
        _gate(
            "full_block_capture_audits",
            observed=bool(
                capture_audit.get("all_required_capture_audits_pass")
            ),
            operator="==",
            threshold=bool(gates["require_full_block_capture_audits"]),
        ),
        _gate(
            "fold_executable_bundle",
            observed=(
                fold_bundle_receipt.get("fold_count") == _FAMILIES
                and fold_bundle_receipt.get("validated") is True
                and fold_bundle_receipt.get("source_safe") is True
            ),
            operator="==",
            threshold=bool(gates["require_fold_executable_bundle"]),
        ),
    )
    base_rows = base.get("gate_table")
    if isinstance(base_rows, (str, bytes)) or not isinstance(
        base_rows,
        Sequence,
    ):
        raise TypeError("A3-compatible base gate table is unavailable")
    gate_rows = [dict(value) for value in base_rows if isinstance(value, Mapping)]
    if len(gate_rows) != len(base_rows):
        raise TypeError("A3-compatible base gate table is invalid")
    gate_rows.extend(additional)
    passed = all(bool(row["passed"]) for row in gate_rows)
    return {
        "protocol_sha256": frozen["artifact_sha256"],
        "decision_policy": gates["decision_policy"],
        "a3_compatible_gate_metrics": base["derived_metrics"],
        "prior_a3_comparison_metrics": {
            "family_macro_a4_kl": current_kl,
            "family_macro_prior_a3_kl": prior_macro_kl,
            "family_macro_kl_improvement": prior_macro_kl - current_kl,
            "held_family_kl_improvement_by_alias": improvement_by_alias,
            "held_family_kl_improvement_count": improvement_count,
        },
        "gate_table": gate_rows,
        "all_required_gates_pass": passed,
        "next_action": (
            "fit_all_eight_families_then_replay_exactly_one_open_a_selection"
            if passed
            else "stop_keep_other_roles_closed_and_revise_a_fit_recipe"
        ),
    }


def _fold_bundle_receipt(
    *,
    bundle: Mapping[str, object],
    companion: Mapping[str, object],
    bundle_path: Path,
) -> dict[str, object]:
    folds = bundle.get("folds")
    if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence):
        raise TypeError("A4 executable fold bundle catalog is unavailable")
    bindings = []
    for raw in folds:
        if not isinstance(raw, Mapping):
            raise TypeError("A4 executable fold bundle record is invalid")
        lowerings = raw.get("lowering_records")
        if isinstance(lowerings, (str, bytes)) or not isinstance(
            lowerings,
            Sequence,
        ):
            raise TypeError("A4 executable fold lowerings are unavailable")
        lowering_sha256_by_node = {}
        for record in lowerings:
            if not isinstance(record, Mapping):
                raise TypeError("A4 executable lowering record is invalid")
            lowering_sha256_by_node[str(record["node_name"])] = _require_sha256(
                record.get("lowering_sha256"),
                label="A4 executable lowering",
            )
        bindings.append(
            {
                "fold_index": raw["fold_index"],
                "fold_id": raw["fold_id"],
                "held_family_alias": raw["held_family_alias"],
                "protocol_fold_sha256": raw["protocol_fold_sha256"],
                "graph_sha256": raw["graph_sha256"],
                "fit_split_sha256": raw["fit_split_sha256"],
                "held_split_sha256": raw["held_split_sha256"],
                "lowering_sha256_by_node": lowering_sha256_by_node,
                "application_boundary": raw["application_boundary"],
            }
        )
    companion_path = bundle_path.with_suffix(".json")
    receipt = {
        "tensor_file": bundle_path.name,
        "tensor_file_sha256": _require_sha256(
            companion.get("tensor_file_sha256"),
            label="A4 executable fold bundle file",
        ),
        "companion_report_file": companion_path.name,
        "companion_report_sha256": _require_sha256(
            companion.get("report_sha256"),
            label="A4 executable fold bundle report",
        ),
        "scientific_payload_sha256": _require_sha256(
            companion.get("scientific_payload_sha256"),
            label="A4 executable fold bundle scientific payload",
        ),
        "protocol_sha256": companion.get("protocol_sha256"),
        "fold_count": companion.get("fold_count"),
        "fold_bindings": bindings,
        "contains_executable_generator_weights": companion.get(
            "contains_executable_generator_weights"
        ),
        "contains_prompt_or_activation_rows": companion.get(
            "contains_prompt_or_activation_rows"
        ),
        "source_safe": companion.get("source_safe"),
        "validated": True,
    }
    if (
        companion.get("tensor_file") != bundle_path.name
        or receipt["protocol_sha256"]
        != FROZEN_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_PROTOCOL_SHA256
        or receipt["fold_count"] != _FAMILIES
        or len(bindings) != _FAMILIES
        or receipt["contains_executable_generator_weights"] is not True
        or receipt["contains_prompt_or_activation_rows"] is not False
        or receipt["source_safe"] is not True
        or _file_sha256(bundle_path) != receipt["tensor_file_sha256"]
    ):
        raise ValueError("A4 executable fold bundle receipt drifted")
    return receipt


def _validate_fold_report(
    value: object,
    *,
    index: int,
    protocol: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "fold_index",
        "fold_id",
        "held_family_alias",
        "training_family_aliases",
        "protocol_fold_sha256",
        "row_receipt",
        "a4_split_receipt",
        "correction_fit",
        "corrected_layer17_graph_sha256",
        "corrected_primary_graph_sha256",
        "corrected_edgeless_graph_sha256",
        "corrected_lowering_sha256_by_node",
        "relocated_basis_sha256_by_node",
        "composition_receipt",
        "shared_control_receipt",
        "evaluation",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"A4 fold {index} report fields are invalid")
    protocol_folds = protocol.get("folds")
    if isinstance(protocol_folds, (str, bytes)) or not isinstance(
        protocol_folds,
        Sequence,
    ):
        raise TypeError("A4 protocol fold catalog is unavailable")
    expected = protocol_folds[index]
    if not isinstance(expected, Mapping):
        raise TypeError("A4 protocol fold is invalid")
    if (
        value.get("fold_index") != expected.get("fold_index")
        or value.get("fold_id") != expected.get("fold_id")
        or value.get("held_family_alias")
        != expected.get("held_family_alias")
        or value.get("training_family_aliases")
        != expected.get("training_family_aliases")
        or value.get("protocol_fold_sha256")
        != expected.get("artifact_sha256")
    ):
        raise ValueError(f"A4 fold {index} differs from frozen split")
    row_receipt = value.get("row_receipt")
    split_receipt = value.get("a4_split_receipt")
    fit = value.get("correction_fit")
    lowerings = value.get("corrected_lowering_sha256_by_node")
    relocated_bases = value.get("relocated_basis_sha256_by_node")
    if not all(
        isinstance(item, Mapping)
        for item in (row_receipt, split_receipt, fit, lowerings, relocated_bases)
    ):
        raise TypeError(f"A4 fold {index} fit lineage is incomplete")
    assert isinstance(row_receipt, Mapping)
    assert isinstance(split_receipt, Mapping)
    assert isinstance(fit, Mapping)
    assert isinstance(lowerings, Mapping)
    assert isinstance(relocated_bases, Mapping)
    expected_split = _a4_split_receipt(
        row_receipt=row_receipt,
        protocol_fold_sha256=str(expected["artifact_sha256"]),
        protocol_sha256=str(protocol["artifact_sha256"]),
    )
    node_order = tuple(fit.get("node_order", ()))
    if (
        _canonical_json_bytes(split_receipt)
        != _canonical_json_bytes(expected_split)
        or not _projection_receipt_matches_protocol(row_receipt, protocol)
        or row_receipt.get(
            "held_family_excluded_from_projection_fit_and_generator_fit"
        )
        is not True
        or row_receipt.get("held_family_alias")
        != expected.get("held_family_alias")
        or tuple(row_receipt.get("fit_family_aliases", ()))
        != tuple(expected.get("training_family_aliases", ()))
        or row_receipt.get("fit_sequences") != 224
        or row_receipt.get("held_sequences") != 32
        or fit.get("graph_sha256")
        != value.get("corrected_layer17_graph_sha256")
        or fit.get("parameter_count") != 163_094
        or fit.get("macs_per_token") != 160_352
        or fit.get("interaction_count") != 0
        or len(node_order) != 4
        or set(lowerings) != set(node_order)
        or fit.get("lowering_sha256_by_node") != lowerings
        or set(relocated_bases) != set(node_order)
        or any(
            _SHA256.fullmatch(str(digest)) is None
            for digest in (*lowerings.values(), *relocated_bases.values())
        )
    ):
        raise ValueError(f"A4 fold {index} projection/refit lineage drifted")
    for field in (
        "corrected_layer17_graph_sha256",
        "corrected_primary_graph_sha256",
        "corrected_edgeless_graph_sha256",
    ):
        _require_sha256(value.get(field), label=f"A4 fold {index} {field}")
    composition = value.get("composition_receipt")
    if not isinstance(composition, Mapping):
        raise TypeError(f"A4 fold {index} composition receipt is unavailable")
    composition_payload = {
        key: child
        for key, child in composition.items()
        if key != "receipt_sha256"
    }
    if (
        composition.get("receipt_sha256")
        != _domain_sha256(_COMPOSITION_DOMAIN, composition_payload)
        or composition.get("corrected_layer17_graph_sha256")
        != value.get("corrected_layer17_graph_sha256")
        or composition.get("corrected_primary_graph_sha256")
        != value.get("corrected_primary_graph_sha256")
        or composition.get("corrected_edgeless_graph_sha256")
        != value.get("corrected_edgeless_graph_sha256")
        or composition.get("application_policy", {}).get(
            "layer17_generated_output_boundary"
        )
        != _POST_DELTA_BOUNDARY
    ):
        raise ValueError(f"A4 fold {index} composition receipt drifted")
    shared = value.get("shared_control_receipt")
    if (
        not isinstance(shared, Mapping)
        or shared.get("matched") is not True
        or shared.get("token_counts_match") is not True
        or float(shared.get("maximum_absolute_difference", float("inf")))
        > float(
            shared.get("maximum_allowed_absolute_difference", float("-inf"))
        )
    ):
        raise ValueError(f"A4 fold {index} shared controls drifted")
    _validate_full_block_evaluation(
        value.get("evaluation"),
        label=f"A4 fold {index}",
    )
    return value


def _validate_fold_bundle_receipt(
    value: object,
    *,
    folds: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> Mapping[str, object]:
    fields = {
        "tensor_file",
        "tensor_file_sha256",
        "companion_report_file",
        "companion_report_sha256",
        "scientific_payload_sha256",
        "protocol_sha256",
        "fold_count",
        "fold_bindings",
        "contains_executable_generator_weights",
        "contains_prompt_or_activation_rows",
        "source_safe",
        "validated",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("A4 fold-bundle receipt fields are invalid")
    bindings = value.get("fold_bindings")
    if (
        value.get("protocol_sha256") != protocol.get("artifact_sha256")
        or value.get("fold_count") != _FAMILIES
        or value.get("contains_executable_generator_weights") is not True
        or value.get("contains_prompt_or_activation_rows") is not False
        or value.get("source_safe") is not True
        or value.get("validated") is not True
        or isinstance(bindings, (str, bytes))
        or not isinstance(bindings, Sequence)
        or len(bindings) != _FAMILIES
    ):
        raise ValueError("A4 fold-bundle safety/boundary drifted")
    for field in (
        "tensor_file_sha256",
        "companion_report_sha256",
        "scientific_payload_sha256",
    ):
        _require_sha256(value.get(field), label=f"A4 fold bundle {field}")
    protocol_folds = protocol.get("folds")
    if isinstance(protocol_folds, (str, bytes)) or not isinstance(
        protocol_folds,
        Sequence,
    ):
        raise TypeError("A4 protocol folds are unavailable")
    for index, (binding, fold, protocol_fold) in enumerate(
        zip(bindings, folds, protocol_folds, strict=True)
    ):
        if not all(
            isinstance(item, Mapping)
            for item in (binding, fold, protocol_fold)
        ):
            raise TypeError("A4 fold-bundle cross-binding is invalid")
        assert isinstance(binding, Mapping)
        assert isinstance(fold, Mapping)
        assert isinstance(protocol_fold, Mapping)
        row = fold["row_receipt"]
        lowerings = fold["corrected_lowering_sha256_by_node"]
        if not isinstance(row, Mapping) or not isinstance(lowerings, Mapping):
            raise TypeError("A4 fold report split/lowerings are unavailable")
        if (
            binding.get("fold_index") != index
            or binding.get("fold_index") != fold.get("fold_index")
            or binding.get("fold_id") != fold.get("fold_id")
            or binding.get("fold_id") != protocol_fold.get("fold_id")
            or binding.get("held_family_alias")
            != fold.get("held_family_alias")
            or binding.get("held_family_alias")
            != protocol_fold.get("held_family_alias")
            or binding.get("protocol_fold_sha256")
            != fold.get("protocol_fold_sha256")
            or binding.get("protocol_fold_sha256")
            != protocol_fold.get("artifact_sha256")
            or binding.get("graph_sha256")
            != fold.get("corrected_layer17_graph_sha256")
            or binding.get("fit_split_sha256")
            != row.get("fit_split_sha256")
            or binding.get("held_split_sha256")
            != row.get("held_split_sha256")
            or binding.get("lowering_sha256_by_node") != lowerings
            or binding.get("application_boundary") != _POST_DELTA_BOUNDARY
        ):
            raise ValueError(f"A4 executable fold {index} is cross-bound wrong")
    return value


def build_full_block_closure_lofo_report(
    *,
    protocol: Mapping[str, object],
    authorization: Mapping[str, object],
    prior_a3_comparison: Mapping[str, object],
    runtime: Mapping[str, object],
    fit_collection: Mapping[str, object],
    fold_executable_bundle: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
    interaction_factorial: Mapping[str, object],
    decision: Mapping[str, object],
    source_model_unchanged: bool,
    layer10_unchanged: bool,
) -> dict[str, object]:
    frozen = validate_gemma3_l10_l17_full_block_closure_protocol(protocol)
    payload: dict[str, object] = {
        "schema": GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_SCHEMA,
        "format_version": GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_FORMAT_VERSION,
        "scientific_role": (
            "calibration_a_fit_adaptive_full_block_closure_development"
        ),
        "heldout_confirmation": False,
        "protocol": frozen,
        "authorization": dict(authorization),
        "prior_a3_comparison": dict(prior_a3_comparison),
        "runtime": dict(runtime),
        "fit_collection": dict(fit_collection),
        "fold_executable_bundle": dict(fold_executable_bundle),
        "folds": [dict(value) for value in folds],
        "aggregate": dict(aggregate),
        "resources": dict(_PRIMARY_RESOURCES),
        "interaction_factorial": dict(interaction_factorial),
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
        "safety": dict(_SAFETY),
    }
    _reject_forbidden_output_fields(payload)
    report = {
        **payload,
        "report_sha256": _domain_sha256(_REPORT_DOMAIN, payload),
    }
    return validate_gemma3_l10_l17_full_block_closure_lofo_report(report)


def validate_gemma3_l10_l17_full_block_closure_lofo_report(
    raw: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on A4 lineage, safety, metric, or bundle tampering."""

    if not isinstance(raw, Mapping) or set(raw) != _REPORT_FIELDS:
        raise ValueError("A4 full-block LOFO report fields are invalid")
    _reject_forbidden_output_fields(raw)
    if (
        raw.get("schema") != GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_SCHEMA
        or raw.get("format_version")
        != GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_FORMAT_VERSION
        or raw.get("scientific_role")
        != "calibration_a_fit_adaptive_full_block_closure_development"
        or raw.get("heldout_confirmation") is not False
        or raw.get("source_model_unchanged") is not True
        or raw.get("layer10_unchanged") is not True
        or raw.get("full_model_logits_scored") is not True
        or any(
            raw.get(field) is not False
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
        or raw.get("safety") != _SAFETY
    ):
        raise ValueError("A4 full-block LOFO claim boundary is invalid")
    protocol_raw = raw.get("protocol")
    if not isinstance(protocol_raw, Mapping):
        raise TypeError("A4 full-block protocol is unavailable")
    protocol = validate_gemma3_l10_l17_full_block_closure_protocol(protocol_raw)
    authorization = raw.get("authorization")
    runtime = raw.get("runtime")
    if not isinstance(authorization, Mapping) or not isinstance(runtime, Mapping):
        raise TypeError("A4 authorization/runtime is unavailable")
    bundle_binding = authorization.get("bundle")
    source_runtime_catalog = authorization.get("source_runtime_catalog")
    if not isinstance(bundle_binding, Mapping):
        raise TypeError("A4 authenticated bundle binding is unavailable")
    _validate_source_runtime_catalog(
        source_runtime_catalog,
        protocol=protocol,
        bundle_binding=bundle_binding,
    )
    if (
        authorization.get("authorization_kind")
        != "frozen_composition_then_a4_fit_only_family_lofo"
        or authorization.get("authorization_completed_before_fit_open")
        is not True
        or authorization.get("protocol_sha256")
        != protocol.get("artifact_sha256")
        or authorization.get("authenticated_source_protocol_sha256")
        != protocol.get("source_authority", {})
        .get("prior_a3_evidence", {})
        .get("protocol_sha256")
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
        or runtime.get("model_id")
        != protocol.get("source_authority", {}).get("model", {}).get("model_id")
        or runtime.get("requested_revision")
        != protocol.get("source_authority", {})
        .get("model", {})
        .get("requested_revision")
        or runtime.get("local_files_only") is not True
    ):
        raise ValueError("A4 authorization/runtime boundary drifted")
    prior = raw.get("prior_a3_comparison")
    expected_prior = protocol.get("source_authority", {}).get(
        "prior_a3_evidence"
    )
    if not isinstance(prior, Mapping) or not isinstance(expected_prior, Mapping):
        raise TypeError("A4 prior-A3 comparison is unavailable")
    prior_kl = prior.get("native_to_candidate_kl_per_token_by_family_alias")
    prior_macro = prior.get("equal_family_macro")
    if (
        prior.get("report_file") != expected_prior.get("report_file")
        or prior.get("report_file_sha256")
        != expected_prior.get("report_file_sha256")
        or prior.get("report_sha256") != expected_prior.get("report_sha256")
        or prior.get("protocol_sha256")
        != expected_prior.get("protocol_sha256")
        or prior.get("source_safe") is not True
        or not isinstance(prior_kl, Mapping)
        or len(prior_kl) != _FAMILIES
        or not isinstance(prior_macro, Mapping)
        or not _metric_close(
            prior_macro.get("native_to_candidate_kl_per_token"),
            expected_prior.get(
                "family_macro_native_to_candidate_kl_per_token"
            ),
        )
    ):
        raise ValueError("A4 prior-A3 evidence binding drifted")
    fit_collection = raw.get("fit_collection")
    if not isinstance(fit_collection, Mapping):
        raise TypeError("A4 fit collection is unavailable")
    materialization = fit_collection.get("materialization")
    capture = fit_collection.get("capture")
    capture_audit = fit_collection.get("capture_audit")
    if not isinstance(materialization, Mapping) or not isinstance(
        capture,
        Mapping,
    ) or not isinstance(capture_audit, Mapping):
        raise TypeError("A4 materialization/capture is unavailable")
    validate_gemma3_layer17_family_lofo_materialization_metadata(
        materialization
    )
    recomputed_capture_audit = _full_block_capture_audit_receipt(
        capture,
        protocol,
    )
    if (
        _canonical_json_bytes(capture_audit)
        != _canonical_json_bytes(recomputed_capture_audit)
        or capture_audit.get("all_required_capture_audits_pass") is not True
        or fit_collection.get("capture_count") != 1
        or fit_collection.get("captured_examples") != 256
        or fit_collection.get("captured_sequences") != 256
        or fit_collection.get("model_rows_recollected_per_fold") is not False
        or fit_collection.get("a4_target_construction")
        != protocol.get("target_contract", {}).get("raw_target_formula")
    ):
        raise ValueError("A4 fit/capture provenance drifted")
    folds_raw = raw.get("folds")
    if isinstance(folds_raw, (str, bytes)) or not isinstance(
        folds_raw,
        Sequence,
    ) or len(folds_raw) != _FAMILIES:
        raise ValueError("A4 full-block LOFO requires exactly eight folds")
    folds = tuple(
        _validate_fold_report(value, index=index, protocol=protocol)
        for index, value in enumerate(folds_raw)
    )
    aliases = tuple(str(value["held_family_alias"]) for value in folds)
    if len(set(aliases)) != _FAMILIES or set(aliases) != set(prior_kl):
        raise ValueError("A4 held-family alias coverage drifted")
    bundle_receipt = _validate_fold_bundle_receipt(
        raw.get("fold_executable_bundle"),
        folds=folds,
        protocol=protocol,
    )
    evaluations = tuple(value["evaluation"] for value in folds)
    aggregate = aggregate_full_block_closure_folds(evaluations)  # type: ignore[arg-type]
    if _canonical_json_bytes(raw.get("aggregate")) != _canonical_json_bytes(
        aggregate
    ):
        raise ValueError("A4 aggregate was not reproduced")
    factorial = _interaction_factorial(evaluations)  # type: ignore[arg-type]
    if _canonical_json_bytes(raw.get("interaction_factorial")) != (
        _canonical_json_bytes(factorial)
    ):
        raise ValueError("A4 interaction factorial was not reproduced")
    exact_projection = all(
        _projection_receipt_matches_protocol(value["row_receipt"], protocol)  # type: ignore[arg-type]
        for value in folds
    )
    decision = evaluate_full_block_closure_lofo_gates(
        protocol=protocol,
        fold_aliases=aliases,
        fold_evaluations=evaluations,  # type: ignore[arg-type]
        aggregate=aggregate,
        prior_a3_comparison=prior,
        exact_projection_metadata_match=exact_projection,
        capture_audit=capture_audit,
        fold_bundle_receipt=bundle_receipt,
        source_model_unchanged=True,
        layer10_unchanged=True,
    )
    if _canonical_json_bytes(raw.get("decision")) != _canonical_json_bytes(
        decision
    ):
        raise ValueError("A4 gate decision was not reproduced")
    if _canonical_json_bytes(raw.get("resources")) != _canonical_json_bytes(
        _PRIMARY_RESOURCES
    ):
        raise ValueError("A4 exact primary resources drifted")
    supplied = _require_sha256(raw.get("report_sha256"), label="A4 report")
    payload = {key: raw[key] for key in _REPORT_FIELDS if key != "report_sha256"}
    if supplied != _domain_sha256(_REPORT_DOMAIN, payload):
        raise ValueError("A4 full-block LOFO report hash mismatch")
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


def save_gemma3_l10_l17_full_block_closure_lofo_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A4 full-block LOFO report")
    validated = validate_gemma3_l10_l17_full_block_closure_lofo_report(report)
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
                "refusing to overwrite A4 full-block LOFO report"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_gemma3_l10_l17_full_block_closure_lofo_report(
    path: Path | str,
) -> dict[str, object]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("A4 full-block LOFO report is not strict JSON") from error
    if not isinstance(raw, dict):
        raise TypeError("A4 full-block LOFO report must contain one object")
    return validate_gemma3_l10_l17_full_block_closure_lofo_report(raw)


def run_gemma3_l10_l17_full_block_closure_lofo(
    *,
    revision: str,
    output: Path | str = DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    fold_bundle_output: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE
    ),
    prior_a3_report_path: Path | str = (
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
    """Run the sealed A-fit-only eight-family A4 full-block LOFO."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    fold_destination = Path(fold_bundle_output)
    fold_companion = fold_destination.with_suffix(".json")
    prior_path = Path(prior_a3_report_path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A4 full-block LOFO report")
    if fold_destination.suffix != ".pt":
        raise ValueError("A4 fold bundle output must use the .pt suffix")
    if fold_destination.exists() or fold_companion.exists():
        raise FileExistsError("refusing to overwrite A4 executable fold bundle")
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)

    _progress("preflight: authenticate A3 source and sealed prior evidence")
    bundle, authority, a3_protocol, a3_authorization = (
        _authenticate_before_fit_access(
            bundle_path=composition_bundle_path,
            corpus_receipt_path=corpus_receipt_path,
            corpus_artifact_path=corpus_artifact_path,
            fit_input_path=fit_input_path,
        )
    )
    protocol = validate_gemma3_l10_l17_full_block_closure_protocol(
        build_default_gemma3_l10_l17_full_block_closure_protocol()
    )
    prior_a3_report, prior_a3_comparison = _prior_a3_comparison(
        protocol=protocol,
        prior_path=prior_path,
    )
    if (
        getattr(bundle, "model_id", None) != model_id
        or getattr(bundle, "requested_revision", None) != revision
    ):
        raise ValueError("requested model identity differs from frozen bundle")
    device = resolve_torch_device(device_name)
    randomness_receipt = _a4_randomness_receipt(device)
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
    edgeless_graph = getattr(bundle, "edgeless", None)
    bundle_lowerings = getattr(bundle, "lowerings", None)
    if (
        not isinstance(bundle_binding, Mapping)
        or not isinstance(primary_graph, ModalGeneratorGraphPlan)
        or not isinstance(edgeless_graph, ModalGeneratorGraphPlan)
        or not isinstance(bundle_lowerings, tuple)
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
    _validate_source_runtime_catalog(
        source_runtime_catalog,
        protocol=protocol,
        bundle_binding=bundle_binding,
    )
    authorization = {
        **a3_authorization,
        "authorization_kind": (
            "frozen_composition_then_a4_fit_only_family_lofo"
        ),
        "authenticated_source_protocol_sha256": a3_protocol[
            "artifact_sha256"
        ],
        "protocol_sha256": protocol["artifact_sha256"],
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
        model_fingerprint != primary_graph.model_fingerprint
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
    _reject_forbidden_output_fields(materialization, path="materialization")
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
    prior_folds = prior_a3_report.get("folds")
    if isinstance(prior_folds, (str, bytes)) or not isinstance(
        prior_folds,
        Sequence,
    ):
        raise TypeError("prior A3 folds are unavailable")
    prior_evaluation_by_alias = {
        str(raw["held_family_alias"]): raw["evaluation"]
        for raw in prior_folds
        if isinstance(raw, Mapping)
    }
    if len(prior_evaluation_by_alias) != _FAMILIES:
        raise ValueError("prior A3 family evaluation catalog is incomplete")

    source_lowering_by_name = {
        node.name: lowering
        for node, lowering in zip(
            primary_graph.nodes,
            bundle_lowerings,
            strict=True,
        )
    }
    layer10_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer10_graph,
        tuple(
            layer10_lowerings[name] for name in layer10_graph.traversal_order
        ),
    )
    source_layer17_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        layer17_graph,
        tuple(
            layer17_lowerings[name] for name in layer17_graph.traversal_order
        ),
    )
    frozen_primary_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        primary_graph,
        _ordered_restored_lowerings(primary_graph, source_lowering_by_name),
    )
    frozen_edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless_graph,
        _ordered_restored_lowerings(edgeless_graph, source_lowering_by_name),
    )

    _progress("rows: capture exact Layer17 full-block closure trajectories")
    capture: Gemma3Layer17FullBlockClosureCapture = (
        capture_gemma3_layer17_full_block_closure(
            adapter,
            all_batches,
            selection=selection,
            leaf_activation_site=leaf_site,
            layer10_executor=layer10_executor,
            layer17_executor=source_layer17_executor,
        )
    )
    expected_sequences = sum(batch.batch_size for batch in all_batches)
    if (
        capture.native_rows.sequences != expected_sequences
        or capture.compiled_rows.sequences != expected_sequences
        or expected_sequences != 256
    ):
        raise RuntimeError("A4 capture sequence accounting drifted")
    capture_metadata = capture.metadata()
    capture_audit = _full_block_capture_audit_receipt(
        capture_metadata,
        protocol,
    )
    if capture_audit["all_required_capture_audits_pass"] is not True:
        raise RuntimeError("A4 full-block capture audits failed")
    fragment_ids = tuple(capture.trajectory_rows.fragment_ids)
    fit_view = _TrajectoryCorrectionFitView(
        compiled_input=_shared_compiled_input(
            capture.compiled_rows,
            fragment_ids,
        ),
        a3_target=capture.a4_full_block_closure_target,
        native_fisher_weights_by_fragment={
            fragment_id: capture.native_rows.rows_by_fragment[
                fragment_id
            ].fisher_weights
            for fragment_id in fragment_ids
        },
        row_keys=capture.native_rows.row_keys,
        row_key_sha256=capture.native_rows.row_key_sha256,
        sequences=capture.native_rows.sequences,
        fragment_ids=fragment_ids,
        capture_sha256=capture.capture_sha256,
    )
    captured_observations = fit_view.observations
    del capture

    fold_reports: list[dict[str, object]] = []
    fold_evaluations: list[Mapping[str, object]] = []
    fold_executables: list[dict[str, object]] = []
    held_batches_by_alias = dict(blocks)
    for index, fold in enumerate(_fold_catalog(protocol)):
        held = str(fold["held_family_alias"])
        training = tuple(fold["training_family_aliases"])  # type: ignore[arg-type]
        _progress(f"fold {index + 1}/{_FAMILIES}: fit seven, hold {held}")
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
        split_receipt = _a4_split_receipt(
            row_receipt=row_receipt,
            protocol_fold_sha256=str(fold["artifact_sha256"]),
            protocol_sha256=str(protocol["artifact_sha256"]),
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
            output_boundary=_POST_DELTA_BOUNDARY,
        )
        if (
            correction.graph_plan.parameter_count != 163_094
            or correction.graph_plan.macs_per_token != 160_352
            or correction.graph_plan.interactions
            or any(
                node.output_boundary != _POST_DELTA_BOUNDARY
                for node in correction.graph_plan.nodes
            )
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
            raise RuntimeError("A4 fold correction escaped fixed capacity")
        corrected_primary = replace_layer_nodes_in_composed_graph(
            primary_graph,
            correction.graph_plan,
            layer_ordinal=17,
        )
        corrected_edgeless = replace_layer_nodes_in_composed_graph(
            edgeless_graph,
            correction.graph_plan,
            layer_ordinal=17,
        )
        if (
            corrected_primary.parameter_count != 295_129
            or corrected_primary.macs_per_token != 289_600
            or len(corrected_primary.interactions) != 3
            or corrected_edgeless.parameter_count != 290_710
            or corrected_edgeless.macs_per_token != 285_280
            or corrected_edgeless.interactions
        ):
            raise RuntimeError("A4 corrected composition resources drifted")
        corrected_primary_lowerings = _merge_corrected_composition_lowerings(
            corrected_primary,
            layer10_lowerings_by_node=layer10_lowerings,
            corrected_layer17_lowerings_by_node=correction.lowerings_by_node,
        )
        corrected_edgeless_lowerings = _merge_corrected_composition_lowerings(
            corrected_edgeless,
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
            corrected_primary_lowerings,
            post_feedforward_delta_layer_ordinals=(17,),
        )
        corrected_edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            corrected_edgeless,
            corrected_edgeless_lowerings,
            post_feedforward_delta_layer_ordinals=(17,),
        )
        _progress(f"fold {index + 1}/{_FAMILIES}: score held full-model logits")
        core = score_trajectory_correction_fold(
            adapter=adapter,
            layer10_executor=layer10_executor,
            corrected_layer17_executor=corrected_layer17_executor,
            frozen_composition_executor=frozen_primary_executor,
            corrected_composition_executor=corrected_primary_executor,
            batches=held_batches_by_alias[held],
        )
        auxiliary = _score_auxiliary_conditions(
            adapter=adapter,
            executors={
                "source_layer17_only": source_layer17_executor,
                "l10_edgeless_frozen_l17_composition": (
                    frozen_edgeless_executor
                ),
                "l10_edgeless_a4_composition": corrected_edgeless_executor,
            },
            batches=held_batches_by_alias[held],
        )
        evaluation = _combine_fold_evaluations(core, auxiliary)
        shared_control = _shared_control_receipt(
            evaluation,
            prior_evaluation_by_alias[held],  # type: ignore[arg-type]
        )
        composition_receipt = _a4_composition_receipt(
            source_runtime_catalog=source_runtime_catalog,
            source_layer17_lowerings=layer17_lowerings,
            corrected_layer17_graph=correction.graph_plan,
            corrected_layer17_lowerings=correction.lowerings_by_node,
            corrected_primary_graph=corrected_primary,
            corrected_edgeless_graph=corrected_edgeless,
        )
        corrected_lowering_sha256_by_node = {
            name: correction.lowerings_by_node[name].artifact_sha256
            for name in correction.graph_plan.traversal_order
        }
        relocated_basis_sha256_by_node = {
            name: correction.lowerings_by_node[
                name
            ].computational_mode_basis.artifact_sha256
            for name in correction.graph_plan.traversal_order
        }
        fold_reports.append(
            {
                "fold_index": fold["fold_index"],
                "fold_id": fold["fold_id"],
                "held_family_alias": held,
                "training_family_aliases": list(training),
                "protocol_fold_sha256": fold["artifact_sha256"],
                "row_receipt": row_receipt,
                "a4_split_receipt": split_receipt,
                "correction_fit": correction.metadata(),
                "corrected_layer17_graph_sha256": (
                    correction.graph_plan.artifact_sha256
                ),
                "corrected_primary_graph_sha256": (
                    corrected_primary.artifact_sha256
                ),
                "corrected_edgeless_graph_sha256": (
                    corrected_edgeless.artifact_sha256
                ),
                "corrected_lowering_sha256_by_node": (
                    corrected_lowering_sha256_by_node
                ),
                "relocated_basis_sha256_by_node": (
                    relocated_basis_sha256_by_node
                ),
                "composition_receipt": composition_receipt,
                "shared_control_receipt": shared_control,
                "evaluation": evaluation,
            }
        )
        fold_evaluations.append(evaluation)
        fold_executables.append(
            {
                "fold_id": fold["fold_id"],
                "held_family_alias": held,
                "protocol_fold_sha256": fold["artifact_sha256"],
                "fit_split_sha256": row_receipt["fit_split_sha256"],
                "held_split_sha256": row_receipt["held_split_sha256"],
                "graph_plan": correction.graph_plan,
                "lowerings_by_node": correction.lowerings_by_node,
            }
        )
        del (
            fit_rows,
            held_rows,
            correction,
            corrected_primary,
            corrected_edgeless,
            corrected_primary_lowerings,
            corrected_edgeless_lowerings,
            corrected_layer17_executor,
            corrected_primary_executor,
            corrected_edgeless_executor,
            core,
            auxiliary,
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
        raise RuntimeError("A4 LOFO mutated frozen source state")
    aggregate = aggregate_full_block_closure_folds(fold_evaluations)
    interaction_factorial = _interaction_factorial(fold_evaluations)

    _progress("bundle: validate and publish eight fold-local executables")
    executable_bundle = build_gemma3_l10_l17_full_block_closure_fold_bundle(
        model_fingerprint=model_fingerprint,
        source_runtime_catalog_sha256=str(
            source_runtime_catalog["catalog_sha256"]
        ),
        source_composition_graph_sha256=primary_graph.artifact_sha256,
        folds=fold_executables,
    )
    bundle_written = False
    try:
        bundle_companion = (
            save_gemma3_l10_l17_full_block_closure_fold_bundle(
                fold_destination,
                executable_bundle,
            )
        )
        bundle_written = True
        fold_bundle_receipt = _fold_bundle_receipt(
            bundle=executable_bundle,
            companion=bundle_companion,
            bundle_path=fold_destination,
        )
        decision = evaluate_full_block_closure_lofo_gates(
            protocol=protocol,
            fold_aliases=tuple(
                str(value["held_family_alias"]) for value in fold_reports
            ),
            fold_evaluations=fold_evaluations,
            aggregate=aggregate,
            prior_a3_comparison=prior_a3_comparison,
            exact_projection_metadata_match=all(
                _projection_receipt_matches_protocol(
                    value["row_receipt"],  # type: ignore[arg-type]
                    protocol,
                )
                for value in fold_reports
            ),
            capture_audit=capture_audit,
            fold_bundle_receipt=fold_bundle_receipt,
            source_model_unchanged=source_model_unchanged,
            layer10_unchanged=layer10_unchanged,
        )
        projection_contract = protocol["projection_contract"]
        assert isinstance(projection_contract, Mapping)
        report = build_full_block_closure_lofo_report(
            protocol=protocol,
            authorization=authorization,
            prior_a3_comparison=prior_a3_comparison,
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
                "capture_audit": capture_audit,
                "capture_count": 1,
                "captured_examples": 256,
                "captured_sequences": fit_view.sequences,
                "captured_observations": captured_observations,
                "model_rows_recollected_per_fold": False,
                "a4_target_construction": protocol["target_contract"][  # type: ignore[index]
                    "raw_target_formula"
                ],
                "decoder_span_sha256": projection_contract[
                    "decoder_span_sha256"
                ],
                "summed_mean_sha256": projection_contract[
                    "summed_mean_sha256"
                ],
                "splitter_lineage": (
                    "sealed_a3_equal_family_fisher_splitter_wrapped_by_a4_"
                    "split_receipt"
                ),
            },
            fold_executable_bundle=fold_bundle_receipt,
            folds=fold_reports,
            aggregate=aggregate,
            interaction_factorial=interaction_factorial,
            decision=decision,
            source_model_unchanged=source_model_unchanged,
            layer10_unchanged=layer10_unchanged,
        )
        _progress("report: publish tensor-free strict JSON without overwrite")
        return save_gemma3_l10_l17_full_block_closure_lofo_report(
            destination,
            report,
        )
    except BaseException:
        if bundle_written:
            fold_destination.unlink(missing_ok=True)
            fold_companion.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed fit-only eight-family Gemma L10+L17 A4 "
            "full-block closure LOFO diagnostic."
        )
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--fold-bundle-output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    )
    parser.add_argument(
        "--prior-a3-report",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_TRAJECTORY_CORRECTION_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--composition-bundle",
        type=Path,
        default=DEFAULT_COMPOSITION_BUNDLE_PATH,
    )
    parser.add_argument(
        "--corpus-receipt",
        type=Path,
        default=DEFAULT_RECEIPT_OUTPUT,
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=DEFAULT_CORPUS_OUTPUT,
    )
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l10_l17_full_block_closure_lofo(
        revision=args.revision,
        output=args.output,
        fold_bundle_output=args.fold_bundle_output,
        prior_a3_report_path=args.prior_a3_report,
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
