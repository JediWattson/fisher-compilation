"""Strict A4 decoder-span versus generator-map oracle attribution.

This development-only rung replays the already-published A4 artifacts and
changes no learned parameter.  One ephemeral A-fit capture supplies the exact
Layer-17 full-block target.  For every sealed LOFO fold, the published frozen
decoder projection is rebuilt without fitting and three full-model-logit
conditions are scored on that fold's held family:

* the ordinary published A4 generator;
* the published generator attenuated by exactly 1/16;
* the exact held target projected into the frozen decoder span; and
* the exact held full-block target.

The intervention is diagnostic only.  It makes no resource, latency, serving,
held-out-confirmation, or whole-model-compilation claim.  The report is strict,
source-safe JSON containing scalars, counts, and hashes only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from .compiler.calibration import CalibrationBatch
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_l10_l17_a4_attenuation_sweep import (
    DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT,
    _metrics_equal,
    load_gemma3_l10_l17_a4_attenuation_sweep_report,
)
from .gemma3_l10_l17_full_block_closure_bundle import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    load_gemma3_l10_l17_full_block_closure_fold_bundle,
    restore_gemma3_l10_l17_full_block_closure_fold,
)
from .gemma3_l10_l17_full_block_closure_lofo import (
    DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    _EXPECTED_SOURCE_RUNTIME_CATALOG_SHA256,
    _TrajectoryCorrectionFitView,
    _build_source_runtime_catalog,
    _build_trajectory_correction_fold_rows_from_fit_view,
    _canonical_json_bytes,
    _fold_catalog,
    _full_block_capture_audit_receipt,
    _reject_forbidden_output_fields,
    _source_lowering_maps,
    _validate_frozen_selection,
    _validate_source_decoder_contract,
    _validate_source_runtime_catalog,
    load_gemma3_l10_l17_full_block_closure_lofo_report,
)
from .gemma3_l10_l17_open_a_progressive_evaluation import (
    DEFAULT_COMPOSITION_BUNDLE_PATH,
)
from .gemma3_l10_l17_trajectory_correction_fitting import (
    replace_layer_nodes_in_composed_graph,
)
from .gemma3_l10_l17_trajectory_correction_lofo import (
    _authenticate_before_fit_access,
    _merge_corrected_composition_lowerings,
    _shared_compiled_input,
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
    Gemma3Layer17FullBlockClosureCapture,
    capture_gemma3_layer17_full_block_closure,
)
from .gemma3_layer17_open_a_capacity_evaluation import (
    _add_comparison,
    _add_native,
    _candidate_comparison,
    _file_sha256,
    _finalize_metric_accumulator,
    _native_nll,
    _new_metric_accumulator,
    _selected_logits_and_targets,
    _validate_metric_container,
)
from .gemma3_layer17_v8_fit_lofo import _blocks_to_device, _family_blocks
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
    _batch_row_keys,
)
from .gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    select_top_fisher_same_layer_fragments,
)
from .modal_generator_graph import ModalGeneratorGraphPlan


__all__ = [
    "DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT",
    "GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_FORMAT_VERSION",
    "GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_SCHEMA",
    "aggregate_a4_oracle_folds",
    "build_a4_oracle_attribution_report",
    "classify_a4_oracle_attribution",
    "load_gemma3_l10_l17_a4_oracle_attribution_report",
    "run_gemma3_l10_l17_a4_oracle_attribution",
    "save_gemma3_l10_l17_a4_oracle_attribution_report",
    "score_a4_oracle_fold",
    "validate_gemma3_l10_l17_a4_oracle_attribution_report",
]


GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_SCHEMA = (
    "fisher_graph.gemma3_l10_l17_a4_oracle_attribution"
)
GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_FORMAT_VERSION = 1
DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "layer10-layer17-a4-oracle-attribution-v1.json"
)

_REPORT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-oracle-report:v1\0"
_ROW_RECEIPT_DOMAIN = b"fisher-graph:gemma3-l10-l17-a4-oracle-rows:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FAMILY_COUNT = 8
_VOCABULARY_CHUNK_SIZE = 16_384
_CONDITIONS = (
    "ordinary_a4_generated",
    "attenuated_a4_alpha_1_over_16",
    "exact_frozen_decoder_span",
    "exact_full_block_target",
)
_METRIC_FIELDS = (
    "nll_per_token",
    "delta_nll_per_token",
    "native_to_candidate_kl_per_token",
    "top1_agreement_to_native",
)
_SUCCESS_THRESHOLDS = {
    "maximum_delta_nll_per_token": 0.08,
    "maximum_native_to_candidate_kl_per_token": 0.09,
    "minimum_top1_agreement_to_native": 0.84,
}
_CONDITION_CONTRACT = {
    "ordinary_a4_generated": "published_fold_generator_replay",
    "attenuated_a4_alpha_1_over_16": (
        "published_generator_scaled_by_exact_binary_alpha_0_0625"
    ),
    "exact_frozen_decoder_span": (
        "held_full_block_target_exact_projection_into_frozen_decoder_span"
    ),
    "exact_full_block_target": "captured_exact_held_full_block_target",
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
    "source_bindings",
    "runtime",
    "capture",
    "conditions",
    "success_thresholds",
    "external_alpha_1_over_16_benchmark",
    "folds",
    "aggregate",
    "attribution",
    "capture_count",
    "refit_performed",
    "selection_opened",
    "guard_opened",
    "calibration_b_opened",
    "validation_opened",
    "test_opened",
    "full_model_logits_scored",
    "full_model_compiled",
    "heldout_confirmation",
    "serving_authorized",
    "resource_or_latency_claim",
    "safety",
    "report_sha256",
}
_SOURCE_BINDING_FIELDS = {
    "a4_report_file_sha256",
    "a4_report_sha256",
    "attenuation_report_file_sha256",
    "attenuation_report_sha256",
    "fold_bundle_file_sha256",
    "fold_bundle_payload_sha256",
    "composition_bundle_file_sha256",
    "composition_payload_sha256",
    "protocol_sha256",
    "source_runtime_catalog_sha256",
}
_RUNTIME_FIELDS = {
    "model_id",
    "requested_revision",
    "model_fingerprint",
    "device",
    "dtype",
    "local_files_only",
    "vocabulary_chunk_size",
}
_EVALUATION_FIELDS = {
    "supervised_tokens",
    "native",
    "conditions",
    "layer17_output_audit",
    "held_row_count",
    "span_rows_consumed",
    "exact_rows_consumed",
    "span_padded_positions_preserved",
    "exact_padded_positions_preserved",
    "execution_path",
    "application_boundary",
    "native_state_and_logits_same_forward",
    "candidate_state_and_logits_same_forward",
    "target_capture_and_scoring_same_forward",
    "full_model_logits_scored",
    "full_model_compiled",
    "heldout_confirmation",
    "serving_authorized",
    "resource_or_latency_claim",
    "refit_performed",
}
_STATE_AUDIT_FIELDS = {
    "valid_scalar_count",
    "max_abs_difference",
    "rms_difference",
    "reference_rms",
    "normalized_rms_difference",
}


def _domain_sha256(domain: bytes, value: object) -> str:
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


def _metric_copy(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_METRIC_FIELDS):
        raise ValueError(f"{label} metric fields are invalid")
    result = {
        field: _finite(value.get(field), label=f"{label} {field}")
        for field in _METRIC_FIELDS
    }
    if (
        result["nll_per_token"] < 0.0
        or result["native_to_candidate_kl_per_token"] < 0.0
        or not 0.0 <= result["top1_agreement_to_native"] <= 1.0
    ):
        raise ValueError(f"{label} metric range is invalid")
    return result


def _condition_succeeds(metric: Mapping[str, object]) -> bool:
    return (
        float(metric["delta_nll_per_token"])
        <= _SUCCESS_THRESHOLDS["maximum_delta_nll_per_token"]
        and float(metric["native_to_candidate_kl_per_token"])
        <= _SUCCESS_THRESHOLDS[
            "maximum_native_to_candidate_kl_per_token"
        ]
        and float(metric["top1_agreement_to_native"])
        >= _SUCCESS_THRESHOLDS["minimum_top1_agreement_to_native"]
    )


def classify_a4_oracle_attribution(
    *,
    generated_metric: Mapping[str, object],
    span_metric: Mapping[str, object],
    exact_metric: Mapping[str, object],
) -> dict[str, object]:
    """Classify the first failed boundary in the A4 approximation chain."""

    generated = _condition_succeeds(generated_metric)
    span = _condition_succeeds(span_metric)
    exact = _condition_succeeds(exact_metric)
    span_improves_generated = (
        float(span_metric["native_to_candidate_kl_per_token"])
        < float(generated_metric["native_to_candidate_kl_per_token"])
        and float(span_metric["delta_nll_per_token"])
        < float(generated_metric["delta_nll_per_token"])
    )
    if not exact:
        classification = "target_boundary_or_row_mapping"
        next_action = "audit_exact_target_boundary_and_runtime_row_mapping"
    elif span and not generated:
        classification = "generator_map"
        next_action = "revise_coordinate_generator_without_changing_decoder_span"
    elif not span and span_improves_generated:
        classification = "decoder_geometry_and_generator_map"
        next_action = "change_decoder_geometry_then_refit_generator_map"
    elif not span:
        classification = "euclidean_projection_or_span_geometry"
        next_action = (
            "test_downstream_sensitive_projection_before_expanding_decoder_span"
        )
    elif generated:
        classification = "no_attributed_failure"
        next_action = "confirm_with_sealed_non_fit_data"
    else:  # defensive completeness; the booleans above exhaust this state.
        raise RuntimeError("oracle attribution state is unreachable")
    return {
        "classification": classification,
        "ordinary_generated_succeeds": generated,
        "exact_decoder_span_succeeds": span,
        "exact_full_block_target_succeeds": exact,
        "frozen_span_capacity_resolved": span,
        "exact_span_improves_generated_on_kl_and_delta_nll": (
            span_improves_generated
        ),
        "next_action": next_action,
    }


@dataclass(slots=True)
class _RowCorrectionProvider:
    """One-shot exact-key override which leaves padded positions untouched."""

    adapter: Gemma3CausalLMAdapter
    batch: CalibrationBatch
    rows_by_key: Mapping[tuple[str, int], Tensor]
    consumed_keys: set[tuple[str, int]]
    call_count: int = 0
    padded_position_count: int = 0

    def __call__(self, generated: Tensor) -> Tensor:
        if self.call_count != 0:
            raise RuntimeError("row correction provider is one-shot")
        if generated.ndim != 3 or not generated.is_floating_point():
            raise ValueError("generated correction must have shape [B, S, D]")
        if self.batch.example_ids is None:
            raise ValueError("oracle replacement requires stable example ids")
        context = self.adapter.prepare_sequence(self.batch.model_inputs)
        valid = self.batch.valid_positions.to(device=context.logical_positions.device)
        if (
            tuple(generated.shape[:2]) != tuple(valid.shape)
            or valid.shape != context.logical_positions.shape
            or bool((valid & ~context.query_valid_mask).any())
        ):
            raise ValueError("oracle replacement sequence grid drifted")
        ordered_keys: list[tuple[str, int]] = []
        for batch_index, example_id in enumerate(self.batch.example_ids):
            positions = context.logical_positions[batch_index, valid[batch_index]]
            ordered_keys.extend(
                (example_id, int(position))
                for position in positions.detach().cpu().tolist()
            )
        if len(ordered_keys) != int(valid.sum().item()):
            raise RuntimeError("oracle replacement row accounting drifted")
        if len(ordered_keys) != len(set(ordered_keys)):
            raise ValueError("oracle replacement row keys are not unique")
        if any(key in self.consumed_keys for key in ordered_keys):
            raise RuntimeError("oracle replacement reused a row key")
        try:
            rows = torch.stack([self.rows_by_key[key] for key in ordered_keys])
        except KeyError as error:
            raise ValueError("oracle replacement is missing an exact row key") from error
        if rows.shape != (len(ordered_keys), generated.shape[-1]):
            raise ValueError("oracle replacement width drifted")
        replacement = generated.clone()
        runtime_valid = valid.to(device=generated.device)
        runtime_rows = rows.to(device=generated.device, dtype=generated.dtype)
        replacement[runtime_valid] = runtime_rows
        padding = ~runtime_valid
        if bool(padding.any()) and not torch.equal(
            replacement[padding], generated[padding]
        ):
            raise RuntimeError("oracle replacement changed padded positions")
        self.padded_position_count = int(padding.sum().item())
        self.consumed_keys.update(ordered_keys)
        self.call_count = 1
        return replacement


def _row_map(
    row_keys: Sequence[tuple[str, int]],
    rows: Tensor,
) -> dict[tuple[str, int], Tensor]:
    keys = tuple(row_keys)
    if (
        not keys
        or len(keys) != len(set(keys))
        or not isinstance(rows, Tensor)
        or rows.ndim != 2
        or rows.shape[0] != len(keys)
        or not bool(torch.isfinite(rows).all())
    ):
        raise ValueError("oracle row map is invalid")
    return {key: rows[index].detach().cpu() for index, key in enumerate(keys)}


def _summed_projected_rows(rows: AlignedFragmentRows) -> Tensor:
    contributions = tuple(
        value.contributions for value in rows.rows_by_fragment.values()
    )
    if not contributions or any(
        value.shape != contributions[0].shape for value in contributions[1:]
    ):
        raise ValueError("projected contribution catalog is invalid")
    # Avoid a transient [node, row, width] stack: the projection matrices are
    # the largest objects in this diagnostic.
    result = contributions[0].clone()
    for contribution in contributions[1:]:
        result.add_(contribution)
    return result


def _new_state_accumulator() -> dict[str, float | int]:
    return {
        "valid_scalar_count": 0,
        "squared_difference_sum": 0.0,
        "reference_squared_sum": 0.0,
        "max_abs_difference": 0.0,
    }


def _add_state_difference(
    accumulator: dict[str, float | int],
    *,
    native: Tensor,
    candidate: Tensor,
    valid_positions: Tensor,
) -> None:
    if native.shape != candidate.shape or native.ndim != 3:
        raise ValueError("Layer17 output audit shape drifted")
    valid = valid_positions.to(device=native.device)
    selected_native = native.detach()[valid].to(device="cpu", dtype=torch.float64)
    selected_candidate = candidate.detach()[valid].to(
        device="cpu", dtype=torch.float64
    )
    difference = selected_candidate - selected_native
    if difference.numel() == 0 or not bool(torch.isfinite(difference).all()):
        raise ValueError("Layer17 output audit is empty or non-finite")
    accumulator["valid_scalar_count"] = int(
        accumulator["valid_scalar_count"]
    ) + difference.numel()
    accumulator["squared_difference_sum"] = float(
        accumulator["squared_difference_sum"]
    ) + float(difference.square().sum().item())
    accumulator["reference_squared_sum"] = float(
        accumulator["reference_squared_sum"]
    ) + float(selected_native.square().sum().item())
    accumulator["max_abs_difference"] = max(
        float(accumulator["max_abs_difference"]),
        float(difference.abs().max().item()),
    )


def _finalize_state_difference(
    accumulator: Mapping[str, float | int],
) -> dict[str, float | int]:
    count = int(accumulator["valid_scalar_count"])
    if count <= 0:
        raise ValueError("Layer17 output audit has no valid scalars")
    rmse = math.sqrt(float(accumulator["squared_difference_sum"]) / count)
    reference_rms = math.sqrt(float(accumulator["reference_squared_sum"]) / count)
    return {
        "valid_scalar_count": count,
        "max_abs_difference": float(accumulator["max_abs_difference"]),
        "rms_difference": rmse,
        "reference_rms": reference_rms,
        "normalized_rms_difference": rmse / max(reference_rms, 1e-30),
    }


def score_a4_oracle_fold(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
    span_rows_by_key: Mapping[tuple[str, int], Tensor],
    exact_rows_by_key: Mapping[tuple[str, int], Tensor],
) -> dict[str, object]:
    """Score one held family under ordinary, span-oracle, and exact-oracle paths."""

    materialized = tuple(batches)
    if not materialized or any(
        not isinstance(batch, CalibrationBatch) for batch in materialized
    ):
        raise ValueError("oracle batches must contain CalibrationBatch values")
    if executor.affected_layer_ordinals != (10, 17):
        raise ValueError("oracle executor must compose Layers 10 and 17")
    if 17 not in executor.post_feedforward_delta_layer_ordinals:
        raise ValueError("oracle executor lacks the Layer17 delta boundary")
    expected_keys = tuple(
        key for batch in materialized for key in _batch_row_keys(adapter, batch)
    )
    if (
        not expected_keys
        or len(expected_keys) != len(set(expected_keys))
        or set(span_rows_by_key) != set(expected_keys)
        or set(exact_rows_by_key) != set(expected_keys)
    ):
        raise ValueError("oracle row catalogs do not exactly cover held batches")

    accumulator = _new_metric_accumulator(_CONDITIONS)
    state = {condition: _new_state_accumulator() for condition in _CONDITIONS}
    consumed = {
        "exact_frozen_decoder_span": set(),
        "exact_full_block_target": set(),
    }
    padded_counts = {name: 0 for name in consumed}
    with executor.validated_transaction():
        for batch in materialized:
            # The adapter owns use_cache/return_dict.  Supplying those boolean
            # runtime flags here would violate its Tensor-only input contract.
            call_inputs = batch.model_inputs
            with torch.no_grad():
                native_run = adapter.forward(
                    call_inputs,
                    capture_sites=("layer.17.output",),
                    retain_gradients=False,
                )
            native_logits, targets = _selected_logits_and_targets(
                native_run.logits, batch
            )
            _add_native(
                accumulator,
                nll_sum=_native_nll(native_logits, targets),
                token_count=targets.numel(),
            )
            native_state = native_run.activations["layer.17.output"].detach()
            del native_run
            gc.collect()

            def score_candidate(condition: str, candidate_run: object) -> None:
                logits = getattr(candidate_run, "logits", None)
                activations = getattr(candidate_run, "activations", None)
                if not isinstance(logits, Tensor) or not isinstance(
                    activations, Mapping
                ):
                    raise TypeError("oracle candidate run is invalid")
                candidate_logits, candidate_targets = (
                    _selected_logits_and_targets(logits, batch)
                )
                if not torch.equal(targets, candidate_targets):
                    raise RuntimeError(f"{condition} held targets drifted")
                _add_comparison(
                    accumulator,
                    condition,
                    _candidate_comparison(
                        native_logits,
                        candidate_logits,
                        targets,
                        vocabulary_chunk_size=_VOCABULARY_CHUNK_SIZE,
                    ),
                )
                candidate_state = activations.get("layer.17.output")
                if not isinstance(candidate_state, Tensor):
                    raise TypeError("oracle candidate Layer17 state is unavailable")
                _add_state_difference(
                    state[condition],
                    native=native_state,
                    candidate=candidate_state,
                    valid_positions=batch.valid_positions,
                )
                del candidate_logits, candidate_targets, candidate_state

            with torch.no_grad():
                ordinary_run = executor.run_with_generated_overlay(
                    lambda: adapter.forward(
                        call_inputs,
                        capture_sites=("layer.17.output",),
                        retain_gradients=False,
                    ),
                    expected_forward_calls=1,
                )
            score_candidate("ordinary_a4_generated", ordinary_run)
            del ordinary_run
            gc.collect()

            with torch.no_grad():
                attenuated_run = (
                    executor.run_with_diagnostic_post_feedforward_delta_attenuation(
                        lambda: adapter.forward(
                            call_inputs,
                            capture_sites=("layer.17.output",),
                            retain_gradients=False,
                        ),
                        layer_ordinal=17,
                        alpha=0.0625,
                        expected_forward_calls=1,
                    )
                )
            score_candidate("attenuated_a4_alpha_1_over_16", attenuated_run)
            del attenuated_run
            gc.collect()

            for condition, rows in (
                ("exact_frozen_decoder_span", span_rows_by_key),
                ("exact_full_block_target", exact_rows_by_key),
            ):
                provider = _RowCorrectionProvider(
                    adapter=adapter,
                    batch=batch,
                    rows_by_key=rows,
                    consumed_keys=consumed[condition],
                )
                with torch.no_grad():
                    candidate_run = (
                        executor.run_with_diagnostic_post_feedforward_delta_override(
                            lambda: adapter.forward(
                                call_inputs,
                                capture_sites=("layer.17.output",),
                                retain_gradients=False,
                            ),
                            layer_ordinal=17,
                            correction_provider=provider,
                            expected_forward_calls=1,
                        )
                    )
                if provider.call_count != 1:
                    raise RuntimeError("oracle provider did not execute exactly once")
                padded_counts[condition] += provider.padded_position_count
                score_candidate(condition, candidate_run)
                del candidate_run, provider
                gc.collect()
            del native_logits, native_state, targets, score_candidate
            gc.collect()

    for condition in consumed:
        if consumed[condition] != set(expected_keys):
            raise RuntimeError(f"{condition} did not consume every held row exactly once")
    metrics = _finalize_metric_accumulator(accumulator, conditions=_CONDITIONS)
    _validate_metric_container(
        metrics,
        label="A4 oracle fold",
        conditions=_CONDITIONS,
    )
    return {
        **metrics,
        "layer17_output_audit": {
            condition: _finalize_state_difference(state[condition])
            for condition in _CONDITIONS
        },
        "held_row_count": len(expected_keys),
        "span_rows_consumed": len(consumed["exact_frozen_decoder_span"]),
        "exact_rows_consumed": len(consumed["exact_full_block_target"]),
        "span_padded_positions_preserved": padded_counts[
            "exact_frozen_decoder_span"
        ],
        "exact_padded_positions_preserved": padded_counts[
            "exact_full_block_target"
        ],
        "execution_path": "full_model_logits_a4_oracle_attribution",
        "application_boundary": "layer.17.mlp.delta",
        "native_state_and_logits_same_forward": True,
        "candidate_state_and_logits_same_forward": True,
        "target_capture_and_scoring_same_forward": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "refit_performed": False,
    }


def aggregate_a4_oracle_folds(
    evaluations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    values = tuple(evaluations)
    if len(values) != _FAMILY_COUNT:
        raise ValueError("oracle aggregate requires exactly eight folds")
    token_total = sum(int(value["supervised_tokens"]) for value in values)
    if token_total <= 0:
        raise ValueError("oracle aggregate has no supervised tokens")

    def aggregate_metric(*, micro: bool, condition: str | None) -> dict[str, float]:
        def source(value: Mapping[str, object]) -> Mapping[str, object]:
            if condition is None:
                raw = value["native"]
            else:
                raw = value["conditions"][condition]  # type: ignore[index]
            if not isinstance(raw, Mapping):
                raise TypeError("oracle fold metric is unavailable")
            return raw

        fields = ("nll_per_token",) if condition is None else _METRIC_FIELDS
        result: dict[str, float] = {}
        for field in fields:
            numerator = sum(
                float(source(value)[field])
                * (int(value["supervised_tokens"]) if micro else 1)
                for value in values
            )
            denominator = token_total if micro else len(values)
            result[field] = numerator / denominator
        return result

    def scope(micro: bool) -> dict[str, object]:
        native = aggregate_metric(micro=micro, condition=None)
        conditions = {
            condition: aggregate_metric(micro=micro, condition=condition)
            for condition in _CONDITIONS
        }
        # Match the sealed attenuation report's aggregation semantics exactly:
        # aggregate native and candidate NLL independently, then derive delta.
        # Averaging the already-derived per-fold deltas is mathematically the
        # same but can differ by one floating-point ulp and would make an exact
        # authenticated replay fail at publication time.
        for metric in conditions.values():
            metric["delta_nll_per_token"] = (
                metric["nll_per_token"] - native["nll_per_token"]
            )
        return {
            "supervised_tokens": token_total,
            "native": native,
            "conditions": conditions,
        }

    return {
        "family_count": len(values),
        "supervised_tokens": token_total,
        "micro": scope(True),
        "equal_family_macro": scope(False),
    }


def _downstream_geometry_receipt(
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    witness_count = 0
    attenuation_better_kl_count = 0
    attenuation_worse_state_count = 0
    for fold in folds:
        evaluation = fold.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise TypeError("oracle geometry fold evaluation is unavailable")
        conditions = evaluation.get("conditions")
        state = evaluation.get("layer17_output_audit")
        if not isinstance(conditions, Mapping) or not isinstance(state, Mapping):
            raise TypeError("oracle geometry evidence is unavailable")
        alpha_metric = conditions.get("attenuated_a4_alpha_1_over_16")
        span_metric = conditions.get("exact_frozen_decoder_span")
        alpha_state = state.get("attenuated_a4_alpha_1_over_16")
        span_state = state.get("exact_frozen_decoder_span")
        if not all(
            isinstance(value, Mapping)
            for value in (alpha_metric, span_metric, alpha_state, span_state)
        ):
            raise TypeError("oracle geometry condition evidence is unavailable")
        assert isinstance(alpha_metric, Mapping)
        assert isinstance(span_metric, Mapping)
        assert isinstance(alpha_state, Mapping)
        assert isinstance(span_state, Mapping)
        better_kl = float(alpha_metric["native_to_candidate_kl_per_token"]) < float(
            span_metric["native_to_candidate_kl_per_token"]
        )
        worse_state = float(alpha_state["normalized_rms_difference"]) > float(
            span_state["normalized_rms_difference"]
        )
        attenuation_better_kl_count += int(better_kl)
        attenuation_worse_state_count += int(worse_state)
        witness_count += int(better_kl and worse_state)
    return {
        "definition": (
            "alpha_1_over_16_has_lower_kl_but_higher_layer17_output_nrmse_"
            "than_exact_euclidean_decoder_span"
        ),
        "family_count": len(folds),
        "attenuation_lower_kl_family_count": attenuation_better_kl_count,
        "attenuation_higher_state_nrmse_family_count": attenuation_worse_state_count,
        "joint_misalignment_witness_family_count": witness_count,
        "all_families_witness_downstream_geometry_misalignment": (
            witness_count == len(folds)
        ),
        "attenuation_preserves_original_affine_decoder_image": False,
        "does_not_resolve_frozen_span_capacity": True,
        "scientific_scope": "diagnostic_evidence_not_selection_or_gate",
    }


def build_a4_oracle_attribution_report(
    *,
    source_bindings: Mapping[str, object],
    runtime: Mapping[str, object],
    capture: Mapping[str, object],
    attenuation_macro_benchmark: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fold_values = [dict(value) for value in folds]
    evaluations = [value["evaluation"] for value in fold_values]
    if any(not isinstance(value, Mapping) for value in evaluations):
        raise TypeError("oracle fold evaluation is unavailable")
    aggregate = aggregate_a4_oracle_folds(evaluations)  # type: ignore[arg-type]
    macro_conditions = aggregate["equal_family_macro"]["conditions"]  # type: ignore[index]
    if not _metrics_equal(
        macro_conditions["attenuated_a4_alpha_1_over_16"],
        attenuation_macro_benchmark,
    ):
        raise ValueError("rescored alpha=1/16 macro differs from publication")
    attribution = {
        **classify_a4_oracle_attribution(
            generated_metric=macro_conditions["ordinary_a4_generated"],
            span_metric=macro_conditions["exact_frozen_decoder_span"],
            exact_metric=macro_conditions["exact_full_block_target"],
        ),
        "downstream_geometry": _downstream_geometry_receipt(fold_values),
    }
    payload = {
        "schema": GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_SCHEMA,
        "format_version": GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_FORMAT_VERSION,
        "scientific_role": "calibration_a_fit_a4_oracle_attribution",
        "source_bindings": dict(source_bindings),
        "runtime": dict(runtime),
        "capture": dict(capture),
        "conditions": dict(_CONDITION_CONTRACT),
        "success_thresholds": dict(_SUCCESS_THRESHOLDS),
        "external_alpha_1_over_16_benchmark": {
            "alpha": 0.0625,
            "condition_id": "alpha_2m4",
            "role": "external_published_benchmark_exactly_rescored_not_selected",
            "equal_family_macro": _metric_copy(
                attenuation_macro_benchmark,
                label="external alpha=1/16 benchmark",
            ),
        },
        "folds": fold_values,
        "aggregate": aggregate,
        "attribution": attribution,
        "capture_count": 1,
        "refit_performed": False,
        "selection_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "resource_or_latency_claim": False,
        "safety": dict(_SAFETY),
    }
    _reject_forbidden_output_fields(payload, path="A4 oracle report")
    report = {
        **payload,
        "report_sha256": _domain_sha256(_REPORT_DOMAIN, payload),
    }
    return validate_gemma3_l10_l17_a4_oracle_attribution_report(report)


def validate_gemma3_l10_l17_a4_oracle_attribution_report(
    report: object,
) -> dict[str, object]:
    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise ValueError("A4 oracle report fields are invalid")
    raw = dict(report)
    if (
        raw.get("schema") != GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_SCHEMA
        or raw.get("format_version")
        != GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_FORMAT_VERSION
        or raw.get("scientific_role")
        != "calibration_a_fit_a4_oracle_attribution"
        or raw.get("capture_count") != 1
        or raw.get("refit_performed") is not False
        or raw.get("full_model_logits_scored") is not True
        or raw.get("full_model_compiled") is not False
        or raw.get("heldout_confirmation") is not False
        or raw.get("serving_authorized") is not False
        or raw.get("resource_or_latency_claim") is not False
        or any(
            raw.get(field) is not False
            for field in (
                "selection_opened",
                "guard_opened",
                "calibration_b_opened",
                "validation_opened",
                "test_opened",
            )
        )
        or raw.get("safety") != _SAFETY
        or raw.get("success_thresholds") != _SUCCESS_THRESHOLDS
    ):
        raise ValueError("A4 oracle report boundary is invalid")
    source_bindings = raw.get("source_bindings")
    runtime = raw.get("runtime")
    if not isinstance(source_bindings, Mapping) or not isinstance(
        runtime, Mapping
    ):
        raise TypeError("A4 oracle source/runtime binding is unavailable")
    if set(source_bindings) != _SOURCE_BINDING_FIELDS:
        raise ValueError("A4 oracle source binding fields are invalid")
    for field in _SOURCE_BINDING_FIELDS:
        _require_sha256(source_bindings.get(field), label=f"oracle source {field}")
    if (
        set(runtime) != _RUNTIME_FIELDS
        or not isinstance(runtime.get("model_id"), str)
        or not runtime.get("model_id")
        or not isinstance(runtime.get("requested_revision"), str)
        or _REVISION.fullmatch(str(runtime.get("requested_revision"))) is None
        or runtime.get("local_files_only") is not True
        or runtime.get("vocabulary_chunk_size") != _VOCABULARY_CHUNK_SIZE
        or not isinstance(runtime.get("device"), str)
        or not runtime.get("device")
        or not isinstance(runtime.get("dtype"), str)
        or not runtime.get("dtype")
    ):
        raise ValueError("A4 oracle runtime identity is invalid")
    _require_sha256(runtime.get("model_fingerprint"), label="oracle model")
    condition_contract = raw.get("conditions")
    if condition_contract != _CONDITION_CONTRACT:
        raise ValueError("A4 oracle condition contract is invalid")
    capture = raw.get("capture")
    if (
        not isinstance(capture, Mapping)
        or set(capture)
        != {
            "capture_sha256",
            "capture_audit_sha256",
            "row_key_sha256",
            "observations",
            "sequences",
            "all_required_capture_audits_pass",
        }
        or capture.get("all_required_capture_audits_pass") is not True
        or type(capture.get("observations")) is not int
        or int(capture["observations"]) <= 0
        or capture.get("sequences") != 256
    ):
        raise ValueError("A4 oracle capture receipt is invalid")
    for field in ("capture_sha256", "capture_audit_sha256", "row_key_sha256"):
        _require_sha256(capture.get(field), label=f"oracle capture {field}")
    folds = raw.get("folds")
    if (
        isinstance(folds, (str, bytes))
        or not isinstance(folds, Sequence)
        or len(folds) != _FAMILY_COUNT
    ):
        raise ValueError("A4 oracle requires exactly eight folds")
    evaluations: list[Mapping[str, object]] = []
    aliases: set[str] = set()
    held_observations_total = 0
    held_sequences_total = 0
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or set(fold) != {
            "fold_index",
            "fold_id_sha256",
            "held_family_alias_sha256",
            "protocol_fold_sha256",
            "row_receipt_sha256",
            "held_row_key_sha256",
            "held_observations",
            "held_sequences",
            "a4_layer17_graph_sha256",
            "a4_composition_graph_sha256",
            "evaluation",
            "external_alpha_1_over_16_benchmark",
        }:
            raise ValueError(f"A4 oracle fold {index} fields are invalid")
        if fold.get("fold_index") != index:
            raise ValueError("A4 oracle fold order drifted")
        for field in (
            "fold_id_sha256",
            "held_family_alias_sha256",
            "protocol_fold_sha256",
            "row_receipt_sha256",
            "held_row_key_sha256",
            "a4_layer17_graph_sha256",
            "a4_composition_graph_sha256",
        ):
            _require_sha256(fold.get(field), label=f"oracle fold {index} {field}")
        alias_hash = str(fold["held_family_alias_sha256"])
        if alias_hash in aliases:
            raise ValueError("A4 oracle held-family hash repeated")
        aliases.add(alias_hash)
        if (
            type(fold.get("held_observations")) is not int
            or int(fold["held_observations"]) <= 0
            or type(fold.get("held_sequences")) is not int
            or int(fold["held_sequences"]) <= 0
        ):
            raise ValueError("A4 oracle held counts are invalid")
        held_observations_total += int(fold["held_observations"])
        held_sequences_total += int(fold["held_sequences"])
        evaluation = fold.get("evaluation")
        if not isinstance(evaluation, Mapping) or set(evaluation) != _EVALUATION_FIELDS:
            raise TypeError("A4 oracle fold evaluation is unavailable")
        _validate_metric_container(
            evaluation,
            label=f"A4 oracle fold {index}",
            conditions=_CONDITIONS,
        )
        if (
            evaluation.get("held_row_count") != fold["held_observations"]
            or evaluation.get("span_rows_consumed") != fold["held_observations"]
            or evaluation.get("exact_rows_consumed") != fold["held_observations"]
            or evaluation.get("native_state_and_logits_same_forward") is not True
            or evaluation.get("candidate_state_and_logits_same_forward")
            is not True
            or evaluation.get("target_capture_and_scoring_same_forward")
            is not False
            or evaluation.get("execution_path")
            != "full_model_logits_a4_oracle_attribution"
            or evaluation.get("application_boundary") != "layer.17.mlp.delta"
            or evaluation.get("full_model_logits_scored") is not True
            or any(
                evaluation.get(field) is not False
                for field in (
                    "full_model_compiled",
                    "heldout_confirmation",
                    "serving_authorized",
                    "resource_or_latency_claim",
                    "refit_performed",
                )
            )
        ):
            raise ValueError("A4 oracle exact-row coverage is invalid")
        for field in (
            "span_padded_positions_preserved",
            "exact_padded_positions_preserved",
        ):
            if type(evaluation.get(field)) is not int or int(evaluation[field]) < 0:
                raise ValueError("A4 oracle preserved-position count is invalid")
        state_audit = evaluation.get("layer17_output_audit")
        if not isinstance(state_audit, Mapping) or set(state_audit) != set(
            _CONDITIONS
        ):
            raise ValueError("A4 oracle Layer17 output audit catalog is invalid")
        state_scalar_counts: set[int] = set()
        for condition, record in state_audit.items():
            if not isinstance(record, Mapping) or set(record) != _STATE_AUDIT_FIELDS:
                raise ValueError(
                    f"A4 oracle {condition} Layer17 output audit fields are invalid"
                )
            if (
                type(record.get("valid_scalar_count")) is not int
                or int(record["valid_scalar_count"]) <= 0
            ):
                raise ValueError("A4 oracle state-audit scalar count is invalid")
            state_scalar_counts.add(int(record["valid_scalar_count"]))
            maximum = _finite(
                record.get("max_abs_difference"),
                label=f"oracle {condition} state max",
            )
            rmse = _finite(
                record.get("rms_difference"),
                label=f"oracle {condition} state RMSE",
            )
            reference_rms = _finite(
                record.get("reference_rms"),
                label=f"oracle {condition} state reference RMS",
            )
            normalized = _finite(
                record.get("normalized_rms_difference"),
                label=f"oracle {condition} state NRMSE",
            )
            if (
                maximum < 0.0
                or rmse < 0.0
                or normalized < 0.0
                or reference_rms <= 0.0
            ):
                raise ValueError(
                    "A4 oracle state-audit magnitudes are outside range"
                )
            expected_normalized = rmse / reference_rms
            if not math.isclose(
                normalized,
                expected_normalized,
                rel_tol=0.0,
                abs_tol=max(math.ulp(expected_normalized) * 2.0, 1e-15),
            ):
                raise ValueError("A4 oracle state-audit NRMSE identity drifted")
        if len(state_scalar_counts) != 1:
            raise ValueError("A4 oracle state-audit scalar counts disagree")
        evaluations.append(evaluation)
        _metric_copy(
            fold.get("external_alpha_1_over_16_benchmark"),
            label=f"fold {index} alpha benchmark",
        )
        if not _metrics_equal(
            evaluation["conditions"]["attenuated_a4_alpha_1_over_16"],  # type: ignore[index]
            fold.get("external_alpha_1_over_16_benchmark"),
        ):
            raise ValueError("rescored fold alpha=1/16 differs from publication")
    if (
        held_observations_total != capture["observations"]
        or held_sequences_total != capture["sequences"]
    ):
        raise ValueError("A4 oracle fold coverage differs from capture")
    reproduced = aggregate_a4_oracle_folds(evaluations)
    if not _metrics_equal(reproduced, raw.get("aggregate")):
        raise ValueError("A4 oracle aggregate was not reproduced")
    macro = reproduced["equal_family_macro"]["conditions"]  # type: ignore[index]
    expected_attribution = {
        **classify_a4_oracle_attribution(
            generated_metric=macro["ordinary_a4_generated"],
            span_metric=macro["exact_frozen_decoder_span"],
            exact_metric=macro["exact_full_block_target"],
        ),
        "downstream_geometry": _downstream_geometry_receipt(
            [dict(value) for value in folds]  # type: ignore[arg-type]
        ),
    }
    if raw.get("attribution") != expected_attribution:
        raise ValueError("A4 oracle attribution was not reproduced")
    external = raw.get("external_alpha_1_over_16_benchmark")
    if (
        not isinstance(external, Mapping)
        or external.get("alpha") != 0.0625
        or external.get("condition_id") != "alpha_2m4"
        or external.get("role")
        != "external_published_benchmark_exactly_rescored_not_selected"
    ):
        raise ValueError("A4 oracle external benchmark is invalid")
    _metric_copy(external.get("equal_family_macro"), label="alpha benchmark")
    if not _metrics_equal(
        external.get("equal_family_macro"),
        macro["attenuated_a4_alpha_1_over_16"],
    ):
        raise ValueError("A4 oracle external macro benchmark was not replayed")
    supplied = _require_sha256(raw.pop("report_sha256"), label="A4 oracle report")
    if supplied != _domain_sha256(_REPORT_DOMAIN, raw):
        raise ValueError("A4 oracle report hash mismatch")
    raw["report_sha256"] = supplied
    _reject_forbidden_output_fields(raw, path="A4 oracle report")
    return raw


def save_gemma3_l10_l17_a4_oracle_attribution_report(
    path: Path | str,
    report: Mapping[str, object],
) -> dict[str, object]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A4 oracle report")
    validated = validate_gemma3_l10_l17_a4_oracle_attribution_report(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
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
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if destination.exists():
            raise FileExistsError("refusing to overwrite A4 oracle report")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def load_gemma3_l10_l17_a4_oracle_attribution_report(
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
        raise ValueError("A4 oracle report is not strict JSON") from error
    if not isinstance(raw, Mapping):
        raise TypeError("A4 oracle report must contain one object")
    return validate_gemma3_l10_l17_a4_oracle_attribution_report(raw)


def _progress(message: str) -> None:
    print(f"[a4-oracle] {message}", file=os.sys.stderr, flush=True)


def _string_sha256(value: str, *, domain: bytes) -> str:
    return hashlib.sha256(domain + value.encode("utf-8")).hexdigest()


def run_gemma3_l10_l17_a4_oracle_attribution(
    *,
    revision: str,
    output: Path | str = DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
    source_a4_report_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT
    ),
    fold_bundle_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE
    ),
    attenuation_report_path: Path | str = (
        DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT
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
    """Run exactly one A-fit capture and the eight-fold oracle attribution."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite A4 oracle report")
    a4_path = Path(source_a4_report_path)
    executable_path = Path(fold_bundle_path)
    attenuation_path = Path(attenuation_report_path)

    _progress("preflight: authenticate A4, attenuation, folds, corpus, composition")
    source_report = load_gemma3_l10_l17_full_block_closure_lofo_report(a4_path)
    fold_bundle = load_gemma3_l10_l17_full_block_closure_fold_bundle(
        executable_path
    )
    attenuation = load_gemma3_l10_l17_a4_attenuation_sweep_report(
        attenuation_path
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
        raise ValueError("oracle runtime must exactly replay published A4 runtime")
    attenuation_source = attenuation.get("source_a4_report")
    attenuation_folds = attenuation.get("fold_executable_bundle")
    if not isinstance(attenuation_source, Mapping) or not isinstance(
        attenuation_folds, Mapping
    ):
        raise TypeError("published attenuation source bindings are unavailable")
    if (
        attenuation_source.get("file") != a4_path.name
        or attenuation_source.get("file_sha256") != _file_sha256(a4_path)
        or attenuation_source.get("report_sha256")
        != source_report.get("report_sha256")
        or attenuation_source.get("protocol_sha256")
        != protocol.get("artifact_sha256")
        or attenuation_folds.get("file") != executable_path.name
        or attenuation_folds.get("file_sha256") != _file_sha256(executable_path)
        or attenuation_folds.get("scientific_payload_sha256")
        != fold_bundle.get("scientific_payload_sha256")
        or attenuation.get("runtime")
        != {
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": source_runtime.get("model_fingerprint"),
            "device": device_name,
            "dtype": dtype,
            "local_files_only": True,
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
        }
    ):
        raise ValueError("published attenuation evidence is not bound to A4")
    source_fold_receipt = source_report.get("fold_executable_bundle")
    if not isinstance(source_fold_receipt, Mapping) or (
        fold_bundle.get("scientific_payload_sha256")
        != source_fold_receipt.get("scientific_payload_sha256")
        or fold_bundle.get("protocol_sha256")
        != source_fold_receipt.get("protocol_sha256")
        or _file_sha256(executable_path)
        != source_fold_receipt.get("tensor_file_sha256")
    ):
        raise ValueError("published A4 fold executable bundle drifted")

    bundle, authority, _, fit_authorization = _authenticate_before_fit_access(
        bundle_path=composition_bundle_path,
        corpus_receipt_path=corpus_receipt_path,
        corpus_artifact_path=corpus_artifact_path,
        fit_input_path=fit_input_path,
    )
    if (
        not _metrics_equal(
            source_authorization.get("bundle"), fit_authorization.get("bundle")
        )
        or not _metrics_equal(
            source_authorization.get("fit_authority"),
            fit_authorization.get("fit_authority"),
        )
        or source_authorization.get("fit_authority_sha256")
        != fit_authorization.get("fit_authority_sha256")
    ):
        raise ValueError("live A-fit authority differs from published A4")
    if (
        getattr(bundle, "model_id", None) != model_id
        or getattr(bundle, "requested_revision", None) != revision
    ):
        raise ValueError("live composition model identity drifted")
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
        raise ValueError("Layer17 selected fragment order differs from protocol")
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
        not _metrics_equal(
            source_authorization.get("source_runtime_catalog"), catalog
        )
        or fold_bundle.get("source_runtime_catalog_sha256")
        != catalog.get("catalog_sha256")
        or fold_bundle.get("source_composition_graph_sha256")
        != primary_graph.artifact_sha256
    ):
        raise ValueError("published A4 artifacts differ from live composition")

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
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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

    _progress("tokenize: replay the exact 256-example calibration-A fit stream")
    raw_blocks, materialization = materialize_gemma3_layer17_family_lofo(
        authority, tokenizer
    )
    validate_gemma3_layer17_family_lofo_materialization_metadata(materialization)
    fit_collection = source_report.get("fit_collection")
    if not isinstance(fit_collection, Mapping) or not _metrics_equal(
        materialization, fit_collection.get("materialization")
    ):
        raise ValueError("live A-fit materialization differs from published A4")
    blocks = _blocks_to_device(_family_blocks(raw_blocks), device)
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
    source_lowering_by_name = {
        node.name: lowering
        for node, lowering in zip(
            primary_graph.nodes, bundle_lowerings, strict=True
        )
    }
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

    _progress("capture: one exact Layer17 full-block closure replay over all 256")
    capture: Gemma3Layer17FullBlockClosureCapture = (
        capture_gemma3_layer17_full_block_closure(
            adapter,
            all_batches,
            selection=selection,
            leaf_activation_site=selection.execution_order[0].input_site,
            layer10_executor=layer10_executor,
            layer17_executor=source_layer17_executor,
        )
    )
    expected_sequences = sum(batch.batch_size for batch in all_batches)
    if expected_sequences != 256 or capture.native_rows.sequences != 256:
        raise RuntimeError("A4 oracle capture sequence accounting drifted")
    capture_metadata = capture.metadata()
    if not _metrics_equal(capture_metadata, fit_collection.get("capture")):
        raise ValueError("fresh oracle capture differs from published A4 capture")
    capture_audit = _full_block_capture_audit_receipt(capture_metadata, protocol)
    if (
        capture_audit.get("all_required_capture_audits_pass") is not True
        or not _metrics_equal(capture_audit, fit_collection.get("capture_audit"))
    ):
        raise RuntimeError("fresh A4 oracle capture audits failed or drifted")
    fragment_ids = tuple(capture.trajectory_rows.fragment_ids)
    fit_view = _TrajectoryCorrectionFitView(
        compiled_input=_shared_compiled_input(capture.compiled_rows, fragment_ids),
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
    capture_receipt = {
        "capture_sha256": capture.capture_sha256,
        "capture_audit_sha256": _domain_sha256(
            b"a4-oracle:capture-audit:\0", capture_audit
        ),
        "row_key_sha256": fit_view.row_key_sha256,
        "observations": fit_view.observations,
        "sequences": fit_view.sequences,
        "all_required_capture_audits_pass": True,
    }
    # Float32 is the exact runtime injection dtype for this pinned run.  Keep
    # one dense target matrix plus one integer lookup instead of tens of
    # thousands of long-lived row views.
    exact_target = fit_view.a3_target.to(dtype=torch.float32).contiguous()
    exact_index_by_key = {
        key: index for index, key in enumerate(fit_view.row_keys)
    }
    # The lean fit view now owns every tensor required to reproduce the fold
    # projections.  Drop the much broader capture immediately.
    del capture, layer10_executor, source_layer17_executor
    gc.collect()
    held_batches_by_alias = dict(blocks)
    source_folds = source_report.get("folds")
    attenuation_fold_values = attenuation.get("folds")
    if (
        isinstance(source_folds, (str, bytes))
        or not isinstance(source_folds, Sequence)
        or isinstance(attenuation_fold_values, (str, bytes))
        or not isinstance(attenuation_fold_values, Sequence)
        or len(source_folds) != _FAMILY_COUNT
        or len(attenuation_fold_values) != _FAMILY_COUNT
    ):
        raise ValueError("published oracle source fold catalogs are incomplete")

    # Projection is the only operation which needs the large capture-bound
    # fit view.  Rebuild all eight disjoint held projections first, retaining
    # only their float32 predictions and scalar/hash lineage.  The capture and
    # capture-only executors are freed before any full-model scoring forward.
    prepared_folds: list[dict[str, object]] = []
    for index, (fold, source_fold, attenuation_fold) in enumerate(
        zip(
            _fold_catalog(protocol),
            source_folds,
            attenuation_fold_values,
            strict=True,
        )
    ):
        if not isinstance(source_fold, Mapping) or not isinstance(
            attenuation_fold, Mapping
        ):
            raise TypeError("published oracle source fold is invalid")
        held = str(fold["held_family_alias"])
        _progress(f"fold {index + 1}/{_FAMILY_COUNT}: rebuild held projection {held}")
        fit_rows, held_rows, row_receipt = (
            _build_trajectory_correction_fold_rows_from_fit_view(
                fit_view,
                family_alias_by_example=family_alias_by_example,
                training_family_aliases=tuple(fold["training_family_aliases"]),
                held_family_alias=held,
                fold_sha256=str(fold["artifact_sha256"]),
                protocol_sha256=str(protocol["artifact_sha256"]),
                authenticated_stream_sha256=stream_sha256,
                source_graph=layer17_graph,
                source_lowerings_by_node=layer17_lowerings,
            )
        )
        if not _metrics_equal(row_receipt, source_fold.get("row_receipt")):
            raise ValueError(f"rebuilt oracle row receipt differs for fold {index}")
        if (
            held_rows.row_key_sha256
            != source_fold.get("row_receipt", {}).get("held_row_key_sha256")
            or fit_rows.row_key_sha256
            != source_fold.get("row_receipt", {}).get("fit_row_key_sha256")
        ):
            raise ValueError("rebuilt A4 row-key hash differs from publication")
        span_prediction = _summed_projected_rows(held_rows).to(
            dtype=torch.float32
        ).contiguous()
        external_metric = attenuation_fold["evaluation"]["conditions"][  # type: ignore[index]
            "alpha_2m4"
        ]
        prepared_folds.append(
            {
                "fold": dict(fold),
                "source_fold": source_fold,
                "held": held,
                "held_row_keys": held_rows.row_keys,
                "held_row_key_sha256": held_rows.row_key_sha256,
                "held_observations": held_rows.observations,
                "held_sequences": held_rows.sequences,
                "row_receipt_sha256": _domain_sha256(
                    _ROW_RECEIPT_DOMAIN, row_receipt
                ),
                "span_prediction": span_prediction,
                "external_metric": _metric_copy(
                    external_metric, label=f"fold {index} alpha=1/16 benchmark"
                ),
            }
        )
        del fit_rows, held_rows, span_prediction, row_receipt
        gc.collect()

    del fit_view
    gc.collect()

    fold_results: list[dict[str, object]] = []
    for index, prepared in enumerate(prepared_folds):
        fold = prepared["fold"]
        source_fold = prepared["source_fold"]
        held = str(prepared["held"])
        held_row_keys = tuple(prepared["held_row_keys"])  # type: ignore[arg-type]
        span_prediction = prepared["span_prediction"]
        if (
            not isinstance(fold, Mapping)
            or not isinstance(source_fold, Mapping)
            or not isinstance(span_prediction, Tensor)
        ):
            raise TypeError("prepared oracle fold is invalid")
        _progress(f"fold {index + 1}/{_FAMILY_COUNT}: score held {held}")
        span_map = _row_map(held_row_keys, span_prediction)
        exact_indices = torch.tensor(
            [exact_index_by_key[key] for key in held_row_keys], dtype=torch.long
        )
        exact_map = _row_map(
            held_row_keys,
            exact_target.index_select(0, exact_indices),
        )
        graph, lowerings = restore_gemma3_l10_l17_full_block_closure_fold(
            fold_bundle, index
        )
        if (
            graph.artifact_sha256
            != source_fold.get("corrected_layer17_graph_sha256")
            or {
                name: lowerings[name].artifact_sha256
                for name in graph.traversal_order
            }
            != source_fold.get("corrected_lowering_sha256_by_node")
        ):
            raise ValueError("restored oracle fold executable is cross-bound")
        composition = replace_layer_nodes_in_composed_graph(
            primary_graph, graph, layer_ordinal=17
        )
        if composition.artifact_sha256 != source_fold.get(
            "corrected_primary_graph_sha256"
        ):
            raise ValueError("restored oracle composition graph drifted")
        merged = _merge_corrected_composition_lowerings(
            composition,
            layer10_lowerings_by_node=layer10_lowerings,
            corrected_layer17_lowerings_by_node=lowerings,
        )
        executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            composition,
            merged,
            post_feedforward_delta_layer_ordinals=(17,),
        )
        evaluation = score_a4_oracle_fold(
            adapter=adapter,
            executor=executor,
            batches=held_batches_by_alias[held],
            span_rows_by_key=span_map,
            exact_rows_by_key=exact_map,
        )
        published_a4 = source_fold["evaluation"]["conditions"][  # type: ignore[index]
            "a4_full_block_corrected_composition"
        ]
        if not _metrics_equal(
            evaluation["conditions"]["ordinary_a4_generated"],  # type: ignore[index]
            published_a4,
        ):
            raise RuntimeError("ordinary A4 oracle replay differs from publication")
        if not _metrics_equal(
            evaluation["conditions"][  # type: ignore[index]
                "attenuated_a4_alpha_1_over_16"
            ],
            prepared["external_metric"],
        ):
            raise RuntimeError(
                "alpha=1/16 oracle replay differs from attenuation publication"
            )
        fold_results.append(
            {
                "fold_index": index,
                "fold_id_sha256": _string_sha256(
                    str(fold["fold_id"]), domain=b"a4-oracle:fold-id:\0"
                ),
                "held_family_alias_sha256": _string_sha256(
                    held, domain=b"a4-oracle:held-family:\0"
                ),
                "protocol_fold_sha256": fold["artifact_sha256"],
                "row_receipt_sha256": prepared["row_receipt_sha256"],
                "held_row_key_sha256": prepared["held_row_key_sha256"],
                "held_observations": prepared["held_observations"],
                "held_sequences": prepared["held_sequences"],
                "a4_layer17_graph_sha256": graph.artifact_sha256,
                "a4_composition_graph_sha256": composition.artifact_sha256,
                "evaluation": evaluation,
                "external_alpha_1_over_16_benchmark": prepared[
                    "external_metric"
                ],
            }
        )
        del span_map, exact_map, exact_indices, executor, prepared["span_prediction"]
        gc.collect()

    attenuation_macro = attenuation["aggregate"]["equal_family_macro"][  # type: ignore[index]
        "conditions"
    ]["alpha_2m4"]
    source_bindings = {
        "a4_report_file_sha256": _file_sha256(a4_path),
        "a4_report_sha256": source_report["report_sha256"],
        "attenuation_report_file_sha256": _file_sha256(attenuation_path),
        "attenuation_report_sha256": attenuation["report_sha256"],
        "fold_bundle_file_sha256": _file_sha256(executable_path),
        "fold_bundle_payload_sha256": fold_bundle["scientific_payload_sha256"],
        "composition_bundle_file_sha256": bundle_binding["bundle_file_sha256"],
        "composition_payload_sha256": bundle_binding[
            "composition_payload_sha256"
        ],
        "protocol_sha256": protocol["artifact_sha256"],
        "source_runtime_catalog_sha256": catalog["catalog_sha256"],
    }
    report = build_a4_oracle_attribution_report(
        source_bindings=source_bindings,
        runtime={
            "model_id": model_id,
            "requested_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "device": str(device),
            "dtype": dtype,
            "local_files_only": True,
            "vocabulary_chunk_size": _VOCABULARY_CHUNK_SIZE,
        },
        capture=capture_receipt,
        attenuation_macro_benchmark=attenuation_macro,
        folds=fold_results,
    )
    _progress("report: publish strict scalar/count/hash JSON without overwrite")
    return save_gemma3_l10_l17_a4_oracle_attribution_report(destination, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed A4 decoder-span/generator oracle attribution."
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A4_ORACLE_ATTRIBUTION_OUTPUT,
    )
    parser.add_argument(
        "--source-a4-report",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_LOFO_OUTPUT,
    )
    parser.add_argument(
        "--fold-bundle",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_FULL_BLOCK_CLOSURE_FOLD_BUNDLE,
    )
    parser.add_argument(
        "--attenuation-report",
        type=Path,
        default=DEFAULT_GEMMA3_L10_L17_A4_ATTENUATION_SWEEP_OUTPUT,
    )
    parser.add_argument(
        "--composition-bundle", type=Path, default=DEFAULT_COMPOSITION_BUNDLE_PATH
    )
    parser.add_argument("--corpus-receipt", type=Path, default=DEFAULT_RECEIPT_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument("--fit-input", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_gemma3_l10_l17_a4_oracle_attribution(
        revision=arguments.revision,
        output=arguments.output,
        source_a4_report_path=arguments.source_a4_report,
        fold_bundle_path=arguments.fold_bundle,
        attenuation_report_path=arguments.attenuation_report,
        composition_bundle_path=arguments.composition_bundle,
        corpus_receipt_path=arguments.corpus_receipt,
        corpus_artifact_path=arguments.corpus,
        fit_input_path=arguments.fit_input,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
