"""Run the first joint lexical, attention, MLP, and residual graph arm.

This development experiment is the first standalone candidate in the research
ladder that places every learned Gemma boundary inside one compiled runtime:

* one physically tied lexical factor performs embedding ingress and logit
  egress;
* all 18 attention mutations execute through reduced, source-independent
  Q/K/V/O generators;
* all 18 MLP mutations execute through Fisher-weighted activation charts; and
* strictly-forward attention-to-MLP transports cross block boundaries while a
  single ordered residual carrier preserves the model state.

The rank-reduced attention initializer is a deterministic weight-SVD capacity
control.  MLP charts and cross-block transports are Fisher/activation fitted.
The fit/evaluation panel is prompt-disjoint but not family-disjoint, so this
runner produces development evidence only and never authorizes a compression
claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .gemma3_ablation_experiment import _FrozenModelTensorGuard
from .gemma3_attention_generator import (
    Gemma3AttentionGenerator,
    compile_svd_reduced_attention_generator,
)
from .gemma3_cross_layer_chart_transport_model_experiment import (
    _ordered_example_ids,
    _tokenize_keyed_batches,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_joint_cross_block_chart_executor import (
    Gemma3JointCrossBlockChartExecutor,
    PINNED_GEMMA3_EVENT_COUNT,
    PINNED_GEMMA3_MUTATION_COUNT,
    build_joint_cross_block_event_schedule,
)
from .gemma3_joint_cross_block_compiler import (
    Gemma3JointGraphCompilation,
    compile_gemma3_joint_cross_block_candidate,
    compile_gemma3_joint_exact_control,
)
from .gemma3_joint_cross_block_model_fit import (
    JointCrossBlockFitResult,
    fit_ewc_joint_cross_block_whole_model,
    fit_joint_cross_block_mlp_charts,
    fit_pullback_ewc_shared_tied_lexical_factor,
)
from .gemma3_l10_l17_a5e_functional_mlp_channel_coalescing_experiment import (
    DEFAULT_PANEL_PATH,
    DEFAULT_REVISION,
    _load_prompt_split,
)
from .gemma3_layer_agnostic_whole_model_compression_protocol import (
    DEFAULT_PROTOCOL_OUTPUT,
    PINNED_HIDDEN_WIDTH,
    PINNED_LAYER_COUNT,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_SOURCE_UNIQUE_PARAMETER_COUNT,
    build_layer_agnostic_whole_model_compression_protocol,
    load_layer_agnostic_whole_model_compression_protocol,
)
from .gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
    _source_lexical_modules,
    fit_shared_tied_lexical_factor,
)
from .gemma3_one_block_residual_form_experiment import (
    _LogitMetricAccumulator,
    _native_nll_sum,
)


GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_SCHEMA = (
    "fisher_graph.gemma3_joint_cross_block_whole_model.experiment.v1"
)
GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_FORMAT_VERSION = 1
DEVELOPMENT_ARM_ID = "joint_lexical_attention_mlp_cross_block_development"
PROTOCOL_RELATIONSHIP = "unrestricted_layer_agnostic_primary_precursor"
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "joint-lexical-attention-mlp-cross-block-dev-v1.json"
)
DEFAULT_LEXICAL_RANK = 512
DEFAULT_ATTENTION_INPUT_RANK = 256
DEFAULT_ATTENTION_OUTPUT_RANK = 64
DEFAULT_MLP_GENERATOR_RANK = 4
DEFAULT_MAXIMUM_ATTENTION_SOURCES_PER_TARGET_CHART = 1
DEFAULT_TRANSPORT_RELATIVE_RIDGE = 1.0e-4
DEFAULT_BATCH_SIZE = 4
DETERMINISTIC_SEED = 1729
_REPORT_HASH_DOMAIN = (
    b"fisher_graph:gemma3-joint-cross-block-whole-model-report:v1\0"
)


@dataclass(frozen=True, slots=True)
class _JointCandidateBundle:
    compilation: Gemma3JointGraphCompilation
    fit: JointCrossBlockFitResult
    attention_generators: tuple[Gemma3AttentionGenerator, ...]


def _progress(message: str) -> None:
    print(f"[joint-cross-block] {message}", flush=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _report_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _REPORT_HASH_DOMAIN + _canonical_json_bytes(value)
    ).hexdigest()


def _protocol_receipt(path: Path | str) -> dict[str, object]:
    observed = load_layer_agnostic_whole_model_compression_protocol(Path(path))
    current = build_layer_agnostic_whole_model_compression_protocol()
    if observed != current:
        raise ValueError("joint experiment protocol artifact is not canonical")
    arms = current.get("arms")
    primary = [
        item
        for item in arms if isinstance(item, Mapping)
        and item.get("arm_id") == "unrestricted_layer_agnostic_primary"
    ] if isinstance(arms, list) else []
    if len(primary) != 1:
        raise ValueError("layer-agnostic primary protocol arm is unavailable")
    return {
        "schema": current.get("schema"),
        "format_version": current.get("format_version"),
        "protocol_sha256": current.get("protocol_sha256"),
        "relationship": PROTOCOL_RELATIONSHIP,
        "primary_arm": copy.deepcopy(dict(primary[0])),
        "primary_arm_fully_executed": False,
    }


def _compile_exact_control(
    adapter: Gemma3CausalLMAdapter,
) -> Gemma3JointGraphCompilation:
    return compile_gemma3_joint_exact_control(
        adapter,
        dtype=torch.float32,
        device=next(adapter.module.parameters()).device,
    )


def _exact_attention_catalog(
    compilation: Gemma3JointGraphCompilation,
) -> tuple[Gemma3AttentionGenerator, ...]:
    mutations = tuple(compilation.executor.mutations)
    result = tuple(
        mutation.compiled_operator
        for mutation in mutations
        if mutation.event_spec.event_kind == "attention_mutation"
    )
    if (
        len(result) != PINNED_LAYER_COUNT
        or any(not isinstance(item, Gemma3AttentionGenerator) for item in result)
    ):
        raise RuntimeError("exact control omitted its attention generator catalog")
    return result  # type: ignore[return-value]


def _compile_joint_executors(
    *,
    adapter: Gemma3CausalLMAdapter,
    exact_control: Gemma3JointGraphCompilation,
    fit_batches: Sequence[Mapping[str, Tensor]],
    fit_batch_row_keys: Sequence[Sequence[tuple[str, int]]],
    lexical_rank: int,
    attention_input_rank: int,
    attention_output_rank: int,
    mlp_generator_rank: int,
    maximum_attention_sources_per_target_chart: int,
    transport_relative_ridge: float,
    use_ewc: bool = False,
    fuse_mlp_modes: bool = False,
) -> _JointCandidateBundle:
    """Fit and compile the candidate after the exact gate has passed."""

    exact_attention = _exact_attention_catalog(exact_control)
    _, _, source_weight, embed_scale = _source_lexical_modules(adapter)
    if use_ewc:
        lexical = fit_pullback_ewc_shared_tied_lexical_factor(
            adapter=adapter,
            exact_executor=exact_control.executor,
            batches=fit_batches,
            source_weight=source_weight,
            lexical_rank=lexical_rank,
            embed_scale=embed_scale,
        )
        if not isinstance(lexical, SharedTiedLexicalFactor) or lexical.is_exact:
            raise RuntimeError("joint candidate lexical boundary was not compressed")
        reduced_attention, fit = fit_ewc_joint_cross_block_whole_model(
            adapter=adapter,
            exact_executor=exact_control.executor,
            lexical=lexical,
            exact_attention=exact_attention,
            batches=fit_batches,
            batch_row_keys=fit_batch_row_keys,
            model_fingerprint=adapter.model_fingerprint(),
            attention_input_rank=attention_input_rank,
            attention_output_rank=attention_output_rank,
            generator_rank=mlp_generator_rank,
            maximum_incoming_attention_sources_per_target_chart=(
                maximum_attention_sources_per_target_chart
            ),
            fuse_mlp_modes=fuse_mlp_modes,
            progress=_progress,
        )
    else:
        reduced_attention = tuple(
            compile_svd_reduced_attention_generator(
                item,
                input_rank=attention_input_rank,
                output_rank=attention_output_rank,
            )
            for item in exact_attention
        )
        lexical = fit_shared_tied_lexical_factor(
            source_weight,
            lexical_rank,
            embed_scale=embed_scale,
        )
        if not isinstance(lexical, SharedTiedLexicalFactor) or lexical.is_exact:
            raise RuntimeError("joint candidate lexical boundary was not compressed")
        fit = fit_joint_cross_block_mlp_charts(
            adapter=adapter,
            batches=fit_batches,
            batch_row_keys=fit_batch_row_keys,
            attention_generators=reduced_attention,
            model_fingerprint=adapter.model_fingerprint(),
            generator_rank=mlp_generator_rank,
            maximum_incoming_attention_sources_per_target_chart=(
                maximum_attention_sources_per_target_chart
            ),
            transport_relative_ridge=transport_relative_ridge,
            progress=_progress,
        )
    compilation = compile_gemma3_joint_cross_block_candidate(
        adapter,
        lexical=lexical,
        attention_generators=reduced_attention,
        mlp_chart_generators=fit.mlp_generators,
        transports=fit.transports,
        dtype=torch.float32,
        device=next(adapter.module.parameters()).device,
    )
    return _JointCandidateBundle(
        compilation=compilation,
        fit=fit,
        attention_generators=reduced_attention,
    )


def _source_call_modules(
    adapter: Gemma3CausalLMAdapter,
) -> dict[str, nn.Module]:
    ingress, egress, _, _ = _source_lexical_modules(adapter)
    backbone = getattr(adapter.module, "model", None)
    terminal = getattr(backbone, "norm", None)
    if not isinstance(terminal, nn.Module):
        raise TypeError("Gemma terminal RMSNorm is unavailable")
    result: dict[str, nn.Module] = {
        "source.lexical_ingress": ingress,
        "source.terminal_rmsnorm": terminal,
        "source.lexical_egress": egress,
    }
    for ordinal, layer in enumerate(adapter.layers):
        source = adapter.source_module(layer.id)
        result[f"source.block.{ordinal}"] = source
        result[f"source.attention.{ordinal}"] = source.self_attn
        result[f"source.mlp.{ordinal}"] = source.mlp
    return result


def _evaluate_joint_executor(
    *,
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3JointCrossBlockChartExecutor,
    batches: Sequence[Mapping[str, Tensor]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Compare one standalone joint executor with the loaded teacher."""

    materialized = tuple(batches)
    if not materialized:
        raise ValueError("joint evaluation requires tokenized batches")
    modules = _source_call_modules(adapter)
    native_calls = {name: 0 for name in modules}
    compiled_calls = {name: 0 for name in modules}
    phase = "idle"
    handles: list[Any] = []

    def count(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
        *,
        name: str,
    ) -> None:
        if phase == "native":
            native_calls[name] += 1
        elif phase == "compiled":
            compiled_calls[name] += 1
        else:
            raise RuntimeError("source module executed outside an audited phase")

    for name, module in modules.items():
        handles.append(
            module.register_forward_hook(
                lambda module, args, output, *, name=name: count(
                    module, args, output, name=name
                )
            )
        )
    accumulator = _LogitMetricAccumulator()
    native_nll_sum = 0.0
    supervised_tokens = 0
    event_receipts: list[list[str]] = []
    operator_receipts: list[list[dict[str, object]]] = []
    peak_packets = 0
    peak_packet_bytes = 0
    try:
        for batch in materialized:
            phase = "native"
            with torch.inference_mode():
                native = adapter.forward(batch)
            nll_sum, tokens = _native_nll_sum(native.logits, batch)
            native_nll_sum += nll_sum
            supervised_tokens += tokens
            phase = "compiled"
            with torch.inference_mode():
                sequence = adapter.prepare_sequence(batch)
                execution = executor.execute(batch["input_ids"], sequence)
            phase = "idle"
            accumulator.update(native.logits, execution.logits, batch)
            event_receipts.append(list(execution.event_ids))
            receipts = [item.metadata() for item in execution.operator_receipts]
            if len(receipts) != PINNED_GEMMA3_MUTATION_COUNT:
                raise RuntimeError("joint operator receipt count drifted")
            operator_receipts.append(receipts)
            peak_packets = max(peak_packets, execution.peak_live_transport_packets)
            peak_packet_bytes = max(
                peak_packet_bytes,
                execution.peak_live_transport_packet_bytes,
            )
    finally:
        phase = "idle"
        for handle in reversed(handles):
            handle.remove()
    if supervised_tokens <= 0:
        raise RuntimeError("joint evaluation produced no supervised tokens")
    if any(compiled_calls.values()):
        raise RuntimeError("standalone joint executor called a teacher module")
    expected_events = list(executor.ordered_event_ids)
    if any(receipt != expected_events for receipt in event_receipts):
        raise RuntimeError("joint 39-event execution order drifted")
    attention_receipts = [
        item
        for batch in operator_receipts
        for item in batch
        if item["event_kind"] == "attention_mutation"
    ]
    if len(attention_receipts) != PINNED_LAYER_COUNT * len(materialized) or any(
        item["source_projection_call_count"] != 0
        or item["source_fallback_call_count"] != 0
        for item in attention_receipts
    ):
        raise RuntimeError("compiled attention source-call receipt failed")
    native_nll = native_nll_sum / supervised_tokens
    metrics = {
        "native": {
            "supervised_tokens": supervised_tokens,
            "nll_per_token": native_nll,
        },
        "candidate": accumulator.result(native_nll_per_token=native_nll),
    }
    audit = {
        "batch_count": len(materialized),
        "native_source_calls": native_calls,
        "compiled_source_calls": compiled_calls,
        "compiled_source_call_total": sum(compiled_calls.values()),
        "zero_live_teacher_calls": True,
        "event_count_per_execution": PINNED_GEMMA3_EVENT_COUNT,
        "mutation_count_per_execution": PINNED_GEMMA3_MUTATION_COUNT,
        "event_receipts": event_receipts,
        "operator_receipts": operator_receipts,
        "attention_operator_receipt_count": len(attention_receipts),
        "attention_source_projection_call_total": sum(
            int(item["source_projection_call_count"])
            for item in attention_receipts
        ),
        "attention_source_fallback_call_total": sum(
            int(item["source_fallback_call_count"])
            for item in attention_receipts
        ),
        "peak_live_transport_packets": peak_packets,
        "peak_live_transport_packet_bytes": peak_packet_bytes,
    }
    return metrics, audit


def _assert_exact_control_gate(metrics: Mapping[str, object]) -> None:
    candidate = metrics.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("exact joint control metrics are incomplete")
    required = {
        "delta_nll_per_token": 0.0,
        "native_to_candidate_kl_per_query": 0.0,
        "top1_agreement_to_native": 1.0,
        "logit_nrmse": 0.0,
        "maximum_absolute_logit_error": 0.0,
    }
    if any(candidate.get(key) != value for key, value in required.items()):
        raise RuntimeError("exact joint 39-event control failed literal parity")


def _resource_report(
    compilation: Gemma3JointGraphCompilation,
    *,
    lexical_rank: int,
    attention_input_rank: int,
    attention_output_rank: int,
    mlp_generator_rank: int,
) -> dict[str, object]:
    inventory = compilation.executor.resource_inventory()
    candidate = int(inventory["candidate_deployed_scalars"])
    source = PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    difference = source - candidate
    return {
        "source_unique_parameter_count": source,
        "candidate_deployed_scalar_count": candidate,
        "parameter_difference": difference,
        "parameter_reduction_fraction": difference / source,
        "projected_source_float32_bytes": 4 * source,
        "projected_candidate_float32_bytes": 4 * candidate,
        "projected_byte_reduction_fraction": difference / source,
        "lexical_rank": lexical_rank,
        "attention_input_rank": attention_input_rank,
        "attention_output_rank": attention_output_rank,
        "mlp_generator_rank": mlp_generator_rank,
        "executor_inventory": inventory,
        "lexical_factor_counted_once": (
            inventory["lexical_ingress_egress_factor_counted_once"] is True
        ),
        "source_independent_in_memory_executor": True,
        "serialized_candidate_artifact_written": False,
        "serialized_byte_reduction_authoritative": False,
        "whole_model_mac_ledger_closed": False,
        "whole_model_mac_reduction_claimed": False,
        "latency_reduction_claimed": False,
    }


def _qualification(
    metrics: Mapping[str, object],
    resources: Mapping[str, object],
    *,
    exact_control_passed: bool,
    cross_block_edge_count: int,
) -> dict[str, object]:
    candidate = metrics.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate metrics are incomplete")

    def number(name: str) -> float | None:
        value = candidate.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result = float(value)
            return result if math.isfinite(result) else None
        return None

    delta_nll = number("delta_nll_per_token")
    kl = number("native_to_candidate_kl_per_query")
    top1 = number("top1_agreement_to_native")
    parameter_reduction = float(resources["parameter_reduction_fraction"])
    available_fidelity = {
        "delta_nll_at_most_0_10": delta_nll is not None and delta_nll <= 0.10,
        "kl_at_most_0_10": kl is not None and kl <= 0.10,
        "top1_at_least_0_90": top1 is not None and top1 >= 0.90,
    }
    reasons = []
    if not exact_control_passed:
        reasons.append("exact_39_event_control_failed")
    if not all(available_fidelity.values()):
        reasons.append("available_fidelity_screen_failed")
    if parameter_reduction < 0.20:
        reasons.append("whole_model_parameter_reduction_below_20_percent")
    if cross_block_edge_count <= 0:
        reasons.append("no_positive_cross_block_transport")
    reasons.extend(
        (
            "family_disjoint_guard_not_run",
            "downstream_retention_not_run",
            "matched_svd_and_deletion_controls_not_run",
            "zero_and_shuffled_transport_controls_not_run",
            "serialized_source_free_artifact_not_written",
            "whole_model_mac_ledger_not_closed",
        )
    )
    return {
        "exact_control_passed": exact_control_passed,
        "available_fidelity_screen": available_fidelity,
        "available_fidelity_screen_passed": all(available_fidelity.values()),
        "whole_model_parameter_reduction_observed": parameter_reduction,
        "whole_model_parameter_gate_passed": parameter_reduction >= 0.20,
        "positive_cross_block_transport_present": cross_block_edge_count > 0,
        "family_disjoint_guard_passed": False,
        "downstream_retention_passed": False,
        "serialized_source_free_artifact_established": False,
        "qualified": False,
        "passed": False,
        "compression_claim_authorized": False,
        "compression_success_claimed": False,
        "failure_reasons": reasons,
    }


def _validate_report_semantics(report: Mapping[str, object]) -> None:
    if (
        report.get("schema") != GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_SCHEMA
        or report.get("format_version")
        != GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_FORMAT_VERSION
        or report.get("contains_model_weights") is not False
        or report.get("contains_executor_weights") is not False
        or report.get("contains_prompt_text") is not False
        or report.get("contains_token_ids") is not False
    ):
        raise ValueError("joint report header is invalid")
    scientific = report.get("scientific_status")
    compilation = report.get("compilation")
    metrics = report.get("metrics")
    audit = report.get("execution_audit")
    resources = report.get("resources")
    qualification = report.get("qualification")
    protocol = report.get("protocol")
    configuration = report.get("configuration")
    model = report.get("model")
    data = report.get("data")
    if not all(
        isinstance(value, Mapping)
        for value in (
            scientific,
            compilation,
            metrics,
            audit,
            resources,
            qualification,
            protocol,
            configuration,
            model,
            data,
        )
    ):
        raise ValueError("joint report sections are incomplete")
    assert isinstance(scientific, Mapping)
    assert isinstance(compilation, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(audit, Mapping)
    assert isinstance(resources, Mapping)
    assert isinstance(qualification, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(configuration, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(data, Mapping)
    current_protocol = build_layer_agnostic_whole_model_compression_protocol()
    if (
        protocol.get("protocol_sha256")
        != current_protocol.get("protocol_sha256")
        or protocol.get("relationship") != PROTOCOL_RELATIONSHIP
        or protocol.get("primary_arm_fully_executed") is not False
    ):
        raise ValueError("joint report protocol receipt is invalid")
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("requested_revision") != PINNED_MODEL_REVISION
        or model.get("parameter_count") != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or model.get("hidden_size") != PINNED_HIDDEN_WIDTH
        or model.get("num_hidden_layers") != PINNED_LAYER_COUNT
        or model.get("resolved_commit") != PINNED_MODEL_REVISION
        or model.get("device") != "cpu"
        or model.get("dtype") != "torch.float32"
        or model.get("weights_in_artifact") is not False
        or data.get("prompt_disjoint") is not True
        or data.get("family_disjoint") is not False
        or type(data.get("fit_example_count")) is not int
        or int(data["fit_example_count"]) <= 0
        or type(data.get("evaluation_example_count")) is not int
        or int(data["evaluation_example_count"]) <= 0
        or any(
            not isinstance(data.get(name), str)
            or len(str(data[name])) != 64
            for name in (
                "panel_sha256",
                "fit_example_ids_sha256",
                "evaluation_example_ids_sha256",
            )
        )
    ):
        raise ValueError("joint model or data provenance is invalid")
    if any(
        scientific.get(name) is not False
        for name in (
            "family_disjoint_evaluation",
            "heldout_confirmation",
            "compression_success_claimed",
            "compression_claim_authorized",
            "latency_or_kernel_speed_claimed",
        )
    ):
        raise ValueError("joint report claim boundary is invalid")
    if (
        scientific.get("development_only") is not True
        or scientific.get("exact_39_event_control_passed") is not True
        or scientific.get("tied_lexical_embedding_and_head_compiled") is not True
        or scientific.get("all_18_attention_mutations_compiled") is not True
        or scientific.get("all_18_mlp_mutations_compiled") is not True
        or scientific.get("single_residual_carrier_compiled") is not True
        or scientific.get(
            "cross_block_cross_component_transports_compiled"
        )
        is not True
        or scientific.get("prefill_only") is not True
        or scientific.get("cached_decode_tested") is not False
    ):
        raise ValueError("joint report execution scope is invalid")
    candidate_manifest = compilation.get("candidate_manifest")
    exact_manifest = compilation.get("exact_control_manifest")
    exact_compilation = compilation.get("exact_control")
    candidate_compilation = compilation.get("joint_candidate")
    if not isinstance(candidate_manifest, Mapping) or not isinstance(
        exact_manifest, Mapping
    ) or not isinstance(exact_compilation, Mapping) or not isinstance(
        candidate_compilation, Mapping
    ):
        raise ValueError("joint report manifests are missing")
    fit_row_identity = compilation.get("fit_row_identity_sha256")
    if not isinstance(fit_row_identity, str) or len(fit_row_identity) != 64:
        raise ValueError("joint fit-row identity is invalid")
    canonical_schedule = build_joint_cross_block_event_schedule()
    canonical_event_ids = [item.event_id for item in canonical_schedule]
    if (
        list(exact_manifest.get("ordered_event_ids", ()))
        != canonical_event_ids
        or list(candidate_manifest.get("ordered_event_ids", ()))
        != canonical_event_ids
    ):
        raise ValueError("joint report event schedule is not canonical")
    exact_inventory = exact_manifest.get("resource_inventory")
    candidate_inventory = candidate_manifest.get("resource_inventory")
    exact_lexical = exact_manifest.get("lexical")
    candidate_lexical = candidate_manifest.get("lexical")
    exact_carrier = exact_manifest.get("residual_carrier")
    candidate_carrier = candidate_manifest.get("residual_carrier")
    exact_inventory_sum = (
        sum(
            int(exact_inventory.get(name, -1))
            for name in (
                "lexical_shared_physical_scalars",
                "mutation_scalars",
                "transport_scalars",
                "terminal_norm_scalars",
            )
        )
        if isinstance(exact_inventory, Mapping)
        else -1
    )
    candidate_inventory_sum = (
        sum(
            int(candidate_inventory.get(name, -1))
            for name in (
                "lexical_shared_physical_scalars",
                "mutation_scalars",
                "transport_scalars",
                "terminal_norm_scalars",
            )
        )
        if isinstance(candidate_inventory, Mapping)
        else -1
    )
    if (
        exact_manifest.get("event_count") != 39
        or exact_manifest.get("attention_mutation_count") != 18
        or exact_manifest.get("mlp_mutation_count") != 18
        or exact_manifest.get("direct_compiled_attention_mutation_count") != 18
        or exact_manifest.get("exact_control_arm") is not True
        or exact_manifest.get("live_teacher_reference_present") is not False
        or exact_manifest.get("native_fallback_present") is not False
        or exact_manifest.get(
            "source_checkpoint_coefficients_retained_by_mutations"
        )
        is not True
        or candidate_manifest.get("event_count") != 39
        or candidate_manifest.get("attention_mutation_count") != 18
        or candidate_manifest.get("mlp_mutation_count") != 18
        or candidate_manifest.get("direct_compiled_attention_mutation_count")
        != 18
        or candidate_manifest.get("candidate_arm") is not True
        or candidate_manifest.get("live_teacher_reference_present") is not False
        or candidate_manifest.get("native_fallback_present") is not False
        or candidate_manifest.get(
            "source_checkpoint_coefficients_retained_by_mutations"
        )
        is not False
        or candidate_manifest.get("lexical_factor_registered_once") is not True
        or candidate_manifest.get("strictly_forward_transports") is not True
        or candidate_manifest.get(
            "cross_block_cross_component_transport_count"
        )
        is None
        or int(
            candidate_manifest["cross_block_cross_component_transport_count"]
        )
        <= 0
        or not isinstance(exact_inventory, Mapping)
        or exact_inventory.get("inventory_closes") is not True
        or exact_inventory.get("candidate_deployed_scalars")
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or exact_inventory_sum != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or not isinstance(candidate_inventory, Mapping)
        or candidate_inventory.get("inventory_closes") is not True
        or candidate_inventory_sum
        != candidate_inventory.get("candidate_deployed_scalars")
        or candidate_inventory.get("serialized_artifact_written") is not False
        or candidate_inventory.get("source_unique_parameter_reduction_claimed")
        is not False
        or candidate_inventory.get("whole_model_mac_reduction_claimed") is not False
        or candidate_inventory.get("latency_reduction_claimed") is not False
        or not isinstance(exact_lexical, Mapping)
        or exact_lexical.get("physical_storage_count") != 1
        or exact_lexical.get("second_head_copy_present") is not False
        or exact_lexical.get("rank") != PINNED_HIDDEN_WIDTH
        or exact_lexical.get("is_exact_copied_matrix") is not True
        or not isinstance(candidate_lexical, Mapping)
        or candidate_lexical.get("physical_storage_count") != 1
        or candidate_lexical.get("second_head_copy_present") is not False
        or not isinstance(exact_carrier, Mapping)
        or exact_carrier.get("mutation_count") != 36
        or exact_carrier.get("initial_version") != 0
        or exact_carrier.get("final_version") != 36
        or exact_carrier.get("identity_preserved_by_ordered_delta_updates")
        is not True
        or not isinstance(candidate_carrier, Mapping)
        or candidate_carrier.get("mutation_count") != 36
        or candidate_carrier.get("initial_version") != 0
        or candidate_carrier.get("final_version") != 36
        or candidate_carrier.get(
            "identity_preserved_by_ordered_delta_updates"
        )
        is not True
        or exact_compilation.get("exact_parameter_inventory_parity") is not True
        or exact_compilation.get("source_model_parameter_count")
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or not isinstance(
            exact_compilation.get("source_independence_audit"), Mapping
        )
        or exact_compilation["source_independence_audit"].get("passed")
        is not True
        or not isinstance(
            candidate_compilation.get("source_independence_audit"), Mapping
        )
        or candidate_compilation["source_independence_audit"].get("passed")
        is not True
        or candidate_compilation.get("source_model_parameter_count")
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
    ):
        raise ValueError("joint report did not authenticate the complete graph")
    lexical_manifest = candidate_lexical
    if (
        configuration.get("arm_id") != DEVELOPMENT_ARM_ID
        or configuration.get("deterministic_seed") != DETERMINISTIC_SEED
        or configuration.get("device") != "cpu"
        or configuration.get("dtype") != "float32"
        or configuration.get("batch_size") != DEFAULT_BATCH_SIZE
        or configuration.get("lexical_rank") != DEFAULT_LEXICAL_RANK
        or configuration.get("attention_input_rank")
        != DEFAULT_ATTENTION_INPUT_RANK
        or configuration.get("attention_output_rank")
        != DEFAULT_ATTENTION_OUTPUT_RANK
        or configuration.get("mlp_generator_rank")
        != DEFAULT_MLP_GENERATOR_RANK
        or configuration.get(
            "maximum_attention_sources_per_target_chart"
        )
        != DEFAULT_MAXIMUM_ATTENTION_SOURCES_PER_TARGET_CHART
        or configuration.get("transport_relative_ridge")
        != DEFAULT_TRANSPORT_RELATIVE_RIDGE
        or configuration.get("lexical_fit_kind")
        != "deterministic_weight_svd_capacity_control"
        or configuration.get("attention_fit_kind")
        != "deterministic_weight_svd_capacity_control"
        or configuration.get("mlp_fit_kind")
        != "fisher_weighted_activation_charts_full_native_suffix"
        or not isinstance(lexical_manifest, Mapping)
        or lexical_manifest.get("rank") != configuration.get("lexical_rank")
        or lexical_manifest.get("is_exact_copied_matrix") is not False
        or lexical_manifest.get(
            "ingress_and_egress_share_the_same_storage"
        )
        is not True
    ):
        raise ValueError("joint fitted configuration binding is invalid")
    exact_metrics = metrics.get("exact_control")
    candidate_metrics = metrics.get("joint_candidate")
    if not isinstance(exact_metrics, Mapping) or not isinstance(
        candidate_metrics, Mapping
    ):
        raise ValueError("joint report metrics are incomplete")
    for metric_set in (exact_metrics, candidate_metrics):
        native_metric = metric_set.get("native")
        candidate_metric = metric_set.get("candidate")
        if not isinstance(native_metric, Mapping) or not isinstance(
            candidate_metric, Mapping
        ):
            raise ValueError("joint metric payload is incomplete")
        supervised = native_metric.get("supervised_tokens")
        compared = candidate_metric.get("compared_query_rows")
        numeric_names = (
            "nll_per_token",
            "delta_nll_per_token",
            "native_to_candidate_kl_per_query",
            "top1_agreement_to_native",
            "logit_nrmse",
            "maximum_absolute_logit_error",
        )
        if (
            type(supervised) is not int
            or supervised <= 0
            or candidate_metric.get("supervised_tokens") != supervised
            or type(compared) is not int
            or compared < supervised
            or not isinstance(native_metric.get("nll_per_token"), (int, float))
            or isinstance(native_metric.get("nll_per_token"), bool)
            or not math.isfinite(float(native_metric["nll_per_token"]))
            or float(native_metric["nll_per_token"]) < 0.0
            or any(
                not isinstance(candidate_metric.get(name), (int, float))
                or isinstance(candidate_metric.get(name), bool)
                or not math.isfinite(float(candidate_metric[name]))
                for name in numeric_names
            )
            or float(candidate_metric["nll_per_token"]) < 0.0
            or float(candidate_metric["native_to_candidate_kl_per_query"])
            < 0.0
            or not 0.0
            <= float(candidate_metric["top1_agreement_to_native"])
            <= 1.0
            or float(candidate_metric["logit_nrmse"]) < 0.0
            or float(candidate_metric["maximum_absolute_logit_error"]) < 0.0
            or float(candidate_metric["delta_nll_per_token"])
            != float(candidate_metric["nll_per_token"])
            - float(native_metric["nll_per_token"])
        ):
            raise ValueError("joint metric closure or range is invalid")
    _assert_exact_control_gate(exact_metrics)
    canonical_mutation_ids = canonical_event_ids[1:-2]
    expected_source_names = {
        "source.lexical_ingress",
        "source.terminal_rmsnorm",
        "source.lexical_egress",
        *(f"source.block.{index}" for index in range(PINNED_LAYER_COUNT)),
        *(f"source.attention.{index}" for index in range(PINNED_LAYER_COUNT)),
        *(f"source.mlp.{index}" for index in range(PINNED_LAYER_COUNT)),
    }
    expected_batch_count = (
        int(data["evaluation_example_count"]) + DEFAULT_BATCH_SIZE - 1
    ) // DEFAULT_BATCH_SIZE
    for condition in ("exact_control", "joint_candidate"):
        receipt = audit.get(condition)
        if not isinstance(receipt, Mapping):
            raise ValueError("joint execution audit is incomplete")
        events = receipt.get("event_receipts")
        native_calls = receipt.get("native_source_calls")
        compiled_calls = receipt.get("compiled_source_calls")
        if (
            receipt.get("batch_count") != expected_batch_count
            or receipt.get("zero_live_teacher_calls") is not True
            or receipt.get("compiled_source_call_total") != 0
            or receipt.get("event_count_per_execution") != 39
            or receipt.get("mutation_count_per_execution") != 36
            or receipt.get("attention_source_projection_call_total") != 0
            or receipt.get("attention_source_fallback_call_total") != 0
            or not isinstance(native_calls, Mapping)
            or set(native_calls) != expected_source_names
            or any(type(value) is not int or value <= 0 for value in native_calls.values())
            or not isinstance(compiled_calls, Mapping)
            or set(compiled_calls) != expected_source_names
            or any(type(value) is not int or value != 0 for value in compiled_calls.values())
            or sum(compiled_calls.values()) != 0
            or not isinstance(events, list)
            or len(events) != receipt.get("batch_count")
            or any(event != list(candidate_manifest["ordered_event_ids"])
                   for event in events)
        ):
            raise ValueError("joint source-call or event audit is invalid")
        operator_batches = receipt.get("operator_receipts")
        if not isinstance(operator_batches, list) or len(operator_batches) != (
            receipt.get("batch_count")
        ):
            raise ValueError("joint operator receipts are incomplete")
        for batch_receipts in operator_batches:
            if (
                not isinstance(batch_receipts, list)
                or len(batch_receipts) != 36
                or [
                    item.get("event_id")
                    for item in batch_receipts
                    if isinstance(item, Mapping)
                ]
                != canonical_mutation_ids
            ):
                raise ValueError("joint mutation receipt order is invalid")
            attention_receipts = [
                item
                for item in batch_receipts
                if isinstance(item, Mapping)
                and item.get("event_kind") == "attention_mutation"
            ]
            if len(attention_receipts) != 18 or any(
                item.get("operator_receipt_present") is not True
                or item.get("source_projection_call_count") != 0
                or item.get("source_fallback_call_count") != 0
                for item in attention_receipts
            ):
                raise ValueError("joint attention operator receipt is invalid")
            mlp_receipts = [
                item
                for item in batch_receipts
                if isinstance(item, Mapping)
                and item.get("event_kind") == "mlp_mutation"
            ]
            if len(mlp_receipts) != 18:
                raise ValueError("joint MLP operator receipt is invalid")
            for index, item in enumerate(batch_receipts, start=1):
                if not isinstance(item, Mapping):
                    raise ValueError("joint mutation receipt is not an object")
                event = canonical_schedule[index]
                if (
                    item.get("event_kind") != event.event_kind
                    or (
                        event.event_kind == "attention_mutation"
                        and item.get("attention_kind")
                        != (
                            "global_causal"
                            if event.attention_kind == "global"
                            else "sliding_causal"
                        )
                    )
                ):
                    raise ValueError("joint operator contract receipt drifted")
                if condition == "exact_control":
                    expected_in = 640
                    expected_out = 1024 if index % 2 else 640
                    expected_kind = "exact_full_rank" if index % 2 else None
                else:
                    expected_in = (
                        DEFAULT_ATTENTION_INPUT_RANK
                        if index % 2
                        else DEFAULT_MLP_GENERATOR_RANK
                    )
                    expected_out = (
                        DEFAULT_ATTENTION_OUTPUT_RANK
                        if index % 2
                        else DEFAULT_MLP_GENERATOR_RANK
                    )
                    expected_kind = (
                        "reduced_coordinates" if index % 2 else None
                    )
                if (
                    item.get("incoming_generator_rank") != expected_in
                    or item.get("emitted_generator_rank") != expected_out
                    or item.get("generator_kind") != expected_kind
                ):
                    raise ValueError("joint exact/candidate rank receipt drifted")
        if receipt.get("attention_operator_receipt_count") != 18 * int(
            receipt["batch_count"]
        ):
            raise ValueError("joint attention receipt count did not close")
        if condition == "exact_control":
            if (
                receipt.get("peak_live_transport_packets") != 0
                or receipt.get("peak_live_transport_packet_bytes") != 0
            ):
                raise ValueError("exact control unexpectedly carried transport")
        elif (
            type(receipt.get("peak_live_transport_packets")) is not int
            or int(receipt["peak_live_transport_packets"]) <= 0
            or type(receipt.get("peak_live_transport_packet_bytes")) is not int
            or int(receipt["peak_live_transport_packet_bytes"]) <= 0
        ):
            raise ValueError("joint candidate transport was not traversed")
    candidate_audit = audit["joint_candidate"]
    assert isinstance(candidate_audit, Mapping)
    before_model = audit.get("source_model_fingerprint_before")
    after_model = audit.get("source_model_fingerprint_after")
    before_execution = audit.get("source_execution_fingerprint_before")
    after_execution = audit.get("source_execution_fingerprint_after")
    if (
        audit.get("source_unchanged") is not True
        or not isinstance(before_model, str)
        or len(before_model) != 64
        or before_model != after_model
        or not isinstance(before_execution, str)
        or len(before_execution) != 64
        or before_execution != after_execution
    ):
        raise ValueError("joint source restoration receipt is invalid")
    operator_batches = candidate_audit.get("operator_receipts")
    if not isinstance(operator_batches, list) or len(operator_batches) != (
        candidate_audit.get("batch_count")
    ):
        raise ValueError("joint candidate operator receipts are incomplete")
    for batch_receipts in operator_batches:
        if not isinstance(batch_receipts, list) or len(batch_receipts) != 36:
            raise ValueError("joint candidate mutation receipts are incomplete")
        attention_receipts = [
            item
            for item in batch_receipts
            if isinstance(item, Mapping)
            and item.get("event_kind") == "attention_mutation"
        ]
        mlp_receipts = [
            item
            for item in batch_receipts
            if isinstance(item, Mapping)
            and item.get("event_kind") == "mlp_mutation"
        ]
        if (
            len(attention_receipts) != 18
            or len(mlp_receipts) != 18
            or any(
                item.get("generator_kind") != "reduced_coordinates"
                or item.get("operator_receipt_present") is not True
                or item.get("incoming_generator_rank")
                != configuration.get("attention_input_rank")
                or item.get("emitted_generator_rank")
                != configuration.get("attention_output_rank")
                or item.get("source_projection_call_count") != 0
                or item.get("source_fallback_call_count") != 0
                for item in attention_receipts
            )
            or any(
                item.get("incoming_generator_rank")
                != configuration.get("mlp_generator_rank")
                or item.get("emitted_generator_rank")
                != configuration.get("mlp_generator_rank")
                for item in mlp_receipts
            )
        ):
            raise ValueError("joint generator-rank execution binding is invalid")
    transports = candidate_manifest.get("transports")
    if not isinstance(transports, (list, tuple)):
        raise ValueError("joint transport catalog is missing")
    mutation_by_id = {
        item.event_id: item
        for item in canonical_schedule
        if item.event_kind in ("attention_mutation", "mlp_mutation")
    }
    cross_block = 0
    cross_family = 0
    cross_both = 0
    transport_ids: set[str] = set()
    transport_coordinates: set[tuple[str, int, str, int]] = set()
    for edge in transports:
        if not isinstance(edge, Mapping):
            raise ValueError("joint transport receipt is invalid")
        source_event = mutation_by_id.get(edge.get("source_event_id"))
        target_event = mutation_by_id.get(edge.get("target_event_id"))
        transport_id = edge.get("transport_id")
        source_chart = edge.get("source_chart_index")
        target_chart = edge.get("target_chart_index")
        if (
            source_event is None
            or target_event is None
            or source_event.event_index >= target_event.event_index
            or edge.get("strictly_forward") is not True
            or source_event.event_kind != "attention_mutation"
            or target_event.event_kind != "mlp_mutation"
            or source_event.block_ordinal == target_event.block_ordinal
            or edge.get("token_relation") != "same_token"
            or not isinstance(transport_id, str)
            or not transport_id
            or type(source_chart) is not int
            or source_chart != 0
            or type(target_chart) is not int
            or not 0 <= target_chart < 8
            or edge.get("source_generator_rank")
            != DEFAULT_ATTENTION_OUTPUT_RANK
            or edge.get("target_generator_rank") != DEFAULT_MLP_GENERATOR_RANK
            or edge.get("stored_scalar_count")
            != DEFAULT_ATTENTION_OUTPUT_RANK * DEFAULT_MLP_GENERATOR_RANK
        ):
            raise ValueError("joint transport topology is invalid")
        coordinate = (
            str(edge["source_event_id"]),
            source_chart,
            str(edge["target_event_id"]),
            target_chart,
        )
        if transport_id in transport_ids or coordinate in transport_coordinates:
            raise ValueError("joint transport catalog contains duplicates")
        transport_ids.add(transport_id)
        transport_coordinates.add(coordinate)
        different_block = source_event.block_ordinal != target_event.block_ordinal
        different_family = (
            source_event.component_family != target_event.component_family
        )
        cross_block += int(different_block)
        cross_family += int(different_family)
        cross_both += int(different_block and different_family)
    if (
        len(transports) != candidate_manifest.get("transport_edge_count")
        or cross_block != candidate_manifest.get("cross_block_transport_count")
        or cross_family
        != candidate_manifest.get("cross_component_family_transport_count")
        or cross_both
        != candidate_manifest.get(
            "cross_block_cross_component_transport_count"
        )
        or cross_both <= 0
    ):
        raise ValueError("joint transport counts do not close")
    candidate_resources = resources.get("joint_candidate")
    exact_resources = resources.get("exact_control")
    if not isinstance(candidate_resources, Mapping):
        raise ValueError("joint candidate resources are missing")
    if (
        not isinstance(exact_resources, Mapping)
        or exact_resources.get("source_unique_parameter_count")
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or exact_resources.get("candidate_deployed_scalar_count")
        != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or exact_resources.get("parameter_difference") != 0
        or exact_resources.get("parameter_reduction_fraction") != 0.0
        or exact_resources.get("executor_inventory") != exact_inventory
    ):
        raise ValueError("joint exact-control resource ledger is invalid")
    source = candidate_resources.get("source_unique_parameter_count")
    candidate = candidate_resources.get("candidate_deployed_scalar_count")
    if (
        source != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT
        or type(candidate) is not int
        or candidate <= 0
        or candidate_resources.get("parameter_difference")
        != source - candidate
        or candidate_resources.get("parameter_reduction_fraction")
        != (source - candidate) / source
        or candidate_resources.get("projected_source_float32_bytes")
        != 4 * source
        or candidate_resources.get("projected_candidate_float32_bytes")
        != 4 * candidate
        or candidate_resources.get("projected_byte_reduction_fraction")
        != (source - candidate) / source
        or candidate_resources.get("lexical_rank") != DEFAULT_LEXICAL_RANK
        or candidate_resources.get("attention_input_rank")
        != DEFAULT_ATTENTION_INPUT_RANK
        or candidate_resources.get("attention_output_rank")
        != DEFAULT_ATTENTION_OUTPUT_RANK
        or candidate_resources.get("mlp_generator_rank")
        != DEFAULT_MLP_GENERATOR_RANK
        or candidate_resources.get("lexical_factor_counted_once") is not True
        or candidate_resources.get("source_independent_in_memory_executor")
        is not True
        or candidate_resources.get("serialized_candidate_artifact_written")
        is not False
        or candidate_resources.get("serialized_byte_reduction_authoritative")
        is not False
        or candidate_resources.get("whole_model_mac_ledger_closed") is not False
        or candidate_resources.get("whole_model_mac_reduction_claimed") is not False
        or candidate_resources.get("latency_reduction_claimed") is not False
        or candidate_resources.get("executor_inventory")
        != candidate_manifest.get("resource_inventory")
        or not isinstance(candidate_resources.get("executor_inventory"), Mapping)
        or candidate_resources["executor_inventory"].get("inventory_closes")
        is not True
        or candidate_resources["executor_inventory"].get(
            "candidate_deployed_scalars"
        )
        != candidate
    ):
        raise ValueError("joint resource ledger is invalid")
    recomputed = _qualification(
        candidate_metrics,
        candidate_resources,
        exact_control_passed=True,
        cross_block_edge_count=int(
            candidate_manifest["cross_block_cross_component_transport_count"]
        ),
    )
    if dict(qualification) != recomputed:
        raise ValueError("joint qualification was not recomputed")
    without_hash = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    if report.get("report_sha256") != _report_sha256(without_hash):
        raise ValueError("joint report SHA-256 is invalid")


def load_gemma3_joint_cross_block_whole_model_report(
    source: Path | str,
) -> dict[str, object]:
    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("joint experiment report is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("joint experiment report must be an object")
    _validate_report_semantics(value)
    return copy.deepcopy(dict(value))


def run_gemma3_joint_cross_block_whole_model_experiment(
    *,
    revision: str = DEFAULT_REVISION,
    output: Path | str = DEFAULT_OUTPUT,
    protocol_path: Path | str = DEFAULT_PROTOCOL_OUTPUT,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    batch_size: int = DEFAULT_BATCH_SIZE,
    lexical_rank: int = DEFAULT_LEXICAL_RANK,
    attention_input_rank: int = DEFAULT_ATTENTION_INPUT_RANK,
    attention_output_rank: int = DEFAULT_ATTENTION_OUTPUT_RANK,
    mlp_generator_rank: int = DEFAULT_MLP_GENERATOR_RANK,
    maximum_attention_sources_per_target_chart: int = (
        DEFAULT_MAXIMUM_ATTENTION_SOURCES_PER_TARGET_CHART
    ),
    transport_relative_ridge: float = DEFAULT_TRANSPORT_RELATIVE_RIDGE,
    use_ewc: bool = False,
    fuse_mlp_modes: bool = False,
) -> dict[str, object]:
    """Fit and evaluate the joint standalone development candidate."""

    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("joint experiment output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a joint experiment report")
    if revision != PINNED_MODEL_REVISION or model_id != PINNED_MODEL_ID:
        raise ValueError("joint experiment requires the pinned Gemma checkpoint")
    if device_name != "cpu" or dtype != "float32":
        raise ValueError("joint experiment requires deterministic CPU float32")
    integer_options = {
        "batch_size": batch_size,
        "lexical_rank": lexical_rank,
        "attention_input_rank": attention_input_rank,
        "attention_output_rank": attention_output_rank,
        "mlp_generator_rank": mlp_generator_rank,
        "maximum_attention_sources_per_target_chart": (
            maximum_attention_sources_per_target_chart
        ),
    }
    if any(type(value) is not int or value <= 0 for value in integer_options.values()):
        raise ValueError("joint experiment integer options must be positive")
    frozen_numerics = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "lexical_rank": DEFAULT_LEXICAL_RANK,
        "attention_input_rank": DEFAULT_ATTENTION_INPUT_RANK,
        "attention_output_rank": DEFAULT_ATTENTION_OUTPUT_RANK,
        "mlp_generator_rank": DEFAULT_MLP_GENERATOR_RANK,
        "maximum_attention_sources_per_target_chart": (
            DEFAULT_MAXIMUM_ATTENTION_SOURCES_PER_TARGET_CHART
        ),
    }
    if integer_options != frozen_numerics:
        raise ValueError("joint experiment v1 numerics are frozen")
    if (
        not isinstance(transport_relative_ridge, (int, float))
        or isinstance(transport_relative_ridge, bool)
        or not math.isfinite(float(transport_relative_ridge))
        or float(transport_relative_ridge) <= 0
    ):
        raise ValueError("joint transport ridge must be finite and positive")
    if float(transport_relative_ridge) != DEFAULT_TRANSPORT_RELATIVE_RIDGE:
        raise ValueError("joint experiment v1 transport ridge is frozen")

    protocol = _protocol_receipt(protocol_path)
    fit_prompts, evaluation_prompts, split = _load_prompt_split(Path(panel_path))
    if (
        not split.get("prompt_disjoint")
        or split.get("family_disjoint") is not False
        or not fit_prompts
        or not evaluation_prompts
    ):
        raise RuntimeError("joint development split contract is unavailable")
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("load pinned local Gemma teacher")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    source_count = sum(parameter.numel() for parameter in model.parameters())
    if source_count != PINNED_SOURCE_UNIQUE_PARAMETER_COUNT:
        raise RuntimeError("pinned source parameter inventory drifted")
    before_model = adapter.model_fingerprint()
    before_execution = adapter.execution_fingerprint()
    guard = _FrozenModelTensorGuard(model)
    fit_ids = _ordered_example_ids(Path(panel_path), fit_prompts)
    evaluation_ids = _ordered_example_ids(Path(panel_path), evaluation_prompts)
    fit_keyed = _tokenize_keyed_batches(
        tokenizer,
        fit_prompts,
        fit_ids,
        batch_size=batch_size,
        device=device,
    )
    evaluation_keyed = _tokenize_keyed_batches(
        tokenizer,
        evaluation_prompts,
        evaluation_ids,
        batch_size=batch_size,
        device=device,
    )
    fit_batches = tuple(item.model_inputs for item in fit_keyed)
    fit_row_keys = tuple(item.row_keys for item in fit_keyed)
    evaluation_batches = tuple(item.model_inputs for item in evaluation_keyed)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DETERMINISTIC_SEED)
        torch.use_deterministic_algorithms(True)
        try:
            _progress("compile exact tied lexical + attention + MLP control")
            exact = _compile_exact_control(adapter)
            exact_manifest = exact.executor.architecture_manifest()
            exact_resources = _resource_report(
                exact,
                lexical_rank=PINNED_HIDDEN_WIDTH,
                attention_input_rank=PINNED_HIDDEN_WIDTH,
                attention_output_rank=1024,
                mlp_generator_rank=PINNED_HIDDEN_WIDTH,
            )
            _progress("evaluate and gate exact 39-event control")
            exact_metrics, exact_audit = _evaluate_joint_executor(
                adapter=adapter,
                executor=exact.executor,
                batches=evaluation_batches,
            )
            _assert_exact_control_gate(exact_metrics)

            _progress("fit tied lexical, reduced attention, MLP charts, and edges")
            candidate_bundle = _compile_joint_executors(
                adapter=adapter,
                exact_control=exact,
                fit_batches=fit_batches,
                fit_batch_row_keys=fit_row_keys,
                lexical_rank=lexical_rank,
                attention_input_rank=attention_input_rank,
                attention_output_rank=attention_output_rank,
                mlp_generator_rank=mlp_generator_rank,
                maximum_attention_sources_per_target_chart=(
                    maximum_attention_sources_per_target_chart
                ),
                transport_relative_ridge=float(transport_relative_ridge),
                use_ewc=use_ewc,
                fuse_mlp_modes=fuse_mlp_modes,
            )
            candidate = candidate_bundle.compilation
            candidate_manifest = candidate.executor.architecture_manifest()
            candidate_resources = _resource_report(
                candidate,
                lexical_rank=lexical_rank,
                attention_input_rank=attention_input_rank,
                attention_output_rank=attention_output_rank,
                mlp_generator_rank=mlp_generator_rank,
            )
            _progress("evaluate standalone joint cross-block candidate")
            candidate_metrics, candidate_audit = _evaluate_joint_executor(
                adapter=adapter,
                executor=candidate.executor,
                batches=evaluation_batches,
            )
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

    guard.assert_unchanged()
    after_model = adapter.model_fingerprint()
    after_execution = adapter.execution_fingerprint()
    if after_model != before_model or after_execution != before_execution:
        raise RuntimeError("source Gemma changed during joint experiment")
    cross_edges = int(
        candidate_manifest["cross_block_cross_component_transport_count"]
    )
    qualification = _qualification(
        candidate_metrics,
        candidate_resources,
        exact_control_passed=True,
        cross_block_edge_count=cross_edges,
    )
    report: dict[str, object] = {
        "schema": GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_SCHEMA,
        "format_version": GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_FORMAT_VERSION,
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "protocol": protocol,
        "scientific_status": {
            "development_only": True,
            "prompt_disjoint_evaluation": True,
            "family_disjoint_evaluation": False,
            "heldout_confirmation": False,
            "exact_39_event_control_passed": True,
            "tied_lexical_embedding_and_head_compiled": True,
            "all_18_attention_mutations_compiled": True,
            "all_18_mlp_mutations_compiled": True,
            "single_residual_carrier_compiled": True,
            "cross_block_cross_component_transports_compiled": True,
            "unrestricted_primary_fully_executed": False,
            "attention_basis_is_weight_svd_capacity_control": True,
            "mlp_charts_are_fisher_activation_fitted": True,
            "candidate_fit_trajectory": "one_shot_native_development_v1",
            "prefill_only": True,
            "cached_decode_tested": False,
            "compression_success_claimed": False,
            "compression_claim_authorized": False,
            "latency_or_kernel_speed_claimed": False,
        },
        "model": _model_provenance(
            model,
            model_id=model_id,
            requested_revision=revision,
        ),
        "data": split,
        "configuration": {
            "arm_id": DEVELOPMENT_ARM_ID,
            "deterministic_seed": DETERMINISTIC_SEED,
            "device": device_name,
            "dtype": dtype,
            "batch_size": batch_size,
            "lexical_rank": lexical_rank,
            "attention_input_rank": attention_input_rank,
            "attention_output_rank": attention_output_rank,
            "mlp_generator_rank": mlp_generator_rank,
            "maximum_attention_sources_per_target_chart": (
                maximum_attention_sources_per_target_chart
            ),
            "transport_relative_ridge": float(transport_relative_ridge),
            "lexical_fit_kind": "deterministic_weight_svd_capacity_control",
            "attention_fit_kind": "deterministic_weight_svd_capacity_control",
            "mlp_fit_kind": (
                "fisher_weighted_activation_charts_full_native_suffix"
            ),
            "transport_fit_kind": "fisher_weighted_partial_residual_gain",
        },
        "compilation": {
            "exact_control": exact.metadata(),
            "joint_candidate": candidate.metadata(),
            "exact_control_manifest": exact_manifest,
            "candidate_manifest": candidate_manifest,
            "fit_row_identity_sha256": (
                candidate_bundle.fit.fit_row_identity_sha256
            ),
            "fit_layer_metrics": candidate_bundle.fit.layer_metrics,
        },
        "metrics": {
            "exact_control": exact_metrics,
            "joint_candidate": candidate_metrics,
        },
        "execution_audit": {
            "exact_control": exact_audit,
            "joint_candidate": candidate_audit,
            "source_model_fingerprint_before": before_model,
            "source_model_fingerprint_after": after_model,
            "source_execution_fingerprint_before": before_execution,
            "source_execution_fingerprint_after": after_execution,
            "source_unchanged": True,
        },
        "resources": {
            "exact_control": exact_resources,
            "joint_candidate": candidate_resources,
        },
        "qualification": qualification,
    }
    report["report_sha256"] = _report_sha256(report)
    _validate_report_semantics(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        raise FileExistsError(
            "refusing to overwrite a joint experiment report"
        ) from None
    _progress(f"wrote {destination}")
    gc.collect()
    return copy.deepcopy(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone joint lexical, attention, MLP, residual, "
            "and cross-block Gemma development experiment."
        )
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_OUTPUT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lexical-rank", type=int, default=DEFAULT_LEXICAL_RANK)
    parser.add_argument(
        "--attention-input-rank",
        type=int,
        default=DEFAULT_ATTENTION_INPUT_RANK,
    )
    parser.add_argument(
        "--attention-output-rank",
        type=int,
        default=DEFAULT_ATTENTION_OUTPUT_RANK,
    )
    parser.add_argument(
        "--mlp-generator-rank",
        type=int,
        default=DEFAULT_MLP_GENERATOR_RANK,
    )
    parser.add_argument(
        "--maximum-attention-sources-per-target-chart",
        type=int,
        default=DEFAULT_MAXIMUM_ATTENTION_SOURCES_PER_TARGET_CHART,
    )
    parser.add_argument(
        "--transport-relative-ridge",
        type=float,
        default=DEFAULT_TRANSPORT_RELATIVE_RIDGE,
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="use Pullback-Fisher + EWC sequential trajectory fitting across all 39 events",
    )
    parser.add_argument(
        "--fuse-mlp-modes",
        action="store_true",
        help="fuse MLP chart modes into the 640-D post-norm residual space (39.1%% whole-model reduction)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_gemma3_joint_cross_block_whole_model_experiment(
        revision=args.revision,
        output=args.output,
        protocol_path=args.protocol,
        panel_path=args.panel,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device_name=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        lexical_rank=args.lexical_rank,
        attention_input_rank=args.attention_input_rank,
        attention_output_rank=args.attention_output_rank,
        mlp_generator_rank=args.mlp_generator_rank,
        maximum_attention_sources_per_target_chart=(
            args.maximum_attention_sources_per_target_chart
        ),
        transport_relative_ridge=args.transport_relative_ridge,
        use_ewc=args.ewc,
        fuse_mlp_modes=args.fuse_mlp_modes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_FORMAT_VERSION",
    "GEMMA3_JOINT_CROSS_BLOCK_EXPERIMENT_SCHEMA",
    "load_gemma3_joint_cross_block_whole_model_report",
    "run_gemma3_joint_cross_block_whole_model_experiment",
]
