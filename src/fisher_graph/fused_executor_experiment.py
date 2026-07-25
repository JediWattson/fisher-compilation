"""Build, lock, validate, and benchmark the fused two-layer modal runtime."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from .associative import (
    AssociativeRecallSplit,
    AssociativeRecallTaskConfig,
    associative_recall_answer_logits,
    associative_recall_metrics_from_logits,
    build_associative_recall_splits,
)
from .config import TransformerConfig
from .compiler import (
    manifest_from_legacy_runtime,
    save_runtime_manifest,
)
from .fused_benchmark import benchmark_batch_sizes
from .fused_executor import (
    FusedToyTransformer,
    FusedTwoLayerModalStack,
    LazyFusedTwoLayerModalStack,
    PackedTriangularFusedTwoLayerModalStack,
    load_fused_modal_stack,
    load_lazy_fused_modal_stack,
    save_fused_modal_stack,
    save_lazy_fused_modal_stack,
)
from .modal_artifacts import (
    fused_executor_artifact_paths,
    modal_completion_artifact_paths,
    modal_executor_artifact_paths,
)
from .modal_completion import (
    PositionConditionedCompletedModalGraphExecutor,
)
from .modal_composition_experiment import (
    _layer_accounting,
    _module_state_sha256,
    _sha256,
    _tensor_sha256,
    _validate_layer_artifacts,
)
from .modal_executor import (
    ModalExecutorConfig,
    PositionConditionedModalGraphExecutor,
)
from .modal_executor_experiment import _estimated_block_multiplies
from .model import ToyTransformer


_VALIDATION_GATE: dict[str, float] = {
    "maximum_absolute_nll_delta": 1e-6,
    "maximum_mean_answer_kl": 1e-6,
    "maximum_answer_logit_difference": 5e-4,
}

_FUSED_REPORT_FORMAT_VERSION = 3

_EXPECTED_ARITHMETIC = {
    "original_two_block_estimated_multiplies": 139_264,
    "unfused_modal_logical_multiplies": 72_384,
    "fused_dense_executed_multiplies": 49_152,
    "fused_triangular_nonzero_multiplies": 30_336,
}

_EXPECTED_LAZY_STORAGE = {
    "fast_stack_resident_tensor_bytes": 199_808,
    "sidecar_resident_tensor_bytes": 203_648,
    "model_shell_tensor_bytes": 6_144,
    "default_full_runtime_resident_tensor_bytes": 205_952,
    "loaded_full_runtime_resident_tensor_bytes": 409_600,
}


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _new_teacher(
    model_config: TransformerConfig,
    state_dict: Mapping[str, Tensor],
) -> ToyTransformer:
    model = ToyTransformer(model_config)
    model.load_state_dict(state_dict)
    _freeze(model)
    return model


def _build_unfused_model(
    *,
    model_config: TransformerConfig,
    state_dict: Mapping[str, Tensor],
    first: PositionConditionedCompletedModalGraphExecutor,
    second: PositionConditionedCompletedModalGraphExecutor,
) -> ToyTransformer:
    model = _new_teacher(model_config, state_dict)
    model.replace_layer(0, first)
    model.replace_layer(1, second)
    _freeze(model)
    return model


def _answer_kl(reference_logits: Tensor, candidate_logits: Tensor) -> Tensor:
    reference = reference_logits.to(torch.float64)
    candidate = candidate_logits.to(torch.float64)
    reference_probabilities = reference.softmax(dim=-1)
    return (
        reference_probabilities
        * (
            reference.log_softmax(dim=-1)
            - candidate.log_softmax(dim=-1)
        )
    ).sum(dim=-1)


def _compare_fused_to_unfused(
    *,
    split: AssociativeRecallSplit,
    unfused_logits: Tensor,
    fused_logits: Tensor,
) -> dict[str, object]:
    """Summarize the behavioral and numerical runtime-equivalence contract."""

    unfused_metrics = associative_recall_metrics_from_logits(
        split,
        unfused_logits,
    )
    fused_metrics = associative_recall_metrics_from_logits(
        split,
        fused_logits,
    )
    unfused_predictions = unfused_logits.argmax(dim=-1)
    fused_predictions = fused_logits.argmax(dim=-1)
    per_answer_kl = _answer_kl(unfused_logits, fused_logits)
    return {
        "answer_accuracy_exactly_equal": (
            fused_metrics.answer_accuracy
            == unfused_metrics.answer_accuracy
        ),
        "paired_context_accuracy_exactly_equal": (
            fused_metrics.paired_context_accuracy
            == unfused_metrics.paired_context_accuracy
        ),
        "argmax_predictions_exactly_equal": bool(
            torch.equal(fused_predictions, unfused_predictions)
        ),
        "unfused_argmax_sha256": _tensor_sha256(
            unfused_predictions,
        ),
        "fused_argmax_sha256": _tensor_sha256(fused_predictions),
        "absolute_hard_nll_delta": abs(
            fused_metrics.hard_nll - unfused_metrics.hard_nll
        ),
        "mean_unfused_to_fused_answer_kl": (
            per_answer_kl.mean().item()
        ),
        "maximum_unfused_to_fused_answer_kl": (
            per_answer_kl.max().item()
        ),
        "maximum_answer_logit_difference": (
            (fused_logits - unfused_logits).abs().max().item()
        ),
    }


def _passes_fusion_gate(
    comparison: Mapping[str, object],
    gate: Mapping[str, float],
) -> bool:
    return bool(
        comparison["answer_accuracy_exactly_equal"]
        and comparison["paired_context_accuracy_exactly_equal"]
        and comparison["argmax_predictions_exactly_equal"]
        and float(comparison["absolute_hard_nll_delta"])
        <= gate["maximum_absolute_nll_delta"]
        and float(comparison["mean_unfused_to_fused_answer_kl"])
        <= gate["maximum_mean_answer_kl"]
        and float(comparison["maximum_answer_logit_difference"])
        <= gate["maximum_answer_logit_difference"]
    )


def _evaluate_systems(
    *,
    split: AssociativeRecallSplit,
    teacher: ToyTransformer,
    unfused: ToyTransformer,
    monolithic: FusedToyTransformer,
    lazy: FusedToyTransformer,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    logits = {
        "teacher": associative_recall_answer_logits(teacher, split),
        "unfused": associative_recall_answer_logits(unfused, split),
        "monolithic": associative_recall_answer_logits(
            monolithic,
            split,
        ),
        "lazy": associative_recall_answer_logits(lazy, split),
    }
    metrics = {
        name: asdict(
            associative_recall_metrics_from_logits(split, values)
        )
        for name, values in logits.items()
    }
    lazy_vs_monolithic = {
        "logits_bit_exact": bool(
            torch.equal(logits["lazy"], logits["monolithic"])
        ),
        "maximum_logit_difference": (
            (logits["lazy"] - logits["monolithic"]).abs().max().item()
        ),
        "monolithic_argmax_sha256": _tensor_sha256(
            logits["monolithic"].argmax(dim=-1)
        ),
        "lazy_argmax_sha256": _tensor_sha256(
            logits["lazy"].argmax(dim=-1)
        ),
    }
    return (
        {
            "systems": metrics,
            "monolithic_vs_unfused": _compare_fused_to_unfused(
                split=split,
                unfused_logits=logits["unfused"],
                fused_logits=logits["monolithic"],
            ),
            "lazy_vs_unfused": _compare_fused_to_unfused(
                split=split,
                unfused_logits=logits["unfused"],
                fused_logits=logits["lazy"],
            ),
            "lazy_vs_monolithic": lazy_vs_monolithic,
        },
        logits,
    )


def _state_storage(module: nn.Module) -> dict[str, int]:
    parameter_tensors = list(module.parameters())
    buffer_tensors = list(module.buffers())

    def elements(tensors: list[Tensor]) -> int:
        return sum(tensor.numel() for tensor in tensors)

    def size_bytes(tensors: list[Tensor]) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )

    trainable_tensors = [
        parameter
        for parameter in parameter_tensors
        if parameter.requires_grad
    ]
    return {
        "parameter_elements": elements(parameter_tensors),
        "parameter_bytes": size_bytes(parameter_tensors),
        "trainable_parameter_elements": elements(trainable_tensors),
        "trainable_parameter_bytes": size_bytes(trainable_tensors),
        "buffer_elements": elements(buffer_tensors),
        "buffer_bytes": size_bytes(buffer_tensors),
        "total_state_elements": (
            elements(parameter_tensors) + elements(buffer_tensors)
        ),
        "total_state_bytes": (
            size_bytes(parameter_tensors) + size_bytes(buffer_tensors)
        ),
    }


def _status_dict(
    stack: LazyFusedTwoLayerModalStack,
) -> dict[str, object]:
    return asdict(stack.instrumentation_status())


def _require_no_sidecar_activity(
    status: Mapping[str, object],
    *,
    phase: str,
) -> None:
    if (
        status["residency"] != "unloaded"
        or status["loaded"] is not False
        or int(status["load_attempts"]) != 0
        or int(status["successful_loads"]) != 0
        or int(status["failed_loads"]) != 0
        or int(status["instrumented_path_calls"]) != 0
        or int(status["resident_sidecar_tensor_bytes"]) != 0
        or int(status["sidecar_file_bytes_read"]) != 0
    ):
        raise RuntimeError(
            f"lazy fast runtime touched instrumentation during {phase}"
        )


def _lazy_storage_contract(
    *,
    lazy_model: FusedToyTransformer,
    lazy_fast_status: Mapping[str, object],
    loaded_status: Mapping[str, object],
) -> dict[str, int]:
    fast_stack_bytes = int(
        lazy_fast_status["resident_fast_tensor_bytes"]
    )
    sidecar_bytes = int(
        loaded_status["resident_sidecar_tensor_bytes"]
    )
    default_model_state = _state_storage(lazy_model)[
        "total_state_bytes"
    ]
    shell_bytes = default_model_state - fast_stack_bytes
    actual = {
        "fast_stack_resident_tensor_bytes": fast_stack_bytes,
        "sidecar_resident_tensor_bytes": sidecar_bytes,
        "model_shell_tensor_bytes": shell_bytes,
        "default_full_runtime_resident_tensor_bytes": (
            shell_bytes + fast_stack_bytes
        ),
        "loaded_full_runtime_resident_tensor_bytes": (
            shell_bytes + fast_stack_bytes + sidecar_bytes
        ),
    }
    if actual != _EXPECTED_LAZY_STORAGE:
        raise RuntimeError(
            f"lazy storage contract changed: {actual!r}"
        )
    return actual


def _lazy_benchmark_comparison(
    benchmark: list[dict[str, object]],
) -> dict[str, object]:
    per_batch: list[dict[str, object]] = []
    latency_ratios: list[float] = []
    for batch in benchmark:
        timings = batch["timings"]
        speedups = batch["speedup_ratios"]
        assert isinstance(timings, dict)
        assert isinstance(speedups, dict)
        monolithic = float(
            timings["monolithic"]["median_microseconds"]
        )
        lazy = float(timings["lazy"]["median_microseconds"])
        latency_ratio = lazy / monolithic
        latency_ratios.append(latency_ratio)
        per_batch.append(
            {
                "batch_size": int(batch["batch_size"]),
                "lazy_to_monolithic_latency_ratio": latency_ratio,
                "lazy_vs_monolithic_speedup": float(
                    speedups["lazy_vs_monolithic"]
                ),
                "lazy_latency_regression_fraction": (
                    latency_ratio - 1.0
                ),
            }
        )
    geometric_mean_ratio = math.exp(
        sum(math.log(value) for value in latency_ratios)
        / len(latency_ratios)
    )
    return {
        "per_batch": per_batch,
        "geometric_mean_lazy_to_monolithic_latency_ratio": (
            geometric_mean_ratio
        ),
        "geometric_mean_lazy_latency_regression_fraction": (
            geometric_mean_ratio - 1.0
        ),
        "maximum_lazy_to_monolithic_latency_ratio": max(
            latency_ratios
        ),
        "maximum_lazy_latency_regression_fraction": (
            max(latency_ratios) - 1.0
        ),
        "hard_latency_gate_applied": False,
        "interpretation": (
            "positive regression fractions mean the lazy fast wrapper was "
            "slower; negative values mean it was faster"
        ),
    }


def _triangular_benchmark_comparison(
    benchmark: list[dict[str, object]],
) -> dict[str, object]:
    """Derive triangular latency ratios from one fair rotating cohort."""

    if not benchmark:
        raise ValueError("triangular benchmark cannot be empty")
    per_batch: list[dict[str, object]] = []
    lazy_speedups: list[float] = []
    unfused_speedups: list[float] = []
    for batch in benchmark:
        timings = batch["timings"]
        speedups = batch["speedup_ratios"]
        assert isinstance(timings, dict)
        assert isinstance(speedups, dict)
        lazy = float(timings["lazy"]["median_microseconds"])
        unfused = float(timings["unfused"]["median_microseconds"])
        triangular = float(
            timings["triangular"]["median_microseconds"]
        )
        triangular_vs_lazy = float(
            speedups["triangular_vs_lazy"]
        )
        triangular_vs_unfused = float(
            speedups["triangular_vs_unfused"]
        )
        if min(
            lazy,
            unfused,
            triangular,
            triangular_vs_lazy,
            triangular_vs_unfused,
        ) <= 0:
            raise ValueError(
                "triangular benchmark timings and ratios must be positive"
            )
        expected_lazy = lazy / triangular
        expected_unfused = unfused / triangular
        if not math.isclose(
            triangular_vs_lazy,
            expected_lazy,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "triangular_vs_lazy is inconsistent with the medians"
            )
        if not math.isclose(
            triangular_vs_unfused,
            expected_unfused,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "triangular_vs_unfused is inconsistent with the medians"
            )
        lazy_speedups.append(triangular_vs_lazy)
        unfused_speedups.append(triangular_vs_unfused)
        per_batch.append(
            {
                "batch_size": int(batch["batch_size"]),
                "triangular_vs_lazy_speedup": triangular_vs_lazy,
                "triangular_to_lazy_latency_ratio": (
                    triangular / lazy
                ),
                "triangular_vs_unfused_speedup": (
                    triangular_vs_unfused
                ),
                "triangular_to_unfused_latency_ratio": (
                    triangular / unfused
                ),
            }
        )

    def geometric_mean(values: list[float]) -> float:
        return math.exp(
            sum(math.log(value) for value in values) / len(values)
        )

    return {
        "per_batch": per_batch,
        "geometric_mean_triangular_vs_lazy_speedup": geometric_mean(
            lazy_speedups
        ),
        "geometric_mean_triangular_vs_unfused_speedup": (
            geometric_mean(unfused_speedups)
        ),
        "hard_latency_gate_applied": False,
        "interpretation": (
            "speedups above one mean the packed triangular runtime was "
            "faster; latency ratios below one mean it was faster"
        ),
    }


def _fused_arithmetic(
    *,
    stack: FusedTwoLayerModalStack,
    model_config: TransformerConfig,
    first_base: PositionConditionedModalGraphExecutor,
    first_completed: PositionConditionedCompletedModalGraphExecutor,
    first_config: ModalExecutorConfig,
    second_base: PositionConditionedModalGraphExecutor,
    second_completed: PositionConditionedCompletedModalGraphExecutor,
    second_config: ModalExecutorConfig,
) -> dict[str, object]:
    """Count the dense fast path and its causal triangular nonzero subset."""

    if not stack.uses_cross_layer_bypass:
        raise ValueError(
            "the locked fused build requires the exact cross-layer bypass"
        )
    sequence_length = stack.config.first.sequence_length
    width = stack.config.first.width
    first_routing = stack.config.first.routing_width
    second_routing = stack.config.second.routing_width
    causal_pairs = sequence_length * (sequence_length + 1) // 2

    dense_components = {
        "input_to_layer_0_hidden": (
            sequence_length
            * sequence_length
            * width
            * first_routing
        ),
        "layer_0_hidden_to_layer_1_hidden": (
            sequence_length
            * sequence_length
            * first_routing
            * second_routing
        ),
        "layer_1_hidden_to_residual_output": (
            sequence_length * second_routing * width
        ),
    }
    triangular_components = {
        "input_to_layer_0_hidden": (
            causal_pairs * width * first_routing
        ),
        "layer_0_hidden_to_layer_1_hidden": (
            causal_pairs * first_routing * second_routing
        ),
        "layer_1_hidden_to_residual_output": (
            sequence_length * second_routing * width
        ),
    }
    layer0 = _layer_accounting(
        executor=first_base,
        completed=first_completed,
        config=first_config,
        sequence_length=sequence_length,
        width=width,
    )
    layer1 = _layer_accounting(
        executor=second_base,
        completed=second_completed,
        config=second_config,
        sequence_length=sequence_length,
        width=width,
    )
    original = 2 * _estimated_block_multiplies(
        model_config,
        sequence_length=sequence_length,
    )
    modal_logical = (
        int(layer0["completed_estimated_multiplies"])
        + int(layer1["completed_estimated_multiplies"])
    )
    dense = sum(dense_components.values())
    triangular = sum(triangular_components.values())
    actual = {
        "original_two_block_estimated_multiplies": original,
        "unfused_modal_logical_multiplies": modal_logical,
        "fused_dense_executed_multiplies": dense,
        "fused_triangular_nonzero_multiplies": triangular,
    }
    if actual != _EXPECTED_ARITHMETIC:
        raise RuntimeError(
            f"fused arithmetic contract changed: {actual!r}"
        )
    return {
        **actual,
        "dense_components": dense_components,
        "triangular_components": triangular_components,
        "fused_dense_vs_original_ratio": dense / original,
        "fused_triangular_vs_original_ratio": triangular / original,
        "fused_dense_vs_unfused_modal_ratio": dense / modal_logical,
        "fused_triangular_vs_unfused_modal_ratio": (
            triangular / modal_logical
        ),
        "counting_scope": (
            "two replaced blocks only; scalar multiplies; bias, GELU, "
            "embedding, final norm, and vocabulary head excluded"
        ),
        "dense_interpretation": (
            "current einsum fast path includes structural zeros in its "
            "dense kernels"
        ),
        "triangular_interpretation": (
            "packed causal-pair PyTorch reference executes only the "
            "lower-triangular position pairs; wall-clock behavior is "
            "reported in its separate benchmark"
        ),
    }


def _trace_contract(
    *,
    validation_inputs: Tensor,
    unfused: ToyTransformer,
    lazy: FusedToyTransformer,
    lazy_stack: LazyFusedTwoLayerModalStack,
    expected_sidecar_file_bytes: int,
) -> dict[str, object]:
    status_before = _status_dict(lazy_stack)
    _require_no_sidecar_activity(
        status_before,
        phase="instrumentation setup",
    )
    with torch.inference_mode():
        unfused_output = unfused(
            validation_inputs,
            capture_activations=True,
            retain_activation_gradients=False,
        )
        lazy_trace_output = lazy(
            validation_inputs,
            capture_activations=True,
            retain_activation_gradients=False,
        )
    status_after_first_capture = _status_dict(lazy_stack)
    with torch.inference_mode():
        identity_intervention_output = lazy(
            validation_inputs,
            activation_interventions={
                "layer.0.modal.hidden": lambda values: values,
            },
        )
    status_after_reused_intervention = _status_dict(lazy_stack)
    with torch.inference_mode():
        lazy_fast_output = lazy(validation_inputs)
    status_after_fast_reuse = _status_dict(lazy_stack)
    eviction_returned_true = lazy_stack.evict_instrumentation()
    status_after_explicit_eviction = _status_dict(lazy_stack)

    if (
        status_after_first_capture["residency"] != "loaded"
        or status_after_first_capture["loaded"] is not True
        or int(status_after_first_capture["load_attempts"]) != 1
        or int(status_after_first_capture["successful_loads"]) != 1
        or int(status_after_first_capture["failed_loads"]) != 0
        or int(
            status_after_first_capture["instrumented_path_calls"]
        )
        != 1
        or int(status_after_first_capture["cache_hits"]) != 0
        or int(
            status_after_first_capture[
                "derived_kernel_verifications"
            ]
        )
        != 1
        or int(
            status_after_first_capture[
                "resident_sidecar_tensor_bytes"
            ]
        )
        != _EXPECTED_LAZY_STORAGE["sidecar_resident_tensor_bytes"]
        or int(
            status_after_first_capture["sidecar_file_bytes_read"]
        )
        != expected_sidecar_file_bytes
    ):
        raise RuntimeError(
            "first lazy instrumentation capture did not load exactly once"
        )
    if (
        int(status_after_reused_intervention["load_attempts"]) != 1
        or int(status_after_reused_intervention["successful_loads"]) != 1
        or int(status_after_reused_intervention["cache_hits"]) != 1
        or int(
            status_after_reused_intervention[
                "instrumented_path_calls"
            ]
        )
        != 2
        or int(
            status_after_reused_intervention[
                "derived_kernel_verifications"
            ]
        )
        != 1
    ):
        raise RuntimeError(
            "repeated lazy instrumentation did not reuse its sidecar"
        )
    if (
        status_after_fast_reuse["last_dispatch"] != "fast_cross_layer"
        or int(status_after_fast_reuse["fast_path_calls"]) != 1
        or int(status_after_fast_reuse["successful_loads"]) != 1
    ):
        raise RuntimeError(
            "loaded lazy runtime did not retain its independent fast path"
        )
    if (
        not eviction_returned_true
        or status_after_explicit_eviction["residency"] != "unloaded"
        or status_after_explicit_eviction["loaded"] is not False
        or int(
            status_after_explicit_eviction[
                "resident_sidecar_tensor_bytes"
            ]
        )
        != 0
        or int(status_after_explicit_eviction["evictions"]) != 1
    ):
        raise RuntimeError(
            "explicit lazy instrumentation eviction did not release state"
        )
    if (
        unfused_output.activations is None
        or lazy_trace_output.activations is None
    ):
        raise RuntimeError("capture_activations did not return a trace")
    unfused_trace = unfused_output.activations
    lazy_trace = lazy_trace_output.activations
    common = tuple(
        name for name in unfused_trace.names if name in lazy_trace
    )
    maximum_by_tap = {
        name: (
            lazy_trace[name] - unfused_trace[name]
        ).abs().max().item()
        for name in common
    }
    names_equal = lazy_trace.names == unfused_trace.names
    identity_difference = (
        identity_intervention_output.logits
        - lazy_trace_output.logits
    ).abs().max().item()
    if not names_equal:
        raise RuntimeError(
            "lazy logical trace taps differ from the unfused modal model"
        )
    if identity_difference != 0.0:
        raise RuntimeError("identity trace intervention changed fused output")
    return {
        "default_dispatch": (
            "seven-tensor forward_fast with exact cross-layer modal bypass"
        ),
        "trace_dispatch": (
            "load the verified four-artifact logical sidecar on first "
            "capture or intervention, then reuse it"
        ),
        "fast_path_has_no_activation_trace": (
            lazy_fast_output.activations is None
        ),
        "capture_returns_activation_trace": True,
        "trace_names_exactly_equal_to_unfused": names_equal,
        "trace_names": list(lazy_trace.names),
        "maximum_unfused_to_fused_difference_by_trace_tap": (
            maximum_by_tap
        ),
        "maximum_unfused_to_fused_trace_logit_difference": (
            (
                lazy_trace_output.logits - unfused_output.logits
            ).abs().max().item()
        ),
        "maximum_fast_to_trace_logit_difference": (
            (
                lazy_fast_output.logits - lazy_trace_output.logits
            ).abs().max().item()
        ),
        "identity_intervention_tap": "layer.0.modal.hidden",
        "identity_intervention_applied": True,
        "identity_intervention_maximum_logit_difference": (
            identity_difference
        ),
        "status_before_instrumentation": status_before,
        "status_after_first_capture": status_after_first_capture,
        "status_after_reused_intervention": (
            status_after_reused_intervention
        ),
        "status_after_fast_reuse": status_after_fast_reuse,
        "explicit_eviction_returned_true": eviction_returned_true,
        "status_after_explicit_eviction": (
            status_after_explicit_eviction
        ),
        "sidecar_loaded_exactly_once": True,
        "repeated_instrumentation_reused_cache": True,
        "fast_dispatch_remained_available_while_sidecar_loaded": True,
        "explicit_eviction_released_sidecar_tensors": True,
    }


def _benchmark(
    *,
    validation_inputs: Tensor,
    teacher: ToyTransformer,
    unfused: ToyTransformer,
    monolithic: FusedToyTransformer,
    lazy: FusedToyTransformer,
) -> list[dict[str, object]]:
    operations_by_batch = {}
    for batch_size in (1, 8, 64, 256):
        inputs = validation_inputs[:batch_size]
        operations_by_batch[batch_size] = {
            "teacher": lambda model=teacher, values=inputs: model(
                values
            ).logits,
            "unfused": lambda model=unfused, values=inputs: model(
                values
            ).logits,
            "monolithic": (
                lambda model=monolithic, values=inputs: model(
                    values
                ).logits
            ),
            "lazy": lambda model=lazy, values=inputs: model(
                values
            ).logits,
        }
    return [
        asdict(report)
        for report in benchmark_batch_sizes(
            operations_by_batch,
            speedup_pairs={
                "monolithic_vs_unfused": (
                    "unfused",
                    "monolithic",
                ),
                "lazy_vs_unfused": ("unfused", "lazy"),
                "monolithic_vs_teacher": (
                    "teacher",
                    "monolithic",
                ),
                "lazy_vs_teacher": ("teacher", "lazy"),
                "lazy_vs_monolithic": ("monolithic", "lazy"),
            },
        )
    ]


def _triangular_benchmark(
    *,
    validation_inputs: Tensor,
    teacher: ToyTransformer,
    unfused: ToyTransformer,
    monolithic: FusedToyTransformer,
    lazy: FusedToyTransformer,
    triangular: FusedToyTransformer,
) -> list[dict[str, object]]:
    """Benchmark all five systems together for directly comparable timing."""

    operations_by_batch = {}
    for batch_size in (1, 8, 64, 256):
        inputs = validation_inputs[:batch_size]
        operations_by_batch[batch_size] = {
            "teacher": lambda model=teacher, values=inputs: model(
                values
            ).logits,
            "unfused": lambda model=unfused, values=inputs: model(
                values
            ).logits,
            "monolithic": (
                lambda model=monolithic, values=inputs: model(
                    values
                ).logits
            ),
            "lazy": lambda model=lazy, values=inputs: model(
                values
            ).logits,
            "triangular": (
                lambda model=triangular, values=inputs: model(
                    values
                ).logits
            ),
        }
    return [
        asdict(report)
        for report in benchmark_batch_sizes(
            operations_by_batch,
            speedup_pairs={
                "monolithic_vs_unfused": (
                    "unfused",
                    "monolithic",
                ),
                "lazy_vs_unfused": ("unfused", "lazy"),
                "monolithic_vs_teacher": (
                    "teacher",
                    "monolithic",
                ),
                "lazy_vs_teacher": ("teacher", "lazy"),
                "lazy_vs_monolithic": ("monolithic", "lazy"),
                "triangular_vs_lazy": ("lazy", "triangular"),
                "triangular_vs_unfused": (
                    "unfused",
                    "triangular",
                ),
                "triangular_vs_teacher": (
                    "teacher",
                    "triangular",
                ),
                "triangular_vs_monolithic": (
                    "monolithic",
                    "triangular",
                ),
            },
        )
    ]


def _environment(
    *,
    systems: tuple[str, ...] = (
        "teacher",
        "unfused",
        "monolithic",
        "lazy",
    ),
) -> dict[str, object]:
    if not systems or any(
        not isinstance(system, str) or not system
        for system in systems
    ):
        raise ValueError("benchmark systems must be nonempty strings")
    if len(set(systems)) != len(systems):
        raise ValueError("benchmark systems must be unique")
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": "cpu",
        "dtype": "float32",
        "torch_intraop_threads_outside_benchmark": (
            torch.get_num_threads()
        ),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "benchmark_contract": {
            "input_split": "validation_fisher",
            "batch_sizes": [1, 8, 64, 256],
            "systems": list(systems),
            "intraop_threads": 1,
            "inference_mode": True,
            "repeats": 9,
            "minimum_block_seconds": 0.2,
            "warmup_iterations": 100,
            "minimum_warmup_seconds": 1.0,
            "ordering": "deterministic rotating system order",
            "scope": (
                "end-to-end fixed-length model forward including embedding, "
                "compiled blocks, final norm, and vocabulary head"
            ),
        },
    }


def _write_markdown(
    path: Path,
    report: Mapping[str, object],
) -> None:
    validation = report["validation"]
    test = report["test"]
    arithmetic = report["arithmetic"]
    benchmark = report["benchmark"]
    benchmark_comparison = report["lazy_vs_monolithic_benchmark"]
    triangular_section = report["triangular_runtime_benchmark"]
    trace = report["dispatch_and_trace_contract"]
    storage = report["storage"]
    assert isinstance(validation, dict)
    assert isinstance(test, dict)
    assert isinstance(arithmetic, dict)
    assert isinstance(benchmark, list)
    assert isinstance(benchmark_comparison, dict)
    assert isinstance(triangular_section, dict)
    assert isinstance(trace, dict)
    assert isinstance(storage, dict)
    triangular_contract = triangular_section["runtime_contract"]
    triangular_validation = triangular_section["validation"]
    triangular_benchmark = triangular_section["benchmark"]
    triangular_comparison = triangular_section["comparison"]
    triangular_source_before = triangular_section[
        "source_lazy_status_before"
    ]
    triangular_source_after = triangular_section[
        "source_lazy_status_after"
    ]
    assert isinstance(triangular_contract, dict)
    assert isinstance(triangular_validation, dict)
    assert isinstance(triangular_benchmark, list)
    assert isinstance(triangular_comparison, dict)
    assert isinstance(triangular_source_before, dict)
    assert isinstance(triangular_source_after, dict)
    triangular_validation_comparison = triangular_validation[
        "triangular_vs_lazy"
    ]
    assert isinstance(triangular_validation_comparison, dict)
    storage_contract = storage["lazy_storage_contract"]
    assert isinstance(storage_contract, dict)
    packed_full_model_bytes = int(
        triangular_contract["packed_fast_state_tensor_bytes"]
    ) + int(storage_contract["model_shell_tensor_bytes"])
    lazy_artifact_file_bytes = int(storage["lazy_artifact_file_bytes"])
    sidecar_total_file_bytes = int(
        storage["instrumentation_sidecar_total_file_bytes"]
    )
    geometric_latency_ratio = float(
        benchmark_comparison[
            "geometric_mean_lazy_to_monolithic_latency_ratio"
        ]
    )

    lines = [
        "# Fused Two-Layer Modal Executor",
        "",
        "The two locked completed modal layers were algebraically folded into",
        "a seven-tensor fast runtime. Normal inference keeps only those",
        "coefficients resident. The first activation capture or intervention",
        "loads the existing logical modal artifacts as a verified sidecar;",
        "later instrumented calls reuse that cache, and explicit eviction",
        "returns the runtime to its default footprint.",
        "",
        "## Equivalence",
        "",
        "| Split | System | Answer accuracy | Paired accuracy | Hard NLL |",
        "|---|---|---:|---:|---:|",
    ]
    for split_name, section in (
        ("Validation", validation),
        ("Exploratory test", test),
    ):
        systems = section["systems"]
        assert isinstance(systems, dict)
        for system in (
            "teacher",
            "unfused",
            "monolithic",
            "lazy",
        ):
            metrics = systems[system]
            assert isinstance(metrics, dict)
            lines.append(
                f"| {split_name} | {system} | "
                f"{float(metrics['answer_accuracy']):.3%} | "
                f"{float(metrics['paired_context_accuracy']):.3%} | "
                f"{float(metrics['hard_nll']):.6f} |"
            )
    validation_comparison = validation["lazy_vs_unfused"]
    lazy_vs_monolithic = validation["lazy_vs_monolithic"]
    assert isinstance(validation_comparison, dict)
    assert isinstance(lazy_vs_monolithic, dict)
    lines.extend(
        [
            "",
            "The validation equivalence gate passed before test evaluation:",
            "",
            f"- Exact argmax predictions: "
            f"{validation_comparison['argmax_predictions_exactly_equal']}",
            f"- Absolute NLL delta: "
            f"{float(validation_comparison['absolute_hard_nll_delta']):.3e}",
            f"- Mean answer KL: "
            f"{float(validation_comparison['mean_unfused_to_fused_answer_kl']):.3e}",
            f"- Maximum answer-logit difference: "
            f"{float(validation_comparison['maximum_answer_logit_difference']):.3e}",
            f"- Lazy and monolithic fast logits bit-exact: "
            f"{lazy_vs_monolithic['logits_bit_exact']}",
            "",
            "## Arithmetic",
            "",
            "| Runtime accounting | Scalar multiplies | Original ratio |",
            "|---|---:|---:|",
            f"| Original two blocks | "
            f"{arithmetic['original_two_block_estimated_multiplies']} | 100.000% |",
            f"| Unfused logical modal stack | "
            f"{arithmetic['unfused_modal_logical_multiplies']} | "
            f"{float(arithmetic['unfused_modal_logical_multiplies']) / float(arithmetic['original_two_block_estimated_multiplies']):.3%} |",
            f"| Current fused dense path | "
            f"{arithmetic['fused_dense_executed_multiplies']} | "
            f"{float(arithmetic['fused_dense_vs_original_ratio']):.3%} |",
            f"| Packed triangular reference | "
            f"{arithmetic['fused_triangular_nonzero_multiplies']} | "
            f"{float(arithmetic['fused_triangular_vs_original_ratio']):.3%} |",
            "",
            "The authenticated dense paths execute kernels that contain",
            "causal zeros. The separate packed triangular reference uses",
            "packed causal-pair PyTorch contractions that execute only the",
            "lower-triangular position pairs; its wall-clock behavior is",
            "measured separately below.",
            "",
            "## CPU latency",
            "",
            "| Batch | Teacher us | Unfused us | Monolithic us | Lazy us | "
            "Lazy vs unfused | Lazy/monolithic latency |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for batch in benchmark:
        timings = batch["timings"]
        speedups = batch["speedup_ratios"]
        assert isinstance(timings, dict)
        assert isinstance(speedups, dict)
        monolithic_median = float(
            timings["monolithic"]["median_microseconds"]
        )
        lazy_median = float(
            timings["lazy"]["median_microseconds"]
        )
        lines.append(
            f"| {batch['batch_size']} | "
            f"{float(timings['teacher']['median_microseconds']):.3f} | "
            f"{float(timings['unfused']['median_microseconds']):.3f} | "
            f"{monolithic_median:.3f} | "
            f"{lazy_median:.3f} | "
            f"{float(speedups['lazy_vs_unfused']):.3f}x | "
            f"{lazy_median / monolithic_median:.3f}x |"
        )
    lines.extend(
        [
            "",
            "No hard latency threshold was applied. Across the four batch",
            "sizes, the geometric-mean lazy/monolithic latency ratio was",
            f"{geometric_latency_ratio:.4f}x "
            "(positive regression means the lazy wrapper was slower).",
            "",
        ]
    )
    lines.extend(
        [
            "## Packed triangular reference benchmark",
            "",
            "This is an ephemeral runtime derived in memory from the",
            "authenticated lazy artifact. It is neither a serialized artifact",
            "nor the default backend, it updates no weights, and it does not",
            "change the authenticated dense runtime ABI. Validation and the",
            "separate five-system benchmark use only `validation_fisher`; the",
            "test split is not used.",
            "",
            f"- Implementation: {triangular_contract['implementation']}",
            f"- Validation gate passed: "
            f"{triangular_validation['gate_passed']}",
            f"- Exact lazy/triangular argmax predictions: "
            f"{triangular_validation_comparison['argmax_predictions_exactly_equal']}",
            f"- Absolute lazy/triangular NLL delta: "
            f"{float(triangular_validation_comparison['absolute_hard_nll_delta']):.3e}",
            f"- Mean lazy-to-triangular answer KL: "
            f"{float(triangular_validation_comparison['mean_unfused_to_fused_answer_kl']):.3e}",
            f"- Maximum lazy/triangular answer-logit difference: "
            f"{float(triangular_validation_comparison['maximum_answer_logit_difference']):.3e}",
            f"- Packed causal position pairs: "
            f"{triangular_contract['packed_causal_pair_count']}",
            f"- Packed fast-state tensor bytes: "
            f"{triangular_contract['packed_fast_state_tensor_bytes']}",
            f"- Source lazy sidecar loads before/after: "
            f"{triangular_source_before['successful_loads']}/"
            f"{triangular_source_after['successful_loads']}",
            f"- Source lazy sidecar bytes read after benchmark: "
            f"{triangular_source_after['sidecar_file_bytes_read']}",
            "",
            "| Batch | Teacher us | Unfused us | Monolithic us | Lazy us | "
            "Triangular us | Triangular vs lazy | "
            "Triangular vs unfused |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for batch in triangular_benchmark:
        timings = batch["timings"]
        speedups = batch["speedup_ratios"]
        assert isinstance(timings, dict)
        assert isinstance(speedups, dict)
        lines.append(
            f"| {batch['batch_size']} | "
            f"{float(timings['teacher']['median_microseconds']):.3f} | "
            f"{float(timings['unfused']['median_microseconds']):.3f} | "
            f"{float(timings['monolithic']['median_microseconds']):.3f} | "
            f"{float(timings['lazy']['median_microseconds']):.3f} | "
            f"{float(timings['triangular']['median_microseconds']):.3f} | "
            f"{float(speedups['triangular_vs_lazy']):.3f}x | "
            f"{float(speedups['triangular_vs_unfused']):.3f}x |"
        )
    lines.extend(
        [
            "",
            "Across the four batch sizes, the geometric-mean triangular",
            "speedup was "
            f"{float(triangular_comparison['geometric_mean_triangular_vs_lazy_speedup']):.4f}x "
            "versus lazy and "
            f"{float(triangular_comparison['geometric_mean_triangular_vs_unfused_speedup']):.4f}x "
            "versus unfused. Speedups above 1 mean triangular was faster;",
            "values below 1 mean it was slower. No hard latency threshold was",
            "applied.",
            "",
            "## Resident tensor storage",
            "",
            "| State | Bytes |",
            "|---|---:|",
            f"| Monolithic fused full runtime | "
            f"{storage['fused_full_model']['total_state_bytes']} |",
            f"| Lazy full runtime, default | "
            f"{storage_contract['default_full_runtime_resident_tensor_bytes']} "
            "|",
            f"| Logical sidecar after first instrumentation | "
            f"{storage_contract['sidecar_resident_tensor_bytes']} |",
            f"| Lazy full runtime, sidecar loaded | "
            f"{storage_contract['loaded_full_runtime_resident_tensor_bytes']} "
            "|",
            f"| Packed triangular full model | "
            f"{packed_full_model_bytes} |",
            "",
            "On disk, the compact runtime can be deployed by itself for",
            "uninstrumented inference. An instrumentable bundle also carries",
            "the four existing modal source artifacts:",
            "",
            "| Artifact files | Bytes |",
            "|---|---:|",
            f"| Compact lazy runtime | "
            f"{lazy_artifact_file_bytes} |",
            f"| Four logical sidecar source files | "
            f"{sidecar_total_file_bytes} |",
            f"| Compact runtime plus sidecars | "
            f"{lazy_artifact_file_bytes + sidecar_total_file_bytes} |",
            f"| Monolithic fused artifact | "
            f"{storage['fused_artifact_file_bytes']} |",
            "",
            "## Instrumentation contract",
            "",
            f"- Default dispatch: {trace['default_dispatch']}",
            f"- Trace dispatch: {trace['trace_dispatch']}",
            f"- Sidecar loaded exactly once: "
            f"{trace['sidecar_loaded_exactly_once']}",
            f"- Repeated instrumentation reused the cache: "
            f"{trace['repeated_instrumentation_reused_cache']}",
            f"- Explicit eviction released sidecar tensors: "
            f"{trace['explicit_eviction_released_sidecar_tensors']}",
            f"- Trace names equal the unfused runtime: "
            f"{trace['trace_names_exactly_equal_to_unfused']}",
            f"- Fast-to-traced maximum logit difference: "
            f"{float(trace['maximum_fast_to_trace_logit_difference']):.3e}",
            "",
            "Benchmark timings are hardware- and backend-specific. Arithmetic",
            "counts cover only the two replaced blocks; timings cover the",
            "complete model forward. This remains an exploratory",
            "single-checkpoint result because the test split was inspected",
            "during earlier development.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_fused_executor_experiment(
    *,
    artifact_dir: Path,
) -> dict[str, object]:
    """Fuse locked modal artifacts, validation-gate, test once, and benchmark."""

    started = time.perf_counter()
    output_paths = fused_executor_artifact_paths(artifact_dir)
    checkpoint_path = artifact_dir / "checkpoint.pt"
    fisher_path = artifact_dir / "fisher_modes.pt"
    manifest_path = artifact_dir / "split_manifest.json"
    checkpoint_hash = _sha256(checkpoint_path)
    fisher_hash = _sha256(fisher_path)
    manifest_hash = _sha256(manifest_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = TransformerConfig(**checkpoint["model_config"])
    if model_config.n_layers != 2:
        raise ValueError("the fused experiment requires exactly two layers")
    task_config = AssociativeRecallTaskConfig(**checkpoint["task_config"])
    if task_config.sequence_length != model_config.max_sequence_length:
        raise ValueError("task and transformer sequence lengths differ")
    splits = build_associative_recall_splits(task_config)
    manifest = json.loads(manifest_path.read_text())
    for split_name, split in (
        ("train", splits.train),
        ("validation_fisher", splits.validation),
        ("test", splits.test),
    ):
        manifest_split = manifest[split_name]
        if manifest_split["context_ids"] != split.context_ids.tolist():
            raise ValueError(
                f"split manifest context IDs mismatch: {split_name}"
            )
        if (
            manifest_split["context_ids_sha256"]
            != _tensor_sha256(split.context_ids)
        ):
            raise ValueError(
                f"split manifest context hash mismatch: {split_name}"
            )

    teacher = _new_teacher(
        model_config,
        checkpoint["model_state_dict"],
    )
    teacher_state_hash = _module_state_sha256(teacher)
    executor_paths = {
        index: modal_executor_artifact_paths(artifact_dir, index)
        for index in (0, 1)
    }
    completion_paths = {
        index: modal_completion_artifact_paths(artifact_dir, index)
        for index in (0, 1)
    }
    source_paths = {
        "checkpoint": checkpoint_path,
        "fisher_modes": fisher_path,
        "split_manifest": manifest_path,
        "layer_0_executor": executor_paths[0].executor,
        "layer_0_output_completion": (
            completion_paths[0].output_completion
        ),
        "layer_1_executor": executor_paths[1].executor,
        "layer_1_output_completion": (
            completion_paths[1].output_completion
        ),
    }
    source_hashes = {
        name: _sha256(path) for name, path in source_paths.items()
    }
    instrumentation_sidecar_paths = {
        name: source_paths[name]
        for name in (
            "layer_0_executor",
            "layer_0_output_completion",
            "layer_1_executor",
            "layer_1_output_completion",
        )
    }
    instrumentation_sidecar_file_bytes = {
        name: path.stat().st_size
        for name, path in instrumentation_sidecar_paths.items()
    }
    instrumentation_sidecar_total_file_bytes = sum(
        instrumentation_sidecar_file_bytes.values()
    )

    loaded: list[
        tuple[
            PositionConditionedModalGraphExecutor,
            PositionConditionedCompletedModalGraphExecutor,
            ModalExecutorConfig,
            dict[str, object],
        ]
    ] = []
    boundaries = (
        ("layer.0.input", "layer.0.output"),
        ("layer.0.output", "layer.1.output"),
    )
    for layer_index, (input_name, output_name) in enumerate(boundaries):
        loaded.append(
            _validate_layer_artifacts(
                layer_index=layer_index,
                checkpoint_hash=checkpoint_hash,
                fisher_hash=fisher_hash,
                executor_path=executor_paths[layer_index].executor,
                output_completion_path=(
                    completion_paths[layer_index].output_completion
                ),
                expected_input=input_name,
                expected_output=output_name,
                sequence_length=task_config.sequence_length,
                width=model_config.d_model,
                teacher_state_sha256=teacher_state_hash,
                fit_context_ids_sha256=manifest["train"][
                    "context_ids_sha256"
                ],
                selection_context_ids_sha256=manifest[
                    "validation_fisher"
                ]["context_ids_sha256"],
            )
        )
    first_base, first_completed, first_config, first_provenance = (
        loaded[0]
    )
    second_base, second_completed, second_config, second_provenance = (
        loaded[1]
    )
    unfused = _build_unfused_model(
        model_config=model_config,
        state_dict=checkpoint["model_state_dict"],
        first=first_completed,
        second=second_completed,
    )

    built_stack = FusedTwoLayerModalStack.from_executors(
        first_completed,
        second_completed,
        require_cross_layer_bypass=True,
    )
    build_metadata: dict[str, object] = {
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "split_manifest_sha256": manifest_hash,
        "teacher_state_sha256": teacher_state_hash,
        "source_artifacts": {
            name: {
                "filename": path.name,
                "sha256": source_hashes[name],
            }
            for name, path in source_paths.items()
        },
        "layer_provenance": {
            "layer_0": first_provenance,
            "layer_1": second_provenance,
        },
        "build_contract": {
            "operation": "algebraic_fusion_only",
            "weights_fitted_or_updated": False,
            "cross_layer_bypass_required": True,
            "test_used_for_build_or_selection": False,
        },
    }
    save_fused_modal_stack(
        output_paths.stack,
        stack=built_stack,
        metadata=build_metadata,
    )
    save_lazy_fused_modal_stack(
        output_paths.runtime,
        stack=built_stack,
        sidecar_paths=instrumentation_sidecar_paths,
        metadata=build_metadata,
    )
    fused_hash_before_validation = _sha256(output_paths.stack)
    lazy_hash_before_validation = _sha256(output_paths.runtime)
    reloaded_stack, reloaded_config, reloaded_metadata = (
        load_fused_modal_stack(output_paths.stack)
    )
    if reloaded_config != built_stack.config:
        raise RuntimeError("reloaded fused config differs from built config")
    if reloaded_metadata != build_metadata:
        raise RuntimeError("reloaded fused metadata differs from build metadata")
    if not reloaded_stack.uses_cross_layer_bypass:
        raise RuntimeError("reloaded stack lost its cross-layer bypass")
    monolithic = FusedToyTransformer.from_teacher(
        teacher,
        reloaded_stack,
    )
    _freeze(monolithic)
    if list(monolithic.parameters()):
        raise RuntimeError("fused runtime unexpectedly contains parameters")

    lazy_fast_stack, lazy_config, lazy_metadata = (
        load_lazy_fused_modal_stack(output_paths.runtime)
    )
    if lazy_config != reloaded_config:
        raise RuntimeError("lazy fused config differs from monolithic config")
    if lazy_metadata != build_metadata:
        raise RuntimeError("lazy fused metadata differs from build metadata")
    lazy = FusedToyTransformer.from_teacher(
        teacher,
        lazy_fast_stack,
    )
    _freeze(lazy)
    if list(lazy.parameters()):
        raise RuntimeError(
            "lazy fused runtime unexpectedly contains parameters"
        )
    fast_status_before_validation = _status_dict(lazy_fast_stack)
    _require_no_sidecar_activity(
        fast_status_before_validation,
        phase="validation setup",
    )

    trace_stack, trace_config, trace_metadata = (
        load_lazy_fused_modal_stack(output_paths.runtime)
    )
    if trace_config != lazy_config or trace_metadata != lazy_metadata:
        raise RuntimeError(
            "fresh lazy trace runtime differs from lazy fast runtime"
        )
    trace_runtime = FusedToyTransformer.from_teacher(
        teacher,
        trace_stack,
    )
    _freeze(trace_runtime)

    arithmetic = _fused_arithmetic(
        stack=reloaded_stack,
        model_config=model_config,
        first_base=first_base,
        first_completed=first_completed,
        first_config=first_config,
        second_base=second_base,
        second_completed=second_completed,
        second_config=second_config,
    )
    trace_contract = _trace_contract(
        validation_inputs=splits.validation.input_ids[:8],
        unfused=unfused,
        lazy=trace_runtime,
        lazy_stack=trace_stack,
        expected_sidecar_file_bytes=(
            instrumentation_sidecar_total_file_bytes
        ),
    )
    loaded_trace_status = trace_contract[
        "status_after_first_capture"
    ]
    assert isinstance(loaded_trace_status, dict)
    lazy_storage = _lazy_storage_contract(
        lazy_model=lazy,
        lazy_fast_status=fast_status_before_validation,
        loaded_status=loaded_trace_status,
    )

    print("Evaluating fused runtime equivalence on validation", flush=True)
    validation, _ = _evaluate_systems(
        split=splits.validation,
        teacher=teacher,
        unfused=unfused,
        monolithic=monolithic,
        lazy=lazy,
    )
    monolithic_comparison = validation["monolithic_vs_unfused"]
    lazy_comparison = validation["lazy_vs_unfused"]
    lazy_monolithic_comparison = validation["lazy_vs_monolithic"]
    assert isinstance(monolithic_comparison, dict)
    assert isinstance(lazy_comparison, dict)
    assert isinstance(lazy_monolithic_comparison, dict)
    gate_passed = bool(
        _passes_fusion_gate(
            monolithic_comparison,
            _VALIDATION_GATE,
        )
        and _passes_fusion_gate(
            lazy_comparison,
            _VALIDATION_GATE,
        )
        and lazy_monolithic_comparison["logits_bit_exact"]
    )
    if not gate_passed:
        raise RuntimeError(
            "fused runtime failed validation equivalence gate"
        )
    fast_status_after_validation = _status_dict(lazy_fast_stack)
    _require_no_sidecar_activity(
        fast_status_after_validation,
        phase="validation",
    )
    if _sha256(output_paths.stack) != fused_hash_before_validation:
        raise RuntimeError("fused artifact changed during validation")
    if _sha256(output_paths.runtime) != lazy_hash_before_validation:
        raise RuntimeError(
            "lazy fused artifact changed during validation"
        )
    if {
        name: _sha256(path) for name, path in source_paths.items()
    } != source_hashes:
        raise RuntimeError("source artifacts changed during fused validation")
    if _module_state_sha256(teacher) != teacher_state_hash:
        raise RuntimeError("teacher changed during fused validation")

    print("Fused runtime locked; evaluating exploratory test once", flush=True)
    test, _ = _evaluate_systems(
        split=splits.test,
        teacher=teacher,
        unfused=unfused,
        monolithic=monolithic,
        lazy=lazy,
    )
    fast_status_after_test = _status_dict(lazy_fast_stack)
    _require_no_sidecar_activity(
        fast_status_after_test,
        phase="test",
    )

    print(
        "Benchmarking teacher, unfused, monolithic, and lazy CPU runtimes",
        flush=True,
    )
    benchmark = _benchmark(
        validation_inputs=splits.validation.input_ids,
        teacher=teacher,
        unfused=unfused,
        monolithic=monolithic,
        lazy=lazy,
    )
    lazy_vs_monolithic_benchmark = _lazy_benchmark_comparison(
        benchmark
    )
    fast_status_after_benchmark = _status_dict(lazy_fast_stack)
    _require_no_sidecar_activity(
        fast_status_after_benchmark,
        phase="benchmark",
    )

    triangular_sidecar_root = (
        artifact_dir / ".triangular-runtime-sidecars-unavailable"
    )
    if triangular_sidecar_root.exists():
        raise RuntimeError(
            "packed triangular benchmark requires an unavailable sidecar root"
        )
    (
        triangular_source_stack,
        triangular_source_config,
        triangular_source_metadata,
    ) = load_lazy_fused_modal_stack(
        output_paths.runtime,
        sidecar_root=triangular_sidecar_root,
    )
    if triangular_source_config != lazy_config:
        raise RuntimeError(
            "packed triangular source config differs from lazy config"
        )
    if triangular_source_metadata != lazy_metadata:
        raise RuntimeError(
            "packed triangular source metadata differs from lazy metadata"
        )
    triangular_source_status_before = _status_dict(
        triangular_source_stack
    )
    _require_no_sidecar_activity(
        triangular_source_status_before,
        phase="packed triangular setup",
    )
    triangular_source = FusedToyTransformer.from_teacher(
        teacher,
        triangular_source_stack,
    )
    _freeze(triangular_source)
    triangular_stack = (
        PackedTriangularFusedTwoLayerModalStack.from_lazy(
            triangular_source_stack
        )
    )
    triangular = FusedToyTransformer.from_teacher(
        teacher,
        triangular_stack,
    )
    _freeze(triangular)
    if list(triangular_source.parameters()) or list(
        triangular.parameters()
    ):
        raise RuntimeError(
            "packed triangular benchmark runtimes unexpectedly contain "
            "parameters"
        )

    print(
        "Validating ephemeral packed triangular runtime against lazy source",
        flush=True,
    )
    triangular_source_logits = associative_recall_answer_logits(
        triangular_source,
        splits.validation,
    )
    triangular_logits = associative_recall_answer_logits(
        triangular,
        splits.validation,
    )
    triangular_validation_comparison = _compare_fused_to_unfused(
        split=splits.validation,
        unfused_logits=triangular_source_logits,
        fused_logits=triangular_logits,
    )
    triangular_gate_passed = _passes_fusion_gate(
        triangular_validation_comparison,
        _VALIDATION_GATE,
    )
    if not triangular_gate_passed:
        raise RuntimeError(
            "packed triangular runtime failed validation equivalence gate"
        )

    print(
        "Benchmarking separate five-system packed triangular CPU cohort",
        flush=True,
    )
    triangular_benchmark = _triangular_benchmark(
        validation_inputs=splits.validation.input_ids,
        teacher=teacher,
        unfused=unfused,
        monolithic=monolithic,
        lazy=triangular_source,
        triangular=triangular,
    )
    triangular_benchmark_comparison = (
        _triangular_benchmark_comparison(triangular_benchmark)
    )
    triangular_source_status_after = _status_dict(
        triangular_source_stack
    )
    _require_no_sidecar_activity(
        triangular_source_status_after,
        phase="packed triangular validation and benchmark",
    )
    triangular_runtime_benchmark: dict[str, object] = {
        "source_lazy_artifact": {
            "filename": output_paths.runtime.name,
            "sha256": lazy_hash_before_validation,
            "artifact_kind": "lazy_fused_two_layer_modal_stack",
            "format_version": 2,
        },
        "runtime_contract": {
            "implementation": "packed_triangular_prefix_v1",
            "serialized_artifact": False,
            "default_backend": False,
            "weights_updated": False,
            "test_used": False,
            "validation_split": "validation_fisher",
            "benchmark_split": "validation_fisher",
            "packed_causal_pair_count": (
                triangular_stack.causal_pair_count
            ),
            "packed_fast_state_tensor_bytes": (
                triangular_stack.packed_fast_state_bytes
            ),
        },
        "source_lazy_status_before": triangular_source_status_before,
        "source_lazy_status_after": triangular_source_status_after,
        "validation": {
            "gate": dict(_VALIDATION_GATE),
            "gate_passed": triangular_gate_passed,
            "triangular_vs_lazy": triangular_validation_comparison,
        },
        "benchmark_environment": _environment(
            systems=(
                "teacher",
                "unfused",
                "monolithic",
                "lazy",
                "triangular",
            )
        ),
        "benchmark": triangular_benchmark,
        "comparison": triangular_benchmark_comparison,
    }

    fused_hash_after = _sha256(output_paths.stack)
    lazy_hash_after = _sha256(output_paths.runtime)
    source_hashes_after = {
        name: _sha256(path) for name, path in source_paths.items()
    }
    teacher_state_after = _module_state_sha256(teacher)
    if fused_hash_after != fused_hash_before_validation:
        raise RuntimeError("fused artifact changed after lock")
    if lazy_hash_after != lazy_hash_before_validation:
        raise RuntimeError("lazy fused artifact changed after lock")
    if source_hashes_after != source_hashes:
        raise RuntimeError("source artifacts changed after lock")
    if teacher_state_after != teacher_state_hash:
        raise RuntimeError("teacher changed after lock")

    report: dict[str, object] = {
        "format_version": _FUSED_REPORT_FORMAT_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "fisher_sha256": fisher_hash,
        "split_manifest_sha256": manifest_hash,
        "teacher_state_sha256_before": teacher_state_hash,
        "teacher_state_sha256_after": teacher_state_after,
        "protocol": {
            "operation": "algebraic_fusion_of_locked_modal_artifacts",
            "validation_split": "validation_fisher",
            "evaluation_split": "test",
            "test_used_for_build_or_selection": False,
            "fused_artifact_saved_and_reloaded_before_validation": True,
            "fused_artifact_saved_and_reloaded_before_test": True,
            "lazy_artifact_saved_and_reloaded_before_validation": True,
            "lazy_artifact_saved_and_reloaded_before_test": True,
            "validation_test_and_benchmark_runtime": (
                "fresh_lazy_fast_runtime"
            ),
            "instrumentation_runtime": (
                "separate_fresh_lazy_runtime"
            ),
            "fast_runtime_sidecar_loads_during_validation_test_benchmark": 0,
            "validation_gate": dict(_VALIDATION_GATE),
            "validation_gate_passed": gate_passed,
            "test_evaluated_once_after_gate": True,
        },
        "validation": validation,
        "test": test,
        "arithmetic": arithmetic,
        "dispatch_and_trace_contract": trace_contract,
        "benchmark_environment": _environment(),
        "benchmark": benchmark,
        "lazy_vs_monolithic_benchmark": (
            lazy_vs_monolithic_benchmark
        ),
        "triangular_runtime_benchmark": (
            triangular_runtime_benchmark
        ),
        "lazy_fast_runtime_status": {
            "before_validation": fast_status_before_validation,
            "after_validation": fast_status_after_validation,
            "after_test": fast_status_after_test,
            "after_benchmark": fast_status_after_benchmark,
            "zero_sidecar_loads_throughout": True,
        },
        "storage": {
            "teacher_full_model": _state_storage(teacher),
            "unfused_full_model": _state_storage(unfused),
            "unfused_two_layer_executors": _state_storage(
                nn.ModuleList([first_completed, second_completed])
            ),
            "fused_full_model": _state_storage(monolithic),
            "fused_two_layer_stack": _state_storage(reloaded_stack),
            "lazy_default_full_model_state": _state_storage(lazy),
            "lazy_default_fast_stack_state": _state_storage(
                lazy_fast_stack
            ),
            "lazy_storage_contract": lazy_storage,
            "fused_artifact_file_bytes": (
                output_paths.stack.stat().st_size
            ),
            "lazy_artifact_file_bytes": (
                output_paths.runtime.stat().st_size
            ),
            "instrumentation_sidecar_file_bytes": (
                instrumentation_sidecar_file_bytes
            ),
            "instrumentation_sidecar_total_file_bytes": (
                instrumentation_sidecar_total_file_bytes
            ),
            "source_artifact_file_bytes": {
                name: path.stat().st_size
                for name, path in source_paths.items()
            },
        },
        "source_artifacts": {
            name: {
                "filename": path.name,
                "sha256": source_hashes[name],
            }
            for name, path in source_paths.items()
        },
        "fused_artifact": {
            "filename": output_paths.stack.name,
            "sha256": fused_hash_before_validation,
            "config": asdict(reloaded_config),
            "metadata": reloaded_metadata,
        },
        "lazy_fused_artifact": {
            "filename": output_paths.runtime.name,
            "sha256": lazy_hash_before_validation,
            "config": asdict(lazy_config),
            "metadata": lazy_metadata,
            "format_version": 2,
            "artifact_kind": "lazy_fused_two_layer_modal_stack",
            "fast_state_tensor_bytes": (
                fast_status_before_validation[
                    "resident_fast_tensor_bytes"
                ]
            ),
            "sidecar_descriptors": {
                name: {
                    "filename": path.name,
                    "sha256": source_hashes[name],
                    "size_bytes": path.stat().st_size,
                }
                for name, path in instrumentation_sidecar_paths.items()
            },
        },
        "artifacts_locked_before_validation_and_test": True,
        "scientific_status": (
            "exploratory_single_checkpoint_validation_fisher_informed_"
            "test_previously_inspected"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_paths.report_json.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _write_markdown(output_paths.report_markdown, report)
    save_runtime_manifest(
        artifact_dir / "runtime_manifest.json",
        manifest_from_legacy_runtime(artifact_dir),
    )
    fused_test = test["systems"]["lazy"]  # type: ignore[index]
    fused_vs_unfused = test["lazy_vs_unfused"]  # type: ignore[index]
    print(
        "Lazy fused runtime complete: "
        f"test accuracy={float(fused_test['answer_accuracy']):.3%}, "
        f"paired={float(fused_test['paired_context_accuracy']):.3%}, "
        f"NLL={float(fused_test['hard_nll']):.6f}, "
        "max fused/unfused logit delta="
        f"{float(fused_vs_unfused['maximum_answer_logit_difference']):.3e}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse and benchmark the locked two-layer modal runtime."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/associative_recall"),
    )
    args = parser.parse_args()
    run_fused_executor_experiment(artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
