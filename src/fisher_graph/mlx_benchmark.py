"""Reproducible same-device MLX benchmarks for the packed modal stack.

The benchmark compares three executions of the same two-layer modal stack:

* ``mlx_dense_compiled`` reconstructs zero-filled dense causal kernels and
  evaluates them through an ``mx.compile`` graph;
* ``mlx_packed_compiled`` evaluates the ordinary packed MLX reference graph;
* ``mlx_packed_metal`` evaluates the custom packed causal Metal kernel.

MLX is imported lazily by :mod:`fisher_graph.mlx_executor`, so importing this
module remains safe on systems where the optional MLX dependency is absent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from .fused_executor import (
    PackedTriangularFusedTwoLayerModalStack,
    load_lazy_fused_modal_stack,
)
from .mlx_executor import (
    MLXPackedTriangularFusedTwoLayerModalStack,
    mlx_runtime_provenance,
)


MLXBenchmarkOperation: TypeAlias = Callable[[], object]
SpeedupPair: TypeAlias = tuple[str, str]

MLX_BENCHMARK_SYSTEMS = (
    "mlx_dense_compiled",
    "mlx_packed_compiled",
    "mlx_packed_metal",
)
MLX_BENCHMARK_FORMAT_VERSION = 1

_DEFAULT_SPEEDUP_PAIRS: dict[str, SpeedupPair] = {
    "packed_compiled_vs_dense_compiled": (
        "mlx_dense_compiled",
        "mlx_packed_compiled",
    ),
    "packed_metal_vs_dense_compiled": (
        "mlx_dense_compiled",
        "mlx_packed_metal",
    ),
    "packed_metal_vs_packed_compiled": (
        "mlx_packed_compiled",
        "mlx_packed_metal",
    ),
}


@dataclass(frozen=True, slots=True)
class MLXTiming:
    """First-observed and steady-state timing for one MLX system."""

    first_observed_call_microseconds: float
    warmup_calls: int
    iterations_per_round: int
    rounds: int
    raw_microseconds: tuple[float, ...]
    median_microseconds: float
    minimum_microseconds: float
    maximum_microseconds: float
    p10_microseconds: float
    p90_microseconds: float


@dataclass(frozen=True, slots=True)
class MLXBatchTimingReport:
    """Same-input timings for all MLX systems at one batch size."""

    batch_size: int
    timings: dict[str, MLXTiming]
    examples_per_second: dict[str, float]
    speedup_ratios: dict[str, float]
    round_orders: tuple[tuple[str, ...], ...]


def unpack_packed_triangular_kernel(
    packed: np.ndarray,
    *,
    sequence_length: int,
) -> np.ndarray:
    """Expand target-major packed causal pairs into a zero-filled kernel.

    The input shape is ``[S * (S + 1) / 2, input_width, output_width]``.
    The returned shape is ``[S, S, input_width, output_width]`` and entries
    where ``source > target`` remain exactly zero.
    """

    if not isinstance(packed, np.ndarray):
        raise TypeError("packed must be a numpy.ndarray")
    if packed.ndim != 3:
        raise ValueError(
            "packed must have shape [causal_pair, input, output]"
        )
    if type(sequence_length) is not int or sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    expected_pairs = sequence_length * (sequence_length + 1) // 2
    if packed.shape[0] != expected_pairs:
        raise ValueError(
            f"packed has {packed.shape[0]} pairs; expected {expected_pairs}"
        )

    dense = np.zeros(
        (
            sequence_length,
            sequence_length,
            packed.shape[1],
            packed.shape[2],
        ),
        dtype=packed.dtype,
    )
    offset = 0
    for target in range(sequence_length):
        count = target + 1
        dense[target, :count] = packed[offset : offset + count]
        offset += count
    return dense


def make_synchronized_mlx_operation(
    core: Any,
    forward: Callable[..., Any],
    *arguments: Any,
    **keyword_arguments: Any,
) -> MLXBenchmarkOperation:
    """Wrap a lazy MLX forward so every invocation is fully measured.

    ``forward`` is called inside the returned operation. It must therefore
    produce a fresh lazy output on every call. ``mx.eval`` materializes that
    output and ``mx.synchronize`` prevents asynchronous GPU work from escaping
    the timed interval.
    """

    if not callable(forward):
        raise TypeError("forward must be callable")
    if not callable(getattr(core, "eval", None)):
        raise TypeError("core must provide eval")
    if not callable(getattr(core, "synchronize", None)):
        raise TypeError("core must provide synchronize")

    def operation() -> object:
        output = forward(*arguments, **keyword_arguments)
        core.eval(output)
        core.synchronize()
        return output

    return operation


def _validate_benchmark_options(
    *,
    rounds: int,
    warmup_calls: int,
    minimum_warmup_seconds: float,
    iterations_per_round: int,
) -> None:
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if type(warmup_calls) is not int or warmup_calls < 0:
        raise ValueError("warmup_calls must be a nonnegative integer")
    if (
        not math.isfinite(minimum_warmup_seconds)
        or minimum_warmup_seconds < 0
    ):
        raise ValueError(
            "minimum_warmup_seconds must be finite and nonnegative"
        )
    if type(iterations_per_round) is not int or iterations_per_round <= 0:
        raise ValueError(
            "iterations_per_round must be a positive integer"
        )


def _validate_operations(
    operations_by_batch: Mapping[
        int,
        Mapping[str, MLXBenchmarkOperation],
    ],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not operations_by_batch:
        raise ValueError("at least one benchmark batch is required")
    batch_sizes = tuple(sorted(operations_by_batch))
    if any(
        type(batch_size) is not int or batch_size <= 0
        for batch_size in batch_sizes
    ):
        raise ValueError("benchmark batch sizes must be positive integers")

    system_names = tuple(operations_by_batch[batch_sizes[0]])
    if not system_names:
        raise ValueError("at least one benchmark system is required")
    if any(not isinstance(name, str) or not name for name in system_names):
        raise ValueError("benchmark system names must be nonempty strings")
    for batch_size in batch_sizes:
        operations = operations_by_batch[batch_size]
        if tuple(operations) != system_names:
            raise ValueError(
                "every batch must define benchmark systems in the same order"
            )
        if any(not callable(operation) for operation in operations.values()):
            raise TypeError("benchmark operations must be callable")
    return batch_sizes, system_names


def _validate_speedup_pairs(
    pairs: Mapping[str, SpeedupPair],
    *,
    system_names: tuple[str, ...],
) -> None:
    known = set(system_names)
    for label, pair in pairs.items():
        if not isinstance(label, str) or not label:
            raise ValueError("speedup labels must be nonempty strings")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "speedup pairs must be (reference, candidate) tuples"
            )
        reference, candidate = pair
        if reference not in known or candidate not in known:
            raise ValueError(
                "speedup pairs must refer to benchmark system names"
            )
        if reference == candidate:
            raise ValueError(
                "speedup reference and candidate must be different"
            )


def _run_calls(
    operation: MLXBenchmarkOperation,
    calls: int,
) -> float:
    garbage_collection_enabled = gc.isenabled()
    if garbage_collection_enabled:
        gc.disable()
    try:
        started = time.perf_counter_ns()
        for _ in range(calls):
            operation()
        return (time.perf_counter_ns() - started) / 1e3
    finally:
        if garbage_collection_enabled:
            gc.enable()


def _percentile(
    sorted_values: tuple[float, ...],
    fraction: float,
) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _summarize(
    *,
    first_observed_call_microseconds: float,
    warmup_calls: int,
    iterations_per_round: int,
    raw_microseconds: list[float],
) -> MLXTiming:
    ordered = tuple(sorted(raw_microseconds))
    return MLXTiming(
        first_observed_call_microseconds=(
            first_observed_call_microseconds
        ),
        warmup_calls=warmup_calls,
        iterations_per_round=iterations_per_round,
        rounds=len(raw_microseconds),
        raw_microseconds=tuple(raw_microseconds),
        median_microseconds=statistics.median(raw_microseconds),
        minimum_microseconds=ordered[0],
        maximum_microseconds=ordered[-1],
        p10_microseconds=_percentile(ordered, 0.1),
        p90_microseconds=_percentile(ordered, 0.9),
    )


def benchmark_synchronized_operations(
    operations_by_batch: Mapping[
        int,
        Mapping[str, MLXBenchmarkOperation],
    ],
    *,
    rounds: int = 9,
    warmup_calls: int = 100,
    minimum_warmup_seconds: float = 1.0,
    iterations_per_round: int = 100,
    speedup_pairs: Mapping[str, SpeedupPair] | None = None,
) -> tuple[MLXBatchTimingReport, ...]:
    """Measure synchronized operations with first-call and rotating rounds.

    Operations must perform their own device synchronization. For MLX forwards,
    construct them with :func:`make_synchronized_mlx_operation`.

    Raw steady-state samples are per-call microseconds averaged over one
    ``iterations_per_round`` block. A speedup pair is ``(reference,
    candidate)`` and is reported as ``reference_median / candidate_median``.
    """

    _validate_benchmark_options(
        rounds=rounds,
        warmup_calls=warmup_calls,
        minimum_warmup_seconds=minimum_warmup_seconds,
        iterations_per_round=iterations_per_round,
    )
    batch_sizes, system_names = _validate_operations(operations_by_batch)
    pairs = (
        {
            label: pair
            for label, pair in _DEFAULT_SPEEDUP_PAIRS.items()
            if pair[0] in system_names and pair[1] in system_names
        }
        if speedup_pairs is None
        else dict(speedup_pairs)
    )
    _validate_speedup_pairs(pairs, system_names=system_names)

    reports: list[MLXBatchTimingReport] = []
    for batch_size in batch_sizes:
        operations = operations_by_batch[batch_size]
        first_observed = {
            name: _run_calls(operations[name], 1)
            for name in system_names
        }
        actual_warmup_calls: dict[str, int] = {}
        for name in system_names:
            calls = 0
            started = time.perf_counter_ns()
            while True:
                elapsed_seconds = (
                    time.perf_counter_ns() - started
                ) / 1e9
                if (
                    calls >= warmup_calls
                    and elapsed_seconds >= minimum_warmup_seconds
                ):
                    break
                operations[name]()
                calls += 1
            actual_warmup_calls[name] = calls

        raw = {name: [] for name in system_names}
        round_orders: list[tuple[str, ...]] = []
        for round_index in range(rounds):
            offset = round_index % len(system_names)
            order = system_names[offset:] + system_names[:offset]
            round_orders.append(order)
            for name in order:
                elapsed_microseconds = _run_calls(
                    operations[name],
                    iterations_per_round,
                )
                raw[name].append(
                    elapsed_microseconds / iterations_per_round
                )

        timings = {
            name: _summarize(
                first_observed_call_microseconds=first_observed[name],
                warmup_calls=actual_warmup_calls[name],
                iterations_per_round=iterations_per_round,
                raw_microseconds=raw[name],
            )
            for name in system_names
        }
        examples_per_second = {
            name: batch_size * 1e6 / timing.median_microseconds
            for name, timing in timings.items()
        }
        speedup_ratios = {
            label: (
                timings[reference].median_microseconds
                / timings[candidate].median_microseconds
            )
            for label, (reference, candidate) in pairs.items()
        }
        reports.append(
            MLXBatchTimingReport(
                batch_size=batch_size,
                timings=timings,
                examples_per_second=examples_per_second,
                speedup_ratios=speedup_ratios,
                round_orders=tuple(round_orders),
            )
        )
    return tuple(reports)


def build_mlx_dense_compiled_forward(
    runtime: MLXPackedTriangularFusedTwoLayerModalStack,
) -> Callable[[Any], Any]:
    core = runtime._core
    neural = runtime._neural
    core.eval(
        runtime.packed_first_input_kernel,
        runtime.packed_bridge_kernel,
    )
    core.synchronize()
    first_host = np.array(
        runtime.packed_first_input_kernel,
        copy=True,
    )
    bridge_host = np.array(
        runtime.packed_bridge_kernel,
        copy=True,
    )
    dense_first = core.array(
        unpack_packed_triangular_kernel(
            first_host,
            sequence_length=runtime.sequence_length,
        ),
        dtype=core.float32,
    )
    dense_bridge = core.array(
        unpack_packed_triangular_kernel(
            bridge_host,
            sequence_length=runtime.sequence_length,
        ),
        dtype=core.float32,
    )
    core.eval(dense_first, dense_bridge)
    core.synchronize()

    def forward(hidden_states: Any) -> Any:
        first_hidden = neural.gelu(
            core.einsum(
                "bsi,tsio->bto",
                hidden_states - runtime.first_input_mean,
                dense_first,
            )
            + runtime.first_hidden_bias
        )
        second_hidden = neural.gelu(
            core.einsum(
                "bsi,tsio->bto",
                first_hidden,
                dense_bridge,
            )
            + runtime.bridge_bias
        )
        return (
            core.einsum(
                "bsh,shw->bsw",
                second_hidden,
                runtime.second_fused_output_weight,
            )
            + runtime.second_fused_output_bias
        )

    return core.compile(forward)


def build_mlx_stack_benchmark_operations(
    runtime: MLXPackedTriangularFusedTwoLayerModalStack,
    *,
    batch_sizes: Sequence[int] = (1, 8, 64, 256),
    seed: int = 1701,
    input_standard_deviation: float = 0.05,
) -> dict[int, dict[str, MLXBenchmarkOperation]]:
    """Prepare identical deterministic inputs and the three MLX systems."""

    if not isinstance(
        runtime,
        MLXPackedTriangularFusedTwoLayerModalStack,
    ):
        raise TypeError(
            "runtime must be an MLX packed triangular modal stack"
        )
    normalized_batches = tuple(batch_sizes)
    if not normalized_batches:
        raise ValueError("at least one batch size is required")
    if any(
        type(batch_size) is not int or batch_size <= 0
        for batch_size in normalized_batches
    ):
        raise ValueError("batch sizes must be positive integers")
    if len(set(normalized_batches)) != len(normalized_batches):
        raise ValueError("batch sizes must be unique")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (
        not math.isfinite(input_standard_deviation)
        or input_standard_deviation <= 0
    ):
        raise ValueError(
            "input_standard_deviation must be finite and positive"
        )

    core = runtime._core
    dense_compiled = build_mlx_dense_compiled_forward(runtime)
    core.eval(runtime.first_input_mean)
    core.synchronize()
    input_mean = np.array(runtime.first_input_mean, copy=True)

    operations_by_batch: dict[
        int,
        dict[str, MLXBenchmarkOperation],
    ] = {}
    for batch_size in sorted(normalized_batches):
        random = np.random.default_rng(
            np.random.SeedSequence([seed, batch_size])
        )
        noise = random.standard_normal(
            (
                batch_size,
                runtime.sequence_length,
                runtime.width,
            )
        ).astype(np.float32)
        hidden_states = core.array(
            input_mean[None, :, :]
            + np.float32(input_standard_deviation) * noise,
            dtype=core.float32,
        )
        core.eval(hidden_states)
        core.synchronize()
        operations_by_batch[batch_size] = {
            "mlx_dense_compiled": make_synchronized_mlx_operation(
                core,
                dense_compiled,
                hidden_states,
            ),
            "mlx_packed_compiled": make_synchronized_mlx_operation(
                core,
                runtime._compiled_reference,
                hidden_states,
            ),
            "mlx_packed_metal": make_synchronized_mlx_operation(
                core,
                runtime._compiled_metal,
                hidden_states,
            ),
        }
    return operations_by_batch


def _portable_device_info(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _portable_device_info(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_portable_device_info(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(core: Any) -> dict[str, object]:
    device_info = (
        core.device_info()
        if callable(getattr(core, "device_info", None))
        else core.metal.device_info()
    )
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": platform.release(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "mlx_version": _package_version("mlx"),
        "mlx_metal_version": _package_version("mlx-metal"),
        "default_device": str(core.default_device()),
        "device_info": _portable_device_info(device_info),
    }


def _validate_benchmark_outputs(
    operations: Mapping[int, Mapping[str, MLXBenchmarkOperation]],
) -> dict[str, object]:
    batches: list[dict[str, object]] = []
    passed = True
    for batch_size in sorted(operations):
        outputs = {
            name: np.array(operation(), copy=True)
            for name, operation in operations[batch_size].items()
        }
        reference = outputs["mlx_dense_compiled"]
        expected_shape = tuple(reference.shape)
        systems: dict[str, dict[str, object]] = {}
        for name, output in outputs.items():
            difference = output.astype(np.float64) - reference.astype(
                np.float64
            )
            rms = float(np.sqrt(np.mean(np.square(difference))))
            reference_rms = float(
                np.sqrt(
                    np.mean(
                        np.square(reference.astype(np.float64))
                    )
                )
            )
            relative_rms = (
                rms / reference_rms if reference_rms > 0 else rms
            )
            maximum_absolute_difference = float(
                np.max(np.abs(difference))
            )
            reference_peak = float(np.max(np.abs(reference)))
            peak_relative_difference = (
                maximum_absolute_difference / reference_peak
                if reference_peak > 0
                else maximum_absolute_difference
            )
            finite = bool(np.isfinite(output).all())
            shape_matches = tuple(output.shape) == expected_shape
            dtype_matches = output.dtype == np.float32
            system_passed = bool(
                finite
                and shape_matches
                and dtype_matches
                and relative_rms < 0.005
                and peak_relative_difference < 0.01
            )
            passed = passed and system_passed
            systems[name] = {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "finite": finite,
                "shape_matches_dense": shape_matches,
                "dtype_is_float32": dtype_matches,
                "mean_absolute_difference_from_dense": float(
                    np.mean(np.abs(difference))
                ),
                "rms_difference_from_dense": rms,
                "relative_rms_difference_from_dense": relative_rms,
                "maximum_absolute_difference_from_dense": (
                    maximum_absolute_difference
                ),
                "maximum_difference_relative_to_dense_peak": (
                    peak_relative_difference
                ),
                "gate_passed": system_passed,
            }
        batches.append(
            {
                "batch_size": batch_size,
                "reference_system": "mlx_dense_compiled",
                "systems": systems,
            }
        )
    if not passed:
        raise RuntimeError(
            "MLX benchmark output-equivalence gate failed"
        )
    return {
        "gate": {
            "finite": True,
            "shape_matches_dense": True,
            "dtype": "float32",
            "maximum_relative_rms_difference_from_dense": 0.005,
            "maximum_difference_relative_to_dense_peak": 0.01,
        },
        "gate_passed": True,
        "batches": batches,
    }


def benchmark_mlx_modal_stack(
    runtime: MLXPackedTriangularFusedTwoLayerModalStack,
    *,
    batch_sizes: Sequence[int] = (1, 8, 64, 256),
    seed: int = 1701,
    input_standard_deviation: float = 0.05,
    rounds: int = 9,
    warmup_calls: int = 100,
    minimum_warmup_seconds: float = 1.0,
    iterations_per_round: int = 100,
) -> dict[str, object]:
    """Run and return a portable stack-only MLX/Metal benchmark report."""

    core = runtime._core
    if core.default_device() != core.gpu:
        raise ValueError(
            "MLX benchmark requires the GPU as the default device"
        )
    reset_peak_memory = getattr(core, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()
    operations = build_mlx_stack_benchmark_operations(
        runtime,
        batch_sizes=batch_sizes,
        seed=seed,
        input_standard_deviation=input_standard_deviation,
    )
    reports = benchmark_synchronized_operations(
        operations,
        rounds=rounds,
        warmup_calls=warmup_calls,
        minimum_warmup_seconds=minimum_warmup_seconds,
        iterations_per_round=iterations_per_round,
        speedup_pairs=_DEFAULT_SPEEDUP_PAIRS,
    )
    output_validation = _validate_benchmark_outputs(operations)
    dense_pairs = runtime.sequence_length * runtime.sequence_length
    return {
        "schema": "fisher_graph.mlx_metal_benchmark",
        "format_version": MLX_BENCHMARK_FORMAT_VERSION,
        "benchmark_scope": "two_layer_modal_stack_only",
        "claim_scope": {
            "same_mlx_device_for_all_systems": True,
            "gpu_default_device_enforced": True,
            "benchmark_inputs": "synthetic_mean_centered_gaussian",
            "benchmark_output_equivalence_validated": True,
            "task_validation_included": False,
            "test_split_used": False,
            "weights_updated": False,
            "default_runtime_changed": False,
        },
        "measurement_contract": {
            "first_observed_call_reported_separately": True,
            "first_observed_call_process_isolated": False,
            "first_observed_call_may_include_compilation": True,
            "fresh_lazy_output_per_call": True,
            "completion_barrier": "mx.eval_then_mx.synchronize",
            "steady_state_system_order": "deterministic_rotation",
            "raw_sample_unit": "microseconds_per_call",
            "rounds": rounds,
            "warmup_calls": warmup_calls,
            "minimum_warmup_seconds": minimum_warmup_seconds,
            "iterations_per_round": iterations_per_round,
            "seed": seed,
            "input_standard_deviation": (
                input_standard_deviation
            ),
        },
        "systems": {
            "mlx_dense_compiled": {
                "execution": "mx.compile",
                "kernel_storage": "zero_filled_dense",
                "position_pair_count": dense_pairs,
            },
            "mlx_packed_compiled": {
                "execution": "mx.compile",
                "kernel_storage": "packed_causal",
                "position_pair_count": runtime.causal_pair_count,
            },
            "mlx_packed_metal": {
                "execution": "mx.compile_with_custom_metal_kernel",
                "kernel_storage": "packed_causal",
                "position_pair_count": runtime.causal_pair_count,
            },
        },
        "runtime": mlx_runtime_provenance(runtime),
        "environment": _environment(runtime._core),
        "output_validation": output_validation,
        "memory": {
            "active_bytes": (
                core.get_active_memory()
                if callable(getattr(core, "get_active_memory", None))
                else None
            ),
            "peak_bytes": (
                core.get_peak_memory()
                if callable(getattr(core, "get_peak_memory", None))
                else None
            ),
            "cache_bytes": (
                core.get_cache_memory()
                if callable(getattr(core, "get_cache_memory", None))
                else None
            ),
        },
        "batches": [asdict(report) for report in reports],
    }


def render_mlx_benchmark_markdown(
    report: Mapping[str, object],
) -> str:
    environment = report["environment"]
    assert isinstance(environment, Mapping)
    validation = report["output_validation"]
    assert isinstance(validation, Mapping)
    batches = report["batches"]
    assert isinstance(batches, list)
    runtime = report["runtime"]
    assert isinstance(runtime, Mapping)
    measurement = report["measurement_contract"]
    assert isinstance(measurement, Mapping)

    lines = [
        "# MLX/Metal packed modal-stack benchmark",
        "",
        "This format-1 report compares three stack-only MLX executions on "
        "one GPU. It is an exploratory accelerator measurement. It is not "
        "a task-validation report, a serialized runtime, or a change to "
        "the authenticated default backend.",
        "",
        f"- Device: {environment['device_info']['device_name']}",
        f"- MLX: {environment['mlx_version']}",
        "- Dtype: float32",
        "- Output-equivalence gate: "
        + ("passed" if validation["gate_passed"] else "failed"),
        "- Test split used: no",
        "- Weights updated: no",
        "",
        "## Steady-state stack latency",
        "",
        "| Batch | Dense compiled | Packed compiled | Packed Metal | "
        "Metal vs dense | Metal vs packed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in batches:
        assert isinstance(batch, Mapping)
        timings = batch["timings"]
        speedups = batch["speedup_ratios"]
        assert isinstance(timings, Mapping)
        assert isinstance(speedups, Mapping)

        def median(name: str) -> float:
            timing = timings[name]
            assert isinstance(timing, Mapping)
            return float(timing["median_microseconds"])

        lines.append(
            f"| {batch['batch_size']} "
            f"| {median('mlx_dense_compiled'):.3f} us "
            f"| {median('mlx_packed_compiled'):.3f} us "
            f"| {median('mlx_packed_metal'):.3f} us "
            "| "
            f"{float(speedups['packed_metal_vs_dense_compiled']):.3f}x "
            "| "
            f"{float(speedups['packed_metal_vs_packed_compiled']):.3f}x |"
        )
    lines.extend(
        [
            "",
            "Each timed call creates a fresh lazy result, evaluates it, and "
            "synchronizes the GPU. "
            f"{measurement['rounds']} measurement rounds rotate system "
            "order. First-observed calls are retained in JSON but are not "
            "process-isolated cold-start measurements.",
            "",
            "## Runtime structure",
            "",
            f"- Causal pairs stored: {runtime['causal_pair_count']}",
            f"- MLX stack state: {runtime['state_bytes']:,} bytes",
            "- Custom kernel: `fisher_packed_causal_gelu`",
            "- Custom kernel accumulation: FP32, safe math, no atomics",
            "- Activation capture/differentiation fallback: ordinary MLX "
            "graph",
            "- Activation Fisher oracle: authenticated PyTorch "
            "instrumentation path",
            "",
            "The custom kernel removes gathered-pair temporaries and indexed "
            "reduction. At this toy size, MLX's dense kernels remain highly "
            "competitive despite executing structural zeros, so no hard "
            "latency gate is applied.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark dense, ordinary packed, and custom-Metal MLX "
            "executions of the two-layer modal stack."
        )
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path(
            "artifacts/associative_recall/fused_modal_runtime.pt"
        ),
        help="lazy fused runtime artifact",
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=None,
        help="optional sidecar root; sidecars are not loaded",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 64, 256],
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--warmup-calls", type=int, default=100)
    parser.add_argument(
        "--minimum-warmup-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--iterations-per-round",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path; otherwise print to stdout",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help=(
            "optional Markdown output path; defaults beside --output "
            "when JSON is written"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output is not None and args.output.suffix != ".json":
        raise ValueError("--output must use a .json suffix")
    markdown_output = args.markdown_output
    if args.output is not None and markdown_output is None:
        markdown_output = args.output.with_suffix(".md")
    if (
        args.output is not None
        and markdown_output is not None
        and args.output.resolve() == markdown_output.resolve()
    ):
        raise ValueError("JSON and Markdown output paths must differ")
    lazy, _, _ = load_lazy_fused_modal_stack(
        args.runtime,
        sidecar_root=args.sidecar_root,
    )
    source_status_before = asdict(lazy.instrumentation_status())
    packed = PackedTriangularFusedTwoLayerModalStack.from_lazy(
        lazy
    ).eval()
    runtime = MLXPackedTriangularFusedTwoLayerModalStack.from_torch(
        packed,
        backend="metal",
    )
    report = benchmark_mlx_modal_stack(
        runtime,
        batch_sizes=args.batch_sizes,
        seed=args.seed,
        rounds=args.rounds,
        warmup_calls=args.warmup_calls,
        minimum_warmup_seconds=args.minimum_warmup_seconds,
        iterations_per_round=args.iterations_per_round,
    )
    source_status_after = asdict(lazy.instrumentation_status())
    if source_status_after != source_status_before:
        raise RuntimeError(
            "MLX benchmark unexpectedly touched source instrumentation"
        )
    report["source_artifact"] = {
        "path": str(args.runtime),
        "sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
    }
    report["source_lazy_instrumentation"] = {
        "before": source_status_before,
        "after": source_status_after,
        "unchanged": True,
    }
    serialized = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(
            render_mlx_benchmark_markdown(report)
        )


__all__ = [
    "MLX_BENCHMARK_FORMAT_VERSION",
    "MLX_BENCHMARK_SYSTEMS",
    "MLXBatchTimingReport",
    "MLXTiming",
    "benchmark_mlx_modal_stack",
    "benchmark_synchronized_operations",
    "build_mlx_dense_compiled_forward",
    "build_mlx_stack_benchmark_operations",
    "make_synchronized_mlx_operation",
    "render_mlx_benchmark_markdown",
    "unpack_packed_triangular_kernel",
]


if __name__ == "__main__":
    main()
