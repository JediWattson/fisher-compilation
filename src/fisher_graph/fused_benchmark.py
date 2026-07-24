"""Deterministic CPU microbenchmarks for interchangeable layer executors."""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeAlias

import torch

BenchmarkOperation: TypeAlias = Callable[[], object]
SpeedupPair: TypeAlias = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Timing:
    """Steady-state timing for one system at one batch size."""

    repeats: int
    iterations_per_block: int
    target_block_seconds: float
    calibration_seconds: float
    warmup_calls: int
    raw_microseconds: tuple[float, ...]
    median_microseconds: float
    minimum_microseconds: float
    maximum_microseconds: float
    p10_microseconds: float
    p90_microseconds: float

@dataclass(frozen=True, slots=True)
class BatchTimingReport:
    """Timings and relative speedups for one batch size."""

    batch_size: int
    timings: dict[str, Timing]
    examples_per_second: dict[str, float]
    speedup_ratios: dict[str, float]
    round_orders: tuple[tuple[str, ...], ...]


@contextmanager
def one_torch_thread() -> Iterator[None]:
    """Temporarily select one PyTorch intra-op thread and restore it safely.

    PyTorch's inter-op thread count cannot in general be changed after parallel
    work has started, so this context deliberately leaves it untouched. A
    benchmark process that needs a fixed inter-op count should set it once,
    before invoking this utility.
    """

    previous_threads = torch.get_num_threads()
    try:
        if previous_threads != 1:
            torch.set_num_threads(1)
        yield
    finally:
        if torch.get_num_threads() != previous_threads:
            torch.set_num_threads(previous_threads)


def _validate_options(
    *,
    repeats: int,
    minimum_block_seconds: float,
    warmup_iterations: int,
    minimum_warmup_seconds: float,
    maximum_iterations_per_block: int,
) -> None:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if (
        not math.isfinite(minimum_block_seconds)
        or minimum_block_seconds <= 0
    ):
        raise ValueError("minimum_block_seconds must be finite and positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")
    if (
        not math.isfinite(minimum_warmup_seconds)
        or minimum_warmup_seconds < 0
    ):
        raise ValueError(
            "minimum_warmup_seconds must be finite and nonnegative"
        )
    if maximum_iterations_per_block <= 0:
        raise ValueError("maximum_iterations_per_block must be positive")


def _validate_operations(
    operations_by_batch: Mapping[
        int,
        Mapping[str, BenchmarkOperation],
    ],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not operations_by_batch:
        raise ValueError("at least one benchmark batch is required")
    batch_sizes = tuple(sorted(operations_by_batch))
    if any(
        not isinstance(batch_size, int) or batch_size <= 0
        for batch_size in batch_sizes
    ):
        raise ValueError("benchmark batch sizes must be positive integers")
    first_operations = operations_by_batch[batch_sizes[0]]
    if not first_operations:
        raise ValueError("at least one benchmark system is required")
    system_names = tuple(first_operations)
    if any(not name for name in system_names):
        raise ValueError("benchmark system names cannot be empty")
    if len(set(system_names)) != len(system_names):
        raise ValueError("benchmark system names must be unique")
    expected_names = set(system_names)
    for batch_size in batch_sizes:
        operations = operations_by_batch[batch_size]
        if set(operations) != expected_names:
            raise ValueError(
                "every batch size must define the same benchmark systems"
            )
        if any(not callable(operation) for operation in operations.values()):
            raise TypeError("benchmark operations must be callable")
    return batch_sizes, system_names


def _warmup(
    operation: BenchmarkOperation,
    *,
    minimum_iterations: int,
    minimum_seconds: float,
) -> int:
    calls = 0
    started = time.perf_counter_ns()
    while True:
        elapsed_seconds = (time.perf_counter_ns() - started) / 1e9
        if calls >= minimum_iterations and elapsed_seconds >= minimum_seconds:
            return calls
        operation()
        calls += 1


def _run_block(
    operation: BenchmarkOperation,
    iterations: int,
) -> float:
    garbage_collection_enabled = gc.isenabled()
    if garbage_collection_enabled:
        gc.disable()
    try:
        started = time.perf_counter_ns()
        for _ in range(iterations):
            operation()
        return (time.perf_counter_ns() - started) / 1e9
    finally:
        if garbage_collection_enabled:
            gc.enable()


def _calibrate_iterations(
    operation: BenchmarkOperation,
    *,
    minimum_block_seconds: float,
    maximum_iterations: int,
) -> tuple[int, float]:
    iterations = 1
    while True:
        elapsed_seconds = _run_block(operation, iterations)
        if elapsed_seconds >= minimum_block_seconds:
            return iterations, elapsed_seconds
        if iterations > maximum_iterations // 2:
            raise RuntimeError(
                "could not reach minimum_block_seconds before "
                "maximum_iterations_per_block"
            )
        iterations *= 2


def _percentile(sorted_values: tuple[float, ...], fraction: float) -> float:
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


def _summarize_timing(
    *,
    iterations: int,
    target_block_seconds: float,
    calibration_seconds: float,
    warmup_calls: int,
    block_seconds: list[float],
) -> Timing:
    raw_microseconds = tuple(
        seconds * 1e6 / iterations for seconds in block_seconds
    )
    ordered = tuple(sorted(raw_microseconds))
    return Timing(
        repeats=len(raw_microseconds),
        iterations_per_block=iterations,
        target_block_seconds=target_block_seconds,
        calibration_seconds=calibration_seconds,
        warmup_calls=warmup_calls,
        raw_microseconds=raw_microseconds,
        median_microseconds=statistics.median(raw_microseconds),
        minimum_microseconds=ordered[0],
        maximum_microseconds=ordered[-1],
        p10_microseconds=_percentile(ordered, 0.1),
        p90_microseconds=_percentile(ordered, 0.9),
    )


def _default_speedup_pairs(
    system_names: tuple[str, ...],
) -> dict[str, SpeedupPair]:
    names = set(system_names)
    pairs: dict[str, SpeedupPair] = {}
    if {"unfused", "fused"} <= names:
        pairs["fused_vs_unfused"] = ("unfused", "fused")
    if {"teacher", "fused"} <= names:
        pairs["fused_vs_teacher"] = ("teacher", "fused")
    return pairs


def _validate_speedup_pairs(
    pairs: Mapping[str, SpeedupPair],
    *,
    system_names: tuple[str, ...],
) -> None:
    known_systems = set(system_names)
    for label, pair in pairs.items():
        if not label:
            raise ValueError("speedup labels cannot be empty")
        if len(pair) != 2:
            raise ValueError(
                "speedup pairs must contain reference and candidate names"
            )
        reference, candidate = pair
        if reference not in known_systems or candidate not in known_systems:
            raise ValueError(
                "speedup pairs must refer to benchmark system names"
            )
        if reference == candidate:
            raise ValueError(
                "speedup reference and candidate must be different"
            )


def benchmark_batch_sizes(
    operations_by_batch: Mapping[
        int,
        Mapping[str, BenchmarkOperation],
    ],
    *,
    repeats: int = 9,
    minimum_block_seconds: float = 0.2,
    warmup_iterations: int = 100,
    minimum_warmup_seconds: float = 1.0,
    maximum_iterations_per_block: int = 1 << 30,
    speedup_pairs: Mapping[str, SpeedupPair] | None = None,
) -> tuple[BatchTimingReport, ...]:
    """Benchmark CPU operations with deterministic rotating system order.

    Each zero-argument operation should close over already-prepared inputs.
    Input construction, artifact loading, and compilation therefore remain
    outside the steady-state measurement.

    A speedup pair is ``(reference, candidate)`` and is reported as
    ``reference_median / candidate_median``; values greater than one mean that
    the candidate was faster.
    """

    _validate_options(
        repeats=repeats,
        minimum_block_seconds=minimum_block_seconds,
        warmup_iterations=warmup_iterations,
        minimum_warmup_seconds=minimum_warmup_seconds,
        maximum_iterations_per_block=maximum_iterations_per_block,
    )
    batch_sizes, system_names = _validate_operations(operations_by_batch)
    pairs = (
        _default_speedup_pairs(system_names)
        if speedup_pairs is None
        else dict(speedup_pairs)
    )
    _validate_speedup_pairs(pairs, system_names=system_names)

    reports: list[BatchTimingReport] = []
    with one_torch_thread(), torch.inference_mode():
        for batch_size in batch_sizes:
            operations = operations_by_batch[batch_size]
            warmup_calls: dict[str, int] = {}
            iterations: dict[str, int] = {}
            calibration_seconds: dict[str, float] = {}
            for name in system_names:
                operation = operations[name]
                warmup_calls[name] = _warmup(
                    operation,
                    minimum_iterations=warmup_iterations,
                    minimum_seconds=minimum_warmup_seconds,
                )
                (
                    iterations[name],
                    calibration_seconds[name],
                ) = _calibrate_iterations(
                    operation,
                    minimum_block_seconds=minimum_block_seconds,
                    maximum_iterations=maximum_iterations_per_block,
                )

            measurements = {name: [] for name in system_names}
            round_orders: list[tuple[str, ...]] = []
            for repeat in range(repeats):
                offset = repeat % len(system_names)
                order = (
                    system_names[offset:] + system_names[:offset]
                )
                round_orders.append(order)
                for name in order:
                    measurements[name].append(
                        _run_block(
                            operations[name],
                            iterations[name],
                        )
                    )

            timings = {
                name: _summarize_timing(
                    iterations=iterations[name],
                    target_block_seconds=minimum_block_seconds,
                    calibration_seconds=calibration_seconds[name],
                    warmup_calls=warmup_calls[name],
                    block_seconds=measurements[name],
                )
                for name in system_names
            }
            examples_per_second = {
                name: (
                    batch_size
                    * 1e6
                    / timing.median_microseconds
                )
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
                BatchTimingReport(
                    batch_size=batch_size,
                    timings=timings,
                    examples_per_second=examples_per_second,
                    speedup_ratios=speedup_ratios,
                    round_orders=tuple(round_orders),
                )
            )
    return tuple(reports)
