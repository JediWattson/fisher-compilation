"""Generic rotating benchmark for prepared full-model runtime operations."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

ModelRuntimeOperation: TypeAlias = Callable[[], object]
PrepareSystem: TypeAlias = Callable[[str], object]
Synchronize: TypeAlias = Callable[[], object]
PerfCounter: TypeAlias = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ModelRuntimeTiming:
    """Cold and steady-state timings for one prepared model runtime."""

    cold_seconds: float
    raw_round_seconds: tuple[float, ...]
    median_seconds: float
    p10_seconds: float
    p90_seconds: float
    p95_seconds: float
    processed_token_count: int
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class ModelRuntimeBenchmarkReport:
    """Rotating benchmark results for one exact set of runtime systems."""

    system_names: tuple[str, ...]
    rounds: int
    warmup_calls: int
    processed_token_count: int
    timings: dict[str, ModelRuntimeTiming]
    round_orders: tuple[tuple[str, ...], ...]


def _require_integer(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        relation = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{label} must be a {relation} integer")
    return value


def _validate_systems(
    operations: Mapping[str, ModelRuntimeOperation],
    *,
    expected_systems: Sequence[str] | None,
) -> tuple[tuple[str, ...], dict[str, ModelRuntimeOperation]]:
    if not isinstance(operations, Mapping) or not operations:
        raise ValueError("at least one model runtime system is required")

    copied = dict(operations)
    if any(
        not isinstance(name, str) or not name
        for name in copied
    ):
        raise ValueError("model runtime system names must be nonempty strings")
    if any(not callable(operation) for operation in copied.values()):
        raise TypeError("model runtime operations must be callable")

    if expected_systems is None:
        return tuple(copied), copied
    if isinstance(expected_systems, (str, bytes)):
        raise TypeError("expected_systems must be a sequence of names")

    ordered = tuple(expected_systems)
    if (
        not ordered
        or any(
            not isinstance(name, str) or not name
            for name in ordered
        )
        or len(set(ordered)) != len(ordered)
    ):
        raise ValueError(
            "expected_systems must contain unique nonempty names"
        )
    if set(copied) != set(ordered):
        raise ValueError(
            "model runtime operations must match the exact expected system set"
        )
    return ordered, copied


def _elapsed_seconds(
    operation: ModelRuntimeOperation,
    *,
    system_name: str,
    phase: str,
    synchronize: Synchronize | None,
    perf_counter: PerfCounter,
) -> float:
    started = perf_counter()
    operation()
    if synchronize is not None:
        synchronize()
    elapsed = perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(
            f"{system_name} {phase} time must be finite and positive"
        )
    return elapsed


def _percentile(
    ordered_values: tuple[float, ...],
    fraction: float,
) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered_values[lower]
    weight = position - lower
    return (
        ordered_values[lower] * (1.0 - weight)
        + ordered_values[upper] * weight
    )


def benchmark_model_runtimes(
    operations: Mapping[str, ModelRuntimeOperation],
    *,
    processed_token_count: int,
    rounds: int = 9,
    warmup_calls: int = 3,
    expected_systems: Sequence[str] | None = None,
    prepare_system: PrepareSystem | None = None,
    synchronize: Synchronize | None = None,
    perf_counter: PerfCounter = time.perf_counter,
) -> ModelRuntimeBenchmarkReport:
    """Benchmark named, already-prepared model runtime operations.

    Each operation is called once for a separately reported cold measurement,
    ``warmup_calls`` times outside the steady-state timer, and once in every
    measured round. System order rotates deterministically between rounds.

    ``prepare_system(name)`` runs before every operation and before the timer
    starts. This lets callers switch prepared model stacks or construct a KV
    cache without charging that work to model execution. If ``synchronize``
    is supplied, it runs once after preparation outside the timer and once
    after the operation; the latter remains inside cold and steady timing.
    """

    rounds = _require_integer(rounds, label="rounds", minimum=1)
    warmup_calls = _require_integer(
        warmup_calls,
        label="warmup_calls",
        minimum=0,
    )
    processed_token_count = _require_integer(
        processed_token_count,
        label="processed_token_count",
        minimum=1,
    )
    if prepare_system is not None and not callable(prepare_system):
        raise TypeError("prepare_system must be callable")
    if synchronize is not None and not callable(synchronize):
        raise TypeError("synchronize must be callable")
    if not callable(perf_counter):
        raise TypeError("perf_counter must be callable")

    system_names, copied_operations = _validate_systems(
        operations,
        expected_systems=expected_systems,
    )

    def prepare(name: str) -> None:
        if prepare_system is not None:
            prepare_system(name)
        # Preparation may enqueue asynchronous accelerator work (for example,
        # building the source-specific KV cache for a decode benchmark).
        # Complete it before starting the measured operation.
        if synchronize is not None:
            synchronize()

    cold_seconds: dict[str, float] = {}
    for name in system_names:
        prepare(name)
        cold_seconds[name] = _elapsed_seconds(
            copied_operations[name],
            system_name=name,
            phase="cold",
            synchronize=synchronize,
            perf_counter=perf_counter,
        )

    for name in system_names:
        for _ in range(warmup_calls):
            prepare(name)
            copied_operations[name]()
            if synchronize is not None:
                synchronize()

    measurements = {name: [] for name in system_names}
    round_orders: list[tuple[str, ...]] = []
    for round_index in range(rounds):
        offset = round_index % len(system_names)
        order = system_names[offset:] + system_names[:offset]
        round_orders.append(order)
        for name in order:
            prepare(name)
            measurements[name].append(
                _elapsed_seconds(
                    copied_operations[name],
                    system_name=name,
                    phase=f"round {round_index}",
                    synchronize=synchronize,
                    perf_counter=perf_counter,
                )
            )

    timings: dict[str, ModelRuntimeTiming] = {}
    for name in system_names:
        raw = tuple(measurements[name])
        ordered = tuple(sorted(raw))
        median = statistics.median(raw)
        timings[name] = ModelRuntimeTiming(
            cold_seconds=cold_seconds[name],
            raw_round_seconds=raw,
            median_seconds=median,
            p10_seconds=_percentile(ordered, 0.10),
            p90_seconds=_percentile(ordered, 0.90),
            p95_seconds=_percentile(ordered, 0.95),
            processed_token_count=processed_token_count,
            tokens_per_second=processed_token_count / median,
        )

    if set(timings) != set(system_names):
        raise RuntimeError("benchmark result does not contain the exact systems")

    return ModelRuntimeBenchmarkReport(
        system_names=system_names,
        rounds=rounds,
        warmup_calls=warmup_calls,
        processed_token_count=processed_token_count,
        timings=timings,
        round_orders=tuple(round_orders),
    )


__all__ = [
    "ModelRuntimeBenchmarkReport",
    "ModelRuntimeOperation",
    "ModelRuntimeTiming",
    "benchmark_model_runtimes",
]
