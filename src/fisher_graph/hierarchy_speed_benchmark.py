"""Shape-only latency probe for prepared hierarchical connectivity factors.

This benchmark deliberately sits below model quality evaluation.  It compares
three executions of the same authenticated boundary decomposition:

* the exact dense source transfer ``H``;
* the retained candidate materialized as one dense matrix ``P @ R``; and
* the retained candidate executed in its compact factorized form ``R -> P``.

The materialized candidate is a control: it has the same approximation as the
factorized candidate but the same dense matrix-multiplication geometry as the
source.  Comparing the two candidates therefore isolates the latency effect of
low-rank execution from the effect of changing the numerical operator.

The synthetic CLI does not establish downstream fidelity, Gemma replacement
authority, deployed storage savings, or end-to-end model speed.  It only finds
the width/rank/row-count region where a validate-once prepared kernel can beat
an optimized dense boundary matrix on the measured device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .fused_benchmark import benchmark_batch_sizes
from .mlx_benchmark import (
    benchmark_synchronized_operations,
)
from .modal_connectivity_modes import (
    CausalBoundaryTransfer,
    MessageMoments,
    ModalBoundaryPort,
    ModalConnectivityDecomposition,
    factor_modal_connectivity,
)
from .prepared_hierarchy_runtime import PreparedTorchHierarchyRuntime


HIERARCHY_SPEED_BENCHMARK_FORMAT_VERSION = 1
HIERARCHY_SPEED_SYSTEMS = (
    "source_dense",
    "candidate_dense",
    "candidate_factorized",
)
HIERARCHY_SPEEDUP_PAIRS = {
    "candidate_dense_vs_source_dense": (
        "source_dense",
        "candidate_dense",
    ),
    "candidate_factorized_vs_source_dense": (
        "source_dense",
        "candidate_factorized",
    ),
    "candidate_factorized_vs_candidate_dense": (
        "candidate_dense",
        "candidate_factorized",
    ),
}
BenchmarkBackend = Literal["torch", "mlx", "both"]


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _require_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_rank(
    value: object,
    *,
    maximum: int,
) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"retained rank must lie in [0, {maximum}]")
    return value


def build_synthetic_connectivity_decomposition(
    *,
    input_width: int,
    output_width: int,
    retained_rank: int | None = None,
    spectrum_floor: float = 0.01,
) -> ModalConnectivityDecomposition:
    """Build a deterministic one-input/one-output benchmark decomposition.

    The source is a dense-stored rectangular diagonal matrix with a geometric,
    strictly descending singular spectrum.  Dense kernels still process all
    entries, while the known spectrum keeps rank-energy accounting stable and
    avoids making the latency probe depend on random SVD conditioning.
    """

    input_width = _require_positive_integer(
        input_width,
        label="input_width",
    )
    output_width = _require_positive_integer(
        output_width,
        label="output_width",
    )
    spectrum_rank = min(input_width, output_width)
    if retained_rank is not None:
        _require_rank(retained_rank, maximum=spectrum_rank)
    if (
        not isinstance(spectrum_floor, float)
        or not math.isfinite(spectrum_floor)
        or not 0.0 < spectrum_floor < 1.0
    ):
        raise ValueError("spectrum_floor must lie in (0, 1)")

    graph_id = (
        f"synthetic-hierarchy-speed-{input_width}x{output_width}"
    )
    input_port = ModalBoundaryPort(
        name=f"{graph_id}.input",
        direction="input",
        causal_order=0,
        width=input_width,
        owner_id=graph_id,
    )
    output_port = ModalBoundaryPort(
        name=f"{graph_id}.output",
        direction="output",
        causal_order=0,
        width=output_width,
        owner_id=graph_id,
    )
    singular_values = torch.logspace(
        0.0,
        math.log10(spectrum_floor),
        spectrum_rank,
        dtype=torch.float64,
    )
    matrix = torch.zeros(
        (output_width, input_width),
        dtype=torch.float64,
    )
    diagonal = torch.arange(spectrum_rank)
    matrix[diagonal, diagonal] = singular_values
    offset = torch.linspace(
        -0.05,
        0.05,
        output_width,
        dtype=torch.float64,
    )
    source_digest = _sha256(f"{graph_id}:source")
    transfer = CausalBoundaryTransfer(
        source_level_sha256=source_digest,
        input_ports=(input_port,),
        output_ports=(output_port,),
        input_prefixes=((input_port.name,),),
        transfer_matrices=(matrix,),
        affine_offsets=(offset,),
    )
    reduction_id = "synthetic-hierarchy-speed"
    input_moments = MessageMoments(
        port=input_port,
        source_level_sha256=source_digest,
        reduction_id=reduction_id,
        sample_count=1024,
        mean=torch.zeros(input_width, dtype=torch.float64),
        covariance=torch.eye(input_width, dtype=torch.float64),
        fisher=torch.eye(input_width, dtype=torch.float64),
    )
    output_moments = MessageMoments(
        port=output_port,
        source_level_sha256=source_digest,
        reduction_id=reduction_id,
        sample_count=1024,
        mean=offset,
        covariance=torch.eye(output_width, dtype=torch.float64),
        fisher=torch.eye(output_width, dtype=torch.float64),
    )
    return factor_modal_connectivity(
        transfer,
        (input_moments,),
        (output_moments,),
        retained_ranks=retained_rank,
    )


def hierarchy_speed_accounting(
    decomposition: ModalConnectivityDecomposition,
) -> dict[str, object]:
    """Return exact arithmetic and compact prepared-state accounting."""

    if not isinstance(decomposition, ModalConnectivityDecomposition):
        raise TypeError(
            "decomposition must be ModalConnectivityDecomposition"
        )
    decomposition.validate_integrity()
    source_macs = decomposition.source_transfer.macs_per_row
    candidate_macs = decomposition.candidate_macs_per_row
    source_scalars = decomposition.source_transfer.stored_scalar_count
    direct_candidate_scalars = (
        decomposition.candidate_stored_scalar_count
    )
    prepared_candidate_scalars = sum(
        factor.restriction.numel()
        + factor.prolongation.numel()
        + factor.output_port.width
        for factor in decomposition.factors
    )
    retained_ranks = tuple(
        factor.retained_rank for factor in decomposition.factors
    )
    arithmetic_speedup = (
        None if candidate_macs == 0 else source_macs / candidate_macs
    )
    total_energy = decomposition.total_weighted_energy
    retained_fraction = (
        1.0
        if total_energy == 0.0
        else decomposition.retained_weighted_energy / total_energy
    )
    return {
        "source_dense_macs_per_row": source_macs,
        "candidate_factorized_macs_per_row": candidate_macs,
        "ideal_arithmetic_speedup": arithmetic_speedup,
        "candidate_has_fewer_macs": candidate_macs < source_macs,
        "source_dense_stored_scalars": source_scalars,
        "direct_candidate_stored_scalars": direct_candidate_scalars,
        "prepared_candidate_stored_scalars": prepared_candidate_scalars,
        "prepared_candidate_has_fewer_scalars": (
            prepared_candidate_scalars < source_scalars
        ),
        "retained_ranks": retained_ranks,
        "retained_weighted_energy_fraction": retained_fraction,
    }


def _numpy_inputs(
    decomposition: ModalConnectivityDecomposition,
    *,
    row_count: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed + row_count * 1009)
    return tuple(
        generator.normal(
            loc=0.0,
            scale=0.25,
            size=(row_count, port.width),
        ).astype(np.float32)
        for port in decomposition.source_transfer.input_ports
    )


def make_synchronized_mlx_chain_operation(
    core: Any,
    forward: Any,
    inputs: tuple[Any, ...],
    *,
    stages_per_call: int,
) -> Any:
    """Execute a dependency chain with one completion barrier.

    A one-stage operation measures standalone boundary invocation latency.
    A deeper chain models several prepared generators embedded in one lazy
    model traversal, where only the outer traversal synchronizes.  Outputs
    feed the next stage, preventing dead-work elimination.
    """

    stages_per_call = _require_positive_integer(
        stages_per_call,
        label="stages_per_call",
    )
    if not callable(forward):
        raise TypeError("forward must be callable")
    if not callable(getattr(core, "eval", None)):
        raise TypeError("core must provide eval")
    if not callable(getattr(core, "synchronize", None)):
        raise TypeError("core must provide synchronize")
    if type(inputs) is not tuple or not inputs:
        raise ValueError("MLX chain inputs must be a nonempty tuple")

    def operation() -> tuple[Any, ...]:
        state = inputs
        for _ in range(stages_per_call):
            state = tuple(forward(state))
        core.eval(state)
        core.synchronize()
        return state

    return operation


def _torch_output_validation(
    runtime: PreparedTorchHierarchyRuntime,
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    with torch.inference_mode():
        source = runtime.source_dense(inputs)
        candidate_dense = runtime.candidate_dense(inputs)
        candidate_factorized = runtime.candidate_factorized(inputs)
    maximum_candidate_difference = max(
        float(
            torch.max(torch.abs(dense - factorized)).item()
            if dense.numel()
            else 0.0
        )
        for dense, factorized in zip(
            candidate_dense,
            candidate_factorized,
            strict=True,
        )
    )
    candidate_peak = max(
        float(torch.max(torch.abs(value)).item() if value.numel() else 0.0)
        for value in candidate_dense
    )
    candidate_square = math.fsum(
        float(value.double().square().sum().item())
        for value in candidate_dense
    )
    candidate_difference_square = math.fsum(
        float(
            (factorized - dense).double().square().sum().item()
        )
        for dense, factorized in zip(
            candidate_dense,
            candidate_factorized,
            strict=True,
        )
    )
    candidate_relative_rms = math.sqrt(
        candidate_difference_square
        / max(candidate_square, torch.finfo(torch.float64).tiny)
    )
    source_square = math.fsum(
        float(value.double().square().sum().item()) for value in source
    )
    difference_square = math.fsum(
        float(
            (candidate - exact).double().square().sum().item()
        )
        for candidate, exact in zip(
            candidate_factorized,
            source,
            strict=True,
        )
    )
    relative_rms = math.sqrt(
        difference_square / max(source_square, torch.finfo(torch.float64).tiny)
    )
    gate_passed = (
        candidate_relative_rms <= 5e-4
        and maximum_candidate_difference
        <= 5e-5 + 5e-4 * candidate_peak
    )
    if not gate_passed:
        raise RuntimeError(
            "Torch factorized candidate does not match its dense control: "
            f"relative RMS {candidate_relative_rms:.6g}, maximum "
            f"difference {maximum_candidate_difference:.6g}, candidate "
            f"peak {candidate_peak:.6g}"
        )
    return {
        "gate_passed": True,
        "maximum_absolute_factorized_vs_candidate_dense": (
            maximum_candidate_difference
        ),
        "relative_rms_factorized_vs_candidate_dense": (
            candidate_relative_rms
        ),
        "maximum_relative_rms_gate": 5e-4,
        "maximum_difference_gate": "5e-5 + 5e-4 * candidate_peak",
        "candidate_vs_source_relative_rms": relative_rms,
        "candidate_vs_source_is_quality_gate": False,
    }


def benchmark_prepared_torch_decomposition(
    decomposition: ModalConnectivityDecomposition,
    *,
    row_counts: Sequence[int],
    seed: int = 1701,
    repeats: int = 7,
    minimum_block_seconds: float = 0.05,
    warmup_iterations: int = 20,
    minimum_warmup_seconds: float = 0.2,
) -> dict[str, object]:
    """Benchmark prepared float32 CPU dense and factorized executions."""

    started = time.perf_counter_ns()
    runtime = PreparedTorchHierarchyRuntime.from_decomposition(
        decomposition,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    preparation_microseconds = (time.perf_counter_ns() - started) / 1e3
    rows = tuple(
        _require_positive_integer(value, label="row count")
        for value in row_counts
    )
    if len(rows) != len(set(rows)):
        raise ValueError("row counts must be unique")
    operations: dict[int, dict[str, Any]] = {}
    validation_inputs: tuple[torch.Tensor, ...] | None = None
    for row_count in rows:
        arrays = _numpy_inputs(
            decomposition,
            row_count=row_count,
            seed=seed,
        )
        inputs = tuple(torch.from_numpy(value) for value in arrays)
        if validation_inputs is None:
            validation_inputs = inputs
        operations[row_count] = {
            "source_dense": (
                lambda inputs=inputs: runtime.source_dense(inputs)
            ),
            "candidate_dense": (
                lambda inputs=inputs: runtime.candidate_dense(inputs)
            ),
            "candidate_factorized": (
                lambda inputs=inputs: runtime.candidate_factorized(inputs)
            ),
        }
    assert validation_inputs is not None
    validation = _torch_output_validation(runtime, validation_inputs)
    reports = benchmark_batch_sizes(
        operations,
        repeats=repeats,
        minimum_block_seconds=minimum_block_seconds,
        warmup_iterations=warmup_iterations,
        minimum_warmup_seconds=minimum_warmup_seconds,
        speedup_pairs=HIERARCHY_SPEEDUP_PAIRS,
    )
    return {
        "backend": "torch_cpu",
        "preparation_microseconds": preparation_microseconds,
        "runtime": {
            "source_transfer_sha256": runtime.source_transfer_sha256,
            "decomposition_sha256": runtime.decomposition_sha256,
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "input_names": runtime.input_names,
            "input_widths": runtime.input_widths,
            "output_names": runtime.output_names,
            "output_widths": runtime.output_widths,
            "retained_ranks": runtime.retained_ranks,
            "accounting": asdict(runtime.accounting),
        },
        "output_validation": validation,
        "measurement_contract": {
            "torch_inference_mode": True,
            "intraop_threads": 1,
            "steady_state_system_order": "deterministic_rotation",
            "raw_sample_unit": "microseconds_per_call",
            "repeats": repeats,
            "minimum_block_seconds": minimum_block_seconds,
            "warmup_iterations": warmup_iterations,
            "minimum_warmup_seconds": minimum_warmup_seconds,
        },
        "rows": [
            {
                "row_count": report.batch_size,
                "timings": {
                    name: asdict(value)
                    for name, value in report.timings.items()
                },
                "rows_per_second": report.examples_per_second,
                "speedup_ratios": report.speedup_ratios,
                "round_orders": report.round_orders,
            }
            for report in reports
        ],
    }


def _mlx_package_version() -> str:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _portable_runtime_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _portable_runtime_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_portable_runtime_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def benchmark_prepared_mlx_decomposition(
    decomposition: ModalConnectivityDecomposition,
    *,
    row_counts: Sequence[int],
    seed: int = 1701,
    rounds: int = 7,
    warmup_calls: int = 20,
    minimum_warmup_seconds: float = 0.2,
    iterations_per_round: int = 50,
    stages_per_synchronized_call: int = 1,
) -> dict[str, object]:
    """Benchmark prepared float32 MLX dense and factorized executions."""

    from .mlx_hierarchy_runtime import PreparedMLXHierarchyRuntime

    import mlx.core as mx

    if mx.default_device() != mx.gpu:
        raise RuntimeError("MLX hierarchy benchmark requires the GPU")
    stages_per_synchronized_call = _require_positive_integer(
        stages_per_synchronized_call,
        label="stages_per_synchronized_call",
    )
    started = time.perf_counter_ns()
    runtime = PreparedMLXHierarchyRuntime.from_decomposition(
        decomposition
    )
    preparation_microseconds = (time.perf_counter_ns() - started) / 1e3
    rows = tuple(
        _require_positive_integer(value, label="row count")
        for value in row_counts
    )
    if len(rows) != len(set(rows)):
        raise ValueError("row counts must be unique")
    if stages_per_synchronized_call > 1 and (
        runtime.input_widths != runtime.output_widths
        or len(runtime.input_widths) != len(runtime.output_widths)
    ):
        raise ValueError(
            "multi-stage MLX chains require matching input/output widths"
        )

    operations: dict[int, dict[str, Any]] = {}
    validation_inputs: tuple[Any, ...] | None = None
    for row_count in rows:
        arrays = _numpy_inputs(
            decomposition,
            row_count=row_count,
            seed=seed,
        )
        inputs = tuple(mx.array(value) for value in arrays)
        if validation_inputs is None:
            validation_inputs = inputs
        operations[row_count] = {
            "source_dense": make_synchronized_mlx_chain_operation(
                mx,
                runtime.source_dense,
                inputs,
                stages_per_call=stages_per_synchronized_call,
            ),
            "candidate_dense": make_synchronized_mlx_chain_operation(
                mx,
                runtime.candidate_dense,
                inputs,
                stages_per_call=stages_per_synchronized_call,
            ),
            "candidate_factorized": make_synchronized_mlx_chain_operation(
                mx,
                runtime.candidate_factorized,
                inputs,
                stages_per_call=stages_per_synchronized_call,
            ),
        }
    assert validation_inputs is not None
    source = runtime.source_dense(validation_inputs)
    candidate_dense = runtime.candidate_dense(validation_inputs)
    candidate_factorized = runtime.candidate_factorized(validation_inputs)
    mx.eval(source, candidate_dense, candidate_factorized)
    mx.synchronize()
    source_numpy = tuple(np.asarray(value) for value in source)
    candidate_dense_numpy = tuple(
        np.asarray(value) for value in candidate_dense
    )
    candidate_factorized_numpy = tuple(
        np.asarray(value) for value in candidate_factorized
    )
    maximum_candidate_difference = max(
        float(np.max(np.abs(dense - factorized), initial=0.0))
        for dense, factorized in zip(
            candidate_dense_numpy,
            candidate_factorized_numpy,
            strict=True,
        )
    )
    candidate_peak = max(
        float(np.max(np.abs(value), initial=0.0))
        for value in candidate_dense_numpy
    )
    candidate_square = math.fsum(
        float(np.square(value.astype(np.float64)).sum())
        for value in candidate_dense_numpy
    )
    candidate_difference_square = math.fsum(
        float(
            np.square(
                factorized.astype(np.float64) - dense.astype(np.float64)
            ).sum()
        )
        for dense, factorized in zip(
            candidate_dense_numpy,
            candidate_factorized_numpy,
            strict=True,
        )
    )
    candidate_relative_rms = math.sqrt(
        candidate_difference_square
        / max(candidate_square, np.finfo(np.float64).tiny)
    )
    source_square = math.fsum(
        float(np.square(value.astype(np.float64)).sum())
        for value in source_numpy
    )
    difference_square = math.fsum(
        float(
            np.square(
                candidate.astype(np.float64) - exact.astype(np.float64)
            ).sum()
        )
        for candidate, exact in zip(
            candidate_factorized_numpy,
            source_numpy,
            strict=True,
        )
    )
    if (
        candidate_relative_rms > 5e-3
        or maximum_candidate_difference
        > 5e-5 + 1e-2 * candidate_peak
    ):
        raise RuntimeError(
            "MLX factorized candidate does not match its dense control: "
            f"relative RMS {candidate_relative_rms:.6g}, maximum "
            f"difference {maximum_candidate_difference:.6g}, candidate "
            f"peak {candidate_peak:.6g}"
        )
    reports = benchmark_synchronized_operations(
        operations,
        rounds=rounds,
        warmup_calls=warmup_calls,
        minimum_warmup_seconds=minimum_warmup_seconds,
        iterations_per_round=iterations_per_round,
        speedup_pairs=HIERARCHY_SPEEDUP_PAIRS,
    )
    return {
        "backend": "mlx_gpu",
        "stages_per_synchronized_call": stages_per_synchronized_call,
        "preparation_microseconds": preparation_microseconds,
        "environment": {
            "mlx_version": _mlx_package_version(),
            "default_device": str(mx.default_device()),
            "device_info": _portable_runtime_value(
                (
                    mx.device_info()
                    if callable(getattr(mx, "device_info", None))
                    else mx.metal.device_info()
                )
            ),
        },
        "runtime": runtime.runtime_provenance(),
        "output_validation": {
            "gate_passed": True,
            "maximum_absolute_factorized_vs_candidate_dense": (
                maximum_candidate_difference
            ),
            "relative_rms_factorized_vs_candidate_dense": (
                candidate_relative_rms
            ),
            "maximum_relative_rms_gate": 5e-3,
            "maximum_difference_gate": (
                "5e-5 + 1e-2 * candidate_peak"
            ),
            "candidate_vs_source_relative_rms": math.sqrt(
                difference_square
                / max(source_square, np.finfo(np.float64).tiny)
            ),
            "candidate_vs_source_is_quality_gate": False,
        },
        "measurement_contract": {
            "first_observed_call_reported_separately": True,
            "fresh_lazy_output_per_call": True,
            "completion_barrier": "mx.eval_then_mx.synchronize",
            "steady_state_system_order": "deterministic_rotation",
            "raw_sample_unit": "microseconds_per_call",
            "rounds": rounds,
            "warmup_calls": warmup_calls,
            "minimum_warmup_seconds": minimum_warmup_seconds,
            "iterations_per_round": iterations_per_round,
            "stages_per_synchronized_call": (
                stages_per_synchronized_call
            ),
            "stage_dependency": "output_feeds_next_input",
        },
        "rows": [
            {
                "row_count": report.batch_size,
                "timings": {
                    name: {
                        **asdict(value),
                        "median_microseconds_per_stage": (
                            value.median_microseconds
                            / stages_per_synchronized_call
                        ),
                    }
                    for name, value in report.timings.items()
                },
                "boundary_rows_per_second": {
                    name: value * stages_per_synchronized_call
                    for name, value in (
                        report.examples_per_second.items()
                    )
                },
                "speedup_ratios": report.speedup_ratios,
                "round_orders": report.round_orders,
            }
            for report in reports
        ],
    }


def run_hierarchy_speed_benchmark(
    *,
    input_width: int = 640,
    output_width: int = 640,
    retained_ranks: Sequence[int] = (80, 160, 256, 320),
    row_counts: Sequence[int] = (1, 8, 32, 128, 512),
    backend: BenchmarkBackend = "torch",
    seed: int = 1701,
    torch_repeats: int = 7,
    torch_minimum_block_seconds: float = 0.05,
    torch_warmup_iterations: int = 20,
    torch_minimum_warmup_seconds: float = 0.2,
    mlx_rounds: int = 7,
    mlx_warmup_calls: int = 20,
    mlx_minimum_warmup_seconds: float = 0.2,
    mlx_iterations_per_round: int = 50,
    mlx_chain_depths: Sequence[int] = (1, 18),
) -> dict[str, object]:
    """Run the synthetic rank/row latency ladder."""

    if backend not in {"torch", "mlx", "both"}:
        raise ValueError("backend must be torch, mlx, or both")
    input_width = _require_positive_integer(
        input_width,
        label="input_width",
    )
    output_width = _require_positive_integer(
        output_width,
        label="output_width",
    )
    maximum_rank = min(input_width, output_width)
    ranks = tuple(
        _require_rank(value, maximum=maximum_rank)
        for value in retained_ranks
    )
    if not ranks or len(ranks) != len(set(ranks)):
        raise ValueError("retained ranks must be nonempty and unique")
    rows = tuple(
        _require_positive_integer(value, label="row count")
        for value in row_counts
    )
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("row counts must be nonempty and unique")
    chain_depths = tuple(
        _require_positive_integer(value, label="MLX chain depth")
        for value in mlx_chain_depths
    )
    if not chain_depths or len(chain_depths) != len(set(chain_depths)):
        raise ValueError("MLX chain depths must be nonempty and unique")

    build_started = time.perf_counter_ns()
    full = build_synthetic_connectivity_decomposition(
        input_width=input_width,
        output_width=output_width,
    )
    factorization_microseconds = (
        time.perf_counter_ns() - build_started
    ) / 1e3
    cases: list[dict[str, object]] = []
    for rank in ranks:
        decomposition = (
            full if rank == maximum_rank else full.truncate(rank)
        )
        case: dict[str, object] = {
            "retained_rank": rank,
            "decomposition_sha256": decomposition.artifact_sha256,
            "accounting": hierarchy_speed_accounting(decomposition),
            "backends": {},
        }
        backend_reports = case["backends"]
        assert isinstance(backend_reports, dict)
        if backend in {"torch", "both"}:
            backend_reports["torch"] = benchmark_prepared_torch_decomposition(
                decomposition,
                row_counts=rows,
                seed=seed,
                repeats=torch_repeats,
                minimum_block_seconds=torch_minimum_block_seconds,
                warmup_iterations=torch_warmup_iterations,
                minimum_warmup_seconds=(
                    torch_minimum_warmup_seconds
                ),
            )
        if backend in {"mlx", "both"}:
            for depth in chain_depths:
                backend_reports[
                    f"mlx_sync_{depth}"
                ] = benchmark_prepared_mlx_decomposition(
                    decomposition,
                    row_counts=rows,
                    seed=seed,
                    rounds=mlx_rounds,
                    warmup_calls=mlx_warmup_calls,
                    minimum_warmup_seconds=(
                        mlx_minimum_warmup_seconds
                    ),
                    iterations_per_round=(
                        mlx_iterations_per_round
                    ),
                    stages_per_synchronized_call=depth,
                )
        cases.append(case)

    return {
        "schema": "fisher_graph.hierarchy_speed_benchmark",
        "format_version": HIERARCHY_SPEED_BENCHMARK_FORMAT_VERSION,
        "benchmark_scope": (
            "synthetic_single_boundary_connectivity_kernel"
        ),
        "claim_scope": {
            "shape_only": True,
            "prepared_validate_once_runtime": True,
            "source_fallback_removed_from_timed_hot_paths": True,
            "task_validation_included": False,
            "real_gemma_messages_used": False,
            "model_level_latency_measured": False,
            "replacement_authorized": False,
            "deployed_storage_reduction_claimed": False,
        },
        "shape": {
            "input_width": input_width,
            "output_width": output_width,
            "row_counts": rows,
            "retained_ranks": ranks,
            "spectrum": "geometric_1_to_0.01",
        },
        "factorization_microseconds": factorization_microseconds,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mlx": (
                _mlx_package_version()
                if backend in {"mlx", "both"}
                else None
            ),
            "mlx_chain_depths": (
                chain_depths if backend in {"mlx", "both"} else ()
            ),
        },
        "cases": cases,
    }


def render_hierarchy_speed_benchmark_markdown(
    report: Mapping[str, object],
) -> str:
    """Render the portable JSON report as a compact audit table."""

    shape = report["shape"]
    assert isinstance(shape, Mapping)
    cases = report["cases"]
    assert isinstance(cases, list)
    mlx_device: str | None = None
    mlx_version: str | None = None
    for case in cases:
        assert isinstance(case, Mapping)
        backends = case["backends"]
        assert isinstance(backends, Mapping)
        for backend_name, backend in backends.items():
            if not str(backend_name).startswith("mlx_"):
                continue
            assert isinstance(backend, Mapping)
            environment = backend.get("environment")
            if not isinstance(environment, Mapping):
                continue
            device_info = environment.get("device_info")
            if isinstance(device_info, Mapping):
                candidate = device_info.get("device_name")
                if isinstance(candidate, str):
                    mlx_device = candidate
            version = environment.get("mlx_version")
            if isinstance(version, str):
                mlx_version = version
            break
        if mlx_device is not None:
            break
    lines = [
        "# Prepared hierarchy speed probe",
        "",
        (
            "This is a synthetic single-boundary kernel benchmark at "
            f"`{shape['input_width']} -> {shape['output_width']}` width. "
            "It is not a Gemma quality or end-to-end latency result."
        ),
        "",
    ]
    if mlx_device is not None:
        version_text = (
            f" with MLX {mlx_version}" if mlx_version is not None else ""
        )
        lines.extend(
            [
                f"Recorded GPU: `{mlx_device}`{version_text}.",
                "",
            ]
        )
    lines.extend(
        [
        (
            "| Backend | Rank | Rows | Retained Fisher energy | "
            "Candidate state vs dense | Ideal MAC speedup | "
            "Factorized vs source | "
            "Factorized vs dense candidate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in cases:
        assert isinstance(case, Mapping)
        accounting = case["accounting"]
        backends = case["backends"]
        assert isinstance(accounting, Mapping)
        assert isinstance(backends, Mapping)
        ideal = accounting["ideal_arithmetic_speedup"]
        ideal_text = "unbounded" if ideal is None else f"{float(ideal):.3f}x"
        energy = float(
            accounting["retained_weighted_energy_fraction"]
        )
        state_fraction = (
            int(accounting["prepared_candidate_stored_scalars"])
            / int(accounting["source_dense_stored_scalars"])
        )
        for backend_name, backend_report in backends.items():
            assert isinstance(backend_report, Mapping)
            rows = backend_report["rows"]
            assert isinstance(rows, list)
            for row in rows:
                assert isinstance(row, Mapping)
                speedups = row["speedup_ratios"]
                assert isinstance(speedups, Mapping)
                lines.append(
                    "| "
                    f"{backend_name} | {case['retained_rank']} | "
                    f"{row['row_count']} | {energy:.3%} | "
                    f"{state_fraction:.3%} | {ideal_text} | "
                    f"{float(speedups['candidate_factorized_vs_source_dense']):.3f}x | "
                    f"{float(speedups['candidate_factorized_vs_candidate_dense']):.3f}x |"
                )
    lines.extend(
        [
            "",
            (
                "Ratios above `1.0x` mean the factorized candidate was "
                "faster. The dense-candidate control represents the same "
                "truncated operator as the factorized path, so their ratio "
                "isolates low-rank execution geometry."
            ),
            "",
            (
                "`mlx_sync_1` synchronizes each standalone boundary call. "
                "`mlx_sync_18` executes an 18-stage dependency chain and "
                "synchronizes only at the outer traversal; its recorded "
                "timings are normalized per stage."
            ),
            "",
            (
                "The benchmark removes proof verification, artifact hashing, "
                "per-call dtype/device copies, fallback execution, and Fisher "
                "error measurement from the timed hot path. Those remain "
                "load-time or validation concerns, not free runtime work."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark prepared dense and factorized hierarchy kernels."
        )
    )
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument(
        "--retained-ranks",
        type=int,
        nargs="+",
        default=[80, 160, 256, 320],
    )
    parser.add_argument(
        "--row-counts",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512],
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "mlx", "both"),
        default="torch",
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--torch-repeats", type=int, default=7)
    parser.add_argument(
        "--torch-minimum-block-seconds",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--torch-warmup-iterations",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--torch-minimum-warmup-seconds",
        type=float,
        default=0.2,
    )
    parser.add_argument("--mlx-rounds", type=int, default=7)
    parser.add_argument("--mlx-warmup-calls", type=int, default=20)
    parser.add_argument(
        "--mlx-minimum-warmup-seconds",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--mlx-iterations-per-round",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--mlx-chain-depths",
        type=int,
        nargs="+",
        default=[1, 18],
        help=(
            "number of dependent prepared stages per outer GPU "
            "synchronization"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path; otherwise print JSON",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="optional Markdown output path",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output is not None and args.output.suffix != ".json":
        raise ValueError("--output must use a .json suffix")
    markdown_output = args.markdown_output
    if args.output is not None and markdown_output is None:
        markdown_output = args.output.with_suffix(".md")
    report = run_hierarchy_speed_benchmark(
        input_width=args.input_width,
        output_width=args.output_width,
        retained_ranks=args.retained_ranks,
        row_counts=args.row_counts,
        backend=args.backend,
        seed=args.seed,
        torch_repeats=args.torch_repeats,
        torch_minimum_block_seconds=(
            args.torch_minimum_block_seconds
        ),
        torch_warmup_iterations=args.torch_warmup_iterations,
        torch_minimum_warmup_seconds=(
            args.torch_minimum_warmup_seconds
        ),
        mlx_rounds=args.mlx_rounds,
        mlx_warmup_calls=args.mlx_warmup_calls,
        mlx_minimum_warmup_seconds=(
            args.mlx_minimum_warmup_seconds
        ),
        mlx_iterations_per_round=args.mlx_iterations_per_round,
        mlx_chain_depths=args.mlx_chain_depths,
    )
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
            render_hierarchy_speed_benchmark_markdown(report)
        )


__all__ = [
    "HIERARCHY_SPEED_BENCHMARK_FORMAT_VERSION",
    "HIERARCHY_SPEED_SYSTEMS",
    "HIERARCHY_SPEEDUP_PAIRS",
    "benchmark_prepared_mlx_decomposition",
    "benchmark_prepared_torch_decomposition",
    "build_synthetic_connectivity_decomposition",
    "hierarchy_speed_accounting",
    "make_synchronized_mlx_chain_operation",
    "render_hierarchy_speed_benchmark_markdown",
    "run_hierarchy_speed_benchmark",
]


if __name__ == "__main__":
    main()
