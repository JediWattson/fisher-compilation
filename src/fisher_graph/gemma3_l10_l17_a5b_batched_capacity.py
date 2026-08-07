"""Vectorized A5 frozen-affine capacity solving with token-local guards.

This module changes throughput, not the A5a scientific question.  It replays
the exact A4 float64 Euclidean initialization, then optimizes independent
per-token coordinates through a batched final-head projection.  Every token
keeps A5a's own RMS-derived Adam learning rate and independently selects its
KL-best checkpoint, including the initial point as a safe abstention.

The batching optimization is allowed only after the adapter demonstrates that
``project_logits`` is token-local on both the native teacher rows and the A4
baseline rows.  Returned tensors remain ephemeral; receipts contain only
scalars, small row aggregates, and hashes.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .downstream_affine_coordinate_solver import (
    DownstreamAffineSolverConfig,
    solve_batched_downstream_sensitive_affine_coordinates,
)
from .gemma3_l10_l17_a5_frozen_affine_capacity_oracle import (
    FrozenAffineCapacitySolution,
    FrozenAffineImage,
    _DEFAULT_SOLVER,
    _contains_tensor,
    _head_only_context,
    _state_error,
    _tensor_sha256,
)
from .gemma3_a5_same_shape_locality import (
    audit_same_shape_off_row_locality,
    validate_same_shape_off_row_locality_receipt,
)


_DEFAULT_ROW_CHUNK_SIZE = 64
# Float32 CPU matrix kernels may change accumulation order between a multirow
# projection and the same rows projected one at a time.  Sixteen-ish float32
# epsilons is still a strict numerical-equivalence gate while rejecting real
# cross-row coupling by orders of magnitude.
_TOKEN_LOCALITY_ATOL = 1.0e-6
_TOKEN_LOCALITY_RTOL = 2.0e-6
TOKEN_LOCALITY_POLICY_BATCHED_VS_SINGLETONS = "batched_vs_singletons_v1"
TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW = (
    "same_shape_directed_off_row_v2"
)
_TOKEN_LOCALITY_POLICIES = {
    TOKEN_LOCALITY_POLICY_BATCHED_VS_SINGLETONS,
    TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW,
}
_RECEIPT_DOMAIN = b"fisher-graph:a5b-batched-capacity-receipt:v1\0"


def _finite_matrix(value: object) -> bool:
    return bool(
        isinstance(value, Tensor)
        and value.layout is torch.strided
        and value.ndim == 2
        and value.is_floating_point()
        and value.numel() > 0
        and torch.isfinite(value.detach()).all().item()
    )


def _validate_inputs(
    *,
    image: FrozenAffineImage,
    native_state: Tensor,
    compiled_post_attention_residual: Tensor,
    compiled_compact_retained_delta: Tensor,
    target_correction: Tensor,
    a4_float64_projection_correction: Tensor,
    row_chunk_size: int,
    token_locality_atol: float,
    token_locality_rtol: float,
    token_locality_policy: str,
) -> None:
    if not isinstance(image, FrozenAffineImage):
        raise TypeError("image must be a FrozenAffineImage")
    if type(row_chunk_size) is not int or row_chunk_size < 1:
        raise ValueError("row_chunk_size must be a positive integer")
    if (
        not isinstance(token_locality_policy, str)
        or token_locality_policy not in _TOKEN_LOCALITY_POLICIES
    ):
        raise ValueError(
            "token_locality_policy must name a supported locality guard"
        )
    for label, value in (
        ("token_locality_atol", token_locality_atol),
        ("token_locality_rtol", token_locality_rtol),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and nonnegative")

    runtime_values = (
        native_state,
        compiled_post_attention_residual,
        compiled_compact_retained_delta,
        a4_float64_projection_correction,
    )
    if not all(_finite_matrix(value) for value in runtime_values):
        raise ValueError("A5b runtime inputs must be finite floating matrices")
    if (
        not _finite_matrix(target_correction)
        or target_correction.dtype != torch.float64
        or target_correction.device.type != "cpu"
    ):
        raise ValueError(
            "A5b target correction must remain canonical CPU float64"
        )
    if (
        native_state.shape != compiled_post_attention_residual.shape
        or native_state.shape != compiled_compact_retained_delta.shape
        or native_state.shape != target_correction.shape
        or native_state.shape != a4_float64_projection_correction.shape
        or native_state.shape[1] != image.residual_width
        or native_state.device.type != "cpu"
        or native_state.device != compiled_post_attention_residual.device
        or native_state.device != compiled_compact_retained_delta.device
        or native_state.device != a4_float64_projection_correction.device
        or native_state.dtype != compiled_post_attention_residual.dtype
        or native_state.dtype != compiled_compact_retained_delta.dtype
        or native_state.dtype != a4_float64_projection_correction.dtype
    ):
        raise ValueError("A5b state inputs are not aligned")


def _project_logits_rows(
    adapter: Gemma3CausalLMAdapter,
    states: Tensor,
) -> Tensor:
    context = _head_only_context(states.shape[0], states.device)
    logits = adapter.project_logits(states[None, :, :], context)
    if (
        not isinstance(logits, Tensor)
        or logits.layout is not torch.strided
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != states.shape[0]
        or logits.shape[2] < 2
        or not logits.is_floating_point()
        or logits.device != states.device
        or not bool(torch.isfinite(logits.detach()).all().item())
    ):
        raise RuntimeError(
            "adapter.project_logits must return finite [1, rows, vocab] logits"
        )
    return logits[0]


def _project_logits_singletons(
    adapter: Gemma3CausalLMAdapter,
    states: Tensor,
) -> Tensor:
    outputs = [
        _project_logits_rows(adapter, states[index : index + 1])[0]
        for index in range(states.shape[0])
    ]
    widths = {int(output.shape[0]) for output in outputs}
    if len(widths) != 1:
        raise RuntimeError("adapter.project_logits vocabulary width is unstable")
    return torch.stack(outputs, dim=0)


def _comparison_error(left: Tensor, right: Tensor) -> dict[str, float]:
    if left.shape != right.shape:
        raise RuntimeError("batched and singleton head projections misalign")
    difference = left.detach().double() - right.detach().double()
    return {
        "max_abs": float(difference.abs().max().item()),
        "rms": float(torch.sqrt(difference.square().mean()).item()),
        "reference_max_abs": float(right.detach().double().abs().max().item()),
    }


def _measure_token_locality(
    *,
    adapter: Gemma3CausalLMAdapter,
    teacher_rows: Tensor,
    initial_rows: Tensor,
) -> tuple[Tensor, Tensor, dict[str, float], dict[str, float]]:
    """Measure both probes without applying an acceptance threshold."""

    with torch.no_grad():
        teacher_batched = _project_logits_rows(adapter, teacher_rows).detach()
        teacher_singletons = _project_logits_singletons(
            adapter, teacher_rows
        ).detach()
        initial_batched = _project_logits_rows(adapter, initial_rows).detach()
        initial_singletons = _project_logits_singletons(
            adapter, initial_rows
        ).detach()
    return (
        teacher_batched,
        initial_batched,
        _comparison_error(teacher_batched, teacher_singletons),
        _comparison_error(initial_batched, initial_singletons),
    )


def _audit_token_locality(
    *,
    adapter: Gemma3CausalLMAdapter,
    teacher_rows: Tensor,
    initial_rows: Tensor,
    atol: float,
    rtol: float,
) -> tuple[Tensor, dict[str, object]]:
    teacher_batched, _, teacher_error, initial_error = _measure_token_locality(
        adapter=adapter,
        teacher_rows=teacher_rows,
        initial_rows=initial_rows,
    )
    for label, error in (
        ("native teacher", teacher_error),
        ("A4 baseline", initial_error),
    ):
        allowed = atol + rtol * error["reference_max_abs"]
        if error["max_abs"] > allowed:
            raise RuntimeError(
                f"adapter.project_logits is not token-local at the {label} "
                "probe: "
                f"max_abs={error['max_abs']:.17g}, "
                f"rms={error['rms']:.17g}, "
                f"reference_max_abs={error['reference_max_abs']:.17g}, "
                f"allowed={allowed:.17g}"
            )
    return teacher_batched, {
        "method": "batched_projection_vs_same_rows_projected_as_singletons",
        "probe_states": ["native_teacher", "a4_euclidean_baseline"],
        "row_count": int(teacher_rows.shape[0]),
        "nontrivial_multirow_probe": teacher_rows.shape[0] > 1,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "teacher": teacher_error,
        "a4_baseline": initial_error,
        "passed": True,
    }


def _audit_directed_same_shape_token_locality(
    *,
    adapter: Gemma3CausalLMAdapter,
    teacher_rows: Tensor,
    initial_rows: Tensor,
    atol: float,
    rtol: float,
) -> tuple[Tensor, dict[str, object]]:
    """Authorize batching by testing every directed off-row dependency."""

    row_count = int(teacher_rows.shape[0])
    if row_count < 2:
        raise RuntimeError(
            "same-shape directed off-row locality fails closed for a "
            "singleton chunk"
        )
    teacher_audit = audit_same_shape_off_row_locality(
        adapter=adapter,
        rows=teacher_rows,
        probe_name="native_teacher",
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        solver_authorization=True,
    )
    a4_audit = audit_same_shape_off_row_locality(
        adapter=adapter,
        rows=initial_rows,
        probe_name="a4_euclidean_baseline",
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        solver_authorization=True,
    )
    validate_same_shape_off_row_locality_receipt(teacher_audit.receipt)
    validate_same_shape_off_row_locality_receipt(a4_audit.receipt)
    teacher_receipt = dict(teacher_audit.receipt)
    a4_receipt = dict(a4_audit.receipt)
    if (
        _tensor_sha256(teacher_audit.baseline_logits)
        != teacher_receipt["baseline_logits_sha256"]
    ):
        raise RuntimeError(
            "directed token-locality native baseline tensor hash mismatch"
        )
    passed = (
        teacher_receipt["passed"] is True
        and a4_receipt["passed"] is True
    )
    locality: dict[str, object] = {
        "policy": TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW,
        "method": "same_shape_directed_off_row_native_and_a4",
        "probe_states": ["native_teacher", "a4_euclidean_baseline"],
        "row_count": row_count,
        "nontrivial_multirow_probe": True,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "teacher": teacher_receipt,
        "a4_baseline": a4_receipt,
        "directed_receipt_sha256_by_probe": {
            "teacher": teacher_receipt["receipt_sha256"],
            "a4_baseline": a4_receipt["receipt_sha256"],
        },
        "native_teacher_baseline_logits_reused_by_solver": True,
        "changes_solver_authorization": True,
        "passed": passed,
    }
    if _contains_tensor(locality):
        raise RuntimeError("directed token-locality receipt contains a tensor")
    if not passed:
        raise RuntimeError(
            "adapter.project_logits failed the same-shape directed off-row "
            "token-locality guard: "
            f"native_passed={teacher_receipt['passed']}, "
            f"a4_passed={a4_receipt['passed']}, "
            "native_receipt_sha256="
            f"{teacher_receipt['receipt_sha256']}, "
            f"a4_receipt_sha256={a4_receipt['receipt_sha256']}"
        )
    return teacher_audit.baseline_logits, locality


def _same_shape_counterfactual_error(
    *,
    adapter: Gemma3CausalLMAdapter,
    rows: Tensor,
    baseline_logits: Tensor,
) -> dict[str, float] | None:
    """Hold each checked row fixed while changing every peer at fixed shape."""

    row_count = int(rows.shape[0])
    if row_count <= 1:
        return None
    differences: list[Tensor] = []
    for checked in range(row_count):
        counterfactual = rows.roll(shifts=1, dims=0).neg().contiguous()
        offsets = torch.linspace(
            0.125,
            0.125 * row_count,
            row_count,
            device=rows.device,
            dtype=rows.dtype,
        )[:, None]
        counterfactual = (counterfactual + offsets).contiguous()
        counterfactual[checked].copy_(rows[checked])
        if not torch.equal(counterfactual[checked], rows[checked]):
            raise RuntimeError("counterfactual changed its checked row")
        with torch.no_grad():
            changed = _project_logits_rows(adapter, counterfactual).detach()
        differences.append(
            baseline_logits[checked].detach().double()
            - changed[checked].detach().double()
        )
    difference = torch.stack(differences, dim=0)
    return {
        "max_abs": float(difference.abs().max().item()),
        "rms": float(torch.sqrt(difference.square().mean()).item()),
        "reference_max_abs": float(
            baseline_logits.detach().double().abs().max().item()
        ),
    }


def _diagnostic_probe(
    error: Mapping[str, float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    allowed = atol + rtol * float(error["reference_max_abs"])
    max_abs = float(error["max_abs"])
    reference = float(error["reference_max_abs"])
    required_rtol = (
        max(0.0, (max_abs - atol) / reference)
        if reference > 0.0
        else None
    )
    return {
        **dict(error),
        "allowed": allowed,
        "max_abs_minus_allowed": max_abs - allowed,
        "max_abs_over_allowed": max_abs / allowed if allowed > 0.0 else None,
        "required_rtol_at_fixed_atol": required_rtol,
        "passed": max_abs <= allowed,
    }


def _nearest_rank_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty diagnostic sequence")
    ordered = sorted(values)

    def select(probability: float) -> float:
        index = max(0, math.ceil(probability * len(ordered)) - 1)
        return ordered[min(index, len(ordered) - 1)]

    return {
        "minimum": ordered[0],
        "p50": select(0.50),
        "p90": select(0.90),
        "p95": select(0.95),
        "p99": select(0.99),
        "maximum": ordered[-1],
    }


def _diagnostic_aggregate(
    chunks: list[dict[str, object]],
    *,
    probe: str,
) -> dict[str, object]:
    rows: list[tuple[int, Mapping[str, object]]] = []
    for chunk in chunks:
        value = chunk[probe]
        if not isinstance(value, Mapping):
            raise TypeError("token-locality diagnostic probe is unavailable")
        rows.append((int(chunk["chunk_index"]), value))
    ratios = [float(value["max_abs_over_allowed"]) for _, value in rows]
    max_abs_values = [float(value["max_abs"]) for _, value in rows]
    rms_values = [float(value["rms"]) for _, value in rows]
    required_rtols = [
        float(value["required_rtol_at_fixed_atol"])
        for _, value in rows
        if value["required_rtol_at_fixed_atol"] is not None
    ]
    worst_index, worst = max(
        rows, key=lambda item: float(item[1]["max_abs_over_allowed"])
    )
    failing = [index for index, value in rows if value["passed"] is not True]
    return {
        "chunk_count": len(rows),
        "failing_chunk_count": len(failing),
        "failing_chunk_indices": failing,
        "max_abs_distribution": _nearest_rank_quantiles(max_abs_values),
        "rms_distribution": _nearest_rank_quantiles(rms_values),
        "max_abs_over_allowed_distribution": _nearest_rank_quantiles(ratios),
        "required_rtol_at_fixed_atol_distribution": (
            _nearest_rank_quantiles(required_rtols) if required_rtols else None
        ),
        "worst_chunk_index": worst_index,
        "worst_chunk": dict(worst),
        "passed": not failing,
    }


def _counterfactual_aggregate(
    chunks: list[dict[str, object]],
    *,
    probe: str,
) -> dict[str, object]:
    covered: list[dict[str, object]] = []
    uncheckable_rows = 0
    for chunk in chunks:
        counterfactual = chunk["same_shape_peer_counterfactual"]
        if not isinstance(counterfactual, Mapping):
            raise TypeError("same-shape counterfactual receipt is unavailable")
        value = counterfactual[probe]
        if value is None:
            uncheckable_rows += int(chunk["row_count"])
            continue
        if not isinstance(value, Mapping):
            raise TypeError("same-shape counterfactual probe is invalid")
        covered.append(
            {
                "chunk_index": chunk["chunk_index"],
                probe: value,
            }
        )
    if not covered:
        return {
            "checked_chunk_count": 0,
            "checked_row_count": 0,
            "uncheckable_singleton_row_count": uncheckable_rows,
            "failing_chunk_count": 0,
            "failing_chunk_indices": [],
            "max_abs_distribution": None,
            "rms_distribution": None,
            "max_abs_over_allowed_distribution": None,
            "required_rtol_at_fixed_atol_distribution": None,
            "worst_chunk_index": None,
            "worst_chunk": None,
            "passed": True,
        }
    synthetic = [
        {
            "chunk_index": item["chunk_index"],
            probe: item[probe],
        }
        for item in covered
    ]
    aggregate = _diagnostic_aggregate(synthetic, probe=probe)
    aggregate["checked_chunk_count"] = len(covered)
    aggregate["checked_row_count"] = sum(
        int(chunk["row_count"])
        for chunk in chunks
        if isinstance(chunk["same_shape_peer_counterfactual"], Mapping)
        and chunk["same_shape_peer_counterfactual"][probe] is not None  # type: ignore[index]
    )
    aggregate["uncheckable_singleton_row_count"] = uncheckable_rows
    aggregate.pop("chunk_count")
    return aggregate


def diagnose_token_locality_envelope(
    *,
    adapter: Gemma3CausalLMAdapter,
    native_state: Tensor,
    a4_compiled_state: Tensor,
    row_chunk_size: int,
    token_locality_atol: float = _TOKEN_LOCALITY_ATOL,
    token_locality_rtol: float = _TOKEN_LOCALITY_RTOL,
) -> dict[str, object]:
    """Measure the complete native/A4 locality envelope without early exit.

    This is diagnostic-only: a failed probe is recorded for every chunk and
    never converted into solver authorization.  The production solver keeps
    its existing fail-closed per-chunk guard and canonical receipt unchanged.
    """

    if (
        not _finite_matrix(native_state)
        or not _finite_matrix(a4_compiled_state)
        or native_state.shape != a4_compiled_state.shape
        or native_state.dtype != a4_compiled_state.dtype
        or native_state.device != a4_compiled_state.device
    ):
        raise ValueError("diagnostic native and A4 states must be aligned")
    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("diagnostic row_chunk_size must be positive")
    for label, value in (
        ("token_locality_atol", token_locality_atol),
        ("token_locality_rtol", token_locality_rtol),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and nonnegative")
    atol = float(token_locality_atol)
    rtol = float(token_locality_rtol)
    total_rows = int(native_state.shape[0])
    chunks: list[dict[str, object]] = []
    for chunk_index, start in enumerate(range(0, total_rows, row_chunk_size)):
        stop = min(start + row_chunk_size, total_rows)
        (
            teacher_batched,
            a4_batched,
            teacher_error,
            a4_error,
        ) = _measure_token_locality(
            adapter=adapter,
            teacher_rows=native_state[start:stop],
            initial_rows=a4_compiled_state[start:stop],
        )
        teacher_counterfactual = _same_shape_counterfactual_error(
            adapter=adapter,
            rows=native_state[start:stop],
            baseline_logits=teacher_batched,
        )
        a4_counterfactual = _same_shape_counterfactual_error(
            adapter=adapter,
            rows=a4_compiled_state[start:stop],
            baseline_logits=a4_batched,
        )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "row_start": start,
                "row_stop": stop,
                "row_count": stop - start,
                "native_teacher": _diagnostic_probe(
                    teacher_error, atol=atol, rtol=rtol
                ),
                "a4_compiled": _diagnostic_probe(
                    a4_error, atol=atol, rtol=rtol
                ),
                "same_shape_peer_counterfactual": {
                    "method": (
                        "hold_each_checked_row_fixed_while_deterministically_"
                        "changing_all_peer_rows_at_identical_batch_shape"
                    ),
                    "peer_mutation": (
                        "roll_one_row_then_negate_and_add_row_index_offset"
                    ),
                    "checked_row_count": stop - start if stop - start > 1 else 0,
                    "native_teacher": (
                        None
                        if teacher_counterfactual is None
                        else _diagnostic_probe(
                            teacher_counterfactual, atol=atol, rtol=rtol
                        )
                    ),
                    "a4_compiled": (
                        None
                        if a4_counterfactual is None
                        else _diagnostic_probe(
                            a4_counterfactual, atol=atol, rtol=rtol
                        )
                    ),
                },
            }
        )
    singleton_native = _diagnostic_aggregate(chunks, probe="native_teacher")
    singleton_a4 = _diagnostic_aggregate(chunks, probe="a4_compiled")
    counterfactual_native = _counterfactual_aggregate(
        chunks, probe="native_teacher"
    )
    counterfactual_a4 = _counterfactual_aggregate(chunks, probe="a4_compiled")
    result: dict[str, object] = {
        "schema": "fisher_graph.a5b_token_locality_envelope.v1",
        "scientific_role": (
            "diagnostic_only_full_chunk_envelope_not_solver_authorization"
        ),
        "method": "batched_projection_vs_same_rows_projected_as_singletons",
        "row_count": total_rows,
        "row_chunk_size": row_chunk_size,
        "chunk_count": len(chunks),
        "probe_states": ["native_teacher", "a4_compiled"],
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "chunks": chunks,
        "aggregate": {
            "batched_vs_singletons": {
                "native_teacher": singleton_native,
                "a4_compiled": singleton_a4,
            },
            "same_shape_peer_counterfactual": {
                "native_teacher": counterfactual_native,
                "a4_compiled": counterfactual_a4,
            },
        },
        "canonical_singleton_guard_passed": (
            singleton_native["passed"] is True
            and singleton_a4["passed"] is True
        ),
        "same_shape_counterfactual_passed": (
            counterfactual_native["passed"] is True
            and counterfactual_a4["passed"] is True
        ),
        "changes_solver_authorization": False,
        "contains_tensor_payloads": False,
    }
    if _contains_tensor(result):
        raise RuntimeError("token-locality diagnostic contains a tensor")
    return result


def _exact_teacher_kl_by_row(
    teacher_logits: Tensor,
    candidate_logits: Tensor,
) -> Tensor:
    if (
        teacher_logits.shape != candidate_logits.shape
        or teacher_logits.ndim != 2
    ):
        raise ValueError("teacher and candidate logits must be aligned matrices")
    teacher = teacher_logits.detach().to(dtype=torch.float64)
    candidate = candidate_logits.to(dtype=torch.float64)
    teacher_log_probability = teacher - torch.logsumexp(
        teacher, dim=-1, keepdim=True
    )
    candidate_log_probability = candidate - torch.logsumexp(
        candidate, dim=-1, keepdim=True
    )
    return (
        teacher_log_probability.exp()
        * (teacher_log_probability - candidate_log_probability)
    ).sum(dim=-1).clamp_min(0.0)


def _receipt_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_RECEIPT_DOMAIN + encoded).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "sum": float(tensor.sum().item()),
        "mean": float(tensor.mean().item()),
        "minimum": float(tensor.min().item()),
        "median": float(tensor.median().item()),
        "maximum": float(tensor.max().item()),
    }


def solve_batched_frozen_affine_capacity_rows(
    *,
    adapter: Gemma3CausalLMAdapter,
    image: FrozenAffineImage,
    native_state: Tensor,
    compiled_post_attention_residual: Tensor,
    compiled_compact_retained_delta: Tensor,
    target_correction: Tensor,
    a4_float64_projection_correction: Tensor,
    row_chunk_size: int = _DEFAULT_ROW_CHUNK_SIZE,
    solver_config: DownstreamAffineSolverConfig = _DEFAULT_SOLVER,
    token_locality_atol: float = _TOKEN_LOCALITY_ATOL,
    token_locality_rtol: float = _TOKEN_LOCALITY_RTOL,
    token_locality_policy: str = (
        TOKEN_LOCALITY_POLICY_BATCHED_VS_SINGLETONS
    ),
) -> FrozenAffineCapacitySolution:
    """Replay A5a with independent token solves sharing batched head calls."""

    _validate_inputs(
        image=image,
        native_state=native_state,
        compiled_post_attention_residual=compiled_post_attention_residual,
        compiled_compact_retained_delta=compiled_compact_retained_delta,
        target_correction=target_correction,
        a4_float64_projection_correction=a4_float64_projection_correction,
        row_chunk_size=row_chunk_size,
        token_locality_atol=token_locality_atol,
        token_locality_rtol=token_locality_rtol,
        token_locality_policy=token_locality_policy,
    )
    if not isinstance(solver_config, DownstreamAffineSolverConfig):
        raise TypeError("solver_config must be a DownstreamAffineSolverConfig")
    if (
        not math.isfinite(float(solver_config.learning_rate))
        or solver_config.learning_rate <= 0.0
    ):
        raise ValueError("solver learning-rate fraction must be positive")

    # Fixed runtime sources are detached so the only differentiable variables
    # in every chunk are the affine coordinates owned by the generic solver.
    teacher_state = native_state.detach().clone().contiguous()
    post_attention = (
        compiled_post_attention_residual.detach().clone().contiguous()
    )
    retained_delta = (
        compiled_compact_retained_delta.detach().clone().contiguous()
    )

    initial64 = image.euclidean_initial_coordinates(target_correction)
    initial_correction64 = (
        image.mean_sum + initial64 @ image.decoder
    ).contiguous()
    initial_correction = initial_correction64.to(
        device=native_state.device,
        dtype=native_state.dtype,
    ).contiguous()
    if not torch.equal(
        initial_correction, a4_float64_projection_correction
    ):
        raise RuntimeError(
            "A5b Euclidean initialization differs from A4 float64-one-cast "
            "projection"
        )
    initial_state = (
        post_attention + (retained_delta + initial_correction)
    ).contiguous()

    initial_row_rms = torch.sqrt(initial64.square().mean(dim=-1)).contiguous()
    minimum_scale = torch.finfo(torch.float64).eps
    effective_learning_rates = (
        initial_row_rms.clamp_min(minimum_scale)
        * solver_config.learning_rate
    ).contiguous()

    selected_chunks: list[Tensor] = []
    chunk_receipts: list[dict[str, object]] = []
    all_initial_kls: list[float] = []
    all_selected_kls: list[float] = []
    all_selected_steps: list[float] = []
    all_projection_counts: list[float] = []
    total_rows = int(native_state.shape[0])

    for chunk_index, start in enumerate(range(0, total_rows, row_chunk_size)):
        stop = min(start + row_chunk_size, total_rows)
        teacher_rows = teacher_state[start:stop]
        post_attention_rows = post_attention[start:stop]
        retained_rows = retained_delta[start:stop]
        initial_state_rows = initial_state[start:stop]
        if (
            token_locality_policy
            == TOKEN_LOCALITY_POLICY_BATCHED_VS_SINGLETONS
        ):
            teacher_logits, locality = _audit_token_locality(
                adapter=adapter,
                teacher_rows=teacher_rows,
                initial_rows=initial_state_rows,
                atol=float(token_locality_atol),
                rtol=float(token_locality_rtol),
            )
        else:
            teacher_logits, locality = (
                _audit_directed_same_shape_token_locality(
                    adapter=adapter,
                    teacher_rows=teacher_rows,
                    initial_rows=initial_state_rows,
                    atol=float(token_locality_atol),
                    rtol=float(token_locality_rtol),
                )
            )

        def loss_callback(
            candidate_correction: Tensor,
        ) -> Mapping[str, Tensor]:
            runtime_correction = candidate_correction.to(
                device=post_attention_rows.device,
                dtype=post_attention_rows.dtype,
            )
            candidate_rows = post_attention_rows + (
                retained_rows + runtime_correction
            )
            candidate_logits = _project_logits_rows(adapter, candidate_rows)
            kl = _exact_teacher_kl_by_row(teacher_logits, candidate_logits)
            return {"loss": kl, "kl": kl}

        solution = solve_batched_downstream_sensitive_affine_coordinates(
            image.mean_sum,
            image.decoder,
            initial64[start:stop],
            loss_callback,
            config=solver_config,
            learning_rate_by_row=effective_learning_rates[start:stop],
        )
        selected_chunks.append(solution.coordinates)
        generic_receipt = solution.receipt
        initial_kls = [
            float(value) for value in generic_receipt["per_row_initial_kl"]
        ]
        selected_kls = [
            float(value) for value in generic_receipt["per_row_selected_kl"]
        ]
        selected_steps = [
            float(value) for value in generic_receipt["per_row_selected_steps"]
        ]
        projection_counts = [
            float(value)
            for value in generic_receipt["per_row_trust_projection_counts"]
        ]
        all_initial_kls.extend(initial_kls)
        all_selected_kls.extend(selected_kls)
        all_selected_steps.extend(selected_steps)
        all_projection_counts.extend(projection_counts)
        chunk_receipts.append(
            {
                "chunk_index": chunk_index,
                "row_start": start,
                "row_stop": stop,
                "row_count": stop - start,
                "initial_kl": _summary(initial_kls),
                "selected_kl": _summary(selected_kls),
                "selected_step": _summary(selected_steps),
                "trust_projection_count": _summary(projection_counts),
                "token_locality": locality,
                "full_solver_receipt_sha256": _receipt_sha256(
                    generic_receipt
                ),
            }
        )

    selected64 = torch.cat(selected_chunks, dim=0).contiguous()
    selected_correction64 = (
        image.mean_sum + selected64 @ image.decoder
    ).contiguous()
    selected_correction = selected_correction64.to(
        device=native_state.device,
        dtype=native_state.dtype,
    ).contiguous()
    selected_state = (
        post_attention + (retained_delta + selected_correction)
    ).contiguous()
    initial_kl = _summary(all_initial_kls)
    selected_kl = _summary(all_selected_kls)

    receipt: dict[str, object] = {
        "schema": "fisher_graph.gemma3_l10_l17_a5b_batched_capacity.v1",
        "objective": (
            "independent_per_token_exact_native_to_candidate_kl_through_"
            "adapter_project_logits"
        ),
        "scientific_method": "a5a_frozen_affine_capacity_oracle",
        "throughput_change_only": True,
        "teacher_boundary": "captured_native_layer17_output",
        "candidate_formula": (
            "compiled_post_attention_plus_parenthesized_exact_compact_delta_"
            "plus_sum_frozen_means_plus_coefficient_times_frozen_decoder"
        ),
        "initialization": "float64_affine_sum_svd_pseudoinverse_minimum_norm",
        "canonical_target_dtype": "torch.float64",
        "affine_arithmetic_dtype": "torch.float64",
        "coordinate_layout": "joint_concatenated_four_node_rank_182",
        "runtime_correction_dtype": str(native_state.dtype),
        "runtime_correction_cast_count_per_materialization": 1,
        "initial_correction_bit_identical_to_a4_float64_one_cast": True,
        "row_count": total_rows,
        "row_chunk_size": row_chunk_size,
        "chunk_count": len(chunk_receipts),
        "batching": {
            "one_batched_head_callback_per_optimizer_evaluation": True,
            "independent_adam_parameter_group_per_token": True,
            "independent_kl_best_checkpoint_per_token": True,
            "token_locality_audited_on_native_and_a4_states": True,
            "token_locality_absolute_tolerance": float(token_locality_atol),
            "token_locality_relative_tolerance": float(token_locality_rtol),
            **(
                {"token_locality_policy": token_locality_policy}
                if token_locality_policy
                == TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW
                else {}
            ),
        },
        "solver": {
            "steps": solver_config.steps,
            "learning_rate_fraction_of_per_token_initial_coefficient_rms": (
                solver_config.learning_rate
            ),
            "minimum_scale_for_zero_rms": minimum_scale,
            "initial_coefficient_rms": _summary(
                [float(value) for value in initial_row_rms.tolist()]
            ),
            "effective_learning_rate": _summary(
                [float(value) for value in effective_learning_rates.tolist()]
            ),
            "scale_is_independent_for_each_token": True,
            "ridge": solver_config.ridge,
            "trust_radius": solver_config.trust_radius,
            "initial_point_evaluated_as_safe_abstention": True,
        },
        "initial_kl": initial_kl,
        "selected_kl": selected_kl,
        "absolute_mean_kl_improvement": (
            initial_kl["mean"] - selected_kl["mean"]
        ),
        "selected_not_worse_than_initial_for_every_token": all(
            selected <= initial
            for selected, initial in zip(
                all_selected_kls, all_initial_kls, strict=True
            )
        ),
        "selected_step": _summary(all_selected_steps),
        "trust_projection_count": _summary(all_projection_counts),
        "initial_state_error": _state_error(initial_state, teacher_state),
        "selected_state_error": _state_error(selected_state, teacher_state),
        "hashes": {
            "initial_coefficient_sha256": _tensor_sha256(initial64),
            "selected_coefficient_sha256": _tensor_sha256(selected64),
            "initial_correction_sha256": _tensor_sha256(initial_correction),
            "selected_correction_sha256": _tensor_sha256(
                selected_correction
            ),
            "initial_state_sha256": _tensor_sha256(initial_state),
            "selected_state_sha256": _tensor_sha256(selected_state),
        },
        "chunk_receipts": chunk_receipts,
        "frozen_affine_membership_by_construction": True,
        "basis_mean_or_decoder_changed": False,
        "deployable_generator_fitted": False,
        "contains_tensor_payloads": False,
    }
    if _contains_tensor(receipt):
        raise RuntimeError("A5b receipt contains a tensor payload")
    return FrozenAffineCapacitySolution(
        initial_coefficients=initial64,
        selected_coefficients=selected64,
        initial_correction=initial_correction,
        selected_correction=selected_correction,
        initial_state=initial_state,
        selected_state=selected_state,
        receipt=receipt,
    )


__all__ = [
    "TOKEN_LOCALITY_POLICY_BATCHED_VS_SINGLETONS",
    "TOKEN_LOCALITY_POLICY_SAME_SHAPE_DIRECTED_OFF_ROW",
    "diagnose_token_locality_envelope",
    "solve_batched_frozen_affine_capacity_rows",
]
