"""Locked V10 float64-objective replay of the V9 scalar-to-joint H4 path.

V10 changes exactly one numerical policy relative to the pinned V9 run:
teacher and candidate logit rows are promoted to float64 before teacher-KL
arithmetic.  The V6-scalar endpoint, V7-joint endpoint, and four GL4 H4 nodes
must otherwise replay the exact V9/V8 executions.  The finite float64 delta is
also evaluated directly as ``p_T * (log p_scalar - log p_joint)`` so it does
not inherit subtraction of two already-rounded float32 KL values.

This is a same-A diagnostic.  It performs no fitting, selection, search,
damping, fallback, routing, serving mutation, or compression authorization.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic as v3diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic as v7diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_finite_diagnostic as v8diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_scalar_joint_path_attribution_diagnostic as v9diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic as v6diag
from . import gemma3_l3_l4_complete_h4_one_pass_transfer as transfer_support
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from .complete_h4_tail_candidate_joint_state_gain_finite import (
    CandidateConditionedK64ThreeArmGainSupport,
    build_candidate_conditioned_k64_three_arm_gain_support,
)
from .complete_h4_tail_candidate_joint_state_objective_precision import (
    CLOSURE_COSINE_MINIMUM,
    DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE,
    FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM,
    OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateObjectivePrecisionComparison,
    CandidateJointStateObjectivePrecisionEvidence,
    summarize_candidate_joint_state_objective_precision,
)
from .complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathAccumulator,
)
from .complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    fit_complete_h4_tail_held_family,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _require_sha256,
    _runtime_tensor_sha256,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_V9_REPORT",
    "V9_REPORT_FILE_SHA256",
    "V9_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_objective_precision_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v9diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v9diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v9diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v9diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v9diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v9diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v9diag.DEFAULT_V6_REPORT
DEFAULT_V7_REPORT = v9diag.DEFAULT_V7_REPORT
DEFAULT_V8_REPORT = v9diag.DEFAULT_V8_REPORT
DEFAULT_V9_REPORT = v9diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-objective-precision-gl4-"
    "lofo-a-fit16-dev-v10.json"
)

V9_REPORT_FILE_SHA256 = (
    "24343d0d4bc633d59a436ebeed95787ea7fdc83c42540b35d63a21b5d1063ed6"
)
V9_REPORT_SHA256 = (
    "00d51133dd579a26198f763d6c21b0c89d47e834c6bdac64e91d583e2858055e"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_objective_precision_gl4_lofo.v10"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-objective-precision:v10\0"
_V9_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-v9-binding:v10\0"
_ENDPOINT_REPLAY_DOMAIN = b"fisher-graph:complete-h4-k64-precision-endpoint:v10\0"
_F32_OBJECTIVE_DOMAIN = b"fisher-graph:complete-h4-k64-f32-objective:v10\0"
_F64_OBJECTIVE_DOMAIN = b"fisher-graph:complete-h4-k64-f64-objective:v10\0"
_NODE_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-precision-node:v10\0"
_NODE_SET_DOMAIN = b"fisher-graph:complete-h4-k64-precision-node-set:v10\0"

_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS = 16
_EXPECTED_NODES_PER_PROMPT = 4
_EXPECTED_NODE_FORWARDS = 64
_EXPECTED_LINEAGE_FORWARDS = 112
_EXPECTED_LINEAGE_BACKWARDS = 494
_EXPECTED_FRESH_FORWARDS = 112
_EXPECTED_FRESH_BACKWARDS = 545
_EXPECTED_TOTAL_FORWARDS = 224
_EXPECTED_TOTAL_BACKWARDS = 1039
_EXPECTED_ROW_BANK_SUPPORT_EXECUTIONS = 2842
_EXPECTED_FRESH_SUPPORT_EXECUTIONS = 4914
_EXPECTED_TOTAL_SUPPORT_EXECUTIONS = 7756


@dataclass(frozen=True, slots=True)
class _PinnedV9Receipts:
    endpoints: Mapping[str, Mapping[str, object]]
    nodes: Mapping[tuple[str, int], Mapping[str, object]]
    evidence_artifacts: Mapping[str, str]


@dataclass(slots=True)
class _PromptPrecisionResult:
    evidence: CandidateJointStateObjectivePrecisionEvidence
    endpoint_receipt: Mapping[str, object]
    node_receipts: tuple[Mapping[str, object], ...]
    resources: Mapping[str, int]


@dataclass(slots=True)
class _LivePrecisionResult:
    evidence: tuple[CandidateJointStateObjectivePrecisionEvidence, ...]
    comparison: CandidateJointStateObjectivePrecisionComparison
    endpoint_receipts: tuple[Mapping[str, object], ...]
    node_receipts: tuple[Mapping[str, object], ...]
    node_receipt_set_sha256: str
    resources: Mapping[str, int]


def _canonical(value: object) -> object:
    return v3diag._canonical(value)


def _load_v9_report(path: Path | str) -> dict[str, object]:
    """Load only the exact inspected V9 closure-unresolved artifact."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V9_REPORT_FILE_SHA256,
        expected_report_sha256=V9_REPORT_SHA256,
        label="candidate joint-state objective precision V9 anchor",
    )
    resources = report.get("resources")
    comparison = report.get("family_equal_scalar_joint_path_attribution")
    if (
        report.get("schema") != v9diag._SCHEMA
        or report.get("classification")
        != "scalar_joint_path_closure_unresolved_same_a"
        or report.get("passed") is not False
        or not isinstance(resources, Mapping)
        or resources.get("total_model_forward_count") != 240
        or resources.get("total_backward_call_count") != 1148
        or resources.get("total_candidate_support_row_executions") != 8575
        or not isinstance(comparison, Mapping)
        or float(comparison.get("closure_relative_rmse", -1.0))
        != 0.089016283613215
        or float(comparison.get("closure_cosine", -1.0))
        != 0.9960743421828786
    ):
        raise RuntimeError("candidate joint-state path V9 anchor differs")
    _index_v9_receipts(report)
    return report


def _index_v9_receipts(report: Mapping[str, object]) -> _PinnedV9Receipts:
    """Authenticate V9 endpoint, finalized evidence, and four-node receipts."""

    raw_endpoints = report.get("endpoint_pair_receipts")
    raw_nodes = report.get("path_node_receipts")
    summary = report.get("family_equal_scalar_joint_path_attribution")
    if (
        not isinstance(raw_endpoints, list)
        or len(raw_endpoints) != _EXPECTED_PROMPTS
        or not isinstance(raw_nodes, list)
        or len(raw_nodes) != _EXPECTED_NODE_FORWARDS
        or not isinstance(summary, Mapping)
    ):
        raise ValueError("pinned V9 receipt grid differs")
    example_ids = summary.get("evidence_example_ids")
    evidence_hashes = summary.get("evidence_artifact_sha256s")
    if (
        not isinstance(example_ids, list)
        or not isinstance(evidence_hashes, list)
        or len(example_ids) != _EXPECTED_PROMPTS
        or len(evidence_hashes) != _EXPECTED_PROMPTS
    ):
        raise ValueError("pinned V9 finalized evidence index differs")
    evidence_artifacts = {
        token_v1._identifier(example, label="pinned V9 evidence example"): (
            _require_sha256(value, label="pinned V9 evidence artifact")
        )
        for example, value in zip(example_ids, evidence_hashes, strict=True)
    }
    endpoints: dict[str, Mapping[str, object]] = {}
    families: dict[str, str] = {}
    for raw in raw_endpoints:
        if not isinstance(raw, Mapping):
            raise ValueError("pinned V9 endpoint receipt differs")
        row = dict(raw)
        example = token_v1._identifier(
            row.get("example_id"), label="pinned V9 endpoint example"
        )
        family = token_v1._identifier(
            row.get("family_id"), label="pinned V9 endpoint family"
        )
        artifact = _require_sha256(
            row.get("artifact_sha256"), label="pinned V9 endpoint pair"
        )
        base = dict(row)
        base.pop("artifact_sha256")
        for key in (
            "core_evidence_artifact_sha256",
            "core_scalar_tangent_receipt_sha256",
            "core_held_unit_tangent_receipt_sha256",
            "additive_prompt_scalars",
            "path_node_receipt_sha256s",
        ):
            if key not in base:
                raise ValueError("pinned V9 endpoint extension differs")
            base.pop(key)
        if (
            artifact
            != token_v1._domain_sha256(base, domain=v9diag._ENDPOINT_PAIR_DOMAIN)
            or example in endpoints
            or row.get("core_evidence_artifact_sha256")
            != evidence_artifacts.get(example)
            or row.get("actual_cast_once_endpoint_pair") is not True
            or row.get("v8_scalar_and_joint_token_KL_hashes_replayed") is not True
        ):
            raise RuntimeError("pinned V9 endpoint pair receipt drifted")
        scalar = row.get("scalar_endpoint")
        joint = row.get("joint_endpoint")
        if (
            not isinstance(scalar, Mapping)
            or not isinstance(joint, Mapping)
            or scalar.get("realized_endpoint_h4_dtype") != "torch.float32"
            or joint.get("realized_endpoint_h4_dtype") != "torch.float32"
            or scalar.get("endpoint_supervised_token_count")
            != joint.get("endpoint_supervised_token_count")
            or not isinstance(scalar.get("endpoint_supervised_token_count"), int)
            or int(scalar["endpoint_supervised_token_count"]) <= 0
        ):
            raise ValueError("pinned V9 endpoint dtype/token geometry differs")
        endpoints[example] = row
        families[example] = family

    nodes: dict[tuple[str, int], Mapping[str, object]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise ValueError("pinned V9 node receipt differs")
        row = dict(raw)
        receipt = _require_sha256(
            row.pop("receipt_sha256", None), label="pinned V9 node receipt"
        )
        if receipt != token_v1._domain_sha256(
            row, domain=v9diag._PATH_NODE_RECEIPT_DOMAIN
        ):
            raise RuntimeError("pinned V9 path node receipt drifted")
        row["receipt_sha256"] = receipt
        example = token_v1._identifier(
            row.get("example_id"), label="pinned V9 node example"
        )
        node_index = row.get("node_index")
        if (
            example not in endpoints
            or type(node_index) is not int
            or not 0 <= node_index < _EXPECTED_NODES_PER_PROMPT
            or row.get("family_id") != families[example]
            or row.get("path_fraction_hex")
            != GL4_UNIT_INTERVAL_NODES[node_index].hex()
            or row.get("quadrature_weight_hex")
            != GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
            or (example, node_index) in nodes
        ):
            raise ValueError("pinned V9 path node identity differs")
        nodes[(example, node_index)] = row

    families_set = tuple(sorted(set(families.values())))
    if (
        set(endpoints) != set(evidence_artifacts)
        or len(families_set) != _EXPECTED_FAMILIES
        or any(sum(value == family for value in families.values()) != 2 for family in families_set)
        or any(
            (example, node) not in nodes
            for example in endpoints
            for node in range(_EXPECTED_NODES_PER_PROMPT)
        )
    ):
        raise RuntimeError("pinned V9 receipt universe differs")
    for example, endpoint in endpoints.items():
        expected = tuple(
            str(nodes[(example, node)]["receipt_sha256"])
            for node in range(_EXPECTED_NODES_PER_PROMPT)
        )
        raw_expected = endpoint.get("path_node_receipt_sha256s")
        if not isinstance(raw_expected, list) or tuple(raw_expected) != expected:
            raise RuntimeError("pinned V9 finalized evidence node chain differs")
        token_count = int(
            endpoint["scalar_endpoint"]["endpoint_supervised_token_count"]  # type: ignore[index]
        )
        backward_count = (token_count + token_v1._VJP_CHUNK_SIZE - 1) // token_v1._VJP_CHUNK_SIZE
        if any(
            int(nodes[(example, node)]["backward_call_count"]) != backward_count
            for node in range(_EXPECTED_NODES_PER_PROMPT)
        ):
            raise RuntimeError("pinned V9 variable-length node chunks differ")
    ordered = tuple(
        str(nodes[(example, node)]["receipt_sha256"])
        for example in sorted(endpoints)
        for node in range(_EXPECTED_NODES_PER_PROMPT)
    )
    if report.get("path_observation_set_sha256") != token_v1._domain_sha256(
        ordered, domain=v9diag._PATH_OBSERVATION_SET_DOMAIN
    ):
        raise RuntimeError("pinned V9 node receipt set drifted")
    return _PinnedV9Receipts(
        endpoints=endpoints,
        nodes=nodes,
        evidence_artifacts=evidence_artifacts,
    )


def _selected_rows(logits: Tensor, indices: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.dtype != torch.float32
        or not isinstance(indices, Tensor)
        or indices.ndim != 1
        or indices.dtype != torch.int64
        or indices.numel() <= 0
    ):
        raise ValueError(f"{label} requires locked float32 B1 logits and indices")
    selected = logits[0].index_select(0, indices.to(logits.device))
    if not bool(torch.isfinite(selected).all()):
        raise ValueError(f"{label} selected logits are nonfinite")
    return selected


def _selected_token_teacher_kl_f32(
    teacher_logits: Tensor, candidate_logits: Tensor, indices: Tensor
) -> Tensor:
    teacher = _selected_rows(teacher_logits, indices, label="float32 teacher KL")
    candidate = _selected_rows(candidate_logits, indices, label="float32 candidate KL")
    if teacher.shape != candidate.shape:
        raise ValueError("float32 teacher/candidate selected grids differ")
    teacher_logp = F.log_softmax(teacher, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    return (
        teacher_logp.exp() * (teacher_logp - candidate_logp)
    ).sum(dim=-1).detach().to(device="cpu").clone().contiguous()


def _selected_token_teacher_kl_f64(
    teacher_logits: Tensor, candidate_logits: Tensor, indices: Tensor
) -> Tensor:
    teacher = _selected_rows(teacher_logits, indices, label="float64 teacher KL").to(
        torch.float64
    )
    candidate = _selected_rows(
        candidate_logits, indices, label="float64 candidate KL"
    ).to(torch.float64)
    if teacher.shape != candidate.shape:
        raise ValueError("float64 teacher/candidate selected grids differ")
    teacher_logp = F.log_softmax(teacher, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    return (
        teacher_logp.exp() * (teacher_logp - candidate_logp)
    ).sum(dim=-1).detach().to(device="cpu").clone().contiguous()


def _selected_token_teacher_kl_direct_delta_f64(
    teacher_logits: Tensor,
    scalar_logits: Tensor,
    joint_logits: Tensor,
    indices: Tensor,
) -> Tensor:
    teacher = _selected_rows(teacher_logits, indices, label="direct teacher KL").to(
        torch.float64
    )
    scalar = _selected_rows(scalar_logits, indices, label="direct scalar KL").to(
        torch.float64
    )
    joint = _selected_rows(joint_logits, indices, label="direct joint KL").to(
        torch.float64
    )
    if teacher.shape != scalar.shape or scalar.shape != joint.shape:
        raise ValueError("direct float64 endpoint logit grids differ")
    teacher_logp = F.log_softmax(teacher, dim=-1)
    scalar_logp = F.log_softmax(scalar, dim=-1)
    joint_logp = F.log_softmax(joint, dim=-1)
    return (
        teacher_logp.exp() * (scalar_logp - joint_logp)
    ).sum(dim=-1).detach().to(device="cpu").clone().contiguous()


def _require_exact_f64_objective(
    *, captured: Tensor, independent: Tensor, label: str
) -> None:
    if (
        captured.dtype != torch.float64
        or independent.dtype != torch.float64
        or captured.shape != independent.shape
        or not torch.equal(captured.detach().to(device="cpu"), independent)
    ):
        raise RuntimeError(f"{label} float64 objective differs exactly")


def _direct_endpoint_crosscheck(
    *, direct: Tensor, scalar_kl: Tensor, joint_kl: Tensor
) -> tuple[Tensor, float, float]:
    endpoint_subtraction = (joint_kl - scalar_kl).contiguous()
    if direct.shape != endpoint_subtraction.shape or direct.dtype != torch.float64:
        raise ValueError("direct endpoint float64 geometry differs")
    residual = (direct - endpoint_subtraction).contiguous()
    scale = max(
        1.0,
        float(direct.abs().max()),
        float(endpoint_subtraction.abs().max()),
    )
    tolerance = DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE * scale
    maximum = float(residual.abs().max())
    if maximum > tolerance:
        raise RuntimeError("direct and endpoint-subtracted float64 deltas differ")
    return endpoint_subtraction, maximum, tolerance


def _legacy_v9_f32_delta(
    *, scalar_token_kl: Tensor, joint_token_kl: Tensor
) -> Tensor:
    """Replay V9 by promoting each float32 endpoint before subtraction."""

    if (
        not isinstance(scalar_token_kl, Tensor)
        or not isinstance(joint_token_kl, Tensor)
        or scalar_token_kl.dtype != torch.float32
        or joint_token_kl.dtype != torch.float32
        or scalar_token_kl.ndim != 1
        or scalar_token_kl.shape != joint_token_kl.shape
        or scalar_token_kl.numel() <= 0
        or not bool(torch.isfinite(scalar_token_kl).all())
        or not bool(torch.isfinite(joint_token_kl).all())
    ):
        raise ValueError("legacy V9 D32 requires paired finite float32 token vectors")
    return (
        joint_token_kl.detach().to(device="cpu", dtype=torch.float64)
        - scalar_token_kl.detach().to(device="cpu", dtype=torch.float64)
    ).clone().contiguous()


def _require_legacy_v9_f32_delta_replay(
    *, delta: Tensor, pinned_endpoint: Mapping[str, object]
) -> float:
    additive = pinned_endpoint.get("additive_prompt_scalars")
    if (
        not isinstance(delta, Tensor)
        or delta.dtype != torch.float64
        or delta.ndim != 1
        or delta.numel() <= 0
        or not bool(torch.isfinite(delta).all())
        or not isinstance(additive, Mapping)
    ):
        raise ValueError("legacy V9 D32 replay evidence differs")
    expected = additive.get("finite_joint_minus_scalar_teacher_KL")
    live = float(delta.mean())
    if (
        isinstance(expected, bool)
        or not isinstance(expected, (int, float))
        or not math.isfinite(float(expected))
        or live != float(expected)
    ):
        raise RuntimeError("V10 legacy D32 prompt scalar did not replay V9")
    return live


def _authenticate_live_lineage_against_v9(
    *,
    v9_report: Mapping[str, object],
    v8_binding: Mapping[str, object],
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> dict[str, object]:
    control = v9_report.get("v8_control_binding")
    pinned_folds = v9_report.get("folds")
    if not isinstance(control, Mapping) or not isinstance(pinned_folds, list):
        raise ValueError("pinned V9 lineage binding differs")
    pinned_live = control.get("live_lineage_reproduction")
    live_folds = tuple(fits[family].metadata() for family in sorted(fits))
    if (
        not isinstance(pinned_live, Mapping)
        or _canonical(v8_binding) != _canonical(pinned_live)
        or _canonical(live_folds) != _canonical(tuple(pinned_folds))
    ):
        raise RuntimeError("live V10 lineage did not reproduce V9")
    payload: dict[str, object] = {
        "v9_report_file_sha256": V9_REPORT_FILE_SHA256,
        "v9_report_sha256": V9_REPORT_SHA256,
        "v9_schema": v9_report.get("schema"),
        "v9_classification": v9_report.get("classification"),
        "v9_passed": v9_report.get("passed"),
        "v8_live_lineage_canonically_equal": True,
        "held_fold_metadata_canonically_equal": True,
        "authenticated_before_any_v10_fresh_forward": True,
        "raw_tensors_serialized": False,
    }
    payload["artifact_sha256"] = token_v1._domain_sha256(
        payload, domain=_V9_BINDING_DOMAIN
    )
    return payload


def _legacy_v9_endpoint_receipt_equal(
    *, live: Mapping[str, object], pinned: Mapping[str, object], arm: str
) -> None:
    expected = pinned.get("scalar_endpoint" if arm == "v6_exact_scalar" else "joint_endpoint")
    if not isinstance(expected, Mapping) or _canonical(live) != _canonical(expected):
        raise RuntimeError(f"V10 {arm} endpoint did not replay V9")


def _execute_prompt_precision(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    gain_support: CandidateConditionedK64ThreeArmGainSupport,
    codec: object,
    record: object,
    refit: object,
    v8_observations: Mapping[tuple[str, str], Mapping[str, object]],
    pinned_endpoint: Mapping[str, object],
    pinned_nodes: Mapping[tuple[str, int], Mapping[str, object]],
    pinned_v9_evidence_artifact_sha256: str,
    path_node_observer: Callable[..., None] | None = None,
) -> _PromptPrecisionResult:
    bridge = getattr(context, "bridge")
    adapter = getattr(context, "adapter")
    model_inputs, indices, targets, teacher_logits = v9diag.v5diag._fresh_native_teacher(
        context=context, trace=trace
    )
    endpoint_indices, _endpoint_targets, endpoint_grid = v9diag.v5diag._endpoint_indices(
        trace, indices, targets
    )
    if teacher_logits.dtype != torch.float32:
        raise RuntimeError("V10 requires the locked float32 teacher logits")
    teacher_sha = _runtime_tensor_sha256(teacher_logits)
    grid_sha = _runtime_tensor_sha256(endpoint_grid)
    support_mask = (
        trace.prefix.complete_h4_causal_support_mask()
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    resources = {
        "fresh_native_teacher_forward_count": 1,
        "scalar_endpoint_f64_vjp_forward_count": 0,
        "scalar_endpoint_f64_vjp_backward_call_count": 0,
        "joint_boundary_forward_count": 0,
        "path_f64_vjp_forward_count": 0,
        "path_f64_vjp_backward_call_count": 0,
        "path_quadrature_node_count": 0,
    }

    scalar_provider, scalar_correction_rows = v9diag._prepare_v8_endpoint_provider(
        trace=trace,
        basis=basis,
        fit=fit,
        support=gain_support,
        arm="v6_exact_scalar",
    )
    scalar_vjp = bridge.execute_h4_token_teacher_kl_vjps(
        adapter,
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=endpoint_grid,
        vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
        h4_head=scalar_provider,
        objective_dtype=torch.float64,
    )
    resources["scalar_endpoint_f64_vjp_forward_count"] += 1
    resources["scalar_endpoint_f64_vjp_backward_call_count"] += scalar_vjp.backward_call_count
    scalar_vjp.validate_integrity()
    v3diag._validate_candidate_execution(
        trace=trace, provider=scalar_provider, execution=scalar_vjp.execution
    )
    if (
        scalar_vjp.objective_dtype != str(torch.float64)
        or scalar_vjp.execution.logits.dtype != torch.float32
        or scalar_vjp.teacher_logits_sha256 != teacher_sha
        or not torch.equal(
            scalar_vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
        )
    ):
        raise RuntimeError("V10 scalar float64 VJP policy differs")
    scalar_kl64 = scalar_vjp.token_kl_divergences.detach().to(device="cpu").contiguous()
    scalar_independent64 = _selected_token_teacher_kl_f64(
        teacher_logits, scalar_vjp.execution.logits, endpoint_indices
    )
    _require_exact_f64_objective(
        captured=scalar_kl64, independent=scalar_independent64, label="scalar endpoint"
    )
    scalar_kl32 = _selected_token_teacher_kl_f32(
        teacher_logits, scalar_vjp.execution.logits, endpoint_indices
    )
    scalar_legacy64 = scalar_kl32.to(torch.float64).contiguous()
    scalar_h4_rows, scalar_legacy_receipt = v9diag._expected_v8_endpoint_execution(
        trace=trace,
        fit=fit,
        support=gain_support,
        codec=codec,
        record=record,
        refit=refit,
        arm="v6_exact_scalar",
        provider=scalar_provider,
        execution=scalar_vjp.execution,
        correction_rows=scalar_correction_rows,
        teacher_logits=teacher_logits,
        endpoint_grid=endpoint_grid,
        token_kl=scalar_legacy64,
        v8_observation=v8_observations[(trace.example_id, "v6_exact_scalar")],
    )
    _legacy_v9_endpoint_receipt_equal(
        live=scalar_legacy_receipt, pinned=pinned_endpoint, arm="v6_exact_scalar"
    )
    scalar_gradients = (
        scalar_vjp.h4_gradients.detach()
        .to(device="cpu", dtype=torch.float64)[:, 0]
        .index_select(1, trace.support_indices)
        .contiguous()
    )
    scalar_future_maximum, scalar_future_nonzero = v9diag.path_diag._causal_future_gradient_summary(
        gradient_rows=scalar_gradients,
        endpoint_indices=endpoint_indices,
        support_indices=trace.support_indices,
        logical_positions=trace.prefix.logical_positions,
    )
    if scalar_future_maximum != 0.0 or scalar_future_nonzero != 0:
        raise RuntimeError("V10 scalar float64 VJP leaks into future H4")

    joint_provider, joint_correction_rows = v9diag._prepare_v8_endpoint_provider(
        trace=trace,
        basis=basis,
        fit=fit,
        support=gain_support,
        arm="v7_joint",
    )
    joint_execution = bridge.execute(adapter, model_inputs, h4_head=joint_provider)
    resources["joint_boundary_forward_count"] += 1
    v3diag._validate_candidate_execution(
        trace=trace, provider=joint_provider, execution=joint_execution
    )
    if joint_execution.logits.dtype != torch.float32:
        raise RuntimeError("V10 requires the locked float32 joint logits")
    joint_kl64 = _selected_token_teacher_kl_f64(
        teacher_logits, joint_execution.logits, endpoint_indices
    )
    joint_kl32 = _selected_token_teacher_kl_f32(
        teacher_logits, joint_execution.logits, endpoint_indices
    )
    joint_legacy64 = joint_kl32.to(torch.float64).contiguous()
    legacy_delta64 = _legacy_v9_f32_delta(
        scalar_token_kl=scalar_kl32,
        joint_token_kl=joint_kl32,
    )
    pinned_legacy_delta_mean = _require_legacy_v9_f32_delta_replay(
        delta=legacy_delta64,
        pinned_endpoint=pinned_endpoint,
    )
    joint_h4_rows, joint_legacy_receipt = v9diag._expected_v8_endpoint_execution(
        trace=trace,
        fit=fit,
        support=gain_support,
        codec=codec,
        record=record,
        refit=refit,
        arm="v7_joint",
        provider=joint_provider,
        execution=joint_execution,
        correction_rows=joint_correction_rows,
        teacher_logits=teacher_logits,
        endpoint_grid=endpoint_grid,
        token_kl=joint_legacy64,
        v8_observation=v8_observations[(trace.example_id, "v7_joint")],
    )
    _legacy_v9_endpoint_receipt_equal(
        live=joint_legacy_receipt, pinned=pinned_endpoint, arm="v7_joint"
    )
    direct_delta64 = _selected_token_teacher_kl_direct_delta_f64(
        teacher_logits,
        scalar_vjp.execution.logits,
        joint_execution.logits,
        endpoint_indices,
    )
    endpoint_subtraction64, direct_maximum, direct_tolerance = _direct_endpoint_crosscheck(
        direct=direct_delta64, scalar_kl=scalar_kl64, joint_kl=joint_kl64
    )

    endpoint_replay: dict[str, object] = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "pinned_v9_endpoint_pair_sha256": pinned_endpoint["artifact_sha256"],
        "pinned_v9_evidence_artifact_sha256": pinned_v9_evidence_artifact_sha256,
        "model_inputs_sha256": trace.model_inputs_sha256,
        "teacher_logits_sha256": teacher_sha,
        "supervised_grid_sha256": grid_sha,
        "support_mask_sha256": _runtime_tensor_sha256(support_mask),
        "support_indices_sha256": _runtime_tensor_sha256(trace.support_indices),
        "scalar_provider_artifact_sha256": scalar_provider.artifact_sha256,
        "scalar_execution_artifact_sha256": scalar_vjp.execution.artifact_sha256,
        "joint_provider_artifact_sha256": joint_provider.artifact_sha256,
        "joint_execution_artifact_sha256": joint_execution.artifact_sha256,
        "scalar_h4_rows_sha256": _runtime_tensor_sha256(scalar_h4_rows),
        "joint_h4_rows_sha256": _runtime_tensor_sha256(joint_h4_rows),
        "scalar_logits_sha256": _runtime_tensor_sha256(scalar_vjp.execution.logits),
        "joint_logits_sha256": _runtime_tensor_sha256(joint_execution.logits),
        "legacy_V9_endpoint_receipts_replayed_exactly": True,
        "raw_tensors_serialized": False,
    }
    endpoint_replay["artifact_sha256"] = token_v1._domain_sha256(
        endpoint_replay, domain=_ENDPOINT_REPLAY_DOMAIN
    )
    f32_binding: dict[str, object] = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "objective_dtype": "torch.float32",
        "teacher_logits_dtype": str(teacher_logits.dtype),
        "scalar_logits_dtype": str(scalar_vjp.execution.logits.dtype),
        "joint_logits_dtype": str(joint_execution.logits.dtype),
        "scalar_token_teacher_kl_f32_sha256": _runtime_tensor_sha256(scalar_kl32),
        "joint_token_teacher_kl_f32_sha256": _runtime_tensor_sha256(joint_kl32),
        "scalar_token_teacher_kl_promoted_f64_sha256": _runtime_tensor_sha256(
            scalar_legacy64
        ),
        "joint_token_teacher_kl_promoted_f64_sha256": _runtime_tensor_sha256(
            joint_legacy64
        ),
        "legacy_f32_delta_sha256": _runtime_tensor_sha256(legacy_delta64),
        "legacy_f32_delta_mean": pinned_legacy_delta_mean,
        "pinned_V9_f32_delta_mean": pinned_legacy_delta_mean,
        "f32_endpoint_operands_promoted_before_subtraction": True,
        "legacy_V8_V9_hashes_replayed": True,
        "raw_tensors_serialized": False,
    }
    f32_binding["artifact_sha256"] = token_v1._domain_sha256(
        f32_binding, domain=_F32_OBJECTIVE_DOMAIN
    )

    accumulator = CandidateJointStatePathAccumulator(
        example_id=trace.example_id,
        family_id=trace.family_id,
        scalar_endpoint_h4_rows=scalar_h4_rows,
        joint_endpoint_h4_rows=joint_h4_rows,
        scalar_token_teacher_kl=scalar_kl64,
        joint_token_teacher_kl=joint_kl64,
        endpoint_pair_binding_sha256=str(endpoint_replay["artifact_sha256"]),
        scalar_endpoint_execution_artifact_sha256=scalar_vjp.execution.artifact_sha256,
        joint_endpoint_execution_artifact_sha256=joint_execution.artifact_sha256,
        supervised_grid_sha256=grid_sha,
        teacher_logits_sha256=teacher_sha,
        scalar_endpoint_token_h4_gradients=scalar_gradients,
        scalar_tangent_vjp_artifact_sha256=scalar_vjp.artifact_sha256,
        scalar_tangent_provider_artifact_sha256=scalar_provider.artifact_sha256,
        scalar_tangent_execution_artifact_sha256=scalar_vjp.execution.artifact_sha256,
        scalar_tangent_maximum_future_gradient_abs=scalar_future_maximum,
        scalar_tangent_future_gradient_nonzero_count=scalar_future_nonzero,
    )

    node_receipts: list[Mapping[str, object]] = []
    f64_node_bindings: list[Mapping[str, object]] = []
    for node_index, (alpha, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS, strict=True)
    ):
        pinned_node = pinned_nodes[(trace.example_id, node_index)]
        provider = v9diag._AuthenticatedV9ScalarJointPathProvider(
            example_id=trace.example_id,
            family_id=trace.family_id,
            endpoint_pair_binding_sha256=str(pinned_endpoint["artifact_sha256"]),
            model_inputs_sha256=trace.model_inputs_sha256,
            bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
            prefix_artifact_sha256=trace.prefix.artifact_sha256,
            node_index=node_index,
            path_fraction=alpha,
            quadrature_weight=weight,
            base_h4=trace.base_h4,
            support_mask=support_mask,
            support_indices=trace.support_indices,
            scalar_endpoint_h4_rows=scalar_h4_rows,
            joint_endpoint_h4_rows=joint_h4_rows,
        )
        node_vjp = bridge.execute_h4_token_teacher_kl_vjps(
            adapter,
            model_inputs,
            teacher_logits=teacher_logits,
            supervised_indices=endpoint_grid,
            vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
            h4_head=provider,
            objective_dtype=torch.float64,
        )
        resources["path_f64_vjp_forward_count"] += 1
        resources["path_f64_vjp_backward_call_count"] += node_vjp.backward_call_count
        resources["path_quadrature_node_count"] += 1
        node_vjp.validate_integrity()
        provider.validate_integrity()
        v9diag._validate_v9_path_node_full_h4(
            trace=trace, provider=provider, execution=node_vjp.execution
        )
        if (
            node_vjp.objective_dtype != str(torch.float64)
            or node_vjp.execution.logits.dtype != torch.float32
            or node_vjp.teacher_logits_sha256 != teacher_sha
            or not torch.equal(
                node_vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
            )
        ):
            raise RuntimeError("V10 path float64 VJP policy differs")
        token_kl64 = node_vjp.token_kl_divergences.detach().to(device="cpu").contiguous()
        independent64 = _selected_token_teacher_kl_f64(
            teacher_logits, node_vjp.execution.logits, endpoint_indices
        )
        _require_exact_f64_objective(
            captured=token_kl64,
            independent=independent64,
            label=f"path node {node_index}",
        )
        legacy32 = _selected_token_teacher_kl_f32(
            teacher_logits, node_vjp.execution.logits, endpoint_indices
        )
        legacy64 = legacy32.to(torch.float64).contiguous()
        gradient_rows = (
            node_vjp.h4_gradients.detach()
            .to(device="cpu", dtype=torch.float64)[:, 0]
            .index_select(1, trace.support_indices)
            .contiguous()
        )
        node_h4_rows = v9diag._realized_support_rows(
            node_vjp.execution.candidate_h4, trace.support_indices
        )
        maximum_future, future_nonzero = v9diag.path_diag._causal_future_gradient_summary(
            gradient_rows=gradient_rows,
            endpoint_indices=endpoint_indices,
            support_indices=trace.support_indices,
            logical_positions=trace.prefix.logical_positions,
        )
        replay_fields = {
            "endpoint_pair_binding_sha256": provider.endpoint_pair_binding_sha256,
            "provider_artifact_sha256": provider.artifact_sha256,
            "execution_artifact_sha256": node_vjp.execution.artifact_sha256,
            "candidate_h4_sha256": _runtime_tensor_sha256(
                node_vjp.execution.candidate_h4
            ),
            "path_node_h4_rows_sha256": _runtime_tensor_sha256(node_h4_rows),
            "candidate_logits_sha256": _runtime_tensor_sha256(
                node_vjp.execution.logits
            ),
            "token_teacher_kl_sha256": _runtime_tensor_sha256(legacy64),
            "backward_call_count": node_vjp.backward_call_count,
        }
        changed = tuple(
            key for key, value in replay_fields.items() if pinned_node.get(key) != value
        )
        if (
            changed
            or maximum_future != 0.0
            or future_nonzero != 0
            or not token_v1._bitwise_equal(node_h4_rows, provider.path_h4_rows)
        ):
            raise RuntimeError(
                f"V10 node did not replay V9 execution: {changed}"
            )
        core_receipt = accumulator.add_node(
            node_index=node_index,
            path_fraction=alpha,
            quadrature_weight=weight,
            path_node_h4_rows=node_h4_rows,
            token_h4_gradients=gradient_rows,
            token_teacher_kl=token_kl64,
            vjp_artifact_sha256=node_vjp.artifact_sha256,
            provider_artifact_sha256=provider.artifact_sha256,
            execution_artifact_sha256=node_vjp.execution.artifact_sha256,
            maximum_future_gradient_abs=maximum_future,
            future_gradient_nonzero_count=future_nonzero,
        )
        if path_node_observer is not None:
            path_node_observer(
                context=context,
                trace=trace,
                model_inputs=model_inputs,
                teacher_logits=teacher_logits,
                endpoint_indices=endpoint_indices,
                endpoint_grid=endpoint_grid,
                scalar_execution=scalar_vjp.execution,
                joint_execution=joint_execution,
                scalar_h4_rows=scalar_h4_rows,
                joint_h4_rows=joint_h4_rows,
                scalar_token_teacher_kl_f64=scalar_kl64,
                joint_token_teacher_kl_f64=joint_kl64,
                direct_delta_f64=direct_delta64,
                node_index=node_index,
                path_fraction=alpha,
                quadrature_weight=weight,
                path_provider=provider,
                path_vjp=node_vjp,
                path_h4_rows=node_h4_rows,
                path_token_teacher_kl_f64=token_kl64,
                path_token_h4_gradients_f64=gradient_rows,
                core_node_receipt=core_receipt,
            )
        f64_binding = {
            "node_index": node_index,
            "path_fraction_hex": alpha.hex(),
            "quadrature_weight_hex": weight.hex(),
            "objective_dtype": str(torch.float64),
            "token_teacher_kl_f64_sha256": _runtime_tensor_sha256(token_kl64),
            "h4_gradient_runtime_sha256": _runtime_tensor_sha256(gradient_rows),
            "vjp_artifact_sha256": node_vjp.artifact_sha256,
            "core_node_receipt_sha256": core_receipt.artifact_sha256,
            "captured_equals_independent_bitwise": True,
        }
        f64_node_bindings.append(f64_binding)
        receipt: dict[str, object] = {
            "example_id": trace.example_id,
            "family_id": trace.family_id,
            "node_index": node_index,
            "pinned_v9_node_receipt_sha256": pinned_node["receipt_sha256"],
            **replay_fields,
            **f64_binding,
            "maximum_future_gradient_abs_hex": maximum_future.hex(),
            "future_gradient_nonzero_count": future_nonzero,
            "legacy_V9_node_replayed_exactly": True,
            "raw_tensors_serialized": False,
        }
        receipt["receipt_sha256"] = token_v1._domain_sha256(
            receipt, domain=_NODE_RECEIPT_DOMAIN
        )
        node_receipts.append(receipt)
        del node_vjp, provider, gradient_rows, token_kl64, independent64, legacy32

    path_evidence = accumulator.finalize()
    f64_binding: dict[str, object] = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "objective_dtype": str(torch.float64),
        "teacher_logits_dtype": str(teacher_logits.dtype),
        "scalar_token_teacher_kl_f64_sha256": _runtime_tensor_sha256(scalar_kl64),
        "joint_token_teacher_kl_f64_sha256": _runtime_tensor_sha256(joint_kl64),
        "finite_delta_f64_direct_sha256": _runtime_tensor_sha256(direct_delta64),
        "finite_delta_f64_endpoint_subtraction_sha256": _runtime_tensor_sha256(
            endpoint_subtraction64
        ),
        "direct_minus_subtraction_max_abs": direct_maximum,
        "direct_endpoint_crosscheck_tolerance": direct_tolerance,
        "scalar_f64_vjp_artifact_sha256": scalar_vjp.artifact_sha256,
        "path_f64_node_bindings": tuple(f64_node_bindings),
        "captured_objectives_equal_independent_bitwise": True,
        "raw_tensors_serialized": False,
    }
    f64_binding["artifact_sha256"] = token_v1._domain_sha256(
        f64_binding, domain=_F64_OBJECTIVE_DOMAIN
    )
    precision_evidence = CandidateJointStateObjectivePrecisionEvidence(
        path_evidence=path_evidence,
        finite_delta_f64_direct=direct_delta64,
        scalar_token_teacher_kl_f32=scalar_kl32,
        joint_token_teacher_kl_f32=joint_kl32,
        pinned_v9_evidence_artifact_sha256=pinned_v9_evidence_artifact_sha256,
        endpoint_replay_binding_sha256=str(endpoint_replay["artifact_sha256"]),
        f32_objective_binding_sha256=str(f32_binding["artifact_sha256"]),
        f64_objective_binding_sha256=str(f64_binding["artifact_sha256"]),
    )
    endpoint_receipt = {
        **endpoint_replay,
        "precision_evidence_artifact_sha256": precision_evidence.artifact_sha256,
        "f32_objective_binding": f32_binding,
        "f64_objective_binding": f64_binding,
        "direct_f64_delta_mean": float(direct_delta64.mean()),
        "legacy_f32_delta_sha256": _runtime_tensor_sha256(legacy_delta64),
        "legacy_f32_delta_mean": pinned_legacy_delta_mean,
        "pinned_V9_f32_delta_mean": pinned_legacy_delta_mean,
        "legacy_V9_D32_hash_and_scalar_replayed": True,
        "GL4_node_receipt_sha256s": tuple(
            str(value["receipt_sha256"]) for value in node_receipts
        ),
    }
    return _PromptPrecisionResult(
        evidence=precision_evidence,
        endpoint_receipt=endpoint_receipt,
        node_receipts=tuple(node_receipts),
        resources=resources,
    )


def _execute_live_precision_grid(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    analytic: v8diag._V8AnalyticPhaseResults,
    v8_observations: Mapping[tuple[str, str], Mapping[str, object]],
    pinned: _PinnedV9Receipts,
    path_node_observer: Callable[..., None] | None = None,
) -> _LivePrecisionResult:
    values = tuple(sorted(traces, key=lambda value: value.example_id))
    fold_records = {
        record.outer_held_family_id: record
        for record in analytic.v7_phases.joint_fold_records
    }
    families = tuple(sorted(fits))
    if (
        len(values) != _EXPECTED_PROMPTS
        or len(families) != _EXPECTED_FAMILIES
        or set(fold_records) != set(families)
        or set(analytic.codecs) != set(families)
        or {value.example_id for value in values} != set(pinned.endpoints)
        or any(sum(value.family_id == family for value in values) != 2 for family in families)
    ):
        raise RuntimeError("V10 objective precision held universe differs")
    totals = {
        "fresh_native_teacher_forward_count": 0,
        "scalar_endpoint_f64_vjp_forward_count": 0,
        "scalar_endpoint_f64_vjp_backward_call_count": 0,
        "joint_boundary_forward_count": 0,
        "path_f64_vjp_forward_count": 0,
        "path_f64_vjp_backward_call_count": 0,
        "path_quadrature_node_count": 0,
    }
    evidence: list[CandidateJointStateObjectivePrecisionEvidence] = []
    endpoint_receipts: list[Mapping[str, object]] = []
    node_receipts: list[Mapping[str, object]] = []
    for trace in values:
        family = trace.family_id
        fit = fits[family]
        record = fold_records[family]
        refit = analytic.v7_phases.v6_phases.row_bank.refits[family]
        codec = analytic.codecs[family]
        directions = v6diag._ordered_k64(fit)
        base_rows = v9diag._realized_support_rows(
            trace.base_h4, trace.support_indices
        ).to(torch.float64)
        gain_support = build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            codec,
            record.full_scalar_control_fit,
            record.full_joint_fit,
            phase="held",
            example_id=trace.example_id,
            family_id=trace.family_id,
            base_h4_support_rows=base_rows,
            ordered_directions=directions,
        )
        result = _execute_prompt_precision(
            context=context,
            trace=trace,
            basis=basis,
            fit=fit,
            gain_support=gain_support,
            codec=codec,
            record=record,
            refit=refit,
            v8_observations=v8_observations,
            pinned_endpoint=pinned.endpoints[trace.example_id],
            pinned_nodes=pinned.nodes,
            pinned_v9_evidence_artifact_sha256=pinned.evidence_artifacts[
                trace.example_id
            ],
            path_node_observer=path_node_observer,
        )
        evidence.append(result.evidence)
        endpoint_receipts.append(result.endpoint_receipt)
        node_receipts.extend(result.node_receipts)
        for key, value in result.resources.items():
            totals[key] += value
    comparison = summarize_candidate_joint_state_objective_precision(evidence)
    ordered_nodes = tuple(
        sorted(
            node_receipts,
            key=lambda value: (str(value["example_id"]), int(value["node_index"])),
        )
    )
    node_set = token_v1._domain_sha256(
        tuple(str(value["receipt_sha256"]) for value in ordered_nodes),
        domain=_NODE_SET_DOMAIN,
    )
    return _LivePrecisionResult(
        evidence=tuple(evidence),
        comparison=comparison,
        endpoint_receipts=tuple(endpoint_receipts),
        node_receipts=ordered_nodes,
        node_receipt_set_sha256=node_set,
        resources=totals,
    )


def _resource_accounting(
    *,
    endpoint_resources: Mapping[str, int],
    gradient_resources: Mapping[str, int],
    live_resources: Mapping[str, int],
    row_bank_candidate_support_row_executions: int,
) -> dict[str, object]:
    parent_forwards = (
        endpoint_resources["base_forward_count"]
        + endpoint_resources["native_forward_count"]
        + endpoint_resources["endpoint_token_vjp_forward_count"]
    )
    parent_backwards = endpoint_resources["endpoint_token_vjp_backward_call_count"]
    analytic_forwards = (
        gradient_resources["gradient_native_forward_count"]
        + gradient_resources["gradient_candidate_vjp_forward_count"]
    )
    analytic_backwards = gradient_resources["gradient_candidate_vjp_backward_call_count"]
    lineage_forwards = parent_forwards + analytic_forwards
    lineage_backwards = parent_backwards + analytic_backwards
    fresh_forwards = (
        live_resources["fresh_native_teacher_forward_count"]
        + live_resources["scalar_endpoint_f64_vjp_forward_count"]
        + live_resources["joint_boundary_forward_count"]
        + live_resources["path_f64_vjp_forward_count"]
    )
    fresh_backwards = (
        live_resources["scalar_endpoint_f64_vjp_backward_call_count"]
        + live_resources["path_f64_vjp_backward_call_count"]
    )
    total_forwards = lineage_forwards + fresh_forwards
    total_backwards = lineage_backwards + fresh_backwards
    support_rows = endpoint_resources["complete_h4_support_row_count"]
    fresh_support = support_rows * 6
    total_support = row_bank_candidate_support_row_executions + fresh_support
    if (
        parent_forwards != 48
        or parent_backwards != 109
        or analytic_forwards != 64
        or analytic_backwards != 385
        or lineage_forwards != _EXPECTED_LINEAGE_FORWARDS
        or lineage_backwards != _EXPECTED_LINEAGE_BACKWARDS
        or live_resources["fresh_native_teacher_forward_count"] != 16
        or live_resources["scalar_endpoint_f64_vjp_forward_count"] != 16
        or live_resources["scalar_endpoint_f64_vjp_backward_call_count"] != 109
        or live_resources["joint_boundary_forward_count"] != 16
        or live_resources["path_f64_vjp_forward_count"] != 64
        or live_resources["path_f64_vjp_backward_call_count"] != 436
        or live_resources["path_quadrature_node_count"] != 64
        or fresh_forwards != _EXPECTED_FRESH_FORWARDS
        or fresh_backwards != _EXPECTED_FRESH_BACKWARDS
        or total_forwards != _EXPECTED_TOTAL_FORWARDS
        or total_backwards != _EXPECTED_TOTAL_BACKWARDS
        or support_rows != 819
        or row_bank_candidate_support_row_executions
        != _EXPECTED_ROW_BANK_SUPPORT_EXECUTIONS
        or fresh_support != _EXPECTED_FRESH_SUPPORT_EXECUTIONS
        or total_support != _EXPECTED_TOTAL_SUPPORT_EXECUTIONS
    ):
        raise RuntimeError("V10 objective precision resource accounting differs")
    return {
        **endpoint_resources,
        **gradient_resources,
        **live_resources,
        "phase_order": (
            "parent_endpoint_recollection",
            "exact_v7_v8_lineage_reproduction_and_v9_authentication",
            "fresh_native_teacher_once_per_outer_held_prompt",
            "scalar_endpoint_float64_teacher_KL_VJP",
            "joint_endpoint_boundary_execution",
            "four_fixed_GL4_float64_teacher_KL_VJPs",
            "family_equal_objective_precision_publication",
        ),
        "lineage_model_forward_count": lineage_forwards,
        "lineage_backward_call_count": lineage_backwards,
        "fresh_model_forward_count": fresh_forwards,
        "fresh_backward_call_count": fresh_backwards,
        "row_bank_candidate_support_row_executions": row_bank_candidate_support_row_executions,
        "fresh_candidate_support_row_executions": fresh_support,
        "total_candidate_support_row_executions": total_support,
        "tune_model_forward_count": 0,
        "selection_model_forward_count": 0,
        "step_grid_model_forward_count": 0,
        "fallback_model_forward_count": 0,
        "routing_model_forward_count": 0,
        "total_model_forward_count": total_forwards,
        "total_backward_call_count": total_backwards,
        "exact_model_forward_count_is_224": total_forwards == 224,
        "exact_backward_call_count_is_1039": total_backwards == 1039,
        "backward_accounting_uses_actual_vjp_backward_call_count": True,
        "raw_logits_gradients_or_H4_retained_in_report": False,
    }


def _scientific_classification(
    comparison: CandidateJointStateObjectivePrecisionComparison,
) -> str:
    if comparison.closure_passed:
        return "high_precision_closure_established_same_a"
    metrics = comparison.metrics
    if metrics.finite_delta_f64_rms <= metrics.relative_rmse_epsilon:
        return "objective_precision_source_unresolved_same_a"
    if (
        metrics.transport_relative_rmse_to_finite_f64
        <= OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
    ):
        return "small_path_transport_live_dtype_or_finite_rounding_supported_same_a"
    if math.isfinite(metrics.transport_relative_rmse_to_finite_f64):
        return "material_path_transport_higher_order_quadrature_earned_same_a"
    return "objective_precision_source_unresolved_same_a"


def _require_integrity_gate_results(gates: Mapping[str, bool]) -> None:
    failures = tuple(sorted(key for key, value in gates.items() if not value))
    if failures:
        raise RuntimeError(
            "V10 objective precision integrity failed before publication: "
            + ", ".join(failures)
        )


def _safety_metadata() -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_gain_vectors": False,
        "contains_token_teacher_kl_tensors": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_objective_precision_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    v5_report_path: Path | str = DEFAULT_V5_REPORT,
    v6_report_path: Path | str = DEFAULT_V6_REPORT,
    v7_report_path: Path | str = DEFAULT_V7_REPORT,
    v8_report_path: Path | str = DEFAULT_V8_REPORT,
    v9_report_path: Path | str = DEFAULT_V9_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the no-fit, no-selection V10 objective-precision replay."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V10 objective precision report")
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = v9diag.v4diag._load_v3_report(v3_report_path)
    v4_report = v9diag.v5diag._load_v4_report(v4_report_path)
    v5_report = v6diag._load_v5_report(v5_report_path)
    v6_report = v7diag._load_v6_report(v6_report_path)
    v7_report = v8diag._load_v7_report(v7_report_path)
    v8_report = v9diag._load_v8_report(v8_report_path)
    v8_observations = v9diag._index_v8_endpoint_observations(v8_report)
    v9_report = _load_v9_report(v9_report_path)
    pinned = _index_v9_receipts(v9_report)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="V10 objective precision rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="V10 objective precision rank320 transfer",
    )
    transfer_receipts = expanded._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = transfer_support._load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = token_v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != _EXPECTED_PROMPTS or len(families) != _EXPECTED_FAMILIES:
            raise RuntimeError("V10 objective precision A16 panel differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        analytic = v8diag._execute_v8_analytic_phases(
            context=context,
            parent=parent,
            v3_report=v3_report,
            v4_report=v4_report,
            v5_report=v5_report,
            v6_report=v6_report,
            v7_report=v7_report,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        v8_binding = v9diag._authenticate_v8_live_lineage(
            v8_report=v8_report, analytic=analytic
        )
        v9_binding = _authenticate_live_lineage_against_v9(
            v9_report=v9_report, v8_binding=v8_binding, fits=fits
        )
        live = _execute_live_precision_grid(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            analytic=analytic,
            v8_observations=v8_observations,
            pinned=pinned,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    resources = _resource_accounting(
        endpoint_resources=endpoint_resources,
        gradient_resources=analytic.v7_phases.v6_phases.row_bank.resources,
        live_resources=live.resources,
        row_bank_candidate_support_row_executions=(
            v8diag._row_bank_candidate_support_row_executions(analytic.v7_phases)
        ),
    )
    comparison = live.comparison
    comparison_metadata = comparison.metadata()
    endpoint_count = len(live.endpoint_receipts)
    node_count = len(live.node_receipts)
    integrity_gates = {
        "exact_V9_file_and_logical_hash_authenticated": (
            v9_binding["v9_report_file_sha256"] == V9_REPORT_FILE_SHA256
            and v9_binding["v9_report_sha256"] == V9_REPORT_SHA256
        ),
        "V7_V8_lineage_and_folds_reproduced_before_fresh_work": (
            v9_binding["authenticated_before_any_v10_fresh_forward"] is True
            and v9_binding["v8_live_lineage_canonically_equal"] is True
            and v9_binding["held_fold_metadata_canonically_equal"] is True
        ),
        "all_16_endpoints_replay_V9_V8_H4_logits_provider_execution_and_KL": (
            endpoint_count == _EXPECTED_PROMPTS
            and all(
                bool(value["legacy_V9_endpoint_receipts_replayed_exactly"])
                for value in live.endpoint_receipts
            )
        ),
        "all_64_GL4_nodes_replay_V9_H4_logits_provider_execution_and_KL": (
            node_count == _EXPECTED_NODE_FORWARDS
            and all(
                bool(value["legacy_V9_node_replayed_exactly"])
                for value in live.node_receipts
            )
        ),
        "every_f64_objective_is_bitwise_equal_to_independent_recomputation": all(
            bool(
                value["f64_objective_binding"][
                    "captured_objectives_equal_independent_bitwise"
                ]
            )
            for value in live.endpoint_receipts
        ),
        "all_logits_and_legacy_objectives_use_explicit_float32": all(
            value["f32_objective_binding"]["objective_dtype"] == "torch.float32"
            and value["f32_objective_binding"]["teacher_logits_dtype"]
            == "torch.float32"
            and value["f32_objective_binding"]["scalar_logits_dtype"]
            == "torch.float32"
            and value["f32_objective_binding"]["joint_logits_dtype"]
            == "torch.float32"
            for value in live.endpoint_receipts
        ),
        "all_D32_hashes_and_prompt_scalars_replay_V9_promote_then_subtract": all(
            bool(value["legacy_V9_D32_hash_and_scalar_replayed"])
            and value["legacy_f32_delta_mean"] == value["pinned_V9_f32_delta_mean"]
            and bool(
                value["f32_objective_binding"][
                    "f32_endpoint_operands_promoted_before_subtraction"
                ]
            )
            for value in live.endpoint_receipts
        ),
        "direct_D64_crosschecks_endpoint_KL64_subtraction": all(
            float(value["f64_objective_binding"]["direct_minus_subtraction_max_abs"])
            <= float(
                value["f64_objective_binding"]["direct_endpoint_crosscheck_tolerance"]
            )
            for value in live.endpoint_receipts
        ),
        "all_scalar_and_path_float64_VJPs_are_causal": all(
            value.path_evidence.scalar_endpoint_tangent_receipt is not None
            and value.path_evidence.scalar_endpoint_tangent_receipt.future_gradient_nonzero_count
            == 0
            and all(
                receipt.future_gradient_nonzero_count == 0
                for receipt in value.path_evidence.node_receipts
            )
            for value in live.evidence
        ),
        "no_fit_selection_search_fallback_or_routing": (
            resources["tune_model_forward_count"] == 0
            and resources["selection_model_forward_count"] == 0
            and resources["step_grid_model_forward_count"] == 0
            and resources["fallback_model_forward_count"] == 0
            and resources["routing_model_forward_count"] == 0
        ),
        "exact_model_forward_count_is_224": (
            resources["total_model_forward_count"] == _EXPECTED_TOTAL_FORWARDS
        ),
        "exact_backward_call_count_is_1039": (
            resources["total_backward_call_count"] == _EXPECTED_TOTAL_BACKWARDS
        ),
        "exact_candidate_support_row_executions_is_7756": (
            resources["total_candidate_support_row_executions"]
            == _EXPECTED_TOTAL_SUPPORT_EXECUTIONS
        ),
    }
    _require_integrity_gate_results(integrity_gates)
    classification = _scientific_classification(comparison)
    passed = comparison.closure_passed
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_precision_diagnostic",
            "outer_held_prompt_count": _EXPECTED_PROMPTS,
            "outer_family_count": _EXPECTED_FAMILIES,
            "prompts_per_outer_family": 2,
            "scalar_endpoint": "exact_frozen_full_seven_V6_scalar_control",
            "joint_endpoint": "exact_frozen_full_seven_V7_joint_field",
            "path": "exact_V9_realized_cast_once_scalar_to_joint_H4_path",
            "quadrature_rule": "fixed_gauss_legendre_order_4_on_unit_interval",
            "quadrature_nodes_hex": tuple(
                value.hex() for value in GL4_UNIT_INTERVAL_NODES
            ),
            "quadrature_weights_hex": tuple(
                value.hex() for value in GL4_UNIT_INTERVAL_WEIGHTS
            ),
            "only_changed_policy": (
                "selected_teacher_and_candidate_logits_promoted_to_float64_"
                "before_teacher_KL_arithmetic"
            ),
            "finite_D64_primary": (
                "sum_p_teacher64_times_logp_scalar64_minus_logp_joint64"
            ),
            "finite_D32_control": (
                "promote_each_float32_endpoint_KL_operand_to_float64_"
                "before_subtraction_exactly_as_V9"
            ),
            "aggregation": "equal_token_then_equal_prompt_then_equal_family",
            "closure_thresholds": {
                "overall_relative_RMSE_maximum": OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM,
                "every_family_relative_RMSE_maximum": (
                    FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "overall_cosine_minimum": CLOSURE_COSINE_MINIMUM,
            },
            "v10_fit_performed": False,
            "selection_or_search_performed": False,
            "fallback_or_routing_allowed": False,
            "result_can_authorize_serving_or_model_mutation": False,
        },
        "v9_control_binding": {
            "file": str(v9_report_path),
            "file_sha256": V9_REPORT_FILE_SHA256,
            "report_sha256": V9_REPORT_SHA256,
            "schema": v9_report.get("schema"),
            "classification": v9_report.get("classification"),
            "passed": v9_report.get("passed"),
            "live_lineage_reproduction": v9_binding,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": token_v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": token_v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": token_v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": v9diag.v5diag._endpoint_prompt_receipts(traces),
        "endpoint_precision_receipts": live.endpoint_receipts,
        "path_precision_node_receipts": live.node_receipts,
        "path_precision_node_receipt_set_sha256": live.node_receipt_set_sha256,
        "family_equal_objective_precision_comparison": comparison_metadata,
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "outcome_matrix": {
            "outcome": classification,
            "integrity_completed": True,
            "high_precision_closure_established": comparison.closure_passed,
            "higher_order_quadrature_earned_only_if_transport_material": (
                not comparison.closure_passed
                and comparison.metrics.transport_relative_rmse_to_finite_f64
                > OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
            ),
            "fresh_confirmation_authorized_next": False,
            "selection_or_serving_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_diagnostic_only": True,
            "all_parent_outcomes_previously_inspected": True,
            "V9_f32_path_is_exact_execution_control": True,
            "direct_D64_is_cancellation_resistant": True,
            "P64_minus_T64_is_path_transport_not_unique_cause": True,
            "D64_minus_D32_is_endpoint_precision_signal_not_unique_cause": True,
            "single_numerical_cause_forced": False,
            "fresh_family_disjoint_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "model_mutation_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
        },
        "safety": _safety_metadata(),
    }
    return _publish(report, output=destination)


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(report, domain=_REPORT_DOMAIN)
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": token_v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run the pinned V10 float64-objective scalar-to-joint GL4 replay."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_objective_precision_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
