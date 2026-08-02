"""Locked V11 suffix-JVP replay of the V10 scalar-to-joint H4 path.

V11 observes the exact V10 GL4 executions and, at each already-evaluated H4
node, replays only the native Gemma post-H4 suffix in forward mode.  The
resulting tokenwise JVP is compared with both the wrapped V10 reverse-mode
contraction and the direct float64 endpoint delta.  Six realized live-dtype
points (scalar, four GL4 nodes, joint) are also described by five adjacent
discrete-cast interval receipts.

This remains a same-A diagnostic.  It performs no fit, search, selection,
correction, routing, fallback, serving mutation, or compression authorization.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_objective_precision_diagnostic as v10diag
from .complete_h4_tail_candidate_joint_state_suffix_jvp import (
    ADJOINT_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateSuffixJVPComparison,
    CandidateJointStateSuffixJVPEvidence,
    CandidateJointStateSuffixJVPNodeEvidence,
    summarize_candidate_joint_state_suffix_jvp,
)
from .gemma3_l3_l4_h4_suffix_jvp_runtime import (
    Gemma3L3L4H4SuffixJVP,
    Gemma3L3L4H4SuffixJVPRuntime,
    gemma3_l3_l4_h4_discrete_cast_interval_stats,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_V10_REPORT",
    "V10_REPORT_FILE_SHA256",
    "V10_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_jvp_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v10diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v10diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v10diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v10diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v10diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v10diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v10diag.DEFAULT_V6_REPORT
DEFAULT_V7_REPORT = v10diag.DEFAULT_V7_REPORT
DEFAULT_V8_REPORT = v10diag.DEFAULT_V8_REPORT
DEFAULT_V9_REPORT = v10diag.DEFAULT_V9_REPORT
DEFAULT_V10_REPORT = v10diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = v10diag.token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-suffix-jvp-gl4-lofo-a-"
    "fit16-dev-v11.json"
)

V10_REPORT_FILE_SHA256 = (
    "48afceefdf9c468a91f07b2004ed9052586e99d41dba5f4c7223b6e4ae638e79"
)
V10_REPORT_SHA256 = (
    "36468470728bb941933ada75b74497108d0c163dc23688b91375462ac96fc77c"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_suffix_jvp_gl4_lofo.v11"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-suffix-jvp:v11\0"
_RUNTIME_NODE_DOMAIN = b"fisher-graph:complete-h4-k64-suffix-runtime-node:v11\0"
_RUNTIME_NODE_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-suffix-runtime-node-set:v11\0"
)
_DISCRETE_PROMPT_DOMAIN = b"fisher-graph:complete-h4-k64-discrete-cast:v11\0"
_DISCRETE_SET_DOMAIN = b"fisher-graph:complete-h4-k64-discrete-cast-set:v11\0"

_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS = 16
_EXPECTED_NODES_PER_PROMPT = 4
_EXPECTED_SUFFIX_JVP_EVALUATIONS = 64
_EXPECTED_SUFFIX_SEGMENT_CALLS = 832
_EXPECTED_SUFFIX_PROJECTIONS = 64
_EXPECTED_SUFFIX_DTYPE_CASTS = 64
_EXPECTED_SUFFIX_H4_SUPPORT_EVALUATIONS = 3276
_EXPECTED_SUFFIX_FULL_H4_ROW_EVALUATIONS = 3788
_EXPECTED_SUFFIX_SUPPORT_ROW_LAYER_OBSERVATIONS = 42588
_EXPECTED_SUFFIX_FULL_H4_ROW_LAYER_CALLS = 49244
_EXPECTED_SUFFIX_OUTSIDE_SUPPORT_ROW_EVALUATIONS = 512
_EXPECTED_SUFFIX_OUTSIDE_SUPPORT_ROW_LAYER_CALLS = 6656
_EXPECTED_SUFFIX_SUPERVISED_TOKEN_KL_ELEMENTS = 3212
_EXPECTED_COMBINED_SUPPORT_EVALUATIONS = 11032


@dataclass(frozen=True, slots=True)
class _PinnedV10Receipts:
    endpoints: Mapping[str, Mapping[str, object]]
    nodes: Mapping[tuple[str, int], Mapping[str, object]]
    evidence_artifacts: Mapping[str, str]
    comparison_artifact_sha256: str


@dataclass(slots=True)
class _ObservedNode:
    node_index: int
    path_fraction: float
    quadrature_weight: float
    directional_token_teacher_kl_f64: Tensor
    core_node_receipt: object
    suffix_runtime_receipt_sha256: str


@dataclass(slots=True)
class _ObservedPrompt:
    example_id: str
    family_id: str
    scalar_h4: Tensor
    scalar_token_kl_f64: Tensor
    joint_h4: Tensor
    joint_token_kl_f64: Tensor
    direction_h4_f64: Tensor
    runtime: Gemma3L3L4H4SuffixJVPRuntime
    nodes: dict[int, _ObservedNode] = field(default_factory=dict)
    runtime_receipts: dict[int, Mapping[str, object]] = field(default_factory=dict)
    live_node_h4: dict[int, Tensor] = field(default_factory=dict)
    node_token_kl_f64: dict[int, Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _SuffixCollectionResult:
    evidence: tuple[CandidateJointStateSuffixJVPEvidence, ...]
    comparison: CandidateJointStateSuffixJVPComparison
    runtime_node_receipts: tuple[Mapping[str, object], ...]
    runtime_node_receipt_set_sha256: str
    discrete_cast_receipts: tuple[Mapping[str, object], ...]
    discrete_cast_receipt_set_sha256: str
    resources: Mapping[str, int]


def _canonical(value: object) -> object:
    return v10diag._canonical(value)


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _load_v10_report(path: Path | str) -> dict[str, object]:
    """Load only the exact inspected V10 precision artifact."""

    report = v10diag.token_v1._load_pinned_report(
        path,
        expected_file_sha256=V10_REPORT_FILE_SHA256,
        expected_report_sha256=V10_REPORT_SHA256,
        label="candidate joint-state suffix JVP V10 anchor",
    )
    resources = report.get("resources")
    comparison = report.get("family_equal_objective_precision_comparison")
    if (
        report.get("schema") != v10diag._SCHEMA
        or report.get("classification")
        != "small_path_transport_live_dtype_or_finite_rounding_supported_same_a"
        or report.get("passed") is not False
        or not isinstance(resources, Mapping)
        or resources.get("total_model_forward_count") != 224
        or resources.get("total_backward_call_count") != 1039
        or resources.get("total_candidate_support_row_executions") != 7756
        or not isinstance(comparison, Mapping)
        or float(comparison.get("closure_relative_rmse", -1.0))
        != 0.08673388652083237
        or float(comparison.get("closure_rmse", -1.0))
        != 1.3509587725610062e-6
        or float(comparison.get("closure_cosine", -1.0))
        != 0.99626247056314
        or float(comparison.get("transport_relative_rmse_to_finite_f64", -1.0))
        != 0.0006221383476886703
        or float(comparison.get("transport_rmse", -1.0))
        != 9.690367770556959e-9
        or float(comparison.get("finite_precision_relative_rmse_to_finite_f64", -1.0))
        != 0.02003952674974365
        or float(comparison.get("finite_precision_rmse", -1.0))
        != 3.121337639359058e-7
        or float(comparison.get("finite_delta_f64_rms", -1.0))
        != 1.5575904951941974e-5
    ):
        raise RuntimeError("candidate joint-state objective precision V10 anchor differs")
    _index_v10_receipts(report)
    return report


def _index_v10_receipts(report: Mapping[str, object]) -> _PinnedV10Receipts:
    """Authenticate the exact V10 16-endpoint, 64-node receipt universe."""

    raw_endpoints = report.get("endpoint_precision_receipts")
    raw_nodes = report.get("path_precision_node_receipts")
    comparison = report.get("family_equal_objective_precision_comparison")
    if (
        not isinstance(raw_endpoints, list)
        or len(raw_endpoints) != _EXPECTED_PROMPTS
        or not isinstance(raw_nodes, list)
        or len(raw_nodes) != _EXPECTED_SUFFIX_JVP_EVALUATIONS
        or not isinstance(comparison, Mapping)
    ):
        raise ValueError("pinned V10 receipt grid differs")
    raw_examples = comparison.get("evidence_example_ids")
    raw_evidence = comparison.get("evidence_artifact_sha256s")
    if (
        not isinstance(raw_examples, list)
        or not isinstance(raw_evidence, list)
        or len(raw_examples) != _EXPECTED_PROMPTS
        or len(raw_evidence) != _EXPECTED_PROMPTS
    ):
        raise ValueError("pinned V10 evidence index differs")
    evidence_artifacts = {
        v10diag.token_v1._identifier(example, label="pinned V10 evidence example"): (
            v10diag._require_sha256(value, label="pinned V10 evidence artifact")
        )
        for example, value in zip(raw_examples, raw_evidence, strict=True)
    }

    endpoints: dict[str, Mapping[str, object]] = {}
    families: dict[str, str] = {}
    node_extensions: dict[tuple[str, int], Mapping[str, object]] = {}
    for raw in raw_endpoints:
        row = dict(_require_mapping(raw, label="pinned V10 endpoint"))
        example = v10diag.token_v1._identifier(
            row.get("example_id"), label="pinned V10 endpoint example"
        )
        family = v10diag.token_v1._identifier(
            row.get("family_id"), label="pinned V10 endpoint family"
        )
        artifact = v10diag._require_sha256(
            row.get("artifact_sha256"), label="pinned V10 endpoint replay"
        )
        base = dict(row)
        for key in (
            "artifact_sha256",
            "precision_evidence_artifact_sha256",
            "f32_objective_binding",
            "f64_objective_binding",
            "direct_f64_delta_mean",
            "legacy_f32_delta_sha256",
            "legacy_f32_delta_mean",
            "pinned_V9_f32_delta_mean",
            "legacy_V9_D32_hash_and_scalar_replayed",
            "GL4_node_receipt_sha256s",
        ):
            if key not in base:
                raise ValueError("pinned V10 endpoint extension differs")
            base.pop(key)
        if (
            artifact
            != v10diag.token_v1._domain_sha256(
                base, domain=v10diag._ENDPOINT_REPLAY_DOMAIN
            )
            or example in endpoints
            or row.get("precision_evidence_artifact_sha256")
            != evidence_artifacts.get(example)
            or row.get("legacy_V9_endpoint_receipts_replayed_exactly") is not True
            or row.get("legacy_V9_D32_hash_and_scalar_replayed") is not True
        ):
            raise RuntimeError("pinned V10 endpoint receipt drifted")
        f32 = dict(
            _require_mapping(
                row.get("f32_objective_binding"), label="pinned V10 f32 binding"
            )
        )
        f64 = dict(
            _require_mapping(
                row.get("f64_objective_binding"), label="pinned V10 f64 binding"
            )
        )
        f32_artifact = v10diag._require_sha256(
            f32.pop("artifact_sha256", None), label="pinned V10 f32 binding"
        )
        f64_artifact = v10diag._require_sha256(
            f64.pop("artifact_sha256", None), label="pinned V10 f64 binding"
        )
        raw_bindings = f64.get("path_f64_node_bindings")
        if (
            f32_artifact
            != v10diag.token_v1._domain_sha256(
                f32, domain=v10diag._F32_OBJECTIVE_DOMAIN
            )
            or f64_artifact
            != v10diag.token_v1._domain_sha256(
                f64, domain=v10diag._F64_OBJECTIVE_DOMAIN
            )
            or f32.get("objective_dtype") != "torch.float32"
            or f64.get("objective_dtype") != "torch.float64"
            or f64.get("captured_objectives_equal_independent_bitwise") is not True
            or not isinstance(raw_bindings, list)
            or len(raw_bindings) != _EXPECTED_NODES_PER_PROMPT
        ):
            raise RuntimeError("pinned V10 objective binding drifted")
        for node_index, binding_raw in enumerate(raw_bindings):
            binding = _require_mapping(
                binding_raw, label="pinned V10 f64 node binding"
            )
            if (
                binding.get("node_index") != node_index
                or binding.get("path_fraction_hex")
                != v10diag.GL4_UNIT_INTERVAL_NODES[node_index].hex()
                or binding.get("quadrature_weight_hex")
                != v10diag.GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
                or binding.get("captured_equals_independent_bitwise") is not True
            ):
                raise ValueError("pinned V10 f64 node binding differs")
            node_extensions[(example, node_index)] = binding
        endpoints[example] = row
        families[example] = family

    nodes: dict[tuple[str, int], Mapping[str, object]] = {}
    for raw in raw_nodes:
        row = dict(_require_mapping(raw, label="pinned V10 path node"))
        receipt = v10diag._require_sha256(
            row.pop("receipt_sha256", None), label="pinned V10 path node receipt"
        )
        if receipt != v10diag.token_v1._domain_sha256(
            row, domain=v10diag._NODE_RECEIPT_DOMAIN
        ):
            raise RuntimeError("pinned V10 path node receipt drifted")
        row["receipt_sha256"] = receipt
        example = v10diag.token_v1._identifier(
            row.get("example_id"), label="pinned V10 node example"
        )
        node_index = row.get("node_index")
        key = (example, node_index) if type(node_index) is int else (example, -1)
        binding = node_extensions.get(key)
        if (
            example not in endpoints
            or type(node_index) is not int
            or not 0 <= node_index < _EXPECTED_NODES_PER_PROMPT
            or row.get("family_id") != families[example]
            or row.get("path_fraction_hex")
            != v10diag.GL4_UNIT_INTERVAL_NODES[node_index].hex()
            or row.get("quadrature_weight_hex")
            != v10diag.GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
            or row.get("objective_dtype") != "torch.float64"
            or row.get("captured_equals_independent_bitwise") is not True
            or row.get("legacy_V9_node_replayed_exactly") is not True
            or binding is None
            or row.get("core_node_receipt_sha256")
            != binding.get("core_node_receipt_sha256")
            or row.get("token_teacher_kl_f64_sha256")
            != binding.get("token_teacher_kl_f64_sha256")
            or row.get("vjp_artifact_sha256") != binding.get("vjp_artifact_sha256")
            or key in nodes
        ):
            raise ValueError("pinned V10 path node identity differs")
        nodes[key] = row

    if (
        set(endpoints) != set(evidence_artifacts)
        or len(set(families.values())) != _EXPECTED_FAMILIES
        or any(
            sum(value == family for value in families.values()) != 2
            for family in set(families.values())
        )
        or any(
            (example, node) not in nodes
            for example in endpoints
            for node in range(_EXPECTED_NODES_PER_PROMPT)
        )
    ):
        raise RuntimeError("pinned V10 receipt universe differs")
    for example, endpoint in endpoints.items():
        expected = tuple(
            str(nodes[(example, node)]["receipt_sha256"])
            for node in range(_EXPECTED_NODES_PER_PROMPT)
        )
        raw_chain = endpoint.get("GL4_node_receipt_sha256s")
        if not isinstance(raw_chain, list) or tuple(raw_chain) != expected:
            raise RuntimeError("pinned V10 endpoint node chain differs")
    ordered = tuple(
        str(nodes[(example, node)]["receipt_sha256"])
        for example in sorted(endpoints)
        for node in range(_EXPECTED_NODES_PER_PROMPT)
    )
    if report.get("path_precision_node_receipt_set_sha256") != (
        v10diag.token_v1._domain_sha256(ordered, domain=v10diag._NODE_SET_DOMAIN)
    ):
        raise RuntimeError("pinned V10 path node receipt set drifted")
    comparison_artifact = v10diag._require_sha256(
        comparison.get("artifact_sha256"), label="pinned V10 comparison artifact"
    )
    return _PinnedV10Receipts(
        endpoints=endpoints,
        nodes=nodes,
        evidence_artifacts=evidence_artifacts,
        comparison_artifact_sha256=comparison_artifact,
    )


def _bitwise_equal_across_devices(left: Tensor, right: Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(
        torch.equal(
            left.detach().to(device="cpu").contiguous().view(torch.uint8),
            right.detach().to(device="cpu").contiguous().view(torch.uint8),
        )
    )


def _full_path_point_f64(
    scalar_h4: Tensor, joint_h4: Tensor, path_fraction: float
) -> tuple[Tensor, Tensor]:
    """Construct the exact cast-once V10 full-H4 point and direction."""

    if (
        not isinstance(scalar_h4, Tensor)
        or not isinstance(joint_h4, Tensor)
        or scalar_h4.ndim != 3
        or scalar_h4.shape != joint_h4.shape
        or scalar_h4.dtype != joint_h4.dtype
        or scalar_h4.device != joint_h4.device
        or scalar_h4.requires_grad
        or joint_h4.requires_grad
        or not scalar_h4.is_contiguous()
        or not joint_h4.is_contiguous()
        or isinstance(path_fraction, bool)
        or not isinstance(path_fraction, (int, float))
        or not 0.0 <= float(path_fraction) <= 1.0
    ):
        raise ValueError("V11 full-H4 cast-once path geometry differs")
    scalar_cpu = scalar_h4.detach().to(device="cpu", dtype=torch.float64)
    joint_cpu = joint_h4.detach().to(device="cpu", dtype=torch.float64)
    direction_cpu = (joint_cpu - scalar_cpu).contiguous()
    point_cpu = scalar_cpu.add(float(path_fraction) * direction_cpu).contiguous()
    return (
        point_cpu.to(device=scalar_h4.device).contiguous(),
        direction_cpu.to(device=scalar_h4.device).contiguous(),
    )


def _build_suffix_node_evidence(
    *, precision: object, observed: _ObservedNode, node_index: int
) -> CandidateJointStateSuffixJVPNodeEvidence:
    """Bridge runtime evidence into the wrapped pure-core hash domain."""

    core_receipt = precision.path_evidence.node_receipts[node_index]
    if observed.core_node_receipt is not core_receipt:
        raise RuntimeError("V11 suffix core receipt ownership differs")
    return CandidateJointStateSuffixJVPNodeEvidence(
        node_index=node_index,
        path_fraction=observed.path_fraction,
        quadrature_weight=observed.quadrature_weight,
        token_directional_derivative_f64=(
            observed.directional_token_teacher_kl_f64
        ),
        pinned_v10_node_receipt_artifact_sha256=core_receipt.artifact_sha256,
        suffix_runtime_receipt_sha256=observed.suffix_runtime_receipt_sha256,
        # These three are deliberately wrapped pure-core hashes.  The caller
        # has already authenticated the runtime-domain tensors bitwise.
        primal_token_teacher_kl_sha256=core_receipt.token_teacher_kl_sha256,
        provider_artifact_sha256=core_receipt.provider_artifact_sha256,
        execution_artifact_sha256=core_receipt.execution_artifact_sha256,
        path_h4_sha256=core_receipt.path_node_h4_sha256,
        supervised_grid_sha256=precision.path_evidence.supervised_grid_sha256,
        endpoint_pair_binding_sha256=(
            precision.path_evidence.endpoint_pair_binding_sha256
        ),
    )


class _SuffixJVPCollector:
    """Observe V10 nodes and retain only typed transient tensors plus receipts."""

    def __init__(self, *, context: object) -> None:
        self._context = context
        self._prompts: dict[str, _ObservedPrompt] = {}

    def __call__(self, **observation: object) -> None:
        context = observation.get("context")
        trace = observation.get("trace")
        model_inputs = observation.get("model_inputs")
        teacher_logits = observation.get("teacher_logits")
        endpoint_grid = observation.get("endpoint_grid")
        scalar_execution = observation.get("scalar_execution")
        joint_execution = observation.get("joint_execution")
        scalar_kl = observation.get("scalar_token_teacher_kl_f64")
        joint_kl = observation.get("joint_token_teacher_kl_f64")
        node_index = observation.get("node_index")
        path_fraction = observation.get("path_fraction")
        quadrature_weight = observation.get("quadrature_weight")
        path_vjp = observation.get("path_vjp")
        path_rows = observation.get("path_h4_rows")
        path_kl = observation.get("path_token_teacher_kl_f64")
        core_receipt = observation.get("core_node_receipt")
        if (
            context is not self._context
            or trace is None
            or not isinstance(model_inputs, Mapping)
            or not isinstance(teacher_logits, Tensor)
            or not isinstance(endpoint_grid, Tensor)
            or scalar_execution is None
            or joint_execution is None
            or not isinstance(scalar_kl, Tensor)
            or not isinstance(joint_kl, Tensor)
            or type(node_index) is not int
            or not 0 <= node_index < _EXPECTED_NODES_PER_PROMPT
            or isinstance(path_fraction, bool)
            or not isinstance(path_fraction, (int, float))
            or isinstance(quadrature_weight, bool)
            or not isinstance(quadrature_weight, (int, float))
            or path_vjp is None
            or not isinstance(path_rows, Tensor)
            or not isinstance(path_kl, Tensor)
            or core_receipt is None
        ):
            raise RuntimeError("V11 suffix observer contract differs")
        example = v10diag.token_v1._identifier(
            getattr(trace, "example_id", None), label="V11 suffix example"
        )
        family = v10diag.token_v1._identifier(
            getattr(trace, "family_id", None), label="V11 suffix family"
        )
        scalar_h4 = getattr(scalar_execution, "candidate_h4", None)
        joint_h4 = getattr(joint_execution, "candidate_h4", None)
        full_h4 = getattr(getattr(path_vjp, "execution", None), "candidate_h4", None)
        full_logits = getattr(getattr(path_vjp, "execution", None), "logits", None)
        if not all(
            isinstance(value, Tensor)
            for value in (scalar_h4, joint_h4, full_h4, full_logits)
        ):
            raise RuntimeError("V11 suffix observer executions differ")
        scalar_h4 = scalar_h4.detach().contiguous()
        joint_h4 = joint_h4.detach().contiguous()
        full_h4 = full_h4.detach().contiguous()
        full_logits = full_logits.detach().contiguous()
        scalar_kl = scalar_kl.detach().to(device="cpu").contiguous()
        joint_kl = joint_kl.detach().to(device="cpu").contiguous()
        path_kl = path_kl.detach().to(device="cpu").contiguous()

        state = self._prompts.get(example)
        if state is None:
            if node_index != 0:
                raise RuntimeError("V11 suffix prompt did not begin at GL4 node zero")
            adapter = getattr(context, "adapter", None)
            sequence = adapter.prepare_sequence(model_inputs)
            _point_zero, direction = _full_path_point_f64(
                scalar_h4, joint_h4, 0.0
            )
            runtime = Gemma3L3L4H4SuffixJVPRuntime(
                adapter,
                sequence,
                teacher_logits=teacher_logits.detach().contiguous(),
                supervised_indices=endpoint_grid.detach()
                .to(device="cpu")
                .clone()
                .contiguous(),
            )
            state = _ObservedPrompt(
                example_id=example,
                family_id=family,
                scalar_h4=scalar_h4.clone().contiguous(),
                scalar_token_kl_f64=scalar_kl.clone().contiguous(),
                joint_h4=joint_h4.clone().contiguous(),
                joint_token_kl_f64=joint_kl.clone().contiguous(),
                direction_h4_f64=direction,
                runtime=runtime,
            )
            self._prompts[example] = state
        elif (
            state.family_id != family
            or node_index != len(state.nodes)
            or not _bitwise_equal_across_devices(state.scalar_h4, scalar_h4)
            or not _bitwise_equal_across_devices(state.joint_h4, joint_h4)
            or not _bitwise_equal_across_devices(
                state.scalar_token_kl_f64, scalar_kl
            )
            or not _bitwise_equal_across_devices(state.joint_token_kl_f64, joint_kl)
        ):
            raise RuntimeError("V11 suffix prompt endpoint state drifted")

        point, direction = _full_path_point_f64(
            state.scalar_h4, state.joint_h4, float(path_fraction)
        )
        if not _bitwise_equal_across_devices(direction, state.direction_h4_f64):
            raise RuntimeError("V11 suffix direction drifted between GL4 nodes")
        result = state.runtime.execute(
            point,
            direction,
            full_h4=full_h4,
            full_logits=full_logits,
        )
        if not isinstance(result, Gemma3L3L4H4SuffixJVP):
            raise RuntimeError("V11 suffix runtime returned the wrong result type")
        result.validate_integrity()
        support_indices = getattr(trace, "support_indices", None)
        if not isinstance(support_indices, Tensor):
            raise RuntimeError("V11 suffix support index contract differs")
        replay_rows = (
            full_h4.detach()
            .to(device="cpu")[0]
            .index_select(0, support_indices.detach().to(device="cpu"))
            .contiguous()
        )
        if (
            not _bitwise_equal_across_devices(replay_rows, path_rows)
            or not _bitwise_equal_across_devices(
                result.primal_token_teacher_kl, path_kl
            )
            or result.path_h4_sha256
            != v10diag._runtime_tensor_sha256(point)
            or result.full_h4_sha256
            != v10diag._runtime_tensor_sha256(full_h4)
            or result.full_logits_sha256
            != v10diag._runtime_tensor_sha256(full_logits)
            or result.primal_token_teacher_kl_sha256
            != v10diag._runtime_tensor_sha256(path_kl)
            or result.suffix_segment_call_count != 13
            or result.logit_projection_call_count != 1
            or result.h4_dtype_cast_count != 1
        ):
            raise RuntimeError("V11 suffix runtime did not replay the V10 primal")

        core_artifact = v10diag._require_sha256(
            getattr(core_receipt, "artifact_sha256", None),
            label="V11 core node artifact",
        )
        runtime_metadata = result.metadata()
        runtime_receipt: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "node_index": node_index,
            "path_fraction_hex": float(path_fraction).hex(),
            "quadrature_weight_hex": float(quadrature_weight).hex(),
            "pinned_v10_core_node_receipt_artifact_sha256": core_artifact,
            "suffix_runtime_receipt": runtime_metadata,
            "runtime_path_h4_matches_constructed_f64_path": True,
            "runtime_full_h4_matches_V10_full_node": True,
            "runtime_full_logits_matches_V10_full_node": True,
            "runtime_primal_token_KL_matches_V10_f64_bitwise": True,
            "core_domain_hashes_used_only_after_bitwise_runtime_replay": True,
            "raw_tensors_serialized": False,
        }
        runtime_receipt["receipt_sha256"] = v10diag.token_v1._domain_sha256(
            runtime_receipt, domain=_RUNTIME_NODE_DOMAIN
        )
        state.nodes[node_index] = _ObservedNode(
            node_index=node_index,
            path_fraction=float(path_fraction),
            quadrature_weight=float(quadrature_weight),
            directional_token_teacher_kl_f64=(
                result.directional_token_teacher_kl.detach()
                .to(device="cpu")
                .clone()
                .contiguous()
            ),
            core_node_receipt=core_receipt,
            suffix_runtime_receipt_sha256=result.artifact_sha256,
        )
        state.runtime_receipts[node_index] = runtime_receipt
        state.live_node_h4[node_index] = full_h4.clone().contiguous()
        state.node_token_kl_f64[node_index] = path_kl.clone().contiguous()

    def _discrete_cast_receipt(self, state: _ObservedPrompt) -> Mapping[str, object]:
        roles = (
            "scalar_endpoint",
            "GL4_node_0",
            "GL4_node_1",
            "GL4_node_2",
            "GL4_node_3",
            "joint_endpoint",
        )
        fractions = (
            0.0,
            *v10diag.GL4_UNIT_INTERVAL_NODES,
            1.0,
        )
        h4_points = (
            state.scalar_h4,
            *(state.live_node_h4[index] for index in range(4)),
            state.joint_h4,
        )
        ideal_h4_points = tuple(
            _full_path_point_f64(
                state.scalar_h4,
                state.joint_h4,
                fraction,
            )[0]
            for fraction in fractions
        )
        kl_points = (
            state.scalar_token_kl_f64,
            *(state.node_token_kl_f64[index] for index in range(4)),
            state.joint_token_kl_f64,
        )
        intervals = tuple(
            gemma3_l3_l4_h4_discrete_cast_interval_stats(
                left_h4,
                right_h4,
                ideal_left_h4_f64=ideal_left_h4,
                ideal_right_h4_f64=ideal_right_h4,
                left_path_fraction=left_fraction,
                right_path_fraction=right_fraction,
                left_token_teacher_kl=left_kl,
                right_token_teacher_kl=right_kl,
            )
            for (
                left_h4,
                right_h4,
                ideal_left_h4,
                ideal_right_h4,
                left_fraction,
                right_fraction,
                left_kl,
                right_kl,
            ) in zip(
                h4_points[:-1],
                h4_points[1:],
                ideal_h4_points[:-1],
                ideal_h4_points[1:],
                fractions[:-1],
                fractions[1:],
                kl_points[:-1],
                kl_points[1:],
                strict=True,
            )
        )
        for interval in intervals:
            interval.validate_integrity()
        if any(
            intervals[index].right_h4_sha256
            != intervals[index + 1].left_h4_sha256
            or intervals[index].ideal_right_h4_f64_sha256
            != intervals[index + 1].ideal_left_h4_f64_sha256
            or intervals[index].right_token_teacher_kl_sha256
            != intervals[index + 1].left_token_teacher_kl_sha256
            for index in range(len(intervals) - 1)
        ):
            raise RuntimeError("V11 discrete-cast interval chain differs")
        receipt: dict[str, object] = {
            "example_id": state.example_id,
            "family_id": state.family_id,
            "point_roles": roles,
            "path_fraction_hexes": tuple(value.hex() for value in fractions),
            "point_h4_sha256s": tuple(
                v10diag._runtime_tensor_sha256(value) for value in h4_points
            ),
            "ideal_point_h4_f64_sha256s": tuple(
                v10diag._runtime_tensor_sha256(value)
                for value in ideal_h4_points
            ),
            "point_token_teacher_kl_f64_sha256s": tuple(
                v10diag._runtime_tensor_sha256(value) for value in kl_points
            ),
            "adjacent_intervals": tuple(value.metadata() for value in intervals),
            "interval_count": len(intervals),
            "coordinate_interval_evaluation_count": sum(
                value.coordinate_count for value in intervals
            ),
            "ideal_changed_coordinate_interval_count": sum(
                value.ideal_changed_coordinate_count for value in intervals
            ),
            "live_changed_coordinate_interval_count": sum(
                value.live_changed_coordinate_count for value in intervals
            ),
            "preserved_change_coordinate_interval_count": sum(
                value.preserved_change_coordinate_count for value in intervals
            ),
            "cast_collision_coordinate_interval_count": sum(
                value.cast_collision_coordinate_count for value in intervals
            ),
            "static_coordinate_interval_count": sum(
                value.static_coordinate_count for value in intervals
            ),
            "unchanged_live_coordinate_interval_count": sum(
                value.unchanged_live_coordinate_count for value in intervals
            ),
            "ideal_displacement_squared_l2_sum": sum(
                value.ideal_displacement_squared_l2 for value in intervals
            ),
            "live_displacement_squared_l2_sum": sum(
                value.live_displacement_squared_l2 for value in intervals
            ),
            "token_teacher_kl_delta_squared_l2_sum": sum(
                float(value.token_teacher_kl_delta_squared_l2)
                for value in intervals
            ),
            "token_teacher_kl_normalized_secant_squared_l2_sum": sum(
                float(value.token_teacher_kl_normalized_secant_squared_l2)
                for value in intervals
            ),
            "cast_collision_excludes_static_coordinates": True,
            "all_six_points_and_five_adjacent_intervals_authenticated": True,
            "descriptive_only_no_fit_or_correction": True,
            "raw_tensors_serialized": False,
        }
        receipt["artifact_sha256"] = v10diag.token_v1._domain_sha256(
            receipt, domain=_DISCRETE_PROMPT_DOMAIN
        )
        return receipt

    def finalize(
        self,
        precision_evidence: Sequence[object],
    ) -> _SuffixCollectionResult:
        values = tuple(sorted(precision_evidence, key=lambda value: value.example_id))
        if (
            len(values) != _EXPECTED_PROMPTS
            or set(self._prompts) != {value.example_id for value in values}
            or len({value.family_id for value in values}) != _EXPECTED_FAMILIES
        ):
            raise RuntimeError("V11 suffix prompt universe differs")
        evidence: list[CandidateJointStateSuffixJVPEvidence] = []
        runtime_receipts: list[Mapping[str, object]] = []
        discrete_receipts: list[Mapping[str, object]] = []
        candidate_support_observations = 0
        full_h4_row_evaluations = 0
        for precision in values:
            precision.validate_integrity()
            state = self._prompts[precision.example_id]
            if (
                state.family_id != precision.family_id
                or tuple(sorted(state.nodes)) != (0, 1, 2, 3)
                or tuple(sorted(state.runtime_receipts)) != (0, 1, 2, 3)
                or tuple(sorted(state.live_node_h4)) != (0, 1, 2, 3)
                or tuple(sorted(state.node_token_kl_f64)) != (0, 1, 2, 3)
            ):
                raise RuntimeError("V11 suffix prompt collection is incomplete")
            nodes: list[CandidateJointStateSuffixJVPNodeEvidence] = []
            for node_index, _core_receipt in enumerate(
                precision.path_evidence.node_receipts
            ):
                observed = state.nodes[node_index]
                nodes.append(
                    _build_suffix_node_evidence(
                        precision=precision,
                        observed=observed,
                        node_index=node_index,
                    )
                )
                runtime_receipts.append(state.runtime_receipts[node_index])
            prompt_evidence = CandidateJointStateSuffixJVPEvidence(
                precision_evidence=precision,
                nodes=tuple(nodes),
            )
            prompt_evidence.validate_integrity()
            evidence.append(prompt_evidence)
            candidate_support_observations += precision.h4_shape[0] * 4
            full_h4_row_evaluations += (
                int(state.scalar_h4.shape[0])
                * int(state.scalar_h4.shape[1])
                * 4
            )
            discrete_receipts.append(self._discrete_cast_receipt(state))

        comparison = summarize_candidate_joint_state_suffix_jvp(evidence)
        ordered_runtime = tuple(
            sorted(
                runtime_receipts,
                key=lambda value: (
                    str(value["example_id"]),
                    int(value["node_index"]),
                ),
            )
        )
        ordered_discrete = tuple(
            sorted(discrete_receipts, key=lambda value: str(value["example_id"]))
        )
        runtime_set = v10diag.token_v1._domain_sha256(
            tuple(str(value["receipt_sha256"]) for value in ordered_runtime),
            domain=_RUNTIME_NODE_SET_DOMAIN,
        )
        discrete_set = v10diag.token_v1._domain_sha256(
            tuple(str(value["artifact_sha256"]) for value in ordered_discrete),
            domain=_DISCRETE_SET_DOMAIN,
        )
        node_count = len(ordered_runtime)
        segment_calls = sum(
            int(value["suffix_runtime_receipt"]["suffix_segment_call_count"])
            for value in ordered_runtime
        )
        projections = sum(
            int(value["suffix_runtime_receipt"]["logit_projection_call_count"])
            for value in ordered_runtime
        )
        casts = sum(
            int(value["suffix_runtime_receipt"]["h4_dtype_cast_count"])
            for value in ordered_runtime
        )
        supervised_token_kl_elements = sum(
            int(value["suffix_runtime_receipt"]["token_count"])
            for value in ordered_runtime
        )
        outside_support_row_evaluations = (
            full_h4_row_evaluations - candidate_support_observations
        )
        resources = {
            "suffix_jvp_evaluation_count": node_count,
            "suffix_differentiated_token_KL_evaluation_count": node_count,
            "suffix_full_token_KL_crosscheck_evaluation_count": node_count,
            "suffix_forward_structural_segment_call_count": segment_calls,
            "suffix_logit_projection_call_count": projections,
            "suffix_h4_dtype_cast_count": casts,
            "suffix_directional_token_KL_element_count": (
                supervised_token_kl_elements
            ),
            "suffix_primal_token_KL_element_count": supervised_token_kl_elements,
            "suffix_full_crosscheck_token_KL_element_count": (
                supervised_token_kl_elements
            ),
            "suffix_full_H4_row_evaluation_count": full_h4_row_evaluations,
            "candidate_support_row_observation_count": (
                candidate_support_observations
            ),
            "suffix_outside_candidate_support_H4_row_evaluation_count": (
                outside_support_row_evaluations
            ),
            "suffix_forward_structural_row_layer_call_count": (
                full_h4_row_evaluations * 13
            ),
            "candidate_support_row_layer_observation_count": (
                candidate_support_observations * 13
            ),
            "suffix_outside_candidate_support_row_layer_call_count": (
                outside_support_row_evaluations * 13
            ),
            "discrete_cast_interval_count": _EXPECTED_PROMPTS * 5,
            "discrete_cast_endpoint_validation_cast_count": (
                _EXPECTED_PROMPTS * 5 * 2
            ),
        }
        if (
            node_count != _EXPECTED_SUFFIX_JVP_EVALUATIONS
            or segment_calls != _EXPECTED_SUFFIX_SEGMENT_CALLS
            or projections != _EXPECTED_SUFFIX_PROJECTIONS
            or casts != _EXPECTED_SUFFIX_DTYPE_CASTS
            or supervised_token_kl_elements
            != _EXPECTED_SUFFIX_SUPERVISED_TOKEN_KL_ELEMENTS
            or candidate_support_observations
            != _EXPECTED_SUFFIX_H4_SUPPORT_EVALUATIONS
            or full_h4_row_evaluations
            != _EXPECTED_SUFFIX_FULL_H4_ROW_EVALUATIONS
            or resources["suffix_forward_structural_row_layer_call_count"]
            != _EXPECTED_SUFFIX_FULL_H4_ROW_LAYER_CALLS
            or resources["candidate_support_row_layer_observation_count"]
            != _EXPECTED_SUFFIX_SUPPORT_ROW_LAYER_OBSERVATIONS
            or outside_support_row_evaluations
            != _EXPECTED_SUFFIX_OUTSIDE_SUPPORT_ROW_EVALUATIONS
            or resources[
                "suffix_outside_candidate_support_row_layer_call_count"
            ]
            != _EXPECTED_SUFFIX_OUTSIDE_SUPPORT_ROW_LAYER_CALLS
        ):
            raise RuntimeError(
                "V11 suffix runtime resource accounting differs: "
                f"observed={resources!r}"
            )
        return _SuffixCollectionResult(
            evidence=tuple(evidence),
            comparison=comparison,
            runtime_node_receipts=ordered_runtime,
            runtime_node_receipt_set_sha256=runtime_set,
            discrete_cast_receipts=ordered_discrete,
            discrete_cast_receipt_set_sha256=discrete_set,
            resources=resources,
        )


def _authenticate_live_v10_replay(
    *,
    report: Mapping[str, object],
    pinned: _PinnedV10Receipts,
    live: object,
    fits: Mapping[str, object],
    traces: Sequence[object],
) -> dict[str, object]:
    """Require callback-enabled V10 execution to serialize exactly as V10."""

    live_endpoints = getattr(live, "endpoint_receipts")
    live_nodes = getattr(live, "node_receipts")
    live_node_set = getattr(live, "node_receipt_set_sha256")
    live_comparison = getattr(live, "comparison")
    live_folds = tuple(fits[family].metadata() for family in sorted(fits))
    live_prompts = v10diag.v9diag.v5diag._endpoint_prompt_receipts(traces)
    if (
        _canonical(live_endpoints)
        != _canonical(report.get("endpoint_precision_receipts"))
        or _canonical(live_nodes)
        != _canonical(report.get("path_precision_node_receipts"))
        or live_node_set != report.get("path_precision_node_receipt_set_sha256")
        or _canonical(live_comparison.metadata())
        != _canonical(report.get("family_equal_objective_precision_comparison"))
        or _canonical(live_folds) != _canonical(report.get("folds"))
        or _canonical(live_prompts) != _canonical(report.get("prompt_receipts"))
        or live_comparison.artifact_sha256
        != pinned.comparison_artifact_sha256
        or tuple(sorted(value.example_id for value in live.evidence))
        != tuple(sorted(pinned.evidence_artifacts))
        or any(
            value.artifact_sha256 != pinned.evidence_artifacts[value.example_id]
            for value in live.evidence
        )
    ):
        raise RuntimeError("callback-enabled V10 serialization or evidence drifted")
    return {
        "v10_report_file_sha256": V10_REPORT_FILE_SHA256,
        "v10_report_sha256": V10_REPORT_SHA256,
        "v10_comparison_artifact_sha256": pinned.comparison_artifact_sha256,
        "endpoint_receipt_count": len(live_endpoints),
        "path_node_receipt_count": len(live_nodes),
        "endpoint_receipts_canonically_equal": True,
        "path_node_receipts_canonically_equal": True,
        "path_node_receipt_set_equal": True,
        "comparison_canonically_equal": True,
        "folds_canonically_equal": True,
        "prompt_receipts_canonically_equal": True,
        "precision_evidence_artifacts_equal": True,
        "optional_observer_did_not_change_default_V10_serialization": True,
    }


def _resource_accounting(
    *,
    v10_resources: Mapping[str, object],
    suffix_resources: Mapping[str, int],
) -> dict[str, object]:
    """Combine the exact V10 ledger with the separately labelled suffix work."""

    if (
        v10_resources.get("total_model_forward_count") != 224
        or v10_resources.get("total_backward_call_count") != 1039
        or v10_resources.get("total_candidate_support_row_executions") != 7756
        or suffix_resources.get("suffix_jvp_evaluation_count")
        != _EXPECTED_SUFFIX_JVP_EVALUATIONS
        or suffix_resources.get("suffix_differentiated_token_KL_evaluation_count")
        != _EXPECTED_SUFFIX_JVP_EVALUATIONS
        or suffix_resources.get("suffix_full_token_KL_crosscheck_evaluation_count")
        != _EXPECTED_SUFFIX_JVP_EVALUATIONS
        or suffix_resources.get("suffix_forward_structural_segment_call_count")
        != _EXPECTED_SUFFIX_SEGMENT_CALLS
        or suffix_resources.get("suffix_logit_projection_call_count")
        != _EXPECTED_SUFFIX_PROJECTIONS
        or suffix_resources.get("suffix_h4_dtype_cast_count")
        != _EXPECTED_SUFFIX_DTYPE_CASTS
        or any(
            suffix_resources.get(key)
            != _EXPECTED_SUFFIX_SUPERVISED_TOKEN_KL_ELEMENTS
            for key in (
                "suffix_directional_token_KL_element_count",
                "suffix_primal_token_KL_element_count",
                "suffix_full_crosscheck_token_KL_element_count",
            )
        )
        or suffix_resources.get("suffix_full_H4_row_evaluation_count")
        != _EXPECTED_SUFFIX_FULL_H4_ROW_EVALUATIONS
        or suffix_resources.get("suffix_forward_structural_row_layer_call_count")
        != _EXPECTED_SUFFIX_FULL_H4_ROW_LAYER_CALLS
        or suffix_resources.get("candidate_support_row_observation_count")
        != _EXPECTED_SUFFIX_H4_SUPPORT_EVALUATIONS
        or suffix_resources.get("candidate_support_row_layer_observation_count")
        != _EXPECTED_SUFFIX_SUPPORT_ROW_LAYER_OBSERVATIONS
        or suffix_resources.get(
            "suffix_outside_candidate_support_H4_row_evaluation_count"
        )
        != _EXPECTED_SUFFIX_OUTSIDE_SUPPORT_ROW_EVALUATIONS
        or suffix_resources.get(
            "suffix_outside_candidate_support_row_layer_call_count"
        )
        != _EXPECTED_SUFFIX_OUTSIDE_SUPPORT_ROW_LAYER_CALLS
        or suffix_resources.get("discrete_cast_interval_count") != 80
        or suffix_resources.get("discrete_cast_endpoint_validation_cast_count")
        != 160
    ):
        raise RuntimeError("V11 suffix JVP resource accounting differs")
    combined = int(v10_resources["total_candidate_support_row_executions"]) + int(
        suffix_resources["candidate_support_row_observation_count"]
    )
    if combined != _EXPECTED_COMBINED_SUPPORT_EVALUATIONS:
        raise RuntimeError("V11 combined support evaluation accounting differs")
    phase_order = tuple(v10_resources.get("phase_order", ())) + (
        "sixty_four_post_H4_suffix_forward_mode_JVPs",
        "six_point_five_interval_discrete_cast_description",
        "family_equal_suffix_JVP_VJP_finite_comparison",
    )
    return {
        **dict(v10_resources),
        **dict(suffix_resources),
        "phase_order": phase_order,
        "combined_candidate_and_suffix_h4_support_evaluations": combined,
        "suffix_segment_calls_are_forward_structural_calls_not_AD_sweep_count": True,
        "suffix_JVP_AD_internal_sweeps_not_misreported_as_model_backwards": True,
        "total_model_forward_count": 224,
        "total_backward_call_count": 1039,
        "total_candidate_support_row_executions": 7756,
        "tune_model_forward_count": 0,
        "selection_model_forward_count": 0,
        "step_grid_model_forward_count": 0,
        "fallback_model_forward_count": 0,
        "routing_model_forward_count": 0,
        "exact_suffix_jvp_evaluation_count_is_64": True,
        "exact_suffix_forward_structural_segment_call_count_is_832": True,
        "exact_suffix_forward_structural_row_layer_call_count_is_49244": True,
        "exact_suffix_full_H4_row_evaluation_count_is_3788": True,
        "exact_suffix_directional_token_KL_element_count_is_3212": True,
        "exact_suffix_primal_token_KL_element_count_is_3212": True,
        "exact_suffix_full_crosscheck_token_KL_element_count_is_3212": True,
        "exact_candidate_support_row_observation_count_is_3276": True,
        "exact_candidate_support_row_layer_observation_count_is_42588": True,
        "exact_suffix_outside_candidate_support_H4_row_count_is_512": True,
        "exact_suffix_outside_candidate_support_row_layer_count_is_6656": True,
        "exact_combined_support_evaluation_count_is_11032": True,
        "raw_logits_gradients_H4_or_JVP_tensors_retained_in_report": False,
        "FLOP_or_total_compute_claim": False,
    }


def _require_integrity_gate_results(gates: Mapping[str, bool]) -> None:
    failures = tuple(sorted(key for key, value in gates.items() if not value))
    if failures:
        raise RuntimeError(
            "V11 suffix JVP integrity failed before publication: "
            + ", ".join(failures)
        )


def _strict_forward_mode_receipts_valid(
    receipts: Sequence[Mapping[str, object]],
) -> bool:
    try:
        return len(receipts) == _EXPECTED_SUFFIX_JVP_EVALUATIONS and all(
            isinstance(value.get("suffix_runtime_receipt"), Mapping)
            and value["suffix_runtime_receipt"].get("ad_mechanism")
            == "torch.func.jvp.forward_mode"
            and value["suffix_runtime_receipt"].get("jvp_strict") is True
            and value["suffix_runtime_receipt"].get("jvp_has_aux") is True
            and value["suffix_runtime_receipt"].get("h4_dtype_cast_count") == 1
            for value in receipts
        )
    except (AttributeError, TypeError):
        return False


def _discrete_cast_receipts_valid(
    receipts: Sequence[Mapping[str, object]],
) -> bool:
    try:
        return len(receipts) == _EXPECTED_PROMPTS and all(
            value.get("interval_count") == 5
            and value.get(
                "all_six_points_and_five_adjacent_intervals_authenticated"
            )
            is True
            and value.get("descriptive_only_no_fit_or_correction") is True
            and value.get("cast_collision_excludes_static_coordinates") is True
            for value in receipts
        )
    except (AttributeError, TypeError):
        return False


def _safety_metadata() -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_JVP_tensors": False,
        "contains_token_teacher_kl_tensors": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_jvp_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    v5_report_path: Path | str = DEFAULT_V5_REPORT,
    v6_report_path: Path | str = DEFAULT_V6_REPORT,
    v7_report_path: Path | str = DEFAULT_V7_REPORT,
    v8_report_path: Path | str = DEFAULT_V8_REPORT,
    v9_report_path: Path | str = DEFAULT_V9_REPORT,
    v10_report_path: Path | str = DEFAULT_V10_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the no-fit, no-selection V11 suffix-JVP diagnostic."""

    destination = v10diag.token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V11 suffix JVP report")

    # Authenticate every frozen authority before any fresh model work.
    v10_report = _load_v10_report(v10_report_path)
    pinned_v10 = _index_v10_receipts(v10_report)
    parent = v10diag.v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = v10diag.v9diag.v4diag._load_v3_report(v3_report_path)
    v4_report = v10diag.v9diag.v5diag._load_v4_report(v4_report_path)
    v5_report = v10diag.v6diag._load_v5_report(v5_report_path)
    v6_report = v10diag.v7diag._load_v6_report(v6_report_path)
    v7_report = v10diag.v8diag._load_v7_report(v7_report_path)
    v8_report = v10diag.v9diag._load_v8_report(v8_report_path)
    v8_observations = v10diag.v9diag._index_v8_endpoint_observations(v8_report)
    v9_report = v10diag._load_v9_report(v9_report_path)
    pinned_v9 = v10diag._index_v9_receipts(v9_report)
    materialization = v10diag.token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=v10diag.token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=v10diag.token_v1.MATERIALIZATION_REPORT_SHA256,
        label="V11 suffix JVP rank320 materialization",
    )
    transfer = v10diag.token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=v10diag.token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=v10diag.token_v1.TRANSFER_REPORT_SHA256,
        label="V11 suffix JVP rank320 transfer",
    )
    transfer_receipts = v10diag.expanded._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = (
        v10diag.transfer_support._load_committed_basis(
            materialization_report_path=materialization_report_path,
            expected_materialization_report_sha256=(
                v10diag.token_v1.MATERIALIZATION_REPORT_SHA256
            ),
            basis_sidecar_path=basis_sidecar_path,
        )
    )
    context = v10diag.prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = v10diag.token_v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != _EXPECTED_PROMPTS or len(families) != _EXPECTED_FAMILIES:
            raise RuntimeError("V11 suffix JVP A16 panel differs")
        fits = {
            family: v10diag.fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        analytic = v10diag.v8diag._execute_v8_analytic_phases(
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
        v8_binding = v10diag.v9diag._authenticate_v8_live_lineage(
            v8_report=v8_report, analytic=analytic
        )
        v9_binding = v10diag._authenticate_live_lineage_against_v9(
            v9_report=v9_report, v8_binding=v8_binding, fits=fits
        )
        collector = _SuffixJVPCollector(context=context)
        live = v10diag._execute_live_precision_grid(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            analytic=analytic,
            v8_observations=v8_observations,
            pinned=pinned_v9,
            path_node_observer=collector,
        )
        suffix = collector.finalize(live.evidence)
        v10_replay = _authenticate_live_v10_replay(
            report=v10_report,
            pinned=pinned_v10,
            live=live,
            fits=fits,
            traces=traces,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    reproduced_v10_resources = v10diag._resource_accounting(
        endpoint_resources=endpoint_resources,
        gradient_resources=analytic.v7_phases.v6_phases.row_bank.resources,
        live_resources=live.resources,
        row_bank_candidate_support_row_executions=(
            v10diag.v8diag._row_bank_candidate_support_row_executions(
                analytic.v7_phases
            )
        ),
    )
    if _canonical(reproduced_v10_resources) != _canonical(v10_report["resources"]):
        raise RuntimeError("callback-enabled V10 resource serialization drifted")
    resources = _resource_accounting(
        v10_resources=reproduced_v10_resources,
        suffix_resources=suffix.resources,
    )
    comparison = suffix.comparison
    comparison_metadata = comparison.metadata()
    runtime_receipts = suffix.runtime_node_receipts
    discrete_receipts = suffix.discrete_cast_receipts
    integrity_gates = {
        "exact_V10_file_and_logical_hash_authenticated_before_model_work": (
            v10_replay["v10_report_file_sha256"] == V10_REPORT_FILE_SHA256
            and v10_replay["v10_report_sha256"] == V10_REPORT_SHA256
        ),
        "callback_enabled_V10_replays_all_16_endpoints_exactly": (
            v10_replay["endpoint_receipt_count"] == _EXPECTED_PROMPTS
            and v10_replay["endpoint_receipts_canonically_equal"] is True
        ),
        "callback_enabled_V10_replays_all_64_GL4_nodes_exactly": (
            v10_replay["path_node_receipt_count"]
            == _EXPECTED_SUFFIX_JVP_EVALUATIONS
            and v10_replay["path_node_receipts_canonically_equal"] is True
            and v10_replay["path_node_receipt_set_equal"] is True
        ),
        "optional_observer_preserves_default_V10_serialization": all(
            v10_replay[key] is True
            for key in (
                "comparison_canonically_equal",
                "folds_canonically_equal",
                "prompt_receipts_canonically_equal",
                "precision_evidence_artifacts_equal",
                "optional_observer_did_not_change_default_V10_serialization",
            )
        ),
        "all_64_suffix_primals_replay_V10_H4_logits_and_token_KL": (
            len(runtime_receipts) == _EXPECTED_SUFFIX_JVP_EVALUATIONS
            and all(
                value["runtime_path_h4_matches_constructed_f64_path"] is True
                and value["runtime_full_h4_matches_V10_full_node"] is True
                and value["runtime_full_logits_matches_V10_full_node"] is True
                and value["runtime_primal_token_KL_matches_V10_f64_bitwise"]
                is True
                for value in runtime_receipts
            )
        ),
        "all_64_suffix_derivatives_use_strict_true_forward_mode_once_cast": (
            _strict_forward_mode_receipts_valid(runtime_receipts)
        ),
        "pure_core_uses_wrapped_V10_hash_domain_after_runtime_bitwise_replay": (
            all(
                value["core_domain_hashes_used_only_after_bitwise_runtime_replay"]
                is True
                for value in runtime_receipts
            )
            and all(
                node.pinned_v10_node_receipt_artifact_sha256
                == evidence.precision_evidence.path_evidence.node_receipts[
                    node.node_index
                ].artifact_sha256
                and node.primal_token_teacher_kl_sha256
                == evidence.precision_evidence.path_evidence.node_receipts[
                    node.node_index
                ].token_teacher_kl_sha256
                and node.path_h4_sha256
                == evidence.precision_evidence.path_evidence.node_receipts[
                    node.node_index
                ].path_node_h4_sha256
                for evidence in suffix.evidence
                for node in evidence.nodes
            )
        ),
        "all_16_discrete_cast_receipts_cover_scalar_four_nodes_and_joint": (
            _discrete_cast_receipts_valid(discrete_receipts)
        ),
        "V11_replays_exact_V10_comparison_inside_pure_core": (
            comparison.replayed_v10_comparison_artifact_sha256
            == pinned_v10.comparison_artifact_sha256
            and comparison.vjp_closure_passed == live.comparison.closure_passed
        ),
        "no_fit_search_selection_correction_fallback_or_routing": (
            resources["tune_model_forward_count"] == 0
            and resources["selection_model_forward_count"] == 0
            and resources["step_grid_model_forward_count"] == 0
            and resources["fallback_model_forward_count"] == 0
            and resources["routing_model_forward_count"] == 0
        ),
        "exact_V10_model_forward_backward_and_support_ledger_replayed": (
            resources["total_model_forward_count"] == 224
            and resources["total_backward_call_count"] == 1039
            and resources["total_candidate_support_row_executions"] == 7756
        ),
        "exact_suffix_and_combined_resource_ledger": (
            resources["suffix_jvp_evaluation_count"] == 64
            and resources["suffix_forward_structural_segment_call_count"] == 832
            and resources["suffix_forward_structural_row_layer_call_count"]
            == 49244
            and resources["suffix_full_H4_row_evaluation_count"] == 3788
            and resources["suffix_directional_token_KL_element_count"] == 3212
            and resources["suffix_primal_token_KL_element_count"] == 3212
            and resources["suffix_full_crosscheck_token_KL_element_count"]
            == 3212
            and resources["candidate_support_row_observation_count"] == 3276
            and resources["candidate_support_row_layer_observation_count"]
            == 42588
            and resources[
                "suffix_outside_candidate_support_H4_row_evaluation_count"
            ]
            == 512
            and resources[
                "suffix_outside_candidate_support_row_layer_call_count"
            ]
            == 6656
            and resources[
                "combined_candidate_and_suffix_h4_support_evaluations"
            ]
            == 11032
        ),
    }
    _require_integrity_gate_results(integrity_gates)
    classification = comparison.classification
    passed = classification == (
        "suffix_adjoint_passed_both_closures_established_same_a"
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_suffix_JVP_diagnostic",
            "outer_held_prompt_count": _EXPECTED_PROMPTS,
            "outer_family_count": _EXPECTED_FAMILIES,
            "prompts_per_outer_family": 2,
            "scalar_endpoint": "exact_frozen_full_seven_V6_scalar_control",
            "joint_endpoint": "exact_frozen_full_seven_V7_joint_field",
            "path": "exact_V10_realized_cast_once_scalar_to_joint_H4_path",
            "quadrature_rule": "fixed_gauss_legendre_order_4_on_unit_interval",
            "suffix": "native_Gemma_segments_layer_5_through_layer_17_then_logits",
            "suffix_derivative": (
                "forward_mode_token_teacher_KL_JVP_along_complete_H4_"
                "scalar_to_joint_displacement"
            ),
            "comparison": "J64_suffix_JVP_vs_P64_V10_VJP_vs_D64_direct_finite",
            "discrete_cast_points": "scalar_plus_four_GL4_nodes_plus_joint",
            "discrete_cast_intervals": 5,
            "aggregation": "equal_token_then_equal_prompt_then_equal_family",
            "thresholds": {
                "adjoint_relative_RMSE_maximum": ADJOINT_RELATIVE_RMSE_MAXIMUM,
                "overall_closure_relative_RMSE_maximum": (
                    v10diag.OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "every_family_closure_relative_RMSE_maximum": (
                    v10diag.FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "overall_closure_cosine_minimum": v10diag.CLOSURE_COSINE_MINIMUM,
            },
            "v11_fit_performed": False,
            "search_or_selection_performed": False,
            "correction_or_refit_performed": False,
            "fallback_or_routing_allowed": False,
            "result_can_authorize_serving_or_model_mutation": False,
        },
        "v10_control_binding": {
            "file": str(v10_report_path),
            "file_sha256": V10_REPORT_FILE_SHA256,
            "report_sha256": V10_REPORT_SHA256,
            "schema": v10_report.get("schema"),
            "classification": v10_report.get("classification"),
            "passed": v10_report.get("passed"),
            "live_reproduction": v10_replay,
            "v9_live_lineage_reproduction": v9_binding,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": (
                v10diag.token_v1.MATERIALIZATION_REPORT_FILE_SHA256
            ),
            "materialization_report_sha256": (
                v10diag.token_v1.MATERIALIZATION_REPORT_SHA256
            ),
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": (
                v10diag.token_v1.TRANSFER_REPORT_FILE_SHA256
            ),
            "transfer_report_sha256": v10diag.token_v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": v10diag.v9diag.v5diag._endpoint_prompt_receipts(traces),
        "replayed_v10_endpoint_precision_receipts": live.endpoint_receipts,
        "replayed_v10_path_precision_node_receipts": live.node_receipts,
        "replayed_v10_path_precision_node_receipt_set_sha256": (
            live.node_receipt_set_sha256
        ),
        "suffix_jvp_evidence": tuple(value.metadata() for value in suffix.evidence),
        "suffix_runtime_node_receipts": runtime_receipts,
        "suffix_runtime_node_receipt_set_sha256": (
            suffix.runtime_node_receipt_set_sha256
        ),
        "discrete_cast_prompt_receipts": discrete_receipts,
        "discrete_cast_prompt_receipt_set_sha256": (
            suffix.discrete_cast_receipt_set_sha256
        ),
        "family_equal_suffix_jvp_comparison": comparison_metadata,
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "outcome_matrix": {
            "outcome": classification,
            "integrity_completed": True,
            "adjoint_established": comparison.adjoint_passed,
            "suffix_JVP_closure_established": comparison.jvp_closure_passed,
            "replayed_V10_VJP_closure_established": comparison.vjp_closure_passed,
            "discrete_cast_interval_description_published": True,
            "fresh_confirmation_authorized_next": False,
            "selection_or_serving_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_diagnostic_only": True,
            "V10_endpoint_node_and_summary_serialization_replayed_exactly": True,
            "suffix_primal_parity_is_integrity_not_scientific_success": True,
            "suffix_JVP_is_forward_mode_not_a_fitted_correction": True,
            "discrete_cast_receipts_are_descriptive_not_causal_proof": True,
            "scientific_miss_is_publishable_after_integrity": True,
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
    v10diag.frozen._scalar_report(report)
    reservation = v10diag.frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = v10diag.frozen._json_sha256(
            report, domain=_REPORT_DOMAIN
        )
        stage = v10diag.frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": v10diag.token_v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Run the pinned V11 post-H4 suffix-JVP GL4 diagnostic."
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_jvp_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
