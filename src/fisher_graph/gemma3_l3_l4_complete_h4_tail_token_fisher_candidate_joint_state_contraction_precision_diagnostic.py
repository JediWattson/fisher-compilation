"""Locked V12 contraction-precision replay of the authenticated V11 run.

V12 adds no model or suffix evaluation.  A composite observer first validates
and streams the already-produced raw float32 V10 H4 gradient bank into the
pure contraction-precision ladder, then gives the untouched observation to
the unchanged V11 suffix-JVP collector.  The
ladder separates operation order, direction-cast rounding, product rounding,
and a fixed float32 reduction boundary.  It is a same-A diagnostic only: no
fit, search, selection, correction, routing, serving mutation, or compression
authorization is performed here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_jvp_diagnostic as v11diag
from .complete_h4_tail_candidate_joint_state_contraction_precision import (
    ADJOINT_RELATIVE_RMSE_MAXIMUM,
    CONTRACTION_CORRECTED_STAGE_ORDER,
    CONTRACTION_PUBLISHED_STAGE_ORDER,
    CandidateJointStateContractionPrecisionAccumulator,
    CandidateJointStateContractionPrecisionComparison,
    CandidateJointStateContractionPrecisionEvidence,
    summarize_candidate_joint_state_contraction_precision,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_V11_REPORT",
    "V11_REPORT_FILE_SHA256",
    "V11_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic",
    "main",
]


v10diag = v11diag.v10diag
DEFAULT_MATERIALIZATION_REPORT = v11diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v11diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v11diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v11diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v11diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v11diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v11diag.DEFAULT_V6_REPORT
DEFAULT_V7_REPORT = v11diag.DEFAULT_V7_REPORT
DEFAULT_V8_REPORT = v11diag.DEFAULT_V8_REPORT
DEFAULT_V9_REPORT = v11diag.DEFAULT_V9_REPORT
DEFAULT_V10_REPORT = v11diag.DEFAULT_V10_REPORT
DEFAULT_V11_REPORT = v11diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = v10diag.token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-contraction-precision-gl4-"
    "lofo-a-fit16-dev-v12.json"
)

V11_REPORT_FILE_SHA256 = (
    "7fe73e921d62308964af790c5ae39cf352da5ab0ba441543c22bb51841355e65"
)
V11_REPORT_SHA256 = (
    "b20d3a8d5775ff407fc2a5bf21a417ba2933f4816c67c7d6cfb2022a65df02e1"
)
V11_COMPARISON_ARTIFACT_SHA256 = (
    "052dd31cf6c1a9ac0b485b3940de327b6330fc830453c69553e360eb1b705b30"
)
V11_SUFFIX_RUNTIME_SET_SHA256 = (
    "d6be74f64e3a7311025fb32c1dc23bee2e5c53627f8a97d6c87ef9b0b87bff2b"
)
V11_DISCRETE_CAST_SET_SHA256 = (
    "75a558c89de4c137d487d9467601aecd8836eee69946c656c2c771a996632d3f"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_contraction_precision_gl4_lofo.v12"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-contraction-precision:v12\0"
_RAW_GRADIENT_DOMAIN = (
    b"fisher-graph:complete-h4-k64-contraction-raw-gradient:v12\0"
)
_RAW_GRADIENT_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-contraction-raw-gradient-set:v12\0"
)
_CAST_ONLY_JVP_DOMAIN = (
    b"fisher-graph:complete-h4-k64-contraction-cast-jvp:v12\0"
)
_CAST_ONLY_JVP_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-contraction-cast-jvp-set:v12\0"
)
_REDUCTION_ENVIRONMENT_DOMAIN = (
    b"fisher-graph:complete-h4-k64-contraction-reduction-environment:v12\0"
)

_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS = 16
_EXPECTED_NODES = 64
_EXPECTED_SUPERVISED_TOKENS = 803
_EXPECTED_FULL_H4_ROWS = 947
_EXPECTED_SUPPORT_H4_ROWS = 819
_EXPECTED_OUTSIDE_SUPPORT_H4_ROWS = 128
_EXPECTED_H4_WIDTH = 640
_EXPECTED_RAW_FULL_GRADIENT_ELEMENTS = 130_048_000
_EXPECTED_RAW_SUPPORT_GRADIENT_ELEMENTS = 113_602_560
_EXPECTED_V10_FINAL_CONTRACTION_PRODUCTS = 28_400_640
_EXPECTED_NODEWISE_CONTRACTION_COORDINATE_OBSERVATIONS = 454_410_240
_EXPECTED_ACTUAL_COORDINATE_PRODUCTS = 340_807_680
_EXPECTED_NODEWISE_GL4_WEIGHT_APPLICATIONS = 12_848
_EXPECTED_CAST_ONLY_JVP_ELEMENTS = 606_080
_EXPECTED_OUTSIDE_SUPPORT_ZERO_VALIDATION_ELEMENTS = 163_840


@dataclass(slots=True)
class _ObservedContractionPrompt:
    example_id: str
    family_id: str
    support_indices: Tensor
    full_displacement_f64: Tensor
    full_cast_tangent_f32: Tensor
    accumulator: CandidateJointStateContractionPrecisionAccumulator
    raw_gradient_receipts: dict[int, Mapping[str, object]] = field(
        default_factory=dict
    )
    cast_only_jvp_receipt: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class _ContractionCollectionResult:
    evidence: tuple[CandidateJointStateContractionPrecisionEvidence, ...]
    comparison: CandidateJointStateContractionPrecisionComparison
    suffix: object
    raw_gradient_receipts: tuple[Mapping[str, object], ...]
    raw_gradient_receipt_set_sha256: str
    cast_only_jvp_receipts: tuple[Mapping[str, object], ...]
    cast_only_jvp_receipt_set_sha256: str
    resources: Mapping[str, object]


def _canonical(value: object) -> object:
    return v11diag._canonical(value)


def _load_v11_report(path: Path | str) -> dict[str, object]:
    """Load only the exact inspected V11 suffix-JVP artifact."""

    report = v10diag.token_v1._load_pinned_report(
        path,
        expected_file_sha256=V11_REPORT_FILE_SHA256,
        expected_report_sha256=V11_REPORT_SHA256,
        label="candidate joint-state contraction precision V11 anchor",
    )
    comparison = report.get("family_equal_suffix_jvp_comparison")
    resources = report.get("resources")
    if (
        report.get("schema") != v11diag._SCHEMA
        or report.get("classification") != "suffix_adjoint_ambiguity_same_a"
        or report.get("passed") is not False
        or not isinstance(comparison, Mapping)
        or comparison.get("artifact_sha256")
        != V11_COMPARISON_ARTIFACT_SHA256
        or float(comparison.get("adjoint_relative_rmse", -1.0))
        != 0.0002564458592054882
        or float(comparison.get("jvp_closure_relative_rmse", -1.0))
        != 0.08675453755642996
        or float(comparison.get("vjp_closure_relative_rmse", -1.0))
        != 0.08673388652083237
        or report.get("suffix_runtime_node_receipt_set_sha256")
        != V11_SUFFIX_RUNTIME_SET_SHA256
        or report.get("discrete_cast_prompt_receipt_set_sha256")
        != V11_DISCRETE_CAST_SET_SHA256
        or not isinstance(resources, Mapping)
        or resources.get("total_model_forward_count") != 224
        or resources.get("total_backward_call_count") != 1039
        or resources.get("suffix_jvp_evaluation_count") != 64
        or resources.get("combined_candidate_and_suffix_h4_support_evaluations")
        != 11032
    ):
        raise RuntimeError("candidate joint-state suffix JVP V11 anchor differs")
    if (
        not isinstance(report.get("suffix_jvp_evidence"), list)
        or len(report["suffix_jvp_evidence"]) != _EXPECTED_PROMPTS
        or not isinstance(report.get("suffix_runtime_node_receipts"), list)
        or len(report["suffix_runtime_node_receipts"]) != _EXPECTED_NODES
        or not isinstance(report.get("discrete_cast_prompt_receipts"), list)
        or len(report["discrete_cast_prompt_receipts"]) != _EXPECTED_PROMPTS
    ):
        raise RuntimeError("candidate joint-state suffix JVP V11 evidence grid differs")
    return report


def _typed_reduction_environment() -> dict[str, object]:
    """Bind every public torch setting relevant to the fixed CPU reductions."""

    receipt: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "reduction_device_type": "cpu",
        "cpu_default_dtype": str(torch.get_default_dtype()),
        "raw_gradient_source_dtype": str(torch.float32),
        "promoted_gradient_dtype": str(torch.float64),
        "full_displacement_dtype": str(torch.float64),
        "cast_tangent_dtype": str(torch.float32),
        "P64_node_reduction_dtype": str(torch.float64),
        "P_dir_reduction_dtype": str(torch.float64),
        "P_prod_reduction_dtype": str(torch.float64),
        "P_live_product_and_reduction_dtype": str(torch.float32),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "reduction_order": (
            "canonical_contiguous_support_row_major_width_minor_flatten_"
            "then_typed_torch_sum"
        ),
        "internal_torch_sum_kernel_schedule_claimed": False,
        "environment_is_reproducibility_binding_not_cross_platform_bitwise_guarantee": (
            True
        ),
    }
    receipt["artifact_sha256"] = v10diag.token_v1._domain_sha256(
        receipt, domain=_REDUCTION_ENVIRONMENT_DOMAIN
    )
    return receipt


def _validate_contraction_receipts(
    *,
    raw_receipts: Sequence[Mapping[str, object]],
    raw_receipt_set_sha256: str,
    cast_receipts: Sequence[Mapping[str, object]],
    cast_receipt_set_sha256: str,
    reduction_environment: Mapping[str, object],
) -> dict[str, object]:
    """Rehash every mutable V12 receipt and its canonical ownership set."""

    raw = tuple(raw_receipts)
    casts = tuple(cast_receipts)
    environment = dict(reduction_environment)
    environment_artifact = v10diag._require_sha256(
        environment.pop("artifact_sha256", None),
        label="V12 typed reduction environment",
    )
    if environment_artifact != v10diag.token_v1._domain_sha256(
        environment, domain=_REDUCTION_ENVIRONMENT_DOMAIN
    ):
        raise RuntimeError("V12 typed reduction environment receipt drifted")
    if (
        environment.get("reduction_device_type") != "cpu"
        or environment.get("raw_gradient_source_dtype") != "torch.float32"
        or environment.get("promoted_gradient_dtype") != "torch.float64"
        or environment.get("P_live_product_and_reduction_dtype")
        != "torch.float32"
        or environment.get("internal_torch_sum_kernel_schedule_claimed") is not False
        or _canonical(reduction_environment)
        != _canonical(_typed_reduction_environment())
    ):
        raise RuntimeError("V12 typed reduction environment semantics differ")

    raw_keys: list[tuple[str, int]] = []
    raw_families: dict[str, str] = {}
    raw_artifacts: list[str] = []
    raw_hash_fields = (
        "support_indices_sha256",
        "raw_full_gradient_f32_sha256",
        "path_vjp_artifact_sha256",
        "selected_support_gradient_f32_sha256",
        "selected_then_promoted_gradient_f64_sha256",
        "pinned_v10_promoted_gradient_f64_sha256",
        "contraction_node_evidence_artifact_sha256",
        "pinned_v10_core_node_receipt_artifact_sha256",
    )
    for value in raw:
        payload = dict(value)
        artifact = v10diag._require_sha256(
            payload.pop("artifact_sha256", None),
            label="V12 raw gradient receipt",
        )
        if artifact != v10diag.token_v1._domain_sha256(
            payload, domain=_RAW_GRADIENT_DOMAIN
        ):
            raise RuntimeError("V12 raw gradient receipt drifted")
        example = v10diag.token_v1._identifier(
            payload.get("example_id"), label="V12 raw gradient receipt example"
        )
        family = v10diag.token_v1._identifier(
            payload.get("family_id"), label="V12 raw gradient receipt family"
        )
        node = payload.get("node_index")
        if (
            type(node) is not int
            or not 0 <= node < 4
            or payload.get("path_fraction_hex")
            != v10diag.GL4_UNIT_INTERVAL_NODES[node].hex()
            or payload.get("quadrature_weight_hex")
            != v10diag.GL4_UNIT_INTERVAL_WEIGHTS[node].hex()
            or payload.get("raw_full_gradient_dtype") != "torch.float32"
            or payload.get("raw_float32_selected_before_float64_promotion")
            is not True
            or payload.get("selected_promoted_equals_V10_bank_bitwise") is not True
            or payload.get("transient_gradient_retained_after_callback") is not False
            or payload.get("raw_tensors_serialized") is not False
        ):
            raise RuntimeError("V12 raw gradient receipt semantics differ")
        for key in raw_hash_fields:
            v10diag._require_sha256(
                payload.get(key), label=f"V12 raw gradient receipt {key}"
            )
        if example in raw_families and raw_families[example] != family:
            raise RuntimeError("V12 raw gradient family ownership differs")
        raw_families[example] = family
        raw_keys.append((example, node))
        raw_artifacts.append(artifact)
    expected_raw_keys = tuple(
        (example, node)
        for example in sorted(raw_families)
        for node in range(4)
    )
    if (
        len(raw) != _EXPECTED_NODES
        or len(raw_families) != _EXPECTED_PROMPTS
        or tuple(raw_keys) != expected_raw_keys
        or len(set(raw_keys)) != len(raw_keys)
    ):
        raise RuntimeError("V12 raw gradient receipt order or ownership differs")
    expected_raw_set = v10diag.token_v1._domain_sha256(
        tuple(raw_artifacts), domain=_RAW_GRADIENT_SET_DOMAIN
    )
    if (
        v10diag._require_sha256(
            raw_receipt_set_sha256, label="V12 raw gradient receipt set"
        )
        != expected_raw_set
    ):
        raise RuntimeError("V12 raw gradient receipt set drifted")

    cast_examples: list[str] = []
    cast_artifacts: list[str] = []
    cast_hash_fields = (
        "point_f64_sha256",
        "direction_f64_sha256",
        "primal_f32_sha256",
        "tangent_f32_sha256",
        "live_v10_full_h4_f32_sha256",
    )
    for value in casts:
        payload = dict(value)
        artifact = v10diag._require_sha256(
            payload.pop("artifact_sha256", None),
            label="V12 cast-only JVP receipt",
        )
        if artifact != v10diag.token_v1._domain_sha256(
            payload, domain=_CAST_ONLY_JVP_DOMAIN
        ):
            raise RuntimeError("V12 cast-only JVP receipt drifted")
        example = v10diag.token_v1._identifier(
            payload.get("example_id"), label="V12 cast-only JVP receipt example"
        )
        family = v10diag.token_v1._identifier(
            payload.get("family_id"), label="V12 cast-only JVP receipt family"
        )
        if (
            payload.get("node_index") != 0
            or payload.get("path_fraction_hex")
            != v10diag.GL4_UNIT_INTERVAL_NODES[0].hex()
            or payload.get("ad_mechanism") != "torch.func.jvp.forward_mode"
            or payload.get("jvp_strict") is not True
            or payload.get("primal_matches_direct_cast_bitwise") is not True
            or payload.get("primal_matches_live_V10_full_H4_bitwise") is not True
            or payload.get("tangent_matches_direct_cast_bitwise") is not True
            or payload.get("typed_reduction_environment_sha256")
            != environment_artifact
            or payload.get("model_or_suffix_segment_evaluated") is not False
            or payload.get("raw_tensors_serialized") is not False
            or raw_families.get(example) != family
        ):
            raise RuntimeError("V12 cast-only JVP receipt semantics differ")
        for key in cast_hash_fields:
            v10diag._require_sha256(
                payload.get(key), label=f"V12 cast-only JVP receipt {key}"
            )
        cast_examples.append(example)
        cast_artifacts.append(artifact)
    if (
        len(casts) != _EXPECTED_PROMPTS
        or tuple(cast_examples) != tuple(sorted(raw_families))
        or len(set(cast_examples)) != len(cast_examples)
    ):
        raise RuntimeError("V12 cast-only JVP receipt order or ownership differs")
    expected_cast_set = v10diag.token_v1._domain_sha256(
        tuple(cast_artifacts), domain=_CAST_ONLY_JVP_SET_DOMAIN
    )
    if (
        v10diag._require_sha256(
            cast_receipt_set_sha256, label="V12 cast-only JVP receipt set"
        )
        != expected_cast_set
    ):
        raise RuntimeError("V12 cast-only JVP receipt set drifted")
    return {
        "raw_gradient_receipt_count": len(raw),
        "cast_only_jvp_receipt_count": len(casts),
        "prompt_count": len(raw_families),
        "raw_gradient_receipt_set_sha256": expected_raw_set,
        "cast_only_jvp_receipt_set_sha256": expected_cast_set,
        "typed_reduction_environment_sha256": environment_artifact,
        "all_receipts_payload_set_order_and_ownership_rehashed": True,
    }


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return v11diag._bitwise_equal_across_devices(left, right)


def _full_displacement_and_tangent(
    scalar_h4: Tensor,
    joint_h4: Tensor,
) -> tuple[Tensor, Tensor]:
    _point, direction = v11diag._full_path_point_f64(scalar_h4, joint_h4, 0.0)
    if direction.ndim != 3 or direction.shape[0] != 1:
        raise RuntimeError("V12 requires a singleton-batch full-H4 direction")
    displacement = direction.detach().to(device="cpu")[0].clone().contiguous()
    tangent = displacement.to(torch.float32).contiguous()
    return displacement, tangent


def _cast_only_jvp_receipt(
    *,
    example_id: str,
    family_id: str,
    point_f64: Tensor,
    direction_f64: Tensor,
    live_h4_f32: Tensor,
    reduction_environment_sha256: str,
) -> Mapping[str, object]:
    """Validate the isolated float64-to-float32 tangent at GL4 node zero."""

    if (
        point_f64.dtype != torch.float64
        or direction_f64.dtype != torch.float64
        or not isinstance(live_h4_f32, Tensor)
        or live_h4_f32.dtype != torch.float32
        or live_h4_f32.shape != point_f64.shape
    ):
        raise RuntimeError("V12 cast-only JVP requires float64 primal and tangent")

    def cast_only(value: Tensor) -> Tensor:
        return value.to(torch.float32)

    primal, tangent = torch.func.jvp(
        cast_only,
        (point_f64,),
        (direction_f64,),
        strict=True,
    )
    expected_primal = point_f64.to(torch.float32).contiguous()
    expected_tangent = direction_f64.to(torch.float32).contiguous()
    if (
        not _bitwise_equal(primal, expected_primal)
        or not _bitwise_equal(tangent, expected_tangent)
        or not _bitwise_equal(primal, live_h4_f32)
    ):
        raise RuntimeError(
            "V12 isolated cast-only JVP differs from exact cast or live V10 H4"
        )
    receipt: dict[str, object] = {
        "example_id": v10diag.token_v1._identifier(
            example_id, label="V12 cast-only JVP example"
        ),
        "family_id": v10diag.token_v1._identifier(
            family_id, label="V12 cast-only JVP family"
        ),
        "node_index": 0,
        "path_fraction_hex": v10diag.GL4_UNIT_INTERVAL_NODES[0].hex(),
        "ad_mechanism": "torch.func.jvp.forward_mode",
        "jvp_strict": True,
        "function": "isolated_float64_to_float32_cast_only",
        "device_type": point_f64.device.type,
        "input_dtype": str(point_f64.dtype),
        "output_dtype": str(primal.dtype),
        "shape": tuple(int(size) for size in point_f64.shape),
        "element_count": int(point_f64.numel()),
        "point_f64_sha256": v10diag._runtime_tensor_sha256(point_f64),
        "direction_f64_sha256": v10diag._runtime_tensor_sha256(direction_f64),
        "primal_f32_sha256": v10diag._runtime_tensor_sha256(primal),
        "tangent_f32_sha256": v10diag._runtime_tensor_sha256(tangent),
        "live_v10_full_h4_f32_sha256": (
            v10diag._runtime_tensor_sha256(live_h4_f32)
        ),
        "primal_matches_direct_cast_bitwise": True,
        "primal_matches_live_V10_full_H4_bitwise": True,
        "tangent_matches_direct_cast_bitwise": True,
        "typed_reduction_environment_sha256": reduction_environment_sha256,
        "model_or_suffix_segment_evaluated": False,
        "raw_tensors_serialized": False,
    }
    receipt["artifact_sha256"] = v10diag.token_v1._domain_sha256(
        receipt, domain=_CAST_ONLY_JVP_DOMAIN
    )
    return receipt


class _CompositeContractionPrecisionCollector:
    """Validate raw V10 float32 banks, then run V11 on the untouched callback."""

    def __init__(self, *, context: object) -> None:
        self._context = context
        self._suffix = v11diag._SuffixJVPCollector(context=context)
        self._prompts: dict[str, _ObservedContractionPrompt] = {}
        self._environment = _typed_reduction_environment()
        self._raw_full_gradient_element_count = 0

    @property
    def reduction_environment(self) -> Mapping[str, object]:
        return dict(self._environment)

    def __call__(self, **observation: object) -> None:
        context = observation.get("context")
        trace = observation.get("trace")
        scalar_execution = observation.get("scalar_execution")
        joint_execution = observation.get("joint_execution")
        path_vjp = observation.get("path_vjp")
        promoted_gradient = observation.get("path_token_h4_gradients_f64")
        core_receipt = observation.get("core_node_receipt")
        node_index = observation.get("node_index")
        path_fraction = observation.get("path_fraction")
        quadrature_weight = observation.get("quadrature_weight")
        if (
            context is not self._context
            or trace is None
            or scalar_execution is None
            or joint_execution is None
            or path_vjp is None
            or not isinstance(promoted_gradient, Tensor)
            or core_receipt is None
            or type(node_index) is not int
            or not 0 <= node_index < 4
            or isinstance(path_fraction, bool)
            or not isinstance(path_fraction, (int, float))
            or isinstance(quadrature_weight, bool)
            or not isinstance(quadrature_weight, (int, float))
        ):
            raise RuntimeError("V12 contraction observer contract differs")
        example = v10diag.token_v1._identifier(
            getattr(trace, "example_id", None), label="V12 contraction example"
        )
        family = v10diag.token_v1._identifier(
            getattr(trace, "family_id", None), label="V12 contraction family"
        )
        support = getattr(trace, "support_indices", None)
        scalar_h4 = getattr(scalar_execution, "candidate_h4", None)
        joint_h4 = getattr(joint_execution, "candidate_h4", None)
        raw_gradient = getattr(path_vjp, "h4_gradients", None)
        path_execution = getattr(path_vjp, "execution", None)
        full_live_h4 = getattr(path_execution, "candidate_h4", None)
        validate_path_vjp = getattr(path_vjp, "validate_integrity", None)
        if (
            not isinstance(support, Tensor)
            or support.ndim != 1
            or support.dtype != torch.int64
            or not isinstance(scalar_h4, Tensor)
            or not isinstance(joint_h4, Tensor)
            or not isinstance(raw_gradient, Tensor)
            or raw_gradient.dtype != torch.float32
            or raw_gradient.ndim != 4
            or raw_gradient.shape[1] != 1
            or raw_gradient.shape[2:] != scalar_h4.shape[1:]
            or not isinstance(full_live_h4, Tensor)
            or full_live_h4.dtype != torch.float32
            or full_live_h4.shape != scalar_h4.shape
            or not callable(validate_path_vjp)
        ):
            raise RuntimeError("V12 raw float32 H4 gradient geometry differs")
        validate_path_vjp()
        path_vjp_artifact = v10diag._require_sha256(
            getattr(path_vjp, "artifact_sha256", None),
            label="V12 path VJP artifact",
        )
        core_node_index = getattr(core_receipt, "node_index", None)
        core_path_fraction = getattr(core_receipt, "path_fraction", None)
        core_quadrature_weight = getattr(core_receipt, "quadrature_weight", None)
        core_vjp_artifact = getattr(core_receipt, "vjp_artifact_sha256", None)
        if (
            core_node_index != node_index
            or not isinstance(core_path_fraction, float)
            or not isinstance(core_quadrature_weight, float)
            or core_path_fraction.hex() != float(path_fraction).hex()
            or core_quadrature_weight.hex() != float(quadrature_weight).hex()
            or core_vjp_artifact != path_vjp_artifact
        ):
            raise RuntimeError(
                "V12 observation differs from core node coordinates or VJP"
            )
        support_cpu = support.detach().to(device="cpu").clone().contiguous()
        state = self._prompts.get(example)
        if state is None:
            if node_index != 0:
                raise RuntimeError("V12 contraction prompt did not begin at node zero")
            displacement, tangent = _full_displacement_and_tangent(
                scalar_h4.detach().contiguous(), joint_h4.detach().contiguous()
            )
            accumulator = CandidateJointStateContractionPrecisionAccumulator(
                support_indices=support_cpu,
                full_displacement_f64=displacement,
                full_cast_tangent_f32=tangent,
            )
            point, direction = v11diag._full_path_point_f64(
                scalar_h4.detach().contiguous(),
                joint_h4.detach().contiguous(),
                float(path_fraction),
            )
            cast_receipt = _cast_only_jvp_receipt(
                example_id=example,
                family_id=family,
                point_f64=point,
                direction_f64=direction,
                live_h4_f32=full_live_h4.detach().contiguous(),
                reduction_environment_sha256=str(
                    self._environment["artifact_sha256"]
                ),
            )
            state = _ObservedContractionPrompt(
                example_id=example,
                family_id=family,
                support_indices=support_cpu,
                full_displacement_f64=displacement,
                full_cast_tangent_f32=tangent,
                accumulator=accumulator,
                cast_only_jvp_receipt=cast_receipt,
            )
            self._prompts[example] = state
        elif (
            state.family_id != family
            or node_index != state.accumulator.node_count
            or not torch.equal(state.support_indices, support_cpu)
        ):
            raise RuntimeError("V12 contraction prompt state drifted")

        raw_selected_f32 = (
            raw_gradient.detach()[:, 0]
            .index_select(1, support_cpu.to(device=raw_gradient.device))
            .to(device="cpu")
            .clone()
            .contiguous()
        )
        selected_promoted_f64 = raw_selected_f32.to(torch.float64).contiguous()
        promoted_cpu = (
            promoted_gradient.detach().to(device="cpu").clone().contiguous()
        )
        if (
            promoted_cpu.dtype != torch.float64
            or not torch.equal(selected_promoted_f64, promoted_cpu)
        ):
            raise RuntimeError(
                "V12 raw float32 selection does not equal V10 promoted support bank"
            )
        node = state.accumulator.add_node(
            node_receipt=core_receipt,
            token_support_h4_gradients_f64=selected_promoted_f64,
        )
        if node.node_index != node_index:
            raise RuntimeError("V12 contraction core node order differs")
        raw_receipt: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "node_index": node_index,
            "path_fraction_hex": float(path_fraction).hex(),
            "quadrature_weight_hex": float(quadrature_weight).hex(),
            "raw_full_gradient_shape": tuple(
                int(size) for size in raw_gradient.shape
            ),
            "raw_full_gradient_dtype": str(raw_gradient.dtype),
            "raw_full_gradient_f32_sha256": (
                v10diag._runtime_tensor_sha256(raw_gradient)
            ),
            "path_vjp_artifact_sha256": path_vjp_artifact,
            "selected_batch_index": 0,
            "support_indices_sha256": v10diag._runtime_tensor_sha256(support_cpu),
            "selected_support_gradient_shape": tuple(
                int(size) for size in raw_selected_f32.shape
            ),
            "selected_support_gradient_f32_sha256": (
                v10diag._runtime_tensor_sha256(raw_selected_f32)
            ),
            "selected_then_promoted_gradient_f64_sha256": (
                v10diag._runtime_tensor_sha256(selected_promoted_f64)
            ),
            "pinned_v10_promoted_gradient_f64_sha256": (
                v10diag._runtime_tensor_sha256(promoted_cpu)
            ),
            "contraction_node_evidence_artifact_sha256": node.artifact_sha256,
            "pinned_v10_core_node_receipt_artifact_sha256": (
                core_receipt.artifact_sha256
            ),
            "raw_float32_selected_before_float64_promotion": True,
            "selected_promoted_equals_V10_bank_bitwise": True,
            "transient_gradient_retained_after_callback": False,
            "raw_tensors_serialized": False,
        }
        raw_receipt["artifact_sha256"] = v10diag.token_v1._domain_sha256(
            raw_receipt, domain=_RAW_GRADIENT_DOMAIN
        )
        state.raw_gradient_receipts[node_index] = raw_receipt
        self._raw_full_gradient_element_count += int(raw_gradient.numel())

        # Raw provenance is authenticated first.  The untouched observation is
        # then handed to the unchanged V11 collector; no tensor is mutated.
        self._suffix(**observation)

    def finalize(self, precision_evidence: Sequence[object]) -> _ContractionCollectionResult:
        suffix = self._suffix.finalize(precision_evidence)
        suffix_by_example = {
            value.example_id: value for value in suffix.evidence
        }
        if (
            len(self._prompts) != _EXPECTED_PROMPTS
            or set(self._prompts) != set(suffix_by_example)
            or _canonical(self._environment) != _canonical(
                _typed_reduction_environment()
            )
        ):
            raise RuntimeError("V12 contraction collection universe differs")
        evidence: list[CandidateJointStateContractionPrecisionEvidence] = []
        raw_receipts: list[Mapping[str, object]] = []
        cast_receipts: list[Mapping[str, object]] = []
        for example in sorted(self._prompts):
            state = self._prompts[example]
            if (
                tuple(sorted(state.raw_gradient_receipts)) != (0, 1, 2, 3)
                or state.cast_only_jvp_receipt is None
                or state.accumulator.node_count != 4
            ):
                raise RuntimeError("V12 contraction prompt collection is incomplete")
            value = state.accumulator.finalize(
                suffix_jvp_evidence=suffix_by_example[example]
            )
            if (
                getattr(state.accumulator, "_integrated_gradient64", None)
                is not None
                or getattr(state.accumulator, "_token_p_v10_f64", None)
                is not None
            ):
                raise RuntimeError(
                    "V12 contraction accumulator retained a transient gradient bank"
                )
            value.validate_integrity()
            evidence.append(value)
            raw_receipts.extend(
                state.raw_gradient_receipts[index] for index in range(4)
            )
            cast_receipts.append(state.cast_only_jvp_receipt)
        comparison = summarize_candidate_joint_state_contraction_precision(evidence)
        ordered_raw = tuple(
            sorted(
                raw_receipts,
                key=lambda value: (
                    str(value["example_id"]), int(value["node_index"])
                ),
            )
        )
        ordered_cast = tuple(
            sorted(cast_receipts, key=lambda value: str(value["example_id"]))
        )
        raw_set = v10diag.token_v1._domain_sha256(
            tuple(str(value["artifact_sha256"]) for value in ordered_raw),
            domain=_RAW_GRADIENT_SET_DOMAIN,
        )
        cast_set = v10diag.token_v1._domain_sha256(
            tuple(str(value["artifact_sha256"]) for value in ordered_cast),
            domain=_CAST_ONLY_JVP_SET_DOMAIN,
        )
        resources = _contraction_resource_accounting(
            evidence=tuple(evidence),
            raw_full_gradient_element_count=self._raw_full_gradient_element_count,
            cast_receipts=ordered_cast,
        )
        return _ContractionCollectionResult(
            evidence=tuple(evidence),
            comparison=comparison,
            suffix=suffix,
            raw_gradient_receipts=ordered_raw,
            raw_gradient_receipt_set_sha256=raw_set,
            cast_only_jvp_receipts=ordered_cast,
            cast_only_jvp_receipt_set_sha256=cast_set,
            resources=resources,
        )


def _contraction_resource_accounting(
    *,
    evidence: Sequence[CandidateJointStateContractionPrecisionEvidence],
    raw_full_gradient_element_count: int,
    cast_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    values = tuple(evidence)
    if len(values) != _EXPECTED_PROMPTS or len(cast_receipts) != _EXPECTED_PROMPTS:
        raise RuntimeError("V12 contraction resource prompt grid differs")
    per_prompt = tuple(value.resource_accounting for value in values)

    def total(key: str) -> int:
        return sum(int(resources[key]) for resources in per_prompt)

    resources = {
        "contraction_prompt_count": len(values),
        "contraction_quadrature_node_count": total("quadrature_node_count"),
        "contraction_supervised_token_count": total("supervised_token_count"),
        "contraction_full_H4_row_count": total("full_h4_row_count"),
        "contraction_support_H4_row_count": total("support_h4_row_count"),
        "contraction_outside_support_H4_row_count": total(
            "outside_support_h4_row_count"
        ),
        "contraction_H4_width": values[0].h4_width,
        "raw_f32_full_gradient_observation_element_count": (
            raw_full_gradient_element_count
        ),
        "raw_f32_support_gradient_selection_element_count": total(
            "gradient_f64_to_f32_roundtrip_validation_element_count"
        ),
        "selected_f32_gradient_f64_promotion_element_count": total(
            "gradient_f64_to_f32_roundtrip_validation_element_count"
        ),
        "gradient_f64_to_f32_roundtrip_validation_element_count": total(
            "gradient_f64_to_f32_roundtrip_validation_element_count"
        ),
        "full_direction_cast_validation_element_count": total(
            "full_direction_cast_validation_element_count"
        ),
        "outside_support_zero_validation_element_count": total(
            "outside_support_zero_validation_element_count"
        ),
        "v10_gradient_weighted_add_element_count": total(
            "v10_gradient_weighted_add_element_count"
        ),
        "v10_final_contraction_product_count": total(
            "v10_final_contraction_product_count"
        ),
        "nodewise_contraction_stage_count": len(CONTRACTION_CORRECTED_STAGE_ORDER),
        "nodewise_contraction_coordinate_observation_count_per_stage": total(
            "nodewise_contraction_coordinate_observation_count_per_stage"
        ),
        "nodewise_contraction_coordinate_observation_count_total": total(
            "nodewise_contraction_coordinate_observation_count_total"
        ),
        "actual_coordinate_product_bank_count_per_prompt": 3,
        "actual_coordinate_product_bank_observation_count_total": total(
            "actual_coordinate_product_bank_count"
        ),
        "actual_coordinate_product_count_total": total(
            "actual_coordinate_product_count_total"
        ),
        "P_prod_and_P_live_share_one_f32_product_bank": True,
        "nodewise_GL4_token_weight_application_count": total(
            "nodewise_GL4_token_weight_application_count"
        ),
        "published_contraction_stage_count": len(CONTRACTION_PUBLISHED_STAGE_ORDER),
        "cast_only_jvp_validation_count": len(cast_receipts),
        "cast_only_jvp_primal_element_count": sum(
            int(value["element_count"]) for value in cast_receipts
        ),
        "cast_only_jvp_tangent_element_count": sum(
            int(value["element_count"]) for value in cast_receipts
        ),
        "additional_model_forward_count": 0,
        "additional_model_backward_count": 0,
        "additional_suffix_jvp_evaluation_count": 0,
        "maximum_transient_integrated_gradient_bank_count_per_prompt": 1,
        "retained_integrated_gradient_bank_count_after_node_three": 0,
    }
    expected = {
        "contraction_prompt_count": _EXPECTED_PROMPTS,
        "contraction_quadrature_node_count": _EXPECTED_NODES,
        "contraction_supervised_token_count": _EXPECTED_SUPERVISED_TOKENS,
        "contraction_full_H4_row_count": _EXPECTED_FULL_H4_ROWS,
        "contraction_support_H4_row_count": _EXPECTED_SUPPORT_H4_ROWS,
        "contraction_outside_support_H4_row_count": (
            _EXPECTED_OUTSIDE_SUPPORT_H4_ROWS
        ),
        "contraction_H4_width": _EXPECTED_H4_WIDTH,
        "raw_f32_full_gradient_observation_element_count": (
            _EXPECTED_RAW_FULL_GRADIENT_ELEMENTS
        ),
        "raw_f32_support_gradient_selection_element_count": (
            _EXPECTED_RAW_SUPPORT_GRADIENT_ELEMENTS
        ),
        "selected_f32_gradient_f64_promotion_element_count": (
            _EXPECTED_RAW_SUPPORT_GRADIENT_ELEMENTS
        ),
        "gradient_f64_to_f32_roundtrip_validation_element_count": (
            _EXPECTED_RAW_SUPPORT_GRADIENT_ELEMENTS
        ),
        "full_direction_cast_validation_element_count": (
            _EXPECTED_CAST_ONLY_JVP_ELEMENTS
        ),
        "outside_support_zero_validation_element_count": (
            _EXPECTED_OUTSIDE_SUPPORT_ZERO_VALIDATION_ELEMENTS
        ),
        "v10_gradient_weighted_add_element_count": (
            _EXPECTED_RAW_SUPPORT_GRADIENT_ELEMENTS
        ),
        "v10_final_contraction_product_count": (
            _EXPECTED_V10_FINAL_CONTRACTION_PRODUCTS
        ),
        "nodewise_contraction_stage_count": len(CONTRACTION_CORRECTED_STAGE_ORDER),
        "nodewise_contraction_coordinate_observation_count_per_stage": (
            _EXPECTED_RAW_SUPPORT_GRADIENT_ELEMENTS
        ),
        "nodewise_contraction_coordinate_observation_count_total": (
            _EXPECTED_NODEWISE_CONTRACTION_COORDINATE_OBSERVATIONS
        ),
        "actual_coordinate_product_bank_count_per_prompt": 3,
        "actual_coordinate_product_bank_observation_count_total": 48,
        "actual_coordinate_product_count_total": (
            _EXPECTED_ACTUAL_COORDINATE_PRODUCTS
        ),
        "P_prod_and_P_live_share_one_f32_product_bank": True,
        "nodewise_GL4_token_weight_application_count": (
            _EXPECTED_NODEWISE_GL4_WEIGHT_APPLICATIONS
        ),
        "published_contraction_stage_count": len(CONTRACTION_PUBLISHED_STAGE_ORDER),
        "cast_only_jvp_validation_count": _EXPECTED_PROMPTS,
        "cast_only_jvp_primal_element_count": _EXPECTED_CAST_ONLY_JVP_ELEMENTS,
        "cast_only_jvp_tangent_element_count": _EXPECTED_CAST_ONLY_JVP_ELEMENTS,
        "additional_model_forward_count": 0,
        "additional_model_backward_count": 0,
        "additional_suffix_jvp_evaluation_count": 0,
        "maximum_transient_integrated_gradient_bank_count_per_prompt": 1,
        "retained_integrated_gradient_bank_count_after_node_three": 0,
    }
    if resources != expected:
        raise RuntimeError(
            "V12 contraction resource accounting differs: "
            f"observed={resources!r}"
        )
    return resources


def _live_v11_sections(
    *,
    fits: Mapping[str, object],
    traces: Sequence[object],
    live: object,
    suffix: object,
    v10_replay: Mapping[str, object],
    v9_binding: Mapping[str, object],
    materialization: Mapping[str, object],
    materialization_report_path: Path | str,
    transfer_report_path: Path | str,
    materialization_binding: Mapping[str, object],
    basis_binding: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    return {
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
        "v10_live_reproduction": v10_replay,
        "v9_live_lineage_reproduction": v9_binding,
        "folds": tuple(fits[family].metadata() for family in sorted(fits)),
        "prompt_receipts": v10diag.v9diag.v5diag._endpoint_prompt_receipts(traces),
        "replayed_v10_endpoint_precision_receipts": live.endpoint_receipts,
        "replayed_v10_path_precision_node_receipts": live.node_receipts,
        "replayed_v10_path_precision_node_receipt_set_sha256": (
            live.node_receipt_set_sha256
        ),
        "suffix_jvp_evidence": tuple(
            value.metadata() for value in suffix.evidence
        ),
        "suffix_runtime_node_receipts": suffix.runtime_node_receipts,
        "suffix_runtime_node_receipt_set_sha256": (
            suffix.runtime_node_receipt_set_sha256
        ),
        "discrete_cast_prompt_receipts": suffix.discrete_cast_receipts,
        "discrete_cast_prompt_receipt_set_sha256": (
            suffix.discrete_cast_receipt_set_sha256
        ),
        "family_equal_suffix_jvp_comparison": suffix.comparison.metadata(),
        "resources": dict(resources),
    }


def _authenticate_live_v11_sections(
    *, report: Mapping[str, object], live_sections: Mapping[str, object]
) -> dict[str, object]:
    expected = {
        "input_binding": report.get("input_binding"),
        "v10_live_reproduction": (
            report.get("v10_control_binding", {}).get("live_reproduction")
            if isinstance(report.get("v10_control_binding"), Mapping)
            else None
        ),
        "v9_live_lineage_reproduction": (
            report.get("v10_control_binding", {}).get(
                "v9_live_lineage_reproduction"
            )
            if isinstance(report.get("v10_control_binding"), Mapping)
            else None
        ),
        **{
            key: report.get(key)
            for key in (
                "folds",
                "prompt_receipts",
                "replayed_v10_endpoint_precision_receipts",
                "replayed_v10_path_precision_node_receipts",
                "replayed_v10_path_precision_node_receipt_set_sha256",
                "suffix_jvp_evidence",
                "suffix_runtime_node_receipts",
                "suffix_runtime_node_receipt_set_sha256",
                "discrete_cast_prompt_receipts",
                "discrete_cast_prompt_receipt_set_sha256",
                "family_equal_suffix_jvp_comparison",
                "resources",
            )
        },
    }
    changed = tuple(
        key
        for key in expected
        if _canonical(expected[key]) != _canonical(live_sections.get(key))
    )
    if changed:
        raise RuntimeError(
            "V12 live execution did not exactly replay V11 sections: "
            + ", ".join(changed)
        )
    return {
        "v11_report_file_sha256": V11_REPORT_FILE_SHA256,
        "v11_report_sha256": V11_REPORT_SHA256,
        "v11_comparison_artifact_sha256": V11_COMPARISON_ARTIFACT_SHA256,
        "section_names": tuple(expected),
        "section_count": len(expected),
        "all_live_sections_canonically_equal": True,
    }


def _resource_accounting(
    *,
    v11_resources: Mapping[str, object],
    contraction_resources: Mapping[str, object],
) -> dict[str, object]:
    if (
        v11_resources.get("total_model_forward_count") != 224
        or v11_resources.get("total_backward_call_count") != 1039
        or v11_resources.get("suffix_jvp_evaluation_count") != 64
        or contraction_resources.get("contraction_quadrature_node_count") != 64
        or contraction_resources.get("cast_only_jvp_validation_count") != 16
        or contraction_resources.get("additional_model_forward_count") != 0
        or contraction_resources.get("additional_model_backward_count") != 0
        or contraction_resources.get("additional_suffix_jvp_evaluation_count") != 0
    ):
        raise RuntimeError("V12 combined resource accounting differs")
    phase_order = tuple(v11_resources.get("phase_order", ())) + (
        "stream_raw_float32_v10_gradient_banks_into_precision_ladder",
        "sixteen_isolated_node_zero_cast_only_forward_mode_JVP_validations",
        "family_equal_contraction_precision_and_fixed_telescope_summary",
    )
    return {
        **dict(v11_resources),
        **dict(contraction_resources),
        "phase_order": phase_order,
        "total_model_forward_count": 224,
        "total_backward_call_count": 1039,
        "suffix_jvp_evaluation_count": 64,
        "V12_additional_model_forward_count_is_zero": True,
        "V12_additional_model_backward_count_is_zero": True,
        "V12_additional_suffix_jvp_evaluation_count_is_zero": True,
        "V12_reuses_existing_V10_gradient_banks": True,
        "raw_gradient_selection_and_contraction_counts_are_not_FLOPs": True,
        "FLOP_or_total_compute_claim": False,
    }


def _require_integrity_gate_results(gates: Mapping[str, bool]) -> None:
    failures = tuple(sorted(key for key, value in gates.items() if not value))
    if failures:
        raise RuntimeError(
            "V12 contraction precision integrity failed before publication: "
            + ", ".join(failures)
        )


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


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic(
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
    v11_report_path: Path | str = DEFAULT_V11_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the no-fit, zero-extra-model-work V12 precision diagnostic."""

    destination = v10diag.token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V12 contraction report")

    # Authenticate every authority, including the complete V11 file, before
    # constructing a live model context.
    v11_report_before = _load_v11_report(v11_report_path)
    v10_report = v11diag._load_v10_report(v10_report_path)
    pinned_v10 = v11diag._index_v10_receipts(v10_report)
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
        label="V12 contraction rank320 materialization",
    )
    transfer = v10diag.token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=v10diag.token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=v10diag.token_v1.TRANSFER_REPORT_SHA256,
        label="V12 contraction rank320 transfer",
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
            raise RuntimeError("V12 contraction A16 panel differs")
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
        collector = _CompositeContractionPrecisionCollector(context=context)
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
        contraction = collector.finalize(live.evidence)
        v10_replay = v11diag._authenticate_live_v10_replay(
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
        raise RuntimeError("V12 callback-enabled V10 resource replay drifted")
    reproduced_v11_resources = v11diag._resource_accounting(
        v10_resources=reproduced_v10_resources,
        suffix_resources=contraction.suffix.resources,
    )
    live_v11_sections = _live_v11_sections(
        fits=fits,
        traces=traces,
        live=live,
        suffix=contraction.suffix,
        v10_replay=v10_replay,
        v9_binding=v9_binding,
        materialization=materialization,
        materialization_report_path=materialization_report_path,
        transfer_report_path=transfer_report_path,
        materialization_binding=materialization_binding,
        basis_binding=basis_binding,
        resources=reproduced_v11_resources,
    )
    v11_replay = _authenticate_live_v11_sections(
        report=v11_report_before, live_sections=live_v11_sections
    )
    # Re-read and re-authenticate after all model work, then require the full
    # pinned report to be byte-logically unchanged from the pre-work load.
    v11_report_after = _load_v11_report(v11_report_path)
    if _canonical(v11_report_before) != _canonical(v11_report_after):
        raise RuntimeError("V11 anchor changed during V12 model work")

    reduction_environment = collector.reduction_environment
    comparison = contraction.comparison
    comparison_metadata = comparison.metadata()
    resources = _resource_accounting(
        v11_resources=reproduced_v11_resources,
        contraction_resources=contraction.resources,
    )
    v12_receipt_authentication = _validate_contraction_receipts(
        raw_receipts=contraction.raw_gradient_receipts,
        raw_receipt_set_sha256=contraction.raw_gradient_receipt_set_sha256,
        cast_receipts=contraction.cast_only_jvp_receipts,
        cast_receipt_set_sha256=contraction.cast_only_jvp_receipt_set_sha256,
        reduction_environment=reduction_environment,
    )
    integrity_gates = {
        "exact_V11_file_and_logical_hash_authenticated_before_and_after_model_work": (
            v11_replay["v11_report_file_sha256"] == V11_REPORT_FILE_SHA256
            and v11_replay["v11_report_sha256"] == V11_REPORT_SHA256
            and _canonical(v11_report_before) == _canonical(v11_report_after)
        ),
        "live_execution_replays_every_published_V11_computational_section": (
            v11_replay["all_live_sections_canonically_equal"] is True
        ),
        "all_V12_receipts_payload_set_order_and_ownership_rehashed": (
            v12_receipt_authentication[
                "all_receipts_payload_set_order_and_ownership_rehashed"
            ]
            is True
            and v12_receipt_authentication["raw_gradient_receipt_count"]
            == _EXPECTED_NODES
            and v12_receipt_authentication["cast_only_jvp_receipt_count"]
            == _EXPECTED_PROMPTS
        ),
        "all_64_raw_float32_gradient_banks_selected_then_promoted_exactly": (
            len(contraction.raw_gradient_receipts) == _EXPECTED_NODES
            and all(
                value["raw_float32_selected_before_float64_promotion"] is True
                and value["selected_promoted_equals_V10_bank_bitwise"] is True
                and value["raw_full_gradient_dtype"] == "torch.float32"
                and isinstance(value["raw_full_gradient_f32_sha256"], str)
                and isinstance(value["path_vjp_artifact_sha256"], str)
                for value in contraction.raw_gradient_receipts
            )
        ),
        "all_16_node_zero_cast_only_JVPs_match_direct_float32_cast_bitwise": (
            len(contraction.cast_only_jvp_receipts) == _EXPECTED_PROMPTS
            and all(
                value["ad_mechanism"] == "torch.func.jvp.forward_mode"
                and value["jvp_strict"] is True
                and value["primal_matches_direct_cast_bitwise"] is True
                and value["primal_matches_live_V10_full_H4_bitwise"] is True
                and value["tangent_matches_direct_cast_bitwise"] is True
                and value["model_or_suffix_segment_evaluated"] is False
                for value in contraction.cast_only_jvp_receipts
            )
        ),
        "P_v10_and_suffix_JVP_finite_controls_replay_V11_exactly": (
            comparison.replayed_v11_comparison_artifact_sha256
            == V11_COMPARISON_ARTIFACT_SHA256
            and comparison.metrics_for("P_v10").adjoint_relative_rmse.hex()
            == float(
                v11_report_before["family_equal_suffix_jvp_comparison"][
                    "adjoint_relative_rmse"
                ]
            ).hex()
            and comparison.finite_metrics.closure_relative_rmse.hex()
            == float(
                v11_report_before["family_equal_suffix_jvp_comparison"][
                    "jvp_closure_relative_rmse"
                ]
            ).hex()
        ),
        "fixed_ladder_telescope_closes_overall_and_in_every_family": (
            comparison.telescope_metrics.passed
            and all(
                family.telescope_metrics.passed
                for family in comparison.family_summaries
            )
        ),
        "typed_CPU_reduction_environment_is_bound_and_unchanged": (
            _canonical(reduction_environment)
            == _canonical(_typed_reduction_environment())
            and reduction_environment["reduction_device_type"] == "cpu"
        ),
        "streaming_integrated_gradient_bank_is_released_after_node_three": (
            resources["retained_integrated_gradient_bank_count_after_node_three"]
            == 0
        ),
        "resource_ledger_distinguishes_logical_stage_observations_from_three_product_banks": (
            resources[
                "nodewise_contraction_coordinate_observation_count_total"
            ]
            == _EXPECTED_NODEWISE_CONTRACTION_COORDINATE_OBSERVATIONS
            and resources["actual_coordinate_product_bank_count_per_prompt"] == 3
            and resources[
                "actual_coordinate_product_bank_observation_count_total"
            ]
            == 48
            and resources["actual_coordinate_product_count_total"]
            == _EXPECTED_ACTUAL_COORDINATE_PRODUCTS
            and resources["P_prod_and_P_live_share_one_f32_product_bank"] is True
        ),
        "zero_additional_model_backward_or_suffix_evaluation_beyond_V11": (
            resources["additional_model_forward_count"] == 0
            and resources["additional_model_backward_count"] == 0
            and resources["additional_suffix_jvp_evaluation_count"] == 0
            and resources["total_model_forward_count"] == 224
            and resources["total_backward_call_count"] == 1039
            and resources["suffix_jvp_evaluation_count"] == 64
        ),
        "no_fit_search_selection_correction_routing_or_serving_mutation": True,
    }
    _require_integrity_gate_results(integrity_gates)
    classification = comparison.classification
    passed = comparison.finite_correction_eligible
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_contraction_precision",
            "outer_held_prompt_count": _EXPECTED_PROMPTS,
            "outer_family_count": _EXPECTED_FAMILIES,
            "prompts_per_outer_family": 2,
            "V11_control": "exact_pinned_suffix_JVP_v11_report_and_live_replay",
            "path": "exact_V10_realized_scalar_to_joint_H4_GL4_path",
            "gradient_source": "raw_float32_V10_H4_VJP_bank_before_promotion",
            "published_stage_order": CONTRACTION_PUBLISHED_STAGE_ORDER,
            "corrected_stage_order": CONTRACTION_CORRECTED_STAGE_ORDER,
            "aggregation": "equal_token_then_equal_prompt_then_equal_family",
            "adjoint_threshold": {
                "overall_and_every_family_relative_RMSE_maximum": (
                    ADJOINT_RELATIVE_RMSE_MAXIMUM
                )
            },
            "classification_rule": "earliest_corrected_stage_passing_frozen_adjoint_gate",
            "isolated_cast_only_JVP_validation_count": _EXPECTED_PROMPTS,
            "P_prod_and_P_live_share_one_f32_product_bank": True,
            "v12_fit_performed": False,
            "search_or_selection_performed": False,
            "finite_correction_or_refit_performed": False,
            "fallback_or_routing_allowed": False,
            "result_can_authorize_serving_or_model_mutation": False,
        },
        "v11_control_binding": {
            "file": str(v11_report_path),
            "file_sha256": V11_REPORT_FILE_SHA256,
            "report_sha256": V11_REPORT_SHA256,
            "schema": v11_report_before.get("schema"),
            "classification": v11_report_before.get("classification"),
            "passed": v11_report_before.get("passed"),
            "live_reproduction": v11_replay,
            "authenticated_before_and_after_model_work": True,
        },
        "typed_reduction_environment": reduction_environment,
        "v12_receipt_authentication": v12_receipt_authentication,
        "input_binding": live_v11_sections["input_binding"],
        "folds": live_v11_sections["folds"],
        "prompt_receipts": live_v11_sections["prompt_receipts"],
        "replayed_v11_suffix_jvp_evidence": live_v11_sections[
            "suffix_jvp_evidence"
        ],
        "replayed_v11_suffix_runtime_node_receipts": live_v11_sections[
            "suffix_runtime_node_receipts"
        ],
        "replayed_v11_suffix_runtime_node_receipt_set_sha256": (
            live_v11_sections["suffix_runtime_node_receipt_set_sha256"]
        ),
        "replayed_v11_discrete_cast_prompt_receipts": live_v11_sections[
            "discrete_cast_prompt_receipts"
        ],
        "replayed_v11_discrete_cast_prompt_receipt_set_sha256": (
            live_v11_sections["discrete_cast_prompt_receipt_set_sha256"]
        ),
        "replayed_v11_family_equal_suffix_jvp_comparison": live_v11_sections[
            "family_equal_suffix_jvp_comparison"
        ],
        "raw_float32_gradient_selection_receipts": (
            contraction.raw_gradient_receipts
        ),
        "raw_float32_gradient_selection_receipt_set_sha256": (
            contraction.raw_gradient_receipt_set_sha256
        ),
        "cast_only_jvp_validation_receipts": contraction.cast_only_jvp_receipts,
        "cast_only_jvp_validation_receipt_set_sha256": (
            contraction.cast_only_jvp_receipt_set_sha256
        ),
        "contraction_precision_evidence": tuple(
            value.metadata() for value in contraction.evidence
        ),
        "family_equal_contraction_precision_comparison": comparison_metadata,
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "outcome_matrix": {
            "outcome": classification,
            "integrity_completed": True,
            "earliest_passing_stage": comparison.earliest_passing_stage,
            "stage_gate_results": tuple(
                sorted(comparison.stage_gate_results.items())
            ),
            "fixed_telescope_passed": comparison.telescope_metrics.passed,
            "finite_correction_eligible": comparison.finite_correction_eligible,
            "finite_correction_executed": False,
            "selection_or_serving_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_diagnostic_only": True,
            "V11_file_and_live_computational_sections_replayed_exactly": True,
            "cast_only_JVP_is_validation_not_model_or_suffix_work": True,
            "P_live_is_fixed_counterfactual_not_internal_VJP_schedule_proof": True,
            "precision_stage_pass_does_not_establish_finite_closure": True,
            "finite_correction_rung_opened_only_if_all_eligibility_gates_pass": (
                comparison.finite_correction_eligible
            ),
            "candidate_serving_authorized": False,
            "model_mutation_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "FLOP_or_total_compute_claim": False,
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
        description="Run the pinned V12 post-H4 contraction-precision diagnostic."
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
