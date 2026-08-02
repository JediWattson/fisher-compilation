"""Locked V13-B same-suffix native-VJP diagnostic over exact V12.

V13-B replays the complete authenticated V12/V11 computation unchanged and,
at each of the same sixty-four GL4 nodes, evaluates a native reverse-mode VJP
of the post-H4 suffix.  The native vectors are compared with the already
published V11 forward-mode JVP vectors both nodewise and after GL4 integration.

This is a same-A, truth-leaking diagnostic.  It performs no fit, correction,
search, route, fallback, serving mutation, or compression selection.  Raw H4,
logit, KL, JVP, and VJP tensors never enter the JSON artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor

from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic as v12diag
from . import gemma3_l3_l4_h4_suffix_vjp_runtime as native_runtime
from .complete_h4_tail_candidate_joint_state_suffix_native_vjp import (
    ADJOINT_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateSuffixNativeVJPComparison,
    CandidateJointStateSuffixNativeVJPEvidence,
    build_candidate_joint_state_suffix_native_vjp_evidence,
    summarize_candidate_joint_state_suffix_native_vjp,
)
from .gemma3_l3_l4_h4_suffix_vjp_runtime import (
    Gemma3L3L4H4SuffixVJP,
    Gemma3L3L4H4SuffixVJPReceipt,
    Gemma3L3L4H4SuffixVJPRuntime,
    gemma3_l3_l4_h4_suffix_vjp_resource_accounting,
    require_gemma3_l3_l4_h4_suffix_vjp_complete_panel,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_V12_REPORT",
    "V12_REPORT_FILE_SHA256",
    "V12_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_native_vjp_diagnostic",
    "main",
]


v11diag = v12diag.v11diag
v10diag = v12diag.v10diag
DEFAULT_MATERIALIZATION_REPORT = v12diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v12diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v12diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v12diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v12diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v12diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v12diag.DEFAULT_V6_REPORT
DEFAULT_V7_REPORT = v12diag.DEFAULT_V7_REPORT
DEFAULT_V8_REPORT = v12diag.DEFAULT_V8_REPORT
DEFAULT_V9_REPORT = v12diag.DEFAULT_V9_REPORT
DEFAULT_V10_REPORT = v12diag.DEFAULT_V10_REPORT
DEFAULT_V11_REPORT = v12diag.DEFAULT_V11_REPORT
DEFAULT_V12_REPORT = v12diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = v10diag.token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-suffix-native-vjp-gl4-"
    "lofo-a-fit16-dev-v13.json"
)

V12_REPORT_FILE_SHA256 = (
    "3d8cf470b7fb1a10cbaaa2758e59d960b62e850e67bd41b54842c95b1f4c22d7"
)
V12_REPORT_SHA256 = (
    "795e509774435b188dbc6f50590543ce22eda6ef63fd1ff899092a02cc32358f"
)
V12_COMPARISON_ARTIFACT_SHA256 = (
    "2948019f59f5557a74b81a95bb2bb9294144d8d0b4a72eecf1bbbf7496e9c0d0"
)
V12_RAW_GRADIENT_SET_SHA256 = (
    "08b794227948fc7ff1d7d9102668798476cdd129058fc4b21a3b026126914481"
)
V12_CAST_ONLY_JVP_SET_SHA256 = (
    "0a83a6c6bf530f6f62f651d18c46651ce9f0928525576a29fd55bfdf8b8b9be1"
)
V12_REDUCTION_ENVIRONMENT_SHA256 = (
    "203d48e3d1628516c4cfdc7a87335c5b814943920fc680b475487e914057e720"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_suffix_native_vjp_gl4_lofo.v13"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-suffix-native-vjp:v13\0"
_NATIVE_RUNTIME_NODE_DOMAIN = (
    b"fisher-graph:complete-h4-k64-suffix-native-vjp-runtime-node:v13\0"
)
_NATIVE_RUNTIME_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-suffix-native-vjp-runtime-set:v13\0"
)
_NATIVE_RESOURCE_NODE_DOMAIN = (
    b"fisher-graph:complete-h4-k64-suffix-native-vjp-resource-node:v13\0"
)
_NATIVE_RESOURCE_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-suffix-native-vjp-resource-set:v13\0"
)
_NATIVE_RESOURCE_COMPONENT_DOMAIN = (
    b"fisher-graph:complete-h4-k64-suffix-native-vjp-resource-component:v13\0"
)

_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS = 16
_EXPECTED_NODES = 64
_EXPECTED_SUPERVISED_TOKENS = 803
_EXPECTED_NATIVE_TOKEN_NODE_COVERAGE = 3_212
_EXPECTED_NATIVE_PULLBACK_CHUNKS = 436
_EXPECTED_NATIVE_SUFFIX_FORWARDS = 64
_EXPECTED_FULL_H4_ROWS = 947
_EXPECTED_H4_WIDTH = 640
_EXPECTED_NATIVE_INPUT_GRADIENT_COORDINATES = 130_048_000
_EXPECTED_NATIVE_SUPPORT_CONTRACTION_PRODUCTS = 113_602_560
_EXPECTED_NATIVE_OUTSIDE_SUPPORT_GRADIENT_COORDINATES = 16_445_440
_EXPECTED_DENSE_OUTPUT_COTANGENT_COORDINATES = 174_292
_EXPECTED_NATIVE_SUFFIX_SEGMENT_CALLS = 832
_EXPECTED_DIRECTION_COORDINATE_VALIDATIONS = 2_424_320
_EXPECTED_OUTSIDE_DIRECTION_ZERO_VALIDATIONS = 327_680


@dataclass(frozen=True, slots=True)
class _NativeVJPCollectionResult:
    contraction: object
    evidence: tuple[CandidateJointStateSuffixNativeVJPEvidence, ...]
    comparison: CandidateJointStateSuffixNativeVJPComparison
    runtime_receipts: tuple[Mapping[str, object], ...]
    runtime_receipt_set_sha256: str
    resource_receipts: tuple[Mapping[str, object], ...]
    resource_receipt_set_sha256: str
    runtime_resources: Mapping[str, object]
    core_resources: Mapping[str, object]


@dataclass(slots=True)
class _ObservedNativeVJPPrompt:
    example_id: str
    family_id: str
    scalar_h4: Tensor
    joint_h4: Tensor
    direction_h4_f64: Tensor
    runtime: Gemma3L3L4H4SuffixVJPRuntime
    vectors: dict[int, Tensor] = field(default_factory=dict)
    typed_receipts: dict[int, Gemma3L3L4H4SuffixVJPReceipt] = field(
        default_factory=dict
    )
    runtime_receipts: dict[int, Mapping[str, object]] = field(
        default_factory=dict
    )
    resource_receipts: dict[int, Mapping[str, object]] = field(
        default_factory=dict
    )


def _canonical(value: object) -> object:
    return v12diag._canonical(value)


def _load_v12_report(path: Path | str) -> dict[str, object]:
    """Load and rehash only the exact independently authenticated V12 file."""

    report = v10diag.token_v1._load_pinned_report(
        path,
        expected_file_sha256=V12_REPORT_FILE_SHA256,
        expected_report_sha256=V12_REPORT_SHA256,
        label="candidate joint-state suffix native VJP V12 anchor",
    )
    logical_payload = dict(report)
    logical_payload.pop("report_sha256", None)
    comparison = report.get("family_equal_contraction_precision_comparison")
    environment = report.get("typed_reduction_environment")
    receipt_authentication = report.get("v12_receipt_authentication")
    integrity = report.get("integrity_gate_results")
    if (
        v10diag.token_v1._domain_sha256(
            logical_payload, domain=v12diag._REPORT_DOMAIN
        )
        != V12_REPORT_SHA256
        or report.get("schema") != v12diag._SCHEMA
        or report.get("classification")
        != "unresolved_forward_reverse_ad_kernel_mismatch"
        or report.get("passed") is not False
        or not isinstance(comparison, Mapping)
        or comparison.get("artifact_sha256")
        != V12_COMPARISON_ARTIFACT_SHA256
        or comparison.get("earliest_passing_stage") is not None
        or comparison.get("finite_correction_eligible") is not False
        or report.get("raw_float32_gradient_selection_receipt_set_sha256")
        != V12_RAW_GRADIENT_SET_SHA256
        or report.get("cast_only_jvp_validation_receipt_set_sha256")
        != V12_CAST_ONLY_JVP_SET_SHA256
        or not isinstance(environment, Mapping)
        or environment.get("artifact_sha256")
        != V12_REDUCTION_ENVIRONMENT_SHA256
        or not isinstance(receipt_authentication, Mapping)
        or receipt_authentication.get("raw_gradient_receipt_set_sha256")
        != V12_RAW_GRADIENT_SET_SHA256
        or receipt_authentication.get("cast_only_jvp_receipt_set_sha256")
        != V12_CAST_ONLY_JVP_SET_SHA256
        or receipt_authentication.get("typed_reduction_environment_sha256")
        != V12_REDUCTION_ENVIRONMENT_SHA256
        or not isinstance(integrity, list)
        or len(integrity) != 12
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or value[1] is not True
            for value in integrity
        )
    ):
        raise RuntimeError("V12 suffix native-VJP anchor differs")
    authentication = v12diag._validate_contraction_receipts(
        raw_receipts=report.get("raw_float32_gradient_selection_receipts", ()),
        raw_receipt_set_sha256=str(
            report.get("raw_float32_gradient_selection_receipt_set_sha256", "")
        ),
        cast_receipts=report.get("cast_only_jvp_validation_receipts", ()),
        cast_receipt_set_sha256=str(
            report.get("cast_only_jvp_validation_receipt_set_sha256", "")
        ),
        reduction_environment=environment,
    )
    if _canonical(authentication) != _canonical(receipt_authentication):
        raise RuntimeError("V12 receipt authentication differs")
    return report


def _live_v12_sections(
    *,
    live_v11_sections: Mapping[str, object],
    v11_replay: Mapping[str, object],
    contraction: object,
    reduction_environment: Mapping[str, object],
    resources: Mapping[str, object],
) -> dict[str, object]:
    authentication = v12diag._validate_contraction_receipts(
        raw_receipts=contraction.raw_gradient_receipts,
        raw_receipt_set_sha256=contraction.raw_gradient_receipt_set_sha256,
        cast_receipts=contraction.cast_only_jvp_receipts,
        cast_receipt_set_sha256=contraction.cast_only_jvp_receipt_set_sha256,
        reduction_environment=reduction_environment,
    )
    return {
        "v11_live_reproduction": dict(v11_replay),
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
        "family_equal_contraction_precision_comparison": (
            contraction.comparison.metadata()
        ),
        "typed_reduction_environment": dict(reduction_environment),
        "v12_receipt_authentication": authentication,
        "resources": dict(resources),
    }


def _authenticate_live_v12_sections(
    *, report: Mapping[str, object], live_sections: Mapping[str, object]
) -> dict[str, object]:
    control = report.get("v11_control_binding")
    expected: dict[str, object] = {
        "v11_live_reproduction": (
            control.get("live_reproduction")
            if isinstance(control, Mapping)
            else None
        ),
        **{
            key: report.get(key)
            for key in (
                "input_binding",
                "folds",
                "prompt_receipts",
                "replayed_v11_suffix_jvp_evidence",
                "replayed_v11_suffix_runtime_node_receipts",
                "replayed_v11_suffix_runtime_node_receipt_set_sha256",
                "replayed_v11_discrete_cast_prompt_receipts",
                "replayed_v11_discrete_cast_prompt_receipt_set_sha256",
                "replayed_v11_family_equal_suffix_jvp_comparison",
                "raw_float32_gradient_selection_receipts",
                "raw_float32_gradient_selection_receipt_set_sha256",
                "cast_only_jvp_validation_receipts",
                "cast_only_jvp_validation_receipt_set_sha256",
                "contraction_precision_evidence",
                "family_equal_contraction_precision_comparison",
                "typed_reduction_environment",
                "v12_receipt_authentication",
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
            "V13-B live execution did not exactly replay V12 sections: "
            + ", ".join(changed)
        )
    return {
        "v12_report_file_sha256": V12_REPORT_FILE_SHA256,
        "v12_report_sha256": V12_REPORT_SHA256,
        "v12_comparison_artifact_sha256": V12_COMPARISON_ARTIFACT_SHA256,
        "v12_raw_gradient_receipt_set_sha256": V12_RAW_GRADIENT_SET_SHA256,
        "v12_cast_only_jvp_receipt_set_sha256": V12_CAST_ONLY_JVP_SET_SHA256,
        "v12_reduction_environment_sha256": V12_REDUCTION_ENVIRONMENT_SHA256,
        "section_names": tuple(expected),
        "section_count": len(expected),
        "all_live_sections_canonically_equal": True,
    }


class _CompositeSuffixNativeVJPCollector:
    """Run untouched V12/V11 first, then one native VJP at the same node."""

    def __init__(self, *, context: object) -> None:
        self._context = context
        self._v12 = v12diag._CompositeContractionPrecisionCollector(
            context=context
        )
        self._prompts: dict[str, _ObservedNativeVJPPrompt] = {}
        self._live_result_count = 0

    @property
    def reduction_environment(self) -> Mapping[str, object]:
        return self._v12.reduction_environment

    def __call__(self, **observation: object) -> None:
        # This exact call is intentionally first.  It includes the unchanged
        # V12 raw-gradient ladder and unchanged V11 suffix-JVP collector.
        self._v12(**observation)
        self._observe_native(**observation)

    def _observe_native(self, **observation: object) -> None:
        context = observation.get("context")
        trace = observation.get("trace")
        model_inputs = observation.get("model_inputs")
        teacher_logits = observation.get("teacher_logits")
        endpoint_grid = observation.get("endpoint_grid")
        scalar_execution = observation.get("scalar_execution")
        joint_execution = observation.get("joint_execution")
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
            or type(node_index) is not int
            or not 0 <= node_index < 4
            or isinstance(path_fraction, bool)
            or not isinstance(path_fraction, (int, float))
            or isinstance(quadrature_weight, bool)
            or not isinstance(quadrature_weight, (int, float))
            or path_vjp is None
            or not isinstance(path_rows, Tensor)
            or not isinstance(path_kl, Tensor)
            or core_receipt is None
        ):
            raise RuntimeError("V13-B native observer contract differs")
        example = v10diag.token_v1._identifier(
            getattr(trace, "example_id", None), label="V13-B native example"
        )
        family = v10diag.token_v1._identifier(
            getattr(trace, "family_id", None), label="V13-B native family"
        )
        support = getattr(trace, "support_indices", None)
        scalar_h4 = getattr(scalar_execution, "candidate_h4", None)
        joint_h4 = getattr(joint_execution, "candidate_h4", None)
        path_execution = getattr(path_vjp, "execution", None)
        full_h4 = getattr(path_execution, "candidate_h4", None)
        full_logits = getattr(path_execution, "logits", None)
        if not all(
            isinstance(value, Tensor)
            for value in (support, scalar_h4, joint_h4, full_h4, full_logits)
        ):
            raise RuntimeError("V13-B native observer tensor contract differs")
        support = support.detach().to(device="cpu").clone().contiguous()
        scalar_h4 = scalar_h4.detach().contiguous()
        joint_h4 = joint_h4.detach().contiguous()
        full_h4 = full_h4.detach().contiguous()
        full_logits = full_logits.detach().contiguous()
        path_kl = path_kl.detach().to(device="cpu").contiguous()

        state = self._prompts.get(example)
        if state is None:
            if node_index != 0:
                raise RuntimeError("V13-B native prompt did not begin at node zero")
            adapter = getattr(context, "adapter", None)
            sequence = adapter.prepare_sequence(model_inputs)
            _point_zero, direction = v11diag._full_path_point_f64(
                scalar_h4, joint_h4, 0.0
            )
            runtime = Gemma3L3L4H4SuffixVJPRuntime(
                adapter,
                sequence,
                teacher_logits=teacher_logits.detach().contiguous(),
                supervised_indices=endpoint_grid.detach()
                .to(device="cpu")
                .clone()
                .contiguous(),
            )
            state = _ObservedNativeVJPPrompt(
                example_id=example,
                family_id=family,
                scalar_h4=scalar_h4.clone().contiguous(),
                joint_h4=joint_h4.clone().contiguous(),
                direction_h4_f64=direction,
                runtime=runtime,
            )
            self._prompts[example] = state
        elif (
            state.family_id != family
            or node_index != len(state.vectors)
            or not v11diag._bitwise_equal_across_devices(
                state.scalar_h4, scalar_h4
            )
            or not v11diag._bitwise_equal_across_devices(
                state.joint_h4, joint_h4
            )
        ):
            raise RuntimeError("V13-B native prompt state drifted")

        point, direction = v11diag._full_path_point_f64(
            state.scalar_h4, state.joint_h4, float(path_fraction)
        )
        if not v11diag._bitwise_equal_across_devices(
            direction, state.direction_h4_f64
        ):
            raise RuntimeError("V13-B native direction drifted between nodes")
        result = state.runtime.execute(
            point,
            direction,
            support_indices=support,
            full_h4=full_h4,
            full_logits=full_logits,
        )
        if not isinstance(result, Gemma3L3L4H4SuffixVJP):
            raise RuntimeError("V13-B native runtime returned wrong result type")
        result.validate_integrity()
        receipt = result.receipt
        replay_rows = (
            full_h4.detach()
            .to(device="cpu")[0]
            .index_select(0, support)
            .contiguous()
        )
        if (
            not v11diag._bitwise_equal_across_devices(replay_rows, path_rows)
            or not v11diag._bitwise_equal_across_devices(
                result.primal_token_teacher_kl, path_kl
            )
            or receipt.path_h4_sha256
            != v10diag._runtime_tensor_sha256(point)
            or receipt.direction_h4_sha256
            != v10diag._runtime_tensor_sha256(direction)
            or receipt.full_h4_sha256
            != v10diag._runtime_tensor_sha256(full_h4)
            or receipt.full_logits_sha256
            != v10diag._runtime_tensor_sha256(full_logits)
            or receipt.primal_token_teacher_kl_sha256
            != v10diag._runtime_tensor_sha256(path_kl)
            or receipt.support_indices_sha256
            != v10diag._runtime_tensor_sha256(support)
        ):
            raise RuntimeError("V13-B native runtime did not replay V10 primal")

        v12_state = self._v12._prompts.get(example)
        v11_state = self._v12._suffix._prompts.get(example)
        if v12_state is None or v11_state is None:
            raise RuntimeError("V13-B native observation lacks V12/V11 ownership")
        raw_receipt = v12_state.raw_gradient_receipts.get(node_index)
        observed_jvp_node = v11_state.nodes.get(node_index)
        if not isinstance(raw_receipt, Mapping) or observed_jvp_node is None:
            raise RuntimeError("V13-B native same-node authorities are incomplete")
        core_artifact = v10diag._require_sha256(
            getattr(core_receipt, "artifact_sha256", None),
            label="V13-B native V10 core node",
        )
        contraction_node_artifact = v10diag._require_sha256(
            raw_receipt.get("contraction_node_evidence_artifact_sha256"),
            label="V13-B native V12 contraction node",
        )
        if (
            raw_receipt.get("pinned_v10_core_node_receipt_artifact_sha256")
            != core_artifact
            or not v11diag._bitwise_equal_across_devices(
                direction, state.direction_h4_f64
            )
        ):
            raise RuntimeError("V13-B native authority pin differs")

        self._live_result_count += 1
        runtime_metadata = result.metadata()
        runtime_receipt: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "node_index": node_index,
            "path_fraction_hex": float(path_fraction).hex(),
            "quadrature_weight_hex": float(quadrature_weight).hex(),
            "live_result_ordinal": self._live_result_count,
            "pinned_v10_core_node_receipt_artifact_sha256": core_artifact,
            "pinned_v12_contraction_node_artifact_sha256": (
                contraction_node_artifact
            ),
            "native_suffix_vjp_runtime_artifact_sha256": result.artifact_sha256,
            "provider_artifact_sha256": core_receipt.provider_artifact_sha256,
            "execution_artifact_sha256": core_receipt.execution_artifact_sha256,
            "path_h4_sha256": receipt.path_h4_sha256,
            "supervised_grid_sha256": receipt.supervised_indices_sha256,
            "primal_token_teacher_kl_sha256": (
                receipt.primal_token_teacher_kl_sha256
            ),
            "native_suffix_vjp_runtime_receipt": runtime_metadata,
            "native_vjp_primal_count": 1,
            "runtime_path_h4_matches_constructed_f64_path_bitwise": True,
            "runtime_cast_h4_matches_V10_full_node_bitwise": True,
            "runtime_full_logits_matches_V10_full_node_bitwise": True,
            "runtime_primal_token_KL_matches_V10_f64_bitwise": True,
            "same_node_direction_matches_V11_bitwise": True,
            "no_discarded_preflight_result": True,
            "raw_tensors_serialized": False,
        }
        runtime_receipt["receipt_sha256"] = (
            v10diag.token_v1._domain_sha256(
                runtime_receipt, domain=_NATIVE_RUNTIME_NODE_DOMAIN
            )
        )

        chunks = receipt.chunk_receipts

        def component_hash(name: str) -> str:
            return v10diag.token_v1._domain_sha256(
                tuple(getattr(chunk, name) for chunk in chunks),
                domain=_NATIVE_RESOURCE_COMPONENT_DOMAIN,
            )

        resource_receipt: dict[str, object] = {
            "example_id": example,
            "family_id": family,
            "node_index": node_index,
            "native_runtime_receipt_sha256": runtime_receipt["receipt_sha256"],
            "token_count": receipt.token_count,
            "full_h4_row_count": receipt.h4_shape[1],
            "support_h4_row_count": receipt.support_row_count,
            "outside_support_h4_row_count": receipt.outside_support_row_count,
            "h4_width": receipt.h4_shape[2],
            "native_suffix_forward_count": 1,
            "native_vjp_primal_count": 1,
            "native_vjp_pullback_count": receipt.vjp_pullback_chunk_call_count,
            "suffix_segment_call_count": receipt.suffix_segment_call_count,
            "logit_projection_call_count": receipt.logit_projection_call_count,
            "h4_dtype_cast_count": receipt.h4_dtype_cast_count,
            "canonical_output_cotangent_row_count": (
                receipt.token_cotangent_coverage_count
            ),
            "dense_output_cotangent_coordinate_count": (
                receipt.token_cotangent_element_count
            ),
            "full_input_gradient_coordinate_count": (
                receipt.full_h4_cotangent_coordinate_count
            ),
            "support_contraction_product_count": (
                receipt.direction_contraction_coordinate_product_count
            ),
            "outside_support_gradient_coordinate_count": (
                receipt.outside_support_h4_cotangent_coordinate_count
            ),
            "direction_coordinate_validation_count": (
                receipt.direction_coordinate_validation_count
            ),
            "outside_support_direction_zero_validation_count": (
                receipt.outside_support_direction_zero_validation_count
            ),
            "outside_direction_nonzero_count": (
                receipt.outside_direction_nonzero_count
            ),
            "maximum_outside_direction_abs": receipt.outside_direction_max_abs,
            "outside_direction_zero_proved_before_support_contraction": True,
            "support_indices_sha256": receipt.support_indices_sha256,
            "token_basis_cotangent_sha256": component_hash(
                "token_cotangent_sha256"
            ),
            "transient_full_cotangent_sha256": component_hash(
                "full_h4_cotangent_sha256"
            ),
            "transient_full_input_gradient_sha256": component_hash(
                "full_h4_cotangent_sha256"
            ),
            "support_input_gradient_sha256": component_hash(
                "support_h4_cotangent_sha256"
            ),
            "contracted_directional_vector_sha256": component_hash(
                "contracted_directional_token_teacher_kl_sha256"
            ),
            "chunk_artifact_sha256s": tuple(
                chunk.artifact_sha256 for chunk in chunks
            ),
            "transient_full_cotangent_hashed": (
                receipt.full_h4_cotangents_hashed
            ),
            "transient_full_input_gradient_hashed": (
                receipt.full_h4_cotangents_hashed
            ),
            "transient_tensors_retained_after_result": False,
            "raw_tensors_serialized": False,
        }
        resource_receipt["artifact_sha256"] = (
            v10diag.token_v1._domain_sha256(
                resource_receipt, domain=_NATIVE_RESOURCE_NODE_DOMAIN
            )
        )
        state.vectors[node_index] = (
            result.directional_token_teacher_kl.detach()
            .to(device="cpu")
            .clone()
            .contiguous()
        )
        state.typed_receipts[node_index] = receipt
        state.runtime_receipts[node_index] = runtime_receipt
        state.resource_receipts[node_index] = resource_receipt

    def finalize(self, precision_evidence: Sequence[object]) -> _NativeVJPCollectionResult:
        contraction = self._v12.finalize(precision_evidence)
        contraction_by_example = {
            value.example_id: value for value in contraction.evidence
        }
        if (
            self._live_result_count != _EXPECTED_NODES
            or len(self._prompts) != _EXPECTED_PROMPTS
            or set(self._prompts) != set(contraction_by_example)
        ):
            raise RuntimeError("V13-B native collection universe differs")
        evidence: list[CandidateJointStateSuffixNativeVJPEvidence] = []
        runtime_receipts: list[Mapping[str, object]] = []
        resource_receipts: list[Mapping[str, object]] = []
        typed_receipts: list[Gemma3L3L4H4SuffixVJPReceipt] = []
        for example in sorted(self._prompts):
            state = self._prompts[example]
            if any(
                tuple(sorted(values)) != (0, 1, 2, 3)
                for values in (
                    state.vectors,
                    state.typed_receipts,
                    state.runtime_receipts,
                    state.resource_receipts,
                )
            ):
                raise RuntimeError("V13-B native prompt collection is incomplete")
            value = build_candidate_joint_state_suffix_native_vjp_evidence(
                contraction_precision_evidence=contraction_by_example[example],
                token_native_vjp_node_vectors_f64=tuple(
                    state.vectors[index] for index in range(4)
                ),
                native_suffix_runtime_receipt_sha256s=tuple(
                    str(state.runtime_receipts[index]["receipt_sha256"])
                    for index in range(4)
                ),
                native_resource_receipt_sha256s=tuple(
                    str(state.resource_receipts[index]["artifact_sha256"])
                    for index in range(4)
                ),
                native_suffix_forward_counts=(1, 1, 1, 1),
                native_vjp_pullback_counts=tuple(
                    state.typed_receipts[index].vjp_pullback_chunk_call_count
                    for index in range(4)
                ),
            )
            value.validate_integrity()
            evidence.append(value)
            runtime_receipts.extend(
                state.runtime_receipts[index] for index in range(4)
            )
            resource_receipts.extend(
                state.resource_receipts[index] for index in range(4)
            )
            typed_receipts.extend(
                state.typed_receipts[index] for index in range(4)
            )
        comparison = summarize_candidate_joint_state_suffix_native_vjp(evidence)
        ordered_runtime = tuple(runtime_receipts)
        ordered_resources = tuple(resource_receipts)
        runtime_set = v10diag.token_v1._domain_sha256(
            tuple(str(value["receipt_sha256"]) for value in ordered_runtime),
            domain=_NATIVE_RUNTIME_SET_DOMAIN,
        )
        resource_set = v10diag.token_v1._domain_sha256(
            tuple(str(value["artifact_sha256"]) for value in ordered_resources),
            domain=_NATIVE_RESOURCE_SET_DOMAIN,
        )
        runtime_resources = gemma3_l3_l4_h4_suffix_vjp_resource_accounting(
            typed_receipts
        )
        require_gemma3_l3_l4_h4_suffix_vjp_complete_panel(runtime_resources)
        core_resources = _native_core_resource_accounting(evidence)
        _validate_native_vjp_receipts(
            runtime_receipts=ordered_runtime,
            runtime_receipt_set_sha256=runtime_set,
            resource_receipts=ordered_resources,
            resource_receipt_set_sha256=resource_set,
        )
        return _NativeVJPCollectionResult(
            contraction=contraction,
            evidence=tuple(evidence),
            comparison=comparison,
            runtime_receipts=ordered_runtime,
            runtime_receipt_set_sha256=runtime_set,
            resource_receipts=ordered_resources,
            resource_receipt_set_sha256=resource_set,
            runtime_resources=runtime_resources,
            core_resources=core_resources,
        )


def _native_core_resource_accounting(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
) -> dict[str, object]:
    """Aggregate the pure-core ledger and enforce the locked A16 panel."""

    values = tuple(evidence)
    if len(values) != _EXPECTED_PROMPTS:
        raise RuntimeError("V13-B native core resource prompt grid differs")
    ledgers = tuple(value.resource_accounting for value in values)

    def total(key: str) -> int:
        return sum(int(value[key]) for value in ledgers)

    widths = {int(value["h4_width"]) for value in ledgers}
    resources: dict[str, object] = {
        "native_prompt_count": len(values),
        "native_family_count": len({value.family_id for value in values}),
        "native_quadrature_node_count": total("quadrature_node_count"),
        "native_supervised_token_count": total("supervised_token_count"),
        "native_full_H4_row_count": total("full_h4_row_count"),
        "native_H4_width": next(iter(widths)) if len(widths) == 1 else -1,
        "native_suffix_forward_count": total("native_suffix_forward_count"),
        "native_vjp_pullback_count": total("native_vjp_pullback_count"),
        "logical_native_vjp_input_gradient_coordinate_count": total(
            "logical_native_vjp_input_gradient_coordinate_count"
        ),
        "canonical_output_cotangent_row_count": total(
            "canonical_output_cotangent_row_count"
        ),
        "native_token_directional_derivative_count": total(
            "native_token_directional_derivative_count"
        ),
        "published_v11_jvp_node_token_reference_element_count": total(
            "published_v11_jvp_node_token_reference_element_count"
        ),
        "native_GL4_token_weight_application_count": total(
            "native_GL4_token_weight_application_count"
        ),
        "fresh_full_model_forward_count": total(
            "fresh_full_model_forward_count"
        ),
        "fresh_full_model_backward_count": total(
            "fresh_full_model_backward_count"
        ),
        "resource_counts_are_not_FLOPs_or_total_model_compute": True,
    }
    expected = {
        "native_prompt_count": _EXPECTED_PROMPTS,
        "native_family_count": _EXPECTED_FAMILIES,
        "native_quadrature_node_count": _EXPECTED_NODES,
        "native_supervised_token_count": _EXPECTED_SUPERVISED_TOKENS,
        "native_full_H4_row_count": _EXPECTED_FULL_H4_ROWS,
        "native_H4_width": _EXPECTED_H4_WIDTH,
        "native_suffix_forward_count": _EXPECTED_NATIVE_SUFFIX_FORWARDS,
        "native_vjp_pullback_count": _EXPECTED_NATIVE_PULLBACK_CHUNKS,
        "logical_native_vjp_input_gradient_coordinate_count": (
            _EXPECTED_NATIVE_INPUT_GRADIENT_COORDINATES
        ),
        "canonical_output_cotangent_row_count": (
            _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
        ),
        "native_token_directional_derivative_count": (
            _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
        ),
        "published_v11_jvp_node_token_reference_element_count": (
            _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
        ),
        "native_GL4_token_weight_application_count": (
            _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
        ),
        "fresh_full_model_forward_count": 0,
        "fresh_full_model_backward_count": 0,
    }
    if any(resources[key] != value for key, value in expected.items()):
        raise RuntimeError("V13-B native core resource accounting differs")
    return resources


def _validate_native_vjp_receipts(
    *,
    runtime_receipts: Sequence[Mapping[str, object]],
    runtime_receipt_set_sha256: str,
    resource_receipts: Sequence[Mapping[str, object]],
    resource_receipt_set_sha256: str,
) -> dict[str, object]:
    """Rehash all V13-B node/resource receipts and their exact ownership sets."""

    runtimes = tuple(runtime_receipts)
    resources = tuple(resource_receipts)
    runtime_artifacts: list[str] = []
    resource_artifacts: list[str] = []
    runtime_keys: list[tuple[str, int]] = []
    resource_keys: list[tuple[str, int]] = []
    families: dict[str, str] = {}

    for value in runtimes:
        payload = dict(value)
        artifact = v10diag._require_sha256(
            payload.pop("receipt_sha256", None),
            label="V13-B native runtime receipt",
        )
        if artifact != v10diag.token_v1._domain_sha256(
            payload, domain=_NATIVE_RUNTIME_NODE_DOMAIN
        ):
            raise RuntimeError("V13-B native runtime receipt drifted")
        example = v10diag.token_v1._identifier(
            payload.get("example_id"), label="V13-B native runtime example"
        )
        family = v10diag.token_v1._identifier(
            payload.get("family_id"), label="V13-B native runtime family"
        )
        node = payload.get("node_index")
        ordinal = payload.get("live_result_ordinal")
        runtime_metadata = payload.get("native_suffix_vjp_runtime_receipt")
        runtime_token_count = (
            runtime_metadata.get("token_count")
            if isinstance(runtime_metadata, Mapping)
            else None
        )
        if (
            type(node) is not int
            or not 0 <= node < 4
            or type(ordinal) is not int
            or not 1 <= ordinal <= _EXPECTED_NODES
            or payload.get("path_fraction_hex")
            != v10diag.GL4_UNIT_INTERVAL_NODES[node].hex()
            or payload.get("quadrature_weight_hex")
            != v10diag.GL4_UNIT_INTERVAL_WEIGHTS[node].hex()
            or not isinstance(runtime_metadata, Mapping)
            or type(runtime_token_count) is not int
            or runtime_token_count <= 0
            or runtime_metadata.get("ad_mechanism")
            != "torch.func.vjp.reverse_mode"
            or runtime_metadata.get("vjp_has_aux") is not True
            or runtime_metadata.get("vjp_transform_count") != 1
            or runtime_metadata.get("suffix_segment_call_count") != 13
            or runtime_metadata.get("logit_projection_call_count") != 1
            or runtime_metadata.get("h4_dtype_cast_count") != 1
            or runtime_metadata.get("token_cotangent_chunk_size") != 8
            or runtime_metadata.get("pullback_vectorization")
            != "torch.vmap_over_canonical_token_cotangents"
            or runtime_metadata.get("vjp_pullback_chunk_call_count")
            != (runtime_token_count + 7) // 8
            or runtime_metadata.get("vmap_pullback_call_count")
            != (runtime_token_count + 7) // 8
            or runtime_metadata.get("token_cotangent_coverage_count")
            != runtime_token_count
            or runtime_metadata.get("token_cotangent_nonzero_count")
            != runtime_token_count
            or runtime_metadata.get("token_cotangent_element_count")
            != runtime_token_count * runtime_token_count
            or runtime_metadata.get("full_suffix_h4_bitwise_equal") is not True
            or runtime_metadata.get("full_suffix_logits_bitwise_equal") is not True
            or runtime_metadata.get("full_suffix_token_teacher_kl_bitwise_equal")
            is not True
            or runtime_metadata.get("direction_is_exactly_zero_outside_support")
            is not True
            or payload.get("native_vjp_primal_count") != 1
            or payload.get("native_suffix_vjp_runtime_artifact_sha256")
            != runtime_metadata.get("artifact_sha256")
            or payload.get("runtime_path_h4_matches_constructed_f64_path_bitwise")
            is not True
            or payload.get("runtime_cast_h4_matches_V10_full_node_bitwise")
            is not True
            or payload.get("runtime_full_logits_matches_V10_full_node_bitwise")
            is not True
            or payload.get("runtime_primal_token_KL_matches_V10_f64_bitwise")
            is not True
            or payload.get("same_node_direction_matches_V11_bitwise") is not True
            or payload.get("no_discarded_preflight_result") is not True
            or payload.get("raw_tensors_serialized") is not False
        ):
            raise RuntimeError("V13-B native runtime receipt semantics differ")
        runtime_payload = dict(runtime_metadata)
        runtime_artifact = v10diag._require_sha256(
            runtime_payload.pop("artifact_sha256", None),
            label="V13-B nested native runtime artifact",
        )
        if runtime_artifact != v10diag.token_v1._domain_sha256(
            runtime_payload, domain=native_runtime._RECEIPT_DOMAIN
        ):
            raise RuntimeError("V13-B nested native runtime receipt drifted")
        chunk_values = runtime_metadata.get("chunk_receipts")
        if not isinstance(chunk_values, (tuple, list)):
            raise RuntimeError("V13-B native runtime chunks differ")
        expected_start = 0
        for chunk_value in chunk_values:
            if not isinstance(chunk_value, Mapping):
                raise RuntimeError("V13-B native runtime chunk differs")
            chunk_payload = dict(chunk_value)
            chunk_artifact = v10diag._require_sha256(
                chunk_payload.pop("artifact_sha256", None),
                label="V13-B nested native chunk artifact",
            )
            start = chunk_payload.get("token_start")
            stop = chunk_payload.get("token_stop")
            if (
                chunk_artifact
                != v10diag.token_v1._domain_sha256(
                    chunk_payload, domain=native_runtime._CHUNK_DOMAIN
                )
                or type(start) is not int
                or type(stop) is not int
                or start != expected_start
                or not start < stop <= runtime_token_count
                or stop - start > 8
                or chunk_payload.get("token_count") != runtime_token_count
                or chunk_payload.get("pullback_mechanism")
                != "torch.func.vmap(pullback)"
                or chunk_payload.get("canonical_one_hot_token_cotangents")
                is not True
                or chunk_payload.get("vmap_pullback_call_count") != 1
                or chunk_payload.get("token_cotangent_nonzero_count")
                != stop - start
                or chunk_payload.get("token_cotangent_element_count")
                != (stop - start) * runtime_token_count
            ):
                raise RuntimeError("V13-B native runtime chunk semantics differ")
            expected_start = stop
        if (
            expected_start != runtime_token_count
            or len(chunk_values) != (runtime_token_count + 7) // 8
        ):
            raise RuntimeError("V13-B native runtime chunk coverage differs")
        for key in (
            "pinned_v10_core_node_receipt_artifact_sha256",
            "pinned_v12_contraction_node_artifact_sha256",
            "native_suffix_vjp_runtime_artifact_sha256",
            "provider_artifact_sha256",
            "execution_artifact_sha256",
            "path_h4_sha256",
            "supervised_grid_sha256",
            "primal_token_teacher_kl_sha256",
        ):
            v10diag._require_sha256(
                payload.get(key), label=f"V13-B native runtime {key}"
            )
        if example in families and families[example] != family:
            raise RuntimeError("V13-B native runtime family ownership differs")
        families[example] = family
        runtime_keys.append((example, node))
        runtime_artifacts.append(artifact)

    runtime_artifact_by_key = dict(
        zip(runtime_keys, runtime_artifacts, strict=True)
    )
    for value in resources:
        payload = dict(value)
        artifact = v10diag._require_sha256(
            payload.pop("artifact_sha256", None),
            label="V13-B native resource receipt",
        )
        if artifact != v10diag.token_v1._domain_sha256(
            payload, domain=_NATIVE_RESOURCE_NODE_DOMAIN
        ):
            raise RuntimeError("V13-B native resource receipt drifted")
        example = v10diag.token_v1._identifier(
            payload.get("example_id"), label="V13-B native resource example"
        )
        family = v10diag.token_v1._identifier(
            payload.get("family_id"), label="V13-B native resource family"
        )
        node = payload.get("node_index")
        token_count = payload.get("token_count")
        full_rows = payload.get("full_h4_row_count")
        support_rows = payload.get("support_h4_row_count")
        outside_rows = payload.get("outside_support_h4_row_count")
        width = payload.get("h4_width")
        if (
            type(node) is not int
            or not 0 <= node < 4
            or type(token_count) is not int
            or token_count <= 0
            or type(full_rows) is not int
            or full_rows <= 0
            or type(support_rows) is not int
            or not 0 < support_rows <= full_rows
            or type(outside_rows) is not int
            or outside_rows != full_rows - support_rows
            or type(width) is not int
            or width != _EXPECTED_H4_WIDTH
            or families.get(example) != family
            or payload.get("native_runtime_receipt_sha256")
            != runtime_artifact_by_key.get((example, node))
            or payload.get("native_suffix_forward_count") != 1
            or payload.get("native_vjp_primal_count") != 1
            or payload.get("logit_projection_call_count") != 1
            or payload.get("h4_dtype_cast_count") != 1
            or payload.get("suffix_segment_call_count") != 13
            or payload.get("native_vjp_pullback_count")
            != (token_count + 7) // 8
            or payload.get("canonical_output_cotangent_row_count")
            != token_count
            or payload.get("dense_output_cotangent_coordinate_count")
            != token_count * token_count
            or payload.get("full_input_gradient_coordinate_count")
            != token_count * full_rows * width
            or payload.get("support_contraction_product_count")
            != token_count * support_rows * width
            or payload.get("outside_support_gradient_coordinate_count")
            != token_count * outside_rows * width
            or payload.get("direction_coordinate_validation_count")
            != full_rows * width
            or payload.get("outside_support_direction_zero_validation_count")
            != outside_rows * width
            or payload.get("outside_direction_nonzero_count") != 0
            or float(payload.get("maximum_outside_direction_abs", -1.0)) != 0.0
            or payload.get("outside_direction_zero_proved_before_support_contraction")
            is not True
            or payload.get("transient_full_cotangent_hashed") is not True
            or payload.get("transient_full_input_gradient_hashed") is not True
            or payload.get("transient_tensors_retained_after_result") is not False
            or payload.get("raw_tensors_serialized") is not False
        ):
            raise RuntimeError("V13-B native resource receipt semantics differ")
        for key in (
            "native_runtime_receipt_sha256",
            "support_indices_sha256",
            "token_basis_cotangent_sha256",
            "transient_full_cotangent_sha256",
            "transient_full_input_gradient_sha256",
            "support_input_gradient_sha256",
            "contracted_directional_vector_sha256",
        ):
            v10diag._require_sha256(
                payload.get(key), label=f"V13-B native resource {key}"
            )
        resource_keys.append((example, node))
        resource_artifacts.append(artifact)

    expected_keys = tuple(
        (example, node)
        for example in sorted(families)
        for node in range(4)
    )
    if (
        len(runtimes) != _EXPECTED_NODES
        or len(resources) != _EXPECTED_NODES
        or len(families) != _EXPECTED_PROMPTS
        or tuple(runtime_keys) != expected_keys
        or tuple(resource_keys) != expected_keys
        or len(set(runtime_keys)) != _EXPECTED_NODES
        or len(set(resource_keys)) != _EXPECTED_NODES
        or sorted(int(value["live_result_ordinal"]) for value in runtimes)
        != list(range(1, _EXPECTED_NODES + 1))
    ):
        raise RuntimeError("V13-B native receipt order or ownership differs")
    expected_runtime_set = v10diag.token_v1._domain_sha256(
        tuple(runtime_artifacts), domain=_NATIVE_RUNTIME_SET_DOMAIN
    )
    expected_resource_set = v10diag.token_v1._domain_sha256(
        tuple(resource_artifacts), domain=_NATIVE_RESOURCE_SET_DOMAIN
    )
    if (
        v10diag._require_sha256(
            runtime_receipt_set_sha256,
            label="V13-B native runtime receipt set",
        )
        != expected_runtime_set
    ):
        raise RuntimeError("V13-B native runtime receipt set drifted")
    if (
        v10diag._require_sha256(
            resource_receipt_set_sha256,
            label="V13-B native resource receipt set",
        )
        != expected_resource_set
    ):
        raise RuntimeError("V13-B native resource receipt set drifted")

    totals = {
        "native_suffix_node_count": len(resources),
        "native_vjp_primal_count": sum(
            int(value["native_vjp_primal_count"]) for value in resources
        ),
        "native_suffix_forward_count": sum(
            int(value["native_suffix_forward_count"]) for value in resources
        ),
        "native_vjp_pullback_count": sum(
            int(value["native_vjp_pullback_count"]) for value in resources
        ),
        "suffix_segment_call_count": sum(
            int(value["suffix_segment_call_count"]) for value in resources
        ),
        "logit_projection_call_count": sum(
            int(value["logit_projection_call_count"]) for value in resources
        ),
        "h4_dtype_cast_count": sum(
            int(value["h4_dtype_cast_count"]) for value in resources
        ),
        "canonical_output_cotangent_row_count": sum(
            int(value["canonical_output_cotangent_row_count"])
            for value in resources
        ),
        "dense_output_cotangent_coordinate_count": sum(
            int(value["dense_output_cotangent_coordinate_count"])
            for value in resources
        ),
        "full_input_gradient_coordinate_count": sum(
            int(value["full_input_gradient_coordinate_count"])
            for value in resources
        ),
        "support_contraction_product_count": sum(
            int(value["support_contraction_product_count"])
            for value in resources
        ),
        "outside_support_gradient_coordinate_count": sum(
            int(value["outside_support_gradient_coordinate_count"])
            for value in resources
        ),
        "direction_coordinate_validation_count": sum(
            int(value["direction_coordinate_validation_count"])
            for value in resources
        ),
        "outside_support_direction_zero_validation_count": sum(
            int(value["outside_support_direction_zero_validation_count"])
            for value in resources
        ),
    }
    expected_totals = {
        "native_suffix_node_count": _EXPECTED_NODES,
        "native_vjp_primal_count": _EXPECTED_NODES,
        "native_suffix_forward_count": _EXPECTED_NATIVE_SUFFIX_FORWARDS,
        "native_vjp_pullback_count": _EXPECTED_NATIVE_PULLBACK_CHUNKS,
        "suffix_segment_call_count": _EXPECTED_NATIVE_SUFFIX_SEGMENT_CALLS,
        "logit_projection_call_count": _EXPECTED_NODES,
        "h4_dtype_cast_count": _EXPECTED_NODES,
        "canonical_output_cotangent_row_count": (
            _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
        ),
        "dense_output_cotangent_coordinate_count": (
            _EXPECTED_DENSE_OUTPUT_COTANGENT_COORDINATES
        ),
        "full_input_gradient_coordinate_count": (
            _EXPECTED_NATIVE_INPUT_GRADIENT_COORDINATES
        ),
        "support_contraction_product_count": (
            _EXPECTED_NATIVE_SUPPORT_CONTRACTION_PRODUCTS
        ),
        "outside_support_gradient_coordinate_count": (
            _EXPECTED_NATIVE_OUTSIDE_SUPPORT_GRADIENT_COORDINATES
        ),
        "direction_coordinate_validation_count": (
            _EXPECTED_DIRECTION_COORDINATE_VALIDATIONS
        ),
        "outside_support_direction_zero_validation_count": (
            _EXPECTED_OUTSIDE_DIRECTION_ZERO_VALIDATIONS
        ),
    }
    if totals != expected_totals:
        raise RuntimeError("V13-B native receipt resource totals differ")
    return {
        **totals,
        "runtime_receipt_count": len(runtimes),
        "resource_receipt_count": len(resources),
        "prompt_count": len(families),
        "runtime_receipt_set_sha256": expected_runtime_set,
        "resource_receipt_set_sha256": expected_resource_set,
        "first_live_result_ordinal": min(
            int(value["live_result_ordinal"]) for value in runtimes
        ),
        "last_live_result_ordinal": max(
            int(value["live_result_ordinal"]) for value in runtimes
        ),
        "all_receipts_payload_set_order_and_ownership_rehashed": True,
    }


def _resource_accounting(
    *,
    v12_resources: Mapping[str, object],
    native_core_resources: Mapping[str, object],
    native_receipt_resources: Mapping[str, object],
) -> dict[str, object]:
    if (
        v12_resources.get("total_model_forward_count") != 224
        or v12_resources.get("total_backward_call_count") != 1_039
        or v12_resources.get("suffix_jvp_evaluation_count") != 64
        or native_core_resources.get("native_suffix_forward_count")
        != _EXPECTED_NATIVE_SUFFIX_FORWARDS
        or native_core_resources.get("native_vjp_pullback_count")
        != _EXPECTED_NATIVE_PULLBACK_CHUNKS
        or native_core_resources.get(
            "logical_native_vjp_input_gradient_coordinate_count"
        )
        != native_receipt_resources.get("full_input_gradient_coordinate_count")
        or native_core_resources.get("canonical_output_cotangent_row_count")
        != native_receipt_resources.get("canonical_output_cotangent_row_count")
        or native_core_resources.get("native_token_directional_derivative_count")
        != _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
        or native_receipt_resources.get("support_contraction_product_count")
        != _EXPECTED_NATIVE_SUPPORT_CONTRACTION_PRODUCTS
        or native_receipt_resources.get(
            "outside_support_gradient_coordinate_count"
        )
        != _EXPECTED_NATIVE_OUTSIDE_SUPPORT_GRADIENT_COORDINATES
    ):
        raise RuntimeError("V13-B combined resource accounting differs")
    phases = tuple(v12_resources.get("phase_order", ()))
    return {
        **dict(v12_resources),
        **dict(native_core_resources),
        "native_dense_output_cotangent_coordinate_count": (
            native_receipt_resources[
                "dense_output_cotangent_coordinate_count"
            ]
        ),
        "native_support_contraction_product_count": (
            native_receipt_resources["support_contraction_product_count"]
        ),
        "native_outside_support_gradient_coordinate_count": (
            native_receipt_resources[
                "outside_support_gradient_coordinate_count"
            ]
        ),
        "native_suffix_segment_call_count": native_receipt_resources[
            "suffix_segment_call_count"
        ],
        "native_logit_projection_call_count": native_receipt_resources[
            "logit_projection_call_count"
        ],
        "native_h4_dtype_cast_count": native_receipt_resources[
            "h4_dtype_cast_count"
        ],
        "native_runtime_receipt_count": native_receipt_resources[
            "runtime_receipt_count"
        ],
        "native_resource_receipt_count": native_receipt_resources[
            "resource_receipt_count"
        ],
        "native_runtime_receipt_set_sha256": native_receipt_resources[
            "runtime_receipt_set_sha256"
        ],
        "native_resource_receipt_set_sha256": native_receipt_resources[
            "resource_receipt_set_sha256"
        ],
        "first_native_live_result_ordinal": native_receipt_resources[
            "first_live_result_ordinal"
        ],
        "last_native_live_result_ordinal": native_receipt_resources[
            "last_live_result_ordinal"
        ],
        "additional_full_model_forward_count": 0,
        "additional_full_model_backward_count": 0,
        "additional_suffix_jvp_evaluation_count": 0,
        "V13_native_suffix_forwards_are_not_full_model_forwards": True,
        "V13_native_pullback_chunks_are_not_reported_as_full_model_backwards": (
            True
        ),
        "no_discarded_native_preflight_evaluation": True,
        "raw_logits_H4_JVP_or_VJP_tensors_retained_in_report": False,
        "FLOP_or_total_compute_claim": False,
        "phase_order": (
            *phases,
            "sixty_four_same_node_native_suffix_reverse_mode_VJP_primals",
            "four_hundred_thirty_six_canonical_token_cotangent_pullback_chunks",
            "family_equal_native_VJP_nodewise_and_integrated_summary",
        ),
    }


def _require_integrity_gate_results(gates: Mapping[str, bool]) -> None:
    failed = tuple(sorted(key for key, value in gates.items() if value is not True))
    if failed:
        raise RuntimeError(
            "V13-B integrity gates failed before publication: " + ", ".join(failed)
        )


def _safety_metadata() -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_JVP_tensors": False,
        "contains_native_VJP_tensors": False,
        "contains_token_teacher_KL_tensors": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


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


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_native_vjp_diagnostic(
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
    v12_report_path: Path | str = DEFAULT_V12_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the locked V13-B same-node native reverse-mode diagnostic."""

    destination = v10diag.token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V13-B native VJP report")

    # Authenticate the complete V12 authority before constructing a model.
    v12_report_before = _load_v12_report(v12_report_path)
    v11_report_before = v12diag._load_v11_report(v11_report_path)
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
        label="V13-B native VJP rank320 materialization",
    )
    transfer = v10diag.token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=v10diag.token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=v10diag.token_v1.TRANSFER_REPORT_SHA256,
        label="V13-B native VJP rank320 transfer",
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
    context = v10diag.prepare_complete_h4_rank320_live_context(
        cache_dir=cache_dir
    )
    try:
        traces, endpoint_resources = v10diag.token_v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if len(traces) != _EXPECTED_PROMPTS or len(families) != _EXPECTED_FAMILIES:
            raise RuntimeError("V13-B native VJP A16 panel differs")
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
        collector = _CompositeSuffixNativeVJPCollector(context=context)
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
        native = collector.finalize(live.evidence)
        contraction = native.contraction
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
        raise RuntimeError("V13-B callback-enabled V10 resource replay drifted")
    reproduced_v11_resources = v11diag._resource_accounting(
        v10_resources=reproduced_v10_resources,
        suffix_resources=contraction.suffix.resources,
    )
    live_v11_sections = v12diag._live_v11_sections(
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
    v11_replay = v12diag._authenticate_live_v11_sections(
        report=v11_report_before, live_sections=live_v11_sections
    )
    v11_report_after = v12diag._load_v11_report(v11_report_path)
    if _canonical(v11_report_before) != _canonical(v11_report_after):
        raise RuntimeError("V11 anchor changed during V13-B model work")

    v12_resources = v12diag._resource_accounting(
        v11_resources=reproduced_v11_resources,
        contraction_resources=contraction.resources,
    )
    live_v12_sections = _live_v12_sections(
        live_v11_sections=live_v11_sections,
        v11_replay=v11_replay,
        contraction=contraction,
        reduction_environment=collector.reduction_environment,
        resources=v12_resources,
    )
    v12_replay = _authenticate_live_v12_sections(
        report=v12_report_before, live_sections=live_v12_sections
    )
    # Re-read the exact V12 authority only after every native pullback and
    # require the complete logical payload to remain unchanged.
    v12_report_after = _load_v12_report(v12_report_path)
    if _canonical(v12_report_before) != _canonical(v12_report_after):
        raise RuntimeError("V12 anchor changed during V13-B model work")

    native_authentication = _validate_native_vjp_receipts(
        runtime_receipts=native.runtime_receipts,
        runtime_receipt_set_sha256=native.runtime_receipt_set_sha256,
        resource_receipts=native.resource_receipts,
        resource_receipt_set_sha256=native.resource_receipt_set_sha256,
    )
    resources = _resource_accounting(
        v12_resources=v12_resources,
        native_core_resources=native.core_resources,
        native_receipt_resources=native_authentication,
    )
    comparison = native.comparison
    comparison_metadata = comparison.metadata()
    integrity_gates = {
        "exact_V12_file_and_logical_hash_authenticated_before_and_after_model_work": (
            v12_replay["v12_report_file_sha256"] == V12_REPORT_FILE_SHA256
            and v12_replay["v12_report_sha256"] == V12_REPORT_SHA256
            and _canonical(v12_report_before) == _canonical(v12_report_after)
        ),
        "live_execution_replays_every_published_V12_computational_section": (
            v12_replay["all_live_sections_canonically_equal"] is True
            and v12_replay["section_count"] == 19
        ),
        "all_64_native_runtime_and_resource_receipts_rehashed_with_exact_ownership": (
            native_authentication[
                "all_receipts_payload_set_order_and_ownership_rehashed"
            ]
            is True
            and native_authentication["runtime_receipt_count"] == _EXPECTED_NODES
            and native_authentication["resource_receipt_count"] == _EXPECTED_NODES
        ),
        "all_64_native_primals_casts_logits_and_token_KLs_replay_bitwise": (
            len(native.runtime_receipts) == _EXPECTED_NODES
            and all(
                value[
                    "runtime_path_h4_matches_constructed_f64_path_bitwise"
                ]
                is True
                and value["runtime_cast_h4_matches_V10_full_node_bitwise"]
                is True
                and value["runtime_full_logits_matches_V10_full_node_bitwise"]
                is True
                and value["runtime_primal_token_KL_matches_V10_f64_bitwise"]
                is True
                for value in native.runtime_receipts
            )
        ),
        "canonical_one_hot_vmap_pullbacks_cover_exactly_3212_tokens_in_436_chunks": (
            native_authentication["canonical_output_cotangent_row_count"]
            == _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
            and native_authentication["native_vjp_pullback_count"]
            == _EXPECTED_NATIVE_PULLBACK_CHUNKS
            and native.runtime_resources[
                "canonical_token_cotangent_coverage_count"
            ]
            == _EXPECTED_NATIVE_TOKEN_NODE_COVERAGE
            and native.runtime_resources["vmap_pullback_call_count"]
            == _EXPECTED_NATIVE_PULLBACK_CHUNKS
        ),
        "outside_direction_zero_is_proved_before_every_support_only_contraction": (
            all(
                value[
                    "outside_direction_zero_proved_before_support_contraction"
                ]
                is True
                and value["outside_direction_nonzero_count"] == 0
                and float(value["maximum_outside_direction_abs"]) == 0.0
                for value in native.resource_receipts
            )
        ),
        "native_core_evidence_binds_exact_V12_and_V11_nodes": (
            comparison.replayed_v12_comparison_artifact_sha256
            == V12_COMPARISON_ARTIFACT_SHA256
            and comparison.replayed_v11_comparison_artifact_sha256
            == v12diag.V11_COMPARISON_ARTIFACT_SHA256
            and all(
                value.pinned_v12_evidence_artifact_sha256
                == value.contraction_precision_evidence.artifact_sha256
                and value.pinned_v11_evidence_artifact_sha256
                == value.contraction_precision_evidence.suffix_jvp_evidence.artifact_sha256
                for value in native.evidence
            )
        ),
        "nodewise_and_integrated_frozen_core_gates_are_both_published": (
            set(comparison.nodewise_gate_results)
            == {
                "overall_nodewise_symmetric_relative_RMSE_at_most_0_0001",
                "every_family_nodewise_symmetric_relative_RMSE_at_most_0_0001",
            }
            and set(comparison.integrated_gate_results)
            == {
                "overall_integrated_symmetric_relative_RMSE_at_most_0_0001",
                "every_family_integrated_symmetric_relative_RMSE_at_most_0_0001",
            }
        ),
        "fixed_native_VJP_telescope_closes_overall_and_in_every_family": (
            comparison.telescope_metrics.passed
            and all(
                family.telescope_metrics.passed
                for family in comparison.family_summaries
            )
        ),
        "first_live_native_result_is_one_of_64_with_no_discarded_preflight": (
            native_authentication["first_live_result_ordinal"] == 1
            and native_authentication["last_live_result_ordinal"] == 64
            and resources["no_discarded_native_preflight_evaluation"] is True
        ),
        "exact_native_resource_panel_and_zero_extra_full_model_work": (
            resources["native_suffix_forward_count"] == 64
            and resources["native_vjp_pullback_count"] == 436
            and resources["native_token_directional_derivative_count"] == 3_212
            and resources["additional_full_model_forward_count"] == 0
            and resources["additional_full_model_backward_count"] == 0
            and resources["total_model_forward_count"] == 224
            and resources["total_backward_call_count"] == 1_039
        ),
        "no_fit_correction_search_routing_serving_or_compression_mutation": True,
    }
    _require_integrity_gate_results(integrity_gates)
    passed = comparison.nodewise_passed and comparison.integrated_passed
    classification = comparison.classification
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_same_suffix_native_vjp",
            "outer_held_prompt_count": _EXPECTED_PROMPTS,
            "outer_family_count": _EXPECTED_FAMILIES,
            "prompts_per_outer_family": 2,
            "V12_control": "exact_pinned_contraction_precision_v12_and_live_replay",
            "V11_reference": "exact_same_node_forward_mode_suffix_JVP_vectors",
            "native_AD_mechanism": "torch.func.vjp.reverse_mode",
            "pullback_batching": "torch.func.vmap_over_canonical_one_hot_token_cotangents",
            "cotangent_chunk_size": 8,
            "same_node_GL4_evaluation_count": _EXPECTED_NODES,
            "aggregation": "equal_node_and_token_then_prompt_then_family",
            "adjoint_threshold": {
                "overall_and_every_family_symmetric_relative_RMSE_maximum": (
                    ADJOINT_RELATIVE_RMSE_MAXIMUM
                )
            },
            "nodewise_and_integrated_gates_are_independently_required": True,
            "first_native_execution_is_retained_as_live_result_one_of_64": True,
            "v13_fit_or_correction_performed": False,
            "search_selection_routing_or_fallback_performed": False,
            "result_can_authorize_serving_compression_or_model_mutation": False,
        },
        "v12_control_binding": {
            "file": str(v12_report_path),
            "file_sha256": V12_REPORT_FILE_SHA256,
            "report_sha256": V12_REPORT_SHA256,
            "comparison_artifact_sha256": V12_COMPARISON_ARTIFACT_SHA256,
            "raw_gradient_receipt_set_sha256": V12_RAW_GRADIENT_SET_SHA256,
            "cast_only_jvp_receipt_set_sha256": V12_CAST_ONLY_JVP_SET_SHA256,
            "reduction_environment_sha256": V12_REDUCTION_ENVIRONMENT_SHA256,
            "schema": v12_report_before.get("schema"),
            "classification": v12_report_before.get("classification"),
            "passed": v12_report_before.get("passed"),
            "live_reproduction": v12_replay,
            "authenticated_before_and_after_model_work": True,
        },
        "replayed_v12_computational_sections": dict(live_v12_sections),
        "native_suffix_vjp_runtime_receipts": native.runtime_receipts,
        "native_suffix_vjp_runtime_receipt_set_sha256": (
            native.runtime_receipt_set_sha256
        ),
        "native_suffix_vjp_resource_receipts": native.resource_receipts,
        "native_suffix_vjp_resource_receipt_set_sha256": (
            native.resource_receipt_set_sha256
        ),
        "native_suffix_vjp_receipt_authentication": native_authentication,
        "native_suffix_vjp_evidence": tuple(
            value.metadata() for value in native.evidence
        ),
        "family_equal_native_suffix_vjp_comparison": comparison_metadata,
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "outcome_matrix": {
            "outcome": classification,
            "integrity_completed": True,
            "nodewise_gate_results": tuple(
                sorted(comparison.nodewise_gate_results.items())
            ),
            "integrated_gate_results": tuple(
                sorted(comparison.integrated_gate_results.items())
            ),
            "nodewise_passed": comparison.nodewise_passed,
            "integrated_passed": comparison.integrated_passed,
            "fixed_telescope_passed": comparison.telescope_metrics.passed,
            "finite_correction_eligible": False,
            "fit_search_route_or_serving_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_diagnostic_only": True,
            "V12_file_and_all_live_computational_sections_replayed_exactly": True,
            "native_VJP_uses_the_same_post_H4_suffix_and_GL4_nodes_as_V11": True,
            "success_supports_a_V10_gradient_source_or_execution_path_difference": (
                passed
            ),
            "persistent_failure_retains_forward_reverse_AD_or_nondifferentiable_boundary_ambiguity": (
                not passed
            ),
            "finite_correction_eligible": False,
            "fit_or_correction_performed": False,
            "search_selection_routing_or_fallback_performed": False,
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


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Run locked Gemma-3 V13-B same-suffix native-VJP diagnostic."
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_native_vjp_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
