"""Pure V13-B comparison of same-suffix native VJP against V11 JVP.

V12 showed that replaying the V10 gradient banks with several explicit
contraction orders did not, by itself, identify the source of the small
forward/reverse adjoint gap.  This module adds no model execution.  It accepts
four native-VJP token vectors produced by the *same live suffix* used by V11,
binds them to the exact V12, V11, and GL4-node authorities, and compares:

* all four node vectors before quadrature integration;
* the GL4-integrated native VJP and V11 JVP vectors; and
* the integrated native VJP descriptively against every published V12 stage.

The frozen decision gate is symmetric relative RMSE at most ``1e-4`` overall
and in every family, independently for nodewise and integrated comparisons.
No stage is selected, fitted, corrected, searched, or authorized for serving.
Raw vectors remain typed, hashed evidence and never enter serialized metadata.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .complete_h4_tail_candidate_joint_state_contraction_precision import (
    CONTRACTION_PUBLISHED_STAGE_ORDER,
    CandidateJointStateContractionPrecisionEvidence,
    summarize_candidate_joint_state_contraction_precision,
)
from .complete_h4_tail_candidate_joint_state_suffix_jvp import (
    ADJOINT_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateSuffixJVPNodeEvidence,
    summarize_candidate_joint_state_suffix_jvp,
)
from .complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


__all__ = [
    "SUFFIX_NATIVE_VJP_TELESCOPE_POINTS",
    "SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS",
    "CandidateJointStateSuffixNativeVJPComparison",
    "CandidateJointStateSuffixNativeVJPEvidence",
    "CandidateJointStateSuffixNativeVJPFamilySummary",
    "CandidateJointStateSuffixNativeVJPNodeEvidence",
    "CandidateJointStateSuffixNativeVJPNodeMetrics",
    "CandidateJointStateSuffixNativeVJPPairMetrics",
    "CandidateJointStateSuffixNativeVJPStageMetrics",
    "CandidateJointStateSuffixNativeVJPTelescopeMetrics",
    "build_candidate_joint_state_suffix_native_vjp_evidence",
    "classify_candidate_joint_state_suffix_native_vjp",
    "summarize_candidate_joint_state_suffix_native_vjp",
]


SUFFIX_NATIVE_VJP_TELESCOPE_POINTS = (
    *CONTRACTION_PUBLISHED_STAGE_ORDER,
    "J64_suffix",
    "N64_native_vjp",
)
SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS = tuple(
    f"{target}_minus_{source}"
    for source, target in zip(
        SUFFIX_NATIVE_VJP_TELESCOPE_POINTS[:-1],
        SUFFIX_NATIVE_VJP_TELESCOPE_POINTS[1:],
        strict=True,
    )
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-native-vjp-tensor:v13\0"
_NODE_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-native-vjp-node:v13\0"
_EVIDENCE_DOMAIN = (
    b"fisher-graph:candidate-joint-state-suffix-native-vjp-evidence:v13\0"
)
_FAMILY_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-native-vjp-family:v13\0"
_SUMMARY_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-native-vjp-summary:v13\0"
_WEIGHTING = (
    "mean_tokens_and_GL4_nodes_within_prompt_then_equal_prompts_within_family_"
    "then_equal_families"
)
_INTEGRATED_WEIGHTING = (
    "mean_tokens_within_prompt_then_equal_prompts_within_family_then_equal_families"
)
_EPSILON = 64.0 * torch.finfo(torch.float64).eps
_TELESCOPE_TOLERANCE_FACTOR = 512.0 * torch.finfo(torch.float64).eps


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_float(value: object, *, label: str, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return float(value)


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise TypeError("hashed value must be a materialized strided tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + _canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "shape": tuple(int(size) for size in tensor.shape),
            }
        )
        + payload
    ).hexdigest()


def _float64_vector(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.numel() <= 0
        or value.dtype != torch.float64
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty float64 vector")
    return value.detach().to(device="cpu").clone().contiguous()


def _same_float(left: float, right: float) -> bool:
    return float(left).hex() == float(right).hex()


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
    )


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPNodeEvidence:
    """One native-VJP token vector bound to the corresponding V11 node."""

    node_index: int
    path_fraction: float
    quadrature_weight: float
    token_native_vjp_f64: Tensor = field(repr=False)
    pinned_v11_node_artifact_sha256: str
    pinned_v10_node_receipt_artifact_sha256: str
    native_suffix_runtime_receipt_sha256: str
    native_resource_receipt_sha256: str
    primal_token_teacher_kl_sha256: str
    provider_artifact_sha256: str
    execution_artifact_sha256: str
    path_h4_sha256: str
    supervised_grid_sha256: str
    endpoint_pair_binding_sha256: str
    native_suffix_forward_count: int
    native_vjp_pullback_count: int
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or not 0 <= self.node_index < 4:
            raise ValueError("native VJP node index must be in [0, 3]")
        node = _finite_float(self.path_fraction, label="native VJP path fraction")
        weight = _finite_float(
            self.quadrature_weight,
            label="native VJP quadrature weight",
            nonnegative=True,
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
        ):
            raise ValueError("native VJP node does not use the exact GL4 rule")
        vector = _float64_vector(
            self.token_native_vjp_f64,
            label="native VJP token vector",
        )
        for name in (
            "pinned_v11_node_artifact_sha256",
            "pinned_v10_node_receipt_artifact_sha256",
            "native_suffix_runtime_receipt_sha256",
            "native_resource_receipt_sha256",
            "primal_token_teacher_kl_sha256",
            "provider_artifact_sha256",
            "execution_artifact_sha256",
            "path_h4_sha256",
            "supervised_grid_sha256",
            "endpoint_pair_binding_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), label=name),
            )
        forwards = _positive_int(
            self.native_suffix_forward_count,
            label="native suffix forward count",
        )
        pullbacks = _positive_int(
            self.native_vjp_pullback_count,
            label="native VJP pullback count",
        )
        object.__setattr__(self, "path_fraction", node)
        object.__setattr__(self, "quadrature_weight", weight)
        object.__setattr__(self, "token_native_vjp_f64", vector)
        object.__setattr__(self, "native_suffix_forward_count", forwards)
        object.__setattr__(self, "native_vjp_pullback_count", pullbacks)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def token_count(self) -> int:
        return int(self.token_native_vjp_f64.shape[0])

    def token_vector_f64(self) -> Tensor:
        return self.token_native_vjp_f64.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        vector = self.token_vector_f64()
        result: dict[str, object] = {
            "node_index": self.node_index,
            "path_fraction_hex": self.path_fraction.hex(),
            "quadrature_weight_hex": self.quadrature_weight.hex(),
            "token_count": self.token_count,
            "token_native_vjp_f64_sha256": _tensor_sha256(vector),
            "token_native_vjp_mean": float(vector.mean()),
            "token_native_vjp_rms": float(vector.square().mean().sqrt()),
            "token_native_vjp_maximum_abs": float(vector.abs().max()),
            "pinned_v11_node_artifact_sha256": (
                self.pinned_v11_node_artifact_sha256
            ),
            "pinned_v10_node_receipt_artifact_sha256": (
                self.pinned_v10_node_receipt_artifact_sha256
            ),
            "native_suffix_runtime_receipt_sha256": (
                self.native_suffix_runtime_receipt_sha256
            ),
            "native_resource_receipt_sha256": self.native_resource_receipt_sha256,
            "primal_token_teacher_kl_sha256": self.primal_token_teacher_kl_sha256,
            "provider_artifact_sha256": self.provider_artifact_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "path_h4_sha256": self.path_h4_sha256,
            "supervised_grid_sha256": self.supervised_grid_sha256,
            "endpoint_pair_binding_sha256": self.endpoint_pair_binding_sha256,
            "native_suffix_forward_count": self.native_suffix_forward_count,
            "native_vjp_pullback_count": self.native_vjp_pullback_count,
            "raw_tensor_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        vector = self.token_native_vjp_f64
        if (
            vector.dtype != torch.float64
            or vector.device.type != "cpu"
            or vector.ndim != 1
            or vector.numel() <= 0
            or vector.requires_grad
            or not vector.is_contiguous()
            or not bool(torch.isfinite(vector).all())
            or self.path_fraction.hex()
            != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or self.quadrature_weight.hex()
            != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
            or _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256,
                label="native VJP node artifact",
            )
        ):
            raise RuntimeError("candidate joint-state native VJP node drifted")


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPEvidence:
    """One prompt's native VJPs bound to exact V12 and V11 authorities."""

    contraction_precision_evidence: CandidateJointStateContractionPrecisionEvidence = (
        field(repr=False)
    )
    nodes: tuple[CandidateJointStateSuffixNativeVJPNodeEvidence, ...]
    pinned_v12_evidence_artifact_sha256: str
    pinned_v11_evidence_artifact_sha256: str
    example_id: str = field(init=False)
    family_id: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        v12 = self.contraction_precision_evidence
        if not isinstance(v12, CandidateJointStateContractionPrecisionEvidence):
            raise TypeError("native VJP evidence requires V12 contraction evidence")
        v12.validate_integrity()
        v11 = v12.suffix_jvp_evidence
        pinned_v12 = _require_sha256(
            self.pinned_v12_evidence_artifact_sha256,
            label="pinned V12 evidence artifact",
        )
        pinned_v11 = _require_sha256(
            self.pinned_v11_evidence_artifact_sha256,
            label="pinned V11 evidence artifact",
        )
        if pinned_v12 != v12.artifact_sha256 or pinned_v11 != v11.artifact_sha256:
            raise ValueError("native VJP authority pin differs from V12 or V11")
        nodes = tuple(self.nodes)
        if (
            len(nodes) != 4
            or any(
                not isinstance(node, CandidateJointStateSuffixNativeVJPNodeEvidence)
                or node.node_index != index
                for index, node in enumerate(nodes)
            )
        ):
            raise ValueError("native VJP evidence requires ordered four GL4 nodes")
        for node, suffix_node in zip(nodes, v11.nodes, strict=True):
            node.validate_integrity()
            self._validate_node_provenance(node=node, suffix_node=suffix_node)
        if (
            len({node.native_suffix_runtime_receipt_sha256 for node in nodes}) != 4
            or len({node.native_resource_receipt_sha256 for node in nodes}) != 4
            or len({node.artifact_sha256 for node in nodes}) != 4
        ):
            raise ValueError("native VJP node ownership must be node-distinct")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "pinned_v12_evidence_artifact_sha256", pinned_v12)
        object.__setattr__(self, "pinned_v11_evidence_artifact_sha256", pinned_v11)
        object.__setattr__(self, "example_id", v12.example_id)
        object.__setattr__(self, "family_id", v12.family_id)
        if self.maximum_telescope_abs_error > self.telescope_tolerance:
            raise ValueError("native VJP float64 telescope failed")
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @staticmethod
    def _validate_node_provenance(
        *,
        node: CandidateJointStateSuffixNativeVJPNodeEvidence,
        suffix_node: CandidateJointStateSuffixJVPNodeEvidence,
    ) -> None:
        if (
            node.token_count != suffix_node.token_count
            or node.path_fraction.hex() != suffix_node.path_fraction.hex()
            or node.quadrature_weight.hex() != suffix_node.quadrature_weight.hex()
            or node.pinned_v11_node_artifact_sha256 != suffix_node.artifact_sha256
            or node.pinned_v10_node_receipt_artifact_sha256
            != suffix_node.pinned_v10_node_receipt_artifact_sha256
            or node.primal_token_teacher_kl_sha256
            != suffix_node.primal_token_teacher_kl_sha256
            or node.provider_artifact_sha256 != suffix_node.provider_artifact_sha256
            or node.execution_artifact_sha256 != suffix_node.execution_artifact_sha256
            or node.path_h4_sha256 != suffix_node.path_h4_sha256
            or node.supervised_grid_sha256 != suffix_node.supervised_grid_sha256
            or node.endpoint_pair_binding_sha256
            != suffix_node.endpoint_pair_binding_sha256
        ):
            raise ValueError("native VJP node provenance differs from exact V11")

    @property
    def supervised_tokens(self) -> int:
        return self.contraction_precision_evidence.supervised_tokens

    def native_node_matrix_f64(self) -> Tensor:
        return torch.stack(
            tuple(node.token_vector_f64() for node in self.nodes),
            dim=1,
        ).contiguous()

    def jvp_node_matrix_f64(self) -> Tensor:
        return torch.stack(
            tuple(
                node.directional_derivative_f64()
                for node in self.contraction_precision_evidence.suffix_jvp_evidence.nodes
            ),
            dim=1,
        ).contiguous()

    def integrated_native_vjp_f64(self) -> Tensor:
        total = torch.zeros(self.supervised_tokens, dtype=torch.float64)
        for node in self.nodes:
            total.add_(node.token_vector_f64(), alpha=node.quadrature_weight)
        return total.contiguous()

    def integrated_jvp_f64(self) -> Tensor:
        return (
            self.contraction_precision_evidence.suffix_jvp_evidence
            .integrated_suffix_jvp_f64()
        )

    def v12_stage_f64(self, stage: str) -> Tensor:
        return self.contraction_precision_evidence.stage_vector_f64(stage)

    def _telescope_points(self) -> dict[str, Tensor]:
        return {
            **{
                stage: self.v12_stage_f64(stage)
                for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
            },
            "J64_suffix": self.integrated_jvp_f64(),
            "N64_native_vjp": self.integrated_native_vjp_f64(),
        }

    def telescope_residual_f64(self) -> Tensor:
        points = self._telescope_points()
        reconstructed = torch.zeros(self.supervised_tokens, dtype=torch.float64)
        for source, target in zip(
            SUFFIX_NATIVE_VJP_TELESCOPE_POINTS[:-1],
            SUFFIX_NATIVE_VJP_TELESCOPE_POINTS[1:],
            strict=True,
        ):
            reconstructed = (
                reconstructed + points[target] - points[source]
            ).contiguous()
        direct = (
            points[SUFFIX_NATIVE_VJP_TELESCOPE_POINTS[-1]]
            - points[SUFFIX_NATIVE_VJP_TELESCOPE_POINTS[0]]
        ).contiguous()
        return (reconstructed - direct).contiguous()

    @property
    def telescope_tolerance(self) -> float:
        points = self._telescope_points()
        scale = max(
            1.0,
            *(float(value.abs().max()) for value in points.values()),
        )
        return float(
            _TELESCOPE_TOLERANCE_FACTOR
            * len(SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS)
            * scale
        )

    @property
    def maximum_telescope_abs_error(self) -> float:
        return float(self.telescope_residual_f64().abs().max())

    @property
    def resource_accounting(self) -> dict[str, int]:
        full_rows = self.contraction_precision_evidence.full_h4_row_count
        width = self.contraction_precision_evidence.h4_width
        return {
            "quadrature_node_count": 4,
            "supervised_token_count": self.supervised_tokens,
            "full_h4_row_count": full_rows,
            "h4_width": width,
            "native_suffix_forward_count": sum(
                node.native_suffix_forward_count for node in self.nodes
            ),
            "native_vjp_pullback_count": sum(
                node.native_vjp_pullback_count for node in self.nodes
            ),
            "logical_native_vjp_input_gradient_coordinate_count": (
                4 * self.supervised_tokens * full_rows * width
            ),
            "canonical_output_cotangent_row_count": (
                4 * self.supervised_tokens
            ),
            "native_token_directional_derivative_count": (
                4 * self.supervised_tokens
            ),
            "published_v11_jvp_node_token_reference_element_count": (
                4 * self.supervised_tokens
            ),
            "native_GL4_token_weight_application_count": (
                4 * self.supervised_tokens
            ),
            "fresh_full_model_forward_count": 0,
            "fresh_full_model_backward_count": 0,
        }

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        native_nodes = self.native_node_matrix_f64()
        jvp_nodes = self.jvp_node_matrix_f64()
        native = self.integrated_native_vjp_f64()
        jvp = self.integrated_jvp_f64()
        stages = {
            stage: self.v12_stage_f64(stage)
            for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
        }
        points = self._telescope_points()
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "pinned_v12_evidence_artifact_sha256": (
                self.pinned_v12_evidence_artifact_sha256
            ),
            "pinned_v11_evidence_artifact_sha256": (
                self.pinned_v11_evidence_artifact_sha256
            ),
            "nodes": tuple(node.metadata() for node in self.nodes),
            "native_node_matrix_f64_sha256": _tensor_sha256(native_nodes),
            "v11_jvp_node_matrix_f64_sha256": _tensor_sha256(jvp_nodes),
            "nodewise_native_minus_jvp_f64_sha256": _tensor_sha256(
                (native_nodes - jvp_nodes).contiguous()
            ),
            "integrated_native_vjp_f64_sha256": _tensor_sha256(native),
            "integrated_v11_jvp_f64_sha256": _tensor_sha256(jvp),
            "integrated_native_minus_jvp_f64_sha256": _tensor_sha256(
                (native - jvp).contiguous()
            ),
            "v12_stage_vectors": tuple(
                (
                    stage,
                    _tensor_sha256(stages[stage]),
                    _tensor_sha256((native - stages[stage]).contiguous()),
                )
                for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
            ),
            "telescope_points": tuple(
                (name, _tensor_sha256(points[name]))
                for name in SUFFIX_NATIVE_VJP_TELESCOPE_POINTS
            ),
            "telescope_transitions": SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS,
            "telescope_residual_f64_sha256": _tensor_sha256(
                self.telescope_residual_f64()
            ),
            "maximum_telescope_abs_error": self.maximum_telescope_abs_error,
            "telescope_tolerance": self.telescope_tolerance,
            "telescope_passed": True,
            "resource_accounting": tuple(sorted(self.resource_accounting.items())),
            "native_and_jvp_share_exact_V11_node_provenance": True,
            "V12_and_V11_authority_pins_exact": True,
            "descriptive_V12_stage_comparison_only": True,
            "raw_tensors_serialized": False,
            "fits_corrects_searches_or_selects_candidates": False,
            "authorizes_serving_compression_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        v12 = self.contraction_precision_evidence
        v12.validate_integrity()
        v11 = v12.suffix_jvp_evidence
        try:
            if (
                self.pinned_v12_evidence_artifact_sha256 != v12.artifact_sha256
                or self.pinned_v11_evidence_artifact_sha256 != v11.artifact_sha256
                or self.example_id != v12.example_id
                or self.family_id != v12.family_id
                or len(self.nodes) != 4
            ):
                raise RuntimeError
            for node, suffix_node in zip(self.nodes, v11.nodes, strict=True):
                node.validate_integrity()
                self._validate_node_provenance(node=node, suffix_node=suffix_node)
            if (
                len({node.native_suffix_runtime_receipt_sha256 for node in self.nodes})
                != 4
                or len({node.native_resource_receipt_sha256 for node in self.nodes})
                != 4
                or self.maximum_telescope_abs_error > self.telescope_tolerance
            ):
                raise RuntimeError
            resources = self.resource_accounting
            if (
                resources["quadrature_node_count"] != 4
                or resources["native_token_directional_derivative_count"]
                != 4 * self.supervised_tokens
                or resources["published_v11_jvp_node_token_reference_element_count"]
                != 4 * self.supervised_tokens
                or resources["fresh_full_model_forward_count"] != 0
                or resources["fresh_full_model_backward_count"] != 0
            ):
                raise RuntimeError
            if (
                _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False))
                != _require_sha256(
                    self.artifact_sha256,
                    label="native VJP evidence artifact",
                )
            ):
                raise RuntimeError
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "candidate joint-state native VJP evidence drifted"
            ) from error


def build_candidate_joint_state_suffix_native_vjp_evidence(
    *,
    contraction_precision_evidence: CandidateJointStateContractionPrecisionEvidence,
    token_native_vjp_node_vectors_f64: Sequence[Tensor],
    native_suffix_runtime_receipt_sha256s: Sequence[str],
    native_resource_receipt_sha256s: Sequence[str],
    native_suffix_forward_counts: Sequence[int],
    native_vjp_pullback_counts: Sequence[int],
) -> CandidateJointStateSuffixNativeVJPEvidence:
    """Bind four already-computed native VJPs to exact V12/V11 evidence."""

    v12 = contraction_precision_evidence
    if not isinstance(v12, CandidateJointStateContractionPrecisionEvidence):
        raise TypeError("native VJP builder requires typed V12 evidence")
    v12.validate_integrity()
    vectors = tuple(token_native_vjp_node_vectors_f64)
    runtime = tuple(native_suffix_runtime_receipt_sha256s)
    resource = tuple(native_resource_receipt_sha256s)
    forwards = tuple(native_suffix_forward_counts)
    pullbacks = tuple(native_vjp_pullback_counts)
    if any(len(values) != 4 for values in (vectors, runtime, resource, forwards, pullbacks)):
        raise ValueError("native VJP builder requires four values for every node field")
    v11 = v12.suffix_jvp_evidence
    nodes = tuple(
        CandidateJointStateSuffixNativeVJPNodeEvidence(
            node_index=suffix_node.node_index,
            path_fraction=suffix_node.path_fraction,
            quadrature_weight=suffix_node.quadrature_weight,
            token_native_vjp_f64=vectors[index],
            pinned_v11_node_artifact_sha256=suffix_node.artifact_sha256,
            pinned_v10_node_receipt_artifact_sha256=(
                suffix_node.pinned_v10_node_receipt_artifact_sha256
            ),
            native_suffix_runtime_receipt_sha256=runtime[index],
            native_resource_receipt_sha256=resource[index],
            primal_token_teacher_kl_sha256=(
                suffix_node.primal_token_teacher_kl_sha256
            ),
            provider_artifact_sha256=suffix_node.provider_artifact_sha256,
            execution_artifact_sha256=suffix_node.execution_artifact_sha256,
            path_h4_sha256=suffix_node.path_h4_sha256,
            supervised_grid_sha256=suffix_node.supervised_grid_sha256,
            endpoint_pair_binding_sha256=(
                suffix_node.endpoint_pair_binding_sha256
            ),
            native_suffix_forward_count=forwards[index],
            native_vjp_pullback_count=pullbacks[index],
        )
        for index, suffix_node in enumerate(v11.nodes)
    )
    return CandidateJointStateSuffixNativeVJPEvidence(
        contraction_precision_evidence=v12,
        nodes=nodes,
        pinned_v12_evidence_artifact_sha256=v12.artifact_sha256,
        pinned_v11_evidence_artifact_sha256=v11.artifact_sha256,
    )


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPPairMetrics:
    mean_native_vjp_f64: float
    mean_reference_f64: float
    mean_native_minus_reference_f64: float
    native_vjp_f64_rms: float
    reference_f64_rms: float
    difference_rmse: float
    symmetric_relative_rmse: float
    cosine: float
    maximum_absolute_error: float
    relative_rmse_epsilon: float

    def __post_init__(self) -> None:
        signed = {
            "mean_native_vjp_f64",
            "mean_reference_f64",
            "mean_native_minus_reference_f64",
            "cosine",
        }
        for name in self.__dataclass_fields__:
            value = _finite_float(
                getattr(self, name),
                label=f"native VJP pair metric {name}",
                nonnegative=name not in signed,
            )
            if name == "cosine" and not -1.0 <= value <= 1.0:
                raise ValueError("native VJP cosine must be in [-1, 1]")
            object.__setattr__(self, name, value)

    def metadata(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPNodeMetrics:
    node_index: int
    path_fraction: float
    quadrature_weight: float
    metrics: CandidateJointStateSuffixNativeVJPPairMetrics

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or not 0 <= self.node_index < 4:
            raise ValueError("native VJP node metric index must be in [0, 3]")
        node = _finite_float(self.path_fraction, label="node metric path fraction")
        weight = _finite_float(
            self.quadrature_weight,
            label="node metric quadrature weight",
            nonnegative=True,
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
            or not isinstance(self.metrics, CandidateJointStateSuffixNativeVJPPairMetrics)
        ):
            raise ValueError("native VJP node metric is invalid")
        object.__setattr__(self, "path_fraction", node)
        object.__setattr__(self, "quadrature_weight", weight)

    def metadata(self) -> dict[str, object]:
        return {
            "node_index": self.node_index,
            "path_fraction_hex": self.path_fraction.hex(),
            "quadrature_weight_hex": self.quadrature_weight.hex(),
            **self.metrics.metadata(),
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPStageMetrics:
    stage: str
    metrics: CandidateJointStateSuffixNativeVJPPairMetrics

    def __post_init__(self) -> None:
        if (
            self.stage not in CONTRACTION_PUBLISHED_STAGE_ORDER
            or not isinstance(self.metrics, CandidateJointStateSuffixNativeVJPPairMetrics)
        ):
            raise ValueError("native VJP stage metric is invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "orientation": f"N64_native_vjp_minus_{self.stage}",
            "descriptive_only": True,
            **self.metrics.metadata(),
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPTelescopeMetrics:
    residual_rmse: float
    maximum_absolute_residual: float
    maximum_tolerance: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    label=f"native VJP telescope {name}",
                    nonnegative=True,
                ),
            )
        if self.maximum_absolute_residual > self.maximum_tolerance:
            raise ValueError("native VJP telescope residual exceeds tolerance")

    @property
    def passed(self) -> bool:
        return self.maximum_absolute_residual <= self.maximum_tolerance

    def metadata(self) -> dict[str, object]:
        return {
            "points": SUFFIX_NATIVE_VJP_TELESCOPE_POINTS,
            "transitions": SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS,
            "residual_rmse": self.residual_rmse,
            "maximum_absolute_residual": self.maximum_absolute_residual,
            "maximum_tolerance": self.maximum_tolerance,
            "passed": self.passed,
        }


def classify_candidate_joint_state_suffix_native_vjp(
    *,
    nodewise_passed: bool,
    integrated_passed: bool,
) -> str:
    """Classify only whether same-suffix native VJP closes both adjoint gates."""

    if type(nodewise_passed) is not bool or type(integrated_passed) is not bool:
        raise TypeError("native VJP classification gates must be boolean")
    if nodewise_passed and integrated_passed:
        return "v10_gradient_source_or_execution_path_difference_supported"
    return "persistent_same_suffix_forward_reverse_ad_or_nondifferentiable_boundary_ambiguity"


def _canonical_evidence(
    evidence: Iterable[CandidateJointStateSuffixNativeVJPEvidence],
) -> tuple[CandidateJointStateSuffixNativeVJPEvidence, ...]:
    values = tuple(evidence)
    if not values or any(
        not isinstance(value, CandidateJointStateSuffixNativeVJPEvidence)
        for value in values
    ):
        raise TypeError("native VJP comparison requires typed evidence")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("native VJP example ids must be unique")

    # Runtime, resource, node, and upstream evidence artifacts have single
    # ownership.  Reuse across prompts or families is a protocol failure rather
    # than a weighting shortcut.
    ownership_groups = (
        tuple(value.pinned_v12_evidence_artifact_sha256 for value in ordered),
        tuple(value.pinned_v11_evidence_artifact_sha256 for value in ordered),
        tuple(node.artifact_sha256 for value in ordered for node in value.nodes),
        tuple(
            node.native_suffix_runtime_receipt_sha256
            for value in ordered
            for node in value.nodes
        ),
        tuple(
            node.native_resource_receipt_sha256
            for value in ordered
            for node in value.nodes
        ),
    )
    if any(len(values_) != len(set(values_)) for values_ in ownership_groups):
        raise ValueError("native VJP duplicate or cross-family ownership detected")
    return ordered


def _nested_scalar_mean(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
    values: Mapping[str, Tensor],
) -> float:
    if set(values) != {value.example_id for value in evidence}:
        raise ValueError("native VJP statistic keys differ from evidence")
    by_family: dict[str, list[Tensor]] = defaultdict(list)
    trailing_shape: tuple[int, ...] | None = None
    for value in evidence:
        statistic = values[value.example_id]
        if (
            not isinstance(statistic, Tensor)
            or statistic.ndim < 1
            or statistic.shape[0] != value.supervised_tokens
            or not statistic.is_floating_point()
            or not bool(torch.isfinite(statistic).all())
        ):
            raise ValueError("native VJP statistic must be finite and token aligned")
        statistic64 = statistic.detach().to(device="cpu", dtype=torch.float64)
        shape = tuple(int(size) for size in statistic64.shape[1:])
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("native VJP statistic shapes differ")
        by_family[value.family_id].append(statistic64.mean())
    return float(
        torch.stack(
            tuple(
                torch.stack(tuple(by_family[family])).mean()
                for family in sorted(by_family)
            )
        ).mean()
    )


def _rms(second_moment: float) -> float:
    return math.sqrt(max(second_moment, 0.0))


def _symmetric_relative_rmse(rmse: float, left_rms: float, right_rms: float) -> float:
    if left_rms <= _EPSILON and right_rms <= _EPSILON and rmse <= _EPSILON:
        return 0.0
    return rmse / max(left_rms, right_rms, _EPSILON)


def _cosine(*, cross: float, left_rms: float, right_rms: float) -> float:
    if left_rms <= _EPSILON and right_rms <= _EPSILON:
        return 1.0
    if left_rms <= _EPSILON or right_rms <= _EPSILON:
        return 0.0
    return min(max(cross / (left_rms * right_rms), -1.0), 1.0)


def _pair_metrics(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
    native: Mapping[str, Tensor],
    reference: Mapping[str, Tensor],
) -> CandidateJointStateSuffixNativeVJPPairMetrics:
    if set(native) != set(reference):
        raise ValueError("native VJP and reference membership differs")
    difference = {key: native[key] - reference[key] for key in native}

    def mean(values: Mapping[str, Tensor]) -> float:
        return _nested_scalar_mean(evidence, values)

    native_rms = _rms(mean({key: value.square() for key, value in native.items()}))
    reference_rms = _rms(
        mean({key: value.square() for key, value in reference.items()})
    )
    rmse = _rms(
        mean({key: value.square() for key, value in difference.items()})
    )
    return CandidateJointStateSuffixNativeVJPPairMetrics(
        mean_native_vjp_f64=mean(native),
        mean_reference_f64=mean(reference),
        mean_native_minus_reference_f64=mean(difference),
        native_vjp_f64_rms=native_rms,
        reference_f64_rms=reference_rms,
        difference_rmse=rmse,
        symmetric_relative_rmse=_symmetric_relative_rmse(
            rmse, native_rms, reference_rms
        ),
        cosine=_cosine(
            cross=mean({key: native[key] * reference[key] for key in native}),
            left_rms=native_rms,
            right_rms=reference_rms,
        ),
        maximum_absolute_error=max(
            float(value.abs().max()) for value in difference.values()
        ),
        relative_rmse_epsilon=_EPSILON,
    )


def _node_metrics(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
) -> tuple[CandidateJointStateSuffixNativeVJPNodeMetrics, ...]:
    result: list[CandidateJointStateSuffixNativeVJPNodeMetrics] = []
    for node_index, (node, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS, strict=True)
    ):
        result.append(
            CandidateJointStateSuffixNativeVJPNodeMetrics(
                node_index=node_index,
                path_fraction=node,
                quadrature_weight=weight,
                metrics=_pair_metrics(
                    evidence,
                    {
                        value.example_id: value.nodes[node_index].token_vector_f64()
                        for value in evidence
                    },
                    {
                        value.example_id: (
                            value.contraction_precision_evidence.suffix_jvp_evidence
                            .nodes[node_index]
                            .directional_derivative_f64()
                        )
                        for value in evidence
                    },
                ),
            )
        )
    return tuple(result)


def _aggregate_nodewise_metrics(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
) -> CandidateJointStateSuffixNativeVJPPairMetrics:
    return _pair_metrics(
        evidence,
        {value.example_id: value.native_node_matrix_f64() for value in evidence},
        {value.example_id: value.jvp_node_matrix_f64() for value in evidence},
    )


def _integrated_metrics(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
) -> CandidateJointStateSuffixNativeVJPPairMetrics:
    return _pair_metrics(
        evidence,
        {value.example_id: value.integrated_native_vjp_f64() for value in evidence},
        {value.example_id: value.integrated_jvp_f64() for value in evidence},
    )


def _stage_metrics(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
) -> tuple[CandidateJointStateSuffixNativeVJPStageMetrics, ...]:
    native = {
        value.example_id: value.integrated_native_vjp_f64() for value in evidence
    }
    return tuple(
        CandidateJointStateSuffixNativeVJPStageMetrics(
            stage=stage,
            metrics=_pair_metrics(
                evidence,
                native,
                {
                    value.example_id: value.v12_stage_f64(stage)
                    for value in evidence
                },
            ),
        )
        for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
    )


def _telescope_metrics(
    evidence: Sequence[CandidateJointStateSuffixNativeVJPEvidence],
) -> CandidateJointStateSuffixNativeVJPTelescopeMetrics:
    residuals = {
        value.example_id: value.telescope_residual_f64() for value in evidence
    }
    return CandidateJointStateSuffixNativeVJPTelescopeMetrics(
        residual_rmse=_rms(
            _nested_scalar_mean(
                evidence,
                {key: value.square() for key, value in residuals.items()},
            )
        ),
        maximum_absolute_residual=max(
            float(value.abs().max()) for value in residuals.values()
        ),
        maximum_tolerance=max(value.telescope_tolerance for value in evidence),
    )


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPFamilySummary:
    family_id: str
    example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    supervised_token_count: int
    nodewise_metrics: CandidateJointStateSuffixNativeVJPPairMetrics
    integrated_metrics: CandidateJointStateSuffixNativeVJPPairMetrics
    node_metrics: tuple[CandidateJointStateSuffixNativeVJPNodeMetrics, ...]
    stage_metrics: tuple[CandidateJointStateSuffixNativeVJPStageMetrics, ...]
    telescope_metrics: CandidateJointStateSuffixNativeVJPTelescopeMetrics
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        family = _identifier(self.family_id, label="native VJP family_id")
        examples = tuple(
            _identifier(value, label="native VJP family example_id")
            for value in self.example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="native VJP evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        nodes = tuple(self.node_metrics)
        stages = tuple(self.stage_metrics)
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or not isinstance(
                self.nodewise_metrics, CandidateJointStateSuffixNativeVJPPairMetrics
            )
            or not isinstance(
                self.integrated_metrics, CandidateJointStateSuffixNativeVJPPairMetrics
            )
            or tuple(value.node_index for value in nodes) != tuple(range(4))
            or tuple(value.stage for value in stages) != CONTRACTION_PUBLISHED_STAGE_ORDER
            or not isinstance(
                self.telescope_metrics,
                CandidateJointStateSuffixNativeVJPTelescopeMetrics,
            )
            or not self.telescope_metrics.passed
        ):
            raise ValueError("native VJP family membership is invalid")
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "node_metrics", nodes)
        object.__setattr__(self, "stage_metrics", stages)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FAMILY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "family_id": self.family_id,
            "example_ids": self.example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "prompt_count": len(self.example_ids),
            "supervised_token_count": self.supervised_token_count,
            "nodewise_metrics": self.nodewise_metrics.metadata(),
            "integrated_metrics": self.integrated_metrics.metadata(),
            "node_metrics": tuple(value.metadata() for value in self.node_metrics),
            "descriptive_native_vjp_vs_V12_stage_metrics": tuple(
                value.metadata() for value in self.stage_metrics
            ),
            "telescope_metrics": self.telescope_metrics.metadata(),
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            _sha256(_FAMILY_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256,
                label="native VJP family artifact",
            )
        ):
            raise RuntimeError("candidate joint-state native VJP family drifted")


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixNativeVJPComparison:
    evidence_example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    family_summaries: tuple[CandidateJointStateSuffixNativeVJPFamilySummary, ...]
    supervised_token_count: int
    nodewise_metrics: CandidateJointStateSuffixNativeVJPPairMetrics
    integrated_metrics: CandidateJointStateSuffixNativeVJPPairMetrics
    node_metrics: tuple[CandidateJointStateSuffixNativeVJPNodeMetrics, ...]
    stage_metrics: tuple[CandidateJointStateSuffixNativeVJPStageMetrics, ...]
    telescope_metrics: CandidateJointStateSuffixNativeVJPTelescopeMetrics
    replayed_v12_comparison_artifact_sha256: str
    replayed_v11_comparison_artifact_sha256: str
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        examples = tuple(
            _identifier(value, label="native VJP comparison example_id")
            for value in self.evidence_example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="native VJP evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        families = tuple(self.family_summaries)
        nodes = tuple(self.node_metrics)
        stages = tuple(self.stage_metrics)
        replayed_v12 = _require_sha256(
            self.replayed_v12_comparison_artifact_sha256,
            label="replayed V12 comparison artifact",
        )
        replayed_v11 = _require_sha256(
            self.replayed_v11_comparison_artifact_sha256,
            label="replayed V11 comparison artifact",
        )
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or not families
            or tuple(family.family_id for family in families)
            != tuple(sorted({family.family_id for family in families}))
            or set(examples)
            != {example for family in families for example in family.example_ids}
            or sum(len(family.example_ids) for family in families) != len(examples)
            or {
                example: digest
                for family in families
                for example, digest in zip(
                    family.example_ids,
                    family.evidence_artifact_sha256s,
                    strict=True,
                )
            }
            != dict(zip(examples, hashes, strict=True))
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or self.supervised_token_count
            != sum(family.supervised_token_count for family in families)
            or not isinstance(
                self.nodewise_metrics, CandidateJointStateSuffixNativeVJPPairMetrics
            )
            or not isinstance(
                self.integrated_metrics, CandidateJointStateSuffixNativeVJPPairMetrics
            )
            or tuple(value.node_index for value in nodes) != tuple(range(4))
            or tuple(value.stage for value in stages) != CONTRACTION_PUBLISHED_STAGE_ORDER
            or not isinstance(
                self.telescope_metrics,
                CandidateJointStateSuffixNativeVJPTelescopeMetrics,
            )
            or not self.telescope_metrics.passed
        ):
            raise ValueError("native VJP comparison membership is invalid")
        for family in families:
            family.validate_integrity()
        object.__setattr__(self, "evidence_example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "family_summaries", families)
        object.__setattr__(self, "node_metrics", nodes)
        object.__setattr__(self, "stage_metrics", stages)
        object.__setattr__(self, "replayed_v12_comparison_artifact_sha256", replayed_v12)
        object.__setattr__(self, "replayed_v11_comparison_artifact_sha256", replayed_v11)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SUMMARY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def nodewise_gate_results(self) -> dict[str, bool]:
        return {
            "overall_nodewise_symmetric_relative_RMSE_at_most_0_0001": (
                self.nodewise_metrics.symmetric_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
            ),
            "every_family_nodewise_symmetric_relative_RMSE_at_most_0_0001": all(
                family.nodewise_metrics.symmetric_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
                for family in self.family_summaries
            ),
        }

    @property
    def integrated_gate_results(self) -> dict[str, bool]:
        return {
            "overall_integrated_symmetric_relative_RMSE_at_most_0_0001": (
                self.integrated_metrics.symmetric_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
            ),
            "every_family_integrated_symmetric_relative_RMSE_at_most_0_0001": all(
                family.integrated_metrics.symmetric_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
                for family in self.family_summaries
            ),
        }

    @property
    def nodewise_passed(self) -> bool:
        return all(self.nodewise_gate_results.values())

    @property
    def integrated_passed(self) -> bool:
        return all(self.integrated_gate_results.values())

    @property
    def classification(self) -> str:
        return classify_candidate_joint_state_suffix_native_vjp(
            nodewise_passed=self.nodewise_passed,
            integrated_passed=self.integrated_passed,
        )

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence_example_ids": self.evidence_example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "family_summaries": tuple(
                family.metadata() for family in self.family_summaries
            ),
            "family_count": len(self.family_summaries),
            "prompt_count": len(self.evidence_example_ids),
            "supervised_token_count": self.supervised_token_count,
            "nodewise_metrics": self.nodewise_metrics.metadata(),
            "integrated_metrics": self.integrated_metrics.metadata(),
            "node_metrics": tuple(value.metadata() for value in self.node_metrics),
            "descriptive_native_vjp_vs_V12_stage_metrics": tuple(
                value.metadata() for value in self.stage_metrics
            ),
            "telescope_metrics": self.telescope_metrics.metadata(),
            "replayed_v12_comparison_artifact_sha256": (
                self.replayed_v12_comparison_artifact_sha256
            ),
            "replayed_v11_comparison_artifact_sha256": (
                self.replayed_v11_comparison_artifact_sha256
            ),
            "V12_and_V11_authorities_replayed_exactly": True,
            "nodewise_gate_results": tuple(sorted(self.nodewise_gate_results.items())),
            "integrated_gate_results": tuple(
                sorted(self.integrated_gate_results.items())
            ),
            "nodewise_passed": self.nodewise_passed,
            "integrated_passed": self.integrated_passed,
            "classification": self.classification,
            "nodewise_weighting": _WEIGHTING,
            "integrated_weighting": _INTEGRATED_WEIGHTING,
            "adjoint_gate": (
                "overall_and_every_family_symmetric_relative_RMSE_at_most_0_0001"
            ),
            "adjoint_relative_RMSE_maximum": ADJOINT_RELATIVE_RMSE_MAXIMUM,
            "fixed_telescope_reconstructs_N64_native_vjp_minus_P_v10": True,
            "all_V12_stage_comparisons_published_without_selection": True,
            "duplicate_and_cross_family_ownership_rejected": True,
            "same_suffix_hypothesis_test_only": True,
            "fits_corrects_searches_or_selects_candidates": False,
            "authorizes_serving_compression_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        for family in self.family_summaries:
            family.validate_integrity()
        if (
            _sha256(_SUMMARY_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256,
                label="native VJP comparison artifact",
            )
        ):
            raise RuntimeError("candidate joint-state native VJP comparison drifted")


def summarize_candidate_joint_state_suffix_native_vjp(
    evidence: Iterable[CandidateJointStateSuffixNativeVJPEvidence],
) -> CandidateJointStateSuffixNativeVJPComparison:
    """Build immutable V13-B nodewise and integrated adjoint summaries."""

    values = _canonical_evidence(evidence)
    v12 = summarize_candidate_joint_state_contraction_precision(
        tuple(value.contraction_precision_evidence for value in values)
    )
    v11 = summarize_candidate_joint_state_suffix_jvp(
        tuple(
            value.contraction_precision_evidence.suffix_jvp_evidence
            for value in values
        )
    )
    if v12.replayed_v11_comparison_artifact_sha256 != v11.artifact_sha256:
        raise RuntimeError("V13-B exact V12 authority does not bind replayed V11")

    by_family: dict[str, list[CandidateJointStateSuffixNativeVJPEvidence]] = (
        defaultdict(list)
    )
    for value in values:
        by_family[value.family_id].append(value)
    families: list[CandidateJointStateSuffixNativeVJPFamilySummary] = []
    for family_id in sorted(by_family):
        members = tuple(
            sorted(by_family[family_id], key=lambda value: value.example_id)
        )
        families.append(
            CandidateJointStateSuffixNativeVJPFamilySummary(
                family_id=family_id,
                example_ids=tuple(value.example_id for value in members),
                evidence_artifact_sha256s=tuple(
                    value.artifact_sha256 for value in members
                ),
                supervised_token_count=sum(
                    value.supervised_tokens for value in members
                ),
                nodewise_metrics=_aggregate_nodewise_metrics(members),
                integrated_metrics=_integrated_metrics(members),
                node_metrics=_node_metrics(members),
                stage_metrics=_stage_metrics(members),
                telescope_metrics=_telescope_metrics(members),
            )
        )
    return CandidateJointStateSuffixNativeVJPComparison(
        evidence_example_ids=tuple(value.example_id for value in values),
        evidence_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        family_summaries=tuple(families),
        supervised_token_count=sum(value.supervised_tokens for value in values),
        nodewise_metrics=_aggregate_nodewise_metrics(values),
        integrated_metrics=_integrated_metrics(values),
        node_metrics=_node_metrics(values),
        stage_metrics=_stage_metrics(values),
        telescope_metrics=_telescope_metrics(values),
        replayed_v12_comparison_artifact_sha256=v12.artifact_sha256,
        replayed_v11_comparison_artifact_sha256=v11.artifact_sha256,
    )
