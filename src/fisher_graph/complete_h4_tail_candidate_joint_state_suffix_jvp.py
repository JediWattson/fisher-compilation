"""Pure V11 suffix-JVP audit for the candidate joint-state H4 path.

V10 measures the GL4 path derivative with reverse-mode contractions.  V11
replays the same four nodes with a forward-mode directional derivative through
the post-H4 suffix.  This module binds those tokenwise JVPs to one authenticated
V10 evidence object, integrates them with the exact V9/V10 GL4 rule, and
compares three independently labelled quantities:

``J64``
    The GL4 integral of the supplied suffix JVP directional derivatives.
``P64``
    The wrapped V10 reverse-mode path integral, replayed without alteration.
``D64``
    The wrapped V10 direct float64 finite endpoint delta.

Statistics use token means inside prompts, equal prompt means inside families,
and equal family means.  Prompt token counts and H4 row counts may vary.  Raw
tensors stay in typed evidence and never enter serialized metadata.
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

from .complete_h4_tail_candidate_joint_state_objective_precision import (
    CLOSURE_COSINE_MINIMUM,
    FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM,
    OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateObjectivePrecisionEvidence,
    summarize_candidate_joint_state_objective_precision,
)
from .complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


__all__ = [
    "ADJOINT_RELATIVE_RMSE_MAXIMUM",
    "CandidateJointStateSuffixJVPComparison",
    "CandidateJointStateSuffixJVPEvidence",
    "CandidateJointStateSuffixJVPFamilySummary",
    "CandidateJointStateSuffixJVPMetrics",
    "CandidateJointStateSuffixJVPNodeEvidence",
    "classify_candidate_joint_state_suffix_jvp",
    "summarize_candidate_joint_state_suffix_jvp",
]


ADJOINT_RELATIVE_RMSE_MAXIMUM = 1.0e-4

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-jvp-tensor:v11\0"
_NODE_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-jvp-node:v11\0"
_EVIDENCE_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-jvp-evidence:v11\0"
_FAMILY_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-jvp-family:v11\0"
_SUMMARY_DOMAIN = b"fisher-graph:candidate-joint-state-suffix-jvp-summary:v11\0"
_WEIGHTING = (
    "mean_tokens_within_prompt_then_equal_prompts_within_family_then_"
    "equal_families"
)
_EPSILON = 64.0 * torch.finfo(torch.float64).eps


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


def _float64_token_vector(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.numel() <= 0
        or value.dtype != torch.float64
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty float64 token vector")
    return value.detach().to(device="cpu").clone().contiguous()


def _same_float(left: float, right: float) -> bool:
    return float(left).hex() == float(right).hex()


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixJVPNodeEvidence:
    """One ordered GL4 suffix-JVP vector and its provenance authorities.

    ``primal_token_teacher_kl_sha256`` and ``path_h4_sha256`` deliberately use
    the wrapped V10/V9 core receipt domains.  A diagnostic first checks its
    runtime tensors bitwise against those live values; its separately supplied
    ``suffix_runtime_receipt_sha256`` then transitively binds runtime-domain
    hashes and the suffix execution provenance.
    """

    node_index: int
    path_fraction: float
    quadrature_weight: float
    token_directional_derivative_f64: Tensor = field(repr=False)
    pinned_v10_node_receipt_artifact_sha256: str
    suffix_runtime_receipt_sha256: str
    primal_token_teacher_kl_sha256: str
    provider_artifact_sha256: str
    execution_artifact_sha256: str
    path_h4_sha256: str
    supervised_grid_sha256: str
    endpoint_pair_binding_sha256: str
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or not 0 <= self.node_index < 4:
            raise ValueError("suffix JVP GL4 node index must be in [0, 3]")
        node = _finite_float(self.path_fraction, label="suffix JVP path fraction")
        weight = _finite_float(
            self.quadrature_weight,
            label="suffix JVP quadrature weight",
            nonnegative=True,
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
        ):
            raise ValueError("suffix JVP node does not use the exact GL4 rule")
        vector = _float64_token_vector(
            self.token_directional_derivative_f64,
            label="suffix JVP directional derivative",
        )
        for name in (
            "pinned_v10_node_receipt_artifact_sha256",
            "suffix_runtime_receipt_sha256",
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
        object.__setattr__(self, "path_fraction", node)
        object.__setattr__(self, "quadrature_weight", weight)
        object.__setattr__(self, "token_directional_derivative_f64", vector)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def token_count(self) -> int:
        return int(self.token_directional_derivative_f64.shape[0])

    def directional_derivative_f64(self) -> Tensor:
        return self.token_directional_derivative_f64.clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        vector = self.directional_derivative_f64()
        result: dict[str, object] = {
            "node_index": self.node_index,
            "path_fraction_hex": self.path_fraction.hex(),
            "quadrature_weight_hex": self.quadrature_weight.hex(),
            "token_count": self.token_count,
            "token_directional_derivative_f64_sha256": _tensor_sha256(vector),
            "token_directional_derivative_mean": float(vector.mean()),
            "token_directional_derivative_rms": float(vector.square().mean().sqrt()),
            "token_directional_derivative_maximum_abs": float(vector.abs().max()),
            "pinned_v10_node_receipt_artifact_sha256": (
                self.pinned_v10_node_receipt_artifact_sha256
            ),
            "suffix_runtime_receipt_sha256": self.suffix_runtime_receipt_sha256,
            "primal_token_teacher_kl_sha256": (
                self.primal_token_teacher_kl_sha256
            ),
            "provider_artifact_sha256": self.provider_artifact_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "path_h4_sha256": self.path_h4_sha256,
            "supervised_grid_sha256": self.supervised_grid_sha256,
            "endpoint_pair_binding_sha256": self.endpoint_pair_binding_sha256,
            "raw_tensors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        value = self.token_directional_derivative_f64
        if (
            value.dtype != torch.float64
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.numel() <= 0
            or value.requires_grad
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
            or self.path_fraction.hex()
            != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or self.quadrature_weight.hex()
            != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
            or _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256, label="suffix JVP node artifact"
            )
        ):
            raise RuntimeError("candidate joint-state suffix JVP node drifted")


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixJVPEvidence:
    """One prompt's four suffix JVPs bound to exact V10 evidence."""

    precision_evidence: CandidateJointStateObjectivePrecisionEvidence = field(
        repr=False
    )
    nodes: tuple[CandidateJointStateSuffixJVPNodeEvidence, ...]
    example_id: str = field(init=False)
    family_id: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.precision_evidence, CandidateJointStateObjectivePrecisionEvidence
        ):
            raise TypeError("suffix JVP evidence requires V10 precision evidence")
        self.precision_evidence.validate_integrity()
        nodes = tuple(self.nodes)
        if (
            len(nodes) != 4
            or any(
                not isinstance(node, CandidateJointStateSuffixJVPNodeEvidence)
                or node.node_index != index
                for index, node in enumerate(nodes)
            )
        ):
            raise ValueError("suffix JVP evidence requires ordered four GL4 nodes")
        path = self.precision_evidence.path_evidence
        for node, receipt in zip(nodes, path.node_receipts, strict=True):
            node.validate_integrity()
            if node.token_count != self.precision_evidence.supervised_tokens:
                raise ValueError("suffix JVP token geometry differs from V10")
            if (
                node.path_fraction.hex() != receipt.path_fraction.hex()
                or node.quadrature_weight.hex() != receipt.quadrature_weight.hex()
                or node.pinned_v10_node_receipt_artifact_sha256
                != receipt.artifact_sha256
                or node.primal_token_teacher_kl_sha256
                != receipt.token_teacher_kl_sha256
                or node.provider_artifact_sha256
                != receipt.provider_artifact_sha256
                or node.execution_artifact_sha256
                != receipt.execution_artifact_sha256
                or node.path_h4_sha256 != receipt.path_node_h4_sha256
                or node.supervised_grid_sha256 != path.supervised_grid_sha256
                or node.endpoint_pair_binding_sha256
                != path.endpoint_pair_binding_sha256
            ):
                raise ValueError("suffix JVP node provenance differs from V10")
        suffix_receipts = tuple(node.suffix_runtime_receipt_sha256 for node in nodes)
        if len(set(suffix_receipts)) != 4:
            raise ValueError("suffix JVP runtime receipts must be node-distinct")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "example_id", self.precision_evidence.example_id)
        object.__setattr__(self, "family_id", self.precision_evidence.family_id)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return self.precision_evidence.supervised_tokens

    @property
    def h4_shape(self) -> tuple[int, int]:
        return self.precision_evidence.h4_shape

    def integrated_suffix_jvp_f64(self) -> Tensor:
        total = torch.zeros(self.supervised_tokens, dtype=torch.float64)
        for node in self.nodes:
            total.add_(node.directional_derivative_f64(), alpha=node.quadrature_weight)
        return total.contiguous()

    def replayed_vjp_integral_f64(self) -> Tensor:
        return self.precision_evidence.path_integral_f64()

    def finite_delta_f64(self) -> Tensor:
        return self.precision_evidence.finite_delta_f64()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        jvp = self.integrated_suffix_jvp_f64()
        vjp = self.replayed_vjp_integral_f64()
        finite = self.finite_delta_f64()
        residual = self.precision_evidence.direct_endpoint_crosscheck_residual()
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "supervised_token_count": self.supervised_tokens,
            "h4_support_row_count": self.h4_shape[0],
            "h4_width": self.h4_shape[1],
            "precision_evidence_artifact_sha256": (
                self.precision_evidence.artifact_sha256
            ),
            "path_evidence_artifact_sha256": (
                self.precision_evidence.path_evidence.artifact_sha256
            ),
            "nodes": tuple(node.metadata() for node in self.nodes),
            "integrated_suffix_jvp_f64_sha256": _tensor_sha256(jvp),
            "replayed_vjp_integral_f64_sha256": _tensor_sha256(vjp),
            "finite_delta_f64_sha256": _tensor_sha256(finite),
            "jvp_minus_vjp_f64_sha256": _tensor_sha256((jvp - vjp).contiguous()),
            "jvp_minus_finite_f64_sha256": _tensor_sha256(
                (jvp - finite).contiguous()
            ),
            "vjp_minus_finite_f64_sha256": _tensor_sha256(
                (vjp - finite).contiguous()
            ),
            "maximum_direct_endpoint_crosscheck_abs_error": float(
                residual.abs().max()
            ),
            "maximum_direct_endpoint_crosscheck_tolerance": (
                self.precision_evidence.direct_endpoint_crosscheck_tolerance
            ),
            "direct_endpoint_crosscheck_passed": (
                float(residual.abs().max())
                <= self.precision_evidence.direct_endpoint_crosscheck_tolerance
            ),
            "quadrature_rule": "gauss_legendre_order_4_on_unit_interval",
            "jvp_direction": (
                "complete_H4_scalar_to_joint_displacement_through_post_H4_suffix"
            ),
            "raw_tensors_serialized": False,
            "fits_selects_or_routes_candidates": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.precision_evidence.validate_integrity()
        path = self.precision_evidence.path_evidence
        try:
            if len(self.nodes) != 4:
                raise RuntimeError
            for index, (node, receipt) in enumerate(
                zip(self.nodes, path.node_receipts, strict=True)
            ):
                node.validate_integrity()
                if (
                    node.node_index != index
                    or node.token_count != self.supervised_tokens
                    or node.pinned_v10_node_receipt_artifact_sha256
                    != receipt.artifact_sha256
                    or node.primal_token_teacher_kl_sha256
                    != receipt.token_teacher_kl_sha256
                    or node.provider_artifact_sha256
                    != receipt.provider_artifact_sha256
                    or node.execution_artifact_sha256
                    != receipt.execution_artifact_sha256
                    or node.path_h4_sha256 != receipt.path_node_h4_sha256
                    or node.supervised_grid_sha256 != path.supervised_grid_sha256
                    or node.endpoint_pair_binding_sha256
                    != path.endpoint_pair_binding_sha256
                ):
                    raise RuntimeError
            if (
                len({node.suffix_runtime_receipt_sha256 for node in self.nodes})
                != 4
                or self.example_id != self.precision_evidence.example_id
                or self.family_id != self.precision_evidence.family_id
                or _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False))
                != _require_sha256(
                    self.artifact_sha256, label="suffix JVP evidence artifact"
                )
            ):
                raise RuntimeError
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "candidate joint-state suffix JVP evidence drifted"
            ) from error


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixJVPMetrics:
    mean_suffix_jvp_f64: float
    mean_vjp_path_integral_f64: float
    mean_finite_delta_f64: float
    mean_jvp_minus_vjp_f64: float
    mean_jvp_minus_finite_f64: float
    mean_vjp_minus_finite_f64: float
    suffix_jvp_f64_rms: float
    vjp_path_integral_f64_rms: float
    finite_delta_f64_rms: float
    adjoint_rmse: float
    adjoint_relative_rmse: float
    adjoint_cosine: float
    maximum_absolute_adjoint_error: float
    jvp_closure_rmse: float
    jvp_closure_relative_rmse: float
    jvp_closure_cosine: float
    maximum_absolute_jvp_closure_error: float
    vjp_closure_rmse: float
    vjp_closure_relative_rmse: float
    vjp_closure_cosine: float
    maximum_absolute_vjp_closure_error: float
    relative_rmse_epsilon: float

    def __post_init__(self) -> None:
        signed = {
            "mean_suffix_jvp_f64",
            "mean_vjp_path_integral_f64",
            "mean_finite_delta_f64",
            "mean_jvp_minus_vjp_f64",
            "mean_jvp_minus_finite_f64",
            "mean_vjp_minus_finite_f64",
            "adjoint_cosine",
            "jvp_closure_cosine",
            "vjp_closure_cosine",
        }
        cosines = {
            "adjoint_cosine",
            "jvp_closure_cosine",
            "vjp_closure_cosine",
        }
        for name in self.__dataclass_fields__:
            value = _finite_float(
                getattr(self, name),
                label=f"suffix JVP metric {name}",
                nonnegative=name not in signed,
            )
            if name in cosines and not -1.0 <= value <= 1.0:
                raise ValueError(f"suffix JVP {name} must be in [-1, 1]")
            object.__setattr__(self, name, value)

    def metadata(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        }


def _direction(value: float) -> str:
    if abs(value) <= _EPSILON:
        return "zero"
    return "positive" if value > 0.0 else "negative"


def _directions(metrics: CandidateJointStateSuffixJVPMetrics) -> dict[str, str]:
    return {
        "suffix_jvp": _direction(metrics.mean_suffix_jvp_f64),
        "vjp_path_integral": _direction(metrics.mean_vjp_path_integral_f64),
        "finite_delta": _direction(metrics.mean_finite_delta_f64),
        "jvp_minus_vjp": _direction(metrics.mean_jvp_minus_vjp_f64),
        "jvp_minus_finite": _direction(metrics.mean_jvp_minus_finite_f64),
        "vjp_minus_finite": _direction(metrics.mean_vjp_minus_finite_f64),
    }


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixJVPFamilySummary:
    family_id: str
    example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    supervised_token_count: int
    maximum_direct_endpoint_crosscheck_abs_error: float
    maximum_direct_endpoint_crosscheck_tolerance: float
    metrics: CandidateJointStateSuffixJVPMetrics
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        family = _identifier(self.family_id, label="suffix JVP family_id")
        examples = tuple(
            _identifier(value, label="suffix JVP family example_id")
            for value in self.example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="suffix JVP evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        error = _finite_float(
            self.maximum_direct_endpoint_crosscheck_abs_error,
            label="family direct endpoint cross-check error",
            nonnegative=True,
        )
        tolerance = _finite_float(
            self.maximum_direct_endpoint_crosscheck_tolerance,
            label="family direct endpoint cross-check tolerance",
            nonnegative=True,
        )
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or error > tolerance
            or not isinstance(self.metrics, CandidateJointStateSuffixJVPMetrics)
        ):
            raise ValueError("suffix JVP family membership is invalid")
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(
            self, "maximum_direct_endpoint_crosscheck_abs_error", error
        )
        object.__setattr__(
            self, "maximum_direct_endpoint_crosscheck_tolerance", tolerance
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FAMILY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def directions(self) -> dict[str, str]:
        return _directions(self.metrics)

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "family_id": self.family_id,
            "example_ids": self.example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "prompt_count": len(self.example_ids),
            "supervised_token_count": self.supervised_token_count,
            "maximum_direct_endpoint_crosscheck_abs_error": (
                self.maximum_direct_endpoint_crosscheck_abs_error
            ),
            "maximum_direct_endpoint_crosscheck_tolerance": (
                self.maximum_direct_endpoint_crosscheck_tolerance
            ),
            "direct_endpoint_crosscheck_passed": True,
            "directions": tuple(sorted(self.directions.items())),
            **self.metrics.metadata(),
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _FAMILY_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(
            self.artifact_sha256, label="suffix JVP family artifact"
        ):
            raise RuntimeError("candidate joint-state suffix JVP family drifted")


def classify_candidate_joint_state_suffix_jvp(
    *,
    adjoint_passed: bool,
    jvp_closure_passed: bool,
    vjp_closure_passed: bool,
) -> str:
    """Classify what the three-way JVP/VJP/finite comparison establishes."""

    if any(
        type(value) is not bool
        for value in (adjoint_passed, jvp_closure_passed, vjp_closure_passed)
    ):
        raise TypeError("suffix JVP classification gates must be boolean")
    # JVP-only finite closure is the one asymmetric outcome with a direct
    # authority: forward-mode reaches D64 while the wrapped reverse contraction
    # does not.  It is therefore reported before the general adjoint-ambiguity
    # branch (and will ordinarily be accompanied by an adjoint-gate failure).
    if jvp_closure_passed and not vjp_closure_passed:
        return "suffix_jvp_only_closure_reverse_contraction_failure_same_a"
    if not adjoint_passed:
        return "suffix_adjoint_ambiguity_same_a"
    if jvp_closure_passed and vjp_closure_passed:
        return "suffix_adjoint_passed_both_closures_established_same_a"
    if not jvp_closure_passed and not vjp_closure_passed:
        # Adjoint agreement localizes the discrepancy beyond the materialized
        # reverse contraction, but it does not by itself distinguish cast
        # discontinuities from unsampled curvature/quadrature remainder.
        return "suffix_adjoint_passed_both_closures_miss_finite_path_remainder_same_a"
    return "suffix_adjoint_ambiguity_same_a"


@dataclass(frozen=True, slots=True)
class CandidateJointStateSuffixJVPComparison:
    evidence_example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    family_summaries: tuple[CandidateJointStateSuffixJVPFamilySummary, ...]
    supervised_token_count: int
    maximum_direct_endpoint_crosscheck_abs_error: float
    maximum_direct_endpoint_crosscheck_tolerance: float
    replayed_v10_comparison_artifact_sha256: str
    metrics: CandidateJointStateSuffixJVPMetrics
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        examples = tuple(
            _identifier(value, label="suffix JVP example_id")
            for value in self.evidence_example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="suffix JVP evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        families = tuple(self.family_summaries)
        error = _finite_float(
            self.maximum_direct_endpoint_crosscheck_abs_error,
            label="comparison direct endpoint cross-check error",
            nonnegative=True,
        )
        tolerance = _finite_float(
            self.maximum_direct_endpoint_crosscheck_tolerance,
            label="comparison direct endpoint cross-check tolerance",
            nonnegative=True,
        )
        replay = _require_sha256(
            self.replayed_v10_comparison_artifact_sha256,
            label="replayed V10 comparison artifact",
        )
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or not families
            or any(
                not isinstance(value, CandidateJointStateSuffixJVPFamilySummary)
                for value in families
            )
            or tuple(value.family_id for value in families)
            != tuple(sorted({value.family_id for value in families}))
            or set(examples)
            != {example for family in families for example in family.example_ids}
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or self.supervised_token_count
            != sum(family.supervised_token_count for family in families)
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
            or error > tolerance
            or not isinstance(self.metrics, CandidateJointStateSuffixJVPMetrics)
        ):
            raise ValueError("suffix JVP comparison membership is invalid")
        for family in families:
            family.validate_integrity()
        object.__setattr__(self, "evidence_example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "family_summaries", families)
        object.__setattr__(self, "maximum_direct_endpoint_crosscheck_abs_error", error)
        object.__setattr__(
            self, "maximum_direct_endpoint_crosscheck_tolerance", tolerance
        )
        object.__setattr__(
            self, "replayed_v10_comparison_artifact_sha256", replay
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SUMMARY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def directions(self) -> dict[str, str]:
        return _directions(self.metrics)

    @property
    def adjoint_gate_results(self) -> dict[str, bool]:
        return {
            "overall_adjoint_relative_RMSE_at_most_0_0001": (
                self.metrics.adjoint_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
            ),
            "every_family_adjoint_relative_RMSE_at_most_0_0001": all(
                family.metrics.adjoint_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
                for family in self.family_summaries
            ),
        }

    @property
    def adjoint_passed(self) -> bool:
        return all(self.adjoint_gate_results.values())

    def _closure_gate_results(self, *, jvp: bool) -> dict[str, bool]:
        prefix = "jvp" if jvp else "vjp"
        relative_name = (
            "jvp_closure_relative_rmse" if jvp else "vjp_closure_relative_rmse"
        )
        cosine_name = "jvp_closure_cosine" if jvp else "vjp_closure_cosine"
        return {
            f"overall_{prefix}_closure_relative_RMSE_at_most_0_05": (
                getattr(self.metrics, relative_name)
                <= OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
            ),
            f"every_family_{prefix}_closure_relative_RMSE_at_most_0_10": all(
                getattr(family.metrics, relative_name)
                <= FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                for family in self.family_summaries
            ),
            f"overall_{prefix}_closure_cosine_at_least_0_99": (
                getattr(self.metrics, cosine_name) >= CLOSURE_COSINE_MINIMUM
            ),
        }

    @property
    def jvp_closure_gate_results(self) -> dict[str, bool]:
        return self._closure_gate_results(jvp=True)

    @property
    def vjp_closure_gate_results(self) -> dict[str, bool]:
        return self._closure_gate_results(jvp=False)

    @property
    def jvp_closure_passed(self) -> bool:
        return all(self.jvp_closure_gate_results.values())

    @property
    def vjp_closure_passed(self) -> bool:
        return all(self.vjp_closure_gate_results.values())

    @property
    def classification(self) -> str:
        return classify_candidate_joint_state_suffix_jvp(
            adjoint_passed=self.adjoint_passed,
            jvp_closure_passed=self.jvp_closure_passed,
            vjp_closure_passed=self.vjp_closure_passed,
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
            "maximum_direct_endpoint_crosscheck_abs_error": (
                self.maximum_direct_endpoint_crosscheck_abs_error
            ),
            "maximum_direct_endpoint_crosscheck_tolerance": (
                self.maximum_direct_endpoint_crosscheck_tolerance
            ),
            "direct_endpoint_crosscheck_passed": True,
            "replayed_v10_comparison_artifact_sha256": (
                self.replayed_v10_comparison_artifact_sha256
            ),
            "v10_vjp_closure_replayed_exactly": True,
            **self.metrics.metadata(),
            "directions": tuple(sorted(self.directions.items())),
            "adjoint_gate_results": tuple(sorted(self.adjoint_gate_results.items())),
            "jvp_closure_gate_results": tuple(
                sorted(self.jvp_closure_gate_results.items())
            ),
            "vjp_closure_gate_results": tuple(
                sorted(self.vjp_closure_gate_results.items())
            ),
            "adjoint_passed": self.adjoint_passed,
            "jvp_closure_passed": self.jvp_closure_passed,
            "vjp_closure_passed": self.vjp_closure_passed,
            "classification": self.classification,
            "weighting": _WEIGHTING,
            "adjoint_orientation": "J64_suffix_GL4_minus_P64_V10_VJP_GL4",
            "jvp_closure_orientation": "J64_suffix_GL4_minus_D64_direct_finite",
            "vjp_closure_orientation": "P64_V10_VJP_GL4_minus_D64_direct_finite",
            "adjoint_relative_RMSE_denominator": (
                "max(J64_RMS,P64_RMS,numerical_epsilon)"
            ),
            "thresholds": {
                "adjoint_relative_RMSE_maximum": ADJOINT_RELATIVE_RMSE_MAXIMUM,
                "overall_closure_relative_RMSE_maximum": (
                    OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "every_family_closure_relative_RMSE_maximum": (
                    FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "overall_closure_cosine_minimum": CLOSURE_COSINE_MINIMUM,
            },
            "variable_prompt_token_and_support_row_counts_allowed": True,
            "raw_tensors_serialized": False,
            "same_A_hypothesis_use_only": True,
            "authorizes_serving_compression_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        for family in self.family_summaries:
            family.validate_integrity()
        if _sha256(
            _SUMMARY_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(
            self.artifact_sha256, label="suffix JVP comparison artifact"
        ):
            raise RuntimeError("candidate joint-state suffix JVP comparison drifted")


def _canonical_evidence(
    evidence: Iterable[CandidateJointStateSuffixJVPEvidence],
) -> tuple[CandidateJointStateSuffixJVPEvidence, ...]:
    values = tuple(evidence)
    if not values or any(
        not isinstance(value, CandidateJointStateSuffixJVPEvidence)
        for value in values
    ):
        raise TypeError("suffix JVP comparison requires typed evidence")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("suffix JVP example ids must be unique")
    return ordered


def _nested_mean(
    evidence: Sequence[CandidateJointStateSuffixJVPEvidence],
    token_values: Mapping[str, Tensor],
) -> Tensor:
    if set(token_values) != {value.example_id for value in evidence}:
        raise ValueError("suffix JVP statistic keys differ from evidence")
    by_family: dict[str, list[Tensor]] = defaultdict(list)
    trailing_shape: tuple[int, ...] | None = None
    for value in evidence:
        statistic = token_values[value.example_id]
        if (
            not isinstance(statistic, Tensor)
            or statistic.ndim < 1
            or statistic.shape[0] != value.supervised_tokens
            or not statistic.is_floating_point()
            or not bool(torch.isfinite(statistic).all())
        ):
            raise ValueError("suffix JVP statistic must be token aligned")
        statistic64 = statistic.detach().to(device="cpu", dtype=torch.float64)
        shape = tuple(int(size) for size in statistic64.shape[1:])
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("suffix JVP statistic shapes differ")
        by_family[value.family_id].append(statistic64.mean(dim=0))
    return torch.stack(
        tuple(
            torch.stack(tuple(by_family[family])).mean(dim=0)
            for family in sorted(by_family)
        )
    ).mean(dim=0).contiguous()


def _rms(second_moment: float) -> float:
    return math.sqrt(max(second_moment, 0.0))


def _relative_rmse(rmse: float, target_rms: float) -> float:
    if target_rms <= _EPSILON and rmse <= _EPSILON:
        return 0.0
    return rmse / max(target_rms, _EPSILON)


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


def _metrics(
    evidence: Sequence[CandidateJointStateSuffixJVPEvidence],
) -> CandidateJointStateSuffixJVPMetrics:
    jvp = {value.example_id: value.integrated_suffix_jvp_f64() for value in evidence}
    vjp = {value.example_id: value.replayed_vjp_integral_f64() for value in evidence}
    finite = {value.example_id: value.finite_delta_f64() for value in evidence}
    adjoint = {key: jvp[key] - vjp[key] for key in jvp}
    jclosure = {key: jvp[key] - finite[key] for key in jvp}
    vclosure = {key: vjp[key] - finite[key] for key in jvp}

    def mean(values: Mapping[str, Tensor]) -> float:
        return float(_nested_mean(evidence, values))

    def second(values: Mapping[str, Tensor]) -> float:
        return mean({key: value.square() for key, value in values.items()})

    jvp_rms = _rms(second(jvp))
    vjp_rms = _rms(second(vjp))
    finite_rms = _rms(second(finite))
    adjoint_rmse = _rms(second(adjoint))
    jclosure_rmse = _rms(second(jclosure))
    vclosure_rmse = _rms(second(vclosure))
    return CandidateJointStateSuffixJVPMetrics(
        mean_suffix_jvp_f64=mean(jvp),
        mean_vjp_path_integral_f64=mean(vjp),
        mean_finite_delta_f64=mean(finite),
        mean_jvp_minus_vjp_f64=mean(adjoint),
        mean_jvp_minus_finite_f64=mean(jclosure),
        mean_vjp_minus_finite_f64=mean(vclosure),
        suffix_jvp_f64_rms=jvp_rms,
        vjp_path_integral_f64_rms=vjp_rms,
        finite_delta_f64_rms=finite_rms,
        adjoint_rmse=adjoint_rmse,
        adjoint_relative_rmse=_symmetric_relative_rmse(
            adjoint_rmse, jvp_rms, vjp_rms
        ),
        adjoint_cosine=_cosine(
            cross=mean({key: jvp[key] * vjp[key] for key in jvp}),
            left_rms=jvp_rms,
            right_rms=vjp_rms,
        ),
        maximum_absolute_adjoint_error=max(
            float(value.abs().max()) for value in adjoint.values()
        ),
        jvp_closure_rmse=jclosure_rmse,
        jvp_closure_relative_rmse=_relative_rmse(jclosure_rmse, finite_rms),
        jvp_closure_cosine=_cosine(
            cross=mean({key: jvp[key] * finite[key] for key in jvp}),
            left_rms=jvp_rms,
            right_rms=finite_rms,
        ),
        maximum_absolute_jvp_closure_error=max(
            float(value.abs().max()) for value in jclosure.values()
        ),
        vjp_closure_rmse=vclosure_rmse,
        vjp_closure_relative_rmse=_relative_rmse(vclosure_rmse, finite_rms),
        vjp_closure_cosine=_cosine(
            cross=mean({key: vjp[key] * finite[key] for key in jvp}),
            left_rms=vjp_rms,
            right_rms=finite_rms,
        ),
        maximum_absolute_vjp_closure_error=max(
            float(value.abs().max()) for value in vclosure.values()
        ),
        relative_rmse_epsilon=_EPSILON,
    )


def _assert_v10_replay_exact(
    *,
    metrics: CandidateJointStateSuffixJVPMetrics,
    v10_metrics: object,
) -> None:
    pairs = (
        (metrics.mean_finite_delta_f64, getattr(v10_metrics, "mean_finite_delta_f64")),
        (
            metrics.mean_vjp_path_integral_f64,
            getattr(v10_metrics, "mean_path_integral_f64"),
        ),
        (
            metrics.mean_vjp_minus_finite_f64,
            getattr(v10_metrics, "mean_path_minus_finite_f64"),
        ),
        (metrics.finite_delta_f64_rms, getattr(v10_metrics, "finite_delta_f64_rms")),
        (
            metrics.vjp_path_integral_f64_rms,
            getattr(v10_metrics, "path_integral_f64_rms"),
        ),
        (metrics.vjp_closure_rmse, getattr(v10_metrics, "closure_rmse")),
        (
            metrics.vjp_closure_relative_rmse,
            getattr(v10_metrics, "closure_relative_rmse"),
        ),
        (metrics.vjp_closure_cosine, getattr(v10_metrics, "closure_cosine")),
        (
            metrics.maximum_absolute_vjp_closure_error,
            getattr(v10_metrics, "maximum_absolute_closure_error"),
        ),
        (metrics.relative_rmse_epsilon, getattr(v10_metrics, "relative_rmse_epsilon")),
    )
    if any(not _same_float(left, right) for left, right in pairs):
        raise RuntimeError("suffix JVP summary did not exactly replay V10 VJP closure")


def summarize_candidate_joint_state_suffix_jvp(
    evidence: Iterable[CandidateJointStateSuffixJVPEvidence],
) -> CandidateJointStateSuffixJVPComparison:
    """Build immutable family-equal V11 adjoint and closure summaries."""

    values = _canonical_evidence(evidence)
    v10 = summarize_candidate_joint_state_objective_precision(
        tuple(value.precision_evidence for value in values)
    )
    metrics = _metrics(values)
    _assert_v10_replay_exact(metrics=metrics, v10_metrics=v10.metrics)
    by_family: dict[str, list[CandidateJointStateSuffixJVPEvidence]] = defaultdict(list)
    for value in values:
        by_family[value.family_id].append(value)
    v10_families = {family.family_id: family for family in v10.family_summaries}
    families: list[CandidateJointStateSuffixJVPFamilySummary] = []
    for family_id in sorted(by_family):
        members = tuple(sorted(by_family[family_id], key=lambda value: value.example_id))
        family_metrics = _metrics(members)
        v10_family = v10_families[family_id]
        _assert_v10_replay_exact(
            metrics=family_metrics,
            v10_metrics=v10_family.metrics,
        )
        family = CandidateJointStateSuffixJVPFamilySummary(
            family_id=family_id,
            example_ids=tuple(value.example_id for value in members),
            evidence_artifact_sha256s=tuple(
                value.artifact_sha256 for value in members
            ),
            supervised_token_count=sum(value.supervised_tokens for value in members),
            maximum_direct_endpoint_crosscheck_abs_error=max(
                float(
                    value.precision_evidence.direct_endpoint_crosscheck_residual()
                    .abs()
                    .max()
                )
                for value in members
            ),
            maximum_direct_endpoint_crosscheck_tolerance=max(
                value.precision_evidence.direct_endpoint_crosscheck_tolerance
                for value in members
            ),
            metrics=family_metrics,
        )
        if (
            not _same_float(
                family.maximum_direct_endpoint_crosscheck_abs_error,
                v10_family.maximum_direct_endpoint_crosscheck_abs_error,
            )
            or not _same_float(
                family.maximum_direct_endpoint_crosscheck_tolerance,
                v10_family.maximum_direct_endpoint_crosscheck_tolerance,
            )
        ):
            raise RuntimeError("suffix JVP family did not replay V10 endpoint cross-check")
        families.append(family)
    comparison = CandidateJointStateSuffixJVPComparison(
        evidence_example_ids=tuple(value.example_id for value in values),
        evidence_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        family_summaries=tuple(families),
        supervised_token_count=sum(value.supervised_tokens for value in values),
        maximum_direct_endpoint_crosscheck_abs_error=max(
            float(
                value.precision_evidence.direct_endpoint_crosscheck_residual()
                .abs()
                .max()
            )
            for value in values
        ),
        maximum_direct_endpoint_crosscheck_tolerance=max(
            value.precision_evidence.direct_endpoint_crosscheck_tolerance
            for value in values
        ),
        replayed_v10_comparison_artifact_sha256=v10.artifact_sha256,
        metrics=metrics,
    )
    if (
        comparison.vjp_closure_passed != v10.closure_passed
        or not _same_float(
            comparison.maximum_direct_endpoint_crosscheck_abs_error,
            v10.maximum_direct_endpoint_crosscheck_abs_error,
        )
        or not _same_float(
            comparison.maximum_direct_endpoint_crosscheck_tolerance,
            v10.maximum_direct_endpoint_crosscheck_tolerance,
        )
    ):
        raise RuntimeError("suffix JVP comparison did not exactly replay V10")
    return comparison
