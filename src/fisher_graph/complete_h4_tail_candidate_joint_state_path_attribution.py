"""Pure GL4 attribution for the realized V6-scalar to V7-joint path.

This module does not execute a model, fit a gain field, select an arm, or
authorize serving.  It represents one deliberately narrow finite experiment:

``H(alpha) = H_scalar_actual + alpha * (H_joint_actual - H_scalar_actual)``.

The two endpoints are the H4 tensors *after* the runtime's cast-once update.
Four strictly interior Gauss--Legendre nodes supply teacher-KL gradients.  The
core streams those gradients into one weighted integral, binds every transient
evaluation to hash-only receipts, and compares the resulting contraction with
``KL_joint - KL_scalar``.  An endpoint gradient at the scalar arm is optional
and, when present, is reported only as a separately labelled first-order
tangent; it is never substituted for the GL4 integral.

All aggregation is token-then-prompt-then-family equal.  Raw node-gradient
banks are not retained.  The retained endpoint tensors and integrated gradient
are defensive CPU copies protected by content-addressed metadata.
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

from .complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


__all__ = [
    "CandidateJointStatePathAccumulator",
    "CandidateJointStatePathAttribution",
    "CandidateJointStatePathEvidence",
    "CandidateJointStatePathFamilyAttribution",
    "CandidateJointStatePathNodeReceipt",
    "CandidateJointStateEndpointTangentReceipt",
    "candidate_joint_state_finite_kl_delta",
    "candidate_joint_state_path_displacement",
    "candidate_joint_state_path_integrated_contraction",
    "candidate_joint_state_held_unit_endpoint_tangent_contraction",
    "candidate_joint_state_scalar_endpoint_tangent_contraction",
    "summarize_candidate_joint_state_path_attribution",
]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NODE_DOMAIN = b"fisher-graph:candidate-joint-state-path-node:v9\0"
_TANGENT_DOMAIN = b"fisher-graph:candidate-joint-state-endpoint-tangent:v9\0"
_EVIDENCE_DOMAIN = b"fisher-graph:candidate-joint-state-path-evidence:v9\0"
_SUMMARY_DOMAIN = b"fisher-graph:candidate-joint-state-path-summary:v9\0"
_TENSOR_DOMAIN = b"fisher-graph:candidate-joint-state-path-tensor:v9\0"
_PATH_GEOMETRY = (
    "continuous_straight_complete_H4_scalar_to_joint_path_"
    "sampled_with_one_endpoint_dtype_cast_per_node"
)
_PATH_OBJECTIVE = "teacher_KL_along_scalar_to_joint_candidate_path"
_FTC_ORIENTATION = "joint_KL_minus_scalar_KL_compared_with_GL4_path_integral"
_WEIGHTING = (
    "mean_tokens_within_prompt_then_equal_prompts_within_family_"
    "then_equal_families"
)


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


def _float64(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or 0 in value.shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()


def _realized_endpoint(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 2
        or 0 in value.shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite nonempty floating H4 rows")
    # Preserve the supplied runtime dtype: the point of this evidence is to
    # bind the values that survived the runtime's cast, not a pre-cast ideal.
    return value.detach().to(device="cpu").clone().contiguous()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise TypeError("hashed value must be a strided tensor")
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


def _shape3(value: object, *, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(type(size) is not int or size <= 0 for size in value)
    ):
        raise ValueError(f"{label} must be a positive rank-three shape")
    return value


@dataclass(frozen=True, slots=True)
class CandidateJointStatePathNodeReceipt:
    """Hash/scalar receipt for one transient scalar-to-joint GL4 node."""

    node_index: int
    path_fraction: float
    quadrature_weight: float
    token_count: int
    h4_gradient_shape: tuple[int, int, int]
    path_node_h4_shape: tuple[int, int]
    path_node_h4_dtype: str
    path_node_h4_sha256: str
    token_teacher_kl_sha256: str
    token_teacher_kl_mean: float
    token_teacher_kl_minimum: float
    token_teacher_kl_maximum: float
    h4_gradient_sha256: str
    h4_gradient_frobenius: float
    integrated_gradient_sha256_before: str
    integrated_gradient_sha256_after: str
    vjp_artifact_sha256: str
    provider_artifact_sha256: str
    execution_artifact_sha256: str
    maximum_future_gradient_abs: float
    future_gradient_nonzero_count: int
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or not 0 <= self.node_index < 4:
            raise ValueError("GL4 node index must be in [0, 3]")
        node = _finite_float(self.path_fraction, label="path fraction")
        weight = _finite_float(
            self.quadrature_weight,
            label="quadrature weight",
            nonnegative=True,
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
            or not 0.0 < node < 1.0
            or weight <= 0.0
        ):
            raise ValueError("node receipt does not use the exact GL4 rule")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("node token count must be positive")
        shape = _shape3(self.h4_gradient_shape, label="node gradient shape")
        if shape[0] != self.token_count:
            raise ValueError("node token count and gradient shape differ")
        if (
            not isinstance(self.path_node_h4_shape, tuple)
            or len(self.path_node_h4_shape) != 2
            or any(
                type(size) is not int or size <= 0
                for size in self.path_node_h4_shape
            )
            or shape[1:] != self.path_node_h4_shape
            or not isinstance(self.path_node_h4_dtype, str)
            or not (
                self.path_node_h4_dtype.startswith("torch.float")
                or self.path_node_h4_dtype == "torch.bfloat16"
            )
        ):
            raise ValueError("path node H4 geometry is invalid")
        mean = _finite_float(self.token_teacher_kl_mean, label="node token KL mean")
        minimum = _finite_float(
            self.token_teacher_kl_minimum, label="node token KL minimum"
        )
        maximum = _finite_float(
            self.token_teacher_kl_maximum, label="node token KL maximum"
        )
        tolerance = 64.0 * torch.finfo(torch.float64).eps * max(
            abs(mean), abs(minimum), abs(maximum), 1.0
        )
        if mean < minimum - tolerance or mean > maximum + tolerance:
            raise ValueError("node token KL summary is inconsistent")
        norm = _finite_float(
            self.h4_gradient_frobenius,
            label="node gradient Frobenius norm",
            nonnegative=True,
        )
        future_maximum = _finite_float(
            self.maximum_future_gradient_abs,
            label="maximum future gradient",
            nonnegative=True,
        )
        if (
            type(self.future_gradient_nonzero_count) is not int
            or self.future_gradient_nonzero_count < 0
            or (self.future_gradient_nonzero_count == 0) != (future_maximum == 0.0)
        ):
            raise ValueError("node future-gradient scalars are inconsistent")
        for value, label in (
            (self.token_teacher_kl_sha256, "node token KL"),
            (self.path_node_h4_sha256, "path node H4"),
            (self.h4_gradient_sha256, "node gradient"),
            (self.integrated_gradient_sha256_before, "prior integrated gradient"),
            (self.integrated_gradient_sha256_after, "updated integrated gradient"),
            (self.vjp_artifact_sha256, "node VJP artifact"),
            (self.provider_artifact_sha256, "node provider artifact"),
            (self.execution_artifact_sha256, "node execution artifact"),
        ):
            _require_sha256(value, label=label)
        object.__setattr__(self, "path_fraction", node)
        object.__setattr__(self, "quadrature_weight", weight)
        object.__setattr__(self, "h4_gradient_shape", shape)
        object.__setattr__(self, "token_teacher_kl_mean", mean)
        object.__setattr__(self, "token_teacher_kl_minimum", minimum)
        object.__setattr__(self, "token_teacher_kl_maximum", maximum)
        object.__setattr__(self, "h4_gradient_frobenius", norm)
        object.__setattr__(self, "maximum_future_gradient_abs", future_maximum)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False)),
        )

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "node_index": self.node_index,
            "path_fraction_hex": self.path_fraction.hex(),
            "quadrature_weight_hex": self.quadrature_weight.hex(),
            "token_count": self.token_count,
            "h4_gradient_shape": self.h4_gradient_shape,
            "path_node_h4_shape": self.path_node_h4_shape,
            "path_node_h4_dtype": self.path_node_h4_dtype,
            "path_node_h4_sha256": self.path_node_h4_sha256,
            "token_teacher_kl_sha256": self.token_teacher_kl_sha256,
            "token_teacher_kl_mean_hex": self.token_teacher_kl_mean.hex(),
            "token_teacher_kl_minimum_hex": self.token_teacher_kl_minimum.hex(),
            "token_teacher_kl_maximum_hex": self.token_teacher_kl_maximum.hex(),
            "h4_gradient_sha256": self.h4_gradient_sha256,
            "h4_gradient_frobenius_hex": self.h4_gradient_frobenius.hex(),
            "integrated_gradient_sha256_before": (
                self.integrated_gradient_sha256_before
            ),
            "integrated_gradient_sha256_after": (
                self.integrated_gradient_sha256_after
            ),
            "vjp_artifact_sha256": self.vjp_artifact_sha256,
            "provider_artifact_sha256": self.provider_artifact_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "maximum_future_gradient_abs_hex": (
                self.maximum_future_gradient_abs.hex()
            ),
            "future_gradient_nonzero_count": self.future_gradient_nonzero_count,
            "raw_node_gradient_or_token_KL_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _NODE_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(self.artifact_sha256, label="path node artifact"):
            raise RuntimeError("candidate joint-state path node receipt drifted")


@dataclass(frozen=True, slots=True)
class CandidateJointStateEndpointTangentReceipt:
    """Provenance and causality for an optional contextual endpoint tangent."""

    endpoint_role: str
    token_count: int
    h4_gradient_shape: tuple[int, int, int]
    endpoint_h4_gradient_sha256: str
    endpoint_h4_gradient_frobenius: float
    endpoint_h4_rows_sha256: str
    endpoint_token_teacher_kl_sha256: str
    endpoint_displacement_sha256: str
    token_tangent_contraction_sha256: str
    token_tangent_contraction_mean: float
    token_tangent_contraction_minimum: float
    token_tangent_contraction_maximum: float
    vjp_artifact_sha256: str
    provider_artifact_sha256: str
    execution_artifact_sha256: str
    supervised_grid_sha256: str
    teacher_logits_sha256: str
    maximum_future_gradient_abs: float
    future_gradient_nonzero_count: int
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        role = str(self.endpoint_role)
        if role not in {"scalar_endpoint", "held_unit_endpoint"}:
            raise ValueError("endpoint tangent role differs")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("scalar tangent token count must be positive")
        shape = _shape3(
            self.h4_gradient_shape, label="scalar tangent gradient shape"
        )
        if shape[0] != self.token_count:
            raise ValueError("scalar tangent token count and gradient shape differ")
        norm = _finite_float(
            self.endpoint_h4_gradient_frobenius,
            label="endpoint tangent gradient Frobenius norm",
            nonnegative=True,
        )
        mean = _finite_float(
            self.token_tangent_contraction_mean,
            label="scalar tangent contraction mean",
        )
        minimum = _finite_float(
            self.token_tangent_contraction_minimum,
            label="scalar tangent contraction minimum",
        )
        maximum = _finite_float(
            self.token_tangent_contraction_maximum,
            label="scalar tangent contraction maximum",
        )
        tolerance = 64.0 * torch.finfo(torch.float64).eps * max(
            abs(mean), abs(minimum), abs(maximum), 1.0
        )
        if mean < minimum - tolerance or mean > maximum + tolerance:
            raise ValueError("scalar tangent contraction summary is inconsistent")
        future_maximum = _finite_float(
            self.maximum_future_gradient_abs,
            label="scalar tangent maximum future gradient",
            nonnegative=True,
        )
        if (
            type(self.future_gradient_nonzero_count) is not int
            or self.future_gradient_nonzero_count < 0
            or (self.future_gradient_nonzero_count == 0) != (future_maximum == 0.0)
        ):
            raise ValueError("scalar tangent future-gradient scalars are inconsistent")
        for value, label in (
            (
                self.endpoint_h4_gradient_sha256,
                "endpoint tangent H4 gradient",
            ),
            (self.endpoint_h4_rows_sha256, "tangent endpoint H4"),
            (
                self.endpoint_token_teacher_kl_sha256,
                "tangent endpoint token KL",
            ),
            (self.endpoint_displacement_sha256, "scalar tangent displacement"),
            (
                self.token_tangent_contraction_sha256,
                "scalar tangent contraction",
            ),
            (self.vjp_artifact_sha256, "scalar tangent VJP artifact"),
            (self.provider_artifact_sha256, "scalar tangent provider artifact"),
            (self.execution_artifact_sha256, "scalar tangent execution artifact"),
            (self.supervised_grid_sha256, "scalar tangent supervised grid"),
            (self.teacher_logits_sha256, "scalar tangent teacher logits"),
        ):
            _require_sha256(value, label=label)
        object.__setattr__(self, "endpoint_role", role)
        object.__setattr__(self, "h4_gradient_shape", shape)
        object.__setattr__(
            self, "endpoint_h4_gradient_frobenius", norm
        )
        object.__setattr__(self, "token_tangent_contraction_mean", mean)
        object.__setattr__(self, "token_tangent_contraction_minimum", minimum)
        object.__setattr__(self, "token_tangent_contraction_maximum", maximum)
        object.__setattr__(self, "maximum_future_gradient_abs", future_maximum)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_TANGENT_DOMAIN, self.metadata(include_artifact=False)),
        )

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "endpoint_role": self.endpoint_role,
            "token_count": self.token_count,
            "h4_gradient_shape": self.h4_gradient_shape,
            "endpoint_h4_gradient_sha256": self.endpoint_h4_gradient_sha256,
            "endpoint_h4_gradient_frobenius_hex": (
                self.endpoint_h4_gradient_frobenius.hex()
            ),
            "endpoint_h4_rows_sha256": self.endpoint_h4_rows_sha256,
            "endpoint_token_teacher_kl_sha256": (
                self.endpoint_token_teacher_kl_sha256
            ),
            "endpoint_displacement_sha256": self.endpoint_displacement_sha256,
            "token_tangent_contraction_sha256": (
                self.token_tangent_contraction_sha256
            ),
            "token_tangent_contraction_mean_hex": (
                self.token_tangent_contraction_mean.hex()
            ),
            "token_tangent_contraction_minimum_hex": (
                self.token_tangent_contraction_minimum.hex()
            ),
            "token_tangent_contraction_maximum_hex": (
                self.token_tangent_contraction_maximum.hex()
            ),
            "vjp_artifact_sha256": self.vjp_artifact_sha256,
            "provider_artifact_sha256": self.provider_artifact_sha256,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "supervised_grid_sha256": self.supervised_grid_sha256,
            "teacher_logits_sha256": self.teacher_logits_sha256,
            "maximum_future_gradient_abs_hex": (
                self.maximum_future_gradient_abs.hex()
            ),
            "future_gradient_nonzero_count": self.future_gradient_nonzero_count,
            "raw_endpoint_gradient_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _TANGENT_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(
            self.artifact_sha256, label="scalar tangent receipt artifact"
        ):
            raise RuntimeError("candidate joint-state scalar tangent receipt drifted")


@dataclass(frozen=True, slots=True)
class CandidateJointStatePathEvidence:
    """One prompt's authenticated scalar-to-joint path evidence."""

    example_id: str
    family_id: str
    scalar_endpoint_h4_rows: Tensor = field(repr=False)
    joint_endpoint_h4_rows: Tensor = field(repr=False)
    integrated_token_h4_gradients: Tensor = field(repr=False)
    scalar_token_teacher_kl: Tensor = field(repr=False)
    joint_token_teacher_kl: Tensor = field(repr=False)
    held_unit_endpoint_h4_rows: Tensor | None = field(default=None, repr=False)
    held_unit_token_teacher_kl: Tensor | None = field(default=None, repr=False)
    scalar_endpoint_tangent_contraction: Tensor | None = field(default=None, repr=False)
    scalar_endpoint_tangent_receipt: (
        CandidateJointStateEndpointTangentReceipt | None
    ) = None
    held_unit_endpoint_tangent_contraction: Tensor | None = field(
        default=None, repr=False
    )
    held_unit_endpoint_tangent_receipt: (
        CandidateJointStateEndpointTangentReceipt | None
    ) = None
    node_receipts: tuple[CandidateJointStatePathNodeReceipt, ...] = ()
    endpoint_pair_binding_sha256: str = ""
    scalar_endpoint_execution_artifact_sha256: str = ""
    joint_endpoint_execution_artifact_sha256: str = ""
    supervised_grid_sha256: str = ""
    teacher_logits_sha256: str = ""
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        example = _identifier(self.example_id, label="path example_id")
        family = _identifier(self.family_id, label="path family_id")
        scalar_h4 = _realized_endpoint(
            self.scalar_endpoint_h4_rows, label="realized scalar endpoint"
        )
        joint_h4 = _realized_endpoint(
            self.joint_endpoint_h4_rows, label="realized joint endpoint"
        )
        gradient = _float64(
            self.integrated_token_h4_gradients,
            label="integrated path token H4 gradients",
            ndim=3,
        )
        scalar_kl = _float64(
            self.scalar_token_teacher_kl,
            label="scalar endpoint token teacher KL",
            ndim=1,
        )
        joint_kl = _float64(
            self.joint_token_teacher_kl,
            label="joint endpoint token teacher KL",
            ndim=1,
        )
        unit_h4 = (
            None
            if self.held_unit_endpoint_h4_rows is None
            else _realized_endpoint(
                self.held_unit_endpoint_h4_rows,
                label="realized held-unit endpoint",
            )
        )
        unit_kl = (
            None
            if self.held_unit_token_teacher_kl is None
            else _float64(
                self.held_unit_token_teacher_kl,
                label="held-unit endpoint token teacher KL",
                ndim=1,
            )
        )
        scalar_tangent = (
            None
            if self.scalar_endpoint_tangent_contraction is None
            else _float64(
                self.scalar_endpoint_tangent_contraction,
                label="scalar endpoint tangent contraction",
                ndim=1,
            )
        )
        tangent_receipt = self.scalar_endpoint_tangent_receipt
        unit_tangent = (
            None
            if self.held_unit_endpoint_tangent_contraction is None
            else _float64(
                self.held_unit_endpoint_tangent_contraction,
                label="held-unit endpoint tangent contraction",
                ndim=1,
            )
        )
        unit_tangent_receipt = self.held_unit_endpoint_tangent_receipt
        if (
            scalar_h4.shape != joint_h4.shape
            or scalar_h4.dtype != joint_h4.dtype
            or gradient.shape[1:] != scalar_h4.shape
            or scalar_kl.shape != joint_kl.shape
            or gradient.shape[0] != scalar_kl.shape[0]
            or (
                scalar_tangent is not None
                and scalar_tangent.shape != scalar_kl.shape
            )
            or (scalar_tangent is None) != (tangent_receipt is None)
            or (unit_tangent is not None and unit_tangent.shape != scalar_kl.shape)
            or (unit_tangent is None) != (unit_tangent_receipt is None)
            or (unit_h4 is None) != (unit_kl is None)
            or (unit_h4 is None) != (unit_tangent is None)
            or (
                unit_h4 is not None
                and (
                    unit_h4.shape != scalar_h4.shape
                    or unit_h4.dtype != scalar_h4.dtype
                    or unit_kl is None
                    or unit_kl.shape != scalar_kl.shape
                )
            )
        ):
            raise ValueError("scalar-to-joint endpoint, gradient, and KL geometry differ")
        receipts = tuple(self.node_receipts)
        if (
            len(receipts) != 4
            or any(
                not isinstance(receipt, CandidateJointStatePathNodeReceipt)
                or receipt.node_index != index
                for index, receipt in enumerate(receipts)
            )
        ):
            raise ValueError("path evidence requires the ordered four GL4 receipts")
        for receipt in receipts:
            receipt.validate_integrity()
            if receipt.h4_gradient_shape != tuple(gradient.shape):
                raise ValueError("path node and integrated gradient shapes differ")
            expected_node_h4 = (
                scalar_h4.to(torch.float64)
                + receipt.path_fraction
                * (joint_h4.to(torch.float64) - scalar_h4.to(torch.float64))
            ).to(dtype=scalar_h4.dtype).contiguous()
            if (
                receipt.path_node_h4_shape != tuple(scalar_h4.shape)
                or receipt.path_node_h4_dtype != str(scalar_h4.dtype)
                or receipt.path_node_h4_sha256 != _tensor_sha256(expected_node_h4)
            ):
                raise ValueError("path node H4 receipt differs from frozen interpolation")
        zero_sha = _tensor_sha256(torch.zeros_like(gradient))
        if receipts[0].integrated_gradient_sha256_before != zero_sha:
            raise ValueError("path accumulation does not start from zero")
        for before, after in zip(receipts, receipts[1:]):
            if (
                before.integrated_gradient_sha256_after
                != after.integrated_gradient_sha256_before
            ):
                raise ValueError("path accumulation receipt chain is broken")
        if receipts[-1].integrated_gradient_sha256_after != _tensor_sha256(gradient):
            raise ValueError("path integrated gradient differs from receipt chain")
        for value, label in (
            (self.endpoint_pair_binding_sha256, "endpoint pair binding"),
            (
                self.scalar_endpoint_execution_artifact_sha256,
                "scalar endpoint execution artifact",
            ),
            (
                self.joint_endpoint_execution_artifact_sha256,
                "joint endpoint execution artifact",
            ),
            (self.supervised_grid_sha256, "supervised grid"),
            (self.teacher_logits_sha256, "teacher logits"),
        ):
            _require_sha256(value, label=label)
        if tangent_receipt is not None:
            if not isinstance(
                tangent_receipt, CandidateJointStateEndpointTangentReceipt
            ):
                raise TypeError("scalar endpoint tangent receipt type differs")
            tangent_receipt.validate_integrity()
            displacement = (
                joint_h4.to(torch.float64) - scalar_h4.to(torch.float64)
            ).contiguous()
            if (
                tangent_receipt.endpoint_role != "scalar_endpoint"
                or
                tangent_receipt.token_count != int(scalar_kl.shape[0])
                or tangent_receipt.h4_gradient_shape != tuple(gradient.shape)
                or tangent_receipt.endpoint_h4_rows_sha256
                != _tensor_sha256(scalar_h4)
                or tangent_receipt.endpoint_token_teacher_kl_sha256
                != _tensor_sha256(scalar_kl)
                or tangent_receipt.endpoint_displacement_sha256
                != _tensor_sha256(displacement)
                or tangent_receipt.token_tangent_contraction_sha256
                != _tensor_sha256(scalar_tangent)
                or tangent_receipt.supervised_grid_sha256
                != self.supervised_grid_sha256
                or tangent_receipt.teacher_logits_sha256
                != self.teacher_logits_sha256
            ):
                raise ValueError("scalar endpoint tangent receipt binding differs")
        if unit_tangent_receipt is not None:
            if not isinstance(
                unit_tangent_receipt, CandidateJointStateEndpointTangentReceipt
            ):
                raise TypeError("held-unit endpoint tangent receipt type differs")
            unit_tangent_receipt.validate_integrity()
            displacement = (
                joint_h4.to(torch.float64) - scalar_h4.to(torch.float64)
            ).contiguous()
            if (
                unit_tangent_receipt.endpoint_role != "held_unit_endpoint"
                or unit_tangent_receipt.token_count != int(scalar_kl.shape[0])
                or unit_tangent_receipt.h4_gradient_shape != tuple(gradient.shape)
                or unit_tangent_receipt.endpoint_h4_rows_sha256
                != _tensor_sha256(unit_h4)
                or unit_tangent_receipt.endpoint_token_teacher_kl_sha256
                != _tensor_sha256(unit_kl)
                or unit_tangent_receipt.endpoint_displacement_sha256
                != _tensor_sha256(displacement)
                or unit_tangent_receipt.token_tangent_contraction_sha256
                != _tensor_sha256(unit_tangent)
                or unit_tangent_receipt.supervised_grid_sha256
                != self.supervised_grid_sha256
                or unit_tangent_receipt.teacher_logits_sha256
                != self.teacher_logits_sha256
            ):
                raise ValueError("held-unit endpoint tangent receipt binding differs")
        object.__setattr__(self, "example_id", example)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "scalar_endpoint_h4_rows", scalar_h4)
        object.__setattr__(self, "joint_endpoint_h4_rows", joint_h4)
        object.__setattr__(self, "integrated_token_h4_gradients", gradient)
        object.__setattr__(self, "scalar_token_teacher_kl", scalar_kl)
        object.__setattr__(self, "joint_token_teacher_kl", joint_kl)
        object.__setattr__(self, "held_unit_endpoint_h4_rows", unit_h4)
        object.__setattr__(self, "held_unit_token_teacher_kl", unit_kl)
        object.__setattr__(
            self, "scalar_endpoint_tangent_contraction", scalar_tangent
        )
        object.__setattr__(self, "scalar_endpoint_tangent_receipt", tangent_receipt)
        object.__setattr__(
            self, "held_unit_endpoint_tangent_contraction", unit_tangent
        )
        object.__setattr__(
            self, "held_unit_endpoint_tangent_receipt", unit_tangent_receipt
        )
        object.__setattr__(self, "node_receipts", receipts)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return int(self.scalar_token_teacher_kl.shape[0])

    @property
    def h4_shape(self) -> tuple[int, int]:
        return tuple(int(size) for size in self.scalar_endpoint_h4_rows.shape)

    @property
    def has_scalar_endpoint_tangent(self) -> bool:
        return self.scalar_endpoint_tangent_contraction is not None

    @property
    def has_held_unit_endpoint_tangent(self) -> bool:
        return self.held_unit_endpoint_tangent_contraction is not None

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "h4_shape": self.h4_shape,
            "realized_endpoint_dtype": str(self.scalar_endpoint_h4_rows.dtype),
            "scalar_endpoint_h4_rows_sha256": _tensor_sha256(
                self.scalar_endpoint_h4_rows
            ),
            "joint_endpoint_h4_rows_sha256": _tensor_sha256(
                self.joint_endpoint_h4_rows
            ),
            "endpoint_displacement_sha256": _tensor_sha256(
                (
                    self.joint_endpoint_h4_rows.to(torch.float64)
                    - self.scalar_endpoint_h4_rows.to(torch.float64)
                ).contiguous()
            ),
            "integrated_token_h4_gradients_sha256": _tensor_sha256(
                self.integrated_token_h4_gradients
            ),
            "scalar_token_teacher_kl_sha256": _tensor_sha256(
                self.scalar_token_teacher_kl
            ),
            "joint_token_teacher_kl_sha256": _tensor_sha256(
                self.joint_token_teacher_kl
            ),
            "held_unit_endpoint_h4_rows_sha256": (
                None
                if self.held_unit_endpoint_h4_rows is None
                else _tensor_sha256(self.held_unit_endpoint_h4_rows)
            ),
            "held_unit_token_teacher_kl_sha256": (
                None
                if self.held_unit_token_teacher_kl is None
                else _tensor_sha256(self.held_unit_token_teacher_kl)
            ),
            "scalar_endpoint_tangent_contraction_sha256": (
                None
                if self.scalar_endpoint_tangent_contraction is None
                else _tensor_sha256(self.scalar_endpoint_tangent_contraction)
            ),
            "scalar_endpoint_tangent_receipt": (
                None
                if self.scalar_endpoint_tangent_receipt is None
                else self.scalar_endpoint_tangent_receipt.metadata()
            ),
            "held_unit_endpoint_tangent_contraction_sha256": (
                None
                if self.held_unit_endpoint_tangent_contraction is None
                else _tensor_sha256(self.held_unit_endpoint_tangent_contraction)
            ),
            "held_unit_endpoint_tangent_receipt": (
                None
                if self.held_unit_endpoint_tangent_receipt is None
                else self.held_unit_endpoint_tangent_receipt.metadata()
            ),
            "supervised_token_count": self.supervised_tokens,
            "node_receipts": tuple(receipt.metadata() for receipt in self.node_receipts),
            "endpoint_pair_binding_sha256": self.endpoint_pair_binding_sha256,
            "scalar_endpoint_execution_artifact_sha256": (
                self.scalar_endpoint_execution_artifact_sha256
            ),
            "joint_endpoint_execution_artifact_sha256": (
                self.joint_endpoint_execution_artifact_sha256
            ),
            "supervised_grid_sha256": self.supervised_grid_sha256,
            "teacher_logits_sha256": self.teacher_logits_sha256,
            "quadrature_rule": "gauss_legendre_order_4_on_unit_interval",
            "path_geometry": _PATH_GEOMETRY,
            "path_objective": _PATH_OBJECTIVE,
            "FTC_orientation": _FTC_ORIENTATION,
            "endpoints_are_supplied_realized_cast_once_H4_values": True,
            "pre_cast_ideal_endpoints_used": False,
            "uses_only_strictly_interior_path_nodes": True,
            "path_node_construction": (
                "interpolate_supplied_realized_endpoints_in_float64_then_cast_once_to_endpoint_dtype"
            ),
            "every_path_node_H4_authenticated_against_frozen_interpolation": True,
            "scalar_endpoint_tangent_available": self.has_scalar_endpoint_tangent,
            "scalar_endpoint_tangent_causal": (
                None
                if self.scalar_endpoint_tangent_receipt is None
                else self.scalar_endpoint_tangent_receipt.future_gradient_nonzero_count
                == 0
                and self.scalar_endpoint_tangent_receipt.maximum_future_gradient_abs
                == 0.0
            ),
            "scalar_endpoint_tangent_substituted_for_GL4_integral": False,
            "held_unit_endpoint_tangent_available": (
                self.has_held_unit_endpoint_tangent
            ),
            "held_unit_endpoint_tangent_causal": (
                None
                if self.held_unit_endpoint_tangent_receipt is None
                else self.held_unit_endpoint_tangent_receipt.future_gradient_nonzero_count
                == 0
                and self.held_unit_endpoint_tangent_receipt.maximum_future_gradient_abs
                == 0.0
            ),
            "held_unit_endpoint_tangent_substituted_for_GL4_integral": False,
            "held_unit_tangent_displacement": (
                "joint_actual_minus_scalar_actual_same_displacement_different_reference"
            ),
            "finite_endpoint_KLs_used_as_integrand_nodes": False,
            "all_path_nodes_causal": all(
                receipt.future_gradient_nonzero_count == 0
                and receipt.maximum_future_gradient_abs == 0.0
                for receipt in self.node_receipts
            ),
            "full_node_gradient_banks_retained": False,
            "raw_evidence_serialized": False,
            "fits_or_selects_gain_fields": False,
            "hypothesis_use_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        for receipt in self.node_receipts:
            receipt.validate_integrity()
            expected_node_h4 = (
                self.scalar_endpoint_h4_rows.to(torch.float64)
                + receipt.path_fraction
                * (
                    self.joint_endpoint_h4_rows.to(torch.float64)
                    - self.scalar_endpoint_h4_rows.to(torch.float64)
                )
            ).to(dtype=self.scalar_endpoint_h4_rows.dtype).contiguous()
            if (
                receipt.path_node_h4_shape != self.h4_shape
                or receipt.path_node_h4_dtype
                != str(self.scalar_endpoint_h4_rows.dtype)
                or receipt.path_node_h4_sha256 != _tensor_sha256(expected_node_h4)
            ):
                raise RuntimeError(
                    "candidate joint-state path node H4 binding drifted"
                )
        if self.scalar_endpoint_tangent_receipt is not None:
            self.scalar_endpoint_tangent_receipt.validate_integrity()
            displacement = (
                self.joint_endpoint_h4_rows.to(torch.float64)
                - self.scalar_endpoint_h4_rows.to(torch.float64)
            ).contiguous()
            if (
                self.scalar_endpoint_tangent_contraction is None
                or self.scalar_endpoint_tangent_receipt.endpoint_role
                != "scalar_endpoint"
                or self.scalar_endpoint_tangent_receipt.endpoint_h4_rows_sha256
                != _tensor_sha256(self.scalar_endpoint_h4_rows)
                or self.scalar_endpoint_tangent_receipt.endpoint_token_teacher_kl_sha256
                != _tensor_sha256(self.scalar_token_teacher_kl)
                or self.scalar_endpoint_tangent_receipt.endpoint_displacement_sha256
                != _tensor_sha256(displacement)
                or self.scalar_endpoint_tangent_receipt.token_tangent_contraction_sha256
                != _tensor_sha256(self.scalar_endpoint_tangent_contraction)
                or self.scalar_endpoint_tangent_receipt.supervised_grid_sha256
                != self.supervised_grid_sha256
                or self.scalar_endpoint_tangent_receipt.teacher_logits_sha256
                != self.teacher_logits_sha256
            ):
                raise RuntimeError(
                    "candidate joint-state scalar tangent binding drifted"
                )
        if self.held_unit_endpoint_tangent_receipt is not None:
            self.held_unit_endpoint_tangent_receipt.validate_integrity()
            displacement = (
                self.joint_endpoint_h4_rows.to(torch.float64)
                - self.scalar_endpoint_h4_rows.to(torch.float64)
            ).contiguous()
            if (
                self.held_unit_endpoint_tangent_contraction is None
                or self.held_unit_endpoint_tangent_receipt.endpoint_role
                != "held_unit_endpoint"
                or self.held_unit_endpoint_h4_rows is None
                or self.held_unit_token_teacher_kl is None
                or self.held_unit_endpoint_tangent_receipt.endpoint_h4_rows_sha256
                != _tensor_sha256(self.held_unit_endpoint_h4_rows)
                or self.held_unit_endpoint_tangent_receipt.endpoint_token_teacher_kl_sha256
                != _tensor_sha256(self.held_unit_token_teacher_kl)
                or self.held_unit_endpoint_tangent_receipt.endpoint_displacement_sha256
                != _tensor_sha256(displacement)
                or self.held_unit_endpoint_tangent_receipt.token_tangent_contraction_sha256
                != _tensor_sha256(self.held_unit_endpoint_tangent_contraction)
                or self.held_unit_endpoint_tangent_receipt.supervised_grid_sha256
                != self.supervised_grid_sha256
                or self.held_unit_endpoint_tangent_receipt.teacher_logits_sha256
                != self.teacher_logits_sha256
            ):
                raise RuntimeError(
                    "candidate joint-state held-unit tangent binding drifted"
                )
        tensors = (
            self.scalar_endpoint_h4_rows,
            self.joint_endpoint_h4_rows,
            self.integrated_token_h4_gradients,
            self.scalar_token_teacher_kl,
            self.joint_token_teacher_kl,
        ) + (
            ()
            if self.scalar_endpoint_tangent_contraction is None
            else (self.scalar_endpoint_tangent_contraction,)
        ) + (
            ()
            if self.held_unit_endpoint_tangent_contraction is None
            else (self.held_unit_endpoint_tangent_contraction,)
        ) + (
            ()
            if self.held_unit_endpoint_h4_rows is None
            else (
                self.held_unit_endpoint_h4_rows,
                self.held_unit_token_teacher_kl,
            )
        )
        if (
            any(
                tensor.device.type != "cpu"
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all())
                for tensor in tensors
            )
            or self.integrated_token_h4_gradients.dtype != torch.float64
            or self.scalar_token_teacher_kl.dtype != torch.float64
            or self.joint_token_teacher_kl.dtype != torch.float64
            or (
                self.scalar_endpoint_tangent_contraction is not None
                and self.scalar_endpoint_tangent_contraction.dtype != torch.float64
            )
            or (
                self.held_unit_endpoint_tangent_contraction is not None
                and self.held_unit_endpoint_tangent_contraction.dtype
                != torch.float64
            )
            or self.node_receipts[-1].integrated_gradient_sha256_after
            != _tensor_sha256(self.integrated_token_h4_gradients)
            or _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="path evidence artifact")
        ):
            raise RuntimeError("candidate joint-state path evidence drifted")


class CandidateJointStatePathAccumulator:
    """Stream four GL4 nodes between authenticated realized H4 endpoints."""

    __slots__ = (
        "_example_id",
        "_family_id",
        "_scalar_h4",
        "_joint_h4",
        "_scalar_kl",
        "_joint_kl",
        "_scalar_tangent_contraction",
        "_scalar_tangent_receipt",
        "_held_unit_tangent_contraction",
        "_held_unit_tangent_receipt",
        "_held_unit_h4",
        "_held_unit_kl",
        "_endpoint_pair_binding_sha256",
        "_scalar_endpoint_execution_artifact_sha256",
        "_joint_endpoint_execution_artifact_sha256",
        "_supervised_grid_sha256",
        "_teacher_logits_sha256",
        "_integrated",
        "_node_receipts",
        "_sealed",
    )

    def __init__(
        self,
        *,
        example_id: str,
        family_id: str,
        scalar_endpoint_h4_rows: Tensor,
        joint_endpoint_h4_rows: Tensor,
        scalar_token_teacher_kl: Tensor,
        joint_token_teacher_kl: Tensor,
        endpoint_pair_binding_sha256: str,
        scalar_endpoint_execution_artifact_sha256: str,
        joint_endpoint_execution_artifact_sha256: str,
        supervised_grid_sha256: str,
        teacher_logits_sha256: str,
        scalar_endpoint_token_h4_gradients: Tensor | None = None,
        scalar_tangent_vjp_artifact_sha256: str | None = None,
        scalar_tangent_provider_artifact_sha256: str | None = None,
        scalar_tangent_execution_artifact_sha256: str | None = None,
        scalar_tangent_maximum_future_gradient_abs: float | None = None,
        scalar_tangent_future_gradient_nonzero_count: int | None = None,
        held_unit_endpoint_h4_rows: Tensor | None = None,
        held_unit_token_teacher_kl: Tensor | None = None,
        held_unit_endpoint_token_h4_gradients: Tensor | None = None,
        held_unit_tangent_vjp_artifact_sha256: str | None = None,
        held_unit_tangent_provider_artifact_sha256: str | None = None,
        held_unit_tangent_execution_artifact_sha256: str | None = None,
        held_unit_tangent_maximum_future_gradient_abs: float | None = None,
        held_unit_tangent_future_gradient_nonzero_count: int | None = None,
    ) -> None:
        self._example_id = _identifier(example_id, label="path example_id")
        self._family_id = _identifier(family_id, label="path family_id")
        self._scalar_h4 = _realized_endpoint(
            scalar_endpoint_h4_rows, label="realized scalar endpoint"
        )
        self._joint_h4 = _realized_endpoint(
            joint_endpoint_h4_rows, label="realized joint endpoint"
        )
        self._scalar_kl = _float64(
            scalar_token_teacher_kl,
            label="scalar endpoint token teacher KL",
            ndim=1,
        )
        self._joint_kl = _float64(
            joint_token_teacher_kl,
            label="joint endpoint token teacher KL",
            ndim=1,
        )
        scalar_tangent_gradient = (
            None
            if scalar_endpoint_token_h4_gradients is None
            else _float64(
                scalar_endpoint_token_h4_gradients,
                label="scalar endpoint token H4 gradients",
                ndim=3,
            )
        )
        if (
            self._scalar_h4.shape != self._joint_h4.shape
            or self._scalar_h4.dtype != self._joint_h4.dtype
            or self._scalar_kl.shape != self._joint_kl.shape
            or (
                scalar_tangent_gradient is not None
                and scalar_tangent_gradient.shape
                != (self._scalar_kl.shape[0], *self._scalar_h4.shape)
            )
        ):
            raise ValueError("scalar-to-joint endpoint geometry differs")
        self._endpoint_pair_binding_sha256 = _require_sha256(
            endpoint_pair_binding_sha256, label="endpoint pair binding"
        )
        self._scalar_endpoint_execution_artifact_sha256 = _require_sha256(
            scalar_endpoint_execution_artifact_sha256,
            label="scalar endpoint execution artifact",
        )
        self._joint_endpoint_execution_artifact_sha256 = _require_sha256(
            joint_endpoint_execution_artifact_sha256,
            label="joint endpoint execution artifact",
        )
        self._supervised_grid_sha256 = _require_sha256(
            supervised_grid_sha256, label="supervised grid"
        )
        self._teacher_logits_sha256 = _require_sha256(
            teacher_logits_sha256, label="teacher logits"
        )
        tangent_provenance = (
            scalar_tangent_vjp_artifact_sha256,
            scalar_tangent_provider_artifact_sha256,
            scalar_tangent_execution_artifact_sha256,
            scalar_tangent_maximum_future_gradient_abs,
            scalar_tangent_future_gradient_nonzero_count,
        )
        if scalar_tangent_gradient is None:
            if any(value is not None for value in tangent_provenance):
                raise ValueError(
                    "scalar tangent provenance requires scalar endpoint gradients"
                )
            self._scalar_tangent_contraction = None
            self._scalar_tangent_receipt = None
        else:
            if any(value is None for value in tangent_provenance):
                raise ValueError(
                    "scalar endpoint gradients require complete tangent provenance"
                )
            displacement = (
                self._joint_h4.to(torch.float64)
                - self._scalar_h4.to(torch.float64)
            ).contiguous()
            contraction = torch.einsum(
                "rw,trw->t", displacement, scalar_tangent_gradient
            ).contiguous()
            self._scalar_tangent_contraction = contraction
            self._scalar_tangent_receipt = (
                CandidateJointStateEndpointTangentReceipt(
                    endpoint_role="scalar_endpoint",
                    token_count=int(contraction.shape[0]),
                    h4_gradient_shape=tuple(
                        int(size) for size in scalar_tangent_gradient.shape
                    ),
                    endpoint_h4_gradient_sha256=_tensor_sha256(
                        scalar_tangent_gradient
                    ),
                    endpoint_h4_gradient_frobenius=float(
                        torch.linalg.vector_norm(scalar_tangent_gradient)
                    ),
                    endpoint_h4_rows_sha256=_tensor_sha256(self._scalar_h4),
                    endpoint_token_teacher_kl_sha256=_tensor_sha256(
                        self._scalar_kl
                    ),
                    endpoint_displacement_sha256=_tensor_sha256(displacement),
                    token_tangent_contraction_sha256=_tensor_sha256(contraction),
                    token_tangent_contraction_mean=float(contraction.mean()),
                    token_tangent_contraction_minimum=float(contraction.min()),
                    token_tangent_contraction_maximum=float(contraction.max()),
                    vjp_artifact_sha256=str(
                        scalar_tangent_vjp_artifact_sha256
                    ),
                    provider_artifact_sha256=str(
                        scalar_tangent_provider_artifact_sha256
                    ),
                    execution_artifact_sha256=str(
                        scalar_tangent_execution_artifact_sha256
                    ),
                    supervised_grid_sha256=self._supervised_grid_sha256,
                    teacher_logits_sha256=self._teacher_logits_sha256,
                    maximum_future_gradient_abs=float(
                        scalar_tangent_maximum_future_gradient_abs
                    ),
                    future_gradient_nonzero_count=int(
                        scalar_tangent_future_gradient_nonzero_count
                    ),
                )
            )
        unit_inputs = (
            held_unit_endpoint_h4_rows,
            held_unit_token_teacher_kl,
            held_unit_endpoint_token_h4_gradients,
        )
        unit_provenance = (
            held_unit_tangent_vjp_artifact_sha256,
            held_unit_tangent_provider_artifact_sha256,
            held_unit_tangent_execution_artifact_sha256,
            held_unit_tangent_maximum_future_gradient_abs,
            held_unit_tangent_future_gradient_nonzero_count,
        )
        if all(value is None for value in unit_inputs):
            if any(value is not None for value in unit_provenance):
                raise ValueError(
                    "held-unit tangent provenance requires endpoint evidence"
                )
            self._held_unit_tangent_contraction = None
            self._held_unit_tangent_receipt = None
            self._held_unit_h4 = None
            self._held_unit_kl = None
        else:
            if any(value is None for value in unit_inputs + unit_provenance):
                raise ValueError(
                    "held-unit tangent requires complete endpoint evidence and provenance"
                )
            unit_h4 = _realized_endpoint(
                held_unit_endpoint_h4_rows,
                label="realized held-unit endpoint",
            )
            unit_kl = _float64(
                held_unit_token_teacher_kl,
                label="held-unit endpoint token teacher KL",
                ndim=1,
            )
            unit_gradient = _float64(
                held_unit_endpoint_token_h4_gradients,
                label="held-unit endpoint token H4 gradients",
                ndim=3,
            )
            if (
                unit_h4.shape != self._scalar_h4.shape
                or unit_h4.dtype != self._scalar_h4.dtype
                or unit_kl.shape != self._scalar_kl.shape
                or unit_gradient.shape
                != (self._scalar_kl.shape[0], *self._scalar_h4.shape)
            ):
                raise ValueError("held-unit tangent endpoint geometry differs")
            displacement = (
                self._joint_h4.to(torch.float64)
                - self._scalar_h4.to(torch.float64)
            ).contiguous()
            contraction = torch.einsum(
                "rw,trw->t", displacement, unit_gradient
            ).contiguous()
            self._held_unit_h4 = unit_h4
            self._held_unit_kl = unit_kl
            self._held_unit_tangent_contraction = contraction
            self._held_unit_tangent_receipt = CandidateJointStateEndpointTangentReceipt(
                endpoint_role="held_unit_endpoint",
                token_count=int(contraction.shape[0]),
                h4_gradient_shape=tuple(int(size) for size in unit_gradient.shape),
                endpoint_h4_gradient_sha256=_tensor_sha256(unit_gradient),
                endpoint_h4_gradient_frobenius=float(
                    torch.linalg.vector_norm(unit_gradient)
                ),
                endpoint_h4_rows_sha256=_tensor_sha256(unit_h4),
                endpoint_token_teacher_kl_sha256=_tensor_sha256(unit_kl),
                endpoint_displacement_sha256=_tensor_sha256(displacement),
                token_tangent_contraction_sha256=_tensor_sha256(contraction),
                token_tangent_contraction_mean=float(contraction.mean()),
                token_tangent_contraction_minimum=float(contraction.min()),
                token_tangent_contraction_maximum=float(contraction.max()),
                vjp_artifact_sha256=str(held_unit_tangent_vjp_artifact_sha256),
                provider_artifact_sha256=str(
                    held_unit_tangent_provider_artifact_sha256
                ),
                execution_artifact_sha256=str(
                    held_unit_tangent_execution_artifact_sha256
                ),
                supervised_grid_sha256=self._supervised_grid_sha256,
                teacher_logits_sha256=self._teacher_logits_sha256,
                maximum_future_gradient_abs=float(
                    held_unit_tangent_maximum_future_gradient_abs
                ),
                future_gradient_nonzero_count=int(
                    held_unit_tangent_future_gradient_nonzero_count
                ),
            )
        self._integrated = torch.zeros(
            (int(self._scalar_kl.shape[0]), *self._scalar_h4.shape),
            dtype=torch.float64,
        )
        self._node_receipts: list[CandidateJointStatePathNodeReceipt] = []
        self._sealed = False

    @property
    def node_count(self) -> int:
        return len(self._node_receipts)

    def add_node(
        self,
        *,
        node_index: int,
        path_fraction: float,
        quadrature_weight: float,
        path_node_h4_rows: Tensor,
        token_h4_gradients: Tensor,
        token_teacher_kl: Tensor,
        vjp_artifact_sha256: str,
        provider_artifact_sha256: str,
        execution_artifact_sha256: str,
        maximum_future_gradient_abs: float,
        future_gradient_nonzero_count: int,
    ) -> CandidateJointStatePathNodeReceipt:
        if self._sealed:
            raise RuntimeError("path accumulator is already sealed")
        if type(node_index) is not int or node_index != len(self._node_receipts):
            raise ValueError("path nodes must be added once in canonical GL4 order")
        if node_index >= 4:
            raise ValueError("path accumulator already has all four GL4 nodes")
        node = _finite_float(path_fraction, label="path fraction")
        weight = _finite_float(
            quadrature_weight, label="quadrature weight", nonnegative=True
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
        ):
            raise ValueError("path node does not match the exact GL4 rule")
        evaluated_h4 = _realized_endpoint(
            path_node_h4_rows, label="evaluated path node H4"
        )
        expected_h4 = (
            self._scalar_h4.to(torch.float64)
            + node
            * (
                self._joint_h4.to(torch.float64)
                - self._scalar_h4.to(torch.float64)
            )
        ).to(dtype=self._scalar_h4.dtype).contiguous()
        if (
            evaluated_h4.shape != self._scalar_h4.shape
            or evaluated_h4.dtype != self._scalar_h4.dtype
            or not torch.equal(evaluated_h4, expected_h4)
        ):
            raise ValueError(
                "evaluated path node H4 differs from frozen float64-interpolate-cast-once policy"
            )
        gradient = _float64(
            token_h4_gradients, label="path node token H4 gradients", ndim=3
        )
        token_kl = _float64(
            token_teacher_kl, label="path node token teacher KL", ndim=1
        )
        if gradient.shape != self._integrated.shape or token_kl.shape != (
            self._integrated.shape[0],
        ):
            raise ValueError("path node gradient or token KL geometry differs")
        before = _tensor_sha256(self._integrated)
        updated = (self._integrated + weight * gradient).contiguous()
        receipt = CandidateJointStatePathNodeReceipt(
            node_index=node_index,
            path_fraction=node,
            quadrature_weight=weight,
            token_count=int(token_kl.shape[0]),
            h4_gradient_shape=tuple(int(size) for size in gradient.shape),
            path_node_h4_shape=tuple(int(size) for size in evaluated_h4.shape),
            path_node_h4_dtype=str(evaluated_h4.dtype),
            path_node_h4_sha256=_tensor_sha256(evaluated_h4),
            token_teacher_kl_sha256=_tensor_sha256(token_kl),
            token_teacher_kl_mean=float(token_kl.mean()),
            token_teacher_kl_minimum=float(token_kl.min()),
            token_teacher_kl_maximum=float(token_kl.max()),
            h4_gradient_sha256=_tensor_sha256(gradient),
            h4_gradient_frobenius=float(torch.linalg.vector_norm(gradient)),
            integrated_gradient_sha256_before=before,
            integrated_gradient_sha256_after=_tensor_sha256(updated),
            vjp_artifact_sha256=vjp_artifact_sha256,
            provider_artifact_sha256=provider_artifact_sha256,
            execution_artifact_sha256=execution_artifact_sha256,
            maximum_future_gradient_abs=maximum_future_gradient_abs,
            future_gradient_nonzero_count=future_gradient_nonzero_count,
        )
        self._integrated = updated
        self._node_receipts.append(receipt)
        return receipt

    def finalize(self) -> CandidateJointStatePathEvidence:
        if self._sealed:
            raise RuntimeError("path accumulator is already sealed")
        if len(self._node_receipts) != 4:
            raise RuntimeError("path accumulator requires all four GL4 nodes")
        evidence = CandidateJointStatePathEvidence(
            example_id=self._example_id,
            family_id=self._family_id,
            scalar_endpoint_h4_rows=self._scalar_h4,
            joint_endpoint_h4_rows=self._joint_h4,
            integrated_token_h4_gradients=self._integrated,
            scalar_token_teacher_kl=self._scalar_kl,
            joint_token_teacher_kl=self._joint_kl,
            held_unit_endpoint_h4_rows=self._held_unit_h4,
            held_unit_token_teacher_kl=self._held_unit_kl,
            scalar_endpoint_tangent_contraction=self._scalar_tangent_contraction,
            scalar_endpoint_tangent_receipt=self._scalar_tangent_receipt,
            held_unit_endpoint_tangent_contraction=(
                self._held_unit_tangent_contraction
            ),
            held_unit_endpoint_tangent_receipt=self._held_unit_tangent_receipt,
            node_receipts=tuple(self._node_receipts),
            endpoint_pair_binding_sha256=self._endpoint_pair_binding_sha256,
            scalar_endpoint_execution_artifact_sha256=(
                self._scalar_endpoint_execution_artifact_sha256
            ),
            joint_endpoint_execution_artifact_sha256=(
                self._joint_endpoint_execution_artifact_sha256
            ),
            supervised_grid_sha256=self._supervised_grid_sha256,
            teacher_logits_sha256=self._teacher_logits_sha256,
        )
        self._sealed = True
        return evidence


def _path_evidence(value: object) -> CandidateJointStatePathEvidence:
    if not isinstance(value, CandidateJointStatePathEvidence):
        raise TypeError("value must be candidate joint-state path evidence")
    value.validate_integrity()
    return value


def candidate_joint_state_path_displacement(
    evidence: CandidateJointStatePathEvidence,
) -> Tensor:
    """Return ``H_joint_actual - H_scalar_actual`` as a float64 copy."""

    value = _path_evidence(evidence)
    return (
        value.joint_endpoint_h4_rows.to(torch.float64)
        - value.scalar_endpoint_h4_rows.to(torch.float64)
    ).contiguous()


def candidate_joint_state_path_integrated_contraction(
    evidence: CandidateJointStatePathEvidence,
) -> Tensor:
    """Contract the GL4-integrated gradient with the realized displacement."""

    value = _path_evidence(evidence)
    displacement = candidate_joint_state_path_displacement(value)
    return torch.einsum(
        "rw,trw->t", displacement, value.integrated_token_h4_gradients
    ).contiguous()


def candidate_joint_state_finite_kl_delta(
    evidence: CandidateJointStatePathEvidence,
) -> Tensor:
    """Return finite ``KL_joint - KL_scalar`` in FTC orientation."""

    value = _path_evidence(evidence)
    return (value.joint_token_teacher_kl - value.scalar_token_teacher_kl).contiguous()


def candidate_joint_state_scalar_endpoint_tangent_contraction(
    evidence: CandidateJointStatePathEvidence,
) -> Tensor | None:
    """Return the optional scalar-endpoint tangent, never a path substitute."""

    value = _path_evidence(evidence)
    if value.scalar_endpoint_tangent_contraction is None:
        return None
    return value.scalar_endpoint_tangent_contraction.clone().contiguous()


def candidate_joint_state_held_unit_endpoint_tangent_contraction(
    evidence: CandidateJointStatePathEvidence,
) -> Tensor | None:
    """Return the optional unit-reference tangent along scalar-to-joint delta."""

    value = _path_evidence(evidence)
    if value.held_unit_endpoint_tangent_contraction is None:
        return None
    return value.held_unit_endpoint_tangent_contraction.clone().contiguous()


def _canonical_evidence(
    evidence: Iterable[CandidateJointStatePathEvidence],
) -> tuple[CandidateJointStatePathEvidence, ...]:
    values = tuple(evidence)
    if not values or any(
        not isinstance(value, CandidateJointStatePathEvidence) for value in values
    ):
        raise TypeError("path attribution requires nonempty typed evidence")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("path evidence example ids must be unique")
    # Sequence lengths legitimately produce different causal-support row
    # counts.  Only the model width is a cross-prompt invariant.
    if len({value.h4_shape[1] for value in ordered}) != 1:
        raise ValueError("path evidence H4 widths differ")
    tangent_presence = {value.has_scalar_endpoint_tangent for value in ordered}
    if len(tangent_presence) != 1:
        raise ValueError("scalar-endpoint tangent coverage must be uniform")
    unit_tangent_presence = {
        value.has_held_unit_endpoint_tangent for value in ordered
    }
    if len(unit_tangent_presence) != 1:
        raise ValueError("held-unit endpoint tangent coverage must be uniform")
    return ordered


def _family_prompt_token_mean(
    evidence: Sequence[CandidateJointStatePathEvidence],
    token_values: Mapping[str, Tensor],
) -> Tensor:
    if not isinstance(token_values, Mapping) or set(token_values) != {
        value.example_id for value in evidence
    }:
        raise ValueError("path token statistic keys differ from evidence")
    by_family: dict[str, list[Tensor]] = defaultdict(list)
    trailing_shape: tuple[int, ...] | None = None
    for value in evidence:
        statistic = token_values[value.example_id]
        if (
            not isinstance(statistic, Tensor)
            or statistic.ndim < 1
            or not statistic.is_floating_point()
            or statistic.shape[0] != value.supervised_tokens
            or not bool(torch.isfinite(statistic).all())
        ):
            raise ValueError("path token statistic must be finite and token aligned")
        statistic64 = statistic.detach().to(dtype=torch.float64, device="cpu")
        shape = tuple(int(size) for size in statistic64.shape[1:])
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("path token statistic trailing shapes differ")
        by_family[value.family_id].append(statistic64.mean(dim=0))
    return torch.stack(
        tuple(
            torch.stack(by_family[family]).mean(dim=0)
            for family in sorted(by_family)
        )
    ).mean(dim=0).contiguous()


def _sign_agreement(prediction: Tensor, target: Tensor) -> Tensor:
    # Zeros agree only with zeros.  This avoids declaring a no-op prediction to
    # have learned a direction merely because its product with the target is 0.
    return (
        ((prediction > 0.0) & (target > 0.0))
        | ((prediction < 0.0) & (target < 0.0))
        | ((prediction == 0.0) & (target == 0.0))
    ).to(torch.float64)


def _metrics(
    evidence: Sequence[CandidateJointStatePathEvidence],
) -> dict[str, float | None]:
    path = {
        value.example_id: candidate_joint_state_path_integrated_contraction(value)
        for value in evidence
    }
    finite = {
        value.example_id: candidate_joint_state_finite_kl_delta(value)
        for value in evidence
    }
    error = {key: path[key] - finite[key] for key in path}

    def mean(values: Mapping[str, Tensor]) -> float:
        return float(_family_prompt_token_mean(evidence, values))

    target_second = mean({key: value.square() for key, value in finite.items()})
    path_second = mean({key: value.square() for key, value in path.items()})
    error_second = mean({key: value.square() for key, value in error.items()})
    cross = mean({key: path[key] * finite[key] for key in path})
    target_rms = math.sqrt(max(target_second, 0.0))
    path_rms = math.sqrt(max(path_second, 0.0))
    rmse = math.sqrt(max(error_second, 0.0))
    epsilon = 64.0 * torch.finfo(torch.float64).eps
    denominator = target_rms * path_rms
    cosine = 0.0 if denominator <= epsilon else cross / denominator
    result: dict[str, float | None] = {
        "mean_finite_kl_delta": mean(finite),
        "mean_path_integral": mean(path),
        "mean_closure_error": mean(error),
        "mean_absolute_closure_error": mean(
            {key: value.abs() for key, value in error.items()}
        ),
        "finite_delta_rms": target_rms,
        "path_integral_rms": path_rms,
        "closure_rmse": rmse,
        "closure_relative_rmse": rmse / max(target_rms, epsilon),
        "closure_cosine": min(max(cosine, -1.0), 1.0),
        "maximum_absolute_closure_error": max(
            float(value.abs().max()) for value in error.values()
        ),
        "family_equal_token_sign_agreement_rate": mean(
            {key: _sign_agreement(path[key], finite[key]) for key in path}
        ),
        "relative_rmse_epsilon": epsilon,
        "mean_scalar_endpoint_tangent": None,
        "family_equal_tangent_finite_sign_agreement_rate": None,
        "mean_path_minus_scalar_tangent": None,
        "mean_held_unit_endpoint_tangent": None,
        "family_equal_held_unit_tangent_finite_sign_agreement_rate": None,
        "mean_path_minus_held_unit_tangent": None,
    }
    if evidence[0].has_scalar_endpoint_tangent:
        tangent = {
            value.example_id: candidate_joint_state_scalar_endpoint_tangent_contraction(
                value
            )
            for value in evidence
        }
        # Uniform coverage is checked above, so the optional values are tensors.
        typed_tangent = {key: value for key, value in tangent.items() if value is not None}
        result.update(
            {
                "mean_scalar_endpoint_tangent": mean(typed_tangent),
                "family_equal_tangent_finite_sign_agreement_rate": mean(
                    {
                        key: _sign_agreement(typed_tangent[key], finite[key])
                        for key in typed_tangent
                    }
                ),
                "mean_path_minus_scalar_tangent": mean(
                    {key: path[key] - typed_tangent[key] for key in path}
                ),
            }
        )
    if evidence[0].has_held_unit_endpoint_tangent:
        unit_tangent = {
            value.example_id: (
                candidate_joint_state_held_unit_endpoint_tangent_contraction(value)
            )
            for value in evidence
        }
        typed_unit_tangent = {
            key: value for key, value in unit_tangent.items() if value is not None
        }
        result.update(
            {
                "mean_held_unit_endpoint_tangent": mean(typed_unit_tangent),
                "family_equal_held_unit_tangent_finite_sign_agreement_rate": mean(
                    {
                        key: _sign_agreement(typed_unit_tangent[key], finite[key])
                        for key in typed_unit_tangent
                    }
                ),
                "mean_path_minus_held_unit_tangent": mean(
                    {key: path[key] - typed_unit_tangent[key] for key in path}
                ),
            }
        )
    return result


@dataclass(frozen=True, slots=True)
class CandidateJointStatePathFamilyAttribution:
    family_id: str
    example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    supervised_token_count: int
    mean_finite_kl_delta: float
    mean_path_integral: float
    mean_closure_error: float
    mean_absolute_closure_error: float
    finite_delta_rms: float
    path_integral_rms: float
    closure_rmse: float
    closure_relative_rmse: float
    closure_cosine: float
    maximum_absolute_closure_error: float
    family_equal_token_sign_agreement_rate: float
    relative_rmse_epsilon: float
    mean_scalar_endpoint_tangent: float | None
    family_equal_tangent_finite_sign_agreement_rate: float | None
    mean_path_minus_scalar_tangent: float | None
    mean_held_unit_endpoint_tangent: float | None
    family_equal_held_unit_tangent_finite_sign_agreement_rate: float | None
    mean_path_minus_held_unit_tangent: float | None

    def __post_init__(self) -> None:
        family = _identifier(self.family_id, label="path family summary family_id")
        examples = tuple(
            _identifier(value, label="path family summary example_id")
            for value in self.example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="path family evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
        ):
            raise ValueError("path family attribution membership is invalid")
        _validate_metric_payload(self)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)

    @property
    def finite_family_direction(self) -> int:
        return (self.mean_finite_kl_delta > 0.0) - (
            self.mean_finite_kl_delta < 0.0
        )

    @property
    def path_family_direction(self) -> int:
        return (self.mean_path_integral > 0.0) - (self.mean_path_integral < 0.0)

    @property
    def family_direction_agrees(self) -> bool:
        return self.finite_family_direction == self.path_family_direction

    def metadata(self) -> dict[str, object]:
        return {
            **_metric_metadata(self),
            "family_id": self.family_id,
            "example_ids": self.example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "supervised_token_count": self.supervised_token_count,
            "finite_family_direction": self.finite_family_direction,
            "path_family_direction": self.path_family_direction,
            "family_direction_agrees": self.family_direction_agrees,
        }


_METRIC_NAMES = (
    "mean_finite_kl_delta",
    "mean_path_integral",
    "mean_closure_error",
    "mean_absolute_closure_error",
    "finite_delta_rms",
    "path_integral_rms",
    "closure_rmse",
    "closure_relative_rmse",
    "closure_cosine",
    "maximum_absolute_closure_error",
    "family_equal_token_sign_agreement_rate",
    "relative_rmse_epsilon",
    "mean_scalar_endpoint_tangent",
    "family_equal_tangent_finite_sign_agreement_rate",
    "mean_path_minus_scalar_tangent",
    "mean_held_unit_endpoint_tangent",
    "family_equal_held_unit_tangent_finite_sign_agreement_rate",
    "mean_path_minus_held_unit_tangent",
)


def _validate_metric_payload(value: object) -> None:
    optional = {
        "mean_scalar_endpoint_tangent",
        "family_equal_tangent_finite_sign_agreement_rate",
        "mean_path_minus_scalar_tangent",
        "mean_held_unit_endpoint_tangent",
        "family_equal_held_unit_tangent_finite_sign_agreement_rate",
        "mean_path_minus_held_unit_tangent",
    }
    scalar_optional = {
        "mean_scalar_endpoint_tangent",
        "family_equal_tangent_finite_sign_agreement_rate",
        "mean_path_minus_scalar_tangent",
    }
    unit_optional = optional - scalar_optional
    if any(
        len({getattr(value, name) is not None for name in group}) != 1
        for group in (scalar_optional, unit_optional)
    ):
        raise ValueError("endpoint tangent metrics must be complete within each role")
    for name in _METRIC_NAMES:
        raw = getattr(value, name)
        if raw is None:
            if name not in optional:
                raise ValueError(f"path attribution {name} may not be absent")
            continue
        metric = _finite_float(raw, label=f"path attribution {name}")
        if name in {
            "mean_absolute_closure_error",
            "finite_delta_rms",
            "path_integral_rms",
            "closure_rmse",
            "closure_relative_rmse",
            "maximum_absolute_closure_error",
            "relative_rmse_epsilon",
        } and metric < 0.0:
            raise ValueError(f"path attribution {name} must be nonnegative")
        if name == "closure_cosine" and not -1.0 <= metric <= 1.0:
            raise ValueError("path attribution closure cosine must be in [-1, 1]")
        if "sign_agreement_rate" in name and not 0.0 <= metric <= 1.0:
            raise ValueError("path attribution sign-agreement rate must be in [0, 1]")
        object.__setattr__(value, name, metric)


def _metric_metadata(value: object) -> dict[str, object]:
    return {name: getattr(value, name) for name in _METRIC_NAMES}


@dataclass(frozen=True, slots=True)
class CandidateJointStatePathAttribution:
    evidence_example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    family_summaries: tuple[CandidateJointStatePathFamilyAttribution, ...]
    supervised_token_count: int
    mean_finite_kl_delta: float
    mean_path_integral: float
    mean_closure_error: float
    mean_absolute_closure_error: float
    finite_delta_rms: float
    path_integral_rms: float
    closure_rmse: float
    closure_relative_rmse: float
    closure_cosine: float
    maximum_absolute_closure_error: float
    family_equal_token_sign_agreement_rate: float
    relative_rmse_epsilon: float
    mean_scalar_endpoint_tangent: float | None
    family_equal_tangent_finite_sign_agreement_rate: float | None
    mean_path_minus_scalar_tangent: float | None
    mean_held_unit_endpoint_tangent: float | None
    family_equal_held_unit_tangent_finite_sign_agreement_rate: float | None
    mean_path_minus_held_unit_tangent: float | None
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        examples = tuple(
            _identifier(value, label="path attribution example_id")
            for value in self.evidence_example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="path attribution evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        families = tuple(self.family_summaries)
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or not families
            or any(
                not isinstance(value, CandidateJointStatePathFamilyAttribution)
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
        ):
            raise ValueError("path attribution membership is invalid")
        _validate_metric_payload(self)
        object.__setattr__(self, "evidence_example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "family_summaries", families)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SUMMARY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def family_direction_agreement_count(self) -> int:
        return sum(family.family_direction_agrees for family in self.family_summaries)

    @property
    def family_finite_joint_improvement_count(self) -> int:
        return sum(
            family.mean_finite_kl_delta < 0.0 for family in self.family_summaries
        )

    @property
    def family_path_predicts_joint_improvement_count(self) -> int:
        return sum(
            family.mean_path_integral < 0.0 for family in self.family_summaries
        )

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            **_metric_metadata(self),
            "evidence_example_ids": self.evidence_example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "family_summaries": tuple(
                family.metadata() for family in self.family_summaries
            ),
            "supervised_token_count": self.supervised_token_count,
            "family_count": len(self.family_summaries),
            "family_direction_agreement_count": self.family_direction_agreement_count,
            "family_finite_joint_improvement_count": (
                self.family_finite_joint_improvement_count
            ),
            "family_path_predicts_joint_improvement_count": (
                self.family_path_predicts_joint_improvement_count
            ),
            "weighting": _WEIGHTING,
            "FTC_orientation": _FTC_ORIENTATION,
            "path_geometry": _PATH_GEOMETRY,
            "path_objective": _PATH_OBJECTIVE,
            "sign_zero_policy": "zero_agrees_only_with_zero",
            "scalar_endpoint_tangent_is_optional_first_order_context": True,
            "scalar_endpoint_tangent_substituted_for_GL4_integral": False,
            "held_unit_endpoint_tangent_is_optional_reference_shift_context": True,
            "held_unit_tangent_uses_scalar_to_joint_displacement": True,
            "held_unit_endpoint_tangent_substituted_for_GL4_integral": False,
            "native_or_D320_endpoint_schema_reused": False,
            "fits_selects_or_ranks_gain_fields": False,
            "endpoint_pair_path_evidence_only": True,
            "raw_evidence_serialized": False,
            "hypothesis_use_only": True,
            "authorizes_serving_compression_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _SUMMARY_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(self.artifact_sha256, label="path attribution artifact"):
            raise RuntimeError("candidate joint-state path attribution drifted")


def summarize_candidate_joint_state_path_attribution(
    evidence: Iterable[CandidateJointStatePathEvidence],
) -> CandidateJointStatePathAttribution:
    """Build family-equal finite/path closure and direction summaries."""

    values = _canonical_evidence(evidence)
    metrics = _metrics(values)
    by_family: dict[str, list[CandidateJointStatePathEvidence]] = defaultdict(list)
    for value in values:
        by_family[value.family_id].append(value)
    families: list[CandidateJointStatePathFamilyAttribution] = []
    for family_id in sorted(by_family):
        members = tuple(
            sorted(by_family[family_id], key=lambda value: value.example_id)
        )
        families.append(
            CandidateJointStatePathFamilyAttribution(
                family_id=family_id,
                example_ids=tuple(value.example_id for value in members),
                evidence_artifact_sha256s=tuple(
                    value.artifact_sha256 for value in members
                ),
                supervised_token_count=sum(
                    value.supervised_tokens for value in members
                ),
                **_metrics(members),
            )
        )
    return CandidateJointStatePathAttribution(
        evidence_example_ids=tuple(value.example_id for value in values),
        evidence_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        family_summaries=tuple(families),
        supervised_token_count=sum(value.supervised_tokens for value in values),
        **metrics,
    )
