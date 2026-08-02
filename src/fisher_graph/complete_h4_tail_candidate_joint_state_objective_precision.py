"""Pure V10 objective-precision comparison for the scalar-to-joint H4 path.

The live V10 diagnostic changes one thing: selected teacher and candidate
logits are promoted to float64 *before* teacher-KL arithmetic.  This module
does not execute a model and does not choose that arithmetic for serving.  It
binds the resulting float64 scalar tangent, GL4 integral, and finite endpoint
delta to the paired float32 endpoint KL values from the exact same forward
executions.

All statistics are computed by taking a token mean inside each prompt, an
equal prompt mean inside each family, and finally an equal family mean.
Prompt token counts and H4 support-row counts may differ.  Only hashes,
counts, and scalar summaries are serialized.
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

from .complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathEvidence,
    candidate_joint_state_finite_kl_delta,
    candidate_joint_state_path_integrated_contraction,
    candidate_joint_state_scalar_endpoint_tangent_contraction,
)


__all__ = [
    "CLOSURE_COSINE_MINIMUM",
    "DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE",
    "FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM",
    "OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM",
    "CandidateJointStateObjectivePrecisionComparison",
    "CandidateJointStateObjectivePrecisionEvidence",
    "CandidateJointStateObjectivePrecisionFamilySummary",
    "CandidateJointStateObjectivePrecisionMetrics",
    "classify_candidate_joint_state_objective_precision",
    "summarize_candidate_joint_state_objective_precision",
]


OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM = 0.05
FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM = 0.10
CLOSURE_COSINE_MINIMUM = 0.99
DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE = (
    128.0 * torch.finfo(torch.float64).eps
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:candidate-joint-state-objective-precision-tensor:v10\0"
_EVIDENCE_DOMAIN = (
    b"fisher-graph:candidate-joint-state-objective-precision-evidence:v10\0"
)
_FAMILY_DOMAIN = (
    b"fisher-graph:candidate-joint-state-objective-precision-family:v10\0"
)
_SUMMARY_DOMAIN = (
    b"fisher-graph:candidate-joint-state-objective-precision-summary:v10\0"
)
_WEIGHTING = (
    "mean_tokens_within_prompt_then_equal_prompts_within_family_then_"
    "equal_families"
)
_EPSILON = 64.0 * torch.finfo(torch.float64).eps
_DOMINANCE_TOLERANCE_FACTOR = 512.0 * torch.finfo(torch.float64).eps


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


def _float32_token_vector(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.numel() <= 0
        or value.dtype != torch.float32
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty float32 token vector")
    return value.detach().to(device="cpu").clone().contiguous()


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


@dataclass(frozen=True, slots=True)
class CandidateJointStateObjectivePrecisionEvidence:
    """One prompt's paired float64-path and float32-endpoint evidence."""

    path_evidence: CandidateJointStatePathEvidence = field(repr=False)
    finite_delta_f64_direct: Tensor = field(repr=False)
    scalar_token_teacher_kl_f32: Tensor = field(repr=False)
    joint_token_teacher_kl_f32: Tensor = field(repr=False)
    pinned_v9_evidence_artifact_sha256: str
    endpoint_replay_binding_sha256: str
    f32_objective_binding_sha256: str
    f64_objective_binding_sha256: str
    example_id: str = field(init=False)
    family_id: str = field(init=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path_evidence, CandidateJointStatePathEvidence):
            raise TypeError("objective-precision evidence requires V9 path evidence")
        self.path_evidence.validate_integrity()
        tangent = candidate_joint_state_scalar_endpoint_tangent_contraction(
            self.path_evidence
        )
        if tangent is None:
            raise ValueError("objective-precision evidence requires a scalar tangent")
        direct64 = _float64_token_vector(
            self.finite_delta_f64_direct,
            label="direct float64 finite teacher-KL delta",
        )
        scalar32 = _float32_token_vector(
            self.scalar_token_teacher_kl_f32,
            label="float32 scalar endpoint teacher KL",
        )
        joint32 = _float32_token_vector(
            self.joint_token_teacher_kl_f32,
            label="float32 joint endpoint teacher KL",
        )
        if (
            scalar32.shape != joint32.shape
            or scalar32.shape[0] != self.path_evidence.supervised_tokens
            or tangent.shape != scalar32.shape
            or direct64.shape != scalar32.shape
        ):
            raise ValueError("objective-precision token geometry differs")
        endpoint_subtraction = candidate_joint_state_finite_kl_delta(
            self.path_evidence
        )
        direct_residual = direct64 - endpoint_subtraction
        direct_tolerance = DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE * max(
            1.0,
            float(direct64.abs().max()),
            float(endpoint_subtraction.abs().max()),
        )
        if float(direct_residual.abs().max()) > direct_tolerance:
            raise ValueError(
                "direct float64 finite delta differs from endpoint-KL subtraction"
            )
        tangent_receipt = self.path_evidence.scalar_endpoint_tangent_receipt
        if (
            tangent_receipt is None
            or tangent_receipt.future_gradient_nonzero_count != 0
            or tangent_receipt.maximum_future_gradient_abs != 0.0
            or any(
                receipt.future_gradient_nonzero_count != 0
                or receipt.maximum_future_gradient_abs != 0.0
                for receipt in self.path_evidence.node_receipts
            )
        ):
            raise ValueError("objective-precision VJPs must be strictly causal")
        for name in (
            "pinned_v9_evidence_artifact_sha256",
            "endpoint_replay_binding_sha256",
            "f32_objective_binding_sha256",
            "f64_objective_binding_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), label=name),
            )
        if (
            self.endpoint_replay_binding_sha256
            != self.path_evidence.endpoint_pair_binding_sha256
        ):
            raise ValueError(
                "endpoint replay binding must equal the path endpoint-pair binding"
            )
        if self.f32_objective_binding_sha256 == self.f64_objective_binding_sha256:
            raise ValueError("float32 and float64 objective bindings must be distinct")
        if self.endpoint_replay_binding_sha256 in {
            self.f32_objective_binding_sha256,
            self.f64_objective_binding_sha256,
        }:
            raise ValueError(
                "endpoint replay and objective bindings must be distinct authorities"
            )
        object.__setattr__(self, "finite_delta_f64_direct", direct64)
        object.__setattr__(self, "scalar_token_teacher_kl_f32", scalar32)
        object.__setattr__(self, "joint_token_teacher_kl_f32", joint32)
        object.__setattr__(self, "example_id", self.path_evidence.example_id)
        object.__setattr__(self, "family_id", self.path_evidence.family_id)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return self.path_evidence.supervised_tokens

    @property
    def h4_shape(self) -> tuple[int, int]:
        return self.path_evidence.h4_shape

    def finite_delta_f64(self) -> Tensor:
        """Return the cancellation-resistant direct float64 finite delta."""

        return self.finite_delta_f64_direct.clone().contiguous()

    def finite_delta_f64_endpoint_subtraction(self) -> Tensor:
        """Return the separately evaluated endpoint-KL subtraction cross-check."""

        return candidate_joint_state_finite_kl_delta(self.path_evidence)

    def direct_endpoint_crosscheck_residual(self) -> Tensor:
        return (
            self.finite_delta_f64()
            - self.finite_delta_f64_endpoint_subtraction()
        ).contiguous()

    @property
    def direct_endpoint_crosscheck_tolerance(self) -> float:
        direct = self.finite_delta_f64()
        endpoint = self.finite_delta_f64_endpoint_subtraction()
        return DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE * max(
            1.0,
            float(direct.abs().max()),
            float(endpoint.abs().max()),
        )

    def scalar_tangent_f64(self) -> Tensor:
        value = candidate_joint_state_scalar_endpoint_tangent_contraction(
            self.path_evidence
        )
        if value is None:  # Construction and integrity make this unreachable.
            raise RuntimeError("objective-precision scalar tangent disappeared")
        return value

    def path_integral_f64(self) -> Tensor:
        return candidate_joint_state_path_integrated_contraction(self.path_evidence)

    def finite_delta_f32(self) -> Tensor:
        """Replay V9's endpoint delta from its paired float32 KL operands.

        V9 retained each endpoint by independently promoting the float32 KL
        vector to CPU float64, then subtracted those retained operands.  The
        ordering here is intentional: float32 subtraction followed by
        promotion can round to a different delta and would not replay V9.
        """

        return (
            self.joint_token_teacher_kl_f32.to(torch.float64)
            - self.scalar_token_teacher_kl_f32.to(torch.float64)
        ).contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        finite64 = self.finite_delta_f64()
        endpoint_subtraction64 = self.finite_delta_f64_endpoint_subtraction()
        direct_residual = self.direct_endpoint_crosscheck_residual()
        direct_tolerance = self.direct_endpoint_crosscheck_tolerance
        tangent64 = self.scalar_tangent_f64()
        path64 = self.path_integral_f64()
        finite32 = self.finite_delta_f32()
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "supervised_token_count": self.supervised_tokens,
            "h4_support_row_count": self.h4_shape[0],
            "h4_width": self.h4_shape[1],
            "path_evidence_artifact_sha256": self.path_evidence.artifact_sha256,
            "pinned_v9_evidence_artifact_sha256": (
                self.pinned_v9_evidence_artifact_sha256
            ),
            "endpoint_replay_binding_sha256": self.endpoint_replay_binding_sha256,
            "f32_objective_binding_sha256": self.f32_objective_binding_sha256,
            "f64_objective_binding_sha256": self.f64_objective_binding_sha256,
            "scalar_token_teacher_kl_f32_sha256": _tensor_sha256(
                self.scalar_token_teacher_kl_f32
            ),
            "joint_token_teacher_kl_f32_sha256": _tensor_sha256(
                self.joint_token_teacher_kl_f32
            ),
            "finite_delta_f32_sha256": _tensor_sha256(finite32),
            "finite_delta_f64_sha256": _tensor_sha256(finite64),
            "finite_delta_f64_endpoint_subtraction_sha256": _tensor_sha256(
                endpoint_subtraction64
            ),
            "finite_delta_f64_direct_minus_endpoint_subtraction_sha256": (
                _tensor_sha256(direct_residual)
            ),
            "finite_delta_f64_direct_minus_endpoint_subtraction_max_abs": (
                float(direct_residual.abs().max())
            ),
            "finite_delta_f64_direct_endpoint_crosscheck_tolerance": (
                direct_tolerance
            ),
            "finite_delta_f64_direct_endpoint_crosscheck_passed": (
                float(direct_residual.abs().max()) <= direct_tolerance
            ),
            "finite_delta_f64_direct_is_primary_metrics_authority": True,
            "scalar_tangent_f64_sha256": _tensor_sha256(tangent64),
            "path_integral_f64_sha256": _tensor_sha256(path64),
            "finite_delta_f32_definition": (
                "legacy_V9_float32_endpoint_operands_independently_promoted_"
                "to_float64_before_subtraction"
            ),
            "f32_endpoint_operands_promoted_before_subtraction": True,
            "f64_logits_required_before_teacher_KL_arithmetic": True,
            "same_forward_endpoint_and_path_replay_required": True,
            "future_H4_gradient_is_exact_zero": True,
            "raw_tensors_serialized": False,
            "fits_selects_or_routes_candidates": False,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.path_evidence.validate_integrity()
        direct_residual = self.direct_endpoint_crosscheck_residual()
        tensors = (
            self.finite_delta_f64_direct,
            self.scalar_token_teacher_kl_f32,
            self.joint_token_teacher_kl_f32,
        )
        if (
            any(
                value.dtype
                != (
                    torch.float64
                    if value is self.finite_delta_f64_direct
                    else torch.float32
                )
                or value.device.type != "cpu"
                or value.requires_grad
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all())
                for value in tensors
            )
            or self.example_id != self.path_evidence.example_id
            or self.family_id != self.path_evidence.family_id
            or self.endpoint_replay_binding_sha256
            != self.path_evidence.endpoint_pair_binding_sha256
            or self.f32_objective_binding_sha256
            == self.f64_objective_binding_sha256
            or self.endpoint_replay_binding_sha256
            in {
                self.f32_objective_binding_sha256,
                self.f64_objective_binding_sha256,
            }
            or self.path_evidence.scalar_endpoint_tangent_receipt is None
            or (
                self.path_evidence.scalar_endpoint_tangent_receipt
                .future_gradient_nonzero_count
                != 0
            )
            or any(
                receipt.future_gradient_nonzero_count != 0
                for receipt in self.path_evidence.node_receipts
            )
            or float(direct_residual.abs().max())
            > self.direct_endpoint_crosscheck_tolerance
            or _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(
                self.artifact_sha256,
                label="objective-precision evidence artifact",
            )
        ):
            raise RuntimeError(
                "candidate joint-state objective-precision evidence drifted"
            )


@dataclass(frozen=True, slots=True)
class CandidateJointStateObjectivePrecisionMetrics:
    """Scalar metrics for one family or the family-equal panel."""

    mean_finite_delta_f64: float
    mean_path_integral_f64: float
    mean_scalar_tangent_f64: float
    mean_finite_delta_f32: float
    mean_path_minus_finite_f64: float
    mean_path_minus_scalar_tangent_f64: float
    mean_finite_f64_minus_f32: float
    finite_delta_f64_rms: float
    path_integral_f64_rms: float
    scalar_tangent_f64_rms: float
    finite_delta_f32_rms: float
    closure_rmse: float
    closure_relative_rmse: float
    closure_cosine: float
    maximum_absolute_closure_error: float
    transport_rmse: float
    transport_relative_rmse_to_finite_f64: float
    transport_cosine: float
    finite_precision_rmse: float
    finite_precision_relative_rmse_to_finite_f64: float
    relative_rmse_epsilon: float

    def __post_init__(self) -> None:
        nonnegative = {
            "finite_delta_f64_rms",
            "path_integral_f64_rms",
            "scalar_tangent_f64_rms",
            "finite_delta_f32_rms",
            "closure_rmse",
            "closure_relative_rmse",
            "maximum_absolute_closure_error",
            "transport_rmse",
            "transport_relative_rmse_to_finite_f64",
            "finite_precision_rmse",
            "finite_precision_relative_rmse_to_finite_f64",
            "relative_rmse_epsilon",
        }
        for name in self.__dataclass_fields__:
            value = _finite_float(
                getattr(self, name),
                label=f"objective-precision metric {name}",
                nonnegative=name in nonnegative,
            )
            if name in {"closure_cosine", "transport_cosine"} and not (
                -1.0 <= value <= 1.0
            ):
                raise ValueError(f"objective-precision {name} must be in [-1, 1]")
            object.__setattr__(self, name, value)

    def metadata(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateObjectivePrecisionFamilySummary:
    family_id: str
    example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    supervised_token_count: int
    maximum_direct_endpoint_crosscheck_abs_error: float
    maximum_direct_endpoint_crosscheck_tolerance: float
    metrics: CandidateJointStateObjectivePrecisionMetrics
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        family = _identifier(self.family_id, label="objective-precision family_id")
        examples = tuple(
            _identifier(value, label="objective-precision family example_id")
            for value in self.example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="objective-precision evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        crosscheck_error = _finite_float(
            self.maximum_direct_endpoint_crosscheck_abs_error,
            label="family direct endpoint cross-check error",
            nonnegative=True,
        )
        crosscheck_tolerance = _finite_float(
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
            or crosscheck_error > crosscheck_tolerance
            or not isinstance(
                self.metrics, CandidateJointStateObjectivePrecisionMetrics
            )
        ):
            raise ValueError("objective-precision family membership is invalid")
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(
            self,
            "maximum_direct_endpoint_crosscheck_abs_error",
            crosscheck_error,
        )
        object.__setattr__(
            self,
            "maximum_direct_endpoint_crosscheck_tolerance",
            crosscheck_tolerance,
        )
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
            "maximum_direct_endpoint_crosscheck_abs_error": (
                self.maximum_direct_endpoint_crosscheck_abs_error
            ),
            "maximum_direct_endpoint_crosscheck_tolerance": (
                self.maximum_direct_endpoint_crosscheck_tolerance
            ),
            "direct_endpoint_crosscheck_passed": True,
            **self.metrics.metadata(),
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _FAMILY_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(
            self.artifact_sha256,
            label="objective-precision family artifact",
        ):
            raise RuntimeError("candidate joint-state precision family drifted")


def classify_candidate_joint_state_objective_precision(
    *,
    closure_passed: bool,
    transport_rmse: float,
    finite_precision_rmse: float,
) -> str:
    """Classify relative diagnostic signals without assigning unique cause."""

    if type(closure_passed) is not bool:
        raise TypeError("closure_passed must be boolean")
    transport = _finite_float(
        transport_rmse, label="transport RMSE", nonnegative=True
    )
    precision = _finite_float(
        finite_precision_rmse,
        label="finite precision RMSE",
        nonnegative=True,
    )
    closure = "f64_closure_established" if closure_passed else "f64_closure_unresolved"
    tolerance = _DOMINANCE_TOLERANCE_FACTOR * max(1.0, transport, precision)
    if abs(transport - precision) <= tolerance:
        signal = "balanced_or_zero_signals"
    elif transport > precision:
        signal = "path_transport_signal_dominant"
    else:
        signal = "endpoint_precision_signal_dominant"
    return f"{closure}_{signal}"


@dataclass(frozen=True, slots=True)
class CandidateJointStateObjectivePrecisionComparison:
    evidence_example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    family_summaries: tuple[CandidateJointStateObjectivePrecisionFamilySummary, ...]
    supervised_token_count: int
    maximum_direct_endpoint_crosscheck_abs_error: float
    maximum_direct_endpoint_crosscheck_tolerance: float
    metrics: CandidateJointStateObjectivePrecisionMetrics
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        examples = tuple(
            _identifier(value, label="objective-precision example_id")
            for value in self.evidence_example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="objective-precision evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        families = tuple(self.family_summaries)
        crosscheck_error = _finite_float(
            self.maximum_direct_endpoint_crosscheck_abs_error,
            label="comparison direct endpoint cross-check error",
            nonnegative=True,
        )
        crosscheck_tolerance = _finite_float(
            self.maximum_direct_endpoint_crosscheck_tolerance,
            label="comparison direct endpoint cross-check tolerance",
            nonnegative=True,
        )
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or not families
            or any(
                not isinstance(
                    value, CandidateJointStateObjectivePrecisionFamilySummary
                )
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
            or crosscheck_error > crosscheck_tolerance
            or not isinstance(
                self.metrics, CandidateJointStateObjectivePrecisionMetrics
            )
        ):
            raise ValueError("objective-precision comparison membership is invalid")
        for family in families:
            family.validate_integrity()
        object.__setattr__(self, "evidence_example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "family_summaries", families)
        object.__setattr__(
            self,
            "maximum_direct_endpoint_crosscheck_abs_error",
            crosscheck_error,
        )
        object.__setattr__(
            self,
            "maximum_direct_endpoint_crosscheck_tolerance",
            crosscheck_tolerance,
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SUMMARY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def closure_gate_results(self) -> dict[str, bool]:
        return {
            "overall_closure_relative_RMSE_at_most_0_05": (
                self.metrics.closure_relative_rmse
                <= OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
            ),
            "every_family_closure_relative_RMSE_at_most_0_10": all(
                family.metrics.closure_relative_rmse
                <= FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                for family in self.family_summaries
            ),
            "overall_closure_cosine_at_least_0_99": (
                self.metrics.closure_cosine >= CLOSURE_COSINE_MINIMUM
            ),
        }

    @property
    def closure_passed(self) -> bool:
        return all(self.closure_gate_results.values())

    @property
    def classification(self) -> str:
        return classify_candidate_joint_state_objective_precision(
            closure_passed=self.closure_passed,
            transport_rmse=self.metrics.transport_rmse,
            finite_precision_rmse=self.metrics.finite_precision_rmse,
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
            **self.metrics.metadata(),
            "closure_gate_results": tuple(sorted(self.closure_gate_results.items())),
            "closure_passed": self.closure_passed,
            "classification": self.classification,
            "weighting": _WEIGHTING,
            "closure_orientation": "P64_GL4_minus_D64_finite",
            "transport_definition": "P64_GL4_minus_T64_scalar_endpoint",
            "finite_precision_definition": (
                "D64_direct_finite_minus_D32_legacy_V9_endpoint_replay"
            ),
            "closure_thresholds": {
                "overall_relative_RMSE_maximum": (
                    OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "every_family_relative_RMSE_maximum": (
                    FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                ),
                "overall_cosine_minimum": CLOSURE_COSINE_MINIMUM,
            },
            "classification_compares_signal_RMSE_not_unique_cause": True,
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
            self.artifact_sha256,
            label="objective-precision comparison artifact",
        ):
            raise RuntimeError("candidate joint-state precision comparison drifted")


def _canonical_evidence(
    evidence: Iterable[CandidateJointStateObjectivePrecisionEvidence],
) -> tuple[CandidateJointStateObjectivePrecisionEvidence, ...]:
    values = tuple(evidence)
    if not values or any(
        not isinstance(value, CandidateJointStateObjectivePrecisionEvidence)
        for value in values
    ):
        raise TypeError("objective-precision comparison requires typed evidence")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("objective-precision example ids must be unique")
    return ordered


def _nested_mean(
    evidence: Sequence[CandidateJointStateObjectivePrecisionEvidence],
    token_values: Mapping[str, Tensor],
) -> Tensor:
    if set(token_values) != {value.example_id for value in evidence}:
        raise ValueError("objective-precision statistic keys differ from evidence")
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
            raise ValueError("objective-precision statistic must be token aligned")
        statistic64 = statistic.detach().to(device="cpu", dtype=torch.float64)
        shape = tuple(int(size) for size in statistic64.shape[1:])
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("objective-precision statistic shapes differ")
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


def _cosine(*, cross: float, left_rms: float, right_rms: float) -> float:
    if left_rms <= _EPSILON and right_rms <= _EPSILON:
        return 1.0
    if left_rms <= _EPSILON or right_rms <= _EPSILON:
        return 0.0
    denominator = left_rms * right_rms
    return min(max(cross / denominator, -1.0), 1.0)


def _metrics(
    evidence: Sequence[CandidateJointStateObjectivePrecisionEvidence],
) -> CandidateJointStateObjectivePrecisionMetrics:
    d64 = {value.example_id: value.finite_delta_f64() for value in evidence}
    p64 = {value.example_id: value.path_integral_f64() for value in evidence}
    t64 = {value.example_id: value.scalar_tangent_f64() for value in evidence}
    d32 = {value.example_id: value.finite_delta_f32() for value in evidence}
    closure = {key: p64[key] - d64[key] for key in d64}
    transport = {key: p64[key] - t64[key] for key in d64}
    precision = {key: d64[key] - d32[key] for key in d64}

    def mean(values: Mapping[str, Tensor]) -> float:
        return float(_nested_mean(evidence, values))

    def second(values: Mapping[str, Tensor]) -> float:
        return mean({key: value.square() for key, value in values.items()})

    d64_rms = _rms(second(d64))
    p64_rms = _rms(second(p64))
    t64_rms = _rms(second(t64))
    d32_rms = _rms(second(d32))
    closure_rmse = _rms(second(closure))
    transport_rmse = _rms(second(transport))
    precision_rmse = _rms(second(precision))
    return CandidateJointStateObjectivePrecisionMetrics(
        mean_finite_delta_f64=mean(d64),
        mean_path_integral_f64=mean(p64),
        mean_scalar_tangent_f64=mean(t64),
        mean_finite_delta_f32=mean(d32),
        mean_path_minus_finite_f64=mean(closure),
        mean_path_minus_scalar_tangent_f64=mean(transport),
        mean_finite_f64_minus_f32=mean(precision),
        finite_delta_f64_rms=d64_rms,
        path_integral_f64_rms=p64_rms,
        scalar_tangent_f64_rms=t64_rms,
        finite_delta_f32_rms=d32_rms,
        closure_rmse=closure_rmse,
        closure_relative_rmse=_relative_rmse(closure_rmse, d64_rms),
        closure_cosine=_cosine(
            cross=mean({key: p64[key] * d64[key] for key in d64}),
            left_rms=p64_rms,
            right_rms=d64_rms,
        ),
        maximum_absolute_closure_error=max(
            float(value.abs().max()) for value in closure.values()
        ),
        transport_rmse=transport_rmse,
        transport_relative_rmse_to_finite_f64=_relative_rmse(
            transport_rmse, d64_rms
        ),
        transport_cosine=_cosine(
            cross=mean({key: p64[key] * t64[key] for key in d64}),
            left_rms=p64_rms,
            right_rms=t64_rms,
        ),
        finite_precision_rmse=precision_rmse,
        finite_precision_relative_rmse_to_finite_f64=_relative_rmse(
            precision_rmse, d64_rms
        ),
        relative_rmse_epsilon=_EPSILON,
    )


def summarize_candidate_joint_state_objective_precision(
    evidence: Iterable[CandidateJointStateObjectivePrecisionEvidence],
) -> CandidateJointStateObjectivePrecisionComparison:
    """Build immutable per-family and family-equal V10 precision summaries."""

    values = _canonical_evidence(evidence)
    by_family: dict[str, list[CandidateJointStateObjectivePrecisionEvidence]] = (
        defaultdict(list)
    )
    for value in values:
        by_family[value.family_id].append(value)
    families: list[CandidateJointStateObjectivePrecisionFamilySummary] = []
    for family_id in sorted(by_family):
        members = tuple(
            sorted(by_family[family_id], key=lambda value: value.example_id)
        )
        families.append(
            CandidateJointStateObjectivePrecisionFamilySummary(
                family_id=family_id,
                example_ids=tuple(value.example_id for value in members),
                evidence_artifact_sha256s=tuple(
                    value.artifact_sha256 for value in members
                ),
                supervised_token_count=sum(
                    value.supervised_tokens for value in members
                ),
                maximum_direct_endpoint_crosscheck_abs_error=max(
                    float(value.direct_endpoint_crosscheck_residual().abs().max())
                    for value in members
                ),
                maximum_direct_endpoint_crosscheck_tolerance=max(
                    value.direct_endpoint_crosscheck_tolerance for value in members
                ),
                metrics=_metrics(members),
            )
        )
    return CandidateJointStateObjectivePrecisionComparison(
        evidence_example_ids=tuple(value.example_id for value in values),
        evidence_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        family_summaries=tuple(families),
        supervised_token_count=sum(value.supervised_tokens for value in values),
        maximum_direct_endpoint_crosscheck_abs_error=max(
            float(value.direct_endpoint_crosscheck_residual().abs().max())
            for value in values
        ),
        maximum_direct_endpoint_crosscheck_tolerance=max(
            value.direct_endpoint_crosscheck_tolerance for value in values
        ),
        metrics=_metrics(values),
    )
