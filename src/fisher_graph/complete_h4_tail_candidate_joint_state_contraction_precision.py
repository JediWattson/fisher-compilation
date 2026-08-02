"""Pure V12 contraction-precision ladder for the complete-H4 path.

V11 established a small but non-zero difference between a true suffix JVP and
the V10 reverse-mode contraction.  This module changes no model execution.  It
replays the four authenticated V10 gradient banks and evaluates, in order, the
precision boundaries between that contraction and the live float32 suffix:

``P_v10``
    V10's control: GL4-integrate float64-lifted gradients, then contract with
    the float64 endpoint displacement.
``P64_node``
    Contract each node in float64, then GL4-integrate token contractions.
``P_dir``
    As above, with the direction rounded through the live float32 cast.
``P_prod``
    Multiply float32 gradient and direction coordinates, lift each product to
    float64, flatten in support-row-major/width-minor order, reduce with the
    bound typed ``torch.sum``, then GL4.
``P_live``
    Multiply in float32, use the same canonical flatten and a float32
    ``torch.sum``, lift the token contraction to float64, then GL4.  This is a
    counterfactual boundary replay, not a claim about an autograd kernel's
    hidden reduction schedule or serial reduction tree.

The first corrected stage whose JVP-relative RMSE is at most ``1e-4`` both
overall and in every family determines the ordered diagnosis.  Every stage is
published regardless of which one passes.  Raw tensors remain typed evidence
and never enter serialized metadata.
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
)
from .complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathNodeReceipt,
    candidate_joint_state_path_displacement,
)
from .complete_h4_tail_candidate_joint_state_suffix_jvp import (
    ADJOINT_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateSuffixJVPEvidence,
    CandidateJointStateSuffixJVPComparison,
    summarize_candidate_joint_state_suffix_jvp,
)
from .complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


__all__ = [
    "CONTRACTION_CONTROL_STAGE",
    "CONTRACTION_CORRECTED_STAGE_ORDER",
    "CONTRACTION_PUBLISHED_STAGE_ORDER",
    "CONTRACTION_TELESCOPE_POINTS",
    "CONTRACTION_TELESCOPE_TRANSITIONS",
    "FINITE_ADJOINT_RESIDUAL_FRACTION_MAXIMUM",
    "FINITE_TO_ADJOINT_RMSE_RATIO_MINIMUM",
    "CandidateJointStateContractionPrecisionComparison",
    "CandidateJointStateContractionPrecisionAccumulator",
    "CandidateJointStateContractionPrecisionEvidence",
    "CandidateJointStateContractionPrecisionFamilySummary",
    "CandidateJointStateContractionPrecisionFiniteMetrics",
    "CandidateJointStateContractionPrecisionNodeEvidence",
    "CandidateJointStateContractionPrecisionStageMetrics",
    "CandidateJointStateContractionPrecisionTelescopeMetrics",
    "CandidateJointStateContractionPrecisionTransitionMetrics",
    "build_candidate_joint_state_contraction_precision_evidence",
    "classify_candidate_joint_state_contraction_precision",
    "summarize_candidate_joint_state_contraction_precision",
]


CONTRACTION_CONTROL_STAGE = "P_v10"
CONTRACTION_CORRECTED_STAGE_ORDER = (
    "P64_node",
    "P_dir",
    "P_prod",
    "P_live",
)
CONTRACTION_PUBLISHED_STAGE_ORDER = (
    CONTRACTION_CONTROL_STAGE,
    *CONTRACTION_CORRECTED_STAGE_ORDER,
)
CONTRACTION_TELESCOPE_POINTS = (
    "P_v10",
    "P64_node",
    "P_dir",
    "P_prod",
    "P_live",
    "J64_suffix",
    "D64_finite",
)
CONTRACTION_TELESCOPE_TRANSITIONS = tuple(
    f"{target}_minus_{source}"
    for source, target in zip(
        CONTRACTION_TELESCOPE_POINTS[:-1],
        CONTRACTION_TELESCOPE_POINTS[1:],
        strict=True,
    )
)
FINITE_ADJOINT_RESIDUAL_FRACTION_MAXIMUM = 0.01
FINITE_TO_ADJOINT_RMSE_RATIO_MINIMUM = 100.0
_TELESCOPE_ABSOLUTE_TOLERANCE_FACTOR = 512.0 * torch.finfo(torch.float64).eps

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:candidate-joint-state-contraction-precision-tensor:v12\0"
_NODE_DOMAIN = b"fisher-graph:candidate-joint-state-contraction-precision-node:v12\0"
_EVIDENCE_DOMAIN = b"fisher-graph:candidate-joint-state-contraction-precision-evidence:v12\0"
_FAMILY_DOMAIN = b"fisher-graph:candidate-joint-state-contraction-precision-family:v12\0"
_SUMMARY_DOMAIN = b"fisher-graph:candidate-joint-state-contraction-precision-summary:v12\0"
_V9_TENSOR_DOMAIN = b"fisher-graph:candidate-joint-state-path-tensor:v9\0"
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


def _tensor_sha256(value: Tensor, *, domain: bytes = _TENSOR_DOMAIN) -> str:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise TypeError("hashed value must be a materialized strided tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(
        domain
        + _canonical_json_bytes(
            {
                "dtype": str(tensor.dtype),
                "shape": tuple(int(size) for size in tensor.shape),
            }
        )
        + payload
    ).hexdigest()


def _float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
    dtype: torch.dtype,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != ndim
        or 0 in value.shape
        or value.dtype != dtype
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{label} must be a finite nonempty rank-{ndim} {dtype} tensor"
        )
    return value.detach().to(device="cpu").clone().contiguous()


def _support_indices(value: Tensor, *, full_rows: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.numel() <= 0
        or value.dtype != torch.int64
        or value.requires_grad
    ):
        raise ValueError("support indices must be a nonempty int64 vector")
    result = value.detach().to(device="cpu").clone().contiguous()
    if (
        int(result[0]) < 0
        or int(result[-1]) >= full_rows
        or result.numel() > 1
        and not bool((result[1:] > result[:-1]).all())
    ):
        raise ValueError("support indices must be strictly increasing and in range")
    return result


def _same_float(left: float, right: float) -> bool:
    return float(left).hex() == float(right).hex()


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
    )


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionNodeEvidence:
    """Token contractions derived from one transient V10 gradient bank."""

    node_index: int
    path_fraction: float
    quadrature_weight: float
    gradient_shape: tuple[int, int, int]
    gradient_f64_sha256: str
    token_p64_node_f64: Tensor = field(repr=False)
    token_p_dir_f64: Tensor = field(repr=False)
    token_p_prod_f64: Tensor = field(repr=False)
    token_p_live_f64: Tensor = field(repr=False)
    pinned_v10_node_receipt_artifact_sha256: str
    pinned_v10_h4_gradient_sha256: str
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.node_index) is not int or not 0 <= self.node_index < 4:
            raise ValueError("contraction node index must be in [0, 3]")
        node = _finite_float(self.path_fraction, label="contraction path fraction")
        weight = _finite_float(
            self.quadrature_weight,
            label="contraction quadrature weight",
            nonnegative=True,
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[self.node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[self.node_index].hex()
        ):
            raise ValueError("contraction node does not use the exact GL4 rule")
        shape = self.gradient_shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise ValueError("contraction gradient shape must be positive rank three")
        gradient_hash = _require_sha256(
            self.gradient_f64_sha256,
            label="transient V12 gradient",
        )
        vectors = tuple(
            _float_tensor(
                getattr(self, name),
                label=name,
                ndim=1,
                dtype=torch.float64,
            )
            for name in (
                "token_p64_node_f64",
                "token_p_dir_f64",
                "token_p_prod_f64",
                "token_p_live_f64",
            )
        )
        if any(tuple(vector.shape) != (shape[0],) for vector in vectors):
            raise ValueError("contraction token vector geometry differs from gradient")
        receipt = _require_sha256(
            self.pinned_v10_node_receipt_artifact_sha256,
            label="pinned V10 node receipt",
        )
        pinned_gradient_hash = _require_sha256(
            self.pinned_v10_h4_gradient_sha256,
            label="pinned V10 H4 gradient",
        )
        object.__setattr__(self, "path_fraction", node)
        object.__setattr__(self, "quadrature_weight", weight)
        object.__setattr__(self, "gradient_shape", shape)
        object.__setattr__(self, "gradient_f64_sha256", gradient_hash)
        for name, vector in zip(
            (
                "token_p64_node_f64",
                "token_p_dir_f64",
                "token_p_prod_f64",
                "token_p_live_f64",
            ),
            vectors,
            strict=True,
        ):
            object.__setattr__(self, name, vector)
        object.__setattr__(self, "pinned_v10_node_receipt_artifact_sha256", receipt)
        object.__setattr__(self, "pinned_v10_h4_gradient_sha256", pinned_gradient_hash)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def token_vector_f64(self, stage: str) -> Tensor:
        names = {
            "P64_node": "token_p64_node_f64",
            "P_dir": "token_p_dir_f64",
            "P_prod": "token_p_prod_f64",
            "P_live": "token_p_live_f64",
        }
        if stage not in names:
            raise ValueError("node token vector requires a corrected stage")
        return getattr(self, names[stage]).clone().contiguous()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "node_index": self.node_index,
            "path_fraction_hex": self.path_fraction.hex(),
            "quadrature_weight_hex": self.quadrature_weight.hex(),
            "gradient_shape": self.gradient_shape,
            "gradient_f64_sha256": self.gradient_f64_sha256,
            "token_contractions": tuple(
                (
                    stage,
                    _tensor_sha256(self.token_vector_f64(stage)),
                    float(self.token_vector_f64(stage).mean()),
                    float(self.token_vector_f64(stage).square().mean().sqrt()),
                )
                for stage in CONTRACTION_CORRECTED_STAGE_ORDER
            ),
            "pinned_v10_node_receipt_artifact_sha256": (
                self.pinned_v10_node_receipt_artifact_sha256
            ),
            "pinned_v10_h4_gradient_sha256": self.pinned_v10_h4_gradient_sha256,
            "gradient_f64_is_exact_f32_lift": True,
            "support_row_major_width_minor_order": True,
            "transient_gradient_retained": False,
            "raw_gradient_or_token_vectors_serialized": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if (
            any(
                vector.dtype != torch.float64
                or vector.device.type != "cpu"
                or vector.ndim != 1
                or tuple(vector.shape) != (self.gradient_shape[0],)
                or vector.requires_grad
                or not vector.is_contiguous()
                or not bool(torch.isfinite(vector).all())
                for vector in (
                    self.token_p64_node_f64,
                    self.token_p_dir_f64,
                    self.token_p_prod_f64,
                    self.token_p_live_f64,
                )
            )
            or _sha256(_NODE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="contraction node artifact")
        ):
            raise RuntimeError("candidate joint-state contraction node drifted")


def _fixed_order_node_contractions(
    gradient64: Tensor,
    displacement64: Tensor,
    cast_tangent32: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flatten support-row-major/width-minor, then use typed torch reductions."""

    tokens = int(gradient64.shape[0])
    flat_gradient64 = gradient64.reshape(tokens, -1)
    flat_gradient32 = gradient64.to(torch.float32).reshape(tokens, -1)
    flat_displacement64 = displacement64.reshape(-1)
    flat_cast32 = cast_tangent32.reshape(-1)
    product32 = (flat_gradient32 * flat_cast32.unsqueeze(0)).contiguous()
    return (
        torch.sum(
            flat_gradient64 * flat_displacement64.unsqueeze(0),
            dim=1,
            dtype=torch.float64,
        ).contiguous(),
        torch.sum(
            flat_gradient64
            * flat_cast32.to(torch.float64).unsqueeze(0),
            dim=1,
            dtype=torch.float64,
        ).contiguous(),
        torch.sum(
            product32.to(torch.float64), dim=1, dtype=torch.float64
        ).contiguous(),
        torch.sum(product32, dim=1, dtype=torch.float32)
        .to(torch.float64)
        .contiguous(),
    )


def _integrate_node_vectors(
    nodes: Sequence[CandidateJointStateContractionPrecisionNodeEvidence],
) -> dict[str, Tensor]:
    outputs = {
        stage: torch.zeros(nodes[0].gradient_shape[0], dtype=torch.float64)
        for stage in CONTRACTION_CORRECTED_STAGE_ORDER
    }
    for node in nodes:
        for stage in CONTRACTION_CORRECTED_STAGE_ORDER:
            outputs[stage] = (
                outputs[stage]
                + node.quadrature_weight * node.token_vector_f64(stage)
            ).contiguous()
    return outputs


class CandidateJointStateContractionPrecisionAccumulator:
    """Stream transient V10 banks into bounded V12 contraction evidence."""

    __slots__ = (
        "_full_cast_tangent32",
        "_full_displacement64",
        "_integrated_gradient64",
        "_nodes",
        "_sealed",
        "_support_cast_tangent32",
        "_support_displacement64",
        "_support_indices",
        "_token_p_v10_f64",
    )

    def __init__(
        self,
        *,
        support_indices: Tensor,
        full_displacement_f64: Tensor,
        full_cast_tangent_f32: Tensor,
    ) -> None:
        displacement = _float_tensor(
            full_displacement_f64,
            label="full H4 displacement",
            ndim=2,
            dtype=torch.float64,
        )
        tangent = _float_tensor(
            full_cast_tangent_f32,
            label="full H4 cast tangent",
            ndim=2,
            dtype=torch.float32,
        )
        if displacement.shape != tangent.shape:
            raise ValueError("full displacement and cast tangent geometry differs")
        indices = _support_indices(support_indices, full_rows=displacement.shape[0])
        if not _bitwise_equal(tangent, displacement.to(torch.float32).contiguous()):
            raise ValueError("full cast tangent is not the exact float32 delta cast")
        mask = torch.zeros(displacement.shape[0], dtype=torch.bool)
        mask.index_fill_(0, indices, True)
        if (
            bool((displacement[~mask] != 0.0).any())
            or bool((tangent[~mask] != 0.0).any())
        ):
            raise ValueError("H4 direction must be exactly zero outside support")
        self._support_indices = indices
        self._full_displacement64 = displacement
        self._full_cast_tangent32 = tangent
        self._support_displacement64 = displacement.index_select(
            0, indices
        ).contiguous()
        self._support_cast_tangent32 = tangent.index_select(0, indices).contiguous()
        self._integrated_gradient64: Tensor | None = None
        self._token_p_v10_f64: Tensor | None = None
        self._nodes: list[CandidateJointStateContractionPrecisionNodeEvidence] = []
        self._sealed = False

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def add_node(
        self,
        *,
        node_receipt: CandidateJointStatePathNodeReceipt,
        token_support_h4_gradients_f64: Tensor,
    ) -> CandidateJointStateContractionPrecisionNodeEvidence:
        if self._sealed:
            raise RuntimeError("contraction precision accumulator is sealed")
        if not isinstance(node_receipt, CandidateJointStatePathNodeReceipt):
            raise TypeError("contraction precision node requires a V10 path receipt")
        node_receipt.validate_integrity()
        node_index = len(self._nodes)
        if node_receipt.node_index != node_index or node_index >= 4:
            raise ValueError("contraction precision nodes must be added in GL4 order")
        gradient = _float_tensor(
            token_support_h4_gradients_f64,
            label="support H4 gradient",
            ndim=3,
            dtype=torch.float64,
        )
        expected_tail = (
            int(self._support_indices.numel()),
            int(self._full_displacement64.shape[1]),
        )
        if tuple(gradient.shape[1:]) != expected_tail:
            raise ValueError("support H4 gradient geometry differs from direction")
        if self._integrated_gradient64 is not None and (
            gradient.shape != self._integrated_gradient64.shape
        ):
            raise ValueError("support H4 gradient geometry differs between nodes")
        if not torch.equal(gradient, gradient.to(torch.float32).to(torch.float64)):
            raise ValueError("support H4 gradient is not an exact float32 lift")
        pinned_hash = _tensor_sha256(gradient, domain=_V9_TENSOR_DOMAIN)
        if (
            pinned_hash != node_receipt.h4_gradient_sha256
            or tuple(gradient.shape) != node_receipt.h4_gradient_shape
        ):
            raise ValueError("support H4 gradient differs from the V10 receipt")
        contractions = _fixed_order_node_contractions(
            gradient,
            self._support_displacement64,
            self._support_cast_tangent32,
        )
        node = CandidateJointStateContractionPrecisionNodeEvidence(
            node_index=node_receipt.node_index,
            path_fraction=node_receipt.path_fraction,
            quadrature_weight=node_receipt.quadrature_weight,
            gradient_shape=tuple(int(size) for size in gradient.shape),
            gradient_f64_sha256=_tensor_sha256(gradient),
            token_p64_node_f64=contractions[0],
            token_p_dir_f64=contractions[1],
            token_p_prod_f64=contractions[2],
            token_p_live_f64=contractions[3],
            pinned_v10_node_receipt_artifact_sha256=node_receipt.artifact_sha256,
            pinned_v10_h4_gradient_sha256=pinned_hash,
        )
        if self._integrated_gradient64 is None:
            integrated = torch.zeros_like(gradient)
        else:
            integrated = self._integrated_gradient64
        updated = (
            integrated + node_receipt.quadrature_weight * gradient
        ).contiguous()
        self._integrated_gradient64 = updated
        self._nodes.append(node)
        if node_index == 3:
            self._token_p_v10_f64 = torch.einsum(
                "rw,trw->t", self._support_displacement64, updated
            ).contiguous()
            self._integrated_gradient64 = None
        return node

    def finalize(
        self,
        *,
        suffix_jvp_evidence: CandidateJointStateSuffixJVPEvidence,
    ) -> CandidateJointStateContractionPrecisionEvidence:
        if self._sealed:
            raise RuntimeError("contraction precision accumulator is sealed")
        if len(self._nodes) != 4 or self._token_p_v10_f64 is None:
            raise RuntimeError("contraction precision accumulator requires four nodes")
        result = CandidateJointStateContractionPrecisionEvidence(
            suffix_jvp_evidence=suffix_jvp_evidence,
            support_indices=self._support_indices,
            full_displacement_f64=self._full_displacement64,
            full_cast_tangent_f32=self._full_cast_tangent32,
            nodes=tuple(self._nodes),
            token_p_v10_f64=self._token_p_v10_f64,
            pinned_v11_evidence_artifact_sha256=suffix_jvp_evidence.artifact_sha256,
        )
        # Returned evidence retains only token contractions and authenticated
        # hashes.  The one streaming integrated bank is deliberately released.
        self._token_p_v10_f64 = None
        self._sealed = True
        return result


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionEvidence:
    """One prompt's V12 ladder bound to exact V11 and V10 authorities."""

    suffix_jvp_evidence: CandidateJointStateSuffixJVPEvidence = field(repr=False)
    support_indices: Tensor = field(repr=False)
    full_displacement_f64: Tensor = field(repr=False)
    full_cast_tangent_f32: Tensor = field(repr=False)
    nodes: tuple[CandidateJointStateContractionPrecisionNodeEvidence, ...]
    token_p_v10_f64: Tensor = field(repr=False)
    pinned_v11_evidence_artifact_sha256: str
    example_id: str = field(init=False)
    family_id: str = field(init=False)
    _stage_vectors: tuple[Tensor, ...] = field(init=False, repr=False)
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.suffix_jvp_evidence, CandidateJointStateSuffixJVPEvidence):
            raise TypeError("contraction precision requires V11 suffix JVP evidence")
        self.suffix_jvp_evidence.validate_integrity()
        pinned = _require_sha256(
            self.pinned_v11_evidence_artifact_sha256,
            label="pinned V11 evidence artifact",
        )
        if pinned != self.suffix_jvp_evidence.artifact_sha256:
            raise ValueError("pinned V11 evidence artifact differs")
        displacement = _float_tensor(
            self.full_displacement_f64,
            label="full H4 displacement",
            ndim=2,
            dtype=torch.float64,
        )
        tangent = _float_tensor(
            self.full_cast_tangent_f32,
            label="full H4 cast tangent",
            ndim=2,
            dtype=torch.float32,
        )
        if displacement.shape != tangent.shape:
            raise ValueError("full displacement and cast tangent geometry differs")
        indices = _support_indices(self.support_indices, full_rows=displacement.shape[0])
        nodes = tuple(self.nodes)
        p_v10 = _float_tensor(
            self.token_p_v10_f64,
            label="P_v10 token contraction",
            ndim=1,
            dtype=torch.float64,
        )
        if (
            len(nodes) != 4
            or any(
                not isinstance(node, CandidateJointStateContractionPrecisionNodeEvidence)
                or node.node_index != index
                for index, node in enumerate(nodes)
            )
        ):
            raise ValueError("contraction precision requires ordered four GL4 nodes")
        precision = self.suffix_jvp_evidence.precision_evidence
        path = precision.path_evidence
        expected_shape = (
            precision.supervised_tokens,
            int(indices.numel()),
            int(displacement.shape[1]),
        )
        if path.h4_shape != expected_shape[1:]:
            raise ValueError("support geometry differs from V10 H4 evidence")
        if tuple(p_v10.shape) != (expected_shape[0],):
            raise ValueError("P_v10 token geometry differs from V11")
        for node, receipt, suffix_node in zip(
            nodes, path.node_receipts, self.suffix_jvp_evidence.nodes, strict=True
        ):
            node.validate_integrity()
            if (
                node.gradient_shape != expected_shape
                or node.path_fraction.hex() != receipt.path_fraction.hex()
                or node.quadrature_weight.hex() != receipt.quadrature_weight.hex()
                or node.pinned_v10_node_receipt_artifact_sha256
                != receipt.artifact_sha256
                or node.pinned_v10_node_receipt_artifact_sha256
                != suffix_node.pinned_v10_node_receipt_artifact_sha256
                or node.pinned_v10_h4_gradient_sha256 != receipt.h4_gradient_sha256
            ):
                raise ValueError("contraction node provenance or geometry differs")
        support_displacement = displacement.index_select(0, indices).contiguous()
        expected_displacement = candidate_joint_state_path_displacement(path)
        if not _bitwise_equal(support_displacement, expected_displacement):
            raise ValueError("full displacement support rows differ from V10")
        expected_tangent = displacement.to(torch.float32).contiguous()
        if not _bitwise_equal(tangent, expected_tangent):
            raise ValueError("full cast tangent is not the exact float32 delta cast")
        support_mask = torch.zeros(displacement.shape[0], dtype=torch.bool)
        support_mask.index_fill_(0, indices, True)
        if (
            bool((displacement[~support_mask] != 0.0).any())
            or bool((tangent[~support_mask] != 0.0).any())
        ):
            raise ValueError("H4 direction must be exactly zero outside support")
        corrected = _integrate_node_vectors(nodes)
        ladder = {CONTRACTION_CONTROL_STAGE: p_v10, **corrected}
        wrapped_v10 = self.suffix_jvp_evidence.replayed_vjp_integral_f64()
        if not _bitwise_equal(ladder[CONTRACTION_CONTROL_STAGE], wrapped_v10):
            raise ValueError("P_v10 did not exactly replay the wrapped V10 contraction")
        object.__setattr__(self, "support_indices", indices)
        object.__setattr__(self, "full_displacement_f64", displacement)
        object.__setattr__(self, "full_cast_tangent_f32", tangent)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "token_p_v10_f64", p_v10)
        object.__setattr__(self, "pinned_v11_evidence_artifact_sha256", pinned)
        object.__setattr__(self, "example_id", self.suffix_jvp_evidence.example_id)
        object.__setattr__(self, "family_id", self.suffix_jvp_evidence.family_id)
        object.__setattr__(
            self,
            "_stage_vectors",
            tuple(
                ladder[stage].clone().contiguous()
                for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
            ),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return self.suffix_jvp_evidence.supervised_tokens

    @property
    def full_h4_row_count(self) -> int:
        return int(self.full_displacement_f64.shape[0])

    @property
    def support_row_count(self) -> int:
        return int(self.support_indices.numel())

    @property
    def outside_support_row_count(self) -> int:
        return self.full_h4_row_count - self.support_row_count

    @property
    def h4_width(self) -> int:
        return int(self.full_displacement_f64.shape[1])

    def stage_vector_f64(self, stage: str) -> Tensor:
        if stage not in CONTRACTION_PUBLISHED_STAGE_ORDER:
            raise ValueError("unknown contraction precision stage")
        return (
            self._stage_vectors[CONTRACTION_PUBLISHED_STAGE_ORDER.index(stage)]
            .clone()
            .contiguous()
        )

    def suffix_jvp_f64(self) -> Tensor:
        return self.suffix_jvp_evidence.integrated_suffix_jvp_f64()

    def finite_delta_f64(self) -> Tensor:
        return self.suffix_jvp_evidence.finite_delta_f64()

    @property
    def resource_accounting(self) -> dict[str, int]:
        node_support_elements = (
            4 * self.supervised_tokens * self.support_row_count * self.h4_width
        )
        nodewise_stage_count = len(CONTRACTION_CORRECTED_STAGE_ORDER)
        return {
            "quadrature_node_count": 4,
            "supervised_token_count": self.supervised_tokens,
            "full_h4_row_count": self.full_h4_row_count,
            "support_h4_row_count": self.support_row_count,
            "outside_support_h4_row_count": self.outside_support_row_count,
            "h4_width": self.h4_width,
            "gradient_f64_to_f32_roundtrip_validation_element_count": (
                node_support_elements
            ),
            "full_direction_cast_validation_element_count": (
                self.full_h4_row_count * self.h4_width
            ),
            "outside_support_zero_validation_element_count": (
                2 * self.outside_support_row_count * self.h4_width
            ),
            "v10_gradient_weighted_add_element_count": node_support_elements,
            "v10_final_contraction_product_count": (
                self.supervised_tokens * self.support_row_count * self.h4_width
            ),
            "nodewise_contraction_stage_count": nodewise_stage_count,
            "nodewise_contraction_coordinate_observation_count_per_stage": (
                node_support_elements
            ),
            "nodewise_contraction_coordinate_observation_count_total": (
                nodewise_stage_count * node_support_elements
            ),
            "actual_coordinate_product_bank_count": 3,
            "actual_coordinate_product_count_total": 3 * node_support_elements,
            "nodewise_GL4_token_weight_application_count": (
                nodewise_stage_count * 4 * self.supervised_tokens
            ),
            "published_stage_count": len(CONTRACTION_PUBLISHED_STAGE_ORDER),
            "fresh_model_forward_count": 0,
            "fresh_model_backward_count": 0,
        }

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        stage_vectors = {
            stage: self.stage_vector_f64(stage)
            for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
        }
        jvp = self.suffix_jvp_f64()
        finite = self.finite_delta_f64()
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "pinned_v11_evidence_artifact_sha256": (
                self.pinned_v11_evidence_artifact_sha256
            ),
            "support_indices_sha256": _tensor_sha256(self.support_indices),
            "full_displacement_f64_sha256": _tensor_sha256(self.full_displacement_f64),
            "full_cast_tangent_f32_sha256": _tensor_sha256(self.full_cast_tangent_f32),
            "nodes": tuple(node.metadata() for node in self.nodes),
            "stage_vectors": tuple(
                (
                    stage,
                    _tensor_sha256(stage_vectors[stage]),
                    float(stage_vectors[stage].mean()),
                    float(stage_vectors[stage].square().mean().sqrt()),
                )
                for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
            ),
            "suffix_jvp_f64_sha256": _tensor_sha256(jvp),
            "finite_delta_f64_sha256": _tensor_sha256(finite),
            "resource_accounting": tuple(sorted(self.resource_accounting.items())),
            "P_v10_replayed_bitwise": True,
            "gradient_f64_is_exact_f32_lift": True,
            "cast_tangent_is_exact_f64_to_f32_delta_cast": True,
            "direction_is_exactly_zero_outside_support": True,
            "reduction_order": (
                "canonical_contiguous_support_row_major_width_minor_flatten_"
                "then_typed_torch_sum"
            ),
            "GL4_integration_dtype": str(torch.float64),
            "P_live_is_counterfactual_not_internal_VJP_schedule_proof": True,
            "P_prod_and_P_live_share_one_f32_product_bank": True,
            "resource_counts_are_not_FLOPs_or_total_model_compute": True,
            "raw_tensors_serialized": False,
            "fits_selects_or_routes_candidates": False,
            "authorizes_serving_compression_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        self.suffix_jvp_evidence.validate_integrity()
        try:
            if (
                self.pinned_v11_evidence_artifact_sha256
                != self.suffix_jvp_evidence.artifact_sha256
                or self.support_indices.dtype != torch.int64
                or self.support_indices.device.type != "cpu"
                or not self.support_indices.is_contiguous()
                or self.full_displacement_f64.dtype != torch.float64
                or self.full_cast_tangent_f32.dtype != torch.float32
                or self.token_p_v10_f64.dtype != torch.float64
                or not self.full_displacement_f64.is_contiguous()
                or not self.full_cast_tangent_f32.is_contiguous()
                or not self.token_p_v10_f64.is_contiguous()
                or len(self.nodes) != 4
                or len(self._stage_vectors) != len(CONTRACTION_PUBLISHED_STAGE_ORDER)
            ):
                raise RuntimeError
            for node in self.nodes:
                node.validate_integrity()
            mask = torch.zeros(self.full_h4_row_count, dtype=torch.bool)
            mask.index_fill_(0, self.support_indices, True)
            if (
                not torch.equal(
                    self.full_cast_tangent_f32,
                    self.full_displacement_f64.to(torch.float32),
                )
                or bool((self.full_displacement_f64[~mask] != 0.0).any())
                or bool((self.full_cast_tangent_f32[~mask] != 0.0).any())
            ):
                raise RuntimeError
            support64 = self.full_displacement_f64.index_select(
                0, self.support_indices
            ).contiguous()
            expected_support = candidate_joint_state_path_displacement(
                self.suffix_jvp_evidence.precision_evidence.path_evidence
            )
            if not _bitwise_equal(support64, expected_support):
                raise RuntimeError
            replay = {
                CONTRACTION_CONTROL_STAGE: self.token_p_v10_f64,
                **_integrate_node_vectors(self.nodes),
            }
            for stage, stored in zip(
                CONTRACTION_PUBLISHED_STAGE_ORDER, self._stage_vectors, strict=True
            ):
                if not _bitwise_equal(replay[stage], stored):
                    raise RuntimeError
            if not _bitwise_equal(
                replay[CONTRACTION_CONTROL_STAGE],
                self.suffix_jvp_evidence.replayed_vjp_integral_f64(),
            ):
                raise RuntimeError
            resources = self.resource_accounting
            node_elements = 4 * self.supervised_tokens * self.support_row_count * self.h4_width
            if (
                resources["gradient_f64_to_f32_roundtrip_validation_element_count"]
                != node_elements
                or resources["v10_gradient_weighted_add_element_count"] != node_elements
                or resources[
                    "nodewise_contraction_coordinate_observation_count_per_stage"
                ]
                != node_elements
                or resources[
                    "nodewise_contraction_coordinate_observation_count_total"
                ]
                != len(CONTRACTION_CORRECTED_STAGE_ORDER) * node_elements
                or resources["actual_coordinate_product_bank_count"] != 3
                or resources["actual_coordinate_product_count_total"]
                != 3 * node_elements
                or resources["full_h4_row_count"]
                != resources["support_h4_row_count"]
                + resources["outside_support_h4_row_count"]
                or resources["fresh_model_forward_count"] != 0
                or resources["fresh_model_backward_count"] != 0
            ):
                raise RuntimeError
            if (
                _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False))
                != _require_sha256(
                    self.artifact_sha256, label="contraction precision evidence artifact"
                )
            ):
                raise RuntimeError
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "candidate joint-state contraction precision evidence drifted"
            ) from error


def build_candidate_joint_state_contraction_precision_evidence(
    *,
    suffix_jvp_evidence: CandidateJointStateSuffixJVPEvidence,
    support_indices: Tensor,
    full_displacement_f64: Tensor,
    full_cast_tangent_f32: Tensor,
    node_support_h4_gradients_f64: Sequence[Tensor],
) -> CandidateJointStateContractionPrecisionEvidence:
    """Bind transient V10 banks to V11 and construct one immutable V12 row."""

    if not isinstance(suffix_jvp_evidence, CandidateJointStateSuffixJVPEvidence):
        raise TypeError("contraction precision builder requires V11 evidence")
    gradients = tuple(node_support_h4_gradients_f64)
    if len(gradients) != 4:
        raise ValueError("contraction precision builder requires four gradient banks")
    accumulator = CandidateJointStateContractionPrecisionAccumulator(
        support_indices=support_indices,
        full_displacement_f64=full_displacement_f64,
        full_cast_tangent_f32=full_cast_tangent_f32,
    )
    path = suffix_jvp_evidence.precision_evidence.path_evidence
    for receipt, gradient in zip(path.node_receipts, gradients, strict=True):
        accumulator.add_node(
            node_receipt=receipt,
            token_support_h4_gradients_f64=gradient,
        )
    return accumulator.finalize(suffix_jvp_evidence=suffix_jvp_evidence)


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionStageMetrics:
    stage: str
    mean_stage_f64: float
    stage_f64_rms: float
    mean_jvp_minus_stage_f64: float
    adjoint_rmse: float
    adjoint_relative_rmse: float
    adjoint_cosine: float
    maximum_absolute_adjoint_error: float
    mean_stage_minus_finite_f64: float
    closure_rmse: float
    closure_relative_rmse: float
    closure_cosine: float
    maximum_absolute_closure_error: float
    relative_rmse_epsilon: float

    def __post_init__(self) -> None:
        stage = _identifier(self.stage, label="contraction precision stage")
        if stage not in CONTRACTION_PUBLISHED_STAGE_ORDER:
            raise ValueError("contraction precision stage is not published")
        signed = {
            "mean_stage_f64",
            "mean_jvp_minus_stage_f64",
            "mean_stage_minus_finite_f64",
            "adjoint_cosine",
            "closure_cosine",
        }
        for name in self.__dataclass_fields__:
            if name == "stage":
                continue
            value = _finite_float(
                getattr(self, name),
                label=f"contraction precision metric {name}",
                nonnegative=name not in signed,
            )
            if name in {"adjoint_cosine", "closure_cosine"} and not -1.0 <= value <= 1.0:
                raise ValueError(f"contraction precision {name} must be in [-1, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "stage", stage)

    def metadata(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionFiniteMetrics:
    mean_suffix_jvp_f64: float
    suffix_jvp_f64_rms: float
    mean_finite_delta_f64: float
    finite_delta_f64_rms: float
    mean_jvp_minus_finite_f64: float
    closure_rmse: float
    closure_relative_rmse: float
    closure_cosine: float
    maximum_absolute_closure_error: float
    relative_rmse_epsilon: float

    def __post_init__(self) -> None:
        signed = {
            "mean_suffix_jvp_f64",
            "mean_finite_delta_f64",
            "mean_jvp_minus_finite_f64",
            "closure_cosine",
        }
        for name in self.__dataclass_fields__:
            value = _finite_float(
                getattr(self, name),
                label=f"contraction finite metric {name}",
                nonnegative=name not in signed,
            )
            if name == "closure_cosine" and not -1.0 <= value <= 1.0:
                raise ValueError("contraction finite closure cosine must be in [-1, 1]")
            object.__setattr__(self, name, value)

    def metadata(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionTransitionMetrics:
    transition: str
    source: str
    target: str
    mean_delta_f64: float
    delta_f64_rms: float
    maximum_absolute_delta: float

    def __post_init__(self) -> None:
        transition = _identifier(
            self.transition, label="contraction telescope transition"
        )
        source = _identifier(self.source, label="contraction telescope source")
        target = _identifier(self.target, label="contraction telescope target")
        if (
            transition not in CONTRACTION_TELESCOPE_TRANSITIONS
            or source not in CONTRACTION_TELESCOPE_POINTS
            or target not in CONTRACTION_TELESCOPE_POINTS
            or transition != f"{target}_minus_{source}"
            or CONTRACTION_TELESCOPE_POINTS.index(target)
            != CONTRACTION_TELESCOPE_POINTS.index(source) + 1
        ):
            raise ValueError("contraction telescope transition is not canonical")
        object.__setattr__(
            self,
            "mean_delta_f64",
            _finite_float(self.mean_delta_f64, label="transition mean delta"),
        )
        object.__setattr__(
            self,
            "delta_f64_rms",
            _finite_float(
                self.delta_f64_rms,
                label="transition delta RMS",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_absolute_delta",
            _finite_float(
                self.maximum_absolute_delta,
                label="transition maximum absolute delta",
                nonnegative=True,
            ),
        )

    def metadata(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionTelescopeMetrics:
    mean_residual_f64: float
    residual_f64_rmse: float
    maximum_absolute_residual: float
    absolute_tolerance: float
    maximum_point_magnitude: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_residual_f64",
            _finite_float(self.mean_residual_f64, label="telescope mean residual"),
        )
        for name in (
            "residual_f64_rmse",
            "maximum_absolute_residual",
            "absolute_tolerance",
            "maximum_point_magnitude",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    label=f"telescope {name}",
                    nonnegative=True,
                ),
            )
        if self.maximum_absolute_residual > self.absolute_tolerance:
            raise ValueError("contraction telescope does not close within tolerance")

    @property
    def passed(self) -> bool:
        return self.maximum_absolute_residual <= self.absolute_tolerance

    def metadata(self) -> dict[str, object]:
        return {
            "mean_residual_f64": self.mean_residual_f64,
            "residual_f64_rmse": self.residual_f64_rmse,
            "maximum_absolute_residual": self.maximum_absolute_residual,
            "absolute_tolerance": self.absolute_tolerance,
            "maximum_point_magnitude": self.maximum_point_magnitude,
            "passed": self.passed,
            "orientation": (
                "sum_adjacent_deltas_minus_(D64_finite_minus_P_v10)"
            ),
            "absolute_tolerance_factor": _TELESCOPE_ABSOLUTE_TOLERANCE_FACTOR,
        }


def _nested_mean(
    evidence: Sequence[CandidateJointStateContractionPrecisionEvidence],
    token_values: Mapping[str, Tensor],
) -> Tensor:
    if set(token_values) != {value.example_id for value in evidence}:
        raise ValueError("contraction precision statistic keys differ from evidence")
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
            raise ValueError("contraction precision statistic must be token aligned")
        statistic64 = statistic.detach().to(device="cpu", dtype=torch.float64)
        shape = tuple(int(size) for size in statistic64.shape[1:])
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("contraction precision statistic shapes differ")
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


def _telescope_points(
    value: CandidateJointStateContractionPrecisionEvidence,
) -> dict[str, Tensor]:
    return {
        **{
            stage: value.stage_vector_f64(stage)
            for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
        },
        "J64_suffix": value.suffix_jvp_f64(),
        "D64_finite": value.finite_delta_f64(),
    }


def _transition_vectors(
    value: CandidateJointStateContractionPrecisionEvidence,
) -> dict[str, Tensor]:
    points = _telescope_points(value)
    return {
        transition: (points[target] - points[source]).contiguous()
        for transition, source, target in zip(
            CONTRACTION_TELESCOPE_TRANSITIONS,
            CONTRACTION_TELESCOPE_POINTS[:-1],
            CONTRACTION_TELESCOPE_POINTS[1:],
            strict=True,
        )
    }


def _transition_metrics(
    evidence: Sequence[CandidateJointStateContractionPrecisionEvidence],
) -> tuple[CandidateJointStateContractionPrecisionTransitionMetrics, ...]:
    by_example = {
        value.example_id: _transition_vectors(value) for value in evidence
    }
    result: list[CandidateJointStateContractionPrecisionTransitionMetrics] = []
    for transition, source, target in zip(
        CONTRACTION_TELESCOPE_TRANSITIONS,
        CONTRACTION_TELESCOPE_POINTS[:-1],
        CONTRACTION_TELESCOPE_POINTS[1:],
        strict=True,
    ):
        vectors = {
            example: values[transition] for example, values in by_example.items()
        }
        mean = float(_nested_mean(evidence, vectors))
        second = float(
            _nested_mean(
                evidence, {key: vector.square() for key, vector in vectors.items()}
            )
        )
        result.append(
            CandidateJointStateContractionPrecisionTransitionMetrics(
                transition=transition,
                source=source,
                target=target,
                mean_delta_f64=mean,
                delta_f64_rms=_rms(second),
                maximum_absolute_delta=max(
                    float(vector.abs().max()) for vector in vectors.values()
                ),
            )
        )
    return tuple(result)


def _telescope_metrics(
    evidence: Sequence[CandidateJointStateContractionPrecisionEvidence],
) -> CandidateJointStateContractionPrecisionTelescopeMetrics:
    residuals: dict[str, Tensor] = {}
    maximum_point = 0.0
    for value in evidence:
        points = _telescope_points(value)
        transitions = _transition_vectors(value)
        summed = torch.zeros(value.supervised_tokens, dtype=torch.float64)
        for transition in CONTRACTION_TELESCOPE_TRANSITIONS:
            summed = (summed + transitions[transition]).contiguous()
        target = (
            points["D64_finite"] - points[CONTRACTION_CONTROL_STAGE]
        ).contiguous()
        residuals[value.example_id] = (summed - target).contiguous()
        maximum_point = max(
            maximum_point,
            *(float(point.abs().max()) for point in points.values()),
        )
    mean = float(_nested_mean(evidence, residuals))
    second = float(
        _nested_mean(
            evidence, {key: value.square() for key, value in residuals.items()}
        )
    )
    tolerance = _TELESCOPE_ABSOLUTE_TOLERANCE_FACTOR * max(1.0, maximum_point)
    return CandidateJointStateContractionPrecisionTelescopeMetrics(
        mean_residual_f64=mean,
        residual_f64_rmse=_rms(second),
        maximum_absolute_residual=max(
            float(value.abs().max()) for value in residuals.values()
        ),
        absolute_tolerance=tolerance,
        maximum_point_magnitude=maximum_point,
    )


def _stage_metrics(
    evidence: Sequence[CandidateJointStateContractionPrecisionEvidence],
    stage: str,
) -> CandidateJointStateContractionPrecisionStageMetrics:
    values = {value.example_id: value.stage_vector_f64(stage) for value in evidence}
    jvp = {value.example_id: value.suffix_jvp_f64() for value in evidence}
    finite = {value.example_id: value.finite_delta_f64() for value in evidence}
    adjoint = {key: jvp[key] - values[key] for key in values}
    closure = {key: values[key] - finite[key] for key in values}

    def mean(items: Mapping[str, Tensor]) -> float:
        return float(_nested_mean(evidence, items))

    def second(items: Mapping[str, Tensor]) -> float:
        return mean({key: value.square() for key, value in items.items()})

    value_rms = _rms(second(values))
    jvp_rms = _rms(second(jvp))
    finite_rms = _rms(second(finite))
    adjoint_rmse = _rms(second(adjoint))
    closure_rmse = _rms(second(closure))
    return CandidateJointStateContractionPrecisionStageMetrics(
        stage=stage,
        mean_stage_f64=mean(values),
        stage_f64_rms=value_rms,
        mean_jvp_minus_stage_f64=mean(adjoint),
        adjoint_rmse=adjoint_rmse,
        adjoint_relative_rmse=_symmetric_relative_rmse(
            adjoint_rmse, jvp_rms, value_rms
        ),
        adjoint_cosine=_cosine(
            cross=mean({key: jvp[key] * values[key] for key in values}),
            left_rms=jvp_rms,
            right_rms=value_rms,
        ),
        maximum_absolute_adjoint_error=max(
            float(value.abs().max()) for value in adjoint.values()
        ),
        mean_stage_minus_finite_f64=mean(closure),
        closure_rmse=closure_rmse,
        closure_relative_rmse=_relative_rmse(closure_rmse, finite_rms),
        closure_cosine=_cosine(
            cross=mean({key: values[key] * finite[key] for key in values}),
            left_rms=value_rms,
            right_rms=finite_rms,
        ),
        maximum_absolute_closure_error=max(
            float(value.abs().max()) for value in closure.values()
        ),
        relative_rmse_epsilon=_EPSILON,
    )


def _finite_metrics(
    evidence: Sequence[CandidateJointStateContractionPrecisionEvidence],
) -> CandidateJointStateContractionPrecisionFiniteMetrics:
    jvp = {value.example_id: value.suffix_jvp_f64() for value in evidence}
    finite = {value.example_id: value.finite_delta_f64() for value in evidence}
    closure = {key: jvp[key] - finite[key] for key in jvp}

    def mean(items: Mapping[str, Tensor]) -> float:
        return float(_nested_mean(evidence, items))

    def second(items: Mapping[str, Tensor]) -> float:
        return mean({key: value.square() for key, value in items.items()})

    jvp_rms = _rms(second(jvp))
    finite_rms = _rms(second(finite))
    closure_rmse = _rms(second(closure))
    return CandidateJointStateContractionPrecisionFiniteMetrics(
        mean_suffix_jvp_f64=mean(jvp),
        suffix_jvp_f64_rms=jvp_rms,
        mean_finite_delta_f64=mean(finite),
        finite_delta_f64_rms=finite_rms,
        mean_jvp_minus_finite_f64=mean(closure),
        closure_rmse=closure_rmse,
        closure_relative_rmse=_relative_rmse(closure_rmse, finite_rms),
        closure_cosine=_cosine(
            cross=mean({key: jvp[key] * finite[key] for key in jvp}),
            left_rms=jvp_rms,
            right_rms=finite_rms,
        ),
        maximum_absolute_closure_error=max(
            float(value.abs().max()) for value in closure.values()
        ),
        relative_rmse_epsilon=_EPSILON,
    )


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionFamilySummary:
    family_id: str
    example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    supervised_token_count: int
    stage_metrics: tuple[CandidateJointStateContractionPrecisionStageMetrics, ...]
    finite_metrics: CandidateJointStateContractionPrecisionFiniteMetrics
    transition_metrics: tuple[
        CandidateJointStateContractionPrecisionTransitionMetrics, ...
    ]
    telescope_metrics: CandidateJointStateContractionPrecisionTelescopeMetrics
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        family = _identifier(self.family_id, label="contraction precision family_id")
        examples = tuple(
            _identifier(value, label="contraction precision family example_id")
            for value in self.example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="contraction precision evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        metrics = tuple(self.stage_metrics)
        transitions = tuple(self.transition_metrics)
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or tuple(metric.stage for metric in metrics)
            != CONTRACTION_PUBLISHED_STAGE_ORDER
            or not isinstance(
                self.finite_metrics,
                CandidateJointStateContractionPrecisionFiniteMetrics,
            )
            or tuple(value.transition for value in transitions)
            != CONTRACTION_TELESCOPE_TRANSITIONS
            or not isinstance(
                self.telescope_metrics,
                CandidateJointStateContractionPrecisionTelescopeMetrics,
            )
            or not self.telescope_metrics.passed
        ):
            raise ValueError("contraction precision family membership is invalid")
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "stage_metrics", metrics)
        object.__setattr__(self, "transition_metrics", transitions)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_FAMILY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def metrics_for(self, stage: str) -> CandidateJointStateContractionPrecisionStageMetrics:
        if stage not in CONTRACTION_PUBLISHED_STAGE_ORDER:
            raise ValueError("unknown contraction precision stage")
        return self.stage_metrics[CONTRACTION_PUBLISHED_STAGE_ORDER.index(stage)]

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "family_id": self.family_id,
            "example_ids": self.example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "prompt_count": len(self.example_ids),
            "supervised_token_count": self.supervised_token_count,
            "stage_metrics": tuple(metric.metadata() for metric in self.stage_metrics),
            "suffix_jvp_finite_metrics": self.finite_metrics.metadata(),
            "transition_metrics": tuple(
                metric.metadata() for metric in self.transition_metrics
            ),
            "telescope_metrics": self.telescope_metrics.metadata(),
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _FAMILY_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(self.artifact_sha256, label="contraction family artifact"):
            raise RuntimeError("candidate joint-state contraction family drifted")


def classify_candidate_joint_state_contraction_precision(
    *,
    p64_node_passed: bool,
    p_dir_passed: bool,
    p_prod_passed: bool,
    p_live_passed: bool,
) -> str:
    """Return the earliest precision boundary that closes the adjoint gap."""

    values = (p64_node_passed, p_dir_passed, p_prod_passed, p_live_passed)
    if any(type(value) is not bool for value in values):
        raise TypeError("contraction precision classification gates must be boolean")
    labels = (
        "f64_operation_ordering_sufficient",
        "direction_cast_rounding_sufficient",
        "f32_product_rounding_sufficient",
        "native_f32_boundary_reduction_sufficient",
    )
    for passed, label in zip(values, labels, strict=True):
        if passed:
            return label
    return "unresolved_forward_reverse_ad_kernel_mismatch"


@dataclass(frozen=True, slots=True)
class CandidateJointStateContractionPrecisionComparison:
    evidence_example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    family_summaries: tuple[CandidateJointStateContractionPrecisionFamilySummary, ...]
    supervised_token_count: int
    stage_metrics: tuple[CandidateJointStateContractionPrecisionStageMetrics, ...]
    finite_metrics: CandidateJointStateContractionPrecisionFiniteMetrics
    transition_metrics: tuple[
        CandidateJointStateContractionPrecisionTransitionMetrics, ...
    ]
    telescope_metrics: CandidateJointStateContractionPrecisionTelescopeMetrics
    replayed_v11_comparison_artifact_sha256: str
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        examples = tuple(
            _identifier(value, label="contraction precision example_id")
            for value in self.evidence_example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="contraction precision evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        families = tuple(self.family_summaries)
        metrics = tuple(self.stage_metrics)
        transitions = tuple(self.transition_metrics)
        replay = _require_sha256(
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
            or tuple(metric.stage for metric in metrics)
            != CONTRACTION_PUBLISHED_STAGE_ORDER
            or not isinstance(
                self.finite_metrics,
                CandidateJointStateContractionPrecisionFiniteMetrics,
            )
            or tuple(value.transition for value in transitions)
            != CONTRACTION_TELESCOPE_TRANSITIONS
            or not isinstance(
                self.telescope_metrics,
                CandidateJointStateContractionPrecisionTelescopeMetrics,
            )
            or not self.telescope_metrics.passed
        ):
            raise ValueError("contraction precision comparison membership is invalid")
        for family in families:
            family.validate_integrity()
        object.__setattr__(self, "evidence_example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "family_summaries", families)
        object.__setattr__(self, "stage_metrics", metrics)
        object.__setattr__(self, "transition_metrics", transitions)
        object.__setattr__(self, "replayed_v11_comparison_artifact_sha256", replay)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_SUMMARY_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def metrics_for(self, stage: str) -> CandidateJointStateContractionPrecisionStageMetrics:
        if stage not in CONTRACTION_PUBLISHED_STAGE_ORDER:
            raise ValueError("unknown contraction precision stage")
        return self.stage_metrics[CONTRACTION_PUBLISHED_STAGE_ORDER.index(stage)]

    @property
    def stage_gate_results(self) -> dict[str, bool]:
        return {
            stage: (
                self.metrics_for(stage).adjoint_relative_rmse
                <= ADJOINT_RELATIVE_RMSE_MAXIMUM
                and all(
                    family.metrics_for(stage).adjoint_relative_rmse
                    <= ADJOINT_RELATIVE_RMSE_MAXIMUM
                    for family in self.family_summaries
                )
            )
            for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
        }

    @property
    def earliest_passing_stage(self) -> str | None:
        for stage in CONTRACTION_CORRECTED_STAGE_ORDER:
            if self.stage_gate_results[stage]:
                return stage
        return None

    @property
    def classification(self) -> str:
        gates = self.stage_gate_results
        return classify_candidate_joint_state_contraction_precision(
            p64_node_passed=gates["P64_node"],
            p_dir_passed=gates["P_dir"],
            p_prod_passed=gates["P_prod"],
            p_live_passed=gates["P_live"],
        )

    def _closure_gate_results(self, *, stage: str | None) -> dict[str, bool]:
        if stage is None:
            overall_relative = self.finite_metrics.closure_relative_rmse
            overall_cosine = self.finite_metrics.closure_cosine
            family_relative = tuple(
                family.finite_metrics.closure_relative_rmse
                for family in self.family_summaries
            )
            prefix = "suffix_jvp"
        else:
            overall = self.metrics_for(stage)
            overall_relative = overall.closure_relative_rmse
            overall_cosine = overall.closure_cosine
            family_relative = tuple(
                family.metrics_for(stage).closure_relative_rmse
                for family in self.family_summaries
            )
            prefix = stage
        return {
            f"overall_{prefix}_closure_relative_RMSE_at_most_0_05": (
                overall_relative <= OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM
            ),
            f"every_family_{prefix}_closure_relative_RMSE_at_most_0_10": all(
                value <= FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM
                for value in family_relative
            ),
            f"overall_{prefix}_closure_cosine_at_least_0_99": (
                overall_cosine >= CLOSURE_COSINE_MINIMUM
            ),
        }

    @property
    def suffix_jvp_closure_gate_results(self) -> dict[str, bool]:
        return self._closure_gate_results(stage=None)

    @property
    def selected_stage_closure_gate_results(self) -> dict[str, bool]:
        if self.earliest_passing_stage is None:
            return {}
        return self._closure_gate_results(stage=self.earliest_passing_stage)

    @property
    def adjoint_residual_fraction_of_finite_closure(self) -> float:
        stage = self.earliest_passing_stage
        if stage is None:
            return 1.0
        remaining = self.metrics_for(stage).adjoint_rmse
        finite = self.finite_metrics.closure_rmse
        if finite <= _EPSILON and remaining <= _EPSILON:
            return 0.0
        return remaining / max(finite, _EPSILON)

    @property
    def finite_closure_to_remaining_adjoint_rmse_ratio(self) -> float:
        stage = self.earliest_passing_stage
        if stage is None:
            return 0.0
        return self.finite_metrics.closure_rmse / max(
            self.metrics_for(stage).adjoint_rmse, _EPSILON
        )

    @property
    def finite_correction_eligibility_gate_results(self) -> dict[str, bool]:
        stage = self.earliest_passing_stage
        selected_closure_passed = bool(self.selected_stage_closure_gate_results) and all(
            self.selected_stage_closure_gate_results.values()
        )
        jvp_closure_passed = all(self.suffix_jvp_closure_gate_results.values())
        return {
            "corrected_contraction_stage_identified": stage is not None,
            "remaining_adjoint_RMSE_fraction_of_finite_closure_at_most_0_01": (
                stage is not None
                and self.adjoint_residual_fraction_of_finite_closure
                <= FINITE_ADJOINT_RESIDUAL_FRACTION_MAXIMUM
            ),
            "corrected_contraction_still_fails_frozen_finite_closure_suite": (
                stage is not None and not selected_closure_passed
            ),
            "suffix_JVP_still_fails_frozen_finite_closure_suite": (
                not jvp_closure_passed
            ),
            "finite_closure_RMSE_over_remaining_adjoint_RMSE_at_least_100": (
                stage is not None
                and self.finite_closure_to_remaining_adjoint_rmse_ratio
                >= FINITE_TO_ADJOINT_RMSE_RATIO_MINIMUM
            ),
        }

    @property
    def finite_correction_eligible(self) -> bool:
        return all(self.finite_correction_eligibility_gate_results.values())

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
            "stage_metrics": tuple(metric.metadata() for metric in self.stage_metrics),
            "suffix_jvp_finite_metrics": self.finite_metrics.metadata(),
            "transition_metrics": tuple(
                metric.metadata() for metric in self.transition_metrics
            ),
            "telescope_metrics": self.telescope_metrics.metadata(),
            "replayed_v11_comparison_artifact_sha256": (
                self.replayed_v11_comparison_artifact_sha256
            ),
            "P_v10_and_V11_JVP_finite_metrics_replayed_exactly": True,
            "stage_gate_results": tuple(sorted(self.stage_gate_results.items())),
            "earliest_passing_stage": self.earliest_passing_stage,
            "classification": self.classification,
            "suffix_jvp_closure_gate_results": tuple(
                sorted(self.suffix_jvp_closure_gate_results.items())
            ),
            "selected_stage_closure_gate_results": tuple(
                sorted(self.selected_stage_closure_gate_results.items())
            ),
            "adjoint_residual_fraction_of_finite_closure": (
                self.adjoint_residual_fraction_of_finite_closure
            ),
            "finite_closure_to_remaining_adjoint_RMSE_ratio": (
                self.finite_closure_to_remaining_adjoint_rmse_ratio
            ),
            "finite_correction_eligibility_gate_results": tuple(
                sorted(self.finite_correction_eligibility_gate_results.items())
            ),
            "finite_correction_eligible": self.finite_correction_eligible,
            "weighting": _WEIGHTING,
            "adjoint_gate": (
                "overall_and_every_family_symmetric_relative_RMSE_at_most_0_0001"
            ),
            "finite_closure_thresholds": {
                "overall_relative_RMSE_maximum": OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM,
                "every_family_relative_RMSE_maximum": FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM,
                "overall_cosine_minimum": CLOSURE_COSINE_MINIMUM,
            },
            "finite_correction_eligibility_thresholds": {
                "remaining_adjoint_fraction_maximum": (
                    FINITE_ADJOINT_RESIDUAL_FRACTION_MAXIMUM
                ),
                "finite_to_remaining_adjoint_RMSE_ratio_minimum": (
                    FINITE_TO_ADJOINT_RMSE_RATIO_MINIMUM
                ),
            },
            "all_precision_stages_published_regardless_of_pass": True,
            "fixed_telescope_reconstructs_D64_finite_minus_P_v10": True,
            "P_live_is_counterfactual_not_internal_VJP_schedule_proof": True,
            "remaining_finite_gap_is_cast_or_quadrature_remainder_not_cast_causality_proof": True,
            "same_A_hypothesis_use_only": True,
            "authorizes_finite_correction_experiment_only_when_eligible": True,
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
            self.artifact_sha256, label="contraction precision comparison artifact"
        ):
            raise RuntimeError("candidate joint-state contraction comparison drifted")


def _canonical_evidence(
    evidence: Iterable[CandidateJointStateContractionPrecisionEvidence],
) -> tuple[CandidateJointStateContractionPrecisionEvidence, ...]:
    values = tuple(evidence)
    if not values or any(
        not isinstance(value, CandidateJointStateContractionPrecisionEvidence)
        for value in values
    ):
        raise TypeError("contraction precision comparison requires typed evidence")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("contraction precision example ids must be unique")
    return ordered


def _assert_v11_replay_exact(
    *,
    stage: CandidateJointStateContractionPrecisionStageMetrics,
    finite: CandidateJointStateContractionPrecisionFiniteMetrics,
    v11_metrics: object,
) -> None:
    pairs = (
        (stage.mean_stage_f64, getattr(v11_metrics, "mean_vjp_path_integral_f64")),
        (stage.stage_f64_rms, getattr(v11_metrics, "vjp_path_integral_f64_rms")),
        (stage.mean_jvp_minus_stage_f64, getattr(v11_metrics, "mean_jvp_minus_vjp_f64")),
        (stage.adjoint_rmse, getattr(v11_metrics, "adjoint_rmse")),
        (stage.adjoint_relative_rmse, getattr(v11_metrics, "adjoint_relative_rmse")),
        (stage.adjoint_cosine, getattr(v11_metrics, "adjoint_cosine")),
        (
            stage.maximum_absolute_adjoint_error,
            getattr(v11_metrics, "maximum_absolute_adjoint_error"),
        ),
        (
            stage.mean_stage_minus_finite_f64,
            getattr(v11_metrics, "mean_vjp_minus_finite_f64"),
        ),
        (stage.closure_rmse, getattr(v11_metrics, "vjp_closure_rmse")),
        (stage.closure_relative_rmse, getattr(v11_metrics, "vjp_closure_relative_rmse")),
        (stage.closure_cosine, getattr(v11_metrics, "vjp_closure_cosine")),
        (
            stage.maximum_absolute_closure_error,
            getattr(v11_metrics, "maximum_absolute_vjp_closure_error"),
        ),
        (finite.mean_suffix_jvp_f64, getattr(v11_metrics, "mean_suffix_jvp_f64")),
        (finite.suffix_jvp_f64_rms, getattr(v11_metrics, "suffix_jvp_f64_rms")),
        (finite.mean_finite_delta_f64, getattr(v11_metrics, "mean_finite_delta_f64")),
        (finite.finite_delta_f64_rms, getattr(v11_metrics, "finite_delta_f64_rms")),
        (finite.mean_jvp_minus_finite_f64, getattr(v11_metrics, "mean_jvp_minus_finite_f64")),
        (finite.closure_rmse, getattr(v11_metrics, "jvp_closure_rmse")),
        (finite.closure_relative_rmse, getattr(v11_metrics, "jvp_closure_relative_rmse")),
        (finite.closure_cosine, getattr(v11_metrics, "jvp_closure_cosine")),
        (
            finite.maximum_absolute_closure_error,
            getattr(v11_metrics, "maximum_absolute_jvp_closure_error"),
        ),
        (finite.relative_rmse_epsilon, getattr(v11_metrics, "relative_rmse_epsilon")),
    )
    if any(not _same_float(left, right) for left, right in pairs):
        raise RuntimeError("contraction precision summary did not exactly replay V11")


def summarize_candidate_joint_state_contraction_precision(
    evidence: Iterable[CandidateJointStateContractionPrecisionEvidence],
) -> CandidateJointStateContractionPrecisionComparison:
    """Build immutable family-equal V12 ladder and finite-rung summaries."""

    values = _canonical_evidence(evidence)
    v11: CandidateJointStateSuffixJVPComparison = (
        summarize_candidate_joint_state_suffix_jvp(
            tuple(value.suffix_jvp_evidence for value in values)
        )
    )
    metrics = tuple(
        _stage_metrics(values, stage) for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
    )
    finite = _finite_metrics(values)
    transitions = _transition_metrics(values)
    telescope = _telescope_metrics(values)
    _assert_v11_replay_exact(stage=metrics[0], finite=finite, v11_metrics=v11.metrics)
    by_family: dict[str, list[CandidateJointStateContractionPrecisionEvidence]] = defaultdict(list)
    for value in values:
        by_family[value.family_id].append(value)
    v11_families = {family.family_id: family for family in v11.family_summaries}
    families: list[CandidateJointStateContractionPrecisionFamilySummary] = []
    for family_id in sorted(by_family):
        members = tuple(sorted(by_family[family_id], key=lambda value: value.example_id))
        family_metrics = tuple(
            _stage_metrics(members, stage)
            for stage in CONTRACTION_PUBLISHED_STAGE_ORDER
        )
        family_finite = _finite_metrics(members)
        family_transitions = _transition_metrics(members)
        family_telescope = _telescope_metrics(members)
        _assert_v11_replay_exact(
            stage=family_metrics[0],
            finite=family_finite,
            v11_metrics=v11_families[family_id].metrics,
        )
        families.append(
            CandidateJointStateContractionPrecisionFamilySummary(
                family_id=family_id,
                example_ids=tuple(value.example_id for value in members),
                evidence_artifact_sha256s=tuple(
                    value.artifact_sha256 for value in members
                ),
                supervised_token_count=sum(
                    value.supervised_tokens for value in members
                ),
                stage_metrics=family_metrics,
                finite_metrics=family_finite,
                transition_metrics=family_transitions,
                telescope_metrics=family_telescope,
            )
        )
    return CandidateJointStateContractionPrecisionComparison(
        evidence_example_ids=tuple(value.example_id for value in values),
        evidence_artifact_sha256s=tuple(value.artifact_sha256 for value in values),
        family_summaries=tuple(families),
        supervised_token_count=sum(value.supervised_tokens for value in values),
        stage_metrics=metrics,
        finite_metrics=finite,
        transition_metrics=transitions,
        telescope_metrics=telescope,
        replayed_v11_comparison_artifact_sha256=v11.artifact_sha256,
    )
