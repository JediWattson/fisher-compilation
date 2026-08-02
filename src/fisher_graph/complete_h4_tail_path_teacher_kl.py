"""Pure authenticated GL4 path evidence for complete-H4 teacher KL.

The endpoint signed-joint screen differentiates teacher KL once at D320.  A
finite displacement has different semantics: for the straight path

``H(alpha) = H_source + alpha * (H_native - H_source)``,

the fundamental theorem of calculus compares ``KL_native - KL_source`` with
the integral of ``d KL / d H`` contracted with the complete-H4 displacement.
This module represents that path experiment without knowing anything about a
model runtime.

Path nodes are deliberately accumulated one at a time.  The final evidence
retains only the GL4-weighted gradient, finite boundary objectives, scalar and
hash receipts, and the prompt-local displacement.  It never retains the four
full node-gradient banks.  Metadata contains hashes and scalars only; all raw
tensors remain hypothesis-use evidence.
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

from .causal_edge_transport import gauss_legendre_unit_interval


__all__ = [
    "CompleteH4TailPathTeacherKLAccumulator",
    "CompleteH4TailPathTeacherKLClosure",
    "CompleteH4TailPathTeacherKLEvidence",
    "CompleteH4TailPathTeacherKLFamilyClosure",
    "CompleteH4TailPathTeacherKLNodeReceipt",
    "GL4_UNIT_INTERVAL_NODES",
    "GL4_UNIT_INTERVAL_WEIGHTS",
    "complete_h4_tail_path_as_endpoint_example",
    "complete_h4_tail_path_basis_contraction",
    "complete_h4_tail_path_direct_contraction",
    "complete_h4_tail_path_family_prompt_token_mean",
    "complete_h4_tail_path_gate_scores",
    "complete_h4_tail_path_ftc_target",
    "complete_h4_tail_path_weighted_gradient",
    "summarize_complete_h4_tail_path_ftc_closure",
]


GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS = (
    gauss_legendre_unit_interval(4)
)

_NODE_DOMAIN = b"fisher-graph:complete-h4-tail-path-teacher-kl-node:v1\0"
_EVIDENCE_DOMAIN = (
    b"fisher-graph:complete-h4-tail-path-teacher-kl-evidence:v1\0"
)
_CLOSURE_DOMAIN = (
    b"fisher-graph:complete-h4-tail-path-teacher-kl-closure:v1\0"
)
_TENSOR_DOMAIN = b"fisher-graph:complete-h4-tail-path-teacher-kl-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WEIGHTING = "equal_family_then_equal_prompt_then_equal_token"
_PATH_OBJECTIVE = "KL_native_teacher_distribution_to_path_candidate_distribution"
_PATH_GEOMETRY = "straight_complete_H4_source_to_native_displacement"
_GRADIENT_SEMANTICS = "GL4_path_integrated_teacher_KL_gradient"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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
        or not value.is_floating_point()
        or 0 in value.shape
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating tensor")
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .clone()
        .contiguous()
    )


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise TypeError("hashed value must be a tensor")
    tensor = _float64(value, label="hashed tensor", ndim=value.ndim)
    payload = tensor.numpy().astype("<f8", copy=False).tobytes(order="C")
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + _canonical_json_bytes(
            {"dtype": "float64-little-endian", "shape": tuple(tensor.shape)}
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


def _canonical_basis(value: Tensor, *, width: int) -> Tensor:
    basis = _float64(value, label="path contraction basis", ndim=2)
    if basis.shape[1] != width:
        raise ValueError("path contraction basis width differs")
    identity = torch.eye(basis.shape[0], dtype=torch.float64)
    if not torch.allclose(
        basis @ basis.T, identity, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("path contraction basis rows must be orthonormal")
    return basis


@dataclass(frozen=True, slots=True)
class CompleteH4TailPathTeacherKLNodeReceipt:
    """Hash/scalar receipt for one transient GL4 integrand evaluation."""

    node_index: int
    path_fraction: float
    quadrature_weight: float
    token_count: int
    h4_gradient_shape: tuple[int, int, int]
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
        mean = _finite_float(
            self.token_teacher_kl_mean, label="node token KL mean"
        )
        minimum = _finite_float(
            self.token_teacher_kl_minimum, label="node token KL minimum"
        )
        maximum = _finite_float(
            self.token_teacher_kl_maximum, label="node token KL maximum"
        )
        scale = max(abs(minimum), abs(maximum), abs(mean), 1.0)
        tolerance = 64.0 * torch.finfo(torch.float64).eps * scale
        if mean < minimum - tolerance or mean > maximum + tolerance:
            raise ValueError("node token KL summary is inconsistent")
        gradient_norm = _finite_float(
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
        object.__setattr__(self, "h4_gradient_frobenius", gradient_norm)
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
            raise RuntimeError("path teacher-KL node receipt drifted")


@dataclass(frozen=True, slots=True)
class CompleteH4TailPathTeacherKLEvidence:
    """One prompt's ephemeral GL4-integrated teacher-KL path evidence."""

    example_id: str
    family_id: str
    residual_rows: Tensor = field(repr=False)
    integrated_token_h4_gradients: Tensor = field(repr=False)
    source_token_teacher_kl: Tensor = field(repr=False)
    native_token_teacher_kl: Tensor = field(repr=False)
    node_receipts: tuple[CompleteH4TailPathTeacherKLNodeReceipt, ...]
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        example_id = _identifier(self.example_id, label="path example_id")
        family_id = _identifier(self.family_id, label="path family_id")
        residual = _float64(
            self.residual_rows, label="path complete-H4 displacement", ndim=2
        )
        gradient = _float64(
            self.integrated_token_h4_gradients,
            label="path integrated token H4 gradients",
            ndim=3,
        )
        source = _float64(
            self.source_token_teacher_kl,
            label="path source token teacher KL",
            ndim=1,
        )
        native = _float64(
            self.native_token_teacher_kl,
            label="path native token teacher KL",
            ndim=1,
        )
        if (
            gradient.shape[0] != source.shape[0]
            or native.shape != source.shape
            or gradient.shape[1:] != residual.shape
        ):
            raise ValueError("path displacement, gradient, and boundary KL differ")
        receipts = tuple(self.node_receipts)
        if (
            len(receipts) != 4
            or any(
                not isinstance(receipt, CompleteH4TailPathTeacherKLNodeReceipt)
                or receipt.node_index != index
                for index, receipt in enumerate(receipts)
            )
        ):
            raise ValueError("path evidence requires the ordered four GL4 receipts")
        for receipt in receipts:
            receipt.validate_integrity()
            if receipt.h4_gradient_shape != tuple(gradient.shape):
                raise ValueError("path node and integrated gradient shapes differ")
        zero_sha256 = _tensor_sha256(torch.zeros_like(gradient))
        if receipts[0].integrated_gradient_sha256_before != zero_sha256:
            raise ValueError("path accumulation does not start from zero")
        for before, after in zip(receipts, receipts[1:]):
            if (
                before.integrated_gradient_sha256_after
                != after.integrated_gradient_sha256_before
            ):
                raise ValueError("path accumulation receipt chain is broken")
        if receipts[-1].integrated_gradient_sha256_after != _tensor_sha256(gradient):
            raise ValueError("path integrated gradient differs from its receipt chain")
        object.__setattr__(self, "example_id", example_id)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "residual_rows", residual)
        object.__setattr__(self, "integrated_token_h4_gradients", gradient)
        object.__setattr__(self, "source_token_teacher_kl", source)
        object.__setattr__(self, "native_token_teacher_kl", native)
        object.__setattr__(self, "node_receipts", receipts)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    @property
    def supervised_tokens(self) -> int:
        return int(self.source_token_teacher_kl.shape[0])

    @property
    def width(self) -> int:
        return int(self.residual_rows.shape[1])

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "example_id": self.example_id,
            "family_id": self.family_id,
            "residual_shape": tuple(self.residual_rows.shape),
            "residual_sha256": _tensor_sha256(self.residual_rows),
            "integrated_token_h4_gradients_shape": tuple(
                self.integrated_token_h4_gradients.shape
            ),
            "integrated_token_h4_gradients_sha256": _tensor_sha256(
                self.integrated_token_h4_gradients
            ),
            "source_token_teacher_kl_sha256": _tensor_sha256(
                self.source_token_teacher_kl
            ),
            "native_token_teacher_kl_sha256": _tensor_sha256(
                self.native_token_teacher_kl
            ),
            "supervised_token_count": self.supervised_tokens,
            "node_receipts": tuple(
                receipt.metadata() for receipt in self.node_receipts
            ),
            "quadrature_rule": "gauss_legendre_order_4_on_unit_interval",
            "path_objective": _PATH_OBJECTIVE,
            "path_geometry": _PATH_GEOMETRY,
            "gradient_semantics": _GRADIENT_SEMANTICS,
            "uses_only_strictly_interior_path_nodes": True,
            "endpoint_gradient_substituted_for_path_integral": False,
            "finite_boundary_KLs_used_as_integrand_nodes": False,
            "all_path_nodes_causal": all(
                receipt.future_gradient_nonzero_count == 0
                and receipt.maximum_future_gradient_abs == 0.0
                for receipt in self.node_receipts
            ),
            "full_node_gradient_banks_retained": False,
            "raw_evidence_serialized": False,
            "hypothesis_use_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        for receipt in self.node_receipts:
            receipt.validate_integrity()
        if (
            self.residual_rows.dtype != torch.float64
            or self.residual_rows.device.type != "cpu"
            or self.residual_rows.requires_grad
            or not self.residual_rows.is_contiguous()
            or not bool(torch.isfinite(self.residual_rows).all())
            or self.integrated_token_h4_gradients.dtype != torch.float64
            or self.integrated_token_h4_gradients.device.type != "cpu"
            or self.integrated_token_h4_gradients.requires_grad
            or not self.integrated_token_h4_gradients.is_contiguous()
            or not bool(torch.isfinite(self.integrated_token_h4_gradients).all())
            or self.source_token_teacher_kl.dtype != torch.float64
            or self.source_token_teacher_kl.device.type != "cpu"
            or self.source_token_teacher_kl.requires_grad
            or not self.source_token_teacher_kl.is_contiguous()
            or not bool(torch.isfinite(self.source_token_teacher_kl).all())
            or self.native_token_teacher_kl.dtype != torch.float64
            or self.native_token_teacher_kl.device.type != "cpu"
            or self.native_token_teacher_kl.requires_grad
            or not self.native_token_teacher_kl.is_contiguous()
            or not bool(torch.isfinite(self.native_token_teacher_kl).all())
            or self.node_receipts[-1].integrated_gradient_sha256_after
            != _tensor_sha256(self.integrated_token_h4_gradients)
            or _sha256(_EVIDENCE_DOMAIN, self.metadata(include_artifact=False))
            != _require_sha256(self.artifact_sha256, label="path evidence artifact")
        ):
            raise RuntimeError("complete-H4 path teacher-KL evidence drifted")


class CompleteH4TailPathTeacherKLAccumulator:
    """Stream four GL4 nodes into one authenticated integrated gradient."""

    __slots__ = (
        "_example_id",
        "_family_id",
        "_residual_rows",
        "_source_token_teacher_kl",
        "_native_token_teacher_kl",
        "_integrated",
        "_node_receipts",
        "_sealed",
    )

    def __init__(
        self,
        *,
        example_id: str,
        family_id: str,
        residual_rows: Tensor,
        source_token_teacher_kl: Tensor,
        native_token_teacher_kl: Tensor,
    ) -> None:
        self._example_id = _identifier(example_id, label="path example_id")
        self._family_id = _identifier(family_id, label="path family_id")
        self._residual_rows = _float64(
            residual_rows, label="path complete-H4 displacement", ndim=2
        )
        self._source_token_teacher_kl = _float64(
            source_token_teacher_kl,
            label="path source token teacher KL",
            ndim=1,
        )
        self._native_token_teacher_kl = _float64(
            native_token_teacher_kl,
            label="path native token teacher KL",
            ndim=1,
        )
        if self._native_token_teacher_kl.shape != self._source_token_teacher_kl.shape:
            raise ValueError("path source and native token KL shapes differ")
        self._integrated = torch.zeros(
            (
                int(self._source_token_teacher_kl.shape[0]),
                int(self._residual_rows.shape[0]),
                int(self._residual_rows.shape[1]),
            ),
            dtype=torch.float64,
        )
        self._node_receipts: list[CompleteH4TailPathTeacherKLNodeReceipt] = []
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
        token_h4_gradients: Tensor,
        token_teacher_kl: Tensor,
        vjp_artifact_sha256: str,
        provider_artifact_sha256: str,
        execution_artifact_sha256: str,
        maximum_future_gradient_abs: float,
        future_gradient_nonzero_count: int,
    ) -> CompleteH4TailPathTeacherKLNodeReceipt:
        """Consume one node and immediately discard its raw gradient bank."""

        if self._sealed:
            raise RuntimeError("path accumulator is already sealed")
        if type(node_index) is not int or node_index != len(self._node_receipts):
            raise ValueError("path nodes must be added once in canonical GL4 order")
        if node_index >= 4:
            raise ValueError("path accumulator already has all four GL4 nodes")
        node = _finite_float(path_fraction, label="path fraction")
        weight = _finite_float(
            quadrature_weight,
            label="quadrature weight",
            nonnegative=True,
        )
        if (
            node.hex() != GL4_UNIT_INTERVAL_NODES[node_index].hex()
            or weight.hex() != GL4_UNIT_INTERVAL_WEIGHTS[node_index].hex()
        ):
            raise ValueError("path node does not match the exact GL4 rule")
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
        receipt = CompleteH4TailPathTeacherKLNodeReceipt(
            node_index=node_index,
            path_fraction=node,
            quadrature_weight=weight,
            token_count=int(token_kl.shape[0]),
            h4_gradient_shape=tuple(int(size) for size in gradient.shape),
            token_teacher_kl_sha256=_tensor_sha256(token_kl),
            token_teacher_kl_mean=float(token_kl.mean()),
            token_teacher_kl_minimum=float(token_kl.min()),
            token_teacher_kl_maximum=float(token_kl.max()),
            h4_gradient_sha256=_tensor_sha256(gradient),
            h4_gradient_frobenius=float(torch.linalg.vector_norm(gradient)),
            integrated_gradient_sha256_before=before,
            integrated_gradient_sha256_after=_tensor_sha256(updated),
            vjp_artifact_sha256=_require_sha256(
                vjp_artifact_sha256, label="path node VJP artifact"
            ),
            provider_artifact_sha256=_require_sha256(
                provider_artifact_sha256, label="path node provider artifact"
            ),
            execution_artifact_sha256=_require_sha256(
                execution_artifact_sha256, label="path node execution artifact"
            ),
            maximum_future_gradient_abs=maximum_future_gradient_abs,
            future_gradient_nonzero_count=future_gradient_nonzero_count,
        )
        self._integrated = updated
        self._node_receipts.append(receipt)
        return receipt

    def finalize(self) -> CompleteH4TailPathTeacherKLEvidence:
        if self._sealed:
            raise RuntimeError("path accumulator is already sealed")
        if len(self._node_receipts) != 4:
            raise RuntimeError("path accumulator requires all four GL4 nodes")
        evidence = CompleteH4TailPathTeacherKLEvidence(
            example_id=self._example_id,
            family_id=self._family_id,
            residual_rows=self._residual_rows,
            integrated_token_h4_gradients=self._integrated,
            source_token_teacher_kl=self._source_token_teacher_kl,
            native_token_teacher_kl=self._native_token_teacher_kl,
            node_receipts=tuple(self._node_receipts),
        )
        self._sealed = True
        return evidence


def _path_evidence(
    value: object,
) -> CompleteH4TailPathTeacherKLEvidence:
    if not isinstance(value, CompleteH4TailPathTeacherKLEvidence):
        raise TypeError("value must be complete-H4 path teacher-KL evidence")
    value.validate_integrity()
    return value


def complete_h4_tail_path_weighted_gradient(
    evidence: CompleteH4TailPathTeacherKLEvidence,
) -> Tensor:
    """Return a defensive copy of the already GL4-weighted token gradient."""

    value = _path_evidence(evidence)
    return value.integrated_token_h4_gradients.clone().contiguous()


def complete_h4_tail_path_gate_scores(
    evidence: CompleteH4TailPathTeacherKLEvidence,
    basis_rows: Tensor,
) -> Tensor:
    """Return per-token, per-direction path-integrated quadratic responses."""

    value = _path_evidence(evidence)
    basis = _canonical_basis(basis_rows, width=value.width)
    amplitudes = value.residual_rows @ basis.T
    gradient_coordinates = torch.einsum(
        "trw,kw->trk", value.integrated_token_h4_gradients, basis
    )
    return torch.einsum("rk,trk->tk", amplitudes, gradient_coordinates).contiguous()


def complete_h4_tail_path_basis_contraction(
    evidence: CompleteH4TailPathTeacherKLEvidence,
    basis_rows: Tensor,
) -> Tensor:
    """Contract through an orthonormal basis, summing its direction scores."""

    return complete_h4_tail_path_gate_scores(evidence, basis_rows).sum(
        dim=1
    ).contiguous()


def complete_h4_tail_path_direct_contraction(
    evidence: CompleteH4TailPathTeacherKLEvidence,
) -> Tensor:
    """Contract the integrated gradient directly with the finite displacement."""

    value = _path_evidence(evidence)
    return torch.einsum(
        "rw,trw->t", value.residual_rows, value.integrated_token_h4_gradients
    ).contiguous()


def complete_h4_tail_path_ftc_target(
    evidence: CompleteH4TailPathTeacherKLEvidence,
) -> Tensor:
    """Return ``KL_native - KL_source`` in path-integral orientation."""

    value = _path_evidence(evidence)
    return (
        value.native_token_teacher_kl - value.source_token_teacher_kl
    ).contiguous()


def complete_h4_tail_path_as_endpoint_example(
    evidence: CompleteH4TailPathTeacherKLEvidence,
):
    """Adapt integrated path evidence to the signed-projector input shape.

    This explicit adapter preserves the semantic boundary: its gradient is a
    GL4 path integral and its target is the finite FTC boundary difference.
    It is not an endpoint VJP, even though the downstream model-agnostic
    projector intentionally consumes the same tensor geometry.
    """

    from .complete_h4_tail_token_fisher import CompleteH4TailEndpointExample

    value = _path_evidence(evidence)
    return CompleteH4TailEndpointExample(
        example_id=value.example_id,
        family_id=value.family_id,
        residual_rows=value.residual_rows,
        token_h4_gradients=value.integrated_token_h4_gradients,
        compensation_target=complete_h4_tail_path_ftc_target(value),
    )


def _canonical_evidence(
    evidence: Iterable[CompleteH4TailPathTeacherKLEvidence],
) -> tuple[CompleteH4TailPathTeacherKLEvidence, ...]:
    values = tuple(evidence)
    if not values or any(
        not isinstance(value, CompleteH4TailPathTeacherKLEvidence)
        for value in values
    ):
        raise TypeError("path closure requires nonempty typed evidence")
    ordered = tuple(sorted(values, key=lambda value: value.example_id))
    for value in ordered:
        value.validate_integrity()
    if len({value.example_id for value in ordered}) != len(ordered):
        raise ValueError("path evidence example ids must be unique")
    if len({value.width for value in ordered}) != 1:
        raise ValueError("path evidence widths differ")
    return ordered


def complete_h4_tail_path_family_prompt_token_mean(
    evidence: Iterable[CompleteH4TailPathTeacherKLEvidence],
    token_values: Mapping[str, Tensor],
) -> Tensor:
    """Average tokens, then prompts, then families with equal weights."""

    values = _canonical_evidence(evidence)
    if not isinstance(token_values, Mapping) or set(token_values) != {
        value.example_id for value in values
    }:
        raise ValueError("path token statistic keys differ from the evidence")
    by_family: dict[str, list[Tensor]] = defaultdict(list)
    trailing_shape: tuple[int, ...] | None = None
    for evidence_value in values:
        statistic = token_values[evidence_value.example_id]
        if (
            not isinstance(statistic, Tensor)
            or statistic.ndim < 1
            or not statistic.is_floating_point()
            or statistic.shape[0] != evidence_value.supervised_tokens
            or not bool(torch.isfinite(statistic).all())
        ):
            raise ValueError("path token statistic must be finite and token aligned")
        statistic64 = statistic.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        shape = tuple(int(size) for size in statistic64.shape[1:])
        if trailing_shape is None:
            trailing_shape = shape
        elif shape != trailing_shape:
            raise ValueError("path token statistic trailing shapes differ")
        by_family[evidence_value.family_id].append(statistic64.mean(dim=0))
    family_means = tuple(
        torch.stack(by_family[family]).mean(dim=0)
        for family in sorted(by_family)
    )
    return torch.stack(family_means).mean(dim=0).contiguous()


def _weighted_metrics(
    evidence: Sequence[CompleteH4TailPathTeacherKLEvidence],
) -> dict[str, float]:
    predictions = {
        value.example_id: complete_h4_tail_path_direct_contraction(value)
        for value in evidence
    }
    targets = {
        value.example_id: complete_h4_tail_path_ftc_target(value)
        for value in evidence
    }
    errors = {
        key: predictions[key] - targets[key]
        for key in predictions
    }

    def mean(values: Mapping[str, Tensor]) -> float:
        return float(
            complete_h4_tail_path_family_prompt_token_mean(evidence, values)
        )

    target_second = mean({key: value.square() for key, value in targets.items()})
    prediction_second = mean(
        {key: value.square() for key, value in predictions.items()}
    )
    error_second = mean({key: value.square() for key, value in errors.items()})
    cross = mean(
        {key: predictions[key] * targets[key] for key in predictions}
    )
    target_rms = math.sqrt(max(target_second, 0.0))
    prediction_rms = math.sqrt(max(prediction_second, 0.0))
    rmse = math.sqrt(max(error_second, 0.0))
    epsilon = 64.0 * torch.finfo(torch.float64).eps
    cosine_denominator = target_rms * prediction_rms
    cosine = 0.0 if cosine_denominator <= epsilon else cross / cosine_denominator
    cosine = min(max(cosine, -1.0), 1.0)
    return {
        "mean_target_delta": mean(targets),
        "mean_path_integral": mean(predictions),
        "mean_error": mean(errors),
        "mean_absolute_error": mean(
            {key: value.abs() for key, value in errors.items()}
        ),
        "target_rms": target_rms,
        "path_integral_rms": prediction_rms,
        "rmse": rmse,
        "relative_rmse": rmse / max(target_rms, epsilon),
        "cosine": cosine,
        "maximum_absolute_error": max(
            float(value.abs().max()) for value in errors.values()
        ),
        "relative_rmse_epsilon": epsilon,
    }


@dataclass(frozen=True, slots=True)
class CompleteH4TailPathTeacherKLFamilyClosure:
    family_id: str
    example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    supervised_token_count: int
    mean_target_delta: float
    mean_path_integral: float
    mean_error: float
    mean_absolute_error: float
    target_rms: float
    path_integral_rms: float
    rmse: float
    relative_rmse: float
    cosine: float
    maximum_absolute_error: float
    relative_rmse_epsilon: float

    def __post_init__(self) -> None:
        family = _identifier(self.family_id, label="path closure family_id")
        examples = tuple(
            _identifier(value, label="path closure example_id")
            for value in self.example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="path closure evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
        ):
            raise ValueError("path family closure membership is invalid")
        for name in (
            "mean_target_delta",
            "mean_path_integral",
            "mean_error",
            "mean_absolute_error",
            "target_rms",
            "path_integral_rms",
            "rmse",
            "relative_rmse",
            "cosine",
            "maximum_absolute_error",
            "relative_rmse_epsilon",
        ):
            value = _finite_float(getattr(self, name), label=f"path closure {name}")
            if name in {
                "mean_absolute_error",
                "target_rms",
                "path_integral_rms",
                "rmse",
                "relative_rmse",
                "maximum_absolute_error",
                "relative_rmse_epsilon",
            } and value < 0.0:
                raise ValueError(f"path closure {name} must be nonnegative")
            if name == "cosine" and not -1.0 <= value <= 1.0:
                raise ValueError("path closure cosine must be in [-1, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)

    def metadata(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "example_ids": self.example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "supervised_token_count": self.supervised_token_count,
            "mean_target_delta": self.mean_target_delta,
            "mean_path_integral": self.mean_path_integral,
            "mean_error": self.mean_error,
            "mean_absolute_error": self.mean_absolute_error,
            "target_rms": self.target_rms,
            "path_integral_rms": self.path_integral_rms,
            "rmse": self.rmse,
            "relative_rmse": self.relative_rmse,
            "cosine": self.cosine,
            "maximum_absolute_error": self.maximum_absolute_error,
            "relative_rmse_epsilon": self.relative_rmse_epsilon,
        }


@dataclass(frozen=True, slots=True)
class CompleteH4TailPathTeacherKLClosure:
    evidence_example_ids: tuple[str, ...]
    evidence_artifact_sha256s: tuple[str, ...]
    family_summaries: tuple[CompleteH4TailPathTeacherKLFamilyClosure, ...]
    supervised_token_count: int
    mean_target_delta: float
    mean_path_integral: float
    mean_error: float
    mean_absolute_error: float
    target_rms: float
    path_integral_rms: float
    rmse: float
    relative_rmse: float
    cosine: float
    maximum_absolute_error: float
    relative_rmse_epsilon: float
    artifact_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        examples = tuple(
            _identifier(value, label="path closure example_id")
            for value in self.evidence_example_ids
        )
        hashes = tuple(
            _require_sha256(value, label="path closure evidence artifact")
            for value in self.evidence_artifact_sha256s
        )
        families = tuple(self.family_summaries)
        if (
            not examples
            or examples != tuple(sorted(set(examples)))
            or len(hashes) != len(examples)
            or not families
            or any(
                not isinstance(
                    family, CompleteH4TailPathTeacherKLFamilyClosure
                )
                for family in families
            )
            or tuple(family.family_id for family in families)
            != tuple(sorted({family.family_id for family in families}))
            or set(examples)
            != {
                example
                for family in families
                for example in family.example_ids
            }
            or type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or self.supervised_token_count
            != sum(family.supervised_token_count for family in families)
        ):
            raise ValueError("path closure membership is invalid")
        for name in (
            "mean_target_delta",
            "mean_path_integral",
            "mean_error",
            "mean_absolute_error",
            "target_rms",
            "path_integral_rms",
            "rmse",
            "relative_rmse",
            "cosine",
            "maximum_absolute_error",
            "relative_rmse_epsilon",
        ):
            value = _finite_float(getattr(self, name), label=f"path closure {name}")
            if name in {
                "mean_absolute_error",
                "target_rms",
                "path_integral_rms",
                "rmse",
                "relative_rmse",
                "maximum_absolute_error",
                "relative_rmse_epsilon",
            } and value < 0.0:
                raise ValueError(f"path closure {name} must be nonnegative")
            if name == "cosine" and not -1.0 <= value <= 1.0:
                raise ValueError("path closure cosine must be in [-1, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "evidence_example_ids", examples)
        object.__setattr__(self, "evidence_artifact_sha256s", hashes)
        object.__setattr__(self, "family_summaries", families)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(_CLOSURE_DOMAIN, self.metadata(include_artifact=False)),
        )
        self.validate_integrity()

    def metadata(self, *, include_artifact: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence_example_ids": self.evidence_example_ids,
            "evidence_artifact_sha256s": self.evidence_artifact_sha256s,
            "family_summaries": tuple(
                family.metadata() for family in self.family_summaries
            ),
            "supervised_token_count": self.supervised_token_count,
            "mean_target_delta": self.mean_target_delta,
            "mean_path_integral": self.mean_path_integral,
            "mean_error": self.mean_error,
            "mean_absolute_error": self.mean_absolute_error,
            "target_rms": self.target_rms,
            "path_integral_rms": self.path_integral_rms,
            "rmse": self.rmse,
            "relative_rmse": self.relative_rmse,
            "cosine": self.cosine,
            "maximum_absolute_error": self.maximum_absolute_error,
            "relative_rmse_epsilon": self.relative_rmse_epsilon,
            "weighting": _WEIGHTING,
            "FTC_orientation": "native_KL_minus_source_KL_equals_path_integral",
            "path_objective": _PATH_OBJECTIVE,
            "path_geometry": _PATH_GEOMETRY,
            "gradient_semantics": _GRADIENT_SEMANTICS,
            "endpoint_first_order_result_reused_as_path_closure": False,
            "raw_evidence_serialized": False,
            "hypothesis_use_only": True,
            "authorizes_serving_or_model_mutation": False,
        }
        if include_artifact:
            result["artifact_sha256"] = self.artifact_sha256
        return result

    def validate_integrity(self) -> None:
        if _sha256(
            _CLOSURE_DOMAIN, self.metadata(include_artifact=False)
        ) != _require_sha256(self.artifact_sha256, label="path closure artifact"):
            raise RuntimeError("complete-H4 path teacher-KL closure drifted")


def summarize_complete_h4_tail_path_ftc_closure(
    evidence: Iterable[CompleteH4TailPathTeacherKLEvidence],
) -> CompleteH4TailPathTeacherKLClosure:
    """Summarize GL4 FTC closure with family/prompt/token equal weighting."""

    values = _canonical_evidence(evidence)
    metrics = _weighted_metrics(values)
    by_family: dict[str, list[CompleteH4TailPathTeacherKLEvidence]] = defaultdict(
        list
    )
    for value in values:
        by_family[value.family_id].append(value)
    family_summaries: list[CompleteH4TailPathTeacherKLFamilyClosure] = []
    for family in sorted(by_family):
        members = tuple(sorted(by_family[family], key=lambda value: value.example_id))
        family_metrics = _weighted_metrics(members)
        family_summaries.append(
            CompleteH4TailPathTeacherKLFamilyClosure(
                family_id=family,
                example_ids=tuple(value.example_id for value in members),
                evidence_artifact_sha256s=tuple(
                    value.artifact_sha256 for value in members
                ),
                supervised_token_count=sum(
                    value.supervised_tokens for value in members
                ),
                **family_metrics,
            )
        )
    return CompleteH4TailPathTeacherKLClosure(
        evidence_example_ids=tuple(value.example_id for value in values),
        evidence_artifact_sha256s=tuple(
            value.artifact_sha256 for value in values
        ),
        family_summaries=tuple(family_summaries),
        supervised_token_count=sum(value.supervised_tokens for value in values),
        **metrics,
    )
