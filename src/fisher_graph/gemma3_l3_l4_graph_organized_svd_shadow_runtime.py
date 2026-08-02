"""Strict source-authoritative shadow runtime for the Gemma L3/L4 SVD edge.

The compiled operator is intentionally a *partial* edge.  It transports the
top source-modal contribution from ``layer.3.mlp.normalized_input`` to a
target-modal delta at ``layer.4.mlp.normalized_input``.  It does not replace a
complete transformer block.

Natural-prompt shadow execution therefore uses three source-model passes:

1. capture the native L3 input/output and native L4 input;
2. subtract only ``m3 @ P3[:, :rank].T`` from the native L3 MLP output, leaving
   its unmodelled complement intact, and capture the resulting L4 reference.
3. repeat the clamped-reference path, inject the graph candidate at the L4
   normalized-input boundary, and retain its logits for metrics only.

The graph-organized executor predicts the omitted target-modal delta.  A
right inverse derived from ``R4[:target_modes]`` decodes that delta back to
the L4 input width.  ``P4`` is an L4 *output* prolongation and is never used
at this boundary.

The third pass retains the clamped-reference residual carrier on causally
affected rows and explicitly restores the authoritative native X4 boundary on
all other rows.  It does not claim to replace the missing full-width carrier
contribution.  Every result remains source authoritative.  Candidate tensors
are exposed for measurement only and are never returned as the served output.

This locked rung supports only ``identity`` and ``all_on``.  The routed arm is
rejected because its numerical certificate and accounting are not yet trusted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from .activations import ActivationIntervention
from .adapters.base import AdapterRun
from .adapters.gemma3 import Gemma3CausalLMAdapter
from .gemma3_l3_l4_basis_package import Gemma3L3L4BasisPackage
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    Gemma3GraphOrganizedSVDCandidate,
)
from .graph_organized_svd import (
    GraphOrganizedSVDExecutionAccounting,
    PreparedGraphOrganizedSVD,
)


ShadowArm = Literal["identity", "all_on"]
OracleSuffixRole = Literal["projection_64", "exact_x4_carrier"]
CorrectionWriteScope = Literal[
    "graph_target_affected_mask",
    "complete_h4_causal_support",
]
CompleteH4CorrectionRole = Literal[
    "projection_oracle",
    "exact_h4_ceiling",
]

__all__ = [
    "AuthenticatedCompleteH4CorrectionArmResult",
    "AuthenticatedCompleteH4IdentityAuditResult",
    "AuthenticatedCompleteH4PairResult",
    "AuthenticatedOracleSuffixResult",
    "CorrectionWriteScope",
    "CompleteH4CorrectionRole",
    "COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING",
    "Gemma3L3L4GraphOrganizedSVDShadowAccounting",
    "Gemma3L3L4GraphOrganizedSVDShadowResult",
    "Gemma3L3L4GraphOrganizedSVDShadowRuntime",
    "Gemma3L3L4CorrectionProvider",
    "Gemma3L3L4OnePassBridge",
    "Gemma3L3L4OnePassExecution",
    "Gemma3L3L4OnePassPrefix",
    "Gemma3L3L4TokenNLLVJP",
    "Gemma3L3L4TokenTeacherKLVJP",
    "OracleSuffixRole",
    "ShadowArm",
    "gemma3_l3_l4_shadow_model_inputs_sha256",
    "gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256",
    "validate_gemma3_l3_l4_shadow_model_inputs_sha256",
]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLAN_KEY = "signed_gfa"
_X3_SITE = "layer.3.mlp.normalized_input"
_Y3_SITE = "layer.3.mlp.operator_output"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_ARMS = frozenset({"identity", "all_on"})
_CORRECTION_WRITE_SCOPES = frozenset(
    {
        "graph_target_affected_mask",
        "complete_h4_causal_support",
    }
)


class Gemma3L3L4CorrectionProvider:
    """Nominal interface for an integrity-bound X4 or H4 correction head.

    The bridge accepts a head object rather than a free callback plus a
    caller-asserted hash.  This keeps execution provenance attached to the
    object whose tensors are authenticated immediately before and after use.
    Providers opting into a nondefault write scope must bind that selector in
    their own authenticated artifact.
    """

    __slots__ = ()

    site: str
    artifact_sha256: str
    write_scope: CorrectionWriteScope = "graph_target_affected_mask"

    def validate_integrity(self) -> None:
        raise NotImplementedError

    def correction(
        self,
        prefix: Gemma3L3L4OnePassPrefix,
        realized_state: Tensor,
    ) -> Tensor:
        raise NotImplementedError
_MAX_R4_CONDITION = 1.0e8
_MAX_DUAL_IDENTITY_ERROR = 1.0e-10
_INTERNAL_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-runtime-tensor:v1\0"
)
_RUNTIME_BINDING_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-runtime-binding:v1\0"
)
_EXECUTION_GRID_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-execution-grid:v1\0"
)
_SHADOW_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-result:v1\0"
)
_ORACLE_SUFFIX_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-oracle-suffix-result:v1\0"
)
_COMPLETE_H4_IDENTITY_AUDIT_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-identity-audit-result:v2\0"
)
_COMPLETE_H4_PAIR_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-pair-result:v2\0"
)
_COMPLETE_H4_CORRECTION_ARM_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-correction-arm-result:v2\0"
)
_COMPLETE_H4_NLL_OBJECTIVE_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-nll-objective-receipt:v1\0"
)
_COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-projection-basis:v1\0"
)
_COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_V2_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-projection-basis:v2\0"
)
_COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_V3_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-projection-basis:v3\0"
)
_MODEL_INPUTS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-model-inputs:v1\0"
)
_ONE_PASS_BRIDGE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-one-pass-bridge:v1\0"
)
_ONE_PASS_RESULT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-one-pass-result:v1\0"
)
_LOCKED_FACTORIZED_ADAPTER_EXECUTION_SHA256 = (
    "911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9"
)
_ADAPTER_EXECUTION_BINDING_SCOPES = frozenset(
    {"locked_factorized_refit", "generic_test"}
)
_ORACLE_SUFFIX_ROLES = frozenset(
    {"projection_64", "exact_x4_carrier"}
)
_COMPLETE_H4_CORRECTION_ROLES = frozenset(
    {"projection_oracle", "exact_h4_ceiling"}
)
_COMPLETE_H4_PROJECTION_ORDERING = (
    "descending_fisher_tilted_residual_eigenvalue"
)
COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING = (
    "unweighted_u192_then_tail_residual_svd_span_then_mgs_u320"
)
_COMPLETE_H4_PROJECTION_ORDERINGS = frozenset(
    {
        _COMPLETE_H4_PROJECTION_ORDERING,
        "descending_unweighted_residual_eigenvalue",
        COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
    }
)
_COMPLETE_H4_TAIL_INFORMED_PROJECTION_CONSTRUCTION = (
    "exact_unweighted_u192_prefix_then_full_numerical_svd_row_span_of_"
    "u192_tail_residual_then_two_pass_modified_gram_schmidt_of_"
    "remaining_u193_through_u320"
)
_COMPLETE_H4_PROJECTION_DEFINITION = (
    "cpu_float64_residual_matmul_D_transpose_matmul_D_cast_once"
)
_COMPLETE_H4_CALLBACK_ORDER = (
    "partial_exact_x4.y3",
    "partial_exact_x4.x4",
    "complete_h4.y3",
    "complete_h4.x4",
    "complete_h4.h4",
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_arm(value: object) -> ShadowArm:
    if value == "routed":
        raise ValueError(
            "routed shadow execution is disabled in the locked all-on rung"
        )
    if value not in _ARMS:
        raise ValueError("arm must be identity or all_on")
    return value  # type: ignore[return-value]


def _require_oracle_suffix_role(value: object) -> OracleSuffixRole:
    if value not in _ORACLE_SUFFIX_ROLES:
        raise ValueError(
            "role must be projection_64 or exact_x4_carrier"
        )
    return value  # type: ignore[return-value]


def _require_correction_write_scope(
    value: object,
) -> CorrectionWriteScope:
    if not isinstance(value, str) or value not in _CORRECTION_WRITE_SCOPES:
        raise ValueError(
            "correction write scope must be graph_target_affected_mask "
            "or complete_h4_causal_support"
        )
    return value  # type: ignore[return-value]


def _require_complete_h4_correction_role(
    value: object,
) -> CompleteH4CorrectionRole:
    if value not in _COMPLETE_H4_CORRECTION_ROLES:
        raise ValueError(
            "role must be projection_oracle or exact_h4_ceiling"
        )
    return value  # type: ignore[return-value]


def _runtime_tensor_sha256(value: Tensor) -> str:
    """Hash one runtime tensor including exact execution representation."""

    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_INTERNAL_TENSOR_DOMAIN)
    digest.update(str(value.device).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        str(tuple(int(width) for width in value.shape)).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def gemma3_l3_l4_shadow_model_inputs_sha256(
    model_inputs: Mapping[str, Tensor],
) -> str:
    """Hash the complete tensor-only model-input mapping without mutation."""

    if not isinstance(model_inputs, Mapping):
        raise TypeError("model_inputs must be a mapping")
    materialized = tuple(model_inputs.items())
    for key, value in materialized:
        if not isinstance(key, str) or not key:
            raise ValueError("model_inputs keys must be nonempty strings")
        if not isinstance(value, Tensor):
            raise TypeError("every model_inputs value must be a Tensor")
        if value.layout != torch.strided or value.device.type == "meta":
            raise ValueError(
                "model_inputs tensors must use materialized strided storage"
            )
    if len({key for key, _ in materialized}) != len(materialized):
        raise ValueError("model_inputs mapping yielded duplicate keys")
    ordered = sorted(materialized, key=lambda item: item[0])
    digest = hashlib.sha256()
    digest.update(_MODEL_INPUTS_DOMAIN)
    digest.update(len(ordered).to_bytes(8, "big", signed=False))
    for key, value in ordered:
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        shape_bytes = _canonical_json_bytes(
            tuple(int(width) for width in value.shape)
        )
        canonical = value.detach().to(device="cpu").contiguous()
        raw = canonical.view(torch.uint8).numpy().tobytes(order="C")
        for payload in (key_bytes, dtype_bytes, shape_bytes, raw):
            digest.update(len(payload).to_bytes(8, "big", signed=False))
            digest.update(payload)
    return digest.hexdigest()


def validate_gemma3_l3_l4_shadow_model_inputs_sha256(
    model_inputs: Mapping[str, Tensor],
    expected_sha256: str,
) -> str:
    """Return the input hash after exact comparison with a frozen digest."""

    expected = _require_sha256(
        expected_sha256,
        label="expected model_inputs",
    )
    observed = gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
    if observed != expected:
        raise ValueError("model_inputs SHA-256 differs from shadow result")
    return observed


def _execution_grid_sha256(
    logical_positions: Tensor,
    valid_target_mask: Tensor,
    source_eligible_mask: Tensor,
    target_affected_mask: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(_EXECUTION_GRID_DOMAIN)
    for label, value in (
        ("logical_positions", logical_positions),
        ("valid_target_mask", valid_target_mask),
        ("source_eligible_mask", source_eligible_mask),
        ("target_affected_mask", target_affected_mask),
    ):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(_runtime_tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _basis_copy(value: Gemma3L3L4BasisPackage) -> Gemma3L3L4BasisPackage:
    """Reconstruct the package so mutated frozen-dataclass tensors fail shut."""

    if not isinstance(value, Gemma3L3L4BasisPackage):
        raise TypeError("basis must be a Gemma3L3L4BasisPackage")
    return Gemma3L3L4BasisPackage(
        basis_payload_sha256=value.basis_payload_sha256,
        source_model_sha256=value.source_model_sha256,
        generator_plan_sha256s=value.generator_plan_sha256s,
        layer3_factor_sha256=value.layer3_factor_sha256,
        layer4_factor_sha256=value.layer4_factor_sha256,
        **value.tensors(),
    )


def _same_sequence(left: AdapterRun, right: AdapterRun) -> bool:
    return (
        left.sequence.phase == right.sequence.phase == "prefill"
        and left.sequence.cache_state is None
        and right.sequence.cache_state is None
        and torch.equal(
            left.sequence.logical_positions,
            right.sequence.logical_positions,
        )
        and torch.equal(
            left.sequence.query_valid_mask,
            right.sequence.query_valid_mask,
        )
    )


def _bitwise_equal(left: Tensor, right: Tensor) -> bool:
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        return False
    return torch.equal(
        left.contiguous().view(torch.uint8),
        right.contiguous().view(torch.uint8),
    )


def _tensor_row_difference_mask(left: Tensor, right: Tensor) -> Tensor:
    """Return rows with any byte difference, preserving the live device."""

    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
        or left.ndim != 3
    ):
        raise ValueError("row-difference tensors must share [B, S, D]")
    byte_shape = (*left.shape[:2], -1)
    left_bytes = (
        left.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .reshape(byte_shape)
    )
    right_bytes = (
        right.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .reshape(byte_shape)
    )
    return (left_bytes != right_bytes).any(dim=-1).to(device=left.device)


def _complete_h4_causal_support(
    logical_positions: Tensor,
    valid_target_mask: Tensor,
    source_eligible_mask: Tensor,
) -> Tensor:
    """Close every eligible L3 source over later valid H4 target rows.

    The locked development rung has at most 128 tokens under a 512-token
    layer-4 window.  Its structural H4 support is therefore the full causal
    closure, deliberately distinct from the graph's finite-lag target mask.
    """

    if (
        not isinstance(logical_positions, Tensor)
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or not isinstance(valid_target_mask, Tensor)
        or valid_target_mask.dtype != torch.bool
        or not isinstance(source_eligible_mask, Tensor)
        or source_eligible_mask.dtype != torch.bool
        or logical_positions.shape != valid_target_mask.shape
        or source_eligible_mask.shape != valid_target_mask.shape
        or logical_positions.device != valid_target_mask.device
        or source_eligible_mask.device != valid_target_mask.device
        or logical_positions.ndim != 2
    ):
        raise ValueError("complete-H4 support requires aligned [B, S] tensors")
    if bool((source_eligible_mask & ~valid_target_mask).any()):
        raise ValueError("complete-H4 sources must be valid rows")
    support = torch.zeros_like(valid_target_mask)
    for batch in range(int(logical_positions.shape[0])):
        valid_positions = logical_positions[batch][valid_target_mask[batch]]
        if valid_positions.numel() == 0:
            raise ValueError("complete-H4 support requires valid rows")
        if int(valid_positions[-1] - valid_positions[0]) >= 512:
            raise ValueError(
                "complete-H4 causal closure exceeds the 512-token window"
            )
        source_positions = logical_positions[batch][
            source_eligible_mask[batch]
        ]
        if source_positions.numel() == 0:
            continue
        valid_indices = torch.nonzero(
            valid_target_mask[batch],
            as_tuple=False,
        ).flatten()
        target_positions = logical_positions[batch][valid_indices]
        support[batch, valid_indices] = (
            target_positions.unsqueeze(1)
            >= source_positions.unsqueeze(0)
        ).any(dim=1)
    return support


def _validated_complete_h4_projection_basis(
    projection_basis: Tensor,
    *,
    projection_rank: int,
    projection_ordering: str,
) -> tuple[str, float]:
    if (
        type(projection_rank) is not int
        or projection_rank <= 0
        or not isinstance(projection_ordering, str)
        or projection_ordering not in _COMPLETE_H4_PROJECTION_ORDERINGS
        or not isinstance(projection_basis, Tensor)
        or projection_basis.dtype != torch.float64
        or projection_basis.device.type != "cpu"
        or projection_basis.ndim != 2
        or projection_basis.shape[0] != projection_rank
        or projection_rank > projection_basis.shape[1]
        or projection_basis.numel() == 0
        or not projection_basis.is_contiguous()
        or not bool(torch.isfinite(projection_basis).all())
    ):
        raise ValueError(
            "projection basis must be contiguous CPU float64 [rank, width] "
            "with an authenticated residual-eigenvalue ordering"
        )
    gram = projection_basis @ projection_basis.T
    identity = torch.eye(projection_rank, dtype=torch.float64)
    orthonormal_error = float((gram - identity).abs().max())
    if (
        not math.isfinite(orthonormal_error)
        or orthonormal_error > 1.0e-10
    ):
        raise ValueError("projection basis rows must be orthonormal")
    return _runtime_tensor_sha256(projection_basis), orthonormal_error


def gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
    projection_basis: Tensor,
    *,
    projection_rank: int,
    projection_ordering: str,
) -> str:
    """Authenticate the one allowed complete-H4 projection-basis format."""

    basis_sha256, orthonormal_error = (
        _validated_complete_h4_projection_basis(
            projection_basis,
            projection_rank=projection_rank,
            projection_ordering=projection_ordering,
        )
    )
    payload: dict[str, object] = {
        "schema": "fisher_graph.gemma3_l3_l4_complete_h4_projection_basis",
        "format_version": 1,
        "projection_basis_sha256": basis_sha256,
        "projection_rank": projection_rank,
        "projection_width": int(projection_basis.shape[1]),
        "projection_ordering": projection_ordering,
        "projection_definition": _COMPLETE_H4_PROJECTION_DEFINITION,
        "orthonormal_max_abs_error": orthonormal_error,
    }
    domain = _COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_DOMAIN
    if projection_ordering == "descending_unweighted_residual_eigenvalue":
        payload["format_version"] = 2
        payload["fit_weighting"] = "unweighted"
        domain = _COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_V2_DOMAIN
    elif projection_ordering == COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING:
        payload["format_version"] = 3
        payload["fit_weighting"] = "unweighted"
        payload["basis_construction"] = (
            _COMPLETE_H4_TAIL_INFORMED_PROJECTION_CONSTRUCTION
        )
        domain = _COMPLETE_H4_PROJECTION_BASIS_ARTIFACT_V3_DOMAIN
    return hashlib.sha256(
        domain + _canonical_json_bytes(payload)
    ).hexdigest()


def _complete_h4_nll_objective_receipt_sha256(
    *,
    supervised_indices_sha256: str,
    supervised_targets_sha256: str,
    partial_exact_x4_logits_sha256: str,
    ignore_index: int,
    reduction: str,
    supervised_token_count: int,
    mean_nll: float,
) -> str:
    payload = {
        "schema": "fisher_graph.gemma3_l3_l4_complete_h4_nll_objective",
        "format_version": 1,
        "supervised_indices_sha256": _require_sha256(
            supervised_indices_sha256,
            label="supervised indices",
        ),
        "supervised_targets_sha256": _require_sha256(
            supervised_targets_sha256,
            label="supervised targets",
        ),
        "partial_exact_x4_logits_sha256": _require_sha256(
            partial_exact_x4_logits_sha256,
            label="partial exact-X4 logits",
        ),
        "ignore_index": ignore_index,
        "reduction": reduction,
        "supervised_token_count": supervised_token_count,
        "mean_nll": mean_nll,
    }
    return hashlib.sha256(
        _COMPLETE_H4_NLL_OBJECTIVE_RECEIPT_DOMAIN
        + _canonical_json_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDShadowAccounting:
    """Local edge work; source-model reference-pass work is kept separate."""

    arm: ShadowArm
    batch_size: int
    sequence_length: int
    residual_width: int
    source_modes: int
    target_modes: int
    valid_target_rows: int
    source_eligible_rows: int
    target_affected_rows: int
    target_fallback_rows: int
    model_forward_count: int
    graph: GraphOrganizedSVDExecutionAccounting | None

    def __post_init__(self) -> None:
        _require_arm(self.arm)
        for name in (
            "batch_size",
            "sequence_length",
            "residual_width",
            "source_modes",
            "target_modes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "valid_target_rows",
            "source_eligible_rows",
            "target_affected_rows",
            "target_fallback_rows",
            "model_forward_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        total = self.batch_size * self.sequence_length
        if (
            self.valid_target_rows > total
            or self.source_eligible_rows > self.valid_target_rows
            or self.target_affected_rows > self.valid_target_rows
            or self.target_affected_rows + self.target_fallback_rows != total
        ):
            raise ValueError("source and target coverage is inconsistent")
        if self.arm == "identity":
            if self.graph is not None:
                raise ValueError("identity accounting cannot contain graph work")
        elif self.source_eligible_rows == 0:
            if self.graph is not None:
                raise ValueError("an empty source grid cannot contain graph work")
        elif self.graph is None:
            raise ValueError("compiled arms require graph accounting")
        elif self.graph.valid_source_rows != self.source_eligible_rows:
            raise ValueError("graph source-row accounting differs from the edge")

    @property
    def source_modal_encode_macs(self) -> int:
        if self.arm == "identity":
            return 0
        return (
            self.source_eligible_rows
            * self.residual_width
            * self.source_modes
        )

    @property
    def layer3_partial_decode_macs(self) -> int:
        if self.arm == "identity":
            return 0
        return (
            self.source_eligible_rows
            * self.source_modes
            * self.residual_width
        )

    @property
    def target_dual_decode_macs(self) -> int:
        if self.arm == "identity":
            return 0
        return (
            self.target_affected_rows
            * self.target_modes
            * self.residual_width
        )

    @property
    def bridge_linear_macs(self) -> int:
        return (
            self.source_modal_encode_macs
            + self.layer3_partial_decode_macs
            + self.target_dual_decode_macs
        )

    @property
    def valid_target_fallback_rows(self) -> int:
        return self.valid_target_rows - self.target_affected_rows

    @property
    def padding_rows(self) -> int:
        return (
            self.batch_size * self.sequence_length
            - self.valid_target_rows
        )

    @property
    def graph_factorized_linear_macs(self) -> int:
        return 0 if self.graph is None else self.graph.factorized_linear_macs

    @property
    def local_factorized_linear_macs(self) -> int:
        return self.bridge_linear_macs + self.graph_factorized_linear_macs

    def metadata(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "residual_width": self.residual_width,
            "source_modes": self.source_modes,
            "target_modes": self.target_modes,
            "valid_target_rows": self.valid_target_rows,
            "source_eligible_rows": self.source_eligible_rows,
            "target_affected_rows": self.target_affected_rows,
            "target_fallback_rows": self.target_fallback_rows,
            "valid_target_fallback_rows": (
                self.valid_target_fallback_rows
            ),
            "padding_rows": self.padding_rows,
            "model_forward_count": self.model_forward_count,
            "source_modal_encode_macs": self.source_modal_encode_macs,
            "layer3_partial_decode_macs": self.layer3_partial_decode_macs,
            "target_dual_decode_macs": self.target_dual_decode_macs,
            "bridge_linear_macs": self.bridge_linear_macs,
            "graph_factorized_linear_macs": (
                self.graph_factorized_linear_macs
            ),
            "local_factorized_linear_macs": (
                self.local_factorized_linear_macs
            ),
            "source_model_reference_forward_macs_counted": False,
            "graph": None if self.graph is None else self.graph.metadata(),
        }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDShadowResult:
    """One source-authoritative partial-edge shadow observation."""

    arm: ShadowArm
    authoritative_logits: Tensor | None
    candidate_logits: Tensor | None
    authoritative_x4: Tensor
    candidate_x4: Tensor
    reference_x4: Tensor
    native_y3: Tensor
    clamped_y3: Tensor
    source_modes: Tensor
    predicted_target_modal_delta: Tensor
    logical_positions: Tensor
    valid_target_mask: Tensor
    source_eligible_mask: Tensor
    target_affected_mask: Tensor
    pack_mask: Tensor
    route_scores: Tensor
    runtime_binding_sha256: str
    model_inputs_sha256: str
    layer3_reconstruction_max_abs_error: float
    target_dual_reconstruction_max_abs_error: float
    accounting: Gemma3L3L4GraphOrganizedSVDShadowAccounting
    execution_grid_sha256: str = ""
    result_artifact_sha256: str = ""

    def __post_init__(self) -> None:
        self._validate_structure()
        computed_grid = _execution_grid_sha256(
            self.logical_positions,
            self.valid_target_mask,
            self.source_eligible_mask,
            self.target_affected_mask,
        )
        if self.execution_grid_sha256:
            if (
                _require_sha256(
                    self.execution_grid_sha256,
                    label="execution grid",
                )
                != computed_grid
            ):
                raise ValueError("shadow result execution grid hash mismatch")
        else:
            object.__setattr__(
                self,
                "execution_grid_sha256",
                computed_grid,
            )
        computed_artifact = self._computed_result_artifact_sha256()
        if self.result_artifact_sha256:
            if (
                _require_sha256(
                    self.result_artifact_sha256,
                    label="shadow result artifact",
                )
                != computed_artifact
            ):
                raise ValueError("shadow result artifact hash mismatch")
        else:
            object.__setattr__(
                self,
                "result_artifact_sha256",
                computed_artifact,
            )

    def _validate_structure(self) -> None:
        _require_arm(self.arm)
        if not isinstance(
            self.accounting,
            Gemma3L3L4GraphOrganizedSVDShadowAccounting,
        ):
            raise TypeError("accounting must be strict shadow accounting")
        boundary_shape = self.authoritative_x4.shape
        expected_boundary_shape = (
            self.accounting.batch_size,
            self.accounting.sequence_length,
            self.accounting.residual_width,
        )
        if (
            boundary_shape != expected_boundary_shape
            or self.candidate_x4.shape != boundary_shape
            or self.reference_x4.shape != boundary_shape
            or self.native_y3.shape != boundary_shape
            or self.clamped_y3.shape != boundary_shape
            or any(
                value.device != self.authoritative_x4.device
                or value.dtype != self.authoritative_x4.dtype
                for value in (
                    self.candidate_x4,
                    self.reference_x4,
                    self.native_y3,
                    self.clamped_y3,
                )
            )
            or self.source_eligible_mask.shape != boundary_shape[:2]
            or self.source_eligible_mask.dtype != torch.bool
            or self.target_affected_mask.shape != boundary_shape[:2]
            or self.target_affected_mask.dtype != torch.bool
            or self.source_modes.shape
            != (*boundary_shape[:2], self.accounting.source_modes)
            or self.predicted_target_modal_delta.shape
            != (*boundary_shape[:2], self.accounting.target_modes)
            or self.logical_positions.shape != boundary_shape[:2]
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or self.valid_target_mask.shape != boundary_shape[:2]
            or self.valid_target_mask.dtype != torch.bool
            or self.pack_mask.shape[:2] != boundary_shape[:2]
            or self.route_scores.shape != self.pack_mask.shape
            or self.pack_mask.dtype != torch.bool
            or bool(
                (
                    self.source_eligible_mask
                    & ~self.valid_target_mask
                ).any()
            )
            or bool(
                (
                    self.target_affected_mask
                    & ~self.valid_target_mask
                ).any()
            )
        ):
            raise ValueError("shadow result tensor geometry differs")
        _require_sha256(
            self.runtime_binding_sha256,
            label="runtime binding",
        )
        _require_sha256(
            self.model_inputs_sha256,
            label="model_inputs",
        )
        for value, label in (
            (
                self.layer3_reconstruction_max_abs_error,
                "layer3 reconstruction error",
            ),
            (
                self.target_dual_reconstruction_max_abs_error,
                "target dual reconstruction error",
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and nonnegative")
        if self.accounting.arm != self.arm:
            raise ValueError("result and accounting arms differ")
        if (
            self.accounting.valid_target_rows
            != int(self.valid_target_mask.sum())
            or self.accounting.source_eligible_rows
            != int(self.source_eligible_mask.sum())
            or self.accounting.target_affected_rows
            != int(self.target_affected_mask.sum())
        ):
            raise ValueError("result grid and accounting coverage differ")
        if self.candidate_logits is not None:
            authoritative = self.authoritative_logits
            if (
                authoritative is None
                or authoritative.shape != self.candidate_logits.shape
                or authoritative.device != self.candidate_logits.device
                or authoritative.dtype != self.candidate_logits.dtype
            ):
                raise ValueError("source and candidate logits must align")

    def _computed_result_artifact_sha256(self) -> str:
        tensors: dict[str, Tensor | None] = {
            "authoritative_logits": self.authoritative_logits,
            "candidate_logits": self.candidate_logits,
            "authoritative_x4": self.authoritative_x4,
            "candidate_x4": self.candidate_x4,
            "reference_x4": self.reference_x4,
            "native_y3": self.native_y3,
            "clamped_y3": self.clamped_y3,
            "source_modes": self.source_modes,
            "predicted_target_modal_delta": (
                self.predicted_target_modal_delta
            ),
            "logical_positions": self.logical_positions,
            "valid_target_mask": self.valid_target_mask,
            "source_eligible_mask": self.source_eligible_mask,
            "target_affected_mask": self.target_affected_mask,
            "pack_mask": self.pack_mask,
            "route_scores": self.route_scores,
        }
        payload = {
            "schema": (
                "fisher_graph.gemma3_l3_l4_graph_organized_svd_"
                "shadow_result"
            ),
            "format_version": 1,
            "arm": self.arm,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "layer3_reconstruction_max_abs_error": (
                self.layer3_reconstruction_max_abs_error
            ),
            "target_dual_reconstruction_max_abs_error": (
                self.target_dual_reconstruction_max_abs_error
            ),
            "accounting": self.accounting.metadata(),
            "tensor_sha256s": {
                name: (
                    None
                    if value is None
                    else _runtime_tensor_sha256(value)
                )
                for name, value in tensors.items()
            },
        }
        return hashlib.sha256(
            _SHADOW_RESULT_DOMAIN + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self._validate_structure()
        if (
            _execution_grid_sha256(
                self.logical_positions,
                self.valid_target_mask,
                self.source_eligible_mask,
                self.target_affected_mask,
            )
            != self.execution_grid_sha256
        ):
            raise ValueError("shadow result execution grid hash mismatch")
        _require_sha256(
            self.runtime_binding_sha256,
            label="runtime binding",
        )
        _require_sha256(
            self.model_inputs_sha256,
            label="model_inputs",
        )
        if (
            self._computed_result_artifact_sha256()
            != _require_sha256(
                self.result_artifact_sha256,
                label="shadow result artifact",
            )
        ):
            raise ValueError("shadow result artifact hash mismatch")

    @property
    def output(self) -> Tensor:
        """The only served boundary: the untouched source-model boundary."""

        return self.authoritative_x4

    @property
    def logits(self) -> Tensor | None:
        """The only served logits: the untouched source-model logits."""

        return self.authoritative_logits

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "execution_mode": "shadow",
            "arm": self.arm,
            "authoritative_path": "source",
            "source_outputs_authoritative": True,
            "candidate_outputs_authoritative": False,
            "candidate_boundary_used_for_metrics_only": True,
            "candidate_logits_used_for_metrics_only": (
                self.candidate_logits is not None
            ),
            "candidate_suffix_executed": self.candidate_logits is not None,
            "candidate_suffix_carrier": (
                "clamped_y3_reference_on_affected_rows_with_"
                "authoritative_native_x4_fallback_elsewhere"
            ),
            "full_hidden_state_replacement": False,
            "native_x4_fallback_used": True,
            "native_x4_fallback_scope": (
                "outside_target_affected_mask"
            ),
            "native_x4_fallback_authoritative_oracle": True,
            "native_x4_fallback_used_for_metrics_only": True,
            "candidate_outputs_must_not_be_served": True,
            "partial_edge_only": True,
            "routing_supported": False,
            "routed_execution_rejected": True,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "model_inputs_authenticated": True,
            "execution_grid_sha256": self.execution_grid_sha256,
            "execution_grid_authenticated": True,
            "result_artifact_sha256": self.result_artifact_sha256,
            "result_artifact_authenticated": True,
            "source_boundary": _X3_SITE,
            "source_intervention_boundary": _Y3_SITE,
            "target_boundary": _X4_SITE,
            "unmodelled_layer3_complement_preserved": True,
            "source_knot_interval_applies_to_targets": False,
            "target_rows_selected_by_causal_lag_reachability": True,
            "native_x4_preserved_outside_target_affected_mask": True,
            "target_decoder": "right_inverse_of_R4_restriction",
            "P4_used_as_target_decoder": False,
            "layer3_reconstruction_max_abs_error": (
                self.layer3_reconstruction_max_abs_error
            ),
            "target_dual_reconstruction_max_abs_error": (
                self.target_dual_reconstruction_max_abs_error
            ),
            "accounting": self.accounting.metadata(),
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedOracleSuffixResult:
    """Hash-bound, metrics-only output from one authenticated oracle suffix."""

    role: OracleSuffixRole
    logits: Tensor
    injected_x4_sha256: str
    shadow_result_artifact_sha256: str
    runtime_binding_sha256: str
    execution_grid_sha256: str
    adapter_execution_sha256: str
    logits_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_oracle_suffix_role(self.role)
        if (
            not isinstance(self.logits, Tensor)
            or not self.logits.is_floating_point()
            or self.logits.ndim != 3
            or self.logits.numel() == 0
            or not bool(torch.isfinite(self.logits).all())
        ):
            raise ValueError(
                "oracle suffix logits must be finite floating [B, S, V]"
            )
        for value, label in (
            (self.injected_x4_sha256, "injected X4"),
            (
                self.shadow_result_artifact_sha256,
                "shadow result artifact",
            ),
            (self.runtime_binding_sha256, "runtime binding"),
            (self.execution_grid_sha256, "execution grid"),
            (self.adapter_execution_sha256, "adapter execution"),
        ):
            _require_sha256(value, label=label)
        computed_logits = _runtime_tensor_sha256(self.logits)
        if self.logits_sha256:
            if (
                _require_sha256(
                    self.logits_sha256,
                    label="oracle suffix logits",
                )
                != computed_logits
            ):
                raise ValueError("oracle suffix logits hash mismatch")
        else:
            object.__setattr__(self, "logits_sha256", computed_logits)
        computed_artifact = self._computed_artifact_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="oracle suffix artifact",
                )
                != computed_artifact
            ):
                raise ValueError("oracle suffix artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    def _computed_artifact_sha256(self) -> str:
        payload = {
            "schema": (
                "fisher_graph.gemma3_l3_l4_graph_organized_svd_"
                "oracle_suffix_result"
            ),
            "format_version": 1,
            "role": self.role,
            "injected_x4_sha256": self.injected_x4_sha256,
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "logits_sha256": self.logits_sha256,
            "model_forward_count": 1,
            "metrics_only": True,
        }
        return hashlib.sha256(
            _ORACLE_SUFFIX_RESULT_DOMAIN
            + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        if (
            _runtime_tensor_sha256(self.logits)
            != _require_sha256(
                self.logits_sha256,
                label="oracle suffix logits",
            )
        ):
            raise ValueError("oracle suffix logits hash mismatch")
        if (
            self._computed_artifact_sha256()
            != _require_sha256(
                self.artifact_sha256,
                label="oracle suffix artifact",
            )
        ):
            raise ValueError("oracle suffix artifact hash mismatch")

    def validate_injected_x4(self, value: Tensor) -> None:
        if (
            not isinstance(value, Tensor)
            or _runtime_tensor_sha256(value) != self.injected_x4_sha256
        ):
            raise ValueError("oracle suffix injected X4 hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "role": self.role,
            "execution_mode": "authenticated_oracle_suffix",
            "metrics_only": True,
            "serving_authorized": False,
            "model_forward_count": 1,
            "injected_x4_sha256": self.injected_x4_sha256,
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "logits_sha256": self.logits_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedCompleteH4IdentityAuditResult:
    """Hash-bound, metrics-only audit of the complete layer-4 carrier."""

    partial_exact_x4_logits: Tensor
    complete_h4_logits: Tensor
    incomplete_h4_difference_mask: Tensor
    native_h4_sha256: str
    incomplete_carrier_h4_sha256: str
    injected_h4_sha256: str
    shadow_result_artifact_sha256: str
    runtime_binding_sha256: str
    model_inputs_sha256: str
    execution_grid_sha256: str
    adapter_execution_sha256: str
    target_affected_rows: int
    incomplete_h4_difference_rows: int
    incomplete_h4_difference_valid_rows: int
    incomplete_h4_difference_padding_rows: int
    incomplete_h4_difference_target_rows: int
    incomplete_h4_difference_outside_target_rows: int
    target_affected_h4_difference_observed: bool
    incomplete_h4_difference_nonvacuous: bool
    boundary_callback_order: tuple[str, ...]
    complete_h4_logits_bitwise_authoritative: bool
    complete_h4_max_abs_logit_error: float
    incomplete_h4_difference_mask_sha256: str = ""
    partial_exact_x4_logits_sha256: str = ""
    complete_h4_logits_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        self._validate_structure()
        mask_sha256 = _runtime_tensor_sha256(
            self.incomplete_h4_difference_mask
        )
        partial_sha256 = _runtime_tensor_sha256(
            self.partial_exact_x4_logits
        )
        complete_sha256 = _runtime_tensor_sha256(self.complete_h4_logits)
        if self.incomplete_h4_difference_mask_sha256:
            if (
                _require_sha256(
                    self.incomplete_h4_difference_mask_sha256,
                    label="incomplete-H4 difference mask",
                )
                != mask_sha256
            ):
                raise ValueError(
                    "incomplete-H4 difference mask hash mismatch"
                )
        else:
            object.__setattr__(
                self,
                "incomplete_h4_difference_mask_sha256",
                mask_sha256,
            )
        if self.partial_exact_x4_logits_sha256:
            if (
                _require_sha256(
                    self.partial_exact_x4_logits_sha256,
                    label="partial exact-X4 logits",
                )
                != partial_sha256
            ):
                raise ValueError("partial exact-X4 logits hash mismatch")
        else:
            object.__setattr__(
                self,
                "partial_exact_x4_logits_sha256",
                partial_sha256,
            )
        if self.complete_h4_logits_sha256:
            if (
                _require_sha256(
                    self.complete_h4_logits_sha256,
                    label="complete-H4 logits",
                )
                != complete_sha256
            ):
                raise ValueError("complete-H4 logits hash mismatch")
        else:
            object.__setattr__(
                self,
                "complete_h4_logits_sha256",
                complete_sha256,
            )
        computed_artifact = self._computed_artifact_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="complete-H4 identity audit artifact",
                )
                != computed_artifact
            ):
                raise ValueError(
                    "complete-H4 identity audit artifact hash mismatch"
                )
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    def _validate_structure(self) -> None:
        for value, label in (
            (self.partial_exact_x4_logits, "partial exact-X4 logits"),
            (self.complete_h4_logits, "complete-H4 logits"),
        ):
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.ndim != 3
                or value.numel() == 0
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{label} must be finite floating [B, S, V]")
        if (
            self.partial_exact_x4_logits.shape
            != self.complete_h4_logits.shape
            or self.partial_exact_x4_logits.dtype
            != self.complete_h4_logits.dtype
            or self.partial_exact_x4_logits.device
            != self.complete_h4_logits.device
        ):
            raise ValueError("partial and complete audit logits must align")
        if (
            not isinstance(self.incomplete_h4_difference_mask, Tensor)
            or self.incomplete_h4_difference_mask.shape
            != self.partial_exact_x4_logits.shape[:2]
            or self.incomplete_h4_difference_mask.dtype != torch.bool
            or self.incomplete_h4_difference_mask.device
            != self.partial_exact_x4_logits.device
        ):
            raise ValueError(
                "incomplete-H4 difference mask must be boolean [B, S] "
                "aligned with logits"
            )
        for value, label in (
            (self.native_h4_sha256, "native H4"),
            (
                self.incomplete_carrier_h4_sha256,
                "incomplete-carrier H4",
            ),
            (self.injected_h4_sha256, "injected H4"),
            (
                self.shadow_result_artifact_sha256,
                "shadow result artifact",
            ),
            (self.runtime_binding_sha256, "runtime binding"),
            (self.model_inputs_sha256, "model inputs"),
            (self.execution_grid_sha256, "execution grid"),
            (self.adapter_execution_sha256, "adapter execution"),
        ):
            _require_sha256(value, label=label)
        if self.native_h4_sha256 != self.injected_h4_sha256:
            raise ValueError("injected H4 must be the authenticated native H4")
        if type(self.target_affected_rows) is not int:
            raise TypeError("target_affected_rows must be an integer")
        total_rows = int(self.incomplete_h4_difference_mask.numel())
        if (
            self.target_affected_rows <= 0
            or self.target_affected_rows > total_rows
        ):
            raise ValueError("complete-H4 audit requires affected rows")
        count_names = (
            "incomplete_h4_difference_rows",
            "incomplete_h4_difference_valid_rows",
            "incomplete_h4_difference_padding_rows",
            "incomplete_h4_difference_target_rows",
            "incomplete_h4_difference_outside_target_rows",
        )
        for name in count_names:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        difference_rows = self.incomplete_h4_difference_rows
        if (
            difference_rows
            != int(self.incomplete_h4_difference_mask.sum())
            or difference_rows
            != self.incomplete_h4_difference_valid_rows
            + self.incomplete_h4_difference_padding_rows
            or difference_rows
            != self.incomplete_h4_difference_target_rows
            + self.incomplete_h4_difference_outside_target_rows
            or difference_rows <= 0
            or self.incomplete_h4_difference_target_rows <= 0
            or self.incomplete_h4_difference_valid_rows > difference_rows
            or self.incomplete_h4_difference_padding_rows > difference_rows
            or self.incomplete_h4_difference_target_rows
            > self.target_affected_rows
            or self.incomplete_h4_difference_outside_target_rows
            > total_rows - self.target_affected_rows
        ):
            raise ValueError(
                "incomplete-H4 difference count partition is inconsistent"
            )
        if self.boundary_callback_order != _COMPLETE_H4_CALLBACK_ORDER:
            raise ValueError("complete-H4 boundary callback order differs")
        for value, label in (
            (
                self.target_affected_h4_difference_observed,
                "target-affected H4 difference",
            ),
            (
                self.incomplete_h4_difference_nonvacuous,
                "incomplete-H4 difference nonvacuity",
            ),
        ):
            if value is not True:
                raise ValueError(f"{label} must be proven true")
        if type(self.complete_h4_logits_bitwise_authoritative) is not bool:
            raise TypeError(
                "complete_h4_logits_bitwise_authoritative must be boolean"
            )
        error = self.complete_h4_max_abs_logit_error
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or float(error) < 0.0
        ):
            raise ValueError(
                "complete-H4 max logit error must be finite and nonnegative"
            )
        if (
            self.complete_h4_logits_bitwise_authoritative
            and float(error) != 0.0
        ):
            raise ValueError(
                "bitwise complete-H4 identity requires zero max logit error"
            )

    def _computed_artifact_sha256(self) -> str:
        payload = {
            "schema": (
                "fisher_graph.gemma3_l3_l4_authenticated_complete_h4_"
                "identity_audit_result"
            ),
            "format_version": 2,
            "native_h4_sha256": self.native_h4_sha256,
            "incomplete_carrier_h4_sha256": (
                self.incomplete_carrier_h4_sha256
            ),
            "injected_h4_sha256": self.injected_h4_sha256,
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "target_affected_rows": self.target_affected_rows,
            "incomplete_h4_difference_mask_sha256": (
                self.incomplete_h4_difference_mask_sha256
            ),
            "incomplete_h4_difference_rows": (
                self.incomplete_h4_difference_rows
            ),
            "incomplete_h4_difference_valid_rows": (
                self.incomplete_h4_difference_valid_rows
            ),
            "incomplete_h4_difference_padding_rows": (
                self.incomplete_h4_difference_padding_rows
            ),
            "incomplete_h4_difference_target_rows": (
                self.incomplete_h4_difference_target_rows
            ),
            "incomplete_h4_difference_outside_target_rows": (
                self.incomplete_h4_difference_outside_target_rows
            ),
            "target_affected_h4_difference_observed": (
                self.target_affected_h4_difference_observed
            ),
            "incomplete_h4_difference_nonvacuous": (
                self.incomplete_h4_difference_nonvacuous
            ),
            "boundary_callbacks_exactly_once": True,
            "boundary_callback_order": self.boundary_callback_order,
            "complete_h4_logits_bitwise_authoritative": (
                self.complete_h4_logits_bitwise_authoritative
            ),
            "complete_h4_max_abs_logit_error": (
                self.complete_h4_max_abs_logit_error
            ),
            "partial_exact_x4_logits_sha256": (
                self.partial_exact_x4_logits_sha256
            ),
            "complete_h4_logits_sha256": self.complete_h4_logits_sha256,
            "model_forward_count": 3,
            "metrics_only": True,
            "serving_authorized": False,
        }
        return hashlib.sha256(
            _COMPLETE_H4_IDENTITY_AUDIT_RESULT_DOMAIN
            + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self._validate_structure()
        self.validate_incomplete_h4_difference_mask(
            self.incomplete_h4_difference_mask
        )
        if (
            _runtime_tensor_sha256(self.partial_exact_x4_logits)
            != _require_sha256(
                self.partial_exact_x4_logits_sha256,
                label="partial exact-X4 logits",
            )
        ):
            raise ValueError("partial exact-X4 logits hash mismatch")
        if (
            _runtime_tensor_sha256(self.complete_h4_logits)
            != _require_sha256(
                self.complete_h4_logits_sha256,
                label="complete-H4 logits",
            )
        ):
            raise ValueError("complete-H4 logits hash mismatch")
        if (
            self._computed_artifact_sha256()
            != _require_sha256(
                self.artifact_sha256,
                label="complete-H4 identity audit artifact",
            )
        ):
            raise ValueError(
                "complete-H4 identity audit artifact hash mismatch"
            )

    def validate_incomplete_h4_difference_mask(self, value: Tensor) -> None:
        if (
            not isinstance(value, Tensor)
            or value.shape != self.incomplete_h4_difference_mask.shape
            or value.dtype != torch.bool
            or value.device != self.incomplete_h4_difference_mask.device
            or _runtime_tensor_sha256(value)
            != _require_sha256(
                self.incomplete_h4_difference_mask_sha256,
                label="incomplete-H4 difference mask",
            )
        ):
            raise ValueError("incomplete-H4 difference mask hash mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "execution_mode": "authenticated_complete_h4_identity_audit",
            "metrics_only": True,
            "serving_authorized": False,
            "model_forward_count": 3,
            "native_h4_sha256": self.native_h4_sha256,
            "incomplete_carrier_h4_sha256": (
                self.incomplete_carrier_h4_sha256
            ),
            "injected_h4_sha256": self.injected_h4_sha256,
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "target_affected_rows": self.target_affected_rows,
            "incomplete_h4_difference_mask_sha256": (
                self.incomplete_h4_difference_mask_sha256
            ),
            "incomplete_h4_difference_rows": (
                self.incomplete_h4_difference_rows
            ),
            "incomplete_h4_difference_valid_rows": (
                self.incomplete_h4_difference_valid_rows
            ),
            "incomplete_h4_difference_padding_rows": (
                self.incomplete_h4_difference_padding_rows
            ),
            "incomplete_h4_difference_target_rows": (
                self.incomplete_h4_difference_target_rows
            ),
            "incomplete_h4_difference_outside_target_rows": (
                self.incomplete_h4_difference_outside_target_rows
            ),
            "target_affected_h4_difference_observed": (
                self.target_affected_h4_difference_observed
            ),
            "incomplete_h4_difference_nonvacuous": (
                self.incomplete_h4_difference_nonvacuous
            ),
            "boundary_callbacks_exactly_once": True,
            "boundary_callback_order": self.boundary_callback_order,
            "complete_h4_logits_bitwise_authoritative": (
                self.complete_h4_logits_bitwise_authoritative
            ),
            "complete_h4_max_abs_logit_error": (
                self.complete_h4_max_abs_logit_error
            ),
            "partial_exact_x4_logits_sha256": (
                self.partial_exact_x4_logits_sha256
            ),
            "complete_h4_logits_sha256": self.complete_h4_logits_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedCompleteH4PairResult:
    """Transient, hash-bound native/incomplete H4 fit observation."""

    native_h4: Tensor
    incomplete_h4: Tensor
    h4_gradient: Tensor
    partial_exact_x4_logits_sha256: str
    supervised_indices_sha256: str
    supervised_targets_sha256: str
    supervised_token_count: int
    objective_ignore_index: int
    objective_reduction: str
    objective_mean_nll: float
    objective_receipt_sha256: str
    source_modes: Tensor
    logical_positions: Tensor
    valid_target_mask: Tensor
    source_eligible_mask: Tensor
    target_affected_mask: Tensor
    complete_h4_support_mask: Tensor
    shadow_result_artifact_sha256: str
    runtime_binding_sha256: str
    model_inputs_sha256: str
    execution_grid_sha256: str
    adapter_execution_sha256: str
    boundary_callback_order: tuple[str, ...]
    native_h4_sha256: str = ""
    incomplete_h4_sha256: str = ""
    h4_gradient_sha256: str = ""
    source_modes_sha256: str = ""
    logical_positions_sha256: str = ""
    valid_target_mask_sha256: str = ""
    source_eligible_mask_sha256: str = ""
    target_affected_mask_sha256: str = ""
    complete_h4_support_mask_sha256: str = ""
    artifact_sha256: str = ""

    _TENSOR_HASH_FIELDS = (
        ("native_h4", "native_h4_sha256", "native H4"),
        ("incomplete_h4", "incomplete_h4_sha256", "incomplete H4"),
        ("h4_gradient", "h4_gradient_sha256", "H4 gradient"),
        ("source_modes", "source_modes_sha256", "source modes"),
        (
            "logical_positions",
            "logical_positions_sha256",
            "logical positions",
        ),
        (
            "valid_target_mask",
            "valid_target_mask_sha256",
            "valid-target mask",
        ),
        (
            "source_eligible_mask",
            "source_eligible_mask_sha256",
            "source-eligible mask",
        ),
        (
            "target_affected_mask",
            "target_affected_mask_sha256",
            "target-affected mask",
        ),
        (
            "complete_h4_support_mask",
            "complete_h4_support_mask_sha256",
            "complete-H4 support mask",
        ),
    )

    def __post_init__(self) -> None:
        self._validate_structure()
        for tensor_field, hash_field, label in self._TENSOR_HASH_FIELDS:
            value = getattr(self, tensor_field)
            computed = _runtime_tensor_sha256(value)
            supplied = getattr(self, hash_field)
            if supplied:
                if _require_sha256(supplied, label=label) != computed:
                    raise ValueError(f"{label} hash mismatch")
            else:
                object.__setattr__(self, hash_field, computed)
        computed_artifact = self._computed_artifact_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="complete-H4 pair artifact",
                )
                != computed_artifact
            ):
                raise ValueError("complete-H4 pair artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    def _validate_structure(self) -> None:
        boundary_shape = self.native_h4.shape
        for value, label in (
            (self.native_h4, "native H4"),
            (self.incomplete_h4, "incomplete H4"),
            (self.h4_gradient, "H4 gradient"),
        ):
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.ndim != 3
                or value.numel() == 0
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{label} must be finite floating [B, S, D]")
        if any(
            value.shape != boundary_shape
            or value.dtype != self.native_h4.dtype
            or value.device != self.native_h4.device
            for value in (self.incomplete_h4, self.h4_gradient)
        ):
            raise ValueError("native, incomplete, and gradient H4 must align")
        _require_sha256(
            self.partial_exact_x4_logits_sha256,
            label="partial exact-X4 logits",
        )
        _require_sha256(
            self.supervised_indices_sha256,
            label="supervised indices",
        )
        _require_sha256(
            self.supervised_targets_sha256,
            label="supervised targets",
        )
        if (
            type(self.supervised_token_count) is not int
            or self.supervised_token_count <= 0
            or type(self.objective_ignore_index) is not int
            or self.objective_reduction != "mean"
            or isinstance(self.objective_mean_nll, bool)
            or not isinstance(self.objective_mean_nll, (int, float))
            or not math.isfinite(float(self.objective_mean_nll))
            or float(self.objective_mean_nll) < 0.0
        ):
            raise ValueError("complete-H4 NLL objective metadata differs")
        expected_receipt = _complete_h4_nll_objective_receipt_sha256(
            supervised_indices_sha256=self.supervised_indices_sha256,
            supervised_targets_sha256=self.supervised_targets_sha256,
            partial_exact_x4_logits_sha256=(
                self.partial_exact_x4_logits_sha256
            ),
            ignore_index=self.objective_ignore_index,
            reduction=self.objective_reduction,
            supervised_token_count=self.supervised_token_count,
            mean_nll=float(self.objective_mean_nll),
        )
        if _require_sha256(
            self.objective_receipt_sha256,
            label="NLL objective receipt",
        ) != expected_receipt:
            raise ValueError("complete-H4 NLL objective receipt mismatch")
        grid_shape = boundary_shape[:2]
        if (
            not isinstance(self.source_modes, Tensor)
            or not self.source_modes.is_floating_point()
            or self.source_modes.ndim != 3
            or self.source_modes.shape[:2] != grid_shape
            or self.source_modes.shape[-1] <= 0
            or self.source_modes.device != self.native_h4.device
            or not bool(torch.isfinite(self.source_modes).all())
            or not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.shape != grid_shape
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or self.logical_positions.device != self.native_h4.device
        ):
            raise ValueError("complete-H4 pair modal or position grid differs")
        for value, label in (
            (self.valid_target_mask, "valid-target mask"),
            (self.source_eligible_mask, "source-eligible mask"),
            (self.target_affected_mask, "target-affected mask"),
            (self.complete_h4_support_mask, "complete-H4 support mask"),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.bool
                or value.shape != grid_shape
                or value.device != self.native_h4.device
            ):
                raise ValueError(f"{label} must be aligned boolean [B, S]")
        expected_support = _complete_h4_causal_support(
            self.logical_positions,
            self.valid_target_mask,
            self.source_eligible_mask,
        )
        if not _bitwise_equal(
            self.complete_h4_support_mask,
            expected_support,
        ):
            raise ValueError("complete-H4 support is not the causal closure")
        if bool(
            (self.target_affected_mask & ~self.complete_h4_support_mask).any()
        ):
            raise ValueError("graph target mask escapes complete-H4 support")
        difference = _tensor_row_difference_mask(
            self.native_h4,
            self.incomplete_h4,
        )
        if bool((difference & ~self.valid_target_mask).any()):
            raise ValueError("native/incomplete H4 differs on padding rows")
        if bool(
            (
                difference
                & self.valid_target_mask
                & ~self.complete_h4_support_mask
            ).any()
        ):
            raise ValueError(
                "native/incomplete H4 difference escapes causal support"
            )
        for value, label in (
            (
                self.shadow_result_artifact_sha256,
                "shadow result artifact",
            ),
            (self.runtime_binding_sha256, "runtime binding"),
            (self.model_inputs_sha256, "model inputs"),
            (self.execution_grid_sha256, "execution grid"),
            (self.adapter_execution_sha256, "adapter execution"),
        ):
            _require_sha256(value, label=label)
        if self.boundary_callback_order != (
            "complete_h4_pair.y3",
            "complete_h4_pair.x4",
            "complete_h4_pair.h4",
        ):
            raise ValueError("complete-H4 pair callback order differs")

    def _computed_artifact_sha256(self) -> str:
        payload = {
            "schema": (
                "fisher_graph.gemma3_l3_l4_authenticated_complete_h4_pair"
            ),
            "format_version": 2,
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "boundary_callback_order": self.boundary_callback_order,
            "tensor_sha256s": {
                tensor_field: getattr(self, hash_field)
                for tensor_field, hash_field, _label in self._TENSOR_HASH_FIELDS
            },
            "partial_exact_x4_logits_sha256": (
                self.partial_exact_x4_logits_sha256
            ),
            "supervised_indices_sha256": self.supervised_indices_sha256,
            "supervised_targets_sha256": self.supervised_targets_sha256,
            "supervised_token_count": self.supervised_token_count,
            "objective_ignore_index": self.objective_ignore_index,
            "objective_reduction": self.objective_reduction,
            "objective_mean_nll": self.objective_mean_nll,
            "objective_receipt_sha256": self.objective_receipt_sha256,
            "model_forward_count": 2,
            "fit_only": True,
            "metrics_only": True,
            "serving_authorized": False,
        }
        return hashlib.sha256(
            _COMPLETE_H4_PAIR_RESULT_DOMAIN + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self._validate_structure()
        for tensor_field, hash_field, label in self._TENSOR_HASH_FIELDS:
            if _runtime_tensor_sha256(getattr(self, tensor_field)) != (
                _require_sha256(getattr(self, hash_field), label=label)
            ):
                raise ValueError(f"{label} hash mismatch")
        if self._computed_artifact_sha256() != _require_sha256(
            self.artifact_sha256,
            label="complete-H4 pair artifact",
        ):
            raise ValueError("complete-H4 pair artifact hash mismatch")

    @property
    def incomplete_h4_difference_mask(self) -> Tensor:
        self.validate_integrity()
        return _tensor_row_difference_mask(
            self.native_h4,
            self.incomplete_h4,
        )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        difference = _tensor_row_difference_mask(
            self.native_h4,
            self.incomplete_h4,
        )
        support = self.complete_h4_support_mask
        target = self.target_affected_mask
        return {
            "execution_mode": "authenticated_complete_h4_pair",
            "fit_only": True,
            "metrics_only": True,
            "serving_authorized": False,
            "model_forward_count": 2,
            "support_rule": "valid_target_at_or_after_any_eligible_source",
            "complete_h4_support_rows": int(support.sum()),
            "graph_target_affected_rows": int(target.sum()),
            "complete_h4_support_outside_graph_rows": int(
                (support & ~target).sum()
            ),
            "incomplete_h4_difference_rows": int(difference.sum()),
            "incomplete_h4_difference_valid_rows": int(
                (difference & self.valid_target_mask).sum()
            ),
            "incomplete_h4_difference_padding_rows": int(
                (difference & ~self.valid_target_mask).sum()
            ),
            "incomplete_h4_difference_outside_support_rows": int(
                (difference & ~support).sum()
            ),
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "boundary_callback_order": self.boundary_callback_order,
            "native_h4_sha256": self.native_h4_sha256,
            "incomplete_h4_sha256": self.incomplete_h4_sha256,
            "h4_gradient_sha256": self.h4_gradient_sha256,
            "partial_exact_x4_logits_sha256": (
                self.partial_exact_x4_logits_sha256
            ),
            "supervised_indices_sha256": self.supervised_indices_sha256,
            "supervised_targets_sha256": self.supervised_targets_sha256,
            "supervised_token_count": self.supervised_token_count,
            "objective_ignore_index": self.objective_ignore_index,
            "objective_reduction": self.objective_reduction,
            "objective_mean_nll": self.objective_mean_nll,
            "objective_receipt_sha256": self.objective_receipt_sha256,
            "source_modes_sha256": self.source_modes_sha256,
            "complete_h4_support_mask_sha256": (
                self.complete_h4_support_mask_sha256
            ),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedCompleteH4CorrectionArmResult:
    """One metrics-only replay with an authenticated H4 correction."""

    role: CompleteH4CorrectionRole
    logits: Tensor
    injected_h4_sha256: str
    native_h4_sha256: str
    incomplete_h4_sha256: str
    projected_delta_sha256: str | None
    projection_basis_sha256: str | None
    projection_basis_artifact_sha256: str | None
    projection_fit_basis_artifact_sha256: str | None
    projection_rank: int | None
    projection_ordering: str | None
    projection_definition: str | None
    projection_basis_orthonormal_max_abs_error: float | None
    complete_h4_pair_artifact_sha256: str
    shadow_result_artifact_sha256: str
    runtime_binding_sha256: str
    model_inputs_sha256: str
    execution_grid_sha256: str
    adapter_execution_sha256: str
    complete_h4_support_mask_sha256: str
    boundary_callback_order: tuple[str, ...]
    logits_bitwise_authoritative: bool
    max_abs_authoritative_logit_error: float
    logits_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        self._validate_structure()
        computed_logits = _runtime_tensor_sha256(self.logits)
        if self.logits_sha256:
            if (
                _require_sha256(
                    self.logits_sha256,
                    label="complete-H4 correction logits",
                )
                != computed_logits
            ):
                raise ValueError("complete-H4 correction logits hash mismatch")
        else:
            object.__setattr__(self, "logits_sha256", computed_logits)
        computed_artifact = self._computed_artifact_sha256()
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="complete-H4 correction arm artifact",
                )
                != computed_artifact
            ):
                raise ValueError(
                    "complete-H4 correction arm artifact hash mismatch"
                )
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    def _validate_structure(self) -> None:
        selected_role = _require_complete_h4_correction_role(self.role)
        if (
            not isinstance(self.logits, Tensor)
            or not self.logits.is_floating_point()
            or self.logits.ndim != 3
            or self.logits.numel() == 0
            or not bool(torch.isfinite(self.logits).all())
        ):
            raise ValueError(
                "complete-H4 correction logits must be finite [B, S, V]"
            )
        for value, label in (
            (self.injected_h4_sha256, "injected H4"),
            (self.native_h4_sha256, "native H4"),
            (self.incomplete_h4_sha256, "incomplete H4"),
            (
                self.complete_h4_pair_artifact_sha256,
                "complete-H4 pair artifact",
            ),
            (
                self.shadow_result_artifact_sha256,
                "shadow result artifact",
            ),
            (self.runtime_binding_sha256, "runtime binding"),
            (self.model_inputs_sha256, "model inputs"),
            (self.execution_grid_sha256, "execution grid"),
            (self.adapter_execution_sha256, "adapter execution"),
            (
                self.complete_h4_support_mask_sha256,
                "complete-H4 support mask",
            ),
        ):
            _require_sha256(value, label=label)
        if selected_role == "projection_oracle":
            if (
                self.projected_delta_sha256 is None
                or self.projection_basis_sha256 is None
                or self.projection_basis_artifact_sha256 is None
                or self.projection_fit_basis_artifact_sha256 is None
                or type(self.projection_rank) is not int
                or self.projection_rank <= 0
                or self.projection_ordering
                not in _COMPLETE_H4_PROJECTION_ORDERINGS
                or self.projection_definition
                != _COMPLETE_H4_PROJECTION_DEFINITION
                or isinstance(
                    self.projection_basis_orthonormal_max_abs_error,
                    bool,
                )
                or not isinstance(
                    self.projection_basis_orthonormal_max_abs_error,
                    (int, float),
                )
                or not math.isfinite(
                    float(
                        self.projection_basis_orthonormal_max_abs_error
                    )
                )
                or float(
                    self.projection_basis_orthonormal_max_abs_error
                )
                > 1.0e-10
            ):
                raise ValueError(
                    "projection oracle requires an authenticated basis"
                )
            _require_sha256(
                self.projected_delta_sha256,
                label="projected H4 delta",
            )
            _require_sha256(
                self.projection_basis_sha256,
                label="projection basis",
            )
            _require_sha256(
                self.projection_basis_artifact_sha256,
                label="projection basis artifact",
            )
            _require_sha256(
                self.projection_fit_basis_artifact_sha256,
                label="projection fit-basis artifact",
            )
        elif any(
            value is not None
            for value in (
                self.projected_delta_sha256,
                self.projection_basis_sha256,
                self.projection_basis_artifact_sha256,
                self.projection_fit_basis_artifact_sha256,
                self.projection_rank,
                self.projection_ordering,
                self.projection_definition,
                self.projection_basis_orthonormal_max_abs_error,
            )
        ):
            raise ValueError(
                "exact-H4 ceiling cannot contain projection fields"
            )
        if selected_role == "exact_h4_ceiling" and (
            self.injected_h4_sha256 != self.native_h4_sha256
            or self.logits_bitwise_authoritative is not True
            or float(self.max_abs_authoritative_logit_error) != 0.0
        ):
            raise ValueError(
                "exact-H4 ceiling must inject native H4 and reproduce logits"
            )
        if self.boundary_callback_order != (
            "complete_h4_correction.y3",
            "complete_h4_correction.x4",
            "complete_h4_correction.h4",
        ):
            raise ValueError("complete-H4 correction callback order differs")
        if type(self.logits_bitwise_authoritative) is not bool:
            raise TypeError("logits_bitwise_authoritative must be boolean")
        error = self.max_abs_authoritative_logit_error
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or float(error) < 0.0
        ):
            raise ValueError(
                "authoritative logit error must be finite and nonnegative"
            )
        if self.logits_bitwise_authoritative and float(error) != 0.0:
            raise ValueError("bitwise authoritative logits require zero error")

    def _computed_artifact_sha256(self) -> str:
        payload = {
            "schema": (
                "fisher_graph.gemma3_l3_l4_authenticated_complete_h4_"
                "correction_arm"
            ),
            "format_version": 2,
            "role": self.role,
            "injected_h4_sha256": self.injected_h4_sha256,
            "native_h4_sha256": self.native_h4_sha256,
            "incomplete_h4_sha256": self.incomplete_h4_sha256,
            "projected_delta_sha256": self.projected_delta_sha256,
            "projection_basis_sha256": self.projection_basis_sha256,
            "projection_basis_artifact_sha256": (
                self.projection_basis_artifact_sha256
            ),
            "projection_fit_basis_artifact_sha256": (
                self.projection_fit_basis_artifact_sha256
            ),
            "projection_rank": self.projection_rank,
            "projection_ordering": self.projection_ordering,
            "projection_definition": self.projection_definition,
            "projection_basis_orthonormal_max_abs_error": (
                self.projection_basis_orthonormal_max_abs_error
            ),
            "complete_h4_pair_artifact_sha256": (
                self.complete_h4_pair_artifact_sha256
            ),
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "complete_h4_support_mask_sha256": (
                self.complete_h4_support_mask_sha256
            ),
            "boundary_callback_order": self.boundary_callback_order,
            "logits_bitwise_authoritative": (
                self.logits_bitwise_authoritative
            ),
            "max_abs_authoritative_logit_error": (
                self.max_abs_authoritative_logit_error
            ),
            "logits_sha256": self.logits_sha256,
            "model_forward_count": 1,
            "metrics_only": True,
            "serving_authorized": False,
        }
        return hashlib.sha256(
            _COMPLETE_H4_CORRECTION_ARM_RESULT_DOMAIN
            + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self._validate_structure()
        if _runtime_tensor_sha256(self.logits) != _require_sha256(
            self.logits_sha256,
            label="complete-H4 correction logits",
        ):
            raise ValueError("complete-H4 correction logits hash mismatch")
        if self._computed_artifact_sha256() != _require_sha256(
            self.artifact_sha256,
            label="complete-H4 correction arm artifact",
        ):
            raise ValueError("complete-H4 correction arm artifact hash mismatch")

    def validate_projected_delta(self, value: Tensor) -> None:
        if self.role != "projection_oracle":
            raise ValueError("exact-H4 ceiling has no projected delta")
        if (
            not isinstance(value, Tensor)
            or _runtime_tensor_sha256(value) != self.projected_delta_sha256
        ):
            raise ValueError("projected H4 delta hash mismatch")

    def validate_projection_basis(self, value: Tensor) -> None:
        if self.role != "projection_oracle":
            raise ValueError("exact-H4 ceiling has no projection basis")
        assert self.projection_rank is not None
        assert self.projection_ordering is not None
        basis_sha256, orthonormal_error = (
            _validated_complete_h4_projection_basis(
                value,
                projection_rank=self.projection_rank,
                projection_ordering=self.projection_ordering,
            )
        )
        artifact_sha256 = (
            gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
                value,
                projection_rank=self.projection_rank,
                projection_ordering=self.projection_ordering,
            )
        )
        if (
            basis_sha256 != self.projection_basis_sha256
            or artifact_sha256 != self.projection_basis_artifact_sha256
            or orthonormal_error
            != self.projection_basis_orthonormal_max_abs_error
        ):
            raise ValueError("projection basis authentication mismatch")

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "role": self.role,
            "execution_mode": "authenticated_complete_h4_correction_arm",
            "projection_rank": self.projection_rank,
            "metrics_only": True,
            "serving_authorized": False,
            "model_forward_count": 1,
            "injected_h4_sha256": self.injected_h4_sha256,
            "native_h4_sha256": self.native_h4_sha256,
            "incomplete_h4_sha256": self.incomplete_h4_sha256,
            "projected_delta_sha256": self.projected_delta_sha256,
            "projection_basis_sha256": self.projection_basis_sha256,
            "projection_basis_artifact_sha256": (
                self.projection_basis_artifact_sha256
            ),
            "projection_fit_basis_artifact_sha256": (
                self.projection_fit_basis_artifact_sha256
            ),
            "projection_ordering": self.projection_ordering,
            "projection_definition": self.projection_definition,
            "projection_basis_orthonormal_max_abs_error": (
                self.projection_basis_orthonormal_max_abs_error
            ),
            "complete_h4_pair_artifact_sha256": (
                self.complete_h4_pair_artifact_sha256
            ),
            "shadow_result_artifact_sha256": (
                self.shadow_result_artifact_sha256
            ),
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "execution_grid_sha256": self.execution_grid_sha256,
            "adapter_execution_sha256": self.adapter_execution_sha256,
            "complete_h4_support_mask_sha256": (
                self.complete_h4_support_mask_sha256
            ),
            "boundary_callback_order": self.boundary_callback_order,
            "logits_bitwise_authoritative": (
                self.logits_bitwise_authoritative
            ),
            "max_abs_authoritative_logit_error": (
                self.max_abs_authoritative_logit_error
            ),
            "logits_sha256": self.logits_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


class Gemma3L3L4GraphOrganizedSVDShadowRuntime:
    """Authenticated partial-edge shadow executor with frozen routing policy."""

    def __init__(
        self,
        candidate: Gemma3GraphOrganizedSVDCandidate,
        basis: Gemma3L3L4BasisPackage,
        *,
        expected_candidate_artifact_sha256: str,
        expected_basis_payload_sha256: str,
        expected_plan_artifact_sha256: str,
        expected_live_model_sha256: str,
        expected_adapter_execution_sha256: str,
        adapter_execution_binding_scope: str = "locked_factorized_refit",
        analysis_device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(candidate, Gemma3GraphOrganizedSVDCandidate):
            raise TypeError(
                "candidate must be a Gemma3GraphOrganizedSVDCandidate"
            )
        candidate.validate_integrity()
        if candidate.artifact_sha256 != _require_sha256(
            expected_candidate_artifact_sha256,
            label="expected candidate artifact",
        ):
            raise ValueError("candidate artifact differs from the frozen identity")
        authenticated_basis = _basis_copy(basis)
        if authenticated_basis.basis_payload_sha256 != _require_sha256(
            expected_basis_payload_sha256,
            label="expected basis payload",
        ):
            raise ValueError("basis package differs from the frozen identity")
        try:
            plan_index = candidate.plan_keys.index(_PLAN_KEY)
        except ValueError as error:
            raise ValueError(
                "candidate lacks the signed-GFA deployment plan"
            ) from error
        plan = candidate.plans[plan_index]
        if (
            plan.artifact_sha256
            != _require_sha256(
                expected_plan_artifact_sha256,
                label="expected deployment plan",
            )
            or plan.organization_kind != "signed_gfa_dyadic"
        ):
            raise ValueError("deployment plan identity or organization differs")
        binding = candidate.binding
        basis_binding = authenticated_basis.binding()
        if any(
            binding.get(name) != value
            for name, value in basis_binding.items()
        ):
            raise ValueError("candidate and basis projection lineage differ")
        if (
            candidate.model.get("source_model_sha256")
            != authenticated_basis.source_model_sha256
            or binding.get("residual_width")
            != authenticated_basis.residual_width
            or binding.get("upstream_edge_rank") != plan.source_modes
            or plan.source_modes > authenticated_basis.residual_width
            or plan.target_modes > authenticated_basis.residual_width
        ):
            raise ValueError("candidate model or modal geometry differs")
        expected_scales = (
            authenticated_basis.source_mode_standard_deviations(
                plan.source_modes
            )
        )
        if not torch.equal(plan.source_scales, expected_scales):
            raise ValueError("candidate source scales differ from the basis")

        r4 = authenticated_basis.R4[: plan.target_modes].contiguous()
        singular_values = torch.linalg.svdvals(r4)
        if (
            singular_values.shape != (plan.target_modes,)
            or not bool(torch.isfinite(singular_values).all())
            or float(singular_values[-1]) <= 0.0
        ):
            raise ValueError("R4 target restriction is not full row rank")
        condition = float(singular_values[0] / singular_values[-1])
        if not math.isfinite(condition) or condition > _MAX_R4_CONDITION:
            raise ValueError("R4 target restriction is too ill-conditioned")
        target_decoder = torch.linalg.pinv(
            r4.T,
            atol=0.0,
            rtol=1.0e-12,
        ).contiguous()
        identity_error = float(
            (
                target_decoder @ r4.T
                - torch.eye(plan.target_modes, dtype=torch.float64)
            )
            .abs()
            .max()
        )
        if (
            not math.isfinite(identity_error)
            or identity_error > _MAX_DUAL_IDENTITY_ERROR
        ):
            raise ValueError("R4 target dual failed its right-inverse check")
        runtime_device = torch.device(analysis_device)
        live_model_sha256 = _require_sha256(
            expected_live_model_sha256,
            label="expected live model",
        )
        adapter_execution_sha256 = _require_sha256(
            expected_adapter_execution_sha256,
            label="expected adapter execution",
        )
        if (
            adapter_execution_binding_scope
            not in _ADAPTER_EXECUTION_BINDING_SCOPES
        ):
            raise ValueError(
                "adapter_execution_binding_scope must be "
                "locked_factorized_refit or generic_test"
            )
        if (
            adapter_execution_binding_scope == "locked_factorized_refit"
            and adapter_execution_sha256
            != _LOCKED_FACTORIZED_ADAPTER_EXECUTION_SHA256
        ):
            raise ValueError(
                "locked factorized-refit adapter execution fingerprint differs"
            )
        self._candidate_sha256 = candidate.artifact_sha256
        self._basis_sha256 = authenticated_basis.basis_payload_sha256
        self._plan = plan
        self._plan_sha256 = plan.artifact_sha256
        self._source_model_sha256 = authenticated_basis.source_model_sha256
        self._live_model_sha256 = live_model_sha256
        self._adapter_execution_sha256 = adapter_execution_sha256
        self._adapter_execution_binding_scope = (
            adapter_execution_binding_scope
        )
        self._device = runtime_device
        self._graph: PreparedGraphOrganizedSVD = plan.prepare(
            device=runtime_device,
            dtype=torch.float64,
        )
        self._x3_mean = authenticated_basis.x3_mean.to(
            runtime_device
        ).contiguous().clone()
        self._r3 = authenticated_basis.R3[: plan.source_modes].to(
            runtime_device
        ).contiguous().clone()
        self._p3 = authenticated_basis.P3[:, : plan.source_modes].to(
            runtime_device
        ).contiguous().clone()
        self._r4 = r4.to(runtime_device).contiguous().clone()
        self._target_decoder = target_decoder.to(
            runtime_device
        ).contiguous().clone()
        self._residual_width = authenticated_basis.residual_width
        self._r4_condition = condition
        self._target_dual_identity_error = identity_error
        self._runtime_binding_sha256 = self._computed_runtime_binding_sha256()
        self._expected_runtime_header = self._runtime_header()
        self._expected_internal_tensor_sha256s = {
            name: _runtime_tensor_sha256(value)
            for name, value in self._internal_tensors().items()
        }
        self.validate_integrity()

    @property
    def candidate_artifact_sha256(self) -> str:
        return self._candidate_sha256

    @property
    def basis_payload_sha256(self) -> str:
        return self._basis_sha256

    @property
    def plan_artifact_sha256(self) -> str:
        return self._plan_sha256

    @property
    def source_model_sha256(self) -> str:
        return self._source_model_sha256

    @property
    def live_model_sha256(self) -> str:
        return self._live_model_sha256

    @property
    def adapter_execution_sha256(self) -> str:
        return self._adapter_execution_sha256

    @property
    def adapter_execution_binding_scope(self) -> str:
        return self._adapter_execution_binding_scope

    @property
    def runtime_binding_sha256(self) -> str:
        return self._runtime_binding_sha256

    @property
    def source_modes(self) -> int:
        return self._plan.source_modes

    @property
    def target_modes(self) -> int:
        return self._plan.target_modes

    @property
    def residual_width(self) -> int:
        return self._residual_width

    def _runtime_binding_payload(self) -> dict[str, object]:
        return {
            "schema": (
                "fisher_graph.gemma3_l3_l4_graph_organized_svd_"
                "shadow_runtime_binding"
            ),
            "format_version": 1,
            "candidate_artifact_sha256": self._candidate_sha256,
            "basis_payload_sha256": self._basis_sha256,
            "plan_key": _PLAN_KEY,
            "plan_artifact_sha256": self._plan_sha256,
            "raw_source_model_sha256": self._source_model_sha256,
            "live_factorized_model_sha256": self._live_model_sha256,
            "adapter_execution_sha256": self._adapter_execution_sha256,
            "adapter_execution_binding_scope": (
                self._adapter_execution_binding_scope
            ),
            "analysis_device": str(self._device),
            "residual_width": self._residual_width,
            "source_modes": self._plan.source_modes,
            "source_rank": self._plan.source_rank,
            "target_modes": self._plan.target_modes,
            "fit_knot_origins": self._plan.fit_knot_origins,
            "lag_count": self._plan.lag_count,
            "routing_supported": False,
            "candidate_serving_authorized": False,
            "native_x4_fallback_policy": (
                "authoritative_native_boundary_outside_target_affected_mask"
            ),
        }

    def _computed_runtime_binding_sha256(self) -> str:
        return hashlib.sha256(
            _RUNTIME_BINDING_DOMAIN
            + _canonical_json_bytes(self._runtime_binding_payload())
        ).hexdigest()

    def _runtime_header(self) -> tuple[object, ...]:
        graph = self._graph
        return (
            self._candidate_sha256,
            self._basis_sha256,
            self._plan_sha256,
            self._source_model_sha256,
            self._live_model_sha256,
            self._adapter_execution_sha256,
            self._adapter_execution_binding_scope,
            self._runtime_binding_sha256,
            str(self._device),
            self._residual_width,
            self._r4_condition,
            self._target_dual_identity_error,
            graph.plan_sha256,
            graph.fit_knot_origins,
            graph.source_modes,
            graph.source_rank,
            graph.target_modes,
            graph.pack_count,
            graph.lag_count,
            False,
        )

    def _internal_tensors(self) -> dict[str, Tensor]:
        result = {
            "basis.x3_mean": self._x3_mean,
            "basis.R3": self._r3,
            "basis.P3": self._p3,
            "basis.R4": self._r4,
            "decoder.target_dual": self._target_decoder,
        }
        if type(self._graph) is not PreparedGraphOrganizedSVD:
            raise RuntimeError("prepared graph runtime type drifted")
        graph_buffers = dict(self._graph.named_buffers(recurse=True))
        if set(graph_buffers) != {
            "source_scales",
            "source_basis",
            "knot_cores",
            "core_operator_norm_bounds",
            "pack_offsets",
        }:
            raise RuntimeError("prepared graph buffer set drifted")
        result.update(
            {
                f"graph.{name}": value
                for name, value in graph_buffers.items()
            }
        )
        return result

    def validate_integrity(self) -> None:
        """Re-authenticate every retained plan, scalar, and runtime tensor."""

        try:
            self._plan.validate_integrity()
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                "frozen graph deployment plan drifted after binding"
            ) from error
        if self._plan.artifact_sha256 != self._plan_sha256:
            raise RuntimeError("frozen graph deployment plan identity drifted")
        if not isinstance(self._graph, PreparedGraphOrganizedSVD):
            raise RuntimeError("prepared graph runtime type drifted")
        if dict(self._graph.named_parameters(recurse=True)):
            raise RuntimeError("prepared graph unexpectedly acquired parameters")
        try:
            header = self._runtime_header()
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "shadow runtime execution geometry drifted"
            ) from error
        if header != self._expected_runtime_header:
            raise RuntimeError("shadow runtime execution geometry drifted")
        if (
            self._computed_runtime_binding_sha256()
            != self._runtime_binding_sha256
        ):
            raise RuntimeError("shadow runtime binding payload drifted")
        tensors = self._internal_tensors()
        if set(tensors) != set(self._expected_internal_tensor_sha256s):
            raise RuntimeError("shadow runtime internal tensor set drifted")
        for name, value in tensors.items():
            if (
                not isinstance(value, Tensor)
                or value.device != self._device
                or not value.is_contiguous()
                or (
                    value.is_floating_point()
                    and not bool(torch.isfinite(value).all())
                )
                or _runtime_tensor_sha256(value)
                != self._expected_internal_tensor_sha256s[name]
            ):
                raise RuntimeError(
                    f"shadow runtime internal tensor {name} drifted"
                )

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            "candidate_artifact_sha256": self._candidate_sha256,
            "basis_payload_sha256": self._basis_sha256,
            "plan_key": _PLAN_KEY,
            "plan_artifact_sha256": self._plan_sha256,
            "source_model_sha256": self._source_model_sha256,
            "live_model_sha256": self._live_model_sha256,
            "adapter_execution_sha256": (
                self._adapter_execution_sha256
            ),
            "locked_factorized_adapter_execution_sha256": (
                _LOCKED_FACTORIZED_ADAPTER_EXECUTION_SHA256
            ),
            "adapter_execution_binding_scope": (
                self._adapter_execution_binding_scope
            ),
            "runtime_binding_sha256": self._runtime_binding_sha256,
            "source_model_role": "raw_artifact_lineage",
            "live_model_role": "executed_factorized_model_state",
            "adapter_execution_role": (
                "executed_factorized_model_nontensor_semantics"
            ),
            "source_and_live_model_hashes_may_differ": True,
            "adapter_execution_reauthenticated_before_and_after_every_forward": (
                True
            ),
            "fit_knot_origins": self._plan.fit_knot_origins,
            "source_modes": self.source_modes,
            "target_modes": self.target_modes,
            "residual_width": self.residual_width,
            "partial_edge_only": True,
            "routing_supported": False,
            "routed_execution_rejected": True,
            "validate_on_use_integrity": True,
            "authenticated_internal_tensor_count": len(
                self._expected_internal_tensor_sha256s
            ),
            "native_x4_fallback_policy": (
                "authoritative_native_boundary_outside_target_affected_mask"
            ),
            "native_x4_fallback_used_for_metrics_only": True,
            "R4_right_inverse_condition_number": self._r4_condition,
            "R4_right_inverse_identity_max_abs_error": (
                self._target_dual_identity_error
            ),
            "P4_used_as_target_decoder": False,
            "candidate_serving_authorized": False,
        }

    def export_one_pass_bridge(self) -> Gemma3L3L4OnePassBridge:
        """Clone the authenticated all-on carrier into a one-prefill bridge.

        The returned bridge has deliberately different semantics from this
        measurement runtime: it preserves the clamped reference carrier
        outside graph support and never asks for an authoritative native X4
        fallback.  Consequently its ``execute`` method owns exactly one model
        forward.
        """

        self.validate_integrity()
        return Gemma3L3L4OnePassBridge(self)

    def encode_target_delta(self, full_width_delta: Tensor) -> Tensor:
        """Encode finite full-width X4 deltas into authenticated target modes."""

        self.validate_integrity()
        if (
            not isinstance(full_width_delta, Tensor)
            or not full_width_delta.is_floating_point()
            or full_width_delta.ndim < 1
            or full_width_delta.numel() == 0
            or full_width_delta.shape[-1] != self.residual_width
            or not bool(torch.isfinite(full_width_delta).all())
        ):
            raise ValueError(
                "full_width_delta must be finite floating [..., residual_width]"
            )
        analysis = full_width_delta.to(
            device=self._device,
            dtype=torch.float64,
        )
        encoded = analysis @ self._r4.T
        if not bool(torch.isfinite(encoded).all()):
            raise RuntimeError("target modal encoding became nonfinite")
        self.validate_integrity()
        return encoded

    def decode_target_modal_delta(self, modes: Tensor) -> Tensor:
        """Decode finite target modes with the authenticated R4 right inverse."""

        self.validate_integrity()
        if (
            not isinstance(modes, Tensor)
            or not modes.is_floating_point()
            or modes.ndim < 1
            or modes.numel() == 0
            or modes.shape[-1] != self.target_modes
            or not bool(torch.isfinite(modes).all())
        ):
            raise ValueError(
                "modes must be finite floating [..., target_modes]"
            )
        analysis = modes.to(device=self._device, dtype=torch.float64)
        decoded = analysis @ self._target_decoder
        recovered = decoded @ self._r4.T
        error = float((recovered - analysis).abs().max())
        tolerance = max(float(analysis.abs().max()), 1.0) * 2.0e-10
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError(
                "target-modal dual failed runtime reconstruction"
            )
        self.validate_integrity()
        return decoded

    def validate_result_binding(
        self,
        result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    ) -> None:
        """Fail unless a strict result belongs to this exact frozen runtime."""

        self.validate_integrity()
        if not isinstance(
            result,
            Gemma3L3L4GraphOrganizedSVDShadowResult,
        ):
            raise TypeError("result must be a strict L3/L4 shadow result")
        result.validate_integrity()
        if result.runtime_binding_sha256 != self._runtime_binding_sha256:
            raise ValueError("shadow result belongs to a different runtime")
        if (
            result.accounting.residual_width != self.residual_width
            or result.accounting.source_modes != self.source_modes
            or result.accounting.target_modes != self.target_modes
        ):
            raise ValueError("shadow result execution geometry differs")

    def _grid(
        self,
        x3: Tensor,
        native_y3: Tensor,
        native_x4: Tensor,
        reference_x4: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tensors = (x3, native_y3, native_x4, reference_x4)
        if (
            any(not isinstance(value, Tensor) for value in tensors)
            or x3.ndim != 3
            or any(value.shape != x3.shape for value in tensors[1:])
            or x3.shape[-1] != self.residual_width
            or any(
                value.device != x3.device or value.dtype != x3.dtype
                for value in tensors[1:]
            )
            or not x3.is_floating_point()
        ):
            raise ValueError(
                "x3, native_y3, native_x4, and reference_x4 must share "
                "finite-runtime [B, S, residual_width] geometry"
            )
        if (
            not isinstance(logical_positions, Tensor)
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or logical_positions.device != x3.device
            or logical_positions.shape != x3.shape[:2]
            or not isinstance(valid_mask, Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.device != x3.device
            or valid_mask.shape != x3.shape[:2]
        ):
            raise ValueError(
                "logical_positions and valid_mask must match [B, S]"
            )
        for batch in range(x3.shape[0]):
            positions = logical_positions[batch][valid_mask[batch]]
            if positions.numel() == 0:
                raise ValueError("every sequence must contain a valid row")
            if bool((positions < 0).any()) or (
                positions.numel() > 1
                and not bool(torch.all(positions[1:] > positions[:-1]))
            ):
                raise ValueError(
                    "valid logical positions must be nonnegative and "
                    "strictly increasing"
                )
        minimum = self._plan.fit_knot_origins[0]
        maximum = self._plan.fit_knot_origins[-1]
        source_eligible = (
            valid_mask
            & (logical_positions >= minimum)
            & (logical_positions <= maximum)
        )
        target_affected = torch.zeros_like(valid_mask)
        for batch in range(x3.shape[0]):
            source_positions = logical_positions[batch][
                source_eligible[batch]
            ]
            if source_positions.numel() == 0:
                continue
            target_indices = torch.nonzero(
                valid_mask[batch],
                as_tuple=False,
            ).flatten()
            target_positions = logical_positions[batch][target_indices]
            lags = (
                target_positions.unsqueeze(1)
                - source_positions.unsqueeze(0)
            )
            target_affected[batch, target_indices] = (
                (lags >= 0) & (lags < self._plan.lag_count)
            ).any(dim=1)
        for value, mask, label in (
            (x3, source_eligible, "x3 source"),
            (native_y3, source_eligible, "native_y3 source"),
            (native_x4, target_affected, "native_x4 target"),
            (reference_x4, target_affected, "reference_x4 target"),
        ):
            if bool(mask.any()) and not bool(torch.isfinite(value[mask]).all()):
                raise ValueError(f"{label} rows must be finite")
        return source_eligible, target_affected, logical_positions

    def _source_and_clamp(
        self,
        x3: Tensor,
        native_y3: Tensor,
        source_eligible: Tensor,
    ) -> tuple[Tensor, Tensor, float]:
        shape = (*x3.shape[:2], self.source_modes)
        modes = torch.zeros(shape, dtype=torch.float64, device=self._device)
        delta = torch.zeros(
            x3.shape,
            dtype=torch.float64,
            device=self._device,
        )
        source_analysis = source_eligible.to(self._device)
        if bool(source_analysis.any()):
            x3_analysis = x3[source_eligible].to(
                device=self._device,
                dtype=torch.float64,
            )
            selected_modes = (x3_analysis - self._x3_mean) @ self._r3.T
            modes[source_analysis] = selected_modes
            delta[source_analysis] = selected_modes @ self._p3.T
        native_analysis = native_y3.to(
            device=self._device,
            dtype=torch.float64,
        )
        clamped_analysis = native_analysis - delta
        reconstruction_error = 0.0
        if bool(source_analysis.any()):
            reconstruction_error = float(
                (
                    clamped_analysis[source_analysis]
                    + delta[source_analysis]
                    - native_analysis[source_analysis]
                )
                .abs()
                .max()
        )
        clamped = native_y3.clone()
        if bool(source_eligible.any()):
            clamped[source_eligible] = clamped_analysis[source_analysis].to(
                device=native_y3.device,
                dtype=native_y3.dtype,
            )
            delta_native = delta[source_analysis].to(
                device=native_y3.device,
                dtype=native_y3.dtype,
            )
            restored = clamped[source_eligible] + delta_native
            scale = max(
                float(native_y3[source_eligible].detach().abs().max()),
                float(delta_native.detach().abs().max()),
                1.0,
            )
            tolerance = torch.finfo(native_y3.dtype).eps * scale * 8.0
            actual = float(
                (
                    restored - native_y3[source_eligible]
                ).detach().abs().max()
            )
            if actual > tolerance:
                raise RuntimeError(
                    "partial L3 clamp failed its dtype reconstruction check"
                )
            reconstruction_error = max(reconstruction_error, actual)
        return modes, clamped, reconstruction_error

    def _graph_execute(
        self,
        source_modes: Tensor,
        logical_positions: Tensor,
        valid_target_mask: Tensor,
        source_eligible_mask: Tensor,
        *,
        arm: ShadowArm,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        GraphOrganizedSVDExecutionAccounting | None,
    ]:
        prediction = torch.zeros(
            (*source_eligible_mask.shape, self.target_modes),
            dtype=torch.float64,
            device=self._device,
        )
        pack_mask = torch.zeros(
            (*source_eligible_mask.shape, self._plan.pack_count),
            dtype=torch.bool,
            device=self._device,
        )
        scores = torch.zeros(
            pack_mask.shape,
            dtype=torch.float64,
            device=self._device,
        )
        if arm == "identity":
            return prediction, pack_mask, scores, None
        active_batches = torch.nonzero(
            source_eligible_mask.any(dim=1),
            as_tuple=False,
        ).flatten()
        if active_batches.numel() == 0:
            return prediction, pack_mask, scores, None
        active_on_analysis = active_batches.to(self._device)
        active_source = source_modes.index_select(0, active_on_analysis)
        active_positions = logical_positions.index_select(
            0,
            active_batches.to(logical_positions.device),
        ).to(self._device)
        active_valid = valid_target_mask.index_select(
            0,
            active_batches.to(valid_target_mask.device),
        ).to(self._device)
        active_sources = source_eligible_mask.index_select(
            0,
            active_batches.to(source_eligible_mask.device),
        ).to(self._device)
        if arm != "all_on":
            raise RuntimeError("locked graph execution reached an invalid arm")
        active_prediction = self._graph(
            active_source,
            logical_positions=active_positions,
            valid_mask=active_valid,
            source_mask=active_sources,
        )
        active_pack_mask = active_sources.unsqueeze(-1).expand(
            -1,
            -1,
            self._plan.pack_count,
        ).clone()
        accounting = self._graph.execution_accounting(
            logical_positions=active_positions,
            valid_mask=active_valid,
            pack_mask=active_pack_mask,
            source_mask=active_sources,
        )
        prediction.index_copy_(0, active_on_analysis, active_prediction)
        pack_mask.index_copy_(0, active_on_analysis, active_pack_mask)
        return prediction, pack_mask, scores, accounting

    def execute_boundary_shadow(
        self,
        *,
        x3: Tensor,
        native_y3: Tensor,
        native_x4: Tensor,
        reference_x4: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
        arm: ShadowArm,
        authoritative_logits: Tensor | None = None,
        model_forward_count: int = 0,
        model_inputs_sha256: str | None = None,
    ) -> Gemma3L3L4GraphOrganizedSVDShadowResult:
        """Execute against an already measured clamped L4 reference.

        ``execute_model_shadow`` is the stronger public path because it owns
        and verifies the reference intervention.  This boundary method exists
        for deterministic unit tests and offline activation replay.
        """

        self.validate_integrity()
        selected_arm = _require_arm(arm)
        if type(model_forward_count) is not int or model_forward_count < 0:
            raise ValueError("model_forward_count must be nonnegative")
        bound_model_inputs_sha256 = (
            gemma3_l3_l4_shadow_model_inputs_sha256({})
            if model_inputs_sha256 is None
            else _require_sha256(
                model_inputs_sha256,
                label="boundary model_inputs",
            )
        )
        source_eligible, target_affected, positions = self._grid(
            x3,
            native_y3,
            native_x4,
            reference_x4,
            logical_positions,
            valid_mask,
        )
        if selected_arm == "identity":
            if not _bitwise_equal(reference_x4, native_x4):
                raise RuntimeError(
                    "identity reference does not reproduce the native boundary"
                )
            source_modes = torch.zeros(
                (*source_eligible.shape, self.source_modes),
                dtype=torch.float64,
                device=self._device,
            )
            clamped = native_y3.clone()
            layer3_error = 0.0
        else:
            source_modes, clamped, layer3_error = self._source_and_clamp(
                x3,
                native_y3,
                source_eligible,
            )
        prediction, pack_mask, scores, graph_accounting = (
            self._graph_execute(
                source_modes,
                positions,
                valid_mask,
                source_eligible,
                arm=selected_arm,
            )
        )
        candidate = native_x4.clone()
        dual_error = 0.0
        target_analysis = target_affected.to(self._device)
        if selected_arm != "identity" and bool(target_analysis.any()):
            selected_prediction = prediction[target_analysis]
            decoded = self.decode_target_modal_delta(selected_prediction)
            recovered = decoded @ self._r4.T
            dual_error = float(
                (recovered - selected_prediction).abs().max()
            )
            tolerance = max(
                float(selected_prediction.abs().max()),
                1.0,
            ) * 2.0e-10
            if dual_error > tolerance:
                raise RuntimeError(
                    "target-modal dual failed runtime reconstruction"
                )
            reference_analysis = reference_x4[target_affected].to(
                device=self._device,
                dtype=torch.float64,
            )
            candidate[target_affected] = (reference_analysis + decoded).to(
                device=native_x4.device,
                dtype=native_x4.dtype,
            )
        fallback = ~target_affected
        if not _bitwise_equal(
            candidate[fallback],
            native_x4[fallback],
        ):
            raise RuntimeError("preserved boundary rows were modified")
        accounting = Gemma3L3L4GraphOrganizedSVDShadowAccounting(
            arm=selected_arm,
            batch_size=int(x3.shape[0]),
            sequence_length=int(x3.shape[1]),
            residual_width=self.residual_width,
            source_modes=self.source_modes,
            target_modes=self.target_modes,
            valid_target_rows=int(valid_mask.sum()),
            source_eligible_rows=int(source_eligible.sum()),
            target_affected_rows=int(target_affected.sum()),
            target_fallback_rows=int((~target_affected).sum()),
            model_forward_count=model_forward_count,
            graph=graph_accounting,
        )
        result = Gemma3L3L4GraphOrganizedSVDShadowResult(
            arm=selected_arm,
            authoritative_logits=authoritative_logits,
            candidate_logits=None,
            authoritative_x4=native_x4,
            candidate_x4=candidate,
            reference_x4=reference_x4,
            native_y3=native_y3,
            clamped_y3=clamped,
            source_modes=source_modes,
            predicted_target_modal_delta=prediction,
            logical_positions=positions.detach().clone(),
            valid_target_mask=valid_mask.detach().clone(),
            source_eligible_mask=source_eligible,
            target_affected_mask=target_affected,
            pack_mask=pack_mask,
            route_scores=scores,
            runtime_binding_sha256=self._runtime_binding_sha256,
            model_inputs_sha256=bound_model_inputs_sha256,
            layer3_reconstruction_max_abs_error=layer3_error,
            target_dual_reconstruction_max_abs_error=dual_error,
            accounting=accounting,
        )
        self.validate_integrity()
        return result

    def _authenticate_adapter(
        self,
        adapter: Gemma3CausalLMAdapter,
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if adapter.module.training or any(
            module.training for module in adapter.module.modules()
        ):
            raise ValueError("source Gemma must be completely in eval mode")
        live_model_sha256 = _require_sha256(
            adapter.model_fingerprint(),
            label="live adapter model fingerprint",
        )
        if live_model_sha256 != self._live_model_sha256:
            raise ValueError(
                "live Gemma differs from the frozen execution scope"
            )
        live_execution_sha256 = _require_sha256(
            adapter.execution_fingerprint(),
            label="live adapter execution fingerprint",
        )
        if live_execution_sha256 != self._adapter_execution_sha256:
            raise ValueError(
                "live Gemma execution semantics differ from the frozen scope"
            )
        sites = {site.id: site for site in adapter.activation_sites}
        if (
            _X3_SITE not in sites
            or _Y3_SITE not in sites
            or _X4_SITE not in sites
            or not sites[_Y3_SITE].intervenable
            or not sites[_X4_SITE].intervenable
        ):
            raise ValueError("Gemma adapter L3/L4 intervention ABI drifted")

    def _authenticated_forward(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        capture_sites: tuple[str, ...],
        interventions: Mapping[str, ActivationIntervention] | None = None,
    ) -> AdapterRun:
        """Authenticate model state and execution semantics around one pass."""

        self.validate_integrity()
        self._authenticate_adapter(adapter)
        try:
            with torch.no_grad():
                result = adapter.forward(
                    model_inputs,
                    capture_sites=capture_sites,
                    interventions=interventions,
                    retain_gradients=False,
                )
        finally:
            self._authenticate_adapter(adapter)
            self.validate_integrity()
        return result

    def _authenticate_complete_h4_audit_adapter(
        self,
        adapter: Gemma3CausalLMAdapter,
    ) -> None:
        self._authenticate_adapter(adapter)
        sites = {site.id: site for site in adapter.activation_sites}
        if _H4_SITE not in sites or not sites[_H4_SITE].intervenable:
            raise ValueError(
                "Gemma adapter complete layer-4 output ABI drifted"
            )

    @staticmethod
    def _sequence_matches_shadow_result(
        run: AdapterRun,
        three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    ) -> bool:
        return (
            run.sequence.phase == "prefill"
            and run.sequence.cache_state is None
            and torch.equal(
                run.sequence.logical_positions,
                three_pass_result.logical_positions,
            )
            and torch.equal(
                run.sequence.query_valid_mask,
                three_pass_result.valid_target_mask,
            )
        )

    def validate_complete_h4_pair_binding(
        self,
        pair: AuthenticatedCompleteH4PairResult,
        three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    ) -> None:
        """Fail unless a complete-H4 pair belongs to this exact shadow."""

        self.validate_result_binding(three_pass_result)
        if not isinstance(pair, AuthenticatedCompleteH4PairResult):
            raise TypeError("pair must be an authenticated complete-H4 pair")
        pair.validate_integrity()
        if (
            pair.shadow_result_artifact_sha256
            != three_pass_result.result_artifact_sha256
            or pair.runtime_binding_sha256 != self._runtime_binding_sha256
            or pair.model_inputs_sha256
            != three_pass_result.model_inputs_sha256
            or pair.execution_grid_sha256
            != three_pass_result.execution_grid_sha256
            or pair.adapter_execution_sha256
            != self._adapter_execution_sha256
        ):
            raise ValueError(
                "complete-H4 pair belongs to a different shadow execution"
            )
        pair_device = pair.native_h4.device
        expected_tensors = (
            (
                pair.source_modes,
                three_pass_result.source_modes.to(pair_device),
                "source modes",
            ),
            (
                pair.logical_positions,
                three_pass_result.logical_positions.to(pair_device),
                "logical positions",
            ),
            (
                pair.valid_target_mask,
                three_pass_result.valid_target_mask.to(pair_device),
                "valid-target mask",
            ),
            (
                pair.source_eligible_mask,
                three_pass_result.source_eligible_mask.to(pair_device),
                "source-eligible mask",
            ),
            (
                pair.target_affected_mask,
                three_pass_result.target_affected_mask.to(pair_device),
                "target-affected mask",
            ),
        )
        for observed, expected, label in expected_tensors:
            if not _bitwise_equal(observed, expected):
                raise ValueError(f"complete-H4 pair {label} differs")

    def execute_complete_h4_pair(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
        *,
        supervised_indices: Tensor,
        supervised_targets: Tensor,
        ignore_index: int = -100,
    ) -> AuthenticatedCompleteH4PairResult:
        """Collect an exact-X4 H4 pair with built-in supervised mean NLL.

        This fit-only method first authenticates native X4, H4, and logits.
        Its second pass replays clamped Y3 plus authoritative X4 and cuts the
        autograd graph at incomplete H4 before evaluating canonical mean NLL.
        """

        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self.validate_result_binding(three_pass_result)
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            three_pass_result.model_inputs_sha256,
        )
        if (
            three_pass_result.arm != "all_on"
            or three_pass_result.accounting.model_forward_count != 3
            or three_pass_result.authoritative_logits is None
            or three_pass_result.candidate_logits is None
        ):
            raise ValueError(
                "complete-H4 pair requires a completed all-on three-pass "
                "shadow result"
            )
        if type(ignore_index) is not int:
            raise TypeError("complete-H4 NLL ignore_index must be an integer")
        if (
            not isinstance(supervised_indices, Tensor)
            or supervised_indices.dtype != torch.int64
            or supervised_indices.device.type != "cpu"
            or supervised_indices.ndim != 2
            or supervised_indices.shape[1:] != (2,)
            or supervised_indices.shape[0] <= 0
            or not supervised_indices.is_contiguous()
        ):
            raise ValueError(
                "supervised_indices must be nonempty contiguous CPU int64 "
                "[N, 2]"
            )
        token_count = int(supervised_indices.shape[0])
        if (
            not isinstance(supervised_targets, Tensor)
            or supervised_targets.dtype != torch.int64
            or supervised_targets.device.type != "cpu"
            or supervised_targets.shape != (token_count,)
            or not supervised_targets.is_contiguous()
        ):
            raise ValueError(
                "supervised_targets must be contiguous CPU int64 [N]"
            )
        batch_size, sequence_length = (
            three_pass_result.valid_target_mask.shape
        )
        batches = supervised_indices[:, 0]
        positions = supervised_indices[:, 1]
        if bool(
            (batches < 0).any()
            or (batches >= batch_size).any()
            or (positions < 0).any()
            or (positions >= sequence_length).any()
        ):
            raise ValueError("supervised indices escape the execution grid")
        canonical_ordinals = batches * sequence_length + positions
        if token_count > 1 and not bool(
            torch.all(canonical_ordinals[1:] > canonical_ordinals[:-1])
        ):
            raise ValueError(
                "supervised indices must be unique batch-major sorted"
            )
        valid_cpu = three_pass_result.valid_target_mask.detach().to(
            device="cpu"
        )
        if not bool(valid_cpu[batches, positions].all()):
            raise ValueError("supervised indices include padding rows")
        vocabulary = int(
            three_pass_result.authoritative_logits.shape[-1]
        )
        if bool(
            (supervised_targets == ignore_index).any()
            or (supervised_targets < 0).any()
            or (supervised_targets >= vocabulary).any()
        ):
            raise ValueError(
                "supervised targets must be non-ignored vocabulary ids"
            )
        supervised_indices_sha256 = _runtime_tensor_sha256(
            supervised_indices
        )
        supervised_targets_sha256 = _runtime_tensor_sha256(
            supervised_targets
        )
        indices_snapshot = supervised_indices.detach().clone().contiguous()
        targets_snapshot = supervised_targets.detach().clone().contiguous()
        self._authenticate_complete_h4_audit_adapter(adapter)
        callback_order: list[str] = []

        def record_callback(event: str) -> None:
            expected = (
                "complete_h4_pair.y3",
                "complete_h4_pair.x4",
                "complete_h4_pair.h4",
            )
            index = len(callback_order)
            if index >= len(expected) or expected[index] != event:
                raise RuntimeError(
                    "complete-H4 pair callback repeated or reordered"
                )
            callback_order.append(event)

        native_h4: Tensor | None = None
        incomplete_h4_leaf: Tensor | None = None
        gradient: Tensor | None = None
        native_h4_sha256: str | None = None
        objective_mean_nll: float | None = None
        objective_receipt_sha256: str | None = None
        partial_logits_sha256: str | None = None
        try:
            native = self._authenticated_forward(
                adapter,
                model_inputs,
                capture_sites=(_X4_SITE, _H4_SITE),
            )
            if (
                not self._sequence_matches_shadow_result(
                    native,
                    three_pass_result,
                )
                or not _bitwise_equal(
                    native.activations[_X4_SITE],
                    three_pass_result.authoritative_x4,
                )
                or not _bitwise_equal(
                    native.logits,
                    three_pass_result.authoritative_logits,
                )
            ):
                raise RuntimeError(
                    "native complete-H4 pair replay differs from the "
                    "authenticated source path"
                )
            native_h4 = native.activations[_H4_SITE]
            if (
                native_h4.shape
                != three_pass_result.authoritative_x4.shape
                or native_h4.dtype
                != three_pass_result.authoritative_x4.dtype
                or native_h4.device
                != three_pass_result.authoritative_x4.device
                or not bool(torch.isfinite(native_h4).all())
            ):
                raise RuntimeError(
                    "native H4 does not match the authenticated residual ABI"
                )
            native_h4_sha256 = _runtime_tensor_sha256(native_h4)

            def at_y3(original: Tensor) -> Tensor:
                record_callback("complete_h4_pair.y3")
                if not _bitwise_equal(original, three_pass_result.native_y3):
                    raise RuntimeError(
                        "complete-H4 pair reached a non-authenticated native Y3"
                    )
                return three_pass_result.clamped_y3

            def at_x4(original: Tensor) -> Tensor:
                record_callback("complete_h4_pair.x4")
                if not _bitwise_equal(
                    original,
                    three_pass_result.reference_x4,
                ):
                    raise RuntimeError(
                        "complete-H4 pair reached a non-authenticated "
                        "reference X4"
                    )
                return three_pass_result.authoritative_x4

            def at_h4(original: Tensor) -> Tensor:
                nonlocal incomplete_h4_leaf
                record_callback("complete_h4_pair.h4")
                if incomplete_h4_leaf is not None:
                    raise RuntimeError("complete-H4 pair H4 callback repeated")
                if (
                    original.shape != native_h4.shape
                    or original.dtype != native_h4.dtype
                    or original.device != native_h4.device
                    or not bool(torch.isfinite(original).all())
                ):
                    raise RuntimeError(
                        "incomplete H4 does not match the native residual ABI"
                    )
                incomplete_h4_leaf = (
                    original.detach().requires_grad_(True)
                )
                return incomplete_h4_leaf

            self.validate_integrity()
            self._authenticate_adapter(adapter)
            try:
                with torch.enable_grad():
                    partial = adapter.forward(
                        model_inputs,
                        capture_sites=(),
                        interventions={
                            _Y3_SITE: at_y3,
                            _X4_SITE: at_x4,
                            _H4_SITE: at_h4,
                        },
                        retain_gradients=False,
                    )
                    if incomplete_h4_leaf is None:
                        raise RuntimeError(
                            "complete-H4 pair did not reach the H4 boundary"
                        )
                    partial_logits_sha256 = _runtime_tensor_sha256(
                        partial.logits
                    )
                    indices_on_logits = indices_snapshot.to(
                        device=partial.logits.device
                    )
                    targets_on_logits = targets_snapshot.to(
                        device=partial.logits.device
                    )
                    supervised_logits = partial.logits[
                        indices_on_logits[:, 0],
                        indices_on_logits[:, 1],
                    ]
                    if supervised_logits.dtype in (
                        torch.float16,
                        torch.bfloat16,
                    ):
                        supervised_logits = supervised_logits.float()
                    loss = F.cross_entropy(
                        supervised_logits,
                        targets_on_logits,
                        ignore_index=ignore_index,
                        reduction="mean",
                    )
                    if (
                        not isinstance(loss, Tensor)
                        or loss.ndim != 0
                        or not loss.is_floating_point()
                        or not bool(torch.isfinite(loss))
                    ):
                        raise ValueError(
                            "complete-H4 mean NLL must be one finite scalar"
                        )
                    objective_mean_nll = float(
                        loss.detach().to(device="cpu", dtype=torch.float64)
                    )
                    objective_receipt_sha256 = (
                        _complete_h4_nll_objective_receipt_sha256(
                            supervised_indices_sha256=(
                                supervised_indices_sha256
                            ),
                            supervised_targets_sha256=(
                                supervised_targets_sha256
                            ),
                            partial_exact_x4_logits_sha256=(
                                partial_logits_sha256
                            ),
                            ignore_index=ignore_index,
                            reduction="mean",
                            supervised_token_count=token_count,
                            mean_nll=objective_mean_nll,
                        )
                    )
                    (gradient,) = torch.autograd.grad(
                        loss,
                        (incomplete_h4_leaf,),
                        retain_graph=False,
                        create_graph=False,
                    )
            finally:
                self._authenticate_adapter(adapter)
                self.validate_integrity()
            if not self._sequence_matches_shadow_result(
                partial,
                three_pass_result,
            ):
                raise RuntimeError(
                    "partial complete-H4 pair grid differs from its shadow"
                )
            if tuple(callback_order) != (
                "complete_h4_pair.y3",
                "complete_h4_pair.x4",
                "complete_h4_pair.h4",
            ):
                raise RuntimeError("complete-H4 pair callbacks were skipped")
            if gradient is None or (
                gradient.shape != incomplete_h4_leaf.shape
                or gradient.dtype != incomplete_h4_leaf.dtype
                or gradient.device != incomplete_h4_leaf.device
                or not bool(torch.isfinite(gradient).all())
            ):
                raise RuntimeError("complete-H4 pair gradient geometry differs")
            if (
                partial_logits_sha256 is None
                or objective_mean_nll is None
                or objective_receipt_sha256 is None
            ):
                raise RuntimeError("complete-H4 NLL objective receipt omitted")
            if _runtime_tensor_sha256(native_h4) != native_h4_sha256:
                raise RuntimeError("native H4 drifted during pair collection")

            pair_device = native_h4.device
            logical_positions = three_pass_result.logical_positions.to(
                pair_device
            ).detach().contiguous()
            valid_target_mask = three_pass_result.valid_target_mask.to(
                pair_device
            ).detach().contiguous()
            source_eligible_mask = three_pass_result.source_eligible_mask.to(
                pair_device
            ).detach().contiguous()
            target_affected_mask = three_pass_result.target_affected_mask.to(
                pair_device
            ).detach().contiguous()
            complete_h4_support_mask = _complete_h4_causal_support(
                logical_positions,
                valid_target_mask,
                source_eligible_mask,
            )
            native_snapshot = native_h4.detach().clone().contiguous()
            incomplete_snapshot = (
                incomplete_h4_leaf.detach().clone().contiguous()
            )
            difference = _tensor_row_difference_mask(
                native_snapshot,
                incomplete_snapshot,
            )
            if bool((difference & ~valid_target_mask).any()):
                raise RuntimeError(
                    "native/incomplete H4 differs on padding rows"
                )
            if bool(
                (
                    difference
                    & valid_target_mask
                    & ~complete_h4_support_mask
                ).any()
            ):
                raise RuntimeError(
                    "native/incomplete H4 difference escapes causal support"
                )
            pair = AuthenticatedCompleteH4PairResult(
                native_h4=native_snapshot,
                incomplete_h4=incomplete_snapshot,
                h4_gradient=gradient.detach().clone().contiguous(),
                partial_exact_x4_logits_sha256=(
                    partial_logits_sha256
                ),
                supervised_indices_sha256=supervised_indices_sha256,
                supervised_targets_sha256=supervised_targets_sha256,
                supervised_token_count=token_count,
                objective_ignore_index=ignore_index,
                objective_reduction="mean",
                objective_mean_nll=objective_mean_nll,
                objective_receipt_sha256=objective_receipt_sha256,
                source_modes=(
                    three_pass_result.source_modes.to(pair_device)
                    .detach()
                    .contiguous()
                ),
                logical_positions=logical_positions,
                valid_target_mask=valid_target_mask,
                source_eligible_mask=source_eligible_mask,
                target_affected_mask=target_affected_mask,
                complete_h4_support_mask=complete_h4_support_mask,
                shadow_result_artifact_sha256=(
                    three_pass_result.result_artifact_sha256
                ),
                runtime_binding_sha256=self._runtime_binding_sha256,
                model_inputs_sha256=three_pass_result.model_inputs_sha256,
                execution_grid_sha256=(
                    three_pass_result.execution_grid_sha256
                ),
                adapter_execution_sha256=self._adapter_execution_sha256,
                boundary_callback_order=tuple(callback_order),
            )
            self.validate_complete_h4_pair_binding(
                pair,
                three_pass_result,
            )
            return pair
        finally:
            validate_gemma3_l3_l4_shadow_model_inputs_sha256(
                model_inputs,
                three_pass_result.model_inputs_sha256,
            )
            if (
                _runtime_tensor_sha256(supervised_indices)
                != supervised_indices_sha256
                or _runtime_tensor_sha256(supervised_targets)
                != supervised_targets_sha256
            ):
                raise RuntimeError(
                    "complete-H4 NLL supervision drifted during use"
                )
            self.validate_result_binding(three_pass_result)
            self._authenticate_complete_h4_audit_adapter(adapter)

    def execute_complete_h4_correction_arm(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
        pair: AuthenticatedCompleteH4PairResult,
        projected_delta: Tensor | None = None,
        *,
        role: CompleteH4CorrectionRole,
        projection_basis: Tensor | None = None,
        projection_basis_artifact_sha256: str | None = None,
        projection_fit_basis_artifact_sha256: str | None = None,
        projection_rank: int | None = None,
        projection_ordering: str | None = None,
    ) -> AuthenticatedCompleteH4CorrectionArmResult:
        """Run one exact-X4 suffix with a projected or exact H4 correction."""

        selected_role = _require_complete_h4_correction_role(role)
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self.validate_complete_h4_pair_binding(pair, three_pass_result)
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            three_pass_result.model_inputs_sha256,
        )
        if (
            three_pass_result.arm != "all_on"
            or three_pass_result.accounting.model_forward_count != 3
            or three_pass_result.authoritative_logits is None
            or three_pass_result.candidate_logits is None
        ):
            raise ValueError(
                "complete-H4 correction requires a completed all-on "
                "three-pass shadow result"
        )
        delta_snapshot: Tensor | None = None
        projected_delta_sha256: str | None = None
        basis_snapshot: Tensor | None = None
        projection_basis_sha256: str | None = None
        projection_basis_orthonormal_error: float | None = None
        if selected_role == "projection_oracle":
            if (
                not isinstance(projected_delta, Tensor)
                or not projected_delta.is_floating_point()
                or projected_delta.shape != pair.incomplete_h4.shape
                or projected_delta.dtype != pair.incomplete_h4.dtype
                or projected_delta.device != pair.incomplete_h4.device
                or not bool(torch.isfinite(projected_delta).all())
            ):
                raise ValueError(
                    "projected H4 delta must be finite and align with H4"
                )
            if (
                not isinstance(projection_basis, Tensor)
                or type(projection_rank) is not int
                or not isinstance(projection_ordering, str)
                or not isinstance(projection_basis_artifact_sha256, str)
                or not isinstance(
                    projection_fit_basis_artifact_sha256,
                    str,
                )
            ):
                raise ValueError(
                    "projection oracle requires basis, artifact, rank, and "
                    "ordering"
                )
            projection_basis_sha256, projection_basis_orthonormal_error = (
                _validated_complete_h4_projection_basis(
                    projection_basis,
                    projection_rank=projection_rank,
                    projection_ordering=projection_ordering,
                )
            )
            if int(projection_basis.shape[1]) != int(
                pair.incomplete_h4.shape[-1]
            ):
                raise ValueError("projection basis width differs from H4")
            computed_basis_artifact = (
                gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
                    projection_basis,
                    projection_rank=projection_rank,
                    projection_ordering=projection_ordering,
                )
            )
            if (
                _require_sha256(
                    projection_basis_artifact_sha256,
                    label="projection basis artifact",
                )
                != computed_basis_artifact
            ):
                raise ValueError("projection basis artifact mismatch")
            _require_sha256(
                projection_fit_basis_artifact_sha256,
                label="projection fit-basis artifact",
            )
            basis_snapshot = projection_basis.detach().clone().contiguous()
            support = pair.complete_h4_support_mask
            expected_delta = torch.zeros_like(pair.incomplete_h4)
            if bool(support.any()):
                native_rows = pair.native_h4[support].to(
                    device="cpu",
                    dtype=torch.float64,
                )
                incomplete_rows = pair.incomplete_h4[support].to(
                    device="cpu",
                    dtype=torch.float64,
                )
                residual_rows = native_rows - incomplete_rows
                projected_rows = (
                    residual_rows @ basis_snapshot.T
                ) @ basis_snapshot
                expected_delta[support] = projected_rows.to(
                    device=pair.incomplete_h4.device,
                    dtype=pair.incomplete_h4.dtype,
                )
            if not _bitwise_equal(projected_delta, expected_delta):
                raise ValueError(
                    "submitted H4 delta differs from authenticated projection"
                )
            delta_snapshot = projected_delta.detach().clone().contiguous()
            projected_delta_sha256 = _runtime_tensor_sha256(delta_snapshot)
        elif any(
            value is not None
            for value in (
                projected_delta,
                projection_basis,
                projection_basis_artifact_sha256,
                projection_fit_basis_artifact_sha256,
                projection_rank,
                projection_ordering,
            )
        ):
            raise ValueError(
                "exact-H4 ceiling does not accept projection fields"
            )

        self._authenticate_complete_h4_audit_adapter(adapter)
        callback_order: list[str] = []
        injected_h4: Tensor | None = None

        def record_callback(event: str) -> None:
            expected = (
                "complete_h4_correction.y3",
                "complete_h4_correction.x4",
                "complete_h4_correction.h4",
            )
            index = len(callback_order)
            if index >= len(expected) or expected[index] != event:
                raise RuntimeError(
                    "complete-H4 correction callback repeated or reordered"
                )
            callback_order.append(event)

        def at_y3(original: Tensor) -> Tensor:
            record_callback("complete_h4_correction.y3")
            if not _bitwise_equal(original, three_pass_result.native_y3):
                raise RuntimeError(
                    "complete-H4 correction reached non-authenticated Y3"
                )
            return three_pass_result.clamped_y3

        def at_x4(original: Tensor) -> Tensor:
            record_callback("complete_h4_correction.x4")
            if not _bitwise_equal(original, three_pass_result.reference_x4):
                raise RuntimeError(
                    "complete-H4 correction reached non-authenticated X4"
                )
            return three_pass_result.authoritative_x4

        def at_h4(original: Tensor) -> Tensor:
            nonlocal injected_h4
            record_callback("complete_h4_correction.h4")
            if not _bitwise_equal(original, pair.incomplete_h4):
                raise RuntimeError(
                    "complete-H4 correction reached a non-authenticated "
                    "incomplete carrier"
                )
            if selected_role == "exact_h4_ceiling":
                injected_h4 = pair.native_h4
            else:
                if delta_snapshot is None:
                    raise RuntimeError("projected H4 delta was not retained")
                injected_h4 = original.clone()
                support = pair.complete_h4_support_mask
                if bool(support.any()):
                    injected_h4[support] += delta_snapshot[support]
                if not _bitwise_equal(
                    injected_h4[~support],
                    original[~support],
                ):
                    raise RuntimeError(
                        "H4 correction modified rows outside complete support"
                    )
            return injected_h4

        try:
            corrected = self._authenticated_forward(
                adapter,
                model_inputs,
                capture_sites=(),
                interventions={
                    _Y3_SITE: at_y3,
                    _X4_SITE: at_x4,
                    _H4_SITE: at_h4,
                },
            )
            if not self._sequence_matches_shadow_result(
                corrected,
                three_pass_result,
            ):
                raise RuntimeError(
                    "complete-H4 correction grid differs from its shadow"
                )
            if tuple(callback_order) != (
                "complete_h4_correction.y3",
                "complete_h4_correction.x4",
                "complete_h4_correction.h4",
            ):
                raise RuntimeError("complete-H4 correction callbacks skipped")
            if injected_h4 is None:
                raise RuntimeError("complete-H4 correction omitted injection")
            authoritative = three_pass_result.authoritative_logits
            bitwise = _bitwise_equal(corrected.logits, authoritative)
            max_abs_error = float(
                (
                    corrected.logits.detach().to(dtype=torch.float64)
                    - authoritative.detach().to(dtype=torch.float64)
                )
                .abs()
                .max()
            )
            if selected_role == "exact_h4_ceiling" and (
                not bitwise or max_abs_error != 0.0
            ):
                raise RuntimeError(
                    "exact-H4 ceiling did not reproduce authoritative logits"
                )
            result = AuthenticatedCompleteH4CorrectionArmResult(
                role=selected_role,
                logits=corrected.logits.detach().clone().contiguous(),
                injected_h4_sha256=_runtime_tensor_sha256(injected_h4),
                native_h4_sha256=pair.native_h4_sha256,
                incomplete_h4_sha256=pair.incomplete_h4_sha256,
                projected_delta_sha256=projected_delta_sha256,
                projection_basis_sha256=projection_basis_sha256,
                projection_basis_artifact_sha256=(
                    projection_basis_artifact_sha256
                ),
                projection_fit_basis_artifact_sha256=(
                    projection_fit_basis_artifact_sha256
                ),
                projection_rank=projection_rank,
                projection_ordering=projection_ordering,
                projection_definition=(
                    _COMPLETE_H4_PROJECTION_DEFINITION
                    if selected_role == "projection_oracle"
                    else None
                ),
                projection_basis_orthonormal_max_abs_error=(
                    projection_basis_orthonormal_error
                ),
                complete_h4_pair_artifact_sha256=pair.artifact_sha256,
                shadow_result_artifact_sha256=(
                    three_pass_result.result_artifact_sha256
                ),
                runtime_binding_sha256=self._runtime_binding_sha256,
                model_inputs_sha256=three_pass_result.model_inputs_sha256,
                execution_grid_sha256=(
                    three_pass_result.execution_grid_sha256
                ),
                adapter_execution_sha256=self._adapter_execution_sha256,
                complete_h4_support_mask_sha256=(
                    pair.complete_h4_support_mask_sha256
                ),
                boundary_callback_order=tuple(callback_order),
                logits_bitwise_authoritative=bitwise,
                max_abs_authoritative_logit_error=max_abs_error,
            )
            result.validate_integrity()
            return result
        finally:
            if delta_snapshot is not None and (
                _runtime_tensor_sha256(delta_snapshot)
                != projected_delta_sha256
            ):
                raise RuntimeError("projected H4 delta drifted during use")
            if basis_snapshot is not None and (
                _runtime_tensor_sha256(basis_snapshot)
                != projection_basis_sha256
            ):
                raise RuntimeError("projection basis drifted during use")
            if projection_basis is not None and (
                _runtime_tensor_sha256(projection_basis)
                != projection_basis_sha256
            ):
                raise RuntimeError("projection basis input drifted during use")
            if projected_delta is not None and (
                _runtime_tensor_sha256(projected_delta)
                != projected_delta_sha256
            ):
                raise RuntimeError("projected H4 delta input drifted during use")
            pair.validate_integrity()
            validate_gemma3_l3_l4_shadow_model_inputs_sha256(
                model_inputs,
                three_pass_result.model_inputs_sha256,
            )
            self.validate_result_binding(three_pass_result)
            self._authenticate_complete_h4_audit_adapter(adapter)

    def execute_complete_h4_identity_audit(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    ) -> AuthenticatedCompleteH4IdentityAuditResult:
        """Audit exact X4 versus the complete layer-4 output boundary.

        Three independent, authenticated replays measure the native H4,
        replay the clamped-Y3 carrier with exact native X4, and finally replace
        that replay's incomplete H4 with the native complete carrier.  All
        outputs are metrics-only and can never authorize serving.
        """

        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self.validate_result_binding(three_pass_result)
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            three_pass_result.model_inputs_sha256,
        )
        if (
            three_pass_result.arm != "all_on"
            or three_pass_result.accounting.model_forward_count != 3
            or three_pass_result.authoritative_logits is None
            or three_pass_result.candidate_logits is None
        ):
            raise ValueError(
                "complete-H4 identity audit requires a completed all-on "
                "three-pass shadow result"
            )
        target_affected = three_pass_result.target_affected_mask
        target_affected_rows = int(target_affected.sum())
        if target_affected_rows <= 0:
            raise ValueError(
                "complete-H4 identity audit requires affected rows"
            )
        self._authenticate_complete_h4_audit_adapter(adapter)
        callback_order: list[str] = []

        def record_callback(event: str) -> None:
            next_index = len(callback_order)
            if (
                next_index >= len(_COMPLETE_H4_CALLBACK_ORDER)
                or _COMPLETE_H4_CALLBACK_ORDER[next_index] != event
            ):
                raise RuntimeError(
                    "complete-H4 boundary callback repeated or reordered"
                )
            callback_order.append(event)

        def sequence_matches_result(run: AdapterRun) -> bool:
            return (
                run.sequence.phase == "prefill"
                and run.sequence.cache_state is None
                and torch.equal(
                    run.sequence.logical_positions,
                    three_pass_result.logical_positions,
                )
                and torch.equal(
                    run.sequence.query_valid_mask,
                    three_pass_result.valid_target_mask,
                )
            )

        try:
            native = self._authenticated_forward(
                adapter,
                model_inputs,
                capture_sites=(_X4_SITE, _H4_SITE),
            )
            if (
                not sequence_matches_result(native)
                or not _bitwise_equal(
                    native.activations[_X4_SITE],
                    three_pass_result.authoritative_x4,
                )
                or not _bitwise_equal(
                    native.logits,
                    three_pass_result.authoritative_logits,
                )
            ):
                raise RuntimeError(
                    "native H4 audit replay differs from the authenticated "
                    "source path"
                )
            native_h4 = native.activations[_H4_SITE]
            if (
                native_h4.shape
                != three_pass_result.authoritative_x4.shape
                or native_h4.dtype
                != three_pass_result.authoritative_x4.dtype
                or native_h4.device
                != three_pass_result.authoritative_x4.device
                or not bool(torch.isfinite(native_h4).all())
            ):
                raise RuntimeError(
                    "native H4 does not match the authenticated residual ABI"
                )
            native_h4_sha256 = _runtime_tensor_sha256(native_h4)

            def partial_y3(original: Tensor) -> Tensor:
                record_callback("partial_exact_x4.y3")
                if not _bitwise_equal(
                    original,
                    three_pass_result.native_y3,
                ):
                    raise RuntimeError(
                        "partial exact-X4 replay reached a non-authenticated "
                        "native Y3"
                    )
                return three_pass_result.clamped_y3

            def partial_x4(original: Tensor) -> Tensor:
                record_callback("partial_exact_x4.x4")
                if not _bitwise_equal(
                    original,
                    three_pass_result.reference_x4,
                ):
                    raise RuntimeError(
                        "partial exact-X4 replay reached a non-authenticated "
                        "reference X4"
                    )
                return three_pass_result.authoritative_x4

            partial = self._authenticated_forward(
                adapter,
                model_inputs,
                capture_sites=(_H4_SITE,),
                interventions={
                    _Y3_SITE: partial_y3,
                    _X4_SITE: partial_x4,
                },
            )
            if not sequence_matches_result(partial):
                raise RuntimeError(
                    "partial exact-X4 replay sequence differs from the "
                    "authenticated grid"
                )
            incomplete_h4 = partial.activations[_H4_SITE]
            if (
                incomplete_h4.shape != native_h4.shape
                or incomplete_h4.dtype != native_h4.dtype
                or incomplete_h4.device != native_h4.device
                or not bool(torch.isfinite(incomplete_h4).all())
            ):
                raise RuntimeError(
                    "incomplete H4 does not match the native residual ABI"
                )
            incomplete_h4_sha256 = _runtime_tensor_sha256(incomplete_h4)
            byte_shape = (*native_h4.shape[:2], -1)
            native_h4_bytes = native_h4.detach().to(
                device="cpu"
            ).contiguous().view(torch.uint8).reshape(byte_shape)
            incomplete_h4_bytes = incomplete_h4.detach().to(
                device="cpu"
            ).contiguous().view(torch.uint8).reshape(byte_shape)
            incomplete_h4_difference_mask = (
                (native_h4_bytes != incomplete_h4_bytes)
                .any(dim=-1)
                .to(device=native_h4.device)
            )
            valid_target = three_pass_result.valid_target_mask.to(
                device=incomplete_h4_difference_mask.device
            )
            target_support = target_affected.to(
                device=incomplete_h4_difference_mask.device
            )
            difference_rows = int(incomplete_h4_difference_mask.sum())
            difference_valid_rows = int(
                (incomplete_h4_difference_mask & valid_target).sum()
            )
            difference_padding_rows = int(
                (incomplete_h4_difference_mask & ~valid_target).sum()
            )
            difference_target_rows = int(
                (incomplete_h4_difference_mask & target_support).sum()
            )
            difference_outside_target_rows = int(
                (incomplete_h4_difference_mask & ~target_support).sum()
            )
            target_difference_observed = difference_target_rows > 0
            difference_nonvacuous = difference_rows > 0
            if not target_difference_observed or not difference_nonvacuous:
                raise RuntimeError(
                    "partial exact-X4 H4 audit is vacuous on affected rows"
                )

            def complete_y3(original: Tensor) -> Tensor:
                record_callback("complete_h4.y3")
                if not _bitwise_equal(
                    original,
                    three_pass_result.native_y3,
                ):
                    raise RuntimeError(
                        "complete-H4 replay reached a non-authenticated "
                        "native Y3"
                    )
                return three_pass_result.clamped_y3

            def complete_x4(original: Tensor) -> Tensor:
                record_callback("complete_h4.x4")
                if not _bitwise_equal(
                    original,
                    three_pass_result.reference_x4,
                ):
                    raise RuntimeError(
                        "complete-H4 replay reached a non-authenticated "
                        "reference X4"
                    )
                return three_pass_result.authoritative_x4

            def complete_h4(original: Tensor) -> Tensor:
                record_callback("complete_h4.h4")
                if not _bitwise_equal(original, incomplete_h4):
                    raise RuntimeError(
                        "complete-H4 replay reached a non-authenticated "
                        "incomplete carrier"
                    )
                if _runtime_tensor_sha256(native_h4) != native_h4_sha256:
                    raise RuntimeError(
                        "authenticated native H4 drifted before injection"
                    )
                return native_h4

            complete = self._authenticated_forward(
                adapter,
                model_inputs,
                capture_sites=(),
                interventions={
                    _Y3_SITE: complete_y3,
                    _X4_SITE: complete_x4,
                    _H4_SITE: complete_h4,
                },
            )
            if not sequence_matches_result(complete):
                raise RuntimeError(
                    "complete-H4 replay sequence differs from the "
                    "authenticated grid"
                )
            observed_callback_order = tuple(callback_order)
            if observed_callback_order != _COMPLETE_H4_CALLBACK_ORDER:
                raise RuntimeError(
                    "complete-H4 boundary callbacks were skipped"
                )
            injected_h4_sha256 = _runtime_tensor_sha256(native_h4)
            if injected_h4_sha256 != native_h4_sha256:
                raise RuntimeError("injected native H4 drifted during use")
            complete_bitwise = _bitwise_equal(
                complete.logits,
                three_pass_result.authoritative_logits,
            )
            complete_max_abs_error = float(
                (
                    complete.logits.detach().to(dtype=torch.float64)
                    - three_pass_result.authoritative_logits.detach().to(
                        dtype=torch.float64
                    )
                )
                .abs()
                .max()
            )
            result = AuthenticatedCompleteH4IdentityAuditResult(
                partial_exact_x4_logits=partial.logits,
                complete_h4_logits=complete.logits,
                incomplete_h4_difference_mask=(
                    incomplete_h4_difference_mask
                ),
                native_h4_sha256=native_h4_sha256,
                incomplete_carrier_h4_sha256=incomplete_h4_sha256,
                injected_h4_sha256=injected_h4_sha256,
                shadow_result_artifact_sha256=(
                    three_pass_result.result_artifact_sha256
                ),
                runtime_binding_sha256=self._runtime_binding_sha256,
                model_inputs_sha256=three_pass_result.model_inputs_sha256,
                execution_grid_sha256=(
                    three_pass_result.execution_grid_sha256
                ),
                adapter_execution_sha256=self._adapter_execution_sha256,
                target_affected_rows=target_affected_rows,
                incomplete_h4_difference_rows=difference_rows,
                incomplete_h4_difference_valid_rows=(
                    difference_valid_rows
                ),
                incomplete_h4_difference_padding_rows=(
                    difference_padding_rows
                ),
                incomplete_h4_difference_target_rows=(
                    difference_target_rows
                ),
                incomplete_h4_difference_outside_target_rows=(
                    difference_outside_target_rows
                ),
                target_affected_h4_difference_observed=(
                    target_difference_observed
                ),
                incomplete_h4_difference_nonvacuous=(
                    difference_nonvacuous
                ),
                boundary_callback_order=observed_callback_order,
                complete_h4_logits_bitwise_authoritative=complete_bitwise,
                complete_h4_max_abs_logit_error=complete_max_abs_error,
            )
            result.validate_integrity()
            return result
        finally:
            validate_gemma3_l3_l4_shadow_model_inputs_sha256(
                model_inputs,
                three_pass_result.model_inputs_sha256,
            )
            self.validate_result_binding(three_pass_result)
            self._authenticate_complete_h4_audit_adapter(adapter)

    def execute_oracle_suffix(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
        injected_x4: Tensor,
        *,
        role: OracleSuffixRole,
    ) -> AuthenticatedOracleSuffixResult:
        """Run one hash-bound suffix from an authenticated X4 injection.

        This is a metrics-only oracle facility.  It replays the measured
        clamped-Y3 carrier, verifies both intervention boundaries against the
        strict three-pass result, and never authorizes its logits for serving.
        """

        selected_role = _require_oracle_suffix_role(role)
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self.validate_result_binding(three_pass_result)
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            three_pass_result.model_inputs_sha256,
        )
        if (
            three_pass_result.accounting.model_forward_count != 3
            or three_pass_result.candidate_logits is None
        ):
            raise ValueError(
                "oracle suffix requires a completed three-pass shadow result"
            )
        reference_x4 = three_pass_result.reference_x4
        if (
            not isinstance(injected_x4, Tensor)
            or injected_x4.shape != reference_x4.shape
            or injected_x4.dtype != reference_x4.dtype
            or injected_x4.device != reference_x4.device
            or not injected_x4.is_floating_point()
        ):
            raise ValueError(
                "injected_x4 must match the authenticated X4 boundary"
            )
        valid_mask = three_pass_result.valid_target_mask
        if bool(valid_mask.any()) and not bool(
            torch.isfinite(injected_x4[valid_mask]).all()
        ):
            raise ValueError("valid injected X4 rows must be finite")
        injected_sha256 = _runtime_tensor_sha256(injected_x4)
        reached_native_y3 = False
        reached_reference_x4 = False

        def intervene_y3(original: Tensor) -> Tensor:
            nonlocal reached_native_y3
            if not _bitwise_equal(
                original,
                three_pass_result.native_y3,
            ):
                raise RuntimeError(
                    "oracle suffix reached a non-authenticated native Y3"
                )
            reached_native_y3 = True
            return three_pass_result.clamped_y3

        def intervene_x4(original: Tensor) -> Tensor:
            nonlocal reached_reference_x4
            if not _bitwise_equal(original, reference_x4):
                raise RuntimeError(
                    "oracle suffix reached a non-authenticated reference X4"
                )
            if _runtime_tensor_sha256(injected_x4) != injected_sha256:
                raise RuntimeError(
                    "oracle suffix injected X4 drifted before intervention"
                )
            reached_reference_x4 = True
            return injected_x4

        suffix = self._authenticated_forward(
            adapter,
            model_inputs,
            capture_sites=(_X3_SITE,),
            interventions={
                _Y3_SITE: intervene_y3,
                _X4_SITE: intervene_x4,
            },
        )
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            three_pass_result.model_inputs_sha256,
        )
        if not reached_native_y3 or not reached_reference_x4:
            raise RuntimeError(
                "oracle suffix did not reach both authenticated boundaries"
            )
        if (
            suffix.sequence.phase != "prefill"
            or suffix.sequence.cache_state is not None
            or not torch.equal(
                suffix.sequence.logical_positions,
                three_pass_result.logical_positions,
            )
            or not torch.equal(
                suffix.sequence.query_valid_mask,
                three_pass_result.valid_target_mask,
            )
        ):
            raise RuntimeError(
                "oracle suffix sequence differs from the authenticated grid"
            )
        if _runtime_tensor_sha256(injected_x4) != injected_sha256:
            raise RuntimeError("oracle suffix injected X4 drifted during use")
        self.validate_result_binding(three_pass_result)
        result = AuthenticatedOracleSuffixResult(
            role=selected_role,
            logits=suffix.logits,
            injected_x4_sha256=injected_sha256,
            shadow_result_artifact_sha256=(
                three_pass_result.result_artifact_sha256
            ),
            runtime_binding_sha256=self._runtime_binding_sha256,
            execution_grid_sha256=(
                three_pass_result.execution_grid_sha256
            ),
            adapter_execution_sha256=self._adapter_execution_sha256,
        )
        result.validate_injected_x4(injected_x4)
        self.validate_integrity()
        return result

    def execute_model_shadow(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        arm: ShadowArm,
    ) -> Gemma3L3L4GraphOrganizedSVDShadowResult:
        """Run native plus verified reference passes; always serve native data."""

        selected_arm = _require_arm(arm)
        if not isinstance(model_inputs, Mapping):
            raise TypeError("model_inputs must be a mapping")
        self.validate_integrity()
        model_inputs_sha256 = (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        )
        capture = (_X3_SITE, _Y3_SITE, _X4_SITE)
        native = self._authenticated_forward(
            adapter,
            model_inputs,
            capture_sites=capture,
        )
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            model_inputs_sha256,
        )
        x3 = native.activations[_X3_SITE]
        native_y3 = native.activations[_Y3_SITE]
        native_x4 = native.activations[_X4_SITE]
        if selected_arm == "identity":
            clamped_y3 = native_y3
        else:
            source_eligible = (
                native.sequence.query_valid_mask
                & (
                    native.sequence.logical_positions
                    >= self._plan.fit_knot_origins[0]
                )
                & (
                    native.sequence.logical_positions
                    <= self._plan.fit_knot_origins[-1]
                )
            )
            _, clamped_y3, _ = self._source_and_clamp(
                x3,
                native_y3,
                source_eligible,
            )

        def intervene_y3(original: Tensor) -> Tensor:
            if not _bitwise_equal(original, native_y3):
                raise RuntimeError(
                    "reference pass reached a non-deterministic native L3 output"
                )
            return clamped_y3

        reference = self._authenticated_forward(
            adapter,
            model_inputs,
            capture_sites=(_X3_SITE, _X4_SITE),
            interventions={_Y3_SITE: intervene_y3},
        )
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            model_inputs_sha256,
        )
        if (
            not _same_sequence(native, reference)
            or not torch.equal(reference.activations[_X3_SITE], x3)
        ):
            raise RuntimeError("native and reference executions are not aligned")
        reference_x4 = reference.activations[_X4_SITE]
        if selected_arm == "identity" and (
            not _bitwise_equal(reference_x4, native_x4)
            or not _bitwise_equal(reference.logits, native.logits)
        ):
            raise RuntimeError("identity shadow arm is not deterministic")
        provisional = self.execute_boundary_shadow(
            x3=x3,
            native_y3=native_y3,
            native_x4=native_x4,
            reference_x4=reference_x4,
            logical_positions=native.sequence.logical_positions,
            valid_mask=native.sequence.query_valid_mask,
            arm=selected_arm,
            authoritative_logits=native.logits,
            model_forward_count=2,
            model_inputs_sha256=model_inputs_sha256,
        )
        reached_candidate_x4 = False

        def intervene_x4(original: Tensor) -> Tensor:
            nonlocal reached_candidate_x4
            if not _bitwise_equal(original, reference_x4):
                raise RuntimeError(
                    "candidate suffix did not reach the authenticated "
                    "reference X4 boundary"
                )
            reached_candidate_x4 = True
            return provisional.candidate_x4

        candidate_suffix = self._authenticated_forward(
            adapter,
            model_inputs,
            capture_sites=(_X3_SITE,),
            interventions={
                _Y3_SITE: intervene_y3,
                _X4_SITE: intervene_x4,
            },
        )
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            model_inputs_sha256,
        )
        if (
            not reached_candidate_x4
            or not _same_sequence(native, candidate_suffix)
            or not torch.equal(
                candidate_suffix.activations[_X3_SITE],
                x3,
            )
        ):
            raise RuntimeError(
                "candidate suffix prefix or sequence is not deterministic"
            )
        if selected_arm == "identity" and not _bitwise_equal(
            candidate_suffix.logits,
            native.logits,
        ):
            raise RuntimeError("identity candidate suffix changed model logits")
        result = replace(
            provisional,
            candidate_logits=candidate_suffix.logits,
            accounting=replace(
                provisional.accounting,
                model_forward_count=3,
            ),
            result_artifact_sha256="",
        )
        self.validate_result_binding(result)
        self.validate_integrity()
        return result


@dataclass(frozen=True, slots=True)
class Gemma3L3L4OnePassPrefix:
    """Authenticated state prepared before the layer-4 X4 intervention."""

    source_modes: Tensor
    clamped_y3: Tensor
    predicted_target_modal_delta: Tensor
    decoded_base_x4_delta: Tensor
    logical_positions: Tensor
    valid_target_mask: Tensor
    source_eligible_mask: Tensor
    target_affected_mask: Tensor
    bridge_binding_sha256: str
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(
            self.bridge_binding_sha256,
            label="one-pass bridge binding",
        )
        if (
            not isinstance(self.source_modes, Tensor)
            or self.source_modes.ndim != 3
            or not self.source_modes.is_floating_point()
            or not isinstance(self.clamped_y3, Tensor)
            or self.clamped_y3.ndim != 3
            or not self.clamped_y3.is_floating_point()
            or not isinstance(self.predicted_target_modal_delta, Tensor)
            or self.predicted_target_modal_delta.ndim != 3
            or not self.predicted_target_modal_delta.is_floating_point()
            or not isinstance(self.decoded_base_x4_delta, Tensor)
            or self.decoded_base_x4_delta.shape != self.clamped_y3.shape
            or not self.decoded_base_x4_delta.is_floating_point()
        ):
            raise ValueError("one-pass prefix floating geometry is invalid")
        grid = self.clamped_y3.shape[:2]
        if (
            self.source_modes.shape[:2] != grid
            or self.predicted_target_modal_delta.shape[:2] != grid
            or not isinstance(self.logical_positions, Tensor)
            or self.logical_positions.shape != grid
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or not isinstance(self.valid_target_mask, Tensor)
            or self.valid_target_mask.shape != grid
            or self.valid_target_mask.dtype != torch.bool
            or not isinstance(self.source_eligible_mask, Tensor)
            or self.source_eligible_mask.shape != grid
            or self.source_eligible_mask.dtype != torch.bool
            or not isinstance(self.target_affected_mask, Tensor)
            or self.target_affected_mask.shape != grid
            or self.target_affected_mask.dtype != torch.bool
            or bool(
                (
                    self.source_eligible_mask
                    & ~self.valid_target_mask
                ).any()
            )
            or bool(
                (
                    self.target_affected_mask
                    & ~self.valid_target_mask
                ).any()
            )
        ):
            raise ValueError("one-pass prefix execution grid is invalid")
        for value, mask, label in (
            (self.source_modes, self.source_eligible_mask, "source modes"),
            (self.clamped_y3, self.source_eligible_mask, "clamped Y3"),
            (
                self.predicted_target_modal_delta,
                self.target_affected_mask,
                "target modes",
            ),
            (
                self.decoded_base_x4_delta,
                self.target_affected_mask,
                "decoded X4",
            ),
        ):
            selected = mask.to(value.device)
            if bool(selected.any()) and not bool(
                torch.isfinite(value[selected]).all()
            ):
                raise ValueError(f"{label} must be finite on active rows")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="one-pass prefix artifact",
            ) != computed:
                raise ValueError("one-pass prefix artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _computed_sha256(self) -> str:
        payload = {
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "tensor_sha256s": {
                name: _runtime_tensor_sha256(value)
                for name, value in (
                    ("source_modes", self.source_modes),
                    ("clamped_y3", self.clamped_y3),
                    (
                        "predicted_target_modal_delta",
                        self.predicted_target_modal_delta,
                    ),
                    (
                        "decoded_base_x4_delta",
                        self.decoded_base_x4_delta,
                    ),
                    ("logical_positions", self.logical_positions),
                    ("valid_target_mask", self.valid_target_mask),
                    ("source_eligible_mask", self.source_eligible_mask),
                    ("target_affected_mask", self.target_affected_mask),
                )
            },
        }
        return hashlib.sha256(
            _ONE_PASS_RESULT_DOMAIN
            + b"prefix\0"
            + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("one-pass prefix tensor payload drifted")

    def complete_h4_causal_support_mask(self) -> Tensor:
        """Return the derived full causal H4 write support.

        The support is recomputed from authenticated execution-grid tensors;
        it is not another caller-controlled or serialized prefix field.
        """

        self.validate_integrity()
        return _complete_h4_causal_support(
            self.logical_positions,
            self.valid_target_mask,
            self.source_eligible_mask,
        )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4OnePassExecution:
    """One complete prefill execution of the bridge and optional heads."""

    logits: Tensor
    reference_x4: Tensor
    candidate_x4: Tensor
    candidate_h4: Tensor
    prefix: Gemma3L3L4OnePassPrefix
    model_inputs_sha256: str
    bridge_binding_sha256: str
    x4_head_sha256: str | None
    h4_head_sha256: str | None
    model_forward_count: int = 1
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.model_inputs_sha256, label="one-pass model inputs")
        _require_sha256(
            self.bridge_binding_sha256,
            label="one-pass bridge binding",
        )
        self.prefix.validate_integrity()
        if self.prefix.bridge_binding_sha256 != self.bridge_binding_sha256:
            raise ValueError("one-pass prefix belongs to another bridge")
        for name in ("x4_head_sha256", "h4_head_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, label=name)
        if (
            not isinstance(self.logits, Tensor)
            or self.logits.ndim != 3
            or not self.logits.is_floating_point()
            or not isinstance(self.reference_x4, Tensor)
            or self.reference_x4.ndim != 3
            or not self.reference_x4.is_floating_point()
            or self.candidate_x4.shape != self.reference_x4.shape
            or self.candidate_h4.shape != self.reference_x4.shape
            or not self.candidate_x4.is_floating_point()
            or not self.candidate_h4.is_floating_point()
            or self.logits.shape[:2] != self.reference_x4.shape[:2]
            or self.model_forward_count != 1
        ):
            raise ValueError("one-pass execution geometry is invalid")
        valid = self.prefix.valid_target_mask
        for value, label in (
            (self.logits, "logits"),
            (self.reference_x4, "reference X4"),
            (self.candidate_x4, "candidate X4"),
            (self.candidate_h4, "candidate H4"),
        ):
            selected = valid.to(value.device)
            if bool(selected.any()) and not bool(
                torch.isfinite(value[selected]).all()
            ):
                raise ValueError(f"one-pass {label} must be finite")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="one-pass execution artifact",
            ) != computed:
                raise ValueError("one-pass execution artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _computed_sha256(self) -> str:
        return hashlib.sha256(
            _ONE_PASS_RESULT_DOMAIN
            + b"execution\0"
            + _canonical_json_bytes(
                {
                    "model_inputs_sha256": self.model_inputs_sha256,
                    "bridge_binding_sha256": self.bridge_binding_sha256,
                    "prefix_sha256": self.prefix.artifact_sha256,
                    "x4_head_sha256": self.x4_head_sha256,
                    "h4_head_sha256": self.h4_head_sha256,
                    "model_forward_count": self.model_forward_count,
                    "tensor_sha256s": {
                        name: _runtime_tensor_sha256(value)
                        for name, value in (
                            ("logits", self.logits),
                            ("reference_x4", self.reference_x4),
                            ("candidate_x4", self.candidate_x4),
                            ("candidate_h4", self.candidate_h4),
                        )
                    },
                }
            )
        ).hexdigest()

    def validate_integrity(self) -> None:
        self.prefix.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("one-pass execution tensor payload drifted")


@dataclass(frozen=True, slots=True)
class Gemma3L3L4TokenNLLVJP:
    """Authenticated exact per-token NLL gradients at the H4 boundary."""

    execution: Gemma3L3L4OnePassExecution
    supervised_indices: Tensor
    token_losses: Tensor
    h4_gradients: Tensor
    targets_sha256: str
    ignore_index: int
    vjp_chunk_size: int
    backward_call_count: int
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.execution, Gemma3L3L4OnePassExecution):
            raise TypeError("token NLL VJP execution has the wrong type")
        self.execution.validate_integrity()
        _require_sha256(self.targets_sha256, label="token NLL VJP targets")
        if type(self.ignore_index) is not int:
            raise ValueError("token NLL VJP ignore_index must be an integer")
        if type(self.vjp_chunk_size) is not int or self.vjp_chunk_size <= 0:
            raise ValueError(
                "token NLL VJP chunk size must be a positive integer"
            )
        if (
            not isinstance(self.supervised_indices, Tensor)
            or self.supervised_indices.ndim != 2
            or self.supervised_indices.shape[1] != 2
            or self.supervised_indices.dtype != torch.int64
            or self.supervised_indices.requires_grad
            or not self.supervised_indices.is_contiguous()
        ):
            raise ValueError(
                "token NLL VJP indices must be contiguous int64 [N, 2]"
            )
        token_count = int(self.supervised_indices.shape[0])
        if (
            token_count <= 0
            or not isinstance(self.token_losses, Tensor)
            or self.token_losses.shape != (token_count,)
            or not self.token_losses.is_floating_point()
            or self.token_losses.requires_grad
            or not self.token_losses.is_contiguous()
            or not bool(torch.isfinite(self.token_losses).all())
            or not isinstance(self.h4_gradients, Tensor)
            or self.h4_gradients.shape
            != (token_count, *self.execution.candidate_h4.shape)
            or not self.h4_gradients.is_floating_point()
            or self.h4_gradients.requires_grad
            or not self.h4_gradients.is_contiguous()
            or not bool(torch.isfinite(self.h4_gradients).all())
        ):
            raise ValueError("token NLL VJP tensor geometry is invalid")
        if (
            self.token_losses.device != self.execution.logits.device
            or self.h4_gradients.device
            != self.execution.candidate_h4.device
        ):
            raise ValueError("token NLL VJP tensors use the wrong device")
        indices = self.supervised_indices.to(
            device=self.execution.prefix.valid_target_mask.device
        )
        batch_count, sequence_length = (
            self.execution.prefix.valid_target_mask.shape
        )
        if (
            bool((indices[:, 0] < 0).any())
            or bool((indices[:, 0] >= batch_count).any())
            or bool((indices[:, 1] < 0).any())
            or bool((indices[:, 1] >= sequence_length).any())
            or not bool(
                self.execution.prefix.valid_target_mask[
                    indices[:, 0], indices[:, 1]
                ].all()
            )
        ):
            raise ValueError("token NLL VJP indices escape the valid grid")
        flattened = indices[:, 0] * sequence_length + indices[:, 1]
        if token_count > 1 and not bool(
            (flattened[1:] > flattened[:-1]).all()
        ):
            raise ValueError("token NLL VJP indices are not canonical")
        expected_backward_calls = (
            token_count + self.vjp_chunk_size - 1
        ) // self.vjp_chunk_size
        if (
            type(self.backward_call_count) is not int
            or self.backward_call_count != expected_backward_calls
        ):
            raise ValueError("token NLL VJP backward count differs")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="token NLL VJP artifact",
            ) != computed:
                raise ValueError("token NLL VJP artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def token_count(self) -> int:
        return int(self.token_losses.shape[0])

    @property
    def model_forward_count(self) -> int:
        return self.execution.model_forward_count

    def _computed_sha256(self) -> str:
        payload = {
            "execution_sha256": self.execution.artifact_sha256,
            "targets_sha256": self.targets_sha256,
            "ignore_index": self.ignore_index,
            "vjp_chunk_size": self.vjp_chunk_size,
            "backward_call_count": self.backward_call_count,
            "tensor_sha256s": {
                name: _runtime_tensor_sha256(value)
                for name, value in (
                    ("supervised_indices", self.supervised_indices),
                    ("token_losses", self.token_losses),
                    ("h4_gradients", self.h4_gradients),
                )
            },
        }
        return hashlib.sha256(
            _ONE_PASS_RESULT_DOMAIN
            + b"token-nll-vjp\0"
            + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self.execution.validate_integrity()
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError("token NLL VJP tensor payload drifted")


@dataclass(frozen=True, slots=True)
class Gemma3L3L4TokenTeacherKLVJP:
    """Authenticated exact per-token teacher-KL gradients at H4."""

    execution: Gemma3L3L4OnePassExecution
    supervised_indices: Tensor
    token_kl_divergences: Tensor
    h4_gradients: Tensor
    teacher_logits_sha256: str
    teacher_logits_shape: tuple[int, int, int]
    h4_head_sha256: str | None
    vjp_chunk_size: int
    backward_call_count: int
    artifact_sha256: str = ""
    objective_dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution, Gemma3L3L4OnePassExecution):
            raise TypeError("token teacher-KL VJP execution has the wrong type")
        self.execution.validate_integrity()
        _require_sha256(
            self.teacher_logits_sha256,
            label="token teacher-KL VJP teacher logits",
        )
        if (
            not isinstance(self.teacher_logits_shape, tuple)
            or len(self.teacher_logits_shape) != 3
            or any(
                type(width) is not int or width <= 0
                for width in self.teacher_logits_shape
            )
            or self.teacher_logits_shape
            != tuple(int(width) for width in self.execution.logits.shape)
        ):
            raise ValueError(
                "token teacher-KL VJP teacher grid differs from execution"
            )
        if self.h4_head_sha256 is not None:
            _require_sha256(
                self.h4_head_sha256,
                label="token teacher-KL VJP H4 head",
            )
        if self.h4_head_sha256 != self.execution.h4_head_sha256:
            raise ValueError(
                "token teacher-KL VJP H4 head binding differs"
            )
        if type(self.vjp_chunk_size) is not int or self.vjp_chunk_size <= 0:
            raise ValueError(
                "token teacher-KL VJP chunk size must be a positive integer"
            )
        if self.objective_dtype not in (None, str(torch.float64)):
            raise ValueError(
                "token teacher-KL VJP objective dtype must be None or "
                "torch.float64"
            )
        if (
            not isinstance(self.supervised_indices, Tensor)
            or self.supervised_indices.ndim != 2
            or self.supervised_indices.shape[1] != 2
            or self.supervised_indices.dtype != torch.int64
            or self.supervised_indices.requires_grad
            or not self.supervised_indices.is_contiguous()
        ):
            raise ValueError(
                "token teacher-KL VJP indices must be contiguous int64 "
                "[N, 2]"
            )
        token_count = int(self.supervised_indices.shape[0])
        if (
            token_count <= 0
            or not isinstance(self.token_kl_divergences, Tensor)
            or self.token_kl_divergences.shape != (token_count,)
            or not self.token_kl_divergences.is_floating_point()
            or self.token_kl_divergences.requires_grad
            or not self.token_kl_divergences.is_contiguous()
            or not bool(torch.isfinite(self.token_kl_divergences).all())
            or not isinstance(self.h4_gradients, Tensor)
            or self.h4_gradients.shape
            != (token_count, *self.execution.candidate_h4.shape)
            or not self.h4_gradients.is_floating_point()
            or self.h4_gradients.requires_grad
            or not self.h4_gradients.is_contiguous()
            or not bool(torch.isfinite(self.h4_gradients).all())
        ):
            raise ValueError("token teacher-KL VJP tensor geometry is invalid")
        if (
            self.token_kl_divergences.device
            != self.execution.logits.device
            or self.h4_gradients.device
            != self.execution.candidate_h4.device
        ):
            raise ValueError("token teacher-KL VJP tensors use the wrong device")
        if (
            self.objective_dtype == str(torch.float64)
            and self.token_kl_divergences.dtype != torch.float64
        ):
            raise ValueError(
                "float64 token teacher-KL VJP objective produced another dtype"
            )
        indices = self.supervised_indices.to(
            device=self.execution.prefix.valid_target_mask.device
        )
        batch_count, sequence_length = (
            self.execution.prefix.valid_target_mask.shape
        )
        if (
            bool((indices[:, 0] < 0).any())
            or bool((indices[:, 0] >= batch_count).any())
            or bool((indices[:, 1] < 0).any())
            or bool((indices[:, 1] >= sequence_length).any())
            or not bool(
                self.execution.prefix.valid_target_mask[
                    indices[:, 0], indices[:, 1]
                ].all()
            )
        ):
            raise ValueError(
                "token teacher-KL VJP indices escape the valid grid"
            )
        flattened = indices[:, 0] * sequence_length + indices[:, 1]
        if token_count > 1 and not bool(
            (flattened[1:] > flattened[:-1]).all()
        ):
            raise ValueError(
                "token teacher-KL VJP indices are not canonical"
            )
        expected_backward_calls = (
            token_count + self.vjp_chunk_size - 1
        ) // self.vjp_chunk_size
        if (
            type(self.backward_call_count) is not int
            or self.backward_call_count != expected_backward_calls
        ):
            raise ValueError("token teacher-KL VJP backward count differs")
        computed = self._computed_sha256()
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="token teacher-KL VJP artifact",
            ) != computed:
                raise ValueError(
                    "token teacher-KL VJP artifact hash mismatch"
                )
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def token_count(self) -> int:
        return int(self.token_kl_divergences.shape[0])

    @property
    def model_forward_count(self) -> int:
        return self.execution.model_forward_count

    def _computed_sha256(self) -> str:
        payload = {
            "execution_sha256": self.execution.artifact_sha256,
            "teacher_logits_sha256": self.teacher_logits_sha256,
            "teacher_logits_shape": self.teacher_logits_shape,
            "h4_head_sha256": self.h4_head_sha256,
            "vjp_chunk_size": self.vjp_chunk_size,
            "backward_call_count": self.backward_call_count,
            "tensor_sha256s": {
                name: _runtime_tensor_sha256(value)
                for name, value in (
                    ("supervised_indices", self.supervised_indices),
                    ("token_kl_divergences", self.token_kl_divergences),
                    ("h4_gradients", self.h4_gradients),
                )
            },
        }
        if self.objective_dtype is not None:
            payload["objective_dtype"] = self.objective_dtype
        return hashlib.sha256(
            _ONE_PASS_RESULT_DOMAIN
            + b"token-teacher-kl-vjp\0"
            + _canonical_json_bytes(payload)
        ).hexdigest()

    def validate_integrity(self) -> None:
        self.execution.validate_integrity()
        if self.h4_head_sha256 != self.execution.h4_head_sha256:
            raise RuntimeError(
                "token teacher-KL VJP H4 head binding drifted"
            )
        if self._computed_sha256() != self.artifact_sha256:
            raise RuntimeError(
                "token teacher-KL VJP tensor payload drifted"
            )


class Gemma3L3L4OnePassBridge:
    """Serving-safe clone of the locked rank-64 all-on bridge.

    Unlike the source-authoritative shadow runtime, this class never consumes
    native X4 from a separate pass.  Rows outside graph support remain the
    clamped reference produced by the same forward.
    """

    def __init__(
        self,
        source: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    ) -> None:
        if not isinstance(
            source,
            Gemma3L3L4GraphOrganizedSVDShadowRuntime,
        ):
            raise TypeError("source must be the locked L3/L4 shadow runtime")
        source.validate_integrity()
        self._parent_runtime_binding_sha256 = source.runtime_binding_sha256
        self._candidate_sha256 = source.candidate_artifact_sha256
        self._live_model_sha256 = source.live_model_sha256
        self._adapter_execution_sha256 = source.adapter_execution_sha256
        self._device = source._device
        self._residual_width = source.residual_width
        self._source_modes = source.source_modes
        self._target_modes = source.target_modes
        self._fit_knot_origins = source._plan.fit_knot_origins
        self._lag_count = source._plan.lag_count
        self._source_rank = source._plan.source_rank
        self._x3_mean = source._x3_mean.detach().clone().contiguous()
        self._r3 = source._r3.detach().clone().contiguous()
        self._p3 = source._p3.detach().clone().contiguous()
        self._target_decoder = (
            source._target_decoder.detach().clone().contiguous()
        )
        self._graph = source._plan.prepare(
            device=self._device,
            dtype=torch.float64,
        )
        self._binding_sha256 = self._computed_binding_sha256()
        self._expected_header = self._header()
        self._expected_tensor_sha256s = {
            name: _runtime_tensor_sha256(value)
            for name, value in self._internal_tensors().items()
        }
        self.validate_integrity()

    @property
    def bridge_binding_sha256(self) -> str:
        return self._binding_sha256

    @property
    def parent_runtime_binding_sha256(self) -> str:
        return self._parent_runtime_binding_sha256

    @property
    def source_modes(self) -> int:
        return self._source_modes

    @property
    def target_modes(self) -> int:
        return self._target_modes

    @property
    def source_rank(self) -> int:
        return self._source_rank

    @property
    def residual_width(self) -> int:
        return self._residual_width

    @property
    def lag_count(self) -> int:
        return self._lag_count

    @property
    def fit_knot_origins(self) -> tuple[int, ...]:
        return self._fit_knot_origins

    @property
    def prepared_float_scalar_count(self) -> int:
        return sum(
            int(value.numel())
            for value in self._internal_tensors().values()
            if value.is_floating_point()
        )

    @property
    def prepared_integer_value_count(self) -> int:
        return sum(
            int(value.numel())
            for value in self._internal_tensors().values()
            if not value.is_floating_point()
        )

    @property
    def prepared_runtime_parameter_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in self._internal_tensors().values()
        )

    @property
    def logical_macs_per_token_upper_bound(self) -> int:
        return (
            2 * self.residual_width * self.source_modes
            + self.source_modes * self.source_rank
            + self.lag_count * self.source_rank * self.target_modes
            + self.target_modes * self.residual_width
        )

    def _internal_tensors(self) -> dict[str, Tensor]:
        graph = dict(self._graph.named_buffers(recurse=True))
        if set(graph) != {
            "source_scales",
            "source_basis",
            "knot_cores",
            "core_operator_norm_bounds",
            "pack_offsets",
        }:
            raise RuntimeError("one-pass prepared graph buffer set drifted")
        return {
            "basis.x3_mean": self._x3_mean,
            "basis.R3": self._r3,
            "basis.P3": self._p3,
            "decoder.target_dual": self._target_decoder,
            **{f"graph.{name}": value for name, value in graph.items()},
        }

    def _header(self) -> tuple[object, ...]:
        return (
            self._parent_runtime_binding_sha256,
            self._candidate_sha256,
            self._live_model_sha256,
            self._adapter_execution_sha256,
            str(self._device),
            self._residual_width,
            self._source_modes,
            self._source_rank,
            self._target_modes,
            self._fit_knot_origins,
            self._lag_count,
            self._graph.plan_sha256,
            self._graph.fit_knot_origins,
            self._graph.source_modes,
            self._graph.source_rank,
            self._graph.target_modes,
            self._graph.pack_count,
            self._graph.lag_count,
            self._binding_sha256,
            "one_model_forward",
            "reference_x4_outside_target_support",
            "float64_reference_plus_decoded_delta_then_single_live_dtype_cast",
        )

    def _computed_binding_sha256(self) -> str:
        return hashlib.sha256(
            _ONE_PASS_BRIDGE_DOMAIN
            + _canonical_json_bytes(
                {
                    "format_version": 2,
                    "parent_runtime_binding_sha256": (
                        self._parent_runtime_binding_sha256
                    ),
                    "candidate_artifact_sha256": self._candidate_sha256,
                    "live_model_sha256": self._live_model_sha256,
                    "adapter_execution_sha256": (
                        self._adapter_execution_sha256
                    ),
                    "analysis_device": str(self._device),
                    "residual_width": self._residual_width,
                    "source_modes": self._source_modes,
                    "source_rank": self._source_rank,
                    "target_modes": self._target_modes,
                    "fit_knot_origins": self._fit_knot_origins,
                    "lag_count": self._lag_count,
                    "model_forward_count": 1,
                    "fallback_policy": (
                        "same_pass_reference_x4_outside_target_support"
                    ),
                    "base_x4_accumulation_policy": (
                        "float64_reference_plus_decoded_delta_then_"
                        "single_live_dtype_cast"
                    ),
                }
            )
        ).hexdigest()

    def validate_integrity(self) -> None:
        if not isinstance(self._graph, PreparedGraphOrganizedSVD):
            raise RuntimeError("one-pass prepared graph type drifted")
        if dict(self._graph.named_parameters(recurse=True)):
            raise RuntimeError("one-pass prepared graph gained parameters")
        if (
            self._computed_binding_sha256() != self._binding_sha256
            or self._header() != self._expected_header
        ):
            raise RuntimeError("one-pass bridge header drifted")
        tensors = self._internal_tensors()
        if set(tensors) != set(self._expected_tensor_sha256s):
            raise RuntimeError("one-pass bridge tensor set drifted")
        for name, value in tensors.items():
            if (
                not value.is_contiguous()
                or value.device != self._device
                or (
                    value.is_floating_point()
                    and not bool(torch.isfinite(value).all())
                )
                or _runtime_tensor_sha256(value)
                != self._expected_tensor_sha256s[name]
            ):
                raise RuntimeError(f"one-pass bridge tensor {name} drifted")

    def _authenticate_adapter(
        self,
        adapter: Gemma3CausalLMAdapter,
    ) -> None:
        if not isinstance(adapter, Gemma3CausalLMAdapter):
            raise TypeError("adapter must be a Gemma3CausalLMAdapter")
        if adapter.module.training or any(
            module.training for module in adapter.module.modules()
        ):
            raise ValueError("source Gemma must remain completely in eval mode")
        if (
            adapter.model_fingerprint() != self._live_model_sha256
            or adapter.execution_fingerprint()
            != self._adapter_execution_sha256
        ):
            raise ValueError(
                "live Gemma or adapter execution differs from the bridge"
            )
        sites = {site.id: site for site in adapter.activation_sites}
        required = (_X3_SITE, _Y3_SITE, _X4_SITE, _H4_SITE)
        if any(
            name not in sites or not sites[name].intervenable
            for name in required
        ):
            raise ValueError("Gemma one-pass intervention ABI drifted")

    def _masks(
        self,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source = (
            valid_mask
            & (logical_positions >= self.fit_knot_origins[0])
            & (logical_positions <= self.fit_knot_origins[-1])
        )
        target = torch.zeros_like(valid_mask)
        for batch in range(valid_mask.shape[0]):
            source_positions = logical_positions[batch][source[batch]]
            if source_positions.numel() == 0:
                continue
            indices = torch.nonzero(
                valid_mask[batch],
                as_tuple=False,
            ).flatten()
            target_positions = logical_positions[batch][indices]
            differences = (
                target_positions.unsqueeze(1)
                - source_positions.unsqueeze(0)
            )
            target[batch, indices] = (
                (differences >= 0) & (differences < self.lag_count)
            ).any(dim=1)
        return source, target

    def _prepare_prefix(
        self,
        *,
        x3: Tensor,
        native_y3: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
    ) -> Gemma3L3L4OnePassPrefix:
        if (
            x3.ndim != 3
            or x3.shape != native_y3.shape
            or x3.shape[-1] != self.residual_width
            or not x3.is_floating_point()
            or native_y3.dtype != x3.dtype
            or native_y3.device != x3.device
            or logical_positions.shape != x3.shape[:2]
            or logical_positions.device != x3.device
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or valid_mask.shape != x3.shape[:2]
            or valid_mask.device != x3.device
            or valid_mask.dtype != torch.bool
        ):
            raise ValueError("one-pass X3/Y3 execution geometry is invalid")
        for batch in range(x3.shape[0]):
            positions = logical_positions[batch][valid_mask[batch]]
            if (
                positions.numel() == 0
                or bool((positions < 0).any())
                or (
                    positions.numel() > 1
                    and not bool(torch.all(positions[1:] > positions[:-1]))
                )
            ):
                raise ValueError(
                    "valid logical positions must be nonnegative and increasing"
                )
        source, target = self._masks(logical_positions, valid_mask)
        modes = torch.zeros(
            (*x3.shape[:2], self.source_modes),
            device=self._device,
            dtype=torch.float64,
        )
        decoded_l3 = torch.zeros(
            x3.shape,
            device=self._device,
            dtype=torch.float64,
        )
        source_analysis = source.to(self._device)
        if bool(source_analysis.any()):
            selected = x3[source].to(self._device, torch.float64)
            selected_modes = (selected - self._x3_mean) @ self._r3.T
            modes[source_analysis] = selected_modes
            decoded_l3[source_analysis] = selected_modes @ self._p3.T
        clamped = native_y3.clone()
        if bool(source.any()):
            clamped[source] = (
                native_y3[source].to(self._device, torch.float64)
                - decoded_l3[source_analysis]
            ).to(device=native_y3.device, dtype=native_y3.dtype)
        prediction = torch.zeros(
            (*x3.shape[:2], self.target_modes),
            device=self._device,
            dtype=torch.float64,
        )
        if bool(source.any()):
            prediction = self._graph(
                modes,
                logical_positions=logical_positions.to(self._device),
                valid_mask=valid_mask.to(self._device),
                source_mask=source.to(self._device),
            )
        decoded_x4 = torch.zeros(
            x3.shape,
            device=self._device,
            dtype=torch.float64,
        )
        target_analysis = target.to(self._device)
        if bool(target_analysis.any()):
            decoded_x4[target_analysis] = (
                prediction[target_analysis] @ self._target_decoder
            )
        return Gemma3L3L4OnePassPrefix(
            source_modes=modes,
            clamped_y3=clamped,
            predicted_target_modal_delta=prediction,
            decoded_base_x4_delta=decoded_x4,
            logical_positions=logical_positions.detach().clone(),
            valid_target_mask=valid_mask.detach().clone(),
            source_eligible_mask=source.detach().clone(),
            target_affected_mask=target.detach().clone(),
            bridge_binding_sha256=self.bridge_binding_sha256,
        )

    @staticmethod
    def _head_correction(
        provider: Gemma3L3L4CorrectionProvider | None,
        *,
        prefix: Gemma3L3L4OnePassPrefix,
        expected_site: str,
        label: str,
        reference: Tensor,
    ) -> tuple[Tensor, str | None, Tensor, CorrectionWriteScope]:
        prefix.validate_integrity()
        if provider is None:
            return (
                torch.zeros_like(reference),
                None,
                prefix.target_affected_mask.to(reference.device),
                "graph_target_affected_mask",
            )
        if not isinstance(provider, Gemma3L3L4CorrectionProvider):
            raise TypeError(
                f"{label} must implement the authenticated correction "
                "provider interface"
            )
        provider.validate_integrity()
        provider_site = provider.site
        if provider_site != expected_site:
            raise ValueError(f"{label} is bound to the wrong activation site")
        write_scope = _require_correction_write_scope(provider.write_scope)
        if write_scope == "complete_h4_causal_support":
            if expected_site != _H4_SITE:
                raise ValueError(
                    "complete-H4 causal correction scope is valid only "
                    "for the H4 head"
                )
            active = prefix.complete_h4_causal_support_mask()
        else:
            active = prefix.target_affected_mask
        head_sha256 = provider.artifact_sha256
        _require_sha256(head_sha256, label=f"{label} artifact")
        reference_sha256 = _runtime_tensor_sha256(reference)
        correction = provider.correction(prefix, reference)
        provider.validate_integrity()
        prefix.validate_integrity()
        if (
            provider.site != provider_site
            or provider.artifact_sha256 != head_sha256
            or provider.write_scope != write_scope
        ):
            raise RuntimeError(f"{label} provider identity drifted")
        if _runtime_tensor_sha256(reference) != reference_sha256:
            raise RuntimeError(f"{label} mutated its realized activation")
        if (
            not isinstance(correction, Tensor)
            or correction.shape != reference.shape
            or not correction.is_floating_point()
        ):
            raise ValueError(
                f"{label} correction must match the residual stream"
            )
        correction_active = active.to(correction.device)
        if bool(correction_active.any()) and not bool(
            torch.isfinite(correction[correction_active]).all()
        ):
            raise ValueError(f"{label} correction is nonfinite")
        inactive = ~correction_active
        if bool(inactive.any()) and not bool(
            (correction[inactive] == 0).all()
        ):
            raise ValueError(f"{label} correction must be zero off support")
        normalized_correction = correction.to(
            device=reference.device,
            dtype=(
                torch.float64
                if write_scope == "complete_h4_causal_support"
                else reference.dtype
            ),
        )
        return (
            normalized_correction,
            head_sha256,
            active.to(reference.device),
            write_scope,
        )

    def _execute(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        x4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_vjp_objective: Callable[[AdapterRun], Tensor] | None = None,
        h4_vjp_chunk_size: int | None = None,
    ) -> tuple[Gemma3L3L4OnePassExecution, Tensor | None]:
        """Execute one bridge pass, optionally retaining H4 VJP rows."""

        self.validate_integrity()
        self._authenticate_adapter(adapter)
        if h4_vjp_chunk_size is not None and (
            h4_vjp_objective is None
            or type(h4_vjp_chunk_size) is not int
            or h4_vjp_chunk_size <= 0
        ):
            raise ValueError(
                "batched H4 VJP requires a positive integer chunk size"
            )
        model_inputs_sha256 = (
            gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        )
        sequence = adapter.prepare_sequence(model_inputs)
        x3: Tensor | None = None
        prefix: Gemma3L3L4OnePassPrefix | None = None
        reference_x4: Tensor | None = None
        candidate_x4: Tensor | None = None
        candidate_h4: Tensor | None = None
        x4_head_sha256: str | None = None
        h4_head_sha256: str | None = None

        def at_x3(original: Tensor) -> Tensor:
            nonlocal x3
            if x3 is not None:
                raise RuntimeError("one-pass X3 intervention repeated")
            x3 = original
            return original

        def at_y3(original: Tensor) -> Tensor:
            nonlocal prefix
            if x3 is None or prefix is not None:
                raise RuntimeError("one-pass Y3 intervention order drifted")
            prefix = self._prepare_prefix(
                x3=x3,
                native_y3=original,
                logical_positions=sequence.logical_positions,
                valid_mask=sequence.query_valid_mask,
            )
            return prefix.clamped_y3

        def at_x4(original: Tensor) -> Tensor:
            nonlocal reference_x4, candidate_x4, x4_head_sha256
            if prefix is None or reference_x4 is not None:
                raise RuntimeError("one-pass X4 intervention order drifted")
            reference_x4 = original
            candidate_x4 = original.clone()
            active = prefix.target_affected_mask
            if bool(active.any()):
                active_on_analysis = active.to(
                    prefix.decoded_base_x4_delta.device
                )
                base = prefix.decoded_base_x4_delta[
                    active_on_analysis
                ]
                # Match the authenticated shadow arithmetic exactly: add the
                # float64 decoded delta to a float64 reference carrier, then
                # cast once into the live residual dtype.  Casting the delta
                # first introduces a distinct float32 rounding path.
                reference = original[active].to(
                    device=base.device,
                    dtype=torch.float64,
                )
                candidate_x4[active] = (reference + base).to(
                    device=original.device,
                    dtype=original.dtype,
                )
            (
                correction,
                x4_head_sha256,
                write_mask,
                write_scope,
            ) = self._head_correction(
                x4_head,
                prefix=prefix,
                expected_site=_X4_SITE,
                label="X4 head",
                reference=original,
            )
            if write_scope != "graph_target_affected_mask":
                raise RuntimeError("X4 correction escaped graph write scope")
            if bool(write_mask.any()):
                candidate_x4[write_mask] += correction[write_mask]
            if h4_vjp_objective is not None:
                candidate_x4 = candidate_x4.detach()
            return candidate_x4

        def at_h4(original: Tensor) -> Tensor:
            nonlocal candidate_h4, h4_head_sha256
            if prefix is None or candidate_x4 is None or candidate_h4 is not None:
                raise RuntimeError("one-pass H4 intervention order drifted")
            candidate_h4 = original.clone()
            (
                correction,
                h4_head_sha256,
                write_mask,
                write_scope,
            ) = self._head_correction(
                h4_head,
                prefix=prefix,
                expected_site=_H4_SITE,
                label="H4 head",
                reference=original,
            )
            if bool(write_mask.any()):
                if write_scope == "complete_h4_causal_support":
                    # Preserve the exact residual identity: independently
                    # rounding the delta before addition can select another
                    # live-dtype value at a midpoint.
                    realized_h4 = original[write_mask].to(
                        device=correction.device,
                        dtype=torch.float64,
                    )
                    candidate_h4[write_mask] = (
                        realized_h4 + correction[write_mask]
                    ).to(
                        device=original.device,
                        dtype=original.dtype,
                    )
                else:
                    candidate_h4[write_mask] += correction[write_mask]
            if h4_vjp_objective is not None:
                candidate_h4 = (
                    candidate_h4.detach().requires_grad_(True)
                )
            return candidate_h4

        h4_gradient: Tensor | None = None
        try:
            context = (
                torch.enable_grad()
                if h4_vjp_objective is not None
                else torch.no_grad()
            )
            with context:
                run = adapter.forward(
                    model_inputs,
                    capture_sites=(),
                    interventions={
                        _X3_SITE: at_x3,
                        _Y3_SITE: at_y3,
                        _X4_SITE: at_x4,
                        _H4_SITE: at_h4,
                    },
                )
                if h4_vjp_objective is not None:
                    if candidate_h4 is None:
                        raise RuntimeError(
                            "H4 VJP pass did not reach the H4 boundary"
                        )
                    loss = h4_vjp_objective(run)
                    if h4_vjp_chunk_size is None:
                        if (
                            not isinstance(loss, Tensor)
                            or loss.ndim != 0
                            or not loss.is_floating_point()
                            or not bool(torch.isfinite(loss))
                        ):
                            raise ValueError(
                                "H4 VJP objective must return one finite "
                                "scalar"
                            )
                        (h4_gradient,) = torch.autograd.grad(
                            loss,
                            (candidate_h4,),
                            retain_graph=False,
                            create_graph=False,
                        )
                    else:
                        if (
                            not isinstance(loss, Tensor)
                            or loss.ndim != 1
                            or loss.numel() <= 0
                            or not loss.is_floating_point()
                            or not bool(torch.isfinite(loss).all())
                        ):
                            raise ValueError(
                                "batched H4 VJP objective must return a "
                                "nonempty finite vector"
                            )
                        token_count = int(loss.shape[0])
                        chunks: list[Tensor] = []
                        for start in range(
                            0, token_count, h4_vjp_chunk_size
                        ):
                            stop = min(
                                start + h4_vjp_chunk_size,
                                token_count,
                            )
                            chunk_count = stop - start
                            cotangents = torch.zeros(
                                (chunk_count, token_count),
                                device=loss.device,
                                dtype=loss.dtype,
                            )
                            chunk_rows = torch.arange(
                                chunk_count,
                                device=loss.device,
                            )
                            loss_rows = torch.arange(
                                start,
                                stop,
                                device=loss.device,
                            )
                            cotangents[chunk_rows, loss_rows] = 1
                            (chunk,) = torch.autograd.grad(
                                loss,
                                (candidate_h4,),
                                grad_outputs=cotangents,
                                retain_graph=stop < token_count,
                                create_graph=False,
                                is_grads_batched=True,
                            )
                            if (
                                chunk.shape
                                != (
                                    chunk_count,
                                    *candidate_h4.shape,
                                )
                                or not chunk.is_floating_point()
                                or not bool(torch.isfinite(chunk).all())
                            ):
                                raise RuntimeError(
                                    "batched H4 VJP chunk geometry differs"
                                )
                            chunks.append(chunk.detach().contiguous())
                        h4_gradient = torch.cat(chunks, dim=0).contiguous()
        finally:
            self.validate_integrity()
            self._authenticate_adapter(adapter)
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            model_inputs_sha256,
        )
        if (
            prefix is None
            or reference_x4 is None
            or candidate_x4 is None
            or candidate_h4 is None
            or x3 is None
        ):
            raise RuntimeError("one-pass bridge did not reach every boundary")
        prefix.validate_integrity()
        if (
            not torch.equal(
                run.sequence.logical_positions,
                sequence.logical_positions,
            )
            or not torch.equal(
                run.sequence.query_valid_mask,
                sequence.query_valid_mask,
            )
        ):
            raise RuntimeError("prepared and executed sequence grids differ")
        inactive = ~prefix.target_affected_mask
        if not _bitwise_equal(
            candidate_x4[inactive],
            reference_x4[inactive],
        ):
            raise RuntimeError(
                "one-pass bridge modified reference X4 outside support"
            )
        detached_prefix = Gemma3L3L4OnePassPrefix(
            source_modes=prefix.source_modes.detach(),
            clamped_y3=prefix.clamped_y3.detach(),
            predicted_target_modal_delta=(
                prefix.predicted_target_modal_delta.detach()
            ),
            decoded_base_x4_delta=(
                prefix.decoded_base_x4_delta.detach()
            ),
            logical_positions=prefix.logical_positions.detach(),
            valid_target_mask=prefix.valid_target_mask.detach(),
            source_eligible_mask=prefix.source_eligible_mask.detach(),
            target_affected_mask=prefix.target_affected_mask.detach(),
            bridge_binding_sha256=prefix.bridge_binding_sha256,
            artifact_sha256=prefix.artifact_sha256,
        )
        execution = Gemma3L3L4OnePassExecution(
            logits=run.logits.detach(),
            reference_x4=reference_x4.detach(),
            candidate_x4=candidate_x4.detach(),
            candidate_h4=candidate_h4.detach(),
            prefix=detached_prefix,
            model_inputs_sha256=model_inputs_sha256,
            bridge_binding_sha256=self.bridge_binding_sha256,
            x4_head_sha256=x4_head_sha256,
            h4_head_sha256=h4_head_sha256,
        )
        return (
            execution,
            None
            if h4_gradient is None
            else h4_gradient.detach().contiguous(),
        )

    def execute(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        x4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_head: Gemma3L3L4CorrectionProvider | None = None,
    ) -> Gemma3L3L4OnePassExecution:
        """Execute the base bridge and optional residual heads in one prefill."""

        execution, gradient = self._execute(
            adapter,
            model_inputs,
            x4_head=x4_head,
            h4_head=h4_head,
        )
        if gradient is not None:
            raise RuntimeError("serving execution unexpectedly retained H4 VJP")
        return execution

    def execute_h4_vjp(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        objective: Callable[[AdapterRun], Tensor],
        x4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_head: Gemma3L3L4CorrectionProvider | None = None,
    ) -> tuple[Gemma3L3L4OnePassExecution, Tensor]:
        """Execute an authenticated fit-only pass and return dLoss/dH4."""

        if not callable(objective):
            raise TypeError("objective must be callable")
        execution, gradient = self._execute(
            adapter,
            model_inputs,
            x4_head=x4_head,
            h4_head=h4_head,
            h4_vjp_objective=objective,
        )
        if gradient is None:
            raise RuntimeError("H4 VJP execution omitted its gradient")
        return execution, gradient

    def execute_h4_token_nll_vjps(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        targets: Tensor,
        ignore_index: int = -100,
        vjp_chunk_size: int = 8,
        x4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_head: Gemma3L3L4CorrectionProvider | None = None,
    ) -> Gemma3L3L4TokenNLLVJP:
        """Return exact per-supervised-token NLL gradients from one forward.

        The retained autograd graph is traversed in bounded batched-VJP
        chunks.  Token rows use canonical batch-major, position-major order.
        """

        if not isinstance(targets, Tensor) or targets.ndim != 2:
            raise ValueError("token NLL VJP targets must have shape [B, S]")
        input_ids = model_inputs.get("input_ids")
        if (
            not isinstance(input_ids, Tensor)
            or targets.shape != input_ids.shape
            or targets.dtype != torch.int64
        ):
            raise ValueError(
                "token NLL VJP targets must be int64 on the input grid"
            )
        if type(ignore_index) is not int:
            raise ValueError("token NLL VJP ignore_index must be an integer")
        if type(vjp_chunk_size) is not int or vjp_chunk_size <= 0:
            raise ValueError(
                "token NLL VJP chunk size must be a positive integer"
            )
        target_snapshot = targets.detach().clone().contiguous()
        targets_sha256 = _runtime_tensor_sha256(target_snapshot)
        captured_indices: Tensor | None = None
        captured_losses: Tensor | None = None

        def token_objective(run: AdapterRun) -> Tensor:
            nonlocal captured_indices, captured_losses
            if (
                run.logits.ndim != 3
                or run.logits.shape[:2] != target_snapshot.shape
            ):
                raise ValueError("token NLL VJP logits grid differs")
            selected_targets = target_snapshot.to(run.logits.device)
            valid = run.sequence.query_valid_mask.to(run.logits.device)
            supervised = selected_targets != ignore_index
            if (
                valid.shape != supervised.shape
                or bool((supervised & ~valid).any())
                or not bool(supervised.any())
            ):
                raise ValueError(
                    "token NLL VJP targets escape the valid execution grid"
                )
            vocabulary = int(run.logits.shape[-1])
            supervised_targets = selected_targets[supervised]
            if bool(
                (
                    (supervised_targets < 0)
                    | (supervised_targets >= vocabulary)
                ).any()
            ):
                raise ValueError(
                    "token NLL VJP target escapes the vocabulary"
                )
            logits = run.logits[supervised]
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            losses = F.cross_entropy(
                logits,
                supervised_targets,
                reduction="none",
            )
            captured_indices = torch.nonzero(
                supervised,
                as_tuple=False,
            ).detach().to(dtype=torch.int64).contiguous()
            captured_losses = losses.detach().contiguous()
            return losses

        execution, gradients = self._execute(
            adapter,
            model_inputs,
            x4_head=x4_head,
            h4_head=h4_head,
            h4_vjp_objective=token_objective,
            h4_vjp_chunk_size=vjp_chunk_size,
        )
        if (
            gradients is None
            or captured_indices is None
            or captured_losses is None
        ):
            raise RuntimeError("token NLL VJP execution omitted its outputs")
        result = Gemma3L3L4TokenNLLVJP(
            execution=execution,
            supervised_indices=captured_indices,
            token_losses=captured_losses,
            h4_gradients=gradients,
            targets_sha256=targets_sha256,
            ignore_index=ignore_index,
            vjp_chunk_size=vjp_chunk_size,
            backward_call_count=(
                int(captured_losses.shape[0]) + vjp_chunk_size - 1
            )
            // vjp_chunk_size,
        )
        result.validate_integrity()
        return result

    def execute_h4_token_teacher_kl_vjps(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        teacher_logits: Tensor,
        supervised_indices: Tensor,
        vjp_chunk_size: int = 8,
        h4_head: Gemma3L3L4CorrectionProvider | None = None,
        objective_dtype: torch.dtype | None = None,
    ) -> Gemma3L3L4TokenTeacherKLVJP:
        """Return exact per-token ``KL(teacher || candidate)`` H4 VJPs.

        Teacher logits are copied into a detached authenticated snapshot and
        are never retained in the result.  Supervised rows must be supplied
        in canonical batch-major, position-major order.  One candidate
        forward is reused across bounded batched-VJP backward chunks.  Passing
        ``objective_dtype=torch.float64`` casts both selected logit rows before
        every teacher-KL operation; ``None`` preserves the legacy computation.
        """

        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError(
                "token teacher-KL VJP model inputs need an [B, S] input grid"
            )
        if (
            not isinstance(teacher_logits, Tensor)
            or teacher_logits.ndim != 3
            or teacher_logits.shape[-1] <= 0
        ):
            raise ValueError(
                "token teacher-KL VJP teacher logits must have shape "
                "[B, S, V]"
            )
        if not teacher_logits.is_floating_point():
            raise ValueError(
                "token teacher-KL VJP teacher logits must be floating-point"
            )
        if (
            teacher_logits.layout != torch.strided
            or teacher_logits.device.type == "meta"
        ):
            raise ValueError(
                "token teacher-KL VJP teacher logits must be materialized"
            )
        if teacher_logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "token teacher-KL VJP teacher logits differ from the model "
                "input grid"
            )
        if not bool(torch.isfinite(teacher_logits).all()):
            raise ValueError(
                "token teacher-KL VJP teacher logits must be finite"
            )
        if (
            not isinstance(supervised_indices, Tensor)
            or supervised_indices.ndim != 2
            or supervised_indices.shape[1] != 2
            or supervised_indices.dtype != torch.int64
            or supervised_indices.requires_grad
            or supervised_indices.shape[0] <= 0
        ):
            raise ValueError(
                "token teacher-KL VJP indices must be nonempty int64 [N, 2]"
            )
        if type(vjp_chunk_size) is not int or vjp_chunk_size <= 0:
            raise ValueError(
                "token teacher-KL VJP chunk size must be a positive integer"
            )
        if objective_dtype not in (None, torch.float64):
            raise ValueError(
                "token teacher-KL VJP objective dtype must be None or "
                "torch.float64"
            )
        objective_dtype_name = (
            None if objective_dtype is None else str(objective_dtype)
        )
        teacher_snapshot = teacher_logits.detach().clone().contiguous()
        indices_snapshot = (
            supervised_indices.detach().clone().contiguous()
        )
        teacher_logits_sha256 = _runtime_tensor_sha256(teacher_snapshot)
        batch_count, sequence_length = input_ids.shape
        indices_on_input = indices_snapshot.to(input_ids.device)
        if (
            bool((indices_on_input[:, 0] < 0).any())
            or bool((indices_on_input[:, 0] >= batch_count).any())
            or bool((indices_on_input[:, 1] < 0).any())
            or bool((indices_on_input[:, 1] >= sequence_length).any())
        ):
            raise ValueError(
                "token teacher-KL VJP indices escape the model input grid"
            )
        flattened = (
            indices_on_input[:, 0] * sequence_length
            + indices_on_input[:, 1]
        )
        if indices_snapshot.shape[0] > 1 and not bool(
            (flattened[1:] > flattened[:-1]).all()
        ):
            raise ValueError(
                "token teacher-KL VJP indices are not canonical"
            )
        captured_indices: Tensor | None = None
        captured_divergences: Tensor | None = None

        def token_objective(run: AdapterRun) -> Tensor:
            nonlocal captured_indices, captured_divergences
            if (
                run.logits.ndim != 3
                or tuple(run.logits.shape)
                != tuple(teacher_snapshot.shape)
            ):
                raise ValueError(
                    "token teacher-KL VJP teacher and candidate grids differ"
                )
            selected_indices = indices_snapshot.to(run.logits.device)
            valid = run.sequence.query_valid_mask.to(run.logits.device)
            if (
                valid.shape != run.logits.shape[:2]
                or not bool(
                    valid[
                        selected_indices[:, 0],
                        selected_indices[:, 1],
                    ].all()
                )
            ):
                raise ValueError(
                    "token teacher-KL VJP indices escape the valid "
                    "execution grid"
                )
            candidate = run.logits[
                selected_indices[:, 0], selected_indices[:, 1]
            ]
            if objective_dtype is None:
                if candidate.dtype in (torch.float16, torch.bfloat16):
                    candidate = candidate.float()
                teacher = teacher_snapshot.to(
                    device=run.logits.device,
                    dtype=candidate.dtype,
                )[
                    selected_indices[:, 0], selected_indices[:, 1]
                ]
            else:
                candidate = candidate.to(dtype=torch.float64)
                teacher = teacher_snapshot.to(device=run.logits.device)[
                    selected_indices[:, 0], selected_indices[:, 1]
                ].to(dtype=torch.float64)
            teacher_log_probabilities = F.log_softmax(teacher, dim=-1)
            candidate_log_probabilities = F.log_softmax(candidate, dim=-1)
            divergences = (
                teacher_log_probabilities.exp()
                * (
                    teacher_log_probabilities
                    - candidate_log_probabilities
                )
            ).sum(dim=-1)
            if not bool(torch.isfinite(divergences).all()):
                raise ValueError(
                    "token teacher-KL VJP divergence is nonfinite"
                )
            captured_indices = selected_indices.detach().to(
                dtype=torch.int64
            ).contiguous()
            captured_divergences = divergences.detach().contiguous()
            return divergences

        execution, gradients = self._execute(
            adapter,
            model_inputs,
            h4_head=h4_head,
            h4_vjp_objective=token_objective,
            h4_vjp_chunk_size=vjp_chunk_size,
        )
        if (
            gradients is None
            or captured_indices is None
            or captured_divergences is None
        ):
            raise RuntimeError(
                "token teacher-KL VJP execution omitted its outputs"
            )
        if (
            _runtime_tensor_sha256(teacher_snapshot)
            != teacher_logits_sha256
        ):
            raise RuntimeError(
                "token teacher-KL VJP teacher snapshot drifted"
            )
        result = Gemma3L3L4TokenTeacherKLVJP(
            execution=execution,
            supervised_indices=captured_indices,
            token_kl_divergences=captured_divergences,
            h4_gradients=gradients,
            teacher_logits_sha256=teacher_logits_sha256,
            teacher_logits_shape=tuple(
                int(width) for width in teacher_snapshot.shape
            ),
            h4_head_sha256=execution.h4_head_sha256,
            vjp_chunk_size=vjp_chunk_size,
            backward_call_count=(
                int(captured_divergences.shape[0])
                + vjp_chunk_size
                - 1
            )
            // vjp_chunk_size,
            objective_dtype=objective_dtype_name,
        )
        result.validate_integrity()
        return result
