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

__all__ = [
    "AuthenticatedOracleSuffixResult",
    "Gemma3L3L4GraphOrganizedSVDShadowAccounting",
    "Gemma3L3L4GraphOrganizedSVDShadowResult",
    "Gemma3L3L4GraphOrganizedSVDShadowRuntime",
    "Gemma3L3L4CorrectionProvider",
    "Gemma3L3L4OnePassBridge",
    "Gemma3L3L4OnePassExecution",
    "Gemma3L3L4OnePassPrefix",
    "OracleSuffixRole",
    "ShadowArm",
    "gemma3_l3_l4_shadow_model_inputs_sha256",
    "validate_gemma3_l3_l4_shadow_model_inputs_sha256",
]


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLAN_KEY = "signed_gfa"
_X3_SITE = "layer.3.mlp.normalized_input"
_Y3_SITE = "layer.3.mlp.operator_output"
_X4_SITE = "layer.4.mlp.normalized_input"
_H4_SITE = "layer.4.output"
_ARMS = frozenset({"identity", "all_on"})


class Gemma3L3L4CorrectionProvider:
    """Nominal interface for an integrity-bound X4 or H4 correction head.

    The bridge accepts a head object rather than a free callback plus a
    caller-asserted hash.  This keeps execution provenance attached to the
    object whose tensors are authenticated immediately before and after use.
    """

    __slots__ = ()

    site: str
    artifact_sha256: str

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
    ) -> tuple[Tensor, str | None]:
        prefix.validate_integrity()
        if provider is None:
            return torch.zeros_like(reference), None
        if not isinstance(provider, Gemma3L3L4CorrectionProvider):
            raise TypeError(
                f"{label} must implement the authenticated correction "
                "provider interface"
            )
        provider.validate_integrity()
        if provider.site != expected_site:
            raise ValueError(f"{label} is bound to the wrong activation site")
        head_sha256 = provider.artifact_sha256
        _require_sha256(head_sha256, label=f"{label} artifact")
        reference_sha256 = _runtime_tensor_sha256(reference)
        correction = provider.correction(prefix, reference)
        provider.validate_integrity()
        prefix.validate_integrity()
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
        active = prefix.target_affected_mask.to(correction.device)
        if bool(active.any()) and not bool(
            torch.isfinite(correction[active]).all()
        ):
            raise ValueError(f"{label} correction is nonfinite")
        inactive = ~active
        if bool(inactive.any()) and not bool(
            (correction[inactive] == 0).all()
        ):
            raise ValueError(f"{label} correction must be zero off support")
        return (
            correction.to(device=reference.device, dtype=reference.dtype),
            head_sha256,
        )

    def _execute(
        self,
        adapter: Gemma3CausalLMAdapter,
        model_inputs: Mapping[str, Tensor],
        *,
        x4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_head: Gemma3L3L4CorrectionProvider | None = None,
        h4_vjp_objective: Callable[[AdapterRun], Tensor] | None = None,
    ) -> tuple[Gemma3L3L4OnePassExecution, Tensor | None]:
        """Execute one bridge pass, optionally retaining only the H4 VJP."""

        self.validate_integrity()
        self._authenticate_adapter(adapter)
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
            correction, x4_head_sha256 = self._head_correction(
                x4_head,
                prefix=prefix,
                expected_site=_X4_SITE,
                label="X4 head",
                reference=original,
            )
            if bool(active.any()):
                candidate_x4[active] += correction[active]
            if h4_vjp_objective is not None:
                candidate_x4 = candidate_x4.detach()
            return candidate_x4

        def at_h4(original: Tensor) -> Tensor:
            nonlocal candidate_h4, h4_head_sha256
            if prefix is None or candidate_x4 is None or candidate_h4 is not None:
                raise RuntimeError("one-pass H4 intervention order drifted")
            candidate_h4 = original.clone()
            correction, h4_head_sha256 = self._head_correction(
                h4_head,
                prefix=prefix,
                expected_site=_H4_SITE,
                label="H4 head",
                reference=original,
            )
            active = prefix.target_affected_mask
            if bool(active.any()):
                candidate_h4[active] += correction[active]
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
                    if (
                        not isinstance(loss, Tensor)
                        or loss.ndim != 0
                        or not loss.is_floating_point()
                        or not bool(torch.isfinite(loss))
                    ):
                        raise ValueError(
                            "H4 VJP objective must return one finite scalar"
                        )
                    (h4_gradient,) = torch.autograd.grad(
                        loss,
                        (candidate_h4,),
                        retain_graph=False,
                        create_graph=False,
                    )
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
