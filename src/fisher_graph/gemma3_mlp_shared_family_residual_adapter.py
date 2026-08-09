"""Standalone first-rung MLP shared-family compression runtime.

This module deliberately does not widen the claims made by
``heterogeneous_generator_atom_bank``.  That module is the authenticated
exact, full-rank v1 runtime.  The classes here consume its final Gemma
compilation as a read-only predecessor and build a separate candidate-owned
runtime for one directed, non-adjacent MLP pair.

Only one temporal invocation is replaced.  The carrier graph, the 39 ordered
invocations, the 75 edge occurrences, all identity carries, lexical tying,
attention generators, and the terminal boundary remain unchanged.  A
low-rank point stores one exact MLP core once and three factorized projection
deltas for the non-anchor invocation; its two RMSNorm vectors remain exact.
No dense fused projection is materialized or registered by the hot path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import SequenceContext
from .causal_carrier_transport_graph import (
    CarrierTransportExecution,
    CarrierTransportExecutionReceipt,
    TransportContext,
    TransportMessage,
    execute_causal_carrier_transport_graph,
    validate_carrier_transport_execution_receipt,
)
from .gemma3_heterogeneous_generator_family_compiler import (
    Gemma3HeterogeneousGeneratorCompilation,
)
from .gemma3_heterogeneous_whole_model_generator_family_protocol import (
    PINNED_HIDDEN_WIDTH,
    PINNED_MLP_WIDTH,
    PINNED_SOURCE_SCALARS,
)
from .gemma3_mlp_shared_family_residual_adapter_setup import (
    build_gemma3_mlp_shared_family_residual_adapter_setup,
)
from .heterogeneous_generator_atom_bank import (
    AttentionAtomRuntime,
    AtomEmission,
    GeneratorInvocationSpec,
    HeterogeneousGeneratorManifest,
    LIFT_SCOPE_FULL_HETEROGENEOUS,
    LexicalAtomRuntime,
    MLPAtomRuntime,
    TerminalRuntime,
    module_physical_scalar_count,
    module_state_sha256,
    tensor_sha256,
    validate_heterogeneous_generator_manifest,
)


MLP_SHARED_FAMILY_PAIR_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_shared_family.pair.v1"
)
MLP_SHARED_FAMILY_ARTIFACT_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_shared_family.artifact.v1"
)
MLP_SHARED_FAMILY_CONTENT_BLOB_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_shared_family.content_blob.v1"
)
MLP_SHARED_FAMILY_CONTENT_POINT_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_shared_family.content_point.v1"
)
MLP_SHARED_FAMILY_CONTENT_STORE_SCHEMA: Final = (
    "fisher_graph.gemma3_mlp_shared_family.content_store.v1"
)
MLP_SHARED_FAMILY_FORMAT_VERSION: Final = 1

MLP_FAMILY_ID: Final = "family.mlp.gated_gelu_h640_i2048"
PRIMARY_RESIDUAL_RANKS: Final = (256, 128, 64, 32)
PRIMARY_POINT_ORDER: Final = (None, *PRIMARY_RESIDUAL_RANKS)

DENSE_EXACT_CONTROL: Final = "dense_exact_control"
SHARED_CORE_LOW_RANK: Final = "shared_core_low_rank"
ZERO_RESIDUAL_SHARED_CORE: Final = "zero_residual_shared_core"
INDEPENDENT_LOW_RANK_NONSHARING: Final = "independent_low_rank_nonsharing"
ARTIFACT_KINDS: Final = frozenset(
    {
        DENSE_EXACT_CONTROL,
        SHARED_CORE_LOW_RANK,
        ZERO_RESIDUAL_SHARED_CORE,
        INDEPENDENT_LOW_RANK_NONSHARING,
    }
)

FITTED_ASSIGNMENT: Final = "fitted"
SWAPPED_ASSIGNMENT: Final = "swapped"
SHUFFLED_ASSIGNMENT: Final = "deterministic_shuffled"
ASSIGNMENT_MODES: Final = frozenset(
    {FITTED_ASSIGNMENT, SWAPPED_ASSIGNMENT, SHUFFLED_ASSIGNMENT}
)

_PAIR_DOMAIN = b"fisher_graph:gemma3-mlp-shared-family-pair:v1\0"
_POINT_DOMAIN = b"fisher_graph:gemma3-mlp-shared-family-point:v1\0"
_ARTIFACT_DOMAIN = b"fisher_graph:gemma3-mlp-shared-family-artifact:v1\0"
_TENSOR_CATALOG_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-tensor-catalog:v1\0"
)
_CONTENT_BLOB_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-content-blob:v1\0"
)
_CONTENT_POINT_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-content-point:v1\0"
)
_CONTENT_STORE_DOMAIN = (
    b"fisher_graph:gemma3-mlp-shared-family-content-store:v1\0"
)
_RUNTIME_DOMAIN = b"fisher_graph:gemma3-mlp-shared-family-runtime:v1\0"
_HETEROGENEOUS_MODULE_DOMAIN = (
    b"fisher_graph:heterogeneous-generator-module:v1\0"
)

_PINNED_MLP_INVOCATION_SCALARS = 3_933_440
_PINNED_MLP_RESIDUAL_FIXED_SCALARS = 1_280
_PINNED_MLP_RESIDUAL_RANK_SLOPE = 8_064
PINNED_SUCCESSOR_SETUP_SHA256: Final = str(
    build_gemma3_mlp_shared_family_residual_adapter_setup()["setup_sha256"]
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() for character in value
    ):
        raise ValueError(f"{label} must be a nonempty canonical identifier")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _storage_keys(module: nn.Module) -> set[tuple[str, int]]:
    return {
        (str(value.device), value.untyped_storage().data_ptr())
        for value in module.state_dict(keep_vars=True).values()
        if value.numel()
    }


def _assert_modules_storage_disjoint(modules: Sequence[nn.Module]) -> None:
    seen: set[tuple[str, int]] = set()
    for module in modules:
        current = _storage_keys(module)
        if seen & current:
            raise ValueError("candidate-owned modules alias physical storage")
        seen.update(current)


def _assert_source_storage_disjoint(
    candidate: nn.Module,
    sources: Sequence[nn.Module],
) -> None:
    candidate_keys = _storage_keys(candidate)
    source_keys: set[tuple[str, int]] = set()
    for source in sources:
        source_keys.update(_storage_keys(source))
    if candidate_keys & source_keys:
        raise ValueError("candidate retained exact-predecessor tensor storage")


def _block_ordinal(invocation: GeneratorInvocationSpec) -> int:
    if (
        invocation.event_kind != "mlp"
        or invocation.invocation_index < 2
        or invocation.invocation_index > 36
        or invocation.invocation_index % 2
    ):
        raise ValueError("invocation is not an ordered Gemma MLP event")
    return (invocation.invocation_index - 2) // 2


@dataclass(frozen=True, slots=True)
class MLPFamilyPairManifest:
    """Authenticated directed non-adjacent pair selected on A-fit."""

    exact_plan_protocol_sha256: str
    successor_setup_sha256: str
    exact_plan_sha256: str
    exact_manifest_sha256: str
    source_model_fingerprint: str
    anchor_invocation_id: str
    member_invocation_id: str
    anchor_atom_id: str
    member_atom_id: str
    anchor_block_ordinal: int
    member_block_ordinal: int
    a_fit_manifest_sha256: str
    selection_evidence_sha256: str
    family_id: str = MLP_FAMILY_ID
    selected_on: str = "A_fit"
    nonadjacency_is_eligibility_only: bool = True
    block_ordinal_used_as_numerical_feature: bool = False
    frozen_before_A_select: bool = True
    schema: str = MLP_SHARED_FAMILY_PAIR_SCHEMA
    format_version: int = MLP_SHARED_FAMILY_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != MLP_SHARED_FAMILY_PAIR_SCHEMA:
            raise ValueError("MLP pair manifest schema is unsupported")
        if self.format_version != MLP_SHARED_FAMILY_FORMAT_VERSION:
            raise ValueError("MLP pair manifest version is unsupported")
        for label, value in (
            ("exact_plan_protocol_sha256", self.exact_plan_protocol_sha256),
            ("successor_setup_sha256", self.successor_setup_sha256),
            ("exact_plan_sha256", self.exact_plan_sha256),
            ("exact_manifest_sha256", self.exact_manifest_sha256),
            ("source_model_fingerprint", self.source_model_fingerprint),
            ("a_fit_manifest_sha256", self.a_fit_manifest_sha256),
            ("selection_evidence_sha256", self.selection_evidence_sha256),
        ):
            _require_sha256(value, label=label)
        for label, value in (
            ("anchor_invocation_id", self.anchor_invocation_id),
            ("member_invocation_id", self.member_invocation_id),
            ("anchor_atom_id", self.anchor_atom_id),
            ("member_atom_id", self.member_atom_id),
            ("family_id", self.family_id),
        ):
            _require_id(value, label=label)
        if self.anchor_invocation_id == self.member_invocation_id:
            raise ValueError("MLP family pair members must be distinct")
        if self.anchor_atom_id == self.member_atom_id:
            raise ValueError("exact pair members must start as distinct atoms")
        for label, value in (
            ("anchor_block_ordinal", self.anchor_block_ordinal),
            ("member_block_ordinal", self.member_block_ordinal),
        ):
            _require_int(value, label=label)
        if abs(self.anchor_block_ordinal - self.member_block_ordinal) <= 1:
            raise ValueError("first MLP family pair must be non-adjacent")
        if (
            self.family_id != MLP_FAMILY_ID
            or self.selected_on != "A_fit"
            or self.nonadjacency_is_eligibility_only is not True
            or self.block_ordinal_used_as_numerical_feature is not False
            or self.frozen_before_A_select is not True
        ):
            raise ValueError("MLP pair selection firewall drifted")

    def payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "exact_plan_protocol_sha256": self.exact_plan_protocol_sha256,
            "successor_setup_sha256": self.successor_setup_sha256,
            "exact_plan_sha256": self.exact_plan_sha256,
            "exact_manifest_sha256": self.exact_manifest_sha256,
            "source_model_fingerprint": self.source_model_fingerprint,
            "family_id": self.family_id,
            "anchor_invocation_id": self.anchor_invocation_id,
            "member_invocation_id": self.member_invocation_id,
            "anchor_atom_id": self.anchor_atom_id,
            "member_atom_id": self.member_atom_id,
            "anchor_block_ordinal": self.anchor_block_ordinal,
            "member_block_ordinal": self.member_block_ordinal,
            "a_fit_manifest_sha256": self.a_fit_manifest_sha256,
            "selection_evidence_sha256": self.selection_evidence_sha256,
            "selected_on": self.selected_on,
            "nonadjacency_is_eligibility_only": (
                self.nonadjacency_is_eligibility_only
            ),
            "block_ordinal_used_as_numerical_feature": (
                self.block_ordinal_used_as_numerical_feature
            ),
            "frozen_before_A_select": self.frozen_before_A_select,
        }

    @property
    def fingerprint_sha256(self) -> str:
        return _fingerprint(_PAIR_DOMAIN, self.payload())

    def to_dict(self) -> dict[str, object]:
        result = self.payload()
        result["pair_manifest_sha256"] = self.fingerprint_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MLPFamilyPairManifest":
        expected = {
            "schema",
            "format_version",
            "exact_plan_protocol_sha256",
            "successor_setup_sha256",
            "exact_plan_sha256",
            "exact_manifest_sha256",
            "source_model_fingerprint",
            "family_id",
            "anchor_invocation_id",
            "member_invocation_id",
            "anchor_atom_id",
            "member_atom_id",
            "anchor_block_ordinal",
            "member_block_ordinal",
            "a_fit_manifest_sha256",
            "selection_evidence_sha256",
            "selected_on",
            "nonadjacency_is_eligibility_only",
            "block_ordinal_used_as_numerical_feature",
            "frozen_before_A_select",
            "pair_manifest_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("MLP pair manifest fields are not exact")
        kwargs = {key: value[key] for key in expected - {"pair_manifest_sha256"}}
        result = cls(**kwargs)  # type: ignore[arg-type]
        _require_sha256(
            value["pair_manifest_sha256"],
            label="pair_manifest_sha256",
        )
        if result.to_dict() != dict(value):
            raise ValueError("MLP pair manifest authentication failed")
        return result


def build_mlp_family_pair_manifest(
    exact_compilation: Gemma3HeterogeneousGeneratorCompilation,
    *,
    anchor_invocation_id: str,
    member_invocation_id: str,
    successor_setup_sha256: str,
    a_fit_manifest_sha256: str,
    selection_evidence_sha256: str,
) -> MLPFamilyPairManifest:
    """Bind a frozen A-fit selection to the exact full-scope predecessor."""

    if not isinstance(
        exact_compilation,
        Gemma3HeterogeneousGeneratorCompilation,
    ):
        raise TypeError("exact_compilation has the wrong type")
    manifest = exact_compilation.plan.manifest_for_scope(
        LIFT_SCOPE_FULL_HETEROGENEOUS
    )
    by_id = {item.invocation_id: item for item in manifest.invocations}
    try:
        anchor = by_id[anchor_invocation_id]
        member = by_id[member_invocation_id]
    except KeyError as error:
        raise ValueError("selected MLP invocation is absent") from error
    if anchor.event_kind != "mlp" or member.event_kind != "mlp":
        raise ValueError("family pair must contain two MLP invocations")
    if (
        anchor.family_id != MLP_FAMILY_ID
        or member.family_id != MLP_FAMILY_ID
        or anchor.atom_id is None
        or member.atom_id is None
    ):
        raise ValueError("selected invocations are not in the full MLP family")
    return MLPFamilyPairManifest(
        exact_plan_protocol_sha256=exact_compilation.plan.protocol_sha256,
        successor_setup_sha256=_require_sha256(
            successor_setup_sha256,
            label="successor_setup_sha256",
        ),
        exact_plan_sha256=exact_compilation.plan.fingerprint_sha256,
        exact_manifest_sha256=manifest.fingerprint_sha256,
        source_model_fingerprint=exact_compilation.plan.source_model_fingerprint,
        anchor_invocation_id=anchor.invocation_id,
        member_invocation_id=member.invocation_id,
        anchor_atom_id=anchor.atom_id,
        member_atom_id=member.atom_id,
        anchor_block_ordinal=_block_ordinal(anchor),
        member_block_ordinal=_block_ordinal(member),
        a_fit_manifest_sha256=_require_sha256(
            a_fit_manifest_sha256,
            label="a_fit_manifest_sha256",
        ),
        selection_evidence_sha256=_require_sha256(
            selection_evidence_sha256,
            label="selection_evidence_sha256",
        ),
    )


def validate_mlp_family_pair_binding(
    pair: MLPFamilyPairManifest | Mapping[str, object],
    *,
    exact_compilation: Gemma3HeterogeneousGeneratorCompilation,
) -> MLPFamilyPairManifest:
    result = (
        pair
        if isinstance(pair, MLPFamilyPairManifest)
        else MLPFamilyPairManifest.from_dict(pair)
    )
    manifest = exact_compilation.plan.manifest_for_scope(
        LIFT_SCOPE_FULL_HETEROGENEOUS
    )
    if (
        result.exact_plan_protocol_sha256
        != exact_compilation.plan.protocol_sha256
        or result.exact_plan_sha256 != exact_compilation.plan.fingerprint_sha256
        or result.exact_manifest_sha256 != manifest.fingerprint_sha256
        or result.source_model_fingerprint
        != exact_compilation.plan.source_model_fingerprint
    ):
        raise ValueError("MLP pair belongs to another exact compilation")
    by_id = {item.invocation_id: item for item in manifest.invocations}
    anchor = by_id.get(result.anchor_invocation_id)
    member = by_id.get(result.member_invocation_id)
    if (
        anchor is None
        or member is None
        or anchor.event_kind != "mlp"
        or member.event_kind != "mlp"
        or anchor.family_id != result.family_id
        or member.family_id != result.family_id
        or anchor.atom_id != result.anchor_atom_id
        or member.atom_id != result.member_atom_id
        or _block_ordinal(anchor) != result.anchor_block_ordinal
        or _block_ordinal(member) != result.member_block_ordinal
    ):
        raise ValueError("MLP pair binding drifted from the exact manifest")
    return result


def pinned_mlp_pair_resource_point(
    residual_rank: int | None,
) -> dict[str, object]:
    """Return the canonical whole-model resource row for the first pair."""

    if residual_rank is not None and residual_rank not in PRIMARY_RESIDUAL_RANKS:
        raise ValueError("primary MLP residual rank is not preregistered")
    adapter = (
        _PINNED_MLP_INVOCATION_SCALARS
        if residual_rank is None
        else _PINNED_MLP_RESIDUAL_FIXED_SCALARS
        + _PINNED_MLP_RESIDUAL_RANK_SLOPE * residual_rank
    )
    family = _PINNED_MLP_INVOCATION_SCALARS + adapter
    candidate = (
        PINNED_SOURCE_SCALARS
        - 2 * _PINNED_MLP_INVOCATION_SCALARS
        + family
    )
    removed = PINNED_SOURCE_SCALARS - candidate
    representation = (
        "exact_full_rank_dense_residual"
        if residual_rank is None
        else f"low_rank_residual_r{residual_rank}"
    )
    return {
        "point_id": (
            "shared_core.residual_dense"
            if residual_rank is None
            else f"shared_core.residual_{residual_rank}"
        ),
        "residual_rank": residual_rank,
        "residual_representation": representation,
        "shared_core_count": 1,
        "shared_core_stored_scalars": _PINNED_MLP_INVOCATION_SCALARS,
        "residual_adapter_count": 1,
        "per_residual_adapter_stored_scalars": adapter,
        "family_candidate_stored_scalars": family,
        "candidate_stored_scalars": candidate,
        "removed_scalars": removed,
        "whole_model_reduction_fraction": removed / PINNED_SOURCE_SCALARS,
        "learned_correction_edge_count": 0,
        "exact_dense_residual_control": residual_rank is None,
    }


def pinned_mlp_pair_resource_ladder() -> tuple[dict[str, object], ...]:
    return tuple(pinned_mlp_pair_resource_point(rank) for rank in PRIMARY_POINT_ORDER)


class LowRankProjectionDelta(nn.Module):
    """One ``left @ right`` projection delta with no fused dense buffer."""

    def __init__(
        self,
        *,
        left: Tensor,
        right: Tensor,
        output_width: int,
        input_width: int,
        rank: int,
    ) -> None:
        super().__init__()
        for label, value in (
            ("output_width", output_width),
            ("input_width", input_width),
            ("rank", rank),
        ):
            _require_int(value, label=label, minimum=1)
        if (
            not isinstance(left, Tensor)
            or not isinstance(right, Tensor)
            or left.shape != (output_width, rank)
            or right.shape != (rank, input_width)
            or not left.is_floating_point()
            or not right.is_floating_point()
            or left.dtype != right.dtype
            or left.device != right.device
            or not bool(torch.isfinite(left).all())
            or not bool(torch.isfinite(right).all())
        ):
            raise ValueError("low-rank projection factors are invalid")
        self.output_width = output_width
        self.input_width = input_width
        self.rank = rank
        self.left = nn.Parameter(left.detach().clone().contiguous())
        self.right = nn.Parameter(right.detach().clone().contiguous())

    @property
    def stored_scalar_count(self) -> int:
        return self.rank * (self.input_width + self.output_width)

    def forward(self, values: Tensor) -> Tensor:
        if not isinstance(values, Tensor) or values.shape[-1] != self.input_width:
            raise ValueError("projection-delta input width drifted")
        return F.linear(F.linear(values, self.right), self.left)

    def materialize_for_test(self) -> Tensor:
        """Materialize only for controlled algebra tests, never runtime state."""

        return self.left @ self.right

    def metadata(self) -> dict[str, object]:
        return {
            "input_width": self.input_width,
            "output_width": self.output_width,
            "rank": self.rank,
            "stored_scalar_count": self.stored_scalar_count,
            "fused_dense_buffer_present": False,
            "state_sha256": module_state_sha256(self),
        }


@dataclass(frozen=True, slots=True)
class _MLPComponents:
    input_norm: nn.Module
    feed_forward: nn.Module
    output_norm: nn.Module
    gate: nn.Linear
    up: nn.Linear
    down: nn.Linear
    activation: str


def _extract_mlp_components(atom: MLPAtomRuntime) -> _MLPComponents:
    mutation = getattr(atom, "mutation", None)
    input_norm = getattr(mutation, "input_norm", None)
    output_norm = getattr(mutation, "output_norm", None)
    chart = getattr(mutation, "chart_generator", None)
    operator = getattr(chart, "operator", None)
    feed_forward = getattr(operator, "operator", operator)
    if not all(
        isinstance(item, nn.Module)
        for item in (input_norm, output_norm, feed_forward)
    ):
        raise TypeError("exact MLP atom omitted its pipeline modules")
    gate = getattr(feed_forward, "gate_proj", None)
    up = getattr(feed_forward, "up_proj", None)
    down = getattr(feed_forward, "down_proj", None)
    if not all(isinstance(item, nn.Linear) for item in (gate, up, down)):
        raise TypeError("shared-family MLP requires gate/up/down projections")
    assert isinstance(gate, nn.Linear)
    assert isinstance(up, nn.Linear)
    assert isinstance(down, nn.Linear)
    if any(item.bias is not None for item in (gate, up, down)):
        raise ValueError("Gemma shared-family MLP projections must be bias-free")
    if (
        gate.in_features != up.in_features
        or gate.out_features != up.out_features
        or down.in_features != gate.out_features
        or down.out_features != gate.in_features
    ):
        raise ValueError("MLP projection shapes are incompatible")
    spec = getattr(feed_forward, "spec", None)
    activation = getattr(spec, "activation", None)
    if activation is None:
        activation = getattr(feed_forward, "activation_name", None)
    if activation not in {"gelu_pytorch_tanh", "gelu", "silu"}:
        raise ValueError("MLP activation contract is unsupported")
    return _MLPComponents(
        input_norm=input_norm,
        feed_forward=feed_forward,
        output_norm=output_norm,
        gate=gate,
        up=up,
        down=down,
        activation=activation,
    )


def _activate(values: Tensor, activation: str) -> Tensor:
    if activation == "gelu_pytorch_tanh":
        return F.gelu(values, approximate="tanh")
    if activation == "gelu":
        return F.gelu(values)
    if activation == "silu":
        return F.silu(values)
    raise RuntimeError("validated MLP activation disappeared")


def _rms_norm_weight_and_epsilon(norm: nn.Module) -> tuple[Tensor, float]:
    weight = getattr(norm, "weight", None)
    spec = getattr(norm, "spec", None)
    epsilon = getattr(spec, "epsilon", getattr(norm, "epsilon", None))
    if (
        not isinstance(weight, Tensor)
        or weight.ndim != 1
        or not weight.is_floating_point()
        or epsilon is None
        or isinstance(epsilon, bool)
        or not isinstance(epsilon, (float, int))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError("shared-family runtime requires Gemma RMSNorm semantics")
    scale_parameterization = getattr(spec, "scale_parameterization", "unit_offset")
    if scale_parameterization != "unit_offset":
        raise ValueError("shared-family runtime requires unit-offset RMSNorm")
    return weight, float(epsilon)


def _apply_rms_norm_delta(
    anchor_norm: nn.Module,
    values: Tensor,
    weight_delta: Tensor | None,
) -> Tensor:
    weight, epsilon = _rms_norm_weight_and_epsilon(anchor_norm)
    if values.shape[-1] != weight.shape[0]:
        raise ValueError("RMSNorm input width drifted")
    if weight_delta is not None and (
        weight_delta.shape != weight.shape
        or weight_delta.dtype != weight.dtype
        or weight_delta.device != weight.device
    ):
        raise ValueError("RMSNorm delta does not match its anchor")
    effective = weight if weight_delta is None else weight + weight_delta
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    return (normalized * (1.0 + effective.float())).to(dtype=values.dtype)


def _norm_weight_delta(member_norm: nn.Module, anchor_norm: nn.Module) -> Tensor:
    member_weight, member_epsilon = _rms_norm_weight_and_epsilon(member_norm)
    anchor_weight, anchor_epsilon = _rms_norm_weight_and_epsilon(anchor_norm)
    if (
        member_weight.shape != anchor_weight.shape
        or member_weight.dtype != anchor_weight.dtype
        or member_weight.device != anchor_weight.device
        or member_epsilon != anchor_epsilon
    ):
        raise ValueError("MLP family RMSNorm contracts are incompatible")
    return (member_weight.detach() - anchor_weight.detach()).contiguous()


class DenseExactControlMLPRuntime(nn.Module):
    """Source-sized exact non-anchor adapter used by the dense control."""

    artifact_kind = DENSE_EXACT_CONTROL

    def __init__(self, exact_atom: MLPAtomRuntime) -> None:
        super().__init__()
        if not isinstance(exact_atom, MLPAtomRuntime):
            raise TypeError("dense control requires an exact MLP atom")
        self.exact_atom = exact_atom
        self.eval().requires_grad_(False)

    @property
    def stored_scalar_count(self) -> int:
        return module_physical_scalar_count(self)

    def emit_with_core(
        self,
        carrier: Tensor,
        sequence: object,
        core_atom: MLPAtomRuntime,
    ) -> AtomEmission:
        del core_atom
        return self.exact_atom.emit(carrier, sequence)

    def metadata(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "stored_scalar_count": self.stored_scalar_count,
            "exact_dense_invocation_owned": True,
            "shared_core_used_by_adapter": False,
            "dense_core_residual_aggregation_live": False,
            "dense_core_residual_relationship_observational": True,
            "fused_dense_buffer_present": False,
            "state_sha256": module_state_sha256(self),
        }


class SharedCoreResidualAdapterMLPRuntime(nn.Module):
    """Norm-weight and projection deltas around one exact shared core."""

    artifact_kind = SHARED_CORE_LOW_RANK

    def __init__(
        self,
        *,
        input_norm_delta: Tensor | None,
        output_norm_delta: Tensor | None,
        activation: str,
        gate_delta: LowRankProjectionDelta | None,
        up_delta: LowRankProjectionDelta | None,
        down_delta: LowRankProjectionDelta | None,
    ) -> None:
        super().__init__()
        deltas = (gate_delta, up_delta, down_delta)
        if any(item is None for item in deltas) and not all(
            item is None for item in deltas
        ):
            raise ValueError("projection deltas must be all present or all absent")
        if any(
            item is not None and not isinstance(item, LowRankProjectionDelta)
            for item in deltas
        ):
            raise TypeError("projection deltas have the wrong type")
        if activation not in {"gelu_pytorch_tanh", "gelu", "silu"}:
            raise ValueError("residual adapter activation is unsupported")
        if (input_norm_delta is None) != (output_norm_delta is None):
            raise ValueError("norm deltas must be both present or both absent")
        if input_norm_delta is not None:
            if (
                not isinstance(input_norm_delta, Tensor)
                or not isinstance(output_norm_delta, Tensor)
                or input_norm_delta.ndim != 1
                or output_norm_delta.shape != input_norm_delta.shape
                or not input_norm_delta.is_floating_point()
                or output_norm_delta.dtype != input_norm_delta.dtype
                or output_norm_delta.device != input_norm_delta.device
                or not bool(torch.isfinite(input_norm_delta).all())
                or not bool(torch.isfinite(output_norm_delta).all())
            ):
                raise ValueError("RMSNorm weight deltas are invalid")
            self.input_norm_delta = nn.Parameter(
                input_norm_delta.detach().clone().contiguous()
            )
            self.output_norm_delta = nn.Parameter(
                output_norm_delta.detach().clone().contiguous()
            )
        else:
            self.register_parameter("input_norm_delta", None)
            self.register_parameter("output_norm_delta", None)
        self.activation = activation
        self.gate_delta = gate_delta
        self.up_delta = up_delta
        self.down_delta = down_delta
        self.eval().requires_grad_(False)

    @property
    def residual_rank(self) -> int:
        return 0 if self.gate_delta is None else self.gate_delta.rank

    @property
    def stored_scalar_count(self) -> int:
        return module_physical_scalar_count(self)

    @property
    def is_zero_residual(self) -> bool:
        return self.gate_delta is None and self.input_norm_delta is None

    def projection_factors(self) -> dict[str, tuple[Tensor, Tensor]]:
        if self.gate_delta is None or self.up_delta is None or self.down_delta is None:
            return {}
        return {
            "gate_proj": (
                self.gate_delta.left.detach().clone(),
                self.gate_delta.right.detach().clone(),
            ),
            "up_proj": (
                self.up_delta.left.detach().clone(),
                self.up_delta.right.detach().clone(),
            ),
            "down_proj": (
                self.down_delta.left.detach().clone(),
                self.down_delta.right.detach().clone(),
            ),
        }

    def emit_with_core(
        self,
        carrier: Tensor,
        sequence: object,
        core_atom: MLPAtomRuntime,
    ) -> AtomEmission:
        if not isinstance(sequence, SequenceContext):
            raise TypeError("Gemma MLP requires a SequenceContext")
        if self.is_zero_residual:
            return core_atom.emit(carrier, sequence)
        core = _extract_mlp_components(core_atom)
        normalized = _apply_rms_norm_delta(
            core.input_norm,
            carrier,
            self.input_norm_delta,
        )
        if normalized.shape != carrier.shape:
            raise ValueError("adapter input norm must preserve carrier shape")
        gate = F.linear(normalized, core.gate.weight)
        up = F.linear(normalized, core.up.weight)
        if self.gate_delta is not None:
            assert self.up_delta is not None
            gate = gate + self.gate_delta(normalized)
            up = up + self.up_delta(normalized)
        features = _activate(gate, self.activation) * up
        raw = F.linear(features, core.down.weight)
        if self.down_delta is not None:
            raw = raw + self.down_delta(features)
        delta = _apply_rms_norm_delta(
            core.output_norm,
            raw,
            self.output_norm_delta,
        )
        if delta.shape != carrier.shape:
            raise ValueError("shared-family MLP must emit a full-width delta")
        return AtomEmission(
            value=delta.masked_fill(~sequence.query_valid_mask.unsqueeze(-1), 0)
        )

    def metadata(self) -> dict[str, object]:
        return {
            "artifact_kind": (
                ZERO_RESIDUAL_SHARED_CORE
                if self.is_zero_residual
                else self.artifact_kind
            ),
            "residual_rank": self.residual_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "norm_delta_stored_scalars": (
                0
                if self.input_norm_delta is None
                else self.input_norm_delta.numel()
                + self.output_norm_delta.numel()
            ),
            "anchor_norm_plus_delta_live": self.input_norm_delta is not None,
            "shared_core_registered_inside_adapter": False,
            "fused_dense_buffer_present": False,
            "state_sha256": module_state_sha256(self),
        }


class IndependentLowRankMLPRuntime(nn.Module):
    """Parameter-matched nonsharing control with direct low-rank weights."""

    artifact_kind = INDEPENDENT_LOW_RANK_NONSHARING

    def __init__(
        self,
        *,
        input_norm: nn.Module,
        output_norm: nn.Module,
        activation: str,
        gate: LowRankProjectionDelta,
        up: LowRankProjectionDelta,
        down: LowRankProjectionDelta,
    ) -> None:
        super().__init__()
        if not isinstance(input_norm, nn.Module) or not isinstance(
            output_norm,
            nn.Module,
        ):
            raise TypeError("independent control requires two exact norms")
        if not all(
            isinstance(item, LowRankProjectionDelta)
            for item in (gate, up, down)
        ):
            raise TypeError("independent control factors are invalid")
        if gate.rank != up.rank or gate.rank != down.rank:
            raise ValueError("independent control ranks must agree")
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.activation = activation
        self.gate = gate
        self.up = up
        self.down = down
        self.eval().requires_grad_(False)

    @property
    def residual_rank(self) -> int:
        return self.gate.rank

    @property
    def stored_scalar_count(self) -> int:
        return module_physical_scalar_count(self)

    def projection_factors(self) -> dict[str, tuple[Tensor, Tensor]]:
        return {
            "gate_proj": (
                self.gate.left.detach().clone(),
                self.gate.right.detach().clone(),
            ),
            "up_proj": (
                self.up.left.detach().clone(),
                self.up.right.detach().clone(),
            ),
            "down_proj": (
                self.down.left.detach().clone(),
                self.down.right.detach().clone(),
            ),
        }

    def emit_with_core(
        self,
        carrier: Tensor,
        sequence: object,
        core_atom: MLPAtomRuntime,
    ) -> AtomEmission:
        del core_atom
        if not isinstance(sequence, SequenceContext):
            raise TypeError("Gemma MLP requires a SequenceContext")
        normalized = self.input_norm(carrier)
        gate = self.gate(normalized)
        up = self.up(normalized)
        raw = self.down(_activate(gate, self.activation) * up)
        delta = self.output_norm(raw)
        if delta.shape != carrier.shape:
            raise ValueError("independent MLP must emit a full-width delta")
        return AtomEmission(
            value=delta.masked_fill(~sequence.query_valid_mask.unsqueeze(-1), 0)
        )

    def metadata(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "residual_rank": self.residual_rank,
            "stored_scalar_count": self.stored_scalar_count,
            "shared_core_used_by_adapter": False,
            "parameter_matched_to_shared_residual": True,
            "fused_dense_buffer_present": False,
            "state_sha256": module_state_sha256(self),
        }


AdapterRuntime = (
    DenseExactControlMLPRuntime
    | SharedCoreResidualAdapterMLPRuntime
    | IndependentLowRankMLPRuntime
)


def _projection_shapes(
    core_atom: MLPAtomRuntime,
) -> dict[str, tuple[int, int]]:
    components = _extract_mlp_components(core_atom)
    return {
        "gate_proj": (
            components.gate.out_features,
            components.gate.in_features,
        ),
        "up_proj": (
            components.up.out_features,
            components.up.in_features,
        ),
        "down_proj": (
            components.down.out_features,
            components.down.in_features,
        ),
    }


def _build_projection_deltas(
    factors: Mapping[str, tuple[Tensor, Tensor]],
    *,
    core_atom: MLPAtomRuntime,
    rank: int,
) -> dict[str, LowRankProjectionDelta]:
    expected = _projection_shapes(core_atom)
    if not isinstance(factors, Mapping) or set(factors) != set(expected):
        raise ValueError("projection factors must cover gate/up/down exactly")
    result: dict[str, LowRankProjectionDelta] = {}
    for name, (output_width, input_width) in expected.items():
        value = factors[name]
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise TypeError(f"{name} factors must be a left/right pair")
        left, right = value
        result[name] = LowRankProjectionDelta(
            left=left,
            right=right,
            output_width=output_width,
            input_width=input_width,
            rank=rank,
        )
    return result


def _truncated_svd_factors(matrix: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    if (
        not isinstance(matrix, Tensor)
        or matrix.ndim != 2
        or not matrix.is_floating_point()
        or not bool(torch.isfinite(matrix).all())
    ):
        raise ValueError("SVD initialization requires a finite floating matrix")
    _require_int(rank, label="rank", minimum=1)
    if rank > min(matrix.shape):
        raise ValueError("SVD initialization rank exceeds matrix rank bound")
    left, singular, right = torch.linalg.svd(
        matrix.detach().to(dtype=torch.float64),
        full_matrices=False,
    )
    scale = singular[:rank].clamp_min(0.0).sqrt()
    factor_left = (left[:, :rank] * scale).to(dtype=matrix.dtype)
    factor_right = (scale.unsqueeze(1) * right[:rank]).to(dtype=matrix.dtype)
    return factor_left.contiguous(), factor_right.contiguous()


def initialize_mlp_projection_factors(
    exact_compilation: Gemma3HeterogeneousGeneratorCompilation,
    *,
    pair_manifest: MLPFamilyPairManifest,
    rank: int,
    independent_nonsharing: bool = False,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Deterministic SVD initialization; this is not an A-fit claim."""

    pair = validate_mlp_family_pair_binding(
        pair_manifest,
        exact_compilation=exact_compilation,
    )
    executor = exact_compilation.assemble(LIFT_SCOPE_FULL_HETEROGENEOUS)
    by_id = {item.invocation_id: item for item in executor.manifest.invocations}
    anchor = executor.atom_for(by_id[pair.anchor_invocation_id])
    member = executor.atom_for(by_id[pair.member_invocation_id])
    if not isinstance(anchor, MLPAtomRuntime) or not isinstance(member, MLPAtomRuntime):
        raise TypeError("pair invocations did not resolve to MLP atoms")
    anchor_components = _extract_mlp_components(anchor)
    member_components = _extract_mlp_components(member)
    result: dict[str, tuple[Tensor, Tensor]] = {}
    for name, anchor_projection, member_projection in (
        ("gate_proj", anchor_components.gate, member_components.gate),
        ("up_proj", anchor_components.up, member_components.up),
        ("down_proj", anchor_components.down, member_components.down),
    ):
        target = member_projection.weight.detach()
        if not independent_nonsharing:
            target = target - anchor_projection.weight.detach()
        result[name] = _truncated_svd_factors(target, rank)
    return result


def _assignment(
    pair: MLPFamilyPairManifest,
    *,
    mode: str,
    shuffle_seed: int | None,
) -> tuple[str, str, int | None]:
    if mode not in ASSIGNMENT_MODES:
        raise ValueError("residual assignment mode is invalid")
    if mode == FITTED_ASSIGNMENT:
        if shuffle_seed is not None:
            raise ValueError("fitted assignment cannot carry a shuffle seed")
        return pair.anchor_invocation_id, pair.member_invocation_id, None
    if mode == SWAPPED_ASSIGNMENT:
        if shuffle_seed is not None:
            raise ValueError("swapped assignment cannot carry a shuffle seed")
        return pair.anchor_invocation_id, pair.anchor_invocation_id, None
    _require_int(shuffle_seed, label="shuffle_seed")
    # The seed family contains one residual.  Its only nontrivial invocation
    # shuffle is therefore fixed: attach the frozen member residual to the
    # anchor occurrence and leave the member occurrence on the bare anchor
    # core.  The seed authenticates the preregistered deterministic replay;
    # it does not refit or alter coefficients.
    return pair.anchor_invocation_id, pair.anchor_invocation_id, shuffle_seed


@dataclass(frozen=True, slots=True)
class MLPSharedFamilyInvocationReceipt:
    invocation_index: int
    invocation_id: str
    event_id: str
    event_kind: str
    provider_kind: str
    output_sha256: str
    routing_sha256: str | None

    def __post_init__(self) -> None:
        _require_int(self.invocation_index, label="invocation_index")
        for label, value in (
            ("invocation_id", self.invocation_id),
            ("event_id", self.event_id),
            ("event_kind", self.event_kind),
            ("provider_kind", self.provider_kind),
        ):
            _require_id(value, label=label)
        _require_sha256(self.output_sha256, label="output_sha256")
        if self.routing_sha256 is not None:
            _require_sha256(self.routing_sha256, label="routing_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "invocation_index": self.invocation_index,
            "invocation_id": self.invocation_id,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "provider_kind": self.provider_kind,
            "output_sha256": self.output_sha256,
            "routing_sha256": self.routing_sha256,
        }


@dataclass(frozen=True, slots=True)
class MLPSharedFamilyExecutionReceipt:
    runtime_sha256: str
    execution_nonce: str
    carrier_execution_receipt: CarrierTransportExecutionReceipt
    invocation_receipts: tuple[MLPSharedFamilyInvocationReceipt, ...]
    candidate_state_before_sha256: str
    candidate_state_after_sha256: str
    final_carrier_sha256: str
    normalized_output_sha256: str
    logits_sha256: str
    physical_stored_scalar_count: int
    learned_cross_event_correction_count: int = 0
    source_projection_call_count: int = 0
    source_fallback_call_count: int = 0
    compression_attempted: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("runtime_sha256", self.runtime_sha256),
            ("candidate_state_before_sha256", self.candidate_state_before_sha256),
            ("candidate_state_after_sha256", self.candidate_state_after_sha256),
            ("final_carrier_sha256", self.final_carrier_sha256),
            ("normalized_output_sha256", self.normalized_output_sha256),
            ("logits_sha256", self.logits_sha256),
        ):
            _require_sha256(value, label=label)
        _require_id(self.execution_nonce, label="execution_nonce")
        if len(self.invocation_receipts) != 39:
            raise ValueError("compressed execution must cover 39 invocations")
        if self.candidate_state_before_sha256 != self.candidate_state_after_sha256:
            raise ValueError("compressed candidate state changed during execution")
        _require_int(
            self.physical_stored_scalar_count,
            label="physical_stored_scalar_count",
            minimum=1,
        )
        if (
            self.learned_cross_event_correction_count != 0
            or self.source_projection_call_count != 0
            or self.source_fallback_call_count != 0
            or type(self.compression_attempted) is not bool
        ):
            raise ValueError("compressed execution crossed its claim firewall")

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_sha256": self.runtime_sha256,
            "execution_nonce": self.execution_nonce,
            "carrier_execution_receipt": self.carrier_execution_receipt.to_dict(),
            "invocation_receipts": [item.to_dict() for item in self.invocation_receipts],
            "candidate_state_before_sha256": self.candidate_state_before_sha256,
            "candidate_state_after_sha256": self.candidate_state_after_sha256,
            "final_carrier_sha256": self.final_carrier_sha256,
            "normalized_output_sha256": self.normalized_output_sha256,
            "logits_sha256": self.logits_sha256,
            "physical_stored_scalar_count": self.physical_stored_scalar_count,
            "learned_cross_event_correction_count": (
                self.learned_cross_event_correction_count
            ),
            "source_projection_call_count": self.source_projection_call_count,
            "source_fallback_call_count": self.source_fallback_call_count,
            "compression_attempted": self.compression_attempted,
        }


@dataclass(frozen=True, slots=True)
class MLPSharedFamilyExecution:
    output: Tensor
    normalized_output: Tensor
    logits: Tensor
    carrier_execution: CarrierTransportExecution
    receipt: MLPSharedFamilyExecutionReceipt


class Gemma3MLPSharedFamilyExecutor(nn.Module):
    """Candidate-owned 39-event executor with one adapted MLP invocation."""

    def __init__(
        self,
        *,
        topology_manifest: HeterogeneousGeneratorManifest,
        lexical_atom: LexicalAtomRuntime,
        attention_atoms: Mapping[str, AttentionAtomRuntime],
        exact_mlp_atoms: Mapping[str, MLPAtomRuntime],
        shared_core_atom: MLPAtomRuntime,
        terminal_runtime: TerminalRuntime,
        adapter: AdapterRuntime,
        pair_invocation_ids: tuple[str, str],
        core_invocation_id: str,
        adapted_invocation_id: str,
        artifact_kind: str,
        assignment_mode: str,
    ) -> None:
        super().__init__()
        topology = validate_heterogeneous_generator_manifest(topology_manifest)
        if topology.lift_scope != LIFT_SCOPE_FULL_HETEROGENEOUS:
            raise ValueError("compressed executor requires the final exact topology")
        if not isinstance(lexical_atom, LexicalAtomRuntime):
            raise TypeError("compressed executor requires one tied lexical atom")
        if not isinstance(terminal_runtime, TerminalRuntime):
            raise TypeError("compressed executor requires one terminal runtime")
        if artifact_kind not in ARTIFACT_KINDS:
            raise ValueError("compressed executor artifact kind is invalid")
        if assignment_mode not in ASSIGNMENT_MODES:
            raise ValueError("compressed executor assignment mode is invalid")
        attention_ids = tuple(
            item.invocation_id
            for item in topology.invocations
            if item.event_kind == "attention"
        )
        mlp_ids = tuple(
            item.invocation_id
            for item in topology.invocations
            if item.event_kind == "mlp"
        )
        if (
            type(pair_invocation_ids) is not tuple
            or len(pair_invocation_ids) != 2
            or len(set(pair_invocation_ids)) != 2
            or not set(pair_invocation_ids) <= set(mlp_ids)
        ):
            raise ValueError("compressed pair invocation inventory is invalid")
        if (
            not isinstance(attention_atoms, Mapping)
            or set(attention_atoms) != set(attention_ids)
            or any(
                not isinstance(item, AttentionAtomRuntime)
                for item in attention_atoms.values()
            )
        ):
            raise ValueError("candidate attention inventory drifted")
        if (
            adapted_invocation_id not in mlp_ids
            or core_invocation_id not in mlp_ids
            or core_invocation_id != pair_invocation_ids[0]
            or adapted_invocation_id not in pair_invocation_ids
            or not isinstance(exact_mlp_atoms, Mapping)
            or set(exact_mlp_atoms) != set(mlp_ids) - set(pair_invocation_ids)
            or any(
                not isinstance(item, MLPAtomRuntime)
                for item in exact_mlp_atoms.values()
            )
        ):
            raise ValueError("candidate MLP inventory drifted")
        if not isinstance(shared_core_atom, MLPAtomRuntime):
            raise TypeError("shared core must remain one exact owned MLP")
        if not isinstance(
            adapter,
            (
                DenseExactControlMLPRuntime,
                SharedCoreResidualAdapterMLPRuntime,
                IndependentLowRankMLPRuntime,
            ),
        ):
            raise TypeError("candidate adapter runtime is invalid")

        self.topology_manifest = topology
        self.lexical_atom = lexical_atom
        self._attention_ids = attention_ids
        self.attention_atoms = nn.ModuleList(
            [attention_atoms[item] for item in attention_ids]
        )
        self._exact_mlp_ids = tuple(
            item for item in mlp_ids if item not in pair_invocation_ids
        )
        self.exact_mlp_atoms = nn.ModuleList(
            [exact_mlp_atoms[item] for item in self._exact_mlp_ids]
        )
        self.shared_core_atom = shared_core_atom
        self.terminal_runtime = terminal_runtime
        self.adapter = adapter
        self.pair_invocation_ids = pair_invocation_ids
        self.core_invocation_id = core_invocation_id
        self.adapted_invocation_id = adapted_invocation_id
        self.artifact_kind = artifact_kind
        self.assignment_mode = assignment_mode
        _assert_modules_storage_disjoint(
            (
                self.lexical_atom,
                *self.attention_atoms,
                *self.exact_mlp_atoms,
                self.shared_core_atom,
                self.terminal_runtime,
                self.adapter,
            )
        )
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> "Gemma3MLPSharedFamilyExecutor":
        if type(mode) is not bool:
            raise TypeError("mode must be bool")
        if mode:
            raise RuntimeError("compiled shared-family executor is inference-only")
        return super().train(False)

    @property
    def deployed_scalar_count(self) -> int:
        return module_physical_scalar_count(self)

    @property
    def state_sha256(self) -> str:
        return module_state_sha256(self)

    @property
    def runtime_sha256(self) -> str:
        return _fingerprint(
            _RUNTIME_DOMAIN,
            {
                "topology_manifest_sha256": (
                    self.topology_manifest.fingerprint_sha256
                ),
                "artifact_kind": self.artifact_kind,
                "assignment_mode": self.assignment_mode,
                "core_invocation_id": self.core_invocation_id,
                "adapted_invocation_id": self.adapted_invocation_id,
                "candidate_state_sha256": self.state_sha256,
                "deployed_scalar_count": self.deployed_scalar_count,
            },
        )

    def attention_atom(self, invocation_id: str) -> AttentionAtomRuntime:
        try:
            index = self._attention_ids.index(invocation_id)
        except ValueError as error:
            raise KeyError(invocation_id) from error
        atom = self.attention_atoms[index]
        assert isinstance(atom, AttentionAtomRuntime)
        return atom

    def exact_mlp_atom(self, invocation_id: str) -> MLPAtomRuntime:
        if invocation_id == self.core_invocation_id:
            return self.shared_core_atom
        try:
            index = self._exact_mlp_ids.index(invocation_id)
        except ValueError as error:
            raise KeyError(invocation_id) from error
        atom = self.exact_mlp_atoms[index]
        assert isinstance(atom, MLPAtomRuntime)
        return atom

    @staticmethod
    def _receipt(
        invocation: GeneratorInvocationSpec,
        *,
        provider_kind: str,
        output: Tensor,
        routing: Tensor | None,
    ) -> MLPSharedFamilyInvocationReceipt:
        return MLPSharedFamilyInvocationReceipt(
            invocation_index=invocation.invocation_index,
            invocation_id=invocation.invocation_id,
            event_id=invocation.event_id,
            event_kind=invocation.event_kind,
            provider_kind=provider_kind,
            output_sha256=tensor_sha256(output),
            routing_sha256=None if routing is None else tensor_sha256(routing),
        )

    def execute(
        self,
        token_input: Tensor,
        sequence: object,
        *,
        execution_nonce: str,
    ) -> MLPSharedFamilyExecution:
        if not isinstance(token_input, Tensor) or token_input.ndim != 2:
            raise ValueError("token_input must be rank-two")
        _require_id(execution_nonce, label="execution_nonce")
        before = self.state_sha256
        invocations = self.topology_manifest.invocations
        by_event = {item.event_id: item for item in invocations[1:37]}
        receipts: dict[int, MLPSharedFamilyInvocationReceipt] = {}
        producers: dict[str, Any] = {}
        carrier = self.topology_manifest.carrier_manifest
        lexical_invocation = invocations[0]

        def lexical_producer(context: TransportContext) -> TransportMessage:
            value = self.lexical_atom.ingress(token_input)
            if value.shape != (*token_input.shape, self.topology_manifest.hidden_width):
                raise ValueError("lexical ingress returned the wrong shape")
            receipts[0] = self._receipt(
                lexical_invocation,
                provider_kind="exact_tied_lexical",
                output=value,
                routing=None,
            )
            return TransportMessage(edge_id=context.edge.edge_id, value=value)

        producers[carrier.edges[0].edge_id] = lexical_producer
        for edge in carrier.edges:
            if edge.message_kind not in {"attention_transport", "mlp_mutation"}:
                continue
            assert edge.target_event_id is not None
            invocation = by_event[edge.target_event_id]

            def producer(
                context: TransportContext,
                *,
                invocation: GeneratorInvocationSpec = invocation,
            ) -> TransportMessage:
                if context.carrier_input is None:
                    raise RuntimeError("mutation did not receive its carrier")
                if invocation.event_kind == "attention":
                    emission = self.attention_atom(invocation.invocation_id).emit(
                        context.carrier_input,
                        sequence,
                    )
                    if emission.attention_routing_weights is None:
                        raise ValueError("attention atom omitted routing weights")
                    provider_kind = "exact_attention"
                elif invocation.invocation_id == self.adapted_invocation_id:
                    emission = self.adapter.emit_with_core(
                        context.carrier_input,
                        sequence,
                        self.shared_core_atom,
                    )
                    if emission.attention_routing_weights is not None:
                        raise ValueError("adapted MLP emitted attention routing")
                    provider_kind = self.artifact_kind
                elif invocation.invocation_id in self.pair_invocation_ids:
                    emission = self.shared_core_atom.emit(
                        context.carrier_input,
                        sequence,
                    )
                    if emission.attention_routing_weights is not None:
                        raise ValueError("shared MLP core emitted attention routing")
                    provider_kind = "exact_shared_mlp_core"
                else:
                    emission = self.exact_mlp_atom(invocation.invocation_id).emit(
                        context.carrier_input,
                        sequence,
                    )
                    if emission.attention_routing_weights is not None:
                        raise ValueError("exact MLP emitted attention routing")
                    provider_kind = "exact_mlp"
                if emission.value.shape != context.carrier_input.shape:
                    raise ValueError("generator must emit one full-width delta")
                receipts[invocation.invocation_index] = self._receipt(
                    invocation,
                    provider_kind=provider_kind,
                    output=emission.value,
                    routing=emission.attention_routing_weights,
                )
                return TransportMessage(
                    edge_id=context.edge.edge_id,
                    value=emission.value,
                    attention_routing_weights=emission.attention_routing_weights,
                )

            producers[edge.edge_id] = producer

        carrier_execution = execute_causal_carrier_transport_graph(
            carrier,
            token_input=token_input,
            message_producers=producers,
            execution_nonce=execution_nonce,
        )
        validate_carrier_transport_execution_receipt(
            carrier_execution.receipt,
            manifest=carrier,
        )
        terminal_invocation = invocations[37]
        normalized = self.terminal_runtime.normalize(carrier_execution.final_carrier)
        receipts[37] = self._receipt(
            terminal_invocation,
            provider_kind="exact_terminal",
            output=normalized,
            routing=None,
        )
        egress_invocation = invocations[38]
        logits = self.lexical_atom.egress(normalized)
        receipts[38] = self._receipt(
            egress_invocation,
            provider_kind="exact_tied_lexical",
            output=logits,
            routing=None,
        )
        if set(receipts) != set(range(39)):
            raise RuntimeError("compressed path did not execute all invocations")
        after = self.state_sha256
        receipt = MLPSharedFamilyExecutionReceipt(
            runtime_sha256=self.runtime_sha256,
            execution_nonce=execution_nonce,
            carrier_execution_receipt=carrier_execution.receipt,
            invocation_receipts=tuple(receipts[index] for index in range(39)),
            candidate_state_before_sha256=before,
            candidate_state_after_sha256=after,
            final_carrier_sha256=tensor_sha256(carrier_execution.final_carrier),
            normalized_output_sha256=tensor_sha256(normalized),
            logits_sha256=tensor_sha256(logits),
            physical_stored_scalar_count=self.deployed_scalar_count,
            compression_attempted=(
                self.artifact_kind != DENSE_EXACT_CONTROL
            ),
        )
        return MLPSharedFamilyExecution(
            output=carrier_execution.final_carrier,
            normalized_output=normalized,
            logits=logits,
            carrier_execution=carrier_execution,
            receipt=receipt,
        )

    forward_components = execute


def _point_id(artifact_kind: str, residual_rank: int | None) -> str:
    if artifact_kind == DENSE_EXACT_CONTROL:
        return "shared_core.residual_dense"
    if artifact_kind == ZERO_RESIDUAL_SHARED_CORE:
        return "ablation.shared_core.zero_residual"
    assert residual_rank is not None
    return f"shared_core.residual_{residual_rank}"


@dataclass(slots=True)
class Gemma3MLPSharedFamilyCompilation:
    pair_manifest: MLPFamilyPairManifest
    topology_manifest: HeterogeneousGeneratorManifest
    executor: Gemma3MLPSharedFamilyExecutor
    artifact_kind: str
    residual_rank: int | None
    assignment_mode: str
    shuffle_seed: int | None
    exact_adapted_atom_stored_scalars: int
    shared_core_stored_scalars: int
    source_independence_audit: Mapping[str, object]
    fit_factor_catalog_sha256: str | None

    def __post_init__(self) -> None:
        self.topology_manifest = validate_heterogeneous_generator_manifest(
            self.topology_manifest
        )
        if not isinstance(self.pair_manifest, MLPFamilyPairManifest):
            raise TypeError("compilation pair manifest is invalid")
        if not isinstance(self.executor, Gemma3MLPSharedFamilyExecutor):
            raise TypeError("compilation executor is invalid")
        if self.executor.topology_manifest != self.topology_manifest:
            raise ValueError("executor topology drifted")
        if self.artifact_kind not in ARTIFACT_KINDS:
            raise ValueError("compilation artifact kind is invalid")
        if self.assignment_mode not in ASSIGNMENT_MODES:
            raise ValueError("compilation assignment mode is invalid")
        if self.artifact_kind in {
            SHARED_CORE_LOW_RANK,
            INDEPENDENT_LOW_RANK_NONSHARING,
        }:
            _require_int(self.residual_rank, label="residual_rank", minimum=1)
            _require_sha256(
                self.fit_factor_catalog_sha256,
                label="fit_factor_catalog_sha256",
            )
        elif self.residual_rank is not None:
            raise ValueError("dense/zero points cannot declare a residual rank")
        for label, value in (
            ("exact_adapted_atom_stored_scalars", self.exact_adapted_atom_stored_scalars),
            ("shared_core_stored_scalars", self.shared_core_stored_scalars),
        ):
            _require_int(value, label=label, minimum=1)
        if (
            not isinstance(self.source_independence_audit, Mapping)
            or self.source_independence_audit.get("passed") is not True
            or self.source_independence_audit.get(
                "candidate_predecessor_storage_overlap_count"
            )
            != 0
        ):
            raise ValueError("source-independence audit did not pass")
        expected = (
            self.topology_manifest.source_stored_scalar_count
            - self.exact_adapted_atom_stored_scalars
            + self.adapter_stored_scalars
        )
        if self.executor.deployed_scalar_count != expected:
            raise ValueError("compressed physical scalar inventory did not close")

    @property
    def adapter_stored_scalars(self) -> int:
        return self.executor.adapter.stored_scalar_count

    @property
    def candidate_stored_scalars(self) -> int:
        return self.executor.deployed_scalar_count

    @property
    def removed_scalars(self) -> int:
        return self.topology_manifest.source_stored_scalar_count - self.candidate_stored_scalars

    @property
    def point_id(self) -> str:
        return _point_id(self.artifact_kind, self.residual_rank)

    @property
    def arm_id(self) -> str:
        if self.artifact_kind == DENSE_EXACT_CONTROL:
            return "exact.shared_core_dense_residual"
        if self.artifact_kind == ZERO_RESIDUAL_SHARED_CORE:
            return "ablation.shared_core_zero_residual"
        assert self.residual_rank is not None
        if self.artifact_kind == INDEPENDENT_LOW_RANK_NONSHARING:
            return f"control.independent_member_r{self.residual_rank}"
        if self.assignment_mode != FITTED_ASSIGNMENT:
            return f"control.shuffled_residual_r{self.residual_rank}"
        return f"primary.shared_core_residual_r{self.residual_rank}"

    @property
    def control_only(self) -> bool:
        return (
            self.artifact_kind
            in {
                ZERO_RESIDUAL_SHARED_CORE,
                INDEPENDENT_LOW_RANK_NONSHARING,
            }
            or self.assignment_mode != FITTED_ASSIGNMENT
        )

    def assemble(self) -> Gemma3MLPSharedFamilyExecutor:
        return self.executor

    def metadata(self) -> dict[str, object]:
        invocation_count = len(self.topology_manifest.invocations)
        occurrence_count = len(self.topology_manifest.edge_occurrences)
        correction_count = self.topology_manifest.carrier_manifest.correction_count
        payload = {
            "arm_id": self.arm_id,
            "point_id": self.point_id,
            "pair_manifest": self.pair_manifest.to_dict(),
            "pair_manifest_sha256": self.pair_manifest.fingerprint_sha256,
            "exact_plan_protocol_sha256": (
                self.pair_manifest.exact_plan_protocol_sha256
            ),
            "successor_setup_sha256": self.pair_manifest.successor_setup_sha256,
            "exact_plan_sha256": self.pair_manifest.exact_plan_sha256,
            "exact_topology_manifest_sha256": (
                self.topology_manifest.fingerprint_sha256
            ),
            "carrier_manifest_sha256": (
                self.topology_manifest.carrier_manifest.fingerprint_sha256
            ),
            "source_model_fingerprint": (
                self.pair_manifest.source_model_fingerprint
            ),
            "artifact_kind": self.artifact_kind,
            "residual_rank": self.residual_rank,
            "assignment_mode": self.assignment_mode,
            "shuffle_seed": self.shuffle_seed,
            "fit_factor_catalog_sha256": self.fit_factor_catalog_sha256,
            "core_invocation_id": self.executor.core_invocation_id,
            "adapted_invocation_id": self.executor.adapted_invocation_id,
            "invocation_count": invocation_count,
            "edge_occurrence_count": occurrence_count,
            "carrier_boundary_count": len(
                self.topology_manifest.carrier_manifest.events
            ),
            "identity_carry_count": 36,
            "learned_cross_event_correction_count": correction_count,
            "source_stored_scalars": self.topology_manifest.source_stored_scalar_count,
            "exact_adapted_atom_stored_scalars": (
                self.exact_adapted_atom_stored_scalars
            ),
            "shared_core_stored_scalars": self.shared_core_stored_scalars,
            "shared_core_counted_once": True,
            "adapter_stored_scalars": self.adapter_stored_scalars,
            "adapter_state_sha256": module_state_sha256(self.executor.adapter),
            "candidate_stored_scalars": self.candidate_stored_scalars,
            "removed_scalars": self.removed_scalars,
            "whole_model_reduction_fraction": (
                self.removed_scalars
                / self.topology_manifest.source_stored_scalar_count
            ),
            "candidate_state_sha256": self.executor.state_sha256,
            "runtime_sha256": self.executor.runtime_sha256,
            "source_independence_audit": dict(self.source_independence_audit),
            "topology_preserved": (
                invocation_count == 39
                and occurrence_count == 75
                and len(self.topology_manifest.carrier_manifest.events) == 36
                and correction_count == 0
            ),
            "tied_lexical_remains_exact": True,
            "all_attention_remains_exact": True,
            "unselected_MLP_invocations_remain_exact": True,
            "terminal_remains_exact": True,
            "fused_dense_projection_buffer_present": False,
            "source_free_candidate_runtime": True,
            "control_only": self.control_only,
            "compression_attempted": self.artifact_kind != DENSE_EXACT_CONTROL,
            "compression_success_claimed": False,
        }
        payload["point_manifest_sha256"] = _fingerprint(_POINT_DOMAIN, payload)
        return payload

    def export_artifact(self) -> "MLPSharedFamilyPointArtifact":
        return MLPSharedFamilyPointArtifact.from_compilation(self)

    def restore_artifact_(
        self,
        artifact: "MLPSharedFamilyPointArtifact",
    ) -> None:
        if not isinstance(artifact, MLPSharedFamilyPointArtifact):
            raise TypeError("artifact has the wrong type")
        if artifact.point_manifest != self.metadata():
            raise ValueError("artifact belongs to another compiled point")
        self.executor.load_state_dict(artifact.tensor_state, strict=True)
        if self.executor.state_sha256 != artifact.point_manifest[
            "candidate_state_sha256"
        ]:
            raise RuntimeError("restored candidate state fingerprint drifted")


def _factor_catalog_sha256(
    factors: Mapping[str, tuple[Tensor, Tensor]],
) -> str:
    rows = []
    for name in sorted(factors):
        left, right = factors[name]
        rows.append(
            {
                "projection": name,
                "left_sha256": tensor_sha256(left),
                "right_sha256": tensor_sha256(right),
            }
        )
    return _fingerprint(_POINT_DOMAIN, rows)


def compile_gemma3_mlp_shared_family_residual_adapter(
    exact_compilation: Gemma3HeterogeneousGeneratorCompilation,
    *,
    pair_manifest: MLPFamilyPairManifest,
    artifact_kind: str,
    residual_rank: int | None = None,
    projection_factors: Mapping[str, tuple[Tensor, Tensor]] | None = None,
    assignment_mode: str = FITTED_ASSIGNMENT,
    shuffle_seed: int | None = None,
) -> Gemma3MLPSharedFamilyCompilation:
    """Compile one candidate-owned pair point from the exact predecessor."""

    if not isinstance(
        exact_compilation,
        Gemma3HeterogeneousGeneratorCompilation,
    ):
        raise TypeError("exact_compilation has the wrong type")
    pair = validate_mlp_family_pair_binding(
        pair_manifest,
        exact_compilation=exact_compilation,
    )
    if artifact_kind not in ARTIFACT_KINDS:
        raise ValueError("artifact_kind is invalid")
    if artifact_kind in {
        SHARED_CORE_LOW_RANK,
        INDEPENDENT_LOW_RANK_NONSHARING,
    }:
        _require_int(residual_rank, label="residual_rank", minimum=1)
        if projection_factors is None:
            raise ValueError("low-rank compilation requires fitted factors")
    else:
        if residual_rank is not None or projection_factors is not None:
            raise ValueError("dense/zero compilation cannot receive factors")
    if artifact_kind != SHARED_CORE_LOW_RANK and assignment_mode != FITTED_ASSIGNMENT:
        raise ValueError("only shared residuals support assignment controls")
    core_id, adapted_id, resolved_seed = _assignment(
        pair,
        mode=assignment_mode,
        shuffle_seed=shuffle_seed,
    )

    source_executor = exact_compilation.assemble(LIFT_SCOPE_FULL_HETEROGENEOUS)
    topology = source_executor.manifest
    by_id = {item.invocation_id: item for item in topology.invocations}
    source_atoms: dict[str, Any] = {}
    for invocation in topology.invocations:
        if invocation.event_kind == "terminal":
            continue
        source_atoms[invocation.invocation_id] = source_executor.atom_for(invocation)
    anchor_source = source_atoms[pair.anchor_invocation_id]
    member_source = source_atoms[pair.member_invocation_id]
    core_source = source_atoms[core_id]
    adapted_source = source_atoms[adapted_id]
    if not all(
        isinstance(item, MLPAtomRuntime)
        for item in (anchor_source, member_source, core_source, adapted_source)
    ):
        raise TypeError("selected pair did not resolve to MLP runtimes")
    assert isinstance(anchor_source, MLPAtomRuntime)
    assert isinstance(member_source, MLPAtomRuntime)
    assert isinstance(core_source, MLPAtomRuntime)
    assert isinstance(adapted_source, MLPAtomRuntime)
    anchor_components = _extract_mlp_components(anchor_source)
    member_components = _extract_mlp_components(member_source)
    if (
        anchor_components.gate.weight.shape != member_components.gate.weight.shape
        or anchor_components.up.weight.shape != member_components.up.weight.shape
        or anchor_components.down.weight.shape != member_components.down.weight.shape
    ):
        raise ValueError("selected MLP family members are shape-incompatible")

    lexical_source = source_atoms[topology.invocations[0].invocation_id]
    if not isinstance(lexical_source, LexicalAtomRuntime):
        raise TypeError("exact topology lost its tied lexical atom")
    lexical = copy.deepcopy(lexical_source)
    attention: dict[str, AttentionAtomRuntime] = {}
    exact_mlps: dict[str, MLPAtomRuntime] = {}
    for invocation in topology.invocations[1:37]:
        atom = source_atoms[invocation.invocation_id]
        if invocation.event_kind == "attention":
            if not isinstance(atom, AttentionAtomRuntime):
                raise TypeError("attention invocation resolved to wrong atom")
            attention[invocation.invocation_id] = copy.deepcopy(atom)
        elif invocation.invocation_id not in {
            pair.anchor_invocation_id,
            pair.member_invocation_id,
        }:
            if not isinstance(atom, MLPAtomRuntime):
                raise TypeError("MLP invocation resolved to wrong atom")
            exact_mlps[invocation.invocation_id] = copy.deepcopy(atom)
    terminal = copy.deepcopy(source_executor.terminal_runtime)

    factor_sha256: str | None = None
    if artifact_kind == DENSE_EXACT_CONTROL:
        adapter: AdapterRuntime = DenseExactControlMLPRuntime(
            copy.deepcopy(adapted_source)
        )
    else:
        # Adapter coefficients and normalization semantics are frozen by the
        # fitted member, even when a no-refit assignment control applies them
        # to the other temporal invocation.
        if artifact_kind == ZERO_RESIDUAL_SHARED_CORE:
            adapter = SharedCoreResidualAdapterMLPRuntime(
                input_norm_delta=None,
                output_norm_delta=None,
                activation=member_components.activation,
                gate_delta=None,
                up_delta=None,
                down_delta=None,
            )
        else:
            assert residual_rank is not None
            assert projection_factors is not None
            factor_sha256 = _factor_catalog_sha256(projection_factors)
            deltas = _build_projection_deltas(
                projection_factors,
                core_atom=core_source,
                rank=residual_rank,
            )
            if artifact_kind == SHARED_CORE_LOW_RANK:
                fitted_anchor_components = _extract_mlp_components(anchor_source)
                adapter = SharedCoreResidualAdapterMLPRuntime(
                    input_norm_delta=_norm_weight_delta(
                        member_components.input_norm,
                        fitted_anchor_components.input_norm,
                    ),
                    output_norm_delta=_norm_weight_delta(
                        member_components.output_norm,
                        fitted_anchor_components.output_norm,
                    ),
                    activation=member_components.activation,
                    gate_delta=deltas["gate_proj"],
                    up_delta=deltas["up_proj"],
                    down_delta=deltas["down_proj"],
                )
            else:
                adapter = IndependentLowRankMLPRuntime(
                    input_norm=copy.deepcopy(member_components.input_norm),
                    output_norm=copy.deepcopy(member_components.output_norm),
                    activation=member_components.activation,
                    gate=deltas["gate_proj"],
                    up=deltas["up_proj"],
                    down=deltas["down_proj"],
                )

    candidate = Gemma3MLPSharedFamilyExecutor(
        topology_manifest=topology,
        lexical_atom=lexical,
        attention_atoms=attention,
        exact_mlp_atoms=exact_mlps,
        shared_core_atom=copy.deepcopy(anchor_source),
        terminal_runtime=terminal,
        adapter=adapter,
        pair_invocation_ids=(
            pair.anchor_invocation_id,
            pair.member_invocation_id,
        ),
        core_invocation_id=core_id,
        adapted_invocation_id=adapted_id,
        artifact_kind=artifact_kind,
        assignment_mode=assignment_mode,
    )
    source_modules = (
        *(source_executor.atom_banks),
        source_executor.terminal_runtime,
    )
    _assert_source_storage_disjoint(candidate, source_modules)
    exact_adapted_count = module_physical_scalar_count(member_source)
    shared_core_count = module_physical_scalar_count(anchor_source)
    result = Gemma3MLPSharedFamilyCompilation(
        pair_manifest=pair,
        topology_manifest=topology,
        executor=candidate,
        artifact_kind=artifact_kind,
        residual_rank=residual_rank,
        assignment_mode=assignment_mode,
        shuffle_seed=resolved_seed,
        exact_adapted_atom_stored_scalars=exact_adapted_count,
        shared_core_stored_scalars=shared_core_count,
        source_independence_audit={
            "passed": True,
            "candidate_predecessor_storage_overlap_count": 0,
            "candidate_owns_exact_unmodified_components": True,
            "live_source_model_reference_present": False,
        },
        fit_factor_catalog_sha256=factor_sha256,
    )
    if topology.source_stored_scalar_count == PINNED_SOURCE_SCALARS:
        if (
            pair.successor_setup_sha256 != PINNED_SUCCESSOR_SETUP_SHA256
            or topology.hidden_width != PINNED_HIDDEN_WIDTH
            or shared_core_count != _PINNED_MLP_INVOCATION_SCALARS
            or exact_adapted_count != _PINNED_MLP_INVOCATION_SCALARS
        ):
            raise RuntimeError("pinned MLP resource profile drifted")
        if artifact_kind == SHARED_CORE_LOW_RANK:
            assert residual_rank is not None
            if residual_rank not in PRIMARY_RESIDUAL_RANKS:
                raise ValueError("pinned primary rank is not preregistered")
            expected = pinned_mlp_pair_resource_point(residual_rank)
            if result.candidate_stored_scalars != expected["candidate_stored_scalars"]:
                raise RuntimeError("pinned low-rank resource point did not close")
        elif artifact_kind == DENSE_EXACT_CONTROL:
            expected = pinned_mlp_pair_resource_point(None)
            if result.candidate_stored_scalars != expected["candidate_stored_scalars"]:
                raise RuntimeError("pinned dense control did not close")
    return result


def projection_factors_from_compilation(
    compilation: Gemma3MLPSharedFamilyCompilation,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Clone already-fitted factors for deterministic no-refit controls."""

    if not isinstance(compilation, Gemma3MLPSharedFamilyCompilation):
        raise TypeError("compilation has the wrong type")
    adapter = compilation.executor.adapter
    callback = getattr(adapter, "projection_factors", None)
    if not callable(callback):
        raise ValueError("compiled point has no reusable low-rank factors")
    factors = callback()
    if not factors:
        raise ValueError("compiled point has no reusable low-rank factors")
    if _factor_catalog_sha256(factors) != compilation.fit_factor_catalog_sha256:
        raise RuntimeError("compiled factors changed after fitting")
    return factors


def compile_no_refit_residual_assignment_control(
    exact_compilation: Gemma3HeterogeneousGeneratorCompilation,
    *,
    fitted_compilation: Gemma3MLPSharedFamilyCompilation,
    assignment_mode: str,
    shuffle_seed: int | None = None,
) -> Gemma3MLPSharedFamilyCompilation:
    """Apply frozen factors under a swapped/shuffled invocation assignment."""

    if fitted_compilation.artifact_kind != SHARED_CORE_LOW_RANK:
        raise ValueError("no-refit assignment controls require a shared residual")
    if assignment_mode not in {SWAPPED_ASSIGNMENT, SHUFFLED_ASSIGNMENT}:
        raise ValueError("control assignment must be swapped or shuffled")
    factors = projection_factors_from_compilation(fitted_compilation)
    result = compile_gemma3_mlp_shared_family_residual_adapter(
        exact_compilation,
        pair_manifest=fitted_compilation.pair_manifest,
        artifact_kind=SHARED_CORE_LOW_RANK,
        residual_rank=fitted_compilation.residual_rank,
        projection_factors=factors,
        assignment_mode=assignment_mode,
        shuffle_seed=shuffle_seed,
    )
    if result.fit_factor_catalog_sha256 != fitted_compilation.fit_factor_catalog_sha256:
        raise RuntimeError("no-refit control changed the fitted factor catalog")
    if module_state_sha256(result.executor.adapter) != module_state_sha256(
        fitted_compilation.executor.adapter
    ):
        raise RuntimeError("no-refit control changed the frozen adapter state")
    return result


def _state_alias_groups(state: Mapping[str, Tensor]) -> tuple[tuple[str, ...], ...]:
    groups: dict[tuple[str, int], list[str]] = {}
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError("serialized candidate state must contain only tensors")
        key = (str(value.device), value.untyped_storage().data_ptr())
        groups.setdefault(key, []).append(name)
    return tuple(
        sorted(
            (tuple(sorted(names)) for names in groups.values() if len(names) > 1),
            key=lambda row: row[0],
        )
    )


def _state_stored_scalar_count(state: Mapping[str, Tensor]) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for value in state.values():
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += value.numel()
    return total


def _state_tensor_storage_bytes(state: Mapping[str, Tensor]) -> int:
    seen: set[tuple[str, int]] = set()
    total = 0
    for value in state.values():
        if not isinstance(value, Tensor):
            raise TypeError("serialized candidate state must contain only tensors")
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def _tensor_catalog_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256(_TENSOR_CATALOG_DOMAIN)
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise TypeError("candidate tensor catalog is invalid")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _serialized_module_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Reproduce the runtime module-state digest from serialized tensors."""

    digest = hashlib.sha256(_HETEROGENEOUS_MODULE_DOMAIN)
    seen: set[tuple[str, int]] = set()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise TypeError("serialized module state is invalid")
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key in seen:
            continue
        seen.add(key)
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


def _clone_state_to_cpu_preserving_aliases(
    state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    clones: dict[tuple[str, int], Tensor] = {}
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key not in clones:
            clones[key] = value.detach().to(device="cpu").clone().contiguous()
        clone = clones[key]
        if clone.shape != value.shape or clone.dtype != value.dtype:
            raise ValueError("unsupported aliased state views in candidate")
        result[name] = clone
    return result


@dataclass(slots=True)
class MLPSharedFamilyPointArtifact:
    """Strict source-free tensor envelope for one independently fitted point."""

    point_manifest: Mapping[str, object]
    tensor_state: Mapping[str, Tensor]
    tensor_state_sha256: str
    ordered_state_names: tuple[str, ...]
    storage_alias_groups: tuple[tuple[str, ...], ...]
    serialized_stored_scalar_count: int
    schema: str = MLP_SHARED_FAMILY_ARTIFACT_SCHEMA
    format_version: int = MLP_SHARED_FAMILY_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.schema != MLP_SHARED_FAMILY_ARTIFACT_SCHEMA:
            raise ValueError("shared-family artifact schema is unsupported")
        if self.format_version != MLP_SHARED_FAMILY_FORMAT_VERSION:
            raise ValueError("shared-family artifact version is unsupported")
        if not isinstance(self.point_manifest, Mapping):
            raise TypeError("point_manifest must be a mapping")
        required_point_fields = {
            "arm_id",
            "point_id",
            "pair_manifest",
            "pair_manifest_sha256",
            "artifact_kind",
            "candidate_stored_scalars",
            "candidate_state_sha256",
            "topology_preserved",
            "source_free_candidate_runtime",
            "fused_dense_projection_buffer_present",
            "point_manifest_sha256",
        }
        if not required_point_fields <= set(self.point_manifest):
            raise ValueError("point manifest omitted required fields")
        point_payload = dict(self.point_manifest)
        point_sha = point_payload.pop("point_manifest_sha256")
        _require_sha256(point_sha, label="point_manifest_sha256")
        if _fingerprint(_POINT_DOMAIN, point_payload) != point_sha:
            raise ValueError("point manifest authentication failed")
        pair = MLPFamilyPairManifest.from_dict(self.point_manifest["pair_manifest"])
        artifact_kind = self.point_manifest.get("artifact_kind")
        residual_rank = self.point_manifest.get("residual_rank")
        assignment_mode = self.point_manifest.get("assignment_mode")
        source_scalars = self.point_manifest.get("source_stored_scalars")
        exact_removed = self.point_manifest.get(
            "exact_adapted_atom_stored_scalars"
        )
        adapter_scalars = self.point_manifest.get("adapter_stored_scalars")
        candidate_scalars = self.point_manifest.get("candidate_stored_scalars")
        removed_scalars = self.point_manifest.get("removed_scalars")
        if (
            self.point_manifest.get("pair_manifest_sha256")
            != pair.fingerprint_sha256
            or self.point_manifest.get("exact_plan_protocol_sha256")
            != pair.exact_plan_protocol_sha256
            or self.point_manifest.get("successor_setup_sha256")
            != pair.successor_setup_sha256
            or self.point_manifest.get("exact_plan_sha256")
            != pair.exact_plan_sha256
            or self.point_manifest.get("source_model_fingerprint")
            != pair.source_model_fingerprint
            or artifact_kind not in ARTIFACT_KINDS
            or assignment_mode not in ASSIGNMENT_MODES
            or self.point_manifest.get("point_id")
            != _point_id(artifact_kind, residual_rank)
            or self.point_manifest.get("invocation_count") != 39
            or self.point_manifest.get("edge_occurrence_count") != 75
            or self.point_manifest.get("carrier_boundary_count") != 36
            or self.point_manifest.get("identity_carry_count") != 36
            or self.point_manifest.get("learned_cross_event_correction_count")
            != 0
            or type(source_scalars) is not int
            or type(exact_removed) is not int
            or type(adapter_scalars) is not int
            or type(candidate_scalars) is not int
            or candidate_scalars != source_scalars - exact_removed + adapter_scalars
            or type(removed_scalars) is not int
            or removed_scalars != source_scalars - candidate_scalars
            or self.point_manifest.get("whole_model_reduction_fraction")
            != removed_scalars / source_scalars
            or self.point_manifest.get("shared_core_counted_once") is not True
            or self.point_manifest.get("tied_lexical_remains_exact") is not True
            or self.point_manifest.get("all_attention_remains_exact") is not True
            or self.point_manifest.get("terminal_remains_exact") is not True
        ):
            raise ValueError("serialized point semantics drifted")
        for label in (
            "exact_topology_manifest_sha256",
            "carrier_manifest_sha256",
            "candidate_state_sha256",
            "adapter_state_sha256",
            "runtime_sha256",
        ):
            _require_sha256(self.point_manifest.get(label), label=label)
        if (
            self.point_manifest.get("topology_preserved") is not True
            or self.point_manifest.get("source_free_candidate_runtime") is not True
            or self.point_manifest.get("fused_dense_projection_buffer_present")
            is not False
        ):
            raise ValueError("serialized point crossed its claim firewall")
        if not isinstance(self.tensor_state, Mapping) or not self.tensor_state:
            raise ValueError("serialized candidate tensor state is empty")
        if tuple(sorted(self.tensor_state)) != self.ordered_state_names:
            raise ValueError("serialized candidate state-name order drifted")
        for name, tensor in self.tensor_state.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(tensor, Tensor)
                or tensor.device.type != "cpu"
                or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
            ):
                raise ValueError("serialized candidate tensor is invalid")
        _require_sha256(self.tensor_state_sha256, label="tensor_state_sha256")
        if _tensor_catalog_sha256(self.tensor_state) != self.tensor_state_sha256:
            raise ValueError("serialized candidate tensor authentication failed")
        if (
            _serialized_module_state_sha256(self.tensor_state)
            != self.point_manifest.get("candidate_state_sha256")
        ):
            raise ValueError("serialized state and candidate-state receipt differ")
        if _state_alias_groups(self.tensor_state) != self.storage_alias_groups:
            raise ValueError("serialized candidate storage aliases drifted")
        if (
            _state_stored_scalar_count(self.tensor_state)
            != self.serialized_stored_scalar_count
            or self.serialized_stored_scalar_count
            != self.point_manifest.get("candidate_stored_scalars")
        ):
            raise ValueError("serialized candidate storage accounting drifted")

    @classmethod
    def from_compilation(
        cls,
        compilation: Gemma3MLPSharedFamilyCompilation,
    ) -> "MLPSharedFamilyPointArtifact":
        if not isinstance(compilation, Gemma3MLPSharedFamilyCompilation):
            raise TypeError("compilation has the wrong type")
        state = _clone_state_to_cpu_preserving_aliases(
            compilation.executor.state_dict(keep_vars=True)
        )
        return cls(
            point_manifest=copy.deepcopy(compilation.metadata()),
            tensor_state=state,
            tensor_state_sha256=_tensor_catalog_sha256(state),
            ordered_state_names=tuple(sorted(state)),
            storage_alias_groups=_state_alias_groups(state),
            serialized_stored_scalar_count=_state_stored_scalar_count(state),
        )

    def metadata_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "point_manifest": copy.deepcopy(dict(self.point_manifest)),
            "tensor_state_sha256": self.tensor_state_sha256,
            "ordered_state_names": list(self.ordered_state_names),
            "storage_alias_groups": [list(group) for group in self.storage_alias_groups],
            "serialized_stored_scalar_count": self.serialized_stored_scalar_count,
            "contains_source_model": False,
            "source_free_serialized_candidate": True,
        }

    @property
    def artifact_sha256(self) -> str:
        return _fingerprint(_ARTIFACT_DOMAIN, self.metadata_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            **self.metadata_payload(),
            "tensor_state": dict(self.tensor_state),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_payload(
        cls,
        value: Mapping[str, object],
        *,
        expected_artifact_sha256: str | None = None,
    ) -> "MLPSharedFamilyPointArtifact":
        expected = {
            "schema",
            "format_version",
            "point_manifest",
            "tensor_state_sha256",
            "ordered_state_names",
            "storage_alias_groups",
            "serialized_stored_scalar_count",
            "contains_source_model",
            "source_free_serialized_candidate",
            "tensor_state",
            "artifact_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("shared-family artifact fields are not exact")
        if (
            value["contains_source_model"] is not False
            or value["source_free_serialized_candidate"] is not True
        ):
            raise ValueError("artifact retained a source-model claim")
        state = value["tensor_state"]
        names = value["ordered_state_names"]
        aliases = value["storage_alias_groups"]
        if not isinstance(state, Mapping):
            raise TypeError("artifact tensor_state must be a mapping")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise TypeError("artifact ordered_state_names must be a sequence")
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise TypeError("artifact storage_alias_groups must be a sequence")
        result = cls(
            point_manifest=value["point_manifest"],
            tensor_state={str(name): tensor for name, tensor in state.items()},
            tensor_state_sha256=value["tensor_state_sha256"],
            ordered_state_names=tuple(names),
            storage_alias_groups=tuple(tuple(group) for group in aliases),
            serialized_stored_scalar_count=value["serialized_stored_scalar_count"],
            schema=value["schema"],
            format_version=value["format_version"],
        )
        observed = _require_sha256(value["artifact_sha256"], label="artifact_sha256")
        if observed != result.artifact_sha256:
            raise ValueError("shared-family artifact authentication failed")
        if expected_artifact_sha256 is not None and observed != _require_sha256(
            expected_artifact_sha256,
            label="expected_artifact_sha256",
        ):
            raise ValueError("artifact does not match the frozen frontier receipt")
        return result


def validate_mlp_shared_family_point_artifact(
    value: MLPSharedFamilyPointArtifact | Mapping[str, object],
    *,
    expected_artifact_sha256: str | None = None,
) -> MLPSharedFamilyPointArtifact:
    if isinstance(value, MLPSharedFamilyPointArtifact):
        return MLPSharedFamilyPointArtifact.from_payload(
            value.to_payload(),
            expected_artifact_sha256=expected_artifact_sha256,
        )
    return MLPSharedFamilyPointArtifact.from_payload(
        value,
        expected_artifact_sha256=expected_artifact_sha256,
    )


def write_mlp_shared_family_point_artifact(
    artifact: MLPSharedFamilyPointArtifact,
    destination: Path | str,
) -> MLPSharedFamilyPointArtifact:
    validated = validate_mlp_shared_family_point_artifact(artifact)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            torch.save(validated.to_payload(), handle)
    except FileExistsError as error:
        raise FileExistsError("refusing to overwrite a shared-family artifact") from error
    return load_mlp_shared_family_point_artifact(
        path,
        expected_artifact_sha256=validated.artifact_sha256,
    )


def load_mlp_shared_family_point_artifact(
    source: Path | str,
    *,
    expected_artifact_sha256: str | None = None,
) -> MLPSharedFamilyPointArtifact:
    value = torch.load(Path(source), map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError("serialized shared-family artifact is not a mapping")
    return validate_mlp_shared_family_point_artifact(
        value,
        expected_artifact_sha256=expected_artifact_sha256,
    )


def _sha256_path_streaming(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_content_store_path(store_root: Path, path: Path) -> None:
    root = store_root.absolute()
    candidate = path.absolute()
    if not candidate.is_relative_to(root):
        raise ValueError("content-addressed path escapes its store root")
    if root.is_symlink():
        raise ValueError("content-addressed store root cannot be a symlink")
    current = root
    if current.is_symlink():
        raise ValueError("content-addressed path traverses a symlink")
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("content-addressed path traverses a symlink")
    resolved_root = root.resolve(strict=False)
    if not candidate.resolve(strict=False).is_relative_to(resolved_root):
        raise ValueError("content-addressed path resolves outside its store root")


def _atomic_write_exclusive(
    path: Path,
    *,
    store_root: Path,
    write_payload: Any,
    validate_temporary: Any | None = None,
) -> bool:
    """Publish a complete file without overwriting an existing store entry."""

    _assert_content_store_path(store_root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_content_store_path(store_root, path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write_payload(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if validate_temporary is not None:
            validate_temporary(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def initialize_mlp_shared_family_content_addressed_store(
    store_root: Path | str,
) -> Path:
    """Atomically claim a new store root before live model work begins."""

    root = Path(store_root)
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            "refusing to reuse a shared-family content-addressed store"
        ) from error
    _assert_content_store_path(root, root)
    return root


def _content_blob_metadata(
    state: Mapping[str, Tensor],
    *,
    blob_kind: str,
) -> dict[str, object]:
    if blob_kind not in {
        "shared_base",
        "dense_exact_member",
        "primary_adapter",
        "independent_adapter",
    }:
        raise ValueError("content-addressed tensor blob kind is invalid")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("content-addressed tensor blob cannot be empty")
    aliases = _state_alias_groups(state)
    return {
        "schema": MLP_SHARED_FAMILY_CONTENT_BLOB_SCHEMA,
        "format_version": MLP_SHARED_FAMILY_FORMAT_VERSION,
        "blob_kind": blob_kind,
        "tensor_state_sha256": _tensor_catalog_sha256(state),
        "ordered_state_names": list(sorted(state)),
        "storage_alias_groups": [list(group) for group in aliases],
        "stored_scalar_count": _state_stored_scalar_count(state),
        "tensor_storage_bytes": _state_tensor_storage_bytes(state),
        "contains_source_model": False,
        "source_free_serialized_component": True,
    }


def _content_blob_payload(
    state: Mapping[str, Tensor],
    *,
    blob_kind: str,
) -> dict[str, object]:
    metadata = _content_blob_metadata(state, blob_kind=blob_kind)
    return {
        **metadata,
        "tensor_state": dict(state),
        "blob_sha256": _fingerprint(_CONTENT_BLOB_DOMAIN, metadata),
    }


def _validate_content_blob_payload(
    value: Mapping[str, object],
    *,
    expected_blob_sha256: str | None = None,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    expected = {
        "schema",
        "format_version",
        "blob_kind",
        "tensor_state_sha256",
        "ordered_state_names",
        "storage_alias_groups",
        "stored_scalar_count",
        "tensor_storage_bytes",
        "contains_source_model",
        "source_free_serialized_component",
        "tensor_state",
        "blob_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("content-addressed tensor blob fields are not exact")
    state_value = value.get("tensor_state")
    if not isinstance(state_value, Mapping):
        raise TypeError("content-addressed tensor state must be a mapping")
    state = {str(name): tensor for name, tensor in state_value.items()}
    if not state:
        raise ValueError("content-addressed tensor blob cannot be empty")
    for name, tensor in state.items():
        if (
            not name
            or not isinstance(tensor, Tensor)
            or tensor.device.type != "cpu"
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
        ):
            raise ValueError("content-addressed tensor blob is invalid")
    blob_kind = value.get("blob_kind")
    metadata = _content_blob_metadata(state, blob_kind=str(blob_kind))
    if any(value.get(name) != expected_value for name, expected_value in metadata.items()):
        raise ValueError("content-addressed tensor blob metadata drifted")
    observed = _require_sha256(value.get("blob_sha256"), label="blob_sha256")
    if _fingerprint(_CONTENT_BLOB_DOMAIN, metadata) != observed:
        raise ValueError("content-addressed tensor blob authentication failed")
    if expected_blob_sha256 is not None and observed != _require_sha256(
        expected_blob_sha256,
        label="expected_blob_sha256",
    ):
        raise ValueError("content-addressed tensor blob does not match its reference")
    return state, {**metadata, "blob_sha256": observed}


def _load_content_blob(
    path: Path,
    *,
    expected_blob_sha256: str,
    store_root: Path | None = None,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    root = path.parent.parent if store_root is None else store_root
    _assert_content_store_path(root, path)
    if not path.is_file():
        raise ValueError("content-addressed tensor blob is not a regular file")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError("serialized content-addressed tensor blob is not a mapping")
    return _validate_content_blob_payload(
        value,
        expected_blob_sha256=expected_blob_sha256,
    )


def _write_content_blob(
    *,
    state: Mapping[str, Tensor],
    blob_kind: str,
    blob_root: Path,
) -> dict[str, object]:
    payload = _content_blob_payload(state, blob_kind=blob_kind)
    blob_sha = str(payload["blob_sha256"])
    path = blob_root / f"{blob_sha}.pt"
    blob_root.mkdir(parents=True, exist_ok=True)
    store_root = blob_root.parent
    created = _atomic_write_exclusive(
        path,
        store_root=store_root,
        write_payload=lambda handle: torch.save(payload, handle),
        validate_temporary=lambda temporary: _load_content_blob(
            temporary,
            expected_blob_sha256=blob_sha,
            store_root=store_root,
        ),
    )
    _loaded_state, metadata = _load_content_blob(
        path,
        expected_blob_sha256=blob_sha,
        store_root=store_root,
    )
    return {
        "blob_kind": blob_kind,
        "blob_sha256": blob_sha,
        "blob_file_basename": path.name,
        "blob_file_sha256": _sha256_path_streaming(path),
        "blob_file_bytes": path.stat().st_size,
        "tensor_state_sha256": metadata["tensor_state_sha256"],
        "ordered_state_names": metadata["ordered_state_names"],
        "stored_scalar_count": metadata["stored_scalar_count"],
        "tensor_storage_bytes": metadata["tensor_storage_bytes"],
        "newly_materialized": created,
    }


def _point_component_states(
    artifact: MLPSharedFamilyPointArtifact,
) -> tuple[tuple[str, dict[str, Tensor]], ...]:
    base = {
        name: tensor
        for name, tensor in artifact.tensor_state.items()
        if not name.startswith("adapter.")
    }
    adapter = {
        name: tensor
        for name, tensor in artifact.tensor_state.items()
        if name.startswith("adapter.")
    }
    if not base:
        raise ValueError("content-addressed point omitted its shared base")
    base_keys = {
        (str(tensor.device), tensor.untyped_storage().data_ptr())
        for tensor in base.values()
    }
    adapter_keys = {
        (str(tensor.device), tensor.untyped_storage().data_ptr())
        for tensor in adapter.values()
    }
    if base_keys & adapter_keys:
        raise ValueError("content-addressed components cross-alias storage")
    artifact_kind = artifact.point_manifest.get("artifact_kind")
    if artifact_kind == ZERO_RESIDUAL_SHARED_CORE:
        if adapter:
            raise ValueError("zero-residual point unexpectedly owns adapter tensors")
        return (("shared_base", base),)
    if not adapter:
        raise ValueError("materialized point omitted its adapter tensors")
    if artifact_kind == DENSE_EXACT_CONTROL:
        adapter_kind = "dense_exact_member"
    elif artifact_kind == INDEPENDENT_LOW_RANK_NONSHARING:
        adapter_kind = "independent_adapter"
    else:
        adapter_kind = "primary_adapter"
    return (("shared_base", base), (adapter_kind, adapter))


def content_addressed_mlp_shared_family_point_manifest_path(
    store_root: Path | str,
    *,
    arm_id: str,
) -> Path:
    _require_id(arm_id, label="arm_id")
    basename = arm_id.replace(".", "--")
    safe_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    if not basename or any(
        character not in safe_characters for character in basename
    ):
        raise ValueError("arm_id cannot be represented as a point-manifest name")
    return Path(store_root) / "points" / f"{basename}.json"


def _content_point_manifest_payload(
    artifact: MLPSharedFamilyPointArtifact,
    *,
    component_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    references = [
        {
            "blob_kind": receipt["blob_kind"],
            "blob_sha256": receipt["blob_sha256"],
            "blob_file_basename": receipt["blob_file_basename"],
            "tensor_state_sha256": receipt["tensor_state_sha256"],
            "ordered_state_names": list(receipt["ordered_state_names"]),
            "stored_scalar_count": receipt["stored_scalar_count"],
            "tensor_storage_bytes": receipt["tensor_storage_bytes"],
        }
        for receipt in component_receipts
    ]
    payload: dict[str, object] = {
        "schema": MLP_SHARED_FAMILY_CONTENT_POINT_SCHEMA,
        "format_version": MLP_SHARED_FAMILY_FORMAT_VERSION,
        "point_manifest": copy.deepcopy(dict(artifact.point_manifest)),
        "point_artifact_sha256": artifact.artifact_sha256,
        "tensor_state_sha256": artifact.tensor_state_sha256,
        "ordered_state_names": list(artifact.ordered_state_names),
        "storage_alias_groups": [
            list(group) for group in artifact.storage_alias_groups
        ],
        "serialized_stored_scalar_count": artifact.serialized_stored_scalar_count,
        "component_blobs": references,
        "contains_source_model": False,
        "source_free_serialized_candidate": True,
        "logical_point_materialized_separately": True,
    }
    payload["content_point_manifest_sha256"] = _fingerprint(
        _CONTENT_POINT_DOMAIN,
        payload,
    )
    return payload


def _validate_content_point_manifest(
    value: Mapping[str, object],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    expected = {
        "schema",
        "format_version",
        "point_manifest",
        "point_artifact_sha256",
        "tensor_state_sha256",
        "ordered_state_names",
        "storage_alias_groups",
        "serialized_stored_scalar_count",
        "component_blobs",
        "contains_source_model",
        "source_free_serialized_candidate",
        "logical_point_materialized_separately",
        "content_point_manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("content-addressed point-manifest fields are not exact")
    payload = dict(value)
    observed = _require_sha256(
        payload.pop("content_point_manifest_sha256"),
        label="content_point_manifest_sha256",
    )
    if _fingerprint(_CONTENT_POINT_DOMAIN, payload) != observed:
        raise ValueError("content-addressed point-manifest authentication failed")
    if expected_manifest_sha256 is not None and observed != _require_sha256(
        expected_manifest_sha256,
        label="expected_manifest_sha256",
    ):
        raise ValueError("content-addressed point manifest changed after freezing")
    if (
        value.get("schema") != MLP_SHARED_FAMILY_CONTENT_POINT_SCHEMA
        or value.get("format_version") != MLP_SHARED_FAMILY_FORMAT_VERSION
        or value.get("contains_source_model") is not False
        or value.get("source_free_serialized_candidate") is not True
        or value.get("logical_point_materialized_separately") is not True
    ):
        raise ValueError("content-addressed point-manifest claim drifted")
    result = copy.deepcopy(dict(value))
    result["content_point_manifest_sha256"] = observed
    return result


def write_mlp_shared_family_content_addressed_point_artifact(
    artifact: MLPSharedFamilyPointArtifact,
    store_root: Path | str,
) -> dict[str, object]:
    """Store one logical point while deduplicating authenticated tensor blobs."""

    validated = validate_mlp_shared_family_point_artifact(artifact)
    root = Path(store_root)
    if not root.exists():
        initialize_mlp_shared_family_content_addressed_store(root)
    _assert_content_store_path(root, root)
    manifest_path = content_addressed_mlp_shared_family_point_manifest_path(
        root,
        arm_id=str(validated.point_manifest["arm_id"]),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    component_receipts = [
        _write_content_blob(
            state=state,
            blob_kind=blob_kind,
            blob_root=root / "blobs",
        )
        for blob_kind, state in _point_component_states(validated)
    ]
    manifest = _content_point_manifest_payload(
        validated,
        component_receipts=component_receipts,
    )

    def validate_temporary_manifest(temporary: Path) -> None:
        raw = json.loads(temporary.read_text(encoding="utf-8"))
        _validate_content_point_manifest(
            raw,
            expected_manifest_sha256=str(
                manifest["content_point_manifest_sha256"]
            ),
        )

    created = _atomic_write_exclusive(
        manifest_path,
        store_root=root,
        write_payload=lambda handle: handle.write(
            _canonical_json_bytes(manifest)
        ),
        validate_temporary=validate_temporary_manifest,
    )
    if not created:
        raise FileExistsError(
            "refusing to overwrite a shared-family logical point manifest"
        )
    loaded = load_mlp_shared_family_content_addressed_point_artifact(
        manifest_path,
        expected_artifact_sha256=validated.artifact_sha256,
        expected_manifest_sha256=str(manifest["content_point_manifest_sha256"]),
    )
    if loaded.to_payload()["artifact_sha256"] != validated.artifact_sha256:
        raise RuntimeError("content-addressed point changed across strict reload")
    referenced_blob_bytes = sum(
        int(receipt["blob_file_bytes"]) for receipt in component_receipts
    )
    referenced_tensor_bytes = sum(
        int(receipt["tensor_storage_bytes"]) for receipt in component_receipts
    )
    new_blob_bytes = sum(
        int(receipt["blob_file_bytes"])
        for receipt in component_receipts
        if receipt["newly_materialized"] is True
    )
    new_tensor_bytes = sum(
        int(receipt["tensor_storage_bytes"])
        for receipt in component_receipts
        if receipt["newly_materialized"] is True
    )
    relative_manifest_path = manifest_path.relative_to(root)
    manifest_file_bytes = manifest_path.stat().st_size
    return {
        "path_basename": relative_manifest_path.as_posix(),
        "artifact_sha256": validated.artifact_sha256,
        "file_sha256": _sha256_path_streaming(manifest_path),
        "serialized_file_bytes": manifest_file_bytes,
        "content_point_manifest_sha256": manifest[
            "content_point_manifest_sha256"
        ],
        "point_manifest_sha256": validated.point_manifest[
            "point_manifest_sha256"
        ],
        "candidate_state_sha256": validated.point_manifest[
            "candidate_state_sha256"
        ],
        "fit_factor_catalog_sha256": validated.point_manifest[
            "fit_factor_catalog_sha256"
        ],
        "tensor_state_sha256": validated.tensor_state_sha256,
        "serialized_stored_scalar_count": (
            validated.serialized_stored_scalar_count
        ),
        "logical_tensor_storage_bytes": _state_tensor_storage_bytes(
            validated.tensor_state
        ),
        "ordered_state_names_sha256": _fingerprint(
            _CONTENT_POINT_DOMAIN,
            list(validated.ordered_state_names),
        ),
        "storage_alias_groups_sha256": _fingerprint(
            _CONTENT_POINT_DOMAIN,
            [list(group) for group in validated.storage_alias_groups],
        ),
        "storage_alias_group_count": len(validated.storage_alias_groups),
        "component_blob_receipts": component_receipts,
        "referenced_blob_count": len(component_receipts),
        "referenced_blob_file_bytes": referenced_blob_bytes,
        "referenced_tensor_storage_bytes": referenced_tensor_bytes,
        "newly_materialized_blob_count": sum(
            receipt["newly_materialized"] is True
            for receipt in component_receipts
        ),
        "newly_materialized_blob_file_bytes": new_blob_bytes,
        "newly_materialized_tensor_storage_bytes": new_tensor_bytes,
        "serialized_file_bytes_newly_materialized": (
            manifest_file_bytes + new_blob_bytes
        ),
        "content_addressed_store": True,
        "per_arm_logical_manifest_materialized": True,
        "source_free_candidate_runtime": True,
        "unique_scalar_inventory_and_physical_alias_receipt_present": True,
    }


def load_mlp_shared_family_content_addressed_point_artifact(
    source: Path | str,
    *,
    expected_artifact_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> MLPSharedFamilyPointArtifact:
    """Rehydrate a legacy-compatible point envelope from authenticated blobs."""

    path = Path(source)
    store_root = path.parent.parent
    _assert_content_store_path(store_root, path)
    if not path.is_file():
        raise ValueError("content-addressed point manifest is not a regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("content-addressed point manifest is not canonical JSON") from error
    manifest = _validate_content_point_manifest(
        raw,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    references = manifest.get("component_blobs")
    if not isinstance(references, list) or not references:
        raise ValueError("content-addressed point omitted component references")
    expected_reference_fields = {
        "blob_kind",
        "blob_sha256",
        "blob_file_basename",
        "tensor_state_sha256",
        "ordered_state_names",
        "stored_scalar_count",
        "tensor_storage_bytes",
    }
    merged: dict[str, Tensor] = {}
    observed_kinds: list[str] = []
    blob_root = path.parent.parent / "blobs"
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != expected_reference_fields:
            raise ValueError("content-addressed component reference fields are not exact")
        blob_sha = _require_sha256(
            reference.get("blob_sha256"),
            label="component_blob_sha256",
        )
        expected_basename = f"{blob_sha}.pt"
        if reference.get("blob_file_basename") != expected_basename:
            raise ValueError("content-addressed component filename drifted")
        state, metadata = _load_content_blob(
            blob_root / expected_basename,
            expected_blob_sha256=blob_sha,
            store_root=store_root,
        )
        for field in (
            "blob_kind",
            "tensor_state_sha256",
            "ordered_state_names",
            "stored_scalar_count",
            "tensor_storage_bytes",
        ):
            if reference.get(field) != metadata.get(field):
                raise ValueError("content-addressed component reference drifted")
        overlap = set(merged) & set(state)
        if overlap:
            raise ValueError("content-addressed component state names overlap")
        merged.update(state)
        observed_kinds.append(str(metadata["blob_kind"]))
    point_manifest = manifest.get("point_manifest")
    if not isinstance(point_manifest, Mapping):
        raise TypeError("content-addressed point metadata must be a mapping")
    artifact_kind = point_manifest.get("artifact_kind")
    expected_kinds = (
        ["shared_base"]
        if artifact_kind == ZERO_RESIDUAL_SHARED_CORE
        else [
            "shared_base",
            (
                "dense_exact_member"
                if artifact_kind == DENSE_EXACT_CONTROL
                else (
                    "independent_adapter"
                    if artifact_kind == INDEPENDENT_LOW_RANK_NONSHARING
                    else "primary_adapter"
                )
            ),
        ]
    )
    if observed_kinds != expected_kinds:
        raise ValueError("content-addressed component role inventory drifted")
    names = manifest.get("ordered_state_names")
    aliases = manifest.get("storage_alias_groups")
    if not isinstance(names, list) or not isinstance(aliases, list):
        raise TypeError("content-addressed point state catalog is invalid")
    artifact = MLPSharedFamilyPointArtifact(
        point_manifest=point_manifest,
        tensor_state=merged,
        tensor_state_sha256=manifest.get("tensor_state_sha256"),
        ordered_state_names=tuple(names),
        storage_alias_groups=tuple(tuple(group) for group in aliases),
        serialized_stored_scalar_count=manifest.get(
            "serialized_stored_scalar_count"
        ),
    )
    manifest_artifact_sha = _require_sha256(
        manifest.get("point_artifact_sha256"),
        label="point_artifact_sha256",
    )
    if artifact.artifact_sha256 != manifest_artifact_sha:
        raise ValueError("content-addressed point and legacy artifact identity differ")
    if expected_artifact_sha256 is not None and manifest_artifact_sha != _require_sha256(
        expected_artifact_sha256,
        label="expected_artifact_sha256",
    ):
        raise ValueError("content-addressed point does not match the frozen frontier")
    return artifact


def _build_content_addressed_store_inventory(
    store_root: Path,
    *,
    expected_arm_ids: Sequence[str],
) -> dict[str, object]:
    root = store_root.absolute()
    if (
        not isinstance(expected_arm_ids, Sequence)
        or isinstance(expected_arm_ids, (str, bytes))
        or not expected_arm_ids
        or len(set(expected_arm_ids)) != len(expected_arm_ids)
    ):
        raise ValueError("content-addressed store arm inventory is invalid")
    ordered_arm_ids = tuple(str(value) for value in expected_arm_ids)
    for arm_id in ordered_arm_ids:
        _require_id(arm_id, label="expected_arm_id")
    points_root = root / "points"
    blobs_root = root / "blobs"
    _assert_content_store_path(root, points_root)
    _assert_content_store_path(root, blobs_root)
    if not points_root.is_dir() or not blobs_root.is_dir():
        raise ValueError("content-addressed store directories are incomplete")
    allowed_root_entries = {"points", "blobs", "store-inventory.json"}
    actual_root_entries = {item.name for item in root.iterdir()}
    if not actual_root_entries <= allowed_root_entries:
        raise ValueError("content-addressed store contains unexpected root entries")
    expected_point_paths = {
        arm_id: content_addressed_mlp_shared_family_point_manifest_path(
            root,
            arm_id=arm_id,
        )
        for arm_id in ordered_arm_ids
    }
    if {item.name for item in points_root.iterdir()} != {
        path.name for path in expected_point_paths.values()
    }:
        raise ValueError("content-addressed store point inventory is not exact")

    manifests: dict[str, dict[str, object]] = {}
    referenced_blob_shas: set[str] = set()
    for arm_id, point_path in expected_point_paths.items():
        _assert_content_store_path(root, point_path)
        if not point_path.is_file():
            raise ValueError("content-addressed point manifest is not a regular file")
        try:
            raw = json.loads(point_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("content-addressed point manifest is invalid JSON") from error
        manifest = _validate_content_point_manifest(raw)
        point_manifest = manifest.get("point_manifest")
        if (
            not isinstance(point_manifest, Mapping)
            or point_manifest.get("arm_id") != arm_id
        ):
            raise ValueError("content-addressed point manifest arm binding drifted")
        references = manifest.get("component_blobs")
        if not isinstance(references, list) or not references:
            raise ValueError("content-addressed point omitted component references")
        reference_fields = {
            "blob_kind",
            "blob_sha256",
            "blob_file_basename",
            "tensor_state_sha256",
            "ordered_state_names",
            "stored_scalar_count",
            "tensor_storage_bytes",
        }
        for reference in references:
            if (
                not isinstance(reference, Mapping)
                or set(reference) != reference_fields
            ):
                raise ValueError("content-addressed component reference is invalid")
            referenced_blob_shas.add(
                _require_sha256(
                    reference.get("blob_sha256"),
                    label="component_blob_sha256",
                )
            )
        manifests[arm_id] = manifest

    expected_blob_names = {f"{value}.pt" for value in referenced_blob_shas}
    if {item.name for item in blobs_root.iterdir()} != expected_blob_names:
        raise ValueError("content-addressed store blob inventory is not exact")
    blob_states: dict[str, dict[str, Tensor]] = {}
    blob_metadata: dict[str, dict[str, object]] = {}
    blob_rows: list[dict[str, object]] = []
    for blob_sha in sorted(referenced_blob_shas):
        blob_path = blobs_root / f"{blob_sha}.pt"
        state, metadata = _load_content_blob(
            blob_path,
            expected_blob_sha256=blob_sha,
            store_root=root,
        )
        blob_states[blob_sha] = state
        blob_metadata[blob_sha] = metadata
        blob_rows.append(
            {
                "blob_kind": metadata["blob_kind"],
                "blob_sha256": blob_sha,
                "blob_file_basename": blob_path.name,
                "blob_file_sha256": _sha256_path_streaming(blob_path),
                "blob_file_bytes": blob_path.stat().st_size,
                "tensor_state_sha256": metadata["tensor_state_sha256"],
                "stored_scalar_count": metadata["stored_scalar_count"],
                "tensor_storage_bytes": metadata["tensor_storage_bytes"],
            }
        )

    point_rows: list[dict[str, object]] = []
    logical_flat_tensor_bytes = 0
    for arm_id in ordered_arm_ids:
        manifest = manifests[arm_id]
        references = manifest["component_blobs"]
        merged: dict[str, Tensor] = {}
        observed_kinds: list[str] = []
        for reference in references:
            blob_sha = str(reference["blob_sha256"])
            metadata = blob_metadata[blob_sha]
            for field in (
                "blob_kind",
                "tensor_state_sha256",
                "ordered_state_names",
                "stored_scalar_count",
                "tensor_storage_bytes",
            ):
                if reference.get(field) != metadata.get(field):
                    raise ValueError(
                        "content-addressed component reference drifted"
                    )
            if set(merged) & set(blob_states[blob_sha]):
                raise ValueError(
                    "content-addressed component state names overlap"
                )
            merged.update(blob_states[blob_sha])
            observed_kinds.append(str(metadata["blob_kind"]))
        point_manifest = manifest["point_manifest"]
        artifact_kind = point_manifest.get("artifact_kind")
        expected_kinds = (
            ["shared_base"]
            if artifact_kind == ZERO_RESIDUAL_SHARED_CORE
            else [
                "shared_base",
                (
                    "dense_exact_member"
                    if artifact_kind == DENSE_EXACT_CONTROL
                    else (
                        "independent_adapter"
                        if artifact_kind == INDEPENDENT_LOW_RANK_NONSHARING
                        else "primary_adapter"
                    )
                ),
            ]
        )
        if observed_kinds != expected_kinds:
            raise ValueError(
                "content-addressed component role inventory drifted"
            )
        artifact = MLPSharedFamilyPointArtifact(
            point_manifest=point_manifest,
            tensor_state=merged,
            tensor_state_sha256=manifest["tensor_state_sha256"],
            ordered_state_names=tuple(manifest["ordered_state_names"]),
            storage_alias_groups=tuple(
                tuple(group) for group in manifest["storage_alias_groups"]
            ),
            serialized_stored_scalar_count=manifest[
                "serialized_stored_scalar_count"
            ],
        )
        if artifact.artifact_sha256 != manifest["point_artifact_sha256"]:
            raise ValueError(
                "content-addressed point and artifact identity differ"
            )
        logical_tensor_bytes = _state_tensor_storage_bytes(merged)
        logical_flat_tensor_bytes += logical_tensor_bytes
        point_path = expected_point_paths[arm_id]
        point_rows.append(
            {
                "arm_id": arm_id,
                "point_manifest_path_basename": (
                    point_path.relative_to(root).as_posix()
                ),
                "content_point_manifest_sha256": manifest[
                    "content_point_manifest_sha256"
                ],
                "point_manifest_file_sha256": _sha256_path_streaming(
                    point_path
                ),
                "point_manifest_file_bytes": point_path.stat().st_size,
                "point_artifact_sha256": artifact.artifact_sha256,
                "candidate_state_sha256": point_manifest[
                    "candidate_state_sha256"
                ],
                "serialized_stored_scalar_count": artifact.serialized_stored_scalar_count,
                "logical_tensor_storage_bytes": logical_tensor_bytes,
                "ordered_component_blob_sha256s": [
                    reference["blob_sha256"] for reference in references
                ],
            }
        )
    unique_tensor_bytes = sum(
        int(row["tensor_storage_bytes"]) for row in blob_rows
    )
    reference_count = sum(
        len(row["ordered_component_blob_sha256s"]) for row in point_rows
    )
    savings_bytes = logical_flat_tensor_bytes - unique_tensor_bytes
    if savings_bytes <= 0:
        raise ValueError("content-addressed store did not deduplicate storage")
    payload: dict[str, object] = {
        "schema": MLP_SHARED_FAMILY_CONTENT_STORE_SCHEMA,
        "format_version": MLP_SHARED_FAMILY_FORMAT_VERSION,
        "ordered_arm_ids": list(ordered_arm_ids),
        "point_manifest_count": len(point_rows),
        "point_manifests": point_rows,
        "unique_blob_count": len(blob_rows),
        "unique_blobs": blob_rows,
        "component_blob_reference_count": reference_count,
        "reused_component_blob_reference_count": (
            reference_count - len(blob_rows)
        ),
        "logical_flat_tensor_storage_bytes": logical_flat_tensor_bytes,
        "unique_tensor_storage_bytes": unique_tensor_bytes,
        "tensor_storage_savings_bytes": savings_bytes,
        "tensor_storage_savings_fraction": (
            savings_bytes / logical_flat_tensor_bytes
        ),
        "unique_blob_file_bytes": sum(
            int(row["blob_file_bytes"]) for row in blob_rows
        ),
        "point_manifest_file_bytes": sum(
            int(row["point_manifest_file_bytes"]) for row in point_rows
        ),
        "logical_candidate_scalar_accounting_unchanged": True,
        "equal_values_without_logical_alias_remain_separate": True,
        "shuffled_adapter_reuses_primary_blob": True,
        "source_free_serialized_candidates": True,
        "exact_point_and_blob_inventory": True,
    }
    payload["store_inventory_sha256"] = _fingerprint(
        _CONTENT_STORE_DOMAIN,
        payload,
    )
    return payload


def write_mlp_shared_family_content_addressed_store_inventory(
    store_root: Path | str,
    *,
    expected_arm_ids: Sequence[str],
) -> dict[str, object]:
    root = Path(store_root)
    inventory = _build_content_addressed_store_inventory(
        root,
        expected_arm_ids=expected_arm_ids,
    )
    path = root / "store-inventory.json"

    def validate_temporary_inventory(temporary: Path) -> None:
        raw = json.loads(temporary.read_text(encoding="utf-8"))
        if raw != inventory:
            raise ValueError("temporary store inventory serialization drifted")

    created = _atomic_write_exclusive(
        path,
        store_root=root,
        write_payload=lambda handle: handle.write(
            _canonical_json_bytes(inventory)
        ),
        validate_temporary=validate_temporary_inventory,
    )
    if not created:
        raise FileExistsError(
            "refusing to overwrite a shared-family store inventory"
        )
    return validate_mlp_shared_family_content_addressed_store_inventory(
        root,
        expected_arm_ids=expected_arm_ids,
        expected_inventory_sha256=str(inventory["store_inventory_sha256"]),
    )


def validate_mlp_shared_family_content_addressed_store_inventory(
    store_root: Path | str,
    *,
    expected_arm_ids: Sequence[str],
    expected_inventory_sha256: str | None = None,
) -> dict[str, object]:
    root = Path(store_root)
    path = root / "store-inventory.json"
    _assert_content_store_path(root, path)
    if not path.is_file():
        raise ValueError("content-addressed store inventory is not a regular file")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("content-addressed store inventory is invalid JSON") from error
    rebuilt = _build_content_addressed_store_inventory(
        root,
        expected_arm_ids=expected_arm_ids,
    )
    if stored != rebuilt:
        raise ValueError("content-addressed store inventory authentication failed")
    observed = _require_sha256(
        rebuilt.get("store_inventory_sha256"),
        label="store_inventory_sha256",
    )
    if expected_inventory_sha256 is not None and observed != _require_sha256(
        expected_inventory_sha256,
        label="expected_inventory_sha256",
    ):
        raise ValueError("content-addressed store inventory changed after freezing")
    inventory_file_bytes = path.stat().st_size
    return {
        **rebuilt,
        "inventory_path_basename": path.name,
        "inventory_file_sha256": _sha256_path_streaming(path),
        "inventory_file_bytes": inventory_file_bytes,
        "serialized_store_file_bytes": (
            int(rebuilt["unique_blob_file_bytes"])
            + int(rebuilt["point_manifest_file_bytes"])
            + inventory_file_bytes
        ),
    }


__all__ = [
    "ARTIFACT_KINDS",
    "ASSIGNMENT_MODES",
    "DENSE_EXACT_CONTROL",
    "FITTED_ASSIGNMENT",
    "Gemma3MLPSharedFamilyCompilation",
    "Gemma3MLPSharedFamilyExecutor",
    "INDEPENDENT_LOW_RANK_NONSHARING",
    "LowRankProjectionDelta",
    "MLPFamilyPairManifest",
    "MLPSharedFamilyExecution",
    "MLPSharedFamilyExecutionReceipt",
    "MLPSharedFamilyPointArtifact",
    "MLP_SHARED_FAMILY_CONTENT_BLOB_SCHEMA",
    "MLP_SHARED_FAMILY_CONTENT_POINT_SCHEMA",
    "MLP_SHARED_FAMILY_CONTENT_STORE_SCHEMA",
    "PINNED_SUCCESSOR_SETUP_SHA256",
    "PRIMARY_POINT_ORDER",
    "PRIMARY_RESIDUAL_RANKS",
    "SHARED_CORE_LOW_RANK",
    "SHUFFLED_ASSIGNMENT",
    "SWAPPED_ASSIGNMENT",
    "SharedCoreResidualAdapterMLPRuntime",
    "ZERO_RESIDUAL_SHARED_CORE",
    "build_mlp_family_pair_manifest",
    "compile_gemma3_mlp_shared_family_residual_adapter",
    "compile_no_refit_residual_assignment_control",
    "content_addressed_mlp_shared_family_point_manifest_path",
    "initialize_mlp_projection_factors",
    "initialize_mlp_shared_family_content_addressed_store",
    "load_mlp_shared_family_point_artifact",
    "load_mlp_shared_family_content_addressed_point_artifact",
    "pinned_mlp_pair_resource_ladder",
    "pinned_mlp_pair_resource_point",
    "projection_factors_from_compilation",
    "validate_mlp_family_pair_binding",
    "validate_mlp_shared_family_point_artifact",
    "validate_mlp_shared_family_content_addressed_store_inventory",
    "write_mlp_shared_family_point_artifact",
    "write_mlp_shared_family_content_addressed_point_artifact",
    "write_mlp_shared_family_content_addressed_store_inventory",
]
