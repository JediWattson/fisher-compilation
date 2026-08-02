"""Frozen-preserving A16 complete-H4 basis/rank capacity ladder.

The runner collects the authenticated exact-X4 H4 pair once per prompt, fits
both Fisher-alignment-tilted and unweighted family/example-macro bases, and
tests ranks 64/96/128/192.  Every prompt owns one evaluation shadow shared by
all eight one-forward projection arms.  This remains a same-A truth-leaking
capacity oracle: it does not learn per-row coordinates and authorizes no
serving, compression, or speed claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_projection as projection_math
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionBasis,
    CompleteH4ProjectionFitSequence,
    ProjectionFitWeighting,
    fit_complete_h4_projection_basis,
    project_complete_h4_residual_rows,
    summarize_complete_h4_projection_geometry,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
    _runtime_tensor_sha256,
    gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_RANK64_PROJECTION_BASELINE",
    "classify_projection_ladder_arm",
    "run_gemma3_l3_l4_complete_h4_projection_basis_rank_ladder",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_RANK64_PROJECTION_BASELINE = frozen.DEFAULT_OUTPUT
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-"
    "complete-h4-projection-basis-rank-ladder-a-fit16-dev-v1.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_complete_h4_projection_basis_rank_ladder_"
    "development"
)
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-projection-basis-rank-ladder:"
    b"v1\0"
)
_FIT_TO_PREFIX_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-fit-to-prefix:v1\0"
)
_TAIL_INFORMED_FIT_TO_PREFIX_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-tail-informed-fit-to-prefix:v1\0"
)
_ARM_RECEIPT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-ladder-arm-receipt:v1\0"
)
_FIT_MANIFEST_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-ladder-fit-manifest:v1\0"
)
_U192_PARENT_INVARIANT_RECEIPTS_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-u192-parent-invariant-receipts:"
    b"v1\0"
)

_U192_PARENT_RECEIPT_TOP_LEVEL_INCLUDED_FIELDS = (
    "example_id",
    "family_id",
    "prompt_sha256",
    "model_inputs_sha256",
    "execution_grid_sha256",
    "complete_h4_support_rows",
    "complete_h4_padding_write_rows",
)
_U192_PARENT_RECEIPT_TOP_LEVEL_EXCLUDED_FIELDS = (
    "fit_to_prefix_before",
    "fit_to_prefix_after",
    "arm",
    "receipt_sha256",
)
_U192_PARENT_RECEIPT_PREFIX_INCLUDED_FIELDS = (
    "schema",
    "format_version",
    "arm_id",
    "fit_weighting",
    "residual_width",
    "prefix_rank",
    "prefix_definition",
    "execution_ordering",
    "execution_basis_sha256",
    "execution_basis_artifact_sha256",
)
_U192_PARENT_RECEIPT_PREFIX_EXCLUDED_FIELDS = (
    # These describe the containing fit.  They legitimately change when the
    # exact same authenticated U192 prefix comes from a U320 fit rather than
    # the frozen parent's U192 fit.
    "fit_basis_artifact_sha256",
    "fit_basis_matrix_sha256",
    "fit_max_rank",
    "fit_to_prefix_lineage_sha256",
)
_U192_PARENT_RECEIPT_ARM_INCLUDED_FIELDS = (
    "role",
    "execution_mode",
    "projection_rank",
    "metrics_only",
    "serving_authorized",
    "model_forward_count",
    "injected_h4_sha256",
    "native_h4_sha256",
    "incomplete_h4_sha256",
    "projected_delta_sha256",
    "projection_basis_sha256",
    "projection_basis_artifact_sha256",
    "projection_ordering",
    "projection_definition",
    "projection_basis_orthonormal_max_abs_error",
    "complete_h4_pair_artifact_sha256",
    "shadow_result_artifact_sha256",
    "runtime_binding_sha256",
    "model_inputs_sha256",
    "execution_grid_sha256",
    "adapter_execution_sha256",
    "complete_h4_support_mask_sha256",
    "boundary_callback_order",
    "logits_bitwise_authoritative",
    "max_abs_authoritative_logit_error",
    "logits_sha256",
)
_U192_PARENT_RECEIPT_ARM_EXCLUDED_FIELDS = (
    # The result artifact incorporates projection_fit_basis_artifact_sha256,
    # so both fields legitimately change with the containing U320 fit.
    "projection_fit_basis_artifact_sha256",
    "artifact_sha256",
)

_RANK_GRID: tuple[int, ...] = (64, 96, 128, 192)
_MAX_RANK = 192
_WIDTH = 640
_WEIGHTINGS: tuple[ProjectionFitWeighting, ...] = (
    "fisher_alignment_tilted",
    "unweighted",
)
_EXECUTION_ORDERING = {
    "fisher_alignment_tilted": (
        "descending_fisher_tilted_residual_eigenvalue"
    ),
    "unweighted": "descending_unweighted_residual_eigenvalue",
}

_EXPECTED_PRIOR_FILE_SHA256 = (
    "c20dc948b280c24a4b7f6f9dd43e54e5660b94092ed4446ba9145e05098ab73a"
)
_EXPECTED_PRIOR_REPORT_SHA256 = (
    "8d767392281502f6fe83a8cb21e68a82fb3a743a26063cc75a8de8b9aa34de70"
)
_EXPECTED_TILTED_RANK64_EXECUTION_BASIS_ARTIFACT_SHA256 = (
    "7ebf58a9db8ec921b33beed207a214bfd6e53d9c32f1a3a8e8d0ccfc112f0bda"
)
_EXPECTED_UNWEIGHTED_RANK192_EXECUTION_BASIS_ARTIFACT_SHA256 = (
    "50ddc74586954663bf2ec0330847f6f83356d7e9e4095643fba7ab1a62c9248c"
)

_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer_state": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_logits": False,
    "contains_activation_tensors": False,
    "contains_gradient_tensors": False,
    "contains_basis_coefficients": False,
    "contains_scalar_metrics": True,
    "truth_leaking_same_a_capacity_oracle": True,
    "calibration_b_opened": False,
    "validation_opened": False,
    "test_opened": False,
    "artifact_must_remain_outside_git": True,
    "committable": False,
}


@dataclass(frozen=True, slots=True)
class _ProjectionArmSpec:
    arm_id: str
    fit_weighting: ProjectionFitWeighting
    rank: int
    fit_basis: CompleteH4ProjectionBasis
    execution_basis: Tensor
    execution_ordering: str
    execution_basis_sha256: str
    execution_basis_artifact_sha256: str
    fit_to_prefix_lineage: Mapping[str, object]
    fit_to_prefix_lineage_sha256: str

    @property
    def projection_fit_artifact_sha256(self) -> str:
        return self.fit_basis.artifact_sha256


@dataclass(frozen=True, slots=True)
class _TailInformedProjectionArmSpec:
    arm_id: str
    fit_weighting: ProjectionFitWeighting
    rank: int
    tail_informed_fit: object
    execution_basis: Tensor
    execution_ordering: str
    execution_basis_sha256: str
    execution_basis_artifact_sha256: str
    fit_to_prefix_lineage: Mapping[str, object]
    fit_to_prefix_lineage_sha256: str

    @property
    def projection_fit_artifact_sha256(self) -> str:
        value = getattr(self.tail_informed_fit, "artifact_sha256", None)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("tail-informed fit artifact is invalid")
        return value


_AnyProjectionArmSpec = _ProjectionArmSpec | _TailInformedProjectionArmSpec


@dataclass(frozen=True, slots=True)
class _TailInformedFactorialConfig:
    schema: str
    format_version: int
    report_domain: bytes
    role: str
    parent_ladder_path: Path | str
    parent_ladder_file_sha256: str
    parent_ladder_report_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema, str)
            or not self.schema
            or type(self.format_version) is not int
            or self.format_version <= 0
            or not isinstance(self.report_domain, bytes)
            or not self.report_domain.endswith(b"\0")
            or not isinstance(self.role, str)
            or not self.role
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.parent_ladder_file_sha256,
                    self.parent_ladder_report_sha256,
                )
            )
        ):
            raise ValueError("tail-informed factorial configuration is invalid")


def _domain_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(
        domain + frozen._canonical_json_bytes(value)
    ).hexdigest()


def _u192_receipt_fields(
    value: object,
    *,
    included_fields: Sequence[str],
    excluded_fields: Sequence[str],
    label: str,
) -> dict[str, object]:
    """Select one closed receipt schema while documenting allowed drift."""

    row = frozen._mapping(value, label=label)
    included = tuple(included_fields)
    excluded = tuple(excluded_fields)
    expected = set(included) | set(excluded)
    if (
        len(expected) != len(included) + len(excluded)
        or set(row) != expected
    ):
        raise ValueError(f"{label} fields differ from the closed U192 policy")
    return {name: row[name] for name in included}


def _u192_parent_invariant_receipt(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    """Remove only fields expected to change between U192 and U320 fits."""

    row = frozen._mapping(value, label=label)
    top_level = _u192_receipt_fields(
        row,
        included_fields=_U192_PARENT_RECEIPT_TOP_LEVEL_INCLUDED_FIELDS,
        excluded_fields=_U192_PARENT_RECEIPT_TOP_LEVEL_EXCLUDED_FIELDS,
        label=label,
    )
    before = _u192_receipt_fields(
        row.get("fit_to_prefix_before"),
        included_fields=_U192_PARENT_RECEIPT_PREFIX_INCLUDED_FIELDS,
        excluded_fields=_U192_PARENT_RECEIPT_PREFIX_EXCLUDED_FIELDS,
        label=f"{label} fit_to_prefix_before",
    )
    after = _u192_receipt_fields(
        row.get("fit_to_prefix_after"),
        included_fields=_U192_PARENT_RECEIPT_PREFIX_INCLUDED_FIELDS,
        excluded_fields=_U192_PARENT_RECEIPT_PREFIX_EXCLUDED_FIELDS,
        label=f"{label} fit_to_prefix_after",
    )
    arm = _u192_receipt_fields(
        row.get("arm"),
        included_fields=_U192_PARENT_RECEIPT_ARM_INCLUDED_FIELDS,
        excluded_fields=_U192_PARENT_RECEIPT_ARM_EXCLUDED_FIELDS,
        label=f"{label} arm",
    )
    example_id = top_level.get("example_id")
    if (
        not isinstance(example_id, str)
        or not example_id
        or example_id != example_id.strip()
    ):
        raise ValueError(f"{label} example_id is invalid")
    return {
        **top_level,
        "fit_to_prefix_before": before,
        "fit_to_prefix_after": after,
        "arm": arm,
    }


def _validate_u192_parent_invariant_receipts(
    *,
    parent_receipts: object,
    live_receipts: object,
    expected_prompt_count: int = 16,
) -> dict[str, object]:
    """Require every non-fit-dependent U192 prompt receipt to remain exact."""

    if type(expected_prompt_count) is not int or expected_prompt_count <= 0:
        raise ValueError("U192 parent expected_prompt_count must be positive")
    if not isinstance(parent_receipts, (tuple, list)) or not isinstance(
        live_receipts,
        (tuple, list),
    ):
        raise TypeError("U192 parent receipts must be sequences")
    if (
        len(parent_receipts) != expected_prompt_count
        or len(live_receipts) != expected_prompt_count
    ):
        raise ValueError("U192 parent receipt prompt count differs")

    def normalize(
        receipts: Sequence[object],
        *,
        label: str,
    ) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for index, raw in enumerate(receipts):
            invariant = _u192_parent_invariant_receipt(
                raw,
                label=f"{label} receipt {index}",
            )
            example_id = str(invariant["example_id"])
            if example_id in result:
                raise ValueError(f"{label} U192 receipt example_id repeats")
            result[example_id] = invariant
        return result

    parent = normalize(parent_receipts, label="parent")
    live = normalize(live_receipts, label="live")
    parent_payload = tuple(parent[name] for name in sorted(parent))
    live_payload = tuple(live[name] for name in sorted(live))
    if (
        frozen._canonical_json_bytes(parent_payload)
        != frozen._canonical_json_bytes(live_payload)
    ):
        raise RuntimeError("U192 live per-prompt invariant receipts differ from parent")
    receipt_sha256 = _domain_sha256(
        parent_payload,
        domain=_U192_PARENT_INVARIANT_RECEIPTS_DOMAIN,
    )
    return {
        "matched": True,
        "prompt_count": expected_prompt_count,
        "invariant_receipts_sha256": receipt_sha256,
        "included_field_policy": {
            "top_level": _U192_PARENT_RECEIPT_TOP_LEVEL_INCLUDED_FIELDS,
            "fit_to_prefix": _U192_PARENT_RECEIPT_PREFIX_INCLUDED_FIELDS,
            "arm": _U192_PARENT_RECEIPT_ARM_INCLUDED_FIELDS,
        },
        "excluded_only_because_u192_fit_became_u320_fit": {
            "top_level": ("receipt_sha256",),
            "fit_to_prefix": _U192_PARENT_RECEIPT_PREFIX_EXCLUDED_FIELDS,
            "arm": _U192_PARENT_RECEIPT_ARM_EXCLUDED_FIELDS,
        },
    }


def _fit_to_prefix_payload(
    *,
    arm_id: str,
    fit_basis: CompleteH4ProjectionBasis,
    rank: int,
    execution_basis: Tensor,
    execution_ordering: str,
    execution_basis_artifact_sha256: str,
) -> dict[str, object]:
    if not isinstance(fit_basis, CompleteH4ProjectionBasis):
        raise TypeError("fit_basis must be a complete-H4 projection basis")
    if type(rank) is not int or rank <= 0 or rank > fit_basis.max_rank:
        raise ValueError("prefix rank must lie inside the fitted basis")
    expected = fit_basis.basis_tensor(ordering="euclidean")[:rank].contiguous()
    if (
        not isinstance(execution_basis, Tensor)
        or execution_basis.dtype != torch.float64
        or execution_basis.device.type != "cpu"
        or not execution_basis.is_contiguous()
        or not torch.equal(execution_basis, expected)
    ):
        raise ValueError("execution basis is not the exact fitted rank prefix")
    expected_ordering = _EXECUTION_ORDERING[fit_basis.fit_weighting]
    if execution_ordering != expected_ordering:
        raise ValueError("execution ordering differs from fit weighting")
    expected_artifact = (
        gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            execution_basis,
            projection_rank=rank,
            projection_ordering=execution_ordering,
        )
    )
    if execution_basis_artifact_sha256 != expected_artifact:
        raise ValueError("execution basis artifact differs from exact prefix")
    basis_metadata = fit_basis.metadata()
    matrix = frozen._mapping(
        basis_metadata.get("basis_rows"),
        label="fit basis rows",
    )
    return {
        "schema": "fisher_graph.complete_h4_fit_to_prefix_lineage",
        "format_version": 1,
        "arm_id": arm_id,
        "fit_basis_artifact_sha256": fit_basis.artifact_sha256,
        "fit_basis_matrix_sha256": matrix["matrix_sha256"],
        "fit_weighting": fit_basis.fit_weighting,
        "fit_max_rank": fit_basis.max_rank,
        "residual_width": fit_basis.width,
        "prefix_rank": rank,
        "prefix_definition": (
            "first_rank_rows_of_fit_basis_in_residual_eigenvalue_order"
        ),
        "execution_ordering": execution_ordering,
        "execution_basis_sha256": _runtime_tensor_sha256(execution_basis),
        "execution_basis_artifact_sha256": expected_artifact,
    }


def _build_projection_arm_specs(
    bases: Mapping[ProjectionFitWeighting, CompleteH4ProjectionBasis],
    *,
    ranks: Sequence[int] = _RANK_GRID,
    weightings: Sequence[ProjectionFitWeighting] = _WEIGHTINGS,
) -> tuple[_ProjectionArmSpec, ...]:
    normalized_weightings = tuple(weightings)
    if (
        not normalized_weightings
        or len(set(normalized_weightings)) != len(normalized_weightings)
        or any(value not in _WEIGHTINGS for value in normalized_weightings)
        or set(bases) != set(normalized_weightings)
    ):
        raise ValueError("ladder bases must exactly match its weightings")
    normalized_ranks = tuple(ranks)
    if (
        not normalized_ranks
        or tuple(sorted(set(normalized_ranks))) != normalized_ranks
        or any(type(rank) is not int or rank <= 0 for rank in normalized_ranks)
    ):
        raise ValueError("ladder ranks must be strictly increasing positive integers")
    specs: list[_ProjectionArmSpec] = []
    for weighting in normalized_weightings:
        fit_basis = bases[weighting]
        if fit_basis.fit_weighting != weighting:
            raise ValueError("fit basis weighting label differs")
        for rank in normalized_ranks:
            execution_basis = fit_basis.basis_tensor(
                ordering="euclidean"
            )[:rank].clone().contiguous()
            ordering = _EXECUTION_ORDERING[weighting]
            artifact = (
                gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
                    execution_basis,
                    projection_rank=rank,
                    projection_ordering=ordering,
                )
            )
            arm_id = f"{weighting}.rank{rank}"
            payload = _fit_to_prefix_payload(
                arm_id=arm_id,
                fit_basis=fit_basis,
                rank=rank,
                execution_basis=execution_basis,
                execution_ordering=ordering,
                execution_basis_artifact_sha256=artifact,
            )
            specs.append(
                _ProjectionArmSpec(
                    arm_id=arm_id,
                    fit_weighting=weighting,
                    rank=rank,
                    fit_basis=fit_basis,
                    execution_basis=execution_basis,
                    execution_ordering=ordering,
                    execution_basis_sha256=str(payload["execution_basis_sha256"]),
                    execution_basis_artifact_sha256=artifact,
                    fit_to_prefix_lineage=payload,
                    fit_to_prefix_lineage_sha256=_domain_sha256(
                        payload,
                        domain=_FIT_TO_PREFIX_DOMAIN,
                    ),
                )
            )
    return tuple(specs)


def _build_tail_informed_projection_arm_specs(
    tail_informed_fit: object,
    *,
    ranks: Sequence[int],
) -> tuple[_TailInformedProjectionArmSpec, ...]:
    validate = getattr(tail_informed_fit, "validate_integrity", None)
    basis_for_rank = getattr(tail_informed_fit, "basis_tensor", None)
    if not callable(validate) or not callable(basis_for_rank):
        raise TypeError("tail-informed fit lacks integrity-bound basis methods")
    validate()
    normalized_ranks = tuple(ranks)
    tail_rank = getattr(tail_informed_fit, "tail_rank", None)
    max_rank = getattr(tail_informed_fit, "max_rank", None)
    if (
        tail_rank != 17
        or type(max_rank) is not int
        or normalized_ranks != (209, 224, 256, 320)
        or max_rank != 320
        or tuple(sorted(set(normalized_ranks))) != normalized_ranks
    ):
        raise ValueError(
            "locked A16 tail-informed ladder requires rT=17 and ranks "
            "209/224/256/320"
        )
    specs: list[_TailInformedProjectionArmSpec] = []
    for rank in normalized_ranks:
        execution_basis = basis_for_rank(rank)
        if not isinstance(execution_basis, Tensor):
            raise TypeError("tail-informed basis_tensor must return a tensor")
        execution_basis = execution_basis.clone().contiguous()
        artifact = gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            execution_basis,
            projection_rank=rank,
            projection_ordering=COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
        )
        arm_id = f"tail_informed.rank{rank}"
        payload = _tail_informed_fit_to_prefix_payload(
            arm_id=arm_id,
            tail_informed_fit=tail_informed_fit,
            rank=rank,
            execution_basis=execution_basis,
            execution_basis_artifact_sha256=artifact,
        )
        specs.append(
            _TailInformedProjectionArmSpec(
                arm_id=arm_id,
                fit_weighting="unweighted",
                rank=rank,
                tail_informed_fit=tail_informed_fit,
                execution_basis=execution_basis,
                execution_ordering=(
                    COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
                ),
                execution_basis_sha256=_runtime_tensor_sha256(
                    execution_basis
                ),
                execution_basis_artifact_sha256=artifact,
                fit_to_prefix_lineage=payload,
                fit_to_prefix_lineage_sha256=_domain_sha256(
                    payload,
                    domain=_TAIL_INFORMED_FIT_TO_PREFIX_DOMAIN,
                ),
            )
        )
    return tuple(specs)


def _validate_projection_arm_spec(spec: _ProjectionArmSpec) -> dict[str, object]:
    if not isinstance(spec, _ProjectionArmSpec):
        raise TypeError("projection arm spec has the wrong type")
    payload = _fit_to_prefix_payload(
        arm_id=spec.arm_id,
        fit_basis=spec.fit_basis,
        rank=spec.rank,
        execution_basis=spec.execution_basis,
        execution_ordering=spec.execution_ordering,
        execution_basis_artifact_sha256=(
            spec.execution_basis_artifact_sha256
        ),
    )
    if (
        frozen._canonical_json_bytes(payload)
        != frozen._canonical_json_bytes(spec.fit_to_prefix_lineage)
        or _runtime_tensor_sha256(spec.execution_basis)
        != spec.execution_basis_sha256
        or _domain_sha256(payload, domain=_FIT_TO_PREFIX_DOMAIN)
        != spec.fit_to_prefix_lineage_sha256
    ):
        raise ValueError("fit-to-prefix lineage receipt differs")
    return {
        **payload,
        "fit_to_prefix_lineage_sha256": spec.fit_to_prefix_lineage_sha256,
    }


def _tail_informed_fit_to_prefix_payload(
    *,
    arm_id: str,
    tail_informed_fit: object,
    rank: int,
    execution_basis: Tensor,
    execution_basis_artifact_sha256: str,
) -> dict[str, object]:
    validate = getattr(tail_informed_fit, "validate_integrity", None)
    basis_for_rank = getattr(tail_informed_fit, "basis_tensor", None)
    metadata_method = getattr(tail_informed_fit, "metadata", None)
    lineage_method = getattr(tail_informed_fit, "lineage", None)
    if not callable(validate) or not callable(basis_for_rank) or not callable(
        metadata_method
    ) or not callable(lineage_method):
        raise TypeError("tail-informed fit lacks integrity-bound basis methods")
    validate()
    expected = basis_for_rank(rank)
    if (
        not isinstance(expected, Tensor)
        or expected.dtype != torch.float64
        or expected.device.type != "cpu"
        or not expected.is_contiguous()
        or not torch.equal(execution_basis, expected)
    ):
        raise ValueError("execution basis is not the exact tail-informed prefix")
    artifact_sha256 = getattr(tail_informed_fit, "artifact_sha256", None)
    tail_rank = getattr(tail_informed_fit, "tail_rank", None)
    max_rank = getattr(tail_informed_fit, "max_rank", None)
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or type(tail_rank) is not int
        or tail_rank <= 0
        or type(max_rank) is not int
        or not 192 + tail_rank <= rank <= max_rank
    ):
        raise ValueError("tail-informed fit rank or artifact is invalid")
    expected_artifact = (
        gemma3_l3_l4_complete_h4_projection_basis_artifact_sha256(
            execution_basis,
            projection_rank=rank,
            projection_ordering=(
                COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
            ),
        )
    )
    if execution_basis_artifact_sha256 != expected_artifact:
        raise ValueError("tail-informed execution basis artifact differs")
    metadata = frozen._mapping(
        metadata_method(),
        label="tail-informed fit metadata",
    )
    if metadata.get("artifact_sha256") != artifact_sha256:
        raise ValueError("tail-informed fit metadata artifact differs")
    fit_lineage = dict(
        frozen._mapping(
            lineage_method(rank, expected_artifact),
            label="tail-informed fit prefix lineage",
        )
    )
    if (
        fit_lineage.get("fit_artifact_sha256") != artifact_sha256
        or fit_lineage.get("rank") != rank
        or fit_lineage.get("tail_rank") != tail_rank
        or fit_lineage.get("ordering")
        != COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
        or fit_lineage.get("execution_basis_artifact_sha256")
        != expected_artifact
        or not isinstance(fit_lineage.get("prefix_artifact_sha256"), str)
        or not isinstance(fit_lineage.get("lineage_sha256"), str)
    ):
        raise ValueError("tail-informed fit prefix lineage differs")
    return {
        "schema": "fisher_graph.complete_h4_tail_informed_fit_to_prefix_lineage",
        "format_version": 1,
        "arm_id": arm_id,
        "tail_informed_fit_artifact_sha256": artifact_sha256,
        "fit_weighting": "unweighted",
        "fit_max_rank": max_rank,
        "global_prefix_rank": 192,
        "tail_residual_span_rank": tail_rank,
        "prefix_rank": rank,
        "prefix_definition": (
            "u192_then_full_tail_residual_svd_span_then_two_pass_mgs_"
            "remaining_u320_prefix"
        ),
        "execution_ordering": COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING,
        "execution_basis_sha256": _runtime_tensor_sha256(execution_basis),
        "execution_basis_artifact_sha256": expected_artifact,
        "tail_informed_fit_prefix_lineage": fit_lineage,
    }


def _validate_tail_informed_projection_arm_spec(
    spec: _TailInformedProjectionArmSpec,
) -> dict[str, object]:
    if not isinstance(spec, _TailInformedProjectionArmSpec):
        raise TypeError("tail-informed projection arm spec has the wrong type")
    payload = _tail_informed_fit_to_prefix_payload(
        arm_id=spec.arm_id,
        tail_informed_fit=spec.tail_informed_fit,
        rank=spec.rank,
        execution_basis=spec.execution_basis,
        execution_basis_artifact_sha256=(
            spec.execution_basis_artifact_sha256
        ),
    )
    if (
        frozen._canonical_json_bytes(payload)
        != frozen._canonical_json_bytes(spec.fit_to_prefix_lineage)
        or _runtime_tensor_sha256(spec.execution_basis)
        != spec.execution_basis_sha256
        or _domain_sha256(
            payload,
            domain=_TAIL_INFORMED_FIT_TO_PREFIX_DOMAIN,
        )
        != spec.fit_to_prefix_lineage_sha256
    ):
        raise ValueError("tail-informed fit-to-prefix lineage receipt differs")
    return {
        **payload,
        "fit_to_prefix_lineage_sha256": spec.fit_to_prefix_lineage_sha256,
    }


def _validate_any_projection_arm_spec(
    spec: _AnyProjectionArmSpec,
) -> dict[str, object]:
    if isinstance(spec, _ProjectionArmSpec):
        return _validate_projection_arm_spec(spec)
    return _validate_tail_informed_projection_arm_spec(spec)


def _project_projection_arm_rows(
    residual_rows: object,
    spec: _AnyProjectionArmSpec,
) -> Tensor:
    if isinstance(spec, _ProjectionArmSpec):
        return project_complete_h4_residual_rows(
            residual_rows,
            spec.fit_basis,
            rank=spec.rank,
            ordering="euclidean",
        )
    if hasattr(residual_rows, "to_tensor"):
        rows = residual_rows.to_tensor()
    elif isinstance(residual_rows, Tensor):
        rows = residual_rows.detach().to(device="cpu", dtype=torch.float64)
    else:
        raise TypeError("projection rows must be a tensor or immutable matrix")
    return (rows @ spec.execution_basis.T) @ spec.execution_basis


def _geometry_with_examples(
    traces: Sequence[frozen._PromptTrace],
    projected_by_example: Mapping[str, Tensor],
    *,
    candidate_semantics: str = "cast_once_correction_submitted_to_runtime",
) -> dict[str, object]:
    """Measure the correction that was actually cast and submitted.

    Pooled geometry is row-pooled.  Family geometry is deliberately an
    unweighted macro over examples, matching the fit panel's example/family
    balance instead of allowing a long prompt to dominate its sibling.  An
    empty graph-core or causal-tail stratum is retained as an explicit N/A;
    only nonempty strata participate in gates.
    """

    pooled = {
        "full": frozen._ProjectionMoments(),
        "graph_core": frozen._ProjectionMoments(),
        "causal_tail": frozen._ProjectionMoments(),
    }
    examples: list[dict[str, object]] = []
    for trace in sorted(traces, key=lambda value: value.example.example_id):
        source = trace.fit_sequence.residual_rows.to_tensor()  # type: ignore[union-attr]
        candidate = projected_by_example[trace.example.example_id]
        if candidate.shape != source.shape:
            raise ValueError("executed correction rows differ from H4 truth")
        core = trace.graph_core_rows
        rows: dict[str, object] = {}
        for name, mask in (
            ("full", torch.ones_like(core, dtype=torch.bool)),
            ("graph_core", core),
            ("causal_tail", ~core),
        ):
            if bool(mask.any()):
                moments = frozen._ProjectionMoments()
                moments.add(source[mask], candidate[mask])
                pooled[name].add(source[mask], candidate[mask])
                rows[name] = {**moments.summary(), "applicable": True}
            else:
                rows[name] = {
                    "rows": 0,
                    "coverage": 0.0,
                    "applicable": False,
                    "status": "not_applicable_zero_rows",
                }
        examples.append(
            {
                "example_id": trace.example.example_id,
                "family_id": trace.example.family_id,
                "strata": rows,
            }
        )

    pooled_rows: dict[str, object] = {}
    for name, moments in pooled.items():
        if moments.rows > 0:
            pooled_rows[name] = {**moments.summary(), "applicable": True}
        else:
            pooled_rows[name] = {
                "rows": 0,
                "coverage": 0.0,
                "applicable": False,
                "status": "not_applicable_zero_rows",
            }

    family_members: dict[str, list[Mapping[str, object]]] = {}
    for row in examples:
        family_members.setdefault(str(row["family_id"]), []).append(row)
    families: list[dict[str, object]] = []
    for family_id in sorted(family_members):
        members = family_members[family_id]
        strata: dict[str, object] = {}
        for stratum in ("full", "graph_core", "causal_tail"):
            materialized = [
                frozen._mapping(
                    frozen._mapping(member["strata"], label="example strata")[
                        stratum
                    ],
                    label=f"{family_id}.{stratum}",
                )
                for member in members
            ]
            applicable = [
                value for value in materialized if value.get("applicable") is True
            ]
            if applicable:
                strata[stratum] = {
                    "rows": sum(int(value["rows"]) for value in applicable),
                    "example_count": len(applicable),
                    "family_example_count": len(members),
                    "example_coverage": len(applicable) / len(members),
                    "normalized_rmse": math.fsum(
                        float(value["normalized_rmse"]) for value in applicable
                    )
                    / len(applicable),
                    "cosine": math.fsum(
                        float(value["cosine"]) for value in applicable
                    )
                    / len(applicable),
                    "aggregation": "unweighted_nonempty_example_macro",
                    "applicable": True,
                }
            else:
                strata[stratum] = {
                    "rows": 0,
                    "example_count": 0,
                    "family_example_count": len(members),
                    "example_coverage": 0.0,
                    "aggregation": "unweighted_nonempty_example_macro",
                    "applicable": False,
                    "status": "not_applicable_zero_rows",
                }
        families.append({"family_id": family_id, "strata": strata})

    result: dict[str, object] = {
        "semantics": {
            "source": "native_h4_minus_incomplete_exact_x4_carrier_h4",
            "candidate": candidate_semantics,
            "full": "complete_h4_causal_support",
            "graph_core": "finite_lag_graph_target_support",
            "causal_tail": "complete_h4_support_outside_graph_core",
            "pooled_aggregation": "row_pooled",
            "family_aggregation": "unweighted_nonempty_example_macro",
            "truth_leaking_same_a_capacity_measurement": True,
        },
        "pooled": pooled_rows,
        "families": tuple(families),
        "per_example": tuple(examples),
    }
    result["gates"] = _executed_geometry_gates(result)
    return result


def _geometry_metric_gate(
    row: Mapping[str, object],
    *,
    nrmse_max: float,
    cosine_min: float,
) -> dict[str, object]:
    if row.get("applicable") is False:
        if row.get("rows") != 0:
            raise ValueError("N/A geometry stratum has nonzero rows")
        return {
            "applicable": False,
            "status": "not_applicable_zero_rows",
            "passed": True,
        }
    gated = frozen._metric_gate(
        row,
        nrmse_max=nrmse_max,
        cosine_min=cosine_min,
    )
    return {"applicable": True, **gated}


def _executed_geometry_gates(
    geometry: Mapping[str, object],
) -> dict[str, object]:
    strata = ("full", "graph_core", "causal_tail")
    pooled = frozen._mapping(geometry.get("pooled"), label="pooled geometry")
    pooled_gates = {
        name: _geometry_metric_gate(
            frozen._mapping(pooled.get(name), label=f"pooled {name}"),
            nrmse_max=0.05,
            cosine_min=0.995,
        )
        for name in strata
    }

    def grouped_gates(
        raw_rows: object,
        *,
        identity_key: str,
        label: str,
    ) -> tuple[dict[str, object], ...]:
        if not isinstance(raw_rows, (tuple, list)) or not raw_rows:
            raise ValueError(f"{label} geometry rows are empty")
        result: list[dict[str, object]] = []
        for raw in raw_rows:
            row = frozen._mapping(raw, label=f"{label} geometry row")
            identity = row.get(identity_key)
            row_strata = frozen._mapping(
                row.get("strata"),
                label=f"{label} geometry strata",
            )
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{label} geometry identity differs")
            gates = {
                name: _geometry_metric_gate(
                    frozen._mapping(
                        row_strata.get(name),
                        label=f"{identity}.{name}",
                    ),
                    nrmse_max=0.10,
                    cosine_min=0.99,
                )
                for name in strata
            }
            result.append(
                {
                    identity_key: identity,
                    "strata": gates,
                    "passed": all(
                        value["passed"] is True for value in gates.values()
                    ),
                }
            )
        return tuple(result)

    family_gates = grouped_gates(
        geometry.get("families"),
        identity_key="family_id",
        label="family",
    )
    example_gates = grouped_gates(
        geometry.get("per_example"),
        identity_key="example_id",
        label="example",
    )
    return {
        "thresholds": {
            "pooled_normalized_rmse_max": 0.05,
            "pooled_cosine_min": 0.995,
            "every_nonempty_family_stratum_normalized_rmse_max": 0.10,
            "every_nonempty_family_stratum_cosine_min": 0.99,
            "every_nonempty_example_stratum_normalized_rmse_max": 0.10,
            "every_nonempty_example_stratum_cosine_min": 0.99,
            "zero_row_strata": "explicit_not_applicable_and_neutral",
        },
        "pooled": pooled_gates,
        "families": family_gates,
        "per_example": example_gates,
        "passed": all(value["passed"] is True for value in pooled_gates.values())
        and all(value["passed"] is True for value in family_gates)
        and all(value["passed"] is True for value in example_gates),
    }


def _behavioral_family_gates(
    summary: Mapping[str, object],
) -> dict[str, object]:
    established = frozen._mapping(
        summary.get("gates"),
        label="behavioral aggregate/prompt gates",
    )
    family_summary = frozen._mapping(
        summary.get("family_summary"),
        label="behavioral family summary",
    )
    raw_families = family_summary.get("families")
    if not isinstance(raw_families, (tuple, list)) or not raw_families:
        raise ValueError("behavioral family summary is empty")
    families: list[dict[str, object]] = []
    for raw in raw_families:
        family = frozen._mapping(raw, label="behavioral family")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("behavioral family identity differs")
        gates = ESTABLISHED_SHADOW_FIDELITY_GATES.evaluate(
            delta_nll_per_token=float(family["delta_nll_per_token"]),
            top1_agreement_to_source=float(
                family["top1_agreement_to_source"]
            ),
            source_to_candidate_kl_per_token=float(
                family["source_to_candidate_kl_per_token"]
            ),
            per_prompt_p90_absolute_delta_nll_per_token=float(
                family["per_prompt_p90_absolute_delta_nll_per_token"]
            ),
            per_prompt_p10_top1_agreement_to_source=float(
                family["per_prompt_p10_top1_agreement_to_source"]
            ),
        )
        families.append(
            {"family_id": family_id, "gates": gates, "passed": gates["passed"]}
        )
    return {
        "established_aggregate_and_prompt": dict(established),
        "every_nonempty_family": tuple(families),
        "passed": established.get("passed") is True
        and all(row["passed"] is True for row in families),
    }


def classify_projection_ladder_arm(
    *,
    fit_weighting: ProjectionFitWeighting,
    rank: int,
    identity_validated: bool,
    exact_h4_ceiling: Mapping[str, object],
    support_integrity: Mapping[str, object],
    boundary_geometry: Mapping[str, object],
    ordinary_behavioral: Mapping[str, object],
    support_behavioral: Mapping[str, object],
    graph_core_behavioral: Mapping[str, object],
    causal_tail_behavioral: Mapping[str, object],
) -> dict[str, object]:
    if fit_weighting not in _WEIGHTINGS:
        raise ValueError("projection arm fit weighting differs")
    if type(rank) is not int or rank <= 0:
        raise ValueError("projection arm rank must be positive")
    if type(identity_validated) is not bool:
        raise TypeError("identity_validated must be boolean")
    behavioral = {
        "ordinary": _behavioral_family_gates(ordinary_behavioral),
        "complete_h4_support": _behavioral_family_gates(support_behavioral),
        "graph_core": _behavioral_family_gates(graph_core_behavioral),
        "causal_tail": _behavioral_family_gates(causal_tail_behavioral),
    }
    axes = {
        "frozen_exact_h4_identity": identity_validated,
        "live_exact_h4_frozen_ceiling": exact_h4_ceiling.get("passed") is True,
        "complete_h4_support_integrity": support_integrity.get("passed") is True,
        "boundary_geometry": frozen._mapping(
            boundary_geometry.get("gates"),
            label="boundary geometry gates",
        ).get("passed")
        is True,
        "ordinary_behavioral_fidelity": behavioral["ordinary"]["passed"] is True,
        "complete_h4_support_behavioral_fidelity": (
            behavioral["complete_h4_support"]["passed"] is True
        ),
        "graph_core_behavioral_fidelity": behavioral["graph_core"]["passed"] is True,
        "causal_tail_behavioral_fidelity": (
            behavioral["causal_tail"]["passed"] is True
        ),
    }
    passed = all(axes.values())
    return {
        "fit_weighting": fit_weighting,
        "rank": rank,
        "classifier_axes": tuple(axes),
        "pass_pattern": "".join(str(int(value)) for value in axes.values()),
        "arm_passes": axes,
        "behavioral_gate_detail": behavioral,
        "classification": (
            "complete_h4_projection_oracle_arm_validated"
            if passed
            else "complete_h4_projection_oracle_arm_insufficient"
        ),
        "later_lofo_fitting_authorized": passed,
        "generator_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }


def _select_projection_ladder(
    comparisons: Mapping[str, Mapping[str, object]],
    *,
    ranks: Sequence[int] = _RANK_GRID,
    weightings: Sequence[ProjectionFitWeighting] = _WEIGHTINGS,
) -> dict[str, object]:
    normalized_ranks = tuple(ranks)
    normalized_weightings = tuple(weightings)
    by_weighting: dict[str, list[tuple[int, str, bool]]] = {
        weighting: [] for weighting in normalized_weightings
    }
    for arm_id, row in comparisons.items():
        weighting = str(row["fit_weighting"])
        if weighting not in by_weighting:
            raise ValueError("ladder comparison has an unknown weighting")
        by_weighting[weighting].append(
            (int(row["rank"]), arm_id, row.get("later_lofo_fitting_authorized") is True)
        )
    stable: list[tuple[str, int, str]] = []
    per_basis: dict[str, str | None] = {}
    for weighting in normalized_weightings:
        rows = sorted(by_weighting[weighting])
        if tuple(row[0] for row in rows) != normalized_ranks:
            raise ValueError("ladder comparisons do not cover the locked rank grid")
        stable_rows = [
            (rank, arm_id)
            for index, (rank, arm_id, passed) in enumerate(rows)
            if passed and all(later[2] for later in rows[index:])
        ]
        per_basis[weighting] = None if not stable_rows else stable_rows[0][1]
        if stable_rows:
            stable.append((weighting, *stable_rows[0]))
    selected = sorted(
        stable,
        key=lambda row: (
            row[1],
            0 if row[0] == "unweighted" else 1,
            row[2],
        ),
    )
    return {
        "selection_rule": (
            "smallest_rank_with_all_larger_same_basis_ranks_passing_then_"
            "unweighted_then_lexical_arm_id"
        ),
        "per_basis_smallest_stable_passing_arm": per_basis,
        "overall_stable_passing_rank": None if not selected else selected[0][1],
        "selected_arm": None if not selected else selected[0][2],
        "any_oracle_arm_passed": bool(selected),
        "later_lofo_fitting_authorized": bool(selected),
        "generator_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }


def _normalize_tail_informed_factorial_comparisons(
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Turn legacy arm passes into capacity candidates without authority."""

    normalized: dict[str, dict[str, object]] = {}
    for arm_id, raw in comparisons.items():
        comparison = dict(raw)
        capacity_passed = (
            comparison.get("later_lofo_fitting_authorized") is True
        )
        comparison["factorial_capacity_gate_passed"] = capacity_passed
        comparison[
            "frozen_basis_one_pass_carrier_transfer_candidate_if_stable_suffix"
        ] = capacity_passed
        comparison["later_lofo_fitting_authorized"] = False
        comparison.pop(
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized",
            None,
        )
        normalized[arm_id] = comparison
    return normalized


def _select_tail_informed_factorial(
    comparisons: Mapping[str, Mapping[str, object]],
    *,
    tail_rank: int,
) -> dict[str, object]:
    if tail_rank != 17:
        raise ValueError("locked A16 factorial requires numerical tail rank rT=17")
    branch_grids = {
        "global_unweighted": (192, 224, 256, 320),
        "tail_informed": (209, 224, 256, 320),
    }
    branch_prefixes = {
        "global_unweighted": "unweighted.rank",
        "tail_informed": "tail_informed.rank",
    }
    expected_arm_ids = {
        f"{branch_prefixes[branch]}{rank}"
        for branch, ranks in branch_grids.items()
        for rank in ranks
    }
    if set(comparisons) != expected_arm_ids:
        raise ValueError("factorial comparisons must contain exactly eight arms")
    stable: list[tuple[int, str, str]] = []
    per_branch: dict[str, str | None] = {}
    for branch, ranks in branch_grids.items():
        prefix = branch_prefixes[branch]
        rows: list[tuple[int, str, bool]] = []
        for rank in ranks:
            arm_id = f"{prefix}{rank}"
            row = comparisons.get(arm_id)
            if row is None or int(row.get("rank", -1)) != rank:
                raise ValueError("factorial comparisons do not cover a locked branch")
            rows.append(
                (
                    rank,
                    arm_id,
                    row.get("factorial_capacity_gate_passed") is True,
                )
            )
        stable_rows = [
            (rank, arm_id)
            for index, (rank, arm_id, passed) in enumerate(rows)
            if passed and all(later[2] for later in rows[index:])
        ]
        per_branch[branch] = None if not stable_rows else stable_rows[0][1]
        if stable_rows:
            stable.append((stable_rows[0][0], branch, stable_rows[0][1]))
    selected = sorted(
        stable,
        key=lambda row: (
            row[0],
            0 if row[1] == "global_unweighted" else 1,
            row[2],
        ),
    )
    return {
        "selection_rule": (
            "smallest_total_rank_with_all_larger_same_branch_ranks_passing_"
            "then_global_unweighted_then_lexical_arm_id"
        ),
        "per_branch_smallest_stable_passing_arm": per_branch,
        "overall_stable_passing_rank": None if not selected else selected[0][0],
        "selected_arm": None if not selected else selected[0][2],
        "any_oracle_arm_passed": bool(selected),
        "factorial_capacity_gate_passed": bool(selected),
        "frozen_basis_one_pass_carrier_transfer_oracle_authorized": bool(
            selected
        ),
        "later_lofo_fitting_authorized": False,
        "generator_authorized": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }


def _expected_resources(*, prompt_count: int, arm_count: int) -> dict[str, int]:
    if type(prompt_count) is not int or prompt_count <= 0:
        raise ValueError("prompt_count must be positive")
    if type(arm_count) is not int or arm_count <= 0:
        raise ValueError("arm_count must be positive")
    collect = prompt_count * 5
    evaluation_shadow = prompt_count * 3
    projection = prompt_count * arm_count
    ceiling = prompt_count
    return {
        "collect_model_forward_count": collect,
        "evaluation_shadow_model_forward_count": evaluation_shadow,
        "projection_arm_model_forward_count": projection,
        "exact_h4_ceiling_model_forward_count": ceiling,
        "evaluation_model_forward_count": evaluation_shadow + projection + ceiling,
        "total_model_forward_count": collect + evaluation_shadow + projection + ceiling,
        "backward_count": prompt_count,
    }


def _full_vocabulary_logit_peak_accounting() -> dict[str, object]:
    """Describe phase-specific full-vocabulary tensors retained at peak.

    Collection retains the two returned shadow logits while the native and
    partial exact-X4 pair replays overlap.  Evaluation retains those same two
    shadow logits plus exactly one ceiling/projection-arm result.
    """

    return {
        "collection_simultaneously_live_full_vocabulary_logit_tensor_peak": 4,
        "collection_simultaneously_live_logit_roles": (
            "authoritative_shadow",
            "candidate_shadow",
            "native_complete_h4_pair_replay",
            "partial_exact_x4_nll_pair_replay",
        ),
        "evaluation_simultaneously_live_full_vocabulary_logit_tensor_peak": 3,
        "evaluation_simultaneously_live_logit_roles": (
            "authoritative_shadow",
            "candidate_shadow",
            "one_ceiling_or_projection_correction",
        ),
        "experiment_simultaneously_live_full_vocabulary_logit_tensor_peak": 4,
    }


def _fit_manifest_receipt(
    sequences: Sequence[CompleteH4ProjectionFitSequence],
    bases: Mapping[ProjectionFitWeighting, CompleteH4ProjectionBasis],
    *,
    weightings: Sequence[ProjectionFitWeighting] = _WEIGHTINGS,
) -> dict[str, object]:
    normalized_weightings = tuple(weightings)
    ordered = tuple(sorted(sequences, key=lambda row: (row.family_id, row.example_id)))
    payload = {
        "schema": "fisher_graph.complete_h4_projection_ladder_fit_manifest",
        "format_version": 1,
        "ordering": "family_id_then_example_id",
        "example_ids": tuple(row.example_id for row in ordered),
        "family_ids_by_example": tuple(row.family_id for row in ordered),
        "sequence_sha256s": tuple(row.sequence_sha256 for row in ordered),
        "row_counts": tuple(row.row_count for row in ordered),
        "width": ordered[0].width,
        "all_sequences_have_gradients": all(row.has_gradients for row in ordered),
    }
    if set(bases) != set(normalized_weightings):
        raise ValueError("fit manifest bases differ from declared weightings")
    for weighting in normalized_weightings:
        basis = bases[weighting]
        if (
            basis.source_example_ids != payload["example_ids"]
            or basis.source_sequence_sha256s != payload["sequence_sha256s"]
            or basis.source_family_ids
            != tuple(sorted(set(payload["family_ids_by_example"])))
        ):
            raise ValueError("fitted bases did not consume the identical manifest")
    result = {
        **payload,
        "fit_manifest_sha256": _domain_sha256(
            payload,
            domain=_FIT_MANIFEST_DOMAIN,
        ),
    }
    if normalized_weightings == _WEIGHTINGS:
        result["both_weightings_consumed_identical_sequence_receipts"] = True
    else:
        result["all_weightings_consumed_identical_sequence_receipts"] = True
    return result


def _basis_numerical_diagnostics(
    sequences: Sequence[CompleteH4ProjectionFitSequence],
    basis: CompleteH4ProjectionBasis,
    *,
    ranks: Sequence[int] = _RANK_GRID,
) -> dict[str, object]:
    covariance, _unweighted, _fisher = (
        projection_math._family_example_macro_moments(
            tuple(sequences),
            fit_weighting=basis.fit_weighting,
        )
    )
    directions = basis.basis_tensor(ordering="euclidean")
    eigenvalues = torch.tensor(
        basis.residual_eigenvalues,
        dtype=torch.float64,
    )
    eigenpair_error = covariance @ directions.T - directions.T * eigenvalues
    residual_norms = torch.linalg.vector_norm(eigenpair_error, dim=0)
    residual_bound = float(residual_norms.max())
    # The covariance is PSD and the retained order begins at its largest
    # eigenvalue, so this avoids an accidental third eigendecomposition.
    covariance_norm = float(basis.residual_eigenvalues[0])
    normalized_bound = residual_bound / max(
        covariance_norm,
        torch.finfo(torch.float64).tiny,
    )
    gram = directions @ directions.T
    orthonormal_error = float(
        (
            gram
            - torch.eye(
                basis.max_rank,
                dtype=torch.float64,
            )
        )
        .abs()
        .max()
    )
    cutoffs: list[dict[str, object]] = []
    for rank in ranks:
        cutoff = float(basis.residual_eigenvalues[rank - 1])
        following = (
            float(basis.residual_eigenvalues[rank])
            if rank < basis.max_rank
            else float(basis.next_residual_eigenvalue)
        )
        gap = max(0.0, cutoff - following)
        stable = gap > 10.0 * residual_bound
        cutoffs.append(
            {
                "rank": rank,
                "cutoff_eigenvalue": cutoff,
                "next_eigenvalue": following,
                "spectral_gap": gap,
                "minimum_required_gap": 10.0 * residual_bound,
                "numerically_selectable": stable,
            }
        )
    result = {
        "fit_weighting": basis.fit_weighting,
        "dtype": "float64",
        "device": "cpu",
        "eigenpair_max_residual_l2": residual_bound,
        "eigenpair_max_residual_relative_to_covariance_spectral_norm": (
            normalized_bound
        ),
        "basis_orthonormal_max_abs_error": orthonormal_error,
        "cutoff_stability_rule": (
            "spectral_gap_strictly_greater_than_10x_max_eigenpair_residual"
        ),
        "cutoffs": tuple(cutoffs),
        "all_cutoffs_numerically_selectable": all(
            row["numerically_selectable"] is True for row in cutoffs
        ),
    }
    if (
        not math.isfinite(residual_bound)
        or not math.isfinite(normalized_bound)
        or not math.isfinite(orthonormal_error)
        or orthonormal_error > 1.0e-9
        or result["all_cutoffs_numerically_selectable"] is not True
    ):
        raise RuntimeError("projection basis eigensolver/cutoff stability failed")
    return result


def _basis_projector_overlap(
    tilted: CompleteH4ProjectionBasis,
    unweighted: CompleteH4ProjectionBasis,
    *,
    ranks: Sequence[int] = _RANK_GRID,
) -> dict[str, object]:
    if tilted.width != unweighted.width:
        raise ValueError("basis projector widths differ")
    left = tilted.basis_tensor(ordering="euclidean")
    right = unweighted.basis_tensor(ordering="euclidean")
    rows: list[dict[str, object]] = []
    for rank in ranks:
        singular = torch.linalg.svdvals(left[:rank] @ right[:rank].T).clamp(
            min=0.0,
            max=1.0,
        )
        angles = torch.acos(singular)
        squared_overlap = float(singular.square().sum())
        rows.append(
            {
                "rank": rank,
                "normalized_projector_overlap": squared_overlap / rank,
                "minimum_principal_cosine": float(singular.min()),
                "mean_principal_angle_radians": float(angles.mean()),
                "maximum_principal_angle_radians": float(angles.max()),
                "projector_frobenius_distance": math.sqrt(
                    max(0.0, 2.0 * rank - 2.0 * squared_overlap)
                ),
            }
        )
    return {
        "definition": "singular_values_of_tilted_D_times_unweighted_D_transpose",
        "diagnostic_only": True,
        "rank_rows": tuple(rows),
    }


def _validate_monotonic_offline_geometry(
    geometry: Mapping[str, object],
    *,
    ranks: Sequence[int] = _RANK_GRID,
) -> dict[str, object]:
    normalized_ranks = tuple(ranks)
    raw_rows = geometry.get("rank_rows")
    if not isinstance(raw_rows, (tuple, list)) or len(raw_rows) != len(
        normalized_ranks
    ):
        raise ValueError("offline projection rank geometry differs")
    rows = [frozen._mapping(row, label="offline rank row") for row in raw_rows]
    if tuple(int(row["rank"]) for row in rows) != normalized_ranks:
        raise ValueError("offline projection rank grid differs")
    retention_metrics = (
        "family_balanced_residual_energy_retention",
        "row_weighted_residual_energy_retention",
    )
    rmse_metrics = (
        "family_balanced_residual_rmse",
        "row_weighted_residual_rmse",
    )
    tolerance = 1.0e-12
    monotonic = all(
        float(rows[index + 1][name]) + tolerance >= float(rows[index][name])
        for name in retention_metrics
        for index in range(len(rows) - 1)
    ) and all(
        float(rows[index + 1][name]) <= float(rows[index][name]) + tolerance
        for name in rmse_metrics
        for index in range(len(rows) - 1)
    )
    if not monotonic:
        raise RuntimeError("nested projection geometry is not monotonic")
    derived: list[dict[str, object]] = []
    for row in rows:
        residual = row.get("fisher_absolute_first_order_residual_coupling")
        error = row.get("fisher_absolute_first_order_error_coupling")
        ratio = None
        if isinstance(residual, (int, float)) and isinstance(error, (int, float)):
            ratio = (float(error) / max(float(residual), 1.0e-30)) ** 2
        derived.append(
            {
                "rank": int(row["rank"]),
                "squared_absolute_first_order_error_ratio": ratio,
            }
        )
    return {
        "nested_rank_energy_and_rmse_monotonic": True,
        "monotonicity_tolerance": tolerance,
        "derived_rank_diagnostics": tuple(derived),
    }


def _load_frozen_rank64_projection(path: Path | str) -> dict[str, object]:
    source = Path(path)
    if frozen._file_sha256(source) != _EXPECTED_PRIOR_FILE_SHA256:
        raise ValueError("frozen rank64 projection report file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    report = dict(frozen._mapping(raw, label="frozen rank64 projection report"))
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    if (
        claimed != _EXPECTED_PRIOR_REPORT_SHA256
        or frozen._json_sha256(payload, domain=frozen._REPORT_DOMAIN) != claimed
        or report.get("schema")
        != "fisher_graph.gemma3_l3_l4_complete_h4_projection_development"
        or report.get("format_version") != 1
        or report.get("role")
        != "reused_calibration_a_truth_leaking_complete_h4_capacity_screen"
    ):
        raise ValueError("frozen rank64 projection report identity differs")
    runtime_basis = frozen._mapping(
        report.get("runtime_projection_basis"),
        label="frozen runtime projection basis",
    )
    comparison = frozen._mapping(
        report.get("comparison"),
        label="frozen rank64 comparison",
    )
    safety = frozen._mapping(report.get("safety"), label="frozen safety")
    if (
        runtime_basis.get("artifact_sha256")
        != _EXPECTED_TILTED_RANK64_EXECUTION_BASIS_ARTIFACT_SHA256
        or runtime_basis.get("rank") != 64
        or runtime_basis.get("ordering")
        != "descending_fisher_tilted_residual_eigenvalue"
        or comparison.get("classification")
        != "rank64_h4_projection_insufficient"
        or safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_logits") is not False
        or safety.get("calibration_b_opened") is not False
        or safety.get("validation_opened") is not False
        or safety.get("test_opened") is not False
    ):
        raise ValueError("frozen rank64 projection premise differs")
    raw_receipts = report.get("correction_receipts")
    if not isinstance(raw_receipts, (tuple, list)) or len(raw_receipts) != 16:
        raise ValueError("frozen rank64 correction receipts differ")
    logits_by_example: dict[str, str] = {}
    for raw_receipt in raw_receipts:
        receipt = frozen._mapping(raw_receipt, label="rank64 correction receipt")
        arm = frozen._mapping(receipt.get("arm"), label="rank64 correction arm")
        example_id = receipt.get("example_id")
        if (
            not isinstance(example_id, str)
            or example_id in logits_by_example
            or arm.get("role") != "projection_oracle"
            or arm.get("projection_rank") != 64
            or arm.get("projection_basis_artifact_sha256")
            != _EXPECTED_TILTED_RANK64_EXECUTION_BASIS_ARTIFACT_SHA256
            or not isinstance(arm.get("logits_sha256"), str)
        ):
            raise ValueError("frozen rank64 correction arm differs")
        logits_by_example[example_id] = str(arm["logits_sha256"])
    return {
        "file": str(source),
        "file_sha256": _EXPECTED_PRIOR_FILE_SHA256,
        "report_sha256": _EXPECTED_PRIOR_REPORT_SHA256,
        "runtime_projection_basis": dict(runtime_basis),
        "ordinary_behavioral": report["ordinary_behavioral"],
        "support_behavioral": report["complete_h4_support_behavioral"],
        "boundary_geometry": report["rank64_boundary_geometry"],
        "logits_by_example": logits_by_example,
    }


def _load_parent_projection_basis_rank_ladder(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
) -> dict[str, object]:
    source = Path(path)
    if frozen._file_sha256(source) != expected_file_sha256:
        raise ValueError("parent complete-H4 basis/rank ladder file differs")
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    report = dict(frozen._mapping(raw, label="parent basis/rank ladder report"))
    payload = dict(report)
    claimed = payload.pop("report_sha256", None)
    status = frozen._mapping(
        report.get("scientific_status"),
        label="parent basis/rank ladder status",
    )
    safety = frozen._mapping(
        report.get("safety"),
        label="parent basis/rank ladder safety",
    )
    arms = frozen._mapping(
        report.get("arms"),
        label="parent basis/rank ladder arms",
    )
    if (
        claimed != expected_report_sha256
        or frozen._json_sha256(payload, domain=_REPORT_DOMAIN) != claimed
        or report.get("schema") != _SCHEMA
        or report.get("format_version") != _FORMAT_VERSION
        or status.get("development_capacity_ladder_complete") is not True
        or status.get("all_eight_arms_executed") is not True
        or safety.get("truth_leaking_same_a_capacity_oracle") is not True
        or safety.get("calibration_b_opened") is not False
        or safety.get("validation_opened") is not False
        or safety.get("test_opened") is not False
        or "unweighted.rank192" not in arms
    ):
        raise ValueError("parent complete-H4 basis/rank ladder identity differs")
    return {
        "file": str(source),
        "file_sha256": expected_file_sha256,
        "report_sha256": expected_report_sha256,
        "report": report,
        "unweighted_rank192": arms["unweighted.rank192"],
    }


def _validate_tilted_rank64_regression(
    *,
    prior: Mapping[str, object],
    spec: _ProjectionArmSpec,
    boundary_geometry: Mapping[str, object],
    ordinary_behavioral: Mapping[str, object],
    support_behavioral: Mapping[str, object],
    logits_by_example: Mapping[str, str],
) -> dict[str, object]:
    if (
        spec.fit_weighting != "fisher_alignment_tilted"
        or spec.rank != 64
        or spec.execution_basis_artifact_sha256
        != _EXPECTED_TILTED_RANK64_EXECUTION_BASIS_ARTIFACT_SHA256
    ):
        raise ValueError("tilted rank64 prefix differs from frozen execution basis")
    prior_boundary = frozen._mapping(
        prior.get("boundary_geometry"),
        label="prior boundary geometry",
    )
    comparable_boundary = {
        name: boundary_geometry[name]
        for name in ("semantics", "pooled", "families", "gates")
    }
    if (
        frozen._canonical_json_bytes(comparable_boundary)
        != frozen._canonical_json_bytes(prior_boundary)
        or frozen._canonical_json_bytes(ordinary_behavioral)
        != frozen._canonical_json_bytes(prior["ordinary_behavioral"])
        or frozen._canonical_json_bytes(support_behavioral)
        != frozen._canonical_json_bytes(prior["support_behavioral"])
        or frozen._canonical_json_bytes(dict(logits_by_example))
        != frozen._canonical_json_bytes(prior["logits_by_example"])
    ):
        raise ValueError("tilted rank64 live regression differs from frozen V1")
    return {
        "matched": True,
        "prior_file_sha256": prior["file_sha256"],
        "prior_report_sha256": prior["report_sha256"],
        "execution_basis_artifact_sha256": (
            _EXPECTED_TILTED_RANK64_EXECUTION_BASIS_ARTIFACT_SHA256
        ),
        "full_scalar_behavior_and_geometry_matched": True,
        "all_16_logits_hashes_matched": True,
    }


def _validate_unweighted_rank192_parent_regression(
    *,
    parent: Mapping[str, object],
    spec: _ProjectionArmSpec,
    exact_float64_geometry: Mapping[str, object],
    executed_cast_once_geometry: Mapping[str, object],
    behavioral_ledgers: Mapping[str, object],
    support_integrity: Mapping[str, object],
    comparison: Mapping[str, object],
    projection_macs: int,
    live_correction_receipts: Sequence[Mapping[str, object]],
    logits_by_example: Mapping[str, str],
) -> dict[str, object]:
    if (
        spec.fit_weighting != "unweighted"
        or spec.rank != 192
        or spec.execution_basis_artifact_sha256
        != _EXPECTED_UNWEIGHTED_RANK192_EXECUTION_BASIS_ARTIFACT_SHA256
    ):
        raise ValueError("unweighted rank192 prefix differs from parent ladder")
    parent_arm = frozen._mapping(
        parent.get("unweighted_rank192"),
        label="parent unweighted rank192 arm",
    )
    parent_lineage = frozen._mapping(
        parent_arm.get("fit_to_prefix_lineage"),
        label="parent unweighted rank192 lineage",
    )
    raw_receipts = parent_arm.get("correction_receipts")
    if (
        parent_arm.get("fit_weighting") != "unweighted"
        or parent_arm.get("rank") != 192
        or parent_lineage.get("execution_basis_artifact_sha256")
        != spec.execution_basis_artifact_sha256
        or not isinstance(raw_receipts, (tuple, list))
        or len(raw_receipts) != 16
    ):
        raise ValueError("parent unweighted rank192 basis or receipts differ")
    invariant_receipt_regression = _validate_u192_parent_invariant_receipts(
        parent_receipts=raw_receipts,
        live_receipts=live_correction_receipts,
    )
    parent_logits: dict[str, str] = {}
    for raw_receipt in raw_receipts:
        receipt = frozen._mapping(
            raw_receipt,
            label="parent unweighted rank192 receipt",
        )
        arm = frozen._mapping(
            receipt.get("arm"),
            label="parent unweighted rank192 correction arm",
        )
        example_id = receipt.get("example_id")
        logits_sha256 = arm.get("logits_sha256")
        if (
            not isinstance(example_id, str)
            or not example_id
            or example_id != example_id.strip()
            or not isinstance(logits_sha256, str)
            or len(logits_sha256) != 64
            or example_id in parent_logits
        ):
            raise ValueError("parent unweighted rank192 logit receipt differs")
        parent_logits[example_id] = logits_sha256
    comparisons = (
        (
            parent_arm.get("exact_float64_geometry"),
            exact_float64_geometry,
        ),
        (
            parent_arm.get("executed_cast_once_geometry"),
            executed_cast_once_geometry,
        ),
        (parent_arm.get("behavioral_ledgers"), behavioral_ledgers),
        (parent_arm.get("support_integrity"), support_integrity),
        (parent_arm.get("comparison"), comparison),
        (parent_arm.get("projection_macs_over_a16_support"), projection_macs),
        (parent_logits, dict(logits_by_example)),
    )
    if any(
        frozen._canonical_json_bytes(expected)
        != frozen._canonical_json_bytes(observed)
        for expected, observed in comparisons
    ):
        raise RuntimeError("unweighted rank192 live regression differs from parent V1")
    return {
        "parent_file_sha256": parent["file_sha256"],
        "parent_report_sha256": parent["report_sha256"],
        "execution_basis_artifact_sha256": (
            spec.execution_basis_artifact_sha256
        ),
        "exact_and_executed_geometry_matched": True,
        "all_four_behavior_ledgers_matched": True,
        "all_scalar_arm_metrics_matched": True,
        "all_16_per_prompt_invariant_receipts_matched": True,
        "per_prompt_invariant_receipt_regression": (
            invariant_receipt_regression
        ),
        "all_16_logits_hashes_matched": True,
    }


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    parts = destination.parts
    if (
        destination.is_absolute()
        or destination.suffix != ".json"
        or len(parts) < 2
        or parts[0] != ".local-runs"
        or parts.count(".local-runs") != 1
        or ".." in parts
    ):
        raise ValueError(
            "H4 basis/rank ladder output must be JSON under .local-runs "
            "as a lexical repo-relative path without traversal or nesting"
        )
    return destination


def _publish(
    report: dict[str, object],
    *,
    output: Path,
    report_domain: bytes = _REPORT_DOMAIN,
) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(
            report,
            domain=report_domain,
        )
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": frozen._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def run_gemma3_l3_l4_complete_h4_projection_basis_rank_ladder(
    *,
    fit_source_artifact_path: Path | str = frozen.DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = frozen.DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = frozen.DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = frozen.DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = frozen.DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        frozen.DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    panel_path: Path | str = frozen.DEFAULT_PANEL,
    rank64_x4_baseline_path: Path | str = frozen.DEFAULT_RANK64_X4_BASELINE,
    complete_h4_identity_path: Path | str = frozen.DEFAULT_COMPLETE_H4_IDENTITY,
    rank64_projection_baseline_path: Path | str = (
        DEFAULT_RANK64_PROJECTION_BASELINE
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = frozen.DEFAULT_MAX_LENGTH,
    _tail_informed_factorial: _TailInformedFactorialConfig | None = None,
) -> dict[str, object]:
    """Run all eight same-A complete-H4 projection capacity arms.

    The function is deliberately fail-closed.  Authentication, integrity,
    finite-number, exact-ceiling, or accounting failures raise immediately;
    only fully executed, numerically valid arms can be classified as having
    insufficient capacity.
    """

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite complete-H4 ladder report")
    if type(max_length) is not int or max_length != frozen.DEFAULT_MAX_LENGTH:
        raise ValueError(
            f"max_length must equal locked value {frozen.DEFAULT_MAX_LENGTH}"
        )

    baseline = frozen._load_rank64_x4_baseline(rank64_x4_baseline_path)
    identity = frozen._load_complete_h4_identity(complete_h4_identity_path)
    prior_rank64 = _load_frozen_rank64_projection(
        rank64_projection_baseline_path
    )
    parent_ladder = (
        None
        if _tail_informed_factorial is None
        else _load_parent_projection_basis_rank_ladder(
            _tail_informed_factorial.parent_ladder_path,
            expected_file_sha256=(
                _tail_informed_factorial.parent_ladder_file_sha256
            ),
            expected_report_sha256=(
                _tail_informed_factorial.parent_ladder_report_sha256
            ),
        )
    )
    examples, panel_receipt = frozen._load_panel(panel_path)
    if (
        frozen._canonical_json_bytes(panel_receipt)
        != frozen._canonical_json_bytes(baseline["panel"])
        or frozen._canonical_json_bytes(panel_receipt)
        != frozen._canonical_json_bytes(identity["panel"])
    ):
        raise ValueError("live A16 panel differs from authenticated reports")

    fit_source = frozen.load_gemma3_spectral_source(
        fit_source_artifact_path,
        expected_file_sha256=frozen.DEFAULT_INTERIOR_ARTIFACT_SHA256,
        expected_report_sha256=frozen.DEFAULT_INTERIOR_REPORT_SHA256,
        expected_origins=frozen.INTERIOR_ORIGINS,
    )
    parent = frozen.load_gemma3_graph_wavelet_candidate(
        parent_artifact_path,
        expected_artifact_sha256=frozen.DEFAULT_PARENT_ARTIFACT_SHA256,
        expected_tensor_file_sha256=(
            frozen.DEFAULT_PARENT_TENSOR_FILE_SHA256
        ),
        expected_report_sha256=frozen.DEFAULT_PARENT_REPORT_SHA256,
    )
    candidate = frozen.load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate(
        candidate_artifact_path,
        expected_artifact_sha256=frozen.DEFAULT_FROZEN_ARTIFACT_SHA256,
        expected_tensor_file_sha256=frozen.DEFAULT_FROZEN_TENSOR_FILE_SHA256,
        expected_report_sha256=frozen.DEFAULT_FROZEN_REPORT_SHA256,
    )
    basis_package = frozen.load_gemma3_l3_l4_basis_package(
        basis_package_path,
        expected_file_sha256=frozen.DEFAULT_BASIS_PACKAGE_FILE_SHA256,
        expected_payload_sha256=frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    )
    plan, plan_receipt = frozen.build_rank64_global_svd_plan(
        fit_source,
        parent,
    )
    if frozen._canonical_json_bytes(plan_receipt) != frozen._canonical_json_bytes(
        baseline["rank64_plan"]
    ):
        raise ValueError("rebuilt rank64 X4 plan differs from corrected V2")
    arm_receipt = frozen._mapping(
        baseline["rank64_arm_receipt"],
        label="rank64 arm",
    )
    common_binding = frozen._mapping(
        arm_receipt.get("common_binding"),
        label="rank64 common binding",
    )
    if (
        arm_receipt.get("artifact_sha256")
        != frozen._EXPECTED_RANK64_ARM_SHA256
        or common_binding.get("signed_g8_candidate_artifact_sha256")
        != candidate.artifact_sha256
        or common_binding.get("fit_response_tensor_file_sha256")
        != fit_source.file_sha256
        or common_binding.get("parent_graph_wavelet_artifact_sha256")
        != parent.artifact_sha256
        or common_binding.get("basis_package_payload_sha256")
        != frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
        or common_binding.get("panel_file_sha256")
        != panel_receipt["file_sha256"]
        or common_binding.get("max_length") != max_length
    ):
        raise ValueError("live rank64 arm inputs differ from corrected V2")

    protocol = frozen.default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    tokenizer, tokenizer_contract = (
        frozen._load_and_validate_frozen_local_tokenizer(protocol=protocol)
    )
    if (
        tokenizer_contract.get("configuration_sha256")
        != frozen._EXPECTED_TOKENIZER_CONFIGURATION_SHA256
        or tokenizer_contract.get("backend_serialized_sha256")
        != frozen._EXPECTED_TOKENIZER_INITIAL_BACKEND_SHA256
    ):
        raise ValueError("live tokenizer differs from corrected V2")
    tokenizer_integrity_check = frozen._frozen_tokenizer_integrity_check(
        tokenizer,
        tokenizer_contract,
    )

    model_metadata = candidate.model
    if (
        model_metadata.get("source_model_sha256")
        != frozen._EXPECTED_RAW_MODEL_SHA256
    ):
        raise ValueError("candidate raw model lineage differs")
    device = frozen.resolve_torch_device("cpu")
    cache = frozen.resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = frozen._load_local_gemma3_model_only(
        model_id=str(model_metadata["model_id"]),
        revision=str(model_metadata["resolved_commit"]),
        cache_dir=cache,
        device=device,
        dtype="float32",
    )
    adapter = frozen.Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != frozen._EXPECTED_RAW_MODEL_SHA256:
        raise ValueError("live raw Gemma differs from corrected V2")
    catalog = frozen.restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    switcher = frozen.PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {frozen._FACTORIZED_SCOPE: catalog.replacements},
    )

    traces: list[frozen._PromptTrace] = []
    collect_receipts: list[dict[str, object]] = []
    behavior_manifests: dict[str, dict[str, str]] = {
        "ordinary": {},
        "complete_h4_support": {},
        "graph_core": {},
        "causal_tail": {},
    }
    collect_forwards = 0
    backward_count = 0
    total_support_rows = 0
    total_graph_core_rows = 0
    total_causal_tail_rows = 0
    total_padding_difference_rows = 0
    retained_pair_tensor_bytes = 0
    largest_retained_pair_tensor_bytes = 0
    tail_informed_fit: object | None = None

    try:
        switcher.switch(frozen._FACTORIZED_SCOPE)
        factorized_model_sha256, factorized_execution_sha256 = (
            frozen._live_factorized_identity(adapter)
        )
        runtime = frozen.Gemma3L3L4ConditionalSpectralShadowRuntime(
            plan,
            basis_package,
            candidate_artifact_sha256=frozen._EXPECTED_RANK64_ARM_SHA256,
            candidate_method="global_svd_rank64_capacity_oracle",
            candidate_binding=candidate.binding,
            candidate_model=candidate.model,
            expected_plan_artifact_sha256=(
                frozen._EXPECTED_RANK64_PLAN_SHA256
            ),
            expected_basis_payload_sha256=(
                frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            expected_live_model_sha256=factorized_model_sha256,
            expected_adapter_execution_sha256=factorized_execution_sha256,
            analysis_device="cpu",
        )
        runtime_metadata = runtime.metadata()
        if (
            runtime_metadata.get("runtime_binding_sha256")
            != frozen._EXPECTED_RANK64_RUNTIME_SHA256
            or frozen._canonical_json_bytes(runtime_metadata)
            != frozen._canonical_json_bytes(baseline["runtime_binding"])
        ):
            raise ValueError("live rank64 runtime differs from corrected V2")

        frozen_receipts = frozen._mapping(
            identity["receipts"],
            label="identity receipts",
        )
        for example in sorted(examples, key=lambda value: value.example_id):
            tokenizer_integrity_check("before")
            model_inputs, supervised_indices, supervised_targets = (
                frozen._tokenize_one(
                    tokenizer,
                    example.prompt,
                    max_length=max_length,
                    model_input_device=device,
                )
            )
            tokenizer_integrity_check("after")
            with torch.inference_mode():
                shadow = runtime.execute_model_shadow(
                    adapter,
                    model_inputs,
                    arm="all_on",
                )
            pair = runtime.execute_complete_h4_pair(
                adapter,
                model_inputs,
                shadow,
                supervised_indices=frozen._supervised_grid_indices(
                    supervised_indices
                ),
                supervised_targets=supervised_targets.detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous(),
                ignore_index=-100,
            )
            if example.example_id not in frozen_receipts:
                raise ValueError("live example is absent from H4 identity report")
            frozen_receipt = frozen._mapping(
                frozen_receipts[example.example_id],
                label="frozen identity receipt",
            )
            if (
                frozen_receipt.get("family_id") != example.family_id
                or frozen_receipt.get("prompt_sha256")
                != frozen._prompt_sha256(example.prompt)
            ):
                raise ValueError("live prompt identity differs from H4 report")
            pair_metadata = frozen._validate_pair_against_frozen(
                pair,
                frozen_receipt=frozen_receipt,
            )
            support = pair.complete_h4_support_mask[0].detach().to(device="cpu")
            target = pair.target_affected_mask[0].detach().to(device="cpu")
            if bool((target & ~support).any()):
                raise RuntimeError("graph-core rows escaped complete-H4 support")
            support_indices = torch.nonzero(
                support,
                as_tuple=False,
            ).flatten().to(dtype=torch.int64)
            graph_core_rows = target.index_select(0, support_indices)
            device_support_indices = support_indices.to(pair.native_h4.device)
            residual_rows = pair.native_h4[0].index_select(
                0,
                device_support_indices,
            ).to(dtype=torch.float64) - pair.incomplete_h4[0].index_select(
                0,
                device_support_indices,
            ).to(dtype=torch.float64)
            gradient_rows = pair.h4_gradient[0].index_select(
                0,
                device_support_indices,
            )
            fit_sequence = CompleteH4ProjectionFitSequence(
                example_id=example.example_id,
                family_id=example.family_id,
                residual_rows=residual_rows,
                gradient_rows=gradient_rows,
            )
            traces.append(
                frozen._PromptTrace(
                    example=example,
                    prompt_sha256=frozen._prompt_sha256(example.prompt),
                    pair=pair,
                    fit_sequence=fit_sequence,
                    support_indices=support_indices,
                    graph_core_rows=graph_core_rows,
                )
            )

            supervised_cpu = supervised_indices.detach().to(
                device="cpu",
                dtype=torch.int64,
            )
            supervised_masks = {
                "ordinary": torch.ones_like(supervised_cpu, dtype=torch.bool),
                "complete_h4_support": support.index_select(0, supervised_cpu),
                "graph_core": target.index_select(0, supervised_cpu),
                "causal_tail": (support & ~target).index_select(
                    0,
                    supervised_cpu,
                ),
            }
            for ledger_name, mask in supervised_masks.items():
                if bool(mask.any()):
                    behavior_manifests[ledger_name][example.example_id] = (
                        example.family_id
                    )

            collect_receipts.append(
                {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "prompt_sha256": frozen._prompt_sha256(example.prompt),
                    "tokenized_tokens": int(model_inputs["input_ids"].shape[1]),
                    "supervised_tokens": int(supervised_indices.numel()),
                    "behavioral_ledger_tokens": {
                        name: int(mask.sum())
                        for name, mask in supervised_masks.items()
                    },
                    "model_inputs_sha256": pair.model_inputs_sha256,
                    "execution_grid_sha256": pair.execution_grid_sha256,
                    "shadow_result_artifact_sha256": (
                        pair.shadow_result_artifact_sha256
                    ),
                    "pair": pair_metadata,
                    "fit_sequence": fit_sequence.metadata(),
                }
            )
            collect_forwards += 5
            backward_count += 1
            total_support_rows += int(support.sum())
            total_graph_core_rows += int(graph_core_rows.sum())
            total_causal_tail_rows += int((~graph_core_rows).sum())
            total_padding_difference_rows += int(
                pair_metadata["incomplete_h4_difference_padding_rows"]
            )
            pair_tensor_bytes = sum(
                value.numel() * value.element_size()
                for value in (
                    pair.native_h4,
                    pair.incomplete_h4,
                    pair.h4_gradient,
                    pair.source_modes,
                    pair.logical_positions,
                    pair.valid_target_mask,
                    pair.source_eligible_mask,
                    pair.target_affected_mask,
                    pair.complete_h4_support_mask,
                )
            )
            retained_pair_tensor_bytes += pair_tensor_bytes
            largest_retained_pair_tensor_bytes = max(
                largest_retained_pair_tensor_bytes,
                pair_tensor_bytes,
            )
            del (
                shadow,
                model_inputs,
                residual_rows,
                gradient_rows,
                supervised_targets,
                supervised_masks,
            )

        if (
            len(traces) != 16
            or len({trace.example.family_id for trace in traces}) != 8
            or total_support_rows != frozen._EXPECTED_COMPLETE_H4_ROWS
            or total_graph_core_rows != frozen._EXPECTED_GRAPH_CORE_ROWS
            or total_causal_tail_rows != frozen._EXPECTED_CAUSAL_TAIL_ROWS
            or total_padding_difference_rows != 0
            or any(not manifest for manifest in behavior_manifests.values())
        ):
            raise ValueError("collected complete-H4 support differs from identity")

        fit_sequences = tuple(trace.fit_sequence for trace in traces)
        fit_weightings: tuple[ProjectionFitWeighting, ...] = (
            _WEIGHTINGS
            if _tail_informed_factorial is None
            else ("unweighted",)
        )
        fit_max_rank = (
            _MAX_RANK if _tail_informed_factorial is None else 320
        )
        offline_rank_grid = (
            _RANK_GRID
            if _tail_informed_factorial is None
            else (192, 224, 256, 320)
        )
        bases: dict[ProjectionFitWeighting, CompleteH4ProjectionBasis] = {}
        offline_geometry: dict[str, object] = {}
        offline_geometry_validation: dict[str, object] = {}
        basis_numerical_diagnostics: dict[str, object] = {}
        for weighting in fit_weightings:
            basis = fit_complete_h4_projection_basis(
                fit_sequences,
                max_rank=fit_max_rank,
                fit_weighting=weighting,
            )
            if (
                basis.max_rank != fit_max_rank
                or basis.width != _WIDTH
                or not basis.has_fisher
                or basis.fit_weighting != weighting
            ):
                raise ValueError("complete-H4 projection basis geometry differs")
            bases[weighting] = basis
            geometry = summarize_complete_h4_projection_geometry(
                fit_sequences,
                basis,
                ranks=offline_rank_grid,
                ordering="euclidean",
            )
            offline_geometry[weighting] = geometry.to_dict()
            offline_geometry_validation[weighting] = (
                _validate_monotonic_offline_geometry(
                    geometry.to_dict(),
                    ranks=offline_rank_grid,
                )
            )
            basis_numerical_diagnostics[weighting] = (
                _basis_numerical_diagnostics(
                    fit_sequences,
                    basis,
                    ranks=offline_rank_grid,
                )
            )
        fit_manifest = _fit_manifest_receipt(
            fit_sequences,
            bases,
            weightings=fit_weightings,
        )
        tilted_rank64_spec: _ProjectionArmSpec | None
        if _tail_informed_factorial is None:
            if len({basis.artifact_sha256 for basis in bases.values()}) != 2:
                raise ValueError("tilted and unweighted fitted bases collapsed")
            projector_overlap: dict[str, object] | None = (
                _basis_projector_overlap(
                    bases["fisher_alignment_tilted"],
                    bases["unweighted"],
                )
            )
            specs: tuple[_AnyProjectionArmSpec, ...] = (
                _build_projection_arm_specs(bases)
            )
            tilted_rank64_spec = next(
                spec
                for spec in specs
                if isinstance(spec, _ProjectionArmSpec)
                and spec.fit_weighting == "fisher_alignment_tilted"
                and spec.rank == 64
            )
            if (
                tilted_rank64_spec.execution_basis_artifact_sha256
                != _EXPECTED_TILTED_RANK64_EXECUTION_BASIS_ARTIFACT_SHA256
            ):
                raise ValueError(
                    "max-rank fit changed the frozen tilted rank64 prefix"
                )
        else:
            from .gemma3_l3_l4_complete_h4_tail_informed_projection import (
                CompleteH4TailProjectionTrace,
                fit_complete_h4_tail_informed_projection,
            )

            projector_overlap = None
            tilted_rank64_spec = None
            global_basis = bases["unweighted"]
            global_specs = _build_projection_arm_specs(
                {"unweighted": global_basis},
                ranks=(192, 224, 256, 320),
                weightings=("unweighted",),
            )
            global_rank192_spec = global_specs[0]
            if (
                global_rank192_spec.execution_basis_artifact_sha256
                != _EXPECTED_UNWEIGHTED_RANK192_EXECUTION_BASIS_ARTIFACT_SHA256
            ):
                raise ValueError(
                    "U320 fit changed the authenticated unweighted U192 prefix"
                )
            tail_traces = tuple(
                CompleteH4TailProjectionTrace.from_fit_sequence(
                    trace.fit_sequence,
                    trace.graph_core_rows,
                    source_pair_sha256=trace.pair.artifact_sha256,
                    source_graph_core_mask_sha256=(
                        trace.pair.target_affected_mask_sha256
                    ),
                )
                for trace in traces
            )
            if sum(trace.tail_row_count for trace in tail_traces) != (
                frozen._EXPECTED_CAUSAL_TAIL_ROWS
            ):
                raise ValueError("tail-informed trace support differs from identity")
            tail_informed_fit = fit_complete_h4_tail_informed_projection(
                tail_traces,
                global_basis,
                anchor_rank=192,
                max_rank=320,
            )
            tail_informed_fit.validate_integrity()
            if int(tail_informed_fit.tail_rank) != 17:
                raise RuntimeError(
                    "locked A16 tail-informed fit requires numerical rank rT=17"
                )
            del tail_traces
            treatment_specs = _build_tail_informed_projection_arm_specs(
                tail_informed_fit,
                ranks=(209, 224, 256, 320),
            )
            specs = (*global_specs, *treatment_specs)
        if len(specs) != 8 or len({spec.arm_id for spec in specs}) != 8:
            raise RuntimeError("projection ladder must contain exactly eight arms")
        for spec in specs:
            _validate_any_projection_arm_spec(spec)

        exact_rows_by_arm: dict[str, dict[str, Tensor]] = {
            spec.arm_id: {} for spec in specs
        }
        exact_float64_geometry: dict[str, object] = {}
        for spec in specs:
            projected = exact_rows_by_arm[spec.arm_id]
            for trace in traces:
                projected[trace.example.example_id] = _project_projection_arm_rows(
                    trace.fit_sequence.residual_rows,
                    spec,
                )
            exact_float64_geometry[spec.arm_id] = _geometry_with_examples(
                traces,
                projected,
                candidate_semantics="exact_float64_projection_before_runtime_cast",
            )
        legacy_tilted_rank64_geometry = (
            None
            if _tail_informed_factorial is not None
            else frozen._geometry_summary(
                traces,
                # The legacy branch establishes this spec above.
                exact_rows_by_arm[tilted_rank64_spec.arm_id],
            )
        )

        behavior_accumulators = {
            spec.arm_id: {
                name: SourceAuthoritativeShadowFidelityAccumulator(
                    manifest,
                    gates=ESTABLISHED_SHADOW_FIDELITY_GATES,
                )
                for name, manifest in behavior_manifests.items()
            }
            for spec in specs
        }
        executed_rows_by_arm: dict[str, dict[str, Tensor]] = {
            spec.arm_id: {} for spec in specs
        }
        correction_receipts: dict[str, list[dict[str, object]]] = {
            spec.arm_id: [] for spec in specs
        }
        ceiling_receipts: list[dict[str, object]] = []
        tilted_rank64_logits_by_example: dict[str, str] = {}
        unweighted_rank192_logits_by_example: dict[str, str] = {}
        write_rows = {spec.arm_id: 0 for spec in specs}
        padding_write_rows = {spec.arm_id: 0 for spec in specs}
        support_supervised_tokens = {spec.arm_id: 0 for spec in specs}
        evaluation_shadow_forwards = 0
        exact_ceiling_forwards = 0
        projection_forwards = 0

        for trace in traces:
            example = trace.example
            pair = trace.pair
            tokenizer_integrity_check("before")
            model_inputs, supervised_indices, supervised_targets = (
                frozen._tokenize_one(
                    tokenizer,
                    example.prompt,
                    max_length=max_length,
                    model_input_device=device,
                )
            )
            tokenizer_integrity_check("after")
            if (
                _runtime_tensor_sha256(
                    frozen._supervised_grid_indices(supervised_indices)
                )
                != pair.supervised_indices_sha256
                or _runtime_tensor_sha256(
                    supervised_targets.detach()
                    .to(device="cpu", dtype=torch.int64)
                    .contiguous()
                )
                != pair.supervised_targets_sha256
            ):
                raise RuntimeError("evaluation supervision differs from fit pair")
            with torch.inference_mode():
                shadow = runtime.execute_model_shadow(
                    adapter,
                    model_inputs,
                    arm="all_on",
                )
            evaluation_shadow_forwards += 3

            ceiling = runtime.execute_complete_h4_correction_arm(
                adapter,
                model_inputs,
                shadow,
                pair,
                role="exact_h4_ceiling",
            )
            ceiling_metadata = ceiling.metadata()
            if (
                ceiling_metadata.get("role") != "exact_h4_ceiling"
                or ceiling_metadata.get("model_forward_count") != 1
                or ceiling_metadata.get("logits_bitwise_authoritative") is not True
                or ceiling_metadata.get("max_abs_authoritative_logit_error")
                != 0.0
                or ceiling_metadata.get("injected_h4_sha256")
                != pair.native_h4_sha256
                or ceiling_metadata.get("complete_h4_pair_artifact_sha256")
                != pair.artifact_sha256
                or ceiling_metadata.get("shadow_result_artifact_sha256")
                != pair.shadow_result_artifact_sha256
            ):
                raise RuntimeError("live exact-H4 ceiling is not authoritative")
            ceiling_receipts.append(
                {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "prompt_sha256": trace.prompt_sha256,
                    "arm": ceiling_metadata,
                }
            )
            exact_ceiling_forwards += 1
            del ceiling

            source_logits = frozen._select_sequence_rows(
                shadow.authoritative_logits,
                supervised_indices,
            )
            support_supervised = pair.complete_h4_support_mask[0].detach().to(
                device="cpu"
            ).index_select(0, supervised_indices)
            core_supervised = pair.target_affected_mask[0].detach().to(
                device="cpu"
            ).index_select(0, supervised_indices)
            selected_by_ledger = {
                "ordinary": torch.arange(
                    supervised_indices.numel(),
                    dtype=torch.int64,
                ),
                "complete_h4_support": torch.nonzero(
                    support_supervised,
                    as_tuple=False,
                ).flatten().to(dtype=torch.int64),
                "graph_core": torch.nonzero(
                    core_supervised,
                    as_tuple=False,
                ).flatten().to(dtype=torch.int64),
                "causal_tail": torch.nonzero(
                    support_supervised & ~core_supervised,
                    as_tuple=False,
                ).flatten().to(dtype=torch.int64),
            }
            for ledger_name, selected in selected_by_ledger.items():
                expected_member = example.example_id in behavior_manifests[
                    ledger_name
                ]
                if expected_member != (selected.numel() > 0):
                    raise RuntimeError("behavioral ledger membership drifted")

            for spec in specs:
                lineage_before = _validate_any_projection_arm_spec(spec)
                exact_rows = exact_rows_by_arm[spec.arm_id][example.example_id]
                cast_rows = exact_rows.to(
                    device=pair.incomplete_h4.device,
                    dtype=pair.incomplete_h4.dtype,
                )
                projected_delta = torch.zeros_like(pair.incomplete_h4)
                projected_delta[0].index_copy_(
                    0,
                    trace.support_indices.to(projected_delta.device),
                    cast_rows,
                )
                if bool(
                    (projected_delta[~pair.complete_h4_support_mask] != 0).any()
                ):
                    raise RuntimeError("projected delta escaped H4 support")
                arm = runtime.execute_complete_h4_correction_arm(
                    adapter,
                    model_inputs,
                    shadow,
                    pair,
                    projected_delta,
                    role="projection_oracle",
                    projection_basis=spec.execution_basis,
                    projection_basis_artifact_sha256=(
                        spec.execution_basis_artifact_sha256
                    ),
                    projection_fit_basis_artifact_sha256=(
                        spec.projection_fit_artifact_sha256
                    ),
                    projection_rank=spec.rank,
                    projection_ordering=spec.execution_ordering,
                )
                arm.validate_projected_delta(projected_delta)
                arm.validate_projection_basis(spec.execution_basis)
                arm_metadata = arm.metadata()
                lineage_after = _validate_any_projection_arm_spec(spec)
                if (
                    frozen._canonical_json_bytes(lineage_before)
                    != frozen._canonical_json_bytes(lineage_after)
                    or arm_metadata.get("role") != "projection_oracle"
                    or arm_metadata.get("model_forward_count") != 1
                    or arm_metadata.get("projection_rank") != spec.rank
                    or arm_metadata.get("projection_ordering")
                    != spec.execution_ordering
                    or arm_metadata.get("projection_basis_sha256")
                    != spec.execution_basis_sha256
                    or arm_metadata.get("projection_basis_artifact_sha256")
                    != spec.execution_basis_artifact_sha256
                    or arm_metadata.get("projection_fit_basis_artifact_sha256")
                    != spec.projection_fit_artifact_sha256
                    or arm_metadata.get("complete_h4_pair_artifact_sha256")
                    != pair.artifact_sha256
                    or arm_metadata.get("shadow_result_artifact_sha256")
                    != pair.shadow_result_artifact_sha256
                    or arm_metadata.get("complete_h4_support_mask_sha256")
                    != pair.complete_h4_support_mask_sha256
                ):
                    raise ValueError("complete-H4 ladder arm binding differs")

                candidate_logits = frozen._select_sequence_rows(
                    arm.logits,
                    supervised_indices,
                )
                for ledger_name, selected in selected_by_ledger.items():
                    if selected.numel() == 0:
                        continue
                    behavior_accumulators[spec.arm_id][ledger_name].add(
                        ShadowFidelityExample(
                            example_id=example.example_id,
                            family_id=example.family_id,
                            source_logits=source_logits.index_select(
                                0,
                                selected.to(source_logits.device),
                            ),
                            candidate_logits=candidate_logits.index_select(
                                0,
                                selected.to(candidate_logits.device),
                            ),
                            targets=supervised_targets.index_select(
                                0,
                                selected.to(supervised_targets.device),
                            ),
                        )
                    )
                executed_rows_by_arm[spec.arm_id][example.example_id] = (
                    projected_delta[0]
                    .index_select(
                        0,
                        trace.support_indices.to(projected_delta.device),
                    )
                    .detach()
                    .to(device="cpu")
                    .clone()
                    .contiguous()
                )
                rows_written = int(pair.complete_h4_support_mask.sum())
                padding_rows_written = int(
                    (
                        pair.complete_h4_support_mask
                        & ~pair.valid_target_mask
                    ).sum()
                )
                write_rows[spec.arm_id] += rows_written
                padding_write_rows[spec.arm_id] += padding_rows_written
                support_supervised_tokens[spec.arm_id] += int(
                    selected_by_ledger["complete_h4_support"].numel()
                )
                arm_receipt_payload = {
                    "example_id": example.example_id,
                    "family_id": example.family_id,
                    "prompt_sha256": trace.prompt_sha256,
                    "model_inputs_sha256": pair.model_inputs_sha256,
                    "execution_grid_sha256": pair.execution_grid_sha256,
                    "complete_h4_support_rows": rows_written,
                    "complete_h4_padding_write_rows": padding_rows_written,
                    "fit_to_prefix_before": lineage_before,
                    "fit_to_prefix_after": lineage_after,
                    "arm": arm_metadata,
                }
                correction_receipts[spec.arm_id].append(
                    {
                        **arm_receipt_payload,
                        "receipt_sha256": _domain_sha256(
                            arm_receipt_payload,
                            domain=_ARM_RECEIPT_DOMAIN,
                        ),
                    }
                )
                if (
                    tilted_rank64_spec is not None
                    and spec.arm_id == tilted_rank64_spec.arm_id
                ):
                    tilted_rank64_logits_by_example[example.example_id] = str(
                        arm_metadata["logits_sha256"]
                    )
                if spec.arm_id == "unweighted.rank192":
                    unweighted_rank192_logits_by_example[
                        example.example_id
                    ] = str(arm_metadata["logits_sha256"])
                projection_forwards += 1
                del arm, candidate_logits, projected_delta, cast_rows

            del (
                shadow,
                model_inputs,
                source_logits,
                supervised_targets,
                selected_by_ledger,
            )

        behavioral_summaries: dict[str, dict[str, object]] = {}
        for spec in specs:
            behavioral_summaries[spec.arm_id] = {
                name: accumulator.finalize()
                for name, accumulator in behavior_accumulators[
                    spec.arm_id
                ].items()
            }
        executed_cast_once_geometry = {
            spec.arm_id: _geometry_with_examples(
                traces,
                executed_rows_by_arm[spec.arm_id],
                candidate_semantics="cast_once_correction_submitted_to_runtime",
            )
            for spec in specs
        }
        runtime.validate_integrity()
        frozen._live_factorized_identity(adapter)
        tokenizer_integrity_check("after")
    finally:
        switcher.close()

    if adapter.model_fingerprint() != frozen._EXPECTED_RAW_MODEL_SHA256:
        raise RuntimeError("complete-H4 ladder did not restore raw Gemma")

    # Everything below is scalar/hash-only report construction.  No model
    # output, activation, gradient, fitted coefficient, or token is retained.
    exact_ceiling_summary = {
        "expected_prompt_count": 16,
        "observed_prompt_count": len(ceiling_receipts),
        "every_prompt_bitwise_authoritative": all(
            frozen._mapping(row["arm"], label="ceiling arm").get(
                "logits_bitwise_authoritative"
            )
            is True
            for row in ceiling_receipts
        ),
        "every_prompt_max_abs_logit_error_zero": all(
            frozen._mapping(row["arm"], label="ceiling arm").get(
                "max_abs_authoritative_logit_error"
            )
            == 0.0
            for row in ceiling_receipts
        ),
    }
    exact_ceiling_summary["passed"] = (
        exact_ceiling_summary["observed_prompt_count"] == 16
        and exact_ceiling_summary["every_prompt_bitwise_authoritative"] is True
        and exact_ceiling_summary["every_prompt_max_abs_logit_error_zero"] is True
    )
    if exact_ceiling_summary["passed"] is not True:
        raise RuntimeError("exact-H4 ceiling ledger is incomplete")

    support_integrity_by_arm: dict[str, dict[str, object]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    for spec in specs:
        support_integrity = {
            "expected_complete_h4_support_rows": (
                frozen._EXPECTED_COMPLETE_H4_ROWS
            ),
            "observed_complete_h4_support_rows": total_support_rows,
            "graph_core_rows": total_graph_core_rows,
            "causal_tail_rows": total_causal_tail_rows,
            "projection_write_rows": write_rows[spec.arm_id],
            "projection_padding_write_rows": padding_write_rows[spec.arm_id],
            "incomplete_h4_padding_difference_rows": (
                total_padding_difference_rows
            ),
            "support_supervised_tokens": support_supervised_tokens[spec.arm_id],
            "prompt_receipt_count": len(correction_receipts[spec.arm_id]),
        }
        support_integrity["support_coverage"] = (
            write_rows[spec.arm_id] / total_support_rows
        )
        support_integrity["passed"] = (
            total_support_rows == frozen._EXPECTED_COMPLETE_H4_ROWS
            and total_graph_core_rows == frozen._EXPECTED_GRAPH_CORE_ROWS
            and total_causal_tail_rows == frozen._EXPECTED_CAUSAL_TAIL_ROWS
            and write_rows[spec.arm_id] == total_support_rows
            and total_padding_difference_rows == 0
            and padding_write_rows[spec.arm_id] == 0
            and support_supervised_tokens[spec.arm_id] > 0
            and len(correction_receipts[spec.arm_id]) == 16
        )
        support_integrity_by_arm[spec.arm_id] = support_integrity
        ledgers = behavioral_summaries[spec.arm_id]
        comparisons[spec.arm_id] = classify_projection_ladder_arm(
            fit_weighting=spec.fit_weighting,
            rank=spec.rank,
            identity_validated=True,
            exact_h4_ceiling=exact_ceiling_summary,
            support_integrity=support_integrity,
            boundary_geometry=executed_cast_once_geometry[spec.arm_id],
            ordinary_behavioral=ledgers["ordinary"],
            support_behavioral=ledgers["complete_h4_support"],
            graph_core_behavioral=ledgers["graph_core"],
            causal_tail_behavioral=ledgers["causal_tail"],
        )

    parent_equivalent_comparisons = {
        arm_id: dict(comparison)
        for arm_id, comparison in comparisons.items()
    }
    if _tail_informed_factorial is not None:
        comparisons = _normalize_tail_informed_factorial_comparisons(
            comparisons
        )

    if _tail_informed_factorial is None:
        assert tilted_rank64_spec is not None
        assert legacy_tilted_rank64_geometry is not None
        selection = _select_projection_ladder(comparisons)
        tilted_rank64_regression = _validate_tilted_rank64_regression(
            prior=prior_rank64,
            spec=tilted_rank64_spec,
            boundary_geometry=legacy_tilted_rank64_geometry,
            ordinary_behavioral=behavioral_summaries[
                tilted_rank64_spec.arm_id
            ]["ordinary"],
            support_behavioral=behavioral_summaries[
                tilted_rank64_spec.arm_id
            ]["complete_h4_support"],
            logits_by_example=tilted_rank64_logits_by_example,
        )
        unweighted_rank192_parent_regression = None
    else:
        if parent_ladder is None or tail_informed_fit is None:
            raise RuntimeError("tail-informed factorial parent or fit was omitted")
        selection = _select_tail_informed_factorial(
            comparisons,
            tail_rank=int(tail_informed_fit.tail_rank),
        )
        tilted_rank64_regression = None
        global_rank192_spec = next(
            spec
            for spec in specs
            if isinstance(spec, _ProjectionArmSpec)
            and spec.fit_weighting == "unweighted"
            and spec.rank == 192
        )
        unweighted_rank192_parent_regression = (
            _validate_unweighted_rank192_parent_regression(
                parent=parent_ladder,
                spec=global_rank192_spec,
                exact_float64_geometry=exact_float64_geometry[
                    global_rank192_spec.arm_id
                ],
                executed_cast_once_geometry=executed_cast_once_geometry[
                    global_rank192_spec.arm_id
                ],
                behavioral_ledgers=behavioral_summaries[
                    global_rank192_spec.arm_id
                ],
                support_integrity=support_integrity_by_arm[
                    global_rank192_spec.arm_id
                ],
                comparison=parent_equivalent_comparisons[
                    global_rank192_spec.arm_id
                ],
                projection_macs=(
                    2
                    * _WIDTH
                    * global_rank192_spec.rank
                    * total_support_rows
                ),
                live_correction_receipts=correction_receipts[
                    global_rank192_spec.arm_id
                ],
                logits_by_example=unweighted_rank192_logits_by_example,
            )
        )

    expected_resources = _expected_resources(
        prompt_count=len(traces),
        arm_count=len(specs),
    )
    observed_resources = {
        "collect_model_forward_count": collect_forwards,
        "evaluation_shadow_model_forward_count": evaluation_shadow_forwards,
        "projection_arm_model_forward_count": projection_forwards,
        "exact_h4_ceiling_model_forward_count": exact_ceiling_forwards,
        "evaluation_model_forward_count": (
            evaluation_shadow_forwards
            + projection_forwards
            + exact_ceiling_forwards
        ),
        "total_model_forward_count": (
            collect_forwards
            + evaluation_shadow_forwards
            + projection_forwards
            + exact_ceiling_forwards
        ),
        "backward_count": backward_count,
    }
    if observed_resources != expected_resources:
        raise RuntimeError("complete-H4 ladder resource accounting differs")

    projection_macs_by_arm = {
        spec.arm_id: 2 * _WIDTH * spec.rank * total_support_rows
        for spec in specs
    }
    total_projection_macs = sum(projection_macs_by_arm.values())
    factorial_rank_sum: int | None = None
    if _tail_informed_factorial is not None:
        if tail_informed_fit is None:
            raise RuntimeError("tail-informed fit vanished before accounting")
        expected_factorial_ranks = (
            192,
            224,
            256,
            320,
            209,
            224,
            256,
            320,
        )
        if tuple(spec.rank for spec in specs) != expected_factorial_ranks:
            raise RuntimeError("tail-informed factorial rank ordering differs")
        factorial_rank_sum = sum(expected_factorial_ranks)
    expected_projection_macs = (
        1_006_387_200
        if _tail_informed_factorial is None
        else 2 * _WIDTH * total_support_rows * int(factorial_rank_sum)
    )
    if total_projection_macs != expected_projection_macs:
        raise RuntimeError("complete-H4 ladder projection MAC count differs")
    if _tail_informed_factorial is not None and (
        int(tail_informed_fit.tail_rank) != 17
        or factorial_rank_sum != 2_001
        or total_projection_macs != 2_097_688_320
    ):
        raise RuntimeError("locked rT=17 factorial MAC accounting differs")

    fit_sequence_matrix_bytes = sum(
        trace.fit_sequence.row_count
        * trace.fit_sequence.width
        * 8
        * (2 if trace.fit_sequence.has_gradients else 1)
        for trace in traces
    )
    fitted_basis_bytes = sum(
        basis.max_rank * basis.width * 8 for basis in bases.values()
    )
    if tail_informed_fit is not None:
        fitted_basis_bytes += (
            int(tail_informed_fit.max_rank)
            * int(tail_informed_fit.width)
            * 8
        )
    execution_prefix_bytes = sum(
        spec.execution_basis.numel() * spec.execution_basis.element_size()
        for spec in specs
    )
    exact_projection_rows_bytes = sum(
        value.numel() * value.element_size()
        for rows in exact_rows_by_arm.values()
        for value in rows.values()
    )
    executed_projection_rows_bytes = sum(
        value.numel() * value.element_size()
        for rows in executed_rows_by_arm.values()
        for value in rows.values()
    )

    arm_results: dict[str, object] = {}
    arm_lineage: list[dict[str, object]] = []
    for spec in specs:
        lineage = _validate_any_projection_arm_spec(spec)
        arm_lineage.append(lineage)
        arm_results[spec.arm_id] = {
            "fit_weighting": spec.fit_weighting,
            "rank": spec.rank,
            "fit_to_prefix_lineage": lineage,
            "exact_float64_geometry": exact_float64_geometry[spec.arm_id],
            "executed_cast_once_geometry": executed_cast_once_geometry[
                spec.arm_id
            ],
            "behavioral_ledgers": behavioral_summaries[spec.arm_id],
            "support_integrity": support_integrity_by_arm[spec.arm_id],
            "correction_receipts": tuple(correction_receipts[spec.arm_id]),
            "comparison": comparisons[spec.arm_id],
            "projection_macs_over_a16_support": projection_macs_by_arm[
                spec.arm_id
            ],
        }

    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "role": (
            "reused_calibration_a_truth_leaking_complete_h4_basis_rank_"
            "capacity_ladder"
        ),
        "lineage": {
            "rank64_x4_baseline_file_sha256": (
                frozen._EXPECTED_BASELINE_FILE_SHA256
            ),
            "rank64_x4_baseline_report_sha256": (
                frozen._EXPECTED_BASELINE_REPORT_SHA256
            ),
            "complete_h4_identity_file_sha256": (
                frozen._EXPECTED_IDENTITY_FILE_SHA256
            ),
            "complete_h4_identity_report_sha256": (
                frozen._EXPECTED_IDENTITY_REPORT_SHA256
            ),
            "frozen_rank64_projection_file_sha256": (
                _EXPECTED_PRIOR_FILE_SHA256
            ),
            "frozen_rank64_projection_report_sha256": (
                _EXPECTED_PRIOR_REPORT_SHA256
            ),
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "basis_package_payload_sha256": (
                frozen.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
            ),
            "fit_response_tensor_file_sha256": fit_source.file_sha256,
            "parent_graph_wavelet_artifact_sha256": parent.artifact_sha256,
            "raw_source_model_sha256": frozen._EXPECTED_RAW_MODEL_SHA256,
            "factorized_live_model_sha256": (
                frozen._EXPECTED_FACTORIZED_MODEL_SHA256
            ),
            "factorized_adapter_execution_sha256": (
                frozen._EXPECTED_FACTORIZED_EXECUTION_SHA256
            ),
        },
        "panel": panel_receipt,
        "protocol": {
            "isolated_target": (
                "native_h4_minus_incomplete_clamped_y3_exact_x4_carrier_h4"
            ),
            "objective": "mean_supervised_next_token_nll",
            "fit_partition": "same_locked_calibration_a_fit16",
            "evaluation_partition": "same_locked_calibration_a_fit16",
            "truth_leaking_same_a_capacity_screen": True,
            "fit_weightings": _WEIGHTINGS,
            "rank_grid": _RANK_GRID,
            "max_fit_rank": _MAX_RANK,
            "arm_count": len(specs),
            "all_arms_run_without_early_stop": True,
            "fit_to_prefix_receipt_revalidated_before_and_after_every_arm": (
                True
            ),
            "one_evaluation_shadow_shared_by_ceiling_and_all_eight_arms": (
                True
            ),
            "model_forwards_per_collect_prompt": 5,
            "model_forwards_per_evaluation_prompt": 12,
            "backwards_per_collect_prompt": 1,
            "exact_h4_ceiling_per_prompt": True,
            "exact_h4_ceiling_required_bitwise_authoritative": True,
            "geometry_hard_gate_uses_actual_cast_once_submitted_delta": True,
            "exact_float64_geometry_is_diagnostic": True,
            "behavioral_ledgers": tuple(behavior_manifests),
            "behavioral_aggregate_prompt_and_every_nonempty_family_gates": (
                True
            ),
            "stable_rank_rule": (
                "smallest_rank_whose_same_basis_arm_and_every_larger_"
                "tested_rank_pass"
            ),
            "deterministic_equal_rank_preference": "unweighted",
            "fisher_semantics": (
                "prompt_mean_nll_activation_gradient_empirical_fisher_proxy"
            ),
            "full_activation_fisher_claim": False,
            "runtime_recomputes_projection_from_authenticated_basis": True,
            "runtime_requires_submitted_delta_to_match_projection": True,
            "exact_x4_isolation": True,
            "learned_prediction": False,
            "bounded_pair_retains_vocab_logits": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "rank64_x4_baseline": {
            "file": baseline["file"],
            "file_sha256": baseline["file_sha256"],
            "report_sha256": baseline["report_sha256"],
            "rank64_plan_artifact_sha256": baseline[
                "rank64_plan_artifact_sha256"
            ],
            "rank64_arm_artifact_sha256": baseline[
                "rank64_arm_artifact_sha256"
            ],
            "runtime_binding_sha256": baseline["runtime_binding_sha256"],
        },
        "complete_h4_identity": {
            "file": identity["file"],
            "file_sha256": identity["file_sha256"],
            "report_sha256": identity["report_sha256"],
            "classification": "complete_h4_identity_validated",
        },
        "frozen_rank64_projection": {
            "file": prior_rank64["file"],
            "file_sha256": prior_rank64["file_sha256"],
            "report_sha256": prior_rank64["report_sha256"],
        },
        "rank64_x4_plan": plan_receipt,
        "rank64_x4_arm_receipt": dict(arm_receipt),
        "runtime_binding": runtime_metadata,
        "fitted_bases": {
            weighting: basis.metadata() for weighting, basis in bases.items()
        },
        "fit_manifest": fit_manifest,
        "basis_numerical_diagnostics": basis_numerical_diagnostics,
        "basis_projector_overlap_and_principal_angles": projector_overlap,
        "offline_rank_geometry": offline_geometry,
        "offline_rank_geometry_validation": offline_geometry_validation,
        "fit_to_prefix_lineage": tuple(arm_lineage),
        "behavioral_ledger_manifests": {
            name: {
                "example_count": len(manifest),
                "family_count": len(set(manifest.values())),
                "example_ids": tuple(sorted(manifest)),
                "family_ids": tuple(sorted(set(manifest.values()))),
            }
            for name, manifest in behavior_manifests.items()
        },
        "live_exact_h4_ceiling": {
            **exact_ceiling_summary,
            "prompt_receipts": tuple(ceiling_receipts),
        },
        "arms": arm_results,
        "tilted_rank64_frozen_regression": tilted_rank64_regression,
        "selection": selection,
        "collect_receipts": tuple(collect_receipts),
        "resource_accounting": {
            "model_load_count": 1,
            "tokenizer_load_count": 1,
            **observed_resources,
            **_full_vocabulary_logit_peak_accounting(),
            "candidate_logits_discarded_after_each_arm": True,
            "retained_pair_count": len(traces),
            "retained_pairs_contain_vocab_logits": False,
            "retained_pair_device": "cpu",
            "retained_pair_tensor_bytes_at_a16_peak": (
                retained_pair_tensor_bytes
            ),
            "largest_single_retained_pair_tensor_bytes": (
                largest_retained_pair_tensor_bytes
            ),
            "fit_sequence_residual_and_gradient_matrix_bytes": (
                fit_sequence_matrix_bytes
            ),
            "single_width_square_float64_fit_matrix_bytes": (
                _WIDTH * _WIDTH * 8
            ),
            "family_moment_width_square_matrices_per_fit": 3,
            "family_moment_matrix_bytes_per_sequential_fit": (
                3 * _WIDTH * _WIDTH * 8
            ),
            "eigensolver_transient_workspace_bytes_measured": False,
            "fitted_basis_matrix_bytes": fitted_basis_bytes,
            "execution_prefix_matrix_bytes": execution_prefix_bytes,
            "exact_float64_projection_rows_bytes_during_ladder": (
                exact_projection_rows_bytes
            ),
            "executed_cast_once_projection_rows_bytes_during_ladder": (
                executed_projection_rows_bytes
            ),
            "fitted_basis_float_coefficient_count": (
                2 * _MAX_RANK * _WIDTH
            ),
            "execution_prefix_float_coefficient_count": sum(
                spec.rank * _WIDTH for spec in specs
            ),
            "projection_macs_by_arm": projection_macs_by_arm,
            "logical_projection_macs_over_a16_all_eight_arms": (
                total_projection_macs
            ),
            "runtime_authentication_recomputes_projection_for_equality": True,
            "projection_mac_count_scope": (
                "one_logical_basis_project_and_reconstruct_per_arm_support_row"
            ),
            "basis_fit_and_eigendecomposition_are_offline_only": True,
            "float64_width_square_eigendecomposition_count": 2,
            "bounded_pair_reuse_assumption_must_be_revalidated_for_gpu_or_"
            "larger_models": True,
            "oracle_per_row_coordinates_are_deployable_parameters": False,
            "oracle_true_residual_rows_are_deployable_parameters": False,
            "whole_model_parameter_reduction_claim": False,
            "latency_or_speed_claim": False,
        },
        "scientific_status": {
            "development_capacity_ladder_complete": True,
            "all_eight_arms_executed": True,
            "live_exact_h4_ceiling_validated": True,
            "stable_projection_arm_selected": selection["selected_arm"],
            "overall_stable_passing_rank": selection[
                "overall_stable_passing_rank"
            ],
            "lofo_learned_generator_fitting_authorized": selection[
                "later_lofo_fitting_authorized"
            ],
            "success_authorizes_only_later_lofo_fitting": True,
            "same_a_truth_leaking_only": True,
            "formal_qualification": False,
            "generator_validated": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
        },
        "artifact": {"file": str(destination), "committable": False},
        "safety": _SAFETY,
    }

    report_domain = _REPORT_DOMAIN
    if _tail_informed_factorial is not None:
        if parent_ladder is None or tail_informed_fit is None:
            raise RuntimeError("tail-informed factorial report inputs were omitted")
        report_domain = _tail_informed_factorial.report_domain
        report["schema"] = _tail_informed_factorial.schema
        report["format_version"] = _tail_informed_factorial.format_version
        report["role"] = _tail_informed_factorial.role
        lineage = dict(frozen._mapping(report["lineage"], label="report lineage"))
        lineage["parent_basis_rank_ladder_file_sha256"] = parent_ladder[
            "file_sha256"
        ]
        lineage["parent_basis_rank_ladder_report_sha256"] = parent_ladder[
            "report_sha256"
        ]
        report["lineage"] = lineage
        protocol_report = dict(
            frozen._mapping(report["protocol"], label="factorial protocol")
        )
        protocol_report.pop("rank_grid", None)
        protocol_report.update(
            {
                "fit_weightings": ("unweighted",),
                "branch_rank_grids": {
                    "global_unweighted": (192, 224, 256, 320),
                    "tail_informed": (209, 224, 256, 320),
                },
                "max_fit_rank": 320,
                "factorial_branches": (
                    "global_unweighted_u320_prefix_control",
                    "u192_tail_residual_svd_span_then_mgs_u320_treatment",
                ),
                "tail_informed_is_one_global_euclidean_projector": True,
                "tail_informed_execution_ordering": (
                    COMPLETE_H4_TAIL_INFORMED_PROJECTION_ORDERING
                ),
                "row_routing": False,
                "all_arms_run_without_early_stop": True,
                "stable_rank_rule": (
                    "smallest_total_rank_whose_same_branch_arm_and_every_"
                    "larger_tested_branch_rank_pass"
                ),
                "deterministic_equal_rank_preference": "global_unweighted",
                "passing_factorial_authorizes_only": (
                    "frozen_basis_one_pass_carrier_transfer_oracle"
                ),
                "learned_lofo_fitting_authorized": False,
            }
        )
        report["protocol"] = protocol_report
        report["parent_basis_rank_ladder"] = {
            "file": parent_ladder["file"],
            "file_sha256": parent_ladder["file_sha256"],
            "report_sha256": parent_ladder["report_sha256"],
        }
        report["tail_informed_fit"] = tail_informed_fit.metadata()
        report["unweighted_rank192_parent_regression"] = (
            unweighted_rank192_parent_regression
        )
        report.pop("tilted_rank64_frozen_regression", None)
        for arm_id, arm in arm_results.items():
            arm["factorial_branch"] = (
                "tail_informed"
                if arm_id.startswith("tail_informed.")
                else "global_unweighted"
            )
        resource_report = dict(
            frozen._mapping(
                report["resource_accounting"],
                label="factorial resource accounting",
            )
        )
        resource_report.update(
            {
                "fitted_basis_float_coefficient_count": 2 * 320 * _WIDTH,
                "float64_width_square_eigendecomposition_count": 1,
                "tail_residual_float64_svd_count": 1,
                "sum_of_eight_total_projection_ranks": factorial_rank_sum,
                "logical_projection_macs_over_a16_all_eight_arms": (
                    total_projection_macs
                ),
                "rT17_locked_rank_sum_and_mac_match": (
                    int(tail_informed_fit.tail_rank) == 17
                    and factorial_rank_sum == 2_001
                    and total_projection_macs == 2_097_688_320
                ),
            }
        )
        report["resource_accounting"] = resource_report
        status_report = dict(
            frozen._mapping(
                report["scientific_status"],
                label="factorial scientific status",
            )
        )
        status_report["tail_informed_factorial_complete"] = True
        status_report["lofo_learned_generator_fitting_authorized"] = False
        status_report["success_authorizes_only_later_lofo_fitting"] = False
        status_report[
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized"
        ] = selection[
            "frozen_basis_one_pass_carrier_transfer_oracle_authorized"
        ]
        status_report[
            "success_authorizes_only_frozen_basis_one_pass_carrier_transfer"
        ] = True
        status_report["token_wise_vjp_branch_opened"] = False
        status_report["token_wise_vjp_branch_is_next_if_neither_stable"] = (
            selection["selected_arm"] is None
        )
        status_report["next_rung"] = (
            "frozen_basis_one_pass_carrier_transfer_oracle"
            if selection["selected_arm"] is not None
            else "token_wise_vjp_basis_branch"
        )
        report["scientific_status"] = status_report

    # Make the no-tensor publication boundary explicit before validation.
    del (
        exact_rows_by_arm,
        executed_rows_by_arm,
        behavior_accumulators,
        bases,
        specs,
        traces,
    )
    return _publish(
        report,
        output=destination,
        report_domain=report_domain,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen A16 complete-H4 basis/rank capacity ladder",
    )
    parser.add_argument(
        "--fit-source-artifact",
        default=frozen.DEFAULT_INTERIOR_ARTIFACT,
    )
    parser.add_argument(
        "--parent-artifact",
        default=frozen.DEFAULT_PARENT_ARTIFACT,
    )
    parser.add_argument(
        "--candidate-artifact",
        default=frozen.DEFAULT_CANDIDATE_ARTIFACT,
    )
    parser.add_argument("--basis-package", default=frozen.DEFAULT_BASIS_PACKAGE)
    parser.add_argument(
        "--base-artifact",
        default=frozen.DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        default=frozen.DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--panel", default=frozen.DEFAULT_PANEL)
    parser.add_argument(
        "--rank64-x4-baseline",
        default=frozen.DEFAULT_RANK64_X4_BASELINE,
    )
    parser.add_argument(
        "--complete-h4-identity",
        default=frozen.DEFAULT_COMPLETE_H4_IDENTITY,
    )
    parser.add_argument(
        "--rank64-projection-baseline",
        default=DEFAULT_RANK64_PROJECTION_BASELINE,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--max-length",
        type=int,
        default=frozen.DEFAULT_MAX_LENGTH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_projection_basis_rank_ladder(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        rank64_x4_baseline_path=arguments.rank64_x4_baseline,
        complete_h4_identity_path=arguments.complete_h4_identity,
        rank64_projection_baseline_path=(
            arguments.rank64_projection_baseline
        ),
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        max_length=arguments.max_length,
    )
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "selection": report["selection"],
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
