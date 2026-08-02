"""Held-only finite validation of the frozen V5/V6/V7 K64 gain arms.

This v8 rung first recollects and exactly reproduces the pinned v7 analytic
capacity result.  Only after that authentication succeeds does it execute
three already-frozen arms on the sixteen outer-held prompts: the V5 positive
one-over-64 carrier, the exact V6 scalar control, and the V7 joint
intercept-plus-state field.  There is no tune phase, arm selection, refit,
step grid, damping, fallback, or per-family routing.

The panel remains reused Calibration A and the finite providers use native
teacher logits and held native H4 state.  A pass is therefore same-A finite
hypothesis evidence only; it is not a serving, compression, deployment, or
speed result.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from . import complete_h4_tail_candidate_state_gain_field as state_field
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic as v5diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic as v3diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic as v4diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic as v7diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic as v6diag
from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from .complete_h4_tail_candidate_gain_refit_v4 import CANDIDATE_GAIN_RANK
from .complete_h4_tail_candidate_joint_state_gain_finite import (
    THREE_ARM_FINITE_NAMES,
    V8_EXPECTED_CANDIDATE_FORWARD_COUNT,
    V8_EXPECTED_HELD_CELL_COUNT,
    CandidateConditionedK64ThreeArmFiniteComparison,
    CandidateConditionedK64ThreeArmFiniteExample,
    CandidateConditionedK64ThreeArmGainSupport,
    build_candidate_conditioned_k64_three_arm_gain_support,
    candidate_conditioned_k64_gain_correction_rows,
    compare_candidate_conditioned_k64_three_arm_finite_examples,
)
from .complete_h4_tail_candidate_state_gain_field import (
    CandidateConditionedK64StateFeatureCodec,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    complete_h4_tail_gate_scores,
    fit_complete_h4_tail_held_family,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import _load_committed_basis
from .gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _require_sha256,
    _runtime_tensor_sha256,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    SourceAuthoritativeShadowFidelityAccumulator,
)


__all__ = [
    "DEFAULT_EXPANDED_PARENT_REPORT",
    "DEFAULT_MATERIALIZATION_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TRANSFER_REPORT",
    "DEFAULT_V3_REPORT",
    "DEFAULT_V4_REPORT",
    "DEFAULT_V5_REPORT",
    "DEFAULT_V6_REPORT",
    "DEFAULT_V7_REPORT",
    "V7_REPORT_FILE_SHA256",
    "V7_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_finite_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v7diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v7diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v7diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v7diag.DEFAULT_V3_REPORT
DEFAULT_V4_REPORT = v7diag.DEFAULT_V4_REPORT
DEFAULT_V5_REPORT = v7diag.DEFAULT_V5_REPORT
DEFAULT_V6_REPORT = v7diag.DEFAULT_V6_REPORT
DEFAULT_V7_REPORT = v7diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-joint-state-gain-held-finite-"
    "lofo-a-fit16-dev-v8.json"
)

V7_REPORT_FILE_SHA256 = (
    "8822529ca2526fe73157a4497e199cadb6594e6c5e8597625f0158965e16b0b6"
)
V7_REPORT_SHA256 = (
    "816a1b7fe25f02d2b17dcb7e8cd9a57105c94dab1e3644941650b5a111cf789a"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_joint_state_gain_held_finite_lofo.v8"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-joint-state-gain-finite:v8\0"
_V7_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-v7-live-binding:v8\0"
_CODEC_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-v7-codec-binding:v8\0"
_V5_PLUS_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-v5-plus-binding:v8\0"
_PROVIDER_DOMAIN = b"fisher-graph:complete-h4-k64-three-arm-provider:v8\0"
_OBSERVATION_DOMAIN = b"fisher-graph:complete-h4-k64-three-arm-observation:v8\0"
_OBSERVATION_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-three-arm-observation-set:v8\0"
)

_EXPECTED_FAMILIES = 8
_EXPECTED_PROMPTS = 16
_EXPECTED_ARMS = 3
_EXPECTED_CANDIDATE_FORWARDS = 48
_EXPECTED_FINITE_FORWARDS = 64
_EXPECTED_TOTAL_FORWARDS = 176
_EXPECTED_TOTAL_BACKWARDS = 494
_TOP1_THRESHOLD = 0.90
_TOP1_REGRESSION_TOLERANCE = 0.01
_KL_RELATIVE_IMPROVEMENT_MINIMUM = 0.02
_KL_FAMILY_WIN_MINIMUM = 6
_KL_WORST_FAMILY_MULTIPLIER = 1.05
_KL_WORST_FAMILY_ABSOLUTE_TOLERANCE = 1.0e-8
_LEDGERS = v5diag._LEDGERS

if (
    V8_EXPECTED_HELD_CELL_COUNT != _EXPECTED_PROMPTS
    or V8_EXPECTED_CANDIDATE_FORWARD_COUNT != _EXPECTED_CANDIDATE_FORWARDS
):
    raise RuntimeError("V8 diagnostic and pure-core resource contracts differ")


def _canonical(value: object) -> object:
    return v3diag._canonical(value)


def _load_v7_report(path: Path | str) -> dict[str, object]:
    """Load only the exact passing live V7 analytic capacity artifact."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V7_REPORT_FILE_SHA256,
        expected_report_sha256=V7_REPORT_SHA256,
        label="candidate joint state-gain capacity v7 control",
    )
    if (
        report.get("schema") != v7diag._SCHEMA
        or report.get("classification")
        != "joint_capacity_supported_for_finite_validation"
        or report.get("passed") is not True
    ):
        raise RuntimeError("candidate joint state-gain capacity v7 differs")
    return report


def _authenticate_live_v7_evidence(
    *,
    v7_report: Mapping[str, object],
    phases: v7diag._JointPhaseResults,
) -> dict[str, object]:
    """Require canonical reproduction of all V7 analytic result geometry."""

    expected_screen = v7_report.get("joint_analytic_capacity_screen")
    expected_folds = v7_report.get("joint_analytic_fold_records")
    expected_inner = v7_report.get("joint_analytic_inner_family_records")
    control = v7_report.get("v6_control_binding")
    expected_scalar = v7_report.get("v6_scalar_comparator_binding")
    resources = v7_report.get("resources")
    if (
        not isinstance(expected_screen, Mapping)
        or not isinstance(expected_folds, list)
        or not isinstance(expected_inner, list)
        or not isinstance(control, Mapping)
        or not isinstance(expected_scalar, Mapping)
        or not isinstance(resources, Mapping)
    ):
        raise ValueError("pinned V7 analytic evidence differs")
    expected_v6_live = control.get("live_evidence_reproduction")
    if not isinstance(expected_v6_live, Mapping):
        raise ValueError("pinned V7 V6 reproduction binding differs")
    live_folds = tuple(record.metadata() for record in phases.joint_fold_records)
    live_inner = tuple(
        inner.metadata()
        for record in phases.joint_fold_records
        for inner in record.inner_family_records
    )
    comparisons = (
        (phases.joint_screen.metadata(), expected_screen, "screen"),
        (live_folds, expected_folds, "fold records"),
        (live_inner, expected_inner, "inner records"),
        (phases.v6_binding, expected_v6_live, "V6 live binding"),
        (phases.scalar_comparator_binding, expected_scalar, "scalar binding"),
    )
    for live, expected, label in comparisons:
        if _canonical(live) != _canonical(expected):
            raise RuntimeError(f"live V8 recollection did not reproduce V7 {label}")
    if (
        len(live_folds) != _EXPECTED_FAMILIES
        or len(live_inner) != 56
        or not phases.joint_screen.capacity_screen_passed
        or phases.joint_screen.outcome
        != "joint_capacity_supported_for_finite_validation"
        or resources.get("total_model_forward_count") != 112
        or resources.get("total_backward_call_count") != 494
        or resources.get("finite_joint_candidate_model_forward_count") != 0
    ):
        raise RuntimeError("pinned V7 analytic geometry differs")
    payload = {
        "v7_report_file_sha256": V7_REPORT_FILE_SHA256,
        "v7_report_sha256": V7_REPORT_SHA256,
        "v7_screen_artifact_sha256": phases.joint_screen.artifact_sha256,
        "v7_fold_artifact_sha256s": tuple(
            record.artifact_sha256 for record in phases.joint_fold_records
        ),
        "v7_inner_record_count": len(live_inner),
        "screen_metadata_canonically_equal": True,
        "fold_metadata_canonically_equal": True,
        "all_56_inner_metadata_rows_canonically_equal": True,
        "v6_live_binding_canonically_equal": True,
        "all_64_scalar_comparators_canonically_equal": True,
        "authenticated_before_any_finite_forward": True,
        "raw_tensors_serialized": False,
    }
    return {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V7_BINDING_DOMAIN
        ),
    }


def _reconstruct_full_v7_codecs(
    *,
    phases: v7diag._JointPhaseResults,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> tuple[
    dict[str, CandidateConditionedK64StateFeatureCodec],
    dict[str, object],
]:
    """Rebuild each full-seven codec without any held prompt or recentering."""

    fold_records = {
        record.outer_held_family_id: record
        for record in phases.joint_fold_records
    }
    if set(fold_records) != set(fits) or len(fold_records) != _EXPECTED_FAMILIES:
        raise RuntimeError("V8 full-fold universe differs")
    codecs: dict[str, CandidateConditionedK64StateFeatureCodec] = {}
    receipts: list[dict[str, object]] = []
    for held_family in sorted(fits):
        fit = fits[held_family]
        directions = v6diag._ordered_k64(fit)
        cells = tuple(
            sorted(
                phases.v6_phases.row_bank.cells[held_family],
                key=lambda value: value.family_id,
            )
        )
        if (
            len(cells) != 7
            or {cell.family_id for cell in cells}
            != set(phases.v6_phases.row_bank.refits[held_family].training_family_ids)
            or any(cell.held_family_id != held_family for cell in cells)
        ):
            raise RuntimeError("V8 full-seven codec training grid differs")
        codec = v6diag.fit_candidate_conditioned_k64_state_feature_codec(
            tuple(v6diag._feature_example(cell) for cell in cells),
            held_family_id=held_family,
            ordered_directions=directions,
        )
        record = fold_records[held_family]
        if (
            codec.artifact_sha256
            != record.full_joint_fit.codec_artifact_sha256
            or codec.training_family_ids
            != record.full_joint_fit.training_family_ids
            or codec.training_example_ids
            != record.full_joint_fit.training_example_ids
        ):
            raise RuntimeError("V8 full-seven codec did not reproduce V7")
        codecs[held_family] = codec
        receipt = {
            "held_family_id": held_family,
            "training_family_ids": codec.training_family_ids,
            "training_example_ids": codec.training_example_ids,
            "codec_artifact_sha256": codec.artifact_sha256,
            "v7_full_joint_codec_artifact_sha256": (
                record.full_joint_fit.codec_artifact_sha256
            ),
            "held_rows_used_for_codec_fit_or_recenter": False,
            "full_seven_codec_canonically_reproduced": True,
            "raw_tensors_serialized": False,
        }
        receipt["receipt_sha256"] = token_v1._domain_sha256(
            receipt, domain=_CODEC_BINDING_DOMAIN
        )
        receipts.append(receipt)
    payload = {
        "full_seven_codec_count": len(codecs),
        "receipts": tuple(receipts),
        "every_codec_reproduces_v7_full_joint_fit": True,
        "held_rows_used_for_codec_fit_or_recenter": False,
        "raw_tensors_serialized": False,
    }
    payload["artifact_sha256"] = token_v1._domain_sha256(
        payload, domain=_CODEC_BINDING_DOMAIN
    )
    return codecs, payload


def _pinned_v5_selected_plus_rows(
    v5_report: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Bind the twelve held V5 plus observations that actually existed."""

    raw = v5_report.get("finite_observation_receipts")
    if not isinstance(raw, list):
        raise ValueError("pinned V5 final observation evidence differs")
    plus: dict[str, dict[str, object]] = {}
    unit_count = 0
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("pinned V5 final observation row differs")
        row = dict(value)
        example_id = token_v1._identifier(
            row.get("example_id"), label="pinned V5 example_id"
        )
        sign = row.get("selected_sign")
        if sign == 1:
            if example_id in plus:
                raise RuntimeError("pinned V5 plus example repeats")
            plus[example_id] = row
        elif sign == 0:
            unit_count += 1
        else:
            raise RuntimeError("pinned V5 selected sign differs")
    if len(raw) != _EXPECTED_PROMPTS or len(plus) != 12 or unit_count != 4:
        raise RuntimeError("pinned V5 plus/unit held split differs")
    payload = {
        "pinned_v5_selected_plus_observation_count": len(plus),
        "counterfactual_plus_observation_count": unit_count,
        "selected_plus_example_ids": tuple(sorted(plus)),
        "selected_plus_token_teacher_kl_sha256s": tuple(
            str(plus[key]["token_teacher_kl_sha256"])
            for key in sorted(plus)
        ),
        "static_plus_is_counterfactual_for_obsidian_and_shell_midden": True,
        "all_16_prompts_claimed_as_previously_executed_plus": False,
        "raw_tensors_serialized": False,
    }
    payload["artifact_sha256"] = token_v1._domain_sha256(
        payload, domain=_V5_PLUS_BINDING_DOMAIN
    )
    return plus, payload


class _AuthenticatedV8GainProvider(Gemma3L3L4CorrectionProvider):
    """Single-use static or row-conditioned correction for one held arm."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "stage",
        "arm",
        "fold_artifact_sha256",
        "gain_support_artifact_sha256",
        "ordered_directions_sha256",
        "gains_sha256",
        "example_id",
        "family_id",
        "model_inputs_sha256",
        "bridge_binding_sha256",
        "prefix_artifact_sha256",
        "base_h4_sha256",
        "support_mask_sha256",
        "correction_sha256",
        "_support",
        "_correction",
        "_used",
    )

    def __init__(
        self,
        *,
        arm: str,
        fold_artifact_sha256: str,
        gain_support_artifact_sha256: str,
        ordered_directions_sha256: str,
        gains: Tensor,
        example_id: str,
        family_id: str,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        if arm not in THREE_ARM_FINITE_NAMES:
            raise ValueError("V8 provider arm differs")
        gain_values = gains.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        if (
            gain_values.ndim not in {1, 2}
            or gain_values.shape[-1] != CANDIDATE_GAIN_RANK
            or not bool(torch.isfinite(gain_values).all())
            or bool((gain_values < 0.0).any())
            or bool((gain_values > 1.5).any())
            or (arm == "v7_joint") != (gain_values.ndim == 2)
        ):
            raise ValueError("V8 provider gain geometry differs")
        if (
            not isinstance(base_h4, Tensor)
            or base_h4.ndim != 3
            or base_h4.shape[-1] != token_v1._WIDTH
            or not base_h4.is_floating_point()
            or not isinstance(support_mask, Tensor)
            or support_mask.shape != base_h4.shape[:2]
            or support_mask.dtype != torch.bool
            or not isinstance(correction, Tensor)
            or correction.shape != base_h4.shape
            or not correction.is_floating_point()
        ):
            raise ValueError("V8 provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        delta = correction.detach().to(
            device="cpu", dtype=torch.float64
        ).clone().contiguous()
        if (
            (gain_values.ndim == 2 and gain_values.shape[0] != int(support.sum()))
            or not bool(torch.isfinite(delta).all())
            or bool((delta[~support] != 0).any())
        ):
            raise ValueError("V8 provider correction escapes held support")
        self.site = token_v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.stage = "held"
        self.arm = arm
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="V8 fold"
        )
        self.gain_support_artifact_sha256 = _require_sha256(
            gain_support_artifact_sha256, label="V8 gain support"
        )
        self.ordered_directions_sha256 = _require_sha256(
            ordered_directions_sha256, label="V8 directions"
        )
        self.gains_sha256 = _runtime_tensor_sha256(gain_values)
        self.example_id = token_v1._identifier(
            example_id, label="V8 example_id"
        )
        self.family_id = token_v1._identifier(
            family_id, label="V8 family_id"
        )
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="V8 model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="V8 bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="V8 prefix"
        )
        self.base_h4_sha256 = _runtime_tensor_sha256(base_h4)
        self._support = support
        self._correction = delta
        self.support_mask_sha256 = _runtime_tensor_sha256(support)
        self.correction_sha256 = _runtime_tensor_sha256(delta)
        self._used = False
        self.artifact_sha256 = token_v1._domain_sha256(
            self._payload(), domain=_PROVIDER_DOMAIN
        )
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.complete_h4_k64_three_arm_provider.v8",
            "rank": CANDIDATE_GAIN_RANK,
            "site": self.site,
            "write_scope": self.write_scope,
            "stage": self.stage,
            "arm": self.arm,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "gain_support_artifact_sha256": self.gain_support_artifact_sha256,
            "ordered_directions_sha256": self.ordered_directions_sha256,
            "gains_sha256": self.gains_sha256,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": (
                "P_D320_R_plus_frozen_static_or_row_conditioned_K64_tail"
            ),
            "single_use": True,
            "truth_leaking_same_A_hypothesis_use_only": True,
            "serving_authorized": False,
        }

    @property
    def used(self) -> bool:
        return self._used

    def validate_integrity(self) -> None:
        if (
            self.site != token_v1._H4_SITE
            or self.write_scope != "complete_h4_causal_support"
            or _runtime_tensor_sha256(self._support) != self.support_mask_sha256
            or _runtime_tensor_sha256(self._correction) != self.correction_sha256
            or bool((self._correction[~self._support] != 0).any())
            or token_v1._domain_sha256(self._payload(), domain=_PROVIDER_DOMAIN)
            != self.artifact_sha256
        ):
            raise RuntimeError("V8 three-arm provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("V8 three-arm provider cannot be reused")
        prefix.validate_integrity()
        if (
            prefix.artifact_sha256 != self.prefix_artifact_sha256
            or prefix.bridge_binding_sha256 != self.bridge_binding_sha256
            or _runtime_tensor_sha256(realized_state) != self.base_h4_sha256
            or _runtime_tensor_sha256(
                prefix.complete_h4_causal_support_mask()
                .detach()
                .to(device="cpu")
                .contiguous()
            )
            != self.support_mask_sha256
        ):
            raise RuntimeError("V8 provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _executed_cast_once_delta_rows(
    *,
    base_h4: Tensor,
    correction: Tensor,
    support_mask: Tensor,
    support_indices: Tensor,
) -> Tensor:
    """Return the delta the model actually sees after its single dtype cast."""

    if (
        not isinstance(base_h4, Tensor)
        or base_h4.ndim != 3
        or base_h4.shape[-1] != token_v1._WIDTH
        or not base_h4.is_floating_point()
        or not isinstance(correction, Tensor)
        or correction.shape != base_h4.shape
        or not correction.is_floating_point()
        or not isinstance(support_mask, Tensor)
        or support_mask.shape != base_h4.shape[:2]
        or support_mask.dtype != torch.bool
        or not isinstance(support_indices, Tensor)
        or support_indices.ndim != 1
        or support_indices.dtype != torch.long
    ):
        raise ValueError("V8 cast-once tensor geometry differs")
    expected_h4 = base_h4.detach().clone()
    support = support_mask.detach().to(device=expected_h4.device)
    delta = correction.detach().to(
        device=expected_h4.device, dtype=torch.float64
    )
    if bool((delta[~support] != 0).any()):
        raise ValueError("V8 cast-once correction escapes support")
    expected_h4[support] = (
        base_h4.detach()[support].to(dtype=torch.float64) + delta[support]
    ).to(dtype=expected_h4.dtype)
    base_rows = (
        base_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, support_indices.detach().to(device="cpu"))
    )
    return (
        expected_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, support_indices.detach().to(device="cpu"))
        .sub(base_rows)
        .contiguous()
    )


def _execute_held_finite_arm(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    support: CandidateConditionedK64ThreeArmGainSupport,
    arm: str,
    model_inputs: Mapping[str, Tensor],
    teacher_logits: Tensor,
    endpoint_indices: Tensor,
) -> tuple[Tensor, object, _AuthenticatedV8GainProvider, Tensor]:
    """Execute one arm with common V5 row algebra and one cast-once write."""

    directions = v6diag._ordered_k64(fit)
    support.validate_integrity()
    support_metadata = support.metadata()
    gain_hash_field = {
        "v5_static_plus": "static_plus_gains_sha256",
        "v6_exact_scalar": "exact_scalar_gains_sha256",
        "v7_joint": "joint_row_gains_sha256",
    }.get(arm)
    if gain_hash_field is None:
        raise ValueError("unknown V8 finite arm")
    gains = support.gains_tensor(arm)
    support_mask = (
        trace.prefix.complete_h4_causal_support_mask()
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    base_support_rows = (
        trace.base_h4.detach()
        .to(device="cpu", dtype=torch.float64)[0]
        .index_select(0, trace.support_indices.detach().to(device="cpu"))
        .contiguous()
    )
    directions_sha256 = _runtime_tensor_sha256(directions)
    gains_sha256 = _runtime_tensor_sha256(gains)
    if (
        support.phase != "held"
        or support.example_id != trace.example_id
        or support.family_id != trace.family_id
        or support.held_family_id != trace.family_id
        or support.parent_fold_artifact_sha256 != fit.artifact_sha256
        or support.ordered_directions_codec_sha256
        != state_field._tensor_sha256(directions)
        or support.ordered_directions_refit_sha256 != directions_sha256
        or support.base_h4_support_rows_sha256
        != _runtime_tensor_sha256(base_support_rows)
        or support.support_row_count != int(support_mask.sum())
        or support_metadata.get(gain_hash_field) != gains_sha256
    ):
        raise RuntimeError("V8 arm gain support binding differs")
    residual = trace.endpoint.residual_rows.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    frozen_basis = basis.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    supported_rows = ((residual @ frozen_basis.T) @ frozen_basis).contiguous()
    tail_rows = v3diag.project_complete_h4_tail_rows(
        residual, frozen_basis
    ).contiguous()
    correction_rows = candidate_conditioned_k64_gain_correction_rows(
        supported_rows=supported_rows,
        tail_rows=tail_rows,
        ordered_directions=directions,
        gains=gains,
    )
    _directions, _tail, reference_rows, _correction = (
        v3diag._candidate_components(
            trace,
            basis=frozen_basis,
            fit=fit,
            gains=gains,
        )
    )
    if (
        not torch.equal(_directions, directions)
        or not torch.equal(_tail, tail_rows)
        or not torch.equal(reference_rows, correction_rows)
    ):
        raise RuntimeError("V8 common row algebra does not replay V5")
    correction = torch.zeros(trace.base_h4.shape, dtype=torch.float64)
    correction[0].index_copy_(
        0, trace.support_indices, correction_rows
    )
    provider = _AuthenticatedV8GainProvider(
        arm=arm,
        fold_artifact_sha256=fit.artifact_sha256,
        gain_support_artifact_sha256=support.artifact_sha256,
        ordered_directions_sha256=directions_sha256,
        gains=gains,
        example_id=trace.example_id,
        family_id=trace.family_id,
        model_inputs_sha256=trace.model_inputs_sha256,
        bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
        prefix_artifact_sha256=trace.prefix.artifact_sha256,
        base_h4=trace.base_h4,
        support_mask=support_mask,
        correction=correction,
    )
    if (
        provider.gains_sha256 != gains_sha256
        or provider.gain_support_artifact_sha256 != support.artifact_sha256
        or provider.correction_sha256 != _runtime_tensor_sha256(correction)
        or not torch.equal(
            provider._correction[0].index_select(0, trace.support_indices),
            correction_rows,
        )
    ):
        raise RuntimeError("V8 provider did not bind exact arm row algebra")
    execution = getattr(context, "bridge").execute(
        getattr(context, "adapter"), model_inputs, h4_head=provider
    )
    v3diag._validate_candidate_execution(
        trace=trace, provider=provider, execution=execution
    )
    token_kl = teacher_kl._selected_token_teacher_kl(
        teacher_logits, execution.logits, endpoint_indices
    ).detach().to(device="cpu", dtype=torch.float64).contiguous()
    return token_kl, execution, provider, correction_rows


def _geometry_trace(trace: object) -> object:
    return SimpleNamespace(
        example=trace.example,
        fit_sequence=CompleteH4ProjectionFitSequence(
            example_id=trace.example_id,
            family_id=trace.family_id,
            residual_rows=trace.endpoint.residual_rows,
        ),
        support_indices=trace.support_indices,
        graph_core_rows=(
            trace.prefix.target_affected_mask.detach()
            .to(device="cpu")[0]
            .index_select(0, trace.support_indices)
        ),
    )


@dataclass(slots=True)
class _HeldFiniteGridResult:
    examples: tuple[CandidateConditionedK64ThreeArmFiniteExample, ...]
    comparison: CandidateConditionedK64ThreeArmFiniteComparison
    gain_support_receipts: tuple[dict[str, object], ...]
    observation_receipts: tuple[dict[str, object], ...]
    observation_set_sha256: str
    behavior_by_arm: Mapping[str, Mapping[str, object]]
    nll_by_arm: Mapping[str, Mapping[str, object]]
    geometry_by_arm: Mapping[str, Mapping[str, object]]
    top1_comparison_by_arm: Mapping[str, Mapping[str, object]]
    plus_replay_binding: Mapping[str, object]
    resources: Mapping[str, int]


def _held_arm_schedule(
    traces: Sequence[object],
) -> tuple[tuple[object, str], ...]:
    """Freeze the complete held 16-by-3 schedule before any finite forward."""

    values = tuple(sorted(traces, key=lambda value: value.example_id))
    families = tuple(sorted({trace.family_id for trace in values}))
    family_counts = tuple(
        sum(trace.family_id == family for trace in values) for family in families
    )
    if (
        len(values) != _EXPECTED_PROMPTS
        or len({trace.example_id for trace in values}) != _EXPECTED_PROMPTS
        or len(families) != _EXPECTED_FAMILIES
        or family_counts != (2,) * _EXPECTED_FAMILIES
    ):
        raise RuntimeError("V8 held schedule requires two prompts in eight families")
    schedule = tuple(
        (trace, arm) for trace in values for arm in THREE_ARM_FINITE_NAMES
    )
    if (
        len(schedule) != _EXPECTED_CANDIDATE_FORWARDS
        or any(
            sum(
                scheduled.example_id == trace.example_id and scheduled_arm == arm
                for scheduled, scheduled_arm in schedule
            )
            != 1
            for trace in values
            for arm in THREE_ARM_FINITE_NAMES
        )
    ):
        raise RuntimeError("V8 held schedule is not the exact 16-by-3 grid")
    return schedule


def _descriptive_nll_summary_by_arm(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize endpoint and ordinary NLL without creating another gate."""

    required = (
        "native_mean_nll",
        "d320_mean_nll",
        "candidate_mean_nll",
        "native_ordinary_mean_nll",
        "d320_ordinary_mean_nll",
        "ordinary_candidate_mean_nll",
    )
    summaries: dict[str, dict[str, object]] = {}
    tiny = torch.finfo(torch.float64).tiny
    for arm in THREE_ARM_FINITE_NAMES:
        selected = tuple(row for row in observations if row.get("arm") == arm)
        families = tuple(sorted({str(row.get("family_id")) for row in selected}))
        grouped = {
            family: tuple(
                row for row in selected if str(row.get("family_id")) == family
            )
            for family in families
        }
        if (
            len(selected) != _EXPECTED_PROMPTS
            or len(families) != _EXPECTED_FAMILIES
            or any(len(rows) != 2 for rows in grouped.values())
        ):
            raise RuntimeError("V8 descriptive NLL grid differs")
        for row in selected:
            for key in required:
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(
                    float(value)
                ):
                    raise ValueError(f"V8 descriptive NLL field {key} differs")

        def prompt_mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
            return math.fsum(float(row[key]) for row in rows) / len(rows)

        def family_macro(key: str) -> float:
            return math.fsum(
                prompt_mean(grouped[family], key) for family in families
            ) / len(families)

        def gap_metrics(
            *, native_key: str, d320_key: str, candidate_key: str
        ) -> dict[str, object]:
            family_before = tuple(
                math.fsum(
                    abs(float(row[d320_key]) - float(row[native_key]))
                    for row in grouped[family]
                )
                / len(grouped[family])
                for family in families
            )
            family_after = tuple(
                math.fsum(
                    abs(float(row[candidate_key]) - float(row[native_key]))
                    for row in grouped[family]
                )
                / len(grouped[family])
                for family in families
            )
            family_improvements = tuple(
                1.0 - after / max(before, tiny)
                for before, after in zip(family_before, family_after, strict=True)
            )
            before = math.fsum(family_before) / len(family_before)
            after = math.fsum(family_after) / len(family_after)
            return {
                "family_macro_absolute_nll_gap_before": before,
                "family_macro_absolute_nll_gap_after": after,
                "family_macro_relative_absolute_nll_gap_improvement": (
                    1.0 - after / max(before, tiny)
                ),
                "family_strict_gap_improvement_count": sum(
                    after_value < before_value
                    for before_value, after_value in zip(
                        family_before, family_after, strict=True
                    )
                ),
                "worst_family_relative_absolute_nll_gap_improvement": min(
                    family_improvements
                ),
            }

        endpoint = gap_metrics(
            native_key="native_mean_nll",
            d320_key="d320_mean_nll",
            candidate_key="candidate_mean_nll",
        )
        ordinary = gap_metrics(
            native_key="native_ordinary_mean_nll",
            d320_key="d320_ordinary_mean_nll",
            candidate_key="ordinary_candidate_mean_nll",
        )
        summaries[arm] = {
            "candidate_arm": arm,
            "aggregation": (
                "prompt_equal_within_each_family_then_equal_over_eight_families"
            ),
            "prompt_count": len(selected),
            "family_count": len(families),
            "prompts_per_family": 2,
            "family_macro_native_endpoint_mean_nll": family_macro(
                "native_mean_nll"
            ),
            "family_macro_d320_endpoint_mean_nll": family_macro(
                "d320_mean_nll"
            ),
            "family_macro_candidate_endpoint_mean_nll": family_macro(
                "candidate_mean_nll"
            ),
            "family_macro_native_ordinary_mean_nll": family_macro(
                "native_ordinary_mean_nll"
            ),
            "family_macro_d320_ordinary_mean_nll": family_macro(
                "d320_ordinary_mean_nll"
            ),
            "family_macro_candidate_ordinary_mean_nll": family_macro(
                "ordinary_candidate_mean_nll"
            ),
            "endpoint_gap": endpoint,
            "ordinary_gap": ordinary,
            "descriptive_only_non_gating": True,
            "threshold_applied": False,
        }
    return summaries


def _execute_held_three_arm_grid(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    phases: v7diag._JointPhaseResults,
    codecs: Mapping[str, CandidateConditionedK64StateFeatureCodec],
    unit_observations: Mapping[str, Mapping[str, object]],
    unit_behavior: Mapping[str, object],
    pinned_plus: Mapping[str, Mapping[str, object]],
) -> _HeldFiniteGridResult:
    """Execute exactly three arms on two outer-held prompts per family."""

    values = tuple(sorted(traces, key=lambda value: value.example_id))
    schedule = _held_arm_schedule(values)
    families = tuple(sorted(fits))
    fold_records = {
        record.outer_held_family_id: record
        for record in phases.joint_fold_records
    }
    if (
        len(values) != _EXPECTED_PROMPTS
        or len(families) != _EXPECTED_FAMILIES
        or set(codecs) != set(families)
        or set(fold_records) != set(families)
        or set(unit_observations) != {trace.example_id for trace in values}
    ):
        raise RuntimeError("V8 held-only execution universe differs")
    by_family = {
        family: tuple(trace for trace in values if trace.family_id == family)
        for family in families
    }
    if any(len(rows) != 2 for rows in by_family.values()):
        raise RuntimeError("V8 requires two outer-held prompts per family")

    manifests = {
        ledger: {
            trace.example_id: trace.family_id
            for trace in values
            if trace.selected_by_ledger[ledger].numel() > 0
        }
        for ledger in _LEDGERS
    }
    fidelity = {
        arm: {
            ledger: SourceAuthoritativeShadowFidelityAccumulator(
                manifests[ledger], gates=ESTABLISHED_SHADOW_FIDELITY_GATES
            )
            for ledger in _LEDGERS
        }
        for arm in THREE_ARM_FINITE_NAMES
    }
    geometry_traces = tuple(_geometry_trace(trace) for trace in values)
    executed_rows: dict[str, dict[str, Tensor]] = {
        arm: {} for arm in THREE_ARM_FINITE_NAMES
    }
    examples: list[CandidateConditionedK64ThreeArmFiniteExample] = []
    support_receipts: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    plus_replay_receipts: list[dict[str, object]] = []
    native_forwards = 0
    candidate_forwards = 0

    for trace in values:
        held_family = trace.family_id
        fit = fits[held_family]
        record = fold_records[held_family]
        refit = phases.v6_phases.row_bank.refits[held_family]
        codec = codecs[held_family]
        directions = v6diag._ordered_k64(fit)
        base_rows = (
            trace.base_h4.detach()
            .to(device="cpu", dtype=torch.float64)[0]
            .index_select(0, trace.support_indices)
            .contiguous()
        )
        gain_support = build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            codec,
            record.full_scalar_control_fit,
            record.full_joint_fit,
            phase="held",
            example_id=trace.example_id,
            family_id=trace.family_id,
            base_h4_support_rows=base_rows,
            ordered_directions=directions,
        )
        support_receipts.append(gain_support.metadata())
        model_inputs, indices, targets, teacher_logits = (
            v5diag._fresh_native_teacher(context=context, trace=trace)
        )
        native_forwards += 1
        endpoint_indices, endpoint_targets, endpoint_grid = (
            v5diag._endpoint_indices(trace, indices, targets)
        )
        source_selected = frozen._select_sequence_rows(teacher_logits, indices)
        full_scores = complete_h4_tail_gate_scores(
            trace.endpoint, fit.ordered_basis_rows()
        )
        unit = dict(unit_observations[trace.example_id])
        if (
            unit.get("family_id") != held_family
            or unit.get("fold_artifact_sha256") != fit.artifact_sha256
            or unit.get("ordered_directions_sha256")
            != _runtime_tensor_sha256(directions)
            or unit.get("teacher_logits_sha256")
            != _runtime_tensor_sha256(teacher_logits)
            or unit.get("endpoint_supervised_grid_sha256")
            != _runtime_tensor_sha256(endpoint_grid)
            or unit.get("endpoint_supervised_token_count")
            != int(endpoint_grid.shape[0])
            or unit.get("token_score_matrix_sha256")
            != _runtime_tensor_sha256(full_scores)
            or unit.get("arm") != "unit_k64"
            or unit.get("candidate_variant") != "unit"
        ):
            raise RuntimeError("fresh V8 held prompt did not rebind pinned unit")

        token_kl_by_arm: dict[str, Tensor] = {}
        provider_artifacts: list[str] = []
        execution_artifacts: list[str] = []
        correction_row_hashes: list[str] = []
        full_correction_hashes: list[str] = []
        cell_input_bindings: list[tuple[str, ...]] = []
        scheduled_arms = tuple(
            arm
            for scheduled_trace, arm in schedule
            if scheduled_trace.example_id == trace.example_id
        )
        if scheduled_arms != THREE_ARM_FINITE_NAMES:
            raise RuntimeError("V8 held arm schedule drifted")
        for arm in scheduled_arms:
            token_kl, execution, provider, correction_rows = (
                _execute_held_finite_arm(
                    context=context,
                    trace=trace,
                    basis=basis,
                    fit=fit,
                    support=gain_support,
                    arm=arm,
                    model_inputs=model_inputs,
                    teacher_logits=teacher_logits,
                    endpoint_indices=endpoint_indices,
                )
            )
            token_kl_by_arm[arm] = token_kl
            provider_artifacts.append(provider.artifact_sha256)
            execution_artifacts.append(execution.artifact_sha256)
            correction_row_hashes.append(_runtime_tensor_sha256(correction_rows))
            full_correction_hashes.append(provider.correction_sha256)
            cell_input_bindings.append(
                (
                    provider.model_inputs_sha256,
                    _runtime_tensor_sha256(teacher_logits),
                    _runtime_tensor_sha256(endpoint_grid),
                    provider.base_h4_sha256,
                    provider.prefix_artifact_sha256,
                    provider.support_mask_sha256,
                )
            )
            candidate_forwards += 1
            candidate_nll = token_v1._selected_token_nll(
                execution.logits, indices, targets
            )
            candidate_endpoint_nll = token_v1._selected_token_nll(
                execution.logits, endpoint_indices, endpoint_targets
            )
            candidate_selected = frozen._select_sequence_rows(
                execution.logits, indices
            )
            for ledger, selected in trace.selected_by_ledger.items():
                if selected.numel() == 0:
                    continue
                fidelity[arm][ledger].add(
                    ShadowFidelityExample(
                        example_id=trace.example_id,
                        family_id=held_family,
                        source_logits=source_selected.index_select(
                            0, selected.to(source_selected.device)
                        ),
                        candidate_logits=candidate_selected.index_select(
                            0, selected.to(candidate_selected.device)
                        ),
                        targets=targets.index_select(
                            0, selected.to(targets.device)
                        ),
                    )
                )
            actual_rows = (
                execution.candidate_h4.detach()
                .to(device="cpu", dtype=torch.float64)[0]
                .index_select(0, trace.support_indices)
                - trace.base_h4.detach()
                .to(device="cpu", dtype=torch.float64)[0]
                .index_select(0, trace.support_indices)
            ).contiguous()
            expected_executed_rows = _executed_cast_once_delta_rows(
                base_h4=trace.base_h4,
                correction=provider._correction,
                support_mask=provider._support,
                support_indices=trace.support_indices,
            )
            if not torch.equal(actual_rows, expected_executed_rows):
                raise RuntimeError("V8 executed cast-once correction rows differ")
            executed_rows[arm][trace.example_id] = actual_rows
            expected_plus = pinned_plus.get(trace.example_id)
            replayed_plus = None
            if arm == "v5_static_plus" and expected_plus is not None:
                if (
                    expected_plus.get("family_id") != held_family
                    or expected_plus.get("fold_artifact_sha256")
                    != fit.artifact_sha256
                    or expected_plus.get("ordered_directions_sha256")
                    != provider.ordered_directions_sha256
                    or expected_plus.get("gains_sha256") != provider.gains_sha256
                    or expected_plus.get("teacher_logits_sha256")
                    != _runtime_tensor_sha256(teacher_logits)
                    or expected_plus.get("endpoint_supervised_grid_sha256")
                    != _runtime_tensor_sha256(endpoint_grid)
                    or expected_plus.get("token_teacher_kl_sha256")
                    != _runtime_tensor_sha256(token_kl)
                ):
                    raise RuntimeError("V8 static plus did not replay selected V5")
                replayed_plus = True
                plus_receipt = {
                    "example_id": trace.example_id,
                    "family_id": held_family,
                    "pinned_v5_observation_sha256": expected_plus[
                        "observation_sha256"
                    ],
                    "pinned_v5_token_teacher_kl_sha256": expected_plus[
                        "token_teacher_kl_sha256"
                    ],
                    "live_v8_token_teacher_kl_sha256": (
                        _runtime_tensor_sha256(token_kl)
                    ),
                    "selected_v5_plus_replayed_exactly": True,
                    "raw_tensors_serialized": False,
                }
                plus_receipt["receipt_sha256"] = token_v1._domain_sha256(
                    plus_receipt, domain=_V5_PLUS_BINDING_DOMAIN
                )
                plus_replay_receipts.append(plus_receipt)
            observation = {
                "example_id": trace.example_id,
                "family_id": held_family,
                "held_family_id": held_family,
                "arm": arm,
                "rank": CANDIDATE_GAIN_RANK,
                "fold_artifact_sha256": fit.artifact_sha256,
                "refit_artifact_sha256": refit.artifact_sha256,
                "codec_artifact_sha256": codec.artifact_sha256,
                "scalar_fit_artifact_sha256": (
                    record.full_scalar_control_fit.artifact_sha256
                ),
                "joint_fit_artifact_sha256": (
                    record.full_joint_fit.artifact_sha256
                ),
                "gain_support_artifact_sha256": gain_support.artifact_sha256,
                "pinned_v4_unit_observation_sha256": unit[
                    "observation_sha256"
                ],
                "pinned_v4_unit_token_teacher_kl_sha256": unit[
                    "token_teacher_kl_sha256"
                ],
                "pinned_v4_unit_mean_teacher_kl": float(
                    unit["complete_h4_support_mean_teacher_kl"]
                ),
                "ordered_directions_sha256": provider.ordered_directions_sha256,
                "gains_sha256": provider.gains_sha256,
                "provider_artifact_sha256": provider.artifact_sha256,
                "execution_artifact_sha256": execution.artifact_sha256,
                "model_inputs_sha256": provider.model_inputs_sha256,
                "bridge_binding_sha256": provider.bridge_binding_sha256,
                "prefix_artifact_sha256": provider.prefix_artifact_sha256,
                "base_h4_sha256": provider.base_h4_sha256,
                "support_mask_sha256": provider.support_mask_sha256,
                "correction_sha256": provider.correction_sha256,
                "analytic_correction_rows_sha256": _runtime_tensor_sha256(
                    correction_rows
                ),
                "teacher_logits_sha256": _runtime_tensor_sha256(
                    teacher_logits
                ),
                "endpoint_supervised_grid_sha256": _runtime_tensor_sha256(
                    endpoint_grid
                ),
                "endpoint_supervised_token_count": int(endpoint_grid.shape[0]),
                "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
                "complete_h4_support_mean_teacher_kl": float(token_kl.mean()),
                "native_mean_nll": float(trace.native_token_nll.mean()),
                "d320_mean_nll": float(trace.d320_token_nll.mean()),
                "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
                "native_ordinary_mean_nll": float(
                    trace.native_ordinary_token_nll.mean()
                ),
                "d320_ordinary_mean_nll": float(
                    trace.d320_ordinary_token_nll.mean()
                ),
                "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
                "candidate_h4_bitwise_native": token_v1._bitwise_equal(
                    execution.candidate_h4.detach().to(device="cpu"),
                    trace.native_h4,
                ),
                "candidate_logits_bitwise_native": (
                    _runtime_tensor_sha256(execution.logits)
                    == trace.native_logits_sha256
                ),
                "executed_correction_rows_sha256": _runtime_tensor_sha256(
                    actual_rows
                ),
                "analytic_and_executed_rows_separated_for_cast_once_semantics": (
                    True
                ),
                "selected_v5_plus_replayed_exactly": replayed_plus,
                "held_family_used_for_fit_or_tune": False,
                "held_family_excluded_from_all_fits": True,
                "candidate_executed": True,
                "raw_tensors_serialized": False,
            }
            observation["observation_sha256"] = token_v1._domain_sha256(
                observation, domain=_OBSERVATION_DOMAIN
            )
            observations.append(observation)
            del (
                token_kl,
                execution,
                provider,
                correction_rows,
                candidate_nll,
                candidate_endpoint_nll,
                candidate_selected,
                actual_rows,
            )

        if len(set(cell_input_bindings)) != 1:
            raise RuntimeError("V8 arms did not share exact native inputs")

        examples.append(
            CandidateConditionedK64ThreeArmFiniteExample(
                gain_support=gain_support,
                model_inputs_sha256=trace.model_inputs_sha256,
                bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
                prefix_artifact_sha256=trace.prefix.artifact_sha256,
                support_mask_sha256=_runtime_tensor_sha256(
                    trace.prefix.complete_h4_causal_support_mask()
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                ),
                teacher_logits_sha256=_runtime_tensor_sha256(teacher_logits),
                endpoint_supervised_grid_sha256=_runtime_tensor_sha256(
                    endpoint_grid
                ),
                pinned_unit_receipt_sha256=str(unit["observation_sha256"]),
                pinned_unit_token_teacher_kl_sha256=str(
                    unit["token_teacher_kl_sha256"]
                ),
                pinned_unit_mean_teacher_kl=float(
                    unit["complete_h4_support_mean_teacher_kl"]
                ),
                arm_provider_artifact_sha256s=tuple(provider_artifacts),
                arm_execution_artifact_sha256s=tuple(execution_artifacts),
                arm_correction_rows_sha256s=tuple(correction_row_hashes),
                arm_full_correction_sha256s=tuple(full_correction_hashes),
                pinned_v5_static_plus_token_teacher_kl_sha256=(
                    None
                    if trace.example_id not in pinned_plus
                    else str(
                        pinned_plus[trace.example_id][
                            "token_teacher_kl_sha256"
                        ]
                    )
                ),
                static_plus_token_teacher_kl=token_kl_by_arm[
                    "v5_static_plus"
                ],
                exact_scalar_token_teacher_kl=token_kl_by_arm[
                    "v6_exact_scalar"
                ],
                joint_token_teacher_kl=token_kl_by_arm["v7_joint"],
            )
        )
        del (
            gain_support,
            model_inputs,
            indices,
            targets,
            teacher_logits,
            endpoint_indices,
            endpoint_targets,
            endpoint_grid,
            source_selected,
            full_scores,
            token_kl_by_arm,
        )

    if (
        native_forwards != _EXPECTED_PROMPTS
        or candidate_forwards != _EXPECTED_CANDIDATE_FORWARDS
        or len(examples) != _EXPECTED_PROMPTS
        or len(support_receipts) != _EXPECTED_PROMPTS
        or len(observations) != _EXPECTED_CANDIDATE_FORWARDS
        or len(plus_replay_receipts) != 12
    ):
        raise RuntimeError("V8 held-only finite resource grid differs")
    comparison = compare_candidate_conditioned_k64_three_arm_finite_examples(
        tuple(examples), phase="held"
    )
    behavior = {
        arm: {
            ledger: fidelity[arm][ledger].finalize() for ledger in _LEDGERS
        }
        for arm in THREE_ARM_FINITE_NAMES
    }
    geometry = {
        arm: ladder._geometry_with_examples(
            geometry_traces,
            executed_rows[arm],
            candidate_semantics=(
                "actual_cast_once_d320_plus_frozen_"
                f"{arm}_complete_h4_tail_k64"
            ),
        )
        for arm in THREE_ARM_FINITE_NAMES
    }
    top1 = {
        arm: {
            **v5diag._top1_comparison(
                unit_behavior=unit_behavior,
                selected_behavior=behavior[arm],
            ),
            "candidate_arm": arm,
        }
        for arm in THREE_ARM_FINITE_NAMES
    }
    ordered_observations = tuple(
        sorted(
            observations,
            key=lambda row: (
                str(row["example_id"]),
                THREE_ARM_FINITE_NAMES.index(str(row["arm"])),
            ),
        )
    )
    nll = _descriptive_nll_summary_by_arm(ordered_observations)
    observation_set_sha256 = token_v1._domain_sha256(
        tuple(str(row["observation_sha256"]) for row in ordered_observations),
        domain=_OBSERVATION_SET_DOMAIN,
    )
    plus_payload = {
        "selected_v5_plus_exact_replay_count": len(plus_replay_receipts),
        "counterfactual_v5_plus_execution_count": 4,
        "selected_v5_plus_replay_receipts": tuple(plus_replay_receipts),
        "all_available_selected_v5_plus_rows_replayed_exactly": True,
        "all_16_plus_rows_claimed_as_previously_executed": False,
        "raw_tensors_serialized": False,
    }
    plus_payload["artifact_sha256"] = token_v1._domain_sha256(
        plus_payload, domain=_V5_PLUS_BINDING_DOMAIN
    )
    return _HeldFiniteGridResult(
        examples=tuple(examples),
        comparison=comparison,
        gain_support_receipts=tuple(support_receipts),
        observation_receipts=ordered_observations,
        observation_set_sha256=observation_set_sha256,
        behavior_by_arm=behavior,
        nll_by_arm=nll,
        geometry_by_arm=geometry,
        top1_comparison_by_arm=top1,
        plus_replay_binding=plus_payload,
        resources={
            "held_native_forward_count": native_forwards,
            "held_candidate_forward_count": candidate_forwards,
            "held_static_plus_forward_count": _EXPECTED_PROMPTS,
            "held_exact_scalar_forward_count": _EXPECTED_PROMPTS,
            "held_joint_forward_count": _EXPECTED_PROMPTS,
            "held_prompt_count": len(examples),
            "candidate_arm_count": len(THREE_ARM_FINITE_NAMES),
            "finite_backward_call_count": 0,
        },
    )


@dataclass(slots=True)
class _V8AnalyticPhaseResults:
    v7_phases: v7diag._JointPhaseResults
    v7_binding: Mapping[str, object]
    codecs: Mapping[str, CandidateConditionedK64StateFeatureCodec]
    codec_binding: Mapping[str, object]
    unit_observations: Mapping[str, Mapping[str, object]]
    unit_behavior: Mapping[str, object]
    unit_geometry: Mapping[str, object]
    unit_binding: Mapping[str, object]
    pinned_plus: Mapping[str, Mapping[str, object]]
    pinned_plus_binding: Mapping[str, object]


def _execute_v8_analytic_phases(
    *,
    context: object,
    parent: Mapping[str, object],
    v3_report: Mapping[str, object],
    v4_report: Mapping[str, object],
    v5_report: Mapping[str, object],
    v6_report: Mapping[str, object],
    v7_report: Mapping[str, object],
    traces: Sequence[object],
    endpoint_resources: Mapping[str, int],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> _V8AnalyticPhaseResults:
    """Authenticate all V7 evidence and frozen inputs before finite work."""

    v7_phases = v7diag._execute_joint_phases(
        context=context,
        parent=parent,
        v3_report=v3_report,
        v4_report=v4_report,
        v5_report=v5_report,
        v6_report=v6_report,
        traces=traces,
        endpoint_resources=endpoint_resources,
        basis=basis,
        fits=fits,
    )
    v7_binding = _authenticate_live_v7_evidence(
        v7_report=v7_report, phases=v7_phases
    )
    codecs, codec_binding = _reconstruct_full_v7_codecs(
        phases=v7_phases, fits=fits
    )
    (
        unit_observations,
        unit_behavior,
        unit_geometry,
        unit_binding,
    ) = v5diag._authenticate_v4_unit_final_baseline(v4_report=v4_report)
    pinned_plus, pinned_plus_binding = _pinned_v5_selected_plus_rows(v5_report)
    return _V8AnalyticPhaseResults(
        v7_phases=v7_phases,
        v7_binding=v7_binding,
        codecs=codecs,
        codec_binding=codec_binding,
        unit_observations=unit_observations,
        unit_behavior=unit_behavior,
        unit_geometry=unit_geometry,
        unit_binding=unit_binding,
        pinned_plus=pinned_plus,
        pinned_plus_binding=pinned_plus_binding,
    )


def _pairwise_finite_gate_results(
    comparison: CandidateConditionedK64ThreeArmFiniteComparison,
) -> tuple[dict[str, object], dict[str, bool]]:
    """Apply the unchanged V5 finite thresholds to both required controls."""

    metadata = comparison.metadata()
    raw_pairwise = metadata.get("pairwise")
    if not isinstance(raw_pairwise, Mapping):
        raise ValueError("V8 finite pairwise metadata differs")
    rows: dict[str, object] = {}
    gates: dict[str, bool] = {}
    for source in ("v6_exact_scalar", "v5_static_plus"):
        key = f"v7_joint_vs_{source}"
        raw = raw_pairwise.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"V8 finite pairwise row {key} differs")
        joint_mean = comparison.family_equal_mean_teacher_kl("v7_joint")
        source_mean = comparison.family_equal_mean_teacher_kl(source)
        relative_improvement = (
            (source_mean - joint_mean)
            / max(source_mean, torch.finfo(torch.float64).tiny)
        )
        deltas = tuple(
            float(value)
            for value in raw["outer_family_delta_candidate_minus_source"]
        )
        family_wins = sum(value < 0.0 for value in deltas)
        all_within_cap = all(
            joint <= _KL_WORST_FAMILY_MULTIPLIER * control
            + _KL_WORST_FAMILY_ABSOLUTE_TOLERANCE
            for joint, control in zip(
                comparison.outer_family_mean_teacher_kl[:, 3],
                comparison.outer_family_mean_teacher_kl[
                    :, THREE_ARM_FINITE_NAMES.index(source) + 1
                ],
                strict=True,
            )
        )
        if (
            float(raw["family_equal_delta_candidate_minus_source"])
            != joint_mean - source_mean
            or int(raw["outer_family_strict_improvement_count"])
            != family_wins
        ):
            raise RuntimeError("V8 core and diagnostic pairwise results differ")
        prefix = "joint_vs_scalar" if source == "v6_exact_scalar" else "joint_vs_plus"
        rows[prefix] = {
            "candidate_arm": "v7_joint",
            "control_arm": source,
            "joint_family_equal_mean_teacher_kl": joint_mean,
            "control_family_equal_mean_teacher_kl": source_mean,
            "family_macro_relative_improvement": relative_improvement,
            "held_family_improvement_count": family_wins,
            "held_family_delta_joint_minus_control": deltas,
            "all_held_families_within_five_percent_plus_1e_minus_8": (
                all_within_cap
            ),
        }
        gates[
            f"{prefix}_family_macro_teacher_kl_improves_at_least_2pct"
        ] = relative_improvement >= _KL_RELATIVE_IMPROVEMENT_MINIMUM
        gates[f"{prefix}_held_family_improvement_count_at_least_6_of_8"] = (
            family_wins >= _KL_FAMILY_WIN_MINIMUM
        )
        gates[
            f"{prefix}_worst_family_regression_at_most_5pct_plus_1e_minus_8"
        ] = all_within_cap
    return rows, gates


def _joint_top1_gate_results(
    top1_comparison_by_arm: Mapping[str, Mapping[str, object]],
) -> dict[str, bool]:
    """Reuse V5's approximate top-1 checks, with joint as the only gate arm."""

    joint = top1_comparison_by_arm.get("v7_joint")
    if not isinstance(joint, Mapping):
        raise ValueError("V8 joint top1 comparison differs")
    gates: dict[str, bool] = {}
    for ledger in ("ordinary", "complete_h4_support", "graph_core"):
        row = joint.get(ledger)
        if not isinstance(row, Mapping):
            raise ValueError(f"V8 joint {ledger} top1 row differs")
        unit_aggregate = float(row["unit_aggregate_top1"])
        joint_aggregate = float(row["candidate_aggregate_top1"])
        unit_macro = float(row["unit_family_macro_top1"])
        joint_macro = float(row["candidate_family_macro_top1"])
        expected = {
            "aggregate_at_least_point_90": joint_aggregate >= 0.90,
            "family_macro_at_least_point_90": joint_macro >= 0.90,
            "aggregate_no_material_regression_vs_unit": (
                joint_aggregate >= unit_aggregate - 0.01
            ),
            "family_macro_no_material_regression_vs_unit": (
                joint_macro >= unit_macro - 0.01
            ),
        }
        if any(bool(row[name]) is not value for name, value in expected.items()):
            raise RuntimeError(f"V8 joint {ledger} top1 boolean drifted")
        gates[f"joint_{ledger}_aggregate_top1_at_least_point_90"] = bool(
            row["aggregate_at_least_point_90"]
        )
        gates[f"joint_{ledger}_family_macro_top1_at_least_point_90"] = bool(
            row["family_macro_at_least_point_90"]
        )
        gates[
            f"joint_{ledger}_aggregate_top1_no_material_regression_vs_unit"
        ] = bool(row["aggregate_no_material_regression_vs_unit"])
        gates[
            f"joint_{ledger}_family_macro_top1_no_material_regression_vs_unit"
        ] = bool(row["family_macro_no_material_regression_vs_unit"])
    return gates


def _row_bank_candidate_support_row_executions(
    phases: v7diag._JointPhaseResults,
) -> int:
    """Count the support rows actually projected by V7's 56 analytic cells."""

    cells_by_fold = phases.v6_phases.row_bank.cells
    if (
        len(cells_by_fold) != _EXPECTED_FAMILIES
        or any(len(cells_by_fold[held]) != 7 for held in sorted(cells_by_fold))
    ):
        raise RuntimeError("V8 analytic row-bank fold count differs")
    cells = tuple(cell for held in sorted(cells_by_fold) for cell in cells_by_fold[held])
    if len(cells) != 56:
        raise RuntimeError("V8 analytic row-bank cell count differs")
    counts: list[int] = []
    for cell in cells:
        rows = cell.base_h4_support_rows
        if (
            not isinstance(rows, Tensor)
            or rows.ndim != 2
            or rows.shape[1] != token_v1._WIDTH
        ):
            raise RuntimeError("V8 analytic support-row geometry differs")
        counts.append(int(rows.shape[0]))
    total = sum(counts)
    if total != 2842:
        raise RuntimeError("V8 analytic support-row execution count differs")
    return total


def _resource_accounting(
    *,
    endpoint_resources: Mapping[str, int],
    gradient_resources: Mapping[str, int],
    finite_resources: Mapping[str, int],
    row_bank_candidate_support_row_executions: int,
) -> dict[str, object]:
    parent_forwards = (
        endpoint_resources["base_forward_count"]
        + endpoint_resources["native_forward_count"]
        + endpoint_resources["endpoint_token_vjp_forward_count"]
    )
    parent_backwards = endpoint_resources[
        "endpoint_token_vjp_backward_call_count"
    ]
    analytic_forwards = (
        gradient_resources["gradient_native_forward_count"]
        + gradient_resources["gradient_candidate_vjp_forward_count"]
    )
    analytic_backwards = gradient_resources[
        "gradient_candidate_vjp_backward_call_count"
    ]
    finite_forwards = (
        finite_resources["held_native_forward_count"]
        + finite_resources["held_candidate_forward_count"]
    )
    row_bank_support_executions = row_bank_candidate_support_row_executions
    held_support_executions = (
        endpoint_resources["complete_h4_support_row_count"] * _EXPECTED_ARMS
    )
    total_candidate_support_executions = (
        row_bank_support_executions + held_support_executions
    )
    total_candidate_executions = (
        gradient_resources["gradient_candidate_vjp_forward_count"]
        + finite_resources["held_candidate_forward_count"]
    )
    d320_logical_macs = total_candidate_support_executions * 2 * 640 * 320
    k64_logical_macs = (
        total_candidate_support_executions
        * 2
        * 640
        * CANDIDATE_GAIN_RANK
    )
    total_forwards = parent_forwards + analytic_forwards + finite_forwards
    total_backwards = parent_backwards + analytic_backwards
    if (
        parent_forwards != 48
        or parent_backwards != 109
        or analytic_forwards != 64
        or analytic_backwards != 385
        or finite_forwards != _EXPECTED_FINITE_FORWARDS
        or finite_resources["held_candidate_forward_count"]
        != _EXPECTED_CANDIDATE_FORWARDS
        or finite_resources["finite_backward_call_count"] != 0
        or row_bank_support_executions != 2842
        or held_support_executions != 2457
        or total_candidate_support_executions != 5299
        or total_candidate_executions != 104
        or d320_logical_macs != 2170470400
        or k64_logical_macs != 434094080
        or total_forwards != _EXPECTED_TOTAL_FORWARDS
        or total_backwards != _EXPECTED_TOTAL_BACKWARDS
    ):
        raise RuntimeError("V8 finite resource accounting differs")
    return {
        **endpoint_resources,
        **gradient_resources,
        **finite_resources,
        "phase_order": (
            "parent_endpoint_recollection",
            "exact_v7_analytic_phase_reproduction_and_authentication",
            "frozen_full_seven_codec_reconstruction",
            "pinned_unit_and_available_v5_plus_authentication",
            "held_only_three_arm_finite_execution",
            "prompt_then_family_equal_scalar_report_publication",
        ),
        "parent_collection_model_forward_count": parent_forwards,
        "parent_collection_backward_call_count": parent_backwards,
        "shared_analytic_model_forward_count": analytic_forwards,
        "shared_analytic_backward_call_count": analytic_backwards,
        "finite_stage_model_forward_count": finite_forwards,
        "row_bank_candidate_support_row_executions": (
            row_bank_support_executions
        ),
        "held_three_arm_candidate_support_row_executions": (
            held_support_executions
        ),
        "total_candidate_support_row_executions": (
            total_candidate_support_executions
        ),
        "analytic_and_finite_candidate_execution_count": (
            total_candidate_executions
        ),
        "analysis_only_d320_supported_projection_logical_macs": (
            d320_logical_macs
        ),
        "analysis_only_k64_tail_projection_logical_macs": k64_logical_macs,
        "projection_mac_scope": (
            "cpu_analysis_correction_materialization_not_model_kernel_speed"
        ),
        "kernel_or_serving_speed_claim": False,
        "tune_model_forward_count": 0,
        "selection_model_forward_count": 0,
        "total_model_forward_count": total_forwards,
        "total_backward_call_count": total_backwards,
        "exact_model_forward_count_is_176": total_forwards
        == _EXPECTED_TOTAL_FORWARDS,
        "exact_backward_call_count_is_494": total_backwards
        == _EXPECTED_TOTAL_BACKWARDS,
        "raw_finite_logits_or_tensors_retained_in_report": False,
        "serving_learned_parameter_count": "not_applicable_finite_diagnostic_only",
        "serving_logical_macs_per_token": "not_applicable_finite_diagnostic_only",
    }


def _safety_metadata() -> dict[str, object]:
    return {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_logits": False,
        "contains_activation_tensors": False,
        "contains_gradient_tensors": False,
        "contains_state_feature_tensors": False,
        "contains_gain_vectors": False,
        "contains_joint_parameter_vectors": False,
        "contains_token_teacher_kl_tensors": False,
        "contains_only_hashes_counts_and_scalar_metrics": True,
        "artifact_must_remain_outside_git": True,
    }


def _integrity_gate_results(
    *,
    analytic: _V8AnalyticPhaseResults,
    finite: _HeldFiniteGridResult,
    resources: Mapping[str, object],
) -> dict[str, bool]:
    arm_counts = {
        arm: sum(row["arm"] == arm for row in finite.observation_receipts)
        for arm in THREE_ARM_FINITE_NAMES
    }
    return {
        "exact_v7_file_and_logical_hash_authenticated": (
            analytic.v7_binding["v7_report_file_sha256"]
            == V7_REPORT_FILE_SHA256
            and analytic.v7_binding["v7_report_sha256"] == V7_REPORT_SHA256
        ),
        "live_recollection_canonically_reproduced_v7_before_finite": (
            analytic.v7_binding["authenticated_before_any_finite_forward"]
            is True
            and analytic.v7_binding["screen_metadata_canonically_equal"]
            is True
            and analytic.v7_binding["fold_metadata_canonically_equal"] is True
            and analytic.v7_binding[
                "all_56_inner_metadata_rows_canonically_equal"
            ]
            is True
        ),
        "all_eight_full_seven_codecs_reproduced_without_held_rows": (
            analytic.codec_binding["full_seven_codec_count"]
            == _EXPECTED_FAMILIES
            and analytic.codec_binding[
                "every_codec_reproduces_v7_full_joint_fit"
            ]
            is True
            and analytic.codec_binding[
                "held_rows_used_for_codec_fit_or_recenter"
            ]
            is False
        ),
        "pinned_unit_reference_covers_all_16_prompts_without_reexecution": (
            len(analytic.unit_observations) == _EXPECTED_PROMPTS
            and resources["unit_reference_model_forward_count"] == 0
        ),
        "all_12_available_v5_selected_plus_rows_replayed_exactly": (
            finite.plus_replay_binding[
                "selected_v5_plus_exact_replay_count"
            ]
            == 12
            and finite.plus_replay_binding[
                "all_available_selected_v5_plus_rows_replayed_exactly"
            ]
            is True
        ),
        "four_counterfactual_plus_rows_reported_without_false_replay_claim": (
            finite.plus_replay_binding[
                "counterfactual_v5_plus_execution_count"
            ]
            == 4
            and finite.plus_replay_binding[
                "all_16_plus_rows_claimed_as_previously_executed"
            ]
            is False
        ),
        "exactly_16_outer_held_prompt_examples_present": (
            len(finite.examples) == _EXPECTED_PROMPTS
            and finite.comparison.cell_count == _EXPECTED_PROMPTS
            and tuple(finite.comparison.prompt_count_by_outer_family)
            == (2,) * _EXPECTED_FAMILIES
        ),
        "exactly_three_canonical_arms_executed_once_per_prompt": (
            arm_counts
            == {arm: _EXPECTED_PROMPTS for arm in THREE_ARM_FINITE_NAMES}
            and len(finite.observation_receipts)
            == _EXPECTED_CANDIDATE_FORWARDS
        ),
        "no_tune_selection_refit_damping_or_fallback_executed": (
            resources["tune_model_forward_count"] == 0
            and resources["selection_model_forward_count"] == 0
        ),
        "exact_model_forward_count_is_176": (
            resources["total_model_forward_count"] == _EXPECTED_TOTAL_FORWARDS
        ),
        "exact_backward_call_count_is_494": (
            resources["total_backward_call_count"] == _EXPECTED_TOTAL_BACKWARDS
        ),
        "exact_held_candidate_forward_count_is_48": (
            resources["held_candidate_forward_count"]
            == _EXPECTED_CANDIDATE_FORWARDS
        ),
    }


def _classification(
    *,
    integrity_passed: bool,
    pairwise_rows: Mapping[str, object],
    pairwise_gates: Mapping[str, bool],
    top1_gates: Mapping[str, bool],
) -> str:
    if not integrity_passed:
        return "integrity_failure"
    scalar = pairwise_rows.get("joint_vs_scalar")
    plus = pairwise_rows.get("joint_vs_plus")
    if not isinstance(scalar, Mapping) or not isinstance(plus, Mapping):
        return "integrity_failure"
    scalar_macro_better = float(
        scalar["joint_family_equal_mean_teacher_kl"]
    ) < float(scalar["control_family_equal_mean_teacher_kl"])
    plus_macro_better = float(plus["joint_family_equal_mean_teacher_kl"]) < float(
        plus["control_family_equal_mean_teacher_kl"]
    )
    if not scalar_macro_better:
        return "analytic_to_finite_attribution_failure_same_a"
    if not plus_macro_better:
        return "no_improvement_over_static_plus_carrier_same_a"
    cap_gates = tuple(
        value
        for name, value in pairwise_gates.items()
        if name.endswith("worst_family_regression_at_most_5pct_plus_1e_minus_8")
    )
    if len(cap_gates) != 2 or not all(cap_gates):
        return "unstable_family_regression_same_a"
    usefulness_gates = tuple(
        value
        for name, value in pairwise_gates.items()
        if name.endswith("family_macro_teacher_kl_improves_at_least_2pct")
        or name.endswith("held_family_improvement_count_at_least_6_of_8")
    )
    if len(usefulness_gates) != 4 or not all(usefulness_gates):
        return "below_predeclared_useful_effect_or_insufficient_breadth_same_a"
    if not all(top1_gates.values()):
        return "joint_finite_failed_approximate_top1_safety_same_a"
    return "joint_finite_cleared_both_controls_and_top1_same_a"


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_finite_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    v5_report_path: Path | str = DEFAULT_V5_REPORT,
    v6_report_path: Path | str = DEFAULT_V6_REPORT,
    v7_report_path: Path | str = DEFAULT_V7_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the locked same-A held-only V8 three-arm finite diagnostic."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite candidate joint state-gain V8 report"
        )
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = v4diag._load_v3_report(v3_report_path)
    v4_report = v5diag._load_v4_report(v4_report_path)
    v5_report = v6diag._load_v5_report(v5_report_path)
    v6_report = v7diag._load_v6_report(v6_report_path)
    v7_report = _load_v7_report(v7_report_path)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate joint state-gain V8 rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate joint state-gain V8 rank320 transfer",
    )
    transfer_receipts = expanded._transfer_receipts(transfer)
    basis, basis_binding, materialization_binding = _load_committed_basis(
        materialization_report_path=materialization_report_path,
        expected_materialization_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        basis_sidecar_path=basis_sidecar_path,
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        traces, endpoint_resources = token_v1._collect_endpoint_traces(
            context=context,
            basis=basis,
            basis_binding=basis_binding,
            transfer_receipts=transfer_receipts,
        )
        families = tuple(sorted({trace.family_id for trace in traces}))
        if (
            len(traces) != _EXPECTED_PROMPTS
            or len(families) != _EXPECTED_FAMILIES
        ):
            raise RuntimeError("candidate joint state-gain V8 A16 panel differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        analytic = _execute_v8_analytic_phases(
            context=context,
            parent=parent,
            v3_report=v3_report,
            v4_report=v4_report,
            v5_report=v5_report,
            v6_report=v6_report,
            v7_report=v7_report,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        finite = _execute_held_three_arm_grid(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            phases=analytic.v7_phases,
            codecs=analytic.codecs,
            unit_observations=analytic.unit_observations,
            unit_behavior=analytic.unit_behavior,
            pinned_plus=analytic.pinned_plus,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    resources = _resource_accounting(
        endpoint_resources=endpoint_resources,
        gradient_resources=analytic.v7_phases.v6_phases.row_bank.resources,
        finite_resources=finite.resources,
        row_bank_candidate_support_row_executions=(
            _row_bank_candidate_support_row_executions(analytic.v7_phases)
        ),
    )
    resources["unit_reference_model_forward_count"] = 0
    comparison_metadata = finite.comparison.metadata()
    pairwise_rows, pairwise_gates = _pairwise_finite_gate_results(
        finite.comparison
    )
    core_gates = comparison_metadata.get("joint_vs_each_control_gates")
    if not isinstance(core_gates, Mapping):
        raise RuntimeError("V8 pure-core held verdict metadata differs")
    for source, prefix in (
        ("v6_exact_scalar", "joint_vs_scalar"),
        ("v5_static_plus", "joint_vs_plus"),
    ):
        row = core_gates.get(source)
        expected_individual = {
            "family_macro_relative_improvement_at_least_2pct": pairwise_gates[
                f"{prefix}_family_macro_teacher_kl_improves_at_least_2pct"
            ],
            "held_family_improvement_count_at_least_6_of_8": pairwise_gates[
                f"{prefix}_held_family_improvement_count_at_least_6_of_8"
            ],
            "every_family_within_1_05_times_control_plus_1e_minus_8": (
                pairwise_gates[
                    f"{prefix}_worst_family_regression_at_most_5pct_plus_1e_minus_8"
                ]
            ),
        }
        if (
            not isinstance(row, Mapping)
            or dict(row.get("gates", ())) != expected_individual
            or bool(row.get("passed")) != all(expected_individual.values())
        ):
            raise RuntimeError("V8 pure-core and diagnostic finite gates differ")
    pairwise_passed = all(pairwise_gates.values())
    if (
        finite.comparison.joint_vs_exact_scalar_passed
        != all(
            value
            for name, value in pairwise_gates.items()
            if name.startswith("joint_vs_scalar_")
        )
        or finite.comparison.joint_vs_static_plus_passed
        != all(
            value
            for name, value in pairwise_gates.items()
            if name.startswith("joint_vs_plus_")
        )
        or finite.comparison.joint_cleared_both_controls != pairwise_passed
        or bool(comparison_metadata.get("joint_cleared_both_controls"))
        != pairwise_passed
    ):
        raise RuntimeError("V8 pure-core joint verdict differs")
    top1_gates = _joint_top1_gate_results(finite.top1_comparison_by_arm)
    integrity_gates = _integrity_gate_results(
        analytic=analytic,
        finite=finite,
        resources=resources,
    )
    integrity_passed = all(integrity_gates.values())
    passed = integrity_passed and pairwise_passed and all(top1_gates.values())
    classification = _classification(
        integrity_passed=integrity_passed,
        pairwise_rows=pairwise_rows,
        pairwise_gates=pairwise_gates,
        top1_gates=top1_gates,
    )
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_hypothesis_use_only",
            "held_only_finite_validation": True,
            "outer_held_prompt_count": _EXPECTED_PROMPTS,
            "outer_family_count": _EXPECTED_FAMILIES,
            "prompts_per_outer_family": 2,
            "canonical_arm_order": THREE_ARM_FINITE_NAMES,
            "v5_static_plus_role": (
                "forced_counterfactual_plus_one_over_64_all_eight_folds"
            ),
            "v6_exact_scalar_role": "exact_full_seven_v6_scalar_control",
            "v7_joint_role": (
                "exact_full_seven_joint_u_plus_pre_gate_z_dot_w"
            ),
            "state_feature_source": "pre_gate_bridge_base_H4_support_rows",
            "codec_fit": "frozen_full_seven_without_held_rows_or_recentering",
            "aggregation": (
                "two_prompts_equal_within_family_then_equal_over_eight_families"
            ),
            "teacher_KL_primary": True,
            "NLL_and_geometry_descriptive_only": True,
            "tune_phase_executed": False,
            "arm_selection_performed": False,
            "finite_refit_performed": False,
            "step_or_damping_grid_searched": False,
            "per_family_fallback_or_routing_allowed": False,
            "posthoc_hyperparameter_search_performed": False,
            "V5_thresholds_reused_unchanged": {
                "family_macro_relative_improvement_minimum": (
                    _KL_RELATIVE_IMPROVEMENT_MINIMUM
                ),
                "held_family_strict_win_minimum": _KL_FAMILY_WIN_MINIMUM,
                "worst_family_cap": (
                    "joint_le_1.05_times_control_plus_1e_minus_8"
                ),
                "applied_independently_against": (
                    "v6_exact_scalar",
                    "v5_static_plus",
                ),
            },
            "top1_threshold": _TOP1_THRESHOLD,
            "top1_no_material_regression_tolerance": (
                _TOP1_REGRESSION_TOLERANCE
            ),
            "causal_tail_top1_reported_but_excluded_from_primary": True,
            "finite_result_can_authorize_serving_or_mutation": False,
        },
        "v7_control_binding": {
            "file": str(v7_report_path),
            "file_sha256": V7_REPORT_FILE_SHA256,
            "report_sha256": V7_REPORT_SHA256,
            "schema": v7_report.get("schema"),
            "classification": v7_report.get("classification"),
            "live_evidence_reproduction": analytic.v7_binding,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": (
                token_v1.MATERIALIZATION_REPORT_FILE_SHA256
            ),
            "materialization_report_sha256": (
                token_v1.MATERIALIZATION_REPORT_SHA256
            ),
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": token_v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": token_v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
        },
        "prompt_role_receipt": {
            "artifact_sha256": analytic.v7_phases.v6_phases.roles.artifact_sha256,
            "analytic_fit_example_ids": (
                analytic.v7_phases.v6_phases.roles.fit_example_ids
            ),
            "held_finite_example_ids": tuple(
                trace.example_id
                for trace in sorted(traces, key=lambda value: value.example_id)
            ),
            "every_prompt_is_outer_held_for_its_own_family_fold": True,
            "no_prompt_used_to_refit_or_select_an_arm_in_v8": True,
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": v5diag._endpoint_prompt_receipts(traces),
        "full_seven_codec_binding": analytic.codec_binding,
        "pinned_v4_unit_reference_binding": analytic.unit_binding,
        "pinned_v5_plus_availability_binding": analytic.pinned_plus_binding,
        "live_v5_plus_replay_binding": finite.plus_replay_binding,
        "gain_support_receipts": finite.gain_support_receipts,
        "finite_example_receipts": tuple(
            example.metadata() for example in finite.examples
        ),
        "finite_observation_receipts": finite.observation_receipts,
        "finite_observation_set_sha256": finite.observation_set_sha256,
        "held_three_arm_comparison": comparison_metadata,
        "joint_pairwise_control_results": pairwise_rows,
        "behavioral_fidelity_by_arm": finite.behavior_by_arm,
        "descriptive_nll_by_arm_non_gating": finite.nll_by_arm,
        "approximate_top1_comparison_vs_pinned_unit_by_arm": (
            finite.top1_comparison_by_arm
        ),
        "geometry_by_arm_descriptive_only": finite.geometry_by_arm,
        "integrity_gate_results": tuple(sorted(integrity_gates.items())),
        "finite_teacher_KL_gate_results": tuple(sorted(pairwise_gates.items())),
        "joint_approximate_top1_gate_results": tuple(sorted(top1_gates.items())),
        "outcome_matrix": {
            "outcome": classification,
            "joint_cleared_both_finite_teacher_KL_controls": pairwise_passed,
            "joint_cleared_approximate_top1_safety": all(top1_gates.values()),
            "finite_hypothesis_supported_same_a": passed,
            "fresh_confirmation_authorized_next": passed,
            "selection_or_serving_authorized": False,
        },
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "all_parent_outcomes_previously_inspected": True,
            "native_teacher_logits_used_for_every_finite_prompt": True,
            "held_native_H4_state_used_to_form_joint_features": True,
            "outer_held_family_excluded_from_its_refit_codec_scalar_and_joint_fit": True,
            "V7_gate_result_used_to_open_this_finite_test": True,
            "fresh_family_disjoint_confirmation_panel_opened": False,
            "finite_teacher_KL_is_displacement_authority_for_this_rung": True,
            "NLL_and_geometry_are_descriptive_not_primary": True,
            "candidate_serving_authorized": False,
            "model_mutation_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
        },
        "safety": _safety_metadata(),
    }
    return _publish(report, output=destination)


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(
            report, domain=_REPORT_DOMAIN
        )
        stage = frozen._stage_json(report, output)
        reservation.publish((stage,))
        return {
            **report,
            "artifact": {
                **dict(report["artifact"]),  # type: ignore[arg-type]
                "file_sha256": token_v1._file_sha256(output),
                "file_bytes": output.stat().st_size,
            },
        }
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """Return the deliberately no-knob V8 command-line interface."""

    return argparse.ArgumentParser(
        description=(
            "Run the pinned held-only V5/V6/V7 K64 three-arm finite test."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_finite_diagnostic()
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
