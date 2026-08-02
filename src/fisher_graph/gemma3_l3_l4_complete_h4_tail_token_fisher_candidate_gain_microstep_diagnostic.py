"""Symmetric finite K64 gain microstep diagnostic on the pinned A16 panel.

This v5 rung asks the narrow question left open by the v4 finite alpha grid:
does the family-equal mean-teacher-KL direction have a locally useful sign at
``epsilon = 1 / 64``?  It recollects and authenticates the exact v4 VJP/refit
bank, reuses (without reexecution) v4's unit tune observations, executes one
positive and one negative microstep on tune prompts, and freezes only the
positive-or-unit decision for held-family evaluation.  The negative step is a
diagnostic and can never be selected or satisfy a primary gate.

The experiment remains same-A and truth leaking.  It does not authorize
serving and does not establish compression, speed, or deployment readiness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic as v3diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic as v4diag
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl
from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from .complete_h4_tail_candidate_gain_microstep import (
    SYMMETRIC_GAIN_MICROSTEP_EPSILON,
    CandidateConditionedK64SymmetricMicrostepExample,
    CandidateConditionedK64SymmetricMicrostepSelection,
    select_candidate_conditioned_k64_symmetric_microstep,
    symmetric_microstep_gains,
)
from .complete_h4_tail_candidate_gain_refit_v4 import (
    CANDIDATE_GAIN_RANK,
    CandidateConditionedK64MeanKLRefit,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    complete_h4_tail_gate_scores,
    fit_complete_h4_tail_held_family,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import _load_committed_basis
from .gemma3_l3_l4_complete_h4_projection import CompleteH4ProjectionFitSequence
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
    "DEFAULT_V4_REPORT",
    "V4_REPORT_FILE_SHA256",
    "V4_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v4diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v4diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v4diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V4_REPORT = v4diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-gain-microstep-lofo-a-fit16-dev-v5.json"
)

V4_REPORT_FILE_SHA256 = (
    "5cad2c81694d9a0122ffe60df1c2bc1222395ddccc661b3e0751e2d78904ed50"
)
V4_REPORT_SHA256 = (
    "2ace5239314d5497e1c50ef17e3820ab041e99d5c19fe9d8443b0d9505f248c2"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_gain_symmetric_microstep_lofo.v5"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-gain-microstep:v5\0"
_PROVIDER_DOMAIN = b"fisher-graph:complete-h4-k64-gain-microstep-provider:v5\0"
_TUNE_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-gain-microstep-tune:v5\0"
_TUNE_RECEIPT_SET_DOMAIN = b"fisher-graph:complete-h4-k64-gain-microstep-tune-set:v5\0"
_OBSERVATION_DOMAIN = b"fisher-graph:complete-h4-k64-gain-microstep-observation:v5\0"
_OBSERVATION_SET_DOMAIN = b"fisher-graph:complete-h4-k64-gain-microstep-observation-set:v5\0"
_V4_REFIT_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-gain-v4-refit-binding:v5\0"
_V4_UNIT_TUNE_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-gain-v4-unit-tune-binding:v5\0"
_V4_UNIT_FINAL_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-gain-v4-unit-final-binding:v5\0"

_LEDGERS = v4diag._LEDGERS
_SELECTED_ARM = "selected_symmetric_microstep_k64"
_EXPECTED_FIT_SUPPORT_TOKENS = 398
_EXPECTED_TUNE_SUPPORT_TOKENS = 405
_EXPECTED_PARENT_FORWARD_COUNT = 48
_EXPECTED_PARENT_BACKWARD_COUNT = 109
_EXPECTED_GRADIENT_NATIVE_FORWARDS = 8
_EXPECTED_GRADIENT_CANDIDATE_FORWARDS = 56
_EXPECTED_GRADIENT_BACKWARDS = 385
_EXPECTED_TUNE_NATIVE_FORWARDS = 8
_EXPECTED_TUNE_CANDIDATE_FORWARDS = 112
_EXPECTED_FINAL_NATIVE_FORWARDS = 16
_EXPECTED_FINAL_CANDIDATE_FORWARDS = 16
_EXPECTED_TOTAL_FORWARDS = 264
_EXPECTED_TOTAL_BACKWARDS = 494

_PromptRoles = v4diag._PromptRoles
_checkerboard_prompt_roles = v4diag._checkerboard_prompt_roles
_fresh_native_teacher = v4diag._fresh_native_teacher
_endpoint_indices = v4diag._endpoint_indices
_ordered_k64 = v4diag._ordered_k64
_authenticate_parent_recollection = v4diag._authenticate_parent_recollection
_authenticate_static_unit_k64_replay = v4diag._authenticate_static_unit_k64_replay
_endpoint_prompt_receipts = v4diag._endpoint_prompt_receipts


def _load_v4_report(path: Path | str) -> dict[str, object]:
    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V4_REPORT_FILE_SHA256,
        expected_report_sha256=V4_REPORT_SHA256,
        label="candidate mean-KL gain-refit v4 control",
    )
    if (
        report.get("schema") != v4diag._SCHEMA
        or report.get("classification")
        != "tested_mean_KL_OPG_direction_not_supported_same_a"
        or report.get("passed") is not False
    ):
        raise RuntimeError("candidate mean-KL gain-refit v4 control differs")
    return report


def _canonical(value: object) -> object:
    return v3diag._canonical(value)


def _selection_step(selection: CandidateConditionedK64SymmetricMicrostepSelection) -> float:
    return float(getattr(selection, "selected_step"))


def _selection_arm(selection: CandidateConditionedK64SymmetricMicrostepSelection) -> str:
    arm = str(getattr(selection, "selected_arm"))
    if arm not in {"unit", "plus_epsilon"}:
        raise ValueError("v5 selected microstep arm differs")
    return arm


def _selection_gains(selection: CandidateConditionedK64SymmetricMicrostepSelection) -> Tensor:
    value = getattr(selection, "selected_gains_tensor")
    gains = value() if callable(value) else value
    if not isinstance(gains, Tensor):
        raise TypeError("v5 selected gains must be a tensor")
    gains = gains.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if gains.shape != (CANDIDATE_GAIN_RANK,) or not bool(torch.isfinite(gains).all()):
        raise ValueError("v5 selected gain geometry differs")
    return gains


def _microstep_gains(refit: CandidateConditionedK64MeanKLRefit, sign: int) -> Tensor:
    if sign not in {-1, 1}:
        raise ValueError("v5 microstep sign must be plus or minus one")
    gains = symmetric_microstep_gains(refit, sign=sign)
    gains = gains.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if (
        gains.shape != (CANDIDATE_GAIN_RANK,)
        or not bool(torch.isfinite(gains).all())
        or bool((gains < 0.0).any())
        or bool((gains > 1.5).any())
    ):
        raise ValueError("v5 symmetric microstep gains differ")
    return gains


class _AuthenticatedCandidateGainMicrostepProvider(Gemma3L3L4CorrectionProvider):
    """Single-use v5 correction bound to one signed finite execution."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "stage",
        "candidate_variant",
        "sign",
        "epsilon_hex",
        "step_hex",
        "fold_artifact_sha256",
        "ordered_directions_sha256",
        "gains_sha256",
        "refit_artifact_sha256",
        "selection_artifact_sha256",
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
        stage: str,
        candidate_variant: str,
        sign: int,
        fold_artifact_sha256: str,
        ordered_directions_sha256: str,
        gains: Tensor,
        refit_artifact_sha256: str,
        selection_artifact_sha256: str | None,
        example_id: str,
        family_id: str,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        if stage not in {"tune", "final"}:
            raise ValueError("v5 candidate provider stage differs")
        if candidate_variant not in {
            "plus_epsilon",
            "minus_epsilon_diagnostic",
            "selected_unit",
            "selected_plus_epsilon",
        }:
            raise ValueError("v5 candidate provider variant differs")
        if type(sign) is not int:
            raise ValueError("v5 candidate provider sign must be an exact integer")
        if stage == "tune":
            if (
                selection_artifact_sha256 is not None
                or sign not in {-1, 1}
                or (candidate_variant == "plus_epsilon" and sign != 1)
                or (candidate_variant == "minus_epsilon_diagnostic" and sign != -1)
                or candidate_variant.startswith("selected_")
            ):
                raise ValueError("v5 tune provider semantics differ")
        elif (
            selection_artifact_sha256 is None
            or sign not in {0, 1}
            or candidate_variant not in {"selected_unit", "selected_plus_epsilon"}
            or (candidate_variant == "selected_unit" and sign != 0)
            or (candidate_variant == "selected_plus_epsilon" and sign != 1)
        ):
            raise ValueError("v5 final provider semantics differ")
        gain_values = gains.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if (
            gain_values.shape != (CANDIDATE_GAIN_RANK,)
            or not bool(torch.isfinite(gain_values).all())
            or bool((gain_values < 0.0).any())
            or bool((gain_values > 1.5).any())
        ):
            raise ValueError("v5 candidate provider gains differ")
        if candidate_variant == "selected_unit" and not torch.equal(
            gain_values, torch.ones_like(gain_values)
        ):
            raise ValueError("v5 selected-unit provider requires all-one gains")
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
            raise ValueError("v5 candidate provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        delta = correction.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
        if not bool(torch.isfinite(delta).all()) or bool((delta[~support] != 0).any()):
            raise ValueError("v5 candidate provider correction escapes support")
        self.site = token_v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.stage = stage
        self.candidate_variant = candidate_variant
        self.sign = sign
        self.epsilon_hex = SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex()
        self.step_hex = (sign * SYMMETRIC_GAIN_MICROSTEP_EPSILON).hex()
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="v5 candidate fold"
        )
        self.ordered_directions_sha256 = _require_sha256(
            ordered_directions_sha256, label="v5 candidate directions"
        )
        self.gains_sha256 = _runtime_tensor_sha256(gain_values)
        self.refit_artifact_sha256 = _require_sha256(
            refit_artifact_sha256, label="v5 candidate refit"
        )
        self.selection_artifact_sha256 = (
            None
            if selection_artifact_sha256 is None
            else _require_sha256(selection_artifact_sha256, label="v5 selection")
        )
        self.example_id = token_v1._identifier(example_id, label="v5 example_id")
        self.family_id = token_v1._identifier(family_id, label="v5 family_id")
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="v5 model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="v5 bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="v5 prefix"
        )
        self.base_h4_sha256 = _runtime_tensor_sha256(base_h4)
        self._support = support
        self._correction = delta
        self.support_mask_sha256 = _runtime_tensor_sha256(support)
        self.correction_sha256 = _runtime_tensor_sha256(delta)
        self._used = False
        self.artifact_sha256 = self._computed_sha256()
        self.validate_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.complete_h4_k64_candidate_gain_microstep_provider.v5",
            "rank": CANDIDATE_GAIN_RANK,
            "site": self.site,
            "write_scope": self.write_scope,
            "stage": self.stage,
            "candidate_variant": self.candidate_variant,
            "sign": self.sign,
            "epsilon_hex": self.epsilon_hex,
            "step_hex": self.step_hex,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "ordered_directions_sha256": self.ordered_directions_sha256,
            "gains_sha256": self.gains_sha256,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "example_id": self.example_id,
            "family_id": self.family_id,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": (
                "P_D320_R_plus_signed_mean_KL_OPG_microstep_gain_scaled_K64_tail"
            ),
            "single_use": True,
            "truth_leaking_hypothesis_use_only": True,
            "serving_authorized": False,
        }

    def _computed_sha256(self) -> str:
        return token_v1._domain_sha256(self._payload(), domain=_PROVIDER_DOMAIN)

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
            or self._computed_sha256() != self.artifact_sha256
        ):
            raise RuntimeError("v5 candidate gain provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("v5 candidate gain provider cannot be reused")
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
            raise RuntimeError("v5 candidate provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _execute_candidate_teacher_kl_forward(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    gains: Tensor,
    stage: str,
    candidate_variant: str,
    sign: int,
    refit_artifact_sha256: str,
    selection_artifact_sha256: str | None,
    model_inputs: Mapping[str, Tensor],
    teacher_logits: Tensor,
    endpoint_indices: Tensor,
) -> tuple[Tensor, object, _AuthenticatedCandidateGainMicrostepProvider, Tensor]:
    directions, _tail, correction_rows, correction = v3diag._candidate_components(
        trace, basis=basis, fit=fit, gains=gains
    )
    provider = _AuthenticatedCandidateGainMicrostepProvider(
        stage=stage,
        candidate_variant=candidate_variant,
        sign=sign,
        fold_artifact_sha256=fit.artifact_sha256,
        ordered_directions_sha256=_runtime_tensor_sha256(directions),
        gains=gains,
        refit_artifact_sha256=refit_artifact_sha256,
        selection_artifact_sha256=selection_artifact_sha256,
        example_id=trace.example_id,
        family_id=trace.family_id,
        model_inputs_sha256=trace.model_inputs_sha256,
        bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
        prefix_artifact_sha256=trace.prefix.artifact_sha256,
        base_h4=trace.base_h4,
        support_mask=trace.prefix.complete_h4_causal_support_mask(),
        correction=correction,
    )
    execution = getattr(context, "bridge").execute(
        getattr(context, "adapter"), model_inputs, h4_head=provider
    )
    v3diag._validate_candidate_execution(
        trace=trace, provider=provider, execution=execution
    )
    token_kl = teacher_kl._selected_token_teacher_kl(
        teacher_logits, execution.logits, endpoint_indices
    )
    return token_kl, execution, provider, correction_rows


def _authenticate_live_v4_refits_and_gradients(
    *,
    v4_report: Mapping[str, object],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
    gradient_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require exact live reproduction of v4 refits and gradient receipts."""

    raw_refits = v4_report.get("candidate_gain_refits")
    raw_receipts = v4_report.get("candidate_gradient_receipts")
    if not isinstance(raw_refits, list) or not isinstance(raw_receipts, list):
        raise ValueError("pinned v4 refit or gradient evidence differs")
    expected_refits = {
        str(row["held_family_id"]): dict(row)
        for row in raw_refits
        if isinstance(row, Mapping)
    }
    if set(expected_refits) != set(refits) or len(expected_refits) != 8:
        raise RuntimeError("pinned v4 refit fold set differs")
    for family in sorted(refits):
        if _canonical(refits[family].metadata()) != _canonical(expected_refits[family]):
            raise RuntimeError("live v5 bank did not reproduce a pinned v4 refit")
    expected_by_cell = {
        (str(row["held_family_id"]), str(row["example_id"])): dict(row)
        for row in raw_receipts
        if isinstance(row, Mapping)
    }
    actual_by_cell = {
        (str(row["held_family_id"]), str(row["example_id"])): dict(row)
        for row in gradient_receipts
    }
    if (
        len(expected_by_cell) != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or len(actual_by_cell) != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or set(actual_by_cell) != set(expected_by_cell)
    ):
        raise RuntimeError("pinned v4 gradient receipt grid differs")
    for identity in sorted(actual_by_cell):
        if _canonical(actual_by_cell[identity]) != _canonical(expected_by_cell[identity]):
            raise RuntimeError("live v5 bank did not reproduce a pinned v4 gradient receipt")
    live_set = v4diag._receipt_set_sha256(
        gradient_receipts,
        expected_count=_EXPECTED_GRADIENT_CANDIDATE_FORWARDS,
        receipt_domain=v4diag._GRADIENT_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-gradient-set:v4\0",
    )
    if live_set != v4_report.get("candidate_gradient_receipt_set_sha256"):
        raise RuntimeError("live v5 gradient receipt-set did not reproduce v4")
    payload = {
        "v4_report_file_sha256": V4_REPORT_FILE_SHA256,
        "v4_report_sha256": V4_REPORT_SHA256,
        "v4_candidate_gradient_receipt_set_sha256": live_set,
        "live_refit_artifact_sha256s": tuple(
            refits[family].artifact_sha256 for family in sorted(refits)
        ),
        "live_refit_count": len(refits),
        "live_gradient_receipt_count": len(gradient_receipts),
        "refit_metadata_canonically_equal": True,
        "gradient_receipts_canonically_equal": True,
        "authenticated_before_microstep_execution": True,
    }
    return {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V4_REFIT_BINDING_DOMAIN
        ),
    }


def _authenticate_v4_unit_tune_baselines(
    *,
    v4_report: Mapping[str, object],
    roles: _PromptRoles,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[str, object]]:
    """Authenticate the full v4 tune set, then bind its 56 unit cells."""

    raw_receipts = v4_report.get("candidate_dual_tune_receipts")
    if not isinstance(raw_receipts, list):
        raise ValueError("pinned v4 tune receipt evidence differs")
    full_set = v4diag._receipt_set_sha256(
        raw_receipts,
        expected_count=v4diag._EXPECTED_TUNE_CANDIDATE_FORWARDS,
        receipt_domain=v4diag._TUNE_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-dual-tune-set:v4\0",
    )
    if full_set != v4_report.get("candidate_dual_tune_receipt_set_sha256"):
        raise RuntimeError("pinned v4 full tune receipt set drifted")
    tune_ids = set(roles.tune_example_ids)
    unit_rows = [
        dict(row)
        for row in raw_receipts
        if isinstance(row, Mapping) and row.get("candidate_variant") == "unit"
    ]
    by_cell: dict[tuple[str, str], dict[str, object]] = {}
    unit_gains_sha256 = _runtime_tensor_sha256(
        torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
    )
    for row in unit_rows:
        held_family = str(row.get("held_family_id"))
        example_id = str(row.get("example_id"))
        identity = (held_family, example_id)
        if identity in by_cell:
            raise ValueError("pinned v4 unit tune grid has a duplicate")
        if (
            held_family not in fits
            or example_id not in tune_ids
            or row.get("family_id") == held_family
            or row.get("fold_artifact_sha256") != fits[held_family].artifact_sha256
            or row.get("candidate_variant") != "unit"
            or row.get("step_kind") != "shared_zero"
            or row.get("step_hex") != 0.0.hex()
            or row.get("gains_sha256") != unit_gains_sha256
            or row.get("mean_refit_artifact_sha256") is not None
            or row.get("stage") != "tune"
            or row.get("held_family_used") is not False
            or row.get("unit_execution_shared_between_alpha_zero_and_beta_zero")
            is not True
            or not isinstance(row.get("mean_teacher_kl"), (int, float))
            or not math.isfinite(float(row["mean_teacher_kl"]))
        ):
            raise RuntimeError("pinned v4 unit tune cell semantics differ")
        for key in (
            "receipt_sha256",
            "dual_tune_example_artifact_sha256",
            "teacher_logits_sha256",
            "endpoint_supervised_grid_sha256",
            "token_teacher_kl_sha256",
            "ordered_directions_sha256",
        ):
            _require_sha256(str(row.get(key)), label=f"v4 unit tune {key}")
        by_cell[identity] = row
    if len(by_cell) != 56:
        raise RuntimeError("pinned v4 unit tune receipt count differs")
    expected_cells = {
        (held_family, example_id)
        for example_id in roles.tune_example_ids
        for held_family in sorted(fits)
        if held_family
        != next(
            str(row["family_id"])
            for row in unit_rows
            if row["example_id"] == example_id
        )
    }
    if set(by_cell) != expected_cells:
        raise RuntimeError("pinned v4 unit tune cell coverage differs")
    canonical_rows = tuple(by_cell[key] for key in sorted(by_cell))
    payload = {
        "v4_report_file_sha256": V4_REPORT_FILE_SHA256,
        "v4_report_sha256": V4_REPORT_SHA256,
        "v4_full_dual_tune_receipt_set_sha256": full_set,
        "canonical_unit_tune_receipts": canonical_rows,
        "unit_tune_receipt_count": len(canonical_rows),
        "unit_tune_cell_order": "held_family_id_then_example_id",
        "unit_tune_candidates_reexecuted_in_v5": False,
    }
    return by_cell, {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V4_UNIT_TUNE_BINDING_DOMAIN
        ),
    }


def _authenticate_v4_unit_final_baseline(
    *, v4_report: Mapping[str, object]
) -> tuple[dict[str, dict[str, object]], Mapping[str, object], Mapping[str, object], dict[str, object]]:
    """Authenticate v4's complete final set and extract its unit baseline."""

    raw_observations = v4_report.get("finite_observation_receipts")
    if not isinstance(raw_observations, list):
        raise ValueError("pinned v4 final observation evidence differs")
    full_set = v4diag._finite_observation_set_sha256(raw_observations)
    if full_set != v4_report.get("finite_observation_set_sha256"):
        raise RuntimeError("pinned v4 final observation set drifted")
    unit_rows = {
        str(row["example_id"]): dict(row)
        for row in raw_observations
        if isinstance(row, Mapping) and row.get("arm") == "unit_k64"
    }
    if len(unit_rows) != token_v1._EXPECTED_EXAMPLES:
        raise RuntimeError("pinned v4 final unit observation count differs")
    raw_behavior = v4_report.get("established_behavioral_fidelity_by_arm")
    raw_geometry = v4_report.get("executed_cast_once_geometry_by_arm")
    if not isinstance(raw_behavior, Mapping) or not isinstance(raw_geometry, Mapping):
        raise ValueError("pinned v4 unit behavior or geometry differs")
    unit_behavior = v3diag._mapping(
        raw_behavior.get("unit_k64"), label="pinned v4 unit behavior"
    )
    unit_geometry = v3diag._mapping(
        raw_geometry.get("unit_k64"), label="pinned v4 unit geometry"
    )
    if set(unit_behavior) != set(_LEDGERS):
        raise RuntimeError("pinned v4 unit behavior ledger set differs")
    v3diag._mapping(unit_geometry.get("gates"), label="pinned v4 unit geometry gates")
    canonical_rows = tuple(unit_rows[key] for key in sorted(unit_rows))
    payload = {
        "v4_report_file_sha256": V4_REPORT_FILE_SHA256,
        "v4_report_sha256": V4_REPORT_SHA256,
        "v4_full_finite_observation_set_sha256": full_set,
        "canonical_unit_final_observations": canonical_rows,
        "unit_behavior": unit_behavior,
        "unit_geometry": unit_geometry,
        "unit_final_observation_count": len(canonical_rows),
        "separate_pinned_v4_unit_final_baselines_reexecuted_in_v5": False,
    }
    return unit_rows, unit_behavior, unit_geometry, {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V4_UNIT_FINAL_BINDING_DOMAIN
        ),
    }


def _collect_symmetric_microstep_tune_selections(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
    roles: _PromptRoles,
    unit_baselines: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[
    dict[str, CandidateConditionedK64SymmetricMicrostepSelection],
    tuple[dict[str, object], ...],
    dict[str, int],
]:
    """Execute +epsilon then -epsilon in each tune cell; never rerun unit."""

    by_id = {trace.example_id: trace for trace in traces}
    examples_by_fold: dict[
        str, list[CandidateConditionedK64SymmetricMicrostepExample]
    ] = {family: [] for family in sorted(fits)}
    pending_receipts: list[tuple[str, dict[str, object]]] = []
    native_forwards = 0
    candidate_forwards = 0
    for example_id in roles.tune_example_ids:
        trace = by_id[example_id]
        model_inputs, indices, targets, teacher_logits = _fresh_native_teacher(
            context=context, trace=trace
        )
        native_forwards += 1
        endpoint_indices, _endpoint_targets, endpoint_grid = _endpoint_indices(
            trace, indices, targets
        )
        teacher_sha256 = _runtime_tensor_sha256(teacher_logits)
        grid_sha256 = _runtime_tensor_sha256(endpoint_grid)
        for held_family in sorted(fits):
            if held_family == trace.family_id:
                continue
            fit = fits[held_family]
            refit = refits[held_family]
            if refit.held_family_id != held_family:
                raise RuntimeError("v5 tune refit fold binding differs")
            unit = dict(unit_baselines[(held_family, trace.example_id)])
            directions_sha256 = _runtime_tensor_sha256(_ordered_k64(fit))
            if (
                unit.get("example_id") != trace.example_id
                or unit.get("family_id") != trace.family_id
                or unit.get("held_family_id") != held_family
                or unit.get("fold_artifact_sha256") != fit.artifact_sha256
                or unit.get("ordered_directions_sha256") != directions_sha256
                or unit.get("teacher_logits_sha256") != teacher_sha256
                or unit.get("endpoint_supervised_grid_sha256") != grid_sha256
                or unit.get("endpoint_supervised_token_count")
                != int(endpoint_grid.shape[0])
            ):
                raise RuntimeError("fresh v5 tune cell did not rebind pinned v4 unit")
            values: dict[int, Tensor] = {}
            cell_receipts: list[dict[str, object]] = []
            for sign, variant in (
                (1, "plus_epsilon"),
                (-1, "minus_epsilon_diagnostic"),
            ):
                gains = _microstep_gains(refit, sign)
                token_kl, execution, provider, _correction_rows = (
                    _execute_candidate_teacher_kl_forward(
                        context=context,
                        trace=trace,
                        basis=basis,
                        fit=fit,
                        gains=gains,
                        stage="tune",
                        candidate_variant=variant,
                        sign=sign,
                        refit_artifact_sha256=refit.artifact_sha256,
                        selection_artifact_sha256=None,
                        model_inputs=model_inputs,
                        teacher_logits=teacher_logits,
                        endpoint_indices=endpoint_indices,
                    )
                )
                values[sign] = token_kl
                refit_is_structural_no_op = bool(refit.mean_no_op) or torch.equal(
                    refit.mean_proposed_gains_tensor(),
                    torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64),
                )
                if refit_is_structural_no_op and (
                    _runtime_tensor_sha256(token_kl)
                    != unit["token_teacher_kl_sha256"]
                    or float(token_kl.mean()) != float(unit["mean_teacher_kl"])
                ):
                    raise RuntimeError(
                        "structural no-op signed execution did not replay v4 unit"
                    )
                cell_receipts.append(
                    {
                        "example_id": trace.example_id,
                        "family_id": trace.family_id,
                        "held_family_id": held_family,
                        "fold_artifact_sha256": fit.artifact_sha256,
                        "refit_artifact_sha256": refit.artifact_sha256,
                        "selection_artifact_sha256": None,
                        "pinned_v4_unit_receipt_sha256": unit["receipt_sha256"],
                        "pinned_v4_tune_example_artifact_sha256": unit[
                            "dual_tune_example_artifact_sha256"
                        ],
                        "pinned_v4_unit_token_teacher_kl_sha256": unit[
                            "token_teacher_kl_sha256"
                        ],
                        "pinned_v4_unit_mean_teacher_kl": float(
                            unit["mean_teacher_kl"]
                        ),
                        "ordered_directions_sha256": directions_sha256,
                        "candidate_variant": variant,
                        "sign": sign,
                        "epsilon_hex": SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
                        "step_hex": (
                            sign * SYMMETRIC_GAIN_MICROSTEP_EPSILON
                        ).hex(),
                        "gains_sha256": provider.gains_sha256,
                        "provider_artifact_sha256": provider.artifact_sha256,
                        "execution_artifact_sha256": execution.artifact_sha256,
                        "model_inputs_sha256": provider.model_inputs_sha256,
                        "bridge_binding_sha256": provider.bridge_binding_sha256,
                        "prefix_artifact_sha256": provider.prefix_artifact_sha256,
                        "base_h4_sha256": provider.base_h4_sha256,
                        "support_mask_sha256": provider.support_mask_sha256,
                        "correction_sha256": provider.correction_sha256,
                        "teacher_logits_sha256": teacher_sha256,
                        "endpoint_supervised_grid_sha256": grid_sha256,
                        "endpoint_supervised_token_count": int(
                            endpoint_grid.shape[0]
                        ),
                        "token_teacher_kl_sha256": _runtime_tensor_sha256(
                            token_kl
                        ),
                        "mean_teacher_kl": float(token_kl.mean()),
                        "stage": "tune",
                        "held_family_used": False,
                        "negative_step_is_diagnostic_sign_control": sign == -1,
                        "refit_was_structural_no_op": refit_is_structural_no_op,
                        "structural_no_op_replayed_pinned_v4_unit_exactly": (
                            True if refit_is_structural_no_op else None
                        ),
                        "raw_tensors_serialized": False,
                    }
                )
                candidate_forwards += 1
                del execution, provider, gains
            if set(values) != {-1, 1} or len(cell_receipts) != 2:
                raise RuntimeError("v5 symmetric tune cell candidate grid differs")
            tune_example = CandidateConditionedK64SymmetricMicrostepExample(
                example_id=trace.example_id,
                family_id=trace.family_id,
                held_family_id=held_family,
                v4_refit_artifact_sha256=refit.artifact_sha256,
                pinned_v4_tune_example_artifact_sha256=str(
                    unit["dual_tune_example_artifact_sha256"]
                ),
                pinned_v4_unit_receipt_sha256=str(unit["receipt_sha256"]),
                pinned_v4_unit_mean_teacher_kl=float(unit["mean_teacher_kl"]),
                pinned_v4_unit_token_teacher_kl_sha256=str(
                    unit["token_teacher_kl_sha256"]
                ),
                structural_no_op_replayed_pinned_v4_unit_exactly=(
                    True if refit_is_structural_no_op else None
                ),
                plus_token_teacher_kl=values[1],
                minus_token_teacher_kl=values[-1],
            )
            examples_by_fold[held_family].append(tune_example)
            for row in cell_receipts:
                row["microstep_tune_example_artifact_sha256"] = (
                    tune_example.artifact_sha256
                )
                pending_receipts.append((held_family, row))
            del values, cell_receipts
        del model_inputs, indices, targets, teacher_logits
    if (
        native_forwards != _EXPECTED_TUNE_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_TUNE_CANDIDATE_FORWARDS
        or len(pending_receipts) != _EXPECTED_TUNE_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("v5 symmetric microstep tune accounting differs")
    selections = {
        held_family: select_candidate_conditioned_k64_symmetric_microstep(
            refits[held_family], examples_by_fold[held_family]
        )
        for held_family in sorted(fits)
    }
    receipts: list[dict[str, object]] = []
    for held_family, pending in pending_receipts:
        bound = {
            **pending,
            "selection_artifact_sha256": selections[
                held_family
            ].artifact_sha256,
            "selection_frozen_after_complete_seven_family_tune_set": True,
            "unit_candidate_reexecuted": False,
        }
        bound["receipt_sha256"] = token_v1._domain_sha256(
            bound, domain=_TUNE_RECEIPT_DOMAIN
        )
        receipts.append(bound)
    return selections, tuple(receipts), {
        "tune_native_forward_count": native_forwards,
        "tune_candidate_forward_count": candidate_forwards,
        "tune_prompt_fold_count": 56,
        "tune_prompt_fold_candidate_count": len(receipts),
        "tune_candidate_count_per_prompt_fold": 2,
        "positive_microstep_execution_count_per_prompt_fold": 1,
        "negative_microstep_execution_count_per_prompt_fold": 1,
        "unit_execution_count_per_prompt_fold": 0,
        "pinned_v4_unit_baseline_reuse_count": 56,
    }


def _authenticate_selected_unit_stable_replay(
    observation: Mapping[str, object], unit: Mapping[str, object]
) -> bool:
    stable_keys = (
        "gains_sha256",
        "ordered_directions_sha256",
        "teacher_logits_sha256",
        "endpoint_supervised_grid_sha256",
        "endpoint_supervised_token_count",
        "token_teacher_kl_sha256",
        "complete_h4_support_mean_teacher_kl",
        "token_score_matrix_sha256",
        "native_mean_nll",
        "d320_mean_nll",
        "candidate_mean_nll",
        "ordinary_candidate_mean_nll",
        "endpoint_baseline_mse",
        "endpoint_prediction_mse",
        "candidate_h4_bitwise_native",
        "candidate_logits_bitwise_native",
        "full_tail_reconstruction_max_abs_error",
        "exact_residual_provider_used",
        "executed_correction_rows_sha256",
    )
    mismatches = [
        key for key in stable_keys if observation.get(key) != unit.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "selected-unit v5 execution did not replay pinned v4 unit: "
            + ", ".join(mismatches)
        )
    return True


def _final_selected_observations(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
    selections: Mapping[str, CandidateConditionedK64SymmetricMicrostepSelection],
    v4_unit_observations: Mapping[str, Mapping[str, object]],
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, int],
]:
    """Run one frozen unit-or-positive arm on each held-family prompt."""

    manifests = {
        ledger: {
            trace.example_id: trace.family_id
            for trace in traces
            if trace.selected_by_ledger[ledger].numel() > 0
        }
        for ledger in _LEDGERS
    }
    fidelity = {
        ledger: SourceAuthoritativeShadowFidelityAccumulator(
            manifests[ledger], gates=ESTABLISHED_SHADOW_FIDELITY_GATES
        )
        for ledger in _LEDGERS
    }
    geometry_traces: list[object] = []
    executed_rows: dict[str, Tensor] = {}
    for trace in traces:
        geometry_traces.append(
            SimpleNamespace(
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
        )
    observations: list[dict[str, object]] = []
    native_forwards = 0
    candidate_forwards = 0
    for trace in sorted(traces, key=lambda value: value.example_id):
        fit = fits[trace.family_id]
        refit = refits[trace.family_id]
        selection = selections[trace.family_id]
        selection_metadata = selection.metadata()
        if (
            selection.held_family_id != trace.family_id
            or selection.refit_artifact_sha256 != refit.artifact_sha256
            or selection_metadata.get("held_family_id") != trace.family_id
        ):
            raise RuntimeError("v5 final selection fold binding differs")
        selected_arm = _selection_arm(selection)
        selected_step = _selection_step(selection)
        if (
            (selected_arm == "unit" and selected_step != 0.0)
            or (
                selected_arm == "plus_epsilon"
                and selected_step != SYMMETRIC_GAIN_MICROSTEP_EPSILON
            )
        ):
            raise RuntimeError("v5 frozen final selection semantics differ")
        sign = 0 if selected_arm == "unit" else 1
        variant = "selected_unit" if sign == 0 else "selected_plus_epsilon"
        gains = _selection_gains(selection)
        expected_selected_gains_sha256 = _runtime_tensor_sha256(gains)
        model_inputs, indices, targets, teacher_logits = _fresh_native_teacher(
            context=context, trace=trace
        )
        native_forwards += 1
        endpoint_indices, endpoint_targets, endpoint_grid = _endpoint_indices(
            trace, indices, targets
        )
        source_selected = frozen._select_sequence_rows(teacher_logits, indices)
        full_scores = complete_h4_tail_gate_scores(
            trace.endpoint, fit.ordered_basis_rows()
        )
        scores = full_scores[:, :CANDIDATE_GAIN_RANK]
        unit = dict(v4_unit_observations[trace.example_id])
        if (
            unit.get("family_id") != trace.family_id
            or unit.get("fold_artifact_sha256") != fit.artifact_sha256
            or unit.get("ordered_directions_sha256")
            != _runtime_tensor_sha256(_ordered_k64(fit))
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
            raise RuntimeError("fresh v5 final prompt did not rebind pinned v4 unit")
        token_kl, execution, provider, _correction_rows = (
            _execute_candidate_teacher_kl_forward(
                context=context,
                trace=trace,
                basis=basis,
                fit=fit,
                gains=gains,
                stage="final",
                candidate_variant=variant,
                sign=sign,
                refit_artifact_sha256=refit.artifact_sha256,
                selection_artifact_sha256=selection.artifact_sha256,
                model_inputs=model_inputs,
                teacher_logits=teacher_logits,
                endpoint_indices=endpoint_indices,
            )
        )
        if provider.gains_sha256 != expected_selected_gains_sha256:
            raise RuntimeError("v5 final provider did not bind exact selection gains")
        candidate_forwards += 1
        candidate_nll = token_v1._selected_token_nll(
            execution.logits, indices, targets
        )
        candidate_endpoint_nll = token_v1._selected_token_nll(
            execution.logits, endpoint_indices, endpoint_targets
        )
        candidate_selected = frozen._select_sequence_rows(execution.logits, indices)
        for ledger, selected in trace.selected_by_ledger.items():
            if selected.numel() == 0:
                continue
            fidelity[ledger].add(
                ShadowFidelityExample(
                    example_id=trace.example_id,
                    family_id=trace.family_id,
                    source_logits=source_selected.index_select(
                        0, selected.to(source_selected.device)
                    ),
                    candidate_logits=candidate_selected.index_select(
                        0, selected.to(candidate_selected.device)
                    ),
                    targets=targets.index_select(0, selected.to(targets.device)),
                )
            )
        actual_rows = (
            execution.candidate_h4.detach().to(device="cpu", dtype=torch.float64)[0]
            .index_select(0, trace.support_indices)
            - trace.base_h4.detach().to(device="cpu", dtype=torch.float64)[0]
            .index_select(0, trace.support_indices)
        ).contiguous()
        executed_rows[trace.example_id] = actual_rows
        prediction = (scores * gains.unsqueeze(0)).sum(dim=1).contiguous()
        target = trace.endpoint.compensation_target
        observation: dict[str, object] = {
            "example_id": trace.example_id,
            "family_id": trace.family_id,
            "arm": _SELECTED_ARM,
            "candidate_variant": variant,
            "selected_arm": selected_arm,
            "selected_sign": sign,
            "epsilon_hex": SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
            "selected_step_hex": selected_step.hex(),
            "rank": CANDIDATE_GAIN_RANK,
            "fold_artifact_sha256": fit.artifact_sha256,
            "refit_artifact_sha256": refit.artifact_sha256,
            "selection_artifact_sha256": selection.artifact_sha256,
            "pinned_v4_unit_observation_sha256": unit["observation_sha256"],
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
            "teacher_logits_sha256": _runtime_tensor_sha256(teacher_logits),
            "endpoint_supervised_grid_sha256": _runtime_tensor_sha256(endpoint_grid),
            "endpoint_supervised_token_count": int(endpoint_grid.shape[0]),
            "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
            "complete_h4_support_mean_teacher_kl": float(token_kl.mean()),
            "token_score_matrix_sha256": _runtime_tensor_sha256(full_scores),
            "native_mean_nll": float(trace.native_token_nll.mean()),
            "d320_mean_nll": float(trace.d320_token_nll.mean()),
            "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
            "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
            "endpoint_baseline_mse": float(target.square().mean()),
            "endpoint_prediction_mse": float((prediction - target).square().mean()),
            "candidate_h4_bitwise_native": token_v1._bitwise_equal(
                execution.candidate_h4.detach().to(device="cpu"), trace.native_h4
            ),
            "candidate_logits_bitwise_native": (
                _runtime_tensor_sha256(execution.logits) == trace.native_logits_sha256
            ),
            "full_tail_reconstruction_max_abs_error": None,
            "exact_residual_provider_used": False,
            "executed_correction_rows_sha256": _runtime_tensor_sha256(actual_rows),
            "held_family_used_for_fit_or_tune": False,
            "held_family_excluded_from_gain_fit_and_tune": True,
            "negative_epsilon_can_be_selected_or_independently_authorize_primary": False,
            "separate_pinned_v4_unit_baseline_reexecuted": False,
            "selected_candidate_executed": True,
        }
        if selected_arm == "unit":
            observation["selected_unit_stable_outputs_replay_v4_exactly"] = (
                _authenticate_selected_unit_stable_replay(observation, unit)
            )
        else:
            observation["selected_unit_stable_outputs_replay_v4_exactly"] = None
        observation["observation_sha256"] = token_v1._domain_sha256(
            observation, domain=_OBSERVATION_DOMAIN
        )
        observations.append(observation)
        del (
            model_inputs,
            indices,
            targets,
            teacher_logits,
            source_selected,
            full_scores,
            scores,
            gains,
            token_kl,
            execution,
            provider,
            candidate_nll,
            candidate_endpoint_nll,
            candidate_selected,
            actual_rows,
            prediction,
        )
    if (
        native_forwards != _EXPECTED_FINAL_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_FINAL_CANDIDATE_FORWARDS
        or len(observations) != _EXPECTED_FINAL_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("v5 selected final execution accounting differs")
    behavior = {ledger: fidelity[ledger].finalize() for ledger in _LEDGERS}
    geometry = ladder._geometry_with_examples(
        geometry_traces,
        executed_rows,
        candidate_semantics=(
            "actual_cast_once_d320_plus_frozen_unit_or_positive_one_over_64_"
            "mean_KL_OPG_microstep_tail_k64"
        ),
    )
    return observations, behavior, geometry, {
        "final_native_forward_count": native_forwards,
        "final_candidate_forward_count": candidate_forwards,
        "final_observation_count": len(observations),
        "final_arm_count": 1,
        "separate_pinned_v4_unit_final_baseline_reexecution_count": 0,
        "selected_candidate_execution_count": candidate_forwards,
    }


def _finite_observation_set_sha256(
    observations: Sequence[Mapping[str, object]],
) -> str:
    if len(observations) != token_v1._EXPECTED_EXAMPLES:
        raise ValueError("v5 candidate observation count differs")
    identities: set[str] = set()
    receipts: list[str] = []
    for raw in observations:
        row = dict(raw)
        receipt = row.pop("observation_sha256", None)
        example_id = token_v1._identifier(
            row.get("example_id"), label="v5 candidate observation example_id"
        )
        if (
            example_id in identities
            or row.get("arm") != _SELECTED_ARM
            or row.get("rank") != CANDIDATE_GAIN_RANK
            or row.get("selected_sign") not in {0, 1}
        ):
            raise ValueError("v5 candidate observation grid differs")
        identities.add(example_id)
        expected = token_v1._domain_sha256(row, domain=_OBSERVATION_DOMAIN)
        if receipt != expected:
            raise RuntimeError("v5 candidate observation receipt drifted")
        receipts.append(expected)
    return token_v1._domain_sha256(tuple(receipts), domain=_OBSERVATION_SET_DOMAIN)


def _receipt_set_sha256(
    receipts: Sequence[Mapping[str, object]],
) -> str:
    if len(receipts) != _EXPECTED_TUNE_CANDIDATE_FORWARDS:
        raise ValueError("v5 tune receipt-set count differs")
    values: list[str] = []
    identities: set[tuple[str, str, int]] = set()
    pair_signs: dict[tuple[str, str], set[int]] = defaultdict(set)
    for raw in receipts:
        row = dict(raw)
        receipt = row.pop("receipt_sha256", None)
        expected = token_v1._domain_sha256(row, domain=_TUNE_RECEIPT_DOMAIN)
        identity = (
            str(row.get("held_family_id")),
            str(row.get("example_id")),
            int(row.get("sign", 0)),
        )
        if receipt != expected:
            raise RuntimeError("v5 tune receipt drifted")
        if (
            identity in identities
            or identity[2] not in {-1, 1}
            or row.get("held_family_id") == row.get("family_id")
        ):
            raise ValueError("v5 tune receipt grid has a duplicate")
        identities.add(identity)
        pair_signs[identity[:2]].add(identity[2])
        values.append(expected)
    if len(pair_signs) != 56 or any(signs != {-1, 1} for signs in pair_signs.values()):
        raise ValueError("v5 tune receipt grid lacks exact symmetric sign coverage")
    return token_v1._domain_sha256(values, domain=_TUNE_RECEIPT_SET_DOMAIN)


def _selected_arm_summary(
    *,
    observations: Sequence[Mapping[str, object]],
    parent: Mapping[str, object],
) -> dict[str, object]:
    parent_rows = parent.get("finite_observation_receipts")
    if not isinstance(parent_rows, list):
        raise ValueError("parent K320 summary anchor differs")
    selected = [v3diag._v1_observation_view(row) for row in observations]
    k320 = [
        dict(row)
        for row in parent_rows
        if isinstance(row, Mapping) and row.get("rank") == 320
    ]
    summaries, _gates = token_v1._summarize_observations(
        selected + k320, ranks=(64, 320)
    )
    result = dict(next(row for row in summaries if row["tail_rank"] == 64))
    result["arm"] = _SELECTED_ARM
    result["k320_reexecuted"] = False
    result["k320_parent_anchor_only"] = True
    return result


def _teacher_kl_comparison(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in observations:
        by_family[str(row["family_id"])].append(row)
    family_rows: list[dict[str, object]] = []
    for family in sorted(by_family):
        rows = by_family[family]
        if len(rows) != 2:
            raise ValueError("v5 held teacher-KL family observation shape differs")
        unit = math.fsum(
            float(row["pinned_v4_unit_mean_teacher_kl"]) for row in rows
        ) / 2.0
        selected = math.fsum(
            float(row["complete_h4_support_mean_teacher_kl"]) for row in rows
        ) / 2.0
        family_rows.append(
            {
                "family_id": family,
                "pinned_v4_unit_k64_mean_teacher_kl": unit,
                "selected_symmetric_microstep_k64_mean_teacher_kl": selected,
                "absolute_delta_selected_minus_unit": selected - unit,
                "relative_improvement": (
                    (unit - selected)
                    / max(unit, torch.finfo(torch.float64).tiny)
                ),
                "improved": selected < unit,
                "within_five_percent_plus_1e_minus_8": (
                    selected <= 1.05 * unit + 1.0e-8
                ),
            }
        )
    if len(family_rows) != token_v1._EXPECTED_FAMILIES:
        raise ValueError("v5 held teacher-KL family count differs")
    unit_macro = math.fsum(
        float(row["pinned_v4_unit_k64_mean_teacher_kl"])
        for row in family_rows
    ) / len(family_rows)
    selected_macro = math.fsum(
        float(row["selected_symmetric_microstep_k64_mean_teacher_kl"])
        for row in family_rows
    ) / len(family_rows)
    return {
        "ledger": "complete_h4_support",
        "aggregation": "family_then_prompt_equal",
        "pinned_v4_unit_k64_family_macro_mean_teacher_kl": unit_macro,
        "selected_symmetric_microstep_k64_family_macro_mean_teacher_kl": (
            selected_macro
        ),
        "family_macro_relative_improvement": (
            (unit_macro - selected_macro)
            / max(unit_macro, torch.finfo(torch.float64).tiny)
        ),
        "held_family_improvement_count": sum(
            bool(row["improved"]) for row in family_rows
        ),
        "all_held_families_within_five_percent_plus_1e_minus_8": all(
            bool(row["within_five_percent_plus_1e_minus_8"])
            for row in family_rows
        ),
        "families": tuple(family_rows),
        "unit_values_are_authenticated_v4_not_reexecuted": True,
    }


def _top1_comparison(
    *,
    unit_behavior: Mapping[str, object],
    selected_behavior: Mapping[str, object],
) -> dict[str, object]:
    # Reuse v4's tested comparison math by placing the selected v5 behavior in
    # its primary-arm slot; the returned scalar semantics are arm-agnostic.
    result = v4diag._top1_comparison(
        {
            "unit_k64": unit_behavior,
            v4diag._PRIMARY_ARM: selected_behavior,
        },
        candidate_arm=v4diag._PRIMARY_ARM,
    )
    return {
        **result,
        "source_arm": "authenticated_v4_unit_k64",
        "candidate_arm": _SELECTED_ARM,
    }


def _geometry_comparison(
    *,
    unit_geometry: Mapping[str, object],
    selected_geometry: Mapping[str, object],
) -> dict[str, object]:
    unit_pooled = v3diag._mapping(
        unit_geometry.get("pooled"), label="v4 unit pooled geometry"
    )
    selected_pooled = v3diag._mapping(
        selected_geometry.get("pooled"), label="v5 selected pooled geometry"
    )
    strata: dict[str, object] = {}
    for stratum in ("full", "graph_core", "causal_tail"):
        unit = v3diag._mapping(unit_pooled.get(stratum), label=f"unit {stratum}")
        selected = v3diag._mapping(
            selected_pooled.get(stratum), label=f"selected {stratum}"
        )
        if bool(unit.get("applicable")) and bool(selected.get("applicable")):
            unit_rmse = float(unit["normalized_rmse"])
            selected_rmse = float(selected["normalized_rmse"])
            unit_cosine = float(unit["cosine"])
            selected_cosine = float(selected["cosine"])
            strata[stratum] = {
                "applicable": True,
                "unit_normalized_rmse": unit_rmse,
                "selected_normalized_rmse": selected_rmse,
                "normalized_rmse_delta_selected_minus_unit": (
                    selected_rmse - unit_rmse
                ),
                "unit_cosine": unit_cosine,
                "selected_cosine": selected_cosine,
                "cosine_delta_selected_minus_unit": selected_cosine - unit_cosine,
            }
        else:
            strata[stratum] = {
                "applicable": False,
                "status": "not_comparable_zero_rows",
            }
    return {
        "source_arm": "authenticated_v4_unit_k64",
        "candidate_arm": _SELECTED_ARM,
        "pooled_strata": strata,
        "unit_geometry_passed": bool(
            v3diag._mapping(
                unit_geometry.get("gates"), label="unit geometry gates"
            ).get("passed")
        ),
        "selected_geometry_passed": bool(
            v3diag._mapping(
                selected_geometry.get("gates"), label="selected geometry gates"
            ).get("passed")
        ),
        "geometry_reported_but_not_primary": True,
    }


def _gate_results(
    *,
    selected_behavior: Mapping[str, object],
    selected_geometry: Mapping[str, object],
    comparison: Mapping[str, object],
    top1: Mapping[str, object],
    selections: Mapping[str, CandidateConditionedK64SymmetricMicrostepSelection],
) -> tuple[dict[str, object], dict[str, object]]:
    ledger_pass = {
        ledger: bool(
            v3diag._mapping(
                v3diag._mapping(
                    selected_behavior[ledger], label=f"selected {ledger} behavior"
                ).get("gates"),
                label=f"selected {ledger} gates",
            ).get("passed")
        )
        for ledger in _LEDGERS
    }
    geometry_passed = bool(
        v3diag._mapping(
            selected_geometry.get("gates"), label="selected geometry gates"
        ).get("passed")
    )
    strict = {
        "behavior_ledger_pass": ledger_pass,
        "behavioral_only_strict_passed": all(ledger_pass.values()),
        "geometry_passed": geometry_passed,
        "full_behavior_plus_geometry_strict_passed": (
            all(ledger_pass.values()) and geometry_passed
        ),
    }
    positive_count = sum(
        _selection_step(selection) > 0.0 for selection in selections.values()
    )
    structural_no_op_count = sum(
        bool(selection.refit_no_op_or_zero_delta)
        for selection in selections.values()
    )
    gates: dict[str, bool] = {
        "family_macro_complete_h4_support_teacher_kl_improves_at_least_2pct": (
            float(comparison["family_macro_relative_improvement"]) >= 0.02
        ),
        "held_family_teacher_kl_improvement_count_at_least_6_of_8": (
            int(comparison["held_family_improvement_count"]) >= 6
        ),
        "worst_held_family_teacher_kl_regression_at_most_5pct_plus_1e_minus_8": bool(
            comparison["all_held_families_within_five_percent_plus_1e_minus_8"]
        ),
        "folds_selecting_positive_microstep_at_least_6_of_8": (
            positive_count >= 6
        ),
    }
    for ledger in ("ordinary", "complete_h4_support", "graph_core"):
        row = v3diag._mapping(top1[ledger], label=f"{ledger} top1")
        gates[f"{ledger}_aggregate_top1_at_least_point_90"] = bool(
            row["aggregate_at_least_point_90"]
        )
        gates[f"{ledger}_family_macro_top1_at_least_point_90"] = bool(
            row["family_macro_at_least_point_90"]
        )
        gates[f"{ledger}_aggregate_top1_no_material_regression_vs_unit"] = bool(
            row["aggregate_no_material_regression_vs_unit"]
        )
        gates[f"{ledger}_family_macro_top1_no_material_regression_vs_unit"] = bool(
            row["family_macro_no_material_regression_vs_unit"]
        )
    approximate = {
        "gates": tuple(sorted(gates.items())),
        "passed": all(gates.values()),
        "positive_microstep_fold_count": positive_count,
        "positive_microstep_required_count": 6,
        "structural_refit_no_op_or_zero_delta_fold_count_reported_separately": (
            structural_no_op_count
        ),
        "negative_microstep_used_only_for_central_slope_sign_guard": True,
        "negative_microstep_can_be_selected_or_independently_authorize_primary": False,
        "top1_threshold": 0.90,
        "top1_no_material_regression_tolerance": 0.01,
        "teacher_kl_required_relative_improvement": 0.02,
        "teacher_kl_worst_family_regression_cap": (
            "candidate_le_1.05_times_unit_plus_1e_minus_8"
        ),
        "causal_tail_reported_but_excluded_from_primary": True,
        "geometry_reported_but_excluded_from_primary": True,
    }
    return strict, approximate


def _outcome_matrix(
    *,
    approximate: Mapping[str, object],
    comparison: Mapping[str, object],
    top1: Mapping[str, object],
    selections: Mapping[str, CandidateConditionedK64SymmetricMicrostepSelection],
) -> dict[str, object]:
    positive_count = sum(
        _selection_step(selection) > 0.0 for selection in selections.values()
    )
    no_op_count = sum(
        bool(selection.refit_no_op_or_zero_delta)
        for selection in selections.values()
    )
    top1_primary_safe = all(
        bool(top1[ledger][key])
        for ledger in ("ordinary", "complete_h4_support", "graph_core")
        for key in (
            "aggregate_at_least_point_90",
            "family_macro_at_least_point_90",
            "aggregate_no_material_regression_vs_unit",
            "family_macro_no_material_regression_vs_unit",
        )
    )
    held_kl_transferred = (
        positive_count >= 6
        and int(comparison["held_family_improvement_count"]) >= 6
        and bool(
            comparison["all_held_families_within_five_percent_plus_1e_minus_8"]
        )
        and float(comparison["family_macro_relative_improvement"]) > 0.0
    )
    if no_op_count == token_v1._EXPECTED_FAMILIES:
        outcome = "one_over_64_symmetric_microstep_all_refits_structural_no_op_same_a"
    elif bool(approximate["passed"]):
        outcome = "symmetric_microstep_mean_KL_OPG_supported_same_a"
    elif positive_count < 6:
        outcome = "one_over_64_symmetric_microstep_fit_to_tune_transfer_failed_same_a"
    elif not held_kl_transferred:
        outcome = "symmetric_microstep_static_cross_family_transfer_blocker_same_a"
    elif not top1_primary_safe:
        outcome = "symmetric_microstep_behavioral_top1_fidelity_failed_same_a"
    elif float(comparison["family_macro_relative_improvement"]) < 0.02:
        outcome = "symmetric_microstep_transferred_but_below_useful_effect_same_a"
    else:
        outcome = "symmetric_microstep_primary_gate_partition_inconsistent"
    return {
        "outcome": outcome,
        "primary_supported": bool(approximate["passed"]),
        "positive_microstep_tune_selection_count": positive_count,
        "positive_microstep_tune_selection_required_count": 6,
        "structural_refit_no_op_or_zero_delta_fold_count": no_op_count,
        "held_family_KL_transfer_detected": held_kl_transferred,
        "behavioral_top1_retained": top1_primary_safe,
        "held_family_macro_relative_improvement": float(
            comparison["family_macro_relative_improvement"]
        ),
        "negative_microstep_is_diagnostic_sign_control": True,
        "negative_microstep_participates_in_central_slope_guard": True,
        "negative_microstep_can_select_or_rescue_primary": False,
        "interpretation_if_fewer_than_six_tune_folds_select_positive": (
            "one_over_64_finite_fit_to_tune_transfer_failure"
        ),
        "interpretation_if_at_least_six_tune_folds_select_but_held_fails": (
            "static_cross_family_transfer_blocker"
        ),
        "interpretation_if_KL_transfers_but_top1_fails": (
            "behavioral_top1_fidelity_failure"
        ),
        "interpretation_if_held_transfers_below_two_percent": (
            "real_but_below_predeclared_useful_effect"
        ),
    }


def _resource_accounting(
    *,
    traces: Sequence[object],
    roles: _PromptRoles,
    endpoint_resources: Mapping[str, int],
    gradient_resources: Mapping[str, int],
    tune_resources: Mapping[str, int],
    final_resources: Mapping[str, int],
) -> dict[str, object]:
    parent_forwards = (
        endpoint_resources["base_forward_count"]
        + endpoint_resources["native_forward_count"]
        + endpoint_resources["endpoint_token_vjp_forward_count"]
    )
    parent_backwards = endpoint_resources[
        "endpoint_token_vjp_backward_call_count"
    ]
    gradient_forwards = (
        gradient_resources["gradient_native_forward_count"]
        + gradient_resources["gradient_candidate_vjp_forward_count"]
    )
    tune_forwards = (
        tune_resources["tune_native_forward_count"]
        + tune_resources["tune_candidate_forward_count"]
    )
    final_forwards = (
        final_resources["final_native_forward_count"]
        + final_resources["final_candidate_forward_count"]
    )
    total_forwards = parent_forwards + gradient_forwards + tune_forwards + final_forwards
    total_backwards = (
        parent_backwards
        + gradient_resources["gradient_candidate_vjp_backward_call_count"]
    )
    if (
        parent_forwards != _EXPECTED_PARENT_FORWARD_COUNT
        or parent_backwards != _EXPECTED_PARENT_BACKWARD_COUNT
        or gradient_forwards
        != _EXPECTED_GRADIENT_NATIVE_FORWARDS
        + _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or tune_forwards
        != _EXPECTED_TUNE_NATIVE_FORWARDS + _EXPECTED_TUNE_CANDIDATE_FORWARDS
        or final_forwards
        != _EXPECTED_FINAL_NATIVE_FORWARDS + _EXPECTED_FINAL_CANDIDATE_FORWARDS
        or total_forwards != _EXPECTED_TOTAL_FORWARDS
        or total_backwards != _EXPECTED_TOTAL_BACKWARDS
        or tune_resources["unit_execution_count_per_prompt_fold"] != 0
        or final_resources[
            "separate_pinned_v4_unit_final_baseline_reexecution_count"
        ]
        != 0
    ):
        raise RuntimeError("v5 symmetric microstep resource accounting differs")
    by_id = {trace.example_id: trace for trace in traces}
    fit_rows = sum(
        int(by_id[example_id].endpoint.residual_rows.shape[0])
        for example_id in roles.fit_example_ids
    )
    tune_rows = sum(
        int(by_id[example_id].endpoint.residual_rows.shape[0])
        for example_id in roles.tune_example_ids
    )
    support_rows = sum(int(trace.endpoint.residual_rows.shape[0]) for trace in traces)
    gradient_row_executions = 7 * fit_rows
    tune_row_executions = 7 * 2 * tune_rows
    final_row_executions = support_rows
    candidate_support_row_executions = (
        gradient_row_executions + tune_row_executions + final_row_executions
    )
    candidate_execution_count = (
        _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        + _EXPECTED_TUNE_CANDIDATE_FORWARDS
        + _EXPECTED_FINAL_CANDIDATE_FORWARDS
    )
    return {
        **endpoint_resources,
        **gradient_resources,
        **tune_resources,
        **final_resources,
        "vjp_chunk_size": token_v1._VJP_CHUNK_SIZE,
        "parent_collection_model_forward_count": parent_forwards,
        "parent_collection_backward_call_count": parent_backwards,
        "gradient_stage_model_forward_count": gradient_forwards,
        "tune_stage_model_forward_count": tune_forwards,
        "final_stage_model_forward_count": final_forwards,
        "total_model_forward_count": total_forwards,
        "total_backward_call_count": total_backwards,
        "candidate_k64_execution_count": candidate_execution_count,
        "candidate_support_row_executions": candidate_support_row_executions,
        "gradient_candidate_support_row_executions": gradient_row_executions,
        "tune_candidate_support_row_executions": tune_row_executions,
        "final_candidate_support_row_executions": final_row_executions,
        "candidate_d320_supported_projection_logical_macs": (
            candidate_support_row_executions
            * 2
            * token_v1._D_RANK
            * token_v1._WIDTH
        ),
        "candidate_k64_tail_projection_logical_macs": (
            candidate_support_row_executions
            * 2
            * CANDIDATE_GAIN_RANK
            * token_v1._WIDTH
        ),
        "candidate_projection_mac_scope": (
            "cpu_analysis_correction_materialization_not_model_kernel_speed"
        ),
        "pinned_v4_unit_tune_baseline_reuse_count": 56,
        "pinned_v4_unit_tune_model_forward_count": 0,
        "pinned_v4_unit_final_baseline_model_forward_count": 0,
        "k320_model_forward_count": 0,
        "k320_reexecution_performed": False,
        "raw_candidate_vjp_banks_retained_in_report": False,
        "serving_learned_parameter_count": "not_applicable_no_serving_artifact",
        "serving_logical_macs_per_token": "not_applicable_no_serving_artifact",
    }


@dataclass(slots=True)
class _CandidatePhaseResults:
    recollection_receipt: str
    static_unit_replay_receipt: str
    roles: _PromptRoles
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit]
    gradient_receipts: tuple[dict[str, object], ...]
    gradient_resources: Mapping[str, int]
    v4_refit_binding: Mapping[str, object]
    unit_tune_baselines: Mapping[tuple[str, str], Mapping[str, object]]
    v4_unit_tune_binding: Mapping[str, object]
    v4_unit_observations: Mapping[str, Mapping[str, object]]
    v4_unit_behavior: Mapping[str, object]
    v4_unit_geometry: Mapping[str, object]
    v4_unit_final_binding: Mapping[str, object]
    selections: Mapping[str, CandidateConditionedK64SymmetricMicrostepSelection]
    tune_receipts: tuple[dict[str, object], ...]
    tune_resources: Mapping[str, int]
    observations: list[dict[str, object]]
    observation_set_sha256: str
    selected_behavior: Mapping[str, object]
    selected_geometry: Mapping[str, object]
    final_resources: Mapping[str, int]


def _execute_candidate_phases(
    *,
    context: object,
    parent: Mapping[str, object],
    v4_report: Mapping[str, object],
    traces: Sequence[object],
    endpoint_resources: Mapping[str, int],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> _CandidatePhaseResults:
    recollection_receipt = _authenticate_parent_recollection(
        parent=parent,
        traces=traces,
        endpoint_resources=endpoint_resources,
        fits=fits,
    )
    static_unit_replay_receipt = _authenticate_static_unit_k64_replay(
        parent=parent, traces=traces, basis=basis, fits=fits
    )
    roles = _checkerboard_prompt_roles(traces)
    refits, _replayed_v3, gradient_receipts, gradient_resources = (
        v4diag._collect_candidate_gradient_refits(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            roles=roles,
        )
    )
    v4_refit_binding = _authenticate_live_v4_refits_and_gradients(
        v4_report=v4_report,
        refits=refits,
        gradient_receipts=gradient_receipts,
    )
    unit_tune_baselines, v4_unit_tune_binding = (
        _authenticate_v4_unit_tune_baselines(
            v4_report=v4_report, roles=roles, fits=fits
        )
    )
    (
        v4_unit_observations,
        v4_unit_behavior,
        v4_unit_geometry,
        v4_unit_final_binding,
    ) = _authenticate_v4_unit_final_baseline(v4_report=v4_report)
    selections, tune_receipts, tune_resources = (
        _collect_symmetric_microstep_tune_selections(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            refits=refits,
            roles=roles,
            unit_baselines=unit_tune_baselines,
        )
    )
    observations, selected_behavior, selected_geometry, final_resources = (
        _final_selected_observations(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            refits=refits,
            selections=selections,
            v4_unit_observations=v4_unit_observations,
        )
    )
    observation_set_sha256 = _finite_observation_set_sha256(observations)
    return _CandidatePhaseResults(
        recollection_receipt=recollection_receipt,
        static_unit_replay_receipt=static_unit_replay_receipt,
        roles=roles,
        refits=refits,
        gradient_receipts=gradient_receipts,
        gradient_resources=gradient_resources,
        v4_refit_binding=v4_refit_binding,
        unit_tune_baselines=unit_tune_baselines,
        v4_unit_tune_binding=v4_unit_tune_binding,
        v4_unit_observations=v4_unit_observations,
        v4_unit_behavior=v4_unit_behavior,
        v4_unit_geometry=v4_unit_geometry,
        v4_unit_final_binding=v4_unit_final_binding,
        selections=selections,
        tune_receipts=tune_receipts,
        tune_resources=tune_resources,
        observations=observations,
        observation_set_sha256=observation_set_sha256,
        selected_behavior=selected_behavior,
        selected_geometry=selected_geometry,
        final_resources=final_resources,
    )


def _publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
    frozen._scalar_report(report)
    reservation = frozen._reserve_outputs((output,))
    stage: Path | None = None
    try:
        report["report_sha256"] = frozen._json_sha256(report, domain=_REPORT_DOMAIN)
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


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v4_report_path: Path | str = DEFAULT_V4_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned same-A v5 symmetric one-over-64 microstep rung."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite candidate gain microstep v5 report")
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v4_report = _load_v4_report(v4_report_path)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate gain v5 rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate gain v5 rank320 transfer",
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
            len(traces) != token_v1._EXPECTED_EXAMPLES
            or len(families) != token_v1._EXPECTED_FAMILIES
        ):
            raise RuntimeError("candidate gain v5 A16 panel shape differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        phases = _execute_candidate_phases(
            context=context,
            parent=parent,
            v4_report=v4_report,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    roles = phases.roles
    refits = phases.refits
    selections = phases.selections
    observations = phases.observations
    comparison = _teacher_kl_comparison(observations)
    top1 = _top1_comparison(
        unit_behavior=phases.v4_unit_behavior,
        selected_behavior=phases.selected_behavior,
    )
    geometry_comparison = _geometry_comparison(
        unit_geometry=phases.v4_unit_geometry,
        selected_geometry=phases.selected_geometry,
    )
    strict_results, approximate_results = _gate_results(
        selected_behavior=phases.selected_behavior,
        selected_geometry=phases.selected_geometry,
        comparison=comparison,
        top1=top1,
        selections=selections,
    )
    outcome = _outcome_matrix(
        approximate=approximate_results,
        comparison=comparison,
        top1=top1,
        selections=selections,
    )
    tune_receipt_set_sha256 = _receipt_set_sha256(phases.tune_receipts)
    selected_summary = _selected_arm_summary(
        observations=observations, parent=parent
    )
    raw_v4_summaries = v4_report.get("finite_arm_summaries")
    if not isinstance(raw_v4_summaries, list):
        raise ValueError("pinned v4 finite arm summaries differ")
    unit_summary = dict(
        next(
            row
            for row in raw_v4_summaries
            if isinstance(row, Mapping) and row.get("arm") == "unit_k64"
        )
    )
    resources = _resource_accounting(
        traces=traces,
        roles=roles,
        endpoint_resources=endpoint_resources,
        gradient_resources=phases.gradient_resources,
        tune_resources=phases.tune_resources,
        final_resources=phases.final_resources,
    )
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    ) and all(
        receipt["maximum_future_gradient_abs"] == 0.0
        and receipt["future_gradient_nonzero_count"] == 0
        for receipt in phases.gradient_receipts
    )
    selected_unit_rows = [
        row for row in observations if row["selected_arm"] == "unit"
    ]
    integrity_gates = {
        "expanded_v2_parent_authenticated": True,
        "v4_control_authenticated_by_exact_file_and_report_sha256": True,
        "live_shared_bank_canonically_reproduced_all_eight_v4_refits": (
            phases.v4_refit_binding["live_refit_count"] == 8
        ),
        "live_shared_bank_canonically_reproduced_all_56_v4_gradient_receipts": (
            phases.v4_refit_binding["live_gradient_receipt_count"] == 56
        ),
        "all_v4_refit_and_gradient_evidence_authenticated_before_microsteps": bool(
            phases.v4_refit_binding["authenticated_before_microstep_execution"]
        ),
        "all_56_v4_unit_tune_cells_authenticated_and_not_reexecuted": (
            phases.v4_unit_tune_binding["unit_tune_receipt_count"] == 56
            and phases.tune_resources["unit_execution_count_per_prompt_fold"] == 0
        ),
        "v4_full_final_set_unit_behavior_and_geometry_authenticated": (
            phases.v4_unit_final_binding["unit_final_observation_count"] == 16
        ),
        "parent_endpoint_traces_folds_and_resources_recollected_exactly": True,
        "checkerboard_fit_and_tune_roles_are_disjoint_with_398_and_405_support_tokens": (
            roles.fit_support_tokens == _EXPECTED_FIT_SUPPORT_TOKENS
            and roles.tune_support_tokens == _EXPECTED_TUNE_SUPPORT_TOKENS
            and set(roles.fit_example_ids).isdisjoint(roles.tune_example_ids)
        ),
        "every_gradient_and_tune_receipt_excludes_its_held_family": all(
            receipt["family_id"] != receipt["held_family_id"]
            for receipt in (*phases.gradient_receipts, *phases.tune_receipts)
        ),
        "all_parent_and_candidate_teacher_kl_vjps_have_zero_future_gradient": (
            causality_passed
        ),
        "unit_k64_static_projection_and_cast_rows_replay_before_refit": True,
        "every_selected_unit_execution_replays_v4_stable_outputs_exactly": all(
            row["selected_unit_stable_outputs_replay_v4_exactly"] is True
            for row in selected_unit_rows
        ),
        "negative_epsilon_is_never_selectable_or_independently_authorizes_primary": all(
            _selection_arm(selection) in {"unit", "plus_epsilon"}
            for selection in selections.values()
        ),
        "exact_model_forward_count_is_264": resources["total_model_forward_count"]
        == _EXPECTED_TOTAL_FORWARDS,
        "exact_backward_call_count_is_494": resources["total_backward_call_count"]
        == _EXPECTED_TOTAL_BACKWARDS,
        "zero_hidden_unit_tune_or_separate_unit_final_candidate_executions": (
            resources["pinned_v4_unit_tune_model_forward_count"] == 0
            and resources["pinned_v4_unit_final_baseline_model_forward_count"]
            == 0
        ),
    }
    integrity_passed = all(integrity_gates.values())
    approximate_passed = bool(approximate_results["passed"])
    passed = integrity_passed and approximate_passed
    if not integrity_passed:
        classification = "candidate_conditioned_k64_microstep_v5_integrity_failed"
    else:
        classification = str(outcome["outcome"])
    primary_gates = {
        **integrity_gates,
        "one_over_64_symmetric_microstep_clears_approximate_90_useful_gates": (
            approximate_passed
        ),
    }
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_hypothesis_use_only",
            "parent_schema": expanded._SCHEMA,
            "frozen_tail_rank": CANDIDATE_GAIN_RANK,
            "frozen_tail_basis_and_order": "whole_family_lofo_token_fisher_v1",
            "candidate_gradient_point": "realized_unit_gain_k64",
            "candidate_gradient_objective": "token_teacher_KL_native_to_candidate",
            "candidate_h4_vjp_semantics": (
                "exact_with_respect_to_realized_post_cast_h4_state"
            ),
            "gain_pullback_cast_semantics": (
                "continuous_local_interpretation_of_final_h4_float_cast"
            ),
            "analytic_gain_pullback_is_finite_displacement_authority": False,
            "finite_executed_post_cast_teacher_KL_is_authority": True,
            "direction": "v4_family_equal_mean_KL_OPG_preconditioned_descent",
            "microstep_epsilon": SYMMETRIC_GAIN_MICROSTEP_EPSILON,
            "microstep_epsilon_hex": SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
            "tune_signed_steps_in_canonical_execution_order": (
                SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex(),
                (-SYMMETRIC_GAIN_MICROSTEP_EPSILON).hex(),
            ),
            "microstep_is_small_finite_not_infinitesimal": True,
            "central_slope_is_finite_central_difference_not_derivative": True,
            "unit_tune_baselines": "authenticated_v4_56_cells_not_reexecuted",
            "positive_microstep_can_select": True,
            "negative_microstep_role": (
                "diagnostic_sign_control_used_only_for_central_slope_guard_"
                "never_selectable_never_final_executed"
            ),
            "tune_selection_objective": "exact_family_equal_mean_teacher_KL",
            "positive_selection_guards": (
                "negative_central_slope",
                "minimum_absolute_or_relative_improvement",
                "at_least_4_of_7_families_nonworse",
                "every_family_within_5pct_plus_1e_minus_8",
            ),
            "central_slope_threshold": 0.0,
            "central_slope_has_independent_numeric_margin": False,
            "positive_improvement_floor": "max_1e_minus_8_or_1e_minus_4_times_unit",
            "posthoc_step_grid_searched": False,
            "candidate_vjp_chunk_size": token_v1._VJP_CHUNK_SIZE,
            "final_arm": _SELECTED_ARM,
            "final_observation_grid": "16_prompts_times_1_frozen_selected_arm",
            "pinned_v4_unit_final_baseline_reexecuted_separately": False,
            "geometry_decides_primary": False,
            "causal_tail_decides_primary": False,
        },
        "expanded_parent_binding": {
            "file": str(expanded_parent_report_path),
            "file_sha256": v3diag.EXPANDED_PARENT_REPORT_FILE_SHA256,
            "report_sha256": v3diag.EXPANDED_PARENT_REPORT_SHA256,
            "schema": parent.get("schema"),
            "classification": parent.get("classification"),
            "k320_reexecuted": False,
        },
        "v4_control_binding": {
            "file": str(v4_report_path),
            "file_sha256": V4_REPORT_FILE_SHA256,
            "report_sha256": V4_REPORT_SHA256,
            "schema": v4_report.get("schema"),
            "classification": v4_report.get("classification"),
            "live_refit_and_gradient_reproduction": phases.v4_refit_binding,
            "unit_tune_baseline_binding": phases.v4_unit_tune_binding,
            "unit_final_behavior_geometry_binding": phases.v4_unit_final_binding,
        },
        "input_binding": {
            "materialization_report_file": str(materialization_report_path),
            "materialization_report_file_sha256": (
                token_v1.MATERIALIZATION_REPORT_FILE_SHA256
            ),
            "materialization_report_sha256": token_v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file": str(transfer_report_path),
            "transfer_report_file_sha256": token_v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": token_v1.TRANSFER_REPORT_SHA256,
            "basis_materialization_binding": materialization_binding,
            "basis_runtime_tensor_sha256": basis_binding["runtime_tensor_sha256"],
            "materialization_schema": materialization.get("schema"),
            "parent_recollection_receipt_sha256": phases.recollection_receipt,
            "unit_k64_static_replay_receipt_sha256": (
                phases.static_unit_replay_receipt
            ),
        },
        "prompt_role_receipt": {
            "artifact_sha256": roles.artifact_sha256,
            "fit_example_ids": roles.fit_example_ids,
            "tune_example_ids": roles.tune_example_ids,
            "fit_support_supervised_token_count": roles.fit_support_tokens,
            "tune_support_supervised_token_count": roles.tune_support_tokens,
            "family_order": families,
            "held_family_excluded_from_its_own_fold_gain_fit_and_tune": True,
        },
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": _endpoint_prompt_receipts(traces),
        "candidate_gain_refits": tuple(refits[family].metadata() for family in families),
        "candidate_microstep_tune_selections": tuple(
            selections[family].metadata() for family in families
        ),
        "candidate_gradient_receipts": phases.gradient_receipts,
        "candidate_gradient_receipt_set_sha256": phases.v4_refit_binding[
            "v4_candidate_gradient_receipt_set_sha256"
        ],
        "candidate_microstep_tune_receipts": phases.tune_receipts,
        "candidate_microstep_tune_receipt_set_sha256": tune_receipt_set_sha256,
        "finite_arm_summaries": (unit_summary, selected_summary),
        "complete_h4_support_teacher_kl_comparison": comparison,
        "selected_top1_comparison_vs_authenticated_v4_unit": top1,
        "established_behavioral_fidelity": {
            "authenticated_v4_unit_k64": phases.v4_unit_behavior,
            _SELECTED_ARM: phases.selected_behavior,
        },
        "executed_cast_once_geometry": {
            "authenticated_v4_unit_k64": phases.v4_unit_geometry,
            _SELECTED_ARM: phases.selected_geometry,
        },
        "geometry_comparison_vs_authenticated_v4_unit": geometry_comparison,
        "strict_established_gate_results_selected_arm": strict_results,
        "approximate_90_useful_microstep_gate_results": approximate_results,
        "outcome_matrix": outcome,
        "finite_observation_receipts": tuple(observations),
        "finite_observation_set_sha256": phases.observation_set_sha256,
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "passed": passed,
        "classification": classification,
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "native_teacher_logits_and_held_native_tails_used": True,
            "tail_basis_and_order_are_family_disjoint": True,
            "gain_fit_and_tune_prompt_roles_are_disjoint": True,
            "gain_fit_and_tune_both_exclude_the_held_family": True,
            "gain_fit_and_tune_are_mutually_family_disjoint": False,
            "frozen_d320_contains_same_a_held_family_information": True,
            "end_to_end_candidate_family_disjoint": False,
            "fresh_confirmation_panel_opened": False,
            "one_over_64_is_a_fixed_small_finite_displacement": True,
            "negative_microstep_is_diagnostic_sign_control": True,
            "negative_microstep_participates_only_in_central_slope_guard": True,
            "negative_result_cannot_select_or_rescue_primary": True,
            "structural_refit_no_op_counted_separately": True,
            "exact_positive_and_negative_one_over_64_tested": True,
            "positive_steps_below_one_over_64_tested": False,
            "positive_steps_between_one_over_64_and_one_over_8_tested": False,
            "other_preconditioners_or_multistep_fits_tested_in_this_rung": False,
            "other_preconditioners_or_multistep_fits_remain_open": True,
            "conditional_gain_fits_tested_in_this_rung": False,
            "conditional_gain_fits_remain_open": True,
            "identically_trained_scalar_gain_control_run": False,
            "fixed_seed_gain_permutation_control_run": False,
            "full_vs_diagonal_preconditioner_control_run": False,
            "diagonal_vs_full_preconditioner_attribution_established": False,
            "mode_specific_value_over_scalar_gain_established": False,
            "missing_controls_block_mode_specific_mechanism_claim": True,
            "one_global_or_deployable_gain_executor_established": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "contains_token_score_matrices": False,
            "contains_gain_vectors": False,
            "contains_basis_coefficients": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
            "artifact_must_remain_outside_git": True,
        },
    }
    return _publish(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authenticated A16 K64 symmetric one-over-64 gain microstep v5 rung."
        )
    )
    parser.add_argument(
        "--expanded-parent-report",
        type=Path,
        default=DEFAULT_EXPANDED_PARENT_REPORT,
    )
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument(
        "--transfer-report", type=Path, default=DEFAULT_TRANSFER_REPORT
    )
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic(
        expanded_parent_report_path=args.expanded_parent_report,
        v4_report_path=args.v4_report,
        materialization_report_path=args.materialization_report,
        transfer_report_path=args.transfer_report,
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":  # pragma: no cover
    main()
