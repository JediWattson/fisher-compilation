"""Shared-bank mean-KL OPG K64 gain diagnostic on the pinned A16 panel.

This v4 rung recollects the exact v3 candidate-gradient bank once, proves that
the residual Gauss-Newton proposal reproduces the pinned v3 artifacts, and
derives a primary family-equal mean-teacher-KL direction from the same OPG
system.  A reversed residual direction is executed as a diagnostic control
only.  Fit, tune, and held-family final prompts remain separated exactly as in
v3; only the selected mean-KL arm can make the primary result pass.

The experiment is same-A, truth-leaking hypothesis evidence.  It does not
authorize serving or establish compression, speed, or deployment readiness.
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
from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from .complete_h4_tail_candidate_gain_refit_v4 import (
    CANDIDATE_GAIN_RANK,
    MEAN_KL_GAIN_ALPHAS,
    REVERSE_RESIDUAL_GAIN_BETAS,
    CandidateConditionedK64GainGradientExampleV4,
    CandidateConditionedK64MeanKLRefit,
    CandidateConditionedK64DualTuneExample,
    CandidateConditionedK64DualTuneSelection,
    fit_candidate_conditioned_k64_mean_kl_gains,
    select_candidate_conditioned_k64_dual_tune_steps,
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
    "DEFAULT_V3_REPORT",
    "V3_REPORT_FILE_SHA256",
    "V3_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = v3diag.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = v3diag.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = v3diag.DEFAULT_EXPANDED_PARENT_REPORT
DEFAULT_V3_REPORT = v3diag.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-gain-refit-lofo-a-fit16-dev-v4.json"
)

V3_REPORT_FILE_SHA256 = (
    "178fafdd9554650a0d90bddc5ce485e252dcd476a395c0423931fbe50202c388"
)
V3_REPORT_SHA256 = (
    "01405ea80856ed4ba2cd78126e7b318f9475f4f65d48aa4ea21d02654e7fe937"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_gain_refit_lofo.v4"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-candidate-gain-refit:v4\0"
_PROVIDER_DOMAIN = b"fisher-graph:complete-h4-k64-candidate-gain-provider:v4\0"
_GRADIENT_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-gain-gradient:v4\0"
_TUNE_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-gain-dual-tune:v4\0"
_OBSERVATION_DOMAIN = b"fisher-graph:complete-h4-k64-gain-observation:v4\0"
_OBSERVATION_SET_DOMAIN = b"fisher-graph:complete-h4-k64-gain-observation-set:v4\0"
_V3_BINDING_DOMAIN = b"fisher-graph:complete-h4-k64-gain-v3-binding:v4\0"
_ARMS = (
    "unit_k64",
    "selected_mean_kl_opg_k64",
    "selected_reverse_residual_k64",
)
_PRIMARY_ARM = "selected_mean_kl_opg_k64"
_CONTROL_ARM = "selected_reverse_residual_k64"
_LEDGERS = v3diag._LEDGERS
_EXPECTED_FIT_SUPPORT_TOKENS = 398
_EXPECTED_TUNE_SUPPORT_TOKENS = 405
_EXPECTED_PARENT_FORWARD_COUNT = 48
_EXPECTED_PARENT_BACKWARD_COUNT = 109
_EXPECTED_GRADIENT_NATIVE_FORWARDS = 8
_EXPECTED_GRADIENT_CANDIDATE_FORWARDS = 56
_EXPECTED_GRADIENT_BACKWARDS = 385
_EXPECTED_TUNE_NATIVE_FORWARDS = 8
_EXPECTED_TUNE_CANDIDATE_FORWARDS = 448
_EXPECTED_FINAL_NATIVE_FORWARDS = 16
_EXPECTED_FINAL_CANDIDATE_FORWARDS = 48
_EXPECTED_TOTAL_FORWARDS = 632
_EXPECTED_TOTAL_BACKWARDS = 494

# Reuse the authenticated v3 trace, fold, and replay machinery while keeping
# every v4 provider, receipt domain, tune grid, and conclusion separate.
_PromptRoles = v3diag._PromptRoles
_checkerboard_prompt_roles = v3diag._checkerboard_prompt_roles
_fresh_native_teacher = v3diag._fresh_native_teacher
_endpoint_indices = v3diag._endpoint_indices
_ordered_k64 = v3diag._ordered_k64
_ordered_k64_relevance = v3diag._ordered_k64_relevance
_authenticate_parent_recollection = v3diag._authenticate_parent_recollection
_authenticate_static_unit_k64_replay = v3diag._authenticate_static_unit_k64_replay
_endpoint_prompt_receipts = v3diag._endpoint_prompt_receipts
_execute_candidate_teacher_kl_vjp = v3diag._execute_candidate_teacher_kl_vjp


def _load_v3_report(path: Path | str) -> dict[str, object]:
    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=V3_REPORT_FILE_SHA256,
        expected_report_sha256=V3_REPORT_SHA256,
        label="candidate gain-refit v3 control",
    )
    if (
        report.get("schema") != v3diag._SCHEMA
        or report.get("classification")
        != "candidate_conditioned_k64_gain_refit_not_supported_same_a"
        or report.get("passed") is not False
    ):
        raise RuntimeError("candidate gain-refit v3 control outcome differs")
    return report


def _tensor_gain_value(value: object, *, label: str) -> Tensor:
    if callable(value):
        value = value()
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must return a tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    if (
        result.shape != (CANDIDATE_GAIN_RANK,)
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError(f"{label} gain geometry differs")
    return result


def _mean_proposed(refit: CandidateConditionedK64MeanKLRefit) -> Tensor:
    return _tensor_gain_value(
        getattr(refit, "mean_proposed_gains_tensor"), label="mean proposed"
    )


def _residual_proposed(refit: CandidateConditionedK64MeanKLRefit) -> Tensor:
    return _tensor_gain_value(
        getattr(refit, "residual_proposed_gains_tensor"),
        label="residual proposed",
    )


def _reverse_residual(
    refit: CandidateConditionedK64MeanKLRefit, beta: float
) -> Tensor:
    method = getattr(refit, "reverse_residual_gains_tensor")
    if not callable(method):
        raise TypeError("reverse residual gain accessor must be callable")
    return _tensor_gain_value(lambda: method(beta), label="reverse residual")


def _selected_mean(selection: CandidateConditionedK64DualTuneSelection) -> float:
    return float(getattr(selection, "selected_mean_alpha"))


def _selected_reverse(selection: CandidateConditionedK64DualTuneSelection) -> float:
    return float(getattr(selection, "selected_reverse_beta"))


def _mean_gains(
    refit: CandidateConditionedK64MeanKLRefit, alpha: float
) -> Tensor:
    if type(alpha) is not float or alpha not in MEAN_KL_GAIN_ALPHAS:
        raise ValueError("mean-KL alpha is outside the fixed grid")
    proposed = _mean_proposed(refit)
    return (1.0 + alpha * (proposed - 1.0)).contiguous()


class _AuthenticatedCandidateGainProviderV4(Gemma3L3L4CorrectionProvider):
    """Single-use correction bound to one v4 variant and finite execution."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "stage",
        "candidate_variant",
        "fold_artifact_sha256",
        "ordered_directions_sha256",
        "gains_sha256",
        "mean_refit_artifact_sha256",
        "selection_artifact_sha256",
        "step_hex",
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
        fold_artifact_sha256: str,
        ordered_directions_sha256: str,
        gains: Tensor,
        mean_refit_artifact_sha256: str | None,
        selection_artifact_sha256: str | None,
        step: float | None,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        if stage not in {"tune", "final"}:
            raise ValueError("v4 candidate provider stage differs")
        if candidate_variant not in {"unit", "mean_kl_opg", "reverse_residual_gn"}:
            raise ValueError("v4 candidate provider variant differs")
        gain_values = gains.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if (
            gain_values.shape != (CANDIDATE_GAIN_RANK,)
            or not bool(torch.isfinite(gain_values).all())
            or bool((gain_values < 0.0).any())
            or bool((gain_values > 1.5).any())
        ):
            raise ValueError("v4 candidate provider gains differ")
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
            raise ValueError("v4 candidate provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        delta = correction.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
        if not bool(torch.isfinite(delta).all()) or bool((delta[~support] != 0).any()):
            raise ValueError("v4 candidate provider correction escapes support")
        unit = torch.equal(gain_values, torch.ones_like(gain_values))
        if candidate_variant == "unit" and not unit:
            raise ValueError("unit v4 provider requires all-one gains")
        if stage == "tune":
            if type(step) is not float or selection_artifact_sha256 is not None:
                raise ValueError("v4 tune provider semantics differ")
            if candidate_variant == "unit" and (
                step != 0.0 or mean_refit_artifact_sha256 is not None
            ):
                raise ValueError("v4 unit tune provider semantics differ")
            if candidate_variant == "mean_kl_opg" and (
                step not in MEAN_KL_GAIN_ALPHAS[1:]
                or mean_refit_artifact_sha256 is None
            ):
                raise ValueError("v4 mean tune provider semantics differ")
            if candidate_variant == "reverse_residual_gn" and (
                step not in REVERSE_RESIDUAL_GAIN_BETAS[1:]
                or mean_refit_artifact_sha256 is None
            ):
                raise ValueError("v4 reverse tune provider semantics differ")
        elif (
            step is not None
            or (
                candidate_variant == "unit"
                and (
                    mean_refit_artifact_sha256 is not None
                    or selection_artifact_sha256 is not None
                )
            )
            or (
                candidate_variant != "unit"
                and (
                    mean_refit_artifact_sha256 is None
                    or selection_artifact_sha256 is None
                )
            )
        ):
            raise ValueError("v4 final provider semantics differ")
        self.site = token_v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.stage = stage
        self.candidate_variant = candidate_variant
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="v4 candidate fold"
        )
        self.ordered_directions_sha256 = _require_sha256(
            ordered_directions_sha256, label="v4 candidate directions"
        )
        self.gains_sha256 = _runtime_tensor_sha256(gain_values)
        self.mean_refit_artifact_sha256 = (
            None
            if mean_refit_artifact_sha256 is None
            else _require_sha256(mean_refit_artifact_sha256, label="v4 mean refit")
        )
        self.selection_artifact_sha256 = (
            None
            if selection_artifact_sha256 is None
            else _require_sha256(selection_artifact_sha256, label="v4 selection")
        )
        self.step_hex = None if step is None else step.hex()
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="v4 model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="v4 bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="v4 prefix"
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
            "schema": "fisher_graph.complete_h4_k64_candidate_gain_provider.v4",
            "rank": CANDIDATE_GAIN_RANK,
            "site": self.site,
            "write_scope": self.write_scope,
            "stage": self.stage,
            "candidate_variant": self.candidate_variant,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "ordered_directions_sha256": self.ordered_directions_sha256,
            "gains_sha256": self.gains_sha256,
            "mean_refit_artifact_sha256": self.mean_refit_artifact_sha256,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "step_hex": self.step_hex,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": (
                "P_D320_R_plus_gain_scaled_frozen_training_token_fisher_K64_tail"
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
            raise RuntimeError("v4 candidate gain provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("v4 candidate gain provider cannot be reused")
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
            raise RuntimeError("v4 candidate provider reached another execution")
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
    mean_refit_artifact_sha256: str | None,
    selection_artifact_sha256: str | None,
    step: float | None,
    model_inputs: Mapping[str, Tensor],
    teacher_logits: Tensor,
    endpoint_indices: Tensor,
) -> tuple[Tensor, object, _AuthenticatedCandidateGainProviderV4, Tensor]:
    directions, _tail, correction_rows, correction = v3diag._candidate_components(
        trace, basis=basis, fit=fit, gains=gains
    )
    provider = _AuthenticatedCandidateGainProviderV4(
        stage=stage,
        candidate_variant=candidate_variant,
        fold_artifact_sha256=fit.artifact_sha256,
        ordered_directions_sha256=_runtime_tensor_sha256(directions),
        gains=gains,
        mean_refit_artifact_sha256=mean_refit_artifact_sha256,
        selection_artifact_sha256=selection_artifact_sha256,
        step=step,
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


def _collect_candidate_gradient_refits(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    roles: _PromptRoles,
) -> tuple[
    dict[str, CandidateConditionedK64MeanKLRefit],
    dict[str, object],
    tuple[dict[str, object], ...],
    dict[str, int],
]:
    """Collect one 8x7 VJP bank and derive v4 plus exact v3 refits."""

    by_id = {trace.example_id: trace for trace in traces}
    v4_examples: dict[
        str, list[CandidateConditionedK64GainGradientExampleV4]
    ] = {family: [] for family in sorted(fits)}
    v3_examples: dict[str, list[object]] = {
        family: [] for family in sorted(fits)
    }
    receipts: list[dict[str, object]] = []
    native_forwards = 0
    candidate_forwards = 0
    backward_calls = 0
    unit_gains = torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
    for example_id in roles.fit_example_ids:
        trace = by_id[example_id]
        model_inputs, indices, targets, teacher_logits = _fresh_native_teacher(
            context=context, trace=trace
        )
        native_forwards += 1
        endpoint_indices, _endpoint_targets, endpoint_grid = _endpoint_indices(
            trace, indices, targets
        )
        for held_family in sorted(fits):
            if held_family == trace.family_id:
                continue
            fit = fits[held_family]
            token_kl, scores, receipt, calls = _execute_candidate_teacher_kl_vjp(
                context=context,
                trace=trace,
                basis=basis,
                fit=fit,
                gains=unit_gains,
                model_inputs=model_inputs,
                teacher_logits=teacher_logits,
                endpoint_indices=endpoint_indices,
                endpoint_grid=endpoint_grid,
            )
            example_v4 = CandidateConditionedK64GainGradientExampleV4(
                example_id=trace.example_id,
                family_id=trace.family_id,
                token_gain_gradients=scores,
                token_teacher_kl=token_kl,
            )
            # This parallel typed view performs no model work.  It is required
            # solely to prove exact v3 residual-refit artifact reproduction.
            example_v3 = v3diag.CandidateConditionedK64GainGradientExample(
                example_id=trace.example_id,
                family_id=trace.family_id,
                token_gain_gradients=scores,
                token_teacher_kl=token_kl,
            )
            v4_examples[held_family].append(example_v4)
            v3_examples[held_family].append(example_v3)
            bound = {
                **receipt,
                "v4_gradient_example_artifact_sha256": example_v4.artifact_sha256,
                "v3_replay_gradient_example_artifact_sha256": (
                    example_v3.artifact_sha256
                ),
                "token_gain_gradients_sha256": example_v4.metadata()[
                    "token_gain_gradients_sha256"
                ],
                "same_vjp_bank_used_for_mean_and_residual_directions": True,
            }
            bound["receipt_sha256"] = token_v1._domain_sha256(
                bound, domain=_GRADIENT_RECEIPT_DOMAIN
            )
            receipts.append(bound)
            candidate_forwards += 1
            backward_calls += calls
            del token_kl, scores
        del model_inputs, indices, targets, teacher_logits
    expected_backward = sum(
        (
            by_id[example_id].endpoint.supervised_tokens
            + token_v1._VJP_CHUNK_SIZE
            - 1
        )
        // token_v1._VJP_CHUNK_SIZE
        * 7
        for example_id in roles.fit_example_ids
    )
    if (
        native_forwards != _EXPECTED_GRADIENT_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
        or backward_calls != _EXPECTED_GRADIENT_BACKWARDS
        or expected_backward != _EXPECTED_GRADIENT_BACKWARDS
        or len(receipts) != _EXPECTED_GRADIENT_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("v4 candidate gradient execution accounting differs")
    refits: dict[str, CandidateConditionedK64MeanKLRefit] = {}
    replayed_v3: dict[str, object] = {}
    for held_family in sorted(fits):
        fit = fits[held_family]
        kwargs = {
            "held_family_id": held_family,
            "parent_fold_artifact_sha256": fit.artifact_sha256,
            "ordered_directions_sha256": _runtime_tensor_sha256(
                _ordered_k64(fit)
            ),
            "ordered_token_fisher_relevance": _ordered_k64_relevance(fit),
        }
        refits[held_family] = fit_candidate_conditioned_k64_mean_kl_gains(
            v4_examples[held_family], **kwargs
        )
        replayed_v3[held_family] = v3diag.fit_candidate_conditioned_k64_gains(
            v3_examples[held_family], **kwargs
        )
    return refits, replayed_v3, tuple(receipts), {
        "gradient_native_forward_count": native_forwards,
        "gradient_candidate_vjp_forward_count": candidate_forwards,
        "gradient_candidate_vjp_backward_call_count": backward_calls,
        "gradient_prompt_fold_count": len(receipts),
        "shared_candidate_gradient_bank_count": len(receipts),
        "additional_backward_calls_for_second_direction": 0,
    }


def _authenticate_reproduced_v3_residuals(
    *,
    v3_report: Mapping[str, object],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
    replayed_v3: Mapping[str, object],
) -> dict[str, object]:
    """Bind exact live residual refits to pinned v3 finite tune evidence."""

    raw_refits = v3_report.get("candidate_gain_refits")
    raw_receipts = v3_report.get("candidate_tune_receipts")
    if not isinstance(raw_refits, list) or not isinstance(raw_receipts, list):
        raise ValueError("pinned v3 residual evidence differs")
    expected = {
        str(row["held_family_id"]): dict(row)
        for row in raw_refits
        if isinstance(row, Mapping)
    }
    if set(expected) != set(refits) or set(replayed_v3) != set(refits):
        raise RuntimeError("pinned v3 residual fold set differs")
    reproduced_rows: list[dict[str, object]] = []
    for family in sorted(refits):
        current = refits[family]
        replay = replayed_v3[family]
        replay_metadata = replay.metadata()
        if (
            v3diag._canonical(replay_metadata)
            != v3diag._canonical(expected[family])
            or not torch.equal(
                current.residual_gradient_c, replay.residual_gradient_c
            )
            or not torch.equal(current.gradient_gram, replay.gradient_gram)
            or not torch.equal(
                current.relevance_regularizer, replay.relevance_regularizer
            )
            or not torch.equal(current.damped_system, replay.damped_system)
            or not torch.equal(
                current.residual_raw_delta, replay.raw_delta
            )
            or not torch.equal(
                current.residual_proposed_gains, replay.proposed_gains
            )
            or current.damping != replay.damping
            or current.residual_raw_delta_rms != replay.raw_delta_rms
            or current.residual_trust_scale != replay.trust_scale
            or current.residual_predicted_derivative != replay.predicted_derivative
            or current.residual_no_op != replay.no_op
        ):
            raise RuntimeError("live v4 bank did not reproduce pinned v3 residual refit")
        reproduced_rows.append(
            {
                "held_family_id": family,
                "v3_residual_refit_artifact_sha256": replay.artifact_sha256,
                "v4_refit_artifact_sha256": current.artifact_sha256,
                "residual_proposed_gains_sha256": replay_metadata[
                    "proposed_gains_sha256"
                ],
            }
        )
    full_receipt_set = v3diag._receipt_set_sha256(
        raw_receipts,
        expected_count=v3diag._EXPECTED_TUNE_CANDIDATE_FORWARDS,
        receipt_domain=v3diag._TUNE_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-tune-set:v1\0",
    )
    if full_receipt_set != v3_report.get("candidate_tune_receipt_set_sha256"):
        raise RuntimeError("pinned v3 residual tune receipt set drifted")
    retained_steps = {0.25.hex(), 0.5.hex()}
    retained = tuple(
        {
            "example_id": row["example_id"],
            "family_id": row["family_id"],
            "held_family_id": row["held_family_id"],
            "refit_artifact_sha256": row["refit_artifact_sha256"],
            "alpha_hex": row["alpha_hex"],
            "gains_sha256": row["gains_sha256"],
            "token_teacher_kl_sha256": row["token_teacher_kl_sha256"],
            "mean_teacher_kl": row["mean_teacher_kl"],
            "receipt_sha256": row["receipt_sha256"],
        }
        for row in raw_receipts
        if isinstance(row, Mapping) and row.get("alpha_hex") in retained_steps
    )
    if len(retained) != 112:
        raise RuntimeError("pinned v3 alpha .25/.5 residual evidence count differs")
    payload = {
        "v3_report_file_sha256": V3_REPORT_FILE_SHA256,
        "v3_report_sha256": V3_REPORT_SHA256,
        "v3_full_tune_receipt_set_sha256": full_receipt_set,
        "exact_residual_refit_reproductions": tuple(reproduced_rows),
        "authenticated_positive_residual_tune_points": retained,
        "authenticated_alpha_hex_grid": tuple(sorted(retained_steps)),
        "positive_residual_tune_point_count": len(retained),
        "positive_residual_points_reexecuted_in_v4": False,
        "binding_role": "pinned_v3_control_not_v4_primary_selection_evidence",
    }
    return {
        **payload,
        "artifact_sha256": token_v1._domain_sha256(
            payload, domain=_V3_BINDING_DOMAIN
        ),
    }


def _tune_candidate_manifest() -> tuple[dict[str, object], ...]:
    manifest = (
        {
            "candidate_variant": "unit",
            "step_kind": "shared_zero",
            "step_hex": 0.0.hex(),
        },
        *(
            {
                "candidate_variant": "mean_kl_opg",
                "step_kind": "alpha",
                "step_hex": alpha.hex(),
            }
            for alpha in MEAN_KL_GAIN_ALPHAS[1:]
        ),
        *(
            {
                "candidate_variant": "reverse_residual_gn",
                "step_kind": "beta",
                "step_hex": beta.hex(),
            }
            for beta in REVERSE_RESIDUAL_GAIN_BETAS[1:]
        ),
    )
    if len(manifest) != 8:
        raise RuntimeError("v4 tune candidate manifest count drifted")
    return manifest


def _collect_dual_tune_selections(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
    roles: _PromptRoles,
) -> tuple[
    dict[str, CandidateConditionedK64DualTuneSelection],
    tuple[dict[str, object], ...],
    dict[str, int],
]:
    """Execute eight unique candidates in each of the 56 tune cells."""

    by_id = {trace.example_id: trace for trace in traces}
    examples_by_fold: dict[str, list[CandidateConditionedK64DualTuneExample]] = {
        family: [] for family in sorted(fits)
    }
    receipts: list[dict[str, object]] = []
    native_forwards = 0
    candidate_forwards = 0
    manifest = _tune_candidate_manifest()
    for example_id in roles.tune_example_ids:
        trace = by_id[example_id]
        model_inputs, indices, targets, teacher_logits = _fresh_native_teacher(
            context=context, trace=trace
        )
        native_forwards += 1
        endpoint_indices, _endpoint_targets, endpoint_grid = _endpoint_indices(
            trace, indices, targets
        )
        for held_family in sorted(fits):
            if held_family == trace.family_id:
                continue
            fit = fits[held_family]
            refit = refits[held_family]
            if refit.held_family_id != held_family:
                raise RuntimeError("v4 tune refit fold binding differs")
            unit_token_kl: Tensor | None = None
            mean_values: list[Tensor] = []
            reverse_values: list[Tensor] = []
            arm_receipts: list[dict[str, object]] = []
            for candidate in manifest:
                variant = str(candidate["candidate_variant"])
                step = float.fromhex(str(candidate["step_hex"]))
                if variant == "unit":
                    gains = torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
                    refit_sha256 = None
                elif variant == "mean_kl_opg":
                    gains = _mean_gains(refit, step)
                    refit_sha256 = refit.artifact_sha256
                elif variant == "reverse_residual_gn":
                    gains = _reverse_residual(refit, step)
                    refit_sha256 = refit.artifact_sha256
                else:  # pragma: no cover - manifest is constructed above.
                    raise RuntimeError("v4 tune candidate variant drifted")
                token_kl, execution, provider, _correction_rows = (
                    _execute_candidate_teacher_kl_forward(
                        context=context,
                        trace=trace,
                        basis=basis,
                        fit=fit,
                        gains=gains,
                        stage="tune",
                        candidate_variant=variant,
                        mean_refit_artifact_sha256=refit_sha256,
                        selection_artifact_sha256=None,
                        step=step,
                        model_inputs=model_inputs,
                        teacher_logits=teacher_logits,
                        endpoint_indices=endpoint_indices,
                    )
                )
                if variant == "unit":
                    if unit_token_kl is not None:
                        raise RuntimeError("v4 tune unit candidate executed twice")
                    unit_token_kl = token_kl
                elif variant == "mean_kl_opg":
                    mean_values.append(token_kl)
                else:
                    reverse_values.append(token_kl)
                arm_receipts.append(
                    {
                        "example_id": trace.example_id,
                        "family_id": trace.family_id,
                        "held_family_id": held_family,
                        "fold_artifact_sha256": fit.artifact_sha256,
                        "mean_refit_artifact_sha256": refit_sha256,
                        "ordered_directions_sha256": (
                            provider.ordered_directions_sha256
                        ),
                        "candidate_variant": variant,
                        "step_kind": candidate["step_kind"],
                        "step_hex": candidate["step_hex"],
                        "gains_sha256": provider.gains_sha256,
                        "provider_artifact_sha256": provider.artifact_sha256,
                        "execution_artifact_sha256": execution.artifact_sha256,
                        "teacher_logits_sha256": _runtime_tensor_sha256(
                            teacher_logits
                        ),
                        "endpoint_supervised_grid_sha256": _runtime_tensor_sha256(
                            endpoint_grid
                        ),
                        "endpoint_supervised_token_count": int(
                            endpoint_grid.shape[0]
                        ),
                        "token_teacher_kl_sha256": _runtime_tensor_sha256(
                            token_kl
                        ),
                        "mean_teacher_kl": float(token_kl.mean()),
                        "stage": "tune",
                        "held_family_used": False,
                        "raw_tensors_serialized": False,
                    }
                )
                candidate_forwards += 1
                del execution, provider, gains
            if (
                unit_token_kl is None
                or len(mean_values) != len(MEAN_KL_GAIN_ALPHAS) - 1
                or len(reverse_values) != len(REVERSE_RESIDUAL_GAIN_BETAS) - 1
            ):
                raise RuntimeError("v4 tune cell candidate grid differs")
            tune_example = CandidateConditionedK64DualTuneExample(
                example_id=trace.example_id,
                family_id=trace.family_id,
                unit_token_teacher_kl=unit_token_kl,
                mean_token_teacher_kl_by_positive_alpha=tuple(mean_values),
                reverse_token_teacher_kl_by_positive_beta=tuple(reverse_values),
            )
            examples_by_fold[held_family].append(tune_example)
            for arm_receipt in arm_receipts:
                bound = {
                    **arm_receipt,
                    "dual_tune_example_artifact_sha256": tune_example.artifact_sha256,
                    "unit_execution_shared_between_alpha_zero_and_beta_zero": True,
                }
                bound["receipt_sha256"] = token_v1._domain_sha256(
                    bound, domain=_TUNE_RECEIPT_DOMAIN
                )
                receipts.append(bound)
            del unit_token_kl, mean_values, reverse_values, arm_receipts
        del model_inputs, indices, targets, teacher_logits
    if (
        native_forwards != _EXPECTED_TUNE_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_TUNE_CANDIDATE_FORWARDS
        or len(receipts) != _EXPECTED_TUNE_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("v4 candidate tune execution accounting differs")
    selections = {
        held_family: select_candidate_conditioned_k64_dual_tune_steps(
            refits[held_family], examples_by_fold[held_family]
        )
        for held_family in sorted(fits)
    }
    return selections, tuple(receipts), {
        "tune_native_forward_count": native_forwards,
        "tune_candidate_forward_count": candidate_forwards,
        "tune_prompt_fold_candidate_count": len(receipts),
        "tune_unique_candidate_count_per_prompt_fold": len(manifest),
        "unit_execution_count_per_prompt_fold": 1,
    }


def _final_candidate_observations(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit],
    selections: Mapping[str, CandidateConditionedK64DualTuneSelection],
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, int],
]:
    """Run native once plus exact unit, mean, and reverse arms on held prompts."""

    manifests = {
        ledger: {
            trace.example_id: trace.family_id
            for trace in traces
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
        for arm in _ARMS
    }
    geometry_traces: list[object] = []
    executed_rows: dict[str, dict[str, Tensor]] = {arm: {} for arm in _ARMS}
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
        if (
            refit.held_family_id != trace.family_id
            or selection.held_family_id != trace.family_id
            or selection.refit_artifact_sha256 != refit.artifact_sha256
        ):
            raise RuntimeError("v4 final candidate fold binding differs")
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
        arms = (
            (
                "unit_k64",
                "unit",
                torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64),
                None,
                None,
            ),
            (
                _PRIMARY_ARM,
                "mean_kl_opg",
                selection.selected_mean_gains_tensor(),
                refit.artifact_sha256,
                selection.artifact_sha256,
            ),
            (
                _CONTROL_ARM,
                "reverse_residual_gn",
                selection.selected_reverse_gains_tensor(),
                refit.artifact_sha256,
                selection.artifact_sha256,
            ),
        )
        for (
            arm,
            variant,
            gains,
            refit_sha256,
            selection_sha256,
        ) in arms:
            token_kl, execution, provider, _correction_rows = (
                _execute_candidate_teacher_kl_forward(
                    context=context,
                    trace=trace,
                    basis=basis,
                    fit=fit,
                    gains=gains,
                    stage="final",
                    candidate_variant=variant,
                    mean_refit_artifact_sha256=refit_sha256,
                    selection_artifact_sha256=selection_sha256,
                    step=None,
                    model_inputs=model_inputs,
                    teacher_logits=teacher_logits,
                    endpoint_indices=endpoint_indices,
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
                        family_id=trace.family_id,
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
                execution.candidate_h4.detach().to(
                    device="cpu", dtype=torch.float64
                )[0].index_select(0, trace.support_indices)
                - trace.base_h4.detach().to(device="cpu", dtype=torch.float64)[0]
                .index_select(0, trace.support_indices)
            ).contiguous()
            executed_rows[arm][trace.example_id] = actual_rows
            prediction = (
                scores.sum(dim=1).contiguous()
                if arm == "unit_k64"
                else (scores * gains.unsqueeze(0)).sum(dim=1).contiguous()
            )
            target = trace.endpoint.compensation_target
            observation: dict[str, object] = {
                "example_id": trace.example_id,
                "family_id": trace.family_id,
                "arm": arm,
                "candidate_variant": variant,
                "rank": CANDIDATE_GAIN_RANK,
                "fold_artifact_sha256": fit.artifact_sha256,
                "mean_refit_artifact_sha256": refit_sha256,
                "selection_artifact_sha256": selection_sha256,
                "selected_mean_alpha_hex": (
                    _selected_mean(selection).hex() if arm == _PRIMARY_ARM else None
                ),
                "selected_reverse_beta_hex": (
                    _selected_reverse(selection).hex()
                    if arm == _CONTROL_ARM
                    else None
                ),
                "ordered_directions_sha256": provider.ordered_directions_sha256,
                "gains_sha256": provider.gains_sha256,
                "provider_artifact_sha256": provider.artifact_sha256,
                "execution_artifact_sha256": execution.artifact_sha256,
                "teacher_logits_sha256": _runtime_tensor_sha256(teacher_logits),
                "endpoint_supervised_grid_sha256": _runtime_tensor_sha256(
                    endpoint_grid
                ),
                "endpoint_supervised_token_count": int(endpoint_grid.shape[0]),
                "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
                "complete_h4_support_mean_teacher_kl": float(token_kl.mean()),
                "token_score_matrix_sha256": _runtime_tensor_sha256(full_scores),
                "native_mean_nll": float(trace.native_token_nll.mean()),
                "d320_mean_nll": float(trace.d320_token_nll.mean()),
                "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
                "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
                "endpoint_baseline_mse": float(target.square().mean()),
                "endpoint_prediction_mse": float(
                    (prediction - target).square().mean()
                ),
                "candidate_h4_bitwise_native": token_v1._bitwise_equal(
                    execution.candidate_h4.detach().to(device="cpu"), trace.native_h4
                ),
                "candidate_logits_bitwise_native": (
                    _runtime_tensor_sha256(execution.logits)
                    == trace.native_logits_sha256
                ),
                "full_tail_reconstruction_max_abs_error": None,
                "exact_residual_provider_used": False,
                "executed_correction_rows_sha256": _runtime_tensor_sha256(
                    actual_rows
                ),
                "held_family_used_for_fit_or_tune": False,
                "held_family_excluded_from_gain_fit_and_tune": True,
                "reverse_control_can_drive_primary": False,
            }
            observation["observation_sha256"] = token_v1._domain_sha256(
                observation, domain=_OBSERVATION_DOMAIN
            )
            observations.append(observation)
            del (
                token_kl,
                execution,
                provider,
                candidate_nll,
                candidate_endpoint_nll,
                candidate_selected,
                actual_rows,
                prediction,
            )
        del (
            model_inputs,
            indices,
            targets,
            teacher_logits,
            source_selected,
            full_scores,
            scores,
        )
    if (
        native_forwards != _EXPECTED_FINAL_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_FINAL_CANDIDATE_FORWARDS
        or len(observations) != _EXPECTED_FINAL_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("v4 final candidate execution accounting differs")
    behavior = {
        arm: {ledger: fidelity[arm][ledger].finalize() for ledger in _LEDGERS}
        for arm in _ARMS
    }
    geometry = {
        "unit_k64": ladder._geometry_with_examples(
            geometry_traces,
            executed_rows["unit_k64"],
            candidate_semantics=(
                "actual_cast_once_d320_plus_training_only_fisher_tail_k64"
            ),
        ),
        _PRIMARY_ARM: ladder._geometry_with_examples(
            geometry_traces,
            executed_rows[_PRIMARY_ARM],
            candidate_semantics=(
                "actual_cast_once_d320_plus_family_equal_mean_KL_OPG_"
                "preconditioned_gain_tail_k64"
            ),
        ),
        _CONTROL_ARM: ladder._geometry_with_examples(
            geometry_traces,
            executed_rows[_CONTROL_ARM],
            candidate_semantics=(
                "actual_cast_once_d320_plus_reversed_residual_GN_gain_"
                "diagnostic_tail_k64"
            ),
        ),
    }
    return observations, behavior, geometry, {
        "final_native_forward_count": native_forwards,
        "final_candidate_forward_count": candidate_forwards,
        "final_observation_count": len(observations),
        "final_arm_count": len(_ARMS),
    }


def _finite_observation_set_sha256(
    observations: Sequence[Mapping[str, object]],
) -> str:
    if len(observations) != token_v1._EXPECTED_EXAMPLES * len(_ARMS):
        raise ValueError("v4 candidate observation count differs")
    identities: set[tuple[str, str]] = set()
    receipts: list[str] = []
    for raw in observations:
        row = dict(raw)
        receipt = row.pop("observation_sha256", None)
        example_id = token_v1._identifier(
            row.get("example_id"), label="v4 candidate observation example_id"
        )
        arm = row.get("arm")
        if arm not in _ARMS or row.get("rank") != CANDIDATE_GAIN_RANK:
            raise ValueError("v4 candidate observation arm/rank differs")
        identity = (example_id, str(arm))
        if identity in identities:
            raise ValueError("v4 candidate observation grid has a duplicate")
        identities.add(identity)
        expected = token_v1._domain_sha256(row, domain=_OBSERVATION_DOMAIN)
        if receipt != expected:
            raise RuntimeError("v4 candidate observation receipt drifted")
        receipts.append(expected)
    if len({example for example, _arm in identities}) != token_v1._EXPECTED_EXAMPLES:
        raise ValueError("v4 candidate observation example grid is incomplete")
    return token_v1._domain_sha256(tuple(receipts), domain=_OBSERVATION_SET_DOMAIN)


def _arm_summary(
    *,
    arm: str,
    observations: Sequence[Mapping[str, object]],
    parent: Mapping[str, object],
) -> dict[str, object]:
    if arm not in _ARMS:
        raise ValueError("v4 candidate summary arm differs")
    parent_rows = parent.get("finite_observation_receipts")
    if not isinstance(parent_rows, list):
        raise ValueError("parent K320 summary anchor differs")
    selected = [
        v3diag._v1_observation_view(row)
        for row in observations
        if row.get("arm") == arm
    ]
    k320 = [
        dict(row)
        for row in parent_rows
        if isinstance(row, Mapping) and row.get("rank") == 320
    ]
    summaries, _gates = token_v1._summarize_observations(
        selected + k320, ranks=(64, 320)
    )
    result = dict(next(row for row in summaries if row["tail_rank"] == 64))
    result["arm"] = arm
    result["k320_reexecuted"] = False
    result["k320_parent_anchor_only"] = True
    return result


def _teacher_kl_comparison(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_arm_family: dict[str, dict[str, list[float]]] = {
        arm: defaultdict(list) for arm in _ARMS
    }
    for row in observations:
        by_arm_family[str(row["arm"])][str(row["family_id"])].append(
            float(row["complete_h4_support_mean_teacher_kl"])
        )
    family_rows: list[dict[str, object]] = []
    for family in sorted(by_arm_family["unit_k64"]):
        values = {
            arm: by_arm_family[arm][family]
            for arm in _ARMS
        }
        if any(len(rows) != 2 for rows in values.values()):
            raise ValueError("v4 held teacher-KL family observation shape differs")
        means = {
            arm: math.fsum(rows) / len(rows) for arm, rows in values.items()
        }
        unit = means["unit_k64"]
        primary = means[_PRIMARY_ARM]
        control = means[_CONTROL_ARM]
        family_rows.append(
            {
                "family_id": family,
                "unit_k64_mean_teacher_kl": unit,
                "selected_mean_kl_opg_k64_mean_teacher_kl": primary,
                "selected_reverse_residual_k64_mean_teacher_kl": control,
                "mean_absolute_delta_minus_unit": primary - unit,
                "reverse_absolute_delta_minus_unit": control - unit,
                "mean_relative_improvement": (
                    (unit - primary)
                    / max(unit, torch.finfo(torch.float64).tiny)
                ),
                "reverse_relative_improvement": (
                    (unit - control)
                    / max(unit, torch.finfo(torch.float64).tiny)
                ),
                "mean_improved": primary < unit,
                "reverse_improved": control < unit,
                "mean_within_five_percent_plus_1e_minus_8": (
                    primary <= 1.05 * unit + 1.0e-8
                ),
                "reverse_within_five_percent_plus_1e_minus_8": (
                    control <= 1.05 * unit + 1.0e-8
                ),
            }
        )
    if len(family_rows) != token_v1._EXPECTED_FAMILIES:
        raise ValueError("v4 held teacher-KL family count differs")

    def macro(key: str) -> float:
        return math.fsum(float(row[key]) for row in family_rows) / len(family_rows)

    unit_macro = macro("unit_k64_mean_teacher_kl")
    primary_macro = macro("selected_mean_kl_opg_k64_mean_teacher_kl")
    control_macro = macro("selected_reverse_residual_k64_mean_teacher_kl")
    return {
        "ledger": "complete_h4_support",
        "aggregation": "family_then_prompt_equal",
        "unit_k64_family_macro_mean_teacher_kl": unit_macro,
        "selected_mean_kl_opg_k64_family_macro_mean_teacher_kl": primary_macro,
        "selected_reverse_residual_k64_family_macro_mean_teacher_kl": (
            control_macro
        ),
        "mean_family_macro_relative_improvement": (
            (unit_macro - primary_macro)
            / max(unit_macro, torch.finfo(torch.float64).tiny)
        ),
        "reverse_family_macro_relative_improvement": (
            (unit_macro - control_macro)
            / max(unit_macro, torch.finfo(torch.float64).tiny)
        ),
        "mean_held_family_improvement_count": sum(
            bool(row["mean_improved"]) for row in family_rows
        ),
        "reverse_held_family_improvement_count": sum(
            bool(row["reverse_improved"]) for row in family_rows
        ),
        "mean_all_held_families_within_five_percent_plus_1e_minus_8": all(
            bool(row["mean_within_five_percent_plus_1e_minus_8"])
            for row in family_rows
        ),
        "reverse_all_held_families_within_five_percent_plus_1e_minus_8": all(
            bool(row["reverse_within_five_percent_plus_1e_minus_8"])
            for row in family_rows
        ),
        "families": tuple(family_rows),
        "reverse_arm_is_diagnostic_only": True,
    }


def _top1_comparison(
    behavior: Mapping[str, Mapping[str, object]], *, candidate_arm: str
) -> dict[str, object]:
    if candidate_arm not in {_PRIMARY_ARM, _CONTROL_ARM}:
        raise ValueError("v4 top1 candidate arm differs")
    rows: dict[str, object] = {}
    for ledger in _LEDGERS:
        unit = v3diag._mapping(
            behavior["unit_k64"][ledger], label=f"unit {ledger} behavior"
        )
        candidate = v3diag._mapping(
            behavior[candidate_arm][ledger],
            label=f"{candidate_arm} {ledger} behavior",
        )

        def score(row: Mapping[str, object], *, family: bool) -> float:
            if family:
                return float(
                    v3diag._mapping(
                        v3diag._mapping(
                            row.get("family_summary"), label="family summary"
                        ).get("macro"),
                        label="family macro",
                    ).get("top1_agreement_to_source")
                )
            return float(
                v3diag._mapping(row.get("aggregate"), label="aggregate").get(
                    "top1_agreement_to_source"
                )
            )

        unit_aggregate = score(unit, family=False)
        candidate_aggregate = score(candidate, family=False)
        unit_macro = score(unit, family=True)
        candidate_macro = score(candidate, family=True)
        rows[ledger] = {
            "unit_aggregate_top1": unit_aggregate,
            "candidate_aggregate_top1": candidate_aggregate,
            "aggregate_delta_candidate_minus_unit": (
                candidate_aggregate - unit_aggregate
            ),
            "unit_family_macro_top1": unit_macro,
            "candidate_family_macro_top1": candidate_macro,
            "family_macro_delta_candidate_minus_unit": candidate_macro - unit_macro,
            "aggregate_at_least_point_90": candidate_aggregate >= 0.90,
            "family_macro_at_least_point_90": candidate_macro >= 0.90,
            "aggregate_no_material_regression_vs_unit": (
                candidate_aggregate >= unit_aggregate - 0.01
            ),
            "family_macro_no_material_regression_vs_unit": (
                candidate_macro >= unit_macro - 0.01
            ),
            "decides_primary": candidate_arm == _PRIMARY_ARM and ledger != "causal_tail",
        }
    return rows


def _gate_results(
    *,
    behavior: Mapping[str, Mapping[str, object]],
    geometry: Mapping[str, Mapping[str, object]],
    teacher_kl_comparison: Mapping[str, object],
    primary_top1: Mapping[str, object],
    selections: Mapping[str, CandidateConditionedK64DualTuneSelection],
) -> tuple[dict[str, object], dict[str, object]]:
    strict: dict[str, object] = {}
    for arm in _ARMS:
        ledger_pass = {
            ledger: bool(
                v3diag._mapping(
                    v3diag._mapping(
                        behavior[arm][ledger], label=f"{arm} {ledger} behavior"
                    ).get("gates"),
                    label=f"{arm} {ledger} gates",
                ).get("passed")
            )
            for ledger in _LEDGERS
        }
        geometry_pass = bool(
            v3diag._mapping(
                v3diag._mapping(geometry[arm], label=f"{arm} geometry").get(
                    "gates"
                ),
                label=f"{arm} geometry gates",
            ).get("passed")
        )
        strict[arm] = {
            "behavior_ledger_pass": ledger_pass,
            "behavioral_only_strict_passed": all(ledger_pass.values()),
            "geometry_passed": geometry_pass,
            "full_behavior_plus_geometry_strict_passed": (
                all(ledger_pass.values()) and geometry_pass
            ),
        }
    positive_mean_count = sum(
        _selected_mean(selection) > 0.0 for selection in selections.values()
    )
    positive_reverse_count = sum(
        _selected_reverse(selection) > 0.0 for selection in selections.values()
    )
    gates: dict[str, bool] = {
        "mean_family_macro_complete_h4_support_teacher_kl_improves_at_least_2pct": (
            float(teacher_kl_comparison["mean_family_macro_relative_improvement"])
            >= 0.02
        ),
        "mean_held_family_teacher_kl_improvement_count_at_least_6_of_8": (
            int(teacher_kl_comparison["mean_held_family_improvement_count"]) >= 6
        ),
        "mean_worst_held_family_teacher_kl_regression_at_most_5pct_plus_1e_minus_8": bool(
            teacher_kl_comparison[
                "mean_all_held_families_within_five_percent_plus_1e_minus_8"
            ]
        ),
        "folds_selecting_positive_mean_alpha_at_least_6_of_8": (
            positive_mean_count >= 6
        ),
    }
    for ledger in ("ordinary", "complete_h4_support", "graph_core"):
        row = v3diag._mapping(primary_top1[ledger], label=f"{ledger} top1")
        gates[f"mean_{ledger}_aggregate_top1_at_least_point_90"] = bool(
            row["aggregate_at_least_point_90"]
        )
        gates[f"mean_{ledger}_family_macro_top1_at_least_point_90"] = bool(
            row["family_macro_at_least_point_90"]
        )
        gates[
            f"mean_{ledger}_aggregate_top1_no_material_regression_vs_unit"
        ] = bool(row["aggregate_no_material_regression_vs_unit"])
        gates[
            f"mean_{ledger}_family_macro_top1_no_material_regression_vs_unit"
        ] = bool(row["family_macro_no_material_regression_vs_unit"])
    approximate = {
        "gates": tuple(sorted(gates.items())),
        "passed": all(gates.values()),
        "positive_mean_alpha_fold_count": positive_mean_count,
        "positive_mean_alpha_required_count": 6,
        "positive_reverse_beta_fold_count_report_only": positive_reverse_count,
        "reverse_control_used_by_any_primary_gate": False,
        "squared_KL_used_by_any_selection_or_primary_gate": False,
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


def _receipt_set_sha256(
    receipts: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
    receipt_domain: bytes,
    set_domain: bytes,
) -> str:
    if len(receipts) != expected_count:
        raise ValueError("v4 candidate receipt-set count differs")
    values: list[str] = []
    identities: set[tuple[object, ...]] = set()
    for raw in receipts:
        row = dict(raw)
        receipt = row.pop("receipt_sha256", None)
        expected = token_v1._domain_sha256(row, domain=receipt_domain)
        if receipt != expected:
            raise RuntimeError("v4 candidate gain receipt drifted")
        identity = (
            row.get("held_family_id"),
            row.get("example_id"),
            row.get("candidate_variant"),
            row.get("step_hex"),
        )
        if identity in identities:
            raise ValueError("v4 candidate receipt set has a duplicate")
        identities.add(identity)
        values.append(expected)
    return token_v1._domain_sha256(tuple(values), domain=set_domain)


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
    ):
        raise RuntimeError("v4 candidate gain full resource accounting differs")
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
    tune_row_executions = 7 * 8 * tune_rows
    final_row_executions = 3 * support_rows
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
        "candidate_vjp_chunk_size_rationale": (
            "fixed_at_8_for_bounded_memory_matching_parent_vjp_protocol"
        ),
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
        "k320_model_forward_count": 0,
        "k320_reexecution_performed": False,
        "unique_recollected_candidate_teacher_kl_vjp_bank_count_per_prompt_fold": 1,
        "contracted_gradient_typed_views_per_prompt_fold": 2,
        "contracted_gradient_typed_view_roles": (
            "v4_shared_mean_and_residual_fit",
            "v3_exact_artifact_reproduction",
        ),
        "raw_candidate_vjp_banks_retained_in_report": False,
        "additional_model_or_backward_work_for_second_fit_direction": 0,
        "serving_learned_parameter_count": "not_applicable_no_serving_artifact",
        "serving_logical_macs_per_token": "not_applicable_no_serving_artifact",
    }


def _outcome_matrix(
    *,
    approximate: Mapping[str, object],
    comparison: Mapping[str, object],
    selections: Mapping[str, CandidateConditionedK64DualTuneSelection],
) -> dict[str, object]:
    mean_count = sum(_selected_mean(value) > 0.0 for value in selections.values())
    reverse_count = sum(
        _selected_reverse(value) > 0.0 for value in selections.values()
    )
    mean_supported = bool(approximate["passed"])
    reverse_mechanism_signal = (
        float(comparison["reverse_family_macro_relative_improvement"]) >= 0.02
        and int(comparison["reverse_held_family_improvement_count"]) >= 6
    )
    if mean_supported:
        outcome = "mean_KL_OPG_direction_supported_same_a"
    elif reverse_mechanism_signal:
        outcome = "reverse_residual_sign_control_signal_without_primary_support"
    elif mean_count == 0:
        outcome = "mean_KL_OPG_grid_abstained_at_alpha_at_least_one_eighth"
    else:
        outcome = "tested_mean_KL_OPG_direction_not_supported_same_a"
    return {
        "outcome": outcome,
        "mean_primary_supported": mean_supported,
        "mean_positive_alpha_fold_count": mean_count,
        "reverse_positive_beta_fold_count": reverse_count,
        "reverse_residual_mechanism_signal_report_only": reverse_mechanism_signal,
        "reverse_control_can_rescue_primary": False,
        "all_mean_folds_abstained": mean_count == 0,
        "smallest_tested_positive_mean_alpha": MEAN_KL_GAIN_ALPHAS[1],
        "if_all_mean_folds_abstain_claim_is_limited_to": (
            "no_safe_mean_KL_step_found_on_predeclared_alpha_grid_0.125_to_1.0"
        ),
        "does_not_exclude_smaller_than_one_eighth_step": True,
        "does_not_exclude_other_preconditioner_or_multistep_fit": True,
        "squared_KL_can_change_reverse_control_diagnostics_but_not_primary_pass": (
            True
        ),
    }


@dataclass(slots=True)
class _CandidatePhaseResults:
    recollection_receipt: str
    static_unit_replay_receipt: str
    roles: _PromptRoles
    refits: Mapping[str, CandidateConditionedK64MeanKLRefit]
    replayed_v3_refits: Mapping[str, object]
    v3_residual_binding: Mapping[str, object]
    gradient_receipts: tuple[dict[str, object], ...]
    gradient_resources: Mapping[str, int]
    selections: Mapping[str, CandidateConditionedK64DualTuneSelection]
    tune_receipts: tuple[dict[str, object], ...]
    tune_resources: Mapping[str, int]
    observations: list[dict[str, object]]
    observation_set_sha256: str
    behavior: Mapping[str, Mapping[str, object]]
    geometry: Mapping[str, Mapping[str, object]]
    final_resources: Mapping[str, int]
    unit_summary: dict[str, object]
    unit_replay_receipt: str


def _execute_candidate_phases(
    *,
    context: object,
    parent: Mapping[str, object],
    v3_report: Mapping[str, object],
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
    refits, replayed_v3, gradient_receipts, gradient_resources = (
        _collect_candidate_gradient_refits(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            roles=roles,
        )
    )
    v3_residual_binding = _authenticate_reproduced_v3_residuals(
        v3_report=v3_report, refits=refits, replayed_v3=replayed_v3
    )
    selections, tune_receipts, tune_resources = _collect_dual_tune_selections(
        context=context,
        traces=traces,
        basis=basis,
        fits=fits,
        refits=refits,
        roles=roles,
    )
    observations, behavior, geometry, final_resources = (
        _final_candidate_observations(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            refits=refits,
            selections=selections,
        )
    )
    observation_set_sha256 = _finite_observation_set_sha256(observations)
    unit_summary, unit_replay_receipt = v3diag._authenticate_unit_k64_replay(
        parent=parent,
        observations=observations,
        behavior=behavior,
        geometry=geometry,
    )
    return _CandidatePhaseResults(
        recollection_receipt=recollection_receipt,
        static_unit_replay_receipt=static_unit_replay_receipt,
        roles=roles,
        refits=refits,
        replayed_v3_refits=replayed_v3,
        v3_residual_binding=v3_residual_binding,
        gradient_receipts=gradient_receipts,
        gradient_resources=gradient_resources,
        selections=selections,
        tune_receipts=tune_receipts,
        tune_resources=tune_resources,
        observations=observations,
        observation_set_sha256=observation_set_sha256,
        behavior=behavior,
        geometry=geometry,
        final_resources=final_resources,
        unit_summary=unit_summary,
        unit_replay_receipt=unit_replay_receipt,
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


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    v3_report_path: Path | str = DEFAULT_V3_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned same-A v4 mean-KL/shared-bank diagnostic."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite candidate gain-refit v4 report")
    parent = v3diag._load_expanded_parent(expanded_parent_report_path)
    v3_report = _load_v3_report(v3_report_path)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate gain v4 rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate gain v4 rank320 transfer",
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
            raise RuntimeError("candidate gain v4 A16 panel shape differs")
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
            v3_report=v3_report,
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
    behavior = phases.behavior
    geometry = phases.geometry
    arm_summaries = tuple(
        _arm_summary(arm=arm, observations=observations, parent=parent)
        for arm in _ARMS
    )
    comparison = _teacher_kl_comparison(observations)
    primary_top1 = _top1_comparison(behavior, candidate_arm=_PRIMARY_ARM)
    control_top1 = _top1_comparison(behavior, candidate_arm=_CONTROL_ARM)
    strict_results, approximate_results = _gate_results(
        behavior=behavior,
        geometry=geometry,
        teacher_kl_comparison=comparison,
        primary_top1=primary_top1,
        selections=selections,
    )
    gradient_receipt_set_sha256 = _receipt_set_sha256(
        phases.gradient_receipts,
        expected_count=_EXPECTED_GRADIENT_CANDIDATE_FORWARDS,
        receipt_domain=_GRADIENT_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-gradient-set:v4\0",
    )
    tune_receipt_set_sha256 = _receipt_set_sha256(
        phases.tune_receipts,
        expected_count=_EXPECTED_TUNE_CANDIDATE_FORWARDS,
        receipt_domain=_TUNE_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-dual-tune-set:v4\0",
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
    integrity_gates = {
        "expanded_v2_parent_authenticated": True,
        "failed_v3_control_report_authenticated_by_file_and_report_sha256": True,
        "live_shared_bank_reproduced_all_eight_v3_residual_refit_artifacts": (
            len(phases.v3_residual_binding["exact_residual_refit_reproductions"])
            == 8
        ),
        "pinned_v3_positive_residual_alpha_point25_and_point5_receipts_bound": (
            phases.v3_residual_binding["positive_residual_tune_point_count"]
            == 112
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
        "unit_k64_stable_outputs_summary_behavior_and_geometry_replay_exactly": True,
        "unit_k64_static_projection_and_cast_rows_replay_before_refit": True,
        "exact_model_forward_count_is_632": resources["total_model_forward_count"]
        == _EXPECTED_TOTAL_FORWARDS,
        "exact_backward_call_count_is_494": resources["total_backward_call_count"]
        == _EXPECTED_TOTAL_BACKWARDS,
        "second_fit_direction_added_zero_model_or_backward_calls": (
            resources["additional_model_or_backward_work_for_second_fit_direction"]
            == 0
        ),
    }
    approximate_passed = bool(approximate_results["passed"])
    primary_gates = {
        **integrity_gates,
        "mean_KL_OPG_refit_clears_approximate_90_useful_gates": (
            approximate_passed
        ),
    }
    outcome = _outcome_matrix(
        approximate=approximate_results,
        comparison=comparison,
        selections=selections,
    )
    integrity_passed = all(integrity_gates.values())
    passed = integrity_passed and approximate_passed
    if not integrity_passed:
        classification = "candidate_conditioned_k64_mean_KL_OPG_v4_integrity_failed"
    elif approximate_passed:
        classification = (
            "candidate_conditioned_k64_mean_KL_OPG_refit_supported_same_a"
        )
    else:
        classification = str(outcome["outcome"])
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
            "shared_candidate_gradient_bank": True,
            "shared_bank_quantities": (
                "token_gain_gradient_q",
                "token_teacher_KL",
                "gradient_gram_F",
                "relevance_regularizer_R",
                "damped_system_H",
            ),
            "primary_mean_gradient": "b_equal_family_equal_expected_q",
            "primary_direction": "negative_H_inverse_b",
            "primary_direction_name": "mean_KL_OPG_preconditioned_descent",
            "preconditioner_structure": (
                "full_64_by_64_OPG_plus_diagonal_relevance_regularizer"
            ),
            "primary_direction_is_not": (
                "Gauss_Newton",
                "natural_gradient",
                "Hessian",
                "exact_GGN",
            ),
            "residual_control_direction": "negative_H_inverse_expected_q_times_KL",
            "reverse_control_formula": "one_minus_beta_times_residual_proposal_minus_one",
            "reverse_control_role": "diagnostic_only_never_primary",
            "reverse_control_clamp_applied": False,
            "reverse_beta_maximum_keeps_gains_within_fixed_bounds": 0.5,
            "mean_alpha_grid": MEAN_KL_GAIN_ALPHAS,
            "reverse_beta_grid": REVERSE_RESIDUAL_GAIN_BETAS,
            "alpha_zero_and_beta_zero_share_exact_unit_execution": True,
            "unique_tune_candidates_per_prompt_fold": 8,
            "tune_selection_objective": "exact_family_equal_mean_teacher_KL",
            "squared_KL_is_report_only": True,
            "finite_executed_teacher_KL_is_selection_authority": True,
            "candidate_vjp_chunk_size": token_v1._VJP_CHUNK_SIZE,
            "candidate_vjp_chunk_size_policy": (
                "bounded_memory_chunk8_authoritative_not_single_batch_chunk128"
            ),
            "final_arms": _ARMS,
            "final_observation_grid": "16_prompts_times_3_arms",
            "only_mean_arm_drives_primary": True,
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
        "failed_v3_control_binding": {
            "file": str(v3_report_path),
            "file_sha256": V3_REPORT_FILE_SHA256,
            "report_sha256": V3_REPORT_SHA256,
            "schema": v3_report.get("schema"),
            "classification": v3_report.get("classification"),
            "residual_reproduction_and_tune_binding": phases.v3_residual_binding,
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
            "unit_k64_replay_receipt_sha256": phases.unit_replay_receipt,
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
        "candidate_gain_refits": tuple(
            refits[family].metadata() for family in families
        ),
        "candidate_dual_tune_selections": tuple(
            selections[family].metadata() for family in families
        ),
        "candidate_gradient_receipts": phases.gradient_receipts,
        "candidate_gradient_receipt_set_sha256": gradient_receipt_set_sha256,
        "candidate_dual_tune_receipts": phases.tune_receipts,
        "candidate_dual_tune_receipt_set_sha256": tune_receipt_set_sha256,
        "finite_arm_summaries": arm_summaries,
        "complete_h4_support_teacher_kl_comparison": comparison,
        "primary_mean_top1_comparison": primary_top1,
        "reverse_control_top1_comparison_report_only": control_top1,
        "established_behavioral_fidelity_by_arm": behavior,
        "executed_cast_once_geometry_by_arm": geometry,
        "strict_established_gate_results_by_arm": strict_results,
        "approximate_90_useful_mean_refit_gate_results": approximate_results,
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
            "mean_KL_OPG_direction_is_primary": True,
            "reverse_residual_direction_is_diagnostic_only": True,
            "reverse_result_cannot_rescue_primary": True,
            "squared_KL_does_not_select_or_gate": True,
            "identically_trained_scalar_gain_control_run": False,
            "fixed_seed_gain_permutation_control_run": False,
            "full_vs_diagonal_preconditioner_control_run": False,
            "diagonal_vs_full_preconditioner_attribution_established": False,
            "mode_specific_value_over_scalar_gain_established": False,
            "missing_controls_block_mode_specific_mechanism_claim": True,
            "all_mean_abstention_only_rejects_tested_alpha_grid": True,
            "alpha_below_one_eighth_not_tested": True,
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
            "Run the authenticated A16 K64 shared-bank mean-KL OPG v4 rung."
        )
    )
    parser.add_argument(
        "--expanded-parent-report",
        type=Path,
        default=DEFAULT_EXPANDED_PARENT_REPORT,
    )
    parser.add_argument("--v3-report", type=Path, default=DEFAULT_V3_REPORT)
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
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic(
        expanded_parent_report_path=args.expanded_parent_report,
        v3_report_path=args.v3_report,
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
