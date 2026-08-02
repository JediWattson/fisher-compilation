"""Candidate-conditioned K64 gain refit on the authenticated A16 panel.

This rung keeps every upstream token-Fisher basis and ordering frozen.  For
each held-family fold it differentiates teacher KL at the *realized* unit-gain
K64 candidate on one checkerboard prompt per training family, fits one bounded
residual Gauss-Newton gain step, and selects a fixed alpha on the complementary
checkerboard prompts.  For each fold, both prompts from that fold's held
family are excluded from its gain fit and tune, then used in its final two-arm
observation.  They still materialize in parent recollection and train the
other seven LOFO folds.

The experiment is same-A, truth-leaking hypothesis evidence.  It neither
authorizes serving nor claims compression, speed, or deployment readiness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen
from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1
from . import gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded
from . import gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl
from .complete_h4_tail_candidate_gain_refit import (
    CANDIDATE_GAIN_ALPHAS,
    CANDIDATE_GAIN_RANK,
    CandidateConditionedK64GainGradientExample,
    CandidateConditionedK64GainRefit,
    CandidateConditionedK64GainTuneExample,
    CandidateConditionedK64GainTuneSelection,
    contract_candidate_teacher_kl_gain_scores,
    fit_candidate_conditioned_k64_gains,
    select_candidate_conditioned_k64_gain_alpha,
)
from .complete_h4_tail_token_fisher import (
    CompleteH4TailHeldFamilyFit,
    complete_h4_tail_gate_scores,
    fit_complete_h4_tail_held_family,
    project_complete_h4_tail_prefix,
    project_complete_h4_tail_rows,
)
from .gemma3_l3_l4_complete_h4_one_pass_transfer import (
    _load_committed_basis,
    _native_boundary,
    _retokenize,
)
from .gemma3_l3_l4_complete_h4_projection import CompleteH4ProjectionFitSequence
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
    Gemma3L3L4OnePassPrefix,
    _require_sha256,
    _runtime_tensor_sha256,
    gemma3_l3_l4_shadow_model_inputs_sha256,
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
    "EXPANDED_PARENT_REPORT_FILE_SHA256",
    "EXPANDED_PARENT_REPORT_SHA256",
    "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic",
    "main",
]


DEFAULT_MATERIALIZATION_REPORT = token_v1.DEFAULT_MATERIALIZATION_REPORT
DEFAULT_TRANSFER_REPORT = token_v1.DEFAULT_TRANSFER_REPORT
DEFAULT_EXPANDED_PARENT_REPORT = expanded.DEFAULT_OUTPUT
DEFAULT_OUTPUT = token_v1._LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-tail-"
    "token-fisher-k64-candidate-gain-refit-lofo-a-fit16-dev-v3.json"
)

EXPANDED_PARENT_REPORT_FILE_SHA256 = (
    "e7736b60084c8e5bbb83f44cc77613e09242848230137def3afa862162284721"
)
EXPANDED_PARENT_REPORT_SHA256 = (
    "26010938b5b81dbce9e05607acd46e5b9e0beea1d981edbd91d5e841365799fa"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_tail_token_fisher_"
    "candidate_gain_refit_lofo.v3"
)
_REPORT_DOMAIN = b"fisher-graph:complete-h4-k64-candidate-gain-refit:v3\0"
_PROVIDER_DOMAIN = b"fisher-graph:complete-h4-k64-candidate-gain-provider:v1\0"
_ROLE_DOMAIN = b"fisher-graph:complete-h4-k64-candidate-gain-roles:v1\0"
_RECOLLECTION_DOMAIN = b"fisher-graph:complete-h4-k64-gain-recollection:v1\0"
_GRADIENT_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-gain-gradient:v1\0"
_TUNE_RECEIPT_DOMAIN = b"fisher-graph:complete-h4-k64-gain-tune:v1\0"
_OBSERVATION_DOMAIN = b"fisher-graph:complete-h4-k64-gain-observation:v1\0"
_OBSERVATION_SET_DOMAIN = (
    b"fisher-graph:complete-h4-k64-gain-observation-set:v1\0"
)
_ARMS = ("unit_k64", "selected_refit_k64")
_LEDGERS = ("ordinary", "complete_h4_support", "graph_core", "causal_tail")
_EXPECTED_FIT_SUPPORT_TOKENS = 398
_EXPECTED_TUNE_SUPPORT_TOKENS = 405
_EXPECTED_PARENT_FORWARD_COUNT = 48
_EXPECTED_PARENT_BACKWARD_COUNT = 109
_EXPECTED_GRADIENT_NATIVE_FORWARDS = 8
_EXPECTED_GRADIENT_CANDIDATE_FORWARDS = 56
_EXPECTED_GRADIENT_BACKWARDS = 385
_EXPECTED_TUNE_NATIVE_FORWARDS = 8
_EXPECTED_TUNE_CANDIDATE_FORWARDS = 224
_EXPECTED_FINAL_NATIVE_FORWARDS = 16
_EXPECTED_FINAL_CANDIDATE_FORWARDS = 32


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _canonical(value: object) -> object:
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def _load_expanded_parent(path: Path | str) -> dict[str, object]:
    """Authenticate the exact completed expanded-v2 parent and K320 anchor."""

    report = token_v1._load_pinned_report(
        path,
        expected_file_sha256=EXPANDED_PARENT_REPORT_FILE_SHA256,
        expected_report_sha256=EXPANDED_PARENT_REPORT_SHA256,
        label="candidate gain expanded token-Fisher parent",
    )
    authenticated = teacher_kl._load_expanded_control_report(path)
    if authenticated != report:
        raise RuntimeError("expanded parent authenticators disagree")
    protocol = _mapping(report.get("protocol"), label="expanded parent protocol")
    science = _mapping(
        report.get("scientific_status"), label="expanded parent science"
    )
    safety = _mapping(report.get("safety"), label="expanded parent safety")
    decisions = _mapping(
        report.get("fidelity_and_geometry_pass_by_rank"),
        label="expanded parent decisions",
    )
    expected_decisions = {
        "64": False,
        "96": False,
        "128": False,
        "160": False,
        "192": False,
        "256": True,
        "320": True,
    }
    if (
        report.get("schema") != expanded._SCHEMA
        or report.get("classification")
        != "adaptive_same_a_smallest_tail_rank_256_cleared_established_gates"
        or report.get("passed") is not True
        or tuple(protocol.get("tail_ranks", ())) != expanded.EXPANDED_TAIL_RANKS
        or report.get(
            "smallest_tail_rank_below_320_clearing_established_fidelity_and_geometry_gates"
        )
        != 256
        or dict(decisions) != expected_decisions
        or science.get("same_a_adaptive_hypothesis_use_only") is not True
        or science.get("fresh_confirmation_panel_opened") is not False
        or science.get("candidate_serving_authorized") is not False
        or science.get("compression_claim") is not False
        or safety.get("contains_prompt_text") is not False
        or safety.get("contains_token_ids") is not False
        or safety.get("contains_only_hashes_counts_and_scalar_metrics") is not True
    ):
        raise ValueError("expanded token-Fisher parent semantics differ")
    raw = report.get("finite_observation_receipts")
    if not isinstance(raw, list):
        raise ValueError("expanded parent observation grid differs")
    observations = [
        _mapping(value, label="expanded parent observation") for value in raw
    ]
    if (
        token_v1._finite_observation_set_sha256(
            observations, ranks=expanded.EXPANDED_TAIL_RANKS
        )
        != report.get("finite_observation_set_sha256")
    ):
        raise RuntimeError("expanded parent observation receipt drifted")
    ladder = report.get("finite_ladder")
    behavior = _mapping(
        report.get("established_behavioral_fidelity_by_rank"),
        label="expanded parent behavior",
    )
    geometry = _mapping(
        report.get("executed_cast_once_geometry_by_rank"),
        label="expanded parent geometry",
    )
    if not isinstance(ladder, list):
        raise ValueError("expanded parent finite ladder differs")
    k320 = next(
        (row for row in ladder if isinstance(row, Mapping) and row.get("tail_rank") == 320),
        None,
    )
    k320_behavior = _mapping(behavior.get("320"), label="parent K320 behavior")
    k320_geometry = _mapping(geometry.get("320"), label="parent K320 geometry")
    if (
        k320 is None
        or k320.get("every_prompt_h4_bitwise_native") is not True
        or k320.get("every_prompt_logits_bitwise_native") is not True
        or float(k320.get("maximum_full_tail_reconstruction_abs_error", math.inf))
        > 1.0e-9
        or any(
            _mapping(k320_behavior.get(ledger), label=f"parent K320 {ledger}")
            .get("gates", {})
            .get("passed")
            is not True
            for ledger in _LEDGERS
        )
        or _mapping(k320_geometry.get("gates"), label="parent K320 geometry gates")
        .get("passed")
        is not True
    ):
        raise ValueError("expanded parent K320 exact/pass anchor differs")
    return report


@dataclass(frozen=True, slots=True)
class _PromptRoles:
    fit_example_ids: tuple[str, ...]
    tune_example_ids: tuple[str, ...]
    fit_support_tokens: int
    tune_support_tokens: int
    artifact_sha256: str


def _checkerboard_prompt_roles(traces: Sequence[object]) -> _PromptRoles:
    """Choose one fit and one tune prompt per sorted family deterministically."""

    by_family: dict[str, list[object]] = defaultdict(list)
    for trace in traces:
        family = token_v1._identifier(
            getattr(trace, "family_id", None), label="role family_id"
        )
        token_v1._identifier(
            getattr(trace, "example_id", None), label="role example_id"
        )
        by_family[family].append(trace)
    if len(traces) != token_v1._EXPECTED_EXAMPLES or len(by_family) != 8:
        raise ValueError("checkerboard role panel differs")
    fit: list[object] = []
    tune: list[object] = []
    for family_index, family in enumerate(sorted(by_family)):
        values = sorted(by_family[family], key=lambda value: value.example_id)
        if len(values) != 2 or values[0].example_id == values[1].example_id:
            raise ValueError("checkerboard roles require two prompts per family")
        fit_index = 0 if family_index % 2 == 0 else 1
        fit.append(values[fit_index])
        tune.append(values[1 - fit_index])
    fit_ids = tuple(value.example_id for value in fit)
    tune_ids = tuple(value.example_id for value in tune)
    fit_tokens = sum(value.endpoint.supervised_tokens for value in fit)
    tune_tokens = sum(value.endpoint.supervised_tokens for value in tune)
    if (
        set(fit_ids) & set(tune_ids)
        or fit_tokens != _EXPECTED_FIT_SUPPORT_TOKENS
        or tune_tokens != _EXPECTED_TUNE_SUPPORT_TOKENS
    ):
        raise RuntimeError("checkerboard role support accounting differs")
    payload = {
        "schema": "fisher_graph.complete_h4_k64_gain_checkerboard_roles.v1",
        "family_order": tuple(sorted(by_family)),
        "fit_example_ids": fit_ids,
        "tune_example_ids": tune_ids,
        "fit_support_tokens": fit_tokens,
        "tune_support_tokens": tune_tokens,
        "even_family_index_fit_role": "lexicographically_first_prompt",
        "odd_family_index_fit_role": "lexicographically_second_prompt",
        "held_family_excluded_from_its_own_fold_gain_fit_and_tune": True,
    }
    return _PromptRoles(
        fit_example_ids=fit_ids,
        tune_example_ids=tune_ids,
        fit_support_tokens=fit_tokens,
        tune_support_tokens=tune_tokens,
        artifact_sha256=token_v1._domain_sha256(payload, domain=_ROLE_DOMAIN),
    )


class _AuthenticatedCandidateGainProvider(Gemma3L3L4CorrectionProvider):
    """Single-use K64 correction bound to one fold, gain vector, and execution."""

    __slots__ = (
        "site",
        "write_scope",
        "artifact_sha256",
        "stage",
        "gain_kind",
        "fold_artifact_sha256",
        "ordered_directions_sha256",
        "gains_sha256",
        "refit_artifact_sha256",
        "selection_artifact_sha256",
        "alpha_hex",
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
        gain_kind: str,
        fold_artifact_sha256: str,
        ordered_directions_sha256: str,
        gains: Tensor,
        refit_artifact_sha256: str | None,
        selection_artifact_sha256: str | None,
        alpha: float | None,
        model_inputs_sha256: str,
        bridge_binding_sha256: str,
        prefix_artifact_sha256: str,
        base_h4: Tensor,
        support_mask: Tensor,
        correction: Tensor,
    ) -> None:
        if stage not in {"gradient", "tune", "final"}:
            raise ValueError("candidate gain provider stage differs")
        if gain_kind not in {"unit", "interpolated", "selected_refit"}:
            raise ValueError("candidate gain provider gain kind differs")
        gain_values = gains.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if (
            gain_values.shape != (CANDIDATE_GAIN_RANK,)
            or not bool(torch.isfinite(gain_values).all())
            or bool((gain_values < 0.0).any())
            or bool((gain_values > 1.5).any())
        ):
            raise ValueError("candidate gain provider gains differ")
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
            raise ValueError("candidate gain provider tensor geometry differs")
        support = support_mask.detach().to(device="cpu").clone().contiguous()
        delta = correction.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
        if not bool(torch.isfinite(delta).all()) or bool((delta[~support] != 0).any()):
            raise ValueError("candidate gain correction escapes support")
        self.site = token_v1._H4_SITE
        self.write_scope = "complete_h4_causal_support"
        self.stage = stage
        self.gain_kind = gain_kind
        self.fold_artifact_sha256 = _require_sha256(
            fold_artifact_sha256, label="candidate gain fold"
        )
        self.ordered_directions_sha256 = _require_sha256(
            ordered_directions_sha256, label="candidate gain directions"
        )
        self.gains_sha256 = _runtime_tensor_sha256(gain_values)
        self.refit_artifact_sha256 = (
            None
            if refit_artifact_sha256 is None
            else _require_sha256(refit_artifact_sha256, label="candidate gain refit")
        )
        self.selection_artifact_sha256 = (
            None
            if selection_artifact_sha256 is None
            else _require_sha256(
                selection_artifact_sha256, label="candidate gain selection"
            )
        )
        self.alpha_hex = None if alpha is None else float(alpha).hex()
        unit = torch.equal(gain_values, torch.ones_like(gain_values))
        if gain_kind == "unit" and not unit:
            raise ValueError("unit candidate gain provider requires all-one gains")
        if stage == "gradient" and (
            gain_kind != "unit"
            or refit_artifact_sha256 is not None
            or selection_artifact_sha256 is not None
            or alpha is not None
        ):
            raise ValueError("gradient provider semantics differ")
        if stage == "tune" and (
            gain_kind != "interpolated"
            or refit_artifact_sha256 is None
            or selection_artifact_sha256 is not None
            or type(alpha) is not float
            or alpha not in CANDIDATE_GAIN_ALPHAS
        ):
            raise ValueError("tune provider semantics differ")
        if stage == "final" and (
            alpha is not None
            or (
                gain_kind == "unit"
                and (
                    refit_artifact_sha256 is not None
                    or selection_artifact_sha256 is not None
                )
            )
            or (
                gain_kind == "selected_refit"
                and (
                    refit_artifact_sha256 is None
                    or selection_artifact_sha256 is None
                )
            )
            or gain_kind == "interpolated"
        ):
            raise ValueError("final provider semantics differ")
        self.model_inputs_sha256 = _require_sha256(
            model_inputs_sha256, label="candidate gain model inputs"
        )
        self.bridge_binding_sha256 = _require_sha256(
            bridge_binding_sha256, label="candidate gain bridge"
        )
        self.prefix_artifact_sha256 = _require_sha256(
            prefix_artifact_sha256, label="candidate gain prefix"
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
            "schema": "fisher_graph.complete_h4_k64_candidate_gain_provider.v1",
            "rank": CANDIDATE_GAIN_RANK,
            "site": self.site,
            "write_scope": self.write_scope,
            "stage": self.stage,
            "gain_kind": self.gain_kind,
            "fold_artifact_sha256": self.fold_artifact_sha256,
            "ordered_directions_sha256": self.ordered_directions_sha256,
            "gains_sha256": self.gains_sha256,
            "refit_artifact_sha256": self.refit_artifact_sha256,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "alpha_hex": self.alpha_hex,
            "model_inputs_sha256": self.model_inputs_sha256,
            "bridge_binding_sha256": self.bridge_binding_sha256,
            "prefix_artifact_sha256": self.prefix_artifact_sha256,
            "base_h4_sha256": self.base_h4_sha256,
            "support_mask_sha256": self.support_mask_sha256,
            "correction_sha256": self.correction_sha256,
            "correction_semantics": (
                "P_D320_R_plus_gain_scaled_frozen_training_token_fisher_K64_tail"
            ),
            "exact_residual_provider_substitution_used": False,
            "held_native_tail_instantiated": True,
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
            raise RuntimeError("candidate gain provider payload drifted")

    def correction(
        self, prefix: Gemma3L3L4OnePassPrefix, realized_state: Tensor
    ) -> Tensor:
        self.validate_integrity()
        if self._used:
            raise RuntimeError("candidate gain provider cannot be reused")
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
            raise RuntimeError("candidate gain provider reached another execution")
        self._used = True
        return self._correction.to(device=realized_state.device).clone()

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def _endpoint_prompt_receipts(traces: Sequence[object]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            **trace.endpoint.metadata(),
            "model_inputs_sha256": trace.model_inputs_sha256,
            "base_x4_sha256": trace.base_x4_sha256,
            "supervised_indices_sha256": trace.supervised_indices_sha256,
            "supervised_targets_sha256": trace.supervised_targets_sha256,
            "endpoint_support_indices_sha256": trace.endpoint_indices_sha256,
            "endpoint_support_targets_sha256": trace.endpoint_targets_sha256,
            "endpoint_support_supervised_token_count": trace.endpoint.supervised_tokens,
            "native_logits_sha256": trace.native_logits_sha256,
            "endpoint_vjp_artifact_sha256": trace.endpoint_vjp_artifact_sha256,
            "endpoint_execution_artifact_sha256": trace.endpoint_execution_artifact_sha256,
            "endpoint_provider_artifact_sha256": trace.endpoint_provider_artifact_sha256,
            "backward_call_count": trace.backward_call_count,
            "compensation_target_semantics": (
                "native_token_nll_minus_d320_endpoint_token_nll"
            ),
            "compensation_target_sign_used_in_fisher_q2_ordering": False,
            "maximum_future_gradient_abs": trace.maximum_future_gradient_abs,
            "future_gradient_nonzero_count": trace.future_gradient_nonzero_count,
            "causality_receipt_sha256": trace.causality_receipt_sha256,
        }
        for trace in sorted(traces, key=lambda value: value.example_id)
    )


def _authenticate_parent_recollection(
    *,
    parent: Mapping[str, object],
    traces: Sequence[object],
    endpoint_resources: Mapping[str, int],
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> str:
    """Fail before refit unless traces, folds, and collection exactly replay."""

    parent_prompts = parent.get("prompt_receipts")
    parent_folds = parent.get("folds")
    parent_resources = _mapping(parent.get("resources"), label="parent resources")
    if not isinstance(parent_prompts, list) or not isinstance(parent_folds, list):
        raise ValueError("expanded parent recollection receipts differ")
    current_prompts = _endpoint_prompt_receipts(traces)
    current_folds = tuple(fits[family].metadata() for family in sorted(fits))
    resource_keys = (
        "base_forward_count",
        "native_forward_count",
        "endpoint_token_vjp_forward_count",
        "endpoint_token_vjp_backward_call_count",
        "ordinary_supervised_token_count",
        "endpoint_support_supervised_token_count",
        "complete_h4_support_row_count",
        "graph_core_row_count",
        "causal_tail_row_count",
        "graph_core_supervised_token_count",
        "causal_tail_supervised_token_count",
    )
    if (
        _canonical(current_prompts) != _canonical(parent_prompts)
        or _canonical(current_folds) != _canonical(parent_folds)
        or any(
            endpoint_resources.get(key) != parent_resources.get(key)
            for key in resource_keys
        )
    ):
        raise RuntimeError("expanded parent traces/folds/resources did not replay")
    payload = {
        "prompt_receipts": current_prompts,
        "folds": current_folds,
        "collection_resources": tuple(
            (key, endpoint_resources[key]) for key in resource_keys
        ),
        "parent_report_sha256": EXPANDED_PARENT_REPORT_SHA256,
        "replay_completed_before_candidate_refit": True,
    }
    return token_v1._domain_sha256(payload, domain=_RECOLLECTION_DOMAIN)


def _ordered_k64(fit: CompleteH4TailHeldFamilyFit) -> Tensor:
    directions = fit.ordered_basis_rows()[:CANDIDATE_GAIN_RANK].contiguous()
    if directions.shape != (CANDIDATE_GAIN_RANK, token_v1._WIDTH):
        raise RuntimeError("frozen token-Fisher K64 directions differ")
    return directions


def _ordered_k64_relevance(fit: CompleteH4TailHeldFamilyFit) -> Tensor:
    return torch.tensor(
        tuple(
            fit.token_fisher_relevance[index]
            for index in fit.token_fisher_order[:CANDIDATE_GAIN_RANK]
        ),
        dtype=torch.float64,
    ).contiguous()


def _candidate_components(
    trace: object,
    *,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    gains: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    directions = _ordered_k64(fit)
    residual = trace.endpoint.residual_rows
    supported = ((residual @ basis.T) @ basis).contiguous()
    tail = project_complete_h4_tail_rows(residual, basis)
    gain_values = gains.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if torch.equal(
        gain_values,
        torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64),
    ):
        # Preserve the exact parent K64 recipe at the replay/control point.
        prefix = project_complete_h4_tail_prefix(
            tail, fit, rank=CANDIDATE_GAIN_RANK
        )
    else:
        prefix = (((tail @ directions.T) * gain_values) @ directions).contiguous()
    correction_rows = (supported + prefix).contiguous()
    correction = torch.zeros(trace.base_h4.shape, dtype=torch.float64)
    correction[0].index_copy_(0, trace.support_indices, correction_rows)
    return directions, tail, correction_rows, correction


def _authenticate_static_unit_k64_replay(
    *,
    parent: Mapping[str, object],
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> str:
    """Replay every deterministic unit-K64 field before expensive refitting.

    This deliberately reconstructs the cast-once correction rows without a
    model forward.  Candidate logits and their NLLs remain covered by the
    later executed replay, but any basis/order/projection drift now fails
    immediately after parent recollection instead of after 494 backwards.
    """

    raw_parent = parent.get("finite_observation_receipts")
    if not isinstance(raw_parent, list):
        raise ValueError("expanded parent static replay evidence differs")
    parent_k64 = {
        str(row["example_id"]): row
        for row in raw_parent
        if isinstance(row, Mapping) and row.get("rank") == CANDIDATE_GAIN_RANK
    }
    if len(parent_k64) != token_v1._EXPECTED_EXAMPLES:
        raise ValueError("expanded parent static K64 grid differs")
    unit_gains = torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64)
    stable_keys = (
        "example_id",
        "family_id",
        "rank",
        "fold_artifact_sha256",
        "token_score_matrix_sha256",
        "native_mean_nll",
        "d320_mean_nll",
        "endpoint_baseline_mse",
        "endpoint_prediction_mse",
        "candidate_h4_bitwise_native",
        "full_tail_reconstruction_max_abs_error",
        "exact_residual_provider_used",
        "executed_correction_rows_sha256",
    )
    rows: list[dict[str, object]] = []
    mismatches: list[tuple[str, tuple[str, ...]]] = []
    for trace in sorted(traces, key=lambda value: value.example_id):
        fit = fits[trace.family_id]
        support = trace.prefix.complete_h4_causal_support_mask().detach().to(
            device="cpu"
        )
        support_indices = torch.nonzero(
            support[0], as_tuple=False
        ).flatten().to(dtype=torch.int64)
        if not torch.equal(support_indices, trace.support_indices):
            raise RuntimeError("unit K64 static support ordering drifted")
        _directions, _tail, correction_rows, _correction = (
            _candidate_components(
                trace,
                basis=basis,
                fit=fit,
                gains=unit_gains,
            )
        )
        full_scores = complete_h4_tail_gate_scores(
            trace.endpoint, fit.ordered_basis_rows()
        )
        scores = full_scores[:, :CANDIDATE_GAIN_RANK]
        base_rows = (
            trace.base_h4.detach()
            .to(device="cpu")[0]
            .index_select(0, trace.support_indices)
            .contiguous()
        )
        realized_rows = (
            base_rows.to(dtype=torch.float64) + correction_rows
        ).to(dtype=base_rows.dtype)
        actual_rows = (
            realized_rows.to(dtype=torch.float64)
            - base_rows.to(dtype=torch.float64)
        ).contiguous()
        candidate_h4 = trace.base_h4.detach().to(device="cpu").clone()
        candidate_h4[0].index_copy_(0, trace.support_indices, realized_rows)
        prediction = scores.sum(dim=1).contiguous()
        target = trace.endpoint.compensation_target
        row: dict[str, object] = {
            "example_id": trace.example_id,
            "family_id": trace.family_id,
            "rank": CANDIDATE_GAIN_RANK,
            "fold_artifact_sha256": fit.artifact_sha256,
            "token_score_matrix_sha256": _runtime_tensor_sha256(full_scores),
            "native_mean_nll": float(trace.native_token_nll.mean()),
            "d320_mean_nll": float(trace.d320_token_nll.mean()),
            "endpoint_baseline_mse": float(target.square().mean()),
            "endpoint_prediction_mse": float(
                (prediction - target).square().mean()
            ),
            "candidate_h4_bitwise_native": token_v1._bitwise_equal(
                candidate_h4, trace.native_h4
            ),
            "full_tail_reconstruction_max_abs_error": None,
            "exact_residual_provider_used": False,
            "executed_correction_rows_sha256": _runtime_tensor_sha256(
                actual_rows
            ),
        }
        expected = parent_k64.get(trace.example_id)
        if expected is None:
            mismatches.append((trace.example_id, ("missing_parent_row",)))
        else:
            changed = tuple(key for key in stable_keys if row[key] != expected[key])
            if changed:
                mismatches.append((trace.example_id, changed))
        rows.append(row)
    if len(rows) != token_v1._EXPECTED_EXAMPLES or mismatches:
        raise RuntimeError(
            "unit K64 static replay differs from expanded parent: "
            f"{mismatches[:4]}"
        )
    return token_v1._domain_sha256(
        tuple(rows), domain=b"fisher-graph:complete-h4-k64-static-replay:v1\0"
    )


def _fresh_native_teacher(
    *, context: object, trace: object
) -> tuple[Mapping[str, Tensor], Tensor, Tensor, Tensor]:
    model_inputs, indices, targets = _retokenize(
        getattr(context, "tokenize"), trace.example
    )
    if (
        gemma3_l3_l4_shadow_model_inputs_sha256(model_inputs)
        != trace.model_inputs_sha256
        or _runtime_tensor_sha256(indices) != trace.supervised_indices_sha256
        or _runtime_tensor_sha256(targets) != trace.supervised_targets_sha256
    ):
        raise RuntimeError("candidate gain retokenization drifted")
    teacher_logits, native_h4, positions, valid = _native_boundary(
        getattr(context, "adapter"), model_inputs
    )
    if (
        _runtime_tensor_sha256(teacher_logits) != trace.native_logits_sha256
        or not token_v1._bitwise_equal(
            native_h4.detach().to(device="cpu"), trace.native_h4
        )
        or not token_v1._bitwise_equal(positions, trace.prefix.logical_positions)
        or not token_v1._bitwise_equal(valid, trace.prefix.valid_target_mask)
    ):
        raise RuntimeError("candidate gain native boundary drifted")
    return model_inputs, indices, targets, teacher_logits


def _validate_candidate_execution(
    *, trace: object, provider: _AuthenticatedCandidateGainProvider, execution: object
) -> None:
    provider.validate_integrity()
    expected_h4 = trace.base_h4.detach().clone()
    support = provider._support.to(device=expected_h4.device)
    expected_h4[support] = (
        trace.base_h4.detach()[support].to(dtype=torch.float64)
        + provider._correction.to(device=expected_h4.device)[support]
    ).to(dtype=expected_h4.dtype)
    if (
        getattr(execution, "model_forward_count", None) != 1
        or not provider.used
        or execution.model_inputs_sha256 != trace.model_inputs_sha256
        or execution.bridge_binding_sha256 != trace.prefix.bridge_binding_sha256
        or execution.prefix.artifact_sha256 != trace.prefix.artifact_sha256
        or execution.h4_head_sha256 != provider.artifact_sha256
        or _runtime_tensor_sha256(execution.candidate_x4) != trace.base_x4_sha256
        or not token_v1._bitwise_equal(execution.candidate_h4.detach(), expected_h4)
    ):
        raise RuntimeError("candidate gain execution binding differs")


def _endpoint_indices(
    trace: object, indices: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    selected = trace.selected_by_ledger["complete_h4_support"]
    endpoint_indices = indices.index_select(0, selected.to(indices.device))
    endpoint_targets = targets.index_select(0, selected.to(targets.device))
    grid = teacher_kl._canonical_support_grid(
        endpoint_indices.detach().to(device="cpu", dtype=torch.int64)
    )
    if (
        _runtime_tensor_sha256(endpoint_indices) != trace.endpoint_indices_sha256
        or _runtime_tensor_sha256(endpoint_targets) != trace.endpoint_targets_sha256
    ):
        raise RuntimeError("candidate gain endpoint supervision drifted")
    return endpoint_indices, endpoint_targets, grid


def _future_gradient_evidence(
    *,
    trace: object,
    endpoint_indices: Tensor,
    gradient_rows: Tensor,
) -> tuple[float, int]:
    logical = trace.prefix.logical_positions.detach().to(device="cpu")
    supervised_logical = logical[0].index_select(
        0, endpoint_indices.detach().to(device="cpu")
    )
    support_logical = logical[0].index_select(0, trace.support_indices)
    maximum = 0.0
    nonzero = 0
    for token_index in range(int(endpoint_indices.numel())):
        later = support_logical > supervised_logical[token_index]
        if bool(later.any()):
            future = gradient_rows[token_index, later]
            maximum = max(maximum, float(future.abs().max()))
            nonzero += int((future != 0.0).sum())
    if maximum != 0.0 or nonzero != 0:
        raise RuntimeError("candidate teacher-KL VJP leaks into future H4 rows")
    return maximum, nonzero


def _execute_candidate_teacher_kl_vjp(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    gains: Tensor,
    model_inputs: Mapping[str, Tensor],
    teacher_logits: Tensor,
    endpoint_indices: Tensor,
    endpoint_grid: Tensor,
) -> tuple[Tensor, Tensor, dict[str, object], int]:
    directions, tail, _correction_rows, correction = _candidate_components(
        trace, basis=basis, fit=fit, gains=gains
    )
    directions_sha256 = _runtime_tensor_sha256(directions)
    provider = _AuthenticatedCandidateGainProvider(
        stage="gradient",
        gain_kind="unit",
        fold_artifact_sha256=fit.artifact_sha256,
        ordered_directions_sha256=directions_sha256,
        gains=gains,
        refit_artifact_sha256=None,
        selection_artifact_sha256=None,
        alpha=None,
        model_inputs_sha256=trace.model_inputs_sha256,
        bridge_binding_sha256=trace.prefix.bridge_binding_sha256,
        prefix_artifact_sha256=trace.prefix.artifact_sha256,
        base_h4=trace.base_h4,
        support_mask=trace.prefix.complete_h4_causal_support_mask(),
        correction=correction,
    )
    vjp = getattr(context, "bridge").execute_h4_token_teacher_kl_vjps(
        getattr(context, "adapter"),
        model_inputs,
        teacher_logits=teacher_logits,
        supervised_indices=endpoint_grid,
        vjp_chunk_size=token_v1._VJP_CHUNK_SIZE,
        h4_head=provider,
    )
    vjp.validate_integrity()
    _validate_candidate_execution(
        trace=trace, provider=provider, execution=vjp.execution
    )
    if (
        vjp.teacher_logits_sha256 != _runtime_tensor_sha256(teacher_logits)
        or not torch.equal(
            vjp.supervised_indices.detach().to(device="cpu"), endpoint_grid
        )
    ):
        raise RuntimeError("candidate teacher-KL VJP authority differs")
    token_kl = vjp.token_kl_divergences.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    independent = teacher_kl._selected_token_teacher_kl(
        teacher_logits, vjp.execution.logits, endpoint_indices
    )
    if not torch.allclose(token_kl, independent, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("candidate teacher-KL objective authority differs")
    gradient_rows = (
        vjp.h4_gradients.detach()
        .to(device="cpu", dtype=torch.float64)[:, 0]
        .index_select(1, trace.support_indices)
        .contiguous()
    )
    maximum_future, future_nonzero = _future_gradient_evidence(
        trace=trace,
        endpoint_indices=endpoint_indices,
        gradient_rows=gradient_rows,
    )
    scores = contract_candidate_teacher_kl_gain_scores(
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradient_rows,
    )
    evidence = {
        "example_id": trace.example_id,
        "family_id": trace.family_id,
        "held_family_id": fit.held_family_id,
        "fold_artifact_sha256": fit.artifact_sha256,
        "ordered_directions_sha256": directions_sha256,
        "gains_sha256": provider.gains_sha256,
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_artifact_sha256": vjp.execution.artifact_sha256,
        "teacher_kl_vjp_artifact_sha256": vjp.artifact_sha256,
        "teacher_logits_sha256": vjp.teacher_logits_sha256,
        "supervised_grid_sha256": _runtime_tensor_sha256(endpoint_grid),
        "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
        "token_teacher_kl_mean": float(token_kl.mean()),
        "token_gain_gradients_runtime_sha256": _runtime_tensor_sha256(scores),
        "backward_call_count": vjp.backward_call_count,
        "maximum_future_gradient_abs": maximum_future,
        "future_gradient_nonzero_count": future_nonzero,
        "stage": "fit_gradient",
        "candidate_is_realized_unit_gain_k64": True,
        "held_family_used": False,
        "raw_tensors_serialized": False,
    }
    backward_count = vjp.backward_call_count
    del vjp, provider, correction, independent, gradient_rows, tail, directions
    return token_kl, scores, evidence, backward_count


def _collect_candidate_gradient_refits(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    roles: _PromptRoles,
) -> tuple[
    dict[str, CandidateConditionedK64GainRefit],
    tuple[dict[str, object], ...],
    dict[str, int],
]:
    """Collect prompt-major 8x7 unit-K64 VJPs and fit eight folds."""

    by_id = {trace.example_id: trace for trace in traces}
    examples_by_fold: dict[
        str, list[CandidateConditionedK64GainGradientExample]
    ] = {family: [] for family in sorted(fits)}
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
            token_kl, scores, receipt, calls = (
                _execute_candidate_teacher_kl_vjp(
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
            )
            example = CandidateConditionedK64GainGradientExample(
                example_id=trace.example_id,
                family_id=trace.family_id,
                token_gain_gradients=scores,
                token_teacher_kl=token_kl,
            )
            examples_by_fold[held_family].append(example)
            bound_receipt = {
                **receipt,
                "gradient_example_artifact_sha256": example.artifact_sha256,
                "token_gain_gradients_sha256": example.metadata()[
                    "token_gain_gradients_sha256"
                ],
            }
            bound_receipt["receipt_sha256"] = token_v1._domain_sha256(
                bound_receipt, domain=_GRADIENT_RECEIPT_DOMAIN
            )
            receipts.append(bound_receipt)
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
        raise RuntimeError("candidate gradient execution accounting differs")
    refits: dict[str, CandidateConditionedK64GainRefit] = {}
    for held_family in sorted(fits):
        fit = fits[held_family]
        refits[held_family] = fit_candidate_conditioned_k64_gains(
            examples_by_fold[held_family],
            held_family_id=held_family,
            parent_fold_artifact_sha256=fit.artifact_sha256,
            ordered_directions_sha256=_runtime_tensor_sha256(_ordered_k64(fit)),
            ordered_token_fisher_relevance=_ordered_k64_relevance(fit),
        )
    return refits, tuple(receipts), {
        "gradient_native_forward_count": native_forwards,
        "gradient_candidate_vjp_forward_count": candidate_forwards,
        "gradient_candidate_vjp_backward_call_count": backward_calls,
        "gradient_prompt_fold_count": len(receipts),
    }


def _execute_candidate_teacher_kl_forward(
    *,
    context: object,
    trace: object,
    basis: Tensor,
    fit: CompleteH4TailHeldFamilyFit,
    gains: Tensor,
    stage: str,
    gain_kind: str,
    refit_artifact_sha256: str | None,
    selection_artifact_sha256: str | None,
    alpha: float | None,
    model_inputs: Mapping[str, Tensor],
    teacher_logits: Tensor,
    endpoint_indices: Tensor,
) -> tuple[Tensor, object, _AuthenticatedCandidateGainProvider, Tensor]:
    directions, _tail, correction_rows, correction = _candidate_components(
        trace, basis=basis, fit=fit, gains=gains
    )
    provider = _AuthenticatedCandidateGainProvider(
        stage=stage,
        gain_kind=gain_kind,
        fold_artifact_sha256=fit.artifact_sha256,
        ordered_directions_sha256=_runtime_tensor_sha256(directions),
        gains=gains,
        refit_artifact_sha256=refit_artifact_sha256,
        selection_artifact_sha256=selection_artifact_sha256,
        alpha=alpha,
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
    _validate_candidate_execution(
        trace=trace, provider=provider, execution=execution
    )
    token_kl = teacher_kl._selected_token_teacher_kl(
        teacher_logits, execution.logits, endpoint_indices
    )
    return token_kl, execution, provider, correction_rows


def _collect_candidate_tune_selections(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    refits: Mapping[str, CandidateConditionedK64GainRefit],
    roles: _PromptRoles,
) -> tuple[
    dict[str, CandidateConditionedK64GainTuneSelection],
    tuple[dict[str, object], ...],
    dict[str, int],
]:
    """Evaluate the fixed 4-alpha grid on the eight complementary prompts."""

    by_id = {trace.example_id: trace for trace in traces}
    examples_by_fold: dict[str, list[CandidateConditionedK64GainTuneExample]] = {
        family: [] for family in sorted(fits)
    }
    receipts: list[dict[str, object]] = []
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
        for held_family in sorted(fits):
            if held_family == trace.family_id:
                continue
            fit = fits[held_family]
            refit = refits[held_family]
            proposed = refit.proposed_gains_tensor()
            token_kl_by_alpha: list[Tensor] = []
            arm_receipts: list[dict[str, object]] = []
            for alpha in CANDIDATE_GAIN_ALPHAS:
                gains = (1.0 + alpha * (proposed - 1.0)).contiguous()
                token_kl, execution, provider, _correction_rows = (
                    _execute_candidate_teacher_kl_forward(
                        context=context,
                        trace=trace,
                        basis=basis,
                        fit=fit,
                        gains=gains,
                        stage="tune",
                        gain_kind="interpolated",
                        refit_artifact_sha256=refit.artifact_sha256,
                        selection_artifact_sha256=None,
                        alpha=alpha,
                        model_inputs=model_inputs,
                        teacher_logits=teacher_logits,
                        endpoint_indices=endpoint_indices,
                    )
                )
                token_kl_by_alpha.append(token_kl)
                arm_receipts.append(
                    {
                        "example_id": trace.example_id,
                        "family_id": trace.family_id,
                        "held_family_id": held_family,
                        "fold_artifact_sha256": fit.artifact_sha256,
                        "refit_artifact_sha256": refit.artifact_sha256,
                        "ordered_directions_sha256": provider.ordered_directions_sha256,
                        "alpha_hex": alpha.hex(),
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
                        "token_teacher_kl_sha256": _runtime_tensor_sha256(token_kl),
                        "mean_teacher_kl": float(token_kl.mean()),
                        "stage": "tune",
                        "held_family_used": False,
                        "raw_tensors_serialized": False,
                    }
                )
                candidate_forwards += 1
                del execution, provider, gains
            tune_example = CandidateConditionedK64GainTuneExample(
                example_id=trace.example_id,
                family_id=trace.family_id,
                token_teacher_kl_by_alpha=tuple(token_kl_by_alpha),
            )
            examples_by_fold[held_family].append(tune_example)
            for arm_receipt in arm_receipts:
                bound = {
                    **arm_receipt,
                    "tune_example_artifact_sha256": tune_example.artifact_sha256,
                }
                bound["receipt_sha256"] = token_v1._domain_sha256(
                    bound, domain=_TUNE_RECEIPT_DOMAIN
                )
                receipts.append(bound)
            del token_kl_by_alpha, arm_receipts, proposed
        del model_inputs, indices, targets, teacher_logits
    if (
        native_forwards != _EXPECTED_TUNE_NATIVE_FORWARDS
        or candidate_forwards != _EXPECTED_TUNE_CANDIDATE_FORWARDS
        or len(receipts) != _EXPECTED_TUNE_CANDIDATE_FORWARDS
    ):
        raise RuntimeError("candidate tune execution accounting differs")
    selections = {
        held_family: select_candidate_conditioned_k64_gain_alpha(
            refits[held_family], examples_by_fold[held_family]
        )
        for held_family in sorted(fits)
    }
    return selections, tuple(receipts), {
        "tune_native_forward_count": native_forwards,
        "tune_candidate_forward_count": candidate_forwards,
        "tune_prompt_fold_alpha_count": len(receipts),
    }


def _final_candidate_observations(
    *,
    context: object,
    traces: Sequence[object],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
    refits: Mapping[str, CandidateConditionedK64GainRefit],
    selections: Mapping[str, CandidateConditionedK64GainTuneSelection],
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, int],
]:
    """Run native once plus exact unit/refit K64 arms for all sixteen prompts."""

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
            raise RuntimeError("final candidate fold binding differs")
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
                torch.ones(CANDIDATE_GAIN_RANK, dtype=torch.float64),
                "unit",
                None,
                None,
            ),
            (
                "selected_refit_k64",
                selection.selected_gains_tensor(),
                "selected_refit",
                refit.artifact_sha256,
                selection.artifact_sha256,
            ),
        )
        for arm, gains, gain_kind, refit_sha256, selection_sha256 in arms:
            token_kl, execution, provider, _correction_rows = (
                _execute_candidate_teacher_kl_forward(
                    context=context,
                    trace=trace,
                    basis=basis,
                    fit=fit,
                    gains=gains,
                    stage="final",
                    gain_kind=gain_kind,
                    refit_artifact_sha256=refit_sha256,
                    selection_artifact_sha256=selection_sha256,
                    alpha=None,
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
                "rank": CANDIDATE_GAIN_RANK,
                "fold_artifact_sha256": fit.artifact_sha256,
                "refit_artifact_sha256": refit_sha256,
                "selection_artifact_sha256": selection_sha256,
                "selected_alpha_hex": (
                    selection.selected_alpha.hex()
                    if arm == "selected_refit_k64"
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
                "token_score_matrix_sha256": _runtime_tensor_sha256(
                    full_scores
                ),
                "native_mean_nll": float(trace.native_token_nll.mean()),
                "d320_mean_nll": float(trace.d320_token_nll.mean()),
                "candidate_mean_nll": float(candidate_endpoint_nll.mean()),
                "ordinary_candidate_mean_nll": float(candidate_nll.mean()),
                "endpoint_baseline_mse": float(target.square().mean()),
                "endpoint_prediction_mse": float(
                    (prediction - target).square().mean()
                ),
                "candidate_h4_bitwise_native": token_v1._bitwise_equal(
                    execution.candidate_h4.detach().to(device="cpu"),
                    trace.native_h4,
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
        raise RuntimeError("final candidate execution accounting differs")
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
        "selected_refit_k64": ladder._geometry_with_examples(
            geometry_traces,
            executed_rows["selected_refit_k64"],
            candidate_semantics=(
                "actual_cast_once_d320_plus_candidate_conditioned_gain_refit_"
                "training_token_fisher_tail_k64"
            ),
        ),
    }
    return observations, behavior, geometry, {
        "final_native_forward_count": native_forwards,
        "final_candidate_forward_count": candidate_forwards,
        "final_observation_count": len(observations),
    }


def _finite_observation_set_sha256(
    observations: Sequence[Mapping[str, object]],
) -> str:
    if len(observations) != token_v1._EXPECTED_EXAMPLES * len(_ARMS):
        raise ValueError("candidate gain observation count differs")
    identities: set[tuple[str, str]] = set()
    receipts: list[str] = []
    for raw in observations:
        row = dict(raw)
        receipt = row.pop("observation_sha256", None)
        example_id = token_v1._identifier(
            row.get("example_id"), label="candidate observation example_id"
        )
        arm = row.get("arm")
        if arm not in _ARMS or row.get("rank") != CANDIDATE_GAIN_RANK:
            raise ValueError("candidate observation arm/rank differs")
        identity = (example_id, str(arm))
        if identity in identities:
            raise ValueError("candidate observation grid has a duplicate")
        identities.add(identity)
        expected = token_v1._domain_sha256(row, domain=_OBSERVATION_DOMAIN)
        if receipt != expected:
            raise RuntimeError("candidate observation receipt drifted")
        receipts.append(expected)
    if len({example for example, _arm in identities}) != token_v1._EXPECTED_EXAMPLES:
        raise ValueError("candidate observation example grid is incomplete")
    return token_v1._domain_sha256(
        tuple(receipts), domain=_OBSERVATION_SET_DOMAIN
    )


def _v1_observation_view(row: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "example_id",
        "family_id",
        "rank",
        "fold_artifact_sha256",
        "provider_artifact_sha256",
        "execution_artifact_sha256",
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
    return {key: row[key] for key in keys}


def _authenticate_unit_k64_replay(
    *,
    parent: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
    behavior: Mapping[str, Mapping[str, object]],
    geometry: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], str]:
    """Require stable unit-K64 outputs, summary, behavior, and geometry exactly."""

    raw_parent = parent.get("finite_observation_receipts")
    raw_ladder = parent.get("finite_ladder")
    if not isinstance(raw_parent, list) or not isinstance(raw_ladder, list):
        raise ValueError("expanded parent replay evidence differs")
    parent_k64 = {
        str(row["example_id"]): row
        for row in raw_parent
        if isinstance(row, Mapping) and row.get("rank") == 64
    }
    parent_k320 = [
        dict(row)
        for row in raw_parent
        if isinstance(row, Mapping) and row.get("rank") == 320
    ]
    unit = [row for row in observations if row.get("arm") == "unit_k64"]
    stable_keys = (
        "example_id",
        "family_id",
        "rank",
        "fold_artifact_sha256",
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
    mismatches: list[tuple[str, tuple[str, ...]]] = []
    for row in unit:
        example_id = str(row["example_id"])
        expected = parent_k64.get(example_id)
        if expected is None:
            mismatches.append((example_id, ("missing_parent_row",)))
            continue
        changed = tuple(
            key for key in stable_keys if row[key] != expected[key]
        )
        if changed:
            mismatches.append((example_id, changed))
    if (
        len(parent_k64) != token_v1._EXPECTED_EXAMPLES
        or len(unit) != token_v1._EXPECTED_EXAMPLES
        or mismatches
    ):
        raise RuntimeError(
            "unit K64 stable outputs differ from expanded parent: "
            f"{mismatches[:4]}"
        )
    summary_input = [_v1_observation_view(row) for row in unit] + parent_k320
    summaries, _secondary = token_v1._summarize_observations(
        summary_input, ranks=(64, 320)
    )
    unit_summary = next(row for row in summaries if row["tail_rank"] == 64)
    parent_summary = next(
        row for row in raw_ladder if row.get("tail_rank") == 64
    )
    parent_behavior = _mapping(
        parent.get("established_behavioral_fidelity_by_rank"),
        label="parent replay behavior",
    )
    parent_geometry = _mapping(
        parent.get("executed_cast_once_geometry_by_rank"),
        label="parent replay geometry",
    )
    if (
        _canonical(unit_summary) != _canonical(parent_summary)
        or _canonical(behavior["unit_k64"])
        != _canonical(parent_behavior.get("64"))
        or _canonical(geometry["unit_k64"])
        != _canonical(parent_geometry.get("64"))
    ):
        raise RuntimeError("unit K64 summary/behavior/geometry replay differs")
    receipt = token_v1._domain_sha256(
        {
            "stable_unit_observations": tuple(
                {key: row[key] for key in stable_keys} for row in unit
            ),
            "unit_summary": unit_summary,
            "unit_behavior": behavior["unit_k64"],
            "unit_geometry": geometry["unit_k64"],
            "parent_report_sha256": EXPANDED_PARENT_REPORT_SHA256,
        },
        domain=b"fisher-graph:complete-h4-k64-unit-replay:v1\0",
    )
    return unit_summary, receipt


def _arm_summary(
    *,
    arm: str,
    observations: Sequence[Mapping[str, object]],
    parent: Mapping[str, object],
) -> dict[str, object]:
    if arm not in _ARMS:
        raise ValueError("candidate summary arm differs")
    parent_rows = parent.get("finite_observation_receipts")
    if not isinstance(parent_rows, list):
        raise ValueError("parent K320 summary anchor differs")
    selected = [
        _v1_observation_view(row)
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
        arm = str(row["arm"])
        by_arm_family[arm][str(row["family_id"])].append(
            float(row["complete_h4_support_mean_teacher_kl"])
        )
    family_rows: list[dict[str, object]] = []
    for family in sorted(by_arm_family["unit_k64"]):
        unit_values = by_arm_family["unit_k64"][family]
        refit_values = by_arm_family["selected_refit_k64"][family]
        if len(unit_values) != 2 or len(refit_values) != 2:
            raise ValueError("held teacher-KL family observation shape differs")
        unit = math.fsum(unit_values) / len(unit_values)
        refit = math.fsum(refit_values) / len(refit_values)
        family_rows.append(
            {
                "family_id": family,
                "unit_k64_mean_teacher_kl": unit,
                "selected_refit_k64_mean_teacher_kl": refit,
                "absolute_delta_refit_minus_unit": refit - unit,
                "relative_improvement": (
                    (unit - refit)
                    / max(unit, torch.finfo(torch.float64).tiny)
                ),
                "improved": refit < unit,
                "within_five_percent_plus_1e_minus_8": (
                    refit <= 1.05 * unit + 1.0e-8
                ),
            }
        )
    if len(family_rows) != token_v1._EXPECTED_FAMILIES:
        raise ValueError("held teacher-KL family count differs")
    unit_macro = math.fsum(
        float(row["unit_k64_mean_teacher_kl"]) for row in family_rows
    ) / len(family_rows)
    refit_macro = math.fsum(
        float(row["selected_refit_k64_mean_teacher_kl"])
        for row in family_rows
    ) / len(family_rows)
    return {
        "ledger": "complete_h4_support",
        "aggregation": "family_then_prompt_equal",
        "unit_k64_family_macro_mean_teacher_kl": unit_macro,
        "selected_refit_k64_family_macro_mean_teacher_kl": refit_macro,
        "family_macro_absolute_delta_refit_minus_unit": refit_macro - unit_macro,
        "family_macro_relative_improvement": (
            (unit_macro - refit_macro)
            / max(unit_macro, torch.finfo(torch.float64).tiny)
        ),
        "held_family_improvement_count": sum(
            bool(row["improved"]) for row in family_rows
        ),
        "worst_held_family_relative_improvement": min(
            float(row["relative_improvement"]) for row in family_rows
        ),
        "all_held_families_within_five_percent_plus_1e_minus_8": all(
            bool(row["within_five_percent_plus_1e_minus_8"])
            for row in family_rows
        ),
        "families": tuple(family_rows),
    }


def _top1_comparison(
    behavior: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for ledger in _LEDGERS:
        unit = _mapping(
            behavior["unit_k64"][ledger], label=f"unit {ledger} behavior"
        )
        refit = _mapping(
            behavior["selected_refit_k64"][ledger],
            label=f"refit {ledger} behavior",
        )
        unit_aggregate = float(
            _mapping(unit.get("aggregate"), label="unit aggregate").get(
                "top1_agreement_to_source"
            )
        )
        refit_aggregate = float(
            _mapping(refit.get("aggregate"), label="refit aggregate").get(
                "top1_agreement_to_source"
            )
        )
        unit_macro = float(
            _mapping(
                _mapping(unit.get("family_summary"), label="unit family summary").get(
                    "macro"
                ),
                label="unit family macro",
            ).get("top1_agreement_to_source")
        )
        refit_macro = float(
            _mapping(
                _mapping(
                    refit.get("family_summary"), label="refit family summary"
                ).get("macro"),
                label="refit family macro",
            ).get("top1_agreement_to_source")
        )
        rows[ledger] = {
            "unit_aggregate_top1": unit_aggregate,
            "selected_refit_aggregate_top1": refit_aggregate,
            "aggregate_delta_refit_minus_unit": refit_aggregate - unit_aggregate,
            "unit_family_macro_top1": unit_macro,
            "selected_refit_family_macro_top1": refit_macro,
            "family_macro_delta_refit_minus_unit": refit_macro - unit_macro,
            "aggregate_at_least_point_90": refit_aggregate >= 0.90,
            "family_macro_at_least_point_90": refit_macro >= 0.90,
            "aggregate_no_material_regression_vs_unit": (
                refit_aggregate >= unit_aggregate - 0.01
            ),
            "family_macro_no_material_regression_vs_unit": (
                refit_macro >= unit_macro - 0.01
            ),
            "decides_approximate_90_result": ledger != "causal_tail",
        }
    return rows


def _gate_results(
    *,
    behavior: Mapping[str, Mapping[str, object]],
    geometry: Mapping[str, Mapping[str, object]],
    teacher_kl_comparison: Mapping[str, object],
    top1_comparison: Mapping[str, object],
    selections: Mapping[str, CandidateConditionedK64GainTuneSelection],
) -> tuple[dict[str, object], dict[str, object]]:
    strict: dict[str, object] = {}
    for arm in _ARMS:
        ledger_pass = {
            ledger: bool(
                _mapping(
                    _mapping(
                        behavior[arm][ledger], label=f"{arm} {ledger} behavior"
                    ).get("gates"),
                    label=f"{arm} {ledger} gates",
                ).get("passed")
            )
            for ledger in _LEDGERS
        }
        geometry_pass = bool(
            _mapping(
                _mapping(geometry[arm], label=f"{arm} geometry").get("gates"),
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
    positive_alpha_count = sum(
        selection.selected_alpha > 0.0 for selection in selections.values()
    )
    approximate_gates: dict[str, bool] = {
        "family_macro_complete_h4_support_teacher_kl_improves_at_least_2pct": (
            float(teacher_kl_comparison["family_macro_relative_improvement"])
            >= 0.02
        ),
        "held_family_teacher_kl_improvement_count_at_least_6_of_8": (
            int(teacher_kl_comparison["held_family_improvement_count"]) >= 6
        ),
        "worst_held_family_teacher_kl_regression_at_most_5pct_plus_1e_minus_8": bool(
            teacher_kl_comparison[
                "all_held_families_within_five_percent_plus_1e_minus_8"
            ]
        ),
        "folds_selecting_positive_alpha_at_least_6_of_8": (
            positive_alpha_count >= 6
        ),
    }
    for ledger in ("ordinary", "complete_h4_support", "graph_core"):
        row = _mapping(top1_comparison[ledger], label=f"{ledger} top1 comparison")
        approximate_gates[f"{ledger}_aggregate_top1_at_least_point_90"] = bool(
            row["aggregate_at_least_point_90"]
        )
        approximate_gates[f"{ledger}_family_macro_top1_at_least_point_90"] = bool(
            row["family_macro_at_least_point_90"]
        )
        approximate_gates[
            f"{ledger}_aggregate_top1_no_material_regression_vs_unit"
        ] = bool(row["aggregate_no_material_regression_vs_unit"])
        approximate_gates[
            f"{ledger}_family_macro_top1_no_material_regression_vs_unit"
        ] = bool(row["family_macro_no_material_regression_vs_unit"])
    approximate: dict[str, object] = {
        "gates": tuple(sorted(approximate_gates.items())),
        "passed": all(approximate_gates.values()),
        "positive_alpha_fold_count": positive_alpha_count,
        "positive_alpha_required_count": 6,
        "top1_threshold": 0.90,
        "top1_no_material_regression_tolerance": 0.01,
        "teacher_kl_required_relative_improvement": 0.02,
        "teacher_kl_worst_family_regression_cap": (
            "candidate_le_1.05_times_unit_plus_1e_minus_8"
        ),
        "causal_tail_reported_but_excluded_from_approximate_decision": True,
        "causal_tail_exclusion_reason": (
            "only_13_supervised_tokens_in_the_frozen_panel"
        ),
        "geometry_reported_but_excluded_from_refit_hypothesis_classification": True,
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
        raise ValueError("candidate gain receipt-set count differs")
    values: list[str] = []
    identities: set[tuple[object, ...]] = set()
    for raw in receipts:
        row = dict(raw)
        receipt = row.pop("receipt_sha256", None)
        expected = token_v1._domain_sha256(row, domain=receipt_domain)
        if receipt != expected:
            raise RuntimeError("candidate gain receipt drifted")
        identity = (
            row.get("held_family_id"),
            row.get("example_id"),
            row.get("alpha_hex"),
        )
        if identity in identities:
            raise ValueError("candidate gain receipt set has a duplicate")
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
        or total_forwards != 392
        or total_backwards != 494
    ):
        raise RuntimeError("candidate gain full resource accounting differs")
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
    tune_row_executions = 7 * len(CANDIDATE_GAIN_ALPHAS) * tune_rows
    final_row_executions = 2 * support_rows
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
        "peak_simultaneously_retained_candidate_teacher_kl_vjp_bank_count": 1,
        "raw_candidate_vjp_banks_retained_in_report": False,
        "serving_learned_parameter_count": "not_applicable_no_serving_artifact",
        "serving_logical_macs_per_token": "not_applicable_no_serving_artifact",
    }


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


@dataclass(slots=True)
class _CandidatePhaseResults:
    recollection_receipt: str
    static_unit_replay_receipt: str
    roles: _PromptRoles
    refits: Mapping[str, CandidateConditionedK64GainRefit]
    gradient_receipts: tuple[dict[str, object], ...]
    gradient_resources: Mapping[str, int]
    selections: Mapping[str, CandidateConditionedK64GainTuneSelection]
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
    traces: Sequence[object],
    endpoint_resources: Mapping[str, int],
    basis: Tensor,
    fits: Mapping[str, CompleteH4TailHeldFamilyFit],
) -> _CandidatePhaseResults:
    """Execute replay, per-fold-held fit/tune, then held-fold final phases."""

    recollection_receipt = _authenticate_parent_recollection(
        parent=parent,
        traces=traces,
        endpoint_resources=endpoint_resources,
        fits=fits,
    )
    static_unit_replay_receipt = _authenticate_static_unit_k64_replay(
        parent=parent,
        traces=traces,
        basis=basis,
        fits=fits,
    )
    roles = _checkerboard_prompt_roles(traces)
    refits, gradient_receipts, gradient_resources = (
        _collect_candidate_gradient_refits(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            roles=roles,
        )
    )
    selections, tune_receipts, tune_resources = (
        _collect_candidate_tune_selections(
            context=context,
            traces=traces,
            basis=basis,
            fits=fits,
            refits=refits,
            roles=roles,
        )
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
    unit_summary, unit_replay_receipt = _authenticate_unit_k64_replay(
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


def run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic(
    *,
    expanded_parent_report_path: Path | str = DEFAULT_EXPANDED_PARENT_REPORT,
    materialization_report_path: Path | str = DEFAULT_MATERIALIZATION_REPORT,
    transfer_report_path: Path | str = DEFAULT_TRANSFER_REPORT,
    basis_sidecar_path: Path | str | None = None,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the pinned A16 K64 candidate-conditioned gain-refit diagnostic."""

    destination = token_v1._validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite candidate gain-refit report")
    parent = _load_expanded_parent(expanded_parent_report_path)
    materialization = token_v1._load_pinned_report(
        materialization_report_path,
        expected_file_sha256=token_v1.MATERIALIZATION_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.MATERIALIZATION_REPORT_SHA256,
        label="candidate gain rank320 materialization",
    )
    transfer = token_v1._load_pinned_report(
        transfer_report_path,
        expected_file_sha256=token_v1.TRANSFER_REPORT_FILE_SHA256,
        expected_report_sha256=token_v1.TRANSFER_REPORT_SHA256,
        label="candidate gain rank320 transfer",
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
            raise RuntimeError("candidate gain A16 panel shape differs")
        fits = {
            family: fit_complete_h4_tail_held_family(
                (trace.endpoint for trace in traces),
                supported_basis=basis,
                held_family_id=family,
            )
            for family in families
        }
        # This exact replay is deliberately ordered before roles, candidate
        # gradients, refits, or tune evidence can influence the experiment.
        phases = _execute_candidate_phases(
            context=context,
            parent=parent,
            traces=traces,
            endpoint_resources=endpoint_resources,
            basis=basis,
            fits=fits,
        )
        context.validate_immutable_inputs()
    finally:
        context.close()

    recollection_receipt = phases.recollection_receipt
    static_unit_replay_receipt = phases.static_unit_replay_receipt
    roles = phases.roles
    refits = phases.refits
    gradient_receipts = phases.gradient_receipts
    gradient_resources = phases.gradient_resources
    selections = phases.selections
    tune_receipts = phases.tune_receipts
    tune_resources = phases.tune_resources
    observations = phases.observations
    observation_set_sha256 = phases.observation_set_sha256
    behavior = phases.behavior
    geometry = phases.geometry
    final_resources = phases.final_resources
    unit_summary = phases.unit_summary
    unit_replay_receipt = phases.unit_replay_receipt
    selected_summary = _arm_summary(
        arm="selected_refit_k64", observations=observations, parent=parent
    )
    teacher_kl_comparison = _teacher_kl_comparison(observations)
    top1_comparison = _top1_comparison(behavior)
    strict_results, approximate_results = _gate_results(
        behavior=behavior,
        geometry=geometry,
        teacher_kl_comparison=teacher_kl_comparison,
        top1_comparison=top1_comparison,
        selections=selections,
    )
    gradient_receipt_set_sha256 = _receipt_set_sha256(
        gradient_receipts,
        expected_count=_EXPECTED_GRADIENT_CANDIDATE_FORWARDS,
        receipt_domain=_GRADIENT_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-gradient-set:v1\0",
    )
    tune_receipt_set_sha256 = _receipt_set_sha256(
        tune_receipts,
        expected_count=_EXPECTED_TUNE_CANDIDATE_FORWARDS,
        receipt_domain=_TUNE_RECEIPT_DOMAIN,
        set_domain=b"fisher-graph:complete-h4-k64-gain-tune-set:v1\0",
    )
    resources = _resource_accounting(
        traces=traces,
        roles=roles,
        endpoint_resources=endpoint_resources,
        gradient_resources=gradient_resources,
        tune_resources=tune_resources,
        final_resources=final_resources,
    )
    causality_passed = all(
        trace.maximum_future_gradient_abs == 0.0
        and trace.future_gradient_nonzero_count == 0
        for trace in traces
    ) and all(
        receipt["maximum_future_gradient_abs"] == 0.0
        and receipt["future_gradient_nonzero_count"] == 0
        for receipt in gradient_receipts
    )
    integrity_gates = {
        "expanded_v2_parent_authenticated": True,
        "expanded_parent_k320_exact_and_pass_authenticated_without_reexecution": True,
        "parent_endpoint_traces_folds_and_resources_recollected_exactly_before_refit": True,
        "checkerboard_fit_and_tune_roles_are_disjoint_with_398_and_405_support_tokens": (
            roles.fit_support_tokens == _EXPECTED_FIT_SUPPORT_TOKENS
            and roles.tune_support_tokens == _EXPECTED_TUNE_SUPPORT_TOKENS
            and set(roles.fit_example_ids).isdisjoint(roles.tune_example_ids)
        ),
        "every_gradient_and_tune_receipt_excludes_its_held_family": all(
            receipt["family_id"] != receipt["held_family_id"]
            for receipt in (*gradient_receipts, *tune_receipts)
        ),
        "all_parent_and_candidate_teacher_kl_vjps_have_zero_future_gradient": (
            causality_passed
        ),
        "unit_k64_stable_outputs_summary_behavior_and_geometry_replay_exactly": True,
        "unit_k64_static_projection_and_cast_rows_replay_before_refit": True,
        "exact_model_forward_count_is_392": resources["total_model_forward_count"]
        == 392,
        "exact_backward_call_count_is_494": resources["total_backward_call_count"]
        == 494,
    }
    approximate_passed = bool(approximate_results["passed"])
    primary_gates = {
        **integrity_gates,
        "candidate_conditioned_refit_clears_approximate_90_useful_refit_gates": (
            approximate_passed
        ),
    }
    role_receipt = {
        "artifact_sha256": roles.artifact_sha256,
        "fit_example_ids": roles.fit_example_ids,
        "tune_example_ids": roles.tune_example_ids,
        "fit_support_supervised_token_count": roles.fit_support_tokens,
        "tune_support_supervised_token_count": roles.tune_support_tokens,
        "family_order": families,
        "even_family_index_fit_role": "lexicographically_first_prompt",
        "odd_family_index_fit_role": "lexicographically_second_prompt",
        "tune_role": "complementary_prompt_within_family",
        "held_family_excluded_from_its_own_fold_gain_fit_and_tune": True,
    }
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "artifact": {"file": str(destination), "committable": False},
        "protocol": {
            "panel": "reused_calibration_a_fit16_same_a_hypothesis_use_only",
            "parent_schema": expanded._SCHEMA,
            "parent_classification": (
                "adaptive_same_a_smallest_tail_rank_256_cleared_established_gates"
            ),
            "frozen_tail_rank": CANDIDATE_GAIN_RANK,
            "frozen_tail_basis_and_order": "whole_family_lofo_token_fisher_v1",
            "initial_gains": "all_ones",
            "candidate_gradient_point": "realized_unit_gain_k64",
            "candidate_gradient_objective": "token_teacher_KL_native_to_candidate",
            "candidate_h4_vjp_semantics": (
                "exact_with_respect_to_realized_post_cast_h4_state"
            ),
            "gain_pullback_cast_semantics": (
                "continuous_local_interpretation_of_final_h4_float_cast"
            ),
            "analytic_gain_pullback_is_finite_displacement_authority": False,
            "finite_alpha_executed_teacher_kl_is_selection_authority": True,
            "candidate_vjp_chunk_size": token_v1._VJP_CHUNK_SIZE,
            "candidate_vjp_chunk_size_policy": (
                "bounded_memory_chunk8_authoritative_not_single_batch_chunk128"
            ),
            "fit_objective": "one_half_expected_squared_token_teacher_KL",
            "fit_method": "one_step_damped_residual_Gauss_Newton",
            "not_claimed_as": "mean_KL_natural_gradient_or_exact_GGN",
            "fit_aggregation": "family_then_prompt_then_token_equal",
            "tune_alpha_grid": CANDIDATE_GAIN_ALPHAS,
            "tune_selects_largest_eligible_positive_alpha_else_zero": True,
            "final_arms": _ARMS,
            "final_observation_grid": "16_prompts_times_2_arms",
            "finite_shadow_ledgers": {
                "ordinary": token_v1._EXPECTED_ORDINARY_TOKENS,
                "complete_h4_support": token_v1._EXPECTED_SUPPORT_TOKENS,
                "graph_core": token_v1._EXPECTED_GRAPH_CORE_TOKENS,
                "causal_tail": token_v1._EXPECTED_CAUSAL_TAIL_TOKENS,
            },
            "k320_reexecuted": False,
            "geometry_decides_refit_hypothesis_classification": False,
            "causal_tail_decides_approximate_90_classification": False,
        },
        "expanded_parent_binding": {
            "file": str(expanded_parent_report_path),
            "file_sha256": EXPANDED_PARENT_REPORT_FILE_SHA256,
            "report_sha256": EXPANDED_PARENT_REPORT_SHA256,
            "schema": parent.get("schema"),
            "classification": parent.get("classification"),
            "rank_decisions": parent.get("fidelity_and_geometry_pass_by_rank"),
            "smallest_passing_subsentinel_rank": 256,
            "k320_authenticated_exact_and_pass": True,
            "k320_reexecuted": False,
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
            "parent_recollection_receipt_sha256": recollection_receipt,
            "unit_k64_static_replay_receipt_sha256": static_unit_replay_receipt,
            "unit_k64_replay_receipt_sha256": unit_replay_receipt,
        },
        "prompt_role_receipt": role_receipt,
        "folds": tuple(fits[family].metadata() for family in families),
        "prompt_receipts": _endpoint_prompt_receipts(traces),
        "candidate_gain_refits": tuple(
            refits[family].metadata() for family in families
        ),
        "candidate_gain_tune_selections": tuple(
            selections[family].metadata() for family in families
        ),
        "candidate_gradient_receipts": gradient_receipts,
        "candidate_gradient_receipt_set_sha256": gradient_receipt_set_sha256,
        "candidate_tune_receipts": tune_receipts,
        "candidate_tune_receipt_set_sha256": tune_receipt_set_sha256,
        "finite_arm_summaries": (unit_summary, selected_summary),
        "complete_h4_support_teacher_kl_comparison": teacher_kl_comparison,
        "top1_comparison_with_exact_deltas": top1_comparison,
        "established_behavioral_fidelity_by_arm": behavior,
        "executed_cast_once_geometry_by_arm": geometry,
        "strict_established_gate_results_by_arm": strict_results,
        "approximate_90_useful_refit_gate_results": approximate_results,
        "finite_observation_receipts": tuple(observations),
        "finite_observation_set_sha256": observation_set_sha256,
        "primary_gate_results": tuple(sorted(primary_gates.items())),
        "passed": all(primary_gates.values()),
        "classification": (
            "candidate_conditioned_k64_gain_refit_approximate_90_supported_same_a"
            if all(primary_gates.values())
            else "candidate_conditioned_k64_gain_refit_not_supported_same_a"
        ),
        "resources": resources,
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "expanded_parent_outcome_was_inspected": True,
            "native_teacher_logits_and_held_native_tails_used": True,
            "tail_basis_and_order_are_family_disjoint": True,
            "gain_fit_and_tune_prompt_roles_are_disjoint": True,
            "gain_fit_and_tune_both_exclude_the_held_family": True,
            "gain_fit_and_tune_are_mutually_family_disjoint": False,
            "frozen_d320_contains_same_a_held_family_information": True,
            "end_to_end_candidate_family_disjoint": False,
            "fresh_confirmation_panel_opened": False,
            "geometry_is_report_only_for_refit_hypothesis_classification": True,
            "strict_full_behavior_plus_geometry_result_is_still_reported": True,
            "identically_trained_scalar_gain_control_run": False,
            "mode_specific_value_over_scalar_gain_is_established": False,
            "gain_vectors_are_eight_fold_specific_lofo_vectors": True,
            "one_global_or_deployable_gain_executor_established": False,
            "fixed_seed_gain_permutation_control_run": False,
            "mechanism_gate_present": False,
            "missing_scalar_and_permutation_controls_block_mode_specific_claim": True,
            "candidate_serving_authorized": False,
            "compression_claim": False,
            "speed_or_latency_claim": False,
            "deployment_claim": False,
            "next_rung_only_if_approximate_supported": (
                "freeze_recipe_then_family_disjoint_shadow_and_NLL_confirmation"
            ),
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
            "Run the authenticated A16 K64 candidate-conditioned gain-refit rung."
        )
    )
    parser.add_argument(
        "--expanded-parent-report",
        type=Path,
        default=DEFAULT_EXPANDED_PARENT_REPORT,
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--transfer-report", type=Path, default=DEFAULT_TRANSFER_REPORT)
    parser.add_argument("--basis-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic(
        expanded_parent_report_path=args.expanded_parent_report,
        materialization_report_path=args.materialization_report,
        transfer_report_path=args.transfer_report,
        basis_sidecar_path=args.basis_sidecar,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"report: {report['artifact']['file']}")  # type: ignore[index]
    print(f"report sha256: {report['report_sha256']}")
    print(f"classification: {report['classification']}")


if __name__ == "__main__":
    main()
