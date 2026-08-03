"""V20d fit-only natural-response smoke on the frozen V20c pair.

V20c found a useful but just-subthreshold fixed signed-log response on one
post-hoc reed/sundial diagnostic pair.  V20d keeps that exact development-only
scope and learns only three response weights from the six-family complement.
Signed-log and parameter-matched linear laws are fit independently from their
own exact full-suffix KL VJPs, family-equal empirical Fishers, one damped
natural direction, and the frozen finite alpha ladder.  Both fits and every
held control are frozen before either reciprocal held capability can exist.

This remains analysis, not a serving or compression artifact.  The report is
write-once 0600 scalar/hash-only JSON; learned weights are three explicit
scalars plus their provider-domain tensor hash, and no provider sidecar or raw
model tensor is serialized.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import complete_h4_fisher_natural_response as _core
from .complete_h4_autonomous_residual import _tensor_sha256 as _provider_tensor_sha256
from .complete_h4_fisher_conditional_pedal import _training_parent_modal
from .complete_h4_fisher_continuous_transfer import (
    AutonomousCompleteH4FisherContinuousTransferProvider,
    build_autonomous_complete_h4_fisher_continuous_constant_control,
    build_autonomous_complete_h4_fisher_continuous_transfer,
    fisher_continuous_bilinear_box_max_abs,
    fisher_continuous_bilinear_corner_values,
    fisher_continuous_transfer_modal_terms,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_continuous_response_smoke as _v20c
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as _v20b
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_natural_response_smoke",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V20C_OUTPUT = _v20c.DEFAULT_OUTPUT
_V20C_LOGICAL_SHA256 = (
    "fb744e3d5fdcfa81bf455ceb29b1aec4721dea9fc8bd0ccc247f7d358038d831"
)
_V20C_FILE_SHA256 = (
    "485c7997e06aa671e25aa70e8e7827a1ad1f66b03f8a8e0452c5eb9edadd658f"
)
_V20C_CLASSIFICATION = "fixed_continuous_response_pair_smoke_failed"

DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-natural-response-pair-smoke-"
    "r16-k256-a-fit16-dev-v20d.json"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_natural_response_smoke.v20d"
_FORMAT_VERSION = 20
_REPORT_DOMAIN = b"fisher-graph:natural-response-smoke-report:v20d\0"
_SOURCE_DOMAIN = b"fisher-graph:natural-response-smoke-source:v20d\0"
_GRADIENT_EVIDENCE_DOMAIN = b"fisher-graph:natural-response-gradient:v20d\0"
_CANDIDATE_EVIDENCE_DOMAIN = b"fisher-graph:natural-response-candidate:v20d\0"
_FIT_EVIDENCE_DOMAIN = b"fisher-graph:natural-response-fit-evidence:v20d\0"
_FIT_EXECUTION_DOMAIN = b"fisher-graph:natural-response-fit-execution:v20d\0"
_HELD_EXECUTION_DOMAIN = b"fisher-graph:natural-response-held-execution:v20d\0"
_PROVIDER_RECEIPT_DOMAIN = b"fisher-graph:natural-response-provider:v20d\0"
_PROVIDER_BUNDLE_DOMAIN = b"fisher-graph:natural-response-provider-bundle:v20d\0"
_ROLE_EVIDENCE_DOMAIN = b"fisher-graph:natural-response-role-evidence:v20d\0"
_QUALIFICATION_DOMAIN = b"fisher-graph:natural-response-qualification:v20d\0"

_INITIAL_WEIGHT = (0.0, 1.0, 0.0)
_ALPHAS = (0.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0)
_FIT_LAWS = ("signed_log", "linear")
_HELD_ARMS = (
    "base",
    "constant_plus_one",
    "fixed_signed_log",
    "fixed_linear",
    "learned_signed_log",
    "learned_linear",
    "learned_signed_log_sign_flip",
)
_PROMPTS_PER_FAMILY = 2
_FIT_FAMILY_COUNT = 6
_FAMILY_COUNT = 8
_RANK = 256
_CONDITIONAL_RANK = 16

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20d_three_parameter_family_equal_empirical_fisher_response_fit",
    "scientific_status": "development_only_post_hoc_reused_a16_pair",
    "prerequisite": "exact_immutable_failed_v20c_pair_smoke",
    "excluded_pair": _v20c._FROZEN_EXCLUDED,
    "fit_family_count": _FIT_FAMILY_COUNT,
    "fit_prompt_count": _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY,
    "initial_response_weight": _INITIAL_WEIGHT,
    "response_features": "c1_c2_c1_times_c2",
    "laws_fit_independently": _FIT_LAWS,
    "gradient": "exact_full_suffix_teacher_KL_VJP_then_local_response_contraction",
    "fisher": "family_equal_empirical_outer_product_of_per_example_gradients",
    "direction": "single_damped_natural_descent_direction",
    "global_feasibility": "four_corner_bilinear_box_bound_at_most_one",
    "proposal_projection": "radial_to_global_bilinear_box",
    "alpha_ladder": _ALPHAS,
    "alpha_zero_execution": "reused_exactly_from_initial_VJP_execution",
    "selection": "fit_only_family_equal_exact_objective_then_smaller_alpha",
    "held_arms": _HELD_ARMS,
    "held_barrier": "both_fits_and_all_seven_providers_frozen_before_capability",
    "provider_sidecar_or_raw_tensor": False,
    "serving_compression_speed_or_fresh_fidelity_claim": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(
    _FIXED_PROTOCOL, domain=_SOURCE_DOMAIN
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} mapping is missing")
    return value


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} sequence is missing")
    return tuple(value)


def _sha(value: object, *, label: str) -> str:
    return _v19._sha256_identifier(value, label=label)


def _identifier(value: object, *, label: str) -> str:
    return _v14._identifier(value, label=label)


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or not _v20b._is_under_local_runs(output):
        raise ValueError("V20d output must be JSON under .local-runs")
    if _v20b._same_destination(output, _V20C_OUTPUT):
        raise ValueError("V20d must preserve the immutable V20c report")
    return output


def _load_authenticated_v20c_source(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Authenticate V20c and its frozen V20b pair before model creation."""

    _v20b._secure_stat(_V20C_OUTPUT, label="pinned V20c report")
    if _v14._file_sha256(_V20C_OUTPUT) != _V20C_FILE_SHA256:
        raise RuntimeError("pinned V20c report file hash drifted")
    report = _v20c._load_existing_report(_V20C_OUTPUT)
    if (
        report.get("report_sha256") != _V20C_LOGICAL_SHA256
        or report.get("classification") != _V20C_CLASSIFICATION
        or report.get("passed") is not False
        or report.get("next_full_reused_panel_screen_authorized") is not False
        or report.get("fresh_family_disjoint_claim_authorized") is not False
        or report.get("serving_authorized") is not False
        or report.get("compression_claim") is not False
        or report.get("candidate") is not None
        or report.get("provider_sidecar") is not None
    ):
        raise RuntimeError("pinned V20c scientific authority differs")
    logical = dict(report)
    logical.pop("report_sha256", None)
    if _v14._sha256(logical, domain=_v20c._REPORT_DOMAIN) != _V20C_LOGICAL_SHA256:
        raise RuntimeError("pinned V20c logical hash drifted")
    roles = tuple(
        _mapping(item, label="V20c role")
        for item in _sequence(report.get("roles"), label="V20c roles")
    )
    selected = {
        int(role["law_receipt"]["coordinate_statistics"]["selected_coordinate_index"])
        for role in roles
    }
    if selected != {1}:
        raise RuntimeError("pinned V20c did not freeze coordinate two")
    source_payload = {
        "path": _V20C_OUTPUT.as_posix(),
        "report_logical_sha256": _V20C_LOGICAL_SHA256,
        "report_file_sha256": _V20C_FILE_SHA256,
        "classification": _V20C_CLASSIFICATION,
        "passed": False,
        "selected_coordinate_index": 1,
        "v20b_pair_key": report["shared_fit_receipt"]["fit_key"],
        "v20b_pair_fragment_sha256": report["source"]["pair_fragment_sha256"],
        "base_provider_artifact_sha256": report["shared_fit_receipt"][
            "base_provider_artifact_sha256"
        ],
        "proposal_provider_artifact_sha256": report["shared_fit_receipt"][
            "proposal_provider_artifact_sha256"
        ],
        "authenticated_before_model_work": True,
    }
    source = {
        **source_payload,
        "artifact_sha256": _v14._sha256(source_payload, domain=_SOURCE_DOMAIN),
    }
    _v14._scalar_report(source)
    _source_v20c, _v20b_report, fragment = _v20c._load_authenticated_v20b_source()
    if (
        _v14._canonical_json_bytes(fragment.get("shared_fit_receipt"))
        != _v14._canonical_json_bytes(report.get("shared_fit_receipt"))
        or _v14._canonical_json_bytes(fragment.get("fit_training_evidence"))
        != _v14._canonical_json_bytes(report.get("fit_training_evidence"))
    ):
        raise RuntimeError("V20c endpoint evidence differs from frozen V20b pair")
    return source, dict(report), dict(fragment)


def _source_sha256s(source: Mapping[str, object]) -> dict[str, str]:
    """Return the exact V20c/V20b hash lineage consumed by both fits."""

    return {
        "v20c_report_logical_sha256": _sha(
            source.get("report_logical_sha256"), label="V20c logical report"
        ),
        "v20c_report_file_sha256": _sha(
            source.get("report_file_sha256"), label="V20c report file"
        ),
        "v20c_source_artifact_sha256": _sha(
            source.get("artifact_sha256"), label="V20c source receipt"
        ),
        "v20b_pair_fragment_sha256": _sha(
            source.get("v20b_pair_fragment_sha256"), label="V20b pair fragment"
        ),
    }


def _initial_weight_tensor() -> Tensor:
    return torch.tensor(_INITIAL_WEIGHT, dtype=torch.float64)


def _weight_tensor(value: Sequence[float]) -> Tensor:
    weight = torch.tensor(tuple(float(item) for item in value), dtype=torch.float64)
    if weight.shape != (3,) or not bool(torch.isfinite(weight).all()):
        raise ValueError("V20d response weight geometry differs")
    return weight


def _build_response_provider(
    workspace: object,
    *,
    law: str,
    response_weight: Sequence[float] | Tensor,
    polarity: int,
    evidence_sha256: str,
) -> AutonomousCompleteH4FisherContinuousTransferProvider:
    weight = (
        response_weight
        if isinstance(response_weight, Tensor)
        else _weight_tensor(response_weight)
    )
    return build_autonomous_complete_h4_fisher_continuous_transfer(
        workspace.base_provider,
        workspace.proposal_provider,
        response_law=law,
        response_source="direct",
        response_weight=weight,
        polarity=polarity,
        transfer_protocol_sha256=_core.NATURAL_RESPONSE_PROTOCOL_SHA256,
        transfer_evidence_sha256=evidence_sha256,
        signed_log_kappa=9.0,
    )


def _local_response_weight_gradient(
    provider: AutonomousCompleteH4FisherContinuousTransferProvider,
    sequence: object,
    h4_gradient: Tensor,
) -> Tensor:
    """Contract exact suffix dKL/dH4 through only the 3D response weight."""

    if (
        not isinstance(h4_gradient, Tensor)
        or h4_gradient.shape != (1, *sequence.base_h4.shape)
        or not h4_gradient.is_floating_point()
        or not bool(torch.isfinite(h4_gradient).all())
    ):
        raise ValueError("V20d suffix H4 gradient geometry differs")
    parent = _training_parent_modal(provider.parent_provider, sequence)
    coordinates = provider.bounded_coordinates(parent)
    weight = provider.response_weight.detach().clone().requires_grad_(True)
    delta = fisher_continuous_transfer_modal_terms(
        parent,
        coordinates,
        provider.base_provider.direction_left,
        provider.base_provider.direction_right,
        provider.proposal_provider.direction_left,
        provider.proposal_provider.direction_right,
        provider.base_provider.pedal_weight,
        provider.base_provider.pedal_bias,
        provider.proposal_provider.pedal_weight,
        provider.proposal_provider.pedal_bias,
        weight,
        response_source="direct",
        response_law=provider.response_law,
        polarity=provider.polarity,
        signed_log_kappa=provider.signed_log_kappa,
        trust_fraction=provider.trust_fraction,
    )[-1]
    decoded = delta @ provider.parent_provider.output_decoder
    suffix = h4_gradient[0].detach().to(device="cpu", dtype=torch.float64)
    surrogate = (decoded * suffix).sum()
    gradient = torch.autograd.grad(
        surrogate, weight, retain_graph=False, create_graph=False
    )[0]
    if gradient.shape != (3,) or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("V20d local response gradient became invalid")
    return gradient.detach().to(device="cpu", dtype=torch.float64).contiguous()


@dataclass(slots=True)
class _InitialFitEvidence:
    provider: AutonomousCompleteH4FisherContinuousTransferProvider
    capability: object
    training_records: tuple[object, ...]
    gradients_by_family: dict[str, dict[str, tuple[float, float, float]]]
    objectives_by_family: dict[str, dict[str, float]]
    h4_sha256s: dict[str, str]
    logits_sha256s: dict[str, str]
    execution_sha256s: dict[str, str]
    gradient_evidence: dict[str, object]


@dataclass(slots=True)
class _CandidateLive:
    provider: AutonomousCompleteH4FisherContinuousTransferProvider
    evidence: dict[str, object]


@dataclass(slots=True)
class _FitLive:
    law: str
    initial: _InitialFitEvidence
    direction_receipt: dict[str, object]
    candidates: tuple[_CandidateLive, ...]
    candidate_receipts: tuple[dict[str, object], ...]
    fit_receipt: dict[str, object]
    fit_evidence: dict[str, object]
    selected_provider: AutonomousCompleteH4FisherContinuousTransferProvider | None


def _fit_execution_sha256(
    *,
    law: str,
    phase: str,
    provider_artifact_sha256: str,
    example_id: str,
    family_id: str,
    objective: float,
    h4_sha256: str,
    logits_sha256: str,
) -> str:
    return _v14._sha256(
        {
            "law": law,
            "phase": phase,
            "provider_artifact_sha256": provider_artifact_sha256,
            "example_id": example_id,
            "family_id": family_id,
            "objective": objective,
            "post_cast_h4_sha256": h4_sha256,
            "supervised_full_vocab_logits_sha256": logits_sha256,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )


def _collect_initial_fit_evidence(
    context: object,
    workspace: object,
    teacher_vault: object,
    *,
    law: str,
    source_artifact_sha256: str,
) -> _InitialFitEvidence:
    if law not in _FIT_LAWS:
        raise ValueError("V20d fit law differs")
    training = _v20b._ordered_records(workspace.training_records)
    families = tuple(sorted({record.sequence.family_id for record in training}))
    if (
        len(training) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or len(families) != _FIT_FAMILY_COUNT
        or set(families) & set(_v20c._FROZEN_EXCLUDED)
    ):
        raise RuntimeError("V20d initial fit complement differs")
    initial_evidence_sha256 = _v14._sha256(
        {
            "source_artifact_sha256": source_artifact_sha256,
            "fit_receipt_sha256": workspace.fit_receipt["artifact_sha256"],
            "law": law,
            "initial_response_weight": _INITIAL_WEIGHT,
            "training_sequence_sha256s": tuple(
                record.sequence.artifact_sha256 for record in training
            ),
            "held_family_ids": _v20c._FROZEN_EXCLUDED,
            "held_rows_used": False,
        },
        domain=_GRADIENT_EVIDENCE_DOMAIN,
    )
    provider = _build_response_provider(
        workspace,
        law=law,
        response_weight=_initial_weight_tensor(),
        polarity=1,
        evidence_sha256=initial_evidence_sha256,
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in training),
        held_family_id=None,
    )
    gradients: dict[str, dict[str, tuple[float, float, float]]] = {
        family: {} for family in families
    }
    objectives: dict[str, dict[str, float]] = {family: {} for family in families}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
    gradient_hashes: dict[str, str] = {}
    for record in training:
        model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
            context, record
        )
        teacher = capability.get(
            record.sequence.example_id, family_id=record.sequence.family_id
        )
        objective, captured = _v19._teacher_kl_objective(
            teacher, supervised_indices
        )
        execution, h4_gradient = context.bridge.execute_h4_vjp(
            context.adapter,
            model_inputs,
            objective=objective,
            h4_head=provider,
        )
        score, h4_sha, logits_sha = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=provider.artifact_sha256,
        )
        if len(captured) != 1 or score != captured[0]:
            raise RuntimeError("V20d initial objective capture drifted")
        gradient = _local_response_weight_gradient(
            provider, record.sequence, h4_gradient
        )
        example = record.sequence.example_id
        family = record.sequence.family_id
        values = tuple(float(item) for item in gradient.tolist())
        gradients[family][example] = values  # type: ignore[assignment]
        objectives[family][example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        gradient_hashes[example] = _v14._tensor_sha256(gradient)
        execution_hashes[example] = _fit_execution_sha256(
            law=law,
            phase="initial_vjp_alpha_zero",
            provider_artifact_sha256=provider.artifact_sha256,
            example_id=example,
            family_id=family,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
        )
        del model_inputs, teacher, execution, h4_gradient, gradient
    phase_capability = capability.receipt()
    _v20b._validate_capability_receipt(
        phase_capability,
        expected_example_ids=tuple(
            record.sequence.example_id for record in training
        ),
        expected_family_count=_FIT_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label=f"V20d {law} initial fit capability",
    )
    payload = {
        "law": law,
        "initial_response_weight": _INITIAL_WEIGHT,
        "initial_response_weight_sha256": _provider_tensor_sha256(
            _initial_weight_tensor()
        ),
        "provider_artifact_sha256": provider.artifact_sha256,
        "provider_transfer_evidence_sha256": provider.transfer_evidence_sha256,
        "fit_receipt_sha256": workspace.fit_receipt["artifact_sha256"],
        "training_family_ids": families,
        "training_sequence_sha256s": tuple(
            record.sequence.artifact_sha256 for record in training
        ),
        "training_example_sequence_sha256s": {
            record.sequence.example_id: record.sequence.artifact_sha256
            for record in training
        },
        "training_example_family_ids": {
            record.sequence.example_id: record.sequence.family_id
            for record in training
        },
        "gradient_sha256s": dict(sorted(gradient_hashes.items())),
        "initial_objectives_by_family": objectives,
        "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
        "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
        "execution_receipt_sha256s": dict(sorted(execution_hashes.items())),
        "initial_phase_capability_receipt": phase_capability,
        "full_suffix_vjp_count": len(training),
        "local_response_autograd_contraction_count": len(training),
        "alpha_zero_exact_execution_reusable": True,
        "held_family_ids": _v20c._FROZEN_EXCLUDED,
        "held_data_or_objectives_used": False,
        "raw_gradients_h4_logits_or_tensors_serialized": False,
    }
    evidence = {
        **payload,
        "artifact_sha256": _v14._sha256(
            payload, domain=_GRADIENT_EVIDENCE_DOMAIN
        ),
    }
    _v14._scalar_report(evidence)
    return _InitialFitEvidence(
        provider=provider,
        capability=capability,
        training_records=training,
        gradients_by_family=gradients,
        objectives_by_family=objectives,
        h4_sha256s=h4_hashes,
        logits_sha256s=logits_hashes,
        execution_sha256s=execution_hashes,
        gradient_evidence=evidence,
    )


def _score_fit_candidate(
    context: object,
    workspace: object,
    initial: _InitialFitEvidence,
    *,
    law: str,
    alpha: float,
    response_weight: Sequence[float],
    candidate_evidence_sha256: str,
) -> _CandidateLive:
    if alpha not in _ALPHAS or law not in _FIT_LAWS:
        raise ValueError("V20d fit candidate geometry differs")
    weight = _weight_tensor(response_weight)
    if fisher_continuous_bilinear_box_max_abs(weight) > 1.0:
        raise ValueError("V20d candidate escaped global bilinear box")
    if alpha == 0.0:
        provider = initial.provider
        if not torch.equal(provider.response_weight, weight):
            raise ValueError("V20d alpha-zero weight differs from initial provider")
        objectives = {
            family: dict(values)
            for family, values in initial.objectives_by_family.items()
        }
        h4_hashes = dict(initial.h4_sha256s)
        logits_hashes = dict(initial.logits_sha256s)
        execution_hashes = dict(initial.execution_sha256s)
        alpha_zero_reused = True
    else:
        provider = _build_response_provider(
            workspace,
            law=law,
            response_weight=weight,
            polarity=1,
            evidence_sha256=candidate_evidence_sha256,
        )
        objectives = {
            family: {}
            for family in sorted(
                {record.sequence.family_id for record in initial.training_records}
            )
        }
        h4_hashes: dict[str, str] = {}
        logits_hashes: dict[str, str] = {}
        execution_hashes: dict[str, str] = {}
        for record in initial.training_records:
            model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
                context, record
            )
            teacher = initial.capability.get(
                record.sequence.example_id, family_id=record.sequence.family_id
            )
            execution = context.bridge.execute(
                context.adapter, model_inputs, h4_head=provider
            )
            score, h4_sha, logits_sha = _v20a._execution_hashes_and_score(
                execution=execution,
                record=record,
                teacher=teacher,
                supervised_indices=supervised_indices,
                provider_artifact_sha256=provider.artifact_sha256,
            )
            example = record.sequence.example_id
            family = record.sequence.family_id
            objectives[family][example] = score
            h4_hashes[example] = h4_sha
            logits_hashes[example] = logits_sha
            execution_hashes[example] = _fit_execution_sha256(
                law=law,
                phase=f"finite_alpha_{alpha.hex()}",
                provider_artifact_sha256=provider.artifact_sha256,
                example_id=example,
                family_id=family,
                objective=score,
                h4_sha256=h4_sha,
                logits_sha256=logits_sha,
            )
            del model_inputs, teacher, execution
        alpha_zero_reused = False
    prompt_scores = {
        example: score
        for values in objectives.values()
        for example, score in values.items()
    }
    macro, family_scores = _v19._family_equal_mean(
        prompt_scores, initial.training_records
    )
    trace_with_values = _v20c._response_runtime_trace(
        provider, initial.training_records, arm=f"fit_{law}_{alpha.hex()}"
    )
    trace = _v20c._strip_transient_trace(trace_with_values)
    transient_gains = _mapping(
        trace_with_values.get("_transient_gain_values"),
        label="fit response gains",
    )
    joined_gains = torch.cat(
        tuple(
            value.reshape(-1)
            for _example, value in sorted(transient_gains.items())
            if isinstance(value, Tensor)
        )
    )
    gain_min = float(joined_gains.min())
    gain_max = float(joined_gains.max())
    gain_range = gain_max - gain_min
    corners = fisher_continuous_bilinear_corner_values(weight)
    payload = {
        "law": law,
        "alpha": alpha,
        "response_weight": tuple(float(item) for item in weight.tolist()),
        "response_weight_sha256": _provider_tensor_sha256(weight),
        "bilinear_corner_values": corners,
        "bilinear_box_max_abs": max(abs(float(item)) for item in corners),
        "global_bilinear_box_feasible": max(abs(float(item)) for item in corners)
        <= 1.0,
        "provider_artifact_sha256": provider.artifact_sha256,
        "provider_transfer_evidence_sha256": provider.transfer_evidence_sha256,
        "family_equal_objective": macro,
        "family_objectives": dict(sorted(family_scores.items())),
        "objectives_by_family": objectives,
        "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
        "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
        "execution_receipt_sha256s": dict(sorted(execution_hashes.items())),
        "response_trace": trace,
        "response_gain_min_on_fit_support": gain_min,
        "response_gain_max_on_fit_support": gain_max,
        "response_gain_range_on_fit_support": gain_range,
        "response_gain_nonconstant_on_fit_support": gain_range > 0.0,
        "exact_finite_full_model_forward": True,
        "alpha_zero_reused_from_exact_initial_vjp": alpha_zero_reused,
        "held_data_or_objectives_used": False,
        "raw_tensors_h4_logits_or_gradients_serialized": False,
    }
    evidence = {
        **payload,
        "artifact_sha256": _v14._sha256(
            payload, domain=_CANDIDATE_EVIDENCE_DOMAIN
        ),
    }
    _v14._scalar_report(evidence)
    return _CandidateLive(provider=provider, evidence=evidence)


def _nested_execution_sha256s(
    evidence: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    family_ids = _mapping(
        evidence.get("objectives_by_family"), label="candidate objectives"
    )
    flat = _mapping(
        evidence.get("execution_receipt_sha256s"), label="candidate executions"
    )
    result: dict[str, dict[str, str]] = {}
    for family, raw_rows in sorted(family_ids.items()):
        rows = _mapping(raw_rows, label=f"{family} candidate objectives")
        result[str(family)] = {
            str(example): _sha(flat.get(example), label="candidate execution")
            for example in sorted(rows)
        }
    return result


def _fit_response_law(
    context: object,
    workspace: object,
    teacher_vault: object,
    *,
    law: str,
    source: Mapping[str, object],
    family_ids: Sequence[str],
) -> _FitLive:
    """Fit one law without constructing any held-family capability."""

    if law not in _FIT_LAWS:
        raise ValueError("V20d fit law differs")
    source_hashes = _source_sha256s(source)
    initial = _collect_initial_fit_evidence(
        context,
        workspace,
        teacher_vault,
        law=law,
        source_artifact_sha256=str(source["artifact_sha256"]),
    )
    if initial.provider.response_law != law:
        raise RuntimeError("V20d initial provider law differs")
    direction = _core.build_natural_response_direction_receipt(
        v20c_source_sha256s=source_hashes,
        family_ids=family_ids,
        excluded_family_ids=_v20c._FROZEN_EXCLUDED,
        fit_gradients_by_family=initial.gradients_by_family,
        base_provider_artifact_sha256=workspace.base_provider.artifact_sha256,
        proposal_provider_artifact_sha256=workspace.proposal_provider.artifact_sha256,
        gradient_evidence_sha256=str(
            initial.gradient_evidence["artifact_sha256"]
        ),
        response_law=law,
        held_objectives_or_gradients_used=False,
    )
    if direction.get("response_law") != law:
        raise RuntimeError("V20d direction law binding differs")

    live_candidates: list[_CandidateLive] = []
    receipts: list[dict[str, object]] = []
    for alpha in _ALPHAS:
        unprojected = tuple(
            initial_weight + alpha * step
            for initial_weight, step in zip(
                _INITIAL_WEIGHT, direction["natural_direction"]
            )
        )
        projected = _core.radially_project_bilinear_weights(unprojected)
        weights = tuple(float(value) for value in projected["weights"])
        evidence_seed = _v14._sha256(
            {
                "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                "source_artifact_sha256": source["artifact_sha256"],
                "direction_artifact_sha256": direction["artifact_sha256"],
                "response_law": law,
                "alpha": alpha,
                "response_weights": weights,
                "held_rows_used": False,
            },
            domain=_CANDIDATE_EVIDENCE_DOMAIN,
        )
        live = _score_fit_candidate(
            context,
            workspace,
            initial,
            law=law,
            alpha=alpha,
            response_weight=weights,
            candidate_evidence_sha256=evidence_seed,
        )
        if live.provider.response_law != law:
            raise RuntimeError("V20d candidate provider law differs")
        candidate = _core.build_natural_response_alpha_candidate(
            direction_receipt=direction,
            alpha=alpha,
            provider_artifact_sha256=live.provider.artifact_sha256,
            exact_fit_objectives_by_family=_mapping(
                live.evidence["objectives_by_family"],
                label="exact fit objectives",
            ),
            fit_execution_receipt_sha256s_by_family=(
                _nested_execution_sha256s(live.evidence)
            ),
            objective_source="exact_finite_fit_execution",
            held_objectives_used=False,
        )
        if (
            tuple(float(value) for value in candidate["weights"]) != weights
            or candidate["provider_artifact_sha256"]
            != live.provider.artifact_sha256
            or candidate["response_law"] != law
        ):
            raise RuntimeError("V20d core/live candidate binding differs")
        live_candidates.append(live)
        receipts.append(candidate)

    fit = _core.build_natural_response_fit_receipt(
        direction_receipt=direction,
        candidates=receipts,
    )
    if fit.get("response_law") != law:
        raise RuntimeError("V20d selected fit law differs")
    by_provider = {
        value.provider.artifact_sha256: value for value in live_candidates
    }
    selected_provider = None
    if fit["learned_candidate_authorized"] is True:
        selected_provider = by_provider.get(str(fit["selected_provider_artifact_sha256"]))
        if (
            selected_provider is None
            or selected_provider.provider.response_law != law
            or tuple(float(item) for item in selected_provider.provider.response_weight.tolist())
            != tuple(float(item) for item in fit["selected_weights"])
        ):
            raise RuntimeError("V20d selected provider did not bind the learned fit")
        selected_provider_value = selected_provider.provider
        selected_gain_nonconstant = (
            selected_provider.evidence.get(
                "response_gain_nonconstant_on_fit_support"
            )
            is True
            and float(
                selected_provider.evidence.get(
                    "response_gain_range_on_fit_support", 0.0
                )
            )
            > 0.0
        )
    else:
        selected_provider_value = None
        selected_gain_nonconstant = False

    final_capability = initial.capability.receipt()
    expected_accesses = len(_ALPHAS)
    _v20b._validate_capability_receipt(
        final_capability,
        expected_example_ids=tuple(
            record.sequence.example_id for record in initial.training_records
        ),
        expected_family_count=_FIT_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=expected_accesses,
        label=f"V20d {law} complete fit capability",
    )
    fit_payload = {
        "response_law": law,
        "direction_artifact_sha256": direction["artifact_sha256"],
        "fit_artifact_sha256": fit["artifact_sha256"],
        "gradient_evidence": initial.gradient_evidence,
        "candidate_evidence": tuple(value.evidence for value in live_candidates),
        "fit_phase_capability_receipt": final_capability,
        "selected_provider_artifact_sha256": (
            selected_provider_value.artifact_sha256
            if selected_provider_value is not None
            else None
        ),
        "selected_response_gain_nonconstant_on_fit_support": (
            selected_gain_nonconstant
        ),
        "initial_vjp_is_law_specific": True,
        "alpha_zero_reused_only_within_same_law": True,
        "all_six_candidates_exactly_scored_on_six_family_fit_complement": True,
        "fit_frozen_before_any_held_capability": True,
        "held_data_or_objectives_used": False,
        "raw_gradients_tensors_h4_logits_or_targets_serialized": False,
    }
    fit_evidence = {
        **fit_payload,
        "artifact_sha256": _v14._sha256(
            fit_payload, domain=_FIT_EVIDENCE_DOMAIN
        ),
    }
    _v14._scalar_report(fit_evidence)
    return _FitLive(
        law=law,
        initial=initial,
        direction_receipt=direction,
        candidates=tuple(live_candidates),
        candidate_receipts=tuple(receipts),
        fit_receipt=fit,
        fit_evidence=fit_evidence,
        selected_provider=selected_provider_value,
    )


def _provider_receipt(
    provider: object,
    *,
    arm: str,
    fit_bundle_artifact_sha256: str,
    selected_fit_artifact_sha256: str | None,
) -> dict[str, object]:
    if arm not in _HELD_ARMS:
        raise ValueError("V20d provider arm differs")
    metadata = dict(getattr(provider, "metadata")())
    _v14._scalar_report(metadata)
    continuous = isinstance(
        provider, AutonomousCompleteH4FisherContinuousTransferProvider
    )
    if continuous:
        weight = tuple(float(item) for item in provider.response_weight.tolist())
        corners = tuple(
            float(item)
            for item in fisher_continuous_bilinear_corner_values(
                provider.response_weight
            )
        )
        box = max(abs(item) for item in corners)
    else:
        weight = None
        corners = None
        box = None
    payload = {
        "arm": arm,
        "fit_bundle_artifact_sha256": _sha(
            fit_bundle_artifact_sha256, label="two-fit bundle"
        ),
        "selected_fit_artifact_sha256": (
            _sha(selected_fit_artifact_sha256, label="selected law fit")
            if selected_fit_artifact_sha256 is not None
            else None
        ),
        "provider_artifact_sha256": _sha(
            getattr(provider, "artifact_sha256"), label=f"{arm} provider"
        ),
        "provider_metadata_sha256": _v14._sha256(
            metadata, domain=_PROVIDER_RECEIPT_DOMAIN
        ),
        "base_provider_artifact_sha256": (
            provider.base_provider.artifact_sha256
            if continuous
            else provider.artifact_sha256
        ),
        "proposal_provider_artifact_sha256": (
            provider.proposal_provider.artifact_sha256 if continuous else None
        ),
        "response_weight": weight,
        "response_weight_sha256": (
            _provider_tensor_sha256(provider.response_weight)
            if continuous
            else None
        ),
        "bilinear_corner_values": corners,
        "bilinear_box_max_abs": box,
        "global_bilinear_box_feasible": box is None or box <= 1.0,
        "response_source": provider.response_source if continuous else "base_zero",
        "response_law": provider.response_law if continuous else "base_zero",
        "polarity": int(provider.polarity) if continuous else 0,
        "signed_log_kappa": (
            float(provider.signed_log_kappa) if continuous else None
        ),
        "transfer_protocol_sha256": (
            provider.transfer_protocol_sha256 if continuous else None
        ),
        "transfer_evidence_sha256": (
            provider.transfer_evidence_sha256 if continuous else None
        ),
        "rank": int(getattr(provider, "rank")),
        "conditional_rank": int(getattr(provider, "conditional_rank")),
        "prepared_float_scalar_count": int(
            getattr(provider, "prepared_float_scalar_count")
        ),
        "logical_macs_per_token_upper_bound": int(
            getattr(provider, "logical_macs_per_token_upper_bound")
        ),
        "analysis_only": continuous,
        "provider_sidecar_serialized": False,
    }
    result = {
        **payload,
        "artifact_sha256": _v14._sha256(
            payload, domain=_PROVIDER_RECEIPT_DOMAIN
        ),
    }
    _v14._scalar_report(result)
    return result


def _build_frozen_held_providers(
    workspace: object,
    *,
    fits: Mapping[str, _FitLive],
    fit_bundle: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    if set(fits) != set(_FIT_LAWS) or fit_bundle.get("held_score_authorized") is not True:
        raise ValueError("V20d two-fit bundle does not authorize held providers")
    signed = fits["signed_log"]
    linear = fits["linear"]
    if signed.selected_provider is None or linear.selected_provider is None:
        raise RuntimeError("V20d learned provider is missing after authorized fit")
    if any(
        fit.fit_evidence.get(
            "selected_response_gain_nonconstant_on_fit_support"
        )
        is not True
        for fit in fits.values()
    ):
        raise PermissionError("V20d learned response is constant on fit support")
    if (
        signed.selected_provider.response_law != "signed_log"
        or linear.selected_provider.response_law != "linear"
    ):
        raise RuntimeError("V20d learned providers crossed law bindings")
    bundle_sha = str(fit_bundle["artifact_sha256"])
    providers: dict[str, object] = {
        "base": workspace.base_provider,
        "constant_plus_one": (
            build_autonomous_complete_h4_fisher_continuous_constant_control(
                workspace.base_provider,
                workspace.proposal_provider,
                alpha=1,
                transfer_protocol_sha256=_core.NATURAL_RESPONSE_PROTOCOL_SHA256,
                transfer_evidence_sha256=bundle_sha,
            )
        ),
        "fixed_signed_log": signed.initial.provider,
        "fixed_linear": linear.initial.provider,
        "learned_signed_log": signed.selected_provider,
        "learned_linear": linear.selected_provider,
        "learned_signed_log_sign_flip": _build_response_provider(
            workspace,
            law="signed_log",
            response_weight=signed.fit_receipt["selected_weights"],
            polarity=-1,
            evidence_sha256=signed.selected_provider.transfer_evidence_sha256,
        ),
    }
    if set(providers) != set(_HELD_ARMS):
        raise RuntimeError("V20d held provider geometry differs")
    provider_hashes = {
        getattr(provider, "artifact_sha256") for provider in providers.values()
    }
    if len(provider_hashes) != len(_HELD_ARMS):
        raise RuntimeError("V20d held provider artifacts are not distinct")
    selected_fit = {
        "learned_signed_log": str(signed.fit_receipt["artifact_sha256"]),
        "learned_signed_log_sign_flip": str(signed.fit_receipt["artifact_sha256"]),
        "learned_linear": str(linear.fit_receipt["artifact_sha256"]),
    }
    receipts = {
        arm: _provider_receipt(
            providers[arm],
            arm=arm,
            fit_bundle_artifact_sha256=bundle_sha,
            selected_fit_artifact_sha256=selected_fit.get(arm),
        )
        for arm in _HELD_ARMS
    }
    bundle_payload = {
        "fit_bundle_artifact_sha256": bundle_sha,
        "base_provider_artifact_sha256": workspace.base_provider.artifact_sha256,
        "proposal_provider_artifact_sha256": (
            workspace.proposal_provider.artifact_sha256
        ),
        "arm_order": _HELD_ARMS,
        "provider_artifact_sha256s": {
            arm: receipts[arm]["provider_artifact_sha256"] for arm in _HELD_ARMS
        },
        "provider_receipt_artifact_sha256s": {
            arm: receipts[arm]["artifact_sha256"] for arm in _HELD_ARMS
        },
        "learned_signed_log_fit_artifact_sha256": signed.fit_receipt[
            "artifact_sha256"
        ],
        "learned_linear_fit_artifact_sha256": linear.fit_receipt[
            "artifact_sha256"
        ],
        "all_seven_providers_frozen_before_held_capability": True,
        "provider_sidecar_or_raw_tensor_serialized": False,
    }
    provider_bundle = {
        **bundle_payload,
        "artifact_sha256": _v14._sha256(
            bundle_payload, domain=_PROVIDER_BUNDLE_DOMAIN
        ),
    }
    _v14._scalar_report(provider_bundle)
    return providers, receipts, provider_bundle


def _validate_provider_bundle(
    provider_receipts: Mapping[str, Mapping[str, object]],
    *,
    provider_bundle: Mapping[str, object],
    fit_bundle: Mapping[str, object],
    fit_evidence_by_law: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if set(provider_receipts) != set(_HELD_ARMS):
        raise ValueError("V20d provider receipt arms differ")
    bundle_sha = _sha(fit_bundle.get("artifact_sha256"), label="two-fit bundle")
    base_sha = _sha(
        fit_bundle.get("base_provider_artifact_sha256"), label="base endpoint"
    )
    proposal_sha = _sha(
        fit_bundle.get("proposal_provider_artifact_sha256"),
        label="proposal endpoint",
    )
    fits = _mapping(
        fit_bundle.get("fit_receipts_by_law"), label="two-fit receipts"
    )
    if set(fit_evidence_by_law) != set(_FIT_LAWS):
        raise ValueError("V20d provider bundle fit evidence differs")
    selected_candidate_evidence: dict[str, Mapping[str, object]] = {}
    for law in _FIT_LAWS:
        selected_provider_sha = fits[law]["selected_provider_artifact_sha256"]
        matches = tuple(
            _mapping(item, label=f"{law} candidate evidence")
            for item in _sequence(
                fit_evidence_by_law[law].get("candidate_evidence"),
                label=f"{law} candidate evidence",
            )
            if isinstance(item, Mapping)
            and item.get("provider_artifact_sha256") == selected_provider_sha
        )
        if len(matches) != 1:
            raise ValueError(f"V20d {law} selected candidate evidence differs")
        selected_candidate_evidence[law] = matches[0]
    expected = {
        "base": (None, "base_zero", "base_zero", 0, None),
        "constant_plus_one": ((0.0, 0.0, 0.0), "constant", "linear", 1, None),
        "fixed_signed_log": (_INITIAL_WEIGHT, "direct", "signed_log", 1, None),
        "fixed_linear": (_INITIAL_WEIGHT, "direct", "linear", 1, None),
        "learned_signed_log": (
            tuple(float(value) for value in fits["signed_log"]["selected_weights"]),
            "direct",
            "signed_log",
            1,
            str(fits["signed_log"]["artifact_sha256"]),
        ),
        "learned_linear": (
            tuple(float(value) for value in fits["linear"]["selected_weights"]),
            "direct",
            "linear",
            1,
            str(fits["linear"]["artifact_sha256"]),
        ),
        "learned_signed_log_sign_flip": (
            tuple(float(value) for value in fits["signed_log"]["selected_weights"]),
            "direct",
            "signed_log",
            -1,
            str(fits["signed_log"]["artifact_sha256"]),
        ),
    }
    normalized: dict[str, dict[str, object]] = {}
    for arm in _HELD_ARMS:
        receipt = dict(_mapping(provider_receipts[arm], label=f"{arm} provider"))
        payload = {
            key: value for key, value in receipt.items() if key != "artifact_sha256"
        }
        if receipt.get("artifact_sha256") != _v14._sha256(
            payload, domain=_PROVIDER_RECEIPT_DOMAIN
        ):
            raise ValueError(f"V20d {arm} provider receipt hash differs")
        weight, source, law, polarity, selected_fit = expected[arm]
        continuous = arm != "base"
        if weight is None:
            expected_weight_hash = None
            expected_corners = None
            expected_box = None
        else:
            tensor = _weight_tensor(weight)
            expected_weight_hash = _provider_tensor_sha256(tensor)
            expected_corners = tuple(
                float(item)
                for item in fisher_continuous_bilinear_corner_values(tensor)
            )
            expected_box = max(abs(item) for item in expected_corners)
        if (
            receipt.get("arm") != arm
            or receipt.get("fit_bundle_artifact_sha256") != bundle_sha
            or receipt.get("selected_fit_artifact_sha256") != selected_fit
            or receipt.get("base_provider_artifact_sha256") != base_sha
            or receipt.get("proposal_provider_artifact_sha256")
            != (proposal_sha if continuous else None)
            or (
                tuple(receipt.get("response_weight", ()))
                if continuous
                else receipt.get("response_weight")
            )
            != weight
            or receipt.get("response_weight_sha256") != expected_weight_hash
            or (
                tuple(receipt.get("bilinear_corner_values", ()))
                if continuous
                else receipt.get("bilinear_corner_values")
            )
            != expected_corners
            or receipt.get("bilinear_box_max_abs") != expected_box
            or receipt.get("global_bilinear_box_feasible") is not True
            or receipt.get("response_source") != source
            or receipt.get("response_law") != law
            or receipt.get("polarity") != polarity
            or receipt.get("signed_log_kappa")
            != (9.0 if continuous else None)
            or receipt.get("transfer_protocol_sha256")
            != (
                _core.NATURAL_RESPONSE_PROTOCOL_SHA256 if continuous else None
            )
            or receipt.get("rank") != _RANK
            or receipt.get("conditional_rank") != _CONDITIONAL_RANK
            or receipt.get("analysis_only") is not continuous
            or receipt.get("provider_sidecar_serialized") is not False
            or not isinstance(receipt.get("prepared_float_scalar_count"), int)
            or int(receipt.get("prepared_float_scalar_count", 0)) <= 0
            or not isinstance(
                receipt.get("logical_macs_per_token_upper_bound"), int
            )
            or int(receipt.get("logical_macs_per_token_upper_bound", 0)) <= 0
        ):
            raise ValueError(f"V20d {arm} provider semantics differ")
        _sha(receipt.get("provider_artifact_sha256"), label=f"{arm} provider")
        _sha(receipt.get("provider_metadata_sha256"), label=f"{arm} metadata")
        if continuous:
            _sha(
                receipt.get("transfer_evidence_sha256"),
                label=f"{arm} transfer evidence",
            )
            if expected_box is None or expected_box > 1.0:
                raise ValueError(f"V20d {arm} global box certificate differs")
        elif (
            receipt.get("transfer_evidence_sha256") is not None
            or receipt.get("provider_artifact_sha256") != base_sha
        ):
            raise ValueError("V20d base provider semantics differ")
        normalized[arm] = receipt
    if (
        normalized["constant_plus_one"]["transfer_evidence_sha256"] != bundle_sha
        or normalized["fixed_signed_log"]["provider_artifact_sha256"]
        != fit_evidence_by_law["signed_log"]["gradient_evidence"][
            "provider_artifact_sha256"
        ]
        or normalized["fixed_signed_log"]["transfer_evidence_sha256"]
        != fit_evidence_by_law["signed_log"]["gradient_evidence"][
            "provider_transfer_evidence_sha256"
        ]
        or normalized["fixed_linear"]["provider_artifact_sha256"]
        != fit_evidence_by_law["linear"]["gradient_evidence"][
            "provider_artifact_sha256"
        ]
        or normalized["fixed_linear"]["transfer_evidence_sha256"]
        != fit_evidence_by_law["linear"]["gradient_evidence"][
            "provider_transfer_evidence_sha256"
        ]
        or normalized["learned_signed_log"]["provider_artifact_sha256"]
        != fit_bundle["selected_provider_artifact_sha256s_by_law"]["signed_log"]
        or normalized["learned_signed_log"]["transfer_evidence_sha256"]
        != selected_candidate_evidence["signed_log"][
            "provider_transfer_evidence_sha256"
        ]
        or normalized["learned_linear"]["provider_artifact_sha256"]
        != fit_bundle["selected_provider_artifact_sha256s_by_law"]["linear"]
        or normalized["learned_linear"]["transfer_evidence_sha256"]
        != selected_candidate_evidence["linear"][
            "provider_transfer_evidence_sha256"
        ]
    ):
        raise ValueError("V20d learned provider fit binding differs")
    positive = normalized["learned_signed_log"]
    mirror = normalized["learned_signed_log_sign_flip"]
    mirror_equal_fields = (
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "response_weight",
        "response_weight_sha256",
        "bilinear_corner_values",
        "bilinear_box_max_abs",
        "response_source",
        "response_law",
        "signed_log_kappa",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "rank",
        "conditional_rank",
        "prepared_float_scalar_count",
        "logical_macs_per_token_upper_bound",
        "selected_fit_artifact_sha256",
    )
    if any(positive[field] != mirror[field] for field in mirror_equal_fields):
        raise ValueError("V20d learned mirror changes more than polarity")
    artifacts = {
        str(receipt["provider_artifact_sha256"])
        for receipt in normalized.values()
    }
    if len(artifacts) != len(_HELD_ARMS):
        raise ValueError("V20d held providers are not artifact-distinct")

    provider_bundle_payload = {
        key: value
        for key, value in provider_bundle.items()
        if key != "artifact_sha256"
    }
    expected_bundle_payload = {
        "fit_bundle_artifact_sha256": bundle_sha,
        "base_provider_artifact_sha256": base_sha,
        "proposal_provider_artifact_sha256": proposal_sha,
        "arm_order": _HELD_ARMS,
        "provider_artifact_sha256s": {
            arm: normalized[arm]["provider_artifact_sha256"]
            for arm in _HELD_ARMS
        },
        "provider_receipt_artifact_sha256s": {
            arm: normalized[arm]["artifact_sha256"] for arm in _HELD_ARMS
        },
        "learned_signed_log_fit_artifact_sha256": fits["signed_log"][
            "artifact_sha256"
        ],
        "learned_linear_fit_artifact_sha256": fits["linear"][
            "artifact_sha256"
        ],
        "all_seven_providers_frozen_before_held_capability": True,
        "provider_sidecar_or_raw_tensor_serialized": False,
    }
    if (
        _v14._canonical_json_bytes(provider_bundle_payload)
        != _v14._canonical_json_bytes(expected_bundle_payload)
        or provider_bundle.get("artifact_sha256")
        != _v14._sha256(
            expected_bundle_payload, domain=_PROVIDER_BUNDLE_DOMAIN
        )
    ):
        raise ValueError("V20d provider bundle receipt differs")
    return normalized, dict(provider_bundle)


def _held_execution_sha256(
    *,
    fit_bundle_artifact_sha256: str,
    arm: str,
    provider_artifact_sha256: str,
    outer_family_id: str,
    scored_family_id: str,
    h4_sha256s: Mapping[str, str],
    logits_sha256s: Mapping[str, str],
    response_trace_sha256: str,
    objective: float,
) -> str:
    return _v14._sha256(
        {
            "fit_bundle_artifact_sha256": fit_bundle_artifact_sha256,
            "arm": arm,
            "provider_artifact_sha256": provider_artifact_sha256,
            "outer_held_family_id": outer_family_id,
            "scored_inner_family_id": scored_family_id,
            "post_cast_h4_sha256s": dict(sorted(h4_sha256s.items())),
            "supervised_full_vocab_logits_sha256s": dict(
                sorted(logits_sha256s.items())
            ),
            "response_trace_sha256": response_trace_sha256,
            "objective": float(objective),
        },
        domain=_HELD_EXECUTION_DOMAIN,
    )


def _score_exact_held_arm(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    arm: str,
    outer_family_id: str,
    fit_bundle_artifact_sha256: str,
    baseline_hashes: tuple[Mapping[str, str], Mapping[str, str]] | None,
) -> tuple[dict[str, object], tuple[dict[str, str], dict[str, str]]]:
    ordered = _v20b._ordered_records(records)
    families = {record.sequence.family_id for record in ordered}
    if (
        arm not in _HELD_ARMS
        or len(ordered) != _PROMPTS_PER_FAMILY
        or len(families) != 1
    ):
        raise RuntimeError("V20d held arm score geometry differs")
    scored_family = next(iter(families))
    trace_with_values = _v20c._response_runtime_trace(
        provider, ordered, arm=arm
    )
    trace = _v20c._strip_transient_trace(trace_with_values)
    transient_gains = _mapping(
        trace_with_values.get("_transient_gain_values"),
        label="held response gains",
    )
    joined_gains = torch.cat(
        tuple(
            value.reshape(-1)
            for _example, value in sorted(transient_gains.items())
            if isinstance(value, Tensor)
        )
    )
    gain_min = float(joined_gains.min())
    gain_max = float(joined_gains.max())
    gain_range = gain_max - gain_min
    prompt_scores: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    for record in ordered:
        model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
            context, record
        )
        teacher = capability.get(
            record.sequence.example_id, family_id=record.sequence.family_id
        )
        execution = context.bridge.execute(
            context.adapter, model_inputs, h4_head=provider
        )
        score, h4_sha, logits_sha = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=provider.artifact_sha256,
        )
        example = record.sequence.example_id
        prompt_scores[example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        del model_inputs, teacher, execution
    objective, family_scores = _v19._family_equal_mean(prompt_scores, ordered)
    if set(family_scores) != {scored_family}:
        raise RuntimeError("V20d held family objective geometry differs")
    changed = False
    if baseline_hashes is not None:
        changed = h4_hashes != dict(baseline_hashes[0]) or logits_hashes != dict(
            baseline_hashes[1]
        )
    execution_sha = _held_execution_sha256(
        fit_bundle_artifact_sha256=fit_bundle_artifact_sha256,
        arm=arm,
        provider_artifact_sha256=provider.artifact_sha256,
        outer_family_id=outer_family_id,
        scored_family_id=scored_family,
        h4_sha256s=h4_hashes,
        logits_sha256s=logits_hashes,
        response_trace_sha256=str(trace["artifact_sha256"]),
        objective=objective,
    )
    result = {
        "arm": arm,
        "objective": objective,
        "prompt_objectives": dict(sorted(prompt_scores.items())),
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_receipt_sha256": execution_sha,
        "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
        "supervised_full_vocab_logits_sha256s": dict(
            sorted(logits_hashes.items())
        ),
        "response_trace": trace,
        "response_gain_min_on_held_support": gain_min,
        "response_gain_max_on_held_support": gain_max,
        "response_gain_range_on_held_support": gain_range,
        "response_gain_nonconstant_on_held_support": gain_range > 0.0,
        "execution_changed_from_base": changed,
    }
    _v14._scalar_report(result)
    result["_transient_trace"] = trace_with_values
    return result, (dict(h4_hashes), dict(logits_hashes))


def _score_reciprocal_role(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    scored_family_id: str,
    providers: Mapping[str, object],
    provider_receipts: Mapping[str, Mapping[str, object]],
    provider_bundle: Mapping[str, object],
    fit_bundle: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        fit_bundle.get("held_score_authorized") is not True
        or set(providers) != set(_HELD_ARMS)
        or set(provider_receipts) != set(_HELD_ARMS)
        or provider_bundle.get("all_seven_providers_frozen_before_held_capability")
        is not True
    ):
        raise PermissionError("V20d held capability barrier is not satisfied")
    selected_records = _v20b._ordered_records(
        tuple(
            record
            for record in records
            if record.sequence.family_id == scored_family_id
        )
    )
    if len(selected_records) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20d held role prompt geometry differs")

    # This is deliberately the first held-family operation in the runner.
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in selected_records),
        held_family_id=outer_family_id,
    )
    raw_by_arm: dict[str, dict[str, object]] = {}
    baseline, baseline_hashes = _score_exact_held_arm(
        context,
        selected_records,
        capability,
        provider=providers["base"],
        arm="base",
        outer_family_id=outer_family_id,
        fit_bundle_artifact_sha256=str(fit_bundle["artifact_sha256"]),
        baseline_hashes=None,
    )
    raw_by_arm["base"] = baseline
    for arm in _HELD_ARMS:
        if arm == "base":
            continue
        row, _hashes = _score_exact_held_arm(
            context,
            selected_records,
            capability,
            provider=providers[arm],
            arm=arm,
            outer_family_id=outer_family_id,
            fit_bundle_artifact_sha256=str(fit_bundle["artifact_sha256"]),
            baseline_hashes=baseline_hashes,
        )
        raw_by_arm[arm] = row
    _v20c._assert_exact_mirror(
        _mapping(
            raw_by_arm["learned_signed_log"]["_transient_trace"],
            label="learned signed-log trace",
        ),
        _mapping(
            raw_by_arm["learned_signed_log_sign_flip"]["_transient_trace"],
            label="learned signed-log mirror trace",
        ),
    )
    learned_nonconstant = bool(
        raw_by_arm["learned_signed_log"][
            "response_gain_nonconstant_on_held_support"
        ]
        is True
        and float(
            raw_by_arm["learned_signed_log"][
                "response_gain_range_on_held_support"
            ]
        )
        > 0.0
    )

    arm_scores: list[dict[str, object]] = []
    persisted: dict[str, dict[str, object]] = {}
    for arm in _HELD_ARMS:
        raw = raw_by_arm[arm]
        trace = _mapping(raw["response_trace"], label=f"{arm} held trace")
        expected_provider = provider_receipts[arm]["provider_artifact_sha256"]
        if raw["provider_artifact_sha256"] != expected_provider:
            raise RuntimeError("V20d held execution escaped the provider bundle")
        score = _core.build_natural_response_held_arm_score(
            fit_bundle_receipt=fit_bundle,
            outer_held_family_id=outer_family_id,
            held_family_id=scored_family_id,
            arm=arm,
            objective=float(raw["objective"]),
            provider_artifact_sha256=str(expected_provider),
            execution_receipt_sha256=str(raw["execution_receipt_sha256"]),
            finite=trace.get("finite") is True,
            pointwise_trust_passed=trace.get("pointwise_trust_passed") is True,
            rank_is_16=trace.get("endpoint_conditional_ranks_are_16") is True,
            execution_changed_from_base=bool(
                raw["execution_changed_from_base"]
            ),
            response_nonconstant=(
                raw.get("response_gain_nonconstant_on_held_support") is True
                and float(raw.get("response_gain_range_on_held_support", 0.0))
                > 0.0
            ),
            score_source="exact_finite_held_execution",
        )
        arm_scores.append(score)
        persisted[arm] = {
            key: value
            for key, value in raw.items()
            if not key.startswith("_transient_")
        }
    role = _core.build_natural_response_held_role_receipt(
        fit_bundle_receipt=fit_bundle,
        arm_scores=arm_scores,
    )
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(
            record.sequence.example_id for record in selected_records
        ),
        expected_family_count=1,
        expected_held_family_id=outer_family_id,
        expected_accesses_per_example=len(_HELD_ARMS),
        label="V20d held role capability",
    )
    evidence_payload = {
        "fit_bundle_artifact_sha256": fit_bundle["artifact_sha256"],
        "provider_bundle_artifact_sha256": provider_bundle["artifact_sha256"],
        "outer_held_family_id": outer_family_id,
        "scored_inner_family_id": scored_family_id,
        "capability_receipt": capability_receipt,
        "arm_execution_evidence": persisted,
        "learned_signed_log_gain_nonconstant_on_held_support": learned_nonconstant,
        "learned_signed_log_mirror_exact_negative": True,
        "both_fits_and_all_seven_providers_frozen_before_capability": True,
    }
    evidence = {
        **evidence_payload,
        "artifact_sha256": _v14._sha256(
            evidence_payload, domain=_ROLE_EVIDENCE_DOMAIN
        ),
    }
    _v14._scalar_report(evidence)
    return role, evidence


def _pair_qualification(
    *,
    fit_bundle: Mapping[str, object],
    provider_bundle: Mapping[str, object],
    provider_receipts: Mapping[str, Mapping[str, object]],
    fits: Mapping[str, Mapping[str, object]],
    roles: Sequence[Mapping[str, object]],
    role_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    core = _core.build_natural_response_pair_qualification(
        fit_bundle_receipt=fit_bundle,
        roles=roles,
    )
    normalized_roles = tuple(
        _core.validate_natural_response_held_role_receipt(
            value, fit_bundle_receipt=fit_bundle
        )
        for value in roles
    )
    roles_by_outer = {
        str(value["outer_held_family_id"]): value for value in normalized_roles
    }
    raw_evidence = tuple(
        _mapping(value, label="held role evidence") for value in role_evidence
    )
    evidence_outer_ids = {
        str(value.get("outer_held_family_id")) for value in raw_evidence
    }
    evidence_hashes = {
        str(value.get("artifact_sha256")) for value in raw_evidence
    }
    expected_outer_ids = set(str(item) for item in fit_bundle["excluded_family_ids"])
    if (
        len(raw_evidence) != 2
        or len(roles_by_outer) != 2
        or set(roles_by_outer) != expected_outer_ids
        or evidence_outer_ids != expected_outer_ids
        or len(evidence_hashes) != 2
    ):
        raise ValueError("V20d qualification requires two role evidence rows")
    evidence = tuple(
        _validate_role_evidence(
            value,
            role_receipt=roles_by_outer[
                str(value.get("outer_held_family_id"))
            ],
            provider_receipts=provider_receipts,
            provider_bundle=provider_bundle,
            fit_bundle=fit_bundle,
        )
        for value in raw_evidence
    )
    fit_nonconstant = all(
        _mapping(fits[law], label=f"{law} fit evidence").get(
            "selected_response_gain_nonconstant_on_fit_support"
        )
        is True
        for law in _FIT_LAWS
    )
    held_nonconstant = all(
        value.get("learned_signed_log_gain_nonconstant_on_held_support") is True
        for value in evidence
    )
    mirror_exact = all(
        value.get("learned_signed_log_mirror_exact_negative") is True
        for value in evidence
    )
    barrier = all(
        value.get("both_fits_and_all_seven_providers_frozen_before_capability")
        is True
        and value.get("provider_bundle_artifact_sha256")
        == provider_bundle.get("artifact_sha256")
        and value.get("fit_bundle_artifact_sha256")
        == fit_bundle.get("artifact_sha256")
        for value in evidence
    )
    gates = {
        "core_exact_pair_qualification_passed": core["passed"] is True,
        "both_learned_laws_nonconstant_on_fit_support": fit_nonconstant,
        "learned_signed_log_nonconstant_on_both_held_roles": held_nonconstant,
        "learned_signed_log_mirror_exact_negative_on_both_roles": mirror_exact,
        "two_fit_and_seven_provider_barrier_bound_to_both_roles": barrier,
    }
    payload = {
        "fit_bundle_artifact_sha256": fit_bundle["artifact_sha256"],
        "provider_bundle_artifact_sha256": provider_bundle["artifact_sha256"],
        "core_pair_qualification": core,
        "role_evidence_artifact_sha256s": tuple(
            value["artifact_sha256"] for value in evidence
        ),
        "runtime_and_barrier_gates": gates,
        "passed": all(gates.values()),
        "scientific_scope": "development_only_reused_a16_pair_smoke",
    }
    result = {
        **payload,
        "artifact_sha256": _v14._sha256(
            payload, domain=_QUALIFICATION_DOMAIN
        ),
    }
    _v14._scalar_report(result)
    return result


def _validate_hashed_payload(
    value: Mapping[str, object], *, domain: bytes, label: str
) -> dict[str, object]:
    selected = dict(value)
    payload = {
        key: item for key, item in selected.items() if key != "artifact_sha256"
    }
    if selected.get("artifact_sha256") != _v14._sha256(payload, domain=domain):
        raise ValueError(f"{label} artifact hash differs")
    _v14._scalar_report(selected)
    return selected


def _validate_response_trace(value: Mapping[str, object], *, arm: str) -> dict[str, object]:
    selected = _validate_hashed_payload(
        value, domain=_v20c._RESPONSE_TRACE_DOMAIN, label=f"{arm} response trace"
    )
    gains = _mapping(
        selected.get("response_gain_sha256s"), label=f"{arm} response gain hashes"
    )
    if (
        selected.get("arm") != arm
        or not gains
        or selected.get("finite") is not True
        or selected.get("raw_response_or_modal_tensors_serialized") is not False
    ):
        raise ValueError(f"{arm} response trace semantics differ")
    for item in gains.values():
        _sha(item, label=f"{arm} response gain")
    return selected


def _validate_role_evidence(
    value: Mapping[str, object],
    *,
    role_receipt: Mapping[str, object],
    provider_receipts: Mapping[str, Mapping[str, object]],
    provider_bundle: Mapping[str, object],
    fit_bundle: Mapping[str, object],
) -> dict[str, object]:
    selected = _validate_hashed_payload(
        value, domain=_ROLE_EVIDENCE_DOMAIN, label="V20d held role evidence"
    )
    outer = _identifier(
        selected.get("outer_held_family_id"), label="role evidence outer family"
    )
    held = _identifier(
        selected.get("scored_inner_family_id"), label="role evidence scored family"
    )
    arms = _mapping(
        selected.get("arm_execution_evidence"), label="role arm evidence"
    )
    scores = {
        str(item["arm"]): item
        for item in _sequence(
            role_receipt.get("arm_scores"), label="role arm scores"
        )
        if isinstance(item, Mapping)
    }
    capability = _mapping(
        selected.get("capability_receipt"), label="held capability"
    )
    base_raw = _mapping(arms.get("base"), label="base held execution evidence")
    base_h4 = _mapping(
        base_raw.get("post_cast_h4_sha256s"), label="base held H4 hashes"
    )
    base_logits = _mapping(
        base_raw.get("supervised_full_vocab_logits_sha256s"),
        label="base held logit hashes",
    )
    expected_prompt_ids = tuple(sorted(str(item) for item in base_h4))
    if len(expected_prompt_ids) != _PROMPTS_PER_FAMILY:
        raise ValueError("V20d held role prompt IDs differ")
    _v20b._validate_capability_receipt(
        capability,
        expected_example_ids=expected_prompt_ids,
        expected_family_count=1,
        expected_held_family_id=outer,
        expected_accesses_per_example=len(_HELD_ARMS),
        label="V20d held role capability",
    )
    if (
        set(arms) != set(_HELD_ARMS)
        or set(scores) != set(_HELD_ARMS)
        or role_receipt.get("outer_held_family_id") != outer
        or role_receipt.get("held_family_id") != held
        or selected.get("fit_bundle_artifact_sha256")
        != fit_bundle.get("artifact_sha256")
        or selected.get("provider_bundle_artifact_sha256")
        != provider_bundle.get("artifact_sha256")
        or selected.get("learned_signed_log_mirror_exact_negative") is not True
        or selected.get(
            "both_fits_and_all_seven_providers_frozen_before_capability"
        )
        is not True
    ):
        raise ValueError("V20d role evidence authority differs")
    for arm in _HELD_ARMS:
        raw = _mapping(arms[arm], label=f"{arm} held execution evidence")
        score = _mapping(scores[arm], label=f"{arm} held score")
        trace = _validate_response_trace(
            _mapping(raw.get("response_trace"), label=f"{arm} held trace"),
            arm=arm,
        )
        provider_sha = provider_receipts[arm]["provider_artifact_sha256"]
        h4 = _mapping(
            raw.get("post_cast_h4_sha256s"), label=f"{arm} held H4 hashes"
        )
        logits = _mapping(
            raw.get("supervised_full_vocab_logits_sha256s"),
            label=f"{arm} held logit hashes",
        )
        prompt_objectives = _mapping(
            raw.get("prompt_objectives"), label=f"{arm} prompt objectives"
        )
        trace_gains = _mapping(
            trace.get("response_gain_sha256s"),
            label=f"{arm} held response gain hashes",
        )
        if (
            len(h4) != _PROMPTS_PER_FAMILY
            or set(h4) != set(logits)
            or set(h4) != set(prompt_objectives)
            or set(h4) != set(expected_prompt_ids)
            or set(trace_gains) != set(expected_prompt_ids)
        ):
            raise ValueError(f"V20d {arm} held prompt evidence differs")
        for item in (*h4.values(), *logits.values()):
            _sha(item, label=f"{arm} held tensor hash")
        objective = math.fsum(float(item) for item in prompt_objectives.values()) / len(
            prompt_objectives
        )
        response_nonconstant = bool(
            raw.get("response_gain_nonconstant_on_held_support") is True
            and float(raw.get("response_gain_range_on_held_support", 0.0)) > 0.0
        )
        changed_from_base = bool(
            arm != "base"
            and (dict(h4) != dict(base_h4) or dict(logits) != dict(base_logits))
        )
        expected_execution = _held_execution_sha256(
            fit_bundle_artifact_sha256=str(fit_bundle["artifact_sha256"]),
            arm=arm,
            provider_artifact_sha256=str(provider_sha),
            outer_family_id=outer,
            scored_family_id=held,
            h4_sha256s={str(key): str(item) for key, item in h4.items()},
            logits_sha256s={str(key): str(item) for key, item in logits.items()},
            response_trace_sha256=str(trace["artifact_sha256"]),
            objective=objective,
        )
        if (
            raw.get("arm") != arm
            or raw.get("provider_artifact_sha256") != provider_sha
            or trace.get("provider_artifact_sha256") != provider_sha
            or float(raw.get("objective", -1.0)) != objective
            or raw.get("execution_receipt_sha256") != expected_execution
            or score.get("objective") != objective
            or score.get("provider_artifact_sha256") != provider_sha
            or score.get("execution_receipt_sha256") != expected_execution
            or score.get("execution_changed_from_base")
            is not changed_from_base
            or raw.get("execution_changed_from_base") is not changed_from_base
            or score.get("response_nonconstant") is not response_nonconstant
            or score.get("finite") is not (trace.get("finite") is True)
            or score.get("pointwise_trust_passed")
            is not (trace.get("pointwise_trust_passed") is True)
            or score.get("rank_is_16")
            is not (trace.get("endpoint_conditional_ranks_are_16") is True)
            or tuple(trace.get("scored_family_ids", ())) != (held,)
            or raw.get("response_gain_range_on_held_support")
            != float(raw["response_gain_max_on_held_support"])
            - float(raw["response_gain_min_on_held_support"])
            or raw.get("response_gain_nonconstant_on_held_support")
            != (float(raw["response_gain_range_on_held_support"]) > 0.0)
        ):
            raise ValueError(f"V20d {arm} held execution binding differs")
    learned_nonconstant = (
        _mapping(arms["learned_signed_log"], label="learned held evidence").get(
            "response_gain_nonconstant_on_held_support"
        )
        is True
    )
    if (
        selected.get("learned_signed_log_gain_nonconstant_on_held_support")
        is not learned_nonconstant
    ):
        raise ValueError("V20d learned held nonconstant gate differs")
    return selected


def _validate_fit_evidence(
    value: Mapping[str, object],
    *,
    fit_receipt: Mapping[str, object],
    law: str,
) -> dict[str, object]:
    selected = _validate_hashed_payload(
        value, domain=_FIT_EVIDENCE_DOMAIN, label=f"{law} fit evidence"
    )
    if law not in _FIT_LAWS or fit_receipt.get("response_law") != law:
        raise ValueError("V20d fit evidence law differs")
    direction = _mapping(
        fit_receipt.get("direction_receipt"), label=f"{law} direction"
    )
    gradient = _validate_hashed_payload(
        _mapping(selected.get("gradient_evidence"), label=f"{law} gradient evidence"),
        domain=_GRADIENT_EVIDENCE_DOMAIN,
        label=f"{law} gradient evidence",
    )
    candidate_evidence = tuple(
        _validate_hashed_payload(
            _mapping(item, label=f"{law} candidate evidence"),
            domain=_CANDIDATE_EVIDENCE_DOMAIN,
            label=f"{law} candidate evidence",
        )
        for item in _sequence(
            selected.get("candidate_evidence"), label=f"{law} candidate evidence"
        )
    )
    candidate_receipts = tuple(
        _mapping(item, label=f"{law} candidate receipt")
        for item in _sequence(
            fit_receipt.get("candidate_receipts"), label=f"{law} candidates"
        )
    )
    by_alpha = {float(item["alpha"]): item for item in candidate_evidence}
    core_by_alpha = {float(item["alpha"]): item for item in candidate_receipts}
    fit_ids_by_family = {
        str(family): tuple(str(example) for example in examples)
        for family, examples in _mapping(
            direction.get("fit_example_ids_by_family"),
            label=f"{law} direction fit IDs",
        ).items()
    }
    exact_example_family_ids = {
        example: family
        for family, examples in fit_ids_by_family.items()
        for example in examples
    }
    exact_example_ids = tuple(sorted(exact_example_family_ids))
    ordered_training_example_ids = tuple(
        sorted(
            exact_example_family_ids,
            key=lambda example: (exact_example_family_ids[example], example),
        )
    )
    if (
        tuple(sorted(fit_ids_by_family))
        != tuple(direction.get("fit_family_ids", ()))
        or len(exact_example_ids) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY
    ):
        raise ValueError(f"V20d {law} direction example geometry differs")
    capability = _mapping(
        selected.get("fit_phase_capability_receipt"),
        label=f"{law} fit capability",
    )
    initial_capability = _mapping(
        gradient.get("initial_phase_capability_receipt"),
        label=f"{law} initial capability",
    )
    _v20b._validate_capability_receipt(
        initial_capability,
        expected_example_ids=exact_example_ids,
        expected_family_count=_FIT_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label=f"V20d {law} initial fit capability",
    )
    _v20b._validate_capability_receipt(
        capability,
        expected_example_ids=exact_example_ids,
        expected_family_count=_FIT_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ALPHAS),
        label=f"V20d {law} complete fit capability",
    )
    expected_initial_transfer_evidence = _v14._sha256(
        {
            "source_artifact_sha256": direction["v20c_source_sha256s"][
                "v20c_source_artifact_sha256"
            ],
            "fit_receipt_sha256": gradient.get("fit_receipt_sha256"),
            "law": law,
            "initial_response_weight": _INITIAL_WEIGHT,
            "training_sequence_sha256s": tuple(
                gradient.get("training_sequence_sha256s", ())
            ),
            "held_family_ids": _v20c._FROZEN_EXCLUDED,
            "held_rows_used": False,
        },
        domain=_GRADIENT_EVIDENCE_DOMAIN,
    )
    sequence_hashes = _mapping(
        gradient.get("training_example_sequence_sha256s"),
        label=f"{law} training sequence hashes",
    )
    if (
        set(sequence_hashes) != set(exact_example_ids)
        or tuple(sequence_hashes[example] for example in ordered_training_example_ids)
        != tuple(gradient.get("training_sequence_sha256s", ()))
    ):
        raise ValueError(f"V20d {law} training sequence geometry differs")
    for item in sequence_hashes.values():
        _sha(item, label=f"{law} training sequence")
    if (
        selected.get("response_law") != law
        or selected.get("direction_artifact_sha256")
        != direction.get("artifact_sha256")
        or gradient.get("artifact_sha256")
        != direction.get("gradient_evidence_sha256")
        or selected.get("fit_artifact_sha256")
        != fit_receipt.get("artifact_sha256")
        or gradient.get("law") != law
        or gradient.get("provider_artifact_sha256")
        != core_by_alpha.get(0.0, {}).get("provider_artifact_sha256")
        or tuple(gradient.get("initial_response_weight", ())) != _INITIAL_WEIGHT
        or gradient.get("initial_response_weight_sha256")
        != _provider_tensor_sha256(_initial_weight_tensor())
        or gradient.get("provider_transfer_evidence_sha256")
        != expected_initial_transfer_evidence
        or tuple(gradient.get("training_family_ids", ()))
        != tuple(direction.get("fit_family_ids", ()))
        or gradient.get("training_example_family_ids")
        != exact_example_family_ids
        or tuple(gradient.get("held_family_ids", ()))
        != _v20c._FROZEN_EXCLUDED
        or gradient.get("full_suffix_vjp_count") != len(exact_example_ids)
        or gradient.get("local_response_autograd_contraction_count")
        != len(exact_example_ids)
        or gradient.get("alpha_zero_exact_execution_reusable") is not True
        or gradient.get("held_data_or_objectives_used") is not False
        or gradient.get("raw_gradients_h4_logits_or_tensors_serialized")
        is not False
        or set(by_alpha) != set(_ALPHAS)
        or set(core_by_alpha) != set(_ALPHAS)
        or len(by_alpha) != len(_ALPHAS)
        or selected.get("initial_vjp_is_law_specific") is not True
        or selected.get("alpha_zero_reused_only_within_same_law") is not True
        or selected.get(
            "all_six_candidates_exactly_scored_on_six_family_fit_complement"
        )
        is not True
        or selected.get("fit_frozen_before_any_held_capability") is not True
        or selected.get("held_data_or_objectives_used") is not False
        or selected.get(
            "raw_gradients_tensors_h4_logits_or_targets_serialized"
        )
        is not False
    ):
        raise ValueError(f"V20d {law} fit evidence semantics differ")

    initial_objectives = _mapping(
        gradient.get("initial_objectives_by_family"),
        label=f"{law} initial objectives",
    )
    initial_h4 = _mapping(
        gradient.get("post_cast_h4_sha256s"), label=f"{law} initial H4 hashes"
    )
    initial_logits = _mapping(
        gradient.get("supervised_full_vocab_logits_sha256s"),
        label=f"{law} initial logit hashes",
    )
    initial_executions = _mapping(
        gradient.get("execution_receipt_sha256s"),
        label=f"{law} initial executions",
    )
    initial_gradient_hashes = _mapping(
        gradient.get("gradient_sha256s"), label=f"{law} initial gradient hashes"
    )
    if (
        set(initial_objectives) != set(fit_ids_by_family)
        or any(
            set(_mapping(initial_objectives[family], label="initial objectives"))
            != set(fit_ids_by_family[family])
            for family in fit_ids_by_family
        )
        or set(initial_h4) != set(exact_example_ids)
        or set(initial_logits) != set(exact_example_ids)
        or set(initial_executions) != set(exact_example_ids)
        or set(initial_gradient_hashes) != set(exact_example_ids)
    ):
        raise ValueError(f"V20d {law} initial execution geometry differs")
    for item in (
        *initial_h4.values(),
        *initial_logits.values(),
        *initial_executions.values(),
        *initial_gradient_hashes.values(),
    ):
        _sha(item, label=f"{law} initial evidence hash")
    for alpha in _ALPHAS:
        evidence = by_alpha[alpha]
        core_candidate = core_by_alpha[alpha]
        weight = tuple(float(item) for item in evidence["response_weight"])
        tensor = _weight_tensor(weight)
        corners = tuple(
            float(item)
            for item in fisher_continuous_bilinear_corner_values(tensor)
        )
        trace = _validate_response_trace(
            _mapping(evidence.get("response_trace"), label=f"{law} fit trace"),
            arm=f"fit_{law}_{alpha.hex()}",
        )
        objectives_by_family = _mapping(
            evidence.get("objectives_by_family"),
            label=f"{law} candidate objectives",
        )
        if set(objectives_by_family) != set(fit_ids_by_family):
            raise ValueError(f"V20d {law} candidate family geometry differs")
        flat_objectives: dict[str, float] = {}
        for family, ids in fit_ids_by_family.items():
            family_objectives = _mapping(
                objectives_by_family[family],
                label=f"{law} {family} candidate objectives",
            )
            if set(family_objectives) != set(ids):
                raise ValueError(f"V20d {law} candidate example geometry differs")
            flat_objectives.update(
                {example: float(family_objectives[example]) for example in ids}
            )
        h4 = _mapping(
            evidence.get("post_cast_h4_sha256s"),
            label=f"{law} candidate H4 hashes",
        )
        logits = _mapping(
            evidence.get("supervised_full_vocab_logits_sha256s"),
            label=f"{law} candidate logit hashes",
        )
        executions = _mapping(
            evidence.get("execution_receipt_sha256s"),
            label=f"{law} candidate executions",
        )
        trace_gains = _mapping(
            trace.get("response_gain_sha256s"),
            label=f"{law} candidate response gains",
        )
        if any(
            set(mapping) != set(exact_example_ids)
            for mapping in (h4, logits, executions, trace_gains)
        ):
            raise ValueError(f"V20d {law} candidate hash geometry differs")
        for item in (*h4.values(), *logits.values(), *executions.values(), *trace_gains.values()):
            _sha(item, label=f"{law} candidate evidence hash")
        phase = (
            "initial_vjp_alpha_zero"
            if alpha == 0.0
            else f"finite_alpha_{alpha.hex()}"
        )
        expected_executions = {
            example: _fit_execution_sha256(
                law=law,
                phase=phase,
                provider_artifact_sha256=str(
                    evidence["provider_artifact_sha256"]
                ),
                example_id=example,
                family_id=exact_example_family_ids[example],
                objective=flat_objectives[example],
                h4_sha256=str(h4[example]),
                logits_sha256=str(logits[example]),
            )
            for example in exact_example_ids
        }
        expected_transfer_evidence = (
            expected_initial_transfer_evidence
            if alpha == 0.0
            else _v14._sha256(
                {
                    "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                    "source_artifact_sha256": direction[
                        "v20c_source_sha256s"
                    ]["v20c_source_artifact_sha256"],
                    "direction_artifact_sha256": direction["artifact_sha256"],
                    "response_law": law,
                    "alpha": alpha,
                    "response_weights": weight,
                    "held_rows_used": False,
                },
                domain=_CANDIDATE_EVIDENCE_DOMAIN,
            )
        )
        nested_executions = _nested_execution_sha256s(evidence)
        if (
            evidence.get("law") != law
            or float(evidence.get("alpha", -1.0)) != alpha
            or weight != tuple(float(item) for item in core_candidate["weights"])
            or evidence.get("response_weight_sha256")
            != _provider_tensor_sha256(tensor)
            or tuple(evidence.get("bilinear_corner_values", ())) != corners
            or evidence.get("bilinear_box_max_abs")
            != max(abs(item) for item in corners)
            or evidence.get("global_bilinear_box_feasible") is not True
            or evidence.get("provider_artifact_sha256")
            != core_candidate.get("provider_artifact_sha256")
            or evidence.get("provider_transfer_evidence_sha256")
            != expected_transfer_evidence
            or evidence.get("objectives_by_family")
            != core_candidate.get("exact_fit_objectives_by_family")
            or nested_executions
            != core_candidate.get("fit_execution_receipt_sha256s_by_family")
            or evidence.get("family_equal_objective")
            != core_candidate.get("family_equal_objective")
            or evidence.get("family_objectives")
            != core_candidate.get("family_objectives")
            or evidence.get("response_gain_range_on_fit_support")
            != float(evidence["response_gain_max_on_fit_support"])
            - float(evidence["response_gain_min_on_fit_support"])
            or evidence.get("response_gain_nonconstant_on_fit_support")
            != (float(evidence["response_gain_range_on_fit_support"]) > 0.0)
            or evidence.get("exact_finite_full_model_forward") is not True
            or evidence.get("alpha_zero_reused_from_exact_initial_vjp")
            is not (alpha == 0.0)
            or evidence.get("held_data_or_objectives_used") is not False
            or evidence.get("raw_tensors_h4_logits_or_gradients_serialized")
            is not False
            or trace.get("provider_artifact_sha256")
            != evidence.get("provider_artifact_sha256")
            or tuple(trace.get("scored_family_ids", ()))
            != tuple(direction.get("fit_family_ids", ()))
            or dict(executions) != expected_executions
        ):
            raise ValueError(f"V20d {law} alpha candidate evidence differs")
        if alpha == 0.0 and (
            evidence.get("objectives_by_family") != initial_objectives
            or evidence.get("post_cast_h4_sha256s") != initial_h4
            or evidence.get("supervised_full_vocab_logits_sha256s")
            != initial_logits
            or evidence.get("execution_receipt_sha256s") != initial_executions
        ):
            raise ValueError(f"V20d {law} alpha-zero replay differs")
    selected_provider = selected.get("selected_provider_artifact_sha256")
    expected_selected = fit_receipt.get("selected_provider_artifact_sha256")
    selected_candidate = (
        next(
            (
                item
                for item in candidate_evidence
                if item.get("provider_artifact_sha256") == selected_provider
            ),
            None,
        )
        if selected_provider is not None
        else None
    )
    expected_nonconstant = bool(
        selected_candidate is not None
        and selected_candidate.get("response_gain_nonconstant_on_fit_support")
        is True
        and float(selected_candidate.get("response_gain_range_on_fit_support", 0.0))
        > 0.0
    )
    if (
        selected_provider != expected_selected
        or selected.get("selected_response_gain_nonconstant_on_fit_support")
        is not expected_nonconstant
    ):
        raise ValueError(f"V20d {law} selected fit evidence differs")
    return selected


def _core_work(*, held_scoring_executed: bool) -> dict[str, object]:
    core = _core.natural_response_work_accounting(
        collection_forward_count=32,
        collection_backward_count=16,
        endpoint_forward_count=12,
        endpoint_backward_count=12,
        endpoint_local_contraction_count=12,
        law_count=2,
        fit_prompt_count=12,
        alpha_count=6,
        alpha_zero_vjp_reused=True,
        held_role_count=2,
        held_arm_count=7,
        held_prompts_per_role=2,
        held_scoring_executed=held_scoring_executed,
    )
    return {
        **core,
        "full_model_forward_count": core["total_model_forward_count"],
        "full_suffix_backward_traversal_count": core["total_model_backward_count"],
        "local_head_autograd_contraction_count": core[
            "total_local_contraction_count"
        ],
        "total_autograd_grad_call_count": core[
            "total_backward_or_local_contraction_count"
        ],
        "teacher_capability_access_count": core[
            "teacher_h4_logit_access_count"
        ],
        "post_cast_h4_hash_check_count": core[
            "teacher_h4_logit_access_count"
        ],
        "supervised_full_vocab_logits_hash_check_count": core[
            "teacher_h4_logit_access_count"
        ],
        "fit_candidate_count": len(_FIT_LAWS) * len(_ALPHAS),
        "held_exact_arm_score_count": (
            2 * len(_HELD_ARMS) if held_scoring_executed else 0
        ),
        "held_capability_count": 2 if held_scoring_executed else 0,
        "provider_only_runtime_trace_count": (
            len(_FIT_LAWS) * len(_ALPHAS)
            + (2 * len(_HELD_ARMS) if held_scoring_executed else 0)
        ),
        "provider_only_runtime_traces_excluded_from_full_model_forward_count": True,
    }


def _work_accounting(*, held_scoring_executed: bool) -> dict[str, object]:
    observed = _core_work(held_scoring_executed=held_scoring_executed)
    planned = _core_work(held_scoring_executed=True)
    if (
        planned["full_model_forward_count"] != 216
        or planned["full_suffix_backward_traversal_count"] != 52
        or planned["local_head_autograd_contraction_count"] != 36
        or planned["total_autograd_grad_call_count"] != 88
        or planned["teacher_capability_access_count"] != 184
        or observed["full_model_forward_count"]
        != (216 if held_scoring_executed else 188)
        or observed["teacher_capability_access_count"]
        != (184 if held_scoring_executed else 156)
    ):
        raise RuntimeError("V20d exact work accounting differs")
    return {
        "planned_full_budget": planned,
        "observed_stage_counters": observed,
        "observed_matches_planned_full_budget": held_scoring_executed,
    }


def _fit_stage_authorized(
    fit_bundle: Mapping[str, object], fits: Mapping[str, Mapping[str, object]]
) -> bool:
    return bool(
        fit_bundle.get("both_fits_authorized") is True
        and fit_bundle.get("held_score_authorized") is True
        and set(fits) == set(_FIT_LAWS)
        and all(
            _mapping(fits[law], label=f"{law} fit evidence").get(
                "selected_response_gain_nonconstant_on_fit_support"
            )
            is True
            for law in _FIT_LAWS
        )
    )


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    v20c_report: Mapping[str, object],
    workspace: object,
    coordinate_trace: Mapping[str, object],
    fits: Mapping[str, _FitLive],
    fit_bundle: Mapping[str, object],
    provider_receipts: Mapping[str, Mapping[str, object]] | None,
    provider_bundle: Mapping[str, object] | None,
    roles: Sequence[Mapping[str, object]],
    role_evidence: Sequence[Mapping[str, object]],
    qualification: Mapping[str, object] | None,
) -> dict[str, object]:
    fit_evidence = {law: fits[law].fit_evidence for law in _FIT_LAWS}
    held_executed = provider_bundle is not None
    fit_authorized = _fit_stage_authorized(fit_bundle, fit_evidence)
    if held_executed != fit_authorized:
        raise RuntimeError("V20d held execution did not follow the fit-stage gate")
    if held_executed:
        if (
            provider_receipts is None
            or len(roles) != 2
            or len(role_evidence) != 2
            or qualification is None
        ):
            raise RuntimeError("V20d held report evidence is incomplete")
        passed = qualification.get("passed") is True
        classification = (
            "natural_response_pair_smoke_passed"
            if passed
            else "natural_response_pair_smoke_failed"
        )
    else:
        if (
            provider_receipts not in (None, {})
            or roles
            or role_evidence
            or qualification is not None
        ):
            raise RuntimeError("V20d fit terminal contains held evidence")
        passed = False
        classification = "fit_only_natural_response_failed"
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "artifact": output.as_posix(),
        "experiment_stage": "A16_development_only_reused_pair_natural_response",
        "scientific_status": "development_only_post_hoc_reused_a16_pair",
        "fixed_protocol": dict(_FIXED_PROTOCOL),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.NATURAL_RESPONSE_PROTOCOL_SHA256,
        "source": dict(source),
        "source_pair_diagnostic": dict(v20c_report["source_pair_diagnostic"]),
        "panel_receipt": dict(v20c_report["panel_receipt"]),
        "shared_fit_receipt": dict(workspace.fit_receipt),
        "fit_training_evidence": dict(workspace.fit_training_evidence),
        "coordinate_trace_receipt": dict(coordinate_trace),
        "natural_response_fit_receipts_by_law": {
            law: fits[law].fit_receipt for law in _FIT_LAWS
        },
        "natural_response_fit_execution_evidence_by_law": fit_evidence,
        "two_fit_bundle_receipt": dict(fit_bundle),
        "fit_stage_authorized_for_held_scoring": fit_authorized,
        "provider_receipts": (
            {arm: dict(provider_receipts[arm]) for arm in _HELD_ARMS}
            if provider_receipts is not None
            else {}
        ),
        "provider_bundle_receipt": (
            dict(provider_bundle) if provider_bundle is not None else None
        ),
        "roles": tuple(dict(role) for role in roles),
        "role_execution_evidence": tuple(dict(value) for value in role_evidence),
        "pair_qualification": dict(qualification) if qualification is not None else None,
        "held_scoring_executed": held_executed,
        "classification": classification,
        "passed": passed,
        "next_full_reused_panel_screen_authorized": passed,
        "empirical_fisher_response_weight_fit_implemented": True,
        "independent_signed_log_and_linear_fits_implemented": True,
        "fresh_family_disjoint_claim_authorized": False,
        "held_fidelity_claim": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "end_to_end_parameter_or_flop_claim": False,
        "candidate": None,
        "provider_sidecar": None,
        "raw_tensors_logits_gradients_targets_or_coordinates_serialized": False,
        "work_accounting": _work_accounting(
            held_scoring_executed=held_executed
        ),
    }
    _v14._scalar_report(report)
    return report


def _validate_report(
    value: Mapping[str, object],
    *,
    output: Path,
    authenticated_source: Mapping[str, object],
    authenticated_v20c: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(value)
    source = _mapping(selected.get("source"), label="V20d source")
    source_payload = {
        key: item for key, item in source.items() if key != "artifact_sha256"
    }
    panel = _v20b._core.validate_nested_microstep_panel_receipt(
        _mapping(selected.get("panel_receipt"), label="V20d panel")
    )
    endpoint_fit = _v20b._core.validate_nested_microstep_fit_receipt(
        _mapping(selected.get("shared_fit_receipt"), label="V20d endpoint fit"),
        panel_receipt=panel,
    )
    _v20b._validate_fit_training_evidence(
        selected.get("fit_training_evidence"), fit_receipt=endpoint_fit
    )
    source_hashes = _source_sha256s(source)
    fit_bundle = _core.validate_natural_response_two_fit_bundle_receipt(
        _mapping(
            selected.get("two_fit_bundle_receipt"), label="V20d two-fit bundle"
        ),
        expected_v20c_source_sha256s=source_hashes,
        expected_base_provider_artifact_sha256=str(
            endpoint_fit["base_provider_artifact_sha256"]
        ),
        expected_proposal_provider_artifact_sha256=str(
            endpoint_fit["proposal_provider_artifact_sha256"]
        ),
    )
    raw_fits = _mapping(
        selected.get("natural_response_fit_receipts_by_law"),
        label="V20d fit receipts",
    )
    raw_evidence = _mapping(
        selected.get("natural_response_fit_execution_evidence_by_law"),
        label="V20d fit execution evidence",
    )
    if set(raw_fits) != set(_FIT_LAWS) or set(raw_evidence) != set(_FIT_LAWS):
        raise ValueError("V20d two-law report geometry differs")
    fits: dict[str, dict[str, object]] = {}
    fit_evidence: dict[str, dict[str, object]] = {}
    for law in _FIT_LAWS:
        fits[law] = _core.validate_natural_response_fit_receipt(
            _mapping(raw_fits[law], label=f"{law} fit receipt"),
            expected_v20c_source_sha256s=source_hashes,
            expected_base_provider_artifact_sha256=str(
                endpoint_fit["base_provider_artifact_sha256"]
            ),
            expected_proposal_provider_artifact_sha256=str(
                endpoint_fit["proposal_provider_artifact_sha256"]
            ),
        )
        fit_evidence[law] = _validate_fit_evidence(
            _mapping(raw_evidence[law], label=f"{law} fit evidence"),
            fit_receipt=fits[law],
            law=law,
        )
        if fit_evidence[law]["gradient_evidence"].get(
            "fit_receipt_sha256"
        ) != endpoint_fit.get("artifact_sha256"):
            raise ValueError("V20d gradient evidence endpoint binding differs")
        if (
            _v14._canonical_json_bytes(fits[law])
            != _v14._canonical_json_bytes(
                fit_bundle["fit_receipts_by_law"][law]
            )
        ):
            raise ValueError("V20d fit/bundle receipt binding differs")
    gradient_rows = {
        law: _mapping(
            fit_evidence[law]["gradient_evidence"],
            label=f"{law} gradient evidence",
        )
        for law in _FIT_LAWS
    }
    if (
        len(
            {
                row["artifact_sha256"]
                for row in gradient_rows.values()
            }
        )
        != 2
        or len(
            {
                row["provider_artifact_sha256"]
                for row in gradient_rows.values()
            }
        )
        != 2
        or len(
            {
                row["provider_transfer_evidence_sha256"]
                for row in gradient_rows.values()
            }
        )
        != 2
    ):
        raise ValueError("V20d per-law initial VJP evidence is not distinct")
    fit_authorized = _fit_stage_authorized(fit_bundle, fit_evidence)
    held_executed = selected.get("held_scoring_executed") is True
    roles: tuple[dict[str, object], ...] = ()
    if held_executed:
        if not fit_authorized:
            raise ValueError("V20d held scoring bypassed the fit-only gate")
        provider_receipts = {
            str(arm): _mapping(row, label=f"{arm} provider receipt")
            for arm, row in _mapping(
                selected.get("provider_receipts"), label="V20d providers"
            ).items()
        }
        provider_bundle_raw = _mapping(
            selected.get("provider_bundle_receipt"), label="V20d provider bundle"
        )
        normalized_receipts, normalized_provider_bundle = _validate_provider_bundle(
            provider_receipts,
            provider_bundle=provider_bundle_raw,
            fit_bundle=fit_bundle,
            fit_evidence_by_law=fit_evidence,
        )
        roles = tuple(
            _core.validate_natural_response_held_role_receipt(
                _mapping(item, label="V20d held role"),
                fit_bundle_receipt=fit_bundle,
            )
            for item in _sequence(selected.get("roles"), label="V20d held roles")
        )
        evidence = tuple(
            _mapping(item, label="V20d held role evidence")
            for item in _sequence(
                selected.get("role_execution_evidence"),
                label="V20d role execution evidence",
            )
        )
        expected_qualification = _pair_qualification(
            fit_bundle=fit_bundle,
            provider_bundle=normalized_provider_bundle,
            provider_receipts=normalized_receipts,
            fits=fit_evidence,
            roles=roles,
            role_evidence=evidence,
        )
        if (
            _v14._canonical_json_bytes(selected.get("pair_qualification"))
            != _v14._canonical_json_bytes(expected_qualification)
        ):
            raise ValueError("V20d pair qualification differs")
        passed = expected_qualification["passed"] is True
        classification = (
            "natural_response_pair_smoke_passed"
            if passed
            else "natural_response_pair_smoke_failed"
        )
    else:
        if (
            fit_authorized
            or selected.get("provider_receipts") != {}
            or selected.get("provider_bundle_receipt") is not None
            or tuple(selected.get("roles", ()))
            or tuple(selected.get("role_execution_evidence", ()))
            or selected.get("pair_qualification") is not None
        ):
            raise ValueError("V20d fit-only terminal contains held authority")
        passed = False
        classification = "fit_only_natural_response_failed"
    expected_v20c_fields = (
        "source_pair_diagnostic",
        "panel_receipt",
        "shared_fit_receipt",
        "fit_training_evidence",
        "coordinate_trace_receipt",
    )
    if any(
        _v14._canonical_json_bytes(selected.get(field))
        != _v14._canonical_json_bytes(authenticated_v20c.get(field))
        for field in expected_v20c_fields
    ):
        raise ValueError("V20d report drifted from immutable V20c endpoint evidence")
    if (
        selected.get("schema") != _SCHEMA
        or selected.get("format_version") != _FORMAT_VERSION
        or selected.get("artifact") != output.as_posix()
        or selected.get("experiment_stage")
        != "A16_development_only_reused_pair_natural_response"
        or selected.get("scientific_status")
        != "development_only_post_hoc_reused_a16_pair"
        or _v14._canonical_json_bytes(selected.get("fixed_protocol"))
        != _v14._canonical_json_bytes(_FIXED_PROTOCOL)
        or selected.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or selected.get("core_protocol_sha256")
        != _core.NATURAL_RESPONSE_PROTOCOL_SHA256
        or source.get("artifact_sha256")
        != _v14._sha256(source_payload, domain=_SOURCE_DOMAIN)
        or _v14._canonical_json_bytes(source)
        != _v14._canonical_json_bytes(authenticated_source)
        or selected.get("fit_stage_authorized_for_held_scoring")
        is not fit_authorized
        or selected.get("classification") != classification
        or selected.get("passed") is not passed
        or selected.get("next_full_reused_panel_screen_authorized") is not passed
        or selected.get("empirical_fisher_response_weight_fit_implemented")
        is not True
        or selected.get("independent_signed_log_and_linear_fits_implemented")
        is not True
        or selected.get("fresh_family_disjoint_claim_authorized") is not False
        or selected.get("held_fidelity_claim") is not False
        or selected.get("serving_authorized") is not False
        or selected.get("compression_claim") is not False
        or selected.get("speed_or_latency_claim") is not False
        or selected.get("end_to_end_parameter_or_flop_claim") is not False
        or selected.get("candidate") is not None
        or selected.get("provider_sidecar") is not None
        or selected.get(
            "raw_tensors_logits_gradients_targets_or_coordinates_serialized"
        )
        is not False
        or selected.get("work_accounting")
        != _work_accounting(held_scoring_executed=held_executed)
    ):
        raise ValueError("V20d report authority or claim boundary differs")
    _v14._scalar_report(selected)
    return selected


def _load_existing_report(output: Path) -> dict[str, object]:
    selected = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20d report",
    )
    source, v20c_report, _fragment = _load_authenticated_v20c_source()
    return _validate_report(
        selected,
        output=output,
        authenticated_source=source,
        authenticated_v20c=v20c_report,
    )


def run_gemma3_l3_l4_complete_h4_natural_response_smoke(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or authenticate the V20d independent natural-response fit."""

    destination = _validate_output(output)
    if destination.exists():
        return _load_existing_report(destination)

    # Both immutable prerequisites authenticate before any model construction.
    source, v20c_report, fragment = _load_authenticated_v20c_source()
    prerequisite, _v20a_payload, _v20a_folds = (
        _v20b._load_authenticated_v20a_artifact()
    )
    panel_receipt = dict(
        _mapping(v20c_report.get("panel_receipt"), label="V20c panel receipt")
    )
    family_ids = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20c panel family prompts",
            )
        )
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != family_ids:
            raise RuntimeError("V20d live family order differs from V20c")
        workspace = _v20b._reconstruct_pair_workspace(
            context,
            records,
            teacher_vault,
            fragment=fragment,
            panel_receipt=panel_receipt,
        )
        _coordinates, coordinate_trace = _v20c._fit_coordinates_by_family(
            workspace.training_records,
            base_provider=workspace.base_provider,
            family_ids=family_ids,
            excluded_family_ids=_v20c._FROZEN_EXCLUDED,
        )
        if (
            _v14._canonical_json_bytes(coordinate_trace)
            != _v14._canonical_json_bytes(
                v20c_report.get("coordinate_trace_receipt")
            )
        ):
            raise RuntimeError("V20d fit coordinates drifted from immutable V20c")

        fits = {
            law: _fit_response_law(
                context,
                workspace,
                teacher_vault,
                law=law,
                source=source,
                family_ids=family_ids,
            )
            for law in _FIT_LAWS
        }
        fit_bundle = _core.build_natural_response_two_fit_bundle_receipt(
            signed_log_fit_receipt=fits["signed_log"].fit_receipt,
            linear_fit_receipt=fits["linear"].fit_receipt,
        )
        fit_evidence = {law: fits[law].fit_evidence for law in _FIT_LAWS}
        if _fit_stage_authorized(fit_bundle, fit_evidence):
            providers, provider_receipts, provider_bundle = (
                _build_frozen_held_providers(
                    workspace,
                    fits=fits,
                    fit_bundle=fit_bundle,
                )
            )
            _validate_provider_bundle(
                provider_receipts,
                provider_bundle=provider_bundle,
                fit_bundle=fit_bundle,
                fit_evidence_by_law=fit_evidence,
            )
            roles: list[dict[str, object]] = []
            role_evidence: list[dict[str, object]] = []
            for outer, held in (
                (_v20c._REED, _v20c._SUNDIAL),
                (_v20c._SUNDIAL, _v20c._REED),
            ):
                role, evidence = _score_reciprocal_role(
                    context,
                    records,
                    teacher_vault,
                    outer_family_id=outer,
                    scored_family_id=held,
                    providers=providers,
                    provider_receipts=provider_receipts,
                    provider_bundle=provider_bundle,
                    fit_bundle=fit_bundle,
                )
                roles.append(role)
                role_evidence.append(evidence)
            qualification = _pair_qualification(
                fit_bundle=fit_bundle,
                provider_bundle=provider_bundle,
                provider_receipts=provider_receipts,
                fits=fit_evidence,
                roles=roles,
                role_evidence=role_evidence,
            )
        else:
            provider_receipts = None
            provider_bundle = None
            roles = []
            role_evidence = []
            qualification = None
        report = _build_report(
            output=destination,
            source=source,
            v20c_report=v20c_report,
            workspace=workspace,
            coordinate_trace=coordinate_trace,
            fits=fits,
            fit_bundle=fit_bundle,
            provider_receipts=provider_receipts,
            provider_bundle=provider_bundle,
            roles=roles,
            role_evidence=role_evidence,
            qualification=qualification,
        )
    finally:
        context.validate_immutable_inputs()
        del context

    try:
        _v20b._publish_scalar_fragment(
            report,
            path=destination,
            domain=_REPORT_DOMAIN,
            hash_key="report_sha256",
            label="V20d report",
        )
    except FileExistsError:
        return _load_existing_report(destination)
    return _load_existing_report(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only V20d fit-only natural-response pair smoke"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_natural_response_smoke(
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
